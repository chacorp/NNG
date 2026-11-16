// Export the graph as torch.nn code with a forward / gradient self-check.

const PYTHON_RESERVED_NAMES = new Set('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield input self torch nn model'.split(' '));
const MODULE_ATTRIBUTE_PREFIX = {Linear: 'fc', Conv2d: 'conv', MaxPool2d: 'pool', BatchNorm2d: 'bn', LayerNorm: 'ln', Dropout: 'drop', Flatten: 'flatten', Upsample: 'up', AdaptiveAvgPool2d: 'gap', Attention: 'attn'};
let current_model_code = null;

/**
 * Valid Python identifier from a name.
 * @param {string} name
 * @returns {string}
 */
function python_identifier(name){
  let identifier = String(name).trim().replace(/[^A-Za-z0-9_]/g, '_');
  if(!identifier || /^\d/.test(identifier)) identifier = '_' + identifier;
  if(PYTHON_RESERVED_NAMES.has(identifier)) identifier += '_';
  return identifier;
}

/**
 * CamelCase → snake_case.
 * @param {string} camel_case_name  e.g. "DecoderBlock"
 * @returns {string} e.g. "decoder_block"
 */
function to_snake_case(camel_case_name){
  return camel_case_name.replace(/([a-z0-9])([A-Z])/g, '$1_$2').replace(/([A-Z])([A-Z][a-z])/g, '$1_$2').toLowerCase();
}

/**
 * Python expression of an operator node.
 * @param {object} operator_node
 * @param {string[]} input_variable_names  operand variable names
 * @param {number[]} first_input_shape     shape of the first operand (no batch)
 * @returns {string}
 */
function operator_python_expression(operator_node, input_variable_names, first_input_shape){
  const params = operator_node.params, [first_name, second_name] = input_variable_names;
  const list_arguments = list_text => String(list_text).split(',').map(part => part.trim()).join(', ');
  switch(operator_node.type){
    case 'Add': return input_variable_names.join(' + ');
    case 'Sub': return `${first_name} - ${second_name}`;
    case 'Hadamard': return input_variable_names.join(' * ');
    case 'Div': return is_valid_divisor_text(params.by) ? `${first_name} / ${divisor_python(params.by)}` : `${first_name} / ${second_name}`;
    case 'MatMul': return `torch.matmul(${first_name}, ${second_name}${params.transpose_b ? '.transpose(-2, -1)' : ''})`;
    case 'Concat': return `torch.cat([${input_variable_names.join(', ')}], dim=${params.dim})`;
    case 'CustomFn': return `custom_function(${first_name})`;
    case 'Reshape': return `${first_name}.reshape(${first_name}.shape[0], ${list_arguments(params.to)})`;
    case 'Permute': return `${first_name}.permute(${list_arguments(params.order)})`;
    case 'Transpose': return `${first_name}.transpose(${params.dim0}, ${params.dim1})`;
    case 'CausalMask': return `${first_name}.masked_fill(torch.ones(${first_input_shape.at(-2)}, ${first_input_shape.at(-1)}, dtype=torch.bool, device=${first_name}.device).triu(1), float("-inf"))`;
  }
  throw `알 수 없는 연산자 ${operator_node.type}`;
}

/**
 * Python self-check appended to the exported file: random inputs (B=4), output shape asserts, dummy-loss backward,
 * and asserts that every parameter and input received a finite gradient.
 * @param {object} main_module  result of the Model generation ({input_nodes, output_nodes, shapes})
 * @returns {string}
 */
function self_check_python(main_module){
  const input_tensors = main_module.input_nodes.map(input_node => `torch.randn(B, ${main_module.shapes[input_node.id].join(', ')})`).join(', ');
  const expected_outputs = main_module.output_nodes.map(output_node => `("${String(output_node.params.name).replace(/["\\]/g, '')}", (B, ${main_module.shapes[output_node.id].join(', ')}))`).join(', ');
  return `

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    model = Model()
    print(model)

    # ---- 1) forward: 더미 입력으로 출력 shape 확인
    inputs = tuple(t.requires_grad_() for t in (${input_tensors},))
    outputs = model(*inputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    expected = [${expected_outputs}]
    print("\\n== Forward ==")
    for (name, shape), out in zip(expected, outputs):
        print(f"  {name}: {tuple(out.shape)} (expected {shape})")
        assert tuple(out.shape) == shape, f"shape mismatch for {name}"

    # ---- 2) backward: 더미 loss로 gradient 흐름 확인
    loss = sum(out.float().pow(2).mean() for out in outputs)
    loss.backward()
    print(f"\\n== Backward ==\\n  dummy loss = {loss.item():.6f}")
    params = list(model.named_parameters())
    for n, p in params:
        g = "None" if p.grad is None else f"{p.grad.norm().item():.3e}"
        print(f"  {n:40s} grad_norm={g}")
    missing = [n for n, p in params if p.grad is None]
    nonfinite = [n for n, p in params if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing, f"gradient did not reach: {missing}"
    assert not nonfinite, f"non-finite gradient: {nonfinite}"
    assert all(t.grad is not None for t in inputs), "gradient did not reach every input"
    print(f"\\nOK: output shapes match, gradients reached all {len(params)} parameter tensors and all inputs")
`;
}

/**
 * Generate model.py for the main graph and every block it uses.
 * Each (block, input shapes) pair becomes one nn.Module class with literal layer sizes.
 * @returns {{code: string}|{problems: string[]}}
 */
function generate_model_code(){
  block_call_input_shapes = {};
  const problems = [], block_classes = [], class_name_by_call = new Map(), used_class_names = new Set(['Model']);
  let uses_custom_function = false;

  const class_for_block_call = (block_name, input_shapes, call_stack) => {
    const call_key = block_name + JSON.stringify(input_shapes);
    if(class_name_by_call.has(call_key)) return class_name_by_call.get(call_key);
    let class_name = block_name, suffix = 2;
    while(used_class_names.has(class_name)) class_name = `${block_name}_${suffix++}`;
    used_class_names.add(class_name);
    class_name_by_call.set(call_key, class_name);
    const block_graph = state.blocks[block_name], input_shape_overrides = {};
    block_ports(block_graph).input_nodes.forEach((input_node, input_index) => input_shape_overrides[input_node.id] = input_shapes[input_index]);
    const block_module = generate_module(class_name, block_graph, input_shape_overrides, [...call_stack, block_name], block_name);
    if(block_module) block_classes.push(block_module);
    return class_name;
  };

  const generate_module = (class_name, graph, input_shape_overrides, call_stack, block_name) => {
    const {shapes, errors, order} = infer_shapes(graph, input_shape_overrides, call_stack);
    const {input_nodes, output_nodes} = block_ports(graph), location = block_name ? `[${block_name}] ` : '';
    if(!input_nodes.length) problems.push(location + 'Input 노드가 없습니다.');
    if(!output_nodes.length) problems.push(location + 'Output 노드가 없습니다.');
    if(block_name && output_nodes.length > 1) problems.push(location + '블록에는 Output이 1개만 있어야 합니다.');
    const needed_ids = new Set(), pending_ids = output_nodes.map(node => node.id);
    while(pending_ids.length){
      const node_id = pending_ids.pop();
      if(needed_ids.has(node_id)) continue;
      needed_ids.add(node_id);
      graph.edges.filter(edge => edge.to === node_id).forEach(edge => pending_ids.push(edge.from));
    }
    input_nodes.forEach(node => needed_ids.add(node.id));
    const problem_label = node => node.type === 'Input' || node.type === 'Output' ? `${node.type} '${node.params.name}'`
      : node.type in OPERATOR_SPECS ? OPERATOR_SPECS[node.type].label : node_title(node);
    for(const node_id of needed_ids) if(errors[node_id] && errors[node_id] !== '상위 노드 오류') problems.push(`${location}${problem_label(find_node(graph, node_id))}: ${errors[node_id]}`);
    const seen_names = new Set();
    for(const node of [...input_nodes, ...output_nodes]){
      const name_key = node.type + ':' + python_identifier(node.params.name);
      if(seen_names.has(name_key)) problems.push(`${location}${node.type} 이름 중복: ${node.params.name}`);
      seen_names.add(name_key);
    }
    if(problems.length) return null;

    const attribute_counts = {}, variable_by_node = {}, init_lines = [], forward_lines = [];
    let hidden_index = 0;
    input_nodes.forEach(node => variable_by_node[node.id] = python_identifier(node.params.name));
    for(const node_id of order){
      if(!needed_ids.has(node_id)) continue;
      const node = find_node(graph, node_id), params = node.params;
      if(node.type === 'Input' || node.type === 'Output') continue;
      const source_ids = input_sources(graph, node), input_names = source_ids.map(source_id => variable_by_node[source_id]), first_input_shape = shapes[source_ids[0]];
      const variable_name = 'h' + (++hidden_index);
      variable_by_node[node_id] = variable_name;
      let expression;
      if(node.type in OPERATOR_SPECS){
        if(node.type === 'CustomFn') uses_custom_function = true;
        expression = operator_python_expression(node, input_names, first_input_shape);
      } else {
        const attribute_base = node.type === 'Custom' ? to_snake_case(params.block) : node.type === 'Activation' ? params.fn.toLowerCase() : MODULE_ATTRIBUTE_PREFIX[node.type];
        attribute_counts[attribute_base] = (attribute_counts[attribute_base] || 0) + 1;
        const attribute_name = /\d$/.test(attribute_base) ? `${attribute_base}_${attribute_counts[attribute_base]}` : `${attribute_base}${attribute_counts[attribute_base]}`;
        let constructor_code;
        if(node.type === 'Custom') constructor_code = `${class_for_block_call(params.block, source_ids.map(source_id => shapes[source_id]), call_stack)}()`;
        else {
          const layer_spec = LAYER_SPECS[node.type];
          constructor_code = layer_spec.input_port_names ? layer_spec.torch_code(params, source_ids.map(source_id => shapes[source_id]))
            : layer_spec.torch_code(params, layer_spec.consumed_dim ? consumed_dim_size(first_input_shape, layer_spec.consumed_dim) : null);
        }
        init_lines.push(`        self.${attribute_name} = ${constructor_code}`);
        expression = LAYER_SPECS[node.type] && LAYER_SPECS[node.type].call_code
          ? LAYER_SPECS[node.type].call_code(attribute_name, input_names) : `self.${attribute_name}(${input_names.join(', ')})`;
      }
      forward_lines.push(`        ${variable_name} = ${expression}  # [B, ${shapes[node_id].join(', ')}]`);
    }
    const returned_names = output_nodes.map(output_node => variable_by_node[input_sources(graph, output_node)[0]]);
    return {class_name, shapes, input_nodes, output_nodes, code:
      `class ${class_name}(nn.Module):\n    def __init__(self):\n        super().__init__()\n${init_lines.length ? init_lines.join('\n') : '        pass'}\n\n`
      + `    def forward(self, ${input_nodes.map(node => variable_by_node[node.id]).join(', ')}):\n${forward_lines.join('\n')}${forward_lines.length ? '\n' : ''}        return ${returned_names.join(', ')}\n`};
  };

  const main_module = generate_module('Model', state.main, {}, [], null);
  if(problems.length || !main_module) return {problems: [...new Set(problems)]};
  const uses_math = all_graphs().some(graph => graph.nodes.some(node => node.type === 'Div' && is_valid_divisor_text(node.params.by) && /sqrt/.test(node.params.by)));
  let model_code = `${uses_math ? 'import math\n' : ''}import torch\nimport torch.nn as nn\n\n`;
  if(uses_custom_function) model_code += '\ndef custom_function(input):\n    # TODO: 사용자 정의 연산을 구현하세요\n    return input\n\n';
  block_classes.forEach(block_module => model_code += '\n' + block_module.code + '\n');
  model_code += '\n' + main_module.code;
  model_code += self_check_python(main_module);
  return {code: model_code};
}

/** Show the generated code (or the problems that block export) in the side panel. */
function render_code_panel(){
  const generation_result = generate_model_code(), problem_list = document.getElementById('problem_list');
  problem_list.innerHTML = '';
  current_model_code = generation_result.code || null;
  if(generation_result.problems){
    const problem_box = document.createElement('div');
    problem_box.className = 'errbox';
    const heading = document.createElement('b');
    heading.textContent = 'Export 불가';
    problem_box.appendChild(heading);
    generation_result.problems.forEach(problem_text => { const line = document.createElement('div'); line.textContent = '• ' + problem_text; problem_box.appendChild(line); });
    problem_list.appendChild(problem_box);
  }
  const code_view = document.getElementById('code_view');
  code_view.textContent = generation_result.code || '';
  code_view.style.display = generation_result.code ? '' : 'none';
  document.getElementById('copy_button').disabled = document.getElementById('save_python_button').disabled = !generation_result.code;
}
