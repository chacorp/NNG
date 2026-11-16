// Shape inference over a graph. Shapes exclude the batch dimension.

/**
 * Shape as text with brackets.
 * @param {number[]|undefined} shape
 * @returns {string} e.g. "[3, 32, 32]" or "?"
 */
function format_shape(shape){
  return shape ? `[${shape.join(', ')}]` : '?';
}

/**
 * Shape as compact text.
 * @param {number[]|undefined} shape
 * @returns {string} e.g. "3×32×32" or "?"
 */
function dims_text(shape){
  return shape ? shape.join('×') : '?';
}

/**
 * @param {number[]} first_shape
 * @param {number[]} second_shape
 * @returns {boolean} true when both shapes are identical
 */
function shapes_equal(first_shape, second_shape){
  return first_shape.length === second_shape.length && first_shape.every((size, dim_index) => size === second_shape[dim_index]);
}

/**
 * Parse an Input shape text.
 * @param {string} shape_text  e.g. "3,32,32"
 * @returns {number[]} e.g. [3, 32, 32]
 */
function parse_shape_text(shape_text){
  const parts = String(shape_text).replace(/[\[\]()\s]/g, '').split(/[,×x*]/).filter(Boolean);
  if(!parts.length) throw 'shape를 입력하세요';
  const shape = parts.map(Number);
  if(shape.some(size => !Number.isInteger(size) || size < 1)) throw `잘못된 shape: ${shape_text}`;
  return shape;
}

/**
 * Size of the input dim a layer constructor needs.
 * @param {number[]} input_shape  shape without batch
 * @param {string} consumed_dim   'last' | 'channel'
 * @returns {number} e.g. last dim (Linear in_features) or first dim (Conv2d in_channels)
 */
function consumed_dim_size(input_shape, consumed_dim){
  return consumed_dim === 'last' ? input_shape[input_shape.length - 1] : input_shape[0];
}

/**
 * Node ids in topological order (nodes on a cycle are left out).
 * @param {object} graph
 * @returns {number[]} node ids
 */
function topological_order(graph){
  const in_degree = new Map(graph.nodes.map(node => [node.id, 0]));
  graph.edges.forEach(edge => in_degree.set(edge.to, in_degree.get(edge.to) + 1));
  const ready_ids = graph.nodes.filter(node => in_degree.get(node.id) === 0).map(node => node.id), ordered_ids = [];
  while(ready_ids.length){
    const node_id = ready_ids.shift();
    ordered_ids.push(node_id);
    graph.edges.filter(edge => edge.from === node_id).forEach(edge => {
      in_degree.set(edge.to, in_degree.get(edge.to) - 1);
      if(in_degree.get(edge.to) === 0) ready_ids.push(edge.to);
    });
  }
  return ordered_ids;
}

let block_call_input_shapes = {};

/**
 * Output shape of a custom block called with the given input shapes.
 * @param {string} block_name
 * @param {number[][]} input_shapes  one shape per block input
 * @param {string[]} call_stack      enclosing block names (recursion check)
 * @returns {number[]} output shape
 */
function infer_block_call(block_name, input_shapes, call_stack){
  const block_graph = state.blocks[block_name];
  if(!block_graph) throw `블록 '${block_name}' 정의 없음`;
  if(call_stack.includes(block_name)) throw `블록 재귀 참조 (${[...call_stack, block_name].join(' → ')})`;
  const {input_nodes, output_nodes} = block_ports(block_graph);
  if(output_nodes.length !== 1) throw `블록 '${block_name}'에는 Output이 정확히 1개 필요`;
  if(!input_nodes.length) throw `블록 '${block_name}'에 Input이 없음`;
  if(!block_call_input_shapes[block_name]) block_call_input_shapes[block_name] = input_shapes;
  const input_shape_overrides = {};
  input_nodes.forEach((input_node, input_index) => input_shape_overrides[input_node.id] = input_shapes[input_index]);
  const block_inference = infer_shapes(block_graph, input_shape_overrides, [...call_stack, block_name]);
  if(!block_inference.shapes[output_nodes[0].id]){
    const first_error = Object.entries(block_inference.errors).find(([, message]) => message !== '상위 노드 오류');
    const error_node = first_error && find_node(block_graph, +first_error[0]);
    const error_label = error_node ? `${error_node.type in OPERATOR_SPECS ? OPERATOR_SPECS[error_node.type].label : node_title(error_node)}: ${first_error[1]}` : '오류';
    throw `블록 '${block_name}' 내부 ${error_label}`;
  }
  return block_inference.shapes[output_nodes[0].id];
}

/**
 * Output shape of a Reshape node.
 * @param {number[]} input_shape
 * @param {string} target_text  e.g. "16,-1"
 * @returns {number[]}
 */
function reshape_output_shape(input_shape, target_text){
  const target_sizes = String(target_text).split(',').map(part => part.trim()).filter(Boolean).map(Number);
  if(!target_sizes.length || target_sizes.some(size => !Number.isInteger(size) || (size < 1 && size !== -1)) || target_sizes.filter(size => size === -1).length > 1)
    throw `잘못된 reshape 크기: ${target_text}`;
  const total_size = input_shape.reduce((product, size) => product * size, 1);
  const known_size = target_sizes.filter(size => size !== -1).reduce((product, size) => product * size, 1);
  if(target_sizes.includes(-1)){
    if(total_size % known_size) throw `${format_shape(input_shape)}을 (${target_text})로 reshape 불가`;
    return target_sizes.map(size => size === -1 ? total_size / known_size : size);
  }
  if(known_size !== total_size) throw `${format_shape(input_shape)}을 (${target_text})로 reshape 불가`;
  return target_sizes;
}

/**
 * Output shape of an operator node.
 * @param {object} operator_node
 * @param {number[][]} input_shapes  operand shapes
 * @param {boolean} divides_by_constant  Div with a constant divisor
 * @returns {number[]}
 */
function operator_output_shape(operator_node, input_shapes, divides_by_constant){
  const params = operator_node.params, first_shape = input_shapes[0];
  switch(operator_node.type){
    case 'CustomFn': return first_shape;
    case 'Div': if(divides_by_constant) return first_shape;
    // falls through: element-wise with a second tensor
    case 'Add': case 'Sub': case 'Hadamard':
      input_shapes.forEach(shape => { if(!shapes_equal(shape, first_shape)) throw `shape 불일치 ${format_shape(first_shape)} vs ${format_shape(shape)}`; });
      return first_shape;
    case 'MatMul': {
      const left_shape = first_shape, right_input = input_shapes[1];
      const right_shape = params.transpose_b && right_input && right_input.length >= 2 ? [...right_input.slice(0, -2), right_input.at(-1), right_input.at(-2)] : right_input;
      if(left_shape.length < 2 || right_shape.length < 2) throw `행렬곱은 [.., n, k] × [.., k, m] 필요 (${format_shape(left_shape)} × ${format_shape(right_shape)})`;
      if(left_shape.length !== right_shape.length || !shapes_equal(left_shape.slice(0, -2), right_shape.slice(0, -2))) throw `앞쪽 차원 불일치 ${format_shape(left_shape)} × ${format_shape(right_shape)}`;
      if(left_shape[left_shape.length - 1] !== right_shape[right_shape.length - 2]) throw `내적 차원 불일치 ${format_shape(left_shape)} × ${format_shape(right_shape)}`;
      return [...left_shape.slice(0, -1), right_shape[right_shape.length - 1]];
    }
    case 'CausalMask':
      if(first_shape.length < 2) throw '마스크는 [.., T, S] 형태 필요';
      return first_shape;
    case 'Reshape': return reshape_output_shape(first_shape, params.to);
    case 'Permute': {
      const rank = first_shape.length, dim_order = String(params.order).split(',').map(part => Number(part.trim()));
      if(dim_order.length !== rank + 1 || dim_order[0] !== 0 || [...dim_order].sort((a, b) => a - b).some((dim, position) => dim !== position))
        throw `permute 순서는 0으로 시작하는 0..${rank} 순열이어야 함 (현재 ${params.order})`;
      return dim_order.slice(1).map(dim => first_shape[dim - 1]);
    }
    case 'Transpose': {
      const rank = first_shape.length, to_positive = dim => dim < 0 ? dim + rank + 1 : dim;
      const first_dim = to_positive(params.dim0), second_dim = to_positive(params.dim1);
      if(first_dim < 1 || second_dim < 1 || first_dim > rank || second_dim > rank) throw 'transpose dim 범위 밖 (배치 dim 0은 불가)';
      const output_shape = [...first_shape];
      [output_shape[first_dim - 1], output_shape[second_dim - 1]] = [output_shape[second_dim - 1], output_shape[first_dim - 1]];
      return output_shape;
    }
    case 'Concat': {
      const rank = first_shape.length, concat_dim = params.dim;
      if(concat_dim === 0) throw 'dim=0(배치)에는 concat 불가';
      const dim_index = concat_dim < 0 ? rank + concat_dim : concat_dim - 1;
      if(dim_index < 0 || dim_index >= rank) throw `dim ${concat_dim} 범위 밖`;
      input_shapes.forEach(shape => {
        if(shape.length !== rank || shape.some((size, index) => index !== dim_index && size !== first_shape[index])) throw `concat shape 불일치 ${format_shape(first_shape)} vs ${format_shape(shape)}`;
      });
      const output_shape = [...first_shape];
      output_shape[dim_index] = input_shapes.reduce((total, shape) => total + shape[dim_index], 0);
      return output_shape;
    }
  }
  throw `알 수 없는 연산자 ${operator_node.type}`;
}

/**
 * Infer every node's output shape.
 * @param {object} graph
 * @param {Object<number, number[]>} input_shape_overrides  Input node id → shape (a block evaluated with its caller's shapes)
 * @param {string[]} call_stack  enclosing block names
 * @returns {{shapes: Object<number, number[]>, errors: Object<number, string>, order: number[]}}
 */
function infer_shapes(graph, input_shape_overrides = {}, call_stack = []){
  const shapes = {}, errors = {}, ordered_ids = topological_order(graph);
  graph.nodes.forEach(node => { if(!ordered_ids.includes(node.id)) errors[node.id] = '순환 연결'; });
  for(const node_id of ordered_ids){
    const node = find_node(graph, node_id), params = node.params;
    try{
      const source_ids = input_sources(graph, node);
      if(is_variadic_operator(node) && source_ids.length < 2) throw '입력 2개 이상 연결 필요';
      const divides_by_constant = node.type === 'Div' && is_valid_divisor_text(params.by);
      if(divides_by_constant && source_ids.length !== 1) throw '상수로 나눌 때는 입력 1개만 연결';
      if(node.type in OPERATOR_SPECS && !is_variadic_operator(node) && !divides_by_constant && source_ids.length !== OPERATOR_SPECS[node.type].input_count)
        throw `입력 ${OPERATOR_SPECS[node.type].input_count}개 필요 (현재 ${source_ids.length}개)`;
      if(source_ids.some(source_id => source_id === null)) throw '입력 미연결';
      if(source_ids.some(source_id => !shapes[source_id])) throw '상위 노드 오류';
      const input_shapes = source_ids.map(source_id => shapes[source_id]);
      let output_shape;
      if(node.type === 'Input'){
        if(input_shape_overrides[node.id]) output_shape = input_shape_overrides[node.id];
        else {
          output_shape = parse_shape_text(params.shape);
          const dim_names = input_dim_names(node);
          if(dim_names.length !== output_shape.length) throw `차원 이름 ${dim_names.length}개(${layout_text(dim_names)})와 크기 ${output_shape.length}개가 맞지 않음`;
        }
      }
      else if(node.type === 'Output') output_shape = input_shapes[0];
      else if(node.type in OPERATOR_SPECS) output_shape = operator_output_shape(node, input_shapes, divides_by_constant);
      else if(node.type === 'Custom') output_shape = infer_block_call(params.block, input_shapes, call_stack);
      else {
        const layer_spec = LAYER_SPECS[node.type];
        output_shape = layer_spec.input_port_names ? layer_spec.output_shape(input_shapes, params) : layer_spec.output_shape(input_shapes[0], params);
      }
      shapes[node_id] = output_shape;
    } catch(error){ errors[node_id] = String(error); }
  }
  return {shapes, errors, order: ordered_ids};
}
