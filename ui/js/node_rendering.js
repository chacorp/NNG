// Drawing nodes, ports, arrows and operator labels on the canvas.

const canvas_element = document.getElementById('canvas');
const world_element = document.getElementById('world');
const wire_group_element = document.getElementById('wire_group');
const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
let current_inference = {shapes: {}, errors: {}};
let current_input_overrides = {};
const side_label_boxes = new Map();

/** Apply pan and zoom to the canvas. */
function apply_view_transform(){
  world_element.style.transform = `translate(${view_transform.x}px,${view_transform.y}px) scale(${view_transform.scale})`;
  document.getElementById('zoom_info').textContent = Math.round(view_transform.scale * 100) + '%';
}

/**
 * Element of a node on the canvas.
 * @param {number} node_id
 * @returns {HTMLElement|null}
 */
function node_element(node_id){
  return world_element.querySelector(`.node[data-id="${node_id}"]`);
}

/**
 * Inline editor for one node parameter (text / number / checkbox / select).
 * @param {object} node
 * @param {string} param_name
 * @returns {HTMLElement} input or select element
 */
function make_param_input(node, param_name){
  const editor = PARAM_EDITORS[param_name] || {type: 'text'}, current_value = node.params[param_name];
  let input_element;
  if(editor.type === 'select'){
    input_element = document.createElement('select');
    editor.options.forEach(option_text => { const option = document.createElement('option'); option.textContent = option_text; input_element.appendChild(option); });
    input_element.value = current_value;
  } else {
    input_element = document.createElement('input');
    input_element.type = editor.type === 'bool' ? 'checkbox' : editor.type === 'text' ? 'text' : 'number';
    if(editor.type === 'bool') input_element.checked = current_value; else input_element.value = current_value;
    if(editor.type === 'float') input_element.step = '0.05';
  }
  input_element.dataset.param = param_name;
  const fit_width = () => {
    if(editor.type === 'bool' || editor.type === 'select') return;
    const text_length = String(input_element.value).length;
    input_element.style.width = (editor.type === 'text' ? Math.max(2, text_length) * 1.15 + 0.6 : Math.max(1, text_length) + 0.4) + 'ch';
  };
  input_element.fit_width = fit_width;
  fit_width();
  const commit_value = is_final => {
    const new_value = editor.type === 'bool' ? input_element.checked
      : (editor.type === 'int' || editor.type === 'float') ? Number(input_element.value) : input_element.value;
    let is_valid = true;
    if(editor.type === 'int') is_valid = input_element.value !== '' && Number.isInteger(new_value) && (editor.min == null || new_value >= editor.min);
    if(editor.type === 'float') is_valid = input_element.value !== '' && Number.isFinite(new_value) && new_value >= (editor.min ?? -Infinity) && new_value <= (editor.max ?? Infinity);
    if(param_name === 'name') is_valid = new_value.trim() !== '';
    if(is_valid) node.params[param_name] = new_value;
    if(is_final && !is_valid) input_element.value = node.params[param_name];
    fit_width();
    refresh_view();
    measure_nodes_and_ports();
    if(param_name === 'name' && node.type === 'Input' && current_tab !== 'main') render_menus();
  };
  input_element.addEventListener('input', () => commit_value(false));
  input_element.addEventListener('change', () => commit_value(true));
  input_element.addEventListener('keydown', event => { if(event.key === 'Enter') input_element.blur(); event.stopPropagation(); });
  return input_element;
}

/**
 * Dim names shown for an Input node (its caller's rank when the shape comes from a calling graph).
 * @param {object} input_node
 * @returns {string[]}
 */
function shown_dim_names(input_node){
  const override_shape = current_input_overrides[input_node.id];
  return override_shape && override_shape.length !== input_dim_names(input_node).length ? default_dim_names(override_shape.length) : input_dim_names(input_node);
}

/**
 * Key that changes whenever the Input dims editor has to be rebuilt.
 * @param {object} input_node
 * @returns {string}
 */
function dim_editor_signature(input_node){
  const override_shape = current_input_overrides[input_node.id];
  return override_shape ? 'override:' + override_shape.join(',') + '|' + shown_dim_names(input_node).join(',')
    : input_node.params.dims + '|' + split_list_text(input_node.params.shape).length;
}

/**
 * One-line Input shape editor "B×3×32×32" (each size editable, dim name as tooltip).
 * @param {object} input_node
 * @returns {HTMLElement}
 */
function make_input_dims_editor(input_node){
  const override_shape = current_input_overrides[input_node.id];
  const sizes = override_shape || split_list_text(input_node.params.shape), dim_names = shown_dim_names(input_node);
  const editor_element = document.createElement('span');
  editor_element.className = 'dims';
  editor_element.dataset.signature = dim_editor_signature(input_node);
  editor_element.title = layout_text(dim_names);
  editor_element.append('B');
  const dim_count = Math.max(sizes.length, dim_names.length);
  for(let dim_index = 0; dim_index < dim_count; dim_index++){
    editor_element.append('×');
    let size_element;
    if(override_shape) size_element = String(sizes[dim_index]);
    else {
      size_element = document.createElement('input');
      size_element.type = 'number';
      size_element.min = 1;
      size_element.value = sizes[dim_index] ?? '';
      size_element.title = dim_names[dim_index] || '?';
      const fit_width = () => size_element.style.width = (Math.max(1, String(size_element.value).length) + 0.4) + 'ch';
      fit_width();
      const commit_size = is_final => {
        const shape_parts = split_list_text(input_node.params.shape), new_size = Number(size_element.value);
        if(size_element.value !== '' && Number.isInteger(new_size) && new_size >= 1){ shape_parts[dim_index] = String(new_size); input_node.params.shape = shape_parts.join(','); }
        else if(is_final) size_element.value = shape_parts[dim_index] ?? '';
        fit_width();
        refresh_view();
        measure_nodes_and_ports();
      };
      size_element.addEventListener('input', () => commit_size(false));
      size_element.addEventListener('change', () => commit_size(true));
      size_element.addEventListener('keydown', event => { if(event.key === 'Enter') size_element.blur(); event.stopPropagation(); });
    }
    editor_element.append(size_element);
  }
  return editor_element;
}

/** Rebuild every node element of the current tab, then refresh shapes, ports and arrows. */
function render_all(){
  all_graphs().forEach(drop_invalid_edges);
  const graph = current_graph();
  world_element.querySelectorAll('.node').forEach(element => element.remove());
  for(const node of graph.nodes){
    const element = document.createElement('div');
    element.dataset.id = node.id;
    element.style.left = node.x + 'px';
    element.style.top = node.y + 'px';
    if(node.type in OPERATOR_SPECS){
      element.className = 'node circ' + (LETTER_OPERATORS.includes(node.type) ? ' fn' : '');
      element.appendChild(document.createTextNode(OPERATOR_SPECS[node.type].symbol));
      const side_text = operator_side_label(node);
      if(side_text){ const label = document.createElement('span'); label.className = 'side'; label.textContent = side_text; element.appendChild(label); }
      if(node.type === 'Concat'){ const label = document.createElement('label'); label.className = 'side'; label.append('dim '); label.appendChild(make_param_input(node, 'dim')); element.appendChild(label); }
    } else {
      element.className = `node box k-${node_kind(node.type)}`;
      const title_element = document.createElement('div');
      title_element.className = 't';
      if(node.type === 'Activation') title_element.appendChild(make_param_input(node, 'fn'));
      else if(node.type === 'Input' || node.type === 'Output') title_element.appendChild(make_param_input(node, 'name'));
      else if(node.type === 'Custom') title_element.appendChild(make_block_name_input(node));
      else title_element.textContent = node_title(node);
      element.appendChild(title_element);
      const dims_element = document.createElement('div');
      dims_element.className = 'io';
      const make_span = class_name => { const span = document.createElement('span'); span.className = class_name; return span; };
      if(node.type === 'Input') dims_element.appendChild(make_input_dims_editor(node));
      else {
        dims_element.appendChild(make_span('in'));
        if(node.type !== 'Output'){
          dims_element.append(' → ');
          const inline_output = INLINE_OUTPUT_PARAM[node.type];
          if(inline_output) dims_element.append(make_span('pre'), make_param_input(node, inline_output.param_name), make_span('post'));
          else dims_element.appendChild(make_span('out'));
        }
      }
      if(node.type !== 'Activation') element.appendChild(dims_element);
    }
    world_element.appendChild(element);
  }
  render_tabs();
  render_menus();
  refresh_view();
  measure_nodes_and_ports();
}

/** Measure node sizes (unscaled), place port dots, redraw arrows. */
function measure_nodes_and_ports(){
  const graph = current_graph();
  for(const node of graph.nodes){
    const element = node_element(node.id);
    if(!element) continue;
    measured_sizes.set(node.id, {w: element.offsetWidth, h: element.offsetHeight});
    element.querySelectorAll('.port').forEach(port_element => port_element.remove());
    const port_count = input_port_count(graph, node);
    for(let port_index = 0; port_index < port_count; port_index++){
      const port_element = document.createElement('div');
      port_element.className = 'port in';
      port_element.dataset.port = port_index;
      const position = port_position(node, 'in', port_index);
      port_element.style.left = (position.x - node.x) + 'px';
      port_element.style.top = (position.y - node.y) + 'px';
      if(node.type === 'Custom'){
        const block_graph = state.blocks[node.params.block];
        const block_inputs = block_graph ? block_ports(block_graph).input_nodes : [];
        if(block_inputs[port_index]) port_element.title = block_inputs[port_index].params.name;
      }
      if(LAYER_SPECS[node.type] && LAYER_SPECS[node.type].input_port_names) port_element.title = LAYER_SPECS[node.type].input_port_names[port_index];
      element.appendChild(port_element);
    }
    if(has_output_port(node)){
      const port_element = document.createElement('div');
      port_element.className = 'port out';
      const position = port_position(node, 'out');
      port_element.style.left = (position.x - node.x) + 'px';
      port_element.style.top = (position.y - node.y) + 'px';
      element.appendChild(port_element);
    }
  }
  render_wires();
}

/**
 * Shapes of the current tab; a block tab uses the input shapes of its first use in the main graph.
 * @returns {{shapes: Object<number, number[]>, errors: Object<number, string>, order: number[]}}
 */
function infer_current_tab(){
  block_call_input_shapes = {};
  const main_inference = infer_shapes(state.main);
  if(current_tab === 'main'){ current_input_overrides = {}; return main_inference; }
  const block_graph = current_graph(), {input_nodes} = block_ports(block_graph), caller_shapes = block_call_input_shapes[current_tab];
  current_input_overrides = {};
  if(caller_shapes && caller_shapes.length === input_nodes.length) input_nodes.forEach((input_node, input_index) => current_input_overrides[input_node.id] = caller_shapes[input_index]);
  return infer_shapes(block_graph, current_input_overrides, [current_tab]);
}

/** Update shapes, errors, selection, arrows and code without rebuilding node elements (keeps input focus). */
function refresh_view(){
  current_inference = infer_current_tab();
  const graph = current_graph();
  for(const node of graph.nodes){
    const element = node_element(node.id);
    if(!element) continue;
    const error_text = current_inference.errors[node.id], output_shape = current_inference.shapes[node.id];
    element.classList.toggle('err', !!error_text);
    element.classList.toggle('sel', selected_node_ids.has(node.id));
    element.title = (node.type in OPERATOR_SPECS ? OPERATOR_SPECS[node.type].label : node_title(node))
      + (output_shape ? `\n→ [B, ${output_shape.join(', ')}]` : '') + (error_text ? `\n⚠ ${error_text}` : '');
    if(node.type === 'Input'){
      const dims_editor = element.querySelector('.dims');
      if(dims_editor.dataset.signature !== dim_editor_signature(node)) dims_editor.replaceWith(make_input_dims_editor(node));
      element.title = `${node.params.name}: (${layout_text(shown_dim_names(node))})` + (output_shape ? ` = [B, ${output_shape.join(', ')}]` : '') + (error_text ? `\n⚠ ${error_text}` : '');
    }
    if(node.type === 'Output'){
      const input_span = element.querySelector('.io .in'), source_id = input_sources(graph, node)[0];
      if(input_span) input_span.textContent = source_id != null && current_inference.shapes[source_id] ? 'B×' + dims_text(current_inference.shapes[source_id]) : '?';
      continue;
    }
    if(node.type in OPERATOR_SPECS || node.type === 'Input') continue;
    const source_ids = input_sources(graph, node);
    const set_text = (class_name, text) => { const span = element.querySelector('.io .' + class_name); if(span) span.textContent = text; };
    if(node.type === 'Custom' && source_ids.length > 1){ element.querySelector('.io').textContent = dims_text(output_shape); continue; }
    const shown_sources = node.type === 'Attention' ? source_ids.slice(0, 1) : source_ids;
    set_text('in', shown_sources.map(source_id => source_id != null ? dims_text(current_inference.shapes[source_id]) : '?').join(', ') || '?');
    const inline_output = INLINE_OUTPUT_PARAM[node.type];
    if(inline_output){
      let dim_index = inline_output.output_dim_index;
      if(output_shape){
        if(dim_index < 0) dim_index = output_shape.length - 1;
        set_text('pre', output_shape.slice(0, dim_index).map(size => size + '×').join(''));
        set_text('post', output_shape.slice(dim_index + 1).map(size => '×' + size).join(''));
      } else { set_text('pre', ''); set_text('post', ''); }
    }
    else set_text('out', dims_text(output_shape));
  }
  render_wires();
  render_code_panel();
  save_to_local_storage();
}

/** Redraw every arrow of the current tab (plus the arrow being dragged). */
function render_wires(){
  wire_group_element.innerHTML = '';
  const graph = current_graph();
  compute_edge_lanes(graph);
  for(const edge of graph.edges){
    const path_data = edge_path(graph, edge), is_selected = selected_edge_id === edge.id;
    const visible_path = document.createElementNS(SVG_NAMESPACE, 'path');
    visible_path.setAttribute('d', path_data);
    visible_path.setAttribute('class', 'w' + (is_selected ? ' sel' : ''));
    visible_path.setAttribute('marker-end', is_selected ? 'url(#arrow_head_selected)' : 'url(#arrow_head)');
    wire_group_element.appendChild(visible_path);
    const hit_path = document.createElementNS(SVG_NAMESPACE, 'path');
    hit_path.setAttribute('d', path_data);
    hit_path.setAttribute('class', 'hit');
    hit_path.dataset.edge = edge.id;
    wire_group_element.appendChild(hit_path);
  }
  if(drag_state && drag_state.kind === 'wire'){
    const dragged_path = document.createElementNS(SVG_NAMESPACE, 'path');
    dragged_path.setAttribute('d', elbow_path(drag_state.start_point, drag_state.end_point));
    dragged_path.setAttribute('class', 'w tmp');
    dragged_path.setAttribute('marker-end', 'url(#arrow_head_selected)');
    wire_group_element.appendChild(dragged_path);
  }
  place_side_labels();
}

/** Put each operator label on the first spot around its circle that no arrow, box or other label touches. */
function place_side_labels(){
  const graph = current_graph();
  side_label_boxes.clear();
  const arrow_segments = [];
  graph.edges.forEach(edge => {
    const points = path_points(edge_path(graph, edge));
    for(let point_index = 1; point_index < points.length; point_index++) arrow_segments.push([points[point_index - 1], points[point_index]]);
  });
  const node_boxes = graph.nodes.map(node => ({id: node.id, x: node.x, y: node.y, w: node_size(node).w, h: node_size(node).h}));
  for(const node of graph.nodes){
    if(!(node.type in OPERATOR_SPECS)) continue;
    const label_element = node_element(node.id).querySelector('.side');
    if(!label_element) continue;
    const label_width = label_element.offsetWidth, label_height = label_element.offsetHeight;
    const radius = OPERATOR_RADIUS, center_x = node.x + radius, center_y = node.y + radius, diagonal_offset = radius * 0.6;
    const candidate_spots = [
      [center_x + diagonal_offset, center_y - radius - label_height + 2], [center_x + diagonal_offset, center_y + radius - 2],
      [center_x - diagonal_offset - label_width, center_y - radius - label_height + 2], [center_x - diagonal_offset - label_width, center_y + radius - 2],
      [center_x + radius + 4, center_y - label_height / 2], [center_x - radius - 4 - label_width, center_y - label_height / 2],
      [center_x + diagonal_offset + 8, center_y - radius - label_height - 8], [center_x + diagonal_offset + 8, center_y + radius + 8],
    ];
    const is_clear = ([spot_x, spot_y]) => {
      const label_rect = {x: spot_x - 2, y: spot_y - 1, w: label_width + 4, h: label_height + 2};
      return !arrow_segments.some(([start_point, end_point]) => segment_hits_rect(start_point, end_point, label_rect))
        && !node_boxes.some(box => box.id !== node.id && rects_overlap(label_rect, box))
        && ![...side_label_boxes.values()].some(other_rect => rects_overlap(label_rect, other_rect));
    };
    const [label_x, label_y] = candidate_spots.find(is_clear) || candidate_spots[0];
    side_label_boxes.set(node.id, {x: label_x, y: label_y, w: label_width, h: label_height});
    label_element.style.left = (label_x - node.x) + 'px';
    label_element.style.top = (label_y - node.y) + 'px';
  }
}
