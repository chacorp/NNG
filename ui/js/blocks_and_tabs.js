// Custom blocks: tabs, renaming, grouping a selection into a block, copy / paste, attention block.

let clipboard = null;

/** Draw the tab bar (Main + open block tabs). */
function render_tabs(){
  const tab_bar = document.getElementById('tab_bar');
  tab_bar.innerHTML = '';
  const add_tab = (tab_key, label) => {
    const tab_element = document.createElement('div');
    tab_element.className = 'tab' + (tab_key === current_tab ? ' on' : '');
    tab_element.dataset.tab = tab_key;
    if(tab_key !== 'main'){ const swatch = document.createElement('span'); swatch.className = 'sw'; tab_element.appendChild(swatch); }
    tab_element.append(label);
    if(tab_key !== 'main'){
      const close_button = document.createElement('button');
      close_button.className = 'x';
      close_button.textContent = '×';
      close_button.title = '탭 닫기';
      close_button.onclick = event => { event.stopPropagation(); close_tab(tab_key); };
      tab_element.appendChild(close_button);
    }
    tab_element.onclick = () => switch_tab(tab_key);
    tab_bar.appendChild(tab_element);
  };
  add_tab('main', 'Main');
  open_tabs.forEach(block_name => add_tab(block_name, block_name));
}

/**
 * Block name editor shown on a Custom node (renames the block everywhere it is used).
 * @param {object} custom_node
 * @returns {HTMLInputElement}
 */
function make_block_name_input(custom_node){
  const input_element = document.createElement('input');
  input_element.type = 'text';
  input_element.value = custom_node.params.block;
  input_element.dataset.param = 'block';
  const fit_width = () => input_element.style.width = (Math.max(2, input_element.value.length) * 1.15 + 0.6) + 'ch';
  fit_width();
  input_element.addEventListener('input', fit_width);
  input_element.addEventListener('keydown', event => {
    if(event.key === 'Enter') input_element.blur();
    if(event.key === 'Escape'){ input_element.value = custom_node.params.block; input_element.blur(); }
    event.stopPropagation();
  });
  input_element.addEventListener('change', () => {
    const rename_error = rename_block(custom_node.params.block, input_element.value.trim());
    if(rename_error){ alert(rename_error); input_element.value = custom_node.params.block; fit_width(); }
  });
  return input_element;
}

/**
 * Show a tab ('main' or a block name).
 * @param {string} tab_key
 */
function switch_tab(tab_key){
  if(tab_key !== 'main' && !state.blocks[tab_key]) tab_key = 'main';
  tab_views[current_tab] = {...view_transform};
  close_popup();
  if(tab_key !== 'main' && !open_tabs.includes(tab_key)) open_tabs.push(tab_key);
  current_tab = tab_key;
  clear_selection();
  measured_sizes.clear();
  render_all();
  if(current_graph().needs_layout){
    delete current_graph().needs_layout;
    if(!grid_layout(current_graph())) auto_layout();
    render_all();
    fit_view();
    return;
  }
  if(tab_views[tab_key]){ view_transform = {...tab_views[tab_key]}; apply_view_transform(); }
  else fit_view();
}

/**
 * Close a block tab.
 * @param {string} tab_key
 */
function close_tab(tab_key){
  open_tabs = open_tabs.filter(open_key => open_key !== tab_key);
  if(current_tab === tab_key) switch_tab('main');
  else { render_tabs(); save_to_local_storage(); }
}

/**
 * Validation message for a new block name.
 * @param {string} new_name
 * @param {string} old_name
 * @returns {string|null} error message, or null when the name is fine
 */
function block_name_error(new_name, old_name){
  if(!/^[A-Za-z_][A-Za-z0-9_]*$/.test(new_name)) return '이름은 영문/숫자/_ 로, 숫자로 시작할 수 없어요.';
  if(['Model', 'nn', 'torch', 'custom_function', 'main'].includes(new_name)) return '예약된 이름입니다.';
  if(new_name !== old_name && state.blocks[new_name]) return '같은 이름의 블록이 이미 있어요.';
  return null;
}

/**
 * Rename a block and every node that uses it.
 * @param {string} old_name
 * @param {string} new_name
 * @returns {string|null} error message, or null on success
 */
function rename_block(old_name, new_name){
  if(new_name === old_name) return null;
  const name_error = block_name_error(new_name, old_name);
  if(name_error) return name_error;
  state.blocks[new_name] = state.blocks[old_name];
  delete state.blocks[old_name];
  all_graphs().forEach(graph => graph.nodes.forEach(node => { if(node.type === 'Custom' && node.params.block === old_name) node.params.block = new_name; }));
  open_tabs = open_tabs.map(open_key => open_key === old_name ? new_name : open_key);
  if(tab_views[old_name]){ tab_views[new_name] = tab_views[old_name]; delete tab_views[old_name]; }
  if(current_tab === old_name) current_tab = new_name;
  render_all();
  return null;
}

/** Ctrl+G: replace the selected nodes by one new custom block (outside sources become its inputs). */
function group_selection_into_block(){
  const graph = current_graph(), grouped_ids = new Set([...selected_node_ids].filter(node_id => find_node(graph, node_id)));
  const is_inside = node_id => grouped_ids.has(node_id);
  if(!grouped_ids.size) return;
  if([...grouped_ids].some(node_id => ['Input', 'Output'].includes(find_node(graph, node_id).type)))
    return alert('Input/Output 노드는 블록으로 묶을 수 없어요. 레이어와 연산자만 선택해 주세요.');
  const entering_edges = graph.edges.filter(edge => !is_inside(edge.from) && is_inside(edge.to));
  const leaving_edges = graph.edges.filter(edge => is_inside(edge.from) && !is_inside(edge.to));
  const internal_edges = graph.edges.filter(edge => is_inside(edge.from) && is_inside(edge.to));
  const leaving_source_ids = [...new Set(leaving_edges.map(edge => edge.from))];
  if(!entering_edges.length) return alert('선택 영역으로 들어오는 연결이 없어요.');
  if(leaving_source_ids.length !== 1)
    return alert(leaving_source_ids.length ? `선택 영역 밖으로 나가는 출력이 ${leaving_source_ids.length}개예요. 커스텀 블록은 출력 1개만 지원해요.` : '선택 영역에서 밖으로 나가는 연결이 없어요.');
  if(leaving_edges.some(edge => [...grouped_ids].some(node_id => path_exists(graph, edge.to, node_id))))
    return alert('선택 영역 밖을 거쳐 다시 들어오는 경로가 있어서 묶을 수 없어요.');
  const outside_source_ids = [...new Set(entering_edges.map(edge => edge.from))].sort((first_id, second_id) => compare_by_position(find_node(graph, first_id), find_node(graph, second_id)));
  let block_number = 1;
  while(state.blocks['Block' + block_number]) block_number++;
  const block_name = 'Block' + block_number;
  const moved_nodes = graph.nodes.filter(node => is_inside(node.id));
  const left = Math.min(...moved_nodes.map(node => node.x)), top = Math.min(...moved_nodes.map(node => node.y));
  const right = Math.max(...moved_nodes.map(node => node.x + node_size(node).w)), bottom = Math.max(...moved_nodes.map(node => node.y + node_size(node).h));

  const block_graph = {nodes: [...moved_nodes], edges: [...internal_edges]};
  const block_input_nodes = outside_source_ids.map((source_id, input_index) => {
    const source_shape = current_inference.shapes[source_id], source_node = find_node(graph, source_id);
    const input_node = add_node('Input', left + input_index * 170, top - 90, {shape: source_shape ? source_shape.join(',') : DEFAULT_HIDDEN_SIZE}, block_graph);
    if(source_node.type === 'Input' && source_shape && shown_dim_names(source_node).length === source_shape.length) input_node.params.dims = shown_dim_names(source_node).join(',');
    return input_node;
  });
  entering_edges.forEach(edge => block_graph.edges.push({id: 'e' + (state.next_id++), from: block_input_nodes[outside_source_ids.indexOf(edge.from)].id, to: edge.to, port: edge.port}));
  const block_output_node = add_node('Output', (left + right) / 2 - 65, bottom + 50, {}, block_graph);
  block_graph.edges.push({id: 'e' + (state.next_id++), from: leaving_source_ids[0], to: block_output_node.id, port: 0});
  state.blocks[block_name] = block_graph;

  graph.nodes = graph.nodes.filter(node => !is_inside(node.id));
  graph.edges = graph.edges.filter(edge => !is_inside(edge.from) && !is_inside(edge.to));
  const custom_node = add_node('Custom', (left + right) / 2 - 65, (top + bottom) / 2 - 20, {block: block_name}, graph);
  outside_source_ids.forEach((source_id, input_index) => graph.edges.push({id: 'e' + (state.next_id++), from: source_id, to: custom_node.id, port: input_index}));
  leaving_edges.forEach(edge => graph.edges.push({id: 'e' + (state.next_id++), from: custom_node.id, to: edge.to, port: edge.port}));
  moved_nodes.forEach(node => measured_sizes.delete(node.id));
  selected_node_ids = new Set([custom_node.id]);
  selected_edge_id = null;
  render_all();
}

/** Ctrl+C: copy the selected nodes and the edges between them. */
function copy_selection(){
  const graph = current_graph(), copied_ids = new Set([...selected_node_ids].filter(node_id => find_node(graph, node_id)));
  clipboard = {
    nodes: deep_copy(graph.nodes.filter(node => copied_ids.has(node.id))),
    edges: deep_copy(graph.edges.filter(edge => copied_ids.has(edge.from) && copied_ids.has(edge.to))),
  };
}

/** Ctrl+V: paste the copied nodes slightly offset into the current tab. */
function paste_clipboard(){
  const graph = current_graph(), new_id_by_old_id = {};
  const pastable_nodes = clipboard.nodes.filter(node => node.type !== 'Custom' || (state.blocks[node.params.block] && can_place_block(node.params.block)));
  pastable_nodes.forEach(node => {
    const pasted_node = deep_copy(node);
    pasted_node.id = state.next_id++;
    pasted_node.x += 30;
    pasted_node.y += 30;
    new_id_by_old_id[node.id] = pasted_node.id;
    graph.nodes.push(pasted_node);
  });
  clipboard.edges.forEach(edge => {
    if(new_id_by_old_id[edge.from] && new_id_by_old_id[edge.to])
      graph.edges.push({id: 'e' + (state.next_id++), from: new_id_by_old_id[edge.from], to: new_id_by_old_id[edge.to], port: edge.port});
  });
  clipboard.nodes.forEach(node => { node.x += 30; node.y += 30; });
  selected_node_ids = new Set(Object.values(new_id_by_old_id));
  selected_edge_id = null;
  render_all();
}

/**
 * Add a scaled dot-product attention block: q, k, v → Linear each → q·kᵀ → ÷√d → (causal mask) → Softmax → ·v → Linear.
 * @param {string} block_name
 * @param {number} embed_dim     E, size of the last dim of q, k, v
 * @param {number} query_length  T, sequence length of q
 * @param {{masked?: boolean, key_length?: number}} options  causal mask; S, sequence length of k and v
 * Block inputs: q [T, E], k [S, E], v [S, E]. Output: [T, E].
 */
function build_attention_block(block_name, embed_dim, query_length = 16, {masked = false, key_length = query_length} = {}){
  const {add, link, chain} = graph_builder(new_block_graph(block_name));
  const query_input = add('Input', {shape: `${query_length},${embed_dim}`});
  const key_input = add('Input', {shape: `${key_length},${embed_dim}`});
  const value_input = add('Input', {shape: `${key_length},${embed_dim}`});
  query_input.params.name = 'q'; key_input.params.name = 'k'; value_input.params.name = 'v';
  const query_projection = add('Linear', {}, {out_features: embed_dim});
  const key_projection = add('Linear', {}, {out_features: embed_dim});
  const value_projection = add('Linear', {}, {out_features: embed_dim});
  link(query_input, query_projection); link(key_input, key_projection); link(value_input, value_projection);
  const attention_scores = add('MatMul', {}, {transpose_b: true});
  link(query_projection, attention_scores, 0); link(key_projection, attention_scores, 2);
  const scale_by_sqrt_dim = add('Div', {}, {by: `sqrt(${embed_dim})`});
  const causal_mask = masked ? add('CausalMask') : null;
  const softmax = add('Activation', {fn: 'Softmax'});
  chain(...[attention_scores, scale_by_sqrt_dim, causal_mask, softmax].filter(Boolean));
  const weighted_values = add('MatMul');
  link(softmax, weighted_values, 0); link(value_projection, weighted_values, 2);
  const output_projection = add('Linear', {}, {out_features: embed_dim}), output_node = add('Output');
  chain(weighted_values, output_projection, output_node);
  const mask_rows = masked ? 1 : 0;
  [[query_input, 0, 0], [key_input, 1, 0], [value_input, 2, 0],
   [query_projection, 0, 1], [key_projection, 1, 1], [value_projection, 2, 1],
   [attention_scores, 0, 2], [scale_by_sqrt_dim, 0, 3], ...(causal_mask ? [[causal_mask, 0, 4]] : []),
   [softmax, 0, 4 + mask_rows], [weighted_values, 0, 5 + mask_rows], [output_projection, 0, 6 + mask_rows], [output_node, 0, 7 + mask_rows]]
    .forEach(([node, grid_column, grid_row]) => { node.grid_column = grid_column; node.grid_row = grid_row; });
}

/** Menu action: add a new attention block (embed dim DEFAULT_HIDDEN_SIZE) and place one instance. */
function add_attention_block(){
  let block_number = 1, block_name = 'Attention';
  while(state.blocks[block_name]) block_name = 'Attention' + (++block_number);
  build_attention_block(block_name, DEFAULT_HIDDEN_SIZE);
  add_node_at_view_center('Custom', {block: block_name});
}
