// Graph data: the main graph and custom block graphs, nodes, edges and input ports.
// Graph  = {nodes: Node[], edges: Edge[], needs_layout?: boolean}
// Node   = {id, type, x, y, params, grid_column?, grid_row?}
// Edge   = {id, from: node id, to: node id, port: input port index of the target}

let state = {main: {nodes: [], edges: []}, blocks: {}, next_id: 1};
let current_tab = 'main';
let open_tabs = [];
let tab_views = {};
let view_transform = {x: 40, y: 40, scale: 1};
let selected_node_ids = new Set();
let selected_edge_id = null;
const measured_sizes = new Map();

/** Clear node and edge selection. */
function clear_selection(){
  selected_node_ids = new Set();
  selected_edge_id = null;
}

/**
 * Deep copy of plain JSON data.
 * @param {*} value
 * @returns {*} copy
 */
function deep_copy(value){
  return JSON.parse(JSON.stringify(value));
}

/**
 * Graph shown in the current tab.
 * @returns {object} graph
 */
function current_graph(){
  return current_tab === 'main' ? state.main : state.blocks[current_tab];
}

/**
 * Main graph followed by every block graph.
 * @returns {object[]} graphs
 */
function all_graphs(){
  return [state.main, ...Object.values(state.blocks)];
}

/**
 * Find a node by id in a graph.
 * @param {object} graph
 * @param {number} node_id
 * @returns {object|undefined} node
 */
function find_node(graph, node_id){
  return graph.nodes.find(node => node.id === node_id);
}

/**
 * Find a node by id in the current tab.
 * @param {number} node_id
 * @returns {object|undefined} node
 */
function find_current_node(node_id){
  return find_node(current_graph(), node_id);
}

/** Sort comparator: left to right, then top to bottom. */
function compare_by_position(first_node, second_node){
  return first_node.x - second_node.x || first_node.y - second_node.y;
}

/** Sort comparator: creation order. */
function compare_by_id(first_node, second_node){
  return first_node.id - second_node.id;
}

/**
 * Name shown on a node.
 * @param {object} node
 * @returns {string} block name, activation name, Input/Output name, or the node type
 */
function node_title(node){
  if(node.type === 'Custom') return node.params.block;
  if(node.type === 'Activation') return node.params.fn;
  if(node.type === 'Input' || node.type === 'Output') return node.params.name;
  return node.type;
}

/**
 * Parameters for a new node.
 * @param {string} node_type
 * @param {object} extra  {preset?, shape?, block?, fn?}
 * @param {object} graph  graph the node goes into (for unique Input/Output names)
 * @returns {object} params
 */
function default_params(node_type, extra, graph){
  const unique_name = (type, base_name) => {
    let suffix = 1, name = base_name;
    while(graph.nodes.some(node => node.type === type && node.params.name === name)) name = base_name + (++suffix);
    return name;
  };
  if(node_type === 'Input'){
    if(extra.preset){
      const preset = INPUT_LAYOUT_PRESETS[extra.preset];
      return {name: unique_name('Input', 'x'), dims: preset.dim_names.join(','), shape: preset.shape.join(',')};
    }
    const shape_parts = split_list_text(extra.shape || '784');
    return {name: unique_name('Input', 'x'), dims: default_dim_names(shape_parts.length).join(','), shape: shape_parts.join(',')};
  }
  if(node_type === 'Output') return {name: unique_name('Output', 'y')};
  if(node_type === 'Concat') return {dim: -1};
  if(node_type === 'MatMul') return {transpose_b: false};
  if(node_type === 'Div') return {by: ''};
  if(node_type === 'Reshape') return {to: '-1'};
  if(node_type === 'Permute') return {order: '0,2,1'};
  if(node_type === 'Transpose') return {dim0: 1, dim1: 2};
  if(node_type in OPERATOR_SPECS) return {};
  if(node_type === 'Custom') return {block: extra.block};
  const params = deep_copy(LAYER_SPECS[node_type].default_params);
  if(node_type === 'Activation' && extra.fn) params.fn = extra.fn;
  return params;
}

/**
 * Create a node and add it to a graph.
 * @param {string} node_type
 * @param {number} x  world position
 * @param {number} y
 * @param {object} extra  see default_params
 * @param {object} graph
 * @returns {object} node
 */
function add_node(node_type, x, y, extra = {}, graph = current_graph()){
  const node = {id: state.next_id++, type: node_type, x: Math.round(x), y: Math.round(y), params: default_params(node_type, extra, graph)};
  graph.nodes.push(node);
  return node;
}

/**
 * Input and Output nodes of a block graph, in creation order (= argument order).
 * @param {object} block_graph
 * @returns {{input_nodes: object[], output_nodes: object[]}}
 */
function block_ports(block_graph){
  return {
    input_nodes: block_graph.nodes.filter(node => node.type === 'Input').sort(compare_by_id),
    output_nodes: block_graph.nodes.filter(node => node.type === 'Output').sort(compare_by_id),
  };
}

/**
 * Whether a node is an operator that takes two or more operands.
 * @param {object} node
 * @returns {boolean}
 */
function is_variadic_operator(node){
  return node.type in OPERATOR_SPECS && !!OPERATOR_SPECS[node.type].variadic;
}

/**
 * Number of operands a node takes (operators: fixed operand count).
 * @param {object} node
 * @returns {number}
 */
function fixed_input_count(node){
  if(node.type === 'Input') return 0;
  if(node.type in OPERATOR_SPECS) return OPERATOR_SPECS[node.type].input_count;
  if(node.type === 'Custom'){
    const block_graph = state.blocks[node.params.block];
    return block_graph ? Math.max(1, block_ports(block_graph).input_nodes.length) : 1;
  }
  if(LAYER_SPECS[node.type] && LAYER_SPECS[node.type].input_port_names) return LAYER_SPECS[node.type].input_port_names.length;
  return 1;
}

/**
 * Edges into a node, ordered by port.
 * @param {object} graph
 * @param {object} node
 * @returns {object[]} edges
 */
function incoming_edges(graph, node){
  return graph.edges.filter(edge => edge.to === node.id).sort((first_edge, second_edge) => first_edge.port - second_edge.port);
}

/**
 * Number of input port positions a node shows (operator circles: top, left, right, plus extra top slots when full).
 * @param {object} graph
 * @param {object} node
 * @returns {number}
 */
function input_port_count(graph, node){
  if(!(node.type in OPERATOR_SPECS)) return fixed_input_count(node);
  if(!is_variadic_operator(node)) return 3;
  const edges_in = incoming_edges(graph, node);
  return Math.max(3, edges_in.length + 1, ...edges_in.map(edge => edge.port + 1));
}

/**
 * Whether a node has an output port.
 * @param {object} node
 * @returns {boolean}
 */
function has_output_port(node){
  return node.type !== 'Output';
}

/**
 * Remove edges whose nodes or ports no longer exist.
 * @param {object} graph
 */
function drop_invalid_edges(graph){
  graph.edges = graph.edges.filter(edge => {
    const target_node = find_node(graph, edge.to);
    if(!target_node || !find_node(graph, edge.from)) return false;
    if(target_node.type in OPERATOR_SPECS) return is_variadic_operator(target_node) || edge.port < 3;
    return edge.port < fixed_input_count(target_node);
  });
}

/**
 * Source node ids of a node's inputs, in operand order (null for an unconnected port of a non-operator).
 * @param {object} graph
 * @param {object} node
 * @returns {(number|null)[]}
 */
function input_sources(graph, node){
  const edges_in = incoming_edges(graph, node);
  if(node.type in OPERATOR_SPECS) return edges_in.map(edge => edge.from);
  return Array.from({length: fixed_input_count(node)}, (_, port_index) => {
    const edge = edges_in.find(candidate => candidate.port === port_index);
    return edge ? edge.from : null;
  });
}

/**
 * Whether block `block_name` uses block `target_name`, directly or through nested blocks.
 * @param {string} block_name
 * @param {string} target_name
 * @param {Set<string>} visited_names
 * @returns {boolean}
 */
function block_contains(block_name, target_name, visited_names = new Set()){
  if(block_name === target_name) return true;
  if(visited_names.has(block_name)) return false;
  visited_names.add(block_name);
  const block_graph = state.blocks[block_name];
  if(!block_graph) return false;
  return block_graph.nodes.some(node => node.type === 'Custom' && block_contains(node.params.block, target_name, visited_names));
}

/**
 * Whether a block may be placed in the current tab (no recursion).
 * @param {string} block_name
 * @returns {boolean}
 */
function can_place_block(block_name){
  return current_tab === 'main' || !block_contains(block_name, current_tab);
}

/**
 * Whether `to_id` is reachable from `from_id` along edges.
 * @param {object} graph
 * @param {number} from_id
 * @param {number} to_id
 * @returns {boolean}
 */
function path_exists(graph, from_id, to_id){
  const pending_ids = [from_id], visited_ids = new Set();
  while(pending_ids.length){
    const node_id = pending_ids.pop();
    if(node_id === to_id) return true;
    if(visited_ids.has(node_id)) continue;
    visited_ids.add(node_id);
    graph.edges.filter(edge => edge.from === node_id).forEach(edge => pending_ids.push(edge.to));
  }
  return false;
}

/**
 * Connect a node's output to another node's input port.
 * @param {number} from_id
 * @param {number} to_id
 * @param {number|null} port_index  null: first free port
 * @param {object} graph
 * @returns {boolean} false when the connection would be invalid (self, Input target, cycle)
 */
function connect_nodes(from_id, to_id, port_index, graph = current_graph()){
  const target_node = find_node(graph, to_id);
  if(from_id === to_id || !target_node || target_node.type === 'Input' || path_exists(graph, to_id, from_id)) return false;
  if(port_index == null){
    const used_ports = new Set(incoming_edges(graph, target_node).map(edge => edge.port));
    port_index = 0;
    while(used_ports.has(port_index)) port_index++;
    const port_limit = target_node.type in OPERATOR_SPECS ? input_port_count(graph, target_node) : fixed_input_count(target_node);
    if(port_index >= port_limit && !is_variadic_operator(target_node)) port_index = port_limit - 1;
  }
  graph.edges = graph.edges.filter(edge => !(edge.to === to_id && edge.port === port_index));
  graph.edges.push({id: 'e' + (state.next_id++), from: from_id, to: to_id, port: port_index});
  return true;
}

/**
 * Remove a node and its edges.
 * @param {number} node_id
 * @param {object} graph
 */
function remove_node(node_id, graph = current_graph()){
  graph.nodes = graph.nodes.filter(node => node.id !== node_id);
  graph.edges = graph.edges.filter(edge => edge.from !== node_id && edge.to !== node_id);
  measured_sizes.delete(node_id);
}
