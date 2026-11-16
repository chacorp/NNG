// Automatic placement of nodes and fitting the view.

/**
 * Place nodes that carry designed grid cells (grid_column, grid_row), centred in each cell.
 * @param {object} graph
 * @returns {boolean} false when no node has a grid cell
 */
function grid_layout(graph){
  const grid_nodes = graph.nodes.filter(node => node.grid_column != null);
  if(!grid_nodes.length) return false;
  const column_width = Math.max(...grid_nodes.map(node => node_size(node).w)) + 40;
  const row_height = Math.max(...grid_nodes.map(node => node_size(node).h)) + 24;
  grid_nodes.forEach(node => {
    node.x = Math.round(node.grid_column * column_width - node_size(node).w / 2);
    node.y = Math.round(node.grid_row * row_height - node_size(node).h / 2);
  });
  return true;
}

/**
 * Top-to-bottom layout of the current tab.
 * Each node sits straight below its first input (the input→output axis is one straight line); a straight edge keeps
 * its column free on the layers it crosses, so side branches step to the right. Sources that only feed others from
 * the side (e.g. a decoder's memory) go to the top right.
 */
function auto_layout(){
  const graph = current_graph();
  if(graph.nodes.length && graph.nodes.every(node => node.grid_column != null)){ grid_layout(graph); return; }
  const ordered_ids = topological_order(graph), depth_by_id = {};
  ordered_ids.forEach(node_id => {
    const parent_depths = graph.edges.filter(edge => edge.to === node_id).map(edge => depth_by_id[edge.from] ?? 0);
    depth_by_id[node_id] = parent_depths.length ? Math.max(...parent_depths) + 1 : 0;
  });
  graph.nodes.forEach(node => { if(depth_by_id[node.id] == null) depth_by_id[node.id] = 0; });

  const anchor_edge_by_id = {};
  graph.nodes.forEach(node => {
    const edges_in = incoming_edges(graph, node);
    if(edges_in.length) anchor_edge_by_id[node.id] = edges_in.find(edge => edge.port === 0) || edges_in[0];
  });

  const chain_root_by_id = {}, chain_width_by_root = {};
  ordered_ids.forEach(node_id => { chain_root_by_id[node_id] = anchor_edge_by_id[node_id] ? chain_root_by_id[anchor_edge_by_id[node_id].from] : node_id; });
  graph.nodes.forEach(node => {
    const root_id = chain_root_by_id[node.id] ?? node.id;
    chain_width_by_root[root_id] = Math.max(chain_width_by_root[root_id] || 0, node_size(node).w);
  });

  const anchoring_source_ids = new Set(Object.values(anchor_edge_by_id).map(edge => edge.from));
  const side_source_ids = new Set(graph.nodes
    .filter(node => !incoming_edges(graph, node).length && !anchoring_source_ids.has(node.id) && graph.edges.some(edge => edge.from === node.id))
    .map(node => node.id));

  const layer_count = Math.max(0, ...Object.values(depth_by_id)) + 1, layers = Array.from({length: layer_count}, () => []);
  graph.nodes.forEach(node => { if(!side_source_ids.has(node.id)) layers[depth_by_id[node.id]].push(node); });
  const horizontal_gap = 28, vertical_gap = 26, center_x_by_id = {};
  let row_top = 0, first_row = null;
  layers.forEach((layer_nodes, layer_depth) => {
    if(!layer_nodes.length) return;
    if(first_row == null) first_row = {top: row_top, height: Math.max(...layer_nodes.map(node => node_size(node).h))};
    const occupied_intervals = [];
    Object.entries(anchor_edge_by_id).forEach(([target_id, edge]) => {
      if(depth_by_id[edge.from] < layer_depth && layer_depth < depth_by_id[+target_id] && center_x_by_id[edge.from] != null)
        occupied_intervals.push([center_x_by_id[edge.from] - 10, center_x_by_id[edge.from] + 10]);
    });
    const placements = layer_nodes.map(node => ({node, width: node_size(node).w, wanted_x: anchor_edge_by_id[node.id] ? center_x_by_id[anchor_edge_by_id[node.id].from] : null}));
    placements.sort((first, second) => (first.wanted_x ?? Infinity) - (second.wanted_x ?? Infinity) || first.node.id - second.node.id);
    for(const placement of placements){
      let center_x = placement.wanted_x;
      if(center_x == null){
        const free_left = occupied_intervals.length ? Math.max(...occupied_intervals.map(interval => interval[1])) + horizontal_gap : 0;
        const chain_width = chain_width_by_root[placement.node.id] || placement.width;
        center_x = free_left + chain_width / 2;
        occupied_intervals.push([free_left, free_left + chain_width]);
      }
      for(let attempt = 0; attempt < 200 && placement.wanted_x != null; attempt++){
        const blocking_interval = occupied_intervals.find(interval => interval[0] < center_x + placement.width / 2 + horizontal_gap / 2 && interval[1] > center_x - placement.width / 2 - horizontal_gap / 2);
        if(!blocking_interval) break;
        center_x = blocking_interval[1] + horizontal_gap / 2 + placement.width / 2 + 0.5;
      }
      if(placement.wanted_x != null) occupied_intervals.push([center_x - placement.width / 2, center_x + placement.width / 2]);
      center_x_by_id[placement.node.id] = center_x;
      placement.center_x = center_x;
    }
    const row_height = Math.max(...layer_nodes.map(node => node_size(node).h));
    placements.forEach(placement => {
      placement.node.x = Math.round(placement.center_x) - placement.width / 2;
      placement.node.y = Math.round(row_top + (row_height - node_size(placement.node).h) / 2);
    });
    row_top += row_height + vertical_gap;
  });

  let right_edge = Math.max(0, ...graph.nodes.filter(node => !side_source_ids.has(node.id)).map(node => node.x + node_size(node).w));
  graph.nodes.filter(node => side_source_ids.has(node.id)).forEach(node => {
    const width = node_size(node).w;
    node.x = Math.round(right_edge + horizontal_gap + width / 2) - width / 2;
    right_edge = node.x + width;
    node.y = Math.round((first_row ? first_row.top : 0) + ((first_row ? first_row.height : node_size(node).h) - node_size(node).h) / 2);
  });
}

/** Zoom and pan so the whole current tab is visible. */
function fit_view(){
  const graph = current_graph();
  if(!graph.nodes.length){ view_transform = {x: 40, y: 40, scale: 1}; return apply_view_transform(); }
  const canvas_rect = canvas_element.getBoundingClientRect();
  const left = Math.min(...graph.nodes.map(node => node.x)), top = Math.min(...graph.nodes.map(node => node.y));
  const right = Math.max(...graph.nodes.map(node => node.x + node_size(node).w)), bottom = Math.max(...graph.nodes.map(node => node.y + node_size(node).h));
  const scale = Math.min(1.3, Math.max(0.25, Math.min((canvas_rect.width - 80) / (right - left || 1), (canvas_rect.height - 60) / (bottom - top || 1))));
  view_transform = {scale, x: (canvas_rect.width - (right - left) * scale) / 2 - left * scale, y: (canvas_rect.height - (bottom - top) * scale) / 2 - top * scale};
  apply_view_transform();
  save_to_local_storage();
}

/** Render, auto-layout, render again with the new positions, and fit the view. */
function arrange_current_tab(){
  render_all();
  auto_layout();
  render_all();
  fit_view();
}
