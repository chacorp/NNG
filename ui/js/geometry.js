// Node sizes, port positions and arrow paths (world coordinates, pixels).
// Point = {x, y, direction?}; direction 'right' / 'left' marks an operator side input entered moving right / left.

const OPERATOR_RADIUS = 14;
let edge_lane_x = new Map();

/**
 * Size of a node box as last measured in the DOM.
 * @param {object} node
 * @returns {{w: number, h: number}} width and height
 */
function node_size(node){
  return measured_sizes.get(node.id) || (node.type in OPERATOR_SPECS ? {w: 2 * OPERATOR_RADIUS, h: 2 * OPERATOR_RADIUS} : {w: 140, h: 44});
}

/**
 * Position of a node's port.
 * @param {object} node
 * @param {string} direction   'in' | 'out'
 * @param {number} port_index  input port index
 * @returns {{x: number, y: number, direction?: string}}
 */
function port_position(node, direction, port_index = 0){
  const {w: width, h: height} = node_size(node);
  if(direction === 'out') return {x: node.x + width / 2, y: node.y + height};
  if(node.type in OPERATOR_SPECS){
    const center_x = node.x + OPERATOR_RADIUS, center_y = node.y + OPERATOR_RADIUS;
    if(port_index === 0) return {x: center_x, y: node.y};
    if(port_index === 1 || port_index === 2)
      return operator_side_of_port(current_graph(), node, port_index) === 'left'
        ? {x: node.x, y: center_y, direction: 'right'}
        : {x: node.x + 2 * OPERATOR_RADIUS, y: center_y, direction: 'left'};
    const extra_slot = port_index - 2, offset_x = (extra_slot % 2 ? -1 : 1) * Math.ceil(extra_slot / 2) * 8;
    return {x: center_x + offset_x, y: center_y - Math.sqrt(OPERATOR_RADIUS * OPERATOR_RADIUS - offset_x * offset_x)};
  }
  return {x: box_input_port_x(current_graph(), node, port_index), y: node.y};
}

/**
 * Center x of each input source, keyed by port index.
 * @param {object} graph
 * @param {object} node
 * @returns {Object<number, number>}
 */
function source_center_x_by_port(graph, node){
  const center_x_by_port = {};
  incoming_edges(graph, node).forEach(edge => {
    const source_node = find_node(graph, edge.from);
    if(source_node) center_x_by_port[edge.port] = source_node.x + node_size(source_node).w / 2;
  });
  return center_x_by_port;
}

/**
 * x of a box's input port: an input from straight above takes the center, others the half facing their source.
 * @param {object} graph
 * @param {object} node
 * @param {number} port_index
 * @returns {number}
 */
function box_input_port_x(graph, node, port_index){
  const width = node_size(node).w, center_x = node.x + width / 2, port_count = input_port_count(graph, node);
  if(port_count < 2) return center_x;
  const source_x = source_center_x_by_port(graph, node);
  const port_indices = [...Array(port_count).keys()];
  const center_port = port_indices.find(port => source_x[port] != null && Math.abs(source_x[port] - center_x) < 1.5);
  if(center_port == null) return node.x + width * (port_slot_by_source(graph, node, port_index) + 1) / (port_count + 1);
  if(port_index === center_port) return center_x;
  const other_ports = port_indices.filter(port => port !== center_port), sort_key = port => source_x[port] ?? Infinity;
  const left_ports = other_ports.filter(port => source_x[port] != null && source_x[port] < center_x).sort((a, b) => sort_key(a) - sort_key(b));
  const right_ports = other_ports.filter(port => !left_ports.includes(port)).sort((a, b) => sort_key(a) - sort_key(b) || a - b);
  if(left_ports.includes(port_index)) return node.x + (width / 2) * (left_ports.indexOf(port_index) + 1) / (left_ports.length + 1);
  return center_x + (width / 2) * (right_ports.indexOf(port_index) + 1) / (right_ports.length + 1);
}

/**
 * Side of an operator circle where side input 1 or 2 is drawn (the side its arrow comes from).
 * @param {object} graph
 * @param {object} operator_node
 * @param {number} port_index  1 or 2
 * @returns {string} 'left' | 'right'
 */
function operator_side_of_port(graph, operator_node, port_index){
  const source_x = {};
  incoming_edges(graph, operator_node).forEach(edge => {
    if(edge.port !== 1 && edge.port !== 2) return;
    const source_node = find_node(graph, edge.from);
    if(source_node) source_x[edge.port] = source_node.x + node_size(source_node).w / 2;
  });
  const center_x = operator_node.x + OPERATOR_RADIUS;
  let swap_sides = false;
  if(source_x[1] != null && source_x[2] != null) swap_sides = source_x[1] > source_x[2];
  else if(source_x[1] != null) swap_sides = source_x[1] > center_x;
  else if(source_x[2] != null) swap_sides = source_x[2] < center_x;
  return (port_index === 1) !== swap_sides ? 'left' : 'right';
}

/**
 * Left-to-right slot of a box input port, ordered by where each input comes from.
 * @param {object} graph
 * @param {object} node
 * @param {number} port_index
 * @returns {number} slot index
 */
function port_slot_by_source(graph, node, port_index){
  const port_count = input_port_count(graph, node);
  if(port_count < 2) return port_index;
  const source_x = source_center_x_by_port(graph, node);
  const slot_order = [...Array(port_count).keys()].sort((a, b) => ((source_x[a] ?? Infinity) - (source_x[b] ?? Infinity)) || a - b);
  return slot_order.indexOf(port_index);
}

/**
 * Orthogonal arrow path between two points.
 * @param {{x: number, y: number}} start_point
 * @param {{x: number, y: number, direction?: string}} end_point
 * @returns {string} SVG path data
 */
function elbow_path(start_point, end_point){
  if(end_point.direction){
    const side_sign = end_point.direction === 'right' ? -1 : 1;
    if((end_point.x - start_point.x) * -side_sign >= 12 && start_point.y < end_point.y)
      return `M${start_point.x},${start_point.y} V${end_point.y} H${end_point.x}`;
    const approach_x = end_point.x + side_sign * 14;
    return `M${start_point.x},${start_point.y} V${start_point.y + 8} H${approach_x} V${end_point.y} H${end_point.x}`;
  }
  if(Math.abs(start_point.x - end_point.x) < 1 && end_point.y > start_point.y) return `M${start_point.x},${start_point.y} V${end_point.y}`;
  if(end_point.y - start_point.y >= 18){
    const bend_y = end_point.y - 10;
    return `M${start_point.x},${start_point.y} V${bend_y} H${end_point.x} V${end_point.y}`;
  }
  const detour_x = Math.max(start_point.x, end_point.x) + 60;
  return `M${start_point.x},${start_point.y} V${start_point.y + 8} H${detour_x} V${end_point.y - 10} H${end_point.x} V${end_point.y}`;
}

/**
 * Pick a side lane for every downward edge whose direct path would cross a box.
 * Fills edge_lane_x: edge id → lane x.
 * @param {object} graph
 */
function compute_edge_lanes(graph){
  edge_lane_x = new Map();
  const node_boxes = graph.nodes.map(node => ({id: node.id, x: node.x, y: node.y, w: node_size(node).w, h: node_size(node).h}));
  const lane_requests = [];
  for(const edge of graph.edges){
    const source_node = find_node(graph, edge.from), target_node = find_node(graph, edge.to);
    if(!source_node || !target_node) continue;
    if(!(target_node.y >= source_node.y + node_size(source_node).h + 6)) continue;
    const obstacle_boxes = node_boxes.filter(box => box.id !== source_node.id && box.id !== target_node.id)
      .map(box => ({x: box.x + 1, y: box.y + 1, w: box.w - 2, h: box.h - 2}));
    const direct_points = path_points(direct_edge_path(graph, edge));
    let is_blocked = false;
    for(let point_index = 1; point_index < direct_points.length && !is_blocked; point_index++)
      is_blocked = obstacle_boxes.some(box => segment_hits_rect(direct_points[point_index - 1], direct_points[point_index], box));
    if(!is_blocked) continue;
    const start_point = port_position(source_node, 'out'), end_point = port_position(target_node, 'in', edge.port);
    const side_sign = end_point.direction ? (end_point.direction === 'left' ? 1 : -1) : (start_point.x > end_point.x ? 1 : -1);
    const min_x = Math.min(start_point.x, end_point.x), max_x = Math.max(start_point.x, end_point.x);
    const nearby_boxes = obstacle_boxes.filter(box => box.y < end_point.y && box.y + box.h > start_point.y && box.x < max_x + 40 && box.x + box.w > min_x - 40);
    const outer_x = side_sign < 0 ? Math.min(min_x, ...nearby_boxes.map(box => box.x)) : Math.max(max_x, ...nearby_boxes.map(box => box.x + box.w));
    lane_requests.push({edge, side_sign, top_y: start_point.y, bottom_y: end_point.y, outer_x});
  }
  lane_requests.sort((first, second) => (first.bottom_y - first.top_y) - (second.bottom_y - second.top_y));
  const placed_lanes = [];
  for(const request of lane_requests){
    let lane_x = request.outer_x + request.side_sign * 14;
    for(const placed of placed_lanes)
      if(placed.side_sign === request.side_sign && placed.top_y < request.bottom_y && placed.bottom_y > request.top_y)
        lane_x = request.side_sign < 0 ? Math.min(lane_x, placed.lane_x - 10) : Math.max(lane_x, placed.lane_x + 10);
    request.lane_x = lane_x;
    placed_lanes.push(request);
    edge_lane_x.set(request.edge.id, lane_x);
  }
}

/**
 * Arrow path of an edge, through its side lane when it has one.
 * @param {object} graph
 * @param {object} edge
 * @returns {string} SVG path data
 */
function edge_path(graph, edge){
  if(edge_lane_x.has(edge.id)){
    const source_node = find_node(graph, edge.from), target_node = find_node(graph, edge.to);
    const start_point = port_position(source_node, 'out'), end_point = port_position(target_node, 'in', edge.port), lane_x = edge_lane_x.get(edge.id);
    if(target_node.y >= source_node.y + node_size(source_node).h + 6)
      return end_point.direction
        ? `M${start_point.x},${start_point.y} V${start_point.y + 8} H${lane_x} V${end_point.y} H${end_point.x}`
        : `M${start_point.x},${start_point.y} V${start_point.y + 8} H${lane_x} V${end_point.y - 10} H${end_point.x} V${end_point.y}`;
  }
  return direct_edge_path(graph, edge);
}

/**
 * Arrow path of an edge from the node geometry (target below, above, or beside the source).
 * @param {object} graph
 * @param {object} edge
 * @returns {string} SVG path data
 */
function direct_edge_path(graph, edge){
  const source_node = find_node(graph, edge.from), target_node = find_node(graph, edge.to);
  const source_size = node_size(source_node), target_size = node_size(target_node), vertical_gap = 6;
  let start_point = port_position(source_node, 'out'), end_point = port_position(target_node, 'in', edge.port);
  const overlaps_horizontally = !(target_node.x >= source_node.x + source_size.w || target_node.x + target_size.w <= source_node.x);
  const target_is_right = target_node.x > source_node.x;
  if(target_node.y >= source_node.y + source_size.h + vertical_gap){
    if(!overlaps_horizontally && !(target_node.type in OPERATOR_SPECS) && input_port_count(graph, target_node) === 1)
      return `M${start_point.x},${start_point.y} V${target_node.y + target_size.h / 2} H${target_is_right ? target_node.x : target_node.x + target_size.w}`;
    return elbow_path(start_point, end_point);
  }
  if(target_node.y + target_size.h <= source_node.y - vertical_gap){
    if(!overlaps_horizontally && !end_point.direction){
      const end_x = target_node.type in OPERATOR_SPECS ? target_node.x + OPERATOR_RADIUS : end_point.x;
      return `M${target_is_right ? source_node.x + source_size.w : source_node.x},${source_node.y + source_size.h / 2} H${end_x} V${target_node.y + target_size.h}`;
    }
    start_point = {x: source_node.x + source_size.w / 2, y: source_node.y};
    if(end_point.direction) return `M${start_point.x},${start_point.y} V${end_point.y} H${end_point.x}`;
    end_point = {x: target_node.type in OPERATOR_SPECS ? target_node.x + OPERATOR_RADIUS : end_point.x, y: target_node.y + target_size.h};
    if(Math.abs(start_point.x - end_point.x) < 1) return `M${start_point.x},${start_point.y} V${end_point.y}`;
    const bend_y = end_point.y + 10;
    return `M${start_point.x},${start_point.y} V${bend_y} H${end_point.x} V${end_point.y}`;
  }
  const target_on_right = target_node.x + target_size.w / 2 > source_node.x + source_size.w / 2;
  start_point = {x: target_on_right ? source_node.x + source_size.w : source_node.x, y: source_node.y + source_size.h / 2};
  if(!(end_point.direction && (end_point.direction === 'right') === target_on_right))
    end_point = {x: target_on_right ? target_node.x : target_node.x + target_size.w, y: target_node.y + target_size.h / 2};
  if(Math.abs(start_point.y - end_point.y) < 1) return `M${start_point.x},${start_point.y} H${end_point.x}`;
  const bend_x = end_point.x + (target_on_right ? -12 : 12);
  return `M${start_point.x},${start_point.y} H${bend_x} V${end_point.y} H${end_point.x}`;
}

/**
 * Corner points of an SVG path made of M / L / H / V commands.
 * @param {string} path_data
 * @returns {{x: number, y: number}[]}
 */
function path_points(path_data){
  const points = [];
  let x = 0, y = 0;
  for(const match of path_data.matchAll(/([MLHV])\s*(-?[\d.]+)(?:[ ,](-?[\d.]+))?/g)){
    if(match[1] === 'H') x = +match[2];
    else if(match[1] === 'V') y = +match[2];
    else { x = +match[2]; y = +match[3]; }
    points.push({x, y});
  }
  return points;
}

/**
 * Whether an axis-aligned segment touches a rectangle.
 * @param {{x: number, y: number}} start_point
 * @param {{x: number, y: number}} end_point
 * @param {{x: number, y: number, w: number, h: number}} rect
 * @returns {boolean}
 */
function segment_hits_rect(start_point, end_point, rect){
  const min_x = Math.min(start_point.x, end_point.x), max_x = Math.max(start_point.x, end_point.x);
  const min_y = Math.min(start_point.y, end_point.y), max_y = Math.max(start_point.y, end_point.y);
  return max_x >= rect.x && min_x <= rect.x + rect.w && max_y >= rect.y && min_y <= rect.y + rect.h;
}

/**
 * Whether two rectangles overlap.
 * @param {{x: number, y: number, w: number, h: number}} first_rect
 * @param {{x: number, y: number, w: number, h: number}} second_rect
 * @returns {boolean}
 */
function rects_overlap(first_rect, second_rect){
  return first_rect.x < second_rect.x + second_rect.w && first_rect.x + first_rect.w > second_rect.x
    && first_rect.y < second_rect.y + second_rect.h && first_rect.y + first_rect.h > second_rect.y;
}
