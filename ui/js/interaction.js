// Mouse and keyboard: selecting, dragging nodes, connecting ports, panning, zooming, shortcuts.

let drag_state = null;

/**
 * Screen position → world position.
 * @param {number} client_x
 * @param {number} client_y
 * @returns {{x: number, y: number}}
 */
function screen_to_world(client_x, client_y){
  const canvas_rect = canvas_element.getBoundingClientRect();
  return {x: (client_x - canvas_rect.left - view_transform.x) / view_transform.scale, y: (client_y - canvas_rect.top - view_transform.y) / view_transform.scale};
}

/**
 * Toggle a node in the selection.
 * @param {number} node_id
 */
function toggle_node_selection(node_id){
  if(selected_node_ids.has(node_id)) selected_node_ids.delete(node_id); else selected_node_ids.add(node_id);
  selected_edge_id = null;
  refresh_view();
}

/** Delete the selected nodes and edge. */
function delete_selection(){
  if(!selected_node_ids.size && !selected_edge_id) return;
  const graph = current_graph();
  if(selected_edge_id) graph.edges = graph.edges.filter(edge => edge.id !== selected_edge_id);
  selected_node_ids.forEach(node_id => remove_node(node_id));
  clear_selection();
  render_all();
}

canvas_element.addEventListener('mousedown', event => {
  if(event.button !== 0) return;
  if(event.target.closest('input,select,button')){
    const clicked_node_element = event.target.closest('.node');
    if(event.shiftKey && clicked_node_element){ event.preventDefault(); toggle_node_selection(+clicked_node_element.dataset.id); }
    return;
  }
  const graph = current_graph(), port_element = event.target.closest('.port'), clicked_node_element = event.target.closest('.node');
  const clicked_edge_id = event.target.dataset && event.target.dataset.edge;
  if(port_element){
    event.preventDefault();
    const node_id = +clicked_node_element.dataset.id;
    if(port_element.classList.contains('out')){
      drag_state = {kind: 'wire', source_id: node_id, start_point: port_position(find_current_node(node_id), 'out'), end_point: screen_to_world(event.clientX, event.clientY)};
    } else {
      const port_index = +port_element.dataset.port, existing_edge = graph.edges.find(edge => edge.to === node_id && edge.port === port_index);
      if(existing_edge){
        graph.edges = graph.edges.filter(edge => edge !== existing_edge);
        drag_state = {kind: 'wire', source_id: existing_edge.from, start_point: port_position(find_current_node(existing_edge.from), 'out'), end_point: screen_to_world(event.clientX, event.clientY)};
        render_all();
      }
    }
    return;
  }
  if(clicked_node_element){
    event.preventDefault();
    const node_id = +clicked_node_element.dataset.id, pointer_world = screen_to_world(event.clientX, event.clientY);
    if(event.shiftKey){ toggle_node_selection(node_id); return; }
    if(!selected_node_ids.has(node_id)) selected_node_ids = new Set([node_id]);
    selected_edge_id = null;
    refresh_view();
    drag_state = {kind: 'node', start_world: pointer_world,
      original_positions: [...selected_node_ids].map(find_current_node).filter(Boolean).map(node => ({node, x: node.x, y: node.y}))};
    return;
  }
  if(clicked_edge_id){ clear_selection(); selected_edge_id = clicked_edge_id; refresh_view(); return; }
  if(document.activeElement && document.activeElement.blur) document.activeElement.blur();
  if(event.shiftKey){ drag_state = {kind: 'box', start_client_x: event.clientX, start_client_y: event.clientY}; return; }
  drag_state = {kind: 'pan', start_client_x: event.clientX, start_client_y: event.clientY, start_view_x: view_transform.x, start_view_y: view_transform.y, has_moved: false};
});

window.addEventListener('mousemove', event => {
  if(!drag_state) return;
  if(drag_state.kind === 'pan'){
    view_transform.x = drag_state.start_view_x + event.clientX - drag_state.start_client_x;
    view_transform.y = drag_state.start_view_y + event.clientY - drag_state.start_client_y;
    if(Math.abs(event.clientX - drag_state.start_client_x) + Math.abs(event.clientY - drag_state.start_client_y) > 3) drag_state.has_moved = true;
    apply_view_transform();
  } else if(drag_state.kind === 'node'){
    const pointer_world = screen_to_world(event.clientX, event.clientY);
    const offset_x = pointer_world.x - drag_state.start_world.x, offset_y = pointer_world.y - drag_state.start_world.y;
    drag_state.original_positions.forEach(original => {
      original.node.x = Math.round((original.x + offset_x) / 5) * 5;
      original.node.y = Math.round((original.y + offset_y) / 5) * 5;
      const element = node_element(original.node.id);
      if(element){ element.style.left = original.node.x + 'px'; element.style.top = original.node.y + 'px'; }
    });
    render_wires();
  } else if(drag_state.kind === 'box'){
    const canvas_rect = canvas_element.getBoundingClientRect(), selection_box = document.getElementById('selection_box');
    Object.assign(selection_box.style, {
      display: 'block',
      left: Math.min(event.clientX, drag_state.start_client_x) - canvas_rect.left + 'px',
      top: Math.min(event.clientY, drag_state.start_client_y) - canvas_rect.top + 'px',
      width: Math.abs(event.clientX - drag_state.start_client_x) + 'px',
      height: Math.abs(event.clientY - drag_state.start_client_y) + 'px',
    });
  } else if(drag_state.kind === 'wire'){
    drag_state.end_point = screen_to_world(event.clientX, event.clientY);
    render_wires();
  }
});

window.addEventListener('mouseup', event => {
  if(!drag_state) return;
  const finished_drag = drag_state;
  drag_state = null;
  if(finished_drag.kind === 'wire'){
    const element_under_pointer = document.elementFromPoint(event.clientX, event.clientY);
    const target_port = element_under_pointer && element_under_pointer.closest('.port.in'), target_node_element = element_under_pointer && element_under_pointer.closest('.node');
    if(target_port) connect_nodes(finished_drag.source_id, +target_node_element.dataset.id, +target_port.dataset.port);
    else if(target_node_element) connect_nodes(finished_drag.source_id, +target_node_element.dataset.id, null);
    render_all();
  } else if(finished_drag.kind === 'node') save_to_local_storage();
  else if(finished_drag.kind === 'pan'){
    if(!finished_drag.has_moved && (selected_node_ids.size || selected_edge_id)){ clear_selection(); refresh_view(); }
    save_to_local_storage();
  } else if(finished_drag.kind === 'box'){
    document.getElementById('selection_box').style.display = 'none';
    const top_left = screen_to_world(Math.min(event.clientX, finished_drag.start_client_x), Math.min(event.clientY, finished_drag.start_client_y));
    const bottom_right = screen_to_world(Math.max(event.clientX, finished_drag.start_client_x), Math.max(event.clientY, finished_drag.start_client_y));
    current_graph().nodes.forEach(node => {
      const size = node_size(node);
      if(node.x < bottom_right.x && node.x + size.w > top_left.x && node.y < bottom_right.y && node.y + size.h > top_left.y) selected_node_ids.add(node.id);
    });
    selected_edge_id = null;
    refresh_view();
  }
});

canvas_element.addEventListener('wheel', event => {
  event.preventDefault();
  const canvas_rect = canvas_element.getBoundingClientRect(), pointer_x = event.clientX - canvas_rect.left, pointer_y = event.clientY - canvas_rect.top;
  const new_scale = Math.min(2.5, Math.max(0.25, view_transform.scale * Math.exp(-event.deltaY * 0.0015)));
  view_transform.x = pointer_x - (pointer_x - view_transform.x) * new_scale / view_transform.scale;
  view_transform.y = pointer_y - (pointer_y - view_transform.y) * new_scale / view_transform.scale;
  view_transform.scale = new_scale;
  apply_view_transform();
  save_to_local_storage();
}, {passive: false});

canvas_element.addEventListener('dragover', event => event.preventDefault());
canvas_element.addEventListener('drop', event => {
  event.preventDefault();
  let dropped_item;
  try{ dropped_item = JSON.parse(event.dataTransfer.getData('text/plain')); } catch{ return; }
  if(dropped_item.type === 'Custom' && !can_place_block(dropped_item.block)) return;
  const pointer_world = screen_to_world(event.clientX, event.clientY);
  const node = add_node(dropped_item.type, pointer_world.x - (dropped_item.type in OPERATOR_SPECS ? OPERATOR_RADIUS : 65), pointer_world.y - 20, dropped_item);
  selected_node_ids = new Set([node.id]);
  selected_edge_id = null;
  render_all();
});

canvas_element.addEventListener('dblclick', event => {
  const clicked_node_element = event.target.closest('.node');
  if(!clicked_node_element || event.target.closest('input,select')) return;
  const node = find_current_node(+clicked_node_element.dataset.id);
  if(node.type === 'Custom') return switch_tab(node.params.block);
  if(node.type === 'Input') return open_input_layout_popup(node, event);
  open_param_popup(node, event);
});

window.addEventListener('keydown', event => {
  if(event.target.closest && event.target.closest('input,select,textarea')) return;
  const with_command_key = event.ctrlKey || event.metaKey, key = event.key.toLowerCase();
  if(event.key === 'Delete' || event.key === 'Backspace'){ event.preventDefault(); delete_selection(); }
  if(event.key === 'Escape'){ clear_selection(); close_popup(); refresh_view(); }
  if(with_command_key && key === 'g'){ event.preventDefault(); group_selection_into_block(); }
  if(with_command_key && key === 'a'){ event.preventDefault(); selected_node_ids = new Set(current_graph().nodes.map(node => node.id)); selected_edge_id = null; refresh_view(); }
  if(with_command_key && key === 'c' && selected_node_ids.size){ event.preventDefault(); copy_selection(); }
  if(with_command_key && key === 'v' && clipboard){ event.preventDefault(); paste_clipboard(); }
});
