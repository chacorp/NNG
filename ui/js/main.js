// Saving / restoring the session and wiring the top-bar buttons.

const STORAGE_KEY = 'nng-graph-v4';
const LEGACY_STORAGE_KEYS = ['nn-graph-builder-v3', 'nn-graph-builder-v2'];

/** Store the graph, open tabs and views in localStorage. */
function save_to_local_storage(){
  try{
    tab_views[current_tab] = {...view_transform};
    localStorage.setItem(STORAGE_KEY, JSON.stringify({state, current_tab, open_tabs, tab_views}));
  } catch{}
}

/**
 * Convert saved data from older versions to the current format.
 * @param {object} saved_state  current {main, blocks, next_id}, or older formats
 * @returns {object} {main, blocks, next_id}
 */
function migrate_saved_state(saved_state){
  let migrated_state;
  if(saved_state.main) migrated_state = {main: saved_state.main, blocks: saved_state.blocks || {}, next_id: saved_state.next_id ?? saved_state.nextId ?? 1};
  else {
    migrated_state = {main: {nodes: saved_state.nodes || [], edges: saved_state.edges || []}, blocks: {}, next_id: saved_state.nextId || 1};
    for(const [block_name, layer_steps] of Object.entries(saved_state.blocks || {})){
      const block_graph = {nodes: [], edges: []};
      let previous_node = null, next_y = 0;
      const append_node = (node_type, params) => {
        const node = {id: migrated_state.next_id++, type: node_type, x: 0, y: next_y, params};
        next_y += 80;
        block_graph.nodes.push(node);
        if(previous_node) block_graph.edges.push({id: 'e' + (migrated_state.next_id++), from: previous_node.id, to: node.id, port: 0});
        previous_node = node;
      };
      append_node('Input', {name: 'x', shape: String(DEFAULT_HIDDEN_SIZE)});
      layer_steps.forEach(step => append_node(step.type, deep_copy(step.params)));
      append_node('Output', {name: 'y'});
      migrated_state.blocks[block_name] = block_graph;
    }
  }
  [migrated_state.main, ...Object.values(migrated_state.blocks)].forEach(graph => {
    if('needsLayout' in graph){ graph.needs_layout = graph.needsLayout; delete graph.needsLayout; }
    graph.nodes.forEach(node => {
      if('gc' in node){ node.grid_column = node.gc; node.grid_row = node.gr; delete node.gc; delete node.gr; }
      if(node.type === 'Input' && !node.params.dims) node.params.dims = default_dim_names(split_list_text(node.params.shape).length).join(',');
    });
  });
  return migrated_state;
}

/**
 * Replace the whole graph state.
 * @param {object} saved_state  see migrate_saved_state
 */
function load_state(saved_state){
  state = migrate_saved_state(saved_state || {});
  current_tab = 'main';
  open_tabs = [];
  tab_views = {};
  clear_selection();
  measured_sizes.clear();
}

/**
 * Load a graph from JSON text and show it.
 * @param {string} json_text
 */
function open_graph_json(json_text){
  try{ load_state(JSON.parse(json_text)); render_all(); fit_view(); }
  catch(error){ alert('Could not read the JSON: ' + error); }
}

document.querySelectorAll('[data-example]').forEach(example_button => example_button.onclick = () => {
  if(state.main.nodes.length && !confirm('Replace the current graph with the example?')) return;
  load_example(example_button.dataset.example);
});
document.getElementById('clear_button').onclick = () => {
  if(!confirm('Clear the graph and all custom blocks?')) return;
  load_state({});
  render_all();
  fit_view();
};
document.getElementById('save_json_button').onclick = () => save_file('nn_graph.json', JSON.stringify(state, null, 2), 'application/json');
document.getElementById('open_json_button').onclick = async () => {
  const app_bridge = get_app_bridge();
  if(!app_bridge) return document.getElementById('file_input').click();
  const opened_file = await app_bridge.open_graph_file();
  if(opened_file) open_graph_json(opened_file.text);
};
document.getElementById('file_input').onchange = async event => {
  const chosen_file = event.target.files[0];
  if(!chosen_file) return;
  open_graph_json(await chosen_file.text());
  event.target.value = '';
};
document.getElementById('auto_layout_button').onclick = arrange_current_tab;
document.getElementById('fit_button').onclick = fit_view;
document.getElementById('save_svg_button').onclick = save_figure_svg;
document.getElementById('save_png_button').onclick = save_figure_png;
document.getElementById('torch_check_button').onclick = run_torch_check;
document.getElementById('copy_button').onclick = async () => {
  const copy_button = document.getElementById('copy_button');
  try{ await navigator.clipboard.writeText(current_model_code); copy_button.textContent = 'Copied'; }
  catch{ copy_button.textContent = 'Copy failed'; }
  setTimeout(() => copy_button.textContent = 'Copy', 1200);
};
document.getElementById('save_python_button').onclick = () => current_model_code && save_file('model.py', current_model_code, 'text/x-python');
window.addEventListener('nngbridgeready', () => { document.getElementById('torch_check_button').style.display = ''; });

/** Restore the last session, or start with the Custom MLP example. */
function initialize_app(){
  let saved_session = null;
  try{ saved_session = JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch{}
  if(!saved_session){
    for(const legacy_key of LEGACY_STORAGE_KEYS){
      try{
        const legacy_session = JSON.parse(localStorage.getItem(legacy_key));
        if(legacy_session && legacy_session.S){ saved_session = {state: legacy_session.S, open_tabs: legacy_session.openTabs, tab_views: legacy_session.views, current_tab: legacy_session.cur}; break; }
      } catch{}
    }
  }
  if(saved_session && saved_session.state){
    load_state(saved_session.state);
    open_tabs = (saved_session.open_tabs || []).filter(block_name => state.blocks[block_name]);
    tab_views = Object.fromEntries(Object.entries(saved_session.tab_views || {}).map(([tab_key, view]) => [tab_key, {x: view.x, y: view.y, scale: view.scale ?? view.k ?? 1}]));
    current_tab = saved_session.current_tab && (saved_session.current_tab === 'main' || state.blocks[saved_session.current_tab]) ? saved_session.current_tab : 'main';
    render_all();
    if(tab_views[current_tab]){ view_transform = {...tab_views[current_tab]}; apply_view_transform(); }
    else fit_view();
  } else {
    build_example_mlp();
    arrange_current_tab();
  }
}

initialize_app();
