// Top-bar dropdown menus and the double-click parameter popup.

const popup_element = document.createElement('div');
popup_element.id = 'popup';
document.body.appendChild(popup_element);

/**
 * Dropdown item that adds a node on click or by drag-and-drop onto the canvas.
 * @param {string} label
 * @param {HTMLElement} icon_element
 * @param {Function} on_add       called on click
 * @param {object|null} drag_data  node description for drag-and-drop ({type, ...}); null disables dragging
 * @returns {HTMLElement}
 */
function make_palette_item(label, icon_element, on_add, drag_data){
  const item_element = document.createElement('div');
  item_element.className = 'pitem';
  item_element.draggable = !!drag_data;
  item_element.appendChild(icon_element);
  const name_element = document.createElement('span');
  name_element.className = 'nm';
  name_element.textContent = label;
  item_element.appendChild(name_element);
  item_element.addEventListener('click', () => on_add());
  item_element.addEventListener('dragstart', event => event.dataTransfer.setData('text/plain', JSON.stringify(drag_data)));
  return item_element;
}

/**
 * Small square icon in the colour of a node kind.
 * @param {string} kind_name
 * @returns {HTMLElement}
 */
function make_box_icon(kind_name){
  const icon_element = document.createElement('span');
  icon_element.className = 'ic ' + kind_name;
  return icon_element;
}

/**
 * Small circle icon with an operator symbol.
 * @param {string} symbol
 * @returns {HTMLElement}
 */
function make_operator_icon(symbol){
  const icon_element = document.createElement('span');
  icon_element.className = 'ic circ';
  icon_element.textContent = symbol;
  return icon_element;
}

/**
 * Add a node in the middle of the visible canvas and select it.
 * @param {string} node_type
 * @param {object} extra  see default_params
 */
function add_node_at_view_center(node_type, extra){
  const canvas_rect = canvas_element.getBoundingClientRect();
  const node = add_node(node_type,
    (canvas_rect.width / 2 - view_transform.x) / view_transform.scale - 60 + Math.random() * 30,
    (canvas_rect.height / 2 - view_transform.y) / view_transform.scale - 20 + Math.random() * 30, extra);
  selected_node_ids = new Set([node.id]);
  selected_edge_id = null;
  render_all();
}

/** Build the top-bar node menus (grouped like the torch.nn documentation). */
function render_menus(){
  const menu_bar = document.getElementById('menu_bar'), open_menu_key = menu_bar.querySelector('.menu.open')?.dataset.menu;
  menu_bar.innerHTML = '';
  const add_menu = (menu_key, label) => {
    const menu_element = document.createElement('div');
    menu_element.className = 'menu' + (menu_key === open_menu_key ? ' open' : '');
    menu_element.dataset.menu = menu_key;
    const menu_button = document.createElement('button');
    menu_button.className = 'mb';
    menu_button.textContent = label;
    const dropdown_element = document.createElement('div');
    dropdown_element.className = 'dd';
    menu_element.append(menu_button, dropdown_element);
    menu_bar.appendChild(menu_element);
    return dropdown_element;
  };
  const add_section = (dropdown_element, label) => {
    const section_element = document.createElement('div');
    section_element.className = 'sec';
    section_element.textContent = label;
    dropdown_element.appendChild(section_element);
  };
  ['Input', 'Output'].forEach(node_type => {
    const button = document.createElement('button');
    button.className = 'mb plain';
    button.textContent = node_type;
    button.draggable = true;
    button.onclick = () => add_node_at_view_center(node_type);
    button.addEventListener('dragstart', event => event.dataTransfer.setData('text/plain', JSON.stringify({type: node_type})));
    menu_bar.appendChild(button);
  });
  const layer_menu = add_menu('layer', 'Layers');
  [['Linear Layers', ['Linear']], ['Convolution Layers', ['Conv2d']], ['Pooling Layers', ['MaxPool2d', 'AdaptiveAvgPool2d']],
   ['Normalization Layers', ['BatchNorm2d', 'LayerNorm']], ['Dropout Layers', ['Dropout']], ['Vision Layers', ['Upsample']], ['Utilities', ['Flatten']]]
    .forEach(([section_label, layer_types]) => {
      add_section(layer_menu, section_label);
      layer_types.forEach(layer_type => layer_menu.appendChild(make_palette_item(layer_type, make_box_icon(LAYER_SPECS[layer_type].kind), () => add_node_at_view_center(layer_type), {type: layer_type})));
    });
  const activation_menu = add_menu('activation', 'Activation');
  const add_activation_item = function_name => activation_menu.appendChild(
    make_palette_item(function_name, make_box_icon('act'), () => add_node_at_view_center('Activation', {fn: function_name}), {type: 'Activation', fn: function_name}));
  add_section(activation_menu, 'Non-linear Activations (weighted sum, nonlinearity)');
  ACTIVATION_FUNCTIONS.filter(function_name => function_name !== 'Softmax').forEach(add_activation_item);
  activation_menu.appendChild(make_palette_item('Attention', make_box_icon('custom'), add_attention_block, null));
  add_section(activation_menu, 'Non-linear Activations (other)');
  add_activation_item('Softmax');
  const operator_menu = add_menu('operator', 'Operators');
  Object.entries(OPERATOR_SPECS).forEach(([operator_type, operator_spec]) =>
    operator_menu.appendChild(make_palette_item(operator_spec.label, make_operator_icon(operator_spec.symbol), () => add_node_at_view_center(operator_type), {type: operator_type})));
}

/**
 * Form fields for editing a set of parameters (popup).
 * @param {object} params  parameter name → value
 * @param {Function} on_change  (param_name, new_value) => void
 * @returns {DocumentFragment}
 */
function make_param_fields(params, on_change){
  const fields_fragment = document.createDocumentFragment();
  for(const [param_name, current_value] of Object.entries(params)){
    const editor = PARAM_EDITORS[param_name] || {type: 'text'};
    const field_element = document.createElement('div');
    field_element.className = 'field' + (editor.type === 'bool' ? ' chk' : '');
    const label_element = document.createElement('label');
    label_element.textContent = param_name;
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
    input_element.addEventListener('keydown', event => event.stopPropagation());
    input_element.addEventListener('change', () => {
      let new_value = editor.type === 'bool' ? input_element.checked : (editor.type === 'int' || editor.type === 'float') ? Number(input_element.value) : input_element.value;
      if(editor.type === 'int'){
        new_value = Math.round(new_value);
        if(!Number.isFinite(new_value)) new_value = current_value;
        if(editor.min != null) new_value = Math.max(editor.min, new_value);
      }
      if(editor.type === 'float'){
        if(!Number.isFinite(new_value)) new_value = current_value;
        new_value = Math.min(editor.max ?? Infinity, Math.max(editor.min ?? -Infinity, new_value));
      }
      if(editor.type !== 'bool' && editor.type !== 'select') input_element.value = new_value;
      on_change(param_name, new_value);
    });
    if(editor.type === 'bool'){ field_element.appendChild(input_element); field_element.appendChild(label_element); }
    else { field_element.appendChild(label_element); field_element.appendChild(input_element); }
    fields_fragment.appendChild(field_element);
  }
  return fields_fragment;
}

/** Hide the parameter popup. */
function close_popup(){
  popup_element.style.display = 'none';
}

/**
 * Show the popup next to the mouse.
 * @param {MouseEvent} mouse_event
 */
function show_popup_at(mouse_event){
  popup_element.style.display = 'block';
  popup_element.style.left = Math.min(mouse_event.clientX + 8, innerWidth - popup_element.offsetWidth - 8) + 'px';
  popup_element.style.top = Math.min(mouse_event.clientY + 8, innerHeight - popup_element.offsetHeight - 8) + 'px';
}

/**
 * Popup for an Input node: layout presets and dim names.
 * @param {object} input_node
 * @param {MouseEvent} mouse_event
 */
function open_input_layout_popup(input_node, mouse_event){
  popup_element.innerHTML = '';
  const heading = document.createElement('h4');
  heading.textContent = input_node.params.name;
  popup_element.appendChild(heading);
  const preset_row = document.createElement('div');
  preset_row.className = 'presets';
  const current_dim_names_text = () => input_dim_names(input_node).join(',');
  Object.values(INPUT_LAYOUT_PRESETS).forEach(preset => {
    const preset_button = document.createElement('button');
    preset_button.textContent = `${preset.label} (${['B', ...preset.dim_names].join('')})`;
    if(preset.dim_names.join(',') === current_dim_names_text()) preset_button.className = 'on';
    preset_button.onclick = () => {
      input_node.params.dims = preset.dim_names.join(',');
      input_node.params.shape = preset.shape.join(',');
      render_all();
      open_input_layout_popup(input_node, mouse_event);
    };
    preset_row.appendChild(preset_button);
  });
  popup_element.appendChild(preset_row);
  const field_element = document.createElement('div');
  field_element.className = 'field';
  const label_element = document.createElement('label');
  label_element.textContent = 'B,';
  const names_input = document.createElement('input');
  names_input.value = current_dim_names_text();
  names_input.addEventListener('keydown', event => { if(event.key === 'Enter') names_input.blur(); event.stopPropagation(); });
  names_input.addEventListener('change', () => {
    let new_dim_names = split_list_text(names_input.value).map(name => name.replace(/[^A-Za-z0-9_]/g, '')).filter(Boolean);
    if(new_dim_names[0] === 'B') new_dim_names = new_dim_names.slice(1);
    if(!new_dim_names.length){ names_input.value = current_dim_names_text(); return; }
    const old_sizes = split_list_text(input_node.params.shape), old_dim_names = input_dim_names(input_node);
    input_node.params.shape = new_dim_names.map(dim_name => {
      const old_index = old_dim_names.indexOf(dim_name);
      return old_index >= 0 && old_sizes[old_index] ? old_sizes[old_index] : (DEFAULT_DIM_SIZES[dim_name] || DEFAULT_HIDDEN_SIZE);
    }).join(',');
    input_node.params.dims = new_dim_names.join(',');
    render_all();
    open_input_layout_popup(input_node, mouse_event);
  });
  field_element.append(label_element, names_input);
  popup_element.appendChild(field_element);
  show_popup_at(mouse_event);
}

/**
 * Popup for a node's remaining parameters (those not editable on the node itself).
 * @param {object} node
 * @param {MouseEvent} mouse_event
 */
function open_param_popup(node, mouse_event){
  const inline_params = ['name', 'fn', 'block', 'shape', ...(INLINE_OUTPUT_PARAM[node.type] ? [INLINE_OUTPUT_PARAM[node.type].param_name] : [])];
  const popup_params = Object.fromEntries(Object.entries(node.params).filter(([param_name]) => !inline_params.includes(param_name)));
  if(!Object.keys(popup_params).length) return;
  popup_element.innerHTML = '';
  const heading = document.createElement('h4');
  heading.textContent = node_title(node);
  popup_element.appendChild(heading);
  popup_element.appendChild(make_param_fields(popup_params, (param_name, new_value) => { node.params[param_name] = new_value; render_all(); }));
  show_popup_at(mouse_event);
}

document.addEventListener('click', event => {
  const menu_button = event.target.closest('.mb:not(.plain)');
  if(menu_button){
    const menu_element = menu_button.parentElement, was_open = menu_element.classList.contains('open');
    document.querySelectorAll('.menu.open').forEach(open_menu => open_menu.classList.remove('open'));
    if(!was_open) menu_element.classList.add('open');
    return;
  }
  document.querySelectorAll('.menu.open').forEach(open_menu => open_menu.classList.remove('open'));
});
document.addEventListener('dragend', () => document.querySelectorAll('.menu.open').forEach(open_menu => open_menu.classList.remove('open')));
window.addEventListener('mousedown', event => { if(!event.target.closest('#popup')) close_popup(); }, true);
