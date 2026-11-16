// Saving files, the desktop-app bridge, the torch check button and figure export (SVG / PNG).

/**
 * Bridge to the NNG desktop app, or null when the page runs in a plain browser.
 * @returns {object|null} window.nng_bridge
 */
function get_app_bridge(){
  return window.nng_bridge || null;
}

/**
 * Save a file: native save dialog in the desktop app, browser download otherwise.
 * @param {string} file_name   suggested name, e.g. "model.py"
 * @param {string|Blob} file_data  text contents, or a PNG Blob
 * @param {string} mime_type   MIME type used for the browser download
 */
async function save_file(file_name, file_data, mime_type){
  const app_bridge = get_app_bridge();
  if(app_bridge){
    if(file_data instanceof Blob){
      const data_url = await new Promise(resolve => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.readAsDataURL(file_data); });
      await app_bridge.save_png_file(file_name, String(data_url).split(',')[1]);
    } else {
      const file_kind = file_name.endsWith('.py') ? 'python' : file_name.endsWith('.svg') ? 'svg' : 'json';
      await app_bridge.save_text_file(file_name, file_data, file_kind);
    }
    return;
  }
  const link_element = document.createElement('a');
  link_element.href = URL.createObjectURL(file_data instanceof Blob ? file_data : new Blob([file_data], {type: mime_type}));
  link_element.download = file_name;
  link_element.click();
  setTimeout(() => URL.revokeObjectURL(link_element.href), 1000);
}

/** Run the exported model.py in the desktop app and show the forward / gradient check result. */
async function run_torch_check(){
  const app_bridge = get_app_bridge(), result_box = document.getElementById('torch_result'), check_button = document.getElementById('torch_check_button');
  if(!app_bridge || !current_model_code) return;
  check_button.disabled = true;
  check_button.textContent = '검증 중…';
  result_box.style.display = '';
  result_box.className = 'check-run';
  result_box.textContent = '';
  try{
    const check_result = await app_bridge.run_torch_check(current_model_code);
    result_box.className = check_result.passed ? 'check-pass' : 'check-fail';
    result_box.textContent = check_result.output.split('\n').filter(line => /^ {2}\S+: \(|^OK|Error|assert/.test(line)).join('\n') || check_result.output;
  } catch(error){
    result_box.className = 'check-fail';
    result_box.textContent = String(error);
  }
  check_button.disabled = false;
  check_button.textContent = 'torch 검증';
}

/**
 * Escape text for SVG.
 * @param {string} text
 * @returns {string}
 */
function escape_svg_text(text){
  return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/**
 * SVG figure of the current tab (white background, same layout as the canvas).
 * @returns {{svg: string, w: number, h: number}|null} SVG text and its size in pixels; null for an empty graph
 */
function build_figure_svg(){
  const graph = current_graph();
  if(!graph.nodes.length) return null;
  compute_edge_lanes(graph);
  const root_style = getComputedStyle(document.documentElement), css_color = variable_name => root_style.getPropertyValue(variable_name).trim(), margin = 24;
  const left = Math.min(...graph.nodes.map(node => node.x)) - margin, top = Math.min(...graph.nodes.map(node => node.y)) - margin;
  const right = Math.max(...graph.nodes.map(node => node.x + node_size(node).w + (node.type === 'Concat' ? 60 : 0))) + margin;
  const bottom = Math.max(...graph.nodes.map(node => node.y + node_size(node).h)) + margin;
  const font_attribute = 'font-family="Helvetica Neue, Helvetica, Arial, sans-serif"';
  let svg_text = `<svg xmlns="http://www.w3.org/2000/svg" width="${right - left}" height="${bottom - top}" viewBox="${left} ${top} ${right - left} ${bottom - top}">`;
  svg_text += `<defs><marker id="a" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.8 L10,5 L0,9.2 z" fill="${css_color('--wire')}"/></marker></defs>`;
  svg_text += `<rect x="${left}" y="${top}" width="${right - left}" height="${bottom - top}" fill="#ffffff"/>`;
  for(const edge of graph.edges)
    svg_text += `<path d="${edge_path(graph, edge)}" fill="none" stroke="${css_color('--wire')}" stroke-width="1.4" marker-end="url(#a)"/>`;
  for(const node of graph.nodes){
    const {w: width, h: height} = node_size(node);
    if(node.type in OPERATOR_SPECS){
      const center_x = node.x + OPERATOR_RADIUS, center_y = node.y + OPERATOR_RADIUS;
      const glyph_style = LETTER_OPERATORS.includes(node.type) ? ' font-style="italic" font-family="Georgia, Times New Roman, serif"' : ` ${font_attribute}`;
      svg_text += `<circle cx="${center_x}" cy="${center_y}" r="${OPERATOR_RADIUS}" fill="#ffffff" stroke="${css_color('--s-op')}" stroke-width="1.4"/>`;
      svg_text += `<text x="${center_x}" y="${center_y}" text-anchor="middle" dominant-baseline="central" font-size="15" fill="${css_color('--ink')}"${glyph_style}>${escape_svg_text(OPERATOR_SPECS[node.type].symbol)}</text>`;
      const side_text = node.type === 'Concat' ? `dim ${node.params.dim}` : operator_side_label(node);
      if(side_text){
        const label_box = side_label_boxes.get(node.id) || {x: node.x + 2 * OPERATOR_RADIUS - 2, y: node.y - 13, h: 13};
        svg_text += `<text x="${label_box.x}" y="${label_box.y + label_box.h / 2}" dominant-baseline="central" font-size="10.5" fill="${css_color('--ink2')}" ${font_attribute}>${escape_svg_text(side_text)}</text>`;
      }
      continue;
    }
    const kind_name = node_kind(node.type);
    svg_text += `<rect x="${node.x + 0.7}" y="${node.y + 0.7}" width="${width - 1.4}" height="${height - 1.4}"${kind_name === 'io' ? ' rx="8"' : ''} fill="${css_color('--f-' + kind_name)}" stroke="${css_color('--s-' + kind_name)}" stroke-width="1.4"/>`;
    const source_ids = input_sources(graph, node), output_shape = current_inference.shapes[node.id];
    const input_text = (node.type === 'Attention' ? source_ids.slice(0, 1) : source_ids).map(source_id => source_id != null ? dims_text(current_inference.shapes[source_id]) : '?').join(', ') || '?';
    const dims_label = node.type === 'Input' ? 'B×' + (output_shape ? dims_text(output_shape) : split_list_text(node.params.shape).join('×'))
      : node.type === 'Output' ? (source_ids[0] != null && current_inference.shapes[source_ids[0]] ? 'B×' + input_text : '?')
      : (node.type === 'Custom' && source_ids.length > 1) ? dims_text(output_shape) : `${input_text} → ${dims_text(output_shape)}`;
    const dims_tspan = node.type === 'Activation' ? '' : `<tspan dx="6" font-size="10.5" font-weight="400" fill="${css_color('--ink2')}">${escape_svg_text(dims_label.replace(' → ', '→'))}</tspan>`;
    svg_text += `<text x="${node.x + width / 2}" y="${node.y + height / 2}" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="600" fill="${css_color('--ink')}" ${font_attribute}>${escape_svg_text(node_title(node))}${dims_tspan}</text>`;
  }
  return {svg: svg_text + '</svg>', w: right - left, h: bottom - top};
}

/**
 * File name stem for figures of the current tab.
 * @returns {string} "network" for the main graph, otherwise the block name
 */
function figure_file_stem(){
  return current_tab === 'main' ? 'network' : current_tab;
}

/** Save the current tab as an SVG figure. */
function save_figure_svg(){
  const figure = build_figure_svg();
  if(figure) save_file(figure_file_stem() + '.svg', figure.svg, 'image/svg+xml');
}

/** Save the current tab as a PNG figure (3× resolution). */
function save_figure_png(){
  const figure = build_figure_svg();
  if(!figure) return;
  const figure_image = new Image(), pixel_scale = 3;
  figure_image.onload = () => {
    const drawing_canvas = document.createElement('canvas');
    drawing_canvas.width = figure.w * pixel_scale;
    drawing_canvas.height = figure.h * pixel_scale;
    const drawing_context = drawing_canvas.getContext('2d');
    drawing_context.scale(pixel_scale, pixel_scale);
    drawing_context.drawImage(figure_image, 0, 0);
    drawing_canvas.toBlob(png_blob => save_file(figure_file_stem() + '.png', png_blob));
  };
  figure_image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(figure.svg);
}
