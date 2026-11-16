// Example networks: Custom MLP, ResNet, U-Net and an encoder–decoder Transformer.

/**
 * Helpers that add nodes and edges to one graph.
 * @param {object} graph
 * @returns {{add: Function, link: Function, chain: Function}}
 *   add(node_type, extra?, params?) → node;  link(from_node, to_node, port_index?);  chain(...nodes) → last node
 */
function graph_builder(graph){
  const add = (node_type, extra = {}, params = {}) => {
    const node = add_node(node_type, 0, 0, extra, graph);
    Object.assign(node.params, params);
    return node;
  };
  const link = (from_node, to_node, port_index = 0) => connect_nodes(from_node.id, to_node.id, port_index, graph);
  const chain = (...nodes) => {
    for(let node_index = 1; node_index < nodes.length; node_index++) link(nodes[node_index - 1], nodes[node_index]);
    return nodes[nodes.length - 1];
  };
  return {add, link, chain};
}

/**
 * Create an empty block graph that is laid out when its tab is first opened.
 * @param {string} block_name
 * @returns {object} block graph
 */
function new_block_graph(block_name){
  const block_graph = {nodes: [], edges: [], needs_layout: true};
  state.blocks[block_name] = block_graph;
  return block_graph;
}

/**
 * Add a "same" padded Conv2d.
 * @param {Function} add  graph_builder add
 * @param {number} out_channels
 * @param {number} kernel_size  odd
 * @returns {object} node
 */
function add_conv(add, out_channels, kernel_size = 3){
  return add('Conv2d', {}, {out_channels, kernel_size, stride: 1, padding: (kernel_size - 1) / 2});
}

/**
 * Add a ReLU activation.
 * @param {Function} add  graph_builder add
 * @returns {object} node
 */
function add_relu(add){
  return add('Activation', {fn: 'ReLU'});
}

/**
 * Conv-BN-ReLU twice.
 * @param {Function} add  graph_builder add
 * @param {number} out_channels
 * @returns {object[]} six nodes in order
 */
function add_double_conv(add, out_channels){
  return [add_conv(add, out_channels), add('BatchNorm2d'), add_relu(add), add_conv(add, out_channels), add('BatchNorm2d'), add_relu(add)];
}

/** x [784] → Linear → ReLU → MLPBlock, residual ⊕, custom function, concat, Linear(10) → y [10]. */
function build_example_mlp(){
  load_state({});
  {
    const {add, chain} = graph_builder(new_block_graph('MLPBlock'));
    chain(add('Input', {shape: DEFAULT_HIDDEN_SIZE}), add('Linear'), add('Activation', {fn: 'GELU'}), add('Linear'), add('Output'));
  }
  const {add, link, chain} = graph_builder(state.main);
  const input_node = add('Input'), first_linear = add('Linear'), relu_node = add_relu(add), mlp_block = add('Custom', {block: 'MLPBlock'});
  const residual_add = add('Add'), custom_function = add('CustomFn'), concat_node = add('Concat'), head = add('Linear', {}, {out_features: 10}), output_node = add('Output');
  chain(input_node, first_linear, relu_node, mlp_block);
  link(relu_node, residual_add, 0); link(mlp_block, residual_add, 2);
  chain(residual_add, custom_function);
  link(relu_node, concat_node, 0); link(custom_function, concat_node, 2);
  chain(concat_node, head, output_node);
}

/** image [3, 32, 32] → stem conv → ResBlock ×2 → global pool → Linear(10) → y [10]. */
function build_example_resnet(){
  load_state({});
  {
    const {add, link, chain} = graph_builder(new_block_graph('ResBlock'));
    const block_input = add('Input', {shape: '64,32,32'});
    const residual_branch_end = chain(block_input, add_conv(add, 64), add('BatchNorm2d'), add_relu(add), add_conv(add, 64), add('BatchNorm2d'));
    const residual_add = add('Add');
    link(block_input, residual_add, 0); link(residual_branch_end, residual_add, 2);
    chain(residual_add, add_relu(add), add('Output'));
  }
  const {add, chain} = graph_builder(state.main);
  chain(add('Input', {preset: 'image'}), add_conv(add, 64), add('BatchNorm2d'), add_relu(add),
    add('Custom', {block: 'ResBlock'}), add('Custom', {block: 'ResBlock'}),
    add('AdaptiveAvgPool2d'), add('Flatten'), add('Linear', {}, {out_features: 10}), add('Output'));
}

/** image [3, 64, 64] → Inc → Down1 → Down2 → Up1(skip=Down1, x) → Up2(skip=Inc, x) → 1×1 Conv → y [1, 64, 64]. */
function build_example_unet(){
  load_state({});
  {
    const {add, chain} = graph_builder(new_block_graph('Inc'));
    chain(add('Input', {shape: '3,64,64'}), ...add_double_conv(add, 32), add('Output'));
  }
  {
    const {add, chain} = graph_builder(new_block_graph('Down1'));
    chain(add('Input', {shape: '32,64,64'}), add('MaxPool2d'), ...add_double_conv(add, 64), add('Output'));
  }
  {
    const {add, chain} = graph_builder(new_block_graph('Down2'));
    chain(add('Input', {shape: '64,32,32'}), add('MaxPool2d'), ...add_double_conv(add, 128), add('Output'));
  }
  const add_up_block = (block_name, out_channels, input_shape_text, skip_shape_text) => {
    const {add, link, chain} = graph_builder(new_block_graph(block_name));
    const skip_input = add('Input', {shape: skip_shape_text}), upsampled_input = add('Input', {shape: input_shape_text});
    upsampled_input.params.name = 'x';
    skip_input.params.name = 'skip';
    const upsample = chain(upsampled_input, add('Upsample')), concat_node = add('Concat', {}, {dim: 1});
    link(upsample, concat_node, 0); link(skip_input, concat_node, 1);
    chain(concat_node, ...add_double_conv(add, out_channels), add('Output'));
  };
  add_up_block('Up1', 64, '128,16,16', '64,32,32');
  add_up_block('Up2', 32, '64,32,32', '32,64,64');
  const {add, link, chain} = graph_builder(state.main);
  const input_node = add('Input', {preset: 'image'}, {shape: '3,64,64'});
  const inc_block = add('Custom', {block: 'Inc'}), down1_block = add('Custom', {block: 'Down1'}), down2_block = add('Custom', {block: 'Down2'});
  const up1_block = add('Custom', {block: 'Up1'}), up2_block = add('Custom', {block: 'Up2'}), head = add_conv(add, 1, 1), output_node = add('Output');
  chain(input_node, inc_block, down1_block, down2_block);
  link(down1_block, up1_block, 0); link(down2_block, up1_block, 1);
  link(inc_block, up2_block, 0); link(up1_block, up2_block, 1);
  chain(up2_block, head, output_node);
}

/**
 * Encoder–decoder Transformer (pre-norm), embed dim 128.
 * src [16, 64] → Linear → EncoderBlock ×2 → LayerNorm = memory;  tgt [12, 64] → Linear → DecoderBlock(x, memory) ×2 → LayerNorm → Linear(10) → y [12, 10].
 */
function build_example_transformer(){
  load_state({});
  const embed_dim = 128, source_length = 16, target_length = 12;
  build_attention_block('SelfAttention', embed_dim, source_length);
  build_attention_block('MaskedSelfAttention', embed_dim, target_length, {masked: true});
  build_attention_block('CrossAttention', embed_dim, target_length, {key_length: source_length});
  const add_feed_forward = (add, chain, from_node) => chain(from_node, add('LayerNorm'), add('Linear', {}, {out_features: 4 * embed_dim}), add('Activation', {fn: 'GELU'}), add('Linear', {}, {out_features: embed_dim}));
  {
    const {add, link, chain} = graph_builder(new_block_graph('EncoderBlock'));
    const block_input = add('Input', {preset: 'seq'}, {shape: `${source_length},${embed_dim}`}), norm = add('LayerNorm'), attention = add('Custom', {block: 'SelfAttention'});
    link(block_input, norm); link(norm, attention, 0); link(norm, attention, 1); link(norm, attention, 2);
    const attention_residual = add('Add');
    link(block_input, attention_residual, 0); link(attention, attention_residual, 2);
    const feed_forward_end = add_feed_forward(add, chain, attention_residual), feed_forward_residual = add('Add');
    link(attention_residual, feed_forward_residual, 0); link(feed_forward_end, feed_forward_residual, 2);
    chain(feed_forward_residual, add('Output'));
  }
  {
    const {add, link, chain} = graph_builder(new_block_graph('DecoderBlock'));
    const block_input = add('Input', {preset: 'seq'}, {shape: `${target_length},${embed_dim}`});
    const memory_input = add('Input', {preset: 'seq'}, {shape: `${source_length},${embed_dim}`});
    memory_input.params.name = 'memory';
    const self_attention_norm = add('LayerNorm'), self_attention = add('Custom', {block: 'MaskedSelfAttention'});
    link(block_input, self_attention_norm); link(self_attention_norm, self_attention, 0); link(self_attention_norm, self_attention, 1); link(self_attention_norm, self_attention, 2);
    const self_attention_residual = add('Add');
    link(block_input, self_attention_residual, 0); link(self_attention, self_attention_residual, 2);
    const cross_attention_norm = add('LayerNorm'), cross_attention = add('Custom', {block: 'CrossAttention'});
    link(self_attention_residual, cross_attention_norm); link(cross_attention_norm, cross_attention, 0); link(memory_input, cross_attention, 1); link(memory_input, cross_attention, 2);
    const cross_attention_residual = add('Add');
    link(self_attention_residual, cross_attention_residual, 0); link(cross_attention, cross_attention_residual, 2);
    const feed_forward_end = add_feed_forward(add, chain, cross_attention_residual), feed_forward_residual = add('Add');
    link(cross_attention_residual, feed_forward_residual, 0); link(feed_forward_end, feed_forward_residual, 2);
    chain(feed_forward_residual, add('Output'));
  }
  const {add, link, chain} = graph_builder(state.main);
  const source_input = add('Input', {preset: 'seq'}, {shape: `${source_length},64`}), target_input = add('Input', {preset: 'seq'}, {shape: `${target_length},64`});
  source_input.params.name = 'src';
  target_input.params.name = 'tgt';
  const memory = chain(source_input, add('Linear', {}, {out_features: embed_dim}), add('Custom', {block: 'EncoderBlock'}), add('Custom', {block: 'EncoderBlock'}), add('LayerNorm'));
  const target_embedding = chain(target_input, add('Linear', {}, {out_features: embed_dim}));
  const first_decoder = add('Custom', {block: 'DecoderBlock'}), second_decoder = add('Custom', {block: 'DecoderBlock'});
  link(target_embedding, first_decoder, 0); link(memory, first_decoder, 1);
  link(first_decoder, second_decoder, 0); link(memory, second_decoder, 1);
  chain(second_decoder, add('LayerNorm'), add('Linear', {}, {out_features: 10}), add('Output'));
}

const EXAMPLE_BUILDERS = {mlp: build_example_mlp, resnet: build_example_resnet, unet: build_example_unet, transformer: build_example_transformer};

/**
 * Replace the graph with an example and lay it out.
 * @param {string} example_name  'mlp' | 'resnet' | 'unet' | 'transformer'
 */
function load_example(example_name){
  EXAMPLE_BUILDERS[example_name]();
  render_all();
  auto_layout();
  render_all();
  fit_view();
}
