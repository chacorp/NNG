// Node type definitions: layers, operators, parameter editors and input layouts.
// Shapes everywhere exclude the batch dimension, e.g. an image tensor [B, 3, 32, 32] is written [3, 32, 32].

const DEFAULT_HIDDEN_SIZE = 256;
const ACTIVATION_FUNCTIONS = ['ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'Softmax'];

/**
 * Throw unless the shape is [C, H, W].
 * @param {number[]} input_shape  shape without batch
 * @param {string} layer_name     used in the error message
 */
function require_chw_shape(input_shape, layer_name){
  if(input_shape.length !== 3) throw `${layer_name}는 [C,H,W] 입력 필요 (현재 ${format_shape(input_shape)})`;
}

/**
 * Layer specs.
 *   default_params            initial parameters of a new node
 *   consumed_dim              'last' | 'channel' | null — which input dim becomes the torch constructor's input size
 *   kind                      'layer' | 'norm' | 'util' | 'act' (border colour)
 *   output_shape(shape, params) → output shape (no batch); for multi-input layers the first argument is a list of shapes
 *   torch_code(params, size)  → constructor code; size is the consumed dim (or the list of input shapes)
 *   input_port_names          names of the input ports when there is more than one
 *   call_code(attribute_name, input_variable_names) → forward expression when not a plain call
 */
const LAYER_SPECS = {
  Linear: {
    default_params: {out_features: DEFAULT_HIDDEN_SIZE, bias: true}, consumed_dim: 'last', kind: 'layer',
    output_shape: (input_shape, params) => { if(!input_shape.length) throw '입력 shape 없음'; return [...input_shape.slice(0, -1), params.out_features]; },
    torch_code: (params, in_features) => `nn.Linear(${in_features}, ${params.out_features}${params.bias ? '' : ', bias=False'})`,
  },
  Conv2d: {
    default_params: {out_channels: DEFAULT_HIDDEN_SIZE, kernel_size: 3, stride: 1, padding: 1}, consumed_dim: 'channel', kind: 'layer',
    output_shape: (input_shape, params) => {
      require_chw_shape(input_shape, 'Conv2d');
      const output_size = size => Math.floor((size + 2 * params.padding - params.kernel_size) / params.stride) + 1;
      const output_height = output_size(input_shape[1]), output_width = output_size(input_shape[2]);
      if(output_height < 1 || output_width < 1) throw '출력 H/W가 0 이하';
      return [params.out_channels, output_height, output_width];
    },
    torch_code: (params, in_channels) => `nn.Conv2d(${in_channels}, ${params.out_channels}, kernel_size=${params.kernel_size}, stride=${params.stride}, padding=${params.padding})`,
  },
  MaxPool2d: {
    default_params: {kernel_size: 2}, consumed_dim: null, kind: 'util',
    output_shape: (input_shape, params) => {
      require_chw_shape(input_shape, 'MaxPool2d');
      const output_height = Math.floor(input_shape[1] / params.kernel_size), output_width = Math.floor(input_shape[2] / params.kernel_size);
      if(output_height < 1 || output_width < 1) throw '출력 H/W가 0 이하';
      return [input_shape[0], output_height, output_width];
    },
    torch_code: params => `nn.MaxPool2d(${params.kernel_size})`,
  },
  BatchNorm2d: {
    default_params: {}, consumed_dim: 'channel', kind: 'norm',
    output_shape: input_shape => { require_chw_shape(input_shape, 'BatchNorm2d'); return input_shape; },
    torch_code: (params, num_features) => `nn.BatchNorm2d(${num_features})`,
  },
  LayerNorm: {
    default_params: {}, consumed_dim: 'last', kind: 'norm',
    output_shape: input_shape => { if(!input_shape.length) throw '입력 shape 없음'; return input_shape; },
    torch_code: (params, normalized_size) => `nn.LayerNorm(${normalized_size})`,
  },
  Dropout: {
    default_params: {p: 0.1}, consumed_dim: null, kind: 'util',
    output_shape: input_shape => input_shape,
    torch_code: params => `nn.Dropout(${params.p})`,
  },
  Flatten: {
    default_params: {}, consumed_dim: null, kind: 'util',
    output_shape: input_shape => [input_shape.reduce((product, size) => product * size, 1)],
    torch_code: () => 'nn.Flatten()',
  },
  Upsample: {
    default_params: {scale_factor: 2}, consumed_dim: null, kind: 'util',
    output_shape: (input_shape, params) => { require_chw_shape(input_shape, 'Upsample'); return [input_shape[0], input_shape[1] * params.scale_factor, input_shape[2] * params.scale_factor]; },
    torch_code: params => `nn.Upsample(scale_factor=${params.scale_factor}, mode="nearest")`,
  },
  AdaptiveAvgPool2d: {
    default_params: {output_size: 1}, consumed_dim: null, kind: 'util',
    output_shape: (input_shape, params) => { require_chw_shape(input_shape, 'AdaptiveAvgPool2d'); return [input_shape[0], params.output_size, params.output_size]; },
    torch_code: params => `nn.AdaptiveAvgPool2d(${params.output_size})`,
  },
  Attention: {
    default_params: {num_heads: 8}, consumed_dim: null, kind: 'layer', input_port_names: ['q', 'k', 'v'],
    output_shape: (input_shapes, params) => {
      const [query_shape, key_shape, value_shape] = input_shapes;
      if(query_shape.length < 2 || key_shape.length < 2 || value_shape.length < 2) throw 'q, k, v는 [.., T, E] 형태 필요';
      if(!shapes_equal(key_shape.slice(0, -1), value_shape.slice(0, -1))) throw `k, v 길이 불일치 ${format_shape(key_shape)} vs ${format_shape(value_shape)}`;
      if(!shapes_equal(query_shape.slice(0, -2), key_shape.slice(0, -2))) throw 'q, k 앞쪽 차원 불일치';
      const embed_dim = query_shape[query_shape.length - 1];
      if(embed_dim % params.num_heads) throw `embed_dim ${embed_dim}이 num_heads ${params.num_heads}로 나누어지지 않음`;
      return query_shape;
    },
    torch_code: (params, input_shapes) => {
      const embed_dim = input_shapes[0].at(-1), key_dim = input_shapes[1].at(-1), value_dim = input_shapes[2].at(-1);
      return `nn.MultiheadAttention(${embed_dim}, ${params.num_heads}${key_dim !== embed_dim ? `, kdim=${key_dim}` : ''}${value_dim !== embed_dim ? `, vdim=${value_dim}` : ''}, batch_first=True)`;
    },
    call_code: (attribute_name, input_variable_names) => `self.${attribute_name}(${input_variable_names.join(', ')}, need_weights=False)[0]`,
  },
  Activation: {
    default_params: {fn: 'ReLU'}, consumed_dim: null, kind: 'act',
    output_shape: input_shape => input_shape,
    torch_code: params => params.fn === 'Softmax' ? 'nn.Softmax(dim=-1)' : `nn.${params.fn}()`,
  },
};
const LAYER_TYPE_NAMES = Object.keys(LAYER_SPECS).filter(type => type !== 'Activation');

/** Operator specs (drawn as circles). input_count: fixed number of operands; variadic: two or more. */
const OPERATOR_SPECS = {
  Add:        {symbol: '+', label: '더하기 (+)', variadic: true},
  Sub:        {symbol: '−', label: '빼기 (−)', input_count: 2},
  Hadamard:   {symbol: '⊙', label: 'Hadamard 곱 (⊙)', variadic: true},
  MatMul:     {symbol: '×', label: '행렬곱 (×, @)', input_count: 2},
  Div:        {symbol: '÷', label: '나누기 (÷)', input_count: 2},
  Concat:     {symbol: '‖', label: 'Concat (‖)', variadic: true},
  CustomFn:   {symbol: 'f', label: 'Custom function (f)', input_count: 1},
  Reshape:    {symbol: 'R', label: 'Reshape', input_count: 1},
  Permute:    {symbol: 'P', label: 'Permute', input_count: 1},
  Transpose:  {symbol: 'T', label: 'Transpose', input_count: 1},
  CausalMask: {symbol: 'M', label: 'Causal mask', input_count: 1},
};
const LETTER_OPERATORS = ['CustomFn', 'Reshape', 'Permute', 'Transpose', 'CausalMask'];

/** Output size editable directly on the box: parameter name and its index in the output shape (-1 = last). */
const INLINE_OUTPUT_PARAM = {
  Linear: {param_name: 'out_features', output_dim_index: -1},
  Conv2d: {param_name: 'out_channels', output_dim_index: 0},
};

/** Input layout presets: dim names after the batch dim B, and default sizes. */
const INPUT_LAYOUT_PRESETS = {
  vector: {label: 'Vector',   dim_names: ['C'],                shape: [784]},
  seq:    {label: 'Sequence', dim_names: ['T', 'C'],           shape: [16, 256]},
  image:  {label: 'Image',    dim_names: ['C', 'H', 'W'],      shape: [3, 32, 32]},
  video:  {label: 'Video',    dim_names: ['F', 'C', 'H', 'W'], shape: [8, 3, 32, 32]},
  mesh:   {label: 'Mesh',     dim_names: ['V', 'C'],           shape: [1024, 3]},
};
const DEFAULT_DIM_SIZES = {C: DEFAULT_HIDDEN_SIZE, H: 32, W: 32, F: 8, T: 16, V: 1024, N: 16, D: 64};

/** Editors for node parameters: value type, limits, choices and label. */
const PARAM_EDITORS = {
  name: {type: 'text'}, shape: {type: 'text'},
  out_features: {type: 'int', min: 1}, out_channels: {type: 'int', min: 1}, bias: {type: 'bool'},
  kernel_size: {type: 'int', min: 1}, stride: {type: 'int', min: 1}, padding: {type: 'int', min: 0},
  scale_factor: {type: 'int', min: 1}, output_size: {type: 'int', min: 1}, num_heads: {type: 'int', min: 1},
  p: {type: 'float', min: 0, max: 1}, fn: {type: 'select', options: ACTIVATION_FUNCTIONS}, dim: {type: 'int'},
  to: {type: 'text', label: 'reshape 크기 (배치 제외, 예: 16,-1)'},
  order: {type: 'text', label: 'permute 순서 (배치=0 고정, 예: 0,2,1)'},
  dim0: {type: 'int', label: 'dim0 (배치=0)'}, dim1: {type: 'int', label: 'dim1 (배치=0)'},
  transpose_b: {type: 'bool', label: '두 번째 입력 전치 (Bᵀ)'},
  by: {type: 'text', label: '상수로 나누기 (예: sqrt(64))'},
};

/**
 * Default dim names for an input of the given rank.
 * @param {number} rank  number of dims without batch
 * @returns {string[]} e.g. rank 3 → ['C', 'H', 'W']
 */
function default_dim_names(rank){
  return ({1: ['C'], 2: ['T', 'C'], 3: ['C', 'H', 'W'], 4: ['F', 'C', 'H', 'W']})[rank]
    || Array.from({length: rank}, (_, dim_index) => 'D' + (dim_index + 1));
}

/**
 * Split "3,32,32" / "[3, 32, 32]" / "3×32×32" into parts.
 * @param {string|number} list_text
 * @returns {string[]} e.g. ['3', '32', '32']
 */
function split_list_text(list_text){
  return String(list_text).replace(/[\[\]()\s]/g, '').split(/[,×x*]/).filter(Boolean);
}

/**
 * Dim names of an Input node (after B).
 * @param {object} input_node
 * @returns {string[]} e.g. ['C', 'H', 'W']
 */
function input_dim_names(input_node){
  return input_node.params.dims ? split_list_text(input_node.params.dims) : default_dim_names(split_list_text(input_node.params.shape).length);
}

/**
 * Layout text with the batch dim first.
 * @param {string[]} dim_names  e.g. ['C', 'H', 'W']
 * @returns {string} e.g. "B,C,H,W"
 */
function layout_text(dim_names){
  return ['B', ...dim_names].join(',');
}

/**
 * Colour kind of a node type.
 * @param {string} node_type
 * @returns {string} 'io' | 'op' | 'custom' | 'layer' | 'norm' | 'util' | 'act'
 */
function node_kind(node_type){
  if(node_type === 'Input' || node_type === 'Output') return 'io';
  if(node_type in OPERATOR_SPECS) return 'op';
  if(node_type === 'Custom') return 'custom';
  return LAYER_SPECS[node_type].kind;
}

/**
 * Whether a Div divisor text is a valid constant, e.g. "sqrt(64)" or "8".
 * @param {string} divisor_text
 * @returns {boolean}
 */
function is_valid_divisor_text(divisor_text){
  return typeof divisor_text === 'string' && divisor_text.trim() !== ''
    && /^[0-9.+\-*/() ]*(sqrt\([0-9.]+\)[0-9.+\-*/() ]*)*$/.test(divisor_text.trim());
}

/**
 * Python expression for a divisor constant.
 * @param {string} divisor_text  e.g. "sqrt(64)"
 * @returns {string} e.g. "(math.sqrt(64))"
 */
function divisor_python(divisor_text){
  return '(' + divisor_text.trim().replace(/sqrt\(/g, 'math.sqrt(') + ')';
}

/**
 * Small label drawn next to an operator circle.
 * @param {object} operator_node
 * @returns {string} e.g. "reshape(16,-1)", "Bᵀ", "√64", or '' when there is nothing to show
 */
function operator_side_label(operator_node){
  const params = operator_node.params;
  switch(operator_node.type){
    case 'CausalMask': return 'causal mask';
    case 'Reshape': return `reshape(${params.to})`;
    case 'Permute': return `permute(${params.order})`;
    case 'Transpose': return `transpose(${params.dim0}, ${params.dim1})`;
    case 'MatMul': return params.transpose_b ? 'Bᵀ' : '';
    case 'Div': return is_valid_divisor_text(params.by) ? params.by.trim().replace(/sqrt\(([^)]*)\)/g, '√$1') : '';
    default: return '';
  }
}
