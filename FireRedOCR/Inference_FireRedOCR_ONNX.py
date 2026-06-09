import os
import time

import numpy as np
import onnxruntime
from PIL import Image
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(SCRIPT_DIR, 'FireRedOCR_Optimized')

tokenizer_path                 = r'/home/DakeQQ/Downloads/FireRed-OCR'                         # Set the FireRedOCR-Tokenizer folder path.
onnx_model_Embed               = os.path.join(EXPORT_DIR, 'LLM_Embed.onnx')
onnx_model_Vision              = os.path.join(EXPORT_DIR, 'LLM_Vision.onnx')
onnx_model_Rotary_Prefill      = os.path.join(EXPORT_DIR, 'Rotary_Prefill.onnx')
onnx_model_Rotary_Decode       = os.path.join(EXPORT_DIR, 'Rotary_Decode.onnx')
onnx_model_Main                = os.path.join(EXPORT_DIR, 'LLM_Main.onnx')
onnx_model_Greedy              = os.path.join(EXPORT_DIR, 'Greedy_Search.onnx')
onnx_model_First_Beam          = os.path.join(EXPORT_DIR, 'First_Beam_Search.onnx')
onnx_model_Second_Beam         = os.path.join(EXPORT_DIR, 'Second_Beam_Search.onnx')
onnx_model_Penalty             = os.path.join(EXPORT_DIR, 'Apply_Penalty.onnx')
onnx_model_Argmax              = os.path.join(EXPORT_DIR, 'Argmax.onnx')
onnx_model_KV_Slice            = os.path.join(EXPORT_DIR, 'KV_Slice.onnx')


# Test input
TEST_IMAGE               = ["./psyduck_2.png"]     # List of test images for the exported onnx model.
TEST_QUERY               = 'Transcribe this document exactly.'

# Model Config
STOP_TOKEN               = [151643, 151645]        # FireRedOCR stop token ids
MAX_SEQ_LEN              = 4096                    # Max context length. Can not edit after export.

# Vision Config
INPUT_IMAGE_SIZE         = [960, 960]              # Input image shape. Keep the same value as exported model.
VISION_BATCH_SIZE        = 1                       # Number of images supported by the prompt. Keep the same value as exported model.
INPUT_IMAGE_DIM          = 5                       # 4 for [batch, 3, height, width]; 5 for [batch, 1, 3, height, width]

# KV cache quantization
KV_QUANT_DTYPE           = "F16"                   # Keep the same value as exported model.
USE_SYM                  = False                   # Keep the same value as exported model.

# Decoding strategy
USE_BEAM_SEARCH          = False                   # Use beam search or greedy search
REPEAT_PENALTY           = 1.0                     # 0.0 ~ 1.0; No penalty = 1.0
PENALTY_RANGE            = 20                      # Recent-token window to apply penalty
TOP_K                    = 3                       # Top-K for beam search
BEAM_SIZE                = 3                       # Beam size for beam search. Must be <= MAX_BEAM_SIZE

# Runtime config
ORT_LOG                  = False                   # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                 = False                   # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers = []                      # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS              = 0                       # 0 = auto
DEVICE_ID                = 0                       # Device ID for GPU


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def bind_ort_in_buf(binding, names, values):
    """Bind OrtValue inputs by name."""
    for name, val in zip(names, values):
        binding.bind_ortvalue_input(name, val)


def bind_ort_out_buf(binding, names, values):
    """Bind OrtValue outputs by name."""
    for name, val in zip(names, values):
        binding.bind_ortvalue_output(name, val)


def bind_ort_out(binding, names, device):
    """Bind outputs by name, letting ORT allocate on `device`."""
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    """Create an OrtValue from a Python list/scalar."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    """Create a zero-filled OrtValue with the given shape."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_ort_from_numpy(array, device, device_id):
    """Create an OrtValue from an existing numpy array."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    """Create an ORT InferenceSession with standard options."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def load_images(image_paths, target_h, target_w):
    """Load test images as uint8 NCHW tensors using aspect-preserving letterbox."""
    if not image_paths:
        raise ValueError('TEST_IMAGE must contain at least one image path.')
    if DYNAMIC_IMAGE_SHAPE:
        if len(image_paths) > VISION_BATCH_SIZE:
            raise ValueError(f'Got {len(image_paths)} images, but VISION_BATCH_SIZE={VISION_BATCH_SIZE}.')
    elif len(image_paths) != VISION_BATCH_SIZE:
        raise ValueError(f'Expected exactly {VISION_BATCH_SIZE} images, but got {len(image_paths)}.')

    resampling = getattr(getattr(Image, 'Resampling', Image), 'BICUBIC')
    pixel_values = np.empty((len(image_paths), 3, target_h, target_w), dtype=np.uint8)
    for i, path in enumerate(image_paths):
        with Image.open(path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            src_w, src_h = image.size
            scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
            resize_w = max(1, min(target_w, int(round(src_w * scale))))
            resize_h = max(1, min(target_h, int(round(src_h * scale))))
            if image.size != (resize_w, resize_h):
                image = image.resize((resize_w, resize_h), resampling)

            canvas = Image.new('RGB', (target_w, target_h), (127, 127, 127))
            offset_x = (target_w - resize_w) // 2
            offset_y = (target_h - resize_h) // 2
            canvas.paste(image, (offset_x, offset_y))

        pixel_values[i] = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)

    if INPUT_IMAGE_DIM == 5:
        pixel_values = np.expand_dims(pixel_values, axis=1)
    return np.ascontiguousarray(pixel_values)


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,                 # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 ** 3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',   # ["default", "high_performance", "minimum_power"] ; Default (Gpus first), HighPerformance (GPUs first), LowPower (NPUs first)
        'device_filter':              'gpu',                # [gpu, npu, any],
        'disable_metacommands':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'               # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
print('Loading ONNX models . . . it could cost minutes.')

# --- Vision Fused (Preprocess + Encoder + Concat) ---
ort_session_Vision = create_session(onnx_model_Vision, **packed_settings)
binding_Vision     = ort_session_Vision.io_binding()
in_name_Vision     = get_in_names(ort_session_Vision)
out_name_Vision    = get_out_names(ort_session_Vision)
deepstack_features_len = len(out_name_Vision) - 1

# --- Read vision config from fused ONNX model metadata ---
_img_meta_shape = ort_session_Vision._inputs_meta[0].shape
if INPUT_IMAGE_DIM == 5:
    _img_h, _img_w = _img_meta_shape[3], _img_meta_shape[4]
    _vision_batch_from_meta = _img_meta_shape[0]
else:
    _img_h, _img_w = _img_meta_shape[2], _img_meta_shape[3]
    _vision_batch_from_meta = _img_meta_shape[0]
if isinstance(_img_h, int) and isinstance(_img_w, int):
    INPUT_IMAGE_SIZE = [_img_h, _img_w]
if isinstance(_vision_batch_from_meta, int):
    VISION_BATCH_SIZE = _vision_batch_from_meta
DYNAMIC_IMAGE_SHAPE = not isinstance(_vision_batch_from_meta, int)

# --- Embed ---
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

# --- Rotary (Prefill) ---
ort_session_Rotary_Prefill = create_session(onnx_model_Rotary_Prefill, **packed_settings)
binding_Rotary_Prefill     = ort_session_Rotary_Prefill.io_binding()
in_name_Rotary_Prefill     = get_in_names(ort_session_Rotary_Prefill)
out_name_Rotary_Prefill    = get_out_names(ort_session_Rotary_Prefill)

# --- Rotary (Decode) ---
ort_session_Rotary_Decode = create_session(onnx_model_Rotary_Decode, **packed_settings)
binding_Rotary_Decode     = ort_session_Rotary_Decode.io_binding()
in_name_Rotary_Decode     = get_in_names(ort_session_Rotary_Decode)[0]
out_name_Rotary_Decode    = get_out_names(ort_session_Rotary_Decode)
out_meta_Rotary_Decode    = ort_session_Rotary_Decode._outputs_meta

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main  = get_in_names(ort_session_Main)
out_name_Main = get_out_names(ort_session_Main)
in_meta_Main  = ort_session_Main._inputs_meta

# Derived index offsets for accessing beam/greedy extra inputs
num_keys_values_Main        = len(out_name_Main)   - 1
num_keys_values_Main_plus_1 = num_keys_values_Main + 1
num_keys_values_Main_plus_2 = num_keys_values_Main + 2
num_keys_values_Main_plus_3 = num_keys_values_Main + 3

# Partitioned name lists
in_name_Main_kv      = in_name_Main[:num_keys_values_Main]
out_name_Main_kv     = out_name_Main[:num_keys_values_Main]
out_name_Main_logits = out_name_Main[num_keys_values_Main]
idx_rotary_cos       = num_keys_values_Main_plus_1 + deepstack_features_len
deepstack_in_name_Main = in_name_Main[num_keys_values_Main_plus_1: idx_rotary_cos]

# Dtype introspection
kv_dtype_str      = in_meta_Main[0].type
hidden_dtype_Main = np.float16 if 'float16' in in_meta_Main[num_keys_values_Main].type else np.float32
vocab_size        = ort_session_Main._outputs_meta[num_keys_values_Main].shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# KV CACHE SETUP
# ══════════════════════════════════════════════════════════════════════════════

if 'uint8' in kv_dtype_str or 'int8' in kv_dtype_str or 'int32' in kv_dtype_str:
    if 'int32' in kv_dtype_str:
        kv_dtype_Main = np.int32
    elif 'uint8' in kv_dtype_str:
        kv_dtype_Main = np.uint8
    else:
        kv_dtype_Main = np.int8
    _is_rotary_rt   = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
    _is_quantized_rt = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
    _kv_sym_rt      = USE_SYM and (_is_rotary_rt or _is_quantized_rt)

    # Determine number of tensor types to find num_layers_Main
    if _kv_sym_rt:
        _num_types = 4
    else:
        _num_types = 6

    num_layers_Main = num_keys_values_Main // _num_types
    scale_dtype     = np.float16 if 'float16' in in_meta_Main[num_layers_Main * 2].type else np.float32

    if _kv_sym_rt:
        # Symmetric: scale only, no bias
        k_scale_shape   = list(in_meta_Main[num_layers_Main * 2].shape)
        k_scale_shape[0] = 1
        k_scale_shape[-1] = 0
        v_scale_shape   = list(in_meta_Main[num_layers_Main * 3].shape)
        v_scale_shape[0] = 1
        v_scale_shape[3] = 0
        k_scales        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        k_biases        = None
        v_scales        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_biases        = None
    else:
        # Asymmetric: scale + bias
        k_scale_shape   = list(in_meta_Main[num_layers_Main * 2].shape)
        k_scale_shape[0] = 1
        k_scale_shape[-1] = 0
        v_scale_idx     = num_layers_Main * 4
        v_scale_shape   = list(in_meta_Main[v_scale_idx].shape)
        v_scale_shape[0] = 1
        v_scale_shape[3] = 0
        k_scales        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        k_biases        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_scales        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_biases        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
else:
    kv_dtype_Main   = np.float16 if 'float16' in kv_dtype_str else np.float32
    num_layers_Main = num_keys_values_Main // 2
    k_scales        = None

past_keys_Main   = create_ort_with_shape((1, in_meta_Main[0].shape[1],               1, in_meta_Main[0].shape[3],               0), kv_dtype_Main, kv_device, DEVICE_ID)
past_values_Main = create_ort_with_shape((1, in_meta_Main[num_layers_Main].shape[1], 1, 0, in_meta_Main[num_layers_Main].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# DECODING STRATEGY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = (REPEAT_PENALTY != 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & STOP TOKENS & PROMPT
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, fix_mistral_regex=True)

STOP_TOKEN_SET = set(STOP_TOKEN)

pixel_values = load_images(TEST_IMAGE, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1])

pixel_values_ort = create_ort_from_numpy(pixel_values, device_type, DEVICE_ID)
binding_Vision.bind_ortvalue_input(in_name_Vision[0], pixel_values_ort)
bind_ort_out(binding_Vision, out_name_Vision, _ort_device_type)
run(ort_session_Vision, binding_Vision)
vision_outputs = binding_Vision.get_outputs()
concat_hidden_states = vision_outputs[deepstack_features_len].numpy().astype(hidden_dtype_Main, copy=False)
del pixel_values, pixel_values_ort
num_prefill = concat_hidden_states.shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════

# --- Input OrtValues ---
hidden_states_prefill = create_ort_from_numpy(concat_hidden_states, device_type, DEVICE_ID)
ids_len               = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
init_history_len      = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
topK                  = create_ort_with_data([TOP_K], np.int64, device_type, DEVICE_ID)
beam_size             = create_ort_with_data([BEAM_SIZE], np.int64, device_type, DEVICE_ID)

# --- Decode-phase placeholder buffers (reused every step) ---
attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1),                                            hidden_dtype_Main, device_type, DEVICE_ID)
rotary_cos_buf     = create_ort_with_shape(out_meta_Rotary_Decode[0].shape,                                   hidden_dtype_Main, device_type, DEVICE_ID)
rotary_sin_buf     = create_ort_with_shape(out_meta_Rotary_Decode[1].shape,                                   hidden_dtype_Main, device_type, DEVICE_ID)
hidden_states_buf  = create_ort_with_shape((BEAM_SIZE, 1, in_meta_Main[num_keys_values_Main].shape[2]), hidden_dtype_Main, device_type, DEVICE_ID)
save_id_buf        = create_ort_with_shape((BEAM_SIZE, 0),                                              np.int32,          device_type, DEVICE_ID)
init_deepstack_features = [
    create_ort_with_shape((1, 1, in_meta_Main[num_keys_values_Main].shape[2]), hidden_dtype_Main, device_type, DEVICE_ID)
    for _ in range(deepstack_features_len)
]

# --- Logits & token-index buffers ---
prefill_logits_buf = create_ort_with_shape((1, vocab_size),         hidden_dtype_Main, device_type, DEVICE_ID)
decode_logits_buf  = create_ort_with_shape((BEAM_SIZE, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
max_idx_buf        = create_ort_with_shape((1, 1),                  np.int32,          device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# DECODE HEAD SESSIONS (Beam Search OR Greedy/Argmax)
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")

    # --- First Beam ---
    ort_session_First_Beam     = create_session(onnx_model_First_Beam, **packed_settings)
    binding_First_Beam         = ort_session_First_Beam.io_binding()
    in_name_First_Beam         = get_in_names(ort_session_First_Beam)
    out_name_First_Beam        = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_parts   = in_name_First_Beam[:num_keys_values_Main_plus_1]
    out_name_First_Beam_parts  = out_name_First_Beam[:num_keys_values_Main_plus_1]
    out_name_First_Beam_others = out_name_First_Beam[num_keys_values_Main_plus_1:]

    # --- Second Beam ---
    ort_session_Second_Beam     = create_session(onnx_model_Second_Beam, **packed_settings)
    binding_Second_Beam         = ort_session_Second_Beam.io_binding()
    in_name_Second_Beam         = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam        = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_parts   = in_name_Second_Beam[:num_keys_values_Main_plus_1]
    out_name_Second_Beam_parts  = out_name_Second_Beam[:num_keys_values_Main_plus_1]
    out_name_Second_Beam_others = out_name_Second_Beam[num_keys_values_Main_plus_1:]

    # --- Beam-specific buffers ---
    beam_ids_buf   = create_ort_with_shape((BEAM_SIZE, 1), np.int32,          device_type, DEVICE_ID)
    beam_score_buf = create_ort_with_shape((BEAM_SIZE, 1), hidden_dtype_Main, device_type, DEVICE_ID)

    # --- Static beam bindings ---
    bind_ort_in_buf(binding_First_Beam, in_name_First_Beam[num_keys_values_Main_plus_1: num_keys_values_Main_plus_3], [save_id_buf, beam_size])
    bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam[num_keys_values_Main_plus_3:], [beam_size, topK])
else:
    # --- Greedy ---
    ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
    binding_Greedy     = ort_session_Greedy.io_binding()
    in_name_Greedy     = get_in_names(ort_session_Greedy)
    out_name_Greedy    = get_out_names(ort_session_Greedy)
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

    # --- Argmax ---
    ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
    binding_Argmax     = ort_session_Argmax.io_binding()
    in_name_Argmax     = get_in_names(ort_session_Argmax)[0]
    out_name_Argmax    = get_out_names(ort_session_Argmax)[0]
    save_id_numpy      = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# PENALTY SESSION (optional)
# ══════════════════════════════════════════════════════════════════════════════
if USE_PENALTY:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
    binding_Penalty     = ort_session_Penalty.io_binding()
    in_name_Penalty     = get_in_names(ort_session_Penalty)
    out_name_Penalty    = get_out_names(ort_session_Penalty)[0]

    penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)

    bind_ort_in_buf(binding_Penalty, in_name_Penalty[2:], [penalty_value, penalty_range])


# ══════════════════════════════════════════════════════════════════════════════
# PREFILL PHASE
# ══════════════════════════════════════════════════════════════════════════════
is_prefill_step = True
prefill_start_time = time.time()

# --- Step 1: Use precomputed multimodal hidden states ---
hidden_states = hidden_states_prefill

# Pre-bind Embed input for decode phase (will read from max_idx_buf)
binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

# --- Step 2: Compute rotary embeddings & causal mask (prefill) ---
bind_ort_in_buf(binding_Rotary_Prefill, in_name_Rotary_Prefill, [ids_len, init_history_len])
bind_ort_out(binding_Rotary_Prefill, out_name_Rotary_Prefill, _ort_device_type)
run(ort_session_Rotary_Prefill, binding_Rotary_Prefill)
rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Prefill.get_outputs()

# --- Step 3: Pre-bind decode rotary outputs (reused every decode step) ---
binding_Rotary_Decode.bind_ortvalue_input(in_name_Rotary_Decode, kv_seq_len)
bind_ort_out_buf(binding_Rotary_Decode, out_name_Rotary_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len])

# --- Step 4: Bind Main model inputs — non-KV (hidden_states, deepstack, rotary, mask) ---
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], hidden_states)
bind_ort_in_buf(binding_Main, deepstack_in_name_Main, vision_outputs[:deepstack_features_len])
bind_ort_in_buf(binding_Main, in_name_Main[idx_rotary_cos:], [rotary_cos, rotary_sin, attention_mask])

# --- Step 5: Bind Main model inputs — empty KV cache (keys, values, optional scales/biases) ---
i = 0
for _ in range(num_layers_Main):
    binding_Main.bind_ortvalue_input(in_name_Main[i], past_keys_Main)
    i += 1
for _ in range(num_layers_Main):
    binding_Main.bind_ortvalue_input(in_name_Main[i], past_values_Main)
    i += 1
if k_scales is not None:
    if k_biases is not None:
        # Asymmetric: bind k_scale, k_bias, v_scale, v_bias
        for j in (k_scales, k_biases, v_scales, v_biases):
            for _ in range(num_layers_Main):
                binding_Main.bind_ortvalue_input(in_name_Main[i], j)
                i += 1
    else:
        # Symmetric: bind k_scale, v_scale only (no bias)
        for j in (k_scales, v_scales):
            for _ in range(num_layers_Main):
                binding_Main.bind_ortvalue_input(in_name_Main[i], j)
                i += 1

# --- Step 6: Bind Main model outputs ---
bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

# --- Step 7: Bind penalty inputs/outputs to prefill logits buffer ---
if USE_PENALTY:
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
    binding_Penalty.bind_ortvalue_output(out_name_Penalty,  prefill_logits_buf)

# --- Step 8: Bind decode head inputs/outputs to prefill logits buffer ---
if USE_BEAM_SEARCH:
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_Main], prefill_logits_buf)
elif USE_PENALTY:
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[0],   prefill_logits_buf)
    binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
else:
    binding_Argmax.bind_ortvalue_input(in_name_Argmax,   prefill_logits_buf)
    binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)


# ══════════════════════════════════════════════════════════════════════════════
# DECODE LOOP
# ══════════════════════════════════════════════════════════════════════════════
print(f'\nTest Question: {TEST_QUERY}\nLLM Answering:')

num_decode     = 0
generate_limit = MAX_SEQ_LEN - num_prefill

while num_decode < generate_limit:

    # ── 1. Run Main Model ────────────────────────────────────────────────
    run(ort_session_Main, binding_Main)
    outputs_Main = binding_Main.get_outputs()

    # ── 2. Apply Repetition Penalty (if enabled and enough tokens) ───────
    if USE_PENALTY and num_decode >= PENALTY_RANGE:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
        run(ort_session_Penalty, binding_Penalty)

    # ── 3. Token Selection ───────────────────────────────────────────────
    if USE_BEAM_SEARCH:
        # ── 3a. Beam Search ─────────────────────────────────────────────
        if is_prefill_step:
            # First beam step: expand single-beam KV into BEAM_SIZE beams
            bind_ort_in_buf(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
            bind_ort_out(binding_First_Beam, out_name_First_Beam_parts, _ort_device_type)
            bind_ort_out_buf(binding_First_Beam, out_name_First_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
            run(ort_session_First_Beam, binding_First_Beam)
            outputs_Beam = binding_First_Beam.get_outputs()
        else:
            # Subsequent beam steps: prune + expand
            bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
            bind_ort_out(binding_Second_Beam, out_name_Second_Beam_parts, _ort_device_type)
            if num_decode < 2:
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_2], beam_score_buf)
                bind_ort_out_buf(binding_Second_Beam, out_name_Second_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
            run(ort_session_Second_Beam, binding_Second_Beam)
            outputs_Beam = binding_Second_Beam.get_outputs()

        # Stop-token check
        max_logits_idx = max_idx_buf.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET:
            break

        # Feed beam KV + save_id back into Main for next step
        save_id = outputs_Beam[num_keys_values_Main]
        bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Beam)
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_1], save_id)

    else:
        # ── 3b. Greedy / Argmax ─────────────────────────────────────────
        if USE_PENALTY:
            binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
            run(ort_session_Greedy, binding_Greedy)
            save_id = binding_Greedy.get_outputs()[1]
        else:
            run(ort_session_Argmax, binding_Argmax)

        # Stop-token check
        max_logits_idx = max_idx_buf.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET:
            break

        # Track generated token IDs
        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
        else:
            save_id_numpy[num_decode] = max_logits_idx

        # Feed greedy KV outputs back into Main
        bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Main)

        # Streaming print
        print(tokenizer.decode(max_logits_idx), end="", flush=True)

    # ── 4. Re-bind Main KV outputs (ORT allocates fresh each step) ───────
    bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

    # ── 5. Transition: prefill → decode (executes once) ──────────────────
    if is_prefill_step:

        # Switch Main to decode-sized non-KV inputs
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], hidden_states_buf)
        bind_ort_in_buf(binding_Main, deepstack_in_name_Main, init_deepstack_features)
        bind_ort_in_buf(binding_Main, in_name_Main[idx_rotary_cos:], [rotary_cos_buf, rotary_sin_buf, attention_mask_buf])
        binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)

        # Switch Embed to write into decode hidden_states buffer
        binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

        # Switch Penalty to decode logits buffer
        if USE_PENALTY:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
            binding_Penalty.bind_ortvalue_output(out_name_Penalty, decode_logits_buf)

        # Switch decode head to decode logits buffer
        if USE_BEAM_SEARCH:
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main], decode_logits_buf)
            binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
        elif USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
        else:
            binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

        is_prefill_step = False

        # Record prefill time and start decode timer
        decode_start_time = time.time()
        prefill_elapsed = decode_start_time - prefill_start_time

    # ── 6. Prepare next step: Embed + Rotary ─────────────────────────────
    run(ort_session_Embed, binding_Embed)
    run(ort_session_Rotary_Decode, binding_Rotary_Decode)
    num_decode += 1


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
decode_end_time = time.time()

# Handle edge case where generation stopped at prefill (0 decode tokens after first)
if num_decode < 2:
    # Only prefill happened (or single token generated during prefill step)
    prefill_elapsed = 0.0
    decode_elapsed = 0.0
else:
    decode_elapsed = decode_end_time - decode_start_time

total_elapsed = decode_end_time - prefill_start_time

# Prefill speed: tokens processed per second
prefill_tokens_per_second = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0

# Decode speed: tokens generated per second (excluding the first token from prefill)
decode_tokens_per_second = num_decode / decode_elapsed if decode_elapsed > 0 else 0.0

# Overall speed
overall_tokens_per_second = (num_decode + 1) / total_elapsed if total_elapsed > 0 else 0.0

if USE_PENALTY or USE_BEAM_SEARCH:
    result = tokenizer.decode(save_id.numpy().flat[:num_decode], skip_special_tokens=True)
else:
    result = tokenizer.decode(save_id_numpy[:num_decode], skip_special_tokens=True)

print(
    f"\n\n{'─' * 56}\n"
    f"  📝 Generated Output\n"
    f"{'─' * 56}\n"
    f"{result}\n"
    f"{'─' * 56}\n\n"
    f"  ⚡ Performance Summary\n"
    f"{'─' * 56}\n"
    f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
    f"  {'─' * 48}\n"
    f"  {'Prefill':<12} {prefill_tokens_per_second:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
    f"  {'Decode':<12} {decode_tokens_per_second:>10.2f} t/s {num_decode:>8d} {decode_elapsed:>8.3f}s\n"
    f"  {'─' * 48}\n"
    f"  {'Overall':<12} {overall_tokens_per_second:>10.2f} t/s {num_decode:>8d} {total_elapsed:>8.3f}s\n"
    f"{'─' * 56}\n"
)
