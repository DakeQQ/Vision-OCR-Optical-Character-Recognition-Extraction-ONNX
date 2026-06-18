import os
import time

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer


# =============================================================================
# CONFIG  -- edit me, then just hit Run.
# =============================================================================
path_lfm_tokenizer = r'/home/DakeQQ/Downloads/LFM2-350M-Extract'                        # local LFM2-Extract model tokenizer path. Note: the LFM2 only accept pure text as input, not multimodal. URL: https://huggingface.co/LiquidAI/LFM2-350M-Extract / https://huggingface.co/LiquidAI/LFM2-1.2B-Extract
onnx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LFM_Optimized')    # directory containing the 9 exported ONNX graphs

onnx_model_Embed                = os.path.join(onnx_dir, 'LLM_Embed.onnx')              # token id   -> hidden state
onnx_model_Rotary_Mask_Prefill  = os.path.join(onnx_dir, 'Rotary_Prefill.onnx')         # prefill rotary + causal mask
onnx_model_Rotary_Mask_Decode   = os.path.join(onnx_dir, 'Rotary_Decode.onnx')          # decode  rotary
onnx_model_Main                 = os.path.join(onnx_dir, 'LLM_Main.onnx')               # decoder layers + LM head
onnx_model_Greedy               = os.path.join(onnx_dir, 'Greedy_Search.onnx')          # argmax + append
onnx_model_First_Beam           = os.path.join(onnx_dir, 'First_Beam_Search.onnx')      # 1 -> beam_size
onnx_model_Second_Beam          = os.path.join(onnx_dir, 'Second_Beam_Search.onnx')     # prune + re-expand
onnx_model_Penalty              = os.path.join(onnx_dir, 'Apply_Penalty.onnx')          # repetition penalty
onnx_model_Argmax               = os.path.join(onnx_dir, 'Argmax.onnx')                 # bare argmax

# -- Decoding ----------------------------------------------------------------
MAX_SEQ_LEN     = 1024              # Maximum decode length (baked into the rotary/mask buffers).
STOP_TOKEN      = [7]               # <|im_end|>  -> stop.
USE_BEAM_SEARCH = False             # True = beam search; False = greedy/argmax fast path.
BEAM_SIZE       = 3                 # Number of beams.
TOP_K           = 3                 # Top-k candidates per step.
REPEAT_PENALITY = 1.0               # Repetition penalty (0..1; 1.0 = none -> argmax fast path).
PENALITY_RANGE  = 10                # Window of recent tokens to penalize.

# -- Hardware ----------------------------------------------------------------
MAX_THREADS              = 0        # CPU threads (0 = auto).
DEVICE_ID                = 0        # Device index.
ORT_LOG                  = False    # Enable ONNXRuntime logging for debugging. Set False for best performance.
ORT_FP16                 = False    # True for FP16 ORT settings. For CPUs, requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers = []       # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']; [] = CPU.


# -- Prompts (document extraction; LFM2-350M-Extract is an extraction fine-tune) -----
SYSTEM_PROMPT = (
    "Return data as a JSON object with the following schema:\n"
    "{\n"
    '  "ceo": string,\n'
    '  "founded_date": string,\n'
    '  "summary": string,\n'
    '  "industry": string,\n'
    '  "tech_framework": string,\n'
    '  "products": [string]\n'
    "}"
)
TEST_PROMPTS = [
    "DakeQQ was founded on February 16, 2024. This blog focuses on exporting various models to the ONNX format and on ONNXRuntime inference applications. The AI model categories currently covered include, but are not limited to: VAD, ASR, TTS, LLM, OCR, and Embedding. In short, this is a content-rich blog—we warmly welcome you to star, follow, and fork. Thank you for your support!",
    "DakeQQ成立于2024年2月16日。本博客专注于各类模型的ONNX格式导出与ONNXRuntime推理应用，目前涵盖的AI模型类别包括但不限于：VAD、ASR、TTS、LLM、OCR、Embedding等。总而言之，这是一个内容丰富的博客，欢迎大家点赞、关注与分支，多多支持！",
]


# =============================================================================
# ORT helpers  (IOBinding, zero-copy)  -- ported from Qwen_Export.py
# =============================================================================
def bind_ort_in_buf(binding, names, values):
    for name, val in zip(names, values):
        binding.bind_ortvalue_input(name, val)


def bind_ort_out_buf(binding, names, values):
    for name, val in zip(names, values):
        binding.bind_ortvalue_output(name, val)


def bind_ort_out(binding, names, device):
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    return onnxruntime.InferenceSession(
        model_path, sess_options=_session_opts, providers=_providers,
        provider_options=_provider_options, disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


_NP_FROM_ORT = {
    'tensor(float)': np.float32, 'tensor(float16)': np.float16,
    'tensor(uint8)': np.uint8, 'tensor(int8)': np.int8,
    'tensor(int32)': np.int32, 'tensor(int64)': np.int64,
}


def np_dtype_of(meta):
    return _NP_FROM_ORT.get(meta.type, np.float32)


def make_empty_like(meta, device, device_id, batch=1):
    """Zero-filled OrtValue matching a cache input meta, with batch=`batch` and the
    (non-batch) dynamic sequence dim collapsed to length 0."""
    shape = []
    for j, d in enumerate(meta.shape):
        if j == 0:
            shape.append(batch)
        elif isinstance(d, int):
            shape.append(d)
        else:
            shape.append(0)           # dynamic seq/history dim -> empty
    return create_ort_with_shape(tuple(shape), np_dtype_of(meta), device, device_id)


# =============================================================================
# ORT Session / Run options  (full tuning block ported from Qwen_Export.py)
# =============================================================================
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.enable_cpu_mem_arena     = True
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


# =============================================================================
# Execution provider configuration
# =============================================================================
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
    _ort_device_kind = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 ** 3),   # 24GB
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
    _ort_device_kind = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',   # ["default", "high_performance", "minimum_power"]
        'device_filter':              'gpu',                # [gpu, npu, any]
        'disable_metacommands':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'               # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_kind = C.OrtDevice.dml()

else:
    ORT_Accelerate_Providers = ['CPUExecutionProvider']
    provider_options = None
    device_type      = 'cpu'
    _ort_device_kind = C.OrtDevice.cpu()

_ort_device = C.OrtDevice(_ort_device_kind, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device   = 'cpu' if 'dml' in device_type else device_type

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers,
}


# =============================================================================
# Load sessions + derive metadata
# =============================================================================
print('Loading ONNX sessions ...')
ort_session_Embed               = create_session(onnx_model_Embed,               **packed_settings)
ort_session_Rotary_Mask_Prefill = create_session(onnx_model_Rotary_Mask_Prefill, **packed_settings)
ort_session_Rotary_Mask_Decode  = create_session(onnx_model_Rotary_Mask_Decode,  **packed_settings)
ort_session_Main                = create_session(onnx_model_Main,                 **packed_settings)
ort_session_Argmax              = create_session(onnx_model_Argmax,               **packed_settings)
print(f"Usable Providers: {ort_session_Main.get_providers()}")

binding_Embed               = ort_session_Embed.io_binding()
binding_Rotary_Mask_Prefill = ort_session_Rotary_Mask_Prefill.io_binding()
binding_Rotary_Mask_Decode  = ort_session_Rotary_Mask_Decode.io_binding()
binding_Main                = ort_session_Main.io_binding()
binding_Argmax              = ort_session_Argmax.io_binding()

in_name_Embed,               out_name_Embed               = get_in_names(ort_session_Embed)[0],              get_out_names(ort_session_Embed)[0]
in_name_Rotary_Mask_Prefill, out_name_Rotary_Mask_Prefill = get_in_names(ort_session_Rotary_Mask_Prefill),   get_out_names(ort_session_Rotary_Mask_Prefill)
in_name_Rotary_Mask_Decode,  out_name_Rotary_Mask_Decode  = get_in_names(ort_session_Rotary_Mask_Decode)[0], get_out_names(ort_session_Rotary_Mask_Decode)
in_name_Argmax,              out_name_Argmax              = get_in_names(ort_session_Argmax)[0],             get_out_names(ort_session_Argmax)[0]
out_meta_Rotary_Mask_Decode = ort_session_Rotary_Mask_Decode._outputs_meta

in_name_Main   = get_in_names(ort_session_Main)
out_name_Main  = get_out_names(ort_session_Main)
in_meta_Main   = ort_session_Main._inputs_meta
out_meta_Main  = ort_session_Main._outputs_meta

num_cache            = len(out_name_Main) - 1                 # all outputs except logits
num_cache_plus_1     = num_cache + 1                          # logits-input offset for beam graphs
num_cache_plus_2     = num_cache + 2                          # save_id / previous_prob offset
num_cache_plus_3     = num_cache + 3                          # beam_size / topK offset
in_name_Main_cache   = in_name_Main[:num_cache]
out_name_Main_cache  = out_name_Main[:num_cache]
in_name_Main_others  = in_name_Main[num_cache:]               # hidden_states, rotary_cos, rotary_sin, attention_mask
out_name_Main_logits = out_name_Main[num_cache]
hidden_dtype_Main    = np_dtype_of(in_meta_Main[num_cache])  # float32
vocab_size           = out_meta_Main[num_cache].shape[1]

# Empty (length-0) cache buffers for the prefill phase (batch=1), one per cache input.
init_cache_buffers = [make_empty_like(in_meta_Main[i], kv_device, DEVICE_ID, batch=1) for i in range(num_cache)]

# Decoding-strategy validation (mirror both references).
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE
if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
RUN_BEAM_SIZE = BEAM_SIZE if USE_BEAM_SEARCH else 1
USE_PENALTY = (REPEAT_PENALITY != 1.0)

# Decode-head + penalty sessions.
if USE_BEAM_SEARCH:
    ort_session_First_Beam  = create_session(onnx_model_First_Beam,  **packed_settings); binding_First_Beam  = ort_session_First_Beam.io_binding()
    ort_session_Second_Beam = create_session(onnx_model_Second_Beam, **packed_settings); binding_Second_Beam = ort_session_Second_Beam.io_binding()
    in_name_First_Beam,  out_name_First_Beam  = get_in_names(ort_session_First_Beam),  get_out_names(ort_session_First_Beam)
    in_name_Second_Beam, out_name_Second_Beam = get_in_names(ort_session_Second_Beam), get_out_names(ort_session_Second_Beam)

    # Pre-slice beam name lists once (avoid per-step slicing in the decode loop).
    in_name_First_Beam_parts    = in_name_First_Beam[:num_cache_plus_1]
    out_name_First_Beam_parts   = out_name_First_Beam[:num_cache_plus_1]
    out_name_First_Beam_others  = out_name_First_Beam[num_cache_plus_1:]
    in_name_First_Beam_logits   = in_name_First_Beam[num_cache]
    in_name_Second_Beam_parts   = in_name_Second_Beam[:num_cache_plus_1]
    out_name_Second_Beam_parts  = out_name_Second_Beam[:num_cache_plus_1]
    out_name_Second_Beam_others = out_name_Second_Beam[num_cache_plus_1:]
    in_name_Second_Beam_logits  = in_name_Second_Beam[num_cache]
    in_name_Second_Beam_save_id = in_name_Second_Beam[num_cache_plus_1]
    in_name_Second_Beam_prob    = in_name_Second_Beam[num_cache_plus_2]
if USE_PENALTY:
    ort_session_Greedy  = create_session(onnx_model_Greedy,  **packed_settings); binding_Greedy  = ort_session_Greedy.io_binding()
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings); binding_Penalty = ort_session_Penalty.io_binding()
    in_name_Greedy,  out_name_Greedy  = get_in_names(ort_session_Greedy),  get_out_names(ort_session_Greedy)
    in_name_Penalty, out_name_Penalty = get_in_names(ort_session_Penalty), get_out_names(ort_session_Penalty)

tokenizer      = AutoTokenizer.from_pretrained(path_lfm_tokenizer)
STOP_TOKEN_SET = set(STOP_TOKEN)

# Persistent OrtValue buffers (allocated once, reused across prompts/steps).
init_history_len  = create_ort_with_data([0],         np.int64, device_type, DEVICE_ID)
topK_buf          = create_ort_with_data([TOP_K],     np.int64, device_type, DEVICE_ID)
beam_size_buf     = create_ort_with_data([BEAM_SIZE], np.int64, device_type, DEVICE_ID)

attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1), hidden_dtype_Main, device_type, DEVICE_ID)
rotary_cos_buf     = create_ort_with_shape(out_meta_Rotary_Mask_Decode[0].shape, hidden_dtype_Main, device_type, DEVICE_ID)
rotary_sin_buf     = create_ort_with_shape(out_meta_Rotary_Mask_Decode[1].shape, hidden_dtype_Main, device_type, DEVICE_ID)
hidden_states_buf  = create_ort_with_shape((RUN_BEAM_SIZE, 1, in_meta_Main[num_cache].shape[2]), hidden_dtype_Main, device_type, DEVICE_ID)
save_id_buf        = create_ort_with_shape((RUN_BEAM_SIZE, 0), np.int32, device_type, DEVICE_ID)
prefill_logits_buf = create_ort_with_shape((1, vocab_size),             hidden_dtype_Main, device_type, DEVICE_ID)
decode_logits_buf  = create_ort_with_shape((RUN_BEAM_SIZE, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
max_idx_buf        = create_ort_with_shape((1, 1), np.int32, device_type, DEVICE_ID)
if USE_BEAM_SEARCH:
    beam_ids_buf   = create_ort_with_shape((BEAM_SIZE, 1), np.int32,        device_type, DEVICE_ID)
    beam_score_buf = create_ort_with_shape((BEAM_SIZE, 1), hidden_dtype_Main,  device_type, DEVICE_ID)
if USE_PENALTY:
    penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ort_with_data([REPEAT_PENALITY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALITY_RANGE],  np.int64,      device_type, DEVICE_ID)
save_id_numpy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


def generate_iobinding(tokens, stream=False):
    """Full prefill + zero-copy IOBinding decode loop. Returns (ids, prefill_tps, decode_tps)."""
    num_prefill = tokens.shape[-1]
    input_ids   = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    ids_len     = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)

    is_prefill   = True
    t_prefill0   = time.time()
    decode_start = t_prefill0
    prefill_elapsed = 0.0
    num_decode   = 0
    generated    = []

    # -- prefill embed --
    binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
    bind_ort_out(binding_Embed, [out_name_Embed], _ort_device)
    run(ort_session_Embed, binding_Embed)
    hidden_states = binding_Embed.get_outputs()[0]
    binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)          # for decode steps

    # -- prefill rotary + mask --
    bind_ort_in_buf(binding_Rotary_Mask_Prefill, in_name_Rotary_Mask_Prefill, [ids_len, init_history_len])
    bind_ort_out(binding_Rotary_Mask_Prefill, out_name_Rotary_Mask_Prefill, _ort_device)
    run(ort_session_Rotary_Mask_Prefill, binding_Rotary_Mask_Prefill)
    rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Mask_Prefill.get_outputs()

    # -- pre-bind decode rotary (feeds itself the length each step) --
    binding_Rotary_Mask_Decode.bind_ortvalue_input(in_name_Rotary_Mask_Decode, kv_seq_len)
    bind_ort_out_buf(binding_Rotary_Mask_Decode, out_name_Rotary_Mask_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len])

    # -- bind Main: non-cache + empty cache + outputs --
    bind_ort_in_buf(binding_Main, in_name_Main_others, [hidden_states, rotary_cos, rotary_sin, attention_mask])
    bind_ort_in_buf(binding_Main, in_name_Main_cache, init_cache_buffers)
    bind_ort_out(binding_Main, out_name_Main_cache, _ort_device)
    binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

    # -- bind decode head / penalty to prefill logits --
    if USE_PENALTY:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
        binding_Penalty.bind_ortvalue_output(out_name_Penalty[0], prefill_logits_buf)
        bind_ort_in_buf(binding_Penalty, in_name_Penalty[2:], [penalty_value, penalty_range])
    if USE_BEAM_SEARCH:
        binding_First_Beam.bind_ortvalue_input(in_name_First_Beam_logits, prefill_logits_buf)
        bind_ort_in_buf(binding_First_Beam, in_name_First_Beam[num_cache_plus_1:num_cache_plus_3], [save_id_buf, beam_size_buf])
        bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam[num_cache_plus_3:], [beam_size_buf, topK_buf])
    elif USE_PENALTY:
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], prefill_logits_buf)
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)
        binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
    else:
        binding_Argmax.bind_ortvalue_input(in_name_Argmax, prefill_logits_buf)
        binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)

    save_id = save_id_buf
    generate_limit = MAX_SEQ_LEN - num_prefill
    while num_decode < generate_limit:
        run(ort_session_Main, binding_Main)
        outputs_Main = binding_Main.get_outputs()

        if USE_PENALTY and num_decode >= PENALITY_RANGE:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
            run(ort_session_Penalty, binding_Penalty)

        if USE_BEAM_SEARCH:
            if is_prefill:
                bind_ort_in_buf(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
                bind_ort_out(binding_First_Beam, out_name_First_Beam_parts, _ort_device)
                bind_ort_out_buf(binding_First_Beam, out_name_First_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
                run(ort_session_First_Beam, binding_First_Beam)
                outputs_beam = binding_First_Beam.get_outputs()
            else:
                bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
                bind_ort_out(binding_Second_Beam, out_name_Second_Beam_parts, _ort_device)
                if num_decode < 2:
                    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_prob, beam_score_buf)
                    bind_ort_out_buf(binding_Second_Beam, out_name_Second_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
                run(ort_session_Second_Beam, binding_Second_Beam)
                outputs_beam = binding_Second_Beam.get_outputs()
            max_logits_idx = max_idx_buf.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN_SET:
                break
            save_id = outputs_beam[num_cache]
            bind_ort_in_buf(binding_Main, in_name_Main_cache, outputs_beam)
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_save_id, save_id)
        else:
            if USE_PENALTY:
                binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device)
                run(ort_session_Greedy, binding_Greedy)
                save_id = binding_Greedy.get_outputs()[1]
            else:
                run(ort_session_Argmax, binding_Argmax)
            max_logits_idx = max_idx_buf.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN_SET:
                break
            if USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
            else:
                save_id_numpy[num_decode] = max_logits_idx
            generated.append(max_logits_idx)
            bind_ort_in_buf(binding_Main, in_name_Main_cache, outputs_Main)
            if stream:
                print(tokenizer.decode([max_logits_idx], skip_special_tokens=False), end="", flush=True)

        # re-request fresh cache outputs (ORT allocates new ones each step)
        bind_ort_out(binding_Main, out_name_Main_cache, _ort_device)

        if is_prefill:
            bind_ort_in_buf(binding_Main, in_name_Main_others,
                            [hidden_states_buf, rotary_cos_buf, rotary_sin_buf, attention_mask_buf])
            binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)
            binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)
            if USE_PENALTY:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
                binding_Penalty.bind_ortvalue_output(out_name_Penalty[0], decode_logits_buf)
            if USE_BEAM_SEARCH:
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_logits, decode_logits_buf)
                binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
            elif USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
            else:
                binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)
            is_prefill = False
            decode_start = time.time()
            prefill_elapsed = decode_start - t_prefill0

        run(ort_session_Embed, binding_Embed)
        run(ort_session_Rotary_Mask_Decode, binding_Rotary_Mask_Decode)
        num_decode += 1

    decode_elapsed = time.time() - decode_start if num_decode > 0 else 0.0
    if USE_BEAM_SEARCH:
        generated = save_id.numpy().flat[:num_decode].tolist() if num_decode > 0 else []
    prefill_tps = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
    decode_tps  = num_decode / decode_elapsed if decode_elapsed > 0 else 0.0
    return generated, prefill_tps, decode_tps


def build_prompt_ids(user_text):
    # Inlined chat template (see chat_template.jinja); tokenizer is used for encode only.
    text = (
        "<|startoftext|><|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + user_text + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return np.array(ids, dtype=np.int32)[None, :]


# =============================================================================
# Dict-feed reference path (no IOBinding) — used only for validation.
# =============================================================================
def _empty_shape(meta):
    return [1 if j == 0 else (d if isinstance(d, int) else 0) for j, d in enumerate(meta.shape)]


_empty_caches_np = [np.zeros(_empty_shape(in_meta_Main[i]), dtype=np_dtype_of(in_meta_Main[i]))
                    for i in range(num_cache)]


def _main_dictfeed(hidden_np, rc, rs, am, caches):
    feed = {n: c for n, c in zip(in_name_Main_cache, caches)}
    feed[in_name_Main_others[0]] = hidden_np
    feed[in_name_Main_others[1]] = rc
    feed[in_name_Main_others[2]] = rs
    feed[in_name_Main_others[3]] = am
    outs = ort_session_Main.run(out_name_Main, feed)
    return outs[num_cache], outs[:num_cache]


def generate_dictfeed(tokens, max_new=64):
    hidden = ort_session_Embed.run([out_name_Embed], {in_name_Embed: tokens})[0]
    n = tokens.shape[-1]
    rc, rs, am, kv = ort_session_Rotary_Mask_Prefill.run(out_name_Rotary_Mask_Prefill, {in_name_Rotary_Mask_Prefill[0]: np.array([n], np.int64),
                                                                                        in_name_Rotary_Mask_Prefill[1]: np.array([0], np.int64)})
    logits, caches = _main_dictfeed(hidden, rc, rs, am, _empty_caches_np)
    ids = []
    mask_zero = np.zeros((1, 1, 1, 1, 1), dtype=hidden_dtype_Main)
    for _ in range(max_new):
        tok = int(np.argmax(logits[-1]))
        if tok in STOP_TOKEN_SET:
            break
        ids.append(tok)
        hidden = ort_session_Embed.run([out_name_Embed], {in_name_Embed: np.array([[tok]], np.int32)})[0]
        rc, rs, kv = ort_session_Rotary_Mask_Decode.run(out_name_Rotary_Mask_Decode, {in_name_Rotary_Mask_Decode: kv})
        logits, caches = _main_dictfeed(hidden, rc, rs, mask_zero, caches)
    return ids


# =============================================================================
# Production run  (IOBinding, streaming)
# =============================================================================
print("\n" + "=" * 70)
print("LFM2-350M-Extract — ONNXRuntime IOBinding decode")
print("=" * 70)
for prompt_idx, prompt_text in enumerate(TEST_PROMPTS):
    # Re-init the per-sequence decode buffers so each prompt starts from an empty
    # history (save_id grows during beam/penalty decoding and must be reset).
    save_id_buf   = create_ort_with_shape((RUN_BEAM_SIZE, 0), np.int32, device_type, DEVICE_ID)
    save_id_numpy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)

    prod_tokens = build_prompt_ids(prompt_text)
    print(f"\n[{prompt_idx + 1}/{len(TEST_PROMPTS)}] Prompt ({prod_tokens.shape[-1]} tokens):\n{prompt_text}\n\nExtracting:\n")
    gen_ids, prefill_tps, decode_tps = generate_iobinding(prod_tokens, stream=True)
    print("\n\n" + "-" * 70)
    if USE_BEAM_SEARCH:
        print("Decoded:", tokenizer.decode(gen_ids, skip_special_tokens=True))
        print("-" * 70)
    print(f"Prefill: {prefill_tps:8.2f} tok/s   |   Decode: {decode_tps:8.2f} tok/s")
