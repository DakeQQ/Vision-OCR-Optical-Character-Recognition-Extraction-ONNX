import os
import time

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from PIL import Image, ImageColor, ImageDraw, ImageFont
from transformers import AutoConfig, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(SCRIPT_DIR, 'Falcon_Perception_Optimized')   # Folder with the exported / optimized ONNX modules.

download_path                  = r'/home/DakeQQ/Downloads/Falcon-Perception-300M'   # Folder of the downloaded Falcon Perception 300M project (for tokenizer + token ids).

onnx_model_Embed               = os.path.join(EXPORT_DIR, 'LLM_Embed.onnx')
onnx_model_Vision              = os.path.join(EXPORT_DIR, 'LLM_Vision.onnx')
onnx_model_Prefill_Mask        = os.path.join(EXPORT_DIR, 'Prefill_Mask.onnx')
onnx_model_Main                = os.path.join(EXPORT_DIR, 'LLM_Main.onnx')
onnx_model_Argmax              = os.path.join(EXPORT_DIR, 'Argmax.onnx')
onnx_model_Greedy              = os.path.join(EXPORT_DIR, 'Greedy_Search.onnx')
onnx_model_First_Beam          = os.path.join(EXPORT_DIR, 'First_Beam_Search.onnx')
onnx_model_Second_Beam         = os.path.join(EXPORT_DIR, 'Second_Beam_Search.onnx')
onnx_model_Penalty             = os.path.join(EXPORT_DIR, 'Apply_Penalty.onnx')
onnx_model_KV_Slice            = os.path.join(EXPORT_DIR, 'KV_Slice.onnx')
onnx_model_Coord_Encoder       = os.path.join(EXPORT_DIR, 'Coord_Encoder.onnx')
onnx_model_Size_Encoder        = os.path.join(EXPORT_DIR, 'Size_Encoder.onnx')


# Test input
TEST_IMAGE               = [os.path.join(SCRIPT_DIR, 'psyduck.jpg')]   # One image for the static detection path (anchored to the script dir, cwd-independent).
TEST_QUERY               = 'psyduck'                # Detection query. A RUNTIME input: change it and re-run (no re-export needed).

# Model / geometry config (keep the same values as the exported model)
MAX_SEQ_LEN              = 8192                     # Max context length. Must match the exported model.

# Detection runtime (coord dedup threshold + max attempts are baked into the Coord_Encoder graph)
MAX_NEW_TOKENS           = 1024                     # Max decode steps.
SAVE_VISUALIZATION       = True                     # Draw YOLO-style boxes on the input image and save the result.
VISUALIZATION_PATH       = os.path.join(SCRIPT_DIR, 'detection_output.jpg')   # Output path for the annotated image.

# Decoding strategy. The coord/size VALUES are decoded greedily inside the Coord/Size encoder
# graphs, so every strategy below only chooses the per-step vocab token (COORD vs SIZE vs text vs
# stop). Greedy (Argmax) and greedy + repetition penalty (Greedy_Search + Apply_Penalty) are the
# primary paths; beam search (First/Second_Beam_Search) searches over the vocab token while the
# per-beam detection state is reordered each step by the parent-beam map recovered from save_id.
USE_BEAM_SEARCH          = False                    # Beam search over the vocab token, or greedy search.
REPEAT_PENALTY           = 1.0                      # Repetition penalty on the vocab logits. 0.0 ~ 1.0; 1.0 = off.
PENALTY_RANGE            = 10                       # Recent decode-token window the penalty looks back over.
TOP_K                    = 3                        # Top-K per beam for beam search. Clamped up to BEAM_SIZE.
BEAM_SIZE                = 3                        # Beam size for beam search. Must be <= MAX_BEAM_SIZE.
MAX_BEAM_SIZE            = 10                        # Max beam size baked into the exported beam graphs.


# Runtime config
ORT_LOG                  = False
ORT_FP16                 = False
ORT_Accelerate_Providers = []                       # e.g. ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS              = 0
DEVICE_ID                = 0


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    return onnxruntime.InferenceSession(
        model_path, sess_options=_session_opts, providers=_providers,
        provider_options=_provider_options, disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


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


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def np_dtype_from_ort(type_str):
    """Map an ONNX Runtime tensor type string to a numpy dtype."""
    if 'float16' in type_str:
        return np.float16
    if 'float' in type_str:
        return np.float32
    if 'int32' in type_str:
        return np.int32
    if 'uint8' in type_str:
        return np.uint8
    if 'int8' in type_str:
        return np.int8
    if 'int64' in type_str:
        return np.int64
    return np.float32


def make_empty_kv(meta, device, device_id):
    """Build a zero-history KV input OrtValue from an LLM_Main input meta.

    The batch dim (0) is set to 1 and the (dynamic, non-batch) history dim is set
    to 0; every other dim is taken from the static meta. This handles key/value and
    any quantized scale/bias side tensors uniformly without hard-coding the layout.
    """
    shape = [d if isinstance(d, int) else 1 for d in meta.shape]
    for d in range(1, len(meta.shape)):
        if not isinstance(meta.shape[d], int):
            shape[d] = 0
            break
    return create_ort_with_shape(tuple(shape), np_dtype_from_ort(meta.type), device, device_id)


def load_image_for_inference(image_path, target_h, target_w, dim5):
    """Load an image for the LLM_Vision graph as a uint8 tensor in the expected layout.

    target_h / target_w come from the vision graph's input meta: when they are ints (a static
    H/W input) the image is resized to that fixed shape on the host; when they are None (the
    graph declares a dynamic H/W input) the image is loaded at its NATIVE resolution and the
    in-graph Resize normalizes it to the fixed export geometry, so no host-side resize is done."""
    resampling = getattr(getattr(Image, 'Resampling', Image), 'BICUBIC')
    with Image.open(image_path) as image:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if target_h is not None and target_w is not None and image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), resampling)
        pixel_values = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)[None]   # (1, 3, H, W)
    if dim5:
        pixel_values = pixel_values[:, None]                                        # (1, 1, 3, H, W)
    return np.ascontiguousarray(pixel_values)


# Static detection prompt; the query is a runtime input (re-tokenized each run, no re-export).
DETECTION_PROMPT_TEMPLATE = "<|image|>Detect these expressions in the image:<|start_of_query|>{query}<|DET|>"


def build_detection_token_list(tokenizer, config, query, num_patches):
    """Tokenize the detection prompt into the token-id list plus the img_id patch span
    [patch_start, patch_end). Mirrors the exporter so the runtime tail (img_end +
    instruction + query + <|DET|>) is embedded for the CURRENT query with no re-export."""
    img_reg_ids = [
        config.image_reg_1_token_id, config.image_reg_2_token_id,
        config.image_reg_3_token_id, config.image_reg_4_token_id,
    ]
    image_token = tokenizer.convert_ids_to_tokens(config.img_id)
    prompt = DETECTION_PROMPT_TEMPLATE.format(query=query)
    prompt_chunks = [tokenizer.encode(chunk) for chunk in prompt.split(image_token)]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    offset = 0
    head_text_ids = []
    if prompt_chunks and prompt_chunks[0] and bos_id is not None and prompt_chunks[0][0] == bos_id:
        offset = 1
        head_text_ids.append(prompt_chunks[0][0])
    image_block = [config.image_cls_token_id, *img_reg_ids,
                   *([config.img_id] * num_patches), config.img_end_id]
    before = prompt_chunks[0][offset:]
    after = prompt_chunks[1] if len(prompt_chunks) > 1 else []
    token_list = head_text_ids + before + image_block + after
    patch_positions = [i for i, t in enumerate(token_list) if t == config.img_id]
    patch_start = patch_positions[0]
    patch_end = patch_positions[-1] + 1
    return token_list, patch_start, patch_end


# ── YOLO-style detection visualization ────────────────────────────────────
VIS_PALETTE = (
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
    "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
    "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF",
    "#FF95C8", "#FF37C7",
)


def _vis_text_color(rgb):
    """Pick black or white label text for contrast against a box color (YOLO rule)."""
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


# Palette-derived constants precomputed once: RGB tuples (handed straight to PIL, no per-draw
# hex re-parse) and their contrast text colors (folds ImageColor.getrgb + _vis_text_color out
# of the per-detection draw loop).
VIS_PALETTE_RGB = tuple(ImageColor.getrgb(color) for color in VIS_PALETTE)
VIS_TEXT_COLOR = tuple(_vis_text_color(rgb) for rgb in VIS_PALETTE_RGB)


def visualize_detections(image_path, detections, query, output_path):
    """Draw YOLO-style boxes + labels for `detections` (normalized center/size) on the
    original image, save to `output_path`, and return it."""
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    draw = ImageDraw.Draw(image)
    width, height = image.size
    width_f, height_f = float(width), float(height)
    min_dim = min(width, height)
    line_width = max(2, round(min_dim / 300))
    pad = max(2, line_width)
    pad2 = pad + pad
    palette_len = len(VIS_PALETTE_RGB)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, round(min_dim / 40)))
    except OSError:
        font = ImageFont.load_default()

    # Resolve the text-measuring path once (textbbox on modern Pillow, textsize on legacy).
    if hasattr(draw, "textbbox"):
        def _measure(text):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
    else:
        def _measure(text):
            return draw.textsize(text, font=font)

    for idx, det in enumerate(detections):
        pidx = idx % palette_len
        color = VIS_PALETTE_RGB[pidx]
        xy, hw = det["xy"], det["hw"]
        center_x = xy["x"] * width
        center_y = xy["y"] * height
        half_w = hw["w"] * width / 2
        half_h = hw["h"] * height / 2
        x0 = max(0.0, min(width_f, center_x - half_w))
        y0 = max(0.0, min(height_f, center_y - half_h))
        x1 = max(0.0, min(width_f, center_x + half_w))
        y1 = max(0.0, min(height_f, center_y + half_h))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)

        label = f"{query} {idx}"
        text_w, text_h = _measure(label)
        tag_h = text_h + pad2
        tag_y0 = y0 - tag_h if y0 - tag_h >= 0 else y0
        tag_x1 = min(width_f, x0 + text_w + pad2)
        draw.rectangle((x0, tag_y0, tag_x1, tag_y0 + tag_h), fill=color)
        draw.text((x0 + pad, tag_y0 + pad), label, fill=VIS_TEXT_COLOR[pidx], font=font)

    image.save(output_path)
    return output_path

# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS (LightOn-style)
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
for opt in (session_opts, run_options):
    opt.log_severity_level = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4
session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
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
        'device_type': 'CPU', 'precision': 'ACCURACY',
        'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8, 'num_streams': 1,
        'enable_opencl_throttling': False, 'enable_qdq_optimizer': False, 'disable_dynamic_shapes': False,
    }]
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id': DEVICE_ID, 'gpu_mem_limit': 24 * (1024 ** 3),
        'arena_extend_strategy': 'kNextPowerOfTwo', 'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'sdpa_kernel': '2', 'use_tf32': '1', 'fuse_conv_bias': '0',
        'cudnn_conv_use_max_workspace': '1', 'do_copy_in_default_stream': '0',
        'enable_cuda_graph': '0', 'prefer_nhwc': '0', 'use_ep_level_unified_stream': '0',
    }]
    device_type = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id': DEVICE_ID, 'performance_preference': 'high_performance', 'device_filter': 'gpu',
        'disable_metacommands': 'false', 'enable_graph_capture': 'false', 'enable_graph_serialization': 'false',
    }]
    device_type = 'dml'
    _ort_device_type = C.OrtDevice.dml()
else:
    provider_options = None
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type

packed_settings = {
    "_session_opts": session_opts,
    "_providers": ORT_Accelerate_Providers,
    "_provider_options": provider_options,
    "_disabled_optimizers": disabled_optimizers,
}


# ══════════════════════════════════════════════════════════════════════════════
# DECODING STRATEGY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
# Beam search here searches over the vocab token only (the coord/size VALUES stay greedily decoded
# inside the encoder graphs). The First/Second beam graphs reorder the KV cache + save_id; the
# host then recovers the parent-beam map from save_id each step to (a) reorder the per-beam coord
# history / pending box / decoded detections and (b) re-embed each child from its PARENT beam's
# coord/size logits. With BEAM_SIZE == 1 the beam graphs are skipped entirely.
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE
if USE_BEAM_SEARCH and BEAM_SIZE > MAX_BEAM_SIZE:
    print(f"[Warning] BEAM_SIZE ({BEAM_SIZE}) > MAX_BEAM_SIZE ({MAX_BEAM_SIZE}); clamping to {MAX_BEAM_SIZE}.")
    BEAM_SIZE = MAX_BEAM_SIZE
if USE_BEAM_SEARCH and (TOP_K < 2 or BEAM_SIZE < 2):
    print("[Warning] Beam search needs TOP_K >= 2 and BEAM_SIZE >= 2; falling back to greedy search.")
    USE_BEAM_SEARCH = False
if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1
USE_PENALTY = (REPEAT_PENALTY != 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
ort_session_Vision = create_session(onnx_model_Vision, **packed_settings)
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
ort_session_Prefill_Mask = create_session(onnx_model_Prefill_Mask, **packed_settings)
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
ort_session_Coord_Encoder = create_session(onnx_model_Coord_Encoder, **packed_settings)
ort_session_Size_Encoder = create_session(onnx_model_Size_Encoder, **packed_settings)
# Repetition penalty pulls in the Greedy_Search head (it maintains the generated-token history the
# penalty looks back over) plus the Apply_Penalty graph. Plain greedy keeps using Argmax only.
# Beam search pulls in the First/Second beam graphs (which also reuse the penalty head).
if USE_PENALTY:
    ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
if USE_PENALTY or USE_BEAM_SEARCH:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
if USE_BEAM_SEARCH:
    ort_session_First_Beam = create_session(onnx_model_First_Beam, **packed_settings)
    ort_session_Second_Beam = create_session(onnx_model_Second_Beam, **packed_settings)

print(f"Usable Providers: {ort_session_Main.get_providers()}")

binding_Vision = ort_session_Vision.io_binding()
binding_Embed = ort_session_Embed.io_binding()
binding_Prefill_Mask = ort_session_Prefill_Mask.io_binding()
binding_Main = ort_session_Main.io_binding()
binding_Argmax = ort_session_Argmax.io_binding()
binding_Coord_Encoder = ort_session_Coord_Encoder.io_binding()
binding_Size_Encoder = ort_session_Size_Encoder.io_binding()
if USE_PENALTY:
    binding_Greedy = ort_session_Greedy.io_binding()
if USE_PENALTY or USE_BEAM_SEARCH:
    binding_Penalty = ort_session_Penalty.io_binding()
if USE_BEAM_SEARCH:
    binding_First_Beam = ort_session_First_Beam.io_binding()
    binding_Second_Beam = ort_session_Second_Beam.io_binding()



# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS (LightOn-style naming)
# ══════════════════════════════════════════════════════════════════════════════
in_name_Vision = get_in_names(ort_session_Vision)[0]
out_name_Vision = get_out_names(ort_session_Vision)[0]
# Image tensor rank read from the Vision graph (4 -> [batch, 3, H, W]; 5 -> [batch, 1, 3, H, W]).
INPUT_IMAGE_DIM = len(ort_session_Vision.get_inputs()[0].shape)

in_name_Main = get_in_names(ort_session_Main)
out_name_Main = get_out_names(ort_session_Main)
in_meta_Main = ort_session_Main.get_inputs()
out_meta_Main = ort_session_Main.get_outputs()

# LLM_Main emits [ ...KV..., logits, coord_logits, size_logits ]; all but the last three are KV.
num_kv_io = len(out_name_Main) - 3
in_name_Main_kv = in_name_Main[:num_kv_io]
out_name_Main_kv = out_name_Main[:num_kv_io]
in_name_Main_others = in_name_Main[num_kv_io:]               # hidden_states, cache_position, attention_mask
out_name_Main_logits = out_name_Main[num_kv_io]
out_name_Main_coord_logits = out_name_Main[num_kv_io + 1]
out_name_Main_size_logits = out_name_Main[num_kv_io + 2]

kv_dtype_str = in_meta_Main[0].type
kv_dtype_Main = np_dtype_from_ort(kv_dtype_str)
is_quantized_rt = kv_dtype_str not in ('tensor(float)', 'tensor(float16)')
hidden_dtype_Main = np_dtype_from_ort(in_meta_Main[num_kv_io].type)
# Host buffer dtypes are read from the IO meta, not hard-coded, so they track the export.
cache_position_dtype_Main = np_dtype_from_ort(in_meta_Main[num_kv_io + 1].type)
mask_dtype_Main = np_dtype_from_ort(in_meta_Main[num_kv_io + 2].type)
logits_dtype_Main = np_dtype_from_ort(out_meta_Main[num_kv_io].type)
coord_logits_dtype_Main = np_dtype_from_ort(out_meta_Main[num_kv_io + 1].type)
size_logits_dtype_Main = np_dtype_from_ort(out_meta_Main[num_kv_io + 2].type)
coord_num_bins = out_meta_Main[num_kv_io + 1].shape[2]
size_num_bins = out_meta_Main[num_kv_io + 2].shape[2]
vocab_size = out_meta_Main[num_kv_io].shape[1]
hidden_size = in_meta_Main[num_kv_io].shape[2]

in_name_Embed = get_in_names(ort_session_Embed)[0]
out_name_Embed = get_out_names(ort_session_Embed)[0]
in_name_Argmax = get_in_names(ort_session_Argmax)[0]
out_name_Argmax = get_out_names(ort_session_Argmax)[0]
if USE_PENALTY:
    in_name_Greedy = get_in_names(ort_session_Greedy)            # ['logits', 'save_id_in']
    out_name_Greedy = get_out_names(ort_session_Greedy)          # ['max_logits_idx', 'save_id_out']
if USE_PENALTY or USE_BEAM_SEARCH:
    in_name_Penalty = get_in_names(ort_session_Penalty)          # ['logits_in', 'save_id_in', 'penalty_value', 'penalty_range']
    out_name_Penalty = get_out_names(ort_session_Penalty)[0]     # 'logits_out'
if USE_BEAM_SEARCH:
    # Beam graph I/O is partitioned as [KV*num_kv_io, <trailing>]; the beam graphs consume only
    # logits (not coord/size). First in: [KV, logits, save_id_in, beam_size]; Second in adds
    # previous_prob + topK. Both out: [KV, save_id_out, top_beam_prob, top_beam_indices, max_idx].
    in_name_First_Beam = get_in_names(ort_session_First_Beam)
    out_name_First_Beam = get_out_names(ort_session_First_Beam)
    in_name_Second_Beam = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam = get_out_names(ort_session_Second_Beam)
    beam_param_dtype = np_dtype_from_ort(ort_session_Second_Beam.get_inputs()[num_kv_io + 3].type)
    beam_prob_dtype = np_dtype_from_ort(ort_session_First_Beam.get_outputs()[num_kv_io + 1].type)
in_names_Coord_Enc = get_in_names(ort_session_Coord_Encoder)     # ['coord_logits', 'existing_coords']
out_names_Coord_Enc = get_out_names(ort_session_Coord_Encoder)   # ['coord_embed', 'coord_xy']
in_names_Size_Enc = get_in_names(ort_session_Size_Encoder)       # ['size_logits']
out_names_Size_Enc = get_out_names(ort_session_Size_Encoder)     # ['size_embed', 'size_hw']

in_name_Prefill_Mask = get_in_names(ort_session_Prefill_Mask)
out_name_Prefill_Mask = get_out_names(ort_session_Prefill_Mask)

# Remaining host feed dtypes also read from the graph IO meta (not hard-coded).
vision_in_dtype = np_dtype_from_ort(ort_session_Vision.get_inputs()[0].type)
# Vision input H/W are the last two axes ([B,1,3,H,W] or [B,3,H,W]). Read them from the ORT
# meta: an int axis is a STATIC input (host pre-resizes the image to it on load); a str symbol
# is a DYNAMIC input (host loads the native image and the in-graph Resize normalizes it to the
# fixed export geometry). None on either axis -> skip the host resize. If either axis is dynamic, load
# native (can't pre-resize to a fixed shape).
vision_in_shape = ort_session_Vision.get_inputs()[0].shape
vision_static_h = vision_in_shape[-2] if isinstance(vision_in_shape[-2], int) else None
vision_static_w = vision_in_shape[-1] if isinstance(vision_in_shape[-1], int) else None
token_id_dtype = np_dtype_from_ort(ort_session_Argmax.get_outputs()[0].type)
embed_in_dtype = np_dtype_from_ort(ort_session_Embed.get_inputs()[0].type)
coord_xy_dtype = np_dtype_from_ort(ort_session_Coord_Encoder.get_outputs()[1].type)
size_hw_dtype = np_dtype_from_ort(ort_session_Size_Encoder.get_outputs()[1].type)
existing_coords_dtype = np_dtype_from_ort(ort_session_Coord_Encoder.get_inputs()[1].type)
prefill_len_dtype = np_dtype_from_ort(ort_session_Prefill_Mask.get_inputs()[0].type)
prefill_hist_dtype = np_dtype_from_ort(ort_session_Prefill_Mask.get_inputs()[1].type)


# ══════════════════════════════════════════════════════════════════════════════
# FALCON DETECTION CONFIG (token ids from the local model)
# ══════════════════════════════════════════════════════════════════════════════
try:
    _falcon_cfg = AutoConfig.from_pretrained(download_path, trust_remote_code=True)
except Exception:
    _falcon_cfg = None
_tokenizer = AutoTokenizer.from_pretrained(download_path, local_files_only=True, trust_remote_code=True)
COORD_TOKEN_ID = _falcon_cfg.coord_token_id if _falcon_cfg is not None else 240
SIZE_TOKEN_ID = _falcon_cfg.size_token_id if _falcon_cfg is not None else 241
EOS_ID = _falcon_cfg.eos_id if _falcon_cfg is not None else 11
END_OF_QUERY_ID = _tokenizer.convert_tokens_to_ids("<|end_of_query|>")
STOP_TOKEN_SET = {EOS_ID, END_OF_QUERY_ID}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED ORTVALUE BUFFERS (allocated once, reused every decode step)
# ══════════════════════════════════════════════════════════════════════════════
hidden_states_buf = create_ort_with_shape((1, 1, hidden_size), hidden_dtype_Main, device_type, DEVICE_ID)
# Per-step KV-cache position of the new decode token (num_prefill + step). The rotary cos/sin
# tables are now fused into LLM_Main and gathered by this; host-tracked, so there is no
# device-side counter and no read-after-write hazard. Prefill uses a full arange (built below).
cache_position_buf = create_ort_with_shape((1,), cache_position_dtype_Main, device_type, DEVICE_ID)
decode_mask_buf = create_ort_with_shape((1, 1, 1, 1), mask_dtype_Main, device_type, DEVICE_ID)   # scalar-zero, broadcasts over kv
logits_buf = create_ort_with_shape((1, vocab_size), logits_dtype_Main, device_type, DEVICE_ID)
# Fused coord / size detection-head logits, written by LLM_Main every step. The host reads
# whichever one the argmax token selects (coord vs size); on text tokens neither is read.
coord_logits_buf = create_ort_with_shape((1, 2, coord_num_bins), coord_logits_dtype_Main, device_type, DEVICE_ID)
size_logits_buf = create_ort_with_shape((1, 2, size_num_bins), size_logits_dtype_Main, device_type, DEVICE_ID)
max_idx_buf = create_ort_with_shape((1, 1), token_id_dtype, device_type, DEVICE_ID)
# Decoded coord (x, y) / size (h, w) — the only values the decode-encoder graphs cross back to
# the host (they write the replacement embedding into hidden_states_buf on device).
coord_xy_buf = create_ort_with_shape((1, 2), coord_xy_dtype, device_type, DEVICE_ID)
size_hw_buf = create_ort_with_shape((1, 2), size_hw_dtype, device_type, DEVICE_ID)
# Fixed-capacity prior-coord history for the in-graph coord dedup, pre-allocated once. Slots start
# as the far-away (-1, -1) sentinel (never matches a real coord, keeps the compare axis non-empty);
# each decoded coord overwrites the next slot, refreshed via update_inplace.
existing_coords_host = np.full((1, MAX_NEW_TOKENS, 2), -1.0, dtype=existing_coords_dtype)
existing_coords_buf = create_ort_from_numpy(existing_coords_host, device_type, DEVICE_ID)
# Repetition-penalty state: Greedy_Search appends each selected vocab token to save_id, and
# Apply_Penalty rescales the logits of the tokens in the trailing PENALTY_RANGE window. Only
# allocated when the penalty is enabled (plain greedy keeps no token history).
if USE_PENALTY or USE_BEAM_SEARCH:
    save_id = create_ort_with_shape((1, 0), token_id_dtype, device_type, DEVICE_ID)
    penalty_value_dtype = np_dtype_from_ort(ort_session_Penalty.get_inputs()[2].type)
    penalty_range_dtype = np_dtype_from_ort(ort_session_Penalty.get_inputs()[3].type)
    penalty_value_ort = create_ort_with_data([REPEAT_PENALTY], penalty_value_dtype, device_type, DEVICE_ID)
    penalty_range_ort = create_ort_with_data([PENALTY_RANGE], penalty_range_dtype, device_type, DEVICE_ID)
# Beam-batched buffers (BEAM_SIZE rows). Prefill stays batch 1 (logits_buf / coord_logits_buf /
# size_logits_buf), so beam decode gets its own (BEAM, ...) decode buffers, plus the per-beam
# parent-gathered coord/size logits, the three feedback embeddings (coord / size / text) the host
# selects between, and the grown save_id / beam-prob state the beam graphs maintain.
if USE_BEAM_SEARCH:
    decode_logits_buf = create_ort_with_shape((BEAM_SIZE, vocab_size), logits_dtype_Main, device_type, DEVICE_ID)
    decode_coord_buf = create_ort_with_shape((BEAM_SIZE, 2, coord_num_bins), coord_logits_dtype_Main, device_type, DEVICE_ID)
    decode_size_buf = create_ort_with_shape((BEAM_SIZE, 2, size_num_bins), size_logits_dtype_Main, device_type, DEVICE_ID)
    beam_hidden_buf = create_ort_with_shape((BEAM_SIZE, 1, hidden_size), hidden_dtype_Main, device_type, DEVICE_ID)
    beam_existing_coords_buf = create_ort_from_numpy(
        np.full((BEAM_SIZE, MAX_NEW_TOKENS, 2), -1.0, dtype=existing_coords_dtype), device_type, DEVICE_ID)
    beam_coord_in_buf = create_ort_with_shape((BEAM_SIZE, 2, coord_num_bins), coord_logits_dtype_Main, device_type, DEVICE_ID)
    beam_size_in_buf = create_ort_with_shape((BEAM_SIZE, 2, size_num_bins), size_logits_dtype_Main, device_type, DEVICE_ID)
    beam_coord_embed_buf = create_ort_with_shape((BEAM_SIZE, 1, hidden_size), hidden_dtype_Main, device_type, DEVICE_ID)
    beam_size_embed_buf = create_ort_with_shape((BEAM_SIZE, 1, hidden_size), hidden_dtype_Main, device_type, DEVICE_ID)
    beam_text_embed_buf = create_ort_with_shape((BEAM_SIZE, 1, hidden_size), hidden_dtype_Main, device_type, DEVICE_ID)
    beam_coord_xy_buf = create_ort_with_shape((BEAM_SIZE, 2), coord_xy_dtype, device_type, DEVICE_ID)
    beam_size_hw_buf = create_ort_with_shape((BEAM_SIZE, 2), size_hw_dtype, device_type, DEVICE_ID)
    beam_ids_buf = create_ort_with_shape((BEAM_SIZE, 1), token_id_dtype, device_type, DEVICE_ID)
    save_id_first_buf = create_ort_with_shape((BEAM_SIZE, 0), token_id_dtype, device_type, DEVICE_ID)
    beam_prob_buf = create_ort_with_shape((BEAM_SIZE, 1), beam_prob_dtype, device_type, DEVICE_ID)
    beam_size_ort = create_ort_with_data([BEAM_SIZE], beam_param_dtype, device_type, DEVICE_ID)
    topK_ort = create_ort_with_data([TOP_K], beam_param_dtype, device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# PREFILL PHASE
# ══════════════════════════════════════════════════════════════════════════════
# Static vision input -> resize on load to the graph's fixed H/W (meta ints); dynamic vision
# input -> load the native image (vision_static_h/w are None, so no resize) and let LLM_Vision
# resize to the fixed export geometry in-graph.
pixel_values = load_image_for_inference(TEST_IMAGE[0], vision_static_h, vision_static_w, INPUT_IMAGE_DIM == 5)

prefill_start_time = time.time()

# 1. Vision: build the image-dependent prefill prefix (head tokens + projected image patches).
pixel_values_ort = create_ort_from_numpy(pixel_values.astype(vision_in_dtype, copy=False), device_type, DEVICE_ID)
binding_Vision.bind_ortvalue_input(in_name_Vision, pixel_values_ort)
binding_Vision._iobinding.bind_output(out_name_Vision, _ort_device_type)
run(ort_session_Vision, binding_Vision)
vision_hidden = binding_Vision.get_outputs()[0].numpy().astype(hidden_dtype_Main, copy=False)

# 1b. Tail = prompt tokens after the image patch span (img_end + instruction + query + <|DET|>) for
#     the current TEST_QUERY, embedded via LLM_Embed (makes the query a runtime input, no re-export).
#     The patch tokens are produced by the Vision graph (their count is baked at export) and the tail
#     is INDEPENDENT of that count, so the host tokenizes with a single nominal patch and slices it
#     off — it needs no knowledge of the patch grid / resize geometry.
if _falcon_cfg is None:
    raise RuntimeError(
        "Falcon config (AutoConfig) failed to load; it is required at inference time to tokenize "
        "the runtime detection query. Check that `download_path` points to the model folder."
    )
_tail_token_list, _, _patch_end_rt = build_detection_token_list(_tokenizer, _falcon_cfg, TEST_QUERY, 1)
tail_ids = np.asarray([_tail_token_list[_patch_end_rt:]], dtype=embed_in_dtype)
tail_ids_ort = create_ort_from_numpy(tail_ids, device_type, DEVICE_ID)
binding_Embed.bind_ortvalue_input(in_name_Embed, tail_ids_ort)
binding_Embed._iobinding.bind_output(out_name_Embed, _ort_device_type)
run(ort_session_Embed, binding_Embed)
tail_hidden = binding_Embed.get_outputs()[0].numpy().astype(hidden_dtype_Main, copy=False)

# 1c. Full prefill hidden states = vision prefix (head + patches) ++ runtime tail (text + query).
concat_hidden_states = np.concatenate([vision_hidden, tail_hidden], axis=1)
num_prefill = concat_hidden_states.shape[1]
hidden_states_prefill = create_ort_from_numpy(concat_hidden_states, device_type, DEVICE_ID)

# 2. Hybrid prefill attention mask (rotary tables are fused into LLM_Main and gathered by
#    cache_position). The prefill cache positions are the contiguous [0, num_prefill) range.
ids_len = create_ort_with_data([num_prefill], prefill_len_dtype, device_type, DEVICE_ID)
init_history_len = create_ort_with_data([0], prefill_hist_dtype, device_type, DEVICE_ID)
bind_ort_in_buf(binding_Prefill_Mask, in_name_Prefill_Mask, [ids_len, init_history_len])
bind_ort_out(binding_Prefill_Mask, out_name_Prefill_Mask, _ort_device_type)
run(ort_session_Prefill_Mask, binding_Prefill_Mask)
attention_mask_p = binding_Prefill_Mask.get_outputs()[0]
cache_position_prefill = create_ort_with_data(list(range(num_prefill)), cache_position_dtype_Main, device_type, DEVICE_ID)

# 3. Pre-bind the always-reused decode sessions. Rotary tables are produced inside LLM_Main
#    (gathered by cache_position), so there is no separate decode rotary session and no
#    device-side kv_seq_len counter — the host writes cache_position_buf each step.
binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)
binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)
if USE_PENALTY:
    # Greedy head selects the token (-> max_idx_buf) and grows save_id; Penalty rescales the
    # repeated-token logits in place. save_id input + save_id_out output are rebound each step.
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], logits_buf)
    binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], logits_buf)
    binding_Penalty.bind_ortvalue_output(out_name_Penalty, logits_buf)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value_ort)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penalty_range_ort)
else:
    binding_Argmax.bind_ortvalue_input(in_name_Argmax, logits_buf)
    binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)
#    Coord/Size decode-encoders consume the fused-head logits and write the replacement embedding
#    into hidden_states_buf; coord also takes the prior coords (existing_coords_buf, bound once).
binding_Coord_Encoder.bind_ortvalue_input(in_names_Coord_Enc[0], coord_logits_buf)
binding_Coord_Encoder.bind_ortvalue_input(in_names_Coord_Enc[1], existing_coords_buf)
binding_Coord_Encoder.bind_ortvalue_output(out_names_Coord_Enc[0], hidden_states_buf)
binding_Coord_Encoder.bind_ortvalue_output(out_names_Coord_Enc[1], coord_xy_buf)
binding_Size_Encoder.bind_ortvalue_input(in_names_Size_Enc[0], size_logits_buf)
binding_Size_Encoder.bind_ortvalue_output(out_names_Size_Enc[0], hidden_states_buf)
binding_Size_Encoder.bind_ortvalue_output(out_names_Size_Enc[1], size_hw_buf)

# 4. LLM_Main prefill: empty KV cache + prefill hidden states / cache positions / mask.
empty_kv = [make_empty_kv(in_meta_Main[j], kv_device, DEVICE_ID) for j in range(num_kv_io)]
bind_ort_in_buf(binding_Main, in_name_Main_kv, empty_kv)
bind_ort_in_buf(binding_Main, in_name_Main_others,
                [hidden_states_prefill, cache_position_prefill, attention_mask_p])
bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
binding_Main.bind_ortvalue_output(out_name_Main_logits, logits_buf)
binding_Main.bind_ortvalue_output(out_name_Main_coord_logits, coord_logits_buf)
binding_Main.bind_ortvalue_output(out_name_Main_size_logits, size_logits_buf)
run(ort_session_Main, binding_Main)
kv_outputs = binding_Main.get_outputs()[:num_kv_io]

decode_start_time = time.time()
prefill_elapsed = decode_start_time - prefill_start_time


# ══════════════════════════════════════════════════════════════════════════════
# DECODE LOOP (coord / size decoding + Fourier-encoder feedback)
# ══════════════════════════════════════════════════════════════════════════════
_decode_mode = 'Beam Search' if USE_BEAM_SEARCH else ('Greedy + Penalty' if USE_PENALTY else 'Greedy')
print(f'\nTest Query: {TEST_QUERY}\nDetecting . . . ({_decode_mode})')

detections = []                 # final list of {"xy": ..., "hw": ...}
num_decode = 0
generate_limit = min(MAX_NEW_TOKENS, MAX_SEQ_LEN - num_prefill)

if USE_BEAM_SEARCH:
    # ── Beam search over the vocab token. The First/Second beam graphs reorder the KV cache +
    #    save_id; the host recovers the parent-beam map from save_id each step to reorder the
    #    per-beam detection state and re-embed each child from its PARENT beam's coord/size logits.
    #    The coord/size VALUES remain greedily decoded inside the encoder graphs. ──
    def _recover_beam_index(new_ids, prev_ids):
        """Parent beam of each child: new_save_id[b, :-1] == prev_save_id[parent] (save_id is the
        gathered parent history with the child token appended), matched row-wise on the host."""
        lut = {}
        for p, row in enumerate(prev_ids.tolist()):
            lut.setdefault(tuple(row), p)
        return [lut.get(tuple(new_ids[b, :-1].tolist()), b) for b in range(new_ids.shape[0])]

    def _beam_feedback(tokens, parent_coord_np, parent_size_np, ex_coords, coord_count, pending, dets):
        """Build the (BEAM, 1, H) feedback hidden state and update the per-beam detection state.
        Coord/Size encoders run on the PARENT-gathered logits (every beam), Embed on the child
        tokens; the host then selects the per-beam row by the chosen token (COORD / SIZE / text)."""
        beam_coord_in_buf.update_inplace(np.ascontiguousarray(parent_coord_np.astype(coord_logits_dtype_Main)))
        beam_size_in_buf.update_inplace(np.ascontiguousarray(parent_size_np.astype(size_logits_dtype_Main)))
        beam_existing_coords_buf.update_inplace(np.ascontiguousarray(ex_coords))
        run(ort_session_Coord_Encoder, binding_Coord_Encoder)
        run(ort_session_Size_Encoder, binding_Size_Encoder)
        run(ort_session_Embed, binding_Embed)                       # reads beam_ids_buf (child tokens)
        coord_embed = beam_coord_embed_buf.numpy()
        size_embed = beam_size_embed_buf.numpy()
        hidden = beam_text_embed_buf.numpy().copy()
        coord_xy = beam_coord_xy_buf.numpy()
        size_hw = beam_size_hw_buf.numpy()
        for b in range(BEAM_SIZE):
            tok = int(tokens[b])
            if tok == COORD_TOKEN_ID:
                hidden[b] = coord_embed[b]
                ex_coords[b, coord_count[b]] = coord_xy[b]
                coord_count[b] += 1
                pending[b] = {'x': float(coord_xy[b, 0]), 'y': float(coord_xy[b, 1])}
            elif tok == SIZE_TOKEN_ID:
                hidden[b] = size_embed[b]
                if pending[b] is not None:
                    dets[b].append({'xy': pending[b],
                                    'hw': {'h': float(size_hw[b, 0]), 'w': float(size_hw[b, 1])}})
                    pending[b] = None
        beam_hidden_buf.update_inplace(np.ascontiguousarray(hidden.astype(hidden_dtype_Main)))

    # Bind the decode-phase detection sessions to the BEAM-batched buffers. Embed reads the child
    # tokens the beam graph writes into beam_ids_buf; Coord/Size read the parent-gathered logits.
    binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
    binding_Embed.bind_ortvalue_output(out_name_Embed, beam_text_embed_buf)
    binding_Coord_Encoder.bind_ortvalue_input(in_names_Coord_Enc[0], beam_coord_in_buf)
    binding_Coord_Encoder.bind_ortvalue_input(in_names_Coord_Enc[1], beam_existing_coords_buf)
    binding_Coord_Encoder.bind_ortvalue_output(out_names_Coord_Enc[0], beam_coord_embed_buf)
    binding_Coord_Encoder.bind_ortvalue_output(out_names_Coord_Enc[1], beam_coord_xy_buf)
    binding_Size_Encoder.bind_ortvalue_input(in_names_Size_Enc[0], beam_size_in_buf)
    binding_Size_Encoder.bind_ortvalue_output(out_names_Size_Enc[0], beam_size_embed_buf)
    binding_Size_Encoder.bind_ortvalue_output(out_names_Size_Enc[1], beam_size_hw_buf)

    # First beam: expand the single prefill hypothesis (batch 1) into BEAM_SIZE beams.
    bind_ort_in_buf(binding_First_Beam, in_name_First_Beam[:num_kv_io], kv_outputs)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_kv_io], logits_buf)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_kv_io + 1], save_id_first_buf)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_kv_io + 2], beam_size_ort)
    bind_ort_out(binding_First_Beam, out_name_First_Beam[:num_kv_io + 1], _ort_device_type)
    bind_ort_out_buf(binding_First_Beam, out_name_First_Beam[num_kv_io + 1:],
                     [beam_prob_buf, beam_ids_buf, max_idx_buf])
    run(ort_session_First_Beam, binding_First_Beam)
    outputs_Beam = binding_First_Beam.get_outputs()
    beam_kv = outputs_Beam[:num_kv_io]
    save_id = outputs_Beam[num_kv_io]
    prev_save_id = save_id.numpy()

    # Per-beam host detection state (reordered each step to the new parent order).
    beam_existing_coords = np.full((BEAM_SIZE, MAX_NEW_TOKENS, 2), -1.0, dtype=existing_coords_dtype)
    beam_coord_count = [0] * BEAM_SIZE
    beam_pending = [None] * BEAM_SIZE
    beam_dets = [[] for _ in range(BEAM_SIZE)]

    if int(max_idx_buf.numpy().flat[0]) not in STOP_TOKEN_SET:
        # First-step feedback: all children share the single prefill parent, so its coord/size
        # logits (the prefill batch-1 outputs) are broadcast to every beam.
        _beam_feedback(beam_ids_buf.numpy().reshape(-1),
                       np.repeat(coord_logits_buf.numpy(), BEAM_SIZE, axis=0),
                       np.repeat(size_logits_buf.numpy(), BEAM_SIZE, axis=0),
                       beam_existing_coords, beam_coord_count, beam_pending, beam_dets)

        # Switch LLM_Main to the beam decode buffers and pre-bind the Second-beam static inputs.
        bind_ort_in_buf(binding_Main, in_name_Main_others, [beam_hidden_buf, cache_position_buf, decode_mask_buf])
        binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)
        binding_Main.bind_ortvalue_output(out_name_Main_coord_logits, decode_coord_buf)
        binding_Main.bind_ortvalue_output(out_name_Main_size_logits, decode_size_buf)
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_kv_io], decode_logits_buf)
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_kv_io + 2], beam_prob_buf)
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_kv_io + 3], beam_size_ort)
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_kv_io + 4], topK_ort)
        if USE_PENALTY:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
            binding_Penalty.bind_ortvalue_output(out_name_Penalty, decode_logits_buf)
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value_ort)
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penalty_range_ort)

        # First decode LLM_Main step (batch == BEAM_SIZE).
        cache_position_buf.update_inplace(np.asarray([num_prefill + num_decode], dtype=cache_position_dtype_Main))
        bind_ort_in_buf(binding_Main, in_name_Main_kv, beam_kv)
        bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
        run(ort_session_Main, binding_Main)
        beam_kv = binding_Main.get_outputs()[:num_kv_io]
        num_decode = 1

        while num_decode < generate_limit:
            # Optional repetition penalty on the per-beam logits (once the window is full).
            if USE_PENALTY and num_decode >= PENALTY_RANGE:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
                run(ort_session_Penalty, binding_Penalty)
            # Second beam: prune + re-expand from the prior step's logits + KV cache.
            bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam[:num_kv_io], beam_kv)
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_kv_io + 1], save_id)
            bind_ort_out(binding_Second_Beam, out_name_Second_Beam[:num_kv_io + 1], _ort_device_type)
            bind_ort_out_buf(binding_Second_Beam, out_name_Second_Beam[num_kv_io + 1:],
                             [beam_prob_buf, beam_ids_buf, max_idx_buf])
            run(ort_session_Second_Beam, binding_Second_Beam)
            outputs_Beam = binding_Second_Beam.get_outputs()
            beam_kv = outputs_Beam[:num_kv_io]
            save_id = outputs_Beam[num_kv_io]
            new_save_id = save_id.numpy()
            beam_index = _recover_beam_index(new_save_id, prev_save_id)
            prev_save_id = new_save_id

            # Reorder the per-beam host detection state into the new parent order.
            beam_existing_coords = np.ascontiguousarray(beam_existing_coords[beam_index])
            beam_coord_count = [beam_coord_count[p] for p in beam_index]
            beam_pending = [beam_pending[p] for p in beam_index]
            beam_dets = [list(beam_dets[p]) for p in beam_index]

            if int(max_idx_buf.numpy().flat[0]) in STOP_TOKEN_SET:
                break

            # Re-embed each child from its PARENT beam's coord/size logits (source-order Main
            # outputs gathered by the recovered parent map), then feed the next LLM_Main step.
            _beam_feedback(beam_ids_buf.numpy().reshape(-1),
                           decode_coord_buf.numpy()[beam_index],
                           decode_size_buf.numpy()[beam_index],
                           beam_existing_coords, beam_coord_count, beam_pending, beam_dets)
            cache_position_buf.update_inplace(np.asarray([num_prefill + num_decode], dtype=cache_position_dtype_Main))
            bind_ort_in_buf(binding_Main, in_name_Main_kv, beam_kv)
            bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
            run(ort_session_Main, binding_Main)
            beam_kv = binding_Main.get_outputs()[:num_kv_io]
            num_decode += 1

    # Best hypothesis is beam 0 (beams stay sorted by score); report its decoded detections.
    detections = beam_dets[0]
    decode_end_time = time.time()
else:
    aux_output = []             # interleaved coord / size dicts
    coord_count = 0             # decoded coords written into existing_coords_buf
    pending_xy = None

    # Switch LLM_Main's non-KV inputs to the reusable decode buffers (KV bound per step).
    bind_ort_in_buf(binding_Main, in_name_Main_others,
                    [hidden_states_buf, cache_position_buf, decode_mask_buf])

    while num_decode < generate_limit:
        # 1. Sample the next token (greedy, optional repetition penalty) from the current logits buffer.
        if USE_PENALTY:
            # Once the trailing window is full, rescale the repeated-token logits in place, then let the
            # Greedy head pick the token and append it to save_id (the next step's penalty history).
            if num_decode >= PENALTY_RANGE:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
                run(ort_session_Penalty, binding_Penalty)
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
            binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
            run(ort_session_Greedy, binding_Greedy)
            save_id = binding_Greedy.get_outputs()[1]
        else:
            run(ort_session_Argmax, binding_Argmax)
        next_token = int(max_idx_buf.numpy().flat[0])
        if next_token in STOP_TOKEN_SET:
            break

        # 2. Conditional feedback into hidden_states_buf: all three producers write the same
        #    buffer (so LLM_Main reads one fixed input) and run on device.
        if next_token == COORD_TOKEN_ID:
            # Coord decode-encoder (argmax + dedup + Fourier gather, all in ONNX): writes the
            # replacement embedding to hidden_states_buf; only the deduped (x, y) crosses back.
            run(ort_session_Coord_Encoder, binding_Coord_Encoder)   # -> hidden_states_buf on device
            pred_x, pred_y = (v for v in coord_xy_buf.numpy().reshape(-1))
            # Record the new coord into the next history slot and refresh the bound buffer in place.
            existing_coords_host[0, coord_count] = (pred_x, pred_y)
            coord_count += 1
            existing_coords_buf.update_inplace(existing_coords_host)
            aux_output.append({'x': pred_x, 'y': pred_y})
            pending_xy = {'x': pred_x, 'y': pred_y}
        elif next_token == SIZE_TOKEN_ID:
            # Size decode-encoder (argmax + process_sizes + Fourier gather, in ONNX): only (h, w)
            # crosses back; the embedding goes to hidden_states_buf.
            run(ort_session_Size_Encoder, binding_Size_Encoder)     # -> hidden_states_buf on device
            size_h, size_w = (v for v in size_hw_buf.numpy().reshape(-1))
            size_pred = {'h': size_h, 'w': size_w}
            aux_output.append(size_pred)
            if pending_xy is not None:
                detections.append({'xy': pending_xy, 'hw': size_pred})
                pending_xy = None
        else:
            run(ort_session_Embed, binding_Embed)   # max_idx_buf -> hidden_states_buf (zero host round-trip)

        # 3. LLM_Main decode step. The new token's KV-cache position (num_prefill + step) is
        #    written into cache_position_buf so LLM_Main gathers the right fused rotary row;
        #    the previous step's KV outputs feed back as inputs (zero-copy on device).
        cache_position_buf.update_inplace(np.asarray([num_prefill + num_decode], dtype=cache_position_dtype_Main))
        bind_ort_in_buf(binding_Main, in_name_Main_kv, kv_outputs)
        bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
        run(ort_session_Main, binding_Main)
        kv_outputs = binding_Main.get_outputs()[:num_kv_io]

        num_decode += 1

    decode_end_time = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
decode_elapsed = decode_end_time - decode_start_time
total_elapsed = decode_end_time - prefill_start_time
prefill_tps = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
decode_tps = num_decode / decode_elapsed if decode_elapsed > 0 else 0.0
overall_tps = (num_decode + num_prefill) / total_elapsed if total_elapsed > 0 else 0.0

print(
    f"\n\n{'-' * 56}\n"
    f"  Detections ({len(detections)})\n"
    f"{'-' * 56}"
)
for idx, det in enumerate(detections):
    xy, hw = det['xy'], det['hw']
    print(f"  [{idx}] center=({xy['x']:.4f}, {xy['y']:.4f})  size=(h={hw['h']:.4f}, w={hw['w']:.4f})")
if not detections:
    print("  (no detections)")

if SAVE_VISUALIZATION and detections:
    saved_path = visualize_detections(TEST_IMAGE[0], detections, TEST_QUERY, VISUALIZATION_PATH)
    print(f"\n  Saved YOLO-style visualization -> {saved_path}")

print(
    f"{'-' * 56}\n\n"
    f"  Performance Summary\n"
    f"{'-' * 56}\n"
    f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
    f"  {'-' * 48}\n"
    f"  {'Prefill':<12} {prefill_tps:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
    f"  {'Decode':<12} {decode_tps:>10.2f} t/s {num_decode:>8d} {decode_elapsed:>8.3f}s\n"
    f"  {'-' * 48}\n"
    f"  {'Overall':<12} {overall_tps:>10.2f} t/s {num_decode:>8d} {total_elapsed:>8.3f}s\n"
    f"{'-' * 56}\n"
)
