import gc
import importlib
import os
import sys
import time
import types

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(SCRIPT_DIR, 'Falcon_Perception_ONNX')

download_path                  = r'/home/DakeQQ/Downloads/Falcon-Perception-300M'   # Folder of the downloaded Falcon Perception 300M project.

onnx_model_Embed               = os.path.join(EXPORT_DIR, 'LLM_Embed.onnx')
onnx_model_Vision              = os.path.join(EXPORT_DIR, 'LLM_Vision.onnx')
onnx_model_Prefill_Mask        = os.path.join(EXPORT_DIR, 'Prefill_Mask.onnx')
onnx_model_Main                = os.path.join(EXPORT_DIR, 'LLM_Main.onnx')
onnx_model_Greedy              = os.path.join(EXPORT_DIR, 'Greedy_Search.onnx')
onnx_model_First_Beam          = os.path.join(EXPORT_DIR, 'First_Beam_Search.onnx')
onnx_model_Second_Beam         = os.path.join(EXPORT_DIR, 'Second_Beam_Search.onnx')
onnx_model_Penalty             = os.path.join(EXPORT_DIR, 'Apply_Penalty.onnx')
onnx_model_Argmax              = os.path.join(EXPORT_DIR, 'Argmax.onnx')
onnx_model_KV_Slice            = os.path.join(EXPORT_DIR, 'KV_Slice.onnx')
onnx_model_Coord_Encoder       = os.path.join(EXPORT_DIR, 'Coord_Encoder.onnx')
onnx_model_Size_Encoder        = os.path.join(EXPORT_DIR, 'Size_Encoder.onnx')


# Test input
TEST_IMAGE               = [os.path.join(SCRIPT_DIR, 'psyduck.jpg')]   # One image for the static detection path (anchored to the script dir, cwd-independent).
TEST_QUERY               = 'psyduck'               # Detection query text. Now a RUNTIME input: change it and re-run with DO_EXPORT=False (no re-export needed).

# Model Config
DO_EXPORT                = True                    # Export the ONNX models. The query is a runtime input (no re-export to change it); set True only after changing image geometry / quant / decode settings.
PREVENT_F16_OVERFLOW     = False                   # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
MAX_SEQ_LEN              = 8192                    # Max context length. Can not edit after export.

# Vision / prompt static geometry (detection prompt). Image is resized to a
# multiple of the patch size; the resulting patch grid is fixed at export time.
PATCH_SIZE               = 16                      # Falcon's spatial patch size is 16x16. The image is resized to a multiple of this.
HEIGHT_FACTOR            = 28
WIDTH_FACTOR             = 28
IMAGE_RESIZE             = [PATCH_SIZE * HEIGHT_FACTOR, PATCH_SIZE * WIDTH_FACTOR] # [height, width], each a multiple of 16. Static export geometry.
INPUT_IMAGE_SIZE         = [448, 448]              # Raw image shape fed to the ONNX vision graph before preprocessing.
VISION_BATCH_SIZE        = 1                       # Fixed at export time. Number of images per prompt.
INPUT_IMAGE_DIM          = 5                       # 4 for [batch, 3, H, W]; 5 for [batch, 1, 3, H, W].
DYNAMIC_VISION_SHAPE     = False                   # True: vision graph accepts ANY raw input H/W (and image count) and resizes each image to IMAGE_RESIZE in-graph, so the patch grid stays fixed (input H/W + batch are dynamic axes). False: static geometry, input H/W locked to INPUT_IMAGE_SIZE (drops the in-graph resize + batch-expand).

IMAGE_MEAN               = [0.5, 0.5, 0.5]
IMAGE_STD                = [0.5, 0.5, 0.5]

# KV cache quantization
KV_QUANT_DTYPE           = "F16"                   # "ROTARY_Q4" | "ROTARY_Q4_CUDA" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "F16" | "F32"
KV_QUANT_GROUP_SIZE      = 32                      # Group size for Q4 / grouped-Q8 quantization. Must divide head_dim evenly.
USE_HADAMARD             = True                    # Apply randomized Walsh-Hadamard mixing within each group before quantization.
HADAMARD_RANDOM_SEED     = 9527                    # Seed for the deterministic Rademacher sign pattern.
USE_CLIP                 = True                    # Clip outliers to mean +/- CLIP_SIGMA*std before quantization.
CLIP_SIGMA               = 3.0                     # Clip threshold in standard deviations.
USE_SHUFFLE              = True                    # Interleave channels across groups so high-variance channels are spread out.
USE_SYM                  = False                   # True: symmetric quantization (no bias); False: asymmetric (min-max with bias).
USE_FLOAT16_SCALE_BIAS   = True                    # Use float16 for scale/bias in all quantized KV modes.

# Decoding strategy
USE_BEAM_SEARCH          = False                   # Beam search or greedy search.
REPEAT_PENALTY           = 1.0                     # 0.0 ~ 1.0; no penalty = 1.0.
PENALTY_RANGE            = 20                      # Recent-token window for the repetition penalty.
MAX_BEAM_SIZE            = 10                      # Max beam size. Can not edit after export.
TOP_K                    = 3                       # Top-K for beam search.
BEAM_SIZE                = 3                       # Beam size for beam search. Must be <= MAX_BEAM_SIZE.

# Detection runtime
MAX_NEW_TOKENS           = 1024                    # Max decode steps for detection generation.
COORD_DEDUP_THRESHOLD    = 0.01                    # Duplicate-coordinate suppression threshold (fraction of image size).
MAX_COORD_ATTEMPTS       = 100                     # Max re-sampling attempts for duplicate coordinates.
SAVE_VISUALIZATION       = True                    # Draw YOLO-style boxes on the input image and save the result.
VISUALIZATION_PATH       = os.path.join(SCRIPT_DIR, 'detection_output.jpg')   # Output path for the annotated image.

# Runtime config
ORT_LOG                  = False                   # Enable ONNX Runtime logging for debugging.
ORT_FP16                 = False                   # Set True for FP16 ONNX Runtime settings (ARM64-v8.2a+ on CPU).
ORT_Accelerate_Providers = []                      # e.g. ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS              = 0                       # 0 = auto.
DEVICE_ID                = 0                       # GPU device id.
OPSET                    = 18                      # ONNX opset version.


SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4", "ROTARY_Q4_CUDA", "Q8", "Q8_CUDA",
    "ROTARY_Q8", "ROTARY_Q8_CUDA", "F16", "F32"
)


# ══════════════════════════════════════════════════════════════════════════════
# Falcon model loader (CPU, float32, triton / flex-attention stubbed out)
# ══════════════════════════════════════════════════════════════════════════════
def _install_runtime_stubs():
    """Stub triton + FlexAttention so the Falcon source imports on a plain CPU env."""
    triton_pkg = types.ModuleType("triton")
    triton_pkg.__path__ = []
    triton_lang = types.ModuleType("triton.language")
    triton_pkg.jit = lambda fn: fn
    triton_pkg.cdiv = lambda a, b: (a + b - 1) // b
    triton_pkg.language = triton_lang
    for name in ("dtype", "program_id", "arange", "load", "store", "where"):
        setattr(triton_lang, name, (lambda *a, **k: None))
    triton_lang.constexpr = lambda x: x
    sys.modules["triton"] = triton_pkg
    sys.modules["triton.language"] = triton_lang
    sys.modules["triton.backends"] = types.ModuleType("triton.backends")
    sys.modules["triton.backends.compiler"] = types.ModuleType("triton.backends.compiler")

    # No-op torch.compile so the model's compile_model() path is inert.
    torch.compile = lambda fn=None, *a, **k: (fn if fn is not None else (lambda f: f))

    import torch.nn.attention.flex_attention as flex_mod

    class _StubBlockMask:
        BLOCK_SIZE = (128, 128)

        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getitem__(self, index):
            return self

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return None

    flex_mod.BlockMask = _StubBlockMask
    flex_mod.AuxRequest = lambda **kwargs: None
    flex_mod.create_block_mask = lambda *a, **k: _StubBlockMask()
    flex_mod.and_masks = lambda *fns: (lambda b, h, q, kv: True)
    flex_mod.or_masks = lambda *fns: (lambda b, h, q, kv: True)
    if not hasattr(flex_mod, "_mask_mod_signature"):
        flex_mod._mask_mod_signature = type(None)


def load_falcon(model_path):
    """Load Falcon Perception 300M and its processing helpers from a local folder."""
    _install_runtime_stubs()

    model_path = os.path.abspath(model_path)
    parent_dir = os.path.dirname(model_path)
    pkg_name = os.path.basename(model_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    init_file = os.path.join(model_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w"):
            pass

    config_mod = importlib.import_module(f"{pkg_name}.configuration_falcon_perception")
    model_mod = importlib.import_module(f"{pkg_name}.modeling_falcon_perception")
    proc_mod = importlib.import_module(f"{pkg_name}.processing_falcon_perception")

    config = config_mod.FalconPerceptionConfig.from_pretrained(model_path)
    model = model_mod.FalconPerceptionForSegmentation.from_pretrained(
        model_path, config=config, dtype=torch.float32, device_map={"": "cpu"},
    ).eval()
    # Rebuild the non-persistent freqs_cis buffer that meta-device loading leaves empty.
    model._weights_fused = False
    model._ensure_device_buffers()
    model._is_compiled = True     # never trigger compile_model()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    return model, tokenizer, config, proc_mod


# ══════════════════════════════════════════════════════════════════════════════
# Static prompt / RoPE / mask layout (baked at export time)
# ══════════════════════════════════════════════════════════════════════════════
DETECTION_PROMPT_TEMPLATE = "<|image|>Detect these expressions in the image:<|start_of_query|>{query}<|DET|>"


def build_detection_token_list(tokenizer, config, query, num_patches):
    """Tokenize the detection prompt into the token-id list plus the img_id patch span
    [patch_start, patch_end). Shared by the export layout and the runtime prefill."""
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


def build_static_layout(model, tokenizer, config, proc_mod, image_resize, query):
    """Run the real Falcon preprocessing once and bake every prompt-dependent constant the
    export needs into a dict (tokens, RoPE positions, masks, patch span, grid)."""
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    model._pad_token_id = pad_token_id

    patch = config.spatial_patch_size
    grid_h = image_resize[0] // patch
    grid_w = image_resize[1] // patch
    num_patches = grid_h * grid_w

    token_list, _, _ = build_detection_token_list(tokenizer, config, query, num_patches)
    tokens = torch.tensor(token_list, dtype=torch.long).unsqueeze(0)

    # Full-resolution pixel mask so get_pos_thw yields this grid's spatial/temporal positions.
    pixel_mask = torch.ones((1, config.temporal_patch_size, image_resize[0], image_resize[1]), dtype=torch.long)
    pos_t, pos_hw = proc_mod.get_pos_thw(
        tokens, pixel_mask, config, config.spatial_patch_size,
        config.temporal_patch_size, pad_token_id=pad_token_id,
    )

    seq = tokens.shape[1]
    img_id_mask = (tokens[0] == config.img_id)
    patch_index = torch.full((seq,), -1, dtype=torch.long)
    patch_index[img_id_mask] = torch.arange(int(img_id_mask.sum()), dtype=torch.long)

    patch_positions = torch.nonzero(img_id_mask, as_tuple=False).flatten().tolist()
    patch_start = patch_positions[0]
    patch_end = patch_positions[-1] + 1
    assert patch_end - patch_start == num_patches, "img_id patch span must be contiguous."
    assert int(img_id_mask.sum()) == num_patches

    # Bidirectional image-block span [img_lo, img_hi): image_cls .. last patch (img_end excluded).
    img_lo = int(torch.nonzero(tokens[0] == config.image_cls_token_id, as_tuple=False)[0])
    img_hi = patch_end

    head_token_ids = tokens[0, :patch_start].clone()
    tail_token_ids = tokens[0, patch_end:].clone()

    attn_mask_add = _build_hybrid_prefill_mask(tokens[0], config, pad_token_id)

    return {
        "tokens": tokens[0],
        "pos_t": pos_t[0],
        "pos_hw": pos_hw[0],
        "img_id_mask": img_id_mask,
        "patch_index": patch_index,
        "attn_mask_add": attn_mask_add,
        "head_token_ids": head_token_ids,
        "tail_token_ids": tail_token_ids,
        "patch_start": patch_start,
        "patch_end": patch_end,
        "img_lo": img_lo,
        "img_hi": img_hi,
        "num_patches": num_patches,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "pad_token_id": pad_token_id,
    }


def _build_hybrid_prefill_mask(tokens_S, config, pad_token_id):
    """Dense additive prefill mask: (image-prefix bidirectional) OR (causal AND
    same-document AND non-left-pad). Matches create_batch_attention_mask for the
    static single-image prompt. Returns (1, 1, S, S) float32 (0 = attend, -inf masked)."""
    seq = tokens_S.shape[0]
    device = tokens_S.device

    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))

    non_pad_cumsum = torch.cumsum((tokens_S != pad_token_id).int(), dim=0)
    non_left_pad = non_pad_cumsum > 0
    non_pad_kv = non_left_pad.unsqueeze(0).expand(seq, -1)

    eos_mask = (tokens_S == config.eos_id).clone()
    eos_mask[-1] = True
    cum_eos = torch.cumsum(eos_mask.int(), dim=0)
    doc_ids = torch.zeros(seq, dtype=torch.int32, device=device)
    doc_ids[1:] = cum_eos[:-1]
    same_doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)

    soi = (tokens_S == config.image_cls_token_id)
    eoi = (tokens_S == config.img_end_id)
    acc_soi = torch.cumsum(soi.int(), dim=0)
    acc_eoi = torch.cumsum(eoi.int(), dim=0)
    is_img = (acc_soi - acc_eoi) > 0
    img_idx = acc_soi * is_img
    img_bidir = is_img.unsqueeze(0) & is_img.unsqueeze(1) & (img_idx.unsqueeze(0) == img_idx.unsqueeze(1))

    allowed = img_bidir | (causal & same_doc & non_pad_kv)
    additive = torch.where(allowed, torch.zeros((), dtype=torch.float32),
                           torch.full((), -float('inf'), dtype=torch.float32))
    return additive.view(1, 1, seq, seq)


def build_temporal_rope_tables(model, max_seq_len):
    """cos/sin tables for Falcon's 1D temporal RoPE, in the permuted half-split
    [cos|cos] / [-sin|sin] layout the flip-based _rotate_half consumes."""
    freqs_cis = model.freqs_cis[:max_seq_len].to(torch.complex64)   # (S, 16)
    angle = torch.angle(freqs_cis).float()                          # (S, 16)
    cos16 = torch.cos(angle)                                        # (S, 16)
    sin16 = torch.sin(angle)                                        # (S, 16)
    cos = torch.cat([cos16, cos16], dim=-1)                         # (S, 32)
    sin = torch.cat([-sin16, sin16], dim=-1)                        # (S, 32)
    return cos, sin


def build_spatial_rope_tables(model, pos_hw, img_id_mask):
    """cos/sin tables for Falcon's per-head 2D golden spatial RoPE, shape (S, H, 32),
    identity on non-patch positions, in the permuted half-split layout."""
    seq = pos_hw.shape[0]
    num_heads, num_freqs, _ = model.freqs_cis_golden.shape  # 16, 16, 2
    golden = model.freqs_cis_golden.float()

    pos = torch.nan_to_num(pos_hw, nan=0.0).float()                 # (S, 2)
    theta = torch.einsum("sp,hfp->shf", pos, golden)                # (S, H, F)
    cos_hf = torch.cos(theta)
    sin_hf = torch.sin(theta)
    # Identity rotation on non-patch positions.
    keep = img_id_mask.view(seq, 1, 1).float()
    cos_hf = cos_hf * keep + (1.0 - keep)
    sin_hf = sin_hf * keep
    cos = torch.cat([cos_hf, cos_hf], dim=-1)                      # (S, H, 32)
    sin = torch.cat([-sin_hf, sin_hf], dim=-1)                     # (S, H, 32)
    return cos, sin


def normalize_kv_quant_settings(head_dim):
    """Validate and normalize KV quant settings once head_dim is known."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized_kv = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"}
    rotary_kv = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    notes = []

    if KV_QUANT_DTYPE in rotary_kv and head_dim % 2 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires an even head_dim, got {head_dim}.")
    if KV_QUANT_DTYPE in {"Q8_CUDA", "ROTARY_Q8_CUDA"} and head_dim % 4 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 4, got {head_dim}.")
    if KV_QUANT_DTYPE == "ROTARY_Q4_CUDA" and head_dim % 8 != 0:
        raise ValueError(f"{KV_QUANT_DTYPE} requires head_dim divisible by 8, got {head_dim}.")

    if KV_QUANT_DTYPE in quantized_kv:
        if KV_QUANT_GROUP_SIZE <= 0:
            raise ValueError(f"KV_QUANT_GROUP_SIZE must be positive, got {KV_QUANT_GROUP_SIZE}.")
        if KV_QUANT_GROUP_SIZE > head_dim:
            notes.append(f"[Warning] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) > head_dim ({head_dim}); clamping.")
            KV_QUANT_GROUP_SIZE = head_dim
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE != 0:
            original = KV_QUANT_GROUP_SIZE
            KV_QUANT_GROUP_SIZE = max(g for g in range(1, KV_QUANT_GROUP_SIZE + 1) if head_dim % g == 0)
            notes.append(f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not divide head_dim; using {KV_QUANT_GROUP_SIZE}.")
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append("[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32.")

    return notes


# ══════════════════════════════════════════════════════════════════════════════
# KV cache quantizer (Q8 / Q8_CUDA / Q4 storage paths, ported from LightOn)
# ══════════════════════════════════════════════════════════════════════════════
class KVQuantizer(torch.nn.Module):
    """KV cache quantizer (Q8 / Q8_CUDA / Q4) with optional Hadamard mixing, channel
    shuffle, sigma clipping, symmetric/asymmetric ranges, and residual bias correction."""

    def __init__(self, head_dim, num_kv_heads, num_kv_groups, is_q4=False, is_q8_cuda=False,
                 use_sym=False, use_hadamard=False, use_clip=False, clip_sigma=2.5, use_shuffle=False):
        super().__init__()
        self.is_q4         = is_q4
        self.is_q8_cuda    = is_q8_cuda
        self.use_sym       = use_sym
        self.use_hadamard  = use_hadamard
        self.use_clip      = use_clip
        self.clip_sigma    = clip_sigma
        self.use_shuffle   = use_shuffle
        self.use_residual_bias_correction = not use_sym
        self.head_dim      = head_dim
        self.head_dim_half = head_dim // 2 if head_dim else 0
        self.num_kv_heads  = num_kv_heads
        self.num_kv_groups = num_kv_groups

        if use_sym:
            self.SIGNED_QMIN = -8 if is_q4 else -128
            self.SIGNED_QMAX = 7 if is_q4 else 127
            self.QMAX        = float(self.SIGNED_QMAX)
        else:
            self.SIGNED_QMIN = None
            self.SIGNED_QMAX = None
            self.QMAX        = 15.0 if is_q4 else 255.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.QMAX]).view(1, 1, 1, 1, -1))

        self.is_grouped          = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle  = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0

        if is_q8_cuda:
            for name, val in [("_256", 256), ("_128", 128), ("_65536", 65536), ("_16777216", 16777216)]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size ** -0.5], dtype=torch.float32))
            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(0, 2, (self.kv_quant_group_size,), generator=sign_generator, dtype=torch.int64)
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)
            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        if self.use_shuffle:
            perm = torch.arange(head_dim).view(self.kv_quant_num_groups, self.kv_quant_group_size).T.contiguous().view(-1)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(head_dim)
            self.register_buffer("shuffle_idx", perm.int())
            self.register_buffer("unshuffle_idx", inv_perm.int())

    @staticmethod
    def _next_power_of_two(n):
        value = 1
        while value < n:
            value *= 2
        return value

    def _apply_hadamard_last_dim(self, x, inverse=False):
        if not self.use_hadamard:
            return x
        if not inverse:
            x = x * self.hadamard_sign
        if self.hadamard_pad:
            x = F.pad(x, (0, self.hadamard_pad))
        for width, half in self._hadamard_levels:
            x = x.view(*x.shape[:-1], -1, width)
            even, odd = torch.split(x, [half, half], dim=-1)
            x = torch.cat([even + odd, even - odd], dim=-1)
            x = x.view(*x.shape[:-2], -1)
        x = x * self.hadamard_inv_sqrt
        if self.hadamard_pad:
            x = x[..., :self.kv_quant_group_size]
        if inverse:
            x = x * self.hadamard_sign
        return x

    def _clip_to_sigma(self, x, dim):
        mean  = x.mean(dim=dim, keepdim=True)
        var   = (x - mean).square().mean(dim=dim, keepdim=True)
        std   = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    def hadamard_k(self, k, batch_size, inverse=False):
        k = k.reshape(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2), inverse=inverse).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size, inverse=False):
        v = v.reshape(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        v = self._apply_hadamard_last_dim(v, inverse=inverse)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
        if self.use_residual_bias_correction:
            block_residual = x - (x_packed * scale + block_min)
            block_min = block_min + block_residual.mean(dim=dim, keepdim=True)
        if not self.is_q8_cuda:
            x_packed = x_packed.to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale     = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def _quantize_signed_to_storage(self, x, scale):
        x_quant = torch.round(x / scale).clamp(self.SIGNED_QMIN, self.SIGNED_QMAX).to(torch.int32)
        if self.is_q4:
            return torch.remainder(x_quant, 16).to(torch.uint8)
        if self.is_q8_cuda:
            return torch.remainder(x_quant, 256).to(torch.uint8)
        return x_quant.to(torch.int8)

    @staticmethod
    def _decode_signed_q4_storage(x):
        x = x.to(torch.int16)
        return torch.remainder(x + 8, 16) - 8

    @staticmethod
    def _decode_signed_q8_storage(x):
        if x.dtype == torch.int8:
            return x.to(torch.int16)
        x = x.to(torch.int16)
        return torch.remainder(x + 128, 256) - 128

    def _quantize_block(self, x, dim, batch_size=1):
        if self.is_grouped:
            return self._quantize_block_grouped(x, dim, batch_size)
        if self.use_sym:
            if self.use_clip:
                x = self._clip_to_sigma(x, dim=dim)
            absmax = x.abs().amax(dim=dim, keepdim=True)
            scale  = absmax * self.inv_qmax
            x_packed = self._quantize_signed_to_storage(x, scale)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        if self.use_clip:
            x = self._clip_to_sigma(x, dim=dim)
        block_min, block_max = torch.aminmax(x, dim=dim, keepdim=True)
        scale        = (block_max - block_min) * self.inv_qmax
        x_normalized = (x - block_min) / scale
        x_packed     = torch.round(x_normalized)
        return self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim)

    def _quantize_block_grouped(self, x, dim, batch_size):
        if self.use_sym:
            if dim == -2:
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax   = x.abs().amax(dim=-2, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                absmax   = x.abs().amax(dim=-1, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        else:
            if dim == -2:
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-2)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-1)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            return x_packed, scale, block_min

    def pack_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216

    def unpack_cuda(self, x_i32, dim, batch_size, num_kv_heads, head_dim):
        r3 = x_i32 % self._16777216
        x3 = (x_i32 - r3) // self._16777216 + self._128
        x2 = r3 // self._65536
        r2 = r3 % self._65536
        x1 = r2 // self._256
        x0 = r2 % self._256
        unpacked = torch.stack([x0, x1, x2, x3], dim=dim)
        if dim != -1:
            return unpacked.reshape(batch_size, num_kv_heads, 1, head_dim, -1)
        return unpacked.reshape(batch_size, num_kv_heads, 1, -1, head_dim)

    def pack_q4_k(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        low, high = torch.unbind(x, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, x, batch_size):
        x = x.view(batch_size, self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        low, high = torch.unbind(x, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, x, batch_size):
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
        if self.use_shuffle:
            keys   = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)
        if self.use_hadamard:
            keys   = self.hadamard_k(keys, batch_size)
            values = self.hadamard_v(values, batch_size)
        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys,   dim=-2, batch_size=batch_size)
            v_packed, v_scale = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, v_packed, v_scale
        else:
            k_packed, k_scale, k_bias = self._quantize_block(keys,   dim=-2, batch_size=batch_size)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias

    def quantize_keys(self, keys, batch_size, num_kv_heads, head_dim_quarter):
        """Quantize the key tensor only. keys: (B, H, 1, head_dim, S)."""
        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
        if self.use_hadamard:
            keys = self.hadamard_k(keys, batch_size)
        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, None
        k_packed, k_scale, k_bias = self._quantize_block(keys, dim=-2, batch_size=batch_size)
        if self.is_q4:
            k_packed = self.pack_q4_k(k_packed, batch_size)
        if self.is_q8_cuda:
            k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
        return k_packed, k_scale, k_bias

    def quantize_values(self, values, batch_size, num_kv_heads, head_dim_quarter):
        """Quantize the value tensor only. values: (B, H, 1, S, head_dim)."""
        if self.use_shuffle:
            values = values.index_select(-1, self.shuffle_idx)
        if self.use_hadamard:
            values = self.hadamard_v(values, batch_size)
        if self.use_sym:
            v_packed, v_scale = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return v_packed, v_scale, None
        v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
        if self.is_q4:
            v_packed = self.pack_q4_v(v_packed, batch_size)
        if self.is_q8_cuda:
            v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
        return v_packed, v_scale, v_bias


# ══════════════════════════════════════════════════════════════════════════════
# LLM_Embed — token embedding lookup for normal text/task tokens
# ══════════════════════════════════════════════════════════════════════════════
class LLM_EMBED(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.tok_embeddings.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Coord / Size Fourier encoders (re-embed coord_token / size_token positions)
# ══════════════════════════════════════════════════════════════════════════════
def fourier_bin_values(out_dim, is_size):
    """Per-bin input value table for a Fourier encoder (num_bins = out_dim // 2):
    coord value(k) = k / (num_bins - 1); size value(k) = process_sizes' inverse."""
    num_bins = out_dim // 2
    pred = torch.arange(num_bins, dtype=torch.float64) / (num_bins - 1)
    if not is_size:
        return pred
    min_size = float(np.log2(1.0 / num_bins))
    return torch.exp2(pred * (-min_size) + min_size)


class FOURIER_ENCODER(torch.nn.Module):
    """Fourier-encode a discrete coord/size bin pair into a (batch, 1, dim) replacement
    hidden state. Per-axis cos/sin angle tables are precomputed at init (the input is a
    deterministic function of the argmax bin), so the runtime gathers four rows and
    recombines them via the angle-addition identity — no embed matmul, no runtime trig."""

    def __init__(self, encoder, bin_values):
        super().__init__()
        embed_w = encoder.embed.weight.data.float()                         # (feat // 2, 2)
        two_pi = 2.0 * float(np.pi)
        vals = bin_values.float().view(-1, 1)                               # (num_bins, 1)
        angle0 = (two_pi * vals * embed_w[:, 0].view(1, -1)).unsqueeze(1)   # (num_bins, 1, feat // 2)
        angle1 = (two_pi * vals * embed_w[:, 1].view(1, -1)).unsqueeze(1)
        self.register_buffer('cos0', angle0.cos(), persistent=False)        # (num_bins, 1, feat // 2)
        self.register_buffer('sin0', angle0.sin(), persistent=False)
        self.register_buffer('cos1', angle1.cos(), persistent=False)
        self.register_buffer('sin1', angle1.sin(), persistent=False)
        self.transform = encoder.transform.float()

    def forward(self, bins):
        # bins: (batch, 2) int — gather each axis' cos/sin row, recombine via angle-addition.
        bins_0, bins_1 = torch.unbind(bins, dim=1)           # each (batch,)
        c0 = torch.index_select(self.cos0, 0, bins_0)        # (batch, 1, feat // 2)
        s0 = torch.index_select(self.sin0, 0, bins_0)
        c1 = torch.index_select(self.cos1, 0, bins_1)
        s1 = torch.index_select(self.sin1, 0, bins_1)
        f = torch.cat([c0 * c1 - s0 * s1, s0 * c1 + c0 * s1], dim=-1)
        return self.transform(f)


class COORD_DECODE_ENCODER(torch.nn.Module):
    """Coord post-process + Fourier re-embed in one ONNX graph: argmax + duplicate
    suppression + Fourier encode, emitting the (batch, 1, dim) hidden state and (x, y).

    The host's mask-best-then-re-argmax loop equals walking each axis' bins in descending
    order, so one top-K per axis gives every candidate (the diagonal of the two sorted
    lists) and the threshold test picks the first non-duplicate — no data-dependent loop."""

    def __init__(self, encoder, bin_values, max_attempts, dedup_threshold):
        super().__init__()
        self.fourier = FOURIER_ENCODER(encoder, bin_values)
        num_bins = int(bin_values.numel())
        self.num_bins = num_bins
        self.K = int(min(max_attempts, num_bins))   # candidates == host's max re-sampling attempts
        self.inv = 1.0 / (num_bins - 1)             # bin -> coord value (matches host pred_x / pred_y)
        self.dedup_threshold = float(dedup_threshold)
        self.register_buffer('iota', torch.arange(self.K, dtype=torch.int64), persistent=False)

    def forward(self, coord_logits, existing_coords):
        # One TopK per axis (rows sorted independently) -> all candidates; decode bin -> value.
        top = torch.topk(coord_logits, self.K, dim=-1, largest=True, sorted=True).indices   # (B, 2, K)
        top_val = top.to(torch.float32) * self.inv                                          # (B, 2, K)
        # Duplicate when x AND y are both within threshold: prod over coord axis, sum over history.
        cand = top_val.transpose(1, 2)                                              # (B, K, 2) -> [x, y]
        within = ((cand.unsqueeze(2) - existing_coords.unsqueeze(1)).abs()
                  < self.dedup_threshold).to(torch.float32)                         # (B, K, N, 2)
        valid = within.prod(dim=-1).sum(dim=-1) == 0.0                              # (B, K)
        # First valid candidate, else the last attempt (host exhaustion fallback).
        masked = torch.where(valid, self.iota, torch.full_like(self.iota, self.K))  # (B, K)
        sel = masked.min(dim=-1, keepdim=True).values.clamp(max=self.K - 1)         # (B, 1)
        # One shared index gathers the chosen bin and value for both axes.
        sel2 = sel.unsqueeze(1).expand(-1, 2, -1)                                   # (B, 2, 1)
        coord_xy = torch.gather(top_val, 2, sel2).squeeze(-1)                       # (B, 2) decoded (x, y)
        coord_embed = self.fourier(torch.gather(top, 2, sel2).squeeze(-1))          # (B, 1, dim)
        return coord_embed, coord_xy


class SIZE_DECODE_ENCODER(torch.nn.Module):
    """Size post-process + Fourier re-embed in one ONNX graph: argmax the (h, w) bins,
    Fourier-encode them into the (batch, 1, dim) hidden state, and emit the decoded (h, w)
    gathered from the per-bin size table (process_sizes' bin -> size mapping)."""

    def __init__(self, encoder, bin_values):
        super().__init__()
        self.fourier = FOURIER_ENCODER(encoder, bin_values)
        self.register_buffer('size_value', bin_values.float(), persistent=False)   # (num_bins,)

    def forward(self, size_logits):
        bins = torch.argmax(size_logits, dim=-1)                                   # (B, 2) int64
        size_embed = self.fourier(bins)                                            # (B, 1, dim)
        size_hw = self.size_value[bins]                                            # (B, 2) decoded (h, w) — one gather
        return size_embed, size_hw


# ══════════════════════════════════════════════════════════════════════════════
# LLM_Vision — Falcon prefill builder for the static detection prompt
# ══════════════════════════════════════════════════════════════════════════════
class LLM_VISION(torch.nn.Module):
    """Build the image-dependent prefill prefix for the detection prompt: resize +
    normalize (folded into the projector), patchify, project, and concat the static
    head-token embeddings with the projected patches. The query tail is added at runtime."""

    def __init__(self, model, config, layout, image_resize, dynamic_shape=False, input_image_size=None):
        super().__init__()
        self.target_h = int(image_resize[0])
        self.target_w = int(image_resize[1])
        self.dynamic_shape = dynamic_shape
        # Static resize decision: a dynamic-shape graph accepts any raw H/W (always resize);
        # otherwise resize only when the locked raw input geometry (INPUT_IMAGE_SIZE) differs
        # from the export target (IMAGE_RESIZE). When they already match the interpolate is a
        # no-op and is dropped from the graph entirely.
        self.do_resize = bool(dynamic_shape) or (
            input_image_size is not None
            and (int(input_image_size[0]) != self.target_h or int(input_image_size[1]) != self.target_w)
        )
        self.patch = int(config.spatial_patch_size)
        self.grid_h = layout["grid_h"]
        self.grid_w = layout["grid_w"]
        self.num_patches = layout["num_patches"]
        self.patch_start = layout["patch_start"]
        self.patch_end = layout["patch_end"]
        self.dim = int(config.dim)

        # Fuse image normalization (1/255 then (x-mean)/std) into the projector: patchify +
        # linear is affine, so normalization folds into an effective weight + bias.
        projector = model.img_projector
        weight = projector.weight.data.float().clone()   # (dim, ph*pw*c)
        in_features = weight.shape[1]
        c = int(config.channel_size)
        ph = pw = self.patch
        # Per-input-feature mean/std in (ph, pw, c) order, c fastest.
        mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(1, 1, c).expand(ph, pw, c).reshape(-1)
        std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 1, c).expand(ph, pw, c).reshape(-1)
        scale = 1.0 / (255.0 * std)
        fused_weight = weight * scale.view(1, in_features)
        fused_bias = -(weight * (mean / std).view(1, in_features)).sum(dim=1)
        self.proj_weight = torch.nn.Parameter(fused_weight)
        self.proj_bias = torch.nn.Parameter(fused_bias)

        # Static head token embeddings only (BOS / image_cls / image_reg_*); the query tail is
        # embedded at runtime via LLM_Embed, so the query can change without re-exporting.
        with torch.no_grad():
            head_emb = model.tok_embeddings(layout["head_token_ids"]).float().unsqueeze(0)
        self.register_buffer("head_embeddings", head_emb, persistent=False)

    def forward(self, pixel_values):
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)          # (B, 3, H, W)
        pixel_values = pixel_values.float()
        if self.do_resize:
            pixel_values = F.interpolate(pixel_values, size=[self.target_h, self.target_w],
                                         mode='bilinear', align_corners=False)

        # Patchify: (B, C, gh*ph, gw*pw) -> (B, gh*gw, ph*pw*C) with C fastest.
        x = pixel_values.reshape(-1, 3, self.grid_h, self.patch, self.grid_w, self.patch)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.grid_h * self.grid_w, self.patch * self.patch * 3)
        patch_feats = F.linear(x, self.proj_weight, self.proj_bias)   # (B, num_patches, dim)

        if self.dynamic_shape:
            b = pixel_values.shape[0]
            # Dynamic graph: broadcast the static head embeddings up to the runtime batch.
            return torch.cat([
                self.head_embeddings.expand(b, -1, -1),
                patch_feats,
            ], dim=1)
        # Static single-image geometry: head buffer already batch-aligned (batch == 1).
        return torch.cat([
            self.head_embeddings,
            patch_feats,
        ], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# PREFILL_MASK — baked hybrid prefill attention mask (rotary is fused into LLM_Main)
# ══════════════════════════════════════════════════════════════════════════════
class PREFILL_MASK(torch.nn.Module):
    """Build the additive hybrid prefill attention mask for any prompt length:
    allowed = causal | image-block-bidirectional. For the single-image prompt (no
    left-pad, no mid-sequence eos) this equals the source mask. The image-block span
    [img_lo, img_hi) is fixed by the image geometry, so the mask is query-independent."""

    def __init__(self, model, config, layout, max_seq_len):
        super().__init__()
        self.img_lo = int(layout["img_lo"])
        self.img_hi = int(layout["img_hi"])
        self.register_buffer("seq_positions", torch.arange(max_seq_len, dtype=torch.int64), persistent=False)
        img_mask = (self.seq_positions >= self.img_lo) & (self.seq_positions < self.img_hi)
        self.register_buffer("is_img_pos", img_mask, persistent=False)
        self.register_buffer("zero_f", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_inf_f", torch.tensor(-float('inf'), dtype=torch.float32), persistent=False)

    def forward(self, ids_len, history_len):
        # Absolute query / key positions for this prefill chunk, then allowed = causal | img_bidir.
        # Build positions with arange (Range op) instead of slicing the baked seq_positions buffer
        # by a dynamic scalar bound: a scalar-bound Slice is mis-folded by BOTH the ORT optimizer
        # and onnxslim into a 2-D `ends` (runtime "Slice ... Ends must be a 1-D array"), while Range
        # stays a clean 1-D producer through both. This also prunes seq_positions from the graph.
        q_len = ids_len.reshape(())
        hist = history_len.reshape(())
        kv_seq_len = q_len + hist
        q_pos = torch.arange(q_len, dtype=torch.int64) + hist        # (ids_len,)
        k_pos = torch.arange(kv_seq_len, dtype=torch.int64)          # (kv_seq_len,)
        causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)            # (ids_len, kv_seq_len)
        q_img = torch.index_select(self.is_img_pos, 0, q_pos)
        k_img = torch.index_select(self.is_img_pos, 0, k_pos)
        img_bidir = q_img.unsqueeze(1) & k_img.unsqueeze(0)
        allowed = causal | img_bidir
        additive = torch.where(allowed, self.zero_f, self.neg_inf_f)
        return additive.unsqueeze(0).unsqueeze(0)                    # (1, 1, ids_len, kv_seq_len)


# ══════════════════════════════════════════════════════════════════════════════
# LLM_Main — full 22-layer Falcon transformer with compact-GQA KV cache
# ══════════════════════════════════════════════════════════════════════════════
class LLM_MAIN(torch.nn.Module):
    """Inlined 22-layer Falcon decoder stack with compact-GQA KV cache.

    Per layer: pre-attn RMSNorm, fused wqkv, per-head QK RMSNorm, temporal + spatial RoPE,
    grouped-query attention, attention-sink scaling, pre-FFN RMSNorm, squared-ReLU gate.
    The final RMSNorm weight and the detection heads' first projection are fused into one
    shared head matmul yielding [logits | coord_pre | size_pre]. Rotary cos/sin tables are
    baked in and gathered by a single `cache_position` input (no separate rotary graphs).
    Keys are stored rotated, values raw; optional Q8 / Q8_CUDA / Q4 storage via KVQuantizer.
    """

    def __init__(self, model, config, num_layers, layout, max_seq_len):
        super().__init__()
        self.config = config
        self.num_heads = int(config.n_heads)             # 16
        self.num_kv_heads = int(config.n_kv_heads)       # 8
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 2
        self.head_dim = int(config.head_dim)             # 64
        self.head_dim_half = self.head_dim // 2          # 32
        self.head_dim_quarter = self.head_dim // 4
        self.kv_pack_head_dim = self.head_dim // 8 if KV_QUANT_DTYPE == "ROTARY_Q4_CUDA" else self.head_dim_quarter
        self.hidden_size = int(config.dim)               # 768
        self.q_dim = self.num_heads * self.head_dim      # 1024
        self.kv_dim = self.num_kv_heads * self.head_dim  # 512
        self.qk_heads = self.num_heads + self.num_kv_heads  # 24 (Q + K heads, normed together)
        self.qk_dim = self.q_dim + self.kv_dim              # 1536 (fused QK projection width)
        self.num_layers = num_layers

        # Layer-count multipliers for indexing the flat KV input list.
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

        # KV cache dtype flags.
        self.kv_f16       = (KV_QUANT_DTYPE == "F16")
        self.kv_q8        = KV_QUANT_DTYPE in ("Q8", "ROTARY_Q8")
        self.kv_q8_cuda   = KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA")
        self.kv_q4        = KV_QUANT_DTYPE in ("ROTARY_Q4",)
        self.kv_q4_cuda   = KV_QUANT_DTYPE in ("ROTARY_Q4_CUDA",)
        self.kv_quantized = self.kv_q8 or self.kv_q8_cuda or self.kv_q4 or self.kv_q4_cuda
        self.kv_sym       = USE_SYM and self.kv_quantized
        self.kv_grouped   = (self.kv_q4 or self.kv_q4_cuda) or \
                            ((self.kv_q8 or self.kv_q8_cuda) and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < self.head_dim)

        # Sum-based RMSNorm: sum_norm(x) = mean_norm(x) / sqrt(D), the sqrt(D) folded into the
        # consuming weights (wqkv, w13, final_norm); eps scales by D.
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        float_eps = float(torch.finfo(torch.float32).eps)
        hidden_eps = self.hidden_size * float_eps
        qk_eps = self.head_dim * float_eps
        if PREVENT_F16_OVERFLOW:
            ov2 = float(self.overflow_scale.square())
            hidden_eps *= ov2
            qk_eps *= ov2
        self.register_buffer("hidden_eps", torch.tensor([hidden_eps], dtype=torch.float32), persistent=False)
        self.register_buffer("qk_eps", torch.tensor([qk_eps], dtype=torch.float32), persistent=False)
        # Attention scale (1/sqrt(head_dim)) folded into a head_dim^0.25 multiply on q and k,
        # so the score matmul needs no runtime scale.
        self.register_buffer("attn_qk_scale", torch.tensor([self.head_dim ** 0.25], dtype=torch.float32), persistent=False)

        # ── Fused rotary cos/sin caches (pre-shaped, gathered by cache_position) ──────────
        # Temporal cos/sin are baked for every position [0, max_seq_len); spatial cos/sin for the
        # prefill span plus one trailing identity row (decode tokens have no spatial rotation).
        # Both are pre-shaped to the GQA group layout (1, S, KVH, G, head_dim_half) and assembled
        # into one full-head_dim cos/sin table, so each layer rotates Q and K with one multiply-add.
        self.prefill_len = int(layout["tokens"].shape[0])
        base_temporal_pos = int(layout["pos_t"][-1].item())
        t_cos_full, t_sin_full = build_temporal_rope_tables(model, max_seq_len)   # (max_seq_len, head_dim_half)
        tpos = torch.empty(max_seq_len, dtype=torch.long)
        tpos[:self.prefill_len] = layout["pos_t"]
        decode_tpos = base_temporal_pos + 1 + torch.arange(max_seq_len - self.prefill_len, dtype=torch.long)
        tpos[self.prefill_len:] = decode_tpos.clamp_(max=max_seq_len - 1)
        # Temporal table is head-independent but pre-expanded across the (KVH, G) group axes so it
        # concatenates directly with the per-head spatial table (folded to head h -> (h // G, h % G)).
        t_cos_cache = t_cos_full[tpos].view(1, max_seq_len, 1, 1, self.head_dim_half).expand(
            1, max_seq_len, self.num_kv_heads, self.num_kv_groups, self.head_dim_half).contiguous()
        t_sin_cache = t_sin_full[tpos].view(1, max_seq_len, 1, 1, self.head_dim_half).expand(
            1, max_seq_len, self.num_kv_heads, self.num_kv_groups, self.head_dim_half).contiguous()
        s_cos_pf, s_sin_pf = build_spatial_rope_tables(model, layout["pos_hw"], layout["img_id_mask"])  # (prefill_len, H, head_dim_half)
        ident_cos = torch.ones(1, self.num_heads, self.head_dim_half)
        ident_sin = torch.zeros(1, self.num_heads, self.head_dim_half)
        s_cos_cache = torch.cat([s_cos_pf, ident_cos], dim=0).view(1, self.prefill_len + 1, self.num_kv_heads, self.num_kv_groups, self.head_dim_half)
        s_sin_cache = torch.cat([s_sin_pf, ident_sin], dim=0).view(1, self.prefill_len + 1, self.num_kv_heads, self.num_kv_groups, self.head_dim_half)
        t_cos_cache = t_cos_cache.float()
        t_sin_cache = t_sin_cache.float()
        s_cos_cache = s_cos_cache.float()
        s_sin_cache = s_sin_cache.float()

        # Prebuild full-head rotary caches at max_seq_len; decode rows reuse the spatial
        # identity row so forward needs only one gather per cache.
        s_cos_full = torch.empty(1, max_seq_len, self.num_kv_heads, self.num_kv_groups, self.head_dim_half, dtype=torch.float32)
        s_sin_full = torch.empty(1, max_seq_len, self.num_kv_heads, self.num_kv_groups, self.head_dim_half, dtype=torch.float32)
        s_cos_full[:, :self.prefill_len + 1] = s_cos_cache
        s_sin_full[:, :self.prefill_len + 1] = s_sin_cache
        if max_seq_len > (self.prefill_len + 1):
            tail_len = max_seq_len - (self.prefill_len + 1)
            s_cos_full[:, self.prefill_len + 1:] = s_cos_cache[:, self.prefill_len:self.prefill_len + 1].expand(
                1, tail_len, self.num_kv_heads, self.num_kv_groups, self.head_dim_half)
            s_sin_full[:, self.prefill_len + 1:] = s_sin_cache[:, self.prefill_len:self.prefill_len + 1].expand(
                1, tail_len, self.num_kv_heads, self.num_kv_groups, self.head_dim_half)

        self.register_buffer("full_cos_cache", torch.cat([t_cos_cache, s_cos_full], dim=-1).contiguous(), persistent=False)
        self.register_buffer("full_sin_cache", torch.cat([t_sin_cache, s_sin_full], dim=-1).contiguous(), persistent=False)

        # Head-dim channel permutation: interleaved-pair (GPT-J) -> [even|odd] half-split
        # layout within each 32-dim temporal/spatial half, so RoPE uses the flip trick.
        perm_half = torch.cat([torch.arange(0, self.head_dim_half, 2),
                               torch.arange(1, self.head_dim_half, 2)])
        perm_head = torch.cat([perm_half, self.head_dim_half + perm_half])   # (head_dim,)
        sqrt_dim = self.hidden_size ** 0.5

        # Per-head attention sinks for the 16 query heads, reshaped to the GQA (KVH, G) group
        # layout (head h == kvh*G + g) so they broadcast over the grouped lse (B, KVH, G, Sq).
        sinks = torch.stack([model.layers[str(i)].attention.sinks.data.float() for i in range(num_layers)])
        self.register_buffer("sinks", sinks.view(num_layers, 1, self.num_kv_heads, self.num_kv_groups, 1), persistent=False)

        # wqkv output-row permutation: reorder each Q/K head's head_dim channels; V identity.
        row_perm = torch.arange(self.q_dim + 2 * self.kv_dim)
        for h in range(self.num_heads):
            base = h * self.head_dim
            row_perm[base:base + self.head_dim] = base + perm_head
        for h in range(self.num_kv_heads):
            base = self.q_dim + h * self.head_dim
            row_perm[base:base + self.head_dim] = base + perm_head

        # Frozen projection weights per layer. The hidden sum-norm's sqrt(dim) is folded
        # into wqkv and w13; wqkv Q/K channels are permuted for the flip-based RoPE. The
        # interleaved gate/up rows of w13 are de-interleaved to contiguous [gate | up]
        # blocks so the FFN splits them with one contiguous Split (no strided Slice step=2).
        self.ffn_hidden = int(config.ffn_dim)
        gate_up_deinterleave = torch.cat([torch.arange(0, 2 * self.ffn_hidden, 2),
                                          torch.arange(1, 2 * self.ffn_hidden, 2)])
        self.wqkv = torch.nn.ParameterList()
        self.wo = torch.nn.ParameterList()
        self.w13 = torch.nn.ParameterList()
        self.w2 = torch.nn.ParameterList()
        for i in range(num_layers):
            layer = model.layers[str(i)]
            wqkv = layer.attention.wqkv.weight.data.float() * sqrt_dim
            wqkv = wqkv.index_select(0, row_perm)
            self.wqkv.append(torch.nn.Parameter(wqkv))
            self.wo.append(torch.nn.Parameter(layer.attention.wo.weight.data.float()))
            w13 = (layer.feed_forward.w13.weight.data.float() * sqrt_dim).index_select(0, gate_up_deinterleave)
            self.w13.append(torch.nn.Parameter(w13))
            self.w2.append(torch.nn.Parameter(layer.feed_forward.w2.weight.data.float()))

        # Final learned RMSNorm weight folded into its consumers: it is a per-channel diagonal
        # scale, so (x * g) @ W.T == x @ (W * g).T. sqrt(dim) is folded in here too.
        final_norm_vec = model.norm.weight.data.float() * sqrt_dim            # (hidden,)
        output_weight = model.output.weight.data.float() * final_norm_vec     # (vocab, hidden)
        self.vocab_size = int(output_weight.shape[0])

        # Inlined coord + size detection heads (bias-free squared-ReLU MLPs w2(relu(w1(x))^2)).
        # Both share dims and consume the same normed hidden: w1 concatenated and stacked under
        # the output projection (one shared head matmul -> [logits | coord_pre | size_pre]); w2
        # stacked into a batched (2, hidden, out) weight (one bmm after the squared-ReLU).
        coord_dec, size_dec = model.coord_decoder, model.size_decoder
        cw1 = coord_dec.w1.weight.data.float()       # (dec_hidden, hidden)
        sw1 = size_dec.w1.weight.data.float()
        cw2 = coord_dec.w2.weight.data.float()       # (out, dec_hidden)
        sw2 = size_dec.w2.weight.data.float()
        assert cw1.shape == sw1.shape and cw2.shape == sw2.shape, \
            "Inlined detection head requires coord/size decoders to share inner + output dims."
        assert config.coord_out_dim == config.size_out_dim, \
            "Inlined detection head assumes equal coord/size bins (one batched bmm + split)."
        self.bbox_dec_hidden = cw1.shape[0]
        self.bbox_bins = int(config.coord_out_dim) // 2
        # w1 (final-norm scale folded in) concatenated and stacked under the output projection.
        bbox_w1 = torch.cat([cw1, sw1], dim=0) * final_norm_vec.view(1, -1)   # (2*dec_hidden, hidden)
        self.head_weight = torch.nn.Parameter(torch.cat([output_weight, bbox_w1], dim=0))
        # Batched second projection, transposed per head for the bmm (dec_hidden -> out).
        self.bbox_w2 = torch.nn.Parameter(torch.stack([cw2.t(), sw2.t()], dim=0))   # (2, dec_hidden, out)

        # Separate quantizers: keys are full 16-head, values are compact 8-head.
        quant_kwargs = dict(
            head_dim=self.head_dim, num_kv_groups=self.num_kv_groups,
            is_q4=(self.kv_q4 or self.kv_q4_cuda), is_q8_cuda=(self.kv_q8_cuda or self.kv_q4_cuda),
            use_sym=self.kv_sym, use_hadamard=USE_HADAMARD, use_clip=USE_CLIP,
            clip_sigma=CLIP_SIGMA, use_shuffle=USE_SHUFFLE,
        )
        self.k_quantizer = KVQuantizer(num_kv_heads=self.num_heads, **quant_kwargs).eval()
        self.v_quantizer = KVQuantizer(num_kv_heads=self.num_kv_heads, **quant_kwargs).eval()

        # Output buffers.
        self.save_key   = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * num_layers
                self.save_v_bias = [None] * num_layers

    def _rms_norm(self, x, eps):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)

    def _rotate_half(self, x, batch_size, groups):
        """Flip-rotate both RoPE halves of Q (groups == G) or the compact K (groups == 1) in one
        op: view head_dim as (2 halves, 2 lo/hi, quarter) and flip the lo/hi axis."""
        x = x.view(batch_size, -1, self.num_kv_heads, groups, 2, 2, self.head_dim_quarter)
        x = x.flip(-2)
        return x.view(batch_size, -1, self.num_kv_heads, groups, self.head_dim)

    @staticmethod
    def _softmax_lse(scores):
        """Single max-subtracted pass producing both softmax probs and logsumexp."""
        m = scores.max(dim=-1, keepdim=True).values
        e = torch.exp(scores - m)
        denom = e.sum(dim=-1, keepdim=True)
        probs = e / denom
        lse = (m + torch.log(denom)).squeeze(-1)
        return probs, lse

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-3]
        cache_position = all_inputs[-2]  # (ids_len,) int32 — KV-cache positions of these tokens
        attention_mask = all_inputs[-1]  # (1, 1, S, kv_seq)
        batch_size = hidden_states.shape[0]

        # Fused rotary: one gather per cache from the prebuilt full-head tables.
        full_cos = torch.index_select(self.full_cos_cache, 1, cache_position)
        full_sin = torch.index_select(self.full_sin_cache, 1, cache_position)

        for i in range(self.num_layers):
            residual = hidden_states
            normed = self._rms_norm(hidden_states, self.hidden_eps)

            # Fused projection: reshape into head slots, then split compact QK from V on the head axis.
            qkv = F.linear(normed, self.wqkv[i])
            qkv = qkv.view(batch_size, -1, self.qk_heads + self.num_kv_heads, self.head_dim)
            # QK norm + folded head_dim^0.25 scale, run once on compact (Q 16 + K 8 heads).
            qk, v = torch.split(qkv, [self.qk_heads, self.num_kv_heads], dim=2)
            qk = self._rms_norm(qk, self.qk_eps) * self.attn_qk_scale

            # Q takes the explicit GQA (KVH, G) layout; compact K keeps a singleton group axis
            # (reshape-based GQA, no expand) and broadcasts up to G via the per-head cos/sin multiply.
            q, k = torch.split(qk, [self.num_heads, self.num_kv_heads], dim=2)
            q = q.view(batch_size, -1, self.num_kv_heads, self.num_kv_groups, self.head_dim)
            k = k.view(batch_size, -1, self.num_kv_heads, 1, self.head_dim)
            # Rotate Q and K each with one fused multiply-add; K's multiply by the per-head cos/sin
            # broadcasts the compact K up to the G groups (distinct full-16 rotated key, no expand).
            q = q * full_cos + self._rotate_half(q, batch_size, self.num_kv_groups) * full_sin
            k = k * full_cos + self._rotate_half(k, batch_size, 1) * full_sin   # (B, seq, KVH, G, head_dim)

            # Cache layout: K full-16 (B, 16, 1, head_dim, S); V compact-8 (B, 8, 1, S, head_dim).
            # Q stays grouped (B, KVH, G, Sq, head_dim) through the whole attention.
            k = k.permute(0, 2, 3, 4, 1).reshape(batch_size, self.num_heads, 1, self.head_dim, -1)
            v = v.permute(0, 2, 1, 3).unsqueeze(2)                # (B, 8, 1, S, head_dim)
            q = q.permute(0, 2, 3, 1, 4)                          # (B, KVH, G, Sq, head_dim)

            attn = self._attention(i, q, k, v, all_inputs, attention_mask, batch_size, -1)

            # Fold the grouped (KVH, G) axes back to the 16-head output order (head == kvh*G + g).
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.q_dim)
            hidden_states = residual + F.linear(attn, self.wo[i])

            # Feed-forward: RMSNorm, squared-ReLU gate, down projection. w13's gate/up rows
            # were de-interleaved at init, so one contiguous split replaces the strided slices.
            residual = hidden_states
            normed = self._rms_norm(hidden_states, self.hidden_eps)
            w13_out = F.linear(normed, self.w13[i])
            gate, up = w13_out.split(self.ffn_hidden, dim=-1)
            hidden_states = residual + F.linear(F.relu(gate).square() * up, self.w2[i])

        # Final RMSNorm over the last token only (per-position norm, so the rest is never read);
        # the learned final-norm scale is folded into the head weight, leaving only the rsqrt.
        normed = self._rms_norm(hidden_states[:, -1:], self.hidden_eps).squeeze(1)
        # Shared head matmul -> [logits | coord_pre | size_pre]; split off the vocab logits.
        head_out = F.linear(normed, self.head_weight)
        logits, bbox_pre = head_out.split([self.vocab_size, 2 * self.bbox_dec_hidden], dim=-1)
        # Detection tail: squared-ReLU, one batched bmm over both heads, then reshape to
        # (2, B, 2, bins) and unbind into coord / size. The explicit reshape keeps `bins` a
        # concrete output dim (the host reads it from the output meta); chunk()/batch-split()
        # would make it symbolic or bake the export batch (verified in _validation).
        mid = F.relu(bbox_pre).square().view(-1, 2, self.bbox_dec_hidden).transpose(0, 1)  # (2, B, dec_hidden)
        bbox_out = torch.bmm(mid, self.bbox_w2).reshape(2, -1, 2, self.bbox_bins)          # (2, B, 2, bins)
        coord_logits, size_logits = bbox_out.unbind(0)                                     # each (B, 2, bins)

        if self.kv_sym:
            return (*self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale, logits, coord_logits, size_logits)
        elif self.kv_quantized:
            return (*self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits, coord_logits, size_logits)
        return (*self.save_key, *self.save_value, logits, coord_logits, size_logits)

    def _attention(self, i, q, k, v, all_inputs, attention_mask, batch_size, seq):
        """Grouped-query attention with attention-sink scaling. The full-16 key cache is viewed
        as (B, KVH, G, head_dim, S) for the score matmul (real group axis, no expand) and the
        compact-8 value cache broadcasts over G; scores/probs/lse stay grouped throughout."""
        if self.kv_f16:
            k = k.half()
            v = v.half()

        if not self.kv_quantized:
            k = torch.cat((all_inputs[i], k), dim=-1)            # (B, 16, 1, head_dim, S) cache IO
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)  # (B, 8, 1, S, head_dim) cache IO
            self.save_key[i] = k
            self.save_value[i] = v
            # Only the F16 cache needs an upcast; the F32 cache is already float.
            if self.kv_f16:
                k = k.float()
                v = v.float()
            # View the full-16 key cache as a real (KVH, G) group axis; compact V broadcasts over G.
            k = k.view(batch_size, self.num_kv_heads, self.num_kv_groups, self.head_dim, -1)
            scores = torch.matmul(q, k) + attention_mask         # (B, KVH, G, Sq, S)
            probs, lse = self._softmax_lse(scores)               # probs (B,KVH,G,Sq,S), lse (B,KVH,G,Sq)
            out = torch.matmul(probs, v)                         # (B, KVH, G, Sq, head_dim)
        else:
            out, lse = self._attention_quantized(i, q, k, v, all_inputs, attention_mask, batch_size, seq)

        # Attention-sink scaling: output * sigmoid(lse - sinks[h]) per query head (grouped).
        sink_scale = torch.sigmoid(lse - self.sinks[i])          # (B, KVH, G, Sq)
        return out * sink_scale.unsqueeze(-1)

    def _attention_quantized(self, i, q, k_new, v_new, all_inputs, attention_mask, batch_size, seq):
        """Q8 / Q8_CUDA / Q4 cache path on the full-16 rotated key cache and compact-8
        value cache. Keys are already rotated, so no rotary fusion is applied here."""
        # Value scale/bias seq axis: dim -3 grouped (B,8,1,S,G,1) vs -2 non-grouped (B,8,1,S,1).
        v_scale_cat_dim = -3 if self.kv_grouped else -2
        if self.kv_sym:
            packed_k, scale_k, _ = self.k_quantizer.quantize_keys(k_new, batch_size, self.num_heads, self.kv_pack_head_dim)
            packed_v, scale_v, _ = self.v_quantizer.quantize_values(v_new, batch_size, self.num_kv_heads, self.kv_pack_head_dim)
            k = torch.cat([all_inputs[i], packed_k], dim=-1)
            v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
            k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
            v_s = torch.cat([all_inputs[i + self.num_layers_3], scale_v], dim=v_scale_cat_dim)
            self.save_key[i] = k
            self.save_value[i] = v
            self.save_k_scale[i] = k_s
            self.save_v_scale[i] = v_s
            if USE_FLOAT16_SCALE_BIAS:
                k_s = k_s.float()
                v_s = v_s.float()
            k_code = self._decode_key_storage(k, batch_size)
            v_code = self._decode_value_storage(v, batch_size)
        else:
            packed_k, scale_k, bias_k = self.k_quantizer.quantize_keys(k_new, batch_size, self.num_heads, self.kv_pack_head_dim)
            packed_v, scale_v, bias_v = self.v_quantizer.quantize_values(v_new, batch_size, self.num_kv_heads, self.kv_pack_head_dim)
            k = torch.cat([all_inputs[i], packed_k], dim=-1)
            v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
            k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
            k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k], dim=-1)
            v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=v_scale_cat_dim)
            v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=v_scale_cat_dim)
            self.save_key[i] = k
            self.save_value[i] = v
            self.save_k_scale[i] = k_s
            self.save_k_bias[i] = k_b
            self.save_v_scale[i] = v_s
            self.save_v_bias[i] = v_b
            if USE_FLOAT16_SCALE_BIAS:
                k_s = k_s.float(); k_b = k_b.float(); v_s = v_s.float(); v_b = v_b.float()
            k_code = self._decode_key_storage(k, batch_size)
            v_code = self._decode_value_storage(v, batch_size)

        scores = self._quantized_scores(q, k_code, k_s, None if self.kv_sym else k_b, attention_mask, batch_size, seq)
        probs, lse = self._softmax_lse(scores)
        out = self._quantized_context(probs, v_code, v_s, None if self.kv_sym else v_b, batch_size, seq)
        return out, lse

    def _decode_key_storage(self, k, batch_size):
        """Decode packed KV storage to raw quantized key codes, without dequant scales."""
        quant = self.k_quantizer
        if self.kv_q8_cuda or self.kv_q4_cuda:
            k = quant.unpack_cuda(k, -2, batch_size, self.num_heads, self.head_dim if not self.kv_q4_cuda else self.head_dim // 2)
        if self.kv_q4 or self.kv_q4_cuda:
            k = quant.unpack_q4_k(k, batch_size)
            return quant._decode_signed_q4_storage(k).float() if self.kv_sym else k.float()
        if self.kv_sym:
            return quant._decode_signed_q8_storage(k).float()
        return k.float()

    def _decode_value_storage(self, v, batch_size):
        """Decode packed KV storage to raw quantized value codes, without dequant scales."""
        quant = self.v_quantizer
        if self.kv_q8_cuda or self.kv_q4_cuda:
            v = quant.unpack_cuda(v, -1, batch_size, self.num_kv_heads, self.head_dim if not self.kv_q4_cuda else self.head_dim // 2)
        if self.kv_q4 or self.kv_q4_cuda:
            v = quant.unpack_q4_v(v, batch_size)
            return quant._decode_signed_q4_storage(v).float() if self.kv_sym else v.float()
        if self.kv_sym:
            return quant._decode_signed_q8_storage(v).float()
        return v.float()

    def _quantized_scores(self, q, k, k_s, k_b, attention_mask, batch_size, seq):
        """Compute QK scores directly in the stored quantized basis: the key codes and their
        scale/bias are reshaped to (B, KVH, G, ...) so the scores stay grouped."""
        if self.kv_grouped:
            quant = self.k_quantizer
            q_g = q
            if quant.use_shuffle:
                q_g = q_g.index_select(-1, quant.shuffle_idx)
            q_g = q_g.reshape(batch_size, self.num_kv_heads, self.num_kv_groups, seq, quant.kv_quant_num_groups, quant.kv_quant_group_size)
            q_g = q_g.transpose(-2, -3)
            if quant.use_hadamard:
                q_g = quant.hadamard_q(q_g)

            k_g = k.reshape(batch_size, self.num_kv_heads, self.num_kv_groups, quant.kv_quant_num_groups, quant.kv_quant_group_size, -1)
            k_s_g = k_s.reshape(batch_size, self.num_kv_heads, self.num_kv_groups, quant.kv_quant_num_groups, 1, -1)
            attn_raw_g = torch.matmul(q_g, k_g)
            if self.kv_sym:
                scores = (attn_raw_g * k_s_g).sum(dim=-3)
            else:
                k_b_g = k_b.reshape(batch_size, self.num_kv_heads, self.num_kv_groups, quant.kv_quant_num_groups, 1, -1)
                q_sum_g = q_g.sum(dim=-1, keepdim=True)
                scores = (attn_raw_g * k_s_g + q_sum_g * k_b_g).sum(dim=-3)
            return scores + attention_mask

        # Non-grouped: view the key codes / scale / bias as (B, KVH, G, ...) so scores stay grouped.
        k_g = k.reshape(batch_size, self.num_kv_heads, self.num_kv_groups, self.head_dim, -1)
        attn_raw = torch.matmul(q, k_g)
        k_s_g = k_s.reshape(batch_size, self.num_kv_heads, self.num_kv_groups, 1, -1)
        if self.kv_sym:
            scores = attn_raw * k_s_g
        else:
            k_b_g = k_b.reshape(batch_size, self.num_kv_heads, self.num_kv_groups, 1, -1)
            q_sum = q.sum(dim=-1, keepdim=True)
            scores = torch.addcmul(q_sum * k_b_g, attn_raw, k_s_g)
        return scores + attention_mask

    def _quantized_context(self, probs, v, v_s, v_b, batch_size, seq):
        """Apply grouped attention probabilities (B, KVH, G, Sq, S) to the stored quantized value
        basis; the output stays grouped (B, KVH, G, Sq, head_dim) (compact V broadcasts over G)."""
        if self.kv_grouped:
            quant = self.v_quantizer
            v_g = v.view(batch_size, self.num_kv_heads, 1, -1, quant.kv_quant_num_groups, quant.kv_quant_group_size)
            v_dequant = (v_g * v_s).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim) if self.kv_sym \
                else (v_g * v_s + v_b).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            out = torch.matmul(probs, v_dequant)
            if quant.use_hadamard:
                out = quant.inverse_hadamard_attn(out, batch_size)
            if quant.use_shuffle:
                out = out.index_select(-1, quant.unshuffle_idx)
        else:
            v_dequant = (v * v_s) if self.kv_sym else torch.addcmul(v_b, v, v_s)
            out = torch.matmul(probs, v_dequant)
        return out

    def _dequant_key(self, k, k_s, k_b, batch_size):
        """Dequantize the stored key cache to (B, 16, 1, head_dim, S) float32."""
        quant = self.k_quantizer
        if self.kv_q8_cuda or self.kv_q4_cuda:
            k = quant.unpack_cuda(k, -2, batch_size, self.num_heads, self.head_dim if not self.kv_q4_cuda else self.head_dim // 2)
        if self.kv_q4 or self.kv_q4_cuda:
            k = quant.unpack_q4_k(k, batch_size)
            k = quant._decode_signed_q4_storage(k).float() if self.kv_sym else k.float()
        elif self.kv_sym:
            k = quant._decode_signed_q8_storage(k).float()
        else:
            k = k.float()
        if self.kv_grouped:
            kg = k.view(batch_size, self.num_heads, 1, quant.kv_quant_num_groups, quant.kv_quant_group_size, -1)
            k = (kg * k_s).reshape(batch_size, self.num_heads, 1, self.head_dim, -1) if self.kv_sym \
                else (kg * k_s + k_b).reshape(batch_size, self.num_heads, 1, self.head_dim, -1)
        else:
            k = (k * k_s) if self.kv_sym else (k * k_s + k_b)
        if quant.use_hadamard:
            # Undo the forward Hadamard (inverse, not forward).
            k = quant.hadamard_k(k, batch_size, inverse=True)
        if quant.use_shuffle:
            k = k.index_select(3, quant.unshuffle_idx)
        return k

    def _dequant_value(self, v, v_s, v_b, batch_size):
        """Dequantize the stored value cache to (B, 8, 1, S, head_dim) float32."""
        quant = self.v_quantizer
        if self.kv_q8_cuda or self.kv_q4_cuda:
            v = quant.unpack_cuda(v, -1, batch_size, self.num_kv_heads, self.head_dim if not self.kv_q4_cuda else self.head_dim // 2)
        if self.kv_q4 or self.kv_q4_cuda:
            v = quant.unpack_q4_v(v, batch_size)
            v = quant._decode_signed_q4_storage(v).float() if self.kv_sym else v.float()
        elif self.kv_sym:
            v = quant._decode_signed_q8_storage(v).float()
        else:
            v = v.float()
        if self.kv_grouped:
            vg = v.view(batch_size, self.num_kv_heads, 1, -1, quant.kv_quant_num_groups, quant.kv_quant_group_size)
            v = (vg * v_s).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim) if self.kv_sym \
                else (vg * v_s + v_b).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
        else:
            v = (v * v_s) if self.kv_sym else (v * v_s + v_b)
        if quant.use_hadamard:
            # Undo the forward Hadamard (inverse, not forward).
            v = quant.hadamard_v(v, batch_size, inverse=True)
        if quant.use_shuffle:
            v = v.index_select(-1, quant.unshuffle_idx)
        return v


# ══════════════════════════════════════════════════════════════════════════════
# Decoding utility heads (ported from LightOn, adapted to Falcon KV layout)
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Greedy decoding: pick the highest-logit token and append it to save_id."""

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class ARGMAX(torch.nn.Module):
    """Argmax over the vocabulary dimension."""

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class APPLY_PENALTY(torch.nn.Module):
    """Apply a repetition penalty to recently generated token logits."""

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized = logits.gather(1, target_indices) * penalty_value
        logits = logits.scatter(1, target_indices, penalized)
        return logits


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * total_layers
        self._ones_tuple = {d: (1,) * d for d in range(8)}

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp
        for i in range(self.total_layers):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *self._ones_tuple[kv.dim() - 1])
        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]
        return (*self.save_keys_values, save_id, top_beam_prob.transpose(0, 1), top_beam_indices, max_logits_idx)


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        top_k = all_inputs[-1]
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]
        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)
        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)
        return (*self.save_keys_values, save_id, top_beam_prob.unsqueeze(-1), top_beam_indices, max_logits_idx)


class KV_SLICE(torch.nn.Module):
    """Slice the compact KV cache (and any quantized side tensors) along the sequence axis."""

    def __init__(self, num_layers, head_dim=0):
        super().__init__()
        self.kv_quantized = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_grouped = (KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")) or \
                          (KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim)
        self.kv_sym = USE_SYM and self.kv_quantized
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * num_layers
                self.save_v_bias = [None] * num_layers

    def forward(self, *all_inputs):
        slice_start = all_inputs[-2]
        slice_end = all_inputs[-1]
        for i in range(self.num_layers):
            self.save_key[i] = all_inputs[i][..., slice_start:slice_end]                 # K: (B,16,1,D,S)
            self.save_value[i] = all_inputs[i + self.num_layers][..., slice_start:slice_end, :]  # V: (B,8,1,S,D)
            if self.kv_quantized:
                # Key scale/bias seq axis is last; value scale/bias seq axis is -3 grouped / -2 non-grouped.
                self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start:slice_end]
                if self.kv_sym:
                    if self.kv_grouped:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end, :, :]
                    else:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end, :]
                else:
                    self.save_k_bias[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end]
                    if self.kv_grouped:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start:slice_end, :, :]
                        self.save_v_bias[i] = all_inputs[i + self.num_layers_5][..., slice_start:slice_end, :, :]
                    else:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start:slice_end, :]
                        self.save_v_bias[i] = all_inputs[i + self.num_layers_5][..., slice_start:slice_end, :]
        if self.kv_sym:
            return (*self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale)
        if self.kv_quantized:
            return (*self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias)
        return (*self.save_key, *self.save_value)


# ══════════════════════════════════════════════════════════════════════════════
# Export driver
# ══════════════════════════════════════════════════════════════════════════════
def _kv_dtype_and_heads():
    """Storage dtype and per-tensor head dims for the Falcon compact KV cache."""
    head_dim = 64
    if KV_QUANT_DTYPE == "F16":
        return torch.float16, head_dim, head_dim
    if KV_QUANT_DTYPE == "F32":
        return torch.float32, head_dim, head_dim
    if KV_QUANT_DTYPE in ("Q8", "ROTARY_Q8"):
        return (torch.int8 if USE_SYM else torch.uint8), head_dim, head_dim
    if KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA"):
        return torch.int32, head_dim // 4, head_dim // 4
    if KV_QUANT_DTYPE == "ROTARY_Q4":
        return torch.uint8, head_dim // 2, head_dim // 2
    if KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
        return torch.int32, head_dim // 8, head_dim // 8
    return torch.float32, head_dim, head_dim


if DO_EXPORT:
    print('Export start ...')
    os.makedirs(EXPORT_DIR, exist_ok=True)
    with torch.inference_mode():
        model, tokenizer, config, proc_mod = load_falcon(download_path)
        layout = build_static_layout(model, tokenizer, config, proc_mod, IMAGE_RESIZE, TEST_QUERY)

        num_layers   = config.n_layers
        num_heads    = config.n_heads
        num_kv_heads = config.n_kv_heads
        head_dim     = config.head_dim
        hidden_size  = config.dim
        vocab_size   = config.vocab_size
        prefill_len  = int(layout["tokens"].shape[0])
        scale_dtype  = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32

        for note in normalize_kv_quant_settings(head_dim):
            print(f"\n{note}")

        kv_dtype, k_head, v_head = _kv_dtype_and_heads()
        is_quantized = KV_QUANT_DTYPE not in ("F16", "F32")
        is_grouped = (KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")) or \
                     (KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim)
        kv_groups = head_dim // KV_QUANT_GROUP_SIZE if is_grouped else 1

        # KV spec: (name, head_count, seq_axis). K full-16, V compact-8. Only the key scale/bias
        # seq axis shifts with grouping (6D grouped -> 5, 5D non-grouped -> 4).
        key_scale_dim = 5 if is_grouped else 4
        kv_specs = [('key', num_heads, 4), ('value', num_kv_heads, 3)]
        if is_quantized:
            if USE_SYM:
                kv_specs += [('key_scale', num_heads, key_scale_dim), ('value_scale', num_kv_heads, 3)]
            else:
                kv_specs += [('key_scale', num_heads, key_scale_dim), ('key_bias', num_heads, key_scale_dim),
                             ('value_scale', num_kv_heads, 3), ('value_bias', num_kv_heads, 3)]

        dummy_history = 0
        beam = BEAM_SIZE

        def make_kv_tensor(name, heads):
            if name == 'key':
                return torch.zeros((beam, heads, 1, k_head, dummy_history), dtype=kv_dtype)
            if name == 'value':
                return torch.zeros((beam, heads, 1, dummy_history, v_head), dtype=kv_dtype)
            if name in ('key_scale', 'key_bias'):
                if is_grouped:
                    return torch.ones((beam, heads, 1, kv_groups, 1, dummy_history), dtype=scale_dtype)
                return torch.ones((beam, heads, 1, 1, dummy_history), dtype=scale_dtype)
            # value_scale / value_bias
            if is_grouped:
                return torch.ones((beam, heads, 1, dummy_history, kv_groups, 1), dtype=scale_dtype)
            return torch.ones((beam, heads, 1, dummy_history, 1), dtype=scale_dtype)

        kv_tensors = {name: make_kv_tensor(name, heads) for name, heads, _ in kv_specs}

        def get_kv_io(batch_axis='batch_size', seq_axis='history_len', out_seq_axis='kv_seq_len'):
            inputs, in_names, out_names, axes = [], [], [], {}
            for name, heads, dim in kv_specs:
                tensor = kv_tensors[name]
                for i in range(num_layers):
                    in_n = f'in_{name}_{i}'
                    out_n = f'out_{name}_{i}'
                    inputs.append(tensor)
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n] = {0: batch_axis, dim: seq_axis}
                    axes[out_n] = {0: batch_axis, dim: out_seq_axis}
            return inputs, in_names, out_names, axes

        # ── LLM_Embed ──────────────────────────────────────────────────────
        input_ids = torch.ones((1, 8), dtype=torch.int32)
        torch.onnx.export(
            LLM_EMBED(model), (input_ids,), onnx_model_Embed,
            input_names=['input_ids'], output_names=['hidden_states'],
            dynamic_axes={'input_ids': {0: 'batch', 1: 'ids_len'}, 'hidden_states': {0: 'batch', 1: 'ids_len'}},
            opset_version=OPSET, dynamo=False,
        )
        del input_ids

        # ── LLM_Vision (static detection-prompt prefill builder) ───────────
        # The raw H/W axes live at 3,4 for the 5-D [B,1,3,H,W] input and 2,3 for the 4-D
        # [B,3,H,W] input. When DYNAMIC_VISION_SHAPE is True they (and the batch axis) are
        # declared dynamic so the graph accepts any raw image size; the in-graph interpolate
        # still normalizes every image to IMAGE_RESIZE, so the patch grid / prefill_len /
        # rotary tables consumed downstream stay fixed (concat_len is constant either way).
        if INPUT_IMAGE_DIM == 5:
            image_input = torch.zeros((VISION_BATCH_SIZE, 1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8)
            vision_h_axis, vision_w_axis = 3, 4
        else:
            image_input = torch.zeros((VISION_BATCH_SIZE, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8)
            vision_h_axis, vision_w_axis = 2, 3
        vision_dynamic_axes = {
            'pixel_values': {0: 'batch', vision_h_axis: 'height', vision_w_axis: 'width'},
            'concat_hidden_states': {0: 'batch', 1: 'concat_len'},
        } if DYNAMIC_VISION_SHAPE else None
        torch.onnx.export(
            LLM_VISION(model, config, layout, IMAGE_RESIZE, dynamic_shape=DYNAMIC_VISION_SHAPE,
                       input_image_size=INPUT_IMAGE_SIZE),
            (image_input,), onnx_model_Vision,
            input_names=['pixel_values'], output_names=['concat_hidden_states'],
            dynamic_axes=vision_dynamic_axes,
            opset_version=OPSET, dynamo=False,
        )
        del image_input

        # ── Prefill_Mask (hybrid prefill attention mask; rotary is fused into LLM_Main) ─────
        torch.onnx.export(
            PREFILL_MASK(model, config, layout, MAX_SEQ_LEN),
            (torch.tensor([prefill_len], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)),
            onnx_model_Prefill_Mask,
            input_names=['ids_len', 'history_len'],
            output_names=['attention_mask'],
            dynamic_axes={'attention_mask': {2: 'ids_len', 3: 'kv_seq_len'}},
            opset_version=OPSET, dynamo=False,
        )

        # ── LLM_Main ───────────────────────────────────────────────────────
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io()
        dummy_seq = 8
        hidden_states = torch.ones((beam, dummy_seq, hidden_size), dtype=torch.float32)
        cache_position = torch.arange(dummy_seq, dtype=torch.int32)
        attention_mask = torch.zeros((1, 1, dummy_seq, dummy_seq), dtype=torch.float32)

        all_inputs = kv_ins + [hidden_states, cache_position, attention_mask]
        input_names = kv_in_names + ['hidden_states', 'cache_position', 'attention_mask']
        output_names = kv_out_names + ['logits', 'coord_logits', 'size_logits']
        dynamic_axes = {
            **kv_axes,
            'hidden_states':  {0: 'batch', 1: 'ids_len'},
            'logits':         {0: 'batch'},
            'coord_logits':   {0: 'batch'},
            'size_logits':    {0: 'batch'},
            'cache_position': {0: 'ids_len'},
            'attention_mask': {2: 'ids_len', 3: 'kv_seq_len'},
        }
        model_Main = LLM_MAIN(model, config, num_layers, layout, MAX_SEQ_LEN)
        torch.onnx.export(
            model_Main, tuple(all_inputs), onnx_model_Main,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=OPSET, dynamo=False,
        )
        del model_Main, hidden_states, attention_mask, all_inputs
        gc.collect()

        # ── Coord / Size decode-encoders (argmax + dedup + process_sizes + Fourier, in-graph) ──
        coord_logits_in = torch.zeros((1, 2, config.coord_out_dim // 2), dtype=torch.float32)
        size_logits_in = torch.zeros((1, 2, config.size_out_dim // 2), dtype=torch.float32)
        existing_coords_in = torch.zeros((1, 3, 2), dtype=torch.float32)
        torch.onnx.export(
            COORD_DECODE_ENCODER(model.coord_encoder, fourier_bin_values(config.coord_out_dim, is_size=False),
                                 MAX_COORD_ATTEMPTS, COORD_DEDUP_THRESHOLD),
            (coord_logits_in, existing_coords_in), onnx_model_Coord_Encoder,
            input_names=['coord_logits', 'existing_coords'], output_names=['coord_embed', 'coord_xy'],
            dynamic_axes={'coord_logits': {0: 'batch'}, 'existing_coords': {0: 'batch', 1: 'num_existing'},
                          'coord_embed': {0: 'batch'}, 'coord_xy': {0: 'batch'}},
            opset_version=OPSET, dynamo=False,
        )
        torch.onnx.export(
            SIZE_DECODE_ENCODER(model.size_encoder, fourier_bin_values(config.size_out_dim, is_size=True)),
            (size_logits_in,), onnx_model_Size_Encoder,
            input_names=['size_logits'], output_names=['size_embed', 'size_hw'],
            dynamic_axes={'size_logits': {0: 'batch'}, 'size_embed': {0: 'batch'}, 'size_hw': {0: 'batch'}},
            opset_version=OPSET, dynamo=False,
        )

        # The squared-ReLU bbox MLPs are fused into LLM_Main; their post-process lives in the
        # decode-encoder graphs above, so no host numpy decode remains.
        del coord_logits_in, size_logits_in, existing_coords_in

        # ── Greedy / Argmax / Penalty ──────────────────────────────────────
        logits = torch.ones((beam, vocab_size), dtype=torch.float32)
        save_id_in = torch.zeros((beam, 8), dtype=torch.int32)
        torch.onnx.export(
            GREEDY_SEARCH(), (logits, save_id_in), onnx_model_Greedy,
            input_names=['logits', 'save_id_in'], output_names=['max_logits_idx', 'save_id_out'],
            dynamic_axes={'logits': {0: 'batch'}, 'save_id_in': {0: 'batch', 1: 'history_len'},
                          'save_id_out': {0: 'batch', 1: 'history_len'}, 'max_logits_idx': {0: 'batch'}},
            opset_version=OPSET, dynamo=False,
        )
        torch.onnx.export(
            ARGMAX(), (logits,), onnx_model_Argmax,
            input_names=['logits'], output_names=['max_logits_idx'],
            dynamic_axes={'logits': {0: 'batch'}, 'max_logits_idx': {0: 'batch'}},
            opset_version=OPSET, dynamo=False,
        )
        penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALTY_RANGE], dtype=torch.int64)
        torch.onnx.export(
            APPLY_PENALTY(), (logits, save_id_in, penalty_value, penalty_range), onnx_model_Penalty,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'], output_names=['logits_out'],
            dynamic_axes={'logits_in': {0: 'batch'}, 'save_id_in': {0: 'batch', 1: 'history_len'}, 'logits_out': {0: 'batch'}},
            opset_version=OPSET, dynamo=False,
        )

        # ── First / Second beam search ─────────────────────────────────────
        num_layers_beam = num_layers * len(kv_specs)
        beam_size_t = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        topK = torch.tensor([TOP_K], dtype=torch.int64)
        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)

        kv_tensors_first = {name: t[[0]] for name, t in kv_tensors.items()}
        first_ins, first_in_names, first_out_names, first_axes = [], [], [], {}
        for name, heads, dim in kv_specs:
            tensor = kv_tensors_first[name]
            for i in range(num_layers):
                in_n = f'in_{name}_{i}'
                first_ins.append(tensor)
                first_in_names.append(in_n)
                first_out_names.append(f'out_{name}_{i}')
                first_axes[in_n] = {0: 'batch_size', dim: 'history_len'}
        torch.onnx.export(
            FIRST_BEAM_SEARCH(num_layers_beam),
            tuple(first_ins + [logits[[0]], save_id_in, beam_size_t]),
            onnx_model_First_Beam,
            input_names=first_in_names + ['logits', 'save_id_in', 'beam_size'],
            output_names=first_out_names + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={**first_axes, 'logits': {0: 'batch'}, 'save_id_in': {0: 'batch', 1: 'history_len'},
                          'top_beam_prob': {0: 'batch'}, 'top_beam_indices': {0: 'batch'},
                          'max_logits_idx': {0: 'batch'}, 'save_id_out': {0: 'batch', 1: 'history_len'}},
            opset_version=OPSET, dynamo=False,
        )

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io()
        torch.onnx.export(
            SECOND_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + [logits, save_id_in, previous_prob, beam_size_t, topK]),
            onnx_model_Second_Beam,
            input_names=kv_in_names + ['logits', 'save_id_in', 'previous_prob', 'beam_size', 'topK'],
            output_names=kv_out_names + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={**kv_axes, 'logits': {0: 'batch'}, 'save_id_in': {0: 'batch', 1: 'history_len'},
                          'previous_prob': {0: 'batch'}, 'save_id_out': {0: 'batch', 1: 'history_len'},
                          'top_beam_prob': {0: 'batch'}, 'top_beam_indices': {0: 'batch'}, 'max_logits_idx': {0: 'batch'}},
            opset_version=OPSET, dynamo=False,
        )
        del previous_prob

        # ── KV_Slice ───────────────────────────────────────────────────────
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(seq_axis='history_len', out_seq_axis='sliced_len')
        slice_start = torch.tensor([0], dtype=torch.int64)
        slice_end = torch.tensor([5], dtype=torch.int64)
        torch.onnx.export(
            KV_SLICE(num_layers, head_dim),
            tuple(kv_ins + [slice_start, slice_end]),
            onnx_model_KV_Slice,
            input_names=kv_in_names + ['slice_start', 'slice_end'], output_names=kv_out_names,
            dynamic_axes=kv_axes, opset_version=OPSET, dynamo=False,
        )
        del slice_start, slice_end, logits, save_id_in
        gc.collect()

    print(
        '\nExport done!\n\n'
        'Start running Falcon Perception by ONNXRuntime.\n'
        'Now loading . . . it could cost minutes.'
    )


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


def load_image_for_export(image_path, target_h, target_w, dim5):
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


# ── ONNX Runtime I/O-binding helpers (zero-copy decode runtime) ──────────────
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
    """Run a session through its pre-populated I/O binding."""
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
    """Build a zero-history KV input OrtValue from an LLM_Main input meta: batch dim -> 1,
    the first dynamic (KV history) axis -> 0, every other dim taken from the static meta."""
    shape = [d if isinstance(d, int) else 1 for d in meta.shape]
    for d in range(1, len(meta.shape)):
        if not isinstance(meta.shape[d], int):
            shape[d] = 0
            break
    return create_ort_with_shape(tuple(shape), np_dtype_from_ort(meta.type), device, device_id)


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS (LightOn-style)
# ══════════════════════════════════════════════════════════════════════════════
from PIL import Image, ImageColor, ImageDraw, ImageFont  # noqa: E402  (imported here so export can run on headless setups)

# Palette-derived constants precomputed once: RGB tuples (handed straight to PIL, no per-draw
# hex re-parse) and their contrast text colors (folds ImageColor.getrgb + _vis_text_color out
# of the per-detection draw loop).
VIS_PALETTE_RGB = tuple(ImageColor.getrgb(color) for color in VIS_PALETTE)
VIS_TEXT_COLOR = tuple(_vis_text_color(rgb) for rgb in VIS_PALETTE_RGB)

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

packed_settings = {
    "_session_opts": session_opts,
    "_providers": ORT_Accelerate_Providers,
    "_provider_options": provider_options,
    "_disabled_optimizers": disabled_optimizers,
}

# Device/memory location for zero-copy OrtValue allocation. KV stays on CPU for DML only.
_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type


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

print(f"\nUsable Providers: {ort_session_Main.get_providers()}")

# One reusable io_binding() per session for the zero-copy decode runtime.
binding_Vision = ort_session_Vision.io_binding()
binding_Embed = ort_session_Embed.io_binding()
binding_Prefill_Mask = ort_session_Prefill_Mask.io_binding()
binding_Main = ort_session_Main.io_binding()
binding_Argmax = ort_session_Argmax.io_binding()
binding_Coord_Encoder = ort_session_Coord_Encoder.io_binding()
binding_Size_Encoder = ort_session_Size_Encoder.io_binding()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
in_name_Vision = get_in_names(ort_session_Vision)[0]
out_name_Vision = get_out_names(ort_session_Vision)[0]

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
    from transformers import AutoConfig
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
# Per-step KV-cache position of the new decode token (num_prefill + step), host-tracked so
# LLM_Main gathers the right rotary row with no device-side counter. Prefill uses an arange.
cache_position_buf = create_ort_with_shape((1,), cache_position_dtype_Main, device_type, DEVICE_ID)
# Constant zero decode mask: the decode row attends to all keys, so a (1, 1, 1, 1) zero
# additive mask broadcasts over the score matrix with no per-step alloc.
decode_mask_buf = create_ort_with_shape((1, 1, 1, 1), mask_dtype_Main, device_type, DEVICE_ID)
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


# ══════════════════════════════════════════════════════════════════════════════
# PREFILL PHASE
# ══════════════════════════════════════════════════════════════════════════════
# Static vision input -> resize on load to the graph's fixed H/W (meta ints); dynamic vision
# input -> load the native image (vision_static_h/w are None, so no resize) and let LLM_Vision
# resize to the fixed export geometry in-graph.
pixel_values = load_image_for_export(TEST_IMAGE[0], vision_static_h, vision_static_w, INPUT_IMAGE_DIM == 5)

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

# 2. Hybrid prefill attention mask; prefill cache positions are the contiguous [0, num_prefill) range.
ids_len = create_ort_with_data([num_prefill], prefill_len_dtype, device_type, DEVICE_ID)
init_history_len = create_ort_with_data([0], prefill_hist_dtype, device_type, DEVICE_ID)
bind_ort_in_buf(binding_Prefill_Mask, in_name_Prefill_Mask, [ids_len, init_history_len])
bind_ort_out(binding_Prefill_Mask, out_name_Prefill_Mask, _ort_device_type)
run(ort_session_Prefill_Mask, binding_Prefill_Mask)
attention_mask_p = binding_Prefill_Mask.get_outputs()[0]
cache_position_prefill = create_ort_with_data(list(range(num_prefill)), cache_position_dtype_Main, device_type, DEVICE_ID)

# 3. Pre-bind the always-reused decode sessions. Text branch is fully on-device:
#    Argmax token id -> Embed -> hidden_states_buf -> Main.
binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)
binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)
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
bind_ort_in_buf(binding_Main, in_name_Main_others, [hidden_states_prefill, cache_position_prefill, attention_mask_p])
bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
binding_Main.bind_ortvalue_output(out_name_Main_logits, logits_buf)
binding_Main.bind_ortvalue_output(out_name_Main_coord_logits, coord_logits_buf)
binding_Main.bind_ortvalue_output(out_name_Main_size_logits, size_logits_buf)
run(ort_session_Main, binding_Main)
kv_outputs = binding_Main.get_outputs()[:num_kv_io]

decode_start_time = time.time()
prefill_elapsed = decode_start_time - prefill_start_time

# One-time prefill -> decode transition: rebind LLM_Main's non-KV inputs to the reusable decode
# buffers (decode_mask_buf, cache_position_buf); outputs stay bound, KV is rebound per step.
bind_ort_in_buf(binding_Main, in_name_Main_others, [hidden_states_buf, cache_position_buf, decode_mask_buf])


# ══════════════════════════════════════════════════════════════════════════════
# DECODE LOOP (detection: coord / size decoding + Fourier-encoder feedback)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\nTest Query: {TEST_QUERY}\nDetecting . . .')

aux_output = []                 # interleaved coord / size dicts
detections = []                 # final list of {"xy": ..., "hw": ...}
coord_count = 0                 # decoded coords written into existing_coords_buf
pending_xy = None

_DEBUG_DECODE = bool(int(os.environ.get('DEBUG_DECODE', '0')))
_decode_token_log = []

num_decode = 0
generate_limit = min(MAX_NEW_TOKENS, MAX_SEQ_LEN - num_prefill)
if _DEBUG_DECODE and int(os.environ.get('DEBUG_MAX_TOKENS', '0')):
    generate_limit = min(generate_limit, int(os.environ['DEBUG_MAX_TOKENS']))

while num_decode < generate_limit:
    # ── 1. Sample the next token (greedy) from the current logits buffer ──
    run(ort_session_Argmax, binding_Argmax)
    next_token = int(max_idx_buf.numpy().flat[0])
    if _DEBUG_DECODE:
        _decode_token_log.append(next_token)
    if next_token in STOP_TOKEN_SET:
        break

    # ── 2. Conditional feedback into hidden_states_buf: all three producers write the same
    #       buffer (so LLM_Main reads one fixed input) and run on device. ──
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

    # ── 3. LLM_Main decode step. cache_position_buf gets the new token's KV position so the
    #       rotary row is gathered; prior KV outputs feed back as inputs (zero-copy on device). ──
    cache_position_buf.update_inplace(np.asarray([num_prefill + num_decode], dtype=cache_position_dtype_Main))
    bind_ort_in_buf(binding_Main, in_name_Main_kv, kv_outputs)
    bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
    run(ort_session_Main, binding_Main)
    kv_outputs = binding_Main.get_outputs()[:num_kv_io]

    num_decode += 1
decode_end_time = time.time()

if _DEBUG_DECODE:
    _lab = {COORD_TOKEN_ID: 'COORD', SIZE_TOKEN_ID: 'SIZE', EOS_ID: 'EOS', END_OF_QUERY_ID: 'EOQ'}
    print(f"\n[DEBUG] num_prefill={num_prefill} decoded {len(_decode_token_log)} tokens, STOP_TOKEN_SET={STOP_TOKEN_SET}")
    print('[DEBUG] token stream:', ' '.join(_lab.get(t, str(t)) for t in _decode_token_log))


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
