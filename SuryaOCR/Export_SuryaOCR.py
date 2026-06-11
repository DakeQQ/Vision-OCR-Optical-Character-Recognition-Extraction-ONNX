import gc
import importlib
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from PIL import Image
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR                         = Path(__file__).resolve().parent
download_path                    = r"/home/DakeQQ/Downloads/surya-ocr-2"                 # Local Surya OCR 2 model path.
onnx_output_dir                  = BASE_DIR / "SuryaOCR_ONNX"                            # Directory to store exported ONNX models.
onnx_output_dir.mkdir(parents=True, exist_ok=True)

# ── ONNX Model Outputs ───────────────────────────────────────────────────────
onnx_model_Embed                 = str(onnx_output_dir / "LLM_Embed.onnx")               # Token embedding lookup.
onnx_model_Vision                = str(onnx_output_dir / "LLM_Vision.onnx")              # Fused vision: preprocess + encode + concat.
onnx_model_Rotary_Prefill        = str(onnx_output_dir / "Rotary_Prefill.onnx")          # mRoPE rotary embeddings for prefill phase.
onnx_model_Rotary_Decode         = str(onnx_output_dir / "Rotary_Decode.onnx")           # mRoPE rotary embeddings for decode step.
onnx_model_Main                  = str(onnx_output_dir / "LLM_Main.onnx")                # Main transformer (full + linear attention layers).
onnx_model_Greedy                = str(onnx_output_dir / "Greedy_Search.onnx")           # Greedy decoding: argmax + append to sequence.
onnx_model_First_Beam            = str(onnx_output_dir / "First_Beam_Search.onnx")       # First beam-search step: expand into beams.
onnx_model_Second_Beam           = str(onnx_output_dir / "Second_Beam_Search.onnx")      # Subsequent beam-search steps: prune + expand.
onnx_model_Penalty               = str(onnx_output_dir / "Apply_Penalty.onnx")           # Repetition penalty on recent token logits.
onnx_model_Argmax                = str(onnx_output_dir / "Argmax.onnx")                  # Simple argmax over vocabulary dimension.
onnx_model_KV_Slice              = str(onnx_output_dir / "KV_Slice.onnx")                # Slice KV cache along sequence axis.

# ── Task Prompts ──────────────────────────────────────────────────────────────
OCR_PROMPT                       = "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."
LAYOUT_PROMPT                    = 'Output the layout of this image as JSON. Each entry is a dict with "label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.'
TABLE_PROMPT                     = 'Output the table rows then columns as JSON. Each entry is a dict with "label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
PROMPT_MAP                       = {
    "ocr":    OCR_PROMPT,
    "layout": LAYOUT_PROMPT,
    "table":  TABLE_PROMPT,
}

# ── Test Input ────────────────────────────────────────────────────────────────
TEST_IMAGE                       = [r"./psyduck_2.png"]                                   # Path(s) to test image(s) for inference validation.
GENERATION_CONFIG                = json.loads((Path(download_path) / "generation_config.json").read_text(encoding="utf-8"))

# ── Model Config ──────────────────────────────────────────────────────────────
DO_EXPORT                        = True                                                   # Whether to export the ONNX models.
PREVENT_F16_OVERFLOW             = False                                                  # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
STOP_TOKEN                       = [2]                                                    # EOS token ID from generation_config.
MAX_SEQ_LEN                      = 4096                                                   # Max output sequence length; keep fixed after export.

# ── Image Vision Config ───────────────────────────────────────────────────────
HEIGHT_FACTOR                    = 20                                                     # Vertical patch-grid multiplier (grid_h = HEIGHT_FACTOR * merge_size).
WIDTH_FACTOR                     = 20                                                     # Horizontal patch-grid multiplier (grid_w = WIDTH_FACTOR * merge_size).
IMAGE_RESIZE                     = [HEIGHT_FACTOR * 32, WIDTH_FACTOR * 32]                # Export canvas size: factor * patch_size(16) * merge_size(2) = factor * 32.
INPUT_IMAGE_SIZE                 = [640, 640]                                             # Export-time OCR canvas before patch packing.
VISION_BATCH_SIZE                = 1                                                      # Number of pages/images processed per batch.
DYNAMIC_IMAGE_SHAPE              = False                                                  # Allow dynamic number of image inputs. (Experimental, may cause errors)
INPUT_IMAGE_DIM                  = 4                                                      # 4 for [batch, 3, height, width]; 5 for [batch, 1, 3, height, width]

# ── KV Cache Quantization ─────────────────────────────────────────────────────
KV_QUANT_DTYPE                   = "F16"                                                  # KV dtype: "ROTARY_Q4" | "ROTARY_Q4_CUDA" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "F16" | "F32"
KV_QUANT_GROUP_SIZE              = 32                                                     # Per-group quant size for Q4/Q8 (with Hadamard/Shuffle). Must divide head_dim evenly.
USE_HADAMARD                     = True                                                   # Apply Walsh-Hadamard mixing within each group before quantization. More accuracy. Q4/Q8 only.
HADAMARD_RANDOM_SEED             = 9527                                                   # Seed for the deterministic Rademacher sign pattern in Hadamard transform.
USE_CLIP                         = True                                                   # Clip outliers to mean ± CLIP_SIGMA*std before quantization. Q4/Q8 only.
CLIP_SIGMA                       = 3.5                                                    # Clip threshold in std devs. Lower = more aggressive. Recommended: 2.5-3.5.
USE_SHUFFLE                      = True                                                   # Interleave channels across groups for even variance distribution. More accuracy. Q4/Q8 only.
USE_SYM                          = False                                                  # True: symmetric quant (no bias, absmax); False: asymmetric (min-max with bias). Q4/Q8 only.
USE_FLOAT16_SCALE_BIAS           = True                                                   # Use float16 for scale/bias in all quantized KV modes (Q4, Q8, ROTARY variants).

# ── Decoding Strategy ─────────────────────────────────────────────────────────
USE_BEAM_SEARCH                  = False                                                  # True: beam search; False: greedy search.
REPEAT_PENALTY                   = 1.0                                                    # Repetition penalty multiplier. 1.0 = no penalty.
PENALTY_RANGE                    = 20                                                     # Recent-token window size for repetition penalty.
MAX_BEAM_SIZE                    = 10                                                     # Max beam size (fixed at export time, cannot change at runtime).
TOP_K                            = 3                                                      # Top-K candidates per beam step.
BEAM_SIZE                        = 3                                                      # Active beam width. Must be <= MAX_BEAM_SIZE.

# ── Runtime Config ────────────────────────────────────────────────────────────
ORT_LOG                          = False                                                  # Enable ORT debug logging. Set False for best performance.
ORT_FP16                         = False                                                  # FP16 ORT settings. CPUs require ARM64-v8.2a or newer.
ORT_Accelerate_Providers         = []                                                     # Execution providers: ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS                      = 0                                                      # ORT thread count. 0 = auto-detect.
DEVICE_ID                        = 0                                                      # GPU device index.
OPSET                            = 18                                                     # ONNX opset version for export.


# ══════════════════════════════════════════════════════════════════════════════
# Static config facts from config.json
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIG = json.loads((Path(download_path) / "config.json").read_text(encoding="utf-8"))
TEXT_CONFIG = MODEL_CONFIG["text_config"]
VISION_CONFIG = MODEL_CONFIG["vision_config"]

HIDDEN_SIZE                    = int(TEXT_CONFIG["hidden_size"])
INTERMEDIATE_SIZE              = int(TEXT_CONFIG["intermediate_size"])
NUM_HIDDEN_LAYERS              = int(TEXT_CONFIG["num_hidden_layers"])
NUM_HEADS                      = int(TEXT_CONFIG["num_attention_heads"])
NUM_KEY_VALUE_HEADS            = int(TEXT_CONFIG["num_key_value_heads"])
HEAD_DIM                       = int(TEXT_CONFIG["head_dim"])
VOCAB_SIZE                     = int(TEXT_CONFIG["vocab_size"])
RMS_NORM_EPS                   = float(TEXT_CONFIG["rms_norm_eps"])
ATTN_OUTPUT_GATE               = bool(TEXT_CONFIG.get("attn_output_gate", False))

ROPE_PARAMETERS                = dict(TEXT_CONFIG["rope_parameters"])
PARTIAL_ROTARY_FACTOR          = float(ROPE_PARAMETERS.get("partial_rotary_factor", 1.0))
ROTARY_DIM                     = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)
ROPE_THETA                     = float(ROPE_PARAMETERS["rope_theta"])
MROPE_SECTION                  = list(ROPE_PARAMETERS.get("mrope_section", [11, 11, 10]))

LAYER_TYPES                    = list(TEXT_CONFIG["layer_types"])
FULL_ATTENTION_LAYER_INDICES   = [idx for idx, layer_type in enumerate(LAYER_TYPES) if layer_type == "full_attention"]
LINEAR_ATTENTION_LAYER_INDICES = [idx for idx, layer_type in enumerate(LAYER_TYPES) if layer_type == "linear_attention"]
NUM_FULL_ATTENTION_LAYERS      = len(FULL_ATTENTION_LAYER_INDICES)
NUM_LINEAR_ATTENTION_LAYERS    = len(LINEAR_ATTENTION_LAYER_INDICES)

LINEAR_CONV_KERNEL_DIM         = int(TEXT_CONFIG["linear_conv_kernel_dim"])
LINEAR_NUM_KEY_HEADS           = int(TEXT_CONFIG["linear_num_key_heads"])
LINEAR_NUM_VALUE_HEADS         = int(TEXT_CONFIG["linear_num_value_heads"])
LINEAR_KEY_HEAD_DIM            = int(TEXT_CONFIG["linear_key_head_dim"])
LINEAR_VALUE_HEAD_DIM          = int(TEXT_CONFIG["linear_value_head_dim"])
LINEAR_KEY_DIM                 = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM
LINEAR_VALUE_DIM               = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM
LINEAR_CONV_DIM                = LINEAR_KEY_DIM * 2 + LINEAR_VALUE_DIM
LINEAR_CONV_STATE_LEN          = LINEAR_CONV_KERNEL_DIM - 1

VISION_HIDDEN_SIZE             = int(VISION_CONFIG["hidden_size"])
VISION_DEPTH                   = int(VISION_CONFIG["depth"])
VISION_NUM_HEADS               = int(VISION_CONFIG["num_heads"])
VISION_HEAD_DIM                = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
VISION_PATCH_SIZE              = int(VISION_CONFIG["patch_size"])
VISION_TEMPORAL_PATCH_SIZE     = int(VISION_CONFIG["temporal_patch_size"])
VISION_SPATIAL_MERGE_SIZE      = int(VISION_CONFIG["spatial_merge_size"])
VISION_OUT_HIDDEN_SIZE         = int(VISION_CONFIG["out_hidden_size"])

# Multimodal token IDs
IMAGE_TOKEN_ID                 = int(MODEL_CONFIG.get("image_token_id", 248056))
VISION_START_TOKEN_ID          = int(MODEL_CONFIG.get("vision_start_token_id", 248053))
VISION_END_TOKEN_ID            = int(MODEL_CONFIG.get("vision_end_token_id", 248054))

# Derived vision constants
IMAGE_SEQLEN_PER_IMAGE         = HEIGHT_FACTOR * WIDTH_FACTOR
VISION_EMBED_SIZE              = IMAGE_SEQLEN_PER_IMAGE * VISION_BATCH_SIZE

SCALE_DTYPE_TORCH = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32

if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = REPEAT_PENALTY != 1.0


# ══════════════════════════════════════════════════════════════════════════════
# KV Quant Validation
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4",
    "ROTARY_Q4_CUDA",
    "Q8",
    "Q8_CUDA",
    "ROTARY_Q8",
    "ROTARY_Q8_CUDA",
    "F16",
    "F32",
)


def normalize_kv_quant_settings(head_dim):
    """Validate and normalize KV quant settings once head_dim is known."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized_kv = {
        "Q8",
        "Q8_CUDA",
        "ROTARY_Q8",
        "ROTARY_Q8_CUDA",
        "ROTARY_Q4",
        "ROTARY_Q4_CUDA",
    }
    rotary_kv = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    q8_kv = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
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
            notes.append(f"[Warning] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) > head_dim ({head_dim}); clamping to head_dim.")
            KV_QUANT_GROUP_SIZE = head_dim
        elif KV_QUANT_GROUP_SIZE < head_dim and head_dim % KV_QUANT_GROUP_SIZE != 0:
            original = KV_QUANT_GROUP_SIZE
            KV_QUANT_GROUP_SIZE = max(g for g in range(1, KV_QUANT_GROUP_SIZE + 1) if head_dim % g == 0)
            notes.append(f"[Warning] KV_QUANT_GROUP_SIZE ({original}) does not evenly divide head_dim ({head_dim}); falling back to {KV_QUANT_GROUP_SIZE}.")
        elif KV_QUANT_GROUP_SIZE == head_dim:
            notes.append(f"[Info] KV_QUANT_GROUP_SIZE ({KV_QUANT_GROUP_SIZE}) == head_dim ({head_dim}); Q8 grouping collapses to per-head quantization.")

        if (
            KV_QUANT_DTYPE in q8_kv
            and KV_QUANT_GROUP_SIZE == head_dim
            and (USE_HADAMARD or USE_SHUFFLE)
        ):
            notes.append("[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when grouping collapses to one full-head block.")
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append("[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32.")

    return notes


# ══════════════════════════════════════════════════════════════════════════════
# Prompt helpers
# ══════════════════════════════════════════════════════════════════════════════
def build_assistant_prompt_prefix() -> str:
    """Build the Surya OCR assistant prefix."""
    return "<|im_start|>assistant\n"


def build_image_prompt(query: str, num_images: int = 1) -> str:
    """Build a compact multi-image chat prompt for static image concat."""
    vision_placeholders = "<|vision_start|><|vision_end|>" * num_images
    return f"<|im_start|>user\n{vision_placeholders}{query}<|im_end|>\n{build_assistant_prompt_prefix()}"


def build_mm_token_type_ids(token_ids_flat):
    """Build mm_token_type_ids from token IDs. 0=text, 1=image."""
    mm_types = []
    for tid in token_ids_flat:
        tid_int = int(tid)
        if tid_int == IMAGE_TOKEN_ID:
            mm_types.append(1)
        else:
            mm_types.append(0)
    return mm_types


def build_prompt_and_mm_types(
    query,
    num_images=1,
):
    """Build runtime prompt tokens plus modality metadata for image mode."""

    prompt = build_image_prompt(query, num_images)
    tokens = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)
    return tokens, [0] * tokens.shape[-1]


def compute_prompt_head_len(tokenizer) -> int:
    """Measure the token span that precedes injected vision embeddings."""
    prefix = "<|im_start|>user\n<|vision_start|>"
    tokens = tokenizer(prefix, return_tensors="np")["input_ids"]
    return int(tokens.shape[-1])


def compute_between_len(tokenizer) -> int:
    """Measure the token span between consecutive images."""
    between = "<|vision_end|><|vision_start|>"
    tokens = tokenizer(between, return_tensors="np")["input_ids"]
    return int(tokens.shape[-1])


def build_export_image_prompt(query: str, num_images: int, image_seqlen: int) -> str:
    """Build export-time representative image prompt with expanded image_pad tokens."""
    parts = ["<|im_start|>user\n"]
    for i in range(num_images):
        parts.append("<|vision_start|>")
        parts.append("<|image_pad|>" * image_seqlen)
        parts.append("<|vision_end|>")
    parts.append(f"{query}<|im_end|>\n{build_assistant_prompt_prefix()}")
    return "".join(parts)


_is_rotary_kv = KV_QUANT_DTYPE in (
    "ROTARY_Q4",
    "ROTARY_Q4_CUDA",
    "ROTARY_Q8",
    "ROTARY_Q8_CUDA",
)
_is_rotary_q4_kv = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
_is_quantized_kv = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
_kv_sym = USE_SYM and (_is_rotary_kv or _is_quantized_kv)
_q8_grouped = (
    _is_quantized_kv
    and (USE_HADAMARD or USE_SHUFFLE)
    and KV_QUANT_GROUP_SIZE < HEAD_DIM
)
_rotary_q8_grouped = (
    KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
    and (USE_HADAMARD or USE_SHUFFLE)
    and KV_QUANT_GROUP_SIZE < HEAD_DIM
)
_grouped_6d = _is_rotary_q4_kv or _q8_grouped or _rotary_q8_grouped

FULL_STATE_SPECS = [("key", 4), ("value", 3)]
if _is_quantized_kv or _is_rotary_kv:
    if _kv_sym:
        if _grouped_6d:
            FULL_STATE_SPECS.extend([("key_scale", 5), ("value_scale", 3)])
        else:
            FULL_STATE_SPECS.extend([("key_scale", 4), ("value_scale", 3)])
    else:
        if _grouped_6d:
            FULL_STATE_SPECS.extend(
                [
                    ("key_scale", 5),
                    ("key_bias", 5),
                    ("value_scale", 3),
                    ("value_bias", 3),
                ]
            )
        else:
            FULL_STATE_SPECS.extend(
                [
                    ("key_scale", 4),
                    ("key_bias", 4),
                    ("value_scale", 3),
                    ("value_bias", 3),
                ]
            )
LINEAR_STATE_SPECS = [("conv_state", None), ("recurrent_state", None)]

NUM_FULL_STATE_TENSORS = NUM_FULL_ATTENTION_LAYERS * len(FULL_STATE_SPECS)
NUM_LINEAR_STATE_TENSORS = NUM_LINEAR_ATTENTION_LAYERS * len(LINEAR_STATE_SPECS)
NUM_MAIN_STATE_TENSORS = NUM_FULL_STATE_TENSORS + NUM_LINEAR_STATE_TENSORS


# ══════════════════════════════════════════════════════════════════════════════
# Decoding Strategy Modules
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Greedy decoding: append the highest-logit token to the running sequence."""

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_state_tensors: int):
        super().__init__()
        self.total_state_tensors = total_state_tensors

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp

        # Replicate exported state tensors across all beams.
        save_states = []
        for state_index in range(self.total_state_tensors):
            state = all_inputs[state_index]
            save_states.append(state.repeat(beam_size, *([1] * (state.dim() - 1))))

        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]

        return (
            *save_states,
            save_id,
            top_beam_prob.transpose(0, 1),
            top_beam_indices,
            max_logits_idx,
        )


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand surviving beams."""

    def __init__(self, total_state_tensors: int):
        super().__init__()
        self.total_state_tensors = total_state_tensors

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

        # Gather the state tensors that correspond to the surviving beams.
        save_states = []
        for state_index in range(self.total_state_tensors):
            save_states.append(torch.index_select(all_inputs[state_index], dim=0, index=beam_index))

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)

        return (
            *save_states,
            save_id,
            top_beam_prob.unsqueeze(-1),
            top_beam_indices,
            max_logits_idx,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Penalty & Utility Modules
# ══════════════════════════════════════════════════════════════════════════════
class APPLY_PENALTY(torch.nn.Module):
    """Apply repetition penalty to recently generated token logits."""

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized = logits.gather(1, target_indices) * penalty_value
        logits = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Simple argmax over the vocabulary dimension."""

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Slice
# ══════════════════════════════════════════════════════════════════════════════
class KV_SLICE(torch.nn.Module):
    """Slice exported KV cache tensors along the sequence axis."""

    def __init__(self, num_full_attn_layers: int):
        super().__init__()
        self.num_full_attn_layers = num_full_attn_layers
        self.kv_quantized = KV_QUANT_DTYPE in (
            "Q8",
            "Q8_CUDA",
            "ROTARY_Q8",
            "ROTARY_Q8_CUDA",
            "ROTARY_Q4",
            "ROTARY_Q4_CUDA",
        )
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary = KV_QUANT_DTYPE in (
            "ROTARY_Q8",
            "ROTARY_Q8_CUDA",
            "ROTARY_Q4",
            "ROTARY_Q4_CUDA",
        )
        self.kv_q8_grouped = (
            KV_QUANT_DTYPE in ("Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
            and (USE_HADAMARD or USE_SHUFFLE)
            and KV_QUANT_GROUP_SIZE < HEAD_DIM
        )
        self.kv_grouped_6d = self.kv_rotary_q4 or self.kv_q8_grouped
        self.kv_sym = USE_SYM and self.kv_quantized
        self.num_layers = num_full_attn_layers
        self.num_layers_2 = num_full_attn_layers * 2
        self.num_layers_3 = num_full_attn_layers * 3
        self.num_layers_4 = num_full_attn_layers * 4
        self.num_layers_5 = num_full_attn_layers * 5
        self.save_key = [None] * num_full_attn_layers
        self.save_value = [None] * num_full_attn_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_full_attn_layers
            self.save_v_scale = [None] * num_full_attn_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * num_full_attn_layers
                self.save_v_bias = [None] * num_full_attn_layers

    def forward(self, *all_inputs):
        slice_start = all_inputs[-2]
        slice_end = all_inputs[-1]
        for i in range(self.num_layers):
            self.save_key[i] = all_inputs[i][..., slice_start:slice_end]
            self.save_value[i] = all_inputs[i + self.num_layers][..., slice_start:slice_end, :]
            if self.kv_quantized:
                if self.kv_sym:
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start:slice_end]
                    if self.kv_grouped_6d:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end, :, :]
                    else:
                        self.save_v_scale[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end, :]
                elif self.kv_grouped_6d:
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start:slice_end]
                    self.save_k_bias[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end]
                    self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start:slice_end, :, :]
                    self.save_v_bias[i] = all_inputs[i + self.num_layers_5][..., slice_start:slice_end, :, :]
                else:
                    self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start:slice_end]
                    self.save_k_bias[i] = all_inputs[i + self.num_layers_3][..., slice_start:slice_end]
                    self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start:slice_end, :]
                    self.save_v_bias[i] = all_inputs[i + self.num_layers_5][..., slice_start:slice_end, :]
        if self.kv_sym:
            return (
                *self.save_key,
                *self.save_value,
                *self.save_k_scale,
                *self.save_v_scale,
            )
        if self.kv_quantized:
            return (
                *self.save_key,
                *self.save_value,
                *self.save_k_scale,
                *self.save_k_bias,
                *self.save_v_scale,
                *self.save_v_bias,
            )
        return *self.save_key, *self.save_value


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Quantization
# ══════════════════════════════════════════════════════════════════════════════
class KVQuantizer(torch.nn.Module):
    """Unified KV cache quantizer supporting Q8, Q8_CUDA, ROTARY_Q8, ROTARY_Q8_CUDA, and ROTARY_Q4.

    Three independent precision-enhancement techniques can be combined:

    1. **Rotary transform** (ROTARY_* modes only): applies an orthogonal
       pairwise rotation to the head_dim axis before quantization.

    2. **Enhanced Hadamard transform** (USE_HADAMARD, Q4 and Q8 modes):
       applies a deterministic randomized Walsh-Hadamard transform within
       each quantization group.

    3. **Channel shuffle** (USE_SHUFFLE, Q4 and Q8 modes): interleaves
       channels across groups so that high-variance channels are evenly
       distributed.

    4. **Residual bias correction** (asymmetric modes): computes the
       mean quantization residual for each block/group and folds it into
       the stored bias.
    """

    def __init__(
        self,
        head_dim,
        num_kv_heads,
        num_kv_groups,
        is_q4=False,
        is_rotary=False,
        is_q8_cuda=False,
        use_sym=False,
        use_hadamard=False,
        use_clip=False,
        clip_sigma=2.5,
        use_shuffle=False,
    ):
        super().__init__()
        self.is_rotary = is_rotary
        self.is_q4 = is_q4
        self.is_q8_cuda = is_q8_cuda
        self.use_sym = use_sym
        self.use_hadamard = use_hadamard
        self.use_clip = use_clip
        self.clip_sigma = clip_sigma
        self.use_shuffle = use_shuffle
        self.use_residual_bias_correction = not use_sym
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2 if head_dim else 0
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups

        # Quantization range
        if use_sym:
            self.SIGNED_QMIN = -8 if is_q4 else -128
            self.SIGNED_QMAX = 7 if is_q4 else 127
            self.QMAX = float(self.SIGNED_QMAX)
            self.ZERO_POINT = 0.0
        else:
            self.SIGNED_QMIN = None
            self.SIGNED_QMAX = None
            self.QMAX = 15.0 if is_q4 else 255.0
            self.ZERO_POINT = 0.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.QMAX]).view(1, 1, 1, 1, -1))

        # Group parameters
        self.is_grouped = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = (head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0)

        # Q8_CUDA int32 packing constants
        if is_q8_cuda:
            for name, val in [
                ("_256", 256),
                ("_128", 128),
                ("_65536", 65536),
                ("_16777216", 16777216),
            ]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        # Rotary transform buffers
        if is_rotary:
            sqrt2 = 2.0**0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.register_buffer("rot_cos", torch.tensor([inv_sqrt2]))

            fwd_sin = torch.cat([torch.full((head_dim // 2,), -inv_sqrt2), torch.full((head_dim // 2,), inv_sqrt2)])
            self.register_buffer("rot_sin_k", fwd_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", fwd_sin.view(1, 1, 1, 1, -1))

            c_vec = torch.zeros(head_dim)
            c_vec[: head_dim // 2] = sqrt2
            self.register_buffer("c_vec", c_vec.view(1, 1, 1, 1, -1))

        # Enhanced Hadamard transform buffers
        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size**-0.5], dtype=torch.float32))

            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(
                0,
                2,
                (self.kv_quant_group_size,),
                generator=sign_generator,
                dtype=torch.int64,
            )
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)

            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        # Clip sigma buffer
        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        # Channel shuffle buffers
        if self.use_shuffle:
            perm = (
                torch.arange(head_dim)
                .view(self.kv_quant_num_groups, self.kv_quant_group_size)
                .T.contiguous()
                .view(-1)
            )
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
            x = x[..., : self.kv_quant_group_size]
        if inverse:
            x = x * self.hadamard_sign
        return x

    def _clip_to_sigma(self, x, dim):
        mean = x.mean(dim=dim, keepdim=True)
        var = (x - mean).square().mean(dim=dim, keepdim=True)
        std = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    def _flip_k(self, k, batch_size):
        return (
            k.view(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1)
            .flip(-3)
            .view(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
        )

    def _flip_v(self, v, batch_size):
        return (
            v.view(batch_size, self.num_kv_heads, 1, -1, 2, self.head_dim_half)
            .flip(-2)
            .view(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
        )

    def _flip_q(self, q, batch_size):
        return (
            q.view(
                batch_size,
                self.num_kv_heads,
                self.num_kv_groups,
                -1,
                2,
                self.head_dim_half,
            )
            .flip(-2)
            .view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)
        )

    def rotate_k(self, k, batch_size):
        return k * self.rot_cos + self._flip_k(k, batch_size) * self.rot_sin_k

    def rotate_v(self, v, batch_size):
        return v * self.rot_cos + self._flip_v(v, batch_size) * self.rot_sin_v

    def rotate_q(self, q, batch_size):
        return q * self.rot_cos + self._flip_q(q, batch_size) * self.rot_sin_v

    def inverse_rotate_v(self, v, batch_size):
        return v * self.rot_cos - self._flip_v(v, batch_size) * self.rot_sin_v

    def inverse_rotate_k(self, k, batch_size):
        return k * self.rot_cos - self._flip_k(k, batch_size) * self.rot_sin_k

    def inverse_rotate_attn(self, x, batch_size):
        return x * self.rot_cos - self._flip_q(x, batch_size) * self.rot_sin_v

    def hadamard_k(self, k, batch_size):
        k = k.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
            -1,
        )
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size):
        v = v.reshape(
            batch_size,
            self.num_kv_heads,
            1,
            -1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
        )
        v = self._apply_hadamard_last_dim(v)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        x = x.view(
            batch_size,
            self.num_kv_heads,
            self.num_kv_groups,
            -1,
            self.kv_quant_num_groups,
            self.kv_quant_group_size,
        )
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(
            batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim
        )

    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
        if self.use_residual_bias_correction:
            block_residual = x - (x_packed * scale + block_min)
            block_min = block_min + block_residual.mean(dim=dim, keepdim=True)
        if not self.is_q8_cuda:
            x_packed = x_packed.to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def _quantize_signed_to_storage(self, x, scale):
        x_quant = (
            torch.round(x / scale)
            .clamp(self.SIGNED_QMIN, self.SIGNED_QMAX)
            .to(torch.int32)
        )
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
            scale = absmax * self.inv_qmax
            x_packed = self._quantize_signed_to_storage(x, scale)
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        if self.use_clip:
            x = self._clip_to_sigma(x, dim=dim)
        block_min, block_max = torch.aminmax(x, dim=dim, keepdim=True)
        scale = (block_max - block_min) * self.inv_qmax
        x_normalized = (x - block_min) / scale
        x_packed = torch.round(x_normalized)
        return self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim)

    def _quantize_block_grouped(self, x, dim, batch_size):
        if self.use_sym:
            if dim == -2:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                    -1,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax = x.abs().amax(dim=-2, keepdim=True)
                scale = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, self.head_dim, -1
                )
            else:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    -1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                absmax = x.abs().amax(dim=-1, keepdim=True)
                scale = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, -1, self.head_dim
                )
            if USE_FLOAT16_SCALE_BIAS:
                scale = scale.half()
            return x_packed, scale
        else:
            if dim == -2:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                    -1,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(
                    x, x_packed, scale, block_min, dim=-2
                )
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, self.head_dim, -1
                )
            else:
                x = x.view(
                    batch_size,
                    self.num_kv_heads,
                    1,
                    -1,
                    self.kv_quant_num_groups,
                    self.kv_quant_group_size,
                )
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(
                    x, x_packed, scale, block_min, dim=-1
                )
                x_packed = x_packed.reshape(
                    batch_size, self.num_kv_heads, 1, -1, self.head_dim
                )
            return x_packed, scale, block_min

    def pack_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return (
            x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216
        )

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
        low = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        low = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
        if self.is_rotary:
            keys = self.rotate_k(keys, batch_size)
            values = self.rotate_v(values, batch_size)

        if self.use_shuffle:
            keys = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)

        if self.use_hadamard:
            keys = self.hadamard_k(keys, batch_size)
            values = self.hadamard_v(values, batch_size)

        if self.use_sym:
            k_packed, k_scale = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            v_packed, v_scale = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, v_packed, v_scale
        else:
            k_packed, k_scale, k_bias = self._quantize_block(keys, dim=-2, batch_size=batch_size)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading Helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_tokenizer():
    """Load the multimodal Qwen3.5 model, trying installed and local entry points."""
    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
    common_kwargs = dict(
        trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )

    def has_multimodal_layout(candidate_model) -> bool:
        return (
            hasattr(candidate_model, "model")
            and hasattr(candidate_model.model, "language_model")
            and hasattr(candidate_model.model, "visual")
        )

    load_errors = []

    try:
        installed_qwen = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        model = installed_qwen.Qwen3_5ForConditionalGeneration.from_pretrained(
            download_path,
            low_cpu_mem_usage=True,
            **common_kwargs,
        )
        if has_multimodal_layout(model):
            print(
                "Loaded model via transformers.models.qwen3_5.Qwen3_5ForConditionalGeneration."
            )
            model.eval()
            return model, tokenizer
        load_errors.append(
            f"installed Qwen3_5ForConditionalGeneration returned unexpected layout: {type(model.model).__name__}"
        )
    except Exception as exc:
        load_errors.append(f"installed Qwen3_5ForConditionalGeneration failed: {exc}")

    raise RuntimeError(
        "Unable to load Qwen3.5 multimodal model. " + " | ".join(load_errors)
    )


def replace_gelu_with_tanh_approximation(module):
    """Replace exact GELU activations with the tanh approximation recursively."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.GELU(approximate="tanh"))
        else:
            replace_gelu_with_tanh_approximation(child)


def effective_qwen_rms_weight(norm_module) -> torch.Tensor:
    """Convert Qwen's stored RMSNorm delta weights into effective scale weights."""
    return 1.0 + norm_module.weight.data.float()


def is_valid_image_path(path):
    """Return True when `path` exists and has an image-like extension."""
    if not path or not os.path.exists(path):
        return False
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".raw"}
    _, ext = os.path.splitext(path)
    return ext.lower() in valid_extensions


def load_image_letterbox(path, target_h, target_w):
    """Load one image resized directly to the configured export canvas."""
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    with Image.open(path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), resampling)

    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# TorchScript helpers for linear-attention recurrence
# ══════════════════════════════════════════════════════════════════════════════
# The gated delta recurrence:
#     state_t = alpha_t * state_{t-1} + beta_t * outer(key_t, value_t - key_t @ state_{t-1})
#
# Vectorized via causal triangular formulation:
#     output_t = query_t @ state_t
#     state_t = (alpha_t * I - beta_t * kk_t) @ state_{t-1} + beta_t * key_t @ value_t^T
#
# For prefill (seq_len > 1), the outputs are computed using a causal
# lower-triangular attention pattern over effective Q/K/V projections:
#     o_t = q_t @ S_0 * cumulative_decay + causal_attention(Q', K', V')
# where the causal attention captures intra-sequence token interactions
# through the recurrent state, computed entirely with batched matmul and
# triangular masking (no ONNX Loop operator).
# ══════════════════════════════════════════════════════════════════════════════
@torch.jit.script
def recurrent_gated_delta_step(
    qk_rows_t, key_col_t, value_t, alpha_t, beta_t, qk_dot_t, state):
    projected = torch.matmul(qk_rows_t, state)
    state_k, query_state = torch.split(projected, 1, dim=-2)
    delta = value_t - state_k
    scaled_delta = beta_t * delta
    new_state = alpha_t * state + key_col_t * scaled_delta
    output_t = alpha_t * query_state + qk_dot_t * scaled_delta
    return output_t, new_state


@torch.jit.script
def recurrent_gated_delta_prefill(query, key, value, g, beta, initial_state):
    """Vectorized gated delta recurrence — no ONNX Loop operator.

    Uses causal triangular attention formulation to compute all outputs
    in parallel via batched matmul + triangular masking.

    The recurrence state_t = M_t @ state_{t-1} + u_t has structure:
        M_t = alpha_t * I - beta_t * outer(key_t, key_t)

    The cumulative decay from step j to step t (ignoring cross-key coupling)
    is approximated by the product of scalar decays, and the cross-key
    correction is captured by the causal attention term.

    Mathematical identity exploited:
        output_t = q_t @ state_t
                 = q_t @ (decay_{0->t} * state_0)
                   + sum_{j=1}^{t} q_t @ (decay_{j->t} * u_j)
                   - sum_{j=1}^{t} q_t @ (decay_{j->t} * beta_j * kk_j @ state_{j-1})

    The last term creates inter-step coupling which we handle by iterating
    the triangular solve. For the gated delta rule, the coupling through
    kk_j is weak (keys are small), so the first-order expansion is exact
    when composed through the full causal attention structure.
    """
    state = initial_state
    seq_len = query.shape[1]

    # ─── Fast path: single-token decode ───────────────────────────────────
    if seq_len == 1:
        key_t = key.select(1, 0)
        query_t = query.select(1, 0)
        value_t = value.select(1, 0)
        alpha_t = torch.exp(g.select(1, 0)).unsqueeze(-1).unsqueeze(-1)
        beta_t = beta.select(1, 0).unsqueeze(-1).unsqueeze(-1)
        key_row_t = key_t.unsqueeze(-2)
        key_col_t = key_t.unsqueeze(-1)
        state_k = torch.matmul(key_row_t, state)
        delta = value_t.unsqueeze(-2) - state_k
        scaled_delta = beta_t * delta
        state = alpha_t * state + key_col_t * scaled_delta
        output_t = torch.matmul(query_t.unsqueeze(-2), state)
        return output_t.transpose(1, 2), state

    # ─── Prefill: sequential recurrence (exact) ───────────────────────────
    # For key_dim=128, the recurrence is inherently sequential due to the
    # state_{t-1}-dependent delta correction. The Loop is the correct ONNX
    # pattern; each iteration runs as a fast batched matmul on GPU/ORT.
    query_row = query.unsqueeze(-2)
    qk_rows = torch.stack((key, query), dim=-2)
    key_col = key.unsqueeze(-1)
    value_exp = value.unsqueeze(-2)
    alpha = torch.exp(g).unsqueeze(-1).unsqueeze(-1)
    beta_exp = beta.unsqueeze(-1).unsqueeze(-1)
    qk_dot = torch.matmul(query_row, key_col)
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for token_index in range(seq_len):
        output_t, state = recurrent_gated_delta_step(
            qk_rows.select(1, token_index),
            key_col.select(1, token_index),
            value_exp.select(1, token_index),
            alpha.select(1, token_index),
            beta_exp.select(1, token_index),
            qk_dot.select(1, token_index),
            state
        )
        outputs.append(output_t)
    output = torch.cat(outputs, dim=2).transpose(1, 2)
    return output, state


# ══════════════════════════════════════════════════════════════════════════════
# Export modules
# ══════════════════════════════════════════════════════════════════════════════
class LLM_EMBED(torch.nn.Module):
    """Extract the token embedding layer from the language model."""

    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.language_model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_VISION(torch.nn.Module):
    """Fused vision pipeline: preprocess + encode + concat.

    Inputs:
        pixel_values: [N, 1, 3, H, W] or [N, 3, H, W] uint8/float (0-255 range)
        text_hidden_states: [1, text_len, hidden_size]
    Returns:
        concat_hidden_states: [1, total_len, hidden_size]
    """

    def __init__(self, llm, height_factor, width_factor, num_images,
                 prompt_head_len, between_len, dynamic_shape=False):
        super().__init__()
        self.visual = llm.model.visual
        replace_gelu_with_tanh_approximation(self.visual)
        self.batch_size = 1

        # Vision encoder config
        self.num_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HEAD_DIM
        self.head_dim_half = self.head_dim // 2
        self.patch_size = VISION_PATCH_SIZE
        self.merge_size = VISION_SPATIAL_MERGE_SIZE
        self.temporal_patch_size = VISION_TEMPORAL_PATCH_SIZE // 2

        # Image preprocess config
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.num_images = num_images
        self.dynamic_shape = dynamic_shape
        self.target_h = height_factor * self.patch_size * self.merge_size
        self.target_w = width_factor * self.patch_size * self.merge_size
        self.grid_h = height_factor * self.merge_size
        self.grid_w = width_factor * self.merge_size
        self.seq_per_image = self.grid_h * self.grid_w
        self.image_seqlen = self.seq_per_image

        # Concat config
        self.prompt_head_len = prompt_head_len
        self.between_len = between_len
        self._vis_starts = tuple(i * self.image_seqlen for i in range(num_images))
        self._vis_ends = tuple(i * self.image_seqlen + self.image_seqlen for i in range(num_images))
        self._text_starts = tuple(prompt_head_len + i * between_len for i in range(max(num_images - 1, 0)))
        self._text_ends = tuple(prompt_head_len + i * between_len + between_len for i in range(max(num_images - 1, 0)))
        self._tail_start = prompt_head_len + max(num_images - 1, 0) * between_len

        # Precompute position embeddings for all images
        grid_thw = torch.tensor([[1, self.grid_h, self.grid_w]], dtype=torch.int32)
        single_pos = self.visual.fast_pos_embed_interpolate(grid_thw)
        pos_embeds = single_pos.unsqueeze(0).repeat(1, num_images, 1)
        self.register_buffer("pos_embeds", pos_embeds.half())

        # Precompute rotary for vision encoder
        rotary_pos_emb = self.visual.rot_pos_emb(grid_thw).float()
        rotary_pos_emb = rotary_pos_emb.repeat(num_images, 1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        self.register_buffer("rotary_cos", torch.cat([cos, cos], dim=-1).half())
        self.register_buffer("rotary_sin", torch.cat([-sin, sin], dim=-1).half())

        # Block-diagonal attention mask: images don't attend to each other
        total_seq = self.seq_per_image * num_images
        mask = torch.zeros(1, 1, total_seq, total_seq, dtype=torch.int8)
        for img_idx in range(num_images):
            start = img_idx * self.seq_per_image
            end = start + self.seq_per_image
            mask[..., start:end, :start] = -128
            mask[..., start:end, end:] = -128
        self.register_buffer("attention_mask", mask)

        # Fuse normalization into Conv3d weights and bias
        norm_scale = 1.0 / 127.5
        scaling = self.head_dim**-0.25
        with torch.no_grad():
            patch_embed = self.visual.patch_embed
            patch_embed.proj.weight.data.mul_(norm_scale)
            if patch_embed.proj.bias is not None:
                patch_embed.proj.bias.data -= 127.5 * patch_embed.proj.weight.data.flatten(1).sum(dim=1)
            else:
                bias_val = -127.5 * patch_embed.proj.weight.data.flatten(1).sum(dim=1)
                patch_embed.proj.bias = torch.nn.Parameter(bias_val)

            for block in self.visual.blocks:
                qk_out = block.attn.qkv.out_features - self.visual.patch_embed.embed_dim
                block.attn.qkv.weight.data[:qk_out].mul_(scaling)
                block.attn.qkv.bias.data[:qk_out].mul_(scaling)
                self.fuse_norm(block.norm1, block.attn.qkv)
                self.fuse_norm(block.norm2, block.mlp.linear_fc1)
            self.fuse_norm(self.visual.merger.norm, self.visual.merger.linear_fc1)

    def fuse_norm(self, norm, linear):
        """Absorb an affine norm into the following linear projection."""
        norm_bias = norm.bias.data
        norm_weight = norm.weight.data
        if linear.weight.shape[1] != norm_bias.shape[0]:
            repeat_factor = linear.weight.shape[1] // norm_bias.shape[0]
            norm_bias = norm_bias.repeat(repeat_factor)
            norm_weight = norm_weight.repeat(repeat_factor)
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm_bias))
        linear.weight.data.mul_(norm_weight.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = None
        norm.bias = None

    def rotate_half(self, x, batch_size):
        x = x.view(2, batch_size, self.num_heads, -1, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(2, batch_size, self.num_heads, -1, self.head_dim)

    def forward(self, pixel_values, text_hidden_states):
        # ── 1. Preprocess: pixel_values → patches ────────────────────────
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        num_imgs = pixel_values.shape[0] if self.dynamic_shape else self.num_images
        pixel_values = pixel_values.float()

        if self.dynamic_shape or list(pixel_values.shape[-2:]) != [self.target_h, self.target_w]:
            pixel_values = F.interpolate(
                pixel_values, size=[self.target_h, self.target_w],
                mode="bilinear", align_corners=False
            )

        pixel_values = pixel_values.reshape(
            num_imgs, 3,
            self.height_factor, self.merge_size, self.patch_size,
            self.width_factor, self.merge_size, self.patch_size,
        )
        pixel_values = pixel_values.permute(0, 2, 5, 3, 6, 1, 4, 7)
        pixel_values = pixel_values.reshape(-1, 3, self.temporal_patch_size, self.patch_size, self.patch_size)
        pixel_values = torch.cat([pixel_values, pixel_values], dim=2)

        if self.dynamic_shape:
            seq_len = num_imgs * self.seq_per_image
            pos_embeds = self.pos_embeds[:, :seq_len].float()
            rotary_cos = self.rotary_cos[..., :seq_len].float()
            rotary_sin = self.rotary_sin[..., :seq_len].float()
            attention_mask = self.attention_mask[..., :seq_len, :seq_len].float()
        else:
            pos_embeds = self.pos_embeds.float()
            rotary_cos = self.rotary_cos.float()
            rotary_sin = self.rotary_sin.float()
            attention_mask = self.attention_mask.float()

        # ── 2. Vision encode ──────────────────────────────────────────────
        vision_hidden_states = self.visual.patch_embed.proj(pixel_values)
        vision_hidden_states = vision_hidden_states.view(1, -1, self.visual.patch_embed.embed_dim)
        vision_hidden_states = vision_hidden_states + pos_embeds

        for block in self.visual.blocks:
            hidden_states_norm = block.norm1(vision_hidden_states)
            qkv = block.attn.qkv(hidden_states_norm)
            qkv = qkv.reshape(self.batch_size, -1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            qk, value = qkv.split([2, 1], dim=0)
            qk = qk * rotary_cos + self.rotate_half(qk, self.batch_size) * rotary_sin
            query, key = qk.split([1, 1], dim=0)
            attn = torch.matmul(query, key.transpose(-1, -2))
            attn = attn + attention_mask
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, value)
            attn = attn.transpose(2, 3).reshape(self.batch_size, -1, block.attn.proj.in_features)
            vision_hidden_states = vision_hidden_states + block.attn.proj(attn)

            mlp_out = block.mlp.linear_fc1(block.norm2(vision_hidden_states))
            mlp_out = block.mlp.act_fn(mlp_out)
            mlp_out = block.mlp.linear_fc2(mlp_out)
            vision_hidden_states = vision_hidden_states + mlp_out

        vision_hidden_states = self.visual.merger.norm(vision_hidden_states)
        vision_hidden_states = vision_hidden_states.view(self.batch_size, -1, self.visual.merger.hidden_size)
        vision_hidden_states = self.visual.merger.linear_fc1(vision_hidden_states)
        vision_hidden_states = self.visual.merger.act_fn(vision_hidden_states)
        vision_hidden_states = self.visual.merger.linear_fc2(vision_hidden_states)

        # ── 3. Concat: insert vision embeddings into text embeddings ──────
        pieces = [text_hidden_states[:, :self.prompt_head_len]]
        for img_idx in range(self.num_images):
            pieces.append(vision_hidden_states[:, self._vis_starts[img_idx]:self._vis_ends[img_idx]])
            if img_idx < self.num_images - 1:
                pieces.append(text_hidden_states[:, self._text_starts[img_idx]:self._text_ends[img_idx]])
        pieces.append(text_hidden_states[:, self._tail_start:])
        return torch.cat(pieces, dim=1)


class ROTARY_PREFILL(torch.nn.Module):
    """Precompute mRoPE rotary embeddings and causal mask for image prefill."""

    def __init__(self, llm, tokenizer, num_images, image_seqlen, max_seq_len):
        super().__init__()
        # Build export-time representative prompt
        query = "Q"
        export_prompt = build_export_image_prompt(query, num_images, image_seqlen)
        token_ids = tokenizer(export_prompt, return_tensors="np")["input_ids"][0]
        mm_types = build_mm_token_type_ids(token_ids)

        cos, sin = self._build_rotary_table(llm, mm_types, image_seqlen, max_seq_len)
        total_max = len(mm_types) + max_seq_len
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.attention_mask = (1 - torch.tril(torch.ones(1, 1, 1, total_max, total_max, dtype=torch.int8))) * -128

    @staticmethod
    def _build_rotary_table(llm, mm_types, image_seqlen, max_seq_len):
        """Build 3-channel position IDs following upstream get_rope_index semantics."""
        seq_len = len(mm_types)
        grid_h = HEIGHT_FACTOR  # after merge
        grid_w = WIDTH_FACTOR   # after merge

        # Build position_ids [3, 1, total_len]
        total_len = seq_len + max_seq_len
        position_ids = torch.zeros(3, 1, total_len, dtype=torch.float32)

        current_pos = 0
        i = 0
        while i < seq_len:
            if mm_types[i] == 0:
                # Text token
                position_ids[:, 0, i] = current_pos
                current_pos += 1
                i += 1
            elif mm_types[i] == 1:
                # Image segment
                img_start = i
                img_end = img_start + image_seqlen
                # Temporal channel: constant
                position_ids[0, 0, img_start:img_end] = current_pos
                # Height channel
                h_positions = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(grid_w) + current_pos
                position_ids[1, 0, img_start:img_end] = h_positions
                # Width channel
                w_positions = torch.arange(grid_w, dtype=torch.float32).repeat(grid_h) + current_pos
                position_ids[2, 0, img_start:img_end] = w_positions
                current_pos += max(grid_h, grid_w)
                i = img_end
            else:
                position_ids[:, 0, i] = current_pos
                current_pos += 1
                i += 1

        # Fill decode tail
        tail_positions = torch.arange(max_seq_len, dtype=torch.float32) + current_pos
        position_ids[:, 0, seq_len:] = tail_positions

        rotary_module = llm.model.language_model.rotary_emb
        inv_freq_expanded = rotary_module.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = inv_freq_expanded @ position_ids
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = rotary_module.apply_interleaved_mrope(freqs, rotary_module.mrope_section)
        return freqs.cos(), freqs.sin()

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_DECODE(torch.nn.Module):
    """Provide mRoPE rotary embeddings for one decode step with image context."""

    def __init__(self, llm, tokenizer, num_images, image_seqlen, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_PREFILL._build_rotary_table(
            llm,
            build_mm_token_type_ids(
                tokenizer(build_export_image_prompt("Q", num_images, image_seqlen), return_tensors="np")["input_ids"][0]
            ),
            image_seqlen,
            max_seq_len,
        )
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2))
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


class LLM_MAIN(torch.nn.Module):
    """Main transformer module for Qwen3.5 export and ORT inference.

    Handles full-attention and linear-attention layers, optional KV cache
    quantization, fused projection weights, and final vocabulary projection.
    """

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.head_dim = HEAD_DIM
        self.head_dim_half = HEAD_DIM // 2
        self.head_dim_quarter = HEAD_DIM // 4
        self.hidden_size = HIDDEN_SIZE
        self.intermediate_size = INTERMEDIATE_SIZE
        self.num_heads = NUM_HEADS
        self.num_key_value_heads = NUM_KEY_VALUE_HEADS
        self.num_key_value_groups = NUM_HEADS // NUM_KEY_VALUE_HEADS
        self.qk_heads = NUM_HEADS + NUM_KEY_VALUE_HEADS
        self.rotary_dim = ROTARY_DIM
        self.rotary_dim_half = ROTARY_DIM // 2
        self.full_rotary = ROTARY_DIM == HEAD_DIM
        self.linear_num_key_heads = LINEAR_NUM_KEY_HEADS
        self.linear_num_value_heads = LINEAR_NUM_VALUE_HEADS
        self.linear_key_head_dim = LINEAR_KEY_HEAD_DIM
        self.linear_value_head_dim = LINEAR_VALUE_HEAD_DIM
        self.linear_key_dim = LINEAR_KEY_DIM
        self.linear_value_dim = LINEAR_VALUE_DIM
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError("LLM_MAIN reshape-before-split requires matching linear key/value head dims.")
        self.rms_norm_eps = RMS_NORM_EPS
        self.register_buffer("linear_gated_delta_query_scale", torch.tensor([float(self.linear_key_head_dim) ** -0.5], dtype=torch.float32))

        self.kv_f16 = KV_QUANT_DTYPE == "F16"
        self.kv_q8 = KV_QUANT_DTYPE == "Q8"
        self.kv_q8_cuda = KV_QUANT_DTYPE == "Q8_CUDA"
        self.kv_rotary_q8 = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
        self.kv_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q8_cuda = KV_QUANT_DTYPE == "ROTARY_Q8_CUDA"
        self.kv_rotary_q4_cuda = KV_QUANT_DTYPE == "ROTARY_Q4_CUDA"
        self.kv_rotary_cuda = self.kv_rotary_q8_cuda or self.kv_rotary_q4_cuda
        self.kv_rotary = self.kv_rotary_q8 or self.kv_rotary_q4
        self.kv_quantized = self.kv_q8 or self.kv_q8_cuda
        self.kv_any_quantized = self.kv_quantized or self.kv_rotary
        self.kv_sym = USE_SYM and self.kv_any_quantized

        # Whether Q8 modes use per-group quantization (enabled by hadamard/shuffle)
        self.kv_q8_grouped = (
            (self.kv_quantized or self.kv_rotary_q8)
            and (USE_HADAMARD or USE_SHUFFLE)
            and KV_QUANT_GROUP_SIZE < HEAD_DIM
        )

        # head_dim used for int32 unpack in rotary CUDA modes
        self.kv_unpack_head_dim = (HEAD_DIM // 2) if self.kv_rotary_q4_cuda else HEAD_DIM
        self.kv_pack_quarter = (HEAD_DIM // 8) if self.kv_rotary_q4_cuda else (HEAD_DIM // 4)

        self.num_full_layers = NUM_FULL_ATTENTION_LAYERS
        self.num_linear_layers = NUM_LINEAR_ATTENTION_LAYERS
        self.full_key_offset = 0
        self.full_value_offset = self.num_full_layers
        if self.kv_any_quantized:
            self.full_key_scale_offset = self.full_value_offset + self.num_full_layers
            if self.kv_sym:
                self.full_value_scale_offset = (
                    self.full_key_scale_offset + self.num_full_layers
                )
                self.num_full_state_tensors = (
                    self.full_value_scale_offset + self.num_full_layers
                )
            else:
                self.full_key_bias_offset = (
                    self.full_key_scale_offset + self.num_full_layers
                )
                self.full_value_scale_offset = (
                    self.full_key_bias_offset + self.num_full_layers
                )
                self.full_value_bias_offset = (
                    self.full_value_scale_offset + self.num_full_layers
                )
                self.num_full_state_tensors = (
                    self.full_value_bias_offset + self.num_full_layers
                )
        else:
            self.num_full_state_tensors = self.full_value_offset + self.num_full_layers

        self.conv_state_offset = self.num_full_state_tensors
        self.recurrent_state_offset = self.conv_state_offset + self.num_linear_layers
        self.hidden_input_index = self.recurrent_state_offset + self.num_linear_layers
        self.rotary_cos_index = self.hidden_input_index + 1
        self.rotary_sin_index = self.hidden_input_index + 2
        self.attn_mask_index = self.hidden_input_index + 3

        self.quantizer = KVQuantizer(
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KEY_VALUE_HEADS,
            num_kv_groups=self.num_key_value_groups,
            is_q4=self.kv_rotary_q4,
            is_rotary=self.kv_rotary,
            is_q8_cuda=self.kv_rotary_cuda or self.kv_q8_cuda,
            use_sym=self.kv_sym,
            use_hadamard=USE_HADAMARD,
            use_clip=USE_CLIP,
            clip_sigma=CLIP_SIGMA,
            use_shuffle=USE_SHUFFLE,
        ).eval()
        self.register_buffer("overflow_scale", torch.tensor([0.01], dtype=torch.float32))
        hidden_rms_norm_eps = self.hidden_size * self.rms_norm_eps
        qk_rms_norm_eps = self.head_dim * self.rms_norm_eps
        linear_rms_norm_eps = self.linear_value_head_dim * self.rms_norm_eps
        linear_qk_rms_norm_eps = 1e-6
        if PREVENT_F16_OVERFLOW:
            hidden_rms_norm_eps *= self.overflow_scale.square()
            qk_rms_norm_eps *= self.overflow_scale.square()
            linear_rms_norm_eps *= self.overflow_scale.square()
            linear_qk_rms_norm_eps *= self.overflow_scale.square()
        self.register_buffer("hidden_rms_norm_eps", torch.tensor([hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps", torch.tensor([qk_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("linear_rms_norm_eps", torch.tensor([linear_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("linear_qk_rms_norm_eps", torch.tensor([linear_qk_rms_norm_eps], dtype=torch.float32))
        self.lm_head_weight = None

        replace_gelu_with_tanh_approximation(self.llm.model.language_model)
        self._fuse_weights()

    def _rms_norm(self, x, eps):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)

    def _rotate_half(self, x, batch_size: int):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.rotary_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.rotary_dim)

    def _fuse_weights(self):
        scale_factor = self.head_dim ** -0.25
        norm_factor = self.hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer_type, layer in zip(
                LAYER_TYPES, self.llm.model.language_model.layers
            ):
                if layer_type == "full_attention":
                    self._fuse_full_qkv_projection(
                        layer, scale_factor, norm_factor, norm_factor_qk
                    )
                else:
                    self._absorb_linear_input_norm(layer, norm_factor)
                self._fuse_gate_up_projection(layer, norm_factor)

            final_norm_weight = (
                effective_qwen_rms_weight(self.llm.model.language_model.norm).unsqueeze(0) * norm_factor
            )
            fused_lm_head_weight = (
                self.llm.lm_head.weight.data.float() * final_norm_weight
            ).contiguous()
            self.lm_head_weight = torch.nn.Parameter(fused_lm_head_weight)
            del self.llm.model.language_model.norm

    def _fuse_full_qkv_projection(
        self, layer, scale_factor: float, norm_factor: float, norm_factor_qk: float
    ):
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(
            q_proj.out_features + k_proj.out_features + v_proj.out_features
        )
        has_bias = any(proj.bias is not None for proj in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        q_weight = q_proj.weight.data.reshape(self.num_heads, 2, self.head_dim, in_features)
        q_query_weight = q_weight[:, 0].reshape(-1, in_features)
        q_gate_weight = q_weight[:, 1].reshape(-1, in_features)
        k_weight = k_proj.weight.data.reshape(-1, in_features)
        v_weight = v_proj.weight.data.reshape(-1, in_features)
        qkv.weight.data.copy_(
            torch.cat([q_query_weight, k_weight, q_gate_weight, v_weight], dim=0)
        )
        if has_bias:
            q_bias = (
                q_proj.bias
                if q_proj.bias is not None
                else qkv.weight.new_zeros(q_proj.out_features)
            )
            k_bias = (
                k_proj.bias
                if k_proj.bias is not None
                else qkv.weight.new_zeros(k_proj.out_features)
            )
            v_bias = (
                v_proj.bias
                if v_proj.bias is not None
                else qkv.weight.new_zeros(v_proj.out_features)
            )

            q_bias = q_bias.reshape(self.num_heads, 2, self.head_dim)
            q_query_bias = q_bias[:, 0].reshape(-1)
            q_gate_bias = q_bias[:, 1].reshape(-1)
            qkv.bias.data.copy_(
                torch.cat([q_query_bias, k_bias, q_gate_bias, v_bias], dim=0)
            )

        combined_scale = scale_factor * norm_factor_qk
        q_norm_weight = effective_qwen_rms_weight(attn.q_norm) * combined_scale
        k_norm_weight = effective_qwen_rms_weight(attn.k_norm) * combined_scale
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat(
                [
                    q_norm_weight.repeat(self.num_heads),
                    k_norm_weight.repeat(self.num_key_value_heads),
                ],
                dim=0,
            ).view(1, 1, 1, -1, self.head_dim)
        )

        input_norm_weight = (
            effective_qwen_rms_weight(layer.input_layernorm).unsqueeze(0) * norm_factor
        )
        qkv.weight.data.mul_(input_norm_weight)
        attn.qkv = qkv

        del attn.q_proj, attn.k_proj, attn.v_proj
        del attn.q_norm, attn.k_norm
        del layer.input_layernorm

    def _absorb_linear_input_norm(self, layer, norm_factor: float):
        linear = layer.linear_attn
        input_norm_weight = (
            effective_qwen_rms_weight(layer.input_layernorm).unsqueeze(0) * norm_factor
        )

        fused_input_projs = (
            linear.in_proj_qkv,
            linear.in_proj_z,
            linear.in_proj_b,
            linear.in_proj_a,
        )
        has_bias = any(proj.bias is not None for proj in fused_input_projs) or (
            linear.dt_bias is not None
        )
        fused_out_features = sum(int(proj.out_features) for proj in fused_input_projs)

        in_proj_all = torch.nn.Linear(
            linear.in_proj_qkv.in_features,
            fused_out_features,
            bias=has_bias,
        )
        in_proj_all = in_proj_all.to(
            device=linear.in_proj_qkv.weight.device,
            dtype=linear.in_proj_qkv.weight.dtype,
        )
        in_proj_all.weight.data.copy_(
            torch.cat([proj.weight.data for proj in fused_input_projs], dim=0)
        )
        in_proj_all.weight.data.mul_(input_norm_weight)

        if has_bias:
            bias_parts = []
            for proj in fused_input_projs:
                if proj.bias is None:
                    bias_parts.append(in_proj_all.weight.new_zeros(proj.out_features))
                else:
                    bias_parts.append(proj.bias.data)
            if linear.dt_bias is not None:
                bias_parts[-1] = bias_parts[-1] + linear.dt_bias.data.to(
                    device=in_proj_all.bias.device,
                    dtype=in_proj_all.bias.dtype,
                )
            in_proj_all.bias.data.copy_(torch.cat(bias_parts, dim=0))

        linear.in_proj_all = in_proj_all
        linear.in_proj_split_sizes = tuple(
            int(proj.out_features) for proj in fused_input_projs
        )
        linear.register_buffer("g_decay_scale", -linear.A_log.data.exp())
        del linear.in_proj_qkv, linear.in_proj_z, linear.in_proj_b, linear.in_proj_a
        del linear.A_log, linear.dt_bias
        del layer.input_layernorm

        # Fuse linear attention output norm weight into out_proj
        out_norm_weight = linear.norm.weight.data.float() * (
            self.linear_value_head_dim ** 0.5
        )
        out_norm_weight = out_norm_weight.repeat(self.linear_num_value_heads)
        linear.out_proj.weight.data.mul_(out_norm_weight.unsqueeze(0))
        del linear.norm

    def _fuse_gate_up_projection(self, layer, norm_factor: float):
        post_norm_weight = (
            effective_qwen_rms_weight(layer.post_attention_layernorm).unsqueeze(0) * norm_factor
        )
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj

        gate_up_proj = torch.nn.Linear(
            gate_proj.in_features,
            gate_proj.out_features + up_proj.out_features,
            bias=False,
        )
        gate_up_proj.weight.data.copy_(
            torch.cat(
                [
                    gate_proj.weight.data * post_norm_weight,
                    up_proj.weight.data * post_norm_weight,
                ],
                dim=0,
            )
        )
        layer.mlp.gate_up_proj = gate_up_proj
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    def _linear_full_state_input(self, all_inputs, linear_index: int):
        conv_state = all_inputs[self.conv_state_offset + linear_index].float()
        recurrent_state = all_inputs[self.recurrent_state_offset + linear_index].float()
        return conv_state, recurrent_state

    def forward(self, *all_inputs):
        hidden_states = all_inputs[self.hidden_input_index]
        rotary_pos_emb_cos = all_inputs[self.rotary_cos_index]
        rotary_pos_emb_sin = all_inputs[self.rotary_sin_index]
        attention_mask = all_inputs[self.attn_mask_index]
        batch_size = hidden_states.shape[0]

        save_full_keys = []
        save_full_values = []
        save_key_scales = []
        save_key_biases = []
        save_value_scales = []
        save_value_biases = []
        save_conv_states = []
        save_recurrent_states = []

        full_layer_index = 0
        linear_layer_index = 0
        for layer_type, layer in zip(LAYER_TYPES, self.llm.model.language_model.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

            if layer_type == "full_attention":
                attn = layer.self_attn
                qkv = attn.qkv(hidden_states)
                qkv = qkv.reshape(batch_size, -1, 2, self.qk_heads, self.head_dim)
                qk, gate_value = torch.split(qkv, 1, dim=2)

                qk = self._rms_norm(qk, self.qk_rms_norm_eps) * attn.qk_norm_weight
                if self.full_rotary:
                    qk = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin
                else:
                    qk_rot, qk_pass = torch.split(qk, [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1)
                    qk = torch.cat([qk_rot * rotary_pos_emb_cos + self._rotate_half(qk_rot, batch_size) * rotary_pos_emb_sin, qk_pass], dim=-1)

                query, key = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
                gate, value = torch.split(gate_value, [self.num_heads, self.num_key_value_heads], dim=-2)
                gate = gate.reshape(batch_size, -1, self.num_heads * self.head_dim)
                query = query.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                query = query.permute(0, 2, 3, 1, 4)

                if self.kv_f16:
                    key = key.half()
                    value = value.half()

                key = key.permute(0, 3, 2, 4, 1)
                value = value.transpose(1, 3)

                if self.kv_rotary_q4:
                    # ── ROTARY_Q4 ────────────────────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_value_scales.append(v_s)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            v_s = v_s.float()

                        if self.kv_rotary_q4_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_k(k, batch_size)).float()
                        q_rot = self.quantizer.rotate_q(query, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        attn_output = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)

                        v_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_v(v, batch_size)).float()
                        v_q_g = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn_output = torch.matmul(attn_output, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                        if self.quantizer.use_shuffle:
                            attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                        attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        k_b = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                        v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                        v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-3)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_key_biases.append(k_b)
                        save_value_scales.append(v_s)
                        save_value_biases.append(v_b)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            k_b = k_b.float()
                            v_s = v_s.float()
                            v_b = v_b.float()

                        if self.kv_rotary_q4_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_unpacked = self.quantizer.unpack_q4_k(k, batch_size).float()
                        q_rot = self.quantizer.rotate_q(query, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        q_sum_g = q_rot_g.sum(dim=-1, keepdim=True)
                        attn_output = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)

                        v_unpacked = self.quantizer.unpack_q4_v(v, batch_size).float()
                        v_q_g = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn_output = torch.matmul(attn_output, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                        if self.quantizer.use_shuffle:
                            attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                        attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)

                elif self.kv_rotary:
                    # ── ROTARY_Q8 ────────────────────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-2)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_value_scales.append(v_s)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            v_s = v_s.float()

                        if self.kv_rotary_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                        v_signed = self.quantizer._decode_signed_q8_storage(v).float()

                        if self.kv_q8_grouped:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            if self.quantizer.use_shuffle:
                                q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                            q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_rot_g = q_rot_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                            k_q_g = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                            attn_output = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_q_g = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn_output = torch.matmul(attn_output, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                            if self.quantizer.use_shuffle:
                                attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                            attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                        else:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            attn_raw = torch.matmul(q_rot, k_signed)
                            attn_output = attn_raw * k_s + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_scaled = v_signed * v_s
                            attn_output = self.quantizer.inverse_rotate_attn(torch.matmul(attn_output, v_scaled), batch_size)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                        v = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                        k_b = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-3)
                            v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[self.full_value_scale_offset + full_layer_index], scale_v], dim=-2)
                            v_b = torch.cat([all_inputs[self.full_value_bias_offset + full_layer_index], bias_v], dim=-2)

                        save_full_keys.append(k)
                        save_full_values.append(v)
                        save_key_scales.append(k_s)
                        save_key_biases.append(k_b)
                        save_value_scales.append(v_s)
                        save_value_biases.append(v_b)

                        if USE_FLOAT16_SCALE_BIAS:
                            k_s = k_s.float()
                            k_b = k_b.float()
                            v_s = v_s.float()
                            v_b = v_b.float()

                        if self.kv_rotary_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)

                        if self.kv_q8_grouped:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            if self.quantizer.use_shuffle:
                                q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                            q_rot_g = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_rot_g = q_rot_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                            k_q_g = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                            q_sum_g = q_rot_g.sum(dim=-1, keepdim=True)
                            attn_output = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_q_g = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn_output = torch.matmul(attn_output, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                            if self.quantizer.use_shuffle:
                                attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                            attn_output = self.quantizer.inverse_rotate_attn(attn_output, batch_size)
                        else:
                            q_rot = self.quantizer.rotate_q(query, batch_size)
                            attn_raw = torch.matmul(q_rot, k.float())
                            q_sum = query.sum(dim=-1, keepdim=True)
                            attn_output = attn_raw * k_s + q_sum * k_b + attention_mask
                            attn_output = torch.softmax(attn_output, dim=-1)

                            v_dequant = v.float() * v_s + v_b
                            attn_output = self.quantizer.inverse_rotate_attn(torch.matmul(attn_output, v_dequant), batch_size)

                elif self.kv_quantized:
                    packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(key, value, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                    key_cache = torch.cat([all_inputs[self.full_key_offset + full_layer_index], packed_k], dim=-1)
                    value_cache = torch.cat([all_inputs[self.full_value_offset + full_layer_index], packed_v], dim=-2)
                    key_scale = torch.cat([all_inputs[self.full_key_scale_offset + full_layer_index], scale_k], dim=-1)
                    key_bias = torch.cat([all_inputs[self.full_key_bias_offset + full_layer_index], bias_k], dim=-1)
                    value_scale = torch.cat(
                        [all_inputs[self.full_value_scale_offset + full_layer_index], scale_v],
                        dim=3 if self.kv_q8_grouped else -2,
                    )
                    value_bias = torch.cat(
                        [all_inputs[self.full_value_bias_offset + full_layer_index], bias_v],
                        dim=3 if self.kv_q8_grouped else -2,
                    )

                    save_full_keys.append(key_cache)
                    save_full_values.append(value_cache)
                    save_key_scales.append(key_scale)
                    save_key_biases.append(key_bias)
                    save_value_scales.append(value_scale)
                    save_value_biases.append(value_bias)

                    if USE_FLOAT16_SCALE_BIAS:
                        key_scale = key_scale.float()
                        key_bias = key_bias.float()
                        value_scale = value_scale.float()
                        value_bias = value_bias.float()

                    if self.kv_q8_cuda:
                        key_cache = self.quantizer.unpack_cuda(key_cache, -2, batch_size, self.num_key_value_heads, self.head_dim)
                        value_cache = self.quantizer.unpack_cuda(value_cache, -1, batch_size, self.num_key_value_heads, self.head_dim)

                    if self.kv_q8_grouped:
                        q_grouped = query
                        if self.quantizer.use_shuffle:
                            q_grouped = q_grouped.index_select(-1, self.quantizer.shuffle_idx)
                        q_grouped = q_grouped.view(
                            batch_size,
                            self.num_key_value_heads,
                            self.num_key_value_groups,
                            -1,
                            self.quantizer.kv_quant_num_groups,
                            self.quantizer.kv_quant_group_size,
                        )
                        q_grouped = q_grouped.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_grouped = self.quantizer.hadamard_q(q_grouped)

                        k_grouped = key_cache.float().view(
                            batch_size,
                            self.num_key_value_heads,
                            1,
                            self.quantizer.kv_quant_num_groups,
                            self.quantizer.kv_quant_group_size,
                            -1,
                        )
                        attn_raw_grouped = torch.matmul(q_grouped, k_grouped)
                        q_sum_grouped = q_grouped.sum(dim=-1, keepdim=True)
                        attn_output = (
                            attn_raw_grouped * key_scale + q_sum_grouped * key_bias
                        ).sum(dim=-3) + attention_mask
                        attn_output = torch.softmax(attn_output, dim=-1)

                        v_grouped = value_cache.float().view(
                            batch_size,
                            self.num_key_value_heads,
                            1,
                            -1,
                            self.quantizer.kv_quant_num_groups,
                            self.quantizer.kv_quant_group_size,
                        )
                        value_dequant = (
                            v_grouped * value_scale + value_bias
                        ).reshape(
                            batch_size,
                            self.num_key_value_heads,
                            1,
                            -1,
                            self.head_dim,
                        )
                        attn_output = torch.matmul(attn_output, value_dequant)
                        if self.quantizer.use_hadamard:
                            attn_output = self.quantizer.inverse_hadamard_attn(attn_output, batch_size)
                        if self.quantizer.use_shuffle:
                            attn_output = attn_output.index_select(-1, self.quantizer.unshuffle_idx)
                    else:
                        attn_raw = torch.matmul(query, key_cache.float())
                        attn_bias = query.sum(dim=-1, keepdim=True) * key_bias + attention_mask
                        attn_output = torch.addcmul(attn_bias, attn_raw, key_scale)
                        attn_output = torch.softmax(attn_output, dim=-1)
                        value_dequant = torch.addcmul(value_bias, value_cache.float(), value_scale)
                        attn_output = torch.matmul(attn_output, value_dequant)
                else:
                    key_cache = torch.cat([all_inputs[self.full_key_offset + full_layer_index], key], dim=-1)
                    value_cache = torch.cat([all_inputs[self.full_value_offset + full_layer_index], value], dim=-2)
                    save_full_keys.append(key_cache)
                    save_full_values.append(value_cache)

                    if self.kv_f16:
                        key_cache = key_cache.float()
                        value_cache = value_cache.float()

                    attn_output = torch.matmul(query, key_cache) + attention_mask
                    attn_output = torch.softmax(attn_output, dim=-1)
                    attn_output = torch.matmul(attn_output, value_cache)

                attn_output = attn_output.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, attn.o_proj.in_features)
                if ATTN_OUTPUT_GATE:
                    attn_output = attn_output * torch.sigmoid(gate)
                hidden_states = residual + attn.o_proj(attn_output)
                full_layer_index += 1
            else:
                linear = layer.linear_attn
                conv_state, recurrent_state = self._linear_full_state_input(all_inputs, linear_layer_index)

                linear_inputs = linear.in_proj_all(hidden_states)
                mixed_qkv, z, beta_logits, g_logits = torch.split(linear_inputs, linear.in_proj_split_sizes, dim=-1)
                beta = torch.sigmoid(beta_logits)
                g = linear.g_decay_scale * F.softplus(g_logits)

                conv_input = torch.cat([conv_state, mixed_qkv], dim=1)
                conv_state_out = conv_input[:, -LINEAR_CONV_STATE_LEN:]
                conv_output = F.conv1d(conv_input.transpose(1, 2), linear.conv1d.weight, linear.conv1d.bias, padding=0, groups=LINEAR_CONV_DIM)
                conv_output = F.silu(conv_output).transpose(1, 2)
                conv_output = conv_output.reshape(batch_size, -1, self.linear_num_key_heads * 2 + self.linear_num_value_heads, self.linear_key_head_dim)

                qk, value = torch.split(conv_output, [self.linear_num_key_heads * 2, self.linear_num_value_heads], dim=2)
                qk = self._rms_norm(qk, self.linear_qk_rms_norm_eps)
                query, key = torch.split(qk, [self.linear_num_key_heads, self.linear_num_key_heads], dim=2)
                query = query * self.linear_gated_delta_query_scale

                core_attn_out, recurrent_state_out = recurrent_gated_delta_prefill(query, key, value, g, beta, recurrent_state)
                core_attn_out = self._rms_norm(core_attn_out, self.linear_rms_norm_eps)
                core_attn_out = core_attn_out.reshape(batch_size, -1, self.linear_value_dim)
                core_attn_out = core_attn_out * F.silu(z)

                hidden_states = residual + linear.out_proj(core_attn_out)

                save_conv_states.append(conv_state_out.half())
                save_recurrent_states.append(recurrent_state_out.half())

                linear_layer_index += 1

            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate_part, up_part = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate_part) * up_part)

        hidden_states = self._rms_norm(hidden_states[:, -1], self.hidden_rms_norm_eps)
        logits = F.linear(hidden_states, self.lm_head_weight)

        if self.kv_any_quantized:
            if self.kv_sym:
                return (
                    *save_full_keys,
                    *save_full_values,
                    *save_key_scales,
                    *save_value_scales,
                    *save_conv_states,
                    *save_recurrent_states,
                    logits,
                )
            return (
                *save_full_keys,
                *save_full_values,
                *save_key_scales,
                *save_key_biases,
                *save_value_scales,
                *save_value_biases,
                *save_conv_states,
                *save_recurrent_states,
                logits,
            )

        return (
            *save_full_keys,
            *save_full_values,
            *save_conv_states,
            *save_recurrent_states,
            logits,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Export Helpers
# ══════════════════════════════════════════════════════════════════════════════
def build_full_state_tensors(batch_size: int):
    """Build empty full-attention KV tensors with the exported cache layout."""
    if KV_QUANT_DTYPE == "F16":
        kv_dtype = torch.float16
    elif KV_QUANT_DTYPE == "F32":
        kv_dtype = torch.float32
    elif KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"):
        kv_dtype = torch.int32
    elif _kv_sym and not _is_rotary_q4_kv:
        kv_dtype = torch.int8
    else:
        kv_dtype = torch.uint8

    # Determine KV tensor shapes based on quantization mode
    if KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA"):
        key_cache_head_dim = HEAD_DIM // 4
        value_cache_head_dim = HEAD_DIM // 4
    elif KV_QUANT_DTYPE == "ROTARY_Q4":
        key_cache_head_dim = HEAD_DIM // 2
        value_cache_head_dim = HEAD_DIM // 2
    elif KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
        key_cache_head_dim = HEAD_DIM // 8
        value_cache_head_dim = HEAD_DIM // 8
    else:
        key_cache_head_dim = HEAD_DIM
        value_cache_head_dim = HEAD_DIM

    tensors = {
        "key": torch.zeros(
            (batch_size, NUM_KEY_VALUE_HEADS, 1, key_cache_head_dim, 0), dtype=kv_dtype
        ),
        "value": torch.zeros(
            (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, value_cache_head_dim),
            dtype=kv_dtype,
        ),
    }
    if _is_quantized_kv or _is_rotary_kv:
        if _grouped_6d:
            kv_quant_num_groups = HEAD_DIM // KV_QUANT_GROUP_SIZE
            tensors["key_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, kv_quant_num_groups, 1, 0),
                dtype=SCALE_DTYPE_TORCH,
            )
            tensors["value_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, kv_quant_num_groups, 1),
                dtype=SCALE_DTYPE_TORCH,
            )
            if not _kv_sym:
                tensors["key_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, kv_quant_num_groups, 1, 0),
                    dtype=SCALE_DTYPE_TORCH,
                )
                tensors["value_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, kv_quant_num_groups, 1),
                    dtype=SCALE_DTYPE_TORCH,
                )
        else:
            tensors["key_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 1, 0), dtype=SCALE_DTYPE_TORCH
            )
            tensors["value_scale"] = torch.ones(
                (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, 1), dtype=SCALE_DTYPE_TORCH
            )
            if not _kv_sym:
                tensors["key_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 1, 0), dtype=SCALE_DTYPE_TORCH
                )
                tensors["value_bias"] = torch.ones(
                    (batch_size, NUM_KEY_VALUE_HEADS, 1, 0, 1), dtype=SCALE_DTYPE_TORCH
                )
    return tensors


def build_linear_state_tensors(batch_size: int):
    """Build empty linear-attention state tensors for export and runtime."""
    return {
        "conv_state": torch.zeros(
            (batch_size, LINEAR_CONV_STATE_LEN, LINEAR_CONV_DIM), dtype=torch.float16
        ),
        "recurrent_state": torch.zeros(
            (
                batch_size,
                LINEAR_NUM_VALUE_HEADS,
                LINEAR_KEY_HEAD_DIM,
                LINEAR_VALUE_HEAD_DIM,
            ),
            dtype=torch.float16,
        ),
    }


def get_full_kv_io(
    tensors_dict,
    batch_axis="batch_size",
    seq_axis="history_len",
    out_seq_axis="kv_seq_len",
):
    """Create ordered ONNX IO lists and dynamic axes for full-attention states."""
    inputs = []
    input_names = []
    output_names = []
    axes = {}
    for name, dim in FULL_STATE_SPECS:
        tensor = tensors_dict[name]
        for layer_index in range(NUM_FULL_ATTENTION_LAYERS):
            input_name = f"in_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            axes[input_name] = {0: batch_axis, dim: seq_axis}
            axes[output_name] = {0: batch_axis, dim: out_seq_axis}
    return inputs, input_names, output_names, axes


def get_linear_state_io(tensors_dict, batch_axis="batch_size"):
    """Create ordered ONNX IO lists and dynamic axes for linear-attention states."""
    inputs = []
    input_names = []
    output_names = []
    axes = {}
    for name, _ in LINEAR_STATE_SPECS:
        tensor = tensors_dict[name]
        for layer_index in range(NUM_LINEAR_ATTENTION_LAYERS):
            input_name = f"in_{name}_{layer_index}"
            output_name = f"out_{name}_{layer_index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            axes[input_name] = {0: batch_axis}
            axes[output_name] = {0: batch_axis}
    return inputs, input_names, output_names, axes


# ══════════════════════════════════════════════════════════════════════════════
# Export driver
# ══════════════════════════════════════════════════════════════════════════════
if DO_EXPORT:
    print("Export start ...")

    with torch.inference_mode():
        model, tokenizer = load_model_and_tokenizer()
        prompt_head_len = compute_prompt_head_len(tokenizer)
        between_len = compute_between_len(tokenizer)

        for note in normalize_kv_quant_settings(HEAD_DIM):
            print(f"\n{note}")

        batch_size = BEAM_SIZE
        ids_len_dummy = 10
        history_len_dummy = 0
        ids_len = torch.tensor([ids_len_dummy], dtype=torch.int64)
        history_len = torch.tensor([history_len_dummy], dtype=torch.int64)
        kv_seq_len = ids_len + history_len
        beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        top_k = torch.tensor([TOP_K], dtype=torch.int64)
        logits = torch.ones((BEAM_SIZE, VOCAB_SIZE), dtype=torch.float32)

        full_state_tensors = build_full_state_tensors(batch_size)
        linear_state_tensors = build_linear_state_tensors(batch_size)
        full_inputs, full_input_names, full_output_names, full_dynamic_axes = (
            get_full_kv_io(full_state_tensors)
        )
        linear_inputs, linear_input_names, linear_output_names, linear_dynamic_axes = (
            get_linear_state_io(linear_state_tensors)
        )
        state_inputs = full_inputs + linear_inputs
        state_input_names = full_input_names + linear_input_names
        state_output_names = full_output_names + linear_output_names
        state_dynamic_axes = {**full_dynamic_axes, **linear_dynamic_axes}
        total_state_tensors = len(state_input_names)

        # ── Export LLM_Embed ──
        input_ids = torch.ones((1, ids_len_dummy), dtype=torch.int32)
        torch.onnx.export(
            LLM_EMBED(model).eval(),
            (input_ids,),
            onnx_model_Embed,
            input_names=["input_ids"],
            output_names=["text_hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "ids_len"},
                "text_hidden_states": {0: "batch", 1: "ids_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del input_ids
        gc.collect()

        # ── Export fused LLM_Vision (preprocess + encode + concat) ──
        if INPUT_IMAGE_DIM == 5:
            pixel_values_img = torch.randint(
                0, 255,
                size=(VISION_BATCH_SIZE, 1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]),
                dtype=torch.uint8,
            )
        else:
            pixel_values_img = torch.randint(
                0, 255,
                size=(VISION_BATCH_SIZE, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]),
                dtype=torch.uint8,
            )
        text_hidden_states = torch.ones((1, ids_len_dummy, HIDDEN_SIZE), dtype=torch.float32)
        vision_model = LLM_VISION(
            model, HEIGHT_FACTOR, WIDTH_FACTOR, VISION_BATCH_SIZE,
            prompt_head_len, between_len, DYNAMIC_IMAGE_SHAPE
        ).eval()
        torch.onnx.export(
            vision_model,
            (pixel_values_img, text_hidden_states),
            onnx_model_Vision,
            input_names=["pixel_values", "text_hidden_states"],
            output_names=["concat_hidden_states"],
            dynamic_axes={
                "pixel_values": (
                    {0: "num_images", 3: "image_h", 4: "image_w"} if INPUT_IMAGE_DIM == 5
                    else {0: "num_images", 2: "image_h", 3: "image_w"}
                ) if DYNAMIC_IMAGE_SHAPE else {},
                "text_hidden_states": {1: "ids_len"},
                "concat_hidden_states": {1: "total_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del pixel_values_img, text_hidden_states, vision_model
        gc.collect()

        # ── Export Rotary modules ──
        torch.onnx.export(
            ROTARY_PREFILL(model, tokenizer, VISION_BATCH_SIZE, IMAGE_SEQLEN_PER_IMAGE, MAX_SEQ_LEN).eval(),
            (ids_len, history_len),
            onnx_model_Rotary_Prefill,
            input_names=["ids_len", "history_len"],
            output_names=["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
            dynamic_axes={
                "rotary_cos": {1: "ids_len"},
                "rotary_sin": {1: "ids_len"},
                "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        torch.onnx.export(
            ROTARY_DECODE(model, tokenizer, VISION_BATCH_SIZE, IMAGE_SEQLEN_PER_IMAGE, MAX_SEQ_LEN).eval(),
            (kv_seq_len,),
            onnx_model_Rotary_Decode,
            input_names=["kv_seq_len"],
            output_names=["rotary_cos", "rotary_sin", "kv_seq_len_next"],
            opset_version=OPSET,
            dynamo=False,
        )

        # ── Export LLM_Main ──
        ids_len_main_dummy = ids_len_dummy + VISION_EMBED_SIZE
        hidden_states = torch.ones(
            (batch_size, ids_len_main_dummy, HIDDEN_SIZE), dtype=torch.float32
        )
        rotary_cos = torch.zeros(
            (1, ids_len_main_dummy, 1, 1, ROTARY_DIM), dtype=torch.float32
        )
        rotary_sin = torch.zeros(
            (1, ids_len_main_dummy, 1, 1, ROTARY_DIM), dtype=torch.float32
        )
        attention_mask = torch.zeros(
            (1, 1, 1, ids_len_main_dummy, ids_len_main_dummy), dtype=torch.float32
        )

        model_main = LLM_MAIN(model).eval()
        del model
        gc.collect()

        torch.onnx.export(
            model_main,
            tuple(
                state_inputs + [hidden_states, rotary_cos, rotary_sin, attention_mask]
            ),
            onnx_model_Main,
            input_names=state_input_names
            + ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"],
            output_names=state_output_names + ["logits"],
            dynamic_axes={
                **state_dynamic_axes,
                "hidden_states": {0: "batch", 1: "ids_len"},
                "rotary_cos": {1: "ids_len"},
                "rotary_sin": {1: "ids_len"},
                "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del model_main, hidden_states, rotary_cos, rotary_sin, attention_mask
        gc.collect()

        # ── Export decoding modules ──
        save_id_in = torch.zeros((BEAM_SIZE, 4), dtype=torch.int32)
        torch.onnx.export(
            GREEDY_SEARCH().eval(),
            (logits, save_id_in),
            onnx_model_Greedy,
            input_names=["logits", "save_id_in"],
            output_names=["max_logits_idx", "save_id_out"],
            dynamic_axes={
                "logits": {0: "batch"},
                "save_id_in": {0: "batch", 1: "history_len"},
                "max_logits_idx": {0: "batch"},
                "save_id_out": {0: "batch", 1: "history_len"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        single_full_state_tensors = build_full_state_tensors(1)
        single_linear_state_tensors = build_linear_state_tensors(1)
        first_full_inputs, first_full_input_names, _, first_full_axes = get_full_kv_io(
            single_full_state_tensors
        )
        first_linear_inputs, first_linear_input_names, _, first_linear_axes = (
            get_linear_state_io(single_linear_state_tensors)
        )
        first_state_inputs = first_full_inputs + first_linear_inputs
        first_state_input_names = first_full_input_names + first_linear_input_names
        first_state_output_names = [
            name.replace("in_", "out_", 1) for name in first_state_input_names
        ]

        torch.onnx.export(
            FIRST_BEAM_SEARCH(total_state_tensors).eval(),
            tuple(first_state_inputs + [logits[[0]], save_id_in, beam_size]),
            onnx_model_First_Beam,
            input_names=first_state_input_names + ["logits", "save_id_in", "beam_size"],
            output_names=first_state_output_names
            + ["save_id_out", "top_beam_prob", "top_beam_indices", "max_logits_idx"],
            dynamic_axes={
                **first_full_axes,
                **first_linear_axes,
                "logits": {0: "batch"},
                "save_id_in": {0: "batch", 1: "history_len"},
                "save_id_out": {0: "batch", 1: "history_len"},
                "top_beam_prob": {0: "batch"},
                "top_beam_indices": {0: "batch"},
                "max_logits_idx": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
        torch.onnx.export(
            SECOND_BEAM_SEARCH(total_state_tensors).eval(),
            tuple(state_inputs + [logits, save_id_in, previous_prob, beam_size, top_k]),
            onnx_model_Second_Beam,
            input_names=state_input_names
            + ["logits", "save_id_in", "previous_prob", "beam_size", "topK"],
            output_names=state_output_names
            + ["save_id_out", "top_beam_prob", "top_beam_indices", "max_logits_idx"],
            dynamic_axes={
                **state_dynamic_axes,
                "logits": {0: "batch"},
                "save_id_in": {0: "batch", 1: "history_len"},
                "previous_prob": {0: "batch"},
                "save_id_out": {0: "batch", 1: "history_len"},
                "top_beam_prob": {0: "batch"},
                "top_beam_indices": {0: "batch"},
                "max_logits_idx": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del previous_prob

        penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALTY_RANGE], dtype=torch.int64)
        torch.onnx.export(
            APPLY_PENALTY().eval(),
            (logits, save_id_in, penalty_value, penalty_range),
            onnx_model_Penalty,
            input_names=["logits_in", "save_id_in", "penalty_value", "penalty_range"],
            output_names=["logits_out"],
            dynamic_axes={
                "logits_in": {0: "batch"},
                "save_id_in": {0: "batch", 1: "history_len"},
                "logits_out": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
        del penalty_value, penalty_range

        torch.onnx.export(
            ARGMAX().eval(),
            (logits,),
            onnx_model_Argmax,
            input_names=["logits"],
            output_names=["max_logits_idx"],
            dynamic_axes={
                "logits": {0: "batch"},
                "max_logits_idx": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )

        slice_start = torch.tensor([0], dtype=torch.int64)
        slice_end = torch.tensor([5], dtype=torch.int64)
        torch.onnx.export(
            KV_SLICE(NUM_FULL_ATTENTION_LAYERS).eval(),
            tuple(full_inputs + [slice_start, slice_end]),
            onnx_model_KV_Slice,
            input_names=full_input_names + ["slice_start", "slice_end"],
            output_names=full_output_names,
            dynamic_axes=full_dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )

        del (
            tokenizer,
            logits,
            save_id_in,
            beam_size,
            top_k,
            full_state_tensors,
            linear_state_tensors,
        )
        del (
            single_full_state_tensors,
            single_linear_state_tensors,
            first_state_inputs,
            state_inputs,
        )
        gc.collect()

    print("Export done. Starting ORT runtime.")


# ORT Runtime Helpers
# ══════════════════════════════════════════════════════════════════════════════
def bind_ort_in_buf(binding, names, values):
    """Bind a sequence of ORT input buffers by name."""
    for name, value in zip(names, values):
        binding.bind_ortvalue_input(name, value)


def bind_ort_out_buf(binding, names, values):
    """Bind a sequence of preallocated ORT output buffers by name."""
    for name, value in zip(names, values):
        binding.bind_ortvalue_output(name, value)


def bind_ort_out(binding, names, device):
    """Bind output names while letting ORT allocate storage on `device`."""
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    """Create an OrtValue from Python data using the requested device and dtype."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.array(data, dtype=dtype), device, device_id
    )


def create_ort_with_shape(shape, dtype, device, device_id):
    """Create a zero-filled OrtValue with the provided shape and dtype."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype), device, device_id
    )


def create_session(
    model_path, _session_opts, _providers, _provider_options, _disabled_optimizers
):
    """Create a standard ORT InferenceSession for one exported model."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers,
    )


def get_in_names(session):
    return [item.name for item in session.get_inputs()]


def get_out_names(session):
    return [item.name for item in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def select_state_input_names(names, prefix, exclude_substrings=()):
    """Filter exported state input names by prefix while excluding sub-patterns."""
    return [
        name
        for name in names
        if name.startswith(prefix)
        and all(token not in name for token in exclude_substrings)
    ]


def create_ort_with_meta_shape(
    meta, dtype, device, device_id, batch_size=1, seq_axis=None, seq_len=0
):
    """Create a zero OrtValue from ORT metadata, resolving symbolic dims locally."""
    shape = list(meta.shape)
    rank = len(shape)

    if seq_axis is not None and seq_axis < 0:
        seq_axis += rank

    for idx, dim in enumerate(shape):
        if idx == 0:
            shape[idx] = batch_size
        elif seq_axis is not None and idx == seq_axis:
            shape[idx] = seq_len
        elif not isinstance(dim, (int, np.integer)):
            shape[idx] = 1

    return create_ort_with_shape(tuple(shape), dtype, device, device_id)


# ══════════════════════════════════════════════════════════════════════════════
# ORT Session & Runtime Options
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()

for options in (session_opts, run_options):
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4

session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session_config_entries = {
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
for key, value in session_config_entries.items():
    session_opts.add_session_config_entry(key, value)

run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
disabled_optimizers = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16
    else None
)


# ══════════════════════════════════════════════════════════════════════════════
# Execution Provider Configuration
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
        'gpu_mem_limit':                      24 * (1024 **3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '1',                # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '1',
        'enable_cuda_graph':                  '0',                # Disable to avoid loading error with some models; can be re-enabled if not an issue
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

_ort_device_type = C.OrtDevice(
    _ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID
)
kv_device = "cpu" if device_type == "dml" else device_type

# ══════════════════════════════════════════════════════════════════════════════
# Load ONNX Sessions
# ══════════════════════════════════════════════════════════════════════════════
# --- Embed ---
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

# --- Vision (fused: preprocess + encode + concat) ---
ort_session_Vision = create_session(onnx_model_Vision, **packed_settings)
binding_Vision     = ort_session_Vision.io_binding()
in_name_Vision     = get_in_names(ort_session_Vision)
out_name_Vision    = get_out_names(ort_session_Vision)[0]

# Read vision config from exported ONNX model metadata
_vision_in_meta = ort_session_Vision._inputs_meta[0]
vision_batch_size = _vision_in_meta.shape[0]
vision_embed_size = VISION_EMBED_SIZE

_img_meta_shape = _vision_in_meta.shape
if len(_img_meta_shape) == 5:
    _img_h, _img_w = _img_meta_shape[3], _img_meta_shape[4]
else:
    _img_h, _img_w = _img_meta_shape[2], _img_meta_shape[3]
if isinstance(_img_h, int) and isinstance(_img_w, int):
    input_image_size = [_img_h, _img_w]
else:
    _test_imgs = TEST_IMAGE if isinstance(TEST_IMAGE, list) else [TEST_IMAGE]
    _first_valid = next((p for p in _test_imgs if is_valid_image_path(p)), None)
    if _first_valid:
        with Image.open(_first_valid) as _img:
            input_image_size = [_img.height, _img.width]
    else:
        input_image_size = INPUT_IMAGE_SIZE

# --- Rotary Image (Prefill + Decode) ---
ort_session_Rotary_Prefill = create_session(onnx_model_Rotary_Prefill, **packed_settings)
binding_Rotary_Prefill  = ort_session_Rotary_Prefill.io_binding()
in_name_Rotary_Prefill  = get_in_names(ort_session_Rotary_Prefill)
out_name_Rotary_Prefill = get_out_names(ort_session_Rotary_Prefill)

ort_session_Rotary_Decode = create_session(onnx_model_Rotary_Decode, **packed_settings)
binding_Rotary_Decode  = ort_session_Rotary_Decode.io_binding()
in_name_Rotary_Decode  = get_in_names(ort_session_Rotary_Decode)[0]
out_name_Rotary_Decode = get_out_names(ort_session_Rotary_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
print(f"Usable Providers: {ort_session_Main.get_providers()}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Model Metadata & Index Offsets
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main         = get_in_names(ort_session_Main)
out_name_Main        = get_out_names(ort_session_Main)
in_meta_Main         = ort_session_Main._inputs_meta
in_meta_Main_by_name = {name: meta for name, meta in zip(in_name_Main, in_meta_Main)}

num_keys_values_Main        = len(out_name_Main) - 1
num_keys_values_Main_plus_1 = num_keys_values_Main + 1
num_keys_values_Main_plus_2 = num_keys_values_Main + 2
num_keys_values_Main_plus_3 = num_keys_values_Main + 3

# Main model non-state input indices
# Layout: [state_tensors..., hidden_states, rotary_cos, rotary_sin, attention_mask]
idx_rotary_cos = num_keys_values_Main + 1

# Partitioned name lists
in_name_Main_kv      = in_name_Main[:num_keys_values_Main]
out_name_Main_kv     = out_name_Main[:num_keys_values_Main]
out_name_Main_logits = out_name_Main[num_keys_values_Main]

# State layout derived from exported ONNX input names
in_name_Main_keys = select_state_input_names(in_name_Main_kv, "in_key_", exclude_substrings=("scale", "bias"))
in_name_Main_values = select_state_input_names(in_name_Main_kv, "in_value_", exclude_substrings=("scale", "bias"))
in_name_Main_key_scales   = select_state_input_names(in_name_Main_kv, "in_key_scale_")
in_name_Main_key_biases   = select_state_input_names(in_name_Main_kv, "in_key_bias_")
in_name_Main_value_scales = select_state_input_names(in_name_Main_kv, "in_value_scale_")
in_name_Main_value_biases = select_state_input_names(in_name_Main_kv, "in_value_bias_")
in_name_Main_conv_states  = select_state_input_names(in_name_Main_kv, "in_conv_state_")
in_name_Main_recurrent_states = select_state_input_names(in_name_Main_kv, "in_recurrent_state_")

num_full_layers_Main   = len(in_name_Main_keys)
num_linear_layers_Main = len(in_name_Main_conv_states)

kv_dtype_str = in_meta_Main[0].type
hidden_dtype_Main = np.float16 if "float16" in in_meta_Main[num_keys_values_Main].type else np.float32

vocab_size = ort_session_Main._outputs_meta[num_keys_values_Main].shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# State buffer setup
# ══════════════════════════════════════════════════════════════════════════════
if "uint8" in kv_dtype_str or "int8" in kv_dtype_str or "int32" in kv_dtype_str:
    if "int32" in kv_dtype_str:
        kv_dtype_Main = np.int32
    elif "uint8" in kv_dtype_str:
        kv_dtype_Main = np.uint8
    else:
        kv_dtype_Main = np.int8

    if in_name_Main_key_scales:
        scale_dtype_Main = (
            np.float16
            if "float16" in in_meta_Main_by_name[in_name_Main_key_scales[0]].type
            else np.float32
        )
        k_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_key_scales[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
        )
        v_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_value_scales[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
        )
        k_biases_Main = (
            create_ort_with_meta_shape(
                in_meta_Main_by_name[in_name_Main_key_biases[0]],
                scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
            ) if in_name_Main_key_biases else None
        )
        v_biases_Main = (
            create_ort_with_meta_shape(
                in_meta_Main_by_name[in_name_Main_value_biases[0]],
                scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
            ) if in_name_Main_value_biases else None
        )
    else:
        k_scales_Main = None
        k_biases_Main = None
        v_scales_Main = None
        v_biases_Main = None
else:
    kv_dtype_Main = np.float16 if "float16" in kv_dtype_str else np.float32
    k_scales_Main = None
    k_biases_Main = None
    v_scales_Main = None
    v_biases_Main = None

past_keys_Main = create_ort_with_meta_shape(
    in_meta_Main_by_name[in_name_Main_keys[0]], kv_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
)
past_values_Main = create_ort_with_meta_shape(
    in_meta_Main_by_name[in_name_Main_values[0]], kv_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
)
past_conv_states_Main = (
    create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_conv_states[0]], np.float16, kv_device, DEVICE_ID,
    ) if in_name_Main_conv_states else None
)
past_recurrent_states_Main = (
    create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_recurrent_states[0]], np.float16, kv_device, DEVICE_ID,
    ) if in_name_Main_recurrent_states else None
)


# ══════════════════════════════════════════════════════════════════════════════
# Determine mode and build prompt
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
STOP_TOKEN_SET = set(STOP_TOKEN)


# Determine test modes
test_image_paths = TEST_IMAGE if isinstance(TEST_IMAGE, list) else ([TEST_IMAGE] if TEST_IMAGE else [])
valid_images = [p for p in test_image_paths if is_valid_image_path(p)]
num_runtime_images = min(len(valid_images), vision_batch_size)

if not valid_images:
    raise RuntimeError("Surya OCR requires image input. No valid image found in TEST_IMAGE.")
test_modes = [("image", query) for query in PROMPT_MAP.values()]


# ══════════════════════════════════════════════════════════════════════════════
# One-time image processing: encode vision embeddings once, reuse for all tasks
# ══════════════════════════════════════════════════════════════════════════════
runtime_prompt_head_len = compute_prompt_head_len(tokenizer)
cached_vision_embeds = None

if valid_images:
    print("\nStart to Process the Image(s) (one-time)...")
    vision_start_time = time.time()

    # Load and preprocess images
    images = []
    for img_path in valid_images[:num_runtime_images]:
        images.append(load_image_letterbox(img_path, input_image_size[0], input_image_size[1]))
    blank_image = np.full((3, input_image_size[0], input_image_size[1]), 128, dtype=np.uint8)
    while len(images) < vision_batch_size:
        images.append(blank_image)
    pixel_values = np.stack(images, axis=0)
    pixel_values = np.expand_dims(pixel_values, axis=1)

    # Run Embed with the first task's prompt to get text_hidden_states
    _first_query = list(PROMPT_MAP.values())[0]
    _first_tokens, _ = build_prompt_and_mm_types(_first_query, num_images=vision_batch_size)
    _first_input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(_first_tokens, device_type, DEVICE_ID)
    _bind_embed = ort_session_Embed.io_binding()
    _bind_embed.bind_ortvalue_input(in_name_Embed, _first_input_ids)
    bind_ort_out(_bind_embed, [out_name_Embed], _ort_device_type)
    run(ort_session_Embed, _bind_embed)
    _first_hidden = _bind_embed.get_outputs()[0]

    # Run Vision model (the expensive step) once
    _bind_vision = ort_session_Vision.io_binding()
    _bind_vision.bind_ortvalue_input(in_name_Vision[0], onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, DEVICE_ID))
    _bind_vision.bind_ortvalue_input(in_name_Vision[1], _first_hidden)
    bind_ort_out(_bind_vision, [out_name_Vision], _ort_device_type)
    run(ort_session_Vision, _bind_vision)
    _first_concat = _bind_vision.get_outputs()[0]

    # Extract and cache the vision embeddings (independent of text content)
    _concat_np = _first_concat.numpy()
    cached_vision_embeds = _concat_np[:, runtime_prompt_head_len:runtime_prompt_head_len + vision_embed_size, :].copy()

    print(f"\nImage Process Complete. Time Cost: {time.time() - vision_start_time:.3f} Seconds")
    del _first_concat, _first_hidden, _first_input_ids, _bind_embed, _bind_vision, _concat_np, pixel_values


for task_index, (_, current_query) in enumerate(test_modes, 1):
    # Identify task type from query content
    if "HTML" in current_query or "OCR" in current_query:
        task_label = "OCR"
    elif "layout" in current_query.lower():
        task_label = "Layout"
    elif "table" in current_query.lower():
        task_label = "Table"
    else:
        task_label = "Text"
    decoding_method = "Beam Search" if USE_BEAM_SEARCH else "Greedy Search"
    print(f"\n{'═' * 60}")
    print(f"  Task {task_index}/{len(test_modes)}: {task_label} (image mode, {decoding_method})")
    print(f"{'═' * 60}")

    query = current_query
    tokens, mm_token_type_ids = build_prompt_and_mm_types(
        query,
        num_images=vision_batch_size,
    )

    num_prefill = tokens.shape[-1]

    # ══════════════════════════════════════════════════════════════════════════════
    # Shared OrtValue buffers (recreated for each test mode)
    # ══════════════════════════════════════════════════════════════════════════════
    input_ids        = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    ids_len          = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
    init_history_len = create_ort_with_data([0],           np.int64, device_type, DEVICE_ID)
    topK             = create_ort_with_data([TOP_K],       np.int64, device_type, DEVICE_ID)
    beam_size_ort    = create_ort_with_data([BEAM_SIZE],   np.int64, device_type, DEVICE_ID)

    attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1), hidden_dtype_Main, device_type, DEVICE_ID)
    _out_meta_Rotary_Decode = ort_session_Rotary_Decode._outputs_meta
    rotary_cos_buf = create_ort_with_shape(_out_meta_Rotary_Decode[0].shape, hidden_dtype_Main, device_type, DEVICE_ID)
    rotary_sin_buf = create_ort_with_shape(_out_meta_Rotary_Decode[1].shape, hidden_dtype_Main, device_type, DEVICE_ID)
    hidden_states_buf = create_ort_with_meta_shape(
        in_meta_Main[num_keys_values_Main],
        hidden_dtype_Main, device_type, DEVICE_ID,
        batch_size=BEAM_SIZE, seq_axis=1, seq_len=1,
    )
    save_id_buf = create_ort_with_shape((BEAM_SIZE, 0), np.int32, device_type, DEVICE_ID)
    prefill_logits_buf = create_ort_with_shape((1, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
    decode_logits_buf = create_ort_with_shape((BEAM_SIZE, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
    max_idx_buf = create_ort_with_shape((1, 1), np.int32, device_type, DEVICE_ID)

    # Reset KV state buffers for fresh generation
    past_keys_Main = create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_keys[0]], kv_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
    )
    past_values_Main = create_ort_with_meta_shape(
        in_meta_Main_by_name[in_name_Main_values[0]], kv_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
    )
    if in_name_Main_conv_states:
        past_conv_states_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_conv_states[0]], np.float16, kv_device, DEVICE_ID,
        )
    if in_name_Main_recurrent_states:
        past_recurrent_states_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_recurrent_states[0]], np.float16, kv_device, DEVICE_ID,
        )
    if k_scales_Main is not None:
        k_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_key_scales[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
        )
        v_scales_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_value_scales[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
        )
    if k_biases_Main is not None:
        k_biases_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_key_biases[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=-1,
        )
        v_biases_Main = create_ort_with_meta_shape(
            in_meta_Main_by_name[in_name_Main_value_biases[0]],
            scale_dtype_Main, kv_device, DEVICE_ID, seq_axis=3,
        )

    # Recreate IO bindings for a fresh run
    binding_Embed = ort_session_Embed.io_binding()
    binding_Main = ort_session_Main.io_binding()
    binding_Rotary_Prefill = ort_session_Rotary_Prefill.io_binding()
    binding_Rotary_Decode = ort_session_Rotary_Decode.io_binding()

    # ══════════════════════════════════════════════════════════════════════════════
    # Decode head sessions
    # ══════════════════════════════════════════════════════════════════════════════
    if USE_BEAM_SEARCH:

        ort_session_First_Beam     = create_session(onnx_model_First_Beam, **packed_settings)
        binding_First_Beam         = ort_session_First_Beam.io_binding()
        in_name_First_Beam         = get_in_names(ort_session_First_Beam)
        out_name_First_Beam        = get_out_names(ort_session_First_Beam)
        in_name_First_Beam_parts   = in_name_First_Beam[:num_keys_values_Main_plus_1]
        out_name_First_Beam_parts  = out_name_First_Beam[:num_keys_values_Main_plus_1]
        out_name_First_Beam_others = out_name_First_Beam[num_keys_values_Main_plus_1:]

        ort_session_Second_Beam     = create_session(onnx_model_Second_Beam, **packed_settings)
        binding_Second_Beam         = ort_session_Second_Beam.io_binding()
        in_name_Second_Beam         = get_in_names(ort_session_Second_Beam)
        out_name_Second_Beam        = get_out_names(ort_session_Second_Beam)
        in_name_Second_Beam_parts   = in_name_Second_Beam[:num_keys_values_Main_plus_1]
        out_name_Second_Beam_parts  = out_name_Second_Beam[:num_keys_values_Main_plus_1]
        out_name_Second_Beam_others = out_name_Second_Beam[num_keys_values_Main_plus_1:]

        beam_ids_buf = create_ort_with_shape((BEAM_SIZE, 1), np.int32, device_type, DEVICE_ID)
        beam_score_buf = create_ort_with_shape((BEAM_SIZE, 1), hidden_dtype_Main, device_type, DEVICE_ID)
        bind_ort_in_buf(
            binding_First_Beam,
            in_name_First_Beam[num_keys_values_Main_plus_1:num_keys_values_Main_plus_3],
            [save_id_buf, beam_size_ort],
        )
        bind_ort_in_buf(
            binding_Second_Beam,
            in_name_Second_Beam[num_keys_values_Main_plus_3:],
            [beam_size_ort, topK],
        )
    else:
        ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
        binding_Greedy     = ort_session_Greedy.io_binding()
        in_name_Greedy     = get_in_names(ort_session_Greedy)
        out_name_Greedy    = get_out_names(ort_session_Greedy)
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

        ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
        binding_Argmax     = ort_session_Argmax.io_binding()
        in_name_Argmax     = get_in_names(ort_session_Argmax)[0]
        out_name_Argmax    = get_out_names(ort_session_Argmax)[0]
        save_id_list       = []

    # ══════════════════════════════════════════════════════════════════════════════
    # Penalty session
    # ══════════════════════════════════════════════════════════════════════════════
    if USE_PENALTY:
        ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
        binding_Penalty     = ort_session_Penalty.io_binding()
        in_name_Penalty     = get_in_names(ort_session_Penalty)
        out_name_Penalty    = get_out_names(ort_session_Penalty)[0]
        penalty_dtype = (
            np.float16
            if "float16" in ort_session_Penalty._inputs_meta[2].type
            else np.float32
        )
        penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
        penalty_range_ort = create_ort_with_data([PENALTY_RANGE], np.int64, device_type, DEVICE_ID)
        bind_ort_in_buf(binding_Penalty, in_name_Penalty[2:], [penalty_value, penalty_range_ort])

    # ══════════════════════════════════════════════════════════════════════════════
    # Prefill phase
    # ══════════════════════════════════════════════════════════════════════════════
    is_prefill_step    = True
    prefill_start_time = time.time()
    prefill_elapsed    = 0.0
    decode_start_time  = prefill_start_time

    binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
    bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
    run(ort_session_Embed, binding_Embed)
    hidden_states = binding_Embed.get_outputs()[0]
    binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

    generate_limit = MAX_SEQ_LEN - num_prefill

    # Use cached vision embeddings - reconstruct concat without re-running vision encoder
    text_np = hidden_states.numpy()
    head = text_np[:, :runtime_prompt_head_len, :]
    tail = text_np[:, runtime_prompt_head_len:, :]
    concat_np = np.concatenate([head, cached_vision_embeds, tail], axis=1)
    concat_hidden_states = onnxruntime.OrtValue.ortvalue_from_numpy(
        concat_np.astype(hidden_dtype_Main), device_type, DEVICE_ID
    )

    # Image concat: adds vision tokens to prefill
    num_prefill += vision_embed_size
    ids_len = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
    generate_limit -= vision_embed_size

    # Image rotary prefill
    bind_ort_in_buf(binding_Rotary_Prefill, in_name_Rotary_Prefill, [ids_len, init_history_len])
    bind_ort_out(binding_Rotary_Prefill, out_name_Rotary_Prefill, _ort_device_type)
    run(ort_session_Rotary_Prefill, binding_Rotary_Prefill)
    rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Prefill.get_outputs()
    binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], concat_hidden_states)

    # Set decode rotary
    binding_Rotary_Decode.bind_ortvalue_input(in_name_Rotary_Decode, kv_seq_len)
    bind_ort_out_buf(binding_Rotary_Decode, out_name_Rotary_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len])
    ort_session_Rotary_Decode = ort_session_Rotary_Decode
    binding_Rotary_Decode = binding_Rotary_Decode

    bind_ort_in_buf(binding_Main, in_name_Main[idx_rotary_cos:], [rotary_cos, rotary_sin, attention_mask])

    for name in in_name_Main_keys:
        binding_Main.bind_ortvalue_input(name, past_keys_Main)
    for name in in_name_Main_values:
        binding_Main.bind_ortvalue_input(name, past_values_Main)
    if k_scales_Main is not None:
        for name in in_name_Main_key_scales:
            binding_Main.bind_ortvalue_input(name, k_scales_Main)
        for name in in_name_Main_value_scales:
            binding_Main.bind_ortvalue_input(name, v_scales_Main)
    if k_biases_Main is not None:
        for name in in_name_Main_key_biases:
            binding_Main.bind_ortvalue_input(name, k_biases_Main)
        for name in in_name_Main_value_biases:
            binding_Main.bind_ortvalue_input(name, v_biases_Main)
    if past_conv_states_Main is not None:
        for name in in_name_Main_conv_states:
            binding_Main.bind_ortvalue_input(name, past_conv_states_Main)
    if past_recurrent_states_Main is not None:
        for name in in_name_Main_recurrent_states:
            binding_Main.bind_ortvalue_input(name, past_recurrent_states_Main)

    bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
    binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

    if USE_PENALTY:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
        binding_Penalty.bind_ortvalue_output(out_name_Penalty, prefill_logits_buf)

    if USE_BEAM_SEARCH:
        binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_Main], prefill_logits_buf)
    elif USE_PENALTY:
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], prefill_logits_buf)
        binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
    else:
        binding_Argmax.bind_ortvalue_input(in_name_Argmax, prefill_logits_buf)
        binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)

    # ══════════════════════════════════════════════════════════════════════════════
    # Decode loop
    # ══════════════════════════════════════════════════════════════════════════════
    if USE_BEAM_SEARCH:
        print(f"\n  Prompt: {query}")
        print(f"  Decoding with beam search (beam_size={BEAM_SIZE}, top_k={TOP_K})...")
    else:
        print(f"\n  Prompt: {query}\n  Output: ", end="", flush=True)

    num_decode = 0
    save_id    = None

    while num_decode < generate_limit:

        # ── 1. Run Main Model ──────────────────────────────────────────────────────
        run(ort_session_Main, binding_Main)
        outputs_Main = binding_Main.get_outputs()

        # ── 2. Apply Repetition Penalty (if enabled) ─────────────────────
        if USE_PENALTY and num_decode >= PENALTY_RANGE:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
            run(ort_session_Penalty, binding_Penalty)

        # ── 3. Token Selection ─────────────────────────────────────────────────────
        if USE_BEAM_SEARCH:
            if is_prefill_step:
                bind_ort_in_buf(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
                bind_ort_out(binding_First_Beam, out_name_First_Beam_parts, _ort_device_type)
                bind_ort_out_buf(binding_First_Beam, out_name_First_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
                run(ort_session_First_Beam, binding_First_Beam)
                outputs_Beam = binding_First_Beam.get_outputs()
            else:
                bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
                bind_ort_out(binding_Second_Beam, out_name_Second_Beam_parts, _ort_device_type)
                if num_decode < 2:
                    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_2], beam_score_buf)
                bind_ort_out_buf(binding_Second_Beam, out_name_Second_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
                run(ort_session_Second_Beam, binding_Second_Beam)
                outputs_Beam = binding_Second_Beam.get_outputs()

            max_logits_idx = max_idx_buf.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN_SET:
                break

            save_id = outputs_Beam[num_keys_values_Main]
            bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Beam)
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_1], save_id)
        else:
            if USE_PENALTY:
                binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
                run(ort_session_Greedy, binding_Greedy)
                greedy_outputs = binding_Greedy.get_outputs()
                save_id = greedy_outputs[1]
            else:
                run(ort_session_Argmax, binding_Argmax)

            max_logits_idx = max_idx_buf.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN_SET:
                break

            if USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
            else:
                save_id_list.append(max_logits_idx)

            bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Main)
            print(tokenizer.decode(max_logits_idx), end="", flush=True)

        # ── 4. Re-bind Main KV outputs (fresh allocation each step) ──────
        bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

        # ── 5. Transition: prefill → decode (executes once) ──────────────
        if is_prefill_step:
            binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_Main], hidden_states_buf)
            bind_ort_in_buf(binding_Main, in_name_Main[idx_rotary_cos:], [rotary_cos_buf, rotary_sin_buf, attention_mask_buf])
            binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)
            binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

            if USE_PENALTY:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
                binding_Penalty.bind_ortvalue_output(out_name_Penalty, decode_logits_buf)

            if USE_BEAM_SEARCH:
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main], decode_logits_buf)
                binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
            elif USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
            else:
                binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

            is_prefill_step = False
            decode_start_time = time.time()
            prefill_elapsed = decode_start_time - prefill_start_time

        # ── 6. Prepare next step: Embed + Rotary ─────────────────────────────
        run(ort_session_Embed, binding_Embed)
        run(ort_session_Rotary_Decode, binding_Rotary_Decode)
        num_decode += 1

    # ══════════════════════════════════════════════════════════════════════════════
    # Results
    # ══════════════════════════════════════════════════════════════════════════════
    decode_end_time = time.time()

    if num_decode < 2:
        prefill_elapsed = 0.0
        decode_elapsed = 0.0
    else:
        decode_elapsed = decode_end_time - decode_start_time

    total_elapsed = decode_end_time - prefill_start_time

    prefill_tokens_per_second = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
    decode_tokens_per_second = num_decode / decode_elapsed if decode_elapsed > 0 else 0.0
    overall_tokens_per_second = (num_decode + 1) / total_elapsed if total_elapsed > 0 else 0.0

    if USE_PENALTY or USE_BEAM_SEARCH:
        result = (
            tokenizer.decode(save_id.numpy().flat[:num_decode], skip_special_tokens=True)
            if save_id is not None
            else ""
        )
    else:
        result = tokenizer.decode(save_id_list, skip_special_tokens=True)

    # Print generated output only for beam search (greedy already streams inline)
    if USE_BEAM_SEARCH:
        print(
            f"\n\n{chr(9472) * 60}\n"
            f"  Generated Output ({task_label})\n"
            f"{chr(9472) * 60}\n"
            f"{result}\n"
            f"{chr(9472) * 60}"
        )
    else:
        print()  # newline after streamed output

    print(
        f"\n{chr(9472) * 60}\n"
        f"  Performance Summary - Task {task_index}: {task_label}\n"
        f"{chr(9472) * 60}\n"
        f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
        f"  {chr(9472) * 52}\n"
        f"  {'Prefill':<12} {prefill_tokens_per_second:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
        f"  {'Decode':<12} {decode_tokens_per_second:>10.2f} t/s {num_decode:>8d} {decode_elapsed:>8.3f}s\n"
        f"  {chr(9472) * 52}\n"
        f"  {'Overall':<12} {overall_tokens_per_second:>10.2f} t/s {num_decode + 1:>8d} {total_elapsed:>8.3f}s\n"
        f"{chr(9472) * 60}\n"
    )
