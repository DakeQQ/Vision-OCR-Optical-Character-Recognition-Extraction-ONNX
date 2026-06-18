import os
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# CONFIG  -- edit me, then just hit Run.
# =============================================================================
path_lfm = r'/home/DakeQQ/Downloads/LFM2-350M-Extract'                                 # local LFM2-Extract model folder. Note: the LFM2 only accept pure text as input, not multimodal. URL: https://huggingface.co/LiquidAI/LFM2-350M-Extract / https://huggingface.co/LiquidAI/LFM2-1.2B-Extract
onnx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LFM_ONNX')        # output directory for the 9 ONNX graphs

onnx_model_Embed                = os.path.join(onnx_dir, 'LLM_Embed.onnx')             # token id   -> hidden state
onnx_model_Rotary_Mask_Prefill  = os.path.join(onnx_dir, 'Rotary_Prefill.onnx')        # prefill rotary + causal mask
onnx_model_Rotary_Mask_Decode   = os.path.join(onnx_dir, 'Rotary_Decode.onnx')         # decode  rotary
onnx_model_Main                 = os.path.join(onnx_dir, 'LLM_Main.onnx')              # decoder layers + LM head
onnx_model_Greedy               = os.path.join(onnx_dir, 'Greedy_Search.onnx')         # argmax + append
onnx_model_First_Beam           = os.path.join(onnx_dir, 'First_Beam_Search.onnx')     # 1 -> beam_size
onnx_model_Second_Beam          = os.path.join(onnx_dir, 'Second_Beam_Search.onnx')    # prune + re-expand
onnx_model_Penalty              = os.path.join(onnx_dir, 'Apply_Penalty.onnx')         # repetition penalty
onnx_model_Argmax               = os.path.join(onnx_dir, 'Argmax.onnx')                # bare argmax

# -- Export ------------------------------------------------------------------
DO_EXPORT              = True       # Export the ONNX graphs.
PREVENT_F16_OVERFLOW   = False      # Pre-scale activations by 0.01 before squaring in RMSNorm (for F16/Q paths).
USE_FLOAT16_KV         = True       # Store the F16/F32 conv-state cache in float16 (less bandwidth).

# -- KV cache quantization (attention layers only; conv state stays F16/F32) -------------------
KV_QUANT_DTYPE         = "F16"      # "F16" | "F32" | "Q8" | "Q8_CUDA" | "ROTARY_Q8" | "ROTARY_Q8_CUDA" | "ROTARY_Q4" | "ROTARY_Q4_CUDA"
KV_QUANT_GROUP_SIZE    = 32         # Group size for Q4 and (hadamard/shuffle-enabled) Q8 per-group quant. Must divide head_dim.
USE_HADAMARD           = True       # True = More Accuracy. Randomized Walsh-Hadamard mixing within each group before quant (Q4 & Q8).
HADAMARD_RANDOM_SEED   = 9527       # Seed for the deterministic Rademacher sign pattern used by the enhanced Hadamard transform.
USE_CLIP               = True       # Clip outliers to mean ± CLIP_SIGMA*std before quant (Q4 & Q8).
CLIP_SIGMA             = 3.0        # Clip threshold in standard deviations. 2.5-3.5 recommended. Only used when USE_CLIP=True.
USE_SHUFFLE            = True       # True = More Accuracy. Interleave channels across groups so high-variance channels spread out (Q4 & Q8).
USE_SYM                = False      # True = Less RAM Bandwidth. True: symmetric (absmax, no bias); False: asymmetric (min-max + bias).
USE_FLOAT16_SCALE_BIAS = True       # Store scale/bias as float16 in all quantized KV modes (Q8, Q4, and ROTARY variants).

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
OPSET                    = 18       # ONNX opset.
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
# KV-quant settings validation  (ported from Qwen_Export.py)
# =============================================================================
SUPPORTED_KV_QUANT_DTYPES = (
    "ROTARY_Q4", "ROTARY_Q4_CUDA", "Q8", "Q8_CUDA",
    "ROTARY_Q8", "ROTARY_Q8_CUDA", "F16", "F32",
)


def normalize_kv_quant_settings(head_dim):
    """Validate and normalize KV-quant settings once head_dim is known."""
    global KV_QUANT_GROUP_SIZE

    if KV_QUANT_DTYPE not in SUPPORTED_KV_QUANT_DTYPES:
        raise ValueError(f"Unsupported KV_QUANT_DTYPE: {KV_QUANT_DTYPE}")

    quantized_kv = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA", "ROTARY_Q4", "ROTARY_Q4_CUDA"}
    rotary_kv    = {"ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
    q8_kv        = {"Q8", "Q8_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA"}
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

        if KV_QUANT_DTYPE in q8_kv and KV_QUANT_GROUP_SIZE == head_dim and (USE_HADAMARD or USE_SHUFFLE):
            notes.append("[Info] USE_HADAMARD and USE_SHUFFLE do not change Q8 accuracy when grouping collapses to one full-head block.")
    elif any((USE_HADAMARD, USE_CLIP, USE_SHUFFLE, USE_SYM, USE_FLOAT16_SCALE_BIAS)):
        notes.append("[Info] Quant-only KV flags are ignored when KV_QUANT_DTYPE is F16 or F32.")

    return notes


# =============================================================================
# Decoding graphs -- ported verbatim from both references.
# =============================================================================
class GREEDY_SEARCH(torch.nn.Module):
    """Select the token with the highest logit (greedy decoding) and append it."""

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id        = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class ARGMAX(torch.nn.Module):
    """Bare argmax over the vocabulary dimension (greedy fast path / beam helper)."""

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class APPLY_PENALTY(torch.nn.Module):
    """Apply a repetition penalty over the most recent `penalty_range` tokens."""

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized      = logits.gather(1, target_indices) * penalty_value
        logits         = logits.scatter(1, target_indices, penalized)
        return logits


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers     = total_layers
        self.save_keys_values = [None] * self.total_layers
        self._ones_tuple      = {d: (1,) * d for d in range(8)}

    def forward(self, *all_inputs):
        logits    = all_inputs[-3]
        save_id   = all_inputs[-2]
        beam_size = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp

        for i in range(self.total_layers):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *self._ones_tuple[kv.dim() - 1])

        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id          = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx   = top_beam_indices[[0]]

        return (*self.save_keys_values, save_id, top_beam_prob.transpose(0, 1), top_beam_indices, max_logits_idx)


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand existing beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers     = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits        = all_inputs[-5]
        save_id       = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size     = all_inputs[-2]
        top_k         = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        top_k_prob   = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)

        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index       = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]

        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx   = top_beam_indices[[0]]
        save_id          = torch.cat([gathered_save_id, top_beam_indices], dim=-1)

        return (*self.save_keys_values, save_id, top_beam_prob.unsqueeze(-1), top_beam_indices, max_logits_idx)


# =============================================================================
# A : LFM_EMBED
# =============================================================================
class LFM_EMBED(torch.nn.Module):
    """Token-id -> hidden-state lookup (float32)."""

    def __init__(self, lfm):
        super().__init__()
        self.embed_tokens = lfm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


# =============================================================================
# R1 / R2 : Rotary + causal-mask graphs  (flip-RoPE f16 buffers + int8 mask)
# =============================================================================
def _build_rotary_cos_sin(lfm, max_seq_len):
    """Precompute flip-RoPE cos=[cos,cos] / sin=[-sin,sin] buffers (float16).

    Shape (1, max_seq_len, 1, 1, head_dim) so they broadcast with the combined
    QK tensor (B, S, 1, qk_heads, head_dim).
    """
    inv_freq          = lfm.model.rotary_emb.inv_freq.float()          # (head_dim/2,)
    attention_scaling = float(getattr(lfm.model.rotary_emb, "attention_scaling", 1.0))
    position_ids      = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)   # (max_seq, 1)
    freqs             = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)  # (1,max_seq,1,1,hd/2)
    cos = torch.cat([freqs.cos(),  freqs.cos()], dim=-1) * attention_scaling           # [cos,  cos ]
    sin = torch.cat([-freqs.sin(), freqs.sin()], dim=-1) * attention_scaling           # [-sin, sin ]
    return cos.half(), sin.half()


class ROTARY_MASK_PREFILL(torch.nn.Module):
    """Prefill: slice rotary cos/sin, build additive causal mask, return kv_seq_len."""

    def __init__(self, lfm, max_seq_len):
        super().__init__()
        cos, sin = _build_rotary_cos_sin(lfm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)
        # int8 additive causal mask: upper triangle -> -128 (≈ -inf after QK-norm scaling).
        self.register_buffer(
            "attention_mask",
            (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128,
            persistent=False,
        )

    def forward(self, ids_len, history_len):
        kv_seq_len     = ids_len + history_len
        rotary_cos     = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin     = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_MASK_DECODE(torch.nn.Module):
    """Decode: index the single new position, advance kv_seq_len (no mask needed)."""

    def __init__(self, lfm, max_seq_len):
        super().__init__()
        cos, sin = _build_rotary_cos_sin(lfm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# =============================================================================
# KVQuantizer  (unified Q8 / Q8_CUDA / ROTARY_Q8 / ROTARY_Q4 KV-cache quantizer)
#   Ported verbatim from Qwen_Export.py — architecture-agnostic; operates on the
#   already-laid-out K (B,KVH,1,D,S) and V (B,KVH,1,S,D) tensors.  Three precision
#   techniques can combine: rotary pairwise rotation, randomized Walsh-Hadamard
#   group mixing, and channel shuffle, plus sigma clipping and residual-bias
#   correction.  See the Qwen reference for the full derivation.
# =============================================================================
class KVQuantizer(torch.nn.Module):
    """Unified KV cache quantizer supporting Q8, Q8_CUDA, ROTARY_Q8, and ROTARY_Q4."""

    def __init__(self, head_dim, num_kv_heads, num_kv_groups, is_q4=False, is_rotary=False, is_q8_cuda=False, use_sym=False, use_hadamard=False, use_clip=False, clip_sigma=2.5, use_shuffle=False):
        super().__init__()
        self.is_rotary     = is_rotary
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

        # ── Quantization range ───────────────────────────────────────
        if use_sym:
            self.SIGNED_QMIN = -8 if is_q4 else -128
            self.SIGNED_QMAX = 7 if is_q4 else 127
            self.QMAX        = float(self.SIGNED_QMAX)
            self.ZERO_POINT  = 0.0
        else:
            self.SIGNED_QMIN = None
            self.SIGNED_QMAX = None
            self.QMAX        = 15.0 if is_q4 else 255.0
            self.ZERO_POINT  = 0.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.QMAX]).view(1, 1, 1, 1, -1))

        # ── Group parameters (ROTARY_Q4 always grouped; Q8/ROTARY_Q8 grouped when hadamard/shuffle enabled) ──
        self.is_grouped          = is_q4 or ((self.use_hadamard or self.use_shuffle) and KV_QUANT_GROUP_SIZE < head_dim)
        if not self.is_grouped and not is_q4:
            self.use_hadamard = False
            self.use_shuffle  = False
        self.kv_quant_group_size = KV_QUANT_GROUP_SIZE if self.is_grouped else 0
        self.kv_quant_num_groups = head_dim // KV_QUANT_GROUP_SIZE if self.is_grouped else 0

        # ── Q8_CUDA int32 packing constants ──────────────────────────
        if is_q8_cuda:
            for name, val in [("_256", 256), ("_128", 128), ("_65536", 65536), ("_16777216", 16777216)]:
                self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

        # ── Rotary transform buffers ─────────────────────────────────
        if is_rotary:
            sqrt2 = 2.0 ** 0.5
            inv_sqrt2 = 1.0 / sqrt2
            self.register_buffer("rot_cos", torch.tensor([inv_sqrt2]))

            fwd_sin = torch.cat([torch.full((head_dim // 2,), -inv_sqrt2), torch.full((head_dim // 2,),  inv_sqrt2)])
            self.register_buffer("rot_sin_k", fwd_sin.view(1, 1, 1, -1, 1))
            self.register_buffer("rot_sin_v", fwd_sin.view(1, 1, 1, 1, -1))

            c_vec = torch.zeros(head_dim)
            c_vec[:head_dim // 2] = sqrt2
            self.register_buffer("c_vec", c_vec.view(1, 1, 1, 1, -1))

        # ── Enhanced Hadamard transform buffers ───────────────────────
        if self.use_hadamard:
            self.hadamard_size = self._next_power_of_two(self.kv_quant_group_size)
            self.hadamard_pad = self.hadamard_size - self.kv_quant_group_size
            self.register_buffer("hadamard_inv_sqrt", torch.tensor([self.hadamard_size ** -0.5], dtype=torch.float32))

            sign_generator = torch.Generator()
            sign_generator.manual_seed(HADAMARD_RANDOM_SEED)
            hadamard_sign = torch.randint(0, 2, (self.kv_quant_group_size,), generator=sign_generator, dtype=torch.int64)
            hadamard_sign = hadamard_sign.float().mul_(2.0).sub_(1.0)
            self.register_buffer("hadamard_sign", hadamard_sign)

            # Pre-compute Hadamard butterfly level widths
            self._hadamard_levels = []
            w = self.hadamard_size
            while w > 1:
                h = w // 2
                self._hadamard_levels.append((w, h))
                w = h

        # ── Clip sigma buffer ─────────────────────────────────────────
        if self.use_clip:
            self.register_buffer("_clip_sigma_t", torch.tensor([clip_sigma]))

        # ── Channel shuffle buffers ──────────────────────────────────
        if self.use_shuffle:
            perm = torch.arange(head_dim).view(self.kv_quant_num_groups, self.kv_quant_group_size).T.contiguous().view(-1)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(head_dim)
            self.register_buffer("shuffle_idx", perm.int())
            self.register_buffer("unshuffle_idx", inv_perm.int())

    # ══════════════════════════════════════════════════════════════════
    # Enhanced Walsh-Hadamard helpers
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _next_power_of_two(n):
        value = 1
        while value < n:
            value *= 2
        return value

    def _apply_hadamard_last_dim(self, x, inverse=False):
        """Apply a deterministic randomized Walsh-Hadamard transform on the last dim."""
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

    # ══════════════════════════════════════════════════════════════════
    # Sigma-based clipping (applied per quantization block before quant)
    # ══════════════════════════════════════════════════════════════════
    def _clip_to_sigma(self, x, dim):
        """Clip values to mean ± clip_sigma*std per quantization block."""
        mean  = x.mean(dim=dim, keepdim=True)
        var   = (x - mean).square().mean(dim=dim, keepdim=True)
        std   = var.sqrt()
        bound = self._clip_sigma_t * std
        return x.clamp(mean - bound, mean + bound)

    # ══════════════════════════════════════════════════════════════════
    # Rotary flip helpers (view + flip + view)
    # ══════════════════════════════════════════════════════════════════
    def _flip_k(self, k, batch_size):
        """Swap halves along head_dim (dim 3). k: (B, KVH, 1, head_dim, S)"""
        return k.view(batch_size, self.num_kv_heads, 1, 2, self.head_dim_half, -1).flip(-3).view(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def _flip_v(self, v, batch_size):
        """Swap halves along head_dim (last dim). v: (B, KVH, 1, S, head_dim)"""
        return v.view(batch_size, self.num_kv_heads, 1, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def _flip_q(self, q, batch_size):
        """Swap halves along head_dim (last dim). q: (B, KVH, G, Qlen, head_dim)"""
        return q.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, 2, self.head_dim_half).flip(-2).view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    # ── Forward rotation (applied during quantization) ───────────────
    def rotate_k(self, k, batch_size):
        """Rotate key pairs along head_dim (dim 3). k: (B, KVH, 1, head_dim, S)"""
        return k * self.rot_cos + self._flip_k(k, batch_size) * self.rot_sin_k

    def rotate_v(self, v, batch_size):
        """Rotate value pairs along head_dim (dim -1). v: (B, KVH, 1, S, head_dim)"""
        return v * self.rot_cos + self._flip_v(v, batch_size) * self.rot_sin_v

    # ── Inverse rotation (fused into attention computation) ──────────
    def rotate_q(self, q, batch_size):
        """Forward-rotate query along head_dim (last dim) for fused key attention.
        By orthogonality: <Q, R^{-1}(K)> = <R(Q), K>, so we need R(Q).
        q: (B, KVH, G, Qlen, head_dim)"""
        return q * self.rot_cos + self._flip_q(q, batch_size) * self.rot_sin_v

    def inverse_rotate_v(self, v, batch_size):
        """Inverse-rotate dequantized V along head_dim (last dim). v: (B, KVH, 1, S, head_dim)"""
        return v * self.rot_cos - self._flip_v(v, batch_size) * self.rot_sin_v

    def inverse_rotate_k(self, k, batch_size):
        """Inverse-rotate dequantized K along head_dim (dim 3). k: (B, KVH, 1, head_dim, S)"""
        return k * self.rot_cos - self._flip_k(k, batch_size) * self.rot_sin_k

    def inverse_rotate_attn(self, x, batch_size):
        """Inverse-rotate attention output along head_dim (last dim).
        x: (B, KVH, G, Qlen, head_dim)"""
        return x * self.rot_cos - self._flip_q(x, batch_size) * self.rot_sin_v

    # ══════════════════════════════════════════════════════════════════
    # Enhanced Hadamard transform helpers (within quantization groups, Q4 and Q8)
    # ══════════════════════════════════════════════════════════════════
    def hadamard_k(self, k, batch_size):
        """Apply randomized Walsh-Hadamard mixing within key quantization groups."""
        k = k.reshape(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
        k = self._apply_hadamard_last_dim(k.transpose(-1, -2)).transpose(-1, -2)
        return k.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def hadamard_v(self, v, batch_size):
        """Apply randomized Walsh-Hadamard mixing within value quantization groups."""
        v = v.reshape(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        v = self._apply_hadamard_last_dim(v)
        return v.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    def hadamard_q(self, q_g):
        """Apply the forward randomized Walsh-Hadamard transform to grouped queries."""
        return self._apply_hadamard_last_dim(q_g)

    def inverse_hadamard_attn(self, x, batch_size):
        """Apply the inverse randomized Walsh-Hadamard transform to attention output."""
        x = x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
        x = self._apply_hadamard_last_dim(x, inverse=True)
        return x.view(batch_size, self.num_kv_heads, self.num_kv_groups, -1, self.head_dim)

    # ══════════════════════════════════════════════════════════════════
    # Block quantization
    # ══════════════════════════════════════════════════════════════════
    def _finalize_asymmetric_quant(self, x, x_packed, scale, block_min, dim):
        """Finalize asymmetric quantization with optional residual bias correction."""
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
        """Quantize to signed integers, then encode into the selected storage container."""
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
        """Per-block quantization. Symmetric (absmax) or asymmetric (min-max)."""
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
        """Per-group quantization (Q4 or Q8). Symmetric (absmax) or asymmetric (min-max)."""
        if self.use_sym:
            if dim == -2:  # keys: (B, KVH, 1, D, S)
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                absmax   = x.abs().amax(dim=-2, keepdim=True)
                scale    = absmax * self.inv_qmax
                x_packed = self._quantize_signed_to_storage(x, scale)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:          # values: (B, KVH, 1, S, D)
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
            if dim == -2:  # keys: (B, KVH, 1, D, S)
                x = x.view(batch_size, self.num_kv_heads, 1, self.kv_quant_num_groups, self.kv_quant_group_size, -1)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-2)
                block_min, block_max = torch.aminmax(x, dim=-2, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-2)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)
            else:          # values: (B, KVH, 1, S, D)
                x = x.view(batch_size, self.num_kv_heads, 1, -1, self.kv_quant_num_groups, self.kv_quant_group_size)
                if self.use_clip:
                    x = self._clip_to_sigma(x, dim=-1)
                block_min, block_max = torch.aminmax(x, dim=-1, keepdim=True)
                scale    = (block_max - block_min) * self.inv_qmax
                x_packed = torch.round((x - block_min) / scale)
                x_packed, scale, block_min = self._finalize_asymmetric_quant(x, x_packed, scale, block_min, dim=-1)
                x_packed = x_packed.reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)
            return x_packed, scale, block_min

    # ══════════════════════════════════════════════════════════════════
    # CUDA packing / unpacking (4 uint8 → 1 int32)
    # ══════════════════════════════════════════════════════════════════
    def pack_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        """Pack 4 uint8 values into a single int32 for CUDA-friendly storage."""
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216

    def unpack_cuda(self, x_i32, dim, batch_size, num_kv_heads, head_dim):
        """Unpack int32 back into 4 uint8 channels."""
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

    # ══════════════════════════════════════════════════════════════════
    # Q4 packing / unpacking (2 nibbles → 1 byte)
    # ══════════════════════════════════════════════════════════════════
    def pack_q4_k(self, x, batch_size):
        """Pack Q4 keys: (B,KVH,1, D, S) → (B,KVH,1, D//2, S)."""
        x = x.view(batch_size, self.num_kv_heads, 1, self.head_dim_half, 2, -1)
        low, high = torch.unbind(x, dim=-2)
        return (low + high * 16).to(torch.uint8)

    def pack_q4_v(self, x, batch_size):
        """Pack Q4 values: (B,KVH,1, S, D) → (B,KVH,1, S, D//2)."""
        x = x.view(batch_size, self.num_kv_heads, 1, -1, self.head_dim_half, 2)
        low, high = torch.unbind(x, dim=-1)
        return (low + high * 16).to(torch.uint8)

    def unpack_q4_k(self, x, batch_size):
        """Unpack Q4 keys: (B,KVH,1, D//2, S) → (B,KVH,1, D, S)."""
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-2).reshape(batch_size, self.num_kv_heads, 1, self.head_dim, -1)

    def unpack_q4_v(self, x, batch_size):
        """Unpack Q4 values: (B,KVH,1, S, D//2) → (B,KVH,1, S, D)."""
        low  = x % 16
        high = x // 16
        return torch.stack([low, high], dim=-1).reshape(batch_size, self.num_kv_heads, 1, -1, self.head_dim)

    # ══════════════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════════════
    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
        if self.is_rotary:
            # 1. Rotate before quantization
            keys   = self.rotate_k(keys, batch_size)
            values = self.rotate_v(values, batch_size)

        if self.use_shuffle:
            # 1b. Interleave channels across groups (spreads high-variance channels)
            keys   = keys.index_select(3, self.shuffle_idx)
            values = values.index_select(-1, self.shuffle_idx)

        if self.use_hadamard:
            # 2. Hadamard within quantization groups (works for Q4 and Q8)
            keys   = self.hadamard_k(keys, batch_size)
            values = self.hadamard_v(values, batch_size)

        if self.use_sym:
            # 3a. Symmetric quantize (no bias)
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
            # 3b. Asymmetric min-max quantize (with bias)
            k_packed, k_scale, k_bias = self._quantize_block(keys,   dim=-2, batch_size=batch_size)
            v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1, batch_size=batch_size)
            if self.is_q4:
                k_packed = self.pack_q4_k(k_packed, batch_size)
                v_packed = self.pack_q4_v(v_packed, batch_size)
            if self.is_q8_cuda:
                k_packed = self.pack_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
                v_packed = self.pack_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
            return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


# =============================================================================
# B : LFM_MAIN  (all 16 decoder layers + tied LM head)
# =============================================================================
class LFM_MAIN(torch.nn.Module):
    """
    Optimized LFM2 main transformer.  Skills applied (see header):
      - sum()-based RMSNorm with rsqrt, sqrt(N) compensation absorbed into the
        next Linear (no divide, no mean, no separate weight multiply)
      - fused QKV projection (one GEMM); operator_norm absorbed into QKV / conv.in_proj
      - QK-norm weights fused with a split attention scale (head_dim^0.25); no /sqrt(d)
      - flip()-based rotate_half + precomputed f16 [-sin, sin] rotary buffers (fed in)
      - GQA via broadcast reshape (no repeat_kv copies)
      - concat-friendly pre-transposed KV cache (F16/F32 or Q8 / Q8_CUDA / ROTARY_Q8 /
        ROTARY_Q8_CUDA / ROTARY_Q4 / ROTARY_Q4_CUDA via the shared KVQuantizer)
      - depthwise causal short-conv with a 2-column f16 conv-state cache
      - ffn_norm absorbed into w1/w3, then w1|w3 fused into one gate_up GEMM (SwiGLU)
      - embedding_norm absorbed into a CLONED (de-aliased) lm_head; additive causal mask
    Cache I/O order (attention layers only; conv state always appended last):
      keys(A), values(A),
      [sym quant: k_scale(A), v_scale(A)] | [asym quant: k_scale,k_bias,v_scale,v_bias (A each)],
      conv(C),  then outputs append logits.
    Non-cache inputs (at the end):  hidden_states, rotary_cos, rotary_sin, attention_mask.
    """

    def __init__(self, lfm, num_heads, num_key_value_heads, head_dim,
                 num_layers, num_conv_layers, num_attn_layers, hidden_size):
        super().__init__()
        self.lfm = lfm

        # -- geometry --
        self.head_dim             = head_dim
        self.head_dim_half        = head_dim // 2
        self.num_heads            = num_heads
        self.num_key_value_heads  = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.num_layers           = num_layers
        self.num_conv_layers      = num_conv_layers
        self.num_attn_layers      = num_attn_layers
        self.qk_heads             = num_heads + num_key_value_heads
        self.total_qkv_heads      = self.qk_heads + num_key_value_heads
        self.qkv_split_sizes      = [self.qk_heads, num_key_value_heads]
        self.qk_split_sizes       = [num_heads, num_key_value_heads]
        self.o_proj_in_features   = num_heads * head_dim

        # -- KV cache mode flags (attention layers only) --
        self.kv_f16             = (KV_QUANT_DTYPE == "F16")
        self.kv_q8              = (KV_QUANT_DTYPE == "Q8")
        self.kv_q8_cuda         = (KV_QUANT_DTYPE == "Q8_CUDA")
        self.kv_rotary_q8       = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA")
        self.kv_rotary_q4       = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        self.kv_rotary_q8_cuda  = (KV_QUANT_DTYPE == "ROTARY_Q8_CUDA")
        self.kv_rotary_q4_cuda  = (KV_QUANT_DTYPE == "ROTARY_Q4_CUDA")
        self.kv_rotary_cuda     = self.kv_rotary_q8_cuda or self.kv_rotary_q4_cuda
        self.kv_rotary          = self.kv_rotary_q8 or self.kv_rotary_q4
        self.kv_quantized       = self.kv_q8 or self.kv_q8_cuda
        self.kv_any_quantized   = self.kv_quantized or self.kv_rotary
        self.kv_sym             = USE_SYM and self.kv_any_quantized
        self.conv_f16           = USE_FLOAT16_KV

        # Q8 modes use per-group quant only when hadamard/shuffle enabled and group < head_dim.
        self.kv_q8_grouped      = (self.kv_quantized or self.kv_rotary_q8) and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim

        # head_dim used for int32 unpack / pack in rotary CUDA modes
        self.kv_unpack_head_dim = (head_dim // 2) if self.kv_rotary_q4_cuda else head_dim
        self.kv_pack_quarter    = (head_dim // 8) if self.kv_rotary_q4_cuda else (head_dim // 4)
        self.head_dim_quarter   = head_dim // 4

        # Flat-cache index bases over the A attention layers (conv layers live after, at conv_base+i).
        #   keys(A), values(A), [k_scale(A), (k_bias(A)), v_scale(A), (v_bias(A))], conv(C)
        A = num_attn_layers
        self.idx_value   = A
        self.idx_k_scale = 2 * A
        self.idx_k_bias  = 3 * A
        if self.kv_sym:
            # symmetric: no bias -> k_scale, v_scale only
            self.idx_v_scale = 3 * A
            self.idx_v_bias  = 4 * A          # unused
            num_kv_groups_in_cache = 4
        elif self.kv_any_quantized:
            self.idx_v_scale = 4 * A
            self.idx_v_bias  = 5 * A
            num_kv_groups_in_cache = 6
        else:
            self.idx_v_scale = 2 * A          # unused
            self.idx_v_bias  = 3 * A          # unused
            num_kv_groups_in_cache = 2
        self.conv_base = num_kv_groups_in_cache * A

        # -- sum()-RMSNorm epsilons (eps_sum = N * eps) --
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        hidden_eps = hidden_size * 1e-5
        qk_eps     = head_dim    * 1e-5
        if PREVENT_F16_OVERFLOW:
            qk_eps *= float(self.overflow_scale.square())
        self.register_buffer("hidden_rms_norm_eps", torch.tensor([hidden_eps], dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps",     torch.tensor([qk_eps],     dtype=torch.float32))

        # -- norm compensation factors --
        norm_factor       = float(hidden_size ** 0.5)
        combined_qk_scale = float(head_dim ** 0.25)            # d^0.5 (sum-norm) * d^-0.25 (split score scale)

        # -- KV quantizer (shared across all attention layers) --
        if self.kv_any_quantized:
            self.quantizer = KVQuantizer(
                head_dim=head_dim,
                num_kv_heads=num_key_value_heads,
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

        # -- per-layer output buffers --
        self.save_key   = [None] * num_attn_layers
        self.save_value = [None] * num_attn_layers
        self.save_conv  = [None] * num_conv_layers
        if self.kv_any_quantized:
            self.save_k_scale = [None] * num_attn_layers
            self.save_v_scale = [None] * num_attn_layers
            if not self.kv_sym:
                self.save_k_bias = [None] * num_attn_layers
                self.save_v_bias = [None] * num_attn_layers

        self._fuse_weights(norm_factor, combined_qk_scale)

    # ------------------------------------------------------------------ fusion
    def _fuse_weights(self, norm_factor, combined_qk_scale):
        with torch.no_grad():
            for layer in self.lfm.model.layers:
                if layer.is_attention_layer:
                    self._fuse_attention_layer(layer, norm_factor, combined_qk_scale)
                else:
                    self._fuse_conv_layer(layer, norm_factor)
                self._fuse_gate_up(layer, norm_factor)

            # embedding_norm -> CLONED lm_head (tie_embedding=True aliases head & embed;
            # mutating in place would corrupt the embed graph).
            head_weight = self.lfm.model.embed_tokens.weight.detach().clone()
            head_weight.mul_(self.lfm.model.embedding_norm.weight.unsqueeze(0) * norm_factor)
            lm_head = torch.nn.Linear(head_weight.shape[1], head_weight.shape[0], bias=False)
            lm_head.weight.copy_(head_weight)
            self.lm_head = lm_head

    def _fuse_attention_layer(self, layer, norm_factor, combined_qk_scale):
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        qkv = torch.nn.Linear(int(q_proj.in_features), out_features, bias=False)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
        del attn.q_proj, attn.k_proj, attn.v_proj

        # operator_norm * sqrt(hidden) -> QKV
        qkv.weight.mul_(layer.operator_norm.weight.unsqueeze(0) * norm_factor)
        attn.qkv = qkv

        # QK-norm weights * head_dim^0.25, repeated and concatenated into one buffer.
        attn.q_layernorm.weight.mul_(combined_qk_scale)
        attn.k_layernorm.weight.mul_(combined_qk_scale)
        q_norm_rep = attn.q_layernorm.weight.repeat(self.num_heads)
        k_norm_rep = attn.k_layernorm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat([q_norm_rep, k_norm_rep], dim=0).view(1, 1, 1, self.qk_heads, self.head_dim),
            requires_grad=False,
        )
        del attn.q_layernorm, attn.k_layernorm

    def _fuse_conv_layer(self, layer, norm_factor):
        # operator_norm * sqrt(hidden) -> conv.in_proj
        layer.conv.in_proj.weight.mul_(layer.operator_norm.weight.unsqueeze(0) * norm_factor)

    def _fuse_gate_up(self, layer, norm_factor):
        # ffn_norm * sqrt(hidden) -> w1 (gate) & w3 (up); then fuse w1|w3 into one gate_up GEMM.
        ffn_w = layer.ffn_norm.weight.unsqueeze(0) * norm_factor
        w1, w3 = layer.feed_forward.w1, layer.feed_forward.w3
        gate_up = torch.nn.Linear(w1.in_features, w1.out_features + w3.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([w1.weight * ffn_w, w3.weight * ffn_w], dim=0))
        layer.feed_forward.gate_up = gate_up
        layer.feed_forward.mlp_split = [w1.out_features, w3.out_features]
        del layer.feed_forward.w1, layer.feed_forward.w3

    # ------------------------------------------------------------- primitives
    def _rms_norm(self, x, eps):
        """sum()-based RMSNorm: x * rsqrt(sum(x^2) + eps_sum).  No divide, no mean."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)

    def _rotate_half_qk(self, x, batch_size):
        """flip()-based rotate_half for the combined QK tensor (B, S, 1, qk_heads, D)."""
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    # ----------------------------------------------------------------- forward
    def forward(self, *all_inputs):
        hidden_states  = all_inputs[-4]
        rotary_cos     = all_inputs[-3]
        rotary_sin     = all_inputs[-2]
        attention_mask = all_inputs[-1]
        ids_len        = hidden_states.shape[1]
        batch_size     = hidden_states.shape[0]

        kv_count   = 0
        conv_count = 0
        for layer in self.lfm.model.layers:
            if layer.is_attention_layer:
                hn = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

                # fused QKV -> (B, S, 1, total_qkv_heads, D) -> split QK / V
                qkv = layer.self_attn.qkv(hn).reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
                qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

                # QK-norm (weight + scale absorbed) then flip-RoPE
                qk = self._rms_norm(qk, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight
                qk = qk * rotary_cos + self._rotate_half_qk(qk, batch_size) * rotary_sin
                q, k = torch.split(qk, self.qk_split_sizes, dim=-2)

                # GQA: Q -> (B, KVH, G, S, D)
                q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                q = q.permute(0, 2, 3, 1, 4)
                # cache layout: K (B,KVH,1,D,S)  V (B,KVH,1,S,D)
                k = k.permute(0, 3, 2, 4, 1)
                v = v.transpose(1, 3)

                kc = kv_count
                if self.kv_rotary_q4:
                    # ── ROTARY_Q4 / ROTARY_Q4_CUDA ───────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k   = torch.cat([all_inputs[kc],                    packed_k], dim=-1)
                        v   = torch.cat([all_inputs[kc + self.idx_value],   packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[kc + self.idx_k_scale], scale_k],  dim=-1)
                        v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v],  dim=-3)
                        self.save_key[kc]     = k
                        self.save_value[kc]   = v
                        self.save_k_scale[kc] = k_s
                        self.save_v_scale[kc] = v_s
                        if USE_FLOAT16_SCALE_BIAS:
                            k_s, v_s = k_s.float(), v_s.float()
                        if self.kv_rotary_q4_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_k(k, batch_size)).float()
                        q_rot      = self.quantizer.rotate_q(q, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g    = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g      = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)
                        v_unpacked = self.quantizer._decode_signed_q4_storage(self.quantizer.unpack_q4_v(v, batch_size)).float()
                        v_q_g      = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k   = torch.cat([all_inputs[kc],                    packed_k], dim=-1)
                        v   = torch.cat([all_inputs[kc + self.idx_value],   packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[kc + self.idx_k_scale], scale_k],  dim=-1)
                        k_b = torch.cat([all_inputs[kc + self.idx_k_bias],  bias_k],   dim=-1)
                        v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v],  dim=-3)
                        v_b = torch.cat([all_inputs[kc + self.idx_v_bias],  bias_v],   dim=-3)
                        self.save_key[kc]     = k
                        self.save_value[kc]   = v
                        self.save_k_scale[kc] = k_s
                        self.save_k_bias[kc]  = k_b
                        self.save_v_scale[kc] = v_s
                        self.save_v_bias[kc]  = v_b
                        if USE_FLOAT16_SCALE_BIAS:
                            k_s, k_b, v_s, v_b = k_s.float(), k_b.float(), v_s.float(), v_b.float()
                        if self.kv_rotary_q4_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_unpacked = self.quantizer.unpack_q4_k(k, batch_size).float()
                        q_rot      = self.quantizer.rotate_q(q, batch_size)
                        if self.quantizer.use_shuffle:
                            q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                        q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        q_rot_g    = q_rot_g.transpose(-2, -3)
                        if self.quantizer.use_hadamard:
                            q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                        k_q_g      = k_unpacked.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                        attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                        q_sum_g    = q_rot_g.sum(dim=-1, keepdim=True)
                        attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                        attn       = torch.softmax(attn, dim=-1)
                        v_unpacked = self.quantizer.unpack_q4_v(v, batch_size).float()
                        v_q_g      = v_unpacked.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                        v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                        attn       = torch.matmul(attn, v_dequant)
                        if self.quantizer.use_hadamard:
                            attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                        if self.quantizer.use_shuffle:
                            attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)

                elif self.kv_rotary:
                    # ── ROTARY_Q8 / ROTARY_Q8_CUDA ───────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k   = torch.cat([all_inputs[kc],                    packed_k], dim=-1)
                        v   = torch.cat([all_inputs[kc + self.idx_value],   packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[kc + self.idx_k_scale], scale_k],  dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v], dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v], dim=-2)
                        self.save_key[kc]     = k
                        self.save_value[kc]   = v
                        self.save_k_scale[kc] = k_s
                        self.save_v_scale[kc] = v_s
                        if USE_FLOAT16_SCALE_BIAS:
                            k_s, v_s = k_s.float(), v_s.float()
                        if self.kv_rotary_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                        v_signed = self.quantizer._decode_signed_q8_storage(v).float()
                        if self.kv_q8_grouped:
                            q_rot      = self.quantizer.rotate_q(q, batch_size)
                            if self.quantizer.use_shuffle:
                                q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                            q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_rot_g    = q_rot_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                            k_q_g      = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                            attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                            attn       = torch.softmax(attn, dim=-1)
                            v_q_g      = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn       = torch.matmul(attn, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                            if self.quantizer.use_shuffle:
                                attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                            attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                        else:
                            q_rot    = self.quantizer.rotate_q(q, batch_size)
                            attn_raw = torch.matmul(q_rot, k_signed)
                            attn     = attn_raw * k_s + attention_mask
                            attn     = torch.softmax(attn, dim=-1)
                            v_scaled = v_signed * v_s
                            attn     = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.kv_pack_quarter)
                        k   = torch.cat([all_inputs[kc],                    packed_k], dim=-1)
                        v   = torch.cat([all_inputs[kc + self.idx_value],   packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[kc + self.idx_k_scale], scale_k],  dim=-1)
                        k_b = torch.cat([all_inputs[kc + self.idx_k_bias],  bias_k],   dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v], dim=-3)
                            v_b = torch.cat([all_inputs[kc + self.idx_v_bias],  bias_v],  dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v], dim=-2)
                            v_b = torch.cat([all_inputs[kc + self.idx_v_bias],  bias_v],  dim=-2)
                        self.save_key[kc]     = k
                        self.save_value[kc]   = v
                        self.save_k_scale[kc] = k_s
                        self.save_k_bias[kc]  = k_b
                        self.save_v_scale[kc] = v_s
                        self.save_v_bias[kc]  = v_b
                        if USE_FLOAT16_SCALE_BIAS:
                            k_s, k_b, v_s, v_b = k_s.float(), k_b.float(), v_s.float(), v_b.float()
                        if self.kv_rotary_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.kv_unpack_head_dim)
                        if self.kv_q8_grouped:
                            q_rot      = self.quantizer.rotate_q(q, batch_size)
                            if self.quantizer.use_shuffle:
                                q_rot = q_rot.index_select(-1, self.quantizer.shuffle_idx)
                            q_rot_g    = q_rot.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_rot_g    = q_rot_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_rot_g = self.quantizer.hadamard_q(q_rot_g)
                            k_q_g      = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_rot_g, k_q_g)
                            q_sum_g    = q_rot_g.sum(dim=-1, keepdim=True)
                            attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                            attn       = torch.softmax(attn, dim=-1)
                            v_q_g      = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn       = torch.matmul(attn, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                            if self.quantizer.use_shuffle:
                                attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                            attn       = self.quantizer.inverse_rotate_attn(attn, batch_size)
                        else:
                            q_rot         = self.quantizer.rotate_q(q, batch_size)
                            attn_raw      = torch.matmul(q_rot, k.float())
                            q_bias_factor = (q * self.quantizer.c_vec).sum(dim=-1, keepdim=True)
                            attn_bias     = q_bias_factor * k_b + attention_mask
                            attn          = torch.addcmul(attn_bias, attn_raw, k_s)
                            attn          = torch.softmax(attn, dim=-1)
                            v_scaled  = v.float() * v_s
                            bias_term = torch.matmul(attn, v_b) * self.quantizer.c_vec
                            attn      = self.quantizer.inverse_rotate_attn(torch.matmul(attn, v_scaled), batch_size) + bias_term

                elif self.kv_quantized:
                    # ── Q8 / Q8_CUDA ─────────────────────────────────────
                    if self.kv_sym:
                        packed_k, scale_k, packed_v, scale_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                        k   = torch.cat([all_inputs[kc],                    packed_k], dim=-1)
                        v   = torch.cat([all_inputs[kc + self.idx_value],   packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[kc + self.idx_k_scale], scale_k],  dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v], dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v], dim=-2)
                        self.save_key[kc]     = k
                        self.save_value[kc]   = v
                        self.save_k_scale[kc] = k_s
                        self.save_v_scale[kc] = v_s
                        if USE_FLOAT16_SCALE_BIAS:
                            k_s, v_s = k_s.float(), v_s.float()
                        if self.kv_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                        k_signed = self.quantizer._decode_signed_q8_storage(k).float()
                        v_signed = self.quantizer._decode_signed_q8_storage(v).float()
                        if self.kv_q8_grouped:
                            q_in = q
                            if self.quantizer.use_shuffle:
                                q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                            q_g    = q_in.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_g    = q_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_g = self.quantizer.hadamard_q(q_g)
                            k_q_g      = k_signed.view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_g, k_q_g)
                            attn       = (attn_raw_g * k_s).sum(dim=-3) + attention_mask
                            attn       = torch.softmax(attn, dim=-1)
                            v_q_g      = v_signed.view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant  = (v_q_g * v_s).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn       = torch.matmul(attn, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                            if self.quantizer.use_shuffle:
                                attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        else:
                            attn_raw = torch.matmul(q, k_signed)
                            attn     = attn_raw * k_s + attention_mask
                            attn     = torch.softmax(attn, dim=-1)
                            v_scaled = v_signed * v_s
                            attn     = torch.matmul(attn, v_scaled)
                    else:
                        packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                        k   = torch.cat([all_inputs[kc],                    packed_k], dim=-1)
                        v   = torch.cat([all_inputs[kc + self.idx_value],   packed_v], dim=-2)
                        k_s = torch.cat([all_inputs[kc + self.idx_k_scale], scale_k],  dim=-1)
                        k_b = torch.cat([all_inputs[kc + self.idx_k_bias],  bias_k],   dim=-1)
                        if self.kv_q8_grouped:
                            v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v], dim=-3)
                            v_b = torch.cat([all_inputs[kc + self.idx_v_bias],  bias_v],  dim=-3)
                        else:
                            v_s = torch.cat([all_inputs[kc + self.idx_v_scale], scale_v], dim=-2)
                            v_b = torch.cat([all_inputs[kc + self.idx_v_bias],  bias_v],  dim=-2)
                        self.save_key[kc]     = k
                        self.save_value[kc]   = v
                        self.save_k_scale[kc] = k_s
                        self.save_k_bias[kc]  = k_b
                        self.save_v_scale[kc] = v_s
                        self.save_v_bias[kc]  = v_b
                        if USE_FLOAT16_SCALE_BIAS:
                            k_s, k_b, v_s, v_b = k_s.float(), k_b.float(), v_s.float(), v_b.float()
                        if self.kv_q8_cuda:
                            k = self.quantizer.unpack_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                            v = self.quantizer.unpack_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                        if self.kv_q8_grouped:
                            q_in = q
                            if self.quantizer.use_shuffle:
                                q_in = q_in.index_select(-1, self.quantizer.shuffle_idx)
                            q_g    = q_in.view(batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            q_g    = q_g.transpose(-2, -3)
                            if self.quantizer.use_hadamard:
                                q_g = self.quantizer.hadamard_q(q_g)
                            k_q_g      = k.float().view(batch_size, self.num_key_value_heads, 1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size, -1)
                            attn_raw_g = torch.matmul(q_g, k_q_g)
                            q_sum_g    = q_g.sum(dim=-1, keepdim=True)
                            attn       = (attn_raw_g * k_s + q_sum_g * k_b).sum(dim=-3) + attention_mask
                            attn       = torch.softmax(attn, dim=-1)
                            v_q_g      = v.float().view(batch_size, self.num_key_value_heads, 1, -1, self.quantizer.kv_quant_num_groups, self.quantizer.kv_quant_group_size)
                            v_dequant  = (v_q_g * v_s + v_b).reshape(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)
                            attn       = torch.matmul(attn, v_dequant)
                            if self.quantizer.use_hadamard:
                                attn = self.quantizer.inverse_hadamard_attn(attn, batch_size)
                            if self.quantizer.use_shuffle:
                                attn = attn.index_select(-1, self.quantizer.unshuffle_idx)
                        else:
                            attn_raw  = torch.matmul(q, k.float())
                            attn_bias = q.sum(dim=-1, keepdim=True) * k_b + attention_mask
                            attn      = torch.addcmul(attn_bias, attn_raw, k_s)
                            attn      = torch.softmax(attn, dim=-1)
                            v_dequant = torch.addcmul(v_b, v.float(), v_s)
                            attn      = torch.matmul(attn, v_dequant)

                else:
                    # ── F16 / F32 (no quantization) ──────────────────────
                    if self.kv_f16:
                        k, v = k.half(), v.half()
                    k = torch.cat([all_inputs[kc],                k], dim=-1)
                    v = torch.cat([all_inputs[kc + self.idx_value], v], dim=-2)
                    self.save_key[kc]   = k
                    self.save_value[kc] = v
                    if self.kv_f16:
                        k, v = k.float(), v.float()
                    attn = torch.matmul(q, k) + attention_mask
                    attn = torch.softmax(attn, dim=-1)
                    attn = torch.matmul(attn, v)

                kv_count += 1
                attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.o_proj_in_features)
                op_out = layer.self_attn.out_proj(attn)

            else:
                # ---- short conv (depthwise causal, gated) ----
                hn = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)
                BCx = layer.conv.in_proj(hn).transpose(-1, -2)          # (B, 3D, S)
                B_val, Cg, x = BCx.chunk(3, dim=-2)
                Bx = B_val * x                                          # (B, D, S)
                conv_state = torch.cat([all_inputs[conv_count + self.conv_base].float(), Bx], dim=-1)
                if conv_count == 0:
                    len_conv_state = conv_state.shape[-1]
                self.save_conv[conv_count] = conv_state[..., -2:].half() if self.conv_f16 else conv_state[..., -2:]
                conv_count += 1
                conv_out = layer.conv.conv(conv_state)[..., :len_conv_state]
                conv_out = conv_out[..., -ids_len:]
                op_out = layer.conv.out_proj((Cg * conv_out).transpose(-1, -2).contiguous())

            # residual #1 + SwiGLU (fused gate_up) + residual #2
            hidden_states = hidden_states + op_out
            ffn_in   = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)
            gate_up  = layer.feed_forward.gate_up(ffn_in)
            gate, up = torch.split(gate_up, layer.feed_forward.mlp_split, dim=-1)
            hidden_states = hidden_states + layer.feed_forward.w2(torch.nn.functional.silu(gate) * up)

        # final norm (embedding_norm absorbed into the cloned lm_head) + tied LM head
        last   = self._rms_norm(hidden_states[:, -1], self.hidden_rms_norm_eps)
        logits = self.lm_head(last)

        if self.kv_sym:
            return (*self.save_key, *self.save_value, *self.save_k_scale, *self.save_v_scale,
                    *self.save_conv, logits)
        if self.kv_any_quantized:
            return (*self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias,
                    *self.save_v_scale, *self.save_v_bias, *self.save_conv, logits)
        return (*self.save_key, *self.save_value, *self.save_conv, logits)


# =============================================================================
# EXPORT
# =============================================================================
if DO_EXPORT:
    os.makedirs(onnx_dir, exist_ok=True)
    print('Export start ...')
    with torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(
            path_lfm, dtype=torch.float32, device_map='cpu', low_cpu_mem_usage=True).eval()

        # -- derive dims from config + VERIFIED weight shapes --
        num_layers          = model.config.num_hidden_layers
        num_attn_layers     = sum(bool(l.is_attention_layer) for l in model.model.layers)
        num_conv_layers     = num_layers - num_attn_layers
        num_heads           = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads
        head_dim            = model.model.layers[2].self_attn.head_dim
        hidden_size         = model.model.embed_tokens.embedding_dim
        vocab_size          = model.config.vocab_size
        ffn_dim             = None
        for l in model.model.layers:
            if not l.is_attention_layer:
                pass
        ffn_dim = model.model.layers[0].feed_forward.w1.weight.shape[0]
        embed_ptr_before    = model.model.embed_tokens.weight.data_ptr()

        print(f"  layers={num_layers}  attn={num_attn_layers}  conv={num_conv_layers}  "
              f"heads={num_heads}/{num_key_value_heads}kv  head_dim={head_dim}")
        print(f"  hidden={hidden_size}  FFN(real)={ffn_dim}  vocab={vocab_size}")
        assert ffn_dim == 4608, f"expected FFN dim 4608, got {ffn_dim}"
        assert num_attn_layers == 6 and num_conv_layers == 10, "unexpected layer split"

        for note in normalize_kv_quant_settings(head_dim):
            print(f"  {note}")

        # ---- KV-cache dtype / storage geometry per quant mode (attention layers only) ----
        scale_dt = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32
        conv_dt  = torch.float16 if USE_FLOAT16_KV else torch.float32
        A, Cc    = num_attn_layers, num_conv_layers

        _is_rotary    = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA", "ROTARY_Q8", "ROTARY_Q8_CUDA")
        _is_rotary_q4 = KV_QUANT_DTYPE in ("ROTARY_Q4", "ROTARY_Q4_CUDA")
        _is_q8        = KV_QUANT_DTYPE in ("Q8", "Q8_CUDA")
        _kv_sym       = USE_SYM and (_is_rotary or _is_q8)
        _q8_grouped   = _is_q8 and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        _rq8_grouped  = KV_QUANT_DTYPE in ("ROTARY_Q8", "ROTARY_Q8_CUDA") and (USE_HADAMARD or USE_SHUFFLE) and KV_QUANT_GROUP_SIZE < head_dim
        _grouped_6d   = _is_rotary_q4 or _q8_grouped or _rq8_grouped
        _quantized    = _is_rotary or _is_q8

        # storage dtype for packed K/V
        if KV_QUANT_DTYPE == "F16":
            cache_dt = torch.float16
        elif KV_QUANT_DTYPE == "F32":
            cache_dt = torch.float32
        elif KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA", "ROTARY_Q4_CUDA"):
            cache_dt = torch.int32
        elif _kv_sym and not _is_rotary_q4:
            cache_dt = torch.int8           # symmetric Q8 / ROTARY_Q8 (non-CUDA) -> true int8
        else:
            cache_dt = torch.uint8          # asymmetric, or any Q4 nibble-packed

        # packed K/V head_dim along the storage axis
        if KV_QUANT_DTYPE in ("Q8_CUDA", "ROTARY_Q8_CUDA"):
            k_head = v_head = head_dim // 4
        elif KV_QUANT_DTYPE == "ROTARY_Q4":
            k_head = v_head = head_dim // 2
        elif KV_QUANT_DTYPE == "ROTARY_Q4_CUDA":
            k_head = v_head = head_dim // 8
        else:
            k_head = v_head = head_dim

        kf = torch.zeros((BEAM_SIZE, num_key_value_heads, 1, k_head, 0), dtype=cache_dt)
        vf = torch.zeros((BEAM_SIZE, num_key_value_heads, 1, 0, v_head), dtype=cache_dt)
        cf = torch.zeros((BEAM_SIZE, hidden_size, 0), dtype=conv_dt)

        # scale/bias dummy tensors + their seq (history) concat axis used in dynamic_axes.
        if _quantized:
            num_groups = head_dim // KV_QUANT_GROUP_SIZE if _grouped_6d else 1
            if _grouped_6d:
                ks_dummy, ks_seq = torch.ones((BEAM_SIZE, num_key_value_heads, 1, num_groups, 1, 0), dtype=scale_dt), 5
                vs_dummy, vs_seq = torch.ones((BEAM_SIZE, num_key_value_heads, 1, 0, num_groups, 1), dtype=scale_dt), 3
            else:
                ks_dummy, ks_seq = torch.ones((BEAM_SIZE, num_key_value_heads, 1, 1, 0), dtype=scale_dt), 4
                vs_dummy, vs_seq = torch.ones((BEAM_SIZE, num_key_value_heads, 1, 0, 1), dtype=scale_dt), 3
        else:
            ks_dummy = vs_dummy = None
            ks_seq = vs_seq = 0

        def build_cache_io(batch_axis='batch', kf_=kf, vf_=vf, cf_=cf, ks_=None, vs_=None):
            """Return (tensors, in_names, out_names, dyn_axes) for the heterogeneous cache.

            Layout: keys(A), values(A),
                    [sym: k_scale,v_scale] | [asym: k_scale,k_bias,v_scale,v_bias] (A each),
                    conv(C).  Matches LFM_MAIN's index scheme and return order.
            """
            ks_t = ks_ if ks_ is not None else ks_dummy
            vs_t = vs_ if vs_ is not None else vs_dummy
            tensors, in_names, out_names, axes = [], [], [], {}

            def add(group, tensor, seq_dim, count):
                for i in range(count):
                    inn, outn = f'in_{group}_{i}', f'out_{group}_{i}'
                    tensors.append(tensor); in_names.append(inn); out_names.append(outn)
                    axes[inn]  = {0: batch_axis, seq_dim: 'history_len'}
                    axes[outn] = {0: batch_axis, seq_dim: 'history_len_plus_ids_len'}

            add('key',   kf_, 4, A)
            add('value', vf_, 3, A)
            if _quantized:
                if _kv_sym:
                    add('key_scale',   ks_t, ks_seq, A)
                    add('value_scale', vs_t, vs_seq, A)
                else:
                    add('key_scale',   ks_t, ks_seq, A)
                    add('key_bias',    ks_t, ks_seq, A)
                    add('value_scale', vs_t, vs_seq, A)
                    add('value_bias',  vs_t, vs_seq, A)
            add('conv', cf_, 2, Cc)
            return tensors, in_names, out_names, axes

        # ---- Embed ----
        input_ids = torch.ones((1, 10), dtype=torch.int32)
        torch.onnx.export(
            LFM_EMBED(model), (input_ids,), onnx_model_Embed,
            input_names=['input_ids'], output_names=['hidden_states'],
            dynamic_axes={'input_ids': {0: 'batch', 1: 'ids_len'}, 'hidden_states': {0: 'batch', 1: 'ids_len'}},
            do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print('  [A] LFM_Embed exported')
        del input_ids

        # ---- Rotary + Mask (prefill) ----
        ids_len_t     = torch.tensor([10], dtype=torch.int64)
        history_len_t = torch.tensor([0],  dtype=torch.int64)
        torch.onnx.export(
            ROTARY_MASK_PREFILL(model, MAX_SEQ_LEN), (ids_len_t, history_len_t), onnx_model_Rotary_Mask_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={'rotary_cos': {1: 'ids_len'}, 'rotary_sin': {1: 'ids_len'},
                          'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}},
            do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print('  [R1] Rotary_Mask_Prefill exported')

        # ---- Rotary (decode) ----
        kv_seq_len_t = ids_len_t + history_len_t
        torch.onnx.export(
            ROTARY_MASK_DECODE(model, MAX_SEQ_LEN), (kv_seq_len_t,), onnx_model_Rotary_Mask_Decode,
            input_names=['kv_seq_len'], output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len_next'],
            dynamic_axes=None, do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print('  [R2] Rotary_Mask_Decode exported')

        # ---- Main ----
        cache_tensors, cache_in_names, cache_out_names, cache_axes = build_cache_io()
        hidden_states  = torch.ones((BEAM_SIZE, 10, hidden_size), dtype=torch.float32)
        rotary_cos     = torch.zeros((1, 10, 1, 1, head_dim), dtype=torch.float32)
        rotary_sin     = rotary_cos
        attention_mask = torch.zeros((1, 1, 1, 10, 10), dtype=torch.float32)

        all_inputs   = cache_tensors + [hidden_states, rotary_cos, rotary_sin, attention_mask]
        input_names  = cache_in_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
        output_names = cache_out_names + ['logits']
        dynamic_axes = {**cache_axes,
                        'hidden_states': {0: 'batch', 1: 'ids_len'},
                        'rotary_cos': {1: 'ids_len'}, 'rotary_sin': {1: 'ids_len'},
                        'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'},
                        'logits': {0: 'batch'}}

        main = LFM_MAIN(model, num_heads, num_key_value_heads, head_dim,
                        num_layers, num_conv_layers, num_attn_layers, hidden_size)

        # de-alias assertion: cloned head must NOT share storage with embed table.
        head_ptr  = main.lm_head.weight.data_ptr()
        embed_ptr = model.model.embed_tokens.weight.data_ptr()
        assert head_ptr != embed_ptr, "lm_head still aliases embed_tokens!"
        assert embed_ptr == embed_ptr_before, "embed_tokens storage moved!"
        print(f"  [B] de-alias OK (embed=0x{embed_ptr:x}  head=0x{head_ptr:x})")

        torch.onnx.export(
            main, tuple(all_inputs), onnx_model_Main,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
            do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print(f'  [B] LFM_Main exported ({len(cache_in_names)} cache tensors)')
        del hidden_states, rotary_cos, attention_mask, all_inputs, main, model
        gc.collect()

        # ---- Greedy / Argmax / Penalty ----
        logits     = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)
        save_id_in = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)

        torch.onnx.export(
            GREEDY_SEARCH(), (logits, save_id_in), onnx_model_Greedy,
            input_names=['logits', 'save_id_in'], output_names=['max_logits_idx', 'save_id_out'],
            dynamic_axes={'logits': {0: 'batch'}, 'save_id_in': {0: 'batch', 1: 'history_len'},
                          'max_logits_idx': {0: 'batch'}, 'save_id_out': {0: 'batch', 1: 'history_len'}},
            do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print('  [C] Greedy_Search exported')

        torch.onnx.export(
            ARGMAX(), (logits,), onnx_model_Argmax,
            input_names=['logits'], output_names=['max_logits_idx'],
            dynamic_axes={'logits': {0: 'batch'}, 'max_logits_idx': {0: 'batch'}},
            do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print('  [H] Argmax exported')

        penalty_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALITY_RANGE],  dtype=torch.int64)
        torch.onnx.export(
            APPLY_PENALTY(), (logits, save_id_in, penalty_value, penalty_range), onnx_model_Penalty,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
            output_names=['logits_out'],
            dynamic_axes={'logits_in': {0: 'batch'}, 'save_id_in': {0: 'batch', 1: 'history_len'},
                          'logits_out': {0: 'batch'}},
            do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print('  [F] Apply_Penalty exported')

        # ---- First beam (single-batch cache -> beam_size) ----
        num_layers_beam = len(cache_in_names)
        kf1, vf1, cf1 = kf[[0]], vf[[0]], cf[[0]]
        ks1 = ks_dummy[[0]] if ks_dummy is not None else None
        vs1 = vs_dummy[[0]] if vs_dummy is not None else None
        c_t, c_in, c_out, c_ax = build_cache_io(batch_axis='batch', kf_=kf1, vf_=vf1, cf_=cf1, ks_=ks1, vs_=vs1)
        beam_size_t = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        c_in_axes = {k: v for k, v in c_ax.items() if k in c_in}
        torch.onnx.export(
            FIRST_BEAM_SEARCH(num_layers_beam),
            tuple(c_t + [logits[[0]], save_id_in, beam_size_t]), onnx_model_First_Beam,
            input_names=c_in + ['logits', 'save_id_in', 'beam_size'],
            output_names=['out_' + n[3:] for n in c_in] + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={**c_in_axes, 'logits': {0: 'batch'}, 'save_id_in': {0: 'batch', 1: 'history_len'},
                          'save_id_out': {0: 'batch', 1: 'history_len'}, 'top_beam_prob': {0: 'batch'},
                          'top_beam_indices': {0: 'batch'}, 'max_logits_idx': {0: 'batch'}},
            do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print('  [D] First_Beam_Search exported')

        # ---- Second beam ----
        c_t, c_in, c_out, c_ax = build_cache_io(batch_axis='batch')
        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
        topK_t = torch.tensor([TOP_K], dtype=torch.int64)
        torch.onnx.export(
            SECOND_BEAM_SEARCH(num_layers_beam),
            tuple(c_t + [logits, save_id_in, previous_prob, beam_size_t, topK_t]), onnx_model_Second_Beam,
            input_names=c_in + ['logits', 'save_id_in', 'previous_prob', 'beam_size', 'topK'],
            output_names=c_out + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={**c_ax, 'logits': {0: 'batch'}, 'save_id_in': {0: 'batch', 1: 'history_len'},
                          'previous_prob': {0: 'batch'}, 'save_id_out': {0: 'batch', 1: 'history_len'},
                          'top_beam_prob': {0: 'batch'}, 'top_beam_indices': {0: 'batch'}, 'max_logits_idx': {0: 'batch'}},
            do_constant_folding=True, opset_version=OPSET, dynamo=False)
        print('  [E] Second_Beam_Search exported')

        del logits, save_id_in, penalty_value, penalty_range, previous_prob
        gc.collect()
    print('\nExport done!\n')


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

tokenizer      = AutoTokenizer.from_pretrained(path_lfm)
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
