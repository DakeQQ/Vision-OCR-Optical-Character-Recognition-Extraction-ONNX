import os
import gc
import glob
import shutil
import onnx
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils,
)


# ==============================================================================
# Path Settings
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
original_folder_path = os.path.join(SCRIPT_DIR, 'PPOCRv6_ONNX')
quanted_folder_path = os.path.join(SCRIPT_DIR, 'PPOCRv6_Optimized')
os.makedirs(quanted_folder_path, exist_ok=True)


# ==============================================================================
# Lazy Settings (set one True for auto-select, or both False for manual)
# ==============================================================================
lazy_setting_CPU = True                  # Auto-select CPU settings.
lazy_setting_GPU = False                 # Auto-select GPU settings.
use_openvino     = False
SAVE_TWO_PARTS   = False
upgrade_opset    = 0


# ==============================================================================
# Model List  (matches the files emitted by Export_PPOCRv6.py)
# ==============================================================================
model_names = [                           # Recommended dtype:
    "PPOCRv6_DocOri",                     # [float32, float16]   conv classifier
    "PPOCRv6_Unwarp",                     # [float32, float16]   conv + GridSample
    "PPOCRv6_Det",                        # [float32, float16]   conv backbone + neck
    "PPOCRv6_DBPost",                     # [float32]            hand-built DB postprocess (Loop/NonZero/Compress)
    "PPOCRv6_Rec",                        # [int8, float32, float16]  conv + textline-If + SVTR/CTC MatMul
]


# ==============================================================================
# Manual Quantization Settings
# ==============================================================================
quant_int4       = False
quant_int8       = False
quant_float16    = False
keep_io_dtype    = True
fp16_op_block_list = [
    'DynamicQuantizeLinear',
    'DequantizeLinear',
    'DynamicQuantizeMatMul',
    'Range',
    'MatMulIntegerToFloat',
    'GridSample',                               # keep the UVDoc warp in float32
    'Resize',
]


# ==============================================================================
# Int4 matmul_nbits_quantizer Settings  (recognition head only)
# ==============================================================================
algorithm        = "k_quant"             # ["DEFAULT", "RTN", "HQQ", "k_quant"]
bits             = 8
block_size       = 32
accuracy_level   = 4                     # 0:default 1:fp32 2:fp16 3:bf16 4:int8
quant_symmetric  = False
nodes_to_exclude = None


# ==============================================================================
# Per-Model Target Dtype Mapping (CPU)
# ==============================================================================
CPU_MODEL_DTYPE = {
    "PPOCRv6_DocOri":   "float32",
    "PPOCRv6_Unwarp":   "float32",
    "PPOCRv6_Det":      "float32",
    "PPOCRv6_DBPost":   "float32",       # control-flow graph, copied verbatim
    "PPOCRv6_Rec":      "float32",       # set "int4" to weight-quantize the SVTR/CTC MatMuls
}


# ==============================================================================
# Per-Model Target Dtype Mapping (GPU)
# ==============================================================================
GPU_MODEL_DTYPE = {
    "PPOCRv6_DocOri":   "float16",
    "PPOCRv6_Unwarp":   "float16",
    "PPOCRv6_Det":      "float16",
    "PPOCRv6_DBPost":   "float32",       # fp16 overflows the CC labels + integral image
    "PPOCRv6_Rec":      "float16",
}

if lazy_setting_CPU and lazy_setting_GPU:
    raise ValueError("Only one of lazy_setting_CPU or lazy_setting_GPU can be True.")


# ==============================================================================
# Helper Functions
# ==============================================================================
def _is_matmul_block(name):
    """Only the recognition head carries weight MatMuls worth quantizing."""
    return name == "PPOCRv6_Rec"


def _is_postprocess_graph(name):
    """The DB postprocess graph is hand-built control flow, not a CNN/transformer."""
    return name == "PPOCRv6_DBPost"


def _opt_level(name):
    return 1 if use_openvino else 2


def _num_heads(name):
    return 6 if name == "PPOCRv6_Rec" else 0    # SVTR: hidden 192 / head_dim 32


def _hidden_size(name):
    return 192 if name == "PPOCRv6_Rec" else 0


# ==============================================================================
# Core Processing Function
# ==============================================================================
def process_single_model(
    model_path,
    quanted_model_path,
    model_name,
    bits,
    block_size,
    quant_int4_flag,
    quant_int8_flag,
    quant_float16_flag,
    keep_io_flag,
    op_block_list,
):
    """Quantize / optimize / slim a single PP-OCRv6 ONNX graph."""
    be_optimized = False

    # ------------------------------------------------------------------
    # Branch 0: DB postprocess (hand-built control-flow graph)
    # ------------------------------------------------------------------
    # Loop / NonZero / Compress / ScatterElements with outer-scope Loop-body
    # references, plus connected-component labels and an integral image that
    # exceed the float16 range.  It is already minimal and numerically exact, so
    # copy it verbatim into the optimized set (float32) rather than risk the
    # transformer optimizer, float16 conversion or onnxslim rewriting the control
    # flow.  Returns early so none of the optimization passes below touch it.
    if _is_postprocess_graph(model_name):
        print("Postprocess control-flow graph: copying verbatim (float32, no rewrite)...")
        model = onnx.load(model_path)
        onnx.save(model, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
        del model
        gc.collect()
        return

    # ------------------------------------------------------------------
    # Branch 1: int4/int8 weight-only quantization (recognition MatMuls)
    # ------------------------------------------------------------------
    if (quant_int4_flag or quant_int8_flag) and _is_matmul_block(model_name):
        op_types = ["MatMul"]
        quant_axes = [0]
        axes_tuple = (("MatMul", 0),)
        model = quant_utils.load_model_with_shape_infer(Path(model_path))

        if quant_int8_flag:
            bits = 8
            algorithm = "DEFAULT"  # int8 only supports the default quantization algorithm


        if algorithm == "RTN":
            quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
            )
        elif algorithm == "k_quant":
            quant_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
            )
        else:
            quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=block_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=axes_tuple,
            )
        quant_config.bits = bits

        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=axes_tuple,
            algo_config=quant_config,
            nodes_to_exclude=nodes_to_exclude,
        )
        quant.process()
        quant.model.save_model_to_file(quanted_model_path, True)

    # ------------------------------------------------------------------
    # Branch 2: Float16 conversion
    # ------------------------------------------------------------------
    elif quant_float16_flag:
        print("Optimizing model before Float16 conversion...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        print("Converting model to Float16...")
        model.convert_float_to_float16(
            keep_io_types=keep_io_flag,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=32767.0,
            op_block_list=op_block_list,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Branch 3: Float32 (optimize only, lossless)
    # ------------------------------------------------------------------
    else:
        print("Target dtype is float32: optimizing without quantization...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Post-quantization optimization pass
    # ------------------------------------------------------------------
    if not be_optimized:
        print("Running additional ONNX Runtime optimization on quantized model...")
        model = optimize_model(
            quanted_model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Slim pass
    # ------------------------------------------------------------------
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=False,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=SAVE_TWO_PARTS,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Optional opset upgrade / final save
    # ------------------------------------------------------------------
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            m = onnx.load(quanted_model_path)
            converted = onnx.version_converter.convert_version(m, upgrade_opset)
            onnx.save(converted, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m, converted
            gc.collect()
        except Exception as exc:
            print(f"Could not upgrade opset: {exc}. Keeping original opset.")
            m = onnx.load(quanted_model_path)
            onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m
            gc.collect()
    else:
        m = onnx.load(quanted_model_path)
        onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
        del m
        gc.collect()


# ==============================================================================
# Main Processing Loop
# ==============================================================================
for model_name in model_names:
    print(f"\n--- Processing model: {model_name} ---")

    if lazy_setting_GPU:
        target_dtype = GPU_MODEL_DTYPE.get(model_name, "float16")
        keep_io_dtype = False
    elif lazy_setting_CPU:
        target_dtype = CPU_MODEL_DTYPE.get(model_name, "float32")
        keep_io_dtype = True
    else:
        target_dtype = None

    if target_dtype:
        quant_int4 = (target_dtype == "int4")
        quant_int8 = (target_dtype == "int8")
        quant_float16 = (target_dtype == "float16")
        if quant_int4:
            bits = 4
        elif quant_int8:
            bits = 8

    print(f"Selected target dtype for {model_name}: {target_dtype}")
    print(f"quant_int4={quant_int4}, quant_int8={quant_int8}, "
          f"quant_float16={quant_float16}, keep_io_dtype={keep_io_dtype}")

    model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
    quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")

    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping.")
        continue

    process_single_model(
        model_path, quanted_model_path, model_name,
        bits, block_size, quant_int4, quant_int8,
        quant_float16, keep_io_dtype, fp16_op_block_list,
    )


# ==============================================================================
# Companion data
# ==============================================================================
# Mirror rec_char_list.npy into the optimized set so PPOCRv6_Optimized is a
# self-contained drop-in for the runtime (the recognition CTC decoder loads the
# char list from alongside the models, exactly as Export_PPOCRv6.py emits it).
char_list_src = os.path.join(original_folder_path, 'rec_char_list.npy')
char_list_dst = os.path.join(quanted_folder_path, 'rec_char_list.npy')
if os.path.exists(char_list_src):
    shutil.copy2(char_list_src, char_list_dst)
    print(f"Copied {char_list_src} -> {char_list_dst}")
else:
    print(f"Warning: {char_list_src} not found; optimized set will lack the char list.")


# ==============================================================================
# Cleanup
# ==============================================================================
print("Cleaning up temporary *.onnx.data files...")
for file_path in glob.glob(os.path.join(quanted_folder_path, '*.onnx.data')):
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as exc:
        print(f"Error deleting {file_path}: {exc}")

print("--- All models processed successfully! ---")
