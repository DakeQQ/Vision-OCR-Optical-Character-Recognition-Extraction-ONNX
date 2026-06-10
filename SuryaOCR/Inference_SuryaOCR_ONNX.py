import time
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR                         = Path(__file__).resolve().parent
TOKENIZER_DIR                    = "/home/DakeQQ/Downloads/surya-ocr-2"                 # Surya OCR 2 tokenizer path.
ONNX_DIR                         = BASE_DIR / "SuryaOCR_Optimized"                      # Directory containing exported ONNX models.

# ── ONNX Model Paths ─────────────────────────────────────────────────────────
onnx_model_Embed                 = str(ONNX_DIR / "LLM_Embed.onnx")                     # Token embedding lookup.
onnx_model_Vision                = str(ONNX_DIR / "LLM_Vision.onnx")                    # Fused vision: preprocess + encode + concat.
onnx_model_Rotary_Prefill        = str(ONNX_DIR / "Rotary_Prefill.onnx")                # mRoPE rotary embeddings for prefill phase.
onnx_model_Rotary_Decode         = str(ONNX_DIR / "Rotary_Decode.onnx")                 # mRoPE rotary embeddings for decode step.
onnx_model_Main                  = str(ONNX_DIR / "LLM_Main.onnx")                      # Main transformer (full + linear attention layers).
onnx_model_Greedy                = str(ONNX_DIR / "Greedy_Search.onnx")                 # Greedy decoding: argmax + append to sequence.
onnx_model_First_Beam            = str(ONNX_DIR / "First_Beam_Search.onnx")             # First beam-search step: expand into beams.
onnx_model_Second_Beam           = str(ONNX_DIR / "Second_Beam_Search.onnx")            # Subsequent beam-search steps: prune + expand.
onnx_model_Penalty               = str(ONNX_DIR / "Apply_Penalty.onnx")                 # Repetition penalty on recent token logits.
onnx_model_Argmax                = str(ONNX_DIR / "Argmax.onnx")                        # Simple argmax over vocabulary dimension.

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
TEST_IMAGE                       = [r"./psyduck_2.png"]                     			 # Path(s) to test image(s) for inference validation.

# ── Export-Aligned Runtime Config ─────────────────────────────────────────────
STOP_TOKEN 						 = [2]													 # SuryaOCR stop_id=2
MAX_SEQ_LEN                      = 4096                                                  # Max output sequence length; must match export setting.
INPUT_IMAGE_SIZE                 = [640, 640]                                            # Image canvas [H, W]; must match export setting.
VISION_BATCH_SIZE                = 1                                                     # Number of pages/images processed per batch; must match export setting.

# ── Decoding Strategy ─────────────────────────────────────────────────────────
USE_BEAM_SEARCH                  = False                                                 # True: beam search; False: greedy search.
REPEAT_PENALTY                   = 1.0                                                   # Repetition penalty multiplier. 1.0 = no penalty.
PENALTY_RANGE                    = 20                                                    # Recent-token window size for repetition penalty.
TOP_K                            = 3                                                     # Top-K candidates per beam step.
BEAM_SIZE                        = 3                                                     # Active beam width. Must be <= MAX_BEAM_SIZE at export.

# ── Runtime Config ────────────────────────────────────────────────────────────
ORT_LOG                          = False                                                 # Enable ORT debug logging. Set False for best performance.
ORT_FP16                         = False                                                 # FP16 ORT settings. CPUs require ARM64-v8.2a or newer.
ORT_Accelerate_Providers         = []                                                    # Execution providers: ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS                      = 0                                                     # ORT thread count. 0 = auto-detect.
DEVICE_ID                        = 0                                                     # GPU device index.


def build_assistant_prompt_prefix():
	return "<|im_start|>assistant\n"


def build_image_prompt(query, num_images=1):
	vision_placeholders = "<|vision_start|><|vision_end|>" * num_images
	return (
		f"<|im_start|>user\n{vision_placeholders}{query}<|im_end|>\n"
		f"{build_assistant_prompt_prefix()}"
	)


def build_prompt_tokens(tokenizer, query, num_images=1):
	prompt = build_image_prompt(query, num_images)
	return tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)


def compute_prompt_head_len(tokenizer):
	prefix = "<|im_start|>user\n<|vision_start|>"
	tokens = tokenizer(prefix, return_tensors="np")["input_ids"]
	return int(tokens.shape[-1])


def is_valid_image_path(path):
	candidate = Path(path)
	return candidate.exists() and candidate.suffix.lower() in {
		".jpg",
		".jpeg",
		".png",
		".bmp",
		".gif",
		".tiff",
		".raw",
	}


def load_image_resize(path, target_h, target_w):
	resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
	with Image.open(path) as image:
		if image.mode != "RGB":
			image = image.convert("RGB")
		if image.size != (target_w, target_h):
			image = image.resize((target_w, target_h), resampling)
	return np.ascontiguousarray(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))


def load_images(image_paths, target_h, target_w, batch_size, dynamic_shape):
	if not image_paths:
		raise ValueError("TEST_IMAGE must contain at least one valid image path.")

	images = [load_image_resize(path, target_h, target_w) for path in image_paths]
	if not dynamic_shape:
		blank_image = np.full((3, target_h, target_w), 128, dtype=np.uint8)
		while len(images) < batch_size:
			images.append(blank_image)

	pixel_values = np.stack(images, axis=0)
	pixel_values = np.expand_dims(pixel_values, axis=1)
	return np.ascontiguousarray(pixel_values)


def bind_ort_in_buf(binding, names, values):
	for name, value in zip(names, values):
		binding.bind_ortvalue_input(name, value)


def bind_ort_out_buf(binding, names, values):
	for name, value in zip(names, values):
		binding.bind_ortvalue_output(name, value)


def bind_ort_out(binding, names, device):
	for name in names:
		binding._iobinding.bind_output(name, device)


def create_ort_from_numpy(array, device, device_id):
	return onnxruntime.OrtValue.ortvalue_from_numpy(
		np.ascontiguousarray(array), device, device_id
	)


def create_ort_with_data(data, dtype, device, device_id):
	return onnxruntime.OrtValue.ortvalue_from_numpy(
		np.array(data, dtype=dtype), device, device_id
	)


def create_ort_with_shape(shape, dtype, device, device_id):
	return onnxruntime.OrtValue.ortvalue_from_numpy(
		np.zeros(shape, dtype=dtype), device, device_id
	)


def create_session(
	model_path, _session_opts, _providers, _provider_options, _disabled_optimizers
):
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
	return [
		name
		for name in names
		if name.startswith(prefix)
		and all(token not in name for token in exclude_substrings)
	]


def create_ort_with_meta_shape(
	meta, dtype, device, device_id, batch_size=1, seq_axis=None, seq_len=0
):
	shape = list(meta.shape)
	rank = len(shape)

	if seq_axis is not None and seq_axis < 0:
		seq_axis += rank

	for index, dim in enumerate(shape):
		if index == 0:
			shape[index] = batch_size
		elif seq_axis is not None and index == seq_axis:
			shape[index] = seq_len
		elif not isinstance(dim, (int, np.integer)):
			shape[index] = 1

	return create_ort_with_shape(tuple(shape), dtype, device, device_id)


def main():
	global run_options

	use_beam_search = USE_BEAM_SEARCH
	top_k_value = TOP_K
	beam_size_value = BEAM_SIZE
	if use_beam_search and top_k_value < beam_size_value:
		top_k_value = beam_size_value
	if top_k_value < 2 or beam_size_value < 2:
		use_beam_search = False
	if not use_beam_search:
		beam_size_value = 1
	use_penalty = REPEAT_PENALTY != 1.0

	tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
	stop_token_set = set(STOP_TOKEN)
	runtime_prompt_head_len = compute_prompt_head_len(tokenizer)

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
		"session.set_denormal_as_zero": "1",
		"session.intra_op.allow_spinning": "1",
		"session.inter_op.allow_spinning": "1",
		"session.enable_quant_qdq_cleanup": "1",
		"session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
		"session.use_device_allocator_for_initializers": "1",
		"session.graph_optimizations_loop_level": "2",
		"optimization.enable_gelu_approximation": "1",
		"optimization.minimal_build_optimizations": "",
		"optimization.enable_cast_chain_elimination": "1",
		"optimization.disable_specified_optimizers": (
			"CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
			if ORT_FP16
			else ""
		),
	}
	for key, value in session_config_entries.items():
		session_opts.add_session_config_entry(key, value)

	run_options.add_run_config_entry(
		"disable_synchronize_execution_providers", "0"
	)
	disabled_optimizers = (
		["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
		if ORT_FP16
		else None
	)

	if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
		provider_options = [
			{
				"device_type": "CPU",
				"precision": "ACCURACY",
				"num_of_threads": MAX_THREADS if MAX_THREADS != 0 else 8,
				"num_streams": 1,
				"enable_opencl_throttling": False,
				"enable_qdq_optimizer": False,
				"disable_dynamic_shapes": False,
			}
		]
		device_type = "cpu"
		ort_device_type = C.OrtDevice.cpu()
	elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
		provider_options = [
			{
				"device_id": DEVICE_ID,
				"gpu_mem_limit": 24 * (1024**3),
				"arena_extend_strategy": "kNextPowerOfTwo",
				"cudnn_conv_algo_search": "EXHAUSTIVE",
				"sdpa_kernel": "2",
				"use_tf32": "1",
				"fuse_conv_bias": "1",
				"cudnn_conv_use_max_workspace": "1",
				"cudnn_conv1d_pad_to_nc1d": "0",
				"tunable_op_enable": "0",
				"tunable_op_tuning_enable": "0",
				"tunable_op_max_tuning_duration_ms": 10,
				"do_copy_in_default_stream": "1",
				"enable_cuda_graph": "0",
				"prefer_nhwc": "0",
				"enable_skip_layer_norm_strict_mode": "0",
				"use_ep_level_unified_stream": "0",
			}
		]
		device_type = "cuda"
		ort_device_type = C.OrtDevice.cuda()
	elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
		provider_options = [
			{
				"device_id": DEVICE_ID,
				"performance_preference": "high_performance",
				"device_filter": "gpu",
				"disable_metacommands": "false",
				"enable_graph_capture": "false",
				"enable_graph_serialization": "false",
			}
		]
		device_type = "dml"
		ort_device_type = C.OrtDevice.dml()
	else:
		provider_options = None
		device_type = "cpu"
		ort_device_type = C.OrtDevice.cpu()

	packed_settings = {
		"_session_opts": session_opts,
		"_providers": ORT_Accelerate_Providers,
		"_provider_options": provider_options,
		"_disabled_optimizers": disabled_optimizers,
	}

	ort_device_type = C.OrtDevice(
		ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID
	)
	kv_device = "cpu" if device_type == "dml" else device_type

	print("Loading ONNX models ... it could take a while.")

	ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
	ort_session_Vision = create_session(onnx_model_Vision, **packed_settings)
	ort_session_Rotary_Prefill = create_session(
		onnx_model_Rotary_Prefill, **packed_settings
	)
	ort_session_Rotary_Decode = create_session(
		onnx_model_Rotary_Decode, **packed_settings
	)
	ort_session_Main = create_session(onnx_model_Main, **packed_settings)

	in_name_Embed = get_in_names(ort_session_Embed)[0]
	out_name_Embed = get_out_names(ort_session_Embed)[0]

	in_name_Vision = get_in_names(ort_session_Vision)
	out_name_Vision = get_out_names(ort_session_Vision)[0]
	vision_input_meta = ort_session_Vision._inputs_meta[0]
	vision_meta_shape = vision_input_meta.shape
	vision_batch_meta = vision_meta_shape[0]
	dynamic_image_shape = not isinstance(vision_batch_meta, int)
	vision_batch_size = (
		vision_batch_meta if isinstance(vision_batch_meta, int) else VISION_BATCH_SIZE
	)

	image_height = vision_meta_shape[3]
	image_width = vision_meta_shape[4]
	if isinstance(image_height, int) and isinstance(image_width, int):
		input_image_size = [image_height, image_width]
	else:
		input_image_size = INPUT_IMAGE_SIZE[:]

	in_name_Rotary_Prefill = get_in_names(ort_session_Rotary_Prefill)
	out_name_Rotary_Prefill = get_out_names(ort_session_Rotary_Prefill)

	in_name_Rotary_Decode = get_in_names(ort_session_Rotary_Decode)[0]
	out_name_Rotary_Decode = get_out_names(ort_session_Rotary_Decode)
	out_meta_Rotary_Decode = ort_session_Rotary_Decode._outputs_meta

	print(f"Usable Providers: {ort_session_Main.get_providers()}")

	in_name_Main = get_in_names(ort_session_Main)
	out_name_Main = get_out_names(ort_session_Main)
	in_meta_Main = ort_session_Main._inputs_meta
	in_meta_Main_by_name = {
		name: meta for name, meta in zip(in_name_Main, in_meta_Main)
	}

	num_keys_values_main = len(out_name_Main) - 1
	num_keys_values_main_plus_1 = num_keys_values_main + 1
	num_keys_values_main_plus_2 = num_keys_values_main + 2
	num_keys_values_main_plus_3 = num_keys_values_main + 3
	idx_rotary_cos = num_keys_values_main + 1

	in_name_Main_kv = in_name_Main[:num_keys_values_main]
	out_name_Main_kv = out_name_Main[:num_keys_values_main]
	out_name_Main_logits = out_name_Main[num_keys_values_main]

	in_name_Main_keys = select_state_input_names(
		in_name_Main_kv, "in_key_", exclude_substrings=("scale", "bias")
	)
	in_name_Main_values = select_state_input_names(
		in_name_Main_kv, "in_value_", exclude_substrings=("scale", "bias")
	)
	in_name_Main_key_scales = select_state_input_names(in_name_Main_kv, "in_key_scale_")
	in_name_Main_key_biases = select_state_input_names(in_name_Main_kv, "in_key_bias_")
	in_name_Main_value_scales = select_state_input_names(
		in_name_Main_kv, "in_value_scale_"
	)
	in_name_Main_value_biases = select_state_input_names(
		in_name_Main_kv, "in_value_bias_"
	)
	in_name_Main_conv_states = select_state_input_names(
		in_name_Main_kv, "in_conv_state_"
	)
	in_name_Main_recurrent_states = select_state_input_names(
		in_name_Main_kv, "in_recurrent_state_"
	)

	kv_dtype_str = in_meta_Main[0].type
	hidden_dtype_main = (
		np.float16
		if "float16" in in_meta_Main[num_keys_values_main].type
		else np.float32
	)
	vocab_size = ort_session_Main._outputs_meta[num_keys_values_main].shape[1]

	if "int32" in kv_dtype_str:
		kv_dtype_main = np.int32
	elif "uint8" in kv_dtype_str:
		kv_dtype_main = np.uint8
	elif "int8" in kv_dtype_str:
		kv_dtype_main = np.int8
	else:
		kv_dtype_main = np.float16 if "float16" in kv_dtype_str else np.float32

	scale_dtype_main = None
	if in_name_Main_key_scales:
		scale_dtype_main = (
			np.float16
			if "float16" in in_meta_Main_by_name[in_name_Main_key_scales[0]].type
			else np.float32
		)

	def build_state_buffers(batch_size):
		past_keys = create_ort_with_meta_shape(
			in_meta_Main_by_name[in_name_Main_keys[0]],
			kv_dtype_main,
			kv_device,
			DEVICE_ID,
			batch_size=batch_size,
			seq_axis=-1,
		)
		past_values = create_ort_with_meta_shape(
			in_meta_Main_by_name[in_name_Main_values[0]],
			kv_dtype_main,
			kv_device,
			DEVICE_ID,
			batch_size=batch_size,
			seq_axis=3,
		)
		key_scales = None
		value_scales = None
		key_biases = None
		value_biases = None
		if in_name_Main_key_scales:
			key_scales = create_ort_with_meta_shape(
				in_meta_Main_by_name[in_name_Main_key_scales[0]],
				scale_dtype_main,
				kv_device,
				DEVICE_ID,
				batch_size=batch_size,
				seq_axis=-1,
			)
			value_scales = create_ort_with_meta_shape(
				in_meta_Main_by_name[in_name_Main_value_scales[0]],
				scale_dtype_main,
				kv_device,
				DEVICE_ID,
				batch_size=batch_size,
				seq_axis=3,
			)
		if in_name_Main_key_biases:
			key_biases = create_ort_with_meta_shape(
				in_meta_Main_by_name[in_name_Main_key_biases[0]],
				scale_dtype_main,
				kv_device,
				DEVICE_ID,
				batch_size=batch_size,
				seq_axis=-1,
			)
			value_biases = create_ort_with_meta_shape(
				in_meta_Main_by_name[in_name_Main_value_biases[0]],
				scale_dtype_main,
				kv_device,
				DEVICE_ID,
				batch_size=batch_size,
				seq_axis=3,
			)
		conv_states = None
		recurrent_states = None
		if in_name_Main_conv_states:
			conv_states = create_ort_with_meta_shape(
				in_meta_Main_by_name[in_name_Main_conv_states[0]],
				np.float16,
				kv_device,
				DEVICE_ID,
				batch_size=batch_size,
			)
		if in_name_Main_recurrent_states:
			recurrent_states = create_ort_with_meta_shape(
				in_meta_Main_by_name[in_name_Main_recurrent_states[0]],
				np.float16,
				kv_device,
				DEVICE_ID,
				batch_size=batch_size,
			)
		return (
			past_keys,
			past_values,
			key_scales,
			value_scales,
			key_biases,
			value_biases,
			conv_states,
			recurrent_states,
		)

	test_image_paths = TEST_IMAGE if isinstance(TEST_IMAGE, list) else [TEST_IMAGE]
	valid_images = [str(path) for path in test_image_paths if is_valid_image_path(path)]
	if not valid_images:
		raise RuntimeError(
			"Surya OCR requires image input. No valid image found in TEST_IMAGE."
		)

	if dynamic_image_shape:
		runtime_images = valid_images
		prompt_num_images = len(runtime_images)
	else:
		runtime_images = valid_images[:vision_batch_size]
		prompt_num_images = vision_batch_size

	print("\nStart to Process the Image(s) (one-time)...")
	vision_start_time = time.time()
	pixel_values = load_images(
		runtime_images,
		input_image_size[0],
		input_image_size[1],
		vision_batch_size,
		dynamic_image_shape,
	)

	first_query = next(iter(PROMPT_MAP.values()))
	first_tokens = build_prompt_tokens(tokenizer, first_query, num_images=prompt_num_images)
	first_input_ids = create_ort_from_numpy(first_tokens, device_type, DEVICE_ID)

	bind_embed = ort_session_Embed.io_binding()
	bind_embed.bind_ortvalue_input(in_name_Embed, first_input_ids)
	bind_ort_out(bind_embed, [out_name_Embed], ort_device_type)
	run(ort_session_Embed, bind_embed)
	first_hidden = bind_embed.get_outputs()[0]
	first_hidden_np = first_hidden.numpy()

	bind_vision = ort_session_Vision.io_binding()
	bind_vision.bind_ortvalue_input(
		in_name_Vision[0], create_ort_from_numpy(pixel_values, device_type, DEVICE_ID)
	)
	bind_vision.bind_ortvalue_input(in_name_Vision[1], first_hidden)
	bind_ort_out(bind_vision, [out_name_Vision], ort_device_type)
	run(ort_session_Vision, bind_vision)
	first_concat_np = bind_vision.get_outputs()[0].numpy()

	vision_embed_size = first_concat_np.shape[1] - first_hidden_np.shape[1]
	cached_vision_embeds = first_concat_np[
		:, runtime_prompt_head_len : runtime_prompt_head_len + vision_embed_size, :
	].copy()
	print(
		f"Image Process Complete. Time Cost: {time.time() - vision_start_time:.3f} Seconds"
	)

	for task_index, (task_name, query) in enumerate(PROMPT_MAP.items(), start=1):
		task_label = task_name.upper()
		decoding_method = "Beam Search" if use_beam_search else "Greedy Search"
		print("\n" + "=" * 60)
		print(
			f"  Task {task_index}/{len(PROMPT_MAP)}: {task_label} (image mode, {decoding_method})"
		)
		print("=" * 60)

		tokens = build_prompt_tokens(tokenizer, query, num_images=prompt_num_images)
		num_prefill = tokens.shape[-1]

		input_ids = create_ort_from_numpy(tokens, device_type, DEVICE_ID)
		ids_len = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
		init_history_len = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
		top_k = create_ort_with_data([top_k_value], np.int64, device_type, DEVICE_ID)
		beam_size = create_ort_with_data(
			[beam_size_value], np.int64, device_type, DEVICE_ID
		)

		attention_mask_buf = create_ort_with_shape(
			(1, 1, 1, 1, 1), hidden_dtype_main, device_type, DEVICE_ID
		)
		rotary_cos_buf = create_ort_with_meta_shape(
			out_meta_Rotary_Decode[0], hidden_dtype_main, device_type, DEVICE_ID
		)
		rotary_sin_buf = create_ort_with_meta_shape(
			out_meta_Rotary_Decode[1], hidden_dtype_main, device_type, DEVICE_ID
		)
		hidden_states_buf = create_ort_with_meta_shape(
			in_meta_Main[num_keys_values_main],
			hidden_dtype_main,
			device_type,
			DEVICE_ID,
			batch_size=beam_size_value,
			seq_axis=1,
			seq_len=1,
		)
		save_id_buf = create_ort_with_shape(
			(beam_size_value, 0), np.int32, device_type, DEVICE_ID
		)
		prefill_logits_buf = create_ort_with_shape(
			(1, vocab_size), hidden_dtype_main, device_type, DEVICE_ID
		)
		decode_logits_buf = create_ort_with_shape(
			(beam_size_value, vocab_size), hidden_dtype_main, device_type, DEVICE_ID
		)
		max_idx_buf = create_ort_with_shape(
			(1, 1), np.int32, device_type, DEVICE_ID
		)

		(
			past_keys_Main,
			past_values_Main,
			k_scales_Main,
			v_scales_Main,
			k_biases_Main,
			v_biases_Main,
			past_conv_states_Main,
			past_recurrent_states_Main,
		) = build_state_buffers(beam_size_value)

		binding_Embed = ort_session_Embed.io_binding()
		binding_Main = ort_session_Main.io_binding()
		binding_Rotary_Prefill = ort_session_Rotary_Prefill.io_binding()
		binding_Rotary_Decode = ort_session_Rotary_Decode.io_binding()

		if use_beam_search:
			ort_session_First_Beam = create_session(
				onnx_model_First_Beam, **packed_settings
			)
			binding_First_Beam = ort_session_First_Beam.io_binding()
			in_name_First_Beam = get_in_names(ort_session_First_Beam)
			out_name_First_Beam = get_out_names(ort_session_First_Beam)
			in_name_First_Beam_parts = in_name_First_Beam[
				:num_keys_values_main_plus_1
			]
			out_name_First_Beam_parts = out_name_First_Beam[
				:num_keys_values_main_plus_1
			]
			out_name_First_Beam_others = out_name_First_Beam[
				num_keys_values_main_plus_1:
			]

			ort_session_Second_Beam = create_session(
				onnx_model_Second_Beam, **packed_settings
			)
			binding_Second_Beam = ort_session_Second_Beam.io_binding()
			in_name_Second_Beam = get_in_names(ort_session_Second_Beam)
			out_name_Second_Beam = get_out_names(ort_session_Second_Beam)
			in_name_Second_Beam_parts = in_name_Second_Beam[
				:num_keys_values_main_plus_1
			]
			out_name_Second_Beam_parts = out_name_Second_Beam[
				:num_keys_values_main_plus_1
			]
			out_name_Second_Beam_others = out_name_Second_Beam[
				num_keys_values_main_plus_1:
			]

			beam_ids_buf = create_ort_with_shape(
				(beam_size_value, 1), np.int32, device_type, DEVICE_ID
			)
			beam_score_buf = create_ort_with_shape(
				(beam_size_value, 1), hidden_dtype_main, device_type, DEVICE_ID
			)
			bind_ort_in_buf(
				binding_First_Beam,
				in_name_First_Beam[
					num_keys_values_main_plus_1:num_keys_values_main_plus_3
				],
				[save_id_buf, beam_size],
			)
			bind_ort_in_buf(
				binding_Second_Beam,
				in_name_Second_Beam[num_keys_values_main_plus_3:],
				[beam_size, top_k],
			)
		else:
			ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
			binding_Greedy = ort_session_Greedy.io_binding()
			in_name_Greedy = get_in_names(ort_session_Greedy)
			out_name_Greedy = get_out_names(ort_session_Greedy)
			binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

			ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
			binding_Argmax = ort_session_Argmax.io_binding()
			in_name_Argmax = get_in_names(ort_session_Argmax)[0]
			out_name_Argmax = get_out_names(ort_session_Argmax)[0]
			save_id_list = []

		if use_penalty:
			ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
			binding_Penalty = ort_session_Penalty.io_binding()
			in_name_Penalty = get_in_names(ort_session_Penalty)
			out_name_Penalty = get_out_names(ort_session_Penalty)[0]
			penalty_dtype = (
				np.float16
				if "float16" in ort_session_Penalty._inputs_meta[2].type
				else np.float32
			)
			penalty_value = create_ort_with_data(
				[REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID
			)
			penalty_range = create_ort_with_data(
				[PENALTY_RANGE], np.int64, device_type, DEVICE_ID
			)
			bind_ort_in_buf(
				binding_Penalty,
				in_name_Penalty[2:],
				[penalty_value, penalty_range],
			)

		is_prefill_step = True
		prefill_start_time = time.time()
		prefill_elapsed = 0.0
		decode_start_time = prefill_start_time

		binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
		bind_ort_out(binding_Embed, [out_name_Embed], ort_device_type)
		run(ort_session_Embed, binding_Embed)
		hidden_states = binding_Embed.get_outputs()[0]
		binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

		generate_limit = MAX_SEQ_LEN - num_prefill
		text_hidden_np = hidden_states.numpy()
		concat_np = np.concatenate(
			[
				text_hidden_np[:, :runtime_prompt_head_len, :],
				cached_vision_embeds,
				text_hidden_np[:, runtime_prompt_head_len:, :],
			],
			axis=1,
		)
		concat_hidden_states = create_ort_from_numpy(
			concat_np.astype(hidden_dtype_main, copy=False), device_type, DEVICE_ID
		)

		num_prefill += vision_embed_size
		ids_len = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
		generate_limit -= vision_embed_size

		bind_ort_in_buf(
			binding_Rotary_Prefill,
			in_name_Rotary_Prefill,
			[ids_len, init_history_len],
		)
		bind_ort_out(
			binding_Rotary_Prefill, out_name_Rotary_Prefill, ort_device_type
		)
		run(ort_session_Rotary_Prefill, binding_Rotary_Prefill)
		rotary_cos, rotary_sin, attention_mask, kv_seq_len = (
			binding_Rotary_Prefill.get_outputs()
		)

		binding_Main.bind_ortvalue_input(
			in_name_Main[num_keys_values_main], concat_hidden_states
		)
		binding_Rotary_Decode.bind_ortvalue_input(in_name_Rotary_Decode, kv_seq_len)
		bind_ort_out_buf(
			binding_Rotary_Decode,
			out_name_Rotary_Decode,
			[rotary_cos_buf, rotary_sin_buf, kv_seq_len],
		)
		bind_ort_in_buf(
			binding_Main,
			in_name_Main[idx_rotary_cos:],
			[rotary_cos, rotary_sin, attention_mask],
		)

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

		bind_ort_out(binding_Main, out_name_Main_kv, ort_device_type)
		binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

		if use_penalty:
			binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
			binding_Penalty.bind_ortvalue_output(out_name_Penalty, prefill_logits_buf)

		if use_beam_search:
			binding_First_Beam.bind_ortvalue_input(
				in_name_First_Beam[num_keys_values_main], prefill_logits_buf
			)
		elif use_penalty:
			binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], prefill_logits_buf)
			binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
		else:
			binding_Argmax.bind_ortvalue_input(in_name_Argmax, prefill_logits_buf)
			binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)

		if use_beam_search:
			print(f"\n  Prompt: {query}")
			print(
				f"  Decoding with beam search (beam_size={beam_size_value}, top_k={top_k_value})..."
			)
		else:
			print(f"\n  Prompt: {query}\n  Output: ", end="", flush=True)

		num_decode = 0
		save_id = None
		while num_decode < generate_limit:
			run(ort_session_Main, binding_Main)
			outputs_Main = binding_Main.get_outputs()

			if use_penalty and num_decode >= PENALTY_RANGE:
				binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
				run(ort_session_Penalty, binding_Penalty)

			if use_beam_search:
				if is_prefill_step:
					bind_ort_in_buf(
						binding_First_Beam, in_name_First_Beam_parts, outputs_Main
					)
					bind_ort_out(
						binding_First_Beam,
						out_name_First_Beam_parts,
						ort_device_type,
					)
					bind_ort_out_buf(
						binding_First_Beam,
						out_name_First_Beam_others,
						[beam_score_buf, beam_ids_buf, max_idx_buf],
					)
					run(ort_session_First_Beam, binding_First_Beam)
					outputs_Beam = binding_First_Beam.get_outputs()
				else:
					bind_ort_in_buf(
						binding_Second_Beam,
						in_name_Second_Beam_parts,
						outputs_Main,
					)
					bind_ort_out(
						binding_Second_Beam,
						out_name_Second_Beam_parts,
						ort_device_type,
					)
					if num_decode < 2:
						binding_Second_Beam.bind_ortvalue_input(
							in_name_Second_Beam[num_keys_values_main_plus_2],
							beam_score_buf,
						)
					bind_ort_out_buf(
						binding_Second_Beam,
						out_name_Second_Beam_others,
						[beam_score_buf, beam_ids_buf, max_idx_buf],
					)
					run(ort_session_Second_Beam, binding_Second_Beam)
					outputs_Beam = binding_Second_Beam.get_outputs()

				max_logits_idx = int(max_idx_buf.numpy().flat[0])
				if max_logits_idx in stop_token_set:
					break

				save_id = outputs_Beam[num_keys_values_main]
				bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Beam)
				binding_Second_Beam.bind_ortvalue_input(
					in_name_Second_Beam[num_keys_values_main_plus_1], save_id
				)
			else:
				if use_penalty:
					binding_Greedy._iobinding.bind_output(
						out_name_Greedy[1], ort_device_type
					)
					run(ort_session_Greedy, binding_Greedy)
					save_id = binding_Greedy.get_outputs()[1]
				else:
					run(ort_session_Argmax, binding_Argmax)

				max_logits_idx = int(max_idx_buf.numpy().flat[0])
				if max_logits_idx in stop_token_set:
					break

				if use_penalty:
					binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
				else:
					save_id_list.append(max_logits_idx)

				bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Main)
				print(tokenizer.decode(max_logits_idx), end="", flush=True)

			bind_ort_out(binding_Main, out_name_Main_kv, ort_device_type)

			if is_prefill_step:
				binding_Main.bind_ortvalue_input(
					in_name_Main[num_keys_values_main], hidden_states_buf
				)
				bind_ort_in_buf(
					binding_Main,
					in_name_Main[idx_rotary_cos:],
					[rotary_cos_buf, rotary_sin_buf, attention_mask_buf],
				)
				binding_Main.bind_ortvalue_output(
					out_name_Main_logits, decode_logits_buf
				)
				binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

				if use_penalty:
					binding_Penalty.bind_ortvalue_input(
						in_name_Penalty[0], decode_logits_buf
					)
					binding_Penalty.bind_ortvalue_output(
						out_name_Penalty, decode_logits_buf
					)

				if use_beam_search:
					binding_Second_Beam.bind_ortvalue_input(
						in_name_Second_Beam[num_keys_values_main], decode_logits_buf
					)
					binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
				elif use_penalty:
					binding_Greedy.bind_ortvalue_input(
						in_name_Greedy[0], decode_logits_buf
					)
				else:
					binding_Argmax.bind_ortvalue_input(
						in_name_Argmax, decode_logits_buf
					)

				is_prefill_step = False
				decode_start_time = time.time()
				prefill_elapsed = decode_start_time - prefill_start_time

			run(ort_session_Embed, binding_Embed)
			run(ort_session_Rotary_Decode, binding_Rotary_Decode)
			num_decode += 1

		decode_end_time = time.time()
		if num_decode < 2:
			prefill_elapsed = 0.0
			decode_elapsed = 0.0
		else:
			decode_elapsed = decode_end_time - decode_start_time

		total_elapsed = decode_end_time - prefill_start_time
		prefill_tokens_per_second = (
			num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
		)
		decode_tokens_per_second = (
			num_decode / decode_elapsed if decode_elapsed > 0 else 0.0
		)
		overall_tokens_per_second = (
			(num_decode + 1) / total_elapsed if total_elapsed > 0 else 0.0
		)

		if use_beam_search and save_id is not None:
			result = tokenizer.decode(
				save_id.numpy()[0, :num_decode], skip_special_tokens=True
			)
		elif use_penalty and save_id is not None:
			result = tokenizer.decode(
				save_id.numpy()[0, :num_decode], skip_special_tokens=True
			)
		else:
			result = tokenizer.decode(save_id_list, skip_special_tokens=True)

		if use_beam_search:
			print("\n\n" + "-" * 60)
			print(f"  Generated Output ({task_label})")
			print("-" * 60)
			print(result)
			print("-" * 60)
		else:
			print()

		print(
			"\n"
			+ "-" * 60
			+ f"\n  Performance Summary - Task {task_index}: {task_label}\n"
			+ "-" * 60
			+ "\n"
			+ f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
			+ f"  {'-' * 52}\n"
			+ f"  {'Prefill':<12} {prefill_tokens_per_second:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
			+ f"  {'Decode':<12} {decode_tokens_per_second:>10.2f} t/s {num_decode:>8d} {decode_elapsed:>8.3f}s\n"
			+ f"  {'-' * 52}\n"
			+ f"  {'Overall':<12} {overall_tokens_per_second:>10.2f} t/s {num_decode + 1:>8d} {total_elapsed:>8.3f}s\n"
			+ "-" * 60
			+ "\n"
		)


if __name__ == "__main__":
	main()
