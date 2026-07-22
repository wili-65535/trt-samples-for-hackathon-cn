# Candidates from NVIDIA/Model-Optimizer

Source repo analyzed: `/work/trt/repos/Model-Optimizer` (NVIDIA Model Optimizer / ModelOpt).
Cookbook target: `/work/trt-samples-for-hackathon-cn/cookbook`.

## Context / de-duplication baseline

The existing example `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/main.py` already covers three
ModelOpt entry points on a tiny MNIST CNN, all landing in a strongly-typed TRT engine:

- `modelopt.onnx.autocast` — FP32 -> mixed FP16/FP32 ONNX (pure cast, no calibration).
- `modelopt.torch.quantization` — INT8 fake-quant + QAT fine-tune -> INT8 Q/DQ ONNX.
- `modelopt.onnx.quantization` — ONNX PTQ inserting **FP8 (E4M3)** Q/DQ (max calibration).

So the following are **already covered** and should NOT be re-proposed as-is: basic AutoCast FP16,
basic torch INT8 QAT, basic FP8 ONNX PTQ (max calib). Candidates below deliberately extend into
formats/algorithms/paths that this example does not touch (NVFP4, MXFP8, INT4-AWQ, entropy calib,
custom plugins during PTQ, Q/DQ autotune, Torch-TensorRT in-framework, pruning, structured sparsity,
diffusion/LLM engines).

Since TRT 11 strong typing removed weak-typing INT8 calibration, ModelOpt is the sanctioned
quantization path — every candidate here emits explicit Q/DQ (or Cast) that a strongly-typed engine consumes.

---

## Ranked candidates

### 1. ONNX PTQ with a custom TensorRT plugin (IdentityConv) — HIGH
- **Repo path:** `examples/onnx_ptq/custom_op_plugin/` (`create_identity_neural_network.py`, `plugin/*.cpp/.h/CMakeLists.txt`); driven from `examples/onnx_ptq/README.md` "Quantize an ONNX model with custom op".
- **Demonstrates:** build a C++ TRT plugin (`libidentity_conv_plugin.so`), create an ONNX graph with a custom `IdentityConv` op, run `modelopt.onnx.quantization` with `--trt_plugins=<.so>` so calibration/Q-DQ placement works *through* the plugin, then deploy with `trtexec --staticPlugins`.
- **Target section:** `05-Plugin` (new leaf, e.g. `05-Plugin/ONNXParserWithPlugin-ModelOptQuant` or `05-Plugin/ModelOptPTQWithPlugin`).
- **Self-contained:** Excellent — no dataset/model downloads; synthetic ONNX + local CMake plugin build.
- **Effort/priority:** Medium effort, HIGH priority. Unique intersection (plugin + PTQ) not covered anywhere in cookbook; fits the removed-weak-typing narrative.

### 2. PyTorch vision -> ONNX in NVFP4 / MXFP8 / INT4-AWQ -> strongly-typed TRT — HIGH
- **Repo path:** `examples/torch_onnx/torch_quant_to_onnx.py` (+ `README.md`).
- **Demonstrates:** `mtq.quantize` a timm model (ViT / Swin) to **FP8, MXFP8, NVFP4, INT8, INT4-AWQ**, ONNX export (opset 20), TRT-compat post-processing, then eval via `onnx_ptq/evaluate.py --engine_precision=stronglyTyped`. Teaches the TRT-specific **Conv2d override** rule (MXFP8/NVFP4 -> FP8, INT4-AWQ -> INT8 for conv) and the `AUTO` mixed-precision mode.
- **Target section:** `04-Feature` (new `04-Feature/Quantization-NVFP4-MXFP8` or under a `03-Workflow` sibling).
- **Self-contained:** Good — pulls a timm checkpoint (tens of MB); ImageNet eval is optional/gated so skip it and verify argmax instead.
- **Effort/priority:** Medium effort, HIGH priority. Headline strong-typing formats (FP4/FP8) that the existing example lacks; needs TRT >= 10.11 for MXFP8/NVFP4.

### 3. ONNX PTQ: INT4-AWQ / INT8-entropy / RTN on a real vision model — HIGH
- **Repo path:** `examples/onnx_ptq/` (`download_example_onnx.py`, `image_prep.py`, `evaluate.py`, `README.md`).
- **Demonstrates:** the full ONNX PTQ CLI/Python API across `--quantize_mode {fp8,int8,int4}` and `--calibration_method {max,entropy,awq_clip,rtn_dq}` on ViT/ResNet, plus **per-node calibration** (`--calibrate_per_node`) for OOM-safe quantization of large graphs.
- **Target section:** `04-Feature` quantization or `03-Workflow` (ONNX-first PTQ, complements the PyTorch-first existing example).
- **Self-contained:** Moderate — downloads a timm ONNX + tiny-imagenet calib subset (non-gated); ImageNet-1k eval is gated (make optional).
- **Effort/priority:** Medium effort, HIGH priority. Directly extends the existing FP8-max-only coverage into INT4-AWQ and entropy calibration.

### 4. Torch-TensorRT in-framework FP8 PTQ (Dynamo) — HIGH
- **Repo path:** `examples/torch_trt/torch_tensorrt_ptq.py`, `torch_tensorrt_accuracy.py` (+ `README.md`).
- **Demonstrates:** `mtq.quantize` a HuggingFace ViT, then `torch_tensorrt.compile(ir="dynamo", min_block_size=1, use_explicit_typing path)` inside `export_torch_mode()` so Q/DQ become native TRT FP8 layers — no ONNX, no separate runtime. Uses non-gated `zh-plus/tiny-imagenet` calib; `--skip_trt` for non-TRT hosts.
- **Target section:** `06-DLFrameworkTRT` (Torch-TensorRT).
- **Self-contained:** Moderate — downloads `google/vit-large-patch16-224` (~1GB); smaller ViT can be substituted.
- **Effort/priority:** Medium effort, HIGH priority. Aligns with the `trt-torch-quickstart` skill and strong-typing default; no equivalent in cookbook.

### 5. Q/DQ placement optimization with Autotune — MEDIUM
- **Repo path:** `examples/onnx_ptq/autotune/` (`README.md`) + `--autotune {quick,default,extensive}` flag in `examples/onnx_ptq/README.md`.
- **Demonstrates:** automated Q/DQ node placement search driven by real TensorRT latency measurements (region discovery, pattern cache, optimize-from-existing-QDQ, remote autotuning). A genuinely TRT-perf-in-the-loop optimization.
- **Target section:** `07-Tool` (or `04-Feature`), complements existing `07-Tool/FP16Tuning`.
- **Self-contained:** Moderate — ResNet50 from ONNX Model Zoo (~100MB) + Polygraphy shape sanitize; needs a working TRT for measurement.
- **Effort/priority:** Medium effort, MEDIUM priority. Novel concept; heavier because it actually builds/times engines.

### 6. FastNAS pruning of a CNN on CIFAR (notebook) -> ONNX -> TRT — MEDIUM
- **Repo path:** `examples/pruning/cifar_resnet.ipynb` (+ `examples/pruning/README.md` FastNAS section).
- **Demonstrates:** `modelopt.torch.prune` (FastNAS) to search a subnet of a CV model under a constraint, `mto.save/restore` the heterogeneous pruned model; export ONNX and build a TRT engine to show the speedup.
- **Target section:** new `04-Feature/Pruning` (or `03-Workflow`).
- **Self-contained:** Good — CIFAR only, no gated/large downloads. "No additional dependencies" for FastNAS per README.
- **Effort/priority:** Medium effort, MEDIUM priority. Pruning is entirely absent from the cookbook; needs a TRT-export tail added (notebook stops at torch).

### 7. CNN INT8 QAT on ResNet-50 with mto.save/restore — MEDIUM
- **Repo path:** `examples/cnn_qat/torchvision_qat.py`, `utils.py` (+ `README.md`).
- **Demonstrates:** PTQ baseline (`mtq.quantize(model, INT8_DEFAULT_CFG, ...)`) -> QAT fine-tune -> `mto.save/restore` of quantizer state -> ONNX/TRT. Larger and more realistic than the MNIST QAT already in the cookbook; teaches save/restore of ModelOpt state.
- **Target section:** `04-Feature` or `03-Workflow`.
- **Self-contained:** Weak — expects an ImageNet-style train/val tree (large). Would need swapping in a small dataset to be cookbook-friendly.
- **Effort/priority:** Medium effort, MEDIUM priority. Partly overlaps existing INT8 QAT; value is mto.save/restore + realistic CNN.

### 8. Diffusion (SDXL / FLUX) INT8/FP8/FP4 PTQ -> ONNX -> strongly-typed TRT engine — MEDIUM
- **Repo path:** `examples/diffusers/quantization/` (`quantize.py`, `build_sdxl_8bit_engine.sh`, `diffusion_trt.py`, `ONNX-TRT-Deployment.md`).
- **Demonstrates:** calibrate + quantize a diffusion backbone (INT8/FP8/FP4, `--quantize-mha`), export ONNX, build the engine with `trtexec --stronglyTyped --builderOptimizationLevel=4` and multi-input dynamic shape profiles, run the E2E image pipeline. Great real-world strong-typing example with realistic shape profiles.
- **Target section:** `03-Workflow` or `04-Feature`.
- **Self-contained:** Weak — SDXL/FLUX weights are multi-GB, needs >=48GB combined memory. Aspirational / heavy.
- **Effort/priority:** High effort, MEDIUM priority. High learning value but download/size cost is large.

### 9. Windows ONNX PTQ for SAM2 / Whisper (INT4/INT8) — MEDIUM/LOW
- **Repo path:** `examples/windows/onnx_ptq/sam2/sam2_onnx_quantization.py`, `examples/windows/onnx_ptq/whisper/whisper_onnx_quantization.py` (+ per-dir `README.md`; Whisper ships `demo.wav`).
- **Demonstrates:** ModelOpt ONNX PTQ on non-classifier architectures (segmentation, ASR) producing standards-compliant INT4/INT8 ONNX. TRT path optional (primary target is ORT/DML), so needs adaptation to land in a TRT engine.
- **Target section:** `04-Feature` quantization.
- **Self-contained:** Moderate — Whisper/SAM2 checkpoints (hundreds of MB); Whisper includes sample audio.
- **Effort/priority:** Medium effort, MEDIUM/LOW priority. Broadens beyond vision classifiers; ORT-centric so extra TRT glue required.

### 10. 2:4 structured sparsity (SparseGPT PTS + SAT) for HF models — LOW
- **Repo path:** `examples/llm_sparsity/weight_sparsity/` (`hf_pts.py`, `finetune.py`, `export_trtllm_ckpt.py`, `README.md`).
- **Demonstrates:** `mts.sparsify(model, mode="sparsegpt")` for 2:4 weight sparsity, sparsity-aware fine-tune, export a TRT-LLM checkpoint. Complements existing `04-Feature/Sparsity` (which only shows the TRT sparsity *builder flag*, not how to produce sparse weights).
- **Target section:** `04-Feature/Sparsity` companion.
- **Self-contained:** Weak — Llama2-7B (gated, ~13GB), ~44GB GPU.
- **Effort/priority:** High effort, LOW priority. Conceptually valuable pairing with the TRT sparsity flag, but heavy and gated.

### 11. LLM PTQ FP8 / INT4-AWQ / NVFP4 -> TRT-LLM engine — LOW
- **Repo path:** `examples/llm_ptq/` (`hf_ptq.py`, `run_tensorrt_llm.py`, `README.md`) and `examples/hf_ptq/`.
- **Demonstrates:** the sanctioned LLM quantization path: `mtq.quantize` -> `export_tensorrt_llm_checkpoint` -> `trtllm-build`. Includes NVFP4/MXFP4 (`cast_mxfp4_to_nvfp4.py`).
- **Target section:** `09-TensorRT-LLM`.
- **Self-contained:** Weak — multi-GB LLMs, TensorRT-LLM install.
- **Effort/priority:** High effort, LOW priority (for a self-contained cookbook leaf). Good pointer/reference for `09-TensorRT-LLM`.

### 12. AutoCast BF16 + node-sensitivity extension — LOW
- **Repo path:** `docs/source/guides/8_autocast.rst`; API `modelopt.onnx.autocast.convert_to_mixed_precision`.
- **Demonstrates:** extends the existing FP16 AutoCast to **BF16** output and the node-sensitivity keep-in-FP32 logic (why some nodes stay FP32).
- **Target section:** fold into `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` (add a BF16 case) rather than a new leaf.
- **Self-contained:** Excellent — reuses the existing tiny CNN, no downloads.
- **Effort/priority:** Low effort, LOW priority. Small incremental enhancement of an existing example.

### 13. Diffusion sparse attention (skip-softmax) — LOW
- **Repo path:** `examples/diffusers/sparsity/wan22_skip_softmax.py` (+ `README.md`); also `examples/diffusers/distillation/` (QAD).
- **Demonstrates:** training-free skip-softmax sparse attention for DiT/video diffusion; QAD (quantization-aware distillation) for recovering FP4 diffusion quality.
- **Target section:** `04-Feature` (advanced).
- **Self-contained:** Weak — large diffusion/video models.
- **Effort/priority:** High effort, LOW priority. Cutting-edge but heavy; note as future reference only.

---

## Skipped (out of scope)
- `examples/vllm_serve`, `examples/llm_eval`, pure HF/vLLM serving paths — never touch TensorRT.
- `examples/deepseek`, `examples/gpt-oss`, `examples/minimax_m3`, `examples/megatron_bridge`, `examples/speculative_decoding`, `examples/alpamayo`, `examples/puzzletron` — massive model downloads / framework-specific, not self-contained.
- Minitron pruning (`examples/pruning/minitron`) — requires NeMo/Megatron containers and multi-B LLMs.

## Recommended first three to build
1. Candidate 1 (custom-op-plugin PTQ) — fully self-contained, unique to `05-Plugin`.
2. Candidate 2 (torch_onnx NVFP4/MXFP8/INT4-AWQ) — headline strong-typing formats missing today.
3. Candidate 3 (ONNX PTQ INT4-AWQ/entropy) — natural extension of the existing FP8-only workflow.
