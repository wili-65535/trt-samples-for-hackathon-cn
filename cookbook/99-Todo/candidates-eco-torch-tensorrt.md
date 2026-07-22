# Candidates from pytorch/Torch-TensorRT

Source repo: `/work/trt/repos/Torch-TensorRT` (github.com/pytorch/TensorRT).
Focus scanned: `examples/dynamo/`, `examples/distributed_inference/`, `examples/custom_converters/`, `notebooks/`, `examples/dynamo/README.rst`.

## Already covered in the cookbook (do NOT duplicate)

Existing `06-DLFrameworkTRT/`:
- `Torch-TensorRT/main.py` — basic `torch_tensorrt.compile(ir="dynamo")` on a toy MNIST CNN plus a `torch.compile` latency comparison. NOTE: it still uses `enabled_precisions={torch.float32}` and `truncate_long_and_double=True` (legacy weak-typing args) — worth modernizing to the strong-typing / `use_explicit_typing` default for TRT 11.
- `ModelOptimizer/main.py` — currently a stub (`print("Finish")`), so FP8/INT8 quantization is effectively NOT yet demonstrated end-to-end.
- Other siblings: `TensorFlow-TensorRT`, `Paddle-TensorRT`, `ONNXRuntime-TensorRT`, `DALI-TensorRT`.

All candidates below are NOT covered by the above.

---

## Ranked candidates

### 1. Engine caching (AOT + JIT)  —  PRIORITY: HIGH  —  effort: LOW
- Repo: `examples/dynamo/engine_caching_example.py` (278 lines); companion `engine_caching_bert_example.py` (73 lines).
- Demonstrates: saving built TRT engines to disk and reusing them across sessions / weight changes; shows both `torch_tensorrt.dynamo.compile` and the `torch.compile` TRT backend, plus a custom `BaseEngineCache` subclass. Times no-cache vs cache-enabled vs cache-reuse.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/` (new leaf, e.g. `EngineCaching/`).
- Self-contained: YES — pretrained ResNet18 from torchvision, no external assets. High learnability, directly maps to a common cookbook theme (caching).

### 2. Compiling with dynamic input shapes  —  PRIORITY: HIGH  —  effort: LOW
- Repo: `examples/dynamo/compile_with_dynamic_inputs.py` (132 lines).
- Demonstrates: `torch_tensorrt.Input` with `min/opt/max` shapes and `torch.export` dynamic dims on a small ViT-style model with expand/reshape ops; single engine serves a range of batch/seq without recompile.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/DynamicShapes/`.
- Self-contained: YES — toy model, no downloads.

### 3. Saving & loading compiled programs (dynamic shapes)  —  PRIORITY: HIGH  —  effort: LOW-MED
- Repo: `examples/dynamo/save_dynamic_shapes_example.py` (181 lines) + `save_dynamic_shapes_both_methods.py` (171 lines).
- Demonstrates: persisting a compiled module and preserving dynamic-shape specs across `torch_tensorrt.save` / `torch.export.save` / `.ep` load; API mirrors `torch.export` dynamic-shape handling.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/SaveLoad/`.
- Self-contained: YES.

### 4. torch.compile advanced usage (JIT frontend internals)  —  PRIORITY: HIGH  —  effort: LOW
- Repo: `examples/dynamo/torch_compile_advanced_usage.py` (107 lines); pairs well with `torch_compile_resnet_example.py` (115 lines, dynamic shapes via torch.compile).
- Demonstrates: `ir="torch_compile"` JIT path, backend options/kwargs, how graph capture/recompile behaves — the natural "AOT (`ir=dynamo`) vs JIT (`torch.compile`)" contrast the cookbook lacks.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/TorchCompileBackend/`.
- Self-contained: YES — small model / ResNet from torchvision.

### 5. Refitting engines with new weights (LoRA-style)  —  PRIORITY: HIGH  —  effort: MED
- Repo: `examples/dynamo/refit_engine_example.py` (133 lines).
- Demonstrates: `refit_module_weights` to swap weights into an already-built engine without recompiling; save/load the graph module first. Very relevant to LoRA / frequently-updated weights.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/Refit/`.
- Self-contained: YES — ResNet18.

### 6. CUDA-graph integration  —  PRIORITY: MED-HIGH  —  effort: LOW
- Repo: `examples/dynamo/torch_export_cudagraphs.py` (126 lines).
- Demonstrates: `torch_tensorrt.runtime.enable_cudagraphs()` context to capture/replay a compiled module and cut launch overhead; notes it also works in the `torch.compile` path. Complements cookbook `08-Advance` CUDA-graph material but from the Torch-TRT angle.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/CudaGraphs/`.
- Self-contained: YES — ResNet18.

### 7. FP8 post-training quantization + compile (ViT)  —  PRIORITY: MED-HIGH  —  effort: MED
- Repo: `examples/dynamo/quantize_vit_fp8.py` (292 lines).
- Demonstrates: NVIDIA ModelOpt FP8 PTQ on a HuggingFace ViT, then Dynamo compile via two attention paths (TRT `IAttention` layer vs decomposed). Directly fills the empty `ModelOptimizer` stub and the FP8/strong-typing focus.
- Target: `06-DLFrameworkTRT/ModelOptimizer/` (flesh out the stub) or a Torch-TRT `QuantizeFP8/` leaf.
- Self-contained: MOSTLY — needs `nvidia-modelopt`, `transformers`, and a Hopper+ GPU (FP8). Downloads a HF ViT.

### 8. INT8 / FP8 quantized deploy (VGG16)  —  PRIORITY: MED  —  effort: MED
- Repo: `examples/dynamo/vgg16_ptq.py` (269 lines); related legacy `examples/int8/{ptq,qat}` (C++/older) and notebooks `qat-ptq-workflow.ipynb`, `vgg-qat.ipynb`.
- Demonstrates: deploying an INT8- or FP8-quantized model through the Dynamo frontend end-to-end (calibration → quantize → compile → accuracy check).
- Target: `06-DLFrameworkTRT/ModelOptimizer/` or Torch-TRT `PTQ/`.
- Self-contained: MOSTLY — trains/quantizes VGG16 on CIFAR (dataset download), needs `nvidia-modelopt`.

### 9. Autocast / mixed-precision compilation  —  PRIORITY: MED  —  effort: LOW
- Repo: `examples/dynamo/autocast_example.py` (120 lines).
- Demonstrates: Torch-TensorRT Autocast combined with `torch.autocast` to build a mixed-precision engine — the modern strong-typing way to get FP16/BF16 without deprecated `enabled_precisions`. Strongly aligned with the TRT 11 strong-typing focus.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/MixedPrecisionAutocast/`.
- Self-contained: YES.

### 10. Multiple optimization profiles (prefill vs decode)  —  PRIORITY: MED  —  effort: MED
- Repo: `examples/dynamo/multi_optimization_profiles.py` (270 lines).
- Demonstrates: `torch_tensorrt.Input(profiles=[...])` — N optimization profiles on one input, one engine, per-call profile selection; compares single-profile vs prefill/decode-tuned engines on Gemma-3-1B. Excellent LLM-shape lesson.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/OptimizationProfiles/`.
- Self-contained: MOSTLY — downloads `google/gemma-3-1b-it` (gated HF model), needs `transformers`.

### 11. Auto-generate converter / plugin for a custom kernel  —  PRIORITY: MED  —  effort: MED
- Repo: `examples/dynamo/auto_generate_converters.py` (183 lines), `auto_generate_plugins.py` (224 lines), and AOT variant `aot_plugin.py` (248 lines).
- Demonstrates: TRT 10.7+ Quick-Deployable-Plugin (QDP) system — register a Triton kernel as a TRT plugin and auto-generate the Torch-TRT converter; JIT-plugin vs AOT-plugin (PTX embedded, no Python at runtime). Avoids graph breaks for unsupported ops.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/CustomKernelPlugin/` (or cross-link to `05-Plugin`).
- Self-contained: MOSTLY — requires `triton`; kernels are small/inline.

### 12. Overloading converters with a custom converter  —  PRIORITY: MED  —  effort: LOW-MED
- Repo: `examples/dynamo/converter_overloading.py` (212 lines).
- Demonstrates: registering a custom `@dynamo_tensorrt_converter` to override Torch-TRT's default lowering (example overrides `gelu`) — a clean, self-contained intro to the converter registry.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/ConverterOverloading/`.
- Self-contained: YES — no external deps beyond torch/torch_tensorrt.

### 13. Mutable Torch-TensorRT module (hot-swap weights / LoRA / SD)  —  PRIORITY: MED-LOW  —  effort: MED
- Repo: `examples/dynamo/mutable_torchtrt_module_example.py` (238 lines).
- Demonstrates: `MutableTorchTensorRTModule` for interacting with / modifying a compiled module in place (auto-refit on weight change), save/load, LoRA integration into a HF Stable Diffusion pipeline, and dynamic shapes.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/MutableModule/`.
- Self-contained: PARTIAL — ResNet18 part is; the LoRA/SD part downloads `diffusers` + HF weights.

### 14. Weight streaming for large models  —  PRIORITY: MED-LOW  —  effort: MED
- Repo: `examples/dynamo/weight_streaming_example.py` (206 lines).
- Demonstrates: `enable_weight_streaming` compile option + runtime budget context manager to run models larger than GPU memory (throughput vs streaming-overhead tradeoff).
- Target: `06-DLFrameworkTRT/Torch-TensorRT/WeightStreaming/`.
- Self-contained: PARTIAL — uses a Llama-2 model (gated HF, large download).

### 15. Tensor-parallel distributed inference  —  PRIORITY: LOW  —  effort: HIGH
- Repo: `examples/distributed_inference/tensor_parallel_simple_example.py` + `tensor_parallel_initialize_dist.py`, `data_parallel_gpt2.py`, `data_parallel_stable_diffusion.py`.
- Demonstrates: compiling tensor-parallel / data-parallel models with Torch-TRT across GPUs (NCCL, `torch.distributed`). Good for a `08-Advance` multi-device cross-link.
- Target: `06-DLFrameworkTRT/Torch-TensorRT/DistributedInference/` or `08-Advance`.
- Self-contained: NO — needs multiple GPUs + `torchrun`/MPI; heavier setup.

---

## Notes / recommendations
- Modernize the existing `Torch-TensorRT/main.py` to the strong-typing default (drop `enabled_precisions` / `truncate_long_and_double`, rely on `use_explicit_typing`) while adding new leaves — several candidates above (#7, #9) reinforce the TRT 11 strong-typing story.
- Best "quick wins" (self-contained, no gated downloads, high learnability): #1 engine caching, #2 dynamic shapes, #3 save/load, #4 torch.compile AOT-vs-JIT, #6 CUDA graphs, #9 autocast, #12 converter overloading.
- Fill the empty `ModelOptimizer` stub with #7 (FP8 ViT) and/or #8 (VGG16 PTQ) to cover the FP8/INT8-via-ModelOpt gap.
- LLM/SD compilation candidates: #10 (Gemma prefill/decode), #13 (SD + LoRA), plus `torch_compile_gpt2.py`, `torch_compile_stable_diffusion.py`, `torch_export_sam2.py`, `torch_export_flux_dev.py` (all HF/gated, heavier — lower priority but high demo value).
