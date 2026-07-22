# Candidates from the NVIDIA TensorRT ecosystem repos

Repos cloned to `/work/trt/repos/` (shallow) and analyzed 2026-07-22. Per-repo detail lives in the
sibling `candidates-eco-*.md` files; this is the consolidated, cross-repo index and ranking.

| Repo (`/work/trt/repos/…`) | Detail file | Verdict |
| -------------------------- | ----------- | ------- |
| NVIDIA/TensorRT-LLM | [candidates-eco-tensorrt-llm.md](candidates-eco-tensorrt-llm.md) | Fills the empty `09-TensorRT-LLM`; repo has moved to the high-level PyTorch-backend `LLM` API |
| NVIDIA/Model-Optimizer | [candidates-eco-model-optimizer.md](candidates-eco-model-optimizer.md) | Sanctioned quantization path post weak-typing removal; target formats the existing example lacks |
| pytorch/Torch-TensorRT | [candidates-eco-torch-tensorrt.md](candidates-eco-torch-tensorrt.md) | Existing `06-DLFrameworkTRT` coverage is thin; all picks net-new |
| NVIDIA/TensorRT-Incubator (Tripy) | [candidates-eco-tripy.md](candidates-eco-tripy.md) | Pre-1.0 (v0.1.7) — small labeled "experimental frontend" intro only |
| NVIDIA/TensorRT-RTX + wang-xinyu/tensorrtx | [candidates-eco-rtx-tensorrtx.md](candidates-eco-rtx-tensorrtx.md) | RTX: combined weightless-refit workflow; tensorrtx: network-API build patterns |

> TensorRT (OSS) itself → see `candidates-github.md`. TensorRT-Model-Connect clone failed (auth-gated / private) — it is a "no action" repo. tensorflow/tensorrt is archived (skipped).

---

## Cross-repo top picks (highest ROI first)

| # | Candidate | Repo path | Target section | Priority |
|---|-----------|-----------|----------------|----------|
| 1 | **LLM API quickstart** — ~30-line `LLM.generate()` "hello world" | `TensorRT-LLM/examples/llm-api/quickstart_example.py` | 09-TensorRT-LLM (seed the empty stub) | High |
| 2 | **LLM quantization** — FP8 / INT4-AWQ via ModelOpt, or pass a pre-quantized HF checkpoint | `TensorRT-LLM/examples/quantization/quantize.py` | 09-TensorRT-LLM | High |
| 3 | **ONNX PTQ through a custom C++ TRT plugin** — self-contained, no downloads | `Model-Optimizer/examples/onnx_ptq/custom_op_plugin/` | 05-Plugin / 04-Feature | High |
| 4 | **PyTorch→ONNX in NVFP4 / MXFP8 / INT4-AWQ → strongly-typed TRT** | `Model-Optimizer/examples/torch_onnx/torch_quant_to_onnx.py` | 04-Feature / 03-Workflow | High |
| 5 | **Torch-TRT FP8 PTQ (Dynamo)** and **INT8 VGG16 PTQ** — fill the empty `ModelOptimizer` stub | `Torch-TensorRT/examples/dynamo/{quantize_vit_fp8.py,vgg16_ptq.py}` | 06-DLFrameworkTRT | High |
| 6 | **RTX weightless-refit deploy workflow** — STRIP_PLAN+REFIT → Refitter reweight → runtime cache + CUDA graph | `TensorRT-RTX/samples/apiUsage/python/api_usage.py` | 10-TensorRT-RTX (new leaf) | High |
| 7 | **tensorrtx network-API build** — LeNet from `.wts` (incl. matmul-replaces-FullyConnected TRT-10 lesson) + `.wts`/`gen_wts.py` weight convention | `tensorrtx/lenet/`, `tensorrtx/*/gen_wts.py` | 02-API/Network | High |
| 8 | **LLM async + streaming + sampling** — short TinyLlama lessons | `TensorRT-LLM/examples/llm-api/llm_inference_async*.py`, `llm_sampling.py` | 09-TensorRT-LLM | Med-High |
| 9 | **Torch-TRT quick wins** — engine caching, dynamic shapes, save/load compiled program, AOT-vs-JIT, CUDA graphs, autocast | `Torch-TensorRT/examples/dynamo/*` | 06-DLFrameworkTRT | Medium |
| 10 | **Tripy intro** — tiny Conv+ReLU `tp.Module`, run eager → `tp.compile` to TRT | `TensorRT-Incubator/tripy/README.md`, `docs/pre0_user_guides/00-introduction-to-tripy.md` | 06-DLFrameworkTRT (experimental) | Medium |
| 11 | **LLM speculative decoding (NGram)** and **KV-cache offloading** | `TensorRT-LLM/examples/llm-api/llm_speculative_decoding.py`, `llm_kv_cache_offloading.py` | 09-TensorRT-LLM | Medium |
| 12 | **Multi-GPU TP/PP/EP LLM** (needs ≥2 GPUs) | `TensorRT-LLM/examples/llm-api/llm_inference_distributed.py` | 09-TensorRT-LLM / 08-Advance | Medium |
| 13 | **ONNX PTQ INT4-AWQ / entropy / RTN + Q/DQ autotune** | `Model-Optimizer/examples/onnx_ptq/`, `.../autotune/` | 04-Feature | Medium |
| 14 | **trtllm-serve + OpenAI-compatible client** | `TensorRT-LLM/docs/source/quick-start-guide.md`, `examples/serve/openai_*_client.py` | 09-TensorRT-LLM / 07-Tool | Medium |
| 15 | **BatchNorm folded into IScaleLayer** (TRT has no native BN — common question) | `tensorrtx/resnet/resnet34.cpp` (`addBatchNorm2d`) | 02-API/Layer/Scale | Med |
| 16 | **FastNAS pruning (CIFAR ResNet)** — pruning absent from cookbook, self-contained | `Model-Optimizer/examples/pruning/cifar_resnet.ipynb` | 04-Feature (new) | Med |
| 17 | **Torch-TRT custom Triton QDP plugin (JIT/AOT) + converter overloading** | `Torch-TensorRT/examples/dynamo/{*triton*,cuda_kernel_op.py}` | 05-Plugin / 06-DLFrameworkTRT | Med |
| 18 | **Diffusion quantization SDXL/FLUX (INT8/FP8/FP4)→ONNX→TRT** (multi-GB, gated) | `Model-Optimizer/examples/diffusers/quantization/` | 03-Workflow (pairs with demoDiffusion) | Low-Med |

---

## Themes & recommendations

1. **Biggest single gap = LLM (`09-TensorRT-LLM`).** Start with #1 (quickstart) + #2 (quantization) + #8
   (async/streaming). Note: this TensorRT-LLM checkout dropped the old `trtllm-build` engine workflow and
   per-model scripts (and Whisper) in favor of the high-level `LLM` API — cookbook lessons should target
   that API. All LLM examples need a full GPU TensorRT-LLM install, so ship **script + expected-output log**
   with CI execution gated off.

2. **Quantization is now ModelOpt's job** (weak-typing INT8 calibration was removed in TRT 11). The existing
   `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` covers AutoCast-FP16 / torch-INT8-QAT / FP8-ONNX-PTQ on
   MNIST; extend toward **NVFP4 / MXFP8 / INT4-AWQ / entropy calib / custom-plugin PTQ** (#3, #4, #13).

3. **Torch-TensorRT coverage is thin** — the empty `06-DLFrameworkTRT/ModelOptimizer` stub should be filled
   with FP8 ViT / INT8 VGG16 PTQ (#5), and `main.py` modernized to `use_explicit_typing` (strong typing).

4. **Network-API teaching** — tensorrtx (#7, #15) is the best source for "build a net without ONNX" and the
   `.wts` weight-transfer convention, plus TRT-10/11 migration lessons (matmul-replaces-FullyConnected,
   BN-as-Scale). Import *patterns*, not the whole model zoo. Its YOLO plugin uses deprecated
   `IPluginV2IOExt` — cite the decode-in-plugin idea, don't port the code (cookbook already teaches
   `IPluginV3`).

5. **Tripy (#10)** is worth a *small, clearly-labeled experimental* intro only — pre-1.0, API still moving.

6. **RTX (#6)** — the one high-value addition is the end-to-end **weightless AOT + refit + runtime-cache +
   CUDA-graph** workflow; the individual RTX APIs are already covered in `10-TensorRT-RTX/01..07`.
