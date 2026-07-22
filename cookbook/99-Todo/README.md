# 99-Todo

Todo list and research notes for the cookbook.

## Standing todo items

+ Usage of each flag in `trt.Builder`
+ 07-Tool/TritonServerDeploy
+ (No action) [TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)
+ (No action) [TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX)
+ (No action) [TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI)
+ (No action) [TensorRT-Incubator](https://github.com/NVIDIA/TensorRT-Incubator)
+ (archived) [trt-engine-explorer](https://gitlab-master.nvidia.com/wili/trt-engine-explorer)

---

## Content-intake research (2026-07-22)

Detailed, ranked candidate lists live in sibling files — this section is the index/outline.

- **[candidates-github.md](candidates-github.md)** — from the OSS repo `/work/trt/TensorRT-GitHub` (release/11.0).
- **[candidates-gitlab.md](candidates-gitlab.md)** — from the internal repo `/work/trt/TensorRT-GitLab` (superset; **vet before use**, includes an EXCLUDE list of internal/sensitive material).
- **[candidates-ecosystem.md](candidates-ecosystem.md)** — cross-repo index for the NVIDIA ecosystem repos cloned to `/work/trt/repos/` (TensorRT-LLM, Model-Optimizer, Torch-TensorRT, Tripy, RTX, tensorrtx); per-repo detail in the `candidates-eco-*.md` files.

### Top candidates from TensorRT OSS (GitHub)

| Priority | Candidate | Repo path | Target section |
| :------: | --------- | --------- | -------------- |
| High | demoDiffusion (SD / SDXL / SD3 / Flux / ControlNet / video) | `demo/Diffusion` | 03-Workflow (new end-to-end demo) — biggest gap |
| High | RMSNorm `IPluginV3` written in CuteDSL (CUTLASS Python DSL) | `samples/python/cute_dsl_plugin` | 05-Plugin |
| High | Multi-device attention (context parallelism, polygraphy shard) | `samples/python/attention_mdtrt` | 08-Advance/MultiDevice |
| High | Aliased-I/O in-place Scatter-add (`IPluginV3OneBuildV2`) | `samples/python/aliased_io_plugin` | 05-Plugin |
| High | ModelOpt AutoCast FP32→mixed for strong typing | `samples/python/strongly_type_autocast` | 03-Workflow / 04-Feature/StronglyTyped |
| Med-High | Weight-stripped engine build + refit | `samples/python/sample_weight_stripping` | 04-Feature |
| Medium | DDS detection with `IOutputAllocator` (Faster R-CNN) | `samples/python/dds_faster_rcnn` | 04-Feature/OutputAllocator |
| Medium | Editable timing cache (force a tactic) | `samples/sampleEditableTimingCache` | 04-Feature/TimingCache |
| Medium | DeBERTa transformer end-to-end (ONNX-GS + plugin) | `demo/DeBERTa` | 03-Workflow |
| Medium | Multi-backend Python plugin (Triton/Numba/CuPy/Torch) | `samples/python/python_plugin` | 05-Plugin/PythonPlugin |
| Medium | Refit ONNX via GS node replacement (BiDAF) | `samples/python/engine_refit_onnx_bidaf` | 04-Feature/Refit |
| Medium | Detectron2 Mask R-CNN, PackNet depth, cuDLA runtime | `samples/python/{detectron2,onnx_packnet}`, `samples/sampleCudla` | 03-Workflow / 05-Plugin / 04-Feature |

### Top candidates from TensorRT internal (GitLab) — vet before use

| Priority | Candidate | Repo path | Target section |
| :------: | --------- | --------- | -------------- |
| High | Empty-tensor use cases — fills existing TODO stubs | `tests/unitTests/tutEmptyTensorTests.cpp` | 04-Feature/EmptyTensor-TODO, 08-Advance/EmptyTensor-TODO |
| High | CuteDSL RMSNorm plugin (same as OSS) | `samples/python/cute_dsl_plugin` | 05-Plugin |
| High | ModelOpt AutoCast → strong typing (same as OSS) | `samples/python/strongly_type_autocast` | 03-Workflow / 04-Feature |
| Medium | Chained dynamic-reshape preprocessing engine | `samples/sampleDynamicReshape` | 08-Advance / 03-Workflow |
| Medium | Parallel multi-GPU engine deserialization timing | `samples_internal/deserializeTimer` | 08-Advance/MultiDevice |
| Medium | Concurrent multi-engine serving + per-task CUDA graph | `samples_internal/sampleMultiTasks` (Apache-2.0) | 08-Advance |
| Medium | `IPluginV2DynamicExt`→`IPluginV3` migration cheat-sheet | `samples/python/sample_plugin_v2_to_v3_migration` | 05-Plugin |
| Medium | FP8 QDQ scale/numeric semantics ("picky FP8") | `tests/unitTests/tutPickyFP8.cpp` | 02-API/Layer/QDQStructure |
| Medium | Shape-tensor I/O & host↔device shape transfers | `tests/unitTests/tutApiShapeTensorTests.cpp` | 02-API shape-input area |
| Low-Med | Layer-info JSON → multi-format graph viz (relicense first) | `tools/engine_visualizer` | 07-Tool |

> **Caveat (GitLab):** internal unit tests `#include` internal harness headers; the *layer APIs* are public but any intake must be **rewritten from scratch against the public API — copy the idea, never the file**. See the EXCLUDE list in `candidates-gitlab.md` before touching anything under `tools/`, `samples_internal/`, or the fusion/Myelin tests.

---

## NVIDIA official TensorRT ecosystem repos (GitHub) — reference

Surveyed 2026-07-22. Candidates for cross-linking / future intake:

| Repo | What it is | Cookbook relevance |
| ---- | ---------- | ------------------ |
| [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) | TensorRT OSS: parsers, plugins, samples, tools (Polygraphy, onnx-graphsurgeon, TREX) | Primary upstream — see `candidates-github.md` |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | LLM inference library (Python API, custom kernels, runtime) | LLM demos → 09-TensorRT-LLM |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | Quantization/sparsity/distillation/NAS/speculative-decoding (was "TensorRT Model Optimizer", **rebranded Dec 2025**) | Quantization workflows → 04-Feature, 03-Workflow |
| [NVIDIA/TensorRT-Incubator](https://github.com/NVIDIA/TensorRT-Incubator) | Experimental: **Tripy** (Pythonic TRT frontend, eager debug) + MLIR-TensorRT | New authoring frontend — potential 06-DLFrameworkTRT companion |
| [NVIDIA/TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX) | TensorRT for RTX (Windows/consumer GPUs) | Already tracked in 10-TensorRT-RTX |
| [NVIDIA/TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect) | Model-connect utilities | No action (tracked above) |
| [pytorch/TensorRT (Torch-TensorRT)](https://github.com/pytorch/TensorRT) | Torch→TRT compiler (docs at docs.pytorch.org/TensorRT) | Already covered in 06-DLFrameworkTRT |
| [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) | Network-API model zoo (active, TRT10, 7.8k★) | Reference link |
| [tensorflow/tensorrt (TF-TRT)](https://github.com/tensorflow/tensorrt) | TF-TRT integration | **Archived Feb 2025, read-only** — legacy reference only |

---

## Community / third-party TensorRT repos (non-NVIDIA-official) — survey 2026-07-22

Broader search beyond NVIDIA's own orgs. Grouped by what a cookbook could learn from them.

### Converters / frontends (technique intake)
| Repo | ★ | What it is | Cookbook relevance |
| ---- | -- | ---------- | ------------------ |
| [NVIDIA-AI-IOT/torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) | 4.9k | PyTorch→TRT converter via per-op network-API "converters" registry | Great teaching pattern for Network-API building & custom op converters → 06-DLFrameworkTRT / 02-API |
| [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) | 7.8k | Popular nets built purely with the **network definition API** (no ONNX), TRT 7–10 | Reference for hand-built networks → 02-API/Network, already linked |
| [open-mmlab/mmdeploy](https://github.com/open-mmlab/mmdeploy) | 3.1k | Multi-backend deploy (TRT/ORT/ncnn…), custom TRT plugins for MM models | Plugin + deploy patterns → 05-Plugin / 07-Tool |

### End-to-end deployment (demo intake)
| Repo | ★ | What it is | Cookbook relevance |
| ---- | -- | ---------- | ------------------ |
| [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference) | 8.9k | Jetson TRT deploy (detection/segmentation/pose), C++ & Python | Embedded C++ runtime demos → 01-SimpleDemo / edge note |
| [laugh12321/TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) | — | YOLO deploy toolkit: CUDA pre/post kernels + TRT **plugins for NMS/post-proc**, C++ & Python | Efficient-NMS plugin + full detection pipeline → 05-Plugin / 03-Workflow |
| [Linaom1214/TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) | — | YOLOv5–v12 export + NMS-plugin inference | Detection workflow reference |
| [collabora/WhisperLive](https://github.com/collabora/WhisperLive) | 4.1k | Near-real-time Whisper (TensorRT-LLM backend) | Streaming ASR demo idea → 09-TensorRT-LLM |
| [stochasticai/x-stable-diffusion](https://github.com/stochasticai/x-stable-diffusion) | — | SD latency shootout: TRT vs AITemplate/nvFuser/FlashAttention | Benchmarking framing for a diffusion demo |
| [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT) | — | TRT extension for A1111 WebUI (SD1.5/2.1/SDXL/LCM) | Dynamic-shape engine mgmt for diffusion |

> Not TRT-specific (general inference engines, lower intake value): Tencent/TNN, OAID/Tengine, xlite-dev/lite.ai.toolkit, Megvii YOLOX, ultralytics/yolov5. Useful as ecosystem context only.

**Takeaway for the cookbook:** the community's strongest, non-duplicated ideas are (1) **torch2trt's per-op converter registry** as a Network-API teaching device, (2) **YOLO NMS/post-processing plugins + fused CUDA pre/post** as a realistic detection pipeline, and (3) **streaming ASR / diffusion** end-to-end demos — all reinforcing the same top gaps found in the NVIDIA repos (diffusion/LLM demos, richer plugins).

---

## README "Useful Links" health check (2026-07-22)

Checked links in `cookbook/README.outline.txt` "## Useful Links". Fixes applied to that file:

| Link | Status | Action |
| ---- | ------ | ------ |
| Operators Document `.../tensorrt/operators/docs/` | **404 (dead)** | → `https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/index.html` |
| Torch-TensorRT `https://pytorch.org/TensorRT/` | 301 redirect | → `https://docs.pytorch.org/TensorRT/` |
| TensorRT Download `.../nvidia-tensorrt-download` | login-gated redirect | → `https://developer.nvidia.com/tensorrt/download` |
| TF-TRT `github.com/tensorflow/tensorrt` | live but **archived Feb 2025** | annotated "(archived, read-only)" |
| C++ API, Python API, Polygraphy API + doc tree, TRTOSS repo, repo git address, tensorrtx, TREX (`tools/experimental/trt-engine-explorer`), ONNX/CUDA links | OK | none |
