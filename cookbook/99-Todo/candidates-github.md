# Candidates from TensorRT-GitHub (OSS)

Source repo: `/work/trt/TensorRT-GitHub` (TensorRT OSS, release/11.0).
Cross-checked against the existing cookbook so proposals below are **not already well covered**.
Priority = value-to-a-learner vs. effort to port into the cookbook's `main.py` + `TRTWrapper` style.

---

## Summary of top picks

| # | Candidate | Repo path | Target section | Priority |
|---|-----------|-----------|----------------|----------|
| 1 | demoDiffusion (SD / SDXL / SD3 / Flux / ControlNet / video) | `demo/Diffusion` | 03-Workflow (new end-to-end demo) | High |
| 2 | RMSNorm PluginV3 written in CuteDSL | `samples/python/cute_dsl_plugin` | 05-Plugin | High |
| 3 | Multi-device attention (context parallelism) | `samples/python/attention_mdtrt` | 08-Advance/MultiDevice | High |
| 4 | Aliased-I/O in-place Scatter-add plugin (`IPluginV3OneBuildV2`) | `samples/python/aliased_io_plugin` | 05-Plugin | High |
| 5 | ModelOpt AutoCast FP32→mixed for strong typing | `samples/python/strongly_type_autocast` | 03-Workflow / 04-Feature/StronglyTyped | High |
| 6 | Weight-stripped engine build + refit | `samples/python/sample_weight_stripping` | 04-Feature | Med-High |
| 7 | DDS object detection with `IOutputAllocator` (Faster R-CNN) | `samples/python/dds_faster_rcnn` | 04-Feature/OutputAllocator | Medium |
| 8 | Editable timing cache (force a tactic) | `samples/sampleEditableTimingCache` | 04-Feature/TimingCache | Medium |
| 9 | DeBERTa transformer end-to-end (ONNX-GS modify + TRT) | `demo/DeBERTa` | 03-Workflow | Medium |
| 10 | Multi-backend Python plugin (Triton/Numba/CuPy/Torch) | `samples/python/python_plugin` | 05-Plugin/PythonPlugin | Medium |
| 11 | Refit ONNX via GS node replacement (BiDAF) | `samples/python/engine_refit_onnx_bidaf` | 04-Feature/Refit | Medium |
| 12 | Detectron2 Mask R-CNN R50-FPN conversion | `samples/python/detectron2` | 03-Workflow | Medium |
| 13 | PackNet custom-layer + ONNX-GS depth estimation | `samples/python/onnx_packnet` | 05-Plugin / 03-Workflow | Medium |
| 14 | cuDLA API to run a TRT engine | `samples/sampleCudla` | 04-Feature/DLAStandalone | Medium |
| 15 | EfficientDet TensorRT notebook | `demo/EfficientDet/notebooks` | 03-Workflow | Low-Med |
| 16 | Semantic segmentation C++ runtime tutorial | `quickstart/SemanticSegmentation` | 01-SimpleDemo / 06 | Low-Med |

---

## 03-Workflow (framework → ONNX → TRT, end-to-end demos)

- **demoDiffusion** — `demo/Diffusion` — *High.*
  The single most valuable gap. Full text-to-image / image-to-image / text-to-video pipelines accelerated with TensorRT: Stable Diffusion 1.5/2.1, SDXL, SD3/SD3.5, **Flux**, Stable Cascade, ControlNet, Cosmos, WAN video (`demo_txt2img*.py`, `demo_txt2img_flux.py`, `demo_controlnet*.py`, `demo_img2vid.py`, `demo_txt2vid_wan.py`). Shows multi-engine pipelines, FP8/INT8 calibration (`calibration_data/`), and the `demo_diffusion/` engine-management library. Cookbook currently has **no diffusion/LLM end-to-end demo**. Could seed a new `03-Workflow/Diffusion-TensorRT` (or a top-level demo section). High effort but high payoff — port a single trimmed SDXL or SD1.5 txt2img path first.

- **DeBERTa** — `demo/DeBERTa` — *Medium.*
  Transformer (BERT-family) end-to-end: PyTorch→ONNX (`deberta_pytorch2onnx.py`), ONNX-GS graph surgery to insert an optimized disentangled-attention plugin (`deberta_onnx_modify.py`), then TRT vs. ORT inference comparison. Good companion to the (deprecated) `plugin/disentangledAttentionPlugin`. Cookbook has no transformer workflow demo.

- **strongly_type_autocast** — `samples/python/strongly_type_autocast` — *High.*
  Uses **ModelOpt AutoCast** to convert an FP32 ONNX model to mixed FP32/FP16, then builds a strongly-typed engine and verifies numerics. Directly relevant to the cookbook's TRT-11 strong-typing focus and the weak→strong migration story. Cookbook's `pyTorch-ModelOptimizer-ONNX-TensorRT` covers PTQ/QAT but not the AutoCast mixed-precision conversion path — worth a dedicated `04-Feature/StronglyTyped` or workflow example.

- **detectron2** — `samples/python/detectron2` — *Medium.* Mask R-CNN R50-FPN 3x conversion/run/validate; recently migrated to strongly-typed APIs. Realistic instance-segmentation pipeline.

- **onnx_packnet** — `samples/python/onnx_packnet` — *Medium.* Self-supervised monocular depth net; demonstrates custom-layer registration via `REGISTER_TENSORRT_PLUGIN` plus ONNX-GS subgraph rewrites (GroupNorm/upsample/pad cleanup). Good bridge between 05-Plugin and 07-Tool/OnnxGraphSurgeon.

- **EfficientDet notebook** — `demo/EfficientDet/notebooks` — *Low-Med.* Object-detection notebook; lower priority (notebook-only, older TRT8 baseline).

## 05-Plugin

- **cute_dsl_plugin (RMSNorm)** — `samples/python/cute_dsl_plugin` — *High.*
  Authors an `IPluginV3` whose `enqueue()` launches a **CuteDSL (CUTLASS Python DSL)** RMSNorm kernel. Novel and LLM-relevant: zero-copy buffer sharing (`cupy.UnownedMemory`→`torch.as_tensor`→`cute.runtime.from_dlpack`), launching on the TRT-provided CUDA stream, FP32-accumulate/FP16-store. No cookbook analog for DSL-authored kernels. Distinct from existing `PythonPlugin`/`QuickDeployablePlugin`.

- **aliased_io_plugin** — `samples/python/aliased_io_plugin` — *High.*
  In-place Scatter-add using the **`IPluginV3OneBuildV2` aliased-I/O** capability (`get_aliased_input`), motivated by GNN neighborhood aggregation. Cookbook's `05-Plugin/InPlacePlugin` is a CUDA `AddScalar` and does not demonstrate the V2 aliased-input API — this is the canonical example of it.

- **python_plugin (circular padding, multi-backend)** — `samples/python/python_plugin` — *Medium.*
  Same op implemented across **Triton, Numba, CuPy, CUDA-Python, PyTorch, and C++** backends, plus a multi-tactic variant and an `INetworkDefinition`-vs-ONNX variant. Cookbook's `PythonPlugin` is narrower; the backend matrix is a great teaching artifact. Could enrich `05-Plugin/PythonPlugin` rather than a new dir.

- **onnx_packnet** (also listed above) — plugin registration angle for 05-Plugin.

> Skipped as near-duplicates of existing cookbook dirs: `samples/python/quickly_deployable_plugins` (≈ `QuickDeployablePlugin`), `samples/python/onnx_custom_plugin` (≈ `ONNXParserWithPlugin`), `samples/python/non_zero_plugin` (≈ `DataDependentShape`/Layer/NonZero). Most C++ `plugin/*` dirs (bertQKV, efficientNMS, groupNorm, etc.) are **DEPRECATED since 10.12–10.15** — do not port.

## 04-Feature

- **sample_weight_stripping** — `samples/python/sample_weight_stripping` — *Med-High.*
  Build a **weight-stripped** engine (smaller artifact), then refit from the original ONNX via the parser refitter with no accuracy/perf loss (ResNet50). Cookbook `Refit` touches stripping only in passing; a dedicated weight-stripping example is distinct from `WeightStreaming` and worth its own dir.

- **dds_faster_rcnn** — `samples/python/dds_faster_rcnn` — *Medium.*
  End-to-end **data-dependent-shape** output handling via a custom `IOutputAllocator` (`reallocate_output_async`/`notify_shape`) on Faster R-CNN. Cookbook has `OutputAllocator` and `TRTWrapperDDS` primitives but no realistic detection use-case tying them together.

- **sampleEditableTimingCache** — `samples/sampleEditableTimingCache` — *Medium.*
  Force a specific (non-fastest) **tactic** by editing the timing cache from the profiling log (MatMul→Softmax→MatMul). Cookbook `TimingCache` covers reuse but not the editing API / deterministic-tactic-selection workflow.

- **engine_refit_onnx_bidaf** — `samples/python/engine_refit_onnx_bidaf` — *Medium.*
  Refit workflow that first uses ONNX-GS to replace unsupported nodes (HardMax/Compress), builds a refittable engine, then refits fake→real weights. Complements `04-Feature/Refit` with a parser-based refit + GS story.

- **sampleCudla** — `samples/sampleCudla` — *Medium.* Run a TRT engine through the **cuDLA** API (hybrid/standalone DLA). Cookbook `DLAStandalone` exists but cuDLA runtime path is distinct (Orin/embedded relevance).

- **stream_writer** — `samples/python/stream_writer` — *Low.* `IStreamWriter` + `build_serialized_network_to_stream()`. Likely already partially shown in `02-API/Builder`; low incremental value.

## 08-Advance

- **attention_mdtrt** — `samples/python/attention_mdtrt` — *High.*
  Multi-GPU self-attention via **context parallelism (CP)**: `polygraphy multi-device shard` inserts DistCollective (ReduceScatter/AllGather) ops from a `hint.json`, run under MPI+NCCL, with a single-GPU baseline. Cookbook has `Layer/DistCollective` and `08-Advance/MultiDevice` primitives but not this end-to-end sharding workflow. Strong, modern, LLM-relevant.

## 01-SimpleDemo / quickstart

- **quickstart/SemanticSegmentation** — `quickstart/SemanticSegmentation` — *Low-Med.* Clean FCN-ResNet101 export + **C++ runtime** tutorial (`tutorial-runtime.cpp`) and matching notebook. Good minimal C++ deploy example aligned with the cookbook's C++ runtime skill; overlaps partly with `01-SimpleDemo/TensorRT-10`.

> Skipped near-duplicates in quickstart/samples: `quickstart/deploy_to_triton` (≈ `07-Tool/TritonServerDeploy`), `sampleIOFormats` (≈ `04-Feature/DataFormat`), `sampleNamedDimensions` (≈ `LabeledDimension`), `sampleProgressMonitor` (≈ `04-Feature/ProgressMonitor`), `sampleDistCollective` (≈ Layer/DistCollective), `network_api_pytorch_mnist`. Also `tools/pytorch-quantization` and `tools/tensorflow-quantization` are legacy (superseded by ModelOpt) — low priority.

---

## Notes on effort

- **Highest ROI, moderate effort:** #2 cute_dsl_plugin, #3 attention_mdtrt, #4 aliased_io_plugin, #5 strongly_type_autocast — self-contained Python, map cleanly onto existing sections.
- **High value, high effort:** #1 demoDiffusion — port one trimmed pipeline (SD1.5 or SDXL txt2img) rather than the whole family.
- **Quick wins:** #6 weight stripping, #8 editable timing cache, #11 BiDAF refit — small scripts filling clear feature gaps.
