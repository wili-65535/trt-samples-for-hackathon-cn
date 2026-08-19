# 99-Todo

Single file. Everything that was spread over `candidates-*.md` and `trex*.md` is merged here
(2026-08-28); the per-repo files are gone.

+ **§1 is the pick list** — every still-open candidate, one row each, grouped by source. Cross out
  what is not wanted.
+ §2 records what was **decided against**, so it is not re-proposed.
+ §3 is the GitLab **EXCLUDE** list. Read it before importing anything internal.
+ §4 is reference material (ecosystem / community repos, finished migrations).
+ [`chatglm-6b.md`](chatglm-6b.md) is the one sibling file left: an intake analysis of the
  hand-written ChatGLM-6B PyTorch→ONNX→TensorRT pipeline at `/work/chatglm-6b`, kept separate
  because it is a source reading rather than a pick list. Its conclusions feed §1.1 as C1–C12.

Sources: `/work/trt/TensorRT-GitHub` (OSS, `VERSION` 11.2.1.2), `/work/trt/TensorRT-GitLab`
(internal), `/work/trt/repos/{Model-Optimizer,Torch-TensorRT,TensorRT-RTX,tensorrtx,TensorRT-Incubator,TensorRT-LLM}`.
Installed runtime is TensorRT 11.1.0.106.

---

## 1. Open candidates

### 1.1 Standing items (not from any repo survey)

| # | Item | Note |
| - | ---- | ---- |
| S1 | Usage of each flag in `trt.Builder` | |
| S2 | Release the TensorRT objects in the three C++ examples that still leak them | `01-SimpleDemo/TensorRT-8.0`, `-8.6`, `08-Advance/Safety`. Left alone because they **do not compile against TensorRT 11** (`kEXPLICIT_BATCH` removed; `nvinfer1::safe::ICudaEngine` gone), so a fix cannot be verified. Every other C++ example is clean under `compute-sanitizer --leak-check full` |
| S3 | `IRefitterObserver` / `IParserRefitter::setRefitObserver` → `04-Feature/Refit` | The **only** public API added in 11.0 → 11.2 (`NvOnnxParser.h`), C++ only, no Python binding. Needs the 11.2 upgrade first. Header symbol diff otherwise shows **530 vs 530 symbols, empty diff**, so that upgrade is non-breaking |
| S4 | Multi-device attention / context parallelism → `08-Advance/MultiDevice` | `samples/python/attention_mdtrt`; needs >1 GPU + MPI + NCCL |
| S6 | **ChatGLM-6B pipeline intake — DONE 2026-08-28.** (a) `07-Tool/OnnxGraphSurgeon/11..13` (bisect an accuracy bug with `mark_graph_output`, host-side constant tables, splicing two ONNX files + an ArgMax tail); (b) `03-Workflow/pyTorch-KVCache-ONNX-TensorRT` on gpt2. See [`chatglm-6b.md`](chatglm-6b.md) for the source reading. ~~a new `03-Workflow` leaf building a **toy** decoder-only model end to end — packed KV-cache I/O (56 in + 56 out → 1 + 1), input and output cache bound to the same address, sampling moved inside the engine (returns a token id, not a 130528-wide logit vector), export driven from the real `forward`, prefill/decode shape regimes. **The cookbook has no autoregressive workflow at all.** ~~ **Both landed**; the `np.ascontiguousarray` trap turned out to be already documented in `10-advanceAPI.py`, and `mark_graph_output` was already in `tensorrt_cookbook` but had no example. | Both self-contained once re-expressed on a tiny model; the TRT-8.6 build recipe in the original is illegal under TRT 11 (`EXPLICIT_BATCH`, `BuilderFlag.FP16`, `OBEY_PRECISION_CONSTRAINTS`, `set_output_type`) so the build stage is a rewrite, not a port |
| S5 | TREx sub-repo endgame | The migration is finished (§4.3); still to decide whether to wipe the remaining docs/`.git` and commit the sub-repo deletion |

### 1.2 TensorRT OSS — `/work/trt/TensorRT-GitHub`

| # | Candidate | Repo path | Target | Prio |
| - | --------- | --------- | ------ | ---- |
| G1 | **DDS detection with `IOutputAllocator`** — `reallocate_output_async` / `notify_shape` on Faster R-CNN. Cookbook has the primitives (`OutputAllocator`, `TRTWrapperDDS`) but no realistic detection use-case tying them together | `samples/python/dds_faster_rcnn` | 04-Feature/OutputAllocator | Med |
| G2 | **DeBERTa end-to-end** — PyTorch→ONNX, ONNX-GS surgery to insert a disentangled-attention plugin, TRT vs ORT. Cookbook has no transformer workflow demo | `demo/DeBERTa` | 03-Workflow | Med |
| G3 | **Multi-backend Python plugin** — one op across Triton / Numba / CuPy / CUDA-Python / PyTorch / C++, plus multi-tactic and `INetworkDefinition`-vs-ONNX variants. Could enrich `05-Plugin/PythonPlugin` in place | `samples/python/python_plugin` | 05-Plugin/PythonPlugin | Med |
| G4 | **Refit ONNX via GS node replacement (BiDAF)** — replace unsupported nodes (HardMax/Compress), build refittable, refit fake→real weights | `samples/python/engine_refit_onnx_bidaf` | 04-Feature/Refit | Med |
| G5 | **Detectron2 Mask R-CNN R50-FPN** convert/run/validate, already strongly typed | `samples/python/detectron2` | 03-Workflow | Med |
| G6 | **PackNet** — `REGISTER_TENSORRT_PLUGIN` + ONNX-GS subgraph rewrites; bridges 05-Plugin and 07-Tool/OnnxGraphSurgeon | `samples/python/onnx_packnet` | 05-Plugin / 03-Workflow | Med |
| G7 | **`fftPlugin`** (new in 11.2) — cuFFT C2C/R2C/C2R backing the ONNX `DFT` op. Cookbook has **zero** FFT content | `plugin/fftPlugin` | 05-Plugin / 02-API/ONNXParser | Med |
| G8 | **Transformer-Engine ONNX → opset19 Q/DQ converter** — required step on the FP8-LLM path; cookbook has none of it. Cost is dominated by producing a Transformer-Engine-exported model | `scripts/convert_te_onnx_to_trt_onnx.py` | 03-Workflow / 07-Tool | Med |
| G9 | **Declarative data download** — `download.yml` manifest + MD5 `verifyChecksum` + retries; more robust than `00-Data`'s ad-hoc downloading | `samples/python/downloader.py` | 00-Data | Med |
| G10 | **Percentile latency / layer-profile aggregation (C++)** — only interesting for a C++ consumer; `07-Tool/trtexec/parse_export_json.py` already does percentiles in Python | `samples/common/sampleReporting.*` | 07-Tool | Low-Med |
| G11 | EfficientDet notebook (TRT8-era); semantic-segmentation **C++ runtime** tutorial (`tutorial-runtime.cpp`) | `demo/EfficientDet/notebooks`, `quickstart/SemanticSegmentation` | 03-Workflow / 01-SimpleDemo | Low-Med |
| G12 | `topkLastDimPlugin` ("Faster Air Top-K", new in 11.1) as a perf footnote; assorted C++ helpers (`bfloat16.*`/`half.h` vs the cookbook's hand-rolled `numpy_fp32_to_bf16`, `streamReader.h`, `ErrorRecorder.h`, `safeCudaAllocator.h`); `plugin_utils.py::CudaCtxManager`; `common_runtime.py::ArrayWithOwner` ownership trick; cross-compilation Dockerfiles (`ubuntu-26.04`, `ubuntu-cross-aarch64`) | `plugin/topkLastDimPlugin`, `samples/common/`, `docker/` | 02-API/Layer/TopK, include/cookbookHelper, docs | Low |

**Effort notes.** Quick wins: G4, G7, G12. Moderate but good payoff: G1, G3, G9. Dominated by
"first get the model": G2, G5, G6, G8.

**Deliberately skipped as near-duplicates** — `quickly_deployable_plugins` (≈ `QuickDeployablePlugin`),
`onnx_custom_plugin` (≈ `ONNXParserWithPlugin`), `non_zero_plugin` (≈ `DataDependentShape`),
`deploy_to_triton` (≈ `07-Tool/TritonServerDeploy`), `sampleIOFormats` (≈ `04-Feature/DataFormat`),
`sampleNamedDimensions` (≈ `LabeledDimension`), `sampleProgressMonitor`, `sampleDistCollective`,
`network_api_pytorch_mnist`, `samples/python/refactored/`, `stream_writer` (≈ `02-API/Builder` +
`CookbookStreamWriter`), `common.py::setup_timing_cache` (≈ `04-Feature/TimingCache`).
Most C++ `plugin/*` dirs (bertQKV, efficientNMS, groupNorm …) are **deprecated since 10.12–10.15**;
11.0 additionally *removed* `batchTile`, `clip`, `coordConvAC`, `cropAndResize`, `gelu`, `leakyRelu`,
`normalize`, `singleStepLSTM`, `specialSlice`, `split`, `nms`, `proposal`. `tools/pytorch-quantization`
and `tools/tensorflow-quantization` are legacy (superseded by ModelOpt).

### 1.3 TensorRT internal (GitLab) — **vet before use**

The GitLab repo **no longer tracks `samples/`** (`oss_components.yml` lists `samples/**/*` as
`detracked`), and `plugin/` + `tools/trtexecCommon/` are identical to GitHub's. The GitLab-exclusive
surface is `tests/unitTests/`, `tools/`, `samples_internal/`, `testing/`, `scripts/`,
`plugin_internal/`, `projects/`, `documentation/`.

> Internal unit tests `#include` internal harness headers. The *layer APIs* are public, but any
> intake must be **rewritten from scratch against the public API — copy the idea, never the file.**

| # | Candidate | Repo path | Target | Prio |
| - | --------- | --------- | ------ | ---- |
| L1 | **Polygraphy / trtexec debugging workflows** — `accuracy_scatterplot.py` (plot every output value vs its absolute error against a golden run: shows whether FP16 error is spread or driven by a few outliers, far better than one max-abs-diff), `polygraphy_autoreduce.py` (materialise one directory per `debug reduce` iteration so the reduction is auditable), `polygraphy_inputs_to_trtexec.py` (`--save-inputs` JSON → per-tensor `.bin` + the matching `--loadInputs=`), `accuracy_analyzer.py`, `dump_io_data.py`. **All five are `LicenseRef-NvidiaProprietary` — reimplement, do not copy** | `scripts/` | 07-Tool/Polygraphy, 07-Tool/trtexec | Med |
| L2 | Parallel multi-GPU engine deserialization (`std::async` vs sequential) — **needs >1 GPU** | `samples_internal/deserializeTimer` | 08-Advance/MultiDevice | Med |
| L3 | Concurrent multi-engine serving + per-task CUDA graph + device pinning. Cookbook covers MultiContext / MultiStream / MultiDevice / CudaGraph *separately*; the combined "serve N engines at once" is missing. Heavy trtexec-internal coupling → full rewrite | `samples_internal/sampleMultiTasks` (Apache-2.0) | 08-Advance | Med |
| L4 | **"Picky FP8"** — an FP8 Q/DQ/MatMul network deliberately engineered to be numerically sensitive to wrong types/scales (alpha = 1+2⁻¹¹ tricks) | `tests/unitTests/tutPickyFP8.cpp` | 02-API/Layer/QDQStructure | Med |
| L5 | Shape-tensor I/O and host↔device shape transfers — INT32 vs INT64 shape tensors, 0-D/N-D, transfer corner cases; thinly covered today | `tutApiShapeTensorTests.cpp`, `tutShapeH2DTests.cpp`, `tutShuffleShapeTests.cpp`, `tutApiShapeReduceTensorTests.cpp` | 02-API shape-input area | Med |
| L6 | Corner cases: `IEngineInspector` levels, runtime allocation strategies, systematic DDS, several plugins in one graph / many-I/O plugins, `INetworkDefinition` API boundaries, UINT8 / BOOL / NaN semantics, in-place shuffle (and when multiple consumers block it), high-fan-out loops, DynamicQuantize + whole-network QAT | `tut{Inspect,RuntimeAllocation,DataDependent,MultiPlugins,PluginMultipleIO,NetworkAPI,UInt8,Bool,IsNaN,InPlaceShuffle,MultiConsumerInPlaceShuffle,HighDegreeLoop,DynamicQuantize,QATNetwork}Tests.cpp` | 04-Feature, 02-API/Layer | Med |
| L7 | Engine layer-info JSON → Graphviz / Dagre HTML / yEd GraphML (one JSON, three formats). **Scrub the proprietary SPDX header, internal email and `gitlab-master.nvidia.com` links first** | `tools/engine_visualizer/plotEngine.py` | 07-Tool | Low-Med |
| L8 | Per-layer depth: dynamically-quantized conv weights; MXFP / block QDQ with E8M0 scales; **structured-sparsity conv + refit sparse-vs-dense** (a rare genuine refit demo: refit sparse OK, dense→error); GridSample 5-D/CUBIC/FILL/CLAMP/REFLECT; OneHot negative axis; ReverseSequence axis variants; Assertion via the shape machine; Loop/If corner cases (nested, kWHILE/kCOUNT, reverse iterator, lazy-vs-eager) | `tutApiQuantDynamicWeightTest.cpp`, `tutBlockQDQ.cpp`, `tutPackedQDQ.cpp`, `tutApiSparseConvTests.cpp`, `tutGridSampleTests.cpp`, `tutOneHotTests.cpp`, `tutReverseSequenceTests.cpp`, `tutAssertionLayerTests.cpp`, `tutLoopTests.cpp`, `tutIfConditionalTests.cpp` | 02-API/Layer/*, 04-Feature/Sparsity | Low |
| L9 | Golden-output accuracy checker with per-output mixed metrics (cosine for TopK/ArgMax outputs, elementwise otherwise) — overlaps Polygraphy; relicense + drop internal `common/harness*` deps | `tools/infer_ref_check/` | 07-Tool | Low |
| L10 | **Operator reference docs as a QA source** — the authoritative per-operator text to cross-check `02-API/Layer/*/README.md` against. Not content to import. Low effort, good accuracy payoff | `documentation/operators/*.rst` | 02-API/Layer (QA) | Low |
| L11 | Resource-probing *concepts* only (code is proprietary): allocate successive chunks until failure to find the real usable VRAM ceiling (`cudaGetMemInfo` is unreliable on embedded parts); resource-release ordering at process exit | `testing/tools/{allocation_checker,trt_shutdown_test}/` | — | Low |

### 1.4 Model-Optimizer — `/work/trt/repos/Model-Optimizer`

Baseline: `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` already covers **AutoCast FP16**,
**torch INT8 QAT**, and **FP8 ONNX PTQ (max calibration)** on a tiny MNIST CNN. Do not re-propose
those. Since TRT 11 strong typing removed weak-typing INT8 calibration, ModelOpt is the sanctioned
quantization path, and every candidate below emits explicit Q/DQ a strongly-typed engine consumes.

| # | Candidate | Repo path | Target | Prio |
| - | --------- | --------- | ------ | ---- |
| M1 | **ONNX PTQ *through* a custom TRT plugin** — build `libidentity_conv_plugin.so`, make an ONNX graph with a custom `IdentityConv` op, run `modelopt.onnx.quantization --trt_plugins=<.so>` so calibration and Q/DQ placement work through the plugin, deploy with `trtexec --staticPlugins`. **Fully self-contained** (synthetic ONNX + local CMake build), and a unique plugin×PTQ intersection | `examples/onnx_ptq/custom_op_plugin/` | 05-Plugin (new leaf) | **High** |
| M2 | **PyTorch → ONNX in NVFP4 / MXFP8 / INT4-AWQ → strongly-typed TRT** — `mtq.quantize` a timm ViT/Swin, opset-20 export, TRT-compat post-processing. Teaches the TRT **Conv2d override** rule (MXFP8/NVFP4→FP8, INT4-AWQ→INT8 for conv) and `AUTO` mixed precision. Needs TRT ≥ 10.11 | `examples/torch_onnx/torch_quant_to_onnx.py` | 04-Feature / 03-Workflow | **High** |
| M3 | **ONNX PTQ: INT4-AWQ / INT8-entropy / RTN** on a real vision model, plus **per-node calibration** (`--calibrate_per_node`) for OOM-safe quantization of large graphs. Natural extension of the FP8-max-only coverage | `examples/onnx_ptq/` | 04-Feature / 03-Workflow | **High** |
| M4 | **Q/DQ placement autotune** — automated Q/DQ placement search driven by **real TensorRT latency** (region discovery, pattern cache, optimize-from-existing-QDQ). Genuinely TRT-perf-in-the-loop; complements `07-Tool/FP16Tuning` | `examples/onnx_ptq/autotune/` | 07-Tool / 04-Feature | Med |
| M5 | **FastNAS pruning** (CIFAR ResNet) → ONNX → TRT. Pruning is entirely absent from the cookbook; CIFAR-only, no gated downloads; needs a TRT-export tail added (the notebook stops at torch) | `examples/pruning/cifar_resnet.ipynb` | 04-Feature/Pruning (new) | Med |
| M6 | CNN INT8 QAT on ResNet-50 with **`mto.save`/`restore`** of quantizer state. Partly overlaps the existing MNIST QAT; the value is save/restore + a realistic CNN. Expects an ImageNet-style tree — would need a small dataset swapped in | `examples/cnn_qat/torchvision_qat.py` | 04-Feature / 03-Workflow | Med |
| M7 | Diffusion (SDXL / FLUX) INT8/FP8/FP4 PTQ → ONNX → `trtexec --stronglyTyped --builderOptimizationLevel=4` with multi-input dynamic profiles. Great real-world strong-typing example; multi-GB weights, ≥48 GB memory | `examples/diffusers/quantization/` | 03-Workflow / 04-Feature | Med |
| M8 | ONNX PTQ for **non-classifiers** (SAM2 segmentation, Whisper ASR; Whisper ships `demo.wav`). ORT/DML-centric, so extra TRT glue is required | `examples/windows/onnx_ptq/{sam2,whisper}/` | 04-Feature | Med-Low |
| M9 | 2:4 structured sparsity (SparseGPT PTS + SAT) — pairs with `04-Feature/Sparsity`, which only shows the TRT sparsity *builder flag*, never how to produce sparse weights. Llama2-7B, gated, ~44 GB GPU | `examples/llm_sparsity/weight_sparsity/` | 04-Feature/Sparsity | Low |
| M10 | LLM PTQ FP8 / INT4-AWQ / NVFP4 → TRT-LLM engine. Keep as a **pointer**, not an example | `examples/llm_ptq/` | 09-TensorRT-LLM | Low |
| M11 | **AutoCast BF16 + node-sensitivity** (why some nodes stay FP32) — fold a BF16 case into the *existing* example rather than a new leaf. Reuses the tiny CNN, no downloads | `docs/source/guides/8_autocast.rst` | 03-Workflow (existing) | Low |
| M12 | Diffusion sparse attention (training-free skip-softmax) + quantization-aware distillation for FP4 recovery. Cutting-edge but heavy | `examples/diffusers/{sparsity,distillation}/` | 04-Feature | Low |

Skipped: `vllm_serve`, `llm_eval`, pure HF/vLLM serving (never touch TensorRT); `deepseek`,
`gpt-oss`, `minimax_m3`, `megatron_bridge`, `speculative_decoding`, `alpamayo`, `puzzletron`,
Minitron pruning (massive downloads / NeMo-Megatron containers).

### 1.5 TensorRT-RTX and tensorrtx

`10-TensorRT-RTX/{01..07}` already cover the RTX-only APIs **individually**; `02-API/Layer/Scale`
covers `add_scale` mechanics; `02-API/Network` only pokes at Network-object attributes.

| # | Candidate | Repo path | Target | Prio |
| - | --------- | --------- | ------ | ---- |
| R1 | **The RTX value proposition, end to end** — AOT compute-capability targeting (`num_compute_capabilities` + `set_compute_capability`), **weightless engine** (`STRIP_PLAN` + `REFIT` + `set_weights_name`, then `Refitter.set_named_weights().refit_cuda_engine()` at deploy), runtime cache serialized to disk so JIT kernels survive a restart, dynamic-shape specialization `LAZY`/`EAGER`/`NONE`, `WHOLE_GRAPH_CAPTURE`, and engine-validity preflight. The novelty is the **combined build-here / refit-and-run-there flow**, not the individual APIs | `TensorRT-RTX/samples/apiUsage/python/api_usage.py` | 10-TensorRT-RTX/08-WeightlessRefitDeploy (new) | **High** |
| R2 | Polygraphy driving TensorRT-RTX: set `POLYGRAPHY_USE_TENSORRT_RTX=1` **before** importing polygraphy (needs polygraphy ≥ 0.49.24); plus runtime cache on disk guarded by a `LockFile` for safe concurrent access | `TensorRT-RTX/demo/utils/engine.py` | 10-TensorRT-RTX or 07-Tool/Polygraphy | Med |
| R3 | Same tiny FC net built by hand vs parsed from ONNX, both under the **mandatory** RTX `STRONGLY_TYPED` flag (ships a 1 KB `.onnx`). Mostly redundant with existing demos | `TensorRT-RTX/samples/helloWorld/python/hello_world.py` | 10-TensorRT-RTX / 01-SimpleDemo | Low-Med |
| T1 | **LeNet-5 built entirely through the network API from raw weights** — conv → relu → avgpool ×2 → shuffle-flatten → `add_constant` + `add_matrix_multiply` + `add_elementwise` for the FC layers. Includes the TRT-10 migration lesson (`addFullyConnected` is gone, MatMul+bias replaces it). Make it self-contained by *generating* the weights | `tensorrtx/lenet/lenet.py` | 02-API/Network/BuildFromWeights (new) | **High** |
| T2 | **The `.wts` convention** (pairs with T1) — line 1 = blob count, then `name count hex0 hex1 …` with each float as big-endian IEEE-754 hex (`struct.pack('>f', v).hex()`); plus the PyTorch `state_dict` → flat buffer exporter and its register-buffer trick for folding derived tensors (anchor grids) into the weight file. The canonical "move trained weights into a hand-built TRT network without ONNX" recipe | `tensorrtx/tutorials/getting_started.md`, `lenet/lenet.py`, `yolov5/gen_wts.py` | 02-API/Network helper / 90-Misc | **High** |
| T3 | **BatchNorm folded into `IScaleLayer`** — TRT has no native BN, so BN becomes a channel-wise scale with `scale = gamma/sqrt(var+eps)`, `shift = beta - mean*gamma/sqrt(var+eps)`, `power = 1`. One of the most-asked network-API questions; the novelty is the *application*, not the Scale API | `tensorrtx/resnet/resnet34.cpp` (`addBatchNorm2d`) | 02-API/Network or 02-API/Layer/Scale | Med-High |
| T4 | "Push non-standard post-processing into a plugin" — YOLO anchor-grid → box decode inside the kernel so the engine outputs decoded detections. **Do not port the code**: it is `IPluginV2IOExt`, deprecated in TRT 10. At most a note re-expressed on `IPluginV3` | `tensorrtx/yolov5/plugin/yololayer.cu` | 05-Plugin (note) | Low-Med |

### 1.6 Torch-TensorRT and Tripy leftovers

Eight Torch-TensorRT examples landed under `06-DLFrameworkTRT/Torch-TensorRT/`; Tripy landed as
`07-Tool/nvtriPy`. What is left:

| # | Candidate | Repo path | Target | Why not done |
| - | --------- | --------- | ------ | ------------ |
| P1 | `refit_module_weights` — swap weights into a built engine without recompiling (LoRA-shaped) | `Torch-TensorRT/examples/dynamo/refit_engine_example.py` | `06-DLFrameworkTRT/Torch-TensorRT/Refit/` | **Nothing blocks it** — was PRIORITY HIGH |
| P2 | FP8 / INT8 PTQ through the Dynamo frontend (see also M2/M3) | `.../quantize_vit_fp8.py`, `.../vgg16_ptq.py` | `06-DLFrameworkTRT/ModelOptimizer/` (still a 20-line stub) | needs `nvidia-modelopt` + HF/CIFAR downloads |
| P3 | `enable_weight_streaming` + runtime budget, for models larger than VRAM | `.../weight_streaming_example.py` | `.../WeightStreaming/` | gated Llama-2 download |
| P4 | Tensor-/data-parallel compile across GPUs | `.../distributed_inference/` | `.../DistributedInference/` | needs >1 GPU |
| P5 | `Input(profiles=[...])` — N optimization profiles on one engine, runtime selection by index or `"auto"` | `.../multi_optimization_profiles.py` | `.../OptimizationProfiles/` | **Blocked**, see below |
| P6 | ResNet50 / NanoGPT / Stable Diffusion / SAM2 / ModelOpt quantization in Tripy | `TensorRT-Incubator/tripy/{notebooks,examples}/` | — | gated or multi-GB downloads plus `torch`+`transformers` *inside the nvtripy venv*, for a pre-1.0 API |

**P5 is blocked on the installed build.** `torch_tensorrt` 2.14.0a0 has none of the API:
`Input(profiles=...)` raises `ValueError`, `torch_tensorrt.runtime.optimization_profile` does not
exist, `torch.classes.tensorrt.Engine` exposes no profile methods, and
`dynamo/conversion/_TRTInterpreter.py` hard-codes a single `create_optimization_profile()`. The
source clone carries the feature under the *same* `2.14.0a0` version string, but `set_active_profile`
lives in `core/runtime/TRTEngine.cpp` — overlaying the newer Python files would not work, it needs a
full source build. Revisit after a Torch-TensorRT upgrade. The TensorRT-API-level equivalent already
exists as `08-Advance/MultiOptimizationProfile`.

### 1.7 Community repos — ideas, not imports

| Repo | Idea worth stealing |
| ---- | ------------------- |
| [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) (4.9k★) | the **per-op converter registry** as a Network-API teaching device |
| [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO), [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) | NMS/post-processing plugins + fused CUDA pre/post as a realistic detection pipeline |
| [mmdeploy](https://github.com/open-mmlab/mmdeploy) (3.1k★) | multi-backend deploy + custom TRT plugins |
| [jetson-inference](https://github.com/dusty-nv/jetson-inference) (8.9k★) | embedded C++ runtime demos |
| [WhisperLive](https://github.com/collabora/WhisperLive) (4.1k★), [x-stable-diffusion](https://github.com/stochasticai/x-stable-diffusion), [SD-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT) | streaming-ASR / diffusion end-to-end framing, dynamic-shape engine management |

Not TRT-specific, ecosystem context only: TNN, Tengine, lite.ai.toolkit, YOLOX, yolov5.

---

## 2. Decided against — do not re-propose

+ **DLA.** Every DLA path (`tutDLA*.cpp`, `samples/sampleCudla`, a `04-Feature/DLA/` group) would
  take the "no DLA core available" early return on this H100 (`builder.num_DLA_cores == 0`) and
  could never be verified. **Shipping unverifiable examples is worse than shipping none.** Reopen
  only on Orin / DRIVE / Jetson; the API to build from was confirmed present in TRT 11.0:
  `trt.MemoryPoolType.DLA_{MANAGED_SRAM,LOCAL_DRAM,GLOBAL_DRAM}`;
  `IBuilderConfig.{default_device_type, set_device_type, get_device_type, is_device_type_set,
  reset_device_type, can_run_on_DLA, DLA_core}` with `trt.DeviceType.DLA`;
  `Runtime.{DLA_core, num_DLA_cores}`; `trt.EngineCapability.DLA_STANDALONE`.
+ **demoDiffusion.** A faithful port means large downloads, a multi-engine pipeline manager and
  FP8/FP4 calibration data. Point users upstream instead of shipping a trimmed copy that drifts.
+ **Parsing the `TRT_*` custom operators from ONNX.** `TRT_Attention` / `TRT_MoE` /
  `TRT_KVCacheUpdate` are already covered as network-API layers in `02-API/Layer/*`.
+ **`sampleDevice.h`-style C++ RAII wrappers.** Tried, added to `cookbookHelper.cuh`, refactored
  `08-Advance/CudaGraph` onto it, then reverted in full. The C++ examples keep explicit
  `cudaMalloc` / `cudaStreamCreate` / `cudaGraph*` calls paired with explicit releases.
+ **TensorRT-LLM intake — closed 2026-08-28.** Only 9 non-3rdparty files in that repo still
  `import tensorrt`; the engine-build stack (`builder.py`, `network.py`, `module.py`,
  `python_plugin.py`, `tools/plugin_gen/`) is deleted and all three plugin examples fail to import.
  For a Python-written TRT plugin take OSS `samples/python/python_plugin` (G3) instead.
+ **Two "gaps" that were not gaps:** `05-Plugin/InPlacePlugin` **does** already use aliased I/O in
  C++ (`v_2_0::IPluginV3OneBuild` + `getAliasedInput`; an earlier grep missed the camelCase), and
  `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` **does** already cover AutoCast.
+ RTX Flux.1[dev] demo and the tensorrtx model zoo: cite, never import.

---

## 3. EXCLUDE — internal / sensitive (do **not** import)

+ `tools/infer_fuzzing/` — **security**: safety-team ONNX fuzzer; the README leaks a real customer
  model path (Toyota) and internal attack methodology.
+ `tools/memory_usage_safe/` — leaks an internal SSH endpoint / hardcoded IP and an unreleased
  remote QNX timing-server flow.
+ `tools/infer_ref_check_safe/` — undocumented internal debug hooks (`__LUNOWUD`,
  `MYELIN_DUMP_ALL_VALUES`, `MYELIN_SAVE_TENSOR_VALUES`); Safe/LWE runtime.
+ `samples_internal/dlaLoadableExtractor/`, `tools/engine_dumper/`, `tools/plan_converter/` —
  downcast to non-public classes / parse engine-plan binary internals (`api/engine.h`,
  `dispatch/planHeaders.h`).
+ `tools/infer_device*/`, `tools/boot_time_bench/` — internal Myelin / NVRTC / Safe-runtime plumbing.
+ `plugin_internal/` (dlrmBottomMLP, rnRes2*, rnntEncoder, smallTileGEMM) — unreleased plugins.
+ `projects/customer_plugins/{zeekr,edge-llm}/` — **named after customers. Absolutely exclude.**
+ `include/NvInferSerialize.h` — internal `serializeNetwork_INTERNAL()`, marked `@private` and absent
  from the public headers. The cookbook's `utils_network_serialization.py` is an independent
  implementation; do **not** try to align it with this API.
+ All fusion / optimizer / Myelin unit tests (`tutApiFoldReformatIntoMyelin*`, `tutHorizontalMerge*`,
  `tutFuseGELU*`, `tutPointWiseFusion*`, `tutDisableFusion*`, `tutMyelin*`, `tutRaggedTensorLayer*` …)
  — they encode internal pass names, tactic heuristics and nvbug IDs.
+ `documentation/architecture/` (BuilderArch/RuntimeArch + UML) — read it, do not mirror it.
+ `dev_docs/`, `scripts/gitlab_ci/`, `scripts/ai-agents/`, `infrastructure/`, `coverity/`, `capture/`,
  `multigen/`, `plc/`, `optimizer/`, `runtime/`, `samples/README_internal.md` — internal plumbing.
+ Safety samples (`sampleSafeMNIST`, `sampleSafePluginV3`, `trtSafeExec`) — `trtSafeExec`'s own README
  says it is NOT safety-certified and may violate AUTOSAR; the cookbook already has `04-Feature/Safety`.

**Ambiguous, vet first:** `tools/engine_visualizer/` (L7) and `tools/infer_ref_check/` (L9) carry
proprietary SPDX headers and internal URLs/emails — relicense and scrub before adapting.

---

## 4. Reference

### 4.1 Ecosystem repos

| Repo | Status for the cookbook |
| ---- | ----------------------- |
| [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) | primary upstream → §1.2 |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) (rebranded from "TensorRT Model Optimizer", Dec 2025) | §1.4 |
| [pytorch/TensorRT](https://github.com/pytorch/TensorRT) | `06-DLFrameworkTRT`, leftovers in §1.6 |
| [NVIDIA/TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX) | `10-TensorRT-RTX`, R1–R3 |
| [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) (7.8k★) | T1–T4 |
| [NVIDIA/TensorRT-Incubator](https://github.com/NVIDIA/TensorRT-Incubator) | **done** → `07-Tool/nvtriPy` (own venv: it brings TensorRT 10 with it) |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | **no action**, intake closed (§2) |
| [NVIDIA/TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect), [TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) | no action |
| [tensorflow/tensorrt](https://github.com/tensorflow/tensorrt) | **archived Feb 2025**, legacy reference only |

### 4.2 Repo facts worth remembering

+ `.agents/skills/` appeared in the OSS repo at 11.1 (`trt-onnx-quickstart`,
  `trt-cpp-runtime-quickstart`, `trt-perf-analysis`, `trt-strong-typing-migration`,
  `trt-torch-quickstart`). Not cookbook content, but `trt-perf-analysis/scripts` validates
  layer-info/profile JSON pairs and overlaps usefully with `07-Tool/trex`.
+ `cookbook/README.outline.txt` "Useful Links" was health-checked 2026-07-22 and fixed: the
  Operators Document URL was a **404**, Torch-TensorRT and the TensorRT download page were
  redirects, TF-TRT is annotated "(archived, read-only)".

### 4.3 Finished: TREx migration

`trt-engine-explorer` (TREx, ~4.7k lines) was dismantled into the cookbook: **all 15 mapped features
plus EngineCard/Summarize** now live as `07-Tool/trex/00..15`, with every helper in the single module
`tensorrt_cookbook/utils_engine_explorer.py` — **no pandas, no plotly**; graphviz / onnx / openpyxl /
pynvml / tensorrt are lazily imported. Decisions that shaped it: baseline `origin/main` merged with
`origin/dev-trt-10.9-update`; matplotlib only, writing files instead of interactive/browser output
(Range Slider → script-level parameters, animation → gif/png, 3D → 2D); pandas replaced by
numpy + list/dict. Deliberately dropped: `interactive.py`, `notebook.py`, `misc.display_df`,
`bin/trex`, the notebooks, `install.sh`, `setup.py`.

Afterwards `git rm -r trex/ utils/ bin/ notebooks/ examples/ tests/` (118 files) was run in
`trt-engine-explorer` — **not** in `trt-engine-explorer-backup`, which is kept intact — leaving only
the docs, requirements, `setup.py`, `install.sh`, `images/` and `.git`. That deletion is staged but
**not committed**. Remaining decision → S5.
