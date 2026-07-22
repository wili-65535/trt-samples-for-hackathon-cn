# Candidates from TensorRT-GitLab (internal — vet before use)

Source repo explored: `/work/trt/TensorRT-GitLab` (internal/GitLab superset of public OSS).
Target cookbook: `/work/trt-samples-for-hackathon-cn/cookbook` (TensorRT 11, strong typing).

This is a curated shortlist of content that is (a) genuinely valuable/interesting, (b) NOT
already well covered by the cookbook, and (c) suitable for a PUBLIC educational cookbook.
Everything below was cross-checked against existing cookbook sections. A hard EXCLUDE list of
internal/sensitive material is at the bottom — read it before importing anything.

Note: the cookbook is already extremely comprehensive. Many obvious samples in the GitLab tree
(sampleOnnxMNIST, sampleProgressMonitor, sampleNamedDimensions→LabeledDimension, stream_writer→
IStreamWriter, sample_weight_stripping→Refit/STRIP_PLAN, dds_faster_rcnn→OutputAllocator,
editable timing cache, IStreamWriter, RuntimePlatform cross-platform, python_plugin/non_zero_plugin/
QDP) are ALREADY covered and are intentionally omitted here.

---

## HIGH priority

### 1. Empty-tensor use cases — fill the existing TODO stubs
- Repo: `tests/unitTests/tutEmptyTensorTests.cpp` (also `tutNonZeroTests.cpp` empty-input cases `{10,0}`).
- Demonstrates: building graphs that legitimately produce zero-volume tensors (neg/slice/prod
  chains), zero-trip behavior, and how TRT handles empty memory blocks.
- Cookbook target: `04-Feature/EmptyTensor-TODO/` and `08-Advance/EmptyTensor-TODO/` are currently
  empty stubs (README + unit_test.yaml only, no `main.py`). This directly fills a known gap.
- Effort/priority: Medium effort, **High** priority (unblocks two placeholder dirs).

### 2. CUTLASS Python-DSL kernel inside a plugin
- Repo: `samples/python/cute_dsl_plugin/`
- Demonstrates: an `IPluginV3` whose `enqueue()` launches a CuteDSL (CUTLASS Python DSL) RMSNorm
  kernel on TRT's stream; zero-copy cupy→torch→dlpack hand-off, FP32-reduce/FP16-store.
- Cookbook target: `05-Plugin/` (new leaf, e.g. `CuteDSLPlugin`). Complements existing
  PythonPlugin/QuickDeployablePlugin by showing a real modern kernel backend.
- Effort/priority: Medium–High effort (extra CuteDSL dep), **High** interest (novel, nothing similar
  in cookbook).

### 3. ModelOpt AutoCast: FP32 ONNX → mixed FP16/BF16 for strong typing
- Repo: `samples/python/strongly_type_autocast/`
- Demonstrates: using ModelOpt's AutoCast to auto-insert casts and keep accuracy-critical nodes in
  FP32, then building with TRT strong typing. Positioned explicitly as the replacement for the now
  deprecated/removed weak typing — highly relevant since the cookbook targets TRT 11 strong typing.
- Cookbook target: `03-Workflow/` (new leaf) or `04-Feature/StronglyTyped/` extension. Cookbook has
  `pyTorch-ModelOptimizer-ONNX-TensorRT` and `FP16Tuning` but not the ONNX-side AutoCast flow.
- Effort/priority: Medium effort, **High** priority.

---

## MEDIUM priority

### 4. Chained dynamic-reshape preprocessing engine
- Repo: `samples/sampleDynamicReshape/` (C++)
- Demonstrates: a small dynamic-shape preprocessor network (`-1` dims + `IResizeLayer`) whose output
  feeds a second fixed-shape ONNX MNIST engine — i.e. running two engines back-to-back so a static
  model can accept arbitrary input sizes.
- Cookbook target: `08-Advance/` (multi-engine pattern) or `01-SimpleDemo`/`03-Workflow`. The
  "two chained engines / preprocessing engine" pattern is not shown as its own example.
- Effort/priority: Low–Medium effort, **Medium** priority.

### 5. Multi-device context parallelism via polygraphy shard
- Repo: `samples/python/attention_mdtrt/`
- Demonstrates: sharding an attention ONNX across GPUs along the sequence dim using
  `polygraphy multi-device shard` + a `hint.json`, inserting `DistCollective` ReduceScatter/AllGather,
  run under MPI + NCCL.
- Cookbook target: `08-Advance/MultiDevice/` (or a DistCollective workflow). Cookbook has the
  DistCollective *layer* and MPIUtils, but not this end-to-end sharding workflow.
- Effort/priority: Medium–High effort (needs multi-GPU), **Medium** priority. Advanced but public.

### 6. Parallel multi-GPU engine deserialization timing
- Repo: `samples_internal/deserializeTimer/`  (uses only public `NvInfer.h` + trtexec sample utils)
- Demonstrates: deserialization is per-device and can be parallelized across GPUs
  (`std::async` vs sequential) to cut multi-GPU startup latency.
- Cookbook target: `08-Advance/MultiDevice/`. Concept is clean; would need the trtexec sample-util
  includes stripped/vendored.
- Effort/priority: Medium effort (needs >1 GPU), **Medium** priority.

### 7. Concurrent multi-engine serving with per-task CUDA graph + device pinning
- Repo: `samples_internal/sampleMultiTasks/` (Apache-2.0 in-tree)
- Demonstrates: running several independent engines concurrently, each pinned to a device with
  optional CUDA-graph capture (`--task=engine=...,device=N,graph=0`).
- Cookbook target: `08-Advance/` — the cookbook covers MultiContext, MultiStream, MultiDevice and
  CudaGraph *separately*; a combined "serve N engines concurrently" example is missing.
- Effort/priority: High effort (heavy trtexec-internal coupling → needs rewrite against public API),
  **Medium** priority (valuable concept).

### 8. IPluginV2DynamicExt → IPluginV3 migration cheat-sheet
- Repo: `samples/python/sample_plugin_v2_to_v3_migration/`
- Demonstrates: side-by-side V2 vs V3 scale plugin with a full method-mapping table; key win is
  serialization via `get_fields_to_serialize()`.
- Cookbook target: `05-Plugin/` (e.g. `MigrationV2toV3`). Cookbook has V2-deprecated and V3 examples
  but no explicit migration walkthrough.
- Effort/priority: Low–Medium effort, **Medium** priority.

### 9. FP8 QDQ scale/numeric semantics ("picky FP8")
- Repo: `tests/unitTests/tutPickyFP8.cpp`
- Demonstrates: an FP8 Q/DQ/MatMul network deliberately engineered to be numerically sensitive to
  wrong types/scales (alpha = 1+2⁻¹¹ tricks) — an excellent teaching aid for FP8 scale handling.
- Cookbook target: `02-API/Layer/QDQStructure/` extension or `04-Feature` low-bit precision note.
- Effort/priority: Medium effort, **Medium** priority. (Use as inspiration; rewrite in Python API.)

### 10. Shape-tensor I/O and host↔device shape transfers
- Repo: `tests/unitTests/tutApiShapeTensorTests.cpp`, `tutShapeH2DTests.cpp`, `tutShuffleShapeTests.cpp`
- Demonstrates: shape tensors as network I/O (shuffle→shape→cast), INT32 vs INT64 shape tensors,
  0-D/N-D shape tensors, and shape host-to-device transfer corner cases.
- Cookbook target: `02-API` shape-input area / `05-Plugin/ShapeInputTensor` companion. These corner
  cases (esp. INT64 shape tensors, 0-D) are thinly covered.
- Effort/priority: Medium effort, **Medium** priority (internal `*Runner.h` includes are just harness).

### 11. Engine layer-info → multi-format graph visualization
- Repo: `tools/engine_visualizer/` (`plotEngine.py`)
- Demonstrates: turning `trtexec --exportLayerInfo` / `IEngineInspector` JSON (+ optional
  `--exportProfile` timings) into Graphviz (dot/PDF), browser HTML (Dagre), and yEd GraphML.
- Cookbook target: `07-Tool/` — overlaps in spirit with `trex`, but the one-JSON→three-formats export
  is a nice standalone. NOTE: scrub proprietary SPDX header, internal contact email, and
  `gitlab-master.nvidia.com` links before use.
- Effort/priority: Low effort (pure Python on public JSON), **Medium** priority.

### 12. I/O format validation on one engine
- Repo: `samples/sampleIOFormats/` (C++)
- Demonstrates: the same engine consuming `kLINEAR` / `kHWC` / `kCHW32` I/O via `setAllowedFormats`,
  validating each layout against golden output.
- Cookbook target: `04-Feature/DataFormat/` deepening. Cookbook DataFormat already shows format
  reformats; the "same engine, multiple validated I/O layouts" angle is a useful addition.
- Effort/priority: Low effort, **Medium** priority.

---

## LOWER priority (targeted enhancements to existing sections)

Each maps to an existing cookbook layer/feature dir and adds corner cases from the internal unit
tests. All are genuine user-facing API patterns (NOT the flagged fusion/Myelin tests below).

- **13. Dynamically-quantized weights for conv** — `tests/unitTests/tutApiQuantDynamicWeightTest.cpp`;
  quantizing runtime-provided filter/bias, grouped conv. → `02-API/Layer/Convolution` or `Quantize`. Low.
- **14. MXFP / block weight-only QDQ (E8M0 scales)** — `tests/unitTests/tutBlockQDQ.cpp`,
  `tutPackedQDQ.cpp`; per-block scales along axis 0/1, MXFP. → `02-API/Layer/QDQStructure` (already
  covers block-rank scale; MX/E8M0 specifics could be added). Low.
- **15. Structured-sparsity conv + refit sparse-vs-dense** — `tests/unitTests/tutApiSparseConvTests.cpp`;
  one of the few genuine refit demos (refit sparse OK, refit dense→error). → `04-Feature/Sparsity` +
  `Refit`. Low–Medium.
- **16. GridSample corner cases** — `tests/unitTests/tutGridSampleTests.cpp`; 5-D input, CUBIC mode,
  FILL/CLAMP/REFLECT, align-corners. → `02-API/Layer/GridSample`. Low.
- **17. OneHot negative-axis handling** — `tests/unitTests/tutOneHotTests.cpp`. → `02-API/Layer/OneHot`. Low.
- **18. ReverseSequence batch/sequence axis variants** — `tests/unitTests/tutReverseSequenceTests.cpp`.
  → `02-API/Layer/ReverseSequence`. Low.
- **19. Assertion via shape machine** — `tests/unitTests/tutAssertionLayerTests.cpp`; assert runtime
  dimension relationships via shape→slice→elementwise. → `02-API/Layer/Assertion`. Low.
- **20. Loop/If corner cases** — `tests/unitTests/tutLoopTests.cpp`, `tutTrivialLoopTests.cpp`,
  `tutIfConditionalTests.cpp`; nested loops, kWHILE/kCOUNT, reverse iterator, lazy-vs-eager
  conditionals, nested conditionals. → `02-API/Layer/LoopStructure` / `IfConditionStructure`. Low
  (already covered; these add depth).
- **21. Golden-output accuracy checker** — `tools/infer_ref_check/`; per-output mixed-metric
  comparison (cosine for TopK/ArgMax outputs, elementwise otherwise). → `07-Tool/` (concept overlaps
  Polygraphy). NOTE: relicense + replace internal `common/harness*` deps. Low.
- **22. C++ NonZero DDS plugin** — `samples/sampleNonZeroPlugin/`; C++ counterpart to the Python DDS
  plugin. → `05-Plugin/DataDependentShape` (C++ variant). Low.

---

## EXCLUDE — internal / sensitive / not adaptable (do NOT import)

Flagged during exploration; listed so they are not accidentally revisited.

- `tools/infer_fuzzing/` — SECURITY: safety-team ONNX security fuzzer; README leaks a real customer
  model path (Toyota) and internal attack methodology. Exclude.
- `tools/memory_usage_safe/` — leaks internal SSH endpoint / hardcoded IP `10.172.54.89` and an
  unreleased `--remoteAutoTuningConfig` remote QNX timing-server flow. Exclude.
- `tools/infer_ref_check_safe/` — uses undocumented internal debug hooks `__LUNOWUD`,
  `MYELIN_DUMP_ALL_VALUES`, `MYELIN_SAVE_TENSOR_VALUES`; Safe/LWE runtime. Exclude.
- `samples_internal/dlaLoadableExtractor/` — downcasts to non-public `rt::dla::DLABaseRunner`,
  includes internal `api/engine.h`; exposes engine/DLA internals. Exclude.
- `tools/engine_dumper/`, `tools/plan_converter/` — parse non-public engine/plan binary internals
  (`api/engine.h`, `dispatch/planHeaders.h`). Exclude.
- `tools/infer_device/`, `tools/infer_device_safe/`, `tools/boot_time_bench/` — probe internal
  Myelin/NVRTC/Safe-runtime plumbing (QNX/aarch64, `myelin/*`, `NvInferSafeRuntime.h`). Exclude.
- `samples_internal/sampleBuildOptions/` — empty internal RCCA (compile smoke test for an nvbug);
  teaches nothing. Exclude.
- Safety samples/tool: `samples/sampleSafeMNIST/`, `samples/sampleSafePluginV3/`, `samples/trtSafeExec/`
  — automotive certified-safety runtime; cookbook already has `04-Feature/Safety`. `trtSafeExec` README
  itself states it is NOT safety-certified / may violate AUTOSAR. Low value for a public cookbook; treat
  as out of scope.
- `samples/README_internal.md` — internal-only doc; do not mirror.
- All fusion/optimizer/Myelin unit tests — e.g. `tutApiFoldReformatIntoMyelinTests.cpp`,
  `tutApiConv*Tests.cpp`, `tutHorizontalMergeTests.cpp`, `tutFuseGELUTests.cpp`,
  `tutPointWiseFusion*Tests.cpp`, `tutDisableFusionTests.cpp`, `tutMyelin*Tests.cpp`,
  `tutDistributiveIndepenceTest.cpp`, `tutReducePrivateOpTests.cpp`,
  `tutRaggedTensorLayerTests.cpp` (gated behind unreleased `ENABLE_FEATURE_RAGGED_TENSOR`). These
  encode internal pass names, tactic heuristics, and nvbug IDs — they would leak internals and are NOT
  user-facing API. Exclude.

## Ambiguous — vet before use
- Internal unit tests generally `#include` internal harness headers (`*Builder.h`, `*Runner.h`,
  `optimizer/`, `myelin/`). The *layer APIs* they exercise are public, but any import must be rewritten
  from scratch against the public Python/C++ API — copy the *idea*, never the file.
- `tools/engine_visualizer/` and `tools/infer_ref_check/` carry proprietary SPDX headers and internal
  URLs/emails; only adapt after relicensing and scrubbing internal references.
