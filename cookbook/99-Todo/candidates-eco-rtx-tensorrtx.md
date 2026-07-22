# Candidates from TensorRT-RTX and tensorrtx

Two source repos analyzed for content adaptable into the cookbook's runnable-example style
(`main.py` + `TRTWrapper`, one focused concept per leaf directory):

- **(A) TensorRT-RTX** — `/work/trt/repos/TensorRT-RTX` (NVIDIA, RTX/consumer-GPU flavor, Turing→Blackwell, JIT runtime optimizer).
- **(B) tensorrtx** — `/work/trt/repos/tensorrtx` (wang-xinyu, popular nets built purely with the TRT network-definition API from raw `.wts` weights, no ONNX).

Priority = value-to-a-learner vs. effort to port. Cross-checked against the existing cookbook
(`10-TensorRT-RTX`, `02-API/Network`, `02-API/Layer/*`, `05-Plugin`) so proposals below are **not already well covered**.

---

## Section A — TensorRT-RTX

This repo is small: only two samples (`helloWorld`, `apiUsage`) and one demo (`flux1.dev`).
The existing `cookbook/10-TensorRT-RTX` already covers the RTX-only **APIs individually**
(ComputeCapability, RuntimeCache, EngineValidity, RuntimeConfig strategy, stream-capturable,
`REQUIRE_USER_ALLOCATION`, enums). The gap is that it does **not** show the headline RTX
**end-to-end workflow** (weightless AOT engine + refit-at-deploy), the ONNX-parse path, nor the
Polygraphy-with-RTX toggle.

### Summary of picks (A)

| # | Candidate | Repo path | Target section | Priority | Self-contained |
|---|-----------|-----------|----------------|----------|----------------|
| A1 | Weightless AOT engine + refit-at-deploy + runtime-cache-to-disk (full RTX deployment story) | `samples/apiUsage/python/api_usage.py` (+ `cpp/apiUsage.cpp`) | `10-TensorRT-RTX/08-WeightlessRefitDeploy` (new) | **High** | Yes (synthesizes weights) |
| A2 | Polygraphy driving TensorRT-RTX via `POLYGRAPHY_USE_TENSORRT_RTX=1` + runtime cache to disk w/ file lock | `demo/utils/engine.py` | `10-TensorRT-RTX` (small note/example) or `07-Tool/Polygraphy` | Medium | Yes (env-var toggle only) |
| A3 | ONNX-parse path vs hand-built network, side-by-side, in the RTX runtime | `samples/helloWorld/python/hello_world.py` (+ `helloWorld.onnx`) | `10-TensorRT-RTX` / `01-SimpleDemo` | Low-Med | Yes (onnx shipped, 1 KB) |
| A4 | Flux.1[dev] diffusion pipeline on RTX (low-VRAM, fp8/fp4, dynamic shape, cache modes) | `demo/flux1.dev/` | (do not port — too heavy) | Low | No (HF token, big weights) |

### Details

**A1 — Weightless AOT + refit-at-deploy (the RTX value prop). HIGH.**
`/work/trt/repos/TensorRT-RTX/samples/apiUsage/python/api_usage.py`
This single sample is the exemplary combined RTX workflow and the strongest candidate. It chains:
- AOT compute-capability targeting: `builder_config.num_compute_capabilities` +
  `set_compute_capability(trt.ComputeCapability.CURRENT/SM89/SM120/SM75, i)` — build once,
  deploy OS-independently to Ampere+ (or explicit SM list; Turing opt-in).
- **Weightless engine**: `BuilderFlag.STRIP_PLAN` + `BuilderFlag.REFIT`, `network.set_weights_name(...)`,
  then at deploy `trt.Refitter(engine).set_named_weights(...).refit_cuda_engine()`.
- Runtime cache **serialized to persistent storage** to keep JIT-compiled kernels between runs
  (`runtime_config.create_runtime_cache()` → `set_runtime_cache` → `runtime_cache.serialize()`).
- Dynamic-shape kernel specialization strategy (`LAZY`/`EAGER`/`NONE`) and CUDA-graph
  `WHOLE_GRAPH_CAPTURE`.
- Engine-validity preflight (`runtime.get_engine_validity` + `EngineInvalidityDiagnostics` bit decode).

Why it belongs: the cookbook shows each of these APIs in isolation but never the **weightless-build-here,
refit-and-run-there** deployment story that is the entire reason RTX exists. Port as one new
`10-TensorRT-RTX/08-WeightlessRefitDeploy/main.py`, reusing the existing `tensorrt_rtx as trt` +
`print_enumerated_members` conventions already in `10-TensorRT-RTX/00-AllInOne`. Effort: moderate
(the sample is ~350 lines but mostly boilerplate CUDA glue that `TRTWrapper`-style helpers replace).

**A2 — Polygraphy + TensorRT-RTX integration. MEDIUM.**
`/work/trt/repos/TensorRT-RTX/demo/utils/engine.py`
Demonstrates the one non-obvious integration nugget: set `os.environ["POLYGRAPHY_USE_TENSORRT_RTX"]="1"`
**before** importing polygraphy so `polygraphy.backend.trt` builds/loads RTX engines
(needs polygraphy >= 0.49.24). Also shows runtime-cache persisted to disk guarded by `LockFile`
(safe concurrent access) and a trt-dtype↔torch-dtype map. Small, self-contained concept; good as a
short note or a `07-Tool/Polygraphy` RTX variant. Do not port the surrounding `Engine`/metadata class.

**A3 — Hand-built vs ONNX-parsed network in one file. LOW-MED.**
`/work/trt/repos/TensorRT-RTX/samples/helloWorld/python/hello_world.py`
Same tiny FC net built two ways: manually via `add_constant`/`add_matrix_multiply`/`add_activation`,
or by `trt.OnnxParser().parse_from_file("helloWorld.onnx")`, both under the mandatory RTX
`STRONGLY_TYPED` flag. The shipped `helloWorld.onnx` (~1 KB) makes it self-contained. Mostly
redundant with existing basic demos; value is only the strong-typing-is-mandatory-on-RTX point and
the ONNX path. Low priority.

**A4 — Flux.1[dev] diffusion demo. LOW (reference only).**
`/work/trt/repos/TensorRT-RTX/demo/flux1.dev/` (`flux_demo.py`, `flux_demo.ipynb`, `models/`, `pipelines/`).
Real-world RTX deployment: `--precision {bf16,fp8,fp4}`, `--dynamic-shape`, `--enable-runtime-cache`,
`--low-vram`, cache modes `full`/`lean`. Excellent illustration of low-VRAM + JIT cache on consumer
GPUs but requires an HF token and multi-GB weights — **do not import**; cite as the canonical
end-to-end reference only.

---

## Section B — tensorrtx

tensorrtx builds ~60 networks purely through the network-definition API from plain-text `.wts` weights.
Do **not** import the model zoo. The reusable, teachable patterns are the *techniques*, not the models.
The cookbook's `02-API/Network/main.py` only exercises Network-object methods on a trivial graph, and
`02-API/Layer/Scale` shows `add_scale` mechanics but not the BN-folding use case — so the patterns
below fill a real gap: "how do I hand-build a real CNN and load real weights."

### Summary of picks (B)

| # | Candidate | Repo path | Target section | Priority | Self-contained |
|---|-----------|-----------|----------------|----------|----------------|
| B1 | LeNet-5 built end-to-end from `.wts` purely via network API (Conv/Act/Pool/MatMul/bias-add) | `lenet/lenet.py`, `lenet/lenet.cpp` | `02-API/Network/BuildFromWeights` (new) | **High** | Yes (weights can be synthesized) |
| B2 | `.wts` weight convention: text hex format + `load_weights()` + PyTorch `gen_wts.py` exporter | `tutorials/getting_started.md`, `lenet/lenet.py` (`load_weights`), `yolov5/gen_wts.py` | `02-API/Network` helper / `90-Misc` | **High** | Yes |
| B3 | BatchNorm-as-IScaleLayer folding (TRT has no native BN layer) | `resnet/resnet34.cpp` (`addBatchNorm2d`), also arcface/densenet/dbnet/detr | `02-API/Network` or `02-API/Layer/Scale` | Med-High | Yes |
| B4 | YOLO detection decode fused into a custom plugin (anchors→boxes) | `yolov5/plugin/yololayer.cu`, `yololayer.h` | `05-Plugin` (pattern note) | Low-Med | Partial (deprecated API) |

### Details

**B1 — LeNet-5 from raw weights via network API. HIGH.**
Python: `/work/trt/repos/tensorrtx/lenet/lenet.py`  · C++: `/work/trt/repos/tensorrtx/lenet/lenet.cpp`
The cleanest, smallest complete example of the tensorrtx philosophy: no parser, just
`add_convolution_nd` → `add_activation(RELU)` → `add_pooling_nd(AVERAGE)` ×2 → flatten via
`add_shuffle` → `add_constant`+`add_matrix_multiply`+`add_elementwise(SUM)` for FC layers, ending in
softmax. The Python file already uses the modern explicit-batch + `add_matrix_multiply` idiom
(note: original `addFullyConnected` is gone in TRT 10, and they show the MatMul+bias replacement
inline — a genuinely useful migration lesson). Port as `02-API/Network/BuildFromWeights/main.py`
under `TRTWrapperV1`; make it self-contained by generating random weights into the `.wts` format
rather than shipping `lenet5.wts`. Effort: low. This directly enriches `02-API/Network`, which today
only pokes at Network-object attributes.

**B2 — The `.wts` weight-loading convention. HIGH (pairs with B1).**
Format spec: `/work/trt/repos/tensorrtx/tutorials/getting_started.md` (section "The .wts content format").
Loader: `load_weights()` in `/work/trt/repos/tensorrtx/lenet/lenet.py`.
Exporter: `/work/trt/repos/tensorrtx/yolov5/gen_wts.py`.
The convention: line 1 = blob count; each line `name count hex0 hex1 …` with each float as big-endian
IEEE-754 hex (`struct.pack('>f', v).hex()` / `struct.unpack('>f', bytes.fromhex(h))`). This is the
canonical "how to move trained weights into a hand-built TRT network without ONNX" recipe and is a
natural small utility/example for `02-API/Network` (or a `90-Misc` weight-IO helper). Effort: low.
The exporter side (`gen_wts.py`) also teaches the PyTorch `state_dict` → flat buffer step and the
register-buffer trick for folding derived tensors (e.g. anchor grids) into the weight file.

**B3 — BatchNorm folded into IScaleLayer. MED-HIGH.**
`/work/trt/repos/tensorrtx/resnet/resnet34.cpp` (`addBatchNorm2d`, ~lines 78-108); same helper
recurs across `arcface/`, `densenet121.cpp`, `dbnet/common.hpp`, `detr/backbone.hpp`, `crnn/`, etc.
TensorRT has no native BatchNorm layer, so BN is folded into a channel-wise
`addScale(kCHANNEL, shift, scale, power)` with `scale = gamma/sqrt(var+eps)`,
`shift = beta - mean*gamma/sqrt(var+eps)`, `power = 1`. This is one of the most-asked
"how do I do BN in the network API" questions and is a perfect focused addition to
`02-API/Network` or an applied companion to the existing `02-API/Layer/Scale`. Self-contained
(synthesize BN params). Effort: low.

**B4 — YOLO detection decode as a plugin. LOW-MED (pattern, not import).**
`/work/trt/repos/tensorrtx/yolov5/plugin/yololayer.cu` + `yololayer.h`.
Demonstrates fusing model-specific post-processing (anchor grid → (x,y,w,h,conf,cls) decode,
per-cell iteration in a CUDA kernel) inside a TRT plugin so the engine outputs decoded detections
directly. Learnable pattern = "push non-standard postprocess into a plugin". **Caveat**: it is built
on `IPluginV2IOExt` (deprecated in TRT 10; `enqueue`/`getOutputDimensions` signatures), whereas the
cookbook's `05-Plugin/BasicExample` already teaches the modern `IPluginV3` path. So do **not** port
the code; at most add a short `05-Plugin` note describing the "decode-inside-plugin" idea and
re-expressing it on `IPluginV3`. Low-Med priority.

---

## Cross-check notes (avoid duplication)

- `10-TensorRT-RTX/{01..07}` already cover ComputeCapability, EngineValidity, RuntimeCache,
  RuntimeConfig strategy, stream-capturable, `REQUIRE_USER_ALLOCATION`, enums — **A1's novelty is the
  combined weightless+refit deployment flow**, not the individual APIs.
- `02-API/Layer/Scale/main.py` covers `add_scale` UNIFORM/CHANNEL/ELEMENTWISE/`add_scale_nd`
  mechanics — **B3's novelty is the BN-folding application**, not the Scale API itself.
- `02-API/Network/main.py` only exercises Network-object attributes on a 1-layer graph — **B1/B2 add
  the missing "build a real net from real weights" story**.
- `05-Plugin` already has modern `IPluginV3` examples (`BasicExample`, `PythonPlugin`,
  `DataDependentShape`) — **B4 offers only a pattern, on a deprecated API**; low value to import.
