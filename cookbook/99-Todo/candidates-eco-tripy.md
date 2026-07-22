# Candidates from NVIDIA/TensorRT-Incubator (Tripy)

Source repo explored: `/work/trt/repos/TensorRT-Incubator` (the `tripy/` subproject).
Tripy is a debuggable, **Pythonic / NumPy-PyTorch-like eager-mode frontend** for TensorRT:
you define a model with `nvtripy.Module`, run it eagerly for debugging, then `tp.compile(...)`
to JIT it into a TensorRT `Executable`. This makes it a natural fit for the framework-frontends
family alongside Torch-TensorRT.

- Package: `nvtripy`, version **0.1.7** (public, `pip install nvtripy -f https://nvidia.github.io/TensorRT-Incubator/packages.html`).
- Public docs site: <https://nvidia.github.io/TensorRT-Incubator/>. Container: `ghcr.io/nvidia/tensorrt-incubator/nvtripy:latest`.
- Repo snapshot: commit `e5b2b595` (2026-03-25).
- Target cookbook: `/work/trt-samples-for-hackathon-cn/cookbook` (TensorRT 11). Most relevant home: `06-DLFrameworkTRT/` (new `06-DLFrameworkTRT/Tripy-TensorRT/`), with a possible pointer from `01-SimpleDemo`.

**Maturity note:** Pre-1.0 (`0.1.7`), API surface still moving, but it is publicly pip-installable,
has a hosted docs site, versioned releases, CI badges, SPDX headers, and doc-tested examples
(`Tripy: TEST: EXPECTED_STDOUT` blocks). Mature enough for a small, clearly-labeled "experimental
frontend" intro example; not mature enough to build a large cookbook section around yet.

Priority = value-to-a-learner vs. effort to port into the cookbook's `main.py` style.

---

## Summary of top picks

| # | Candidate | Repo path | Target section | Priority | Self-contained |
|---|-----------|-----------|----------------|----------|----------------|
| 1 | Hello-Tripy: define → eager → compile → run (Conv+ReLU) | `tripy/README.md` Quick Start + `docs/pre0_user_guides/00-introduction-to-tripy.md` | 06-DLFrameworkTRT/Tripy-TensorRT (intro) | **High** | Yes (no weights/net) |
| 2 | Compiler guide: `InputInfo`, dynamic shapes, save/load `Executable` | `tripy/docs/pre0_user_guides/01-compiler.md` | 06-DLFrameworkTRT/Tripy-TensorRT/Compile | **High** | Yes |
| 3 | Eager-mode debugging + timing pitfalls | `tripy/docs/pre0_user_guides/00-introduction-to-tripy.md` (Pitfalls section) | 06-DLFrameworkTRT/Tripy-TensorRT/EagerDebug | **High** | Yes |
| 4 | ResNet50 image classification (notebook) | `tripy/notebooks/resnet50.ipynb` | 06-DLFrameworkTRT/Tripy-TensorRT/ResNet50 | Med-High | Mostly (HF weights+dataset) |
| 5 | NanoGPT text generation (compile + dynamic seq len) | `tripy/examples/nanogpt/` | 06-DLFrameworkTRT/Tripy-TensorRT/NanoGPT | Medium | Mostly (HF gpt2 weights) |
| 6 | INT8/INT4/FP8 quantization via ModelOpt (Linear scales) | `tripy/docs/pre0_user_guides/02-quantization.md` + `tripy/examples/nanogpt/quantization.py` | 04-Feature (Quantization) or 06 Tripy/Quantization | Medium | No (needs modelopt+HF+dataset) |
| 7 | Modules: composable `tp.Module` (MLP) + `load_state_dict` | `tripy/docs/pre0_user_guides/00-introduction-to-tripy.md` (Modules) | 06-DLFrameworkTRT/Tripy-TensorRT/Module | Medium | Yes |
| 8 | Custom op via Quickly-Deployable-Plugin (Triton kernel) | `tripy/docs/pre0_user_guides/03-custom-operations.md` | 05-Plugin (Tripy/QDP) | Med-Low | No (needs triton) |
| 9 | Stable Diffusion txt2img (multi-component pipeline) | `tripy/examples/diffusion/` | 03-Workflow or 06 Tripy/Diffusion | Low-Med | No (HF ckpt, heavy) |
| 10 | Segment Anything v2 (SAM2) image/video | `tripy/examples/segment-anything-model-v2/` | 03-Workflow | Low | No (large, many deps) |

---

## 06-DLFrameworkTRT — proposed new `Tripy-TensorRT/` subsection

- **#1 Hello-Tripy (single best "hello Tripy → TRT")** — `tripy/README.md` Quick Start + `docs/pre0_user_guides/00-introduction-to-tripy.md` — **High.**
  The canonical minimal story: a tiny `tp.Module` (single `tp.Conv` + `tp.relu`), `load_state_dict` with `tp.ones` weights, run **eagerly** (`model(dummy_input)`), then `tp.compile(model, args=[tp.InputInfo(shape=(1,1,4,4), dtype=tp.float32)])` and run the compiled TensorRT executable. No external weights, dataset, or network — pure `tp.ones`/`tp.iota` inputs. This is the recommended flagship intro example. Self-contained; only needs `nvtripy`. Low effort.

- **#2 Compiler / dynamic shapes / serialize** — `tripy/docs/pre0_user_guides/01-compiler.md` — **High.**
  GEGLU module compiled with `tp.InputInfo`; demonstrates dynamic dims via `shape=((1,2,4), 2)` (min/opt/max), `tp.NamedDimension` for tied dynamic dims, and `Executable.save()` / `Executable.load()`. Maps directly onto cookbook themes of dynamic shapes + engine (de)serialization, but expressed in the eager frontend. Self-contained. Low effort.

- **#3 Eager-mode debugging** — `tripy/docs/pre0_user_guides/00-introduction-to-tripy.md` (Pitfalls & Best Practices) — **High.**
  Tripy's distinguishing feature vs. graph-based TRT: interactive eager execution and lazy-evaluation timing pitfalls (tensors only evaluated when used; naive `time.time()` around definition is wrong). A short, high-teaching-value example on "why eager mode, and how to time it correctly." Self-contained.

- **#4 ResNet50 classification** — `tripy/notebooks/resnet50.ipynb` — **Med-High.**
  18-cell notebook building ResNet50 from `tp.Module` blocks (`ResNetConvLayer`/`Embeddings`/`Encoder`/`Classifier`), loading pretrained HF weights, and classifying `comet-team/coco-500` images. Good "real CNN in Tripy" example and already notebook-formatted. Requires HF weights + `datasets`/`transformers`/`torch`; otherwise clean. Medium effort to trim into a `main.py`.

- **#5 NanoGPT generation** — `tripy/examples/nanogpt/` (`example.py`, `model.py`, `weight_loader.py`) — **Medium.**
  GPT-2 implemented as `nvtripy.Module`, compiled with a **dynamic sequence-length** `InputInfo` (`(1,(1,len,padded))`), autoregressive loop interoperating with PyTorch via DLPack (`torch.from_dlpack`). Demonstrates dynamic shapes + framework interop on an LLM. Needs `tiktoken`, `torch`, HF gpt2 weights. Medium effort.

## 04-Feature — quantization

- **#6 Quantization with ModelOpt** — `tripy/docs/pre0_user_guides/02-quantization.md` + `tripy/examples/nanogpt/quantization.py` — **Medium.**
  Post-training calibration with `modelopt.torch.quantization` (INT8/INT4-AWQ/FP8 configs), converting ModelOpt `amax` dynamic ranges into Tripy scales, then loading them into a quantization-enabled `tp.Linear` (`quant_dtype`, `weight_quant_dim`, `input_scale`, `weight_scale`). Also shows manual `tp.quantize`/`tp.dequantize` with the important Q/DQ-rotation caveat (don't `.eval()` between Q and DQ). `nanogpt/quantization.py` has ready INT8/INT4/FP8 config recipes. Complements existing cookbook quantization content from the eager-frontend angle. Not self-contained (modelopt + HF + dataset). Medium effort.

## 06 — modules primitive

- **#7 Modules (MLP)** — `tripy/docs/pre0_user_guides/00-introduction-to-tripy.md` (Organizing Code With Modules) — **Medium.**
  Composable `tp.Module` MLP (`tp.Linear` + `tp.gelu`), `load_state_dict`, eager execution, then `tp.compile`. Slightly higher-level companion to #1 focused on the module system. Self-contained. Low effort. Could be merged into #1.

## 05-Plugin — custom operations

- **#8 Custom op via QDP + Triton** — `tripy/docs/pre0_user_guides/03-custom-operations.md` — **Med-Low.**
  "Increment by 1" plugin using TensorRT's Quickly-Deployable-Plugin (`tensorrt.plugin as trtp`, `@trtp.register`) with an OpenAI Triton kernel and PTX generation, invoked from Tripy. Interesting bridge between Tripy and the plugin story, but depends on `triton` and overlaps the existing plugin section's QDP coverage. Medium effort, narrower audience.

## 03-Workflow — heavy end-to-end demos (lower priority)

- **#9 Stable Diffusion txt2img** — `tripy/examples/diffusion/` — **Low-Med.**
  Full SD pipeline (`models/clip_model.py`, `unet_model.py`, `vae_model.py`) in Tripy, HF weight loading, fp16/fp32, SSIM accuracy check vs. torch (`compare_images.py`). Impressive but large/heavy; only worth it if a Tripy showcase demo is desired. Not self-contained.

- **#10 SAM2 (Segment Anything v2)** — `tripy/examples/segment-anything-model-v2/` — **Low.**
  Meta SAM2 image + video segmentation reimplemented in Tripy (large `sam2/` model tree, checkpoint download, text-to-segmentation demo). Most complex; heavy deps and downloads. Reference-only; not a good cookbook fit.

---

## Recommendation

Add a small, clearly-labeled **experimental** `06-DLFrameworkTRT/Tripy-TensorRT/` subsection seeded by
**#1 (Hello-Tripy)** as the flagship, plus **#2 (compile + dynamic shapes)** and **#3 (eager debugging)**
as the two follow-ups — all three are fully self-contained (only `nvtripy` required) and directly teach
Tripy's value proposition (eager debug → JIT to TensorRT). Defer the model-scale examples (#4-#6) until
the API stabilizes past 0.1.x.
