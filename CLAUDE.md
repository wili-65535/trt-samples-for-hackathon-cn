# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**The container is reset between sessions.** Everything *installed* is lost — the pip packages, the
apt packages, and `tensorrt_cookbook` itself — while the repository under `/work` survives. So the
first thing to do in a new session is put the environment back:

```bash
cd cookbook/
./setup_environment.sh
```

That installs the apt packages, the pip packages and `tensorrt_cookbook` (editable), preferring the
local caches in `cookbook/dependence_packages/` over the network. With the caches warm it takes
about **6 seconds** and touches the network not at all. It ends with a table of what is present, so
a partial setup is visible rather than assumed:

```txt
  [OK  ] the cookbook package
  [OK  ] TensorRT
  ...
  [OK  ] graphviz `dot` binary (apt)
  Everything is present
```

If something reports `MISS`, or `requirements.txt` changed, refresh the caches first — this needs
the network, and the `.deb` step needs `sudo`:

```bash
./setup_environment.sh --refresh
```

The caches are **not** tracked by git (see `cookbook/dependence_packages/README.md`). They are a
local accelerator: a fresh clone downloads everything once, and only later setups on that machine
are offline.

**Recommended environment:** NVIDIA Docker image `nvcr.io/nvidia/pytorch:26.07-py3` (Python 3.12,
CUDA 13.3, TensorRT 11.0). The image already provides torch, TensorRT, numpy and much else, which is
why the pip cache holds only the ~23 packages that are actually missing rather than the whole
dependency tree.

### Doing it by hand

```bash
cd cookbook/
export TRT_COOKBOOK_PATH=$(pwd)   # Optional: the package auto-discovers this when it can
pip install -r requirements.txt
pip install -e .                  # Editable install for development

# Release build:
python -m pip install -U build && python -m build && pip install dist/*.whl
```

Two flags in `setup_environment.sh` are not decoration, and hand-installs offline need them too:
`pip install --no-index --find-links=dependence_packages/pip` to stay off the network, and
`pip install --no-build-isolation -e .` because build isolation otherwise downloads a fresh
`setuptools` into a throwaway environment and fails when there is no network.

## Common Commands

**Run a single example:**
```bash
cd 02-API/Layer/Cast
python3 main.py
```

**Run examples via the unified runner:**
```bash
python3 tests/run_tests.py                          # run all discoverable examples
python3 tests/run_tests.py --case 02-API/Layer/Cast # run one specific example
python3 tests/run_tests.py --include "02-API/**"    # glob filter
python3 tests/run_tests.py --list                   # list without running
python3 tests/run_tests.py --dry-run                # print commands only
python3 tests/run_tests.py --tags plugin            # filter by tag
python3 tests/run_tests.py --since HEAD             # only what your changes affect
python3 tests/run_tests.py --since HEAD --explain-since   # ... and why each file selected it
python3 tests/run_tests.py --framework tensorflow   # ONLY the TensorFlow examples
```

**Run the full test suite (unified runner):**
```bash
python3 tests/run_tests.py
```

**Run pytest tests (NetworkSerialization):**
```bash
pytest tests/NetworkSerialization/
pytest tests/NetworkSerialization/test_convolution.py  # single test file
```

**Linting / formatting:**
```bash
pre-commit run --all-files     # run all hooks (yapf, clang-format, autoflake, codespell…)
pre-commit install             # install hooks into .git/hooks
```

**Regenerate README.md:**
```bash
python3 build-README.py
```

**Add SPDX license headers to new files:**
```bash
python3 build-Copyright.py
```

## Code Architecture

### `tensorrt_cookbook/` — shared utility package

All examples import from this package. Key modules:

- **`utils_class.py`** — TensorRT wrapper classes that are the primary abstraction used across examples:
  - `TRTWrapperV1` / `TRTWrapperV2` — high-level build + inference workflows
  - `TRTWrapperDDS`, `TRTWrapperShapeInput`, `TRTWrapperV2Torch` — specialised variants
  - `CookbookLogger`, `CookbookProfiler`, `CookbookErrorRecorder` — diagnostic helpers
  - `CookbookGpuAllocator`, `CookbookOutputAllocator` — custom memory management
- **`utils_cookbook.py`** — cookbook infrastructure: path resolution (`cookbook_path`), the `case_mark` decorator used by every `main.py`, logging, API-coverage inspection, README/copyright generation
- **`utils_function.py`** — framework-agnostic data helpers only: maths, array printing/comparison, dtype casting. Depends on nothing heavier than numpy/torch/tensorrt
- **`utils_network.py`** — building and inspecting `INetworkDefinition`: layer-type helpers (`layer_dynamic_cast`, …), `parse_onnx`, `print_network`, `export_network_as_onnx`
- **`utils_engine.py`** — inspecting a built engine: plan-file header (`print_engine_information`), engine/context I/O tables, engine-information JSON → ONNX
- **`utils_workflow.py`** — `check_torch_operator`: run one Torch model through Torch → ONNX → ONNX-Runtime → Polygraphy → TensorRT and report which stage breaks
- **`utils_network_serialization.py`** — network serialization/deserialization (also tested by `tests/NetworkSerialization/`)
- **`utils_onnx.py`** — ONNX / onnx-graphsurgeon utilities. Deliberately does **not** import `tensorrt`
- **`utils_plugin.py`** — plugin development helpers
- **`utils_engine_explorer.py`** — TREx-derived engine profiling/visualisation

When adding a helper, place it by dependency direction, not by topic name: `utils_function` may not
import from the package, `utils_onnx` may not import `tensorrt`, and anything that touches a built
engine belongs in `utils_engine.py`. Because `__init__.py` re-exports every module with `import *`,
moving a function between these modules does not affect examples.

### Numbered example sections

| Directory            | Content                                                     |
| -------------------- | ----------------------------------------------------------- |
| `00-Data/`           | Dataset and model preparation                               |
| `01-SimpleDemo/`     | Minimal end-to-end TensorRT examples                        |
| `02-API/`            | TensorRT API coverage (Builder, Network, Layers, etc.)      |
| `03-Workflow/`       | Framework-to-TRT pipelines (PyTorch/TF/Paddle → ONNX → TRT) |
| `04-Feature/`        | Advanced features: quantization, profiling, caching, safety |
| `05-Plugin/`         | Custom plugin development and ONNX parser integration       |
| `06-DLFrameworkTRT/` | Torch-TensorRT                                              |
| `07-Tool/`           | External tools: trtexec, Polygraphy, Netron, ONNX utilities |
| `08-Advance/`        | Advanced patterns: CUDA graphs, multi-device, multi-stream  |
| `90-Misc/`           | General utilities not specific to TensorRT                  |

Each leaf directory is independently runnable. The standard entry point is `main.py`.

### Test orchestration

Examples are discovered and run by `tests/run_tests.py`. Discovery rules:
1. If a directory contains `unit_test.yaml`, its `run:` commands are used.
2. If a directory has `main.py` but no `unit_test.yaml`, the runner defaults to `python3 main.py > log-main.py.log`.
3. Paths matched by `tests/skip_tests.yaml` are globally excluded from discovery.
4. Presence of a `.skip_unit_test` file disables that directory entirely.

`unit_test.yaml` fields: `enabled`, `tags`, `timeout`, `env`, `pre`, `run`, `post`, `clean`. See `tests/run_tests.py` for the full spec.

**Two selectors that change what a default sweep means:**

+ **`--since REF`** runs only the examples your changes affect, which turns a ~40 minute sweep into
  seconds. A changed file is attributed to the example directory that owns it; a change under
  `tensorrt_cookbook/`, `include/`, `tests/` or `00-Data/`, or to a shared root file, selects
  everything; a `.md`/`.txt` selects nothing; and **a path no rule recognises selects everything**,
  because widening is the safe direction. `--explain-since` prints the reason per file.
+ **`--framework NAME`** is how the TensorFlow / JAX / PaddlePaddle examples run. They are skipped
  by default so a torch-only container does not watch them fail, and this flag is **exclusive**:
  it runs those and nothing else. The workflow is "default sweep, then `pip install tensorflow`,
  then `--framework tensorflow`". `--framework all` covers every gated framework.

## Code Style

- **Python**: formatted by YAPF (config in `.style.yapf`); isort is intentionally disabled to avoid conflicts.
- **C++/CUDA**: formatted by clang-format v16 (LLVM style, config in `.clang-format`).
- **CMake**: formatted by cmake-format.
- All source files require an SPDX Apache-2.0 header. Run `python3 copyright.py` to add/update headers.
- Pre-commit hooks enforce all of the above automatically on commit.
