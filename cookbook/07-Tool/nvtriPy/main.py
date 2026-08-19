# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""nvtripy: an eager-mode Python frontend that compiles to TensorRT.

Tripy (package `nvtripy`, from NVIDIA/TensorRT-Incubator) lets you write a model as a `tp.Module`,
run it **eagerly** while debugging, then `tp.compile` it into a TensorRT `Executable`.

This file is only the driver. It exists because **`pip install nvtripy` into the cookbook's
environment breaks the cookbook**: nvtripy 0.1.7 depends on `tensorrt-cu12 10.x` and
`mlir-tensorrt ... cuda12.trt109`, which shadow the system TensorRT 11 and downgrade NumPy.
Verified the hard way, see `README.md`. So the real example, `tripy_cases.py`, runs inside a
private virtual environment that this script creates.

+ Steps to run.

```bash
python3 main.py
```
"""

import subprocess
import sys
from pathlib import Path

import tensorrt as trt

current_path = Path(__file__).parent
venv_path = current_path / ".venv"
venv_python = venv_path / "bin" / "python"
PACKAGE_INDEX = "https://nvidia.github.io/TensorRT-Incubator/packages.html"

def prepare_venv() -> bool:
    """Create the private environment and install nvtripy into it. False if that is not possible."""
    if venv_python.exists():
        print(f"    reusing {venv_path.name}/")
        return True

    print(f"    creating {venv_path.name}/ and installing nvtripy (first run only, needs network)")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True, capture_output=True)
        subprocess.run([str(venv_path / "bin" / "pip"), "install", "-q", "nvtripy", "-f", PACKAGE_INDEX], check=True, capture_output=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    could not install nvtripy ({type(e).__name__}); this example needs network access")
        return False
    return True

def report_isolation():
    """The point of the venv: two TensorRT versions in one container, neither disturbing the other."""
    cookbook_version = trt.__version__
    command = "import tensorrt, nvtripy; print(tensorrt.__version__, nvtripy.__version__, tensorrt.__file__)"
    process = subprocess.run([str(venv_python), "-c", command], capture_output=True, text=True)
    venv_version, tripy_version, venv_tensorrt_path = process.stdout.split()

    print(f"    cookbook interpreter : TensorRT {cookbook_version}")
    print(f"    {venv_path.name}                 : TensorRT {venv_version} (nvtripy {tripy_version}), from {venv_tensorrt_path}")
    assert venv_version != cookbook_version, "the venv is not actually isolated"
    print("    -> nvtripy brings its own TensorRT; installing it next to the cookbook's would replace TensorRT 11")
    return

if __name__ == "__main__":
    print(f"{'=' * 30} Start [prepare]")
    if not prepare_venv():
        print("Skipped")
        sys.exit(0)
    report_isolation()
    print(f"{'=' * 30} End   [prepare]")

    # `nvtripy` logs one "WARNING The logger passed into createInferRuntime differs ..." line per
    # engine it builds, which buries the output; drop those lines and keep everything else.
    process = subprocess.run([str(venv_python), str(current_path / "tripy_cases.py")], capture_output=True, text=True, cwd=current_path)
    print(process.stdout, end="")

    error_line_list = [line for line in process.stderr.splitlines() if not line.startswith("WARNING The logger passed into")]
    if error_line_list:
        # Worth showing: the out-of-range shape in `case_dynamic_shapes` is rejected by TensorRT
        # itself, and the message names the profile it failed against.
        print("\nWhat TensorRT wrote to stderr underneath Tripy:")
        for line in error_line_list:
            print(f"    {line}")
    assert process.returncode == 0, "the Tripy cases failed"
