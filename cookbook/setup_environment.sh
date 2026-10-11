#!/usr/bin/env bash
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
#
# Install everything the cookbook needs, preferring the local cache in `dependence_packages/`
# over the network.
#
#   ./setup_environment.sh           install: apt packages, pip packages, then `pip install -e .`
#   ./setup_environment.sh --refresh re-download the caches first (needs network), then install
#
# The container is reset between sessions and loses all of this, which is why the caches live in
# the repository rather than in `~/.cache`.

set -o errexit
set -o nounset
set -o pipefail

COOKBOOK_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_PATH="${COOKBOOK_PATH}/dependence_packages"
PIP_CACHE="${CACHE_PATH}/pip"
APT_CACHE="${CACHE_PATH}/apt"
APT_PACKAGE_LIST=(graphviz git-lfs)

B_REFRESH=0
[ "${1:-}" = "--refresh" ] && B_REFRESH=1

mkdir -p "${PIP_CACHE}" "${APT_CACHE}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- apt packages
say "apt packages: ${APT_PACKAGE_LIST[*]}"

if [ "${B_REFRESH}" = "1" ]; then
    # `apt-get download` fetches into the working directory and does not need root, but it does not
    # resolve dependencies, so ask apt for the full URI list of the package and everything it pulls
    # in. `--reinstall` makes apt list packages that are already installed here, which is the whole
    # point: they will not be installed in the next container.
    echo "Refreshing the apt cache (needs network)"
    (
        cd "${APT_CACHE}"
        apt-get install --print-uris -y --reinstall "${APT_PACKAGE_LIST[@]}" \
            | grep -oP "(?<=^')[^']+" \
            | while read -r url; do
                  file="$(basename "${url}")"
                  [ -f "${file}" ] || wget --quiet --show-progress "${url}" -O "${file}"
              done
    )
fi

if ls "${APT_CACHE}"/*.deb > /dev/null 2>&1; then
    echo "Installing $(ls "${APT_CACHE}"/*.deb | wc -l) .deb from the cache"
    # `dpkg -i` needs root. Everything else in this script does not, so the failure is made
    # survivable: the packages it installs are optional for most of the cookbook.
    if ! sudo dpkg --install --skip-same-version "${APT_CACHE}"/*.deb 2>&1 | tail -5; then
        echo "  [SKIP] could not install the .deb files (needs root). Run by hand:"
        echo "         sudo dpkg -i ${APT_CACHE}/*.deb"
    fi
else
    echo "  [SKIP] ${APT_CACHE} is empty. Populate it once, with network and root:"
    echo "         ./setup_environment.sh --refresh"
fi

# ---------------------------------------------------------------- pip packages
say "pip packages from ${PIP_CACHE}"

if [ "${B_REFRESH}" = "1" ]; then
    echo "Refreshing the pip cache (needs network)"
    # Resolve against *this* container: the NGC image already provides torch, tensorrt, numpy and
    # more, and `pip download` would otherwise fetch gigabytes of them. The report lists only what
    # pip would actually install here.
    python3 -m pip install --dry-run --quiet --report "${CACHE_PATH}/report.json" -r "${COOKBOOK_PATH}/requirements.txt"
    python3 - "${CACHE_PATH}/report.json" > "${CACHE_PATH}/resolved.txt" <<'PYEOF'
import json, sys
for item in json.load(open(sys.argv[1])).get("install", []):
    print(f"{item['metadata']['name']}=={item['metadata']['version']}")
PYEOF
    python3 -m pip download --no-deps --dest "${PIP_CACHE}" -r "${CACHE_PATH}/resolved.txt"
    # Build dependencies for anything that arrives as a source distribution, since `--no-index`
    # cannot fetch them later
    python3 -m pip download --no-deps --dest "${PIP_CACHE}" setuptools wheel
    # Turn every source distribution into a wheel now, while there is a network. Offline, pip would
    # have to build it, and building needs build dependencies that `--no-index` cannot reach:
    # `cppimport` fails on `setuptools_scm>=6.2` no matter how many of them are cached.
    for archive in "${PIP_CACHE}"/*.tar.gz "${PIP_CACHE}"/*.zip; do
        [ -e "${archive}" ] || continue
        echo "Building a wheel for $(basename "${archive}")"
        python3 -m pip wheel --no-deps --wheel-dir "${PIP_CACHE}" "${archive}" && rm -f "${archive}"
    done
fi

# `--no-index` first: if the cache is complete this never touches the network. If it is not - a
# different base image, a changed requirements.txt - fall back rather than fail.
if ! python3 -m pip install --no-index --find-links="${PIP_CACHE}" -r "${COOKBOOK_PATH}/requirements.txt"; then
    echo "  Cache is incomplete; falling back to the network for the rest"
    python3 -m pip install --find-links="${PIP_CACHE}" -r "${COOKBOOK_PATH}/requirements.txt"
fi

# ---------------------------------------------------------------- the cookbook itself
say "tensorrt_cookbook (editable)"
# `--no-build-isolation` uses the setuptools already present instead of fetching a fresh copy into a
# throwaway build environment, which is what makes this step work with no network at all.
python3 -m pip install --no-build-isolation --no-index -e "${COOKBOOK_PATH}"

# ---------------------------------------------------------------- report
say "Checking"
python3 - <<'PYEOF'
import importlib.util
import shutil

module_list = [
    ("tensorrt_cookbook", "the cookbook package"),
    ("tensorrt", "TensorRT"),
    ("torch", "pyTorch"),
    ("onnx_graphsurgeon", "onnx-graphsurgeon"),
    ("onnxruntime", "ONNX Runtime"),
    ("polygraphy", "Polygraphy"),
    ("cupy", "CuPy"),
    ("transformers", "transformers"),
    ("graphviz", "graphviz (python binding)"),
]
n_missing = 0
for name, description in module_list:
    b_ok = importlib.util.find_spec(name) is not None
    n_missing += not b_ok
    print(f"  [{'OK  ' if b_ok else 'MISS'}] {description}")
for binary, description in [("dot", "graphviz `dot` binary (apt)"), ("git-lfs", "git-lfs (apt)")]:
    b_ok = shutil.which(binary) is not None
    n_missing += not b_ok
    print(f"  [{'OK  ' if b_ok else 'MISS'}] {description}")
print(f"\n  {n_missing} missing" if n_missing else "\n  Everything is present")
PYEOF
