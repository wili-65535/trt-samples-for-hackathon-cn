# dependence_packages

+ Local caches so that setting up the cookbook does not need the network.

```bash
cd <PathToCookbook>
./setup_environment.sh             # install everything, cache first
./setup_environment.sh --refresh   # re-download the caches (needs network), then install
```

## Why this exists

The container is reset between sessions. What is lost is everything *installed* — the pip packages
under `dist-packages`, the apt packages, and `tensorrt_cookbook` itself. What survives is `/work`,
which is where this repository lives.

So the caches are kept here rather than in `~/.cache`, and `setup_environment.sh` reinstalls from
them in seconds instead of re-downloading half a gigabyte.

## What is cached

| Directory | Holds | Size |
| :-------- | :---- | :--- |
| `pip/` | wheels for everything `pip install -r requirements.txt` would fetch | ~455 MB, 25 files |
| `apt/` | `.deb` files for `graphviz` and `git-lfs`, with their dependencies | populated by `--refresh` |

**Neither is committed**, and that is deliberate rather than an oversight. These are a *local*
accelerator:

+ A fresh clone has empty caches, so **the first setup on a machine still downloads everything**.
  `./setup_environment.sh --refresh` is that first run.
+ Every setup after that on the same machine is offline, which is the point — the container is reset
  often and the packages have to go back in each time.

Tracking them would add ~455 MB of wheels to git history permanently and would buy nothing, since
`/work` already survives the resets that wipe the installed packages.

## The pip cache is resolved against *this* image, on purpose

`pip download -r requirements.txt` would fetch the whole dependency tree, including the several
gigabytes of `torch`, `tensorrt` and `numpy` that the NGC image already provides. Instead
`--refresh` asks pip what it would actually install here:

```bash
pip install --dry-run --report report.json -r requirements.txt
```

which on this image is **23 packages**, and downloads exactly those.

In principle that ties the cache to one base image. Measured, it does not: comparing `pip freeze`
from `nvcr.io/nvidia/pytorch:26.05-py3` (TensorRT 10.16.1.11) and `26.06-py3` (TensorRT 11.0.0.114),
the two images differ in **61** package versions — `cuda-python` 13.2.0 to 13.3.1, `datasets` 4.8.5
to 5.0.0, `nvidia-modelopt` 0.43.0 to 0.44.0, and so on — while **all 25 packages this cache holds
are at identical versions in both**.

That is not a coincidence. The 61 differences are all packages the *image* ships, which move when
the image does; the cached set is what the image does **not** ship, so those follow PyPI and the
`requirements.txt` pins rather than the image. To redo this after the next image bump, take a
`pip freeze` inside each image and diff it against the wheel names in `pip/`.

Should a future image break that, the script falls back to the network for whatever is missing
rather than failing, so it degrades instead of breaking.

## Source distributions are turned into wheels while the network is up

`--refresh` runs `pip wheel` over anything that arrived as a `.tar.gz` and deletes the archive. This
is not tidiness. Installing a source distribution means *building* it, and a build needs build
dependencies that `--no-index` cannot go and fetch:

```txt
ERROR: Could not find a version that satisfies the requirement setuptools_scm>=6.2 (from versions: none)
ERROR: Failed to build 'cppimport' when installing build dependencies for cppimport
```

Caching `setuptools` and `wheel` does not fix this, because the next source distribution wants
something else. Building once, online, removes the problem at the root. `cppimport` is the only
package in the current set that needs it.

For the same reason the final step uses `--no-build-isolation`:

```bash
pip install --no-build-isolation --no-index -e .
```

Without it, pip creates a throwaway build environment and downloads a fresh `setuptools` into it,
which fails with no network — even though a perfectly good one is already installed.

## The apt half needs root

`.deb` installation is the one step that does. `--refresh` downloads them without root, using
`apt-get install --print-uris` to resolve the dependency list and `wget` to fetch it, but
`dpkg --install` needs `sudo`. If that fails the script says so and carries on; both packages are
optional for most of the cookbook.

+ `graphviz` provides the `dot` **binary**. `pip install graphviz` does not — that package only
  shells out to it. Needed to render a picture in `07-Tool/EngineVisualization` and by
  `07-Tool/trex/11-ProcessEnginePipeline`.
+ `git-lfs` is usually already present in the NGC image.

## Checking

The script finishes with a table, so a partial setup is visible rather than assumed:

```txt
  [OK  ] the cookbook package
  [OK  ] TensorRT
  [OK  ] onnx-graphsurgeon
  ...
  [MISS] graphviz `dot` binary (apt)
  1 missing
```
