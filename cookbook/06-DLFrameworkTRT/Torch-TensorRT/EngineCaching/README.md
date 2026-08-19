# Engine Caching

+ Reuse built TensorRT engines across compilations, sessions and weight changes.

+ Steps to run.

```shell
python3 main.py
```

## Why

Building the engine is the expensive part of Torch-TensorRT. AOT
(`torch_tensorrt.dynamo.compile`) pays it once per process; JIT
(`torch.compile`) pays it again every time dynamo invalidates a graph. Engine
caching writes each built engine to disk under a hash of the PyTorch sub-graph,
so a later compilation loads it instead of rebuilding.

Two switches, both required, plus one precondition:

```python
torch_tensorrt.dynamo.compile(
    exported, example,
    immutable_weights=False,     # precondition: a cached engine is reused by REFITTING it
    cache_built_engines=True,    # write into the cache
    reuse_cached_engines=True,   # read back from it
    engine_cache_dir="...", engine_cache_size=1 << 30,
)
```

`immutable_weights=False` is not optional. A cache hit does not hand back a
finished engine, it hands back an engine that then gets refitted with the current
weights, and only a refittable engine can do that.

## Measured (resnet18, 8x3x224x224, H100)

| case | no cache | build + save | load from cache | speed-up |
| --- | ---: | ---: | ---: | ---: |
| AOT `dynamo.compile` | 25.5 s | 23.0 s | **0.93 s** | **27x** |
| JIT `torch.compile` | 27.4 s | 25.0 s | **1.49 s** | **18x** |
| AOT + custom RAM cache | 22.5 s | 23.0 s | **0.91 s** | **25x** |

Note the middle column: enabling the cache on a *cold* cache is not faster, it is
slightly slower, because the engine has to be serialised and written. The win
only arrives on the third compilation.

## The cache key ignores weights

The hash covers the lowered sub-graph, not the parameter values. `case_weight_agnostic`
replaces **every** parameter and compiles again:

```
every parameter replaced, then recompiled in 0.91 s
engines in cache: 1 -> 1 (unchanged, so the hash ignored the weights)
max |eager - TensorRT| = 1.634e-07 -> the engine was refitted, not just reused
```

That is what makes caching useful for fine-tuned or LoRA-style checkpoints that
share one architecture. The numeric check is the important half: a cache hit that
failed to refit would silently return the *previous* weights' answer, and nothing
else in the pipeline would notice.

This is also why the example uses `models.resnet18(weights=None)` -- pretrained
weights would only add a download without changing anything the cache does.

## Custom cache back end

Subclass `BaseEngineCache` and implement `save` / `load` to put engines somewhere
else (shared storage, an object store, or just RAM as here), then pass it as
`custom_engine_cache=`. `case_custom_cache` reports `1 blob, 1 save, 1 hit,
2 miss` and `0 files on disk`, which is the whole contract.

## Timing cache is a different thing

`main.py` deletes `TIMING_CACHE_PATH` before every compilation. That cache stores
kernel *timings*, not engines; leaving it in place would speed up the uncached
baseline too and blur the comparison. Only the measurement needs this -- in
production leave both caches alone.
