# Green Context

Give a TensorRT engine a fixed slice of one GPU's SMs, from inside the process.

+ Steps to run.

```bash
python3 main.py
```

A green context (CUDA 12.4+) partitions the SMs of a GPU and hands out a stream bound to the
partition; everything launched on that stream is confined to those SMs. TensorRT needs no API for
it — `execute_async_v3(green_stream)` is the whole integration — which is exactly why the two
surprises below are easy to walk into.

All numbers measured on H100 PCIe 80 GB (114 SM), TensorRT 11.1.0.106, CUDA 13.3.

## The partition is real, and free

The engine is a chain of eight 1024x1024 matrix multiplies, chosen to be SM bound:

| Stream | Latency | vs whole GPU | SM ratio |
| ------ | ------- | ------------ | -------- |
| default (114 SM) | 0.137 ms | 1.00x | 1.00x |
| green, 16 SM | 0.546 ms | 3.99x | 7.12x |
| green, 32 SM | 0.299 ms | 2.18x | 3.56x |
| green, 64 SM | 0.166 ms | 1.21x | 1.78x |
| green, 114 SM | 0.141 ms | **1.03x** | 1.00x |

The last row matters: a partition containing every SM costs nothing, so the mechanism itself adds
no overhead. The scaling is sub-linear (16 SM is 7.1x fewer SMs but only 4.0x slower) because a
smaller partition still gets the whole L2 and memory system.

Partitions are not arbitrary: `minSmPartitionSize` and `smCoscheduledAlignment` are both 8 here, so
`cuDevSmResourceSplitByCount` rounds the request. Ask the returned resource what you got.

## What it is for: the noisy neighbour

A latency-critical engine, with a throughput-hungry engine running beside it in the same process:

| Setup | median | p95 |
| ----- | ------ | --- |
| alone | 0.035 ms | 0.039 ms |
| background job, both on the default stream | 0.146 ms (**4.12x**) | 0.296 ms (**7.51x**) |
| background job, green 32 SM / 82 SM | 0.066 ms (1.86x) | 0.095 ms (2.41x) |

This is the experiment MIG is usually sold with, done **without** MIG: same process, no root, no
host configuration, no container restart, and the partitions can be created and destroyed at will.
See [`../MIG/README.md`](../MIG/README.md) for what MIG buys that this does not (hardware-level
memory and bandwidth isolation, separate fault domains, cross-container assignment).

## The hole: auxiliary streams escape the partition

TensorRT runs independent branches of a network on **auxiliary streams**, and those are created
from the current context, not from the green one. The background engine below is pinned to its own
82 SM, the latency job to a disjoint 32 SM, and yet:

| Background engine built with | latency median | p95 |
| ---------------------------- | -------------- | --- |
| `max_aux_streams = 0` | 0.064 ms (1.18x) | 0.102 ms (1.79x) |
| `max_aux_streams = 4` | 0.112 ms (2.06x) | 0.193 ms (**3.40x**) |
| default (`-1`, TensorRT decides) | 0.131 ms (2.41x) | 0.185 ms (**3.25x**) |

The default is `-1`, so **nobody has to ask for this to happen**. If the isolation matters, build
the engines that live in a partition with `max_aux_streams = 0` and pay for it in intra-engine
concurrency — or measure the p95, which is where the leak shows up first.

## The free 19%: build where you will run

Building the engine while the green context is current makes it faster on that partition:

```txt
primary context: cudaGetDeviceProperties.multiProcessorCount=114, cuCtxGetDevResource=114
green context  : cudaGetDeviceProperties.multiProcessorCount=114, cuCtxGetDevResource=32
built on the whole GPU (1), run on 32 SM:   0.303 ms
built on the whole GPU (2), run on 32 SM:   0.303 ms
built inside the partition, run on 32 SM:   0.246 ms
build-to-build spread of two identical builds: 0.001 ms; partition effect: 0.057 ms
```

Two builds under identical conditions differ by 0.001 ms, so the 0.057 ms (19%) is **57x the
noise** and not an artefact (a second, independent run reproduced it: 0.301 / 0.301 / 0.243 ms) — the two control builds are in the example for exactly this reason.

The mechanism is worth understanding, because the obvious explanation is wrong: TensorRT cannot
*ask* how large the partition is. `cudaGetDeviceProperties.multiProcessorCount`, which is what it
reads, still reports 114 inside the green context; only the driver-level `cuCtxGetDevResource`
knows about the 32. What adapts is the **tactic search**, which is empirical: candidate kernels are
timed in whatever context is current, so a build done inside the partition measures the partition's
real behaviour and picks different winners.

This is the measured version of the argument in [`../MIG/README.md`](../MIG/README.md) — build on
the profile you deploy on.

## Lifetime trap

An engine built while a green context is current owns CUDA resources belonging to that context.
Destroying the context first makes TensorRT's destructors fail with
`Error Code 1: Cuda Runtime (In deallocate ...)` and then takes the process down with a SIGSEGV
**at exit**, far from the mistake. Either let the context outlive the TensorRT objects (what case 4
does) or destroy the objects first. Cases 1-3 can free their partitions normally, because their
engines live in the primary context and only the *stream* comes from the partition.

## Related

+ [`../MIG/README.md`](../MIG/README.md) — the hardware-partitioning alternative, and why this
  directory is a note instead of an example.
+ [`../MultiStream/`](../MultiStream/README.md), [`../MultiContext/`](../MultiContext/README.md) —
  sharing one GPU without partitioning it.
+ [`../../04-Feature/AuxStream/`](../../04-Feature/AuxStream/README.md) — what `max_aux_streams`
  does when nothing is partitioned.
