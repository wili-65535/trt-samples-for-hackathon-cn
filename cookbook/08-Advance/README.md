# 08-Advance

+ Tool combinations of using TensorRT and other CUDA / pyTorch features.

## C++ Static Compilation

+ Static compilation the TensorRT engine into a executable file.

## CUDA graph

+ Use CUDA graph to solve launch bound issue (usually appear in small TensorRT engines).

## Empty Tensor

+ Zero-volume tensors where they actually occur: a detector that keeps no box, an empty batch from
  a serving stack. TensorRT handles them; the example measures the three ways the surrounding
  program does not (`cudaMalloc(0)` returns NULL and `enqueueV3` then silently refuses to run,
  `MAX`/`AVG` over an empty axis give `-inf`/`NaN`, and a profile whose minimum is 1 rejects the
  shape through a return value).

## Green Context

+ Partition the SMs of one GPU from inside the process (CUDA 12.4+) and run an engine on a stream
  bound to the partition: the MIG experiment without MIG, no root and no restart. Measured here:
  the partition scales latency as expected and a full-GPU partition is free (1.03x); a noisy
  neighbour costs a latency-critical engine 7.51x at p95 unpartitioned and 2.41x partitioned;
  **TensorRT's auxiliary streams escape the partition, and `max_aux_streams` defaults to -1**;
  and building inside the partition is **19% faster** than building on the whole GPU, against a
  build-to-build spread of 0.3%.

## MIG

+ **A note, not a runnable example.** MIG is host-side configuration and no TensorRT API behaves
  differently because of it, so the directory only records the one consequence that is about
  TensorRT: an engine is tuned for the SM count and memory it sees at build time, so it must be
  built on the MIG profile it will be served on (114 SM / 79 GiB whole-GPU versus 14 SM / 9.75 GiB
  on a `1g.10gb` slice). The measurement backing that up needs a MIG-enabled machine and is listed
  as open.

## Multi Context

+ Use multiple execution context to do inference.

## Multi Device

+ Example to show `engine_bytes` can be shared cross devices, but `engine` can not.

## Multi Optimization Profile

+ Use multiple Optimization-Profile to do inference.

## Multi-Stream

+ Use one execution context with multiple CUDA stream.

## Stream and Async

+ Example  to use pinned memory.

##

+ Safety mode is only for Drive Platform (QNX)，https://github.com/NVIDIA/TensorRT/issues/2156

## Subgraph

+ Use cases of parsing ONNX file with subgraph into TensorRT.

## TensorRT Graph Surgeon

+ Edit a network at the `INetworkDefinition` level, after the ONNX parser and before the builder:
  walk the layers the parser produced (3 ONNX nodes -> 7 TensorRT layers here), append a layer,
  delete one by rewiring its consumer (**there is no `remove_layer`**), and replace a layer with
  your own plugin on a model TensorRT already runs fine. That last one is bit-identical and costs
  **2.13x**, because the replaced op had been fused into its neighbours.

## Steps to run
