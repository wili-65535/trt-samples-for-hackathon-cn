# 05-Plugin

+ Examples of using TensorRT plugins.

## APIs

+ Example of showing all APIs of plugin.

## AliasedIOPlugin

+ A Python plugin that writes into one of its own inputs, using the aliased-I/O capability of `IPluginV3OneBuildV2`.

## Basic Example

+ Basic example of using `PluginV3` to add a scalar onto the input tensor.

## Basic Example - V2DynamicExt (deprecated)

+ The same as Basic Example, but use `IPluginV2DynamicExt` class (deprecated).

## Basic Example - static register (deprecated)

+ The same as Basic Example, but register the plugin in a static way (deprecated).

## CuteDSLPlugin

+ An `IPluginV3` whose kernel is written in **CuteDSL**, CUTLASS's Python DSL.

## Data Dependent Shape

+ Example of using a Data-Dependent-Shape plugin to move all non-zero elements to the left side.

## INT8-QDQ-Plugin

+ Minimal example combining QDQ layers with a plugin insertion point.

## Identity plugin

+ Basic example of using `PluginV3` to copy input to output.

## In-Place Plugin

+ The same as Basic Example, but use in-place plugin (input and output tensor share the same buffer).

## MigrationV2toV3

+ Migrate a Python plugin from the deprecated `IPluginV2DynamicExt` to `IPluginV3`.

## Multi-Version

+ The same as BasicExample, but multiple versions of the plugin are provided to be chose at runtime.

## NcclPlugin

+ Minimal TensorRT `PluginV3` + NCCL `send/recv` example.

## ONNX Parser and Plugin

+ Example of combinating the usage of model from ONNX and plugin.

## Pass Host Data

+ Example of passing a host pointer (pointing to anything like array, structure or even nullptr) into plugin at runtime.

## Plugin Inside Engine - C++

+ Example of serializing a plugin inside a TensorRT engine (no `.so` needed at runtime) using C++ APIs.

## Plugin Inside Engine - Python

+ Example of serializing a plugin inside a TensorRT engine (no `.so` needed at runtime) using C++ APIs.

## PythonPlugin

+ The same as BasicExample, but we make the workflow totally in Python script.

## Quick Deployable Python plugin

+ The same as BasicExample, but use decorated functions to simplift the workflow in Python plugin.

## Resource

+ Example of using TensorRT `IPluginResource` to share information between two `PluginV3` layers in one network.

## Shape Input Tensor

+ Example of sending a shape input tensor into plugin to reshape another execution tensor by the values of it.

## Tactic+TimingCache

+ The same as BasicExample, but we use our own tactics and timing-cache in the plugin.

## Triton AOT Plugin

+ Ship an **OpenAI-Triton** kernel inside a C++ `IPluginV3` with no Python, no Triton and no JIT at
  run time: `triton.tools.compile` bakes the cubin into C source, `triton.tools.link` gives it a
  stable symbol name. The AOT counterpart of the Triton backend in `PythonPlugin`. Also a generator
  that writes the whole plugin from a kernel plus a small spec, and maps several AOT variants onto
  TensorRT **tactics** so the builder times them and keeps the best (**1.83x** over a fixed choice
  here). Watch out for the `fp32` argument that the generated C declares as `double` and the kernel
  then reads as `0`, with a successful launch either way.

## Use cuBLAS

+ Example of using cuBLAS in plugin.

## UseFP16

+ The same as BasicExample, but enabling FP16 mode.
