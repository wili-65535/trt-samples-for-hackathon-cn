# 07-Tool

+ Tools of using TensorRT beyond the original APIs.

## Check pyTorch Operator

+ A template to check whether a operator in pyTorch can be parsed into TensorRT.

## Context Printer

+ Print information of input / output shape set for the context.

## Debug Utils

+ Demonstrates migrated runtime/debug helpers in `tensorrt_cookbook`:

## Engine Printer

+ Print information of layers and tensors in the TensorRT engine.

## FP16 Tuning

+ Thanks Xuewei Li for providing the solution

## List APIs

+ List all the APIs in TensorRT package.

## MPI Utils

+ Complete MPI utility example based on `tensorrt_cookbook` wrappers.

## Netron

+ A visualization tool for neural-network graphs, including ONNX and many other formats.

## Network Printer

+ Print information of layers and tensors in the network.

## Network Serialization and Deserialization

+ Serialize a network into a json file, and deserialize it back into a INetwork.

## Nsight Deep Learning Designer

+ An integrated development environment that helps developers efficiently design and optimize deep neural networks for high-performance inference.

## Nsight Systems

+ Program performance analysis tool (replacing the old performance analysis tools nvprof and nvvp).

## Onnx

+ An open source format for AI models, both deep learning and traditional ML.

## Onnx Graphsurgeon

+ A python library for ONNX compute graph edition, which different from the library *onnx*.

## Onnx Weight Separator

+ A tool to separate weights from a ONNX file, usually for visualization of a remote large ONNX file.

## Onnx Runtime

+ Run ONNX Runtime with TensorRT Execution Provider (EP), and compare latency with CUDA EP.

## Polygraphy - Client tool

+ CLI tool of polygraphy (deep learning model debugger).

## TritonServerDeploy

+ Minimal skeleton to generate a TensorRT plan and Triton model repository layout.

## nvtriPy

+ An **eager-mode Python frontend for TensorRT** (package `nvtripy`): write the model as a
  `tp.Module`, run it immediately while debugging, then `tp.compile` it into a TensorRT
  `Executable` (dynamic shapes via `InputInfo`, `save`/`load` the artifact). Pre-1.0, so it is a
  tour rather than a stable interface. **`pip install nvtripy` into the cookbook's environment
  replaces TensorRT 11 with its own TensorRT 10 and downgrades NumPy**, so the example installs it
  into a private virtual environment instead.

## nvtx

+ Use NVIDIA®Tools Extension SDK to add mark in timeline of Nsight systems.

## trex - TensorRT Engine Explorer

Explore the structure and performance of a **built** TensorRT engine by analysing

## trtexec

+ Command-line tool of TensorRT, attached with an end-to-end performance test tool.
