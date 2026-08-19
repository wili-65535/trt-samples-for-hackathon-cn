# 03-Workflow

+ Common workflow of using TensorRT from DL frameworks.

## Workflow of JAX -> ONNX -> TensorRT

+ A workflow of: train model in JAX, export model to ONNX through `jax2tf` + `tf2onnx`, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of Mindspore -> ONNX -> TensorRT

+ A workflow of: train model in Mindspore, export model to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of OneFlow -> ONNX -> TensorRT

+ A workflow of: train model in OneFlow, export model to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of Paddlepaddle -> ONNX -> TensorRT

+ A workflow of: export trained model from Paddlepaddle to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of TensorFlow2 -> ONNX -> TensorRT

+ A workflow of: export trained model from TensorFlow2 to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of ModelOptimizer post-processing -> TensorRT

+ A workflow of: train a model in pyTorch, post-process it into a reduced-precision model with **NVIDIA TensorRT Model Optimizer (ModelOptimizer)**, then parse the resulting ONNX in TensorRT, build a **strongly-typed** engine and do inference.

## Workflow of pyTorch -> ONNX -> TensorRT

+ A workflow of: export trained model from pyTorch to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of pyTorch (KV cache) -> ONNX -> TensorRT

+ The same road for an **autoregressive** model, where the KV cache is the interesting part: gpt2's
  decode graph has 51 I/O tensors (gpt2-medium's has **99**), and wrapping the model so that the
  whole cache is one packed tensor brings that to **5**. The engine's cache input and output are
  then bound to **one allocation** — which works only because the cache is repacked sequence-first;
  in the layout `transformers` uses, a shared buffer is silently wrong. Greedy sampling is moved
  into the graph, so the engine returns a token id instead of 50257 floats per step.

## Workflow of pyTorch -> TensorRT

+ A workflow of: rebuild model in TensorRT with exported weights, build TensorRT engine and do inference.
