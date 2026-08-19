# Onnx Graphsurgeon

+ A python library for ONNX compute graph edition, which different from the library *onnx*.

+ Installation: `pip install nvidia-pyindex onnx-graphsungeon`

+ Document [Link](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html)

+ The example code here refers to the NVIDIA official repository about TensorRT tools [Link](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/examples).

+ Function:
  + Modify metadata/node / tensor / weight data of compute graph.
  + Modify subgraph: Add / Delete / Replace / Isolate
  + Optimize: constant folding / topological sorting / removing useless layers.

+ `11` to `13` are the toolbox distilled from a hand-written ChatGLM-6B pipeline
  ([`../../99-Todo/chatglm-6b.md`](../../99-Todo/chatglm-6b.md)) -- the three things graph surgery is
  actually used for on a real model, as opposed to the API tour in `01` to `10`:

  + **`11-mark_output_to_bisect.py`** -- turn "the FP16 answer is wrong" into a node name.
    `mark_graph_output` cuts the graph down to one node and makes its output the graph output, so
    walking forward finds the first node where the precisions diverge. The example's FP16 graph
    silently returns zeros -- no `inf`, no `nan` to warn you -- and the bisect pins the overflow on
    the middle `Mul`, not on the node that produced the wrong final value.
  + **`12-constant_table.py`** -- constant folding cannot touch a subgraph that depends on a runtime
    tensor, even when that tensor only ever takes values from a bounded set. A rotary position
    embedding is the classic case: evaluate `cos(position * inv_freq)` once on the host for every
    reachable position and it becomes a single `Gather`. 4 nodes per step become 1; the catch is
    that the table size is now a hard ceiling the original graph did not have.
  + **`13-merge_two_models.py`** -- join a body and a separately exported head, both by lifting a
    weight out of the second file as a `gs.Constant` and by transplanting its nodes, then append an
    `ArgMax` so the graph returns a class index instead of a score vector.

+ Steps to run.

```bash
./main.sh
```
