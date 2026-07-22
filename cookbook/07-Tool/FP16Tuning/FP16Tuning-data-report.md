# FP16 Tuning Report

+ Generated at 2026-07-22 00:46:19

+ Layers specified [  0]: []
+ Layers skipped [  0] : []
+ Layers forced in FP32 [  0]: []
+ Layers could be tuned [  7]: ['Pure FP32 🟩', 'Pure FP16 🟦', 'FP16 + ForceFP32 🟪', 'node_conv2d', 'node_conv2d_1', 'node_linear', 'node_linear_1']
+ Layers actually tune in this session: 4

+ Focus tensor for BestAcc ranking: y
|   No. | LayerName           | TensorName   |   GPUTime (ms) |   MaxAbsError |   MeanAbsError | BestPerf   | BestAcc   |
|-------|---------------------|--------------|----------------|---------------|----------------|------------|-----------|
|     1 | Pure FP32 🟩        | y            |          0.038 |             0 |              0 |            |           |
|     1 | Pure FP32 🟩        | z            |          0.038 |             0 |              0 |            |           |
|     2 | Pure FP16 🟦        | y            |          0.034 |     0.0066223 |      0.0025724 |            |           |
|     2 | Pure FP16 🟦        | z            |          0.034 |             0 |              0 |            |           |
|     3 | FP16 + ForceFP32 🟪 | y            |          0.034 |     0.0066223 |      0.0025724 |            |           |
|     3 | FP16 + ForceFP32 🟪 | z            |          0.034 |             0 |              0 |            |           |
|     4 | node_conv2d         | y            |          0.036 |     0.0066223 |      0.0014007 | 1 🔴       | 2 🔴      |
|     4 | node_conv2d         | z            |          0.036 |             0 |              0 | 2 🔴       |           |
|     5 | node_conv2d_1       | y            |          0.038 |     0.0066223 |      0.0025724 | 7 🟠       | 3 🔴      |
|     5 | node_conv2d_1       | z            |          0.038 |             0 |              0 | 8 🟠       |           |
|     6 | node_linear         | y            |          0.036 |     0.0066223 |      0.0025724 | 5 🔴       | 4 🔴      |
|     6 | node_linear         | z            |          0.036 |             0 |              0 | 6 🟠       |           |
|     7 | node_linear_1       | y            |          0.036 |     0.0051246 |      0.0021989 | 3 🔴       | 1 🔴      |
|     7 | node_linear_1       | z            |          0.036 |             0 |              0 | 4 🔴       |           |

+ Layers performs best in improving accuracy (sorted by `MaxAbsError`):

"node_linear_1", "node_conv2d", "node_conv2d_1", "node_linear",
