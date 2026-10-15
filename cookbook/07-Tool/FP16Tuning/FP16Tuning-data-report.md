# FP16 Tuning Report

+ Generated at 2026-10-15 08:17:25

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
|     2 | Pure FP16 🟦        | y            |          0.036 |      0.013803 |      0.0045699 |            |           |
|     2 | Pure FP16 🟦        | z            |          0.036 |             0 |              0 |            |           |
|     3 | FP16 + ForceFP32 🟪 | y            |          0.034 |      0.013803 |      0.0045699 |            |           |
|     3 | FP16 + ForceFP32 🟪 | z            |          0.034 |             0 |              0 |            |           |
|     4 | node_conv2d         | y            |          0.036 |      0.013803 |      0.0060834 | 3 🔴       | 2 🔴      |
|     4 | node_conv2d         | z            |          0.036 |             0 |              0 | 4 🔴       |           |
|     5 | node_conv2d_1       | y            |          0.036 |      0.013803 |      0.0045699 | 5 🔴       | 3 🔴      |
|     5 | node_conv2d_1       | z            |          0.036 |             0 |              0 | 6 🟠       |           |
|     6 | node_linear         | y            |          0.038 |      0.029428 |      0.0056425 | 7 🟠       | 4 🔴      |
|     6 | node_linear         | z            |          0.038 |             0 |              0 | 8 🟠       |           |
|     7 | node_linear_1       | y            |          0.034 |     0.0063276 |      0.0016509 | 1 🔴       | 1 🔴      |
|     7 | node_linear_1       | z            |          0.034 |             0 |              0 | 2 🔴       |           |

+ Layers performs best in improving accuracy (sorted by `MaxAbsError`):

"node_linear_1", "node_conv2d", "node_conv2d_1", "node_linear",
