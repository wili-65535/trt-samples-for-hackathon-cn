# 99-Todo

+ cookbook 的待办清单与调研笔记。

## 1. 待办 —— 还剩什么

只剩一项。**M4 和 M11 已于 2026-09-07 落地**；只剩 P5，而它卡在 Torch-TensorRT 的升级上，不在我们这边。

+ **M4** -> [`07-Tool/QDQPlacementAutotune/`](../07-Tool/QDQPlacementAutotune/README.md) —— 用真实
  TensorRT 延迟计时的 Q/DQ 放置搜索：14 次 engine 构建换来 **1.174 倍**（0.110 -> 0.090 ms）。同时
  记下了两个诚实的负面结果：在这么小的模型上，模式缓存**什么也没省**（开不开都是 14 次基准测试）；
  而且搜索是**随机的** —— 带缓存的 model-B 那一行一次读出 1.388 倍、另一次 1.149 倍，所以
  「有缓存 vs 无缓存」的延迟差**不能**被当成缓存的效果来读。
+ **M11** -> 并入
  [`03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT`](../03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT/README.md)
  成为 `case_autocast_bf16` + `case_node_sensitivity` —— 在一个从不溢出的模型上（opset 19 -> 22），
  BF16 比 FP16 **离 FP32 远 6.0 倍**；而 `data_max` 扫描显示有三个阈值产出**逐字节相同的图**，却报出
  不同的误差。

已完成的条目不列在这里 —— 每一条的发现都写在它自己那个例子的 README 里，值得跨例子携带的经验在 §2。
到目前为止做过的一切都在 `wili/dev` 分支的工作区里，**尚未提交**。

| #   | 条目 | 来源 | 目标 | 就绪？ |
| --- | ---- | ------ | ------ | ------ |
| P5  | `Input(profiles=[...])` —— 一个 engine 上挂 N 个 optimization profile，运行时按序号或 `"auto"` 选择 | `.../multi_optimization_profiles.py` | `06-DLFrameworkTRT/Torch-TensorRT/OptimizationProfiles/` | **受阻，2026-09-07 再次复核**：`torch_tensorrt` 2.14.0a0 一样都没有 —— `Input` 没有 `profiles` 参数，`torch_tensorrt.runtime.optimization_profile` 不存在。`set_active_profile` 住在 `core/runtime/TRTEngine.cpp` 里，所以需要完整源码构建，覆盖几个 Python 文件是不行的。TensorRT API 层面的等价物已经有了，就是 `08-Advance/MultiOptimizationProfile` |

另外还有一项来自 §3.1 仍未关闭：**S8** —— 把剩下那些标着 H100 的数字在 B200 上重测。动手前先读 §2.1
里那条关于热降频的教训，它会改变这件事该怎么做。

### 1.0 Engine 可视化：收敛到两套系统

当前有三套实现把 engine 的 layer-info JSON 变成图，计划是收敛到两套。对比表在
[`07-Tool/EngineVisualization/README.md`](../07-Tool/EngineVisualization/README.md)；这里记的是工作。

**V1. 控制流 —— 已完成。** `export_engine_as_onnx` 不再拒绝带 `Loop` 的 engine，并且把 `model-loop`
的**全部 8 条**依赖都表示了出来（6 条前向 + 2 条回边），而另外两个写出器只保留 5 条且一声不吭。两种
模式，都由 `07-Tool/EngineVisualization/export_as_onnx.py` 在 `model-loop.onnx` 和 `model-for.onnx`
上覆盖：

+ **默认**把回边保留为真实的环。`onnx.checker` 会拒绝它 —— ONNX 的定义要求单静态赋值 —— 但
  **Netron 能正确画出来**，而这些文件是用来读的、不是用来执行的。已用一个手工搭的三节点环状 ONNX
  验证过，该文件由测试现场重建而非提交进仓库。
+ **`b_break_cycle=True`** 把每一次写入重命名为一个 SSA 版本，并加入 `BackEdge` 标记节点，得到一个
  能通过 `onnx.checker` 的 DAG。任何建立在 `onnx_graphsurgeon` 之上的东西都需要它，否则会抛
  `Cycle detected in graph!`。不设为默认是因为它更难读：那些标记是死胡同，得按名字去查；而且去掉回边
  之后，图常常会裂成互不相连的几块。

不含控制流的 engine 在两种模式下逐字节相同。

仍未解决的是**结构化**形式：一个带真实子图 body 的 ONNX `Loop`，那样 Netron 就能把 body 折叠起来。
这件事比乍看起来难得多 —— **engine 里根本没有子图可供还原。** TensorRT 把循环编译成一条扁平的指令流，
带一个显式的 `cjmp` 层和一条回边，所以要产出 ONNX `Loop` 就等于反编译：要从分支推断 body 的边界，还要
把 recurrence 映射回 carried dependency。嵌套循环、多出口、body 里再嵌一个 `If`，每一条都让事情更复杂；
而一个**细微错误的重建比我们现在输出的那个诚实的环更糟**，因为它看起来是对的。不排期。

**V1b. 另外两个写出器会静默丢边，而且这不是控制流的 bug。** `EngineVisualization/main.py` 和
`utils_engine_explorer.py` 都用 `producer[name] = index` 建生产者索引（后写的覆盖先写的），然后用
`source != index` 跳过自环。任何复用了张量名的 engine 都会无警告地丢边；一个 `Loop` engine 会丢掉
8 条中的 3 条。最起码它们应当检测出「一个名字有多个生产者」并说出来。目前刻意搁置，先把 ONNX 这条路
做完。

**V2. 把 `trex/02-DrawEngineGraph` 的着色吸收进 `EngineVisualization`，然后删掉它 —— 已完成。**
这些写出器现在做了 `trex/02` 做的一切，外加一些它没做的：节点按层类型填色、连线按张量精度着色、
逐层延迟并把最慢的层描红框、绑定节点、各种显示开关，以及在 `.gv` 之外栅格化出 `.svg`。超出 `trex` 的
部分还包括：把 JSON 的每个字段留在节点 tooltip 里、把循环的回边画成虚线、过滤多 profile engine 并打印
出计数而不是悄悄处理。

调色板是**重建的而不是拷贝的**：`layer_colormap` 和旧的 `TYPE_COLOUR` 用的都是 TRT 8/9 时代的首字母
大写名字，而 TensorRT 11 吐出来的是 `kgen`、`gemm`、`correlation`、`maxpool`、`memset`、`add`、
`cjmp`、`custom_layer`。本机实测，旧调色板在一个 MNIST engine 的五种类型里最多只能着色两种，所以两个
工具画出来的都是一张近乎全灰的图。

`fold_no_ops` 是唯一一个刻意**没有**拿过来的特性。一个建好的 engine 里的 `NoOp` 通常标记着 builder
消不掉的 reformat 或拷贝，所以删掉它只是**把证据删了换来图更好看** —— 而且这和 ONNX 写出器遵循的规则
冲突：不要静默改变你被要求去画的那个东西的结构。

`07-Tool/trex/02-DrawEngineGraph/` 已于 2026-09-10 删除，引用也一并清理。它调用的辅助函数
`utils_engine_explorer.py` 里的 `render_engine_graph` **保留**：`trex/11-ProcessEnginePipeline` 把画图
作为其流水线的一步，所以消失的只是那个「纯粹为了画图而存在」的例子。可发现用例数 256 -> 255。

注意这只删掉了 `trex/02-DrawEngineGraph`。`07-Tool/trex/` 的其余部分 —— 那些 report card —— 不受影响，
保留。

### 1.1 下一步：TensorRT 11.2 升级

11.0 的工作已完成：2026-09-08 **257 / 257**（`tests/logs/REPORT-2026-09-08.md`）。下面这份检查单是升级
必须处理的事，按优先级排列。

**1. 版本标签已经不对了，升级会让它更不对。** 有 46 个例子的 README 写着 **TensorRT 11.1.0.106**；
容器当前是 **11.0.0.114**，之后会是 11.2。那些数字当初**确实**是在 11.1.0.106 上测的，所以一直没改标签
—— 不重测就改标签，是这个仓库唯一拒绝做的事。升级之后它们就落后两个版本了。要在下面两条里**刻意**
选一条：
  + 重测并改标签（正确，昂贵 —— 这就是 S8 的翻版），或者
  + 每个 README 加一行带日期的「measured on」，数字原样保留（便宜，诚实）。

**绝对不要**让批量查找替换发生。

**2. `04-Feature/RefitObserver` 应该可以甩掉它外挂的那套 TensorRT。** 它现在通过 `TRT_OBSERVER_PATH`
对着一个解包的 **11.3** tarball 编译，因为 `IRefitterObserver` 是 11.2 才有的、而系统安装的版本更老。
在 11.2 上系统头文件应该就够了 —— 把 `TRT_OBSERVER_PATH` 这层间接和它 `unit_test.yaml` 里的
`LD_LIBRARY_PATH` 要求一起退役掉。仍然是 **C++ only**：11.3 的 cp312 wheel 里没有 `set_refit_observer`，
所以在假定存在 Python 绑定之前，先去查 11.2 的 wheel。

**3. 复查 API 移除导致的修复。** 有两个例子在 11.0 移除弱类型时挂了，已在那轮扫描中迁移。11.2 可能
移除更多：
  + `02-API/Layer/QDQStructure/main.py` —— `BuilderFlag.INT8`（已移除），以及 `add_quantize` /
    `add_dequantize` 现在要求显式的 `output_type`；
  + `07-Tool/trtexec/main.sh` —— `--fp16` / `--int8` / `--best` / `--precisionConstraints` 全部被
    拒为 "Unknown option"。
  `trt-strong-typing-migration` skill 是参考。它的 `scripts/migrate.py --write`
  **会重排整个文件并抹掉 SPDX 头**（它是通过 AST unparse 往返的），所以拿它看 diff 和结论，然后手工改。

**4. 每一次都要先重装 pip 附加包**，见 §1.2。

**5. P5** 仍然卡在 Torch-TensorRT 而不是 TensorRT 上，这次升级不会推动它。

### 1.2 环境供给：容器在会话之间会重置

`tensorrt_rtx`、`tensorrt-lean` 和 `tensorrt-dispatch` **不会**在重置后幸存，每次全量扫描前都必须重装：

```bash
pip install tensorrt_rtx tensorrt-lean tensorrt-dispatch
```

Graphviz（`/usr/bin/dot`，apt 装的）到目前为止都活下来了。**一次全绿的扫描并不能证明附加包装上了**：
那 16 个 `optional-package` 用例会守卫自己的 import 并带着一行 `[SKIP]` 以 0 退出，所以一次没装附加包
的运行照样报 257/257。事后要逐个 log 去 grep `^[SKIP]` 这个标记来核实 —— 不要 grep "skip" 这个词，它会
在 `10-TensorRT-RTX/01-ComputeCapability` 的正常输出里出现（"skip setting"），造成假阳性。

## 2. 值得携带的发现

### 2.1 测量的教训

有三个错误反复出现到值得专门写一次。每一个都产出了**自信而错误**的结果，看起来和真实测量一模一样。

**热降频。** 这台机器上的 GPU 0/1 在持续负载下会到 88-91 °C，并从约 1900 MHz 掉到 **352 MHz**
（`clocks_throttle_reasons.active = 0x20`，软件热降频，功耗 437 W 而上限是 1000 W）。同一个 engine
在凉的 GPU 上测出 2.8 ms，在热的上测出 11.0 ms；有一次扫描**报出 111.8 ms 而正确答案是 10.9 ms**，
还把一个看似的交叉点挪动了一倍。`08-Advance/ContextParallelism` 现在会挑最凉的 GPU、等它们凉下来，
并在每个延迟旁边打印 SM 时钟。**这台机器上任何背靠背取得的延迟数字，如果没有同时记录时钟，都值得怀疑。**

**编排开销淹没被测量。** `08-Advance/MultiTask` 和
`06-DLFrameworkTRT/Torch-TensorRT/DistributedInference` 一开始都把并发测成了**退步**（0.35x、0.65x、
0.79x），因为每次迭代两道 Python 屏障的开销比被计时的 GPU 工作还大。修法是把工作量放大，并改测吞吐
（屏障只付一次）而不是每轮延迟。`MultiTask` 现在显式测量屏障的地板值并打印在每个结果旁边。
**当每次迭代的工作量逼近它周围同步的开销时，这个基准测试就不再是关于 TensorRT 的了。**

**被探针选择做局的对比。** `04-Feature/OnnxPTQMethod` 曾经只在「离群点被**种进去**的那份数据」上比较
entropy 与 max 校准 —— 在那种数据上 `max` 按构造必赢。加上一个普通探针之后**排名反转了**。另外，
`04-Feature/LowBitQuantization` 曾经在数据显示相反结论的情况下断言「块缩放胜过位数」。
**只要一个对比能被输入的选择做局，就要把它的两面都测。**

还有一个小一点的：`08-Advance/ResourceProbe` 曾经断言它当时恰好观察到的那个具体的析构症状。那个症状是
非确定性的（要么以 0 退出但记录 5 条错误，要么静默 SIGSEGV），所以断言是不稳定的。**断言不变量，不要
断言观察到的现象。**

### 2.2 我们正在扛着的上游缺陷与绕行方案

这些都**没有修好** —— 例子只是绕开它们。每一条都应该在下一次 ModelOpt / Torch-TensorRT / TRT 升级后
重测，第一条应当上报上游。

| 是什么 | 在哪里咬人 | 例子改用什么做法 |
| ---- | -------------- | ----------------------------- |
| **ModelOpt 的 Q/DQ 自动调优器在 TRT 11 上用不了它自己的默认基准后端。** `autotune/benchmark.py:361` 读 `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`，而显式 batch 成为唯一模式后该枚举已被移除。那个 `AttributeError` 被吞掉，只报成 `Failed to initialize TensorRT benchmark`，什么都没说 | `07-Tool/QDQPlacementAutotune` | 传 `--use_trtexec` 走 `TrtExecBenchmark` 那条路，可以工作。**应与 `plugin_creator_list` 一并上报 ModelOpt** |
| **ModelOpt 0.44.0 与 TRT 11 不兼容。** TRT 11 移除了 `IPluginRegistry.plugin_creator_list`，而 ModelOpt 仍在读它 | `05-Plugin/ONNXPTQWithPlugin` | 自带 `apply_modelopt_trt11_shim()`，把该属性作为 `all_creators` 的别名加回去。**应上报 ModelOpt** —— 这个 shim 是为了让例子能跑，不是修复 |
| **`trt_plugins=` 在这里不可用。** 即使过了 shim，ONNX Runtime 在**加载 session 时**就会拒绝含 TensorRT 自定义算子的图；`calibration_eps=["trt"]` 也没用。三条报错，没有一条说出自己的原因 | `05-Plugin/ONNXPTQWithPlugin` | 校准一个可用 ONNX 表达的替身，再把得到的 Q/DQ 移植到 plugin 图上 |
| **低于 8 bit 的格式根本到不了 engine。** Torch-TRT 的转换器直接拒绝 NVFP4 / MXFP8 / INT4-AWQ；ONNX 路线对**全部五种**格式都失败；把 ModelOpt 量化过的 torch 模块导出到 ONNX 对每种格式都失败；INT4 weight-only 能量化但在 TRT 构建时因块大小整除失败（`inputSize % scaleSize == 0`，3211264 / 25600） | `04-Feature/LowBitQuantization`、`04-Feature/OnnxPTQMethod` | 把负面结果原样发布。**这套栈上只有 INT8 和 FP8 能到达 engine** |
| **TRT-LLM 的 plugin 拒绝 CUDA 13**，所以张量并行演示不了 | `06-DLFrameworkTRT/Torch-TensorRT/DistributedInference` | 只做数据并行（4 卡 3.11x）；TP 被记录为受阻 |
| **`GroupNormalizationPlugin` 已被移除**（TRT ≥ 10.7，Blackwell） | `07-Tool/OnnxGraphSurgeon/14_fold_exporter_subgraphs.py` | 改为发出原生的 ONNX `GroupNormalization`（opset 21）而不是 plugin 节点 |
| **E8M0 块缩放构建不起来。** 用一个 2x2 测试隔离过 —— 裸的 Q/DQ 对能正常构建，**E8M0 这个 scale 才是阻塞点**；FP4 的块 QDQ 则被直接拒绝 | `02-API/Layer/QDQStructure/block_quantization.py` | 作为实测边界记录在案 |

### 2.3 静默失败 —— 这一轮最有价值的产出

这些没有一个会抛异常。每一个都会给出一个看似合理的数字和一个错误的答案。

+ **`refit_module_weights` 接受结构不同的模型。** 用一个 8 层的模块去 refit 一个 6 层的 engine 是
  **被接受的**；之后这个 engine 与该模块的**前 6 层完全一致**（差值 0.0），与它真正的输出则毫无关系。
  多出来的层被一声不吭地丢掉。这是本页最危险的一条 —— `06-DLFrameworkTRT/Torch-TensorRT/Refit`。
+ **`use_explicit_typing=True` 与 `enabled_precisions=...` 同时出现会被静默接受**，而不是被拒绝。
  我预测会报错，预测错了 —— `06-DLFrameworkTRT/ModelOptimizer`。
+ **给 shape tensor 绑一个设备指针是 SIGSEGV，不是报错。** shape tensor 在主机侧读取，而没有任何地方
  检查这个指针 —— `02-API/ShapeTensor`（该用例在子进程里跑，正是因为它会杀掉进程）。
+ **`MAX` / `MIN` 归约会静默丢掉 NaN**，而 NumPy 会传播它（8.0 / 5.0 对 `nan`），所以 max 归约
  **不是**一个 NaN 检测器 —— `04-Feature/CornerCase`。
+ **TensorRT 可以记录 5 条析构错误却仍然 `exit 0`**，当它的 CUDA 上下文先被销毁时 —— 对任何基于退出码
  的检查都不可见。而且症状是非确定性的：有时是带错误的 exit 0，有时是静默 SIGSEGV ——
  `08-Advance/ResourceProbe`。
+ **相同的权重在 plan 里会被去重**，把一个 18 MiB 的 engine 压成 2 MiB，于是任何建立在「重复权重」之上
  的加载/流式测量都失去意义 —— `08-Advance/MultiDevice`。
+ **`polygraphy run --trt --fp16` 在 `POLYGRAPHY_USE_TENSORRT_RTX=1` 下会丢掉该标志并通过。**
  同样的请求走 `CreateConfig(fp16=True)` 则会抛异常。只看退出码的检查分不出「标志被丢掉」和「标志被
  执行」—— `07-Tool/Polygraphy/TensorRTRTX`。
+ **TF32 默认开启，是手工搭建的网络「对不上」时的头号嫌疑人。** 它在一个正确的网络和 torch 之间制造了
  2.643e-04 的差距，而 float32 下是 8.941e-08，**2956 倍**的差异，读起来和一个转置错的权重块一模一样
  —— `02-API/Network`、`04-Feature/RefitObserver`。
+ **`trt.Weights` 包的是一个指针。** 值是在 `build()` 时读的，不是在 `add_*()` 时。在这中间改动数组会
  静默改变 engine；让它被回收则是一次没有任何提示的 use-after-free —— `02-API/Network/weight_transport.py`。
+ **TensorRT-RTX 的 `set_compute_capability` 在 `num_compute_capabilities` 未赋值时返回 `False` 而不是
  抛异常**，随后构建会悄悄以当前设备为目标 —— `10-TensorRT-RTX/08-DeployTimeRuntimeConfig`。
+ **`IRuntimeConfig` 不会让它的 runtime cache 保活。** 在 `set_runtime_cache()` 之后丢掉最后一个 Python
  引用，进程会段错误，静默地 —— 同一个例子。
+ **两种误差模式可以有相同的 max-abs-diff**，而其中一个是 80/80 个元素都错、另一个只错 1/80。单一的
  标量容差分不出它们 —— `07-Tool/Polygraphy/DebugWorkflow`、`07-Tool/AccuracyCheck`。

## 3. 待定候选

### 3.1 常驻条目（不来自任何仓库调研）

| #   | 条目 | 备注 |
| --- | ---- | ---- |
| S8  | **在 B200 上重测所有标着 H100 的实测数字。9/10 完成，2026-09-06。** 重测并改写了 10 处中的 9 处。**最后一处 `07-Tool/NsightSystems` 被环境阻塞**：`nsys` 目前在这台机器上对**任何**目标都挂死 —— `nsys profile -o /tmp/x /bin/true` 不返回，GPU 全程 0%、日志为空、不产出 report；加 `--sample=none --cpuctxsw=none` 仍挂，`perf_event_paranoid=-1`，所以不是 CPU 采样、不是 TensorRT、也不是并发争用，是 nsys 自身。该例的数字**保持 H100 标签不动**（改标签而不重测等于造假），README 里已写明原因，等 nsys 恢复后再补。**四条结论因平台变化而改变**：(1) `08-Advance/GreenContext` —— **148 SM** 而非 114（`minSmPartitionSize`/`smCoscheduledAlignment` **仍是 8**），噪声邻居下加绿色上下文后 p95 = **0.99x**，即与独占无异（H100 上是 2.41x）；build-in-partition 收益 19% → **8%**（三次复现，0.010 ms vs 0.000–0.001 ms 噪声）。(2) `07-Tool/trtexec` —— `--infStreams=4` 在 H100 上是 2.75x 吞吐，B200 上**完全无收益**（26274.9 qps 反而低于 2 流的 26451.2），流数调优不跨 GPU 迁移。(3) `07-Tool/OnnxVisualization/90-Research/01-SubgraphInONNX` —— flat 与 loop **不再逐比特相同**（MaxDiff 0 → 1.19e-07~1.76e-05）；flat 的 build 爆炸从 2.0x 变成 **8.6x**（9.3 s → 79.4 s），折叠成 `Loop` 更划算。(4) `06-DLFrameworkTRT/Torch-TensorRT/EngineCaching` —— 原文「冷缓存下 build+save 会略慢」**在旧的 H100 数据上就不成立**（三行里两行更快），已改写为「中间那列被 warm-up 主导，不要当成开销读」。其余重测但结论不变：`08-Advance/MIG`（profile 表换成 `7g.180gb`/148 SM 至 `1g.23gb`/18 SM）、`08-Advance/TensorRTGraphSurgeon`（2.13x → 1.84x）、`05-Plugin/TritonAOTPlugin`、`07-Tool/TritonServerDeploy`（13.14 → 7.23 ms）、`07-Tool/OnnxVisualization`（3 处共用表） | 全部在 1965 MHz / 26 °C 下测，见 §2.1 |

### 3.2 TensorRT OSS —— `/work/trt/TensorRT-GitHub`

**该仓库的吸纳已全部关闭（2026-09-04）。** G1、G3、G6、G7、G8、G10、G11、G12 已落地；G2、G4、G5、G9
被否决并移入 §4。

**G12 的剩余部分，刻意不取。** G12 是个杂项包；其中两部分做了，其余记在这里而不是悬着：

+ **已做** —— `topkLastDimPlugin` 作为性能脚注（实测：k=64 时 2.12x，但 **k=8 时 0.83x**，而且
  `ITopKLayer` 对 k > 3840 会让 `add_topk` 返回 `None` 来拒绝）；以及 `bfloat16.*` 的对比，它找到了一个
  真实缺陷 —— `numpy_fp32_to_bf16` 做的是**截断**而不是四舍五入到最近偶数，因此在一个随机数组上与
  `torch.to(torch.bfloat16)` **有 49.8% 不一致**。已修复并向量化（快 13 倍，现在与 torch 逐比特一致，
  NaN 保留）。
+ **未取** —— `streamReader.h`（cookbook 的 `cookbookHelper.cuh` 里已有 `FileStreamReader`）、
  `ErrorRecorder.h`（已有 `CookbookErrorRecorder`）、`safeCudaAllocator.h`（safety 运行时，而且
  `04-Feature/Safety` 已覆盖该边界）、`plugin_utils.py::CudaCtxManager` 和
  `common_runtime.py::ArrayWithOwner`（两者都是 sample 管线的所有权惯用法，cookbook 的 wrapper 已经
  自己拥有了），以及交叉编译的 Dockerfile（`ubuntu-26.04`、`ubuntu-cross-aarch64`）——
  **在这台机器上无法验证**，因为这里没有任何东西能跑 aarch64 目标。

**作为近似重复刻意跳过的** —— `quickly_deployable_plugins`（≈ `QuickDeployablePlugin`）、
`onnx_custom_plugin`（≈ `ONNXParserWithPlugin`）、`non_zero_plugin`（≈ `DataDependentShape`）、
`deploy_to_triton`（≈ `07-Tool/TritonServerDeploy`）、`sampleIOFormats`（≈ `04-Feature/DataFormat`）、
`sampleNamedDimensions`（≈ `LabeledDimension`）、`sampleProgressMonitor`、`sampleDistCollective`、
`network_api_pytorch_mnist`、`samples/python/refactored/`、`stream_writer`（≈ `02-API/Builder` +
`CookbookStreamWriter`）、`common.py::setup_timing_cache`（≈ `04-Feature/TimingCache`）。
大多数 C++ 的 `plugin/*` 目录（bertQKV、efficientNMS、groupNorm 等）**自 10.12–10.15 起已废弃**；
11.0 还额外*移除*了 `batchTile`、`clip`、`coordConvAC`、`cropAndResize`、`gelu`、`leakyRelu`、
`normalize`、`singleStepLSTM`、`specialSlice`、`split`、`nms`、`proposal`。`tools/pytorch-quantization`
和 `tools/tensorflow-quantization` 是遗留物（已被 ModelOpt 取代）。

### 3.3 TensorRT 内部仓（GitLab）—— **使用前必须甄别**

GitLab 仓库**已不再跟踪 `samples/`**（`oss_components.yml` 把 `samples/**/*` 列为 `detracked`），而
`plugin/` 和 `tools/trtexecCommon/` 与 GitHub 上的完全相同。GitLab 独有的面是 `tests/unitTests/`、
`tools/`、`samples_internal/`、`testing/`、`scripts/`、`plugin_internal/`、`projects/`、
`documentation/`。

> 内部单元测试会 `#include` 内部的 harness 头文件。那些*层 API* 是公开的，但任何吸纳都必须
> **对着公开 API 从头重写 —— 可以拿走想法，绝不要拿走文件。**

**所有 L 条目均已决定。** L1、L2、L3、L5、L6、L7、L8、L9、L11 已落地；L4 和 L10 被否决并移入 §4。
这个仓库没有悬而未决的条目了。

### 3.4 Model-Optimizer —— `/work/trt/repos/Model-Optimizer`

基线：`03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` 已经在一个很小的 MNIST CNN 上覆盖了
**AutoCast FP16**、**torch INT8 QAT** 和 **FP8 ONNX PTQ（max 校准）**。不要重新提这些。由于 TRT 11 的
强类型移除了弱类型 INT8 校准，ModelOpt 成为被认可的量化路径，下面每个候选都会产出强类型 engine 能消费
的显式 Q/DQ。

**所有 M 条目均已决定（2026-09-06）。** M4 和 M11 在 §1；M5、M6、M8、M9、M10 被否决并移入 §4。
这个仓库没有悬而未决的条目了。

跳过：`vllm_serve`、`llm_eval`、纯 HF/vLLM 服务（完全不碰 TensorRT）；`deepseek`、`gpt-oss`、
`minimax_m3`、`megatron_bridge`、`speculative_decoding`、`alpamayo`、`puzzletron`、Minitron 剪枝
（巨量下载 / NeMo-Megatron 容器）。

### 3.5 TensorRT-RTX 与 tensorrtx

`10-TensorRT-RTX/{01..07}` 已经**逐个**覆盖了 RTX 独有的 API；`02-API/Layer/Scale` 覆盖 `add_scale` 的
机制；`02-API/Network` 只是戳了戳 Network 对象的属性。

**所有 R 和 T 条目均已决定（2026-09-06）。** R1、R2、T2 已落地；R3、T1、T3、T4 被否决并移入 §4。
这两个仓库没有悬而未决的条目了。

### 3.6 Torch-TensorRT 与 Tripy 的剩余项

八个 Torch-TensorRT 例子落在 `06-DLFrameworkTRT/Torch-TensorRT/` 下；Tripy 落为 `07-Tool/nvtriPy`。

**所有 P 条目均已决定。** P1、P2、P3、P4 已落地，P5 仍受阻（§1）；P6 被否决并移入 §4。
P5 的阻塞点没有变化，随条目一并记录：

**P5 卡在已安装的构建上。** `torch_tensorrt` 2.14.0a0 这套 API 一样都没有：`Input(profiles=...)` 抛
`ValueError`，`torch_tensorrt.runtime.optimization_profile` 不存在，`torch.classes.tensorrt.Engine`
不暴露任何 profile 方法，而 `dynamo/conversion/_TRTInterpreter.py` 把单个
`create_optimization_profile()` 写死了。源码克隆里带着这个特性，而且版本号字符串*也是* `2.14.0a0`，但
`set_active_profile` 住在 `core/runtime/TRTEngine.cpp` 里 —— 覆盖较新的 Python 文件是不行的，需要完整
源码构建。等 Torch-TensorRT 升级后再看。TensorRT API 层面的等价物已经有了，就是
`08-Advance/MultiOptimizationProfile`。

### 3.8 TensorRT-RTX-EP-ABI —— 已调研，暂不做

**仓库的作用。** [`NVIDIA/TensorRT-RTX-EP-ABI`](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) 是
**TensorRT-RTX 的 ONNX Runtime execution provider**，以独立插件
`onnxruntime_providers_nv_tensorrt_rtx.{dll,so}` 发布。它实现 ORT 1.23.0 引入的 **EP ABI**
（`OrtEpFactory` / `OrtEp` / `OrtNodeComputeInfo` / `OrtDataTransferImpl`），因此只依赖 ORT 的 **SDK**
而不必与 ONNX Runtime 一起编译 —— 这正是它存在的理由。流程是：ORT 把子图交给它 → 转成 ONNX proto →
跑一组重写 pass → 交给 TensorRT-RTX 的 ONNX parser → 建 engine → 还给 ORT 一个可调用对象。

对我们的价值集中在那 21000 行 `src/` 里：它基本上是「把真实 ONNX 喂给 TensorRT parser 会出什么问题」
的答案集，每条都带修法和一句「它故意不修什么」。详细候选清单见 [`trt-ep-abi.md`](trt-ep-abi.md)。

**我们的进展：曾经做过一个端到端例子，2026-10-12 已整体清理，全部条目暂不做。**

那个例子（`12-TensorRT-RTX-EP-ABI/`）走完了「编译 .so + 在 onnxruntime 里跑一个最简模型」的闭包，
第 1/2/3/5 步都验证通过，**唯独第 4 步构建始终没跑成**，原因是一个无法绕开的供给问题：

+ **TensorRT-RTX 的 C++ SDK 拿不到。** developer.nvidia.com 需登录，
  `developer.download.nvidia.com/compute/tensorrt-rtx/redist/` 是 404。
+ **`pip install tensorrt_rtx` 不能替代。** 实测装上 1.6.1.120 之后，全盘递归搜 `NvInfer*.h` **零命中**，
  只有 `libtensorrt_rtx.so.1` 和 `libtensorrt_onnxparser_rtx.so.1`；PyPI 上也没有头文件专用包
  （`tensorrt-rtx-headers` / `tensorrt_rtx_headers` / `tensorrt-rtx-dev` 全 404）。
+ **`NVIDIA/TensorRT-RTX` 仓库也不是 SDK。** 那是样例仓库（`samples/{helloWorld,apiUsage,
  memoryManagement}` + `demo/flux1.dev`），不含 `NvInfer*.h`，它自己的 CMake 还在等别人提供
  `TRTRTX_INCLUDE_DIR`。
+ 即便拿到 SDK，**本机 B200 也是错的硬件** —— TensorRT-RTX 面向 Ampere 及以后的 RTX 卡，预期结果是
  插件能加载注册但报告零设备。

所以这条路的阻塞点是环境而不是技术判断。等哪天手上有 RTX 卡和 SDK 了再从 `trt-ep-abi.md` 重新开始。
其中 B1（每个形状一个 CUDA graph）已经通过另一条线落地了，见 `08-Advance/CudaGraph` 的
`case_graph_per_shape`。

### 3.7 社区仓库 —— 取想法，不要导入

| 仓库 | 值得偷的想法 |
| ---- | ---- |
| [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)（4.9k★） | 把**逐算子的转换器注册表**当作讲 Network API 的教具 |
| [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO)、[TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) | NMS/后处理 plugin + 融合的 CUDA 前后处理，作为一条真实的检测流水线 |
| [mmdeploy](https://github.com/open-mmlab/mmdeploy)（3.1k★） | 多后端部署 + 自定义 TRT plugin |
| [jetson-inference](https://github.com/dusty-nv/jetson-inference)（8.9k★） | 嵌入式 C++ 运行时演示 |
| [WhisperLive](https://github.com/collabora/WhisperLive)（4.1k★）、[x-stable-diffusion](https://github.com/stochasticai/x-stable-diffusion)、[SD-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT) | 流式 ASR / 扩散模型的端到端框架，动态 shape 的 engine 管理 |

与 TRT 无关、仅作生态背景：TNN、Tengine、lite.ai.toolkit、YOLOX、yolov5。

---

## 4. 已否决 —— 不要再提

+ **DLA。** 每一条 DLA 路径（`tutDLA*.cpp`、`samples/sampleCudla`、一个 `04-Feature/DLA/` 分组）在这台
  B200 上都会走「没有可用 DLA core」的提前返回（`builder.num_DLA_cores == 0`），永远无法验证。
  **发布无法验证的例子比不发布更糟。** 只在 Orin / DRIVE / Jetson 上重开；用来构建的 API 已确认在
  TRT 11.0 中存在：`trt.MemoryPoolType.DLA_{MANAGED_SRAM,LOCAL_DRAM,GLOBAL_DRAM}`；
  `IBuilderConfig.{default_device_type, set_device_type, get_device_type, is_device_type_set,
  reset_device_type, can_run_on_DLA, DLA_core}` 配 `trt.DeviceType.DLA`；
  `Runtime.{DLA_core, num_DLA_cores}`；`trt.EngineCapability.DLA_STANDALONE`。
+ **demoDiffusion。** 忠实移植意味着大量下载、一个多 engine 的流水线管理器和 FP8/FP4 校准数据。
  不如把用户指向上游，而不是发一个会逐渐漂移的裁剪版。
+ **从 ONNX 解析 `TRT_*` 自定义算子。** `TRT_Attention` / `TRT_MoE` / `TRT_KVCacheUpdate` 已经在
  `02-API/Layer/*` 里以 network API 层的形式覆盖了。
+ **`sampleDevice.h` 风格的 C++ RAII 封装。** 试过，加进了 `cookbookHelper.cuh`，把
  `08-Advance/CudaGraph` 重构到它上面，然后全部回滚。C++ 例子保留显式的 `cudaMalloc` /
  `cudaStreamCreate` / `cudaGraph*` 调用并配对显式释放。
+ **TensorRT-LLM 的吸纳 —— 2026-08-28 关闭。** 那个仓库里非 3rdparty 的文件中只剩 9 个还
  `import tensorrt`；engine 构建栈（`builder.py`、`network.py`、`module.py`、`python_plugin.py`、
  `tools/plugin_gen/`）已被删除，三个 plugin 例子全部 import 失败。要看用 Python 写的 TRT plugin，
  改看 OSS 的 `samples/python/python_plugin`（G3）。
+ **两个「缺口」其实不是缺口：** `05-Plugin/InPlacePlugin` **确实**已经在 C++ 里用了 aliased I/O
  （`v_2_0::IPluginV3OneBuild` + `getAliasedInput`；早先的 grep 漏掉了驼峰命名），而
  `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` **确实**已经覆盖了 AutoCast。
+ RTX 的 Flux.1[dev] demo 和 tensorrtx 的模型库：引用，绝不导入。
+ **又五个候选，2026-09-04 关闭**（原 L4 / L10 / M7 / M12 / P6）：
  + **L4 —— "Picky FP8"**（`tests/unitTests/tutPickyFP8.cpp` → 02-API/Layer/QDQStructure）。一个被
    刻意设计得对错误类型或错误 scale 极其敏感的 FP8 Q/DQ/MatMul 网络（alpha = 1+2⁻¹¹ 这类技巧）。
    它是一个针对 TensorRT 的*测试*，而不是一堂关于它的课：载荷是那个对抗性的常数，复现它教不会读者
    任何可复用的东西。FP8 的 Q/DQ 放置本身已由 `02-API/Layer/QDQStructure` 覆盖，现在还有
    `07-Tool/OnnxFP8QDQConvert`。
  + **L10 —— 把算子参考文档当作 QA 来源**（`documentation/operators/*.rst`）。明确标注为「不是要导入
    的内容」：想法是拿 `02-API/Layer/*/README.md` 去和权威的逐算子文本交叉核对。这是一件没有产物的
    文档审查杂活，而且同样的保证已经由层例子里的 `check_api_coverage` 机械地强制执行了 —— 当 TensorRT
    的 API 和例子不一致时它会失败。
  + **M7 —— 扩散模型（SDXL / FLUX）的 INT8/FP8/FP4 PTQ**（`examples/diffusers/quantization/`）。
    数 GB 的权重和 ≥48 GB 显存，而其中 TensorRT 特有的内容 —— 强类型加多输入动态 profile —— 已经由
    `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` 和 `08-Advance/MultiOptimizationProfile`
    在不用下载的前提下承载了。另见上面因同样理由被否决的 `demoDiffusion` 条目。
  + **M12 —— 扩散模型的稀疏注意力 + 量化感知蒸馏**（`examples/diffusers/{sparsity,distillation}/`）。
    前沿但很重，而且它是一项*模型研究*技术而非 TensorRT 技术：其中没有任何东西会触及 cookbook 尚未
    展示的 API。
  + **P6 —— Tripy 里的 ResNet50 / NanoGPT / Stable Diffusion / SAM2 / ModelOpt 量化**
    （`TensorRT-Incubator/tripy/{notebooks,examples}/`）。需要受限或数 GB 的下载，*外加*在 nvtripy 的
    venv 里再装 `torch`+`transformers`，而那还是个 1.0 之前的 API。`07-Tool/nvtriPy` 已经覆盖了 Tripy
    本身。
+ **四个 TensorRT-OSS 候选，2026-09-04 关闭**（原 §3.2 中的 G2 / G4 / G5 / G9）：
  + **DeBERTa 端到端**（`demo/DeBERTa` → 03-Workflow）。PyTorch→ONNX、用 ONNX-GS 做手术插入一个
    disentangled-attention plugin、TRT 对比 ORT。成本被「先把模型弄到手」主导，而「用图手术插入
    plugin」这个想法已经由 G6（PackNet）在一个不需要受限下载的模型上承载了。
  + **用 GS 节点替换来 refit ONNX，BiDAF**（`samples/python/engine_refit_onnx_bidaf` →
    04-Feature/Refit）。替换不支持的节点（HardMax/Compress），构建成可 refit，再把假权重 refit 成真
    权重。`04-Feature/Refit` 已经覆盖了 refit API 本身；它额外带来的只是某个特定模型对不支持算子的
    绕行方案。
  + **Detectron2 Mask R-CNN R50-FPN**（`samples/python/detectron2` → 03-Workflow）。转换 / 运行 /
    验证，而且已经是强类型的。和 DeBERTa 一样被「先把模型弄到手」的成本主导，依赖还更重
    （detectron2 本身），而且没有任何 TensorRT 特有的东西是更小的模型展示不了的。
  + **声明式的数据下载**（`samples/python/downloader.py` → 00-Data）。`download.yml` 清单 + MD5
    `verifyChecksum` + 重试。这是基础设施而不是一堂 TensorRT 课，而且它现在指错了方向：自 2026-09-01
    起，例子已经**刻意不在运行时下载**任何东西（见 `00-Data/`），所以一个更好的下载器无事可做。
+ **M5 —— FastNAS 结构化剪枝，CIFAR ResNet**（`examples/pruning/cifar_resnet.ipynb`），2026-09-06
  关闭。上游的 notebook 停在 torch，所以整个 TensorRT 的那一半 —— 导出、构建、稠密 vs 剪枝的对比 ——
  都要从零写起，还得先跑一次真正的 FastNAS 搜索加微调。它本来会携带的两条教训，已经在本页上、而且不用
  付训练成本：「结构化剪枝改变张量形状，2:4 稀疏只改变其内容」这个对照作为 L8 的负面结果记录在案
  （builder 在这里*拒绝*稀疏），而「代理指标不是延迟」正是 M4 用真实 TensorRT 计时测出来的东西。
  剪枝作为一个刻意的选择，在 cookbook 中保持缺席。
+ **又八个候选，2026-09-06 关闭**（原 R3 / T1 / T3 / T4 / M6 / M8 / M9 / M10）：
  + **R3 —— 在 RTX 强制的 `STRONGLY_TYPED` 标志下，手工搭建 vs 解析**
    （`TensorRT-RTX/samples/helloWorld/`）。「同一个网络两种搭法」的对比已经是 `01-SimpleDemo`，而
    RTX 的强类型要求已经写在 `10-TensorRT-RTX` 里。R1 已经承载了所有值得展示的 RTX 特有内容。
  + **T1 —— 用 network API 从裸权重搭 LeNet-5**（`tensorrtx/lenet/lenet.py`）。逐层搭网络已经被
    `02-API/Layer/*` 详尽覆盖，而 `addFullyConnected` -> MatMul+bias 的迁移是 TRT 8 时代的注记。这里
    真正新颖的是权重传输的约定，那个作为 **T2** 单独取走了；T2 落地之后，T1 就只是用 cookbook 已经
    记录过的 API 再搭一个模型。
  + **T3 —— 把 BatchNorm 折进 `IScaleLayer`**（`tensorrtx/resnet/resnet34.cpp`）。内容是一个代数恒等
    式，不是一堂 TensorRT 课：`02-API/Layer/Scale` 已经记录了该 API，而进入 TensorRT 的每条受支持路径
    （ONNX parser、ModelOpt、Torch-TensorRT）都会自动折叠 BatchNorm，所以手工折叠只有手搭网络才需要。
  + **T4 —— 在 plugin 里做 YOLO 的 anchor-grid 解码**（`tensorrtx/yolov5/plugin/yololayer.cu`）。它自己
    的条目就写着「不要移植这段代码」：它是 `IPluginV2IOExt`，在 TRT 10 已废弃。在 `IPluginV3` 上重写
    等于做一个新的检测 plugin 而不是导入，而 `05-Plugin` 已经在无需 YOLO checkpoint 即可验证的例子上
    讲了 `IPluginV3`。
  + **M6 —— ResNet-50 INT8 QAT 配 `mto.save`/`restore`**（`examples/cnn_qat/torchvision_qat.py`）。
    与 `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` 里已有的 MNIST QAT 重叠；唯一真正新的部分是
    量化器状态的保存/恢复，这不足以换来一棵 ImageNet 规模的数据集目录树 —— 而「运行时不下载」的政策
    还得扛着它。
  + **M8 —— 非分类模型的 ONNX PTQ，SAM2 / Whisper**（`examples/windows/onnx_ptq/`）。以 ORT 和 DirectML
    为中心，所以大部分工作会是上游 sample 没有的 TensorRT 胶水代码，还要加上数 GB 的权重。PTQ 这堂课
    本身已经是 `04-Feature/OnnxPTQMethod`。
  + **M9 —— 用 SparseGPT 做 2:4 结构化稀疏**（`examples/llm_sparsity/weight_sparsity/`）。Llama2-7B，
    受限访问，约 44 GB 显存。而且它会落在一个**实测的负面结果**之上：builder 在这台机器上*拒绝*稀疏
    （L8），所以就算拿到了稀疏权重，这个例子也展示不出收益。
  + **M10 —— LLM 的 FP8 / INT4-AWQ / NVFP4 PTQ 到 TRT-LLM engine**（`examples/llm_ptq/`）。它自己的
    条目就写着「这是个指针，不是一个例子」，而它的目标章节 `09-TensorRT-LLM` 已经关闭 ——
    TensorRT-LLM 的吸纳已于 2026-08-28 关闭（见上）。低于 8 bit 的格式在这套栈上不可达，也已单独
    实测（§2.2）。

---

## 5. 排除 —— 内部 / 敏感（**不要**导入）

+ `tools/infer_fuzzing/` —— **安全**：安全团队的 ONNX fuzzer；其 README 泄露了一个真实客户的模型路径
  （Toyota）和内部攻击方法论。
+ `tools/memory_usage_safe/` —— 泄露了一个内部 SSH 端点 / 硬编码 IP，以及一套未发布的远程 QNX 计时
  服务器流程。
+ `tools/infer_ref_check_safe/` —— 未公开的内部调试钩子（`__LUNOWUD`、`MYELIN_DUMP_ALL_VALUES`、
  `MYELIN_SAVE_TENSOR_VALUES`）；Safe/LWE 运行时。
+ `samples_internal/dlaLoadableExtractor/`、`tools/engine_dumper/`、`tools/plan_converter/` ——
  向下转型到非公开类 / 解析 engine plan 的二进制内部结构（`api/engine.h`、`dispatch/planHeaders.h`）。
+ `tools/infer_device*/`、`tools/boot_time_bench/` —— 内部的 Myelin / NVRTC / Safe 运行时管线。
+ `plugin_internal/`（dlrmBottomMLP、rnRes2*、rnntEncoder、smallTileGEMM）—— 未发布的 plugin。
+ `projects/customer_plugins/{zeekr,edge-llm}/` —— **以客户命名。绝对排除。**
+ `include/NvInferSerialize.h` —— 内部的 `serializeNetwork_INTERNAL()`，标注为 `@private` 且不在公开
  头文件中。cookbook 的 `utils_network_serialization.py` 是一份独立实现；**不要**试图把它向这套 API
  对齐。
+ 所有融合 / 优化器 / Myelin 的单元测试（`tutApiFoldReformatIntoMyelin*`、`tutHorizontalMerge*`、
  `tutFuseGELU*`、`tutPointWiseFusion*`、`tutDisableFusion*`、`tutMyelin*`、`tutRaggedTensorLayer*` …）
  —— 它们编码了内部的 pass 名称、tactic 启发式和 nvbug ID。
+ `documentation/architecture/`（BuilderArch/RuntimeArch + UML）—— 读它，不要镜像它。
+ `dev_docs/`、`scripts/gitlab_ci/`、`scripts/ai-agents/`、`infrastructure/`、`coverity/`、`capture/`、
  `multigen/`、`plc/`、`optimizer/`、`runtime/`、`samples/README_internal.md` —— 内部管线。
+ Safety 相关的 sample（`sampleSafeMNIST`、`sampleSafePluginV3`、`trtSafeExec`）—— `trtSafeExec` 自己的
  README 就说它并未通过安全认证、可能违反 AUTOSAR；cookbook 已经有 `04-Feature/Safety`。

**模糊地带，使用前先甄别：** `tools/engine_visualizer/`（L7）和 `tools/infer_ref_check/`（L9）带着专有
SPDX 头和内部 URL / 邮箱 —— 改编前要重新授权并清洗。

---

## 6. 参考

### 6.1 生态仓库

| 仓库 | 在 cookbook 中的状态 |
| ---- | ---- |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)（2025 年 12 月由 "TensorRT Model Optimizer" 更名） | §3.4 |
| [NVIDIA/TensorRT-Incubator](https://github.com/NVIDIA/TensorRT-Incubator) | **已完成** → `07-Tool/nvtriPy`（独立 venv：它会带着 TensorRT 10 一起来） |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | **无动作**，吸纳已关闭（§4） |
| [NVIDIA/TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect) | 已调研并部分落地 → `11-TensorRT-Model-Connect/`（GraphTransform / GpuTelemetry / ParameterizedPlugin / Glossary），另含 `tests/run_tests.py` 的 `--since` 与 `--framework`、`08-Advance/CudaGraph` 的 `case_graph_per_shape`、skill `trt-debug-mismatch` |
| [NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) | 已调研 → [`trt-ep-abi.md`](trt-ep-abi.md)，**暂不做**，见 §3.8。例子目录已于 2026-10-12 清理 |
| [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | 已调研 → [`trtellm.md`](trtellm.md)，已落地 `13-TensorRT-Edge-LLM/StageProfiler/` 与 `05-Plugin/ONNXParserWithPlugin` 的 `case_pytorch_to_onnx_to_trt` |
| [NVIDIA/TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX) | `10-TensorRT-RTX`，R1–R3。注意这是**样例仓库而非 SDK**，不含 `NvInfer*.h` |
| [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) | 主要上游 → §3.2 |
| [pytorch/TensorRT](https://github.com/pytorch/TensorRT) | `06-DLFrameworkTRT`，剩余项见 §3.6 |
| [tensorflow/tensorrt](https://github.com/tensorflow/tensorrt) | **2025 年 2 月归档**，仅作遗留参考 |
| [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT) | 已调研（2024-03，TRT 9）。两条已落地：`04-Feature/Refit` 的 `case_refit_only_what_changed`（哈希比对只 refit 变了的权重）、`04-Feature/ProgressMonitor` 的 `case_tqdm` + `case_cancel`。其余不做 |
| [NVIDIA/tensorrt-laboratory](https://github.com/NVIDIA/tensorrt-laboratory) | 已调研，**不做**。2020-09，TRT 5/6 时代：`IPluginFactory`、`enqueue()`/`enqueueV2`、隐式 batch，在 TRT 11 上编译不过。真正的价值（allocator / fiber 调度）与 TensorRT 无关 |
| [NVIDIA/object-detection-tensorrt-example](https://github.com/NVIDIA/object-detection-tensorrt-example) | 已调研，**不做**。2019-06，`trt.UffParser` + `max_workspace_size` + `max_batch_size` + `IInt8EntropyCalibrator2` + `build_cuda_engine`，没有一行 TensorRT 调用今天还能跑 |
| [NVIDIA/healthcare-on-tap-TRT-TRITON-demo](https://github.com/NVIDIA/healthcare-on-tap-TRT-TRITON-demo) | 已调研，**不做**。2021-02，TRT 7 + Triton v1（上游自己让人去看 v2_api 分支），且依赖 MONAI 与两个需申请的医疗数据集 |

### 6.2 本目录的其他文件

| 文件 | 内容 |
| ---- | ---- |
| [`trt-ep-abi.md`](trt-ep-abi.md) | TensorRT-RTX-EP-ABI 调研（结论：暂不做，见 §3.8） |
| [`trtellm.md`](trtellm.md) | TensorRT Edge-LLM 调研与落地状态 |
| [`chatglm-6b.md`](chatglm-6b.md) | ChatGLM-6B 的调研笔记 |

**TensorRT-Model-Connect、Stable-Diffusion-WebUI-TensorRT、tensorrt-laboratory、
object-detection-tensorrt-example、healthcare-on-tap** 五个仓库的详细调研文件已于 2026-10-12 删除，
结论压缩进上面的 §6.1 表格。落地的内容在各自的例子 README 里。

### 6.3 调研旧仓库的一条经验

分水岭在 **TensorRT 10**。隐式 batch、UFF、`max_workspace_size`、INT8 校准器这批 API 在 10 和 11 两个
大版本里被成批移除，所以 2022 年以前的 TensorRT 示例代码基本整体失效 —— 不是风格过时，是编译/运行
不过去。调研这类仓库时，**先看最后一次提交的日期和基础镜像的 tag**，比读代码快得多，也更能决定要不要
继续读下去。
