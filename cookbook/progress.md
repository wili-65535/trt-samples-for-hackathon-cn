# Cookbook 工作进度

> 断线续做用。新会话先读本文件，找到「下一步」一节直接接着干。
> 路径若无特殊说明均相对 `/work/trt-samples-for-hackathon-cn-wili/cookbook/`。

## 当前任务：扩展 `07-Tool/Polygraphy/More/`（**已全部完成**，2026-08-27）

**目标**：现有 `07-Tool/Polygraphy/` 有 9 个 CLI 子目录，Python API 侧只有一个 `API/`
（`main.py` 是 `CreateConfig` 的 kwarg 清单，`gs_workflow.py` 是写着
`# Do something with gs APIs` 的空壳）。在 `More/` 下补齐 Python API 的范例。

**源码参考**：`/work/trt/TensorRT-GitHub/tools/Polygraphy`（仓库 0.49.27）。
**装的版本**：**0.50.3**，比仓库新，对照时注意。

### 环境（2026-08-27 确认）

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA H100 PCIe 80GB（**单卡**） |
| TensorRT | 11.1.0.106 |
| polygraphy | 0.50.3 |
| torch / torch_tensorrt | 2.13.0a0+9186a08b2c.nv26.07 / 2.14.0a0 |
| onnx / onnx_graphsurgeon / onnxruntime-gpu / onnxslim | 1.21.0 / 0.6.1 / 1.29.0 / 0.1.96 |
| triton | 3.7.1 |

注意：`from cuda.bindings import runtime as cudart`（老写法已失效）。

**⚠️ 容器重启会丢包**（2026-08-27 B5 开工时遇到）：`onnxruntime` / `onnx_graphsurgeon` /
`onnxslim` 不在镜像里，是前面会话 pip 装的。容器一重启就没了，
而 `import tensorrt_cookbook` 依赖它们 → 所有 main.py 都 import 失败。恢复：

```bash
pip install onnxruntime-gpu onnx_graphsurgeon onnxslim
```

### 任务清单与进度

按 A → B → C 顺序做。**每做完一个立即回来更新本表并跑 runner。**

| # | 主题 | 核心 API | 状态 |
| --- | --- | --- | --- |
| A1 | 懒加载器 vs 立即求值两种风格 | `EngineFromNetwork` vs `engine_from_network` | ✅ `More/01-LazyVsImmediate/` |
| A2 | 跨框架输出比对 | `Comparator.run` + `CompareFunc.simple`/`.indices` | ✅ `More/02-ComparingBackends/` |
| A3 | 新的比对函数 | `CompareFunc.distance_metrics`/`quality_metrics`/`perceptual_metrics` | ✅ **已并入 A2**（上游 `01_comparing_frameworks` 本就覆盖，拆开是重复） |
| A4 | 与 TensorRT 原生 API 互操作 | `@polygraphy.func.extend` | ✅ `More/04-ExtendInterop/` |
| A5 | 用 TRT network API 手搭网络 | `CreateNetwork` + `extend` | ✅ `More/05-BuildNetworkByHand/` |
| A6 | INT8 校准 | `trt.Calibrator` | ✅ `More/06-Int8IsNowExplicit/`（**API 已移除，改为迁移指南**） |
| A7 | 动态 shape 与多 profile | `Profile` + `TrtRunner(optimization_profile=)` | ✅ `More/07-ProfilesAndDynamicShapes/` |
| A8 | 在真实数据集上验证 | 直接用 runner，不用 `Comparator` | ✅ `More/08-ValidatingOnADataset/` |
| A9 | 保存/复用推理输入输出 | `RunResults` / `IterationResult` / JSON | ✅ `More/09-SavedInputsAndResults/` |
| A10 | PyTorch 张量直接进 runner | `TrtRunner` 收发 `torch.Tensor`（含 BF16） | ✅ `More/10-PyTorchTensors/` |
| B1 | TRT network 导出成 ONNX-like | `OnnxLikeFromNetwork` | ✅ `More/11-NetworkAsOnnxLike/` |
| B2 | 战术记录与回放 | `TacticRecorder`/`TacticReplayer`/`TacticReplayData` | ✅ `More/12-TacticsAndReproducibility/`（**API 已移除，转向 timing cache**） |
| B3 | 逐层精度控制 | `SetLayerPrecisions`/`SetTensorDatatypes`/`SetTensorFormats`/`PostprocessNetwork` | ✅ `More/13-PerLayerPrecision/` |
| B4 | 调试张量 | `MarkDebug` | ✅ `More/14-DebugTensors/` |
| B5 | 零拷贝设备内存 | `polygraphy.cuda.DeviceArray`/`DeviceView`/`Stream` | ✅ `More/15-DeviceMemory/` |
| B6 | 插件参考实现比对 | `PluginRefRunner` | ✅ `More/16-PluginReference/` |
| B7 | 版本兼容 / lean runtime | `LoadRuntime` + `CreateConfig(version_compatible=)` | ✅ `More/17-VersionCompatibility/` |
| C1 | 用 Polygraphy 基类写自己的 CLI | `polygraphy.tools` | ✅ `More/18-WritingACliTool/` |
| C2 | 给 `polygraphy run` 加自定义 backend | extension module | ✅ `More/19-CustomBackend/` |

### 约定（沿用 `06-DLFrameworkTRT/Torch-TensorRT/` 的做法）

- 每个候选一个子目录 `More/<Name>/`，含 `main.py` + `README.md` + `unit_test.yaml`
- `main.py` 用 `@case_mark`（来自 `tensorrt_cookbook`）分 case
- **不照抄上游范例**：先跑一遍上游写法、测它宣称的效果，把**对不上的地方**作为主要产出
- 每个结论都要有实测数字或断言撑着，不写没有证据的话
- 不往仓库里放 `.onnx` / `.trt`（`cookbook/.gitignore` 已挡住 `*.onnx` `*.trt` `*.json`）
- 模型优先用 `00-Data/model/` 下已有的，或代码现造

### 已完成的产出

- **A1 `More/01-LazyVsImmediate/`**（4 个 case，runner 60s 通过）
  - 懒 0.05 ms vs 立即 11240 ms；前者返回 `EngineFromNetwork`，后者返回 `ICudaEngine`
  - **坑**：懒加载器**没有记忆化**，调用两次 build 两次（8.35s + 8.29s，对象不同），无警告
  - 懒加载器可 `deepcopy`/`pickle`，`ICudaEngine` 不行 → 这就是懒 API 存在的理由
  - 改网络：立即式直接改；懒式要 `@func.extend`
  - 注意：Polygraphy 每次 build 都打印整张 config 表，脚本里用
    `G_LOGGER.module_severity = G_LOGGER.ERROR` 压掉

- **A2 `More/02-ComparingBackends/`**（5 个 case，runner 25s 通过）**含原 A3**
  - 真实差异 `max |trt-ort| = 7.451e-09`（fp32 舍入）；上游用 identity 模型，两边永远相等，
    读者看不到比对失败长什么样。这里所有比对都对着这个真实数字跑
  - `simple(atol=1e-8)` PASS / `1e-9` FAIL —— 一个数量级就翻盘
  - `distance_metrics` L2=9.80e-09 cos=1；`quality_metrics` PSNR=144.69dB SNR=140.52dB；
    收紧后同样 FAIL —— 它们**不是更宽松**，是判断整个张量而非最差单点
  - **坑1**：`CompareFunc.indices` 是给 Top-K 输出用的，直接套到 logits 上会把浮点当索引比，
    必然 FAIL。正确用法 `PostprocessFunc.top_k(k={"y": (5,1)})` 再比。
    `k` 要按输出名限定，`z` 是 1-D，`top_k` 在 axis=1 上会抛 `AxisError`
  - **坑2**：`perceptual_metrics` 缺 `lpips` 包时**仍然返回 PASS**，那个 PASS 无意义。
    脚本里改成检测到缺包就 SKIP
  - **发现**：`CreateConfig(fp16=True)` 在 TRT 11 上**抛 `PolygraphyException`**
    （ONNX 解析出的网络是 STRONGLY_TYPED）。与 Torch-TRT 的 `enabled_precisions`
    静默忽略**正好相反** —— 已在两边 README 交叉引用
  - 注意：`CreateConfig` 的校验发生在**应用到 network 时**，光 `CreateConfig(fp16=True)()`
    只会得到 `TypeError: missing 2 required positional arguments`，要真 build 才看得到拒绝

- **A4 `More/04-ExtendInterop/`**（4 个 case，runner 42s 通过）
  - `@func.extend(Loader())` 机制：函数收 loader 的产物、不用 return，链仍是懒的
  - `config.set_flag(BuilderFlag.REFIT)` 与 `CreateConfig(refittable=True)` 等价
  - **重要发现**：**`trt.BuilderFlag` 里 `FP16`/`INT8`/`BF16` 在 TRT 11 全被移除**，
    只剩 `TF32`。所以上游这个范例里 `config.set_flag(trt.BuilderFlag.FP16)` 那一行
    在 TRT 11 上直接 `AttributeError`——**上游范例本身跑不了**。
    也说明 A2 里 `CreateConfig(fp16=True)` 的拒绝不是保守，是底层 API 没了
  - **坑**：`TrtRunner` 复用输出缓冲区。留着 `outputs["y"]` 的引用再 `infer()` 一次，
    那个引用会**变成新结果**，旧值无声丢失。要跨调用留存必须 `copy.deepcopy`

- **A5 `More/05-BuildNetworkByHand/`**（4 个 case，runner 通过）
  - `CreateNetwork()` 默认就是 **STRONGLY_TYPED**，且没有关掉它的 flag
  - **闭环**：FP16 靠 `add_input(dtype=trt.float16)` + fp16 权重得到，**不碰 config**。
    这是 02/04 那条「FP16 不再是 builder flag」的正面答案
  - 类型不匹配（bfloat16 输入 + float16 权重）**build 直接失败**，
    报 `ElementWiseOperation SUM must have same input types`——弱类型时代会静默插 cast

- **A6 `More/06-Int8IsNowExplicit/`**（3 个 case，runner 通过）
  - **整个 INT8 校准 API 在 TRT 11 上已删除**：`BuilderFlag.INT8`、`IInt8Calibrator`、
    `IInt8EntropyCalibrator2`、`IInt8MinMaxCalibrator` 全部 REMOVED
  - `CreateConfig(int8=True)` → `PolygraphyException`；
    `CreateConfig(calibrator=Calibrator(...))` → `AttributeError`（**从 TRT 内部抛的**）
  - **关键**：`from polygraphy.backend.trt import Calibrator` **能 import、能构造**，
    只有 build 时才失败 → 照上游 `04_int8_calibration_in_tensorrt` 写的代码看着没问题、一跑就炸
  - 替代路径：QDQ 显式量化。`00-Data/model/model-trained-int8-qat.onnx` 有 8+8 个 QDQ 节点，
    scale 是 initializer，**不向 builder 传任何东西**
  - 实测 float vs qat：logits 差 **9.96**（本该差），但 **argmax 相同** ——
    量化模型该问的是排序是否保住，不是逐元素容差

- **A7 `More/07-ProfilesAndDynamicShapes/`**（4 个 case，runner 通过）
  - 一个 engine 装 3 个 profile，权重只存一份
  - `TrtRunner(optimization_profile=N)` 绑定后，**适用范围是该 profile 的，不是 engine 的并集**
    （profile 0 拒绝 batch 4，尽管 profile 1 支持）
  - **诚实的负面结果**：pinned 0.601ms vs dynamic 0.607ms = **1.01x**，
    小模型上 profile 调优几乎不影响。上游只建三个 profile 就收尾，从不测量
  - profile 只改 tactic 不改数学：`max |diff| = 7.451e-09`（与 A2 同一个舍入）
  - 模型用 `00-Data/model/model-trained.onnx`，输入是 `['nBS',1,28,28]` 动态 batch

- **A8 `More/08-ValidatingOnADataset/`**（4 个 case，runner 通过）
  - 数据集 `00-Data/data/TestData.npz`：500 张 MNIST，**标签是 one-hot**，要 `argmax`
  - **更重要的理由不是大小**：`Comparator` 只比较 runner 之间，**没有放 label 的地方**。
    直接循环得到 accuracy **194/200 = 97.0%**
  - 大小论断实测：`Comparator.run` 保留 200 个 IterationResult 共 **9600 B（48 B/迭代）**。
    外推：10000 迭代下 MNIST 仅 0.4MB，但分割模型 (1,3,1024,1024) 要 **120 GB**。
    注意：**用 `ru_maxrss` 测不出来**（高水位 + MNIST 输出太小），要直接累加 `nbytes`
  - 批处理：batch 1 → 135.9ms，batch 50 → 11.6ms，**11.7x**，准确率不变
  - 坑：默认 profile 只接受 batch 1，批处理要先建够宽的 `Profile`

- **A9 `More/09-SavedInputsAndResults/`**（4 个 case，runner 通过）
  - 上游只演示**加载** CLI 产生的文件；这里从 Python **生成**再读回，往返 bit-exact
  - `load_json` 用于普通对象，Polygraphy 自己的类型用 `save`/`load`
  - **CLI↔Python 打通**：`RunResults.load` 能直接读 `polygraphy run --save-outputs` 的产物。
    注意 CLI 的 runner 名带时间戳（`trt-runner-N0-08/27/26-00:52:01`），**不要写死**
  - `RunResults.add(iterations, runner_name=)` 可把两个进程的结果并成一个对象再比对
  - **查证过的事实**：Polygraphy **默认 data loader 是确定性的** —— 两个独立进程
    生成的输入逐比特相同（`DataLoader(seed=None)` 但实际有固定种子）。
    这才使得跨进程比对有意义。初稿我写成「不该一致」，与输出的 True 自相矛盾，已改
  - CLI 调用要用 `polygraphy` 可执行文件，`python3 -m polygraphy` 会报
    `No module named polygraphy.__main__`

- **A10 `More/10-PyTorchTensors/`**（5 个 case，runner 通过）
  - numpy 进 → ndarray 出；torch 进 → Tensor 出（默认在 CPU，
    `copy_outputs_to_host=False` 则留在 cuda:0）
  - **未文档化的坑**：`copy_outputs_to_host=False` 返回什么，
    由 runner **第一次** infer 的 feed dict 决定，不是当前这次。
    先喂 numpy 再喂 torch → 返回 `DeviceView`（没有 `.device`/`.cpu()`），
    下游 torch 代码在离原因很远处炸 → **一个 runner 只喂一种数组类型**
  - BF16 端到端可用（手搭 bf16 网络 + `torch.bfloat16` 张量，exact match）
  - **关于「NumPy 没有 BF16」要小心**：干净解释器里 `np.dtype('bfloat16')` 抛 TypeError，
    但 `ml_dtypes`（本镜像的传递依赖）被 import 后就能解析。依赖它是碰运气。
    `torch.bfloat16` 无需注册——这才是用 torch 的稳妥理由
  - 上游这个范例同样用了 `Calibrator`+`int8=True`，**在 TRT 11 上跑不了**（第三个失效的上游范例）

- **B1 `More/11-NetworkAsOnnxLike/`**（3 个 case，runner 通过）
  - **parser 插了一堆东西**：源 ONNX 12 节点 → TRT network 27 层。
    `Gemm`→`MATRIX_MULTIPLY`+`ELEMENTWISE`，`ArgMax`→`TOPK`+`SQUEEZE`，
    `Reshape` 拖出 `SHAPE`/`CAST` 链。build log 里的层名对不上 ONNX 时就该看这个
  - 确实不是合法 ONNX：`onnx.checker` 报
    `No Op registered for CONVOLUTION with domain_version of 11`
  - **initializer=0 不等于没权重**（我初稿写错了，实测纠正）：TRT 层的所有参数
    都变成 ONNX **attribute**，权重也一样（首个 `CONVOLUTION` 带 kernel=800 floats、
    bias=32 floats）。protobuf 存 `FLOATS` 列表远不如 initializer 的 `raw_data` 紧凑，
    所以**导出文件比源模型还大**：15.6 MB vs 12.5 MB。大模型上要当心
  - **与 cookbook 自己的 `export_network_as_onnx` 正面对比**（同一个 `Loop`+`If` 网络）：
    Polygraphy 在 `gs.export_onnx` 里抛
    `ValueError: Could not infer the attribute type from the elements of the passed Iterable`，
    cookbook 的写出全部 38 层。**有控制流就只能用 cookbook 那个**
  - `unit_test.yaml` 的 `clean` 要删 `*.onnx`（16 MB 产物）

- **B2 `More/12-TacticsAndReproducibility/`**（5 个 case，runner 65s 通过）
  - **tactic replay 整条路没了**：`trt.IAlgorithmSelector` 及整个 `IAlgorithm*` 家族
    在 TRT 11 全部移除，`IBuilderConfig.algorithm_selector` 也没了。
    Polygraphy 那边同时标了 deprecated（0.55.0 移除）。
    `TacticRecorder(...)`/`TacticReplayer(...)` **构造就抛**
    `AttributeError: module 'tensorrt' has no attribute 'IAlgorithmSelector'`
  - 但 `TacticReplayData()` 能构造（只是个 OrderedDict，不碰 TRT）→
    与 A6 的 `Calibrator` 同一种坑。**第 4 个跑不了的上游范例**
  - **重要实测坑**：`EngineFromNetwork(..., save_timing_cache=path)` 是**只写的**。
    build 后写盘（还会和已有文件 combine），但**从不读回来喂给下次 build**。
    无 cache 8.43s / 传 kwarg 第一次 8.36s / 第二次 8.38s = **1.00x，完全没加速**，
    而文件是满的、内容也对 → 典型"看着对、实际没生效"
  - 正确做法：`CreateConfig` 没有 load 的 kwarg，要用 `@func.extend` 手动
    `config.set_timing_cache(config.create_timing_cache(bytes), ignore_mismatch=False)`
    → 8.38s → **7.05s，1.19x**
  - **测量方法学**：进程内第一次 build 含 CUDA 初始化（11.22s vs 8.4s）。
    不先跑一次丢弃的 warm-up，就会看成"第二次快了 = cache 生效了"——会得出完全相反的结论
  - **怎么才能发现**：`BuilderFlag.ERROR_ON_TIMING_CACHE_MISS`，
    没命中就直接 build 失败（`Assertion !errorOnMiss failed`）。CI 里该开
  - `ITimingCache.queryKeys()`/`query()` 就是现在的 tactic 记录（本例 30 条，
    含 tacticHash + timingMSec）。**强制换用另一个 tactic** 要 `EDITABLE_TIMING_CACHE`，
    `04-Feature/TimingCache/` 已覆盖，本例不重复、只交叉引用

- **B3 `More/13-PerLayerPrecision/`**（6 个 case，runner 43s 通过）
  - **前提**：TRT 11 里 `create_network()` 不传任何 flag 也已是 `STRONGLY_TYPED`（flags=1），
    **弱类型网络彻底没了**，也没有关掉它的 flag
  - **坑**：`NetworkFromOnnxPath(strongly_typed=False)` **被静默忽略**，
    flag 查回来仍是 True。无 warning 无 error
  - 四个 loader 三种结局：
    | loader | 结果 |
    | `SetLayerPrecisions` | 干净拒绝：`PolygraphyException: layer precision ... not available on TensorRT version 11.1.0.106` |
    | `SetTensorDatatypes` | 裸 pybind 报错：`AttributeError: property of 'ITensor' object has no setter` |
    | `SetTensorFormats` | **可用**，format 是布局不是类型 |
    | `PostprocessNetwork` | **可用**，通用逃生口 |
  - `trt.ILayer.precision`/`precision_is_set` 已移除；
    `BuilderFlag` 里 `OBEY_PRECISION_CONSTRAINTS`/`PREFER_PRECISION_CONSTRAINTS` 也没了；
    `ITensor.set_dynamic_range` 同样没了
  - **两种移除方式的对照**：`SetLayerPrecisions` 报错带版本号（好），
    `SetTensorDatatypes` 只说"某属性没 setter"（差，离病因两层）。
    这个对照本身值得记——A6 的 `Calibrator`、B2 的 `TacticRecorder` 都属后者
  - **format 也被 dtype 反锁**：`LINEAR`/`CHW32`/`HWC` 能建且 engine 上可见；
    `CHW4`(int8)/`HWC8`(fp16) 报
    `has dataType Float unsupported by tensor's allowed TensorFormats`，
    而"改 dtype"正是 `SetTensorDatatypes` 干不了的事 →
    **解析 ONNX 得到的网络，可达 format 集合由 ONNX 文件决定**
  - `PostprocessNetwork(network, func)` 收任意函数、保持懒求值；
    用它 mark 中间 ReLU 输出成额外 output（`['y','z','relu']`），
    是那三个 loader 都做不到的

- **B4 `More/14-DebugTensors/`**（5 个 case，runner 85s 通过）
  - 端到端可用：`MarkDebug` 标网络 + `trt.IDebugListener` 收回调，
    取到 `relu` 的 (1,32,28,28)、range [0, 272.04]，
    而 **engine I/O 仍是 `['x','y','z']`** —— 这就是它相对 mark_output 的唯一卖点
  - 回调里的指针**只在回调期间有效**，要留就得在回调里 copy 出来
  - **坑（我初稿写错、实测纠正）**：`set_tensor_debug_state(name, True)` 是**空操作**，
    `MarkDebug` 已经把状态置 True（`get_debug_state` 查得到）。
    真正必需的只有 listener，**忘了 listener 是静音失败**：推理正常、输出正确、
    captured 为空 dict，读起来像"这张量没被写过"。
    `False` 才是有用的方向——不重建 engine，按需静音单个张量
  - **代价实测（全程没开 listener/debug state）**：
    baseline 0.605ms/125440B → `MarkDebug` 0.943ms/201216B = **1.56x**，
    `mark_output` 0.740ms/75264B = 1.22x。
    **只要标了就付钱，且付在 build 期**（可观测的张量不能被 fuse 掉）→ 调试专用 build
  - `mark_unfused_tensors_as_debug_tensors=True` **确实生效**（捕到 9 个 vs False 的 0 个）。
    **但 `network.is_debug_tensor` 查不到**——哪些张量"未被 fuse"要 build 完才知道。
    我最初就是查错了地方、差点误判成"静默忽略"（与 B3 的 `strongly_typed` 不同，**这个是真生效的**）
  - 捕回来的名字是 TRT fuse 后的内部名（`__myln_k_arg__bb1_3_myl4`），
    只能看出"某个值不对"，定位不到原模型的层

- **B5 `More/15-DeviceMemory/`**（6 个 case，runner 63s 通过）
  - 接口全表：`DeviceView` 只有 `copy_to`/`numpy`/`ptr`/`shape`/`dtype`/`nbytes`；
    `DeviceArray` 多 `copy_from`/`view`/`resize`/`free`/`raw`/`allocated_nbytes`；
    `Stream` 只有 `synchronize`/`free`/`ptr`。`view()` 返回**同一个 ptr**，不分配不拷贝
  - **A10 的伏笔收口**：那个没有 `.device`/`.cpu()` 的对象就是 `DeviceView`，上表就是它的全部
  - **零拷贝是真的**：`context.get_tensor_address("x")` 作证 ——
    numpy 喂 → 绑 runner 自己的 buffer（0x4578020000）；
    `DeviceView` 喂 → 绑我们的地址（0x4578000000），输出逐位相同
  - **数字（本例唯一值得给的）**：4 种组合 × 2 个模型
    | 模型 | host→host | dev→dev | 比 |
    | MNIST 3 KB | 0.582 ms | 0.397 ms | 1.46x（省 0.185 ms） |
    | 手搭 25 MB `y=x+x` | 9.328 ms | 0.290 ms | **32.1x**（省 9.0 ms，97% 是拷贝） |
    → MNIST 上省的几乎不是带宽、是两次拷贝+同步的固定开销。**技术随字节数缩放**，
    1 MB 以下不值得为它承担下面两个生命周期坑
  - **坑1 输出 `DeviceView` 是 runner 的 buffer**：`copy_outputs_to_host=False` 返回的
    view 指向 output allocator，**下次 infer 地址不变**。留着它跨一次 infer，
    读到的是**第二次**的结果。比 A4 的 host 版更难发现（view 根本没有"内容"可看）
  - **坑2 view 不持有内存**（两种死法都是 owner 自己干的、都无声）：
    `free()` 后下一次分配落回同一地址 → stale view 读到别人的张量（实测 1.0 → 9.0，无任何报错）；
    `resize()` 变大内部是 free+malloc → 之前的 view 也悬空，新分配落在别处时读它
    **直接 SIGSEGV**（子进程验证 returncode -11，3/3 稳定，Python 层没有 traceback）。
    注意这与 `with DeviceArray(...) as a:` 的习惯冲突：view 逃出 with 块就已经悬空
  - **坑3 `stream=` 在 pageable 内存上根本不异步**：256 MB H2D，
    numpy 的 `copy_from(buf, stream)` **阻塞 22.6 ms** 才返回，pinned(torch) 只要 0.102 ms。
    后果：忘了 `synchronize()` 在 numpy 上**观察不到**（D2H 同步前数据 100% 已到），
    换成 pinned 立刻炸（同步前只到 2.5%）→ **优化的那天旧 bug 才开始发作**
  - `resize()` 从不缩小分配：`resize((1,3))` 后 `nbytes 12` 而 `allocated_nbytes 24`
  - **读 `.dtype` 本身已 deprecated**（0.55.0 移除）：现在返回 numpy dtype，将来返回
    Polygraphy `DataType` → `np.empty(..., dtype=view.dtype)` 是有保质期的代码

- **B6 `More/16-PluginReference/`**（5 个 case，runner 29s 通过）
  - **名字两个词都不对**：`PluginRefRunner` 不加载任何插件，且 `OP_REGISTRY` 只有
    `Identity`/`InstanceNormalization`/`MeanVarianceNormalization` 三个。
    它跑**整张图**，一个没注册的节点就停 —— 连 `Conv`、`Mul` 都不支持
  - **最贵的坑：`LoadPlugins` 用 `ctypes.CDLL`，注册不了 `IPluginCreatorV3One`**。
    实测 creator 数：启动 16 → `LoadPlugins` 后仍 16 → `init_libnvinfer_plugins` 后 44
    （AddScalar 仍不在）→ `trt.get_plugin_registry().load_library()` 后 45，**才有**。
    `LoadPlugins` 还会 INFO 打印 "Loading plugin library"，报错在两层外的 ONNX parser：
    `Plugin not found, are the plugin name, version, and namespace correct?`
    —— 把人引去查名字/版本/namespace。**CLI 的 `--plugins` 同一条路，同样加载不了**
    （实测 `polygraphy run --trt --plugins ...` exit 1）。
    正确写法就是 `tensorrt_cookbook/utils_plugin.py` 里已经在用的 `registry.load_library()`
  - **`register` 没有被导出**：必须 `from polygraphy.backend.pluginref.references import register`。
    签名 `(attrs, *inputs)`，**返回 list**（每个 node output 一项）；常量输入已是 numpy
  - **比对通过说明不了什么**：参考实现硬编码 `1.0`，模型 attr 恰好也是 1.0 → PASS；
    换成 attr=5.0 的模型 → FAIL，`max |diff| = 4.0`。参考一直是错的，只有第二个模型能看见。
    `OP_REGISTRY` 是普通 dict，重复 register **静默覆盖**
  - **真实模型要先切子图**：`gs` 改 `inputs`/`outputs` + `cleanup()`。
    坑：`graph.copy()` 产生**新的 tensor 对象**，拿旧 `Variable` 去赋值会让 `cleanup()` 抛
    `Encountered a node not in the graph` → 必须 `copy.tensors()` 按名字取
  - 用的插件是 `05-Plugin/ONNXParserWithPlugin/AddScalarPlugin.so`（`main.py` 缺了会自己 make）

- **B7 `More/17-VersionCompatibility/`**（5 个 case，runner 150s 通过）
  - **`version_compatible=True` 就是把 lean runtime 塞进 plan**：
    13.190 MB → 118.036 MB（8.95x），差值 104.847 MB，
    而 `libnvinfer_lean.so.11.1.0` 正好 104.845 MB（差 1528 B）。
    **是固定 ~105 MB 不是百分比**：手搭一层网络 0.011 MB → 104.856 MB = **9221x**
  - `exclude_lean_runtime=True` 把体积**完全**还原（与 baseline 差 0 B），
    然后由 `LoadRuntime(路径)` 在加载时补上 runtime。N 个 engine 共用一个 105 MB 库
  - `exclude_lean_runtime` 不配 `version_compatible` → Polygraphy 自己拦下并说清怎么办
    （与 B3 那种 pybind 深处报错形成对照）
  - **代价在加载不在推理**：反序列化 3.71 → 127.70 ms = **34.4x**；
    推理 0.578 vs 0.572 ms，差 6 us 是噪声。所以伤的是进程启动/镜像体积，不是吞吐
  - **裸 `trt.Runtime` 反序列化 VC engine 返回 `None`（不是抛异常）**：
    `Cannot deserialize engine with lean runtime since getEngineHostCodeAllowed() is false`。
    Polygraphy 的 `EngineFromBytes` 会替你设 `engine_host_code_allowed = True`
    （包在裸 `try/except AttributeError` 里）。
    **名字反直觉**：**含** lean runtime 的那个被裸 runtime 拒绝，**排除**的那个反而能加载
  - `hardware_compatibility_level` 是另一条轴：`AMPERE_PLUS` 只多 0.46 MB、延迟看不出差别
    （诚实结论：MNIST 用不到被限制掉的 kernel，不代表 transformer 也这样）
  - **一条过期建议**：开 VC 时 Polygraphy 提示要加 `NATIVE_INSTANCENORM`，
    TRT 11 上实测加不加**产物字节数完全一样**（104860708 B）

- **C1 `More/18-WritingACliTool/`**（5 个 case，runner 17s 通过）
  - 产出一个真的工具 `plan-size`（可执行文件，50 行）：建 engine 并报告耗时与 plan 大小
  - **订阅 arg group 就白拿 CLI 选项**：空工具 7 个（`LoggerArgs` 是基类自动订阅的）→
    +`DataLoaderArgs` 15 个 → 9 个订阅 **95 个**。
    `plan-size` 自己只声明 `--json`，却能吃 `--version-compatible`（B7 的 118 MB 从它跑出来）
  - **坑：依赖关系只写在 docstring 的 `Depends on:` 里，没有任何校验**。
    漏订阅 `OnnxInferShapesArgs` → 参数解析成功、模型读完、然后
    `KeyError: <class ...OnnxInferShapesArgs>`，从 `ArgGroups.__getitem__` 抛出，7 层深
  - `main()` 里是 `sys.exit`，进程内跑要自己拼
    `setup_parser` / `parse_args(列表)` / `parse` / `run`
  - 基类是 `run_impl`，**上游范例覆盖的是 `run`**（能跑，但跳过了打印模块版本、None→0）；
    两个都不写 → `NotImplementedError`。没有类 docstring → `AssertionError`（`-O` 下变成空 help）
  - **不能给 `polygraphy` 可执行文件加子命令**：`tools/registry.py` 是 import 时执行的
    一串硬编码 `try_register_tool`，没有 entry point。整个 CLI 唯一的扩展点是
    `polygraphy.run.plugins`（见 C2）

- **C2 `More/19-CustomBackend/`**（7 个 case，runner 27s 通过）
  - 产出可安装包 `extension/polygraphy_cookbook_ref`：加 `--cookbook-ref`，
    一个 NumPy 解释器，支持比 `--pluginref` 多的 op，且能选 float32/float64 工作精度
  - **必须真安装，PYTHONPATH 不够**：entry point 来自已安装的 dist 元数据。
    包能 import、选项就是不出现，**没有任何提示**。
    半吊子状态：build 一次会留下 `*.egg-info`，那也是元数据 → `pip uninstall` 后
    PYTHONPATH 又"能用了"，其实是残留产物（main.py 卸载时会一并删掉）
  - entry point group 名固定 `polygraphy.run.plugins`，只对 `polygraphy run` 生效，
    `convert`/`inspect` 都看不到
  - **runner 是被生成的脚本调起来的**：`add_to_script_impl` 是往脚本里加行，
    `--gen-script -` 可以把那段代码打出来
  - **float64 参考的价值**：模型 `(x/3 + 1e7) - 1e7`，float32 下 onnxrt 与 float32 参考
    `max_absdiff=0` 完全一致 —— 两个后端一致**不等于**结果对。
    换 float64 参考后 `max_absdiff=0.29271, max_reldiff=1`（全军覆没），
    TRT 与 onnxrt 给出完全相同的错误答案 → 错在模型不在后端
  - **import 即注册**：`__init__.py` 里 `register("AddScalar")`，装上之后
    `--pluginref` 就能跑 AddScalar（B6 说的"没有 CLI flag"的唯一出路）；
    卸载后立刻恢复报错。反过来说，装扩展模块 = 执行它的代码
  - **选项名必须加前缀**：与内置选项重名会在**构建 parser 时**抛
    `ArgumentError: conflicting option string`，于是 `polygraphy run --help`
    对该环境的所有人都坏掉

### 下一步

**A/B/C 三组 19 个候选已全部完成**（A3 并入 A2，其余 18 个各有一个目录，
编号 `More/01-` 到 `More/19-`）。

**CLI 侧的复查与补齐也已完成**（2026-08-27，见下节），`MultiDevice/` 也已建好。剩下的方向：
- **`MultiDevice/` 的 case 06：多卡实跑**。ONNX 改写部分（case 01~05）单卡已跑通并进 CI，
  缺的是每 rank 一个进程把 `model-TP_tp2_rank<i>.onnx` 建成 engine 跑在 GPU i 上，
  再与单卡 ONNX 的结果比对。**等多卡环境**
- `polygraphy plugin autotune`：需要 TensorRT 能真正加载并计时的插件，`toyPlugin` 只有 pattern
  没有实现，目前只抓了 help
- 把 B6 发现的 `LoadPlugins` 用 ctypes 加载不了 V3 插件这件事反馈给上游

## CLI 目录复查（2026-08-27）

10 个 CLI 目录在 TRT 11.1.0.106 + polygraphy 0.50.3 上**逐个实跑，rc 全为 0**，
问题不在"跑不了"，而在下面三类。已改完并重新生成了全部 `result-*.log`，
`tests/run_tests.py --case 07-Tool/Polygraphy/*` 11 个用例全过。

**实测：CLI 上已死但 help 里还在的选项**（用了才炸，这是最坑的）

| 选项 | 结果 |
| --- | --- |
| `--fp16` / `--bf16` / `--fp8` | `PolygraphyException: ... not available on TensorRT version 11.1.0.106` |
| `--int8` | `AttributeError: module 'tensorrt' has no attribute 'IInt8EntropyCalibrator2'` |
| `--precision-constraints` | 抛（`config.py:226 try_set_flag`） |
| `--layer-precisions <真实层名>:float16` | 抛；**写不存在的层名则静默 PASS** |
| `--save-tactics` / `--load-tactics` | `AttributeError: ... 'IAlgorithmSelector'` |
| `polygraphy debug precision` | 整个子命令死 |
| 仍可用 | `--tf32`、`--sparse-weights`、`--strongly-typed`、`debug build`、`debug repeat` |

连带：`inspect tactics` / `inspect diff-tactics` 在 TRT 11 上**永远拿不到输入文件**
（唯一生产者 `--save-tactics` 已死）。这三处的 help 抓取保留了，但 `unit_test.yaml` 里加了注释说明。

**修掉的硬伤**

- `Inspect/main.sh` 建了个 `model-trained-FP16.trt`，**命令里没有 `--fp16`**（删标志时漏删文件名），
  而且后面没人用它 —— 白建一个和上一个完全同精度的 engine。已删，改成 `--visual --save-visual` 那一步
- `Convert/data_loader.py` 是 INT8 校准步骤删掉后留下的孤儿（全仓库无人引用）。
  移到 `Run/`，用 `polygraphy run --data-loader-script` 救活；`Convert/` 的 `*.Int8Cache` 残留与 clean 规则一并处理
- `Plugin-TODO/` → **`Plugin/`**：原来 `MODEL_ADDSCALAR` 实际指向 `model-trained.onnx`，
  又往 `--plugin-dir` 拷了个 `.so`（`polygraphy plugin` 只读 `pattern.py`，从不加载 `.so`），
  三个日志全是 `{}`；且 `plugin match` 不给 `-o` 会把 `config.yaml` **写进 `00-Data/model/`**。
  现在 `build_toy_subgraph.py` 造一个能匹配 `toyPlugin` 的图（另植两个诱饵子图，让拒绝理由可见），
  `list`/`match`/`replace` 全部产出真实结果（`{'toyPlugin': 1}` → 5 节点折成 1 个 `CustomToyPlugin`），
  并补了 `unit_test.yaml`（此前 runner 根本不跑这个目录）
- `More/p3.py`~`p7.py`：B5 的临时探针脚本被误提交进仓库（`d8a00d36`），已删
- `Data/main.sh` 缺 `echo "Finish"`；`README.md` 里 `pip install polygraph\` 拼错
- `API/main.py` 那份 `CreateConfig` 全参数清单少了 **`runtime_platform`、`tiling_optimization_level`**；
  7 个已死参数逐个加了 DEAD 注释并指向 `More/` 对应目录
- `API/gs_workflow.py` 从空壳改成真的工作流：gs 把动态 batch 钉成 `[7,2,3,4]` →
  `fold_constants` 把整条 `Shape→ReduceProd→Gather→Concat` 折成常量，**8 → 2 节点**

**补上的覆盖缺口**（都能进 CI、单卡即可）

| 位置 | 新增 |
| --- | --- |
| `Surgeon/` | `weight-strip` / `weight-reconstruct`（重建出的是 **proxy 权重**，不是原值） |
| `Inspect/` | `inspect model --visual --save-visual`（81 KB / 100 KB 独立 HTML）。**坑：只给 `--save-visual` 不给 `--visual` 什么都不写、也不报错** |
| `Debug/` | `debug build --until 3 --artifacts`、`debug repeat`。**坑：产物进 `polygraphy_artifacts/{good,bad}/`，不是 `good/`、`bad/`** |
| `Data/` | `data concat`（沿迭代轴拼，与 `merge` 的输入/输出轴正交） |
| `Run/` | `--data-loader-script`（真实 MNIST 而非随机数）、`--warm-up`、`--check-error-stat`、`--postprocess y:top-1 --compare indices`（`More/02` 那个坑的 CLI 正解） |
| `Convert/` | `--convert-to onnx`（**没有 `--fold-constants`，会报 Unrecognized Options**）、`--fp-to-fp16`（只改权重与内部张量，I/O 仍是 float32，边界插 `Cast`） |

**`MultiDevice/`（2026-08-27 已建，多卡执行部分待办）**

关键发现：**切分是模式驱动的，CP 和 TP 找的根本不是同一个东西**
（`polygraphy/tools/multi_device/subtool/shard.py`）：

| | CP | TP |
| --- | --- | --- |
| 切什么 | **序列** | **权重** |
| 认什么模式 | `MatMul(Q,K)→Softmax→MatMul(·,V)` | **SwiGLU MLP**（以 `Sigmoid` 为锚），或 `AttentionPlugin` 节点 |
| 产物 | 1 个文件，initializer 不变，插 **6** 个 `DistCollective` | **每 rank 一个文件** `_tp<N>_rank<i>.onnx`，`w_gate`/`w_up` (8,16)→(8,8) 按列切、`w_down` (16,8)→(8,8) 按行切，插 **1** 个 `all_reduce` |

- `00-Data/model/` 里**没有任何模型含这两种模式**，所以 `build_transformer_block.py` 现造一个
  10 节点的 block（attention + SwiGLU MLP，`B=1 S=4 H=8 I=16`），且是能在 onnxruntime 跑的真模型
  —— 将来多卡比对时单卡参考是免费的
- **最大的坑：模型里没有那两个模式时，整条流程静默变成"复制一份"**。拿 MNIST 跑，
  hints 里 `attention_layers: []`、`inputs/outputs` 空，sharder 报成功，输出 12 节点 → 12 节点、
  0 个 collective，**全程无任何 warning**。要看的是 JSON 里的 `attention_layers`，不是退出码
- **rank 数来自 `--nb-rank` 而不是 `--gpus`**：只给 `--gpus 2`，TP 会老老实实写一个
  `_tp1_rank0` 的原样副本；`--gpus` 只填 `dist_collectives.group_size`
- `template shard-hints` 的 `-o` **必须 `.json` 结尾**，且这个检查发生在模型已经加载分析**之后**
- `shard --one-shot` 与「先 `shard-hints` 再 `-s hints.json`」结果一致（实测 16 节点 / 6 个
  collective 完全相同），只有三个输入张量的访问顺序不同
- `DistCollective` 是 NCCL 集合通信、不是标准 ONNX，onnxruntime 跑不了 → **case 06 必须多卡**，
  已在 `main.sh` 末尾写清缺什么、去哪找多卡管线（`05-Plugin/NcclPlugin/`、`08-Advance/MultiDevice/`）

## `07-Tool/trtexec` 复查（2026-08-27）

对照上游 `samples/trtexec/`（本地 checkout `/work/trt/TensorRT-GitHub`）与**实际安装的 trtexec
11.1.0.106** 逐项核对。**装的版本比目录里假设的新**：`Help.txt` 还是 v11.0.0.114 抓的，
而 README 里写着「11.2 才有、本机没跑过」的两个特性，11.1 就已经有了 —— 于是那两步真的被执行了，
**而且两步都是坏的**：

| 问题 | 证据 |
| --- | --- |
| 步骤 11 精度校验 **直接失败**，`set -e` 让整个 main.sh 中断（后面步骤全没跑） | `[E] When using --refPair, you need at least two pairs of I/O.` —— 例子里只给了一个 `--refPair=0` |
| 步骤 12 调优表达式**知识性错误** | `[E] Failed to parse --tuneBuildRoutes expression: Unknown knob: -builderOptimizationLevel` |

**tuner 的 knob 不是 trtexec 的 build 选项**，而是 209 个**编译器内部 knob**
（`--helpBuildRoute`，tuner_version 2.19.45），如 `-conv_lowering=[on|off]`、`-kgen:tiling=[0|1|2]`。
改对之后实测：`fast` 展开 4 条路线、`full` 展开 6 条；4 条路线 gpu_time 只差 **1.00%**
（28.36~28.64 us），即 MNIST 上花 1 分钟搜索**什么也换不到** —— 诚实结论写进 README。
tuning cache 是 JSON-lines，第一行 metadata 里的 `default_build_route` 是看 TRT 默认值的最好入口。

精度校验改对后补了三件事：`--accuracyThreshold` **是必填**（一旦用了 `--loadRefOutputs`/`--refPair`）；
单对时**不要**加 `--refPair`；以及五种算法在**同一份错配**上的读数完全不可比 ——
L0=1.0、L1=0.001432、L2=0.000003、LInf=0.003492、Cos=0.001497，
所以 `--accuracyThreshold=1e-3` 下 L2 **通过**而其余四个失败。还加了一个**故意失败**的用例
（喂 a 的输入配 b 的参考输出），否则「一直 PASS」证明不了校验在工作。

**关于「TRT 11 废除 FP16 等选项」**（本次任务的起点）：trtexec 侧**不需要删任何东西**，
main.sh 本来就没用；而且 trtexec 的处理比 polygraphy 干净得多 ——
`--fp16`/`--int8`/`--best`/`--precisionConstraints`/`--layerPrecisions` **在 help 里根本不存在**，
用了直接 `[E] Unknown option: --fp16`；polygraphy 则是 help 里还在、build 时才抛。已在两边 README 交叉引用。

**新增覆盖**（对齐上游 README 的 Example 5/6 + TRT 11 特性）：

- 步骤 13 `--stronglyTyped`：上游 Example 6 教你加这个 flag，**TRT 11 上是空操作** ——
  不加任何 flag 的默认 build 就打印 `Precision: Strongly Typed`（见 result-02.log）
- 步骤 14 多流吞吐（上游 Example 5）：`--infStreams` 1/2/4 → 15473 / 26030 / 42889 qps，
  中位延迟 0.0632 → 0.0737 → 0.0894 ms，**2.77x 吞吐换 1.4x 延迟**。
  （上游用的 `--streams` 仍被接受，但文档里已改名 `--infStreams`）
- 步骤 15 权重剥离：13,250,084 B → 157,308 B（**84x**）。
  **坑：剥离后的 engine 照样能加载、能跑、报 PASSED，输出全 0，没有任何 warning**；
  且 CLI 侧 `--refitFromOnnx` 不打印任何日志、输出仍是 0，`--dumpRefit` 什么都不输出 ——
  能用的是 Python 路径 `04-Feature/WeightStripping/`（同一模型，refit 后输出完全恢复）

其余：`Help.txt` 重新抓（v11.0.0.114 → 11.1.0.106）；`unit_test.yaml` 加 `slow` 标签与
`timeout: 3600`（tuner 每条路线 fork 一个子 trtexec）。
上游 `tracer.py` 用的 `startInMs`/`inMs`/`outMs` 与现版 trtexec 的 `startH2dMs`/`h2dMs`/`d2hMs`
不一致这件事，之前 `parse_export_json.py` 已经兼容，README 里保留并补齐成一张「上游哪些地方过期了」的表。

## `07-Tool/TritonServerDeploy`（2026-08-27）

原来只有一个 `print("Finish")` 的空壳，现在是完整的部署范例：`main.py` 按
`start_server` / `wait_for_server` / 客户端请求 / `finally` 收尸的形状分四个阶段。

| 阶段 | 需要 | 本容器 |
| --- | --- | --- |
| 1 建 engine + 铺 model repository | `tensorrt` | ✅ |
| 2 本地跑一遍拿参考答案 | `+cuda-python` | ✅ |
| 3 起 tritonserver → KServe v2 请求 → 与参考比对 | **`tritonserver` 二进制** | ⏭️ 跳过 |
| 4 关服务、不留进程 | — | 属于阶段 3 |

**本容器是 `nvcr.io/nvidia/pytorch`，没有 tritonserver 也没有 docker。**
查证过：PyPI 上的 `tritonserver` 包只有 Python binding 的 `.so`（整包 25 个文件，
无 `libtritonserver.so`、无 backend），`pip install` 装不出服务端。
**但后来把服务端弄起来了，四个阶段全部实跑通过，见下一节。**

为了不让「跑不了的那一半」烂掉（正是 trtexec 里抓到的那类问题），加了 `test_client.py`：
用只讲 KServe v2 的 stub server 顶替 tritonserver，驱动**真实的客户端代码** ——
就绪轮询（前两次故意 503）、metadata、infer 请求编码/响应解码、与参考比对、
`finally` 里的 terminate，**16 项检查全过**。测不到的只有「Triton 认不认这份 `config.pbtxt`」，
作为部分替代，最后一个 case 拿 config 与 engine 的真实张量形状交叉校验。

`config.pbtxt` 的三个坑（都写进 README）：
- `max_batch_size > 0` 时 **Triton 占用第一维**，`dims` 写单样本形状
  （engine 的 `x` 是 `(-1,1,28,28)`，config 里写 `[1,28,28]`）—— 测试专门校验这条
- 每样本标量（输出 `z`）不能写空 `dims`，要 `dims: [1]` + `reshape { shape: [] }`
- `max_batch_size` 不能超出 engine 的 profile（本例 min1/opt4/max16）

客户端用纯 `requests` 走 KServe v2 JSON（不引入 `tritonclient`，方便直接拷进 Triton 容器），
README 里给了 `tritonclient` 的等价写法。另强调 **plan 与 Triton 容器的 TensorRT 版本必须一致**，
否则 `Engine plan file is generated on an incompatible version`。runner 用例通过（14s）。

### 免 docker、免 root 跑起 tritonserver（2026-08-27 实测成功）

**不需要编译源码**。容器镜像本质就是 HTTP API 后面的一堆 tar，NGC 对
`nvidia/tritonserver` 发匿名 token，所以直接把需要的两个目录扒出来就行。
脚本固化在 `07-Tool/TritonServerDeploy/install-tritonserver-without-docker.sh`：

| 步骤 | 要点 |
| --- | --- |
| 匿名 token | `https://nvcr.io/proxy_auth?scope=repository:nvidia/tritonserver:pull`。**token 约一分钟就过期**，必须每个 blob 重新取；用过期 token 拿到的是 JSON 错误体，`tar` 报 "not in gzip format" —— 我一开始就是被这个坑了，白扫了好几层 |
| 选镜像 | `26.07-py3` 的 `TRT_VERSION=11.1.0.106`，**与本容器的 TensorRT 完全一致** —— 这才是 main.py 建的 plan 能被加载的原因 |
| 解包 | 46 层里只有一层（1.9 GB）含 `opt/tritonserver`，只取 `bin/`、`lib/`、`backends/tensorrt/` → 装完 **53 MB** |
| 补两个库 | 二进制还缺 `libb64.so.0d`（Ubuntu universe）与 `libdcgm.so.4`（CUDA 源的 datacenter-gpu-manager-4-core），从 `.deb` 里 `dpkg-deb -x` 取出即可，**不需要 root** |
| 写脚本时踩的两个 shell 坑 | ① `set -o pipefail` 下 `tar -tzf ... \| grep -q` 中 grep 提前关管道 → tar 被 SIGPIPE 打死 → 整条管道返回 141，于是**真正含服务端的那一层被判成没有**（要先把清单落盘再 grep）；② `find A -o -name B -exec cp` 的 `-exec` 只绑定到最后一个 `-name`，`libb64` 被找到但没被拷 → 服务端起不来（要加括号） |
| `--backend-directory` | 默认写死 `/opt/tritonserver/backends`；装到别处会报 `unable to find backend library for backend 'tensorrt'`，**报错里不提目录**。已让 `main.py` 从二进制路径推出这个参数 |

脚本从零跑一遍：装完 **60 MB**，`All shared libraries resolve`。实测端到端（H100 / TRT 11.1.0.106 / Triton 2.71.0-26.07）：
`Server ready after 1.0 s` → HTTP 推理 `13.14 ms` → 与本地 TRT 参考
`max |diff| = 0.000e+00, same argmax = True` → `Server stopped with code 0`。

两条限制：宿主 glibc 要够新（本机 Ubuntu 24.04，与镜像一致）；这**不是受支持的安装方式**，
除 `bin/`/`lib/`/一个 backend 外什么都没有（Python / PyTorch / ONNX-Runtime backend 都不在），
要跑多后端还是得用容器。源码编译（`build.py --no-container-build`）也可行但要拉整条工具链，
在已经验证扒包可行的情况下没必要。

## FP16 ONNX 用量统计（2026-08-27）

**`00-Data/model/` 里一个 FP16 ONNX 都没有**（`model-half-mnist.onnx` 的 "half" 是「半张图」
不是半精度，命名本身误导）。需要它的地方分四类：

**A. 已经坏了的**（用了 TRT 11 已删除的 `trt.BuilderFlag.FP16`，一跑就 `AttributeError`；
`trt.BuilderFlag` 现在只剩 `TF32`）。五个都是 `enabled: false`（框架没装），**CI 抓不到**：

| 例子 | 状态 |
| --- | --- |
| `03-Workflow/JAX-ONNX-TensorRT` | `case_normal(is_fp16=True)` 活跃 |
| `03-Workflow/Mindspore-ONNX-TensorRT` | 活跃 |
| `03-Workflow/Paddlepaddle-ONNX-TensorRT` | 活跃 |
| `03-Workflow/TensorFlow2-ONNX-TensorRT` | 活跃 |
| `03-Workflow/OneFlow-ONNX-TensorRT` | 同样代码，fp16 那行已注释 |

**B. 名字叫 FP16、其实是 FP32 的**
- `07-Tool/trex/get_data.py` 的 `model.fp16.*` 用**普通 FP32 ONNX** 建（注释里也承认精度由图决定），
  于是 CompareEngines 实际在比 INT8-QAT vs FP32
- 顺带发现两处**写死的旧仓库路径**（少了 `-wili`，目录不存在）：
  `07-Tool/trex/get_data.py:36`、`07-Tool/trex/11-ProcessEnginePipeline/main.py:45`

**C. 自己会造 FP16 ONNX 的 —— 现成参考实现**

| 例子 | 工具 |
| --- | --- |
| `03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` | **`modelopt.onnx.autocast`**，含 `op_types_to_exclude` |
| `07-Tool/FP16Tuning` | **`modelopt.onnx.autocast.convert_to_mixed_precision`**，逐节点搜索 |
| `07-Tool/Polygraphy/Convert`（今天新增） | `polygraphy convert --fp-to-fp16`（底层 onnxconverter_common） |

**D. 不需要 FP16 ONNX 的**（走 TRT network API / 插件 / torch）：`02-API/CudaEngine`、
`02-API/Layer/Cast`、`04-Feature/DataFormat`、`05-Plugin/UseFP16`、`05-Plugin/CuteDSLPlugin`、
`Polygraphy/More/05`、`More/13`，以及 `tests/NetworkSerialization`、`06-DLFrameworkTRT/Torch-TensorRT`。

**未验证**：`06-DLFrameworkTRT/ONNXRuntime-TensorRT` 的 `trt_fp16_enable`（ORT-TRT EP 选项）在
TRT 11 上是报错还是静默忽略 —— 本容器的 onnxruntime 加载不了 TRT EP
（`Please install TensorRT libraries...`），需要对 TRT 11 构建的 onnxruntime-gpu 才能定论。

**缺口**：最主线的 `03-Workflow/pyTorch-ONNX-TensorRT` **完全没有 FP16 case**。

**三条生成路径的产物不一样，不是口味问题**：

| 路径 | 产物 | I/O dtype |
| --- | --- | --- |
| `modelopt.onnx.autocast` | 混合精度，敏感节点留 FP32，无需校准数据 | FP32（边界插 Cast） |
| `polygraphy convert --fp-to-fp16` | 权重与内部张量全 FP16（实测：8 个 initializer 转 FP16、多 2 个 `Cast`） | FP32（边界插 Cast） |
| `torch.onnx.export(model.half(), ...)` | 纯 FP16 | **FP16** |

**建议**：默认用 **ModelOpt AutoCast** 生成 `00-Data/model/model-trained-fp16.onnx`
（cookbook 已有两处在用、保留敏感节点、不需要校准数据）；若还要覆盖「I/O 就是 FP16」的场景
（对应 `05-Plugin/UseFP16`、`DataFormat` 那类），再用 `torch.half()` 导出一个
`model-trained-fp16-pure.onnx`，两个文件把「混合精度」与「纯半精度」两种形态都摆出来。
A、B 两类**尚未动手修**。

## `08-Advance/EmptyTensor`（2026-08-27）

原来是 `EmptyTensor-TODO` 空壳，现在是**围绕真实场景**的例子：检测器按分数过滤后一个框都不剩、
推理服务塞进来一个 0 行的 batch —— 这两件事天天发生。TensorRT 本身处理得很好，
例子讲的是**外围程序**出错的三种方式（全部实测于 TRT 11.1.0.106）：

| case | 结论 |
| --- | --- |
| `case_no_detection` | `Greater`→`NonZero`→`Gather` 的检测尾巴。没有框通过时输出 `(0,4)`，同一个引擎同一个调用在 3 个框通过时输出 `(3,4)` —— **空结果是正常结果**，不需要 host 侧的 if 分支 |
| `case_empty_tensor_needs_a_valid_address` | **`cudaMalloc(0)` 成功但返回地址 0**；把 NULL 绑成输入后 `enqueueV3` 返回 `False` 且什么都不跑，输出缓冲区保持原样 |
| `case_reduce_over_empty_axis` | 空轴归约：`SUM=0`，但 **`MAX=-inf`、`AVG=NaN`** |
| `case_profile_must_cover_zero` | shape 里的 0 必须落在 profile 内，`min=1` 时 `set_input_shape([0,2])` 返回 `False` 并保留旧 shape |

**这个例子自己踩了自己要讲的坑**：第一版第三个 case 是通过 cookbook 的 `TRTWrapperV1` 跑的，
而 `utils_class.py` 的四个缓冲区分配处都写着 `cudaMalloc(n_byte)` —— 张量为空时 `n_byte=0`、
拿到 NULL 地址、`enqueueV3` 拒绝执行、输出缓冲区没被写过，于是读回来是三个 0，
差点得出「MAX over nothing = 0」这个**错误结论**。绑对之后才是 `-inf` / `NaN`。
已把 `tensorrt_cookbook/utils_class.py` 的 **4 处** `cudaMalloc(n_byte)` 改成
`cudaMalloc(max(n_byte, 1))` 并加注释（这是共享代码的真 bug，任何喂空张量的例子都会中招）。
回归：`01-SimpleDemo`、`02-API/Layer/NonZero`、`04-Feature/DebugTensor`、
`08-Advance/MultiOptimizationProfile` 均通过。

顺带修了 `04-Feature/EmptyTensor-TODO/unit_test.yaml` 里 `name:` 写成 `08-Advance/EmptyTensor`
的复制粘贴错误（会与本例重名）。

## `08-Advance/MIG`（2026-08-27，决定不写成例子）

`MIG-TODO/` 里那 20 行（跑 `nvidia-smi -L`、数实例、打印一句 `export CUDA_VISIBLE_DEVICES=`）
已删除，目录改名 `MIG/`，**只留一个 README**。理由：MIG 是宿主机配置，
**进程里看一个切片就是一张小卡，没有任何 TensorRT API 因它而不同**，写成例子等于抄 nvidia-smi 手册。

**唯一属于 TensorRT 的那条**（已写进 README）：**engine 要在将要部署的那个 MIG profile 上 build**。
TRT 在 build 时按当时可见的 SM 数与显存挑 tactic（tile 尺寸、split-k、occupancy），
本机实测 profile 两端差 8 倍：`7g.80gb` = 114 SM / 79.25 GiB，`1g.10gb` = **14 SM / 9.75 GiB**。
整卡上 build、切片上服务的 engine 照样加载、照样算对，**只是 tactic 选择彻底盲了**，
而且没有任何东西会报告 —— 最难查的那类性能 bug。`--memPoolSize=workspace` 同理。

**未做的测量**（等有开了 MIG 的机器）：同一个 ONNX 分别在 `7g.80gb` 和 `1g.10gb` 上 build，
都放到 `1g.10gb` 上跑，比吞吐。差距明显则升格为例子；差距在噪声内同样是有用结论，笔记就保持笔记。

本机实测记录：H100 PCIe 80GB **支持 MIG 但 `mig.mode.current=Disabled`**；
容器内 `nvidia-smi -i 0 -mig 1` 报 `Insufficient Permissions`（非 root，且切换 MIG 模式/建实例
本就是宿主机操作，要求 GPU 上无进程）。容器只能在 `docker run` 时被分到已有切片，
**运行期不能重新分区**；切片间 `P2P: No`，NCCL 那套不适用。

## `08-Advance/GreenContext`（2026-08-27，MIG 的进程内替代品）

问「值不值得给 MIG 写例子」时顺出来的方向：**green context（CUDA 12.4+）把一张卡的 SM 切开，
给出一个绑定到分区的 stream，在那个 stream 上启动的一切都被限制在这些 SM 里**。
对 TensorRT 而言不需要任何新 API —— `execute_async_v3(green_stream)` 就是全部集成，
而这正是它危险的地方。全部实测于 H100 PCIe（114 SM）/ TRT 11.1.0.106 / CUDA 13.3。

**1. 分区是真的，而且免费**（8 层 1024×1024 matmul，SM bound）

| stream | 延迟 | 相对整卡 | SM 比 |
| --- | --- | --- | --- |
| default（114 SM） | 0.137 ms | 1.00x | 1.00x |
| green 16 SM | 0.546 ms | 3.98x | 7.12x |
| green 32 SM | 0.299 ms | 2.17x | 3.56x |
| green 64 SM | 0.166 ms | 1.21x | 1.78x |
| **green 114 SM** | 0.141 ms | **1.02x** | 1.00x |

最后一行是关键：**切出全部 SM 的分区不花钱**，机制本身零开销。切分有粒度
（`minSmPartitionSize` = `smCoscheduledAlignment` = 8），要按返回值算实际拿到多少。

**2. 用途：吵闹邻居**（同进程内一个延迟敏感 engine + 一个吃满吞吐的 engine）

| 场景 | median | p95 |
| --- | --- | --- |
| 单独跑 | 0.036 ms | 0.038 ms |
| 有背景任务，都在 default stream | 0.161 ms（**4.53x**） | 0.309 ms（**8.23x**） |
| 有背景任务，green 32 SM / 82 SM | 0.066 ms（1.86x） | 0.115 ms（3.05x） |

**这就是 MIG 的招牌实验，但不用 MIG**：同进程、免 root、免宿主机配置、免重启容器，随建随拆。

**3. 隔离有个洞：TRT 的 aux stream 会逃出分区**（本例存在的主要理由）

背景 engine 被钉在自己的 82 SM、延迟任务在不相交的 32 SM 上，然而：

| 背景 engine 的 build 选项 | 延迟 median | p95 |
| --- | --- | --- |
| `max_aux_streams = 0` | 0.064 ms（1.16x） | 0.103 ms（1.79x） |
| `max_aux_streams = 4` | 0.117 ms（2.15x） | 0.222 ms（**3.86x**） |
| **默认（-1，TRT 自己定）** | 0.112 ms（2.04x） | 0.211 ms（**3.66x**） |

TRT 的 aux stream 是从**当前 context** 创建的，不是 green context —— 而默认值就是 -1，
**不需要谁主动要求就会发生**。要隔离就得用 `max_aux_streams = 0` build，代价是牺牲引擎内并发。

**4. 白捡的 19%：在哪跑就在哪 build**

| engine | 在 32 SM 上的延迟 |
| --- | --- |
| 整卡上 build（第 1 次） | 0.301 ms |
| 整卡上 build（第 2 次，对照组） | 0.301 ms |
| **在 32 SM green context 里 build** | **0.243 ms** |

两次相同条件 build 只差 **0.001 ms**，而分区内 build 的效应是 **0.058 ms（19%）= 抖动的 58 倍**。
**我第一次得出的结论是错的**（早期原型显示"完全相同"），是加了对照组才判定的 —— 对照组现在留在例子里。

机制值得记：**TRT 问不到分区大小** —— 它读的 `cudaGetDeviceProperties.multiProcessorCount`
在 green context 里**仍然报 114**，只有驱动层的 `cuCtxGetDevResource` 知道是 32。
真正适应分区的是**经验式的 tactic 搜索**：候选 kernel 是在当前 context 里计时的，
所以在分区内 build 就量到了分区的真实行为，选出了不同的赢家。
这条正好是 `MIG/README.md` 里那句「在部署用的 profile 上 build」的**实测版**。

**5. 生命周期坑**：在 green context 里 build 的 engine 持有该 context 的 CUDA 资源，
先销毁 context 会让 TRT 析构函数报 `Error Code 1: Cuda Runtime` 然后**在进程退出时 SIGSEGV**，
离出错点很远。case 4 因此故意不释放分区。（case 1~3 可以正常释放：它们的 engine 在 primary
context 里，只有 stream 来自分区。）

**测量方法学教训**：中途有一次崩溃留下的僵尸进程占着 GPU，导致基线从 0.055 ms 变成 1.446 ms、
整组数字失真；发现后 kill 掉重测。跑这类隔离实验前必须先 `nvidia-smi --query-compute-apps` 确认卡是干净的。

## `08-Advance/TensorRTGraphSurgeon`（2026-08-27）

原来只有一个空 README。现在做的是**在 `INetworkDefinition` 层面动刀** —— ONNX parser 之后、
builder 之前，与 `07-Tool/OnnxGraphSurgeon`（parser 之前动 ONNX）互补。
实测于 H100 / TRT 11.1.0.106，模型 `x -> Add(1.0) -> Relu -> Mul(2.0) -> y`，张量 [8,512,512]。

**能用的 API 就这么点**：`num_layers`/`get_layer` 走图、`add_*` 加层、`layer.set_input` 改连线、
`mark_output`/`unmark_output` 挪边界，**没有 `remove_layer`**。

| case | 结论 |
| --- | --- |
| 1 走一遍 parse 结果 | 3 个 ONNX 节点变成 **7 个 TRT 层**；**ONNX 节点名活了下来**（`node_add` 等），这是后面能按名字找层的前提；多出来的 CONSTANT/SHUFFLE 是 parser 在广播标量 |
| 2 追加一层 | `unmark_output` + 加 NEG + `mark_output`；忘了 unmark 不会报错，只是网络变成两个输出 |
| 3 删一层 | 没有删除 API，做法是**把消费者接到生产者的输入上**，让 builder 丢掉没人读的层。`num_layers` 仍是 7（层对象还在），**证据在数字里**：Relu 没了之后出现负值 -4.0 |
| 4 换成自己的 plugin | 用户点名要的那个：**ONNX 本来就能跑**，仍想换成自己的实现。`add_plugin_v3` + `set_input` 改线即可，**输出逐位相同** |

**case 4 的代价必须量**：engine 层数 1 → 2，kernel 时间 0.0060 → 0.0128 ms = **2.13x**。
这不是 plugin 本身慢，而是 TRT 原本把 `Add+Relu+Mul` 融成了**一个 kernel**，plugin 融不进去，
一趟内存访问变成两趟 —— 在带宽受限的链上正好约 2 倍。**换可融合的算子要先算这笔账。**
（计时故意绕开 `TRTWrapperV1.infer`：它的 H2D/D2H 拷贝是 kernel 时间的 ~150 倍，会把效应全盖住。）

**两个生命周期坑，都是撞出来的**：
- **插件库加载后，先建一个 engine 再释放它（重新绑定变量就够），下一次
  `trt.get_plugin_registry().get_creator(...)` 会 SIGSEGV** —— 可复现、无 Python traceback、
  崩在 libnvinfer 里。我用 A~H 八种顺序做了二分才定位到触发条件（H 崩、G 不崩：区别只在于第一个
  wrapper 是否已被释放）。例子因此**先加载库、先取 plugin 对象，并把两个 wrapper 都留着**。
  只记录触发条件与规避方法，未下根因结论。
- 改线之后只能靠**输出数值**验证，不能看层数（层数不变）。

与 `GreenContext` 的坑正好是镜像：那边是先销毁 green context 导致 TRT 析构崩。

## `90-Misc/Number` 数据类型表复查（2026-08-27）

对照 `torch` 2.13、`ml_dtypes` 0.5.4 的 `finfo` 与逐 bit `view()` 复查 `mannuscript.md` 和
`output/*.md`。**所有数值结论均为实测，不是查文档抄的。**

### `mannuscript.md` 中改掉的错

| 位置 | 原值 | 实测 | 说明 |
| ---- | ---- | ---- | ---- |
| FP8E8M0 `0x80` | `2^0=1.0` | **2.0** | bias 127，`0x7F` 才是 1.0 |
| FP8E8M0 NaN | ❌ | **✓** | `0xFF` 是唯一的 NaN（`fnu` 的 `n` 就是 NaN） |
| FP8E4M3 Inf | ❌ | **✓** | 该行描述的是 IEEE 风格的 `ml_dtypes.float8_e4m3`，它保留 Inf、上限 240 |
| FP8E4M3 "OCP standard" | 该行 | **移到 fn 行** | OCP OFP8 的 E4M3 就是 max=448 的 `fn`，不是 240 那个 |
| UINT64 `0x80` | 1.15e+19 | **9.22e+18** | $2^{63}$ |
| Min Positive 列 | 混用 | 拆成 **Min Normal / Min Subnormal** | 原表 FP64 给的是 min normal、FP4E2M1 给的是 min subnormal（0.5），同一列两个含义 |
| FP9E5M4 max | 57344 | 63488 | 抄了 E5M2 的上限；该行连同 E5M0 一并改对 |
| FP7E4M3 | 7 bit / 1-4-3 | **删除** | 1+4+3=8，位数自相矛盾，且查不到这个格式 |
| NVFP8 | 独立一行 | **删除** | NVIDIA 没有叫 NVFP8 的格式；FP8 是 per-tensor scale，不是 block 格式，已在概念表里写清 |
| MXFP8 max | 57344 | 448 或 57344 | OCP 允许 E4M3fn 与 E5M2 两种 element |

补充：加了 **FP6E3M2 / FP6E2M3 / MXFP6** 三行（`output/` 里本来就在生成 FP6，表里却没有）；
FP4E2M1 的 PyTorch 名字是 `float4_e2m1fn_x2`（**打包是类型的一部分**）；新增一节讲 E4M3 的三个
变体（`e4m3` / `e4m3fn` / `e4m3fnuz`，同一 layout 三套取值），以及 `fn`/`uz`/`u` 后缀的含义。

### `output/*.md` 中改掉的错（都改在 `build-number-md.py` 里）

- **FP8E8M0 的 `0xFF` 印成了 `3.402824e+38`**，应为 NaN；最大值是 `0xFE` = 1.70e38。根因是 spec 里
  写了 `has_nan=False`。
- **FP4E2M1 的「Largest number < 1」印成 `1 - 2^{-2}` = 0.75**，而 FP4E2M1 根本没有 0.75。原代码只给
  FP6E2M3 开了特例，其实条件是 `q == 2`（bias=1 时 1.0 下面一档就是次正规数）。
- **IEEE 格式的 NaN 行重复**：qNaN 与「NaN」是同一个 bit pattern，改成非 IEEE 格式才印。
- **`E = 2 - 2^{q-1} = -0`**：q=2 时模板里硬写的负号，改成带符号整数。
- **`Integer.md` 根本不是生成的，而且是错的**：UINT8/INT8 行填的是 16 位的范围（65535 / 32767 /
  -32768），INT16/UINT16 整行缺失，INT64 最小值被抄成 `-9223372036854775800`（末位截断），INT2 最小值
  写成 -1。改为由 `build_integer_md()` 现算。
- `build-number-picture.py` 里 `FP32(E11M23)`、`FP16(E5M11)` 两个标签写错，且 FP16 的位宽给成
  `[1,5,11]`（=17 bit），图会画错。
- `README.md` 里的运行命令写成了 `build_number_md.py`（下划线），实际文件名是连字符。

### 验证

把重新生成的 6 张全值表（FP8E4M3/E5M2/E8M0、FP4E2M1、FP6E2M3/E3M2）逐 bit pattern 与 `ml_dtypes`
对拍：**256/256、16/16、64/64 全部一致，NaN/Inf 位置也一致**。

### torch 有没有更新的浮点类型？

没有。当前（`torch` 2.13 与 main 文档）全部浮点 dtype 是：`float64/32/16`、`bfloat16`、
`float8_e4m3fn`、`float8_e5m2`、`float8_e4m3fnuz`、`float8_e5m2fnuz`、`float8_e8m0fnu`、
`float4_e2m1fn_x2` —— 表里都已覆盖。**没有 `float6_*`**（只有 `ml_dtypes` 有 `float6_e2m3fn` /
`float6_e3m2fn`），也没有裸的 `float8_e4m3`。另外本机 build 里有 `int1..int7` / `uint1..uint7`
这些 shell dtype，以及一个新的复数类型 `bcomplex32`（bfloat16 复数），与本表无关。

## `99-Todo/candidates-eco-tensorrt-llm.md` 复查（2026-08-28，结论：关档）

按「TensorRT-LLM 里有哪些**跟 TensorRT 相关的工具**可以搬进 cookbook」这个标准复查，**原文 16 条候选
全部作废**，且没有新的可搬项，文件改写为关档说明。

- 原文 16 条（quickstart / async / streaming / sampling / 投机解码 / KV-cache 配置 / multi-LoRA /
  guided decoding / 多模态 / trtllm-serve / chat REPL / eval）教的都是 `tensorrt_llm.LLM` 这套
  **PyTorch 后端**的 API，没有一条讲 TensorRT。属于「在 cookbook 里把 TRT-LLM 跑起来」，标准不符。
- 更关键的是**这个仓库的 TensorRT 面已经几乎没了**（实测，非推断）：
  - 全仓非 3rdparty 的 Python 文件里，**只有 9 个还 `import tensorrt`**（注意：直接 grep
    `^import tensorrt` 会把 `import tensorrt_llm` 一起匹配进来，得到 299 个的假象）。
  - 建 engine 那一层整体删除：`builder.py`、`network.py`、`module.py`、`python_plugin.py`、
    `tools/plugin_gen/{core,plugin_gen,shape_infer}.py`、`tools/onnx_utils.py` 都还列在
    `legacy-files.txt` 里但磁盘上不存在（该文件 1301 条里有 451 条已失效）。
  - `_deprecation.py::emit_engine_arch_deprecation` 专门用来在残留的 engine 路径上打
    「legacy TensorRT engine-build workflow，改用 PyTorch backend」。
- 因此三个本来最值得看的插件例子**全部 import 不起来**：`examples/python_plugin/`（依赖已删的
  `tensorrt_llm.python_plugin`，那个 `@trtllm_plugin` + `PluginBase` 的纯 Python 插件写法是这里唯一
  真正 cookbook 形状的点子）、`examples/openai_triton/manual_plugin/`（依赖已删的 `module/builder/
  network`，C++ 侧还是 TRT 11.1 里标了 `TRT_DEPRECATED` 的 `IPluginV2DynamicExt`）、
  `plugin_autogen/`（依赖已删的 `tools/plugin_gen`）。
- 仍能 import 的 5 个只是普通的 torch → ONNX → TRT 建引擎/跑引擎（qwenvl 的 ViT、qwen2audio），
  `03-Workflow/pyTorch-ONNX-TensorRT` 已覆盖且更干净。`scripts/` 全是仓库自身的构建/lint 管道。
- **想要「用 Python 写 TRT 插件」的例子，去 TensorRT OSS 的 `samples/python/python_plugin` 拿**
  （已记在 `candidates-github.md`），不要从这里拿。
- 顺带更新：`candidates-ecosystem.md` 的跨仓表里删掉 6 条 TRT-LLM 行（原 #1/#2/#8/#11/#12/#14）并改写
  主题 1；`99-Todo/README.md` 的生态表把 TRT-LLM 改成 No action。

## `05-Plugin/TritonAOTPlugin`（2026-08-28）

把 TensorRT-LLM 那两个已经跑不起来的例子（Triton kernel 手工包成 C++ 插件 / 自动生成插件）**从零重写**，
不依赖它的任何实现。用的是 Triton 自带的 AOT 工具链（`triton.tools.compile` + `link`，本机 3.7.1
可用），插件按 cookbook 的 `IPluginV3` 写法。

四个 case：

1. 手写插件。AOT 把 kernel 编成**内嵌 cubin 的 C 源码 + `cuLaunchKernel` 包装**，`link` 再给它一个
   不带 hash 的稳定符号名（每个 variant 的文件名里带 content hash，改 kernel 就变，所以必须调
   linker 出来的 `add_scalar_default`）。
2. **实测出 Triton AOT 的一个静默算错 bug**：`fp32` 的 kernel 标量参数，生成的 C 原型写成
   `double`，却把 `&scalar` 直接塞进 `cuLaunchKernel` 的参数数组——kernel 那一格只有 4 字节，于是读到
   double 的低半部分。`1.0` 是 `0x3FF0000000000000`，低半部 = 0，**结果变成 `x + 0`，launch 还返回
   `CUDA_SUCCESS`**。`0.1` 会变成 `-1.59e-23`。修法是 `sed 's/double scalar/float scalar/'`，同时
   调用方声明也要跟着从 `c_double` 改 `c_float`（只改一半仍然错，我第一次就踩了）。case 2 用纯
   ctypes 复现，并对「坏」和「好」两种都下断言，将来 Triton 修了这个 bug 例子会直接失败而不是留着
   一个没用的 sed。
3. 证明「AOT = 不再依赖 Python/Triton」：子进程里把 `import triton` 屏蔽掉重新建 engine 并跑，
   数值正确；`readelf -d` 只有 libcudart/libcuda/libstdc++/libgcc/libc（顺带发现 **libnvinfer 也不在
   NEEDED 里**，插件只实现接口，TRT 加载时才解析）。
4. `plugin_gen.py`：从一个 9 键的 spec 生成 339 行 C++ + Makefile 并编译。故意换成 **GELU**（不同
   kernel、不同参数表、没有 attribute）来证明生成器不是把隔壁那个文件抄一遍。更重要的是把**多个 AOT
   variant 映射成 TensorRT tactic**：3 个 `BLOCK_SIZE`/`num_warps` 组合 → 3 个 tactic，builder 计时后
   选中 `0x3`（BLOCK=4096/warps=8），**比固定用第一个 variant 快 1.83x**（0.0134 vs 0.0245 ms，两次
   运行都是 1.83~1.84x）。等于把 Triton 的 autotune 挪到 build 期，部署进程里没有 autotuner。

踩到并记录的坑：

- **插件库必须同时导出 `setLoggerFinder` 和 `getCreators`**。只导出后者不报链接错误、`loadLibrary`
  也正常返回 handle，但 creator 根本不会注册，错误延迟到 `Cannot find plugin: ...` 才爆出来。
- 带 `:16` 对齐提示时，linker 生成的 dispatcher 会检查 `ptr % 16 == 0`，不满足就**不发射 kernel**
  直接返回 `CUDA_ERROR_INVALID_VALUE`——不查返回值的话输出 buffer 保持原样。
- `TacticValue` 只有在 `profiling_verbosity = DETAILED` 时才出现在 engine information 里。
- Makefile 里 `$(wildcard)` 对「recipe 运行时才生成的文件」无效（目录列表在 parse 期就缓存了），
  要用 shell 的 `$$(ls ...)`。
- 另外看到但没动的两处 Triton 生成代码缺陷：`assert(algo_id < sizeof(kernels))` 用的是字节数不是元素
  个数；launcher 在 `gX*gY*gZ == 0` 的路径上没有 `return`。

定位：最初按要求放在 `08-Advance/`，发现 `05-Plugin/PythonPlugin/add_scalar_triton.py` 已经有一个
**JIT** 版的 Triton 插件之后，改放 `05-Plugin/`（同一主题的 JIT / AOT 两面，一个用于开发、一个用于
部署，README 互相链接）。tag 也从 `advance,plugin,compile` 改成 05-Plugin 惯例的 `plugin,compile`。

## `07-Tool/nvtriPy` + 两个 candidates md 关档（2026-08-28）

### 先说一个我自己造成的事故

为了看 Tripy，我直接 `pip install nvtripy` 装进了 cookbook 的环境，**把环境搞坏了**：

```
Successfully installed ... mlir-tensorrt-*-0.1.43+cuda12.trt109 numpy-1.26.0
                          nvtripy-0.1.7 tensorrt-cu12-10.16.1.11 ...
>>> tensorrt.__version__  →  '10.16.1.11'   （原本 11.1.0.106）
>>> numpy.__version__     →  '1.26.0'       （原本 2.1.0）
```

nvtripy 不是纯 Python 包，它依赖 `tensorrt-cu12 10.x` + `mlir-tensorrt ... cuda12.trt109`，装进来
就把系统的 TensorRT 11 顶掉了。已全部卸载回滚（`tensorrt` 恢复 11.1.0.106、`numpy` 恢复 2.1.0、
`colored` 恢复 2.3.2），并用 `05-Plugin/BasicExample` + `02-API/Layer/Cast` 跑通验证环境完好。
**教训：任何 `pip install` 之前先确认它会不会动 tensorrt/numpy，装第三方 TRT 前端一律进 venv。**

### 例子本身

这个坑反而成了例子的主线。`07-Tool/nvtriPy/` 由 `main.py`（驱动，建私有 `.venv` 并在里边跑）+
`tripy_cases.py`（真正的 Tripy 代码，只能在 venv 里跑）组成，开头就打印两边的 TRT 版本证明隔离：
cookbook 解释器 TensorRT 11.1.0.106，`.venv` 里 TensorRT 10.16.1.11 + nvtripy 0.1.7。

四个 case（对应原 md 的 #1/#7、#3、#2）：

1. **eager → compile**：同一个 `tp.Module` 先即时跑再 `tp.compile`。**发现两者并不一致**：
   相对差 9.8e-05。定位后是 **matmul**：纯 elementwise 图（gelu + 乘法）逐位相同，单个 `tp.Linear`
   则不同，而且**偏的是 eager 那边**（精确值 0.28，eager 给 0.28000974655151367，compiled 给
   0.2800000011920929，TF32 量级）。所以 eager 适合查形状和逻辑，不适合验证部署前的最后几位数。
2. **惰性求值**：定义张量 2.7 ms，第一次 `.eval()` 几百 ms（编译发生在这里），第二次 0.017 ms。
   直接给定义计时会严重低估。同源的表现：`Executable` **拒绝**未求值的输入而不是替你求值。
3. **动态形状**：`InputInfo(shape=((1,4,8), 4))` 就是 min/opt/max，一个 Executable 服务 batch 1/4/8；
   batch 9 报错，而且底下透出来的是 TensorRT 自己的 `satisfyProfile` 消息（`Valid range for
   profile 0: [1,4]..[8,4]`）——出范围是报错，不是偷偷重建。
4. **save/load Executable**：编译 176.5 ms vs 加载 2.4 ms（54 KiB），**74x**，输出一致。

原目录其实已有一个 stub（`enabled: false` + `.skip_unit_test`，而且它 README 里的安装命令正是会
炸环境的那条），已整体替换并删掉 `.skip_unit_test`，现在 runner 能发现并通过（tags `tool,slow`）。
#4~#10（ResNet50 / NanoGPT / SD / SAM2 / ModelOpt 量化）没做，理由写进了 README：都要 gated 或
数 GB 的 HF 下载 + 在 venv 里再装 torch/transformers，而 API 还没到 1.0。

### 两个 md 的处置

+ `candidates-eco-tripy.md`：自包含的几条已全部落地，**删除**。
+ `candidates-eco-torch-tensorrt.md`：**删除**，但里边还开着 5 条（#5 Refit 标着 HIGH 且确实没做、
  #7 FP8 ViT、#8 VGG16 PTQ、#14 weight streaming、#15 多卡），已整表迁进 `99-Todo/README.md`
  的「Still open from the ecosystem repos」一节，没有丢。顺带纠正：原 md 里 #5 底下那段
  "**As implemented**: 4 cases..." 讲的其实是 CUDA graph，是 **#6 的笔记错位**贴到了 #5 下面。
+ **#10 multi-profile 做不了**（本轮唯一没完成的委托）：装的 `torch_tensorrt` 2.14.0a0 完全没有这套
  API —— `Input(profiles=...)` 抛 `ValueError`、`runtime.optimization_profile` 不存在、
  `torch.classes.tensorrt.Engine` 上没有任何 profile 方法、`_TRTInterpreter.py` 里写死单个
  `create_optimization_profile()`。源码 clone（同为 2.14.0a0 但更新的 commit）有这功能，但
  `set_active_profile` 在 `core/runtime/TRTEngine.cpp` 里，光覆盖 py 文件没用，得整套源码重建。
  已记进 README 待办。TRT API 层的等价物 `08-Advance/MultiOptimizationProfile` 本来就有。

## `99-Todo/` 合并 + ChatGLM-6B 流水线取经（2026-08-28）

### 合并

8 个 md（README + 5 个 candidates + trex 2 个）**合成一个 `README.md`**，1288 行压到约 250 行。
去重方式：原 README 里的「Top candidates」两张表其实是 `candidates-github.md` /
`candidates-gitlab.md` 摘要表的副本，只留一份并把详情合进行内；已完成项的叙述（trex 全部 16 项、
链接体检表、已落地的生态例子）压成一句话。结构改成便于「晒需」：

+ **§1 = 挑选清单**，每条一行、带稳定编号（S1-6 常驻 / G1-12 OSS / L1-11 GitLab / M1-12 ModelOpt /
  R1-3 RTX + T1-4 tensorrtx / P1-6 Torch-TRT+Tripy 遗留 / 社区仓库），划掉不要的即可。
+ §2 = **已决定不做**（DLA、demoDiffusion、TRT_* 算子、sampleDevice RAII、TRT-LLM 关档、两个假缺口），
  防止重新提案。
+ §3 = GitLab **EXCLUDE 名单**（客户名目录、泄露内网 IP/客户模型路径的工具、Myelin/fusion 内部测试等），
  这份必须原样保留。
+ §4 = 参考（生态仓库、11.0→11.2 头文件 530 vs 530 空 diff、TREx 迁移始末）。

### ChatGLM-6B（`/work/chatglm-6b`，TRT 8.6 时代的手写 LLM 流水线）

读完写进 `99-Todo/chatglm-6b.md`，并在 README 里立了 **S6**。这套东西的价值在于**它把 TRT-LLM
后来产品化的那些招数摊开在 850 行里**，而且多数招数跟 LLM 无关：

值得搬的（C1-C12，前四条最硬）：

1. **把整个 KV cache 打成一个 I/O 张量**——28 层 ×(K,V) = 56 进 56 出，用一个 `Split(num_outputs=56)`
   和一个 `Concat` 变成 1 进 1 出，115 个 I/O 张量降到 6 个。
2. **KV 的输入和输出绑同一个地址**，cache 原地增长；`backupFile/main-twoBuffer.py` 保留了更早的
   乒乓版本，正好做 before/after。
3. **把采样搬进 engine**：surgery 阶段接上 Slice(最后一个位置)→MatMul(lm_head 常量)→Softmax→ArgMax，
   engine 直接返回 token id。词表 130528，等于每个 token 少搬 26 万个数。
4. **靠改模型自己的 `forward` 来导出**：`exportONNX.py` 只有 14 行且不含 export 调用，真正的
   `torch.onnx.export` 被塞进 patch 过的 `modeling_chatglm.py`，并且**卡在第二次调用（`past_key_values`
   非空）才触发**——导出的是带真实 cache 的 decode 图，不是拼出来的假图。
5. `markGraphOutput`：把任意节点的输出/输入标成图输出并截断图，用来二分精度 bug。
6. 旋转位置编码的 cos/sin 表在 surgery 时算成常量；改 `Transpose.perm` 原地重排 attention 布局；
   把第二个 ONNX（lm_head）的权重提出来当 Constant 并进主图；
   `gs.Constant` 必须 `np.ascontiguousarray`，否则 **TRT 把形状读成 `(0)`**（作者原注三个感叹号）。

不要搬的：**它的 build 配方在 TRT 11 下基本是非法的**——`EXPLICIT_BATCH` flag、`BuilderFlag.FP16`
+ `OBEY_PRECISION_CONSTRAINTS`、逐层 `set_output_type`、`FASTER_DYNAMIC_SHAPES_0805` 全部已移除，
build 那一段是重写不是移植（反过来说，它是 strong-typing 迁移的好素材）；
`builder_optimization_level=5` 绕开 myelin 是时代特定的 workaround，别照抄结论；
ChatGLM 特有的 token id / 两行 position_ids / 中文标点正则 / `.i(2).i().i().i().i()` 链也别搬。

建议形态是**两块而不是一个大例子**：`07-Tool/OnnxGraphSurgeon` 补工具箱（C5/C6/C8/C9），
外加一个新的 `03-Workflow` 叶子——用**玩具级 2 层 decoder** 把 C1/C2/C3/C4/C10 串起来。
**cookbook 目前完全没有自回归工作流**，这是真缺口。原 6B 模型不搬，只作为全尺寸参照引用。

## `03-Workflow/pyTorch-KVCache-ONNX-TensorRT`（2026-08-28）

把 ChatGLM-6B 那套东西用 **gpt2-small** 重写成 cookbook 例子。选 small 不选 medium 的依据是实测对比
（词表都是 50257，架构、cache 布局、导出路径全同；差别只是 51→5 与 99→5 的标题数字，以及下载/体积
2.5 倍），已记在 `99-Todo/chatglm-6b.md`。

五个 case，全部实跑：

1. **I/O 爆炸**：直接读 `00-Data` 里现成的 `model-large.onnx`（就是 gpt2-medium 的 prefill 图），
   2 进 **49 出**；对应的 decode 图是 **99 个 I/O 张量**——每个 token 都要 set 一遍。零下载。
2. **打包导出**：wrapper `Module` 把整个 cache 收成一个张量，导出后 **5 个 I/O**（2538 节点）。
   在 `L_past=3` 上 trace，在 1/7/16 上与 eager PyTorch **token 完全一致**——`transformers` 刷的那堆
   `TracerWarning` 看着像把图冻死在 trace 长度上，实测没有（但 `is_causal` 确实被烤进去了，所以这是
   decode 图，prefill 要单独导）。engine 476 MiB，**build 11.6 s**。
3. **原地 cache**：`past_kv` 和 `present_kv` 绑**同一块 18 MiB 显存**，跑完 17 步，
   输出 "TensorRT is a new type of neural network that can be used to train a neural"。
4. **布局决定能不能 alias**（这条是新发现，比原版更有价值）：
   `[2N,B,H,L,D]`（transformers 的存法）前 92160 个值 **≠** 输入；`[L,2N,B,H,D]`（本例导出的）**=**。
   序列轴不在最外层时新数据是插进去的，共享 buffer **静默算错**。ChatGLM 的 cache 恰好是 `[L,B,32,128]`，
   所以它那个"原地增长"不是巧思，是布局的必然结果。
5. **采样进图**：`next_token (1,1) int32` 0.782 ms/token vs `logit (1,50257) fp32` 0.866 ms/token
   ——每 token 少搬 **196 KiB**，整步 **1.11x**（复跑 1.09x）。两个 engine 由同一个 wrapper 的
   `b_return_logit` 开关产出，是同类对比。

顺带修的两处：

+ `requirements.txt` 里的 **`tripy` 删掉**——包名就是错的（真名 `nvtripy`），而且真装进来会把
  TensorRT 11 换成 10.16。已在文件里写明原因防止再加回来。新增 `transformers`（`--dry-run` 查过，
  只多拉 typer/shellingham/annotated-doc，不碰 numpy 和 tensorrt）。
+ `.pre-commit-config.yaml` 的 codespell `--skip=".git,3rdparty,*.ipynb,*.txt"` **引号是字面量的一部分**，
  导致 `*.txt` 这条 skip 从来没生效过（`requirements.txt` 里的 `lief` 被判成拼写错误才暴露出来）。
  去掉引号。

runner 通过（clean 状态 61 s，HF 权重已缓存的前提下）。生成物 ~2.2 GB（两个 ONNX + 两个 engine）
全部 gitignore 且 clean 清掉。

## `07-Tool/OnnxGraphSurgeon` 补 ChatGLM 工具箱（S6a，2026-08-28）

先纠正我上一轮的判断：**S6a 比我说的小**——`mark_graph_output`（C5 的那个 helper）和 `add_node`
（C-addNode）**早就已经移植进 `tensorrt_cookbook/utils_onnx.py` 并导出了**，`np.ascontiguousarray`
的坑也已经写在 `10-advanceAPI.py` 的注释里。真正缺的是：`mark_graph_output` **在整个 cookbook 里
零使用**，没人演示那个「二分精度 bug」的工作流；以及 C6 常量表、C8 合并两个 ONNX。

新增三个文件（`01`~`10` 是 API 巡礼，`11`~`13` 是真模型上实际会用到的三件事）：

+ **`11-mark_output_to_bisect.py`**：把「FP16 结果不对」变成一个节点名。构造
  `((x*300)*300)/90000 == x`，FP16 下 90000 和中间值都溢出成 `inf`。**关键观察：最终输出既不是 inf
  也不是 nan，而是静默变成 0**（ORT 的 fp16 Div 行为），所以没有任何告警。逐节点截图后
  `node_scale_up_1` 第一个不一致（FP32 90000 vs FP16 inf）——错的是它，不是产生错误最终值的那个节点。
+ **`12-constant_table.py`**：常量折叠管不到「依赖运行时张量、但取值范围有界」的子图。旋转位置编码是
  典型：`cos(position*inv_freq)` 在主机上按所有可达 position 算一遍就变成一个 `Gather`，4 节点 → 1，
  误差 5.96e-08。**代价写清楚了**：表的大小从此是硬上限，原图没有这个限制。
+ **`13-merge_two_models.py`**：把单独导出的 head 接到 body 上，两种做法都演示（把权重当
  `gs.Constant` 提出来 / 整批搬节点），再接一个 `ArgMax` 尾巴，让图直接返回类别号而不是分数向量。

顺带修了个**移植 bug**：`mark_graph_output` 的 `b_mark_input=True` 分支对每个输入无条件写
`.dtype`，碰到 `gs.Constant` 会抛 `property 'dtype' of 'Constant' object has no setter`。
ChatGLM 原版在输入分支**没有**这行，是移植时加进去的。已加 `isinstance(..., gs.Variable)` 判断。
（这条路径此前全 cookbook 无人调用，所以一直没暴露。）

runner 通过（13.0 s）。

## 进度流水

- 2026-08-27：完成 Polygraphy 调研，产出 A/B/C 三类共 19 个候选；建 `More/` 与本文件。
- 2026-08-27：**A1 完成**。
- 2026-08-27：**A2 完成**（合并了原 A3）。
- 2026-08-27：**A4 完成**。
- 2026-08-27：**A5 完成**。
- 2026-08-27：**A7 完成**。
- 2026-08-27：**A8 完成**。
- 2026-08-27：**A9 完成**。
- 2026-08-27：**A10 完成，A 组全部结束**。
- 2026-08-27：**A6 完成**（发现 INT8 校准 API 在 TRT 11 已整体移除，改写为迁移指南）。
- 2026-08-27：**B1 完成**（发现 initializer=0 但权重存在 attribute 里，导出比源模型还大；控制流网络 Polygraphy 导不出、cookbook 的能）。
- 2026-08-27：**B2 完成**（tactic replay API 在 TRT 11 整体移除；实测 `save_timing_cache=` 只写不读、加速 1.00x）。
- 2026-08-27：**B3 完成**（弱类型网络已消失、`strongly_typed=False` 被静默忽略；四个 loader 三种结局）。
- 2026-08-27：**B4 完成**（`set_tensor_debug_state(True)` 是空操作；标记本身就有 1.56x 代价）。
- 2026-08-27：**B5 完成**（零拷贝在 25 MB 上 32.1x、MNIST 上只有 1.46x；view 悬空后 free 路径静默读错数据、resize 路径 SIGSEGV；`stream=` 在 pageable 内存上不异步，掩盖了漏掉的 synchronize）。
- 2026-08-27：容器重启导致 `onnxruntime`/`onnx_graphsurgeon`/`onnxslim`/`tensorrt_cookbook` 全丢，已重装（见「环境」一节）。
- 2026-08-27：**B6 完成**（`LoadPlugins`/CLI `--plugins` 走 ctypes，注册不了 V3 插件 creator，必须用 `registry.load_library()`；`register` 未导出；registry 只有 3 个 op、跑整张图）。
- 2026-08-27：**B7 完成**（VC 把 105 MB 的 lean runtime 塞进每个 plan，一层网络也是 9221x；代价在反序列化 34.4x、推理为 0；裸 runtime 反序列化 VC engine 返回 None）。
- 2026-08-27：**C1 完成**（订阅 arg group 白拿 95 个 CLI 选项；依赖只写在 docstring 里，漏订阅是运行期裸 KeyError；不能给 polygraphy 加子命令）。
- 2026-08-27：**C2 完成，A/B/C 全部结束**（entry point 必须真安装，PYTHONPATH 不够；import 即注册补上了 B6 缺的那条路；float64 参考揭出两个后端一致但都错 100%）。
- 2026-08-27：**CLI 侧复查完成**（10 个目录逐个实跑；修掉 Inspect 的假 FP16 步骤、Convert 的孤儿 data_loader、Plugin-TODO 的空结果与 config.yaml 污染、误提交的 p3~p7.py；补上 weight-strip/reconstruct、--save-visual、debug build/repeat、data concat、--data-loader-script、--convert-to onnx；multi-device 只出方案，单卡环境无法验证）。
- 2026-08-27：**`MultiDevice/` 建好**（CP 切序列插 6 个 collective、TP 切权重每 rank 一个文件；模型里没有 attention/SwiGLU 时整条流程静默变成复制一份；rank 数来自 `--nb-rank` 不是 `--gpus`。ONNX 改写部分单卡跑通并进 CI，多卡实跑留 case 06 待办）。
- 2026-08-27：**`07-Tool/trtexec` 复查完成**（发现步骤 11 因 `--refPair` 只给一对而整段中断、步骤 12 的 tuner knob 名根本不存在，两处都改对并跑通；补 `--stronglyTyped`（TRT 11 上是空操作）、多流吞吐、权重剥离 84x 且 CLI refit 静默失效；trtexec 的 `--fp16` 等选项是直接 Unknown option，与 polygraphy 的「help 里还在、build 才炸」形成对照）。
- 2026-08-27：**`07-Tool/TritonServerDeploy` 建好**（空壳 → 四阶段部署范例；本容器无 tritonserver 亦无 docker，阶段 3 跳过，但用 stub server 把客户端半边测了 16 项；记下 config.pbtxt 的 batch 维/标量 reshape/profile 三个坑）。
- 2026-08-27：**tritonserver 免 docker 跑通**（从 nvcr.io 匿名扒镜像层，只取 53 MB + 两个 .deb 里的 so；26.07-py3 的 TRT 恰为 11.1.0.106 与本机一致；TritonServerDeploy 四阶段全部实跑，输出与本地参考逐位相同）。
- 2026-08-27：**`08-Advance/EmptyTensor` 建好**（检测器无框 / 空 batch 的真实场景；发现并修掉 `utils_class.py` 四处 `cudaMalloc(0)` 返回 NULL 导致 `enqueueV3` 静默拒跑的共享代码 bug —— 这个 bug 一开始让例子自己得出了「MAX over nothing = 0」的错误结论，实为 `-inf`/`NaN`）。
- 2026-08-27：**`08-Advance/MIG-TODO` → `MIG`，决定只留 README 不写例子**（MIG 是宿主机配置、TRT 侧无差异；唯一值得记的是「engine 必须在部署用的 profile 上 build」，本机 114 SM/79 GiB vs 14 SM/9.75 GiB；对应测量因本机未开 MIG 列为待办）。
- 2026-08-27：**`08-Advance/GreenContext` 建好**（MIG 的进程内替代：切 SM 分区、免 root 免重启；吵闹邻居 p95 8.23x → 3.05x；**发现 TRT 的 aux stream 会逃出分区且默认就会发生**；**在分区内 build 白捡 19%**，用对照组推翻了我最初「无差异」的错误结论）。
- 2026-08-27：**`08-Advance/TensorRTGraphSurgeon` 建好**（INetwork 层面动刀：走图/加层/靠改线删层/换 plugin；换 plugin 输出逐位相同但**丢了融合、代价 2.13x**；发现「插件库已加载 + 释放过 engine → 查注册表必崩」的触发条件）。
- 2026-08-27：**FP16 ONNX 用量统计完成**（00-Data 里没有任何 FP16 ONNX；5 个 03-Workflow 例子用着已删除的 `BuilderFlag.FP16`、trex 的 model.fp16 其实是 FP32；建议默认用 ModelOpt AutoCast 生成，另备一个 torch.half() 的纯 FP16 版）。
- 2026-08-27：**`90-Misc/Number` 数据类型表复查完成**（对照 torch/ml_dtypes 逐 bit 实测：E8M0 的 `0x80` 是 2.0 不是 1.0、`0xFF` 是 NaN 不是最大值；FP4E2M1 的「最大的小于 1 的数」印了个它根本没有的 0.75；`Integer.md` 不是生成的且 UINT8/INT8 填的是 16 位范围；torch 目前没有比表里更新的浮点类型，补上了 FP6/MXFP6，删掉了并不存在的 NVFP8 与 FP7E4M3）。
- 2026-08-28：**TensorRT-LLM 候选清单关档**（按「只要跟 TensorRT 相关的工具」的标准，16 条候选全废；实测该仓非 3rdparty 里只剩 9 个文件 `import tensorrt`，建 engine 那层已整体删除，三个插件例子全部 import 不起来；纯 Python 写 TRT 插件的例子改从 TensorRT OSS 拿）。
- 2026-08-28：**`05-Plugin/TritonAOTPlugin` 建好**（从零重写 TRT-LLM 那两个已失效的 Triton 插件例子：AOT 内嵌 cubin 的 C++ 插件 + 从 spec 生成插件；**实测出 Triton AOT 把 `fp32` 标量参数声明成 `double` 导致静默算成 `x+0` 且 launch 仍返回成功**；多个 AOT variant 映射成 TRT tactic，builder 选出的比固定第一个快 1.83x；另记录「插件库不导出 `setLoggerFinder` 就静默注册失败」）。
- 2026-08-28：**`07-Tool/nvtriPy` 建好，Tripy / Torch-TensorRT 两个 candidates md 关档**（先踩了个大坑：`pip install nvtripy` 把 cookbook 环境的 TensorRT 11 换成了 10.16、numpy 降到 1.26，已回滚验证；例子改成在私有 venv 里跑，并把这件事写成主线。发现 **Tripy 的 eager 与 compiled 在 matmul 上不一致且偏的是 eager**；Torch-TRT 的 multi-profile 因装的版本根本没有该 API 而未完成，连同另外 4 条未做项迁进 `99-Todo/README.md`）。
- 2026-08-28：**`99-Todo/` 8 个 md 合并成一个 README（1288 → ~250 行）**，按 §1 挑选清单 / §2 不做 / §3 EXCLUDE / §4 参考 重组，每条候选给了稳定编号便于晒需；**读完 `/work/chatglm-6b` 写出 `99-Todo/chatglm-6b.md`**（TRT 8.6 时代手写 LLM 流水线：KV cache 打包成单张量、输入输出同地址原地增长、采样搬进 engine、靠 patch `forward` 在第二次调用时导出 decode 图；其 build 配方在 TRT 11 下已非法）。
- 2026-08-28：**`03-Workflow/pyTorch-KVCache-ONNX-TensorRT` 建好**（用 gpt2-small 重写 ChatGLM-6B 的那套招数：KV cache 打包成单张量 99→5、输入输出绑同一块显存原地增长、采样进图省 196 KiB/token、导出图与 PyTorch token 逐个一致；**新发现：能不能 alias 完全由 cache 布局决定**，transformers 的 `[2N,B,H,L,D]` 共享 buffer 会静默算错。顺带删掉 requirements 里危险且名字就错的 `tripy`，修好 codespell 那条因引号而从未生效的 `--skip`）。
- 2026-08-28：**S6a 完成**（`07-Tool/OnnxGraphSurgeon` 新增 `11`~`13`：二分精度 bug / 主机端常量表 / 合并两个 ONNX + ArgMax 尾巴。先纠正判断：`mark_graph_output` 与 `add_node` 早已移植进 utils，只是**零使用**，缺的是演示；顺带修好 `mark_graph_output` 的 `b_mark_input` 对 `gs.Constant` 写 dtype 崩溃的移植 bug）。
