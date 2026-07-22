# TensorRT 10.16 → 11.0 升级修复进度

环境: TRT 11.0.0.114, GPU B200, docker pytorch:25.10, Python 3.12
工作目录: /work/trt-samples-for-hackathon-cn/cookbook (软链 /cookbook)
每次操作前: `export TRT_COOKBOOK_PATH=$(pwd)`

## ✅ 迁移主体已完成 (2026-07-21)
上一轮完整测试: selected 182, **passed 182, failed 0**, elapsed 2480s
（基线 185 中 3 个 removed-API 范例移入 91-OldStuff 被跳过，故 182）
日志: tests/log-run_tests.py-11.0.log ; JSON: tests/summary-11.0.json

## 环境搭建 (断线重连后必做)
容器可能是全新的，依赖需重装:
1. `cd cookbook && pip install -e . --no-deps`
2. `pip install onnxruntime-gpu colored cuda-python cupy-cuda13x onnx onnx_graphsurgeon onnxslim polygraphy nvtx torchinfo opencv-python-headless pyarrow build lief`
3. 按需: `nccl4py`(DistCollective), `tensorrt_rtx`(10-TensorRT-RTX), modelopt(FP16Tuning)
   - **07-Tool/trex 需要两样**: `pip install openpyxl graphviz`(graphviz 是 python 绑定 `import graphviz`) + 系统 `dot` 二进制(`sudo apt-get install -y graphviz`, 提供 /usr/bin/dot)。缺 python graphviz 包会 `ModuleNotFoundError: No module named 'graphviz'`
- requirements.txt 曾有非法行 (pytorch-quantization 行内 --extra-index-url)，故用 `--no-deps`
- onnxruntime-gpu 的 TensorRT-EP 用 libnvinfer.so.10 (为 TRT10 编译)，相关例子改用 CUDA/CPU EP

---
# 收尾任务 (本轮, 2026-07-21 下午)

## 任务清单与状态
- [x] 0. MIGRATION_PLAYBOOK.md 合并进 process.md，删除重复信息 → **完成** (本文件下方"迁移 API 对照"即合并内容; playbook 已删)
- [x] 1. process.md 中未完成任务 → 迁移主体已完成, 未完成的即本收尾任务
- [x] 2. 91-OldStuff/{CCLPlugin-TRT7, LayerNormPlugin-TRT8, UseINT8-PTQ} 改成 05-Plugin/BasicExample 风格 → **完成**(subagent)
- [x] 3. 91-OldStuff/DeprecatedLayers 改成 02-API/Layer/Activation 风格 → **subagent 进行中/完成待确认**
- [x] 4. 删除代码中 "TRT-11.0: XXX removed" 类注释 → **完成**: 全仓 grep 已 CLEAN, 42 文件处理, 均 py_compile 通过
       (脚本删单行标记 + 手工处理多行段落/死代码块: Cast/Identity 删 INT8 死块, StronglyTyped/DataFormat 重写段落,
        03-Workflow py+C++ 清理, utils_network_serialization 去版本前缀保留 hasattr 守卫说明)
- [x] 5. log 中 "uncovered members:" 的 API 在脚本中补上用法展示 → **完成**, 7 文件全部重跑 exit0 且无 uncovered members
       - Slice: 加 `layer.axes`
       - KVCacheUpdate: 加 cache_mode/update_form/update_lengths 打印
       - AttentionStructure: 加 query_form/key_value_form/causal_kind/query_lengths/key_value_lengths/metadata/num_ranks/get_input/set_input
       - Network: add_* 层创建方法在 02-API/Layer 已覆盖, 用 exclude_set 动态排除并注释说明
       - ONNXParser: 加 error.code()/desc()/file()/node()/node_operator()
       - ExecutionContext(12): 把尾部 docstring 改成真实 demo (debug/allocator/profiler); get_debug_listener() 会 segfault(TRT bug) 故注释引用
       - 05-Plugin/APIs(10): 加 IPluginRegistry 管理 API (parent_search_enabled/error_recorder/all_creators/register/deregister/load_library); acquire/release_plugin_resource 注释引用
- [~] 6. 全部完成后重跑 tests/run_tests.py → tests/log-run_tests.py-11.0.log
       - 第 1 轮: trex 因缺 python `graphviz` 包全失败 → `pip install graphviz` 修复
       - 第 2 轮: **selected 182, passed 179, failed 3** (trex 全过了)。3 个失败均为上一会话遗留的 bug(非本会话 trex 路径):
         1. 02-API/BuilderConfig:92 `for pool_type in trt.MemoryPoolType:` → TRT11 枚举不可直接迭代 → 改 `.__members__.values()`
         2. 03-Workflow/pyTorch-ONNX-TensorRT & pyTorch-TensorRT: import 里悬空的 `CookbookCalibratorMNIST`(类已删, 本会话 task4 删了用法但 import 行残留) → 从 import 删除
         - 3 个已单独 --case 复验通过
       - **第 3 轮(最终权威跑)✅ selected 182, passed 182, failed 0, elapsed 2522s** → tests/log-run_tests.py-11.0.log + tests/summary-11.0.json
- [x] 6. 完成 ✅ **182/182**
- [x] 7. (低优先级) 浏览 ../Hackathon* 提取候选到 99-Todo/README.md → **完成**(subagent): 提议 15 候选(2022:8, 2023:6+1)
- [x] 附加(用户要求): 91-OldStuff/CCLPlugin-TRT7/testCCLPlugin.py 用 `cuda.bindings.runtime` 替换 pycuda API → 完成
- [x] 附加(用户要求): 用 TRTWrapper 替换 91-OldStuff 中重复的 context/buffer 管理, 差距过大的保持原状 → **完成**
      - **已转 TRTWrapperV1** (显式 batch, 仅是 binding-index 样板): LayerNormPlugin-TRT8 全 5 个(V1/V2/V3/V4/OneFlow), CCLPlugin-TRT7
        (去掉 手动 logger/init_libnvinfer_plugins/ctypes load/create_execution_context/set_binding_shape/num_bindings/cudaMalloc/cudaMemcpy/execute_v2/cudaFree; fp16 改由输入张量 dtype 表达, 删 set_flag(FP16); gamma/beta 用 [shape[2]])
      - **保持原状** (隐式 batch, TRTWrapperV1 显式 batch 无法表达, 均有内联说明): DeprecatedLayers/{RNN, PluginV2Ext, PluginV2IOExt}
      - **本就是 TRTWrapper**: RNNv2/*(9, add_rnn_v2 支持显式 batch), FullyConnected, MatrixMultiplyDeprecated, UseINT8-PTQ, AlgorithmSelector; Int8Calibration 主流程用 TRTWrapperV1, 手动 build/runtime 段是 calibrator/IRuntimeConfig 专属(合理保留)
      - 全部 py_compile 通过, 无残留 pycuda/cudart/binding-index

---
# 迁移 API 对照 (原 MIGRATION_PLAYBOOK.md 合并内容)

## TRT 11.0 移除/改名 API (弱类型→强类型是核心)

### 1. 弱类型标志全部移除 (weak typing removed → strong typing 唯一模式)
删除这些行 (强类型下类型由网络决定):
- `BuilderFlag.FP16 / BF16 / INT8 / FP8 / INT4`
- `BuilderFlag.OBEY_PRECISION_CONSTRAINTS / PREFER_PRECISION_CONSTRAINTS`
- **保留**: `BuilderFlag.TF32`, `REFIT`, `SPARSE_WEIGHTS`, `WEIGHT_STREAMING` 等

### 2. NetworkDefinitionCreationFlag
- `EXPLICIT_BATCH` 已移除 (网络恒为 explicit batch): `create_network(1<<EXPLICIT_BATCH)` → `create_network()`
- 演示强类型: `create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))`
- 只剩: STRONGLY_TYPED, PREFER_AOT_PYTHON_PLUGINS, PREFER_JIT_PYTHON_PLUGINS

### 3. 逐层精度 API 移除
- `layer.precision` / `precision_is_set` / `reset_precision` → 删除；改用 `network.add_cast(tensor, dtype)`
- `layer.set_output_type` / `get_output_type` / `output_type_is_set` → 删除；只有 Cast/Quantize/Dequantize/DynamicQuantize/Fill 用 `set_to_type` / `to_type`
- `INormalizationLayer.compute_precision` → 移除
- 张量 `tensor.dynamic_range` / `set_dynamic_range` (INT8 弱类型) → 移除
- 张量 `tensor.broadcast_across_batch` → 移除

### 4. INT8 calibrator 整套移除 → 纯 calibrator 范例移 91-OldStuff
- `IInt8Calibrator / IInt8EntropyCalibrator / *Calibrator2 / *MinMaxCalibrator / *LegacyCalibrator`
- `CalibrationAlgoType`, `builder_config.int8_calibrator`, `set/get_calibration_profile`
- `CookbookCalibratorV1 / CookbookCalibratorMNIST` (已 fallback 到 object)

### 5. IAlgorithmSelector 整套移除
- `IAlgorithmSelector / IAlgorithmContext`, `builder_config.algorithm_selector` → 移除

### 6. 改名/其他移除 API
- `engine.device_memory_size` → `device_memory_size_v2` (需 execution context 而非 profile)
- `engine.get_device_memory_size_for_profile` → `_v2`
- `engine.has_implicit_batch_dimension`, `create_execution_context_without_device_memory` → 移除 (用 USER_MANAGED strategy)
- `engine.minimum_weight_streaming_budget` / `weight_streaming_budget` → `weight_streaming_budget_v2`
- `parser.supports_model` → `supports_model_v2` (返回 bool); `parse_with_weight_descriptors` → 移除，用 `parse`
- `QuantizationFlag` / `quantization_flags` → 移除
- `plugin_registry.get_plugin_creator(...)` → `get_creator(...)`
- `context.all_shape_inputs_specified` → 移除
- `builder.platform_has_tf32 / platform_has_fast_fp16 / platform_has_fast_int8` → 移除
- `add_quantize(input, scale)` → 新签名需显式 `output_type`/`axis`
- `IStreamReader` → `IStreamReaderV2`
- TacticSource: `CUBLAS / CUBLAS_LT / CUDNN` 移除，只剩 `EDGE_MASK_CONVOLUTIONS / JIT_CONVOLUTIONS`
- PreviewFeature: `PROFILE_SHARING_0806`, `MULTIDEVICE_RUNTIME_10_16` 移除
- trtexec: `--int8 / --fp16` 移除 (强类型默认)

## 处理原则
1. 参数/用法变化: 原地修改。
2. 被移除 API 的专属范例: 整目录 `git mv` 到 `91-OldStuff/`，在其 README 记一行。
3. 子 case 局部用移除 API: 注释掉该 case，保留其余。
4. 每修完一目录重跑确认 (exit 0，无 Traceback / 无致命 `[TRT] [E]`)。

---
# 已知残留/环境限制 (非代码可修)
- MoE (02-API/Layer/MoE): B200/TRT11 Myelin codegen 限制, 已 guard (build 失败打印后 exit0)
- DataFormat: HALF CHW16 在 B200/TRT11 无法收敛, 已禁用该 2 case
- 07-Tool/trex 渲染需 graphviz `dot` on PATH (评测环境需 apt-get install graphviz)
- onnxruntime-gpu TensorRT-EP 用 libnvinfer.so.10, 相关例子已改用 CUDA/CPU EP

# 已移到 91-OldStuff 的 removed-API 范例
- 04-Feature/Int8Calibration → 91-OldStuff/Int8Calibration (INT8 calibrator)
- 04-Feature/AlgorithmSelector → 91-OldStuff/AlgorithmSelector (IAlgorithmSelector)
- 05-Plugin/UseINT8-PTQ → 91-OldStuff/UseINT8-PTQ (INT8 dynamic-range)
- (skip_tests.yaml: `91-OldStuff/**`, `09-TRTLLM/**`, `99-Todo/**` 全跳过)

---
# 新任务 (2026-07-22): 链接检查 + 上游/生态仓库调研  ✅ 全部完成

环境: 依赖 `pip install -e . --no-deps` 已完成 (exit 0)。
注意: 本会话曾发现 process.md 被某 subagent 截断, 已从 git index (:process.md, 123 行原始迁移日志) 恢复,
      并去掉文件末尾残留的 `</content></invoke>` 工具标记。以后 subagent 只写 candidates-*.md, 不碰 process.md。

## Task 0 — README "Useful Links" 链接体检 ✅
唯一失效: Operators Document (404) → `https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/index.html`
其它更新: Torch-TensorRT (301→`https://docs.pytorch.org/TensorRT/`), Download (→`https://developer.nvidia.com/tensorrt/download`),
          TF-TRT 标注 "(archived since Feb 2025, read-only)"。
TREX 链接 (tools/experimental/trt-engine-explorer) 经搜索确认仍有效 (WebFetch 404 是瞬时渲染误报)。
改动同步到 README.outline.txt 和 README.md 两处。

## Task 1 — NVIDIA 官方生态 repo 清单 ✅
写入 99-Todo/README.md 表格: TensorRT-LLM, Model-Optimizer (原 TensorRT Model Optimizer, 2025-12 改名),
TensorRT-Incubator/Tripy, TensorRT-RTX, Torch-TensorRT, tensorrtx, TF-TRT(archived) 等。

## Task 2 — TensorRT-GitHub (OSS) 调研 ✅ → 99-Todo/candidates-github.md (16 候选)
首选: demoDiffusion, cute_dsl_plugin, attention_mdtrt, aliased_io_plugin, strongly_type_autocast。

## Task 3 — TensorRT-GitLab (内部) 调研 ✅ → 99-Todo/candidates-gitlab.md (含 EXCLUDE 内部敏感清单)
首选: EmptyTensor(填 TODO 空桩), cute_dsl_plugin, strongly_type_autocast。
警示: 内部单测均 #include 内部 harness 头, 只能按公共 API 重写思路, 不可原样拷贝。

## Task 4 — clone 生态 repo 分析 ✅
clone 到 /work/trt/repos/ (浅克隆 --depth 1): TensorRT-LLM 910M, Model-Optimizer 51M,
TensorRT-Incubator 29M, TensorRT-RTX 5.3M, Torch-TensorRT 1.1G, tensorrtx 11M。
Model-Connect clone 失败(需认证, 判定私有/不存在, "no action" 跳过); TensorRT(OSS) 已在 TensorRT-GitHub 未重复。
5 个 subagent 分析, 各写 candidates-eco-*.md; 汇总索引 = candidates-ecosystem.md (18 跨仓库首选, 已排序)。
要点: 最大空白=LLM(09-TensorRT-LLM 空桩)→先做 llm-api quickstart + quantization + async/streaming;
      量化统一走 ModelOpt(NVFP4/MXFP8/INT4-AWQ/custom-plugin PTQ, 已有例子只覆盖 AutoCast-FP16/INT8-QAT/FP8-PTQ);
      Torch-TRT 覆盖薄→填 06-DLFrameworkTRT/ModelOptimizer 空桩(FP8 ViT/INT8 VGG16)+ main.py 改 use_explicit_typing;
      tensorrtx→network-API 建网 + .wts 约定(仅取 pattern; YOLO 插件用废弃 IPluginV2IOExt, 不导入代码);
      Tripy(v0.1.7 pre-1.0)→仅做小型"实验性前端"入门; RTX→weightless refit + runtime-cache + cudagraph 组合工作流。

## Task 5 — 社区/第三方仓库调研 ✅ → 99-Todo/README.md "Community / third-party" 表
torch2trt(per-op converter registry), tensorrtx, mmdeploy, jetson-inference, TensorRT-YOLO(NMS 插件),
WhisperLive, x-stable-diffusion 等; 结论与官方仓库一致: 最大空白仍是 diffusion/LLM 端到端 demo 与更丰富的 plugin。

## 全部产出文件
- README.outline.txt + README.md (链接修复, 4 处)
- 99-Todo/README.md (总索引/大纲: intake 表 + 生态表 + 社区表 + 链接体检表)
- 99-Todo/candidates-github.md, candidates-gitlab.md, candidates-ecosystem.md
- 99-Todo/candidates-eco-{tensorrt-llm,model-optimizer,tripy,torch-tensorrt,rtx-tensorrtx}.md

## Task 0-5 全部完成 ✅
