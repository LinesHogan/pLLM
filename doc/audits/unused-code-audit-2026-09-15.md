# 未接入功能与迁移遗留审计

审计日期：2026-09-15。代码基线：`2ed84dc`，当前工作区。此次只新增本报告，未修改运行代码，也未改动工作区已有文档。

结论：最值得处理的是跨进程页授权未接入、mutation 粒度仅参与描述与签名，以及 In-Place 调优未跟随新的 native tuning 策略。另有专用改写入口、调试绑定入口和少量无调用辅助方法可以收敛。不能把所有“只有测试调用”的接口都判定为无用。

## 审计方法与结论边界

- 检查 `pllm/`、`tools/`、`examples/`、`test/` 的 Python 定义、导入、名称和属性引用；以 AST 扫描筛选，再人工追踪调用链。
- 补查 Rust FFI、Rust 内部消费、文档、字符串 RPC、装饰器/属性发现及近期提交。
- “无生产调用”表示仓库内正常库实现、工具、示例中未找到调用，不能证明仓库外用户从未使用这些低层接口。
- 声明类经 codec 序列化、在 Rust 使用，并不需要 Python 属性读取；这类情况已排除。
- 下文分别标注：未接入、行为缺口、可合并接口、无调用候选、测试基础设施。建议不等于已经验证可以无损删除。

## 1. PageBundle 的跨进程页授权：完整实现，只有测试发起

**分类：未接入；优先级高。**

入口位于 `pllm/execution/ptts/page_bundle.py`：

- `send_field`，812 行：将 pin 下的原字段授予远端读者。
- `send_write_field`，823 行：将私有写字段授予远端写者。
- `receive_field`，864 行：接收并映射原字段。

仓库内调用集中在 `test/boundary/test_shared_page.py` 和 `test/gpu/test_page_bundle_cuda.py`。正常 vLLM/host provider 未启用 `PageBundleProviderFacts.shared_memory`；该选项默认 False。

这不是几行闲置封装。它还引入 `PageAccess`、`PageFieldTransfer`、`BorrowedPage`、CPU shared pool、`_Write.remote_writes`、pin 上的远端读者集合，并进入 publish、abort 和 collect 的生命周期逻辑：

- publish 检查远端写完成：`page_bundle.py:473`。
- abort 等待远端写：`page_bundle.py:502`。
- collect 等待/查询远端读者：`page_bundle.py:977`。

真实 EngineCore 辅助输入/输出使用 `SharedTensor.export/allocate/map`：`pllm/backends/vllm/runtime.py:1148`、`1152`，`pllm/worker.py:65`。这与直接授权 arena 字段是不同层次的功能。

**建议：** 若近期没有远端直接训练/读取 arena 页的交付任务，优先将直接页授权隔离为实验扩展，收窄核心 arena 的远端状态。不要连带删除已用于 EngineCore 的 `SharedTensor`、CUDA IPC 底层工具或正常设备完成处理。这里未测量额外分支的性能成本。

## 2. mutation 粒度有三种选择，分配/复制实现没有消费这种选择

**分类：行为缺口；优先级高。**

`pllm/contracts/tensor_layout.py:26` 定义 `WHOLE_TENSOR / ROW / PAGE`，用户创作层可通过 `InstanceState.mutation` 传入；ESamp 也显式声明 PAGE（`pllm/workloads/esamp_request.py:94`）。

字段在 Python/Rust 两端被编码，参与布局身份和摘要。但 `page_bundle_planning.py:52` 的 stride 计算与 `page_bundle.py:388` 的私有分配、`445` 行起的复制逻辑均不依据 mutation 粒度分支。实际粒度是选中的整个字段；ROW 没有让运行时支持只复制某几行。

CPU 复现使用同一 `(2, 3)` 字段，分别修改三种粒度并重新合法编译整份计划：

| 粒度 | arena 预留字节 | 一次字段更新新增载荷 | 新增私有字段页 |
| --- | ---: | ---: | ---: |
| whole_tensor | 468 | 24 | 1 |
| row | 468 | 24 | 1 |
| page | 468 | 24 | 1 |

三份布局摘要均不同。这证明该例中身份发生变化、执行分配不变；源码消费检查进一步表明当前分配路径没有读取此字段。

**建议：** 将现有字段级 COW 语义说清楚，删除或拒绝尚无执行语义的粒度选项，或者明确标成描述性元数据。删除字段涉及跨语言 schema/布局身份迁移。创作层对 overlay buffer 仍有 WHOLE_TENSOR 限制，因此不能声称此字段完全没有校验效果。

## 3. In-Place 调优没有接入新的 heuristic 开关

**分类：迁移后的策略不一致；优先级高。**

新的 native 路径在 `pllm/backends/vllm/runtime.py:815` 遇到 `tuning="heuristic"` 时直接使用参考计划。配置注释将这项约定限定为 native selection（`pllm/config.py:29`）。

但 In-Place 的调用链仍为：

```text
InPlaceTTTWorkload.operations
  -> _tune_chunk_scan
  -> TensorScan.optimized
  -> optimize_region
  -> warmup + measurement
```

对应位置：`pllm/workloads/inplace_ttt.py:88`、`166`、`pllm/execution/tensor_scan.py:125`、`pllm/execution/region_optimization.py:501`。

`RegionOptimizationOptions` 只声明 `allow_numerical_transforms` 和 `schedule_policy`，没有 `tuning`（`region_optimization.py:52`）。因此传入的 `RegionSelectionPolicy.tuning` 不控制这条测量路径。

复现传入 `RegionSelectionPolicy(False, "serial", "heuristic")`，使用原 `_chunk_region/_ChunkTransition/_tune_chunk_scan`：测量边界进入 **16 次**，返回对象仍有 measured selection。复现将 `_measure` 替换为计数函数，所以此数字用于证明调用路径，不是耗时或性能结果。另有真实生产绑定测试通过并确认两种布局都会生成选择结果。

**建议：** 明确是否要让 tuning 覆盖全部 workload；若是，在 In-Place 冷绑定处增加统一策略消费；若只控制 native replay，就把公开名字和说明限定到该范围。不能仅凭此审计删掉 In-Place 的实际优化能力。

## 4. 矩阵乘专用搜索接口与通用改写接口并存

**分类：可合并接口；优先级中。**

`pllm/execution/tensor_rewrites.py` 仍有：

- `matmul_reassociation_candidates`（102 行）；
- `replay_matmul_reassociation`（118 行）；
- `replay_matmul_reassociations`（127 行）；
- `explore_matmul_reassociations`（160 行）；
- `explore_matmul_combinations`（172 行）。

前四个公开入口加 combinations 的直接外部调用均在边界测试中；其中复数 replay 还被单步 replay 内部调用。生产优化用 `explore_tensor_combinations`、`replay_tensor_rewrites`，将矩阵重结合与 batching 放在同一个规则集合中。

这些入口共用 `_steps/_explore/_replay`，所以不能把整份 231 行模块算成废代码。真正多出来的是专用 API、相应测试入口以及读者需要区分的两套操作方式。

**建议：** 生产只保留通用接口；测试若需隔离矩阵规则，使用明确的测试辅助入口或规则集合参数。保留矩阵重结合规则及独立数值测试，不要把算法功能一起删除。

## 5. 多种重建/绑定捷径只供测试或调试

**分类：可合并接口；优先级中。**

| 入口 | 仓库内调用方 | 当前主路径 |
| --- | --- | --- |
| `PageBundleArena.reconstruct`，`page_bundle.py:220` | `test_worker_reconstruction.py` | worker 重建 workload，`BackendWorkload.bind_backend` 校验并创建 arena |
| `RegionStreamExecution`，`region_execution.py:202` | 四份 stream/borrow/kernel 边界测试 | 通用 `RegionInstructionExecution` 接受 stream plan |
| `bind_outlined_region`，`region_outlining.py:87` | `test_region_outlining.py` | `_replay_outlines` 从原程序调用 `outline_region_calls` 并检查证据 |
| `bind_selected_schedule`，`region_optimization.py:151` | `test_region_borrows.py` | 正常优化直接保留已测执行器；通用候选绑定支持原函数与变换 |

部分接口在设计文档中明确保留为外部/调试入口，因此不能称为“作者忘删的死代码”。但四种特殊入口增加了重复的接线、限制和验证用例，值得评估是否还需要这么多。

**建议：** 保留一个清晰的外部重建/检查入口，其余改成测试辅助或合并。迁移时先保留“拒绝变换后函数冒充原函数”等约束，不能只替换构造器就删除检查。

## 6. 两个没有任何仓库调用的辅助 API

**分类：无调用删除候选；优先级低，但确定性高。**

- `PageBundleArena.field_rows`，`pllm/execution/ptts/page_bundle.py:929`：stack 所有 pin 对应字段的张量。源码、测试、工具、示例、文档均未发现该方法的调用。它是一个额外的复制式读取入口。
- `StateTaskContext.publication_settled`，`pllm/execution/state_execution.py:113`：将 publication 集合归约为 bool。除定义外未发现引用。

**建议：** 先检查是否有仓库外低层调用者；否则可以直接删除候选。这两项只有约十行实现，价值在于减少接口歧义，不能夸大为主要代码体积或性能问题。

## 7. 固定的 cleanup_stage 作为可配置契约传输

**分类：冗余描述；优先级低。**

`AttachmentCleanupStage` 只有 `REQUEST_FINISH`，见 `pllm/contracts/backend_attachment.py:47`；构造端固定填入它，见 `pllm/execution/backend_attachment.py:181`。Rust 定义对应字段，见 `rust/pllm_core/src/planning/backend_attachment/types.rs:226`。

Python/Rust 执行代码均未按该字段选择清理时机。其作用是序列化、枚举合法性和参与整体摘要；清理行为由现有请求结束流程实现。

**建议：** 把固定清理语义作为当前 schema 的约定，在下次 schema 收敛时去掉这类单值选择字段。它本身不大，但属于“看起来有策略，实际只有一个固定值”的契约负担。

## 8. 测试解释器放在生产包中

**分类：测试基础设施的位置问题；不是无用功能。**

`pllm/execution/region_reference.py` 共 243 行，模块说明明确写着 serial oracle、不是生产执行路径。`interpret` 的仓库调用在 11 份边界测试中，覆盖结果和效果序列对照。

**建议：** 如果没有将它作为调试 API 发布的需求，可移到 `test/support`，使生产目录更准确地反映实际运行时。应保留独立 oracle：它能发现调度/执行实现的语义错误，删除反而损失验证能力。

## 已排除的误报

- `GPUWorker`、`workload_command`：通过字符串类路径和 `collective_rpc` 使用，见 `runtime.py:96`、`925`。
- `dense_fast_weights`、`request_distiller` 等属性：由创作层反射及声明字符串发现。
- `TraceSnapshot.host_task_times/device_task_times`、`load_loreft_interventions`：对用户有用的诊断/加载接口，无内部调用并不意味着无用。
- `TensorUpdate.metrics`：会随 `OverlayTargetResult` 返回，属于算法输出；框架不解释字典内容是合理的。
- 带 `legacy` 错误字符串的 `StateTaskContext.write/publish/publication`：仍被 `TensorOperation` 和 replay 使用，属于尚在使用的兼容语义，不能直接删。
- Rust retention、tiling、memory profile 的 Python FFI：某些 wrapper 只被测试调用，但实际分析可在 Rust 内部被生产规划调用，不能以 Python 引用数删除整个功能。
- 大多数签名字段、统计快照字段：序列化或对外观测就是消费者，不能因缺少显式 Python 读取判死。

前次发现的跨执行域规划无本地执行支持、七种状态表示只有 dense arena 落地，仍成立；本报告不重复把它们算作新增发现。

## 验证与建议顺序

pytest 使用 **`/workspace/pllm/.venv/bin/python`**：

```bash
/workspace/pllm/.venv/bin/python -m pytest -q \
  test/workloads/test_inplace_region.py::test_production_tuning_uses_signed_policy_and_remaining_transition_budget \
  test/boundary/test_region_outlining.py \
  test/boundary/test_region_borrows.py \
  test/boundary/test_shared_page.py
```

结果：**21 passed，20 warnings，19.20 秒**。warnings 为 PyTorch TreeSpec 弃用提示。此次没有启动模型或执行 GPU benchmark；测试通过不证明这些低层功能已有生产调用。

额外 CPU 复现脚本：`/tmp/pllm_audit_reproduce.py`；输出：`/tmp/pllm_audit_reproduction.json`。临时脚本不属于仓库测试套件，调优探针通过计数替换测量边界，不声称得到真实耗时。

建议先处理第 3 项的策略一致性；再处理第 2 项对用户暴露的粒度语义；第 1 项按近期功能目标决定隔离或接入。第 4—6 项可作为小规模接口收敛，第 7 项随 schema 更新，第 8 项随测试目录整理。不要以减少行数为目标删除语义 oracle、已使用的 IPC 或仍在生产调用的兼容接口。
