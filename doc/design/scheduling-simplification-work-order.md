# 调度精简：实现与验收记录

代码基线：`2ed84dc`。本记录说明当前工作区的实现，取代此前逐步讨论的施工草案。

## 范围

计算捕获、自动分块、数学变换规则和依赖/效果分析保持原样。本轮消费它们已有的结果，收敛安排、候选编排、绑定和执行接入。未新增运行模块、执行器类或公开配置字段。

`module_graph.py`、`exported_region.py`、`region_outlining.py`、`tensor_rewrites.py`、`tensor_batching.py`、`tensor_access.py`、`region_storage.py`、`region_resources.py`、`region_frame.py`、`region_control.py`、`usage.py`、PageBundle arena 与 capture 的文件哈希均与基线一致。Rust `derive_analysis` 实现逐字一致；同文件只修改 schedule 生成与调用接线。

## 现在从哪里读代码

| 职责 | 入口 | 输出 |
| --- | --- | --- |
| 固定调度与物理安排 | `region_streams.py::plan_region_schedule` | 一个预算内的原有 proposal，或明确无可行安排 |
| 共享候选准备和准入 | `region_optimization.py::prepare_region_candidates` | 本地产物、原始请求、完整 Rust assessment |
| 本地绑定 | `region_optimization.py::bind_region_candidate` | `RegionInstructionExecution` |
| 静态准备 | `region_optimization.py::prepare_region_execution` | 确定的执行器，无测量报告 |
| 显式纯图调优 | `region_optimization.py::optimize_region` | 完整的 `SelectedRegion` 报告 |
| EngineCore 调优事务 | `tokenwise_tuning.py::TokenwiseTuning` | 复用上述准备/绑定，只管理原引擎的试验、恢复和选择 |
| 执行 | `region_execution.py::RegionInstructionExecution` | 消费已绑定计算与原资源；不决定调优模式 |

`SelectedRegion` 现在仅是具有 `execution/request/selection/rewrites` 四个确定字段的冻结报告，没有 `run`，不是第二个执行包装。执行器没有 `request=None`、`selection=None` 或调优模式字段。

## 默认行为

- 使用一个 ISL parallel 配置；保留原 reference 作为容量与合法性基线。
- 默认不建立 window/tile 候选目录，不调用 `_measure`、私有 generate 或 TUNE RPC。
- 复用双 stream，生产安排固定 `lanes=2`、`pipeline_depth=2`。对给定顺序按稳定 axis 顺序找到第一个合法、预算内的映射即停止。
- 优先使用已有独立工作上的并发安排。不能容纳时使用已有合法顺序；连 reference 都不能容纳则准备失败，运行失败后不重跑有状态任务。
- 在签署原 task workspace 之前计算并发 payload/retention，worker 只能消费已签署额度。
- 当前参数、设备、数值选项进入冷缓存身份。完整函数就绪时在安装阶段绑定，不把准备拖入首次请求。
- In-Place 在冷绑定时分配辅助 stream，运行时复用；默认不生成计时用的零值试验输入。

默认并发消费分析已经给出的计算块，不以 workload 名字分派，也没有改写分析层来扩充其当前支持范围。

## 显式调优与强类型边界

`tuning="measured"` 才进入测量。候选上限为 8；已有高级 source 顺序与最多 4 项实现变体形成线性列表，不做 schedule × rewrite × compile subset × stream depth 的乘积。变体预算从尾部截断，因此改写族内 compiled 先于 reference 发射：截掉中间项可再生，截掉完整组合（outlining+rewrite+compile）不可恢复。

给定一个高级顺序，同样只做一次固定物理安排。图未改变时，编译实现沿用已有 source 顺序并重做物理事实；outlining 改变程序后才对新程序作固定安排。

- `prepare_region_execution(...)` 不接收样例输入，返回执行器。
- `optimize_region(..., inputs, ...)` 要求明确的 measured 策略和真实样例，返回完整报告。
- `TensorScan.prepared(...)` 返回静态 scan；`TensorScan.optimized(...)` 返回 `(scan, report)`。
- 调优报告不进入执行器状态。原本就存在的“源尚未绑定”、外部资源/stream 是否提供，以及 RPC 命令先后顺序，在相应安装或程序边界检查。
- phase/event/引用保留、host fence、取消和资源回收实现保持不变。

## 真正删除的重复结构

删除了 `_physical_candidates`、`_compiled_variants`、`bind_selected_schedule`、`RegionStreamExecution`，以及旧的 `_candidate_sources`、`_replay_outlines`、`_replay_compilation`、`bind_selected_region` 调试重建路径。

Tokenwise 的 `measurement_schedule`、`with_measured_selection`、`with_schedule` 等专用安装路径被移除。worker 已经本地生成并通过准入的产物只绑定一次，不再把同一个数学描述符从头重建一次。矩阵规则及其自身的 replay 校验仍在原模块中。

外来数据继续经原解码、源引用复核和 Rust 准入；本地 binder 检查完整 assessment 的可行性、预算和实际 imports。原 EngineCore 不传输 callable 的边界保持不变。本轮不再提供从外来数学候选描述符重建任意执行体的旧调试 API。

## 规模

| 核心模块 | 基线行数 | 当前行数 | 基线非文档 AST 语句 | 当前非文档 AST 语句 |
| --- | ---: | ---: | ---: | ---: |
| `region_optimization.py` | 518 | 332 | 317 | 223 |
| `tokenwise_tuning.py` | 181 | 117 | 132 | 90 |
| 合计 | 699 | 449 | 449 | 313 |

核心行数减少约 36%，非文档 AST 语句减少约 30%。运行代码 16 个文件新增 401 行、删除 629 行，净减少 228 行；没有把删除的实现迁到别处。测试和文档不计入运行代码缩减。机器可读记录见 [scheduling-implementation-results.json](../audits/scheduling-implementation-results.json)。

## 已有验证记录

pytest 的 Python 环境统一为 `/workspace/pllm/.venv/bin/python`；GPU 为 RTX 4090。

- Rust 库测试：22 passed。
- 最终 CPU 边界与 workload 回归：509 passed，106 skipped，72.36 秒。
- 静态检查：typing 零错误；import layers、region authority 和 diff 检查通过。
- Qwen 进程内默认：14 次独立数值比较、两个实际 issuing stream、KV/权重/清理检查通过，fallback=0。
- EngineCore 默认：无调优时间样本，无 TUNE；12 个 replay 行；父进程 CUDA 未初始化，正常关闭。
- 显式 worker 调优：本地计算候选、并发绑定、原 owner、RNG/trace 恢复检查通过。
- EngineCore measured：7 个可行候选，有并发和变换候选；12 个 replay 行，父进程 CUDA 未初始化，正常关闭。
- GPU stream/native kernel/In-Place 定向组合：33 项通过；其中原 model-proposal 测试调整到当前固定 mapping 接口后另行通过。
- In-Place 真实模型回归：dense、low-rank、imported、routed 四类场景通过；更新/应用计数、版本/原模型对照和清理断言通过，fallback=0。

上述验证证明功能与所有权行为，不等于所有 workload 都更快。

## 性能验收边界

完整的长请求、batch 8 对照尚未完成，不能声称已通过所有场景的 5% 吞吐门槛。

旧版基准在较大配置上首先遇到 ISL 操作上限；收窄并发容量后，长请求又暴露旧执行路径的 schedule expression 类型错误。这些不通过改动本轮冻结的分析层来掩盖。后续按用户要求停止持续基准追查，优先完成代码结构。

当前不据私有调优样本声称普遍加速，也不以回退全串行冒充默认并发验收。任何性能结果必须单独注明模型、请求长度、实际并发容量、轮数与冷/热阶段。
