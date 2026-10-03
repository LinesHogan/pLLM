# 调度策略复杂度审计

日期：2026-09-15。基线：`2ed84dc` 当前工作区。此次只审计，不修改调度实现。

结论：以“合法移动计算、组织可编译计算块、让独立计算在 stream 上重叠”为主线是合理的。现有代码的主要问题是优化目标与执行能力没有完全对齐，以及在重排到融合的连接尚不完整时，已经扩展了较多候选维度。不能因此把 window、wavefront、事件和生存期管理一并视为遗留代码。

## 用户提出的主线需要补充什么

1. 提前计算以输入可用、读取版本正确、效果顺序不变为前提。更早产生大张量会延长其生存期；“尽可能早”还需受到内存预算和实际收益约束。
2. 调用在时间上相邻并不会自动融合。必须将相关计算组合成同一可捕获/编译的图，Inductor 才有机会融合；进入同一图也不保证最终只有一个 kernel。
3. CUDA stream 不能绕过依赖。有依赖的 A→B 仍然必须等待；额外 stream 只可能使其与其他独立工作重叠。提前计算与多 stream 是可以组合的两个维度。

## 现有机制各自做什么

| 机制 | 位置 | 实际职责 | 判断 |
| --- | --- | --- | --- |
| 外层 task/anchor/AER 选择 | `planning/scheduler.rs`、`planning/aer.rs` | 选 task 实现、分组、生命周期时机、执行域 | 生命周期调度必要；分组代价模型需要重新审视，分布式域仍未被本地执行器支持 |
| ISL locality/parallel 候选 | `planning/region_dependence.rs:1309` | 根据相同依赖图，用两个 ISL 配置生成调用顺序 | 同一调度器的两种候选，非两套执行器；不是纯 ASAP 策略 |
| wavefront | ISL 产生的合法安排 | 在跨 token/layer 依赖下推进可同时进行的调用 | 属于依赖分析产生的结果，不需要单独作为算法特例维护 |
| window | `planning/region_tiling.rs:156` | 限制调用偏离参考顺序的范围 | 对提前计算导致的保留内存增长有实际作用 |
| tiling | `planning/region_tiling.rs:42` | 对可置换循环维度生成方形/矩形 tile | 此处改变调用实例的顺序，不是 Inductor 的 kernel 内部 tile；泛化搜索可后置 |
| outlining | `contracts/region_outlining.py:38` | 把相邻纯 tensor 调用组合成同一图 | 是交给 Torch 编译的必要桥接之一，但目前只看原源码相邻关系 |
| reassociation/batching | `execution/tensor_rewrites.py`、`tensor_batching.py` | 改变矩阵计算结构或把独立 mm 合为 bmm | 属于数学/图优化，不是另一套 stream 调度；可以和调度核心解耦 |
| stream placement/pipeline | `planning/region_streams.rs`、`execution/region_streams.py` | 分配 lane，发出跨 stream 等待，限制在途 phase 与存储 | placement/completion 必要；可搜索 axis/depth 的部分可以按证据收敛 |
| 实测选择 | `region_optimization.py`、`tokenwise_tuning.py` | 组合候选、测量、选中、安装 | 能比较实际执行；候选覆盖与样本代表性值得控制 |

## 发现一：重排没有完整地反馈给计算块融合

`call_outline_sites` 只从 `region.block.operations` 中扫描连续、纯张量的 `Call`；遇到其他操作即截断。它不接收 schedule map。

`region_optimization.py:381` 在原程序上生成 outlining 候选；`_candidate_family` 再为原始或已 outlining 的程序生成顺序、改写和编译实现候选。这里不是“先按候选顺序移动原图，再查找新形成的可融合区域”。

CPU 最小复现使用两个矩阵调用，中间插入一个独立且被返回的整数常量：

```text
原程序：F → Constant(7) → G(F)
合法调度：F → G(F) → Constant(7)
```

原程序的 outlining 位置数为 **0**。Rust 校验接受重新排序的 schedule，执行顺序为 statement **[0, 2, 1]**。显式将原程序重排后，outlining 位置数为 **1**；合并后的执行结果与前述合法调度逐位相同，常量仍为 7。

这证明的是一个具体能力缺口：合法调度产生的邻接关系没有自动变成新的跨调用编译机会。该复现手工提交合法 schedule，不声称现有自动选择器会选中该顺序；也未运行 Inductor/GPU 去证明融合速度。

**建议：优先打通候选安排与可编译图分组的连接。** 同时保留原实例身份、输入版本、条件和逃逸值的验证，不把任意状态调用吸收进纯计算图。

## 发现二：外层“少物化、少分组”评分并不直接兑现为跨 task 融合

外层 `aer.rs:726` 枚举 task 分组；其代价向量在 `aer.rs:998` 按下列顺序比较：

```text
跨组/导出值字节、跨域传输成本、分组数量、峰值内存、组间依赖数
```

`materialization` 在 `aer.rs:918` 按“值是否导出或跨组消费”计算。可是在 `execution/execution.py:82`，组内仍逐项调用原 task 的 `execute(context)`，然后执行各 task 的值释放。这里没有根据分组生成共同 Torch 图，也没有因为组内消费就自动消除中间 tensor。

因此，“更少的跨组物化字节/更少的组”只能作为待验证的结构性代理指标，不能直接解释成实际减少的 tensor 分配、kernel launch 或融合收益。外层分组仍影响合法时机和顺序，因此并非整体无效。Region 内部已有的 outlining/Inductor 功能是另一层，不应混为一谈。

**建议：** 为每个保留的评分项指出实际执行端的收益来源。若外层不负责跨 task 生成融合图，应考虑将它收窄为生命周期窗口、版本依赖和资源准入，将纯计算块的组合交给统一的 Region/Torch 路径。此审计未测量外层搜索耗时，不据此声称它是性能瓶颈。

## 发现三：多个维度的候选搜索先于明确的收益理由

当前共享调度目录轮流抽取 global、window、tile 三类候选（`region_tiling.rs:267`）。window 使用尺寸 2、4，tile 使用 2、4、8、16 并生成矩形变化。

stream 搜索随后遍历调度的时间维度 axis，并尝试 pipeline depth（`region_streams.rs:223`）。Python 层另组合 outlining、数学改写、单函数编译/编译组合，以及串行/并发实现（`region_optimization.py:285`、`316`、`347`、`477`）。

这些搜索有上限、缓存、合法性校验和去重，不能称为无界组合爆炸。问题是有限名额主要由枚举顺序和轮转分配，而不是先判断该 workload 面临的是大中间张量、可融合计算、launch 开销还是 GPU 空闲。

**建议：** 默认只生成有明确理由的少量候选：参考安排、形成编译机会的合法移动、确有独立工作的 stream 安排。window 可作为预算限制下的移动范围；泛化矩形 tiling、多 axis/depth、编译子集搜索保留在显式调优模式。通过真实 workload 比较冷启动、吞吐、峰值内存后，再决定退出哪些候选家族。

## 发现四：stream 实现带有较强的 phase 同步模型

Rust 将 schedule 的一个时间坐标按 lane 数做整除，得到 phase，再按余数分 lane（`region_streams.rs:108`）。它要求依赖跨 phase，并为 pipeline 推导跨 lane 前驱。

Python `RegionStreamFrame._retire` 调用 `_join`，后者既插入 owner stream 等待，又逐个 `synchronize()`（`region_streams.py:159`）。depth=1 时按 phase 完成再推进；更大 depth 保留更多在途引用，等窗口满或遇到宿主操作时收敛。

因此当前并发执行比“在原始 DAG 上给跨 stream 边插 event”多了一层 phase、待完成队列和预算模型。这些复杂度用于限制内存和保护 host/page 操作，具有实际目的。不能简单删除同步，否则提前释放、复用 scratch、宿主读取等行为可能错误。

**建议：** 将 stream 编排视为同一合法计算 DAG 的物理实现。初期收敛为 owner+一条辅助 stream、明确的依赖事件和在途内存上限；仅在需要回收、宿主观察或结束时等待。是否能替代 phase 模型，需要先明确张量与 scratch 的实际所有权，并做 CUDA 验证，此次未实施。

## 哪些不能依据“策略多”直接删

- **window：** 现有测试证明，同一模型区域中全局前移候选可能超出保留内存预算，而窗口候选可行。测试使用固定评分，证明容量差异，不证明窗口的实际速度。
- **tiling：** 存在与 global/window 不同的合法执行顺序；不能说它只是同名重复策略。但不同顺序也不足以证明值得默认搜索。
- **pipeline 的 event 与引用保留：** 是异步执行正确性的必要工作，能收敛表示，不能取消保证。
- **request replica lifting：** 压缩重复请求的编译搜索，是编译器自身的成本优化；它不是又一套 GPU 并行策略。
- **CUDA Graph 与 GEMM batching：** 分别减少提交成本和改变计算形状；不能仅因也有性能收益，就当成和 stream 相互重复。

## 当前默认与测量边界

native replay 默认 `tuning="heuristic"` 直接保留参考计划，并未开启这些候选搜索（`tokenwise_region.py:427`、`runtime.py:815`）。前一份审计已确认 In-Place 的冷调优没有消费这个开关，不能将 native 默认行为推广到所有 workload。

native measured 路径用一个短 prompt、`max_tokens=1`，每候选一轮预热及两轮计时（`runtime.py:841`）。它可以校验并选择这个样本的执行，但不能自动证明长 prompt、大 batch 或长期吞吐收益。特别是复杂候选家族的取舍，不宜只依赖该样本。

## 验证

使用 **`/workspace/pllm/.venv/bin/python`** 运行：

```bash
/workspace/pllm/.venv/bin/python -m pytest -q \
  test/boundary/test_region_windows.py \
  test/boundary/test_region_tiling.py::test_shared_catalog_keeps_distinct_global_window_and_tiled_orders \
  test/boundary/test_region_streams.py::test_pipeline_frontiers_cover_each_grid_predecessor_and_fund_pending_values
```

结果：**3 passed，2.32 秒**。另运行上述重排与 outlining 的 CPU 复现。未修改调度实现，未进行本轮 GPU/模型性能验证。
