# 附加计算负载自动调度系统架构报告（修订版）

## 修订说明

本修订基于 CUDA 硬件洞察（`cuda_insight.md`）。修订保留原报告的骨架与产品形态，仅对**被证伪或不足**的要点进行勘误与补强。

---

## 1. 对“系统定位”的勘误

### 原报告
> 系统定位是编译器中端（Middle-End）的调度综合层……默认策略优先选择内存开销最小的顺序重叠。

### 修订
系统定位不变，但**默认策略的优先级需要修正**。`cuda_insight.md` 明确指出：现代 GPU 的硬件调度器具有亚微秒级反馈循环，只要提交的 CTA 总数不少于物理 SM 数，硬件自己就能高效填满 SM。**静态微调度（如精确指定 kernel 时序、手动分配 SM、精细编排 stream event）在 GPU 上是被证伪的方向**——Intel Itanium（VLIW/静态调度）的死亡已经说明，在具有完整架构状态知识和动态投机能力的硬件面前，软件侧的计划经济式微调度没有胜算。

因此，本系统的目标不是“替 GPU 决定每个子任务何时跑”，而是：

1. **发现合法的并行与重排机会**（isl 多面体分析）；
2. **把足够大的合法计算区域交给 GPU**，让硬件动态调度发挥能力；
3. **避免人为制造不必要的边界**（materialization、graph break、过早切分），这些边界会阻断硬件调度的视野。

**结论**：顺序重叠只是安全 fallback，不是默认最优。默认策略应是**在内存预算硬约束下，尽可能形成大的 ExecutionRegion，让 region 内部有足够的并行度使 GPU 饱和**。

---

## 2. 新增核心抽象：ExecutionRegion

### 原报告
> 调度决策包含子任务列表、插入锚点、并行策略、内存分配方案。

### 修订
上述输出粒度太细，会导致“每个子任务一次 launch、一次 event、一次 buffer 分配”的碎片化执行。修订建议正确指出的核心缺口是：**缺少“粒度选择层”**。

系统必须在逻辑 task 与物理执行之间引入 **ExecutionRegion**：

- **ExecutionRegion** 是一组 task（或一个 affine tile/batch）的集合，它们共同编译、共同捕获、或由一个 executor 批量执行；
- **Region 是物理执行的基本单位**，不是单个 task；
- Region 内部的依赖保留为图内边（Tensor/FX edge 或 CUDA Graph 内依赖），**不转化为跨 stream event**；
- 只有跨 Region、跨设备或跨 backend boundary 的依赖才需要显式 fence。

**为什么必须加这一层**：`cuda_insight.md` 指出，kernel launch 开销仅约 5μs，CUDA Graph 可将数百个 kernel 一次性提交。真正决定性能的不是“launch 够不够快”，而是**Region 内是否有足够的 CTA 让 GPU 填满 SM**，以及**Region 是否因过早拆分而被迫 materialize 中间结果**。

---

## 3. 对“设计原则”的修正

### 3.1 数据搬运仍是一等公民，但“敌人”重新定义

原报告已正确将数据搬运成本置于计算成本之上。现进一步明确优先级：

| 优先级 | 成本项 | 理由 |
|--------|--------|------|
| 硬约束 | 内存预算 | 大模型推理内存已接近物理上限 |
| **第一** | **Materialization bytes** | HBM round-trip 是毫秒级，直接吃掉 overlap 收益 |
| **第二** | **Graph break / Capture boundary** | 破坏 CUDA Graph 机会，阻断 Inductor 融合视野 |
| **第三** | **Peak live memory** | 决定能否在显存内完成 |
| 第四 | Cross-stream fence / launch | 5μs 级，通常可忽略 |

**结论**：中端不应花大量精力优化 stream event 数量或 kernel launch 时序。这些属于“已经被硬件和 CUDA Graph 解决得足够好”的问题。中端的优化目标是**减少 materialization 和 graph break**。

### 3.2 纯函数假设 → Effect-aware

原报告要求附加负载必须是“纯函数”。这在实际中过于严格，会排除大量合法 workload（如 state write、extern call）。

修订为 **Effect-aware**：
- 用户可声明 task 的 effect 类型（pure、state_write、extern_io 等）；
- Rust Core 根据 effect 自动限制重排空间（例如 state transition 不能跨越 version publish）；
- 纯 task 可最大化重排，effect task 在约束内重排。

### 3.3 新增原则：硬件动态调度优先于静态微调度

> 本系统不预测 GPU 的精确执行时序，不分配时间片，不指定 SM。系统只输出**依赖关系**（谁必须在谁之前）和**资源约束**（内存预算、合法并行组），时序填充完全交给 GPU 硬件调度器。

---

## 4. 对“技术选型与软件架构”的修正

### 4.1 Rust Core 新增 Granularity Planner

Rust Core 在原有三模块（Dependency Analyzer、Polyhedral Scheduler、Memory Planner）基础上，**新增 Granularity Planner**：

- 输入：isl 产出的 schedule tree（逻辑调度）；
- 处理：根据 materialization 成本、graph-safety、backend capability 和内存预算，**将逻辑 task 联合 regroup 成 ExecutionRegion**；
- 输出：包含 Region 列表的 `ScheduleDecision`。

**关键**：不是“拆得越细越好”，而是**“分析粒度足够细（暴露依赖和并行机会），物理粒度足够粗（减少边界开销）”**。

### 4.2 成本模型扩展

`CostModel` 在原有 `compute_cost`、`memory_cost`、`transfer_cost` 基础上，增加：
- `materialization_bytes`：因 Region 分割而保存或复制的中间值；
- `capture_boundary_count`：破坏 CUDA Graph 或 `torch.compile` 视野的边界数；
- `launch_group_count`：物理执行单元数（越少越好）。

### 4.3 torch.compile 策略简化

原报告未明确 torch.compile 的渐进路径。现明确：

- **短中期**：附加负载的 task 组成 Region 后，**直接整体交给 `torch.compile` 做内部 fusion/codegen**。不需要分阶段文档化 `external` / `captured_tap` / `joint_lowering`；
- **长期**：当 backend 支持时，自然延伸到与 vLLM piece 的联合 lowering。这是一个远景，不进入当前执行计划。

---

## 5. 对“正确性与安全”的强化

### 5.1 数值等价性

原报告认为“近似相当即可”。修订为：

- **Exact policy**（如 ESamp）：reschedule 输出必须与 serial oracle 做 **数值 parity**（bit-wise 或 epsilon-level）；
- **Approximation policy**（如 Jacobi）：允许语义偏差，但必须显式标记为独立 policy，进入独立的 cache key 和验证路径。

### 5.2 失败策略

原报告提到“优雅降级到顺序执行”。修订为：

> 无合法 candidate 时 **fail closed**。已证明合法的 serial plan 可作为显式 fallback decision，但“fallback”是编译出的显式决策，不是 runtime 静默跳过。

---

## 6. 对“产品形态”的不变声明

以下原则**不因修订而改变**：
- 装饰器 / 声明式 API 仍是面向用户的产品形态；
- Rust Core 仍是“只决策、不生成”的调度引擎；
- Backend trait 仍负责对接具体框架；
- 不复用 PyTorch/Inductor/CUDA 已有能力的立场不变；
- 内存预算仍是硬约束；
- fail-closed 原则不变。

---

## 7. 一句话总结修订

> **中端用 isl 发现合法的并行与重排机会，用 Granularity Planner 把这些机会打包成足够大的 ExecutionRegion，让 CUDA Graph 和 GPU 硬件动态调度填满 SM。Materialization 和 graph break 是主要敌人，5μs 的 stream launch 不是。**