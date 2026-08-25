# 架构

## 当前执行模型

`tllm.workload` 声明用于细粒度分析的图，但 `ExecutionRegion` 才是粗粒度的物理执行单位。Rust
推导、校验并签名每个 Region 的 binding、materialization、fence、lifetime、artifact identity 与
capture policy。Python 不推断 task 语义、不重新调度，也不选择 capture；它只机械物化已签名
artifact，并在已签名 anchor 提交 Region。event/fence 只能出现在 Region、设备或 backend 边界；
Region 内依赖是普通 FX/Tensor/CUDA-Graph 边。

`RegionCaptureMode` 是随 decision 签名的 Rust 闭合 enum。它在 Python lowering 前确定允许的
capture 行为（`disabled` 或 `require`）；Python 不会从 task 类型、stream 或运行期名称推断
capture。缺 capability、artifact 或 binding 会在 cold install 时拒绝。

在 `torch==2.13.0+cu129` 与 `vllm==0.27.1` 上，GPU 正确性已验证 native Dummy、native
Recirculation 与 native ESamp 的 exact eager in-process 执行，均通过 MRv1/MRv2。
Recirculation 观察到 typed layer replay 且 fallback 为零。最新 ESamp 短 smoke：
`loss_count=6`、`update_count=6`、`state_version=6`、`fallback_count=0`。native ESamp exact
eager Region 还通过了两种 runner 的 EngineCore MP 正确性：loss、update、state-version、
intervention 均非零，fallback/replay 均为零。这不是吞吐结论；model-backed native piecewise
CUDA Graph、Recirculation EngineCore MP/capture 集成和 shape-miss epoch switching 仍未验证或未完成。

## 分层结构

```
┌─────────────────────────────────────────────┐
│ workflows/          入口（benchmark、       │
│                     starter、esamp 支持）   │
├─────────────────────────────────────────────┤
│ consumers/          legacy adapter 与       │
│                     native workload shell   │
├─────────────────────────────────────────────┤
│ backends/           vLLM v1 适配器          │
├──────────────────────┬──────────────────────┤
│ runtime/             │ workload/            │
│ Region executor、    │ 声明 → 图 → 已签名   │
│ sampler bridge、统计 │ Region plan           │
├──────────────────────┴──────────────────────┤
│ workgraph/          声明式语言表面          │
│                     （frontend、lowering、  │
│                     placement、admission）  │
├─────────────────────────────────────────────┤
│ common/ producers/  共享机制、张量生产      │
├─────────────────────────────────────────────┤
│ ports/              面向 consumer 的抽象    │
├─────────────────────────────────────────────┤
│ contracts/          纯数据类型              │
├─────────────────────────────────────────────┤
│ core/               Rust 扩展：小而内聚，   │
│                     只做决策                │
└─────────────────────────────────────────────┘
```

依赖只能向下。由 `tools/check_import_layers.py` 机械强制：AST 级门禁，每个包声明允许的 import 目标，违规计数由只降不升的 baseline 钉住。不抬高 baseline 就引入不了新的跨层债务。

两条原则：

- **Rust 决策，Python 执行。** `core/` 扩展做调度决策；所有张量执行留在 Python/torch，可检查、可调试。
- **Fail closed。** 每个编译期拒绝都是稳定错误码；不存在静默 fallback 路径（`PG190` 的存在就是为了让这种企图变成显式报错）。

## Native Region 管线

```
@workload / @task
      │ 类型化 value、effect、timing、affine access
      ▼
ScheduleGraph ──► Rust 合法性 + 粒度 + memory planning
      ▼
已签名 ScheduleDecision v2 ──► ExecutionRegion + RegionCaptureMode
      ▼
PhysicalFlowPlan ──► 每个 Region 一个 artifact/callable，在签名 anchor 提交
```

planner 从最大的合法 Region 开始，仅在 anchor window 无交集、必须可见的 effect/version 或
typed-extern 边界、backend/device/graph-safety 边界，或硬 memory constraint 时拆分。它最小化
显式 materialization 与边界；不选择逐 task stream、不预测 kernel 时序，也不分配 SM。

## Legacy Program 兼容管线

```
ESampProgram（你的 Module）
      │  @flow / @extern 声明
      ▼
declare_program ──► 静态 IR（flow、extern、state spec、边）
      ▼
slice_flow_stages ──► 每个 flow 的 torch.export 图
      ▼
compile_program ──► optimizer 降级为函数式 transition；
                    oracle 校验：extern 体 ≡ 函数式 transition（PG145）
      ▼
ProgramConsumer ──► 运行期执行（见下）
```

## Legacy 运行期执行

以下 lane/state-bank 描述只适用于保留的 Program 兼容路径，不是 native Region 的优化契约。

生成的 consumer 在 vLLM 生成过程中运行。执行策略从声明的 DAG 推导，不硬编码：

- **State bank**（`program/state.py`）：带版本的 slot，持有参数与 optimizer 状态，scope 为 shared 或 per-request。发布是引用交换 + 事件 fence，读者永远看到一致快照。
- **Lane**：训练 lane 独占状态写入；reader lane 服务 `run_flow` 预测；lane 间的边恰是声明的数据依赖（输入就绪等待 + 完成事件 fence）—— 绝不 blanket 等整条流。
- **Grad 模式**：由 `result()` 边推导。输出喂给到期 extern 的 flow 以 grad 模式跑一遍；否则 `no_grad`。
- **CUDA Graph 捕获**：对持久 buffer 上固定 DAG 的步（CUDA 上恒等 batch 组、AdamW 标量），按 batch 形状 bucket 捕获后 replay。首次 replay 与 eager 函数式 transition 校验（`PG146`）。graph-vs-eager 是内部执行决策，对 consumer 不可见。

## Sampler bridge

`runtime/sampler_bridge/` 把 consumer 预测接到 vLLM sampler：provider 经 `run_flow`（reader lane）读当前 state bank，产出候选级 logit 调整，在 top-k/top-p/min-p 过滤后应用。Consumer 永不触碰 sampler 内部。

## 门禁

开发由三重门禁保护：import 分层门禁、改动面的严格 typing（其余只降不升 baseline）、benchmark 功能计数（loss 数、干预数）——吞吐数字永远伴随"算法真的跑了"的证据。
