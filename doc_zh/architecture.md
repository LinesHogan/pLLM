# 架构

## 分层结构

```
┌─────────────────────────────────────────────┐
│ workflows/          入口（benchmark、       │
│                     starter、esamp 支持）   │
├─────────────────────────────────────────────┤
│ consumers/          叶子：esamp（legacy +   │
│                     program）、你的 consumer│
├─────────────────────────────────────────────┤
│ backends/           vLLM v1 适配器          │
├──────────────────────┬──────────────────────┤
│ runtime/             │ program/             │
│ 执行：lane、sampler  │ 编译器：声明 → 切片  │
│ bridge、统计、插件   │ → 校验 → 生成 consumer│
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

## 程序编译管线

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

## 运行期执行

生成的 consumer 在 vLLM 生成过程中运行。执行策略从声明的 DAG 推导，不硬编码：

- **State bank**（`program/state.py`）：带版本的 slot，持有参数与 optimizer 状态，scope 为 shared 或 per-request。发布是引用交换 + 事件 fence，读者永远看到一致快照。
- **Lane**：训练 lane 独占状态写入；reader lane 服务 `run_flow` 预测；lane 间的边恰是声明的数据依赖（输入就绪等待 + 完成事件 fence）—— 绝不 blanket 等整条流。
- **Grad 模式**：由 `result()` 边推导。输出喂给到期 extern 的 flow 以 grad 模式跑一遍；否则 `no_grad`。
- **CUDA Graph 捕获**：对持久 buffer 上固定 DAG 的步（CUDA 上恒等 batch 组、AdamW 标量），按 batch 形状 bucket 捕获后 replay。首次 replay 与 eager 函数式 transition 校验（`PG146`）。graph-vs-eager 是内部执行决策，对 consumer 不可见。

## Sampler bridge

`runtime/sampler_bridge/` 把 consumer 预测接到 vLLM sampler：provider 经 `run_flow`（reader lane）读当前 state bank，产出候选级 logit 调整，在 top-k/top-p/min-p 过滤后应用。Consumer 永不触碰 sampler 内部。

## 门禁

开发由三重门禁保护：import 分层门禁、改动面的严格 typing（其余只降不升 baseline）、benchmark 功能计数（loss 数、干预数）——吞吐数字永远伴随"算法真的跑了"的证据。
