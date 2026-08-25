# tLLM 文档

tLLM 是 vLLM 的附加载荷编译器。新工作负载用 `tllm.workload` 声明；Rust
生成已签名的 `ScheduleDecision`。task 图用于细粒度分析，`ExecutionRegion` 才是粗粒度的
物理执行单位；Python 只机械绑定签名计划。Program/Consumer 的 lane、stream、event 保留为
迁移兼容层，不是优化模型。

## 阅读顺序

1. [附加载荷编译器](workload-compiler.md) — 当前 frontend、已签名 Region 契约与迁移边界。
2. [快速上手](quickstart.md) — native ESamp 导览及保留的 Program 示例。
3. [编写 consumer](write-a-consumer.md) — `tllm.workload` 指南和 legacy Program 兼容 API。
4. [架构](architecture.md) — Region 执行栈与兼容层。
5. [ESamp 参考](esamp.md) — 内置的 test-time training consumer。
6. [适配 In-Place TTT](inplace-ttt.md) — 改模型自身权重的算法：学习一半是 Program，应用一半是 backend 提供的权重 overlay。
7. [TLM q/v LoRA](tlm.md) — 完整 HF/PEFT side training，以及通过 vLLM 原生 LoRA 路径发布不可变版本。

## 一段话心智模型

对于 native workload，用 `@workload` 与 `@task` 声明类型化输入、输出、effect、timing
以及可选 affine access。Rust 校验完整图，并签名 Region 边界、binding、materialization、
lifetime 与 capture policy。旧的 `@flow`/`@extern` Program lane 在 workload 迁移时仍可作为
adapter，但不会用于新的物理调度决策。

不支持的 native 或兼容声明都会 fail closed；没有静默 fallback。

在 `torch==2.13.0+cu129` 与 `vllm==0.27.1` 上，GPU 正确性已覆盖 exact eager in-process 的
native Dummy 与 native ESamp，均通过 MRv1/MRv2。最新 ESamp 短 smoke：`loss_count=6`、
`update_count=6`、`state_version=6`、`fallback_count=0`。native ESamp exact eager Region
还通过了 MRv1/MRv2 的 EngineCore MP 正确性：loss、update、state-version、sampler intervention
计数均非零，`fallback_count=0`、`replay_count=0`。这不是吞吐结论；model-backed native
piecewise CUDA Graph、真实 native Recirculation vLLM prefill 接管和 shape-miss epoch switching
仍未验证或未完成。

English docs: [doc/README.md](../doc/README.md)。
