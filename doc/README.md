# 文档

这份目录按 PyTorch 文档的认知分层组织：**教程**带你学会，**使用指南**回答
"怎么做 X"，**参考**供逐条查证，**设计说明**解释为什么这样。同一事实只在
一层展开，其他层只给链接。

```text
第一次接触 → 教程 00 → 按兴趣分流：算法向 01–05 / 系统向 internals 三篇
写自己的负载 → 使用指南（任务导向）
查某条声明的精确边界 → 参考：创作契约 / 词汇表
理解架构与所有权 → 设计说明
```

## 教程（学习导向，可执行 notebook）

**算法向**（GPU）：从心智模型到完整场景，按顺序读。

- [00 快速上手：心智模型与最小观测负载](tutorials/00_quickstart.ipynb)
- [01 观测：给三层 hidden 装 SAE](tutorials/01_observe_sae.ipynb)
- [02 自定义训练头：从线性头到 CNN](tutorials/02_custom_head.ipynb)
- [03 每请求 LoRA：路由共享、混合与 prefill 学习](tutorials/03_request_lora.ipynb)
- [04 三种循环：层间、步间、会话](tutorials/04_loops.ipynb)
- [05 decode 时在线训练：跨步与跨请求目标](tutorials/05_decode_training.ipynb)

**系统向**（内部机制，写给想搞清框架怎么运转的读者，纯 CPU 可运行）：

- [签名编译管线：声明如何变成签名计划](tutorials/internals/01_signed_compilation.ipynb)
- [托管状态与页 arena：版本、COW、pin、预算](tutorials/internals/02_state_arena.ipynb)
- [进程边界：按引用传输与签名复核](tutorials/internals/03_process_boundary.ipynb)

## 使用指南（任务导向）

- [写一个负载：声明、时机、输入注解](guides/write-a-workload.md)
- [状态与记忆：作用域、冻结、route 键、memory 与捕获](guides/state-and-memory.md)
- [训练与观测：loss 约定、梯度边、导出与 trace](guides/training-and-observation.md)
- [Overlay 与重放：引擎内训练、外部导入、激活检查点](guides/overlays-and-replay.md)
- [让代理写负载：输入与工作流](guides/agent-workloads.md)
- [验证与发布检查](guides/validation.md)
- [基准解读](guides/benchmarking.md)

## 参考（信息导向）

- [负载创作契约](reference/authoring-contract.md) —— 全部能力边界与禁止事项
- [词汇表：一个术语一个含义](reference/glossary.md)

## 设计说明（解释导向）

- [当前架构与所有权](design/architecture.md)
- [In-Place-TTT 稠密 overlay 语义](design/inplace-ttt.md)
- [整合计划：验收与排除项](design/architecture-consolidation-10k-plan.md)
- [可编程推理设计与共享边界](design/programmable-inference-next-design.md)
- [共享高性能原语设计](design/pllm-next-stage-design.md)

整合计划记录验收与排除项。可执行证据清单在 `test/evidence_manifest.toml`；
不存在第二份进度台账。已退役的 API 指南与迁移报告不与当前架构并存维护。
