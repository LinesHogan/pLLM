# 文档

这里的指南描述当前实现。已退役的 API 指南与迁移报告不与当前架构并存维护。

## 教程（中文，按顺序阅读）

- [00 快速上手：心智模型与最小观测负载](tutorials/00_quickstart.ipynb)
- [01 观测：给三层 hidden 装 SAE](tutorials/01_observe_sae.ipynb)
- [02 自定义训练头：从线性头到 CNN](tutorials/02_custom_head.ipynb)
- [03 每请求 LoRA：路由共享、混合与 prefill 学习](tutorials/03_request_lora.ipynb)
- [04 三种循环：层间、步间、会话](tutorials/04_loops.ipynb)
- [05 decode 时在线训练：跨步与跨请求目标](tutorials/05_decode_training.ipynb)

## 指南与参考

- [安装与 ESamp 示例](../README.md)
- [让代理写负载：输入与工作流](guides/agent-workloads.md)
- [负载创作契约](guides/authoring.md)
- [词汇表：一个术语一个含义](glossary.md)
- [验证与发布检查](guides/validation.md)
- [基准解读](guides/benchmarking.md)

## 设计文档

- [In-Place-TTT 稠密 overlay](inplace-ttt.md)
- [当前架构与所有权](architecture.md)
- [整合计划](architecture-consolidation-10k-plan.md)
- [可编程推理设计与共享边界](programmable-inference-next-design.md)
- [共享高性能原语设计](pllm-next-stage-design.md)

整合计划记录验收与排除项。可执行证据清单在 `test/evidence_manifest.toml`；
不存在第二份进度台账。
