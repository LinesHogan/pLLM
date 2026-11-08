# 文档

这份目录按 PyTorch 文档的认知分层组织：**教程**带你学会，**使用指南**回答
"怎么做 X"，**参考**供逐条查证。同一事实只在一层展开，其他层只给链接。
改 pllm 框架本身的贡献者文档在 [dev/](dev/README.md)。

```text
第一次接触 → 教程 00 → 按兴趣继续 01–13
某类算法能不能写 → 能力地图
写自己的负载 → 使用指南（任务导向）
查某条声明的精确边界 → 参考：创作契约 / 词汇表
改框架本身 → 贡献者文档（dev/）
```

## 教程（学习导向，可执行 notebook）

需要 GPU。每篇围绕一个算法，大多在 Qwen2.5-0.5B 上跑完（08 用一个 4B 的 In-Place TTT 社区检查点），输出保留在 notebook 里。
13 会调用编码 agent 跑一个搜索循环，需要 bubblewrap 和已登录的 agent 命令行。

- [00 快速上手：把激活引导（CAA）搬进 vLLM](tutorials/00_quickstart.ipynb)
- [01 观测：用探针逐 token 监控生成](tutorials/01_observe_probe.ipynb)
- [02 每请求 LoRA：Text-to-LoRA 式超网络](tutorials/02_request_adapters.ipynb)
- [03 decode 时训练：ESamp](tutorials/03_decode_training.ipynb)
- [04 改写 forward：循环深度](tutorials/04_custom_forward.ipynb)
- [05 反馈循环：TTRL](tutorials/05_host_feedback.ipynb)
- [06 读模型内部：残差流、逐层 logits 与 DoLa](tutorials/06_reading_inside.ipynb)
- [07 改写残差连接：value residual、超连接与记忆子层](tutorials/07_custom_residual.ipynb)
- [08 In-Place TTT：一边读一边改权重的模型](tutorials/08_inplace_ttt.ipynb)
- [09 在负载里用自己写的算子（Triton）](tutorials/09_custom_kernels.ipynb)
- [10 自定义 attention：监控、只读 top-K、改 KV 布局](tutorials/10_custom_attention.ipynb)
- [11 查表记忆：n-gram 先验与 Engram 式残差](tutorials/11_lookup_memory.ipynb)
- [12 自适应深度：让每个 token 自己决定跳过哪些层](tutorials/12_adaptive_depth.ipynb)
- [13 让 agent 搜索算法：一个可复现的打分循环](tutorials/13_algorithm_search.ipynb)

## 使用指南（任务导向）

- [能力地图：哪些推理时算法今天能用 pllm 写](guides/capability-map.md)
- [写一个负载：声明、时机、输入来源](guides/write-a-workload.md)
- [写代码时就看到成本：pllm.explain](guides/explain.md)
- [性能与显存指南：为什么几行算法会拖慢解码](guides/performance.md)
- [状态与记忆：作用域、冻结、route 键与 carry](guides/state-and-memory.md)
- [训练与观测：loss 约定、梯度边、导出与 trace](guides/training-and-observation.md)
- [Overlay 与重放：引擎内训练、外部导入、激活检查点](guides/overlays-and-replay.md)
- [用 agent 开发 pllm workload：agent skill](agent/SKILL.md)（配套[按任务组织的 API 参考](agent/reference.md)）
- [让代理写负载：输入与工作流](guides/agent-workloads.md)
- [检查自己写的负载](guides/validation.md)
- [基准解读](guides/benchmarking.md)

## 参考（信息导向）

- [负载创作契约](reference/authoring-contract.md) —— 全部能力边界与禁止事项
- [词汇表：一个术语一个含义](reference/glossary.md)

支持哪些模型、有哪些限制，见仓库首页的[支持的模型](../README.md#支持的模型)。
