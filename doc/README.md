# 文档

pLLM 的文档按用途分成五类，每一页只做一件事：**教程**带你从零学会写负载；**用户指南**按功能讲清每个声明怎么用、代价是什么；
**操作指南**是解决一个具体问题的步骤；**参考**供逐条查证；**解释**讲规则背后的原因。另有一个**示例库**，每篇实现一个算法族、
附带实测。

```text
第一次接触         → 教程 00，然后沿主线 01–05
想写自己的负载     → 用户指南：负载的组成，再按需要查各专题
遇到具体问题       → 操作指南（报错、加载权重、上生产、测吞吐……）
想知道某类算法能不能写 → 能力地图；为什么写不了 → 设计取舍
查字段、配置、报错原文 → 参考
```

改 pLLM 框架本身的贡献者文档在 [dev/](dev/README.md)；让编码 agent 写负载，把 [agent skill](agent/SKILL.md) 和
[agent 参考](agent/reference.md)（英文，含全部公开 API 的签名）交给它。

## 教程

可执行的 notebook，需要 GPU，输出保留在 notebook 里。大多在 Qwen2.5-0.5B-Instruct 上跑；测吞吐的 03 用 Qwen2.5-7B-Instruct（bf16 权重约 15 GiB，需要一张 24 GiB 的卡）。每篇从一个具体算法出发，结尾有常见错误和自测题。

| 教程 | 算法 | 你会学到 |
| --- | --- | --- |
| [00 快速上手](tutorials/00_quickstart.ipynb) | 激活引导（CAA） | 负载的三个问题：读什么、存什么、结果去哪；启动前用 `pllm.explain` 和 `pllm.testing.check` 检查 |
| [01 观测](tutorials/01_observe_probe.ipynb) | 线性探针 / SAE 监控 | 逐 token 读 hidden，导出到宿主机；不改变输出、不关 CUDA graph |
| [02 每请求 LoRA](tutorials/02_request_adapters.ipynb) | Text-to-LoRA 式超网络 | `RequestState`、`Adapter`、`@pllm.write`：每个请求一份由 prompt 生成的 LoRA |
| [03 decode 时训练](tutorials/03_decode_training.ipynb) | ESamp | `@pllm.train`、`@pllm.guide_sampling`、`publish_lag`；测吞吐 |
| [04 改写 forward](tutorials/04_custom_forward.ipynb) | 循环深度（Ouro / Huginn 式） | `@pllm.custom_forward`；循环里 KV 怎么处理 |
| [05 反馈循环](tutorials/05_host_feedback.ipynb) | TTRL / GRPO | 宿主批训练：生成、投票、训练在同一个引擎里闭环 |

## 用户指南

按功能组织，每页先说是什么、何时用，再给可运行的短例子，然后讲代价和限制的原因。

- [能力地图](guides/capability-map.md)：哪些推理时算法今天能用 pLLM 写、用什么写、现成的负载
- [负载的组成](guides/workload-anatomy.md)：三类声明、方法体的规则、方法何时运行、选择器、开发循环
- [读取模型内部](guides/reading.md)：hidden、残差流、逐层 logits、token 历史、采样结果
- [状态](guides/state.md)：请求状态与实例状态、字段、初值来源、精度、carry 字段
- [结果去哪](guides/results.md)：导出、logits 加法、采样引导、写状态
- [在线训练](guides/training.md)：`@pllm.train`、归约、选择器、`publish_lag`、梯度边
- [改模型](guides/model-edits.md)：激活补丁、Adapter、前向内训练
- [改 forward 结构](guides/forward-structure.md)：定制残差、定制 attention 与 KV 布局、custom forward
- [宿主循环与批训练](guides/host-loops.md)：宿主批次、因果模型、`make_host`、`generate_feedback`
- [自写算子](guides/custom-ops.md)：用 `torch.library.custom_op` 接入 Triton / CUDA kernel
- [组合负载](guides/composition.md)：把几个负载放进一个引擎
- [看成本：pllm.explain](guides/explain.md)：启动前在 CPU 上读执行计划
- [检查正确性](guides/checking.md)：`testing.check`、`check_on_gpu`、`check_distribution`

## 操作指南

- [排查报错](howto/troubleshooting.md)
- [加载训练好的权重](howto/load-trained-weights.md)
- [让一组请求共享训练成果](howto/share-training-across-requests.md)
- [上生产：多进程、张量并行与量化底座](howto/production.md)
- [公平地测吞吐](howto/measure-throughput.md)
- [让负载更快](howto/faster.md)
- [在新模型家族上使用](howto/new-model-families.md)

## 参考

- [支持矩阵](reference/support-matrix.md)：模型 × 功能 × vLLM 版本 × 并行方式
- [组合规则](reference/composition-rules.md)：哪些写法可以并存、哪些被拒绝及原因
- [配置项](reference/config.md)：`ModelConfig`、`ExecutionOptions` 等的全部字段
- [explain 报告与运行时诊断字段](reference/diagnostics.md)
- [报错目录](reference/errors.md)：报错原文、原因与改法
- [术语表](reference/glossary.md)
- API 签名：[agent 参考的签名列表](agent/reference.md#all-signatures)（由代码生成）

## 解释

- [核心概念](explain/core-concepts.md)：为什么不是 hook、行、一步的时间线、状态版本与 `publish_lag`、为什么有边界
- [设计取舍](explain/design-tradeoffs.md)：为什么有些算法今天写不了，以及替代做法
- [性能与显存](explain/performance.md)：decode 的瓶颈、pLLM 替你省了什么、各写法的实测代价

## 示例库

每篇实现一个算法族，按"问题 → 负载 → explain → check → 生成 → 测量 → 代价与边界"展开；实测数字都注明了运行环境。
吞吐、显存这类性能数字都在 Qwen2.5-7B-Instruct 上测（RTX 4090，vLLM 0.32.0）：07、12 全程用 7B，06、09、10、11 先在 0.5B 上看行为、再在 7B 上测吞吐；08 用一个 4B 的 In-Place TTT 社区检查点；13 的搜索在 0.5B 上进行，需要 bubblewrap 和已登录的 agent 命令行。

| 示例 | 算法 | 重点 |
| --- | --- | --- |
| [06 读模型内部](examples/06_reading_inside.ipynb) | logit lens / DoLa | 残差流和逐层 logits 留在 CUDA graph 里；DoLa 的计算代价 |
| [07 改写残差连接](examples/07_custom_residual.ipynb) | value residual / 超连接 / 记忆子层 | `@pllm.custom_residual`；恒等时与 eager 原生逐位相同；吞吐代价 |
| [08 In-Place TTT](examples/08_inplace_ttt.ipynb) | In-Place TTT（社区 4B 检查点） | 按块更新的快权重；decode 中跨块更新；与官方实现比吞吐 |
| [09 自写算子](examples/09_custom_kernels.ipynb) | 相对引导（Triton kernel） | `torch.library.custom_op`；explain 的算子行；`check_on_gpu` 的设备等价检查 |
| [10 自定义 attention](examples/10_custom_attention.ipynb) | 注意力监控 / Quest / KV 布局 | 只换读法；只读 top-K；滑动窗口、prompt 加窗口、跨层共享各省多少 KV |
| [11 查表记忆](examples/11_lookup_memory.ipynb) | n-gram 先验 / Engram 式残差 | `pllm.input_tokens`；`pllm.Pretrained` 读表；表放显存还是主机内存 |
| [12 自适应深度](examples/12_adaptive_depth.ipynb) | 按 token 跳层（early exit） | `active=`；跳过的层怎么补 KV；什么时候真的省 |
| [13 让 agent 搜索算法](examples/13_algorithm_search.ipynb) | agent 驱动的搜索循环 | explain + check + 打分 + 重放；怎样判断提升是不是噪声 |
