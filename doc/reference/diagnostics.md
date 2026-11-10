# explain 报告与运行时诊断字段

本页逐行、逐字段解释 `pllm.explain(workload, config)` 的报告和 `llm.diagnostics()` 的返回值。怎么用它们找问题，见
[看成本：pllm.explain](../guides/explain.md) 和 [让负载更快](../howto/faster.md)。

```python
import torch

import pllm


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


report = pllm.explain(StreamNorm(), pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8))
print(report.eager_reason == "", report.workload_host_syncs, [export.rows for export in report.exports])
```

## explain 的文本报告

`print(report)` 每行以一个词开头，这个词说明这一行讲什么。

| 行首 | 内容 |
| --- | --- |
| `model` | 模型类、hidden 宽度、层数、词表大小、精度，来自 `config.json`。只读模型另有一行 `model read-only: ...` 写明原因 |
| `parallel` | 张量并行卡数；后面所有预留都是每张卡的数 |
| `forward` | 模型 forward 是否在 CUDA graph 里：`CUDA graphs: no effect needs eager`，或 `eager: <原因>` |
| `base` | 底座模型每个 token 的稠密投影与 LM head FLOP 估计，不含对 KV 历史的 attention |
| `member` 表头之后的每一行 | 你的一个方法的一段计算：`方法名  效果  锚点 / lane / graph  读的状态 (origin, distance)`。一个方法可能有几段 |
| 锚点 | 这段计算在一步里的位置：`request.prepare`、`model.execute`、`state.publish`、`sampler.pre_draw`、`sampler.complete`、`token.commit`、`request.finish`、`next.prepare`（含义见[核心概念](../explain/core-concepts.md)） |
| lane | `main`：主 stream；`side`：旁路 stream，与 forward 重叠 |
| graph | `captured`：录进 CUDA graph；`eager`：逐算子执行 |
| `(origin, distance)` | 读的状态版本：`latest` 最新版本，`boundary` 上一个发布边界的版本，`entry` 步开始时的版本；distance 是离最新差几步 |
| `model site or inactive method` | 直接作用在模型里的声明（`Adapter`、`ActivationPatch`）或没有被安排的辅助方法，没有锚点 |
| `forward eager because <方法>` | 这个声明让模型 forward 退回 eager |
| `CUDA-graph pool reservation` | 这段计算的 CUDA graph 私有内存池预留 |
| `estimate / step (N rows)` | 按 `max_num_seqs` 行估计的每步 matmul 与投影 FLOP、状态读写字节（含优化器字段）、采样器字节 |
| `kernels` | 每步发射的算子数估计（融合前），`graph` 表示在 graph 里重放 |
| `tap` | 从模型里取出的 hidden 位置、哪些 token（decode / prefill / all）、缓冲字节和共用它的方法 |
| `state` 表头之后的每一行 | `状态名  作用域  roots  bytes/root  reserved bytes`，再列出每个字段的存储精度、初值来源（`source`）、放置位置（`placement`）、主机内存字节、每行和每步跨 PCIe 的字节。`roots` 是预留的份数，比并发请求多，因为新旧版本要同时存在 |
| `memory` | pLLM 自己要的显存峰值（状态页、任务工作区、CUDA graph 池等，不含权重和 KV）；第二行给出两个阈值：低于第一个，旁路任务改成主 stream 内联；低于第二个，规划失败 |
| `budget` | 你设的预算装不下时的报错原文（与 `make_llm` 相同）；报告仍给出不受预算限制的计划 |
| `preempt` | 请求被抢占后状态保留（`keep`）还是重置（`reset`），以及重置时为什么不能重算 |
| `host` | 你的数学里 GPU→CPU 同步的次数；模型内部的方法标 `unverified`，要上 GPU 用 `check_on_gpu` 确认 |
| `export` | 每个导出每步多少行、多少列、多少字节、拆成几个事件；行数超过 `max_rows` 时注明多出的行会被丢弃 |
| `timing` | 时机类设置的取值和后果，`(default)` 是默认值、`(you set)` 是你设的 |
| `op` | 自写算子：调用位置、每行 FLOP（有 `register_flop_formula` 时）、每行字节下界、有没有 CPU 参考实现和 autograd |
| `attention layer` | `@pllm.custom_attention` 每层的读法和每 token 的 FLOP、KV 字节估计 |
| `KV layer` | 静态 KV 布局下每层每个请求最多占多少 KV 块和字节 |
| `note` | 这份报告不知道或没计入的事（例如 GPU 上的 BLAS 工作区、模型权重与 KV） |

## ExplainResult 的字段

`pllm.explain` 返回 `ExplainResult`，`dataclasses.asdict(report)` 可以直接 `json.dumps`。常用字段：

| 字段 | 含义 |
| --- | --- |
| `eager_reason` | 模型 forward 为什么不在 CUDA graph 里；空字符串表示在 |
| `members` | 每个方法的效果、各段计算的锚点、lane、是否捕获、读的状态、估计 |
| `states` | 每个状态的作用域、`roots`、每份字节数、`reserved_bytes`、字段精度、初值来源、放置位置 |
| `taps` | 取出的 hidden 位置和缓冲字节 |
| `peak_memory_bytes`、`overlap_threshold_bytes`、`inline_peak_memory_bytes` | `memory` 行的三个数 |
| `budget_errors` | 预算装不下时的报错，空元组表示装得下 |
| `preemption`、`recompute_rejection` | `preempt` 行 |
| `workload_host_syncs` | 同步次数；`None` 表示无法在 CPU 上确认 |
| `exports` | 每个导出的行数、列数、字节、事件数 |
| `escape_hatches` | `timing` 行：设置、取值、是否默认、后果 |
| `custom_ops`、`attention`、`kv_capacity`、`kernels` | 对应的 `op`、`attention layer`、`KV layer`、`kernels` 行 |
| `limitations` | `note` 行 |
| `workload_digest` | 负载声明、方法代码和取值的摘要；相同即同一个负载 |
| `flops_per_token_estimate` | 底座加所有方法每 token 的 FLOP 估计；底座未知时为 `None` |

## llm.diagnostics() 的字段

引擎运行中随时可以调用；返回的是查询那一刻的值，pLLM 会先等设备上的计数写完。

| 字段 | 含义 |
| --- | --- |
| `eager_reason` | 同 explain，空字符串表示模型在 CUDA graph 里 |
| `loss_count` | 训练方法运行的次数 |
| `update_count` | 实际发布的优化器更新次数 |
| `trained_rows` | 参与训练的行数累计 |
| `active_states` | 当前存活的状态副本数 |
| `state_usage` | 状态显存的快照，见下表 |
| `numeric_health` | 每个存活的已发布字段的 L2 范数、最大绝对值和非有限值个数（`norm`、`max_abs`、`nonfinite_count`）；范数或最大值不是有限数时为 `None`。只在查询时计算，不增加每步的同步；字段退役后不保留历史 |
| `kv_cache_blocks`、`kv_cache_max_concurrency` | vLLM 实际分配的 KV 块数，以及按 `max_model_len` 算能同时容纳的请求数（每张卡） |
| `hold_count`、`last_hold` | 带 `on_missing="hold"` 的激活补丁沿用旧输入的次数和最近一次的情况 |
| `inference_pins`、`completed_sampler_pins`、`pending_state_releases` | 框架调试用的内部计数 |

`state_usage` 的字段：

| 字段 | 含义 |
| --- | --- |
| `reserved_bytes` | 为状态预留的显存 |
| `logical_live_bytes` | 所有存活版本的字段字节之和；新旧版本共用的字段会被重复计入 |
| `allocated_payload_bytes` | 实际占用的字段字节；共用的字段只计一次，所以可能远小于 `logical_live_bytes` |
| `reserved_padding_bytes` | 为对齐填充的字节 |
| `free_field_pages`、`live_field_pages`、`private_field_pages`、`retiring_field_pages` | 字段页的使用情况：空闲、已发布、正在写、等待回收 |
| `private_versions`、`retiring_versions`、`pinned_versions` | 正在写、等待回收、被读者占用的版本数（另有一个等待设备完成的发布计数，供框架调试） |
| `admission_capacity_failures` | 因为状态容量用完而被拒绝准入的请求数 |

## 导出事件

`on_export` 收到的 `pllm.io.TraceEvent` 中用户常读的字段：`tensor_name`（导出名；组件的导出去掉了字段名前缀）、
`tensor()`（独立的 CPU 张量）、`identities`（每行一个 `pllm.testing.RequestIdentity`：`prompt_index`、`sample_index`、
pLLM 内部的 `request_id`）、`positions`（每行在请求中的 token 位置）、`step_id`、`dropped_rows`（这一步因 `max_rows`
被丢弃的行数）。`llm.trace_snapshot()` 返回 `pllm.io.TraceSnapshot`：保留的事件 `events`、缓冲满时丢掉的最旧事件数 `dropped`，
以及导出投递队列满时丢掉的事件数 `export_dropped`。
