# 检查自己写的负载

```python
import pllm
from pllm import testing
from pllm.workloads.esamp import ESampConfig, ESampWorkload

model_config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
workload = ESampWorkload(ESampConfig(rank=4, intervention_beta=0.5))
disabled_workload = ESampWorkload(ESampConfig(rank=4, intervention_beta=0.0))
result = testing.check(workload, model_config, neutral=disabled_workload)
print(result)
```

`neutral` 可省略。检查只在 CPU 上运行：读取模型配置，用固定的三个请求比较独立、合批、
倒序、其他请求输入扰动、混合 prefill/decode、分块 prompt、`publish_lag` 允许读到的版本，以及抢占后重算。它复用 runner 的行标签和
真实 planner/executor，比较方法输出、状态、采样干预和已交付的 export；浮点比较使用 FP32
`torch.testing.assert_close` 默认容差。checkpoint-backed state 还需要配置指向的本地权重。

失败抛出 `testing.ContractError`，列出所有差异，并区分方法混合请求行和框架错写目的行。
成功返回不可变的 `CheckResult(passed, skipped)`；`dataclasses.asdict(result)` 可转成 JSON。
共享训练后的依赖输出、export 的限行规则、host batch/feed 和需要模型的效果都有明确的
skip 原因。CPU 检查不执行模型里的实际应用（模型站点上的方法），这部分要在 GPU 上确认。
想知道负载把模型的输出分布推开了多少，用 `testing.check_distribution(workload, config, max_mean_kl=..., max_p99_kl=...)`：
它在 GPU 上以原模型的续写做 teacher forcing，给出两个方向的全词表 KL 与 TV（见
[agent 参考](../agent/reference.md#inspect-the-plan-and-check-contracts)）。

CPU 检查通过不等于 GPU 通过：模型里的实际应用、抢占与 CUDA graph 要用 `testing.check_on_gpu` 在 GPU 上确认，
报告结果时写明哪些检查在 CPU、哪些在 GPU 上跑过。
