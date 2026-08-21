# ESamp 参考

ESamp（*Large Language Models Explore by Latent Distilling*）是 tLLM 的参考 consumer：一个在**生成过程中**运行的 test-time training 算法。

## 它做什么

1. **捕获**：每个 decode step 读取第一层（`source`）与最后一层（`target`）hidden state。
2. **训练**：一个低秩门控残差模型学习从 `source` 预测 `target`，随生成用 AdamW 持续更新。
3. **引导**：sampler bridge 用训练好的模型预测调整候选 token 的 logits，在 top-k/top-p/min-p 过滤之后应用（`--enable-distiller-intervention`，强度 `--distiller-beta`）。

## 两个实现

| `implementation=` | 是什么 |
|---|---|
| `"program"`（推荐） | `tllm/consumers/esamp/program.py` —— 整个算法约 80 行 `@flow`/`@extern` 声明，由 `tllm.program` 编译。训练走训练 lane，预测走 reader lane，整个训练步按 batch 形状捕获为 CUDA Graph。 |
| `"legacy"` | 手工调优管线（专用 train stream、per-slot 事件、Triton kernel）。保留为 benchmark 显式基线。 |

配置选择：

```python
from tllm.consumers.esamp import ESampConsumerConfig, activate
activate(ESampConsumerConfig(implementation="program"))
```

两个实现暴露同一统计面（`loss_count`、`loss_avg`、干预计数），benchmark 对比口径一致。

## 主要配置项（`ESampConsumerConfig`）

- `implementation`：`"program"` 或 `"legacy"`。
- `distiller_hidden_dim`、`distiller_lr`：模型 rank 容量与 AdamW 学习率。
- `enable_distiller_intervention`、`distiller_beta`：sampler 引导开关与强度。
- `distiller_sampler_backend`：候选过滤后端（默认 `post_filter_exact`）。
- `model_bank_train_cudagraph`（仅 legacy）：legacy 引擎的整步 CUDA Graph。program 路径自行捕获，此开关不适用。

## Benchmark

```bash
python -m tllm.workflows.benchmarks.per_request_esamp_benchmark \
  --case-filter single_on --implementation program \
  --benchmark-batch-size 8 --sampling-n 16 \
  --benchmark-max-new-tokens 256 --max-model-len 512 \
  --enable-distiller-intervention --emit-json-summary
```

摘要在输出 tok/s **之外**还报告功能计数（`loss_count`、`distiller_intervention_state_count`、fallback 计数）。任何训练数或干预数为零的吞吐数字都应视为坏掉的运行，而不是快的运行。

## 实测数字

- legacy 管线 vs 优化 vLLM 基线（Qwen2.5-7B，min-p 活跃路径，RTX 4090）：**比值 0.9878**（5304.9 vs 5370.6 tok/s）。
- program 路径单训练步 host 开销（合成探针，128×896 bf16）：**0.082 ms，legacy eager 为 0.51 ms**；整步 CUDA Graph replay ≈ 3–5 次 kernel launch，对比约 70–90 次。
- 5 轮配对 in-engine benchmark（门禁 program/legacy ≥ 0.98）脚本已备好，结果写入 `docs/p5_paired_benchmark_report.json`，撰写本文时等待 GPU 空出。
