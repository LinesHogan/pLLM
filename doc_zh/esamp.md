# ESamp 参考

ESamp（*Large Language Models Explore by Latent Distilling*）是 tLLM 的参考 consumer：一个在**生成过程中**运行的 test-time training 算法。

## 它做什么

1. **捕获**：每个 decode step 读取第一层（`source`）与最后一层（`target`）hidden state。
2. **训练**：一个低秩门控残差模型学习从 `source` 预测 `target`，随生成用 AdamW 持续更新。
3. **引导**：sampler bridge 用训练好的模型预测调整候选 token 的 logits，在 top-k/top-p/min-p 过滤之后应用（`--enable-distiller-intervention`，强度 `--distiller-beta`）。

## 实现与验证边界

| `implementation=` | 是什么 |
|---|---|
| `"native"` | native exact eager workload。Rust 签名编译后的 task 契约，Python 只机械绑定；不使用 legacy ESamp authority。已通过 MRv1/MRv2 的 in-process 与 EngineCore MP 正确性，loss/update/state-version/intervention 非零，fallback/replay 为零。这不是吞吐或 CUDA Graph 结论。 |
| `"program"` | 保留的 Program 迁移兼容 adapter：`tllm/consumers/esamp/program.py` 由 `@flow`/`@extern` 声明编译，使用 legacy lane 模型。 |
| `"legacy"` | 手工调优管线（专用 train stream、per-slot 事件、Triton kernel）。保留为 benchmark 显式基线。 |

配置选择：

```python
from tllm.consumers.esamp import ESampConsumerConfig, build_esamp_workload
from tllm.runtime import TllmRuntime

workload = build_esamp_workload(ESampConsumerConfig(implementation="native"))
runtime = TllmRuntime(workloads=(workload,))
```

`"native"` 会拒绝 legacy-only 的 per-request/model-bank 模式、compact capture lane、row
selection 与非 `post_filter_exact` sampler。model-backed native piecewise CUDA Graph 路径仍未验证
且未完成；native ESamp 的证据仅为 eager 正确性。Program 与 legacy 仍是迁移/benchmark 路径，
不是 native workload 吞吐证据。

## 主要配置项（`ESampConsumerConfig`）

- `implementation`：`"native"`、`"program"` 或 `"legacy"`。
- `distiller_hidden_dim`、`distiller_lr`：模型 rank 容量与 AdamW 学习率。
- `enable_distiller_intervention`、`distiller_beta`：sampler 引导开关与强度。
- `distiller_sampler_backend`：候选过滤后端（默认 `post_filter_exact`）。
- `model_bank_train_cudagraph`（仅 legacy）：legacy 引擎的整步 CUDA Graph；不适用于 native eager workload 路径。

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
