# ESamp Reference

ESamp (*Large Language Models Explore by Latent Distilling*) is tLLM's reference consumer: a test-time training algorithm that runs **during** generation.

## What it does

1. **Capture**: on every decode step, read the first-layer (`source`) and final-layer (`target`) hidden states.
2. **Train**: a low-rank gated residual model learns to predict `target` from `source`, updated with AdamW as generation proceeds.
3. **Guide**: the sampler bridge uses the trained model's predictions to adjust candidate-token logits after top-k/top-p/min-p filtering (`--enable-distiller-intervention`, strength `--distiller-beta`).

## Two implementations

| `implementation=` | What it is |
|---|---|
| `"program"` (recommended) | `tllm/consumers/esamp/program.py` — the whole algorithm as ~80 lines of `@flow`/`@extern` declarations, compiled by `tllm.program`. Training runs on the training lane, predictions on the reader lane, and the whole train step is captured into a CUDA graph per batch shape. |
| `"legacy"` | The hand-tuned pipeline (dedicated train stream, per-slot staging events, Triton kernels). Kept as an explicit baseline for benchmarks. |

Select via config:

```python
from tllm.consumers.esamp import ESampConsumerConfig, activate
activate(ESampConsumerConfig(implementation="program"))
```

Both implementations expose the same stats surface (`loss_count`, `loss_avg`, intervention counters), so benchmark comparisons are apples-to-apples.

## Key config knobs (`ESampConsumerConfig`)

- `implementation`: `"program"` or `"legacy"`.
- `distiller_hidden_dim`, `distiller_lr`: model rank capacity and AdamW learning rate.
- `enable_distiller_intervention`, `distiller_beta`: sampler guidance on/off and strength.
- `distiller_sampler_backend`: candidate filtering backend (default `post_filter_exact`).
- `model_bank_train_cudagraph` (legacy only): whole-step CUDA graph capture in the legacy engine. The program path captures by itself; this flag does not apply to it.

## Benchmark

```bash
python -m tllm.workflows.benchmarks.per_request_esamp_benchmark \
  --case-filter single_on --implementation program \
  --benchmark-batch-size 8 --sampling-n 16 \
  --benchmark-max-new-tokens 256 --max-model-len 512 \
  --enable-distiller-intervention --emit-json-summary
```

The summary reports output tok/s **and** functional counters (`loss_count`, `distiller_intervention_state_count`, fallback counts). Treat any throughput number with zero training or zero interventions as a broken run, not a fast one.

## Measured numbers

- Legacy pipeline vs optimized vLLM baseline (Qwen2.5-7B, min-p active path, RTX 4090): **0.9878 ratio** (5304.9 vs 5370.6 tok/s).
- Program path host cost per train step (synthetic probe, 128×896 bf16): **0.082 ms vs 0.51 ms legacy eager**; whole-step CUDA graph replay ≈ 3–5 kernel launches vs ~70–90.
- The 5-round paired in-engine benchmark (gate ≥ 0.98 program/legacy) is scripted at `docs/p5_paired_benchmark_report.json` and pending GPU availability at time of writing.
