# ESamp Reference

ESamp (*Large Language Models Explore by Latent Distilling*) is tLLM's reference consumer: a test-time training algorithm that runs **during** generation.

## What it does

1. **Capture**: on every decode step, read the first-layer (`source`) and final-layer (`target`) hidden states.
2. **Train**: a low-rank gated residual model learns to predict `target` from `source`, updated with AdamW as generation proceeds.
3. **Guide**: the sampler bridge uses the trained model's predictions to adjust candidate-token logits after top-k/top-p/min-p filtering (`--enable-distiller-intervention`, strength `--distiller-beta`).

## Implementations and verification boundary

| `implementation=` | What it is |
|---|---|
| `"native"` | Native exact eager workload. Rust signs its compiled task contract and Python mechanically binds it; it does not use a legacy ESamp authority. Correctness passed in-process and EngineCore MP on MRv1/MRv2, with nonzero loss/update/state-version/intervention and zero fallback/replay counts. This is not a throughput or CUDA Graph result. |
| `"program"` | Retained Program compatibility adapter: `tllm/consumers/esamp/program.py` is compiled from `@flow`/`@extern` declarations and uses the legacy lane model. |
| `"legacy"` | The hand-tuned pipeline (dedicated train stream, per-slot staging events, Triton kernels). Kept as an explicit baseline for benchmarks. |

Select via config:

```python
from tllm.consumers.esamp import ESampConsumerConfig, build_esamp_workload
from tllm.runtime import TllmRuntime

workload = build_esamp_workload(ESampConsumerConfig(implementation="native"))
runtime = TllmRuntime(workloads=(workload,))
```

`"native"` supports both replica state and eager request-scoped state through
`per_request_models=True`. Request state is keyed by the stable parent request
ID, so parallel `SamplingParams(n=...)` children share one slot, loss,
optimizer update, and version transition. The first request-state tranche does
not yet support sampler intervention, `first_per_prompt` row selection, model
bank state, or disabling the output layer norm; those combinations fail at
construction. Compact capture lanes and non-`post_filter_exact` samplers also
remain legacy-only.

The request-scoped path passed MRv1/MRv2 × in-process/EngineCore-MP eager GPU
correctness with two parents, two children per parent, and two decode updates:
every cell reported four losses and four updates, retired two parent slots at
version 2, cleared live slots and request mappings, and kept fallback/replay at
zero. Its signed request-state Regions are explicitly capture-ineligible, so
this is not CUDA Graph or throughput evidence. Program and legacy remain
migration/benchmark paths until the complete deletion gate passes.

## Key config knobs (`ESampConsumerConfig`)

- `implementation`: `"native"`, `"program"`, or `"legacy"`.
- `distiller_hidden_dim`, `distiller_lr`: model rank capacity and AdamW learning rate.
- `enable_distiller_intervention`, `distiller_beta`: sampler guidance on/off and strength.
- `distiller_sampler_backend`: candidate filtering backend (default `post_filter_exact`).
- `model_bank_train_cudagraph` (legacy only): whole-step CUDA graph capture in the legacy engine. It does not apply to the native eager workload path.

## Benchmark

```bash
python -m tllm.workflows.benchmarks.per_request_esamp_benchmark \
  --case-filter single_on --implementation program \
  --benchmark-batch-size 8 --sampling-n 16 \
  --benchmark-max-new-tokens 256 --max-model-len 512 \
  --enable-distiller-intervention --emit-json-summary
```

The summary reports output tok/s **and** functional counters (`loss_count`, `distiller_intervention_state_count`, fallback counts). Treat any throughput number with zero training or zero interventions as a broken run, not a fast one.
Use the exact alignment and memory preconditions in
[`guides/benchmarking.md`](guides/benchmarking.md), and do not retire either
baseline before the gate in [`guides/validation.md`](guides/validation.md).

## Measured numbers

- Legacy pipeline vs optimized vLLM baseline (Qwen2.5-7B, min-p active path, RTX 4090): **0.9878 ratio** (5304.9 vs 5370.6 tok/s).
- Program path host cost per train step (synthetic probe, 128×896 bf16): **0.082 ms vs 0.51 ms legacy eager**; whole-step CUDA graph replay ≈ 3–5 kernel launches vs ~70–90.
- The recorded 5-round Program/legacy report at
  `docs/p5_paired_benchmark_report.json` has median ratio **0.7968** and
  `gate_pass=false`. It is not native throughput evidence and explicitly keeps
  deletion blocked.
