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
from tllm.consumers.esamp import ESampConfig, ESampWorkload
from tllm.runtime import TllmRuntime

workload = ESampWorkload(ESampConfig())
runtime = TllmRuntime(workloads=(workload,))
```

`"native"` supports both replica state and eager request-scoped state through
`per_request_models=True`. Request state is keyed by the stable parent request
ID, so parallel `SamplingParams(n=...)` children share one slot, loss,
optimizer update, and version transition. Request-state sampler intervention
uses a generic typed compact-row payload: full step identity remains
authoritative while only slots whose frozen before-version is positive publish
hidden predictions. Version-zero rows therefore cannot receive fabricated
guidance.

PTTS-0 now also provides a signed `fixed_batched` request-state layout. Native
ESamp can map the compatibility `per_request_model_bank/model_bank_slots`
inputs to one capacity-first bank module and one optimizer whose moment and
step tensors are independently indexed by logical slot. Admission restores
only newly bound rows from the cold snapshot; transition updates only the
selection's unique slots; retire/reuse remains owned by the same
`RequestSlotArena`. The Rust scheduler receives `fixed_batched` only as
storage evidence and still derives these packed request Regions as eager.
There is no ESamp lane, priority, flush, pipeline, or CUDA-Graph policy.

The SVD initializer, compact capture, non-`post_filter_exact` samplers, GPU
model-bank differential, and aligned throughput gate remain open.

The request-scoped path passed MRv1/MRv2 × in-process/EngineCore-MP eager GPU
correctness with two parents, two children per parent, and two decode updates:
every cell reported four losses and four updates, retired two parent slots at
version 2, cleared live slots and request mappings, and kept fallback/replay at
zero. With intervention enabled, each request case produced exactly one
modifier state: prefill and the first version-zero decode step cannot consume
guidance, while the final decode step consumes version-one guidance.

The retained legacy and native arms in
`test/gpu/native_esamp_request_state_smoke.py` now run in independent
processes from one signed case fingerprint. On MRv2 in-process with fixed
sampling RNG, both intervention-off and `beta=0.5` runs produced identical
tokens, four losses/updates, one intervention state when enabled, and zero
sampler fallback. Their aggregate losses match exactly after the
capture-disabled request optimizer was aligned to eager non-fused AdamW.
An earlier six-versus-four report was invalid: its legacy probe passed a
`TokensPrompt` through a text-only helper, which stringified the object and
changed the prompt/step trace. The corrected workload is decode-only on both
implementations.

```bash
.venv/bin/python test/gpu/native_esamp_request_state_smoke.py \
  --implementation legacy --runner mrv2 --intervention --out legacy.json
.venv/bin/python test/gpu/native_esamp_request_state_smoke.py \
  --implementation native --runner mrv2 --intervention --out native.json
```

The two JSON documents must carry the same `case_signature`; compare their
`output_token_ids`, `loss_count`, `loss_avg`, `update_count`, intervention, and
fallback fields with:

```bash
.venv/bin/python test/gpu/native_esamp_request_state_smoke.py \
  --compare legacy.json native.json
```

The comparator is green for both MRv1 and MRv2 in-process. MRv1 aggregate
loss differs by `2.38e-7`, within the declared `rtol=1e-5, atol=1e-6`; MRv2
is exact. The MP differential is not green: after clearing initialization
statistics, the retained MRv1 legacy child performs two intervention states
for a two-decode request instead of one. A current native MP rerun is also
environment-blocked before workload startup by a `torchcodec 0.16.0+cu129`
spawn-child FFmpeg-op import mismatch. Neither blocker is being hidden by a
relaxed comparator.

Native request cleanup remains the authoritative target: it retires both
slots at version 2 and clears request mappings. The retained legacy baseline
has no equivalent request-finish transaction and reports its two prompt-keyed
trainers/mappings as still live. This legacy lifetime limitation is recorded,
not copied into native. Program and legacy remain present because the fixed
bank has not yet passed the GPU model-bank differential, SVD initialization,
compact-capture, CUDA Graph, aligned throughput, and the rest of the deletion
gate.

The neutral packed-row contract preserves Rust-planned prefill/decode rows,
stable parent identities, resource-keyed hidden snapshots, and explicit
`logits_indices` sampler links across MRv1/MRv2. ESamp consumes those signed
inputs directly. Its request-state selection additionally signs
`phase=decode`; the runtime projects only decode identities while retaining
their original packed tensor indices. A non-empty pure-prefill batch therefore
skips the stateful Region without admitting or updating a slot. This is a
generic packed state-selection facility, not an ESamp scheduling rule.

## Key config knobs (`ESampConsumerConfig`)

- `implementation`: `"native"`, `"program"`, or `"legacy"`.
- `distiller_hidden_dim`, `distiller_lr`: model rank capacity and AdamW learning rate.
- `per_request_models`, `per_request_model_bank`: select per-owner or
  model-bank algorithm semantics. Native maps the latter to a typed
  capacity-first layout; adapter and scheduling decisions remain structural.
- `model_bank_slots`, `model_bank_rank`, `model_bank_use_output_layernorm`:
  fixed-bank layout inputs now accepted by the native builder.
- `enable_distiller_intervention`, `distiller_beta`: sampler guidance on/off and strength.
- `distiller_sampler_backend`: candidate filtering backend (default `post_filter_exact`).
- `model_bank_train_cudagraph` (legacy only): whole-step CUDA graph capture in the legacy engine. It does not apply to the native eager workload path.

## Benchmark

```bash
python test/gpu/esamp_ratio_benchmark.py run --mode native --runner mrv2 --out native.json
```

The ratio harness reports output tok/s **and** functional counters (`loss_count`, intervention counts, fallback counts). Treat any throughput number with zero training or zero interventions as a broken run, not a fast one.
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
