# TLM-Inspired Consumer

This consumer is a tLLM-native rewrite of Test-Time Learning's online/offline
TTL pattern. It maintains a local low-rank adapter bank and selects samples for
adaptation from sentence-level cross entropy.

It does not import LLaMA-Factory or TLM training dependencies.

## Supported Core

`TLMConsumer` supports:

- `online` mode: selected samples update the adapter immediately,
- `offline` mode: samples are buffered and updated on `flush_offline()`,
- CE-threshold sample selection,
- weighted one-hot KL/CE training with `lamb * exp(CE - threshold)`,
- hidden-to-vocab low-rank adapter logits when hidden/logits/labels are
  available,
- offline base-logits selection when `base_logits` is supplied,
- explicit online/offline ordering through `run_tlm_ordered_step()`,
- a closed-loop helper, `run_lora_learning_step()`, that runs one
  predict/train step and stages a strict `TestTimeLoRAUpdate`,
- disabled no-op behavior.

Rows with raw sentence CE strictly above `threshold` are selected for
adaptation. The logits path trains the first adapter slot as a hidden-to-vocab
logits delta and applies the shared weighted one-hot KL/CE helper to
`logits + adapter(hidden)`.

Online updates compute selection CE from the current logits by default. Offline
updates can pass `base_logits`; those base logits determine which rows are
selected, while the adapter is trained against the provided training logits.

The former scalar-score update remains available only as an explicit fallback:
call `update_online(..., scores, allow_scalar_fallback=True)` or
`update_offline(..., scores, allow_scalar_fallback=True)`. That fallback keeps
the legacy low-score selection and MSE objective for smoke compatibility.

`run_tlm_ordered_step()` exposes the upstream loop ordering:

- online TTL: predict, then train,
- offline TTL: train, then predict.

`run_lora_learning_step()` applies the same ordering while packaging the local
adapter as a request-scoped or global test-time LoRA update. After staging the
update it calls an explicit `apply_lora_update` callback when supplied, or the
service's `commit_staged_updates()` method when available. Offline callers that
need the next `predict()` to see the updated adapter should provide one of those
barriers. If the service exposes target modules with ordinary 2D `weight`
tensors, the helper checks the exported LoRA tensor shape before staging.

The caller must provide already-aligned hidden/logits/labels tensors and an
explicit runtime target module name. For raw causal-LM tensors, shift logits to
`[:, :-1, :]` and labels to `[:, 1:]` before calling the helper.

The helper deliberately does not compute base logits by disabling runtime LoRA
inside vLLM. Offline callers that need upstream-style base-model CE selection
should pass `base_logits` that were computed before the adapter was applied.

## Counters

Use `read_stats()` to inspect:

- `update_count`
- `loss_count`
- `selected_count`
- `buffered_count`
- `last_loss`

An enabled smoke run should report `selected_count > 0` and `update_count > 0`.

## Guarded Smoke

`test/consumers/tlm/test_tlm_cached_model_smoke.py` contains a local-files-only
cached-model smoke. It is skipped by default and runs only when
`TLLM_TLM_RUN_CACHED_MODEL_SMOKE=1` is set. The smoke loads a cached tiny causal
LM, shifts raw logits/labels into the TTL alignment contract, runs one
closed-loop TLM step, and verifies that a LoRA update is staged.
