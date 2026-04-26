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

## Counters

Use `read_stats()` to inspect:

- `update_count`
- `loss_count`
- `selected_count`
- `buffered_count`
- `last_loss`

An enabled smoke run should report `selected_count > 0` and `update_count > 0`.
