# TLM-Inspired Consumer

This consumer is a tLLM-native rewrite of Test-Time Learning's online/offline
TTL pattern. It maintains a local low-rank adapter bank and selects samples for
adaptation from scalar confidence or quality scores.

It does not import LLaMA-Factory or TLM training dependencies.

## Supported Core

`TLMConsumer` supports:

- `online` mode: selected samples update the adapter immediately,
- `offline` mode: samples are buffered and updated on `flush_offline()`,
- threshold-based sample selection,
- disabled no-op behavior.

Scores below `threshold` are selected for adaptation. The local objective trains
the first adapter slot toward `1 - score` for selected hidden rows.

## Counters

Use `read_stats()` to inspect:

- `update_count`
- `loss_count`
- `selected_count`
- `buffered_count`
- `last_loss`

An enabled smoke run should report `selected_count > 0` and `update_count > 0`.
