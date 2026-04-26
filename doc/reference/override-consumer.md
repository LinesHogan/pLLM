# OverRIDE-Inspired Consumer

This consumer is a tLLM-native rewrite of OverRIDE's core idea: iterative
low-rank logits reweighting.

It does not import the OverRIDE repository, vLLM patches, or OverRIDE's custom
model runner. It is intended to connect through tLLM's existing consumer and
sampler surfaces.

## Supported Core

`OverRIDEReweightEngine` keeps a sequence of low-rank heads. For iteration `i`,
it can:

- use the previous head to compute residual logits,
- return reweighted logits as `logits - lambd * residual`,
- compute `q_logits_next = logits + current_residual`,
- update the current head from sampled token ids with cross entropy.

The consumer exposes these operations directly through
`OverRIDEConsumer.reweight_logits()` and
`OverRIDEConsumer.update_from_sampled_tokens()`. It declares no
`ConsumerFlow`s because the implemented surface is logits/sampler side, not a
hidden-row bundle consumer.

## Runtime Bridge

`OverRIDEConsumer.reweight_logits_batch()` is the sampler-facing bridge for a
decode batch. It accepts either:

- `request_ids`, using the upstream OverRIDE convention where the integer before
  the first underscore is a zero-based processor index, so `"0_parent"` maps to
  iteration `1` and `"2_parent"` maps to iteration `3`; malformed or unprefixed
  ids map to iteration `1`.
- explicit one-based `iterations`, one value per row.

Rows in the same batch may use different iterations. The bridge groups rows by
iteration, calls the existing `reweight_logits()` path for each group, scatters
the reweighted logits back to the original row order, and caches each row's
`q_logits_next` when a current head exists. After sampling, call
`update_from_sampled_tokens_batch(sampled_token_ids)` with one sampled token per
batch row. The update uses the cached q-logits rows, so rows from different
iterations train their corresponding current heads without changing the
reweight/update formulas.

The bridge returns `OverRIDEBatchLogitsResult` with:

- `logits`: reweighted logits in original row order,
- `q_logits_next`: grouped q-logits tensors for rows that can train a current
  head,
- `q_indices`: original row indices for those grouped q-logits,
- `iterations`: the one-based iteration for each q-logits group.

## Dependency Policy

The implementation depends only on PyTorch and tLLM. Do not install OverRIDE's
requirements or patch vLLM to use this consumer. Future integration should use
the sampler bridge or other existing tLLM runtime hooks.

## Minimal Usage

```python
from tllm.consumers.devlist import OverRIDEConsumer, OverRIDEConsumerConfig

consumer = OverRIDEConsumer(
    OverRIDEConsumerConfig(hidden_size=896, vocab_size=151936, rank=16)
)

result = consumer.reweight_logits(hidden, logits, iteration=1)
consumer.update_from_sampled_tokens(result.q_logits_next, sampled_token_ids)
stats = consumer.read_stats()
```

Runtime bridge usage:

```python
result = consumer.reweight_logits_batch(
    hidden,
    logits,
    request_ids=("0_parent", "1_parent", "bad"),
)
sampled_token_ids = sampler(result.logits)
consumer.update_from_sampled_tokens_batch(sampled_token_ids)
```

Useful counters:

- `reweight_count`
- `modified_row_count`
- `loss_count`
- `update_count`
- `last_loss`

For throughput checks, treat an enabled run with `loss_count == 0` as a signal
that the update path did not execute.
