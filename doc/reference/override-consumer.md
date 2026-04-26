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

The first version exposes these operations directly through
`OverRIDEConsumer.reweight_logits()` and
`OverRIDEConsumer.update_from_sampled_tokens()`. It declares no
`ConsumerFlow`s because the implemented surface is logits/sampler side, not a
hidden-row bundle consumer.

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

Useful counters:

- `reweight_count`
- `modified_row_count`
- `loss_count`
- `update_count`
- `last_loss`

For throughput checks, treat an enabled run with `loss_count == 0` as a signal
that the update path did not execute.
