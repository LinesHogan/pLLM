# TTRL-Inspired Consumer

This consumer is a tLLM-native rewrite of TTRL's lightweight test-time reward
idea. It does not import verl or the reference training stack.

## Supported Core

The first version implements two local pieces:

- majority-vote reward construction over multiple completions for the same
  prompt,
- a small low-rank score adapter trained from hidden rows and scalar rewards.

Tie handling is deterministic: if completions tie, the first normalized
completion seen for that prompt is treated as the winner.

## Usage

```python
from tllm.consumers.devlist import TTRLConsumer, TTRLConsumerConfig, TTRLSample

consumer = TTRLConsumer(TTRLConsumerConfig(hidden_size=896, rank=16))
samples = [
    TTRLSample(prompt_idx=0, sample_idx=0, completion="answer"),
    TTRLSample(prompt_idx=0, sample_idx=1, completion="answer"),
]
rewards = consumer.rewards_for_samples(samples)
consumer.update_from_rewards(hidden_rows, rewards)
```

## Counters

Use `read_stats()` to inspect:

- `update_count`
- `loss_count`
- `reward_count`
- `last_loss`

An enabled smoke run should report `reward_count > 0` and `update_count > 0`.
