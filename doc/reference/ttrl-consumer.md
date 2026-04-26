# TTRL-Inspired Consumer

This consumer is a tLLM-native rewrite of TTRL's lightweight test-time reward
idea. It does not import verl or the reference training stack.

## Supported Core

The first version implements two local pieces:

- majority-vote reward construction over multiple completions for the same
  prompt,
- upstream-style vote/sample separation: vote over `n_votes_per_prompt`
  completions, then keep the first `n_samples_per_prompt` indices per prompt
  for training,
- structured vote results with majority label, majority ratio, per-sample
  normalized label, and scalar reward,
- a small low-rank score adapter trained from hidden rows and scalar rewards.

Majority voting uses `normalize_answer_text`, which prefers the last boxed
answer before normalizing. This means different reasoning traces with the same
boxed answer vote together. Tie handling is deterministic: if labels tie, the
first normalized label seen for that prompt is treated as the winner.

## Usage

```python
from tllm.consumers.devlist import TTRLConsumer, TTRLConsumerConfig, TTRLSample
from tllm.consumers.devlist.ttrl import build_ttrl_reward_result

consumer = TTRLConsumer(TTRLConsumerConfig(hidden_size=896, rank=16))
samples = [
    TTRLSample(prompt_idx=0, sample_idx=0, completion=r"Trace A. \boxed{4}"),
    TTRLSample(prompt_idx=0, sample_idx=1, completion=r"Trace B. \boxed{4}"),
    TTRLSample(prompt_idx=0, sample_idx=2, completion=r"Trace C. \boxed{5}"),
]
rewards = consumer.rewards_for_samples(samples)
consumer.update_from_rewards(hidden_rows, rewards)

result = build_ttrl_reward_result(samples)
assert result.majority_by_prompt[0].label == "4"
assert result.majority_by_prompt[0].ratio == 2 / 3
assert result.rewards == [1.0, 1.0, 0.0]
```

For upstream-style vote/sample separation:

```python
from tllm.consumers.devlist.ttrl import select_ttrl_training_indices

indices = select_ttrl_training_indices(
    num_prompts=2,
    n_votes_per_prompt=4,
    n_samples_per_prompt=2,
)
assert indices == [0, 1, 4, 5]
```

## Counters

Use `read_stats()` to inspect:

- `update_count`
- `loss_count`
- `reward_count`
- `last_loss`

An enabled smoke run should report `reward_count > 0` and `update_count > 0`.
