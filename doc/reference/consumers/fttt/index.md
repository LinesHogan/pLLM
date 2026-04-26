# FTTT-Inspired Consumer

This consumer is a tLLM-native rewrite of Feedback-based Test-Time Training's
core edit loop. It consumes explicit feedback records and trains a local
low-rank edit adapter from hidden rows, scalar rewards, and hashed feedback
text features.

It does not call external reward-model APIs, code-execution services, or
reference repository modules.

## Supported Core

`FTTTConsumer` supports:

- bounded feedback enqueueing,
- explicit feedback draining,
- hidden-row plus reward/text-conditioned updates,
- hidden and logits edit application,
- per-instance retry accounting,
- stop-on-correct behavior with per-instance reset,
- disabled no-op behavior.

The first version expects callers to provide feedback and rewards. It does not
judge completions by itself.

## Feedback-Conditioned Edits

`apply_feedback_update(hidden, feedback)` now trains an edit adapter that emits
a hidden-space delta. The adapter input concatenates:

- the hidden row,
- deterministic feature-hashed feedback text,
- the scalar reward.

The adapter can be queried without another training step:

- `edit_delta(hidden, feedback)` returns the learned hidden delta.
- `apply_edit_to_hidden(hidden, feedback)` returns `hidden + delta`.
- `apply_edit_to_logits(logits, hidden, feedback)` adds the delta directly when
  logits and hidden share the same width.
- `apply_edit_to_logits(..., output_weight=lm_head_weight)` projects the hidden
  delta through an LM-head-style weight before adding it to logits.

The implementation is intentionally local and lightweight: feedback text affects
the edit through stable text feature hashing, not through an upstream tokenizer,
reward model, or external service.

## Instance State

Instances are keyed by `FeedbackRecord.prompt_idx`.

Incorrect or negative feedback increments that instance's retry count and can
train the edit adapter. Positive feedback stops the instance from future
adaptation. Positive feedback is detected when `reward >=
positive_reward_threshold` or when the feedback text contains a positive marker
such as `correct` without a negative marker such as `incorrect`.

Use:

- `instance_state(prompt_idx)` to inspect `retries` and `stopped`.
- `reset_instance(prompt_idx)` to clear stop/retry state and allow adaptation to
  resume for that instance.

## Counters

Use `read_stats()` to inspect:

- `queued_feedback`
- `dropped_feedback`
- `feedback_count`
- `update_count`
- `loss_count`
- `last_loss`
- `skipped_feedback`
- `stopped_instances`

An enabled smoke run should report `feedback_count > 0` and `update_count > 0`.
