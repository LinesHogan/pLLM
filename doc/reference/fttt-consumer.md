# FTTT-Inspired Consumer

This consumer is a tLLM-native rewrite of Feedback-based Test-Time Training's
core edit loop. It consumes explicit feedback records and trains a local
low-rank edit adapter from hidden rows and scalar rewards.

It does not call external reward-model APIs, code-execution services, or
reference repository modules.

## Supported Core

`FTTTConsumer` supports:

- bounded feedback enqueueing,
- explicit feedback draining,
- hidden-row plus reward updates,
- disabled no-op behavior.

The first version expects callers to provide feedback and rewards. It does not
judge completions by itself.

## Counters

Use `read_stats()` to inspect:

- `queued_feedback`
- `dropped_feedback`
- `feedback_count`
- `update_count`
- `loss_count`
- `last_loss`

An enabled smoke run should report `feedback_count > 0` and `update_count > 0`.
