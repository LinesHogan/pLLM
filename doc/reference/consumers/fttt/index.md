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
- completed-generation trace judging into reward outcomes and queued feedback.

The trace judge path is intentionally lightweight. Callers may inject a local
judge callable that receives each completed generation trace entry and returns
`FTTTJudgeResult`. `process_generation_trace(trace)` then returns a
`RewardOutcomeBatch`; incorrect outputs are also converted into `FeedbackRecord`
items for the existing feedback queue, while correct outputs are marked with
`skip_reason="correct-output"` and are not queued for adaptation.

The default judge is only a toy heuristic for smoke tests. It does not call an
external reward model, code runner, dataset oracle, or upstream FTTT module.

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

## Trace Judge Toy Loop

`FTTTConsumer` declares a consumer flow that reads completed generation traces
and writes completed reward outcomes. The runtime port storage remains owned by
the runtime; the consumer-side helper only builds the batch:

- `FTTTJudgeResult(is_correct=False, reward=0.0, feedback_text=...)` marks a
  failed completion, emits a `RewardOutcomeEntry`, and queues feedback.
- `FTTTJudgeResult(is_correct=True, reward=1.0, ...)` emits a
  `RewardOutcomeEntry` with `skip_reason="correct-output"` and skips feedback
  queueing.
- identity fields (`request_id`, `prompt_idx`, `sample_idx`, `output_index`)
  are passed through to `RewardOutcomeEntry` without implicit coercion, so the
  runtime outcome type checks remain strict.

This closes the local trace -> judge -> feedback/outcome mechanism needed for a
toy FTTT loop. It still does not train on feedback tokens or attach PEFT LoRA
modules to the model.

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
