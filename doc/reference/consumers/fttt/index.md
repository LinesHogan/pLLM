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
- failed-completion context capture, including failed text and token ids,
- hidden-row plus reward/text/token-conditioned updates,
- hidden and logits edit application,
- per-instance retry accounting,
- adapter-version tracking on updated instances,
- stop-on-correct behavior with per-instance reset,
- disabled no-op behavior.
- completed-generation trace judging into reward outcomes and queued feedback.
- consumer-side toy closed-loop execution from trace to local update to retry
  intent.

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
- deterministic feature-hashed failed-completion text and failed token ids,
- deterministic feature-hashed feedback token ids,
- the scalar reward.

The adapter can be queried without another training step:

- `edit_delta(hidden, feedback)` returns the learned hidden delta.
- `apply_edit_to_hidden(hidden, feedback)` returns `hidden + delta`.
- `apply_edit_to_logits(logits, hidden, feedback)` adds the delta directly when
  logits and hidden share the same width.
- `apply_edit_to_logits(..., output_weight=lm_head_weight)` projects the hidden
  delta through an LM-head-style weight before adding it to logits.

The implementation is intentionally local and lightweight: feedback text,
feedback token ids, and failed-completion context affect the edit through stable
feature hashing, not through an upstream tokenizer, reward model, or external
service.

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

`run_toy_closed_loop_step(trace, hidden, ...)` is the dependency-light closed-loop
helper for tests. It:

- judges completed generation entries;
- creates feedback records that retain failed completion text/token ids and
  feedback token ids;
- applies one local adapter update from the emitted feedback records;
- returns retry control with the updated adapter version in the retry reason;
- returns instance states after the update;
- optionally exports the updated adapter as a request-scoped
  `TestTimeLoRAUpdate`.

This closes a consumer-side trace -> judge -> feedback -> local update -> retry
intent loop. It still does not train a full LM feedback-token loss or schedule a
real vLLM regenerate step.

## Instance State

Trace-derived instances are keyed by `request_id + prompt_idx`.
Hand-written `FeedbackRecord` values that leave `request_id` empty use the
legacy prompt-only key, so older toy tests and direct feedback updates remain
compatible.

Incorrect or negative feedback increments that instance's retry count and can
train the edit adapter. A successful local update increments the consumer's
adapter version and records that version on each updated instance state. Positive
feedback stops the instance from future adaptation. Positive feedback is detected
when `reward >= positive_reward_threshold` or when the feedback text contains a
positive marker such as `correct` without a negative marker such as `incorrect`.
Closed-loop intent construction uses the same lifecycle decision as update
application and simulates retry progression across records in the same trace
batch, so retry control and update barriers do not promise an update for a
stopped instance.

Use:

- `instance_state(prompt_idx)` to inspect `retries` and `stopped`.
- `reset_instance(prompt_idx)` to clear stop/retry state and allow adaptation to
  resume for that instance.

## Request-Scoped LoRA Export

`export_test_time_lora_update(...)` packages the current local feedback adapter
as a strict `TestTimeLoRAUpdate`. The toy closed-loop helper can call this after
the local update and fill request scope from the failed completion records. Unit
coverage stages that update through a fake `TestTimeLoRAService`, commits it to
a fake LoRA module, and verifies the request id maps to the exported LoRA id.

This is a service-boundary proof only. The consumer does not register LoRA slots,
patch vLLM, or apply the update during live decoding.

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
- `adapter_version`

An enabled smoke run should report `feedback_count > 0` and `update_count > 0`.
