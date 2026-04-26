# FTTT Reproduction Assessment

## Upstream Mechanism Summary

The upstream project has two relevant implementations of "Learning to Reason from Feedback at Test-Time":

- The preliminary FTTT loop generates an answer, checks it with the dataset judge, and stops immediately on a correct answer. On an incorrect but parseable answer, it appends verbal feedback such as "Your answer is incorrect." and performs test-time optimization on a fresh LoRA-wrapped model for that individual test instance. The loss is language-model loss on the feedback tokens, using the generation `past_key_values` from the failed trial, then the adapted model is used for later trials.
- The self-reflection variant first asks the base model to produce a short critique of the failed answer, then trains on the combined incorrect-feedback/reflection text.
- The OpTune path is a learned test-time optimizer. It records generated completions, judges, and rewards. During evaluation, it scores each completion with either ground-truth correctness or a reward-model API, then calls `EditableModel` on `question + completion + feedback`. `EditableModel` backpropagates LM loss through selected inner parameters, captures activation/gradient factors with hooks, transforms them through `GradientTransform`, and returns temporary `named_delta` weight updates for the next generation trial.
- Positive and negative feedback are represented upstream as `judge`/`reward` records. The shipped default training path uses ground-truth feedback and usually excludes correct trials; an alternate strategy can include correct trials or use correct completions as ground-truth targets. At evaluation time, correct trials stop the loop and incorrect trials trigger local adaptation.

The key reproduction target is therefore not merely "a scalar reward changes some parameters"; it is a per-instance generate -> judge/feedback -> local model edit -> regenerate loop, with token-level feedback conditioning and adapter/weight-delta effects on subsequent decoding.

Source files checked for this update:

- Upstream preliminary FTTT loop: `<upstream>/prelim/fttt.py`.
- Upstream OpTune edit model: `<upstream>/optim/editable_model.py`.
- Upstream activation/gradient hooks and transform:
  `<upstream>/optim/hooks.py` and `<upstream>/optim/modules.py`.
- Upstream CLI/dependency surface: `<upstream>/optim/run.py` and
  `<upstream>/README.md`.
- tLLM runtime contract/service:
  `tllm/contracts/test_time_lora.py`, `tllm/runtime/test_time_lora.py`, and
  `tllm/runtime/vllm_test_time_lora.py`.

## tLLM Mapping

The tLLM implementation in `tllm/consumers/fttt/consumer.py` provides:

- `FeedbackRecord(prompt_idx, feedback, reward)`.
- A bounded feedback queue and drain method.
- A small zero-output low-rank edit adapter that maps
  `hidden + hashed feedback text features + reward` to a hidden-space delta.
- MSE training of the hidden delta toward the feedback-conditioned edit target.
- `edit_delta`, `apply_edit_to_hidden`, and `apply_edit_to_logits` helpers that
  let the learned edit affect later hidden rows or logits.
- `export_test_time_lora_update`, which packages the current feedback adapter
  as a strict `TestTimeLoRAUpdate` after the caller provides an explicit runtime
  target module name, positive LoRA id, rank, scope, request ids, and step id.
- Per-instance retry state, stop-on-correct handling, and explicit reset.
- Stats for queued, dropped, consumed feedback, update count, loss count, and last loss.
- A lightweight trace judge toy loop:
  - completed generation trace entries are passed to an injectable local judge;
  - failed completions produce `RewardOutcomeEntry` records and queued
    `FeedbackRecord` items;
  - correct completions produce reward outcomes with `skip_reason="correct-output"`
    and are not queued for adaptation;
  - identity fields are handed to the strict reward outcome runtime types without
    consumer-side coercion.
- A lightweight closed-loop intent path:
  - `RewardOutcomeBatch` records the judge result and feedback text;
  - `RetryControlBatch` retries failed completions below the configured retry
    limit, stops at the limit, and accepts correct outputs;
  - `UpdateBarrierBatch` marks failed-completion feedback rows ready for update;
  - `FTTTClosedLoopIntent.feedback_records` feeds the local low-rank edit
    adapter so tests can observe nonzero update counts.
- No import of external feedback APIs or heavy upstream modules.

This maps to a local feedback-conditioned edit primitive. It is closer than a
scalar reward head because feedback text can change the learned edit and the
edit can be applied to hidden/logit tensors. The trace judge toy loop also
aligns the consumer with the upstream generate -> judge/feedback bookkeeping at
a lightweight data-contract level. It still does not map to the upstream FTTT
loop's full runtime behavior:

- The `feedback` text is feature-hashed, not tokenized or trained with an LM
  feedback-token loss.
- The scalar reward is used as an edit input/training signal; the update does
  not train on feedback tokens or the failed completion context.
- The adapter output can be applied to hidden/logit tensors, but tLLM does not
  yet own an integrated runtime retry loop that applies it during vLLM decoding.
- There is no bundled dataset judge or runtime retry orchestrator.
- There is no equivalent of upstream LoRA test-time optimization or OpTune's activation/gradient transform.
- The `TestTimeLoRAUpdate` export is consumer-side packaging only. It transposes
  the local feedback adapter factors into the runtime contract shape, but it
  does not prove that a concrete vLLM target module accepts or applies the
  update during decoding.
- Positive/negative feedback is reduced to numeric regression targets, not the upstream target construction around incorrect feedback, correct completions, and ground-truth solutions.

## External Services and Dependencies

The tLLM consumer satisfies the "no external service/API/large dependency" constraint. Static inspection found only standard library plus `torch` and local tLLM imports in the consumer. Existing unit tests also check that importing the module does not import `requests` or `fastapi`.

By contrast, the upstream repository can use local services for code-evaluation tasks and can use a reward-model HTTP API for `feedback_type=rm`. Its declared dependencies include `peft`, `fastapi`, `uvicorn`, and related packages. Those are not pulled into the tLLM consumer.

## Upstream Reproduction Feasibility

I attempted to run upstream code with the repository virtual environment and local cache only, avoiding downloads. A true upstream minimal run was not practical in this environment:

- The repository virtual environment uses Python 3.11, while upstream documents Python 3.12. Importing the preliminary FTTT module hits a Python-version-sensitive f-string syntax error.
- The upstream OpTune entry point fails before help text because the current environment lacks the `datasets` package.
- Upstream FTTT scripts are oriented around Llama/Mistral-family models, PEFT LoRA, dataset metadata, and GPU inference. The local model cache contains small models, but the upstream runnable path still requires missing dependencies and dataset assets.

Therefore, I did not treat upstream execution failure as evidence against the algorithm; it is an environment/entrypoint mismatch. I used source inspection plus tLLM-only lightweight checks for this assessment.

## Checks Executed

Commands are summarized with placeholders to avoid machine-local paths:

- `<repo>/.venv/bin/python -m pytest -q test/consumers/fttt/test_fttt_consumer_unit.py`
  - Earlier result before LoRA export coverage: `13 passed`.
  - Result after LoRA export coverage: `16 passed`.
  - Current result after closed-loop adoption coverage: `17 passed`.
- Dependency probe in `<repo>/.venv/bin/python`
  - `torch`, `transformers`, `fastapi`, `requests`, and `sentencepiece` present.
  - `peft` and `adapters` missing.
- Local model cache probe
  - Confirmed small cached models such as `sshleifer/tiny-gpt2`, `gpt2`, and several Qwen models.
- Upstream preliminary import with `<repo>/.venv/bin/python`
  - Result: blocked by f-string syntax under Python 3.11.
- Upstream OpTune entry with `<repo>/.venv/bin/python <upstream>/optim/run.py --help`
  - Result: blocked by missing `datasets`.
- Static dependency scan of the tLLM FTTT consumer
  - No `requests`, `fastapi`, `uvicorn`, `datasets`, `peft`, `transformers`, HTTP, socket, or subprocess usage in the consumer.
- Toy export handoff through fake `TestTimeLoRAService`
  - Result: a trained feedback adapter exported a nonzero
    `TestTimeLoRAUpdate`, committed into a fake LoRA module, and assigned two
    request ids to the LoRA id.

## Lightweight Model/Toy Results

Toy hidden-row update:

- Hidden rows: two orthogonal rows.
- Feedback rewards: incorrect -> `0.0`, correct -> `1.0`.
- After 20 updates, predictions moved from `[0.0, 0.0]` to approximately `[0.005, 1.026]`.
- MSE decreased from approximately `0.413` after the first update to `0.00036`.
- Stats: `feedback_count=40`, `update_count=20`, `loss_count=20`.

Cached real-model hidden-row update:

- Model: local cached `sshleifer/tiny-gpt2`, offline mode.
- Hidden rows: final hidden states for two short answer prompts.
- Feedback rewards: incorrect -> `0.0`, correct -> `1.0`.
- After 15 updates, predictions moved from `[0.0, 0.0]` to approximately `[-0.646, 0.691]`.
- MSE decreased from approximately `0.470` after the first update to `0.256`.
- Stats: `feedback_count=30`, `update_count=15`, `loss_count=15`.

Test-time LoRA export toy check:

- An FTTT feedback adapter was trained on two incorrect feedback records before
  export, so the exported `lora_b` factor was nonzero.
- The exported contract used explicit request scope:
  `adapter_key="fttt-req-7"`, `lora_int_id=23`, `rank=2`,
  `request_ids=("req-7", "req-8")`, `step_id=9`, and a caller-supplied target
  module name.
- Contract tensor shapes were `lora_a=(8, 2)` and `lora_b=(2, 4)`, matching the
  feedback edit input width, rank, and hidden output width.
- A fake `TestTimeLoRAService` commit called `set_lora` on the named module and
  assigned both request ids to LoRA id `23`. This proves the consumer export is
  service-consumable without requiring vLLM, PEFT, or a downloaded model.

These checks validate that the tLLM adapter can consume feedback records, change
parameters, reduce a local edit objective, apply feedback-conditioned hidden or
logit deltas, and update counters. They do not validate upstream-equivalent FTTT
behavior because tLLM still lacks the integrated generate -> judge -> edit ->
retry loop.

## Closed-Loop Toy Evidence

The current consumer branch adds toy coverage for the runtime-facing loop shape:

- a failed `GenerationTraceBatch` entry becomes a reward outcome, a feedback
  record, a retry intent, and an update-barrier entry;
- retry control switches from `retry` to `stop` when the instance retry lifecycle
  reaches the configured limit;
- `consume_bundle()` writes reward, retry, and update-barrier sinks when those
  ports are present;
- the local adapter update path consumes the emitted feedback records and reports
  nonzero update/loss counts.

This remains a consumer-side closed-loop intent, not full upstream FTTT. The
runtime still needs to schedule an actual regenerate-after-update loop and apply
feedback-token or LoRA updates to the model path.

## Alignment Conclusion

Current status: partial mechanism-level alignment, not a complete upstream
reproduction.

The implementation is a clean, local, dependency-light feedback consumer. Its
tests demonstrate bounded queue behavior, feedback-conditioned edits,
hidden/logit edit application, stop-on-correct behavior, per-instance reset, and
trace -> judge -> feedback/outcome conversion. However, it omits the central
training mechanism: failed-generation feedback is not converted into token-level
LM loss, no PEFT LoRA path trains actual model target modules, and tLLM does not
yet orchestrate retry generation from judge outcomes. The new
`TestTimeLoRAUpdate` export gives the runtime team a strict handoff object for
the toy adapter factors, but runtime/vLLM apply smoke remains unverified. As a
result, the consumer should be described as an FTTT-inspired feedback edit
primitive with a toy trace judge loop rather than a complete reproduction of the
upstream method.

## Gaps and Risks

- Feedback text affects the local edit through deterministic feature hashing,
  but this is not equivalent to upstream token-level feedback training.
- Reward sign/value changes train a hidden/logit edit primitive, not the
  language model's target modules or reasoning trajectory.
- Runtime apply gap: no smoke test has staged and applied the exported
  `TestTimeLoRAUpdate` through the real runtime/vLLM LoRA service. The current
  evidence stops at the dependency-light fake `TestTimeLoRAService` commit.
- Positive feedback semantics are not aligned with upstream correct-trial handling.
- No integration test shows that feedback changes final answers inside a real
  generate/retry loop.
- No test covers judge-driven retry behavior across multiple runtime generation
  rounds.
- The existing no-external-API constraint is satisfied, but any future attempt to reproduce reward-model feedback should keep it optional and local unless explicitly allowed.
