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
  Feedback records may also carry request id, sample/output index, group id,
  failed completion text, failed completion token ids, and feedback token ids.
- A bounded feedback queue and drain method.
- A small zero-output low-rank edit adapter that maps
  `hidden + hashed feedback text/context/token features + reward` to a
  hidden-space delta.
- MSE training of the hidden delta toward the feedback-conditioned edit target.
- `edit_delta`, `apply_edit_to_hidden`, and `apply_edit_to_logits` helpers that
  let the learned edit affect later hidden rows or logits.
- `export_test_time_lora_update`, which packages the current feedback adapter
  as a strict `TestTimeLoRAUpdate` after the caller provides an explicit runtime
  target module name, positive LoRA id, rank, scope, request ids, and step id.
- Per-instance retry state, stop-on-correct handling, and explicit reset.
  Trace-derived feedback uses a `request_id + prompt_idx`
  lifecycle key; hand-written records without `request_id` keep the legacy
  prompt-only key.
  Successful updates increment an adapter version and store the version on each
  updated instance state.
- Stats for queued, dropped, consumed feedback, update count, loss count, and last loss.
- A lightweight trace judge toy loop:
  - completed generation trace entries are passed to an injectable local judge;
  - failed completions produce `RewardOutcomeEntry` records and queued
    `FeedbackRecord` items that retain failed completion context and feedback
    token ids;
  - correct completions produce reward outcomes with `skip_reason="correct-output"`
    and are not queued for adaptation;
  - identity fields are handed to the strict reward outcome runtime types without
    consumer-side coercion.
- A lightweight closed-loop intent path:
  - `RewardOutcomeBatch` records the judge result and feedback text;
  - `RetryControlBatch` retries failed completions below the configured retry
    limit, stops at the limit, and accepts correct outputs;
  - retry and barrier intent uses the same lifecycle decision as update
    application, including request-scoped instance state and batch-local retry
    progression;
  - `UpdateBarrierBatch` marks adaptable failed-completion feedback rows ready
    for update and blocks stopped instances instead of promising an update;
  - `FTTTClosedLoopIntent.feedback_records` feeds the local low-rank edit
    adapter so tests can observe nonzero update counts.
- A toy closed-loop helper:
  - `run_toy_closed_loop_step(trace, hidden, ...)` runs judge, feedback record
    construction, local adapter update, updated retry-control construction, and
    optional request-scoped LoRA export in one consumer-layer step;
  - failed completion text/token ids and feedback token ids are part of the
    hashed adapter input, so changing either can change the resulting adapter
    update even when the scalar reward is unchanged;
  - retry intent is returned after the update and includes the updated adapter
    version in the retry reason, while updated instance state exposes retry
    count and adapter version.
- An optional trainer-bridge handoff:
  - `feedback_training_batch(feedback_records)` converts failed-completion
    feedback records into `feedback_lm` examples with failed tokens masked and
    feedback token ids used as the label tokens;
  - failed completion text, failed token ids, feedback token ids, request id,
    sample id, output id, and group id are preserved in strict contract fields
    and example metadata;
  - `run_trainer_bridge_closed_loop_step(...)` submits the feedback batch to a
    dependency-light `TestTimeTrainerBridge` and stages returned
    `TestTimeLoRAUpdate` values through `TestTimeLoRAService` when supplied;
  - retry control references the staged adapter key and version only after an
    update is returned, and blocks retry-after-update when the trainer returns
    no LoRA update.
- No import of external feedback APIs or heavy upstream modules.

This maps to a local feedback-conditioned edit primitive. It is closer than a
scalar reward head because feedback text can change the learned edit and the
edit can be applied to hidden/logit tensors. The trace judge toy loop also
aligns the consumer with the upstream generate -> judge/feedback bookkeeping at
a lightweight data-contract level. It still does not map to the upstream FTTT
loop's full runtime behavior:

- The feedback text, feedback token ids, and failed completion context are
  feature-hashed, not tokenized or trained with an LM feedback-token loss.
- The scalar reward is used as an edit input/training signal; the update does
  not reproduce upstream loss over feedback tokens with generation cache reuse.
- The adapter output can be applied to hidden/logit tensors, but tLLM does not
  yet own an integrated runtime retry loop that applies it during vLLM decoding.
- There is no bundled dataset judge or runtime retry orchestrator.
- There is no bundled equivalent of upstream LoRA test-time optimization or
  OpTune's activation/gradient transform. Full feedback-LM training is expected
  to live behind an optional external trainer bridge.
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

- `<repo-venv>/bin/python -m pytest -q test/consumers/fttt/test_fttt_consumer_unit.py`
  - Earlier result before LoRA export coverage: `13 passed`.
  - Result after LoRA export coverage: `16 passed`.
  - Current result after closed-loop adoption and LoRA boundary coverage: `19 passed`.
  - Current result after toy closed-loop helper and context/token coverage:
    `22 passed`.
  - Current result after request-scoped lifecycle and retry-barrier fixes:
    `25 passed`.
- `<repo-venv>/bin/python -m pytest -q test/consumers/ttrl test/consumers/fttt`
  - Reviewer-observed result after TTRL and FTTT combination coverage:
    `42 passed`.
  - Current result after FTTT lifecycle/retry regression coverage:
    `45 passed`.
- `<repo-venv>/bin/python -m pytest -q test/test_runtime_post_generate_dispatch_unit.py`
  - Current result: `8 passed, 2 subtests passed`.
- Dependency probe in `<repo-venv>/bin/python`
  - `torch`, `transformers`, `fastapi`, `requests`, and `sentencepiece` present.
  - `peft` and `adapters` missing.
- Local model cache probe
  - Confirmed small cached models such as `sshleifer/tiny-gpt2`, `gpt2`, and several Qwen models.
- Upstream preliminary import with `<repo-venv>/bin/python`
  - Result: blocked by f-string syntax under Python 3.11.
- Upstream OpTune entry with `<repo-venv>/bin/python <upstream>/optim/run.py --help`
  - Result: blocked by missing `datasets`.
- Static dependency scan of the tLLM FTTT consumer
  - No `requests`, `fastapi`, `uvicorn`, `datasets`, `peft`, `transformers`, HTTP, socket, or subprocess usage in the consumer.
- Toy export handoff through fake `TestTimeLoRAService`
  - Result: a trained feedback adapter exported a nonzero
    `TestTimeLoRAUpdate`, committed into a fake LoRA module, and assigned two
    request ids to the LoRA id.
- Trainer-bridge handoff through fake `TestTimeTrainerBridge` and fake
  `TestTimeLoRAService`
  - Result: failed-completion feedback became a `feedback_lm`
    `TestTimeTrainingBatch`, the fake bridge returned a request-scoped
    `TestTimeLoRAUpdate`, the update staged and committed through the fake LoRA
    service, retry control referenced the staged adapter key/version, and an
    empty trainer result blocked retry-after-update.

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

Toy closed-loop helper check:

- A failed `GenerationTraceBatch` entry with text and token ids was judged into
  a feedback record carrying failed completion context and feedback token ids.
- `run_toy_closed_loop_step()` applied a local adapter update and returned
  `adapter_version=1`, updated instance retry state, and retry control whose
  reason references the updated adapter version.
- With fixed random seed, hidden row, scalar reward, and feedback text, changing
  only the failed completion text produced different adapter weights.
- With fixed random seed, hidden row, scalar reward, failed completion, and
  feedback text, changing only the feedback token ids produced different adapter
  weights.
- The same helper exported a request-scoped `TestTimeLoRAUpdate` after the
  update. A fake `TestTimeLoRAService` committed it into a fake target module
  and assigned the request id to the exported LoRA id.

These checks validate that the tLLM adapter can consume feedback records, change
parameters, reduce a local edit objective, apply feedback-conditioned hidden or
logit deltas, and update counters. They do not validate upstream-equivalent FTTT
behavior because tLLM still lacks the integrated generate -> judge -> edit ->
retry loop.

## Closed-Loop Toy Evidence

The current consumer branch adds toy coverage for the runtime-facing loop shape:

- a failed `GenerationTraceBatch` entry becomes a reward outcome, a feedback
  record, a retry intent, and an update-barrier entry;
- failed completion text/token ids and feedback token ids are retained in the
  feedback record and participate in the local adapter update;
- retry control switches from `retry` to `stop` when the instance retry lifecycle
  reaches the configured limit;
- two trace-derived requests with the same prompt index keep separate lifecycle
  state, while records without request ids retain prompt-only compatibility;
- closed-loop intent simulates multiple failed completions for the same
  request-scoped instance in one batch, so later failures see earlier in-batch
  retry decisions;
- stopped instances no longer emit ready update barriers or feedback records in
  closed-loop intent;
- the consumer-side toy helper returns retry intent after the adapter update and
  references the updated adapter version;
- `consume_bundle()` writes reward, retry, and update-barrier sinks when those
  ports are present;
- post-generate fake runtime dispatch smoke uses the real `FTTTConsumer` to
  turn `runtime.last_generation_trace` into runtime reward, retry, and update
  barrier sink writes without vLLM, GPU, or model downloads;
- the local adapter update path consumes the emitted feedback records and reports
  nonzero update/loss counts.
- optional external training receives a `feedback_lm` batch without importing
  heavy trainer dependencies, and returned LoRA updates can be staged through the
  shared runtime LoRA service before retry intent is released.

This is a consumer-side closed-loop intent plus trainer handoff, with full
feedback-token optimization intentionally outside tLLM core. Runtime retry
scheduling is represented by strict retry/update-barrier contracts so optional
trainer integrations can provide the heavy PEFT/OpTune or equivalent gradient
path without becoming a core dependency.

## Alignment Conclusion

Current status: scoped mechanism alignment with an explicit external-trainer
boundary.

The implementation is a clean, local, dependency-light feedback consumer. Its
tests demonstrate bounded queue behavior, feedback-conditioned edits,
hidden/logit edit application, stop-on-correct behavior, per-instance reset, and
trace -> judge -> feedback/outcome conversion. Failed-generation feedback is
converted into `feedback_lm` training batches, a dependency-light trainer bridge
returns request-scoped `TestTimeLoRAUpdate` values, and retry-after-update is
released only when an update is staged. The new guarded
`test_fttt_real_model_smoke.py` path loads a cached tiny causal LM, builds a
failed-completion feedback batch, stages a request-scoped LoRA update through
`TestTimeLoRAService`, commits it into a fake target module, and verifies retry
intent references the post-update adapter key/version. The remaining boundary is
intentional: full PEFT/OpTune feedback-token optimization belongs in an optional
external trainer integration, not in unconditional tLLM core imports.

## Gaps and Risks

- Feedback text affects the local edit through deterministic feature hashing,
  but this is not equivalent to upstream token-level feedback training.
- Reward sign/value changes train a hidden/logit edit primitive, not the
  language model's target modules or reasoning trajectory.
- Runtime apply boundary: the guarded cached-model smoke and unit tests stage
  and commit `TestTimeLoRAUpdate` values through the dependency-light
  `TestTimeLoRAService`; vLLM-native in-memory LoRA materialization remains
  covered by the shared runtime LoRA tests.
- Positive feedback semantics are not aligned with upstream correct-trial handling.
- No integration test shows that feedback changes final answers inside a real
  generate/retry loop.
- No test covers judge-driven retry behavior across multiple runtime generation
  rounds.
- The existing no-external-API constraint is satisfied, but any future attempt
  to reproduce reward-model feedback or full feedback-LM training should keep it
  optional and local unless explicitly allowed.
