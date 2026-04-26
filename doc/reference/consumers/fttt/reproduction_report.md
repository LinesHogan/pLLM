# FTTT Reproduction Assessment

## Upstream Mechanism Summary

The upstream project has two relevant implementations of "Learning to Reason from Feedback at Test-Time":

- The preliminary FTTT loop generates an answer, checks it with the dataset judge, and stops immediately on a correct answer. On an incorrect but parseable answer, it appends verbal feedback such as "Your answer is incorrect." and performs test-time optimization on a fresh LoRA-wrapped model for that individual test instance. The loss is language-model loss on the feedback tokens, using the generation `past_key_values` from the failed trial, then the adapted model is used for later trials.
- The self-reflection variant first asks the base model to produce a short critique of the failed answer, then trains on the combined incorrect-feedback/reflection text.
- The OpTune path is a learned test-time optimizer. It records generated completions, judges, and rewards. During evaluation, it scores each completion with either ground-truth correctness or a reward-model API, then calls `EditableModel` on `question + completion + feedback`. `EditableModel` backpropagates LM loss through selected inner parameters, captures activation/gradient factors with hooks, transforms them through `GradientTransform`, and returns temporary `named_delta` weight updates for the next generation trial.
- Positive and negative feedback are represented upstream as `judge`/`reward` records. The shipped default training path uses ground-truth feedback and usually excludes correct trials; an alternate strategy can include correct trials or use correct completions as ground-truth targets. At evaluation time, correct trials stop the loop and incorrect trials trigger local adaptation.

The key reproduction target is therefore not merely "a scalar reward changes some parameters"; it is a per-instance generate -> judge/feedback -> local model edit -> regenerate loop, with token-level feedback conditioning and adapter/weight-delta effects on subsequent decoding.

## tLLM Mapping

The tLLM implementation in `tllm/consumers/fttt/consumer.py` provides:

- `FeedbackRecord(prompt_idx, feedback, reward)`.
- A bounded feedback queue and drain method.
- A small zero-output low-rank edit adapter that maps
  `hidden + hashed feedback text features + reward` to a hidden-space delta.
- MSE training of the hidden delta toward the feedback-conditioned edit target.
- `edit_delta`, `apply_edit_to_hidden`, and `apply_edit_to_logits` helpers that
  let the learned edit affect later hidden rows or logits.
- Per-instance retry state, stop-on-correct handling, and explicit reset.
- Stats for queued, dropped, consumed feedback, update count, loss count, and last loss.
- No import of external feedback APIs or heavy upstream modules.

This maps to a local feedback-conditioned edit primitive. It is closer than a
scalar reward head because feedback text can change the learned edit and the
edit can be applied to hidden/logit tensors. It still does not map to the
upstream FTTT loop's full runtime behavior:

- The `feedback` text is feature-hashed, not tokenized or trained with an LM
  feedback-token loss.
- The scalar reward is used as an edit input/training signal; the update does
  not train on feedback tokens or the failed completion context.
- The adapter output can be applied to hidden/logit tensors, but tLLM does not
  yet own an integrated generate/check/retry loop that applies it during vLLM
  decoding.
- There is no dataset judge or runtime retry orchestrator.
- There is no equivalent of upstream LoRA test-time optimization or OpTune's activation/gradient transform.
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
  - Result: `10 passed in 1.17s`.
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

These checks validate that the tLLM adapter can consume feedback records, change
parameters, reduce a local edit objective, apply feedback-conditioned hidden or
logit deltas, and update counters. They do not validate upstream-equivalent FTTT
behavior because tLLM still lacks the integrated generate -> judge -> edit ->
retry loop.

## Alignment Conclusion

Current status: partial mechanism-level alignment, not a complete upstream
reproduction.

The implementation is a clean, local, dependency-light feedback consumer. Its
tests demonstrate bounded queue behavior, feedback-conditioned edits,
hidden/logit edit application, stop-on-correct behavior, and per-instance reset.
However, it omits the central runtime loop: failed-generation feedback is not
converted into token-level LM training, no LoRA/named-delta path edits actual
model target modules, and tLLM does not yet orchestrate retry generation from
judge outcomes. As a result, the consumer should be described as an
FTTT-inspired feedback edit primitive rather than a complete reproduction of the
upstream method.

## Gaps and Risks

- Feedback text affects the local edit through deterministic feature hashing,
  but this is not equivalent to upstream token-level feedback training.
- Reward sign/value changes train a hidden/logit edit primitive, not the
  language model's target modules or reasoning trajectory.
- Positive feedback semantics are not aligned with upstream correct-trial handling.
- No integration test shows that feedback changes final answers inside a real
  generate/retry loop.
- No test covers judge-driven retry behavior across runtime generations.
- The existing no-external-API constraint is satisfied, but any future attempt to reproduce reward-model feedback should keep it optional and local unless explicitly allowed.
