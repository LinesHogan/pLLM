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
- A small zero-output low-rank adapter from hidden rows to one scalar.
- MSE training from hidden rows to scalar rewards.
- Stats for queued, dropped, consumed feedback, update count, loss count, and last loss.
- No import of external feedback APIs or heavy upstream modules.

This maps only to a small local reward-learning primitive. It does not map to the upstream FTTT loop's core runtime behavior:

- The `feedback` text is stored but not tokenized, modeled, or used in the loss.
- The update predicts scalar reward; it does not train on feedback tokens or the failed completion context.
- The adapter output is not connected to next-token logits, model weights, or a named-delta mechanism, so the update cannot by itself change future reasoning/generation.
- There is no generate/check/retry loop, no dataset judge, no stop-on-correct behavior, and no per-test-instance reset.
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
  - Result: `6 passed in 1.11s`.
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

These checks validate that the tLLM adapter can consume feedback records, change parameters, reduce a scalar reward-regression objective, and update counters. They do not validate upstream-equivalent FTTT behavior because the adapted scalar head is not connected to generation.

## Alignment Conclusion

Current status: partial inspiration, not a correct core reproduction.

The implementation is a clean, local, dependency-light feedback consumer, and its tests demonstrate bounded queue behavior and reward-regression updates. However, it omits the central FTTT mechanism: failed-generation feedback is not converted into token-level training, no local model/LoRA/named-delta edit affects subsequent decoding, and there is no per-instance retry loop governed by correctness feedback. As a result, the consumer should be described as an FTTT-inspired feedback adapter primitive rather than a reproduction of the upstream method.

## Gaps and Risks

- The `feedback` string currently has no behavioral effect.
- Reward sign/value changes train only a scalar predictor, not the language model's reasoning trajectory.
- Positive feedback semantics are not aligned with upstream correct-trial handling.
- No integration test shows that feedback changes next-token probabilities or final answers.
- No test covers per-instance reset, repeated trials, stop-on-correct, or judge-driven retry behavior.
- The existing no-external-API constraint is satisfied, but any future attempt to reproduce reward-model feedback should keep it optional and local unless explicitly allowed.
