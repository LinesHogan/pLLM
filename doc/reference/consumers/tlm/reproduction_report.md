# TLM Reproduction Report

## Upstream Mechanism Summary

The upstream TLM repository implements Test-Time Learning through the LLaMA-Factory
`ttl` training stage. The core implementation lives under
`<upstream>/src/llamafactory/train/ttl/`, with example configs in
`<upstream>/examples/train_lora/online_ttl.yaml` and
`<upstream>/examples/train_lora/offline_ttl.yaml`.

The important mechanics are:

- `online_ttl`: for each streaming batch, predict first with the current adapted
  model, then train on that same batch.
- `offline_ttl`: train first, then predict with the adapted model.
- Sample selection is based on sentence-level cross entropy. Samples are kept
  when `sentence_ce > threshold`; lower-CE samples are masked out.
- The per-sample coefficient is `lamb * exp(sentence_ce - threshold)`.
- The training loss is a weighted sentence-level KL-style loss between model
  logits and one-hot labels, averaged over selected samples.
- Offline CE is computed with the adapter disabled, so selection reflects the
  base model before the adapter update. Online CE is computed from the model
  currently being trained.
- The actual upstream adaptation is LoRA fine-tuning of target model modules
  such as `q_proj,v_proj`; it is not a detached reward head over hidden rows.

## tLLM Mapping

The tLLM implementation reviewed is `tllm/consumers/tlm/consumer.py`, with unit
tests in `test/consumers/tlm/test_tlm_consumer_unit.py` and docs in
`doc/reference/consumers/tlm/index.md`.

What maps reasonably:

- It exposes explicit `online` and `offline` modes.
- Offline mode buffers updates until `flush_offline()`.
- It has a threshold-selection helper.
- It has a local low-rank adapter and verifies that adapter weights change.
- It intentionally avoids importing LLaMA-Factory or upstream training
  dependencies.

What does not map to the upstream core:

- Selection direction is inverted for raw upstream CE. tLLM selects scores below
  threshold, while upstream selects CE above threshold. This can align only if
  callers pass a quality/confidence score that decreases as CE increases, such
  as `exp(-CE)`, and transform the threshold accordingly.
- The objective is different. tLLM trains `adapter(hidden)` with MSE toward
  `1 - score`; upstream trains LoRA parameters using weighted KL over token
  logits and labels.
- The `lamb * exp(CE - threshold)` weighting is not implemented.
- tLLM adapts a standalone scalar low-rank adapter head, not the model's LoRA
  target modules.
- `adapter_slots` is allocated but only slot 0 is trained.
- `flows()` is empty, so this consumer is not wired to collect hidden states,
  logits, labels, request metadata, or generation events from the tLLM runtime.
- Online/offline ordering is a local API convention only. It does not enforce
  upstream's "predict then train" online loop or "train then predict" offline
  loop inside decode/runtime execution.

## Upstream Reproduction Feasibility

I did not run the upstream `llamafactory-cli train` entrypoint. The current
environment can import `torch` and `transformers`, but not the upstream training
stack pieces needed by the original entrypoint, including `accelerate`,
`datasets`, `peft`, and an installed `llamafactory` package. Importing the
upstream TTL trainer through `PYTHONPATH=<upstream>/src` fails immediately on
missing `accelerate`.

Installing those dependencies or downloading the example LLaMA model would
violate the requested lightweight, no-download evaluation. Instead, I validated
the upstream formulas from source with a local cached model and synthetic
tensors.

## Checks Run

All Python checks used the repo virtual environment as `<repo>/.venv/bin/python`.

- `pytest -q -p no:cacheprovider test/consumers/tlm/test_tlm_consumer_unit.py`
  - Result: `7 passed`.
- Checked availability of local packages:
  - `torch`, `transformers`, and `vllm` are available.
  - `datasets`, `peft`, and installed `llamafactory` are not available.
- Tried importing upstream TTL trainer with `PYTHONPATH=<upstream>/src`.
  - Result: `ModuleNotFoundError: No module named 'accelerate'`.
- Ran a cached `sshleifer/tiny-gpt2` local-only check to compute upstream-style
  CE/KL and compare threshold selection with tLLM selection.
- Ran a synthetic online/offline tLLM state-machine check.

## Local Model / Toycase Results

Cached tiny model check:

- Model: `sshleifer/tiny-gpt2`, loaded with local files only.
- Two short prompt/answer strings were tokenized and scored.
- Upstream-style CE values were approximately `[10.8277, 10.8280]`.
- Upstream-style KL values matched CE for one-hot labels, approximately
  `[10.8277, 10.8280]`.
- With threshold set to the mean CE, upstream selection
  `CE > threshold` selected row `[1]`.
- tLLM selection on transformed quality scores `exp(-CE) < exp(-threshold)`
  also selected row `[1]`.
- tLLM selection on raw CE with its default "score below threshold" rule selected
  row `[0]`, the opposite side of the upstream boundary.
- A tLLM online update changed adapter weights and produced
  `update_count=1`, `loss_count=1`, `selected_count=1`.

Synthetic state-machine check:

- Scores `[0.9, 0.2, 0.7, 0.1]` with tLLM threshold `0.5` selected two rows.
- Online mode updated immediately:
  `update_count=1`, `loss_count=1`, `selected_count=2`.
- Offline mode buffered all four rows before flush:
  before flush `update_count=0`, `buffered_count=4`;
  after flush `update_count=1`, `loss_count=1`, `selected_count=2`.

These checks confirm the tLLM implementation's own local behavior, and they
also show that raw upstream CE cannot be passed directly into tLLM selection
without reversing the selection semantics.

## Alignment Conclusion

The current tLLM TLM consumer is not a correct reproduction of upstream TLM's
core learning mechanism. It is better described as a lightweight TLM-inspired
state-machine smoke implementation.

It partially captures the surface ideas of online/offline test-time adaptation,
threshold-based sample filtering, and low-rank parameter updates. It does not
reproduce the upstream training objective, CE-based selection semantics,
loss weighting, base-model CE selection in offline TTL, model-LoRA update target,
or integration into an actual predict/train test-time loop.

## Gaps And Risks

- Correctness gap: upstream selects difficult samples by high CE; tLLM selects
  low scalar scores. This needs an explicit score contract or a CE-specific
  selection path.
- Objective gap: MSE to `1 - score` is not equivalent to weighted KL over token
  logits and labels.
- Adaptation gap: training a detached scalar adapter cannot reproduce LoRA
  updates to model attention projections.
- Runtime gap: no declared flows means the consumer cannot run autonomously in
  tLLM's producer-consumer path.
- Offline gap: tLLM has no concept of disabling the adapter to compute base CE
  before selecting samples.
- Online/offline loop gap: tLLM exposes update calls but does not own the
  upstream predict/train ordering.
- Coverage gap: tests verify local counters and updates, but not equivalence to
  upstream CE/KL formulas.

