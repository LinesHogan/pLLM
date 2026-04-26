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
- It has a CE threshold-selection helper that selects rows with
  `sentence_ce > threshold`, matching the upstream direction.
- Its logits path implements the selected-row weighted one-hot KL/CE objective
  with `lamb * exp(sentence_ce - threshold)`.
- Offline mode accepts caller-supplied `base_logits` for base-model CE
  selection before the local adapter update.
- It has a local low-rank hidden-to-vocab adapter and verifies that adapter
  weights change.
- It intentionally avoids importing LLaMA-Factory or upstream training
  dependencies.

What does not map to the upstream core:

- The scalar-score fallback path still trains `adapter(hidden)` with MSE toward
  `1 - score`; that fallback is not equivalent to upstream TTL and should not
  be used as the primary aligned path.
- tLLM adapts a local hidden-to-vocab low-rank head, not the model's LoRA target
  modules such as `q_proj,v_proj`.
- `adapter_slots` is allocated but only slot 0 is trained.
- `flows()` is empty, so this consumer is not wired to collect hidden states,
  logits, labels, request metadata, or generation events from the tLLM runtime.
- Offline base-logit selection is caller supplied. tLLM does not yet
  autonomously disable a runtime LoRA adapter and compute base logits inside the
  engine.
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
  - Result: `13 passed`.
- Checked availability of local packages:
  - `torch`, `transformers`, and `vllm` are available.
  - `datasets`, `peft`, and installed `llamafactory` are not available.
- Tried importing upstream TTL trainer with `PYTHONPATH=<upstream>/src`.
  - Result: `ModuleNotFoundError: No module named 'accelerate'`.
- Ran a cached `sshleifer/tiny-gpt2` local-only check to compute upstream-style
  CE/KL and compare threshold selection with the tLLM CE path.
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
- tLLM selection on raw CE also selected row `[1]`.
- A tLLM online logits update changed adapter weights and produced
  `update_count=1`, `loss_count=1`, `selected_count=1`.

Synthetic state-machine check:

- Scores `[0.9, 0.2, 0.7, 0.1]` with tLLM threshold `0.5` selected two rows.
- Online mode updated immediately:
  `update_count=1`, `loss_count=1`, `selected_count=2`.
- Offline mode buffered all four rows before flush:
  before flush `update_count=0`, `buffered_count=4`;
  after flush `update_count=1`, `loss_count=1`, `selected_count=2`.

These checks confirm that the tLLM logits path now matches the upstream CE
selection direction and weighted one-hot KL/CE formula at the local objective
level.

## Alignment Conclusion

The current tLLM TLM consumer is a partial formula-level reproduction of the
upstream TTL objective, but not a full runtime reproduction.

It now captures online/offline test-time adaptation APIs, CE-threshold sample
filtering, selected-row exponential weighting, and a local logits/label loss. It
still does not reproduce the upstream model-LoRA update target or integration
into an actual predict/train runtime loop. Offline base-model selection is
available only when the caller supplies base logits.

## Gaps And Risks

- Fallback gap: the scalar-score path remains approximate and should be treated
  as a legacy fallback, not the aligned TTL path.
- Adaptation gap: training a local hidden-to-vocab adapter cannot reproduce LoRA
  updates to model attention projections.
- Runtime gap: no declared flows means the consumer cannot run autonomously in
  tLLM's producer-consumer path.
- Offline gap: tLLM does not yet disable runtime adapters itself to compute base
  CE before selecting samples.
- Online/offline loop gap: tLLM exposes update calls but does not own the
  upstream predict/train ordering.
- Coverage gap: tests verify local CE/KL formulas and counters, but not a real
  vLLM LoRA target-module update.
