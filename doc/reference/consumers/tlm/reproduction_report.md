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
- Upstream causal language-model scoring shifts the tensors before computing
  CE/KL: raw LM logits are sliced as `logits[:, :-1, :]` and labels as
  `labels[:, 1:]`. Raw unshifted logits are not an aligned TTL selection input.
- Offline CE is computed with the adapter disabled, so selection reflects the
  base model before the adapter update. Online CE is computed from the model
  currently being trained.
- The actual upstream adaptation is LoRA fine-tuning of target model modules
  such as `q_proj,v_proj`; it is not a detached reward head over hidden rows.

Source files checked for this update:

- Upstream TTL trainer: `<upstream>/src/llamafactory/train/ttl/trainer.py`.
- Upstream online/offline wrapper:
  `<upstream>/src/llamafactory/train/ttl/workflow.py`.
- Upstream LoRA target examples:
  `<upstream>/examples/train_lora/online_ttl.yaml` and
  `<upstream>/examples/train_lora/offline_ttl.yaml`.
- tLLM runtime contract/service:
  `tllm/contracts/test_time_lora.py`, `tllm/runtime/test_time_lora.py`, and
  `tllm/runtime/vllm_test_time_lora.py`.

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
- It can export the trained local toy adapter as a strict
  `TestTimeLoRAUpdate` when the caller supplies an explicit runtime target
  module name, positive LoRA id, rank, scope, request ids, and step id.
- It now exposes `run_lora_learning_step()`, a small closed-loop helper that
  preserves upstream online/offline ordering while staging the exported
  `TestTimeLoRAUpdate` into a runtime LoRA service and invoking an explicit
  commit/apply barrier when one is available.
- When the LoRA service exposes ordinary target modules with 2D weights, the
  helper rejects updates whose exported LoRA factors do not match the target
  module input/output shape.
- It intentionally avoids importing LLaMA-Factory or upstream training
  dependencies.
- Its CE/KL helpers assume logits and labels are already aligned. For raw
  causal-LM outputs, the caller must pass shifted logits and shifted labels.

What does not map to the upstream core:

- The scalar-score fallback path still trains `adapter(hidden)` with MSE toward
  `1 - score`; that fallback is not equivalent to upstream TTL and should not
  be used as the primary aligned path.
- tLLM adapts a local hidden-to-vocab low-rank head, not the model's LoRA target
  modules such as `q_proj,v_proj`.
- Because of that target-module difference, local adapter-weight changes prove
  that the tLLM toy objective trains, but they do not prove parity with an
  upstream LoRA runtime update inside attention or MLP modules.
- The `TestTimeLoRAUpdate` export is consumer-side packaging only. It transposes
  the local toy adapter factors into the runtime contract shape, and the
  closed-loop helper can validate ordinary exposed target-module shapes, but it
  still does not prove vLLM applies the update during decoding.
- `adapter_slots` is allocated but only slot 0 is trained by the closed-loop
  helper.
- `flows()` is empty, so this consumer is not wired to collect hidden states,
  logits, labels, request metadata, or generation events from the tLLM runtime.
- Offline base-logit selection is caller supplied. tLLM does not yet
  autonomously disable a runtime LoRA adapter and compute base logits inside the
  engine.
- Online/offline ordering is enforced by `run_lora_learning_step()` for callers
  using that helper. Offline predict can be made to see the staged update by
  supplying `apply_lora_update` or a service with `commit_staged_updates()`, but
  the helper is not yet automatically invoked by decode/runtime execution.

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

- `pytest -q test/consumers/tlm/test_tlm_consumer_unit.py`
  - Result: `21 passed`.
- `pytest -q test/consumers/tlm test/runtime/test_time_lora`
  - Result: `38 passed, 2 skipped`.
- `TLLM_TLM_RUN_CACHED_MODEL_SMOKE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  pytest -q test/consumers/tlm/test_tlm_cached_model_smoke.py`
  - Result: `1 passed`.
- Checked availability of local packages:
  - `torch`, `transformers`, and `vllm` are available.
  - `datasets`, `peft`, and installed `llamafactory` are not available.
- Tried importing upstream TTL trainer with `PYTHONPATH=<upstream>/src`.
  - Result: `ModuleNotFoundError: No module named 'accelerate'`.
- Ran a cached `sshleifer/tiny-gpt2` local-only check to compute upstream-style
  CE/KL and compare threshold selection with the tLLM CE path.
- Ran a synthetic online/offline tLLM state-machine check.
- Ran a toy export handoff check where a nonzero TLM adapter is converted into
  `TestTimeLoRAUpdate`, staged in `TestTimeLoRAService`, and committed into a
  fake LoRA module with request assignment.

## Local Model / Toycase Results

Synthetic causal-shift toy check:

- Raw causal-LM logits were built so that direct unshifted CE selects row `[1]`
  while upstream-shifted CE selects row `[0]`.
- The shifted toy path computes CE from `logits[:, :-1, :]` and `labels[:, 1:]`
  and matches the local shared `sentence_cross_entropy` helper once those
  tensors are pre-aligned.
- This establishes the tLLM contract: helper and consumer logits inputs are
  already-aligned tensors. Runtime callers must perform causal-LM shifting
  before passing raw generation logits into TLM selection or training.
- The offline toy check also passes shifted `base_logits` separately from
  shifted training logits. Selection uses the caller-supplied base logits, while
  the loss trains against the training logits.

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

Test-time LoRA export toy check:

- A TLM hidden-to-vocab adapter was trained with the logits/labels path before
  export, so the exported `lora_b` factor was nonzero.
- The exported contract used explicit request scope:
  `adapter_key="tlm-req-1"`, `lora_int_id=17`, `rank=2`,
  `request_ids=("req-1",)`, `step_id=5`, and a caller-supplied target module
  name.
- Contract tensor shapes were `lora_a=(4, 2)` and `lora_b=(2, 3)`, matching the
  local toy adapter's input width, rank, and output width.
- A fake `TestTimeLoRAService` commit called `set_lora` on the named module and
  assigned request `req-1` to LoRA id `17`. This proves the consumer export is
  service-consumable without requiring vLLM or a downloaded model.

Closed-loop LoRA staging toy check:

- Online mode called `predict()` before training and staged the request-scoped
  LoRA update afterward, then committed staged updates when the service exposed
  a commit method.
- Offline mode trained, flushed, staged the request-scoped LoRA update, applied
  the commit/update barrier, and only then called `predict()`.
- A separate toy service without `commit_staged_updates()` used an explicit
  `apply_lora_update` callback, modeling the fuller vLLM commit/materialize and
  request-mapping barrier.
- Both paths used the same strict `TestTimeLoRAUpdate` contract as the runtime
  service tests.

Cached-model closed-loop smoke:

- A local-files-only tiny causal LM produced real hidden states and raw logits.
- The smoke shifted logits to `[:, :-1, :]` and labels to `[:, 1:]`, then ran one
  online `run_lora_learning_step()` call.
- The consumer reported a nonzero update and selected samples, and a request
  scoped `TestTimeLoRAUpdate` for the model's logits head was staged.

These checks confirm that the tLLM logits path now matches the upstream CE
selection direction and weighted one-hot KL/CE formula at the local objective
level.

## Alignment Conclusion

The current tLLM TLM consumer is scoped formula and handoff alignment for the
upstream TTL objective, with full trainer-module optimization kept outside core.

It now captures online/offline test-time adaptation APIs, CE-threshold sample
filtering, selected-row exponential weighting, causal-shifted toy parity, and a
local logits/label loss when the caller supplies aligned tensors. It also now
has a consumer-side closed-loop helper that stages a strict
`TestTimeLoRAUpdate` for a caller-named target module after the ordered training
step. It still does not reproduce the upstream model-LoRA update target or
autonomously capture hidden/logit tensors from vLLM. Offline base-model
selection is available only when the caller supplies already-aligned base
logits.

## Gaps And Risks

- Fallback gap: the scalar-score path remains approximate and should be treated
  as a legacy fallback, not the aligned TTL path.
- Adaptation gap: training a local hidden-to-vocab adapter cannot reproduce LoRA
  updates to model attention projections.
- Runtime apply gap: no smoke test has staged and applied the exported
  `TestTimeLoRAUpdate` through the real runtime/vLLM LoRA service. The current
  evidence stops at the dependency-light fake `TestTimeLoRAService` commit.
- Causal-shift gap: the tLLM helper does not infer or apply causal-LM shifting;
  passing raw unshifted logits and labels can select the wrong samples.
- Runtime gap: no declared flows means the consumer cannot yet collect tensors
  autonomously in tLLM's producer-consumer path.
- Offline gap: tLLM does not yet disable runtime adapters itself to compute base
  CE before selecting samples.
- Online/offline loop gap: `run_lora_learning_step()` owns ordering and an
  optional commit/apply barrier only for explicit callers; the runtime does not
  yet call it automatically.
- Coverage gap: tests verify local CE/KL formulas and counters, but not a real
  vLLM LoRA target-module update.
