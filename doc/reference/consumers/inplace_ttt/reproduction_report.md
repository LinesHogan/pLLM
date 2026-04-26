# In-Place TTT reproduction assessment

Date: 2026-04-26
Updated: 2026-04-27

## Scope

This assessment read the local In-Place TTT consumer, its tests and reference doc, then compared it with the upstream implementation. The 2026-04-27 update adds consumer-local exact toy math coverage. Runtime contracts for MLP activation, hidden target, module weight view, and fast-weight state now exist. A narrow runtime down-projection effective-weight overlay is now covered by fake-module tests and an opt-in cached Qwen smoke; full MLP target capture/update orchestration remains outside the completed scope.

## Upstream mechanism summary

The upstream implementation modifies standard Transformer MLP blocks rather than adding an external adapter. TTT-enabled layers keep the usual `gate_proj`, `up_proj`, and `down_proj`, then add a depthwise `ttt_conv`, optional projection `ttt_proj`, a fast-weight learning rate, and a chunk size.

In the training model path, the MLP computes normal intermediate activations `h`, chunks both `h` and the target tensor `t`, derives per-chunk `down_proj` fast-weight increments, cumulatively applies those increments, and produces the MLP output with the updated down-projection weight. The update direction is LM-aligned fast-weight update on the MLP down projection, not a generic hidden reconstruction loss.

In the inference model path, `TTTDynamicCache` stores per-layer `(past_h, past_t, past_w)`. A call with fewer than one full chunk does not update weights; it only emits with the current fast weight and stores the remainder. Once accumulated states reach a full chunk, the implementation applies the chunk update and keeps only the incomplete tail for later calls.

The recommended long-context inference/evaluation setting is effectively prefill-driven: the prompt/context is long enough to form chunks, while configured generation lengths are much smaller than the default/recommended chunk size. The upstream code does not expose a clean public `train_decode` flag in the files reviewed; nevertheless, its cache can technically accumulate single-token generation calls until a full chunk exists. For tLLM reproduction, the safe interpretation is: default to prefill, and require an explicit user choice before enabling decode-side adaptation.

## tLLM mapping

The local consumer is a tLLM-native, In-Place-TTT-inspired consumer rather than an exact upstream module transplant.

- `InPlaceTTTConsumerConfig.phases` defaults to `("prefill",)`.
- A decode flow is created only when `phases` explicitly contains `"decode"`.
- Prefill delivery is declared as ordinary `bundle` with borrowed tensors.
- Decode delivery is declared as `device_lease` with runtime-owned active rows.
- The local engine trains a `LowRankAdapter(source_hidden)` to predict `target_hidden - source_hidden` using MSE and Adam.
- The engine trains only complete chunks and carries incomplete tail rows across
  calls. It does not pad tail rows into training examples.
- The package now includes a dependency-light exact toy helper,
  `run_exact_toy_down_proj_step()`, that models the upstream-shaped
  `past_h`, `past_t`, and `past_w` cache contract for `down_proj.weight`.
  This helper is for consumer math alignment tests and is not connected to
  runtime model execution.

The live consumer maps the broad online/local hidden-state adaptation idea, but it does not reproduce the exact upstream in-place MLP `down_proj` fast-weight update. The exact toy helper covers cumulative per-chunk `past_w` and tail-cache math in isolation, but runtime execution still does not implement upstream `ttt_conv`, optional `ttt_proj`, integrated fast-weight overlay, or per-layer cache ownership.

## Checks run

Commands were run with the repo virtual environment, summarized without machine-local absolute paths:

- `<repo>/.venv/bin/python -m pytest -q test/consumers/inplace_ttt/test_inplace_ttt_consumer_unit.py`
  - Result: `10 passed`.
- Upstream import smoke with `PYTHONPATH=<upstream>`:
  - Result: failed because `opt_einsum` is not installed in the current virtual environment.
- Upstream import with a temporary in-memory `opt_einsum.contract` shim:
  - Result: failed later with a Transformers decorator API mismatch, so a direct upstream lightweight run was not reliable in this environment.
- Local toycase for tLLM flow/update behavior:
  - Default flow phases: `prefill`, `prefill`.
  - With a deliberately small non-default chunk size, a prefill bundle updated
    once: `update_count=1`, `loss_count=1`, `trained_rows=3`.
  - With a deliberately small non-default chunk size, an explicit decode-only
    flow used `device_lease` and updated only active rows:
    `update_count=1`, `loss_count=1`, `trained_rows=2`.
- Local cached real-model hidden smoke using `sshleifer/tiny-gpt2` with `local_files_only=True`:
  - Prompt hidden rows: `tokens=9`, `hidden=2`.
  - Feeding source/target hidden rows to the tLLM consumer with a deliberately
    small non-default chunk size produced `update_count=1`, `loss_count=1`,
    `trained_rows=9`.
  - With the default `chunk_size=4096`, these short toy/model rows remain tail
    rows and do not train immediately.
- Upstream-mechanism toycase:
  - Sequence length below chunk: zero updates.
  - Sequence length equal to chunk: one update.
  - Four single-token calls with chunk size four: no update for the first three calls, one update on the fourth accumulated call.
- Consumer exact toy math tests:
  - Full chunks match a direct PyTorch reference implementation of cumulative
    effective `past_w` updates.
  - Below-chunk rows produce no update and are cached only in `past_h/past_t`.
  - Cached tail rows combine with later rows to form the next full chunk, then
    the new incomplete remainder is retained as tail.
- Runtime fast-weight overlay tests:
  - `FastWeightState` now validates row counts and checks that `past_h` width
    matches the down-projection input while `past_t` width matches the output.
  - The down-projection wrapper uses stored `past_w` as the effective weight,
    replacing the module output instead of treating `past_w` as a delta.
  - Fake-module tests prove no-slot behavior matches the base module, a stored
    effective weight changes output, shape mismatches fail, and restore returns
    to base output.
- Opt-in cached Qwen smoke:
  - With local cached `Qwen/Qwen2.5-0.5B-Instruct`, single TP, eager execution,
    and non-quantized `model.layers.0.mlp.down_proj`, installing an effective
    `past_w` overlay changes generated token ids.
  - The smoke is guarded by `TLLM_INPLACE_TTT_RUN_CACHED_MODEL_SMOKE=1` because
    it requires CUDA and a cached model.
- Runtime binding sanity check:
  - The default consumer declares required residual layers with phase `prefill`.
  - Current generic residual binding helper returned no raw tap paths for those prefill locators when given the existing decode-only raw locator map.

## Alignment conclusion

At the consumer API and unit-test level, the tLLM implementation correctly
avoids making decode the default. Its default flow is prefill-only, and decode
has to be explicitly requested. The consumer also performs local online updates,
complete-chunk training, tail carryover, and active decode row handling in the
checks above.

As an exact runtime reproduction of upstream In-Place TTT, it is still incomplete. The upstream core update is an in-place fast-weight update of the MLP down projection with chunkwise cumulative weights; tLLM now has the contract-level runtime surface for MLP activation, hidden target, module weight view, fast-weight state, and a narrow effective-weight down-projection overlay. The current completion covers exact chunk/tail/update math and model-output-changing overlay application, while the shipped consumer engine still uses a separate low-rank delta adapter for live bundle consumption and does not yet own a full vLLM MLP capture/update loop.

The largest practical gap is end-to-end prefill delivery. The consumer declares prefill flows, but the currently inspected generic residual flow/binding path is decode-oriented, while existing prefill support lives in a repro/capture path. I did not find evidence that the default prefill `ConsumerFlow` is fully wired through the normal runtime bundle dispatch. Until that is closed with an end-to-end test or smoke run, `update_count > 0` from direct bundle tests should not be treated as proof of integrated prefill runtime reproduction.

## Risks and recommended follow-up

- Add an end-to-end prefill `ConsumerFlow` dispatch test that drives real prefill hidden rows into `InPlaceTTTConsumer` through the normal runtime path.
- Keep decode opt-in. Do not change the default to include `"decode"`.
- Complete the vLLM MLP capture/update orchestration before claiming exact
  integrated reproduction: MLP internal activation `h`, hidden target `t`,
  `down_proj.weight` view, and per-request/per-sample/per-layer fast-weight
  state now have contracts and an overlay primitive, but the update loop is not
  yet wired into model execution.
- Once runtime support exists, replace or augment the low-rank delta adapter
  with a fast-weight `down_proj` update path matching upstream `ttt_conv`,
  `ttt_proj`, chunk accumulation, and cache-tail semantics.
- Add a small regression test that verifies default `InPlaceTTTConsumerConfig()` does not cause decode residual bindings or decode bundle dispatch.
