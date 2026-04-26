# In-Place TTT reproduction assessment

Date: 2026-04-26

## Scope

This assessment read the local In-Place TTT consumer, its tests and reference doc, then compared it with the upstream implementation. Only this report file was written.

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
- Rows are padded to the configured chunk multiple, while the loss and counters use only valid rows.

This maps the broad online/local hidden-state adaptation idea, but it does not reproduce the exact upstream in-place MLP `down_proj` fast-weight update. It also does not implement upstream `ttt_conv`, optional `ttt_proj`, cumulative per-chunk fast weights, or per-layer cache tails.

## Checks run

Commands were run with the repo virtual environment, summarized without machine-local absolute paths:

- `<repo>/.venv/bin/python -m pytest -q test/devlist/inplace_ttt/test_inplace_ttt_consumer_unit.py`
  - Result: `7 passed`.
- Upstream import smoke with `PYTHONPATH=<upstream>`:
  - Result: failed because `opt_einsum` is not installed in the current virtual environment.
- Upstream import with a temporary in-memory `opt_einsum.contract` shim:
  - Result: failed later with a Transformers decorator API mismatch, so a direct upstream lightweight run was not reliable in this environment.
- Local toycase for tLLM flow/update behavior:
  - Default flow phases: `prefill`, `prefill`.
  - Default prefill bundle updated once: `update_count=1`, `loss_count=1`, `trained_rows=3`.
  - Explicit decode-only flow used `device_lease` and updated only active rows: `update_count=1`, `loss_count=1`, `trained_rows=2`.
- Local cached real-model hidden smoke using `sshleifer/tiny-gpt2` with `local_files_only=True`:
  - Prompt hidden rows: `tokens=9`, `hidden=2`.
  - Feeding source/target hidden rows to the tLLM consumer produced `update_count=1`, `loss_count=1`, `trained_rows=9`.
- Upstream-mechanism toycase:
  - Sequence length below chunk: zero updates.
  - Sequence length equal to chunk: one update.
  - Four single-token calls with chunk size four: no update for the first three calls, one update on the fourth accumulated call.
- Runtime binding sanity check:
  - The default consumer declares required residual layers with phase `prefill`.
  - Current generic residual binding helper returned no raw tap paths for those prefill locators when given the existing decode-only raw locator map.

## Alignment conclusion

At the consumer API and unit-test level, the tLLM implementation correctly avoids making decode the default. Its default flow is prefill-only, and decode has to be explicitly requested. The consumer also performs local online updates and accounts for chunk padding and active decode rows correctly in the checks above.

As an exact reproduction of upstream In-Place TTT, it is incomplete. The upstream core update is an in-place fast-weight update of the MLP down projection with chunkwise cumulative weights; the tLLM consumer trains a separate low-rank delta adapter from source to target hidden states. This is a reasonable tLLM-native approximation, but it should be described as inspired/approximate, not exact.

The largest practical gap is end-to-end prefill delivery. The consumer declares prefill flows, but the currently inspected generic residual flow/binding path is decode-oriented, while existing prefill support lives in a repro/capture path. I did not find evidence that the default prefill `ConsumerFlow` is fully wired through the normal runtime bundle dispatch. Until that is closed with an end-to-end test or smoke run, `update_count > 0` from direct bundle tests should not be treated as proof of integrated prefill runtime reproduction.

## Risks and recommended follow-up

- Add an end-to-end prefill `ConsumerFlow` dispatch test that drives real prefill hidden rows into `InPlaceTTTConsumer` through the normal runtime path.
- Keep decode opt-in. Do not change the default to include `"decode"`.
- If exact upstream reproduction is required, replace or augment the low-rank delta adapter with a fast-weight `down_proj` update path matching upstream `ttt_conv`, `ttt_proj`, chunk accumulation, and cache-tail semantics.
- Add a small regression test that verifies default `InPlaceTTTConsumerConfig()` does not cause decode residual bindings or decode bundle dispatch.
