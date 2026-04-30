# In-Place TTT reproduction assessment

Date: 2026-04-26
Updated: 2026-04-27
Updated: 2026-05-01

## Scope

This assessment read the local In-Place TTT consumer, its tests and reference doc, then compared it with the upstream implementation. The 2026-04-27 update adds consumer-local exact toy math coverage. The 2026-05-01 update adds an official-shaped parity harness for the runtime fast-weight step using a toy Qwen-style MLP with `gate_proj`, `up_proj`, `down_proj`, optional random-normal `ttt_proj`, and depthwise `ttt_conv`. Runtime contracts for MLP activation, hidden target, module weight view, and fast-weight state now exist. Runtime fast-weight support now includes both a narrow down-projection effective-weight overlay and a limited Qwen/Llama-style MLP wrapper for single-TP, non-quantized MLPs. The wrapper can be installed from the vLLM runner lifecycle behind an explicit runtime opt-in using either a static single-request key or a bounded request/sample key mode for small multi-request batches.

The runtime fast-weight configuration now exposes
`inplace_ttt_fast_weight_mode` with allowed values `exact` and `low_rank`.
`exact` is the default and preserves the existing `past_h`, `past_t`, and full
effective `past_w` behavior. `low_rank` has memory-estimate helpers, but it is
not an executable runtime representation yet; selecting it fails fast with a
documented `NotImplementedError` instead of pretending to match the upstream
fast-weight recurrence.

## Upstream mechanism summary

The upstream implementation modifies standard Transformer MLP blocks rather than adding an external adapter. TTT-enabled layers keep the usual `gate_proj`, `up_proj`, and `down_proj`, then add a depthwise `ttt_conv`, optional projection `ttt_proj`, a fast-weight learning rate, and a chunk size.

The upstream inference initialization path initializes `ttt_proj` with the model's normal linear initializer and zero-initializes `ttt_conv`. The local parity harness uses a fixed random seed for `ttt_proj` so the toy case is deterministic, and it keeps the official zero-init `ttt_conv` as the default. A separate nonzero-convolution ablation copies deterministic depthwise weights only to prove that tLLM and the upstream-shaped reference agree when the convolution path actually contributes a nonzero fast-weight update.

In the training model path, the MLP computes normal intermediate activations `h`, chunks both `h` and the target tensor `t`, derives per-chunk `down_proj` fast-weight increments, cumulatively applies those increments, and produces the MLP output with the updated down-projection weight. The update direction is LM-aligned fast-weight update on the MLP down projection, not a generic hidden reconstruction loss.

In the inference model path, `TTTDynamicCache` stores per-layer `(past_h, past_t, past_w)`. A call with fewer than one full chunk does not update weights; it only emits with the current fast weight and stores the remainder. Once accumulated states reach a full chunk, the implementation applies the chunk update and keeps only the incomplete tail for later calls.

The recommended long-context inference/evaluation setting is effectively prefill-driven: the prompt/context is long enough to form chunks, while configured generation lengths are much smaller than the default/recommended chunk size. The upstream code does not expose a clean public `train_decode` flag in the files reviewed; nevertheless, its cache can technically accumulate single-token generation calls until a full chunk exists. For tLLM reproduction, the safe interpretation is: default to prefill, and require an explicit user choice before enabling decode-side adaptation.

The upstream inference code is not a general batched serving implementation. Its public generation path asserts batch size one, and its chunked MLP loop handles the first batch lane. The tLLM `request_sample` mode therefore aligns to the intended per-sample recurrence by running independent state streams per request/sample instead of reproducing that batch-lane limitation.

## tLLM mapping

The local consumer is a tLLM-native, In-Place-TTT-inspired consumer rather than an exact upstream module transplant.

- `InPlaceTTTConsumerConfig.phases` defaults to `("prefill",)`.
- A decode flow is created only when `phases` explicitly contains `"decode"`.
- Prefill delivery is declared as ordinary `bundle` with borrowed tensors.
- Decode delivery is declared as `device_lease` with runtime-owned active rows.
- The local engine trains a `LowRankAdapter(source_hidden)` to predict `target_hidden - source_hidden` using MSE and Adam.
- The engine trains only complete chunks and carries incomplete tail rows across
  calls. It does not pad tail rows into training examples.
- The default consumer package includes a dependency-light exact toy helper,
  `run_exact_toy_down_proj_step()`, that models the upstream-shaped
  `past_h`, `past_t`, and `past_w` cache contract for `down_proj.weight`.
  This helper is for consumer math alignment tests and is not connected to
  the default low-rank consumer engine.

The live consumer maps the broad online/local hidden-state adaptation idea, but its default bundle-driven engine still does not replace the MLP module the way upstream does. The exact toy helper covers cumulative per-chunk `past_w`, pre-update chunk output, optional `ttt_proj`, and tail-cache math in isolation. Runtime fast-weight execution now uses the same pre-update output order and contract-style weight update; when a wrapped module exposes upstream-style `ttt_conv` or `ttt_proj` weights, the runtime step consumes them. Ordinary vLLM MLPs without those modules run the no-convolution/no-projection subset of the formula.

The runtime MLP wrapper now owns per-layer fast-weight state and can separate packed rows by request/sample in `request_sample` mode. It uses typed packed-token metadata rather than overloading `row_ids`: each row records a `request_id`, `prompt_idx`, `sample_idx`, `packed_row_idx`, `logical_row_idx`, token offset, phase, and engine step id.

## Checks run

Commands were run with the repo virtual environment, summarized without machine-local absolute paths:

- `<repo>/.venv/bin/python -m pytest -q test/consumers/inplace_ttt/test_inplace_ttt_consumer_unit.py`
  - Result: `10 passed`.
- `<repo>/.venv/bin/python -m pytest -q test/consumers/inplace_ttt/test_inplace_ttt_official_parity_unit.py`
  - Result: `5 passed`.
- `<repo>/.venv/bin/python -m pytest -q test/consumers/inplace_ttt test/runtime/inplace_ttt`
  - Result: `45 passed, 4 skipped`.
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
  - Fake Qwen-like MLP tests prove the wrapper captures the MLP input target
    and down-projection activation, caches below-chunk prefill tails, updates
    when tail plus current rows reaches a full chunk, emits chunk outputs before
    applying that chunk's update, restores the original forward method, and
    applies decode without training by default.
  - Runtime step tests cover the upstream projection formula
    `contract("c h, c d, d e -> e h", h, t, ttt_proj.weight)` and verify that
    `past_w` is updated after output emission.
  - Official-shaped parity tests:
    - use a Qwen-style toy MLP with separate `gate_proj`, `up_proj`,
      `down_proj`, optional `ttt_proj`, and depthwise `ttt_conv`;
    - initialize `ttt_proj` with fixed-seed random normal weights and
      `ttt_conv` with official zero weights unless an explicit nonzero
      convolution ablation is under test;
    - compare runtime `run_down_proj_fast_weight_step()` output, updated
      effective `past_w`, and cached activated tail rows against an
      upstream-shaped inference reference;
    - cover no-projection/no-effective-convolution, random `ttt_proj` with
      zero-init `ttt_conv`, copied nonzero-convolution ablation, below-chunk
      tail-only behavior, and tail plus next rows forming a full chunk.
  - Runner lifecycle tests prove the explicit opt-in path installs wrappers
    at model load, stores handles and state on the runtime, avoids duplicate
    installs, isolates configured layer keys, skips unsupported modules when
    configured to do so, and updates the prefill/decode phase from runner
    input-batch metadata, restores wrappers when disabled, and rejects
    multi-request runner batches in static-key mode.
  - Request/sample lifecycle tests prove packed prefill token metadata is
    created from vLLM request-order counts, numpy count arrays are accepted,
    static-key multi-request rejection is bypassed only in `request_sample`
    mode, and grouped rows update independent fast-weight slots.
  - `FastWeightState` now reports live slot count and estimated tensor bytes,
    and can enforce max-slot and max-byte limits before accepting a new slot.
  - Fast-weight mode tests prove that explicit `exact` mode keeps storing
    `past_w` as the full effective down-projection weight, that `low_rank`
    state/runtime creation raises `NotImplementedError`, and that invalid mode
    names are rejected.
  - Memory budgeting helpers cover the documented formulas:
    `exact = slots * layers * (H * I * bytes + tail_rows * (H + I) * bytes)`
    and `low_rank = slots * layers * rank * (H + I) * bytes`.
- Opt-in cached Qwen smoke:
  - With local cached `Qwen/Qwen2.5-0.5B-Instruct`, single TP, eager execution,
    and non-quantized `model.layers.0.mlp.down_proj`, installing an effective
    `past_w` overlay changes generated token ids.
  - The same guarded smoke installs the MLP wrapper on
    `model.layers.0.mlp`, proves below-chunk calls only cache tail rows,
    a following call reaches a full chunk and increments update counters, and
    the adapted MLP output differs from the restored base output.
  - A runner-lifecycle smoke enables the runtime setting before constructing
    the model and verifies `load_model` installs the wrapper without manually
    patching `model.layers.0.mlp`.
  - The smoke is guarded by `TLLM_INPLACE_TTT_RUN_CACHED_MODEL_SMOKE=1` because
    it requires CUDA and a cached model.
- Opt-in cached Qwen request/sample smoke:
  - Batch 4 with one wrapped layer, `chunk_size=2`, and max slots 8 completed
    generation, installed one wrapper, created four request slots, recorded
    nonzero updates for all four slots, trained 32 rows, and estimated about
    34.9 MB of fast-weight tensors.
  - Batch 8 with the same setup completed generation, created eight request
    slots, recorded nonzero updates for all eight slots, trained 64 rows, and
    estimated about 69.8 MB of fast-weight tensors.
- Runtime binding sanity check:
  - The default consumer declares required residual layers with phase `prefill`.
  - Current generic residual binding helper returned no raw tap paths for those prefill locators when given the existing decode-only raw locator map.

Practical memory examples:

- Exact bfloat16 Qwen-style state with `slots=8`, `layers=1`, `H=896`,
  `I=4864`, and no cached tail is
  `8 * 1 * (896 * 4864 * 2) = 69,763,072` bytes, about 66.5 MiB.
- The same dimensions with a hypothetical low-rank rank-64 state would be
  `8 * 1 * 64 * (896 + 4864) * 2 = 5,898,240` bytes, about 5.6 MiB, but that
  is only a planning estimate until low-rank update semantics are implemented.

## Alignment conclusion

At the consumer API and unit-test level, the tLLM implementation correctly
avoids making decode the default. Its default flow is prefill-only, and decode
has to be explicitly requested. The consumer also performs local online updates,
complete-chunk training, tail carryover, and active decode row handling in the
checks above.

The 2026-05-01 official-shaped random-init parity harness did not expose a
runtime math bug in `run_down_proj_fast_weight_step()`. The runtime step matches
the upstream-shaped reference for output emission before each chunk update,
effective `past_w` updates, official random `ttt_proj` plus zero-init
`ttt_conv`, nonzero-convolution ablation, below-chunk tail cache, and tail plus
next rows forming a later chunk. No runtime or consumer math change was required
for this parity slice.

As an exact runtime reproduction of upstream In-Place TTT, it is still incomplete. The upstream core update is an in-place fast-weight update of the MLP down projection with chunkwise cumulative weights; tLLM now has the contract-level runtime surface for MLP activation, hidden target, module weight view, fast-weight state, a narrow effective-weight down-projection overlay, a limited runtime MLP wrapper that captures `h`/`t` inside Qwen/Llama-style MLPs, an explicit runner lifecycle install path, bounded per-request/per-sample packed-row separation for small batched workloads, and the upstream-style pre-update/`ttt_proj` fast-weight update formula. The shipped consumer engine still uses a separate low-rank delta adapter for live bundle consumption, and ordinary vLLM model classes do not expose upstream-trained `ttt_conv`/`ttt_proj` weights unless a compatible module or checkpoint provides them.

The largest practical gap is end-to-end prefill delivery. The consumer declares prefill flows, but the currently inspected generic residual flow/binding path is decode-oriented, while existing prefill support lives in a repro/capture path. I did not find evidence that the default prefill `ConsumerFlow` is fully wired through the normal runtime bundle dispatch. Until that is closed with an end-to-end test or smoke run, `update_count > 0` from direct bundle tests should not be treated as proof of integrated prefill runtime reproduction.

## Risks and recommended follow-up

- Add an end-to-end prefill `ConsumerFlow` dispatch test that drives real prefill hidden rows into `InPlaceTTTConsumer` through the normal runtime path.
- Keep decode opt-in. Do not change the default to include `"decode"`.
- Replace or augment the low-rank delta adapter with the runtime fast-weight
  `down_proj` path for live bundle consumption.
- Expand the request/sample smoke into a repeatable throughput comparison
  with modules/checkpoints that expose upstream `ttt_conv` and `ttt_proj`
  weights.
- Add a small regression test that verifies default `InPlaceTTTConsumerConfig()` does not cause decode residual bindings or decode bundle dispatch.
