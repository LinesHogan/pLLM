# In-Place-TTT-Inspired Consumer

This consumer is a tLLM-native rewrite of the core In-Place TTT idea: use
test-time hidden-state signals to update a fast adaptation component during
inference.

The implementation does not import VeOmni, custom HuggingFace model classes, or
reference repository code. The default consumer path interacts with the engine
through tLLM `ConsumerFlow`s and hidden-row ports. An explicit runtime opt-in
can additionally install a limited vLLM runner wrapper for Qwen/Llama-style
MLPs.

## Supported Core

`InPlaceTTTConsumer` reads a source hidden stream and a target hidden stream.
Its local engine keeps a small low-rank side adapter as the tLLM-native
fast-weight approximation and trains it to predict the hidden delta:

```text
desired_delta = target_hidden - source_hidden
adapter(source_hidden) ~= desired_delta
```

Rows are accumulated in an engine-local cache until at least one complete
`chunk_size` is available. No update is performed for an incomplete chunk. Each
complete chunk triggers one adapter update, and any incomplete tail is retained
across later `consume_bundle()` calls. This mirrors the upstream inference
cache shape, where `(past_h, past_t, past_w)` carries partial chunks and the
fast-weight state across calls.

The package also exposes `run_exact_toy_down_proj_step()` for dependency-light
consumer-side alignment tests. That helper is not wired to runtime execution;
it is a small PyTorch reference surface for the upstream-shaped exact math:
`past_h` and `past_t` cache incomplete rows, `past_w` stores the effective
`down_proj.weight`, output rows are emitted with the pre-update weight for
their chunk, and only complete chunks update the fast weight. Below-chunk calls
only cache the tail. The helper supports the upstream optional `ttt_proj`
projection formula.

Runtime support now also includes a limited Qwen/Llama-style MLP wrapper for
single-TP, non-quantized `gate_up_proj -> act_fn -> down_proj` blocks. The
wrapper treats the MLP input as the hidden target `t`, the activation before
`down_proj` as `h`, stores `past_w` as an effective weight, emits each chunk
with its pre-update weight, updates on prefill chunks by default, and keeps
decode update opt-in. If the wrapped module exposes upstream-style `ttt_proj`
or `ttt_conv` modules, their weights are used in the update formula; ordinary
vLLM MLPs without those modules run the no-projection/no-convolution subset.

The runner lifecycle can install this wrapper at `load_model` time behind the
explicit `inplace_ttt_fast_weight_enabled` runtime setting. The current runner
integration supports two key modes:

- `static`: one fixed request key for single-request experiments. This mode
  still rejects multi-request runner batches.
- `request_sample`: a bounded multi-request mode. The runtime builds typed
  packed-token metadata after `_prepare_inputs`; each hidden row is mapped to
  an explicit `(request_id, sample_idx, layer, target_module)` fast-weight key.
  Rows are grouped by key, updated independently, and scattered back to packed
  order.

`request_sample` is intentionally bounded. `inplace_ttt_fast_weight_max_slots`
limits the number of live per-request fast-weight slots, and
`inplace_ttt_fast_weight_max_bytes` can enforce an approximate tensor byte
limit. Profile and warmup forwards that happen before request metadata exists
fall through to the original MLP and do not create synthetic slots.

The runtime fast-weight state has an explicit
`inplace_ttt_fast_weight_mode` surface:

- `exact` is the default and the only executable runtime mode today. It keeps
  the upstream-shaped state contract: `past_h` and `past_t` cache incomplete
  tail rows, and `past_w` stores the full effective `down_proj.weight`.
- `low_rank` is accepted as a public configuration value for budgeting and
  future compatibility, but runtime state creation fails fast with
  `NotImplementedError`. tLLM does not currently define a clear low-rank
  recurrence that is equivalent to the upstream cumulative `past_w` contract,
  so it does not silently approximate exact mode.

Use these formulas for budget planning, where `H` is hidden size, `I` is the
MLP intermediate/down-projection input width, `bytes` is bytes per element, and
`slots` counts live request/sample streams per configured layer:

```text
exact_bytes = slots * layers * (H * I * bytes + tail_rows * (H + I) * bytes)
low_rank_bytes = slots * layers * rank * (H + I) * bytes
```

For example, bfloat16 exact state with `slots=8`, `layers=1`, `H=896`,
`I=4864`, and no cached tail uses about 69.8 MB, or 66.5 MiB:

```text
8 * 1 * (896 * 4864 * 2) = 69,763,072 bytes
```

A hypothetical bfloat16 low-rank state with the same dimensions and
`rank=64` would use about 5.9 MB, or 5.6 MiB:

```text
8 * 1 * 64 * (896 + 4864) * 2 = 5,898,240 bytes
```

That low-rank number is only a sizing estimate until the low-rank recurrence
semantics are implemented and covered by parity tests.

## Delivery

By default, the enabled consumer declares a prefill bundle flow:

- `source`: prefill hidden rows from `source_layer`
- `target`: prefill hidden rows from `target_layer`
- `request_meta`: row metadata

This default follows the original method's long-context setting, where the
model adapts while reading the prompt/context. Decode adaptation can be enabled
explicitly with `phases=("prefill", "decode")`; that optional decode flow uses
the tLLM device-lease lane. A default `InPlaceTTTConsumerConfig()` does not
declare a decode flow.

The consumer treats delivered tensors as read-only and copies only what is
needed for training compatibility with PyTorch inference-mode tensors.

## Counters

Use `read_stats()` to inspect:

- `update_count`
- `loss_count`
- `trained_rows`
- `tail_rows`
- `last_loss`

`update_count`, `loss_count`, and `trained_rows` count complete chunk updates,
not arbitrary delivered rows. An enabled smoke run whose delivered hidden rows
reach at least one full chunk should report `update_count > 0` and
`loss_count > 0`.
