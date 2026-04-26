# In-Place-TTT-Inspired Consumer

This consumer is a tLLM-native rewrite of the core In-Place TTT idea: use
test-time hidden-state signals to update a fast adaptation component during
inference.

The implementation does not import VeOmni, custom HuggingFace model classes, or
reference repository code. It also does not patch vLLM. Engine interaction goes
through tLLM `ConsumerFlow`s and hidden-row ports.

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
