# In-Place-TTT-Inspired Consumer

This consumer is a tLLM-native rewrite of the core In-Place TTT idea: use
test-time hidden-state signals to update a fast adaptation component during
inference.

The implementation does not import VeOmni, custom HuggingFace model classes, or
reference repository code. It also does not patch vLLM. Engine interaction goes
through tLLM `ConsumerFlow`s and hidden-row ports.

## Supported Core

`InPlaceTTTConsumer` reads a source hidden stream and a target hidden stream.
Its local engine trains a small low-rank side adapter to predict the hidden
delta:

```text
desired_delta = target_hidden - source_hidden
adapter(source_hidden) ~= desired_delta
```

Rows are padded to a chunk multiple before the update, matching the prefill
chunk discipline used by In-Place TTT, while the loss is computed only on valid
rows.

## Delivery

By default, the enabled consumer declares a prefill bundle flow:

- `source`: prefill hidden rows from `source_layer`
- `target`: prefill hidden rows from `target_layer`
- `request_meta`: row metadata

This default follows the original method's long-context setting, where the
model adapts while reading the prompt/context. Decode adaptation can be enabled
explicitly with `phases=("prefill", "decode")`; that optional decode flow uses
the tLLM device-lease lane.

The consumer treats delivered tensors as read-only and copies only what is
needed for training compatibility with PyTorch inference-mode tensors.

## Counters

Use `read_stats()` to inspect:

- `update_count`
- `loss_count`
- `trained_rows`
- `last_loss`

An enabled smoke run should report `update_count > 0` and `loss_count > 0`.
