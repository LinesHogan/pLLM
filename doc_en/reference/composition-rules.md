# Composition Rules

Each effect in a workload declares its position within a step and the states it reads or writes. pLLM orders effects by position, rejects only genuinely conflicting combinations, and reports an error with remediation instructions at build time (`pllm.explain` or `pllm.make_llm`). This page lists which combinations can coexist, which are rejected, and why. For how to compose workloads, see [Composing Workloads](../guides/composition.md).

## Allowed Combinations

| Combination | Conditions & Notes |
| --- | --- |
| Multiple `@pllm.train` methods, each training its own state | Must use the same training mode (all on request rows, or all on host batches) |
| `@pllm.export` with any effect | Exports are read-only |
| Multiple `@pllm.add_logits` methods (including from different components) | Summed before filtering, then sampling guidance is applied; addition is order-independent |
| `@pllm.add_logits` / `@pllm.guide_sampling` with `Adapter`, `ActivationPatch`, `@pllm.train_in_forward` | Sampling effects run after the forward pass; model edits apply inside the forward pass |
| One `@pllm.custom_forward` with a training overlay and an `ActivationPatch` | Every layer executed by custom forward passes through them; `Adapter` deltas apply on each projection call |
| `@pllm.custom_residual` with `@pllm.custom_forward` | Every layer executed by custom forward also passes through the custom residual (except stateful sublayers; see table below) |
| `pllm.residual` / `pllm.layer_logits` with `@pllm.custom_forward` | Reads the last computed residual stream for each layer |
| Producer and consumer of the same state | For example, `@pllm.write` or `@pllm.train` writing and `Adapter` reading; each state may have at most one writer method per step |

## Rejected Combinations

| Combination | Why | Fix |
| --- | --- | --- |
| Stacking two decorators of the same kind on one method (e.g., `@pllm.export` and `@pllm.train`) | The later decorator overwrites the earlier one, silently dropping one effect; different effects also have different default phases | Use one method per effect and share computation via `pllm.result(name)` |
| Two `@pllm.guide_sampling` methods (including across components) | The sampler reweights against only one prediction per step; how two predictions should be combined is an algorithmic choice | Combine into one prediction in a single method; use `@pllm.add_logits` for additional logit adjustments |
| Request-row training and host-batch training in the same workload (including components) | The two training modes differ in update timing and memory reservation; a workload supports only one training mode | Split across two engines; state read only by host code can be trained separately with `pllm.host.make_host` |
| Standalone request-row training, or training alongside `@pllm.guide_sampling`, with `rows=` / `update_rows=` | This path performs one update and publishes a new version per state on every step, and does not support skipping rows | Select rows on `@pllm.guide_sampling(rows=...)`, or use `reduction="sum"` and multiply the loss of excluded rows by 0 (see [Online Training](../guides/training.md)) |
| `@pllm.custom_forward` with a method reading `pllm.hidden` / `pllm.final_hidden` | In a custom forward, a layer may execute multiple times or not at all, so "that layer's output" is not unique | Read `pllm.residual(layer)` or `pllm.layer_logits(...)` |
| `active=` used outside `tokens="decode"` | Skipped layers on prompt tokens leave only approximate K/V entries that every subsequent token will read | Use `@pllm.custom_forward(tokens="decode")` |
| Stateful `@pllm.custom_residual` (`state=`, `chunk=`) with `@pllm.custom_forward` | Memory advances once per token, whereas custom forward visits each layer a variable number of times | Choose one or the other |
| Two `@pllm.custom_residual` methods on the same sublayer; two `@pllm.custom_attention` methods on the same layer | A sublayer's residual update and a layer's attention read can each have only one owner | Merge into a single method |
| Static KV layouts (`keep="window"` / `"none"`) with `pllm.PositionState` or `@pllm.custom_forward` | Static layouts cause vLLM to partition KV into multiple groups, while both features require a single group | Choose one or the other |
| `@pllm.custom_attention` inside `@pllm.custom_forward` with `kv="peek"` | Read-only layer invocations without KV writes do not yet support custom attention reads | Use `kv="commit"` |
| Two `pllm.Fed` states; reading a fed state directly from a method | Data from an external process enters a state read by `ActivationPatch` only through a single `@pllm.write` | Use one fed state per workload and copy/transform it via a single `@pllm.write` |
| Two `init=pllm.Imported()` states | A single `generate(batch=StateImport(...))` call imports only one state | Merge the fields into a single state |
| A component reading another component's state (state names containing `.`) | A component must behave identically to when it runs standalone | Place methods that need to share state in the same workload |
| One class inheriting from two workloads | Declarations with the same name from the two base classes overwrite each other | Declare them as dataclass fields (components) |
| Two methods reading each other's results within the same step | Results within a step can only flow in one direction | Remove one read, or move it to the next step by reading from state |
| A component requiring a different `publish_lag` than the engine | The entire engine shares a single `publish_lag` | Use the same value across all components |
| Passing `Adapter`, `ActivationPatch`, `@pllm.custom_forward`, or `@pllm.custom_residual` to `pllm.host.make_host` | They operate inside the model forward pass, and host training has no model | Use `pllm.make_llm` |

## Combinations with Engine Settings

| Pattern | Requires | Reason |
| --- | --- | --- |
| `@pllm.custom_attention`, model-writing `@pllm.train_in_forward`, `ActivationPatch` with `last_tokens > 0`, or model edits reading `RequestState` / mutable `InstanceState` | `enable_prefix_caching=False` | Cached KV depends on per-request state, dynamic parameters, or trailing generated tokens; read-only workloads, stateless `@pllm.custom_residual`, and static edits reading only immutable `InstanceState` can safely enable prefix caching |
| `@pllm.custom_attention`, `@pllm.custom_residual` (without `pllm.value` / `pllm.input_tokens`), `ActivationPatch`, dense/low-rank `Adapter` on `FromModule`/`Pretrained`/`Computed`/`Imported`/`Zeros` | Supports `tensor_parallel_size > 1` (vLLM >= 0.32.0) | Executes per local head, residual slice, or projection row/column shard |
| `@pllm.custom_attention` | FlashAttention 2 (or DeepSeek V2/V3 latent reads supporting MLA), unquantized KV; MLA requires `keep='all'` and no `window` buffer; `pllm.attention(...)` fact reads require `tensor_parallel_size=1` | Custom reads directly access paged KV / MLA latent pages |
| `@pllm.custom_residual` fed `pllm.input_tokens` / `pllm.value`, `per_head` adapters, side-stream optimizers/exports | `tensor_parallel_size=1` | Requires a full token/head view or is not closed within local shards |
| `cuda_graphs=True` | No patterns requiring sliced graphs or eager execution (such as `@pllm.custom_forward`) | See [Support Matrix](support-matrix.md#cuda-graphs-and-eager) |

For additional per-model and parallelism restrictions, see the [Support Matrix](support-matrix.md).
