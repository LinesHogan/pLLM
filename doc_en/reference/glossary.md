# Glossary

Terms that appear throughout the documentation and `pllm.explain` output, one meaning per term. Each entry links to the page where it is explained in detail.

## Authoring Workloads

- **workload**: A subclass of `pllm.Workload` containing only declarations and tensor math. Read once when building the engine and scheduled by pLLM thereafter; its methods are not runtime callbacks. See [Workload Anatomy](../guides/workload-anatomy.md).
- **declaration**: The three constructs in a workload that answer "what to read, what to store, and where results go": parameter defaults, state attributes, and effect decorators or model-edit attributes.
- **input marker**: A value placed in a parameter's default specifying where the input comes from, such as `pllm.hidden(0)` or `pllm.parameter("s", "w")`. Type annotations only document what the method body receives. See [Reading Inside the Model](../guides/reading.md).
- **effect**: A decorator specifying where a method's output goes: `@pllm.export`, `@pllm.add_logits`, `@pllm.guide_sampling`, `@pllm.write`, `@pllm.train`, and so on. See [Where Results Go](../guides/results.md).
- **model site**: A declaration acting directly inside the model forward pass: `pllm.model.ActivationPatch`, `pllm.model.Adapter`, `@pllm.train_in_forward`, `@pllm.custom_residual`, and `@pllm.custom_attention`. CPU checks do not execute model sites.
- **setting method**: A method passed to a decorator or input marker instead of a fixed literal, such as `beta=strength` with `def strength(self) -> float`. pLLM calls it once per workload instance.
- **row**: A single entry along dimension 0 of a tensor received by a method: one active request (or one token to be sampled, or one token computed in the step). Row order is determined by the scheduler. See [Core Concepts](../explain/core-concepts.md).
- **selector (`rows=`)**: A method returning a `bool [rows]` mask, passed to an effect's `rows=` argument so the effect applies only to selected rows.
- **component**: A workload held as a dataclass field inside another workload; its names carry the field name prefix, and its states are independent. See [Composing Workloads](../guides/composition.md).
- **custom op**: A kernel wrapped with `torch.library.custom_op` and called inside a workload method. See [Custom Ops](../guides/custom-ops.md).

## State

- **state**: The only mutable construct in a workload: a collection of named tensor fields whose VRAM is allocated and version-updated by pLLM. See [State](../guides/state.md).
- **RequestState / InstanceState**: `pllm.RequestState` maintains one copy per request (or group of requests); `pllm.InstanceState` maintains one copy for the entire engine.
- **`key_fields`**: Determines which requests share a `RequestState` copy: per request, samples of the same prompt, or requests sharing a route (`routes=`). See [Sharing Training Across Requests](../howto/share-training-across-requests.md).
- **field**: A single tensor inside a state. `shape=` declares a single field named `"value"`; `pllm.Field` specifies a projection-scaled shape or an explicit dtype.
- **initializer (`init=`)**: Where a state's initial values come from: `pllm.Zeros`, `pllm.Computed`, `pllm.Pretrained`, `pllm.ModelCheckpoint`, `pllm.FromModule`, `pllm.Fed`, or `pllm.Imported`. See [Loading Trained Weights](../howto/load-trained-weights.md).
- **frozen state**: A state without an optimizer: readable anywhere, and mutable only through import or `@pllm.write`.
- **version**: Each state update publishes a new version; readers observe published versions, while in-flight requests retain their starting version. `pllm.version(state)` gives the number of times each row's state has been updated.
- **`publish_lag`**: An update computed on a step becomes visible to readers starting at the `(1 + publish_lag)`-th step on which the workload actually executes afterward; steps where the workload reads zero rows (such as mid-prompt chunks or recomputed history) are skipped by the engine and do not count. Defaults to 1 to overlap training with the next forward pass.
- **carry field**: A field declared with `carry=` that is published alongside training updates without being updated by the optimizer; written via `pllm.carry`.
- **PositionState (`pllm.PositionState`)**: One hidden-width vector per KV block, read and written via `layers.load` / `layers.save` exclusively inside `@pllm.custom_forward`.
- **placement (`placement=`)**: Whether a read-only large table resides in GPU VRAM (`"device"`) or pinned host memory (`"host"`).

## Training

- **request-row training**: A `@pllm.train` method reads rows of active requests and returns a per-row loss; pLLM reduces per state, runs backward, and updates. See [Online Training](../guides/training.md).
- **reduction (`reduction=`)**: When multiple rows share one state in a step, whether losses are averaged (`"mean"`) or summed (`"sum"`).
- **host batch**: A training batch assembled by host code and passed in via `generate(batch=...)`: `WeightedRowBatch`, `CausalTrainBatch`, or `WeightedCEBatch`. See [Host Loops and Batch Training](../guides/host-loops.md).
- **causal model (`pllm.causal_model`)**: The full model callable inside a training method with specified `Adapter` overlays attached, allowing gradients to flow into adapter state.
- **`HostTrainer`**: The object returned by `pllm.host.make_host`, which trains a workload's state on host batches without loading a model.
- **in-forward training (`@pllm.train_in_forward`)**: Trains a `RequestState` inside the forward pass using a projection's inputs and outputs and adds its delta to that projection (as in TTT / In-Place TTT). See [Model Edits](../guides/model-edits.md).

## Model Edits & Forward Structure

- **ActivationPatch**: Reads values from state and modifies the outputs of selected layers (steering vectors, ReFT).
- **Adapter**: Applies a state as a weight delta onto designated projections (e.g., LoRA), regardless of how the state is written.
- **custom residual (`@pllm.custom_residual`)**: Takes over the residual update at a specific site in selected layers, value rows, or the final stream. See [Customizing Forward Structure](../guides/forward-structure.md).
- **stateful sublayer**: A custom residual with `state=` and `chunk=`, maintaining a per-request carry updated in chunks (fast-weight memory).
- **custom attention (`@pllm.custom_attention`)**: Replaces how selected layers read KV; can also declare a static **KV layout** (sliding window, prompt + window, cross-layer sharing).
- **custom forward (`@pllm.custom_forward`)**: Uses Python to orchestrate which layers each token runs, how many times, and when to stop.

## Execution & Verification

- **planner**: The component of pLLM that reads workload declarations at engine build time, assigns each method's step position and stream, and computes memory reservations; `pllm.explain` prints its plan. See [Inspecting Costs: pllm.explain](../guides/explain.md).
- **admission**: The moment pLLM accepts something. Request admission is when a request enters the engine, receives a state copy, and binds the state version it reads; model or pattern admission checks at build time whether a model or feature is supported, raising an error if not.
- **Inductor**: PyTorch `torch.compile`'s code generator, which fuses adjacent small ops into a single kernel; vLLM uses it when compiling model CUDA graphs, and pLLM uses it to fuse read-only lookups.
- **phase (`pllm.Phase`)**: Time windows within a step: `BEFORE_FORWARD`, `IN_FORWARD`, `AFTER_FORWARD`, `BEFORE_SAMPLE`, `AFTER_SAMPLE`, plus `AUTOMATIC` (chosen by pLLM). Each effect has a default phase that `@pllm.on(phase)` can override.
- **anchor**: Named positions within a step printed by `pllm.explain`: `request.prepare`, `model.execute`, `state.publish`, `sampler.pre_draw`, `token.commit`, and so on.
- **main stream / side stream (`lane`)**: `main` runs serially with the forward pass; `side` overlaps with the next forward pass.
- **eager reason**: Why the model forward is not inside a CUDA graph (`eager_reason`); empty string if captured.
- **neutral configuration (`neutral=`)**: The same workload with zero intervention strength; `testing.check` verifies that it leaves sampling unchanged.
- **contract check**: Properties verified by `pllm.testing.check` (CPU) and `check_on_gpu` (GPU): row-order invariance, request isolation, chunking equivalence, preemption recomputation, version visibility, and neutrality. See [Checking Correctness](../guides/checking.md).
- **budget**: The three fields in `ExecutionOptions` that cap pLLM memory: `state_page_budget_bytes`, `execution_peak_budget_bytes`, and `cuda_graph_pool_budget_bytes`. See [Configuration Options](config.md).
