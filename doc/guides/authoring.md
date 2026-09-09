# Workload Authoring

For a copyable user brief and instructions for your coding agent, start with
[the agent workload guide](agent-workloads.md). This document remains the API contract.

Use `import pllm` for ordinary tensor methods. [Dummy](../../pllm/workloads/dummy.py),
[OverRIDE](../../pllm/workloads/override.py), [In-Place-TTT](../../pllm/workloads/inplace_ttt.py)
and [ReFT](../../pllm/workloads/reft.py) use this interface; [Recirculation](../../pllm/workloads/recirculation.py)
uses its ordinary `Recurrence` declaration. [TTRL](../../pllm/workloads/ttrl.py) uses
`CompletionWorkload`; FTTT and GTTA use the structured `transaction` binding.
[TLM](../../pllm/workloads/tlm/workload.py) uses `CausalAdapterWorkload`.
[ESamp](../../pllm/workloads/esamp_request.py) uses ordinary named parameters, prediction
and loss, with explicit paged/capture replacements on that same graph. Output LoRA
uses ordinary `Batch` inputs and multi-step SGD. All eleven built-ins now use their
applicable shared authoring entry; host transactions are not Tensor/autograd methods.
All construct the original graph and use the same execution owners.
The independent [Huber example](../../examples/huber.py) expresses request-local low-rank
learning with persistent AdamW through ordinary Tensor methods, without core changes.

## Tensor Methods and State

```python
import torch
import pllm

class CenteredHidden(pllm.Workload):
    @pllm.on(pllm.Phase.OBSERVE)
    @pllm.hook(0)
    @pllm.export(name="centered_hidden", max_rows=2, max_cols=16)
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden - hidden.mean(dim=-1, keepdim=True)
```

`hook` declares a supported block/model-output capture, not an arbitrary eager
PyTorch hook. A single plain Tensor parameter inherits that source. Multiple sources
use `Annotated[Tensor, ...]` with `Hidden`, `Samples`, `BaseLogits`,
`SampledTokens`, `Parameter(state, field)`, `StateIndex(state)`, `StateVersion(state)`
or `Result(operation)`.
Ambiguous inputs fail at binding; parameter names are not guessed.
Use evaluated annotations; the binder never evaluates annotation strings.

`InstanceState` and `RequestState` declare a single `shape=` or named fields,
dtype, optimizer and optional initializer. Declare them on the class, instance,
or a typed property when shapes depend on configuration/model dimensions.
Request state defaults to actual request ID isolation. It is not reclaimed by
Python object destruction: original PTTS pins and completion govern its lifetime.
Instance parameters have their declared shapes; request parameters add an
operation-row axis. The ordinary implementation packs these views explicitly;
it is **not native paged GEMM**.
Inputs and parameter views are borrowed read-only values; ordinary mathematics must
not mutate them in place. Shape/dtype declarations do not prove arbitrary Python
side-effect freedom. Optimizer writes still go through original private versions.

`@train(state=..., update=...)` binds a scalar loss and an optional state-eligibility
selector; `@on(phase, rows=...)` independently selects its forward rows. Selectors
are named Tensor methods using a subset of that operation's annotated inputs. They
return boolean masks over its original row axis and become ordinary tasks in the
same signed graph, not execution-time callbacks. The old `train(select=...)` option
is removed; OverRIDE expresses its former training subset with `on(..., rows=...)`.
Without `update`, states represented by the forward rows are updated. An explicit
update mask authorizes the original state of each selected row (any selected row
authorizes a shared state once), independently of forward or gradient eligibility.
Empty update selection preserves every field and version; authorized zero gradients
still advance Adam step, momentum and decay, including an empty-forward invocation.
The scalar function owns its reduction; the binding does not invent normalization.

`@train(state=..., reduction="state_mean")` instead returns one mean loss per original
selected state. The binder differentiates their **sum**, not a mean across states, and
stages loss and optimizer as separate tasks in the original graph, preserving the
declared timing. This form currently requires unconditional forward/update domains.
`StateIndex` gives each row's invocation-local state index for reductions;
`StateVersion` gives its original input-pin version, never latest or optimizer step.
Both are nondifferentiable. Explicit `RequestState.keys` remain the original isolation
keys; omitting them uses request ID. ESamp keeps its request-ID-only backend restriction.

`@guide_sampling(beta=...)` returns post-filter hidden predictions, optionally selected
by `@on(..., rows=...)`. The binding associates them with original sampler links,
including reordered links; it does not require vocabulary dimensions. ESamp uses this
for version-aware guidance and ordinary Tensor loss/initialization. Its five-field
Adam layout and initialization seed progression are unchanged.

`@add_logits` and `@train` connect pre-filter addition to actual sampled feedback.
`@export` owns bounded, stride-gated CPU output; `Config("config.layer")` and
similar attribute paths bind immutable configuration at loading time.
No `build`, input dictionaries, `prepare/output`, field IDs, digests or explicit
publication nodes are needed in ordinary workloads.

A local function replacement is just `dataclasses.replace(workload, transform=fn)`;
see the [runnable example](../../examples/centered_hidden.py). Users do not assign
implementation IDs. The cold binder signs code, checked dependencies, configuration,
initializers and replacements, and snapshots Python functions for the bound instance.
Reflective mathematics or unsupported dependencies require explicit native binding.
Custom Python definitions remain in-process; EngineCore rejects replacements rather
than silently reconstructing default mathematics.

Only `pllm/usage.py` allows cold authoring reflection/type erasure. Strict typing
still covers that module, and runtime execution does not inspect signatures.
The previous `TensorWorkload`, `TensorBuilder` and `TensorFunction` public API is
removed. Arbitrary module anchors, capture and structured host transactions still
require advanced bindings.

`@differentiate(inputs=("features",), parameters=False)` keeps selected input
derivative edges while freezing parameter derivatives at that operation. Upstream
methods use `parameters=True`, and `@train` consumes their `Result` inputs through
the declared edges. Other inputs are detached. This is first-order autograd inside
one invocation, with one optimizer state and one training sink, not full-model
backpropagation through vLLM or across replay/publication boundaries.
The same shared state may be differentiable at another operation; `parameters=False`
does not permanently freeze its optimizer. `train(state=...)` authorizes that state's
update, including its existing Adam step/momentum/decay semantics.
Gradients of explicitly marked external floating inputs are returned as
`result.input_gradients[(operation, argument)]` in that operation's row order.
Intermediate result edges propagate derivatives without a separate gradient DTO.
`differentiate(input_rows="dx", parameter_rows="dw", ...)` independently masks the
declared derivative edges without changing the forward values. Selectors use original
rows, not positions after forward compaction. For shared state, row-specific parameter
derivatives require `Parameter(state, field, per_row=True)`, a logical
`[rows, *parameter_shape]` view consumed by row-aware math such as `bmm`. Request
parameters already have this row axis. Unbatched shared parameters with a parameter
gradient mask are rejected; no hidden shape conversion or alternative backward runs.
Forward-selected non-loss methods preserve the row axis and accept empty batches;
their result is restored to original row positions with zero values outside the domain.
Loss methods receive only selected rows and are not invoked for an empty forward set.
Bounded exports retain their own row limits and do not accept these row selectors.
Sampled-token predicates execute only after sampler completion and before the signed
update; earlier feedback reads, effectful helper reads and post-update reads are rejected.
`result.parameter_gradients` maps original input `StateRef`s to flat gradients, even
for states excluded from publication. Flattening multiple fields is an explicit
inspection cost; `result.parameter_gradient_fields` borrows the original owned
gradient tensors in declared field order, without eager per-request concatenation.
Captured results retain independent storage across later replays. Loss work without publication is counted on
successful completion. All derivatives remain invocation-local and first-order.
This is ordinary eager Tensor execution, not sparse backward or native paged lowering:
forward selection precedes request-parameter packing, and shared per-row parameters
retain their zero-stride view. Consecutive forward rows borrow input/mask slices;
irregular rows pack. Full-row results need no scatter; partial results use one zero-filled
destination to restore original positions. Derivative masks act on a separate PyTorch gradient edge
per consumer, without a parameter-sized forward copy or changing another consumer's
gradient. The original selection owns its shared read-only identity index.
Result tensors share the invocation's original `(task, output)` store. Rust plans their
last use, including downstream declared gradient consumers, and signs release IDs on
the original physical tasks. Python only applies these releases; exported values survive
through completion. There is no separate TensorFrame result cache or Python liveness planner.
Compaction can synchronize and autograd can still save intermediates for detached rows.
Full scratch/saved-tensor planning
and selective-backward fusion remain pending; no capture or allocation-free claim.
Completion/cancellation releases invocation data and any retained backward graph;
failed settlement retains it only until the original receipt can be settled.

## Request Overlays

`@on(Phase.MODEL_UPDATE)` and `@overlay(state=..., targets=..., weight=...)` bind
a dense additive overlay and its publication together. Declare `RequestState`
without an optimizer: fixed `shape=` uses the default `value` field; named `Buffer`
fields can resolve `(input_size, output_size)` for each target. Buffer dtype defaults
to the target model dtype; explicit integer buffers retain their own dtype.
`Operand.INPUT/TARGET/WEIGHT` name captured tensors; `Parameter(state, field)` names
private buffer views. Only these authorized buffers may be mutated, never the operands.
The method returns `TensorUpdate(output, changed=..., update_count=..., trained_rows=...)`.
Optional `metrics` belong to the algorithm, not the state or replay protocol.

`prefill`/`decode` determine update eligibility. Disabled phases only apply the
existing overlay. `replay=ConditionalLayerReplay(...)` reuses the original published
version and runs only for `changed` states: tail-buffer changes alone need not replay.
The shared binding owns target coverage, shape/device checks, root selection and
result provenance. Results expose `entries` with original request/before/after refs
and per-target updates. The current ordinary effect supports request DENSE_ADD,
not low-rank, activation or arbitrary replay composition.

## Activation Checkpoints

Declare ordinary `InstanceState` or `RequestState` buffers and an `Activation`
property naming the state, layers, first/last prefill tokens and supported operator.
`LOW_RANK_AFFINE` uses weight/bias/rotation fields per layer; `ADD` uses one FP32
vector. Buffer shape callbacks receive the activation input/output widths.
Supply the original `pllm.StateTensorUpdate` as `auxiliary=`. The shared binding
validates the checkpoint, selects roots, copies private fields and publishes before
model execution. Instance replacements may repeat; request replacements use original
prompt keys and import only into version-zero request state. Missing updates do not
publish, and existing inference pins continue to see their exact old versions.
See the maintained ReFT workload; it no longer implements prepare/copy/publish tasks.

## Tokenwise Recurrence

Declare `pllm.Recurrence(source_layer=..., destination_layer=..., alpha=..., beta=...,
ramp_tokens=..., enabled=...)` as a class/instance attribute or typed property on a
`pllm.Workload`. Negative layer indices resolve against the actual model. `beta=None`
uses the complementary destination weight; the source is norm-matched before mixing.
This is the existing standalone tokenwise prefill effect: normal logits are preserved,
replay updates the selected token's layer KV, and decode rows only run normally.
No RequestState, hidden capture, external training input, build or prepare method is needed.
Mixing this effect with ordinary training/activation operations is not yet supported and
fails during binding; host-only execution has no model replay capability.

## Weighted Batch Training

[Output LoRA](../../pllm/workloads/lora.py) declares named state fields with `SGD(lr)`
and an initializer, a normal `@add_logits` prediction, and an `AFTER_SAMPLE` method
with `@train(state=..., reset="reset", steps=Config(...), reduction="weighted_state_mean_then_mean")`.
The reset method receives named local parameter tensors and a generator. Reset is
per training batch, distinct from admission initialization; both functions are frozen
and signed with the ordinary math. SGD preserves raw state fields, without Adam slots.

Training inputs use `Annotated[Tensor, Batch("hidden"|"base_logits"|"labels"|"weights")]`
and `Parameter(state, field)`. They consume the original `WeightedCEBatch`, not a new
DTO. Shared inputs use `pllm.WeightedCEBatch(hidden, logits, labels, weights)`;
independent inputs use `pllm.WeightedCEBatch.for_prompts(hidden, logits, labels,
weights, prompt_indices)`, with the same indices supplied to `generate` and optional
`sample_idx=0`. Both borrow the supplied tensors without copies. The binding aligns
the original keys, rejects out-of-envelope inputs, and masks
nonpositive weights/out-of-vocabulary labels. Each selected state must have a valid row.
Training tensors have a leading selected-state axis, including a singleton for shared
state. Return one normalized loss per state; the binding differentiates their **mean**
on each SGD step. This is not ESamp's independent per-state Adam reduction.
Missing batch means no reset, private write or publication. Successful training publishes
once after all steps; cancellation/failure preserves the original pinned state.
Normalization/reset runs in the original graph's `__train_prepare` task before sampling,
not during invocation preparation. Its result edge feeds `train`; the same frame owns
the normalized tensors until consumption, then releases them. LoRA signs preparation,
prediction and training workspaces plus prepared/output tensor sizes, using one formula
for specialization and binding checks. Undersized signed facts fail binding.
The preparation task signs both `[states, rows, hidden]` and `[states, rows, vocab]`
input bounds. Public submission, feedback transfer and worker STAGE preflight the
original auxiliary tensors/descriptors before shared-storage export, mapping or device
copies; dtype comes from the original state layout, with int64 labels. Shape checks
do not inspect tensor contents or replace the later finite-value/selection checks.
Reset training and activation imports fully overwrite their private fields, so their
bindings use `copy_fields=()` to skip copying old fields into new pages. The default
`copy_fields=None` copies all private fields; an explicit ordered subset copies only
those fields. Built-in Adam bindings preserve parameters, steps and moments but not
old gradients, which the update fully overwrites (including every TLM factor group).
There is no separate `replace` execution mode. This uses the same arena, pins and
receipts; callers must fully overwrite omitted private fields before publication and
must not read their uninitialized contents. The private-page capacity is unchanged.
This entry currently supports reset SGD and weighted batches, not persistent momentum,
arbitrary external DTOs, inference-row selectors or cross-step autograd.
`SamplerWorkload` and `LoRAData` are removed; there is no parallel native authoring path.

Ordinary `Workload` also exposes typed `implementations` and `operations` hooks for
explicit native replacement of its original signed tasks, not another graph/runtime.
ESamp and LoRA use them for their existing capture and paged kernels. Ordinary
math can be replaced locally; incompatible replacements are rejected when selecting
the specialized kernels (`native_methods` freezes reference code/source/effect fingerprints;
`require_native_methods` checks them at binding, including in-place annotation edits).
This is not automatic lowering of arbitrary modified math to a specialized kernel.
Inherited mathematical `resource_facts` formulas are specific to the built-in math. A
modified method must supply its own cost declaration instead of silently inheriting
that envelope, even on the ordinary tensor path. The check runs at specialization
and binding; an explicit cost override does not authorize an incompatible native kernel.

## Advanced Dataflow

For custom advanced dataflow, subclass `a.BackendWorkload[Data, Result]` and return
the original graph/layout/attachment from `declare`, with native `bind_math` and
`backend_requirements`. The old `a.Workload` build wrapper is removed.
New local operations receive workload/name-based IDs unless an explicit implementation
is bound. Persistent layout namespaces and implementation identities are not mechanically renamed.

`Call["output"]` names an original task output. It establishes a dependency, not a
Python execution order. `When` names semantic timing windows; Rust still chooses
anchors and ordering. Input, state and output names must describe the mathematical
functions' actual accesses. No function-body inspection or automatic tracing occurs.
The typed `Data` record holds invocation-local intermediates used by those functions;
declaring a port does not automatically extract or assign a Python attribute.

Hidden requirements are derived from the declaration. Shape/dtype callables run only
after model facts are available; requirement discovery does not guess dimensions or
allocate parameters. `state.read()` is a logical input-version reference.
`state.version(publication)` explicitly reads a producer's version, never latest.

## Write Mathematics

Advanced `BackendWorkload.bind_math(facts, plan)` returns the original `MathBinding`.
Its `FunctionMath` supplies initialization, input row selection, preparation and
result construction; no executor is constructed by the workload. Structured authoring
entry points assemble this binding themselves.

Native functions receive typed `Data` and `TensorBatch`. The latter holds original
selection/provenance and borrowed tensor operands, not an arena or publication API.
Read operands are immutable by contract and must not escape their invocation.
Only explicitly selected transition roots are writable. Stream ordering, capture,
private allocation, rollback and publication remain in the original execution layer.

Return `TaskOutput` with mathematical gradients, guidance and effective-work counts.
An empty gradient tuple skips an optimizer update; a tuple of zero gradients still
updates step, momentum and weight decay. For multiple parameter groups, gradients
are state-major, then declared parameter-group order, with one publication for all
groups. Direct transitions mutate authorized private views and use a separate
`b.publish`; replay consumes that original publication and the changed-state selection.
Optimizer-backed transitions also use `b.publish` when publication must be deferred,
as in TLM and TTRL. No private allocation or publication occurs for skipped gradients.

`InvocationOutput` exposes the original guidance and publication, allowing result
construction without lifecycle ownership. `FunctionMath` is generic over input/data/
result types: completion and host payloads are not converted into a tensor-only DTO.
Host transactions continue to use the existing host service and receipt protocol,
not a second state database or a model-inference callback.

For structured host transactions, use `pllm.transaction(name, facts, config=config,
state_factory=make_state, service=service, update=update, query=query, publish=publish)`.
The three ordinary typed functions receive the same original context. The shared binding
declares update-before-query-before-result dependencies and the host timing windows;
Rust still schedules them. `publish` only constructs the result: the original coordinator
alone calls service commit and settles its receipt. State and service objects are retained
by identity, with their existing prepare/snapshot/restore/output and begin/commit/abort
contracts. No TensorFrame conversion, copied queue or second state owner is introduced.
These are explicit native function bindings, not automatically differentiated or frozen
Tensor methods; changes to their native helper implementations require rebinding.
See [FTTT](../../pllm/workloads/fttt/workload.py) and
[GTTA](../../pllm/workloads/gtta/workload.py), which no longer handwrite a graph, port/schema
dictionary or implementation IDs. `declare_host` still supplies device checks and the
state factory. More general host dataflow can use the advanced builder.
The original host executor calls `state_factory` only after Rust admission, so a
finite-budget rejection does not first allocate state. Advanced declarations can pass
their original `timeline`; it cannot loosen `HostRuntimeFacts.memory_budget_bytes`.
FTTT supplies finite tensor costs with `max_input_rows` and `memory_budget_bytes`:
parameters/moments, the rollback snapshot, active hidden input, feedback features,
training/query workspace and result storage. Input cardinality is checked before the
snapshot. Queue storage, text/judge/Python allocations and library scratch are outside
this per-execution tensor budget; GTTA has no complete Python-heap cost declaration.

`Observation` carries selected row or identity provenance to the existing trace.
Explicit bounded exports use `export=True`; optional mathematical observations still
follow the original trace selection. Neither requires access to the executor or arena.

## Completion Learning

Subclass `pllm.CompletionWorkload[Data, Result]` for instance-scoped learning from
the original `CompletionInput`. Declare `parameters` as an `InstanceState` with
named shapes, an optimizer and optional initialization. `prepare` computes the
algorithm's structured context; `loss(data, parameters)` receives named tensor views
and returns a scalar, or `None` to skip the entire optimizer update. `trained_rows`
reports the effective work of a returned loss. Store only detached diagnostics in
the context. `output(data, invocation)` receives the original `InvocationOutput`;
it may expose the original publication and state reference without owning either.

The binding handles input shape/device/readiness, Adam fields, autograd, private writes
and publication. No build, tensor-field indices or manual gradient collection is needed.
Named parameters share one instance root and retain their own Adam fields. A zero gradient
still advances optimizer state and publishes; `None` leaves parameters, moments and step
unchanged. There is no request state, cross-invocation backward or automatic capture here.
Initialization and loss use the cold tensor dependency contract; structured prepare/output
helpers remain explicit native host math. Scalar configuration is snapshotted at binding,
including class attributes, so a later monkey patch only changes newly bound executions.
TTRL is the maintained example: majority voting, tie rules, sampling limits and retry/barrier
construction remain its mathematics, with the original completion objects preserved.

For bounded TTRL tensor execution, pass `HostRuntimeFacts(device, hidden_size=...,
max_input_rows=..., memory_budget_bytes=...)` to the same public factory. The built-in
formula includes the original hidden buffer, selected rows, head/MSE/Adam workspace
and retained results in the original SchedulerTimeline. The row cap is checked before
preparation; CUDA inputs record their consuming stream. Modified prepare/loss/output
math must provide its own `resource_facts(facts)` instead of inheriting that formula.
The returned timeline cannot erase the user's limit; missing task/output costs fail
finite Rust admission. With no budget, ordinary behavior is unchanged. This is not a
bound on completion text, Python metadata/history or external-library scratch.
Host declarations retain their original `HostRuntimeFacts`; advanced callers use
`binding.compile()`, not a second set of facts at compilation.

## Causal Adapters

Subclass `pllm.CausalAdapterWorkload[Data, Result]` with a frozen `pllm.CausalModel`,
an optimizer and `phase` set to `BEFORE_MODEL` or `AFTER_SAMPLE`. It uses the same
prepare/loss/trained_rows/output mathematical contract as completion learning.
`prepare` receives the original `CausalTrainBatch`, or `None` when no training input
was supplied. The named loss parameters follow the model's ordered A/B factor pairs.
The shared binding derives Adam fields and LOW_RANK_ADD attachments, validates target
shapes/device/dtype and causal inputs, and publishes atomically at the declared phase.
TLM retains its q/v restriction and detached sentence-selection objective; there is no
TLM-specific layout, gradient collection or publication implementation in the workload.

The base model and initialization templates must remain immutable. They are retained
by identity, not deep-copied with workload configuration; live factors are the original
PTTS views. The model's fingerprint, resolved optimizer/scale and loss enter the signed
declaration. A meta model may declare against the worker's actual device facts, but
executable mathematics require the materialized model on that device. The built-in
TLM definition supports closed EngineCore reconstruction without initializing parent CUDA.
This interface is instance-scoped and does not add request adapters, capture or replay.
`CausalModel.resident_bytes` declares frozen parameter/buffer storage outside PTTS.
`HfQvModel` derives it from unique tensor storages, including meta models: tied weights
and the borrowed factor templates are not counted twice. The original backend probe
adds it to every timeline anchor alongside input buffers, and worker reconstruction
checks the materialized requirement. This is tensor residency, not allocator overhead
or backward/library scratch. HF calls explicitly disable KV, hidden-state and attention
output retention; the causal objective only consumes logits.
An explicit training cost declaration must include `update.input_shapes=((sequences,
tokens),)`. Both axes are capacity bounds, checked before transport and before causal
math; they are independent of packed inference rows. The standard HF/PEFT TLM binding
still has no complete model-backward scratch cost declaration: its unbounded mode is
not a finite-memory guarantee, and Rust rejects missing costs in a finite plan.
Offline TLM reduces the frozen model's logits to detached sentence losses before the
adapted forward. Only this vector survives, not the full batch/token/vocabulary tensor.
The objective lives with TLM mathematics, not in the shared primitive catalog.

Closing a model handle detaches and shuts down the engine, then drops its runner,
execution and engine references. Caller-owned models and returned tensors retain their
own lifetime. A failed shutdown closes the handle to new work but retains the engine
and any unacknowledged shared-input scope; `close()` can retry, releasing that scope
only after shutdown succeeds. Retaining a closed handle does not itself retain the
successfully shut-down engine.

## Gradients and PLLM Primitives

Backend-bound input uses the original probe's `max_packed_rows`: the limit counts all
packed model tokens, including prefill, not selected decode rows or distinct requests.
Over-capacity input is rejected before state selection; staged submission cancels its
existing invocation on rejection. vLLM checks before opening the step or capture.
Stateless backend observers also enforce the signed hidden width. Auxiliary input has
its own preflight against original task input bounds or checkpoint field layouts and
state scope. Request checkpoint inputs may include queued requests beyond simultaneous
arena capacity; the original arena still admits each active cohort within its entitlement.
Missing task costs still mean unbounded execution, not inferred safety;
shape/dtype checks do not prove arbitrary user-math scratch costs. Host-only observation
has no backend probe.

The [PLLM design](../pllm-next-stage-design.md) distinguishes ordinary mathematical
execution from explicit high-performance primitive binding. `compute` is an opaque
native operation; `a.ops` contains maintained primitive bindings. The current catalog
contains gated prediction/loss/update implementation
candidates and the existing low-rank loss/AdamW capture binding. It is not a universal
operator library or automatic lowering of arbitrary autograd programs.

`MathContract` lives on the original SemanticTask and is included in its signature.
It separately declares input gradients, parameter-state gradients, detached inputs,
forward/input-gradient/parameter-gradient row domains, update-state domain and reduction.
Domains name original input ports; native math performs its explicit filtering within
those domains, and the transition selector identifies actual writable states.
Rust rejects nonexistent ports/resources and contradictory detach/gradient declarations.
Changing reduction or demand invalidates the old signature. These checks do not prove
that arbitrary Python math implements its declared derivative or reduction.

Native input gradients must be computed explicitly and carried by the declared output/data
boundary; ordinary first-order methods use `differentiate`. Freezing parameter gradients does not imply detaching inputs. Shared loss
reduction and per-request reduction are different algorithms, not scheduling choices.
Output LoRA's independent path uses per-adapter weighted normalization followed by a
mean across adapters; it must not become one global weighted mean during migration.

Explicit tensor and paged math now share `pllm.TensorRows` / `pllm.PagedRows` operations:
`linear`, `input_gradient`, `parameter_gradient`, `values`, `reduce` and `write`.
`TensorRows(packed_fields, row_state_indices)` uses explicit tensor packing; an integer
second argument means contiguous equal-sized row groups and uses batched GEMM without
per-token matrix expansion. Negative tensor indices are inactive padding. Writes require
caller-owned private fields; `PagedRows` continues to require the original arena-issued
write capability. Neither object owns logical versions, publication or reclamation.
ESamp uses one residual/loss/backward expression for both implementations. Output LoRA
uses one explicit reset/weighted-CE/multi-step-SGD loop for tensor/paged eager and capture;
its former separate `weighted_lora.py` implementation has been removed. Ordinary Tensor
methods declare the graph and inputs; these explicit native replacements bind the same graph.

Logical parameters remain in the original graph until binding selects an implementation.
The tensor implementation borrows or packs views; its captured training still performs
arena gather/scatter. **Capture does not mean direct page-table GEMM.**
ESamp and shared/independent Output LoRA additionally accept `paged=True` in their
configuration. That signed implementation uses a shared Triton grouped kernel directly on
the original FP32 matrix pools, including input backward and logical-state parameter
gradient reduction. CUDA is required (verified on SM89, PyTorch 2.13, Triton 3.7).
Arbitrary positive matrix dimensions and strided pool slices are supported; IEEE FP32
is used rather than TF32. Bias/norm vectors still use row reads.
The default tensor implementation remains explicit, not a fallback after paged failure.
The previous library grouped call, which synchronized offsets to host on SM89, has been
removed. Public traces confirm device-side grouping without DtoH for the ESamp loss.
Specialization keys use layout/direction and two reduction-capacity buckets, not request
identities, versions or individual row counts. Small reductions use direct FMA instead
of padding to a full dot tile. Long inner dimensions with narrow outputs use a shared
split-inner kernel and FP32 partial reduction, reading the same authorized slot/active
mapping directly. No workload-specific GPU implementation or per-token matrix copy is used.

Advanced `batch.paged(fields, row_state_indices)` returns one view per field, borrowing immutable invocation-local
mapping operands from the original exact pin or private write. `linear`, `input_gradient`,
`parameter_gradient` and `reduce` share those mappings; matrix offsets/shapes are logical
field slices, not physical slot choices. `parameter_gradient(inputs, gradient, selected)`
can return only selected logical states, in that device index order. Indices must belong
to the original mapping; this does not grant write authority. Paged reduction writes the
compact result directly, without an all-state gradient followed by a gather. Omitting
the selection retains the full logical-state result. `PagedRows.select_rows(fields, mask)` restricts
computation using a device boolean mask over the original rows. Separate selections
can feed forward, input gradients and parameter gradients; matrix kernels do not load
excluded row operands, including NaNs, and their outputs/gradient contributions are zero. All selected
fields share one stable filtered ordering. Selection cannot re-enable excluded rows or
grant writes: update eligibility still comes from the original private PageWrite.
This is explicit native binding, not automatic lowering of ordinary `differentiate`.
Filtering uses one prefix-scan launch up to 16384 rows; larger capacities use fixed-size
tiles, block-count cumsum and scatter. It owns an int64 row order, a boolean active mask
and int32 state ends, plus one int32 count per tile for large inputs; no tensor operands
or parameters are gathered. These allocations remain part of signed workspace/capture
residency. Empty rows, strided masks and 65537-row dynamic capture are covered.
Paged primitives reject `requires_grad` operands instead of silently dropping
an ordinary autograd graph. Mappings cannot be used after release/abort or on another stream.
Eager fields share one row ordering, host-computed state ends and one field-slot table
per borrow. Fully private fields reuse that table for writes; mixed access retains the
read-only slot mask. Scalar fields, including Adam steps, use the same page views.
ESamp's captured loss instead derives its
device mapping from the original arena's stable slot table, with original cross-stream
ordering before replacement and fixed-capacity padding excluded from loss/gradients.
Captured fields share one logical-state grouping. Matrix forward and derivatives consume
the original row order directly inside the kernel, without physical-page sorting,
inverse mappings or explicit operand/output row gathers.
Page slots stay indexed by logical state, not duplicated per row and field; matrix
kernels dereference row/state/slot directly. Vector bias/norm reads still gather.
ESamp's explicit LayerNorm derivative calls the same native PyTorch backward operator
without constructing an autograd tape; raw inputs, mean and inverse standard deviation
remain necessary scratch. The ordinary user function retains normal autograd semantics.
It does not gather matrix parameters. `PagedRows.adamw(fields, gradients, config)` accepts
five original FP32 fields in parameter/gradient/step/first-moment/second-moment order and
contiguous `[states, parameters]` gradients. Only original private slots may be updated;
filtering computation rows does not change write authority. ESamp paged eager and Adam
capture share one step/bias kernel and one parameter/gradient/moment kernel. Eager still
constructs a launch slot table; its known workspace is rounded to 96 bytes per state
(80 slot-table bytes, 4 state-end bytes, 8 correction bytes), excluding allocator overhead.
Capture uses its original stable slot table and charges graph-pool reservation.
Ordinary eager AdamW remains the original PyTorch implementation; a library fused switch
did not preserve the established normalized ESamp tolerances and is not selected.
Shared/independent LoRA also supports paged write capture: reset and SGD write directly
to original private fields through `write(values, selected, add=..., scale=...)`.
`selected` is a distinct subset of the original authorized state indices; values retain
that order. Read-only slots cannot be mutated. Eager and captured training share the
same math and weighted reduction; the capture's absent slots use -1 and never write.
The write kernel reads selected states and original slots directly, including strided
selection, without an intermediate selected-slot tensor. Scaling applies to both
replacement and additive writes. The explicit LoRA loop reuses log-softmax storage
for CE derivatives and hoists fixed row normalization out of its SGD iterations.
LoRA excludes zero-weight rows in its existing captured row-state input; eager filtering
uses the already-computed effective row count and does not regroup fully active batches.
Direct `arena.capture_kernel` restores caller-owned private fields before returning.
Framework `KernelCapture.bind` uses disposable admission pages and aborts them after
capture, without allocating field backups. KernelCapture input factories accept a device;
they must produce the same distinct-buffer shapes/dtypes on `meta` and CUDA. All captures
are sized on `meta` against the original signed resident budget before allocating any
capture input buffers. Real input storage and field scratch are checked again before
cold admission; graph-pool reservation is
checked after capture, with cold pages reclaimed on rejection. Actual replay, cross-stream
ordering, publication and abort remain under the original arena, with no field-gather buffer.
`resource_facts(facts, options)` returns a SchedulerTimeline derived from `facts.timeline`,
with original task/value resource facts, not a separate cost graph.
`ExecutionOptions(memory_budget_bytes=...)` constrains that original AER plan. The original
timeline, returned resource facts and user limit are intersected: no declaration can
loosen or erase an existing limit. `None` adds no limit; zero is a zero-byte limit.
Finite planning requires explicit costs for every task and output, including zero-byte
outputs. The limit covers declared state, resident, reserved, workspace and live-result
bytes, not undeclared vLLM model/KV, allocator or external-library costs. It is distinct
from arena capacity (`state_memory_bytes`) and per-task capture capacity.
ESamp also declares
ordinary tensor packing/saved intermediates, the predicate, Adam workspace and exported
loss/gradient storage. ESamp and LoRA reuse their declaration formulas when rejecting
underfunded task/output facts, and enforce the signed row/state capacities at invocation.
Dummy, OverRIDE, plain In-Place-TTT and ReFT additionally provide built-in task/output
costs when a finite budget is requested (or task costs already exist). Unbounded ordinary
usage remains unchanged. Their worker bindings recompute the minimum costs from original
facts, rejecting even re-signed underfunded claims. Dummy accounts for full-width irregular
row packing before bounded export; OverRIDE covers bounded head/CE/Adam tensors;
In-Place-TTT covers tails, chunk outputs and dense updates, enforcing total operand rows
per target; ReFT checks finite content within the budgeted update before copying fields.
Conditional In-Place-TTT and Recirculation model replay, full HF backward and external
library scratch have no built-in complete cost declaration. Finite use of these paths
requires an explicit declaration; none of these formulas is a total GPU-memory guarantee.
Custom declarations own the cost claim for their changed mathematics; neither a signed
graph nor this inheritance check proves arbitrary autograd/library scratch bounds.
Native source/target packing occurs within the consuming task;
prediction packs only eligible sampler rows. Training results are exported graph values,
so AER retains their storage through completion, not just until optimizer consumption.
Consecutive hidden rows borrow the original snapshot, including strided columns; irregular
rows still pack. vLLM creates one owned FP32 snapshot per resource and reserves its maximum
capacity across the original timeline; worker reconstruction preserves that timeline and
rejects lower snapshot reservations. Readiness orders production before consumption, while
recording the consumer stream prevents allocator reuse before its reads finish.
The existing backend requirements project sampler stage/feedback from the original graph.
vLLM reserves known projection/conversion/row-arithmetic buffers at the sampler anchors,
and owned feedback logits/tokens through completion. Probe and attachment installation
check the same minima using actual sampler capacity and padded head width; installation
reads requirements from the original signed graph. Full-row addition/reweighting avoids
gather/copy-back, while partial rows preserve exact indices and finite-support semantics.
These reservations do not cover upstream sampler/GEMM scratch or the model/KV pools.
Specialization also derives effect-buffer reservations from the original attachment,
field shapes and replay tasks. The conservative envelope retains dense operand snapshots,
low-rank/activation intermediates, replay carry/current-slot KV backups and normal outputs
across the timeline; installation rejects an underfunded re-signed plan before hooks.
The Qwen2 adapters update their fresh projection outputs directly, never state pages or
residual inputs. Contiguous low-rank/replay rows borrow views; dense training captures
still own their inputs. Tokenwise replay shares one immutable sequence tensor across
layers and retains normal outputs without cloning. Its per-layer KV backup still precedes
the normal token; conditional replay backs up the successful normal token instead.
Weighted-CE preparation borrows all-valid ordinary inputs; invalid rows require masked
copies, and inference-mode tensors require owned ordinary storage for autograd.
Ordinary request-parameter reads borrow a broadcast view when all selected rows use
one original state, including singleton/empty selections. Mixed states still pack;
backward sums repeated-row contributions into the same original derivative leaf.
Those leaves alias pinned parameters, not a second parameter allocation. Consecutive
sampler-feedback rows likewise borrow their owned feedback buffer; reordered rows pack.
These inputs are read-only to user mathematics. Settled contexts release guidance/feedback;
returned results keep their original tensors, and failed settlement keeps its retry data.
At the sole ordinary training sink, derivative leaves and retained result graph references
are released after success, including the no-training branch. Exported tensor storage,
computed gradients and staged optimizer updates are retained; this relies on the existing
requirement that every differentiated path reaches that sink.
These conservative known-tensor envelopes are not library scratch guarantees or bounds
on arbitrary customized autograd. Unsupported capture is rejected at specialization.

Keep semantic execution, direct paged access, and capture/fusion as separate capability
claims. New optimized primitives follow measurement across reusable scenarios, with
the independent reference fixed before changing the implementation.

## All-Workload Coverage

| Workload | Declaration and mathematical requirements |
| --- | --- |
| Dummy | No state; selected hidden rows, result and bounded observation |
| OverRIDE | Instance heads; pre-filter addition, actual sampled feedback, conditional Adam update |
| ESamp | Request keys; two hidden inputs, version-aware guidance, per-state mean loss, eager/captured update |
| In-Place-TTT | Multiple targets and tail fields; direct transition, publication, optional new-version replay |
| TLM | Checkpoint/model facts, causal input, grouped q/v fields, atomic publication, before-model offline vs after-sample online timing |
| Output LoRA | Instance/request scope, original keys, external weighted CE, separate training/update selection, SGD and capture |
| ReFT | External field replacement without optimizer, publication, activation targets and selected rows |
| Recirculation | Stateless tokenwise replay; ordered prefill, original KV/RNG and normal output |
| FTTT | Completion/judge input, update-before-predict, original transaction, receipt and retry output |
| GTTA | Host state without tensor fields; FIFO trajectories, update-before-query, abort/requeue and commit |
| TTRL | Structured completion/reward input, conditional gradients, publication and combined retry/barrier result |

`Builder.input` names existing typed service ports; `state` without fields describes
host state without allocating a tensor arena. Host graph declarations continue into
the existing `TensorHostBinding` or `TransactionHostBinding`. Overlay/activation bindings
and the two original replay effects remain distinct. Expressibility is not support for
arbitrary combinations: backend capability checks still reject unsupported compositions.
All entries in this table execute declarations built through this interface. Their
existing mathematical oracles remain independent of the builder; arbitrary cross-workload
compositions are not implied by this migration.

## Evidence and Boundaries

Human authors and coding agents use this same contract and the maintained source examples.
Do not change the independent reference to match an optimization. Extend existing evidence
for outputs, gradients, optimizer state, no-update, isolation, cancellation and settlement;
do not recreate a workload/runtime matrix. Public benchmarks use the same factory.

Use the original bounded trace for explicitly selected observations and kernel timing.
No reflective hooks, duplicate profiling database, runtime agent service or generated-code
download is required. Custom Python workloads remain in-process; EngineCore uses explicitly
implemented closed definitions, never arbitrary function serialization.

Evidence 47 checks the independent Huber adaptation against a frozen PyTorch reference,
including CPU/CUDA gradients, all Adam fields and lifecycle. It claims ordinary eager only.
Remaining optimizer fusion, full resource auditing and release/performance research are
not implied by that experiment or by the completed shared mapping replacement.
