# WorkGraph Phase 1

This guide tracks the typed WorkGraph front end that builds on the frozen
Phase 1a compatibility shadow. WorkGraph remains a control-plane artifact;
`DispatchPlan` and `ExecutionPlan` are still authoritative for execution.

## Milestone 1: Public Contracts

The public API exports `flow`, `extern`, and `BeforeNextUse` from both `tllm`
and `tllm.workgraph`.

`@flow` is transparent during normal Python calls. Decoration and normal calls
do not capture, compile, or inspect a graph. Only `.capture(...)` enters the
control plane, and it validates the original function signature without
executing the function.

The current milestone deliberately produces a conservative opaque artifact:

- `CapturedFlow.status` is `opaque_barrier`;
- `node_kind` is `barrier`;
- mobility is `pinned`;
- graph safety is `eager_only`;
- effects contain `opaque`;
- a source-aware typed diagnostic explains the fallback.

This is the required failure behavior. Milestone 2 replaces it only when
`torch.export.export(..., strict=True)` and the effect analysis both prove a
complete leaf graph. There is no `strict=False` success fallback and no
partially captured graph.

The frozen public descriptors are `CapturedFlow`, `FlowCaptureDiagnostic`,
`FlowCaptureCode`, `SourceSpan`, `ShapeGuard`, `ArtifactKey`, and
`PortBindingDesc`. `ArtifactKey` is separate from `NodeKey` and `ExecutorKey`;
it uses callable module and qualified name, never an object address, source
line, Tensor representation, runtime handle, or request id.

`PortBindingDesc` maps a structured `PortId` or `PortAddressDesc` to one typed
input/output pytree path. Raw locator strings and duplicate leaf mappings are
rejected. The closed-world structural serializer accepts these descriptors
without accepting arbitrary mappings or runtime objects.

`@extern` is also transparent at call time. Its control-plane contract is
derived from resolved input and output annotations and requires typed
`Effect`, `StateScope`, `Device`, and `Capability` values. Raw string policy
and missing annotations fail during decoration. No `@task`, `@stream`, or
`@buffer` decorator is introduced.

## Milestone 2: Strict Capture And Inference

`.capture(...)` now runs a source/closure preflight before calling PyTorch. A
supported pure Tensor program is wrapped in an internal `nn.Module` and passed
to `torch.export.export` exactly once with `strict=True`. Decoration and normal
calls still do not import or invoke the capture path.

Successful captures retain only frozen scalar descriptors:

- tensor shape, dtype, typed CPU/CUDA device, layout, and stride from
  FakeTensor metadata;
- SSA producer/input/user ordinals and logical value version;
- stable alias sets propagated from operator schema alias annotations;
- bounded static/range shape guards;
- a leaf `ArtifactKey` and ordered ATen `LeafOperatorDesc` records.

FakeTensor, Tensor, FX node, operator objects, and `ExportedProgram` never enter
`CapturedFlow` or its structural signature. The wrapper owns successful
`ExportedProgram` handles in a private control-plane specialization cache.

The preflight rejects Tensor data escape through `.item()`, `.numpy()`, or
`.tolist()`, host I/O (including referenced aliases), unregistered mutable
closure state, unsupported runtime-object inputs, and obvious Python/in-place
mutation. Postflight scans every `call_function` operator schema, including
nodes with no users. Any mutable schema or unknown schema discards the whole
leaf and produces an opaque barrier. Functionalization is not currently used
to claim mutation equivalence.

Dynamic shape declarations use frozen `DynamicDim` descriptors. The current
front end accepts bounded positional Tensor dimensions and rejects an
unbounded dimension before export with `WG206_UNBOUNDED_DYNAMIC_SHAPE`.
Unknown symbolic bounds returned by PyTorch are rejected in the same way.

## Current Limits

- No root/task/leaf hierarchical graph or `CaptureAttachment` exists yet.
- The accepted success subset is pure ATen Tensor dataflow with static or
  finitely bounded positional dynamic dimensions. Structured multi-result and
  schema-less operators remain conservative barriers.
- Mutation is detected and retained as a typed write effect, but no mutation
  is yet proven equivalent and accepted as a captured compute leaf.
- `@extern` contracts are typed but are not yet partitioned into an opaque leaf
  region inside a captured flow.
- No WorkGraph artifact affects runtime execution or the token hot path.
- FULL graph mode remains typed unsupported in the Phase 1a compatibility
  compiler.

The next accepted milestone is the root/task/leaf hierarchical graph plus
explicit `CaptureAttachment` and typed port bindings into the existing
compatibility lowering. The existing runtime plans remain authoritative.

## Verification

Use the repository virtual environment:

```bash
/workspace/pllm/.venv/bin/python -m pytest -q test/workgraph
/workspace/pllm/.venv/bin/python -m pytest -q
/workspace/pllm/.venv/bin/pyright -p pyrightconfig.strict.json
cargo test --manifest-path rust/tllm_core/Cargo.toml
cargo fmt --manifest-path rust/tllm_core/Cargo.toml -- --check
git diff --check
```
