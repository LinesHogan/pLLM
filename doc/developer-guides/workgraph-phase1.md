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

This is the required failure behavior, not a successful capture. Milestone 2
will replace it only when `torch.export.export(..., strict=True)` and the
effect analysis both prove a complete leaf graph. There is no `strict=False`
success fallback and no partially captured graph.

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

## Current Limits

- No successful strict export capture or FakeTensor inference exists yet.
- No mutation, alias, closure-state, host-I/O, or dynamic-shape proof exists.
- No root/task/leaf hierarchical graph or `CaptureAttachment` exists yet.
- No WorkGraph artifact affects runtime execution or the token hot path.
- FULL graph mode remains typed unsupported in the Phase 1a compatibility
  compiler.

The next accepted milestone is strict export capture plus source-aware typed
diagnostics and conservative effect analysis. An export result is not accepted
merely because PyTorch returned an `ExportedProgram`: host I/O, unregistered
closure state, unknown effects, and unproven mutation must still lower to one
opaque barrier.

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
