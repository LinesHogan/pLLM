# Region Authority v1

This freezes the ExecutionRegion v1 boundary. The executable companion,
`tools/check_region_authority.py`, scans only generic production Region files;
tests and compatibility adapters are not treated as Region core.

## Authority Table

| Concern | Authority | Python may do | Python must not do |
| --- | --- | --- | --- |
| Dependence, anchor window, Region partition, materialization/lifetime, fence, memory | Rust `compile_schedule` / `validate_schedule` | Transport JSON and present errors | Split/reorder Regions or invent a boundary/lifetime |
| Compiler adapter, fullgraph, dynamic/static shapes, autograd, fallback | Rust `RegionCompileCommand` | Dispatch a registered enum adapter | Inspect a task/callable/options to select policy |
| Artifact and decision identity | Rust signed decision | Bind exact key | Recompute key from local facts |
| Physical DAG | Signed Region records and physical descriptor | Bind Tensor, buffer, anchor, cross-Region fence | Read task membership, `compile_safe`, or raw `compile_policy` to re-decide |
| Typed extern | Rust implementation identity/capability | Resolve exact registered stager | Parse implementation strings or select a workload stager |
| State version | Rust typed effects/version publish | Audit observed transition | Reorder effects or fabricate a publish |

`compile_safe` is static capability evidence. Changing
`CompileOptions.compile_policy` must only change the signed compile command,
artifact key, and decision signature, not capability evidence.

## Stable Interfaces

| Interface | Required stable fields | Failure behavior |
| --- | --- | --- |
| `ScheduleDecision` v2 / `ExecutionRegion` | ids, anchors, dependencies, materializations, effects, capture/stream/fallback, artifact key, `compile_command` | unknown, missing, unsigned, or noncanonical field rejects |
| `RegionCompileCommand` | adapter, fullgraph, dynamic, ordered shape specialization, autograd, fallback | unsupported adapter or compiler failure raises; no eager retry |
| `RegionCallableArtifact` | Region id/key, ordered input/output keys, exact typed bindings/effects, local callable, compiled marker, command | missing/mismatched callable/key/binding rejects before execution |
| `BackendRegionBinderSlot` | signed implementation id and event context; exact-once `stage` then `invoke`, `clear`, or `abort` | duplicate/stale/missing stage rejects; a failed callback is never retried |
| `StateVersionBinding` / `StateVersionRegistry` | Rust state descriptor, process-local version snapshot, typed read/write/transition/publish effects | binding coverage or slot identity mismatch rejects; each signed mutation requires one observed increment |
| Physical primitive | signed plan/node/fence/buffer descriptor | unknown node or mismatched command rejects |

The FFI remains `compile_schedule(graph, timeline, options)` and
`validate_schedule(graph, timeline, options, decision)`. No callable, Tensor,
stream, or backend model object crosses it.

## Fail-Closed Matrix

| Condition | Required outcome |
| --- | --- |
| Unknown or omitted Region/compile-command field | Reject during decode/validation |
| Tampered command, artifact key, shape specialization, or signature | Reject before binding |
| Missing callable, stager, capability, residual binding, or static resource | Reject before forward or at signed event |
| `torch.compile` failure | Propagate error; never rebuild eagerly |
| Shape miss | Request a coordinator specialization epoch; do not mutate live Region |
| Internal materialization or per-task fence | Schedule/physical validation rejects |
| State/version mismatch | Reject step; do not fabricate version |
| Native declaration asks for legacy-only feature | Reject selection; do not mix authority in a step |

## Migration Template

1. Declare typed input/output/effect/timing/affine data with `tllm.workload`.
2. Add a failing schedule-contract test for Region boundaries, effects,
   materializations, and unavailable capabilities.
3. Assert schema v2, complete Region coverage, and a Rust-signed command for
   every compiled Region.
4. Bind only local declarations, Tensor/buffers, sinks, residual paths, and
   exact backend stagers. Do not create a task plan, stream placement, or event
   schedule.
5. Verify state/version, request isolation, artifact coverage, then GPU
   correctness. Use either the complete native executor or complete legacy
   adapter for a workload.

Checklist: every task belongs to exactly one Region; every cross-Region value
has a signed lifetime; no internal value materializes; compiled Regions have
commands and artifacts; extern Regions have explicit null commands and stager
identities; and all missing-resource tests fail closed.

## Typed Recirculation Core v1

Recirculation prefill uses a Rust-signed family of exact one-request stage
artifacts. Core v1 bounds that cold family to 16 prompt rows (or the backend's
smaller `max_prefill_tokens`); a larger observed span is a
`specialization-required` failure. Expanding the family is a new coordinator
cold epoch, never hook-time compilation or eager fallback.

Each row specialization has exactly one signed artifact. Its fusion legality,
operation order, grouping kind, and `release_after` values are Rust-derived;
the native request has no serial/wavefront selector. The serial implementation
remains a numerical oracle and test fixture only.

## Diagnostics

`NativeWorkloadExecution.read_resource_diagnostics()` returns the stable local
resource view used by worker status reporting. Its outer keys are workload ids;
each value maps a non-empty diagnostic name to a finite JSON number (`int` or
`float`, with booleans rejected). Diagnostics are observational only: they do
not enter scheduling, artifact identity, cache selection, or signatures. An
invalid reader result fails the status request instead of being normalized or
used as a scheduling hint. MP status reports the same validated per-worker
mapping alongside immutable plan/artifact identity.

## Frozen Verification

The Core v1 checkpoint was verified on 2026-08-25 with
`/workspace/pllm/.venv/bin/python`: Python 3.11.11, pytest 9.0.3,
torch 2.13.0+cu129, vLLM 0.27.1, maturin 1.14.1, and Rust 1.97.1.

```bash
cargo fmt --manifest-path rust/tllm_core/Cargo.toml -- --check
cargo check --manifest-path rust/tllm_core/Cargo.toml
env LD_LIBRARY_PATH=/home/user/.local/share/uv/python/cpython-3.11.11-linux-x86_64-gnu/lib \
  cargo test --manifest-path rust/tllm_core/Cargo.toml --lib
/workspace/pllm/.venv/bin/python -m pytest -q
PATH=.venv/bin:$PATH /workspace/pllm/.venv/bin/python tools/check_typing.py
/workspace/pllm/.venv/bin/python tools/check_region_authority.py
```

Results: Rust `46 passed`; Python `1067 passed, 1079 subtests passed`; strict
typing `0 errors`; basic typing `163 errors` against a baseline of 176; and the
authority gate passed across 20 production files. The frozen schema examples
are `test/workload/golden/schedule_v2_region.json` for a maximal compiled
Region and `test/workload/golden/schedule_v2_complex_region.json` for typed
extern staging, state effects, materialization, and a fence. Schedule,
binder/state, recurrence, worker reconstruction, and authority-boundary
regressions are in
`test/workload/test_schedule_contract_unit.py`,
`test/runtime/test_backend_region_binders_unit.py`,
`test/runtime/test_native_workload_execution_unit.py`,
`test/core/test_typed_recurrence_plan_unit.py`, and
`test/runtime/test_execution_region_core_v1_boundary_unit.py`.

A fresh release wheel imported from an isolated extraction directory and ran a
schema-v2 `compile_schedule`/`validate_schedule` static-isl smoke. Its extension
had only libc, libgcc, and loader dependencies; it did not dynamically link
`libisl`.

With `CUDA_VISIBLE_DEVICES=0`, offline model loading, and
`--gpu-mem 0.15`, serial correctness smokes passed for Region CUDA Graph,
native Dummy MRv1/MRv2, native Recirculation MRv1/MRv2, native ESamp
in-process MRv1/MRv2, and native ESamp EngineCore MP MRv1/MRv2. ESamp reported
nonzero loss/update/state/intervention evidence and zero fallback; Recirculation
reported typed layer replay and zero fallback. Throughput benchmarks were
intentionally not run for this freeze.

## Part II: Legacy Inventory

| Surface | Permitted role until migrated |
| --- | --- |
| `tllm.program` `@flow` / `@extern` | Compatibility declaration/lowering lane |
| `tllm.consumers.*` | Complete workload adapter or benchmark baseline |
| Legacy ESamp program/hand-tuned training | Explicit migration and paired-benchmark baseline |
| Override, TTRL, FTTT, TLM, In-Place-TTT, GTTA | Native workload authority; no legacy consumer or Program implementation |
| Consumer streams/events/plan stores/recurrence schedulers/shadow snapshots | Legacy detail; never Region physical authority |
| vLLM model-specific adapters | Backend quarantine exposing typed capability/binder contracts only |

Deleting a legacy item requires a reviewed migration preserving numerical and
state semantics with a complete signed Region executor.

## Deferred Authoring API TODO

After workload-authority migration and legacy-stack retirement, replace
stringly task/value references such as `result("task", "output")` and the
remaining textual hidden/resource selectors with typed declaration
descriptors. The authoring surface should provide IDE completion, static
validation, and rename-safe references while lowering to the same stable
symbol names used by Rust signatures. This ergonomics work must not introduce
Python-side scheduling choices or weaken deterministic declaration lowering.
