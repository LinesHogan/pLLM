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
| `ScheduleDecision` v4 / `ExecutionRegion` | ids, anchors, dependencies, materializations, effects, capture/stream/fallback, signed side waits and host commands, artifact key, `compile_command` | unknown, missing, unsigned, stale-v2/v3, or noncanonical field rejects |
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
are `test/workload/golden/schedule_v4_region.json` for a maximal compiled
Region and `test/workload/golden/schedule_v4_complex_region.json` for typed
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

A 2026-08-30 follow-up extended native Recirculation eager correctness to
MRv1/MRv2 × in-process/EngineCore-MP. Every cell replayed two requests and
seven prompt rows with `fallback_count=0`; the MRv1 MP cell also exercised a
mixed prefill/decode packed step, where the Rust-signed admission selected
only prefill request rows. CUDA Graph capture, hook retirement, speculative
decode, and distributed TP/PP/DP support remain outside this evidence.

The same follow-up closed native OverRIDE eager correctness on
MRv1/MRv2 × in-process/EngineCore-MP. Every cell executed three training
iterations followed by the signed final no-op iteration, advanced versions
exactly `[2, 4, 6]`, reported six losses and six optimizer updates, and kept
`fallback_count=0`. Fixed-RNG tokens matched exactly between in-process and MP
within each runner family; the smoke freezes a separate golden for MRv1 and
MRv2 because those upstream sampler implementations do not share one RNG
sequence. Finished request mappings were empty in every cell.

Mapped generation uses one generic bulk-admission transaction: level-0 pause,
one typed identity publication, enqueue the complete public batch, then resume
scheduling. This preserves the batch boundary across EngineCore transport but
does not select lanes, priorities, fusion, or execution order; after admission,
the existing scheduler remains the sole physical scheduling authority. The
shared Region invocation boundary also owns inference tensors before entering
autograd, rather than assigning that responsibility to OverRIDE or an MP
adapter. Decode localization stages the exact request/sample identity for the
signed step before automatically placed work can run; the later sampler view
owns only same-step logits and sampled tokens. Parallel sampling keeps child
sample identities when no explicit map is supplied, while parent completion
retires every child alias and overlay lease. An explicit one-to-one sample map
is rejected when `SamplingParams.n > 1` instead of being silently reinterpreted.

Native TTRL now closes one host transaction as an atomic staged pair: full
trace-aligned hidden rows plus a typed majority-reward intent. A signed extern
prepare task performs explicit row selection and reward-tensor construction;
the compiled train task receives tensors only, and its conditional optimizer
does not advance state for an empty trainable set. The post-transition publish
task emits independent typed reward, retry, update-barrier, and aggregate
control outputs. Host settlement requires the pair to be consumed and all four
outputs to agree before acknowledgement. This adds no TTRL lane, priority,
fusion, partition, or execution-order policy.

The MRv1/MRv2 × in-process/EngineCore-MP eager matrix passed with one loss,
one update, state version 1, reward oracle `[1, 1, 0]`, one version-pinned
retry, complete typed-output quorum, and `fallback_count=0`. The smoke executes
the emitted retry request, but uses a deterministic test verifier and staged
hidden rows; it is correctness evidence for the native closed-loop contract,
not a claim that this side reward head is an official policy-GRPO trainer.

Staged-port lifecycle/status/resolution enums and the closed `SKIPPED` marker
now live in `tllm.contracts.workload_ports`; workload storage, runtime
execution, kit services, and FTTT all import the same canonical objects from
the neutral layer. The obsolete workload-owned marker module was removed.
The import-layer gate is back to module/local `0/0`, snapshots its exact and
sanctioned edges, rejects unreviewed rule drift during ordinary checks, and
refuses a rebase that would increase violations.

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

## Deferred Typed Port Transport TODO

Replace the MP port wire's dynamic dataclass type-path import and reflective
reconstruction with a closed, registered typed codec whose schema is fixed at
the Python–Rust interaction boundary. Workload and runtime Python must receive
the declared payload type directly; they must not infer classes from strings,
walk attributes with reflection, or add workload-specific codecs. Preserve
shared-memory tensor transport, exact transaction identity, and worker-quorum
validation while making unsupported payload schemas fail at declaration time.

## Deferred Profiling Support TODO

At the end of the migration work, add one workload-independent profiling
surface built on the existing Region cost probes and workgraph statistics. A
typed `ProfileSpec`/`ProfileWindow` must coordinate the parent and EngineCore
workers and attach stable NVTX/`record_function` correlation metadata to the
signed workload, Region, artifact, plan epoch, physical node, step, anchor,
stream, fence, host-service, state-transition, and eager/capture/replay
boundaries. It must not introduce per-workload profiler callbacks or become a
source of lanes, priorities, fusion, or execution-order policy.

The supported analysis workflow should use Nsight Systems for the
multi-process CPU/GPU timeline and overlap analysis, `torch.profiler` for
operator attribution, and Nsight Compute only after a kernel hotspot has been
identified. CUDA Graph analysis must preserve three separate, correlated
windows for eager attribution, capture cost, and warmed steady-state replay
under the same signed plan and shapes; capture overhead must never be folded
into replay throughput. Stream-local CUDA events should be read after the
window rather than inserting global synchronization that destroys the overlap
being measured.

Expose the same method to maintainers and workload authors through a minimal
runtime profile-window API plus an external profiler-launch CLI. Reports should
retain Region GPU duration, host launch and queue delay, fence wait, overlap,
capture/replay/fallback counts, memory peaks, and training evidence such as
loss/update counts. Tool-specific dynamic integration belongs only in its
explicit adapter quarantine; the remaining Python profiling surface must stay
strictly typed and context-deterministic.
