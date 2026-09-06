# Current Architecture

The active path is:

```text
typed workload declaration
    -> SemanticGraph
    -> Rust schedule and representation decisions
    -> signed PhysicalPlan
    -> Python task execution and PTTS
    -> typed host service or vLLM attachment
```

## Owners

| Owner | Responsibility |
| --- | --- |
| `tllm/contracts` | Typed graph, state, layout and attachment data |
| `rust/tllm_core/src/canonical` | Closed codecs, dense IDs, signed scheduling and representation decisions |
| `tllm/canonical/execution.py` | Execution of signed task/anchor order |
| `tllm/canonical/state_execution.py` | Shared tensor-state invocation, sampler borrowing, cancellation and settlement |
| `tllm/canonical/backend_program.py` | Workload-neutral cold facts and original packed tensor inputs; deployment references the existing execution and signed attachment |
| `tllm/canonical/host_runtime.py` | Typed host transaction settlement |
| `tllm/canonical/ptts` | One PageBundle arena for versions, field pools, COW, issued receipts, exact pins and completion; request directory only selects lineages |
| `tllm/canonical/workloads` | Workload declarations, tensor math, initialization and mathematical inputs/outputs |
| `tllm/canonical_vllm` | Current public MRv2 capture, sampler and dense-overlay integration |
| `tllm/backends/vllm` | Remaining probe/replay primitives awaiting consolidation |

`tllm.make_llm` is the public factory. Without a workload it constructs plain
vLLM; with a supported canonical workload it installs the instance-owned
attachment. No residual runtime or process-wide bootstrap plugin is installed.

Result inputs imply dependencies directly; explicit `SemanticEdge` values add
ordering constraints that are not already expressed by dataflow. Rust derives
one index of predecessors, timing windows and execution order from this graph.
Overlapping conflicting state effects require a causal path; disjoint windows
can prove their order. Ambiguous accesses fail closed instead of acquiring an
accidental order from task names. This is not yet exact-version hazard analysis.

Python serializes these same immutable descriptors at the FFI boundary, adding
only the closed input-variant tag. There is no second wire DTO hierarchy or
workload-maintained copy of the dependency graph.

## Incomplete Boundaries

ESamp and In-Place use one StateExecution on CPU/CUDA. Its invocation retains
the original physical plan, pin, write and publication; settlement does not
replay training. In-Place math consumes original captured operand records.
The vLLM factory and session consume BackendWorkload without importing either
workload. Dense application borrows arena pages under the original invocation
pin, using the original signed field binding and packed rows, without a second
overlay plan or copied tensor bank. Borrowers must not mutate published views
or use them after pin release; owned external exports still use snapshots.
Schema 3 binds target, operator and field IDs together. Unsupported combined
hidden/sampler/overlay programs fail closed; generic integration is not yet
composition support or public support for every workload.

PageBundle/generational state is implemented, but captured page-table version
selection and TLM/global-overlay integration remain pending. Pool address
stability is not capture-stable version selection. Complete AER regionization,
distributed placement, closed typed MP and workload-independent profiling also
remain pending. Backend-local overlay version storage is deleted; global
overlay must use this same arena. Callback-based replay contracts still need
replacement through the shared semantic/effect boundary.

The maintained support matrix is in the [README](../README.md); implementation
debt and acceptance criteria are in the [handoff](architecture-consolidation-handoff.md).
