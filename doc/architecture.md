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
| `tllm/canonical/host_runtime.py` | Typed host transaction settlement |
| `tllm/canonical/ptts` | Current request/global state and provisional dense-page providers |
| `tllm/canonical/workloads` | Workload declarations, tensor math and current execution implementations |
| `tllm/canonical_vllm` | Current public MRv2 capture, sampler and dense-overlay integration |
| `tllm/backends/vllm` | Remaining probe/replay/overlay primitives awaiting consolidation |

`tllm.make_llm` is the public factory. Without a workload it constructs plain
vLLM; with a supported canonical workload it installs the instance-owned
attachment. No residual runtime or process-wide bootstrap plugin is installed.

## Incomplete Boundaries

The existing task executor and host coordinator are shared foundations.
The vLLM integration still branches on ESamp and In-Place-TTT, and workload
execution contains duplicated lifecycle code. Their consolidation remains
required before claiming one workload-independent execution layer.

The dense-page providers are provisional. PageBundle/generational state,
complete AER regionization, distributed placement, closed typed MP and
workload-independent profiling remain pending. Backend-local overlay version
storage and callback-based replay contracts must also be replaced.

The maintained support matrix is in the [README](../README.md); implementation
debt and acceptance criteria are in the [handoff](architecture-consolidation-handoff.md).
