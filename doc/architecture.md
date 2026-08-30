# Architecture

## Current execution model

`tllm.workload` declares a fine-grained graph for analysis, but
`ExecutionRegion` is the coarse-grained physical unit of execution. Rust
derives, validates, and signs every Region's bindings, materializations,
fences, lifetimes, artifact identity, and capture policy. Python does not
infer task meaning, reschedule, or choose capture: it mechanically materializes
the signed artifact and submits the Region at its signed anchor. A fence/event
may exist only at a Region, device, or backend boundary; Region-internal
dependencies are ordinary FX/Tensor/CUDA-Graph edges.

`RegionCaptureMode` is a Rust closed enum carried in the signed decision. It
selects either `disabled` or `require` before Python lowering; Python does not
infer capture from task types, streams, or runtime names. Unsupported
capability, artifact, or binding is a cold-install rejection.

With `torch==2.13.0+cu129` and `vllm==0.27.1`, verified GPU correctness covers
native Dummy, native Recirculation, and native ESamp exact eager, in-process
execution on MRv1 and MRv2. Recirculation also passed EngineCore MP on both
runners, replaying two requests and seven prompt rows with zero fallback. The
latest short ESamp smoke had `loss_count=6`, `update_count=6`,
`state_version=6`, and `fallback_count=0`. Native ESamp exact eager Regions
also passed EngineCore MP correctness on both runners: loss, update,
state-version, and intervention counters were nonzero, while fallback and
replay counts were zero. This is not a throughput claim. Model-backed native
piecewise CUDA Graph execution, Recirculation capture/hook-retirement
integration, and shape-miss epoch switching remain unverified or incomplete.

## The layered stack

```
┌─────────────────────────────────────────────┐
│ workflows/          entry points (bench,    │
│                     starter, esamp support) │
├─────────────────────────────────────────────┤
│ consumers/          legacy adapters and     │
│                     native workload shells  │
├─────────────────────────────────────────────┤
│ backends/           vLLM v1 adapters        │
├──────────────────────┬──────────────────────┤
│ runtime/             │ workload/            │
│ Region executor,     │ declaration → graph  │
│ sampler bridge,      │ → signed Region plan │
├──────────────────────┴──────────────────────┤
│ workgraph/          declaration language    │
│                     (frontend, lowering,    │
│                     placement, admission)   │
├─────────────────────────────────────────────┤
│ common/ producers/  shared mechanisms,      │
│                     tensor production       │
├─────────────────────────────────────────────┤
│ ports/              consumer-facing         │
│                     abstractions & flows    │
├─────────────────────────────────────────────┤
│ contracts/          pure data types         │
├─────────────────────────────────────────────┤
│ core/               Rust extension: small,  │
│                     decision-making         │
└─────────────────────────────────────────────┘
```

Dependencies only point downward. This is enforced mechanically by `tools/check_import_layers.py`, an AST-level gate that runs in CI/tests: each package declares its allowed import targets, and violation counts are pinned by an only-decrease baseline. New cross-layer debt cannot be introduced without explicitly raising the baseline.

Two principles:

- **Rust decides, Python executes.** The `core/` extension makes scheduling decisions; all tensor execution stays in Python/torch where it is inspectable and debuggable.
- **Fail closed.** Every compiler rejection is a stable error code; there is no silent fallback path (`PG190` exists precisely to make that attempt a loud error).

## Native Region pipeline

```
@workload / @task
      │ typed values, effects, timing, affine access
      ▼
ScheduleGraph ──► Rust legality + granularity + memory planning
      ▼
signed ScheduleDecision v3 ──► ExecutionRegions + RegionCaptureMode
      ▼
PhysicalFlowPlan ──► one artifact/callable per Region at signed anchors
```

The planner starts from the largest legal Region and splits only for an
unintersectable anchor window, required effect/version or typed-extern
boundary, backend/device/graph-safety boundary, or hard memory constraint. It
minimizes explicit materialization and boundaries; it does not choose per-task
streams, predict kernel timing, or allocate SMs.

## Legacy Program compatibility pipeline

```
ESampProgram (your Module)
      │  @flow / @extern declarations
      ▼
declare_program ──► static IR (flows, externs, state specs, edges)
      ▼
slice_flow_stages ──► torch.export graphs per flow
      ▼
compile_program ──► optimizer lowered to functional transitions;
                    oracle check: extern body ≡ functional transition (PG145)
      ▼
ProgramConsumer ──► runtime execution (below)
```

## Legacy runtime execution

The following lane/state-bank description applies to the retained Program
compatibility path. It is not the native Region optimizer contract.

The generated consumer runs inside vLLM generation. Execution policy is derived from the declared DAG, not hardcoded:

- **State banks** (`program/state.py`): versioned slots holding parameters and optimizer state, scoped shared or per-request. Publishing is reference-swap with event fences, so readers always see a consistent snapshot.
- **Lanes**: a training lane owns state writes; a reader lane serves `run_flow` predictions; edges between lanes are exactly the declared data dependencies (input-readiness waits plus completion-event fences) — never blanket stream waits.
- **Grad mode**: inferred from `result()` edges. A flow whose output feeds a due extern runs once in grad mode; otherwise it runs under `no_grad`.
- **CUDA graph capture**: steps that are fixed DAGs over persistent buffers (identity batch groups on CUDA, AdamW scalars) are captured per batch-shape bucket and replayed. The first replay is verified against the eager functional transition (`PG146`). Graph-vs-eager is an internal execution decision, invisible to consumers.

## The sampler bridge

`runtime/sampler_bridge/` connects consumer predictions to vLLM's sampler: a provider reads the current state bank via `run_flow` (reader lane) and emits candidate-level logit adjustments applied after top-k/top-p/min-p filtering. Consumers never touch sampler internals.

## Gates

Development is guarded by: the import-layer gate, strict typing on the changed surface (only-decrease baselines elsewhere), and functional counters in benchmarks (loss counts, intervention counts) so throughput numbers are always accompanied by proof the algorithm actually ran.
