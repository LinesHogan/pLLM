# Architecture

## The layered stack

```
┌─────────────────────────────────────────────┐
│ workflows/          entry points (bench,    │
│                     starter, esamp support) │
├─────────────────────────────────────────────┤
│ consumers/          leaves: esamp (legacy + │
│                     program), your consumer │
├─────────────────────────────────────────────┤
│ backends/           vLLM v1 adapters        │
├──────────────────────┬──────────────────────┤
│ runtime/             │ program/             │
│ execution: lanes,    │ compiler: declare →  │
│ sampler bridge,      │ slice → verify →     │
│ stats, plugins       │ generate consumer    │
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

## The program pipeline

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

## Runtime execution

The generated consumer runs inside vLLM generation. Execution policy is derived from the declared DAG, not hardcoded:

- **State banks** (`program/state.py`): versioned slots holding parameters and optimizer state, scoped shared or per-request. Publishing is reference-swap with event fences, so readers always see a consistent snapshot.
- **Lanes**: a training lane owns state writes; a reader lane serves `run_flow` predictions; edges between lanes are exactly the declared data dependencies (input-readiness waits plus completion-event fences) — never blanket stream waits.
- **Grad mode**: inferred from `result()` edges. A flow whose output feeds a due extern runs once in grad mode; otherwise it runs under `no_grad`.
- **CUDA graph capture**: steps that are fixed DAGs over persistent buffers (identity batch groups on CUDA, AdamW scalars) are captured per batch-shape bucket and replayed. The first replay is verified against the eager functional transition (`PG146`). Graph-vs-eager is an internal execution decision, invisible to consumers.

## The sampler bridge

`runtime/sampler_bridge/` connects consumer predictions to vLLM's sampler: a provider reads the current state bank via `run_flow` (reader lane) and emits candidate-level logit adjustments applied after top-k/top-p/min-p filtering. Consumers never touch sampler internals.

## Gates

Development is guarded by: the import-layer gate, strict typing on the changed surface (only-decrease baselines elsewhere), and functional counters in benchmarks (loss counts, intervention counts) so throughput numbers are always accompanied by proof the algorithm actually ran.
