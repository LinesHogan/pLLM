# Architecture

This guide explains how data leaves vLLM, passes through tLLM, and reaches a consumer.

If you only want to write a simple consumer, you do not need to memorize every detail. Read [Write Your First Consumer](../getting-started/write-your-first-consumer.md) first, then come back here when a concept becomes relevant.

tLLM separates two concerns:

- **Producer**: find and extract the correct rows from vLLM's packed tensors.
- **Consumer**: decide what to do with those rows.
- **Runtime**: install hooks, assemble bundles, and schedule consumer work.

## The Problem

Suppose you want to capture hidden states during vLLM generation and optionally feed something back into the generation process. Three things make this hard.

### 1. vLLM Packs Many Requests Into One Tensor

With multiple active requests, vLLM uses tensors shaped like `[total_tokens, hidden_size]`. You need to know which row belongs to which request and phase. tLLM calls this **localization**.

### 2. Useful Hook Points Are Version-Sensitive

`_prepare_inputs`, `execute_model`, and model-layer forward calls can change across vLLM versions. Putting algorithm code directly into those places creates a brittle fork.

### 3. Training Must Not Block Inference

Running ESamp distiller backward directly inside model forward would block the main inference stream. The work needs to be asynchronous, but still synchronized correctly with the data it consumes.

## tLLM's Split

tLLM keeps the boundaries narrow:

- Runtime handles capture, localization, and bundle assembly.
- Consumer code receives `PortBundle` objects.
- Algorithm internals stay inside the consumer.

ESamp follows this split:

- `ESampConsumer` reads bundles and owns consumer-facing state.
- `ESampTrainEngine` owns the training pipeline and model-bank mechanics.
- Runtime does not know ESamp's training details.

## Language and Compiler Boundary

tLLM's declarative model is both a small dataflow language and a compiler
pipeline. The useful boundary is not "Python versus Rust" by file count. It is
the boundary between declarations, decisions, and execution:

| Layer | Owns | Does not own |
|------|------|--------------|
| Python typed front-end | `ConsumerFlow`, producer descriptors, plugin specs, structural port addresses, typed consumer input views | Cross-flow scheduling decisions |
| Rust compiler mid-end | startup validation, provider selection, dependency DAGs, legal hook placement, fusion eligibility metadata | Tensors, CUDA streams/events, vLLM objects, plugin instances |
| Python runtime executor | vLLM adapters, Tensor/CUDA materialization, bundle assembly, deferred work, stream/event launch | Re-deciding the compiled plan on every token |

The FFI carries frozen attribute-based descriptors. It does not carry nested
magic dictionaries or Torch objects. Dynamic probing remains only at foreign
API and plugin boundaries; after a boundary value is validated, downstream
code uses typed attributes. `PortBundle.entries` remains a compatibility and
extension API, while built-in consumers lower it once into typed input views.

Phase 1a WorkGraph is a startup/control-plane shadow only. `DispatchPlan` and
`ExecutionPlan` remain authoritative for every runtime decision and dispatch;
the shadow records and validates their typed structure but cannot replace,
reorder, or execute them. Its explicit anchor edge represents one fixed
`AnchorNode -> WorkNode` attachment. It does not yet encode the future
"between two vLLM lifecycle points" interval constraint.

The Phase 1 typed front end adds transparent `@flow` and `@extern` control
plane declarations. Normal calls still execute their original Python
functions. Until strict export and effect analysis prove a complete leaf
artifact, `.capture(...)` records one source-aware opaque barrier and cannot
affect runtime execution.

Producer objects are declarations, not miniature executors. They describe
outputs, requirements, hooks, and graph safety. Backend drivers remain
responsible for materializing those declarations from vLLM tensors.

## DAG and Automatic Placement

At startup, tLLM lowers flow reads and writes to tagged structural
`PortAddress` values. The Rust core builds dependency edges for same-consumer
state order, read-after-write/write-after-read hazards, and explicitly ordered
write-after-write hazards. A stable Kahn topological sort rejects cycles and
produces execution order.

Placement then computes legal hook bounds from mobility, declared and earliest
hooks, graph mode, backend-materialized hooks, and predecessor selections. The
current scheduler is a constrained greedy list scheduler. For each flow in
topological order it maximizes, in order:

1. capture eligibility bonus;
2. configured or default hook `bubble_score`;
3. proximity to the declared hook;
4. stable hook order as a deterministic tie-break.

This is intentionally smaller than a resource-constrained project scheduler.
The current cost model has no measured task duration, SM/HBM/copy-engine
capacity, or transfer cost, so tLLM does not claim global optimality. Fusion
output is eligibility metadata only: a group must be topologically contiguous,
use the selected materialized fusion hook, have stable leased buffers and a
static row bucket. Torch/vLLM compilation remains responsible for kernel and
graph optimization.

Future algorithms should follow the optimization opportunity and available
cost data:

| Opportunity | Appropriate algorithm |
|-------------|-----------------------|
| Dependency legality and stable ordering | Kahn; Tarjan only when richer SCC diagnostics are needed |
| Earliest/latest legal hook windows | ASAP/ALAP constraint propagation |
| Online deadlines after duration/resource profiling exists | critical-path or EDF-driven list scheduling with admission control |
| Capture-safe fusion regions | topologically convex region formation |
| Stable buffer reuse after order is fixed | liveness intervals plus linear scan/interval coloring |
| Multiple alternative providers with shared dependencies | AND-OR selection; bounded search or ILP only when the graph warrants it |

CP-SAT or exhaustive search can be useful as a small-graph oracle, but adding a
solver before adding a real resource/cost model would only optimize invented
numbers.

## Port: The Public Data Interface

Consumers do not manipulate vLLM internals directly. They declare ports:

| Port | Meaning | Example |
|------|---------|---------|
| `residual_stream` | Hidden states at a layer/site | `ResidualStream.read(layer=0, site="block_output")` |
| `request_meta` | Request identity metadata | `RequestMeta.read()` |
| `cpu_export` | Asynchronous CPU export | `CpuExport.write(channel="debug")` |
| `logits` | Pre-sampling logits | `Logits.read()` |
| `kv_cache` | KV cache state | `KVCache.read(layer=12)` |

Example flow:

```python
ConsumerFlow(
    reads=(
        ResidualStream.read(layer=0, site="block_output", phase="decode"),
        RequestMeta.read(),
    ),
    writes=(CpuExport.write(channel="debug"),),
    window="background",
)
```

The runtime installs hooks, gathers rows, groups frames by `bundle_key`, and calls `consume_bundle(bundle, ctx)`.

## A Decode Step

From a consumer's point of view, one decode step looks like this:

1. `_prepare_inputs` runs.
2. Runtime snapshots request order and decode localization metadata.
3. vLLM runs model forward.
4. Runtime layer hooks gather the requested hidden rows into fixed buffers.
5. `execute_model` returns.
6. Runtime assembles `PortBundle` objects and dispatches consumers.
7. Step-end feedback or async training is scheduled if the consumer needs it.

For sampler-guidance consumers, tLLM also uses the `compute_logits` and sampler boundaries so prediction can happen before candidate logits are modified.

## Localization

### Decode Localization

During decode, each active request contributes one row: the token being generated now.

The producer:

1. Reads `logits_indices`.
2. Selects requests marked as decode.
3. Writes row indices into a fixed GPU buffer.
4. Uses graph-safe gather in the layer hook.
5. Applies a valid mask for active rows.

Fixed buffers matter because CUDA Graph replay cannot allocate new tensors every step.

### Prefill Localization

During prefill, each request may occupy a contiguous span of rows.

The producer computes:

```text
prefill_len = min(scheduled, prompt_len - computed)
```

Then it records the `[start, start + prefill_len)` range for each request. Prefill currently uses an eager-first path, separate from decode's graph-safe path.

## Writing a Consumer

The simplest consumer implements:

```python
class MyConsumer(BaseConsumer):
    @property
    def consumer_id(self):
        return "my_consumer"

    def flows(self):
        return [
            ConsumerFlow(
                reads=(ResidualStream.read(layer=0, site="block_output", phase="decode"),),
                window="background",
            )
        ]

    def consume_bundle(self, bundle, ctx):
        hidden = bundle.entries["hidden"]
        # Your logic here.
```

If you need step-end work, implement `on_step_end(ctx)` as an optional compatibility hook.

## Related Docs

- [WorkGraph Phase 1a](workgraph-phase1a.md)
- [WorkGraph Phase 1](workgraph-phase1.md)
- [WorkGraph Phase 2](workgraph-phase2.md)
- [WorkGraph Phase 3](workgraph-phase3.md)
- [WorkGraph Phase 4](workgraph-phase4.md)
- [Glossary](../reference/glossary.md)
- [Project Structure](../reference/project-structure.md)
- [Write Your First Consumer](../getting-started/write-your-first-consumer.md)
