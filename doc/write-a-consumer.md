# Write a Consumer

A consumer is a `torch.nn.Module` whose methods are annotated with the tLLM declaration vocabulary. You declare **what** depends on **what**; the compiler decides **where** and **how** it runs.

## The declaration vocabulary

```python
from tllm.workgraph.frontend import extern, flow, hidden, local, result, sink
```

| Declaration | Meaning |
|---|---|
| `@flow(inputs, outputs, state=...)` | A stage that reads engine-internal tensors and/or state, and produces outputs. |
| `hidden(layer, site, phase)` | An input read from the residual stream (e.g. `layer=0` / `layer=-1`, `site="block_output"`, `phase="decode"`). |
| `sink(name)` | An output published to a named sink (other consumers or the sampler bridge subscribe to it). |
| `local()` | An output consumed only inside the program (e.g. fed to an extern). |
| `@extern(inputs, after, state, deadline)` | A training step: mutates the named state, scheduled after its dependencies, bounded by a timing deadline. |
| `result(flow_name, output_name)` | An edge carrying another flow's output **with its autograd graph** — this is how the compiler knows grad mode is required. |

State names (`state="distiller"`) refer to **state banks**: versioned, per-scope (shared or per-request) parameter + optimizer-state stores owned by the runtime. Your module never manages buffers, streams, or events.

## A complete example

See `tllm/consumers/esamp/program.py` — the ESamp algorithm in ~80 lines. The shape is always:

1. `__init__` builds the model and a plain `torch.optim` optimizer (SGD / Adam / AdamW, one parameter group).
2. One or more `@flow` methods compute on hidden-state inputs.
3. One `@extern` method consumes a `result(...)` edge and performs the optimizer step. Its Python body is the **reference oracle**: the compiler lowers the optimizer to a functional transition and checks the two agree (within dtype tolerance) at compile time.

## What the compiler derives

Everything below falls out of the declared DAG — none of it is hand-written per consumer:

- **Slicing**: each flow body is exported via `torch.export`; unsupported constructs (data-dependent control flow, non-tensor leaves) are rejected at compile time.
- **Gradient mode**: if a due extern has a `result()` edge into a flow's output, that flow executes once in grad mode and its outputs serve both sinks (detached) and `autograd.grad`. No double forward.
- **Lane placement**: state-writing externs run on a dedicated training lane; read-only invocations (`run_flow`, used by the sampler bridge) run on a low-priority reader lane; the decode stream waits only on the output events it actually consumes.
- **CUDA graph capture**: when a step is a fixed DAG over persistent state buffers (same flow, same batch shape), the runtime captures the whole step once and replays it — ~3–5 kernel launches instead of ~70–90. Capture correctness is self-checked at capture time (`PG146`).
- **Functional optimizer transitions**: your optimizer becomes a pure-tensor update over versioned slot buffers, so updates can be fenced and replayed safely.

## Error codes

Compilation and runtime rejections use stable codes on `WorkGraphError`:

| Code | Condition |
|---|---|
| PG100 | `result()` references an unknown flow/output |
| PG110 | dependency cycle |
| PG120 | duplicate producer / binding / state name |
| PG130 | export graph contains unsupported structure |
| PG140 | autograd graph saved across stages |
| PG145 | oracle mismatch: extern Python body diverges from the functional transition |
| PG146 | captured CUDA graph replay diverges from the eager transition |
| PG150 | undeclared mutation / unmarked RNG |
| PG160 | unsupported optimizer or optimizer combination |
| PG170 | no legal hook can materialize the declared timing window |
| PG180 | irregular row mapping on request-scoped state |
| PG190 | any attempt at silent fallback (always raised; there is no such path) |

The full decision record lives in the development docs; the contract here is: **fail loudly at compile time, never degrade silently**.

## Sampler guidance

If your consumer also steers sampling, implement a `SamplerModifierProvider` (see `tllm/runtime/sampler_bridge/`). It calls `ProgramConsumer.run_flow(...)` on the reader lane and returns candidate-level logit adjustments; the runtime applies them after top-k/top-p/min-p filtering. ESamp's distiller intervention is the reference implementation.
