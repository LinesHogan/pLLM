# Write a Consumer

## Native workload first

For new code, write a `tllm.workload` declaration using `@workload`, `@task`,
typed values, effects, timing windows, and optional affine domains/accesses.
Rust then creates and signs `ExecutionRegion`s; Python has no authority to
split them, pick their order, infer capture behavior, or add per-task stream
placement. `RegionCaptureMode` is a Rust-signed closed enum, and only explicit
cross-Region dependencies may create fences/events. See
[the workload compiler guide](workload-compiler.md) for the contract.

### FTTT: a typed host-service boundary

FTTT illustrates how host work remains explicit without regaining workload
authority:

```python
from tllm.consumers.fttt import FTTTConfig, FTTTFeedbackService, FTTTWorkload
from tllm.runtime import TllmRuntime

config = FTTTConfig(rank=16, learning_rate=1e-3, max_retries=2)
workload = FTTTWorkload(config)
runtime = TllmRuntime(workloads=(workload,))
feedback = FTTTFeedbackService(config)
```

The service judges completed trajectories, owns the bounded feedback queue,
and produces an `FTTTFeedbackTransaction`. `stage_next(runtime)` atomically
stages a typed plan with its hidden rows. Selection, feature construction,
loss, Adam, edit prediction, state-version reads, and closed-loop publication
are native signed tasks. `commit()` accepts only the intent published for that
transaction and acknowledges lifecycle state exactly once; abort discards the
dependent projection only while its staged inputs are still pending. Once the
signed graph consumes those inputs, a matching published intent is settled
even if user code raises. Consumption without a matching publication
fail-stops the runtime; optimizer state is never presented as rolled back.

None of those semantics select a physical schedule. FTTT declares dependencies,
effects, and legal timing windows; the generic Rust scheduler automatically
chooses Region formation, placement, and overlap. Workloads do not declare
lanes, priorities, or fusion strategies.

The rest of this page describes the retained Program/`BaseConsumer` API for
legacy adapters. It is useful while a workload has not completed native
takeover, but its lanes, streams, events, and capture behavior are not the
current optimization model.

A consumer is a `torch.nn.Module` whose methods are annotated with the tLLM declaration vocabulary. You declare **what** depends on **what**; the compiler decides **where** and **how** it runs.

## Legacy Program declaration vocabulary

```python
from tllm.workgraph.frontend import extern, flow, hidden, local, provided, result, sink
```

| Declaration | Meaning |
|---|---|
| `@flow(inputs, outputs, state=...)` | A stage that reads engine-internal tensors and/or state, and produces outputs. |
| `hidden(layer, site, phase)` | An input read from the residual stream (e.g. `layer=0` / `layer=-1`, `site="block_output"`, `phase="decode"`). |
| `provided()` | An input supplied by the caller as a plain tensor (no engine tap). The consumer builds the bundle itself with `provided_bundle(...)`; the lease entry key is the declared input name. |
| `sink(name)` | An output published to a named sink (other consumers or the sampler bridge subscribe to it). |
| `local()` | An output consumed only inside the program (e.g. fed to an extern). |
| `@extern(inputs, after, state, deadline)` | A training step: mutates the named state, scheduled after its dependencies, bounded by a timing deadline. |
| `result(flow_name, output_name)` | An edge carrying another flow's output **with its autograd graph** — this is how the compiler knows grad mode is required. |

State names (`state="distiller"`) refer to **state banks**: versioned, per-scope (shared or per-request) parameter + optimizer-state stores owned by the runtime. Your module never manages buffers, streams, or events.

## The thin consumer shell

The compiled `ProgramConsumer` is driven by a small wrapper: declare what
the engine should feed you (`flows()`), compile lazily once real shapes
arrive, and delegate. This pattern applies only to retained compatibility
implementations, not to native FTTT:

```python
class MyConsumer(BaseConsumer[MyConfig]):
    def _ensure_compiled(self, hidden: torch.Tensor) -> None:
        if self._compiled is not None:
            return
        program = MyProgram(hidden_size=int(hidden.shape[1]), ...)
        rows = 2  # export requires example rows strictly above the dynamic lower bound
        self._compiled = compile_program(
            program,
            states={"adapter": StateSpec(scope=StateScope.REPLICA, optimizer=torch.optim.Adam)},
            examples={"forward": (torch.zeros(rows, hidden.shape[1]), ...)},
            dynamic_dims={
                # one DynamicDim per row-dynamic input, a distinct object each
                "forward": tuple(
                    DynamicDim(value_path=(i,), dimension=0, lower_bound=1, upper_bound=65536)
                    for i in range(num_inputs)
                ),
            },
        )

    def train(self, hidden: torch.Tensor, ...) -> None:
        self._ensure_compiled(hidden)
        bundle = provided_bundle(consumer_id=self.consumer_id, step_id=self._step_id,
                                 entries={"hidden": hidden, ...})
        self._compiled.consume_flow(self._compiled.flow_ordinal("forward"), bundle, None)

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        self._ensure_compiled(hidden)
        return self._compiled.run_flow("predict", hidden)["score"]
```

Rules of thumb:

- **Tensor math goes in the Program; control logic stays plain Python.**
  Judging, voting, queues, lifecycle state machines, row grouping — all of
  that lives on the consumer, not in flows.
- **Make scalar/switch inputs per-row tensors** (`head_idx: (rows,)`,
  `scale: (rows, 1)`). Bundle entries must share a row count, and per-row
  selection (`index_select` + `bmm`) exports cleanly where Python indexing
  of a `ModuleList` does not.
- **Every `DynamicDim` object is used for exactly one input** — reusing the
  same instance twice fails capture ("declared more than once").
- **Read stats from the compiled consumer** (`update_count`, `loss_count`,
  `loss_sum` deltas); never reimplement counters.
- Programs compile lazily at first real input and **fail closed**: an
  unsupported configuration raises at compile time, never silently falls
  back.

The retained ESamp migration baseline in `tllm/consumers/esamp/` is the
reference for this compatibility surface. Do not use native FTTT as a Program
or `BaseConsumer` example: its host lifecycle is the typed service described
above, and its schedulable computation belongs to the signed workload graph.


## A complete example

See `tllm/consumers/esamp/program.py` — the ESamp algorithm in ~80 lines. The shape is always:

1. `__init__` builds the model and a plain `torch.optim` optimizer (SGD / Adam / AdamW, one parameter group).
2. One or more `@flow` methods compute on hidden-state inputs.
3. One `@extern` method consumes a `result(...)` edge and performs the optimizer step. Its Python body is the **reference oracle**: the compiler lowers the optimizer to a functional transition and checks the two agree (within dtype tolerance) at compile time.

## What the legacy runtime derives

Everything below falls out of the declared DAG — none of it is hand-written per consumer:

- **Slicing**: each flow body is exported via `torch.export`; unsupported constructs (data-dependent control flow, non-tensor leaves) are rejected at compile time.
- **Gradient mode**: if a due extern has a `result()` edge into a flow's output, that flow executes once in grad mode and its outputs serve both sinks (detached) and `autograd.grad`. No double forward.
- **Lane placement**: state-writing externs run on a dedicated training lane; read-only invocations (`run_flow`, used by the sampler bridge) run on a low-priority reader lane; the decode stream waits only on the output events it actually consumes. This is compatibility-runtime behavior.
- **CUDA graph capture**: eligible legacy fixed DAGs may be captured and replayed. Native capture instead follows signed `RegionCaptureMode` and Region graph safety.
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
