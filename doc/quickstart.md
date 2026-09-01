# Quickstart: ESamp and Execution Regions

New native work uses `tllm.workload` (`@workload`/`@task`) and runs as signed
`ExecutionRegion`s. Rust determines Region boundaries and `RegionCaptureMode`;
Python only binds artifacts and resources. The Program example below is kept
for legacy-adapter users, not as the scheduling or stream/event model for new
workloads. See [the workload compiler guide](workload-compiler.md) first.

This page teaches tLLM by implementing **ESamp** — the built-in test-time
training algorithm — as a retained short Program example. Native ESamp is
declared through the Region compiler; this page does not claim that the
compatibility Program controls Region capture or physical scheduling.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
pip install -e ".[vllm027]"
```

Requirements: Python ≥ 3.11, PyTorch with CUDA, vLLM v1 (validated on `vllm==0.27.1`).

## The algorithm

During decode, a tiny low-rank model learns online to predict the
final-layer hidden state from the first-layer hidden state. Its predictions
are then used to nudge candidate-token logits — the model steers itself
while it generates.

## The whole implementation

```python
import torch
import torch.nn.functional as F

from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, hidden, local, result


class ESampProgram(torch.nn.Module):
    """Predict the final-layer hidden state from the first-layer one."""

    def __init__(self, *, hidden_size: int, rank: int, lr: float,
                 device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.down = torch.nn.Linear(hidden_size, rank, bias=False, device=device, dtype=dtype)
        self.up = torch.nn.Linear(rank, hidden_size, bias=False, device=device, dtype=dtype)
        torch.nn.init.zeros_(self.up.weight)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, foreach=False)

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="decode"),
            "target": hidden(layer=-1, site="block_output", phase="decode"),
        },
        outputs={"loss": local()},
        state="distiller",
    )
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = self.up(self.down(source))
        return F.mse_loss(prediction.float(), target.float())

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="distiller",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
```

That is the entire algorithm. Reading the declarations:

- `@flow(inputs=...)` — `forward` reads two engine-internal tensors per
  decode step: `hidden(layer=0, ...)` taps the residual stream after the
  first block, `hidden(layer=-1, ...)` after the last.
- `outputs={"loss": local()}` — the loss stays inside the program (it is
  consumed by the extern below). The shipped version also publishes its
  prediction as `sink("guidance")`, which the sampler bridge subscribes to
  for candidate-level guidance.
- `state="distiller"` — the module's parameters and optimizer state live in
  a versioned **state bank** owned by the runtime. You never manage buffers,
  streams, or events.
- `@extern(inputs={"loss": result("forward", "loss")}, ...)` — `backward`
  consumes `forward`'s loss **with its autograd graph**. This edge is how
  the compiler knows grad mode is required; `deadline=` bounds when the
  update must land.
- The `backward` body is plain PyTorch — and it is also the **oracle**: the
  compiler lowers your optimizer to a functional transition over bank slots
  and verifies the two agree at compile time (`PG145`). No silent fallback.

The shipped compatibility version (`tllm/consumers/esamp/program.py`) is the
same shape with a gated residual model and a separate `predict` flow. This is
an ESamp migration baseline, not a pattern followed by native FTTT.

Canonical FTTT has a strict host-transaction entry point:

```python
import torch
from tllm.canonical.host_runtime import (
    CanonicalHostRuntimeFacts,
    HostTransactionCoordinator,
)
from tllm.canonical.workloads.fttt import (
    FTTTCompileOptions, FTTTConfig, FTTTDefaultJudge,
    FTTTFeedbackService, FTTTSpecialization, FTTTWorkload,
)
from tllm.contracts.tensor_layout import TensorDType

config = FTTTConfig(rank=16, learning_rate=1e-3, max_retries=2)
device = torch.device("cpu")
compiled = FTTTWorkload(config).compile(
    FTTTSpecialization(896, TensorDType.FLOAT32, device, 0),
    FTTTCompileOptions(CanonicalHostRuntimeFacts(device).timeline()),
)
feedback = FTTTFeedbackService(config, FTTTDefaultJudge())
fttt = HostTransactionCoordinator(feedback, compiled.execution)
```

The feedback service converts typed completions and hidden rows into an atomic
host transaction, and the coordinator owns execution plus receipt settlement.
Selection, features, loss, Adam, edit prediction, and publication execute as
Rust-signed tasks. The same graph exposes CPU/CUDA candidates and the generic
scheduler chooses from runtime capabilities; FTTT declares no lane, priority,
fusion, provider, or execution order. The old `tllm.consumers.fttt` /
`TllmRuntime` route is retired with no compatibility wrapper.

Canonical GTTA keeps environment I/O outside GPU state while signing all
stateful computation:

```python
import torch
from tllm.canonical.host_runtime import (
    CanonicalHostRuntimeFacts,
    HostTransactionCoordinator,
)
from tllm.canonical.workloads.gtta import (
    GTTACompileOptions, GTTAConfig, GTTAHostRequest, GTTAPromptPosition,
    GTTAQuery, GTTAWorkload, TrajectoryFeedback,
)
from tllm.canonical.workloads.gtta.environment import GTTAEnvironmentService

config = GTTAConfig(score_momentum=0.25)
compiled = GTTAWorkload(config).compile(GTTACompileOptions(
    CanonicalHostRuntimeFacts(torch.device("cpu")).timeline()
))
environment = GTTAEnvironmentService(config)
gtta = HostTransactionCoordinator(environment, compiled.execution)
environment.enqueue_feedback(TrajectoryFeedback(
    task_id="maps", action="open directions",
    previous_observation="map results", next_observation="directions panel",
    environmental_dynamics="", reward=1.0,
))
result = gtta.run(GTTAHostRequest(GTTAQuery(
    prompt_idx=0, task_id="maps",
    candidate_actions=("clear", "open directions"),
    observation="map results", state_signature="", action="",
    max_records=4, position=GTTAPromptPosition.PREFIX,
)))
```

The typed service owns the bounded FIFO; the Rust-selected CPU tasks perform
sequential EMA update, query and publication. Receipt settlement makes abort
requeue the batch in FIFO order, and an empty batch leaves the instance version
unchanged. The old `tllm.consumers.gtta`/`TllmRuntime` implementation has been
deleted rather than retained as a second authority.

## Run it

```python
from vllm import SamplingParams
from tllm import make_llm
from tllm.consumers.esamp import ESampConsumerConfig, activate

activate(ESampConsumerConfig(implementation="program"))
llm = make_llm(model_name="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.8, n=8))
```

Or the starter script (Qwen2.5-7B, 16 parallel answers, prints loss and
intervention counters):

```bash
python starter.py --max-new-tokens 32
```

## What the legacy compiler did with your 40 lines

- **Sliced** each flow into `torch.export` graphs (unsupported constructs →
  compile-time `PG###` error).
- **Scheduled** training on a dedicated compatibility lane and read-only predictions on a
  low-priority reader lane — the decode stream only waits on the events it
  actually consumes. All edges come from your declarations.
- **Captured** eligible legacy train steps into a CUDA graph per batch shape
  (~3–5 kernel launches instead of ~70–90), self-verified at capture time.
- **Lowered** `torch.optim.AdamW` into a functional update over versioned
  bank slots, checked against your `backward` body.

## Next steps

- [Write a consumer](write-a-consumer.md) — the full declaration vocabulary
  (including caller-supplied `provided()` inputs), the thin consumer shell,
  and the error-code table.
- [ESamp reference](esamp.md) — config knobs, the legacy baseline, and
  measured numbers.
- [Adapting an In-Place-TTT-style algorithm](inplace-ttt.md) — when your
  algorithm edits the model's own weights instead of a side module.

## Troubleshooting

- **A `PG###` error at startup?** The compiler rejecting an unsupported
  construct — see the [error code table](write-a-consumer.md#error-codes).
  These are fail-closed; nothing silently degrades.
- **CUDA graph capture disabled?** Capture is controlled for native workloads
  by signed `RegionCaptureMode`; legacy Program capture is a separate
  compatibility behavior.
- **Throughput sanity checks**: the benchmark workflow reports functional
  counters (loss count, intervention states) alongside tok/s, so a broken
  no-op consumer cannot masquerade as a fast one.
