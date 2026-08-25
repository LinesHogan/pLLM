<div align="center">

# tLLM

**An auxiliary-workload compiler middle-end for vLLM.**

[Quick Start](#quick-start) · [Documentation](#documentation) · [Performance](#performance) · [中文文档](doc_zh/README.md)

</div>

tLLM treats vLLM as a pinned main workload and compiles developer-defined
auxiliary workloads around the real lifecycle boundaries exported by its
backend. It does not capture or reorder vLLM's complete FX/ATen graph.

`tllm.workload` is the current declaration surface. The task graph is
fine-grained for analysis, while `ExecutionRegion` is the coarse-grained
physical execution unit. Rust validates and signs a Region-authoritative
`ScheduleDecision`; Python only mechanically binds its artifact, resources,
anchor, and explicit cross-Region fences. Existing
Program and `BaseConsumer` implementations remain migration compatibility
inputs for workloads that have not completed native takeover; they are not the
optimization model. A workload without an eligible native executor or complete
legacy adapter fails before vLLM construction instead of being silently skipped.

> [!WARNING]
> **Recirculation/model-region support is ongoing experimental work.** The
> compiler has finite CPU-request serial/wavefront Regions, but it does not
> yet have a verified native vLLM prefill row/model-Region binding. Do not
> treat it as taking over vLLM prefill. For Recirculation, CUDA Graph,
> EngineCore multiprocessing, decode recurrence, speculative decoding, PP/DP,
> and general model-family support are not complete. Unsupported configurations
> fail closed instead of silently falling back. See the
> [Recirculation implementation notes](tllm/consumers/Recirculation/introduce.md).

Its core is a compiler middle-end with three data contracts:
`MainWorkloadTimeline`, fine-grained `ScheduleGraph`, and Region-authoritative
`ScheduleDecision`. The legacy `@flow`/`@extern` Program model is lowered into
the same dependence/effect vocabulary only during migration.

**Current GPU correctness evidence (not a throughput result):** with
`torch==2.13.0+cu129` and `vllm==0.27.1`, native Dummy and native ESamp passed
exact eager, in-process MRv1/MRv2 short smokes. The ESamp smoke reported
`loss_count=6`, `update_count=6`, `state_version=6`, and `fallback_count=0`.
The same native ESamp exact eager Region also passed EngineCore MP correctness
on MRv1/MRv2: loss, update, state-version, and sampler-intervention counters
were nonzero, with `fallback_count=0` and `replay_count=0`. This establishes
non-capture execution correctness only. No throughput result is claimed; the
model-backed native piecewise CUDA Graph path, real native Recirculation vLLM
prefill takeover, and shape-miss epoch switching remain unverified or
incomplete.

The canonical declaration surface uses one node decorator:

```python
from tllm.workload import (
    AdamWSpec, Before, Between, State, StateScope,
    hidden, read, request_key, result, sink, task, temporary,
    transition, workload,
)

@workload(id="esamp")
class ESamp:
    distiller = State(
        scope=StateScope.REQUEST,
        key=request_key("request_id", "sample_idx"),
        parameters="model",
        optimizer=AdamWSpec(lr=1e-3),
    )

    @task(
        inputs={"source": hidden(0), "target": hidden(-1)},
        outputs={"prediction": sink("guidance"), "loss": temporary()},
        effects=(read(distiller),),
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def predict_and_loss(self, source, target): ...

    @task(
        inputs={"loss": result("predict_and_loss", "loss")},
        effects=(transition(distiller),),
        timing=Before("next.prepare"),
    )
    def update(self, loss): ...
```

See [the workload compiler guide](doc/workload-compiler.md) for the current
contract and migration boundary.

## Migration Note

The Program example below documents a retained compatibility API. New native
workloads should use `tllm.workload` and the Region compiler guide; neither
per-task stream placement nor CUDA-event choreography is part of the current
planning contract.

## Quick Start

ESamp — an algorithm that trains a low-rank distiller *during* generation to predict final-layer hidden states from first-layer ones, then uses the predictions to guide sampling — remains a useful compatibility example. The native reference is the `tllm.workload` declaration near the top of this README; the following Program is retained for legacy adapters:

```python
import torch
import torch.nn.functional as F

from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, hidden, local, result, sink


class ESampProgram(torch.nn.Module):
    """Predict the final-layer hidden state from the first-layer one."""

    def __init__(self, *, hidden_size: int, rank: int, lr: float,
                 device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.model = LowRankGatedResidualModel(
            hidden_size=hidden_size, rank=rank,
            use_output_layernorm=True, device=device, dtype=dtype,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, foreach=False
        )

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="decode"),
            "target": hidden(layer=-1, site="block_output", phase="decode"),
        },
        outputs={"prediction": sink("guidance"), "loss": local()},
        state="distiller",
    )
    def forward(self, source, target):
        prediction = self.model(source)
        return prediction, F.mse_loss(prediction.float(), target.float())

    @flow(
        inputs={"source": hidden(layer=0, site="block_output", phase="decode")},
        outputs={"prediction": local()},
        state="distiller",
    )
    def predict(self, source):
        # Invoked on demand by the sampler bridge against the current
        # bank state.
        return self.model(source)

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="distiller",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
```

There is no stream, buffer, event, lease, or graph-capture code here. In this legacy lane the runtime derives:

- **Gradient mode from dataflow.** The `result("forward", "loss")` edge asks for a value *with* its autograd graph, so when the training extern is due, `forward` runs once in grad mode; the outputs serve the sink and the loss serves `autograd.grad` — no double forward.
- **Lane placement from node roles.** State-writing externs run on a dedicated training lane; read-only `run_flow` calls (sampler predictions) run on a low-priority reader lane fenced by events; the decode stream only ever waits on the single output event it consumes.
- **Whole-step CUDA graph capture.** When a step is a fixed DAG over persistent state buffers, the runtime captures it once per batch shape and replays it thereafter — one input copy plus one `graph.replay()` per step.
- **Functional optimizer transitions.** The declared AdamW step is lowered to a pure-tensor functional update, verified at compile time against the Python body (error code `PG145` on divergence), so updates can be versioned, fenced, and captured.

Run it against vLLM:

```python
from tllm import make_llm
from tllm.consumers.esamp import ESampConsumerConfig, activate

activate(ESampConsumerConfig(implementation="program"))
llm = make_llm(model_name="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
outputs = llm.generate(prompts, sampling_params)
```

## From vLLM Generation to tLLM Algorithms

If you already have a vLLM generation script, tLLM keeps the same prompts and `SamplingParams` and adds your algorithm before `generate`:

```diff
- from vllm import LLM, SamplingParams
+ from vllm import SamplingParams
+ from tllm import make_llm, register_consumer
+ from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig

+ register_consumer(ESampConsumer(ESampConsumerConfig()))
+ llm = make_llm(model_name="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
  outputs = llm.generate(
      [f"Surprise me with an unexpected story about {i} evil sorcerers." for i in range(2, 16)],
      SamplingParams(max_tokens=64, temperature=0.8, n=8),
  )
```

`make_llm` installs tLLM's vLLM v1 runtime hooks; `register_consumer` (or `activate`, which builds from config) attaches the consumer that uses them.

## Why tLLM

| Task | Direct vLLM modification | With tLLM |
|------|--------------------------|-----------|
| Read hidden states during decode | Patch runner internals, chase vLLM releases | Declare a `@flow` over public ports |
| Train a model during generation | Hand-own streams, events, buffers, CUDA graphs | Declare an `@extern`; the compiler derives lanes and capture |
| Add sampler guidance | Patch logits/sampler code directly | Implement a sampler provider behind tLLM's bridge |
| Keep throughput measurable | Easy to accidentally benchmark a broken no-op | Standard ratio checks plus functional counters |
| Stay fast | Deep vLLM internals knowledge required | Declared DAG → compiler-derived optimization |

## Performance

The figures below are historical legacy/program measurements. They do not
establish Region throughput; no Region throughput claim is currently made.

**Legacy hand-tuned pipeline** (measured, RTX 4090): in the aligned 7B min-p ESamp benchmark, optimized ESamp reaches **98.8% of a vLLM baseline** with modern inference optimizations enabled (vLLM V1 engine, CUDA graph execution, FlashInfer sampling, bfloat16, prefix-cache control for fairness, identical sampling workload):

| Model / workload | Optimized vLLM baseline | ESamp (legacy, Triton kernel) | Ratio |
|------------------|-------------------------|-------------------------------|-------|
| Qwen2.5-7B, batch=8, n=16, min-p active path | 5370.6 tok/s | 5304.9 tok/s | 0.9878 |

**Program path** (measured, RTX 4090, synthetic probe: 128 rows × hidden 896, bf16, 50 decode steps): the compiled consumer's train step costs **0.082 ms/step host time vs 0.51 ms/step** for the legacy eager pipeline, and 0.35 ms/step vs 0.55 ms/step wall time — the whole step replays as one captured CUDA graph instead of ~70–90 kernel launches.

The full paired in-engine benchmark (5 alternating rounds, gate: program/legacy ≥ 0.98 output tok/s) is scripted and pending GPU availability; results land in `docs/p5_paired_benchmark_report.json`.

## What You Can Build

- Test-time training algorithms that learn during generation. `esamp`, `ttrl`, `fttt`, and `override` use `tllm.program`; TLM uses a full HF/PEFT side model because its official-family update trains internal `q_proj/v_proj` LoRA rather than a hidden-to-logits surrogate.
- Candidate-level sampler guidance, such as ESamp distiller intervention.
- Algorithms that edit the model's own weights (In-Place-TTT style) via the fast-weight overlay — see [adapting In-Place TTT](doc/inplace-ttt.md).
- Experimental prefill recurrences such as Recirculation through the eager
  model-region/KV materializer. This path is ongoing and is not yet covered by
  the CUDA Graph performance claims above.
- Activation / hidden-state export and editing pipelines; runtime analysis of LLM internals.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
pip install -e ".[vllm027]"

python starter.py --max-new-tokens 32
```

The starter runs ESamp with `Qwen/Qwen2.5-7B-Instruct`, generates 16 answers in parallel, and prints runtime adaptation statistics. It defaults to `--seed-mode shared`, which keeps the FlashInfer sampler backend available; use `--seed-mode per-request` if independent `seed + i` request streams matter more.

## Documentation

- English: [doc/README.md](doc/README.md) — [quickstart](doc/quickstart.md) · [write a consumer](doc/write-a-consumer.md) · [architecture](doc/architecture.md) · [ESamp reference](doc/esamp.md) · [adapting In-Place TTT](doc/inplace-ttt.md) · [TLM q/v LoRA](doc/tlm.md)
- 中文: [doc_zh/README.md](doc_zh/README.md)

## Requirements

- Python ≥ 3.11
- vLLM v1 engine (validated on `vllm==0.27.1`)
- PyTorch with CUDA

## FAQ

> What is the name "tLLM" for?

tLLM is a test-time intervention layer for vLLM. The name is a recursive-style abbreviation: **tLLM is a test-time intervention layer for vLLM**.

> I have a test-time training algorithm. How do I implement it with tLLM?

Write it as a `torch.nn.Module` with `@flow` and `@extern` declarations — see [write a consumer](doc/write-a-consumer.md). If your algorithm needs something the program language rejects, the compiler fails loudly with a stable error code (`PG###`) instead of silently falling back, so you always know what is and isn't supported.

## License

Apache-2.0.
