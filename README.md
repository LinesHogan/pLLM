<div align="center">

# tLLM

**A test-time intervention layer for vLLM — declare what your algorithm computes, and the compiler makes it fast.**

[Quick Start](#quick-start) · [Documentation](#documentation) · [Performance](#performance) · [中文文档](doc_zh/README.md)

</div>

tLLM is a runtime layer on top of the vLLM v1 inference engine for building producer/consumer extensions: capture model-internal data such as hidden states, route it through stable public ports, and run custom consumers during generation — without maintaining a fork of vLLM.

Its core is a **program compiler**: you declare your algorithm as a small dataflow graph (`@flow` nodes and `@extern` training steps over named state), and tLLM derives the execution strategy from those declarations — stream placement, gradient mode, batching, and whole-step CUDA graph capture all fall out of the dependency edges, not from hand-written orchestration code.

## Latest News

- **[2026-08-21]** The ESamp algorithm is re-implemented as a tLLM Program in ~80 lines. The compiled path runs the full train step in ~3–5 kernel launches per step via whole-step CUDA graph capture, faster than the hand-tuned legacy pipeline in host probes.
- **[2026-08-18]** tLLM reorganized into a strict layered architecture (`core ← contracts ← ports ← workgraph ← runtime ← backends`), enforced by an AST-level import gate; all phase-numbered module names replaced with functional names.

## Quick Start

ESamp — an algorithm that trains a low-rank distiller *during* generation to predict final-layer hidden states from first-layer ones, then uses the predictions to guide sampling — is the reference consumer. Expressed as a tLLM Program, this is the **entire algorithm**:

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

There is no stream, buffer, event, lease, or graph-capture code here. From these declarations the compiler derives:

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

**Legacy hand-tuned pipeline** (measured, RTX 4090): in the aligned 7B min-p ESamp benchmark, optimized ESamp reaches **98.8% of a vLLM baseline** with modern inference optimizations enabled (vLLM V1 engine, CUDA graph execution, FlashInfer sampling, bfloat16, prefix-cache control for fairness, identical sampling workload):

| Model / workload | Optimized vLLM baseline | ESamp (legacy, Triton kernel) | Ratio |
|------------------|-------------------------|-------------------------------|-------|
| Qwen2.5-7B, batch=8, n=16, min-p active path | 5370.6 tok/s | 5304.9 tok/s | 0.9878 |

**Program path** (measured, RTX 4090, synthetic probe: 128 rows × hidden 896, bf16, 50 decode steps): the compiled consumer's train step costs **0.082 ms/step host time vs 0.51 ms/step** for the legacy eager pipeline, and 0.35 ms/step vs 0.55 ms/step wall time — the whole step replays as one captured CUDA graph instead of ~70–90 kernel launches.

The full paired in-engine benchmark (5 alternating rounds, gate: program/legacy ≥ 0.98 output tok/s) is scripted and pending GPU availability; results land in `docs/p5_paired_benchmark_report.json`.

## What You Can Build

- Activation / hidden-state export and editing pipelines.
- Runtime analysis of LLM internals.
- Test-time training algorithms that learn during generation.
- Candidate-level sampler guidance, such as ESamp distiller intervention.

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

- English: [doc/README.md](doc/README.md) — [quickstart](doc/quickstart.md) · [write a consumer](doc/write-a-consumer.md) · [architecture](doc/architecture.md) · [ESamp reference](doc/esamp.md)
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
