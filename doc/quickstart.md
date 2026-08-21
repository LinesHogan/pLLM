# Quickstart

## Requirements

- Python ≥ 3.11, PyTorch with CUDA, vLLM v1 (validated on `vllm==0.27.1`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
pip install -e ".[vllm027]"
```

## Run the starter

```bash
python starter.py --max-new-tokens 32
```

This runs ESamp (the built-in test-time training consumer) with `Qwen/Qwen2.5-7B-Instruct`, generates 16 answers in parallel, and prints adaptation statistics: training loss counters, sampler-intervention counters, and port publish hits.

Useful flags:

- `--seed-mode shared` (default) keeps the FlashInfer sampler backend; `--seed-mode per-request` gives each request an independent `seed + i` stream at the cost of that backend.
- `--enable-distiller-intervention / --no-enable-distiller-intervention` toggles sampler guidance.
- `--implementation program|legacy` selects the compiled consumer or the hand-tuned baseline.

## Use tLLM in your own script

```python
from vllm import SamplingParams
from tllm import make_llm, register_consumer
from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig

register_consumer(ESampConsumer(ESampConsumerConfig()))
llm = make_llm(model_name="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.8, n=8))
```

- `make_llm(...)` wraps `vllm.LLM` and installs tLLM's v1 runtime hooks (the entry point rehydrates them in every vLLM process under multiprocessing).
- `register_consumer(...)` attaches a consumer instance. To build from config instead, use `tllm.consumers.esamp.activate(ESampConsumerConfig(...))`.

## Troubleshooting

- **CUDA graph capture disabled?** The program path captures whole train steps automatically on CUDA; CPU execution stays eager by design.
- **A `PG###` error at startup?** That's the program compiler rejecting an unsupported construct — see the [error code table](write-a-consumer.md#error-codes). These are fail-closed; nothing silently degrades to a slower path without telling you.
- **Throughput sanity checks**: the benchmark workflow reports functional counters (loss count, intervention states) alongside tok/s, so a broken no-op consumer cannot masquerade as a fast one.

Next: [write your own consumer](write-a-consumer.md).
