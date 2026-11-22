# pLLM

> **[中文文档](README.md)** · [Technical Blog Series (EN/ZH)](blog/README.md) · [Agent Skill](doc/agent/SKILL.md) · [Migration Guide](doc/agent/migration.md) · [Agent Reference](doc/agent/reference.md) · [Documentation Index (中文)](doc/README.md)

Prototyping an inference-time LLM algorithm—activation steering, linear probes, per-request LoRA, or decode-time test-time training—takes a few dozen lines of PyTorch on a single sequence. Running that same logic inside vLLM runs into continuous batching (packed, dynamically permuted token rows), CUDA graphs (Python hooks do not execute during replay), per-request paged state allocation, and side-stream synchronization.

pLLM acts like an eBPF layer for vLLM: instead of patching `GPUModelRunner` or maintaining a fork, you subclass `pllm.Workload` to declare **what internal activations to read, what state each request or engine owns, and where the result goes**, and write pure PyTorch tensor math. pLLM verifies scheduling contracts on CPU before touching a GPU, then compiles your math, autograd, and state transitions into vLLM's continuous batching and CUDA graphs.

## What pLLM Does (in 30 Seconds)

| Inference-time need | Why raw PyTorch / vLLM patches break | How you declare it in pLLM |
| --- | --- | --- |
| **Read hidden states or per-layer logits** (probes, SAEs, Logit Lens, DoLa) | Requests interleave every step; Python hooks never fire inside replayed CUDA graphs | `pllm.hidden(layer)`, `pllm.residual(layer)`, `pllm.layer_logits(...)`, `@pllm.export` |
| **Modify activations or weights per request / session** (CAA, ReFT, multi-tenant LoRA) | Manual slot allocation, eviction, and quantized projection overlays leak or corrupt state | `pllm.RequestState` + `pllm.model.ActivationPatch` / `pllm.model.Adapter` |
| **Train parameters during decode** (ESamp, In-Place TTT, Test-Time Learning, TTRL) | Synchronous backward steps stall decoding; concurrent streams tear in-place weights | `@pllm.train` / `@pllm.train_in_forward` (CUDA-graph captured and side-stream overlapped) |
| **Rewrite forward loops, attention reads, or KV layouts** (early exit, looped depth, window + sink KV, lookup memory) | Custom layer schedules and paged KV reads easily violate block-table invariants | `@pllm.custom_forward`, `@pllm.custom_attention`, `@pllm.custom_residual`, `pllm.input_tokens` |

A minimal layer-contrast workload, planned and contract-checked on CPU without loading weights or initializing CUDA:

```python
import torch

import pllm
from pllm import testing


class LayerContrast(pllm.Workload):
    @pllm.add_logits
    def contrast(self, logits: torch.Tensor = pllm.layer_logits((12, -1))) -> torch.Tensor:
        return 0.5 * (logits[:, 1] - logits[:, 0])


config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=16)
print(testing.check(LayerContrast(), config))
```

---

## Hand pLLM to Your Coding Agent in 1 Minute

`pllm` ships its authoring rules, generated API signatures, error catalog, and codebase migration guide inside the Python package. From your project repository:

```bash
pllm skill --install claude   # writes .claude/skills/pllm/*.md
# or:
pllm skill --install codex    # writes .agents/skills/pllm/*.md
pllm skill --install cursor   # writes .cursor/rules/pllm/*.md
# or print to stdout for any LLM context: pllm skill --print && pllm skill --print migration
```

Then give your agent this prompt:

> Read the installed `pllm` skill (`SKILL.md` and `migration.md`, consulting `reference.md` and `pitfalls.md` as needed). Audit any existing Hugging Face hooks, `LogitsProcessor`, PEFT adapters, or custom PyTorch inference loops in this repo against `migration.md`'s pre-flight table, flag anything outside pLLM's supported scope, implement an importable `pllm.Workload`, verify `pllm.explain(...)` and `pllm.testing.check(...)` on CPU first, and wire it into `pllm.make_llm`, `pllm.make_async_llm`, or `pllm serve`.

---

## Online Serving & Runtime Entry Points

A single `pllm.Workload` runs across three execution entry points:

- **OpenAI-compatible HTTP server (`pllm serve`)**: continuous per-request admission, SSE token streaming, disconnect cancellation, and automatic state retirement:
  ```bash
  pllm serve \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --workload pllm.workloads:DoLaWorkload \
    --port 8000 \
    --gpu-memory-utilization 0.85
  ```
  When a workload declares `pllm.RequestState(..., key_fields=("route",))`, callers persist and isolate state per conversation across turns via `vllm_xargs`:
  ```python
  # check-docs: skip: connects to a running pllm serve endpoint
  from openai import OpenAI

  client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
  stream = client.chat.completions.create(
      model="Qwen/Qwen2.5-0.5B-Instruct",
      messages=[{"role": "user", "content": "Explain KV caching in two sentences."}],
      stream=True,
      extra_body={"vllm_xargs": {"pllm_route": "session-42"}},
  )
  ```
- **Embedded async Python handle (`pllm.make_async_llm`)** and **Synchronous handle (`pllm.make_llm`)**: share the same runtime control plane. `ModelConfig(..., multiprocessing="auto")` defaults to an isolated EngineCore subprocess for `make_async_llm` / `pllm serve` and in-process execution for `make_llm`, while both support concurrent streaming (`stream()`, `generate()`, `abort()`, `diagnostics()`), live `import_state()` / `train_batch()` / `feed()` / `generate_feedback()`, and optional workloads (`None`).

---

## Example: Decode-Time Per-Request Distillation (ESamp)

Each concurrent request trains its own rank-4 MLP online to predict final-layer hidden states from layer 0, while sampling pushes away from tokens already explained by that shallow prediction:

```python
from dataclasses import dataclass
from torch import nn


@dataclass(frozen=True)
class ESampLite(pllm.Workload):
    beta: float = 0.5
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)


esamp_cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=128)
print(pllm.explain(ESampLite(), esamp_cfg))
print(testing.check(ESampLite(), esamp_cfg))
```

Measured overhead on a single RTX 4090 running Qwen2.5-7B-Instruct (8 prompts × 16 samples, 256 tokens each): this architecture reaches 97.4% of native vLLM throughput when training online without guidance, and 96.3% with both online training and sampling guidance enabled ([measurements.toml](doc/reference/measurements.toml)). On a 4B In-Place TTT community checkpoint (vLLM 0.32.0, CUDA graphs on, 256 greedy tokens at 4096 / 16384 prompt tokens), pLLM runs at 59.64 / 42.27 token/s for a single request vs 54.79 / 34.89 token/s for the official Hugging Face implementation, and 111.38 token/s across 4 concurrent requests.

---

## Installation & vLLM Compatibility (0.27 – 0.34)

Requires Python 3.11+, a Rust toolchain when building from source, and **vLLM 0.27 through 0.34** (`torch>=2.13,<2.16`, `triton>=3.7,<4`):

- **Install into an existing vLLM 0.27–0.34 environment:**
  ```bash
  python -m pip install -e . --no-deps   # or `pip install -e .` to pull pure-Python dependencies
  ```
- **Create a fresh virtual environment:**
  ```bash
  python -m venv .venv && source .venv/bin/activate
  python -m pip install -e ".[vllm032,test]"   # or "[vllm034,test]", "[vllm027,test]", "[vllm,test]"
  ```
- **CPU-only authoring (`pllm.explain` & `pllm.testing.check`):** install `".[test]"`. pLLM registers no global vLLM plugins.

Admitted decoders cover 15 architectures: Qwen2 / Qwen2.5, Llama, Mistral, Phi-3, GLM-4, Gemma 2 / 3, Mixtral, Qwen2-MoE, Qwen3, Qwen3-MoE, Qwen3.5 / Qwen3.5-MoE, DeepSeek V2 / V3 (including latent MLA `@pllm.custom_attention` and sublayer residuals), and gpt-oss, with GPTQ / AWQ / FP8 quantization and single-node tensor parallelism (`tensor_parallel_size > 1` requires `vllm >= 0.32.0`). `0.27.1` and `0.32.0` serve as full GPU regression baselines in this repository; `0.33`–`0.34` pull PyTorch `2.14+cu130` and require a CUDA 13 NVIDIA driver. Pretraining new architectures from scratch and score-based physical KV eviction (H2O, SnapKV) are out of scope; see [rules.md](doc/agent/rules.md) and [support-matrix.md](doc/reference/support-matrix.md).

## Documentation & Built-in Workloads

- **Standalone Engineering Essays & Deep Dives (EN & ZH)**: [blog/README.md](blog/README.md) — ten independent pieces on post-*Bitter Lesson* inference economics, 40-line timeline scheduling vs. polyhedral IRs, PCIe/weight bandwidth quotas, IEEE 754 bit-flipping at the decode tail, compiled fast-weight training, frozen CUDA Graph pointer tables, lock-free GPU MVCC/RCU, GQA/MLA attention layouts, on-GPU `attn_metadata` unpacking, and schedule-as-data multi-tenant verification.

- **Reference workloads (`pllm.workloads`)**: `ESampWorkload`, `DoLaWorkload`, `ReFTWorkload`, `OverRIDEWorkload`, `OutputLoRAWorkload`, `InPlaceTTTWorkload`, `PrefillAdaptersWorkload`, `RecirculationWorkload`, `StepLoopWorkload`, `RewardHeadWorkload`, `TLMWorkload`, `FTTTWorkload`, `DummyWorkload`.
- **English Agent & API Docs**: [SKILL.md](doc/agent/SKILL.md) · [migration.md](doc/agent/migration.md) · [reference.md](doc/agent/reference.md) · [examples.md](doc/agent/examples.md) · [rules.md](doc/agent/rules.md) · [pitfalls.md](doc/agent/pitfalls.md) · [llms.txt](llms.txt)
- **Chinese Guides & Tutorials**: [doc/README.md](doc/README.md) · [Online Serving Guide](doc/howto/online-serving.md) · [Production Guide](doc/howto/production.md)
