# Migrating an existing HF / PyTorch / vLLM project to pLLM

Use this guide when converting an existing Hugging Face `transformers`, PyTorch hook, PEFT, or custom vLLM
prototype into a `pllm.Workload` and wiring it into an engine entry point. Read [rules.md](rules.md) for
invariants and [pitfalls.md](pitfalls.md) for rejections and contract failures.

## 1. Pre-flight codebase audit

Map every custom piece of the existing codebase before writing code:

| Existing PyTorch / HF / vLLM pattern | pLLM declaration | Key checks and limits |
| --- | --- | --- |
| `module.register_forward_hook` reading activations (`hidden_states`, probes, SAEs) | `pllm.hidden(layer)`, `pllm.residual(layer)`, `pllm.final_hidden()`, `pllm.layer_logits(layers)` + `@pllm.export` | Verify whether the original hook reads the MLP branch output (`hidden`) or the post-layer residual stream (`residual`, matching HF `hidden_states[layer + 1]`). |
| Forward hook adding a steering vector (`x + alpha * v`) | `pllm.model.ActivationPatch` + `pllm.InstanceState` (or `pllm.RequestState`) | Stays inside CUDA graphs. Use `Zeros`, `Computed`, `Pretrained`, or `Imported` (`llm.import_state(...)` / `stream(..., state_import=...)` also work online). |
| `transformers.LogitsProcessor` / custom logit bias | `@pllm.add_logits` returning `[rows, vocab_size]`, or `@pllm.guide_sampling(beta=...)` returning `[rows, hidden_size]` | Compute along `dim=-1` only; axis 0 packs concurrent requests in scheduler order. Prefer `@pllm.guide_sampling` over multiplying the LM head yourself. |
| Per-request or per-tenant LoRA (`peft` adapter dict swapped per prompt) | `pllm.RequestState` or `pllm.InstanceState` + `pllm.model.Adapter(..., operator="low_rank")` | Both stay in CUDA graphs. Updated via `@pllm.write`, `@pllm.train`, `@pllm.train_in_forward`, `generate(batch=StateImport(...))`, or live `llm.import_state(...)`. Add `key_fields=("route",)` for multi-turn session state. |
| Inner optimization loop per token / per request (TTT, ESamp, fast weights) | `@pllm.train(state=...)` + `pllm.AdamW` or `@pllm.train_in_forward` + `pllm.SGD` | Request-row `@pllm.train` returns `[rows]` losses (never `.mean()`). Never mutate state or input tensors in place. |
| Early exit / dynamic layer skipping / looped depth | `@pllm.custom_forward` + `pllm.model.Layers` (`active=` on decode) | `active=` requires `tokens="decode"`; carry cross-position state in `pllm.PositionState`. |
| Custom attention masking / top-K KV read / sliding window + sink / MLA latent reads | `@pllm.custom_attention` + `pllm.model.PagedKV` | Works on dense/KV-MoE decoders and DeepSeek V2/V3 MLA (`keep="all"`, absorbed scoring in compressed latent space). Supports `TP > 1` with `keep="all"` and no cross-head `pllm.attention(...)` facts. Requires unquantized KV and prefix caching off. |
| Host feedback / RL loop (`generate -> judge -> update`) | `make_llm` or `make_async_llm` + `generate_completions` / `generate_feedback` + `train_batch` | `import_state`, `train_batch` (including `CausalTrainBatch`), `feed`, `generate_completions`, and `generate_feedback` share a unified control plane across sync and async engines. |
| Out-of-scope mechanisms (stop and report to the user) | — | Pretraining new architectures, score-based physical KV eviction (H2O / SnapKV), speculative decoding / MTP, non-admitted model classes (outside Qwen2/2.5, Llama, Mistral, Phi3, GLM-4, Gemma 2/3, Mixtral, Qwen2-MoE, Qwen3, Qwen3-MoE, Qwen3.5 / Qwen3.5-MoE, DeepSeek V2/V3, and gpt-oss). |

## 2. Translate stateful hooks into a declarative Workload

Single-request Hugging Face prototypes typically stash activations in a mutable Python dict inside a forward
hook and consume them inside a `LogitsProcessor`:

```python
# Before (single-request HF prototype; breaks under continuous batching and CUDA graphs):
#   cache = {}
#   handle = model.model.layers[12].register_forward_hook(lambda m, i, o: cache.__setitem__("r", o[0]))
#   # later before sampling: push away from proj(cache["r"][:, -1]) projected through lm_head
```

Replace Python dicts, hooks, and ad-hoc LM-head projections with a frozen `pllm.Workload` holding state
declarations, input markers, and pure row-wise tensor math:

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing


@dataclass(frozen=True)
class ResidualSteer(pllm.Workload):
    beta: float = 0.25

    probe = pllm.InstanceState(
        init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, m.hidden_size, bias=False)),
    )

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def steer(
        self,
        r: torch.Tensor = pllm.residual(12),
        proj: nn.Module = pllm.module("probe"),
    ) -> torch.Tensor:
        return proj(r)


# Verify execution plan and multi-request scheduling contracts on CPU before touching a GPU.
config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
report = pllm.explain(ResidualSteer(), config)
result = testing.check(ResidualSteer(), config, neutral=ResidualSteer(beta=0.0))
assert report.eager_reason == ""
assert "neutral" in result.passed
print(result)
```

Four rules eliminate most migration failures:

1. Never reduce or index across batch axis `0`; every row belongs to an independent concurrent request.
2. Never call `.item()`, `.cpu()`, `.tolist()`, `print(tensor)`, or in-place ops (`x.add_(...)`) inside methods.
3. Read configurable strengths through `pllm.cfg("beta")` or a workload method (`beta=strength`), not a bare class-body attribute, so `neutral=Workload(beta=0.0)` disables the effect cleanly.
4. Run `pllm.explain` and `pllm.testing.check` on CPU before starting any GPU engine.

## 3. Wire into the right runtime entry point

Pick the engine handle that matches how the project serves or evaluates requests; both `make_llm` and `make_async_llm` share the same live control methods (`import_state`, `train_batch`, `feed`, `generate_completions`, `generate_feedback`):

| Deployment target | Entry point | Requirements |
| --- | --- | --- |
| Offline eval, batched sweeps | `with pllm.make_llm(config, workload=w) as llm:` | `llm.generate(..., batch=StateImport/HostBatch)`, `llm.generate_completions`, `llm.generate_feedback`, `llm.import_state`, `llm.train_batch`, `llm.feed`. |
| Custom async Python app (FastAPI, WebSocket, SSE streaming, live control plane) | `async with pllm.make_async_llm(config, workload=w) as llm:` | Put `w` in an importable `.py` module (`multiprocessing="auto"` enables worker isolation automatically). Use `llm.stream(..., route=..., state_import=...)`, `llm.generate(...)`, `llm.abort(...)`, plus live `await llm.import_state(...)`, `await llm.train_batch(...)`, `await llm.feed(...)`. |
| Drop-in OpenAI HTTP server (`openai.OpenAI`, LiteLLM, HTTP services) | `pllm serve --model ... --workload pkg.mod:WorkloadCls` | Pass multi-turn session keys via `extra_body={"vllm_xargs": {"pllm_route": "session-42"}}`. |

Multi-turn stateful online workload verified on CPU (`key_fields=("route",)` persists and isolates state per
session ID; `reduction="sum"` keeps per-token gradient scale invariant when a session has multiple active rows):

```python
@dataclass(frozen=True)
class SessionDistiller(pllm.Workload):
    beta: float = 0.5
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
        key_fields=("route",),
    )

    @pllm.train(state="distiller", reduction="sum")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


async_cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=16)
check_res = testing.check(SessionDistiller(), async_cfg, neutral=SessionDistiller(beta=0.0))
assert "neutral" in check_res.passed
```

Remember to size `max_num_seqs` for the total number of dormant + active `route` keys when using
`key_fields=("route",)` online, since route-keyed states stay resident across turns until the engine closes.
