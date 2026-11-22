# Online Serving: AsyncVllmLlm and pllm serve

`pllm.make_llm` and `pllm.make_async_llm` share the same control-plane capabilities—you can combine offline evaluation, batched external tensor imports, concurrent online streaming requests, and live parameter updates. When your service needs to continuously admit arriving concurrent requests, stream tokens as they are generated, immediately cancel requests on client disconnect, or expose an OpenAI `/v1/chat/completions` endpoint directly, use the two online entry points on this page:

- **OpenAI-compatible HTTP server**: CLI `pllm serve` (or `pllm.backends.vllm.serving.serve_openai_llm`)
- **Async Python engine handle**: `pllm.make_async_llm` (returns `AsyncVllmLlm`)

## Prerequisites

1. **Place the workload in an importable Python module** with picklable fields (the same multiprocessing rules apply; see [Going to Production: Multiprocessing, Tensor Parallelism, and Quantized Base Models](production.md#multiprocessing-enginecore-isolation-and-serialization-requirements)).
2. **`multiprocessing="auto"` runs in isolation by default**: `ModelConfig` defaults to `multiprocessing="auto"`, so passing it to `pllm.make_async_llm` or `pllm serve` automatically launches a dedicated EngineCore worker subprocess (no need to pass `multiprocessing=True` manually).
3. **In-flight concurrency and dynamic updates are supported**: `init=pllm.Imported()`, `init=pllm.Fed(...)`, `pllm.causal_model`, and `pllm.batch` host batch training all work directly with concurrent online streams. In-flight requests read the state version active at their admission time and do not conflict with subsequent `import_state()`, `train_batch()`, or `feed()` calls.

---

## Step 1: Write a Workload and Check It on CPU

As an example, consider an online distillation workload isolated by user session (`route`) that evolves continuously across turns. Requests in the same session share the same `distiller` state (hence `reduction="sum"`; see [Sharing Training Across Requests](share-training-across-requests.md#reduction-for-shared-group-training-why-reductionsum-is-required) for why):

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing


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


# Verify CUDA graph status and multi-request contracts on CPU before starting the GPU server
async_cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=16)
plan = pllm.explain(SessionDistiller(), async_cfg)
assert plan.eager_reason == ""
print(testing.check(SessionDistiller(), async_cfg, neutral=SessionDistiller(beta=0.0)))
```

---

## Step 2: Launch an OpenAI-Compatible Server with `pllm serve`

Mount any importable workload class or instance (formatted as `module.path:ClassOrObjectName`) onto an HTTP port:

```bash
pllm serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --workload my_project.workloads:SessionDistiller \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
```

Call it with the standard OpenAI streaming SDK. For workloads declaring `key_fields=("route",)`, pass the session ID in `extra_body` via `vllm_xargs={"pllm_route": ...}`; subsequent turns in the same session automatically resume the state trained in earlier turns:

```python
# check-docs: skip: connects to a running OpenAI server
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Explain paged attention in two sentences."}],
    max_tokens=64,
    stream=True,
    extra_body={"vllm_xargs": {"pllm_route": "session-42"}},
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

> [!IMPORTANT]
> Request states grouped by `"route"` transition to a dormant state after the current turn finishes so they can be reused by the next turn, and they continue to occupy slots in the `max_num_seqs` capacity budget. Before deploying to production, size `max_num_seqs` for the expected cardinality of coexisting sessions—see [Sharing Training Across Requests](share-training-across-requests.md#3-route-state-lifecycle-and-capacity-planning).

If a workload uses only the default `key_fields=("request_id",)` (such as the built-in `pllm.workloads:DoLaWorkload` or `pllm.workloads:ESampWorkload`), clients do not need to pass `pllm_route`, and each request returns its state pages immediately upon completion or disconnect cancellation.

---

## Step 3: Embed `pllm.make_async_llm` in an Async Python Service

When embedding streaming inference into custom FastAPI routes, WebSocket services, or async task orchestrators, use `pllm.make_async_llm` directly:

```python
# check-docs: skip: runs GPU async engine
import asyncio
from vllm import SamplingParams
import pllm
from my_project.workloads import SessionDistiller

config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=16,
    gpu_memory_utilization=0.5,
)


async def main() -> None:
    async with pllm.make_async_llm(config, workload=SessionDistiller()) as llm:
        async def ask(prompt: str, route: str) -> str:
            last = None
            async for chunk in llm.stream(prompt, SamplingParams(max_tokens=32), route=route):
                last = chunk
            return "" if last is None else last.outputs[0].text

        reply_a, reply_b = await asyncio.gather(
            ask("Explain gradient descent:", route="user-a"),
            ask("Explain KV caching:", route="user-b"),
        )
        print(reply_a, reply_b)


asyncio.run(main())
```

When the caller breaks out of `async for ... in llm.stream(...)` early (or when the enclosing coroutine is cancelled because the client disconnected), `stream()` automatically issues `abort(request_id)` to the engine on exit, freeing the request's KV blocks and non-persistent request state. When token-by-token streaming is not needed, call `await llm.generate(prompt, params, route=...)` directly.

---

## Verify Runtime Behavior

After starting the service or inside integration tests, call `await llm.diagnostics()` and check three metrics (see [explain and diagnostics Fields](../reference/diagnostics.md) for field definitions):

1. **`diag.eager_reason == ""`**: confirms the model forward runs inside CUDA graphs without silently falling back to eager mode.
2. **`diag.inference_pins == 0` and `diag.pending_state_releases == 0` (when no requests are in flight)**: confirms completed and cancelled requests have drained and been reclaimed with no state page leaks.
3. **`diag.active_states`**: for `key_fields=("request_id",)` workloads, returns to `0` when idle; for `key_fields=("route",)` workloads, equals the number of distinct routes visited.

---

## Online Mode Support Scope and Boundaries

| Feature / Pattern | Online Serving (`make_async_llm` / `pllm serve`) | Notes |
| --- | --- | --- |
| Hidden / residual / per-layer logits reads, `@pllm.export` | ✓ | Bounded asynchronous export; never blocks the main stream |
| `@pllm.add_logits`, `@pllm.guide_sampling` | ✓ | Applied row-wise to in-flight requests; supports workload-free inference and online callbacks |
| `RequestState` (including `request_id`, `prompt_index`, `route`), `@pllm.write`, `@pllm.train`, `@pllm.train_in_forward` | ✓ | Requests arrive and exit independently; state pages are allocated and reclaimed automatically |
| `ActivationPatch`, `Adapter`, `@pllm.custom_residual`, `@pllm.custom_attention`, `@pllm.custom_forward` | ✓ | Both `RequestState` and shared `InstanceState` stay inside graphs (`@pllm.custom_forward` uses sliced graphs) |
| `init=pllm.Imported()`, `init=pllm.Fed(...)`, `pllm.causal_model`, `pllm.batch` host batches | ✓ (`make_async_llm` control plane) | Supports in-flight `import_state()`, `feed()`, and `train_batch()` (or per-call `state_import` attached to a request); see the [Support Matrix](../reference/support-matrix.md) for model-level boundaries |

For exact error messages, root causes, and fixes around online admission and runtime limits, see [Requests and run time](../reference/errors.md#requests-and-run-time) in the Error Catalog.
