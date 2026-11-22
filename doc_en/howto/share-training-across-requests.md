# Sharing Training Across Requests

By default, `pllm.RequestState` is strictly isolated per request: each request receives a private copy of the state at admission, updates it independently during decoding, and frees its VRAM when the request completes and retires.

In many algorithms, however, state should be **owned and evolved jointly by a group of related requests**:
- **Multiple samples for the same prompt (Best-of-N / Self-Consistency)**: One generation call draws $N$ candidate completions that jointly train the same online LoRA adapter or probe during decoding.
- **Multi-turn sessions or multi-tenant serving across calls**: Requests belonging to the same session, tenant, or task domain (such as math or code) persist and reuse the same adapter across multiple `llm.generate` calls.
- **Engine-wide global sharing**: Every request in the inference engine shares a single global state copy.

pLLM expresses request grouping and state lifetimes through `key_fields` on a state declaration. This guide shows how to configure each sharing granularity, plan capacity, and avoid common pitfalls in group training.

---

## 1. Sharing Across Multiple Samples of the Same Prompt: `key_fields=("prompt_index",)`

### Motivation

In parallel sampling (such as `SamplingParams(n=4)`), a single prompt forks into 4 sibling samples decoded concurrently. With the default `RequestState`, each sample allocates its own isolated adapter copy and learns in a silo, unable to pool insights explored by its siblings.

To let all samples from the same prompt pool what they learn into a single shared adapter, configure them to share one state copy.

### Declaration

When declaring `RequestState`, set `key_fields` explicitly to `("prompt_index",)`:

```python
import torch

import pllm

adapter = pllm.RequestState(
    {"weight": lambda m: (m.hidden_size, m.hidden_size)},
    optimizer=pllm.AdamW(lr=1e-3),
    key_fields=("prompt_index",),
)
```

When executing `llm.generate(prompts, sampling_params)`, pLLM automatically tags every request with its source prompt's `prompt_index`. All sample rows sharing the same `prompt_index` align onto and update the same physical state copy during continuous batching.

### Reduction for Shared-Group Training: Why `reduction="sum"` Is Required

In a `@pllm.train` method, per-request training returns one scalar loss per active row in the current batch, which the framework reduces inside the state:

```python
@pllm.train(state="adapter", reduction="sum")
def learn(
    self,
    h: torch.Tensor = pllm.hidden(0),
    w: torch.Tensor = pllm.parameter("adapter", "weight"),
) -> torch.Tensor:
    # Compute loss for each row in the current batch belonging to this state
    return (torch.bmm(h.unsqueeze(1), w).squeeze(1) - h).square().mean(-1)
```

**Why is `reduction="sum"` strongly recommended over the default `reduction="mean"` for `key_fields` group training?**

- **The pitfall of `reduction="mean"`**: `"mean"` averages the losses of the $n_s$ active rows selected for that state on the current step (dividing by $n_s$). Across a group of concurrent sibling samples, sequences vary in length and hit EOS at different decode steps. With `"mean"`, as early samples finish and exit, the denominator $n_s$ shrinks (for instance, from 4 down to 1). That **artificially amplifies the per-token gradient contribution of the remaining unfinished samples several-fold**, destabilizing the effective learning rate.
- **The guarantee of `reduction="sum"`**: `"sum"` adds the losses of all selected rows on the current step directly. Each surviving token contributes a gradient proportional strictly to its own loss—whether 4 sibling samples are decoding together or only the last sample remains running, every token's gradient weight stays constant.

For mathematical details on online loss reduction and optimizer interactions, see [Online Training](../guides/training.md).

---

## 2. Sharing Across Requests and Calls: Business Route Keys `key_fields=("route",)`

### Motivation

Many online learning algorithms span beyond a single `generate` call: long-lived multi-turn conversations (where the same `session_id` continuously updates conversational memory), lightweight per-tenant experts (where the same `tenant_id` reuses a dedicated LoRA), or domain-routed specialists (where math tasks train a math adapter and code tasks train a code adapter).

Here, grouping is determined by external host business logic, and the state must outlive any single `generate` call.

### Declaration and Invocation

#### 1. Declare the route key on the state

Set `key_fields` to `("route",)`:

```python
adapter = pllm.RequestState(
    {"weight": lambda m: (m.hidden_size, m.hidden_size)},
    optimizer=pllm.AdamW(lr=1e-3),
    key_fields=("route",),
)
```

#### 2. Pass route values from host code

When calling `llm.generate`, pass one route string per prompt via the `routes` argument:

```python
# check-docs: skip: requires a CUDA engine and GPU generate
from vllm import SamplingParams

prompts = [
    "Derive the Fundamental Theorem of Calculus",
    "Write a quicksort algorithm",
    "Prove the Pythagorean theorem",
]


def route_of(prompt: str) -> str:
    # Host business logic computes the route key via classifier, regex, or tenant ID; pLLM never interprets it
    if "theorem" in prompt.lower():
        return "math"
    return "code"


routes = tuple(route_of(p) for p in prompts)
# routes is ("math", "code", "math")

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

with pllm.make_llm(CONFIG, workload=workload) as llm:
    # First call: prompts 0 and 2 share the "math" copy; prompt 1 uses a separate "code" copy
    outputs_1 = llm.generate(prompts, sampling_params, routes=routes)

    # Second call: another request with route "math" reuses the already-trained copy in the VRAM pool
    follow_up_prompts = ["Explain the geometric meaning of the Cauchy-Schwarz inequality"]
    outputs_2 = llm.generate(follow_up_prompts, sampling_params, routes=("math",))
```

### Semantics of Route Values and Isolation

- **Computed entirely by host code**: A route value is a plain Python string that can come from regex rules, an external router model, a tenant ID, or a session ID.
- **pLLM never interprets route contents**: The framework treats the string purely as an opaque key for state addressing.
- **Sharing and isolation**:
  - **Requests with the same route share one state copy**: Whether those requests arrive together in the same batch or sequentially across multiple `generate` calls, they read and update the exact same physical state copy. Parameter updates produced by any request take effect for every request under that route.
  - **Different routes are strictly isolated**: `"math"` and `"code"` maintain completely separate parameter tensors and optimizer slots with zero cross-talk.

---

## 3. Route State Lifecycle and Capacity Planning

### Why Route States Persist After Requests Finish

- **Standard request state** (`key_fields=("request_id",)`): Freed immediately when the request completes and retires, returning its VRAM to the pool.
- **Route-keyed state** (`key_fields=("route",)`): When the last active request for a route retires, its state copy is **not freed**. Instead, it stays alive in the engine's VRAM pool in a **dormant** state, bound to that route key. When a later `generate` call arrives with the same route, the engine wakes that copy and continues training it until the engine shuts down (`llm.close()`).

### Capacity Planning: Leave Headroom in `max_num_seqs`

Even when dormant state copies have no active requests running forward passes, they **still count against the engine's admission capacity**.

At engine build time, `pllm.ModelConfig`'s `max_num_seqs` sets the maximum number of state copies the engine can hold simultaneously. With route keys, steady-state memory usage is governed by:

$$\text{Steady-State Capacity Usage} = (\text{Active + Dormant Route Cardinality}) \times \text{Bytes per State Copy}$$

**You must leave sufficient headroom in `max_num_seqs`**: If you size `max_num_seqs` only for instantaneous peak concurrency, dormant route copies will gradually fill the admission budget as requests with distinct routes arrive and retire. When a new route arrives with no free slots left, the request is rejected at admission. Plan `max_num_seqs` to accommodate the expected total number of distinct routes plus a concurrency buffer.

---

## 4. Global Sharing: When to Use `InstanceState`

If your algorithm's semantics dictate that every request across the entire inference engine shares a single set of physical parameters (with no per-prompt or per-route grouping), do not use `RequestState`.

Use `pllm.InstanceState` directly:

```python
global_table = pllm.InstanceState(
    {"table": lambda m: (10000, m.hidden_size)},
    init=pllm.Zeros(),
)
```

### Comparison of the Four State Sharing Granularities

| Declaration | Sharing Scope | Lifetime | Typical Use Cases |
| :--- | :--- | :--- | :--- |
| `RequestState(key_fields=("request_id",))` (default) | Strictly private to one request | Freed when request finishes and retires | Per-request independent LoRA, per-request online distillation |
| `RequestState(key_fields=("prompt_index",))` | Shared across samples of the same prompt | Freed once all samples of that prompt finish | Best-of-N group exploration, cross-sample collaborative guidance |
| `RequestState(key_fields=("route",))` | Shared across requests with the same route value | Persists dormant after completion until engine shutdown | Multi-turn sessions, per-tenant adapters, continual learning across calls |
| `InstanceState(...)` | Shared across all requests in the engine instance | Initialized at engine creation until engine shutdown | Global frozen probes, large cross-request lookup tables, model-wide statistics |

See [State](../guides/state.md) for full details on state scopes and field declarations.

---

## 5. Complete Runnable Example: Declaration and Cost Check

Below is a complete workload declaration and execution-plan check that runs directly on CPU. It declares an online adapter shared by route key and minimizes a reconstruction loss during decoding with `reduction="sum"`:

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class SharedRouteLoRA(pllm.Workload):
    # What it owns: adapter weights and AdamW optimizer shared by route key
    adapter = pllm.RequestState(
        {"weight": lambda m: (m.hidden_size, m.hidden_size)},
        optimizer=pllm.AdamW(lr=1e-3),
        key_fields=("route",),
    )

    # Where results go: online training during decode; sum reduction keeps per-token weight constant
    @pllm.train(state="adapter", reduction="sum")
    def learn(
        self,
        h: torch.Tensor = pllm.hidden(0),
        w: torch.Tensor = pllm.parameter("adapter", "weight"),
    ) -> torch.Tensor:
        # [rows, 1, H] @ [rows, H, H] -> [rows, H]
        reconstructed = torch.bmm(h.unsqueeze(1), w).squeeze(1)
        return (reconstructed - h).square().mean(-1)


# Plan on CPU without allocating GPU VRAM
workload = SharedRouteLoRA()
plan = pllm.explain(workload, CONFIG)

print(f"CUDA Graph mode: {plan.eager_reason or 'CUDA graphs: no effect needs eager'}")
print(f"State name: {plan.states[0].name}, scope: {plan.states[0].scope}")
```

Using [`pllm.explain`](../guides/explain.md), you can inspect physical state reservations and CUDA graph capture status before launching a live GPU engine.

---

## 6. Verifying That Sharing Works

Once your workload is written, use these two checks to verify that sharing and isolation behave as intended:

### 1. Zero-State Determinism Check

The quickest way to confirm state channels are bound properly is zero-initialized determinism:
1. Send two requests with **new and distinct** routes (e.g., `"route_a"` and `"route_b"`) to the engine simultaneously.
2. Given identical input prompts, because both routes start from zero-initialized parameters, their **step-1 training losses must be identical**.
3. After `"route_a"` trains for several steps, send another `"route_a"` request alongside a brand-new `"route_c"` request. Now `"route_a"`'s step-1 loss should diverge noticeably from its initial value, while `"route_c"` still exhibits the exact zero-state baseline loss. If `"route_a"` and `"route_c"` still match, updates were not published or route keys were not mapped properly.

### 2. Monitor Runtime State via `llm.diagnostics()`

`llm.diagnostics()` reports live internal counters from the engine:
- `trained_rows`: total number of rows that participated in optimizer updates.
- `update_count`: total number of optimizer weight update steps executed.
- `loss_count`: number of loss evaluations performed.
- `active_states`: total number of state copies currently allocated and retained by the engine. Use this to confirm that requests sharing a route reuse a single state copy rather than allocating new ones.

For descriptions of all diagnostic fields, see [explain and diagnostics Fields](../reference/diagnostics.md).

---

## 7. Common Errors and Troubleshooting

When using route keys, most errors stem from mismatches between the state declaration and the arguments passed at call time.

### Why Are Requests Missing a Route Value Rejected Immediately?

When a state declares `key_fields=("route",)`, any request arriving without a route value is **rejected immediately with an error at admission**.

**Why**: The route key is part of the state declaration and solely determines which copy a request uses. Without a route value, the engine cannot tell which copy belongs to the request—silently assigning a default copy would mix unrelated requests into the same training state (for example, letting two tenants share one adapter), so admission rejects it up front.

### Other Errors

For exact error messages, root causes, and fixes when patterns on this page are rejected, see [Requests and run time](../reference/errors.md#requests-and-run-time) and [Method inputs and declarations](../reference/errors.md#method-inputs-and-declarations) in the Error Catalog.

---

## Self-Check

1. After declaring `key_fields=("route",)`, why can't `max_num_seqs` stay small even when instantaneous request concurrency is low?
<details>
<summary>Answer</summary>

Because route-keyed state copies are not freed when their last request retires; they remain alive in a dormant state and continue occupying admission capacity (the <code>max_num_seqs</code> quota). As requests arrive for many distinct routes, dormant copies gradually exhaust the quota and cause requests with new routes to be rejected at admission.
</details>

2. When training online across multiple sibling samples of the same prompt, why is `reduction="sum"` recommended over the default `reduction="mean"`?
<details>
<summary>Answer</summary>

Sibling samples typically finish at different decode steps due to differing output lengths. With <code>"mean"</code>, as some samples finish and exit, the active-row denominator $n_s$ shrinks, passively amplifying the per-token gradient contribution of remaining samples several-fold. With <code>"sum"</code>, each token's gradient weight stays constant regardless of how many sibling rows are currently active.
</details>

3. Can host code pass integers in `routes` (such as `routes=(1, 2, 1)`)?
<details>
<summary>Answer</summary>

No. <code>routes</code> expects a tuple of strings (one non-empty string per prompt). If you are grouping by integer IDs, convert them to strings first, e.g. <code>routes=("tenant_1", "tenant_2", "tenant_1")</code>.
</details>

---

## Next Steps

- Learn how to train state and configure gradient edges in the decode loop → [Online Training](../guides/training.md)
- Dive deeper into state scopes, fields, and initializers → [State](../guides/state.md)
- Inspect memory reservations and graph capture status on CPU before startup → [Inspecting Cost: pllm.explain](../guides/explain.md)
- Troubleshoot additional runtime and admission errors → [Troubleshooting](troubleshooting.md)
