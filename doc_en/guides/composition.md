# Composing Workloads

In inference-time algorithm research, you often develop multiple independent modules side by side: for example, an algorithm that trains a lightweight network online during decode to guide sampling (such as ESamp), alongside a probe that monitors residual norms at a specific layer in the background (such as StreamNorm). To run them inside the same inference engine, you do not need to merge all state declarations and tensor methods into one monolithic class. In pLLM, you simply write an outer `@dataclass(frozen=True)` workload and declare each sub-workload as a field.

At engine build time, pLLM walks all fields and flattens them into a unified execution graph—allocating memory, capturing CUDA graphs, and scheduling them concurrently under continuous batching. Every component keeps its own isolated namespace and state with zero cross-contamination, and any conflicts are caught and named explicitly at build time.

---

## Declaring an Outer Workload: Dataclass Fields

**Motivation**: Combine two independent `pllm.Workload` classes without modifying a single line of either sub-workload.

**Pattern**: Define the outer workload as a `@dataclass(frozen=True)` subclass of `pllm.Workload`, and declare each sub-workload as a field using `field(default_factory=...)`.

Why `default_factory`? Assigning a class instance directly as a dataclass default would cause all outer workload instances to share the exact same sub-workload object; `default_factory=MyWorkload` ensures each outer workload instance receives its own independent sub-workload instance and configuration.

```python
from dataclasses import dataclass, field

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


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
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


@dataclass(frozen=True)
class Both(pllm.Workload):
    esamp: ESampLite = field(default_factory=ESampLite)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

---

## Namespace Isolation and Export Routing

When you compose `ESampLite` and `StreamNorm` into `Both`, the outer workload's field names (`esamp` and `monitor`) become the namespace prefixes for each independent component:

1. **State and method prefixes**:
   In the execution plan and memory report from [`pllm.explain`](explain.md), all states, methods, and task names are automatically prefixed by their field name. For example:
   - `ESampLite`'s state `distiller` becomes `esamp.distiller`;
   - training method `loss` becomes `esamp.loss`, and sampling guidance method `predict` becomes `esamp.predict`;
   - monitoring method `norm` becomes `monitor.norm`.
   Even if two sub-workloads happen to use identical internal names (such as `loss` or `state`), no symbol collision occurs.

2. **Export event routing (`on_export`)**:
   `StreamNorm` declares `@pllm.export(name="stream_norm", ...)`. Inside the engine, the global exported tensor name is qualified as `monitor.stream_norm`. However, when export events arrive from GPU to the CPU host process, pLLM strips the field prefix `monitor.` based on the originating component and delivers the local tag `stream_norm` to `monitor.on_export(event)`.
   **Benefit**: Component authors write `on_export` callbacks without needing to know what field name the component will be mounted under later—existing code works out of the box.

Let's inspect the plan on CPU with `pllm.explain` and verify the scheduling contracts with `pllm.testing.check`:

```python
# Inspect the plan on CPU: methods and states carry esamp.* and monitor.* prefixes
print(pllm.explain(Both(), CONFIG))

# Run scheduling contract checks on CPU (including component independence verification)
print(testing.check(Both(), CONFIG, neutral=Both(esamp=ESampLite(beta=0.0))).passed)
```

---

## Component Independence and State Privacy

In `Both`, can `monitor` read `esamp`'s state—for example, by declaring `f: nn.Module = pllm.module("esamp.distiller")` inside one of `monitor`'s methods?

**No—this is strictly forbidden.** Using any dotted name (`.`) in an input marker or state reference raises an exception at build time:
`components are independent; to share state, put both methods in one workload`.

### Why Enforce State Privacy?

This rule comes directly from pLLM's **component independence guarantee**:
- **State trajectories must match standalone execution**: Any component inside a composed workload must produce the exact bitwise-identical computation and state evolution trajectory (every step's state tensor values, optimizer step counts, and version updates) as when run alone in its own engine.
- **Preventing implicit timing coupling**: Allowing cross-field state reads/writes would introduce implicit execution-order dependencies (who runs first, and which intermediate version gets read), breaking pipelined and CUDA-graph overlapped scheduling and causing hard-to-reproduce numerical drift when standalone-tested components are combined.

### Contract Check: `component independence`

When you run [`pllm.testing.check`](checking.md) on a composed workload, the checker verifies not only standard batching scheduling contracts (row permutation, request isolation, preemption replay, etc.), but also **`component independence`**:
1. Runs the full outer composed workload `Both` on CPU;
2. Runs each sub-component (`ESampLite` and `StreamNorm`) individually;
3. Compares every output, state, and export event of each component across both runs, request by request and token by token; any mismatch raises `ContractError`.

> [!TIP]
> **What if two methods genuinely need to share state?**
> If two methods must read and write the same state (for example, one method trains small weights via `@pllm.train` and another reads those weights to rewrite forward residuals), they are logically two steps of a single algorithm rather than two orthogonal components. Declare them inside the same `pllm.Workload` class instead of splitting them across components. See [Workload Anatomy](workload-anatomy.md) and [State](state.md).

---

## Merging Effects: Multiple `@pllm.add_logits`

How do results from different components combine onto the model?

- **Multiple `@pllm.add_logits` can coexist**:
  If component A and component B both declare `@pllm.add_logits`, they do not conflict. During the sampler epilogue, pLLM accumulates each component's logit delta onto the model's base logits in the declaration order of the outer dataclass fields.
- **Logit additions run before sampling guidance**:
  All `@pllm.add_logits` deltas are accumulated before `@pllm.guide_sampling` runs. Thus, if one component shifts the logits, a downstream sampling guidance method still performs filtering and log-reweighting on top of the updated distribution. See [Where Results Go](results.md).

---

## Composition Conflicts and Build-Time Errors

Not every pair of effects can be stacked unconditionally. When two components make mutually exclusive demands on the same physical resource of the base engine, pLLM **rejects the build immediately at build time (`pllm.explain` or `pllm.make_llm`) and names both conflicting sides**, rather than letting a later declaration silently overwrite an earlier one.

Here is an intentional conflict where two components both declare `@pllm.guide_sampling`:

```python
@dataclass(frozen=True)
class ConflictingBoth(pllm.Workload):
    a: ESampLite = field(default_factory=ESampLite)
    b: ESampLite = field(default_factory=ESampLite)


try:
    pllm.explain(ConflictingBoth(), CONFIG)
except ValueError as e:
    print(f"Caught expected build conflict: {e}")
```

Running this raises `ValueError`:
`a.predict and b.predict both declare @pllm.guide_sampling; keep at most one guidance method`. The message explicitly names both conflicting methods, `a.predict` and `b.predict`.

### Other Composition Conflicts

For error messages covering other composition conflicts, see the [Effects and results](../reference/errors.md#effects-and-results) and [Model sites and custom forwards](../reference/errors.md#model-sites-and-custom-forwards) sections of the Error Catalog; for the full matrix of what can coexist vs. what is rejected and why, see [Composition Rules](../reference/composition-rules.md).

---

## Self-Check

1. A composed workload `Both` contains a monitoring component `monitor: StreamNorm`, whose method declares `@pllm.export(name="stream_norm", ...)`. When consuming events on the host, does `monitor.on_export(event)` receive `event.tensor_name` as `"monitor.stream_norm"` or `"stream_norm"`? Why?

<details>
<summary>Answer</summary>

`"stream_norm"`. Before dispatching a GPU export event into a component's own `on_export` callback, pLLM's export router strips the outer field prefix `monitor.`. This preserves component reusability: sub-components can be written and unit-tested without knowing what field name an outer workload will mount them under.
</details>

2. A researcher wants component A to compute per-row classification features into state `feat`, and component B to read `feat` to produce logit biases. They write A and B as two sub-workloads in an outer dataclass, and hit an error when declaring `pllm.parameter("a.feat", "value")` in B. Why does pLLM disallow this, and what is the right implementation?

<details>
<summary>Answer</summary>

pLLM enforces component independence: each component's state trajectory must match running it standalone. Cross-component state reads/writes would introduce implicit timing dependencies and break the `component independence` contract check.

Right approach: Because `feat` is a tightly shared intermediate state between the two methods, they belong to the same algorithm semantically. Put both the producer method and the consumer method of `feat` inside a single `pllm.Workload` class.
</details>
