# Workload Anatomy

pLLM separates **algorithm declarations** from **engine scheduling** (for why it does not use hooks, see [Core Concepts](../explain/core-concepts.md)): you write a `pllm.Workload` subclass whose declarations answer three questions—**what to read**, **what to store**, and **where results go**—and write standard PyTorch tensor math inside its methods. When building the engine, pLLM reads these declarations once, automatically schedules when and on which CUDA stream each method runs, manages per-request state memory, and captures the computation into CUDA graphs.

Start with a minimal runnable workload: it stores no state, reads the residual stream after layer 12, computes the L2 norm of each request's current token, and exports it to the CPU.

```python
from dataclasses import dataclass, field

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)  # Where results go: export to host on_export
    def norm(
        self,
        r: torch.Tensor = pllm.residual(12),                  # What to read: residual stream after layer 12
    ) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)           # [rows, 1]: norm along the feature dim (dim=-1)


print(pllm.explain(StreamNorm(), CONFIG))
```

There are no registration handles or runtime callbacks in this code. If the algorithm also needs each request to maintain its own trainable state (for example, online distillation in ESamp, where each request predicts deep activations from shallow activations as it generates and uses that shallow prediction to guide sampling), simply add a **what to store** declaration to the same class:

```python
@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    # What to store: each request owns an independent small network and AdamW optimizer state
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # What to read: default values declare input sources; where results go: @pllm.train minimizes the per-row loss
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)  # [rows]

    # Method that reads configuration from the instance, referenced by the decorator below
    def strength(self) -> float:
        return self.beta

    # Where results go: guide sampling using the predicted hidden vector at strength beta
    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())  # [rows, hidden]
```

Below we break down the structure of a workload class, the three declaration categories, rules for method bodies, and common configuration and composition patterns. If this is your first time running an engine with a workload, you can also follow along with [Tutorial 00: Quickstart](../tutorials/00_quickstart.ipynb).

---

## The Workload Class and Three Declaration Categories

A workload is a subclass of `pllm.Workload`. Whenever a workload carries hyperparameters or switches (such as `beta: float = 0.5` above), define it as `@dataclass(frozen=True)`.

Why freeze it (`frozen=True`)? Because pLLM reads the configuration on the workload instance once when building the engine and generates a static execution plan; mutating attributes on the Python instance after the engine starts will not alter the execution graph already compiled on the GPU. `make_llm` deep-copies the workload; mutating module globals or objects captured by methods afterward produces undefined behavior (as is always true under `torch.compile`). Using a frozen dataclass prevents accidental runtime mutation and lets you construct a zero-strength baseline with `Distill(beta=0.0)` or `dataclasses.replace(workload, beta=0.0)` for `pllm.testing.check` to verify that disabling the algorithm matches the base model strictly.

Every declaration in a workload falls into one of three categories:

| Question Answered | Syntax | Public Names | Detailed Guides |
| --- | --- | --- | --- |
| **What to read** (method data sources) | Default values on method parameters | Model activations & logits: `pllm.hidden`, `pllm.residual`, `pllm.final_hidden`, `pllm.layer_logits`, `pllm.attention`<br>State & metadata: `pllm.parameter`, `pllm.module`, `pllm.version`, `pllm.index`, `pllm.carry`<br>Tokens & sampling feedback: `pllm.input_tokens`, `pllm.sample_index`, `pllm.base_logits`, `pllm.sampled_tokens`<br>Outputs from other methods in the same step: `pllm.result`<br>In-layer site & projection operands: `pllm.depth`, `pllm.value`, `pllm.operand`<br>Host training batches & differentiable model: `pllm.batch`, `pllm.causal_model` | [Reading Model Internals](reading.md)<br>[State](state.md) |
| **What to store** (tensors held across steps or requests) | Class attributes, or `@property` with a return type annotation | State scopes: `pllm.RequestState` (one private copy per request), `pllm.InstanceState` (one shared copy across the engine), `pllm.PositionState`<br>Field specs: `pllm.Field`, `pllm.Dimensions`<br>Initializers `init=`: `pllm.Zeros` (default), `pllm.Computed`, `pllm.Pretrained`, `pllm.ModelCheckpoint`, `pllm.FromModule`, `pllm.Fed`, `pllm.Imported`<br>Optimizers `optimizer=`: `pllm.AdamW`, `pllm.SGD` | [State](state.md)<br>[Online Training](training.md) |
| **Where results go** (how return values or state take effect) | Method decorators, or model-edit class attributes | Observation & sampling: `@pllm.export`, `@pllm.add_logits`, `@pllm.guide_sampling`<br>State updates & training: `@pllm.write`, `@pllm.train`, `@pllm.differentiate`<br>In-model edits: `pllm.model.ActivationPatch`, `pllm.model.Adapter`, `@pllm.train_in_forward`<br>Forward structure rewrites: `@pllm.custom_residual`, `@pllm.custom_attention`, `@pllm.custom_forward` | [Where Results Go](results.md)<br>[Model Edits](model-edits.md)<br>[Forward Structure](forward-structure.md)<br>[Host Loops and Batch Training](host-loops.md) |

See the [API Reference](../reference/api.md) for full signatures and descriptions of all public interfaces. Three basic rules apply when writing these declarations:

1. **Every input parameter except `self` must have an input marker as its default value.** Python parameter names (like `shallow`, `deep`) are just local variable names inside the method body—pLLM never guesses which layer you want from a parameter name. The default value (such as `= pllm.hidden(0)`) is the declaration read by the planner, and the type annotation (`: torch.Tensor`) tells your IDE and type checker what type the method receives. (The only exceptions are the stream tensors `x, y` in `@pllm.custom_residual`, `q, kv` in `@pllm.custom_attention`, and `layers` in `@pllm.custom_forward`, which are passed as leading positional arguments while any trailing parameters still use input-marker defaults.)
2. **Methods that produce effects must carry an effect decorator.** A plain method that merely returns a tensor without `@pllm.export` / `@pllm.train` or another effect decorator gives pLLM no target (sampler, export queue, or optimizer), so it will not be scheduled on its own unless another method references it via `pllm.result("method_name")` or `rows="method_name"`, or it is explicitly decorated with `@pllm.on`. If the workload declares no effects at all, engine build fails with an error.
3. **At most one pLLM effect decorator of the same kind per method.** Do not stack `@pllm.export` and `@pllm.train` on a single method—if allowed, the outer decorator would overwrite the inner decorator's effect metadata and silently drop one effect. pLLM rejects stacked effect decorators during declaration checking; when you need two effects from the same computation, split them into two methods and reuse the output with `pllm.result("method_name")`.

---

## What Goes in a Method Body

Once inputs and effects are declared, the method body is standard PyTorch tensor math (if you need a custom Triton kernel, call it via `torch.library.custom_op`; see [Custom Operators](custom-ops.md)). Because methods run under continuous batching and CUDA graphs, method bodies must obey four boundaries:

### 1. Dimension 0 is the request row: reduce along features (`dim=-1`), never across rows (`dim=0`)

For activation tensors, request-state parameters, and sampled-row tensors passed into a method, dimension 0 (`[rows, ...]`) always indexes the **request rows in the current step** (one row per active request, or one row per token drawn by the sampler).

- **Why**: Under continuous batching, which requests run together in a step and which row index each request occupies are decided dynamically by the scheduler as requests arrive, finish, or get preempted. If you write `h - h.mean(dim=0)`, `softmax(dim=0)`, BatchNorm, or sort across rows, one request's result mixes in activations from unrelated concurrent requests—changing the batch size or row order changes its output.
- **What to do**: Always reduce along feature dimensions (e.g., `.mean(dim=-1, keepdim=True)`, `.norm(dim=-1)`), so each row's output depends only on that row's own inputs and its own request state. `pllm.testing.check` explicitly tests this via row permutation and neighbor-request perturbation (`request isolation`).

### 2. Inputs and state parameters are borrowed read-only views: do not mutate in-place

Never perform in-place writes on method parameters, such as `h.mul_(2)`, `w.add_(1)`, or `torch.matmul(..., out=h)`.

- **Why**: The activation and state tensors passed to your method are read-only views borrowed from the framework. Other methods in the same step may read the same activations, and older pages in the state pool may still be read by concurrent tasks. Mutating them in-place corrupts shared memory and bypasses versioned state publishing.
- **What to do**: Use out-of-place tensor operations that return new tensors (e.g., `h * 2`, `w + 1`); if you genuinely need slice assignment on a local intermediate, `.clone()` it first. Any persistent state update must be published as a new version through `@pllm.write` or `@pllm.train` (see [State](state.md) and [Where Results Go](results.md)).

### 3. No GPU→CPU synchronization and no Python branches on tensor values

Do not call `.item()`, `.tolist()`, `.cpu()`, or `print(tensor)` inside a method body, and do not write data-dependent Python control flow such as `if h.norm() > 1:` or `while not converged:`.

- **Why**: For two reasons. First, during engine build (and `pllm.explain`), pLLM traces every method with PyTorch `meta` tensors (carrying shape and dtype with no backing storage) to infer shapes and memory budgets; calling `.item()` or branching on a `meta` tensor immediately raises `RuntimeError: Tensor.item() cannot be called on meta tensors`. Second, on GPU at runtime, methods are captured into CUDA graphs and often run on a side CUDA stream overlapped with the model forward pass; CUDA graphs record a fixed operator sequence and cannot capture dynamic Python branches, while GPU→CPU syncs stall the host thread and break stream capture.
- **What to do**:
  - Express conditional logic with boolean masks and `torch.where(mask, a, b)`;
  - To inspect tensor values at runtime, export them asynchronously with `@pllm.export` to `on_export`;
  - **Allowed Python loops**: Bounded `for` loops whose trip count comes from a workload config field (such as `for _ in range(self.loops):`) are completely fine. Because `self.loops` is already a fixed Python integer at build time, tracing unrolls the loop into a fixed sequence of operators.

### 4. Watch tensor dtypes and return shapes

- **Dtype**: Model activations (`pllm.hidden`, `pllm.residual`, `pllm.final_hidden`) are captured in the model's compute dtype (typically `bfloat16`), whereas `pllm.layer_logits` is `float32`. Cast with `.float()` before squaring, taking norms, or accumulating to avoid `bfloat16` underflow or precision loss.
- **Return shapes**:
  - Per-request `@pllm.train` methods must return a 1D `[rows]` loss tensor (one scalar per row)—do **not** call `.mean()` at the end to collapse it to a 0D scalar. Different rows in the same batch may belong to different request states (and padding or unselected rows in the capture table must be masked out first); pLLM reduces valid rows per state (`reduction="mean"` by default, or `"sum"`). By contrast, host batch-training methods reading `pllm.batch(...)` take one step over an external batch and return a single scalar loss (see [Online Training](training.md) and [Host Loops and Batch Training](host-loops.md)).
  - `@pllm.add_logits` must return `[rows, vocab_size]`; `@pllm.guide_sampling` must return `[rows, hidden_size]` (projected into vocabulary space by the model's LM head).

---

## Reading Configuration from the Instance

When writing reusable workloads, hyperparameters such as layer index, guidance strength `beta`, or low-rank `rank` are usually dataclass fields. Passing those fields into decorators, input markers, or state declarations has two common pitfalls.

### Why you cannot write `beta=beta` directly in a decorator

Look at this broken pattern:

```text
@dataclass(frozen=True)
class BadDistill(pllm.Workload):
    beta: float = 0.5
    ...
    @pllm.guide_sampling(beta=beta)   # Here beta evaluates to the class-level default 0.5 at definition time
    def predict(self, ...): ...
```

- **Why**: Python evaluates `@pllm.guide_sampling(beta=beta)` while executing the `class` body, before any `self` instance exists. The name `beta` binds the class attribute's default `0.5`. Later, when you instantiate `BadDistill(beta=0.0)` and pass it to `testing.check(..., neutral=BadDistill(beta=0.0))`, the decorator still holds `0.5`, and the neutral check fails with: `[neutral] guide_sampling still declares a sampler stage; set its beta to zero.`.
- **What to do**: Whenever a decorator argument or input marker argument (such as `@pllm.guide_sampling(beta=...)`, `pllm.hidden(layer)`, `pllm.layer_logits(layers)`, `@pllm.export(every=...)`, or `@pllm.train(steps=...)`) depends on an instance field, pass a **method taking the workload instance and returning that config value** (or `lambda self: self.beta`). pLLM calls it once when binding the concrete instance. Writing a typed method (e.g., `def strength(self) -> float:`) also gives you IDE go-to-definition and static type checking.

### Why a `@property` state must have a return type annotation

When a state's shape depends on an instance field (such as `self.rank`), a class attribute cannot see `self`, so you write the state as a `@property`. In that case, you **must** add a return type annotation to the `@property` (such as `-> pllm.RequestState` or `-> pllm.InstanceState`; for model-edit properties, `-> pllm.model.Adapter`, `-> pllm.model.ActivationPatch`, or `-> pllm.PositionState`). If the class defines any ordinary helper `@property` methods too, annotate their return types as well (e.g., `-> int`).

- **Why**: When analyzing a workload class, pLLM must identify which members are state or model-edit declarations and snapshot method/function dependencies without blindly executing arbitrary user getters, and only then evaluate state declarations in order. Inspecting `@property` return type annotations (including stringified annotations under `from __future__ import annotations`) lets the framework recognize declaration properties without running unknown property code.

The example below combines passing `self.layer` to an input marker via a method and declaring a `self.rank`-dependent state via a typed `@property`:

```python
@dataclass(frozen=True)
class ConfigurableProbe(pllm.Workload):
    layer: int = 12
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )

    def source_layer(self) -> int:
        return self.layer

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(
        self,
        r: torch.Tensor = pllm.residual(source_layer),
        head: nn.Module = pllm.module("head"),
    ) -> torch.Tensor:
        return head(r.float())


print(testing.check(ConfigurableProbe(layer=6, rank=4), CONFIG).passed)
```

---

## When Methods Run

You rarely need to specify when a method runs manually, because every effect decorator carries a default phase (`pllm.Phase`) matching its semantics:

| Effect | Default Phase | Window Within a Step |
| --- | --- | --- |
| `@pllm.train_in_forward`, `@pllm.custom_residual`, `@pllm.custom_attention` | `Phase.IN_FORWARD` | During the model forward pass |
| `@pllm.export` | `Phase.AFTER_FORWARD` | After the model forward pass and state publish, before sampling |
| `@pllm.add_logits`, `@pllm.guide_sampling` | `Phase.BEFORE_SAMPLE` | After the model forward pass, before the sampler draws tokens |
| `@pllm.write`, `@pllm.train` | `Phase.AUTOMATIC` | Placed by the planner based on what the method reads: runs as soon as its inputs are ready; per-request training runs after sampling on a side stream, overlapping with the next step's forward pass |

### `pllm.Phase` is a window within a step, not a runtime hook

`pllm.Phase` has six enum values: `BEFORE_FORWARD`, `IN_FORWARD`, `AFTER_FORWARD`, `BEFORE_SAMPLE`, `AFTER_SAMPLE`, `AUTOMATIC`.

They represent **windows** on the engine's single-step timeline, not callbacks invoked at runtime. At engine build time, pLLM places each method into its assigned window and topologically orders methods according to their data dependencies (for example, method B reading method A via `pllm.result("A")`, or reading a published version of state).

This is why **a value can only be read in the phase that produces it or a later phase**:
- Current-step `pllm.hidden`, `pllm.residual`, and `pllm.final_hidden` exist only after the model forward pass and cannot be read in `BEFORE_FORWARD`;
- `pllm.base_logits()` (the unmodified base-model logits) and `pllm.sampled_tokens()` (tokens drawn in the current step) exist only after sampling and can only be read in `@pllm.train` or methods explicitly marked `@pllm.on(pllm.Phase.AFTER_SAMPLE)`. If you need model logits before sampling (for example inside `@pllm.add_logits`), use `pllm.layer_logits((-1,))`.

When you need to move a method to a non-default window, or schedule an undecorated helper method in a specific window (so other methods can read it via `pllm.result`), decorate it with `@pllm.on(phase)` (taking either a `pllm.Phase` value or a config method returning a `pllm.Phase`):

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(self, mean: torch.Tensor = pllm.parameter("stats", "mean")) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        return (0.9 * mean + 0.1 * h.float(),)


print(testing.check(RunningMean(), CONFIG).passed)
```

---

## Selectors: Acting on a Subset of Rows

Returning to the `Distill` workload from the introduction: on decode step 0 of a newly admitted request, its `distiller` network has not trained even once (`version == 0`), so guiding sampling with its random initial weights only injects noise. We want to enable sampling guidance only for requests whose state has been updated at least once.

Because method bodies execute inside CUDA graphs, you cannot write `if version > 0:`, nor can you dynamically drop rows from a tensor (which would change tensor shapes). pLLM solves this with **row selectors (`rows="method_name"`)**: define a method that reads marked inputs and returns a 1D boolean mask `bool [rows]`, and pass its name to the effect decorator's `rows=` argument.

```python
@dataclass(frozen=True)
class GatedDistill(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # Selector: returns a bool [rows] mask selecting only requests updated at least once
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

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

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())
```

The selector method (`trained`) is compiled as a regular tensor task in the graph, keeping the mask entirely on the GPU: in `@pllm.guide_sampling` and `@pllm.add_logits`, rows whose mask is `False` receive zero weight and keep their original logits untouched with no GPU→CPU sync.

### Where `rows=` is supported and where it is rejected

Not all decorators accept `rows=`. The table below summarizes support and rationale by effect:

| Effect | Supports `rows=`? | Behavior and Rationale |
| --- | --- | --- |
| `@pllm.add_logits(rows="...")`<br>`@pllm.guide_sampling(beta=..., rows="...")` | Supported | Unselected rows keep the sampler's original logits unchanged. |
| Per-request `@pllm.train` paired with in-model effects (e.g., `Adapter`, `ActivationPatch`, without `@pllm.guide_sampling`) or paired with `@pllm.add_logits` | Supports `rows=` and `update_rows=` | The method still evaluates on all rows; `rows=` masks which rows contribute to the reduced loss, and `update_rows=` determines which states run an optimizer step and publish a new version. |
| **Standalone** per-request `@pllm.train` (no in-model effects and no `@pllm.add_logits`), or `@pllm.train` paired with **`@pllm.guide_sampling`** | **Unsupported** (rejected at declaration time) | Both paths take the unconditional post-sample update path (running on a side stream overlapped with the next model forward): every step unconditionally advances the optimizer and publishes a new version for each active request's state, with no per-row conditional skipping. |
| Host batch training reading `pllm.batch(...)` | **Unsupported** | Host batch training performs one update over the whole batch; to skip samples, set `weights` to `0` for those rows in `WeightedRowBatch` or `WeightedCEBatch`. |
| `@pllm.write` | **Unsupported** | `@pllm.write` writes all rows of the state on every step; to keep selected rows unchanged, read the old value and return `torch.where(mask, new_val, old_val)`. |

> [!TIP]
> **How do you suppress certain rows in standalone training or training paired with `@pllm.guide_sampling`?**
> 1. If your goal is to prevent untrained state from affecting generation, attach `rows="trained"` to `@pllm.guide_sampling` as in `GatedDistill` above;
> 2. If your goal is to produce zero gradient for certain rows on a given step, set `@pllm.train(state=..., reduction="sum")` and multiply those rows' losses by `0` inside `loss` (do not use the default `reduction="mean"`, because zeroed rows still count toward the denominator and dilute the gradient of other rows sharing the same state).

---

## Composing Multiple Workloads into One

If you already have an online distillation workload `GatedDistill` and an activation-monitoring workload `StreamNorm` and want to run both in the same engine, you do not need to copy-paste methods into a single class—simply declare them as fields on an outer dataclass workload:

```python
@dataclass(frozen=True)
class Combined(pllm.Workload):
    distill: GatedDistill = field(default_factory=GatedDistill)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

Each field acts as an isolated component: states, methods, and export names are prefixed with the field name (shown in `explain` as `distill.loss`, `monitor.norm`), components cannot read each other's state, and genuinely conflicting combinations (such as two components both guiding sampling) are rejected at build time with both sides named. For why composition is designed this way and which combinations are rejected, see [Composing Workloads](composition.md).

---

## From Code to GPU: The Development Loop

The fastest workflow is not to spin up the full GPU engine on every edit, but to spend a few seconds inspecting the execution plan and running contract checks on CPU first before moving to GPU:

1. **Author the workload**: Define it in a script, notebook cell, or `.py` module. When using the multiprocessing engine (`ModelConfig(multiprocessing=True)`), workload fields must be picklable (no locks, open file handles, etc.) so worker subprocesses can reconstruct and plan the workload independently (see [Production Deployment](../howto/production.md)).
2. **Inspect costs on CPU (`pllm.explain`)**: Reads only the model's `config.json` metadata without loading weights or allocating GPU memory, and prints whether the model stays in CUDA graph, when and on which stream each method runs, how many state bytes are reserved, and whether any method contains a GPU→CPU sync. See [Inspecting Costs: pllm.explain](explain.md).
3. **Check contracts on CPU (`pllm.testing.check`)**: Runs a fixed three-request schedule through the real planner and executor on CPU to verify row permutation invariance, request isolation, mixed prefill/decode, chunked prefill, preemption recomputation, version visibility (`publish_lag`), component isolation, and zero-strength equivalence (`neutral=`). See [Checking Correctness](checking.md).
4. **Run on GPU (`pllm.make_llm`)**: Load the model, generate outputs, and inspect runtime metrics with `llm.diagnostics()`.
5. **Run full GPU contract checks (`pllm.testing.check_on_gpu`)**: Verifies zero synchronization across consecutive decode steps, KV-eviction recomputation, and bitwise equivalence of `neutral=` against native vLLM (in-model edits such as `ActivationPatch`, `Adapter`, and `@pllm.custom_residual` are marked `SKIP` on CPU and require this step for GPU verification).

Run steps 2 and 3 on CPU for `GatedDistill` and `Combined`:

```python
print(pllm.explain(GatedDistill(), CONFIG))
print(testing.check(GatedDistill(), CONFIG, neutral=GatedDistill(beta=0.0)))
print(testing.check(Combined(), CONFIG, neutral=Combined(distill=GatedDistill(beta=0.0))).passed)
```

Once CPU checks pass, run steps 4 and 5 on GPU:

```python
# check-docs: skip: model generation and GPU contract check require CUDA
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)

with pllm.make_llm(GPU_CONFIG, workload=GatedDistill()) as llm:
    outputs = llm.generate(["Explain gradient descent:"], SamplingParams(n=4, max_tokens=32, seed=0))
    report = llm.diagnostics()
print(repr(report.eager_reason), report.update_count, report.trained_rows)

print(testing.check_on_gpu(GatedDistill(), GPU_CONFIG, neutral=GatedDistill(beta=0.0)))
```

---

## Common Errors

When patterns on this page are rejected, exact error messages, causes, and fixes are listed in the [Method inputs and declarations](../reference/errors.md#method-inputs-and-declarations), [Method bodies](../reference/errors.md#method-bodies), and [Contract check errors](../reference/errors.md#contract-check-errors) sections of the error catalog.

---

## Self-Check

1. Why does `hidden - hidden.mean(dim=0, keepdim=True)` pass Python syntax checks yet fail `pllm.testing.check` with `mixes rows`?
<details>
<summary>Answer</summary>

Dimension 0 indexes the distinct request rows packed together in the current step (`[rows, hidden]`). Taking a mean over `dim=0` mixes activations across unrelated requests in the same batch, making one request's output depend on whichever other requests happen to be batched alongside it. Reduce along the feature dimension instead: `hidden - hidden.mean(dim=-1, keepdim=True)`.
</details>

2. In a `@dataclass(frozen=True)` workload, why does `@pllm.guide_sampling(beta=beta)` break `neutral=MyWorkload(beta=0.0)`, and how do you fix it?
<details>
<summary>Answer</summary>

When the `class` body executes, no `self` instance exists yet, so `beta=beta` captures the class attribute's default value (e.g., `0.5`) inside the decorator; constructing `MyWorkload(beta=0.0)` later does not change it. Define a method `def strength(self) -> float: return self.beta` and write `@pllm.guide_sampling(beta=strength)` so pLLM reads `self.beta` when binding the instance.
</details>

3. To save a new tensor into each request's own state on every decode step, can a plain method read `w: torch.Tensor = pllm.parameter("state", "w")` and call `w.copy_(new_val)`? Why or why not?
<details>
<summary>Answer</summary>

No. `pllm.parameter` returns a read-only view, and the same state version may be read concurrently by other methods in the same step. Calling `w.copy_()` in-place breaks version isolation and triggers a `mutates borrowed input` failure in `testing.check`. Decorate the method with `@pllm.write(state="state", fields=("w",))` and return `(new_val,)` so the framework atomically publishes the next version of the state.
</details>

---

## Next Steps

- Learn which activations and logits you can read at every layer → [Reading Model Internals](reading.md)
- Declare per-request state, instance-wide state, initializers, and `key_fields` sharing → [State](state.md)
- Export observations, bias logits, guide sampling, and write back state → [Where Results Go](results.md)
- Train state during decoding and control gradient edges and `publish_lag` → [Online Training](training.md)
- Modify layer activations or linear projection weights directly from state → [Model Edits](model-edits.md)
- Rewrite residual connections, attention reads, or looped layer depth → [Forward Structure](forward-structure.md)
