# State

Probe weights, per-request adapters, large tables shared across requests, and running statistics during generation: anything an inference-time algorithm keeps across steps or across requests must be declared as state in pLLM.

**State is the only mutable thing in pLLM.** Whenever a value needs to survive from step to step or across requests, declare it explicitly as state. Once declared, pLLM handles GPU memory allocation, version-safe publishing, CUDA graph capture, and overlapping background training with forward passes.

---

## 1. Why State Is the Only Mutable Thing

Method bodies replay inside CUDA graphs, so Python assignments (`self.mean += 1`) do not execute on each step; input tensors and parameters are shared read-only views that cannot be mutated in-place; and row ordering changes on every step. Therefore, any value that persists across steps must be declared as state and only published as a new version via `@pllm.train` or `@pllm.write`. For the complete set of rules and rationale, see [What Goes in a Method Body](workload-anatomy.md#what-goes-in-a-method-body).

---

## 2. Two Scopes and Routing Keys

When declaring state, the first decision is its **lifetime and sharing scope**. pLLM provides two scopes:

| Scope | Declaration | Semantics & Lifetime | Typical Uses |
| :--- | :--- | :--- | :--- |
| **Request scope** | `pllm.RequestState(...)` | Each request (or group of requests) owns a private copy of the state. Allocated upon request admission and freed when the request finishes. | Per-request LoRA, per-request online distillation networks, per-request running token statistics. |
| **Instance scope** | `pllm.InstanceState(...)` | A single copy of the state shared across the entire inference engine instance. All concurrent requests read and operate on the same physical tensors. | Frozen probe weights, shared embedding/lookup tables, large global tables, engine-wide statistics. |

### Grouping request state with `key_fields`

By default, `RequestState` is strictly isolated per request—each request gets its own private state copy. When a group of requests should share one copy (for example, multiple candidate samples for the same prompt sharing one adapter, or requests from the same tenant sharing trained updates), specify routing keys via `key_fields`:

- `("request_id",)` (default): One isolated state copy per request.
- `("prompt_index",)`: All samples (`sample`) drawn for the same prompt within a single `generate` call share one state copy.
- `("sample_index",)`: Shares state across prompts aligned by relative sample index.
- `("route",)`: Shares state by an external application routing key. The host passes routing keys via `generate(prompts, routes=...)`, and requests with matching route values join the same state copy. Even after a request finishes, that state copy remains retained for the session so subsequent requests with the same route reuse it. For advanced routing patterns and capacity settings, see [Share Training Across Requests](../howto/share-training-across-requests.md).

---

## 3. Declaring Fields: Shapes, `Dimensions`, and `Field`

State consists of one or more named fields, declared either with a single shape or with a field dictionary:

### Single-field and multi-field state

- **Single field**: Pass `shape=(...)` directly; the state gets a single default field named `"value"`:
  ```python
  from dataclasses import dataclass

  import torch
  from torch import nn

  import pllm

  bias = pllm.InstanceState(shape=(896,))
  ```
- **Multiple fields**: Pass `fields={"name": spec, ...}` (or pass the dictionary as the first positional argument) to declare named fields:
  ```python
  linear = pllm.RequestState({
      "weight": lambda m: (m.hidden_size, m.hidden_size),
      "bias": (896,),
  })
  ```

### Forms of field shape specs

Each field's shape spec accepts three forms:
1. **Static fixed tuple**: such as `(896,)` or `(4, 896)`.
2. **Callable taking `pllm.Dimensions`**: such as `lambda m: (m.vocab_size,)`. The argument `m` exposes static model metadata:
   - `m.hidden_size`: hidden dimension (e.g., 896 for Qwen2.5-0.5B);
   - `m.vocab_size`: vocabulary size (e.g., 151,936);
   - `m.layer_count`: number of layers (e.g., 24);
   - `m.target_shapes`: `(in_features, out_features)` of target projections when the state is read by `pllm.model.Adapter`;
   - `m.dtype`: the model's compute dtype.
3. **`pllm.Field`**: Used when a field's shape depends dynamically on each target projection layer (for instance, with `pllm.model.Adapter`):
   ```python
   lora_a = pllm.Field(lambda in_features, out_features: (4, in_features))
   ```
   `pllm.Field` also lets you override the storage dtype per field (such as `pllm.Field((896,), dtype=torch.float32)`).
   Plain shape tuples and `pllm.Field` entries can be mixed freely within the same state (for example, `{"w": pllm.Field(lambda width, _: (width,)), "b": (1,)}`).

### Class attribute vs. typed `@property`

- **Class attribute**: If shapes and config are known when writing the class, declare the state directly as a class attribute.
- **Typed `@property`**: If a state's shape depends on instance parameters (such as a user-supplied `rank` on a `@dataclass(frozen=True)`), declare it as a `@property` **and annotate its return type** (`-> pllm.RequestState` or `-> pllm.InstanceState`):

```python
@dataclass(frozen=True)
class DynamicProbe(pllm.Workload):
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )
```

**Why is an explicit return type annotation required?**
When building the engine and running [`pllm.explain`](explain.md), pLLM inspects class attributes and methods via static reflection. Type annotations let the planner identify which members are state declarations and their scopes without instantiating unknown code or triggering side effects.

---

## 4. Initial Values: `init=`

Every state must have a well-defined initial value at admission (request arrival or engine startup). That initializer is declared via `init=` (each new request's initial state copy is copied from this source):

| `init=` Option | Scope | Initializer Source & Behavior |
| :--- | :--- | :--- |
| `pllm.Zeros()` | Request / Instance | Initializes all fields to zero (the default). |
| `pllm.Computed(function)` | Request / Instance | Calls `function(fields, generator)` once in-place with a seeded generator at engine build time; subsequent admissions copy that initial buffer. |
| `pllm.Pretrained(path, placement="device")` | Request / Instance | Loads tensors matching field names from a `.safetensors` file (whose hash enters workload identity). `placement="host"` pins large read-only tables in host pinned memory; the default `"device"` never silently falls back to host if it does not fit. |
| `pllm.ModelCheckpoint(names)` | `InstanceState` only | Reads extra tensors from the served model checkpoint not used by vLLM's model class (`names` maps `{field: tensor_name}` and may include `{layer}` placeholders); those tensors are excluded from vLLM's own weight load. Read-only, no optimizer. Do not use this to read weights that vLLM itself consumes. |
| `pllm.FromModule(factory, pretrained="")` | Request / Instance | Uses the parameters of the `nn.Module` returned by `factory(dimensions)` as the state's sole field, initialized by the module itself or loaded from `pretrained`. |
| `pllm.Fed(order="ordered" \| "latest")` | `InstanceState` only | Dynamically pushed by an external producer process via `llm.feed(state)`; `"ordered"` consumes every published update strictly in order, while `"latest"` keeps only the newest. At most one per workload. |
| `pllm.Imported()` | Request / Instance | Supplied per inference call via `generate(batch=pllm.io.StateImport(...))` before the request's first forward pass; `RequestState` supplies one row per prompt, while `InstanceState` supplies the whole tensor. |

For loading pretrained weights or external `.safetensors` files and troubleshooting them, see [Load Trained Weights](../howto/load-trained-weights.md).

---

## 5. Frozen State (Read-Only State)

If a state does not need to be updated by `@pllm.train` or `@pllm.write`, omit `optimizer=` (or set `optimizer=None`). This is called **frozen state**.

Frozen state has three key properties:
1. **Zero extra overhead**: No gradient slots or Adam momentum slots are allocated; physical memory equals only the field tensors themselves.
2. **Never modified by an optimizer**: Its value comes solely from its initializer, `pllm.io.StateImport`, or new versions written by `@pllm.write`.
3. **Freely readable anywhere**: Can be read via `pllm.parameter` from any method (including `@pllm.export`, `@pllm.guide_sampling`, `@pllm.custom_residual`, etc.).

---

## 6. Automatic Storage Dtype (`dtype="auto"`) and Explicit Overrides

State parameters default to `dtype="auto"`. At build time, pLLM statically analyzes each field's read/write roles across methods and balances numerical stability against memory cost:

| Role of Field in Methods | Dtype Chosen by `dtype="auto"` | Rationale & Measured Impact |
| :--- | :--- | :--- |
| **Parameters updated by an optimizer** (`@pllm.train`) | `torch.float32` | Prevents small gradients and momentum updates from underflowing in low precision. Long-sequence gradient accumulation in BF16 easily diverges. |
| **Accumulators read and written back** (`@pllm.write`) | `torch.float32` | Cross-step accumulations (such as running means) make small updates each step whose truncation error compounds in low precision. |
| **Cross-chunk / cross-step carry fields** (`carry`) | `torch.float32` | Sequential cross-step recurrence tests show relative error up to ~ $1.4 \times 10^{-1}$ in BF16 vs. ~ $3.7 \times 10^{-4}$ in FP32. |
| **Read-only fields** (frozen probes, steering vectors) or **write-only fields** | Matches model dtype (e.g., `torch.bfloat16`) | Cuts state VRAM and memory bandwidth in half and matches the backbone model's GEMM compute dtype directly. |

### Explicit dtype override

If you need to run FP32-only operators directly on a read-only field, set `dtype=torch.float32` on `InstanceState` / `RequestState`, or override a single field in `pllm.Field`:

```python
state = pllm.InstanceState({
    "w_fp32": pllm.Field(lambda width, _: (width,), dtype=torch.float32),  # Explicit FP32
    "w_model": pllm.Field(lambda width, _: (width,)),                     # Auto-follows model dtype (BF16)
})
```

---

## 7. Reading State and Visible Versions

Inside workload methods, state and state metadata are read via four input markers (declared as parameter defaults):

| Input Marker | Return Type & Shape | Semantics |
| :--- | :--- | :--- |
| `pllm.parameter(state, field, *, per_row=False)` | `torch.Tensor` | Reads a named field of the given state.<br>• **`RequestState`**: Automatically carries a leading row axis `[rows, *S]` aligned to the current op's scheduled rows (each row sees its own request's state).<br>• **`InstanceState`**: Keeps its declared shape `[*S]`; setting `per_row=True` expands a broadcast row axis `[rows, *S]` for batched per-row math. |
| `pllm.module(state)` | `nn.Module` | Reads a module initialized via `init=pllm.FromModule(...)`.<br>For `RequestState`, each request row runs using that request's private parameter copy. |
| `pllm.version(state)` | `torch.Tensor` (`int64 [rows]`) | Returns how many updates have been published for each request's state (0 upon admission). Commonly used to gate on "has trained yet" (e.g., `torch.where(version > 0, ...)`). |
| `pllm.index(state)` | `torch.Tensor` (`int64 [rows]`) | Returns each request's slot index in the underlying state pool, for per-row indexing into external pooled tensors. |

### Which version does a read see?

Whenever any method reads state during a step, it **always sees the version already published before the current step began** (the pre-publish version).
- Even if a side stream is training during the current step and preparing a new version, the main stream's forward pass still reads the version published on the previous step;
- Running requests always observe a consistent snapshot and never read half-updated intermediate data. For the version-publishing timeline and stream overlap model, see [Core Concepts](../explain/core-concepts.md).

---

## 8. Cross-Step Memory: `carry` Fields and `pllm.carry`

When you need to carry non-optimizer continuous memory across decode steps (such as the previous step's prediction error or shallow feature), use **carry fields**:

1. **Declare a carry field**: Configure `carry={"field_name": shape}` on the state. Carry fields allocate no gradient or momentum slots and are not updated by the optimizer.
2. **Write during training**: Wrap an input marker on `@pllm.train` with `pllm.carry(source, state=..., field=...)`. When the training update publishes, the framework automatically writes each request's last row (latest token) of that input into the corresponding carry field.
3. **Read on the next step**: Read the previously published value via `pllm.parameter(state, "field_name")`.

**Why is the carry field always zero on step 0?**
Carry fields are zero-initialized at request admission. Under the rule that reads observe the version published before the current step starts, at the start of step 0 no update has ever been published for that request, so step 0 necessarily sees zeros—a direct consequence of versioned publishing semantics.

---

## 9. `PositionState` and Host Read-Only Tables

Beyond ordinary tensor state, pLLM supports two specialized state layouts:

### `PositionState`

`pllm.PositionState(initial=0.0)` holds one hidden-width vector per KV-cache block, accessible only inside `@pllm.custom_forward` via `layers.load` / `layers.save` (see [Forward Structure](forward-structure.md)).

### Large host read-only tables (`placement="host"`)

When an algorithm uses a massive read-only table (such as a million-entry n-gram prior table or retrieval memory), keeping it in GPU VRAM severely shrinks available KV cache capacity. With `pllm.InstanceState(..., init=pllm.Pretrained(path, placement="host"))`, the table stays in pinned CPU memory:
- **Overlapped side-stream transfer**: Lookup methods that depend only on token IDs are scheduled during `request.prepare` at the start of each step on a dedicated side CUDA stream (`side` lane). They consume only roughly a quarter of SM capacity to asynchronously stream needed rows over PCIe, hiding transfer latency behind the model forward pass.
- **Allowed ops**: To guarantee bounded static transfers, host tables allow only row-gathering operations: slice indexing (`table[idx]`, including sliced views such as `table[:, ::2]` before indexing), `torch.index_select`, `torch.nn.functional.embedding`, and `torch.vmap`. Attempting any other op on a host table is rejected at plan time. See [Performance and Memory](../explain/performance.md) for the bandwidth model.

---

## 10. Memory Reservation and Budget Controls

Although state slots are assigned on demand, their total memory must be reserved before engine startup:

- **Why are more pages reserved than `max_num_seqs`?**
  In [`pllm.explain`](explain.md) output, the page count per field (`x N pages`) is typically noticeably larger than `max_num_seqs`. Under the default cross-step overlap mode (`publish_lag=1`), the side stream writes step $n+1$'s new version while the main stream concurrently reads step $n$'s old version, so multiple versions must coexist in memory simultaneously. Old versions retain only fields that readers read: AdamW step counts and first/second moments are read only by the optimizer, so they are reserved only for the latest and currently-being-written versions.
- **Budget knobs**:
  `ExecutionOptions.state_page_budget_bytes` caps total state pool memory and raises `ValueError` at compile time if exceeded. `ExecutionOptions.execution_peak_budget_bytes` caps total execution peak memory (including state, method workspaces, and CUDA graph reservations), automatically falling back side-stream tasks to inline main-stream execution under memory pressure to avoid OOM. Inspect runtime memory usage and numerical health via `llm.diagnostics().state_usage` and `numeric_health`; see [Configuration](../reference/config.md) and [explain and diagnostics Fields](../reference/diagnostics.md).

---

## 11. Code Examples

All examples below run and pass verification directly on CPU (no GPU required).

### Example 1: Frozen probe with `Computed` initialization (`InstanceState`)

This example declares a frozen linear probe shared across the engine, populates its weights in-place via a seeded generator at build time, and exports per-token scores via `@pllm.export`.

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


def init_probe(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["weight"].normal_(generator=generator)
    fields["bias"].zero_()


class FrozenProbe(pllm.Workload):
    # Instance scope + frozen state: initialized once at startup, never updated by an optimizer
    classifier = pllm.InstanceState(
        {"weight": lambda m: (m.hidden_size,), "bias": (1,)},
        init=pllm.Computed(init_probe),
    )

    @pllm.export(name="probe_score", max_rows=8, max_cols=1)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        w: torch.Tensor = pllm.parameter("classifier", "weight"),
        b: torch.Tensor = pllm.parameter("classifier", "bias"),
    ) -> torch.Tensor:
        # [rows, 896] * [896] -> [rows, 1]
        return torch.sigmoid((h.float() * w).sum(dim=-1, keepdim=True) + b)


print(testing.check(FrozenProbe(), CONFIG).passed)
```

### Example 2: Per-request neural network module (`RequestState`)

This example declares a small neural module dynamically allocated per request so each request owns independent weights and runs its own private module row-by-row.

```python
@dataclass(frozen=True)
class PerRequestHead(pllm.Workload):
    rank: int = 4

    # Shape depends on instance attribute rank, so declare as a typed property
    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        head: nn.Module = pllm.module("head"),
    ) -> torch.Tensor:
        # head automatically binds each request row to that request's private weights
        return head(h.float())


print(testing.check(PerRequestHead(), CONFIG).passed)
```

### Example 3: Updating a per-request running mean with `@pllm.write`

This example updates state with custom tensor math: on each step it reads the previous running mean and current token representation, computes a weighted update, and atomically publishes it via `@pllm.write`.

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        # Accumulator reads and rewrites itself, so dtype="auto" resolves to float32
        return (0.9 * mean + 0.1 * h.float(),)

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(
        self,
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)


print(testing.check(RunningMean(), CONFIG).passed)
```

### Example 4: Cross-step memory with a `carry` field

This example passes information across steps during online training: the training method computes a prediction loss while saving shallow activations into a carry field via `pllm.carry`, then reads that carried vector back on the next step.

```python
@dataclass(frozen=True)
class StepMemory(pllm.Workload):
    beta: float = 0.5

    head = pllm.RequestState(
        {"w": lambda m: (m.hidden_size, m.hidden_size)},
        optimizer=pllm.AdamW(lr=1e-3),
        carry={"prev_act": lambda m: (m.hidden_size,)},
    )

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> torch.Tensor:
        return torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)

    @pllm.train(state="head")
    def loss(
        self,
        # On publish, writes each request's latest shallow row into prev_act
        shallow: torch.Tensor = pllm.carry(pllm.hidden(0), state="head", field="prev_act"),
        deep: torch.Tensor = pllm.hidden(-1),
        w: torch.Tensor = pllm.parameter("head", "w"),
        # Reads the value published on the previous step (all zeros on step 0)
        prev: torch.Tensor = pllm.parameter("head", "prev_act"),
    ) -> torch.Tensor:
        pred = torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)
        return (pred + prev - deep.float()).square().mean(-1)


print(pllm.explain(StepMemory(), CONFIG))
```

---

## 12. Common Errors

When patterns on this page are rejected, exact error messages, causes, and fixes are listed in the [State and training](../reference/errors.md#state-and-training) and [Loading and placing state](../reference/errors.md#loading-and-placing-state) sections of the error catalog.

---

## 13. Self-Check

<details>
<summary>1. Why does <code>dtype="auto"</code> pick <code>float32</code> rather than the more memory-efficient <code>bfloat16</code> for optimizer-updated state or accumulator fields?</summary>

During autoregressive decoding, per-step online training gradients and running averages are typically small. In `bfloat16` (only 7 mantissa bits), tiny gradient steps or weighted increments added to existing weights underflow and round straight to zero (update swallowing), preventing learning or causing divergence. Measurements show carry recurrence has ~ $1.4 \times 10^{-1}$ relative error in `bfloat16` versus ~ $3.7 \times 10^{-4}$ in `float32`. pLLM therefore uses `float32` automatically for fields with small accumulations while keeping read-only and write-only fields in the model dtype (e.g., `bfloat16`) to save memory.
</details>

<details>
<summary>2. Why does reading a <code>carry</code> field on step 0 always return all zeros?</summary>

Because of pLLM's version-publishing semantics: any method reading state during step $n$ always sees the version already published before step $n$ started (i.e., the output published on step $n-1$). At the start of step 0 upon request admission, no training update has run yet, and carry fields are zero-initialized at admission—so step 0 necessarily reads zeros.
</details>

<details>
<summary>3. Why do fields read from <code>RequestState</code> automatically carry a leading row axis (<code>[rows, *S]</code>), whereas fields read from <code>InstanceState</code> do not by default?</summary>

Under continuous batching, multiple concurrent requests are packed into the same batch. `RequestState` keeps a private parameter copy per request, so on every read pLLM gathers and aligns each scheduled row to its own request's slice, yielding `[rows, *S]`. `InstanceState` has a single global copy shared across the entire engine that does not scale with concurrent request count, so it keeps its declared shape `[*S]` by default (pass `per_row=True` if you want a broadcast row axis for batched per-row math).
</details>
