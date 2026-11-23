# Where Results Go: Export, Logits, Sampling Guidance, and State Writes

In pLLM, a method reads input tensors and runs tensor math. Under continuous batching and CUDA graph replay, however, the framework cannot know where to send a computed tensor, how long to keep it, or when to publish it to subsequent steps or external processes unless you declare that destination explicitly.

Alongside gradient-based online training via optimizers (see [Online Training](training.md)), pLLM provides four effect decorators for non-training outputs, each targeting a distinct consumer:

1. **`@pllm.export`**: Bounded, asynchronous device-to-host export to CPU for monitoring, feature logging, and analysis;
2. **`@pllm.add_logits`**: Injects per-token biases into model logits before sampler filtering (e.g., DoLa contrastive decoding);
3. **`@pllm.guide_sampling`**: Produces hidden-state predictions projected through the LM head to reweight sampling distributions after top-k / top-p filtering (e.g., ESamp online distillation);
4. **`@pllm.write`**: Directly computes new state field values with custom tensor math and atomically publishes the next state version without optimizers or backpropagation.

This guide examines each effect's semantics, shape contracts, runtime cost, and constraints.

```python
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
```

---

## 1. Exporting Observations: `@pllm.export` and `on_export`

### Concept and definition

`@pllm.export` performs bounded, asynchronous copies of tensors computed during the forward pass or side tasks to the host CPU callback `on_export`, enabling feature logging, probe analysis, or SAE activation monitoring without blocking or slowing down GPU inference.

### Minimal runnable example

In this example, we read the residual stream after layer 12, compute the L2 norm of each request's current token, and export it to the host:

```python
seen: dict[int, list[float]] = defaultdict(list)


class StreamNorm(pllm.Workload):
    # Declare export channel: at most 8 rows per step, capped at 1 column per row
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)

    # Asynchronous host consumer callback: runs in the engine process
    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().tolist()):
            seen[identity.prompt_index].append(row[0])


print(testing.check(StreamNorm(), CONFIG).passed)
```

When running on GPU, open `make_llm` with a context manager:

```python
# check-docs: skip: requires CUDA engine and GPU
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)
GREEDY = SamplingParams(max_tokens=8, temperature=0.0)

with pllm.make_llm(GPU_CONFIG, workload=StreamNorm()) as llm:
    llm.generate(["The capital of France is", "One plus one is"], GREEDY)
    assert llm.diagnostics().eager_reason == ""  # Reading the residual stream preserves CUDA graph capture
```

### Shape contract and event structure

- **Input/output shape**: The method returns a `[rows, cols]` tensor, where dimension 0 indexes request rows under continuous batching.
- **Truncation and cadence**: Each step exports at most `max_rows` rows and at most the first `max_cols` columns, executing once every `every` steps.
- **Drop counter `event.dropped_rows`**: If active rows in a step exceed `max_rows`, excess rows are truncated and the dropped count is recorded in `event.dropped_rows`.
- **Row identities `event.identities`**: A `tuple[RequestIdentity, ...]` aligned one-to-one with the rows of the event tensor:
  - `prompt_index`: Index of the prompt within the `generate` call (or the explicit index passed via `generate(..., prompt_indices=...)`);
  - `sample_index`: Relative sample index under that prompt (`0` for the first sample);
  - `request_id`: pLLM's internal request identifier (distinct from vLLM's `RequestOutput.request_id`).
- **Token positions `event.positions`**: Integer tuple giving each row's token offset within its request.
- **CPU tensor `event.tensor()`**: Returns an independent CPU tensor copy.
- **All-token reads `tokens="all"`**: By default, methods read the latest row per request; to observe every fresh token computed in a step (including all prefill-chunk tokens), pass `tokens="all"` on the input marker (e.g., `pllm.residual(12, tokens="all")`). All-token reads are available only to `@pllm.export` methods.

To export full-width hidden representations (width 896 on Qwen2.5-0.5B) for the last token of each prompt as prefill completes, read `tokens="prefill"`:

```python
prompt_ends: dict[int, torch.Tensor] = {}


class PromptEnd(pllm.Workload):
    @pllm.export(name="prompt_end", max_rows=4, max_cols=896)
    def last_token(self, r: torch.Tensor = pllm.residual(12, tokens="prefill")) -> torch.Tensor:
        return r.float()  # [rows, 896]: last row of each prompt that just finished prefill

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        assert event.dropped_rows == 0
        for identity, row in zip(event.identities, event.tensor()):
            prompt_ends[identity.prompt_index] = row


print(testing.check(PromptEnd(), CONFIG).passed)
```

### Cost and overhead

- **Device-to-host (D2H) transfer**: Issues a non-blocking asynchronous GPU-to-pinned-CPU copy of $\text{rows} \times \text{cols} \times \text{element\_size}$ bytes per step (upper-bounded by $\text{max\_rows} \times \text{max\_cols} \times 8$ bytes). Measured on RTX 4090 with `Qwen/Qwen2.5-7B-Instruct` (0 dropped rows, reconstructing independent CPU tensors on the background daemon thread): streaming full 3584-dim residuals for a single request (7 KiB/step) yields a pure decode TPOT of **16.47 ms** vs. **16.50 ms** on native vLLM (**100%**), and exporting full residuals across 128 concurrent sequences (896 KiB/step) retains **99.5%** of native throughput (TPOT **22.06 ms** vs. native **21.97 ms**).
- **Static cost reporting**: `pllm.explain` prints the per-step row limit, max bytes transferred, and event split count for every export channel.

### Constraints and rationale

1. **Whole-row packing and 32 KiB per-event limit**:
   - **Rule**: Each export event contains whole rows up to 32 KiB (32,768 bytes), and a single row must fit in one event—enforced as `max_cols * 8 <= 32768` (at most 4,096 columns).
   - **Why**: Export events live in fixed-size ring buffers (256 events, up to 32 KiB each). A single row is never split across two events, so one row must fit; when the rows exported in one step exceed 32 KiB in total, they are automatically split into multiple events.
2. **256-event queue capacity and `overflow="drop"` (default)**:
   - **Rule**: The background queue holds up to 256 pending events. By default (`overflow="drop"`), a full queue drops the oldest events and increments `export_dropped`; generation blocks to wait for queue space only when `overflow="block"` is set explicitly.
   - **Why**: **Observation must never stall inference.** If `on_export` performs slow disk I/O, network calls, or logging and falls behind generation, blocking by default would stall the entire serving engine. Throughput is protected by default; use `overflow="block"` only in offline evaluations where sample completeness matters more than throughput.
3. **`on_export` execution context and no re-entrancy**:
   - **Rule**: `on_export` runs asynchronously on a dedicated daemon thread inside the engine process. **Never call engine methods** (such as `llm.generate`) from inside `on_export`.
   - **Why**: The engine core holds scheduling locks and pipeline state; calling back into the engine from the asynchronous export thread causes deadlocks or state corruption.
4. **Automatic drain on context exit**:
   - **Rule**: Exiting `with pllm.make_llm(...)` or calling `llm.close()` automatically calls `drain()`, waiting until every queued export event has been processed by `on_export`. Your collected data is complete as soon as the context exits, with no tail loss.

### Runtime trace selector (`select_trace`)

To capture tensors, gradients, or kernel timings temporarily at runtime without editing the workload class, use the host-side trace selector:

```python
# check-docs: skip: requires running CUDA engine llm, prompts and params
from pllm.io import TraceSelection

# 1. Enable selector: temporarily capture the tensor named "loss"
llm.select_trace(TraceSelection(names=("loss",), tensors=True, max_tensor_bytes=32768))

# 2. Cursor recipe: record the absolute event index before generation
before = llm.trace_snapshot()
cursor = before.dropped + len(before.events)

# 3. Run inference
llm.generate(prompts, params)

# 4. Extract fresh events belonging exclusively to this call
after = llm.trace_snapshot()
fresh = after.events[max(0, cursor - after.dropped) :]

# 5. Disable observation with an empty selection
llm.select_trace(TraceSelection())
```

- **How the cursor recipe works**: The trace ring persists across `generate` calls rather than clearing automatically. If the queue overflows, the oldest events are dropped and added to `dropped`. Computing `cursor = before.dropped + len(before.events)` yields a monotonic absolute index, and subtracting `after.dropped` isolates only the fresh events produced during that `generate` call.
- **Calling discipline**: Call `llm.select_trace()` and `llm.trace_snapshot()` **serially between `generate` calls**, never concurrently from another thread while `generate` is running.

---

## 2. Adding Vocabulary Biases: `@pllm.add_logits`

### Concept and definition

`@pllm.add_logits` adds a user-computed tensor element-wise to the model's unnormalized logits before the sampler applies temperature, top-k, and top-p filtering—ideal for training-free decoding interventions such as lexical suppression, DoLa contrastive decoding, or prompt-level biases.

### Minimal runnable example

This example injects an externally imported global bias vector scaled by `strength` across the vocabulary:

```python
@dataclass(frozen=True)
class LogitBias(pllm.Workload):
    strength: float = 1.0
    bias = pllm.InstanceState({"b": lambda m: (m.vocab_size,)}, init=pllm.Imported())

    def factor(self) -> float:
        return self.strength

    @pllm.add_logits()
    def add(
        self,
        sample: torch.Tensor = pllm.sample_index(),
        b: torch.Tensor = pllm.parameter("bias", "b"),
    ) -> torch.Tensor:
        # [rows, vocab]: broadcast bias across rows, returning exact zeros when neutral=0
        return self.factor() * b.float().expand(sample.shape[0], -1)


print(testing.check(LogitBias(), CONFIG, neutral=LogitBias(strength=0.0)).passed)
```

In DoLa contrastive decoding, the algorithm compares predictions from a mature (final) layer against a candidate premature layer to suppress surface-level premature patterns and amplify deeper factual predictions:

```python
@dataclass(frozen=True)
class DoLaContrast(pllm.Workload):
    candidate: int = 12

    def lens(self) -> tuple[int, ...]:
        return (self.candidate, -1)

    @pllm.add_logits()
    def contrast(
        self,
        logits: torch.Tensor = pllm.layer_logits(lens),
    ) -> torch.Tensor:
        mature = logits[:, -1].log_softmax(-1)
        premature = logits[:, 0].log_softmax(-1)
        # Add the difference onto the final-layer logits (native sampler logits equal logits[:, -1])
        return (mature - premature) - logits[:, -1]


print(testing.check(DoLaContrast(), CONFIG).passed)
```

For multi-concurrency benchmarks of DoLa, see [Example 06: Logit Lens and DoLa Contrastive Decoding](../examples/06_reading_inside.ipynb).

### Shape contract

- **Return shape**: Must return a floating-point tensor of shape `[rows, vocab]` (`float32`, `float16`, or `bfloat16`).
- **Row alignment**: Dimension 0 (`rows`) must equal the number of tokens drawn by the sampler in the current step (one per sampled token).
- **Column alignment**: Dimension 1 must equal the model's full `vocab_size` (e.g., 151,936 for Qwen2.5-0.5B).

### Cost and overhead

- `@pllm.add_logits` additions are fused into the sampler's existing pass over the vocabulary prior to sampling.
- Compute is purely element-wise memory traffic and addition—typically microseconds of latency with negligible impact on forward time.

### Constraints and rationale

1. **Multiple `@pllm.add_logits` methods are allowed**:
   - **Rule**: A workload (including subcomponents) may declare multiple `@pllm.add_logits` methods.
   - **Why**: Tensor addition is associative and commutative. The framework adds all `@pllm.add_logits` biases onto the model logits in declaration order before sampling, with no conflict between biases.
2. **Read logits with `pllm.layer_logits((-1,))`, not `pllm.base_logits()`**:
   - **Rule**: If an `@pllm.add_logits` method needs the model's unscaled logits, it must read `pllm.layer_logits((-1,))`; reading `pllm.base_logits()` is rejected at graph build time.
   - **Why**: On the engine's step timeline, `pllm.base_logits()` is defined as the unmodified base logits consumed by the sampler and is only materialized after sampling (`AFTER_SAMPLE` or inside `@pllm.train`). `@pllm.add_logits` runs before sampling (`BEFORE_SAMPLE`) to shape the sampler's input, before `base_logits()` exists. Use `pllm.layer_logits((-1,))` to read final-layer logits prior to sampling.
3. **Neutral baseline must return exact zeros**:
   - **Rule**: When verifying neutrality with `testing.check(..., neutral=...)`, passing the neutral instance (e.g., `strength=0.0`) must cause the method to return exact zeros everywhere.
   - **Why**: The neutral contract verifies that disabling the algorithm yields the exact unmodified base-model distribution; even tiny non-zero residuals or NaNs shift probabilities.
4. **Return width is checked at build time**:
   - **Rule**: Returning hidden width (e.g., `[rows, 896]`) instead of vocabulary width (`[rows, 151936]`) is caught during shape tracing in `pllm.explain` and `make_llm`, with an error naming the expected shape and fix.

---

## 3. Sampling Guidance: `@pllm.guide_sampling`

### Concept and definition

`@pllm.guide_sampling` takes a hidden-width state prediction from your method, projects it to vocabulary space through the model's native LM head, and contrastively reweights the base logits over the surviving candidate tokens after top-k / top-p truncation—steering generation away from token patterns already predicted by that vector.

### Minimal runnable example and the ESamp pattern

The canonical online sampling-guidance algorithm is ESamp ([Tutorial 03: Decode-Time Training](../tutorials/03_decode_training.ipynb)): each request owns a private `distiller` network trained during decode to predict deep hidden states from shallow hidden states, and guidance activates once the network has completed at least one training step:

```python
@dataclass(frozen=True)
class ShallowPredictor(pllm.Workload):
    beta: float = 0.5

    # What to store: each request owns a private distiller and an independent AdamW optimizer
    distiller = pllm.RequestState(
        init=pllm.FromModule(lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))),
        optimizer=pllm.AdamW(1e-3),
    )

    # Selector: activate guidance only after at least 1 weight update
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    # Config passing: return the instance field via a method
    def strength(self) -> float:
        return self.beta

    # Where results go: sampling guidance (with strength and row selector)
    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        distiller: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return distiller(shallow)  # Returns hidden-width prediction vector [rows, hidden]

    # Online training: predict deep from shallow with MSE loss
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        distiller: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (distiller(shallow) - deep.float()).square().mean(dim=-1)


print(testing.check(ShallowPredictor(), CONFIG, neutral=ShallowPredictor(beta=0.0)).passed)
```

### Shape contract and guidance formula

- **Return shape**: Must return `[rows, hidden]`, where column count matches the model's `hidden_size` (e.g., 896 on Qwen2.5-0.5B).
- **Reweighting formula**: Given method output $\hat{h}$, the framework projects through the LM head to obtain $\text{pred\_logits} = \hat{h} \cdot W_{\text{lm\_head}}^T$. Over the surviving candidate set after top-k and top-p truncation, guided logits are computed as:
  $$\text{logits}_{\text{guided}} = (1 + \beta) \cdot \text{logits} - \beta \cdot \text{pred\_logits}$$
  For $\beta > 0$, tokens easily predicted from shallow features are downweighted, encouraging tokens that carry genuinely deeper information.
- **Semantics of $\beta = 0$**: Declares no sampling guidance on this call, falling back to standard sampling.
- **Row selector `rows=`**: Bind a method returning a `[rows]` boolean mask via `rows="method_name"`. Unselected (`False`) rows keep unmodified sampling. See [Workload Anatomy: Selectors](workload-anatomy.md#selectors-acting-on-a-subset-of-rows).

### Cost and overhead

- **One extra LM-head projection**: Performs one $\hat{h} \cdot W_{\text{lm\_head}}^T$ GEMM per decode step.
- `pllm.explain` reports this projection's FLOPs and memory budget explicitly under the sampler task.

### Constraints and rationale

1. **At most one `@pllm.guide_sampling` across the entire workload**:
   - **Rule**: A workload (including all nested components) can declare at most one `@pllm.guide_sampling` method. Declaring two raises `ValueError` at compile time.
   - **Why**: Guidance reweights the post–top-k/top-p distribution once per step using a single prediction. Whether two predictions should be summed, averaged, or applied sequentially is an algorithmic choice pLLM refuses to guess—combine them into one method, and use `@pllm.add_logits` for any additional vocabulary-space biases.
2. **Return width must equal `hidden_size`**:
   - **Rule**: Return `[rows, hidden]`. This width is not currently validated on CPU and will fail only on GPU, so verify the trailing dimension yourself.

---

## 4. Custom State Writes: `@pllm.write`

### Concept and definition

`@pllm.write` computes new state field values using ordinary PyTorch tensor math and atomically publishes them as a tuple representing the next state version—without losses, autograd, or optimizers.

### Minimal runnable example

This example maintains a per-request running mean of the final hidden state at every decode step:

```python
class RunningMean(pllm.Workload):
    # What to store: each request owns a hidden-width mean vector
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    # Where results go: custom state write updating the mean field of stats
    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        # Return a tuple containing the new tensor version for each field
        return (0.9 * mean + 0.1 * h.float(),)

    # Export the L2 norm of the running mean
    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(
        self,
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)


print(testing.check(RunningMean(), CONFIG).passed)
```

### Shape contract and return values

- **Must return a tuple**: The return value must be a Python `tuple` whose length and order match `fields` strictly. **Even when updating a single field, return a 1-tuple `(new_tensor,)`**—returning a bare tensor raises an error.
- **Leading row axis for `RequestState`**: Each returned tensor for a `RequestState` must carry the leading request row axis `[rows, *field_shape]`.
- **Shape for `InstanceState`**: Each returned tensor for an `InstanceState` matches the field's declared shape (or carries `[rows, *field_shape]` if `per_row=True`).

### Cost and overhead

- Forward-only tensor math: no autograd graph construction, no backward pass, and no optimizer momentum/variance memory.
- Writes directly into private state slots inside the CUDA graph and atomically flips version pointers at step end with zero CPU involvement.

### Constraints and rationale

1. **No `rows=` parameter; whole-batch writes and conditional updates**:
   - **Rule**: `@pllm.write` does not accept `rows=`. On every execution it writes for all active request rows in the batch.
   - **Why**: Each write publishes a complete new version of the state. "Updating only some rows" is equivalent to copying old values back for the remaining rows, which is expressed directly with `torch.where` without needing a separate mechanism.
   - **Conditional update pattern**: Read the old state into the method and select element-wise with `torch.where`:
     ```python
     class ConditionalUpdate(pllm.Workload):
         stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

         @pllm.write(state="stats", fields=("mean",))
         def update_selected(
             self,
             h: torch.Tensor = pllm.final_hidden(),
             mean: torch.Tensor = pllm.parameter("stats", "mean"),
         ) -> tuple[torch.Tensor]:
             # Update only rows whose norm exceeds the threshold; keep others unchanged
             mask = h.float().norm(dim=-1, keepdim=True) > 1.0
             new_mean = 0.9 * mean + 0.1 * h.float()
             return (torch.where(mask, new_mean, mean),)

         @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
         def report(self, mean: torch.Tensor = pllm.parameter("stats", "mean")) -> torch.Tensor:
             return mean.norm(dim=-1, keepdim=True)


     print(testing.check(ConditionalUpdate(), CONFIG).passed)
     ```
2. **Accumulator fields automatically use `float32`**:
   - **Rule**: When a write method reads a field that it also writes (a self-referential accumulator such as a running mean or step counter), `dtype="auto"` automatically promotes that field to `float32`.
   - **Why**: Over dozens or hundreds of autoregressive steps, small incremental updates added to larger values in `bfloat16` (7 mantissa bits) suffer underflow and truncate to zero (swallowed updates), stalling accumulation. `float32` preserves accuracy over long horizons.
3. **Reads see the version published before the current step began**:
   - **Rule**: A method executing on step $n$ always reads the version published at the end of step $n-1$ (or the initial value on step 0).
   - **Why**: pLLM's versioned publishing isolates reads from concurrent writes within a step, eliminating read-after-write hazards so the forward pass, custom writes, and exports can all be captured in CUDA graphs and run safely in parallel across CUDA streams.

---

## 5. Boundary Between Selectors (`rows=`) and Training (`@pllm.train`)

When designing dataflow in a workload, keep the responsibilities of **row selectors** and **training effects** clear:

### Where row selectors (`rows=`) apply

- On `@pllm.add_logits` and `@pllm.guide_sampling`, `rows="selector_name"` restricts the effect to matching rows in the batch; unselected rows retain unmodified logits or standard sampling.
- In standalone per-request training without in-model intervention, training runs unconditionally overlapped with sampling and therefore rejects `rows=`. For complete selector rules and availability, see [Workload Anatomy: Selectors](workload-anatomy.md#selectors-acting-on-a-subset-of-rows).

### Non-training effects vs. `@pllm.train`

| Mechanism | Driven By | Autograd? | Memory & Optimizer | Typical Uses |
| :--- | :--- | :--- | :--- | :--- |
| **`@pllm.export`** | Forward tensor copy | No | 32 KiB host ring buffer only | Probe analysis, SAE monitoring, activation capture |
| **`@pllm.add_logits`** | Pre-sample logit addition | No | No persistent extra VRAM | Vocabulary biases, DoLa contrastive decoding, constrained decoding |
| **`@pllm.guide_sampling`** | Post-filter LM-head reweighting | No | LM-head GEMM workspace | ESamp online distillation guidance |
| **`@pllm.write`** | Custom tensor math | No | State pages only | Running means, state machines, cross-step counters |
| **`@pllm.train`** | Optimizer gradient step | **Yes** | Depends on optimizer (e.g., AdamW moments) | Online fine-tuning, per-request LoRA, TTT adapters |

When your algorithm needs loss backpropagation and parameter updates, see [Online Training](training.md).

---

## 6. Common Errors and Troubleshooting

When patterns on this page are rejected, exact error messages, causes, and fixes are listed in the [Effects and results](../reference/errors.md#effects-and-results) and [Phases: when a value exists](../reference/errors.md#phases-when-a-value-exists) sections of the error catalog.

---

## 7. Self-Check

<details>
<summary>1. Why does <code>@pllm.export</code> default to <code>overflow="drop"</code> rather than <code>"block"</code>, and why cannot a single row exceed 32 KiB?</summary>

Observation must never slow down generation: if `on_export` hits slow disk I/O, blocking would stall the entire inference engine, so the default drops excess events and increments a counter. Export events sit in fixed-size buffers of at most 32 KiB each, and a single row is never split across two events, so one row must fit (`max_cols * 8 <= 32768`).
</details>

<details>
<summary>2. Why does reading <code>pllm.base_logits()</code> inside <code>@pllm.add_logits</code> fail, and how should you read the current step's model output instead?</summary>

`pllm.base_logits()` represents the unmodified base logits consumed by the sampler and is materialized only after sampling (`Phase.AFTER_SAMPLE`) for downstream training or feedback. `@pllm.add_logits` must run before sampling (`BEFORE_SAMPLE`) to supply a logit bias, when `base_logits()` does not yet exist. Read `pllm.layer_logits((-1,))` instead to obtain the current unnormalized final-layer logits.
</details>

<details>
<summary>3. Why does <code>@pllm.write</code> not support <code>rows=</code>, and how do you update state for only a subset of requests?</summary>

Each write publishes one complete new version of the state. To update only selected requests, read the old state into the method and use `torch.where(mask, new_value, old_value)` to compute per-row conditional updates while writing unselected rows back unchanged.
</details>

<details>
<summary>4. Why are multiple <code>@pllm.add_logits</code> methods allowed in one workload, whereas multiple <code>@pllm.guide_sampling</code> methods are disallowed?</summary>

Bias addition is associative and commutative, so outputs from multiple `@pllm.add_logits` methods can be summed onto the model logits in declaration order before entering the sampler. By contrast, `@pllm.guide_sampling` reweights the post–top-k/top-p candidate set once per step using one prediction; how two predictions should be combined is an algorithmic decision that belongs inside a single method.
</details>
