# Editing Models: ActivationPatch, Adapter, and In-Forward Training

When prototyping algorithms in PyTorch or Hugging Face, modifying a model's activations or effective weights usually means registering a `forward_hook` or rewriting `linear.weight` directly. In a high-concurrency inference engine like vLLM, however, tokens from dozens of requests are flattened into a single 2D tensor by continuous batching, and the model's forward pass is captured into CUDA graphs—Python hooks cannot tell which row belongs to which request and do not even fire during CUDA graph replay, while mutating shared weight matrices breaks request isolation and causes memory hazards.

pLLM provides three declarative mechanisms for editing models so each request or instance can customize computation without mutating shared base weights:

1. [`pllm.model.ActivationPatch`](../reference/api.md#pllmmodelactivationpatch): adds activation corrections onto the residual stream after specified layers (e.g., activation steering, LoReFT), running entirely inside the model's CUDA graph.
2. [`pllm.model.Adapter`](../reference/api.md#pllmmodeladapter): applies weight deltas $\Delta W$ onto specified projection matrices (such as attention Q/K/V or MLP projections, e.g., per-request LoRA, dense deltas, per-head deltas), automatically isolated per request and addressed inside CUDA graphs.
3. [`@pllm.train_in_forward`](../reference/api.md#pllmtrain_in_forward): updates private state directly from a projection layer's inputs and targets *while the forward pass is running* (e.g., In-Place TTT fast weights), closing the train-during-forward loop.

---

## `ActivationPatch`: Patching the Residual Stream

**Problem solved**: Applying an additive intervention or low-rank affine transform to the residual stream output immediately after certain layers—for representation steering or representation fine-tuning (LoReFT)—without slowing down inference.

Start with a minimal example: adding an externally supplied steering vector to layer 12's residual stream on every decode step:

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


print(pllm.explain(Steering(), CONFIG).eager_reason)
```

`pllm.explain(Steering(), CONFIG).eager_reason` prints the empty string `""`, meaning `ActivationPatch` stays completely inside the model's CUDA graph and does not fall back to eager mode.

### Constructor Parameters and Operators

The signature of `pllm.model.ActivationPatch` is:

```text
pllm.model.ActivationPatch(
    state: str,
    layers: tuple[int, ...],
    first_tokens: int = 0,
    last_tokens: int = 2,
    operator: Literal["low_rank_affine", "add"] = "low_rank_affine",
    *,
    decode: bool = False,
    refresh: Literal["admission", "step"] = "admission",
    on_missing: Literal["stop", "hold"] = "stop",
)
```

- **`state`**: name of the state holding intervention parameters (supports `InstanceState` or `RequestState`).
- **`layers`**: tuple of layer indices to intervene on (strictly ascending and unique); the intervention site is the residual stream after that layer (matching `hidden_states[layer + 1]` in Hugging Face outputs).
- **`operator`**:
  - `"add"`: pure additive intervention; requires one `(hidden_size,)` vector field per layer in the state and computes $h \leftarrow h + v$.
  - `"low_rank_affine"`: low-rank affine transform (LoReFT style); requires three fields per layer—weight matrix `(rank, hidden_size)`, bias `(rank,)`, and projection/rotation matrix `(hidden_size, rank)`—and computes $h \leftarrow h + R^T (W h + b - R h)$.
- **Token scope**:
  - `first_tokens` and `last_tokens`: apply only to the first $k$ and last $m$ tokens of the prompt (prefill phase).
  - `decode=True`: keyword-only flag specifying whether the patch remains active on every generated token during decode. If `decode=False`, specifying a non-default `refresh` or `on_missing` fails at build time.
- **Dynamic update policy**:
  - `refresh="admission"` (default): a request keeps reading the state version bound when it was admitted for its entire lifetime.
  - `refresh="step"`: switches to the latest published version on every decode step.
  - `on_missing="stop"` (default) vs. `"hold"`: used with external streaming input `pllm.io.Feed`. If no new input arrives on a step, `"hold"` reuses the last successfully applied version, whereas `"stop"` stops the request.

### Forward-Only KV: Past Tokens Are Never Rewritten

`ActivationPatch` follows **Forward-only KV** semantics:
An activation edit injected at layer $L$ affects the current token in all subsequent layers ($> L$) and enters the K/V cache written by later attention layers, influencing future generated tokens; however, it **never retroactively modifies the K/V cache already written by past tokens at layer $L$ or earlier**.

Why not recompute past tokens' KV? Autoregressive decoding's core throughput advantage comes from reusing cached prefix KVs. If every activation tweak required updating past tokens' KV entries, per-step decode cost would degrade from $O(1)$ to $O(T)$ over the entire context, collapsing continuous-batching throughput.

---

## `Adapter`: Weight Corrections on Projections

**Problem solved**: Attaching per-request LoRA or parameter patches to specific linear projections (such as attention Q/K/V or MLP Gate/Up/Down projections) without duplicating the base model, and computing different requests' deltas concurrently within the same batch.

Consider `PromptLoRA`: on admission, each request loads its own rank-4 low-rank factors $A$ and $B$ via `pllm.io.StateImport`, attached to `q_proj` and `v_proj` at layer 12:

```python
@dataclass(frozen=True)
class PromptLoRA(pllm.Workload):
    rank: int = 4

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Imported(),
        )

    @property
    def apply(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12))
        return pllm.model.Adapter("lora", targets, operator="low_rank")


print(pllm.explain(PromptLoRA(), CONFIG).eager_reason)
```

This also prints `""`: an `Adapter` applied to `RequestState` stays completely inside the CUDA graph.

```python
# check-docs: skip: requires loaded model and GPU runtime
from vllm import SamplingParams

with pllm.make_llm(CONFIG, workload=PromptLoRA()) as llm:
    # Prepare LoRA factors externally per prompt: RequestState imports carry a leading prompt axis
    a_q = torch.randn(1, 4, 896) * 0.02  # q_proj: 896 in, 896 out
    b_q = torch.randn(1, 896, 4) * 0.02
    a_v = torch.randn(1, 4, 896) * 0.02  # v_proj: Qwen2.5-0.5B uses GQA, so output is 2 KV heads * 64 = 128
    b_v = torch.randn(1, 128, 4) * 0.02
    # Field order matches the two targets: a_0, b_0, a_1, b_1
    batch = pllm.io.StateImport((a_q, b_q, a_v, b_v))
    outputs = llm.generate(["Hello world!"], SamplingParams(max_tokens=16), batch=batch)
```

### Constructor Parameters and Target Naming

`pllm.model.Adapter` has the following signature:

```text
pllm.model.Adapter(
    state: str,
    targets: tuple[pllm.model.Projection, ...],
    weight: str = "value",
    operator: Literal["dense", "low_rank", "per_head"] = "dense",
    factors: tuple[str, str] = ("a", "b"),
    scale: float = 1.0,
)
```

- **`targets`**: tuple of target projections. Each is declared via `pllm.model.Projection(name, layer=None)` using standard Hugging Face module names (`"q_proj"`, `"k_proj"`, `"v_proj"`, `"gate_proj"`, `"up_proj"`, `"down_proj"`); `layer=None` targets every layer of the model.
- **`operator`**:
  - `"low_rank"`: low-rank product $\text{scale} \cdot (x A^T) B^T$, reading the two fields named by `factors` (default `("a", "b")`).
  - `"dense"`: dense matrix delta $\text{scale} \cdot x W_{\text{delta}}^T$, reading the field named by `weight` (default `"value"`).
  - `"per_head"`: independent `(head_dim, head_dim)` block transform per attention head, reading the `weight` field.
- **Multi-target field expansion rule**:
  When one `Adapter` targets multiple `targets`, each `pllm.Field` shape callback `lambda in_features, out_features: (...)` in the state is invoked in turn with each target projection's input and output widths. Underlying private fields are automatically expanded and named in target order as `a_0`, `b_0`, `a_1`, `b_1`, ...

### Both `RequestState` and `InstanceState` Stay in CUDA Graphs

- **`RequestState`**: each request owns a dedicated state page. Inside the CUDA graph, `Adapter` maps each batch row's request index to its private factor address and computes the delta without re-capturing graphs, **staying inside the CUDA graph**.
- **`InstanceState`**: all requests share one state (or snapshots pinned at request admission version). Inside the CUDA graph, `Adapter` indexes the bound state page via a unified version-slot table, also **staying inside the CUDA graph** without falling back to eager mode.
  (If `cuda_graphs="auto"` is active alongside another non-whole-graph feature such as `@pllm.custom_forward`, both `explain` and `llm.diagnostics().eager_reason` report that cause.)

### Why Trainable Low-Rank States Must Have Non-Zero Initialization

If a state has an optimizer (for example, online LoRA training) and uses `operator="low_rank"`, pLLM forbids the default `init=pllm.Zeros()` and requires explicit non-zero initialization (such as `init=pllm.Computed(...)`):

**Why**: The low-rank output is $\Delta y = \text{scale} \cdot (x A^T) B^T$. The gradient with respect to factor $A$ is proportional to $B^T$, and the gradient with respect to $B$ is proportional to $A^T$. If both factors start at 0, both gradients computed on the first backward pass are identically zero, and the optimizer can never move the parameters. The standard practice in `pllm.Computed` is to initialize $A$ with random Gaussian noise (e.g., std 0.02) and $B$ with zeros: this keeps the initial effective delta at 0 (preserving base model behavior) while giving $B$ non-zero gradients on the very first step.

### Behavior on Quantized Base Models

When using `Adapter` on AWQ, GPTQ, or FP8 quantized models:
- The base linear transformation still runs through its high-performance quantized kernel.
- The `Adapter`'s low-rank or dense delta is computed separately in high precision (FP16/BF16/FP32) and added as a **high-precision residual correction** onto the quantized kernel's output: $y = \text{quant\_linear}(x) + \Delta W \cdot x$.
- The base model's quantized weights are not unpacked. Only when a training method explicitly requests `pllm.operand("weight")` does the engine dequantize and materialize a high-precision dense copy of that projection's base weight, incurring extra VRAM usage.

### Tensor Parallelism (TP) Sharding Rules

Under `tensor_parallel_size > 1` (vLLM 0.32+), `"dense"` and `"low_rank"` operators follow the base projection's exact communication sharding:
- **Column-parallel projections (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`)**: sharded along output features. Dense weights are row-sharded; in low-rank deltas, $B$ is sharded along output features while $A$ is replicated across ranks.
- **Row-parallel projections (`down_proj`)**: sharded along input features. Dense weights are column-sharded; in low-rank deltas, $A$ is sharded along input features while $B$ is replicated across ranks.
- State must be initialized with `pllm.Zeros` or `pllm.Computed` and may only be updated shard-locally via `@pllm.train_in_forward` (training a replicated $A$ from column-sharded outputs or a replicated $B$ from row-sharded inputs across GPUs is disallowed).
- The `"per_head"` operator and states with optimizers or external imports (`Imported` / checkpoints) currently require `tensor_parallel_size=1`.

---

## Decoupled State Producers: Multiple Sources for One `Adapter`

`pllm.model.Adapter` is purely a **consumer declaration**: it specifies only "which state to read and which projections to apply it to," completely agnostic to where the state values come from. This orthogonal decoupling of producer and consumer lets the exact same `Adapter` structure work seamlessly across different algorithmic patterns:

1. **External import (`StateImport`)**: each request writes hypernetwork-generated factors or preloaded LoRA weights at admission (see Tutorial [02 Per-Request LoRA](../tutorials/02_request_adapters.ipynb)).
2. **Post-forward write (`@pllm.write`)**: a shared lightweight network in `InstanceState` reads the final prompt token's representation and writes the request's factors after the forward pass (e.g., Text-to-LoRA).
3. **Online distillation & training (`@pllm.train`)**: updates factors on the side stream using AdamW and downstream hidden states as supervision (e.g., TLM).
4. **In-forward training (`@pllm.train_in_forward`)**: updates factors in place inside a single model forward pass using projection operands.

The consumer side remains identical: regardless of which producer drives the state, as long as the state name and shapes match, `pllm.model.Adapter("lora", targets, operator="low_rank")` applies it automatically.

---

## `@pllm.train_in_forward`: Inline Training During the Forward Pass

**Problem solved**: Online Test-Time Training (e.g., In-Place TTT) requires fast weights to update **inside the forward pass** as tokens stream through, so subsequent tokens or layers immediately execute under the updated weights.

Consider `PromptAdapters`: during prompt prefill, every 16 tokens (one chunk), the model uses the target projection's input to reconstruct the layer's MLP input and updates fast weights via SGD, verified on CPU with the contract checker:

```python
TARGETS = tuple(pllm.model.Projection("down_proj", layer) for layer in (0, 2))


def start_small(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["a"].normal_(std=0.02, generator=generator)
    fields["b"].zero_()


@dataclass(frozen=True)
class PromptAdapters(pllm.Workload):
    rank: int = 4

    @property
    def adapters(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Computed(start_small),
            optimizer=pllm.SGD(1e-3),
        )

    @pllm.train_in_forward(state="adapters", targets=TARGETS, operator="low_rank", chunk=16, flush="prompt_end")
    def learn(
        self,
        x: torch.Tensor = pllm.operand("input"),
        target: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        a: torch.Tensor = pllm.parameter("adapters", "a"),
        b: torch.Tensor = pllm.parameter("adapters", "b"),
    ) -> torch.Tensor:
        prediction = x @ base.T + (x @ a.to(x.dtype).T) @ b.to(x.dtype).T
        return (prediction.float() - target.float()).square().mean(-1)


print(testing.check(PromptAdapters(), CONFIG).passed)
```

`testing.check(PromptAdapters(), CONFIG).passed` simulates row permutation, request isolation, chunked scheduling, and preemption replay on CPU, returning the tuple of verified contracts.

### Operands and Parameter Constraints

Inside a `@pllm.train_in_forward` method, input parameters use marker defaults:
- `pllm.operand("input")`: the input rows currently arriving at the target projection.
- `pllm.operand("target")`: the MLP input activations at that projection's layer (the standard target feature for In-Place TTT).
- `pllm.operand("weight")`: the projection's base weight matrix (materializes a dense copy on quantized models if requested).
- `pllm.operand("prompt_end")`: a 0-d boolean scalar tensor indicating whether the current chunk has reached the end of the prompt.
- `pllm.parameter(state, field)`: references a request-private state buffer (such as $a, b$, or $\Delta W$).

**Mutation rules**: tensors captured via `pllm.operand` are intermediate activations of the base forward pass and are **strictly read-only**; the only buffers permitted to be mutated in place are private parameter buffers referenced via `pllm.parameter`.

### Chunked SGD and Tail Flushing

- **`chunk=`**: sets the token granularity of updates. When the state binds `optimizer=pllm.SGD(lr)`, the method returns a per-row loss tensor of shape `[rows]`, and the framework accumulates gradients until a full chunk of rows completes before stepping SGD.
- **`flush="prompt_end"`**: when a prompt's token length is not divisible by `chunk` (e.g., length 50 with chunk 16), the trailing 2 tokens would otherwise remain buffered. Setting `flush="prompt_end"` flushes an update at the prompt boundary so every prompt token participates in adaptation.
- **`tokens=`**: selects which tokens train the state; defaults to `tokens="prefill"` (train on prompt tokens, freeze during decode), and can also be `"decode"` or `"all"` (see [08 In-Place TTT](../examples/08_inplace_ttt.ipynb)).

### Visibility Policies: `forward_pinned` vs. the Cost of `exact`

- **`visibility="forward_pinned"` (default)**: the entire forward pass reads the state version pinned on entry; updates produced inside the forward are published for subsequent forward steps.
- **`visibility="exact"`**: within the *same* forward pass, the next chunk's projection computation **immediately sees the parameters updated by the preceding chunk**.
- **Cost and trade-off**: `"exact"` breaks full-batch parallelism across chunks, inserting chunk-level synchronization and parameter dependencies inside the forward stream and requiring private intermediate state buffers; unless your algorithm relies on intra-forward chunk recurrence (such as TTT), keep the default `"forward_pinned"`.

### Manual Updates: Returning `UpdateOutput`

If you derive a closed-form update or custom rule instead of using framework SGD, omit the optimizer on the state and return `pllm.UpdateOutput` directly:

```python
targets = (pllm.model.Projection("down_proj", 0),)


class ManualFastWeights(pllm.Workload):
    factors = pllm.RequestState(
        {
            "a": pllm.Field(lambda in_features, out_features: (2, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 2)),
        }
    )

    @pllm.train_in_forward(state="factors", targets=targets, operator="low_rank", factors=("a", "b"), scale=1.0)
    def update(
        self,
        x: torch.Tensor = pllm.operand("input"),
        y: torch.Tensor = pllm.operand("target"),
        a: torch.Tensor = pllm.parameter("factors", "a"),
        b: torch.Tensor = pllm.parameter("factors", "b"),
    ) -> pllm.UpdateOutput:
        residual = (x @ a.to(x.dtype).T) @ b.to(x.dtype).T  # return only the delta added onto base output
        a.add_((0.1 * y[:, : a.shape[0]].T @ x).to(a))      # update private parameters in place
        b.add_((0.1 * (y.T @ y)[:, : b.shape[1]]).to(b))
        return pllm.UpdateOutput(
            residual,
            residual.new_ones((), dtype=torch.bool),              # changed: whether state changed
            residual.new_ones((), dtype=torch.int64),             # update_count: updates applied
            residual.new_full((), x.shape[0], dtype=torch.int64), # trained_rows: rows trained
        )


print(pllm.explain(ManualFastWeights(), CONFIG).eager_reason)
```

`UpdateOutput` holds four fields: `residual` is the delta tensor added to the base projection output, and the remaining three are 0-d device tensors on the same device as the output, ensuring zero GPU-to-CPU synchronization during the forward pass.

---

## Using `Adapter` Inside `@pllm.custom_forward`: Deltas Follow Every Projection Call

When overriding the model's layer loop with [`@pllm.custom_forward`](../tutorials/04_custom_forward.ipynb) (e.g., looped depth, dynamic layer skipping, or layer replay), the standard forward assumption that "every layer runs once per token" no longer holds. An `Adapter` attached to the model follows this core rule:

> **Core rule: weight corrections follow every actual invocation of the target projection, using that invocation's actual inputs and each row's request state.**

1. **Triggered per invocation**: whenever a target projection actually executes—whether in a normal layer pass, repeated iterations in looped depth, or replay—`Adapter` applies a weight delta using that invocation's actual activations. If a slice of layers loops 3 times in looped depth, its target projections receive the delta on all 3 passes, matching the physical semantics of running 3 passes with the adapted weights.
2. **State ownership pinned at forward entry**: which request state each row maps to is anchored on forward entry. Even if `@pllm.custom_forward` compacts rows with `active=` or reorders requests into waves, underlying device indices preserve each row's request ownership so every row addresses its own private state.
3. **Skipped projections get nothing; propagated projections get theirs**: projections that are skipped never run and receive no delta; however, if layer skipping triggers native QKV propagation to populate K/V caches, the Q/V projections that actually run still receive their weight deltas.
4. **Model adaptation decoupled from depth schedule**: the exact same `Adapter`-equipped model can run under the standard forward pass or be reused verbatim across different `@pllm.custom_forward` schedules without rewriting adapter logic.
5. **Limitation**: `Adapter` represents a physical weight change to that projection and has no per-loop-iteration on/off switch. If an algorithm needs a correction "only on the first loop pass and not subsequent passes," that is a hidden-state intervention and should be applied explicitly in Python to the first pass's hidden tensor inside `@pllm.custom_forward`, rather than using `Adapter`.

---

## Common Errors and Runtime Constraints: Preemption, Recomputation, and Prefix Caching

When deploying model-editing algorithms, concurrent scheduling introduces a few boundary constraints.

### Recomputation After Preemption

When vLLM runs out of GPU memory for new KV blocks, the scheduler triggers **preemption**: it evicts low-priority requests' KV caches and later recomputes those requests' histories from the start of their prompts once memory frees up.

- **Workloads that recompute seamlessly**:
  If request state is keyed by `request_id` (`key_fields` includes `request_id`), and the workload has **no mutable instance state** and **no external feeds, imports, or host inputs**, replaying the prompt history reproduces the exact state, so the engine recomputes directly. The `preempt` line of `pllm.explain` states which category your workload falls into.
- **Workloads that fail with a runtime error on preemption**:
  Any overlay or activation-editing workload that depends on external per-step inputs, mutable `InstanceState`, or cross-step state dependencies **cannot guarantee identical state upon recomputation** when preempted; the engine raises a runtime error instead. See [Requests and run time](../reference/errors.md#requests-and-run-time) in the Error Catalog for exact messages and fixes.

> **Historical KV retention**: in causal autoregressive models, already-written KV cache entries reflect the effective weights at the time they were computed; later updates to `Adapter` factors do not retroactively recompute prefix KV entries.

### Why Prefix Caching Is Mutually Exclusive with Overlays and Activation Patches

In vLLM, `enable_prefix_caching=True` shares KV blocks across requests that share identical prompt prefixes.
**In pLLM, overlays (`Adapter`, `train_in_forward`) and `ActivationPatch` are strictly mutually exclusive with Prefix Caching (except immutable `InstanceState` patches with `last_tokens=0`) and are rejected at build time if combined.**

**Why**:
Prefix Caching assumes "identical token sequences produce identical KV tensors." However, per-request LoRA weights, step-updated fast weights, and layer-wise activation steering all change the mathematical computation of the forward pass:
Two requests with the exact same prompt text but different LoRA factors produce completely different KV caches; sharing prefix blocks would feed the second request activations computed under the first request's weights.
Therefore, pLLM enforces `enable_prefix_caching=False` whenever a workload alters model activations or projection weights in a request-dependent or mutable way.

---

## Self-Check

<details>
<summary>1. Why cannot a trainable low-rank LoRA state use the default init=pllm.Zeros()?</summary>

The low-rank delta is $\Delta y = \text{scale} \cdot (x A^T) B^T$. The partial derivative of the loss with respect to $A$ is multiplied by $B^T$, and the derivative with respect to $B$ is multiplied by $A^T$. If both $A$ and $B$ start at zero, both first-step gradients are mathematically zero, so the optimizer can never move the parameters. Non-zero initialization (such as random Gaussian $A$ and zero $B$) is required.
</details>

<details>
<summary>2. Why does an ActivationPatch at layer 12 not alter the layer-12 KV cache of past tokens?</summary>

Autoregressive decoding follows Forward-only KV semantics: past tokens' KV entries were already computed and committed during the forward pass that processed those tokens. Updating past KV caches to reflect new activation edits would require re-running the forward pass over the entire prefix on every generation step, turning $O(1)$ per-step decode into $O(T)$ and destroying concurrent throughput.
</details>

<details>
<summary>3. Does attaching pllm.model.Adapter to an InstanceState force eager mode?</summary>

No. Whether bound to `RequestState` or a shared `InstanceState`, `pllm.model.Adapter` resolves deltas via a graph-capturable overlay table indexed by row or version slot, staying inside vLLM's CUDA graphs. Whole-model CUDA graphs are omitted only when explicitly disabled (`cuda_graphs=False`) or when combined with a sliced-graph feature such as `@pllm.custom_forward`.
</details>
