# Reading Model Internals

When prototyping inference algorithms in Hugging Face, reading intermediate activations or logits at any layer is as simple as passing `output_hidden_states=True` or registering a PyTorch `forward hook` on `model.layers[i]`. In high-throughput vLLM serving, however, tokens from dozens of concurrent requests are packed into a single tensor by continuous batching, and the model forward pass is captured inside CUDA graphs—plain PyTorch hooks neither know which row belongs to which request nor execute at all during CUDA graph replay.

pLLM lets you declare the internal tensors a workload method needs to read directly in its signature (for example, `r: torch.Tensor = pllm.residual(12)`). Under the hood, pLLM reserves static buffers inside vLLM's CUDA graph and captures the reads inside the graph without breaking graph replay or introducing CPU–GPU synchronization. Dimension 0 of every tensor your method receives is automatically aligned to the active request rows of the current step.

If your algorithm needs to read mutable parameters or module state owned by the workload itself (`pllm.parameter`, `pllm.module`, `pllm.version`, `pllm.index`), those belong to state declarations; see [State](state.md).

---

## Quick Lookup: From Hugging Face to pLLM Input Markers

The table below maps common Hugging Face internal activations and metadata to their corresponding pLLM input markers:

| Hugging Face Concept / Attribute | pLLM Input Marker | Dtype | Shape | Meaning & Typical Use |
| :--- | :--- | :--- | :--- | :--- |
| `hidden_states[layer + 1]` | `pllm.residual(layer)` | Model dtype (`bfloat16`/`float16`) | `[rows, hidden]` | Residual stream after `layer` (accumulating token embeddings and all preceding sublayer outputs); primary input for activation steering, SAEs, and probes |
| Output of `model.layers[layer].mlp` | `pllm.hidden(layer)` | Model dtype | `[rows, hidden]` | Incremental output of `layer`'s MLP before being added back to the residual stream; for observing or distilling that layer's fresh transform |
| Intermediate vector between attention and MLP | `pllm.residual(layer, after="attention")` | Model dtype | `[rows, hidden]` | Residual stream after `layer`'s attention is added back and before entering the MLP |
| `last_hidden_state` (after final norm) | `pllm.final_hidden()` | Model dtype | `[rows, hidden]` | Hidden vector after the final RMSNorm, directly feeding the LM head |
| `model.lm_head(norm(hidden_states[l+1]))` | `pllm.layer_logits(layers)` | `float32` | `[rows, len(layers), vocab]` | Residual stream after each specified layer projected through the final norm and LM head (logit lens); the last layer matches the model's output logits |
| `logits` (model's final output before sampling) | `pllm.layer_logits((-1,))` | `float32` | `[rows, 1, vocab]` | Read the final layer via `layer_logits` when you need the model's output logits before sampling (e.g., to reshape logits based on its own distribution) |
| `logits` (raw output after sampling) | `pllm.base_logits()` | `float32` | `[rows, vocab]` | Unmodified base logits exposed after sampling for training or monitoring (available only after sampling) |
| `input_ids` (current and recent context tokens) | `pllm.input_tokens(history=n)` | `int64` | `[rows, n + 1]` | Current token ID plus the previous `n` token IDs (padded with `-1` before request start); used for lookup memory and n-gram priors |
| Generated token ID | `pllm.sampled_tokens()` | `int64` | `[rows]` | Token ID selected by the sampler in the current step (available only after sampling) |
| Multi-sample index | `pllm.sample_index(limit=None)` | `int64` | `[rows]` | Sample index (`0` to `n - 1`) of each request within its prompt |
| FlashAttention LSE / aux stats | `pllm.attention(layer, "lse" / "aux")` | `float32` / custom | `[rows, heads]` / `[rows, ...]` | Attention Log-Sum-Exp or custom `aux` tensor (paired with `@pllm.custom_attention`) |

---

## A Complete Reading Example

Here is a complete reading workload that simultaneously computes the L2 norm of the layer-12 residual stream and exports the logit-lens top-1 predicted tokens at layers 6, 12, and the final layer.

```python
import torch
import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class Lens(pllm.Workload):
    @pllm.export(name="lens", max_rows=8, max_cols=3)
    def top1(self, logits: torch.Tensor = pllm.layer_logits((6, 12, -1))) -> torch.Tensor:
        return logits.argmax(dim=-1).float()  # [rows, 3]: most likely token at layers 6, 12, and final

    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)  # [rows, 1]: L2 norm of layer-12 residual stream


print(testing.check(Lens(), CONFIG).passed)
```

In this example:
1. `logits: torch.Tensor = pllm.layer_logits((6, 12, -1))` declares reading logits at three layers with shape `[rows, 3, 151936]` and native dtype `float32`.
2. `r: torch.Tensor = pllm.residual(12)` declares reading the residual stream after layer 12 with shape `[rows, 896]` and the model's compute dtype (typically `bfloat16`).
3. Using [`pllm.testing.check`](checking.md), you can verify input shapes, row alignment, and scheduling contracts on CPU without a GPU.

---

## Residual Stream and In-Layer Activations: `residual`, `hidden`, and `final_hidden`

In a Transformer decoder, layer $l$ computes:

$$\begin{aligned}
x'_l &= x_l + \text{Attention}(\text{Norm}(x_l)) \\
x_{l+1} &= x'_l + \text{MLP}(\text{Norm}(x'_l))
\end{aligned}$$

Here $x_l$ is the **residual stream** flowing through the model. Knowing which site you are reading is essential:

- `pllm.residual(layer, *, after="layer", tokens="decode")`:
  Reads the residual stream $x_{l+1}$ after `layer`, equivalent to Hugging Face's `hidden_states[layer + 1]`. It accumulates the token embedding and all sublayer outputs from layer 0 through `layer`. **Activation steering (ActAdd/CAA), sparse autoencoders (SAEs), linear probes, and logit lens all read this tensor**.
  With `after="attention"`, it reads the intermediate stream $x'_l$ after attention is added back and before the MLP.
- `pllm.hidden(layer=0, *, tokens="decode")`:
  Reads the MLP output $\text{MLP}(\text{Norm}(x'_l))$ of `layer` **before the residual addition**. It contains only the fresh increment computed by that layer's MLP, not the accumulated residual stream. Use this when distilling a single layer's fresh transformation online; mistaking it for the residual stream loses all accumulated information from earlier layers.
- `pllm.final_hidden(*, tokens="decode")`:
  Reads the hidden state after the final RMSNorm, equivalent to Hugging Face's `last_hidden_state`—the direct input to the LM head projection.

### Dtypes and When to Call `.float()`

`pllm.hidden`, `pllm.residual`, and `pllm.final_hidden` use the model's compute dtype (usually `torch.bfloat16` or `torch.float16`). Explicitly cast with `.float()` in three situations:

1. **Norms and similarity reductions**: Squaring or summing in half precision for L2 norms (`r.float().norm(dim=-1)`) or dot products easily overflows or loses small increments.
2. **Training losses**: Casting to `float32` when computing MSE or cross-entropy inside [`@pllm.train`](training.md) prevents gradient underflow or NaNs.
3. **Exporting to host**: Cast floating-point metrics to single precision before exporting via [`@pllm.export`](results.md) to preserve numerical accuracy.

---

## Per-Layer Logits and Logit Lens: `layer_logits`

For interpretability and contrastive decoding methods such as DoLa, you often want to ask "what would the model predict if it stopped at layer $l$?" In raw PyTorch, this requires passing intermediate residual streams through the model's final norm and LM head.

`pllm.layer_logits(layers)` packages that operation:

```python
logits = pllm.layer_logits((6, 12, -1))
```

- **Argument**: Takes a tuple of layer indices (negative indices supported; `-1` is the final layer).
- **Shape and dtype**: Returns a `torch.float32` tensor of shape `[rows, len(layers), vocab]`. It is produced natively in `float32` because projecting onto a large vocabulary (151,936 for Qwen2.5) yields large unnormalized magnitudes that can overflow in half precision.
- **Bitwise identical model output**: `layer_logits` runs an independent GEMM for intermediate layers without touching the base model's own LM head path, so the base model's sampled logits remain bitwise identical to native vLLM.
- **Row alignment**: Produces one row per token drawn by the sampler (including the last token of each prompt and every decode token).

For a detailed contrastive decoding walkthrough, see [Example 06: Reading Model Internals and DoLa](../examples/06_reading_inside.ipynb).

---

## Controlling Which Tokens Are Read: `tokens=` Rules and Prefill Capture

Under continuous batching, a single step may contain both chunked prompts undergoing prefill and requests generating decode tokens. The `tokens=` argument controls which tokens dimension 0 of `[rows, hidden]` corresponds to.

`pllm.hidden`, `pllm.residual`, and `pllm.final_hidden` all accept `tokens=`:

- `tokens="decode"` (default):
  **1 row** per request in the decode phase in this step, corresponding to its latest generated decode token. Most token-by-token guidance, probe monitoring, and online training algorithms use this default.
- `tokens="prefill"`:
  **1 row** per request that **just completed full prompt prefill** in this step, corresponding to the last token of its prompt. If a long prompt is split across multiple prefill chunks, it produces no row on intermediate chunks and appears once on the step where prefill completes. Ideal for prompt classification, feature extraction, or decisions based on the prompt-end representation.
- `tokens="all"`:
  **1 row for every freshly computed token** in the current step (including all intermediate tokens of prefill chunks), ordered strictly by token position within each request.
  - **Restriction and rationale**: `tokens="all"` **can only be read in `@pllm.export` methods that take no `pllm.result` inputs**. Methods for online training, logit biasing, and sampling guidance allocate fixed-size buffers and CUDA graphs sized to `max_num_seqs` (such as 8 or 32), whereas the total fresh tokens in a step can reach `max_num_batched_tokens` (such as 4096 or 8192) and cannot fit inside per-request graph buffers. Only `@pllm.export` supports chunking large token batches and streaming them to a CPU queue.
- **One phase per method**:
  **Do not mix inputs with different `tokens=` modes in the same workload method** (e.g., one parameter with `tokens="prefill"` and another with `tokens="decode"`). The set of requests finishing prefill and the set of requests decoding in a step are disjoint or distinct subsets, so their row axes cannot align.

### Example: Capturing Prompt-End Representations

The example below uses `tokens="prefill"` to extract the residual vector at the end of every prompt:

```python
prompt_ends: dict[int, torch.Tensor] = {}


class PromptEnd(pllm.Workload):
    @pllm.export(name="prompt_end", max_rows=4, max_cols=896)
    def last_token(self, r: torch.Tensor = pllm.residual(12, tokens="prefill")) -> torch.Tensor:
        return r.float()  # [rows, 896]: one row per prompt that just finished prefill

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        assert event.dropped_rows == 0
        for identity, row in zip(event.identities, event.tensor()):
            prompt_ends[identity.prompt_index] = row
```

On GPU, submit at most `max_rows` prompts per call so every row is captured with zero drops:

```python
# check-docs: skip: requires a CUDA engine and model weights
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)

texts = ["I love this", "I hate this", "This is wonderful", "This is terrible", "I feel happy", "I feel sad"]
with pllm.make_llm(GPU_CONFIG, workload=PromptEnd()) as llm:
    for start in range(0, len(texts), 4):  # Submit at most max_rows prompts per call
        chunk = texts[start : start + 4]
        # prompt_indices numbers prompts across batches; identity.prompt_index preserves those indices
        llm.generate(chunk, SamplingParams(max_tokens=1), prompt_indices=tuple(range(start, start + len(chunk))))

rows = torch.stack([prompt_ends[index] for index in range(len(texts))])
print(rows.shape)  # torch.Size([6, 896])
```

---

## Context Token History and Sample Index: `input_tokens` and `sample_index`

### `pllm.input_tokens(history=n)`

For context-based n-gram lookup memory (such as Engram) or n-gram language model priors, algorithms need the current token ID and the preceding token IDs.

- **Return value**: `torch.int64` tensor of shape `[rows, history + 1]`. Column 0 (`ids[:, 0]`) is the current token ID, followed by preceding token IDs (`ids[:, 1]` is the previous token, and so on).
- **Boundary padding**: Positions prior to token 0 of a request are padded with `-1`.
- **Row alignment**: When its result feeds into [`@pllm.custom_residual`](forward-structure.md), it covers every scheduled token in the current step; for other methods (such as `@pllm.export` or `@pllm.add_logits`), it aligns with sampled rows.
- **Data source**: Token history comes from per-request inputs already kept on GPU by vLLM (MRv2 runner), so table lookups require no CPU-to-GPU transfers. See [Example 11: Lookup Memory and n-gram Priors](../examples/11_lookup_memory.ipynb).

```python
class TokenWatcher(pllm.Workload):
    @pllm.export(name="tokens", max_rows=8, max_cols=3)
    def recent_tokens(self, ids: torch.Tensor = pllm.input_tokens(history=2)) -> torch.Tensor:
        return ids.float()  # [rows, 3]: current token, previous, two back


print(testing.check(TokenWatcher(), CONFIG).passed)
```

### `pllm.sample_index(limit=None)`

When a client passes `SamplingParams(n=4)` to sample multiple completions per prompt, each branch request receives a distinct sample index.

- **Return value**: `torch.int64` tensor of shape `[rows]` giving each row's sample index within its prompt (`0` to `n - 1`).
- **`limit` argument**: Passing `limit=K` causes `generate` to reject requests with `sample_index >= K`. When your workload only prepared data for up to `K` samples (for example, a K-row lookup table indexed by sample index), use `limit=K` so invalid requests fail at admission rather than reading out of bounds or silently miscomputing.

---

## Post-Sampling Feedback: `base_logits` and `sampled_tokens`

Some algorithms need to inspect the unmodified base-model distribution or the selected token after sampling occurs:

- `pllm.base_logits()`: Returns the model's raw logits before any `@pllm.add_logits` or sampling guidance modifications, with dtype `float32` and shape `[rows, vocab]`.
- `pllm.sampled_tokens()`: Returns the token ID chosen by the sampler in the current step, with dtype `int64` and shape `[rows]`.

### Why they can only be read after sampling

`pllm.base_logits()` and `pllm.sampled_tokens()` **can only be read after sampling (`Phase.AFTER_SAMPLE`)**, such as inside [`@pllm.train`](training.md) or a method decorated with `@pllm.on(pllm.Phase.AFTER_SAMPLE)`.

This follows directly from causality:
1. `sampled_tokens` is the outcome of the sampler's draw and does not exist before sampling runs.
2. `@pllm.add_logits` computes a bias added to logits before sampling; reading `base_logits()` before sampling would create a cyclic dependency in the sampling pipeline. If you need the model's current-step logits before sampling, use `pllm.layer_logits((-1,))`.

---

## Inspecting Attention Internals: `pllm.attention`

To inspect attention distribution properties (such as the softmax log-sum-exp denominator or custom outputs from a custom attention operator), use `pllm.attention`:

- `pllm.attention(layer, "lse")`: Reads the Log-Sum-Exp produced by the first `kv.attend` call at `layer`, with dtype `float32` and shape `[rows, heads]`.
- `pllm.attention(layer, "aux")`: Reads the auxiliary tensor `aux` returned by that layer's custom attention operator, with shape `[rows, ...]`.
- **Restrictions and rationale**: Must be paired with [`@pllm.custom_attention`](forward-structure.md). Inside an attention method, it can only read earlier layers (`layer < current_layer`); inside external tensor methods (such as export or train), it reads data aligned to the current sampled rows.

---

## Performance and Cost: Why Reading Does Not Break CUDA Graphs

A common concern is that tapping intermediate activations during the forward pass will break CUDA graphs and fall back to slow Python eager execution. In pLLM, internal reads are graph-safe and low-overhead:

1. **In-graph capture (Hidden Tap)**: During model compilation, pLLM collects every `pllm.residual`, `pllm.hidden`, and `pllm.layer_logits` declaration and attaches lightweight copy ops at the corresponding module outputs. During CUDA graph capture, those ops copy into statically reserved **tap buffers** with zero host synchronization, keeping `llm.diagnostics().eager_reason` as `""`.
2. **Memory cost**: Each tapped site allocates a fixed GPU buffer scaled by `max_num_seqs` and hidden width; `pllm.layer_logits` additionally reserves an LM-head output buffer sized to the vocabulary width. Multiple methods in the same workload reading the same site **automatically share** one tap buffer.

You can inspect these tap buffers on CPU using [`pllm.explain`](explain.md):

```python
print(pllm.explain(Lens(), CONFIG))
```

The output itemizes the exact reservation for each tap:

```text
tap      model.layers.13.input_layernorm[1], decode,prefill: 28800 bytes shared by norm, top1
tap      model.layers.7.input_layernorm[1], decode,prefill: 28800 bytes shared by top1
tap      model.norm[1], decode,prefill: 28800 bytes shared by top1
tap      lm_head[0], prefill,decode: 29171840 bytes shared by top1
```

The residual stream after layer 12 is tapped at the input layernorm of layer 13 (`model.layers.13.input_layernorm`) and shared by `norm` and `top1`; the `layer_logits` buffer for `lm_head` is vocabulary-wide and therefore much larger (~28 MiB).

---

## Common Errors and Troubleshooting

When patterns on this page are rejected, exact error messages, causes, and fixes are listed in the [Method inputs and declarations](../reference/errors.md#method-inputs-and-declarations) and [Phases: when a value exists](../reference/errors.md#phases-when-a-value-exists) sections of the error catalog.

---

## Self-Check

<details>
<summary>1. An activation steering algorithm wants to add a steering vector onto the output of layer 12. Should it read and modify <code>pllm.residual(12)</code> or <code>pllm.hidden(12)</code>? Why?</summary>

**Answer**: Use `pllm.residual(12)`.
**Why**: The residual stream is the main backbone carrying accumulated information across layers—`residual(12)` holds the sum of token embeddings and all sublayer outputs through layer 12, and the next layer reads directly from it. `pllm.hidden(12)` is only layer 12's fresh MLP increment before the residual addition; steering on `hidden` misses the accumulated context from earlier layers.
</details>

<details>
<summary>2. Why does reading <code>pllm.base_logits()</code> inside a <code>@pllm.add_logits</code> method fail? If an algorithm needs to add a penalty based on current predicted probabilities, what should it read instead?</summary>

**Answer**: `@pllm.add_logits` runs before sampling (`BEFORE_SAMPLE`) and contributes to the logits fed into the sampler, whereas `pllm.base_logits()` is the unmodified base-model logits materialized only after sampling (`AFTER_SAMPLE`). Reading it before sampling creates a causal cycle in the schedule.
Read `pllm.layer_logits((-1,))` instead to access the model's unadjusted final-layer logits before sampling.
</details>

<details>
<summary>3. If you want to extract representations for every token in the prompt for clustering, can you read <code>tokens="all"</code> inside a <code>@pllm.train</code> method? Why or why not?</summary>

**Answer**: No.
**Why**: Online training and in-forward intervention methods such as `@pllm.train` run inside CUDA graphs whose buffers and row dimension are statically sized to `max_num_seqs`. A prompt's total token count is bounded by `max_num_batched_tokens` (typically thousands), far larger than `max_num_seqs`. Only `@pllm.export` can chunk large variable-length token batches and stream them to a host queue.
</details>
