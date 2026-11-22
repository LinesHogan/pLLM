# Editing Forward Structure: Residuals, Attention, and Layer Execution Order

Modifying a model's computation structure in PyTorch or Hugging Face is straightforward: rewrite `forward()`, insert hooks, or call custom functions between layers. In a throughput-oriented serving engine like vLLM, however, dozens of concurrent requests are packed into a single tensor by continuous batching, and the whole model forward pass is captured into CUDA graphs. Forking model source files to insert Python logic breaks operator fusion, forces fallback to eager execution, and leaves per-request private state unmanaged.

pLLM provides three decorators specifically for restructuring the forward pass while preserving CUDA graph throughput and request isolation:
1. `@pllm.custom_residual`: overrides how sublayer outputs merge into the residual stream, replaces the values read by attention, or attaches a chunk-updated fast-weight memory sublayer.
2. `@pllm.custom_attention`: customizes how attention reads KV pages (top-K sparse reads, attention monitoring) or declares physical KV cache retention layouts (sliding window, prompt sink + window, cross-layer KV sharing).
3. `@pllm.custom_forward`: orchestrates layer execution order in ordinary Python (looped depth, token-wise early exit / layer skipping, cross-step continuous thought state).

---

## 1. Custom Residual Connections: `@pllm.custom_residual`

### Use Cases and Motivation
Use `@pllm.custom_residual` when an algorithm applies custom math at the boundaries where Transformer sublayers read from or write back to the residual stream. Common examples include:
- **Value Residual** (e.g., ResFormer): mixes shallow-layer (e.g., layer 0) values into subsequent layers' values.
- **Hyper-Connections**: expands the single residual stream into $n$ parallel streams, applying weighted reads before sublayers and linear mixing afterward.
- **LAuReL**: adds low-rank corrections or gated scaling when merging a sublayer output into the residual stream.
- **Stateful memory sublayers** (e.g., TTT, Titans): attaches a per-request fast-weight memory matrix alongside attention or MLP that reads and updates chunk by chunk.

In native vLLM, residual addition and LayerNorm / RMSNorm are typically compiled into fused kernels (Fused Add+Norm). Hacking model code to alter residuals forces the whole model back to eager mode. `@pllm.custom_residual` lets you take over residual updates at selected sites while native attention projections, MLP computations, and KV cache writes run unchanged—and the rewritten operators remain fully captured inside the model's CUDA graphs.

### Minimal Runnable Example (CPU)
The example below implements ResFormer-style Value Residual: layers 1 through 23 mix layer 0's values into their own values before writing to the KV cache and running attention.

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class ValueResidual(pllm.Workload):
    mixing: float = 0.0

    @pllm.custom_residual(site="value", layers=tuple(range(1, 24)))
    def mix(self, v: torch.Tensor, first: torch.Tensor = pllm.value(0)) -> torch.Tensor:
        return (v.float() + self.mixing * (first.float() - v.float())).to(v.dtype)


print(repr(pllm.explain(ValueResidual(), CONFIG).eager_reason))
```

```text
''
```

`eager_reason` is the empty string `''`, confirming the workload stays completely inside the model's CUDA graphs.

### Sites (`site`) and Input Signatures
`@pllm.custom_residual` uses `site` to select which residual boundary to own:

| `site` | Positional tensors (no defaults) | Return value | Native behavior replaced | Typical use |
| --- | --- | --- | --- | --- |
| `"attention"` | Pre-sublayer residual `x`, attention branch output `y` | Updated residual stream | `x + y` | Attention residual scaling, gated branches |
| `"mlp"` | Pre-sublayer residual `x`, MLP branch output `y` | Updated residual stream | `x + y` | MLP residual rewrites, attached fast-weight memory |
| `"both"` | Same as above, invoked in forward order on attention then MLP | Updated residual stream | `x + y` | LAuReL, Hyper-Connections sublayer mixing |
| `"value"` | Current full-attention layer's value rows `v` `[rows, kv_heads, head_dim]` | Replacement value | Native `v` | Value Residual (feeds attention and writes to KV cache) |
| `"final"` | Residual stream `x` before final norm | Single residual stream `[rows, hidden]` | Native residual stream | Merging multi-stream Hyper-Connections back to a single stream |

**Signature and input rules**:
1. **Positional tensors first**: `attention`, `mlp`, and `both` must take `x, y` as their first two parameters; `value` and `final` must take the site tensor as their first parameter, all without default values.
2. **Input markers on subsequent parameters**: after the positional tensors, you may read:
   - `pllm.parameter(state, field)` or `pllm.module(state)`: read carried fields from an instance or request state.
   - `pllm.depth()`: current layer index (Python `int`). With `site="both"`, attention gets `2 * layer` and MLP gets `2 * layer + 1`.
   - `pllm.value(layer)`: the value tensor produced at an earlier layer for the same token in this forward pass.
   - `pllm.residual(layer)`: the residual stream output at the end of an earlier layer for the same token in this forward pass.
3. **Precision**: incoming tensors have the model's native dtype (e.g., `bfloat16`). For numerically sensitive accumulations or divisions, cast to float32 (`.float()`) inside the method and cast back (`.to(v.dtype)`) before returning.
4. **Multiple residual streams (`streams=n`)**: setting `streams=n` ($n > 1$) expands the residual stream from embedding onward to `[rows, n, hidden]`. Pair this with `read="method"` naming an `(x, ...) -> branch_input` method that collapses the $n$ streams into one before each sublayer, and declare one `site="final"` method to merge back to `[rows, hidden]`. See [07 Custom Residuals](../examples/07_custom_residual.ipynb).

### Identity Restoration and Numerical Equivalence
When a custom residual method returns the identity expression (such as `x + y` for `site="attention"` or `v` for `site="value"`), model outputs are **bitwise equal** to native vLLM in eager mode.

**Why is there a tiny difference from native vLLM in default graph mode?**
In default graph mode, native vLLM enables PyTorch Inductor fused kernels (Fused Add+Norm). When `@pllm.custom_residual` is loaded, vLLM instead captures the model's unfused operators into CUDA graphs. This preserves exact PyTorch arithmetic—matching native vLLM eager mode bitwise—while differing by normal bf16 rounding ulps from Inductor's fused kernels. Compare identity runs against eager native vLLM or use `testing.check_on_gpu`.

### Stateful Memory Sublayers (Fast Weights)
Algorithms like TTT or Titans maintain a per-request fast-weight state $M$ (e.g., a `[hidden, hidden]` matrix) that is read and updated as tokens flow.

To declare a stateful sublayer on `@pllm.custom_residual`:
- `state="name"`: points to a `pllm.RequestState` with a single field (its carried memory) and no optimizer.
- `chunk=N`: fixed chunk size $N$ (a positive integer, e.g., 16).
- Supported only on a single `"attention"` or `"mlp"` site at **one** layer.

```python
@dataclass(frozen=True)
class FastWeightMemory(pllm.Workload):
    gate: float = 0.0
    lr: float = 0.1

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState({"M": lambda m: (m.hidden_size, m.hidden_size)})

    @pllm.custom_residual(site="mlp", layers=(12,), state="memory", chunk=16)
    def remember(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        M: torch.Tensor = pllm.parameter("memory", "M"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = F.normalize(x.float(), dim=-1)
        read = k @ M.T
        stream = (x.float() + y.float() + self.gate * read).to(x.dtype)
        next_M = M - self.lr / len(x) * (read - y.float()).T @ k
        return stream, next_M


print(repr(pllm.explain(FastWeightMemory(), CONFIG).eager_reason))
```

```text
''
```

**Update semantics and design rationale**:
Every row within the same chunk (16 rows) reads the **carried state from the start of that chunk**; only after the final row of the chunk completes does the returned `next_M` take effect as the starting state for the next chunk.
If a request's remaining tokens in a scheduling step do not fill 16 rows, the framework buffers the partial rows and triggers the state update once subsequent steps complete the chunk. Because vLLM's continuous-batching scheduler may arbitrarily split long prompts across steps (Chunked Prefill), anchoring updates strictly to fixed chunk boundaries guarantees that **outputs are completely independent of how the scheduler chunks prompts**.

### Overhead and Limitations
- **Overhead**: custom residual logic is captured directly into CUDA graphs, avoiding per-step Python sync overhead; throughput cost comes solely from the added tensor math. Measured on an RTX 4090 with Qwen2.5-7B (vLLM 0.32.0, 256 tokens, median of 3 rounds after 1 warmup; see [07 Custom Residuals](../examples/07_custom_residual.ipynb)):
  - Value Residual (editing 27 value layers, 32 concurrent requests): **99%** of native CUDA graph throughput.
  - Hyper-Connections ($n=4$, editing 56 sublayers, 32 concurrent requests): **93%** of native CUDA graph throughput.
  - Stateful memory sublayer (reading/writing a `[hidden, hidden]` FP32 matrix per token, ~49 MiB per request on 7B, so run at 8 concurrent requests): **75%** of native CUDA graph throughput at 8 concurrent requests.
- **Limitations and reasons**:
  - `tensor_parallel_size=1`: residual rewrite operators run inside single-GPU layer forward graphs; cross-GPU residual tensor and state synchronization is not supported yet.
  - Prefix caching must be disabled (`enable_prefix_caching=False`): prefix caching assumes identical prompt prefixes yield identical KVs; residual edits and stateful updates produce different activations or cross-token state accumulation, invalidating prefix reuse.
  - Stateful memory sublayers require raising `ExecutionOptions(cuda_graph_pool_budget_bytes=...)`: per-request private state memory and chunk history buffers exceed the default 64 MiB graph capture budget (7B at 8 concurrent requests needs ~1.9 GiB); if the budget is too small, engine startup fails with the exact byte requirement.
  - Stateful memory sublayers **cannot** be combined with `@pllm.custom_forward`: a stateful sublayer advances its memory strictly once per token, whereas a custom forward may loop or skip layers within a token, leaving update timing undefined.

---

## 2. Custom Attention Reads and KV Layouts: `@pllm.custom_attention`

### Use Cases and Motivation
Use `@pllm.custom_attention` when an algorithm modifies how attention reads information or how the KV cache is laid out physically. Typical examples:
- **Attention monitoring**: extracts intermediate statistics (such as the fraction of attention mass falling on a local window) without altering model outputs.
- **Training-free sparse reads** (e.g., Quest): dynamically scores KV pages and selects only top-K logical positions per head for attention.
- **Physical KV cache shrinking**: uses local sliding windows, prompt sink + sliding window (R-SWA), or shares KV caches across adjacent layers.

In an inference engine, attention is a deeply fused kernel whose KV block allocation and reclamation are managed by a paged memory pool. Traditionally, trying sparse reads or changing KV geometry meant hacking C++/CUDA kernels. `@pllm.custom_attention` preserves native Q/K/V projections and KV writes, replacing only the FlashAttention read step and exposing declarative physical KV layouts.

### Minimal Runnable Example (CPU)
The two examples below demonstrate common patterns: one keeps only a 256-token sliding window on even layers (and full KV on odd layers); the other monitors layer 12's local attention mass and exports it to the host.

```python
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """Keep the most recent 256 tokens of KV on even layers, full KV on odd layers."""

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


print(testing.check(LocalGlobal(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

```python
@dataclass(frozen=True)
class RecentMass(pllm.Workload):
    """Monitor the fraction of layer-12 attention falling on the last 64 positions without altering outputs."""

    recent: int = 64

    @pllm.custom_attention(layers=(12,))
    def split(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = kv.attend(q)
        _, near = kv.attend(q, window=self.recent)
        # aux returns the attention share on recent positions: exp(LSE_local - LSE_global)
        return out, (near - lse).exp().mean(-1, keepdim=True)

    @pllm.export(name="recent_mass", max_rows=8, max_cols=1)
    def report(self, share: torch.Tensor = pllm.attention(12, "aux")) -> torch.Tensor:
        return share


print(testing.check(RecentMass(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

### Method Signature and `PagedKV`
A custom attention method has signature `f(self, q: torch.Tensor, kv: pllm.model.PagedKV, ...markers)`:
- `q`: RoPE-applied Query tensor of shape `[rows, heads, head_dim]`.
- `kv`: read-only paged KV view `pllm.model.PagedKV`. The current token's K/V have already been written to the cache. It exposes `key_cache`, `value_cache`, `block_table`, `seq_lens`, and related attributes for custom inspection.
- `kv.attend(q, *, window=0, topk=None)`: runs attention and returns `(output, float32 LSE [rows, heads])`.
  - `window`: lookback window size from the current position (including the current token).
  - `topk`: logical token position tensor of shape `[rows, k]` or `[rows, heads, k]` (padded with `-1`).
- **Optional `aux` return value**: the method may return `(out, aux)`. Downstream layers or export methods can read this tensor via `pllm.attention(layer, "aux")`, while `"lse"` reads the float32 log-sum-exp from the first `kv.attend` call.

### Why Can Such a Minimal `kv.attend` Signature Express Rich Custom Attention?

At first glance, `@pllm.custom_attention` exposes neither a `[rows, heads, seq_len]` attention-weight matrix nor an arbitrary 2D `attn_mask`, nor in-place mutation of cached K/V—its core primitive is just `kv.attend(q, *, window=0, topk=None) -> (out, lse)`. That minimalism is deliberate: it preserves three hardware and serving invariants simultaneously:
1. **Keeping online softmax fused in on-chip SRAM**: FlashAttention is fast precisely because it tiles $Q K^\top$, softmax, and $\times V$ inside SRAM without ever materializing `[rows, heads, seq_len]` matrices in HBM; exposing dense weight or mask tensors would immediately degrade long-context bandwidth and memory back to quadratic.
2. **Keeping continuous batching inside static-shape CUDA graphs**: requests in the same batch have different `seq_len` values, so any intermediate tensor carrying a `seq_len` axis introduces dynamic shapes or host syncs that break full-model graph capture.
3. **Portability across physical cache architectures (GQA and DeepSeek MLA)**: in DeepSeek V2/V3 Multi-head Latent Attention (MLA), the paged cache stores compressed latent vectors (`kv_lora_rank`) plus shared RoPE keys rather than expanded multi-head K/V. Abstracting reads behind `kv.attend` lets the exact same workload code run FlashAttention on GQA models and matrix-absorbed latent attention directly over compressed MLA pages without decompressing the cache.

Within those invariants, four orthogonal mechanisms compose to express nearly every hardware-friendly custom attention variant:
- **`(out, lse)` forms an online-softmax monoid (sufficient statistics for subset mass & disjoint union)**: the returned `float32` `lse` always has fixed shape `[rows, heads]` ($\log \sum_{j \in S} e^{s_j}$), independent of sequence length, yet retains the exact normalizer:
  - **Subset probability mass** (e.g., recent-window mass, Lookback Lens, sink ratio): call `kv.attend` on subset $S$ to obtain `lse_s`, then compute `(lse_s - lse_full).exp()` in two lines with zero `[seq_len]` allocations.
  - **Exact union of disjoint regions** (e.g., anchor/sink positions `topk` $\sqcup$ local `window`, or multi-tier sparse windows): for any two disjoint index sets $A$ and $B$, calling `kv.attend` for `(out_a, lse_a)` and `(out_b, lse_b)` reconstructs exact attention over $A \sqcup B$ via `m = torch.logaddexp(lse_a, lse_b)` and `out = (lse_a - m).exp()[..., None] * out_a + (lse_b - m).exp()[..., None] * out_b` (ensure $A$ and $B$ do not overlap, or shared positions would be double-counted).
  - **Query-space transforms and head-wise composition** (e.g., Differential Attention $\text{Attn}(q_1) - \lambda \text{Attn}(q_2)$, mixing retrieval heads and streaming heads via `torch.where`, per-head gating): transform fixed-shape `q` and `out` directly around `kv.attend`.
- **Fixed-width `-1`-padded `topk` expresses per-head dynamic budgets inside static graphs**: when different layers or heads select different numbers of positions, allocate `[rows, heads, K_max]` at the maximum budget and mask unused slots to `-1` on GPU (`masked_fill(..., -1)`). The kernel treats `-1` as $-\infty$ (returning zero output and `-inf` LSE when all entries are `-1`), enabling adaptive sparsity without dynamic shapes.
- **`aux` and `pllm.attention(...)` enable cross-layer index reuse**: index-sharing schemes (such as IndexCache) can score pages once in an earlier layer, return `(out, chosen)`, and accept `chosen: torch.Tensor = pllm.attention(earlier_layer, "aux")` in deeper layers to call `kv.attend(q, topk=chosen)` without re-scoring.
- **Orthogonal separation of pre-cache value rewrites, physical block reclamation, and read selection**: modifying values before cache write (e.g., Value Residual or LoRA) belongs to `@pllm.custom_residual(site="value")` or `pllm.model.Adapter`; freeing physical blocks outside a window belongs to `keep=` / `window=` / `sink=` / `source=` on the decorator; and `@pllm.custom_attention` focuses purely on reading the committed cache, preserving native KV writes and block reclamation.

### Static KV Layout Declarations
`@pllm.custom_attention` lets you declare physical KV cache retention directly on the decorator:
- `keep="window", window=N`: retains only the most recent $N$ tokens of KV; older physical blocks are reclaimed into the pool by vLLM's `SlidingWindowSpec`.
- `sink=True` (requires `keep="window", window=N`): retains the entire prompt plus the most recent $N$ generated tokens (mapping to vLLM's `RSWASpec`).
- `keep="none", source=k` ($k \ge 1$): allocates no KV memory for this layer and instead reads layer $k$'s KV cache directly (`source` must point to an earlier layer; to read a deeper layer's KV, use `@pllm.custom_forward` with `kv="peek"`).

### Why Top-K Saves Bandwidth Without Saving VRAM, and Why Score-Based Eviction Is Unsupported
- **Why Top-K saves reads, not VRAM**:
  Passing selected indices to `kv.attend(q, topk=...)` makes FlashAttention load only those keys and values during softmax and GEMM, saving FLOPs and HBM read bandwidth. However, all historical tokens' KVs must remain in physical VRAM pages so future tokens can re-score and select from them. Thus Top-K read selection does not free physical KV capacity.
- **Why true score-based eviction is unsupported**:
  Algorithms like H2O and SnapKV that evict physical blocks mid-sequence by attention score are not supported in the current architecture (see [Design Trade-offs](../explain/design-tradeoffs.md)).
  **Why**: vLLM's physical block manager relies on a hard invariant—each request's logical KV sequence is a contiguous 1D array starting at position 0 whose block table grows monotonically. Punching holes mid-sequence to free blocks breaks block-table addressing, positional indexing, and continuous-batching scheduling.

### Overhead and Limitations
- **Overhead**:
  - Static layouts substantially reduce physical VRAM usage. On Qwen2.5-7B (2048-token context), `explain`'s per-request KV memory ceiling drops from **112.0 MiB** (full) to **70.4 MiB** with a 256-token window on even layers, and **56.0 MiB** with cross-layer sharing—doubling concurrency capacity.
  - In the same example measured on RTX 4090 (7B, 128 concurrent requests): passthrough `NativeRead`, even-layer window `LocalGlobal`, and cross-layer `CrossLayerShared` all achieve **99%** of native vLLM throughput (windowed layers use vLLM's FlashInfer window kernel; see [Performance and Memory](../explain/performance.md)). Pure-PyTorch dynamic Top-K scoring (such as Quest without a custom kernel) runs at **13%** of native throughput; production speed requires a fused operator (see [Custom Operators](custom-ops.md) and [09 Custom Triton Operators](../examples/09_custom_kernels.ipynb)).
- **Limitations and reasons**:
  - `tensor_parallel_size=1`: custom FlashAttention reads and paged KV layouts do not yet implement cross-TP head sharding and sync.
  - Requires FlashAttention 2 backend: pLLM reads page tables and sequence lengths from FA2 metadata to organize each read; plain sliding-window reads then dispatch to vLLM's built-in FlashInfer kernel because FA2's window path disables GQA packing (see [Performance and Memory](../explain/performance.md)).
  - Unquantized KV (`kv_cache_dtype="auto"`): quantized formats like FP8 pack storage bytes and would require dedicated dequantization kernels during sparse gather.
  - Standard RoPE attention: attention variants with ALiBi, logit soft-capping, or hardware-specific sink tokens are unsupported.
  - Prefix caching disabled: sliding windows and cross-layer sharing change the KV layout retained per request, breaking prefix caching's global hash-tree assumptions.
  - Fully paged KV models only: supported only on models whose KV state lives entirely in vLLM paged blocks (Qwen2/2.5, Llama, Qwen3, etc.); hybrid models with extra recurrent state (such as Qwen3.5 DeltaNet) are unsupported.
  - Static layouts cannot combine with `PositionState` or `@pllm.custom_forward`: static layouts partition vLLM into multiple KV groups, whereas custom forwards and position state require a single unified KV group.

---

## 3. Custom Layer Execution Order: `@pllm.custom_forward`

### Use Cases and Motivation
Use `@pllm.custom_forward` when an algorithm reorders or dynamically alters the layer execution trajectory. Typical examples:
- **Looped / Recurrent Depth**: models like Huginn or Ouro that repeat a block of layers multiple times during decode to trade extra compute for deeper reasoning.
- **Per-token early exit / Adaptive depth**: methods like CALM or LayerSkip that let easy tokens skip deep layers during generation.
- **Continuous thought cross-step state**: methods like Coconut that feed the hidden vector produced by one token directly into the next token via `pllm.PositionState`.
- **Speculative dry runs**: looking ahead one step without corrupting the main KV cache.

In standard inference, the forward pass runs sequentially from layer 0 to the last layer in a fixed topology. Writing loops or layer skipping externally incurs complex KV cache bookkeeping and heavy host-device blocking syncs. `@pllm.custom_forward` wraps the forward pass into a slice-callable `layers(...)` object so you can orchestrate execution in Python while the framework automatically splits and captures per-slice CUDA graphs under the hood.

### Minimal Runnable Example (CPU)
The example below implements Ouro-style looped depth with per-token early exit: during decode, layers 20–23 loop up to 2 extra passes; any row whose residual change falls below a threshold converges and exits the loop early.

```python
@dataclass(frozen=True)
class LatentLoop(pllm.Workload):
    start: int = 20
    loops: int = 2
    tol: float = 0.05

    @pllm.custom_forward(tokens="decode")  # active= is restricted to decode
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        active = torch.ones_like(hidden[:, :1], dtype=torch.bool)
        for _ in range(self.loops):
            again = layers(start=self.start, hidden=hidden, active=active)[-1]
            change = (again - hidden).float().norm(dim=-1, keepdim=True) / hidden.float().norm(dim=-1, keepdim=True)
            hidden = torch.where(active, again, hidden)
            active = active & (change > self.tol)  # converged rows stop computing in later loops
        return hidden


print(testing.check(LatentLoop(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile', ...)
```

### Calling `layers(...)`, `LayerOutputs` Indexing, and KV Semantics
The `layers` argument is a `pllm.model.Layers` instance with the following contract:
- **Call signature**: `layers(start=0, stop=None, hidden=None, *, kv="commit", active=None) -> LayerOutputs`
  - Runs layers `start` through `stop - 1`. If `stop` is omitted, runs through the final layer; if `hidden` is omitted, starts from token embeddings (valid only when `start=0`).
- **Absolute indexing on `LayerOutputs`**:
  `LayerOutputs` must be indexed by **absolute model layer index**! For example, after `out = layers(12, 16)` (which runs layers 12–15), accessing layer 15's output requires `out[15]`; negative index `out[-1]` always refers to the last layer run by that call. Indexing outside the slice (such as `out[0]`) raises `ValueError: index a tap inside the Layers slice that produced it`.
- **`kv="commit"` vs. `kv="peek"`**:
  - `kv="commit"` (default): computes each layer's K/V in the slice and **writes** it to the KV cache. In a loop, later passes overwrite earlier passes at the current token's slot, so subsequent generated tokens see the K/V written by the final pass.
  - `kv="peek"`: attends over historical committed KV plus the current pass's newly computed K/V, **without writing** to the KV cache (leaving the cache untouched as it was before the call). Useful for speculative lookahead or dry runs.
  - **Empirical results from the Ouro paper**:
    The Ouro paper (arXiv 2510.25741, §5.4.2 Table 14) compared KV caching schemes for decode-time looped depth on GSM8K:
    - Storing a separate KV copy for every loop pass: **78.92** accuracy.
    - Keeping only the last pass's KV (the default overwrite `kv="commit"`): **78.85** accuracy (virtually lossless vs. storing all passes).
    - Keeping only the first pass's KV (commit on pass 1, `kv="peek"` on later passes): accuracy collapses to **18.73**.
    That collapse occurs because subsequent tokens can only read coarse first-pass K/Vs, breaking a model trained on deep looped representations. Hence `kv="commit"` overwriting on each pass is the default.

### Why Per-Token Layer Skipping (`active=`) Is Restricted to `tokens="decode"`
When you pass a boolean mask (`[rows, 1]`) to `layers(..., active=mask)`, rows with `False` skip that slice's matrix math and pass their `hidden` through unchanged.
For skipped rows, the framework propagates state on the first skip at a layer by running that layer's input norm, QKV projection, and RoPE on the stopped hidden state to populate K/V, and keeps that value on subsequent skips of the same layer.

**Why is `active=` strictly limited to `tokens="decode"`?**
Propagated K/Vs on skipped layers are approximations. During decode, a skipped row affects only one new position; every prompt token, by contrast, is context read by every future generated token, so skipping layers during prefill would replace the entire context's deep K/Vs with approximations whose error compounds across every later step. Prompts therefore run the full model, and `active=` is restricted to decode.

### Cross-Step State: `pllm.PositionState` and `load` / `save`
For continuous-thought algorithms like Coconut, the model passes one token's output hidden vector directly as the next token's input via `pllm.PositionState`:

```python
@dataclass(frozen=True)
class StepLoopWorkload(pllm.Workload):
    """Save each token's pre-norm residual and feed it as layer-0 input for generated positions."""

    history = pllm.PositionState()

    @pllm.custom_forward(tokens="all")
    def loop(self, layers: pllm.model.Layers) -> torch.Tensor:
        previous = layers.load("history", offset=-1)
        taps = layers(hidden=torch.where(layers.generated, previous, layers.embeddings))
        layers.save("history", taps[-1])
        return taps[-1]


print(testing.check(StepLoopWorkload(), CONFIG).passed)
```

```text
('tensor_parallel_size=1: admission', 'compile')
```

- **Call rules**: `layers.load("history", offset=-1)` reads the vector saved at the previous position (even if `save` was already called in the current step, `load` still returns the previous token's snapshot); `layers.save("history", value)` commits the current position's snapshot (if called multiple times, the last write wins).
- **Full-prompt spans (`over="span"`)**:
  The default `over="token"` mode steps causally position by position. With `@pllm.custom_forward(over="span")`, a request's contiguous prompt span is passed together as a `[M, H]` tensor, suited for Jacobi-style parallel prompt iteration. Note that `over="span"` has no sequential predecessor token, so `layers.load` is disallowed.

### Control-Flow Rules: Bounded Loops, No Tensor Branching, and `pllm.scan`
1. **Bounded Python loops unroll automatically**: Python `for` loops over workload fields (e.g., `for _ in range(self.loops):`) unroll at trace time into sliced CUDA graphs.
2. **No data-dependent tensor branching**: method bodies must **never** contain `if tensor.norm() > 1:`, `while`, or `break`. Forward graph topology must be static; otherwise tracing raises `methods run inside CUDA graphs, so Python cannot branch on tensor values (if, while or break on a tensor); ...`. Loop a fixed number of times and express conditionals with `torch.where` or `active=` masks (`active=` requires `tokens="decode"`; see [12 Adaptive Depth](../examples/12_adaptive_depth.ipynb)).
3. **Linear scans (`pllm.scan`)**: to express long-sequence linear recurrences (of the form $S_t = A_t S_{t-1} + B_t$) inside a custom forward, call `pllm.scan(step, carry=..., over=..., chunk=...)`. pLLM recognizes affine recurrences and lowers them to a GPU **parallel prefix scan** when applicable, which is much faster than sequential chunk loops (see [Performance and Memory](../explain/performance.md) and [04 Custom Forward](../tutorials/04_custom_forward.ipynb)).

### Overhead, Limitations, and Composition
- **Overhead**:
  With `@pllm.custom_forward`, vLLM's whole-model CUDA graph is replaced, and `eager_reason` explicitly reports `@pllm.custom_forward`. **This does not mean per-operator eager execution**: pLLM captures separate per-slice CUDA graphs for each `layers(...)` call.
  In [12 Adaptive Depth](../examples/12_adaptive_depth.ipynb) (RTX 4090, Qwen2.5-7B, 32 concurrent requests, 256 tokens, vLLM 0.32.0), a sliced `custom_forward` without skipping hits **99%** of native vLLM CUDA graph throughput. (At 7B where a step is ~19 ms, native eager also reaches 99% because launch overhead is hidden; on smaller models with few-millisecond steps, sliced graphs clearly beat eager.)
- **Memory and hardware constraints**:
  - Per-slice graph capture uses its own pool space and can exceed the default 64 MiB model graph budget (Example 12's workload on 7B at 32 concurrency needs ~143 MiB). Raise `ExecutionOptions(cuda_graph_pool_budget_bytes=128 << 20)` when needed (this budget counts toward `explain`'s peak VRAM estimate, so give only a small margin above the reported byte requirement rather than allocating gigabytes). If the budget is too small, generation fails with the exact bytes required.
  - Multi-GPU tensor parallelism requires vLLM 0.32.0 or newer.
- **Mutually exclusive with `pllm.hidden`**:
  Within the same workload, `@pllm.custom_forward` **cannot** be combined with `pllm.hidden(layer)` or `pllm.final_hidden()`.
  **Why**: under looped or skipped layers, a given layer may execute multiple times or zero times, so "layer $L$'s hidden state" has no unique physical meaning. For observation, use `pllm.residual(layer)` or `pllm.layer_logits((layer,))`, which always read the residual stream from the **last visit** to that layer.
- **Interaction with `pllm.model.Adapter`**:
  If a workload also declares a `pllm.model.Adapter` (see [Editing Models](model-edits.md)), weight deltas $\Delta W$ apply **on every actual projection invocation**. If a slice runs $N$ times in a loop, deltas apply $N$ times; native QKV propagation triggered for skipped layers also applies the corresponding Q/V deltas.
- **Combining with training overlays (`apply_updates()`)**:
  When combined with `@pllm.train_in_forward`, calling `layers.apply_updates()` inside `custom_forward` flushes and concatenates projection captures accumulated since the last boundary and steps the optimizer, making the updated private weights visible to subsequent layer calls on the same token.

---

## Common Errors and Troubleshooting

For exact error messages, root causes, and fixes when declarations on this page are rejected, see [Model sites and custom forwards](../reference/errors.md#model-sites-and-custom-forwards) and [Memory budgets](../reference/errors.md#memory-budgets) in the Error Catalog.

---

## Self-Check

1. Why is an identity `@pllm.custom_residual` (returning `x + y` or `v`) bitwise equal to native vLLM in eager mode, yet slightly different in bf16 rounding from native vLLM's default graph mode?
<details>
<summary>Answer</summary>

In default graph mode, native vLLM uses PyTorch Inductor fused kernels (such as Fused Add+Norm). When `@pllm.custom_residual` is attached, vLLM captures the model's unfused operators into CUDA graphs instead. The arithmetic therefore matches eager mode bitwise while exhibiting normal bf16 floating-point rounding differences relative to Inductor's fused kernels.
</details>

2. Why does `kv.attend(q, topk=...)` save read bandwidth and FLOPs without reducing per-request KV VRAM capacity?
<details>
<summary>Answer</summary>

`kv.attend(q, topk=...)` reads only the selected positions when computing attention softmax and GEMM, saving memory read bandwidth. However, all historical tokens' KVs must remain resident in physical VRAM pages so future tokens can score and select from them, so no physical blocks can be freed.
</details>

3. In looped-depth models, how do `layers(...)`'s default `kv="commit"` and `kv="peek"` differ, and why did the Ouro paper find overwriting every pass (keeping the last pass) works well (78.85) while keeping only the first pass collapses (18.73)?
<details>
<summary>Answer</summary>

`kv="commit"` writes each pass's computed K/V into the cache, with later passes overwriting earlier ones, so subsequent tokens see the refined high-quality K/V from that token's final loop pass (78.85 on GSM8K in the Ouro paper). Using `kv="peek"` after the first pass leaves only the coarse shallow K/V from pass 1 in the cache, starving later tokens of looped deep representations and collapsing models trained with looped depth (18.73 on GSM8K).
</details>
