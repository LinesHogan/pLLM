# Why Reading 98% Less History Ran 5x Slower—and Broke on DeepSeek: Rethinking Custom Attention from Memory Layouts

> **TL;DR**: Custom attention code written around unfolded `[batch, heads, seq, dim]` `Key` and `Value` tensors hits two walls inside a production serving engine: passing `window_size` to FlashAttention-2 on Grouped-Query Attention (GQA) models breaks 7:1 head packing and runs **5.5x slower** than reading full history; switching to DeepSeek V2/V3's Multi-head Latent Attention (MLA) fails outright because separate `Key` and `Value` tensors do not physically exist in paged HBM. We walk through how a back-of-the-envelope Roofline calculation pinpoints the GQA grid regression, how dimension-guided matrix associativity derives zero-materialization latent attention, and how both physical layouts unify behind a single capability interface.

Researchers prototyping long-context probes, alternating local/global windows, or sparse top-$k$ attention in standalone scripts are used to reaching directly for `Key` and `Value` tensors—or passing a `window_size` argument to an attention kernel.

Carry that habit into a production vLLM engine, and two surprises show up almost immediately. On standard Qwen or Llama models, telling the attention kernel to read only the 4 most recent tokens instead of 256 makes the layer run **5.5x slower**; switch the model to DeepSeek V2 or V3, and custom attention stops working altogether because the paged cache contains neither `Key` nor `Value` tables.

Viewed separately, one looks like a quirky kernel regression and the other like a missing model integration. Viewed from HBM layout upward, both teach the exact same modeling lesson: **any attention abstraction coupled to unfolded multi-head `K`/`V` tensors breaks the moment it meets real hardware head-packing or compressed latent caches.**

---

Start with the first mystery: **how can reading 98% less history push single-layer decode latency from 16.6 µs up to 91.5 µs?**

Debugging single-layer kernel regressions benefits from **picking the right microscope**. On a 7B or 70B model, twenty-millisecond dense projections drown out a seventy-microsecond attention regression; on `Qwen2.5-0.5B` (14 Query heads sharing 2 KV heads, head dim 64, 128 concurrent decode rows at history length 256), where an entire step takes ~3 ms, kernel regressions jump right out of end-to-end throughput.

Replaying that exact single-layer decode shape 1,000 times in isolation gave:
- Upstream FlashAttention-2 (FA2) reading all 256 history tokens: **16.6 µs**;
- The exact same FA2 call with `window_size=(3, 0)` reading only 4 tokens: **91.5 µs**.

Resist the urge to dive straight into thousands of lines of C++ CUDA templates. Whenever a timing looks upside down, spend thirty seconds doing a **back-of-the-envelope Roofline check** on the `16.6 µs` baseline first.

Autoregressive decode processes 1 query token per request, leaving Tensor Cores largely idle and making runtime pure HBM byte traffic. Crucially, `Qwen2.5-0.5B` (like `7B`) uses **7:1 Grouped-Query Attention (GQA)**: every 7 Query heads share 1 identical KV head. In an optimal kernel, each historical `K` and `V` page streams out of HBM **once**:

$$\text{Minimum HBM traffic} = 128\text{ rows} \times 256\text{ tokens} \times 2\text{ KV heads} \times 2\text{ (K,V)} \times 64\text{ dim} \times 2\text{ Bytes} = 16\text{ MiB}$$

Divide `16 MiB` by the test GPU's `~1,000 GB/s` memory bandwidth, and the physical floor is **`16 µs`**!

That one division solves half the mystery before opening a single source file: hitting `16.6 µs` on full attention proves the baseline kernel **must** be packing all 7 Query heads sharing a KV head into the same thread block (CTA) so they share SRAM. Conversely, regressing to `91.5 µs` (~5.5x slower) sits in the exact same ballpark as the **7:1 GQA sharing ratio**!

Checking the launched grid dimensions in the profiler confirmed the diagnosis on the spot:

```text
1. Full decode (grid organized by 2 KV heads — 7:1 GQA packing active, 16.6 µs):
   ┌───────────────────┐
   │ KV Head 0 in HBM  │ ──(loaded once)──► [ Shared inside 1 CTA: Q0, Q1, Q2, Q3, Q4, Q5, Q6 ]
   └───────────────────┘

2. Raw FA2 window decode (grid falls back to 14 Query heads — packing lost, 91.5 µs):
   ┌───────────────────┐   ┌──────────────► [ CTA 0: Q0 ]
   │ KV Head 0 in HBM  │───┼──────────────► [ CTA 1: Q1 ]  ... (7 separate CTAs reload the same KV!)
   └───────────────────┘   └──────────────► [ CTA 6: Q6 ]
```

Enabling `window_size` on this decode shape dropped FA2 out of its KV-head-packed path into a 14-query-head grid, trading away sequence-length savings for redundant page fetches and window-bound overhead.

Switching windowed GQA reads to FlashInfer—whose decode kernel keeps packing Query groups per KV head even with `window_left` enabled—fixes the grid, provided you respect CUDA Graph constraints: FlashInfer is a two-phase stateful pipeline whose CPU `plan` step precomputes split schedules into fixed workspace buffers before graph replay. Rather than rebuilding workspace management around raw `import flashinfer` calls, subclassing vLLM's existing `FlashInferMetadataBuilder` in a dozen lines reused its persistent graph buffers directly, snapping windowed decode back to **18.0 µs** (>97% of full-engine throughput).

Reclaiming physical HBM blocks on sliding-window layers (`RSWASpec`: retain prompt prefix + recent window, recycling evicted middle blocks to `null_block`) surfaced a reusable lesson for debugging silently ignored config overrides in large frameworks. Setting only a subset of layers to `RSWASpec` while leaving Layer 0 on `FullAttentionSpec` reclaimed zero blocks at runtime. Inspecting vLLM's ten-line KV-cache group merger exposed classic OOP inheritance swallowing:

```text
Class hierarchy:   AttentionSpec ──► FullAttentionSpec ──► RSWASpec (subclass!)

vLLM group check:  Pick entry #0 (FullAttentionSpec) and ask: "Are all remaining layers isinstance(..., FullAttentionSpec)?"
                   ──► Returns True! The permissive parent swallowed its stricter sliding-window subclass.
```

Tightening equality on a thin subclass and sorting `RSWASpec` entries ahead of `FullAttentionSpec` (`sink_first`) before handing the dict to vLLM let the stricter subclass speak first—activating native physical block eviction without touching upstream code.

---

Now look at the second wall on DeepSeek V2 and V3: **how do you formulate custom windowed or sparse attention when `Key` and `Value` tensors do not exist in physical memory?**

Compare the two physical paged-cache layouts side by side:

```text
Standard GQA paged cache (Qwen / Llama):
  Stored per KV head at each token: [ KV Head 0: K, V ] [ KV Head 1: K, V ] ...

DeepSeek MLA compressed latent cache (V2 / V3):
  All heads share ONE low-rank vector: [ Shared latent c_KV (e.g. 512d) | Shared k_rope (64d) ]
  Logical reconstruction (H heads):    K_h^nope = c_KV @ W_UK[h],    V_h = c_KV @ W_UV[h]
```

MLA slashes KV-cache memory precisely because all $H$ heads share that single 512-dim $c_{KV}$ vector. If a custom attention API demands unfolded `[seq_len L, heads H, head_dim]` `K` and `V` tensors, the backend has to multiply all $L$ historical $c_{KV}$ vectors by $W_{UK}$ and $W_{UV}$ on every step—exploding $O(L)$ compressed reads into $O(L \cdot H)$ HBM traffic and wiping out MLA's entire advantage.

How do you compute exact arbitrary-window or sparse `topk` attention without ever materializing `K` or `V`?

Let **tensor dimension complexity** guide the algebra—the exact same associative trick behind evaluating LoRA as `B @ (A @ x)` and DeepSeek-V2's own inference formulation: **reassociate matrix multiplications so per-head projections move off the length-$L$ history onto the length-$1$ decode query and output.**

Follow the parentheses across both stages:

```text
Decode step: query length = 1, history length = L (L >> 1), attention heads = H

[Scoring stage]
  Naive order:      q_nope [1, H, 128]  ·  (c_KV [L, 512] @ W_UK [H, 512, 128])^T   ──► O(L · H) HBM expansion!
  Reassociated:    (q_nope [1, H, 128] @ W_UK^T [H, 128, 512])  ·  c_KV^T [512, L] ──► O(1 · H) entry projection!

[Aggregation stage]
  Naive order:      sum_t  weights [1, H, L]  ·  (c_KV [L, 512] @ W_UV [H, 512, 128]) ──► O(L · H) HBM expansion!
  Reassociated:    (weights [1, H, L] @ c_KV [L, 512])  @  W_UV [H, 512, 128]        ──► O(1 · H) exit projection!
```

Once the dimensions line up, the implementation writes itself. Paired with tiled online `logaddexp` rescaling over 128-token compressed page tiles for numerical stability at any context length, streaming latent attention fits in fifteen lines of plain PyTorch:

```python
def paged_latent_attention(q_nope, q_pe, latent_pages, rope_pages, w_uk_t, w_uv, scale, tile=128):
    # 1. Entry absorption: project the length-1 query once into the 512-d latent space
    q_abs = torch.einsum("rhp,hpl->rhl", q_nope.float(), w_uk_t.float())
    acc = torch.zeros_like(q_abs)
    lse = q_abs.new_full(q_abs.shape[:2], -float("inf"))

    # 2. Stream directly over compressed latent tiles (or gather sparse topk indices), zero K/V materialization
    for c_tile, r_tile, valid_mask in iter_compressed_tiles(latent_pages, rope_pages, tile):
        scores = (torch.einsum("rhl,rkl->rhk", q_abs, c_tile)
                  + torch.einsum("rhd,rkd->rhk", q_pe.float(), r_tile)) * scale
        scores = scores.masked_fill(~valid_mask, -float("inf"))
        tile_lse = torch.logsumexp(scores, dim=-1)
        merged = torch.logaddexp(lse, tile_lse)
        finite = torch.where(merged.isneginf(), 0.0, merged)
        weights = (scores - tile_lse[..., None]).exp().masked_fill(~valid_mask, 0.0)
        acc = (acc * (lse - finite).exp()[..., None]
               + torch.einsum("rhk,rkl->rhl", weights, c_tile) * (tile_lse - finite).exp()[..., None])
        lse = merged

    # 3. Exit absorption: project the merged latent accumulator once per head back to output dim
    return torch.einsum("rhl,hlv->rhv", acc, w_uv.float()).to(q_nope.dtype), lse
```

---

Put both halves together, and the right abstraction boundary becomes clear: **hide physical cache arrays behind a storage-agnostic attention capability.**

In **[pLLM](https://github.com/LinesHogan/pLLM)** (`pllm`), `@pllm.custom_attention` hands user methods an opaque `PagedKV` handle:

```python
@dataclass(frozen=True)
class WindowProbe(pllm.Workload):
    @pllm.custom_attention(layers=(0,))
    def monitor(self, q: torch.Tensor, kv: pllm.PagedKV):
        out, full_lse = kv.attend(q)
        _, recent_lse = kv.attend(q, window=64)
        return out, recent_lse - full_lse  # Runs unchanged across Qwen/Llama (GQA) and DeepSeek (MLA)
```

Those eight lines dispatch to pre-planned FlashInfer GQA-packed kernels on Qwen/Llama and switch automatically to latent matrix absorption on DeepSeek V2/V3—capturing cleanly inside CUDA Graphs on both.

The dimension diagram above also tells you the exact cost envelope up front: enabling vLLM's FlashInfer planning pipeline reserves a `~394 MiB` shared workspace buffer; and MLA matrix absorption wins specifically when attended history $L \gg 1$. If an ultra-sparse `topk` selector attends to only two or three tokens per step ($L \approx 2$), paying the fixed $O(H)$ entry and exit projections ($W_{UK}^\top$ and $W_{UV}$) costs more FLOPs than decompressing those two slots directly. Sketching the dimension flow first makes the crossover point visible before running a single benchmark.
