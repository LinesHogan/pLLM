# Dynamic Layer Switching Without Recapturing CUDA Graphs: Conditional Subgraphs, Masked Evaluation, and Early-Exit Triton Kernels

> **TL;DR**: Once an online serving engine has captured CUDA Graphs across batch buckets, dynamically switching the target layer of an activation intervention, tap, or LoRA adapter by re-recording the graphs stalls traffic for seconds, while evaluating every candidate layer and masking with `torch.where` wastes HBM bandwidth on inactive layers. After comparing the private cuBLAS workspace footprint of CUDA conditional subgraphs, we settled on an early-exit Triton kernel gated by an L2-resident scalar and paired with in-place accumulation on the existing inter-layer normalization output port, letting missed candidate layers skip both weight-pool and activation-pool HBM traffic. We close by tracing a CUDA Graph capture failure hidden inside the `rank > 256` fallback branch, triggered by a host-synchronizing `bool(mask.any())`.

When analyzing internal model representations or running online activation steering, a familiar requirement turns up quickly: the model is already loaded and CUDA Graphs are already warmed up across every batch-size bucket, and now you want to move a residual tap or steering vector from layer 8 to layer 20—or, inside a single continuous-batching step, route Request A through a layer-4 LoRA, Request B through a layer-18 LoRA, and Request C through `-1` to leave the base model untouched.

If the target layer ID is baked into the static graph structure, every layer switch forces you to detach the old hook, recompile, and recapture every CUDA Graph bucket (`[1, 2, 4, 8, ..., 256]`). A single switch takes anywhere from a few seconds to over ten seconds—far too slow for interactive experiments or live serving.

To avoid recapturing graphs, the hooks for every candidate layer in `over=(4, 12, 20, ...)` must be wired into the graph at build time, with a `0-d int64` scalar in the state pool controlling which layer takes effect at runtime. That raises an immediate engineering question: when the candidate set is large—say, sweeping every layer of a 28-layer model with `over=tuple(range(28))`—only 1 layer fires on a given step. How do you keep the other 27 missed layers from draining latency and memory?

---

We compared three approaches for skipping inactive-layer work inside an already captured static graph.

**Approach 1: Device-side conditional graph nodes (Conditional CUDA Graph / `device_branch`, CUDA 12.4+).**  
The idea is to place a hardware-level `IF (active_count > 0)` node at each candidate layer, executing the branch subgraph's matrix multiplications only when at least one request in the step hits that layer.

In practice, memory consumption is the main constraint: when each conditional branch captures its internal operators (such as batched cuBLAS GEMMs), it allocates a private workspace out of its own branch-stream memory pool.

![Conditional CUDA subgraphs multiplying cuBLAS workspace across batch buckets and candidate layers](../../assets/personal-02-fig1-en.svg)

On top of that, under heterogeneous continuous batching, a single request hitting layer $$l$$ flips the entire layer's `IF` condition to true, so the subgraph still has to perform row-level filtering inside anyway. Layer-level conditional branching consumes VRAM meant for the KV cache while failing to remove row masking for mixed batches.

**Approach 2: Full evaluation followed by masking (`torch.where(mask, h + delta, h)`).**  
This is the most straightforward option: every candidate layer gathers parameters by slot and runs the matrix multiplication to compute `delta`, then `torch.where` zeroes out updates for inactive rows. It adds no private workspace and captures cleanly into CUDA Graphs, yet the 27 missed layers still load parameters and hidden states out of HBM and compute through them. While auditing that overhead, we also caught an unconditional `captured[:len(hidden)].copy_(hidden)` in the existing static adapter path—originally there to retain forward inputs for online autograd—and skipping that copy in inference-only mode trimmed per-layer write bandwidth as well.

**Approach 3: CTA Early-Exit Triton kernels gated by an L2-resident scalar, paired with in-place accumulation on existing output ports.**  
Before each forward step, the engine builds a few-hundred-byte `int64` ownership table `owners[rows]` on device (`-1` for inactive rows), small enough to stay resident in on-chip L2 cache throughout the pass. Our custom Triton kernel partitions the grid by `(row, col_block)`, and each thread block's very first instruction loads `owner = tl.load(owners_ptr + row)` from L2 and returns immediately if it is below `0`:

![Full HBM evaluation + torch.where vs. L2 scalar early-exit Triton kernel](../../assets/personal-02-fig2-en.svg)

There is an easy-to-miss detail here: if the activation-patch op is written as an out-of-place function returning a new tensor (`out = masked_patch_add(hidden, table, pool)`), inactive rows skip reading the parameter pool, yet they still have to read the entire `hidden` row from HBM and write it into `out` just to preserve the original activations—effectively adding a full-width memory copy on every missed layer.

Looking at the surrounding inter-layer `RMSNorm` boundary solves this cleanly: the outer wrapper already allocates both output ports `(residual, normed)` before calling the boundary hook and has already copied the branch output into `normed`. Turning the patch op into an in-place accumulation onto `normed` (`masked_patch_add_into(normed, table, pool)`) lets missed rows return without touching `normed` in HBM at all:

```python
@triton.jit
def _masked_patch_add_into_kernel(
    target_ptr, table_ptr, pool_ptr,
    table_stride, target_stride, pool_stride, width, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Check the L2-resident mask first; return immediately on a miss without touching target_ptr or pool_ptr
    if tl.load(table_ptr + row) == 0:
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < width
    slot = tl.load(table_ptr + table_stride + row)
    dst_ptrs = target_ptr + row.to(tl.int64) * target_stride + cols
    h = tl.load(dst_ptrs, mask=col_mask, other=0.0)
    delta = tl.load(pool_ptr + slot.to(tl.int64) * pool_stride + cols, mask=col_mask, other=0.0)
    tl.store(dst_ptrs, h + delta.to(target_ptr.dtype.element_ty), mask=col_mask)
```

Low-rank LoRA adapters follow the same structure: pre-allocate a `[max_rows, rank]` `float32` scratch buffer `routed_mid`, run a two-stage early-exit kernel, and accumulate straight into `output[:, offset:offset+width]` in stage two. When an outer `@pllm.custom_forward` reorders a sub-batch (`call_rows != None`), a single `index_select(0, call_rows)` on `owners` lets the exact same kernels handle it.

---

Reviewing the dispatch wrapper around these kernels uncovered a graph-capture bug tucked inside a low-frequency parameter branch.

Because the two-stage low-rank kernel caps its fast path at `rank <= 256`, calls with `rank > 256`, `float64` inputs, or a non-contiguous trailing stride fall back to standard PyTorch ops. To avoid extra work on that fallback path, the initial version reused the CPU branch's short-circuit guard:

```python
if target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
    mask = table[0, :rows].bool()
    if bool(mask.any()):  # Never reached for rank <= 256; breaks graph capture as soon as rank = 260
        ...
```

Routine tests all ran at `rank = 4` or `8`, so GPU execution always took the Triton fast path and this fallback only ever ran on CPU—where calling `bool(mask.any())` is completely fine. The moment a user declared `rank = 260` on GPU with CUDA Graphs enabled, warmup capture hit `bool(mask.any())`, forcing PyTorch to synchronize the stream and copy the boolean reduction back to the host to construct a Python `bool`. The CUDA driver failed immediately with `cudaErrorStreamCaptureUnsupported`.

The fix separates the CPU short-circuit from the CUDA fallback explicitly: even when taking the slow path, CUDA stays purely on device via `torch.where` without converting any tensor into a Python scalar, backed by a GPU graph-capture test at `rank = 260`:

```python
if not target.is_cuda or target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
    mask = table[0, :rows].bool()
    if not target.is_cuda and not bool(mask.any()):
        # Use Python bool short-circuit on CPU only
        return
    # Keep the CUDA fallback strictly on-device via masking so stream sync never breaks CUDA Graph capture
    ...
    target.copy_(torch.where(mask[:, None], target + delta, target))
    return
```

---

Measured on a single RTX 4090 running `Qwen2.5-7B-Instruct` (`bfloat16`, CUDA Graphs enabled), the actual overhead breaks down as follows:

- With dynamic activation steering and asynchronous residual export attached to 4 candidate layers (`over=(4, 12, 20, 26)`) and live layer switching via `import_state` between turns: decode TPOT at 128 concurrent requests (`batch=8, n=16`) is `24.6 ms`, compared to `24.4 ms` for a static single-layer hook and `23.2 ms` for stock vLLM with no hooks—dynamic gating adds roughly `0.2 ms` over the static single-layer baseline.
- With low-rank LoRA attached across all 28 layers (`over=tuple(range(28))`, 56 `q_proj` / `v_proj` projection hooks total, 1 layer active per step), the entire set of bucketed CUDA Graphs uses `0.30 GiB` of VRAM, and decode TPOT at 128 concurrency is `24.8 ms` (versus `24.0 ms` for a static single-layer LoRA—roughly `0.8 ms` across all 27 missed layers combined).

The code is merged into `pllm.model.LayerRoute` in **[pLLM](https://github.com/LinesHogan/pLLM)**. Its boundary costs remain worth keeping in mind: early-exit kernels eliminate HBM traffic and GEMMs on missed layers, yet each candidate layer still retains one lightweight kernel launch (on the order of ten-plus microseconds per layer), inter-layer patch boundaries still plumb residual and branch ports, and `per_head`, `rank > 256`, or non-contiguous trailing-stride patch inputs fall back to masked evaluation. That is why the API exposes `LayerRoute(..., over=(...))` as an explicit opt-in requiring callers to declare only the candidate layers they actually intend to switch across, rather than blanketing the full model by default.
