# Bandwidth Accounting at the Decode Tail—and a 5-Line IEEE 754 Order-Preserving Bit Trick

> **TL;DR**: Autoregressive decode ends in a memory-bandwidth bottleneck over the vocabulary matrix, where a few innocent lines of logit rescoring can run slower than `lm_head` itself. Measuring how many tokens actually survive `top-p` filtering turns a full-vocabulary GEMM into on-demand dot products over a handful of surviving rows; packing `(float32 logit, uint32 token_id)` into a single `uint64` integer comparison takes a 5-line IEEE 754 order-preserving bijection borrowed from radix sort. We close with two reusable kernel-debugging habits—seed-locked token-by-token parity and profiling batch `max` rather than batch mean—that surface subtle BF16 rounding shifts and tail stragglers early.

Adding custom sampling guidance to the end of an LLM decode loop—say, using an online-trained lightweight predictor $g$ to suppress bland candidate tokens in the trunk hidden state $h$—usually takes one line of math: over the candidate set surviving the trunk's `top-k` / `top-p` filter, rescore each surviving token as $(1 + \beta) s_i - \beta g_i$ before drawing the next token.

In a research script, the natural implementation projects both $h$ and $g$ through `lm_head`, waits for the sampler filter to mask rejected columns to `-inf`, and calls `torch.where`:

```python
scores = torch.where(torch.isfinite(s), (1.0 + beta) * s - beta * pred, s)
```

Drop that snippet into a concurrent serving engine and open a profiler trace, and the timeline delivers an immediate surprise: after more than twenty Transformer layers finish, those last few vocabulary lines burn more GPU time than running `lm_head` itself.

---

To diagnose why, draw the dataflow and byte volumes at the end of a decode step before reading any kernel code.

Take a 7B model serving 128 concurrent decode requests with a vocabulary of `152,064` and hidden dimension `3,584`. Every step pushes data through this bandwidth funnel:

```text
  Hidden state h [128, 3584] (< 1 MB)
        │
        ▼  Multiply by lm_head weights [3584, 152064] (BF16, ~1.1 GB HBM read!)
  Raw vocabulary logits [128, 152064]
        │
        ▼  Materialize FP32 sampler workspace (~78 MB HBM read/write)
  Temperature scaling + top-k / top-p mask (rejected columns set to -inf)
        │
        ▼
  Gumbel-Max stochastic draw ──► 128 next token_ids
```

Staring at the byte counts makes the bottleneck obvious. At 128 rows per step, the arithmetic in `[128, 3584] × [3584, 152064]` barely warms up modern Tensor Cores; runtime is dominated by streaming **1.1 GB** of weights out of HBM. Once past `lm_head`, the FP32 `[128, 152064]` sampler workspace spans nearly **80 MB**.

The naive snippet pays that toll twice over on the most congested stretch of the step: first streaming the 1.1 GB `lm_head` matrix a second time for $g$, then letting eager PyTorch decompose `torch.where` into seven elementwise kernels (`isfinite`, cast, multiply, subtract, select, and friends) that sweep back and forth across the 80 MB workspace seven times.

That reframes the optimization problem cleanly: at the bandwidth-bound decode tail, **the goal is not trimming arithmetic instructions—it is driving extra full-vocabulary HBM round-trips to zero**. Any math piggybacked inside registers during an already-mandatory memory pass is effectively free.

---

How do you avoid reading that 1.1 GB `lm_head` matrix a second time? Flip the question around: **after `top-p = 0.95` (at temperature `0.8`) finishes masking the vocabulary, how many of those `152,064` columns are actually still alive?**

Instead of guessing, instrument a live decode stream and count finite entries per row. Across `652,000` real decode rows, the survivor count is heavily skewed:

```text
Surviving candidates out of 152,064 after top-p = 0.95 filtering:
  • Mean:               4.83 tokens  (> 99.99% of columns are already -inf)
  • Max over 652k rows: 2,785 tokens (< 2% of the vocabulary even at peak entropy)
```

Seeing `4.83` turns the algorithm upside down: when four or five candidates survive on a typical step, multiplying $g$ against all `152,064` columns only to throw away `99.997%` of the dot products is pure waste.

Invert the execution order instead: wait until upstream filtering finishes, collect the surviving scores and indices into a small fixed scratchpad during a single scan over the filtered row, fetch only those four or five rows from `lm_head` (5 rows of BF16 weights total **35 KB**), compute five dot products with $g$, and draw the Gumbel sample inside the same Triton kernel.

---

Writing that Triton kernel hits a concrete low-level obstacle right away.

Whenever the kernel scans survivors, finds a row maximum, or invokes `tl.sort` to cap rare high-entropy rows to the top-$K$ candidates, deterministic tie-breaking requires a composite ordering: **sort by `float32` `logit` descending first; when two tokens tie in score, prefer the smaller `uint32` `token_id` ascending.**

GPU register primitives like Triton's `tl.maximum` and `tl.sort` only compare a **single scalar tensor**—you cannot hand them a two-field C++ `std::sort` comparator lambda, and splitting comparisons into multi-pass masks bloats register pressure and branching.

How do you make hardware execute a composite `(float32_desc, uint32_asc)` comparison using a single plain integer `a > b` instruction?

The inspiration comes from two classic systems domains: **floating-point radix sort** in graphics and HPC (summarized by Michael Herf's 2001 `FloatFlip` note and used inside NVIDIA CUB's `RadixSort`), and **memcomparable composite index keys** in storage engines like RocksDB and MySQL. Both construct an **order-preserving bijection** from IEEE 754 floats onto unsigned integers.

Draw the bit layout of an IEEE 754 `float32`, and the derivation falls out visually:

```text
IEEE 754 float32 bit layout (32 bits total):
  31 30          23 22                                           0
 ┌──┬──────────────┬──────────────────────────────────────────────┐
 │S │ Exponent(8b) │                 Mantissa (23b)               │
 └──┴──────────────┴──────────────────────────────────────────────┘
  ▲
  └── Sign bit S: 0 for positive, 1 for negative (sign-magnitude, NOT two's complement!)

Viewed directly as an unsigned integer uint32, the real line is folded backward on negatives:
  Numeric order:   -inf  ──────►  -0.0   │   +0.0  ──────►  +inf
  Raw bits (hex): 0xFF800000 ──► 0x80000000 │ 0x00000000 ──► 0x7F800000
                  (S=1: more negative = larger bits, inverted!) (S=0: already monotonic!)

A symmetric bit transform unfolds the entire real line monotonically onto uint32:
  • Negatives (S=1) -> bitwise NOT all 32 bits (~bits): MSB flips to 0 (lower half) & order reverses!
  • Positives (S=0) -> XOR only the sign bit (^ 0x80000000): MSB flips to 1 (upper half), order stays!

  Transformed:    0x007FFFFF ──► 0x7FFFFFFF │ 0x80000000 ──► 0xFF800000
                     (-inf)        (-0)         (+0)           (+inf)
```

Why does this work so cleanly? IEEE 754 designed its biased exponent specifically so non-negative floats could be compared with integer hardware; the only obstacles are the sign bit and the reversed magnitude of negatives. Canonicalizing `-0.0` to `+0.0` first, flipping the sign bit on non-negatives to lift them into the upper half, and complementing all 32 bits on negatives to drop and reverse them into the lower half maps every finite float strictly in order onto `uint32`.

Take one more step: shift that 32-bit ordered float into the upper half of a `uint64`, pack `0xFFFFFFFF - token_id` into the lower 32 bits (turning smaller token IDs into larger unsigned values), and reserve `0` for masked `-inf` entries:

```text
Composite uint64 sort key (a single integer key_a > key_b resolves score + tie-break):
  63                              32 31                               0
 ┌──────────────────────────────────┬──────────────────────────────────┐
 │    Order-preserving float32      │       0xFFFFFFFF - token_id      │
 └──────────────────────────────────┴──────────────────────────────────┘
       High 32 bits: logit descending     Low 32 bits: token_id ascending
                   (All-zero key 0 reserved for -inf, sorting last)
```

Translated into Triton, encoding and decoding are symmetric and fit in ten lines:

```python
@triton.jit
def _keys(logits: tl.tensor, tokens: tl.tensor, finite: tl.tensor) -> tl.tensor:
    logits = tl.where(logits == 0, 0.0, logits)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(finite, (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - tokens.to(tl.uint32)), 0)

@triton.jit
def _decode(keys: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where(ordered & 0x80000000 != 0, ordered ^ 0x80000000, ~ordered)
    return bits.to(tl.float32, bitcast=True), (0xFFFFFFFF - keys.to(tl.uint32)).to(tl.int32)
```

---

Once a custom GPU kernel compiles and runs fast, how do you verify that it neither drifts numerically nor hides latency spikes over hundreds of thousands of production steps? Two general diagnostic habits caught every single edge case we hit afterward.

**Habit 1: Replace eyeball fluency checks with seed-locked token-by-token parity.**

Eyeballing generated prose lets 99% of numerical discrepancies slip right past you. Lock the random seed instead, run the fused kernel side by side against the naive full-vocabulary reference across hundreds of thousands of generated tokens, and halt at the very first token divergence.

On our first parity run, several thousand tokens matched verbatim before an entire sentence suddenly branched off. Inspecting the candidate distribution on the diverging row revealed why: our fused kernel kept dot products in `float32` registers (as did the companion logit-bias kernel), whereas upstream's unquantized `lm_head` rounded its projection output through `bfloat16` on the way to memory. Almost everywhere that tiny mantissa difference is harmless—except when a borderline token sits right on the `top-p = 0.95` cumulative probability cutoff. Skipping the intermediate `bfloat16` rounding kept one extra token alive, shifting the downstream Gumbel draw and altering the rest of the sequence.

Once isolated, the fix took half a line in registers: apply an explicit `.to(bf16).to(fp32)` step to mirror upstream's rounding staircase and inline upstream's exact `gumbel_block_argmax` helper. Across nearly `300,000` generated tokens, mismatches dropped to zero.

**Habit 2: In batched GPU kernels, profile the batch `max` (`p99.9`), never just the mean.**

Earlier we saw that a typical row has `4.83` survivors. Thinking in averages, sizing the per-row scratchpad capacity `K` to `256` looks fifty times larger than necessary. Yet under realistic load, profiler traces showed periodic latency spikes every few dozen steps—and raising `K` to `4096` actually ran *faster* than `256`.

Pulling out the specific rows behind those spikes exposed two phenomena that batch averages completely hide:
- **Batch straggler coupling**: 128 requests execute concurrently inside one kernel launch, so step latency equals `max(row_0, ..., row_127)`, not the mean. Even if 127 rows finish 5 dot products in microseconds, a single high-entropy row exceeding `256` survivors falls into the multi-pass `tl.sort` overflow path while the other 127 thread blocks wait. Spending `4 MiB` of scratch memory to set `K = 4096` above the empirical ceiling (`2,785`) buys out the straggler branch;
- **The `0.12%` unmasked confident rows**: A subset of slow-path rows were not uncertain at all—they were maximally confident! When the top token's probability exceeded `0.9999999`, it cleared `top_p = 0.95` by itself, so upstream's filter took an early-exit shortcut and returned the raw logits untouched without writing a single `-inf`. Tracking the softmax denominator `mass` during the scan and short-circuiting directly to the argmax dot product whenever no `-inf` appears and `1.0 / mass >= top_p` eliminated those spikes immediately.

*(Tracking steady-state memory watermarks the same way caught one bonus PyTorch registration trap: wrapping a variable-shape Triton launch with `torch.library.wrap_triton` and passing `tuple[Tensor, ...]` makes the dispatcher memoize the tensor tuple as a compile-time specialization key, pinning temporary tensors forever on every step; registering a `@torch.library.custom_op` taking `list[Tensor]` treats them as normal runtime operands.)*

---

This bandwidth-first tail pipeline powers `@pllm.guide_sampling` and `@pllm.add_logits` in **[pLLM](https://github.com/LinesHogan/pLLM)** (`pllm`), our open-source programmable extension layer on top of vLLM, keeping 7B serving with 128-way online per-request training and guided sampling above `96%` of baseline throughput.

Knowing the exact constraint model also tells you when this sparse gather stops applying: indexing individual vocabulary rows requires an unsharded, non-block-quantized `lm_head` matrix on the local GPU. Under tensor parallelism (`TP > 1`) across the vocabulary axis or block-packed quantized weights, local thread blocks cannot evaluate arbitrary vocabulary dot products independently; the engine then falls back to concatenating $h$ and $g$ along the batch axis (`lm_head(torch.cat((h, g), dim=0))`)—trading away sparsity, while still guaranteeing that the 1.1 GB weight matrix streams out of HBM only once.
