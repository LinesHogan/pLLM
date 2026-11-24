# Two Rounding Divergences Between Fused Triton Kernels and Eager PyTorch: FP16 Absorption and Weak-Scalar Promotion

> **TL;DR**: When fusing a chain of PyTorch operators into a single Triton kernel, keeping intermediates in `float32` registers versus rounding back to half precision after each sub-expression can produce different outputs even when the algebraic formula is identical. Starting from why a 1-ULP difference snowballs under `temperature=0` greedy decoding, this note walks through two real numerical mismatches: an IEEE 754 `float16` tick-spacing trap on `[2048, 0]` (`ULP = 2.0`) where `+ 1` is absorbed into `0` in stepwise PyTorch, and a multi-GPU tensor-parallel (`TP > 1`) `float32` partial-sum divergence caused by PyTorch's weakly-typed Python `float` promotion on `27.0 * 0.123`. We close with three tiny deterministic test probes that catch rounding and stride bugs missed by `torch.randn`.

When building custom operators inside an LLM inference engine, you frequently end up maintaining two implementations of the same math:

- **An eager PyTorch reference path** (used for static single-layer hooks, prototyping, and ground-truth checks), written in three or four readable lines of ATen matrix ops;
- **A handwritten fused Triton kernel** (used on the online serving fast path), which fuses multiple projections and elementwise ops inside registers to avoid streaming intermediate tensors to and from HBM.

Both paths implement the exact same linear algebra, and standard `torch.randn` unit tests pass `torch.testing.assert_close` without complaint. Why bother chasing a 1-ULP (Unit in the Last Place) rounding discrepancy in the least significant bit?

In an image classifier, you rarely care. Under **autoregressive greedy decoding (`temperature = 0`) in an LLM**, however, a single ULP can change the entire generated response:

![Big Picture: Eager Reference vs. Fused Triton Path & Greedy Decode Snowballing](../../assets/personal-03-fig0-en.svg)

Whenever two top candidate tokens sit within `0.0005` of each other at step $t$, a 1-ULP difference flips `argmax`—say from `"However"` on the reference path to `"Therefore"` on the fused path. Because autoregressive decoding feeds the chosen token right back in as step $t+1$'s input, twenty steps later the exact same weights produce two completely different paragraphs, leaving users convinced that your online kernel loaded the wrong checkpoint.

While auditing bit-level parity between our static PyTorch ops and our fused Triton kernels, two minimal boundary inputs exposed surprising rounding divergences.

---

### Trap 1: Why Does `2048 + 1 - 2048` Equal `0` in Stepwise `float16`?

The first mismatch appeared in a low-rank affine intervention (LoReFT), which adds a rank-$r$ update to hidden state $h$ given projection $W$, orthonormal basis $R$, and bias $b$:

$$\Delta h = (h W^\top + b - h R) R^\top$$

Here are the two implementations side by side. The PyTorch reference chains standard operators directly, while our initial fused Triton kernel followed standard GPU practice—**accumulating dot products in `tl.float32` registers and casting back to `float16` once at the end**:

```python
# Path A: Stepwise PyTorch reference (`h, W, R, b` are all float16)
def eager_loreft(h: Tensor, W: Tensor, R: Tensor, b: Tensor) -> Tensor:
    coeff = h @ W.T + b - h @ R   # Four separate ATen ops, each writing a float16 tensor
    return h + coeff @ R.T

# Path B: Initial fused Triton kernel (core math excerpt)
@triton.jit
def fused_loreft_v0(...):
    # Accumulate Wh and R^T h in float32 registers, casting only on final store
    acc_w = tl.dot(h_tile, w_tile)          # tl.float32
    acc_r = tl.dot(h_tile, r_tile)          # tl.float32
    coeff = acc_w + b_fp32 - acc_r          # Stays in tl.float32 throughout!
    ...
```

Across hundreds of `torch.randn(32, 4096, dtype=torch.float16)` tests, both paths agreed closely. Then we fed in a two-element toy input:

```python
h = torch.tensor([[2048.0, 0.0]], dtype=torch.float16, device="cuda")
W = torch.tensor([[1.0, 1.0]], dtype=torch.float16, device="cuda")
R = torch.tensor([[1.0], [1.0]], dtype=torch.float16, device="cuda")
b = torch.tensor([1.0], dtype=torch.float16, device="cuda")
```

In exact arithmetic, $h W^\top = 2048$, $h R = 2048$, the inner coefficient is $2048 + 1 - 2048 = 1$, and adding $[1, 1]$ back onto $h = [2048, 0]$ gives **`[2049, 1]`**.

Running both paths on GPU produced:

- Initial fused Triton kernel: **`[2048., 1.]`** (coordinate 0 rounds to `2048` on final `float16` store; coordinate 1 is `1`);
- Eager PyTorch reference: **`[2048., 0.]`**—the `1` in coordinate 1 vanished completely!

Where did `+ 1.0` go? Here it helps to recall a hardware detail easy to forget in daily ML work: **floating-point ticks are not evenly spaced—the larger the number, the wider the gap between representable values.**

An IEEE 754 `float16` value has 1 sign bit, 5 exponent bits, and 10 explicit mantissa bits. Between any two consecutive powers of two, those 10 mantissa bits divide the interval into $2^{10} = 1024$ equal steps:

- On $[1, 2)$, the interval width is $1$, so adjacent `float16` values sit $1 / 1024 \approx 0.000977$ apart;
- On **$[2048, 4096)$ ($[2^{11}, 2^{12})$)**, the interval width is $2048$, so dividing into $1024$ steps makes **the spacing between adjacent `float16` numbers (`1 ULP`) exactly `2.0`!**

![Trap 1: IEEE 754 float16 Tick Spacing (ULP = 2.0) on [2048, 4096) Swallows +1.0](../../assets/personal-03-fig1-en.svg)

Think of a kitchen scale whose smallest tick mark above 2,048 grams jumps to 2 grams: in `float16`, `2048.0`, `2050.0`, and `2052.0` exist, **but `2049.0` does not exist at all**.

Tracing PyTorch's four eager steps makes the disappearance obvious:

1. `wh = h @ W.T` writes `float16(2048.0)`;
2. `wh_b = wh + b` computes `2048.0 + 1.0 = 2049.0`. Forced back into `float16`, `2049.0` lands halfway between `2048.0` and `2050.0`; IEEE 754 round-to-nearest-even picks `2048.0` (whose least significant mantissa bit is `0`), **swallowing `+ 1.0` whole**;
3. `hr = h @ R` produces `float16(2048.0)`;
4. `wh_b - hr` evaluates `2048.0 - 2048.0 = 0.0`.

Our fused Triton kernel kept `acc_w + b - acc_r` in 23-bit-mantissa `tl.float32` registers (where `1 ULP` at `2048` is `0.000244`), preserving `2049.0` and returning `1.0`.

To make the fused kernel match the eager PyTorch reference bit-for-bit, **we do not need to split the kernel or touch HBM**—four in-register casts reproduce the exact intermediate rounding points at register speed:

```python
# Fixed: match stepwise half-precision rounding inside registers with 0 extra HBM traffic
wh = acc_w.to(elem_ty).to(tl.float32)
wh_b = (wh + b).to(elem_ty).to(tl.float32)
hr = acc_r.to(elem_ty).to(tl.float32)
coeff = (wh_b - hr).to(elem_ty).to(tl.float32)
```

---

### Trap 2: Why Did `27.0 * 0.123` Match on 1 GPU, Then Diverge Under Multi-GPU Tensor Parallelism?

Right after fixing `float16` absorption, multi-GPU Tensor Parallelism (`TP > 1`) tests on our low-rank LoRA kernel uncovered a second discrepancy—one that never appeared on a single GPU.

First, a quick look at what changes under the hood when `TP > 1`. In a Transformer MLP block, `down_proj` is typically a `RowParallelLinear` layer sharded along its input dimension: each GPU computes a local **partial sum**, and a cross-GPU `All-Reduce` sums those partial results together.

- **On a single GPU (`TP = 1`)**: No cross-GPU reduction is needed, so the linear layer's `output` buffer is directly `bfloat16`;
- **Across multiple GPUs (`TP > 1`)**: To avoid rounding each GPU's partial sum to `bfloat16` *before* summing across ranks, the engine keeps `output` in **`float32`** until both the base projection and any LoRA delta have been accumulated and `All-Reduce`d, casting back to `bfloat16` only at the end.

In the eager PyTorch reference path, accumulating the scaled LoRA update into `output` was a one-liner (where `result` is an unscaled `bfloat16` projection tensor, e.g. `27.0`, and `self.scale = lora_alpha / rank` is a plain Python `float`, e.g. `0.123`):

```python
# Eager PyTorch reference:
# result: bfloat16 Tensor (e.g., 27.0)
# self.scale: Python float (e.g., 0.123)
# output: bfloat16 when TP == 1, but float32 when TP > 1!
output[:, offset : offset + width].add_((result * self.scale).to(output.dtype))
```

In our initial Triton kernel, we wrote:

```python
# Initial Triton kernel:
corr = acc.to(tl.bfloat16)                  # Match unscaled bfloat16 projection (27.0)
scaled = corr.to(tl.float32) * scale        # Multiply in float32 -> 3.3210001
delta = scaled.to(out_ptr.dtype.element_ty) # Cast to output dtype (stays float32 when TP > 1)
```

On a single GPU (`TP = 1`), both paths wrote `3.328125`. Under `TP = 2` (`output` in `float32`) with `result = 27.0` and `scale = 0.123`:

- Initial Triton kernel wrote **`3.3210001`**;
- Eager PyTorch wrote **`3.3281250`**!

At first glance that looks backwards: a Python `float` is a 64-bit IEEE double, so why did `bfloat16(27.0) * 0.123` in PyTorch lose *more* precision than `float32`?

The answer lies in PyTorch's **weakly-typed Python scalar promotion rule**.

If multiplying `half_tensor * 0.5` promoted the entire tensor to `float32` or `float64` just because `0.5` is a Python `float`, half-precision models would silently double their memory footprint everywhere. To prevent accidental upcasting, PyTorch treats bare Python `float` and `int` scalars as **weakly typed**: when multiplied with any floating-point `Tensor` (`float16` or `bfloat16`), **the Python scalar adopts the tensor's lower precision instead of promoting it!**

![Trap 2: Multi-GPU FP32 Partial Sums Meet PyTorch's Weakly-Typed Python float](../../assets/personal-03-fig2-en.svg)

Tracing `(result * self.scale).to(output.dtype)` with that rule in mind:

1. `result * 0.123` multiplies a `bfloat16` tensor by a weak Python `float`, so the product immediately rounds to `bfloat16` (`3.321` snaps to the nearest `bfloat16` grid point, `3.328125`);
2. `.to(torch.float32)` simply pads zeros onto an already-rounded `3.328125`;
3. Single-GPU tests hid the discrepancy because `output` was `bfloat16` too, so Triton's `3.3210001` got rounded to `3.328125` on final store anyway!

Adding one explicit cast to `hidden_ptr.dtype.element_ty` after scaling inside the Triton kernel aligned single-GPU and multi-GPU paths completely:

```python
corr = acc.to(hidden_ptr.dtype.element_ty)
# Match PyTorch's (bf16_tensor * python_float) same-dtype rounding before casting to output:
scaled = (corr.to(tl.float32) * scale).to(hidden_ptr.dtype.element_ty)
delta = scaled.to(out_ptr.dtype.element_ty)
```

During the same audit, we also fixed an implicit contiguous-column assumption in the store address (`row * out_stride0 + out_offset + cols`), passing `out_stride1` explicitly as `row * out_stride0 + (out_offset + cols) * out_stride1` so strided column slices like `output[:, ::2]` land in the right memory locations.

---

### Three Practical Habits for Testing Handwritten GPU Kernels

Standard `torch.randn` tests missed both bugs because unit-normal values cluster tightly in `[-3, 3]`—nowhere near `float16`'s `ULP = 2.0` absorption zone at `2048`—and loose default tolerances (`atol=1e-2`) wave away `0.007` scaling differences.

Three tiny additions to any kernel test suite catch these issues early:

1. **Pair `torch.randn` with 2-to-4-element deterministic "microscope probes"**: Test one cross-magnitude cancellation (`[2048.0, 0.0] + 1.0 - 2048.0` in `float16`) and one non-dyadic scale factor (`0.123` into a `float32` output buffer) with `rtol=0, atol=0` against the reference op;
2. **Guard non-contiguous slice stores with sentinel columns**: Allocate a 2x-wide output buffer, fill odd columns with `-99.0` (`buf[:, 1::2] = -99.0`), pass the even-column view `buf[:, ::2]` into the kernel, and assert afterward that odd columns still read `-99.0`;
3. **Lock batch bucket sizes when comparing end-to-end greedy decode**: When checking greedy token-for-token parity on a full 7B model, keep both paths at the exact same batch size so they hit the same CUDA Graph bucket and cuBLAS split-K reduction order.

These kernels run in production inside **[pLLM](https://github.com/LinesHogan/pLLM)**. One closing note on design trade-offs: we inserted in-register half-precision casts specifically because our static and dynamic entry points must produce identical greedy token streams for the same weights; if you are building a greenfield kernel with no legacy eager reference to match, keeping `tl.float32` throughout is numerically cleaner—what matters is knowing exactly where every rounding boundary sits.
