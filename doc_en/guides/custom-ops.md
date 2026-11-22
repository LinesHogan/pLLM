# Custom Operators

When an inference-time algorithm needs an operation that PyTorch cannot express efficiently—such as lookup-table hashing, chunked state scans, or sparse attention page scoring—researchers typically write dedicated Triton or CUDA kernels.

In pLLM, integrating custom kernels does not require learning a proprietary operator framework: wrap your kernel as an operator object using PyTorch's native `torch.library.custom_op`, and call it directly inside a `pllm.Workload` method body just like a regular function. pLLM handles capturing it into CUDA graphs under continuous batching, planning meta-tensor shapes and device memory on CPU, tracking GPU-to-CPU synchronizations, and verifying numerical equivalence between your kernel and its CPU reference implementation via contract checks.

## Quickstart: Using a Custom Operator in a Workload

Below is a complete minimal example. We define a pure-PyTorch row-normalization operator `row_norm`, register its meta shape inference and FLOP formula, and invoke it inside an `@pllm.export` method.

```python
import math
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import register_flop_formula

import pllm
from pllm import testing


@torch.library.custom_op("my_ops::row_norm", mutates_args=())
def row_norm(x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch implementation, which also serves as the CPU reference."""
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)


@row_norm.register_fake
def _row_norm_fake(x: torch.Tensor) -> torch.Tensor:
    """Meta-tensor shape inference: provides output shape and dtype to the planner."""
    return torch.empty(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)


@register_flop_formula(torch.ops.my_ops.row_norm)
def _row_norm_flops(x_shape, *args, **kwargs) -> int:
    """Declare per-call FLOPs: vector norm takes roughly 2 FLOPs per element."""
    return 2 * math.prod(x_shape)


@dataclass(frozen=True)
class ExportNorm(pllm.Workload):
    @pllm.export(name="norm", max_cols=1)
    def observe(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        # Call the op object directly rather than torch.ops.my_ops.row_norm
        return row_norm(h)


config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4)

# 1. Analyze execution plan and operator cost on CPU (zero GPU overhead)
report = pllm.explain(ExportNorm(), config)

# 2. Verify scheduling contracts on CPU
result = testing.check(ExportNorm(), config)
```

When you call `pllm.explain`, the report lists the operator's call site, FLOPs per row, and a lower bound on logical memory traffic in bytes. Calling `testing.check` executes the operator on CPU to verify scheduling contracts such as row permutation isolation and recomputation consistency.

## Integration Rules and Rationale

Calling a custom operator inside a workload method requires following the rules below, each of which corresponds to an execution constraint of continuous batching or CUDA graphs:

### 1. `mutates_args=()` and Returning New Tensors

You must declare `mutates_args=()` when defining the operator, perform no in-place mutations on input tensors inside the body (such as `x.add_()` or slice assignment), and always return new tensors.

- **Why**: Under continuous batching and CUDA graphs, the activations and state tensors passed to a method are shared or paged memory views borrowed by the current task. Other methods in the same step may be reading the same activation buffer, and earlier tasks may still hold references to prior state pages. In-place mutation bypasses pLLM's versioned state publication and corrupts data.
- **How**: Always allocate and return new tensors for operator outputs; publish any persistent state changes exclusively through framework effects such as `@pllm.write` or `@pllm.train`.

### 2. `register_fake` Is Required

After defining the operator, you must register a meta-device shape-inference function via `@op.register_fake`.

- **Why**: `pllm.explain` and the engine's startup memory planner run entirely on CPU. The framework traces the full execution flow using PyTorch `meta` tensors (which track shape and dtype without allocating physical storage) to determine per-step peak memory, task buffers, and GPU-to-CPU syncs. Without a fake implementation, shape inference breaks and planning cannot proceed.
- **How**: Have the fake function take the same arguments as the real operator and return meta tensors via `torch.empty(...)` or `torch.empty_like(...)` with the exact output shape and dtype.

### 3. Output Shapes Must Depend Strictly on Input Shapes

Every dimension of an operator's output tensors must be derivable from input tensor shapes alone—never from values inside the tensors (do not call `ctx.new_dynamic_size()` or slice dynamically based on tensor data).

- **Why**: Workload methods are captured into CUDA graphs. CUDA graphs require the memory topology and tensor sizes across the entire batch to be statically fixed at capture time and cannot accommodate data-dependent size changes.
- **How**: If a result is logically variable-length (for example, top-K sparse scoring), return a fixed maximum-length tensor paired with a boolean mask indicating valid entries.

### 4. Call the Op Object Directly, Not `torch.ops.<namespace>.<name>`

Inside workload methods, invoke the operator object created by `torch.library.custom_op` directly (e.g., `row_norm(h)`) rather than going through the dispatcher (`torch.ops.my_ops.row_norm(h)`).

- **Why**: When compiling and analyzing a workload, pLLM statically inspects the global symbols referenced by each method to extract the operator definition and its underlying Triton/CUDA kernel source. Kernel source changes are hashed into the workload's identity fingerprint so cached graphs are automatically invalidated when the kernel changes. Looking up via `torch.ops` hides the source object and prevents the framework from tracking versions or shipping the registration to worker processes.
- **How**: Define the operator at module top level or import it via `from my_kernels import my_op`, and pass arguments directly to that object inside your method.

### 5. Register `register_autograd` for Online Training

If the operator sits on the backward path of `@pllm.train` online training, you must register a backward rule with `op.register_autograd(...)`.

- **Why**: pLLM's online training relies on PyTorch's autograd engine to build the gradient graph. If gradients must flow through a custom operator that lacks a backward rule, autograd raises an error and aborts. Operators used only in forward inference, residual rewrites, sampling guidance, or activation exports do not need autograd registration.
- **How**: Write a backward function returning gradients for each input and bind it via `op.register_autograd(backward_fn)`.

### 6. Register `register_vmap` in Per-Token Cells

If the operator is called inside an `@pllm.custom_forward` unit executing per token (`over="token"`), you must register a batching rule with `op.register_vmap(...)`.

- **Why**: Under `over="token"`, the framework uses `torch.vmap` to vectorize single-token logic across the whole token batch. `vmap` rejects any custom operator without a registered batching rule.
- **How**: Register the batching behavior with `op.register_vmap(vmap_fn)`, or switch to whole-span scheduling (`over="span"`) when per-token granularity is unnecessary.

### 7. Optional `register_kernel("cpu")`

We recommend providing a pure-PyTorch CPU reference implementation for your GPU kernel. If `device_types="cuda"` is omitted, the Python function body itself serves as the fallback implementation; if `device_types="cuda"` is specified explicitly, register a CPU version via `@op.register_kernel("cpu")`.

- **Why**: `pllm.testing.check` verifies continuous-batching scheduling contracts (such as cross-row isolation and chunked prompt handling) quickly without touching a GPU or loading weights. Without a CPU reference, CPU contract checks cannot execute the operator's math, and subsequent GPU checks cannot compare across devices.
- **How**: Write and register a mathematically equivalent function using native PyTorch ops on CPU.

### 8. Optional `register_flop_formula`

You can register a FLOP formula using `@register_flop_formula(torch.ops.<namespace>.<name>)`.

- **Why**: `pllm.explain` estimates the compute cost of each step in your workload. With a formula registered, the report displays `[declared] FLOP/row`; without one, it marks the call as `unknown [no flop formula, counted 0]`.
- **How**: Register a function that accepts tensor shapes and returns the total floating-point operations for that call.

## Verification and Device Equivalence (`check_on_gpu`)

Passing `testing.check` on CPU only confirms that the CPU reference implementation satisfies the scheduling contracts. When you write a custom Triton or CUDA kernel, the real risks are numerical divergence between the GPU kernel and CPU reference, or hidden GPU-to-CPU host synchronizations inside the kernel.

`pllm.testing.check_on_gpu` catches both issues on a real GPU engine:

```python
# check-docs: skip: requires a GPU engine and compiled CUDA/Triton kernels
print(testing.check_on_gpu(ExportNorm(), config))
```

For workloads containing custom operators, `check_on_gpu` runs a **device equivalence** check: it copies the step-by-step input tensors from the CPU contract test script onto the GPU, executes the actual GPU kernel, and compares outputs against the CPU reference within the default tolerances for the dtype (`torch.testing.assert_close`). It also verifies that 8 consecutive steps of steady decode run without any host synchronization.

If the GPU kernel diverges mathematically from the CPU reference, `check_on_gpu` raises `ContractError`:

```text
1 contract failure(s)
[device equivalence] [device equivalence] add: Tensor-likes are not close!

Mismatched elements: 12715 / 13440 (94.6%)
Greatest absolute difference: 2.53125 at index (5, 581) (up to 1e-05 allowed)
Greatest relative difference: 5536.0 at index (8, 688) (up to 0.016 allowed)
```

The failure report pinpoints the fraction of mismatched elements along with the maximum absolute and relative errors, helping you track down missing terms or precision issues in the kernel.

## Measured Throughput and Practical Guidance

Before writing a custom kernel, keep one reality in mind: **custom kernels do not automatically speed up end-to-end inference**.

In [09 Custom Triton Kernels](../examples/09_custom_kernels.ipynb), we inserted a relative steering operator after layer 12's MLP output (computing residual-stream norms and adding a scaled direction vector). On an RTX 4090 running vLLM 0.32.0 with Qwen2.5-7B-Instruct (bf16, CUDA graphs) across 128 concurrent requests (8 prompts × 16 samples, 256 tokens each, median of 3 rounds after 1 warmup round), we measured:

```text
Native vLLM   :   5872 token/s, 100% of native vLLM
Triton kernel :   5760 token/s,  98% of native vLLM
PyTorch op    :   5755 token/s,  98% of native vLLM
```

The handwritten Triton kernel and the pure-PyTorch op achieve virtually identical throughput (both 98% of native vLLM). A single-layer residual vector addition takes negligible time and is nowhere near the compute bottleneck of full-model inference.

The right motivation for writing a custom kernel is when an algorithm genuinely involves operations PyTorch cannot express efficiently—such as custom hash collision handling or fine-grained masked scans over token history—not rewriting basic linear algebra in hopes of free speedups.

For complete Triton kernel definitions, safetensors weight binding, and benchmark code, see [09 Custom Triton Kernels](../examples/09_custom_kernels.ipynb). For details on method-body boundaries, see [Workload Anatomy](workload-anatomy.md); for pre-launch cost analysis, see [Seeing Costs Early: pllm.explain](explain.md).

## Common Errors and Troubleshooting

For exact error messages, root causes, and fixes when patterns on this page are rejected, see the [Custom operators](../reference/errors.md#custom-operators) and [Method bodies](../reference/errors.md#method-bodies) sections of the Error Catalog.

## Self-Check

<details>
<summary>1. Why must a workload method call the custom op object directly instead of <code>torch.ops.my_lib.my_kernel(...)</code>?</summary>

pLLM statically inspects the global symbols referenced by each method so it can read the custom operator and its underlying Triton/CUDA kernel source directly. Kernel source changes are hashed into the workload's identity fingerprint to trigger graph recompilation automatically when a kernel is edited, and the op object carries its registration to worker processes. Going through the `torch.ops` dispatcher obscures the source object and prevents the framework from tracking kernel changes or shipping the registration cleanly.
</details>

<details>
<summary>2. Suppose a custom operator filters out tokens based on scores so its output length varies per step. Can it return a variable-length tensor directly? Why or why not?</summary>

No. Workload methods run inside CUDA graphs, which require tensor memory layouts and every dimension size to be completely static at capture time. If an output shape depends on tensor data, graph capture will fail. Instead, return a fixed maximum-length tensor alongside a boolean mask, and ignore invalid positions in downstream operations using `torch.where` or masked multiplication.
</details>

<details>
<summary>3. If a custom operator is used only for rewriting the residual stream (<code>@pllm.custom_residual</code>) and exporting observations (<code>@pllm.export</code>), does it need <code>register_autograd</code>?</summary>

No. `register_autograd` is only required when the operator lies on the backward pass of `@pllm.train` online training. Pure inference, activation exports, and forward-only rewrites do not invoke backpropagation and need no autograd rule.
</details>
