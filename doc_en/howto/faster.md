# Making Workloads Faster

Your algorithm workload runs end-to-end and produces correct results, yet autoregressive decode throughput is lower than expected—or the engine fails with out-of-memory errors at startup.

This how-to guide is a prioritized optimization checklist. Unlike offline training, autoregressive decode generates only 1 token per concurrent request per step: arithmetic intensity (FLOPs/byte) is extremely low, compute is usually abundant, and the true bottlenecks come down to three physical constraints: **GPU timeline bubbles (host synchronization or kernel launch stalls)**, **memory and bus bandwidth (full-vocabulary projections, state reads/writes, PCIe transfers)**, and **KV Cache VRAM capacity**.

For the underlying derivations and hardware models behind these constraints, see [Performance and Memory](../explain/performance.md); for definitions of the fields in the execution report, see [Reading Costs: pllm.explain](../guides/explain.md); and for how to benchmark fairly after making changes, see [Measuring Throughput Fairly](measure-throughput.md).

> [!NOTE] Benchmark Environment and Reference Numbers
> Unless stated otherwise, all measured numbers in this guide come from the same baseline environment: **single RTX 4090 (24 GiB), `Qwen/Qwen2.5-7B-Instruct` (BF16), vLLM 0.32.0, 128 concurrent sequences (8 prompts $\times$ 16 sampling branches), 256 generated tokens per sequence, CUDA Graphs enabled; median of 3 rounds after 1 warmup round via `pllm.benchmark.throughput`**. Throughput percentages are relative to native vLLM under the exact same settings with CUDA Graphs enabled (~$5,894\text{ tok/s}$, or ~$21.7\text{ ms}$ per step across 128 rows). With the same settings on a 0.5B model, one step takes only ~$3.0\text{ ms}$; below, both are contrasted wherever conclusions depend on model size.

```python
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import pllm
from pllm import testing
```

---

## 1. Confirm the Forward Pass Is Still Inside a CUDA Graph

### What to Look For

Run `pllm.explain(workload, config)` and inspect the `forward` line:

```text
forward  CUDA graphs: no effect needs eager
```

If it falls back to eager mode, this line displays `forward eager: <reason>`. After launching the engine, you can also inspect `llm.diagnostics().eager_reason` at runtime (an empty string `""` in graph mode; the concrete reason when falling back to eager).

### What It Means

Falling back to eager mode forces Python to launch hundreds of operators individually on every step. Whether that CPU overhead is fatal depends on how long a single GPU step takes: at 128 concurrent sequences on 7B, a step takes ~21.7 ms—mostly reading weights—so **falling back to eager still retains 98% of graph-mode throughput** ($5,773$ vs. $5,894\text{ tok/s}$). On 0.5B under the same settings, a step is only ~3 ms, so **eager drops to 49% of graph-mode throughput** ($20,926$ vs. $42,355\text{ tok/s}$). The smaller the model, the lower the concurrency, and the shorter the step, the more critical CUDA Graphs become; graph mode is also what lets pLLM capture training and model edits into the same graph.

### How to Fix It

Verify that your workload preserves the whole-model CUDA Graph:
1. **Both per-request and shared adapters stay in-graph**: Whether `pllm.model.Adapter` is bound to `pllm.RequestState` (one state page per request) or `pllm.InstanceState` (shared or admission-version-isolated state slots), deltas are computed directly inside the CUDA Graph via versioned page tables without breaking whole-model capture.
2. **Activation patches and sliced graphs**: `pllm.model.ActivationPatch` also stays inside the CUDA Graph. What replaces the whole-model graph is `@pllm.custom_forward` (which switches to per-layer-slice captured graphs, with the reason recorded in `eager_reason`) or uncaptured/non-traceable operators.
3. **Fail fast via config**: Set `ModelConfig("Qwen/Qwen2.5-7B-Instruct", cuda_graphs=True)`. With the default `cuda_graphs="auto"`, constructs that cannot enter the whole-model graph fall back to eager or sliced mode (with the reason printed in `forward` and `eager_reason`). Setting it explicitly to `True` makes any declaration that breaks whole-model capture raise `ValueError` immediately in `explain` or `make_llm`, preventing silent throughput regressions.

```python
@dataclass(frozen=True)
class FastLoRA(pllm.Workload):
    # Use RequestState: each request holds its own state copy, and the adapter stays fully inside the CUDA Graph
    lora = pllm.RequestState(
        {
            "a": pllm.Field(lambda in_features, out_features: (4, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 4)),
        },
        init=pllm.Imported(),
    )
    apply = pllm.model.Adapter(
        "lora",
        (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12)),
        operator="low_rank",
    )


# Enable cuda_graphs=True: any pattern that triggers eager fallback fails immediately at build time.
# With 128 requests each carrying a LoRA, the modified model's CUDA graph needs a larger capture pool
# (if insufficient, startup fails with an error stating the required byte budget).
config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct",
    max_model_len=512,
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    cuda_graphs=True,
    execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=512 << 20),
)
report = pllm.explain(FastLoRA(), config)
assert report.eager_reason == ""
```

### What It Costs

`RequestState` reserves versioned pages per request slot for independent state (for example, on 7B at 128 concurrency, `explain` reports 256 pages per field, totaling ~22 MiB). If your algorithm only needs cross-request shared state (or state advancing via online `import_state` versions), you can place the `Adapter` on `InstanceState` instead and still stay inside the CUDA Graph (reserving `2 + max_num_seqs = 130` pages when in-flight updates are supported, or just 1 page for static read-only state). Both adapter kinds increase the modified model's CUDA Graph capture pool proportionally (startup raises an error naming the exact byte budget if it is too small). Using `@pllm.custom_forward` or falling back to eager costs only ~2% on 7B, whereas eager halves throughput on 0.5B; setting `cuda_graphs=True` adds zero runtime overhead.

---

## 2. Eliminate Host Synchronization in Workload Math

### What to Look For

Inspect the `host` line in `pllm.explain` output:

```text
host     0 syncs in traced workload math
```

This line must read `0 syncs`. For in-layer model rewrites (`@pllm.custom_residual`, `@pllm.custom_attention`) and custom operators, CPU tracing reports `unverified syncs`; verify those with `pllm.testing.check_on_gpu(workload, config)`.

### What It Means

pLLM relies on the CPU running ahead of the GPU to prepare metadata asynchronously. Calling `.item()`, `.tolist()`, `.cpu()`, `print(tensor)`, `float(tensor)`, or a Python value branch `if tensor.any():` inside a workload method forces the CPU to block until the GPU finishes all previously launched kernels and copies the scalar back to host memory. That host sync completely drains the GPU execution pipeline, causing stalls of hundreds of microseconds to milliseconds.

### How to Fix It

Rewrite scalar extraction and dynamic control flow as pure tensor math:
1. **Replace branches**: Convert Python `if condition:` into `torch.where` or boolean mask multiplication.
2. **Forbid scalar conversions**: Never call `.item()` or print tensors inside workload methods.
3. **Asynchronous observation and health diagnostics**: For debugging values or collecting statistics, use `@pllm.export` (which writes into a bounded pinned ring buffer and drops events on overflow by default without blocking the GPU or CPU). At runtime, check `llm.diagnostics().numeric_health` for NaN/Inf in state tensors—this check incurs 0 per-step syncs during steady-state decode.
4. **GPU contract check**: Run `pllm.testing.check_on_gpu(workload, config)`, which executes at least 8 steady-state decode steps under `torch.cuda.set_sync_debug_mode("error")` to catch any implicit in-layer synchronizations.

```python
@dataclass(frozen=True)
class ZeroSyncWorkload(pllm.Workload):
    # Replace Python if branches with torch.where and avoid .item() to guarantee zero host syncs
    @pllm.add_logits()
    def steer(self, logits: torch.Tensor = pllm.layer_logits((-1,))) -> torch.Tensor:
        last = logits[:, -1]  # [rows, vocab]: the model's own logits
        return torch.where(last > 0.0, 0.1 * last, torch.zeros_like(last))  # amplify positive logits only


sync_report = pllm.explain(ZeroSyncWorkload(), config)
assert sync_report.workload_host_syncs == 0
```

### What It Costs

Tensor branching (such as `torch.where`) evaluates both branches mathematically, but given the abundant compute available during decode, that extra arithmetic is negligible compared to a millisecond-scale pipeline drain.

---

## 3. Ensure Side-Stream Tasks Overlap with the Forward Pass

### What to Look For

1. Check the `lane` column for each task in `pllm.explain`'s `member` list: verify whether it shows `side` (side CUDA stream) or has fallen back to `main` (main stream).
2. Check whether the `timing` line shows `publish_lag=1 (default)`.
3. Check the two thresholds printed on the `memory` line:

```text
memory   peak 1512728864 bytes funded by pLLM [per-rank tensor reservations; excludes GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1512728864 bytes tasks run inline; below 1333743904 bytes planning fails
```

Check whether your configured `execution_peak_budget_bytes` is below the overlap threshold on the first line (the example above is `Distill` from [Workload Anatomy](../guides/workload-anatomy.md) on 7B at 128 concurrency: `1512728864 bytes`).

### What It Means

Heavier compute such as online training (`@pllm.train`) is scheduled by default on a side stream (`side`) for asynchronous execution:
* **Default `publish_lag=1`**: After step $n$ produces a token, the side stream asynchronously runs backpropagation and the optimizer update, which become visible at step $n+2$. Step $n+1$'s forward reads the state version from step $n-1$ without waiting for step $n$'s training, overlapping training with the forward pass.
* **Memory budget downgrade**: Side-stream overlap requires holding workspace buffers for two versions simultaneously, raising peak memory above serial execution. If your `ExecutionOptions(execution_peak_budget_bytes=...)` is below the overlap threshold (yet above the planning failure floor), the planner does not error out—it silently downgrades side-stream tasks to inline execution on the main stream (`inline`, changing `lane` from `side` to `main`), serializing them with the model forward and losing overlap.

### How to Fix It

1. **Keep the lag**: Keep the default `publish_lag=1` rather than setting `publish_lag=0` unnecessarily (unless your algorithm mathematically requires the very next step to see the current step's parameter updates immediately).
2. **Raise the memory budget**: If `lane` fell back to `main`, check the `memory` line in `explain` and raise `execution_peak_budget_bytes` above the overlap threshold to restore side-stream overlap.

```python
exec_opts = pllm.ExecutionOptions(
    execution_peak_budget_bytes=1024 << 20,  # above overlap_threshold to preserve side-stream overlap
    publish_lag=1,  # default 1: step n updates take effect at step n+2, overlapping training with step n+1 forward
)
overlap_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct",
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    execution=exec_opts,
)
```

### What It Costs

How much overlap saves depends on how large training is relative to the whole step (`ESamp`, rank 4):
* **7B**: For pure online training ($\beta=0$), `publish_lag=1` achieves **97.4%** of native vLLM throughput vs. **97.2%** for `publish_lag=0`; with training + guided sampling ($\beta=0.1$), it is **96.4%** vs. **96.1%**. Each training step takes only ~0.6 ms of GPU time—against a 21.7 ms step, the cost of serial waiting is within measurement noise.
* **0.5B (same settings, only changing the model)**: For pure online training, `publish_lag=1` reaches **91.9%** vs. **89.8%** for `publish_lag=0`. Shorter steps make overlap significantly more valuable.

The cost is extra memory reservation, incurred only on states read at a lagged version: when a method reads a version lagged by `publish_lag` steps, each request slot must retain `publish_lag` extra old parameter versions (optimizer fields are not retained for old versions; see the formula in [Performance and Memory, Section 3.1](../explain/performance.md#31-paged-state-pool-and-per-field-page-formula-pages)). For `Distill` on 7B (`predict` reads the lagged version) at 128 concurrency, the parameter field grows from 256 to 384 pages and total state reservation rises from ~95 MiB to ~110 MiB; `ESamp` ($\beta=0$) has no lagged reader, so both settings use 256 pages per field.

---

## 4. Cut Bytes Moved per Step

### What to Look For

Inspect the task estimate lines and state lines in `pllm.explain`:
* `projection FLOP`: whether extra full-vocabulary LM Head projections are being run.
* `sampler ... bytes`: bytes read and written for full-vocabulary tensors.
* `state reads (origin, distance)`: bytes read from state.
* `logical transfer bytes/row` and `decode bytes/step` on `state` lines: per-step transfer volume for host lookup tables over PCIe.

### What It Means

Decode is bound by VRAM and bus bandwidth. Every extra byte moved per step adds directly to latency:
1. **LM Head projections and full-vocabulary tensors**: The 7B LM Head weight (3584 $\times$ 152064 BF16) weighs 1.09 GB. Each extra full-vocabulary projection costs **~$+1.35\text{ ms}$** (~6% of a step; on 0.5B, the 272 MB weight takes ~0.36 ms, which is 12% of a step). Each pass in the sampler scanning a $128 \times 152064$ vocabulary tensor takes 78–283 μs.
2. **Large table transfers over PCIe**: On an RTX 4090, PCIe 4.0 x16 reads from host memory at ~23.7 GB/s—roughly 1/40th of GPU VRAM bandwidth. Gathering 38.9 MB of wide vectors across 128 rows takes a full **1.6 ms** (~7% of a 7B step; over half a step on 0.5B).

### How to Fix It

#### 1) Merge Projections and Optimize Guided Sampling

* **Use `@pllm.guide_sampling`**: Guided sampling returns a `[rows, hidden]` hidden-state prediction vector; by default the framework uses a bounded fused path (`guidance_capacity=4096`) that projects and reweights only the surviving top-K tokens, avoiding a separate full-vocabulary GEMM.
* **Avoid multi-layer `pllm.layer_logits`**: Every `layer_logits` entry triggers an independent full-vocabulary GEMM, re-reading the LM Head weights (1.09 GB on 7B). On 7B (32 concurrent requests), 9-layer DoLa drops to **66%** of native throughput; on smaller models, the penalty is even larger.
* **Filter by rows**: For non-global interventions, add a `rows=` filter to your effect method (e.g., `rows="trained"`) to shrink the number of active rows computed.

```python
@dataclass(frozen=True)
class FastGuidance(pllm.Workload):
    beta: float = 0.5

    def strength(self) -> float:
        return self.beta

    # Prefer @pllm.guide_sampling: returns [rows, hidden] and fuses deeply with the sampler
    @pllm.guide_sampling(beta=strength)
    def steer(self, h: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return h


guidance_report = pllm.explain(FastGuidance(), config)
```

#### 2) Use `placement="host"` with Clean Method Splitting for Huge Read-Only Tables

For read-only lookup tables ranging from hundreds of MiB to several GiB, declare `pllm.Pretrained(..., placement="host")` to keep them in pinned host memory. To hide the ~1.6 ms PCIe transfer behind the forward pass, follow this split pattern:

* **Dedicated lookup method**: The lookup method must read only `pllm.input_tokens` and read-only state (with no effect decorator), passing the gathered result to downstream effect methods via `pllm.result("lookup")`.
* **Why this split is mandatory**: Only an independent method depending purely on tokens and read-only tables can be scheduled by the planner at `request.prepare / side` at the start of the step, using a bounded number of SMs (capped at 32 thread blocks on RTX 4090, i.e. 1/4 of SMs) to prefetch on the side stream in parallel with the backbone forward. If the same method also reads `pllm.hidden` or `pllm.final_hidden`, it must wait until the model forward finishes and run at `state.publish / main`, exposing the full 1.6 ms transfer latency on the timeline.

```python
# check-docs: skip: demonstrates host lookup table pattern requiring pretrained weight file
@dataclass(frozen=True)
class FastHostLookup(pllm.Workload):
    buckets: int = 2048

    @property
    def memory(self) -> pllm.InstanceState:
        # One vocab-wide logits prior row per bucket; ~594 MiB on 7B (2048 x 152064 x bf16), placed in pinned host memory
        return pllm.InstanceState({"table": lambda m: (self.buckets, m.vocab_size)},
                                  init=pllm.Pretrained("ngram_table.safetensors", placement="host"))

    # Dedicated lookup method: reads only token history and the read-only table, no effect decorator;
    # the planner schedules it on request.prepare / side
    def lookup(self, ids: torch.Tensor = pllm.input_tokens(history=1),
               table: torch.Tensor = pllm.parameter("memory", "table")) -> torch.Tensor:
        keys = (ids[:, 0] * 31 + ids[:, 1] + 1) % self.buckets
        return table[keys]

    # Effect method consumes the gathered result via pllm.result
    @pllm.add_logits()
    def prior(self, logits: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return logits
```

### What It Costs

In RTX 4090 benchmarks (7B, 128 concurrency, 594 MiB table, `NGramPrior` from Example 11):
* Native vLLM: 5,877 tok/s (~21.8 ms/step);
* Device-resident table (`placement="device"`): 5,742 tok/s (98%), `explain` peak pLLM memory 1,540 MiB;
* Host-resident table (`placement="host"`): 5,615 tok/s (96%), peak pLLM memory drops to **946 MiB** (only ~0.5 ms slower per step than device placement, saving the entire 594 MiB of VRAM);
* Guided sampling capacity (`ESamp`, $\beta=0.5$): default `guidance_capacity=4096` achieves 5,678 tok/s (96.3%), with 256 and 1024 within noise of it; exact dense guidance (`guidance_capacity=None`) drops to 5,533 tok/s (93.9%).

<details>
<summary>Self-check: Why did a host lookup add 1–2 ms per step after also reading final_hidden in the same method?</summary> <!-- check-docs: not measured -->
Because the same method read <code>final_hidden</code>, it had to wait until the entire model forward completed (anchored at <code>state.publish / main</code>) instead of launching at <code>request.prepare / side</code> at the start of the step. As a result, the ~1.6 ms PCIe transfer was completely exposed outside the forward pass. The fix is to isolate the lookup—which depends only on <code>input_tokens</code>—into an undecorated helper method and pass its output to the effect method via <code>pllm.result</code>.
</details>

---

## 5. Optimize KV Memory: Static Layouts vs. Dynamic Reads

### What to Look For

1. Check the `kv_capacity` report in `pllm.explain` (which shows per-layer `KV bytes/request` and total capacity).
2. Inspect `llm.diagnostics().kv_cache_blocks` and `kv_cache_max_concurrency` at runtime.

### What It Means

In vLLM, memory is allocated in order: model weights + CUDA Graph capture pools + workload state reservations $\rightarrow$ **whatever remains goes to the KV Cache**. A smaller KV Cache lowers the maximum number of concurrent requests.

The two mechanisms in `@pllm.custom_attention` behave fundamentally differently:
1. **Static geometric layouts (`keep="window"`, `sink=True`, `keep="none", source=k`)**: Map directly onto vLLM's physical block manager. Physical blocks that slide out of the window are genuinely freed back to the memory pool, **actually saving KV VRAM capacity**, while running through native FlashAttention-2 / FlashInfer kernels to genuinely cut read bandwidth.
2. **Dynamic sparse reads (`kv.attend(q, topk=...)`)**: Every historical token's KV must still remain in VRAM for per-step scoring and selection, **saving zero KV VRAM capacity**; it reduces only logical attention reads. If scoring and gathering are written in plain PyTorch ops, kernel launch and fragmentation overhead far outweigh the saved reads. Score-based physical KV eviction (such as H2O or SnapKV) is unsupported because vLLM's physical block tables require sequential append-only allocation.

### How to Fix It

When your goal is reducing VRAM footprint and raising maximum concurrency, use static layouts:
* **Local sliding window**: `@pllm.custom_attention(layers=tuple(range(0, 28, 2)), keep="window", window=256)` (7B has 28 layers);
* **Cross-layer KV sharing**: `@pllm.custom_attention(layers=tuple(range(1, 28, 2)), keep="none", source=1)`.

```python
@dataclass(frozen=True)
class LocalGlobalAttention(pllm.Workload):
    # Static sliding window: frees physical blocks sliding out of the window, genuinely saving KV capacity
    @pllm.custom_attention(layers=tuple(range(0, 28, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


attn_report = pllm.explain(LocalGlobalAttention(), config)
```

### What It Costs

In benchmarks from [Example 10](../examples/10_custom_attention.ipynb) (RTX 4090, 7B, `max_model_len=2048`, 128-token chunked prefill; KV caps from `explain`):
* **Per-request KV VRAM cap**: Full KV is 112.0 MiB; even-layer window 256 drops to **70.4 MiB**; odd-layer cross-layer sharing drops to **56.0 MiB** (doubling the number of requests that fit in the same KV pool).
* **Throughput**: At 128 concurrency and ~270 tokens per sequence, passthrough reads, even-layer windows, and cross-layer sharing all run at 99% of native vLLM (setup in Example 10).
* **Dynamic sparse read comparison**: `QuestRead` in the [Custom Attention Tutorial](../examples/10_custom_attention.ipynb) (pure PyTorch) still consumes 112.0 MiB of KV VRAM and throughput plummets to **13%** of native (649 tok/s), because unfused PyTorch scoring and gathering cost far more than the saved reads.

---

## 6. Control Precision Finely: Balance Bandwidth and Numerical Stability

### What to Look For

1. Check the resolved `dtype` for each field on `state` lines in `pllm.explain` (e.g., `field 0: bfloat16, field 1: float32`).
2. Check `llm.diagnostics().numeric_health` at runtime.

### What It Means

During bandwidth-bound autoregressive decode, state tensor precision directly sets memory traffic. BF16 halves tensor size compared to FP32, halving both read/write bandwidth and state-pool VRAM usage. However, across multi-step autoregressive recurrences, using low precision for sensitive accumulators introduces severe rounding drift, causing divergence or NaNs.

### How to Fix It

1. **Keep `dtype="auto"` by default**: For read-only weights, single-step activations, or state fields not accumulated across steps, `dtype="auto"` aligns with the base model's dtype (BF16/FP16), cutting state bandwidth by 50%.
2. **Explicitly specify `torch.float32`** for:
   * Cross-step accumulators and optimizer first/second moments;
   * Recurrent carried state fields (`carry=`);
   * Coefficients and state transition matrices in linear prefix scans (`pllm.scan`);
   * Syntax: `pllm.Field(shape_fn, dtype=torch.float32)`.

```python
@dataclass(frozen=True)
class PrecisionAwareWorkload(pllm.Workload):
    state = pllm.RequestState(
        {
            "momentum": pllm.Field((896,), dtype=torch.float32),  # explicit FP32: avoid cumulative rounding error
            "cached_feature": pllm.Field((896,)),  # default auto: follows base model BF16
        }
    )

    @pllm.export(name="momentum_norm", max_rows=8, max_cols=1)
    def report(self, m: torch.Tensor = pllm.parameter("state", "momentum")) -> torch.Tensor:
        return m.norm(dim=-1, keepdim=True)


precision_report = pllm.explain(PrecisionAwareWorkload(), config)
```

### What It Costs

Explicit FP32 doubles physical VRAM and memory traffic for that field, but on small state fields this is the necessary cost for numerical stability and avoiding divergence on long sequences.

---

## 7. When Operators Become the Bottleneck: Plug In Custom Kernels

### What to Look For

1. Check the `custom_ops` report in `pllm.explain` (which shows operator names, `flops_per_row`, and CPU reference implementations).
2. Inspect profiler traces to see if the timeline is packed with tiny fragmented PyTorch kernels.

### What It Means

Writing multi-step tensor logic in plain PyTorch inside model layers (`@pllm.custom_residual`) or attention (`@pllm.custom_attention`) incurs kernel launch overhead and repeated VRAM round-trips on every call. In the [Custom Attention Tutorial](../examples/10_custom_attention.ipynb), Quest's per-page scoring and gather written in PyTorch runs at only **13%** of native throughput on 7B.

### How to Fix It

Implement compute-intensive or memory-fragmented logic as fused Triton or CUDA kernels registered through PyTorch's standard mechanism:
1. Register a named operator with `@torch.library.custom_op` and provide static shape inference via `register_fake`.
2. In your workload methods, **call the operator function object directly** (e.g., `fused_step(x, v)`) rather than `torch.ops.<namespace>.<op_name>`. Direct calls let the planner statically analyze the source and capture the operator straight into the CUDA Graph. See [Writing Custom Operators](../guides/custom-ops.md) for detailed authoring and verification steps.

```python
# check-docs: skip: demonstrates Triton custom_op pattern
@torch.library.custom_op("faster_guide::fused_step", mutates_args=(), device_types="cuda")
def fused_step(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return x + v


@fused_step.register_fake
def _fused_step_fake(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

### What It Costs

In RTX 4090 benchmarks:
* Example 09 attaches a Triton kernel registered as a `custom_op` at layer 12 and achieves **98%** of native vLLM throughput ($5,760$ vs. $5,872\text{ tok/s}$);
* The equivalent PyTorch math also achieves **98%** ($5,755\text{ tok/s}$): a single elementwise add is not a bottleneck in the first place;
* Writing a custom op requires authoring and maintaining a Triton/CUDA kernel and fake shape function. It pays off on steps that are genuinely slow in PyTorch—like Example 10's PyTorch Quest page scoring, which drops to 13% of native.

---

## 8. Rewrite Residuals Only on the Layers That Need It

### What to Look For

The `forward` line in `pllm.explain` and the number of layers carrying `@pllm.custom_residual` or `ActivationPatch`.

### What It Means

In a model graph carrying residual rewrites, vLLM captures the model's un-fused original operators rather than Inductor-fused operators, so even an identity rewrite pays that cost (~3% on 7B at 128 concurrency; see [Performance and Memory, Section 4.4](../explain/performance.md#44-in-layer-residual-rewrites)), and the rewrite's own compute accumulates per attached layer.

### How to Fix It

Use `layers=` to attach only to the layers that actually need rewriting. To measure the cost of the rewrite itself, compare against native vLLM in eager mode or run `testing.check_on_gpu` (which builds a matching native engine). Stateful sublayers also require a larger CUDA Graph capture pool—increase `ExecutionOptions(cuda_graph_pool_budget_bytes=...)` to the byte budget reported in the error message.

---

## 9. Make Layer Skipping Actually Save Time

### What to Look For

The skip rate of `layers(..., active=mask)` inside `@pllm.custom_forward(tokens="decode")`, and whether throughput scales with it.

### What It Means

Decode GEMMs are weight-bandwidth bound: as long as even one row in the batch runs a layer, that layer's entire weight matrix must be read once. Partial-row skipping saves only the attention reads for the skipped rows; you only save weight-read time when the entire batch skips a block of layers together (for the mechanism and measurements, see [Performance and Memory, Section 4.5](../explain/performance.md#45-in-forward-training-linear-scan-and-adaptive-depth)).

### How to Fix It

Design skip predicates so the whole batch skips together more often (for example, fixed layer blocks or step-level rather than row-level thresholds), or apply layer skipping in long-context regimes where attention reads dominate. Prompt tokens never skip layers; `active=` applies only to `tokens="decode"`.

---

## 10. Sampler: Enable FlashInfer Sampling

At batch sizes of 128+ rows or when a workload modifies logits, vLLM's default sampler can take noticeable time. When FlashInfer is installed, set the environment variable below before launch to switch to its fused sampling kernel:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

Because this changes sampling time on both the native baseline and the workload engine, always measure both under the identical setting and record it in your report (see [Measuring Throughput Fairly](measure-throughput.md)).

---

## Summary: Performance Optimization Cheat Sheet

| Check | Metric (`explain` or Runtime) | Target State | Optimization |
| :--- | :--- | :--- | :--- |
| **1. Graph Mode** | `forward` line / `eager_reason` | `CUDA graphs: no effect needs eager` / `""` | Move mutable shared state to `RequestState`; set `cuda_graphs=True` to catch regressions |
| **2. Host Syncs** | `host` line | `0 syncs in traced workload math` | Replace Python `if` with `torch.where`; remove `.item()`; observe asynchronously with `@pllm.export` |
| **3. Side-Stream Overlap** | `member.lane` / `memory` thresholds | `side` / budget above `overlap_threshold` | Keep `publish_lag=1`; raise `execution_peak_budget_bytes` to preserve overlap |
| **4. Bytes Moved** | `projection FLOP` / `sampler bytes` | Minimal full-vocab GEMMs / high lookup overlap | Replace multi-layer `layer_logits` with `@pllm.guide_sampling`; split host tables into a step-start helper |
| **5. KV Memory** | `kv_capacity` / `kv_cache_blocks` | Real VRAM capacity freed | Use static layouts (sliding window or cross-layer sharing); remember dynamic Top-K saves no KV capacity |
| **6. State Precision** | Per-field `dtype` on `state` lines | `auto` -> BF16 for read-only; explicit FP32 for sensitive state | Declare `torch.float32` on accumulators, momentum, and carry fields; keep `auto` elsewhere |
| **7. Kernel Fusion** | `custom_ops` report / small ops in profiler | Fused in-graph, throughput $\ge 95\%$ | Register fused Triton kernels via `@torch.library.custom_op` and call the op object directly |
| **8. Residual Rewrites** | `forward` line / attached layer count | Only required layers attached | Restrict with `layers=`; compare against eager native or `check_on_gpu` to isolate rewrite cost |
| **9. Layer Skipping** | Skip rate vs. throughput | Whole batch skips together | Decide skipping by layer block or step; use `active=` only on decode |
| **10. Sampler** | Sampler time | Fused sampling | Set `VLLM_USE_FLASHINFER_SAMPLER=1` identically on both arms |
