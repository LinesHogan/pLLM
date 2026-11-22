# Going to Production: Multiprocessing, Tensor Parallelism, and Quantized Base Models

After validating an algorithm in a notebook or local script, promoting a `pllm.Workload` to production serving typically involves three engineering requirements: isolating the inference engine into a separate subprocess (`multiprocessing="auto"` or `True`) for online traffic, enabling Tensor Parallelism across multiple GPUs to serve larger models, and running efficiently on quantized base models (FP8, AWQ, GPTQ).

This guide covers how to configure these three capabilities, their constraints, and the mechanisms behind them.

---

## Multiprocessing: EngineCore Isolation and Serialization Requirements

`ModelConfig` sets `multiprocessing="auto"` by default: during local single-GPU debugging with `pllm.make_llm`, the workload and engine run in the same Python process; when using `pllm.make_async_llm`, `pllm serve`, or `tensor_parallel_size > 1`, pLLM automatically isolates the execution engine into a separate EngineCore worker subprocess (you can also pass `multiprocessing=True` / `False` explicitly to override the default policy). When multiprocessing is enabled, the following serialization rules apply:

The driver serializes the workload object via `pickle` and sends it to the EngineCore worker.

```python
from dataclasses import dataclass

import torch
from torch import nn
import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class StreamMonitor(pllm.Workload):
    @pllm.export(name="norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


# Verify planning and scheduling contracts on CPU
report = pllm.explain(StreamMonitor(), CONFIG)
check_result = testing.check(StreamMonitor(), CONFIG)
```

To launch the engine in multiprocessing mode for production serving:

```python
# check-docs: skip: starts a GPU engine in multiprocessing mode
from vllm import SamplingParams

from my_project.workloads import StreamMonitor

mp_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    multiprocessing=True,
)

with pllm.make_llm(mp_config, workload=StreamMonitor()) as llm:
    outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=8))
```

In multiprocessing mode, workload code must satisfy the following constraints:

1. **Workload fields must be picklable (no locks, open files, or live handles)**  
   **Why**: Workloads are sent to the worker subprocess via `cloudpickle`: classes and functions defined in `__main__` or notebook cells (including lambdas and closures) are serialized by value, whereas those in importable modules are serialized by import path. Process-bound objects such as locks, open file handles, and sockets cannot be serialized and fail during encoding.
2. **The workload module must be importable inside the worker**  
   **Why**: For classes passed by import path, the worker re-imports the defining module, so that module must be on the worker's `PYTHONPATH`.
3. **The module must be import-safe**  
   **Why**: When the worker subprocess starts up, it imports your module at top level before initializing the CUDA runtime or GPU memory. Any top-level code that allocates GPU tensors or touches the CUDA context will crash the subprocess before device initialization completes.
4. **The worker plans the workload itself**  
   **Why**: After unpickling the workload, the worker performs planning and binding in its own process, and the driver only retrieves the resulting plan. Modules sent by import path are imported directly by the worker, so—just as in any Python multiprocessing program—modifying those modules between calling `make_llm` and worker startup is undefined behavior.
5. **Model objects cannot be stored as workload fields**  
   **Why**: Base model weights typically span gigabytes to tens of gigabytes and are already loaded and resident in GPU memory by the worker. Workload fields must not hold duplicate model weights. If your algorithm needs to invoke the base model inside a method (for example, causal language model training), use `pllm.causal_model("adapter")`, which directly reuses the base model already loaded in the worker.

<details>
<summary>Self-check: Why does storing a model object in a workload field fail in multiprocessing mode?</summary>

Because the base model occupies large amounts of VRAM and is bound to a specific CUDA device context, making it unsafe to transfer across processes via `pickle`. Moreover, the EngineCore worker already owns the loaded base model—duplicating it wastes memory and causes deserialization errors. Inside workload methods, use `pllm.causal_model` to access the already-loaded model.
</details>

---

## Multi-GPU Tensor Parallelism (TP > 1)

When a single GPU does not have enough VRAM to hold the full model weights, or when higher memory bandwidth is needed, enable multi-GPU Tensor Parallelism via `ModelConfig(tensor_parallel_size=2, ...)`.

Tensor parallelism requires vLLM 0.32.0 or later.

If you need long-running concurrent requests, token-by-token streaming, or an OpenAI-compatible HTTP endpoint, see the [Online Serving Guide](online-serving.md).

```python
# check-docs: skip: starts a multi-GPU engine with tensor parallelism
from vllm import SamplingParams

tp_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    tensor_parallel_size=2,
)

with pllm.make_llm(tp_config, workload=StreamMonitor()) as llm:
    outputs = llm.generate(["Tensor parallel run"], SamplingParams(max_tokens=8))
```

### How It Works: Zero Cross-GPU Collectives and Staggered Publishing

Under Tensor Parallelism, vLLM must execute a cross-GPU synchronization (Custom All-Reduce) after every attention and MLP layer. If pLLM introduced its own cross-GPU collective communication, or if side-stream training compute contended with vLLM's All-Reduce for interconnect bandwidth, every GPU would stall severely.

To prevent this, pLLM uses two mechanisms:
1. **Zero pLLM cross-GPU collectives**: Hidden-state reads (`pllm.hidden`, `pllm.residual`, `pllm.final_hidden`), sampling guidance (`@pllm.guide_sampling`), logits adjustments (`@pllm.add_logits`), and request-level states maintain an identical replica on each TP rank. Every GPU executes the same deterministic computation independently without exchanging data.
2. **Stream priorities and staggered publishing**: The TP inference main stream runs at high priority (`-1`) while the training side stream runs at normal priority (`0`). State updates are written before sampling and published at token commit time, avoiding bus contention between side-stream work and All-Reduce.

### Support Boundaries for TP > 1

| Feature Category | Behavior under TP > 1 | Details and Rationale |
| :--- | :--- | :--- |
| **Hidden states & sampling** | Fully supported | `pllm.hidden`, `pllm.residual`, `pllm.final_hidden`, `@pllm.guide_sampling`, and `@pllm.add_logits` execute deterministically on each rank's local replica. |
| **Request state & concurrent activation patches** | Fully supported | Request-state training such as ESamp and `ActivationPatch` run independently on each rank's local replica and versioned slot table with zero cross-GPU communication overhead. |
| **Custom forward & custom residual** | Supported | `@pllm.custom_forward` and `@pllm.custom_residual` without `pllm.value(...)` or `pllm.input_tokens` dependencies execute across each rank's local shards. |
| **Custom attention (local heads)** | Supported | `@pllm.custom_attention` recomputing directly over each rank's local KV-sharded heads (including DeepSeek V2/V3 latent heads); reading `pllm.attention(...)` facts or static `window` buffers requires `tensor_parallel_size=1`. |
| **Sharded adapter weights** | Supported (dense / low-rank) | Dense and low-rank deltas on `pllm.model.Adapter` (with `Zeros`, `Computed`, `Pretrained`, or `Imported` init) align with projection sharding: `q`/`k`/`v` and `gate`/`up` projections shard weights and factor B along output rows; `down` projections shard weights and factor A along input columns; the other factor is replicated on all ranks and can be updated on local shards via `@pllm.train_in_forward`. |
| **Exports (`@pllm.export`)** | Delivered by rank 0 only | Results are identical across all ranks, so export events are sent back to the CPU from rank 0 only to avoid redundant transfers. |
| **Features requiring TP=1** | Rejected at validation | `operator="per_head"` adapter deltas; adapter states with optimizers, side-stream writes, or `FromModule` / `ModelCheckpoint` / `Fed` init; whole-model training via `pllm.causal_model`; external streaming via `pllm.Fed`; `@pllm.custom_residual` depending on `pllm.value(...)` or `pllm.input_tokens`; `@pllm.custom_attention` using `pllm.attention(...)` or `window` buffers; and `pllm.testing.check_on_gpu` (must be verified at TP=1 first). |

> [!IMPORTANT]
> **TP does not reduce per-GPU VRAM used by request state**: `pllm.explain` notes `reservations are per rank` when reporting memory. Tensor parallelism shards base model weights across GPUs, but pLLM keeps a complete replica of request state on every GPU, so per-rank request-state VRAM does not decrease as you add GPUs.

<details>
<summary>Self-check: Why doesn't per-GPU request-state memory drop by half when enabling TP=2?</summary>

Because Tensor Parallelism shards the base model's projection weights, whereas pLLM maintains an independent replica of request state on each TP rank to guarantee zero cross-GPU communication and deterministic execution. As a result, per-GPU request-state memory requirements remain unchanged.
</details>

---

## Quantized Base Models: FP8, AWQ, and GPTQ

Production deployments commonly serve FP8, AWQ, or GPTQ quantized checkpoints to cut VRAM consumption and improve memory-bandwidth-bound decode throughput.

pLLM supports quantized base models as first-class citizens:
1. **Identical model admission rules**: Quantized checkpoints follow the same model-family admission rules as dense models (e.g., Qwen2/2.5, Llama, Qwen3; see the [Support Matrix](../reference/support-matrix.md)).
2. **LoRA-style deltas never dequantize the base weights**: Weight deltas applied through `pllm.model.Adapter` or `@pllm.train_in_forward` add high-precision (`BF16` or `FP32`) corrections `(x @ A.T) @ B.T` directly onto the output of the quantized GEMM kernel—just like vLLM's native LoRA over quantized models. Base weights stay quantized throughout and are never materialized as dense matrices in VRAM.
3. **On-demand dequantization for whole-model training**: Training through `pllm.causal_model` works similarly to QLoRA: the forward pass uses quantized kernels, while the backward pass temporarily dequantizes the participating weights to compute gradients, balancing VRAM savings with differentiability.
4. **Materializing dense weights**: If `@pllm.train_in_forward` declares a read of `pllm.operand("weight")`, pLLM materializes a dense floating-point copy of the weight in VRAM for that method to use.

---

## Engine Lifecycle and Resource Cleanup

In long-running service processes or batch pipelines, you may create and destroy multiple engine instances sequentially within a single Python process. Always manage the engine lifecycle with a context manager:

```python
# check-docs: skip: demonstrates engine lifecycle
from vllm import SamplingParams

config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)

with pllm.make_llm(config, workload=StreamMonitor()) as llm:
    llm.generate(["Prompt 1"], SamplingParams(max_tokens=8))

# Exiting the with block automatically calls llm.close()
```

Exiting the `with` block (or calling `llm.close()`) shuts down the engine and releases its compiled state and CUDA graph memory, allowing repeated engine creation and teardown within the same process.

### Environment Requirements for GPU Contract Checks

`pllm.testing.check_on_gpu` internally constructs live engines and verifies scheduling contracts including eight-step steady-state decoding, forced KV eviction, and recomputation. Because it requires full control over the underlying engine lifecycle, it must be invoked in a standalone process with:
- `multiprocessing=False`
- `max_num_seqs >= 2`
- `max_model_len >= 32`

```python
# check-docs: skip: check_on_gpu loads GPU engines
testing.check_on_gpu(
    StreamMonitor(),
    pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4, max_model_len=64),
)
```

See [Checking Correctness](../guides/checking.md) for details.

---

## Memory Knobs and Request Preemption

In production, single-GPU VRAM is partitioned into four components:
$$\text{Total VRAM} = \text{Base Model Weights} + \text{CUDA Graph Pool} + \text{pLLM State \& Execution Workspace} + \text{vLLM KV Cache Blocks}$$

Model weights, the CUDA graph pool, and pLLM's state reservations claim fixed shares first; all remaining VRAM goes to vLLM's paged KV cache.

### Core Memory Knobs

Configure memory bounds in `pllm.ModelConfig` and `pllm.ExecutionOptions`:

```python
# Tune execution budgets and scheduling limits
service_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=16,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=0.7,
    execution=pllm.ExecutionOptions(
        state_page_budget_bytes=0,
        cuda_graph_pool_budget_bytes=128 * 1024 * 1024,
    ),
)
service_plan = pllm.explain(StreamMonitor(), service_config)
```

- **`gpu_memory_utilization`** (default `0.5`): Fraction of GPU memory allocated to the inference engine. Raising it gives the KV cache more room, though you must guard against GPU OOM under high concurrency or large models.
- **`max_num_seqs`** (default `8`): Maximum number of concurrent sequences processed per step. This directly sets the number of state slots `RequestState` pre-allocates for concurrent requests.
- **`max_num_batched_tokens`** (default `512`): Maximum total tokens batched across concurrent requests in a single step (including both prefill and decode).
- **`state_page_budget_bytes`** (default `0`): Upper bound on device memory for request state pages; `0` uses the free capacity observed by the framework.
- **`execution_peak_budget_bytes`** (default `None`): Upper bound on combined peak memory across state pages, task workspaces, and intermediate outputs.
- **`cuda_graph_pool_budget_bytes`** (default `None`): Upper bound on private CUDA graph pools for tasks and the model. `None` sizes automatically: tensor tasks reserve according to traced needs and model graphs are managed by vLLM; explicit caps are still enforced.
- **`max_host_training_rows`** (default `1`): Maximum rows per state per step fed by host training batches (request-level decode row training is budgeted separately).

For full parameter details across all execution options, see [Configuration](../reference/config.md). Always check peak memory reservations on CPU with `pllm.explain` (see [Inspecting Cost: pllm.explain](../guides/explain.md)) before deploying.

### Request Preemption and Recomputation

When a surge in concurrent requests exhausts KV cache blocks, the vLLM scheduler preempts active requests, evicts their KV cache blocks, and recomputes their history once memory frees up.

When a preempted request is rescheduled, vLLM recomputes its history. What this means for your workload is shown in the `preempt` line of `pllm.explain`:

- `state keep`: request state is preserved across preemption and is unaffected by history recomputation.
- `state reset`: state resets to its initial value and is re-derived as history recomputes. When request state is keyed by `request_id` with no mutable instance state, external feeds, or imports, recomputation reproduces the original state and the engine recomputes seamlessly. Otherwise, the `preempt` line states why recomputation is unsafe, and if preemption actually occurs at runtime, the engine raises:
  `vLLM preempted a request and would recompute its history; ... To avoid preemption, give vLLM more KV cache (gpu_memory_utilization) or fewer concurrent sequences (max_num_seqs)`

**Mitigation**: Inspect the `preempt` line before going live. If your workload cannot recompute safely, raise `gpu_memory_utilization` to give the KV cache sufficient headroom or lower `max_num_seqs` so preemption is not triggered.

---

## Boundaries Between Forward Edits and Prefix Caching

vLLM supports Prefix Caching (`enable_prefix_caching=True`) to reuse existing KV cache blocks across prompts that share a common prefix, significantly reducing prefill latency on long contexts.

In `ModelConfig`, `enable_prefix_caching` defaults to `False`. Whenever a workload modifies model computation, pLLM determines whether prefix caching is safe based on whether state varies across requests or steps:

### Allowed vs. Mutually Exclusive Boundaries

- **Safe to enable**: Purely read-only or sampling-guidance workloads, stateless `@pllm.custom_residual`, and `Adapter` or `ActivationPatch(..., last_tokens=0)` reading only static immutable `InstanceState` (`Zeros` / `Computed` / `Pretrained`, without `Imported`/`Fed` or write tasks). In all of these cases, identical prompts produce identical KV tensors across all requests.
- **Must disable (rejected at startup admission)**: Model edits using `@pllm.custom_attention` (including reading `pllm.attention(...)` facts), `@pllm.train_in_forward`, `ActivationPatch(..., last_tokens > 0)`, or reading `RequestState` / mutable or externally written `InstanceState`. These patterns cause identical prefixes to produce different KV tensors across requests or state versions, so reusing cached prefixes would silently corrupt numerical semantics.

---

## Common Errors and Troubleshooting

For exact error messages, root causes, and fixes when patterns on this page are rejected, see [Engine settings, models and parallelism](../reference/errors.md#engine-settings-models-and-parallelism) and [Requests and run time](../reference/errors.md#requests-and-run-time) in the Error Catalog.
