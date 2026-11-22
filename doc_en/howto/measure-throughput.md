# Measuring Throughput Fairly

When designing inference-time algorithms for LLMs (such as activation steering, per-request LoRA, decode-time online training like ESamp, or lookup memory), we care not only about quality gains, but also about the exact throughput overhead relative to the native baseline. Yet LLM inference throughput is notoriously easy to skew with hidden artifacts: early EOS stopping shrinking total generated tokens, Python launch overhead masquerading as compute cost, cold-start warmup contamination, CUPTI profiler driver pollution, or mixing incompatible memory definitions.

This guide is written for algorithm researchers and engineers. It establishes an unbiased benchmarking standard, walks through the two benchmarking utilities provided by pLLM—the general-purpose `pllm.benchmark.throughput` and the strictly aligned `pllm.benchmark.compare`—covers the built-in CLI benchmark scripts and flags, and explains how to interpret throughput, memory, and profiling outputs correctly.

---

## 1. Seven Rules for Fair Comparison (and Why They Matter)

Answering honestly "how much does this algorithm slow down vLLM?" rests on two principles: **Equal Work** and **Identical Environment**. Every rule below is essential:

### 1.1 Identical Input Prompts and Concurrent Batch Size
- **Rule**: Control and experiment arms must use the exact same list of prompt strings and batch size (`batch`).
- **Why**: Different prompt lengths and tokenizations produce completely different prefill workloads; the number of concurrent sequences (`batch × sampling_n`) directly sets the row count of continuous-batching tensors at every decode step. If the two sides see different sequence lengths, prefill variance will mask the decode throughput difference.

### 1.2 Enforce `ignore_eos=True` and Identical Sampling Parameters
- **Rule**: Set `ignore_eos=True` in `SamplingParams`, and keep `n` (candidates per prompt), `temperature`, `top_p`, and `seed` identical.
- **Why**: Autoregressive generation time scales directly with the number of tokens generated. Without `ignore_eos=True`, a degraded or terser algorithm may hit EOS earlier; a request that stops after 50 tokens produces a meaningless "tokens per second" number and perversely makes a broken algorithm look "faster." `ignore_eos=True` forces every request to generate the full `tokens` length, guaranteeing strictly equal decode steps and output tokens across both arms. Matching sampling parameters also aligns sampler compute and state allocation.

### 1.3 Enable CUDA Graphs on Both Sides with Identical Engine Settings
- **Rule**: Experiment and native control arms must share the exact same `ModelConfig`: **both must enable CUDA Graphs** (or if testing eager mode, both must run with `--eager`), use the same `max_num_seqs` and `gpu_memory_utilization`, and keep **prefix caching disabled (`enable_prefix_caching=False`)**. Sampler settings (`VLLM_USE_FLASHINFER_SAMPLER`, see [Making Workloads Faster](faster.md#10-sampler-enable-flashinfer-sampling)) must also match and be stated in the report.
- **Why**:
  1. Native vLLM enables CUDA Graphs during decode by default to eliminate per-step Python kernel-launch overhead. Comparing an uncaptured algorithm against graph-enabled native vLLM mostly measures the huge CPU launch vs. GPU execution gap rather than the algorithm itself (during decode, CPU launch overhead alone can halve throughput; see [Performance and Memory](../explain/performance.md)).
  2. `gpu_memory_utilization` (defaulting to `0.25` in the benchmark suite) and `max_num_seqs` determine the VRAM left for KV Cache and state pools; differing headroom can alter scheduler batching behavior.
  3. Prefix caching reuses KV Cache across later rounds and skips prefill work, breaking compute consistency between rounds.

### 1.4 Exclude Warmup Rounds
- **Rule**: Run at least 1 complete untimed warmup round before timing (`warmup=1`).
- **Why**: The first round pays one-time initialization costs: lazy Python imports, JIT compilation, PyTorch memory pool initialization, and CUDA Graph capture. Those are cold-start setup costs and must never be amortized into steady-state decode throughput.

### 1.5 Measure Multiple Rounds and Take the Median
- **Rule**: After warmup, run at least 3 steady-state generation rounds (`rounds=3`) and take the median rate (`median(round_rates)`).
- **Why**: Any single timing run is vulnerable to OS thread scheduling, background GC, driver context switches, or transient clock jitter. Taking the median across rounds filters outliers and yields a reproducible steady-state metric.

### 1.6 Time the Full `generate` Call and Synchronize Explicitly
- **Rule**: The timed interval must wrap the entire `llm.generate` call (covering both prefill and all decode steps) and call `torch.cuda.synchronize()` before stopping the clock. Throughput is defined as total generated output tokens divided by total elapsed seconds.
- **Why**: GPU kernels are enqueued asynchronously onto CUDA streams. Omitting `torch.cuda.synchronize()` after `generate` measures CPU submission time rather than GPU completion time. Measuring end-to-end requests also reflects real serving behavior without cherry-picking slices.

### 1.7 Report Complete Environment Metadata Alongside Numbers
- **Rule**: Every throughput number must be accompanied by the test configuration: GPU model, exact model ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`), dtype (e.g., bfloat16), vLLM and PyTorch versions, `batch`, `sampling_n`, generated `tokens`, and CUDA Graph mode.
- **Why**: Absolute throughput numbers divorced from concurrency, token length, and hardware specs have no physical meaning and cannot be reproduced.

<details>
<summary>Self-check: Why shouldn't throughput be measured with early stopping (omitting ignore_eos)?</summary>
The number of autoregressive steps determines total work. Without <code>ignore_eos=True</code>, a poorer model or shorter answer exits early and emits fewer tokens; dividing "generated tokens / elapsed time" then fails to reflect true decode step rate and can make a degraded algorithm that truncates early appear artificially faster.
</details>

---

## 2. General-Purpose Benchmarking: `pllm.benchmark.throughput`

### When to Use It and How It Works

When you already have an initialized engine handle (with or without a `Workload` attached, or simply to measure native vLLM on a set of prompts), use the lightweight `pllm.benchmark.throughput` helper (see [API Reference](../reference/api.md#throughput-benchmarking)):

```text
pllm.benchmark.throughput(llm, prompts, sampling, *, warmup=1, rounds=3) -> float
```

Given an engine handle, a list of prompts, and `SamplingParams`, it:
1. Runs `warmup` untimed rounds of `llm.generate` so CUDA Graph capture and memory allocation settle;
2. Calls `torch.cuda.synchronize()` to ensure an idle steady state;
3. Loops over `rounds` full `generate` calls, timing each with `perf_counter` and synchronizing the device at the end of every round;
4. Counts actual generated output tokens per round, computes `tokens / elapsed`, and returns the **median** rate across rounds.

### Code Example: Comparing Native vs. Workload Throughput

First, verify the configuration and benchmark options on CPU:

```python
from dataclasses import dataclass

import torch
from torch import nn
from vllm import SamplingParams

import pllm
from pllm.benchmark import BenchmarkOptions, BenchmarkResult, compare, throughput

# Inspect the engine config generated by default benchmark options
options = BenchmarkOptions(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    batch=8,
    tokens=256,
    rounds=3,
    warmup=1,
)
config = options.model_config()
assert config.max_num_seqs == 128  # 8 prompts x 16 candidates each
assert config.cuda_graphs is True
```

Then run native vLLM vs. a custom workload (such as the online distillation workload from [03 Decode-Time Training](../tutorials/03_decode_training.ipynb)) on GPU:

```python
# check-docs: skip: requires a CUDA engine and GPU model
# Define standardized concurrent prompts and aligned sampling parameters
prompts = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(
    n=16,
    max_tokens=256,
    temperature=0.8,
    top_p=0.95,
    ignore_eos=True,
    seed=7,
)

# 1. Measure native vLLM baseline rate
with pllm.make_llm(config) as plain_llm:
    plain_rate = throughput(plain_llm, prompts, sampling, warmup=1, rounds=3)

# 2. Measure rate with the workload attached
with pllm.make_llm(config, workload=my_workload) as guided_llm:
    guided_rate = throughput(guided_llm, prompts, sampling, warmup=1, rounds=3)

print(f"Plain vLLM rate     : {plain_rate:6.1f} token/s")
print(f"Workload rate       : {guided_rate:6.1f} token/s")
print(f"Relative throughput : {guided_rate / plain_rate:.1%}")
```

> [!TIP]
> `throughput` measures full end-to-end forward + generation speed. If your algorithm coordinates host-side training or external feedback loops, pair it with `llm.trace_snapshot()` to inspect host vs. device task timings; see [Host Loops and Batch Training](../guides/host-loops.md).

---

## 3. Standardized Aligned Benchmark: `pllm.benchmark.compare`

### Automated Equal-Work Comparison

When publishing results or reporting numbers, writing manual comparison loops easily leads to configuration drift. `pllm.benchmark.compare` encapsulates the standardized comparison protocol:

```text
pllm.benchmark.compare(
    workload,
    options,
    *,
    prepare=None,
    max_host_training_rows=1,
    require_training=False,
    prompt_indices=None,
    alternatives=(),
) -> tuple[BenchmarkResult, ...]
```

`compare` automatically performs:
1. **Native baseline measurement**: Launches native vLLM with the exact same prompts, sampling params, and CUDA Graph configuration, recording `vanilla_tokens_per_second` on each result;
2. **Arm measurement**: Builds and benchmarks the primary workload arm plus any variants passed in `alternatives`;
3. **Equal-work contract verification**: Asserts that total generated tokens for every arm strictly equal `batch × sampling_n × tokens × rounds`, and checks on teardown that no unfreed state copies or resources leaked;
4. **Structured reporting**: Prints standardized JSON output and returns a tuple of `BenchmarkResult`.

### Key Parameter Rule: When to Set `require_training`

Pay close attention to `require_training` when calling `compare`:

- **Set `True` only for online-training workloads**: When your algorithm includes online training (declares a `@pllm.train` method, such as ESamp or per-request LoRA), set `require_training=True`. The harness strictly asserts `min(report.loss_count, report.update_count, report.trained_rows) > 0`, ensuring backward passes and optimizer updates actually ran on GPU rather than being silently skipped.
- **Keep `False` (default) for non-training workloads**: For read-only observation, guided sampling (`@pllm.guide_sampling`), probe exports (`@pllm.export`), or residual rewrites, **do not** set `require_training=True`. Those workloads have zero training ops, so their training counts are 0 and the assertion will fail.

### Code Example: Running a Standard Comparison

```python
# check-docs: skip: requires a CUDA engine to run compare
options = BenchmarkOptions(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    batch=8,
    tokens=256,
    rounds=3,
    warmup=1,
    gpu_memory_utilization=0.25,
)

# Keep require_training=False (default) for non-training workloads
results = compare(my_workload, options, require_training=False)

for res in results:
    ratio = res.tokens_per_second / res.vanilla_tokens_per_second
    print(f"Arm [{res.implementation}]:")
    print(f"  Steady throughput : {res.tokens_per_second:.1f} token/s")
    print(f"  vs. native        : {res.vanilla_tokens_per_second:.1f} token/s ({ratio:.1%})")
    print(f"  Round rates       : {res.round_rates}")
    print(f"  Planned peak mem  : {res.planned_peak_memory_bytes / 1024**2:.1f} MiB")
    print(f"  Measured delta    : {(res.peak_allocated_bytes - res.baseline_allocated_bytes) / 1024**2:.1f} MiB")
```

---

## 4. Built-in CLI Scripts for Standard Scenarios

The pLLM repository provides several maintained benchmark scripts under `tools/`. They wrap `BenchmarkOptions.parse` so you can run standard comparisons directly from the CLI without glue code:

- **`tools/benchmark_esamp.py`**: Benchmarks ESamp (online training of a small distiller during decode + guided sampling). Defaults to rank 4, supports `--beta` for guidance strength, and sets `require_training=True`;
- **`tools/benchmark_lora.py`**: Benchmarks Output LoRA. Supports `--scope instance|request|both` (shared instance vs. private per-request), `--rank`, `--max-training-rows`, `--training-fraction`, and `--state-groups`;
- **`tools/benchmark_engram.py`**: Benchmarks Engram-style N-gram lookup memory, comparing host-resident (`--placement host`) vs. device-resident (`--placement device`) decode throughput overhead.

### Example CLI Commands

Run the built-in benchmarks directly in your terminal:

```bash
# Run the ESamp benchmark: batch 8, 256 tokens, 3 rounds
python -m tools.benchmark_esamp --batch 8 --tokens 256 --rounds 3

# Run the Output LoRA benchmark with per-request private state
python -m tools.benchmark_lora --batch 8 --tokens 256 --rounds 3 --scope request

# Benchmark 7B on a 24 GiB GPU: set base model memory fraction explicitly; capture & state pools auto-size
python -m tools.benchmark_esamp --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.80
```

By default, task capture pools are sized from the planner's requirements and state pages use the backend's available capacity, while native model graph pools remain managed by vLLM. Budget flags only need to be set when you want to cap resources explicitly, and setting a cap below what is required raises an error. Note that `explain` still reports task capture reservations excluding vLLM's own model graph and activation pools.

### Common CLI Flags

All built-in benchmark scripts share the same flag set:

| Flag | Type & Default | Meaning & Considerations |
| :--- | :--- | :--- |
| `--model` | `str`, default `"Qwen/Qwen2.5-0.5B-Instruct"` | Base model path or Hugging Face model ID. |
| `--batch` | `int`, default `8` | Number of concurrent prompts. |
| `--tokens` | `int`, default `256` | Target output tokens per sequence. |
| `--rounds` | `int`, default `3` | Number of timed steady-state rounds. |
| `--warmup` | `int`, default `1` | Number of excluded warmup rounds. |
| `--sampling-n` | `int`, default `16` | Independent candidate sequences per prompt. Total concurrent sequences = `batch × sampling_n` (default 128). |
| `--gpu-memory-utilization` | `float`, default `0.25` | vLLM memory allocation cap, leaving ample headroom for workload state and workspaces. |
| `--eager` | `bool` flag | Forces all workload arms and the native baseline to disable CUDA Graphs and run in eager mode. |
| `--cuda-graph-pool-budget-mib` | `int`, optional, default auto-sized | Capture pool cap per tensor task and model graph (MiB). Raises an explicit error if insufficient. |
| `--state-page-budget-mib` | `int`, default `0` (backend capacity) | Device-side state page capacity cap (MiB) for reproducible timing and replay. |
| `--execution-peak-budget-mib` | `int`, optional | Caps planned execution peak memory (including state, workspace, and temporaries, excluding model/KV). |
| `--guidance-capacity` | `int` or `None`, default `4096` | Applies guided sampling only to the top K surviving tokens; `None` runs dense full-vocabulary guidance. |
| `--profile` | `Path`, optional | Directory to export Chrome traces (`.json`). See Section 6. |

### Aligned Throughput Settings

The built-in scripts lock down a strict set of inference hyperparameters so no hidden variables taint comparisons:
- **Dtype**: `bfloat16`;
- **EOS handling**: `ignore_eos=True`;
- **Caching**: prefix caching forced off (`enable_prefix_caching=False`);
- **Sampling**: `temperature=0.8`, `top_p=0.95`, `top_k=-1`, `seed=7`;
- **Length cap**: `max_model_len = max(512, tokens + 64)`;
- **Batched token cap**: `max_num_batched_tokens = max(512, batch * sampling_n * 64)`.

Changing any of these defines a new benchmark shape and must not be conflated with the standard shape when reporting results.

---

## 5. Interpreting Benchmark Output: Throughput, Memory, and State Usage

After a benchmark finishes, it prints structured JSON lines to stdout. Interpreting each field accurately prevents misreading results:

### 5.1 Core Throughput Metrics

- **`tokens_per_second`**: Primary reported metric—the median throughput across rounds.
- **`round_rates`**: Tuple of measured rates for each round.
  - *Diagnostic tip*: Variance exceeding 5% across rounds usually indicates host swapping, VRAM thrashing, or competing processes on the GPU.
- **`vanilla_tokens_per_second`**: Median rate of native vLLM under identical hardware, prompts, and sampling settings.
- **`vs_vanilla`**: `tokens_per_second / vanilla_tokens_per_second`, showing the fraction of native throughput retained (e.g., 0.92 means 92% of native throughput, or 8% overhead). <!-- check-docs: not measured -->

### 5.2 Memory Metrics: Do Not Conflate Planned Upper Bounds with Allocator Peaks

Reports list two distinct sets of memory figures with completely different physical meanings:

1. **`planned_peak_memory_bytes` (Static Planned Peak)**:
   - A **conservative theoretical upper bound** derived on CPU during static planning (via [`pllm.explain`](../guides/explain.md)) from your workload declarations.
   - It accounts for all declared request state pages, tensor workspaces, capture pool reservations, and live intermediates; it **excludes** base model weights, KV Cache blocks, and unmanaged external library allocations.
2. **`peak_allocated_bytes` and `peak_extra_allocated_bytes` (Allocator Measured Peak)**:
   - Actual runtime high-water marks recorded by PyTorch's CUDA memory allocator.
   - `baseline_allocated_bytes` is the post-warmup baseline; `peak_extra_allocated_bytes = peak_allocated_bytes - baseline_allocated_bytes` represents the net extra VRAM peak introduced by the workload.

> [!CAUTION]
> **Never interchange conservative planned bounds with measured allocator peaks!**
> The planned bound is a static proof and safety quota, whereas allocator measurements depend on PyTorch's pooling, fragmentation, and alignment behavior. Never report a planned bound as "measured VRAM" in a paper or report, and never cite a single-run measured delta as a theoretical guarantee across all shapes.

### 5.3 State Usage and Training Counts

- **`state_usage`**: End-of-round physical snapshot of state pages showing physical allocation and utilization (reserved pages, logical bytes, physical bytes, padding, and page ownership). Useful for spotting severe state-memory fragmentation.
- **`loss_count`, `update_count`, `trained_rows`**: For online-training workloads, record the actual number of loss evaluations, optimizer steps, and cumulative trained rows executed during timed rounds. Verify these are positive and match mathematical expectations.

---

## 6. Profiling Bottlenecks: Why `--profile` Replays Separately After Timing

To inspect kernel bottlenecks with a profiler, pass `--profile DIRECTORY` to any CLI benchmark script. pLLM uses a **two-phase mechanism** for profiling:

1. **Phase 1: Clean Timing**: All arms run all rounds in a clean environment with no profiler attached, producing untainted baseline throughput numbers;
2. **Phase 2: Isolated Profiling Replay**: Engines are re-initialized and warmed up, then re-run with identical inputs and rounds under the PyTorch Profiler, writing a `<arm>.json` Chrome trace per arm into the specified directory.

### Why Can't We Profile While Timing?

The reason is **CUPTI driver contamination**:
- PyTorch Profiler relies on NVIDIA's CUPTI (CUDA Profiling Tools Interface) driver library.
- Even after the profiler context exits, CUPTI instrumentation can continue slowing down subsequent CUDA API calls in the same process.
- Running the profiler before timing skews throughput downward, and profiling during timed rounds mixes profiler overhead directly into the throughput numbers.

Therefore, pLLM always runs clean timing first and instrumented replay second. **In your own scripts, never run steady-state timing comparisons after running a profiler in the same Python process.**

Open the exported traces in Chrome (`chrome://tracing`) or Perfetto. The summary reports actual GPU active time (`gpu_activity_microseconds`), total kernel count (`kernel_count`), CUDA Graph launch API count (`cuda_graph_launch_api_count`), and memory copy bytes. Note that API durations in traces include host synchronization wait time rather than pure CPU submission time, and overlapping kernel durations are summed arithmetically rather than representing wall-clock time.

---

## 7. Common Errors and Troubleshooting

For exact error strings, causes, and fixes when a configuration or budget is rejected, see the [Memory budgets](../reference/errors.md#memory-budgets) section of the Error Catalog.

When `compare` raises an `AssertionError`:
- If `min(report.loss_count, report.update_count, report.trained_rows) > 0` fails, you passed `require_training=True` for a non-training workload—remove it;
- If the output token count check fails, `ignore_eos=True` was not set, causing some requests to stop early and breaking equal work across arms.

---

## Next Steps

- [Reading Costs: pllm.explain](../guides/explain.md): Derive peak memory and CUDA Graph capture status on CPU before touching a GPU;
- [Making Workloads Faster](faster.md): Optimize state layout, kernel fusion, and side-stream overlap to reduce overhead further;
- [Sharing Training Across Requests](share-training-across-requests.md): Explore throughput characteristics when sharing state by prompt or route key;
- [Performance and Memory](../explain/performance.md): Deep dive into CUDA Graph replay, memory planning, and continuous batching mechanics.
