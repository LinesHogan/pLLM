# Checking Correctness

After writing a workload—and before trusting any numbers it produces—verify that it computes what you intend under real serving schedules. This page covers the three layers of verification provided by pLLM.

---

## Why Correctness Checks Matter

In single-request PyTorch debugging, an algorithm with a wrong reduction axis often throws no error at all. For example, if you accidentally write feature normalization as `h - h.mean(dim=0)` instead of `h - h.mean(dim=-1)`, a batch of size 1 has length 1 along dimension 0, so the output matches and unit tests pass.

Under vLLM continuous batching, however, the runtime dynamically packs tokens from multiple concurrent requests into a single tensor batch:
- **Cross-row mixing**: Dimension 0 holds rows from different requests dynamically. Reducing along `dim=0` with `mean`, `sum`, `softmax`, or sorting rows leaks activations from one request into another and silently corrupts generations.
- **In-place mutation of read-only inputs**: Method input tensors and model parameters are read-only views into executor-shared buffers. An in-place write like `w.add_(1)` or `h.mul_(2)` corrupts data read by concurrent tasks or subsequent steps.
- **Reading the wrong state version**: Online training and sampling guidance have timing dependencies; `publish_lag` defines the step window in which state updates become visible to downstream readers. Reading an uncommitted or out-of-window state version breaks determinism.

To catch these subtle bugs before touching real GPU traffic, pLLM provides three complementary verification layers:
1. **CPU contract check (`pllm.testing.check`)**: Runs fixed three-request schedules in seconds on CPU using the real planner and executor to verify request isolation, row permutation, chunking, preemption replay, and version visibility.
2. **GPU engine check (`pllm.testing.check_on_gpu`)**: Runs 8 steps of steady decode on a real GPU engine, forces KV-cache eviction and recomputation, and verifies CUDA graph capture, zero host syncs, and custom-operator device equivalence.
3. **GPU distribution drift measurement (`pllm.testing.check_distribution`)**: Runs teacher forcing on base-model-generated tokens to measure mean and P99 full-vocabulary bidirectional KL divergence and TV distance, catching hidden collapses that break rare tokens.

---

## Layer 1: CPU Contract Checks (`pllm.testing.check`)

`pllm.testing.check` requires no GPU and loads no model weights. It reads only the model's `config.json` metadata, instantiates the real planner and executor on CPU, and drives three constructed requests through serving schedule variants—row permutation, perturbed inputs, mixed prefill/decode, prompt chunking, and preemption replay—checking that every request state copy produces strictly identical outputs.

### Quickstart and Zero-Strength Baseline (`neutral=`)

The highest-leverage sanity check is passing a zero-strength baseline instance via `neutral=` (for instance, setting guidance strength `beta` to `0.0`). If an intervention still alters model logits or declares an active sampler stage when dialed to zero, its state binding or control branch has a leak.

```python
from dataclasses import dataclass
import torch
from torch import nn
import pllm
from pllm import testing

@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    # Declare per-request state: each request owns an independent small network and AdamW optimizer copy
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        # Compute per-row loss independently, returning a 1-D tensor [rows] reduced by pLLM per state
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(self, shallow: torch.Tensor = pllm.hidden(0),
                f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow.float())

config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
result = testing.check(Distill(), config, neutral=Distill(beta=0.0))
print(result)
```

Output:

```text
status  contract / reason
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    publish_lag
PASS    preempted
PASS    neutral
```

### Full Contract Check Matrix

`pllm.testing.check` verifies each of the following contracts:

| Contract | Failure Condition |
| --- | --- |
| `compile` | The workload fails planner admission (e.g., missing effect decorator on a method, missing input marker on a parameter, optimizer/training scope mismatch). |
| `row permutation`, `request isolation` | A request's output depends on its row position in the batch or on another concurrent request's inputs (e.g., reducing along `dim=0` or mutating Python attributes on `self` across steps). |
| `mixed prefill/decode`, `chunking / tokens=` | Splitting a long prompt into chunks or co-scheduling prefill and decode rows in the same step changes the computed result for the same token. |
| `preempted` | After a request is preempted under memory pressure, replaying its token history fails to reproduce the exact previous state tensors. |
| `publish_lag` | A method reads an uncommitted state version or one outside the window permitted by `publish_lag`. |
| `neutral` | With `neutral=`, the workload still perturbs the base model: `@pllm.add_logits` returns non-zero deltas, or `@pllm.guide_sampling` fails to set its effective strength to 0. |

### Reading `PASS` and `SKIP`

Each line printed by `print(result)` reports the status of one contract:
- `PASS`: Verified deterministically across multiple CPU schedule variants.
- `SKIP`: Explicitly states why the contract was skipped on CPU.

**Why some contracts are `SKIP`**: Logic mounted on model sites (`pllm.model.ActivationPatch`, `pllm.model.Adapter`, layer hooks in `@pllm.custom_residual`, cross-layer execution in `@pllm.custom_forward`, etc.) requires real base-model weights and a live KV cache. CPU checks verify their operand math and state transitions without simulating full Transformer forward passes on CPU, so model-site application is explicitly reported as `SKIP: model-site execution: ...` and must be covered by Layer 2 GPU checks.

### Structured `CheckResult` and `ContractError`

- **`CheckResult`**: Immutable result returned on success, exposing `passed` (tuple of passed contract names) and `skipped` (tuple of skipped entries, each with `property` and `reason`). Convert it with `dataclasses.asdict(result)` into a plain Python dictionary for CI JSON logging and assertions.
- **`ContractError`**: Subclass of `AssertionError` raised immediately if any contract fails. Its `failures` tuple lists every discrepancy, pinpointing whether a user method mixed rows or whether an internal routing mismatch occurred.

### Catching Read-Only Mutations and Row Mixing

If user code mutates an input tensor in place (e.g., `w.add_(1)`) or reduces across the row dimension (e.g., `h.mean(dim=0)`), `pllm.testing.check` catches it immediately:

```python
class BadRowMix(pllm.Workload):
    @pllm.export(name="probe", every=1, max_rows=8, max_cols=32)
    def probe(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        # Bug: reducing along dim=0 mixes across different request rows
        return h - h.mean(dim=0, keepdim=True)

try:
    testing.check(BadRowMix(), config)
except testing.ContractError as err:
    print(f"Caught contract violation with {len(err.failures)} failure(s):")
    for failure in err.failures:
        print(" -", failure)
```

In this example, `testing.check` catches the violation and reports that `probe` mixes rows: `Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0)`.

---

## Layer 2: Real GPU Engine Checks (`pllm.testing.check_on_gpu`)

Passing CPU checks is necessary for correctness, but not sufficient. Actual model-site modifications (weight overlays, residual rewrites), CUDA-graph capture sequences, and physical paged-memory allocation can only be fully verified on GPU.

`pllm.testing.check_on_gpu` boots a real vLLM engine and runs the following checks:
1. **8 steps of steady decode**: Verifies 8 consecutive decode steps run cleanly inside CUDA graphs with zero GPU-to-CPU host syncs.
2. **Forced KV-cache eviction and recompute**: Exhausts KV memory to force native vLLM sequence preemption, eviction, and context recomputation, verifying that request state copies recover losslessly under extreme memory pressure.
3. **Bitwise neutral alignment (`neutral=`)**: Verifies that greedy decoding and seeded stochastic sampling produce bitwise-identical token IDs compared to the unmodified plain base model.
4. **Custom-operator device equivalence**: If the workload uses custom operators, replays the CPU test script inputs on CUDA and verifies outputs match the CPU reference within dtype tolerances.
5. **Causal batch update equivalence**: For causal batch training, permutes input sequences and weights and verifies that updated AdamW weights remain consistent.

### Environment and Parameter Requirements

Calling `pllm.testing.check_on_gpu` requires:
- **A dedicated fresh process**: Because vLLM initialization, GPU memory pool locking, and CUDA graph capture bind deeply to process state, run GPU checks in a clean dedicated subprocess rather than sharing a process with other large jobs.
- **Single-process, single-device config**: Set `multiprocessing=False` and keep tensor, pipeline, and data parallelism all at 1 (`tensor_parallel_size=1`, etc.). The test suite inspects the in-process engine directly and does not support multi-worker process orchestration.
- **Minimum capacity**: Set `max_num_seqs >= 2` and `max_model_len >= 32`. The fixed test script runs concurrent requests and triggers preemption replay, which will fail to load below these thresholds.

```python
# check-docs: skip: check_on_gpu must run the real engine in a dedicated GPU process
gpu_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    max_model_len=128,
    multiprocessing=False,
    gpu_memory_utilization=0.3,
)
gpu_result = testing.check_on_gpu(Distill(), gpu_config, neutral=Distill(beta=0.0))
print(gpu_result)
```

---

## Layer 3: Measuring Distribution Drift (`pllm.testing.check_distribution`)

Passing bitwise checks under `neutral=` proves your workload is clean when turned off—but when turned on (`beta > 0`), how do you know it is not silently wrecking the model's output distribution?

In many experiments, aggregate averages (such as mean cross-entropy or overall BLEU) look healthy while a handful of rare token positions suffer severe logit blowup. Averages hide those outliers, whereas P99 and max values expose dangerous tails immediately.

`pllm.testing.check_distribution` uses **teacher forcing** on GPU to measure full-vocabulary distribution shift:
1. **Generate reference sequences**: The plain base model generates a fixed number of continuation steps (default `tokens=16`) from built-in test prompts (two prompts by default).
2. **Teacher-forced aligned forward**: The workload runs forward over those exact tokens with state updates and sampling guidance enabled, keeping both engines on identical contexts and batch geometries.
3. **Full-vocabulary FP32 distance statistics**: Computes full-vocabulary `log_softmax` directly on device from untruncated raw logits for three metrics:
   - `KL(plain || workload)`: Forward KL divergence from the plain model to the workload (in nats).
   - `KL(workload || plain)`: Reverse KL divergence from the workload to the plain model.
   - `TV`: Full-vocabulary Total Variation distance (half the L1 probability distance).
4. **Summary stats and outlier pinpointing**: Returns a `DistanceSummary` per metric containing `mean`, nearest-rank `p99`, `max`, and the worst token positions (`worst`, identifying prompt index and token offset).

### Measurement vs. Threshold Assertion

- **Measurement-only mode**: Without thresholds, the function acts purely as an observational metric and returns `DistributionResult` without ever raising `ContractError`.
- **Threshold assertion mode**: Passing `max_mean_kl` and `max_p99_kl` enforces both thresholds on both KL directions. If either threshold is exceeded or NaNs appear, `result.passed` becomes `False` and `result.failures` records the violations while preserving all statistics for inspection.
- **What it measures and does not measure**: It measures full-vocabulary probability shift on the given test sequences; it does **not** measure downstream task quality or guarantee behavior on untested long contexts.

```python
# check-docs: skip: check_distribution loads the base model and sampler in a GPU process
dist_result = testing.check_distribution(
    Distill(),
    gpu_config,
    max_mean_kl=0.01,
    max_p99_kl=0.1,
)
print(dist_result)
# Distill with beta=0.5 pushes the distribution noticeably: on an RTX 4090, measured KL(workload || plain)
# has mean ~0.049 and P99 ~0.20, failing both thresholds
print(f"Passed: {dist_result.passed}")
```

---

## Reporting Verification Results

When sharing results in team reviews, papers, or evaluations, follow this reporting standard:

> [!IMPORTANT]
> **Always state explicitly which checks ran on CPU, which ran on GPU, and the exact Python and hardware environment.**
> 
> Never claim vaguely that "the workload passed all tests." Passing CPU contract checks proves scheduling and state isolation mathematically, but because CPU checks explicitly skip model-site weight overlays (`ActivationPatch`, `Adapter`, `custom_residual`) and physical KV-cache management, **passing on CPU never implies passing on GPU**.

Recommended phrasing templates:
- "Passed `pllm.testing.check` on CPU (with `neutral=` baseline; verified `compile`, `row permutation`, `request isolation`, `publish_lag`, and `preempted`)."
- "Passed `pllm.testing.check_on_gpu` on a single <GPU model> under <Python env> (`multiprocessing=False`): 8 steps of steady decode with zero host syncs, forced KV-cache eviction and recompute passed, and greedy decoding matched bitwise with `neutral=`."
- "Measured via `pllm.testing.check_distribution` on default prompts: full-vocabulary `KL(plain || workload)` mean <value> nats, P99 <value> nats."

---

## Model-Free Unit Testing: Low-Level Stepwise Helpers

When you want to unit-test complex tensor math inside a method without even reading a model's `config.json`, use the low-level compilation and single-step helpers in `pllm.testing`:

| Helper API | Description |
| --- | --- |
| `pllm.testing.backend_facts` | Constructs synthetic in-process backend facts (hidden size, attached hidden-layer taps, memory limits, etc.) in place of vLLM metadata. |
| `pllm.testing.hidden_binding(layer, layer_count)` | Creates an input tap binding corresponding to `pllm.hidden(layer)`. |
| `pllm.testing.final_hidden_binding(layer_count)` | Creates an input tap binding corresponding to `pllm.final_hidden()`. |
| `pllm.testing.compile_execution(workload, facts)` | Compiles the workload, allocates instance state, and returns an execution handle. |
| `pllm.testing.RequestIdentity(request_id, prompt_index, sample_index)` | Constructs request identity metadata for a single request. |
| `pllm.testing.step_input(identities, hidden=...)` | Manually assembles one step's packed input batch (one row per request). |
| `pllm.testing.run_step(execution, value)` | Drives the executor through all anchor tasks for that step and produces outputs. |

Complete CPU-only unit test example:

```python
class Centered(pllm.Workload):
    @pllm.export(name="centered", max_rows=2, max_cols=4)
    def center(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return h - h.mean(dim=-1, keepdim=True)


binding = testing.hidden_binding(0, layer_count=1)
execution = testing.compile_execution(Centered(), testing.backend_facts(4, (binding,), layer_count=1))
rows = (testing.RequestIdentity("a", 0, 0),)
execution.admit(rows)
h = torch.randn(1, 4)
testing.run_step(execution, testing.step_input(rows, {binding: h}))
exported = next(event.tensor() for event in execution.trace.snapshot().events if event.tensor_name == "centered")
torch.testing.assert_close(exported, h - h.mean(dim=-1, keepdim=True))
execution.retire(rows, wait=True)
```

---

## Common Errors

For exact error messages, root causes, and fixes when patterns on this page are rejected, see [Contract check errors](../reference/errors.md#contract-check-errors) in the Error Catalog. All other build-time errors are organized by topic in the [Error Catalog](../reference/errors.md).

---

## Self-Check

<details>
<summary>1. Why does a workload using <code>@pllm.custom_residual</code> still print <code>SKIP model-site execution</code> after passing <code>testing.check</code> on CPU?</summary>

Model-site modifications such as `@pllm.custom_residual`, `ActivationPatch`, and `Adapter` inject directly into the forward computation between Transformer layers, which requires real model weights, attention kernels, and GPU KV-cache memory. CPU contract checks use the real planner to verify operand math, state-update timing, and cross-row isolation without simulating full LLM forward passes on CPU, so actual in-layer application is marked `SKIP` and must be verified on a real GPU engine via `testing.check_on_gpu`.
</details>

<details>
<summary>2. Why does <code>testing.check_on_gpu</code> require <code>multiprocessing=False</code> and a dedicated fresh process?</summary>

When the base vLLM engine initializes, it reserves exclusive GPU memory pools and performs CUDA graph warmup captures. Sharing a process with prior jobs leaves fragmented allocations and CUDA context state that break graph capture. Additionally, `check_on_gpu` inspects the in-process engine's executor and tensor state directly, which cannot cross multiprocess worker pipes—hence it requires a dedicated single-process (`multiprocessing=False`) environment.
</details>

<details>
<summary>3. If we already have bitwise alignment checks with <code>neutral=</code>, why do we still need <code>check_distribution</code>?</summary>

`neutral=` verifies baseline equivalence when the intervention is turned completely off (strength 0), whereas `check_distribution` tests numerical health when the intervention is actively running (strength > 0). Many algorithms look fine on average yet suffer extreme logit spikes on a few rare tokens. By computing full-vocabulary bidirectional KL divergence and TV distance and reporting P99 and max outliers, `check_distribution` catches tail numerical collapses hidden by averages.
</details>

---

## Related Documentation

- [Seeing Costs Early: pllm.explain](explain.md): Statically inspect memory reservations, CUDA graph capture status, and compute overlap plans on CPU.
- [Workload Anatomy](workload-anatomy.md): Deep dive into the three declaration categories, method-body rules, and version visibility.
- [Custom Operators](custom-ops.md): Integrating Triton kernels and registering CPU reference implementations for contract checking.
- [Troubleshooting Errors](../howto/troubleshooting.md): Full error directory and debugging workflow.
- [API Reference](../reference/api.md#contract--equivalence-verification): Signatures and details for `pllm.testing` and all public APIs.
