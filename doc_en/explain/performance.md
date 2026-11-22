# Performance & Memory: Where Costs Come From

An algorithm may be only a few lines of tensor math, yet drop straight into a decode loop and slow down throughput—or blow past VRAM limits. This page explains what takes time and memory during decode, and what each pLLM mechanism and coding pattern actually costs. For ways to speed up a workload, see [Making Workloads Faster](../howto/faster.md); for benchmarking methodology, see [Measuring Throughput Fairly](../howto/measure-throughput.md); for symptom diagnosis, see [Troubleshooting Errors](../howto/troubleshooting.md); and for reading `pllm.explain` reports, see [Inspecting Costs: `pllm.explain`](../guides/explain.md).

## 1. Mental Model: Three Physical Bottlenecks of Autoregressive Decode

During prefill or offline training, the GPU processes thousands of tokens at once and is typically compute-bound (FLOPs). During autoregressive decode, however, the model generates only **1 token** per concurrent request on each step. A GEMM's arithmetic intensity (multiply-adds per byte of weights read from VRAM) equals roughly the number of active batch rows. At small-to-medium batch sizes, **GPU compute sits mostly idle, leaving only three real bottlenecks**:

1. **Timeline bubbles**: Can the CPU feed the whole chain of kernels to the GPU in one shot each step? Any CPU wait on the GPU (host sync), or falling back to per-op Python launches (eager mode), leaves the GPU idle.
2. **Extra bytes moved per step (VRAM bandwidth & bus traffic)**: How many bytes of weights, request state, vocabulary-width logits, and host-memory lookups are read or written on every step?
3. **Headroom left for KV cache (VRAM capacity)**: Model weights, CUDA Graph private pools, and pLLM state reservations take fixed slices off the top; **only the remaining VRAM goes to vLLM's KV cache**. Reserving a few hundred extra MiB can sharply cut maximum concurrency.

### Reference Numbers Worth Remembering

Before evaluating your own workload, establish a baseline. Unless noted otherwise, numbers on this page are measured under the following setup: **single RTX 4090 (24 GiB), `Qwen/Qwen2.5-7B-Instruct` (BF16), vLLM 0.32.0, 128 concurrent sequences (8 prompts × 16 samples), 256 generated tokens each (`ignore_eos`), CUDA Graphs enabled, FlashInfer sampler; throughput measured via `pllm.benchmark.throughput` as the median of 3 runs after 1 warmup**. Percentages are relative to native vLLM (with CUDA Graphs) under the same setup; step latencies are converted as `128 rows ÷ throughput` (amortizing prefill). Measurements from other setups (0.5B comparisons, multi-GPU A100, the 4B In-Place TTT checkpoint, earlier profiling) are called out where they appear; exact configurations and dates for every number live in [Measurement Log](../reference/measurements.toml).

In this reference setup, native vLLM takes about **$21.7\text{ ms}$** per decode step (~**$5,894\text{ tok/s}$**). Switching only the model to 0.5B shrinks a step to ~$3.0\text{ ms}$ ($42,355\text{ tok/s}$). Many overheads behave very differently at those two scales:

| Operation or Phenomenon | Measured Latency / Throughput Cost | Physical Reason |
| :--- | :--- | :--- |
| **Eager fallback** (CUDA Graphs disabled) | **98%** of graph throughput on 7B ($5,773\text{ tok/s}$); drops to **49%** on 0.5B under the same setup ($20,926\text{ tok/s}$) | Launching ops from Python has fixed CPU overhead. On 7B, each step streams ~15 GB of weights, so GPU execution hides Python launch time; on a 3 ms 0.5B step, Python cannot feed the GPU fast enough |
| **One extra vocabulary projection** (LM Head GEMM) | ~**$+1.35\text{ ms}$ / step** (~6% of a step) | The 7B LM Head is $3584 \times 152064$ (BF16), weighing **$1.09\text{ GB}$**; latency is almost pure weight-read bandwidth (microbenchmark: $1349\ \mu\text{s}$ for 128 rows; 0.5B's $272\text{ MB}$ LM Head takes $359\ \mu\text{s}$, ~12% of a 0.5B step) |
| **Native sampler vocabulary sweep** | In 7B profiling (`ESamp` $\beta=0$, 128 rows): temperature ~**$125\ \mu\text{s}$**, top-k/top-p ~**$283\ \mu\text{s}$**, Gumbel draw ~**$78\ \mu\text{s}$** (~2% of a step combined) | A $128 \times 152064$ tensor is read and written repeatedly in VRAM; this cost scales with rows and vocab size rather than model size, so it weighs much more on a 3 ms 0.5B step |
| **Transferring one vocab-wide table over PCIe** (Host $\rightarrow$ GPU) | ~**$1.6\text{ ms}$ / step** physical transfer (~7% of a 7B step; over half of a 0.5B step) | Measured PCIe 4.0 x16 host-read bandwidth on a 4090 is ~**$23.7\text{ GB/s}$** (~1/40 of ~1000 GB/s VRAM bandwidth); 128 vocab-wide rows move **$38.9\text{ MB}$** per step |

Around these three bottlenecks, pLLM does three things under the hood:
1. **Keep the timeline unbroken**: zero host syncs per steady-state decode step, capturing forward edits and side-stream training into CUDA Graphs, while hiding per-step CPU prep (~**$700\ \mu\text{s}$** in 0.5B profiling, largely independent of model size) in the shadow of GPU execution.
2. **Save VRAM capacity**: paged state pools, role-based auto-precision, and CPU static tracing preserve VRAM for the KV cache.
3. **Move fewer bytes & overlap smartly**: never read weights twice when one read suffices; when data movement or training is unavoidable, overlap it on a side stream only when the required GPU hardware resource is idle.

---

## 2. Timeline: Synchronization, CUDA Graphs & Side-Stream Overlap

On a 4090, a 7B decode step takes ~$21.7\text{ ms}$; the same 128 rows on 0.5B take ~$3.0\text{ ms}$. Staying close to native throughput with a custom algorithm comes down to keeping the GPU from waiting on the CPU and keeping small custom ops from shattering CUDA Graphs. The shorter the step, the more both matter.

### 2.1 Why Host Syncs Are Fatal

Under asynchronous execution, the CPU prepares metadata one step ahead of the GPU. In a 0.5B, 128-concurrency profile (vLLM 0.27.1), pLLM spends ~**$700\ \mu\text{s}$** total on CPU prep per step and uploads slot indices, masks, and metadata in a single pinned `int64` table (H2D ~**$6.24\ \mu\text{s}$**), completely hidden behind the ~$3.3\text{ ms}$ GPU forward pass and leaving only ~**$4\text{–}6\ \mu\text{s}$** of inter-step GPU idle gap. This CPU work barely depends on model size and is even easier to hide behind a ~$21.7\text{ ms}$ 7B step.

Calling `.item()`, `.tolist()`, `.cpu()`, `print(tensor)`, or `if tensor.any():` inside a method forces the CPU to wait until every queued GPU kernel finishes, draining the pipeline—and such operations cannot be recorded inside CUDA Graphs or side streams at all. pLLM therefore stops them upfront: it traces methods with `meta` tensors during planning and raises an error immediately in both `pllm.explain` and `make_llm`. See [What to Write in Method Bodies](../guides/workload-anatomy.md#what-to-write-in-method-bodies) for how to rewrite branches with masks and `torch.where`.

**Mechanisms & verification**:
* pLLM guarantees zero blocking host syncs per steady-state decode step, verified by `check_on_gpu`. In the 128-concurrency 0.5B RTX 4090 profile, pLLM incurred **0** Event/Stream host waits and **0** `cudaDeviceSynchronize` idle time inside the decode window.
* When internal mechanics genuinely require dynamic branching (such as adaptive-depth bucketing or scan completion checks), pLLM uses device-side CUDA Graph conditional nodes (CUDA IF nodes, requiring a CUDA 12.4+ driver path) to branch directly on GPU without copying conditions to CPU.
* `explain` traces only regular workload methods. To verify in-layer and custom-attention code, run `pllm.testing.check_on_gpu(workload, config)`, which executes at least 8 steady-state decode steps under `torch.cuda.set_sync_debug_mode("error")` and fails on any implicit sync.

### 2.2 What Runs Inside CUDA Graphs

CUDA Graphs record a sequence of GPU kernel launches once and issue the entire sequence with one lightweight replay call per step. pLLM preserves graph mode at three levels:
1. **Whole-bundle capture for tasks and online training**: Online training (e.g., `@pllm.train`) involves dozens of small kernels—forward, loss, masked reduction, `backward()`, state writes, and paged `AdamW`. pLLM records them into one graph sized to max rows and state capacity; inputs use fixed buffers, and unused rows in partial batches point to dummy sink slots that cannot corrupt real state. Even as active requests change every step, **no graph re-recording occurs**: steady-state decode replays 1 training graph per step with 0 recaptures.
2. **In-layer residual edits inside the model graph**: `@pllm.custom_residual` lowers to a named op and is captured directly inside vLLM's model graph (see [Section 4.4](#44-in-layer-residual-rewrites)).
3. **Custom ops inside graphs**: Triton or PyTorch kernels registered via `torch.library.custom_op` with `register_fake` are captured into both graphs and side streams as long as you invoke the op object directly rather than `torch.ops.<namespace>` (see [Example 09](../examples/09_custom_kernels.ipynb)). In Example 09 on 7B at concurrency 128, attaching a custom Triton op at layer 12 achieves **98%** of native vLLM throughput ($5,760$ vs. $5,872\text{ tok/s}$), matching the PyTorch equivalent at **98%** ($5,755\text{ tok/s}$).

`ActivationPatch` executes inside CUDA Graphs via the same `custom_residual` mechanism. Only two constructs drop the model forward to eager (or replace the whole-model graph):
* `pllm.model.Adapter` bound to a shared mutable `InstanceState`: drops to eager.
* `@pllm.custom_forward`: replaces vLLM's whole-model graph with per-layer-slice graphs captured by pLLM (`eager_reason` notes `@pllm.custom_forward`).

### 2.3 Side-Stream Overlap & `publish_lag`: When Overlap Actually Speeds Things Up

When `pllm.explain` assigns a task (like `@pllm.train`) to lane `side`, it runs on a side CUDA stream. Keep one crucial intuition in mind: **a side stream does not give you a second GPU. SMs and VRAM bandwidth are shared on the same chip—side tasks run "for free" only when the main stream leaves the hardware resource they use underutilized; if the main stream already saturates bandwidth, squeezing a side task alongside it just splits the same pie.**

#### Why Does the Default `publish_lag=1` Help?

Requiring step $n$'s gradients to finish and become readable at the start of step $n+1$'s forward pass (`publish_lag=0`) forces every forward pass to wait for backward + AdamW to finish.

By default, pLLM uses `publish_lag=1`:
* Once step $n$'s tokens commit, after a brief copy fence, training launches asynchronously on the `side` stream;
* Step $n+1$'s forward pass reads the older version from step $n-1$ without waiting on step $n$'s training;
* Weights produced on step $n$ take effect on step $n+2$.

This lets side-stream training tuck into inter-step gaps, small kernels, and underutilized portions of the main stream. How much time this saves depends on training's share of a step. Measured with `ESamp` (rank 4):
* **7B (reference setup)**: with training only ($\beta=0$), `publish_lag=1` reaches **97.4%** of native vLLM throughput vs. **97.2%** for `publish_lag=0`; with training + sampling guidance ($\beta=0.1$), **96.4%** vs. **96.1%**. The difference lies near measurement noise: a few hundred microseconds of training is tiny relative to a 21.7 ms step, so serializing it adds very little.
* **0.5B (same setup, model changed only)**: with training only, `publish_lag=1` reaches **91.9%** vs. **89.8%** for `publish_lag=0`. On a 3 ms step, the cost of stalling for training becomes clearly visible.

Setting `ExecutionOptions(publish_lag=0)` moves the training task's lane in `explain` from `side` to `main`, updates the `timing` line to `publish_lag=0 (you set): updates at step n become readable at n + 1; updates serialize before the next forward`, and reduces parameter-field page counts (see [Section 3.1](#31-paged-state-pools-and-the-per-field-page-formula-pages)).

More overlap is not automatically better. With `publish_lag=1`, `ESamp` ($\beta=0$) reaches 97.4% of native on 7B; the remaining gap matches the ~$616\ \mu\text{s}$ of GPU time spent in side-stream training (~2.8% of a step) because training and the main stream share one GPU, not because of CPU scheduling. Forcing training to overlap into the bandwidth-saturated sampler sweep would just make both fight for VRAM bandwidth. Similarly, because pLLM already captures the whole training bundle in a CUDA Graph, wrapping workload methods in `@torch.compile` yields no measurable gain.

---

## 3. Memory: Paged State, Precision & Budgets

vLLM allocates GPU memory in strict order: **model weights + CUDA Graph capture pools + pLLM state & task workspaces $\rightarrow$ everything left goes to the KV cache**. Every few hundred MiB reserved upfront directly shrinks available KV cache and maximum concurrency, so pLLM makes every reserved byte visible and controllable in `explain`.

### 3.1 Paged State Pools and the Per-Field Page Formula (`pages`)

When you declare `pllm.RequestState` or `pllm.InstanceState`, pLLM manages VRAM through paged state pools—one pool per field, with each page holding one value of that field; `x N pages` on `explain`'s `state` line is the total pages reserved for that pool. Updates perform **field-level copy-on-write (COW)**: new pages are allocated only for fields modified by that update, while unmodified fields share pages with older versions.

A state with `optimizer=pllm.AdamW(...)` has four fields: flattened `parameters`, `step`, first moment `exp_avg`, and second moment `exp_avg_sq` (gradients are consumed directly during the step rather than stored as a state field). Training steps in `Distill` and `ESamp` rewrite all four fields, so each new version takes one new page per field. Why does the parameter field reserve more pages than `max_num_seqs`? Because each request slot may concurrently pin several versions, each serving a distinct role:

| Copies | When Needed |
| :--- | :--- |
| 1 | The version read on the current step: retained from step start through step commit while main and side streams read it. |
| $H$ | Older versions retained for lagging readers. Needed only when a method reads a lagging version, shown in `explain`'s `state reads` as `(boundary, H)` where $H$ equals `publish_lag` (e.g., while step $n$ training writes on `side`, step $n+1$'s `predict` reads an earlier version). The newest of these copies is the currently published version, so "latest" needs no extra slot. |
| 1 | The new version currently being written by this step's training; 0 if no method writes this state. |

For request-scoped fields read by other methods (parameters and carry fields), the page formula is $\text{pages} = (1 + H + W) \times \text{max\_num\_seqs}$, where $W = 1$ if any method writes the state and $0$ otherwise. When a request finishes, any older version it left behind for lagging readers shares this budget with the incoming request that reuses its slot (since the incoming request has no old versions yet). In-flight GPU reads need no extra pages either: after a page is freed, the next write waits on GPU for prior reads to finish rather than allocating another copy.

`AdamW`'s `step`, `exp_avg`, and `exp_avg_sq` fields are read only by the optimizer itself on the latest version being updated; older versions pinned by lagging readers or request admission do not retain these three fields. They therefore need only the latest version plus the version being written: $\text{pages} = 2 \times \text{max\_num\_seqs}$ (2 pages for instance state).

* **Writable `RequestState` with a lagging reader**: e.g., `Distill` from [Workload Anatomy](../guides/workload-anatomy.md) (7B, hidden 3584, `predict` reading `(boundary, 1)`): at `max_num_seqs=128` and default `publish_lag=1`, the parameter field reserves $3 \times 128 = \mathbf{384}$ pages, each optimizer field reserves $2 \times 128 = 256$ pages, one version is $387,136\text{ B}$, and total state reservation is $115,623,984\text{ B} \approx \mathbf{110\text{ MiB}}$. Setting `publish_lag=0` drops all four fields to 256 pages (~95 MiB).
* **Writable `RequestState` reading only latest**: e.g., `ESamp` ($\beta=0$, only `loss` reading `(latest, 0)`), exact in-forward updates (`@pllm.train_in_forward(visibility="exact")`), and adapters read in model forward: $H = 0$, so every field reserves $2 \times \text{max\_num\_seqs}$ pages regardless of `publish_lag`.
* **Read-only `RequestState`**: $\text{pages} = (1 + H) \times \text{max\_num\_seqs}$.
* **`InstanceState`**: A writable instance state counts as a single slot, giving $1 + H + 1$ pages for the parameter field (e.g., 3 pages for `OverRIDE` at `publish_lag=1`, 2 pages per optimizer field). When `pllm.model.Adapter` or `pllm.model.ActivationPatch` binds to it at request admission (each request holds its admission snapshot) and another task writes during execution—whether step-level writers (`@pllm.write`, `@pllm.train`) or online control-plane updates (`import_state()`, `train_batch()`, per-batch `StateImport`)—in-flight requests can hold distinct admission snapshots, so reader-visible fields reserve an extra `max_num_seqs` pages: $1 + H + 1 + \text{max\_num\_seqs}$ (e.g., the `Adapter` / `Imported` case in [Inspecting Costs: `pllm.explain`](../guides/explain.md) reserves $2 + 128 = 130$ pages per field at `max_num_seqs=128`). Immutable instance states (`Zeros` / `Computed` / `Pretrained` with no writers or imports) take **1 page**; states supplied by `pllm.Fed` take a fixed **3 pages** (one read by the engine, one queued, one written by the producer).

### 3.2 Auto Precision: When BF16 Works and When FP32 Is Required

State fields default to `dtype="auto"`. pLLM statically analyzes each field's read/write role in the workload to balance VRAM usage against numerical stability (reductions, gradient accumulation, and optimizer internals always stay $\ge$ FP32):

| Field Role in Methods | Precision Chosen by `dtype="auto"` | Reason & Measured Impact |
| :--- | :--- | :--- |
| **Optimizer-updated parameters**<br>or **read-modify-write accumulators / cross-chunk carry** | `torch.float32` | Prevents small updates from underflowing or errors from drifting over long sequences. In a 32k-token (chunk 64) recurrence test, a BF16 carry had relative error **$1.437 \times 10^{-1}$**, whereas FP32 gave **$3.692 \times 10^{-4}$** (three orders of magnitude lower) |
| **Read-only fields** (static steering vectors, frozen weights)<br>or **write-only fields** | Matches model precision (e.g., `torch.bfloat16`) | Cuts state memory and read/write bandwidth in half. In `pllm.explain` (7B, concurrency 128), auto precision drops per-request state for ReFT (`operator="add"`, 1 layer) from $14\text{ KiB}$ (FP32) to **$7\text{ KiB}$**, and FactorAdapter (rank 2, q/v projections) from $88\text{ KiB}$ to **$44\text{ KiB}$**. Forcing large states to FP32 doubles both state VRAM and per-step state traffic |

Override explicitly with `pllm.Field(..., dtype=torch.float32)` when needed; see [State](../guides/state.md).

### 3.3 Workspaces & Memory Budgets

At engine bind time, pLLM runs lifetime analysis over your actual method code (including autograd-saved tensors, optimizers, and scans) to compute exact peak workspace and graph-pool reservations (e.g., on 7B at concurrency 128, `ESamp` rank 4 reserves ~**$133\text{ MiB}$** for `predict` during sampling and ~**$306\text{ MiB}$** for side-stream training; default `OverRIDE` side-stream training reserves ~**$3.5\text{ GiB}$**).

Four knobs on `pllm.ExecutionOptions` bound these reservations:
1. **`execution_peak_budget_bytes` (two-tier graceful fallback)**: Covers state pages, task workspaces and live results, anchor buffers, and capture-pool reservations; excludes model weights and the KV pool. Side-stream overlap (`side` lane) keeps two steps' buffers alive simultaneously and thus has a higher peak. For `ESamp` (rank 4, $\beta=0.5$) on 7B at concurrency 128, the side-overlap plan peaks at ~**$1,775\text{ MiB}$** while the inline floor is ~**$1,525\text{ MiB}$**.
   * Setting `execution_peak_budget_bytes` between those two numbers **does not raise an error—the planner automatically falls back to inline main-stream execution** (moving the task's lane in `explain` from `side` to `main`), trading a little overlap throughput to fit in VRAM.
   * Dropping below the inline floor fails planning: `explain(...).budget_errors` reports the required bytes vs. budget and suggests lowering `max_num_batched_tokens`, raising `execution_peak_budget_bytes`, or moving large read-only tables to `placement="host"`.
2. **`state_page_budget_bytes`**: Caps total state reservation (sum of `pages × bytes/page` across all fields). Exceeding it writes the shortfall into `explain`'s `budget_errors` (suggesting a smaller `max_num_seqs` or larger budget) and raises `ValueError` in `make_llm`. Default `0` uses the backend's actual free VRAM.
3. **`cuda_graph_pool_budget_bytes`**: Per-pool budget for each task's or model's private CUDA Graph memory pool (not a sum across pools). Default `None` reserves per-task traced needs (the default model graph pool is **$64\text{ MiB}$**, and exact decode overlays account for shape workspaces separately). Larger models yield larger rewritten model graphs: on 7B, even a single-layer `@pllm.custom_residual` requires ~$84\text{ MiB}$ (exceeding the default); stateful `@pllm.custom_residual` sublayers (Example 07, concurrency 8) need ~$1.9\text{ GiB}$; and Example 12's `@pllm.custom_forward` (concurrency 32) needs ~$143\text{ MiB}$. This budget counts toward pLLM's peak memory—simply set it to the reported byte count with a small margin. **Insufficient capture budget always raises an error with the required bytes and never silently drops to eager.**
4. **`max_host_training_rows`**: Used only for host-batch training (`WeightedCEBatch`, `CausalTrainBatch` passed via `llm.generate(..., batch=...)`), bounding the maximum rows trained per state per step so training workspaces can be sized upfront. Defaults to `1` and raises an error telling you to raise it if a batch exceeds it. It does not restrict per-request-row `@pllm.train` during decode.

**Online training on quantized bases**: When training online atop GPTQ/Marlin quantized weights, pLLM performs transient dequantization: dequantized dense weights live only across a single backward pass and are freed immediately, adding **zero peak VRAM overhead** (14B GPTQ on A100 takes **$0.193\text{ s}$** per training step; QKV dequantization for 14B on 4090 takes only **$0.104\text{ ms}$**). Pre-caching dequantized weights is slightly faster per step ($0.176\text{ s}$) at the cost of $24.63\text{ GiB}$ of permanent VRAM.

---

## 4. Cost by Pattern

### 4.1 Sampling Stage: Additive Logits, Guidance & Per-Layer Logits

pLLM folds workload logit operations into the LM Head and sampling stage, collectively termed the **sampler epilogue**. Multiple `@pllm.add_logits` contributions are added in declaration order, each rounded to the input logits dtype; `rows=` affects only selected rows before vLLM's standard penalties, masks, temperature, and filtering run. On steps with active logit edits, the framework fuses those additions with the existing FP32 sampler-input copy into a single pass; steps that skip edits keep the LM Head dtype and avoid per-contribution dtype casts, separate adds, and full row-subset copies; `@pllm.guide_sampling` can compose afterward. The per-step path never reads device values to the host. Requests asking for raw logprobs still retain a separate unmodified distribution.

`explain` attributes the input read, FP32 writeback, and one kernel of this shared copy to the first decode addition, and counts only each subsequent addition's own contribution bytes; row selection is estimated at max full batch size. This is a logical traffic estimate, not a latency prediction.

#### Default Bounded Fused Guidance: Why Capacity Is 4096

`ExecutionOptions.guidance_capacity` defaults to **4096**. On an unquantized, unscaled, untransformed LM Head with `TP=1`, the framework projects only the tokens surviving vLLM's filtering and fuses guidance with sampling; other LM Heads automatically take the exact dense path. Setting `guidance_capacity=None` explicitly selects exact dense; `pllm.explain` reports the chosen path, any fallback reason, and corresponding FLOP/byte estimates.

**Capacity should keep typical rows out of the overflow path, because overflow rows require an extra top-K selection.** When surviving finite logits are $\le K$, the kernel compacts, projects, and samples directly with no sorting. Only above $K$ does it select the top-$K$ logits for guidance while leaving the tail unguided in the draw (ties broken by token ID). If upstream filtering was skipped for a full-vocabulary row and the top-1 probability reaches `top_p < 1`, a true single-token nucleus is used.

Under the reference setup (7B, concurrency 128, `ESamp` rank 4, `beta=0.5`), throughput across capacities is (one fresh launch per capacity, median of 3 runs):

| Guidance Capacity | Throughput (tok/s) | vs. Native vLLM |
|---|---:|---:|
| 256 | 5,684 | 96.4% |
| 1024 | 5,684 | 96.4% |
| **4096 (default)** | **5,678** | **96.3%** |
| `None` (exact dense) | 5,533 | 93.9% |

On 7B, differences between 256 and 4096 are within measurement noise, and all bounded paths outperform exact dense by ~2.6%. On a 3 ms 0.5B step the gap widens: in an earlier 0.5B run (same concurrency 128, `beta=0.5`, median of 7 runs), `K=256` was ~3% slower than `4096` and dense was ~6% slower.

Across four measured 0.5B corpora totaling 652,800 rows, the largest true nucleus was **2,785**, so `K=4096` kept the entire support on the fast path; policy error vs. dense guidance had p99.9 TV ~**$6\times10^{-14}$** and **100%** shared-Gumbel sample agreement. Those figures describe that measured workload rather than a universal guarantee for other models, filters, or betas; on overflow, the bounded path guides only the top-$K$, and peaked full-vocab rows follow the nucleus rule above. Set `guidance_capacity=None` when exact dense semantics are required.

The dense path still pays for an extra full-vocabulary LM Head GEMM and reweighting: in an earlier 0.5B RTX 4090 profile, enabling guidance added ~$401\ \mu\text{s}$ per step over `beta=0` (~$118\ \mu\text{s}$ projection, ~$187\ \mu\text{s}$ reweighting); 7B's LM Head is 4× larger, taking ~$1.35\text{ ms}$ for that projection (Section 1 microbenchmark). Those dense costs do not apply to the default bounded fused path, whose `explain` projection and traffic figures are capacity upper bounds—so a smaller upper bound should not be read directly as higher throughput.

Side-stream training costs also check out in profiling: on 7B, 128 rows, `ESamp` (rank 4, `beta=0`), side-stream GPU time is ~$616\ \mu\text{s}$/step (~2.8% of a step), of which paged `AdamW` takes ~**$191\ \mu\text{s}$**—reading parameters and first/second moments directly from source physical slots and writing to destination slots, avoiding whole-page copy-on-write moves across 5 optimizer state fields.

`pllm.layer_logits` (used by `DoLa` to read multiple intermediate layers' vocab distributions) runs a separate GEMM per layer, reading the LM Head weights ($1.09\text{ GB}$ on 7B) again for each layer. On 7B (32 concurrent requests = $8 \times n=4$), 9-layer `DoLa` runs at **66%** of native throughput (see [Example 06](../examples/06_reading_inside.ipynb)); `explain` estimates 1.7× base per-token compute on 7B vs. 3.5× on 0.5B, hitting smaller models harder. Conversely, larger models dilute sampling-guidance overhead: one full-vocab projection is ~12% of a 0.5B step ($359\ \mu\text{s}$ / $3.0\text{ ms}$) vs. ~6% of a 7B step ($1.35\text{ ms}$ / $21.7\text{ ms}$). Under our reference setup, `ESamp` ($\beta=0.5$) reaches **96.3%** of native vLLM throughput (**97.4%** at $\beta=0$).

### 4.2 Large External Tables & Vocabulary-Wide Lookups

Consider a 7B vocabulary-wide $2048 \times 152064$ (BF16) lookup table: the table itself occupies **$594\text{ MiB}$**. With 128 concurrent requests reading one row each ($304,128\text{ B}$/row), every step reads **$38.9\text{ MB}$**. pLLM applies two layers of optimization:

1. **Table in VRAM (`placement="device"`): whole-graph Inductor fusion**
   For non-differentiable lookup methods depending only on `pllm.input_tokens` and immutable state, pLLM compiles and fuses the gather, dtype cast, mask (`where`), and scaling into a **single table-read kernel** at bind time, eliminating round-trip VRAM writes/reads of the $38.9\text{ MB}$ intermediate tensor and launching it right at `request.prepare`. On 7B (`NGramPrior` from Example 11, 128 concurrent requests on 4090), device-side lookup reaches **98%** of native vLLM throughput (**$5,742\text{ tok/s}$**, ~$22.3\text{ ms}$/step). Fusion can alter floating-point rounding; disable it with `ExecutionOptions(allow_numerical_transforms=False)` if strict bitwise equivalence is needed.
2. **Table in host RAM (`placement="host"`): SM-capped side-stream UVA fetch**
   Moving that $594\text{ MiB}$ table to `placement="host"` drops pLLM's GPU peak reservation (`explain`, 7B, concurrency 128, `LookupPrior` above) from **$1,464\text{ MiB}$ to $870\text{ MiB}$**—saving the entire table (allowing a $3\text{ GiB}$ host table inside a $1\text{ GiB}$ GPU budget).
   However, pulling $38.9\text{ MB}$ over PCIe 4.0 x16 (~$23.7\text{ GB/s}$ measured on 4090) still takes **$1.6\text{ ms}$** physically. A naive GPU kernel reading host memory spawns tens of thousands of thread blocks and occupies all 128 SMs on a 4090—those blocks sit stalled on slow PCIe while blocking the main model forward from scheduling. In an early 0.5B profile during design, only $26.2\ \mu\text{s}$ of a $1,753.7\ \mu\text{s}$ table read overlapped with the main stream; in a standalone micro-experiment (4090, 37.1 MiB read alongside a 3.9 ms matmul "forward"), an uncapped fetch hid only $0.07\text{ ms}$.
   **What pLLM does**: saturating PCIe takes very few SMs. pLLM intercepts row reads from host tables and lowers them to a persistent Triton kernel that **caps thread blocks at one-quarter of the GPU's SMs** (32 blocks on a 4090) and launches at `request.prepare / side`. One-quarter of SMs wait on PCIe in the background while **3/4 of SMs remain free for the main model forward**: in the 0.5B profile, **99.14%** of the table read overlapped with main-stream kernels; in the standalone experiment, 32 blocks read 37.1 MiB in $1.64\text{ ms}$ ($23.7\text{ GB/s}$) and added only $0.12\text{ ms}$ when run concurrently with the forward pass.

| Configuration (RTX 4090, 7B, Concurrency 128, 594 MiB Table, `NGramPrior` from Ex. 11) | Throughput (tok/s) | vs. Native vLLM | Step Time | pLLM Peak VRAM (`explain`) |
| :--- | :--- | :--- | :--- | :--- |
| Native vLLM (no lookup) | $5,877$ | 100% | $21.8\text{ ms}$ | — |
| Lookup: `placement="device"` | $5,742$ | **98%** | $22.3\text{ ms}$ | $1,540\text{ MiB}$ |
| Lookup: `placement="host"` | $5,615$ | **96%** | $22.8\text{ ms}$ | **$946\text{ MiB}$** |

Even while streaming $38.9\text{ MB}$ across PCIe every step (~$1.6\text{ ms}$ physical time), **`placement="host"` is only ~$0.5\text{ ms}$/step slower than `placement="device"`** while saving the entire $594\text{ MiB}$ in VRAM. On 7B, lookup overhead itself is small (2%–4% of a 21.7 ms step); on a 3 ms small model, moving the same byte count takes a much larger fraction of the step.

Host lookups stay hidden only when scheduled on the side stream at step start: split the lookup into a dedicated helper reading only `pllm.input_tokens` and the read-only table, and pass its output via `pllm.result` to the effect method. Mixing `pllm.hidden` into the lookup method—or putting it inside a layer—exposes the transfer outside the forward pass (see [Making Workloads Faster](../howto/faster.md)). As a rule of thumb, if `transfer_bytes_per_step ÷ ~22 GB/s` is well below one forward pass, it can mostly hide. pLLM never moves tables to host automatically to fit a budget; you make that tradeoff explicitly.

### 4.3 Custom Attention: Saving Capacity vs. Saving Reads

When writing `@pllm.custom_attention`, the single biggest dividing line is whether your pattern maps onto vLLM's physical block manager and native FlashAttention-2 / FlashInfer kernels:

| Dimension | Static Geometric Layouts (`keep="window"` / `sink=True` / `keep="none", source=k`) | Dynamic Sparse Reads (`kv.attend(q, topk=...)`) |
| :--- | :--- | :--- |
| **Saves KV VRAM capacity?** | **Yes**: maps directly to vLLM's native `SlidingWindowSpec`, `SinkSpec`, and cross-layer sharing groups; physical blocks outside the window are genuinely reclaimed to the pool | **No**: all historical tokens' K/V must stay in VRAM to be scored and selected |
| **Saves read bandwidth?** | **Yes**: dispatches straight to native FA2 / FlashInfer kernels | **Saves logical read bytes** (in `explain`, Ex. 10's `QuestRead` reads at most $1,835,008\text{ B}$/layer/token on 7B vs. $4,194,304\text{ B}$ dense), but scoring and gathers run through PyTorch ops |
| **Per-request KV VRAM cap** (`explain`)<br>(7B, Ex. 10: `max_model_len=2048`, 128-token chunked prefill) | Full **$112.0\text{ MiB}$** $\rightarrow$ even-layer window 256 drops to **$70.4\text{ MiB}$**; sharing half of layers drops to **$56.0\text{ MiB}$** (on 0.5B: $24.0 \rightarrow 16.6 / 12.0\text{ MiB}$, raising concurrent 2048-token capacity from 358 to 518 / 716 requests). `sink=True` retains the full prompt so the upper bound is unchanged, with actual savings depending on generation length | Unchanged (still **$112.0\text{ MiB}$** on 7B) |
| **Measured throughput**<br>(Ex. 10: RTX 4090, 7B, concurrency 128, max 128 tokens/step) | Native read (`NativeRead`), even-layer window (`LocalGlobal`), and cross-layer sharing (`CrossLayerShared`) all hit **99%** of native | `QuestRead` in Ex. 10 reaches only **13%** of native ($649$ vs. $4,855\text{ tok/s}$), because PyTorch scoring and gather overhead outweighs saved reads |

Windows primarily change what history the model sees and reclaim out-of-window physical blocks; whether they also speed up execution depends on whether the kernel groups query heads sharing each KV head. `Qwen2.5-0.5B` has 14 query heads and 2 KV heads, and `Qwen2.5-7B` has 28 query heads and 4 KV heads (a 7:1 GQA group ratio in both). FA2's windowed path disables decode GQA packing—organizing the grid by query head instead of KV head and re-reading KV per query head—so a smaller window is not automatically faster. On Ampere/Ada, regular stored windows use vLLM's FlashInfer decode/prefill wrappers with `window_left`; full, prompt-sink, shared-layer, and temporary window- or head-changing reads take their respective paths; batch-invariant mode, unsupported head dimensions, or non-contiguous query head strides remain on FA2.

FlashInfer accumulates the softmax denominator in BF16-rounded probabilities and converts base-2 LSE to natural log; an FP32 return dtype does not imply bitwise parity with FA2. In an independent reference check of window reclamation, max absolute error vs. FP32 was ~0.0079 for output and ~0.0017 for LSE. On the same queries, both FlashInfer and FA2 differed from FP64 output by up to ~0.0076, while FlashInfer LSE matched an FP64 reference using that BF16 denominator within 0.00005. If your algorithm is sensitive to tiny LSE shifts, account for this rounding magnitude in quality validation.

Dynamic attention-score-based physical block eviction (SnapKV, H2O) is impossible here: vLLM tracks a single `num_computed_tokens` integer per request from which RoPE positions and physical slots derive linearly; block tables are append-only and cannot leave mid-sequence holes. Use static layouts to save VRAM, and `kv.attend(q, topk=...)` only to reduce logical reads. Because layouts alter model math, applying windows or cross-layer sharing to models not trained for them can visibly degrade quality—always validate on your task first.

### 4.4 In-Layer Residual Rewrites

`pllm.model.ActivationPatch` modifies layer outputs inside CUDA Graphs via the same `custom_residual` model-site machinery. `@pllm.custom_residual` lowers to a named `pllm::model_site` op reading and writing private state pages by device-side index, letting vLLM **capture your residual rewrites directly into the whole-model CUDA Graph**.

Measured throughput in Example 07 on RTX 4090 (7B, 256 generated tokens) as a percentage of native graph mode:
* **Value Residual** (rewriting values across 27 layers, concurrency 32): **99%**
* **Hyper-Connections ($n=4$, 56 sublayers, concurrency 32)**: **93%**
* **Stateful memory sublayer** (one $3584 \times 3584$ FP32 matrix per request, ~49 MiB, so concurrency 8; ~$1.9\text{ GiB}$ model graph pool): **75%** of native graph mode at the same concurrency 8.

In that 7B setup (~19 ms/step), native vLLM eager mode itself hits 99% of graph mode, masking how much graph capture saves; earlier on 0.5B, the same rewrites without graph capture dropped to 12%–34% of graph throughput.

When `custom_residual` is attached, vLLM captures raw ops rather than Inductor-fused ops, so even an identity rewrite (`x + y`) has a cost: on 7B at concurrency 128, attaching an identity rewrite at layer 12 alone yields **97%** of native graph throughput, and attaching it on every layer's MLP is still **97%**. That ~3% gap comes from losing Inductor fusion across the model graph rather than layer count; any compute added by your rewrite itself is extra, so still restrict hooks to the layers you actually modify (`layers=...`). To isolate the cost of your rewrite math alone, compare against native vLLM in eager mode, or use `check_on_gpu` (which builds a matching native engine).

### 4.5 In-Forward Training, Linear Scans & Adaptive Depth

**Linear recurrences (`pllm.scan`)**: Stepping a chunk recurrence in a Python loop leaves GPU SMs underutilized. When you express the recurrence with `pllm.scan`, pLLM's planner detects affine recurrences of the form $S_t = A_t S_{t-1} + B_t$ and—when workspace budgets allow and coefficients are FP32—rewrites them into a **Parallel Prefix Scan**:
* **Microbenchmark** (scan only, independent of model size): a single-head $64 \times 64$ state scan drops from $1.439\text{ ms}$ serial to **$0.219\text{ ms}$ (6.57×)**; 14-head $64 \times 64$ speeds up **1.85×**; full state transition drops from $5.879\text{ ms}$ to **$1.759\text{ ms}$ (3.34×)**.
* **Automatic fallback**: When large state dimensions (e.g., $d=896$) make tree scans unprofitable, shape guards keep the serial loop; CPU execution, partial scans, and differentiable calls also stay serial. You can set `ExecutionOptions(tuning="measured")` to benchmark implementations at bind time and pick the fastest, or `allow_numerical_transforms=False` to forbid parallel reassociation.

**Exact in-forward updates (`@pllm.train_in_forward(visibility="exact")`)**: By default, `@pllm.train_in_forward` uses `visibility="forward_pinned"` (all chunks in one forward pass read the same initial state). Algorithms such as In-Place TTT—where later chunks in the same forward pass must immediately see fast weights updated by earlier chunks—declare `visibility="exact"`:
* **Automatic lowering & measured throughput**: pLLM captures `visibility="exact"` decode inside CUDA Graphs (prefill runs eager) and wraps the rare chunk-boundary fast-weight update in a device-side conditional node (CUDA IF). By statically tracing the authored step graph, pLLM also applies three automatic lowerings: (1) sharing unchanged fast-weight pages across non-boundary decode steps with zero copy-on-write traffic while allocating private pages only for tail buffers; (2) borrowing read-only `ModelCheckpoint` invariants in place without per-step copies; and (3) lowering linear readouts of the form `inputs @ delta.to(inputs.dtype).T` into a paged streaming kernel that skips idle padding lanes. In Example 08 running a community 4B In-Place TTT checkpoint on RTX 4090 (vLLM 0.32.0, CUDA Graphs on, FP32 fast weights, 256 greedy tokens, median of 3 runs after 1 warmup): a single request hits **$73.02\text{ tok/s}$** at 4096 prompt tokens and **$47.97\text{ tok/s}$** at 16384 prompt tokens (vs. $54.79 / 34.89\text{ tok/s}$ for the official Hugging Face implementation under the exact same hardware and protocol). Across 4 concurrent requests, pLLM scales to **$171.56\text{ tok/s}$** (the official implementation hardcodes `bs=1`).

**Adaptive depth (`@pllm.custom_forward` + `active=`)**: Skipped layers perform state propagation: when a row has `active=False` over a span of layers, its residual stream passes through unchanged (equivalent to treating those layers as identity maps). Because later tokens attending at every layer still need K/V entries at that position, pLLM backfills K/V for the skipped layers by passing the row's stopped hidden state through each skipped layer's input norm, K/V projections, and RoPE and writing into the KV cache (computed once on the first skip, costing ~7% of a layer's compute; subsequent skips of the same layer preserve the written values). This is CALM's "state propagation": it yields the exact K/V that position would have produced had those layers acted as identity, confining approximation error to that skipped token's own deep state.

Why other schemes are not the default: copying the previous layer's K/V is wrong (different layers' projections live in different subspaces); leaving holes (skipping attention writes entirely, as in Mixture-of-Depths) works only on models trained for it and requires per-layer KV lengths incompatible with vLLM's KV manager; deferring true deep recomputation (as in FREE) makes token counts per request vary unpredictably across steps, conflicting with fixed-shape CUDA Graphs.

How much skipping actually saves:
* **Why skipping 50% of 128 tokens does not make those layers 2× faster**:
  Decode GEMMs are **weight-bandwidth bound** (a single layer of a 7B model weighs $466\text{ MB}$). Even if only 1 out of 128 rows has `active=True`, the GPU must still stream all several hundred MB of that layer's weights into SMs to compute that 1 row. Skipped rows save only their own attention KV reads (and on their first skip still run ~7% of a layer's norm/QKV/RoPE cost via native ops to keep KV contiguous). <!-- check-docs: not measured -->
* **When do you actually save weight-read bandwidth?**
  pLLM uses CUDA Graph conditional nodes to compact active rows into power-of-2 buckets. **Only when the entire batch skips a layer (0 active rows, landing in `bucket 0`) does the GPU skip the layer's kernels completely via the conditional node and genuinely avoid reading its weights.**
* **Measured numbers (Example 12: RTX 4090, 7B, 32 concurrent requests = 8 prompts × $n=4$, 256 tokens, layers 16–19 of 28 skippable)**:
  * Never skip: **99%** of native graph throughput (native eager is also 99% here: at ~19 ms/step on 7B, launch overhead is hidden)
  * ~23% tokens skipped (threshold 0.85): **98%** of native graph throughput
  * Entire batch skipped (landing in `bucket 0`): **112%** of native graph throughput, matching skipping weight reads for 4 of 28 layers.
* **Longer contexts and larger batches change the balance.** The numbers above describe short-context decode, where time is dominated by reading weights. At very long contexts (thousands to tens of thousands of tokens), each row reads its entire K/V history, which can outweigh reading one layer's weights; skipped rows bypass that layer's attention and save KV reads proportionally to skipped rows (by mechanics; not yet benchmarked). At large batches or compute-bound prefill, compacting active rows directly saves matrix-multiply FLOPs.

Why prompt tokens do not skip layers: their sole job is supplying K/V for every token that follows; approximating deep prompt K/V degrades the entire response and propagates through prefix caches and across $n$ samples (in the Ouro paper, approximate prefill K/V drops >10 GSM8K points while decode skipping costs almost nothing). Furthermore, prompts execute in parallel blocks where row compaction breaks contiguity and extra attention overhead largely cancels matmul savings. Prompts therefore run full depth by default.

---

## 5. Multi-GPU Tensor Parallelism (`TP > 1`)

When a model does not fit on one GPU or needs higher memory bandwidth, enable multi-GPU tensor parallelism via `ModelConfig(tensor_parallel_size=...)` (requires vLLM 0.32.0+).

Under tensor parallelism, vLLM performs a full-cluster barrier (Custom All-Reduce) after every attention and MLP layer. If either (1) pLLM launches its own cross-GPU collectives or (2) side-stream online training collides with vLLM's All-Reduce for memory bandwidth on one rank, all ranks stall together.

pLLM avoids this with two mechanisms:
1. **Zero pLLM collectives**: Hidden taps, router math, `PositionState`, and small request-level states are replicated identically on every TP rank, with each GPU independently executing deterministic logic (measured logits RMS difference vs. native vLLM is 0 across TP sizes); `@pllm.export` delivers results from rank 0 only.
2. **Stream priorities & staggered publishing**: The TP inference main stream gets high priority (`-1`), the side training stream gets normal priority (`0`), and automatic state updates write to private state before sampling and publish at token commit—driving **overlap between side-stream training and vLLM Custom All-Reduce down to $0\ \mu\text{s}$/step**.

**Measured performance (A100-SXM4-80GB, `Qwen/Qwen2.5-7B-Instruct`, `ESamp` $\beta=0$, concurrency 128)**:
* **TP = 1**: **95.84%** of native vLLM throughput
* **TP = 2**: **95.63%** of native vLLM throughput
* **TP = 4**: **97.60%** of native vLLM throughput

TP does not reduce per-GPU VRAM used by pLLM states: states are replicated per rank rather than sharded (`pllm.explain` notes `reservations are per rank`); TP saves VRAM on the base model weights. See [Support Matrix](../reference/support-matrix.md) for per-feature TP coverage.

---

## Self-Check

1. **Question**: Since `lane: side` runs on a background stream, can you put a heavy task that reads and writes hundreds of MB of VRAM per step onto `side` and get it "for free"?
   <details><summary>Click for answer</summary>
   No. During decode, the base model itself is VRAM-bandwidth bound. If a side task also performs heavy VRAM reads and writes, it contends with the main stream's projections or sampler for the same memory bandwidth and slows down the main stream. Overlap yields a net win only when the side task uses resources complementary to the main stream (e.g., a few SMs waiting on slow PCIe, or filling brief inter-step gaps).
   </details>

2. **Question**: You wrote a lookup method that reads `pllm.input_tokens()` to query a large `placement="host"` table, and in the same method also reads `pllm.final_hidden()` for gating. Why did adding the table suddenly add $1\text{–}2\text{ ms}$ per step? <!-- check-docs: not measured -->
   <details><summary>Click for answer</summary>
   Because the same method reads <code>pllm.final_hidden()</code>, it cannot launch until the model forward finishes (its anchor/lane in <code>explain</code> becomes <code>state.publish / main</code>) instead of launching at the start of the step on <code>request.prepare / side</code>. As a result, the ~1.6 ms PCIe transfer sits completely exposed outside the forward pass. Split the host lookup depending only on <code>input_tokens</code> into its own helper method and pass its output via <code>pllm.result</code> to the effect method reading <code>final_hidden</code>, allowing the planner to hoist the lookup to step start and overlap it with the forward pass.
   </details>

3. **Question**: On a 7B model, you implemented early exit with `@pllm.custom_forward(tokens="decode")` and observed 30% of tokens skipping the last 4 layers at concurrency 128, yet end-to-end throughput barely budged. Why?
   <details><summary>Click for answer</summary>
   Layer execution during decode is weight-bandwidth bound. As long as 70%—or even just 1—of the 128 concurrent tokens still need those last 4 layers, the GPU must stream all weights for all 4 layers from VRAM into SMs. Partial skipping saves only the KV cache read traffic for those 30% of tokens; you save weight-read bandwidth only when the entire batch skips those 4 layers (landing in <code>bucket 0</code>).
   </details>
