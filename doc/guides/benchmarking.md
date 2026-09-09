# Benchmarking

The maintained benchmark scenarios call `pllm.benchmark.compare`, which constructs
the public model handle and uses its normal generation, diagnostics and close
operations. Scenarios supply algorithms and data, not arena, scheduler, capture,
worker or profiler implementations.

```bash
.venv/bin/python -m tools.benchmark_esamp --batch 8 --tokens 256 --rounds 3
.venv/bin/python -m tools.benchmark_lora --batch 8 --tokens 256 --rounds 3
```

The shared CLI accepts `--model`, `--batch`, `--tokens`, `--rounds`, `--warmup`,
`--gpu-memory-utilization`, `--capture-memory-mib`, `--state-memory-mib`, optional
`--execution-memory-mib` and `--profile DIRECTORY`. Defaults are Qwen2.5-0.5B, batch 8, 256 tokens,
three rounds, one warmup and 0.25 GPU memory utilization. Inputs use seed 7,
temperature 0.8, top-p 0.95, ignored EOS, disabled prefix caching and `n=1`.
Capture has a default 256 MiB resident limit per task, passed unchanged to the public
`ExecutionOptions`. Larger shapes may need a larger explicit limit; an underfunded
capture fails instead of changing shape, disabling capture or selecting a fallback.
Benchmark state capacity is fixed at 256 MiB by default and enters the original
signed plan through `ExecutionOptions.state_memory_bytes`. It must fit the backend's
observed available memory; Rust rejects insufficient state capacity. This is an arena
budget, not total model/KV/scratch memory. Normal execution defaults to the observed
capacity (`state_memory_bytes=0`), whose changes legitimately change plan signatures;
fixed benchmark capacity makes exact timing/profile reconstruction reproducible.
`--execution-memory-mib` passes a finite `ExecutionOptions.memory_budget_bytes` to the
original AER timeline; omitted means no additional limit, zero means zero capacity.
It includes all declared state/resident/reserved/workspace/live-result costs, but does
not bound undeclared model/KV, allocator or library memory. Missing task/output costs
or insufficient capacity reject planning, without reducing work or disabling capture.
Reports include `planned_peak_memory_bytes`, separately from measured CUDA peaks;
conservative planning bounds and actual allocator measurements are not interchangeable.

The public model-workload entry does not require training or capture:

```python
from pllm.benchmark import BenchmarkOptions, compare

results = compare(my_workload, BenchmarkOptions(batch=4, tokens=64, rounds=1))
```

This measures the ordinary implementation. `capture=True` adds a captured arm;
`explicit_math=True` adds the explicit tensor arm. Unsupported requests fail at the
normal binding boundary. `require_training=True` requires positive loss, update and
trained-row counts; leave it off for observation or activation-only algorithms.
All runs still check token counts, cleanup, fallback and equal effective work.
This entry measures model generation; typed host-service timing uses the shared
execution trace described in [validation](validation.md), not a synthetic GPU loop.

The internal ESamp and LoRA commands explicitly enable both optimized arms and the
training requirement. ESamp compares the same request-local algorithm with ordinary
tensor, explicit tensor, tensor capture and paged eager/capture implementations.
Output LoRA compares the same five execution choices separately for shared and independent
heads. Both scenarios default to rank 4; LoRA defaults to two training rows per state and two SGD iterations.
LoRA additionally accepts `--rank`, `--training-rows`, `--training-fraction`,
`--scope instance|request|both` and `--state-groups N` (default: one request state per
request). The active row count is rounded up and reported; remaining weights are zero.
For request scope, groups share prompt/sample keys and identical prompt text;
training input has one batch per state group, not per request. This models a synchronous
cohort admitted and finished together with ignored EOS, not independently timed joins
or partial retirement of a shared key. Instance scope remains a separate algorithm.
Synthetic training
tensors are created after model initialization using the public model dimensions.
The same input is submitted through the normal typed auxiliary-input boundary.
Generation, page selection, publication and sampler guidance are inside the measured
path. Shared and independent algorithms have different reductions and are not
advertised as equivalent workloads.

Cold model/compilation/capture setup is reported separately. Warmup is excluded.
JSON reports software versions, benchmark settings, signed plan identity, measured
round times, actual output tokens, loss/update counts, trained rows and peak
allocated memory. `round_process_cpu_seconds` measures process user+system CPU time
around each complete generation and final device synchronization, using `process_time`.
This includes Python, native submission and other threads in that process, excludes
sleeping/descheduled time and child processes, and can exceed wall time. It is not
CUDA API duration or a claim that all CPU work is exclusively launch submission.
Profile replay reports its own `profile_round_process_cpu_seconds`; those instrumented
times are not substituted for steady measurements. All internal training-comparison arms must produce the requested token count, perform
nonzero training, finish with no pending request pins/releases or fallback, and
match effective work. Peak memory includes persistent state and capture pools.
`baseline_allocated_bytes` is measured after warmup; `peak_extra_allocated_bytes`
is the peak above that baseline, including any resident growth, not a proven
temporary-only bound. LoRA prints its additional algorithm settings before each comparison.
Results also expose the original `state_usage` snapshot after generation: arena
reservation, logical/physical payload, padding, page ownership and pending receipts.
This is end-of-round state, not a sampled peak or an independent allocation ledger.

`--profile` first completes throughput timing for **all** arms. Only then does it
reconstruct each arm, repeat the same warmup, and profile exactly the same measurement
rounds. No extra generation is extrapolated to stand in for earlier rounds; updates
may legitimately vary between rounds. The reconstructed plan signature and aggregate
effective work must match the original measurement. `profile_rounds` records the
number of instrumented rounds; profile costs and work are totals across those rounds.
CUPTI instrumentation can affect subsequent CUDA API
latency after its context exits, so do not start another timed comparison in that
process. LoRA's `--scope both --profile ...` uses one fresh process per scope.
It exports a Kineto Chrome trace per arm, with memory events but no tensor snapshots,
and reports device kernel count, summed kernel microseconds, summed CUDA runtime/driver
API microseconds and actual GPU memcpy bytes. API time includes synchronization;
it is not pure CPU submission time. Overlapping kernels are summed, not reported as
GPU wall time. `cuda_launch_api_count` counts outer host launch intervals per thread,
merging nested runtime/driver calls; `cuda_graph_launch_api_count` is its graph-launch
subset. Neither counts the device kernels inside each replay. Memcpy bytes exclude
memory accesses inside kernels. These are full
generation costs, including vLLM, not isolated workload-task measurements. Profiling
never contributes to the steady-state counters or timings and must pass the same
effective-work/cleanup checks. Existing trace files are not overwritten. For example:

```bash
.venv/bin/python -m tools.benchmark_lora --batch 2 --tokens 4 --rounds 1 \
  --training-rows 4 --training-fraction 0.25 --scope request --profile /tmp/lora-profile
```

Independent inputs use `pllm.WeightedCEBatch.for_prompts(hidden, logits, labels,
weights, prompt_indices)`. This constructs the original auxiliary object and its
prompt/sample keys, without copying tensors. `prompt_indices` must match those
passed to `generate`; shared input still uses `pllm.WeightedCEBatch(...)` without keys.
The scenario does not import execution frames or construct request identity objects.
For example, `--batch 8 --scope request --state-groups 2` shares two states among eight
requests. `compare(..., prompt_indices=...)` passes those same keys to every public
generation, including warmup and profiling; settings record them for reproduction.

Use `alternatives=(("label", workload, captured), ...)` to add explicit implementations
to the same comparison. Paged arms now use device-side grouped kernels without matrix
packing or host offsets synchronization. ESamp's captured loss reads the original pool;
its captured Adam now updates all five original fields directly in two launches, with
two scalar bias factors per state as scratch. LoRA's paged capture resets and updates
original private fields directly, retaining its seeded reset and per-adapter normalization.
Captured fields now share logical grouping, and matrix kernels consume original row
indices directly without separate operand/output gathers. Fully overwritten capture
inputs no longer receive a redundant fill; smaller inputs still fill the signed padding.

2026-09-09 acceptance run, default batch 8 / tokens 256 / rounds 3, 6144 output tokens per arm:

| Algorithm | Tensor | Explicit tensor | Tensor capture | Paged | Paged capture | Loss/Update/Rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ESamp | 865 | 994 | 1046 | 937 | 1040 | 6120/6120/6120 |
| Shared LoRA | 1463 | 1450 | 1445 | 1385 | 1394 | 6/3/6 |
| Request LoRA | 1353 | 1358 | 1359 | 1325 | 1323 | 48/24/48 |

Values are output tok/s; compare within a row only. LoRA trains once per generate
call, not every token. All arms passed equal-work, cleanup and zero-fallback checks.
Logs are `outputs/acceptance-5Uinna/benchmark-{45,46}.log`. This run follows the
selected-state gradient replacement; it uses no profiler and does not justify a
paged default. The GPU also held another user's allocation.

The current-source eight-configuration study is in
`outputs/study-current-AV1WUw/RESULTS.md`: 40 timed arms and 40 independently
reconstructed profiler arms, three rounds each. Request count, training rows/fraction,
state sharing, rank, instance/request and smaller memory headroom all passed equal-work,
signature and cleanup checks. Increasing public vLLM KV reservation raised warmed
allocation by about 3.5 GiB; it was not an OOM/exhaustion test. No stable paged throughput
advantage was established. Capture reduced extra allocated peaks but added resident
storage and memcpy; do not infer lower total memory. The GPU remained shared.
The report separates enqueue/wait API spans and GPU-active intervals; device kernel
execution count is not host launch-call count, and enqueue time is not full Python
submission time. These limitations remain explicit rather than inferred from counters.
The subsequent CPU/launch instrumentation check and attribution of existing traces are
in `outputs/cpu-launch-euODct/RESULTS.md`. It does not rerun or supersede the eight-case
throughput matrix: only reporting changed, not workload mathematics or execution.

The earlier seven-configuration bounded matrix is preserved in
`outputs/gpu-matrix-uP6yrj/RESULTS.md`, including its settings and limitations.
It predates that gradient replacement and is not a current full performance study.
Older interleaved timing/profiling runs are excluded from implementation rankings:
CUPTI can affect subsequent timing. Their raw logs remain available; obsolete
throughput tables are no longer maintained alongside the current run.

An independent numerical reference remains in the LoRA workload evidence. The
benchmark does not rewrite that reference, call it as an alternative runtime,
or claim arbitrary numerical equivalence from matching counters.

Historical aligned ESamp experiments used batch 8, `sampling_n=16`, at least
256 tokens, bfloat16, temperature 0.8, top-p 0.95 and max model length >=512.
The current public attachment supports `n=1`; its absolute throughput is not
comparable with those historical 4k+ tok/s figures. Small verification runs are
ratio/behavior checks, not evidence of a throughput regression or improvement.

Use the repository virtualenv and compare identical workloads. After abnormal
termination inspect GPU process ownership before retrying; never terminate
unrelated workers. Do not restore a retired runtime as a permanent baseline.
