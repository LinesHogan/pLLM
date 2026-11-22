# explain Report and Runtime Diagnostics

This page documents every line and field of the `pllm.explain(workload, config)` report and the `llm.diagnostics()` return value. To learn how to use them for troubleshooting, see [Inspecting Costs: pllm.explain](../guides/explain.md) and [Making Workloads Faster](../howto/faster.md).

```python
import torch

import pllm


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


report = pllm.explain(StreamNorm(), pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8))
print(report.eager_reason == "", report.workload_host_syncs, [export.rows for export in report.exports])
```

## Text Report from explain

Each line in `print(report)` starts with a leading token indicating what that line describes.

| Line Prefix | Content |
| --- | --- |
| `model` | Model class, hidden width, layer count, vocabulary size, and dtype from `config.json`. Read-only models include an extra `model read-only: ...` line explaining why |
| `parallel` | Tensor parallelism degree; all subsequent reservations are per GPU |
| `forward` | Whether the model forward runs inside a CUDA graph: `CUDA graphs: no effect needs eager`, or `eager: <reason>` |
| `base` | Estimated per-token dense projection and LM-head FLOPs for the base model, excluding attention over KV history |
| Each line after the `member` header | One computation segment of one of your methods: `method_name  effect  anchor / lane / graph  state_read (origin, distance)`. A single method may span multiple segments |
| Anchor | Where the segment runs within a step: `request.prepare`, `model.execute`, `state.publish`, `sampler.pre_draw`, `sampler.complete`, `token.commit`, `request.finish`, or `next.prepare` (see [Core Concepts](../explain/core-concepts.md)) |
| lane | `main`: main stream; `side`: side stream overlapped with the forward pass |
| graph | `captured`: captured into a CUDA graph; `eager`: executed op-by-op |
| `(origin, distance)` | State version read: `latest` for the newest version, `boundary` for the previous publish boundary, `entry` for the version at step start; distance is the number of steps behind latest |
| `model site or inactive method` | A declaration acting directly inside the model (`Adapter`, `ActivationPatch`) or an unscheduled helper method, with no anchor |
| `forward eager because <method>` | This declaration forces the model forward back to eager mode |
| `CUDA-graph pool reservation` | Private CUDA-graph memory pool reservation for this segment |
| `estimate / step (N rows)` | Estimated per-step matmul/projection FLOPs, state read/write bytes (including optimizer fields), and sampler bytes at `max_num_seqs` rows |
| `kernels` | Estimated kernels launched per step (before fusion); `graph` indicates graph replay |
| `tap` | Hidden positions tapped from the model, which tokens (`decode` / `prefill` / `all`), buffer bytes, and the methods sharing that tap |
| Each line after the `state` header | `state_name  scope  bytes/root  reserved bytes`, followed by each field's storage dtype and reserved pages (`x N pages`), initializer source (`source`), `placement`, host memory bytes, and PCIe bytes per row and per step. Page counts exceed concurrent requests because old and new versions coexist; AdamW step counts and moments are read only by the optimizer without retaining old versions, so they use fewer pages than parameter fields |
| `memory` | Peak VRAM required by pLLM itself (state pages, task workspaces, CUDA graph pools, etc., excluding weights and KV cache); the second line gives two thresholds: below the first, side-stream tasks fall back to main-stream inline execution; below the second, planning fails |
| `budget` | Exact error text when your configured budget cannot fit (identical to `make_llm`); the report still prints the unrestricted plan |
| `preempt` | Whether state is kept (`keep`) or reset (`reset`) on request preemption, and why recomputation is rejected when reset |
| `host` | Number of GPU-to-CPU synchronizations in your math; methods inside the model are marked `unverified` and must be confirmed on GPU with `check_on_gpu` |
| `export` | Rows, columns, bytes, and event count per step for each export; notes when rows exceeding `max_rows` are dropped |
| `timing` | Values and consequences of timing settings, tagged `(default)` or `(you set)` |
| `op` | Custom ops: call sites, per-row FLOPs (when `register_flop_formula` is present), per-row byte lower bound, and whether a CPU reference and autograd exist |
| `attention layer` | Per-layer read pattern and estimated per-token FLOPs and KV bytes for `@pllm.custom_attention` |
| `KV layer` | Maximum KV blocks and bytes per request per layer under static KV layouts |
| `note` | Things omitted from or unknown to this report (such as GPU BLAS workspaces, model weights, and KV cache) |

## ExplainResult Fields

`pllm.explain` returns an `ExplainResult`; `dataclasses.asdict(report)` can be passed directly to `json.dumps`. Commonly read fields:

| Field | Meaning |
| --- | --- |
| `eager_reason` | Why the model forward is not inside a CUDA graph; empty string if it is |
| `members` | Effect, per-segment anchor, lane, graph capture status, state reads, and estimates for each method |
| `states` | Scope, per-field page count (`pages`), bytes per root, `reserved_bytes`, field dtypes, initializer source, and placement for each state |
| `taps` | Tapped hidden locations and buffer bytes |
| `peak_memory_bytes`, `overlap_threshold_bytes`, `inline_peak_memory_bytes` | The three numbers printed on the `memory` lines |
| `budget_errors` | Errors when configured budgets are exceeded; empty tuple if everything fits |
| `preemption`, `recompute_rejection` | The `preempt` line |
| `workload_host_syncs` | Synchronization count; `None` if it cannot be verified on CPU |
| `exports` | Row count, column count, bytes, and event count for each export |
| `escape_hatches` | The `timing` lines: setting, value, whether default, and consequence |
| `custom_ops`, `attention`, `kv_capacity`, `kernels` | Corresponding `op`, `attention layer`, `KV layer`, and `kernels` lines |
| `limitations` | The `note` lines |
| `flops_per_token_estimate` | Estimated per-token FLOPs for base model plus all methods; `None` if base model is unknown |

## llm.diagnostics() Fields

Callable at any time while the engine is running; returns values at the moment of the query after pLLM waits for device counters to flush.

| Field | Meaning |
| --- | --- |
| `eager_reason` | Same as in `explain`; empty string when the model runs inside a CUDA graph |
| `loss_count` | Number of training method invocations |
| `update_count` | Number of optimizer updates actually published |
| `trained_rows` | Cumulative rows trained |
| `active_states` | Number of currently live state roots |
| `state_usage` | Snapshot of state VRAM usage; see table below |
| `numeric_health` | L2 norm, maximum absolute value, and non-finite count (`norm`, `max_abs`, `nonfinite_count`) for each live published field; `None` if norm or max is non-finite. Computed only on query without adding per-step syncs; no history is retained after a field retires |
| `kv_cache_blocks`, `kv_cache_max_concurrency` | Actual KV blocks allocated by vLLM and maximum concurrent requests sized by `max_model_len` (per GPU) |
| `hold_count`, `last_hold` | Count and most recent details of `ActivationPatch` with `on_missing="hold"` reusing stale inputs |
| `inference_pins`, `completed_sampler_pins`, `pending_state_releases` | Internal counters for framework debugging |

`state_usage` fields:

| Field | Meaning |
| --- | --- |
| `reserved_bytes` | VRAM reserved for state |
| `logical_live_bytes` | Sum of field bytes across all live versions; fields shared between old and new versions are counted multiple times |
| `allocated_payload_bytes` | Actual allocated field bytes; shared fields are counted once, so this can be much smaller than `logical_live_bytes` |
| `reserved_padding_bytes` | Bytes reserved for alignment padding |
| `free_field_pages`, `live_field_pages`, `private_field_pages`, `retiring_field_pages` | Field page counts: free, published, currently being written, and awaiting reclamation |
| `private_versions`, `retiring_versions`, `pinned_versions` | Version counts: currently being written, awaiting reclamation, and pinned by readers (plus a pending-device-completion publish count for framework debugging) |
| `admission_capacity_failures` | Number of requests rejected at admission because state capacity was exhausted |

## Export Events

Commonly read fields on `pllm.io.TraceEvent` delivered to `on_export`: `tensor_name` (export name; component prefixes are stripped for component exports), `tensor()` (detached CPU tensor), `identities` (one `pllm.testing.RequestIdentity` per row: `prompt_index`, `sample_index`, and pLLM internal `request_id`), `positions` (each row's token position in its request), `step_id`, and `dropped_rows` (rows dropped on this step due to `max_rows`). `llm.trace_snapshot()` returns a `pllm.io.TraceSnapshot` with retained `events`, `dropped` (oldest events evicted when the snapshot buffer filled), and `export_dropped` (events dropped when the export delivery queue filled).
