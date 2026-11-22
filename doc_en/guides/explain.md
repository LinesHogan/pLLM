# Seeing Costs Early: pllm.explain

Written for readers who have already gone through Tutorial 03 (ESamp, per-request online training): with a single call to `pllm.explain(workload, config)`—without loading weights or touching a GPU—you can see in seconds whether your workload enters CUDA graphs, where each computation is scheduled, how much memory pLLM will reserve, and which setting to adjust if a budget is too tight.

## Why You Need It

The brute-force way to find out whether a workload falls back to eager mode or how much memory it uses is to spin up the full engine: load weights, allocate the KV cache, capture CUDA graphs, and then inspect `llm.diagnostics().eager_reason` or wait for an error.

`pllm.explain` runs the exact same declarations, admission rules, and planner on CPU to show you the plan up front. It only reads the model config (`config.json`) and traces tensor shapes on CPU—no model weights loaded, no CUDA initialized, and no GPU memory allocated. Using Tutorial 03's `MiniESamp` with a 7B config as an example, the first call takes ~0.2 s on CPU (including reading the config, excluding `import pllm`), and a second call in the same process takes ~0.1 s.

## A CUDA-Graph-Captured Workload: MiniESamp from Tutorial 03

To check whether your online-training workload fits inside CUDA graphs and how its per-step compute overlaps, call `pllm.explain` directly on `MiniESamp` from Tutorial 03:

```python
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import pllm

class Distiller(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.up(F.silu(self.down(x)))
        return F.layer_norm(y, y.shape[-1:])

@dataclass(frozen=True)
class MiniESamp(pllm.Workload):
    beta: float
    rank: int = 4

    @property
    def distiller(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: Distiller(m.hidden_size, self.rank)), optimizer=pllm.AdamW(1e-3)
        )

    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)

config = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128,
                          max_num_batched_tokens=8192)
print(pllm.explain(MiniESamp(beta=0.5), config))
```

Output (excerpted; `kernels`, `note`, and a few `timing` details omitted):

```text
model    Qwen2ForCausalLM, hidden 3584, 28 layers, vocab 152064, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  CUDA graphs: no effect needs eager
base     14140571648 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0); one AdamW step per state per engine step; prompt rows train once per prefill chunk
  CUDA-graph pool reservation: 240756768 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 18350080 matmul FLOP + 0 projection FLOP; state read/write 44040704/44040704 bytes, optimizer 88081408 bytes (included), sampler 0 bytes
predict  guide_sampling  model.execute / main / captured  distiller (boundary, 1)
  CUDA-graph pool reservation: 2150432 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 0 matmul FLOP + 0 projection FLOP; state read/write 0/0 bytes, optimizer 0 bytes (included), sampler 0 bytes
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
  CUDA-graph pool reservation: 98902048 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  estimate / step (128 rows): 7340032 matmul FLOP + 3758096384 projection FLOP; state read/write 14680064/0 bytes, optimizer 0 bytes (included), sampler 4080928768 bytes
tap      model.layers.0[0], decode: 1837056 bytes shared by loss, predict
tap      model.layers.27[0], decode: 1837056 bytes shared by loss
state  scope  bytes/root  reserved bytes [exact; state_page_budget_bytes]
distiller  request  344080  102764592  field 0: float32 x 384 pages, field 1: float32 x 256 pages, field 2: float32 x 256 pages, field 3: float32 x 256 pages; source: module init; placement: device; host bytes: 0; logical transfer bytes/row: 0; decode bytes/step: 0
memory   peak 1330848404 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1330848404 bytes tasks run inline; below 1168476308 bytes planning fails
preempt  state keep; replay rejection: not applicable to kept state
host     0 syncs in traced workload math
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

Here is how to read the plan line by line:

- **`forward`**: The entire model forward can run inside CUDA graphs; no effect forces eager execution.
- **`base`**: Estimated FLOPs per token for the base model's dense projections and LM head, excluding attention over KV history.
- **`member`**: Where each of your methods is scheduled. A single method may span multiple lines, one per execution stage (for example, `predict` appears twice):
  - `anchor` (when it runs): `token.commit` runs after tokens for the step are committed; `model.execute` runs during model forward; `state.publish` runs when new state versions are published.
  - `lane` (which CUDA stream): `side` is the side stream overlapping with the next forward; `main` is the main stream.
  - `graph`: `captured` means the task is captured into a CUDA graph.
  - `state reads (origin, distance)`: Which state it reads and how many steps behind the latest version that read sits. `loss` reads the latest version (distance 0), while `predict` reads a version 1 step behind (which is why Tutorial 03 notes that "updates from step N become readable at step N+2", enabling overlap). So for `MiniESamp`, `loss` trains on the side stream inside a CUDA graph, and `predict` runs on the main stream inside a CUDA graph.
  - `CUDA-graph pool reservation`: Private GPU memory pool reserved to capture this task's CUDA graph. Because this comes out of the memory available to the KV cache, it is called out explicitly.
  - `estimate / step`: Estimated per-step FLOPs (sized at `max_num_seqs` rows), state read/write bytes, and sampler bytes. `predict`'s `projection FLOP` and `sampler` bytes come from projecting predictions through the LM head to guide sampling.
- **`tap`**: Hidden activations extracted from the model (layer 0 and layer 27) and their reserved buffer sizes (1,837,056 bytes each); layer 0's buffer is shared between `loss` and `predict`.
- **`state`**: `distiller` is a request-scoped state with a page pool per field, where `x N pages` shows the pages reserved for that field. One version across all fields totals 344,080 bytes, with 102,764,592 bytes (~98 MiB) reserved overall; `[exact]` indicates an exact calculation. Field 0 holds the parameters and reserves 384 = 3 × 128 pages: each concurrent request needs to retain the version read in the current step, the older version read one step behind by `predict`, and the new version currently being written by training. Fields 1–3 hold AdamW's step count and first/second moments, which only training reads—old versions never need to keep them—so each request only retains the latest version and the new version being written, totaling 256 pages each (see [Performance & Memory, Section 3.1](../explain/performance.md#31-paged-state-pools-and-per-field-page-counts-pages)).
- **`memory`**: Peak memory funded by pLLM under `execution_peak_budget_bytes` (including state pages, task workspaces and results, graph pools, and anchor reservations): 1,330,848,404 bytes (~1,269 MiB). The second line explains the thresholds: below 1,330,848,404 bytes, side-stream tasks fall back to running inline on the main stream (no longer overlapping); below 1,168,476,308 bytes, planning fails.
- **`host`**: No GPU-to-CPU synchronizations appear in the traced workload math (`0 syncs`).
- **`timing`**: Timing settings and their consequences; `(default)` means you kept the default value.
- **`note`**: Things this report cannot know statically (see the limitations section below).

## A Sliced-Graph Workload: Conditional Early Exit (`custom_forward`)

`Adapter` (whether bound to `RequestState` or a shared `InstanceState`) and `ActivationPatch` stay inside vLLM's whole-model CUDA graph. What actually splits the whole-model CUDA graph into per-slice wave graphs (and explains why in `eager_reason`) is `@pllm.custom_forward`. Running `pllm.explain` before starting the engine shows the pool reservations and slicing notes up front:

```python
@dataclass(frozen=True)
class LoopTail(pllm.Workload):
    threshold: float = 0.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h = layers(stop=26)[25]
        active = torch.ones_like(h[:, :1], dtype=torch.bool)
        for layer in range(26, 28):
            active = active & (h.float().square().mean(-1, keepdim=True) > self.threshold)
            h = layers(layer, layer + 1, hidden=h, active=active)[layer]
        return h


print(pllm.explain(LoopTail(), config))
```

The leading lines of output look like this (`timing` and `note` lines omitted):

```text not-output
model    Qwen2ForCausalLM, hidden 3584, 28 layers, vocab 152064, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  eager: @pllm.custom_forward schedules model slices and owns their wave graphs
base     14140571648 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
kernels  forward: ~11 (eager) / step
forward  custom_forward  model.execute / main / eager  - (entry, 0)
  CUDA-graph pool reservation: 4244635648 bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]
  matmul/byte estimate / step: unavailable; profile this model-site or host-batch body with a loaded model
  forward eager because forward: @pllm.custom_forward schedules model slices and owns their wave graphs
kernels  total: ~11 (eager) / step [estimate: source operators; excludes base model]
state  scope  bytes/root  reserved bytes [exact; state_page_budget_bytes]
memory   peak 4273995778 bytes funded by pLLM [per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace; execution_peak_budget_bytes]
preempt  state reset; replay rejection: None
host     0 syncs in traced workload math
```

Two key differences stand out here:

- **`forward eager`**: `@pllm.custom_forward` conditionally controls the execution order of layer spans, so the outer wrapper runs eagerly while individual layer spans are still captured as wave CUDA graphs (budgeted in `CUDA-graph pool reservation`). By contrast, simply adding or subtracting weight overlays with `Adapter` on a shared `InstanceState` or per-request `RequestState` never triggers this line.
- **`model.execute / main / eager`**: Sliced execution replaces the model's main forward itself, so it attaches to `model.execute` on the main lane.

Comparing the two workloads side by side:

| Workload | Model Forward & Execution Mode |
| --- | --- |
| `MiniESamp` | vLLM whole-model CUDA graph; training overlaps subsequent forwards on the side stream |
| `LoopTail` | Outer conditional slice scheduling; layer spans run inside wave CUDA graphs |

You can weigh the trade-off before running anything: when you want whole-model static graph overhead and minimal dispatch latency, prefer standard layer sites with `Adapter` / `ActivationPatch`; when you need dynamic cross-layer routing or early exit, `pllm.explain` sizes the wave-graph pool for you up front.

## When a Budget Is Too Tight, It Tells You What to Change

If you configure a memory budget and want to check whether the current config fits—or which parameter to tune if it does not—inspect `budget_errors` on the report:

```python
small = pllm.ModelConfig("Qwen/Qwen2.5-7B-Instruct", max_model_len=512, max_num_seqs=128,
                         max_num_batched_tokens=8192,
                         execution=pllm.ExecutionOptions(execution_peak_budget_bytes=64 << 20))
print(pllm.explain(MiniESamp(beta=0.5), small).budget_errors)
```

Output:

```text
('pLLM execution needs 1168476308 bytes at its peak but only 67108864 fit; lower max_num_batched_tokens or raise ExecutionOptions.execution_peak_budget_bytes; reduce device tables or explicitly declare placement="host" for read-only lookup tables',)
```

An empty `budget_errors` tuple means the workload fits within the declared budget. When non-empty, the report still shows the unconstrained plan and attaches the exact error message `make_llm` would raise. Other admission failures (such as unsupported compositions) cause `pllm.explain` to raise directly with the same error message as engine creation.

## What It Does Not Know

Because `pllm.explain` runs on CPU, it has the following boundaries:

- Whether everything fits on your GPU is only decided at `make_llm`: reported memory excludes model weights, the KV cache, and vLLM's own activations and fragmentation.
- BLAS workspace sizes used by projections and scans depend on the GPU and are excluded; affected peaks are lower bounds.
- FLOPs and logical memory traffic bytes are shape-based estimates (lower bounds for custom operators), with no throughput or latency predictions—measure throughput empirically (as in Tutorial 03).
- Sync counts cover only CPU shape-traced workload math; computations inside model sites (`ActivationPatch`, `custom_residual`, etc.) are marked `unverified`.
- Config-only inspection currently derives projection shapes for dense Qwen2, Qwen3, and Llama configs; projection admission for other model families is checked at `make_llm`.

## For Scripts and Coding Agents

If you are writing automation scripts or having an agent check workloads automatically, there is no need to parse the text output. The object returned by `pllm.explain` converts via `dataclasses.asdict(report)` into a JSON-serializable dictionary containing `eager_reason`, `members` (per-method tasks), `states`, `peak_memory_bytes`, `budget_errors`, `exports`, and more; `flops_per_token_estimate` gives the combined per-token FLOP estimate across the base model and all tasks:

```python
import json
from dataclasses import asdict

report = pllm.explain(MiniESamp(beta=0.5), config)
print(report.eager_reason == "", [state.reserved_bytes for state in report.states])
print(len(json.dumps(asdict(report))) > 0)
```

This prints `True [102764592]` and `True`.
