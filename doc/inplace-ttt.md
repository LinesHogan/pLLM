# Adapting an In-Place-TTT-Style Algorithm

In-Place TTT updates the model's **own weights** (e.g. an MLP `down_proj`)
while generation proceeds. tLLM now provides the general backend skeleton
needed to install and remove such a weight overlay. It does **not** yet claim
mathematical or checkpoint parity with the official In-Place-TTT model.

| Half | What it is | tLLM mechanism |
|---|---|---|
| **Learning** | read hidden pairs, compute an update when the algorithm permits it | an ordinary `@flow`/`@extern` **Program** |
| **Application** | the target module's forward must run with an *effective* weight | a **weight overlay** — a backend capability, not a training flow |

## 1. What can be a Program

Reading hidden states and training on them needs no special support when the
algorithm admits an asynchronous hidden-pair formulation. A minimal program:

```python
import torch
import torch.nn.functional as F

from tllm.consumers._common.alignment import LowRankAdapter
from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, hidden, local, result


class InPlaceTTTProgram(torch.nn.Module):
    def __init__(self, *, hidden_size: int, rank: int, lr: float, device, dtype) -> None:
        super().__init__()
        self.adapter = LowRankAdapter(in_dim=hidden_size, out_dim=hidden_size,
                                      rank=rank, device=device, dtype=dtype)
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=lr, foreach=False)

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="prefill"),
            "target": hidden(layer=-1, site="block_output", phase="prefill"),
        },
        outputs={"loss": local()},
        state="adapter",
    )
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.adapter(source), target - source)

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="adapter",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
```

Chunked updates (upstream In-Place TTT updates only on full
`chunk_size` blocks and caches the tail) are plain Python around
`consume_flow` — see the built-in `inplace_ttt` consumer. The pure
algorithmic reference for the chunk-wise least-squares update on an
effective `down_proj` weight is `run_exact_toy_down_proj_step` in
`tllm/consumers/inplace_ttt/engine.py`; keep your own math in a pure helper
like that so it stays CPU-testable.

This decomposition is **not equivalent to the official algorithm**. The
official implementation performs synchronous chunk updates inside the target
layer and immediately uses the effective weight in that layer's forward. That
ordering cannot generally be rewritten as a Program that consumes hidden
pairs after model execution. An official adapter will need a dedicated
in-layer overlay provider in addition to the backend skeleton described here.

## 2. The application half is a weight overlay

Applying what was learned means a module inside the model runs its forward
with a modified weight (`W + Δ`, or a rewritten `W`). That is an
intervention **inside the model's forward**, not a training flow — the
Program model deliberately has no weight-rewriting construct, so this half
goes through the overlay stack:

- **The port**: `ModuleWeightView` (`tllm/ports/module_weight_view.py`).
  Declare the write in your consumer's `flows()`:
  ```python
  ModuleWeightView.write(layer=target_layer, module="mlp", phase="prefill", mode="wrapper")
  ```
  This gives the plan write-conflict visibility: two consumers writing the
  same module weight is a scheduling error, not a race.
- **The mechanism**: `tllm/runtime/fast_weight/` — versioned state keyed by
  `(request_id, sample_idx, layer)`, bounded chunk updates, packed-row phase
  metadata, and wrapper install/restore. Different requests in one packed
  prefill batch never share a tail or effective weight. Static-key mode is
  still restricted to one active request.
- **The backend capability**: `BackendCapabilities.module_weight_overlay`
  declares the supported module, write mode, phases, request-keying, graph
  safety and TP sizes. The vLLM 0.27 MRv1 and MRv2 drivers resolve
  `ModuleWeightViewLocator` (including negative layer indices) and own the
  instance-scoped wrapper handles. Runtime disable, backend unbind, reload,
  and partial installation failure restore the original `forward`.
  **A new backend adapter must provide the same capability before an
  overlay algorithm can run on it** — this is the piece to port, and it is
  the reason In-Place-TTT-style consumers are not pure Programs.

The reusable boundary stops there. A backend driver receives a generic
`installer(module) -> restorable_handle` callback and has no dependency on
fast-weight state or update math. The current `exact` fast-weight runtime is
the first reference provider for that callback, not a universal learning
algorithm. A task with different state, update timing, loss, parameter set,
or checkpoint format should supply a different provider while reusing the
same locator, admission, request context, and lifecycle machinery.

Runtime configuration lives on the runtime config / control plane
(`set_fast_weight_enabled`, `fast_weight_layers`, `fast_weight_chunk_size`,
`fast_weight_lr`, `fast_weight_mode`, key mode, slot/byte caps) — the
overlay is shared infrastructure, configured once, not per consumer.

An active `ModuleWeightView.write` is admitted only when all of the following
are true: the exact `module_weight_view/wrapper/<phase>` capability exists,
the overlay provider is enabled, configured layers match declared locators,
the runner is eager, and tensor parallel size is 1. Validation happens before
model execution and always fails closed.

## Current support boundary

- vLLM 0.27 MRv1 and MRv2; eager execution; TP=1.
- Packed multi-request prefill, chunked prefill, decode, and mixed
  prefill/decode row plans.
- The existing `exact` fast-weight provider. `low_rank` remains explicitly
  unsupported.
- No speculative decode, quantized `down_proj`, PP/DP, compile mode, or CUDA
  graph guarantee in this phase.
- No loading of official checkpoint-only `ttt_proj`/`ttt_conv` parameters and
  no official Qwen3/In-Place-TTT mathematical parity claim.

## Checklist

1. Write the learning half as a Program (hidden taps + `@flow` + `@extern`).
2. Declare `ModuleWeightView.write(...)` in `flows()` for every module the
   algorithm overlays.
3. Keep the update math in a pure, CPU-testable helper.
4. Drive/verify the overlay through the fast-weight runtime; confirm the
   backend installs and restores wrappers (GPU smoke).
5. Fail closed: if the backend cannot overlay the declared module, raise —
   never degrade to "trains but has no effect".

## Status

The built-in `inplace_ttt` consumer and exact fast-weight path are a
dependency-light integration reference for the generic overlay backend. They
are not the official In-Place-TTT implementation. The next phase is model
parameter loading plus official Qwen3 mathematical parity, followed
separately by compile/CUDA-graph support. The built-in consumers `esamp`,
`ttrl`, `fttt`, and `override` are Program-backed templates for learning over
exported tensors. TLM is a different boundary: it runs a full HF/PEFT
causal-LM side trainer and publishes immutable internal `q_proj/v_proj` LoRA
versions through vLLM's native LoRA manager. It does not use the fast-weight
overlay and is not a hidden-to-vocab Program surrogate.
