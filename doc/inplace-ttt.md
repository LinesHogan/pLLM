# Canonical In-Place TTT

In-Place-TTT now runs only on the clean-room canonical core. Its workload
declares typed targets, chunk math, request state, effects, and timing; Rust
selects the implementation and signs schedule, PTTS representation, and the
parameter-overlay attachment.

```python
import tllm
from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTConfig, InPlaceTTTWorkload,
)
from tllm.canonical_vllm.runtime import CanonicalVllmConfig, CanonicalVllmDType
from tllm.contracts.parameter_overlay import (
    ParameterModuleKind, ParameterRole, ParameterTarget,
)

workload = InPlaceTTTWorkload(InPlaceTTTConfig(
    targets=(ParameterTarget(
        0,
        ParameterModuleKind.MLP,
        ParameterRole.DOWN_PROJECTION_WEIGHT,
    ),),
    chunk_size=4096,
    learning_rate=1e-3,
    update_prefill=True,
    update_decode=False,
))
llm = tllm.make_llm(
    workloads=(workload,),
    canonical_config=CanonicalVllmConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype=CanonicalVllmDType.BFLOAT16,
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.8,
    ),
)
```

Each request owns one `StateRef(LayoutId, LineageId, VersionId)` root containing
all configured targets' delta weights and fixed tail buffers. Full chunks emit
with the current effective weight and then apply
`delta += lr * target.T @ hidden`; incomplete rows remain in the immutable next
root. A tail-only transition advances PTTS state but does not fabricate a
parameter publication. When a weight changes, one atomic publication exposes
all target field bindings as `DENSE_ADD`.

The eager MRv2 adapter never mutates model parameters and owns no version table.
It pins exact PTTS roots, adds `F.linear(hidden, delta)` only to signed packed
rows, captures the real MLP/down-projection operand pair, runs one batched
signed transition/publication, and releases the old pin through the sampler
completion fence. Request finish retires roots only after pin and writer fences
complete.

Current real-runtime evidence is Qwen2, vLLM 0.27.1 MRv2, eager in-process,
TP1/PP1/DP1, non-speculative decode. One two-request run observed six updates,
six trained rows, eight dense-apply rows (two from versions above V0), zero
active roots, zero pending releases, and zero fallback. MRv1, EngineCore-MP,
TP/PP>1, quantized targets, speculative decode, and CUDA Graph are not claimed;
backend adapters remain separate from workload scheduling.

Publication is deliberately model-step-granular: a model call pins one version
and publishes its successor for a later call. Updating between chunks inside a
single long prefill would require signed layer/chunk subregions and is not hidden
inside the adapter.
