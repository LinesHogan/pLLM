# Native In-Place TTT

The built-in In-Place-TTT implementation is a typed native workload. It no
longer uses `BaseConsumer`, `ModuleWeightView`, or the process-global
`runtime.fast_weight` stack.

```python
from tllm.consumers.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload
from tllm.runtime import TllmRuntime

workload = InPlaceTTTWorkload(
    InPlaceTTTConfig(
        layers=(0,),
        chunk_size=4096,
        learning_rate=1e-3,
        update_prefill=True,
        update_decode=False,
    )
)
runtime = TllmRuntime(workloads=(workload,))
```

The signed graph owns exact chunk math, request/sample/layer state, tail
carry, version publication, and request cleanup. Full chunks produce output
with the current effective `down_proj` weight and then apply
`W += lr * target.T @ hidden`; the remaining rows are carried to the next
invocation. Decode updates are a semantic config choice, not a scheduling
policy.

Published weights use the shared `OverlayVersionStore`: multi-tensor
publication is transactional, payloads are privately owned, request leases
are stable, old unpinned versions are collected, and finish events release
both tail state and leases. Physical placement, streams, fusion, and overlap
remain Rust scheduler decisions.

Current validation includes exact CPU tensor parity, multi-step tail state,
immutable version publication, and request cleanup. Dense-version application
inside vLLM remains a backend-adapter validation item; do not claim official
checkpoint parity or CUDA-Graph parity until that test is present.
