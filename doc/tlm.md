# Native TLM q/v LoRA

TLM is a native HF/PEFT workload. Model loading is a cold typed service;
forward, official sentence loss, backward/AdamW, q/v snapshot, and immutable
publication are signed tasks.

```python
from tllm.consumers.tlm import TLMConfig, TLMWorkload
from tllm.runtime import TllmRuntime

workload = TLMWorkload(
    TLMConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        mode="online",       # or "offline"
        rank=8,
        lora_alpha=16,
        learning_rate=5e-5,
        local_files_only=True,
    )
)
runtime = TllmRuntime(workloads=(workload,))
```

`mode` is a Rust-normalized finite config choice. It selects one semantic DAG
at signing time:

- online trains in the post-token window;
- offline trains in the prepare window before model execution.

This is semantic ordering, not a TLM scheduler. The generic scheduler still
selects physical placement and overlap inside the legal window.

Only PEFT q/v LoRA parameters are trainable. Every batch performs one optimizer
transition and one publication. `OverlayVersionStore` owns the tensors,
provides stable request leases, and safely collects unpinned versions. The
vLLM quarantine adapter translates generic `low_rank_add` versions into native
`LoRAModel`/`LoRARequest` objects; it contains no TLM training decisions.

The GPU smoke `test/gpu/native_tlm_workload_smoke.py` verifies real tiny-model
HF/PEFT training, `loss_count/update_count > 0`, immutable version 1, vLLM
adapter installation, and LoRA kernel execution on MRv2 eager.
