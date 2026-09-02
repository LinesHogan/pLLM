# Canonical In-Place TTT

In-Place-TTT 现在只运行在 clean-room canonical core 上。workload 仅声明 typed
target、chunk 数学、request state、effect 与 timing；Rust 自动选择 implementation，
并签名 schedule、PTTS representation 和 parameter-overlay attachment。

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

每个 request 只有一个 `StateRef(LayoutId, LineageId, VersionId)` root，其中包含
所有 target 的 delta weight 和固定 tail buffer。满 chunk 先用当前有效权重输出，再执行
`delta += lr * target.T @ hidden`；不足一块的行进入下一个 immutable root。只有 tail
变化时 PTTS 版本会推进，但不会伪造 parameter publication；weight 更新时，所有 target
field 通过一次原子 `DENSE_ADD` publication 可见。

eager MRv2 adapter 不修改模型参数，也不维护第二套版本表。它 pin 精确 PTTS root，只对
signed packed row 加上 `F.linear(hidden, delta)`，捕获真实 MLP/down-projection operand，
执行一个 batched signed transition/publication，并通过 sampler completion fence 释放旧
版本。request finish 只在 pin/writer fence 完成后回收 root。

当前真实 runtime 证据限定为 Qwen2、vLLM 0.27.1 MRv2、eager in-process、
TP1/PP1/DP1、non-speculative decode。一次双 request 运行观察到 6 次 update、6 个训练行、
8 个 dense-apply 行（其中 2 行来自 V0 以上版本），最终 active root、pending release 和
fallback 均为 0。当前不宣称 MRv1、EngineCore-MP、TP/PP>1、量化 target、speculative
decode 或 CUDA Graph 支持；backend adapter 与 workload 调度保持分离。

publication 以 model step 为粒度：一次 model call pin 一个版本，并把后继版本留给之后
的 call。若要在单次长 prefill 内跨 chunk 同步更新，必须引入 signed layer/chunk
subregion，不能把顺序逻辑藏进 adapter。
