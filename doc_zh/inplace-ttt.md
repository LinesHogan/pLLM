# Native In-Place TTT

内置 In-Place-TTT 已是 typed native workload，不再依赖 `BaseConsumer`、
`ModuleWeightView` 或进程全局 `runtime.fast_weight` 栈。

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

签名图拥有 exact chunk 数学、request/sample/layer 状态、tail 延续、版本发布和
request cleanup。满块先用当前有效 `down_proj` 权重产生输出，再执行
`W += lr * target.T @ hidden`；不足一块的行留到下一次调用。decode 是否更新是算法
语义配置，不是 workload 调度策略。

权重通过通用 `OverlayVersionStore` 发布：多 tensor 事务提交、私有 payload、稳定
request lease、未 pin 旧版本回收以及 finish 释放都由共享能力负责。placement、stream、
fusion 和 overlap 仍由 Rust 调度器选择。

当前证据覆盖 CPU exact tensor parity、多步 tail、immutable publication 与 request
cleanup。vLLM 内 dense version 的真实应用仍是 backend adapter 验收项；在该 GPU 测试和
CUDA Graph parity 完成前，不宣称官方 checkpoint parity。
