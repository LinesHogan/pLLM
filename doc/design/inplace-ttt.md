# In-Place TTT

In-Place-TTT 运行在共享执行核心上。它的负载声明带类型的 targets、块数学、
请求状态、效果与时机；Rust 选择实现并签名调度、PTTS 表示与参数 overlay
附着。

```python
import pllm
from vllm import SamplingParams
from pllm.workloads.inplace_ttt import (
    InPlaceTTTConfig, InPlaceTTTWorkload,
)
from pllm import ModelConfig
from pllm.model import Projection

workload = InPlaceTTTWorkload(InPlaceTTTConfig(
    targets=(Projection("down_proj", 0),),
    chunk_size=4096,
    learning_rate=1e-3,
    tokens="prefill",
))
with pllm.make_llm(ModelConfig("Qwen/Qwen2.5-0.5B-Instruct"), workload=workload) as llm:
    outputs = llm.generate(["Explain matrix multiplication:"], SamplingParams(max_tokens=32))
```

每个请求拥有一个 `StateRef(LayoutId, LineageId, VersionId)` 根，内含所有配置
target 的 delta 权重与固定尾部缓冲。完整块以当前有效权重发射，然后应用
`delta += lr * target.T @ hidden`；不完整的行留在不可变的 next 根里。仅尾部的
迁移推进 PTTS 状态，但不伪造参数发布。权重变化时，一次原子发布把所有 target
字段绑定暴露为 `"dense"`。

eager MRv2 适配器从不修改模型参数，也不拥有版本表。它 pin 住精确的 PTTS 根，
只对签名打包行加 `F.linear(hidden, delta)`，捕获真实的 MLP/down 投影操作数对，
跑一次批处理的签名迁移/发布，并通过采样器完成栅栏释放旧 pin。请求结束只在
pin 与写入栅栏完成后才退役根。

当前真实运行时证据为 Qwen2、vLLM 0.27.1 MRv2、eager 进程内、TP1/PP1/DP1、
非投机 decode。一次两请求运行观察到六次更新、六个训练行、八次稠密应用行
（两次来自 V0 以上的版本）、零活跃根、零待释放与零回退。MRv1、
EngineCore-MP、TP/PP>1、投机 decode 与 CUDA Graph 未被声称（量化 target 另见 README）；
后端适配器与负载调度保持分离。

发布刻意是模型步粒度的：一次模型调用 pin 一个版本，为之后的调用发布其后继。
在单个长 prefill 内的块之间更新，需要签名的层/块子区域 —— 这不会被藏在
适配器里。
