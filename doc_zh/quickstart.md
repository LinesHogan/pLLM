# 快速上手：ESamp 与 Execution Region

新的 native work 使用 `tllm.workload`（`@workload`/`@task`），并作为已签名的
`ExecutionRegion` 运行。Rust 决定 Region 边界与 `RegionCaptureMode`；Python 只绑定
artifact 和资源。下方 Program 示例为 legacy adapter 用户保留，不是新 workload 的调度或
stream/event 模型。请先阅读[附加载荷编译器](workload-compiler.md)。

本页通过一个保留的短 Program 示例介绍 **ESamp**（内置 test-time training 算法）。native
ESamp 由 Region 编译器声明；本页不宣称兼容 Program 控制 Region capture 或物理调度。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
pip install -e ".[vllm027]"
```

要求：Python ≥ 3.11、带 CUDA 的 PyTorch、vLLM v1（已在 `vllm==0.27.1` 验证）。

## 算法

decode 期间，一个微型低秩模型在线学习：从第一层 hidden state 预测最后一层 hidden state。它的预测随后被用来微调候选 token 的 logits——模型在生成的同时自我引导。

## 完整实现

```python
import torch
import torch.nn.functional as F

from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, hidden, local, result


class ESampProgram(torch.nn.Module):
    """Predict the final-layer hidden state from the first-layer one."""

    def __init__(self, *, hidden_size: int, rank: int, lr: float,
                 device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.down = torch.nn.Linear(hidden_size, rank, bias=False, device=device, dtype=dtype)
        self.up = torch.nn.Linear(rank, hidden_size, bias=False, device=device, dtype=dtype)
        torch.nn.init.zeros_(self.up.weight)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, foreach=False)

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="decode"),
            "target": hidden(layer=-1, site="block_output", phase="decode"),
        },
        outputs={"loss": local()},
        state="distiller",
    )
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = self.up(self.down(source))
        return F.mse_loss(prediction.float(), target.float())

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="distiller",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
```

这就是整个算法。逐条读声明：

- `@flow(inputs=...)`——`forward` 每个 decode step 读两个引擎内部张量：`hidden(layer=0, ...)` 在第一个 block 后取 residual stream，`hidden(layer=-1, ...)` 在最后一层后取。
- `outputs={"loss": local()}`——loss 只在程序内部流转（被下面的 extern 消费）。内置版本还会把预测发布为 `sink("guidance")`，sampler bridge 订阅它做候选级引导。
- `state="distiller"`——模块参数与 optimizer 状态住在 runtime 持有的版本化 **state bank** 里。你不需要管 buffer、stream 或 event。
- `@extern(inputs={"loss": result("forward", "loss")}, ...)`——`backward` 消费 `forward` 的 loss **连同其 autograd 图**。这条边让编译器知道需要 grad 模式；`deadline=` 约束更新必须落定的时机。
- `backward` 函数体是普通 PyTorch——它同时是 **oracle**：编译器把你的 optimizer 降级为 bank slot 上的函数式 transition，并在编译期校验两者一致（`PG145`）。没有静默 fallback。

仓库内置的兼容版本（`tllm/consumers/esamp/program.py`）形状相同，只是换成 gated
residual 模型并多一个 `predict` flow。这是 ESamp 迁移基线，不是 native FTTT 遵循的模式。

Canonical FTTT 使用严格类型的 host transaction 入口：

```python
import torch
from tllm.canonical.host_runtime import (
    CanonicalHostRuntimeFacts,
    HostTransactionCoordinator,
)
from tllm.canonical.workloads.fttt import (
    FTTTCompileOptions, FTTTConfig, FTTTDefaultJudge,
    FTTTFeedbackService, FTTTSpecialization, FTTTWorkload,
)
from tllm.contracts.tensor_layout import TensorDType

config = FTTTConfig(rank=16, learning_rate=1e-3, max_retries=2)
device = torch.device("cpu")
compiled = FTTTWorkload(config).compile(
    FTTTSpecialization(896, TensorDType.FLOAT32, device, 0),
    FTTTCompileOptions(CanonicalHostRuntimeFacts(device).timeline()),
)
feedback = FTTTFeedbackService(config, FTTTDefaultJudge())
fttt = HostTransactionCoordinator(feedback, compiled.execution)
```

feedback service 将 judge 与队列状态转换成 typed transaction；coordinator 负责签名执行与
receipt settlement。选择、特征构造、loss、Adam、edit prediction 和 publication 都作为
signed tasks 执行。FTTT 只声明依赖、effect 和语义 timing；物理 Region 划分、放置与重叠
由通用 Rust scheduler 自动选择，不存在 FTTT 专属 lane、priority 或 fusion 策略。旧
`tllm.consumers.fttt` / `TllmRuntime` 路径已经退役，不提供兼容 wrapper。

## 运行

```python
from vllm import SamplingParams
from tllm import make_llm
from tllm.consumers.esamp import ESampConsumerConfig, activate

activate(ESampConsumerConfig(implementation="program"))
llm = make_llm(model_name="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.8, n=8))
```

ESamp 的 GPU 证据使用 canonical 验证场景：

```bash
.venv/bin/python tools/verification/automated_tests.py \
  --scenario cuda_graph_replay_preserves_training_and_zero_fallback
```

## Legacy 编译器从这 40 行里推导出了什么

- **切分**：每个 flow 切成 `torch.export` 图（不支持的结构 → 编译期 `PG###` 报错）。
- **调度**：训练走兼容专用 lane，只读预测走低优先级 reader lane——decode 主流只等它实际消费的 event。所有的边都来自你的声明。
- **捕获**：捕获符合条件的 legacy train step，并按 batch 形状生成 CUDA Graph。
- **降级**：把 `torch.optim.AdamW` 变成版本化 bank slot 上的函数式更新，并与你的 `backward` 函数体对照校验。

## 下一步

- [编写 consumer](write-a-consumer.md)——完整声明词汇（含调用方喂张量的 `provided()`）、薄 consumer 外壳、错误码表。
- [ESamp 参考](esamp.md)——配置项、legacy baseline、实测数字。
- [适配 In-Place-TTT 风格算法](inplace-ttt.md)——当你的算法要改模型自身权重（而非旁挂模块）时怎么做。

## 排错

- **启动时报 `PG###`？** 编译器在拒绝不支持的结构——见[错误码表](write-a-consumer.md#error-codes)。全部 fail-closed，不会静默降级。
- **CUDA Graph 没启用？** native workload 的 capture 由已签名的 `RegionCaptureMode` 控制；legacy Program capture 是独立兼容行为。
- **吞吐合理性检查**：benchmark workflow 会连同 tok/s 一起报告功能计数器（loss 数、干预状态数），坏掉的空转 consumer 无法伪装成快 consumer。
