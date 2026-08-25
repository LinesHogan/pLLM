# 适配 In-Place-TTT 风格算法

In-Place TTT 在生成过程中更新模型**自身的权重**（例如 MLP 的 `down_proj`）。tLLM 当前提供的是安装、恢复这类权重 overlay 所需的通用 backend 骨架；现阶段**不宣称**与官方 In-Place-TTT 的数学或 checkpoint parity。

| 一半 | 是什么 | tLLM 机制 |
|---|---|---|
| **学习** | 在算法允许时读 hidden 对、计算更新 | 普通的 `@flow`/`@extern` **Program** |
| **应用** | 目标模块的 forward 必须用*有效权重*运行 | **权重 overlay**——backend 能力，不是训练 flow |

## 1. 哪些部分可以是 Program

当算法允许异步 hidden-pair 表达时，读 hidden state 并在上面训练不需要特殊支持。最小程序：

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

分块更新（上游 In-Place TTT 只在满 `chunk_size` 的块上更新、尾部缓存到下一步）是 `consume_flow` 外围的纯 Python——见内置 `inplace_ttt` consumer。`down_proj` 有效权重上逐块最小二乘更新的纯算法参考是 `tllm/consumers/inplace_ttt/engine.py` 里的 `run_exact_toy_down_proj_step`；把你自己的数学也放进这种纯 helper，保持 CPU 可测。

但这种拆分**不等价于官方算法**。官方实现会在目标层内部同步做 chunk 更新，并立即在该层 forward 中使用更新后的有效权重；这种顺序通常不能改写成模型执行后再消费 hidden 对的 Program。官方适配还需要在本页所述 backend 骨架之上实现专用的层内 overlay provider。

## 2. 应用一半是权重 overlay

把学到的东西应用上，意味着模型内部某个模块要用修改后的权重（`W + Δ` 或重写后的 `W`）跑 forward。这是对**模型自身 forward** 的干预，不是训练 flow——Program 模型刻意没有权重改写结构，所以这一半走 overlay 栈：

- **端口**：`ModuleWeightView`（`tllm/ports/module_weight_view.py`）。在 consumer 的 `flows()` 里声明写：
  ```python
  ModuleWeightView.write(layer=target_layer, module="mlp", phase="prefill", mode="wrapper")
  ```
  这让执行计划看到写冲突：两个 consumer 写同一个模块权重是调度错误，而不是竞态。
- **机制**：`tllm/runtime/fast_weight/`——状态按 `(request_id, sample_idx, layer)` 版本化，包含有界分块更新、packed row phase metadata 以及 wrapper 安装/恢复。同一个 packed prefill batch 中，不同请求不会共享 tail 或有效权重；static-key 模式仍只允许一个活跃请求。
- **backend 能力**：`BackendCapabilities.module_weight_overlay` 类型化声明 module、write mode、phase、request-keyed、graph safety 和 TP 范围。vLLM 0.27 MRv1/MRv2 driver 负责解析 `ModuleWeightViewLocator`（含负 layer index），并持有 instance-scoped wrapper handle。runtime disable、backend unbind、reload 或部分安装失败都会恢复原始 `forward`。**新的 backend adapter 必须先提供同等能力，overlay 算法才能运行**。

通用边界到这里为止。backend driver 只接收通用的 `installer(module) -> restorable_handle` 回调，不依赖 fast-weight 状态或更新数学。当前 `exact` fast-weight runtime 是这个回调的第一个 reference provider，而不是跨任务通用的学习算法。若任务具有不同的状态、更新时序、loss、参数集合或 checkpoint 格式，应实现不同 provider，同时复用 locator、准入、逐请求上下文和生命周期机制。

运行时配置在 runtime config / control plane 上（`set_fast_weight_enabled`、`fast_weight_layers`、`fast_weight_chunk_size`、`fast_weight_lr`、`fast_weight_mode`、key 模式、slot/字节上限）——overlay 是共享基建，全局配置一次，不属于单个 consumer。

活跃 flow 声明 `ModuleWeightView.write` 时，必须同时满足：backend 具有精确的 `module_weight_view/wrapper/<phase>` capability、overlay provider 已启用、配置 layer 与声明 locator 一致、runner 为 eager、TP=1。所有检查都在模型执行前完成，并一律 fail closed。

## 当前支持边界

- vLLM 0.27 MRv1/MRv2、eager 执行、TP=1。
- packed 多请求 prefill、chunked prefill、decode，以及 prefill/decode 混合 row plan。
- 保留现有 `exact` fast-weight provider；`low_rank` 仍显式不支持。
- 首期不承诺 speculative decode、量化 `down_proj`、PP/DP、compile 或 CUDA graph。
- 不加载官方 checkpoint 独有的 `ttt_proj`/`ttt_conv` 参数，也不宣称官方 Qwen3/In-Place-TTT 数学 parity。

## 清单

1. 把学习一半写成 Program（hidden tap + `@flow` + `@extern`）。
2. 对算法要 overlay 的每个模块，在 `flows()` 里声明 `ModuleWeightView.write(...)`。
3. 更新数学放进纯的、CPU 可测的 helper。
4. 通过 fast-weight runtime 驱动/验证 overlay；确认 backend 能安装并恢复 wrapper（GPU smoke）。
5. Fail closed：backend 无法 overlay 声明的模块时显式报错——绝不降级为"训练但不生效"。

## 现状

内置 `inplace_ttt` consumer 与 exact fast-weight 路径是通用 overlay backend 的轻依赖集成参考，不是官方 In-Place-TTT 实现。下一阶段应增加模型参数加载和官方 Qwen3 数学 parity；compile/CUDA graph 再单独解决。`esamp`、`ttrl`、`fttt`、`override` 是从导出 tensor 学习的 Program 模板。TLM 的边界不同：它用完整 HF/PEFT causal-LM side trainer 训练内部 `q_proj/v_proj` LoRA，再通过 vLLM 原生 LoRA manager 发布不可变版本；它不使用 fast-weight overlay，也不是 hidden-to-vocab Program 替身。
