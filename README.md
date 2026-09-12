# PLLM

PyTorch 会训练，vLLM 会推理。但如果你想让模型在**服务的同时继续学习**——
观测中间层激活、按请求叠 LoRA、在 decode 的间隙更新参数——你会发现两边都不
管这段路：训练框架不认识 KV cache，推理引擎不认识优化器。

PLLM 补上这段路。算法只声明数学、数据流、状态与效果；Rust 签名出唯一的
执行计划，托管运行时负责版本、发布、捕获与完成。框架不认识任何算法名——
约束只来自形状、dtype、依赖、效果、设备与容量。

第一次来？先跑下面的[五分钟起步](#五分钟起步)，再到[教程](#教程从基础到场景)
按兴趣分流。

编译中段改造的当前能力、验收证据与剩余工作见
[功能与进展收口](doc/design/implementation-status.md)。

## 五分钟起步

```python
from vllm import SamplingParams
from pllm import ExecutionOptions, ModelConfig, make_llm
from pllm.workloads import ESampRequestConfig, ESampRequestWorkload

config = ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", execution=ExecutionOptions(capture=True))
with make_llm(config, workload=ESampRequestWorkload(ESampRequestConfig(rank=2))) as llm:
    outputs = llm.generate(["Explain gradient descent:"], SamplingParams(max_tokens=32))
    print(outputs[0].outputs[0].text)
    print(llm.diagnostics())
```

省略 `workload` 就是普通的 vLLM 推理。宿主机算法用同一工厂函数配合
`HostRuntimeFacts(device, hidden_size=...)`，无需构造模型。

> 迁移说明：Python 导入与发行名现在是 `pllm`，不再安装 `tllm` 兼容包；
> 旧的模块路径与 `canonical_config` 不是兼容 API。既有调用方请迁移到
> `workload=` 与 `config=` 参数。

## 教程：从基础到场景

入门路径是 `doc/tutorials/` 下的中文 notebook，读法和 PyTorch 官方教程一样：
**边跑边改**。每篇从一个能跑的最小例子出发，逐步加概念，最后落到完整场景。
主线六篇按顺序读，跑完 00 之后就可以按兴趣分流——做算法的继续 01–05，
想搞清框架内部机制的直接跳系统向支线（纯 CPU 即可运行）：

**算法向（GPU 环境，见[安装](#安装)）**

| 教程 | 你会学到什么 |
| --- | --- |
| [00 快速上手](doc/tutorials/00_quickstart.ipynb) | PLLM 的心智模型：声明一个 Workload、让 Rust 签名、交给托管运行时。最小观测负载 + trace 读取 |
| [01 观测：给三层 hidden 装 SAE](doc/tutorials/01_observe_sae.ipynb) | 观测任务、导出语义（`max_rows`/`max_cols`）、冻结状态、后台线程消费 trace、落盘 pkl |
| [02 自定义训练头](doc/tutorials/02_custom_head.ipynb) | 训练任务与采样引导、loss 约定、参数按行就位的 `bmm` 写法；把头从 MLP 换成 CNN |
| [03 每请求 LoRA](doc/tutorials/03_request_lora.ipynb) | 参数 overlay、route 键共享 lineage、判别器混合两个头、prefill 逐块学习（波前 vs Jacobi） |
| [04 三种循环](doc/tutorials/04_loops.ipynb) | 层间重放、逐步 hidden 再循环、跨请求会话循环（memory + Capture） |
| [05 decode 时在线训练](doc/tutorials/05_decode_training.ipynb) | 跨步与跨请求的训练目标：当前步、下一步、上一 VIP 请求终态，全靠输入声明组合 |

**系统向（内部机制，纯 CPU 可运行）**

| 教程 | 你会学到什么 |
| --- | --- |
| [签名编译管线](doc/tutorials/internals/01_signed_compilation.ipynb) | 声明如何变成签名计划 |
| [托管状态与页 arena](doc/tutorials/internals/02_state_arena.ipynb) | 版本、COW、pin、预算 |
| [进程边界](doc/tutorials/internals/03_process_boundary.ipynb) | 按引用传输与签名复核 |

## 自己写负载

一个负载就是一个声明，不是运行时钩子：

```python
import torch
import pllm

class CenteredHidden(pllm.Workload):
    @pllm.on(pllm.Phase.OBSERVE)
    @pllm.hook(0)
    @pllm.export(name="centered_hidden", every=256, max_cols=16)
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden - hidden.mean(dim=-1, keepdim=True)
```

这段代码声明的是捕获与一个输出效果。完整的
[负载创作契约](doc/reference/authoring-contract.md) 涵盖参数作用域、注解输入、
选择、一阶梯度边、激活检查点与带条件重放的请求 overlay；按任务的写法见
[使用指南](doc/README.md#使用指南任务导向)。

打算把需求交给编码代理？先看[负载需求模板与代理指南](doc/guides/agent-workloads.md)：
要提供什么、参考哪些例子、代理应该交付什么。

## 内置算法

| 算法 | 行为 |
| --- | --- |
| Dummy | 选中 hidden 行的导出与有界 trace |
| ESamp | 请求内训练与采样引导 |
| In-Place-TTT | 请求内稠密参数 overlay |
| TLM | 实例级 q/v LoRA、AdamW 与在线/离线发布 |
| Output LoRA | 共享或独立的请求头、加权 CE 与 SGD |
| ReFT | 签名激活编辑与检查点发布 |
| Recirculation | 按 token 的 prefill 重放（原始 KV） |
| OverRIDE | 迭代重加权与真实采样 token 反馈 |
| FTTT | 编辑学习、宿主反馈与有界重试 |
| GTTA | CPU 环境评分与事务化更新 |
| TTRL | 多数奖励学习与重试/栅栏输出 |

它们没有特权：全部内置算法使用同一套图、签名计划与执行所有者。新算法实现的
是[创作契约](doc/reference/authoring-contract.md)，而不是自己的执行器或 arena。
见[架构与支持边界](doc/design/architecture.md)与
[In-Place-TTT 语义](doc/design/inplace-ttt.md)。

## 安装

需要 Python 3.11+、Rust 工具链，以及相互匹配的 PyTorch/CUDA 包。
先装好平台匹配的 PyTorch 与 vLLM wheel，再安装 PLLM：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[vllm027,test]"
```

当前后端面向未量化的 Qwen2 模型与 vLLM 0.27.1：默认 MRv2，TP1/PP1/DP1，
非投机执行，采样 `n=1`。MRv1 支持共享附着路径但不支持 KV 重放。
本仓库的本地环境是 PyTorch 2.13/CUDA 12.9 与 vLLM 0.27.1+cu129——请选择
相互匹配的 wheel，不要假设任意上游 CUDA 变体都兼容。仅 CPU 开发可安装
`.[test]`。不安装任何全局 vLLM 插件。

`multiprocessing=True` 是 EngineCore 生产部署路径：负载按**引用**传输——
类坐标（模块 + 限定名）、模块摘要与冻结字段值，从不序列化函数体——worker
进程重新导入、重建并重新绑定，安装时其编译签名必须与驱动端签名计划逐字节
相等。内置负载与用户负载走同一条路（内置即最佳实践参考）。代价是一条纪律：
workload 类必须可导入（不能是 `__main__`/notebook 里的临时类）、是冻结
dataclass、字段可封闭传输（原始配置值、typed contract、模块级函数引用或
WorkerTransportable）。默认 `multiprocessing=False` 进程内执行没有这些要求。

`ExecutionOptions(capture=True)` 捕获的是受支持的训练内核，不是整个模型。
模型维度与有界资源事实在签名前完成特化；`training_rows` 限制每个状态的
外部训练行数，`capture_memory_bytes` 是每个任务的驻留上限。容量不足是
错误，不是静默退回 eager。Overlay 不能使用共享前缀缓存。

## 基准

内部场景使用与公开负载完全相同的 API：

```bash
.venv/bin/python -m tools.benchmark_esamp --batch 8 --tokens 256 --rounds 3
.venv/bin/python -m tools.benchmark_lora --batch 8 --tokens 256 --rounds 3
```

两者都在等效工作量下对比 eager 与捕获执行；LoRA 分别跑共享状态与独立状态。
它不再是脱离框架的内核微基准。[测量细节](doc/guides/benchmarking.md)说明了
负载构成、限制与历史对比的注意事项。

## 下一步去哪

- 想写自己的负载 → [使用指南](doc/README.md#使用指南任务导向)（任务导向）
- 想查某条声明的精确边界 → [负载创作契约](doc/reference/authoring-contract.md)与[词汇表](doc/reference/glossary.md)
- 想理解架构与所有权 → [设计说明](doc/README.md#设计说明解释导向)
- 想做验证与发布检查 → [验证指南](doc/guides/validation.md)

开发使用 `/workspace/pllm/.venv/bin/python`。
[文档索引](doc/README.md)汇总维护中的设计文档。
