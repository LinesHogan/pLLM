# PLLM

可编程的张量负载，与 LLM 推理并肩运行。算法只声明数学、数据流、状态与效果；
Rust 签名出唯一的执行计划，托管运行时负责版本、发布、捕获与完成。
框架不认识任何算法名 —— 约束只来自形状、dtype、依赖、效果、设备与容量。

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

最好的入门路径是 `doc/tutorials/` 下的六篇中文 notebook，按顺序读，
每一篇都可在 GPU 上直接执行（环境准备见[安装](#安装)）：

| 教程 | 你会学到什么 |
| --- | --- |
| [00 快速上手](doc/tutorials/00_quickstart.ipynb) | PLLM 的心智模型：声明一个 Workload、让 Rust 签名、交给托管运行时。最小观测负载 + trace 读取 |
| [01 观测：给三层 hidden 装 SAE](doc/tutorials/01_observe_sae.ipynb) | 观测任务、导出语义（`max_rows`/`max_cols`）、冻结状态、后台线程消费 trace、落盘 pkl |
| [02 自定义训练头](doc/tutorials/02_custom_head.ipynb) | 训练任务与采样引导、loss 约定、参数按行就位的 `bmm` 写法；把头从 MLP 换成 CNN |
| [03 每请求 LoRA](doc/tutorials/03_request_lora.ipynb) | 参数 overlay、route 键共享 lineage、判别器混合两个头、prefill 逐块学习（波前 vs Jacobi） |
| [04 三种循环](doc/tutorials/04_loops.ipynb) | 层间重放、逐步 hidden 再循环、跨请求会话循环（memory + Capture） |
| [05 decode 时在线训练](doc/tutorials/05_decode_training.ipynb) | 跨步与跨请求的训练目标：当前步、下一步、上一 VIP 请求终态，全靠输入声明组合 |

教程的读法假设你像读 PyTorch/CUDA 教程一样**边跑边改**：每篇从一个能跑的
最小例子出发，逐步加概念，最后落到完整场景。

## 自己写负载

把需求交给编码代理？先看[负载需求模板与代理指南](doc/guides/agent-workloads.md)：
要提供什么、参考哪些例子、代理应该交付什么。

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

这段代码声明的是捕获与一个输出效果，不是运行时钩子或图节点。完整的
[负载创作契约](doc/guides/authoring.md) 涵盖参数作用域、注解输入、选择、
一阶梯度边、激活检查点与带条件重放的请求 overlay。

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

全部内置算法使用同一套图、签名计划与执行所有者。新算法实现的是
[创作契约](doc/guides/authoring.md)，而不是自己的执行器或 arena。
见[架构与支持边界](doc/architecture.md)与 [In-Place-TTT 语义](doc/inplace-ttt.md)。

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
`multiprocessing=True` 会为封闭的内置负载定义启用 EngineCore；任意用户
Python 类不会在 worker 里被序列化或动态导入。

`ExecutionOptions(capture=True)` 捕获的是受支持的训练内核，不是整个模型。
模型维度与有界资源事实在签名前完成特化；`training_rows` 限制每个状态的
外部训练行数，`capture_memory_bytes` 是每个任务的驻留上限。容量不足是
错误，不是静默退回 eager。Overlay 不能使用共享前缀缓存。本仓库的本地
环境是 PyTorch 2.13/CUDA 12.9 与 vLLM 0.27.1+cu129 —— 请选择相互匹配的
wheel，不要假设任意上游 CUDA 变体都兼容。仅 CPU 开发可安装 `.[test]`。
不安装任何全局 vLLM 插件。

## 基准

内部场景使用与公开负载完全相同的 API：

```bash
.venv/bin/python -m tools.benchmark_esamp --batch 8 --tokens 256 --rounds 3
.venv/bin/python -m tools.benchmark_lora --batch 8 --tokens 256 --rounds 3
```

两者都在等效工作量下对比 eager 与捕获执行；LoRA 分别跑共享状态与独立状态。
它不再是脱离框架的内核微基准。[测量细节](doc/guides/benchmarking.md)说明了
负载构成、限制与历史对比的注意事项。

开发使用 `/workspace/pllm/.venv/bin/python`。
[验证指南](doc/guides/validation.md)列出聚焦检查与发布门禁；
[文档索引](doc/README.md)汇总维护中的设计文档。
