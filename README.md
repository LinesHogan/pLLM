# pllm：可编程的 LLM

你写了一个小算法：比如一个引导向量，让模型的语气偏向积极；或者一个每个请求自带的小网络，边生成边训练，用它的预测去调整采样。在 Hugging Face 的 `generate` 里，几十行代码就跑通了。

下一个问题是：放进真实的推理服务会怎样？真实服务里几十上百个请求挤在一个 batch 里连续批处理，模型被 CUDA graph 捕获，每个请求的 token 被拼进同一个张量。这时候你会接连撞上几件事：

- forward hook 不知道张量的哪一行属于哪个请求，在 CUDA graph 下甚至不会被调用；
- 每个请求要有自己的一份参数和优化器状态，请求来了要分配显存、走了要回收；
- 训练如果和生成串行，吞吐会掉一大截，要重叠就得自己管 CUDA stream；
- 最后还得让这一切在 CUDA graph 下跑。

到了这一步，研究者往往被迫变成推理引擎工程师：拆开 vLLM 的模型代码，自己做这些事。

pllm 把这部分接了过来：你声明读哪一层、存什么状态、结果用来做什么，再写自己的张量数学；pllm 在 vLLM（0.27.1 / 0.32.0）里运行它，负责每个请求的状态和显存、CUDA graph 捕获、和 forward 的重叠、优化器更新。

我们的目标：让天下没有难做的 LLM modeling 算法。

## 尝鲜：十几行，一个请求一个在线训练的小网络

下面是 ESamp 的核心：每个请求在生成的同时训练一个小网络，用第 0 层的 hidden 预测最后
一层的 hidden；采样时把这个浅层预测从模型的 logits 里对减一部分，压低"浅层就能猜到"的 token。

```python
import torch
from torch import nn
from vllm import SamplingParams

import pllm


class ESampLite(pllm.Workload):
    distiller = pllm.RequestState(
        module=lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size)),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    @pllm.guide_sampling(beta=0.5)
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)


llm = pllm.make_llm(pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=128), workload=ESampLite())
outputs = llm.generate(["Explain gradient descent:"] * 8, SamplingParams(n=16, max_tokens=256))
```

这段代码里没有钩子、没有 CUDA 流、没有显存管理。128 个并发请求各自拥有一份小网络和
AdamW 状态；反向传播和优化器更新被捕获进 CUDA graph，并在旁路 stream 上和下一步 forward
重叠。[教程 03](doc/tutorials/03_decode_training.ipynb) 在一张 RTX 4090 上测了同样结构的负载
（8 个 prompt × 16 个样本 × 256 token）：只训练不引导采样时，吞吐是原生 vLLM 的 95%；
同时引导采样时是 84%。

[教程 08](doc/tutorials/08_inplace_ttt.ipynb) 在同一张卡上跑一个 4B 的 In-Place TTT 社区检查点（每个请求一份
随 token 更新的快权重）：单个请求、prompt 4096 / 16384 token 时，pllm 是 57.95 / 43.76 token/s，官方
Hugging Face 实现是 53.89 / 34.20 token/s；官方实现一次只能跑一个请求，pllm 同时跑 4 个请求合计 125.31 token/s。

## 能做什么

一个算法适合 pllm，当它在推理时对模型做的改动属于下面四类之一（可以组合）：

| 你的算法需要…… | 例子 | 教程 |
| --- | --- | --- |
| 在已有 forward 的某个位置**读或加一点计算** | 激活引导（CAA / ActAdd）、线性探针与 SAE 监控、ReFT / LoReFT 干预、在 logits 上加修正、读残差流与逐层 logits（logit lens、DoLa）、改写残差连接（LAuReL、超连接）或 attention 的 value（value residual） | [00](doc/tutorials/00_quickstart.ipynb)、[01](doc/tutorials/01_observe_probe.ipynb)、[06](doc/tutorials/06_reading_inside.ipynb)、[07](doc/tutorials/07_custom_residual.ipynb) |
| 给每个请求（或整个实例）一份**随 token 演化的状态** | 从 prompt 现场生成 LoRA（Text-to-LoRA 式超网络）、decode 时在线训练（ESamp）、测试时学习（TLM）、某一层旁边的快权重记忆（TTT / Titans 式）、In-Place TTT（把 MLP 的投影当快权重，按块更新） | [02](doc/tutorials/02_request_adapters.ipynb)、[03](doc/tutorials/03_decode_training.ipynb)、[07](doc/tutorials/07_custom_residual.ipynb)、[08](doc/tutorials/08_inplace_ttt.ipynb) |
| 决定**哪些层、哪些 token 再跑一遍**，并说清 KV 怎么处理 | 循环深度 / 潜在推理（Ouro、Huginn 式）、把上一步的 hidden 当下一步输入（Coconut 式连续思维）、层间再循环 | [04](doc/tutorials/04_custom_forward.ipynb) |
| 控制**生成—打分—更新**的循环 | 测试时强化学习（TTRL：多数投票当伪标签）、带重试的反馈循环 | [05](doc/tutorials/05_host_feedback.ipynb) |

不在范围内的：从头预训练、需要全新算子或全新模型结构的改动。

`pllm.workloads` 里有一批现成实现，和你写的负载走完全相同的路径：ESamp、ReFT/LoReFT、
DoLa、TLM（实例级 q/v LoRA 测试时学习）、In-Place TTT（见[教程 08](doc/tutorials/08_inplace_ttt.ipynb)）、
输出层 LoRA、层间再循环、TTRL、OverRIDE、FTTT、GTTA。

## 框架替你做了什么

- **每个请求的状态**：声明 `pllm.RequestState`，每个请求自动得到一份独立参数（和优化器
  状态），请求结束即回收；pllm 在启动时为它自己的状态页和捕获缓冲核对预算（vLLM 自己的
  KV cache 和激活不在这个核对里），预算不够是一个明确的报错，不会悄悄退回慢路径。
- **CUDA graph**：普通的逐行预测、训练（含反向和 AdamW）和状态写入会被自动捕获。
  暂时不能进 graph 的功能（例如激活补丁、`@pllm.custom_forward`）会以 eager 运行，原因写在
  `llm.diagnostics().eager_reason` 里。
- **自动调度**：你只声明每个方法读什么、写什么。pllm 的规划器据此决定每段计算放在哪里、
  能否放到旁路 stream 和下一步 forward 重叠、用哪种实现，你不用写一行同步代码。
- **训练语义**：训练方法只返回 loss；屏蔽填充行、求平均、应用优化器、发布新版本都由框架
  负责。默认情况下，第 N 步的训练结果在第 N+2 步被读到，这一步的延迟换来了重叠；需要严格
  顺序时设 `ExecutionOptions(publish_lag=0)`。
- **启动前先看计划**：`pllm.explain(workload, config)` 在 CPU 上、不加载权重，列出每个方法
  会不会进 CUDA graph、放在主 stream 还是旁路 stream、状态预留多少显存、为什么要退回 eager；
  它不估计吞吐和延迟，能不能装进 GPU 要到 `make_llm` 时才确定。详见[写代码时就看到成本](doc/guides/explain.md)。

## 安装

需要 Python 3.11+、Rust 工具链，以及相互匹配的 PyTorch / CUDA / vLLM。先装好平台匹配的
PyTorch 与 vLLM wheel，再安装 pllm：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[vllm032,test]"   # 或 ".[vllm027,test]"
```

本仓库在 PyTorch 2.13 / CUDA 12.9 上开发，vLLM 0.27.1（`.venv`）与 0.32.0（`.venv-vllm032`）
都在用。只做 CPU 开发可以只装 `.[test]`。pllm 不安装全局 vLLM 插件。

## 支持的模型

pllm 按 vLLM 的具体模型类准入（子类不算，因为 Phi3、Mistral 这类近亲可能改变投影布局或
logits）。

| 模型 | 能用的功能 |
| --- | --- |
| Qwen2 / Qwen2.5、Llama、Qwen3（稠密） | 全部功能 |
| Qwen3-MoE | 只读功能和 `@pllm.custom_forward` 可用；LoRA 类改写可用于 q/v 投影，不能用于专家层；不支持整模型训练（TLM / TTRL 式） |
| Qwen3.5（Gated DeltaNet 混合注意力，含 MoE 版） | 只读功能全部可用；LoRA 类改写只能用于全注意力层的 v 投影和稠密 MLP；不支持 `@pllm.custom_forward` 与整模型训练 |
| DeepSeek V2 / V3（MLA + MoE）、gpt-oss | 只读功能：hidden 读取与导出、探针、ESamp、采样引导、logits 修正、激活补丁 |

- **量化底座是一等公民**：GPTQ、AWQ、FP8 检查点按同一家族的规则准入。LoRA 类改写像 vLLM
  对量化模型做 LoRA 一样，在量化算子的输出上加一份高精度修正，不需要稠密权重；整模型训练
  按 QLoRA 的方式在反向时临时反量化。
- **vLLM 版本**：0.27.1 与 0.32.0，其他版本启动即报错。
- **多卡**：vLLM 0.32.0 的 MRv2 runner 上支持单节点张量并行（`ModelConfig(tensor_parallel_size=2)`）。
  hidden 读取、采样引导、按请求训练的状态（ESamp、输出层 LoRA 等）都能用；LoRA 类改写、整模型训练、
  `@pllm.custom_forward` 目前需要 `tensor_parallel_size=1`，构造时就会提示。

Kimi K2.x 的 INT4 文本塔可以切成六层的 DeepSeek V3 切片做结构验证，见
[Kimi 切片](doc/guides/kimi-slice.md)。

## 下一步去哪

教程是 `doc/tutorials/` 下的中文 notebook，每篇围绕一个你可能听过的算法，在本地
GPU 上大多在 Qwen2.5-0.5B 上跑完（08 用一个 4B 的 In-Place TTT 社区检查点），输出都保留在 notebook 里：

| 教程 | 算法 | 你会学到 |
| --- | --- | --- |
| [00 快速上手](doc/tutorials/00_quickstart.ipynb) | 激活引导（CAA） | 写第一个负载：一份状态 + 一个激活补丁；把自己算好的张量送进去 |
| [01 观测](doc/tutorials/01_observe_probe.ipynb) | 线性探针 / SAE 监控 | 逐 token、逐请求读 hidden，导出到宿主机，不改变输出，不关 CUDA graph |
| [02 每请求 LoRA](doc/tutorials/02_request_adapters.ipynb) | Text-to-LoRA 式超网络 | 每个请求一份 LoRA，由读完 prompt 的超网络现场写入 |
| [03 decode 时训练](doc/tutorials/03_decode_training.ipynb) | ESamp | 每个请求边生成边训练，训练结果引导采样；测吞吐 |
| [04 改写 forward](doc/tutorials/04_custom_forward.ipynb) | 循环深度（Ouro / Huginn 式） | 让上半部分的层每个 token 多跑几遍；KV 怎么处理 |
| [05 反馈循环](doc/tutorials/05_host_feedback.ipynb) | TTRL | 生成、投票、训练在同一个引擎里闭环 |
| [06 读模型内部](doc/tutorials/06_reading_inside.ipynb) | logit lens / DoLa | 读残差流和逐层 logits，不关 CUDA graph；DoLa 的成本 |
| [07 改写残差连接](doc/tutorials/07_custom_residual.ipynb) | value residual / 超连接 / 记忆子层 | `@pllm.custom_residual`：恒等时与原模型逐位相同；eager 的代价 |
| [08 In-Place TTT](doc/tutorials/08_inplace_ttt.ipynb) | In-Place TTT（社区 4B 检查点） | 按块更新的快权重；decode 中跨块更新；与官方实现比吞吐 |

之后：

- 写好负载、启动之前先看计划 → [写代码时就看到成本](doc/guides/explain.md)
- 按任务查写法 → [使用指南](doc/README.md#使用指南任务导向)
- 查某条声明的精确边界 → [负载创作契约](doc/reference/authoring-contract.md)
- 理解架构 → [设计说明](doc/design/architecture.md)
- 跑基准 → [基准解读](doc/guides/benchmarking.md)；做验证 → [验证指南](doc/guides/validation.md)
