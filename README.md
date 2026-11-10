# pLLM

当你有一个推理时算法的想法——无论是通过激活引导（activation steering）修正生成风格，用探针或 SAE 实时监测内部特征，给每个请求维护一份动态 LoRA，还是在解码（decode）时在线训练一个小网络并引导采样（如 ESamp）——在 PyTorch 或 Hugging Face 的单请求环境里，通常写几十行代码、挂一两个 forward hook 就能验证原型。

然而，一旦要把算法放进真实的高并发推理服务（如 vLLM），原型验证就会演变成一场繁重的推理引擎工程攻坚：

- **请求交错在同一个批次**：在连续批处理（continuous batching）中，数十上百个并发请求的 token 被紧密拼接在同一张扁平张量里，传统的 PyTorch hook 无法分辨张量的哪一行属于哪个请求；
- **CUDA graph 旁路了 Python hook**：为了极致吞吐，现代推理引擎普遍依赖 CUDA graph 捕获前向图，在图重放期间 Python hook 根本不会被调用；
- **每个请求需要独立的状态与显存管理**：如果算法带状态（例如每请求参数或优化器动量），必须在请求到达时动态分配显存、请求结束时立即回收，还要避免显存碎片与泄漏；
- **额外计算容易拖垮生成吞吐**：在解码循环中串行运行训练或辅助前向，会导致整体推理吞吐断崖式下跌；要想让额外计算与主前向流水线重叠，必须手动管理底层 CUDA stream 和同步事件；
- **修改注意力与 KV 结构成本高昂**：若算法需要改变某层 attention 的读取策略或按 token 历史查询外部记忆表，往往需要拆开并深度改写引擎底层代码。

面对这些工程屏障，算法研究者往往不得不花费大量时间去钻研 vLLM 底层架构、拆改模型源码。pLLM 把这一整套工程脏活接了过来：你的工作重新回到算法本身——只需**声明**读取模型的哪些内部位置（如特定层的 hidden 或残差流）、持有怎样的状态（如请求级网络与优化器）、计算结果送往何处（如修正 logits、引导采样或写回状态），然后书写纯粹的 PyTorch 张量数学。

pLLM 与 vLLM 的关系，大致就像 eBPF 之于 Linux 内核：你在外部编写小巧的程序，在引擎启动前通过契约检查（contracts），随后装入引擎全速执行；每个程序可以持有自己的状态，计算过程被完整捕获进 CUDA graph，并可通过 `pllm.explain` 和 trace 进行显存与开销观测，而无需侵入或修改底层引擎源码。

## 尝鲜：十几行代码，每请求在线训练一个小网络

下面是 ESamp 的核心逻辑：每个请求在生成的同时在线训练一个小网络，用第 0 层的 hidden 预测最后一层的 hidden；采样时把这个浅层预测从模型的 logits 里对减一部分，压低“浅层就能猜到”的 token。

```python
import torch
from torch import nn

import pllm
from pllm import testing


class ESampLite(pllm.Workload):
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    @pllm.guide_sampling(beta=0.5)
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)
```

在这段代码里，你没有写任何钩子、没有管理 CUDA 流，也没有编写任何显存管理代码。并发请求到来时，每个请求各自拥有一份独立的小网络参数和 AdamW 状态；反向传播和优化器更新被自动捕获进 CUDA graph，并在旁路 stream 上与下一步 forward 重叠执行。

在真正触碰 GPU 之前，有两个 CPU 工具可以在几秒钟内看清执行计划、检查正确性契约（不加载模型权重，无需 GPU）：

```python
config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=128)

# 1. 在 CPU 上检查执行计划与显存预算
print(pllm.explain(ESampLite(), config))

# 2. 在 CPU 上检查正确性契约（验证请求隔离、行置换、抢占重算等）
print(pllm.testing.check(ESampLite(), config))
```

`pllm.explain` 在 CPU 上解析模型配置，展示哪些方法进 CUDA graph、放在哪个 stream、为状态预留多少显存；`pllm.testing.check` 用固定调度在 CPU 上模拟运行几个请求，验证请求之间是否隔离、行顺序打乱或请求被抢占重算时结果是否严格一致。

确认计划与契约通过后，装入引擎即可全速运行：

```python
# check-docs: skip: requires GPU engine
from vllm import SamplingParams

with pllm.make_llm(config, workload=ESampLite()) as llm:
    outputs = llm.generate(["Explain gradient descent:"] * 8, SamplingParams(n=16, max_tokens=256))
```

真实硬件上的吞吐开销如何？[教程 03](doc/tutorials/03_decode_training.ipynb) 在单张 RTX 4090 上测了同样结构的负载（Qwen2.5-0.5B，8 个 prompt × 16 个样本 × 256 token，vLLM 0.27.1）：只在线训练不引导采样时，吞吐与原生 vLLM 相同（该次运行达到 100%）；同时开启训练与引导采样时，吞吐仍保持在原生的 97%。

此外，[示例 08](doc/examples/08_inplace_ttt.ipynb) 在同一张卡上以 eager 模式跑一个 4B 的 In-Place TTT 社区检查点（每个请求一份随 token 更新的快权重）：单个请求、prompt 4096 / 16384 token 时，pLLM 是 61.61 / 42.91 token/s，官方 Hugging Face 实现是 53.89 / 34.20 token/s；官方实现一次只能跑一个请求，pLLM 同时跑 4 个请求合计 123.76 token/s。

## 能做什么

一个算法适合用 pLLM 编写，当它在推理时对模型做的改动属于下面几类之一（这些改动可以互相组合）：

| 你的算法需要…… | 例子 | 教程与示例 |
| --- | --- | --- |
| 在 forward 的某个位置读或加一点计算 | 激活引导（CAA / ActAdd）、探针与 SAE 监控、ReFT / LoReFT、logits 修正、logit lens 与 DoLa、改写残差连接（LAuReL、超连接）或 attention 的 value（value residual） | [00](doc/tutorials/00_quickstart.ipynb)、[01](doc/tutorials/01_observe_probe.ipynb)、[06](doc/examples/06_reading_inside.ipynb)、[07](doc/examples/07_custom_residual.ipynb) |
| 换掉某层 attention 的读法，或改 KV 怎么存 | 注意力监控、Quest 式只读 top-K、滑动窗口与局部/全局交替、保留 prompt 加最近窗口、跨层共享 KV | [10](doc/examples/10_custom_attention.ipynb) |
| 按 token 历史查一张大表 | n-gram 先验、Engram 式查表记忆；表放显存还是主机内存由你决定 | [11](doc/examples/11_lookup_memory.ipynb) |
| 给每个请求（或整个实例）一份随 token 演化的状态 | 从 prompt 现场生成 LoRA（Text-to-LoRA 式）、decode 时在线训练（ESamp）、测试时学习（TLM）、快权重记忆（TTT / Titans 式）、In-Place TTT | [02](doc/tutorials/02_request_adapters.ipynb)、[03](doc/tutorials/03_decode_training.ipynb)、[07](doc/examples/07_custom_residual.ipynb)、[08](doc/examples/08_inplace_ttt.ipynb) |
| 决定哪些层、哪些 token 再跑一遍或跳过 | 循环深度 / 潜在推理（Ouro、Huginn 式）、Coconut 式连续思维、early exit 与按 token 跳层 | [04](doc/tutorials/04_custom_forward.ipynb)、[12](doc/examples/12_adaptive_depth.ipynb) |
| 一步 PyTorch 写不出来或太慢的计算 | 用 Triton 写的 kernel，经 `torch.library.custom_op` 放进负载 | [09](doc/examples/09_custom_kernels.ipynb) |
| 控制生成—打分—更新的循环 | 测试时强化学习（TTRL） | [05](doc/tutorials/05_host_feedback.ipynb) |

完整的清单（包括哪些算法写不了、为什么）见 [能力地图](doc/guides/capability-map.md)。从头预训练全新模型结构、按分数真正释放 KV 显存（如 H2O、SnapKV）等不在范围内。

`pllm.workloads` 里提供了一批现成实现，与手写负载走完全相同的执行路径：ESamp、ReFT/LoReFT、DoLa、TLM、In-Place TTT、输出层 LoRA、层间再循环、多数投票奖励头、OverRIDE、FTTT。

## 安装

需要 Python 3.11+、Rust 工具链，以及相互匹配的 PyTorch / CUDA / vLLM。先安装平台匹配的 PyTorch 与 vLLM wheel，再安装 pLLM：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[vllm032,test]"   # 或 ".[vllm027,test]"
```

只做 CPU 开发可以只装 `.[test]`。pLLM 不安装全局 vLLM 插件。

## 支持的版本与模型

pLLM 目前支持 vLLM 0.27.1 与 0.32.0，准入模型覆盖 Qwen2 / Qwen2.5、Llama、Qwen3、Qwen3.5 以及 DeepSeek V2 / V3 等系列，并原生支持量化底座（GPTQ、AWQ、FP8）和单节点张量并行。各模型支持的具体功能边界、多卡配置与硬件要求详见 [支持矩阵](doc/reference/support-matrix.md)。

## 下一步

- **快速上手**：跟随 [00 快速上手](doc/tutorials/00_quickstart.ipynb) 写出你的第一个负载，再沿主线教程 01–05 往下走；
- **文档概览**：[文档首页](doc/README.md) 按教程、用户指南、操作指南、参考和解释列出全部文档；
- **写自己的负载**：阅读 [负载的组成](doc/guides/workload-anatomy.md)，了解声明、执行时机与数据流；想知道这些规则背后的原因，读 [核心概念](doc/explain/core-concepts.md)；
- **算法示例**：浏览 [示例库](doc/README.md#示例库)，查看从 DoLa、自定义注意力和 Engram 查表到自适应深度的完整实现；
- **编码代理**：如果使用编码代理开发负载，将 [agent skill](doc/agent/SKILL.md) 与 [agent 参考](doc/agent/reference.md) 交给代理（包含完整工作流、契约规则与公开 API 签名）；
- **参与开发**：若要深入底层运行机制或为框架贡献代码，请参阅 [贡献者文档](doc/dev/README.md)。
