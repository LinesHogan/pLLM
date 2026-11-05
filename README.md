# pllm：可编程的 LLM

早期的网页多是服务器拼好的静态文档，加一点脚本就够了；后来网页变成了在线邮箱、在线文档这样的单页应用（SPA），状态多、交互多，手写“数据变了就去改哪块 DOM”的代码撑不住了。React 流行起来的原因是它换了分工：你声明界面在某个状态下长什么样，什么时候更新、怎么更新交给框架。

LLM 推理正在发生类似的事。推理时的算法正在变复杂、变“脏”：带人为先验的查表记忆（DeepSeek 的 Engram 按 n-gram 查一张大表）、稀疏 attention 的索引器以及把选择结果跨层复用、按 token 决定算几层（early exit、循环深度）、推理时训练的快权重（In-Place TTT）。一个具体的数据点：vLLM 0.32.0 为 DeepSeek V4.1 的 Engram 单独写了一个 1740 行的文件。这些算法有状态、有循环、是稀疏的，而且彼此要组合。

pllm 换了同样的分工：你声明数据——读模型的哪里、存什么状态、结果用来做什么——再写自己的张量数学；pllm 负责控制：显存与分页、和 forward 的重叠、CUDA graph、不在每步做 GPU→CPU 同步。约束换来可预期：每个声明在启动前就能检查；成本在写代码时就看得到（`pllm.explain`）；正确性可以检查（`pllm.testing.check` / `check_on_gpu`）；负载可以组合；编码 agent 是一等用户。

我们的目标：让天下没有难做的 LLM modeling 算法。

## 你的算法遇上真实服务

你写了一个小算法：比如一个引导向量，让模型的语气偏向积极；或者一个每个请求自带的小网络，边生成边训练，用它的预测去调整采样。在 Hugging Face 的 `generate` 里，几十行代码就跑通了。

下一个问题是：放进真实的推理服务会怎样？真实服务里几十上百个请求挤在一个 batch 里连续批处理，模型被 CUDA graph 捕获，每个请求的 token 被拼进同一个张量。这时候你会接连撞上几件事：

- forward hook 不知道张量的哪一行属于哪个请求，在 CUDA graph 下甚至不会被调用；
- 每个请求要有自己的一份参数和优化器状态，请求来了要分配显存、走了要回收；
- 训练如果和生成串行，吞吐会掉一大截，要重叠就得自己管 CUDA stream；
- 想换 attention 的读法、想按 token 历史查一张大表，就得改 vLLM 的模型代码和 attention 后端；
- 最后还得让上面这些事都在 CUDA graph 下跑。

到了这一步，研究者往往被迫变成推理引擎工程师：拆开 vLLM 的模型代码，自己做这些事。

pllm 把这部分接了过来：你声明读哪一层、存什么状态、结果用来做什么，再写自己的张量数学；pllm 在 vLLM（0.27.1 / 0.32.0）里运行它，负责每个请求的状态和显存、CUDA graph 捕获、和 forward 的重叠、优化器更新。

## 尝鲜：十几行，一个请求一个在线训练的小网络

下面是 ESamp 的核心：每个请求在生成的同时训练一个小网络，用第 0 层的 hidden 预测最后一层的 hidden；采样时把这个浅层预测从模型的 logits 里对减一部分，压低“浅层就能猜到”的 token。

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

这段代码里没有钩子、没有 CUDA 流、没有显存管理。128 个并发请求各自拥有一份小网络和 AdamW 状态；反向传播和优化器更新被捕获进 CUDA graph，并在旁路 stream 上和下一步 forward 重叠。[教程 03](doc/tutorials/03_decode_training.ipynb) 在一张 RTX 4090 上测了同样结构的负载（8 个 prompt × 16 个样本 × 256 token）：只训练不引导采样时，吞吐是原生 vLLM 的 95%；同时引导采样时是 84%。

[教程 08](doc/tutorials/08_inplace_ttt.ipynb) 在同一张卡上以 eager 模式跑一个 4B 的 In-Place TTT 社区检查点（每个请求一份随 token 更新的快权重）：单个请求、prompt 4096 / 16384 token 时，pllm 是 61.61 / 42.91 token/s，官方 Hugging Face 实现是 53.89 / 34.20 token/s；官方实现一次只能跑一个请求，pllm 同时跑 4 个请求合计 123.76 token/s。

## 能做什么

一个算法适合 pllm，当它在推理时对模型做的改动属于下面几类之一（可以组合）：

| 你的算法需要…… | 例子 | 教程 |
| --- | --- | --- |
| 在 forward 的某个位置读或加一点计算 | 激活引导（CAA / ActAdd）、探针与 SAE 监控、ReFT / LoReFT、logits 修正、logit lens 与 DoLa、改写残差连接（LAuReL、超连接）或 attention 的 value（value residual） | [00](doc/tutorials/00_quickstart.ipynb)、[01](doc/tutorials/01_observe_probe.ipynb)、[06](doc/tutorials/06_reading_inside.ipynb)、[07](doc/tutorials/07_custom_residual.ipynb) |
| 换掉某层 attention 的读法，或改 KV 怎么存 | 注意力监控、Quest 式只读 top-K、滑动窗口与局部/全局交替、保留 prompt 加最近窗口、跨层共享 KV | [10](doc/tutorials/10_custom_attention.ipynb) |
| 按 token 历史查一张大表 | n-gram 先验、Engram 式查表记忆；表放显存还是主机内存由你决定 | [11](doc/tutorials/11_lookup_memory.ipynb) |
| 给每个请求（或整个实例）一份随 token 演化的状态 | 从 prompt 现场生成 LoRA（Text-to-LoRA 式）、decode 时在线训练（ESamp）、测试时学习（TLM）、快权重记忆（TTT / Titans 式）、In-Place TTT | [02](doc/tutorials/02_request_adapters.ipynb)、[03](doc/tutorials/03_decode_training.ipynb)、[07](doc/tutorials/07_custom_residual.ipynb)、[08](doc/tutorials/08_inplace_ttt.ipynb) |
| 决定哪些层、哪些 token 再跑一遍或跳过 | 循环深度 / 潜在推理（Ouro、Huginn 式）、Coconut 式连续思维、early exit 与按 token 跳层 | [04](doc/tutorials/04_custom_forward.ipynb)、[12](doc/tutorials/12_adaptive_depth.ipynb) |
| 一步 PyTorch 写不出来或太慢的计算 | 用 Triton 写的 kernel，经 `torch.library.custom_op` 放进负载 | [09](doc/tutorials/09_custom_kernels.ipynb) |
| 控制生成—打分—更新的循环 | 测试时强化学习（TTRL） | [05](doc/tutorials/05_host_feedback.ipynb) |

完整的清单（包括哪些写不了、为什么）见 [能力地图](doc/guides/capability-map.md)。例子：按分数释放 KV 显存（H2O、SnapKV）不支持；DSA 一类训练出来的稀疏 attention 需要 sm90 以上的 GPU；从头预训练和全新模型结构不在范围内。

`pllm.workloads` 里有一批现成实现，和你写的负载走完全相同的路径：ESamp、ReFT/LoReFT、DoLa、TLM、In-Place TTT、输出层 LoRA、层间再循环、TTRL、OverRIDE、FTTT、GTTA。

## 框架替你做了什么

- **每个请求的状态**：声明 `pllm.RequestState`，每个请求自动得到一份独立参数（和优化器状态），请求结束即回收；pllm 在启动时为它自己的状态页和捕获缓冲核对预算（vLLM 自己的 KV cache 和激活不在这个核对里），预算不够是一个明确的报错，不会悄悄退回慢路径。
- **CUDA graph**：逐行预测、训练（含反向和 AdamW）、状态写入、`@pllm.custom_residual`、`@pllm.custom_attention` 和按 token 历史查表都在 CUDA graph 里运行。`@pllm.custom_forward` 的程序由 pllm 按段调度并捕获自己的 graph（[教程 12](doc/tutorials/12_adaptive_depth.ipynb) 实测，一个不跳层的 custom forward 是原生 CUDA graph 吞吐的 93%）。暂时不能进 graph 的（例如激活补丁）以 eager 运行，原因写在 `llm.diagnostics().eager_reason` 里。
- **自动调度**：你只声明每个方法读什么、写什么。pllm 据此决定每段计算放在哪里、能否放到旁路 stream 和下一步 forward 重叠、用哪种实现，你不用写一行同步代码。
- **训练语义**：训练方法只返回 loss；屏蔽填充行、求平均、应用优化器、发布新版本都由框架负责。默认情况下，第 N 步的训练结果在第 N+2 步被读到，这一步的延迟换来了重叠；需要严格顺序时设 `ExecutionOptions(publish_lag=0)`。
- **写代码时就看到成本**：`pllm.explain(workload, config)` 在 CPU 上、不加载权重，列出每个方法会不会进 CUDA graph、放在哪个 stream、状态预留多少显存；也列出自写算子每行的 FLOP 和字节、每层每个请求的 KV 上限、放在主机内存的表每行要搬多少字节。它不估计吞吐和延迟。详见 [写代码时就看到成本](doc/guides/explain.md)。
- **正确性可以检查**：`pllm.testing.check` 在 CPU 上用几种固定调度跑几个请求：行的顺序变了、请求混在一起、prompt 被切块、请求被抢占后重算，结果都不能变；关闭开关（`neutral=`）时采样不能变。`pllm.testing.check_on_gpu` 加载真实引擎：连续 8 个 decode 步里没有 GPU→CPU 同步、强制驱逐后重算能复现、关闭时与同样设置的原生 vLLM 逐位相同；用了自写算子时还比较 GPU kernel 和 CPU 参考的输出。
- **组合**：一个负载可以把别的负载声明成 dataclass 字段，例如把 `ESampLite` 和一个监控放在一起：

```python
from dataclasses import dataclass, field


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


@dataclass(frozen=True)
class Both(pllm.Workload):
    esamp: ESampLite = field(default_factory=ESampLite)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

每个字段是一个独立的组件，方法和状态带上字段名（explain 里显示为 `esamp.loss`、`esamp.predict`、`monitor.norm`）；真实的冲突会被拒绝，报错点名双方（例如两个组件都用 `@pllm.guide_sampling` 引导采样时，报错会写出这两个方法的名字）；`pllm.testing.check` 多一条“组件独立”的检查。

- **贵的选择由你写**：默认值只替你省掉小的决定；贵的选择写在你的声明里——大表放显存还是主机内存（`placement=`）、KV 怎么存（`keep=` / `window=`）、训练结果几步后可见（`publish_lag`），explain 把代价列出来。

## 给编码 agent 用

- 把 [agent skill](doc/agent/SKILL.md) 和 [agent 参考](doc/agent/reference.md) 交给你的编码代理：工作流、报错修法和全部公开名字，示例都可执行（`check_docs` 会跑它们）。
- `tools/ai_eval`：一套只给文档、在沙箱里让 agent 写负载再打分的评测，用来检查“文档够不够 agent 写对”。
- `examples/algorithm_search`：agent 改写负载、pllm 检查打分、记账、可重放的搜索循环；[教程 13](doc/tutorials/13_algorithm_search.ipynb) 跑了一次，分数差别在噪声范围内，演示的是循环本身。

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

`@pllm.custom_attention` 只用于 KV 全在分页缓存里的模型（稠密 Qwen2/2.5、Llama、Qwen3 和 Qwen3-MoE）。

- **量化底座是一等公民**：GPTQ、AWQ、FP8 检查点按同一家族的规则准入。LoRA 类改写像 vLLM
  对量化模型做 LoRA 一样，在量化算子的输出上加一份高精度修正，不需要稠密权重；整模型训练
  按 QLoRA 的方式在反向时临时反量化。
- **vLLM 版本**：0.27.1 与 0.32.0，其他版本启动即报错。
- **多卡**：vLLM 0.32.0 的 MRv2 runner 上支持单节点张量并行（`ModelConfig(tensor_parallel_size=2)`）。hidden 读取、采样引导、按请求训练的状态（ESamp、输出层 LoRA 等）、`@pllm.custom_forward` 都能用；LoRA 类改写、整模型训练、状态 feed、`@pllm.custom_residual` 和 `@pllm.custom_attention` 目前需要 `tensor_parallel_size=1`，构造时就会提示。

Kimi K2.x 的 INT4 文本塔可以切成六层的 DeepSeek V3 切片做结构验证，见
[Kimi 切片](doc/guides/kimi-slice.md)。

## 下一步去哪

教程是 `doc/tutorials/` 下的中文 notebook，大多在 Qwen2.5-0.5B 上跑完（08 用一个 4B 的 In-Place TTT 社区检查点），输出都保留在 notebook 里：

| 教程 | 算法 | 你会学到 |
| --- | --- | --- |
| [00 快速上手](doc/tutorials/00_quickstart.ipynb) | 激活引导（CAA） | 写第一个负载：一份状态 + 一个激活补丁；把自己算好的张量送进去 |
| [01 观测](doc/tutorials/01_observe_probe.ipynb) | 线性探针 / SAE 监控 | 逐 token、逐请求读 hidden，导出到宿主机，不改变输出，不关 CUDA graph |
| [02 每请求 LoRA](doc/tutorials/02_request_adapters.ipynb) | Text-to-LoRA 式超网络 | 每个请求一份 LoRA，由读完 prompt 的超网络现场写入 |
| [03 decode 时训练](doc/tutorials/03_decode_training.ipynb) | ESamp | 每个请求边生成边训练，训练结果引导采样；测吞吐 |
| [04 改写 forward](doc/tutorials/04_custom_forward.ipynb) | 循环深度（Ouro / Huginn 式） | 让上半部分的层每个 token 多跑几遍；KV 怎么处理 |
| [05 反馈循环](doc/tutorials/05_host_feedback.ipynb) | TTRL | 生成、投票、训练在同一个引擎里闭环 |
| [06 读模型内部](doc/tutorials/06_reading_inside.ipynb) | logit lens / DoLa | 读残差流和逐层 logits，不关 CUDA graph；DoLa 的成本 |
| [07 改写残差连接](doc/tutorials/07_custom_residual.ipynb) | value residual / 超连接 / 记忆子层 | `@pllm.custom_residual`：恒等时与 eager 原生逐位相同；在 CUDA graph 里的吞吐代价 |
| [08 In-Place TTT](doc/tutorials/08_inplace_ttt.ipynb) | In-Place TTT（社区 4B 检查点） | 按块更新的快权重；decode 中跨块更新；与官方实现比吞吐 |
| [09 自写算子](doc/tutorials/09_custom_kernels.ipynb) | 相对引导（Triton kernel） | 用 `torch.library.custom_op` 放进负载；explain 的算子行；check_on_gpu 的设备等价检查 |
| [10 自定义 attention](doc/tutorials/10_custom_attention.ipynb) | 注意力监控 / Quest / KV 布局 | 只换读法；只读 top-K；滑动窗口、prompt 加窗口、跨层共享各省多少 KV |
| [11 查表记忆](doc/tutorials/11_lookup_memory.ipynb) | n-gram 先验 / Engram 式残差 | `pllm.input_tokens`；`pretrained=` 读表；表放显存还是主机内存 |
| [12 自适应深度](doc/tutorials/12_adaptive_depth.ipynb) | 按 token 跳层（early exit） | `active=`；跳过的层怎么补 KV；什么时候真的省 |
| [13 让 agent 搜索算法](doc/tutorials/13_algorithm_search.ipynb) | agent 驱动的搜索循环 | explain + check + 打分 + 重放；怎样判断提升是不是噪声 |

之后：

- 用 agent 开发 pllm workload → 把 [agent skill](doc/agent/SKILL.md) 和 [agent 参考](doc/agent/reference.md)交给你的编码代理：工作流、报错修法和全部公开名字，示例都可执行
- 想知道某类算法能不能写 → [能力地图](doc/guides/capability-map.md)
- 写好负载、启动之前先看计划 → [写代码时就看到成本](doc/guides/explain.md)
- 按任务查写法 → [使用指南](doc/README.md#使用指南任务导向)
- 查某条声明的精确边界 → [负载创作契约](doc/reference/authoring-contract.md)
- 理解架构 → [设计说明](doc/design/architecture.md)
- 跑基准 → [基准解读](doc/guides/benchmarking.md)；做验证 → [验证指南](doc/guides/validation.md)
