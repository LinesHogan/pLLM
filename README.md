# pLLM

> **[English README](README.en.md)** · [独立技术博客 (中/英)](blog/README.md) · [文档首页](doc/README.md) · [在线服务指南](doc/howto/online-serving.md) · [Agent 迁移指南](doc/agent/migration.md)

在 PyTorch 里验证一个推理期算法（激活引导、探针监控、每请求 LoRA、decode 时在线训练），通常只需几十行代码；但一旦放进生产级的 vLLM，连续批处理会把并发请求拼进同一张二维张量，CUDA graph 会绕开运行期的 Python hook，每请求状态的分配、同步与回收也都得自己重做。

pLLM 与 vLLM 的关系大致像 eBPF 之于 Linux 内核：你不改 vLLM 源码，只声明**读模型的哪个位置、持有什么状态、计算结果送往哪里**，再写普通 PyTorch 张量数学；pLLM 在启动前完成形状追踪与契约检查，把计算、反向传播与状态管理装进连续批处理和 CUDA graph。

## 30 秒看懂 pLLM 能做什么

| 推理期想做的事 | 直接改 PyTorch / vLLM 的阻碍 | 在 pLLM 里的声明 |
| --- | --- | --- |
| **读取内部激活或逐层 logits**（探针、SAE、Logit Lens、DoLa） | 多请求交错排布，CUDA graph 重放时不触发 Python hook | `pllm.hidden(layer)`、`pllm.residual(layer)`、`pllm.layer_logits(...)`、`@pllm.export` |
| **按请求或按会话改激活 / 权重**（CAA、ReFT、多租户动态 LoRA） | 请求进出时的显存页分配、回收与量化投影叠加容易出错 | `pllm.RequestState` + `pllm.model.ActivationPatch` / `pllm.model.Adapter` |
| **生成过程中在线训练**（ESamp、In-Place TTT、测试时学习、TTRL） | 同步反向传播阻塞主解码流，双流并发又会导致参数读写撕裂 | `@pllm.train` / `@pllm.train_in_forward`（捕获进图，在旁路 stream 与前向重叠） |
| **改写 forward、注意力读法或 KV 布局**（早退跳层、循环深度、滑窗/Sink、大表记忆） | 手改块表与层调度极易破坏分页 KV 的内存不变量 | `@pllm.custom_forward`、`@pllm.custom_attention`、`@pllm.custom_residual`、`pllm.input_tokens` |

一个最小可运行示例（对比第 12 层与末层 logits 的简化 DoLa，无需 GPU 即可在 CPU 上完成规划与契约检查）：

```python
import torch

import pllm
from pllm import testing


class LayerContrast(pllm.Workload):
    @pllm.add_logits
    def contrast(self, logits: torch.Tensor = pllm.layer_logits((12, -1))) -> torch.Tensor:
        return 0.5 * (logits[:, 1] - logits[:, 0])


config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=16)
print(testing.check(LayerContrast(), config))
```

---

## 1 分钟交给 Coding Agent

包内自带面向编码代理（Claude Code、Codex、Cursor 等）的完整工作流规范、API 签名与旧项目迁移清单。安装 `pllm` 后，在你的项目目录执行：

```bash
pllm skill --install claude   # 写入 .claude/skills/pllm/
# 或：
pllm skill --install codex    # 写入 .agents/skills/pllm/
pllm skill --install cursor   # 写入 .cursor/rules/pllm/
# 也可直接打印给任意对话模型：pllm skill --print && pllm skill --print migration
```

随后把下面这段提示交给你的 Agent（改造旧项目或开发新算法均适用）：

> 阅读已安装的 `pllm` skill（先读 `SKILL.md` 与 `migration.md`，按需查 `reference.md` 和 `pitfalls.md`）。若当前仓库已有 Hugging Face hook、`LogitsProcessor`、PEFT 或自定义推理循环，先对照 `migration.md` 做准入审计，明确哪些部分可直接声明、哪些属于不支持的边界；再编写可导入的 `pllm.Workload`，在 CPU 上通过 `pllm.explain(...)` 与 `pllm.testing.check(...)` 后，接入 `pllm.make_llm`、`pllm.make_async_llm` 或 `pllm serve`。

---

## 在线服务与批处理接入

同一个 `pllm.Workload` 可以按场景接入三种运行入口（详见 [在线服务指南](doc/howto/online-serving.md)）：

- **OpenAI 兼容 HTTP 服务（`pllm serve`）**：支持连续请求准入、流式 SSE、客户端断连自动取消与状态回收：
  ```bash
  pllm serve \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --workload pllm.workloads:DoLaWorkload \
    --port 8000 \
    --gpu-memory-utilization 0.85
  ```
  若负载的 `pllm.RequestState` 声明了 `key_fields=("route",)`，客户端通过 `vllm_xargs` 传入会话标识即可在多轮对话间持久复用同一份状态，不同会话彼此隔离：
  ```python
  # check-docs: skip: connects to a running pllm serve endpoint
  from openai import OpenAI

  client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
  stream = client.chat.completions.create(
      model="Qwen/Qwen2.5-0.5B-Instruct",
      messages=[{"role": "user", "content": "用两句话解释 KV cache。"}],
      stream=True,
      extra_body={"vllm_xargs": {"pllm_route": "session-42"}},
  )
  ```
- **嵌入异步 Python 服务（`pllm.make_async_llm`）**：默认的 `ModelConfig(multiprocessing="auto")` 会自动开启独立 worker 隔离，提供 `async with`、`stream(..., route=..., state_import=...)`、`generate()` 与 `abort()`。
- **统一控制面（同步与在线通用）**：`make_llm` 与 `make_async_llm` 均支持并发期间的 `import_state(state_import, route=...)` 热更新、`train_batch(HostBatch)`（含 `CausalTrainBatch`）、跨进程/多卡 `feed(name)` 流投喂、`generate_completions` 与 `generate_feedback`（含无负载模式），详见 [在线服务指南](doc/howto/online-serving.md)。

---

## 深入示例：decode 时每请求在线训练小网络（ESamp）

下面是 ESamp 的核心逻辑：每个请求在生成的同时在线训练一个双层小网络，用第 0 层的 hidden 预测最后一层的 hidden；采样时将浅层预测从词表 logits 中对减一部分，压低“浅层就能猜到”的 token。

```python
from dataclasses import dataclass
from torch import nn


@dataclass(frozen=True)
class ESampLite(pllm.Workload):
    beta: float = 0.5
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

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)
```

并发请求到来时，每个请求各自持有一份小网络参数与 AdamW 动量；反向传播和优化器更新被捕获进 CUDA graph，并在旁路 stream 上与下一步 forward 重叠执行。在接触 GPU 前，先在 CPU 上核对执行计划与多请求隔离契约：

```python
esamp_cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=128)
print(pllm.explain(ESampLite(), esamp_cfg))
print(testing.check(ESampLite(), esamp_cfg))
```

通过检查后，装入 GPU 引擎运行：

```python
# check-docs: skip: requires GPU engine
from vllm import SamplingParams

with pllm.make_llm(esamp_cfg, workload=ESampLite()) as llm:
    outputs = llm.generate(["Explain gradient descent:"] * 8, SamplingParams(n=16, max_tokens=256))
```

实测开销：在单张 RTX 4090 上运行 Qwen2.5-7B-Instruct（8 条 prompt × 16 个样本，各生成 256 token），同样结构的负载只做在线训练、不引导采样时，吞吐是原生 vLLM 的 97.4%；同时开启训练与引导采样时，是原生的 96.3%（完整设置见 [实测记录](doc/reference/measurements.toml)，可在 [教程 03](doc/tutorials/03_decode_training.ipynb) 复现）。

[示例 08](doc/examples/08_inplace_ttt.ipynb) 在同一张卡上运行 4B 的 In-Place TTT 社区检查点（每请求一份随 token 更新的快权重；vLLM 0.32.0，CUDA graph 开启，贪心生成 256 token，预热 1 轮后 3 轮取中位数）：单请求、prompt 4096 / 16384 token 时，pLLM 为 59.64 / 42.27 token/s，官方 Hugging Face 实现按同一协议为 54.79 / 34.89 token/s；官方实现一次只跑单请求，pLLM 并发跑 4 个请求合计 111.38 token/s。

---

## 安装与版本兼容（vLLM 0.27 – 0.34）

需要 Python 3.11+、Rust 工具链（从源码构建时），支持 **vLLM 0.27 至 0.34**（`torch>=2.13,<2.16`、`triton>=3.7,<4`）：

- **装入已有的 vLLM 0.27–0.34 环境**：
  ```bash
  python -m pip install -e . --no-deps   # 若需补齐纯 Python 依赖可用 python -m pip install -e .
  ```
- **从零创建虚拟环境**：
  ```bash
  python -m venv .venv && source .venv/bin/activate
  python -m pip install -e ".[vllm032,test]"   # 亦可选 "[vllm034,test]"、"[vllm027,test]" 或 "[vllm,test]"
  ```
- **仅在 CPU 上编写与检查负载**：安装 `".[test]"` 即可运行 `pllm.explain` 和 `pllm.testing.check`。pLLM 不注册全局 vLLM 插件。

准入模型覆盖 15 类解码器：`Qwen2 / Qwen2.5`、`Llama`、`Mistral`、`Phi-3`、`GLM-4`、`Gemma 2 / 3`、`Mixtral`、`Qwen2-MoE`、`Qwen3 / Qwen3-MoE`、`Qwen3.5`（混合架构含 MoE）、`DeepSeek V2 / V3`（含 MLA 压缩潜空间 `@pllm.custom_attention` 与子层残差）与 `gpt-oss`，支持 GPTQ / AWQ / FP8 量化底座与单节点张量并行（`tensor_parallel_size > 1` 需 `vllm >= 0.32.0`）。其中 `0.27.1` 与 `0.32.0` 为本仓库的完整 GPU 回归基线；`0.33`–`0.34` 依赖的 PyTorch `2.14+cu130` 需要支持 CUDA 13 的 NVIDIA 驱动。各模型与特性的精确边界见 [支持矩阵](doc/reference/support-matrix.md)。

---

## 能力一览与教程

| 你的算法需要…… | 例子 | 教程与示例 |
| --- | --- | --- |
| 在 forward 的某个位置读或加一点计算 | 激活引导（CAA / ActAdd）、探针与 SAE 监控、ReFT / LoReFT、logits 修正、logit lens 与 DoLa、改写残差连接（LAuReL、超连接）或 attention 的 value（value residual） | [00](doc/tutorials/00_quickstart.ipynb)、[01](doc/tutorials/01_observe_probe.ipynb)、[06](doc/examples/06_reading_inside.ipynb)、[07](doc/examples/07_custom_residual.ipynb) |
| 换掉某层 attention 的读法，或改 KV 怎么存 | 注意力监控、Quest 式只读 top-K、滑动窗口与局部/全局交替、保留 prompt 加最近窗口、跨层共享 KV | [10](doc/examples/10_custom_attention.ipynb) |
| 按 token 历史查一张大表 | n-gram 先验、Engram 式查表记忆；表放显存还是主机内存由你决定 | [11](doc/examples/11_lookup_memory.ipynb) |
| 给每个请求（或整个实例）一份随 token 演化的状态 | 从 prompt 现场生成 LoRA（Text-to-LoRA 式）、decode 时在线训练（ESamp）、测试时学习（TLM）、快权重记忆（TTT / Titans 式）、In-Place TTT | [02](doc/tutorials/02_request_adapters.ipynb)、[03](doc/tutorials/03_decode_training.ipynb)、[07](doc/examples/07_custom_residual.ipynb)、[08](doc/examples/08_inplace_ttt.ipynb) |
| 决定哪些层、哪些 token 再跑一遍或跳过 | 循环深度 / 潜在推理（Ouro、Huginn 式）、Coconut 式连续思维、early exit 与按 token 跳层 | [04](doc/tutorials/04_custom_forward.ipynb)、[12](doc/examples/12_adaptive_depth.ipynb) |
| 一步 PyTorch 写不出来或太慢的计算 | 用 Triton 写的 kernel，经 `torch.library.custom_op` 放进负载 | [09](doc/examples/09_custom_kernels.ipynb) |
| 控制生成—打分—更新的循环 | 测试时强化学习（TTRL） | [05](doc/tutorials/05_host_feedback.ipynb) |

完整的边界说明见 [能力地图](doc/guides/capability-map.md) 与 [设计取舍](doc/explain/design-tradeoffs.md)。从头预训练新模型结构、按注意力分数动态释放物理 KV 显存（如 H2O、SnapKV）不在支持范围内。`pllm.workloads` 内置了 ESamp、ReFT/LoReFT、DoLa、TLM、In-Place TTT、输出层 LoRA、层间再循环、多数投票奖励头、OverRIDE 与 FTTT 等参考实现，与用户手写负载走同一条编译和执行路径。

## 下一步

- **动手入门**：从 [00 快速上手](doc/tutorials/00_quickstart.ipynb) 开始，或按 [文档首页](doc/README.md) 查阅用户指南、操作指南与 API 参考（另附面向检索工具的 [llms.txt](llms.txt)）；
- **部署上线**：阅读 [在线服务指南](doc/howto/online-serving.md) 与 [上生产指南](doc/howto/production.md)；
- **独立技术随笔（中/英）**：[随笔目录](blog/README.md) 收录十篇围绕推理期算法演进、调度器复盘、带宽配额、IEEE 754 位变换、编译图快权重训练、静态图指针寻址、显存 MVCC/RCU、跨代际注意力、`attn_metadata` 解包与多租户核验的独立文章。
