# PLLM

PyTorch 会训练，vLLM 会推理。但如果你想让模型在**服务的同时继续学习**——
观测中间层激活、按请求叠 LoRA、在 decode 的间隙更新参数——你会发现两边都不
管这段路：训练框架不认识 KV cache，推理引擎不认识优化器。

PLLM 补上这段路。算法只声明数学、数据流、状态与效果；Rust 签名出唯一的
执行计划，托管运行时负责版本、发布、捕获与完成。框架不认识任何算法名——
约束只来自形状、dtype、依赖、效果、设备与容量。

第一次来？先跑下面的[五分钟起步](#五分钟起步)，再到[教程](#教程从基础到场景)
按兴趣分流。支持哪些模型、有哪些限制，见[支持的模型与限制](#支持的模型与限制)。

## 五分钟起步

```python
from vllm import SamplingParams
from pllm import ExecutionOptions, ModelConfig, make_llm
from pllm.workloads import ESampRequestConfig, ESampRequestWorkload

config = ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", execution=ExecutionOptions())
with make_llm(config, workload=ESampRequestWorkload(ESampRequestConfig(rank=2))) as llm:
    outputs = llm.generate(["Explain gradient descent:"], SamplingParams(max_tokens=32))
    print(outputs[0].outputs[0].text)
    print(llm.diagnostics())
```

省略 `workload` 就是普通的 vLLM 推理。宿主机算法（generate → 打分 → 反馈）是传给
`llm.generate_feedback(feedback=...)` 的普通回调；需要框架训练的张量状态写成普通
`pllm.Workload`，用 `pllm.host.make_host(pllm.host.HostConfig(device, hidden_size=...), workload)`
编译，无需构造模型。

> 迁移说明：Python 导入与发行名现在是 `pllm`，不再安装 `tllm` 兼容包；
> 旧的模块路径与 `canonical_config` 不是兼容 API。既有调用方请迁移到
> `workload=` 与 `config=` 参数。

## 教程：从基础到场景

入门路径是 `doc/tutorials/` 下的中文 notebook，读法和 PyTorch 官方教程一样：
**边跑边改**。每篇从一个能跑的最小例子出发，逐步加概念，最后落到完整场景。
六篇按顺序读，跑完 00 之后可以按兴趣挑 01–05（需要 GPU 环境，见[安装](#安装)）：

| 教程 | 你会学到什么 |
| --- | --- |
| [00 快速上手](doc/tutorials/00_quickstart.ipynb) | PLLM 的心智模型：声明一个 Workload、让 Rust 签名、交给托管运行时。最小观测负载 + trace 读取 |
| [01 观测：给三层 hidden 装 SAE](doc/tutorials/01_observe_sae.ipynb) | 观测任务、导出语义（`max_rows`/`max_cols`）、冻结状态、后台线程消费 trace、落盘 pkl |
| [02 自定义训练头](doc/tutorials/02_custom_head.ipynb) | 训练任务与采样引导、loss 约定、参数按行就位的 `bmm` 写法；把头从 MLP 换成 CNN |
| [03 每请求 LoRA](doc/tutorials/03_request_lora.ipynb) | 参数 overlay、route 键共享 lineage、判别器混合两个头、prefill 逐块学习（波前 vs Jacobi） |
| [04 三种循环](doc/tutorials/04_loops.ipynb) | 层间重放、逐步 hidden 再循环、跨请求会话循环（memory + Capture） |
| [05 decode 时在线训练](doc/tutorials/05_decode_training.ipynb) | 跨步与跨请求的训练目标：当前步、下一步、上一 VIP 请求终态，全靠输入声明组合 |

## 自己写负载

一个负载就是一个声明，不是运行时钩子：

```python
import torch
import pllm

class CenteredHidden(pllm.Workload):
    @pllm.export(name="centered_hidden", every=256, max_cols=16)
    def forward(self, hidden: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return hidden - hidden.mean(dim=-1, keepdim=True)
```

这段代码声明了一个输入（参数默认值 `pllm.hidden(0)`：第 0 层的 hidden）和一个输出效果
（有界导出，在负载的 `on_export` 里收到）。完整的
[负载创作契约](doc/reference/authoring-contract.md) 涵盖参数作用域、输入来源、
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
python -m pip install -e ".[vllm032,test]"   # 或 ".[vllm027,test]"
```

本仓库的本地环境是 PyTorch 2.13/CUDA 12.9 与 vLLM 0.27.1+cu129（`.venv`）、0.32.0+cu129（`.venv-vllm032`）——请选择
相互匹配的 wheel，不要假设任意上游 CUDA 变体都兼容。仅 CPU 开发可安装
`.[test]`。不安装任何全局 vLLM 插件。

`multiprocessing=True` 是 EngineCore 生产部署路径：负载经 pickle 送进 worker 进程，
类和模块级函数按导入路径传递，worker 导入同一份源码重建负载；安装时 worker 的声明
必须与驱动端的计划一致，源码不一致会在安装时报错。内置负载与用户负载走同一条路。
代价是两条纪律：workload 类必须可导入（不能是 `__main__`/notebook 里的临时类），
字段必须能被 pickle（引用模块级函数，不要放 lambda 或闭包）。默认
`multiprocessing=False` 进程内执行没有这些要求。

`ModelConfig(tensor_parallel_size=2)`（或模型支持的 4）在 vLLM 0.32 MRv2 上启用单节点 TP；
省略 `workload` 的普通推理也使用该设置。0.27.1 与 MRv1 仅允许 TP=1；PP/DP 固定为 1，其他值会提示设置 `pipeline_parallel_size=1` / `data_parallel_size=1`。
TP>1 支持 hidden 观测、采样引导/反馈，以及各 rank 上一致训练的 RequestState/InstanceState
（ESamp、TTRL、输出 LoRA 等）；激活补丁仍需 `cuda_graphs=False`。参数 overlay、causal_model/TLM、
Feed、layer_route 和 PositionState 在 TP>1 构造时拒绝，并提示设置 `tensor_parallel_size=1`。
TP>1 始终使用多进程 worker，负载须满足上面的 pickle 传输约束；不能直接访问 `llm.runner` 或 `llm.feed()`。
辅助张量经 CPU 共享内存复制到各 rank；completion 只由输出 rank 写入，默认返回 CPU 张量。
计划和附件不再保存设备 ordinal，绑定使用 worker 的本地设备；旧的序列化计划需要重新编译。
`diagnostics()` / `worker_state()` 检查所有 rank 的逻辑计数，不一致时抛出断言；内存分配与回收时序不参与比较。
`command()` 现在返回按 rank 排序的回复列表（TP=1 也返回单元素列表）；`trace_snapshots()` 返回所有 rank
的 trace，`trace_snapshot()` 保留输出 rank 的视图。观测和导出回调在每个 worker 中运行。
TP>1 的普通推理与负载推理使用相同的 graph capture 行数上限（`max_num_batched_tokens`）。
批次 AdamW 训练的自动 phase 现在默认 AFTER_SAMPLE，与批次 SGD 一致；此前 TTRL 等未显式
标注 phase 的宿主批次负载用于模型后端时会报错，宿主执行的 HOST_COMPUTE 行为不变。

CUDA 后端自动将普通逐行预测、训练和字段写入捕获为 CUDA graph，包含 autograd、
私有页写入和 AdamW；ESamp 使用此通用路径。逐请求训练的方法返回逐行 loss，由框架屏蔽填充行
并执行 `mean` 或 `state_mean`，不能在用户方法中混合行；在宿主批次（`WeightedRowBatch`、
`CausalTrainBatch`）上训练的方法返回一个标量 loss。详见
[训练与观测](doc/guides/training-and-observation.md#capture-an-ordinary-tensor-step)。
`ExecutionOptions(publish_lag=1)` 默认让采样后的任务在框架 side stream 上执行，与下一次
forward 重叠。采样前只读的任务固定读取延迟版本：步 N+1 可读到步 N-1 的训练结果，
训练本身始终从最新版本继续；版本选择不取决于 GPU 完成时机。需要严格顺序的算法使用
`publish_lag=0`，非负整数可配置更长延迟。框架先保存 hidden/feedback，再允许后端复用
这些缓冲区；模型内修改、replay、借用 activation 和跨采样阶段的梯度链仍保持原有顺序。
分块递推（`pllm.scan`，如 In-Place-TTT 的 chunk 递推）默认使用预算内的固定实现；
`ExecutionOptions(tuning="measured")` 让它在首次运行时实测各候选实现并选最快的。
调度层不按算法名字分派。
模型维度与有界资源事实在签名前完成特化；`training_rows` 限制每个状态的
外部训练行数，`capture_memory_bytes` 是每个任务的驻留上限。容量不足是
错误，不是静默退回 eager。Overlay 不能使用共享前缀缓存。
`ModelConfig(cuda_graphs=True)` 是默认值：模型走 vLLM 的 CUDA graph。激活补丁、
`@pllm.layer_route`、`runner="mrv1"` 以及读取 `InstanceState` 的模型内 overlay 不能在 CUDA graph
下运行，`make_llm` 会报错并提示设置 `cuda_graphs=False`。

## 支持的模型与限制

- **模型**：准入按 vLLM 的精确类判断，不接受子类：Phi3、Mistral 等相近架构可能改变投影布局、
  注意力或 logits，尚未逐一验证。示例用 Qwen2.5 检查点只是为了方便。
  - 未量化的 `Qwen2ForCausalLM`、`LlamaForCausalLM`、`Qwen3ForCausalLM` 支持全部功能。
  - 未量化的 Qwen3-MoE 与 Qwen3.5 混合模型支持所有只读功能（见下一条），改写类功能按结构取舍：

    | 家族（vLLM 类） | q/v overlay | MLP down overlay | 层路由、PositionState | 因果训练（TLM） |
    |---|---|---|---|---|
    | Qwen3-MoE（`Qwen3MoeForCausalLM`） | 支持 | 专家层不支持 | 支持 | 不支持 |
    | Qwen3.5（`Qwen3_5ForCausalLM`、`Qwen3_5MoeForCausalLM`） | 只有全注意力层的 v | 稠密层支持，专家层不支持 | 不支持 | 不支持 |

    Qwen3-MoE 的注意力和 KV 与 Qwen3 相同；它的 MLP 由多个专家组成，每个专家有自己的 down 投影，
    按专家做 overlay 还没有设计。Qwen3.5 的大部分层是 Gated DeltaNet，状态是随 token 更新的递归矩阵和
    卷积窗口，不在 KV 页里，而层路由和 PositionState 只保存和重放 KV；这些层也没有 q/v 投影。
    Qwen3.5 的全注意力层把每个头的 query 和输出门交错放在同一个投影里，只有 v 是连续的一段。
    因果训练只实现了稠密 Qwen2、Llama、Qwen3 的前向。Qwen3.5 的官方检查点是多模态的
    （`Qwen3_5ForConditionalGeneration`），需先做成纯文本模型：把 `text_config` 展开为顶层配置，
    `architectures` 改为 `Qwen3_5ForCausalLM`，只保留 `model.language_model.*` 权重并改名为 `model.*`。
  - 量化检查点按上面同一家族的规则准入：层路由与 PositionState 只重跑 vLLM 自己的层模块，量化的
    投影照常执行（已验证 GPTQ-Int4 的 Qwen2.5-14B：路由输出、全部层 KV 与 StepLoop 波次和原生 vLLM
    逐位相等）。参数 overlay 和 vLLM 对量化模型做 LoRA 一样，在量化 kernel 的输出上加一份由输入 x
    算出的 BF16/FP32 修正，不需要稠密权重；零修正时输出与原生 vLLM 逐位相等，CUDA graph 路径同样可用。
    overlay 函数读取 `pllm.operand("weight")` 时，pllm 用 vLLM 自己的反量化工具为该投影生成一份
    稠密副本并常驻（例如 14B 的一个 down 投影约 135 MiB），不读取则没有这份内存。
    因果训练（TLM）暂时仍需未量化的检查点。
  - `DeepseekV2ForCausalLM`、`DeepseekV3ForCausalLM`（MLA 注意力 + MoE）、`GptOssForCausalLM`
    （sink 注意力 + MXFP4 MoE）只支持只读功能：
    hidden tap 与导出、ESamp、采样器引导与反馈、`add_logits`、激活补丁。它们只读取或改写层输出与 logits，
    不碰权重和 KV 缓存；block tap 仍是该层新算出的输出（不含残差），最终位点仍是模型自己的 final norm。
    Kimi K2.5–K2.7 的 INT4 文本塔可用 `tools/model_slice.py --text-only` 展开为上述 DeepSeek V3 类；
    六层切片保留 MLA、全部专家与量化尺度，约 51 GiB，只用于结构与数值验证。见
    [Kimi 切片与 tokenizer](doc/guides/kimi-slice.md)。
  - 参数 overlay、因果训练（TLM）、`@pllm.layer_route` 与 PositionState 需要稠密的 q/v/down 投影和逐层 KV：
    MLA 没有独立的 V 投影，KV 缓存是压缩后的潜变量；MoE 的 down 投影按专家拆分。在这些模型上声明它们，
    会在模型加载后、工作负载绑定前报错并说明原因。
- **vLLM**：0.32.0 或 0.27.1，其他版本启动即报错。默认 MRv2，张量、流水线与数据并行都为 1，
  不支持投机解码。MRv1 支持共享附着路径，但不支持 KV 重放。
- **训练与重放**：因果训练（如 TLM）要求 BF16 权重；层间重放要求 FlashAttention 2，
  其余组合限制见[Overlay 与重放](doc/guides/overlays-and-replay.md)。
- **多样本**：`SamplingParams(n=k)` 把每个 prompt 的 k 个样本作为独立请求运行，样本序号依次为
  `sample_index, …, sample_index+k-1`，按 `(PROMPT_IDX, SAMPLE_IDX)` 键的请求状态互不共享。
- **数值**：重放得到的 carry、tap 归一化结果和 KV 与同形状的原生 vLLM 逐位相等。
  不同批宽、填充或前缀命中会改变 BF16 舍入，所以不同调度之间不保证逐 token 相同，这与原生 vLLM 一致。

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
