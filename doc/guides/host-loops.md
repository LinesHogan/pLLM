# 宿主循环与批训练

> **先决条件**：[状态](state.md)；教程 [05 反馈循环](../tutorials/05_host_feedback.ipynb)（TTRL、宿主批训练）。
>
> **本文回答**：如何在 Python 宿主侧组织“生成 → 打分 → 训练”闭环（如 TTRL、GRPO、奖励头拟合）？怎样向引擎喂入外部批次，或者在没有模型引擎时单测宿主状态？

## 动机：为什么需要宿主批训练与反馈循环

在很多测试时强化学习（TTRL、GRPO）、测试时训练（TTT）或自适应搜索算法中，更新逻辑无法在单步 token 的前向过程中就地完成：
1. **依赖全局反馈或完整序列**：例如多数投票（Majority Vote）必须等一组采样的所有答案生成完毕后，统计投票选出伪标签；GRPO 需要拿到组内所有候选序列的奖励以计算优势值并进行组内归一化。
2. **需要训练整模型或外接头**：算法可能需要通过因果语言模型（Causal Model）计算损失反向更新 LoRA 因子，或者仅仅拟合一个轻量级的奖励头（Reward Head）。

如果为此维护两套独立的推理系统与训练框架，每一轮循环都需要在两个系统间跨进程序列化、传输并同步权重，带来巨大的工程复杂度与开销。

在 pLLM 中，宿主代码与引擎使用同一套状态与显存：
- 宿主 Python 循环负责生成请求、计算打分/奖励并组织批次数据；
- 负载通过声明式方法读取宿主送来的批次字段（`pllm.batch(...)`）；
- 引擎在下一次调用 `generate(..., batch=...)` 开始时自动完成一步优化器更新，更新后的参数立刻被后续生成使用；或者通过 `pllm.host.make_host` 在宿主侧独立步进训练。

---

## 核心机制：声明读取宿主批次（`pllm.batch`）

在 `@pllm.train` 装饰的方法中，声明参数的默认值为 `pllm.batch("字段名")`，即可将外部宿主送入的批次注入训练方法。

### 三种批次类型

方法的输入参数读取了哪些 `pllm.batch` 字段，直接决定了该方法消费的批次类型与绑定的优化器：

| 批次类型 | 包含字段 | 绑定优化器与步进规则 | 适用场景 |
| :--- | :--- | :--- | :--- |
| `pllm.host.WeightedRowBatch` | `hidden`, `targets`, `weights` | `pllm.AdamW`，每批一步更新 | 奖励头训练、特征回归、FTTT 等基于隐层向量的快速拟合（可完全在宿主运行） |
| `pllm.host.CausalTrainBatch` | `input_ids`, `labels`, 可选 `attention_mask`, 可选 `weights` | `pllm.AdamW`，每批一步更新 | 因果语言模型训练（如 TTRL、在线 GRPO、TLM），通过整模型对 LoRA 因子做端到端反向传播 |
| `pllm.host.WeightedCEBatch` | `hidden`, `base_logits`, `labels`, `weights` | `pllm.SGD`，配合 `reset=` 执行 `steps=` 步 | 加权交叉熵批量训练（如 Output LoRA），支持单提示词独立批次（`WeightedCEBatch.for_prompts`） |

> [!IMPORTANT]
> 一个训练方法读取的所有 `pllm.batch` 字段必须全部属于同一个批次类型，不能跨批次类型混合读取。

### 因果语言模型端口：`pllm.causal_model`

当算法需要对语言模型本身做端到端自回归训练时，在方法参数中声明：
```text
model: pllm.model.CausalModel = pllm.causal_model("adapter_name")
```
其中 `"adapter_name"` 是负载中声明的 `pllm.model.Adapter` 的属性名。注入的 `CausalModel` 提供以下接口：
- `model(input_ids, attention_mask=None)`：返回注入该 Adapter 后的模型 logits，计算梯度时会直接流向该 Adapter 绑定的状态因子。
- `model.base(input_ids, attention_mask=None)`：返回未注入 Adapter 的冻结基座模型的 logits（通常用于离线重要性权重或基线打分）。
- `model.token_logprobs(input_ids, labels, attention_mask=None, ignore_index=-100)`：分块计算每个位置上下一个 token 标签的对数概率，返回形状为 `[序列数, token 数 - 1]` 的 float32 张量。
  - **为什么分块计算**：自回归模型的词表往往很大（例如 15 万词表）。如果先输出整批的 `[序列数, 序列长度, 词表大小]` logits 张量再求交叉熵，会瞬间消耗数 GB 显存甚至引发显存耗尽。`token_logprobs` 在内部按块计算 LM head 投影，避免整批全词表 logits 在显存中物化。标签为 `ignore_index`（默认 -100）的位置对数概率为 0。

---

## 损失归约约定：方法自行归约并返回单标量

在普通的逐请求训练中，方法为每行返回一个 loss（形状为 `[rows]`），由框架依据 `reduction="mean"` 或 `"sum"` 归约。但在宿主批训练中，规则完全不同：

**宿主批训练的方法必须自行完成损失归约，并直接返回一个标量（或 `pllm.UpdateOutput`）。在 `@pllm.train` 上显式传入 `reduction=` 会在声明期被直接拒绝。**

### 为什么宿主批训练不接受外部 `reduction=`

逐请求训练在每个引擎步处理多少个请求的行，是由连续批处理调度器动态决定的，算法代码无法预知调度结果，因此归约必须交给框架。

而宿主批次是你自己在 Python 宿主端拼装好的完整序列。在序列级强化学习与对齐算法中，序列长度 $|o_i|$ 与候选组大小 $G$ 的归一化方式是算法核心数学的一部分，在线逐 token 生成时无法完成：

| 算法 | 标量损失归约公式 | 归一化特点与数学含义 |
| :--- | :--- | :--- |
| **GRPO** | $\frac{1}{G}\sum_{i=1}^{G} A_i \frac{1}{\|o_i\|} \sum_{t} \ell_{i,t}$ | 先在每条序列内部除以其自身生成长度 $\|o_i\|$ 取平均，再在候选组内 $G$ 条序列间取均值。各序列权重相等，不受长度差异影响。 |
| **DAPO** | $\frac{\sum_{i=1}^{G} \sum_{t} A_i \ell_{i,t}}{\sum_{i=1}^{G} \|o_i\|}$ | 将组内所有序列的所有 token 损失求和后，除以总 token 数。长序列由于 token 数量更多，对总梯度的贡献更大。 |
| **RLOO / REINFORCE** | $\frac{1}{G}\sum_{i=1}^{G} A_i \sum_{t} \ell_{i,t}$ | 序列内部对 token 损失直接求和，组间按候选序列数 $G$ 取均值。 |
| **Dr.GRPO** | $\frac{1}{G}\sum_{i=1}^{G} A_i \frac{1}{L_{\max}} \sum_{t} \ell_{i,t}$ | 用固定的最大长度常数 $L_{\max}$ 代替动态序列长度 $\|o_i\|$，消除长度归一化带来的隐式偏置。 |

由于只有算法作者清楚自己希望采用上述哪种归约公式，且这些公式依赖完整的序列长度与优势权重，因此框架要求方法显式完成归约并返回单标量。

---

## 批次如何进入引擎：`llm.generate(batch=...)`

在典型的“生成 → 投票/打分 → 更新”循环中，训练批次通过 `llm.generate` 送入：

```text
outputs = llm.generate(prompts, sampling, batch=batch)
```

1. **更新时机与可见性**：当传入 `batch` 时，引擎在本次 `generate` 调用的起始阶段执行训练方法并执行一次优化器步进。状态（例如 LoRA 参数）被更新后，本次调用的后续生成 token 会立刻使用更新后的参数。
2. **显存预算：`max_host_training_rows`**：
   - 整模型的反向传播需要大量的临时激活值显存。
   - pLLM 在引擎初始化时根据 `ExecutionOptions(max_host_training_rows=N)` 静态预留这部分工作区显存，确保在生成途中不会因为临时训练反向而突发显存耗尽。
   - 如果送入的批次行数超过了 `max_host_training_rows`，引擎会在执行前直接报错拒绝。
3. **初次编译耗时**：
   - 首次遇到一个此前未见过的批次形状（例如序列数、最大 token 长度发生变化）时，pLLM 需要编译该形状的训练计算图，该轮用时会明显延长。
   - 批次形状稳定后，后续训练复用已编译的图，恢复正常运行速度。

### 因果批训练的约束与原因

当使用 `CausalTrainBatch` 与 `pllm.causal_model` 时，底层有以下明确要求：
1. **状态类型**：必须更新 `InstanceState`，并配置 `pllm.AdamW` 优化器。原因：因果批训练作用于整模型的前向与反向，更新的是所有请求共享的模型权重增量。
2. **基座模型数据类型**：基座模型必须为 BF16（`bfloat16`），且词表嵌入层（`embed_tokens`）与输出头（`lm_head`）不能被量化。原因：自动微分必须以 BF16 精度穿透词表嵌入与 LM head，量化层无法直接支持浮点梯度反向传播。
3. **单卡执行**：必须配置 `tensor_parallel_size=1`。原因：因果整模型的前向与反向目前在单卡上下文内完整执行，多卡张量并行下的跨卡反向传播与梯度聚合目前尚未支持。
4. **低秩初始化**：使用 LoRA 适配器时，必须显式提供 `init=pllm.Computed(...)`（例如 $A$ 随机初始化、$B$ 为零）。原因：若 $A$ 与 $B$ 均初始化为零，则两者的初始梯度均恒为零，LoRA 永远无法被优化器更新。

---

## 纯宿主状态：无需模型引擎的 `pllm.host.make_host`

如果一个状态只被宿主代码读取与训练（例如从生成的最终隐层向量学习打分的奖励头、外部文本分类器），它完全不需要启动占用 GPU 的 vLLM 模型引擎。

你可以使用 `pllm.host.make_host` 将负载编译为一个宿主训练器 `HostTrainer`，它完全支持在 CPU 上单步调试与运行。

### 可运行示例：CPU 上的 RewardHead

以下代码完整演示了如何声明一个奖励头负载，并使用 `make_host` 在 CPU 上对 `WeightedRowBatch` 执行 AdamW 训练：

```python
from dataclasses import dataclass
import torch
import pllm
from pllm.host import HostConfig, HostTrainer, TrainStep, WeightedRowBatch, make_host

@dataclass(frozen=True)
class RewardHead(pllm.Workload):
    head = pllm.InstanceState({"w": pllm.Field(lambda width, _: (width,))}, optimizer=pllm.AdamW(1e-2))

    @pllm.train(state="head")
    def fit(
        self,
        hidden: torch.Tensor = pllm.batch("hidden"),
        targets: torch.Tensor = pllm.batch("targets"),
        weights: torch.Tensor = pllm.batch("weights"),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> torch.Tensor:
        error = (hidden @ w - targets).square()
        return (weights * error).sum() / weights.sum().clamp_min(1)


with make_host(HostConfig(torch.device("cpu"), hidden_size=896), RewardHead()) as head:
    batch = WeightedRowBatch(torch.randn(4, 896), torch.ones(4), torch.ones(4))
    step = head.train(batch)
    print(step.loss is not None, tuple(step.parameters["w"].shape), head.version)
```

- `make_host(config, workload)`：编译负载并初始化其实例状态。
- `head.train(batch)`：执行一步训练，返回 `TrainStep`（包含标量 `loss` 以及当前更新后的参数字典 `parameters`）。
- `head.version`：该状态成功发布的版本计数器（从 0 开始递增）。
- `head.diagnostics()`：返回当前的训练诊断统计（`loss_count`、`update_count`、`trained_rows` 等）。
- `head.close()` 或使用 `with` 上下文管理器：退出时释放相关资源。

---

## 宿主端因果训练定义与成本推导（`pllm.explain`）

在向 GPU 启动因果批训练前，你可以通过 `pllm.explain` 在 CPU 上推导 TTRL 负载的执行计划与显存开销：

```python
import math
from collections.abc import Mapping

def lora_init(fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    for name, value in fields.items():
        if name.startswith("a_"):
            torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
        else:
            value.zero_()


@dataclass(frozen=True)
class TTRL(pllm.Workload):
    rank: int = 8
    learning_rate: float = 1e-4

    @property
    def lora(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            optimizer=pllm.AdamW(self.learning_rate),
            init=pllm.Computed(lora_init),
        )

    @property
    def adapter(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj"), pllm.model.Projection("v_proj"))
        return pllm.model.Adapter("lora", targets, operator="low_rank", scale=2.0)

    @pllm.train(state="lora")
    def learn(
        self,
        input_ids: torch.Tensor = pllm.batch("input_ids"),
        labels: torch.Tensor = pllm.batch("labels"),
        attention_mask: torch.Tensor | None = pllm.batch("attention_mask"),
        model: pllm.model.CausalModel = pllm.causal_model("adapter"),
    ) -> torch.Tensor:
        logprobs = model.token_logprobs(input_ids, labels, attention_mask=attention_mask)
        return -logprobs.sum() / labels[:, 1:].ne(-100).sum()


config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=640,
    max_num_seqs=8,
    execution=pllm.ExecutionOptions(max_host_training_rows=8),
)
report = pllm.explain(TTRL(), config)
print(report.states[0].name)
```

`pllm.explain` 完全在 CPU 上运行，不加载真实模型权重，也不占用 CUDA 存储，用于在编码阶段静态验证负载结构与显存预算。

---

## 闭环构建：重试循环与 `llm.generate_feedback`

pLLM 提供了两种构建宿主循环的形态：
1. **显式全局循环（`llm.generate(batch=...)`）**：每轮对所有提示词重新生成样本，宿主计算全局奖励并组织 `CausalTrainBatch`，在下一次生成开始时更新模型权重（如 TTRL / GRPO）。
2. **细粒度重试循环（`llm.generate_feedback(...)`）**：针对需要多次重试部分未通过样本的场景（如数学自洽性重试、FTTT），支持局部重试与同步栅栏。

### 细粒度重试机制：`generate_feedback`

`llm.generate_feedback` 的调用模式如下：

```python
# check-docs: skip: GPU engine and CUDA are required
from vllm import SamplingParams

from pllm.workloads import DummyWorkload
from pllm.workloads.reward_head import RewardHeadConfig, RewardHeadFeedback, RewardHeadWorkload

gpu_config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)
prompts = ["What is 2 + 3? Put the final answer in \\boxed{}.", "What is 7 * 6? Put the final answer in \\boxed{}."]
sampling = SamplingParams(n=4, max_tokens=64, temperature=0.8, seed=0)  # 每个 prompt 4 个样本参与投票
head_workload = RewardHeadWorkload(RewardHeadConfig(rank=16, learning_rate=1e-3))
host_cfg = HostConfig(torch.device("cuda"), hidden_size=896)

# generate_feedback 要拿每个回答的最终 hidden，引擎需要带一个负载；只做观测的 DummyWorkload 就够了
with pllm.make_llm(gpu_config, workload=DummyWorkload()) as llm, make_host(host_cfg, head_workload) as head_trainer:
    feedback_handler = RewardHeadFeedback(head_trainer, head_workload.config)
    for completion_input, vote_result in llm.generate_feedback(
        prompts,
        sampling,
        feedback=feedback_handler,
        group_id="vote_group_1",
        max_rounds=3,
    ):
        print(f"本轮样本数: {len(completion_input.completions.entries)}, 更新次数: {vote_result.update_count}")
```

- **输入参数**：
  - `prompts`：待评测的提示词列表。
  - `sampling`：采样参数（`SamplingParams`），每个 prompt 采 `n` 个样本。重试按样本进行：被标为 `RETRY` 的样本下一轮单独重采（`n=1`），并保留它在本 prompt 样本中的位置——第 k 轮的第 j 个样本编号为 `k*n + j`。设了 `seed` 时，编号为 i 的样本用 `seed + i`，所以重试会抽到新的样本。
  - `feedback`：用户定义的回调函数，类型为 `Callable[[CompletionInput], FeedbackT | None]`。
  - `group_id`：本批请求所属的组标识符。
  - `max_rounds`：最大迭代轮数。
- **回调接收 `CompletionInput`**：
  - `completions`（`CompletionBatch`）：包含本轮采样的全部 `Completion` 对象（每个包含 `identity`、`text`、`token_ids`、`finish_reason` 等）。
  - `hidden`：每个生成样本最终位置的隐层激活向量（`torch.Tensor`，形状为 `[样本数, hidden_size]`，float32 精度）。
- **回调返回结果**：返回的对象需要包含以下属性（若返回 `None` 则提前终止循环）：
  - `retry_controls`（`tuple[RetryControl, ...]`）：对每个样本的调度决策。每个 `RetryControl` 包含 `action`（`RetryAction.ACCEPT` 接受、`RetryAction.RETRY` 重试、`RetryAction.STOP` 停止）、`reason`（`ControlReason` 原因）以及 `retry_after_update`（布尔值，是否必须等待更新完成）。
  - `update_barriers`（`tuple[UpdateBarrier, ...]`）：同步栅栏，声明一组请求（`request_ids`）与状态（`BarrierState.READY` 或 `BLOCKED`），仅当栅栏就绪且 `update_count > 0` 时，依赖更新的请求才会被重新采样。
  - `update_count`（整数）：本轮是否产生了参数更新。
- **内置辅助工具**：
  - `majority_vote(value: CompletionInput, config: RewardHeadConfig)`：自动提取数学题 `\boxed{}` 中的答案并进行多数投票，一致者标记为 ACCEPT，不一致者标记为 RETRY，同时构造 `UpdateBarrier` 与用于训练奖励头的 `WeightedRowBatch`。
  - `pllm.workloads.RewardHeadFeedback(head, config)`：封装好的反馈回调，内部调用 `majority_vote`，并在 `head.train(vote.batch)` 训练奖励头后返回投票结果。

### 选型建议：何时用 `generate(batch=)`，何时用 `generate_feedback`

- **使用 `generate(batch=)`**：算法每轮需要评估当前策略对全部提示词的整体表现（例如在线强化学习、TTRL、GRPO），每轮重新采样所有请求，训练的是注入模型前向的适配器（如 LoRA）。
- **使用 `generate_feedback`**：算法旨在解决一批具体任务（如推理链自我修正、带判别器的生成重试），只要某个提示词给出了合格答案就立即退出，仅对失败的请求进行重试；或者模型本身保持冻结，仅更新宿主侧维护的奖励头或打分器。

---

## 空选择语义（Empty Selection）

在实际反馈循环中，有时一整批候选样本可能均未通过筛选（例如全票一致导致 GRPO 优势全为 0，或者没有一个样本提取出了有效答案）。

此时训练方法可以通过以下方式返回：
1. 返回 `UpdateOutput(loss, changed, update_count, trained_rows)`，其中 `changed = False`（或设备上的 0 维布尔张量），`update_count = 0`；
2. 或者在损失中使用 `torch.where(total_weights > 0, loss, 0.0)`，权重为 0 时损失为 0。

**行为保证**：当选择为空（`changed` 为假）时，pLLM 自动发布未更改的状态参数，优化器内部状态（动量、步数等）保持不变。更重要的是，`changed` 标志作为一个设备张量传递，整个过程**不需要发生 CPU 与 GPU 之间的同步等待**，保障流水线吞吐。

---

## 完整 GPU 循环示例（TTRL 多数投票）

以下是在真实的 GPU 环境中运行 TTRL 的循环代码（来自 [教程 05](../tutorials/05_host_feedback.ipynb) 的精简版本）：

```python
# check-docs: skip: GPU engine and CUDA are required
from collections import Counter
import re
from vllm import SamplingParams
from pllm.host import CausalTrainBatch

def get_boxed(text: str) -> str | None:
    found = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return found[-1].strip() if found else None

def prepare_batch(outputs, prompts, tokenizer) -> CausalTrainBatch | None:
    candidates = []
    for prompt, output in zip(prompts, outputs):
        answers = [get_boxed(s.text) for s in output.outputs]
        counts = Counter(a for a in answers if a is not None)
        if not counts:
            continue
        majority, _ = counts.most_common(1)[0]
        sample = next(s for s, a in zip(output.outputs, answers) if a == majority)
        candidates.append((tokenizer(prompt).input_ids, list(sample.token_ids)))
    if not candidates:
        return None
    width = max(len(p) + len(a) for p, a in candidates)
    ids = torch.full((len(candidates), width), tokenizer.pad_token_id, dtype=torch.int64)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for row, (p, a) in enumerate(candidates):
        ids[row, : len(p) + len(a)] = torch.tensor(p + a)
        labels[row, len(p) : len(p) + len(a)] = torch.tensor(a)
        mask[row, : len(p) + len(a)] = 1
    return CausalTrainBatch(ids.cuda(), labels.cuda(), mask.cuda())

gpu_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=640,
    max_num_seqs=128,
    max_num_batched_tokens=2048,
    gpu_memory_utilization=0.25,
    seed=0,
    execution=pllm.ExecutionOptions(max_host_training_rows=8),
)
sampling = SamplingParams(n=8, temperature=0.8, top_p=0.95, max_tokens=256, seed=0)

with pllm.make_llm(gpu_config, workload=TTRL()) as llm:
    batch = None
    for round_index in range(5):
        outputs = llm.generate(prompts, sampling, batch=batch)
        batch = prepare_batch(outputs, prompts, tokenizer)
        print(f"第 {round_index} 轮完成，已更新 {llm.diagnostics().update_count} 次")
```

---

## 常见错误与排查

| 错误信息 | 原因 | 解决办法 |
| :--- | :--- | :--- |
| `host-batch training returns one scalar and owns its reduction; remove reduction=` | 宿主批训练方法传入了非默认的 `reduction=`（如 `"sum"`）。宿主批次的完整序列长度与权重仅在宿主侧已知，损失归约必须在方法内部自行完成并返回标量。 | 去掉 `@pllm.train` 上的 `reduction=`，在方法代码中完成除以长度或样本数的归约。 |
| `training on WeightedRowBatch or CausalTrainBatch fields takes one AdamW step over the whole batch: remove rows=, update_rows=, steps=, reset=, @pllm.differentiate and pllm.carry` | 宿主批训练（`WeightedRowBatch` 或 `CausalTrainBatch`）指定了逐请求训练特有的参数（如 `rows=`、`steps=`、`reset=` 等）。宿主批训练每次对整批执行一步 AdamW。 | 从 `@pllm.train` 装饰器中移除 `rows=`、`update_rows=`、`steps=`、`reset=` 等参数。 |
| `...: add init=pllm.Computed(...) (e.g. random A, zero B); with both low_rank factors at zero every gradient is zero and the adapter never learns` | 低秩适配器（LoRA）的状态字段未显式指定初始化，默认全零（`pllm.Zeros()`）。当 $A$ 与 $B$ 两个因子均为零时，梯度恒为零，优化器无法更新参数。 | 在声明 `InstanceState` 时传入 `init=pllm.Computed(...)`，将 $A$ 因子随机初始化、$B$ 因子初始化为零。 |
| `...: host batch shapes ... exceed the planned ...; raise ExecutionOptions(max_host_training_rows=N) for more training rows per state, or HostConfig(max_input_rows=N) for a host workload` | 送入引擎的批次序列数超过了启动时声明的显存预算上限。框架在启动时按声明的上限预留训练显存。 | 在 `ModelConfig` 中调大 `execution=pllm.ExecutionOptions(max_host_training_rows=N)`，或宿主端调大 `HostConfig(max_input_rows=N)`。 |
| `causal training needs a BF16 model whose embedding and LM head are not quantized` | 因果批训练要求基座模型为 BF16，且词表嵌入与 LM head 未量化。反向传播需要以 BF16 精度穿透嵌入与输出层。 | 使用未量化词表嵌入与 LM head 的 BF16 基座模型。 |

---

## 自测

1. 为什么 `max_host_training_rows` 必须在启动引擎时的 `ExecutionOptions` 中事先声明？

<details><summary>答案</summary>

对整模型执行反向传播需要临时占用大量激活值与计算工作区显存。pLLM 在启动引擎时按照声明的最大批次静态预留这部分显存，避免在生成途中因临时申请大块显存而发生不可预期的显存耗尽。送入超过该行数的批次会被直接拒绝。
</details>

2. 在 TTRL 或 GRPO 的循环中，为什么某几轮的执行耗时明显长于其他轮？

<details><summary>答案</summary>

当送入训练的批次形状（例如序列数、最大 token 长度）发生变化时，pLLM 需要为新的批次形状编译训练计算图，该轮会产生一次性的编译开销。当批次形状稳定后，后续训练复用已编译的图，耗时恢复正常。
</details>

3. 为什么在 `@pllm.train` 方法中不能使用 `reduction="mean"`，而必须在方法内自己做除法？

<details><summary>答案</summary>

不同的序列强化学习与对齐算法（如 GRPO、DAPO、RLOO、Dr.GRPO）对待序列长度与候选组的归一化方式不同（例如是按序列自身长度归一化、按整批总 token 数归一化、还是除以最大长度常数）。这些数学逻辑只能由算法作者在宿主批次上显式计算，框架无法用一个通用的 reduction 参数代表所有算法。
</details>

4. 什么时候应当使用 `llm.generate(batch=...)`，什么时候应当使用 `llm.generate_feedback(...)`？

<details><summary>答案</summary>

当算法每一轮都要重新对全部提示词采样并训练模型内的适配器（如 TTRL、在线强化学习）时，使用 `generate(batch=...)`；当算法执行细粒度的局部重试（例如只重试未通过判别的样本，或者模型冻结只在宿主侧拟合奖励头）时，使用 `generate_feedback(...)`。
</details>

---

## 相关文档

- 教程：[05 反馈循环：TTRL](../tutorials/05_host_feedback.ipynb)
- 指南：[在线训练](training.md) · [状态](state.md) · [负载的组成](workload-anatomy.md) · [看成本：pllm.explain](explain.md)
- 参考：[配置项](../reference/config.md) · [核心概念](../explain/core-concepts.md)
