# 设计取舍：为什么有些算法今天写不了

在独立 PyTorch 脚本或 Hugging Face `generate` 循环中，研究员只需几行 `register_forward_hook`、在循环体里插入一段条件分支，或者对张量调用 `.item()`，就能迅速验证一项新算法构想。然而，当尝试把同样的算法直接搬入以连续批处理（continuous batching）和 CUDA Graph 异步回放为基石的高吞吐服务引擎（如 vLLM）时，往往会撞上各种各样的物理与架构限制。

pLLM 的定位不是一个收录海量现成算法的黑盒算法库，而是连接前沿建模算法与现代服务引擎之间的**可编程状态化底座（substrate）**。底座的职责是把连续批处理打包、CUDA Graph 静态图捕获、版本化快照发布以及显存生命周期规划等底层工程脏活接管下来，让能够与现代高吞吐物理机制相容的算法以极致性能运行。

当某种算法的设计假设与底层物理机制（固定内存形状、分页块表、流式异步执行、请求间严格隔离）发生根本冲突时，pLLM 明确选择**不静默妥协、不隐式降级**，而是将系统边界和数学原因清晰地暴露给开发者。这一页梳理了目前不支持或受限的 11 项典型设计决策。每一节都按照统一的逻辑展开：**要解决的问题 $\to$ pLLM 的选择 $\to$ 为什么这样选 $\to$ 替代方案**。

---

## 1. 按分数驱逐 KV 显存与 KV Cache 的连续性（H2O、SnapKV）

### 要解决的问题

在长序列生成中，KV 缓存的物理显存占用与上下文长度呈线性增长。以 H2O、SnapKV、PyramidKV 为代表的“按分数动态驱逐”算法，希望在自回归生成过程中持续统计每个历史 token 的累积注意力分数，并据此在物理显存中直接释放那些注意力权重低于阈值的“不重要”token，仅保留开头若干注意力和关键 heavy hitters，从而在理论上把显存占用压到极低水平。

### pLLM 的选择

pLLM 不支持按注意力分数动态释放 KV 物理显存。

### 为什么这样选

vLLM 的底层显存管理模型（PagedAttention）建立在一项核心约定之上：**每个请求的 KV 缓存是逻辑上从位置 0 开始的一整段连续序列**。

1. **逻辑计数与槽位绑定**：引擎仅用一个标量整数记录该请求“已经计算了多少个 token”。每一层每个 token 应该写在物理页块表的哪一页、哪一槽，以及旋转位置编码（RoPE）对应的绝对位置，都是由这个连续递增的计数直接推出的。每个请求在 vLLM 内部的块表只会向后追加分配新页，从不设计从中间抠除某一槽位的机制。
2. **中间留洞破坏分页索引**：如果根据注意力分数丢弃序列中间任意分散的 token，就会在这段连续内存中形成“空洞”。这不仅需要彻底重构底层 PagedAttention 算子的块查找与步长计算逻辑，而且每一步驱逐哪些 token 依赖于当前步运行时的数值计算，无法提前推断。
3. **破坏 CUDA Graph 捕获**：如果在解码的每一步动态释放中间页块、重新压缩排列块表，就必须在 GPU 与 CPU 之间触发阻塞式的元数据同步，或者破坏 CUDA Graph 所依赖的静态张量指针与内存布局，直接摧毁图模式的吞吐收益。

### 替代方案

- **稀疏读取（省算力与访存带宽，不省显存容量）**：使用 [`@pllm.custom_attention`](../guides/forward-structure.md)，在方法体内部调用 `kv.attend(q, topk=...)` 或自定义打分逻辑（如 Quest 算法）。在每一层每个注意力头内，只挑选得分最高的前 $K$ 个位置参与注意力计算，其余位置直接跳过读取。这能成倍降低注意力内核的 FLOP 与访存带宽，完全在 GPU 上运行且兼容 CUDA Graph，详见 [10 自定义注意力](../examples/10_custom_attention.ipynb)。
- **规则驱动的规整物理释放（真正释放物理显存）**：如果目标是真正降低单请求的显存占用上限，应使用与底层块管理模式对齐的静态 KV 布局。例如滑动窗口（`keep="window", window=N`）、保留 prompt 加最近生成窗口（`sink=True`，即 R-SWA），或者奇数层直接复用偶数层 KV 的跨层共享（`keep="none", source=k`）。这些规则在块的边界对齐释放，不需要在块内挖洞。

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing


@dataclass(frozen=True)
class LocalGlobalAttention(pllm.Workload):
    """偶数层只保留最近 256 个 token 的 KV（滑动窗口），奇数层保留全部上下文。"""

    window_size: int = 256

    def window(self) -> int:
        return self.window_size

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=window)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q, window=self.window_size)[0]


CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4)
report = pllm.explain(LocalGlobalAttention(), CONFIG)
```

---

## 2. 压缩远端记忆与分页填充损耗（Activation Beacon）

### 要解决的问题

Activation Beacon 或各类型压缩 KV 算法试图将远端历史上下文（例如过去成百上千个 token）压缩为少量浓缩的表示（Summary Tokens 或压缩条目），以此兼顾长上下文检索与有限的显存容量。

### pLLM 的选择

pLLM 目前未将压缩远端记忆（compressed remote memory）集成到底座的标准流水线中。

### 为什么这样选

将压缩远端记忆深度集成到连续批处理引擎中存在三项物理与算法层面的根本障碍：

1. **分页格式的块填充损耗（Page padding）**：vLLM 的显存按固定大小的“页”（Block，通常为 16 或 32 个 token）分配。将远端 token 深度压缩后存入分页结构时，若压缩比很高（例如每 16 个 token 压缩为 1 个条目），剩余条目仍须补齐整页。整页补齐之后，实际物理显存并未得到实质性削减，反而引入了严重的内存碎片。
2. **分块调度的非确定性（Chunk-dependent state）**：若远端记忆的生成依赖较深层的隐藏状态，在开启分块预填充（`enable_chunked_prefill`）的并发场景下，某个位置的压缩条目在当前步是否已经计算完毕，取决于系统调度器在当前步把 prompt 划分为多大的分块。不同的并发压力会导致切块边界动态变化，进而导致同一输入在不同负载下的压缩特征发生漂移，破坏推理的确定性。
3. **双段注意力的延迟惩罚（Two-pass attention overhead）**：在解码阶段，将近处窗口的全量 KV 与远处的压缩条目分两次执行注意力内核再在 Python/Triton 中拼接合并，其核函数启动开销与访存重排成本远高于直接执行一次原生的全量注意力读（在单请求单层微基准测试中，分两次读合并耗时约 30.31 微秒，而原生全量读仅需 11.88 微秒）。

### 替代方案

- **固定布局削减**：使用滑动窗口（`keep="window"`）或跨层共享（`keep="none", source=k`），它们能够直接按整页物理回收显存，且不引入双阶段注意力的额外调度开销。
- **查表先验注入**：如果需要模型具备长程背景记忆或跨文档先验知识，推荐使用按 token 历史检索的查表记忆机制（参见 [11 查表记忆](../examples/11_lookup_memory.ipynb) 中的 Engram 结构）。通过 `pllm.Pretrained(..., placement="host")` 把大规模预存记忆放在主机内存或独立存储中，前向时跨 PCIe 异步预取并叠加在残差流上，避免干扰 KV 缓存流水线。

---

## 3. DSA 的 Top-K 索引复用与 sm90 硬件要求

### 要解决的问题

在 DeepSeek 稀疏注意力（DSA，如 DeepSeek V3/V4 系列）中，模型使用轻量索引器提前计算 query 与 key 的匹配程度并选出 top-k 个最重要的块或 token。研究员通常希望直接读取该索引器的选择结果（IndexCache），并在后续若干层中直接复用这一索引，避免重复计算打分。

### pLLM 的选择

pLLM 目前不支持读取并跨层复用底层原生 DSA 的 top-k 索引结果。

### 为什么这样选

DSA 是模型结构本身的一部分，要在训练出 DSA 的模型（DeepSeek V3.2、V4 一类）上才有可读的选择结果；而 vLLM 运行这些模型的 attention 路径只支持 sm90 及以上的 GPU（H100 一类）。在 sm80（A100）、sm89（RTX 4090）上根本跑不起来这类模型，也就没有可以复用的 top-k。`pllm.attention(layer, ...)` 今天只提供 `"aux"` 和 `"lse"` 两种读法。

### 替代方案

- 在 sm80/sm89 硬件上，可以在 `@pllm.custom_attention` 中使用自定义 PyTorch 或 Triton 算子实现注意力过滤，例如通过 [09 自写算子](../examples/09_custom_kernels.ipynb) 配合 [10 自定义注意力](../examples/10_custom_attention.ipynb) 中的 `kv.attend(q, topk=...)` 显式计算并指定位置。
- 若只需跨层降低计算量，可直接在模型结构上采用跨层共享 KV 布局（`keep="none", source=k`）。

---

## 4. 批次竞争式路由（MoR Expert-Choice）与原版 MoD

### 要解决的问题

Mixture-of-Depths (MoD) 与 Mixture-of-Recursions (MoR) 旨在为不同的 token 分配不同的计算深度。其中：
1. **MoR 的 expert-choice 路由**：让每一层在整个 batch 内部展开自顶向下的竞争，挑选分数排在前 $k$ 名的 token 执行计算；
2. **原版 MoD**：被跳过的 token 完全不进入该层计算，也不在该层写入任何 KV。

### pLLM 的选择

pLLM 明确不支持批次竞争式的 expert-choice 路由，也不支持原版 MoD 这种完全不写 KV 的跳层模式；pLLM 支持的是基于每行独立掩码的 **token-choice 自适应深度** 与 **早退（Early Exit）**。

### 为什么这样选

1. **绝对禁止破坏请求隔离（Request Isolation）**：在连续批处理中，同一个二维张量里打包着不同客户端发来的完全独立的请求行。如果在整个 batch 里做 top-k 竞争，那么某个 token 到底算不算这一层，将取决于这一瞬间刚好有哪些并发请求被打包在同一个 batch 里。当并发数增减或排队顺序发生毫秒级变化时，同一个 prompt 生成的文本内容就会发生改变，彻底破坏了系统的幂等性与确定性契约（`pllm.testing.check` 会直接捕获并抛出 `ContractError: method ... mixes rows`）。
2. **KV 序列不可留洞**：在原版 MoD 中，如果第 $t$ 个 token 跳过了第 $L$ 层且不写入任何 KV，第 $L$ 层的 KV 序列在位置 $t$ 就会出现缺失。当后续第 $t+1$ 个 token 在第 $L$ 层做自回归注意力时，因果掩码和分页块表预期能够回溯完整的历史序列，缺失的位置会导致注意力因果对齐失效，产生与按分数驱逐 KV 相同的系统性崩溃。

### 替代方案

- **使用 token-choice 自适应路由**：每个 token 仅根据自身的隐藏特征或局部路由判定网络独立决定是否执行某一层（例如通过 sigmoid 阈值独立过滤），不与其他请求争抢配额。
- **GPU 端自动补齐 KV 并在图内调度**：在 `@pllm.custom_forward` 中传入动态布尔掩码 `active=mask`（形状为 `[行, 1]`）。未激活的行跳过该层前向，而 pLLM 会在 GPU 上通过被跳过层的归一化和投影自动计算近似的 K/V 并写入缓存，确保后续 token 永远读到结构完整的 KV 序列；同时，跳层判定完全留在 GPU 上通过 CUDA Graph 的分组分支节点执行，不产生任何 GPU 到 CPU 的同步阻断。详见 [12 自适应深度](../examples/12_adaptive_depth.ipynb)。

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing


@dataclass(frozen=True)
class SkipMiddle(pllm.Workload):
    """token-choice 自适应深度：仅在 decode 阶段根据自身状态决定跳层。"""

    threshold: float = 2.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h12 = layers(0, 12)[11]
        h16 = layers(12, 16, hidden=h12)[15]
        moved = (
            torch.nn.functional.cosine_similarity(h16.float(), h12.float(), dim=-1).unsqueeze(-1)
            <= self.threshold
        )
        h20 = layers(16, 20, hidden=h16, active=moved)[19]
        return layers(20, hidden=h20)[-1]


check = pllm.testing.check(SkipMiddle(), CONFIG)
```

---

## 5. 早退跳层仅限 Decode Token

### 要解决的问题

在自适应深度算法（如 LayerSkip、CALM 等）中，开发者有时希望将早退逻辑推广到 prefill（处理 prompt）阶段，让 prompt 中的部分 token 在浅层及早退出以缩短首字延迟（TTFT）。

### pLLM 的选择

`@pllm.custom_forward` 中的动态跳层掩码 `active=` 严格限制为仅在 `tokens="decode"` 时生效；若在 `tokens="all"` 或 prefill 阶段尝试传入 `active=`，构建期将抛出异常：

```text
Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. Prompt rows run the full model by default; add an unmasked tokens="prefill" method only to change prompt computation (prefer over="span").
```

### 为什么这样选

当某一行跳过后面的层时，pLLM 要为被跳过的层补上近似的 K/V，否则之后的 token 读不到这些位置。
- decode 时被跳过的只是这一个新位置；
- prompt 的每个 token 都是之后所有生成 token 要读的上下文。在 prefill 时跳层，会把整段上下文在高层的 K/V 换成近似值，误差会影响此后生成的每一步。

所以 prompt 默认跑完整模型，`active=` 只用于 `tokens="decode"`。

### 替代方案

- 默认保持 prompt 跑完完整的模型层数，以建立高精度的上下文 KV 缓存基石；仅在解码生成阶段使用 `@pllm.custom_forward(tokens="decode")` 结合 `layers(..., active=mask)` 动态早退。
- 若确实需要在 prefill 阶段改写计算拓扑（例如循环深度或跨度级前向），应声明不带 `active=` 掩码的统一前向方法，或针对特定跨度定制前向（如 `over="span"`）。

---

## 6. MTP、投机解码与解码外环控制

### 要解决的问题

多 Token 预测（Multi-Token Prediction / MTP）、投机解码（Speculative Decoding，如草稿模型投机或 Medusa 多头投机）以及基于树状搜索的可回滚解码（Rollback / Tree Speculation），希望在单步推理中同时预测多个未来 token，或者根据验证结果将请求的生成进度与已写入的 KV 缓存向后回滚若干步。

### pLLM 的选择

pLLM 目前不支持接管外层解码调度循环的算法（包括草稿生成与验证循环、KV 缓存回滚和动态树状分支展开）。

### 为什么这样选

pLLM 的定位是**模型前向内部与单步推理周围的可编程扩展底座**。它介入的生命周期是单步推理时间线（从请求准入、前向计算、隐藏状态读写、残差流改写、注意力拦截到采样前后处理与在线梯度反向）。

投机解码与回滚算法的本质是**改变外层调度循环的控制状态机**：它们需要管理草稿模型的调度流、候选树的分支剪枝、生成位置计数器的减小、以及物理 KV 块表的事务性回滚操作。这些控制逻辑属于服务调度器（Scheduler）与批处理引擎的外环状态机，超出了单步前向建模底座的职责范围。

### 替代方案

- 若生产环境需要通过投机解码提升整体吞吐，应使用 vLLM 原生内置的投机解码服务架构。
- 若需要在当前步改变候选概率或根据中间层预测修正分布，可以使用 `@pllm.add_logits` 在词表上注入修正偏置，或使用 `@pllm.guide_sampling` 进行对比重加权。
- 若算法包含多轮交互、候选重采样或验证反馈，可以在宿主侧构建反馈循环（参见 [05 反馈循环](../tutorials/05_host_feedback.ipynb)），由外部宿主驱动批量生成与参数更新。

---

## 7. 从头预训练新结构不在职责范围内

### 要解决的问题

算法研究员可能希望从零开始发明一种全新的神经网络拓扑，例如一种完全摒弃自注意力机制的新型 token mixer、重新设计的底层 MoE 路由分发器，或者用随机初始化从头预训练一个全新架构的模型。

### pLLM 的选择

从零开始预训练全新架构（Pretraining from scratch）不在 pLLM 的工作范围内。

### 为什么这样选

pLLM 的核心使命在于解决**推理期与测试期算法**（如激活引导、在线微调、自适应计算、快权重记忆、查表残差等）在服务引擎上的落地难题。

从头预训练一个全新的基础模型，需要的是大规模分布式训练集群编排（如 Megatron-LM、FSDP、DeepSpeed）、海量数据集流式加载与容错检查点系统，这是一套完全不同的工程技术栈。pLLM 的全部抽象都围绕“现有预训练权重的服务与适配”展开。

### 替代方案

- **在现有基座上接入新子层**：通过 [`@pllm.custom_residual`](../guides/forward-structure.md)、`pllm.model.ActivationPatch`、`pllm.model.Adapter` 等接口，把基于现有基座微调或增量训练好的权重挂载到模型中运行。
- **支持新型已准入混合架构**：对于社区中已经完成预训练的新型混合结构（例如线性注意力与全注意力混合的 Qwen3.5），只要其已被列入支持矩阵，即可直接在 pLLM 中加载使用。
- **测试期在线学习**：测试期训练（Test-Time Training / TTT / ESamp / TLM）完全受支持，可通过 `@pllm.train` 或 `@pllm.train_in_forward` 在推理过程中持续更新状态，详见 [在线训练](../guides/training.md)。

---

## 8. 按精确原生模型类准入

### 要解决的问题

开发者尝试传入自定义的模型包装类、派生子类（如继承自 `Qwen2ForCausalLM` 的子类），或者来自相似模型家族（如 Phi3、Mistral）的非登记类。

### pLLM 的选择

pLLM 严格按照 vLLM 的具体原生模型类进行准入验证（`type(model) in architectures`），明确拒绝任何未登记的模型子类：

```text
choose a native vLLM model class; workload admission does not accept model subclasses
```

### 为什么这样选

pLLM 的高效与零拷贝能力建立在底层模型精细结构的不变性契约之上：

1. **残差流拦截点契约**：pLLM 在模型内部各层拦截激活时，依赖于被准入家族统一遵循的子层规范——即 $2L+1$ 个 LayerNorm / RMSNorm 必须严格接受 `(branch, residual)` 二元组并返回 `(normalized, sum)`；任何未经准入的子类一旦修改了 Norm 签名或调用顺序，残差挂载将完全失效。
2. **权重投影的命名与排布**：`pllm.model.Adapter` 和模型权重重写直接绑定到模型具体的投影属性（如 `q_proj`、`k_proj`、`v_proj`、`gate_proj` 等）。许多衍生模型或自定义子类会合并投影（例如将 QKV 打包为单个矩阵）、改变张量对齐或改变 LM Head 的计算方式。如果不做白名单式的精确类准入，一旦底层张量切分不匹配，不仅会导致 CUDA Graph 录制崩溃，甚至可能在显存中读写错误偏移的数据。

### 替代方案

- 使用 [支持矩阵](../reference/support-matrix.md) 中已准入的原生模型架构（如 `Qwen2ForCausalLM`、`LlamaForCausalLM`、`Qwen3ForCausalLM`、`Qwen3MoeForCausalLM` 等）。
- 若权重属于兼容家族，确保模型目录下的 `config.json` 中的 `architectures` 字段指向标准的 vLLM 原生类名称。

---

## 9. 拒绝静默回退（Fail-Closed）与显式昂贵取舍

### 要解决的问题

在复杂的实验环境或服务部署中，当出现显存不足、图捕获受阻、或者组件之间产生依赖竞争时，传统深度学习库往往倾向于“静默回退（Silent Fallback）”——例如悄悄关掉 CUDA Graph 退回极其缓慢的 eager 解释执行、悄悄把张量从 GPU 溢写到 CPU 内存、或者默默忽略某些更新。

### pLLM 的选择

pLLM 坚持**闭合失败（Fail-Closed）**原则：任何语义歧义、未声明的超额资源占用或不相容的组合，在构建阶段（`pllm.explain` 或 `pllm.make_llm`）立刻抛出异常明确拒绝，不悄悄换一条路径执行。唯一的自动退让是 CUDA graph：默认 `cuda_graphs="auto"` 下，个别需要自定义切片调度的写法（例如 `@pllm.custom_forward` 改由 pLLM 按层切片独立捕获图）会让 vLLM 跳过整模型图，原因写在 `explain` 的 `forward` 行和 `eager_reason` 里，设 `cuda_graphs=True` 则直接报错。昂贵的系统取舍必须由开发者在代码中显式声明：
- **存储位置**：大查找表是放 GPU 显存还是 CPU 主机内存（`placement="device"` 还是 `"host"`）；
- **KV 物理布局**：是否牺牲某些位置释放显存（`keep="window"`、`sink=True` 还是 `keep="none", source=k`）；
- **训练结果可见滞后**：参数更新是严格无滞后阻塞等待（`publish_lag=0`），还是隔步发布以与下一步前向完全异步重叠（`publish_lag=1`）。

### 为什么这样选

在现代高并发在线服务中，静默回退是灾难性的：
- 如果大查找表因为显存不足被系统静默搬移到主机内存，跨 PCIe 传输会把每一步的延迟拖慢数十倍，线上吞吐断崖式下跌，且排查时极难定位根因；
- 如果系统在遇到小算子时悄悄关闭 CUDA Graph，小模型或低并发下解码吞吐会跌掉一半（RTX 4090 上 0.5B、128 并发掉到 49%；7B 一步更长，只慢约 2%，见 [性能与显存](performance.md)）；
- 如果在线学习或快权重更新被框架悄悄推迟或丢弃，模型生成质量将出现难以复现的劣化，算法研究员根本无法区分“是自己的算法理论不对”还是“框架在底层偷工减料”。

### 替代方案

- 在启动 GPU 服务之前，先在 CPU 上调用 `pllm.explain(workload, config)` 查看完整的执行计划与显存账单（参见 [看成本：pllm.explain](../guides/explain.md)）。
- 根据算法的物理代价主动显式配置：若需要严格的更新时序，显式传入 `ExecutionOptions(publish_lag=0)`；若显存吃紧，主动配置 `placement="host"` 或调整静态 KV 布局。

---

## 10. 采样引导方法仅限单一来源

### 要解决的问题

开发者在一个复合负载（例如组合了两个独立算法子模块）中，尝试在两个不同的方法上同时添加 `@pllm.guide_sampling` 装饰器，希望两个模型或两套特征同时引导最终的采样分布。

### pLLM 的选择

一个负载（含所有嵌套子组件）中最多只能包含一个 `@pllm.guide_sampling` 方法。若声明了两个或更多，编译期将直接报错：

```text
... and ... both declare @pllm.guide_sampling; keep at most one guidance method
```

### 为什么这样选

这与 `@pllm.add_logits` 的性质有着本质区别：
- `@pllm.add_logits` 只是在过滤前向未归一化的 logits 上累加一个偏置向量。加法运算满足交换律与结合律（$A + B = B + A$），多个修改先后相加，数学结果完全一致，因此 pLLM 允许任意多个 `@pllm.add_logits` 并存。
- `@pllm.guide_sampling` 则是运行在采样器内部、经过 top-k / top-p 截断过滤之后的**非线性分布重加权**（基于对比引导公式 `(1 + beta) * logits - beta * pred_logits`）。当存在两个不同的隐藏层预测向量时，它们应该在特征空间先求平均？还是在概率空间先做乘法？亦或是先后级联应用两次不同的截断？不同的组合方式代表着完全不同的算法语义。pLLM 坚决不替开发者做任何主观猜想。

### 替代方案

- 将多个引导信号的融合逻辑显式写在同一个方法中：读取多个隐藏层或模块输出，按照你的算法设计完成加权、插值或门控计算后，返回一个统一的张量；
- 如果只需简单的词表得分增减，改用可多重叠加的 `@pllm.add_logits`。

```python
@dataclass(frozen=True)
class VocabularyBiases(pllm.Workload):
    """多个 @pllm.add_logits 可以并存，因为词表偏置相加满足交换律。"""

    @pllm.add_logits()
    def shallow_bias(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return torch.zeros((h.shape[0], 151936), device=h.device, dtype=torch.float32)

    @pllm.add_logits()
    def deep_bias(self, h: torch.Tensor = pllm.hidden(-1)) -> torch.Tensor:
        return torch.zeros((h.shape[0], 151936), device=h.device, dtype=torch.float32)


bias_report = pllm.explain(VocabularyBiases(), CONFIG)
```

---

## 11. 改写前向时关闭前缀缓存（Prefix Caching）

### 要解决的问题

在多轮对话或系统提示词（System Prompt）较长的场景下，开发者希望开启前缀缓存（`enable_prefix_caching=True`），同时又在负载中使用了 `@pllm.custom_residual`、`@pllm.custom_attention`、参数覆盖（`pllm.model.Adapter`）或激活补丁（`pllm.model.ActivationPatch`）来改写模型的前向计算。

### pLLM 的选择

当负载改写了模型内部计算节点时，必须保持前缀缓存关闭（`enable_prefix_caching=False`，默认值）；若在改写前向的同时开启了前缀缓存，系统在准入期直接报错拦截：

```text
disable prefix caching (enable_prefix_caching=False) when a workload rewrites model sites; cached KV would not match
```
或
```text
disable prefix caching when using parameter overlays or activation patches
```

### 为什么这样选

前缀缓存（Prefix Caching）的核心原理是假定“相同的输入 token 序列，在同一个模型上必定产生逐位完全相同的 KV Cache”，因此只需对 token id 序列计算哈希值即可跨请求复用已有的 KV 页。

但是，当你的负载包含动态修改前向的逻辑时：
1. 激活补丁或残差改写使得每一层写入 KV 缓存的张量数值不仅取决于 token id，还取决于该请求当前的动态状态、在线更新的权重版本、或外部注入的引导向量；
2. 如果开启前缀缓存，当后续一个携带相同 prompt 的新请求进入时，引擎会直接复用前一个请求在不同状态下计算出的旧 KV 缓存，导致新请求的前向计算直接建立在脏数据之上，彻底破坏了算法的数学正确性。

### 替代方案

- 保持 `ModelConfig(enable_prefix_caching=False)`（这是 pLLM 的默认配置）。
- 对于纯只读观测型负载（如通过 `@pllm.export` 提取探针特征）或纯词表偏置型负载（`@pllm.add_logits`），由于它们并不修改模型层内的前向激活或 KV 生成过程，前缀缓存可以安全开启。

---

这些限制对应的报错原文和改法见[报错目录](../reference/errors.md)。

---

## 算法不是 pLLM 的工作：pLLM 提供的是底座

在评估一项新算法能否在服务引擎中高效运行时，一个核心心智模型是区分**算法数学**与**系统基质**：

- **算法数学属于开发者**：你的前向投影公式、隐藏状态插值、损失函数计算、甚至自定义 Triton 核函数，都由你自由定义与编写。
- **系统基质属于 pLLM 与 vLLM**：张量的显存排布、CUDA Graph 的静态拓扑录制、连续批处理的动态打包切分、跨请求的严格内存隔离，是系统保证高吞吐、低延迟与确定性输出的底层防线。

当一个算法因为触碰了物理法则或服务化边界而“今天写不了”时，说明该算法需要与高吞吐并发推理的现实条件做一次明确的理论对齐。理解这些设计取舍，不仅能帮助你写出正确可运行的 `pllm.Workload`，也能让你在设计新算法时，从第一天起就具备服务化与系统级性能的全局视野。

---

## 自测

1. **为什么在 pLLM 中可以声明多个 `@pllm.add_logits`，却只能声明一个 `@pllm.guide_sampling`？**
   <details><summary>答案</summary>
   <code>@pllm.add_logits</code> 只是在采样前向词表 logits 累加偏置向量，加法运算满足交换律与结合律（$A + B = B + A$），无论添加顺序如何，数学结果完全唯一确定。而 <code>@pllm.guide_sampling</code> 是在经过 top-k / top-p 过滤后的分布上执行非线性的对比重加权（基于 $(1 + \beta)\text{logits} - \beta\,\text{pred\_logits}$）。如果存在多个预测特征，它们应该在特征空间加权平均、在对数概率空间相加、还是分层级联截断，代表着完全不同的算法意图，pLLM 拒绝替开发者做出未经证实的语义猜测。
   </details>

2. **如果一位研究员希望通过丢弃 prompt 中不重要的 token 来缩减首字延迟（TTFT），为什么不能在 `@pllm.custom_forward` 中直接给 prompt 传入 `active=` 掩码？**
   <details><summary>答案</summary>
   当某一行早退跳过某一层时，为了保证 KV 缓存的物理与结构完整，pLLM 会用当前隐藏状态经过该层的投影补齐近似的 K/V。Prompt 通常由数十到数千个连续 token 组成，如果在 prompt 阶段大面积早退，高层网络中整段上下文的 KV 将全被合成的近似值占据；在接下来的数百步自回归解码中，后续生成的每一个 decode token 在做因果注意力时都会读取这些严重失真的 prompt KV，导致生成文本迅速退化。
   </details>

3. **为什么开启前缀缓存（Prefix Caching）与使用 `@pllm.custom_residual` 或参数覆盖是互斥的？**
   <details><summary>答案</summary>
   前缀缓存的基本假设是“相同的 prompt token 序列必然产生逐位完全一致的 KV 缓存”，并仅以 token id 的哈希值作为复用依据。而当使用 <code>@pllm.custom_residual</code> 或参数覆盖改写前向计算时，模型层内产生的 KV 数值不仅取决于输入的 token id，还取决于该请求当前的动态状态、在线微调的权重版本或外部注入的引导向量。如果开启前缀缓存，后续相同 prompt 的新请求会错误地复用前一个请求在不同状态下产生的旧 KV 缓存，破坏推理的数学确定性。
   </details>

---

## 下一步去哪

- 查阅完整的功能可行性清单：[能力地图](../guides/capability-map.md)
- 了解核心物理机制与执行时间线：[核心概念](core-concepts.md)
- 查看各项机制的实测吞吐与开销：[性能与显存](performance.md)
- 学习如何在自定义注意力中编写合法读取：[改 forward 结构](../guides/forward-structure.md) 与 [10 自定义注意力](../examples/10_custom_attention.ipynb)
- 学习如何编写 token-choice 自适应深度：[12 自适应深度](../examples/12_adaptive_depth.ipynb)
- 查看完整的报错信息与修复建议：[报错目录](../reference/errors.md)
