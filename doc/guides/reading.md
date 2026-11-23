# 读取模型内部

在 Hugging Face 中研究推理算法时，通常只需传入 `output_hidden_states=True` 或在 `model.layers[i]` 上注册 PyTorch `forward hook`，就能拿到每一层的中间激活或 logits。但在高吞吐的 vLLM 服务中，几十个并发请求的 token 会被连续批处理拼进同一个张量，且模型前向传播被捕获进了 CUDA graph——普通 PyTorch hook 既无法分辨张量中的哪一行属于哪个请求，在 CUDA graph 重放时也根本不会执行。

pLLM 允许你在负载方法中直接声明需要读取的内部张量（例如 `r: torch.Tensor = pllm.residual(12)`）。底层在 vLLM 的 CUDA graph 中预留静态缓冲并在图内完成捕获，既不破坏 CUDA graph，也不引入 CPU 与 GPU 之间的同步等待。方法收到的张量第 0 维自动对齐到当前步的请求行。

如果你的算法需要读取负载自身持有的可变参数或模块状态（如 `pllm.parameter`、`pllm.module`、`pllm.version`、`pllm.index`），这些属于状态管理声明，请参阅 [状态](state.md)。

---

## 快速查阅：从 Hugging Face 到 pLLM 来源标记

下表汇总了常见的内部激活与元数据在 Hugging Face 中的概念与 pLLM 来源标记的对应关系：

| Hugging Face 概念 / 属性 | pLLM 来源标记 | 数据类型（dtype） | 形状（shape） | 含义与典型用途 |
| :--- | :--- | :--- | :--- | :--- |
| `hidden_states[layer + 1]` | `pllm.residual(layer)` | 模型 dtype (`bfloat16`/`float16`) | `[rows, hidden]` | 第 `layer` 层之后的残差流（累加了 embedding 与前面所有子层输出）；激活引导、SAE、探针的首选输入 |
| `model.layers[layer].mlp` 的输出 | `pllm.hidden(layer)` | 模型 dtype | `[rows, hidden]` | 第 `layer` 层 MLP 的增量输出（尚未加回残差流）；用于观察或蒸馏该层的新鲜变换 |
| attention 与 MLP 之间的中间向量 | `pllm.residual(layer, after="attention")` | 模型 dtype | `[rows, hidden]` | 第 `layer` 层 attention 加回残差流后、进入 MLP 前的残差流 |
| `last_hidden_state`（经过最终 norm） | `pllm.final_hidden()` | 模型 dtype | `[rows, hidden]` | 经过最终 RMSNorm 之后的向量，LM head 的直接输入 |
| `model.lm_head(norm(hidden_states[l+1]))` | `pllm.layer_logits(layers)` | `float32` | `[rows, len(layers), vocab]` | 若干层之后的残差流分别通过最终 norm 和 LM head（logit lens）；最后一层对应模型输出 logits |
| `logits`（采样前的模型最终输出） | `pllm.layer_logits((-1,))` | `float32` | `[rows, 1, vocab]` | 采样前若需读取模型输出 logits（例如按自身分布改写 logits），通过层 logits 读取最后一层 |
| `logits`（采样后的原始输出） | `pllm.base_logits()` | `float32` | `[rows, vocab]` | 采样完成后供训练或监控读取的原始未修改 logits（仅在采样后可用） |
| `input_ids`（当前与历史上下文 token） | `pllm.input_tokens(history=n)` | `int64` | `[rows, n + 1]` | 当前 token 及前 `n` 个 token ID（早于请求起始处填 `-1`）；用于查表记忆与 n-gram 先验 |
| 生成的 token ID | `pllm.sampled_tokens()` | `int64` | `[rows]` | 采样器本步选中的 token ID（仅在采样后可用） |
| 多采样序号 | `pllm.sample_index(limit=None)` | `int64` | `[rows]` | 每个请求在 prompt 内的采样编号（`0` 到 `n - 1`） |
| FlashAttention 的 LSE / aux 统计 | `pllm.attention(layer, "lse" / "aux")` | `float32` / 自定义 | `[rows, heads]` / `[rows, ...]` | Attention 的 Log-Sum-Exp 或自定义 aux 统计量（配合 `@pllm.custom_attention`） |

---

## 一个完整的读取示例

先看一个完整的读取负载：同时读取第 12 层的残差流范数，并读取第 6 层、第 12 层与最后一层的预测 token（logit lens top-1）。

```python
import torch
import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class Lens(pllm.Workload):
    @pllm.export(name="lens", max_rows=8, max_cols=3)
    def top1(self, logits: torch.Tensor = pllm.layer_logits((6, 12, -1))) -> torch.Tensor:
        return logits.argmax(dim=-1).float()  # [rows, 3]：第 6 层、第 12 层与最后一层最可能的 token

    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)  # [rows, 1]：第 12 层残差流的 L2 范数


print(testing.check(Lens(), CONFIG).passed)
```

在上面的代码中：
1. `logits: torch.Tensor = pllm.layer_logits((6, 12, -1))` 声明读取三层的 logits，形状为 `[rows, 3, 151936]`，数据类型原生为 `float32`。
2. `r: torch.Tensor = pllm.residual(12)` 声明读取第 12 层之后的残差流，形状为 `[rows, 896]`，数据类型与模型一致（通常为 `bfloat16`）。
3. 借助 [`pllm.testing.check`](checking.md)，你可以在不需要 GPU 的 CPU 环境下验证负载的输入形状、对齐方式以及调度逻辑。

---

## 残差流与层内激活：`residual`、`hidden` 与 `final_hidden`

在 Transformer 解码器中，第 $l$ 层的内部计算流程如下：

$$\begin{aligned}
x'_l &= x_l + \text{Attention}(\text{Norm}(x_l)) \\
x_{l+1} &= x'_l + \text{MLP}(\text{Norm}(x'_l))
\end{aligned}$$

其中 $x_l$ 是贯穿全模型的**残差流**。分清你读取的是哪一个位置至关重要：

- `pllm.residual(layer, *, after="layer", tokens="decode")`：
  读取第 `layer` 层之后的残差流 $x_{l+1}$，相当于 Hugging Face 的 `hidden_states[layer + 1]`。它累加了最初的词嵌入向量（embedding）以及从第 0 层到第 `layer` 层所有子层的输出。**激活引导（ActAdd/CAA）、稀疏自编码器（SAE）、线性探针和 logit lens 读的都是它**。
  若指定 `after="attention"`，则读取 attention 加回残差流之后、进入 MLP 之前的中间流 $x'_l$。
- `pllm.hidden(layer=0, *, tokens="decode")`：
  读取第 `layer` 层 MLP 的输出 $\text{MLP}(\text{Norm}(x'_l))$，**位于残差加法之前**。它只包含这一层 MLP 刚算出来的新增特征，不包含主干残差流。如果你的算法是在线蒸馏某一层的单层变换能力，使用它；如果误把它当成残差流，会丢失之前所有层的累加信息。
- `pllm.final_hidden(*, tokens="decode")`：
  读取模型通过最终 RMSNorm 之后的隐藏状态，相当于 Hugging Face 的 `last_hidden_state`。这是 LM head 投影前的最终特征向量。

### 数据类型与 `.float()` 的使用时机

`pllm.hidden`、`pllm.residual` 与 `pllm.final_hidden` 的数据类型由模型的运行精度决定（通常为 `torch.bfloat16` 或 `torch.float16`）。在以下场景中，应当主动调用 `.float()`：

1. **范数与相似度归约**：计算 L2 范数（`r.float().norm(dim=-1)`）或点积时，半精度的平方与累加极易溢出（overflow）或丢失小量精度。
2. **训练损失计算**：在 [`@pllm.train`](training.md) 方法中计算均方误差或交叉熵时，转换为 `float32` 可以防止梯度下溢或 NaN。
3. **导出到宿主**：[`@pllm.export`](results.md) 导出浮点指标时，将其转换为单精度浮点数以保证数值精度。

---

## 逐层 Logits 与 Logit Lens：`layer_logits`

在模型解释性与 DoLa 等对比解码算法中，我们常常需要观察“如果模型在第 $l$ 层就停下来，它会预测什么”。在原生 PyTorch 中，这需要将中间残差流依次通过模型的最终 norm 和 LM head。

`pllm.layer_logits(layers)` 封装了这一过程：

```python
logits = pllm.layer_logits((6, 12, -1))
```

- **参数**：接收一个层索引元组（支持负数索引，`-1` 代表最后一层）。
- **形状与精度**：返回 `[rows, len(layers), vocab]` 的 `torch.float32` 张量。之所以原生提供 `float32`，是因为 LM head 在词表维度（Qwen2.5 为 151936）投影后数值范围较大，在 FP32 下运算能防止未归一化 logits 溢出。
- **与模型输出逐位相同**：`layer_logits` 对中间层使用独立开辟的矩阵乘法（GEMM）计算 logits，完全不侵入基座模型本身的 LM head 路径。因此，模型自身采样的输出 logits 与原生 vLLM 保持严格的逐位相同（bitwise identical）。
- **行对齐**：为采样器抽取的每个 token 生成一行（包括 prompt 的最后一个 token 以及每个 decode token）。

详细的对比实验可参阅 [示例 06：读模型内部与 DoLa](../examples/06_reading_inside.ipynb)。如果需要在服务运行期间不重捕获 CUDA graph 动态切换读取哪一层的 `residual`、`hidden` 或 `layer_logits`，可使用高级入口 [`pllm.model.LayerRoute`](model-edits.md#高级入口运行时动态切层pllmmodellayerroute)。

---

## 控制读取哪些 Token：`tokens=` 规则与预填充捕获

在连续批处理中，同一个计算步（step）可能同时包含正在做预填充（prefill）的分块 prompt 和正在逐 token 生成（decode）的请求。`tokens=` 参数决定了 `[rows, hidden]` 的行轴对应哪个时刻的 token。

`pllm.hidden`、`pllm.residual` 与 `pllm.final_hidden` 均支持 `tokens=` 参数：

- `tokens="decode"`（默认）：
  当前步中处于解码阶段的每个请求各占 **1 行**，对应其最新生成的 decode token。绝大多数逐 token 引导、探针监控与在线训练算法都使用此默认值。
- `tokens="prefill"`：
  当前步中**刚刚完成整个 prompt 预填充**的请求各占 **1 行**，对应其 prompt 的最后一个 token。如果一个长 prompt 被拆分成多个 chunk 分步预填充，它在中间的 chunk 步不会出现，仅在预填充彻底完成的那一步出现一次。这适用于 prompt 分类、特征提取或基于 prompt 结尾向量的决策。
- `tokens="all"`：
  当前步新计算出的**每一个 token** 各占 **1 行**（包含 prefill chunk 的所有中间 token），且严格按照请求内 token 位置先后排列。
  - **使用限制与原因**：`tokens="all"` **只能在没有 `pllm.result` 输入的 `@pllm.export` 方法中读取**。这是因为像在线训练、修改 logits、引导采样等其他方法，其显存和 CUDA graph 都是按请求数（`max_num_seqs`，如 8 或 32）预先分配固定大小缓冲的；而一个 step 计算的 fresh token 总数可能达到 `max_num_batched_tokens`（如 4096 或 8192），无法塞进按请求固定的图缓冲内。只有 `@pllm.export` 具备将大批量数据切块并流式传递到 CPU 队列的能力。
- **一个方法只读一个阶段**：
  同一个负载方法中**不要混合读取不同 `tokens=` 阶段的输入**（例如一个参数读 `tokens="prefill"`，另一个参数读 `tokens="decode"`）。原因在于预填充完成的请求集与正在解码的请求集在同一个 step 中是互斥或不同的行子集，行轴无法对齐。

### 示例：捕获 Prompt 结尾的表征向量

下面的示例展示了如何使用 `tokens="prefill"` 提取每个 prompt 结尾处的残差流：

```python
prompt_ends: dict[int, torch.Tensor] = {}


class PromptEnd(pllm.Workload):
    @pllm.export(name="prompt_end", max_rows=4, max_cols=896)
    def last_token(self, r: torch.Tensor = pllm.residual(12, tokens="prefill")) -> torch.Tensor:
        return r.float()  # [rows, 896]：每个刚完成预填充的 prompt 取最后一行

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        assert event.dropped_rows == 0
        for identity, row in zip(event.identities, event.tensor()):
            prompt_ends[identity.prompt_index] = row
```

在 GPU 上运行时，每次传入不超过 `max_rows` 个 prompt，即可在不丢失任何一行的情况下完整采集：

```python
# check-docs: skip: requires a CUDA engine and model weights
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)

texts = ["I love this", "I hate this", "This is wonderful", "This is terrible", "I feel happy", "I feel sad"]
with pllm.make_llm(GPU_CONFIG, workload=PromptEnd()) as llm:
    for start in range(0, len(texts), 4):  # 每次最多送入 max_rows 个 prompt
        chunk = texts[start : start + 4]
        # prompt_indices 给跨 batch 的 prompt 编号；identity.prompt_index 会原样带回
        llm.generate(chunk, SamplingParams(max_tokens=1), prompt_indices=tuple(range(start, start + len(chunk))))

rows = torch.stack([prompt_ends[index] for index in range(len(texts))])
print(rows.shape)  # torch.Size([6, 896])
```

---

## 上下文 Token 历史与采样序号：`input_tokens` 与 `sample_index`

### `pllm.input_tokens(history=n)`

在实现基于上下文 n-gram 查表（如 Engram）或 n-gram 语言模型先验时，算法需要拿到当前及前面几个 token 的 ID。

- **返回值**：`torch.int64` 形状为 `[rows, history + 1]`。第 0 列 `ids[:, 0]` 是当前 token ID，后续列依次是向前回溯的 token ID（`ids[:, 1]` 为前一个，以此类推）。
- **边界填充**：如果请求刚开始生成、回溯位置早于 prompt 第 0 个 token，相应位置填充 `-1`。
- **行对齐**：当其计算结果被送入 [`@pllm.custom_residual`](forward-structure.md) 时，针对当前步调度的每一个 token 计算；对于其他方法（如 `@pllm.export`、`@pllm.add_logits`），则对应采样的行。
- **数据来源**：token 历史取自 vLLM（MRv2 runner）保存在 GPU 上的请求输入，所以查表不需要从 CPU 搬数据。详细用法可参考 [示例 11：查表记忆与 n-gram 先验](../examples/11_lookup_memory.ipynb)。

```python
class TokenWatcher(pllm.Workload):
    @pllm.export(name="tokens", max_rows=8, max_cols=3)
    def recent_tokens(self, ids: torch.Tensor = pllm.input_tokens(history=2)) -> torch.Tensor:
        return ids.float()  # [rows, 3]：当前 token、前一个、前两个


print(testing.check(TokenWatcher(), CONFIG).passed)
```

### `pllm.sample_index(limit=None)`

当客户端设置 `SamplingParams(n=4)` 对同一个 prompt 采样多个回答时，每个分支请求具有不同的采样序号。

- **返回值**：`torch.int64` 形状为 `[rows]`，表示该行在其所属 prompt 中的采样索引（从 `0` 到 `n - 1`）。
- **`limit` 参数**：传入 `limit=K` 时，`generate` 拒绝 `sample_index >= K` 的请求。负载只为前 K 个样本准备了数据（例如按样本序号查的表只有 K 行）时，用它在请求进入时就报错，而不是读越界、悄悄算错。

---

## 采样后的反馈：`base_logits` 与 `sampled_tokens`

在某些算法中，我们需要在采样发生之后，观察模型的原始未修改分布或记录被选中的 token：

- `pllm.base_logits()`：返回模型自身的原始 logits（未经过任何 `@pllm.add_logits` 或采样引导修改），数据类型为 `float32`，形状为 `[rows, vocab]`。
- `pllm.sampled_tokens()`：返回采样器在当前步选中的 token ID，数据类型为 `int64`，形状为 `[rows]`。

### 为什么只能在采样后读取

`pllm.base_logits()` 与 `pllm.sampled_tokens()` **只能在采样之后（`Phase.AFTER_SAMPLE`）读取**，例如在 [`@pllm.train`](training.md) 或带有 `@pllm.on(pllm.Phase.AFTER_SAMPLE)` 的后处理方法中。

其原因在于因果依赖性（Causality）：
1. `sampled_tokens` 是采样器掷骰子的结果，在采样执行之前这个 token 根本尚未产生。
2. `@pllm.add_logits` 的任务是计算加到 logits 上的偏置向量。如果在采样前读取 `base_logits()`，会在采样管线中形成环形依赖。如果你需要在采样前根据模型当前层的预测趋势做调整，请使用 `pllm.layer_logits((-1,))`。

---

## 注意力内部观察：`pllm.attention`

当需要分析注意力的分布特性（例如计算 Softmax 分母的对数和、或提取自定义注意力算子的内部状态）时，使用 `pllm.attention`：

- `pllm.attention(layer, "lse")`：读取第 `layer` 层第一次调用 `kv.attend` 时产出的 Log-Sum-Exp，数据类型为 `float32`，形状为 `[rows, heads]`。
- `pllm.attention(layer, "aux")`：读取该层自定义注意力算子返回的辅助张量 `aux`，形状为 `[rows, ...]`。
- **使用限制与原因**：该标记必须配合 [`@pllm.custom_attention`](forward-structure.md) 使用。在注意力函数内部，它只能读取更早的层（`layer < current_layer`）；在外部张量方法（如 export 或 train）中，它读取当前采样行的数据。

---

## 性能与成本：为什么读取不破坏 CUDA Graph

许多开发者担心在模型前向传播中“截取”激活值会导致 CUDA graph 打破、回退到低效的 Python eager 执行。在 pLLM 中，读取内部激活对性能是透明且友好的：

1. **图内捕获（Hidden Tap）**：pLLM 在模型编译阶段识别所有 `pllm.residual`、`pllm.hidden`、`pllm.layer_logits` 声明，并在底层相应模块输出处挂载轻量级的拷贝算子。在 CUDA graph 捕获时，该算子直接将数据复制到静态预留的**抽头缓冲（tap buffer）**中，不发生主机等待，因此 `llm.diagnostics().eager_reason` 保持为空字符串 `""`。
2. **显存开销**：每个被读取的位置在 GPU 显存中分配一块固定大小的缓冲，大小随 `max_num_seqs` 和 hidden 宽度增长；读 `pllm.layer_logits` 时还要为 LM head 的输出（词表宽度）预留缓冲。同一个负载中多个方法读取同一位置时，它们**自动共享**同一块缓冲。

你可以通过 [`pllm.explain`](explain.md) 在 CPU 上直接看到这些抽头显存：

```python
print(pllm.explain(Lens(), CONFIG))
```

输出中会明确打印出每条抽头的预留情况（格式如下）：

```text
tap      model.layers.13.input_layernorm[1], decode,prefill: 28800 bytes shared by norm, top1
tap      model.layers.7.input_layernorm[1], decode,prefill: 28800 bytes shared by top1
tap      model.norm[1], decode,prefill: 28800 bytes shared by top1
tap      lm_head[0], prefill,decode: 29171840 bytes shared by top1
```

第 12 层之后的残差流在第 13 层开头的 norm 处取出（`model.layers.13.input_layernorm`），由 `norm` 和 `top1` 共用；逐层 logits 的 LM head 缓冲是词表宽度，所以大得多（约 28 MiB）。

---

## 常见报错与排查

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[声明与输入](../reference/errors.md#声明与输入)、[阶段](../reference/errors.md#阶段值什么时候存在)几节。

---

## 自测

<details>
<summary>1. 激活引导（Activation Steering）算法希望在第 12 层的输出上叠加一个引导向量，应该读取并改写 <code>pllm.residual(12)</code> 还是 <code>pllm.hidden(12)</code>？为什么？</summary>

**答案**：应当使用 `pllm.residual(12)`。
**原因**：残差流（residual stream）是模型跨层传递累积信息的主干通道，第 12 层的 `residual` 包含了词嵌入和前 12 层所有子层的输出之和，下一层也是直接从残差流上读取输入的。而 `pllm.hidden(12)` 仅仅是第 12 层 MLP 自身新计算出来的单层增量，尚未加回残差流。如果针对 `hidden` 进行引导，会脱离主干残差流，丢失此前各层的累积上下文。
</details>

<details>
<summary>2. 为什么在 <code>@pllm.add_logits</code> 方法中读取 <code>pllm.base_logits()</code> 会报错？如果算法需要基于当前预测概率增加惩罚项，该怎么写？</summary>

**答案**：`@pllm.add_logits` 运行在采样之前（`BEFORE_SAMPLE`），其输出直接参与决定最终送进采样器的 logits；而 `pllm.base_logits()` 是采样阶段记录的基座模型原始 logits，只能在采样之后（`AFTER_SAMPLE`）读取。如果在采样前读取，会在图调度中产生因果循环依赖。
如果算法需要在采样前读取模型的未调整输出，应声明读取 `pllm.layer_logits((-1,))`。
</details>

<details>
<summary>3. 如果想要提取整个 prompt 中每个 token 的表征做聚类分析，能否直接在一个 <code>@pllm.train</code> 方法中读取 <code>tokens="all"</code>？为什么？</summary>

**答案**：不能。
**原因**：`@pllm.train` 等在线训练与前向干预方法运行在 CUDA graph 内，其显存和张量行数是按照请求数量（`max_num_seqs`）预先分配固定大小的；而 prompt 中的 token 总数上限是 `max_num_batched_tokens`（通常为数千），远大于请求数。只有 `@pllm.export` 具有将变长的大批 token 行切块分批流式传输回主机端队列的能力。
</details>

