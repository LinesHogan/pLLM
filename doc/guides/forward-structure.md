# 改 forward 结构：残差、attention 与层的执行顺序

在 PyTorch 或 Hugging Face 中修改模型的计算结构非常直接：改写 `forward()` 函数、插入 hook，或者在层与层之间调用自定义函数。但在以吞吐为第一目标的 vLLM 服务中，数十个并发请求会被连续批处理（continuous batching）拼进同一个大张量，整模型的前向传播又被捕获进 CUDA graph。此时如果直接 fork 模型源码插入 Python 逻辑，不仅会破坏算子融合退回 eager 执行，更无法妥善管理每个并发请求的私有状态。

`pllm` 提供了三个专用于改写前向计算结构的装饰器，让你在保持 CUDA graph 高吞吐和请求隔离的前提下重构前向传播：
1. `@pllm.custom_residual`：改写残差流汇入方式、替换 attention 读入的 value，或挂载按块更新的快权重记忆子层。
2. `@pllm.custom_attention`：改写 attention 的读取方式（只读 top-K、监控注意力分布），或声明物理 KV 缓存保留布局（滑动窗口、保留 prompt 加窗口、跨层共享 KV）。
3. `@pllm.custom_forward`：用普通 Python 重新编排各层的执行顺序（循环深度、逐 token 早退跳层、连续思维跨步状态）。

---

## 1. 定制残差连接：`@pllm.custom_residual`

### 适用场景与动机
当算法需要在 Transformer 子层进出残差流的切入点上做数学变换时，使用 `@pllm.custom_residual`。常见算法包括：
- **Value Residual**（如 ResFormer）：将浅层（例如第 0 层）计算出的 value 混入后续各层的 value。
- **超连接（Hyper-Connections）**：将单一残差流扩展为 $n$ 条并行流，在子层前后分别执行加权读取与线性混合。
- **LAuReL**：在子层输出并入残差流时加入低秩修正或门控缩放。
- **带状态的记忆子层**（如 TTT、Titans）：在 attention 或 MLP 旁挂载一个由当前请求独立拥有的快权重记忆矩阵，边读边按块更新。

在原生 vLLM 中，残差加法与 LayerNorm / RMSNorm 通常被编译为融合算子（Fused Add+Norm）。如果为了改动残差而修改模型代码，往往会迫使整个模型退回 eager 模式。`@pllm.custom_residual` 让你接管指定位置的残差更新，而原生的 attention 投影、MLP 计算与 KV cache 写入照常进行，且改写后的算子仍完整捕获在模型的 CUDA graph 中。

### 最小可运行示例（CPU）
下面的示例实现 ResFormer 风格的 Value Residual：第 1 到 23 层在将 value 写入 KV cache 和送入 attention 之前，按比例混入第 0 层的 value。

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class ValueResidual(pllm.Workload):
    mixing: float = 0.0

    @pllm.custom_residual(site="value", layers=tuple(range(1, 24)))
    def mix(self, v: torch.Tensor, first: torch.Tensor = pllm.value(0)) -> torch.Tensor:
        return (v.float() + self.mixing * (first.float() - v.float())).to(v.dtype)


print(repr(pllm.explain(ValueResidual(), CONFIG).eager_reason))
```

```text
''
```

输出的 `eager_reason` 为空字符串 `''`，说明该负载完全运行在模型的 CUDA graph 中。

### 挂载位置表（`site`）与输入参数
`@pllm.custom_residual` 通过 `site` 参数指定接管哪一个位置的残差逻辑：

| `site` | 方法接收的位置张量（无默认值） | 返回值 | 替换的原生行为 | 典型用途 |
| --- | --- | --- | --- | --- |
| `"attention"` | 子层前的残差流 `x`、attention 分支输出 `y` | 更新后的残差流 | `x + y` | 注意力残差缩放、门控分支 |
| `"mlp"` | 子层前的残差流 `x`、MLP 分支输出 `y` | 更新后的残差流 | `x + y` | MLP 残差改写、外挂快权重记忆 |
| `"both"` | 同上，按前向顺序依次作用于 attention 与 MLP | 更新后的残差流 | `x + y` | LAuReL、超连接子层混合 |
| `"value"` | 当前全注意力层的 value 行 `v` `[rows, kv_heads, head_dim]` | 替换后的 value | 原生 `v` | Value Residual（替换后供 attention 读取并写入 KV cache） |
| `"final"` | 进入最终 norm 前的残差流 `x` | 单条残差流 `[rows, hidden]` | 原生残差流 | 超连接多流合并回单流 |

**方法签名与输入规则**：
1. **位置张量在前**：`attention`、`mlp`、`both` 必须在前两个参数接收 `x, y`；`value` 与 `final` 必须在第一个参数接收位置张量，均不带默认值。
2. **后续参数使用标记默认值**：位置张量之后可以读取以下标记：
   - `pllm.parameter(state, field)` 或 `pllm.module(state)`：读取实例状态或请求状态的携带量。
   - `pllm.depth()`：当前层号（Python `int`）。若 `site="both"`，attention 处为 `2 * layer`，MLP 处为 `2 * layer + 1`。
   - `pllm.value(layer)`：同一 token 在本次前向中更早层算出的 value 张量。
   - `pllm.residual(layer)`：同一 token 在本次前向中更早层最后算出的残差流。
3. **精度处理**：传入的张量为模型原生类型（如 `bfloat16`）。涉及数值敏感的累加或除法时，在方法内部转为 float32 计算（`.float()`），返回前转回原类型（`.to(v.dtype)`）。
4. **多残差流（`streams=n`）**：设 `streams=n`（$n > 1$）时，残差流自 embedding 起扩展为 `[rows, n, hidden]`。配合 `read="method"` 命名一个 `(x, ...) -> branch_input` 方法在进入子层前将多流压为一条；最后必须声明一个 `site="final"` 的方法将多流合并回 `[rows, hidden]`。详见例程 [07 改写残差连接](../examples/07_custom_residual.ipynb)。

### 恒等还原与数值等价性
当自定义残差方法返回恒等结果（如 `site="attention"` 返回 `x + y`，或 `site="value"` 返回 `v`）时，模型输出与 eager 模式的原生 vLLM **逐位一致（bitwise equal）**。

**为什么它与默认图模式的原生 vLLM 存在微小差异？**
默认图模式的原生 vLLM 会启用 PyTorch Inductor 融合内核（Fused Add+Norm）。当加载 `@pllm.custom_residual` 时，vLLM 改为捕获模型原本的非融合算子到 CUDA graph 中。这种方式保留了纯正的 PyTorch 算术语义，因此与 eager 模式的原生 vLLM 逐位相同，但和 Inductor 融合内核之间会有极细微的 bf16 舍入差异。验证恒等行为时，请与 eager 模式的原生引擎对比，或直接调用 `testing.check_on_gpu`。

### 带状态的记忆子层（快权重）
在实现 TTT 或 Titans 等算法时，算法需要维护每个请求独立的快权重状态 $M$（例如 `[hidden, hidden]` 矩阵），并在生成过程中不断读写。

在 `@pllm.custom_residual` 中声明状态子层：
- `state="name"`：指向一个只有一个字段（携带量 carry）的 `pllm.RequestState`，不带优化器。
- `chunk=N`：固定块长 $N$（正整数，例如 16）。
- 仅支持挂载在某**一层**的单个 `"attention"` 或 `"mlp"` 位置。

```python
@dataclass(frozen=True)
class FastWeightMemory(pllm.Workload):
    gate: float = 0.0
    lr: float = 0.1

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState({"M": lambda m: (m.hidden_size, m.hidden_size)})

    @pllm.custom_residual(site="mlp", layers=(12,), state="memory", chunk=16)
    def remember(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        M: torch.Tensor = pllm.parameter("memory", "M"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = F.normalize(x.float(), dim=-1)
        read = k @ M.T
        stream = (x.float() + y.float() + self.gate * read).to(x.dtype)
        next_M = M - self.lr / len(x) * (read - y.float()).T @ k
        return stream, next_M


print(repr(pllm.explain(FastWeightMemory(), CONFIG).eager_reason))
```

```text
''
```

**更新规则与设计原因**：
同一 chunk（16 行）内的每一行，在计算读取时读到的都是**该 chunk 起始时的携带量**；只有该 chunk 的最后一行执行完毕后，方法返回的 `next_M` 才会正式生效并更新为下一 chunk 的起始状态。
如果某次调度步中某请求剩下的 token 未凑满 16 行，未完成的行由框架暂存，待后续步骤凑满整块后再触发状态更新。这样设计的原因在于：vLLM 的连续批处理调度器可能会将长 prompt 任意切块（Chunked Prefill），通过将更新严格锚定在固定块长边界上，保证了**生成结果完全不依赖调度器如何分步切分 prompt**。

### 运行开销与限制
- **开销**：改写逻辑直接录入 CUDA graph，避免了逐步 Python 同步开销。吞吐下降仅来自于算法多出的张量算术。在 RTX 4090 上对 Qwen2.5-0.5B 实测（详见 [07 改写残差连接](../examples/07_custom_residual.ipynb)）：
  - Value Residual（改写 23 层 value）：吞吐为原生 CUDA graph 的 **92%**。
  - 超连接（$n=4$，改写 48 个子层）：吞吐为原生 CUDA graph 的 **74%**。
  - 带状态记忆子层（每 token 读写 `[hidden, hidden]` 矩阵）：吞吐为原生 CUDA graph 的 **60%**。
- **限制与原因**：
  - `tensor_parallel_size=1`：残差改写算子在单卡层前向图内执行，暂不支持跨多卡的残差张量与状态同步。
  - 关闭前缀缓存（`enable_prefix_caching=False`）：前缀缓存假定 prompt 前缀的 KV 恒定不变；残差改写或带状态更新会导致同一前缀产生不同的激活或跨 token 状态积累，使前缀缓存失效。
  - 带状态记忆子层需要调大 `ExecutionOptions(cuda_graph_pool_budget_bytes=...)`：为每个请求分配的私有状态显存与 chunk 历史缓冲会超出默认 64 MiB 的图捕获预算，建议配置为 `1 << 30`（1 GiB）；若预算不足，引擎启动时会明确报错并给出所需字节数。
  - 带状态记忆子层**不能**与 `@pllm.custom_forward` 组合：状态子层的记忆按 token 严格推进，而自定义前向可能在同一 token 内多次循环或跳过某层，导致步进时机未定义。

---

## 2. 定制 attention 读法与 KV 布局：`@pllm.custom_attention`

### 适用场景与动机
当算法聚焦于 attention 计算过程中的信息获取或 KV cache 的存储形态时，使用 `@pllm.custom_attention`。典型场景包括：
- **注意力监控**：提取中间统计量（如注意力落在局部窗口上的比例），且不影响模型输出。
- **训练无关的稀疏读取**（如 Quest）：动态计算每页键值的分数，每头仅选择 top-K 个逻辑位置送入 attention。
- **物理 KV 缓存瘦身**：使用局部滑动窗口、保留 prompt 加滑动窗口（R-SWA），或在相邻层之间共享 KV 缓存。

在推理引擎中，attention 通常是高度融合的底层算子，KV 块的分配与回收由分页内存池全权接管。传统实现若想尝试稀疏读取或改写 KV 几何分布，必须修改 C++/CUDA 内核。`@pllm.custom_attention` 保留原生的 Q/K/V 投影和 KV 写入过程，仅替换 FlashAttention 读取步骤，并提供声明物理 KV 布局的接口。

### 最小可运行示例（CPU）
下面的示例展示两种典型用法：一种声明偶数层仅保留 256 token 的滑动窗口（奇数层保留全量）；另一种监控第 12 层注意力分布并导出到主机。

```python
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """偶数层保留最近 256 个 token 的 KV，奇数层保留全量 KV。"""

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


print(testing.check(LocalGlobal(), CONFIG).passed)
```

```text
True
```

```python
@dataclass(frozen=True)
class RecentMass(pllm.Workload):
    """监控第 12 层落在最近 64 个位置上的注意力占比，输出保持原模型不变。"""

    recent: int = 64

    @pllm.custom_attention(layers=(12,))
    def split(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = kv.attend(q)
        _, near = kv.attend(q, window=self.recent)
        # aux 返回最近位置分到的注意力比例：exp(LSE_local - LSE_global)
        return out, (near - lse).exp().mean(-1, keepdim=True)

    @pllm.export(name="recent_mass", max_rows=8, max_cols=1)
    def report(self, share: torch.Tensor = pllm.attention(12, "aux")) -> torch.Tensor:
        return share


print(testing.check(RecentMass(), CONFIG).passed)
```

```text
True
```

### 方法签名与 `PagedKV`
自定义 attention 方法的签名为 `f(self, q: torch.Tensor, kv: pllm.model.PagedKV, ...markers)`：
- `q`：加过 RoPE 的 Query 张量，形状为 `[rows, heads, head_dim]`。
- `kv`：只读的分页 KV 视图 `pllm.model.PagedKV`。当前 token 的 K/V 已在此前写入缓存。该对象暴露 `key_cache`、`value_cache`、`block_table`、`seq_lens` 等属性供自定义算法分析。
- `kv.attend(q, *, window=0, topk=None)`：执行 attention 计算，返回元组 `(输出, float32 LSE [rows, heads])`。
  - `window`：从当前位置往前数的回溯窗口大小（含当前 token）。
  - `topk`：逻辑 token 位置张量，形状为 `[rows, k]` 或 `[rows, heads, k]`（不足处用 -1 补齐）。
- **可选返回值 `aux`**：方法可返回 `(out, aux)`。后续层或导出方法可通过 `pllm.attention(layer, "aux")` 读取该张量；`"lse"` 则读取第一次调用 `kv.attend` 返回的 float32 log-sum-exp。

### 静态 KV 布局声明
`@pllm.custom_attention` 允许在装饰器中直接声明底层 KV 缓存的物理留存策略：
- `keep="window", window=N`：仅保留最近 $N$ 个 token 的 KV，更早的物理块由 vLLM 的 `SlidingWindowSpec` 自动回收至显存池。
- `sink=True`（需配合 `keep="window", window=N`）：保留整个 prompt 加上最近 $N$ 个生成的 token（对应 vLLM 的 `RSWASpec`）。
- `keep="none", source=k`（$k \ge 1$）：本层完全不分配独立的 KV 显存，直接读取下方第 $k$ 层的 KV 缓存（源层必须是更早的层；若需读取更深层的 KV，请改用 `@pllm.custom_forward` 的 `kv="peek"`）。

### Top-K 选择只省读取不省显存，以及按分数驱逐不支持
- **Top-K 为什么只省读不省显存**：
  在 `kv.attend(q, topk=...)` 中传入选中的位置索引，FlashAttention 内核在执行 softmax 和 GEMM 时只加载这部分 key 和 value，因而节省了计算 FLOPs 与显存读取带宽。然而，所有历史 token 的 KV 仍然必须完整存留在物理显存页中，供后续新生成的 token 重新计算打分并挑选。因此，Top-K 选择无法释放物理显存容量。
- **为什么不支持按分数真正驱逐（Score-based Eviction）**：
  像 H2O、SnapKV 这类按注意力分数在序列中间剔除物理块的算法，在当前架构下不支持（详见[设计取舍](../explain/design-tradeoffs.md)）。
  **原因**：vLLM 的物理块管理器基于强假设构建——每个请求的逻辑 KV 序列是从位置 0 开始连续递增的一维数组，其块表只增不减。如果在中间任意“挖洞”释放显存块，会导致块表寻址、位置编码计算与连续批处理调度逻辑彻底失效。

### 运行开销与限制
- **开销**：
  - 静态布局能实质减少物理显存占用。在 Qwen2.5-0.5B（上下文 2048 token）实测中，单请求 KV 显存上限从全量的 **24.0 MiB** 降至偶数层窗口 256 的 **16.6 MiB**，跨层共享则降至 **12.0 MiB**，并发容量翻倍。
  - 同一个示例在 RTX 4090 上测得：原样读取的 `NativeRead` 吞吐为原生 vLLM 的 **98%**；偶数层窗口 `LocalGlobal` 为 **97%**，跨层共享 `CrossLayerShared` 为 **96%**（窗口层走 vLLM 的 FlashInfer 窗口内核，见[性能与显存](../explain/performance.md)）。纯 PyTorch 实现的 Top-K 动态打分（如未写成专用内核的 Quest）吞吐仅为原生的 **7%**，生产环境加速需使用融合算子（详见[自写算子](custom-ops.md)与 [09 自写 Triton 算子](../examples/09_custom_kernels.ipynb)）。
- **限制与原因**：
  - `tensor_parallel_size=1`：自定义 FlashAttention 读法与分页 KV 布局尚未实现跨 TP rank 的注意力头切分与通信同步。
  - 仅限 FlashAttention 2 后端：底层依赖 FA2 暴露的低级接口与元数据。
  - 未量化 KV（`kv_cache_dtype="auto"`）：FP8 等量化格式打包了存储字节，在稀疏读取 gather 时需要专用的反量化内核。
  - 普通 RoPE 注意力：不支持带有 ALiBi、Logit Soft-capping 或硬件特定 Sink Token 的注意力模型。
  - 关闭前缀缓存：滑动窗口与跨层共享会改变每个请求实际留存的 KV 结构，破坏前缀缓存的全局哈希树假设。
  - 仅限全部分页模型：仅支持 KV 全部存放在 vLLM 分页块中的模型（Qwen2/2.5、Llama、Qwen3 等）；带有额外递归状态的模型（如 Qwen3.5 DeltaNet）不支持。
  - 静态布局需要 `runner="mrv2"` 且不能与 `PositionState` 或 `@pllm.custom_forward` 组合：静态布局会将 vLLM 划分为多个独立的 KV 组，而自定义前向与跨步状态要求单一全局的 KV 组管理。

---

## 3. 改写层的执行顺序：`@pllm.custom_forward`

### 适用场景与动机
当算法需要完全重排甚至动态改变层的前向执行轨迹时，使用 `@pllm.custom_forward`。典型场景包括：
- **循环深度（Looped / Recurrent Depth）**：如 Huginn、Ouro 等模型，在 decode 阶段将一段层重复执行多遍，用额外的推理时间换取更强的推理能力。
- **逐 token 提前退出 / 自适应深度（Early Exit / Adaptive Depth）**：如 CALM、LayerSkip，在生成过程中让简单的 token 提前跳过深层。
- **连续思维跨步状态（Continuous Thought）**：如 Coconut，通过 `pllm.PositionState` 将上一个 token 产生的 hidden 向量直接喂给下一个 token 作为输入。
- **投机试算**：在不破坏主干 KV 缓存的前提下，向前“试算一步”。

在标准推理中，模型前向按固定拓扑从第 0 层顺序跑至最后一层。如果要在外部写循环或跳层，不仅需要处理极其复杂的 KV 缓存状态维护，还会带来密集的 host-device 阻塞同步。`@pllm.custom_forward` 将整个前向过程封装为可分段调用的 `layers(...)` 对象，由用户在 Python 中编排计算逻辑，框架在底层自动切分并捕获分段 CUDA graph。

### 最小可运行示例（CPU）
下面的示例实现 Ouro 风格的循环深度与逐 token 提前退出：在 decode 阶段将第 20 到 23 层重复执行最多 2 遍；若某行残差变化量小于阈值，则提前收敛停止循环。

```python
@dataclass(frozen=True)
class LatentLoop(pllm.Workload):
    start: int = 20
    loops: int = 2
    tol: float = 0.05

    @pllm.custom_forward(tokens="decode")  # active= 仅限 decode 阶段
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        active = torch.ones_like(hidden[:, :1], dtype=torch.bool)
        for _ in range(self.loops):
            again = layers(start=self.start, hidden=hidden, active=active)[-1]
            change = (again - hidden).float().norm(dim=-1, keepdim=True) / hidden.float().norm(dim=-1, keepdim=True)
            hidden = torch.where(active, again, hidden)
            active = active & (change > self.tol)  # 收敛的行在后续循环中停算
        return hidden


print(testing.check(LatentLoop(), CONFIG).passed)
```

```text
True
```

### `layers(...)` 调用、`LayerOutputs` 索引与 KV 机制
`layers` 参数是 `pllm.model.Layers` 实例，提供以下调用契约：
- **调用形式**：`layers(start=0, stop=None, hidden=None, *, kv="commit", active=None) -> LayerOutputs`
  - 运行第 `start` 到 `stop - 1` 层。若不传 `stop`，默认运行至最后一层；若不传 `hidden`，默认以 token embedding 作为起始输入（仅在 `start=0` 时合法）。
- **`LayerOutputs` 的绝对索引规则**：
  `LayerOutputs` 必须按**模型的绝对层号**索引！例如 `out = layers(12, 16)` 运行了第 12–15 层，访问第 15 层的输出必须写 `out[15]`；负数索引 `out[-1]` 恒定指向本次调用运行的最后一层。若使用了切片范围之外的索引（如 `out[0]`），会抛出 `ValueError: index a tap inside the Layers slice that produced it`。
- **`kv="commit"` 与 `kv="peek"` 的本质区别**：
  - `kv="commit"`（默认）：计算当前分段各层的 K/V 并**写入** KV cache。在循环调用中，后一遍计算出的 K/V 会直接覆盖前一遍在当前位置的槽位，后续生成的 token 将看到最后一遍循环写入的 K/V。
  - `kv="peek"`：结合历史已提交的 KV 缓存与本遍新计算的 K/V 参与 attention 计算，但**不写入** KV cache（缓存被完全保留在本次调用前的状态）。常用于投机预演或试算。
  - **Ouro 论文的对比实验数据**：
    Ouro 论文（arXiv 2510.25741，§5.4.2 表 14）深入对比了 decode 阶段层循环中不同 KV 处理机制在 GSM8K 上的表现：
    - 每一遍循环都独立存一份 KV：准确率 **78.92**。
    - 只留最后一遍的 KV（即默认的每遍覆盖写回 `kv="commit"`）：准确率 **78.85**（与全量保留几乎无损）。
    - 只留第一遍的 KV（第一遍 commit，后续循环改用 `kv="peek"`）：准确率暴跌至 **18.73**。
    这是因为如果后续 token 只能读到浅层第一遍产出的粗糙 K/V，整个基于深层特征训练的模型会彻底崩塌。因此循环中默认使用 `kv="commit"` 覆盖写回。

### 逐 Token 跳层（`active=`）为什么仅限 `tokens="decode"`
当向 `layers(..., active=mask)` 传入布尔掩码（形状 `[rows, 1]`）时，掩码为 `False` 的行不参与该分段的矩阵计算，其 `hidden` 原样穿透传递。
对于被跳过的行，框架会在其第一次跳过某层时，用该行停下的隐藏状态经过该层的输入 norm、QKV 投影与 RoPE 补写对应的 K/V（状态传播），之后再跳过该层时保留该值。

**为什么 `active=` 严格限制在 `tokens="decode"` 中？**
跳层补写的 K/V 是近似值。decode 时被跳过的只是一个新位置；prompt 的每个 token 都是之后所有生成 token 要读的上下文，在 prefill 时跳层会把整段上下文在高层的 K/V 换成近似值，误差影响此后生成的每一步。因此 prompt 默认跑完整模型，`active=` 只用于 decode。

### 跨步状态：`pllm.PositionState` 与 `load` / `save`
对于类似 Coconut 的“连续思维”算法，模型需要把上一个 token 产生的 hidden 向量直接带给下一个 token。这可以通过声明 `pllm.PositionState` 实现：

```python
@dataclass(frozen=True)
class StepLoopWorkload(pllm.Workload):
    """保存每个 token 的 pre-norm 残差，在生成位置作为第 0 层的起始输入。"""

    history = pllm.PositionState()

    @pllm.custom_forward(tokens="all")
    def loop(self, layers: pllm.model.Layers) -> torch.Tensor:
        previous = layers.load("history", offset=-1)
        taps = layers(hidden=torch.where(layers.generated, previous, layers.embeddings))
        layers.save("history", taps[-1])
        return taps[-1]


print(testing.check(StepLoopWorkload(), CONFIG).passed)
```

```text
True
```

- **调用规则**：`layers.load("history", offset=-1)` 读取前一个位置保存的向量（即使当前步骤已先调用过 `save`，它依然返回上一个 token 的快照）；`layers.save("history", value)` 提交当前位置的快照（多次调用以最后一次为准）。
- **整段 Prompt 跨度（`over="span"`）**：
  默认 `over="token"` 模式按因果顺序逐位置推进。设 `@pllm.custom_forward(over="span")` 时，一个请求的一整段连续 prompt 被打包为 `[M, H]` 张量统一送入，适用于 Jacobi 式全 prompt 并行迭代。注意 span 模式下没有串行前驱 token，不能调用 `layers.load`。

### 控制流规范：有界循环、禁止张量分支、`pllm.scan`
1. **有界 Python 循环自动展开**：基于负载字段的 Python `for` 循环（如 `for _ in range(self.loops):`）会在追踪期直接展开录入分段 CUDA graph。
2. **禁止基于张量值的控制流**：方法体内部绝对**不能**出现 `if tensor.norm() > 1:`、`while` 或 `break`。前向图的拓扑结构必须确定，否则追踪时报错 `methods run inside CUDA graphs, so Python cannot branch on tensor values (if, while or break on a tensor); ...`。循环固定次数，条件用 `torch.where` 或 `active=` 掩码表达（`active=` 需 `tokens="decode"`，见 [12 自适应深度](../examples/12_adaptive_depth.ipynb)）。
3. **线性扫描（`pllm.scan`）**：若在自定义前向中需要实现长序列线性递推（形如 $S_t = A_t S_{t-1} + B_t$），应调用 `pllm.scan(step, carry=..., over=..., chunk=...)`。pllm 会识别仿射递推，在条件允许时改用 GPU 上的**并行前缀扫描**，比逐块串行的循环快得多（见[性能与显存](../explain/performance.md)）。详见 [04 改写 forward](../tutorials/04_custom_forward.ipynb)。

### 运行开销、限制与组合
- **运行开销**：
  使用 `@pllm.custom_forward` 时，vLLM 原生的大模型整图不再生效，`eager_reason` 会明确标明 `@pllm.custom_forward`。**但这绝不意味着模型退回了逐算子 eager 运行**：`pllm` 为每次调用的层分段分别捕获了独立的分段 CUDA graph。
  在 [12 自适应深度](../examples/12_adaptive_depth.ipynb) 实测中，一个无跳层、分段调用的 `custom_forward` 吞吐达到了原生 vLLM CUDA graph 的 **93%**（远高于原生 eager 模式的 41%）。
- **显存与硬件限制**：
  - 分段图捕获需要更大的池空间，建议在配置中指定较大的捕获预算（例如 `ExecutionOptions(cuda_graph_pool_budget_bytes=4 << 30)` 即 4 GiB）；若预算不足，生成时会直接报错并指明所需的精确字节数。
  - 多卡张量并行要求使用 vLLM 0.32.0 与 `runner="mrv2"`。
- **与状态读取器（`pllm.hidden`）互斥**：
  在同一个负载中，挂了 `@pllm.custom_forward` 后**不能**再声明读取 `pllm.hidden(layer)` 或 `pllm.final_hidden()`。
  **原因**：在循环或跳层前向中，某一层可能被运行了多次，也可能一次都没运行，“第 $L$ 层的隐藏状态”不再具备唯一的物理定义。如需观测，请改用语义明确的 `pllm.residual(layer)` 或 `pllm.layer_logits((layer,))`，它们稳定读取该层**最后一次被访问时**输出的残差流。
- **与 `pllm.model.Adapter` 的交互**：
  如果负载中同时包含 `pllm.model.Adapter`（详见[改模型](model-edits.md)），权重增量 $\Delta W$ 严格**跟随每一次实际发生的投影调用**生效。若某段层在循环中被执行了 $N$ 遍，增量就会被施加 $N$ 遍；跳层时为补齐 K/V 而触发的原生 QKV 计算也会施加对应的 Q/V 增量。
- **训练 Overlay 与当步生效（`apply_updates()`）**：
  当与 `@pllm.train_in_forward` 组合时，在 `custom_forward` 内部调用 `layers.apply_updates()` 会立即结算并拼接自上个边界以来的投影捕获，执行优化器更新。当前 token 后续的层调用将当场读到更新后的私有权重版本。

---

## 常见报错与排查

在编写或运行改写 forward 结构的负载时，如果违反了前向契约，引擎在声明期或准入期会给出明确的报错信息：

| 报错信息（节选自源码） | 触发原因 | 正确改法 |
| --- | --- | --- |
| `...: take the site tensors ... first, without defaults, then markers` | `@pllm.custom_residual` 方法没有把位置张量（如 `x, y`）写在最前面，或为其赋予了默认值 | 将位置张量作为前两个位置参数（不赋默认值），标记参数（如 `pllm.parameter`）写在后面 |
| `... advances its memory once per token, but a @pllm.custom_forward may run its layer several times or not at all per token; drop state= or the custom forward` | 将带有 `state=` 的状态化残差记忆子层与 `@pllm.custom_forward` 放在了同一个负载中 | 将两者解耦：状态子层按单 token 线性步进，与层循环/跳层互斥 |
| `disable prefix caching (enable_prefix_caching=False) with pllm.model.Adapter, pllm.model.ActivationPatch, @pllm.train_in_forward, @pllm.custom_residual or @pllm.custom_attention: ...` | 在开启前缀缓存（`enable_prefix_caching=True`）的同时使用了改写模型站点的装饰器 | 设置 `ModelConfig(enable_prefix_caching=False)` |
| `@pllm.custom_attention needs tensor_parallel_size=1: each rank sees only its heads` | 在多卡张量并行（`tensor_parallel_size > 1`）下启用了 `@pllm.custom_attention` | 将张量并行大小设为 `1` |
| `@pllm.custom_attention needs vLLM's FlashAttention 2 backend; set VLLM_ATTENTION_BACKEND=FLASH_ATTN and VLLM_FLASH_ATTN_VERSION=2` | 注意力后端未配置为 FlashAttention 2 | 保持默认或配置环境变量 `VLLM_ATTENTION_BACKEND=FLASH_ATTN` |
| `@pllm.custom_attention needs unquantized KV; set kv_cache_dtype=auto` | 在量化 KV 缓存（如 FP8、INT4）的环境下使用了自定义 attention | 将 `kv_cache_dtype` 保持为默认的 `"auto"`（非量化） |
| `static KV layouts cannot accompany PositionState or @pllm.custom_forward (including kv_copy)` | 在静态 KV 布局（滑动窗口/共享）下声明了 `PositionState` 或 `@pllm.custom_forward` | 静态布局要求独立的 KV 组，与需要单一全局 KV 组的自定义前向互斥 |
| `Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. Prompt rows run the full model by default; add an unmasked tokens="prefill" method only to change prompt computation (prefer over="span").` | 在包含 prompt 阶段（`tokens="all"` 或 `tokens="prefill"`）的自定义前向中传入了 `active=` 掩码 | 将装饰器指定为 `@pllm.custom_forward(tokens="decode")`，让 prompt 阶段保持运行完整模型 |
| `... reads block or model outputs, which a @pllm.custom_forward does not produce; use pllm.residual or pllm.layer_logits to read each layer's last visited stream` | 自定义前向负载中声明了读取 `pllm.hidden` 或 `pllm.final_hidden` 的方法 | 改用 `pllm.residual(layer)` 或 `pllm.layer_logits((layer,))` 读取每层最后一次访问的流 |
| `index a tap inside the Layers slice that produced it` | 索引 `LayerOutputs` 时使用了超出该分段起始/终止范围的绝对层号 | 使用该分段范围内的绝对层号（例如 `layers(12, 16)[15]`），或使用 `[-1]` 取最后运行的一层 |
| `custom-forward CUDA graphs need ... bytes; raise ExecutionOptions.cuda_graph_pool_budget_bytes above ...` | 自定义前向分段 CUDA graph 捕获内存超出预算 | 调大 `ExecutionOptions(cuda_graph_pool_budget_bytes=...)` 至报错提示的字节数 |

---

## 自测

1. 为什么 `@pllm.custom_residual` 的恒等改写（返回 `x + y` 或 `v`）与 eager 模式的原生 vLLM 逐位相同，却与默认图模式的原生 vLLM 存在微小 bf16 舍入差异？
<details>
<summary>答案</summary>

默认图模式下的原生 vLLM 会使用 PyTorch Inductor 融合算子（例如 Fused Add+Norm）。当挂载 `@pllm.custom_residual` 时，vLLM 改为捕获模型原本的非融合算子到 CUDA graph 中。因此数学运算与 eager 模式完全逐位一致，但和 Inductor 融合算子之间存在正常的 bf16 浮点舍入差异。
</details>

2. 为什么 `kv.attend(q, topk=...)` 能够节省读取带宽与 FLOPs，却无法降低每个请求占用的 KV 显存容量？
<details>
<summary>答案</summary>

`kv.attend(q, topk=...)` 只是在计算 attention softmax 和 GEMM 时仅读取被选中的位置，从而节省了显存读取带宽。但所有历史 token 的 KV 仍然必须常驻在物理显存页中，供未来的 token 计算相关性分数并从中挑选，因此无法释放物理显存块。
</details>

3. 在循环深度模型中，`layers(...)` 默认 `kv="commit"` 与 `kv="peek"` 分别有什么影响？为什么 Ouro 论文发现每一遍都覆盖写回（保留最后一遍）效果优异（78.85），而只留第一遍（18.73）会导致模型崩掉？
<details>
<summary>答案</summary>

`kv="commit"` 每一遍都把算出的 K/V 写进缓存，后一遍覆盖前一遍，因此后续 token 看到的是这个 token 最后一遍循环提炼后的高质量 K/V；Ouro 论文实测 GSM8K 达 78.85。若改用 `kv="peek"`（或只在第一遍 commit、后续 peek），缓存里留下的只是第一遍浅层计算的粗糙 K/V，使得后续 token 无法利用深层循环的特征，导致按循环训练过的模型崩掉（GSM8K 仅 18.73）。
</details>
