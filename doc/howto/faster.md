# 让负载更快

你的算法负载已经跑通、结果正确，但发现自回归解码（decode）的端到端吞吐低于预期，或者服务刚启动就报显存不足。

这篇操作指南是一份按排查优先级排序的优化清单。自回归解码与离线训练不同：GPU 每步只为每个并发请求生成 1 个 token，计算强度（FLOPs/字节）极低，算力通常过剩，真正的瓶颈集中在三个物理制约：**GPU 时间线气泡（主机同步或算子发射阻塞）**、**显存与总线搬运带宽（全词表投影、状态读写、PCIe 传输）**，以及 **KV Cache 显存容量**。

关于这些物理制约的底层推导与硬件模型，参见 [性能与显存](../explain/performance.md)；阅读执行报告的字段定义，参见 [看成本：pllm.explain](../guides/explain.md)。

> [!NOTE] 基准测试环境与标尺数字
> 除非另有说明，本文引用的实测数字都来自同一基准环境：**单卡 RTX 4090（24 GiB）、`Qwen/Qwen2.5-7B-Instruct`（BF16）、vLLM 0.32.0、128 条并发序列（8 条 prompt $\times$ 16 个采样分支）、每条生成 256 个 token、开启 CUDA Graph；`pllm.benchmark.throughput` 预热 1 轮后取 3 轮中位数**。吞吐百分比均相对于同样设置下原生 vLLM（开启 CUDA Graph）的吞吐（约 $5,894\text{ tok/s}$，按 128 行折算单步约 $21.7\text{ ms}$）。同样设置换成 0.5B 模型，一步只有约 $3.0\text{ ms}$；下文在结论随模型大小改变的地方给出两者对照。

```python
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import pllm
from pllm import testing
```

---

## 1. 确认前向是否仍在 CUDA Graph 中

### 看什么

运行 `pllm.explain(workload, config)`，观察输出的 `forward` 行：

```text
forward  CUDA graphs: no effect needs eager
```

若退回 eager 模式，该行会显示 `forward eager: <reason>`。在启动引擎后，也可以通过查询运行时对象 `llm.diagnostics().eager_reason` 确认（图模式下为空字符串 `""`；退回 eager 时为具体原因）。

### 意味着什么

如果退回 eager 模式，每一步都要从 Python 逐个发射几百个算子。这份 CPU 开销是否致命，取决于一步 GPU 执行有多长：7B 在 128 并发下一步约 21.7 ms，主要花在读权重上，**退回 eager 后吞吐仍有图模式的 98%**（$5,773$ 对 $5,894\text{ tok/s}$）；同样设置的 0.5B 一步只有约 3 ms，**eager 掉到图模式的 49%**（$20,926$ 对 $42,355\text{ tok/s}$）。模型越小、并发越低、步越短，CUDA Graph 越要紧；图模式还是 pLLM 把训练和改写录进同一张图的前提。

### 怎么改

排查哪些负载声明破坏了整模型图：
1. **共享可变实例状态上的适配器**：给共享可变的 `pllm.InstanceState` 绑定 `pllm.model.Adapter` 会迫使模型前向退回 eager（`forward eager: CUDA-graph overlays do not support adapters backed by shared InstanceState`）。替代方案是将状态声明为 `pllm.RequestState`（每个请求持有一份状态副本），或者使用静态只读参数；这样适配器就可以完整录入 CUDA Graph。
2. **激活补丁与切片图**：`pllm.model.ActivationPatch` 在 CUDA Graph 内部执行，不会导致退回 eager。`@pllm.custom_forward` 则会将整模型图替换为切片捕获图（`eager_reason` 中会显式注明 `@pllm.custom_forward`）。
3. **配置强制报错**：在配置中声明 `ModelConfig("Qwen/Qwen2.5-7B-Instruct", cuda_graphs=True)`。默认的 `cuda_graphs="auto"` 遇到暂时不能进图的写法时以 eager 运行（原因写在 `forward` 行和 `eager_reason` 里）；将其显式设为 `True` 后，任何导致退回 eager 的写法都会在 `explain` 或 `make_llm` 阶段直接抛出 `ValueError`，杜绝吞吐隐蔽下降。

```python
@dataclass(frozen=True)
class FastLoRA(pllm.Workload):
    # 使用 RequestState：每个请求持有一份状态副本，适配器可完整录入 CUDA Graph
    lora = pllm.RequestState(
        {
            "a": pllm.Field(lambda in_features, out_features: (4, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 4)),
        },
        init=pllm.Imported(),
    )
    apply = pllm.model.Adapter(
        "lora",
        (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12)),
        operator="low_rank",
    )


# 开启 cuda_graphs=True：若有任何导致退回 eager 的写法，构建期直接报错。
# 128 个请求各带一份 LoRA 时，改写后模型的 CUDA graph 要更大的捕获池（不够时启动报错并给出需要的字节数）。
config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct",
    max_model_len=512,
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    cuda_graphs=True,
    execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=512 << 20),
)
report = pllm.explain(FastLoRA(), config)
assert report.eager_reason == ""
```

### 代价是什么

将共享 `InstanceState` 改为每请求 `RequestState` 会使适配器权重的显存预留随最大并发请求数线性扩展（每个请求槽位多存一份状态副本；上例在 7B、128 并发下 `explain` 报 514 份 × 88 KiB ≈ 44 MiB），改写后模型的 CUDA graph 也要更大的捕获池（不够时启动报错并给出需要的字节数），但换来的是前向完全进图。图模式能挽回多少吞吐取决于步长：7B 上 eager 本来只慢约 2%，0.5B 上 eager 只有图模式的一半。设置 `cuda_graphs=True` 没有任何运行时性能开销。

---

## 2. 消除数学运算中的主机同步

### 看什么

查看 `pllm.explain` 输出中的 `host` 行：

```text
host     0 syncs in traced workload math
```

该行必须显示为 `0 syncs`。对于模型层内部改写（`@pllm.custom_residual`、`@pllm.custom_attention`）和自定义算子，CPU 追踪会显示为 `unverified syncs`，需运行 `pllm.testing.check_on_gpu(workload, config)` 验证。

### 意味着什么

pLLM 依赖 CPU 超前 GPU 异步准备元数据。一旦在负载方法中调用了 `.item()`、`.tolist()`、`.cpu()`、`print(tensor)`、`float(tensor)` 或 Python 值分支 `if tensor.any():`，CPU 就必须阻塞等待 GPU 将之前发射的所有算子执行完毕并把标量拷回主机内存。这一主机同步会彻底排空 GPU 执行流水线，造成数百微秒至毫秒级的 GPU 停顿。

### 怎么改

将标量提取与动态条件分支改写为全张量数学：
1. **替代分支**：Python `if condition:` 改用 `torch.where` 或布尔掩码乘法。
2. **禁止标量转换**：不要在方法内调用 `.item()` 或在循环中打印张量。
3. **异步观察与健康诊断**：调试数值或收集统计量时，改用 `@pllm.export` 异步导出（数据写入有界锁页环形缓冲区，队列满时默认丢弃而不阻塞 GPU 或 CPU）；运行时可调用 `llm.diagnostics().numeric_health` 查询状态张量中是否存在 NaN/Inf，这一检查在稳态解码期间产生 0 次每步同步。
4. **GPU 契约检查**：运行 `pllm.testing.check_on_gpu(workload, config)`，它会在 `torch.cuda.set_sync_debug_mode("error")` 下实跑至少 8 个稳态解码步，抓出层内所有隐式同步。

```python
@dataclass(frozen=True)
class ZeroSyncWorkload(pllm.Workload):
    # 用 torch.where 替代 Python if 分支，不调用 .item()，确保零主机同步
    @pllm.add_logits()
    def steer(self, logits: torch.Tensor = pllm.layer_logits((-1,))) -> torch.Tensor:
        last = logits[:, -1]  # [rows, vocab]：模型自己的 logits
        return torch.where(last > 0.0, 0.1 * last, torch.zeros_like(last))  # 只放大正的 logits


sync_report = pllm.explain(ZeroSyncWorkload(), config)
assert sync_report.workload_host_syncs == 0
```

### 代价是什么

张量分支（如 `torch.where`）会同时执行两个分支的数学计算，但在 GPU 充裕的算力下，这点计算耗时相比排空流水线的毫秒级停顿微不足道。

---

## 3. 确保旁路流任务与前向重叠

### 看什么

1. 查看 `pllm.explain` 的 `member` 列表中各任务的 `lane` 列是否显示为 `side`（旁路 CUDA stream）还是退回了 `main`（主 stream）。
2. 查看 `timing` 行是否显示 `publish_lag=1 (default)`。
3. 查看 `memory` 行打印的两个阈值：

```text
memory   peak 1512728864 bytes funded by pLLM [per-rank tensor reservations; excludes GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1512728864 bytes tasks run inline; below 1333743904 bytes planning fails
```

检查你配置的 `execution_peak_budget_bytes` 是否低于第一行提示的重叠阈值（上例是[性能与显存](../explain/performance.md)第 2 节的 `Distill` 在 7B、128 并发下的输出：`1512728864 bytes`）。

### 意味着什么

在线训练（`@pllm.train`）等较重的计算默认调度在旁路 stream（`side`）上异步执行。
* **默认 `publish_lag=1`**：第 $n$ 步 token 产生后，旁路流异步执行反向传播和优化器更新，更新在第 $n+2$ 步生效；第 $n+1$ 步前向直接读取第 $n-1$ 步的状态版本，无需等待第 $n$ 步的训练，实现训练与前向的并行重叠。
* **显存预算降级**：旁路重叠需要同时保留两个版本的工作区缓冲，显存峰值高于串行执行。如果你设置的 `ExecutionOptions(execution_peak_budget_bytes=...)` 低于重叠阈值（但高于规划失败下限），规划器不会直接报错，而是自动将旁路任务退回主 stream 内联执行（`inline`，`lane` 从 `side` 变为 `main`），任务与模型前向前向串行，失去重叠。

### 怎么改

1. **保持滞后**：保持默认的 `publish_lag=1`，不要无故设为 `publish_lag=0`（除非算法在数学上严格要求下一步立即看到当前步的参数更新）。
2. **调大显存预算**：如果 `lane` 退回了 `main`，检查 `explain` 的 `memory` 行，将 `execution_peak_budget_bytes` 调大至超过重叠阈值，恢复旁路重叠。

```python
exec_opts = pllm.ExecutionOptions(
    execution_peak_budget_bytes=1024 << 20,  # 高于 overlap_threshold 阈值，保障旁路重叠
    publish_lag=1,  # 默认 1：第 n 步更新在第 n+2 步生效，训练与第 n+1 步前向重叠
)
overlap_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct",
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    execution=exec_opts,
)
```

### 代价是什么

重叠省下多少，取决于训练在整步里占多大比例（`ESamp`，rank 4）：
* **7B**：纯在线训练（$\beta=0$）时，`publish_lag=1` 为原生 vLLM 吞吐的 **97.4%**，`publish_lag=0` 为 **97.2%**；训练 + 引导采样（$\beta=0.1$）时为 **96.4%** 和 **96.1%**。每步训练只有约 0.6 ms GPU 时间，相对 21.7 ms 的一步，串行等待的代价在噪声附近。
* **0.5B（同样设置，只换模型）**：纯在线训练时 `publish_lag=1` 为 **91.9%**，`publish_lag=0` 为 **89.8%**。步越短，重叠越值钱。

代价是显存预留增加：`publish_lag=1` 下每个可写请求状态需要预留更多版本槽位（槽位公式为 $\text{roots} = (3 + \text{publish\_lag}) \times \text{副本数} + 2$，128 并发下从 386 份增至 514 份）；对 7B 上的 `Distill`，状态预留从约 190 MiB 增至约 253 MiB。

---

## 4. 削减每步搬运字节数

### 看什么

查看 `pllm.explain` 的任务估算行与状态行：
* `projection FLOP`：是否有额外的 LM Head 全词表投影。
* `sampler ... bytes`：全词表张量的读写字节数。
* `state reads (origin, distance)`：状态读取字节数。
* `state` 行的 `logical transfer bytes/row` 与 `decode bytes/step`：跨 PCIe 主机查表的每步传输字节。

### 意味着什么

解码阶段受限于显存与总线带宽。每步多搬运数据都会直接增加延迟：
1. **LM Head 投影与全词表张量**：7B 的 LM Head 权重（3584 $\times$ 152064 BF16）重达 1.09 GB。多算一次全词表投影耗时约 **$+1.35\text{ ms}$**（约一步的 6%；0.5B 的 272 MB 权重约 0.36 ms，却占它一步的 12%）；采样器扫描一遍 $128 \times 152064$ 的词表张量，每个阶段需 78–283 μs。
2. **大表跨 PCIe 传输**：4090 上 PCIe 4.0 x16 读主机内存实测约 23.7 GB/s，约为 GPU 显存带宽的 1/40。128 行查表搬运 38.9 MB 宽向量需要实打实的 **1.6 ms**（7B 一步的约 7%；0.5B 上超过半步）。

### 怎么改

#### 1) 投影合并与引导采样优化

* **使用 `@pllm.guide_sampling`**：引导采样返回 `[rows, hidden]` 隐藏层预测向量，框架默认使用 `guidance_capacity=4096` 的有界融合路径，只对过滤后保留的 top-K token 做投影和加权，不产生额外的独立全词表 GEMM。
* **避免多层 `pllm.layer_logits`**：每个 `layer_logits` 都会触发一次独立的全词表 GEMM，重复读取 LM Head 权重（7B 为 1.09 GB）。在 7B 上（32 个并发请求），读 9 层的 DoLa 吞吐为原生的 **66%**；模型越小，这部分占比越大。
* **按行过滤**：针对非全局干预，给效果方法加上 `rows=` 过滤器（如 `rows="trained"`），缩小参与计算的请求行数。

```python
@dataclass(frozen=True)
class FastGuidance(pllm.Workload):
    beta: float = 0.5

    def strength(self) -> float:
        return self.beta

    # 优先使用 @pllm.guide_sampling：只返回 [rows, hidden]，与采样器深度融合
    @pllm.guide_sampling(beta=strength)
    def steer(self, h: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return h


guidance_report = pllm.explain(FastGuidance(), config)
```

#### 2) 超大只读表使用 `placement="host"` 并规范方法切分

对于几百 MiB 至几 GiB 的超大只读查找表，在 `pllm.Pretrained(..., placement="host")` 中声明放在主机内存。要让约 1.6 ms 的 PCIe 传输藏在前向计算中，必须遵循以下切分规则：

* **独立查表方法**：查表方法必须只读取 `pllm.input_tokens` 和只读状态（不带效果装饰器），通过 `pllm.result("lookup")` 把查表结果传给后续效果方法。
* **为什么必须这样写**：只有纯依赖 token 和只读表的独立方法，规划器才能将其调度在步首的 `request.prepare / side` 上，使用限制数量的 SM（RTX 4090 上限制为 32 个线程块，即 1/4 SM）在旁路拉取，让传输与主干模型前向重叠；若在查表方法中同时读取了 `pllm.hidden` 或 `pllm.final_hidden`，方法必须等待模型前向结束在 `state.publish / main` 执行，1.6 ms 的传输延迟将完全暴露在时间线上。

```python
# check-docs: skip: demonstrates host lookup table pattern requiring pretrained weight file
@dataclass(frozen=True)
class FastHostLookup(pllm.Workload):
    buckets: int = 2048

    @property
    def memory(self) -> pllm.InstanceState:
        # 每个桶一行词表宽的 logits 先验；7B 上 2048 × 152064 × bf16 约 594 MiB，放在锁页主机内存
        return pllm.InstanceState({"table": lambda m: (self.buckets, m.vocab_size)},
                                  init=pllm.Pretrained("ngram_table.safetensors", placement="host"))

    # 独立查表方法：只读 token 历史和只读表，不加效果装饰器；规划器把它排在 request.prepare / side
    def lookup(self, ids: torch.Tensor = pllm.input_tokens(history=1),
               table: torch.Tensor = pllm.parameter("memory", "table")) -> torch.Tensor:
        keys = (ids[:, 0] * 31 + ids[:, 1] + 1) % self.buckets
        return table[keys]

    # 效果方法通过 pllm.result 读取查表结果
    @pllm.add_logits()
    def prior(self, logits: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return logits
```

### 代价是什么

在 RTX 4090（7B，128 并发，594 MiB 表，示例 11 的 `NGramPrior`）基准测试中：
* 原生 vLLM：5,877 tok/s（约 21.8 ms/步）；
* 设备端查表（`placement="device"`）：5,742 tok/s（98%），`explain` 报的 pLLM 显存峰值 1,540 MiB；
* 主机端查表（`placement="host"`）：5,615 tok/s（96%），显存峰值降至 **946 MiB**（单步只比放显存慢约 0.5 ms，省下整张表的 594 MiB 显存）；
* 引导采样容量（`ESamp`，$\beta=0.5$）：默认 `guidance_capacity=4096` 吞吐为 5,678 tok/s（96.3%），256 和 1024 与它相差在噪声内；精确 dense 路径（`guidance_capacity=None`）为 5,533 tok/s（93.9%）。

<details>
<summary>自测：为什么主机查表混读了 final_hidden 之后每步多出了 1–2 ms？</summary>
因为同一个方法读取了 <code>final_hidden</code>，该方法必须等待模型前向全部计算完毕才能启动（anchor 是 <code>state.publish / main</code>），无法在步首的 <code>request.prepare / side</code> 发射。结果约 1.6 ms 的 PCIe 传输完全暴露在前向之外。解决办法是将只依赖 <code>input_tokens</code> 的查表拆成独立方法，通过 <code>pllm.result</code> 传递给效果方法。
</details>

---

## 5. 优化 KV 显存：区分静态布局与动态读取

### 看什么

1. 查看 `pllm.explain` 的 `kv_capacity` 报告（显示各层的 `KV bytes/request` 与总容量）。
2. 运行时查看 `llm.diagnostics().kv_cache_blocks` 与 `kv_cache_max_concurrency`。

### 意味着什么

在 vLLM 中，显存分配顺序是：模型权重 + CUDA Graph 捕获池 + 负载状态预留 $\rightarrow$ **剩下的显存全部归 KV Cache**。KV Cache 越小，最大并发请求数就越低。

`@pllm.custom_attention` 的两种机制有着根本区别：
1. **静态几何布局（`keep="window"`, `sink=True`, `keep="none", source=k`）**：直接映射到 vLLM 底层的物理块管理器。滑出窗口的物理块被真正回收给显存池，**真正释放 KV 显存容量**，并走原生 FlashAttention-2 / FlashInfer 内核，真正降低读取带宽。
2. **动态稀疏读取（`kv.attend(q, topk=...)`）**：所有历史 token 的 KV 仍然必须全部保存在显存中供每步打分挑选，**完全不节省 KV 显存容量**；它只省注意力的逻辑读取量。如果打分和挑选使用纯 PyTorch 算子写，碎片开销远超省下的读取收益。底层不支持基于分数的显存物理驱逐（如 H2O/SnapKV），因为 vLLM 的物理块表要求按顺序连续分配。

### 怎么改

若目标是降低显存占用、提升系统并发上限，采用静态布局：
* **局部滑动窗口**：`@pllm.custom_attention(layers=tuple(range(0, 28, 2)), keep="window", window=256)`（7B 共 28 层）；
* **跨层共享 KV**：`@pllm.custom_attention(layers=tuple(range(1, 28, 2)), keep="none", source=1)`。

```python
@dataclass(frozen=True)
class LocalGlobalAttention(pllm.Workload):
    # 静态滑动窗口：真正释放滑出窗口的物理块，真省 KV 显存容量
    @pllm.custom_attention(layers=tuple(range(0, 28, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


attn_report = pllm.explain(LocalGlobalAttention(), config)
```

### 代价是什么

在[示例 10](../examples/10_custom_attention.ipynb) 的测试中（RTX 4090，7B，`max_model_len=2048`，128 token 分块 prefill；KV 上限来自 `explain`）：
* **单请求 KV 显存上限**：全量 112.0 MiB；偶数层窗口 256 降至 **70.4 MiB**；奇数层跨层共享降至 **56.0 MiB**（同样的 KV 池能装下的请求数翻倍）。
* **吞吐表现**：128 并发、每个序列约 270 token 时，原样读取为原生 vLLM 的 99%，偶数层窗口 100%，跨层共享 98%（设置见示例 10）。
* **动态稀疏读取对比**：[自定义 attention 教程](../examples/10_custom_attention.ipynb) 中的 `QuestRead`（纯 PyTorch 实现）显存依然为 112.0 MiB，吞吐断崖式下跌至原生的 **14%**（697 tok/s），因为未融合的 PyTorch 打分与 gather 开销远超节省的读取量。

---

## 6. 精细控制精度：平衡带宽与数值稳定性

### 看什么

1. 查看 `pllm.explain` 的 `state` 行中各个字段解析出的 `dtype`（如 `field 0: bfloat16, field 1: float32`）。
2. 运行时查看 `llm.diagnostics().numeric_health`。

### 意味着什么

在带宽受限的自回归解码阶段，状态张量的精度直接决定显存读写搬运量。BF16 相比 FP32 体积减半，读写带宽和状态显存池占用均减半。但在自回归多步递推中，某些敏感变量若使用低精度会产生严重的累积舍入误差，引发数值发散或 NaN。

### 怎么改

1. **默认使用 `dtype="auto"`**：对于只读权重、单步激活或不需要跨步累加的状态字段，`dtype="auto"` 会自动对齐模型的基座精度（BF16/FP16），节省 50% 的状态显存带宽。
2. **显式指定 `torch.float32`**：
   * 跨步累加器、优化器一阶/二阶矩状态；
   * 循环状态携带字段（`carry=`）；
   * 线性前缀扫描 `pllm.scan` 的系数与状态转移矩阵；
   * 写法：`pllm.Field(shape_fn, dtype=torch.float32)`。

```python
@dataclass(frozen=True)
class PrecisionAwareWorkload(pllm.Workload):
    state = pllm.RequestState(
        {
            "momentum": pllm.Field((896,), dtype=torch.float32),  # 显式 FP32：避免累加舍入误差
            "cached_feature": pllm.Field((896,)),  # 默认 auto：跟随模型基座自动选用 BF16
        }
    )

    @pllm.export(name="momentum_norm", max_rows=8, max_cols=1)
    def report(self, m: torch.Tensor = pllm.parameter("state", "momentum")) -> torch.Tensor:
        return m.norm(dim=-1, keepdim=True)


precision_report = pllm.explain(PrecisionAwareWorkload(), config)
```

### 代价是什么

显式声明 FP32 使该字段的物理显存和搬运量增加一倍，但在小尺寸状态字段上，这点开销是保障数值稳定、避免长序列发散的必要成本。

---

## 7. 算子成为瓶颈时：接入自定义算子

### 看什么

1. 查看 `pllm.explain` 的 `custom_ops` 报告（显示算子名称、`flops_per_row` 和 CPU 参考实现）。
2. 查看性能剖析器中是否被大量细碎的 PyTorch 小算子填满时间线。

### 意味着什么

在模型层内（`@pllm.custom_residual`）或注意力（`@pllm.custom_attention`）中直接用 PyTorch 编写多步张量操作，每次调用都会产生算子发射开销与显存往返写入。[自定义 attention 教程](../examples/10_custom_attention.ipynb) 中的 Quest 算法用 PyTorch 编写每页打分和 gather，在 7B 上吞吐仅有原生的 **14%**。

### 怎么改

将计算密集的逻辑用 Triton 或 CUDA 实现为融合 kernel，并用 PyTorch 标准机制注册：
1. 用 `@torch.library.custom_op` 注册具名算子，并通过 `register_fake` 提供静态形状推导。
2. 在负载方法中**直接调用算子函数对象本身**（例如 `fused_step(x, v)`），不要使用 `torch.ops.<namespace>.<op_name>`。直接调用能让规划器静态分析源码并把算子直接录入 CUDA Graph。详细编写与检查步骤参见 [自写算子](../guides/custom-ops.md)。

```python
# check-docs: skip: demonstrates Triton custom_op pattern
@torch.library.custom_op("faster_guide::fused_step", mutates_args=(), device_types="cuda")
def fused_step(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return x + v


@fused_step.register_fake
def _fused_step_fake(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

### 代价是什么

在 RTX 4090 基准测试中：
* 示例 09 在第 12 层挂一个注册为 `custom_op` 的 Triton 算子，吞吐为原生 vLLM 的 **98%**（$5,760$ 对 $5,872\text{ tok/s}$）；
* 同样数学的 PyTorch 写法也是 **98%**（$5,755\text{ tok/s}$）：一次逐行加法本来就不是瓶颈；
* 编写算子需要额外手写 Triton 或 CUDA kernel 并维护 fake 形状函数。值得这样做的是 PyTorch 写起来太慢的步骤，例如示例 10 里用 PyTorch 写的 Quest 页打分，吞吐只有原生的 14%。

---

## 8. 公平地测量吞吐

### 看什么

使用基准测试工具：`pllm.benchmark.throughput(llm, prompts, sampling)` 与 `pllm.benchmark.compare(workload, options)`，观察结果中的 `tokens_per_second` 与 `vanilla_tokens_per_second`。

### 意味着什么

不严谨的测试条件会得出虚假的加速比：
* 若对比的原生 vLLM 运行在 eager 模式，会得出虚高加速比；
* 若未排除首轮预热（warmup），CUDA Graph 录制和 JIT 编译会拉低统计吞吐；
* 并发数、序列长度或采样参数（如 `ignore_eos`）不一致会导致实际计算量不对等。

### 怎么改

遵循公平测试准则（详见 [公平地测吞吐](measure-throughput.md)）：
1. **同配置对比**：同一张 GPU、同一模型权重、相同 CUDA Graph 模式。
2. **固定工作量**：使用相同批次的输入 `prompts`、固定的 `max_tokens`，并设置 `SamplingParams(ignore_eos=True)` 确保生成的总 token 数严格相等。
3. **排除预热**：至少运行 1 次未计时的 warmup，正式测试测量 3 轮以上取中位数。

```python
# check-docs: skip: requires GPU engine and vLLM runtime
from vllm import SamplingParams
from pllm.benchmark import throughput

prompts = ["Explain gradient descent:"] * 8
timed = SamplingParams(max_tokens=256, temperature=0.8, seed=0, ignore_eos=True)

with pllm.make_llm(config) as plain:
    vanilla_tps = throughput(plain, prompts, timed, warmup=1, rounds=3)

with pllm.make_llm(config, workload=FastLoRA()) as llm:
    workload_tps = throughput(llm, prompts, timed, warmup=1, rounds=3)

print(f"相对原生吞吐: {workload_tps / vanilla_tps:.2%}")
```

### 代价是什么

规范的基准测试需要多花费数分钟执行时间，但能提供可信、可复现的真实加速数据。

---

## 总结：性能优化速查清单

| 检查项 | 观察指标（`explain` 或运行时） | 目标状态 | 优化手段 |
| :--- | :--- | :--- | :--- |
| **1. 图模式** | `forward` 行 / `eager_reason` | `CUDA graphs: no effect needs eager` / `""` | 共享可变状态改为 `RequestState`；设置 `cuda_graphs=True` 捕获异常 |
| **2. 主机同步** | `host` 行 | `0 syncs in traced workload math` | Python `if` 改用 `torch.where`；去除 `.item()`；用 `@pllm.export` 异步观察 |
| **3. 旁路重叠** | `member.lane` / `memory` 阈值 | `side` / 预算高于 `overlap_threshold` | 保持 `publish_lag=1`；调大 `execution_peak_budget_bytes` 保障重叠 |
| **4. 搬运字节** | `projection FLOP` / `sampler bytes` | 尽量少全词表 GEMM / 查表重叠率高 | 用 `@pllm.guide_sampling` 替代多层 `layer_logits`；主机表拆独立方法走步首拉取 |
| **5. KV 显存** | `kv_capacity` / `kv_cache_blocks` | 显存容量真正释放 | 采用滑动窗口或跨层共享等静态几何布局；认清动态 Top-K 不省显存容量 |
| **6. 状态精度** | `state` 各字段 `dtype` | 只读 auto 为 BF16，敏感变量显式 FP32 | 累加器、动量与 carry 显式声明 `torch.float32`，其余字段保持 `auto` |
| **7. 算子融合** | `custom_ops` 报告 / 剖析器小算子 | 融合进图，吞吐 $\ge 95\%$ | 用 `@torch.library.custom_op` 接入 Triton 融合 kernel，直接调用算子对象 |
| **8. 公平测量** | `throughput` 中位数对比 | 同硬件、同请求数、同 ignore_eos | 遵循 [公平地测吞吐](measure-throughput.md) 流程，对齐基准条件 |
