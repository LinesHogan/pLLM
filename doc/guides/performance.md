# pllm 替你省下了什么——以内存为中心的性能指南

写推理时算法（在线训练、记忆查表、残差改写、引导采样、稀疏注意力、自适应深度）时，最常见的惊讶是：算法本身没几行张量运算，放进自回归 decode 循环里却慢了一截，或者干脆撑爆显存。

本指南用一条主线解释这件事：**在自回归 decode 阶段，性能几乎完全由“数据停在哪、每步搬多少、搬的时候有没有挡住别人”决定**。读完这篇，你会知道：

1. 你的 workload 额外消耗了哪一级资源；
2. `pllm` 在这一级已经自动替你省下了什么、边界在哪；
3. 你还能通过声明（或预算参数）做什么取舍，以及怎么在 [`pllm.explain`](explain.md) 和运行时诊断里看到它们。

---

## 0. 为什么用“内存”做主线

自回归 decode 每步只为每个请求生成 1 个 token。此时矩阵乘（GEMM）的算术强度（每读一字节权重能做多少次乘加）约等于有效 batch 行数；在小到中等 batch 下，**步时主要由读权重和读写状态/词表的显存带宽决定，而不是由算力决定**。

用一个具体数字看量级：Qwen2.5-0.5B 的 LM head 权重形状是 $896 \times 151936$（BF16），共约 $272\text{ MB}$。在 RTX 4090 上，128 行的 LM head 实测约 $0.38\text{ ms}$（几乎都花在读这块权重上），占原生 vLLM 整步耗时（约 $3.30\text{ ms}$）的十分之一以上。你的算法如果在每步多扫一次词表、多读一份状态、或者阻塞主线程等一次 GPU，开销立刻就能在这个尺度上被看到。

把 GPU 推理涉及的资源按“容量从大到小、带宽从慢到快、停顿代价从高到低”排成一条阶梯，一个推理时算法的额外开销几乎总能归到其中一级：

```text
显存容量                     ── 放不下就无法启动，或挤占 KV cache 降低并发
   ↓
显存带宽                     ── 每步多搬几百 MB，步时直接加几百微秒
   ↓
PCIe 链路 (Host ↔ GPU)       ── ~22 GB/s，搬几十 MB 就要 ~1.7 ms，必须藏在计算后面
   ↓
卡间链路 (TP All-Reduce)     ── 多卡每层同步，旁路任务撞上集合通信就会拖慢主干
   ↓
GPU 时间线气泡 (Timeline)    ── CPU 等 GPU、逐 op 启动开销、过晚发射
   ↓
未填满的 SM (Overlap)        ── GPU 算力单元空闲时能不能塞进旁路工作
   ↓
主机 CPU (Host Bookkeeping)  ── 每步调度与簿记若超过 GPU 步时，就会露出步间空闲
```

**怎么读这篇指南**：写好 workload 后，先跑一行不占 GPU 的 [`pllm.explain(workload, config)`](explain.md)，看它在每一级花了多少（状态字节、张量预留峰值、逻辑读写字节、FLOP、挂在哪个生命周期锚点、走 `main` 还是 `side` 通道），再按下面的章节对照 `pllm` 自动做了什么、你还能改什么。

---

## 1. 显存容量：让状态和 KV 放得下

### 为什么会成为瓶颈

显存容量是硬门槛：模型权重、CUDA graph 私有内存池、pllm 的状态与中间激活预留先分走固定份额，**剩下的才归 vLLM 的 KV cache**。任何一处多预留几百 MiB，要么在 `make_llm` 时直接因预算不足报错，要么把可用 KV 挤到只剩几千个 token——例如在 7B 模型（`gpu_memory_utilization=0.85`）上，如果错误地按调度器最大 token 数（8192）强行捕获模型图，KV 容量会从原生 vLLM 的 76,144 个 token 暴跌到 3,872 个 token，吞吐只剩约 21%（见第 9 章 N15）。

### 核心机制

#### 1) 分页状态池 PTTS（A1）与字段级写时复制（A2）

* **术语**：**PTTS**（Paged Tensor Transition System，分页张量状态转移系统）是 pllm 管理请求状态（`RequestState`）和实例状态（`InstanceState`）的固定页长显存池；**`publish_lag`**（发布滞后）指“第 $n$ 步算出的状态更新，允许推迟到第 $n+1+\text{lag}$ 步才被后续前向读到”，用来换取训练与前向的重叠。
* **它省了什么**：不为每个请求预留整块生命周期缓冲区，也不做整块状态的深拷贝，而是按固定容量页（page）分配，并用引用计数（pin）和完成栅栏（completion fence）回收。当状态发生版本转移时，**只为被写入的字段分配新物理页，未修改字段直接与旧版本共享物理槽位**（字段级写时复制，A2）。
* **需要多少份版本（公式）**：并发读写同一状态时，正在被前向读取的旧版本和正在被旁路训练写入的新版本必须同时存在。pllm 用精确公式决定槽位数（`roots`）：
  * 普通请求状态：$\text{roots} = (3 + \text{publish\_lag}) \times \text{lineages} + 2$
  * 前向内精确更新（`visibility="exact"` decode）：$\text{roots} = 2 \times \text{lineages} + 2$
  * 实例状态：最多按单个 lineage 的版本公式取槽位（例如下面的 `Steering` 为 6 个），不可变的实例状态只需 1 个。
* **实测数字**：以 `explain.md` 中的单向量引导负载 `Steering`（1 个 896 维向量）为例，收紧实例状态容量并配合自动精度（A3）后，预留从 `514 roots × 3,584 B = 1,842,188 B` 降到 `6 roots × 1,792 B = 10,766 B`（`explain@5cdc9785`）。而在 128 并发序列、rank 4 的 `ESamp` 在线训练中，请求状态预留为 `514 roots × 200,784 B = 103,203,036 B`（约 98.4 MiB）。
* **适用边界**：超出 `ExecutionOptions.state_memory_bytes` 时在绑定阶段直接报 `ValueError`；运行时若因路由分支休眠占满槽位，会报 `PageBundle admission capacity is exhausted` 并累加诊断计数。在张量并行（TP>1）下，小状态按 rank 复制，不省单卡容量（见第 4 章 D1）。
* **怎么看到**：`pllm.explain` 的 `state` 行会列出 `scope`、`roots`、`bytes/root` 和 `reserved bytes [exact]`；运行时通过 `llm.diagnostics().state_usage` 查看 `logical_live_bytes` 与实际分配的 `allocated_payload_bytes`，以及 `admission_capacity_failures`。

#### 2) 按声明角色自动选存储精度（A3）

* **它省了什么**：默认 `dtype="auto"` 时，pllm 检查每个状态字段怎么被使用：**只有被优化器更新的参数、以及在方法里“读出再写回”的累加器/跨块进位（carry）才用 FP32**；只写或只读的字段一律用模型精度（如 BF16）。规约、梯度累加和优化器内部数学始终保持至少 FP32。
* **为什么不能全用 BF16，也不能全用 FP32**：
  * **递推累加必须用 FP32**：在 32k tokens、chunk 大小 64 的长序列递推测试中，若跨块 carry 状态用 BF16，相对误差会累积到 $1.437 \times 10^{-1}$；用 FP32 则为 $3.692 \times 10^{-4}$（低三个数量级）。
  * **不需要累加的字段若误用 FP32 会既占显存又掉速**：在 RTX 4090（batch $8 \times n=16$，256 tokens）上，ReFT ADD 状态从 FP32 改为按角色选精度后，每请求状态从 $3.50\text{ KiB}$ 降到 $1.75\text{ KiB}$，FactorAdapter 从 $22.00\text{ KiB}$ 降到 $11.00\text{ KiB}$（Torch 常驻显存分别减少 $1.82\text{ MiB}$ 和 $4.59\text{ MiB}$）；而在稠密状态负载上，把 $9.00\text{ MiB}$/请求的状态强行提到 FP32（$17.32\text{ MiB}$/请求），吞吐会从 $3,156\text{ tok/s}$ 掉到 $2,724\text{ tok/s}$（86.31%）。
* **适用边界与触发**：自动生效；作者可在 `RequestState` / `InstanceState` 或 `pllm.Field(..., dtype=...)` 上显式覆盖。若在冻结字段上直接做仅支持 FP32 的矩阵乘，需显式声明 `dtype=torch.float32` 或在数学中转换。`explain` 的 `state` 行会打印每个字段最终解析出的 dtype。

#### 3) 从真实任务代码推导显存峰值（A4）与预算降级（A5）

* **它省了什么**：早期的手写估算容易高估工作区。pllm 现在在 CPU 上用 meta/fake 张量对真实任务体（前向、反向、优化器、scan）做生命周期与别名追踪（含 autograd 保存的中间张量、512 B 对齐和 1 MiB 分配器尾余量），由 Rust 规划器按精确的生命周期区间计算峰值，并在候选实现中优先选择 `[峰值显存最小, 任务数最少]` 的方案。
* **实测数字**：将手写工作区公式替换为真实代码追踪后，`ESamp` 的 `predict` 工作区预留从 $177.6\text{ MB}$ 降到 $27.8\text{ MB}$；`OverRIDE` 的训练工作区从 $1,056\text{ MB}$ 降到 $267\text{ MB}$（实测预热后 CUDA 峰值为 $264,242,176\text{ B} \le$ 推导上限 $267,269,120\text{ B}$）。
* **放不下时自动退回主通道（A5）**：旁路重叠（`side` lane）需要同时保存两步的缓冲。若 `ExecutionOptions.memory_budget_bytes` 不够放旁路计划，规划器会自动把任务退回主通道（`inline`）执行，同时保留逻辑读取边界；再低于内联下限才报错并指出该改哪个预算。例如 `ESamp`（$\beta=0.5$，128 并发序列）在 `explain@5cdc9785` 中报告：旁路峰值 $1,091,375,392\text{ B}$（约 $1041\text{ MiB}$）；低于该值退回内联；低于 $1,020,590,368\text{ B}$（约 $973\text{ MiB}$）规划失败（内联降级的吞吐代价未单列测量）。
* **适用边界**：不覆盖无界的自定义 autograd、第三方库临时缓冲、首次调用 cuBLAS 时的初始化开销（约 $8.5\text{ MB}$）以及 vLLM 自身的模型与 KV 显存。

#### 4) 静态 KV 布局映射为 vLLM 原生规格（A8）

* **它省了什么**：在 `@pllm.custom_attention`（[教程 10](../tutorials/10_custom_attention.ipynb)）里声明 `keep="window"`（滑动窗口）、`sink=True`（保留完整 prompt + 最近窗口）或 `keep="none", source=k`（直接复用低层第 $k$ 层的 KV cache），pllm 会将它们直接映射为 vLLM 原生的 `SlidingWindowSpec`、`SinkSpec`（`RSWASpec` 子类）或跨层共享组，由 vLLM 的块管理器在生成过程中真正回收和少分物理块。
* **实测数字**：在 [教程 10](../tutorials/10_custom_attention.ipynb)（S-tut，`max_model_len=2048`）中，单请求 KV 上限从全量注意力的 $24.0\text{ MiB}$ 降到 $16.6\text{ MiB}$（偶数层窗口 256）和 $12.0\text{ MiB}$（跨层共享一半层），最大并发 2048-token 请求数达到 717.6。在另一组固定长度测试（prompt 8 + 生成 112）中，12 个奇数层共享省下 **50.00%** KV 字节，2/24 层窗口省下 **7.29%**，sink 布局省下 **6.25%**。
* **适用边界（什么省不了）**：仅支持 MRv2、TP=1、FA2、非量化 KV 且关闭 prefix caching；不能与 `PositionState` 或 `custom_forward` 同时使用。更重要的是：**它只支持静态几何布局，不支持按注意力分数动态淘汰已写入的 KV 块**（如 SnapKV/H2O，原因见第 9 章 N1；压缩远端记忆 M7 见 N2）。在 `explain` 中可看到每层准入容量，运行时看 `diagnostics().kv_cache_blocks` 和 `kv_cache_max_concurrency`。

#### 5) 只读大表放主机内存 `placement="host"`（A9）

* **它省了什么**：像 Engram 这类外部知识库或词表级查找表（如 [教程 11](../tutorials/11_lookup_memory.ipynb) 中 $2048 \times 151936$ 的 BF16 表，光表本身就要 $594\text{ MiB}$）如果常驻显存，会严重挤压 KV cache。作者显式声明 `InstanceState(pretrained=..., placement="host")` 后，该表的设备端预留降为 `reserved_bytes=0`，变成锁页（pinned）主机内存并通过 PCIe 按需读取行。
* **实测数字**：在 [教程 11](../tutorials/11_lookup_memory.ipynb) 中，把这张 $594\text{ MiB}$ 的表从 `"device"` 改为 `"host"`，pllm 负责的设备端显存峰值从 $736,523,358\text{ B}$（约 $702\text{ MiB}$）降到 $114,193,488\text{ B}$（约 $109\text{ MiB}$）；测试中可以在仅 $1\text{ GiB}$ 的设备端预算下运行 $3\text{ GiB}$ 的主机表。
* **为什么必须由作者显式指定**：放主机内存是用显存容量换 PCIe 带宽（见第 3 章），是否划算取决于每步查多少行，因此 pllm 不做后台自动迁移。仅限只读预训练 `InstanceState` 的按行读取，不能用于可写状态或直接在模型站点内读取。`explain` 会显示 `placement: host; host bytes` 及每行逻辑传输字节数。

### 显存容量的其他机制（速览）

| 机制（ID） | 它省了什么 / 做了什么 | 代表数字或状态 | 触发与查看方式 |
| :--- | :--- | :--- | :--- |
| **显式捕获预算**（A6）与常规模型图尺寸（A7） | 每个 `StepCapture` 任务使用受 `resident_bytes` 限制的私有图内存池；无参数目标时不强行捕获到调度器最大 token 数，避免挤空 KV（N15）。预算不足直接报错，绝不静默退回 eager。 | 模型图默认池 $64\text{ MiB}$（LAuReL 实占 $46\text{ MiB}$、value residual $22\text{ MiB}$、HC $26\text{ MiB}$）；`custom_forward` 按实际推导（32 并发 $748\text{ MiB}$，128 并发 $1,392\text{ MiB}$）。 | `ExecutionOptions.capture_memory_bytes`；`explain` 的 `capture reservation`（注：`explain` 是否估算 `custom_forward` 捕获池在文档与代码记录间尚待核实，标为未验证）。 |
| **引擎关闭确定性释放**（A10） | `with make_llm(...)` 退出时按序清理 vLLM、cuBLAS 工作区、CUDA graph 条件池引用与编译缓存并 `empty_cache`。 | 修复前连续建两个 4B 捕获引擎会残留约 $14.4\text{ GiB}$；修复后两次关闭均回到 $21.42\text{ GiB}$ 空闲（起始 $21.53\text{ GiB}$）。边界：带捕获的精确前向更新目前关闭时仍不归还全部显存。 | 自动生效（`with pllm.make_llm(...)`）。 |
| **量化权重瞬态反量化**（A11） | 在 GPTQ/Marlin 等量化底座上做训练或 overlay 读取时，用 Triton kernel 瞬态反量化，不保存常驻稠密权重副本（对比第 9 章 N9）。 | 14B GPTQ TLM（A100）步时从 $1.117\text{ s}$ 降到 $0.193\text{ s}$，峰值显存不变；4090 上 14B qkv 反量化从 $2.535\text{ ms}$ 降到 $0.104\text{ ms}$（注：目前仅有脚本验证，尚无合入仓库的专项回归测试，标为未验证）。 | 量化模型上自动生效。 |
| **共享 hidden tap 缓冲**（A12）与**跨消费者结果复用**（A13） | 同一 `(layer, phase)` 的激活只建一份 tap 缓冲供所有方法共享（A12）；sampler adapter 消费的 token 辅助计算只算一次（A13）。 | A12：`explain` 显示 `tap model.layers.0[0], decode: 460800 bytes shared by loss, predict`（吞吐收益未单测）。A13：128 行 decode 下每行省 $303,872\text{ B}$（从 $607,744\text{ B}$ 减半），每步少搬 $38,895,616\text{ B}$。 | 自动生效；`explain` 的 `tap` 行。 |
| **有界诊断与观测内存**（A14） | `numeric_health` 在设备端分块（FP64）归约快照，仅在查询时拷回标量；trace 队列上限 256 事件 $\times 32\text{ KiB}$；CPU 校验省略不变大表。 | 594 MiB 表的健康检查额外显存从 $5,912,134,656\text{ B}$（$5.51\text{ GiB}$）降到 $19,923,456\text{ B}$（$19.0\text{ MiB}$）；64 桶下主机 RSS 从 $9,181,812\text{ KiB}$ 降到 $1,935,532\text{ KiB}$。 | `diagnostics().numeric_health`、`trace_snapshot()`。 |
| **KV CPU 卸载**（A15）、**感知负载的 Prefix Cache**（A16）、**同卡零拷贝状态投喂**（A17） | A15 通过 vLLM 连接器把 KV 页卸载到 CPU；A16 在 prefix cache 块哈希中混入负载命名空间，只有数学一致时才复用；A17 通过 CUDA IPC 页授权让消费者直接借用生产者的物理页（`data_ptr` 相同，无第二份池）。 | 均为容量/正确性机制，端到端吞吐收益未测。边界：A15 拒绝 LMCache；A16 与 `custom_residual`/`custom_attention` 互斥；A17 仅限同 GPU 只读状态。 | A15：`ModelConfig(kv_offload=...)`；A16：`enable_prefix_caching=True, generated_kv="reuse"/"isolate"`；A17：`InstanceState(feed=...)` + `llm.feed()`。 |

---

## 2. 显存带宽：每步少搬字节

### 为什么会成为瓶颈：用 ESamp guided 的账单算一笔账

哪怕显存完全放得下，**每步在显存里多读或写几百 MB，也会立刻变成几百微秒的步时**。最典型的例子是 [教程 03](../tutorials/03_decode_training.ipynb) 的 `ESamp` 在线蒸馏引导采样（`@pllm.guide_sampling`）。

在 S-4090 设置下（Qwen2.5-0.5B，128 条并发序列 = $8 \times 16$），只训练不引导（$\beta=0$）能跑到原生 vLLM 的 **92.48%**，而开启引导采样（$\beta=0.5$）是 **83.99%**，每步多花 **$401\ \mu\text{s}$**。把这 $401\ \mu\text{s}$ 拆开，每一项都对应着显存带宽或词表扫描：

1. **LM head 行数从 128 行翻倍到 256 行：$+118\ \mu\text{s}$**。引导采样需要为每个请求的引导分支也算一份全词表 logits（151,936 词），`explain` 会直接报出这笔开销：`projection FLOP: 34,850,471,936`。为什么工作量翻倍只多了 $118\ \mu\text{s}$，远小于单独再跑一次 128 行的 LM head（实测约 $380\ \mu\text{s}$）？因为 pllm 把引导行拼进了 vLLM 原有的那次 LM head 调用里，**$272.27\text{ MB}$ 的 LM head 权重每步依然只从显存读一次**（机制 B2）。
2. **词表宽度的重新加权（reweight）：$+187\ \mu\text{s}$**。把主干 logits 和引导预测按 $(1+\beta)\,\text{logits} - \beta\,\text{pred}$ 混合，128 行 $\times 151936$ 词表的一次读写就是 $194,478,080\text{ B}$（约 $195\text{ MB}$），$187\ \mu\text{s}$ 已经贴近 RTX 4090 的显存带宽极限（机制 B3）。
3. **两次额外的小 CUDA graph 回放：约 $+56\ \mu\text{s}$**；**过滤（filter）：$+8\ \mu\text{s}$**。

作为参照，vLLM 自己的原生采样器在每步 decode 里也要把词表扫好几遍：temperature 缩放约 $120\ \mu\text{s}$、top-k/top-p 过滤约 $360\ \mu\text{s}$、Gumbel 采样约 $95\ \mu\text{s}$。在 0.5B 这样的小模型上，主干只有 24 层小权重，多一次词表投影和加权就会占到步时的约 10%；而到了 7B 模型上，主干每步要读十几 GB 权重，同样这点词表开销占比极小（7B 模型在 2048-token 捕获设置下，$\beta=0.5$ 约为原生 vLLM 的 94%，见 A7 的出处）。

### 核心机制

#### 1) 引导行与主干共用一次 LM head GEMM（B2）+ 融合重加权（B3）

* **它省了什么**：
  * **B2（共用 GEMM）**：`@pllm.guide_sampling` 在过滤后产生的引导隐状态行，会被追加到 vLLM 原本的 `compute_logits` 调用末尾，把矩阵乘从 $[128, 896] \times [896, 151936]$ 扩成 $[256, 896] \times [896, 151936]$。这样每步省下了再读一遍 $272.27\text{ MB}$ 权重的带宽；当 $\beta=0$ 或没有受影响的引导行时，不追加任何行。
  * **B3（融合重加权）**：对存活有限值做 $(1+\beta)\,\text{logits} - \beta\,\text{pred}$ 时，pllm 用单个 `torch.compile` 融合 kernel 完成，避免在 $[128, 151936]$ 大张量上做 7 次逐元素往返读写（逻辑流量固定为 $194,478,080\text{ B}$/步）。
* **适用边界**：共用 GEMM 会改变 cuBLAS 选用的 tile 大小，带来的浮点舍入差异不超过 vLLM 自身随 batch 大小变化的幅度（Qwen 0.5B 上 $\le 0.0625$）。注意：用来读取中间层词表分布的 `layer_logits`（如 [教程 06](../tutorials/06_reading_inside.ipynb) 的 DoLa）默认仍使用独立 GEMM，因此每步多读一次或多次 LM head 权重，在 0.5B 上吞吐为原生 vLLM 的 25%。更激进的跨步算术融合因产生 $1.9 \times 10^{-6}$ 的数值偏差而被拒绝（见第 9 章 N14）。
* **怎么看到**：`explain` 中会列出 `projection FLOP`（如 `34,850,471,936`）和 `sampler 194478080 bytes`。

#### 2) 查表与周边逐元素运算的 Inductor 整图融合（B4）

* **它省了什么**：针对只依赖 `input_tokens` 的查表任务（如从记忆表 gather 出向量后紧接着做 `cast`、`where` 掩码和缩放），pllm 在绑定引擎时用 Inductor `fullgraph` 把它们编译并预热成**单次表读取 kernel**，省去把查出来的宽向量写回显存再读出来做逐元素运算的往返流量。
* **实测数字**：在 [教程 11](../tutorials/11_lookup_memory.ipynb) 设置下（结合提前到 `request.prepare` 旁路发射，见第 5 章 E7），设备端表吞吐从 $26,133\text{ tok/s}$ 提升到 $32,881\text{ tok/s}$（原生 vLLM 的 66% $\rightarrow$ 84%），主机端表（C2 之前）从 $12,557\text{ tok/s}$ 提升到 $23,044\text{ tok/s}$（32% $\rightarrow$ 59%）；16-token 剖析中 D2D 拷贝调用从 36 次减半到 18 次，CUDA 耗时从 $84.671\text{ ms}$ 降到 $71.754\text{ ms}$。
* **适用边界**：仅对 CUDA 上不可微的 `input_tokens` 任务自动生效；若由于浮点结合律顺序变化需要严格位级一致，可通过 `allow_numerical_transforms=False` 关闭（`explain` 会在提示中点名该开关）。编译失败会在启动时立刻报错。

#### 3) 自适应深度压实（B6）：为什么 decode 跳层只在整批全跳时才省读权重

* **它省了什么与成本模型**：在 `@pllm.custom_forward`（[教程 12](../tutorials/12_adaptive_depth.ipynb)）里用 `layers(..., active=mask)` 让部分 token 跳过中间若干层时，pllm 会把活跃行稳定压实（compact）到 2 的幂次大小桶里执行，被跳过的行只在第一次跳过时用原生算子补齐 norm/QKV/RoPE 以维持 KV 连续（约占单层开销的 7%）。
  但要注意 **decode 阶段是权重受限的**：哪怕一个层里 128 个 token 只有 1 个活跃，GPU 也必须把这一层几百 MB 的权重（7B 模型单层约 $466\text{ MB}$）完整读进 SM 才能算这 1 行。因此，**只有当活跃行数压实进 bucket 0（即整批 token 在该层全部跳过）时，才真正省下读该层权重的带宽**；部分跳过省下的是与不活跃行数成正比的注意力 KV 读取流量。
* **实测数字**：在 [教程 12](../tutorials/12_adaptive_depth.ipynb)（S-tut，$n=4$，第 16–19 层可跳过）中：从不跳过为原生图吞吐的 **93%**；阈值 0.85（约 14% token 跳过）为 **90%**；全部跳过（落入 bucket 0，真正不读这 4 层权重）达到 **102%**；作为对比，原生 vLLM 关掉 CUDA graph（eager）只有 **45%**。
* **适用边界**：仅对 decode 阶段 token 生效；不支持 expert-choice 式路由；不释放被跳层的 KV 显存。`explain` 会显示 `timing forward: active=CUDA conditional nodes ... 0 blocking host syncs per step`。

#### 4) 布局注意力直接复用 vLLM 原生启动器（B8）与稀疏读取 Top-K（B7）

* **原生启动器（B8）**：第 1 章提到的窗口注意力（`window`）、跨层共享（`shared`）和保留首尾的 `sink`（R-SWA），在读取 KV 时直接调用 vLLM 原生的 FlashAttention-2（支持 `window_size` 与 `return_softmax_lse`）和上游 Triton `unified_attention`，输出与原生启动器位级一致，避免了早期用 Python/PyTorch 拼装掩码和拷贝块的高昂带宽开销。在 S-4090 上，把 `sink` 从旧路径换到原生 R-SWA 后，吞吐从 $8,964\text{ tok/s}$ 升至 $36,458\text{ tok/s}$（每轮拷贝耗时从 $561.4\text{ ms}$ 降到 $2.0\text{ ms}$）。在 [教程 10](../tutorials/10_custom_attention.ipynb) 中，`NativeRead` 达原生吞吐的 **97%**、`LocalGlobal`（sink）达 **97%**、`CrossLayerShared` 达 **95%**。
  * *边界*：FlashAttention-2 的滑动窗口在窗口极小（如窗口大小为 3）时反而比全量注意力慢（见第 9 章 N7）。
* **稀疏 Top-K 读取 `PagedKV.attend(q, topk=...)`（B7）**：按块打分只挑 Top-K 的 KV 块来读——**只省读取带宽，不省 KV 显存容量**。`explain` 能算出每 token 省下的最大 KV 读取字节（例如 `QuestRead <= 458,752` B/token，对比全量 `<= 1,048,576` B/token）。
  * *当前限制（如实说明）*：由于当前打分与 gather 读取路径走的是 PyTorch 算子，在 [教程 10](../tutorials/10_custom_attention.ipynb) 的设置下 `QuestRead` 吞吐只有 $2,727\text{ tok/s}$，即原生 vLLM 的 **7%**（在另一组早期的 `cattn` 基准设置下为 75%）。vLLM 原生的高效 DSA Top-K 路径要求 SM90+（Hopper 架构），在 RTX 4090（SM89）和 A100（SM80）上不可用（见第 9 章 N3）。

### 显存带宽的其他机制（速览）

| 机制（ID） | 它省了什么 | 代表数字（设置与来源） | 触发与查看方式 |
| :--- | :--- | :--- | :--- |
| **捕获训练内的分页 AdamW**（B1） | `paged_adamw` 直接从源物理槽位读参数与一阶/二阶矩、直接写回目标槽位，省掉对 5 个优化器字段做写时复制（COW）的整页搬运。 | S-4090 $\beta=0$ 下，177.6 $\mu$s 的旁路 GPU 训练时间里 AdamW 仅占 **29.3 $\mu$s**（单独隔离收益未测）。 | FP32 AdamW 状态在 `StepCapture` 下自动启用；`explain` 显示 `optimizer ... bytes (included)`。 |
| **只在受影响行上工作**（B5） | `add_logits(rows=...)` 只在命中的采样行上加偏置；非残差消费 `input_tokens` 的方法只看采样行。 | 端到端收益未单列测量。 | 声明 `rows=` 掩码或自动按行空间裁剪。 |
| **削减冗余梯度与拷贝流量**（B9） | `FieldView` 跳过全宽梯度分配、增大页拷贝 tile、单次参数切分避免重复 padding、目标已知时省略掩码保留 kernel、仅消费 token 的方法跳过梯度快照。 | A100 微基准：模块前向/损失/梯度 $0.544 \rightarrow 0.207\text{ ms}$；字段查表 $187.45 \rightarrow 134.25\ \mu\text{s}$；掩码 gather $109.83 \rightarrow 50.97\ \mu\text{s}$；页拷贝 $24.67 \rightarrow 4.79\ \mu\text{s}$。 | 自动生效。 |

---

## 3. PCIe：主机和 GPU 之间

### 为什么会成为瓶颈

当只读大表因为显存装不下而放到主机内存（`placement="host"`，第 1 章 A9）时，数据必须跨过 PCIe 总线。PCIe 4.0 x16 的实测有效带宽约 $22.4\text{ GB/s}$，只有 RTX 4090 显存带宽（约 $1000\text{ GB/s}$）的约四十五分之一。

以 [教程 11](../tutorials/11_lookup_memory.ipynb) 的宽表为例：每查一行要读 $303,872\text{ B}$，128 个并发请求每步一共要经 PCIe 搬运 $38,895,616\text{ B}$（约 **$38.9\text{ MB}$**）。光在 PCIe 上传这 $38.9\text{ MB}$ 就要约 **$1.7\text{ ms}$**——而 0.5B 模型整步前向也才 $3.30\text{ ms}$。如果这 $1.7\text{ ms}$ 的 PCIe 传输不能和模型前向重叠，它就会直接加到每步上：在读表 kernel 做到下面的重叠之前，同一个教程里表放 host 只有原生 vLLM 的 59%。

### 为什么不能用 DMA 拷贝引擎，也不能直接放任普通 gather 重叠

想把主机查表和模型前向重叠，很容易想到两条路，但实测都走不通（第 9 章 N8）：

1. **为什么不能用独立拷贝引擎（Copy Engine DMA）**：要查哪一行（token ID 索引）是在 GPU 上算出来的。`cudaMemcpy2DAsync` 或块交换 API 都需要 CPU 知道源地址偏移，而在 CUDA graph 捕获下拷贝参数更是固定的；如果先把索引拷回 CPU 再发 DMA，就会违反“每步零主机同步”的铁律（第 5 章 E1）。因此必须由 GPU SM 直接通过 **UVA**（Unified Virtual Addressing，统一虚拟寻址，让 GPU 线程直接用指针读取锁页主机内存）按设备端索引去拉取每一行（C1）。
2. **为什么默认的 UVA gather 哪怕提前到旁路 stream 也没法重叠**：不加限制的普通 gather kernel（按数据量开网格）对这张大表做索引时，会启动 **37,984 个线程块（CTA）**，把 RTX 4090 的 128 个 **SM**（Streaming Multiprocessor，GPU 上的流多处理器核心）占到 100% 满载。虽然这些线程块绝大部分时间只是在傻等慢速的 PCIe 返回数据，但它们占住了 SM 槽位，导致主 stream 上的模型前向根本排不上队！实测显示，不限制规模的提前发射非但没快，反而让每步耗时从 $5.697\text{ ms}$ 恶化到 $5.939\text{ ms}$，$1,753.7\ \mu\text{s}$ 的读表只有 **$26.2\ \mu\text{s}$** 与主干前向重叠（N8）。

### 核心机制

#### 1) UVA 按行读取（C1）+ 限制 SM 占用的常驻读表 Kernel（C2）

* **它省了什么**：实际上，跑满 ~22 GB/s 的 PCIe 带宽只需要很少的并发请求，根本用不着全部 SM。pllm 通过 `HostGather` 拦截对声明为 `placement="host"` 表的 `embedding` / `__getitem__` / `index_select` 操作，改调自定义的常驻（persistent）Triton 算子 `pllm::host_rows`，**把线程块数量硬限制在 GPU SM 总数的四分之一**（在 128 个 SM 的 RTX 4090 上只开 **32 个线程块** $\times 128$ 线程），并配合 `request.prepare` 阶段提前在旁路 stream 启动（E7）。
  这样一来，32 个线程块在后台慢慢等 PCIe 数据，**剩下四分之三的 SM 完全空出来跑主 stream 的模型前向**，把原本暴露在外的 $1.7\text{ ms}$ PCIe 传输藏进了模型前向的阴影里（在剖析中达到 **99.14%** 的主干重叠率）。
* **实测数字**：[教程 11](../tutorials/11_lookup_memory.ipynb) 在 `main`（`ea7ff796`，RTX 4090、vLLM 0.27.1）上重跑的结果：
  * 原生 vLLM：**38,821 token/s**（每步约 **$3.30\text{ ms}$**）
  * 表在 `device`：**31,750 token/s**（原生的 **82%**，每步约 **$4.03\text{ ms}$**）
  * 表在 `host`：**30,174 token/s**（原生的 **78%**，每步约 **$4.24\text{ ms}$**）
  可以看到，尽管每步都要经 PCIe 实打实搬运约 $38.9\text{ MB}$（物理传输耗时约 $1.7\text{ ms}$），**表放 host 比放 device 每步只慢了约 $0.21\text{ ms}$**（在之前的 5 轮中位数剖析对比中，未限制 CTA 时为 $5.697\text{ ms}$/步、$22,469\text{ tok/s}$，限制为四分之一 SM 后降到 $4.194\text{ ms}$/步、$30,522\text{ tok/s}$，吞吐提升 35.8%）。
* **适用边界与估算方法**：
  * 仅限对 `placement="host"` 表的视图和按行读取（由 `GatherTrace` 在绑定时静态检查）；而且该查表任务只能依赖 `input_tokens` 和不可变状态，才能提前到前向开始前发射（若依赖模型中间层的 hidden，就不能提前到 `request.prepare` 执行）。
  * **作者怎么自己估算能不能藏住**：看 `pllm.explain` 输出的 `logical transfer bytes/row`（或 `decode bytes/step`），用
    $$\text{PCIe 传输时间} \approx \frac{\text{活跃行数} \times \text{bytes/row}}{22\text{ GB/s}}$$
    跟模型单步前向时间比。传输时间明显短于一次前向时，大部分搬运能被藏住（教程 11 里 $1.7\text{ ms}$ 的搬运只让每步多了约 $0.21\text{ ms}$）；传输时间超过前向时间时就藏不住了。

#### 2) 单次锁页表上传（C3）与有界异步导出（C4）

* **单次上传（C3）**：每步 decode 都需要把当前活跃请求的槽位 ID、掩码、采样器地址与步幅传给 GPU。pllm 在 CPU 上用 NumPy 把所有这些整数元数据填进**同一张锁页（pinned）`int64` 表**，每个捕获任务只做一次 H2D 拷贝。在 S-4090（`ESamp` $\beta=0$）剖析中，pllm 每步 H2D 仅 $23,692\text{ B}$（耗时 **$6.24\ \mu\text{s}$**，对比原生 vLLM 的 $1,088\text{ B}$ / $2.50\ \mu\text{s}$），D2H 保持 $1,542\text{ B}$/步不变。
* **有界异步导出 `@pllm.export`（C4）**：当作者需要把设备端张量导回主机做观测或日志时，`@pllm.export(max_rows, max_cols, every, overflow="drop")` 使用 `non_blocking=True` 拷入有界锁页缓冲并记录 CUDA 事件，后台异步回调消费，**默认在缓冲区满时丢弃（`"drop"`）而不是阻塞 GPU/CPU**（可在诊断中查看 `export_dropped`；TP 侧计数器快照异步 D2H 仅约 $2.2\ \mu\text{s}$/步，零主机等待）。

---

## 4. 卡间通信（TP）

### 为什么会成为瓶颈

在多卡张量并行（Tensor Parallelism，TP）下，vLLM 每算完一层注意力或 MLP 都要做一次全卡集合通信（Custom All-Reduce）。如果负载自己也去发集合通信，或者旁路 stream 上的训练 kernel 刚好在 vLLM 做 All-Reduce 的同一刻抢占 GPU 带宽和 SM，就会拖慢所有卡同步的步伐：在 4 张 A100-SXM4-80GB 上跑 Qwen2.5-7B + `ESamp`（$\beta=0$），当旁路训练与 vLLM 的 Custom All-Reduce 发生重叠（每步重叠 $193\text{–}205\ \mu\text{s}$）时，吞吐从单卡的 93.79% 掉到了 TP4 的 **87.07%**。

### 核心机制：小状态按 rank 复制 + 错开集合通信（D1）

* **它省了什么**：
  1. **pllm 自己绝不发起集合通信**：hidden tap、路由计算、`PositionState` 以及请求级小状态在每个 TP rank 上各存一份完全相同的副本；`@pllm.export` 回调只在 rank 0 触发。各 rank 独立执行相同的确定性逻辑（pllm 与原生 vLLM 在每个 TP 规模下的 logits 均方根误差为 0）。
  2. **流优先级与训练时机对齐**：把 TP 推理主 stream 设为高优先级（`-1`）、旁路 stream 设为 `0`，并让各 rank 的自动状态更新在采样前写入私有状态、在 token 提交时发布，从而消除旁路训练与 vLLM Custom All-Reduce 的重叠争抢。
* **实测数字**：在 A100-SXM4-80GB（Qwen2.5-7B，`ESamp` $\beta=0$，rank 4，batch $8 \times n=16 \times 256$ tokens，开启 CUDA graphs）上，消除训练与 All-Reduce 重叠后（重叠耗时从 $205/193\ \mu\text{s}$ 降为 **$0\ \mu\text{s}$**/步）：
  * TP1：原生 vLLM 的 $93.79\% \rightarrow \mathbf{95.84\%}$
  * TP2：$89.09\% \rightarrow \mathbf{95.63\%}$
  * TP4：$87.07\% \rightarrow \mathbf{97.60\%}$（仅设置流优先级一项在 TP4 上带来 $14,936 \rightarrow 15,031\text{ tok/s}$）。
* **适用边界**：
  * 因为状态是按 rank 复制而非分片（shard），**TP 不会帮你省 pllm 状态或预留缓冲的显存容量**——`explain` 会打印 `parallel ... reservations are per rank` 提示。
  * 仅支持 vLLM 0.32 MRv2。以下能力涉及跨分片语义或每层改写，在 `tensor_parallel_size > 1` 时会被直接拒绝：`custom_residual`、`custom_attention`、LoRA 式权重改写、全模型训练、状态投喂（`feed`）以及 `check_on_gpu`（后者需在 TP=1 下运行）。

---

## 5. GPU 时间线：消灭气泡

### 为什么会成为瓶颈

即便显存和带宽都很充裕，如果 CPU 每步都要停下来等 GPU 算出一个标量（host sync），或者把几十个小算子一个个从 Python 端发射出去，GPU 时间线上就会布满空隙（气泡）。最直观的对比是：**哪怕什么自定义负载都不加，光把原生 vLLM 的 CUDA graph 关掉（退回 eager 逐算子发射），在 Qwen2.5-0.5B 上吞吐就会直接掉到图模式的 45%**（[教程 07](../tutorials/07_custom_residual.ipynb)、[教程 12](../tutorials/12_adaptive_depth.ipynb)）。

**术语**：**CUDA graph** 是把一串 GPU kernel 的启动参数和依赖关系预先录制（capture）成一张图，之后每步只用一次极轻量的图回放（replay）发射整串计算，消除了 Python 逐个启动算子的开销，但要求图内张量地址固定、且绝不能有把 GPU 数值读回 CPU 做 `if` 判断的操作。

### 核心机制

#### 1) 零每步主机同步，且作为可检验契约强制执行（E1）

* **它省了什么**：在自定义代码里不小心写下一句 `.item()`、`if tensor.any():` 或 `.cpu()`，就会迫使 CPU 停下来等 GPU 排空整条流水线。pllm 把“稳态 decode 每步 0 次主机同步”定为硬契约：
  * 静态阶段：`pllm.explain` 在形状追踪时统计同步次数（显示 `host 0 syncs in traced workload math`）。
  * GPU 校验阶段：[`pllm.testing.check_on_gpu`](validation.md) 在 `torch.cuda.set_sync_debug_mode("error")` 下挂载所有执行入口并实跑至少 8 个稳态 decode 步，任何隐式同步都会直接抛错。
* **实测数字**：在 S-4090 的重叠与主机路径剖析中，128 次事件等待全部来自 vLLM 自身的异步输出获取（`async_utils.get_output`），**pllm 产生的 Event/Stream 主机等待为 0，decode 窗口内 `cudaDeviceSynchronize` 空闲时间为 0**。

#### 2) 负载任务与在线训练整包捕获进 `StepCapture`（E2）

* **它省了什么**：在线训练（如 `ESamp` 的 `loss` 方法）包含按请求收集参数、前向算 loss、掩码归约、`backward()` 求导、写状态字段和 `paged_adamw` 更新，一共几十个小 kernel。`StepCapture` 把这一切按签名的最大行数/状态容量**录进同一张 CUDA graph**，输入用固定 tap 缓冲，未满 batch 的空行指向不会污染真实状态的哑槽位（sink state），因此即使每步活跃请求集合在变，也**不需要重新录制图**。
* **实测数字**：在 128 并发请求的 RTX 4090 测试中，启用 `StepCapture` 后每步耗时从 $6.73\text{ ms}$ 降到 $5.25\text{ ms}$，`commit_token` 阶段从 $3.32\text{ ms}$ 降到 $1.56\text{ ms}$，吞吐从 $19,100\text{ tok/s}$ 升至 $23,829\text{ tok/s}$（**+25%**；对照数字取自当时的规格文档，不是同机对照实测）。在 S-4090 的 765 步剖析中，每步 decode 恰好发射 1 次训练图（共 1,530 次图发射，**0 次重捕获**）。
* **适用边界**：要求按行独立的数学与逐行 loss；动态训练选择器会被拒绝并退回 eager。`explain` 的 `member` 行会标出 `captured`。

#### 3) `@pllm.custom_residual` 捕进 vLLM 模型图（E3）

* **它省了什么**：残差流改写（如 [教程 07](../tutorials/07_custom_residual.ipynb) 的 Value Residual、Hyper-Connections、记忆子层）发生在模型每一层内部。如果每层进出 Python 钩子，整张模型图就会碎掉。pllm 把残差钩子降级为具名自定义算子 `pllm::model_site`，用设备端索引读写 PTTS 私有页，使 **vLLM 在录制它自己的模型 CUDA graph 时能把残差改写一并录进去**。
* **实测数字**：在 RTX 4090（$8 \times n=4 \times 256$ tokens）上，把 `custom_residual` 从 eager 捕进模型图后（相对于原生图 vLLM 的吞吐占比）：
  * 空改写（identity）：$25.3\% \rightarrow \mathbf{65.8\%}$
  * Value Residual：$34.4\% \rightarrow \mathbf{85.2\%}$（[教程 07](../tutorials/07_custom_residual.ipynb) 现为 **85%**）
  * Hyper-Connections（$n=4$）：$19.2\% \rightarrow \mathbf{66.6\%}$（教程 07 现为 **65%**）
  * 状态化记忆子层（当时未进图为 $11.6\%$，进图后在教程 07、1 GiB 捕获预算下为 **41%**；原生 eager 为 45%）。
* **适用边界**：要求 TP=1 且关闭 prefix caching；vLLM 捕获的是源算子而非 Inductor 融合算子。

#### 4) 提前到 `request.prepare` 的旁路发射（E7）

* **它省了什么**：如果一个任务只依赖 `input_tokens` 和不可变状态（例如 [教程 11](../tutorials/11_lookup_memory.ipynb) 的记忆查表），Rust 规划器会自动把它的锚点从模型层间提前到整步最开头的 `request.prepare`，放到旁路 stream 上先跑，并在真正消费该结果的模型层入口放一个 CUDA 事件等待。这样查表就能跟前半段模型层并行。
* **关键配合边界**：如第 3 章所述，**提前发射必须与受限网格（C2）或融合（B4）配合**；如果提前发射一个占满 100% SM 的大 kernel，反而会饿死模型前向（N8）。在 `explain` 的 `member` 行可以看到 `request.prepare / side / captured`。

### GPU 时间线的其他机制（速览）

| 机制（ID） | 它做了什么 | 代表数字 / 边界 | 触发与查看方式 |
| :--- | :--- | :--- | :--- |
| **设备端条件节点 CUDA IF nodes**（E4） | 自适应深度分桶（B6）、scan 完成分桶和精确 decode 分支全部用 CUDA 图内的设备端条件节点跳转，无需把判断条件拷回 CPU。 | 稳态每步 0 次阻塞式主机同步。需 CUDA 12.4+ 驱动路径。 | `active=` 或计数 scan 下自动启用；见 `explain` `timing` 行。 |
| **前向内精确 decode 更新捕获**（E5） | `train_in_forward(visibility="exact")` 在默认数值容差模式下将 decode 阶段捕获进图。 | 4090 上 In-Place TTT 4B（4096 prompt + 256 生成）：bs1 从 $57.59 \rightarrow 59.51\text{ tok/s}$（+3.3%），bs4 从 $107.56 \rightarrow 113.16\text{ tok/s}$（+5.2%）。边界：prefill 仍为 eager；严格位级一致模式比 eager 更慢（$48.84 / 78.66\text{ tok/s}$）；关闭引擎时暂不归还全部显存。 | 默认容差模式自动启用；见第 9 章 N12。 |
| **`cuda_graphs="auto"` 准入与具名原因**（E6） | 默认只要支持就进图，不支持时退回 eager 并给出明确原因；设为 `True` 则在遇到不支持的特性时直接报错。 | 会导致模型退回 eager 的原因：`cuda_graphs=False`、MRv1、`ActivationPatch`、`custom_forward`（改用 pllm 自己的分段 wave 图）、共享可变 `InstanceState` 的 Adapter。 | `diagnostics().eager_reason`、`explain` 的 `forward` 行。 |
| **签名自定义算子进图**（E8） | 作者用 `torch.library.custom_op` 注册并补齐 fake/autograd/vmap 规则的自定义 Triton/PyTorch 算子（[教程 09](../tutorials/09_custom_kernels.ipynb)）可原封不动捕进 CUDA graph 与旁路通道。 | [教程 09](../tutorials/09_custom_kernels.ipynb)（S-tut）：自定义 Triton 算子 $37,518\text{ tok/s}$（原生 $39,623$ 的 **95%**），PyTorch 算子 $37,976\text{ tok/s}$（**96%**）。 | `explain` 的 `op` 行会报告每行 FLOP 与逻辑字节下界。 |

---

## 6. 没填满的 SM：重叠什么时候有用

### 为什么要想清楚“重叠”的本质

把任务放到旁路 CUDA stream（`lane: side`）并不等于白捡一套 GPU。一张卡上的 SM 和显存带宽是固定的：**只有当主 stream 没有把那一项资源吃满时，旁路任务塞进来才叫“重叠收益”；如果主 stream 已经把 SM 或显存带宽吃满了，硬把旁路计算挤进来只是在抢同一块蛋糕（零和博弈）**。

两项实测把这条边界划得非常清楚：

* **正例（等待型工作能藏住）**：第 3 章的主机查表（C2）。旁路 kernel 只占四分之一的 SM，且绝大部分时间在等慢速的 PCIe 总线，不抢显存带宽也不占满 SM。于是 $1,634.6\ \mu\text{s}$ 的查表有 **99.14%** 被藏进了主干前向里，吞吐提高了 35.8%。
* **反例（GPU 满载时的计算/带宽型重叠是零和，N6）**：在 `ESamp`（S-4090，$\beta=0$）中，剖析发现旁路训练还有 $70.33\ \mu\text{s}$ 露在主干外面。尝试去掉两个采样器等待事件后，露在外面的旁路时间确实降到了 $8.57\ \mu\text{s}$/步，但端到端吞吐反而从 **94.49% 掉到了 92.73%**——因为训练没有消失，它只是从步间移到了**正在满负荷扫词表的采样阶段**，跟采样器抢显存带宽，把采样拖慢了同样长的时间。这也印证了 `esamp94` 剖析的核心结论：**$\beta=0$ 时 ESamp 比原生 vLLM 慢的那 ~7.5%，主要是 GPU 上实打实的训练算子与主干流的资源争抢，而不是主机气泡**。

### 核心机制

#### 1) 基于 `publish_lag=1` 的采样后旁路训练重叠（F1）

* **它省了什么**：如果要求第 $n$ 步采完样立刻算完梯度更新、第 $n+1$ 步前向必须读到最新权重（即 `publish_lag=0` 严格串行），那每步前向都得干等上一步的反向传播和 AdamW 跑完。
  pllm 默认设置 `publish_lag=1`：第 $n$ 步 token 确定后，经过一个极短的拷贝栅栏就把训练放到旁路 stream 上跑，而第 $n+1$ 步前向直接读取第 $n-1$ 步发布的旧版本，新版本在第 $n+2$ 步才变得可读。这样旁路训练就能填进主干流在步间、小算子和非满载区段的空隙里。
* **实测数字**：在 128 并发请求的 RTX 4090 测试中，把 `publish_lag` 从串行改为重叠后，`ESamp` 在 $\beta=0$ 下从原生吞吐的 $91.4\%$ 提升到 **$94.3\%$**，在 $\beta=0.1$ 下从 $73.7\%$ 提升到 **$84.1\%$**（1,245 个旁路 kernel 中有 780 个与主干重叠；且在 $\beta=0$ 下 `lag=0` 与 `lag=1` 的输出位级完全一致）。
* **适用边界**：会按公式增加 PTTS 所需的状态版本槽位数（见第 1 章 A1）；显存预算不够时会自动退回主通道内联（A5）；对 `custom_forward` 或显式要求即时可见（`visibility="exact"`）的任务自动禁用。在 `explain` 中会看到 `timing publish_lag=1 ... readable at n + 2` 以及成员的 `lane: side`。

#### 2) 线性递推的分块并行前缀扫描 `pllm.scan`（F2）

* **它省了什么**：当在 chunk 内做形如 $S_t = A_t S_{t-1} + B_t$ 的线性递推时（如线性注意力或测试时训练的快权重更新），一步步串行循环既慢又填不满 SM。Rust 规划器会自动识别仿射递推结构，在显存工作区预算允许时改用有序 / Hillis-Steele / 树形并行前缀扫描（parallel prefix scan），并支持通过 `tuning="measured"` 在绑定时实测挑选最快的候选实现。
* **实测数字**：在早期分支（2026-10-23）的微基准测试中，单头 $64 \times 64$ 状态扫描耗时从串行的 $1.4387\text{ ms}$ 降到 **$0.2190\text{ ms}$**（**6.57 倍**加速），14 头 $64 \times 64$ 为 **1.85 倍**，完整状态转移从 $5.8786\text{ ms}$ 降到 **$1.7585\text{ ms}$**（**3.34 倍**）。当状态维度很大（如 $d=896$）导致树形扫描得不偿失时，形状守卫会自动保留串行循环（注：该组数字尚未在最新 `HEAD` 上复测）。
* **适用边界**：系数需为 FP32；CPU、部分扫描或需要求导的调用保持串行；可通过 `allow_numerical_transforms=False` 禁用并行重排。

### 作者怎么判断自己的旁路工作能不能被藏住

打开 `pllm.explain` 看三行：
1. 看 `member` 行是不是 `side / captured`；
2. 看它的物理性质：如果是 `placement: host` 的查表，它是**等待型（等 PCIe）**，只要满足 C2 的条件，绝大部分时间都能被主干藏住；
3. 如果是设备端张量运算，看 `explain` 给出的每步 FLOP 和逻辑读写字节数——把字节数除以显卡带宽（4090 上约 $1\text{ TB/s}$）、FLOP 除以算力，得到它在 GPU 上大致要占多久；GPU 已经接近满载时（例如 0.5B 模型的 decode），这类计算型或带宽型的旁路工作基本会按这个耗时计入步时（第 6 章反例）。

---

## 7. 主机 CPU

### 为什么会成为瓶颈

GPU 上跑一个 0.5B 模型的 decode 步只要 $3.30\text{ ms}$。在异步调度下，CPU 一边准备第 $n+1$ 步的元数据，GPU 一边跑第 $n$ 步的图回放。只要 **CPU 每步的调度 + pllm 簿记总耗时 $\le$ GPU 单步耗时**，CPU 开销就能完全藏在 GPU 后面；一旦 CPU 耗时超过 GPU（比如模型很小、卡很快，或者 Python 里做了按请求循环的集合排序），GPU 跑完一步后就会停下来等 CPU 喂下一步，出现步间空闲。

### 核心机制与实测现状

#### 1) 已落地的每步 Python/NumPy 簿记削减（G1）

* **它省了什么**：pllm 把设备端表视图在绑定时固定下来，每步用 NumPy 批量填充索引表；用位掩码（slot mask）替代了原本每次完成栅栏都要做的 Python 槽位 ID 集合排序；推迟（lazy）trace 事件创建；跨读取流直接移交版本引用计数（pin）；并用数组支撑 `StateRefs`。
* **实测数字**：仅“去掉槽位 ID 集合排序”一项，就在 256 步里省下了 **$43\text{ ms}$**（约 $168\ \mu\text{s}$/步）的 CPU 时间。综合簿记优化后（S-4090，`gpu_memory_utilization=0.5`，3 轮中位数），`ESamp` 在 vLLM 0.27 上 $\beta=0$ 从 $91.73\%$ 升至 **$92.78\%$**、$\beta=0.5$ 从 $76.50\%$ 升至 **$83.74\%$**；在 vLLM 0.32 上 $\beta=0$ 从 $89.44\%$ 升至 **$92.37\%$**、$\beta=0.5$ 从 $69.23\%$ 升至 **$81.61\%$**。

#### 2) 当前主机路径耗时拆解与异步重叠（G2）

* **实测数字**：在 S-4090（`ESamp` $\beta=0$）上，pllm 每步的主机 CPU 簿记总计约 **$700\ \mu\text{s}$**（各阶段中位数：`encode_mrv2_rows` $122\ \mu\text{s}$、`prepare_input` $174\ \mu\text{s}$、`HiddenTaps.prepare` $110\ \mu\text{s}$、`StateExecution.begin` $79\ \mu\text{s}$、`prepare_transition` $181\ \mu\text{s}$、`publish` $82\ \mu\text{s}$、`release` $58\ \mu\text{s}$，合入 `commit_token` 均值约 $720\ \mu\text{s}$/步；外加 Python GC 约 $58\text{–}66\ \mu\text{s}$/步）。
* **为什么现在没有拖慢 4090**：因为在异步调度下，这约 $700\ \mu\text{s}$ 的 CPU 工作**几乎全部与 GPU 上的模型前向（约 $3.3\text{ ms}$）重叠执行**；实测 **pllm 造成的步间 GPU 空闲只有约 $4\text{–}6\ \mu\text{s}$/步**。
* **为什么仍计划把运行时簿记迁往 Rust（计划中）**：当未来在更快的 GPU 上跑小模型、或者后端调度器本身的 CPU 开销变重时，$700\ \mu\text{s}$ 的余量就会变紧。将运行时批量状态簿记迁入 Rust 目前处于**计划中**。

#### 3) 纯 CPU 规划与 `explain`（G3）

* **它省了什么**：Rust 规划、签名与 [`pllm.explain`](explain.md) 只读模型的 `config.json` 并用 meta 张量推导，**不初始化 CUDA、不调 `mem_get_info`、不打开权重文件**（由测试套件强制拦截验证）。以 [教程 03](../tutorials/03_decode_training.ipynb) 的 `MiniESamp` 为例，进程内首次调用 `pllm.explain` 耗时 **$2.9\text{ s}$**，第二次调用**不到 $0.1\text{ s}$**。

---

## 8. 让上面这些可预测：explain、预算、契约、诊断（H1–H5）

性能机制如果对作者是个黑盒，写代码时就只能靠猜。pllm 把前面 7 级资源全部暴露在静态解释（`H1`）、显式预算（`H2`）、自动契约校验（`H3`）、运行时无同步诊断（`H4`）和跨效果组合规则（`H5`）里。关于如何逐行阅读输出，请参阅 [《写代码时就看到成本》](explain.md) 和 [《验证与发布检查》](validation.md)。

你可以直接按下面这张表，把遇到的性能或显存问题对应到工具输出的具体字段：

| 资源阶梯 | 静态看 [`pllm.explain(workload, config)`](explain.md)（H1） | 设预算提前拦截（H2） | 跑 [`check` / `check_on_gpu`](validation.md) 验契约（H3） | 运行时看 `llm.diagnostics()` / `trace`（H4） |
| :--- | :--- | :--- | :--- | :--- |
| **1. 显存容量** | `state` 行（`roots`、`bytes/root`、`reserved`、字段 `dtype`）；`tap` 行；`memory peak ... funded by pllm` 及 `inline`/`fail` 阈值；每层 KV 准入容量 | `state_memory_bytes`、`memory_budget_bytes`、`capture_memory_bytes`、`max_training_rows`（超预算见 `report.budget_errors`） | `check_on_gpu` 的 `explain/probe` 核对推导峰值与真实加载探针一致 | `state_usage`（逻辑 vs 物理分配）、`admission_capacity_failures`、`kv_cache_blocks`、`kv_cache_max_concurrency`、`numeric_health` |
| **2. 显存带宽** | 任务行的逻辑读写字节（`optimizer ... bytes`、`sampler N bytes`、每层 `KV bytes/token`）、`projection FLOP`、自定义 `op` 字节下界 | `max_training_rows` 限制单步训练行数 | — | `trace_snapshot()` 中的 `device_task_times`、`kernel_calls` |
| **3. PCIe 链路** | `state` 行的 `placement: host; host bytes`、`logical transfer bytes/row`、`decode bytes/step` | 用 `memory_budget_bytes` 检验大表放 host 后设备端是否放得下 | `host syncs`（读主机表不能引入同步） | 诊断与 trace 中的 `export_dropped`（有界异步导出丢包计数） |
| **4. 卡间通信 (TP)** | `parallel ... reservations are per rank` 及状态按 rank 复制提示 | `ModelConfig(tensor_parallel_size=...)` 准入检查直接拒绝不兼容特性 | `check` 在 CPU 上验证多卡路由与独立性（`check_on_gpu` 需 TP=1） | — |
| **5. GPU 时间线气泡** | `forward` 行（是否进 CUDA graph 或 eager 原因）、`member` 行的 `captured`、`host 0 syncs in traced workload math`、`timing` 条件节点说明 | `cuda_graphs=True` 在遇到任何会导致 eager 的特性时直接报错拒绝启动 | `check_on_gpu` 的 `host syncs: eight steady decode steps`（开 `sync_debug_mode("error")` 实跑） | `diagnostics().eager_reason` |
| **6. SM 重叠与并发** | `member` 行的 `anchor / lane`（`request.prepare` vs `token.commit`，`main` vs `side`）及 `state reads (origin, distance)`；`timing publish_lag` | 降低 `memory_budget_bytes` 可观察何时从 `side` 自动退回 `inline` | `check` 验证 `publish_lag` 语义、可见性、行隔离与抢占重放一致性 | `trace` 中主干与旁路任务的时间分布 |
| **7. 主机 CPU** | 纯 CPU 毫秒级运行本身（不触发 CUDA） | — | `check` 纯 CPU 契约快速回归 | `trace_snapshot()` 中的 `host_task_times` |

---

## 9. 试过但不行的事（负面结果精选）

知道“什么路走不通、为什么走不通”，能帮你在写自定义负载或改框架时避开我们已经踩过的坑。下面每一条都来自实测或架构验证：

* **N1（按分数动态淘汰 KV 显存，如 SnapKV / H2O）**：由于 vLLM 对每个请求只维护单一的 `num_computed_tokens` 标量，位置编码与物理槽位映射都由它线性推导，且块表（block table）只能追加不能在中间挖洞，因此无法在不破坏 vLLM 块管理器的前提下靠淘汰中间 token 来回收物理块；已放弃，保留静态几何布局（A8）和省带宽的 Top-K 读取（B7）。
* **N2（LastKV 式的压缩远端记忆 M7）**：vLLM 的 `HiddenStateCacheSpec` 在 $8,192\text{ B}$ 的物理页里只装 $2,048\text{ B}$ 有效内容（零显存节省），且深层可见条目数会随 prefill 分块大小（64/512/1024）在 960/896/768 之间波动，单请求下 `near+far+merge` 每层耗时 **$30.31\ \mu\text{s}$**，远慢于全量 FA2 的 **$11.88\ \mu\text{s}$**。
* **N3（复用 vLLM 原生 DSA Top-K 稀疏注意力内核）**：vLLM 的 DSA 路径硬性要求 **SM90+**（Hopper），在 RTX 4090（SM89）和 A100（SM80）上无法运行，调用 `pllm.attention(layer, "topk")` 会直接报错。
* **N4（用 `torch.compile` 编译作者手写的 loss 前反向）**：在 S-4090 的 `ESamp`（$\beta=0$）7 轮测试中吞吐为 **92.77%**，相比不编译没有任何可测收益。
* **N5（把整个捕获的训练执行器函数丢进 `torch.compile`）**：虽然旁路算子活动数从每步 59 个降到 25 个，但 PyTorch 的函数化（functionalization）为了消除原地写入，把整个分页状态池克隆了一份（多出约 $39\ \mu\text{s}$ 的 `clone` 内核，分页 AdamW 从 $29\ \mu\text{s}$ 恶化到 $64\ \mu\text{s}$），旁路 GPU 总耗时从 $\sim 178\ \mu\text{s}$ 涨到 $\sim 257\ \mu\text{s}$，端到端吞吐掉到 **90.10%**。
* **N6（在 GPU 已满载时强行提高旁路重叠度）**：去掉两个采样器等待事件把露在主干外的旁路时间从 $70.33\ \mu\text{s}$ 压到 $8.57\ \mu\text{s}$，但因为训练被挤进了显存带宽受限的采样阶段，端到端吞吐反而从 **94.49% 降到 92.73%**；把训练输入拷贝提前到采样前的试验也同样从 **94.3% 降到 92.9%**。
* **N7（在极小窗口下复用 FlashAttention-2 的 `window_size`）**：在 128 行、1000 次回放的微基准中，FA2 跑极小窗口 `(3, 0)` 耗时 **$91.47\ \mu\text{s}$**，比跑全量注意力（**$16.56\ \mu\text{s}$**）慢 5.5 倍；为保持架构简洁仍复用原生 FA2，但小窗口吞吐为全量的 **89.3% / 90.7%**。
* **N8（用 DMA 拷贝引擎或无限制 UVA 内核搬运主机表）**：设备端索引使 DMA 无法在不做 CPU 同步的前提下知道源地址；而不限制线程块数量的提前 UVA gather 会启动 **37,984 个 CTA** 占满 100% SM 饿死模型前向，$1,753.7\ \mu\text{s}$ 的读表只与前向重叠了 **$26.2\ \mu\text{s}$**，步时反而从 $5.697\text{ ms}$ 恶化到 **$5.939\text{ ms}$**（由此催生了限制为四分之一 SM 的 C2）。
* **N9（为量化模型的测试时训练缓存反量化后的稠密权重）**：在 A100 上跑 14B GPTQ 虽然能把步时从 $1.122\text{ s}$ 降到 $0.176\text{ s}$（6.38 倍），但会多占 **$24.63\text{ GiB}$** 常驻显存；已被峰值显存不变、步时 **$0.193\text{ s}$** 的瞬态反量化（A11）取代。
* **N11（`esamp94` 的一揽子微优化尝试）**：包括按活跃行/状态裁剪（少 14 个内核/步、省 39 MiB 预留，吞吐 91.70%/93.17%）、无环 `TensorBatch` 减少 GC（GC 从 $52.8 \rightarrow 10.4\text{ ms}$，但吞吐 $36,522.9$ vs $36,529.2\text{ tok/s}$ 无变化）、`NamedTuple` 记录、流优先级调整、CPU 绑核、复用完成选择等，**均未产生可复现的端到端吞吐提升**，因为瓶颈在 GPU 侧那约 $100\text{–}140\ \mu\text{s}$/步的真实训练图回放，而非主机侧。
* **N12（前向内精确 decode 更新的严格位级捕获）**：为支持 `visibility="exact"` decode 捕获增加了约 600 条语句，在默认容差模式下仅带来 **+3.3% / +5.2%** 加速，而在严格位级一致（strict bitwise）模式下捕获反而比 eager 更慢（bs1/bs4 为 **$48.84 / 78.66\text{ tok/s}$** 对比容差模式的 $59.51 / 113.16\text{ tok/s}$）。
* **N13（直接打补丁修改 vLLM 0.32 调度器内部以加速 ESamp）**：只带来约 **1%** 的收益，却引入了极脆弱的跨版本内部耦合，未合入。
* **N14（把引导采样的重加权与更外层的采样算术进一步融合）**：引入了 **$1.9 \times 10^{-6}$** 的数值偏差，未通过精度门槛而被拒绝（保留位级一致的 B3）。
* **N15（无条件按调度器最大 token 预算录制模型 CUDA graph）**：在 7B 模型（`gpu_memory_utilization=0.85`）上，强行录制 8192-token 图在 vLLM 0.27 上直接 OOM，在 0.32 上把可用 KV cache 从 76,144 个 token 挤到只剩 **3,872 个 token**（吞吐跌至约 **21%**）；已由 A7 修复。
* **N16（用 gather 出来的 FP32 张量手算 `sink` 注意力）**：在 S-4090 上吞吐只有 **$8,867\text{ tok/s}$**（对比全量注意力的 $39,193\text{ tok/s}$，仅 22.6%），已被复用原生 R-SWA 内核的 B8（$36,458\text{ tok/s}$）取代。

---

## 附录：测量设置与口径

本文引用的所有数字均来自同一套对齐的基准测试配置或对应教程 notebook 的实测输出，具体口径如下：

### 1. 标准硬件与对齐设置

* **S-4090（主力微基准与剖析设置）**：
  * **硬件与环境**：单卡 NVIDIA RTX 4090（128 个 SM，24 GiB 显存），`.venv`（vLLM 0.27.1，PyTorch 2.13+cu129），测试通过 `flock` 独占 GPU 并使用 `taskset -c 0-15` 固定 CPU 亲和性。
  * **模型与生成参数**：`Qwen/Qwen2.5-0.5B-Instruct`（BF16），8 条 prompt $\times$ `sampling_n=16`（共 128 条并发解码序列），每条序列生成 256 个 token，开启 `ignore_eos`，`temperature=0.8`、`top_p=0.95`、`top_k=-1`，关闭 prefix caching，默认开启 CUDA graphs，1 轮预热（warmup）。
  * **比值口径**：所有百分比（如 `92.48%`、`83.99%`）均为**同一次会话中相对于原生（vanilla）开启 CUDA graph 的 vLLM 吞吐比值**。关于如何运行对齐基准，参见 [《基准解读》](benchmarking.md)。
* **S-tut（教程 notebook 输出设置）**：
  * 来自 `doc/tutorials/` 下各 `.ipynb` 单元的实测输出（见 [《能力地图》](capability-map.md)）。
  * 模型同样为 `Qwen/Qwen2.5-0.5B-Instruct`，8 条 prompt $\times n=16$（注：[教程 06](../tutorials/06_reading_inside.ipynb)、[教程 07](../tutorials/07_custom_residual.ipynb)、[教程 12](../tutorials/12_adaptive_depth.ipynb) 使用 $n=4$），生成 256 个 token，1 轮预热，**正式运行 2 轮取较好的一轮（best of 2 rounds）**，并与同一个 notebook 内测得的原生图模式 vLLM 吞吐求比值。其中 [教程 11](../tutorials/11_lookup_memory.ipynb) 的主机/设备查表数字已更新为在 `main`（`ea7ff796`，RTX 4090、vLLM 0.27.1）上重跑的结果。
* **多卡与大模型专项设置**：
  * 第 4 章（D1）与 B9、A11/N9 的 A100 数据在 NVIDIA A100-SXM4-80GB 上测得（模型为 `Qwen/Qwen2.5-7B-Instruct` 或 14B GPTQ）。
  * 第 5 章（E5）与 [教程 08](../tutorials/08_inplace_ttt.ipynb) 的 In-Place TTT 数据在 RTX 4090 上使用 4B 社区权重测得（4096 或 16k prompt + 256 生成）。
* **静态计划数字（`explain@5cdc9785`）**：
  * 文中引用的 `pllm.explain` 字节数、槽位数与 FLOP 均在 `main` `5cdc9785` 上以 `max_num_seqs=128` 实跑 CPU 规划得出。

### 2. 版本与待复测说明（Gaps）

* **来自较早提交、尚未在最新 `HEAD` 复测的专项数字**：
  * `StepCapture` 开启前后的对比（第 5 章 E2，$19,100 \rightarrow 23,829\text{ tok/s}$，测于 2026-10-18）；
  * `publish_lag` 开启前后的对比（第 6 章 F1，$91.4\% \rightarrow 94.3\%$ 及 $73.7\% \rightarrow 84.1\%$，测于 2026-10-19）；
  * 真实代码追踪替代手写公式的峰值对比（第 1 章 A4，测于 2026-10-19）；
  * `pllm.scan` 并行前缀扫描的微基准加速比（第 6 章 F2，6.57 倍 / 3.34 倍，测于 2026-10-23 早期分支）。
* **尚未单独拆出端到端吞吐数字的机制**：字段级写时复制（A2）、旁路退回内联的吞吐代价（A5）、共享 tap 缓冲（A12）、KV 卸载（A15）、感知负载的 prefix cache（A16）、同卡零拷贝投喂（A17）、分页 AdamW 的独立加速比（B1）、行掩码裁剪（B5）、异步导出开销（C4），以及 `explain` 的 FLOP/字节静态估算与实测硬件计数器的偏差。它们目前作为保障容量安全、按位正确性或组合能力的机制工作。
