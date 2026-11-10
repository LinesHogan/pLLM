# 可编程注意力与 KV 布局

如果你把过去两年顶会上的大模型推理论文摊在桌子上，怕是有相当一部分在打注意力（Attention）的主意。

这些论文的出发点往往优雅又直观：在长文本生成中，KV cache 随着上下文长度线性膨胀，但到了生成后期，绝大多数历史 token 分到的注意力权重几乎为零。既然如此，为什么不把那些没人看的旧 token 扔掉？于是我们看到了五花八门的设计：有的让偶数层只看最近 256 个 token、奇数层看全量（局部与全局交替）；有的让奇数层干脆不存 KV，直接借用下一层的缓存（跨层共享）；有的每一步只挑出注意力得分最高的几个物理页来读（如 Quest）；还有的在生成过程中实时统计每个 token 的累积注意力分数，把低分 token 从显存里踢出去（如 H2O、SnapKV、PyramidKV）。

在 Hugging Face 的 `generate` 循环里验证这些点子，通常只需要一个下午。毕竟 `past_key_values` 就是一串普通的 PyTorch 张量，你想改掩码、做切片、还是在中间删掉几列，全凭一行索引操作。

但当你带着这些跑通了的脚本走进生产环境，试图把它们塞进 vLLM 这样的高吞吐服务引擎时，你会迎头撞上一堵看不见的墙。你会惊讶地发现：那些在论文里被统称为“稀疏注意力”或“KV 压缩”的算法，一旦放进真实的物理引擎，会瞬间分裂成三个截然不同的物种——**有些能实打实地把并发容量翻倍，有些只能少读几个字节却省不下哪怕一兆显存，而另一些，在不把引擎拆骨重装的前提下，根本就无法实现。**

为什么会这样？为了回答这个问题，我们不能只盯着算法公式看，而必须往下走一层，去看看现代推理引擎到底是怎么给显存“记账”的。

> **实验环境说明**：除特别注明用于对比小模型固定开销的 Qwen2.5-0.5B 数据及历史微基准外，本文所有性能与显存容量数据均在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0、PyTorch 2.13（BF16，开启 CUDA Graph 与 FlashInfer 采样器）下基于 `Qwen/Qwen2.5-7B-Instruct` 实测。吞吐测试采用 128 并发序列（8 个 prompt × 每个采样 16 条分支，每步最多调度 128 个 token，生成 256 个 token），预热 1 轮后取 3 轮中位数。

---

## 第一条分界线：“少读”不等于“少存”

在单卡跑实验的时候，研究员最关心的是“这一步少算了多少 FLOPs”或者“少读了多少字节”。但在多租户的在线服务（Serving）里，决定你一台机器能接多少单的头号指标只有一个：**一个请求在最坏情况下，最多会占掉多少物理 KV 显存？**

我们来算一笔最基础的账。以 `Qwen2.5-7B` 为例：模型一共 28 层，采用分组查询注意力（GQA），每层有 28 个 Query 头和 4 个 KV 头，每个头维度是 128，使用 2 字节的 BF16 存储。假设我们的服务允许每个请求最长跑到 `max_model_len = 2048` 个 token。那么这一个请求在 28 层里要存下 K 和 V 两份张量，它的物理显存上限是：

$$2048 \times 28 \times 2 \times 4 \times 128 \times 2\text{ Bytes} = 112.0\text{ MiB}$$

如果你在 CPU 上调用 `pllm.explain` 去检查一个不做任何裁剪的 7B 负载，它报出来的单请求 KV 容量上限分毫不差，正是 **112.0 MiB**。

现在，假设你读了 Quest（Tang et al., 2024）的论文，觉得它的想法非常聪明：既然 KV 是按页（比如每页 16 个 token）存放的，我们可以记录每一页里 Key 向量在各个维度上的最小值和最大值，用当前 Query 向量去和这个极值框做点积，快速估算出每一页注意力分数的上界。然后，每个注意力头只挑出得分最高的 8 页（也就是 $8 \times 16 = 128$ 个 token）来算注意力，其余的页统统跳过不读！

在 pLLM 里，你不需要碰任何底层 C++ 代码，用几十行 Python 就能把这个按页挑选的 `QuestRead` 写出来。我们用 `pllm.explain` 给它算一笔读取账：
*   如果不做筛选（`NativeRead`），在 2048 长度下，第 2 层每个 token 最多要读 **4,194,304 字节** 的 KV；
*   换上 `QuestRead`（每头只读 8 页、128 个位置）后，每个 token 最多只读 **1,835,008 字节** 的 KV（在 0.5B 模型上则是从 1,048,576 字节降到 458,752 字节）。

慢着——你可能会盯着这两个数字皱起眉头：128 个 token 明明只有 2048 的 **1/16**（6.25%），为什么读取字节数的上限还占了全量的 **44%**？

这就是系统视角给我们的第一个惊喜：**GQA 的写时共享与读时分化。** 在 `Qwen2.5-7B` 里，每 7 个 Query 头共享 1 个 KV 头。当做全量注意力时，7 个 Query 头齐步走，同一个 KV 页从显存读进寄存器一次，就能同时喂饱 7 个 Query 头；但一旦你让每个 Query 头按自己的喜好独立挑选 Top-8 页，最坏情况下，同组的 7 个 Query 头挑中的页互不重合，那个共享的 KV 头就得被迫为它们各自去捞不同的物理页！`explain` 在你还没启动 GPU 之前，就把这笔潜藏的带宽账摆在了明面上。

但更关键的问题还在后面：既然每个头每步只读 128 个 token，那这个请求的 KV 显存上限从 112.0 MiB 降下来了吗？

**一分钱都没降，还是 112.0 MiB。**

道理一捅就破：你在第 500 步觉得第 3 页不重要、跳过去没读，不代表第 501 步生成的那个新 Query 也不需要第 3 页。为了保证未来的每一步都能从**全部历史页**里重新打分挑出最好的 8 页，所有的历史 KV 页都必须老老实实地留在显存里待命。当 vLLM 的调度器（Scheduler）决定能不能再放进一个新请求时，它依然得按 2048 个 token 的全额预算给这个请求留出 112.0 MiB 的坑位。

雪上加霜的是吞吐。当我们在 7B 上实测示例 10 里的 `QuestRead` 时，并发吞吐只有原生 vLLM 的 **14%**（697 tok/s 对比 5,165 tok/s）。因为在未编写专用融合算子的情况下，用 PyTorch 每层每步把所有页的 Key 捞出来转成 FP32 算极值、打分、再做动态 gather，花掉的开销远远超过了少读那几页省下的带宽。

这就划出了我们的第一条硬分界线：**动态稀疏读取（Selection）省的是逻辑读取量，省不下已分配的物理显存。** 如果你的目标是让一张卡同时多服务一倍的用户，你不能只跳过不读，你必须让那些不需要的物理内存块**真正回到空闲池（Free Pool）里去**。

那么问题来了：既然按页打分只看不删省不下显存，我们能不能干脆再往前迈一步——像 H2O、SnapKV 或 PyramidKV 那样，在生成过程中算出每个历史 token 的分数，把低分 token 占用的物理显存当场释放掉？

---

## vLLM 的账本：三条不可逾越的硬不变量

在 Hugging Face 里，删掉中间几个 token 的 KV 只是一次张量拼接。但在 vLLM v1 里，当你试图去写一个“按分数释放中间某块 KV”的插件时，你会发现整个引擎的地基都在排斥你。

为什么？很多系统设计的死胡同，表面上看是 API 没暴露，往深处看其实是**记账模型（Accounting Model）的不兼容**。vLLM 之所以能把分页注意力（PagedAttention）和连续批处理跑出极高的吞吐，是因为它在 CPU 调度器和 GPU Worker 之间立下了三条极其冷酷的记账不变量。

让我们假设块大小（`block_size`）是 16 个 token，一个请求刚刚算完了前 40 个 token，正在算第 41 个 token（逻辑位置 `pos = 40`）。掀开 vLLM v1 的引擎盖，你会看到这三条铁律是如何咬合在一起的：

### 铁律一：进度是一个全模型共享的标量

如果你去翻 vLLM 调度器里维护的请求状态，你会惊讶地发现，它根本没有维护什么“每一层、每个头当前保留了哪些 token”的列表。对每一个请求，调度器只记了孤零零的一个整数：

$$\texttt{num\_computed\_tokens} = 40$$

这个标量隐含了一个全局假设：**如果第 0 层算到了第 40 个 token，那么第 27 层也一定不多不少存着 40 个 token。** 如果你想实现 PyramidKV——比如让浅层保留 40 个 token，让第 15 层扔掉 16 个、只留 24 个——调度器的账本上连填“第 15 层少了 16 个”的格子都没有。

### 铁律二：逻辑位置、注意力长度与物理槽位被同一个整数绑死

当第 41 个 token 进入 GPU Worker 时，Worker 不会去查任何复杂的哈希表。它直接拿 `num_computed_tokens` 这一个整数，靠最朴素的整数算术推导出三件事：
1.  **本步注意力的读取长度**：`seq_len = num_computed_tokens + query_len`；
2.  **新 token 的绝对位置**（用于乘上 RoPE 旋转位置编码）：`pos = num_computed_tokens + i`；
3.  **新 token 写入物理显存的槽位（`slot_mapping`）**：直接去块表里查第 `pos // 16` 格拿到物理块号，再算 `pos % 16` 拿到块内偏移。

```text
Logical Pos:    0 .. 15        16 .. 31        32 .. 39 [40]  <-- pos = 40 (new token)
              ┌──────────────┬──────────────┬──────────────┐
Block Table:  │ Physical #7  │ Physical #2  │ Physical #9  │
              └──────┬───────┴──────┬───────┴──────┬───────┘
                     │              │              └──► write slot = 9 * 16 + (40 % 16)
                     ▼              ▼
  (A) Punch a hole (free Physical #2, keep table):
      FlashAttention scans [0, 41) ──► dereferences Physical #2 (now owned by another user!)

  (B) Compact table (shift Physical #9 left to index 1):
      Tokens 32..39 now sit at logical indices 16..23, clashing with their baked-in RoPE!

  (C) How vLLM's own SlidingWindow / R-SWA reclaims blocks safely:
              ┌──────────────┬──────────────┬──────────────┐
Block Table:  │  null_block  │ Physical #2  │ Physical #9  │  (table length unchanged!)
              └──────────────┴──────────────┴──────────────┘
              └── Window mask guarantees kernel NEVER reads [0..15]
```

看清这张图里的死结了吗？**逻辑位置、序列长度和块表下标，被绑死在了同一根数轴上。**

假设你在跑完一步之后，发现中间第 16 到 31 号 token（对应 `Physical #2`）注意力得分很低，想把它驱逐掉。你只有两条路可走，而两条路都是悬崖：
*   **路径 A（原位释放，不改块表长度）**：你把 `Physical #2` 还给了空闲池，池子转手把它分配给了另一个刚进来的用户李四。下一步，由于标准 FlashAttention 默认扫描 `[0, seq_len)` 的整段连续区间，内核会毫无防备地冲进块表第 1 格指向的 `Physical #2`——把李四的私密对话当成张三的历史上下文读进来！这在多租户服务里是绝对不可接受的内存污染与数据越权。
*   **路径 B（释放并左移压缩块表）**：你把后面的 `Physical #9` 往前挪到第 1 格，把 `num_computed_tokens` 改成 24。且不说第 32..39 号 token 在写进缓存前就已经乘过了位置 32..39 的 RoPE 旋转角度、挪位置会导致相对距离错乱；单看下一步第 41 个 token 进来时，Worker 会按 `pos = 24` 给它施加 RoPE——一个本来排在第 40 位的字，硬生生戴上了第 24 位的旋转角！

### 铁律三：Worker 端的块表是一本“只追加日志（Append-only Log）”

就算你愿意改位置编码，还有第三道物理屏障挡在面前：在 vLLM 的架构里，CPU 调度器每一步发给 GPU Worker 的运行中请求更新（`CachedRequestData`），**只有一个叫 `new_block_ids` 的字段，而且 Worker 端在更新本地块表时写死了 `overwrite=False`。**

换句话说，调度器与 Worker 之间的通信协议里，只有“在块表末尾追加几块新页”这一种动词，根本不存在“删掉中间第 $k$ 块”或“把整张块表左移压缩”的指令通道。更何况，每一步根据运行时的注意力分数动态决定删哪一块、再去改写块表，需要 CPU 主机介入同步，这会直接打碎 CUDA Graph 静态回放赖以生存的固定内存拓扑。

你可能会问：*“等等——既然块表只能追加、不能缩短，那 vLLM 自己内置的滑动窗口（Sliding Window）和保留前缀窗口（R-SWA）是怎么把显存还回池子里的？”*

答案就在上面那张图的 **(C)** 行里。当窗口向右滑动、最老的 `Physical #7` 彻底滑出窗口边界时，vLLM 的 `SlidingWindowManager` 确实会把 `Physical #7` 回收到空闲池；但为了不破坏“位置 = 块表下标 × 16”的第二条铁律，也没法缩短 Worker 的块表，它在原位置塞进了一个全局共享的哑块——**`null_block`**（0 号占位块），保持块表长度纹丝不动！

为什么换成 `null_block` 不会读到脏数据？**因为滑动窗口的几何形状是静态且规整的。** 注意力内核自带的窗口掩码（Window Mask）在数学上保证了它只读末尾窗口内的块（或者 R-SWA 下的完整 prompt 块加末尾窗口块），绝对不会去解引用前面那些已经被替换成 `null_block` 的位置。`null_block` 里装的是什么垃圾，内核根本不在乎。

---

## 为什么我们对 SnapKV 和远端记忆压缩说“不”

理解了这三条记账不变量，你就明白为什么我们没有在文档里写一句模棱两可的“暂不支持按分数驱逐 KV，未来版本将加入”，而是直接给出了一个明确的架构结论：**在不推翻 vLLM v1 核心记账假设的前提下，不存在任何干净的方案来实现释放物理显存的、基于分数的动态 KV 驱逐。**

在下这个结论之前，我们并不是没有挣扎过。我们把能想到的四种绕过方案在源码层面逐一推演了一遍，看看如果硬要塞进去会发生什么：

| 尝试过的绕行方案 | 为什么在 vLLM v1 架构下会破产 |
| :--- | :--- |
| **方案 1：写一个块粒度的自定义 `KVCacheSpec`** | Token 级的注意力分数根本对不齐 16 个 token 一块的物理页边界；更致命的是 Worker 端块表只追加不更新，释放后的块留在块表里会被标准 FlashAttention 当场读出脏数据，计算结果直接错误。 |
| **方案 2：每隔 $N$ 步把高分 token 抽出来重新 Prefill** | 被剔除的低分 token 在原本的 Prefill 阶段已经通过浅层注意力把信息注入了保留 token 的深层表示里；如果只拿保留下来的 token 重新跑一遍 Prefill，浅层上下文彻底丢失，数学语义已经不再是 SnapKV，且重算开销极其昂贵。 |
| **方案 3：借道 KV Connector 导出、释放再重载压缩后的 KV** | 受限于单标量 `num_computed_tokens`，所有层、所有头被迫保留完全相同的长度，无法支持 PyramidKV 这类逐层不同预算的算法，且必须在调度器里伪造请求生命周期。 |
| **方案 4：在 GPU 上原地 gather 压缩缓存并截断块表尾部** | 同样无法表达跨层异构长度（除非按全模型最大保留长度补齐，白白浪费显存），而且必须同时侵入修改调度器、块管理器和 Worker 块表同步通道三处核心代码。 |

同样的第一性原理推演，也让我们否决了另一个看起来很美的方向：**压缩远端记忆（如 Activation Beacon、LastKV）**——也就是近处保留原始 KV，把远处的历史每几个 token 浓缩成一个“摘要条目（Summary Entry）”。

当我们动手写完原型去算物理账时，三盆冷水迎面泼来：
1.  **分页的填充损耗（Page Padding Waste）**：vLLM 的页是按固定大小分配的。假设你把每 4 个远端 token 压成 1 个宽 256 的 BF16 条目，一页里装 4 个压缩条目，有效内容只有 $2,048\text{ Bytes}$，但它依然要占掉一个完整的 $8,192\text{ Bytes}$ 物理页！由于块对齐填充，远端显存分配一点也没省下来。
2.  **分块预填充（Chunked Prefill）的非确定性**：如果远端摘要条目依赖较深层的隐藏状态，那么在当前步某个位置的摘要算没算出来，完全取决于调度器在这一步把长 prompt 切成了多大的块。这意味着同一段输入文本，在服务器空闲和繁忙时会算出不同的注意力结果，直接违反了 pLLM 最看重的确定性契约。
3.  **双段注意力的延迟倒挂**：在解码时，先对近处窗口跑一次注意力，再对远端摘要跑一次注意力，最后把两段输出和 log-sum-exp 合并——在微基准测试（历史测量）中，这一套组合拳下来单层耗时 **30.31 µs**，而直接让原生内核一口气读完全量历史只要 **11.88 µs**。

在系统工程里，知道“什么不该做”往往比知道“什么能做”更重要。既然动态挖洞和远端分页压缩都在跟底层硬件与引擎的物理定律作对，那就果断放手。把需求拆回本质：
*   如果你只想**少读数据**（实验稀疏注意力规律），用只读的 `topk`；
*   如果你真的想**省下物理显存、提高并发上限**，那就顺应 vLLM 的记账规则，用静态的几何布局。

---

## 伪装的艺术：零行内存管理代码

那么，在不破坏 vLLM 记账铁律的前提下，到底哪些几何形状是能够真正把物理块还回池子里的？

答案恰恰就是 vLLM 自己的原生管理器已经会管的那三种结构：**连续前缀滑出（滑动窗口）**、**保留完整 Prompt 加末尾滑动窗口（R-SWA）**，以及**整层不分配缓存、直接借用下方某层（跨层共享）**。

```text
1. keep="all" (Full Attention — all blocks stay live):
   [ Block 0 ][ Block 1 ][ Block 2 ][ Block 3 ][ Block 4 ]

2. keep="window", window=256 (Sliding Window — prefix blocks recycled):
   [  null   ][  null   ][  null   ][ Block 3 ][ Block 4 ]
      └─────────┬─────────┘
         Freed to pool!

3. keep="window", window=256, sink=True (Prompt Sink + Window / R-SWA):
   [ Prompt0 ][ Prompt1 ][  null   ][ Block 3 ][ Block 4 ]
                            └──┬──┘
                         Freed to pool!

4. keep="none", source=1 (Cross-Layer Shared — zero blocks allocated):
   Layer L   ──┐ (shares kv_sharing_target_layer_name)
               ▼
   Layer L-1: [ Block 0 ][ Block 1 ][ Block 2 ][ Block 3 ][ Block 4 ]
```

想明白这一点后，pLLM 的设计决策变得简单得近乎偷懒：**我们一行新的内存管理代码都不写。我们只在 vLLM 启动、进行显存探测和构建块管理器之前，走上前去，把用户声明的每一层“伪装”成 vLLM 原本就认识的那三种原生层。**

在 `pllm/contracts/kv_layout.py` 里，每一层的存储契约被浓缩成了五个字段的数据类：

```python
# check-docs: skip: excerpt from pllm/contracts/kv_layout.py
@dataclass(frozen=True)
class KVLayout:
    """One layer's storage; source counts layers below, sink retains the entire prompt."""

    layer: int
    keep: Keep = "all"
    window: int = 0
    source: int = 0
    sink: bool = False
```

而在后端挂载时，`pllm/backends/vllm/kv_layout.py` 里的 `install_layouts` 只用了十来行代码，就把这三种声明翻译成了 vLLM 自己的内部属性：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/kv_layout.py
for layout in layouts:
    attention = model.model.layers[layout.layer].self_attn.attn
    storage = storage_layout(layouts, layout.layer)
    if layout.keep == "none":
        target = model.model.layers[storage.layer].self_attn.attn
        attention.kv_sharing_target_layer_name = target.layer_name
    elif layout.keep == "window":
        if layout.sink:
            attention.get_kv_cache_spec = partial(sink_spec, attention, layout.window)
        else:
            install_window(attention, layout.window)
```

看看这三次四两拨千斤的映射：
*   当 `keep="none"` 时，我们先通过 `storage_layout` 顺着共享链往下找到真正拥有缓存的那一层，然后把当前注意力层的 `kv_sharing_target_layer_name` 指向它。vLLM 原本为跨层共享 KV 的模型（如 Gemma 3n）准备的机制立刻接管一切，当前层连一个字节的 KV 都不会分配；
*   当 `keep="window"` 且不带 `sink` 时，`install_window` 给注意力层打上窗口标记（读取走 FlashInfer 窗口计划，其中的门道我们在《复用上游》一篇里细聊），vLLM 自动为它生成 `SlidingWindowSpec`；
*   当加上 `sink=True` 时，我们把该层的 `get_kv_cache_spec` 替换成返回 `RSWASpec`（保留整个 prompt 加上最近窗口；注意这与 StreamingLLM 只保留开头几个固定 token 略有不同，它是把整个用户 prompt 完整钉在缓存里，只让长生成过程的中间段滑出回收）。

因为这次“伪装”发生在 vLLM 探测显存和初始化 `KVCacheCoordinator` **之前**，接下来无论是启动时的最大并发探测、运行时的块分配与 `null_block` 回收，还是请求被抢占（Preemption）时的引用计数清理，全都是 vLLM 自己的原生管理器在干活。甚至连 `pllm.explain` 在 CPU 上离线预算每个请求需要多少块，都不需要我们自己推导公式——`layout_capacity` 直接在内存里实例化一个 vLLM 的 `SlidingWindowSpec`，调用它原生的 `max_admission_blocks_per_request` 把数字拿出来。当多个静态布局组的页大小相同时，`streamline_allocation` 还会顺手把运行中请求的页数预测合并成一次计算，而底层的追加和释放依然百分之百由原生管理器拍板。

不过，把新概念伪装成宿主系统的原生原语，偶尔也会撞上宿主内部微妙的潜规则。我们在实现 `sink=True`（即 R-SWA）时就踩过一个极有意思的坑。

刚把 `RSWASpec` 挂上去时，我们发现那一层在运行时根本没有按“Prompt + 窗口”去回收中间的块，而是跟普通的全量注意力层混在了同一个管理组里，把所有历史块全留着！

顺着 vLLM 给各层 `KVCacheSpec` 分组的代码一路查下去，真相让人哭笑不得：vLLM 在决定“哪些层可以合并成一个组共享同一套块表”时，会拿出字典里的**第一个** `spec`，调用它的 `is_uniform_with_collection(specs)` 去问：“喂，你们后面这些人跟我兼容吗？”

在基座模型里，第 0 层往往是普通的全量注意力（`FullAttentionSpec`），而 `RSWASpec` 在类继承关系上恰好是它的子类。当全量注意力的 `spec` 排在第一个被询问时，它扫了一眼后面的 `RSWASpec`，看头数、头维、数据类型全都对得上，又是自己的子类，大度地点了点头：“兼容，大家都归我管！”——于是，挑剔的 R-SWA 层就这样被宽容的全量注意力层给“吞并”了。

弄清了这个潜规则，修法只需要十几行代码，写起来像在破解一道脑筋急转弯：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/kv_layout.py
class SinkSpec(RSWASpec):
    """Keep R-SWA allocation separate from full attention; the native RSWAManager owns blocks."""

    def is_uniform_with_collection(self, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """Only layers with the same prompt-sink window may share allocation decisions."""
        return all(
            isinstance(spec, RSWASpec) and spec.rswa_window == self.rswa_window for spec in kv_cache_specs.values()
        )


def sink_first(specs: dict[str, KVCacheSpec]) -> dict[str, KVCacheSpec]:
    """vLLM asks the first spec about compatibility; full specs would absorb R-SWA subclasses."""
    return dict(sorted(specs.items(), key=lambda item: not isinstance(item[1], RSWASpec)))
```

第一步，派生一个 `SinkSpec`，重写它的 `is_uniform_with_collection`，立下规矩：只有大家全都是 `RSWASpec` 且窗口大小一模一样时，我才点头；第二步，用 `sink_first` 把字典排个序，把所有 `RSWASpec` 挪到最前头，确保 vLLM 第一个去问的永远是这个挑剔的 `SinkSpec`。不改 vLLM 一行源码，R-SWA 层立刻乖乖地独立成组，由原生的 `RSWAManager` 把滑出窗口的物理页精准地收回池中。

---

## 读存分离：挂在原生 KV 写入之后

把“怎么存”交给了装饰器参数和原生管理器后，方法体内部就只剩下一件事：**怎么读。**

看看写一个“偶数层只留 256 窗口、奇数层全量”的局部/全局交替注意力有多干净：

```python
# check-docs: skip: excerpt from doc/guides/forward-structure.md
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """偶数层保留最近 256 个 token 的 KV，奇数层保留全量 KV。"""

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]
```

为什么方法体里连一行写缓存的代码都没有？这要归功于我们选择的挂载边界。

在 vLLM 的模型图里，注意力并不是一串散装的 PyTorch 算子，而是包在一个不透明的自定义算子（Opaque Custom Op）里，以便 Dynamo 编译和分段 CUDA Graph 捕获。`@pllm.custom_attention` 没有去拆散外面的 QKV 线性投影和 RoPE 旋转，而是直接把钩子（`AttentionHook`）装在了**这个不透明算子的内部、原生 KV 缓存写入刚刚完成的那一瞬间**：

```text
Inside vLLM's Opaque Attention Op (Graph Ordering & Capture Included):
  Hidden ──► [ QKV Proj & RoPE ] ──► [ Native KV Cache Write ]
                                                 │
                                                 ▼
                           ┌─── PagedKV (Read-Only Borrowed View) ───┐
                           │   @pllm.custom_attention: read(q, kv)   │
                           └─────────────────────┬───────────────────┘
                                                 │ kv.attend(q, window=..., topk=...)
                   ┌──────────────────┬──────────┴──────────┬──────────────────┐
                   ▼                  ▼                     ▼                  ▼
              Full Read         Storage Window         Prompt-Sink       Selected (topk)
               (FA2)             (FlashInfer)      (Triton unified_attn) (PyTorch gather)
```

这个位置选得极其讨巧：当你的 `read(self, q, kv)` 被调用时，当前步新 token 的 Key 和 Value 已经被上游内核妥妥当当地写进了分页缓存，CUDA Graph 的节点顺序和录制上下文也全都由上游算子铺设好了。你拿到的 `q` 是已经做完旋转编码的查询张量，而 `kv` 是一个只读的借用视图 `PagedKV`：

```python
# check-docs: skip: excerpt from pllm/execution/attention.py
@dataclass(frozen=True)
class PagedKV:
    # ……
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_start: torch.Tensor
    query_count: int
    read_attention: AttentionRead
    storage_window: int = 0
    prompt_lens: torch.Tensor | None = None

    @property
    def requests(self) -> torch.Tensor:
        # ……
        rows = torch.arange(self.query_count, device=self.query_start.device)
        return torch.searchsorted(self.query_start[1:], rows, right=True).clamp(max=len(self.seq_lens) - 1)
```

注意看 `PagedKV` 的设计。第一，它是冻结的借用视图，你没法在读方法里把它的字段换掉，框架也从不通过它改写缓存；第二，就像我们在《行模型》里讲过的，此时进来的 `q` 是不同请求拍扁拼在一起的二维紧致张量（末尾还可能跟着为了凑齐 CUDA Graph 尺寸而填进来的 padding 行）。如果你的算法需要知道每一行 Query 属于哪个请求、或者当前处在序列的哪个逻辑位置，`requests` 属性直接用一次 GPU 上的 `torch.searchsorted` 对着 `query_start` 算出来（末尾的图 padding 行被安全地夹到最后一个请求上，反正它们的输出随后会被丢弃），全程没有任何 GPU 到 CPU 的同步。

当你调用 `kv.attend(q, window=..., topk=...)` 时，它会返回 `(输出, float32 LSE)` 元组（如果你在方法里多返回一个 `aux` 张量，后面的方法还能通过 `pllm.attention(layer, "aux")` 把它接走，比如用来监控局部注意力占比）。而在底层，`FlashRead` 会根据当前的布局和参数，自动分派给最合适的内核：
*   **全量读取**：直接走 vLLM 的 FlashAttention-2；
*   **静态存储窗口**：走 vLLM 的 FlashInfer 窗口包装器（这里有个反直觉的教训：早先我们直接传 FA2 的 `window_size`，结果在 0.5B 上因为 FA2 窗口路径关掉了 GQA 的 7:1 Query 头打包，每次读取耗时飙到 86.43 µs，吞吐掉到 78.8%；当窗口设成极小的 3 个 token 时，FA2 甚至要花 91.5 µs，比算全量历史的 16.6 µs 慢了五倍多！改走上游的 FlashInfer 才把吞吐追回来，详情见《复用上游》）；
*   **Prompt-Sink（R-SWA）**：走 vLLM 的 Triton `unified_attention`（由于上游这个内核把 softmax 的中间统计量丢掉了，我们在 `rswa_attention.py` 里额外补了一次不分段的轻量启动，专门把 `log-sum-exp` 救回来）；
*   **带 `topk` 的选择读取**（或者当 R-SWA 遇到无法直接表达的更宽窗口交集与混合精度时）：走 `selected_attention` 的受掩码 gather。这里有一个极容易踩雷的细节：在 gather 之前，代码必须先用 `valid` 掩码把越界或已经滑出窗口的位置拦截下来，把它们的块表索引强制洗成 0，**绝对不能解引用任何已经被回收到空闲池的物理页**——哪怕那个通道在后面的 softmax 里会被 `-inf` 掩掉，一旦硬件真的去读了，就是实打实的越界或越权读取。

---

## 解剖室里的成本账：显存、模型数学与“逐位中性”

现在，所有的拼图都各就各位了。让我们把这几种写法放进同一台测试机，看看真实世界里的收益和代价到底长什么样。

首先是示例 10 在 `max_model_len = 2048` 下的账单（同时列出 0.5B 的数字作为参照，因为小模型权重小，留给 KV 池的显存比例更高，能更直观地看到单卡最大并发请求数的跃升）：

| 负载写法 | 每请求 KV 上限（7B） | 每请求 KV 上限（0.5B） | 2048-token 并发容量（0.5B） | 实测吞吐（7B，相对原生 5,165 tok/s） |
| :--- | ---: | ---: | ---: | ---: |
| **NativeRead**（全量保存，原生读） | 112.0 MiB | 24.0 MiB | 358.0 个 | **99%**（5,111 tok/s） |
| **LocalGlobal**（偶数层窗口 256） | **70.4 MiB** | **16.6 MiB** | **517.8 个**（1.45×） | **100%**（5,160 tok/s） |
| **CrossLayerShared**（奇数层借用下一层） | **56.0 MiB** | **12.0 MiB** | **716.0 个**（2.00×） | **98%**（5,073 tok/s） |
| **PromptPlusWindow**（Prompt + 窗口 256） | 112.0 MiB | 24.0 MiB | 358.0 个 | — |
| **QuestRead**（全量保存，每步挑 8 页读） | 112.0 MiB | 24.0 MiB | 358.0 个 | **14%**（697 tok/s） |

这张表把“读”与“存”的物理分野展现得淋漓尽致：
*   `LocalGlobal` 和 `CrossLayerShared` 不但保持了与原生 vLLM 几乎一致的吞吐（98%–100%），而且实打实地把单请求 KV 显存砍到了 **70.4 MiB** 和 **56.0 MiB**，在 0.5B 上让同一张卡能同时接纳的 2048-token 请求数从 **358 个** 直接跳到了 **518 个** 和 **716 个**！
*   至于 `PromptPlusWindow`（`sink=True`），为什么它的准入上限依然是 112.0 MiB？因为准入预估必须防备最坏情况——万一用户发来的 prompt 本身就有 2048 个 token，而你又承诺保留整个 prompt，那它确实需要 112.0 MiB。它的节省发生在“短 prompt + 长生成”的运行过程中：比如在一次 0.5B 的实测里（prompt 8 个 token + 生成 112 个 token，16-token 页），它在运行期实打实地省下了 **6.25%** 的物理页分配。
*   在短上下文（比如上面吞吐测试里总长约 270 个 token 的序列）下，256 的窗口几乎还没填满，既不会触发大量块回收，也不会因为少读而显著提速——窗口布局的红利，只有在长序列生成越过窗口边界之后才会源源不断地兑现。

但是，作为系统设计者，我必须在这张漂亮的显存成绩单旁边贴上两张醒目的警告标签。

### 第一张标签：改 KV 布局，改的是模型的数学

在示例 10 里，我们在 0.5B 上做了一个小小的“大海捞针”实验：在一篇 1500 个 token 的博物馆介绍长文中间，悄悄埋进一句 `"The secret code for the blue door is 7361."`，然后在末尾问模型密码是多少。

*   原生 vLLM 干脆利落地回答：`'7361'`；
*   保留完整 prompt 的 `PromptPlusWindow` 和按页检索的 `QuestRead(pages=8)` 也都准确答出了 `'7361'`（如果把 `QuestRead` 改成只看最近 8 页的 `recent_only=True`，回答就变成了空字符串 `''`；把预算砍到 `pages=2`，回答就变成了错误的 `'9'`）；
*   然而，把偶数层砍成 256 窗口的 `LocalGlobal` 回答了一个莫名其妙的 `'3'`；而让奇数层直接借用偶数层 KV 的 `CrossLayerShared` 更是当场吐出了一串乱码：`'arga textDecorationugi-ingenSalesproj:oランキング étapelessKey'`！

是我们的跨层共享或滑动窗口代码写出了 bug 吗？不是，布局本身按声明工作。真正的原因是：**`Qwen2.5` 在预训练时，每一层都习惯了看自己完整的历史 KV。** 当你强行抽掉奇数层的 KV 让它去读偶数层的缓存，或者把偶数层 256 步之前的记忆抹掉时，你改写的是模型的前向数学函数。对于原版就按交替窗口或跨层共享训练出来的模型（或者经过轻量微调适配的权重），这些布局是白捡的显存红利；但对于普通的全注意力检查点，在欢呼并发翻倍之前，请务必先在你的真实评测集上验一验生成质量。

### 第二张标签：为了“逐位中性”，我们主动放弃了 Inductor 算子融合

仔细看上面吞吐表的第一行：`NativeRead` 的方法体里只有一句 `return kv.attend(q)[0]`，什么花活都没干，为什么它的吞吐是 **99%**（5,111 对 5,165 tok/s），差了那 1% 左右？

这 1% 不是因为多调了一次 Python 函数（函数早就被录进 CUDA Graph 里了），而是我们为了守住 pLLM 的一条铁律——**“中性设置必须与原生 vLLM 逐位相同（Bitwise Identical）”**——而主动交出的一笔买路钱。

当我们要求 `NativeRead`（或者示例 10 里只导出统计量、不改动输出的 `RecentMass` 监控负载）在贪心解码和固定种子采样下，生成的每一个 logit、每一个 token ID 都必须与原生 vLLM **连最后一个比特都分毫不差**时，我们在 7B 上撞见了一件怪事：明明我们已经核对过，FlashAttention-2 在 `AttentionHook` 里读出来的张量与原生注意力逐元素完全相等，可跑完 28 层出来，最后的 logits 还是在小数点后好几位上出现了极其微小的漂移！

凶手不在注意力层，而在 PyTorch 的 **Inductor 编译器**。

默认情况下，原生 vLLM 在用 `torch.compile` 录制模型图时，会让 Inductor 把残差加法（`add`）和紧随其后的归一化（`RMSNorm`）融合成一个 Triton 内核。而一旦我们在模型里挂上了自定义注意力钩子，图的拓扑结构发生了细微变化，Inductor 对挂钩前后的图做出的融合决策或归约顺序就可能产生一丁点不同。在浮点数世界里，先加再归约和边加边归约，舍入误差在 28 层里层层放大，足以让某个临界 token 的最高分易主。

面对“要 Inductor 融合带来的 1%–3% 吞吐”还是“要绝对可靠的逐位中性基准”，我们毫不犹豫地选了后者。在 `pllm/backends/vllm/overlay_graph.py` 里，只要负载挂了包括自定义注意力在内的模型钩子，我们就把 vLLM 的编译后端切到 `"eager"`：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/overlay_graph.py
return {
    "additional_config": {_BUDGET: config.execution.cuda_graph_pool_budget_bytes or 0},
    "compilation_config": CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        # Attention hooks must keep the same source kernels as the vanilla neutral reference.
        # Inductor fusion can change model rounding even when the FA2 reads are bitwise identical.
        backend="eager",
        # ……
    ),
}
```

请千万别被 `"eager"` 这个名字骗了——在这里它不是“关闭 CUDA Graph 退回慢吞吞的单步解释执行”，它的意思是：**在录制 CUDA Graph 时，直接把模型原本的手写 C++/CUDA 源算子录进图里，不要让 Inductor 去重新生成融合内核。**

关掉 Inductor 融合要付多少钱？在 0.5B 上，单单把原生模型图从 Inductor 换成源算子录图，吞吐会慢 **1.9%**（13,158 对 12,911 tok/s）；在 7B 上挂载残差或注意力改写时，总开销大约在 **1%–3%** 之间，这笔“源算子税”就算在其中（7B 上没有单独测过；我们在《图内挂载点》里拆解过这笔账）。但花掉这几个百分点，换来的是研究者最珍贵的定心丸：当你在实验里看到生成质量变好或变坏时，你永远不需要怀疑是不是底层框架在挂钩子时偷偷改变了浮点舍入——因为 `testing.check_on_gpu` 里的 `neutral` 检查已经向你证明了，只要把你的算法参数拨回中性，它的输出就和原生 vLLM 逐位重合。

---

## 读懂宿主的账本

沿着同样的物理逻辑，你也能一眼看穿 `@pllm.custom_attention` 周围那些看似零碎的使用边界：
*   **为什么要强制关闭前缀缓存（`enable_prefix_caching=False`）？** 因为前缀缓存的前提是“相同的 token 序列必定产生完全相同的 KV”。一旦你在某几层换了窗口布局或改写了注意力读法，后续层的 KV 就变了味，再去复用全局哈希表里的旧 KV 块就是张冠李戴。
*   **为什么要求单卡（`tensor_parallel_size=1`）、FlashAttention-2 后端与非量化 KV（`kv_cache_dtype="auto"`）？** 因为我们是从 FA2 的元数据里借出块表和序列长度来统一驱动各条读路径的，而跨卡切分头部的同步、或是对 FP8 打包字节做任意位置的稀疏 gather，都需要额外的通信或反量化内核。
*   **为什么静态 KV 布局要求 MRv2 runner，且不能和 `@pllm.custom_forward` 或 `pllm.PositionState` 混用？** 因为静态布局会把模型的各层拆成多个各自管理的 KV 组，而自定义层循环（比如跳层或反复执行某几层）要求全模型共享单一的全局 KV 组；同理，跨层共享的 `source=k` 只能往下借用已经算过的浅层，如果你想在层循环里偷看更深层的 KV，那属于《自定义层循环》里 `kv="peek"` 的领地。至于像 DeepSeek 稀疏注意力（DSA）那样直接复用模型自带硬件索引器的 `topk` 结果，由于底层内核绑定了 Hopper 架构（SM90+），在 RTX 4090（SM89）和 A100（SM80）上自然也就无从谈起。

回过头看，这一整篇关于注意力与 KV 布局的设计里，pLLM 自己写的物理块分配与回收代码，一共是**零行**。

每当有人问我，在一个像 vLLM 这样高速演进、内部充满精密假设的工业级引擎之上做扩展层，最核心的诀窍是什么？我的回答总是：**先别急着写代码，先去读懂宿主系统的账本。**

每一个成熟的底层系统，都有一套雷打不动的记账方式——对数据库来说是 WAL 日志与事务 ID，对操作系统内核来说是页表与引用计数，而对 vLLM v1 来说，就是那个单标量的 `num_computed_tokens` 和那张只追加的 `block_table`。当你提出一个新功能时，先问问自己：这个功能能用宿主现有的账本科目记下来吗？
*   如果能（哪怕需要像 `SinkSpec` 和 `sink_first` 那样做一点巧妙的科目映射），那就把它彻底翻译成宿主的原语，让宿主去替你扛下并发、抢占和内存回收的全部脏活；
*   如果不能（比如像 SnapKV 那样非要在连续的账本中间撕掉几页），那就把三条记账不变量摊在桌面上，诚实地告诉用户为什么物理定律不允许，然后指明那条真正走得通的路。

在系统设计里，克制地使用宿主的语言，远比在旁边另立一本新账要走得远得多。
