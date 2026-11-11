# 稀疏注意力在 vLLM 里省不下显存（以及为什么调 `window_size` 反而慢了 5 倍）

> 2026-11-10 · pLLM 提交 `cff7c196` · 测量环境：除文中另注外为单张 RTX 4090（24 GiB）、vLLM 0.32.0、Qwen2.5-7B-Instruct，8 条 prompt × 16 个样本、生成 256 token。
> 这是一篇有日期的文章，不随代码更新；当前的用法以[用户指南](../../README.md#用户指南)为准，当前的实测数字以[性能与显存](../performance.md)为准。

如果你把过去两年顶会上的长上下文推理论文摊在桌子上，相当一部分在打注意力（Attention）和 KV Cache 的主意。

这些论文的逻辑听起来无懈可击：在自回归生成时，KV Cache 随着序列长度线性膨胀，但真正分到注意力权重的往往只有最近的局部窗口和少数关键历史位置。既然绝大多数历史 token 都没人看，那我们少读一点、或者把不重要的 token 从缓存里删掉，不就能既省带宽又省显存了吗？

在单卡的 Hugging Face 脚本里验证这些想法非常轻松。`past_key_values` 只是一串归你独占的 PyTorch 张量，你想传个掩码、切个片，还是沿着序列维删掉几列，全凭一行索引操作。

但当你带着跑通的原型走进生产环境，试图把这些“稀疏注意力”或“KV 压缩”算法放进 vLLM 时，你会迎头撞上两件极其反直觉的事：

第一，很多在论文里号称“只看 10% 的历史 token”的稀疏算法，放进 vLLM 后**连一兆字节的物理显存都省不下来**，单卡能接纳的最大并发请求数纹丝不动；而如果你想更进一步，像 SnapKV、H2O 或 PyramidKV 那样根据运行时注意力分数把低分 token 的显存块释放掉，你会发现除非把 vLLM 的调度器和块管理器拆骨重装，否则这件事在架构上根本走不通。

第二，当你退而求其次，采用真正能回收物理块的静态滑动窗口（Sliding Window），并老老实实调用底层 FlashAttention-2（FA2）原生提供的 `window_size` 接口去读缓存时，**少读了 98% 的历史数据，单次注意力读取反而比读全量历史慢了 5.5 倍**，直接把整机吞吐砸穿了两成。

表面上看，这像是某几个 API 没调对。但如果你顺着 vLLM 的源码往下挖两层，就会发现这两件事背后站着同一条系统工程铁律：在一个成熟的底层引擎之上做定制，成败从来不取决于你的算法在纸面上省了多少乘加运算，而取决于你有没有看穿宿主系统的**记账不变量（Accounting Invariants）**与**内核执行管线（Execution Pipeline）**。

我们先从显存账本算起。

---

## 一、 112 MiB 的算术：“少读”为什么不等于“少存”？

在单请求的研究脚本里，人们习惯盯着“这一步少读了多少字节”看。但在多租户的在线推理服务里，决定一台服务器能同时接多少单的头号指标是另一个：**一个请求在最坏情况下，最多会占掉多少已分配的物理 KV 显存？**

拿 `Qwen/Qwen2.5-7B-Instruct`（BF16）来算一笔最基础的账。这个模型一共有 28 层，采用分组查询注意力（GQA）：每层有 28 个 Query 头，但只有 4 个 KV 头（每 7 个 Query 头共享 1 组 K/V），每个头的维度是 128。假设我们的服务配置允许每个请求最长跑到 `max_model_len = 2048` 个 token。那么这一个请求在 28 层里存满 K 和 V 两份张量，它的物理显存准入上限是：

$$2048 \times 28 \times 2 \times 4 \times 128 \times 2\text{ Bytes} = 112.0\text{ MiB}$$

在 pLLM 中，如果你在 CPU 上调用 `pllm.explain` 去检查一个不做任何裁剪的 7B 负载，它报出来的单请求 KV 容量上限分毫不差，正是 **112.0 MiB**（在 24 层、2 个 KV 头、头维 64 的 `Qwen2.5-0.5B` 上，这个数字是 **24.0 MiB**）。

现在，假设你想实现类似 Quest（Tang et al., 2024）的页级稀疏注意力：既然 vLLM 的 KV Cache 是按页（默认每块 `block_size = 16` 个 token）存放的，我们可以根据当前 Query 向量对每个历史页快速打一个粗略的分数，然后让每个注意力头只挑出得分最高的 8 页（即 $8 \times 16 = 128$ 个 token）来读取并计算精确注意力，其余历史页全部跳过。

从 2,048 个位置缩减到 128 个位置，表面上看只读了原来的 $1/16$（6.25%）。但在真实系统里，你会连续收到两份意外的账单。

第一份账单来自 **GQA 的写时共享与读时分化**。在 `Qwen2.5-7B` 里，同组的 7 个 Query 头在写入时共享同一个 KV 头。当做全量因果注意力时，7 个 Query 头看的都是从位置 $0$ 到当前位置的完整区间，同一个 KV 页从显存读进芯片一次，就能同时喂饱 7 个 Query 头。可一旦你允许每个 Query 头按自己的打分独立挑选 Top-8 页，最坏情况下，同组的 7 个 Query 头挑中的是完全不重合的 8 页！那个共享的 KV 头就得被迫为它们各自去显存里搬运不同的页。用 `pllm.explain` 静态核算第 2 层每个 token 的最大 KV 读取量：全量读取（`NativeRead`）是 `4,194,304` 字节，而每头只读 8 页的 `QuestRead` 依然高达 `1,835,008` 字节（在 0.5B 上则是从 `1,048,576` 字节降到 `458,752` 字节）——逻辑位置砍到了 6.25%，物理读取上界却还剩 **44%**。

第二份账单更致命：**这个请求占用的物理 KV 显存，一分钱都没少，还是 112.0 MiB。**

道理一捅就破：你在第 500 步生成时觉得第 3 页得分低、跳过去没读，并不代表第 501 步生成的新 Query 也不看第 3 页。只要算法保留了“未来任意一步都可以从全部历史页里重新打分挑选”的权利，前面算出来的所有历史 KV 页就必须完好无损地留在显存里待命。当 vLLM 的调度器做准入控制（Admission Control）时，依然必须按 2,048 个 token 的全额预算为这个请求预留 112.0 MiB 的物理块。

更雪上加霜的是吞吐。在单张 NVIDIA RTX 4090（24 GiB GDDR6X 显存）、vLLM 0.32.0（开启 CUDA Graph 与 FlashInfer 采样器）下，以 128 条并发序列（8 个 prompt × `n=16`）、每条生成 256 个 token（上下文上限 512、每步最多 128 个 token）的基准测试 `Qwen2.5-7B-Instruct`，原生 vLLM 的吞吐是 `4,855 tok/s`；而用 PyTorch 算子去捞取各页计算极值打分、再按页 gather 的 `QuestRead`，并发吞吐直接跌到了 **`649 tok/s`（仅剩原生的 13%）**。在没有专门为打分与稀疏收集手写底层融合 CUDA 内核的情况下，每层每步在 PyTorch 里动态打分和搬运索引的开销，远远超过了少读那几页省下的显存带宽。

这就划出了推理引擎里的第一条硬分界线：**动态选择读取（Selection）省的是逻辑读取量，省不下已分配的物理显存。** 如果你的目标是降低单请求显存占用、让同一张显卡多接一倍的并发请求，你不能只“跳过不读”，你必须让那些不需要的物理内存块**真正退回到块管理器的空闲池（Free Pool）里去**。

那么问题自然就来了：既然只看不删省不下显存，我们能不能在生成过程中算出每个历史 token 的重要性分数，把低分 token 占用的物理块当场释放掉？

---

## 二、 vLLM 的三本死账：为什么按分数动态驱逐走不通？

在 Hugging Face 的单机脚本（或者像 `kvpress` 这类构建在 Hugging Face 之上的压缩库）里，实现 SnapKV、H2O 或 PyramidKV 非常顺手：在每次预填充或解码步之后，沿着序列维度把高分 token 的 K 和 V `gather` 出来拼成一张变短的新张量，把旧张量扔掉就行了。

但当你打开 vLLM v1 的源码，试图寻找一个“释放请求中间某几个 token 对应的 KV 空间”的接口时，你会发现整个引擎的地基都在排斥你。

这不是因为 vLLM 的作者忘了写一个 `delete_tokens()` 函数，而是因为 PagedAttention 与连续批处理之所以能跑得快，完全建立在 CPU 调度器（Scheduler）与 GPU Worker 之间立下的**三条硬性记账不变量**之上。

假设块大小 `block_size = 16`，某个请求刚刚算完了前 40 个 token，正在进入第 41 个 token（逻辑位置 `pos = 40`）的解码步。掀开 vLLM v1 的引擎盖，你会看到这三条铁律是如何锁死一切动态挖洞企图的：

### 铁律一：进度是一个全模型共享的标量整数

如果你去翻 vLLM 调度器里维护的请求对象，你会发现它根本没有维护什么“第 $l$ 层、第 $h$ 个头当前保留了哪些位置”的列表。对于每个请求的计算进度，调度器只记了孤零零的一个标量整数：

$$\texttt{num\_computed\_tokens} = 40$$

这个单标量假设意味着：**如果第 0 层算到了第 40 个 token，那么第 27 层也一定不多不少存着 40 个 token 的进度。** 如果你想实现像 PyramidKV 那样“浅层保留较多 token、深层只保留较少 token”的金字塔形跨层异构压缩，调度器的账本上连填这个数字的格子都没有。

### 铁律二：逻辑位置、注意力长度与物理块表下标被同一个整数绑死

当第 41 个 token 进入 GPU Worker 时，Worker 是怎么知道该给它乘上多少度的 RoPE 旋转位置编码、该让注意力内核扫描多长的历史、又该把新算出来的 K 和 V 写进哪块物理显存的？

Worker 不会去查任何位置映射表。它直接拿 `num_computed_tokens`（关于入口处这三个基础进度整数如何决定整批行布局，详见[第 01 篇：你其实不需要读 `attn_metadata`](01-attn-metadata.md)），靠最朴素的整数算术同时推导出三件事：
1. 本步注意力的扫描长度：`seq_len = num_computed_tokens + query_len`；
2. 新 token 的 RoPE 逻辑位置：`pos = num_computed_tokens + i`；
3. 新 token 写入 KV 池的物理槽位（`slot_mapping`）：直接去该请求的块表（`block_table`）里查第 `pos // 16` 格拿到物理块号，再算 `pos % 16` 拿到块内偏移。

```text
Logical Pos:    0 .. 15        16 .. 31        32 .. 39 [40]  <-- pos = 40 (new token)
              ┌──────────────┬──────────────┬──────────────┐
Block Table:  │ Physical #7  │ Physical #2  │ Physical #9  │
              └──────┬───────┴──────┬───────┴──────┬───────┘
                     │              │              └──► write slot = 9 * 16 + (40 % 16)
                     ▼              ▼
  (A) Punch a hole (free Physical #2, keep table length):
      FlashAttention scans [0, 41) ──► dereferences Physical #2 (now owned by another user!)

  (B) Compact table (shift Physical #9 left to index 1, set computed = 24):
      New token gets RoPE pos = 24 instead of 40!

  (C) How vLLM's own SlidingWindow / R-SWA reclaims blocks safely:
              ┌──────────────┬──────────────┬──────────────┐
Block Table:  │  null_block  │ Physical #2  │ Physical #9  │  (table length unchanged!)
              └──────────────┴──────────────┴──────────────┘
              └── Static window mask guarantees kernel NEVER reads [0..15]
```

看清这张图里的死结了吗？**RoPE 位置、序列读取长度和块表数组下标，被绑死在了同一根数轴上。**

假设你在跑完一步后，发现中间第 16 到 31 号 token（对应 `Physical #2`）注意力得分很低，想把它驱逐回空闲池。你面前只有两条路，而两条路都是悬崖：
* **走路径 A（原位释放物理块，不改块表长度）**：你把 `Physical #2` 还给了空闲池，池子在下一毫秒把它分给了另一个刚进来的用户李四。到了下一步，由于标准 FlashAttention 默认扫描 `[0, seq_len)` 的连续区间，内核会毫无防备地顺着块表第 1 格冲进 `Physical #2`——把李四的私密对话当成张三的历史上下文读进来！
* **走路径 B（释放物理块，并把块表左移紧缩）**：你把后面的 `Physical #9` 往前挪到第 1 格，把 `num_computed_tokens` 从 40 改成 24。且不说原先存着的第 32..39 号 token 在写入前就已经烘入了位置 32..39 的 RoPE 角度；单看下一步新进来的第 41 个 token，Worker 会按 `pos = 24` 给它施加旋转位置编码——一个原本排在第 40 位的字，硬生生戴上了第 24 位的旋转角。

### 铁律三：Worker 端的块表是一本“只追加日志（Append-only Log）”

就算你愿意去魔改位置编码，还有第三道物理屏障挡在面前：在 vLLM 的架构里，CPU 调度器每一步发给 GPU Worker 的运行中请求更新（`CachedRequestData`），**只有一个叫 `new_block_ids` 的字段，而且 Worker 端在更新本地块表时写死了 `overwrite=False`。**

换句话说，调度器与 Worker 之间的通信协议里，只有“在块表末尾追加几块新页”这一种操作，根本不存在“删掉中间第 $k$ 块”或“把整张块表左移重排”的通道。更何况，每一步根据 GPU 上刚算出来的注意力分数动态决定删哪一块、再通知 CPU 改写块表，意味着每一步都要发生 GPU 到 CPU 的主机同步，直接破坏了 CUDA Graph 静态回放赖以生存的固定内存拓扑（见[第 02 篇：What Color is Your Tensor?](02-tensor-colors.md)）。

如果把所有试图在不拆引擎的前提下“绕”过这三条铁律的方案放在源码显微镜下审视，你会看到它们是如何逐一破产的：

| 试图绕行的方案 | 撞上了哪条底层铁律而破产 |
| :--- | :--- |
| **1. 写一个块粒度的自定义 `KVCacheSpec`** | Token 级的注意力分数对不齐 16 个 token 一块的物理页边界；更致命的是触犯**铁律三**——Worker 块表只追加不更新，释放后的块留在块表里会被标准注意力内核当场解引用，读出脏数据。 |
| **2. 每隔 $N$ 步挑出高分 token 重新跑一遍 Prefill** | 被剔除的低分 token 在原本的 Prefill 阶段已经通过浅层注意力把信息注入了保留 token 的深层表示里；如果只拿保留下来的 token 重新 Prefill，浅层上下文彻底丢失，数学上已不再是原算法，且重算开销极高。 |
| **3. 借道 KV Connector 导出、释放再重载压缩后的 KV** | 触犯**铁律一**——受制于单标量 `num_computed_tokens`，所有层、所有头被迫保留完全相同的长度，无法支持 PyramidKV 这类逐层不同预算的算法，且必须在调度器里伪造请求生命周期。 |
| **4. 在 GPU 上原地 gather 压缩缓存并截断块表尾部** | 同样触犯**铁律一**（无法表达跨层异构长度，除非按全模型最大保留长度补齐，白白浪费显存），且必须同时侵入修改调度器、块管理器和 Worker 块表同步协议三处核心代码。 |

顺着同样的物理账再算一步，另一个在论文里很常见的思路——**压缩远端记忆（如 Activation Beacon、LastKV）**，即近处保留原始 KV、把远处的历史每几个 token 浓缩成一个“摘要条目”——在分页引擎里同样会撞墙：
1. **分页填充损耗（Page Padding Waste）**：vLLM 的页大小是固定的。假设把每 4 个远端 token 压成 1 个宽 256 的 BF16 条目，一页里装 4 个压缩条目，有效内容只有 $2,048\text{ Bytes}$，但它依然要占掉一个完整的 $8,192\text{ Bytes}$ 物理页，块对齐让远端显存一点没省下来；
2. **分块预填充（Chunked Prefill）的非确定性**：如果远端摘要依赖较深层的隐藏状态，那么在当前步某个位置的摘要算没算出来，完全取决于调度器在这一步把长 prompt 切成了多大的块，导致同一段输入在服务器空闲和繁忙时算出不同的结果；
3. **双段注意力的延迟倒挂**：解码时先对近处窗口跑一次注意力，再对远端摘要跑一次注意力，最后合并两段输出与 `log-sum-exp`（LSE）——在微基准测试中，这一套近段+远段+合并的单层耗时是 **`30.31 µs`**，而直接让原生内核一口气读完全量历史只要 **`11.88 µs`**。

把这本账彻底算清之后，结论非常干脆：在不推翻 vLLM v1 核心记账假设的前提下，不存在任何干净的扩展方案来实现释放物理显存的、基于分数的动态 KV 驱逐。

---

## 三、 零行分配代码的“伪装术”，以及 `sink_first` 的分组陷阱

慢着——既然 Worker 端的块表只能追加、不能缩短，那上面那张图的 **(C)** 行里，vLLM 自己内置的滑动窗口（`SlidingWindowManager`）和保留前缀窗口（`RSWAManager`，即 Prompt + 滑动窗口）又是怎么把显存块安全地还回空闲池的？

答案就在那个不起眼的 **`null_block`** 上。

当滑动窗口向右推进、最老的 `Physical #7` 彻底滑出窗口边界时，vLLM 的原生管理器确实会把 `Physical #7` 回收到空闲池供其他请求复用；但为了不破坏“逻辑位置 = 块表下标 × 16”的第二条铁律，也不去缩短 Worker 的块表，管理器在原位置填进了一个全局预留的 0 号哑块 `null_block`。块表的长度纹丝不动，每个组的逻辑页数依然完全一致！

为什么把前面换成 `null_block` 不会读到脏数据？**因为静态窗口的几何形状是编译期已知且规整的。** 窗口注意力内核自带的边界条件在数学上保证了它只去解引用末尾窗口内的物理块（或者 R-SWA 下的完整 prompt 块加末尾窗口块），绝对不会去碰前面那些已经被替换成 `null_block` 的格子。

一旦看懂了宿主的这套记账法则，在 vLLM 里实现自定义 KV 布局的最佳策略就浮出了水面：**一行新的内存分配与回收代码都不要写。赶在 vLLM 启动显存探测和初始化块管理器之前，走进去把每一层的属性“伪装”成 vLLM 原本就认识的那三种原生层。**

```text
1. keep="all" (Full Attention — all blocks stay live):
   [ Block 0 ][ Block 1 ][ Block 2 ][ Block 3 ][ Block 4 ]

2. keep="window", window=256 (Sliding Window — prefix blocks recycled to null):
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

在 pLLM 的 `pllm/backends/vllm/kv_layout.py` 里，这三种布局的挂载核心只有十来行代码：

```python
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

* 当某层声明 `keep="none", source=1`（跨层共享，例如奇数层不存 KV、直接借用紧邻的下面一层）时，我们只需把该注意力层的 `kv_sharing_target_layer_name` 指向目标层。vLLM 原本为 Gemma 3n 等模型准备的跨层共享机制会立刻接管，该层连一个字节的 KV 都不会分配；
* 当声明 `keep="window"` 时，挂载为 `SlidingWindowSpec`；
* 当声明 `keep="window", sink=True` 时，将 `get_kv_cache_spec` 替换为返回 `RSWASpec`（保留整个用户 prompt 加最近窗口；注意这与 StreamingLLM 只保留开头固定几个 token 略有不同，它是把整个 prompt 完整钉在缓存里，只回收长生成过程的中间段）。

因为这次“伪装”发生在 vLLM 探测可用显存和构建 `KVCacheCoordinator` 之前，随后的准入控制（`layout_capacity` 直接调用上游 `SlidingWindowSpec.max_admission_blocks_per_request`）、运行期的物理块追加、`null_block` 置换回收、乃至请求被抢占时的引用计数清理，100% 由 vLLM 自己的原生管理器完成。

不过，跟大型底层系统的原生机制打交道，总少不了和它的隐式潜规则过招。在落地 `sink=True`（R-SWA）时，我们就撞上了一个极典型的例子。

如果在一个原本全是全量注意力（`FullAttentionSpec`）的模型里，只把其中几层改成 `RSWASpec`，你会惊讶地发现：运行时这几层根本没有按“Prompt + 窗口”去回收中间的块，而是跟普通的全量注意力层混在了同一个管理组里，把所有历史块全留着！

顺着 vLLM 给各层 `KVCacheSpec` 分组的代码查下去，原因让人忍俊不禁：vLLM 在决定“哪些层可以合并成一个组共享同一套块表管理”时，会取出字典里的**第一个** `spec`，调用它的 `is_uniform_with_collection(specs)` 去问：“你们后面这些层跟我兼容吗？”

在面向对象设计里，`RSWASpec` 恰好是 `FullAttentionSpec` 的子类！当排在第 0 层的 `FullAttentionSpec` 第一个被询问时，它用 `isinstance(spec, FullAttentionSpec)` 扫了一眼后面的 `RSWASpec`，看头数、头维、数据类型全对得上，又是自己的子类，大度地返回了 `True`——于是，挑剔的 R-SWA 层就这样被宽容的全量注意力父类给“吞并”了。

看穿了这个潜规则，解法在 `pllm/backends/vllm/kv_layout.py` 里只需要十行代码：

```python
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

先派生一个 `SinkSpec`，规定只有全员都是窗口相同的 `RSWASpec` 时才返回 `True`；再用 `sink_first` 给字典排个序，把所有 `RSWASpec` 挪到字典最前头，确保 vLLM 第一个去问的永远是挑剔的子类而不是宽容的父类。不改上游一行代码，R-SWA 层立刻独立成组，由原生的 `RSWAManager` 把滑出窗口的物理页精准收回池中。

至于注意力的方法体内部，则只负责“怎么读”。钩子装在 vLLM 不透明注意力算子的内部、原生 KV 缓存写入刚刚完成的那一刻，收到的是只读借用视图 `PagedKV`（对于需要按位置做掩码 gather 的 `selected_attention`，代码在解引用前会用 `valid` 掩码把越界或已滑出窗口的块表索引强制钳回 `0`，绝不触碰任何已回收的物理页）：

| 负载写法（`max_model_len = 2048`） | 每请求 KV 上限（7B） | 每请求 KV 上限（0.5B） | 2048-token 并发容量（0.5B） | 实测吞吐（7B，相对原生 4,855 tok/s） |
| :--- | ---: | ---: | ---: | ---: |
| **NativeRead**（全量保存，原生读） | 112.0 MiB | 24.0 MiB | 358.1 个 | **99%**（4,817 tok/s） |
| **LocalGlobal**（偶数层窗口 256） | **70.4 MiB** | **16.6 MiB** | **517.6 个**（1.45×） | **99%**（4,804 tok/s） |
| **CrossLayerShared**（奇数层借用下面一层） | **56.0 MiB** | **12.0 MiB** | **710.4 个**（1.98×） | **99%**（4,798 tok/s） |
| **PromptPlusWindow**（Prompt + 窗口 256） | 112.0 MiB | 24.0 MiB | 358.1 个 | — |
| **QuestRead**（全量保存，每步挑 8 页读） | 112.0 MiB | 24.0 MiB | 与 NativeRead 相同 | **13%**（649 tok/s） |

*(注：KV 上限按 `max_model_len = 2048` 由 `pllm.explain` 算出；吞吐测试环境为单张 RTX 4090、vLLM 0.32.0、128 并发序列生成 256 token，因为序列只有约 270 个 token，上下文上限设为 512、每步最多 128 个 token；列出 0.5B 的并发容量是因为小模型权重占显存小，能更直观地反映 KV 池扩容带来的并发倍数。全量读与跨层共享的模型图走 Inductor，窗口布局保留源算子录图，见[第 04 篇：How to Lose 61,440 Kernels in 10 Lines](04-compiled-model-hooks.md)。)*

看这张表，静态布局与动态只读筛选的物理分野一清二楚：`LocalGlobal` 和 `CrossLayerShared` 把 7B 单请求的 KV 准入上限实打实地砍到了 **70.4 MiB** 和 **56.0 MiB**，在 0.5B 上让单卡可容纳的 2048-token 并发请求数从 **358 个** 直接跳到了 **518 个** 和 **710 个**（而 `PromptPlusWindow` 因准入必须防备整条 2,048 都是 prompt 的最坏情况，准入上限不变，但在 0.5B 上跑 prompt 8 + 生成 112 的长生成时实测省下了 6.25% 的运行期物理页）。

当然，改 KV 布局改的是模型的数学函数。在一篇 1,500 个 token 的博物馆长文中间埋入密码 `"7361"` 做大海捞针测试时，未经滑动窗口或跨层共享微调的原版 `Qwen2.5-0.5B`，在 `PromptPlusWindow` 和 `QuestRead(pages=8)` 下能准确答出 `'7361'`，但在直接砍掉偶数层远端记忆的 `LocalGlobal` 下答出了错误的 `'3'`，在硬抽掉奇数层 KV 的 `CrossLayerShared` 下更是直接吐出乱码。系统底座保证的是物理块安全回收与高速执行，而模型认不认识这种稀疏结构，永远要靠评测集来检验。

---

## 四、 为什么调标准 `window_size` 接口，读取反而慢了 5.5 倍？

现在，存储层面的物理块已经由原生管理器替我们省下来了。接下来是本文的后半个谜题：当注意力层真的去读这个滑动窗口时，发生了什么？

上面表格里 `LocalGlobal` 在 7B 上跑出了 99% 的原生吞吐。但要看清底层注意力算子本身的真实效率，7B 模型其实是一面模糊的镜子——因为 7B 模型每步 decode 要花二十多毫秒去搬运庞大的 MLP 权重，单层注意力那几十微秒的变化很容易被稀释掉。

如果我们把镜头切到单步耗时只有约 **3 ms** 的 `Qwen2.5-0.5B-Instruct`（14 个 Query 头 / 2 个 KV 头，头维 64，同样在单张 RTX 4090 上跑 128 并发序列 × 256 token，vLLM 0.27.1，`max_model_len=2048`、每步最多 256 个 token）这台“显微镜”下，一个惊人的性能悬崖就会暴露出来。

在实现滑动窗口读取时，最天经地义、最符合“不要重复造轮子”原则的做法是什么？当然是直接调用上游 FlashAttention-2 的标准接口 `flash_attn_varlen_func`，把 `window_size=[window - 1, 0]` 传进去！

但当你真的这么写，跑出来的数字会让你怀疑人生：
* 在一个仅把 24 层中的 2 层设为窗口、且窗口大小只有 4 个 token 的测试里，本应比全量注意力更快的窗口配置，整机吞吐却掉到了全量注意力的 **89.3%**（vLLM 0.27.1）和 **90.7%**（vLLM 0.32.0）；
* 在偶数层窗口设为 256 的 `LocalGlobal` 上，整机吞吐更是从原生的 `39,120 tok/s` 暴跌到 **`30,846 tok/s`（78.8%）**，平白丢掉了超过 21% 的吞吐！

为了搞清楚为什么“少读历史”反而导致吞吐崩盘，我们把这一层注意力单独拎出来，在 128 行并发、历史长度 256 的条件下跑 1,000 次 CUDA Graph 回放微基准：
* 让 FA2 读全部 256 个历史 token，单次调用耗时 **`16.56 µs`**；
* 给同一个 FA2 函数传进 `window_size=(3, 0)`，让它只读最近 4 个 token，单次耗时却暴涨到了 **`91.47 µs`**！

少读了 98% 的数据，耗时反而慢了 **5.5 倍**（在 `LocalGlobal` 的 3,024 次稳态窗口读中，单次 FA2 窗口读平均也高达 **`86.43 µs`**）。

为什么会这样？拿信封背面算一笔显存带宽的 Roofline 硬账，真相就浮出水面了。

在自回归 decode 阶段，每条序列每次只进 1 个新 token，算力完全过剩，注意力层的唯一瓶颈是把历史 K 和 V 从显卡的 GDDR6X 显存搬进芯片的带宽。而在 `Qwen2.5-0.5B`（14 个 Query 头、2 个 KV 头）和 `7B`（28 个 Query 头、4 个 KV 头）中，GQA 的比例都是 **7 : 1**——每 7 个 Query 头共享完全相同的一组 K 和 V。

这意味着在理想情况下，有效访存载荷应该只按 **2 个 KV 头** 来算。对于 128 行并发、历史长度 256、头维 64、BF16（2 字节），一步需要搬运的 K/V 字节数是：

$$128 \times 256 \times 2\text{ (K与V)} \times 2\text{ (KV头)} \times 64 \times 2\text{ Bytes} = 16\text{ MiB}$$

在 RTX 4090 标称 `1,008 GB/s` 的显存带宽下，把这 16 MiB 数据完整搬运一遍的物理极限时间（Roofline）大约是 **`16.6 µs`**（历史长度为 128 时则是 8 MiB、约 `8.3 µs`）。你看，FA2 全量读取测出来的 `16.56 µs`，几乎严丝合缝地贴在硬件的物理极限线上！

FA2 全量读取是怎么贴住物理极限的？靠的是 **GQA 打包（Packed GQA）**：内核把共享同一组 KV 的 7 个 Query 头打包进同一个线程块（CTA）里，从显存把 K 和 V 读进片上内存一次，7 个 Query 头一起跟它做点积。

但当我们打开 profiler 检查这个形状下 FA2 开启 `window_size` 时的内核网格（Grid）参数，病根水落石出：

```text
1. Packed GQA Decode (Grid organized by 2 KV Heads — 1x KV traffic, ~16.6 µs):
   ┌─────────────────────────────┐
   │  KV Head 0 Pages (in VRAM)  │ ──(Read Once)──► [ CTA: Q0, Q1, Q2, Q3, Q4, Q5, Q6 ]
   └─────────────────────────────┘

2. Unpacked FA2 Window Decode (Grid organized by 14 Query Heads — up to 7x traffic, ~86.4 µs):
   ┌─────────────────────────────┐    ┌──► [ CTA 0: Q0 ]
   │                             │────┼──► [ CTA 1: Q1 ]
   │  KV Head 0 Pages (in VRAM)  │────┼──►   ...          (7 independent CTAs fetching
   │                             │────┴──► [ CTA 6: Q6 ]   the exact same K/V pages!)
   └─────────────────────────────┘
```

在我们测试的这个形状下，一旦开启滑动窗口，FA2 的 decode 路径就失去了 GQA 打包：它的启动网格从按 **2 个 KV 头** 组织，退化成了按 **14 个 Query 头** 组织！原本只需要读 1 次的 K/V 页，现在被 7 个各自为政的线程块分别去请求一遍（能有多少被 L2 Cache 兜住全看运气），再加上窗口边界判定的开销，硬生生把一个 `16.6 µs` 的内核拖成了 `86.4 µs`。

---

## 五、 调裸函数是伪复用，走上游的执行管线才是真复用

面对这个慢了 5 倍的标准接口，我们该怎么办？

如果自己手写一个 Triton 单段局部读 kernel 再配一个 LSE 收尾 kernel，单次读取确实只要 `16.23 µs`（整机吞吐 `98.4%`）。但每多维护一个私有注意力内核，就意味着要自己背负跨硬件架构适配和数值基准测试的包袱。

那如果不自己写，而是换用 vLLM 集成的另一个上游库 **FlashInfer** 呢？在 Ampere 和 Ada 架构（SM80/SM89，如 A100 和 RTX 4090）上，FlashInfer 的 tensor-core decode 内核采用 `16×128` 的分块与 4 个 warp，即使在开启滑动窗口时，依然保持着按 KV 头组织的打包网格（在我们的测试图里，grid 是规整的 `[183, 1, 2]`）。

但如果你试图在自己的代码里直接 `import flashinfer` 去调它的底层函数，你会立刻掉进另一个泥潭：FlashInfer 不是 FA2 那种无状态的纯函数，它是一个**有状态的两阶段管线**。在每次发射 GPU 内核之前，它必须在 CPU 上先跑一次 `plan`（根据当前批次每条序列的长度计算 split-KV 划分、生成页索引与工作区偏移）；而且为了让内核能被录进 CUDA Graph 反复回放，`plan` 写入的页索引暂存区和归约缓冲区必须拥有固定不变的显存地址，在每次回放前由主机原地更新。

看清这里的两难了吗？如果你把上游当成一个静态的“函数库”，想越过宿主直接去调一个裸函数，你就得为了这一个函数自己重写半套页索引暂存、split-KV 规划和 CUDA Graph 缓冲区管理！

真正的出路不是“自己调 FlashInfer”，而是**像 vLLM 自己那样去调 FlashInfer**。

vLLM 自己的 `vllm.v1.attention.backends.flashinfer` 模块里，本来就有一整套现成的 `FlashInferMetadataBuilder` 和 `FlashInferImpl`，页索引、split 规划和图缓冲区的生命周期早就安排得妥妥当当。在 `pllm/backends/vllm/flashinfer_attention.py` 里，我们只需写一个十来行的 `WindowMetadataBuilder` 继承它：

```python
class WindowMetadataBuilder(FlashInferMetadataBuilder):
    """Let both upstream builders consume the same native page mappings."""

    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert vllm_config.speculative_config is None
        self.flash = FlashAttentionMetadataBuilder(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> WindowMetadata:
        infer = super().build(common_prefix_len, common_attn_metadata, fast_build)
        flash = self.flash.build(common_prefix_len, common_attn_metadata, fast_build)
        return WindowMetadata(**vars(infer), flash=flash)
```

当某层在 Ampere/Ada 架构上被声明为滑动窗口存储时，`install_window` 把它的后端设为 `WindowBackend`。每步推理开始前，vLLM 的 runner 照常调用 `WindowMetadataBuilder.build`，把同一份原生页映射同时喂给两个上游 Builder：主干的 `FlashInferMetadataBuilder` 生成规划好的 `infer` 包装器供存储窗口高速读取；旁边的 `FlashAttentionMetadataBuilder` 顺手生成一份 `flash` 元数据备用（如果用户在代码里临时指定了另一个窗口大小或裁剪了头数，随时可以回退到 FA2）。而在执行读取的 `window_attention` 里，直接调用预规划好的 `metadata.decode.wrapper.run(..., return_lse=True)`，最后原地乘上 `math.log(2)` 把 FlashInfer 以 2 为底的对数还原成自然对数 LSE 即可。

把这三条路径放在同一个 0.5B 显微镜基准下对比，账单清清楚楚：

| 窗口读取路径（0.5B，`LocalGlobal` 偶数层窗口 256） | 原生 vLLM (tok/s) | LocalGlobal (tok/s) | 相对原生 | 单次窗口读耗时（分解） |
|---|---:|---:|---:|---|
| 1. 直接调 FA2 `window_size` | 39,120 | 30,846 | **78.8%** | `86.43 µs`（GQA 打包失效，按 14 个 Q 头组织） |
| 2. 接入 vLLM 的 FlashInfer 计划 | 39,069 | 38,050 | **97.4%** | **`18.04 µs`**（读 `14.84` + 合并 `2.23` + 换底 `0.97 µs`） |
| 3. 手写 Triton 单段读（实验对照） | 39,281 | 38,654 | **98.4%** | `16.23 µs`（本体 `14.17` + 收尾 `2.06 µs`） |

看最后两行：手写 Triton 单段读虽然在局部读取上比 FlashInfer 快约 11%，在 0.5B 整机上高出 1 个百分点（98.4% 对 97.4%），但放到 7B 模型上，FlashInfer 路径已经跑出原生吞吐的 **99%**，几乎没有留下可省的余地，而我们却因此彻底甩掉了维护私有窗口内核的包袱。

当然，接入上游的执行管线并不是没有代价的。当你决定坐上宿主的顺风车，就得在三件事上替它买单：

1. **接受上游的数值脾气与资源胃口**：FlashInfer 在内部用 BF16 舍入后的概率累加 softmax 分母，再从底数 2 换底到自然对数。虽然返回的 LSE 是 `float32`，但它与 FA2 并非逐比特相同——在同一组 query 上对比 FP64 参考基准，FlashInfer 与 FA2 的输出最大绝对误差都在 `0.0076` 左右；LSE 上 FA2 与 FP64 只差约 `0.00003`，而 FlashInfer 因 BF16 分母舍入差约 `0.0013`（若与同样采用 BF16 分母的 FP64 参考相比则差异不超过 `0.00005`）。同时，启用 vLLM 的 FlashInfer 后端会默认划走一块 **`394 MiB`** 的共享工作区显存。
2. **用“带便签的补丁”应对上游缺口**：在 Prompt + 窗口（R-SWA）布局下，vLLM 使用的是它的 Triton `unified_attention` 启动器。它算出的输出完全正确，但它的 2-D 分支（含所有 prefill）直接丢弃了中间的最大值 `M` 与分母和 `L`，3-D 分支在分段归约后同样不返回 LSE。为了不把上游几百行的巨型 Triton 内核复制进自己仓库维护，我们在 `pllm/backends/vllm/rswa_attention.py` 里选择先调原装的 `unified_attention` 拿输出，紧接着用不分段模式再调一次上游同一个 `kernel_unified_attention`（`_lse_statistics`）把丢掉的统计量捞回来：

```python
unified_attention(
    q=q,
    k=kv.key_cache,
    v=kv.value_cache,
    out=output,
    # ...
    rswa_prefix_lens=kv.prompt_lens,
    rswa_window=kv.storage_window,
)
_lse_statistics(kv, q, scale, partial, maximum, denominator)
return output, maximum + denominator.log()
```

多发一次 kernel 显然有额外开销，但它守住了“不维护私有注意力分叉”的底线——只要哪天上游暴露了 LSE 参数，删掉 `_lse_statistics` 这一行即可。
3. **把跨版本差异关进隔离舱**：上游的管线在不断演进。比如在 vLLM 0.27.1 中，`FlashInferMetadataBuilder` 默认会为取主机长度触发一次多余的设备到主机拷贝，需要从 `seq_lens_cpu_upper_bound` 手动喂给它精确的主机长度，而 vLLM 0.32.0 原生就已经修好了这一点。把这几行差异关在 `if VLLM_027:` 的小隔离舱里，远比在每次上游升级时修补一整套自研内核要轻松得多。

---

## 六、 结语

回看我们在注意力与 KV 缓存上走过的这一圈，无论是前半篇里“为什么 Quest 省不下显存、为什么 SnapKV 没法在块表中间挖洞”，还是后半篇里“为什么调 FA2 的标准 `window_size` 反而慢了 5.5 倍”，背后其实都是同一个道理：

当你站在一个像 vLLM 这样高度优化的底层系统之上时，**不要只盯着算子表面的数学签名看**。

在存储层，真正决定物理显存生死的，是调度器里那个单标量的 `num_computed_tokens` 和 Worker 端只追加的 `block_table`——顺着这本账，把新布局伪装成宿主认识的 `SlidingWindowSpec`、`RSWASpec` 和跨层共享，你就能用零行内存管理代码换来翻倍的并发容量；而在算子层，真正决定带宽效率的，是 GQA 的 7:1 头部打包与 CUDA Graph 的预规划缓冲区——克制住“自己写个 kernel”或“随手调个裸函数”的冲动，接入宿主自己的 `MetadataBuilder` 执行管线，上游打磨的所有硬件优化才会真正为你所用。
