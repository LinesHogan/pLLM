# 可编程注意力与 KV 布局

先算一笔显存账。

以 `Qwen/Qwen2.5-7B-Instruct` 为例：模型共有 28 层，采用分组查询注意力（GQA），每层有 28 个 Query 头和 4 个 KV 头，每个头的维度是 128，使用 2 字节的 BF16 存储。如果线上服务允许每个请求最长跑到 `max_model_len = 2048` 个 token，那么单单为了存下这一个请求在 28 层里的 Key 和 Value，物理显存的准入上限就是：

$$2048 \times 28 \times 2 \times 4 \times 128 \times 2\text{ Bytes} = 112.0\text{ MiB}$$

在多租户的在线推理服务里，一张显卡在装下模型权重和静态图缓冲之后，剩下的显存几乎全部交给了 KV 缓存池。一个请求在最坏情况下最多占多少物理 KV 块，直接决定了调度器敢同时放进多少个并发请求。

过去两年里，围绕“如何给注意力与 KV 缓存瘦身”诞生了大量论文。在单卡的研究脚本（比如 Hugging Face 的 `generate` 循环）里，这些想法实现起来都很直接：因为 `past_key_values` 只是一组普通的连续张量，无论是让偶数层只看最近 256 个位置（局部与全局交替）、让奇数层直接复用下一层的缓存（跨层共享）、每一步只挑注意力分数上界最高的几个物理页来读（如 Quest），还是在生成过程中实时把低分历史 token 从张量里切掉（如 H2O、SnapKV、PyramidKV），无非都是改几行掩码或张量索引的事。

但当你试图把这些算法放进像 vLLM 这样基于分页注意力（PagedAttention）和连续批处理的服务引擎时，表面上看起来差不多的“稀疏注意力”方案，会因为底层记账方式的限制而分裂成三类截然不同的结果：第一类只能少读几个字节，却省不下哪怕一兆的已分配显存；第二类在不拆掉引擎核心记账结构的前提下根本无法实现；只有第三类能够真正把物理块还回空闲池，把单卡的并发容量提上去。

要弄清楚为什么会这样，不能只看算法公式，得往下看一眼推理引擎到底是怎么给 KV 显存记账的。

---

## “少读”不等于“少存”

在单请求实验里，人们很容易把“注意力计算时少读了多少历史位置”和“显存里少存了多少历史位置”混为一谈。在服务引擎里，这是两本完全独立的账。

拿 Quest（Tang et al., 2024）这类按页检索的稀疏注意力来说。它的核心思路是：既然 KV 缓存按固定大小的页（例如每页 16 个 token）存放，我们可以在每一页里算出 Key 向量在各个维度上的最小值和最大值，构成一个极值包围盒；每来一个新的 Query 向量，先和每一页的极值框做一次轻量点积，估算出该页注意力得分的上界，然后每个注意力头只挑出得分最高的 8 页（即 $8 \times 16 = 128$ 个位置）做真正的注意力计算，其余历史页全部跳过。

在 pLLM 中，这种动态挑选可以直接通过 `@pllm.custom_attention` 配合 `kv.attend(q, topk=...)` 写出来。如果我们用 `pllm.explain` 在 CPU 上计算 `Qwen2.5-7B`（`max_model_len = 2048`）单层单 token 的最大 KV 读取量，会看到全量注意力（`NativeRead`）在第 2 层每个 token 最多要读 `4,194,304` 字节，而只挑 8 页的 `QuestRead` 最多只读 `1,835,008` 字节（在 `Qwen2.5-0.5B` 上则是从 `1,048,576` 字节降到 `458,752` 字节）。

这里顺便能看出一个常被忽略的硬件细节：128 个位置明明只有 2048 长度的 $1/16$（6.25%），为什么读取字节数的静态上界还占了全量的 44%？

原因在于 GQA 的读写非对称性。在 `Qwen2.5-7B` 中，每 7 个 Query 头共享 1 个 KV 头。做全量注意力时，同组的 7 个 Query 头访问的是完全相同的历史序列，同一个 KV 页从显存读进片上内存一次，就能同时服务 7 个 Query 头；但一旦允许每个 Query 头根据自己的打分独立挑选 8 个页，在最坏情况下，同组的 7 个 Query 头挑中的页可能互不重合，原本共享的那个 KV 头就得被迫去显存里为它们各自拉取不同的物理页。

更关键的问题在于显存容量：既然每个头每步只读 128 个位置，这个请求占用的 KV 显存上限从 `112.0 MiB` 降下来了吗？

一分钱都没降，依然是 `112.0 MiB`。

道理很简单：第 500 步的 Query 觉得第 3 页得分低、跳过去没读，不代表第 501 步生成的新 Query 也不需要第 3 页。只要算法允许未来的 token 从全部历史中重新打分挑选，前面写下的每一个物理页就都必须完好无损地留在显存里待命。当调度器做准入控制、判断显存还够不够接纳下一个新请求时，它仍然必须按 2048 个 token 的全额预算为这个请求留出 `112.0 MiB`。

而且，如果你直接用普通的 PyTorch 算子在每一层去捞出所有页的 Key、转成 FP32 算极值、打分再做动态 gather，这串散碎算子本身的开销会远远盖过少读几页省下的带宽。在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0、开启 CUDA Graph 与 FlashInfer 采样器的环境下，对 `Qwen2.5-7B-Instruct`（BF16，`max_model_len = 512`，128 并发序列 = 8 个 prompt × `n=16`，每步最多调度 128 个 token，生成 256 个 token，预热 1 轮后取 3 轮中位数）实测示例 10 中的 `QuestRead`，其吞吐只有原生 vLLM 的 14%（697 tok/s 对比 5,165 tok/s；同样方式在 `0.5B` 模型上测得为原生的 7%，即 2,827 对比 39,812 tok/s）。不写专门的融合算子，纯靠 Python 层的动态挑选在解码阶段是得不偿失的。

这就划出了第一道分界线：动态稀疏读取（Selection）改变的是单步的读取集，不改变物理块的驻留集。如果你的目标是降低每个请求的显存占用上限、让同一张卡服务更多并发请求，那些不再需要的历史位置就必须真正从块表里解绑，把物理页退回给空闲池。

那么，我们能不能在生成过程中算出每个历史 token 的重要性分数，把低分 token 占用的物理块当场释放掉？

---

## 为什么按分数动态驱逐会撞上引擎的记账不变量

像 H2O、SnapKV 或 PyramidKV 这类算法，核心诉求都是在预填充或解码过程中统计每个历史位置的累积注意力得分，只保留得分最高的重度命中位置（heavy hitters）和最近窗口，把中间低分位置的 KV 释放掉。在自己管理连续张量的独立库（如 Hugging Face 配合 `kvpress` 一类工具）里，这很容易实现，因为每一层、每一个头都可以各自持有形状不同的张量。

但在 vLLM v1 这样的分页推理引擎里，如果你不动手重构引擎的调度器与 Worker 通信协议，这类“按分数动态释放中间位置”的算法是走不通的。表面上看是缺少某个释放 API，往深处看，是因为它直接违背了引擎赖以维持高吞吐的三条底层记账不变量。

我们可以顺着一个具体的请求，看看 vLLM v1 是怎么记账的。假设物理块大小（`block_size`）是 16 个 token，一个请求已经算完了前 40 个 token，现在正要送入第 41 个 token（逻辑位置 `pos = 40`）。

第一条不变量是，调度器对每个请求的进度只维护一个全局标量：`num_computed_tokens`。在调度器的账本上，这个请求只有一个数字 `40`。它默认第 0 层算到了 40，第 27 层也一定存着 40 个位置的 KV。如果你想实现 PyramidKV——让浅层保留 40 个位置，让深层扔掉 16 个位置只留 24 个——调度器里根本没有按层或按头记录不同长度的字段。

第二条不变量更严格：在 GPU Worker 内部，新 token 的逻辑位置（RoPE 旋转角下标）、注意力扫描的序列长度（`seq_len`），以及写入物理缓存的槽位（`slot_mapping`），全部由同一个进度标量直接通过整数算术推出来。当第 41 个 token 进场时，Worker 算出它的位置是 `pos = 40`，本步注意力的扫描范围是 `[0, 41)`，并直接去块表的第 `40 // 16 = 2` 格查出物理块号（比如 `Physical #9`），将新算出的 K/V 写进 `9 * 16 + (40 % 16)` 号物理槽位。

```text
Logical Pos:    0 .. 15        16 .. 31        32 .. 39 [40]  <-- pos = 40 (new token)
              ┌──────────────┬──────────────┬──────────────┐
Block Table:  │ Physical #7  │ Physical #2  │ Physical #9  │
              └──────┬───────┴──────┬───────┴──────┬───────┘
                     │              │              └──► write slot = 9 * 16 + (40 % 16)
                     ▼              ▼
  (A) Free Physical #2 in place, keep table length:
      FlashAttention scans [0, 41) ──► reads Physical #2 (reallocated to another request!)

  (B) Free Physical #2 and shift Physical #9 left to index 1:
      Tokens 32..39 now sit at logical indices 16..23; next token gets RoPE pos = 24!

  (C) Native SlidingWindow / R-SWA reclamation:
              ┌──────────────┬──────────────┬──────────────┐
Block Table:  │  null_block  │ Physical #2  │ Physical #9  │  (table length unchanged)
              └──────────────┴──────────────┴──────────────┘
              └── Static window mask guarantees kernel never reads [0..15]
```

这就形成了一个死结。假设你在跑完一步后，发现中间第 16 到 31 号位置（对应块表第 1 格的 `Physical #2`）注意力分数很低，想把它驱逐回空闲池。此时无论你怎么处理块表，都会出错。

如果你走上面图里的路径（A），把 `Physical #2` 退回空闲池，但不缩短块表：这就撞上了第三条不变量——Worker 端的块表对于运行中的请求是只追加的（append-only）。调度器每步发给 Worker 的增量数据里只有新追加的块 ID（`new_block_ids`），更新本地块表时固定走追加逻辑，没有“删除中间某格”的通道。更致命的是，空闲池转手就会把 `Physical #2` 分配给另一个并发请求；到了下一步，标准 FlashAttention 扫描 `[0, 41)` 时，依然会顺着块表第 1 格冲进 `Physical #2`，把别人的私密数据当成当前请求的历史上下文读进来。

如果你走路径（B），把后面的 `Physical #9` 往左挪到第 1 格填补空缺，并把进度计数减掉 16 呢？那原本排在 32..39 位的 token 在写进缓存前就已经乘过了位置 32..39 的 RoPE 旋转矩阵，左移之后它们的相对位置关系就乱了；而且下一步第 41 个 token 进来时，Worker 会按 `pos = 24` 给它施加旋转位置编码，位置彻底错位。此外，每一步根据运行时算出的分数去重排块表，意味着 CPU 必须停下来等 GPU 算完分数（主机同步），并且破坏了 CUDA Graph 录制所依赖的静态内存拓扑（详见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。

如果把几种常见的绕行念头挨个推演一遍，就会发现每一条都会撞上上面的某一条不变量：
* 试着写一个块粒度的自定义 `KVCacheSpec`？首先 token 级的注意力分数很难刚好凑成 16 个连续位置一整块地淘汰；即便凑齐了一整块，只要 Worker 端块表只追加不擦除，标准注意力内核扫过 `[0, seq_len)` 时照样会去解引用那个已经被回收复用的旧块号。
* 每隔若干步把高分 token 的文本抽出来重新做一次 Prefill？且不说重算的开销有多大，被删掉的低分 token 在原本的 Prefill 阶段已经通过浅层注意力把信息混进了保留 token 的深层表示里；只拿保留下来的 token 重新跑一遍 Prefill，浅层上下文就丢了，算出来的已经不再是原算法的数学结果。
* 借道 KV Connector 把缓存导出去、释放请求、再把压缩后的缓存当新请求导回来？受限于单标量 `num_computed_tokens`，所有层和所有头被迫保留完全相同的长度，无法支持逐层不同预算的算法，还得在调度器里伪造请求状态。
* 在 GPU 上自己写算子把保留的 KV 原地 gather 压紧，再截断块表尾部？同样无法表达跨层异构长度，而且必须同时修改调度器、块管理器和 Worker 块表同步通道三处核心实现。

出于同样的物理账，另一类常见方案——“压缩远端记忆”（如 Activation Beacon 或 LastKV，即保留近处窗口的原始 KV，把远处历史每几个 token 浓缩成一个摘要条目）——在分页引擎里也面临严重的损耗。第一是分页填充损耗：假设块大小是 16，你把远端历史压成 4 个宽 256 的 BF16 摘要条目，有效内容只有 `2,048` 字节，但它依然要占掉一个完整的 `8,192` 字节物理页，块对齐填充把省下来的空间又吃掉了；第二是分块预填充（Chunked Prefill）带来的非确定性：如果远端摘要依赖较深层的隐藏状态，那么当前步某个位置的摘要算没算出来，取决于调度器在这一步把长 prompt 切成了多大的块（见[《行模型》](04-packed-rows.md)），这会让同一段输入在不同负载下算出不同的结果；第三是双段注意力的延迟倒挂：在单层微基准测试（历史测量）中，先读近处窗口、再读远端摘要、最后合并两段输出与 log-sum-exp，单层耗时 `30.31 µs`，而直接用原生内核读完全量历史只需 `11.88 µs`。

把这些账算清楚之后，结论就很明确了：与其在连续编号的分页账本中间强行挖洞，不如顺应宿主的记账规则——只想少读数据时用 `topk`；真想省下物理显存时，用与块边界对齐的静态几何布局。

---

## 把静态布局翻译成宿主认识的三种原语

回头看上面的块表示意图里的路径（C）：既然 Worker 端的块表不能缩短，vLLM 自己内置的滑动窗口（Sliding Window）和保留前缀窗口（R-SWA）是怎么安全回收物理块的？

当窗口向右滑动、最老的 `Physical #7` 滑出窗口边界时，vLLM 的原生管理器确实会把 `Physical #7` 退回空闲池；为了不破坏“逻辑位置 = 块表下标 × 块大小”的线性映射，它在块表的第 0 格原位塞进了一个全局共享的占位块 `null_block`（0 号块），保持块表长度不变。之所以换成 `null_block` 不会读出脏数据，是因为滑动窗口的几何形状是静态且规整的——注意力内核自带的窗口掩码在数学上保证了它只读末尾窗口内的块（或者 R-SWA 下的完整 prompt 块加末尾窗口块），永远不会去解引用那些被涂成 `null_block` 的位置。

换句话说，vLLM v1 的原生块管理器本来就已经会管理三种能真正省下物理显存的结构：连续前缀滑出回收的滑动窗口（`SlidingWindowManager`）、保留完整 prompt 加末尾滑动窗口的 R-SWA（`RSWAManager`），以及整层不分配缓存、直接借用下方某层缓存的跨层共享（`kv_sharing_target_layer_name`）。

```text
1. keep="all" (Full Attention — all blocks stay live):
   [ Block 0 ][ Block 1 ][ Block 2 ][ Block 3 ][ Block 4 ]

2. keep="window", window=256 (Sliding Window — prefix blocks recycled):
   [  null   ][  null   ][  null   ][ Block 3 ][ Block 4 ]
      └─────────┬─────────┘
         Freed to pool

3. keep="window", window=256, sink=True (Prompt Sink + Window / R-SWA):
   [ Prompt0 ][ Prompt1 ][  null   ][ Block 3 ][ Block 4 ]
                            └──┬──┘
                         Freed to pool

4. keep="none", source=1 (Cross-Layer Shared — zero blocks allocated):
   Layer L   ──┐ (shares kv_sharing_target_layer_name)
               ▼
   Layer L-1: [ Block 0 ][ Block 1 ][ Block 2 ][ Block 3 ][ Block 4 ]
```

既然宿主自己就有这三套成熟的账本，扩展层最干净的做法就是**一行新的内存分配代码都不写**。

在 `pllm/contracts/kv_layout.py` 中，每一层的存储声明只是一个包含五个字段的数据结构：

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

当引擎启动时，赶在 vLLM 执行显存探测（profiling）和构建 `KVCacheCoordinator` 之前，`pllm/backends/vllm/kv_layout.py` 中的 `install_layouts` 会走上前去，把目标层的注意力对象改写成 vLLM 认识的原生形态：

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

这十来行代码把三种声明分别落到了上游的三个机制上：
* 当声明 `keep="none", source=k` 时，代码通过 `storage_layout` 找到下方真正持有缓存的目标层（`source=k` 只能指向下方第 `k` 层；想让某层读取更上层的 KV，要改用[《自定义层循环与状态传播》](10-custom-layer-loops.md)中的 `kv="peek"`），将当前层的 `kv_sharing_target_layer_name` 指向它，当前层便不再分配任何物理块；
* 当声明 `keep="window"` 且不带 `sink` 时，`install_window` 为该层设置 `sliding_window` 属性（并接入 FlashInfer 窗口读路径，详见[《复用上游：kernel、管理器与原语》](09-reusing-upstream-kernels.md)），vLLM 随后会自动为它生成 `SlidingWindowSpec`；
* 当声明 `keep="window", sink=True` 时，将该层的 `get_kv_cache_spec` 替换为返回 `RSWASpec` 子类。（需要注意，这里的 `sink=True` 采用的是 vLLM 原生 R-SWA 的语义，即保留**整个用户 prompt** 加上最近窗口，长生成过程中滑出窗口的中间段会被回收；这与 StreamingLLM 只保留开头固定几个 token 的做法不同。）

因为这次改写发生在管理器创建之前，后续的准入上限计算（`layout_capacity` 直接调用原生 `SlidingWindowSpec.max_admission_blocks_per_request`）、运行时的块分配与 `null_block` 回收，全都由 vLLM 原生管理器接管。当多个静态布局组的页大小相同时，`streamline_allocation` 还会将运行中请求的页数预测合并为一次计算——因为滑出回收留下的是 `null_block` 占位符，各组的块表长度始终相等——而真正的块追加和回收变动依然完全由原生管理器执行。

这里有一个同上游分组逻辑打交道的具体细节。在实现 `sink=True`（R-SWA）时，如果直接返回上游原装的 `RSWASpec`，你会发现当模型里同时存在普通全量注意力层和 R-SWA 层时，R-SWA 层根本不会回收中间滑出的块。

去翻 vLLM 对各层 `KVCacheSpec` 分组的代码就能看懂原因：vLLM 在判断“哪些层可以合并成一个管理组、共用同一套块表分配策略”时，会取出字典里的**第一个** `spec`，调用它的 `is_uniform_with_collection(specs)` 去检查其余层的 `spec` 是否与它兼容。在面向对象继承关系上，`RSWASpec` 恰好是 `FullAttentionSpec` 的子类。如果第 0 层是普通的全量注意力层，排在字典第一位被询问，它看到后面的 `RSWASpec` 是自己的子类，头数、头维和数据类型又完全一致，就会直接返回 `True`，把 R-SWA 层并进全量注意力组里一起管，窗口回收自然就失效了。

看清这个机制后，`kv_layout.py` 用两小段代码解开了这个结：

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

先派生一个 `SinkSpec`，规定只有当集合里所有层都是相同窗口大小的 `RSWASpec` 时才允许同组；再用 `sink_first` 对字典排个序，把 `RSWASpec` 挪到最前面，让 vLLM 第一个去问它。不改动上游一行代码，R-SWA 层就能独立成组，交由原生的 `RSWAManager` 回收中间块。

---

## 存法与读法分离：在原生 KV 写入之后接管读取

把“怎么存”交还给声明式参数和宿主管理器之后，用户在方法体里写的就只剩下“怎么读”。比如声明一个偶数层只保留 256 窗口、奇数层保留全量的 `LocalGlobal` 负载，代码只有寥寥几行：

```python
# check-docs: skip: excerpt from doc/guides/forward-structure.md
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """偶数层保留最近 256 个 token 的 KV，奇数层保留全量 KV。"""

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]
```

方法体里之所以不需要操心新 token 的 K/V 是怎么写进分页缓存的，是因为挂载点选在了 vLLM 不透明注意力算子（opaque attention custom op）的**内部、原生 KV 缓存写入刚刚完成之后**。

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

这样选位置有两个直接的好处：一是外层的 QKV 线性投影、RoPE 旋转编码、分页写入以及 CUDA Graph 的分段捕获顺序全部由上游现成算子包办；二是当 `read(self, q, kv)` 被调用时，当前步的新 token 已经躺在分页缓存里了。传给用户的 `kv` 是一个冻结的只读视图 `PagedKV`（包含 `key_cache`、`value_cache`、`block_table`、`seq_lens` 等借用张量）。如果需要知道紧凑打包的每一行 Query 属于哪个请求或处于序列的哪个逻辑位置，`kv.requests` 和 `kv.positions` 会在 GPU 上直接用 `torch.searchsorted` 对着 `query_start` 算出来，不产生任何主机同步。

当调用 `kv.attend(q, window=..., topk=...)` 时，它会返回 `(输出, float32 LSE)` 元组（若方法额外返回第二个张量 `aux`，后续方法可通过 `pllm.attention(layer, "aux")` 读取，例如监控局部窗口占总注意力的比例）。在底层，`FlashRead` 会根据布局和参数把读取分派到四条路径之一：
* 普通全量读取直接调用 vLLM 的 FlashAttention-2（`flash_attn_varlen_func`）；
* 静态存储窗口在 Ampere/Ada 架构上调用 vLLM 的 FlashInfer 窗口包装器（之所以不直接给 FA2 传 `window_size`，是因为从内核网格看，FA2 的窗口路径在 decode 阶段不再按 GQA 7:1 把 Query 头打包；在 `0.5B` 模型、128 并发、vLLM 0.27.1 的测量中，FA2 读 256 窗口耗时 `86.43 µs`、整机吞吐跌至 `78.8%`，窗口缩到最近 4 个位置（`window_size=(3, 0)`）时更是慢达 `91.5 µs`，比读全量的 `16.6 µs` 慢了 5 倍以上；改走上游 FlashInfer 路径后单次读降至 `18.04 µs`，这部分我们在[《复用上游：kernel、管理器与原语》](09-reusing-upstream-kernels.md)中会单独展开）；
* Prompt-Sink（`sink=True`）调用 vLLM 的 Triton `unified_attention`（并补一次不分段启动以取回上游丢弃的 `log-sum-exp`）；
* 当传入了 `topk` 索引（或者在 R-SWA 层上要求更宽窗口的交集或混合精度）时，则走 `pllm/execution/attention.py` 中的 `selected_attention` 进行按位置收集。

在写 `selected_attention` 这种与静态回收共存的 gather 逻辑时，有一条安全红线必须看住：

```python
# check-docs: skip: excerpt from pllm/execution/attention.py
size = kv.key_cache.shape[1]
# Never dereference reclaimed pages, even for masked lanes of a gather.
safe = torch.where(valid, chosen.clamp(0, kv.block_table.shape[1] * size - 1), 0).long()
blocks = torch.where(valid, kv.block_table[requests[:, None, None], safe // size], 0).long()
```

哪怕某个无效位置（`~valid`，例如填充的 `-1` 或已经滑出存储窗口的旧位置）算出来的注意力分数一会儿在 softmax 前就会被涂成 `-inf`，在查块表 `kv.block_table` 和读取 `key_cache` **之前**，也必须先用 `torch.where(valid, ..., 0)` 把无效通道的索引强行洗成 `0`。否则，一个被掩码屏蔽的通道在执行 gather 指令时依然会去解引用块表里的无效槽位，触发越界访问或读出已回收页的脏内存。

---

## 显存、吞吐与两条现实代价

把静态几何布局和动态稀疏读取放到同一个基准测试台（RTX 4090，vLLM 0.32.0；KV 上限与并发容量按 `max_model_len = 2048` 计）上对比，账面数字如下（表中同时列出 `Qwen2.5-0.5B` 的容量数据，因为小模型权重占显存少、留给 KV 池的空间大，能更直观地看到单卡可容纳并发请求数的变化；7B 吞吐测试为 `max_model_len = 512`、128 并发序列、每步最多 128 个 token、生成 256 个 token，原生基线为 `5,165 tok/s`）：

| 负载配置 | 每请求 KV 上限（7B） | 每请求 KV 上限（0.5B） | 2048-token 并发容量（0.5B） | 7B 实测吞吐（相对原生） |
| :--- | ---: | ---: | ---: | ---: |
| **NativeRead**（全量保存，原生读） | 112.0 MiB | 24.0 MiB | 358.0 个 | **99%**（5,111 tok/s） |
| **LocalGlobal**（偶数层窗口 256） | **70.4 MiB** | **16.6 MiB** | **517.8 个**（1.45×） | **100%**（5,160 tok/s） |
| **CrossLayerShared**（奇数层借用下一层） | **56.0 MiB** | **12.0 MiB** | **716.0 个**（2.00×） | **98%**（5,073 tok/s） |
| **PromptPlusWindow**（Prompt + 窗口 256） | 112.0 MiB | 24.0 MiB | 358.0 个 | — |
| **QuestRead**（全量保存，每步挑 8 页读） | 112.0 MiB | 24.0 MiB | 358.0 个 | **14%**（697 tok/s） |

`LocalGlobal` 和 `CrossLayerShared` 在保持 98%–100% 原生吞吐的同时，实打实地把 7B 单请求的 KV 准入上限压到了 `70.4 MiB` 和 `56.0 MiB`，在 0.5B 上让同一张卡能同时接纳的 2048-token 请求数从 358 个提升到了 518 个和 716 个。（在上面这个总长约 270 token 的吞吐测试里，256 的窗口刚开始滑出，因此吞吐与全量相当；而在 `0.5B`、prompt 8 token + 生成 112 token、16-token 页的分配实测中，跨层共享、2/24 层窗口和 `PromptPlusWindow` 分别省下了 50.00%、7.29% 和 6.25% 的实际物理页分配——`PromptPlusWindow` 之所以准入上限仍为 `112.0 MiB`，是因为准入控制必须防备用户发来一条刚好长 2048 token 的完整 prompt，它的显存节省发生在“短 prompt + 长生成”的运行过程中。）

但在使用这些布局之前，有两笔代价必须算在明面上。

第一笔代价是：**修改 KV 布局，改的是模型的数学函数。**

在示例 10 中，我们做了一个简单的长文检索实验：在一篇约 1500 个 token 的博物馆介绍中间塞入一句 `"The secret code for the blue door is 7361."`，然后在末尾让 `Qwen2.5-0.5B-Instruct` 回答密码是多少。
* 原生模型、保留完整 prompt 的 `PromptPlusWindow` 以及按页检索的 `QuestRead(pages=8)` 都准确输出了 `'7361'`（若将 `QuestRead` 改为只看最近 8 页的 `recent_only=True`，输出变为空字符串 `''`；将预算缩至 `pages=2`，则答错为 `'9'`）；
* 然而，把偶数层切成 256 窗口的 `LocalGlobal` 答出了错误的 `'3'`；让奇数层直接读偶数层 KV 的 `CrossLayerShared` 则直接输出了一串乱码 `'arga textDecorationugi-ingenSalesproj:oランキング étapelessKey'`。

这不是因为块回收写坏了内存，而是因为 `Qwen2.5` 在预训练时每一层都依赖自己完整的历史 KV。直接抽掉一半层的独立 KV 或截断偶数层的远端记忆，对未经适配微调的权重来说是一次剧烈的分布偏移。对于原生就按交替窗口或跨层共享训练的模型（如 Gemma 2、Gemma 3n 或 Mistral 的滑窗层），或者经过相应继续训练的检查点，静态布局是免费的显存红利；而对普通的全注意力模型，必须先在目标任务上验证质量，再谈显存节省。

第二笔代价藏在表格第一行的 `NativeRead` 里：它的方法体只有一句原样调用的 `return kv.attend(q)[0]`，为什么吞吐是 99%（5,111 对比 5,165 tok/s），少了大约 1%？

这 1%–3% 的固定差额（在[《图内挂载点：在 CUDA graph 里改残差》](05-in-graph-attach-points.md)中也同样出现），是我们为了守住“中性配置下与原生基准逐位一致（Bitwise Neutrality）”而主动交出的门票。

默认情况下，原生 vLLM 在用 `torch.compile` 录制模型 CUDA Graph 时，会启用 PyTorch Inductor 编译器，把层间的残差加法与 `RMSNorm` 融合成 Triton 内核。但在 7B 模型上核对数值可以看到：哪怕 FlashAttention-2 在 `AttentionHook` 里算出的张量与原生注意力逐元素完全相等，一旦模型图里挂入了自定义钩子，Inductor 对周围加法与归一化的融合边界和归约顺序就可能发生细微改变。BF16 尾数上的一点点舍入差异经过 28 层累积，足以让个别位置的最高分 token 发生变化。

为了保证研究者在挂载一个只读监控（如统计局部注意力占比的 `RecentMass`）或将干预参数设为 0 时，模型的输出与参考基准连最后一个比特都完全重合，pLLM 在挂载模型级钩子时会将 vLLM 的编译后端显式设为 `"eager"`（`pllm/backends/vllm/overlay_graph.py`）：

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

这里的 `backend="eager"` 并不是关闭 CUDA Graph 退回慢速的单步解释执行，而是让 CUDA Graph 直接捕获模型原本的手写 C++/CUDA 源算子，不做 Inductor 的二次融合。在 `0.5B` 模型、vLLM 0.27.1 上单独对照，仅将原生模型图从 Inductor 切为源算子捕获，吞吐会从 13,158 降至 12,911 tok/s（约 1.9%）。用这 1%–3% 的固定成本，换取了 `pllm.testing.check_on_gpu`（见[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）能够严格执行的逐位中性验证。

---

## 结语：顺着宿主的账本做设计

顺着上面剖析的记账规则，也会自然看懂 `@pllm.custom_attention` 周边的一系列使用限制：
* 必须关闭前缀缓存（`enable_prefix_caching=False`）：前缀缓存假设相同的 token 前缀永远算出相同的 KV，一旦某几层改了读取方式或窗口布局，后续层的 KV 就不再与全局缓存树里的旧块一致；
* 要求 `tensor_parallel_size=1`、FlashAttention-2 后端与非量化 KV（`kv_cache_dtype="auto"`）：各条读路径依赖从 FA2 元数据中提取块表和长度，而 FP8 打包字节上的任意位置 gather 或跨卡头切分需要额外的反量化与同步路径；
* 静态 KV 布局要求 MRv2 runner，且不能与 `@pllm.custom_forward` 或 `pllm.PositionState` 混用：静态布局会把各层拆成多个独立管理的 KV 组，而自定义层循环要求全模型共享单一的全局 KV 组；至于直接读取并复用 DeepSeek 稀疏注意力（DSA）模型自带硬件索引器的 `topk` 结果，由于 vLLM 的底层路径绑定了 Hopper 架构（SM90+），在 RTX 4090（SM89）和 A100（SM80）上自然无法启用。

回过头看，为了支持滑动窗口、保留完整 prompt 的窗口以及跨层共享这三种能真正省下显存的布局，pLLM 自己编写的物理块分配与回收代码是零行。

在一个像 vLLM 这样内部假设极其严密的底层引擎上做扩展，最省力、也最稳健的办法，往往不是在旁边另起炉灶造一套新的内存管理器，而是先读懂宿主系统的账本：看清哪些不变量是动不得的铁律（如单标量进度与只追加的块表），果断拒绝与铁律相悖的设计；再把真正可行的需求，翻译成宿主账本里原本就有的科目。
