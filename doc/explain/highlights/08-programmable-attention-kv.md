# 可编程注意力与 KV 布局

注意力的改进大概是这两年推理论文里最常见的题目。想法常常很直观：长序列生成时 KV cache 随上下文线性增长，可大多数旧 token 在后期得不到多少注意力，为什么不扔掉一些？或者：“偶数层只看最近 256 个 token，奇数层看全部”；“奇数层不存 KV，直接用下一层的”；“每步只读打分最高的几页”。在 HF 里试这些，改一下 attention mask 或者 `past_key_values` 的形状就行了。

这篇讲的是：在 vLLM 这样的 serving 引擎里，这些想法里哪些能真的省下显存，哪些只能少读几个字节，哪些根本做不了——以及 pLLM 怎样在不写一行内存管理代码的前提下，把能做的那些交给 vLLM 自己去管。我们从一个请求能占多少显存说起。

## 一个请求要多少 KV

在 serving 里，研究者最在乎的往往不是“这一步少算了多少”，而是**一个请求最多占多少 KV 显存**——它直接决定一张卡能同时服务多少请求。

拿 Qwen2.5-7B 算一下：28 层，每层 4 个 KV 头，头维 128，BF16。一个 2048 token 的请求，每层要存 K 和 V 两份：`2048 × 28 × 2 × 4 × 128 × 2` 字节，正好 112.0 MiB。示例 10 里 `pllm.explain` 报出来的也是这个数。

现在假设我们实现了一个 Quest 式的稀疏读取：每步每个头只读打分最高的 8 页（每页 16 个 token），其余的不读。读的字节确实少了——`explain` 估计每层每 token 最多读 1,835,008 字节，而全量是 4,194,304 字节。那这个请求的显存上限能降吗？

不能，还是 112.0 MiB。为了每一步都能从**所有**历史页里挑出最好的 8 页，所有历史页都得留在显存里供打分；调度器决定能不能再收一个 2048 token 的请求时，也得按它最坏会长到 2048 来算。**少读不等于少存**。更糟的是吞吐：示例 10 在 7B 上（RTX 4090，128 并发，每步最多 128 token），`QuestRead` 只有原生 vLLM 的 14%（697 对 5,165 tok/s）——它的打分和 gather 是 PyTorch 写的，开销远大于省下的读取。

所以第一条分界线是：想省带宽，可以在读法上做文章；想多服务几个请求，就得让物理块**真的回到池子里**。那么能不能像 SnapKV、H2O 那样，在生成过程中按分数把不重要的块还回去？

## vLLM 的 KV 是怎么记账的

要回答这个问题，得先看 vLLM v1 怎么管理 KV。用一个小例子：块大小 16，一个请求已经算到第 100 个 token。

调度器对这个请求只记一个数：`num_computed_tokens = 100`。这是第一条硬约束：**进度是一个标量**。它隐含的假设是，第 0 层算到了 100，第 27 层也算到了 100。假如你想“第 7 层扔掉 16 个 token、第 8 层不扔”，调度器里没有地方记“第 7 层少了 16 个”。

worker 端不查任何字典，而是直接由这个标量推出一切：这个请求本步要看的长度是 `seq_len = num_computed + query_len`，第 i 个新 token 的位置是 `pos = num_computed + i`，它的 K/V 写到哪个物理槽，由块表算出来：`block_table[pos // 16]` 是第几块，`pos % 16` 是块内第几格。这是第二条：**位置、长度、物理槽位是被同一个整数绑死的**。在中间挖掉一块，后面所有位置都会错位；把后面的块往前挪，位置编码又对不上了（K 在写进 cache 之前已经乘过 RoPE 的旋转）。

第三条最容易被忽略：调度器和 worker 之间，运行中的请求只能收到“新追加了哪些块”（`CachedRequestData.new_block_ids`，worker 端写入时 `overwrite=False`）。**worker 的块表只能追加**，没有“删掉中间某块”或“整体左移”的通道。

那 vLLM 自己的滑动窗口是怎么回收显存的？它的 `SlidingWindowManager` 和 `RSWAManager`（保留 prompt 加最近窗口）确实会释放块：滑出窗口的连续前缀块还给空闲池，原位置填上一个占位的 `null_block`，块表长度不变。这之所以安全，是因为窗口掩码本来就保证 kernel 不会去读那些位置——占位块里是什么都无所谓。可如果你按分数释放中间的块，标准 FlashAttention 读的是 `[0, seq_len)` 的整段，就会读到那块已经被分给别的请求、写满了别人上下文的物理页。

再加上 CUDA graph：每一步按分数决定释放哪一块、改写块表，需要主机参与，正好撞上 graph 回放依赖的静态指针（[设计取舍](../design-tradeoffs.md)）。

## SnapKV 为什么做不了

我们没有止步于“看起来很难”。一份针对 vLLM 0.27.1 和 0.32.0 源码的评估，把能想到的绕法一条条试过去：

按块粒度自定义一个 KV spec？token 级的分数对不齐 16 个 token 一块的粒度，而且 worker 的块表不会跟着更新，照样读到脏块，不正确。定期用保留下来的 token 重新 prefill？那些被删掉的 token 在浅层已经注入的上下文就丢了，语义已经不是 SnapKV，代价还极高。用 KV connector 把 KV 存出去、释放、再把压缩后的 KV 加载回来？这要求所有层、所有头保留相同的长度（因为 `num_computed_tokens` 全层只有一个），做不了 PyramidKV 那样的逐层不同预算，还得伪造调度状态。在 GPU 上原地 gather 压缩再截断尾部？同样不支持逐层不同长度，而且必须侵入调度器、管理器和 worker 块表三处。

结论写在评估的第一行：**不存在不破坏 vLLM v1 假设的干净方案，来支持释放显存的、基于分数的 KV 驱逐**。2026 年 11 月我们决定不做。这不是“暂不支持”的客套话，而是读懂约束之后的判断：真要省显存，用规则驱动的静态布局；只想少读，用 `topk`。两条路都在下面。

## 把每一层伪装成 vLLM 认识的层

能真的还块的，只有 vLLM 自己的管理器能处理的那几种几何形状：连续前缀滑出（窗口）、保留 prompt 加窗口（R-SWA）、整层借用别的层（跨层共享）。pLLM 的想法很简单：**不写新的内存管理器，只在 vLLM 建管理器之前，把每一层“伪装”成它已经会管的那种层**。

存法用一个很小的声明表示（`pllm/contracts/kv_layout.py`）：

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

`keep="all"` 是原样；`keep="window"` 只留一个窗口，加上 `sink=True` 就是“整个 prompt 加最近窗口”；`keep="none"` 这一层不分配 KV，直接用下面第 `source` 层的。

翻译成 vLLM 的语言，是 `install_layouts` 里的几行（`pllm/backends/vllm/kv_layout.py`）：

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

三种布局，三个 vLLM 早已支持的机制：`kv_sharing_target_layer_name`（vLLM 自带的跨层 KV 共享）、一个 R-SWA 的 KV spec、带窗口的注意力层（窗口读用 FlashInfer，细节在[《复用上游》](09-reusing-upstream-kernels.md)）。这些改写发生在 vLLM 做显存探测和建管理器**之前**，之后的分配、回收、抢占时的引用计数，全由原生管理器负责。连准入容量都不用自己推导：`layout_capacity` 直接实例化 vLLM 的 `SlidingWindowSpec`，调用它自己的 `max_admission_blocks_per_request`。

伪装也有不顺的时候。prompt-sink 那一层一开始并没有被当成独立的 R-SWA 组来管理，而是被全量注意力那一组“吸收”了。原因在 vLLM 给各层 KV spec 分组的方式：它拿第一个 spec 去问后面的“你和我能不能一组”（`is_uniform_with_collection`），而全量注意力的 spec 看头数、维度、dtype 对得上，就点头了。修法是两处小改动：

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

`SinkSpec` 只肯和窗口相同的 R-SWA 层同组；`sink_first` 把 R-SWA 的 spec 排到最前面，让 vLLM 第一个去问的是这个挑剔的家伙。两段 docstring 把“为什么”交代得很清楚——这是读懂了上游的分组逻辑之后，用最小的改动让它做对的事。

## 读法：挂在原生 KV 写入之后

存法交给了 vLLM，读法则是用户写的。示例里最简单的“局部 / 全局交替”长这样（`doc/guides/forward-structure.md`）：

```python
# check-docs: skip: excerpt from doc/guides/forward-structure.md
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """偶数层保留最近 256 个 token 的 KV，奇数层保留全量 KV。"""

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]
```

方法拿到的是已经做完投影和 RoPE 的 `q`，以及一个 `PagedKV`。挂载的位置很关键：它装在 **vLLM 不透明的 attention 算子里面、原生的 KV 写入之后**（`pllm/backends/vllm/custom_attention.py` 的模块 docstring：“The upstream op already supplies graph ordering and capture”）。于是这一步新 token 的 K/V 已经写好了，graph 里的顺序和捕获都由上游提供，方法本身不碰任何写缓存的事。

`PagedKV`（`pllm/execution/attention.py`）把借来的东西写得很清楚，包括那句最重要的提醒：

```python
# check-docs: skip: excerpt from pllm/execution/attention.py
@dataclass(frozen=True)
class PagedKV:
    """Borrowed paged K/V, already including the current tokens; never changes cache storage.

    Caches have shape [blocks, block_size, kv_heads, head_dim]. ``topk`` contains
    logical token positions [rows, k] or [rows, query_heads, k], padded with -1.
    Different heads and layers may select different budgets. Selection saves reads,
    not allocated KV memory. ``window`` counts tokens including the current token.
    """

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
        """Request index per query, materialized only by reads that need token positions.

        CUDA-graph padding rows past the last query map to the last request; their outputs are discarded.
        """
        rows = torch.arange(self.query_count, device=self.query_start.device)
        return torch.searchsorted(self.query_start[1:], rows, right=True).clamp(max=len(self.seq_lens) - 1)
```

“Selection saves reads, not allocated KV memory”——存法和读法在 API 上就是分开的，不会被混为一谈。顺便看一眼 `requests`：每个 query 行属于哪个请求，用一次 `searchsorted` 在 GPU 上算出来，不需要回 CPU（[《行模型》](04-packed-rows.md)讲过这些行是怎么拼的）。

`kv.attend` 最后落到哪个 kernel，按情况分派：全量读用 FA2；存储窗口用 vLLM 的 FlashInfer 计划；prompt-sink 用 vLLM 的 Triton `unified_attention`；带 `topk` 的选择读用 PyTorch 写的 gather 实现。存储窗口曾经走 FA2 的 `window_size`，在 0.5B 上反而慢——窗口读每次 86.43 µs、吞吐只有原生的 78.8%——换成 FlashInfer 才回来。那是另一个故事，在[《复用上游》](09-reusing-upstream-kernels.md)里讲。

## 能省多少，以及代价

示例 10 的账（`max_model_len=2048`；吞吐在 RTX 4090、vLLM 0.32、7B、128 并发、每步最多 128 token 下测）：

| 写法 | 每请求 KV 上限（7B） | 每请求 KV 上限（0.5B） | 吞吐（7B，相对原生 5,165 tok/s） |
|---|---:|---:|---:|
| NativeRead（只是调用原生读） | 112.0 MiB | 24.0 MiB | 99% |
| LocalGlobal（偶数层窗口 256） | 70.4 MiB | 16.6 MiB | 100% |
| CrossLayerShared（一半的层借用相邻层） | 56.0 MiB | 12.0 MiB | 98% |
| PromptPlusWindow（prompt + 窗口 256） | 112.0 MiB | 24.0 MiB | — |
| QuestRead（每步 8 页） | 112.0 MiB | 24.0 MiB | 14% |

在 0.5B 上，这意味着一张卡能同时容纳的 2048 token 请求从 358 个变成 518 个（LocalGlobal）或 716 个（CrossLayerShared）。PromptPlusWindow 的上限不变——prompt 本身可能就有 2048 个 token——实际能省多少取决于生成长度；一次 0.5B 的测量（prompt 8 + 生成 112 个 token，16 token 一页）里它实际省了 6.25%。

但这张表必须配一个警告：**布局改的是模型的数学**。示例 10 做了一个小小的长文检索：在一段长文中间埋一句“The secret code for the blue door is 7361.”，最后问密码。原生 vLLM 答“7361”；PromptPlusWindow 也答对了（QuestRead 也是）；LocalGlobal 答“3”；CrossLayerShared 输出了一串乱码。这不是实现的 bug：没按这种布局训练过的模型，强行换布局，质量可能明显变差。先在自己的任务上验证质量，再看吞吐。

## 为了逐位中性，放弃一点融合

这一节是 2026 年 11 月刚发生的事，也是“逐位中性”这条约束的一笔真实代价。

pLLM 对挂进模型的东西有一条要求：中性设置下（比如 `NativeRead`，只调用原生读法），结果要和原生 vLLM 逐位相同。在 7B 上，这条差点守不住。我们核对过，FA2 的读在 7B 的头配置下和原生 attention 逐元素相同；不一致来自别处——Inductor 对“挂了 hook 的模型图”和“原生参考图”融合 add 和 RMSNorm 的方式不同，舍入就不同。

我们的选择是不用 Inductor 融合这张模型图：带自定义注意力的 workload，vLLM 用 `"eager"` 编译后端录图，也就是录模型的源 CUDA 算子（`pllm/backends/vllm/overlay_graph.py`）：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/overlay_graph.py
return {
    "additional_config": {_BUDGET: config.execution.cuda_graph_pool_budget_bytes or 0},
    "compilation_config": CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        # Attention hooks must keep the same source kernels as the vanilla neutral reference.
        # Inductor fusion can change model rounding even when the FA2 reads are bitwise identical.
        backend="eager",
        # ... graph mode and capture sizes ...
    ),
}
```

注意这里的 eager 不是“不用 CUDA graph”，而是“graph 里录源算子，不录 Inductor 生成的融合 kernel”。改完之后，7B 上反复对比的 logits 和 token id 都逐位一致。参数覆盖、改残差这些模型 hook 早就是这么录的；这次只是把自定义注意力也纳入同一条规则。代价是失去 Inductor 的融合：在 0.5B 上，仅仅把原生模型图从 Inductor 换成源算子就慢 1.9%（13,158 对 12,911 tok/s），7B 上挂改写时的固定差距约 3%（[《图内挂载点》](05-in-graph-attach-points.md)）。我们愿意用这几个百分点换“中性就是逐位相同”——否则每一次对比实验，都得先分辨差异来自算法还是来自编译器。

## 另一个“不做”

除了 SnapKV，还有一个我们认真试过、最后决定不做的方向：压缩远端记忆（Activation Beacon、LastKV 一类）——近处保留原始 KV，远处每几个 token 压成一个条目。一次原型给出了三个理由。第一，分页的填充损耗：按 4 个 token 压一个、宽 256 的 BF16 条目，一页 4 个条目只有 2,048 字节内容，却要占一个 8,192 字节的物理页，远端分配一点也没省。第二，深层条目能不能算出来，取决于 prefill 怎么分块，同一段文本在不同负载下可能不同。第三，速度：近处窗口一次、远端一次、再合并，一层要 30.31 µs，而原生全量读只要 11.88 µs（微基准，历史测量）。结论同样是不做（[设计取舍](../design-tradeoffs.md)）。

## 做不到的事

`@pllm.custom_attention` 要求 `tensor_parallel_size=1`、FlashAttention-2、未量化的 KV（`kv_cache_dtype="auto"`），并且关闭 prefix caching——改写了注意力，相同前缀的 KV 就不再相同。静态布局要求 MRv2 runner，不能和 `pllm.PositionState`、`@pllm.custom_forward` 同时用。跨层共享只能借用下面、已经算过的层（`source=k` 表示下面第 k 层）；想读还没执行到的更深层，要用 custom_forward 的 `kv="peek"`。复用 DeepSeek 一类模型自带的稀疏选择（DSA 的 top-k）需要 SM90（Hopper），在 RTX 4090（SM89）和 A100（SM80）上不可用。短上下文时窗口还没填满，既不省显存也不省读；只有长上下文才见效。Quest 式的选择读今天只有原生的 14%，它的打分和 gather 还是 PyTorch 写的。

## 可以带走的东西

这一篇里 pLLM 自己写的内存管理代码是零行。能做的布局，都被翻译成了 vLLM 已经会管的那三种层；做不到的，我们在读懂了三条约束之后明确说不。我们觉得这里有一个通用的判断方法：想在别人的系统里加一个能力时，先问它的记账方式是什么——vLLM 用一个整数记进度、用只能追加的块表记位置——然后看你的能力能不能用这套记账方式表达。能表达的，交给它；不能表达的，绕法往往比你想的更贵，也更脆弱。

另一个更朴素的教训是：“少读”和“少存”是两件事，最好在 API 上就分开。很多看起来能省显存的算法，仔细一看只是省了带宽。
