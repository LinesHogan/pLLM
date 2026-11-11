# Falsehoods Programmers Believe About Early Exit：为什么在 vLLM 里跳过 23% 的 Token 一点也不快

> 2026-11-10 · pLLM 提交 `cff7c196` · 测量环境：除文中另注外为单张 RTX 4090（24 GiB）、vLLM 0.32.0、Qwen2.5-7B-Instruct，8 条 prompt × 16 个样本、生成 256 token。
> 这是一篇有日期的文章，不随代码更新；当前的用法以[用户指南](../../README.md#用户指南)为准，当前的实测数字以[性能与显存](../performance.md)为准。

如果把关于“早退（Early Exit）”、“自适应深度（Adaptive Depth）”或者“跳层（Layer Skipping）”的论文摊在桌子上，它们背后的直觉几乎让人无法拒绝：生成一句话的时候，并不是每个词都需要动用整座 28 层或 80 层的大模型去深思熟虑。像句号、`the`、`of` 这样毫无悬念的词，跑到第十几层时隐藏状态的方向就已经基本定型了；让它们提前交卷、跳过后面的几层，岂不是白捡的加速？

在单请求、即时执行（Eager）的研究脚本里，验证这个想法只需要五分钟。你写一个 `for` 循环跑前 16 层，算一下余弦相似度或退出分类头的置信度，然后写一行最朴素的 Python 条件判断：

```python
if cosine_similarity(h16, h12) > 0.85:
    h20 = h16  # 方向几乎没变，直接跳过第 16..19 层！
else:
    h20 = run_layers(16, 20, h16)
```

这段代码在笔记本里跑得天衣无缝。于是我们很容易产生一系列看似顺理成章的推论：跳过一层就是不跑那一层的代码；既然适用于解码，自然也适用于预填充；要是线上有四分之一的 token 跳过了这四层，那这四层的耗时就该省下将近四分之一。

遗憾的是，当你试图把这行 `if` 搬进像 vLLM 这样基于连续批处理（Continuous Batching）、分页 KV 缓存（PagedAttention）和 CUDA Graph 的生产级服务引擎时，上面这些推论会一个接一个撞得粉碎。

借用程序员圈子里那个经典的《Falsehoods Programmers Believe About...》句式，我们来挨个盘点把动态跳层放进现代推理引擎时最容易信以为真的五条错觉，顺着源码和硬件数据看看底层到底发生了什么。

---

## 错觉一：“如果第 $t$ 个 Token 决定跳过第 16 层，直接不跑第 16 层就行了。”

这是所有人写下 `if should_skip: pass` 时的第一反应。你觉得第 $t$ 个位置的 token 自己已经想清楚了，不需要第 16 到 19 层的特征变换，那就让它的残差流（Residual Stream）原封不动地穿过这四层，直接去第 20 层报到。在生成第 $t$ 个词的那一步，确实无事发生。

真正的麻烦发生在**下一步**。

当第 $t+1$ 个位置的 token 进场时，它走到第 16 层，发现自己是个难词，决定**不跳过**第 16 层，老老实实做自回归注意力（Self-Attention）。按照因果注意力的规矩，第 $t+1$ 个 token 需要回头去读取从位置 $0$ 到位置 $t$ 所有前辈留在第 16 层 KV Cache 里的 Key 和 Value。它伸手去摸位置 $t$ 在第 16 层的物理槽位——结果一脚踩空了。

因为刚才第 $t$ 个 token 跳过第 16 层的时候，根本没往那儿写过任何东西。

```text
Layer 16 KV Cache without State Propagation:
          Pos 0 .. t-1           Pos t (Skipped L16!)          Pos t+1 (Computing L16)
      ┌────────────────────┬──────────────────────────────┬──────────────────────────────┐
L15:  │     Valid K/V      │          Valid K/V           │          Valid K/V           │
      ├────────────────────┼──────────────────────────────┼──────────────────────────────┤
L16:  │     Valid K/V      │   [ ??? Uninitialized ]      │   Attends to Pos 0 .. t      │
      └────────────────────┴──────────────▲───────────────┴──────────────┬───────────────┘
                                          └──── Reads garbage / stale! ──┘
```

在 vLLM 的底层架构里，分页块表、FlashAttention 内核、前缀缓存（Prefix Caching）以及抢占重算，全部建立在一条不可动摇的静态不变量之上：**每一层、每一个已经算过的逻辑位置，都必须在对应的物理槽位里躺着一份合法的 K 和 V**（我们在[《稀疏注意力在 vLLM 里省不下显存》](06-sparse-attention-kv.md)中会完整拆解这套记账规则）。如果你在第 16 层的缓存里留下一格未初始化的空洞，第 $t+1$ 个 token 读出来的就是显存里残留的随机垃圾，甚至是上一个刚退房的请求留下的私密数据。

面对这个窟窿，学术界尝试过几种绕行的办法，但放进通用服务引擎里都会水土不服：

第一种办法是“干脆留个洞”——也就是原版 Mixture-of-Depths（MoD）的做法：既然位置 $t$ 没进第 16 层，那就让后续所有 token 在第 16 层做注意力时彻底跳过位置 $t$。这不仅要求模型必须从头按这种缺失注意力的模式训练，而且意味着同一个请求在第 15 层有 100 个 KV、在第 16 层可能只有 63 个 KV。而在 vLLM 里，调度器对每个请求只维护一个全模型共享的标量进度 `num_computed_tokens`，根本容不下逐层参差不齐的块表长度。

第二种办法是“把上一层的 K/V 抄过来”——直接拿第 15 层算好的 K 和 V 填进第 16 层的槽位。这在数学上是不成立的：Transformer 每一层的注意力权重都是独立训练的，第 15 层的 $W_K, W_V$ 投影出的线性子空间与第 16 层的 $W_Q$ 根本不在同一个坐标系里，强行做点积只会算出毫无意义的注意力分布。

第三种办法是“先欠着，下回需要时再补算”（如 FREE 一类方法）——等第 $t+1$ 个 token 走到第 16 层发现缺了位置 $t$ 的数据，再把位置 $t$ 漏掉的层跟当前位置绑在一起补算。这在单请求脚本里很机灵，但在服务引擎里，这意味着每个解码步实际送进模型的 token 数量变成了随历史跳层情况波动的变量，与要求张量形状固定的 CUDA Graph 直接冲突。

排查完这三条死胡同，唯一在数学语义与系统账本两边都站得住脚的解法，是 CALM（Schuster et al., 2022）提出的**状态传播（State Propagation）**。

它的逻辑非常干净：当某一行在某一段层上决定跳过（`active=False`）时，它的残差流确实原样穿过这段层——等价于假设这几层的变换近似于恒等映射 $f(x) = x$。既然你假设这一层没有改变残差流，那么在这个假设下，该位置在这一层本应产生的 K 和 V 是什么？就是拿它停下时的隐藏状态，送进被跳过层**自己的**输入归一化（`input_layernorm`）、被跳过层**自己的** K/V 投影矩阵和旋转位置编码（RoPE），算出来的那份 K 和 V：

```text
State Propagation at Skipped Layer 16 (active = False):
  h15 (Frozen Residual) ────────────────────────────────────────► h16 = h15 (Unchanged)
           │
           └──► [ Layer 16 Norm ] ──► [ Layer 16 QKV Proj & RoPE ] ──► Write Layer 16 KV Cache
                ( ~7% of a layer's FLOPs; skips Attention read & MLP! )
```

这样做有三个立竿见影的好处。首先，第 16 层缓存里写进去的是货真价实经过第 16 层投影矩阵和 RoPE 旋转出来的向量，子空间完全对齐，误差只局限在“跳过层前后的残差差异”之内；其次，每一层、每个位置的 KV 缓存依然满格对齐，底层的分页管理、FlashAttention 和抢占重算完全无感；最后，补写一份 K/V 只需要做一次归一化和 QKV 线性投影，完全跳过了注意力历史扫描和庞大的 MLP——按 Qwen2.5 的结构估算，这笔“补办站台登记”的手续费只占完整一层计算量的 **7%** 左右（如果在循环深度模型里多次跳过同一层，只有第一次跳过时需要补写，后续直接保留已写入的 KV 即可）。

---

## 错觉二：“为了补写这 7% 的 K/V，得把各模型的 QKV 和 RoPE 拆出来手写一套算子。”

既然状态传播只需要跑每一层的前半截（Norm + QKV 投影 + RoPE + 写缓存），不需要跑后半截（注意力计算 + 输出投影 `o_proj` + MLP），直觉的做法似乎是写一个专门的 `propagate_kv` 函数，把层里的权重矩阵拿出来自己调 `F.linear` 和旋转编码。

但只要你翻开 vLLM 的模型库看一眼，就会打消这个念头。在 Qwen2、Qwen3、Llama、DeepSeek 等不同模型家族里，注意力层的前半截细节千差万别：QKV 是合并成一个 `qkv_proj` 还是拆成三个？有没有偏置项？有没有像 Qwen3 那样的 QK-norm？是在重塑多头形状之前做还是之后做？RoPE 是哪一种变体？如果自己去重写一套前半截，不仅要把所有模型的分支全抄一遍，而且稍微漏掉一个细节就会引入隐蔽的数值漂移。

其实我们根本不需要自己写算子。仔细看一眼模型里现成的原生 `self_attn` 模块：它天生就知道怎么把当前层的 QKV 投影、QK-norm、RoPE 旋转乃至任何挂载在 Value 上的改写一丝不差地跑完。它唯一的问题是“太勤快了”——跑完前半截之后，它还会接着调内部的 `attn.forward` 去算注意力，接着调 `o_proj.forward` 去算输出投影。

既然它只是太勤快，那我们在调用它之前，**临时把它的后半截拆掉换成两个几行的替身函数，等它帮我们把 K/V 写进缓存了，再把原装方法装回去**，问题不就解决了吗？

在 pLLM 的 `pllm/backends/vllm/adaptive_depth.py` 中，这两个替身函数加起来只有十来行：

```python
def _store(
    attention: Attention, slots: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    assert isinstance(attention.impl, FlashAttentionImpl)
    key = k.view(-1, attention.num_kv_heads, attention.head_size)
    value = v.view(-1, attention.num_kv_heads, attention.head_size)
    attention.impl.do_kv_cache_update(attention, key, value, attention.kv_cache, slots)
    return q


def _omit_projection(hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
    return hidden, None
```

当原生的 `self_attn` 做完了 QKV 投影和 RoPE 旋转，把算好的 `q, k, v` 递给底层的 `attention.attn.forward` 时，接球的是我们的 `_store`。它根本不启动 FlashAttention，而是直接调用 vLLM 原生的 `do_kv_cache_update`，把新鲜出炉的 `key` 和 `value` 写进指定的物理槽位 `slots`，然后随手把 `q` 退回去充数；紧接着，`self_attn` 把结果送往输出投影 `o_proj.forward`，接球的 `_omit_projection` 连矩阵乘都不做，原样返回。

有了这两个替身，真正执行状态传播的 `propagate` 方法写出来干脆利落：

```python
def propagate(self, step: Segment, hidden: torch.Tensor, active: torch.Tensor) -> None:
    """Use native QKV/QK-norm/RoPE and value rewrites, omitting attention and the output projection."""
    owner = self.wave.custom_forward
    mappings: dict[int, torch.Tensor] = {}
    for ordinal in step.propagate:
        layer = owner.model.layers[ordinal]
        attention = layer.self_attn
        group = self.wave.attention.group_ids[ordinal]
        if group not in mappings:
            mappings[group] = self.wave.attention.metadata[attention.attn.layer_name].slot_mapping.masked_fill(
                active.flatten(), -1
            )
        slots = mappings[group]
        endpoint, projection = attention.attn.forward, attention.o_proj.forward
        try:
            attention.attn.forward = partial(_store, attention.attn, slots)
            attention.o_proj.forward = _omit_projection
            # Propagation normalizes the frozen stream; residual read/write hooks belong to computed rows.
            normal = layer.input_layernorm.forward_cuda(hidden)
            attention(self.wave.positions, normal)
        finally:
            attention.attn.forward, attention.o_proj.forward = endpoint, projection
```

注意看 `slot_mapping.masked_fill(active.flatten(), -1)` 这一行。传进 `propagate` 的 `hidden` 包含整批请求的所有行（既有 `active=False` 的跳过行，也有 `active=True` 一会儿要正常计算的行）。怎么防止状态传播去覆写那些活跃行的 KV 槽位？我们直接利用了 vLLM 缓存写入内核的一个原生约定：**凡是 `slot_mapping` 里槽位号为 `-1` 的行，内核都会将其视为无效 padding 自动跳过写入。** 于是，只需把活跃行的槽位掩成 `-1`，再通过 `try ... finally` 换上替身调一次原生的 `attention(self.wave.positions, normal)`，不活跃行的 K/V 就分毫不差地落袋为安了。

---

## 错觉三：“既然 Decode 阶段能跳层，那长 Prompt 的 Prefill 阶段也可以照搬跳层来降首字延迟。”

把解码阶段的跳层逻辑跑通之后，人们很自然地想把它推广到预填充（prefill）阶段：面对几千个 token 的长 prompt，如果能让其中简单的 token 在浅层提前退出，首字延迟（TTFT）岂不是能省下一大截？

然而，如果你在 pLLM 里试图对 `tokens="prefill"` 或 `tokens="all"` 的前向方法传入 `active=` 掩码，构建期检查会当场抛出一道异常拦住你：

```text
Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. Prompt rows run the full model by default; add an unmasked tokens="prefill" method only to change prompt computation (prefer over="span").
```

为什么框架要故意把 prefill 阶段的动态跳层堵死？因为 prompt token 和 decode token 在整个生成生命周期里扮演的角色截然不同。

在 decode 阶段，每一个新生成的 token 算完当前步的层之后，立刻就要送去采样下一个词。当它在第 16 层决定跳过时，前提是它自己的特征表示已经收敛了；它通过状态传播留下的那份近似 K/V，只会影响同一个请求接下来生成的寥寥几个词。这也是为什么 CALM、LayerSkip 等早退算法都只在生成阶段让 token 退出。

但 prompt 里的 token 完全是另一回事：**除了最后一个位置之外，prompt token 自己根本不需要采样输出任何词；它活在世界上的唯一使命，就是算出高质量的 K 和 V，作为整段对话的上下文地基，供后面生成的几百个 token 每一层、每一步反复查阅。**

如果在 prefill 阶段偷懒跳了层，整段提示词在深层的 K/V 就全变成了未经深层注意力和 MLP 提炼的近似品。这份误差不仅会贯穿后续生成的每一步，还会顺着并行采样（`n > 1`）和前缀缓存（Prefix Caching）传染给所有复用这段前缀的分支。在循环深度模型 Ouro 的论文实测中也能看到同样的现象：在 decode 阶段使用近似 K/V 几乎不损精度，但一旦在 prefill 阶段换用近似 K/V，GSM8K 数学评测的准确率就会暴跌 10 个百分点以上。更何况，prefill 阶段本来是一整块连续位置喂给 FlashAttention 高效并行的；一旦你在中间挑出部分 token 跳层压紧，序列位置不再连续，省下的那点矩阵乘还不够填补拆包重排注意力数据的开销。

因此，守住“prompt 默认跑完整模型，只在 `tokens="decode"` 时按 token 动态跳层”这条边界，是防止算法精度的根基被悄悄掏空的必要纪律。

---

## 错觉四：“在连续批处理里按余弦相似度动态跳层，就只能退回 Eager 模式或者每步回 CPU 同步。”

回到文章开头那段研究代码。在 pLLM 里，同样逻辑的 `SkipMiddle` 负载（比较第 11 层和第 15 层输出的余弦相似度，若方向变化不大则跳过第 16–19 层）写出来是这样的：

```python
@dataclass(frozen=True)
class SkipMiddle(pllm.Workload):
    threshold: float = 0.85  # 余弦相似度大于它就跳过第 16–19 层

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h12 = layers(0, 12)[11]  # 第 0–11 层，所有 token 都跑
        h16 = layers(12, 16, hidden=h12)[15]  # 第 12–15 层
        moved = F.cosine_similarity(h16.float(), h12.float(), dim=-1).unsqueeze(-1) <= self.threshold
        h20 = layers(16, 20, hidden=h16, active=moved)[19]  # 只有 moved 为 True 的行计算
        return layers(20, hidden=h20)[-1]  # 第 20–23 层
```

看着这段代码，熟悉 GPU 编程的人往往会捏一把汗：
第一，用户在 `forward` 里是按“单个 token”的直觉写张量算术的，而在运行时进来的是一整批不同租户拼在一起的二维张量 `[B, H]`，怎么既保证租户之间不串味，又不退化成慢吞吞的逐 token 循环？
第二，`moved` 是在 GPU 上每一步动态算出来的布尔掩码（比如 32 行里这步有 19 行是 `True`，下一步只有 6 行是 `True`）。如果 CPU 不调 `.item()` 把活跃行数拉回来，怎么知道该按什么形状去调第 16–19 层？可一旦调了 `.item()`，不就触发阻塞式主机同步、彻底没法放进 CUDA Graph 了吗（见[《What Color is Your Tensor?》](02-tensor-colors.md)）？

我们先看第一个问题。在加载负载时，`TokenProgram.trace`（位于 `pllm/execution/token_program.py`）会用 FakeTensor 把用户的 `forward` 追踪成一张计算图（如果用户在里面写了依赖张量数值的 Python `if/while`，追踪期就会当场报错拦截，见[《`h - h.mean(dim=0)` 不会报错》](10-isolation-testing.md)）。随后，这张图被切分成交替执行的两种积木：
* 凡是 `layers(start, stop, ...)` 调用，都变成一个模型切片 **`Segment`**——在运行时，它们始终以完整的二维批次 `[B, H]` 直接调用 vLLM 原生的层（逐 token 地去跑层循环会拆散原生批处理，单步耗时会成倍上涨；任何在 GPU 上拆散原生批处理的做法都是自毁长城）；
* 凡是夹在两次 `layers` 之间的算子片段（比如算 `F.cosine_similarity`），都被打包成一个函数式 **`Cell`**，由框架自动套上一层 `torch.vmap` 沿着行维度映射执行。这样，作者写下任何归约或变形时享受的是“单位置”的干净语义，而 GPU 硬件执行的则是地地道道的整批矢量化算子。（顺便一提，`Layers.__call__` 除了默认的 `kv="commit"` 外，还支持不写缓存的 `kv="peek"` 与跨步的 `PositionState`，方便 Ouro、Huginn 这类循环深度模型在正式提交前先“试探”几圈。）

再看第二个问题：**怎么在形状固定、零主机同步的 CUDA Graph 里，只对 `active=True` 的行跑模型层？**

答案是组合两件武器：**CUDA Graph 的设备端条件节点（Conditional `IF` Node）**，以及 **2 的幂次压紧桶（Power-of-Two Compaction Buckets）**。

在录制 CUDA Graph 时，现代 CUDA 支持往图里插入一个绑定在设备端布尔标量上的 `IF` 节点。回放时，GPU 硬件调度器会自己在显存里看一眼那个布尔值是 `0` 还是 `1`，当场决定要不要执行里面的子图，全程不需要通知 CPU。我们在 `pllm/execution/cuda_graph.py` 中将其封装成了 `device_branch`：

```python
@contextmanager
def device_branch(predicate: torch.Tensor) -> Generator[None]:
    """Capture a CUDA IF node; cold warmup runs every branch without reading its predicate."""
    graph = (
        torch.cuda.CUDAGraph.get_currently_capturing_graph()
        if predicate.is_cuda and torch.cuda.is_current_stream_capturing()
        else None
    )
    if graph is not None:
        graph.begin_capture_to_if_node(predicate)  # pyright: ignore[reportUnknownMemberType]
        _release_scope()
    try:
        yield
    finally:
        if graph is not None:
            graph.end_capture_to_conditional_node()  # pyright: ignore[reportUnknownMemberType]
            _release_scope()
```

*(注：代码里那两处不起眼的 `_release_scope()`，修补的是 PyTorch C++ 底层的一个暗坑——PyTorch 在开启和结束条件节点捕获时会各自重新打开一次同一个 graph memory pool，把池子的内部引用计数多加了两次，但在销毁图调用 `CUDAGraph.reset()` 时却只释放一次引用。如果不在这里手动平衡掉多出来的引用计数，只要图里用过条件节点，整块显存池就会永久泄漏在显卡里。)*

有了 `device_branch`，`AdaptiveSlice.__call__`（位于 `pllm/backends/vllm/adaptive_depth.py`）就在设备端搭起了一座完全不需要 CPU 参与的三路道岔：

```python
def __call__(self, step: Segment, hidden: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, ...]:
    output = tuple(hidden.clone() for _ in step.taps)
    count = active.sum()
    width = len(hidden)
    # ……
    with device_branch(count == width):
        with owner.sites_rows(self.wave.indices[0]) as stream:
            full = execute_segment(step, hidden, self.wave.layer, stream)
            for destination, value in zip(output, full, strict=True):
                destination.copy_(value)
    with device_branch(count < width):
        with owner.sites_rows(self.wave.indices[0]):
            self.propagate(step, hidden, active)
        if self.widths:
            with device_branch(count > 0):
                order = torch.argsort(~active.flatten(), stable=True)
                lower = 0
                for size in self.widths:
                    with device_branch((count > lower) & (count <= size)):
                        self.compact(step, hidden, active, order[:size], output)
                    lower = size
    return output
```

把外围维护残差挂载点的辅助上下文放到一边，盯着 `count = active.sum()`（这是一个留在 GPU 显存里的标量张量）往下看，整套分支结构一目了然：

```text
AdaptiveSlice Execution inside CUDA Graph (Batch Width = 32, 0 Host Syncs):

                      ┌── [ IF count == 32 ] ──► Run full batch [32, H] directly (no gather/scatter)
                      │
  count = active.sum()
                      │                          1. propagate() K/V for inactive rows (~7% FLOPs)
                      └── [ IF count < 32  ] ──► 2. if count == 0: no bucket fires, segment skipped!
                                                 3. [ IF count > 0 ] ──► argsort(~active) & compact into
                                                                         power-of-two bucket (1, 2, 4, 8, 16, 32)
```

1. **全员活跃（`count == width`）**：没有任何一行跳层。直接按原装宽度把整批 `hidden` 喂给模型切片跑完，零额外拷贝。
2. **部分跳过（`0 < count < width`）**：先调上一节的 `self.propagate` 为不活跃行补写 K/V；然后用 `torch.argsort(~active.flatten(), stable=True)` 做一次稳定排序，把所有 `active=True` 的行号聚拢到 `order` 数组最前头；接着看 `count` 落在哪个 2 的幂次区间 `(lower, size]` 里（比如 19 行活跃就落进容量为 32 的桶，5 行活跃就落进容量为 8 的桶），触发对应的 `device_branch`，把前 `size` 行抽出来（`compact`）送进预先分配好固定形状注意力元数据的桶里计算（桶内凑数的空位同样把 `slot_mapping` 掩成 `-1`），算完再按原位置散射回 `output`。
3. **全员跳过（`count == 0`）**：做完 `self.propagate` 补写 K/V 后，`count > 0` 为假，里面所有的桶分支统统不走——这几层的注意力和 MLP 算子被条件节点直接短路跳过。

全程 0 次 `.item()`，0 次主机同步。唯一需要多付的代价是显存空间：因为模型被拆成了由 pLLM 按活跃宽度（`ProgramWave`）分段捕获的 CUDA Graph，且条件节点里的每个桶分支在冷预热时都要分配各自的图内缓冲，它的捕获池（Capture Pool）会比普通任务更大（在 Qwen2.5-7B、32 并发下，`SkipMiddle` 大约需要 **143 MiB** 的捕获池预算，超过单任务默认的 64 MiB；较早的一版实现在 0.5B 上测得 32 并发约需 748 MiB、128 并发约需 1,392 MiB。当预算不足时，运行期会直接报错并打印出精确的所需字节数）。

---

## 错觉五：“如果有 23% 的 Token 跳过了这四层，这四层的耗时就能省下近四分之一。”

现在，状态传播补好了 KV，条件节点和 2 的幂次压紧桶也把动态跳层塞进了零同步的 CUDA Graph。是时候把这套装置拉上跑道，看看它到底能给我们省下多少时间了。

我们在单张 NVIDIA RTX 4090（24 GiB GDDR6X 显存）、vLLM 0.32.0（开启 CUDA Graph 与 FlashInfer 采样器）上加载 `Qwen/Qwen2.5-7B-Instruct`（BF16，共 28 层，隐藏维 3584，权重约 15 GiB），运行示例 12 的 `SkipMiddle` 负载（允许跳过第 16–19 层这 4 层）。

先看看在实际生成中，到底有多少 token 会触发跳层。我们对 8 个问题各贪心生成 64 个 token（除去每个回答由 prefill 阶段产出的首词，共收集到 $8 \times 63 = 504$ 个 decode token），用 `@pllm.export` 导出它们在第 12–15 层前后的残差流余弦相似度。统计发现，余弦相似度的 10%、50%、90% 分位数分别是 **`[0.768, 0.824, 0.866]`**：
* 当阈值设为 `0.90` 时，**0%** 的 token 跳层；
* 当阈值设为 **`0.85`** 时，大约有 **23% 的 token** 跳过第 16–19 层；
* 当阈值设为 `0.80` 时，大约有 **72% 的 token** 跳层。

接着，我们打开基准测试器 `pllm.benchmark.throughput`：同时跑 32 个并发请求序列（8 条 prompt $\times$ 每条并行采样 $n=4$），每条序列生成 256 个 token，预热 1 轮后取 3 轮完整生成（含 prefill）的中位数。把五种配置放在一起对比，结果如下：

| 运行配置（Qwen2.5-7B，32 并发，第 16–19 层可跳） | 实测吞吐 (tok/s) | 相对原生 CUDA Graph |
| :--- | ---: | ---: |
| 原生 vLLM，CUDA Graph | 1,727 | 100% |
| 原生 vLLM，Eager（关闭 CUDA Graph） | 1,703 | 99% |
| `SkipMiddle`，从不跳（`threshold=2.0`） | 1,703 | 99% |
| `SkipMiddle`，阈值 `0.85`（**约 23% 的 token 跳过**） | **1,692** | **98%** |
| `SkipMiddle`，总是跳（`threshold=-2.0`，100% 跳过） | **1,928** | **112%** |

*(注：在 7B、32 并发下，一步 decode 耗时约 19 ms，CPU 发射算子的微秒级开销被漫长的矩阵乘掩盖了，所以原生 Eager 也有 99%；在单步极短的 Qwen2.5-0.5B 模型上（较早在 vLLM 0.27.1 上的一次测量），原生 Eager 会暴跌到图模式的 45%，而 `SkipMiddle` 从不跳、约 14% 跳过、总是跳则分别为原生图模式的 93%、90% 和 102%，相对趋势完全一致。)*

请盯住表格里的倒数第二行：在阈值 `0.85` 下，明明有接近四分之一的 token 跳过了第 16 到 19 层，而且我们已经用 `argsort` 把剩下的活跃行压紧到了更小的桶里，**为什么端到端吞吐不仅没涨，反而从 1,703 tok/s 微跌到了 1,692 tok/s（98%）？** 为什么只有当阈值设为 `-2.0`、让批次里 **100% 的 token 全部一起跳过**时，吞吐才猛地蹿升到 **1,928 tok/s（112%）**？

这不是因为压紧桶写慢了，而是因为我们在跟 GPU 在解码阶段的头号物理定律掰手腕：**短上下文的 Batch Decode 是极度受限于权重读取带宽（Memory-Bandwidth Bound）的，而不是受限于算力（Compute Bound）的。**

拿计算器算一笔最简单的账就明白了。`Qwen2.5-7B` 的 BF16 权重一共约 15 GiB，除去词表嵌入和最后的 LM Head，平均下来**每一层 Transformer 的权重矩阵就有大约 466 MB**。

当 32 个并发请求行走到第 16 层门口时，假设有 8 行（25%）决定跳层，剩下的 24 行（或者哪怕压紧进了容量为 16 的桶，甚至极端一点，**哪怕 32 行里只有孤零零的 1 行决定不跳**！）需要继续计算第 16 层。为了给这留下来的一行或十几行算出矩阵乘法 $x W^\top$，GPU 的流多处理器（SM）必须老老实实地把第 16 层那 **466 MB** 的权重，从 GDDR6X 显存里一个字节不落地完整读进芯片一遍！

```text
Why Partial Skipping Doesn't Speed Up Short-Context Decode:

  Case A: 24 of 32 rows active (~23% skipped)
  ┌──────────────────────────┬────────────┐
  │ 24 Active Rows (Compute) │ 8 Skipped  │ ──► Still must read 100% of Layer Weights (466 MB / layer)!
  └──────────────────────────┴────────────┘     + Pays ~7% propagation + argsort/gather/scatter overhead
                                                ──► Result: 1,692 tok/s (98% of vanilla)

  Case B: 0 of 32 rows active (100% skipped -> Bucket 0)
  ┌───────────────────────────────────────┐
  │          All 32 Rows Skipped          │ ──► Conditional IF node short-circuits the whole slice!
  └───────────────────────────────────────┘     Saves 4 / 28 layers of weight reads (~1.86 GB)
                                                ──► Result: 1,928 tok/s (112% of vanilla)
```

在每次只乘几十行的瘦矩阵乘（Skinny GEMM）里，把 466 MB 权重从显存搬进寄存器的时间占了绝对大头，Tensor Core 多算几行或少算几行的浮点指令时间几乎可以忽略不计。因此，当批次里只有一部分行跳层时：
1. 你省下的，仅仅是被跳过行在那一层的注意力历史 KV 读取（在几百 token 的短上下文里，这点 KV 只有几兆字节，九牛一毛）；
2. 你没省下的，是每一层照旧要读一遍的 466 MB 权重；
3. 你反而还要额外倒贴一笔约占单层 7% 的状态传播费，外加一次 `argsort` 排序、一次 `index_select` 收集和一次 `index_copy_` 写回的搬运费！

一进一出，跳过 23% 的 token 自然就从 99% 微跌到了 98%。反过来，当整批 32 行全部决定跳层（`count == 0`）时，`device_branch(count > 0)` 直接把第 16–19 层的执行体短路了——GPU 完完全全省下了 4 层、合计约 **1.86 GB** 的权重读取流量！28 层里少读了 4 层（$4/28 \approx 14.3\%$），扣掉一点点状态传播的开销，吞吐实测提升到 **112%**，跟显存带宽账大致相符。

这也顺便解释了为什么在连续批处理服务里，我们绝不支持像 Mixture-of-Recursions 那样“在整个 batch 内部按分数排名前 $k$（Expert-Choice）”的路由方式：且不说固定挑 $k$ 行永远没法进入全跳短路的 `bucket 0`，更致命的是，跨行 top-$k$ 竞争会让张三的 token 跳不跳层取决于这一毫秒刚好跟谁拼在同一个 batch 里，直接违反了多租户请求隔离铁律（`pllm.testing.check` 会当场将其拒之门外）。

那么，按行的部分跳过与 2 的幂次压紧桶，在什么条件下才能真正省下时间？有两个明确的物理区间：
* **当上下文极长（几千到几万 token）时**：那时每一步 decode 不仅要读一层 466 MB 的权重，每一行还要去扫描自己长达几万个位置的全部历史 KV 缓存。一旦历史 KV 的读取量反超层权重，被跳过的行因为直接略过了 `attn.forward`，就能按跳过比例实打实地省下大块的 KV 带宽（原理如此，我们尚未在长上下文基准上专门实测这一交叉点）。
* **当并发批次极大、计算进入算力受限（Compute-Bound）区时**：一旦矩阵乘不再被权重读取卡脖子，把 128 行压紧成 64 行或 32 行，就能成比例地省下 Tensor Core 的矩阵乘时间。

最后，别忘了验证一件事：在一个像 `Qwen2.5-7B-Instruct` 这样**并没有专门针对跳层训练过**的普通模型上直接跳层，输出质量会发生什么？

我们在示例 12 中用 `pllm.testing.check_distribution` 做了定量测量——先让不跳层的原模型对 8 个问题各贪心生成 64 个 token，再把同样的 token 序列喂给跳层模型（Teacher Forcing），在全部 512 个位置上逐一比对两边预测下一个词的完整概率分布：
* **阈值 `+0.90`（0% 跳层）**：平均 KL 散度为 **`0.000`**，总变差距离（TV）为 **`0.00`**（另外，从不跳的 `SkipMiddle(2.0)` 在 `testing.check_on_gpu` 下与原生 vLLM 的贪心与固定种子采样逐位相同）；
* **阈值 `+0.85`（约 23% 跳层）**：平均 KL 散度升至 **`0.131`**，TV 升至 **`0.07`**；
* **阈值 `+0.80`（约 72% 跳层）**：平均 KL 散度升至 **`0.562`**，TV 升至 **`0.18`**；
* **阈值 `-2.00`（100% 总是跳）**：平均 KL 散度高达 **`0.868`**，TV 达 **`0.25`**。

有趣的是，哪怕在总是跳掉 4 层、KL 散度已经漂移到 `0.868` 的时候，模型生成的英文句子肉眼读起来依然相当通顺（比如 `"The ocean covers a significant portion of the Earth's surface..."`）。这也算是关于早退的最后半条错觉：**“生成的句子看着还通顺，就说明跳层没怎么伤精度。”** 实际上，未经过早退损失或 Layer Dropout 专门训练的模型，一旦抽掉中间几层，下一词的概率分布已经发生了显著偏移。系统底座的职责，是用状态传播守住 KV 缓存的完整、用条件节点和压紧桶守住 CUDA Graph 的零同步与高吞吐；而跳掉哪几层不掉点，永远需要配合专门训练的权重与严格的分布评测来拍板。
