# 自定义层循环与状态传播

在 notebook 里验证一个“自适应深度”的想法，代码大概是这样的（示意）：

```python
# check-docs: skip: illustrative prototype, not pLLM code
h = embeddings
for layer in layers[:16]:
    h = layer(h)
if cosine_similarity(h, previous) > 0.85:  # 已经“想好了”，后面几层不算了
    h = skip(h)
else:
    for layer in layers[16:20]:
        h = layer(h)
```

CALM、LayerSkip 一类的早退，Ouro、Huginn 一类的循环深度，Mixture-of-Depths 一类的按 token 路由，说的都是这件事的不同版本：有的 token 少走几层，有的多绕几圈。在 HF 里，这就是一个 `if` 或者一个 `for`。

这篇讲的是：在 vLLM 里，这个 `if` 会撞上什么；为什么 pLLM 最后选了一个叫“状态传播”的做法作为统一的答案；它怎样在不写 kernel 的情况下实现、怎样在 CUDA graph 里按 GPU 上的条件跳过；以及为什么跳过 23% 的 token 并不会让这几层快 23%。

## 一个 if 会撞上什么

第一个问题最显然。条件 `cosine_similarity(...) > 0.85` 的值在 GPU 上，Python 要判断它，就得把它拷回 CPU——一次主机同步。decode 步是录成 CUDA graph 回放的，回放时 Python 根本不执行（[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)），这个 `if` 写不进去。

第二个问题来自连续批处理。一步里的张量是几十上百个请求的行拼在一起的（[《行模型》](04-packed-rows.md)），第 0 行“想好了”可以跳，第 1 行还得算，第 2 行是别人的 prompt。Python 的 `if` 是全有或全无的，它表达不了“这一行跳、那一行不跳”。

第三个问题最要命，而且和计算无关，和**记忆**有关。假设第 t 个 token 跳过了第 16 层。下一步，第 t+1 个 token 走到第 16 层做注意力时，要读前面每个位置在第 16 层的 K 和 V——包括第 t 个。可第 t 个位置在第 16 层没有 K/V。vLLM 的分页 KV、FlashAttention、prefix cache、抢占后的重算，全都假设**每一层、每个位置都有 K/V**（[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)讲过它是怎么记账的）。

所以要在 serving 里改前向的拓扑，第一个要回答的问题不是“怎么跳”，而是：**被跳过的那一层，KV 怎么交代？**

## 几种交代方式

最直接的想法是留个空洞：第 t 个 token 不进第 16 层的注意力，后面的 token 当它在这一层不存在。这是原版 Mixture-of-Depths 的做法。问题有两个：它只适用于按这种方式训练过的模型；而且每一层要维护自己的 KV 长度，而 vLLM 对一个请求只记一个进度。做不到。

第二个想法听起来便宜：把第 15 层的 K/V 复制到第 16 层。可每一层的 K/V 投影把 hidden 投到不同的子空间，第 15 层的 K 放进第 16 层的注意力，内积没有意义。

第三个想法是推迟补算（FREE 一类的做法）：先跳过，以后需要时再把漏掉的深层补上。可这会让每个请求每步要算的 token 数不固定，和固定形状的 CUDA graph 冲突。

活下来的是 CALM 提出的**状态传播**。一个比喻：把 28 层想成一趟车的 28 个站。第 t 个 token 是一位乘客，到第 15 站时已经“想好了”，不需要在第 16 到 19 站下车办事。他就坐在车上不动——他的残差流原样穿过这几层，等价于把它们当成恒等映射。可每一站的登记簿（这一层的 KV cache）上必须有他一行，否则后面查账的乘客会找不到他。登记什么？用他**路过时的样子**：拿他停在第 15 层末尾的 hidden，经过第 16 层**自己的** input norm、K/V 投影和 RoPE，算出 K/V，写进第 16 层的 cache。（第一次跳过时补写一次，按 `explain` 的估计约占 Qwen2.5 一层计算量的 7%；之后再跳过同一层，保留已写的值。）

这个做法好在哪里，可以说得很具体。KV 的形状和语义一点都没变：每一层、每个位置都有由这一层自己的参数算出的 K/V，所以分页 KV、原生 FlashAttention、prefix cache、抢占重算全都照常。它和“怎么决定跳”无关：余弦阈值、学出来的退出头、置信度、固定的跳层表、循环模型每圈停不停，最后都归结为一件事——“这一行不跑这一段层”。它只依赖这一行自己：补写只用到这一行停下时的状态，和同批的其他请求无关，所以一个请求的结果不会因为和谁拼在一起而变，也能进 graph、不需要同步。

还有一条我们刻意加的限制：`active=` 只能用在生成的 token 上（`tokens="decode"`）。prompt token 不输出任何东西，它唯一的作用就是给后面所有 token 提供 K/V；在 prefill 跳层，整段上下文的深层 K/V 都成了近似值，影响的是整段回答。Ouro 论文的实测是：prefill 阶段用近似的 K/V 会掉 10 个以上的 GSM8K 点，decode 阶段却几乎不掉。CALM、LayerSkip 这些早退方法也只让生成的 token 退出。

## 作者写的程序

在 pLLM 里，上面那个 notebook 原型变成这样（`doc/explain/performance.md` 里的例子）：

```python
# check-docs: skip: excerpt from doc/explain/performance.md
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

`if` 变成了一个张量 `moved`，交给 `layers(16, 20, ..., active=moved)`。`Layers.__call__`（`pllm/execution/token_program.py`）的 docstring 就是它的规格：

```python
# check-docs: skip: excerpt from pllm/execution/token_program.py
def __call__(
    self,
    start: int = 0,
    stop: int | None = None,
    hidden: torch.Tensor | None = None,
    *,
    kv: KV = "commit",
    active: torch.Tensor | None = None,
) -> LayerOutputs:
    """Commit causal K/V, or peek at earlier committed positions plus fresh own K/V.

    A peek leaves the cache untouched and requires an earlier commit to every
    layer in this slice within the same custom forward.
    active is a bool [rows, 1] tensor in token mode. False rows keep hidden
    at every layer. On a first commit their K/V is propagated through each
    skipped layer's own norm, QKV and RoPE; later skips retain the last commit.
    Peek never writes, including skipped rows. Residual and layer-logit readouts
    see the frozen stream of skipped layers; custom residual writes run on computed rows.
    """
```

作者写的是“一个 token 的程序”。框架用 fake 张量追踪它（数据依赖的 Python 控制流在追踪时就会被拒绝），切成“模型切片”和“函数式片段”两种东西：模型切片仍然整批调用 vLLM 原生的层，函数式片段用 `vmap` 映射到每一行，保持“一个位置”的语义。另一种读 KV 的模式 `kv="peek"` 只读已经提交过的位置，从不写——循环深度里“先试着再算一圈”用得上。

## 不写 kernel 的状态传播

状态传播要为被跳过的行，用这一层自己的 norm、QKV 投影和 RoPE 算出 K/V 并写进 cache。最直接的办法是写一个融合 kernel。我们没有写，因为原生的 attention 模块已经会做所有这些事——它只是还会多做两件我们不想要的事：算注意力、过输出投影。那就把这两件临时拿掉（`pllm/backends/vllm/adaptive_depth.py`）：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/adaptive_depth.py
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

`_store` 只做 vLLM 自己的 KV 写入（`do_kv_cache_update`），不算注意力；`_omit_projection` 让输出投影什么也不做。然后，在 `propagate` 里临时换上它们，**照常调用原生的 attention 模块**：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/adaptive_depth.py
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

两个细节值得看。`slot_mapping.masked_fill(active.flatten(), -1)`：活跃的行（它们会正常计算这一层、自己写 K/V）槽位被填成 −1，vLLM 的写入 kernel 看到 −1 就跳过；于是只有被跳过的行写入了 K/V。`try/finally`：无论中间发生什么，原来的 `forward` 都会被换回去。QKV 投影、QK-norm、RoPE、甚至 value 改写的 hook，全都由原生代码完成。我们很喜欢这一招：不是写一个“和原生一样”的 kernel，而是让原生代码自己去做，只拿掉不需要的部分。

## 在 graph 里按 GPU 上的条件跳

剩下的问题是，怎么把“有的行算、有的行跳”放进 CUDA graph。CUDA 12.4 起，graph 里可以有条件（IF）节点：录制时录下一段子图，回放时由 GPU 上一个布尔张量的值决定执行与否，不需要回 CPU。`pllm/execution/cuda_graph.py` 把它包成了一个上下文管理器：

```python
# check-docs: skip: excerpt from pllm/execution/cuda_graph.py
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

（旁注：那两次 `_release_scope()` 是一个真实系统里才会碰到的小坑。它上面的注释说，PyTorch 的条件节点在 begin 和 end 时各重新打开一次同一个显存池，引用计数加了两次，而 `CUDAGraph.reset` 只释放一次——不平衡掉，池子就永远释放不了。）

条件节点解决了“走哪条路”，没解决“形状固定”：这一步 17 行活跃，下一步 5 行。办法是两层梯子：按活跃行数落进 2 的幂次的桶，桶里用稳定排序把活跃行压到前面。`AdaptiveSlice.__call__` 把整件事写得很清楚：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/adaptive_depth.py
def __call__(self, step: Segment, hidden: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, ...]:
    output = tuple(hidden.clone() for _ in step.taps)
    count = active.sum()
    width = len(hidden)
    owner = self.wave.custom_forward
    if owner.sites is not None:
        owner.sites.frozen(step.start, step.stop, hidden)
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

读起来就是三种情况。全都活跃（`count == width`）：直接整批跑，什么都不搬。部分活跃：先为不活跃的行做状态传播；再把活跃行用 `argsort(~active, stable=True)` 压到前面，按活跃行数落进对应的 2 的幂次桶，只算这个桶。全都不活跃：`count > 0` 为假，整段层的计算——包括读这几层的权重——都被条件节点跳过。每种情况的形状都是录制时就定好的，没有一次主机同步。

## 为什么跳过 23% 不会快

现在可以看数字了。示例 12 在 Qwen2.5-7B 上（RTX 4090，vLLM 0.32，32 个并发请求 = 8 个 prompt × n=4，256 token，28 层中的第 16–19 层可跳）：

| | tok/s | 相对原生 graph |
|---|---:|---:|
| 原生 vLLM，CUDA graph | 1,727 | 100% |
| 原生 vLLM，eager | 1,703 | 99% |
| SkipMiddle，从不跳 | 1,703 | 99% |
| SkipMiddle，阈值 0.85（约 23% 的 token 跳过） | 1,692 | 98% |
| SkipMiddle，总是跳 | 1,928 | 112% |

23% 的 token 跳过了 4 层，吞吐不升反降。原因在 decode 的本性：矩阵乘是**读权重受限**的。7B 一层的权重约 466 MB，一个 batch 里只要还有 1 行要算这一层，这 466 MB 就得完整读一遍。跳过的行省下的只是它自己那一行的计算——在带宽面前几乎不值钱——却要付出状态传播（约一层的 7%）和压紧的代价。只有整批都跳过、落进“桶 0”时，条件节点才把整段层连同权重读取一起跳过：4/28 层的权重不读，吞吐 112%，和少读 4/28 的权重大致相符。

所以“跳过 50% 的 token 让这几层快一倍”在短上下文的 decode 里不成立。上下文很长时情况会不同：每行要读自己全部历史的 K/V，这部分可能超过一层权重的量，被跳过的行不做这一层的注意力，这部分读取会按跳过行数成比例省下——原理如此，我们还没有实测。batch 很大或者 prefill 那样受算力限制的场景，压紧活跃行会直接省下矩阵乘。（同一套机制在 0.5B 上的较早测量：从不跳 93%，约 14% 的 token 跳过时 90%，总是跳 102%，eager 只有 45%——0.5B 一步太短，CPU 发射开销藏不住。）

质量也要看。同一个示例里，和不跳层的模型比下一 token 分布：阈值 0.85 时 KL 平均 0.131、TV 0.07；阈值 0.80（72% 的 token 跳过）时 KL 0.562；总是跳时 KL 0.868。拿一个没有为早退训练过的模型直接按相似度跳层，分布会明显偏离。机制只负责让“跳”变得正确、便宜；值不值得跳，是算法和模型的事。

## 走过的路

层循环在 pLLM 里不是一开始就长这样的。2026 年 10 月之前，循环深度一类的程序（recirculation、layer_loop、step_loop）由一套逐 token 的 region 解释器执行。那次被称为 Step 6 的重构删掉了这套路径（包括 `tokenwise_region.py`、`region_layers.py` 在内的四个模块），改为批量调用 vLLM 的原生层。重构前后的对比（RTX 4090，0.5B，FA2，MRv2，batch 8，256 + 256 token，外层模型 eager，历史测量）：recirculation 的 prefill 从 6,689.69 ms 降到 863.52 ms，decode 从每步 26.23 ms 降到 3.26 ms；layer_loop 的 decode 从 16.36 降到 1.93 ms/步；step_loop 从 14.04 降到 1.73 ms/步。逐 token 地解释一段程序，在 GPU 上几乎总是输给“整批调原生层”。这件事对[《自动放置与重叠》](11-automatic-placement-overlap.md)里那个被删掉的编译器也很重要。

## 做不到和更贵的地方

部分跳过在短上下文的 decode 里几乎不提速，上面已经说了；跳层也不会释放被跳层的 KV 显存——状态传播恰恰要往那里写。用了 `@pllm.custom_forward`，vLLM 的整模型 graph 就换成了 pLLM 按层切片、按活跃宽度录制的分段 graph，捕获池会很大：0.5B 上 32 并发约 748 MiB、128 并发约 1,392 MiB；7B、32 并发时示例 12 的负载约需 143 MiB，已经超过默认的 64 MiB（示例里设成了 512 MiB），而 `explain` 还不估算这部分，预算不够时报错信息会给出所需字节数。

custom_forward 下，“第 L 层的唯一输出”不再有定义——一层可能被绕好几圈，也可能被跳过——所以同一个 workload 里不能再读 `pllm.hidden` / `pllm.final_hidden`，要改读 `pllm.residual` 或 `pllm.layer_logits`；也不能用 `pllm.model.Adapter`。`active=` 只用于 decode；要改变 prompt 的计算方式（比如固定的层循环），写一个不带 `active=` 的 `tokens="prefill"` 方法，优先用 `over="span"`，因为逐 token 的程序在 prefill 上会按位置串行执行。expert-choice 式的 MoD（一批里 top-k 竞争）会让一个请求的结果依赖同批的其他请求，pLLM 明确拒绝（[设计取舍](../design-tradeoffs.md)）。

## 可以带走的东西

这一篇的关键一步，是把问题从“怎么跳”换成了“被跳过的地方怎么交代”。一旦答案是“让每一层、每个位置照旧有 K/V，只是用这一行停下时的样子去算”，下游的一切——分页 KV、prefix cache、抢占重算、CUDA graph——都不用改。我们觉得这是处理“动态性”的一个通用思路：别让动态性泄漏成数据结构的形状变化，而是在边界上补齐它，让下游看到的永远是规整的形状。

另一个可以带走的是那张吞吐表的教训：在带宽受限的地方，“少算”不等于“少读”。省下的必须是读，才会变成速度。
