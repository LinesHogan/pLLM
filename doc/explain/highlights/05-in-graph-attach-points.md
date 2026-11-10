# 图内挂载点：在 CUDA graph 里改残差

做可解释性和对齐研究的人很熟悉一种叫激活引导（activation steering，比如 CAA、ActAdd）的技术：在第 12 层的输出上加一个方向向量，`h ← h + α·v`，模型的语气就跟着变。在 HF 里这是一个 forward hook，三行代码。

我们就从这三行开始，看看把它们搬进 vLLM 会发生什么，再一点点把它扩展成能改写任意残差连接的机制。

## 三行 hook，掉了三分之二的吞吐

pLLM 早期有一个叫 `ActivationPatch` 的组件，老老实实地模仿 hook：每一步在 Python 里为每个请求挑出要打补丁的行、上传索引、借用字段，把向量加上去。它只能在 eager 模式下跑，`explain` 里写着一句“activation patches run outside CUDA graphs”。用户这边的声明倒是很短（入门教程里的 `Steering`）：

```python
# check-docs: skip: excerpt from doc/tutorials/00_quickstart.ipynb
class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)
```

代价在入门教程的设置下很清楚（RTX 4090，Qwen2.5-0.5B，最多 32 个并发序列，8 个 prompt × n=4 × 256 token，vLLM 0.27.1）：它只有 4,258 tok/s；同一个补丁进了 CUDA graph 之后是 12,174 tok/s。显式 eager 的对照是 4,034 → 4,263，所以差别几乎全部来自“有没有进 graph”，和加法本身贵不贵没关系。

为什么 hook 进不了 graph，[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)讲过：decode 步是录成 CUDA graph 回放的，回放时 Python 不执行，hook 里的逻辑就不存在了。vLLM 还用 `torch.compile` 编译模型：Dynamo 把 forward 追踪成一张 FX 图，按 attention 切成几段（piecewise），每段按几档 batch 大小录 graph；纯 decode 的 batch 还可以整图录（FULL）。外来的 Python 要么被内联进图——每步会变的东西就被焊死了——要么造成图断点。

所以 `ActivationPatch` 只是开头。真正想做的是一类改残差连接本身的算法：value residual、LAuReL、Hyper-Connections，它们不是“在某层输出上加点东西”，而是把 `x + y` 换掉。

## 第一步：找到那个加法

教科书里的 Transformer 层长这样：

```python
# check-docs: skip: textbook pseudo-code, not vLLM's implementation
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

最自然的想法是：找到那个 `+`，拦截它，换成用户的方法。可你在 vLLM 编译后的图里找不到它。vLLM 里准入的模型家族（Qwen2/2.5/3、Llama、DeepSeek……）都把“子层输出加回残差流”融合进了**下一个 norm**：一个原地的 `fused_add_rms_norm` kernel，`norm(branch, residual) → (normalized, residual + branch)`，加法和归一化一次读写完成。单独的加法节点根本不存在。

这个 kernel 是 vLLM 为了省一次显存往返而做的优化，对我们来说却像一扇关着的门。不过换个角度看，它也正好给出了答案：既然加法藏在下一个 norm 里，**那就挂在 norm 上**。在这个边界上，上一个子层的输出 y 和残差流 x 恰好同时交到 norm 手里——没有改写时照常调用原生 kernel；有改写时，用用户方法的返回值代替 `x + y`，再用模型本来的 norm kernel 归一化。`pllm/backends/vllm/model_sites.py` 的模块 docstring 把这个想法写成了一段话：

> Every admitted family adds each sublayer's output into the residual stream inside the next norm, a fused in-place kernel that returns (normalized branch input, stream). A hook on that norm completes the sublayer with the owning method's write instead of the plain add, keeps copies the methods read later in the forward, forms the next branch input with a read method, and normalizes with the kernel the model uses. Boundaries nothing applies to are not wrapped.

最后一句是性能上的关键：**没人用的边界不包装**。只在第 8、16 层改写，其余几十个 norm 原样不动。装 hook 的代码（`ModelSites._install_boundaries`）一边枚举每个 norm 前后是哪个子层，一边只在真正有人用的边界上替换：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/model_sites.py
for index, (norm, kernel) in enumerate(zip(sublayer_norms(decoder.model), decoder.norms, strict=True)):
    layer, attention = divmod(index, 2)
    before: Place | None = ("attention", layer) if attention else ("mlp", layer - 1) if layer else None
    after: Place | None = None if layer == count else ("mlp", layer) if attention else ("attention", layer)
    ...
    if (
        boundary.owner
        or boundary.reader
        or boundary.final
        or boundary.keeps
        or any(boundary.taps)
        or self.streams > 1
    ):
        boundary.index = len(self.hooks)
        self._install(norm, boundary.graph_forward if graphs else boundary)
        self.hooks.append(boundary)
```

`divmod(index, 2)` 那一行把第 index 个 norm 翻译成“第几层、是 attention 前还是 MLP 前”，`before` 和 `after` 记下它两边是哪个子层；只有改写者、读取者、需要留副本的、挂了 hidden tap 的边界才会被换掉。（要改 V 的算法，比如 value residual，另有一个 `ValueProjection` hook：在 qkv 投影之后、attention 读取和写 KV cache 之前替换 V 那一段。）

这个设计也有代价：它依赖一个结构事实——全模型 2L+1 个 norm 都接受 `(branch, residual)` 并返回 `(normalized, sum)`。所以 pLLM 只准入精确的原生模型类，改过 norm 签名的派生模型会被拒绝（[设计取舍](../design-tradeoffs.md)）。

## 第二步：让 hook 进图——门牌号就是身份证

找到了位置，接下来要让挂在 norm 上的 hook 进入 vLLM 录的 graph。正规的办法是 custom op：用 `torch.library.custom_op` 注册的算子，Dynamo 和 Inductor 不看里面，只按签名把它当作图里的一个不透明节点（你给它一个 fake 实现告诉编译器输出的形状）；录 graph 的时候，盒子里实际发射的 kernel 会被录下来。

问题是这个盒子很挑食：它只收张量和标量。你没法把“当前会话”这个 Python 对象传进去。那传一个整数 `session_id`，进去再查表？也不行——vLLM 编译后的模型图（还会缓存到磁盘）会把标量**烘焙**进去，图就和某一个会话绑死了。

我们的办法是：**用一个张量的地址当身份**。每个会话本来就有一张“行 → 请求”表，是一个在会话存活期间不会释放的 GPU 张量。把它作为 op 的 `reads[0]` 传进去，op 里面用它的 `data_ptr()` 去全局注册表里查会话：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/model_sites.py
# Graph-mode sessions by the address of their row-to-request table; see model_site.
_SITES: dict[int, "ModelSites"] = {}


@torch.library.custom_op("pllm::model_site", mutates_args={"outputs", "writes"})
def model_site(
    inputs: list[Tensor], outputs: list[Tensor], reads: list[Tensor], writes: list[Tensor], hook: int
) -> None:
    """Run a bound hook inside vLLM's capture, with stable state and same-forward scratch ports.

    A custom op takes only tensors and scalars, and vLLM's compiled (and AOT-cached) model graph bakes scalars in,
    so the session travels as `reads[0]`, its row table: a runtime graph input whose address finds the session
    without specializing the graph on one. vLLM routes its own opaque ops through a name registry the same way
    (forward_context.no_compile_layers). A session registers in ModelSites.__init__ and leaves in close(); it
    owns the table meanwhile, so the address cannot be reused while registered.
    """
    _SITES[reads[0].data_ptr()].hooks[hook].emit(inputs, outputs)
```

整个技巧就是最后那一行。对编译器来说，`reads[0]` 只是一个普通的张量输入，图不会因为它特化；对 op 来说，地址是门牌号，门牌号就是身份证。docstring 里交代了让它安全的两件事：会话在 `__init__` 时注册、`close()` 时注销，期间一直持有这张表，地址不会被别的张量复用；而 vLLM 自己的不透明算子也是用一张按名字索引的注册表（`forward_context.no_compile_layers`）做同样的事，我们并不是在发明奇怪的用法。（签名里的整数 `hook` 确实会被烘焙进图，但它表示“这是第几个挂载点”，对同一张模型图来说本来就是固定的。）

每步会变的东西——每行属于哪个请求、每个请求该读哪个状态版本——在回放前由 `ModelSites.prepare` 拷进地址固定的缓冲区，op 只读这些缓冲区。装 hook 的时机也要讲究：必须在 vLLM 编译模型**之前**装好，这样 vLLM 追踪模型时看到的就是带 hook 的 norm，录图时自然把 op 录进去。

用户的方法本身怎么进图？pLLM 不把它交给 vLLM 的编译管线，而是自己用 `torch.compile(fullgraph=True, dynamic=True)` 把它连同“写进输出端口”一起编成一个可调用对象（`compile_site_method`），跨层、跨 batch 大小共享。这样 vLLM 那边可以继续用它自己选的编译后端。另一个方案——把 vLLM 的模型编译后端整体切到 Inductor 来顺带编译用户方法——被我们否决了，因为每个引擎都要多一次重编译。

有了这套机制，最早那个 eager 的 `ActivationPatch` 就没有存在的理由了。2026 年 11 月，我们把它改写成 `custom_residual` 的一层语法糖：要打补丁的行（prompt 的首或末 token、decode 的每个 token）变成设备上的掩码，每个请求的引导向量按设备上的槽号 gather。旧的每请求 Python 实现直接删掉，没有保留双路径。开头那个 4,258 → 12,174 tok/s，就是这次改动带来的。

## 第三步：一个什么也不做的改写，为什么慢

机制能跑了，下一个问题是它有多贵。最好的量法是一个数学上什么也不做的改写：

```python
# check-docs: skip: illustrative identity rewrite, shaped like the benchmark workload
@pllm.custom_residual(site="mlp")
def identity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```

它的吞吐和原生 vLLM 的差距，就是挂载机制本身的开销。下面是这个数在几天里的变化（RTX 4090，Qwen2.5-0.5B BF16，8 个 prompt × n=4 × 256 token，最多 32 个并发序列，FULL_AND_PIECEWISE graph，vLLM 0.27.1，3 轮中位数）：

| 时间 | identity | 相对原生 | 发生了什么 |
|---|---:|---:|---|
| 2026-11-03 之前 | 3,320 tok/s | 25.3% | 任何改写都迫使整个模型 eager |
| 2026-11-03 | 8,810 | 65.8% | 无状态改写通过 `pllm::model_site` 进入 vLLM 的模型 graph |
| 2026-11-05 | 11,819 | 89.8% | 中间若干次宿主侧改进（逐项归因没有单独测）；开始数 kernel |
| 2026-11-06 | 13,076 | ≈99.5%（同期原生 13,142） | 删掉多余拷贝与填零、恒等旁路、把末尾的加法还给原生 norm |

（各行的原生基线略有不同：11-03 那次是 13,135 / 13,381，11-06 是 13,142。）

65.8% 这个数一度被写进了用户文档：“恒等 custom_residual 只有原生的约 66%”。它看起来像一个架构上的天花板。它不是。

11 月 5 日我们不再猜，开始**数 kernel**。先把差距拆开：对一次 8,192 token 的生成，identity 比原生（Inductor 编译后端）慢 70.54 ms，其中 11.92 ms（16.9%）来自模型图换成了源算子、失去了 Inductor 的融合与 kernel 选择，剩下 58.63 ms（83.1%）来自挂载点本身。再数：用源算子录图的原生 vLLM 一共发射 104,837 个 kernel，identity 是 166,277 个——**多出 61,440 个**：37,119 个 memcpy、12,240 个 direct_copy、12,288 个独立的 BF16 add、12,288 个填零，加起来约 52 ms 的 GPU 时间。

账一列出来，原因就很清楚了。每进一次挂载点，为了凑齐输入输出端口，代码都在分配临时张量、来回拷贝、填零；用户写的 `return x + y` 被忠实地录成一个独立的 add kernel，算完再拷进 norm。每一步单独看都合法，乘上层数、乘上每一步，就成了这六万个 kernel。

有一条诱人的捷径：既然大多数改写最后都是个加法，那不管用户写了什么，能用原生融合 norm 就用。它确实快，但相对 FP32 参考的最差 logprob 误差从 0.243 恶化到 0.394（0.243 是原生 BF16 模型自己相对 FP32 的误差）。融合改变了舍入发生的位置。我们否决了它：pLLM 的数值标准是，与 FP32 参考的误差不比原生更差。

最后的做法是先**看懂**用户的方法。`SiteBody._trace_math`（`pllm/backends/vllm/site_body.py`）用 `make_fx` 把方法追踪成一张 FX 图，看输出是不是一个形状、dtype 都和残差一致的 `aten.add`：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/site_body.py
body = make_fx(trace, tracing_mode="fake", _allow_non_fake_inputs=True)(samples, arguments)
output = next(node for node in body.graph.nodes if node.op == "output")
(add,) = output.args[0]
self._bodies[key] = body, False
if self.read or self.rewrite.streams > 1:
    return
...
if not isinstance(add, fx.Node) or add.target != torch.ops.aten.add.Tensor or add.kwargs:
    return
left, right = add.args
if not isinstance(left, fx.Node) or not isinstance(right, fx.Node):
    return
tensors = tuple(tensor_node_metadata(node) for node in (left, right, add))
if any(
    value is None or value.shape != samples[0].shape or value.dtype != samples[0].dtype for value in tensors
):
    return
self.identity = (
    len(samples) == 2
    and {left, right} == set(inputs[:2])
    and len(tuple(body.graph.nodes)) == len(inputs) + 2
    and self._shape_independent(len(samples), arguments)
)
self._store(key, body, output, add, (left, right))
```

如果是，就把这个加法节点从图里删掉，让方法改为输出加法的两个操作数：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/site_body.py
def _store(
    self,
    key: _Key,
    body: fx.GraphModule,
    output: fx.Node,
    terminal: fx.Node,
    operands: tuple[fx.Node, ...],
) -> None:
    set_tuple_output(body, output, operands)
    body.graph.erase_node(terminal)
    body.recompile()
    self._bodies[key] = body, True
```

挂载点拿到这两个操作数，交给模型原本的 `fused_add_rms_norm`——和原生模型一模一样的 kernel、一模一样的舍入。如果方法就是 `x + y` 本身（`self.identity`），挂载点直接旁路，连 op 都不调用。同时删掉端口的临时分配和填零，输出直接写进端口。

效果：decode 图每步的 kernel 从 628 个降到 388 个；整次生成的 kernel 数从 166,277 降到 104,837，和用源算子录图的原生 vLLM 完全相同。identity 到了 13,076 tok/s。同期其他改写：value 11,474 → 11,541，Hyper-Connections 8,569 → 8,692，带状态的记忆子层 7,861 → 8,207。中性设置下 token 和 logprob 与原生逐位相同。

我们最喜欢这一步的方向：没有写一个“既做改写又做归一化”的新 kernel，而是让编译器工具看懂用户的意图，把加法**还给**上游已经写好的 kernel。上游多干一点，我们的代码少一点。

## 7B 上，剩下的 3% 是什么

换到 Qwen2.5-7B（RTX 4090，vLLM 0.32，8 × n=16 = 128 行，256 token，3 轮中位数），原生 vLLM 是 5,894 tok/s。只在第 12 层挂 identity 是 5,722 tok/s（97%）；每一层的 MLP 都挂上，是 5,721 tok/s（还是 97%，模型 graph 池需要约 1.13 GB）。

挂 1 层和挂 28 层几乎一样，说明这 3% 不是按挂载点累加的，而是一笔固定的费用。它买的是逐位中性：挂了 custom_residual 的模型，vLLM 录的是模型的**源 CUDA 算子**（vLLM 的 `"eager"` 编译后端），而不是 Inductor 融合后的 kernel。为什么不用 Inductor？因为 Inductor 对挂了 hook 的图和原生参考图融合 add/RMSNorm 的方式不一样，舍入就不一样，逐位中性就保不住。0.5B 上的对照是：仅仅把原生 vLLM 的模型图从 Inductor 换成源算子，吞吐就从 13,158 降到 12,911（1.9%）。我们拿这一点吞吐换了“中性设置与原生逐位相同”这条保证——[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)里，同样的取舍后来扩展到了自定义注意力。

示例 07 在 7B 上（32 个并发序列，256 token）：value residual 是原生图模式的 99%，Hyper-Connections（n=4）93%；带状态的记忆子层（每个请求一个 3584 × 3584 的 FP32 矩阵，约 49 MiB，只开 8 个并发序列）是同样 8 个序列原生的 75%，模型 graph 池约需 1.9 GiB。这个设置下原生 vLLM 的 eager 模式也有图模式的 99%（一步约 19 ms），所以 7B 上“进图”本身挽回不了多少；在 0.5B 上，同样的改写不进图只有 12%–34%。

## 做不到的事

挂在 norm 上这个想法，也决定了它够不着的地方。`custom_residual` 要求 `tensor_parallel_size=1`。改写之后相同的 token 前缀不再产生相同的 KV，所以 prefix caching 必须关掉（[设计取舍](../design-tradeoffs.md)）。CPU 上没有模型，`pllm.testing.check` 会跳过模型内的方法，只能靠 `check_on_gpu`（[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）。只准入精确的原生模型类，这是“依赖 norm 签名”的直接代价。只要挂了改写，整张模型图就失去 Inductor 融合，7B 上约 3%、0.5B 上约 2%。

算法自己的计算也省不掉。Hyper-Connections 把残差流扩成 4 条，多出来的矩阵乘法和带宽是真实存在的：7B 上 93%，0.5B 上只有约 66%。graph 和挂载点只能消掉框架的开销。有状态的子层还很吃 graph 池，7B 上约 1.9 GiB。它早期甚至进不了图：主机侧的 span 把每个请求绑到本步的私有页上，录下来的图会记住那一次的几何形状；后来改成设备索引的私有槽、有界的分桶调度，才进了图。

和别的工具比，取舍就更清楚了。HF hook、TransformerLens、nnsight、pyvene 易用、表达力强，是研究的好工具，但它们面向单请求或静态 batch，没有连续批处理，也没有 CUDA graph。在 vLLM 上试一个新的残差结构，据我们所知常见的做法是 fork 模型文件——直接有效，代价是每个算法一份 fork，每次升级 vLLM 都要自己合并。

## 可以带走的东西

回头看，这套机制的三个关键想法都是“顺着已有的结构走”：加法藏在 norm 里，就挂在 norm 上；custom op 只收张量，就用张量的地址当身份；原生 norm 会做加法，就把加法还给它。没有一处是在和 vLLM 或 PyTorch 对着干。

还有一个更朴素的教训：65.8% 看起来像天花板，是因为我们还没有去数。性能问题在被数清楚之前，最容易被当成“架构的固有代价”写进文档。
