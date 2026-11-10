# 图内挂载点：在 CUDA graph 里改残差

做模型可解释性或对齐研究的人，大都写过“激活引导（Activation Steering，例如 CAA 或 ActAdd）”这类实验：在模型第 12 层的输出上加一个方向向量 $h \leftarrow h + \alpha \cdot v$，观察模型生成行为的变化。

在单卡研究脚本里，这几乎算不上一个工程问题。往 `model.layers[12]` 上挂一个 PyTorch 的 `register_forward_hook`，写三行代码就能跑起来。在 pLLM 中，对应的声明同样只有两行：

```python
# check-docs: skip: excerpt from doc/tutorials/00_quickstart.ipynb
class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)
```

但如果你在底层按照研究代码的习惯去实现这个功能——每走一步，就在 Python 回调里挑出当前批次需要打补丁的行、上传行索引、从状态池里取出引导向量再做一次加法——放到生产级的 vLLM 引擎里一测，性能会掉得让人怀疑人生。

在单张 RTX 4090 上跑 Qwen2.5-0.5B（BF16，vLLM 0.27.1，8 条 prompt $\times$ `n=4` 共 32 条并发序列，生成 256 个 token），这种逐请求 Python 回调的实现只能跑出 4,258 tok/s；而将同一段逻辑真正放进模型的 CUDA Graph 之后，吞吐是 12,174 tok/s。仅仅在第 12 层做了一次向量加法，图外实现就丢掉了将近三分之二的吞吐。

为什么一次复杂度可以忽略不计的向量加法会这么贵？如果把两种实现都强制锁死在关闭 CUDA Graph 的 Eager 模式下重测一遍，答案就清楚了：旧实现是 4,034 tok/s，新实现是 4,263 tok/s，几乎没有差别。向量加法本身并不花时间；真正拖垮吞吐的，是因为 Python 回调无法进入 CUDA Graph，迫使整座模型退回了逐算子发射的 Eager 模式。

我们在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里已经算过 CUDA Graph 的账：解码阶段每一步只有几十微秒到几毫秒，回放时 GPU 自己顺着录制好的静态指令图往下跑，Python 解释器根本不在场。更棘手的是，vLLM 会用 `torch.compile` 编译整个模型——Dynamo 把模型的 `forward` 追踪成 FX 计算图，按注意力层切成若干段（piecewise），每段按不同批次档位录制 CUDA Graph；到了纯解码阶段，还会把整座模型合成一张大图（FULL）一口气录下来。在这个高度静态化的管线里，外来的 Python 逻辑要么被 Dynamo 当成常数死死内联进图里（每步变化的请求行号与状态直接被焊死），要么把计算图撞出断点。

而且，很多算法要做的事情远不止在某一层上“加一个固定向量”。像 Value Residual（把第 0 层的注意力 Value 掺进后续各层）、LAuReL，以及把单条残差流拓宽成 $n$ 条并行车道的 Hyper-Connections，它们需要深入 Transformer 内部，把残差连接里的那个 `+` 号本身替换成自定义的张量函数（`@pllm.custom_residual`）。

要想在不破坏模型编译与 CUDA Graph 的前提下改写任意一层的残差，并且把框架自身的额外开销压到接近于零，我们得依次解开三个问题：钩子到底挂在哪儿、怎么跨过编译器的边界进入图内、以及那些看不见的微秒级开销究竟藏在哪里。

---

## 一、 寻找消失的 `+` 号：为什么挂载点选在 Norm 上

翻开任何一本深度学习教材，里面的 Transformer 层都是这样写的：

```python
# check-docs: skip: textbook pseudo-code, not vLLM's implementation
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

照着这两行代码想，最自然的方案似乎是：在计算图里找到那个 `+` 号对应的节点，把它拦截下来，替换成用户写的 `@pllm.custom_residual` 函数。

但如果你打开 vLLM 实际运行时的模型代码去翻，会发现图里根本没有独立的 `+` 号。

原因在于显存带宽。在解码阶段，GPU 的瓶颈几乎全在显存读写（HBM bandwidth）上。如果先跑一个独立的加法算子 `x = x + y` 把结果写回显存，紧接着下一层的 `RMSNorm` 再从显存里把 `x` 读出来做归一化，这一写一读就白白浪费了一趟显存往返。因此，在 vLLM 中，pLLM 准入的模型家族（Qwen2、Qwen2.5、Qwen3、Llama、DeepSeek 等）都使用了 `fused_add_rms_norm` 原地融合算子：它把“上一子层的输出加回残差流”和“下一子层入口的归一化”合并在同一个 CUDA 内核里，一次读取同时返回 `(normalized, residual + branch)`。

起初，这看起来是个坏消息：残差加法被藏进了底层的 C++/CUDA 融合内核里，Python 层似乎连下手的接缝都没有。

但仔细看一眼数据流就会意识到，这个融合设计反而替我们把全模型最干净的拦截位置圈了出来：

```text
Textbook Transformer:
  stream x ──────────────► (+) ─────────────────► [ norm2 ] ──► mlp_in
  attn_out y ───────────────▲

vLLM Reality (and where pLLM hooks in):
                            ┌─── SublayerBoundary (hooks norm2) ───┐
  stream x ────────────────►│                                      │──► updated stream
                            │  user write(x, y) ──► native norm2   │
  attn_out y ──────────────►│                                      │──► normalized mlp_in
                            └──────────────────────────────────────┘
```

正因为上一个子层（比如 Attention）末尾没有自己做加法，当执行流走到下一个归一化层（比如 `norm2`）门口时，**上一个子层的分支输出 `y`（`branch`）和主干道上的残差流 `x`（`residual`），恰好完好无损地同时交到了这个 Norm 的入口参数上。**

既然残差加法发生在下一个 Norm 里，最顺理成章的做法就是直接把钩子挂在 Norm 模块上。

在 `pllm/backends/vllm/model_sites.py` 中，每个这样的交界处被称为一个 `SublayerBoundary`。一个 $L$ 层的 Transformer 共有 $2L+1$ 个 Norm（每层两个，外加最末尾的 `model.norm`）。安装边界钩子时，只需用一次整除与取模，就能把线性编号 `index` 还原成它在模型里的空间坐标：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/model_sites.py
for index, (norm, kernel) in enumerate(zip(sublayer_norms(decoder.model), decoder.norms, strict=True)):
    layer, attention = divmod(index, 2)
    before: Place | None = ("attention", layer) if attention else ("mlp", layer - 1) if layer else None
    after: Place | None = None if layer == count else ("mlp", layer) if attention else ("attention", layer)
    # ……
```

当 `divmod(index, 2)` 的余数为 1 时，这个 Norm 夹在第 `layer` 层的 `attention` 之后、`mlp` 之前；余数为 0 时，它夹在上一层的 `mlp` 之后、当前层的 `attention` 之前。当某个边界被挂载时，它先调用用户的改写方法完成上一子层的残差合成（如果需要供后续层跨层读取，顺手保留一份副本；如果是 Hyper-Connections 这类多流残差，再调用读取方法合成下一子层的输入），最后调用模型原本的那个 `norm` 内核完成归一化。

同样重要的是：**没有任何方法触碰的边界，完全不做包装。** 如果用户只声明了 `layers=(8, 16)`，其余几十个 Norm 依然保持 vLLM 原生的模块引用，不增加任何包装层或函数跳转。至于像 Value Residual 那样需要在注意力计算前直接修改 Value 的场景，则通过同类机制在 QKV 线性投影出口挂载 `ValueProjection`，在 Value 切片进入注意力算子和写入 KV Cache 之前原地改写它。

借用 `fused_add_rms_norm` 的入口，也决定了模型准入上的一条硬边界：全模型 $2L+1$ 个 Norm 必须严格遵循接受 `(branch, residual)` 并返回 `(normalized, sum)` 的签名契约。如果有人在外部继承了 `Qwen2ForCausalLM` 的自定义子类，改动了某个 Norm 的参数约定，挂载点就会在错误的张量上读写。因此，框架只准入白名单上精确匹配的原生模型类（`type(model) in architectures`），对派生子类在构建期直接报错拒绝——用入口处的严格契约，换取内部循环的确定性。

---

## 二、 跨越编译器边界：拿张量的显存地址当会话身份

找准了挂钩子的模块，下一步是让挂载点顺畅地进入 `torch.compile` 和 CUDA Graph。

在 PyTorch 2 的编译体系里，要想把一段自定义操作放进 Dynamo 追踪的图里而不引发图断裂（graph break），标准途径是将其注册为不透明的自定义算子 `@torch.library.custom_op`，并配一个只推导形状的 `register_fake` 实现。Dynamo 和 Inductor 不会拆解算子内部，只按签名（哪些输入只读、哪些输出被原地修改）把它当成图里的一个节点；等到 vLLM 录制 CUDA Graph 时，算子内部实际发射的 GPU 内核就会被悉数录进图里。

但动手定义这个 `@torch.library.custom_op("pllm::model_site", ...)` 时，立刻会遇到一个接口限制：`custom_op` 的参数列表只接受 `Tensor` 和基本标量（`int`、`float` 等），不接受自定义的 Python 对象。

而在算子内部执行时，我们必须找到当前引擎会话对应的 `ModelSites` 实例——因为编译好的方法体、钩子列表以及每步刷新的状态缓冲区都挂在这个对象上。

直觉上，既然不能传 Python 对象，传一个整数标量 `session_id: int` 去全局字典 `_SITES: dict[int, ModelSites]` 里查表行不行？

在单次运行的脚本里可以，但在 vLLM 里不行。vLLM 编译后的模型图是跨会话复用的（并且支持通过 AOT 编译缓存持久化）。在 Dynamo 眼里，传给算子的任何 `int` 标量都会被视为编译期常量，直接**烘焙（bake in）**进生成的图代码里。如果第一个会话的 ID 是 `1`，编译出的图就会把 `1` 焊死；等第二个会话进来复用同一张图时，算子依然会去查 `1` 号会话的数据。

不能传 Python 对象，传整数标量又会被焊死成常量，唯一能作为运行时变量传进算子、且不会触发图特化的，就只剩下 **`Tensor`** 了。

而一个驻留在显存里的 `Tensor`，本身就自带一个进程内唯一、且在生命周期内绝不改变的 64 位整数——它在显存里的物理首地址 `tensor.data_ptr()`。

每个 `ModelSites` 会话在初始化时，本来就要分配一张名为 `_requests_of` 的一维 GPU 张量（用来存放当前步每一行属于哪个请求）。只要会话尚未关闭，它就一直持有这张张量，CUDA 显存分配器绝不会把这个地址分给别的张量。于是，我们把 `_requests_of` 放在只读张量列表的第一项 `reads[0]` 传进算子。对编译器来说，`reads[0]` 只是一个普通的运行时输入张量；而在算子内部，调一次 `reads[0].data_ptr()` 就能直接从全局注册表里取回当前会话：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/model_sites.py
_SITES: dict[int, "ModelSites"] = {}


@torch.library.custom_op("pllm::model_site", mutates_args={"outputs", "writes"})
def model_site(
    inputs: list[Tensor], outputs: list[Tensor], reads: list[Tensor], writes: list[Tensor], hook: int
) -> None:
    # ……
    _SITES[reads[0].data_ptr()].hooks[hook].emit(inputs, outputs)
```

签名末尾的 `hook: int` 标量确实会被烘焙进图里，但这恰恰符合预期——它代表的是“当前是模型里的第几个挂载点”，在固定的模型拓扑里，第 3 个钩子永远是第 3 个钩子。真正随会话改变的身份，则跟着运行时张量 `reads[0]` 的显存地址走了进来：

```text
Compiled vLLM Model Graph (Reusable across sessions):
  ┌──────────────────────────────────────────────────────────────────┐
  │  ... ──► pllm::model_site(inputs, outputs, reads, writes, hook=3)│
  └──────────────────────────────────────────────┼───────────────────┘
                                                 │ reads[0] is _requests_of tensor
                                                 ▼
                                       reads[0].data_ptr() ──► 0x7f8a00200000
                                                                     │
  _SITES Registry (Python):    { 0x7f8a00200000 : ModelSites } ◄─────┘
                                                       │
                                                       ▼
                                         hooks[3].emit(inputs, outputs)
```

会话在 `__init__` 时将 `_requests_of.data_ptr()` 登记进 `_SITES`，在 `close()` 时移除（vLLM 自己内部那些不参与 Dynamo 编译的不透明层，例如注意力层，也是通过 `forward_context.no_compile_layers` 注册表以同样思路寻址的）。每一步模型图回放前，主机侧的 `ModelSites.prepare` 把当前步每行对应的请求编号拷进 `_requests_of`，并从 PTTS 状态池（见[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)）把各请求当前版本的状态拷进固定地址的 `state_values` 缓冲（若读取的是旁路流提前查表的结果，则在预取流上异步拷贝并记录 CUDA Event，与[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)衔接）。图回放时，算子只读这些地址稳定的缓冲。

至于用户编写的残差改写函数本身，则由 `compile_site_method` 独立编译：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/site_body.py
def compile_site_method(rewrite: ResidualRewrite, *, read: bool = False) -> Callable[..., None]:
    """Share one fullgraph callable across a method's depths and model capture sizes."""
    function = rewrite.read if read else rewrite.write
    assert function is not None
    inputs = rewrite.read_inputs if read else rewrite.inputs
    return compile_math(partial(_method_ports, function, tuple(name for name, _ in inputs)), dynamic=True)
```

这里故意没有把整个 vLLM 模型的编译后端切到 Inductor 去顺带编译用户方法——那样做的话，每当用户微调一行实验代码，整座几十层的模型都要跟着重新编译一遍。用 `compile_math`（内部为 `torch.compile(fullgraph=True, dynamic=True, ...)`）把用户函数连同输出端口写入单独编译成一个动态形状的可调用对象，就能在所有挂载层和所有 CUDA Graph 批次档位之间直接共享。

有了这套图内机制，文章开头的 `ActivationPatch` 就不再需要维护任何独立的 Python 路径了。它直接降级成了 `@pllm.custom_residual` 的一层语法糖：需要打补丁的位置（如 prompt 的首尾 token，或 decode 的每个 token）编码为设备端的一维布尔掩码，每个请求的向量按设备端槽号 gather 出来做掩码加法。开篇那组 4,258 对 12,174 tok/s 的对比，就是同一个 `ActivationPatch` 走图外回调与走这条图内路径的差别。

---

## 三、 消失的三分之一吞吐：数出多出来的 61,440 个内核

当无状态残差改写通过 `pllm::model_site` 进入 vLLM 的 CUDA Graph 后，按理说 CPU 发射瓶颈已经被消除了。为了测量纯框架层面的底噪，我们可以写一个在数学上什么都不做的恒等改写（`identity`）：

```python
# check-docs: skip: illustrative identity rewrite, shaped like the benchmark workload
@pllm.custom_residual(site="mlp")
def identity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```

在同样的 RTX 4090、Qwen2.5-0.5B 基准（32 并发序列 × 256 token，vLLM 0.27.1）下，当改写被迫跑在图外 Eager 模式时，`identity` 的吞吐是 3,320 tok/s，仅为原生 vLLM 的 25.3%。最直接的进图实现跑出了 8,810 tok/s（占同期原生 13,381 tok/s 的 65.8%）；再去掉一部分外围宿主开销，吞吐升到 11,819 tok/s（按 Inductor 编译的原生为 13,158 tok/s，占 89.8%）。

无论是 65.8% 还是 89.8%，仔细想一想都很不对劲：`return x + y` 做的数学运算和原模型完全一样，而且已经录进了 CUDA Graph，为什么还会凭空慢掉 10% 到 34%？

靠直觉猜瓶颈往往会猜错，最稳妥的办法是打开 profiler，把一次完整生成（8,192 个输出 token）的 GPU 轨迹录下来，直接拆解时间并数一数 GPU 到底执行了多少个内核。

剖析器给出的时间拆解非常明确：在这次 8,192 token 的生成中，挂了 `identity` 的模型比原生（Inductor 编译后端）总共慢了 **70.54 ms**。其中：
* **11.92 ms（占差距的 16.9%）** 来自模型编译后端的切换——挂载自定义算子后，模型图改用源算子（eager backend）捕获，少了 Inductor 的跨算子融合；
* **58.63 ms（占差距的 83.1%）** 完完全全发生在我们的挂载点内部。

接着看内核发射总数。同样使用源算子录图的原生 vLLM，跑完这 8,192 个 token 一共发射了 **104,837 个内核**；而挂了 `return x + y` 的模型，发射了 **166,277 个内核**。

整整多出了 **61,440 个内核**。把这六万多个内核按名字分类列出来，问题一目了然：
* **37,119 个 `memcpy` 与 12,240 个 `direct_copy`**：每次进出 `pllm::model_site`，代码都在把输入输出张量拷进临时暂存区（staging buffer），算完再拷回模型端口；
* **12,288 个填零（zero fill）**：每次准备输出端口时，代码都先调用一次清零，哪怕几微秒之后这块内存就会被计算结果完整覆盖；
* **12,288 个独立的 BF16 `add`**：这是最不该发生的一笔开销。

回想一下第一节的结构：原生 vLLM 的 `fused_add_rms_norm` 本来就能接收 `(branch, residual)` 两个输入，在同一个内核里顺手做完加法和归一化。但因为用户在 Python 里写了句 `return x + y`，挂载点就老老实实地先发射一个独立的 BF16 加法内核算出和 `sum` 写回显存，然后再把 `(sum, 0)` 传给原生的 Norm 内核去归一化！

每一次小拷贝、填零和独立加法只有几微秒，但乘上 24 层和几百个生成步，这 61,440 个多余内核足足吃掉了约 52 ms 的纯 GPU 时间。

找到病根后，怎么把那 12,288 个独立加法消掉？

一个看似省事的捷径是：不管用户在改写方法里写了什么，总是无条件拆分并走原生融合 Norm 的双输入路径。但实测表明这条路走不通：相对于 FP32 黄金参考基准，原生 BF16 模型自身的最大对数概率（logprob）误差是 **0.243**；而无条件硬塞进融合 Norm 会改变非标准算式的中间浮点舍入位置，使最差 logprob 误差恶化到 **0.394**。为了守住“数值误差不劣于原生模型”的底线，不能盲目硬塞。

正确的做法是先用符号追踪看清用户方法的结构：只有当用户方法的最后一步**确实是一个形状与类型完全匹配的张量加法**时，才把这个末尾加法剥离出来，交还给原生的融合 Norm。

在 `pllm/backends/vllm/site_body.py` 的 `SiteBody._trace_math` 中，这件事由 `make_fx` 完成：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/site_body.py
body = make_fx(trace, tracing_mode="fake", _allow_non_fake_inputs=True)(samples, arguments)
output = next(node for node in body.graph.nodes if node.op == "output")
(add,) = output.args[0]
# ……
if not isinstance(add, fx.Node) or add.target != torch.ops.aten.add.Tensor or add.kwargs:
    return
left, right = add.args
# ……
self._store(key, body, output, add, (left, right))
```

在绑定挂载点时，`make_fx` 用假张量（fake tensors）把用户方法追踪成一张 FX 计算图，顺着输出节点 `output` 往回看一步：
1. 如果最后一个节点 `add` 恰好是标准的 `aten.add.Tensor`，且左右操作数 `left`、`right` 与输出具有相同的残差形状和数据类型，代码还会顺手检查它是不是纯粹的恒等函数 `return x + y`（即图里除了输入占位符、这一个加法和输出外没有别的节点，且经 `fx.symbolic_trace` 确认没有依赖形状的分支）。一旦确认为纯恒等改写，外层直接将 `self.owner` 置为 `None`，回放时连 `pllm::model_site` 都直接旁路跳过。
2. 如果用户在加法之前确实做了别的运算（例如 `return x + 0.95 * y`），就调用 `_store` 做一次图手术：

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

这四行代码把 FX 图的输出改成了加法之前的两个操作数 `(left, right)`，然后调用 `body.graph.erase_node(terminal)` 把末尾那个 `add` 节点从图里剪掉并重编译：

```text
Before Surgery (61,440 extra kernels across 8,192 tokens):
  x, y ──► [ scale: 0.95 * y ] ──► [ standalone BF16 add ] ──► HBM ──► [ rms_norm ]

After make_fx Surgery (Terminal add snipped & handed back to upstream):
  x, y ──► [ scale: 0.95 * y ] ──► (x, 0.95 * y) ──► [ native fused_add_rms_norm ]
              (erase_node(add))                       (exact same kernel & rounding!)
```

剪掉末尾加法后，用户方法算出的两个操作数直接写进模型声明的端口，原封不动地交给 vLLM 原生的 `fused_add_rms_norm` 去完成相加与归一化。这样既无需手写任何新的融合内核，又保证了浮点舍入行为与原生模型完全一致。

配合去掉多余的端口暂存拷贝与无脑填零，单步 decode 图里的内核数直接从 **628 个** 降到了 **388 个**；整次 8,192 token 生成的内核总数从 **166,277 个** 降回了 **104,837 个**——与使用源算子录图的原生 vLLM 完全相等。

把这几个阶段的实测数字放在一起（RTX 4090，Qwen2.5-0.5B BF16，32 并发序列 × 256 token，vLLM 0.27.1）：

| 机制状态 | `identity` 吞吐 | 相对同期原生 | 底层实际发生的变化 |
|---|---:|---:|---|
| 1. 图外 Eager 回调 | 3,320 tok/s | 25.3% | 残差改写打断图捕获，迫使整座模型退回逐算子 Eager 执行 |
| 2. 直接进图 (`pllm::model_site`) | 8,810 tok/s | 65.8% | 通过不透明 custom op 进入模型 CUDA Graph（同期原生 13,381 tok/s） |
| 3. 去掉部分宿主开销（剖析基线） | 11,819 tok/s | 89.8% | 打开 profiler 数出每次生成仍多出 61,440 个拷贝、填零与独立加法内核 |
| 4. 零拷贝端口 + `make_fx` 剪除末尾加法 | **13,076 tok/s** | **≈99.5%** | 消除暂存拷贝与填零、恒等旁路、末尾加法交还原生融合 Norm（同期按源算子录图的原生 13,142 tok/s） |

零拷贝端口与末尾加法交还也让真实的非恒等改写受益：在相同 0.5B 设置下，Value Residual 从 11,474 升至 11,541 tok/s，Hyper-Connections 从 8,569 升至 8,692 tok/s，带状态的记忆子层从 7,861 升至 8,207 tok/s；且在中性参数设置下，生成的 token 与 logprob 均与原生模型逐位相同（bitwise identical）。

---

## 四、 7B 模型上的实测账单与固有取舍

在步长只有 3 ms 左右的 0.5B 小模型上，任何多余的内核或拷贝都会被放大。当我们把测试台切换到 **Qwen2.5-7B-Instruct**（单张 RTX 4090 24 GiB，BF16，vLLM 0.32.0，开启 CUDA Graph 与 FlashInfer 采样器，8 条 prompt $\times$ `n=16` 共 128 条并发序列，生成 256 token，取预热后 3 轮中位数）时，单步耗时拉长到了 21.7 ms，原生 vLLM 的图模式吞吐为 **5,894 tok/s**。

在这个 7B 基准下测试 `@pllm.custom_residual` 的恒等改写：
* 仅在第 12 层挂载 1 处 `identity`：吞吐为 **5,722 tok/s（原生的 97%）**；
* 将全模型 28 层的每一个 MLP 残差全部挂满 `identity`：吞吐为 **5,721 tok/s（同样是原生的 97%）**，在最大批次 token 数 8,192 下，模型图捕获池占用约 1.13 GB 显存。

挂 1 层是 5,722 tok/s，挂满 28 层是 5,721 tok/s。这说明单个挂载点的执行开销已经小到了完全淹没在测量噪声里的程度。那么，挂 1 层时就已经存在的那 **3%** 固定差距是从哪儿来的？

答案写在 `pllm.explain` 的一行说明里：`model graphs capture source operators, not inductor fusion`。

只要工作负载启用了 `@pllm.custom_residual`，pLLM 就会指示 vLLM 在录制 CUDA Graph 时直接捕获模型的**源算子**（对应 vLLM 的 `"eager"` 编译后端），而不启用 Inductor 的跨算子二次融合。除了避免每次改动负载都重编译整座大模型之外，更关键的原因是数值可复现性：当计算图里插入了不透明的 `pllm::model_site` 节点后，Inductor 对周围算子的重排与融合边界会发生变化，进而改变 BF16 的浮点舍入。在 0.5B 模型上单独测量这笔“关闭 Inductor 融合、仅录源算子”的固有差价，原生 vLLM 自身会从 13,158 降到 12,911 tok/s（相差约 1.9%）；7B 上没有单独测过这笔差价，但那固定的 3% 最可能的主要来源就是这张门票。作为交换，研究者拿到了一个极其干净的性质：只要把算法超参数拨回中性值，改写后的模型与同样按源算子录图的原生基准保证逐位一致（在[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)与[《复用上游：kernel、管理器与原语》](09-reusing-upstream-kernels.md)中，你会看到同样的取舍）。

当然，挂载机制本身的底噪归零，不代表算法自己新引入的数学计算不花时间。在同一台 RTX 4090 上用 Qwen2.5-7B 实测三种真实架构改写（示例 07，生成 256 token）：
1. **Value Residual**（改写第 1 至 27 层的注意力 Value，32 并发序列）：吞吐为原生图模式的 **99%**。它只在 QKV 投影后做轻量级的线性插值，几乎不增加额外带宽。
2. **Hyper-Connections**（$n=4$ 条并行残差流，覆盖全模型 56 个子层边界，32 并发序列）：在 7B 上为原生的 **93%**（在主干很轻的 0.5B 上、vLLM 0.27.1 下则只有约 **66%**）。把单条残差拓宽成四条车道、每进出一个子层都做混合矩阵乘，实打实地增加了显存读写；模型越大，主干权重读取占的分母越大，这笔多流残差的相对占比就越小。
3. **带状态的记忆子层**（每个请求独立维护一个 $3584 \times 3584$ 的 FP32 快权重记忆矩阵，单请求状态即占约 49 MiB，故并发序列数设为 8）：吞吐为同配置原生图模式的 **75%**，录制模型图需约 **1.9 GiB** 的捕获池预算（0.5B 下 identity、value、HC、memory 的捕获池开销分别为 52 MiB、46 MiB、72 MiB 和 448 MiB；若显存预算不足，构建期会直接报错并给出所需的精确字节数）。这类有状态子层之所以也能录进 CUDA Graph，靠的是 `SiteMemorySchedule` 不在主机端按请求切片，而是用**设备端索引、有界压实的请求分桶**，在固定地址上直接更新 PTTS 私有页。

（值得一提的是，在 7B、32 并发下，由于单步矩阵乘本身长达近 19 ms，原生 vLLM 即使关掉 CUDA Graph 也能跑出图模式的 99%；但在 0.5B 下（vLLM 0.27.1 的较早测量），一旦这三种改写掉出 CUDA Graph 退回 Eager，吞吐就会从 92%/74%/60% 暴跌到 12%–34%。）

最后，使用这套图内挂载点有几条明确的工程边界：
* **单卡与前缀缓存互斥**：目前 `@pllm.custom_residual` 仅支持 `tensor_parallel_size=1`。同时，因为模型中途改写了残差或 Value，相同的 token 前缀在不同请求或不同状态版本下会算出不同的隐藏态与 KV，vLLM 的 Prefix Caching 必须关闭。
* **真机验证与控制流隔离**：CPU 上的 `pllm.testing.check` 不加载几十 GB 的模型权重，因此不会执行模型内部的挂载点（在检查结果里记为 `SKIP`），需要调用 `pllm.testing.check_on_gpu` 在 GPU 上做端到端的隔离与逐位中性验证（见[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）。此外，带 `state=` 的有状态记忆子层要求 token 按固定顺序流经挂载层，不能与[《自定义层循环与状态传播》](10-custom-layer-loops.md)中的 `@pllm.custom_forward` 动态跳层混用。

如果只是在单卡上对一两个句子做探索性干预，Hugging Face 的 forward hook、`TransformerLens`、`nnsight` 或 `pyvene` 依然是最顺手的工具；而如果想在生产级 vLLM 上跑出满吞吐的架构改写，过去往往只能把上游的 `qwen2.py` 或 `llama.py` 复制出来硬改源码。`ModelSites` 做的事情，无非是顺着上游已有的结构借力：趁着残差加法被融进下一个 Norm 的瞬间挂载边界，借用常驻张量的 `data_ptr()` 跨过自定义算子的常量烘焙限制，再用 `make_fx` 把用户方法末尾的加法剪下来还给原生的 `fused_add_rms_norm`。而在性能追查里，当一个空操作跑出 66% 甚至 90% 的吞吐时，不要急着把它当成“扩展框架的必然代价”写进文档——打开剖析器去数一数内核总数，那些偷走性能的元凶，往往就藏在几万次看不见的填零和暂存拷贝里。
