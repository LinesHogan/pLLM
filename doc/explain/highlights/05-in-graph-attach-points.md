# 图内挂载点：在 CUDA graph 里改残差

做模型可解释性或对齐研究的人，大都写过一种叫“激活引导（Activation Steering，比如 CAA 或 ActAdd）”的小实验：在模型第 12 层的输出上加一个方向向量 $h \leftarrow h + \alpha \cdot v$，原本死板的助手就会突然变得幽默、严谨或者诚实。

在 Hugging Face 的 `transformers` 里，这甚至算不上一个工程问题。你往 `model.layers[12]` 上挂一个 PyTorch 的 `register_forward_hook`，写三行代码，泡杯咖啡的工夫就能看到结果。

但如果你试着把这三行代码搬进生产级的 vLLM 引擎里，你会一脚踏进“Eager 模式的惩罚室”。

在 pLLM 的早期版本里，我们写过一个叫 `ActivationPatch` 的组件。用户这边的声明看着倒是一如既往的优雅：

```python
# check-docs: skip: excerpt from doc/tutorials/00_quickstart.ipynb
class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)
```

而在引擎盖底下，初代 `ActivationPatch` 老老实实地模仿了你在研究代码里写的 forward hook：每跑一步，就在 Python 里挑出当前批次里需要打补丁的行、上传行索引、从状态池里借出引导向量，然后做一次向量加法。

结果怎么样呢？在入门教程的 Qwen2.5-0.5B 基准下，挂上这三行逻辑后，吞吐只有 **4,258 tok/s**——后面你会看到，同样的补丁本可以跑到 **12,174 tok/s**，这里足足丢了将近三分之二。

> **实验环境与基准约定**：除非另有说明，本文的 7B 性能数据均在单张 NVIDIA RTX 4090（24 GiB）、Qwen2.5-7B-Instruct（BF16）、vLLM 0.32.0、CUDA Graph 开启、`VLLM_USE_FLASHINFER_SAMPLER=1` 下测得，取预热 1 轮后的 3 轮中位数（标准负载为 8 个 prompt $\times$ `n=16` = 128 个并发序列，生成 256 token）。小模型对比与历史追查数据基于同一张 RTX 4090 上的 Qwen2.5-0.5B（BF16，8 个 prompt $\times$ `n=4` = 32 个并发序列，生成 256 token，vLLM 0.27.1）。

为什么一次微不足道的向量加法会收走三分之二的吞吐？如果把新旧两版实现都显式锁死在 Eager 模式下重测一遍，谜底就揭晓了：旧实现是 4,034 tok/s，后来的新实现是 4,263 tok/s，相差无几。补丁里那点向量加法本身几乎是免费的；真正杀死吞吐的，是因为那个 Python hook 进不了 CUDA Graph，逼着整座模型退回了逐算子发射的 Eager 模式。当你在 `pllm.explain` 里查看执行计划时，会看到一行冷酷的提示：`activation patches run outside CUDA graphs`。

我们在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里已经见识过 CUDA Graph 的脾气了：decode 步就像一盘事先录好的自动钢琴磁带，回放时 GPU 自己顺着磁带往下弹，Python 解释器根本不在场。更棘手的是，vLLM 还会用 `torch.compile` 编译整个模型——Dynamo 把模型的 `forward` 追踪成一张 FX 计算图，按注意力层切成若干段（piecewise），每段按不同的 batch 档位录制 CUDA Graph；到了纯 decode 阶段，甚至会把整张模型合成一张大图（FULL）一口气录下来。

在这个高度静态化的世界里，任何外来的 Python 逻辑都面临两难：要么被 Dynamo 当成常数死死内联进图里（每步会变的行号和请求状态直接被焊死），要么把完整的计算图撞出断层。

而且，我们的野心远不止在某一层上“加一个向量”。看看近两年涌现的架构变体——Value Residual（把第 0 层的注意力 Value 掺进后面每一层）、LAuReL、以及把单条残差流拓宽成 $n$ 条并行车道的 Hyper-Connections——它们要干的事情不是在旁边贴块膏药，而是**撬开 Transformer 的骨架，把残差连接里的那个 `+` 号本身给换掉**。

怎样在不拆掉 vLLM 编译管线和 CUDA Graph 的前提下，深入模型内部去改写任意一层的残差？让我们像侦探一样，一关一关把这件机械装置搭出来。

---

## 第一关：寻找消失的 `+` 号

如果你翻开任何一本深度学习教材，里面的 Transformer 层一定长这样：

```python
# check-docs: skip: textbook pseudo-code, not vLLM's implementation
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

看着这两行伪代码，最直觉的方案脱口而出：“找到那个 `+` 号对应的节点，把它拦截下来，替换成用户写的 `@pllm.custom_residual` 函数，不就完事了吗？”

我最初也是这么想的。直到我打开 vLLM 跑起来的计算图，拿着放大镜从头找到尾——**图里根本没有那个 `+` 号。**

它去哪儿了？

在 GPU 上跑大模型，decode 阶段是极度受限于显存带宽（memory-bandwidth bound）的。如果老老实实先算一个独立的加法 `x = x + y` 把结果写回显存（HBM），紧接着下一层的 `RMSNorm` 再从显存里把 `x` 读出来算归一化，这一写一读就白白浪费了一次显存往返。因此，在 vLLM 里，pLLM 准入的模型家族（Qwen2、Qwen2.5、Qwen3、Llama、DeepSeek……）都用了一个叫 `fused_add_rms_norm` 的原地融合算子：它把“上一子层的残差相加”和“下一子层入口的归一化”捏在了一起，在 GPU 寄存器里顺手把两件事全干了。

起初，这个融合算子看起来像一扇当面摔上的铁门：加法被吞进了 C++/CUDA 内核肚子里，Python 层连下嘴的地方都没有。

但盯着结构图看了几分钟后，我突然意识到，vLLM 不仅没有把路堵死，反而帮我们把全模型最完美的挂载点圈了出来：

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

仔细看右半边那个 `norm2` 的入口。正因为上一子层（比如 Attention）没有自己做加法，当执行流走到 `norm2` 门口时，**上一个子层的分支输出 `y`（`branch`）和主干道上的残差流 `x`（`residual`），恰好完好无损地同时交到了 `norm2` 手里！**

既然加法藏在下一个 Norm 里，那我们就干脆**骑在 Norm 上**。

在 `pllm/backends/vllm/model_sites.py` 里，我们把每一个这样的交界处叫做 `SublayerBoundary`。当某个边界被挂载时，它先替上一个子层完成收尾——不调默认的加法，而是调用用户定义的改写方法算出新的残差流（如果需要供后续层跨层读取，顺手留一份副本；如果是 Hyper-Connections 这类多流结构，再调读取方法组合出下一子层的输入），最后调用模型原本的那个 `norm` 内核完成归一化。

更妙的是，如果某一层什么都没挂呢？**没人用的边界，连碰都不碰。** 如果你只声明了 `layers=(8, 16)`，那么全模型其余几十个 Norm 依然是原汁原味的 vLLM 原生模块，零包装、零额外跳转。来看看 `ModelSites._install_boundaries` 里的安装循环：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/model_sites.py
for index, (norm, kernel) in enumerate(zip(sublayer_norms(decoder.model), decoder.norms, strict=True)):
    layer, attention = divmod(index, 2)
    before: Place | None = ("attention", layer) if attention else ("mlp", layer - 1) if layer else None
    after: Place | None = None if layer == count else ("mlp", layer) if attention else ("attention", layer)
    # ……
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

盯住第一行的 `divmod(index, 2)`。一个 $L$ 层的 Transformer 一共有 $2L+1$ 个 Norm（每层两个，外加模型最末尾的 `model.norm`）。这行简单的整数除法把线性递增的 `index` 翻译成了空间坐标：余数为 1 时，它夹在第 `layer` 层的 `attention` 之后、`mlp` 之前；余数为 0 时，它夹在上一层的 `mlp` 之后、当前层的 `attention` 之前。底下的 `if` 判断守住了大门：只有当这个边界真正承担了改写（`owner`）、读取（`reader`）、跨层保留（`keeps`）或隐藏态探针（`taps`）时，我们才把它的 `forward` 替换掉。（至于像 Value Residual 那样需要直接修改注意力 Value 的算法，我们用同样的思路在 QKV 线性投影出口挂了一个姊妹钩子 `ValueProjection`，在 Value 切片进入注意力计算和写入 KV Cache 之前原地改写它。）

当然，天下没有白捡的便宜。能够如此干净地骑在 Norm 上，前提是我们和底层模型立下了一条严格的结构契约：**全模型 $2L+1$ 个 Norm 必须严格接受 `(branch, residual)` 二元组，并返回 `(normalized, sum)`。**

如果某个用户在外面自己继承了一个 `Qwen2ForCausalLM` 的派生子类，随手改掉了某个 Norm 的调用签名或合并了投影层，我们的挂载点就会在错误的显存偏移上读写数据。所以，我们在模型准入上做出了一个看似严苛、实则必要的取舍：pLLM 只准入白名单上精确匹配的 vLLM 原生模型类（`type(model) in architectures`），任何自定义的模型子类在构建阶段都会被当场拒之门外。用入口处的严格挑剔，换取进入内层循环后的零猜测。

---

## 第二关：把门牌号当成身份证

找准了手术下刀的位置，接下来的问题是：当 vLLM 启动 `torch.compile` 追踪模型、接着录制 CUDA Graph 时，怎样让我们挂在 Norm 上的逻辑顺顺当当地钻进图里？

在 PyTorch 2 的编译器体系里，要把一段自定义逻辑塞进 `torch.compile` 而不引发图断裂（graph break），正统的通行证是 `torch.library.custom_op`。你把自己的函数注册成一个具名的自定义算子，再配一个只算形状不跑数据的 fake 实现。Dynamo 和 Inductor 看到它时，会把它当成一个不透明的黑盒节点留在计算图里；等到录制 CUDA Graph 时，黑盒内部真正发射的那些 GPU kernel 就会被原原本本地录进磁带。

听起来顺理成章，对吧？但当我们动手写这个 `@torch.library.custom_op("pllm::model_site", ...)` 时，迎头撞上了一堵墙：

**`custom_op` 的签名是个认死理的守门员，它只收张量（`Tensor`）和基本标量（`int`, `float` 等），绝不接受任意的 Python 对象。**

而我们的挂载点在执行时，必须拿到当前的 `ModelSites` 会话实例——因为所有的挂载钩子、编译好的方法体、以及每一步动态更新的状态缓冲区，全都挂在这个 Python 对象上。

你可能会问：“既然能传 `int`，那我们在全局建一个字典 `_SITES: dict[int, ModelSites]`，给每个会话发一个递增的整数 `session_id` 传进算子里，不就能查表找回来了吗？”

如果是普通的单次脚本，这招完全可行。但在 vLLM 里不行。因为 vLLM 编译后的模型图是会跨会话复用的（甚至会通过 AOT 编译缓存写到磁盘上）。在 Dynamo 眼里，你传给算子的任何 `int` 标量都是编译期常量，会被直接**烘焙（bake in）**进生成的代码里！如果第一个会话的 ID 是 `1`，编译出来的图就会把 `1` 焊死；等第二个会话（ID 为 `2`）进来复用这张模型图时，算子依然会去查 `1` 号会话，当场张冠李戴。

不能传 Python 对象，又不能传整数标量，我们手里唯一能合法传进算子、且不会被编译器当成常量焊死的，就只剩下 **`Tensor`** 了。

等等——一个 GPU 张量本身，不就天然带着一个全世界独一无二的 64 位整数吗？**它在显存里的物理起始地址：`tensor.data_ptr()`！**

每个 `ModelSites` 会话在初始化时，本来就要分配一张名为 `_requests_of` 的一维 GPU 张量（用来记录当前步每一行属于哪个请求）。只要这个会话还活着，它就一直死死攥着这张表不释放；既然不释放，CUDA 显存分配器就绝不可能把这个物理地址分给别人。

于是，我们把这张 `_requests_of` 表放在算子只读张量列表的第一位 `reads[0]` 传进去。在编译器眼里，`reads[0]` 只是一个平平无奇的运行时输入张量，完全不需要为它特化计算图；而在算子内部，我们只需调一次 `.data_ptr()`，就能拿着显存门牌号当身份证，去全局注册表里精准拎出当前会话：

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

我非常喜欢第 9 行那句浓缩到极致的 `_SITES[reads[0].data_ptr()].hooks[hook].emit(inputs, outputs)`。你看，签名末尾那个 `hook: int` 标量确实会被编译器烘焙进图里，但那完全没关系——因为它代表的是“当前是模型里的第几个挂载点（比如第 3 个钩子）”，对于同一张模型拓扑图来说，第 3 个钩子永远是第 3 个钩子。真正随会话改变的身份，则藏在 `reads[0]` 的显存地址里暗度陈仓：

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

会话在 `__init__` 时把自己的 `_requests_of.data_ptr()` 登记进 `_SITES`，在 `close()` 时注销。顺便说一句，我们并不是唯一这么干的人——vLLM 自己内部那些不能被 Dynamo 编译的不透明层（比如注意力算子），也是通过 `forward_context.no_compile_layers` 这样一张全局注册表在图内外搭桥的。

桥搭通了，每步动态变化的数据怎么喂进去呢？还是老规矩：**固定地址缓冲区（stable buffers）**。每一步在模型图回放之前，主机侧的 `ModelSites.prepare` 会先把当前步每一行对应的请求序号拷进 `_requests_of`，再从 PTTS 状态池里把每个请求当前版本的状态拷进固定地址的 `state_values` 缓冲区（如果是旁路流提前算好的只读查表结果，就在预取流上异步拷贝并记下 CUDA Event）。等模型图回放时，录在图里的算子只需要闭着眼睛去这些固定地址拿数据就行了。

这里还有一个关乎编译速度的重要工程抉择：用户写的残差改写函数本身（比如一段包含多层缩放和混合的张量运算），该交给谁去编译？

一个看似顺手的方案是：干脆把 vLLM 整个模型的编译后端切到 Inductor，让 vLLM 在编译整座 Transformer 时顺带把用户的函数一起融合编译了。但我们算了一笔账后立刻否决了这个想法——如果那么干，每当用户调整一点点工作负载，整个 7B 甚至 70B 的大模型都要跟着从头重编译一遍，启动时间会变得令人抓狂。

相反，我们在 `pllm/backends/vllm/site_body.py` 里选择了一条解耦的路：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/site_body.py
def compile_site_method(rewrite: ResidualRewrite, *, read: bool = False) -> Callable[..., None]:
    """Share one fullgraph callable across a method's depths and model capture sizes."""
    function = rewrite.read if read else rewrite.write
    assert function is not None
    inputs = rewrite.read_inputs if read else rewrite.inputs
    return compile_math(partial(_method_ports, function, tuple(name for name, _ in inputs)), dynamic=True)
```

`compile_math` 在内部调用 `torch.compile(fullgraph=True, dynamic=True, ...)`，把用户写的小函数连同“将结果写进输出端口”的动作一起，单独编译成一个形状无关（dynamic shapes）的可调用对象。无论你在 28 层里挂了多少处，也无论 vLLM 录了多少个不同 batch 档位的 CUDA Graph，大家共享的都是这同一个轻量级编译产物，完全不打扰 vLLM 自己的模型编译器。

当这套图内挂载点跑通之后，我们回头看了一眼文章开头那个慢吞吞的初代 `ActivationPatch`。既然 `custom_residual` 已经能在 CUDA Graph 里随心所欲地改写残差了，`ActivationPatch` 还有什么理由保留一套单独的 Eager Python 路径呢？

我们大刀阔斧地把旧的逐请求 Python 实现彻底删掉，没有留任何历史包袱的“双路径”。现在的 `ActivationPatch` 只是 `custom_residual` 上面薄薄的一层语法糖：哪一行需要打补丁（比如只打 prompt 的首尾 token，或者 decode 的每一个 token）被编码成设备端的一维布尔掩码，每个请求的引导向量通过设备端槽号直接 gather 出来做掩码加法。开篇你看到的从 **4,258 tok/s** 到 **12,174 tok/s** 的近三倍飞跃，正是删掉旧路径、并入这套图内挂载点之后的实测结果。

---

## 第三关：数出那多出来的 61,440 个 Kernel

故事讲到这里，按理说应该开香槟了：无状态残差改写成功钻进了 vLLM 的 CUDA Graph，再也不用退回 Eager 模式了。

为了检验这套新机制的纯框架开销（overhead），我写了一个在数学上什么都不做的恒等改写（`identity`）：

```python
# check-docs: skip: illustrative identity rewrite, shaped like the benchmark workload
@pllm.custom_residual(site="mlp")
def identity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```

在 0.5B 基准上，当所有的改写还被迫跑在 Eager 模式时，这个 `identity` 只能跑 **3,320 tok/s**，是原生 vLLM 的 **25.3%**。现在，它通过 `pllm::model_site` 进了 CUDA Graph，我满怀期待地跑了一遍基准测试。

屏幕上跳出来的数字是：**8,810 tok/s**——占同期原生吞吐（13,381 tok/s）的 **65.8%**。

老实说，65.8% 是一个极其危险的数字。如果它只有 25%，你一眼就知道系统坏了，不修好睡不着觉；但 65.8% 有一种极具迷惑性的“体面感”。你会忍不住在心里替自己找借口：“毕竟我们在模型的每一层里都塞了自定义扩展点，用 Python 扩展底层引擎嘛，交三分之一的架构税，听起来很合理吧？”我们甚至真的差点认命了，在当时的文档里一本正经地写下了：“恒等 `custom_residual` 的吞吐约为原生的 66%。”

但在中间又做了几轮改动、把数字推到 **11,819 tok/s（89.8%）** 之后，我心里的那根刺还是拔不掉：**一个只写了 `return x + y` 的函数，干的活跟原模型一模一样，凭什么还要凭空吃掉 10% 甚至 34% 的吞吐？**

我决定不再靠直觉猜了，打开 profiler，把一整次 8,192 个 token 的生成过程录下来，**一个一个去数 GPU 到底发射了多少个 kernel。**

先看总时间账。在这次生成里，挂了 `identity` 的模型比原生（Inductor 编译后端）慢了 **70.54 ms**。把这 70.54 ms 拆开：其中 **11.92 ms（16.9%）** 是因为挂载后模型图改用源算子捕获、失去了 Inductor 的算子融合；而剩下的整整 **58.63 ms（83.1%）**，全都是我们的挂载点自己惹出来的祸！

再数 kernel 数量。同样用源算子录图的原生 vLLM，跑完这批生成一共发射了 **104,837 个 kernel**；而仅仅挂了一个 `return x + y` 的模型，发射了 **166,277 个**。

多出来的整整 **61,440 个 kernel** 是从哪儿冒出来的？我把分类账单摊在桌子上，看得目瞪口呆：

*   **37,119 个 `memcpy` 与 12,240 个 `direct_copy`**：每次进出 `pllm::model_site` 算子，为了把张量规规矩矩地塞进声明好的输入输出端口（ports），代码都在分配临时 staging 张量，拷进去、算完、再拷出来；
*   **12,288 个填零（zero fill）**：每次准备输出端口时，代码都极其讲礼貌地先调一次清零，哪怕几微秒之后这块内存就会被计算结果完整覆盖；
*   **12,288 个独立的 BF16 `add`**：这是最冤的一笔账！

回想一下我们在第一关发现的那个结构事实：原生 vLLM 的 `fused_add_rms_norm` 本来就能一口吃下 `(branch, residual)`，在寄存器里顺手把加法和归一化一起做了。现在倒好，因为用户在 Python 里写了一句 `return x + y`，我们的编译器就老老实实地发射一个独立的 BF16 加法 kernel，算出 `sum` 写回显存，然后再把 `sum` 交给 Norm 去归一化！

每一处小拷贝、每一次填零、每一个独立加法，单独看只有几微秒；但当它们乘上 24 层、再乘上几百个生成步，就堆成了这六万多个幽灵 kernel，硬生生烧掉了 52 ms 的纯 GPU 时间。

真相大白之后，怎么把这笔冤枉钱省下来？

当时我们面前摆着一条充满诱惑的捷径：既然绝大多数残差改写最后都是把某两个张量加起来，那不管用户在函数里写了什么，我们干脆无条件强行调用原生融合 Norm 的双输入路径不就完了吗？

我们试了一下，吞吐确实瞬间上去了，但数值测试亮起了红灯：相比于 FP32 黄金参考，原生 BF16 模型自己的最大对数概率（logprob）误差是 **0.243**；而走这条粗暴捷径时，最差 logprob 误差恶化到了 **0.394**。原因在于，当用户写的算式末尾不是纯粹的同形状同类型加法时，强行塞进融合算子会改变浮点数中间舍入（rounding）发生的位置。在 pLLM 里，我们的数值底线是死板的：**相对于 FP32 参考的误差，绝不能比原生模型自己更差。** 这条捷径被当场枪毙。

既然不能盲目硬塞，唯一的正道就是**让编译器先看懂用户写了什么**。

在 `SiteBody._trace_math` 里，我们请出了 PyTorch 的符号追踪工具 `make_fx`。在绑定挂载点时，我们先用没有真实数据的假张量（fake tensors）把用户的方法跑一遍，拿到它的 FX 计算图，然后顺着图的输出节点往回摸一步：

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
self.identity = (
    len(samples) == 2
    and {left, right} == set(inputs[:2])
    and len(tuple(body.graph.nodes)) == len(inputs) + 2
    and self._shape_independent(len(samples), arguments)
)
self._store(key, body, output, add, (left, right))
```

这段代码干的事情就像一台精密的显微外科手术：
1. 它先检查计算图的最后一个节点 `add`：它是不是一个标准的 `aten.add.Tensor`？它的左操作数 `left`、右操作数 `right` 以及相加结果，形状和数据类型是不是全都跟模型的残差流严丝合缝？
2. 如果答案是肯定的，它还会顺便做一道算术题（`self.identity`）：如果这张图里除了两个输入占位符、一个加法节点和一个输出节点之外空无一物，而且通过 `fx.symbolic_trace` 确认它没有任何依赖张量形状的 Python 分支，那就证明用户写的就是完完全全的 `return x + y`！对于这种纯恒等挂载点，我们在外层直接把 `self.owner` 设为 `None`，回放时连 `pllm::model_site` 算子都直接旁路跳过。
3. 那如果用户在加法之前确实做了别的运算（比如 `return x + 0.95 * y`）呢？这时候就轮到最后一行 `self._store(...)` 登场了：

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

看看这干净利落的四行：我们修改 FX 图的输出节点，让它直接吐出加法之前的两个操作数 `(left, right)`，然后调用 `body.graph.erase_node(terminal)` **一剪刀把末尾那个 `add` 节点从计算图里剪掉**，重新编译 FX 图！

```text
Before Surgery (Naive Hook — 61,440 extra kernels across generation):
  x, y ──► [ scale: 0.95 * y ] ──► [ standalone BF16 add ] ──► HBM ──► [ rms_norm ]
                                    (extra kernel & copy!)

After make_fx Surgery (Terminal add snipped & handed back to upstream):
  x, y ──► [ scale: 0.95 * y ] ──► (x, 0.95 * y) ──► [ native fused_add_rms_norm ]
              (erase_node(add))                       (exact same kernel & rounding!)
```

剪掉末尾的加法之后，用户函数算出来的 `(left, right)` 直接写进外层声明好的两个端口，转手就交给了 vLLM 原生的 `fused_add_rms_norm`。不仅省掉了那 12,288 次独立的加法 kernel 和显存读写，而且因为最终执行加法和归一化的就是原生模型自己的那个 kernel，浮点数舍入行为与原生模型分毫不差。

与此同时，我们把端口进出时多余的临时缓冲拷贝和那 12,288 次无脑填零一并铲除，让编译后的方法体直接把结果写进目标端口。

当这几刀手术全部完成，我重新跑了一遍 profiler。

在单步 decode 的 CUDA Graph 里，kernel 数量从 **628 个** 降到了 **388 个**；整次 8,192 token 生成发射的 kernel 总数，从 **166,277 个** 降回了 **104,837 个**——和用源算子录图的原生 vLLM **一个不多、一个不少，完全相等**。

0.5B 上的基准测试成绩单也随之定格：

| 演进阶段 | `identity` 吞吐 | 占原生比例 | 发生了什么 |
|---|---:|---:|---|
| 阶段 1：Eager 拦截 | 3,320 tok/s | 25.3% | 任何残差改写都会打断图捕获，迫使整座模型退回 Eager |
| 阶段 2：算子进图 | 8,810 tok/s | 65.8% | 通过 `pllm::model_site` 进入模型 CUDA Graph（同期原生 13,381） |
| 阶段 3：中间改动 | 11,819 tok/s | 89.8% | 中间若干改动；打开 profiler 数出多出的 61,440 个 kernel |
| 阶段 4：剪除加法与零拷贝端口 | **13,076 tok/s** | **≈99.5%** | 砍掉多余拷贝与填零、恒等旁路、`make_fx` 剪掉末尾加法还给原生融合 Norm（同期原生 13,142） |

同一批改动之下，其他真实改写算法的吞吐也应声上涨：Value Residual 从 11,474 涨到 11,541 tok/s，Hyper-Connections 从 8,569 涨到 8,692 tok/s，带状态的记忆子层从 7,861 涨到 8,207 tok/s。而且在中性参数设置下，生成的每一个 token 和每一个 logprob 都与原生模型逐位相同（bitwise identical）。

回顾这四步演进，最让我开心的不是那个 99.5% 的数字，而是我们解决问题的方式：为了消掉最后的开销，我们没有去手写一个“把用户改写逻辑和 RMSNorm 揉在一起”的庞大 Triton 内核，而是用十几行 `make_fx` 代码看懂了用户的意图，把加法物归原主，还给了上游早就写好的融合内核。**让上游多干一点，我们的代码就少一点。**

---

## 放到 7B 模型上：剩下的 3% 买来了什么？

在小巧的 0.5B 模型上，一步 decode 只有 3 ms 左右，任何框架层的风吹草动都会被放大成刺眼的百分比。那么，当我们把舞台切换到真实的 **Qwen2.5-7B-Instruct**（128 个并发序列，每步约 21.7 ms）时，账单长什么样？

在 7B 上，原生 vLLM 的图模式吞吐是 **5,894 tok/s**。
*   如果我们在第 12 层单挂一个 `identity` 改写，吞吐是 **5,722 tok/s（97%）**；
*   如果我们丧心病狂地把全模型 28 层的每一个 MLP 残差全都挂上 `identity`，吞吐是 **5,721 tok/s（依然是 97%）**，此时在最大批次 token 数 8,192 下，模型图捕获池需要约 1.13 GB 显存。

停下来看看这两个数字：只挂 1 层是 5,722，挂满 28 层是 5,721。

这说明挂载点本身的每次调用开销已经小到了完全淹没在测量噪声里的程度！那固定消失的 **3%** 是去哪儿了？

最大的嫌疑写在 `pllm.explain` 的一行提示里：`model graphs capture source operators, not inductor fusion`。

只要你在模型里启用了 `@pllm.custom_residual`，pLLM 就会让 vLLM 在录制 CUDA Graph 时捕获模型原本的**源 CUDA 算子**（对应 vLLM 的 `"eager"` 编译后端），而不是 Inductor 融合后的算子。为什么放着 Inductor 融合不用？一来，让 Inductor 为每个挂了钩子的引擎重新编译整座模型太贵（前面说过）；二来，Inductor 在看到图里多出不透明的 `pllm::model_site` 边界时，它对周围 `add` 和 `RMSNorm` 的重排与融合策略会发生变化，从而改动 BF16 的尾数舍入。如果我们放任 Inductor 自由融合，当你把缩放系数设为 `1.0`（数学上的恒等变换）时，跑出来的文本就会因为微小的舍入漂移而与原模型对不上！

在 0.5B 上做过一次干净的对照：哪怕完全不加载 pLLM，仅仅把原生 vLLM 自己的模型图从 Inductor 切成源算子捕获，吞吐就会从 13,158 降到 12,911 tok/s（相差约 1.9%）；到了 7B 上，我们没有单独测这张“不融合门票”，但挂 1 层和挂 28 层同为 97%，说明那 3% 基本是这类固定成本。我们心甘情愿地掏了这笔固定门票，买下了一条对研究者极其宝贵的性质：**只要你把超参数设回中性值，改写后的模型与同样按源算子录图的原生模型保证逐位一致。**（在后面的[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)里，你会看到我们在自定义注意力上做出了完全相同的取舍。）

当然，挂载机制本身的开销归零了，不代表算法自己新增的数学计算也能白送。我们在 7B 上跑了一遍示例 07 里的三种真实架构改写（生成 256 token）：

1.  **Value Residual**（改写第 1 到 27 层的注意力 Value，32 个并发序列）：跑到了原生图模式的 **99%**。因为它只是在 QKV 投影后做了一次轻量级的线性插值，几乎不增加带宽负担。
2.  **Hyper-Connections**（$n=4$ 条并行残差流，覆盖全模型 56 个子层边界，32 个并发序列）：在 7B 上是原生的 **93%**（而在 0.5B 上只有约 **66%**）。把一条残差流拆成四条车道、每进出一个子层都要做一次混合矩阵乘法，多出来的显存读写和矩阵运算是实打实的物理开销，模型越小，这笔额外带宽占比就越重。
3.  **带状态的记忆子层**（每个请求独立维护一个 $3584 \times 3584$ 的 FP32 快权重记忆矩阵，单个请求就要占约 49 MiB 状态显存，因此并发序列数设为 8）：跑到了同样 8 序列原生图模式的 **75%**，录制模型图需要约 **1.9 GiB** 的捕获池预算（0.5B 下对应的捕获池分别为 identity 52 MiB、value 46 MiB、HC 72 MiB、memory 448 MiB；如果预算不够，pLLM 不会静默回退，而是直接报错并告诉你精确的所需字节数）。

顺便提一句，这个带状态的记忆子层当初为了进 CUDA Graph 也经历了一番波折。最早它只能跑在 Eager 模式下，因为早期实现在主机侧按每个请求的 token 跨度（span）直接绑定 PTTS 私有页，一旦录进图里，那一次的几何切分就被焊死了；后来我们在 `SiteMemorySchedule` 里把它改成了**设备端索引、有界压实的请求分桶**，通过固定地址的槽号表在 GPU 上间接寻址私有页，才终于把它也一并塞进了模型图里。

有趣的是，在 7B、32 并发这个设置下，因为单步矩阵乘本身就要跑将近 19 ms，原生 vLLM 自己的 Eager 模式也能跑到图模式的 99%（128 并发下是 98%）。所以单看 7B 的吞吐数字，你甚至不太容易察觉“进 CUDA Graph”有多关键；但只要切回 0.5B（或者未来在更快的新卡上跑低延迟小 batch），同样的这几个改写一旦掉出图外，吞吐立刻摔回惨不忍睹的 **12%–34%**。

---

## 站在边界上看取舍

现在，我们已经把这套图内挂载点的每一颗螺丝都拆开看过了。如果你准备在自己的研究或服务里用上 `@pllm.custom_residual`，有几条由底层物理结构决定的边界需要记在心里：

*   **单卡运行与关闭前缀缓存**：目前的挂载点要求 `tensor_parallel_size=1`。同时，因为你在模型半路改写了残差或 Value，相同的 token 前缀在经过你的函数（尤其是带每请求状态的改写）后，算出来的隐藏态和 KV Cache 可能跟以前完全不同了。因此，vLLM 的 Prefix Caching 必须关闭，否则复用的旧 KV 会污染新请求的结果。
*   **验证需要在 GPU 上跑**：在[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)里我们会讲到，pLLM 提供了秒级完成的 CPU 契约检查 `pllm.testing.check`。但因为 CPU 上根本没有加载几十 GB 的真实模型权重，模型内部的挂载点在 CPU 检查时会被跳过（在 `explain` 里显示为 `unverified`），你需要调用 `pllm.testing.check_on_gpu` 在真卡上做端到端的隔离与置换验证。
*   **不要混用可变深度循环**：带状态（`state=`）的记忆子层默认每个 token 按固定顺序流经挂载的层，因此它不能和[后面将要讲到的 `@pllm.custom_forward` 自适应跳层](10-custom-layer-loops.md)混用——当不同的 token 动态跳过某些层时，单层的块状扫描（chunk scan）就失去了规整的步进语义。

把视野拉远一点，拿它和社区里现有的工具比一比，各自的位置就很清楚了。如果你只是在单卡上针对一两条句子做可解释性探针，Hugging Face 的原生 hook、`TransformerLens`、`nnsight` 或 `pyvene` 依然是最顺手的瑞士军刀——它们可以在任意子模块里随时打断、取出激活、在 Python 里随心所欲地改完再塞回去，代价是它们活在静态 batch 和 Eager 模式的世界里。反过来，如果你要在 vLLM 上把一个新的残差结构跑出生产级吞吐，以前大家唯一的办法就是把 vLLM 的 `qwen2.py` 或 `llama.py` 复制一份出来直接改源码——快是快了，但每试一个新想法就要维护一个自定义 vLLM 分支。

而我们在 `ModelSites` 里所做的，其实是在这两极之间搭了一座小桥。

回看这座桥上的三个核心机关，你会发现一个很有趣的共同点：**没有一处是在跟现有的系统对着干，全都是在顺水推舟。**
*   找不到独立的残差加法，因为上游把它吞进了下一个 Norm？那我们就顺势骑在 Norm 上，白捡一个同时握着 `x` 和 `y` 的交界口；
*   `custom_op` 不让传 Python 对象、传整数标量又会被焊死？那我们就顺手拿只读张量的显存地址 `data_ptr()` 当身份证；
*   自己做加法多出了 12,288 个 kernel？那我们就用 `make_fx` 把末尾的 `+` 号剪掉，顺手把操作数还给上游的 `fused_add_rms_norm`。

至于那段从 25.3% 到 65.8%、再到 99.5% 的追查故事，则留给了我们一个再朴素不过的工匠教训：当一个系统跑出“66%”的时候，千万别急着把它当成架构的固有代价写进文档。打开 profiler，去数一数那些看不见的 kernel——真正偷走性能的，往往不是什么高深的理论瓶颈，而是你在门口随手多写的那一次填零和拷贝。
