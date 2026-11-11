# How to Lose 61,440 Kernels in 10 Lines：在 `torch.compile` 图里改残差与训权重

> 2026-11-10 · pLLM 提交 `cff7c196` · 测量环境：除文中另注外为单张 RTX 4090（24 GiB）、vLLM 0.32.0、Qwen2.5-7B-Instruct，8 条 prompt × 16 个样本、生成 256 token。
> 这是一篇有日期的文章，不随代码更新；当前的用法以[用户指南](../../README.md#用户指南)为准，当前的实测数字以[性能与显存](../performance.md)为准。

在单请求的研究脚本里，往 Transformer 中间塞一点自定义计算几乎没有门槛。

如果你想做激活引导（Activation Steering，如 CAA 或 ActAdd），在第 12 层的输出上加一个方向向量 $h \leftarrow h + \alpha \cdot v$，挂一个 `register_forward_hook` 写三行代码就够了；如果你想研究架构变体——比如把第 0 层的注意力 Value 掺进后续各层（Value Residual）、把单条残差流拓宽成 $n$ 条并行车道（Hyper-Connections），或者像 In-Place TTT（*In-Place Test-Time Training*）那样把选定层 MLP 的 `down_proj` 当作可写的“快权重（fast weights）”，每读入一个固定大小的块（chunk）就当场做一次外积更新 $\Delta W = \eta \cdot V^\top Z$——在 Hugging Face 的 `generate` 循环里，权重和激活就摆在明面上，你想读就读，想改就改。

但只要你试着把这些代码搬进开启了 `torch.compile` 与 CUDA Graph 的生产级 vLLM 引擎，马上就会迎头撞上三堵墙：

1. **原厂算子已经高度融合**：为了省掉显存（GDDR6X / HBM）的读写往返，生产引擎早就把教科书里独立的残差加法 `+` 吞进了下一层入口的 C++ 融合算子里；
2. **编译器封死了动态 Python 回调**：Dynamo 会把模型前向追踪成静态的 FX 计算图并录制 CUDA Graph，任何未注册的 Python 逻辑要么被当成常量焊死，要么把完整的计算图撞出断层（graph break），逼着整座模型退回逐算子发射的 Eager 模式；
3. **成百上千个请求各怀心事**：当 128 个并发请求被拍扁拼成一张二维张量 `[num_tokens, hidden_size]` 冲进同一个线性层时，第 0 行要乘张三的快权重，第 1 行要乘李四的快权重，而在多卡张量并行（Tensor Parallelism，TP）下，每张显卡手里甚至只有半截特征维度。

因此，In-Place TTT 的官方开源仓库在推理入口处干脆利落地写下了一行断言：

`assert len(input_ids) == 1, "only support bs=1 for TTT inference"`

要在不拆散 vLLM 原厂编译管线、不手写一堆定制 C++/CUDA 内核的前提下，让残差改写与前向内权重更新跑满生产级吞吐，唯一的出路是**顺着编译器和 PyTorch 的抽象层借力**。这篇文章就带你走一遍这趟工程拆解：看看怎么顺着原厂融合算子的接缝钻进图里，怎么用 10 行 `make_fx` 图手术砍掉 61,440 个不该存在的幽灵 kernel，又怎么像套俄罗斯套娃一样把 `grad`、`vmap`、`compile` 和 FX 双图同构叠在一起，实现多卡零额外通信的图内权重训练。

---

## 一、 寻找消失的 `+` 号，以及用门牌号当身份证

翻开任何一本深度学习教材，Transformer 层的残差连接都长这样：

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

如果要允许用户改写残差连接（例如替换那个 `+` 号），最直觉的想法是去计算图里找到那个加法节点并替换它。但如果你打印出 vLLM 跑起来的模型图，从头找到尾也找不到那个独立的 `+` 号。

在解码阶段，GPU 极度受限于显存带宽。如果先算一次 `x = x + y` 写回显存，紧接着下一层的 `RMSNorm` 再从显存把 `x` 读出来归一化，就白白多了一次显存往返。因此，pLLM 准入的模型家族（Qwen2/2.5/3、Llama、DeepSeek 等）在 vLLM 中都使用了一个叫 `fused_add_rms_norm` 的原地融合算子：它把“上一子层的残差相加”和“下一子层入口的归一化”捏在了一起，在寄存器里顺手做完。

起初这看起来像一扇锁死的门，但退后一步看，它反而替我们圈出了全模型最规整的挂载位置：

```text
Textbook Transformer:
  stream x ──────────────► (+) ─────────────────► [ norm2 ] ──► mlp_in
  attn_out y ───────────────▲

vLLM Reality (and where the boundary hook sits):
                            ┌─── SublayerBoundary (hooks norm2) ───┐
  stream x ────────────────►│                                      │──► updated stream
                            │  user write(x, y) ──► native norm2   │
  attn_out y ──────────────►│                                      │──► normalized mlp_in
                            └──────────────────────────────────────┘
```

正因为上一子层（如 Attention）没有自己做加法，当执行流走到 `norm2` 门口时，**上一子层的分支输出 `y`（`branch`）和主干道上的残差流 `x`（`residual`），恰好完好无损地同时交到了 `norm2` 手里。**

既然加法藏在下一个 Norm 里，我们就直接骑在 Norm 上。在 pLLM 的 `pllm/backends/vllm/model_sites.py` 中，每个这样的交界处被称为 `SublayerBoundary`。一个 $L$ 层的 Transformer 共有 $2L+1$ 个 Norm（每层两个，外加末尾的 `model.norm`），并且严格遵守“接受 `(branch, residual)` 二元组、返回 `(normalized, sum)`”的结构契约（这也是为什么 pLLM 在入口处只准入精确匹配的原生模型类，拒绝签名可能被篡改的自定义派生子类）。如果某一层没有被声明改写或读取，对应的 Norm 就连碰都不碰，保持原厂零开销。

找准了挂载位置，接下来的问题是：怎么带着我们的 Python 会话状态穿过 `torch.compile` 和 CUDA Graph？

如果直接在 hook 里跑普通 Python 逻辑来挑出当前批次的行并读取状态，计算图就会在此断裂。在单卡 RTX 4090 上跑 Qwen2.5-0.5B（BF16，8 条 prompt $\times$ `n=4` 共 32 个并发序列，生成 256 token，vLLM 0.27.1）的激活引导基准时，把补丁放在图外、按请求逐个回调 Python 的实现只能跑 **4,258 tok/s**；而一旦进入模型的 CUDA Graph，同样的补丁能跑到 **12,174 tok/s**——图外实现足足丢掉了将近三分之二的吞吐。

在 PyTorch 2 的编译器体系里，要把一段自定义逻辑塞进编译图而不引发图断裂，标准通行证是注册一个不透明算子 `@torch.library.custom_op`。Dynamo 看到它时会把它当成黑盒节点留在图里，录 CUDA Graph 时则会把黑盒内部实际发射的 GPU kernel 录进磁带。

但当你动手写这个 `custom_op` 时，会立刻撞上一条严苛的签名限制：**`custom_op` 只接受张量（`Tensor`）和基本标量（`int`, `float` 等），绝不接受自定义的 Python 会话对象。**

你可能会想：“既然能传 `int`，那我建一个全局字典 `_SITES: dict[int, ModelSites]`，给每个会话发一个整数 `session_id` 传进算子里查表，不就行了？”

在单次脚本里可以，但在 vLLM 里不行。因为 vLLM 编译后的模型图会跨会话复用（甚至通过 AOT 编译缓存落盘）。在 Dynamo 眼里，任何传入算子的 `int` 标量都是编译期常量，会被直接**烘焙（bake in）**进生成的代码里！如果第 1 个会话的 ID 是 `1`，图里就会焊死 `1`；等第 2 个会话进来复用这张图时，算子依然会去查 `1` 号会话，当场张冠李戴。

不能传对象，传整数又会被焊死，唯一能作为运行时变量合法传进算子的就只有 **`Tensor`** 了。而一个常驻显存的 GPU 张量本身，恰恰就带着一个全世界独一无二的 64 位整数——**它在显存里的物理首地址 `tensor.data_ptr()`**。

每个会话在初始化时，本来就要分配一张持久的 GPU 张量表（例如记录每行归属的 `_requests_of` 表，或参数覆盖层的索引表 `table`）。只要会话还活着，它就一直持有这张张量的所有权，显存分配器绝不会把这个物理地址分给别人。于是，我们把这张张量放在算子只读参数的第一位 `reads[0]` 传进去，在算子内部直接拿它的显存门牌号当身份证：

```python
_SITES: dict[int, "ModelSites"] = {}


@torch.library.custom_op("pllm::model_site", mutates_args={"outputs", "writes"})
def model_site(
    inputs: list[Tensor], outputs: list[Tensor], reads: list[Tensor], writes: list[Tensor], hook: int
) -> None:
    # ……
    _SITES[reads[0].data_ptr()].hooks[hook].emit(inputs, outputs)
```

签名末尾的 `hook: int` 标量代表“这是模型里的第几个挂载钩子”，它属于静态模型拓扑的一部分，被烘焙进图里完全正确；而真正随会话改变的身份，则藏在 `reads[0].data_ptr()` 里暗度陈仓（vLLM 自己内部那些不透明的注意力层，也是用类似的全局注册表 `forward_context.no_compile_layers` 在图内外搭桥的）。至于用户编写的残差数学函数本身，则由 `compile_site_method` 调用 `torch.compile(fullgraph=True, dynamic=True, ...)` 单独编译成一个跨层、跨批次档位共享的轻量可调用对象，完全不必为了改一行小函数而去重新编译整座大模型。

---

## 二、 破案：一行 `return x + y` 为什么多出了 61,440 个 Kernel？

通过 `pllm::model_site` 钻进 vLLM 的 CUDA Graph 后，我们写了一个在数学上什么都不做的恒等改写（`identity`），用来检验这套挂载机制的纯框架底噪：

```python
@pllm.custom_residual(site="mlp")
def identity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```

在刚才那套 RTX 4090、Qwen2.5-0.5B（32 并发序列 × 256 token，共生成 8,192 个 token）的基准上，当改写被迫跑在 Eager 模式时，吞吐只有 **3,320 tok/s**（占原生 13,142 tok/s 的 **25.3%**）。现在算子顺利进了 CUDA Graph，最直接的进图实现跑出来的数字是：**8,810 tok/s**——占同期原生吞吐（13,381 tok/s）的 **65.8%**；再去掉一部分外围宿主开销，吞吐升到 **11,819 tok/s**（按 Inductor 编译的原生为 13,158 tok/s，占 **89.8%**）。

89.8% 看起来已经很体面了，但从系统工程的角度看，这依然是不可接受的：**一个只写了 `return x + y` 的函数，干的活跟原模型一模一样，凭什么要凭空收走 10% 甚至三分之一的过路费？**

与其靠直觉猜，不如打开 profiler，把一次完整的 8,192-token 生成过程录下来，一个一个去数 GPU 到底发射了多少个 kernel。

账单一拉出来，真相大白。在这 8,192 个 token 的生成过程中，挂了 `identity` 的模型比原生（Inductor 编译后端）慢了 **70.54 ms**。其中只有 **11.92 ms（16.9%）** 来自模型图改用源算子捕获而少掉的 Inductor 融合；剩下的整整 **58.63 ms（83.1%）**，全都是挂载点自己惹出来的祸。

再看 kernel 总数：同样采用源算子录图的原生 vLLM，跑完这批生成一共发射了 **104,837 个 kernel**；而仅仅挂了一个 `return x + y` 的模型，竟然发射了 **166,277 个 kernel**！

多出来的整整 **61,440 个 kernel**（合计烧掉约 52 ms 的纯 GPU 时间）是从哪儿冒出来的？拆开分类账本一看，全是不经意间写下的“礼貌性操作”：

* **37,119 个 `memcpy` 与 12,240 个 `direct_copy`**：每次进出 `pllm::model_site` 算子，为了把张量规规矩矩地放进声明好的输入输出端口（ports），代码都在先把数据拷进临时 staging 张量，算完再拷回目标端口；
* **12,288 个填零（zero fill）**：每次准备输出端口时，代码都先礼貌地调一次清零，哪怕几微秒之后这块显存就会被计算结果完整覆盖；
* **12,288 个独立的 BF16 `add`**：这是最冤的一笔账。原生的 `fused_add_rms_norm` 本来就能一口吃下 `(branch, residual)`，在寄存器里顺手做完加法和归一化。现在因为用户写了句 `return x + y`，编译器就老老实实地单独发射一个 BF16 加法 kernel，算出和写回显存，然后再调用原生 Norm 把它从显存里读出来归一化！

每一处小拷贝、填零和独立加法，单看只有几微秒；但当它们乘上 24 层、再乘上几百个生成步，就堆成了这六万多个多余的 kernel。

怎么把那 12,288 次独立加法省掉？

这时有一条极其诱人的捷径：既然绝大多数残差改写最后都要合流，那不管用户在函数里写了什么，干脆无条件强行调用原生融合 Norm 的双输入路径不就完了吗？

我们实测了这个方案，数值精度亮起了红灯：相比于 FP32 黄金参考，原生 BF16 模型自己的最大对数概率（logprob）误差是 **0.243**；而无条件硬塞给融合 Norm 时，最差 logprob 误差恶化到了 **0.394**。原因在于，当用户算式的末尾不是纯粹的同形状同类型加法时，强行拆给融合算子会改变浮点数中间舍入（rounding）的位置。**相对于 FP32 参考的误差绝不能比原生模型自己更差**——这条捷径被当场否决。

既然不能盲目硬塞，唯一的正道就是让编译器先看懂用户写了什么。在 `pllm/backends/vllm/site_body.py` 的 `SiteBody._trace_math` 与 `_store` 中，总共只需要十来行代码就能完成这场显微手术：

```python
body = make_fx(trace, tracing_mode="fake", _allow_non_fake_inputs=True)(samples, arguments)
output = next(node for node in body.graph.nodes if node.op == "output")
(add,) = output.args[0]
# ……
if not isinstance(add, fx.Node) or add.target != torch.ops.aten.add.Tensor or add.kwargs:
    return
left, right = add.args
# ……
self._store(key, body, output, add, (left, right))

# …… 以下是 _store 的函数体
set_tuple_output(body, output, operands)
body.graph.erase_node(terminal)
body.recompile()
```

这段代码的逻辑非常直接：
1. 在绑定挂载点时，用不占物理显存的假张量（fake tensors）调用 `make_fx`，把用户的方法追踪成一张符号计算图；
2. 顺着图的输出节点 `output` 往回看一步：如果最后一个节点 `add` 恰好是一个标准的 `aten.add.Tensor`，且它的左右操作数 `(left, right)` 与输出形状、类型全都和残差流严丝合缝，我们就修改图的输出节点，让它直接吐出加法前的两个操作数 `(left, right)`，然后在 `_store` 里调用 `body.graph.erase_node(...)` **一剪刀把末尾那个加法节点从计算图里剪掉**！（如果检查发现整张图里除了两个输入和这一个加法之外再无别的算子，那就证明用户写的就是纯恒等 `return x + y`，外层直接连挂载算子都旁路跳过。）

```text
Before Surgery (Naive Hook — 61,440 extra kernels across generation):
  x, y ──► [ scale: 0.95 * y ] ──► [ standalone BF16 add ] ──► Memory ──► [ rms_norm ]
                                     (extra kernel & copy!)

After make_fx Surgery (Terminal add snipped & handed back to upstream):
  x, y ──► [ scale: 0.95 * y ] ──► (x, 0.95 * y) ──► [ native fused_add_rms_norm ]
               (erase_node(add))                       (exact same kernel & rounding!)
```

剪掉末尾加法后，用户函数算出来的 `(left, right)` 直接写进外层输出端口（同时砍掉了端口进出时的临时缓冲拷贝与填零），转手就交给了 vLLM 原生的 `fused_add_rms_norm`。不仅省掉了那 12,288 次独立加法内核与显存读写，而且因为最终执行加法和归一化的就是原厂自己的那个 kernel，浮点数舍入行为与原生模型分毫不差。

手术完成后，重新打开 profiler 数一遍：单步 decode 的 CUDA Graph 内核数从 **628 个** 降到了 **388 个**；整次 8,192-token 生成发射的 kernel 总数从 **166,277 个** 降回了 **104,837 个**——与源算子录图的原生 vLLM 一个不多、一个不少，吞吐也随之追到了 **13,076 tok/s**（同期按源算子录图的原生为 13,142 tok/s，约 **99.5%**）。

当我们把舞台切换到单步计算重得多的 **Qwen2.5-7B-Instruct**（单卡 RTX 4090，vLLM 0.32.0，128 并发序列，单步约 21.7 ms，原生吞吐 5,894 tok/s）时：
* 只在第 12 层挂一个 `identity`，吞吐是 **5,722 tok/s（97%）**；
* 把全模型 28 层的每一个 MLP 残差全部挂满 `identity`，吞吐是 **5,721 tok/s（依然是 97%）**。

挂 1 层和挂 28 层几乎完全相等，说明每多挂一层几乎不增加开销。那固定差出来的约 3% 去了哪里？7B 上没有单独测过，但最可能的主要来源写在 `pllm.explain` 的一行提示里：`model graphs capture source operators, not inductor fusion`。启用 `@pllm.custom_residual` 时，pLLM 会让 vLLM 在录制 CUDA Graph 时直接捕获模型的源 CUDA 算子（`backend="eager"`），不走 Inductor 融合：在 7B/4090 上，全层挂载的残差改写用 Inductor 反而慢 4.8%，有状态的残差改写也过不了嵌套的 FX 追踪；顺带的好处是，“把超参数设回中性值”时，结果与同样按源算子录图的原生基准保证**逐比特一致（Bitwise Identical）**（在 0.5B、vLLM 0.27.1 上单独测量过这张“源算子录图门票”，原生 Inductor 是 13,158 tok/s，原生源算子录图是 12,911 tok/s，相差约 1.9%）。至于真实的有状态或宽残差算法，32 并发下 7B 的 Value Residual 跑在原生的 **99%**，4 车道 Hyper-Connections 跑在 **93%**，而每请求维护 $3584 \times 3584$ FP32 快权重矩阵的记忆子层（通过设备端索引的有界请求分桶录进图内）在 8 并发下跑在原生的 **75%**。

---

## 三、 在前向里边走边训：顺着 PyTorch 的函数式变换往下走

改写残差解决的是层间激活流，而另一大类前沿算法要动的是线性层内部的**权重**——即参数覆盖层（Parameter Overlays）：无论是每个请求挂一份专属的低秩 LoRA / 稠密增量 $\Delta W$，还是像 In-Place TTT 与 `PrefillAdapters` 那样，在模型前向传播的半路上每攒够一个 chunk 就当场求导更新权重，并要求同一次前向里的后续块立刻读到更新后的新权重（`visibility="exact"`）。

看看在 pLLM 里，作者写的 In-Place TTT 核心更新函数 `step`（`pllm/workloads/inplace_ttt.py`）长什么样：

```python
@pllm.train_in_forward(
    visibility="exact",
    state="dense_fast_weights",
    targets=targets,
    weight="delta",
    chunk=chunk_size,
    tokens="all",
)
def step(
    self,
    delta: torch.Tensor = pllm.parameter("dense_fast_weights", "delta"),
    inputs: torch.Tensor = pllm.operand("input"),
    targets: torch.Tensor = pllm.operand(target_source),
    base: torch.Tensor = pllm.operand("weight"),
    conv: torch.Tensor = pllm.parameter("target_parameters", "conv"),
    proj: torch.Tensor = pllm.parameter("target_parameters", "proj"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve inside this chunk, read the incoming weight, then sum the positive outer products."""
    values = (
        torch.nn.functional.conv1d(targets.T.unsqueeze(0), conv, padding=2, groups=targets.shape[1]).squeeze(0).T
        @ proj
    )
    update = values.T @ inputs if self.dtype is torch.bfloat16 else values.float().T @ inputs.float()
    # ……
    update = update * self.config.learning_rate
    # ……
    return delta + update, inputs @ delta.to(inputs.dtype).T
```

注意这个函数的视角：它的参数里只有**一个请求**的权重增量 `delta`、**一个块**的输入 `inputs` 与目标 `targets`，返回的也只是更新后的 `delta` 和当前块在旧权重下的修正输出。没有 batch 维，没有请求路由，没有页表槽号，完全就是从论文框图里拓印下来的单人数学公式。

甚至，如果作者不想手推外积更新公式（如 `PrefillAdaptersWorkload`），他只需要给状态声明一个 `optimizer=pllm.SGD(lr)`，然后在方法里写三行求均方误差的标量 loss：`(prediction - target).square().mean(-1)`。

为了把这段“单请求、单块”的纯数学代码放进 128 并发、开启 CUDA Graph 的引擎里运行，我们没有写哪怕一个专用 C++/CUDA 内核，而是让 PyTorch 现有的变换原语各司其职：

```text
Author's Code:  step(delta, inputs, targets)  or  loss(x, target, base, a, b)
  │
  ├─► 1. Bounded index_select + bmm          (Amortizes 128-request grouped projection reads)
  ├─► 2. grad ──► vmap(chunk_size=1) ──► compile  (Per-chunk backward without mixing BF16 reductions)
  └─► 3. make_fx (Global vs. Shard trace)    (Proves TP>1 updates need zero extra cross-GPU comm)
```

先看只读（或读取当前生效版本）时的“批内分组矩阵乘”。当 128 个请求混在一个 batch 里流经同一个 `down_proj` 时，有的请求在做 prefill 占了几十行，有的在做 decode 只占 1 行。我们在 `pllm/execution/parameter_projection.py` 的 `projection_read` 里用 5 行纯 ATen 算子解决：把属于同一个请求的连续行切成定长小块（decode 每块 1 行，混入 prefill 时每块固定 `ROWS_PER_BLOCK = 32` 行），每个块只从 **PTTS（Paged Tensor Transition System，分页张量状态迁移系统——注意它开头的 P 是 Paged 分页与页表，和 pLLM 里代表 programmable 的 p 无关）** 的物理池里 `index_select` 抽取一份权重矩阵或低秩因子对，所有块一起跑一次标准的 `torch.bmm`，最后用掩码贴回原位。（至于进入 CUDA Graph 的边界，`pinned_projection` 自定义算子用的正是上一节讲过的同一个技巧：拿索引表张量 `table.data_ptr()` 当身份证从 `_READERS` 里找回读取器。当然，如果你唯一的需求是同时服务上千个静态只读的 LoRA，应该直接用 vLLM 原生基于 SGMV/BGMV 定制内核的 multi-LoRA；我们用纯 ATen 搭这套分块 `bmm`，换来的是它能直接通吃在线训练出来的、带多版本演化的稠密与低秩状态。）

再看更核心的前向内自动求导。当作者只写了逐行的重构误差 `loss` 时，反向传播和批处理在 `pllm/execution/overlay_task.py` 的 `overlay_gradient` 里只有十来行代码：

```python
def overlay_gradient(function: Callable[..., torch.Tensor], *, batched: bool = False) -> Gradient:
    """Compose request mapping with differentiation before compiling; broadcast frozen weights."""

    def loss(parameters: Tensors, operands: Tensors, constants: Tensors, valid: torch.Tensor) -> torch.Tensor:
        rows = function(**parameters, **operands, **constants)
        # ……
        return torch.where(valid, rows.to(torch.promote_types(rows.dtype, torch.float32)), 0).sum()

    eager = torch.func.grad(loss)
    if batched:
        # Folding requests into the frozen-base GEMM changes CUDA's BF16 reduction.
        # Compile the chunks together while retaining each source GEMM's shape.
        eager = torch.vmap(eager, in_dims=(0, 0, None, 0), chunk_size=1)
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        eager, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )
```

这几行代码里藏着两个必须严丝合缝的工程细节：

第一，**套娃的顺序为什么必须是 `grad` $\to$ `vmap` $\to$ `compile`？**
因为 `torch.func.grad` 要求被求导的函数输出一个标量。我们在单个请求内部用 `valid` 掩码把当前块的有效行求和变成标量，算出单请求梯度；接着外层的 `vmap` 沿着请求维铺开，并通过 `in_dims=(0, 0, None, 0)` 声明：参数、输入和掩码是每个请求各有一份的，唯独第三个参数 `constants`（那块巨大的冻结底座权重 `base`）是所有人共享的（`None`，直接广播，绝不复制）；最后最外层的 `torch.compile` 把展开后的前向与 AOTAutograd 反向图一口气融合编译。（顺便一提，当一个块跨越多步才攒满时，由于块内未更新前权重是恒定的，每步对新到达的行求梯度并累加进 PTTS 的梯度字段，在数学上与“跨步保存中间激活、等块满再跑一次反向”完全等价——所以用标准的梯度累加就够了，不必动用私有的 AOTAutograd 最小割分割器跨步保存激活。）

第二，**既然用了 `vmap` 做批处理，为什么要加一个看似自废武功的 `chunk_size=1`？**
看看代码里的那两行注释。当用户的函数里包含 `x @ base.T`（乘共享冻结底座）时，如果让 `vmap` 把 $B$ 个请求的输入全折进同一次矩阵乘里，输入形状就会从 `[C, H]` 变成 `[B * C, H]`。虽然数学上每一行是独立的，但当矩阵的 M 维发生变化时，cuBLAS 的启发式算法会挑选另一个分块（tile）尺寸不同的内核，导致内部 BF16 乘累加的归约顺序发生微调！结果就是，同一个请求在单独跑和跟别人拼批跑时，算出来的梯度在最后几个比特上出现了漂移。加上 `chunk_size=1` 后，每个请求对冻结底座的 GEMM 保持原本的 `[C, H]` 形状，同时依然在同一个 `torch.compile` 图里被一起编译，既守住了跨批次的数值一致性，又没丢掉算子融合。

---

## 四、 多卡 TP > 1 零通信证明：用两张假图“找不同”

现在把难度调到最高档：当模型用 `tensor_parallel_size=2` 或 `4` 切分在多张 GPU 上运行时，覆盖层的读取与在线更新怎么避免引入昂贵的跨卡集合通信？

读取这一半很顺畅。以 MLP 的 `down_proj`（`RowParallelLinear`）为例，在 TP=2 下，0 号卡持有左半边输入特征 $x_0$ 和左半边权重 $W_0$，1 号卡持有右半边 $x_1$ 和 $W_1$。因为矩阵乘法在线性代数上对被收缩的特征维天然可加：

$$x (W + \Delta W)^\top = \sum_j \left( x_j W_j^\top + x_j \Delta W_j^\top \right)$$

只要我们也把稠密增量 $\Delta W$（或低秩因子的输入侧矩阵 $A$）按输入特征维切开，在 `ProjectionHook` 里不拦外层的 `forward`，而是精准拦住里面的 `quant_method.apply`——赶在 vLLM 发起跨卡 `all-reduce` 之前，把本卡的修正量加进局部和，并**保持为宽裕的 FP32 穿过 vLLM 原有的 `all-reduce`**，等所有卡加完后再做唯一一次向 BF16 的舍入（避免每张卡在归约前各自多引入一次 BF16 舍入误差）。全过程搭上游的顺风车，零次额外通信。

但真正棘手的是**在线更新（`@pllm.train_in_forward`）**：当每张卡手里只有半截输入特征 `[C, H/2]` 时，让每张卡各自闷头跑一遍作者写的更新函数，算出来的新权重分片还对吗？

这完全取决于作者在更新函数里写了什么：
* **情况 A（In-Place TTT 的外积更新）**：`update = values.T @ inputs`。目标 `values`（`[C, Out]`）在各卡上完整复制，输入 `inputs`（`[C, H/2]`）沿特征维切分。矩阵乘收缩的是块内行维度 `C`，特征维 `H/2` 互不干扰。两张卡各算各的半边权重，拼起来与单卡分毫不差！
* **情况 B（重构误差求导或特征范数）**：比如先算 `pred = inputs @ base.T` 再求误差梯度，或者写了一句 `inputs.square().sum(dim=-1)`。要想知道完整的预测误差或特征范数，必须在被切开的 `H` 维上做跨卡求和！如果各卡只拿自己的半截特征算梯度，权重就会在毫无报错的情况下悄悄算错。

怎么在模型加载前，自动揪出这种需要跨卡交换信息的代码？写一套覆盖成百上千个 PyTorch 算子的静态分片传播分析器（Sharding Propagation Analyzer）太重了。我们在 `pllm/execution/overlay_parallel.py` 里用了一个四两拨千斤的办法：**用 `make_fx` 把作者的更新函数追踪两遍——第一遍喂单卡的完整形状（特征宽 $H$），第二遍喂切了一半的分片形状（特征宽 $H/2$）——然后把两张计算图并排放在一起“找不同”！**

```text
Case A: In-Place TTT Outer Product (ADMITTED)
  Global Trace (width = H)              Shard Trace (width = H/2)
  inputs [C, H]   values [C, Out]       inputs [C, H/2] (sharded axis=1)  values [C, Out]
         └───┬───┘                             └─────┬─────┘
             ▼ aten.mm                               ▼ aten.mm
       update [Out, H]                         update [Out, H/2]  <-- Axis 1 still sharded! OK!

Case B: Reconstruction Error / Feature Norm (REJECTED)
  Global Trace (width = H)              Shard Trace (width = H/2)
  inputs [C, H]   base.T [H, Out]       inputs [C, H/2] (sharded axis=1)  base.T [H/2, Out]
         └───┬───┘                             └─────┬─────┘
             ▼ aten.mm                               ▼ aten.mm
       pred   [C, Out]                         pred   [C, Out]    <-- Sharded axis vanished! REJECT!
```

我们根本不需要教编译器什么是外积、什么是内积，只需要比较两张图里对应节点输出张量的形状就够了：

```python
def _axis(whole: torch.Tensor, local: torch.Tensor, size: int) -> int | None:
    axes = tuple(i for i, (a, b) in enumerate(zip(whole.shape, local.shape, strict=True)) if a != b)
    if not axes:
        return None
    if len(axes) != 1 or whole.shape[axes[0]] != local.shape[axes[0]] * size:
        raise ValueError("TP updates must preserve one contiguous feature shard")
    return axes[0]
```

拿着全局图 `whole` 和分片图 `local` 对齐后的节点（剪除只用于当前步读出的死代码后）：如果两张图节点数不同，说明有依赖宽度的控制流，拒绝；如果节点吃进了至少一个带分片轴（`axis is not None`）的输入，而它自己的输出在两张图里形状突然变得一模一样了（`axis is None`），就说明**那个被切开的分片维度在收缩运算中消失了**！

一旦分片轴消失，意味着该值必须跨卡求和才能算对，`_check_graphs` 当场举红牌拒绝并给出诊断信息。不用写几万行的算子规则表，两张假张量跑出来的 FX 图自己就把分片守恒律证明了。

---

## 五、 4B TTT 实测与被 BF16 吞掉的 99% 更新

最后，我们把社区训练的 4B In-Place TTT 检查点（`zhongweixie/inplace-ttt-qwen3-4b-32k-ttt`，36 层，第 0、6、12、18、24、30 层的 `down_proj` 为 `[2560, 9728]` 快权重，块大小 `chunk_size=1024`，学习率 3.0）放上单张 RTX 4090（vLLM 0.32.0，贪心解码生成 256 token，取预热后 3 轮中位数），与官方仓库基于 Hugging Face `generate`（BF16 + FlashAttention 2）的实现同台实测。

先看跨块可见性。我们截取 Python 标准库 `argparse.py` 的前 **1023 个 token** 作为 prompt（比块大小 1024 刚好少 1 个词），连续生成 5 个 token，对比正常更新（`chunk_size=1024`）与不更新（设 `chunk_size=2048`）的对数概率：
* 生成第 1 个词（位置 1022）和第 2 个词（位置 1023，即全序列第 1024 个词）时，两组的 logprob 分毫不差（差值 `+0.0000`）——因为第 1024 个词虽然刚好凑满第一块并触发了 6 层外积更新，但按因果序它自己前向时读的仍是旧权重；
* 到了第 3 个词（位置 1024，第二块的头一个词），新版本的 PTTS 状态根已原子生效，logprob 立刻跳变了 `+0.0626`！而且这一整套 `exact` 解码更新同样被完整录进了 CUDA Graph（在另一组较早的、只对纯 decode 阶段计时的测量里——4096 prompt，BF16 官方快权重配 FP32 残差存储——录图让单请求从 eager 的 `57.59` 提升至 `59.51 tok/s`，4 并发从 `107.56` 提升至 `113.16 tok/s`）。

再看包含 prefill 的端到端生成吞吐对比：

| 实现与快权重精度（RTX 4090，4B TTT） | 1 个请求，prompt 4096 | 1 个请求，prompt 16384 | 4 个请求，prompt 4096 |
|---|---:|---:|---:|
| 官方 HF 实现（BF16） | `54.79 tok/s` | `34.89 tok/s` | 不支持（仅限 `bs=1`） |
| pLLM，BF16 快权重 | `57.12 tok/s` | `40.64 tok/s` | — |
| pLLM，FP32 快权重（默认） | **`59.64 tok/s`** | **`42.27 tok/s`** | **`111.38 tok/s`**（4 请求合计） |

在严格复现官方 BF16 快权重舍入的同精度模式下，单请求 4k 长度快约 4%，16k 长度快约 16%；当并发开到 4 个请求时，官方实现因 `bs=1` 断言无法运行，而通过 `vmap` 与分页池批处理的版本跑出了 **111.38 tok/s**。（需要注意的是，6 层 FP32 快权重加上未满块的输入行缓冲，每份状态根重达 `714.0 MiB`，4 个并发预留 8 份根就要吃掉约 5.6 GiB 显存；且因为改写了模型内部权重与残差，所有图内改写与覆盖层都必须关闭 vLLM 的 Prefix Caching。）

看着上面这张表，你可能会好奇最后一个问题：**为什么用 32 位浮点数（FP32）存快权重增量，不仅跑得比 16 位的 BF16 略快，而且被我们设成了默认选项？**

因为官方实现是把 BF16 的底座权重直接复制一份当作快权重，每次算出微小的更新量 $\Delta W$ 就直接往大权重的 BF16 上硬加（$W \leftarrow \text{BF16}(W + \Delta W)$）。

在数值计算里，这会迎头撞上**大数吃小数（Swamping）**。BF16 只有 7 位尾数。我们拿该检查点第 0 层的真实 `down_proj` 权重（共 `24,903,680` 个元素）做过一次算术验证：原论文附录 C.2 说，对 Qwen3-4B 报告的所有评测，每个块的局部更新都按 Frobenius 范数裁剪到 `1e-5`（代码里对应 `track="T2"`）；如果把这样一个范数为 `1e-5` 的更新矩阵直接用 BF16 加到底座权重上，在这 **2,490 万**个参数里，只有区区 **472 个元素**的值发生了改变！其余 99.998% 的更新因为跨不过 BF16 的最小精度台阶，直接被硬件舍入抹平了，实际生效的改变量只有预期值的 **0.97%**。

而一旦把只读的 BF16 底座权重 $W$ 和每个请求从零开始累积的 FP32 增量 `delta` 解耦开来（在前向时只算一次 `inputs @ delta.to(inputs.dtype).T` 加到底座输出上；而复现官方舍入的 BF16 模式要先合成 BF16 权重、再算两次矩阵乘求差），哪怕再微小的梯度更新也能在 FP32 的尾数里颗粒归仓。

---

回头看这趟在编译态模型内部的改造，无论是骑在 `fused_add_rms_norm` 上并用 10 行 `make_fx` 剪掉末尾加法，还是用张量地址 `data_ptr()` 穿过 `custom_op`，抑或是把 `grad`、`vmap`、`compile` 和双图形状对比叠在一起——核心思路其实只有一条：**不要跟底层编译器硬碰硬地重造轮子，去读懂它现有的融合边界与变换原语，把你的代码变成它最擅长处理的形状。**
