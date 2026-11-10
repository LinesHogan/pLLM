# 自定义层循环与状态传播

如果你在自己的研究脚本里验证过“自适应深度”（Adaptive Depth）或者“早退”（Early Exit）的想法，你写出来的原型代码大概长这样：

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

不管是 CALM、LayerSkip 这类看到置信度够高就提前交卷的早退算法，还是 Ouro、Huginn 这类把中间几层像陀螺一样反复转好几圈的循环深度（Looped Depth）模型，抑或是 Mixture-of-Depths（MoD）这种按 token 挑层走的路由机制，脱掉论文里的数学外衣，落到代码里其实就是两个最朴素的控制流原语：一个 `if`，或者一个 `for`。

在单请求、即时执行（Eager）的 Hugging Face 脚本里，这个 `if` 写起来连眼睛都不用眨一下。但当你试图把这个 `if` 搬进像 vLLM 这样的高吞吐服务引擎时，你会发现自己迎头撞上了三堵墙——而且最致命的那堵墙，甚至跟“怎么算”毫无关系。

---

## 一脚踩空的注意力

我们先来看看前两堵比较显眼的墙。

第一堵墙叫 **CUDA Graph**。看看那个条件：`cosine_similarity(h, previous) > 0.85`。余弦相似度的计算结果躺在 GPU 显存里，而 `if` 是由 CPU 上的 Python 解释器执行的。要想让 Python 知道走哪个分支，CPU 就得停下来等 GPU 把这一个布尔值跨过 PCIe 传回来——这是一次阻塞式的主机同步。更糟的是，在 decode 阶段，整个前向计算是要被录进 CUDA graph 反复回放的（见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。回放的时候 Python 早就退场了，你的 `if` 会被永远焊死在录制那一刻走过的分支上。

第二堵墙叫 **连续批处理（Continuous Batching）**。在服务引擎里，穿过模型的张量不是孤零零的一个 token，而是几十上百个请求拼在一起的二维矩阵（见[《行模型》](04-packed-rows.md)）。走到第 15 层出口时，第 0 行的 token 觉得“我已经想清楚了，第 16 到 19 层我要跳过”；第 1 行的 token 还在冥思苦想，必须老老实实算完；第 2 行则是另一个请求的块预填充。Python 的 `if` 是全有或全无的，它没法对一个张量的第 0 行说“跳”，对第 1 行说“算”。

但真正让人后背发凉的，是第三堵墙。它不发生在跳层的那一步，而发生在**下一步**。

假设你是第 $t$ 个位置的 token。你在第 15 层结束时心满意足地拍拍屁股，跳过了第 16 到 19 层，直接空降到了第 20 层，最后顺利生成了下一个词。到此为止，无事发生。

接着，第 $t+1$ 个位置的 token 进场了。它走到第 16 层，没有跳层，准备老老实实做自回归注意力（Self-Attention）。按照因果注意力的规矩，它需要回头去读从位置 $0$ 到位置 $t$ 所有前辈在第 16 层的 Key 和 Value。它伸手去摸位置 $t$ 在第 16 层的 KV 槽位——**一脚踩空了。**

因为你刚才跳过第 16 层的时候，根本没往那儿写过东西！

```text
What happens without KV propagation (Layer 16 KV Cache):
         Pos 0..t-1       Pos t (Skipped L16!)     Pos t+1 (Computing L16)
        ┌──────────────┬────────────────────────┬────────────────────────┐
Layer 15│  Valid K/V   │       Valid K/V        │       Valid K/V        │
        ├──────────────┼────────────────────────┼────────────────────────┤
Layer 16│  Valid K/V   │  [ ??? Uninitialized ] │  Attends to Pos 0..t   │
        └──────────────┴───────────▲────────────┴───────────┬────────────┘
                                   └────── Crash / Garbage! ┘
```

在 vLLM 的世界里，分页 KV 缓存、FlashAttention 内核、前缀缓存（Prefix Caching）以及抢占后的重算，全都建立在一条不可动摇的假设之上：**每一层、每一个已经算过的位置，都必须整整齐齐地躺着一份合法的 K 和 V**（我们在[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)里会详细解剖这本账）。如果你在第 16 层的物理块里留下一个未初始化的空洞，第 $t+1$ 个 token 读出来的就是显存里残留的垃圾数据，甚至是上一个已经退房的客户留下的私密对话。

所以，要在推理引擎里改写前向拓扑，我们真正要回答的第一个问题不是“怎么跳过去”，而是：**当你跳过一层时，怎么向这一层的 KV 缓存交代？**

---

## 站台上的登记簿：为什么状态传播能大一统

面对这个窟窿，学术界其实试过好几种不同的补救方案。我们在设计 pLLM 的时候把它们挨个摆在桌面上推演了一遍：

1. **干脆留个洞（原版 Mixture-of-Depths 的做法）**：既然第 $t$ 个 token 没进第 16 层，那就让后面的 token 在第 16 层做注意力时彻底无视位置 $t$。这听起来很干脆，但在系统上有两个死穴：第一，只有从头按这种“注意力缺斤少两”的方式训练出来的模型才能适应它；第二，这意味着同一个请求在第 15 层有 100 个 KV，在第 16 层可能只有 63 个 KV——每一层都得维护自己独立的块表和序列长度。而 vLLM 的调度器对每个请求只记一个标量进度，底层根本容不下这种参差不齐的账本。
2. **把上一层的 K/V 抄过来**：既然第 15 层算出了 K 和 V，直接顺手拷进第 16 层的缓存行不行？绝对不行。Transformer 每一层的注意力权重都是独立训练的，第 15 层的 $W_K, W_V$ 把隐藏状态投影到的线性子空间，跟第 16 层的 $W_Q$ 压根不在一个频道上。拿第 16 层的 Query 去跟第 15 层的 Key 做内积，算出来的注意力分数纯属胡言乱语。
3. **先欠着，以后需要时再补算（FREE 一类方法的做法）**：先把第 16 到 19 层跳过去，等下一步发现在深层确实需要它的 K/V 时，再把前一个位置漏掉的深层跟当前 token 绑在一起补算。这在单请求脚本里很聪明，但在服务引擎里，这意味着每个请求在每一步要喂给模型的 token 数量变成了捉摸不定的随机数，跟要求形状死板的 CUDA graph 直接迎头相撞。

排查完一圈，唯一在数学语义和系统工程两边都站得住脚的幸存者，是 CALM 论文提出的**状态传播（State Propagation）**。

你可以用一个坐火车的比喻来理解它。把一个 28 层的模型想象成一趟有 28 个车站的列车。第 $t$ 个 token 是一位乘客，车开到第 15 站时，他觉得自己的事情已经办妥了，决定在第 16 到 19 站不下车办事，坐在座位上闭目养神——也就是说，他的残差流（Residual Stream）原封不动地穿过这四层，等价于把第 16 到 19 层当成了恒等映射 $f(x) = x$。

但是，每一站的站台登记簿（也就是那一层的 KV Cache）上，规矩是死的：**只要这趟车上有这名乘客，登记簿上就必须有他的一行记录，否则下一班车的乘客来查账时会直接报错。** 既然他没下车，登记簿上填什么呢？

答案是：**按他坐在车窗边路过这一站时的样子登记。**

```text
State Propagation at Skipped Layer 16 (Active = False):
  h15 (Frozen Residual) ───────────────────────────────► h16 = h15 (Unchanged!)
          │
          └──► [ Layer 16 Norm ] ──► [ Layer 16 K/V Proj & RoPE ] ──► Write Layer 16 KV Cache
               ( ~7% of a layer's FLOPs; skips Attention & MLP! )
```

具体来说，我们拿这名乘客停在第 15 层末尾的隐藏状态 `h15`，送进第 16 层**自己的**输入归一化（`input_layernorm`）、第 16 层**自己的** K/V 投影矩阵和旋转位置编码（RoPE），算出属于第 16 层子空间的 Key 和 Value，恭恭敬敬地写进第 16 层的 KV 缓存里。写完之后，既不做注意力读取，也不跑庞大的 MLP，直接把原来的 `h15` 当作第 16 层的输出往下传。（按 `pllm.explain` 的成本估算，这套“只做归一化 + QKV 投影 + RoPE”的登记手续，大约只占 Qwen2.5 完整一层计算量的 **7%**；而且如果是在循环里多次经过同一层，只有第一次跳过时需要补写，后续再跳过直接保留已写入的值即可。）

仔细品味一下这个设计，你会发现它美得让人叹服。

首先，如果第 16 层的真实效果真的如早退假设所说的那样接近恒等映射（即经过这一层后隐藏状态几乎没变），那么拿 `h15` 通过第 16 层自己的投影算出来的 K/V，**在数学上恰恰就是这一层本该产出的 K/V**！近似误差被严格关在了被跳过的那个 token 自己的深层状态里，子空间完全对齐。

其次，对底层的 vLLM 引擎来说，世界一点都没变。每一层、每一个位置依然整整齐齐地存着合法的 K/V。分页块表、原生 FlashAttention、前缀缓存、抢占后的重算，所有机制连一行代码都不用改，全都照常运转。

更重要的是，它把五花八门的算法统一成了一个最干净的原语。引擎根本不需要关心你是用余弦相似度、用训练出来的退出分类头、用词表熵，还是在循环深度模型里转了三圈后决定提前停下——所有这些算法，最后都坍缩成交给模型切片的一个布尔张量 `active`。而且，补写 K/V 只用到了当前这一行自己的隐藏状态，跟同一个 batch 里拼在一起的其他租户毫无瓜葛。这意味着请求之间严格隔离，天然能塞进 CUDA graph。

不过，读懂了这个机制的代价，我们也在 pLLM 里立下了一条看似专断、实则救命的规矩：**`active=` 动态跳层只允许用在解码生成的 token 上（`tokens="decode"`），绝不允许用在处理 prompt 的 prefill 阶段。**

为什么？想想看，状态传播写进去的 K/V 终究是一份“没跑过当前层 MLP 和注意力”的近似品。在 decode 阶段，某个生成 token 跳了层，它留下的近似 K/V 只会影响当前请求接下来生成的寥寥几个词，况且它敢跳层的前提本来就是“特征已经收敛了”。但 prompt 里的 token 完全是另一码事！Prompt token 自己并不需要采样输出任何词，**它活在世界上的唯一使命，就是算出高质量的 K/V，作为整篇对话的基石供后面几百个生成 token 反复查阅。** 如果你在 prefill 阶段偷懒跳了层，整段上下文在深层的 K/V 全都变成了近似值，还会顺着 prefix cache 和 `n` 路采样污染所有的下游分支。Ouro 论文做过一个非常扎心的实验：在 decode 阶段用近似 K/V 几乎不掉点，但在 prefill 阶段一旦换用近似 K/V，GSM8K 数学准确率当场暴跌 10 个百分点以上。更别提 prefill 本来是整段连续位置一起喂给 FlashAttention 高效并行的，一旦你在中间挑出几个 token 跳层压紧，位置不再连续，省下的那点矩阵乘还不够赔给注意力拆包重排的开销。把这道门直接焊死在框架入口，能帮用户省下无数个怀疑人生的夜晚。

---

## 单 Token 的程序，整批的躯体

想通了“用状态传播守住 KV 满格不变量”，我们来看看用户手里的代码长什么样。在 pLLM 里，文章开头那个简陋的 notebook 原型，写出来是这样一段干净的声明：

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

注意看这段代码的视角：你写的是**一个位置（单 token）的思维过程**。先跑 `0..12` 层拿到第 11 层的输出 `h12`，再跑 `12..16` 层拿到第 15 层的输出 `h16`；然后比一比 `h16` 和 `h12` 的方向变了多少——如果余弦相似度小于等于阈值（说明隐藏状态还在剧烈变化，`moved` 为 `True`），就在 `layers(16, 20, hidden=h16, active=moved)` 里继续算这四层；如果相似度已经大于 `0.85`（`moved` 为 `False`），这四层就自动走状态传播跳过。

所有的玄机都浓缩在 `Layers.__call__`（位于 `pllm/execution/token_program.py`）的签名与契约里：

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
    # ……
    return LayerOutputs(tuple(_layers(value, start, stop, kv, active)), start, self.layer_count)
```

除了 `active` 掩码，这里还有一个精巧的参数 `kv`，它只有两个合法值：`"commit"`（默认）和 `"peek"`。`"commit"` 就是正常的因果写入，算完把 K/V 提交进缓存；而 `"peek"` 则像是一次不留痕迹的侦察——它允许当前层读取前面所有位置已经提交的 K/V 外加自己这一下新算出的 K/V，但**绝不往缓存里写任何东西**。当你想写一个循环深度模型（比如先试着往深层跑一圈看看置信度，如果不满意再退回来重跑），`"peek"` 和跨位置的 `PositionState`（通过 `layers.load` / `layers.save` 读写）就能让你像搭积木一样拼出各种循环拓扑。

那么，一段按“单个 token”视角写的 Python 函数，是怎么在底层变成整批高速执行的呢？

当引擎加载 `SkipMiddle` 时，`TokenProgram.trace` 会用没有实际显存的假张量（FakeTensor）把你的 `forward` 跑一遍（如果你在里面写了依赖张量数值的 Python `if`，PyTorch 在追踪期就会当场报错拦住你，见[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）。追踪出来的计算图会被刀起刀落切成两类交替执行的积木：
* **模型切片（`Segment`）**：也就是每一次 `layers(start, stop, ...)` 调用。在运行时，它们依然作为一个完整的二维 batch `[B, H]`，直接喂给 vLLM 原生的 C++/CUDA 层去整批跑。
* **函数式算子片段（`Cell`）**：也就是夹在两次 `layers(...)` 之间的那些张量运算（比如算余弦相似度、线性插值、求范数）。框架会自动用 `torch.vmap` 把每个 `Cell` 沿着行维度映射开。这样一来，你在代码里写一个标量规约或形状变换，语义上严丝合缝地局限在“当前这一个 token”内部，绝不会失手把别人的请求混进来；而在物理执行时，它又是彻头彻尾的矢量化批处理。

说起来，这套“切分成模型段与 `vmap` 算子段、整批调用原生层”的架构，是我们交过一笔昂贵的学费才换来的。在早期版本里，我们曾走过一条弯路：为了支持极度灵活的循环深度（比如 recirculation、layer_loop、step_loop），我们写了一套基于 Region IR 的逐 token 解释器，在运行期一个 token 一个 token 地解释执行层循环。结果呢？GPU 最恨的就是碎片化的小调用。后来我们在一次大清理中痛下决心，删掉了包括 `tokenwise_region.py` 和 `region_layers.py` 在内的四个解释器模块，全面改成了今天这套“按活跃波次（wave）整批调用 vLLM 原生层”的机制。

在一张 RTX 4090 上用 Qwen2.5-0.5B 做重构前后的对照（FlashAttention-2，MRv2 runner，batch 8，256 预填充 + 256 解码，外层模型 eager 模式的历史测量）：`recirculation` 的 prefill 耗时直接从 **6,689.69 ms 暴跌到 863.52 ms**，decode 从每步 **26.23 ms 降到 3.26 ms**；`layer_loop` 的 decode 从 **16.36 ms/步降到 1.93 ms/步**；`step_loop` 从 **14.04 ms/步降到 1.73 ms/步**——整整快了七八倍！这个教训刻骨铭心：在 GPU 上，任何试图绕开原生批处理层、自己逐个 token 搞解释执行的聪明主意，在硬件吞吐面前都会撞得粉碎。这个发现也为我们后来在[《自动放置与重叠》](11-automatic-placement-overlap.md)里彻底删掉庞大的多面体编译器埋下了伏笔。

---

## 借壳上市：一行 Kernel 不写的状态传播

现在，让我们卷起袖子，深入到本文最有趣的一处代码细节里去。

前面说了，当一批请求走到第 16 层，其中一部分行的 `active` 是 `False` 时，我们要为这些“坐在车上不下车”的行做状态传播：用它们停下时的 `hidden`，跑一遍第 16 层**自己的** `input_layernorm`、QKV 投影、可能存在的 QK-norm 以及 RoPE 旋转，把算出来的 K 和 V 塞进这一层的 KV cache 里，但**略过真正的注意力计算和输出投影 `o_proj`**。

听到这个需求，你的手指是不是已经放在键盘上，准备新建一个 `propagate_kv_kernel.py`，把模型层里的权重偷出来，自己调 `F.linear` 和 `rotary_emb` 了？

千万别。如果你去翻翻 vLLM 里的模型实现，你会发现不同模型家族（Qwen2、Qwen3、Llama、DeepSeek）的注意力前半截简直是五花八门：QKV 是合并成一个大矩阵还是拆成三个？有没有偏置项（bias）？有没有 QK-norm？是在重塑形状前做还是重塑后做？RoPE 是标准版还是 Neox 风格？如果你自己写一套状态传播，你就得把所有模型家族的注意力前半截全抄一遍，而且只要上游改一行代码，你的状态传播就会跟真实层产生微妙的数值漂移。

慢着——vLLM 原生的 `self_attn` 模块本来就站在那儿啊！它天生就知道怎么把这一层的 `input_layernorm`、QKV 投影、QK-norm 和 RoPE 一丝不差地跑完。它唯一的毛病是“太勤快了”：跑完这些之后，它还会接着去调 `attn.forward` 算注意力，接着去调 `o_proj.forward` 乘输出矩阵。

既然它只是太勤快，那我们在叫它干活之前，**临时把它的后半截拆掉换成两个空壳，等它帮我们把 K/V 写进缓存了，再把原装零件给它装回去**，不就完了吗？

来看看 `pllm/backends/vllm/adaptive_depth.py` 顶部那短短十几行的两个替身函数：

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

看到 `_store` 了吗？当原生的 `self_attn` 费尽千辛万苦算好了 `q, k, v`（连 RoPE 旋转都替我们做好了！），满怀期待地把它们交给底层的 `attention.attn.forward` 去跑 FlashAttention 时，半路接球的是我们的 `_store`。`_store` 根本不去读历史缓存算注意力，它只调了一句 vLLM 原生的 `do_kv_cache_update`，把新鲜出炉的 `k` 和 `v` 塞进分页缓存槽位 `slots` 里，然后随手把 `q` 原样退回去充数。紧接着，`self_attn` 准备把结果拿去乘输出投影矩阵 `o_proj`，接球的又是我们的 `_omit_projection`——它连矩阵乘都不做，直接把张量原样奉还！

有了这两个替身，真正执行状态传播的 `propagate` 方法就像一场行云流水的外科手术：

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

这里有两个让人会心一笑的小细节。

第一，盯住 `slot_mapping.masked_fill(active.flatten(), -1)` 那一行。此时传进 `propagate` 的 `hidden` 是整批所有的行（既包含要跳层的 `active=False` 行，也包含一会儿要正常计算的 `active=True` 行）。我们怎么防止状态传播去污染那些活跃行的 KV 槽位呢？答案是利用 vLLM 底层缓存写入内核的一个原生约定：**只要槽位号（slot index）是 `-1`，写入内核就会自动跳过这一行（把 `-1` 当作 padding 处理）。** 所以我们只需要用 `masked_fill` 把所有活跃行的槽位涂成 `-1`，只有不活跃行的真实槽位号留着，就能直接把整批张量喂进去！

第二，看那个 `try ... finally` 块。把 `attention.attn.forward` 和 `attention.o_proj.forward` 临时换成 `_store` 和 `_omit_projection`，调完原生归一化与 `attention(self.wave.positions, normal)`，立刻在 `finally` 里原封不动地换回来。

```text
Native self_attn during propagate() (Monkey-Patched in try/finally):
  hidden ──► [ Native input_layernorm ] ──► [ Native QKV + QK-norm + RoPE ]
                                                           │
             ┌─────────────────────────────────────────────┘
             ▼
     [ _store (replaces attn.forward) ] ──► do_kv_cache_update (active rows masked to slot -1!)
             │
             ▼
     [ _omit_projection (replaces o_proj) ] ──► return (Zero attention FLOPs, zero o_proj!)
```

一行新的 CUDA 或 Triton kernel 都没写。原生的层自己帮我们跑完了归一化、投影、旋转、甚至任何挂在 Value 上的自定义改写 hook，把不活跃行的 K/V 妥妥当当地存进了分页缓存。我每次读到这个 `try/finally`，都忍不住想给它鼓掌：最高明的系统代码，往往不是你新写了多少行算子，而是你怎么四两拨千斤地让现有的代码只干你想让它干的那一半活。

---

## 在自动钢琴里装道岔：CUDA 条件节点与 2 的幂次桶

状态传播解决了“不活跃的行怎么交代”，接下来我们要解决那个棘手的执行问题：**在不能回 CPU 同步、形状必须固定的 CUDA graph 里，怎么根据 GPU 上实时算出来的 `active` 掩码，去跑那些活跃的行？**

如果我们用的是早几年的 CUDA，这题基本无解——那时候的 CUDA graph 就像一架死板的自动钢琴，录制好什么琴谱，回放时就只能一个音符不差地从头砸到尾。但从 **CUDA 12** 的较新版本开始，NVIDIA 给这架自动钢琴装上了“道岔”：**条件节点（Conditional / IF Node）**。

在录制 CUDA graph 时，你可以创建一个 `IF` 节点，把它绑定到 GPU 显存里的一个布尔标量（predicate）上，然后往这个节点里面录制一段子图。到了回放的时候，GPU 的硬件调度器会自己在设备端看一眼那个布尔标量的值是 `0` 还是 `1`，直接决定要不要执行这段子图——全程不需要 CPU 插一句话！

我们在 `pllm/execution/cuda_graph.py` 里把这个能力包装成了一个极简的上下文管理器 `device_branch`：

```python
# check-docs: skip: excerpt from pllm/execution/cuda_graph.py
@contextmanager
def device_branch(predicate: torch.Tensor) -> Generator[None]:
    graph = (
        torch.cuda.CUDAGraph.get_currently_capturing_graph()
        if predicate.is_cuda and torch.cuda.is_current_stream_capturing()
        else None
    )
    if graph is not None:
        graph.begin_capture_to_if_node(predicate)
        _release_scope()
    try:
        yield
    finally:
        if graph is not None:
            graph.end_capture_to_conditional_node()
            _release_scope()
```

> **旁注：那两个不起眼的 `_release_scope()` 是干嘛的？**
> 如果你去翻 `cuda_graph.py` 的源码，你会发现 `_release_scope()` 上面写着三行无奈的注释。这是只有真正把 PyTorch 条件节点搬上生产的人才会踩到的暗坑：PyTorch 的 C++ 底层在调用 `begin_capture_to_if_node` 和 `end_capture_to_conditional_node` 时，会各自重新打开一次同一个 CUDA 显存池（graph pool），导致这个池子的内部引用计数平白无故加了两次；然而当你销毁图调用 `CUDAGraph.reset()` 时，它却只释放第一次捕获时的那一个引用！如果不在这里手动调两次底层接口把多出来的引用计数平衡掉，只要你的图里用了一次条件节点，整块显存池就会像幽灵一样永远驻留在显卡里，再也释放不掉。

有了 `device_branch`，我们可以在图里分叉了。但别忘了，条件节点只解决了“走哪条路”，没解决“张量形状必须固定”的问题。假设当前 batch 的总宽度 `width` 是 32 行：这一步可能有 17 行需要算，下一步可能只有 5 行需要算。总不能为 `1..32` 的每一种可能人数都录一张子图吧？

我们的解法是搭一座两层的小梯子：**先看是不是全员到齐；如果不是，就把活跃行排个队压紧（compact）到最前面，然后装进按 2 的幂次（`1, 2, 4, 8, 16, ...`）预备好的固定尺寸桶（bucket）里去执行。**

`AdaptiveSlice.__call__`（位于 `pllm/backends/vllm/adaptive_depth.py`）把这三条路写得一目了然（`sites_rows` 是给残差挂载点用的辅助上下文，可以先略过）：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/adaptive_depth.py
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

顺着 `count = active.sum()`（注意，这也是一个留在 GPU 上的标量张量！）往下读，GPU 在回放时会根据当场的活跃人数，自动扳动道岔走下面三条路之一：

```text
AdaptiveSlice Execution Branches (Width = 32, Zero Host Syncs):

                    ┌── [ IF count == 32 ] ──► Run full batch [32, H] directly (no gather/scatter)
                    │
  count = sum(active)
                    │                          1. propagate() K/V for inactive rows
                    └── [ IF count < 32  ] ──► 2. if count == 0: no bucket fires, segment skipped!
                                               3. [ IF count > 0  ] ──► argsort(~active) & compact into
                                                                        power-of-two bucket (1, 2, 4, 8, 16, 32)
```

1. **全员活跃（`count == width`）**：没有任何一行跳层。直接跳过状态传播和排序搬运，按原装宽度把整批 `hidden` 喂给模型层跑完。
2. **部分活跃（`0 < count < width`）**：先调我们刚才写的 `self.propagate` 为不活跃的行补齐 KV；然后用 `torch.argsort(~active.flatten(), stable=True)` 做一次稳定排序——因为 `~active` 对活跃行是 `0`、对不活跃行是 `1`，排序后所有活跃行的下标就整整齐齐地聚拢到了 `order` 数组的最前头！接着看 `count` 落在哪个 2 的幂次区间 `(lower, size]` 里（比如 5 行活跃就落进容量为 8 的桶），触发对应的 `device_branch`，把前 `size` 行抽出来（`compact`）跑一遍模型层（桶里多出来的凑数行同样把 KV 槽位涂成 `-1` 屏蔽掉），算完再按原下标写回 `output`。
3. **全员跳过（`count == 0`）**：跑完 `self.propagate` 补齐那 7% 的 K/V 之后，`count > 0` 的条件为假，里面所有的桶分支统统不走——**整段模型层的注意力和 MLP 算子被条件节点直接短路跳过！**

全程没有一次 `.item()`，没有一次主机同步，每一个桶在录制时都有着钉死的张量形状和预先建好的注意力元数据缓冲区。

当然，天下没有白吃的午餐。一旦你用上了 `@pllm.custom_forward`，vLLM 原本那张从第 0 层一气呵成录到最后一层的整模型 CUDA graph 就不能用了——因为我们要在层段之间插手控制流，模型的层改由 pLLM 按切片、按活跃宽度（`ProgramWave`）分段捕获。而条件节点里的每一个桶分支，在录制预热时都得实打实地跑一遍并分配各自的图内中间缓冲区。

这笔账最后会体现在 **CUDA Graph 捕获池（Capture Pool）** 的显存开销上。在 Qwen2.5-7B、32 个并发请求下，上面那个 `SkipMiddle` 负载的分段图捕获池需要大约 **143 MiB** 显存，超过了 pLLM 默认给单个任务划拨的 64 MiB 预算（早先在 Qwen2.5-0.5B 上测得的另一组数字是：32 并发约需 748 MiB，128 并发约需 1,392 MiB）。目前 CPU 侧的 `pllm.explain` 还估算不出这部分由 CUDA 驱动在捕获期动态吃掉的池子大小；不过当预算不够时，运行期会立刻抛出明确的异常，把精确到字节的所需大小（`custom-forward CUDA graphs need ... bytes; raise ExecutionOptions.cuda_graph_pool_budget_bytes above ...`）直接报给你。

与此同时，改写前向拓扑还会带来几条自然的语义边界：既然一层可能被跳过、也可能在循环里被绕好几圈，“第 $L$ 层的唯一输出”这个概念就不再成立了。所以在同一个 Workload 里，你不能再混用读取固定层唯一输出的 `pllm.hidden` / `pllm.final_hidden`（要改读基于残差流的 `pllm.residual` 或 `pllm.layer_logits`），也不能混用 `pllm.model.Adapter`。至于像 Mixture-of-Recursions 里那种“在整个 batch 内部按分数搞 top-$k$ 竞争（Expert-Choice）”的路由写法，因为会让张三的 token 跳不跳层取决于今天跟他拼在同一个 batch 里的李四考了多少分，直接违反了多租户请求隔离铁律，pLLM 的契约检查器会毫不留情地将它拒之门外。

---

## 反直觉的账单：为什么跳过 23% 的 Token 一点也不快？

机制搭好了，零同步也做到了，现在到了最刺激的环节：把这个跳层程序放到真实的显卡上跑一跑，看看它到底能给我们省下多少时间。

> **本篇实验环境声明**：除非另有说明，以下吞吐与分布数据均在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0、PyTorch 2.13（CUDA 12.9）、`Qwen/Qwen2.5-7B-Instruct`（BF16，共 28 层，隐藏维 3584）下测得。吞吐测试使用 `pllm.benchmark.throughput`：32 个并发请求（8 条 prompt $\times$ 每条采样 $n=4$），每条生成 256 个 token，预热 1 轮后取 3 轮完整生成（含 prefill）的中位数。

在示例 12（`doc/examples/12_adaptive_depth.ipynb`）里，我们让 `SkipMiddle` 在第 15 层出口检查第 12 到 15 层前后的余弦相似度，决定要不要跳过第 16 到 19 层（共 4 层，占 28 层的 $1/7$）。我们先用 `@pllm.export` 统计了 504 个 decode token 的余弦相似度分布，发现中位数在 `0.824` 左右：当阈值设为 `0.90` 时，没有任何 token 跳层；当阈值设为 **`0.85`** 时，大约有 **23% 的 token** 会跳过这 4 层；当阈值降到 `0.80` 时，有 72% 的 token 跳层。

凭直觉想：既然在阈值 `0.85` 下有将近四分之一的 token 跳过了这 4 层，而且我们又用 2 的幂次桶把活跃行压紧了，吞吐怎么着也该往上涨一点吧？

看看实测出来的成绩单：

| 运行配置（Qwen2.5-7B，32 并发，第 16–19 层可跳） | 实际吞吐 (tok/s) | 相对原生 CUDA Graph |
|---|---:|---:|
| 原生 vLLM，CUDA graph | 1,727 | 100% |
| 原生 vLLM，eager（关闭 CUDA graph） | 1,703 | 99% |
| `SkipMiddle`，从不跳（`threshold=2.0`） | 1,703 | 99% |
| `SkipMiddle`，阈值 `0.85`（**约 23% 的 token 跳过**） | **1,692** | **98%** |
| `SkipMiddle`，总是跳（`threshold=-2.0`，100% 跳过） | **1,928** | **112%** |

盯住倒数第二行：**23% 的 token 跳过了 4 层，总吞吐不仅没涨，反而从 1,703 tok/s 微跌到了 1,692 tok/s（98%）！** 只有当我们把阈值设成 `-2.0`、强行让 batch 里的**所有 token 100% 一起跳过**这 4 层时，吞吐才猛地蹿升到 **1,928 tok/s（112%）**。

（顺便一提，早先在 0.5B 小模型上跑同样的实验，趋势一模一样：从不跳是原生的 93%，约 14% 的 token 跳过时降到 90%，总是跳才升到 102%——只不过在 0.5B 上一步 decode 只有两三毫秒，如果不开图退回 eager 模式，吞吐会暴跌到 45%；而在 7B 上一步 decode 长达约 19 ms，CPU 发射开销被庞大的矩阵乘掩盖了，所以原生 eager 也能跑出 99%。）

为什么会这样？难道我们的压紧桶白写了？

一点也没白写，只是我们在跟一条冷酷的硬件物理定律掰手腕：**在短上下文的 decode 阶段，GPU 的瓶颈根本不是算力（FLOPs），而是把模型权重从显存搬进寄存器的内存带宽（Memory Bandwidth）！**

让我们算一笔简单的账。Qwen2.5-7B 总共有 28 层，BF16 权重约 15 GiB，除去词表嵌入和 LM head，平均下来**每一层的权重就有大约 466 MB**。当 32 个并发请求（32 行）走到第 16 层时，假设有 8 行（25%）决定跳层，剩下的 24 行决定继续算。我们把这 24 行压紧到了容量为 32（或者当有 16 行跳层时压紧到容量为 16）的桶里去调第 16 层。

这时候，GPU 的 Tensor Core 确实少做了 8 行的浮点乘加运算。可是，为了给剩下的那 24 行——**哪怕只剩下孤零零的 1 行！**——计算矩阵乘法，GPU 依然必须老老实实地把这一层重达 **466 MB** 的权重矩阵，从 HBM 显存里一个字节不落地完整读进 SM 一遍！

```text
Why Partial Skipping Doesn't Speed Up Short-Context Decode:

  Case A: 24 of 32 rows active (23% skipped)
  ┌─────────────────────────┬───────────┐
  │ 24 Active Rows (Compute)│ 8 Skipped │ ──► Still must read 100% of Layer Weights (466 MB / layer)!
  └─────────────────────────┴───────────┘     + Extra cost: ~7% propagation + argsort/gather/scatter

  Case B: 0 of 32 rows active (100% skipped -> Bucket 0)
  ┌─────────────────────────────────────┐
  │          All 32 Rows Skipped        │ ──► Conditional IF node skips the whole segment!
  └─────────────────────────────────────┘     Saves 4 / 28 layers of weight reads (~1.86 GB) -> 112% tok/s!
```

在每次只喂几十行的瘦长矩阵乘（Skinny GEMM）里，把 466 MB 权重从显存搬过来的时间占了绝对大头，而多算几行或少算几行激活值的时间几乎可以忽略不计。部分跳层时，466 MB 的权重读取一分钱没省下，反而还要额外掏腰包付一笔约占单层 7% 的状态传播费，外加一次 `argsort` 和 `index_select` / `index_copy_` 的搬运费。一进一出，自然就从 99% 微跌到了 98%。

反过来，当整批 32 行全部决定跳层（落进桶 0）时，奇迹发生了：条件节点 `device_branch(count > 0)` 直接把第 16 到 19 层的整个执行段短路掉了。GPU 完完全全省下了 4 层、合计约 **1.86 GB** 的权重读取流量！28 层里少读了 4 层（$4/28 \approx 14.3\%$），扣掉一点点状态传播的开销，实测吞吐提升到 **112%**，跟物理带宽账相符。

那么，按行的部分跳过与压紧桶，在什么场景下才能真正显出威力呢？有两个地方：
1. **当上下文变得极长（几千到几万 token）时**：那时候每一步 decode 不仅要读一层 466 MB 的权重，每一行还要去扫自己长达几万个位置的全部历史 KV 缓存。当历史 KV 的读取量反超层权重时，被跳过的行因为直接略过了 `attn.forward`，就能按跳过的行数实打实地省下大块的历史 KV 带宽（原理如此，我们尚未在长上下文基准上专门实测这一交叉点）。
2. **当 batch 极大、计算进入算力受限（Compute-bound）区时**：一旦矩阵乘不再被权重读取卡脖子，把 128 行压紧成 64 行，就能真金白银地省下一半的 Tensor Core 算力时间。

最后，别忘了看一眼生成质量。在同一个实验里，我们用 `pllm.testing.check_distribution`（通过 Teacher Forcing 逐位置对比跳层模型与原模型的下一词概率分布）测量了分布偏移：在阈值 `0.85`（23% 跳过）时，平均 KL 散度就已经到了 **0.131**（TV 距离 **0.07**）；阈值降到 `0.80`（72% 跳过）时，KL 升到 **0.562**（TV **0.18**）；而全部跳过时，KL 高达 **0.868**（TV **0.25**）。虽然生成的英文句子读起来依然通顺，但概率分布其实已经漂移得很远了——因为原装的 `Qwen2.5-7B-Instruct` 在预训练时压根没见过“中间四层突然消失”的场面。系统底座能保证的是让“跳层”这件事在工程上跑得正确、隔离且零同步；但到底能不能跳、跳完之后模型考多少分，永远得靠专门训练过的权重（如 LayerSkip、CALM 或 Ouro 检查点）和你的任务评测来说话。

---

## 回头看那张登记簿

每次回顾 `@pllm.custom_forward` 的演进，我都会想起我们最初面对“跳层之后 KV 缓存少了一块”时的抓耳挠腮。

在系统设计里，当你面对一种动态性（比如有的 token 走 24 层，有的走 28 层），最容易掉进去的陷阱就是顺着动态性去改底层的数据结构——去搞变长的每层页表、去改注意力的掩码内核、去写逐 token 的解释器。一旦走上那条路，你会发现整个引擎里所有假设了“规整形状”的部件都在跟你作对。

状态传播教会我们的是另一种哲学：**不要让上层的动态控制流泄漏成底层数据结构的形状碎片；相反，在动态与静态交界的边界上花 7% 的代价把账补齐，让底层的存储与算子看到的永远是一个满格、规整、风平浪静的世界。**

而当我们需要补上那 7% 的 K/V 时，甚至连一行新算子都不用写——轻轻拧下原生 Attention 模块后半截的两个螺丝，借它的手把活干完，再在 `finally` 里把螺丝拧回去。有时候，最好的代码，确实是你克制住冲动、最终没有写出来的那一行。
