# 自定义层循环与状态传播

在大多数推理系统的眼里，一座 Transformer 模型是一条笔直的单行道：第 0 层进去，第 1 层出来，一路走到第 $L-1$ 层，谁也不能插队，谁也不能抄近道。

但在算法研究者的草稿纸上，前向传播的拓扑结构要活跃得多。比如早退（Early Exit，如 CALM、LayerSkip）与跳过中间层：当一个 token 走到第 15 层出口时，如果它的隐藏状态相比第 11 层已经几乎不再变化，说明模型对这个位置的语义已经拿准了，何必再去跑第 16 到 19 层？再比如循环深度（Looped Depth，如 Ouro、Huginn）：把中间几层当成循环体，同一个 token 反复在里面转上三四圈，以参数复用换取计算深度。

如果在单请求、即时执行（Eager）的研究脚本里验证这些想法，写出来的代码无非是一个 `if` 或一个 `for`：

```python
# check-docs: skip: illustrative prototype, not pLLM code
h = embeddings
for layer in layers[:16]:
    h = layer(h)
if cosine_similarity(h, previous) > 0.85:
    h = skip(h)
else:
    for layer in layers[16:20]:
        h = layer(h)
```

这几行代码在笔记本里跑得顺畅无比。可一旦你试图把它塞进像 vLLM 这样的连续批处理服务引擎，整个执行管线会立刻卡死。

表面上看，最先拦路的是控制流与批处理的冲突。那个 `if` 的条件依赖 GPU 张量的实时数值，如果让 CPU 上的 Python 解释器来判断走哪个分支，CPU 就必须停下来等待 GPU 把标量结果跨过 PCIe 传回来——这不仅触发了阻塞式的主机同步，而且在解码阶段，整座模型是要被录进 CUDA Graph 反复回放的（见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)），回放时 Python 根本不在场，分支会被永远焊死在录制那一刻走过的路径上。就算把分支搬到 GPU 上，连续批处理里穿行的是几十上百个请求拼成的二维矩阵 `[rows, hidden]`（见[《行模型》](04-packed-rows.md)）：走到第 15 层出口时，第 0 行的 token 想跳过第 16 到 19 层，第 1 行的 token 却还得老老实实往下算，同一个批次在层间发生了分流。

然而，哪怕你把零同步分支和批次分流都解决了，还有一个更隐蔽的麻烦在下一步等着你。

假设第 $t$ 个位置的 token 在第 15 层结束时满足了跳过条件，绕开第 16 到 19 层，直接去第 20 层完成了后续计算并采样出了新词。在这一步里，一切风平浪静。接着，第 $t+1$ 个位置的 token 进场了。它在第 16 层没有跳层，需要照常执行自回归因果注意力（Self-Attention）——按照规矩，它要回头读取从位置 $0$ 到位置 $t$ 所有前序位置在第 16 层的 Key 和 Value。

当第 $t+1$ 个 token 去第 16 层的 KV 缓存里查找位置 $t$ 的槽位时，它一脚踩空了。因为位置 $t$ 的 token 刚才根本没有进过第 16 层，那个物理槽位里什么都没写。

```text
Layer 16 KV Cache without state propagation:

         Pos 0..t-1       Pos t (Skipped L16)      Pos t+1 (Computes L16)
        ┌──────────────┬────────────────────────┬────────────────────────┐
Layer 15│  Valid K/V   │       Valid K/V        │       Valid K/V        │
        ├──────────────┼────────────────────────┼────────────────────────┤
Layer 16│  Valid K/V   │  [ Uninitialized ! ]   │  Attends to Pos 0..t   │
        └──────────────┴───────────▲────────────┴───────────┬────────────┘
                                   └───── Reads garbage ────┘
```

在现代推理引擎里，分页 KV 缓存、FlashAttention 内核、前缀缓存（Prefix Caching）以及抢占后的重算，全都建立在一条底层记账不变量之上：每一层、每一个已经算过的逻辑位置，都必须存有一份合法的 K 和 V（详见[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)）。如果在第 16 层的物理块里留下一个未初始化的空洞，第 $t+1$ 个 token 读出来的就是显存里残留的旧数据，甚至是上一个已退场请求留下的私密上下文。

因此，要在服务引擎里改写层间控制流，真正需要先回答的不是“怎么把几层跳过去”，而是：当一个 token 跳过了某一层时，怎么向这一层的 KV 缓存交账？

---

## 为什么选状态传播：几种补救思路的取舍

面对被跳过层留下的 KV 空洞，文献里有过几种截然不同的处理思路。把它们放进分页推理引擎的约束下逐一推演，很快就能看清各自的边界。

第一种思路是“干脆留空洞”，也就是原版 Mixture-of-Depths（MoD）的做法：既然位置 $t$ 没进第 16 层，那就让后续所有 token 在第 16 层做注意力时彻底跳过位置 $t$。这要求模型从预训练开始就适应“注意力历史缺斤少两”的分布；更棘手的是系统账本——这意味着同一个请求在第 15 层有 100 个 KV 条目，在第 16 层可能只有 63 个条目。每一层都得维护自己独立的块表和序列长度，而 vLLM 的调度器对每个请求只维护单一的标量进度 `num_computed_tokens`，底层数据结构根本无法表达这种参差不齐的稀疏度。

第二种思路是“拿上一层的 K/V 充数”：既然第 15 层刚刚算出了合法的 K 和 V，顺手拷一份塞进第 16 层的缓存槽位行不行？在数学上行不通。Transformer 每一层的注意力投影矩阵都是独立训练的，第 15 层的 $W_K, W_V$ 把隐藏状态映射到的线性子空间，与第 16 层的 $W_Q$ 并不对齐。用第 16 层的 Query 去和第 15 层的 Key 做内积，算出来的注意力权重是没有意义的。

第三种思路是“先欠着，等下一步需要时再补算”，比如 FREE 一类早退算法：当位置 $t$ 提前退出后，如果发现位置 $t+1$ 没有早退、确实需要读位置 $t$ 在深层的 K/V，就在跑位置 $t+1$ 的时候把位置 $t$ 漏掉的深层计算捆在一起补上。在单请求脚本里这很巧妙，但在服务引擎里，这意味着每个解码请求在每一步实际要送进深层的 token 数变成了依赖历史路径的动态变量，直接与要求静态张量形状的 CUDA Graph 产生冲突。

排除掉这三条路后，剩下的就是 CALM（Confident Adaptive Language Modeling）提出的**状态传播（State Propagation）**。

状态传播的逻辑非常直接：当某一行在某一段层上 `active=False` 时，它的残差流（Residual Stream）原封不动地穿过这段层——也就是说，我们把这段被跳过的层近似看作了恒等映射 $f(x) = x$。既然假设这一层的隐藏状态没有变，那么这一层本该产出的 K 和 V 是什么？就是拿它停下时的隐藏状态，经过被跳过层**自己的**输入归一化（`input_layernorm`）、**自己的** K/V 线性投影和旋转位置编码（RoPE），算出来的那份 K 和 V。

```text
State Propagation at Skipped Layer 16 (active = False):

  h15 (Frozen Residual) ───────────────────────────────► h16 = h15 (Unchanged)
          │
          └──► [ Layer 16 Norm ] ──► [ Layer 16 K/V Proj & RoPE ] ──► Write Layer 16 KV Cache
               ( ~7% of a layer's compute; omits Attention & MLP )
```

我们在第一次跳过该层时把这份 K/V 补写进缓存（按 Qwen2.5 的结构核算，做完归一化、QKV 投影与 RoPE 大约只占完整一层计算量的 7%；如果是在层循环里后续再次跳过同一层，直接保留上一次已提交的 K/V 即可）。

把状态传播作为统一默认机制，一下子解开了引擎里的几个死结：

首先是**下游机制完全无感**。每一层、每一个位置的物理槽位里都按时填入了子空间对齐的 K/V。分页 KV 管理、原生的 FlashAttention 内核、前缀缓存以及抢占重算，全都照常工作，不需要感知前面有谁跳了层。

其次是**把“怎么决定跳”与“怎么执行跳”彻底解耦**。无论是像上面那样比对余弦相似度阈值、用训练出来的线性退出头、看词表分布的熵，还是在循环深度模型里每转一圈检查哪些行已经收敛可以提前停下，所有五花八门的算法在进入模型切片时，都归结为同一个形状为 `[rows, 1]` 的布尔掩码 `active`。

再者是**天然满足多租户隔离与零同步**。补写 K/V 只用到当前行自己停下时的隐藏状态，与同一批次里的其他请求没有任何数据交叉（相比之下，像 Mixture-of-Recursions 中那种在整个 batch 内按分数做 top-$k$ 竞争的 expert-choice 路由，会让张三的 token 算不算这一层取决于今天刚好和哪个李四拼在同一个 batch 里，破坏了请求隔离，因此被 pLLM 明确排除）。

不过，理解了状态传播是一份“基于恒等假设的近似 K/V”，也就立刻能看懂一条至关重要的工程边界：**动态跳层掩码 `active=` 只应该用在解码生成的 token 上（`tokens="decode"`），而不应该用在处理提示词的预填充（prefill）阶段。**

为什么？在 decode 阶段，某个新生成的 token 跳过深层，它留下的近似 K/V 只会影响同一请求接下来生成的几个词，而且它敢跳层的前提本来就是当前特征已经趋于稳定。但 prompt 里的 token 完全不同：prompt token 自己不需要采样输出任何词，它存在的唯一目的，就是算出精确的 K/V 缓存，作为整段对话的基石供后续所有生成步反复读取。如果在 prefill 阶段跳层，整段上下文在深层的 K/V 都会变成近似值，误差不仅会贯穿整个回答，还会顺着 prefix cache 和并行采样扩散出去。Ouro 论文里的实测也印证了这一点：在 prefill 阶段使用近似 K/V 会让 GSM8K 数学评测掉点超过 10 个百分点，而在 decode 阶段使用则几乎没有损失。此外，prefill 本来是把连续的序列位置整块喂给 FlashAttention 高效计算的，一旦在中间抽出一部分位置跳层压紧，序列位置不再连续，省下的一点矩阵乘往往还抵不上注意力拆包重排的额外开销。因此，在 pLLM 中，prompt 默认总是跑完完整的模型深度；如果你尝试在 prefill 方法上传入 `active=`，构建期会直接报错拦下。

---

## 单位置的程序，整批次的执行

确定了底层语义，我们来看算法作者手里的代码接口。在 pLLM 中，开头那个跳过第 16 到 19 层的想法可以写成下面这个 `SkipMiddle` 负载：

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

方法接收一个 `layers: pllm.model.Layers` 对象。调用 `layers(start, stop, ...)` 就代表运行模型从 `start` 到 `stop` 的左闭右开层切片，并用正负下标（如 `[15]` 或 `[-1]`）取出对应层之后的残差流。在 `pllm/execution/token_program.py` 中，`Layers.__call__` 的签名定义了全部操作原语：

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

除了控制哪些行参与计算的 `active` 掩码，这里还有个 `kv` 参数，支持 `"commit"`（默认）与 `"peek"` 两种模式。`"commit"` 会把算出来的因果 K/V 正式写入当前位置的缓存；而 `"peek"` 则允许当前层在注意力计算时读取此前所有已提交位置的 K/V 外加自己刚算出的 K/V，但**不改动缓存里的任何内容**。当你想写一个循环深度模型——比如先用 `"commit"` 跑过一段层，再在循环里用 `"peek"` 反复迭代修正隐藏状态，或者配合 `PositionState`（通过 `layers.load(name, offset=-1)` 读取上一 token 保存的快照、用 `layers.save(name, value)` 写入当前快照）做跨位置的状态传递——这两个参数就能组合出各种前向拓扑。

要注意的是，作者在 `forward` 里写下的张量算术（如 `F.cosine_similarity`），在心智模型上是针对**单个 token** 编写的，但在运行期必须面对一整个 batch 的并发请求。怎么把这两件事统一起来？

如果用逐 token 的解释器在运行期一个位置接一个位置地去调模型层，GPU 的并行算力会被彻底切碎。我们在 RTX 4090 上用 Qwen2.5-0.5B 做过对比测量（FlashAttention-2，MRv2 runner，batch 大小 8，256 预填充 + 256 解码，贪心采样，外层模型 eager 模式）：如果按逐 token 的区域解释器执行层循环，`recirculation` 负载的 prefill 耗时高达 6,689.69 ms，decode 每步要 26.23 ms；而一旦改为按整批波次（wave）直接调用 vLLM 原生的 `[B, H]` 层，prefill 耗时立刻降到 863.52 ms，decode 降到 3.26 ms/步（同类测试中 `layer_loop` 的 decode 从 16.36 ms/步降到 1.93 ms/步，`step_loop` 从 14.04 ms/步降到 1.73 ms/步）。

为了既保留作者“按单位置写算术”的直觉，又让模型层始终跑在整批原生路径上，`TokenProgram.trace` 在加载阶段用 FakeTensor 对 `forward` 做了一次符号追踪（任何依赖张量数值的 Python `if/while` 都会在此时被拦截报错，见[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)），将整个方法拆解成两类交替排列的步骤：

1. **模型层切片（`Segment`）**：即每次 `layers(start, stop, ...)` 调用。在运行期，它们保持完整的 `[B, H]` 批次形态，直接交给 vLLM 原生层一次性执行。
2. **函数式张量片段（`Cell`）**：即夹在两个层切片之间的普通 PyTorch 运算（如求余弦相似度、范数、门控插值）。框架自动用 `torch.vmap` 将每个 `Cell` 沿着第 0 维（行维度）包装起来。

因为 `Cell` 被 `vmap` 隔离在了单行语义内，作者在里面写任何归约或变形操作都只作用于当前 token 自身，不可能误把同一批里其他租户的数据混进来；而一旦跨过 `Cell` 进入 `Segment`，所有的行又重新合流成紧凑的二维矩阵冲进原生层。

---

## 状态传播的实现：不写新算子，只拆装两个函数指针

现在来看最关键的工程细节：当执行流进入一个带 `active` 掩码的 `Segment`，且其中一部分行的 `active` 为 `False` 时，状态传播（只算 `input_layernorm`、QKV 投影、QK-norm 和 RoPE 并写入 KV 缓存，跳过注意力读取和 `o_proj`）在代码里是怎么实现的？

最容易想到的做法是自己写一个专门的 `propagate_kv` 函数或 Triton 内核：把该层的归一化权重和 QKV 权重取出来，自己调线性层、自己算 RoPE、自己往分页缓存里写。

但只要你翻开 vLLM 的模型库看一眼，就会打消这个念头。在不同的模型家族（Qwen2、Qwen2.5、Qwen3、Llama、DeepSeek）之间，注意力前半截的细节千差万别：QKV 是合并成一个 `qkv_proj` 还是拆成三个矩阵？有没有偏置？有没有像 Qwen3 那样的逐头 QK-norm？RoPE 是哪个变体？如果自己另写一套投影逻辑，就等于要把所有模型家族的前半截全部复制维护一遍，但凡有一处细节没对齐，补写进去的 K/V 就会和正常层算出来的发生数值偏离。

换个角度看：vLLM 原生的 `layer.self_attn` 模块本来就站在那里，它自己最清楚怎么把这一层的 QKV 投影、QK-norm、RoPE 以及任何挂载在 Value 上的自定义改写一丝不差地跑完。它唯一多做的事情，是在算完 `q, k, v` 之后继续调用了 `attention.attn.forward`（执行注意力并写缓存）和 `attention.o_proj.forward`（执行输出投影）。

既然如此，我们根本不需要写任何新算子，只需要在调用原生 `self_attn` 之前，把后面这两个步骤临时替换掉。`pllm/backends/vllm/adaptive_depth.py` 顶部的十来行代码定义了两个极简的替身函数：

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

`_store` 接住原生前半截算好的 `q, k, v`，不去跑 FlashAttention，而是直接调用 vLLM 原生的 `do_kv_cache_update` 把 `k` 和 `v` 写进指定的缓存槽位 `slots`，然后随手把 `q` 原样返回；紧接着 `_omit_projection` 接住返回值，跳过 `o_proj` 矩阵乘直接返回。

有了这两个替身，`AdaptiveSlice.propagate` 的主体就非常清楚了：

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

这里有两处值得留意的配合：

第一处是 `slot_mapping.masked_fill(active.flatten(), -1)`。传进 `propagate` 的 `hidden` 是整批张量，既包含 `active=False` 的跳层行，也包含 `active=True` 的活跃行（活跃行一会儿还要去跑真正的模型层）。为了只写跳层行的 KV 而不覆盖活跃行的槽位，这里直接利用了 vLLM 缓存写入内核的一个原生约定：凡是槽位号为 `-1` 的行，内核都会当作填充行自动跳过写入。因此，把所有 `active=True` 的位置掩码成 `-1`，就能直接把整批张量原样送进去。

第二处是 `try ... finally` 块：在调用 `attention(self.wave.positions, normal)` 前把 `attn.forward` 和 `o_proj.forward` 换成 `_store` 与 `_omit_projection`，调用结束立刻恢复原状。

```text
Native self_attn during propagate():

  hidden ──► [ Native input_layernorm ] ──► [ Native QKV + QK-norm + RoPE ]
                                                           │
             ┌─────────────────────────────────────────────┘
             ▼
     [ _store (replaces attn.forward) ] ──► do_kv_cache_update (active rows masked to -1)
             │
             ▼
     [ _omit_projection (replaces o_proj) ] ──► return (skips attention read & o_proj GEMM)
```

没有写一行新的算子内核，也没有复制任何模型家族的投影代码，仅仅通过临时替换两个函数属性，就借用上游模块自己的手把状态传播干干净净地做完了。

---

## 图内分叉：CUDA 条件节点与 2 的幂次压紧桶

补齐了不活跃行的 KV，接下来轮到那些 `active=True` 的活跃行真正执行模型层了。

难点在于：在录制好的 CUDA Graph 内部，CPU 并不参与每一步的控制流，而且图里的算子要求固定的输入张量形状。如果当前批次总共有 32 行，这一步有 17 行需要算，下一步只有 5 行需要算，怎么在不通知 CPU、也不重录图的前提下动态执行？

在较新的 CUDA 12 版本里，CUDA Graph 引入了设备端**条件节点（Conditional / IF Node）**。在录制图时，我们可以将一个 `IF` 节点绑定到显存里的一个单元素布尔张量（predicate）上，并在节点内部录制一段子图；到了回放阶段，GPU 硬件调度器会自己在设备端读取那个布尔值，决定是否执行子图。`pllm/execution/cuda_graph.py` 里的 `device_branch` 封装了这一机制：

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

（在非捕获的预热阶段，`graph` 为 `None`，代码会直接顺序跑过每一个分支以完成算子初始化，完全不去读取 `predicate` 的值，从而避免触发主机同步。代码里的两次 `_release_scope()` 则是为了填平 PyTorch 底层的一个引用计数缺口：PyTorch 的 `begin_capture_to_if_node` 和 `end_capture_to_conditional_node` 每次都会重新打开同一个图显存池并递增其使用计数，但 `CUDAGraph.reset` 销毁图时只释放最初的那个引用；如果不手动释放多出来的两次作用域引用，用过条件节点的图显存池就会发生泄漏。）

有了 `device_branch`，我们还需要解决“活跃行数每步变化”与“图内形状固定”的矛盾。`AdaptiveSlice.__call__` 的做法是预先为当前波次宽度准备一组 2 的幂次桶（如 `1, 2, 4, 8, 16, ...`，每个桶配有固定形状的注意力元数据缓冲区），然后在设备端按活跃行数 `count = active.sum()` 走三条分支：

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

把这段嵌套的条件分支画成执行流图，逻辑就一目了然了：

```text
AdaptiveSlice device-side branching (width = 32, zero host sync):

                    ┌── [ IF count == 32 ] ──► Run full batch [32, H] directly
                    │
  count = active.sum()
                    │                          1. propagate() K/V for inactive rows
                    └── [ IF count < 32  ] ──► 2. if count == 0: no bucket fires (segment skipped!)
                                               3. [ IF count > 0  ] ──► argsort(~active) -> compact into
                                                                        bucket size in {1, 2, 4, 8, 16}
```

* 当 `count == width`（全员活跃，没有一行跳层）时，直接按原批次跑完模型切片，不做状态传播，也不做排序搬运。
* 当 `0 < count < width`（部分行跳层）时，先调用 `self.propagate` 为不活跃行补齐 K/V；随后通过 `torch.argsort(~active.flatten(), stable=True)` 在设备端将所有 `active=True` 的行排到数组最前端，并根据 `count` 落入的 `(lower, size]` 区间激活唯一对应的那个 2 的幂次桶。`self.compact` 抽出前 `size` 行跑完模型层（桶内凑数的非活跃行同样把 `slot_mapping` 设为 `-1` 屏蔽写入），再把结果散射写回 `output`。
* 当 `count == 0`（整批所有行全都跳层）时，跑完轻量级的 `self.propagate` 后，`count > 0` 条件为假，内部所有桶全部跳过——整段模型层的注意力和 MLP 算子根本不会发射。

---

## 带宽的账单：为什么跳过 23% 的行不会让解码变快？

整套机制在设备端闭环了。现在我们把它放到真实的硬件上量一量：在解码阶段跳过一部分 token，到底能省下多少时间？

我们在单张 NVIDIA RTX 4090（24 GiB，vLLM 0.32.0，PyTorch 2.13，开启 CUDA Graph 与 FlashInfer 采样器）上，用 28 层的 `Qwen/Qwen2.5-7B-Instruct`（BF16，隐藏维 3584）运行示例 12 的 `SkipMiddle` 负载：根据第 12 到 15 层前后的余弦相似度，决定是否跳过第 16 到 19 层（共 4 层，占全模型层数的 $1/7$）。测试并发为 32 条序列（8 条 prompt $\times$ 每条采样 $n=4$），每条生成 256 个 token，取预热 1 轮后的 3 轮中位数。

先用 `@pllm.export` 看一眼 504 个解码 token 上余弦相似度的真实分布：中位数在 `0.824` 附近。当阈值设为 `0.90` 时，没有 token 跳过；当阈值设为 `0.85` 时，约有 **23% 的 token** 跳过了第 16 到 19 层；当阈值降到 `0.80` 时，约有 72% 的 token 跳层。

然而，端到端的吞吐数字初看之下却相当反直觉：

| 运行配置（Qwen2.5-7B，32 并发，第 16–19 层可跳） | 实测吞吐 (tok/s) | 相对原生 CUDA Graph |
| :--- | ---: | ---: |
| 原生 vLLM，CUDA Graph | 1,727 | 100% |
| 原生 vLLM，Eager（关闭 CUDA Graph） | 1,703 | 99% |
| `SkipMiddle`，从不跳（`threshold=2.0`） | 1,703 | 99% |
| `SkipMiddle`，阈值 `0.85`（约 23% 的 token 跳过） | 1,692 | 98% |
| `SkipMiddle`，总是跳（`threshold=-2.0`，100% 跳过） | 1,928 | 112% |

看表格的第四行：明明有 23% 的 token 跳过了 4 层计算，而且活跃行已经被压紧到了更小的桶里，为什么总吞吐不仅没涨，反而从 1,703 tok/s 微跌到了 1,692 tok/s（98%），直到让整批 100% 全部跳过时才涨到 1,928 tok/s（112%）？在 `Qwen2.5-0.5B` 小模型上（较早在 vLLM 0.27.1 上的一次测量，第 16–19 层可跳）也能观察到完全一致的现象：从不跳为原生图吞吐的 93%，约 14% 的 token 跳过时为 90%，只有整批全跳时才升至 102%（0.5B 单步极短，原生关闭 CUDA Graph 退回 eager 会直接掉到 45%；而 7B 在 32 并发下每步耗时约 19 ms，CPU 发射开销被长算子掩盖，原生 eager 仍能保持 99%）。

原因不在于分支或桶的代码不够快，而在于短上下文解码阶段的物理瓶颈：**此时决定一层跑多快的不是算力（FLOPs），而是把模型权重从显存读进计算单元的内存带宽。**

算一笔简单的账就清楚了。`Qwen2.5-7B` 的 BF16 权重一共约 15 GiB；除去词嵌入和 LM head，28 层里每一层的注意力与 MLP 权重加起来大约有 **466 MB**。在 32 个并发请求的解码步里，激活张量只有薄薄的 32 行。假设在某一步，32 行里有 8 行（25%）满足条件跳过了第 16 层，剩下 24 行需要继续计算。

```text
Short-context decode is weight-bandwidth bound (466 MB of weights per 7B layer):

  Partial skip (24 of 32 rows active):
  ┌─────────────────────────┬───────────┐
  │ 24 Active Rows (Compute)│ 8 Skipped │ ──► Must still read 100% of layer weights (466 MB/layer)
  └─────────────────────────┴───────────┘     + pays ~7% KV propagation + argsort/gather/scatter

  Full batch skip (0 of 32 rows active -> Bucket 0):
  ┌─────────────────────────────────────┐
  │          All 32 Rows Skipped        │ ──► Conditional IF node skips the whole 4-layer slice
  └─────────────────────────────────────┘     Saves 4 / 28 layers of weight reads (~1.86 GB) -> 112%
```

为了给剩下的那 24 行——哪怕压紧后桶里只剩下孤零零的 **1 行**——执行矩阵乘法，GPU 依然必须把第 16 层到第 19 层每层 466 MB、四层合计 **1.86 GB** 的权重矩阵，一个字节不落地从显存完整读进芯片一遍！部分跳层一分钱的权重读取带宽都没省下来，反而额外付出了约占单层 7% 的状态传播开销，以及 `argsort`、压紧与散射写回的搬运成本，吞吐自然就从 99% 微降到了 98%。只有当整批 32 行全部跳过（`count == 0`）、条件节点把这 4 层的算子整体短路掉时，那 1.86 GB 的权重读取才真正被省了下来，兑现成 112% 的加速（与少读 $4/28 \approx 14.3\%$ 的层权重大致相符）。

那么，按行的部分跳过与压紧桶在什么条件下才能真正省下时间？有两个场景：
一是**超长上下文解码**。当历史序列长达几万个 token 时，每一行在注意力层需要读取的历史 KV 缓存体积会反超单层权重；此时被跳过的行因为省去了 `attn.forward` 的历史 KV 扫描，就能按跳过行数的比例实打实地省下显存带宽（原理如此，长上下文下的这一交叉点我们尚未做专门的基准实测）。
二是**算力受限（Compute-bound）的大批次场景**。当并发批次大到矩阵乘法不再受限于读权重、而是受限于 Tensor Core 算力时，把活跃行压紧到小一号的 2 的幂次桶里，就能直接按比例减少浮点运算时间。

在关心速度的同时，也别忘了检查跳层对模型输出分布的实际影响。用 `pllm.testing.check_distribution`（通过 Teacher Forcing 逐位置比对跳层模型与原模型的下一词概率分布）在同一个 7B 实验上测量：在阈值 `0.85`（23% 跳过）时，平均 KL 散度为 `0.131`（总变差距离 TV 为 `0.07`）；阈值设为 `0.80`（72% 跳过）时，KL 升至 `0.562`（TV `0.18`）；全部跳过时，KL 达到 `0.868`（TV `0.25`）。因为原生的 `Qwen2.5-7B-Instruct` 在预训练时从未见过“中间四层被当作恒等映射跳过”的情况，尽管生成的文本表面通顺，分布其实已经发生了明显偏移。系统机制能保证跳层在工程上不踩空 KV、不阻塞流水线，但算法上能不能跳、在哪几层跳，必须搭配专门经过层丢弃或早退训练的检查点（如 LayerSkip、CALM、Ouro）并用下游评测来把关。

---

## 工程边界与代价

最后，把 `@pllm.custom_forward` 在系统里的几条硬边界交代清楚。

第一是 **CUDA Graph 捕获池的显存账单**。一旦使用 `@pllm.custom_forward` 接管了层间控制流，vLLM 默认那张从头录到尾的整模型图就会被替换为 pLLM 按层切片、按活跃波次宽度录制的一组图；而条件节点里的每一个 2 的幂次桶分支，在预热捕获时都会在图显存池里分配自己的中间激活缓冲。在 7B、32 并发下，`SkipMiddle` 的图捕获池大约需要 **143 MiB** 显存，超过了默认给单任务配置的 64 MiB 上限（较早的一版实现在 0.5B 模型上，32 并发与 128 并发下测得的捕获池需求分别约为 748 MiB 与 1,392 MiB）。CPU 侧的静态规划器 `pllm.explain` 不会预先估算这部分由 CUDA 驱动在捕获期动态决定的池大小；当预算不足时，运行期会直接报错并给出所需的精确字节数，提示调大 `ExecutionOptions(cuda_graph_pool_budget_bytes=...)`。

第二是 **状态显存与组合约束**。状态传播的本质是把被跳过层的 KV 槽位填满，因此跳层**不会释放或节省任何 KV 显存**。此外，因为一层可能被跳过、也可能在循环里被执行多次，“第 $L$ 层的唯一隐藏状态”不再有明确定义，所以在同一个 Workload 内，`@pllm.custom_forward` 不能与读取固定层输出的 `pllm.hidden` / `pllm.final_hidden` 混用（需改用基于残差流的 `pllm.residual` 或 `pllm.layer_logits`），也不能与静态 KV 布局或带状态的残差记忆子层同时开启（`pllm.model.Adapter` 倒是可以同时使用：增量跟随每一次实际发生的投影调用生效）。

回过头看，处理层间动态控制流最容易掉进去的陷阱，就是顺着算法的动态性去拆改底层的存储结构——试图维护每层不同长度的块表，或者退回逐 token 的解释循环。状态传播与条件压紧桶提供的是另一种解法：**在动态控制流与静态底座的交界处，花 7% 的单层代价通过原生模块把 KV 账本补齐，让底层的块表、注意力内核与 CUDA Graph 看到的依然是一个满格、规整的世界。**
