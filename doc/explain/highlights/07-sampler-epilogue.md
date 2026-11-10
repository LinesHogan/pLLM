# 采样尾声融合

如果你想在解码阶段用一个小网络（distiller）去修正大模型的采样分布——比如在 ESamp 里，把大模型和小网络都觉得平庸的高频 token 往下压一点，把只有深层模型自己看准的 token 往上抬一点——它的数学定义只有一行：

```text
new_logit = (1 + beta) * llm_logit - beta * distiller_logit
```

为了防止能力较弱的小网络把原本根本不靠谱的生僻词抬进候选池，这个修正只作用在大模型自己做完 `top-k`、`top-p` 或 `min-p` 过滤之后留下的幸存候选上，也就是所谓的“后过滤（post-filter）”语义。

要是按最直观的方式在 PyTorch 里实现它，代码大概两分钟就能写完：先算模型的 `lm_head(hidden)`，再算一次小网络的 `lm_head(distiller_hidden)`；等采样器跑完 `top-k` / `top-p` 过滤、把淘汰位置填成 `-inf` 得到 `survivors` 之后，写一行逐元素的条件重加权：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/sampler.py
def _reweight_unfused(survivors: torch.Tensor, prediction: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # ……
    return torch.where(
        torch.isfinite(survivors), (1.0 + beta) * survivors - beta * prediction.to(survivors.dtype), survivors
    )
```

最后把改好的得分交给采样器抽样。逻辑清晰，没有半点多余的状态。

但如果你把这几行代码放进生产推理引擎的解码循环里，打开性能分析器看一眼，就会发现一个尴尬的事实：在跑完几十层 Transformer 主干之后，光是最后这几行与词表（vocabulary）打交道的收尾代码，就能吃掉几百微秒甚至上毫秒的 GPU 时间。

为什么几行最基础的矩阵乘和逐元素算术，一碰到词表就变得这么昂贵？这就得从解码阶段末端的显存带宽账算起了。

---

## 一、 词表上的显存带宽账

在一次性处理几千个输入 token 的预填充（prefill）阶段，GPU 的瓶颈在算力（FLOPs）；但在逐字生成的解码（decode）阶段，每个请求每一步只产生 1 行隐藏向量，算力早就过剩了，决定快慢的几乎只有一件事：你从高带宽显存（HBM）里读写了多少字节。

拿 `Qwen2.5-7B-Instruct`（BF16）来说，它的隐藏层维度是 `3584`，词表大小是 `152064`。模型最后一层把隐藏状态映射为词表得分的 `lm_head` 权重，是一个 `3584 × 152064` 的 BF16 矩阵，光这一个矩阵就占了 **1.09 GB**（在隐藏维为 `896` 的 `Qwen2.5-0.5B-Instruct` 上，词表大小为 `151936`，权重也有 **272 MB**）。

在 128 个并发序列的解码步里，让 `[128, 3584]` 的激活去乘 `[3584, 152064]` 的权重，浮点乘加运算本身还塞不满显卡的 Tensor Core；但为了完成这次乘法，GPU 必须把那 1.09 GB 的权重从显存里完整读进芯片一遍。在单张 NVIDIA RTX 4090（24 GiB，vLLM 0.32.0，128 并发序列）的微基准测试中，7B 模型跑一次 128 行的 `lm_head` 投影需要 **1,349 µs**，0.5B 模型需要 **359 µs**。绝大部分时间，计算单元都在等显存总线送权重过来。

这个物理事实直接引出了两条推论。第一，为小网络的预测向量单独再调一次 `lm_head` 是极不划算的：那意味着让 GPU 把 1.09 GB 的权重再从显存里读第二遍，平白多花 1.35 ms。第二，反过来想，既然权重反正要从显存读进寄存器一遍，如果能把小网络的 128 行预测向量和模型自己的 128 行拼在一起乘，权重的显存读取量一字节不增，多出来的一倍乘加运算几乎可以搭顺风车免费完成。

出了 `lm_head`，接力棒交到采样器手里，带宽的压力并没有减轻。

此时横在显存里的 `logits` 张量形状是 `[128, 152064]`。用 FP32 存放，这一张表就是 **77.9 MB**。采样器每做一步处理，都要把这近 80 MB 的数据从显存里读出来再写回去。在 7B 模型的一次稳态剖析中，温度缩放（temperature scaling）约耗时 **125 µs**，`top-k` / `top-p` 过滤约 **283 µs**，最后的 Gumbel-max 抽样约 **78 µs**（在 0.5B、vLLM 0.27.1 的参考剖析中，这三项也分别需要约 120 µs、360 µs 和 95 µs）。

在这张大表上，任何一次额外的全表扫描都是真金白银的延迟。回头看开头那行未融合的 `_reweight_unfused`：在普通的即时执行（Eager）模式下，PyTorch 会把 `isfinite`、类型转换、`1.0 + beta`、两次乘法、减法和 `where` 选择拆成 **7 次**独立的内核启动，在 `[rows, vocab]` 上来回扫七遍。在 RTX 4090、0.5B、vLLM 0.27.1 的一次测量中，朴素稠密引导每步比不引导（$\beta=0$）多花约 **401 µs**，其中词表投影约 118 µs，而那行重加权自己就吃掉了约 **187 µs**。

所以，要想让词表上的扩展逻辑不拖慢解码，设计准则其实只有一条：**绝不新增任何一遍对全词表的遍历；所有用户声明的词表操作，都必须塞进推理本来就要做的那几遍遍历里。**

在 GPU 内核设计里，趁着数据已经读进寄存器、还没写回显存之前顺手把后续算术做完的做法，叫作 **尾声融合（Epilogue Fusion）**。在一趟标准解码的末端，除了温度与过滤，有三个必经的节点天然适合挂载尾声：

1. `lm_head` 把隐藏状态投影为词表宽度的 BF16 得分；
2. 引擎把 BF16 得分拷贝并转换为一份 FP32 工作副本，交给采样器处理；
3. 采样器扫描过滤后的幸存候选，注入 Gumbel 噪声并选出最终的 token。

```text
朴素的多遍扫描流水线（每个方框都是一次 77.9 MB 的全词表读写或 1.09 GB 的权重读取）：
  hidden ───► [LM Head #1] ──► [BF16->FP32 拷贝] ──► [+偏置 #1] ──► [+偏置 #2] ──► [Top-k/p 过滤]
                                                                                         │
  distiller ► [LM Head #2 (重读权重)] ───────────────────────────► [7 遍重加权扫描] ◄────┘
                                                                          │
                                                                          ▼
                                                                    [Gumbel 抽样]

尾声融合后的采样流水线（零额外全词表遍历）：
  hidden (+ distiller) ──► [1. 共用 LM Head] ──► [2. 拷贝并按序累加偏置] ──► [Top-k/p 过滤]
                                                                                   │
  distiller ──────────────────────────────────────────────► [3. 幸存者点积 + 重加权 + Gumbel]
  (默认有界路径下，distiller 甚至直接跳过 LM Head)
```

下面我们就顺着这三个节点，看每一处是如何把多出来的遍历消掉的。

---

## 二、 节点一：在 `compute_logits` 里拼车共用 LM Head

如果算法确实需要完整的稠密词表预测分布（比如显式关闭了有界近似，或者在做过滤前的词表干预），最先要省掉的就是第二遍读取 `lm_head` 权重。

在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里我们看到，vLLM 将 Transformer 主干录进了 CUDA Graph，而负责调用 `lm_head` 的 `compute_logits` 恰好位于主干图结束后的边界上。当存在需要稠密投影的采样引导时，`pllm/backends/vllm/session.py` 会把模型的 `hidden` 与引导器产出的 `guidance.values` 在第 0 维（行维度）上拼成一张两倍长的表，一次性喂给 `lm_head`：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/session.py
projected = lm_head(torch.cat((hidden, guidance.values.to(hidden.dtype))))
assert projected is not None
logits, predicted = projected[: len(hidden)], projected[len(hidden) :]
```

出来之后按行数一刀切开，前半截归模型，后半截归引导器。1.09 GB 的权重矩阵只进出芯片一次。如果某一步里没有任何需要引导的行，或者引导强度设为 $\beta=0$，框架根本不会声明采样阶段，直接原样调用一次 `lm_head(hidden)`，保持零额外开销。

熟悉 GPU 矩阵乘细节的读者可能会问一个问题：把 128 行拼成 256 行送进 cuBLAS，原本那 128 行模型自身的输出，会不会因为矩阵形状变了而发生微小的数值漂移？

答案是：会有一点点尾数舍入上的差别，但它的变化幅度，和“连续批处理里恰好新加进来几个请求、让当前步的 batch 变大了一点”完全相同。cuBLAS 会根据输入矩阵的行数（M 维）挑选不同分块（tiling）策略的内核，分块不同会导致内部累加顺序略有差异。我们在 `Qwen2.5-0.5B` 和一个 `DeepSeek-V2-Lite` 切片上专门对比过这种拼车带来的偏差：它从未超过原生 vLLM 自己在不同 batch 大小下的天然波动范围（在两个模型上最大差值分别为 `0.0625` 和 `0.5`）。在一个每步活跃行数本来就随请求进出而动态变化的连续批处理引擎里，这种差异完全落在系统固有的数值包络之内。

不过，拼车也有一个天然的前提：必须在同一个时间点拿到所有要投影的行。如果你的算法像 DoLa（对比层解码）那样，通过 `pllm.layer_logits` 去读取模型好几层（例如第 0、2、……、14 这 8 个候选浅层，加上最后一层共 9 层）的词表分布，情况就不同了。为了保证模型自身的最终 `logits` 与原生 vLLM 逐位一致，这些中间层的隐藏状态会在 `lm_head` 处单独做一次矩阵乘，而不与最终层拼车；更重的是，每个 token 都要在 15 万词的词表上为 9 个层算出 FP32 分布与散度。在 RTX 4090 上跑 7B 模型（32 并发 = 8 条 prompt $\times$ `n=4`，生成 256 token），同时读 9 层中间词表分布的 DoLa 吞吐是原生的 **66%**（在主干计算极轻的 0.5B 模型上、vLLM 0.27.1 下则只有原生的 **25%**）。当算法本身要求实打实地多算好几遍词表时，硬件带宽是省不掉的；此时 `pllm.explain` 会在启动前把这 1.7 倍（7B）乃至 3.5 倍（0.5B）的每 token 算力成本如实列在账单上。

---

## 三、 节点二：把 `@pllm.add_logits` 折进必经的 FP32 拷贝

在进入后过滤引导之前，先看另一类常见需求：在采样器做 `top-k` / `top-p` 过滤**之前**，往词表得分上叠加一个或多个稠密的偏置张量（`@pllm.add_logits()`）。比如根据最近几个输入词从外部大表里查出来的 n-gram 先验分布（`NGramPrior`），或者某种软性的词汇约束。一个工作负载里甚至可以按顺序声明多个 `@pllm.add_logits()`。

如果每声明一个 `@pllm.add_logits()` 就单独启动一次加法内核，三个声明就是三次 80 MB 张量的全量读写。

但只要看一眼 vLLM 新版模型执行器（MRv2 runner）的采样入口，就会发现一次现成的搬运：`lm_head` 吐出的原始 `logits` 是 BF16 的，而采样器在做温度缩放、惩罚和过滤之前，第一步本来就要把这张 BF16 张量拷贝并转成一份 FP32 的工作张量。

vLLM 自己的温度、过滤和拷贝启动器并没有预留“顺手加一个稠密张量”的参数，而它自带的稀疏 `logit_bias` 接口又是为少数固定 token 设计的，装不下运行时动态算出来的整表张量。既然如此，我们只需要写一个极简的 Triton 内核 `_copy_additions`（位于 `pllm/backends/vllm/sampler_epilogue.py`），直接接管那一次必经的 BF16 到 FP32 拷贝：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/sampler_epilogue.py
loaded = tl.load(Logits + row * LS + tokens * VS, valid, other=0)
dtype = loaded.dtype
value = loaded.to(tl.float32)
for stage in tl.static_range(len(Additions)):
    selected = row
    if RowMap is not None:
        if not FULL[stage]:  # pyright: ignore[reportIndexIssue]
            selected = tl.load(RowMap + stage * ROWS + row)
    prediction = (
        tl.load(
            Additions[stage] + selected * STRIDES[stage][0] + tokens * STRIDES[stage][1],  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]
            valid & (selected >= 0),
            other=0,
        )
        .to(dtype)
        .to(tl.float32)
    )
    # Each declaration previously materialized logits in the incoming dtype.
    value = tl.where(selected >= 0, (value + prediction).to(dtype).to(tl.float32), value)
tl.store(Output + row * VOCAB + tokens, value, valid)
```

这段内核代码很短，但里面有两个关乎正确性与稳定性的细节值得留心。

第一个细节在循环体的最后一行：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/sampler_epilogue.py
value = tl.where(selected >= 0, (value + prediction).to(dtype).to(tl.float32), value)
```

注意那个 `.to(dtype).to(tl.float32)`。这里的 `dtype` 是原始 `Logits` 的数据类型（即 BF16）。既然 `value` 和 `prediction` 都已经转成了高精度的 `tl.float32`，最后写进 `Output` 的也是 `tl.float32`，为什么每加完一层偏置，还要特地把它截断回粗糙的 BF16，再转回 FP32 呢？直接在 FP32 寄存器里一路加到底，岂不是既少两条指令、精度又更高？

原因写在上面那行注释里：为了严格保持未融合前的分步物化语义。

当用户在代码里先后声明了两个 `@pllm.add_logits` 时，如果不做融合，第一个方法加在 BF16 的 `logits` 上，结果会被写回成一张 BF16 张量；第二个方法是在那张已经舍入过的 BF16 张量上继续加的。如果我们在融合进拷贝内核时，自作主张改成了全程 FP32 累加，在 `top-p` 截断的临界点附近，某两个候选词的概率和可能刚好跨过 `0.95` 的门槛——少了一次 BF16 舍入，原本该被过滤掉的词留在了池子里，随机抽样的结果就会随之改变。保留这一次寄存器内的 `.to(dtype)` 舍入，不花任何额外的显存带宽，却保证了融合前后的数学行为分毫不差（当用户通过 `rows=` 只对批次里的部分行加偏置时，主机每步异步上传一张很小的 `RowMap`，只有 `selected >= 0` 的选中行才会执行累加与舍入；对于旧版 MRv1 执行器，当其 `lm_head` 直接输出 FP32 时没有单独的转换拷贝可省，但也复用同一个保序累加内核）。在 3 个工作负载、每个负载 98,304 个生成 token（合计 **294,912 个 token**）的配对测试中，融合前后的输出不一致 token 数为 **0**。

第二个细节在内核外层的 PyTorch 包装方式上。要把一个 Triton 内核注册进 PyTorch 算子体系，常见做法是用 `torch.library.wrap_triton`。但在这里，因为 `additions` 包含可变数量的偏置张量，如果我们把它们打包成 `tuple[torch.Tensor, ...]` 传给 `wrap_triton`，就会踩中一个隐蔽的内存泄漏：`wrap_triton` 会把传入的张量元组当成编译期常量存进一张内部侧表里，从而跨步攥住这些临时张量的底层显存不放。所以这里用的是标准的 `@torch.library.custom_op`：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/sampler_epilogue.py
@torch.library.custom_op("pllm::sampler_additions", mutates_args=())
def _copy(
    logits: torch.Tensor,
    additions: list[torch.Tensor],
    row_map: torch.Tensor | None,
    full: list[bool],
    dtype: torch.dtype,
) -> torch.Tensor:
    # wrap_triton treats tensor tuples as constants and retains their storage across steps.
    # A custom op keeps variable-length additions as runtime tensor operands.
    count, vocab = logits.shape
    # ……
```

用 `list[torch.Tensor]` 作为 `custom_op` 的运行时张量参数后，每一步的临时偏置张量在算子返回的那一刻就能被正常回收。

在单步只有 3.0 ms、对固定词表开销最敏感的 `Qwen2.5-0.5B` 上（RTX 4090，vLLM 0.27.1，128 并发，生成 256 token，示例 11 `NGramPrior`），单独做偏置加法（类型转换加 BF16 加法两个内核）约耗时 **218 µs**，加上 vLLM 原生自己的 FP32 拷贝约 **111 µs**，两步合计约 **329 µs**（三次内核启动）；折进单次拷贝后，整段耗时降到了约 **180 µs**。端到端吞吐在表驻留显存（`device`）时从 33,110 提升到 **34,452 tok/s（+4.05%）**，表驻留主机内存（`host`）时从 30,786 提升到 **32,273 tok/s（+4.83%）**，带 `rows=` 子集选择时也从 27,174 提升到了 **27,913 tok/s（+2.72%）**。

---

## 四、 节点三：只为“幸存者”取权重

现在来看最重头的 `@pllm.guide_sampling`。

前面在节点一里我们提到，把小网络的预测向量拼在模型 `hidden` 后面一起乘 `lm_head`，虽然省掉了重读 1.09 GB 权重的时间，但它依然老老实实地算出了一整张 `[128, 152064]` 的稠密预测表（在 0.5B 上每步是 **34.85 GFLOP** 的矩阵乘），随后还得在过滤后的词表上跑一遍重加权内核（就算用 `torch.compile` 把那 7 遍扫描融合成单遍，也要实打实读写一遍近 80 MB 的张量；更激进的算术改写融合会引入约 `1.9e-6` 的数值误差，所以没有采用）。

有没有可能连这 15 万列的矩阵乘和全表重加权都省掉？

答案藏在一个简单的统计事实里：**经过 `top-p = 0.95` 过滤之后，152,064 个词里到底还剩几个词是活着的（即不是 `-inf`）？**

我们在 ESamp 的真实生成语料上（`Qwen2.5-0.5B`，温度 0.8，`top-p` 0.95，共 652,800 个引导行）做了一次统计：
* 平均每一行留下的有限值幸存者（survivors），只有 **4.83 个** token；
* 在全部 652,800 行里，出现过的最大真实核（nucleus）也只有 **2,785 个** token。

绝大多数时候，模型 95% 的概率质量都集中在前四五个候选词上，其余 152,059 个位置全都被过滤器涂成了 `-inf`。如果我们事先拿小网络的隐藏状态去乘整个 `lm_head` 矩阵，就等于为了给这 5 个幸存者算修正分，提前算出了 152,064 个数，然后把其中 99.99% 的结果直接扔进垃圾桶。

同理，如果你试图在过滤**之前**就把引导信号加进全词表（pre-filter guidance），虽然能省掉后处理，但小网络会直接改变进入 `top-p` 裁剪的候选集合，破坏后过滤的算法语义；而如果你预先算好全词表的稠密预测存起来、等过滤后再按索引去取（full-vocab dense-cache），全词表的矩阵乘和显存写入一分没少，也并不会自动变快。

真正干脆的做法是彻底倒过来：**在 `compute_logits` 阶段，小网络根本不去碰 `lm_head`。等上游采样器把温度缩放和 `top-k` / `top-p` 过滤全部跑完，我们直接扫描过滤后的 `Logits`，看看哪四五个词还活着，只去 `lm_head` 权重矩阵里把这四五行的权重抽出来，跟小网络的隐藏向量做几次向量点积，当场重加权、当场抽样。**

```text
稠密后过滤引导（预先投影全部 152,064 列，丢弃 99.99%）：
  distiller [1, H] ──► (× LM Head [H, 152064]) ──► [152064] ──► 掩码重加权 ──► 抽样

有界融合引导（只有幸存者才去读取 LM Head 权重行）：
  已过滤的 logits [152064]:  [ -inf, -inf,  2.1, -inf, ...,  0.8, -inf ]
                                             │                │
                         1. 将有限值幸存者压紧至 Scratch（平均仅 ~4.8 个词）
                                             ▼                ▼
                             Scratch:    [ (2.1, tok_2),  (0.8, tok_k) ]
                                                 │                │
                         2. 仅从 LM Head 权重 [152064, H] 中按行号取回这几行
                                                 ▼                ▼
                             dot(hidden, W[tok]) -> 重加权 -> gumbel_block_argmax
```

在 `pllm/backends/vllm/guided_sampling.py` 的 `_sample` 内核里，每一个待采样的请求行由一个独立的 Triton 线程块（program）负责。它先调用 `_scan_survivors` 扫一遍上游过滤好的 `Logits`，把有限值幸存者压紧写进固定容量的暂存区 `Scratch`；随后，只对 `Scratch` 里的这几个幸存者去 `Weight`（即 `lm_head` 权重）里按行寻址：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/guided_sampling.py
# Only finite selected survivors fetch embedding rows. No overflow ever projects the vocabulary.
for group in tl.range(tl.cdiv(count, DOT_ROWS)):
    slot = group * DOT_ROWS + tl.arange(0, DOT_ROWS)
    selected_keys = tl.load(Scratch + row * K + slot, slot < count, other=0)
    selected_logits, selected_tokens = _decode(selected_keys)
    valid = selected_keys != 0
    hidden_axis = tl.arange(0, DOT_WIDTH)
    hidden = tl.load(Hidden + prediction_row * HS + hidden_axis, hidden_axis < WIDTH, other=0)
    weight = tl.load(
        Weight + selected_tokens[:, None].to(tl.int64) * WS + hidden_axis[None, :],
        valid[:, None] & (hidden_axis[None, :] < WIDTH),
        other=0,
    )
    prediction = tl.sum(weight.to(tl.float32) * hidden[None, :].to(tl.float32), 1)
    # Native unquantized LM heads round their accumulation to the weight dtype.
    prediction = prediction.to(Weight.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    selected_logits = tl.where(valid, (1 + beta) * selected_logits - beta * prediction, float("-inf"))
    tl.store(Logits + row * LS + selected_tokens, selected_logits, valid)
    value, token = _draw(
        selected_logits, selected_tokens, valid, row, Mapping, Temperature, Seeds, Positions, VOCAB, FP64, V027
    )
    replace = (value > best) | ((value == best) & (token < winner))
    best, winner = tl.maximum(best, value), tl.where(replace, token, winner)
```

看这段循环里的三个关键动作：

第一，在 `Weight + selected_tokens[:, None].to(tl.int64) * WS + hidden_axis[None, :]` 处，只有被选中的 `selected_tokens` 才会发起显存读取。如果某一行只有 5 个幸存词，7B 模型下它只从 1.09 GB 的权重矩阵里读取 $5 \times 3584 \times 2 \approx 35\text{ KB}$。词表投影的复杂度从 $\mathcal{O}(\text{vocab} \times \text{hidden})$ 降到了 $\mathcal{O}(\text{survivors} \times \text{hidden})$——在 0.5B 模型上，即便按最大容量上界估算，每步也不超过 **0.94 GFLOP**。

第二，算出 FP32 点积 `prediction` 后，紧接着做了一次 `prediction.to(Weight.dtype.element_ty).to(tl.float32)`。这和上一节 `_copy_additions` 里的道理完全一致：原生未量化的 `lm_head` 在输出矩阵乘结果时，会把累加值舍入到权重类型（BF16）。补上这一下类型往返，间接点积算出来的数值就与全词表矩阵乘保持了相同的舍入行为。

第三，重加权写回 `selected_logits` 后，内核直接调用 `_draw` 完成抽样。vLLM 的采样器在生成 Gumbel 噪声时，依赖每个请求的随机种子（`Seeds`）和当前生成位置（`Positions`）；如果我们自己另写一套伪随机数公式，给定相同 `seed` 时的采样可复现性就会被打破。好在 vLLM 自己的 Gumbel 抽样核心本来就是一个 `@triton.jit` 函数 `gumbel_block_argmax`。于是 `_draw` 直接在 Triton 内核内部调用了上游的 `gumbel_block_argmax`（自动适配 vLLM 0.27 与 0.32 两个版本的参数签名，详见[《复用上游：kernel、管理器与原语》](09-reusing-upstream-kernels.md)），让请求种子、位置偏移、FP32/FP64 噪声精度以及贪心退化行为完全由上游代码执行。

这里还处理了一个只有在真实语料上才会遇到的边界情况。在其中一组 163,200 个引导行里，有 **0.12%** 的行在经过上游过滤器之后，整行 15 万个位置居然全都是有限值。把这 0.12% 的行拉出来检查就会发现，它们并不是分布平坦，而是模型极度自信：排名第一的 token 概率已经在 **0.99999994** 以上，比第二名的 logit 高出 **17 到 24**。因为第一个词自己就已经超过了 `top_p = 0.95` 的概率阈值，上游过滤器直接跳过了写 `-inf` 掩码的步骤。如果不做区分，这极度笃定的 0.12% 行反而会因为“全行都是有限值”而触发溢出路径。为此，`_scan_survivors` 在第一遍扫描时会顺手维护在线最大值与概率质量（`mass`）：一旦发现整行未掩码且 `1.0 / mass >= top_p`，内核就会取出 `peak_key`，显式将其还原为单 token 核（one-token nucleus），只对那唯一的第一名做点积。

---

## 五、 把决胜规则编进数的表示：`_keys` 与 `_decode`

在上面的 `_sample` 代码里，幸存者并不是按 `(logit, token_id)` 两个数组分开存进 `Scratch` 的，而是通过 `_keys` 打包成了一个 64 位整数，取出来时再由 `_decode` 还原。

为什么需要这一步编码？

因为当某一行的幸存者数量超过暂存区容量 $K$、需要通过 `_select_top_survivors` 选出前 $K$ 大候选，或者在多个分块之间比较最大值时，我们需要遵守一条确定的决胜规则（tie-breaking）：先按 `logit`（32 位浮点数）**从大到小（降序）**排列；如果两个词的 `logit` 恰好相等，再按 `token_id`（32 位整数）**从小到大（升序）**排列。

在 GPU 的寄存器和排序网络里同时维护一对 `(float32, int32)` 并执行带分支的复合比较是很繁琐的。既然 `float32` 和 `uint32` 各占 32 位，合起来正好是 64 位，最干净的办法就是把它们拼成一个无符号的 `uint64`，让硬件用一条原生的 64 位整数比较指令直接决出胜负。

`guided_sampling.py` 开头的 `_keys` 和 `_decode` 只用了十行代码就完成了这个双向映射：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/guided_sampling.py
@triton.jit
def _keys(logits: tl.tensor, tokens: tl.tensor, finite: tl.tensor) -> tl.tensor:
    logits = tl.where(logits == 0, 0.0, logits)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(finite, (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - tokens.to(tl.uint32)), 0)


@triton.jit
def _decode(keys: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where(ordered & 0x80000000 != 0, ordered ^ 0x80000000, ~ordered)
    return bits.to(tl.float32, bitcast=True), (0xFFFFFFFF - keys.to(tl.uint32)).to(tl.int32)
```

先看高 32 位的浮点数保序变换。IEEE 754 单精度浮点数的最高位（第 31 位）是符号位，接下来 8 位是指数，最低 23 位是尾数。对于两个**正浮点数**，它们的比特位直接当成无符号整数 `uint32` 比大小，顺序本来就是正确的。问题只出在符号位和负数上：正数最高位是 `0`，负数最高位是 `1`；而且 IEEE 754 的负数采用“符号 + 绝对值（sign-magnitude）”表示，绝对值越大（即数值越小、越负）的负数，它的指数和尾数比特反而越大。

`ordered` 那一行用一次条件位运算把整个实数轴捋顺了：
* 如果最高位是 `0`（正数或 `+0.0`），只把最高位异或翻转成 `1`（`bits ^ 0x80000000`），让所有非负数保持原有顺序住进上半区 `[0x80000000, 0xFFFFFFFF]`；
* 如果最高位是 `1`（负数），把全部 32 位**按位取反**（`~bits`）——最高位变成了 `0`（住进下半区 `[0x00000000, 0x7FFFFFFF]`），同时按位取反恰好把负数的大小顺序倒了过来，越负的数取反后无符号整数值越小；
* 第一行的 `tl.where(logits == 0, 0.0, logits)` 则事先把 `-0.0` 规范化为 `+0.0`，避免正负零映射出两个不同的编码。

再看低 32 位的决胜字段。我们要的是“高 32 位相同时，`token_id` 越小越优先（排在降序的前面）”。于是直接用 `0xFFFFFFFF - tokens.to(tl.uint32)` 把升序反转为降序；而被过滤掉的无效位置（`~finite`）则统一填为全零哨兵 `0`，在无符号比较中永远垫底：

```text
 63                              32 31                               0
┌──────────────────────────────────┬──────────────────────────────────┐
│    Order-Preserving Logit Bits   │      0xFFFFFFFF - token_id       │
│  (+: 翻转最高位, -: 32位按位取反)  │     (将升序 token ID 反转为降序)   │
└──────────────────────────────────┴──────────────────────────────────┘
       Logit 越大，高 32 位越大             同分时 token_id 越小，低 32 位越大
                    (全 0 代表被过滤的 -inf 哨兵，永远最小)
```

这样一来，“按浮点得分降序、同分按整数编号升序”的复合规则就彻底写进了数的二进制表示里。内核里的 `tl.sort(..., descending=True)` 和 `tl.maximum` 只需要当作普通 `uint64` 处理，用完后由 `_decode` 对称地还原出原始的 `float32` 得分和 `int32` 词号。

---

## 六、 容量原则：为什么 `K=4096` 比 `K=256` 更快？

为了让 `_sample` 内核能放入静态分配的显存缓冲，暂存区 `Scratch` 的每行容量 `K`（即 `ExecutionOptions.guidance_capacity`）必须在启动时固定下来。

如果某一行过滤后留下的幸存者数量 `count` 真的超过了 `K`，内核不会报错，也不会退回去做全词表矩阵乘，而是走一条**溢出路径**：调用 `_select_top_survivors` 分块扫描并调用 `tl.sort`，挑出 `logit` 最高的前 `K` 个幸存者放进 `Scratch` 接受小网络引导；至于排在 `K` 名以外的其余幸存者，则由 `_draw_tail` 作为“未引导的尾巴”一并参与 Gumbel 抽样。因为要做分块排序和额外遍历，一旦走上溢出路径，耗时就会明显增加。

那么，`guidance_capacity` 默认该设多大？

既然前面统计出平均每行只有 **4.83 个**幸存者，直觉上很容易觉得设 `K = 256` 就绰绰有余了。但当我们把 `K = 256`、`1024`、`4096` 以及 `None`（关闭有界融合，走节点一的精确稠密投影路径）放在一起实测时，得到了下面这组数字：

| 模型与引导配置 | `K = 256` | `K = 1024` | **`K = 4096`（默认）** | `None`（精确稠密路径） | 原生 vLLM 基线 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Qwen2.5-0.5B** ($\beta=0.5$) | 33,737 tok/s | 34,069 tok/s | **34,742 tok/s** | 32,624 tok/s | 约 38,900–39,300 tok/s（逐组配对） |
| **Qwen2.5-7B** ($\beta=0.5$) | 5,684 tok/s (96.4%) | 5,684 tok/s (96.4%) | **5,678 tok/s (96.3%)** | 5,533 tok/s (93.9%) | 5,894 tok/s (100%) |

> *测量设置：单张 NVIDIA RTX 4090（24 GiB），BF16，128 并发序列（8 条 prompt $\times$ `n=16`），生成 256 token，温度 0.8，`top-p` 0.95，开启 CUDA Graph 与 FlashInfer 采样器。7B 数据基于 vLLM 0.32.0 取预热后 3 轮中位数（$\beta=0$ 只训练不引导时为 5,738 tok/s，即原生的 97.4%）；0.5B 容量扫描基于 vLLM 0.27.1、7 轮中位数（在另一组 vLLM 0.27.1 的 5 轮配对测量中，0.5B 下 `K=4096` 相比稠密路径每步减少 169.28 µs，吞吐提升 4.56%）。*

看 0.5B 那一行：把容量从 `4096` 缩小到 `256`，吞吐不仅没有因为内存变小而变快，反而从 **34,742 tok/s** 掉到了 **33,737 tok/s**（慢了约 3%）。

原因在于批处理的木桶效应。在同一步的 128 个并发请求行里，哪怕有 127 行都只有四五个幸存者，只要有**一行**的幸存者数量超过了 256（别忘了真实语料里的最大核达到了 **2,785**），负责那一行的 GPU 线程块就会掉进 `_select_top_survivors` 的分块排序循环里，拖住整个批次的脚步。而在单步长达 21.7 ms 的 7B 模型上，主干每步要读约 15 GB 的模型权重，采样器的相对占比变小，256 到 4096 之间的波动已收敛在测量噪声之内，但有界路径整体依然比精确稠密路径（`None`）快了约 **2.6%**（0.5B 上则快了约 **6.5%**）。

这就引出了一条在设计固定容量数据结构时非常实用的经验——**容量原则**：

**固定容量的大小，不应当按“平均情况”去选，而应当选到足够大，让现实中的常见情况永远看不见那条昂贵的溢出路径。**

把默认容量定在 **4096**（大于实测最大真实核 2,785），意味着在常规生成中，几乎所有行都直接走最轻快的线性压紧路径，那条复杂的排序溢出代码只作为兜底安全网存在。而为 `K = 4096` 付出的空间代价是多少呢？每行 4096 个 `uint64`（8 字节），128 行加起来一共只占 **4 MiB** 的暂存显存。用 4 MiB 的显存彻底避开运行时的分块排序，是一笔再划算不过的账。

在数值一致性上，既然 ESamp 四组实测语料里的最大核都没超过 4096，有界路径与精确稠密路径在 **163,200 个引导行**上的共享 Gumbel 抽样结果做到了 **100% 逐 token 一致**；两边输出概率分布的总变差距离（TV distance）p99.9 仅在 $6 \times 10^{-14}$ 到 $1.5 \times 10^{-10}$ 量级，完全属于浮点舍入噪声。

---

## 七、 适用边界与取舍

把工作塞进推理原有的遍历里，让我们用很小的代价拿到了接近原生的解码吞吐，但同时也划定了几条明确的边界：

首先，**有界融合路径有它的准入条件**。因为 `_sample` 内核需要拿着词号直接去 `lm_head.weight` 里按行读取并做点积，它要求执行器为 MRv2、单卡运行（`tensor_parallel_size=1`），且 `lm_head` 是未量化、未加额外缩放或变换的标准权重。如果开启了多卡张量并行（TP > 1，此时词表维被切分在不同 GPU 上），或者模型使用了量化权重，系统会自动回退到节点一的精确稠密路径（`torch.cat` 拼车投影 + `torch.compile` 融合的 1 遍重加权内核）。同理，当温度极高或分布极平坦导致幸存者确实超过 $K$ 时，只引导前 $K$ 个候选、让尾部保持未引导是一项工程近似；如果你的实验要求在任何极端分布下都严格执行全词表后过滤，只需传入 `ExecutionOptions(guidance_capacity=None)` 即可切回精确稠密路径。

其次，**静态成本估算与实际运行量的差异**。在阅读 `pllm.explain` 的静态报告时，由于规划器在启动前并不知道运行时每行会留下几个幸存者，有界路径是按容量上界 $K$（默认 4096）来汇报最坏字节数与算力的，账面数字有时看起来比固定算子的静态值还高。那是静态缓冲的预算上限，而 GPU 实际搬运的字节数只取决于每步真正活下来的那几个候选词。

最后，也许你会好奇：为什么 pLLM 允许在一个工作负载里叠加任意多个 `@pllm.add_logits`，却在编译期硬性规定最多只能声明**一个** `@pllm.guide_sampling`？

这不是因为内核里写不下第二个循环，而是因为两者的代数性质不同。过滤前的偏置加法满足结合律，多个 `@pllm.add_logits` 按声明顺序逐个加上去（每步保持输入 dtype 舍入），语义是唯一确定的。但后过滤引导是在截断后的概率核上做非线性对比重加权——如果同时存在两个引导器，它们究竟应该在隐藏状态空间相加、在全词表对数空间叠加，还是先后截断重加权两次？不同的组合顺序对应着完全不同的算法。与其在底层框架里武断地替用户选一种组合方式，不如把这道边界留给算法作者：如果有多个引导源，先在自己的计算方法里把它们合成一个明确的预测向量，再交给那唯一的一次尾声抽样。
