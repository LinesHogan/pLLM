# Free FLOPs and Where to Find Them：利用 Memory-Bound 解码尾声顺手做完采样引导

> 2026-11-11 · pLLM 提交 `1f9e03d6` · 测量环境：除文中另注外为单张 RTX 4090（24 GiB）、vLLM 0.32.0、Qwen2.5-7B-Instruct，8 条 prompt × 16 个样本、生成 256 token。
> 这是一篇有日期的文章，不随代码更新；当前的用法以[用户指南](../../README.md#用户指南)为准，当前的实测数字以[性能与显存](../performance.md)为准。

如果你想在自回归解码阶段用一个小网络（比如在线蒸馏器 distiller）去修正大模型的采样分布——把大模型和小网络都觉得好的平庸词往下压一点，把只有大模型深层才看准的词往上抬一点——它的数学公式只有一行：

```text
new_logit = (1 + beta) * llm_logit - beta * distiller_logit
```

为了防止小网络把原本根本不靠谱的生僻词抬进候选池，这个修正只作用在大模型自己做完 `top-k`、`top-p` 或 `min-p` 过滤之后留下的幸存候选词上（即“后过滤”语义，post-filter）。

在单请求的 PyTorch 脚本里，把这个公式写出来大概只需要两分钟：先调一次 `model.lm_head(hidden)` 算出大模型的词表得分，再调一次 `model.lm_head(distiller_hidden)` 算出小网络的词表得分；等采样器的过滤器把淘汰词涂成 `-inf` 变成 `survivors` 之后，写一行张量算术：

```python
def _reweight_unfused(survivors: torch.Tensor, prediction: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # ……
    return torch.where(
        torch.isfinite(survivors), (1.0 + beta) * survivors - beta * prediction.to(survivors.dtype), survivors
    )
```

代码清晰、直观，没有任何花哨的地方。但如果你把这几行代码放进生产级的 vLLM 解码循环里，打开性能分析器（profiler）看一眼，会发现一个让人意外的事实：在跑完了整座二十八层的 Transformer 之后，光是最后这几行和词表打交道的收尾代码，就能生生吃掉几百微秒甚至上毫秒的 GPU 时间。

为什么几行最基础的矩阵乘和逐元素算术，一碰到词表（vocabulary）就变得这么昂贵？反过来，我们又该如何不花一分钱额外带宽，白捡到这些计算？

答案藏在 GPU 解码尾声的物理特性里：**在这里，遍历显存是天价的，但算力（FLOPs）几乎是免费的。** 只要你知道这些免费算力藏在哪里，并顺着数据本来就要走的那几趟班车搭个便车，就能把整套采样改写藏得无影无踪。

---

## 先算一笔显存带宽账：为什么这里的算力不要钱？

很多刚接触推理系统的工程师容易有一个直觉：GPU 跑得慢，是因为浮点乘加运算（FLOPs）太多了。

在一次性处理几千个输入 token 的预填充（prefill）阶段，确实如此。但在一个字接一个字往外蹦的解码（decode）阶段，算力早就严重过剩了，真正的瓶颈只有一个：**显存带宽（Memory Bandwidth）**。

不妨拿计算器按两下。以 `Qwen/Qwen2.5-7B-Instruct`（BF16）为例，它的隐藏层维度是 `3584`，词表大小是 `152064`。这意味着模型最后一层那个把隐藏状态投影成词表得分的 `lm_head` 权重，是一个形状为 `3584 × 152064` 的双字节大矩阵——光这一个矩阵，就重达 **1.09 GB**（在 `Qwen2.5-0.5B-Instruct` 上则是 `896 × 151936`，约 **272 MB**）。

在单张 NVIDIA RTX 4090（24 GiB GDDR6X，vLLM 0.32.0，开启 CUDA Graph 与 FlashInfer 采样器）上，当我们以 128 条并发序列（8 条 prompt $\times$ 每条采样 $n=16$，生成 256 个 token，温度 0.8，`top-p` 0.95）运行解码时，7B 模型跑完一整步需要 **21.7 ms**（原生吞吐 **5,894 tok/s**），0.5B 模型跑完一整步更是只需要 **3.0 ms**（原生吞吐 **42,355 tok/s**）。

而在每一步的最后关头，128 条并发序列只产生 128 行隐藏向量。让 `[128, 3584]` 去乘 `[3584, 152064]`，这点乘加运算还塞不满 RTX 4090 Tensor Core 的牙缝；但为了算出这 128 行的结果，GPU 必须老老实实地把那 **1.09 GB** 的权重从 GDDR6X 显存里完整读进芯片一遍。在微基准测试里，7B 模型跑一次 128 行的 `lm_head` 投影要花 **1,349 µs**（0.5B 模型读 272 MB 权重也要 **359 µs**）。在这 1.35 毫秒里，计算单元绝大部分时间都在干等显存控制器把权重字节送过来。

出了 `lm_head`，接力棒交到采样器手里，带宽的紧箍咒依然没有解除。此时你面前的 `logits` 张量形状是 `[128, 152064]`，用 FP32 存放，一张表就是 **77.9 MB**。采样器要在这张近 80 MB 的大表上来回扫描：在 7B 的一次剖析中，除以温度（temperature）大约要 **125 µs**，`top-k` / `top-p` 过滤大约要 **283 µs**，最后的 Gumbel-max 随机抽样大约要 **78 µs**。

回头看开头那行看似无害的 `_reweight_unfused`：

```python
return torch.where(
    torch.isfinite(survivors), (1.0 + beta) * survivors - beta * prediction.to(survivors.dtype), survivors
)
```

在普通的即时执行（Eager）模式下，PyTorch 会怎么执行这一行代码？`isfinite` 扫一遍 80 MB、`.to(dtype)` 扫一遍、算 `1.0 + beta`、两次乘法、相减、`where` 选择……整整 **七次** 独立的内核启动，在 `[rows, vocab]` 的全表上来回读写！在 0.5B 模型（vLLM 0.27.1）上的一次测量里，朴素的全词表引导每步比不引导要多花约 **401 µs**（其中单独投影约 118 µs，重加权约 187 µs），比 0.5B 模型自己的 `lm_head` 投影（359 µs）还要贵。

看清了这个物理事实，一条系统设计准则立刻浮出水面：在被显存带宽锁死的解码尾声，**每多扫一遍词表都是不可饶恕的浪费；但只要数据已经被读进了寄存器，顺手在上面多做几次算术（Epilogue Fusion，尾声融合），算力就是白送的。**

在一次标准的推理步末尾，有三遍遍历是引擎无论如何都省不掉的：
1. `lm_head` 从显存读取 1.09 GB 权重，把隐藏状态投影成 BF16 的词表得分；
2. 采样器入口把 BF16 的词表得分拷贝并转换为一份 FP32 工作张量；
3. 采样器出口扫描过滤后的幸存候选词，注入 Gumbel 噪声并抽出最终的 token。

```text
朴素的多遍扫描管线 (每一个方框都是一次 1.09 GB 权重读取或 77.9 MB 全词表读写):
  hidden ────► [LM Head #1] ──► [BF16->FP32 拷贝] ──► [+偏置 #1] ──► [+偏置 #2] ──► [Top-k/p 过滤]
                                                                                          │
  distiller ─► [LM Head #2 (再读1.09GB!)] ─────────────────────► [7遍逐元素重加权] ◄──────┘
                                                                        │
                                                                        ▼
                                                                  [Gumbel 抽样]

尾声融合管线 (零新增全词表遍历，在三处必经之路上搭便车):
  hidden (+ distiller) ──► [栖息地1: 拼车共用 LM Head] ──► [栖息地2: 拷贝顺手加偏置] ──► [Top-k/p 过滤]
                                                                                               │
  distiller ──────────────────────────────────────────────► [栖息地3: 仅抽4.8个幸存行点积 + 重加权 + 抽样]
  (默认有界路径：连 LM Head 都不进！)
```

这三遍必经遍历，就是免费算力的三处栖息地。我们挨个去拜访它们。

---

## 第一处栖息地：在 `lm_head` 里拼车共读 1.09 GB 权重

先看第一种情况：如果你的算法确实需要完整的全词表预测分布（比如用户在配置里显式要求不做候选截断，走精确的全词表引导），怎么省掉第二遍读那 1.09 GB 的 `lm_head` 权重？

既然 `lm_head(hidden)` 慢在读那 1.09 GB 的权重，而不慢在算 128 行的乘法，那么当权重好不容易从显存搬进芯片寄存器时，让小网络预测出来的 128 行向量跟模型自己的 128 行向量“拼一辆车”一起乘，权重依然只读一遍，多出来的 128 行乘加运算几乎是免费的。

在 pLLM 的 vLLM 会话后端（`pllm/backends/vllm/session.py`）里，这个拼车动作只有两行代码：

```python
projected = lm_head(torch.cat((hidden, guidance.values.to(hidden.dtype))))
assert projected is not None
logits, predicted = projected[: len(hidden)], projected[len(hidden) :]
```

把模型的 `hidden` 和引导器的 `guidance.values` 在第 0 维（行维度）上用 `torch.cat` 拼成一张两倍长的表，喂给同一个 `lm_head`，出来之后再一刀切开，前半截归模型，后半截归引导器。

如果你熟悉 GPU 矩阵乘法的脾气，这里可能会问一个很内行的问题：**把 128 行拼成 256 行喂给 cuBLAS，模型原本那 128 行的输出会不会因为拼车而发生微小的数值变化？**

会有一点点，但它的变化幅度完全落在连续批处理引擎固有的数值包络之内。cuBLAS 会根据输入矩阵的行数（M 维）启发式地挑选不同的分块（tiling）内核，不同的归约顺序会带来尾数最低位的舍入差异。我们在 `Qwen2.5-0.5B` 和一个 `DeepSeek-V2-Lite` 切片上专门测量过这种拼车带来的偏差：它从未超过原生 vLLM 自己在不同 batch 大小下的天然敏感度（在两个模型上分别是 `0.0625` 和 `0.5`）。在一个每步 batch 行数本来就随着请求进出而动态起伏的服务引擎里，多拼进来的这几行，效果就跟“这一步恰好新到了几个用户请求”一模一样。

当然，拼车有一个物理前提：你得在发车前赶到车站，而且不能拼得太离谱。如果你的算法像 DoLa（对比层解码）那样，通过 `pllm.layer_logits` 去同时读取模型好几层（例如第 0、2、……、14 这 8 个候选浅层，加上最后一层共 9 层）的词表分布，为了保证主干模型自己的 logits 与原生 vLLM 逐位一致，这些中间层会在 `lm_head` 处单独做矩阵乘，并且每个 token 都要在 15 万词的词表上为 9 个层计算 FP32 logits 与散度。在 7B 模型（32 并发 = 8 $\times$ `n=4`）上，9 层 DoLa 的吞吐会降到原生的 **66%**（在主干极轻的 0.5B 模型上、vLLM 0.27.1 下则是 **25%**）——当算法本身要求实打实地做十倍的词表投影与散度归约时，就没有免费午餐可吃了。

---

## 第二处栖息地：把偏置累加折进必经的 FP32 拷贝

走出了 `lm_head`，我们拿到了形状为 `[rows, vocab]` 的 BF16 `logits`。在进入复杂的过滤与引导之前，先看一个更普遍的需求：在采样过滤**之前**往词表得分上叠加稠密的偏置向量（在 pLLM 中声明为 `@pllm.add_logits()`，比如叠加从大表里查出来的 n-gram 先验分布，甚至在一个工作负载里连续叠加好几个偏置）。

朴素的做法是：每声明一个偏置，就发射一个逐元素加法内核，把近 80 MB 的词表张量从显存读出来、加上去、再写回去。三个偏置就是三趟显存往返。

但如果我们掀开 vLLM 新版模型执行器（MRv2 runner）的代码看一眼，就会发现第二趟必经班车：**vLLM 在对 logits 做温度缩放、惩罚和 `top-k` / `top-p` 过滤之前，第一步本来就要把 `lm_head` 输出的 BF16 张量拷贝并转换成一份 FP32 张量。**

遗憾的是，vLLM 自己的温度、过滤和拷贝算子接口都没有留“顺手加几个稠密张量”的参数槽位，而它提供的稀疏 `logit_bias` 又是给个别固定 token 准备的。既然上游没有这个旋钮，我们在 `pllm/backends/vllm/sampler_epilogue.py` 里写了一个极简的 Triton 内核 `_copy_additions`，精准替换掉那一次 BF16 到 FP32 的拷贝：

```python
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

这段只有二十行的代码把“读 BF16 $\to$ 累加任意多个偏置 $\to$ 写出 FP32”压缩在了同一组寄存器里。但在写这段代码时，有两个细节决定了它是“可靠的生产优化”还是“埋雷的聪明反被聪明误”。

### 1. 为什么要故意退回 BF16 舍入一次？

请盯住循环里的最后一行：

```python
value = tl.where(selected >= 0, (value + prediction).to(dtype).to(tl.float32), value)
```

这里的 `dtype` 是输入 `Logits` 的类型（BF16）。你可能会觉得奇怪：既然 `value` 和 `prediction` 都已经在寄存器里转成了高精度的 `tl.float32`，最后写进 `Output` 的也是 `tl.float32`，为什么每加完一层偏置，还要特地调一次 `.to(dtype).to(tl.float32)` 把它截断回粗糙的 BF16、再转回 FP32 呢？直接在 FP32 里一路加到底，既省了两条指令、精度又更高，岂不是更好？

在系统优化里，这是最容易踩的坑：**自作主张替用户“提高精度”，往往意味着破坏可复现性。**

如果没有这个融合内核，当用户按顺序声明两个 `@pllm.add_logits` 时，系统的原始语义是：第一个方法加在 BF16 的 `logits` 上，物化成一张 BF16 张量；第二个方法再加在这张新的 BF16 张量上。如果我们在融合进拷贝算子时偷偷改成了全程 FP32 累加，在 `top-p = 0.95` 的截断边界上，某两个候选词的累积概率可能刚好卡在 `0.9500001`——少了中间那一次 BF16 舍入，原本该被踢出局的词留了下来，下游 Gumbel 随机抽样的随机数消耗随之错位，后面生成的整段话就会彻底分道扬镳。

保留那一次 `.to(dtype)` 寄存器内舍入，不花一分钱显存带宽，却严格复刻了分步物化的浮点语义。我们在 3 个不同的工作负载、每个负载 98,304 个生成 token（合计 **294,912 个 token**）上对融合前后做了逐 token 比对：不一致的 token 数量是严格的 **0**。

### 2. `wrap_triton` 的隐秘显存泄漏

把这个 Triton 内核注册进 PyTorch 时，还藏着一个非常隐蔽的泄漏。

最顺手的包装是 PyTorch 提供的 `torch.library.wrap_triton`。因为一个工作负载里可能有不定数量的偏置阶段，自然会想把这些偏置张量打包成一个 Python 元组 `tuple(additions)` 传进去——而这恰恰会让显存水位跨步上涨。

病根写在 `sampler_epilogue.py` 改用 `@torch.library.custom_op` 的注释里：

```python
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

原来，在 `wrap_triton` 的底层分发胶水里，单独传入的 `torch.Tensor` 会被当作运行时数据参数（只取指针）；但如果你传入一个 `tuple[torch.Tensor, ...]`，它会把这个元组当成**编译期常量配置**缓存进一张全局侧表里！结果就是每跑一步 decode，那一步临时算出来的偏置张量都会被侧表死死攥住引用，跨步驻留在显存里无法释放。换成标准的 `@torch.library.custom_op` 并使用 `list[torch.Tensor]` 作为算子级张量参数，这些偏置就只是运行时张量操作数。

在一步只有约 3 ms、对固定词表开销最敏感的 0.5B 模型上（`NGramPrior` 查表先验负载，RTX 4090，vLLM 0.27.1，128 并发），将 `add_logits` 折进这次必经拷贝，把三次内核发射缩减为一次：原本独立的加法（约 `218 µs`）加上原生自己的 FP32 拷贝（约 `111 µs`）合计约 `329 µs`，融合后整段只需约 **`179 µs`**；端到端吞吐在显存查表下从 33,110 提升至 **34,452 tok/s（+4.05%）**，主机内存查表下从 30,786 提升至 **32,273 tok/s（+4.83%）**。

---

## 第三处栖息地：只为 4.83 个“幸存者”取权重行

现在，我们来到了全篇最核心的一处免费算力栖息地：过滤后的采样引导（`@pllm.guide_sampling`）。

前面在第一处栖息地里我们说过，把小网络的隐藏状态拼在模型 hidden 后面一起乘 `lm_head`，虽然省掉了重读 1.09 GB 权重的时间，但依然实打实地算出了一张 `[128, 152064]` 的稠密预测表（在 0.5B 上每步是 **34.85 GFLOP** 的矩阵乘），随后还得在过滤后的词表上跑一遍重加权内核。

但如果我们停下来，盯着经过 `top-p = 0.95` 过滤之后的 `logits` 张量问一个简单的问题：**在 152,064 个候选词里，经过 `top-p = 0.95` 截断之后，到底还有几个词是活着的（即不是 `-inf`）？**

我们在 ESamp 的真实生成语料上（0.5B 模型，vLLM 0.27.1，$\beta=0.5$，温度 0.8，`top-p` 0.95，共 **163,200 个引导行**）数了一下每一行的有限值个数。统计结果如下：
* 平均每一行留下的幸存者（survivors），只有约 **4.83 个** token；
* 在这 163,200 行里，出现过的最大真实概率核（nucleus）也只有 **2,785 个** token（再加三组语料、共 652,800 行，也没有一行的真实核超过 4096）。

不妨体会一下这两个数字的分量。在绝大多数生成步里，大模型对接下来要说的词非常笃定，95% 的概率质量全部集中在前四五个词上，剩下那 **152,059 个词** 已经被采样器的过滤器涂成了 `-inf`。

可全词表投影在干什么？为了给这平均 4.83 个幸存词算引导修正，我们兴师动众地让小网络跟全部 152,064 列权重做了一次完整的矩阵乘法，算出 152,064 个浮点数，然后把其中 99.997% 的结果直接扔进垃圾桶！

既然只有约 5 个词活到了最后，为什么要在过滤之前投影整个词表？**等上游采样器把 `top-k` / `top-p` 过滤做完，趁着扫描过滤结果的那一趟，看看哪几个词还活着，直接去 `lm_head` 权重矩阵里把那 5 行挑出来，跟小网络的隐藏向量做 5 次点积，当场改分、当场抽样，不就完了吗？**

```text
全词表后过滤引导 (投影全部 152,064 列，丢弃 99.997% 的计算结果):
  distiller [1, H] ──► (x LM Head [H, 152064]) ──► [152064] ──► 掩码重加权 ──► 抽样

有界融合引导 (Bounded Fused Guidance — 只有幸存者才去碰 LM Head 权重):
  已过滤的 logits [152064]:  [ -inf, -inf,  2.1, -inf, ...,  0.8, -inf ]
                                             │                │
                         1. 扫描压紧幸存者进 Scratch (平均仅 ~4.83 个词!)
                                             ▼                ▼
                             Scratch:    [ (2.1, tok_2),  (0.8, tok_k) ]
                                                 │                │
                         2. 仅按幸存 token_id 去 LM Head [152064, H] 抽对应行做点积
                                                 ▼                ▼
                             dot(hidden, W[tok]) -> 重加权 -> 原地调 gumbel_block_argmax!
```

这就是 `pllm/backends/vllm/guided_sampling.py` 中有界融合内核 `_sample` 的全部思想。

在这个 Triton 内核里，每一个待采样的请求行分配一个线程块（program）。它先调用 `_scan_survivors` 扫一遍上游刚刚过滤好的 `Logits`，把有限值压紧写进一块固定容量为 `K` 的暂存区 `Scratch`；紧接着，只为 `Scratch` 里的这几个幸存者去 `lm_head`（即代码中的 `Weight`）里按行号捞权重：

```python
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
```

看看这段代码省下了多少功夫：
1. **从 `O(vocab × hidden)` 降维到 `O(survivors × hidden)`**：看 `Weight + selected_tokens[:, None].to(tl.int64) * WS` 这一行。如果某一行只有 5 个幸存词，内核就只从 1.09 GB 的权重表里读取这 5 行（在 7B 上一共才 $5 \times 3584 \times 2 \approx 35\text{ KB}$）。在 0.5B 模型上，哪怕按最大容量上界向 `pllm.explain` 报账，每步的计算量也不超过 **0.94 GFLOP**，而实际运行时只算平均 4.83 行。
2. **复刻原生 `lm_head` 的舍入行为**：算出 FP32 点积 `prediction` 后，下一行紧跟着 `prediction = prediction.to(Weight.dtype.element_ty).to(tl.float32)`。因为未量化的原生 `lm_head` 矩阵乘在输出时就是落回权重类型（BF16）的，加上这一步转换，保证了按需点积算出来的数值与全词表 GEMM 算出来的值分毫不差。
3. **在自己的内核里直接调用上游的抽样心脏**：重加权算出 `selected_logits` 之后，紧接着的下一行就是调用 `_draw` 完成最终抽样。我们在[《稀疏注意力在 vLLM 里省不下显存》](06-sparse-attention-kv.md)里会谈到复用上游原语的原则——如果在这里自己手写一套伪随机数生成器和 Gumbel 噪声公式，不仅容易出错，更会破坏请求随机种子的可复现性。翻看 vLLM 的源码就会发现，它的 Gumbel-max 抽样核心本来就是一个 `@triton.jit` 函数 `gumbel_block_argmax`。因此，`_draw` 直接在 Triton 内核内部调用了上游的 `gumbel_block_argmax`（自动适配 vLLM 0.27 与 0.32 的签名），让随机种子、位置偏移与噪声精度 100% 沿用上游实现。

在扫描幸存者时，我们还处理了一个只有跑过真实大数据才会撞见的角落情况。在其中一组 163,200 个引导行里，有 **0.12%** 的行在经过上游过滤器之后，竟然**整行 152,064 个词全都是有限值**！把这 0.12% 的行拉出来检查，发现并不是过滤器坏了，而是模型在这些位置极度自信：第一名（top-1）的概率超过了 **0.99999994**，比第二名的 logit 高出 **17 到 24**。因为仅凭第一名一个词就已经满足了 `top_p = 0.95`，上游过滤器干脆跳过了写 `-inf` 掩码的步骤，把整行原样放了过来。

为此，`_scan_survivors` 在扫描时会顺手维护在线最大值与指数和（`mass`）：一旦发现某一行全满（没有任何 `-inf`）、且 `1.0 / mass >= top_p`，内核就知道这是一个被上游走捷径放过来的“单词核（one-token nucleus）”，直接提取唯一的峰值词 `peak_key` 送去点积，绝不傻乎乎地退回全词表处理。

---

## 把决胜规则写进比特里：5 行 IEEE 754 位翻转

在刚才的 `_sample` 代码里，你可能注意到了成对出现的 `_keys` 与 `_decode(selected_keys)`。把幸存者存进 `Scratch` 时，我们并没有分别建立 `logits` 和 `token_ids` 两个数组，而是把每一个候选词打包成了一个单独的 64 位无符号整数 `uint64`。

为什么要这么做？

因为在挑出前 $K$ 个幸存者或者比较候选峰值时，我们需要遵守一条严格的复合决胜规则（tie-breaking）：
1. 首先按 `logit`（32 位浮点数）**从大到小（降序）**排列；
2. 如果两个词的 `logit` 恰好相等，按它们的 `token_id`（32 位整数）**从小到大（升序）**排列。

在 GPU 的寄存器和片上内存里，同时维护两个数组、每次比较都写复合条件分支，既占寄存器又拖慢流水线。既然一个 `float32` 是 32 位，一个 `uint32` 也是 32 位，我们完全可以把它们拼成一个 **64 位无符号整数（`uint64`）**，让硬件用一条原生的整数比较指令 `a > b` 同时完成浮点降序与整数升序决胜。

来看看 `guided_sampling.py` 顶部那 5 行 `_keys` 与对称的 `_decode`：

```python
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

这 5 行 `_keys` 代码里用了两个经典的位操作技巧（bit-twiddling hacks）。

第一个技巧在 `ordered` 那一行：**怎么把 IEEE 754 浮点数变成可以直接按无符号整数比大小的格式？**

IEEE 754 标准把符号位放在最高位（第 31 位），8 位指数位紧随其后（第 30..23 位），23 位尾数位放在最低处。这种布局有一个非常漂亮的性质：对于任意两个**正浮点数**，直接把它们的内存比特当成无符号整数 `uint32` 来比大小，顺序本来就是完全正确的！唯一的障碍在于符号位和负数的“符号 + 绝对值（sign-magnitude）”表示法：正数最高位是 `0`，负数最高位是 `1`；而且负数的绝对值越大（即数值越小、越负），它的指数和尾数比特看起来反而越大。

怎么用一条位运算把整个实数轴捋顺？
* **如果是正数**（最高位为 `0`）：我们只把最高位异或翻转成 `1`（`bits ^ 0x80000000`），让所有正数保持原有顺序住进上半区 `[0x80000000, 0xFFFFFFFF]`；
* **如果是负数**（最高位为 `1`）：我们把它的全部 32 个比特**按位取反**（`~bits`）——最高位变成了 `0`（住进下半区 `[0x00000000, 0x7FFFFFFF]`），同时按位取反正好把“绝对值越大比特越大”倒了个个儿，越负的数取反后无符号整数值就越小！（第一行的 `tl.where(logits == 0, 0.0, logits)` 则是先把 `-0.0` 规范化为 `+0.0`，防止正负零映射出两个不同的整数。）

第二个技巧在最后一行。我们把保序的 `ordered` 左移 32 位放进 `uint64` 的高半边；至于低 32 位，怎么表达“同分时 `token_id` 越小越优先”？直接拿 `0xFFFFFFFF` 减去 `token_id`，把升序反转成降序：

```text
 63                              32 31                               0
┌──────────────────────────────────┬──────────────────────────────────┐
│    Order-Preserving Logit Bits   │      0xFFFFFFFF - token_id       │
│  (+: 翻转最高位, -: 32位按位取反)  │     (将升序决胜反转为降序整数)     │
└──────────────────────────────────┴──────────────────────────────────┘
       Logit 越大，高 32 位越大          同分时 token_id 越小，低 32 位越大
                   (全 0 为被过滤的 -inf 哨兵，永远垫底)
```

把排序规则编码进数字本身的比特表示之后，内核里无论是调用 `tl.sort(..., descending=True)` 还是用 `tl.maximum` 归约，都只需要做最原始的 `uint64` 整数比较。

---

## 容量原则：为什么 `K=4096` 比 `K=256` 更快？

为了让这个有界融合内核能够静态分配显存并进入 CUDA Graph（见[《What Color is Your Tensor?》](02-tensor-colors.md)），存放幸存者的暂存区 `Scratch` 必须在启动时固定每行的最大容量——即配置项 `ExecutionOptions.guidance_capacity`（代码中的常数 `K`）。

如果某一行特别发散，过滤后留下的幸存词数量 `count` 超过了 `K`（发生溢出），内核依然不会去投影整个词表，而是自动切入一条**溢出路径**：调用 `_select_top_survivors`，分块读取词表并用 `tl.sort` 归并挑出 `logit` 最高的前 `K` 个幸存词接受引导，其余排在 `K` 名之外的幸存词则交给 `_draw_tail` 作为未引导的尾巴参与 Gumbel 抽样。因为要做分块排序，溢出路径比直接压紧的主路径慢得多。

那么，这个固定容量 `K` 到底该选多大？

前面我们刚统计过，ESamp 里平均每行只有 **4.83 个**幸存词。按照节省内存的直觉，平均不到 5 个词，选 `K = 256` 似乎就已经富余五十倍了。但当我们在 RTX 4090 上把 `K = 256`、`1024`、`4096` 和 `None`（走第一处栖息地的全词表精确 dense 路径，重加权由 `torch.compile` 融合成单个算子）放在一起实测时，结果却给直觉上了一课：

| `guidance_capacity` (`K`) | 0.5B 吞吐 (tok/s, $\beta=0.5$) | 7B 吞吐 (tok/s, $\beta=0.5$) | 7B 相对原生 (5,894 tok/s) |
| :--- | ---: | ---: | ---: |
| `256` | 33,737 | 5,684 | 96.4% |
| `1024` | 34,069 | 5,684 | 96.4% |
| **`4096`（默认）** | **34,742** | **5,678** | **96.3%** |
| `None`（精确全词表 dense） | 32,624 | 5,533 | 93.9% |

*(注：测试环境为单卡 RTX 4090，128 并发序列，生成 256 token，ESamp $\beta=0.5$；7B 基于 vLLM 0.32.0，取 3 轮中位数；0.5B 容量扫描基于 vLLM 0.27.1，取 7 轮中位数，同期原生约 38,900–39,300 tok/s，所以表里没有 0.5B 的相对原生一列。当 $\beta=0$ 只训练不引导时，pLLM 不声明采样阶段，7B 吞吐为 5,738 tok/s，达原生的 97.4%。)*

看 0.5B 那一列：**把容量从 4096 缩减到 256，吞吐不仅没涨，反而掉了约 3%（从 34,742 跌到 33,737 tok/s）！**

原因就在于批处理的长尾效应。在同一步的 128 行请求里，哪怕有 127 行都只有四五个幸存词，只要有**一行**的幸存词数量超过了 256（别忘了语料里的最大真实核是 **2,785**），负责那一行的 GPU 线程块就会掉进需要反复调用 `tl.sort` 的溢出路径，整批的完成时间就会被这一个慢线程块拖住。

由此我们可以提炼出一条在设计固定容量数据结构时非常实用的**容量原则（The Capacity Principle）**：

> **固定容量的使命不是去拟合“平均情况”，而是要选到足够大，让现实中的常见情况永远看不见那条复杂的溢出路径。**

选 `K = 4096`（大于实测最大真实核 `2,785`），意味着在常规生成中，100% 的行都直接走最轻快的无排序压紧路径。而在同一组 163,200 个引导行的配对检验中，因为 4096 完整覆盖了所有真实核，有界融合路径与全词表精确 dense 路径在共享 Gumbel 抽样下抽出的 token 达到了 **100% 逐个一致**，策略分布的总变差距离（TV distance）p99.9 仅为浮点舍入噪声量级（约 $6 \times 10^{-14}$ 至 $1.5 \times 10^{-10}$）。至于为 `K = 4096` 付出的显存代价？$128\text{ 行} \times 4096 \times 8\text{ Bytes} = \mathbf{4\text{ MiB}}$——在 24 GiB 的显卡上连一朵浪花都算不上。

最后，这套尾声融合机制也有三条明确的边界需要交代清楚：
1. **有界路径的准入条件**：因为 `_sample` 需要拿着 `selected_tokens` 直接去 `lm_head.weight` 里按行寻址做点积，它要求 `tensor_parallel_size = 1`，且 `lm_head` 是未量化、未加额外缩放的标准权重。当开启多卡张量并行（TP > 1）切分了词表维，或使用量化 `lm_head` 时，系统会自动退回第一处栖息地的拼车 dense 路径。同样，如果你在极端温度下需要数学上绝对无溢出的全词表后过滤语义，传 `ExecutionOptions(guidance_capacity=None)` 即可显式切回 dense 路径。
2. **静态账单的容量上界**：当你用 `pllm.explain` 查看静态分析报告时，有界路径报告的字节数和算力是按最大容量 $K=4096$ 估算的悲观安全帽，不能直接当作运行时的平均带宽消耗。
3. **为什么允许多个 `@pllm.add_logits`，却只允许一个 `@pllm.guide_sampling`**：过滤前的偏置加法满足结合律，叠多少层都是良定义的；但后过滤引导是在截断分布上做非线性对比重加权，两个引导器之间该怎么组合属于算法定义问题，框架不在背后替用户瞎猜。

---

回顾解码尾声的这三处设计——在 `compute_logits` 里拼车共读一遍 1.09 GB 权重、在 `_copy_additions` 里趁着转 FP32 顺手加完偏置并保留 BF16 舍入、在 `_sample` 里只为那四五个幸存词抽取权重行并用 `uint64` 位翻转排序——你会发现，跟 Memory-Bound 的硬件打交道其实是一件很讲礼貌的事：不要总是急着另起炉灶去发新内核，先看看显存总线上哪趟班车正要发车，把你的算术塞进它的空座位里，算力自然就成了免费的赠品。
