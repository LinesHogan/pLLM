# 采样尾声融合

如果你想在解码阶段用一个小网络（distiller）去修正大模型的采样分布——比如在 ESamp 里，把大模型和小网络都觉得好的平庸 token 往下压一点，把只有大模型自己看准的 token 往上抬一点——它的数学公式简单得只有一行：

```text
new_logit = (1 + beta) * llm_logit - beta * distiller_logit
```

为了不让小网络把原本根本不靠谱的生僻词抬进候选池，这个公式只作用在大模型自己做完 `top-k` / `top-p` / `min-p` 过滤之后留下的幸存候选上（也就是所谓的“后过滤”语义，post-filter）。

要是让你在 PyTorch 里把这个逻辑写出来，你大概两分钟就能敲完：

```python
# check-docs: skip: illustrative naive version, not pLLM code
llm_logits = model.lm_head(hidden)
distiller_logits = model.lm_head(distiller_hidden)
```

接着跑一遍采样器的 `top-k` / `top-p` 过滤，拿到掩码后的 `survivors`（被淘汰的位置填成了 `-inf`），再写一行逐元素的张量重加权：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/sampler.py
def _reweight_unfused(survivors: torch.Tensor, prediction: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # ……
    return torch.where(
        torch.isfinite(survivors), (1.0 + beta) * survivors - beta * prediction.to(survivors.dtype), survivors
    )
```

最后把改好的 logits 丢给采样器抽样。代码清晰、直观，甚至带着一点函数式编程的优雅。

但如果你把这几行代码原封不动地塞进 vLLM 的 decode 循环里，打开性能分析器（profiler）看一眼，你会倒吸一口凉气：**在模型主体之外，光是最后这几行和词表打交道的收尾代码，就能生生吃掉几百微秒甚至上毫秒的 GPU 时间。**

为什么几行最基础的线性代数和逐元素算术，一碰到词表（vocabulary）就变得这么贵？我们又是怎么在不增加哪怕一遍词表遍历的前提下，把这些自定义计算像变魔术一样塞进推理本来就要做的步骤里的？

这就是标题里“采样尾声（sampler epilogue）”的意思：在 GPU 算子世界里，**Epilogue（尾声）** 指的是趁着数据还热乎地揣在寄存器里、还没写回显存之前，在已有遍历的尾巴上顺手把私活干完。

---

## 先算一笔显存带宽的硬账

> **本文实验基准环境**：除特别注明对比的 0.5B 与历史数据外，本文性能数据均在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0、PyTorch 2.13.0+cu129 下测得，开启 CUDA Graph 与 FlashInfer 采样器（`VLLM_USE_FLASHINFER_SAMPLER=1`）。基准负载为 `Qwen/Qwen2.5-7B-Instruct`（BF16），并发 128 行（8 个 prompt $\times$ `n=16`），生成 256 个 token，温度 0.8，`top-p` 0.95，取预热后 3 轮中位数。在该设置下，原生 vLLM 吞吐为 **5,894 tok/s**（单步耗时 **21.7 ms**）；同配置下 `Qwen2.5-0.5B-Instruct` 原生吞吐为 **42,355 tok/s**（单步耗时 **3.0 ms**）。

很多刚接触推理系统的同学容易有一个错觉：大模型推理慢，是因为 Transformer 那几十层里的矩阵乘法（GEMM）计算量太大。

在一次性处理几千个 prompt token 的预填充（prefill）阶段，确实如此。但在一个字一个字往外蹦的解码（decode）阶段，算力早就过剩了，真正的瓶颈只有一个：**显存带宽（HBM Bandwidth）**。

让我们拿计算器按两下。`Qwen2.5-7B` 的隐藏层维度是 `3584`，词表大小是 `152064`。这意味着模型最后一层那个把隐藏状态映射成词表得分的 `lm_head` 权重，是一个形状为 `3584 × 152064` 的 BF16 大矩阵——光这一个矩阵，就重达 **1.09 GB**（在 `0.5B` 模型上则是 `896 × 151936`，约 **272 MB**）。

在 decode 的每一步里，128 个并发请求只产生 128 行隐藏向量。让 `[128, 3584]` 去乘 `[3584, 152064]`，这点浮点乘加运算还塞不满 RTX 4090 的牙缝；但为了算出这 128 行的结果，GPU 必须老老实实地把那 **1.09 GB** 的权重从显存里完整读进芯片一遍。在微基准测试里，7B 模型跑一次 128 行的 LM head 投影要 **1,349 µs**（0.5B 模型是 **359 µs**）。芯片绝大部分时间都在干等显存控制器送数据过来。

看清了这个物理事实，两条推论立刻浮出水面：

1. **单独再调一次 `lm_head` 是不可饶恕的浪费**。如果你像开头那段朴素代码一样，先算完模型的 `lm_head(hidden)`，再单独调一次 `lm_head(distiller_hidden)`，GPU 就得把那 1.09 GB 的权重从显存里读两遍。白白多花 1.35 ms，在 7B 上直接烧掉整步时间的 6%（在一步只有 3.0 ms 的 0.5B 上更是高达 12%）。
2. **反过来，“搭顺风车”几乎是免费的**。既然权重反正要从显存读进寄存器一遍，如果在读进来的同时，让小网络那 128 行预测向量和模型自己的 128 行拼在一起算，显存读取量一字节不增，多出来的算术全由闲置的 Tensor Core 顺手消化了。

还没完，出了 `lm_head`，接力棒交到了采样器手里。

这时候，你面前的 `logits` 张量形状是 `[128, 152064]`。用 FP32 存，一张张量就是 **77.8 MB**。采样器要在这张大表上来回扫：在 7B 的一次剖析（ESamp $\beta=0$）里，除以温度（temperature）大约要 **125 µs**，`top-k` / `top-p` 过滤大约要 **283 µs**，最后的 Gumbel-max 随机抽样大约要 **78 µs**。

在这张接近 80 MB 的大表上，每多跑一个逐元素（elementwise）小算子，就要多付几十到上百微秒的显存读写税。回头看看刚才那行看似无害的 `_reweight_unfused`：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/sampler.py
return torch.where(
    torch.isfinite(survivors), (1.0 + beta) * survivors - beta * prediction.to(survivors.dtype), survivors
)
```

在默认的不融合（eager）模式下，PyTorch 会怎么执行它？`isfinite` 扫一遍、`to(dtype)` 扫一遍、`1.0 + beta`、乘 `survivors`、乘 `prediction`、相减、`where` 选择……整整 **七遍** 对 `[rows, vocab]` 的遍历！在旧版 0.5B 的测量里，光是朴素 dense 引导的投影（约 118 µs）加词表重加权（约 187 µs）等开销，每步就要多花约 **401 µs**，比 0.5B 模型自己的 LM head 投影（359 µs）还要贵。

所以我们给 pLLM 的采样后端立下了一条铁律：**绝不新增任何一遍对词表的全量遍历；所有用户声明的词表操作，必须全部塞进推理本来就要做的那几遍遍历里。**

推理本来非做不可的遍历里，我们挂靠的是这三遍（温度、过滤那几遍照旧）：
1. **第一遍**：`lm_head` 把 hidden 投影成词表宽度的 BF16 logits；
2. **第二遍**：vLLM 把 BF16 logits 拷成一份 FP32 工作副本交给采样器处理；
3. **第三遍**：采样器扫描过滤后的幸存候选，注入 Gumbel 噪声并抽出最终的 token。

```text
Naive Multi-Pass Pipeline (Every box is a full [128, 152064] HBM sweep or 1.09 GB weight read):
  hidden ──► [LM Head #1] ──► [BF16->FP32 Copy] ──► [+Bias #1] ──► [+Bias #2] ──► [Top-k/p Filter]
                                                                                         │
  distiller ─► [LM Head #2 (1.09 GB!)] ─────────────────────────► [7x Reweight Passes] ◄─┘
                                                                          │
                                                                          ▼
                                                                   [Gumbel Draw]

pLLM Sampler Epilogue (Zero extra full-vocab sweeps):
  hidden (+ distiller) ──► [1. Shared LM Head] ──► [2. Copy + Ordered Biases] ──► [Top-k/p Filter]
                                                                                         │
  distiller ──────────────────────────────────────────────► [3. Survivor Dot + Reweight + Gumbel]
  (default bounded path: distiller skips LM Head!)
```

下面我们就顺着这三遍遍历，看看怎么把三件差事一件一件塞进去。

---

## 第一遍：在 `compute_logits` 里拼车共用 LM Head

第一件差事：当我们需要稠密（dense）的词表预测分布时，怎么省掉第二遍读那 1.09 GB 的 `lm_head` 权重？

在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里我们讲过，vLLM 把 Transformer 主干录进了分段的 CUDA graph，而 `compute_logits`（即调用 `lm_head` 的地方）恰好站在主干图结束后的边界上。pLLM 在 `pllm/backends/vllm/session.py` 里接管了这个入口：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/session.py
for guidance in self._sampler_guidance():
    if guidance.stage is SamplerStage.PACKED_POST_FILTER_EXACT and self._guidance_weight is not None:
        capacity = self.attachment.spec.guidance_capacity
        assert capacity is not None
        stages.append(BoundedSamplerGuidance(guidance, self._guidance_weight, capacity))
        continue
    if guidance.stage is not SamplerStage.PACKED_POST_FILTER_EXACT:
        predicted = guidance.values
    elif logits is None:
        projected = lm_head(torch.cat((hidden, guidance.values.to(hidden.dtype))))
        assert projected is not None
        logits, predicted = projected[: len(hidden)], projected[len(hidden) :]
```

先跳过第一个 `if`（那是我们一会儿要讲的第三遍绝活 `BoundedSamplerGuidance`），盯住 `elif logits is None:` 下面那两行：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/session.py
projected = lm_head(torch.cat((hidden, guidance.values.to(hidden.dtype))))
# ……
logits, predicted = projected[: len(hidden)], projected[len(hidden) :]
```

动作干净利落：把模型的 `hidden` 和小网络预测出来的 `guidance.values` 在第 0 维（行维度）上用 `torch.cat` 拼成一张两倍长的表，喂给同一个 `lm_head`，出来之后再一刀切开，前半截归模型，后半截归引导器。1.09 GB 的权重只从显存里走了一遭。

说到这儿，如果你被 GPU 浮点数的非结合律坑过，你心里可能会犯嘀咕：**把 128 行拼成 256 行喂给 cuBLAS，模型原本那 128 行的输出会不会因为拼车而发生微小的数值漂移？**

答案是：会变一点点，但它的变化方式，跟“连续批处理里恰好新加进来几个请求、让 batch 变大了一点”一模一样。

cuBLAS 会根据输入矩阵的行数（M 维）挑选不同的分块（tiling）内核，不同的分块顺序会带来尾数最低位的舍入差异。我们在 `Qwen2.5-0.5B` 和一个 `DeepSeek-V2-Lite` 切片上专门量过这种拼车带来的偏差：它从未超过原生 vLLM 自己在不同 batch 大小下的天然敏感度（在两个模型上分别是 `0.0625` 和 `0.5`）。在一个每步 batch 大小本来就随着请求进出而动态起伏的连续批处理引擎里，这点差别完全落在系统固有的数值包络之内。

当然，“拼车”也有一条明确的物理边界：**你得等到所有人都到齐了，才能发车。**

如果你的算法不是在最后一步做引导，而是像 DoLa（对比层解码）那样，通过 `pllm.layer_logits` 去读取模型一组中间层（默认 0、2、……、14 层）的词表分布呢？这些层的行会在 LM head 处单独做一次矩阵乘——为了让模型自己的 logits 与原生 vLLM 逐位相同，它们不和模型的行拼车；更重的是，每个 token 都要在 15 万词的词表上为 9 个层算出 FP32 logits 和散度。在我们的示例 06 中，7B 模型（32 并发 = 8 $\times$ `n=4`）同时读 9 层中间分布的 DoLa，吞吐会掉到原生的 **66%**（早先在主干极轻的 0.5B 模型上测得的更是只有 **25%**）。这是硬件定律开出的诚实账单——没有捷径可走的时候，pLLM 会通过 `pllm.explain` 把这 1.7 倍（7B）乃至 3.5 倍（0.5B）的每 token 算力膨胀原原本本地算给你看。

---

## 第二遍：把 `@pllm.add_logits` 折进必经的 FP32 拷贝

现在我们拿到了 `lm_head` 吐出来的 BF16 `logits`。在进入复杂的后过滤引导之前，先看一个更普遍的需求：`@pllm.add_logits()`。

很多算法需要在采样器做 `top-k` / `top-p` 过滤**之前**，往词表分布上加一个稠密的偏置向量——比如根据最近几个输入 token 从一张大表里查出来的 n-gram 先验（`NGramPrior`），或者某种结构化约束的惩罚项。你甚至可以在一个 workload 里叠加声明好几个 `@pllm.add_logits()`。

朴素的做法是什么？用户在插件里算好偏置张量，发射一个加法 kernel 加到 `logits` 上；有三个 `@pllm.add_logits()`，就发射三个加法 kernel，把 80 MB 的词表张量来回读写三遍，然后再交给 vLLM 的采样器。

但如果我们掀开 vLLM 新版模型执行器（MRv2 runner）的盖子看一眼，就会发现一个现成的“顺风车”：**vLLM 在对 logits 做温度缩放、惩罚和过滤之前，第一步本来就要把 `lm_head` 输出的 BF16 张量拷贝并转换成一份 FP32 张量！**

既然 vLLM 反正要把 `[rows, vocab]` 读一遍、写一遍，我们为什么不把所有的 `@pllm.add_logits()` 统统塞进这一次必经的拷贝里？

遗憾的是，去翻翻 vLLM 自己的温度、过滤和拷贝算子接口，它们全都没有留“顺手加一个稠密张量”的参数槽位；而 vLLM 提供的稀疏 `logit_bias` 接口又是给个别固定 token 准备的，根本塞不下运行时动态算出来的整表张量。

既然上游没有这个旋钮，我们就写一个最小的 Triton kernel，精准替换掉那一次 BF16 到 FP32 的拷贝（`pllm/backends/vllm/sampler_epilogue.py`）：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/sampler_epilogue.py
loaded = tl.load(Logits + row * LS + tokens * VS, valid, other=0)
dtype = loaded.dtype
value = loaded.to(tl.float32)
for stage in tl.static_range(len(Additions)):
    selected = row
    if RowMap is not None:
        if not FULL[stage]:
            selected = tl.load(RowMap + stage * ROWS + row)
    prediction = (
        tl.load(
            Additions[stage] + selected * STRIDES[stage][0] + tokens * STRIDES[stage][1],
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

这段只有二十行的代码里，藏着两件特别有意思的工程细节。

### 1. 为什么要故意“降级”回 BF16 再转回 FP32？

请你盯住循环里的最后一行：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/sampler_epilogue.py
value = tl.where(selected >= 0, (value + prediction).to(dtype).to(tl.float32), value)
```

看到那个 `.to(dtype).to(tl.float32)` 了吗？这里的 `dtype` 是输入 `Logits` 的类型（也就是 BF16）。

你肯定觉得奇怪：我们都已经把 `value` 和 `prediction` 转成高精度的 `tl.float32` 放进寄存器了，最后写进 `Output` 的也是 `tl.float32`，为什么每加完一层偏置，还要特地把它截断回粗糙的 BF16，然后再转回 FP32 呢？直接在 FP32 里一路加到底，既省指令、精度又高，岂不是一举两得？

千万不能贪这个小便宜。

想想看，当一个用户在代码里按顺序声明了两个 `@pllm.add_logits` 时，如果不做尾声融合，它的原始数学语义是：第一个方法加在 BF16 的 `logits` 上，物化成一张 BF16 张量；第二个方法再加在这张新的 BF16 张量上。如果我们在融合进拷贝算子时，自作主张改成了全程 FP32 累加，那就不叫“等价融合优化”，而叫“偷偷改了用户的算法”。在 `top-p` 截断边界上，某两个候选词的概率之和可能刚好卡在 `0.9500001`——少了那一次 BF16 舍入，原本该被踢出局的词留了下来，随机抽样的随机数链条随之错位，后面生成的整段话就会面目全非。

**系统优化的第一守则，是只省时间，绝不改命。** 加上那一次 `.to(dtype)` 舍入之后，我们在 3 个不同的工作负载、每个负载 98,304 个生成 token（合计 **294,912 个 token**）上做了融合前后的逐 token 比对：不一致的 token 数量是严格的 **0**。

（顺便说一句，当用户通过 `@pllm.add_logits(rows=...)` 只对批次里的部分行加偏置时，主机会在每一步把行映射列表异步上传成上面代码里的 `RowMap`，只有 `selected >= 0` 的行才会触发累加和舍入。虽然有极小的一点宿主上传开销，但彻底省掉了先把子集拷出来、加完再 `index_copy_` 写回去的整表折腾。）

### 2. `wrap_triton` 的隐秘内存泄漏

在把 `_copy_additions` 这个 Triton kernel 接进 PyTorch 的时候，我们还踩过一个让人后背发凉的坑。

起初，我们用 `torch.library.wrap_triton` 来包装这个 kernel。因为一个工作负载里可能有任意多个 `@pllm.add_logits` 阶段，我们把这些偏置张量打包成一个 Python 元组 `tuple(additions)` 传了进去。短跑测试一切正常，直到我们在一次长跑压测里眼睁睁看着显存水位一格一格往上涨。

抓虫抓到最后，真相让人哭笑不得。来看看现在 `sampler_epilogue.py` 里改用 `@torch.library.custom_op` 后的注释：

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

原来，在 Triton 与 PyTorch 的胶水层里，如果你传进一个单独的 `torch.Tensor`，它认得这是运行时数据，只取裸指针；但如果你传进一个 `tuple[torch.Tensor, ...]`，`wrap_triton` 会把这个元组当成**常量**存进一张侧表里！

这意味着，每跑一步 decode，那一步临时算出来的偏置张量就会被这张侧表攥住、跨步留在显存里，垃圾回收器收不走。改成标准的 `@torch.library.custom_op`、用 `list[torch.Tensor]` 作为算子级运行时输入后，引用在算子返回的那一刻干脆释放。选错一层薄薄的抽象边界，代价往往不是跑得慢，而是静悄悄地漏光你的显存。

在一步只有 3 ms、对固定词表开销最敏感的 0.5B 模型上（示例 11 `NGramPrior`，vLLM 0.27.1 时期测量），把 `add_logits` 折进这次拷贝让三次 kernel 发射缩减为一次：原来独立的加法约 **218 µs**，再加上 vLLM 原生自己那次单纯的 FP32 拷贝约 **111 µs**，这一段约 **329 µs**；融合后整段只要约 **180 µs**。端到端吞吐在显存查表下从 33,110 提升到 **34,452 tok/s（+4.05%）**，主机内存查表下从 30,786 提升到 **32,273 tok/s（+4.83%）**，带 `rows=` 子集选择时也提升了 **2.72%**。

---

## 第三遍：只为“幸存者”取权重

现在，我们终于来到了全篇最精彩的一幕：`@pllm.guide_sampling`。

前面第一遍里我们说过，在 `compute_logits` 里把小网络的预测拼在模型 hidden 后面一起乘 `lm_head`，虽然省掉了重读 1.09 GB 权重的时间，但依然算出了一张实打实的 `[128, 152064]` 稠密预测表（在 0.5B 上每步是 **34.85 GFLOP** 的矩阵乘），随后还得在过滤后的词表上跑一遍重加权 kernel。

有一天，我盯着 `top-p = 0.95` 过滤之后的 logits 张量，突然问了自己一个愚蠢的问题：

**经过 `top-p = 0.95` 截断之后，152,064 个候选词里，到底还有几个词是活着的（即不是 `-inf`）？**

我们去 ESamp 的真实生成语料里（0.5B，温度 0.8，`top-p` 0.95，共 652,800 行）数了一下。结果让人大跌眼镜：
* 平均每一行留下的幸存者（survivors），只有 **4.83 个** token！
* 在整整 652,800 行里，出现过的最大真实核（nucleus）也不过 **2,785 个** token。

想一想这意味着什么。大模型说话时其实非常笃定，绝大多数时候，95% 的概率质量全集中在前四五个词上，剩下那 **152,059 个词** 全都被过滤成了 `-inf`。可我们的稠密做法在干嘛？为了给这 5 个幸存者算引导分，我们兴师动众地把小网络的隐藏向量跟整个 15 万词的 `lm_head` 矩阵做了一次完整乘法，算出 152,064 个数，然后把其中 152,059 个数直接扔进垃圾桶！

既然只有 5 个词活下来了，我们干嘛要在过滤之前投影整个词表？**等上游把 `top-k` / `top-p` 过滤做完，看看哪 5 个词还活着，直接去 `lm_head` 权重矩阵里把那 5 行挑出来，跟小网络的 hidden 做 5 次向量点积，当场改分、当场抽样，不就完了吗？！**

```text
Dense Post-Filter Guidance (Projects all 152,064 columns, discards 99.99%):
  distiller [1, H] ──► (x LM Head [H, 152064]) ──► [152064] ──► mask & reweight ──► draw

Bounded Fused Guidance (Only surviving tokens touch the LM head):
  filtered logits [152064]:  [ -inf, -inf,  2.1, -inf, ...,  0.8, -inf ]
                                             │                │
                         1. Compact survivors into Scratch (avg ~4.8 tokens!)
                                             ▼                ▼
                             Scratch:    [ (2.1, tok_2),  (0.8, tok_k) ]
                                                 │                │
                         2. Gather ONLY those rows from LM Head Weight [152064, H]
                                                 ▼                ▼
                             dot(hidden, W[tok]) -> reweight -> gumbel_block_argmax!
```

这个想法一旦成型，整个采样尾声就坍缩成了一个一气呵成的 Triton kernel——`pllm/backends/vllm/guided_sampling.py` 里的 `_sample`。

在这个 kernel 里，每一行（即每一个待采样的请求序列）分配一个独立的线程块（program）。它先调 `_scan_survivors` 扫一遍上游刚刚过滤好的 `Logits`，把有限值（`finite`）压紧写进一块固定容量的暂存区 `Scratch`；紧接着，只为 `Scratch` 里的这几个幸存者去 `lm_head`（即下面的 `Weight`）里捞权重行：

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
    prediction = prediction.to(Weight.dtype.element_ty).to(tl.float32)
    selected_logits = tl.where(valid, (1 + beta) * selected_logits - beta * prediction, float("-inf"))
    tl.store(Logits + row * LS + selected_tokens, selected_logits, valid)
    value, token = _draw(
        selected_logits, selected_tokens, valid, row, Mapping, Temperature, Seeds, Positions, VOCAB, FP64, V027
    )
    replace = (value > best) | ((value == best) & (token < winner))
    best, winner = tl.maximum(best, value), tl.where(replace, token, winner)
```

我们一行行来品味这段代码里的匠心：

1. **间接寻址，按需取货**：看 `Weight + selected_tokens[:, None].to(tl.int64) * WS + hidden_axis[None, :]` 这一行。`selected_tokens` 是幸存下来的那几个词的 ID。如果这一行只剩 5 个词，我们就只从 1.09 GB 的权重矩阵里精准抽出这 5 行（在 7B 上一共才 $5 \times 3584 \times 2 \approx 35\text{ KB}$！）。投影的计算量直接从 `O(vocab × hidden)` 暴跌到 `O(survivors × hidden)`——在 0.5B 上，就算按最大容量上界给 `pllm.explain` 报账，每步也不超过 **0.94 GFLOP**（而实际平均只有约 5 行）。
2. **连原生矩阵乘的舍入脾气都模仿到位**：算出 FP32 点积 `prediction` 后，下一行紧跟着 `prediction = prediction.to(Weight.dtype.element_ty).to(tl.float32)`。为什么又要转回一次权重类型（BF16）？因为原生未量化的 `lm_head` 在做完矩阵乘输出时，结果就是落回 BF16 的。多这一拍类型转换，保证了间接点积算出来的 `prediction` 和全词表矩阵乘算出来的值严丝合缝。
3. **在自己的 kernel 里直接调上游的 Gumbel 抽样原语**：算完 `selected_logits` 后，怎么把最终的 token 抽出来？如果自己手写一套伪随机数生成器和 Gumbel 噪声公式，不仅容易写错，更致命的是**破坏随机种子的可复现性**——用户给定同一个 `seed`，在原生 vLLM 和 pLLM 里的采样轨迹就会分道扬镳。
   所以我们看了一眼 vLLM 的源码，发现它的 Gumbel-max 抽样核心本来就是一个 `@triton.jit` 函数：`gumbel_block_argmax`！于是我们的 `_draw` 辅助函数直接在 Triton 内核内部调用了上游的 `gumbel_block_argmax`（并自动适配了 vLLM 0.27 与 0.32 两个版本的参数签名）。请求的随机种子、当前生成位置 `Positions`、温度、FP32/FP64 噪声精度乃至贪心决胜规则，百分百由上游自己的代码执行。

这里还藏着一个只有跑过海量真实数据才会撞见的角落案例（corner case）。

在其中一组 163,200 个引导行里，我们发现有 **0.12%** 的行显得极其诡异：上游过滤器跑完之后，这几行里居然**整行 15 万个词全都是有限值**！难道模型在这几行上彻底懵了，觉得 15 万个词雨露均沾？

恰恰相反！我们把这 0.12% 的行拉出来做解剖，发现它们的第一名（top-1）概率全都高得离谱——在 **0.99999994** 以上，第一名比第二名的 logit 足足高出了 **17 到 24**。因为第一名一个词就已经超过了 `top_p = 0.95` 的阈值，上游过滤器干脆跳过了写 `-inf` 掩码的步骤，把整行原样放了过来。

如果不明就里地按“所有有限值都是幸存者”去处理，这极度自信的 0.12% 行反而会退化成全词表处理。于是我们在 `_scan_survivors` 扫描时顺手维护了在线最大值与指数和（`mass`）：一旦发现某一行全满、且 `1.0 / mass >= top_p`（即第一名自己就吃够了 `top_p` 的概率质量），内核就会把它的 `peak_key` 拎出来，显式还原成它真正的**单 token 核（one-token nucleus）**，只投影那唯一的第一名。

---

## 把决胜规则写进比特里：`uint64` 位翻转把戏

在刚才那段内核代码里，不知道你有没有注意到一个神秘的配对函数：`selected_logits, selected_tokens = _decode(selected_keys)`，以及扫描时生成它的 `_keys`。

为什么我们要把 `(logit, token_id)` 打包成一个叫 `key` 的东西？

因为在 GPU 上挑出幸存者（尤其是当幸存者数量超过暂存区容量、需要挑出前 $K$ 大时），我们必须遵守一条严格的排序决胜规则（tie-breaking）：
1. 首先按 `logit`（32 位浮点数）**从大到小（降序）**排；
2. 如果两个词的 `logit` 恰好一模一样，必须按它们的 `token_id`（32 位整数）**从小到大（升序）**排。

在 CPU 上写 Python，你随手写个元组 `(-logit, token_id)` 就交差了。但在 GPU 的寄存器和共享内存里，同时搬运两个数组、每次比较和交换都写一堆带分支的复合条件，既占寄存器又拖慢指令流水线。

既然一个 `float32` 正好是 32 个比特，一个 `token_id`（`uint32`）也正好是 32 个比特，我们能不能施展一点位操作魔法，把它们俩直接拼成一个 **64 位无符号整数（`uint64`）**，让硬件用一条原生的整数大于号 `key_a > key_b`，就把“浮点降序 + 整数升序决胜”同时搞定？

来看看 `guided_sampling.py` 开头这短短五行的 `_keys` 函数。这是整个采样后端里我最偏爱的一段代码：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/guided_sampling.py
@triton.jit
def _keys(logits: tl.tensor, tokens: tl.tensor, finite: tl.tensor) -> tl.tensor:
    logits = tl.where(logits == 0, 0.0, logits)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(finite, (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - tokens.to(tl.uint32)), 0)
```

这五行代码里装着两个老派系统程序员津津乐道的位操作把戏（bit-twiddling hacks）。

第一个把戏在 `ordered` 那一行：**怎么把一个 IEEE 754 浮点数变成可以直接当无符号整数比大小的东西？**

当年设计 IEEE 754 浮点数标准的工程师非常体贴：他们把符号位放在最高位（第 31 位），指数位紧随其后（第 30..23 位），尾数位放在最低位（第 22..0 位）。这种排布带来了一个美妙的性质：对于任意两个**正浮点数**，如果你直接把它们的内存比特当成无符号整数 `uint32` 来比大小，指数大的自然大、指数相同尾数大的自然大——顺序本来就是完全一致的！

唯一的麻烦出在符号位和负数上：
* 正数的符号位是 `0`（落在 `0x00000000 .. 0x7FFFFFFF`），负数的符号位是 `1`（落在 `0x80000000 .. 0xFFFFFFFF`）。直接按无符号整数比，负数反而比正数大；
* 更糟的是，IEEE 754 的负数不是补码，而是“符号 + 绝对值（sign-magnitude）”表示法——绝对值越大（也就是越负、越小）的负数，它的指数和尾数比特反而越大！

怎么用两三条位运算把整个实数轴重新捋顺？
* **如果是正数**（`bits & 0x80000000 == 0`）：我们只把它的最高位异或翻转成 `1`（`bits ^ 0x80000000`）。这样所有正数保持原有顺序，整体搬家到了上半区 `[0x80000000, 0xFFFFFFFF]`；
* **如果是负数**（`bits & 0x80000000 != 0`）：我们把它的全部 32 个比特**按位取反**（`~bits`）！最高位的 `1` 变成了 `0`，让所有负数住进了下半区 `[0x00000000, 0x7FFFFFFF]`；与此同时，按位取反正好把“绝对值越大比特越大”倒了个个儿——原本越负的数，取反之后无符号整数值就越小！
* （至于第一行的 `tl.where(logits == 0, 0.0, logits)`，则是为了把 IEEE 754 里讨厌的 `-0.0` 统一洗成 `+0.0`，防止 `-0.0` 和 `+0.0` 映射出两个不同的整数。）

经过这一下变换，从 $-\infty$ 到 $+\infty$ 的所有浮点数，被严丝合缝、单调保序地映射到了 `0x00000000` 到 `0xFFFFFFFF` 的无符号整数轴上。

第二个把戏在最后一行。我们把保序的 `ordered` 左移 32 位塞进 `uint64` 的高半边；那低 32 位怎么表达“同分时 `token_id` 越小越优先”？很简单，拿 `0xFFFFFFFF` 减去 `token_id`，把升序倒成降序：

```text
 63                              32 31                               0
┌──────────────────────────────────┬──────────────────────────────────┐
│    Order-Preserving Logit Bits   │      0xFFFFFFFF - token_id       │
│  (+: flip MSB,  -: bitwise NOT)  │   (inverts ascending tie-break)  │
└──────────────────────────────────┴──────────────────────────────────┘
       Larger logit wins first           Smaller token_id wins ties
            (0 = masked / -inf sentinel, always loses)
```

看，原本啰嗦的双字段复合排序规则，现在彻底写进了数的二进制表示里。不管是在 `_select_top_survivors` 里调用 `tl.sort(..., descending=True)`，还是用 `tl.maximum` 找峰值，硬件只需要闭着眼睛比较一个 `uint64` 的大小。用完之后，`_decode` 再用完全对称的位运算把 `logit` 和 `token_id` 无损拆回来：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/guided_sampling.py
@triton.jit
def _decode(keys: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where(ordered & 0x80000000 != 0, ordered ^ 0x80000000, ~ordered)
    return bits.to(tl.float32, bitcast=True), (0xFFFFFFFF - keys.to(tl.uint32)).to(tl.int32)
```

---

## 容量原则：为什么 `K=4096` 比 `K=256` 反而更快？

为了让内核能跑在静态的 CUDA 流与图里，我们不能在 GPU 运行时动态 `malloc` 内存。这意味着存放幸存者的那块暂存区 `Scratch` 必须提前定死一个最大行容量——也就是配置项里的 `ExecutionOptions.guidance_capacity`（代码里的常数 `K`）。

如果某一行特别发散，过滤后活下来的幸存者个数 `count` 超过了 `K`（即发生了溢出，overflow），内核会怎么办？

它不会报错，也不会退回去投影整个词表，而是启动一条**溢出路径**：调用 `_select_top_survivors`，每次读入一个词表分块（tile），用 `tl.sort` 归并保留当前 `logit` 最高的前 `K` 个幸存者去接受小网络引导；至于排在 `K` 名以外的那些次要幸存者，则交给 `_draw_tail` 作为“未引导的尾巴”参与 Gumbel 抽样。因为涉及分块排序和额外扫描，这条溢出路径比直接压紧的主路径贵得多。

现在，假设由你来拍板 `guidance_capacity` 的默认值，你会选多少？

前面我们刚数过，ESamp 里平均每行只有 **4.83 个**幸存者！任何一个习惯了“内存越省越好”的程序员，第一反应大概都是：既然平均才 5 个，那选 `K = 256` 简直绰绰有余了吧？数组越小，分配越快，缓存越友好！

我们也这么想过，直到我们把 `K = 256`、`1024`、`4096` 和 `None`（不做有界融合，走第一遍那种精确全词表 dense 路径）放在一起跑了一次实测。

先看一步只有 3.0 ms、采样器耗时占比被放大的 `0.5B` 模型（128 并发，$\beta=0.5$，7 轮中位数）：
* `K = 256`：吞吐 **33,737 tok/s**
* `K = 1024`：吞吐 **34,069 tok/s**
* **`K = 4096`（默认）**：吞吐 **34,742 tok/s**
* `None`（精确 dense）：吞吐 **32,624 tok/s**

你没看错：**把容量从 4096 缩减到 256，吞吐不仅没涨，反而掉了约 3%！**

为什么会这样？因为在 128 个并发请求行里，哪怕有 127 行都只有四五个幸存者，只要有**一行**的幸存者数量超过了 256（别忘了我们在语料里见过的最大真实核是 **2,785**），负责那一行的 GPU 线程块就会一头栽进需要反复调用 `tl.sort` 的 `_select_top_survivors` 溢出路径里。整批请求的完成时间取决于最慢的那一个线程块（长尾效应），你省下的那一点点暂存区空间，全被溢出行昂贵的分块排序给赔光了。

而在 `7B` 模型上（一步 21.7 ms，主干每步要读约 15 GB 权重，采样器的相对占比变小），各容量的实测表现如下：

| `guidance_capacity` | 吞吐（tok/s） | 相对原生 vLLM（5,894 tok/s） |
|---|---:|---:|
| `256` | 5,684 | 96.4% |
| `1024` | 5,684 | 96.4% |
| **`4096`（默认）** | **5,678** | **96.3%** |
| `None`（精确 dense） | 5,533 | 93.9% |

在 7B 上，256 到 4096 之间的差异已经完全收敛在测量噪声之内，而有界融合路径整体比精确 dense 路径（`None`）快了约 **2.6%**（在 0.5B 上则是快了约 **6.5%**，34,742 vs 32,624）。如果将引导强度设为 $\beta=0$，pLLM 根本不会声明采样器阶段，直接实现零采样开销（7B 下 ESamp $\beta=0$ 吞吐为 **5,738 tok/s**，达原生的 **97.4%**）。

这就是为什么我们把默认容量定在了 **4096**，并从中总结出了一条在系统设计里反复应验的经验——我们称之为**容量原则（The Capacity Principle）**：

> **在固定容量的数据结构中，容量的使命不是去拟合“平均情况”，而是要选到足够大，让现实中的常见情况永远看不见那条复杂的溢出路径。**

选 `4096`（大于实测最大真实核 `2,785`），意味着在正常生成中，几乎 100% 的行都直接走最轻快的无排序压紧路径。而为此付出的代价是多少呢？每行 4096 个 `uint64`，128 行加起来一共才 **4 MiB** 的临时显存——在动辄几十 GB 的大模型推理里，连一滴水花都算不上。

那么，有界融合在数值上到底准不准？在 0.5B 的四组实测语料上，既然最大真实核（2,785）也没超过 4096，有界路径与精确 dense 引导在 **163,200 个引导行**上的共享 Gumbel 抽样结果达到了 **100% 逐 token 一致**；两者的策略分布误差（每行概率分布的总变差距离，TV distance）的 p99.9 仅为约 **$6 \times 10^{-14}$**（另一份报告中约为 $1.5 \times 10^{-10}$，均属于浮点舍入噪声量级）。

当然，我们必须把话摊开说清楚：这是在 ESamp 的参数设置下的实测结果，而不是对任意小网络、任意温度或极端 `top-p` 的数学绝对承诺。当确实发生超过 $K$ 的溢出时，只引导前 $K$ 个候选、让尾巴保持未引导，是一个有意识的工程近似。如果你的算法实验要求在任何极端分布下都保持百分百的精确全词表后过滤语义，只需要传一个 `ExecutionOptions(guidance_capacity=None)`，系统就会退回第一遍的拼车 dense 路径（并用 `torch.compile` 把那七遍重加权融合成一个算子）。

---

## 演进的脚印与坦诚的边界

今天你看到的这三处尾声融合，并不是我们第一天坐在白板前就画出来的完美蓝图，而是踩过一串坑之后留下的幸存者。

早在做上一代原型（旧 tLLM）时，我们就围着这最后几百微秒折腾过好几个来回：
* 最早一版老老实实做后过滤精确实现，在一个很小的烟测设置下，吞吐比只有惨淡的 **0.6875**；
* 为了提速，我们试过“前过滤 dense”（先在全词表上把引导加进去，再交给采样器过滤）——速度是快了，但因为小网络在过滤前就篡改了词表分布，把原本不该进候选池的词顶了上来，破坏了后过滤的算法语义，只能当作性能参照；
* 接着我们又试过“后过滤 + 全词表 dense 缓存”（预先算好全词表的预测存起来，过滤后再挑），结果实测下来写了句教训在优化日志里：**“full-vocab dense-cache 不自动更快”**——省不掉全词表读写，怎么倒腾缓存都是隔靴搔痒；
* 直到我们在 7B 上尝试“只在过滤后的候选集合内部做采样”，吞吐比才从约 **85.3%** 跳到了 **89.7%**。

今天的有界融合内核，正是顺着这条路走到头的产物：既不在过滤前污染分布，也不预先投影整个词表，更不在 Python 里按行分组折腾。我们也遵从了[《复用上游：kernel、管理器与原语》](09-reusing-upstream-kernels.md)里的同一条准则：vLLM 的 FlashInfer 采样器已经把过滤和抽样融合得很好了，但它没有“通过选中的 embedding 行做间接点积再抽样”这个入口；所以我们只写上游没有的那一小段，而在抽样的最后一哆嗦，依然调回上游自己的 `gumbel_block_argmax`。

作为交换，这套精巧的机制也有它明确够不着的地方：

1. **有界路径的准入门槛**：因为内核要直接拿着 `selected_tokens` 去 `lm_head.weight` 里按行寻址做点积，它要求执行器是 MRv2、`tensor_parallel_size=1`，且 `lm_head` 是未量化、未加缩放或变换的标准权重。一旦开启多卡张量并行（TP > 1），`lm_head` 的词表维度被切分到了不同 GPU 上；或者当模型使用了量化权重，一行权重不再是连续的 BF16 数组——此时系统会自动回退到精确的 dense 路径（第一遍的 `torch.cat` 拼车依然生效）。
2. **`pllm.explain` 的上界报账**：在阅读 `pllm.explain` 的静态分析报告时，你会发现有界路径报出来的流量和算力是按最大容量 $K$（比如 4096）估算的悲观上界，有时候看起来甚至比某些静态值还大。记住那只是一个静态安全帽，运行时的真实搬运量取决于那平均 4.83 个幸存者。
3. **为什么只允许一个 `@pllm.guide_sampling`**：细心的读者会发现，pLLM 允许你叠任意多个 `@pllm.add_logits`，却在编译期严格限制整个工作负载最多只能声明**一个** `@pllm.guide_sampling`。这不是因为内核写不下循环，而是因为数学性质不同：过滤前的偏置加法满足交换律和结合律（$A + B = B + A$），叠十个也是确定的；但后过滤引导是在截断后的分布上做非线性对比重加权。如果出现两个引导器，它们该在隐藏层特征空间求平均，还是在概率空间相乘，还是先后截断两次？这是算法研究者自己的课题，框架绝不替你瞎猜。

---

回头看这三处改动——在 `compute_logits` 里拼车读一遍权重、在 `_copy_additions` 里顺手加完偏置并保持 BF16 舍入、在 `_sample` 里只为那四五个幸存者取权重行并把决胜规则塞进 `uint64` 的高低位——它们其实都在回答同一个问题：

**在被显存带宽锁死的解码尾声里，数据本来要走哪几趟？**

最快的词表算子，往往不是算术写得最花哨的那一个，而是懂得搭上现有的班车、在别人已经读进寄存器的尾巴上把活干完、最终连一趟多余的显存往返都没有发生的那一个。
