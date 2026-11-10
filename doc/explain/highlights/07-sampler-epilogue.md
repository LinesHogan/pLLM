# 采样尾声融合

ESamp 的采样引导只有一个公式。对一个请求、一步、一个候选 token：

```text
new_logit = (1 + beta) * llm_logit - beta * distiller_logit
```

模型自己觉得好、而一个只看第 0 层的小网络（distiller）也猜得到的 token，被往下压一点；小网络猜不到的，被往上抬一点。而且它只作用在模型自己过滤（top-k / top-p / min-p）之后留下的候选上——这叫“后过滤”（post-filter）语义。

在 PyTorch 里写出来是这样的（示意）：

```python
# check-docs: skip: illustrative naive version, not pLLM code
llm_logits = model.lm_head(hidden)
distiller_logits = model.lm_head(distiller_hidden)
```

再加一行重加权，交给采样器。两分钟就写完了。这篇讲的是：为什么这几行在 decode 里一点也不便宜，以及我们怎样不新增任何一遍对词表的遍历，把它们塞进推理本来就要做的那几遍里——这就是标题里“尾声”（epilogue）的意思：在一个已有的遍历的尾巴上，顺手把活干完。

## 先算一笔带宽账

Qwen2.5-7B 的 LM head 是一个 `3584 × 152064` 的 BF16 矩阵，1.09 GB；0.5B 的是 `896 × 151936`，约 272 MB。decode 时一步只有 128 行 hidden（8 个 prompt × n=16），乘这个矩阵的浮点运算量对 RTX 4090 来说微不足道，可这 1.09 GB 的权重得从显存完整读一遍。微基准里，7B 一次 128 行的 LM head 投影要 1,349 µs，0.5B 是 359 µs（RTX 4090）。这就是 decode GEMM 的本性：**被读权重的带宽锁死**，算得再快也在等数据。

从这个事实能推出两件事。第一，像上面那样单独再调一次 `lm_head`，就要再读一遍 1.09 GB，7B 上每步多约 1.35 ms，占一步（约 21.7 ms）的 6% 左右。第二，反过来，如果把小网络的预测行和模型自己的行拼在一起投影，权重只读一遍，多出来的行几乎是免费的。

采样器也不便宜。它要在 `[128, 152064]` 的词表张量上来回扫：7B 的一次 profile 里（ESamp β=0），temperature 约 125 µs，top-k/top-p 约 283 µs，Gumbel 抽样约 78 µs。在这张大张量上每多一遍逐元素操作，就是几十到上百微秒。而朴素的重加权 `where(isfinite(...), (1+β)·x − β·p, x)`，不融合的话是七遍。

所以我们给自己定了一条规矩：**不新增任何一遍对词表的遍历；所有工作都塞进推理本来就有的那几遍里。**推理本来就有的遍历有三遍：LM head 的那次 GEMM、把 logits 拷成 FP32 交给采样器的那次拷贝、采样器扫描过滤后的 logits 那一遍。下面是三处融合，正好一处一遍。

## 第一遍：和模型共用 LM head

pLLM 在 vLLM 的 `compute_logits` 里接管 LM head（[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)讲过为什么是这个边界）。需要稠密引导的时候，它把小网络预测的 hidden 拼到模型 hidden 后面，一起投影（`pllm/backends/vllm/session.py`）：

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
    else:
        predicted = lm_head(guidance.values.to(hidden.dtype))
        assert predicted is not None
    stages.append(SignedSamplerGuidance(guidance, predicted))
```

关键是 `torch.cat((hidden, guidance.values...))` 那一行：一次投影，前半是模型的 logits，后半是小网络的预测。（第一个分支里的 `BoundedSamplerGuidance` 是第三处融合，后面讲。）

你可能会担心：把 128 行拼成 256 行，cuBLAS 会换一种 kernel，模型自己那 128 行的 logits 会不会变？会，但只像“batch 变大”那样变。函数的 docstring 记着测量结果：在 Qwen2.5-0.5B 和一个 DeepSeek-V2-Lite 切片上，偏差从没超过原生 vLLM 自己对 batch 大小的敏感度（0.0625 和 0.5）。连续批处理下 batch 大小本来每步都在变，这点差别是原生系统里本来就有的。

这一招有它够不着的地方。DoLa 一类方法要读中间层的词表分布（`pllm.layer_logits`），中间层的 hidden 在前向还没结束时就要投影，没法和最后的 LM head 拼车，每读一层就要多读一遍权重。示例 06 在 7B 上（32 个并发请求 = 8 × n=4）读 9 层的 DoLa 只有原生的 66%，0.5B 上只有 25%。这条路径我们还没能用同样的办法融合。

## 第二遍：把加法折进 FP32 拷贝

`@pllm.add_logits()` 是一个更朴素的需求：给 logits 加一个偏置，比如 n-gram 先验、格式约束。朴素做法是插件自己发 kernel 去加，加几个就发几个，然后 vLLM 再做它自己的事。

vLLM 的 MRv2 runner 在采样前本来就要把 LM head 输出（BF16）拷成一份 FP32 交给采样器——这是一次必经的、扫过整个 `[rows, vocab]` 的遍历。既然它反正要读一遍、写一遍，为什么不把加法塞进这次拷贝？可惜 vLLM 的 temperature、过滤、拷贝这几个 launcher 都没有“加一个稠密张量”的入口，它的稀疏 logit bias 接口也表达不了 workload 算出来的张量。于是 pLLM 写了一个最小的 kernel，只替换那一次拷贝（`pllm/backends/vllm/sampler_epilogue.py`）：

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

值得停下来看的是倒数第二行：`(value + prediction).to(dtype).to(tl.float32)`。既然最后要写 FP32，为什么每加一次都先舍入回输入的 dtype（BF16）再转回来？因为语义要和不融合时一模一样：用户声明了两个 `add_logits`，原来的含义是“按顺序、每次在 logits 自己的 dtype 下加一次”。在 FP32 里一路加到底会更精确，但那就是另一个算法了——足以让某个刚好在过滤边界上的 token 换个命运。融合只该省时间，不该改结果。实测在 3 个 workload、各 98,304 个 token（共 294,912 个）上，0 个 token 不同。

包这个 kernel 的方式也有一个教训。最初用的是 PyTorch 推荐的 `wrap_triton`，后来换成了 `custom_op`，原因写在注释里：

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
    ...
```

`wrap_triton` 把张量元组当常量存进一张侧表，结果每一步的张量都被留住了；这个泄漏是在长时间测量里才暴露的。`custom_op` 只在调用期间持有这些张量。一个抽象边界选错，代价不是慢，而是慢慢地漏。

收益（RTX 4090，vLLM 0.27.1，0.5B，8 × n=16，256 token，3 轮中位数；示例 11 的 NGramPrior）：显存里的表 33,110 → 34,452 tok/s（+4.05%），主机内存里的表 30,786 → 32,273（+4.83%），只对部分行加（`rows=`）27,174 → 27,913（+2.72%）。单独的加法原来约 218 µs，融合后约 179 µs——而 vLLM 自己那次 FP32 拷贝本来就要约 111 µs；三次 launch 变成一次。

## 第三遍：只投影幸存者

最后是 `@pllm.guide_sampling` 本身，也是最有意思的一处。

先问一个问题：top-p 0.95 过滤之后，一行里到底剩下几个 token？在 ESamp 的实际生成里（0.5B，T 0.8，top-p 0.95），平均约 4.83 个；在 652,800 行里，最大的真实 nucleus 是 2,785。可稠密的做法为了这几个 token，要把小网络的预测投影到整个词表——0.5B 上每步 34.85 GFLOP，然后把其中十五万列扔掉。

所以第三处融合的想法很直接：**不投影词表，只投影幸存者**。一行剩下 5 个 token，就从 LM head 里取出这 5 行权重，和小网络预测的 hidden 各做一次点积，就地重加权、就地抽样。整个过程是一个 Triton kernel（`pllm/backends/vllm/guided_sampling.py`），每行一个 program。它先扫一遍已经过滤好的 logits，数出有限值，把幸存者压紧写进一块暂存区；然后只为幸存者取权重行：

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

`Weight + selected_tokens[:, None] * WS + hidden_axis[None, :]` 就是“只取幸存者那几行权重”。点积用 FP32 累加，再按原生 LM head 的 dtype 舍入一次——原生未量化的 LM head 就是这么舍入的。投影从“词表 × hidden”变成“幸存者 × hidden”，0.5B 上 `explain` 估计每步不超过 0.94 GFLOP（这是按容量算的上界，实际平均只有 5 行左右）。

抽样那一步（`_draw`）我们没有自己写。它调用的是 vLLM 自己的 `gumbel_block_argmax`，在 vLLM 0.27 和 0.32 两个签名之间分支。于是请求种子、位置、FP32/FP64 的噪声、贪心行为都和原生采样器一致——同一个种子，抽出同一个 token。自己写的 kernel 里调用上游的抽样原语，这种复用最省心：种子语义不需要我们去对齐，因为本来就是同一份代码。

kernel 里还有一个小技巧我们很喜欢。挑选幸存者时要比较“logit 谁大，一样大时 token id 谁小”。在 GPU 上维护一对 (float, int) 很笨，`_keys` 把它们编成一个可以直接比较的 uint64：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/guided_sampling.py
@triton.jit
def _keys(logits: tl.tensor, tokens: tl.tensor, finite: tl.tensor) -> tl.tensor:
    logits = tl.where(logits == 0, 0.0, logits)
    bits = logits.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(finite, (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - tokens.to(tl.uint32)), 0)
```

IEEE 754 的浮点数，正数翻转符号位、负数按位取反之后，当成无符号整数比大小，顺序和原来的浮点数完全一致（第一行把 −0 规范成 0）。把这个保序整数放在高 32 位，`0xFFFFFFFF - token` 放在低 32 位，于是“logit 降序、同分时 token 升序”变成了一次整数比较。决胜规则被写进了数的表示里。`_decode` 再把它拆回来。

还有一种行要单独处理。实测里，有 0.12% 的行整行都是有限值——上游没有对它们做过滤，看起来“幸存者是整个词表”。查下来，这些行的 top-1 概率在 0.99999994 以上，和第二名的 logit 差了 17 到 24：它们的真实 nucleus 就是一个 token。kernel 扫描时顺手维护最大值和概率质量，遇到这种行，就显式使用这个单 token 的 nucleus，而不是去投影整个词表。

## 容量原则

幸存者要先放进一块暂存区，而 GPU kernel 不能在运行时申请内存，所以暂存区的大小——`ExecutionOptions.guidance_capacity`，也就是 K——要提前定。超过 K 的行（罕见）怎么办？kernel 会选出 logit 最高的 K 个去引导，其余的当作“未引导的尾巴”参与抽样；这一段需要排序（每块合并进保留的前 K 个），比主路径贵得多。

一个精打细算的人会想：平均才 5 个幸存者，K=256 绰绰有余，越小越省。测出来恰恰相反。较早的一次 0.5B 测量（RTX 4090，128 并发，β=0.5，7 轮中位数）：K=256 是 33,737 tok/s，1024 是 34,069，**4096 是 34,742**，dense 是 32,624。K=256 比 4096 慢约 3%：更小的 K 不省什么，却把更多的行推进了需要排序的溢出路径。

所以默认值是 4096：比实测最大的真实 nucleus（2,785）还大，常见情况根本看不到溢出路径；代价只是每行 4096 个 uint64 的暂存（128 行约 4 MiB）。我们把这条叫作**容量原则：容量要选到让常见情况永远走不到复杂路径。**

在 7B 上（RTX 4090，vLLM 0.32，128 行，256 token，β=0.5，每个容量独立启动，3 轮中位数），差别小得多：

| guidance_capacity | tok/s | 相对原生（5,894） |
|---|---:|---:|
| 256 | 5,684 | 96.4% |
| 1024 | 5,684 | 96.4% |
| 4096（默认） | 5,678 | 96.3% |
| None（精确 dense） | 5,533 | 93.9% |

256 到 4096 的差别在噪声内；有界路径比 dense 快约 2.6%。一步 21.7 ms 里，采样器和 LM head 的占比本来就小。

数值上，初次实现时在 0.5B 上做过配对比较：有界路径和 dense 引导的抽样在 163,200 个引导行上全部一致，策略误差（每行的 TV 距离）的 p99.9 约为 1.5e-10。这是 ESamp 这一组设置下的测量，不是对任意小网络、任意 β 的保证；溢出时只引导前 K 个，是一个有意识的近似。要精确的 dense 行为，设 `guidance_capacity=None`。

## 走过的路，和别人比

旧 tLLM 在 2025 年走过几乎同样的路。第一个后端就是后过滤的精确实现，在一个很小的烟测设置下（0.5B，batch 4、n 4、16 token）吞吐比只有 0.6875；接着试过“前过滤 dense”——先改整个词表再过滤，快一些，但候选集合已经被引导改过，语义不对；又试过“后过滤 + dense 缓存”——预先把整个词表的预测算好、过滤后再挑，结论记在优化记录里：“full-vocab dense-cache 不自动更快”。7B 上还做过“在候选集合内部采样”，吞吐比从约 85.3% 提到约 89.7%（历史测量）。今天的有界 kernel 是这条路的终点：既不预先算整个词表，也不在 Python 里分组。

vLLM 自己的 FlashInfer 采样器已经把过滤和抽样融合了，但它没有“通过选中的 embedding 行做间接投影再抽样”这个入口。这是我们自己写这个 kernel 的理由——上游有过滤、有 Gumbel-max，我们就用上游的；上游没有的那一小段，才自己写。

## 做不到和更差的地方

有界路径的门槛不低：它要求 MRv2 runner、`tensor_parallel_size=1`、未量化且没有缩放或变换的 LM head。TP 下 LM head 按词表切到不同的卡上，量化权重也不是一行一行连续的 BF16，不满足条件时自动走精确的 dense 路径（这时第一遍的拼车仍然有效）。`explain` 给有界路径报的字节数是按容量 K 算的上界，可能比 dense 的估计还大，不能直接拿来预测吞吐。`add_logits(rows=...)` 的行映射每一步从 Python 列表上传一次，非阻塞，但有宿主开销。

还有一条是规则而不是性能：一个 workload 里最多只能有一个 `guide_sampling`。多个 `add_logits` 可以随便叠，因为加法可交换；可引导是过滤之后的非线性重加权，两个引导该怎么组合——先后？平均？——是算法问题，框架不替你猜（[设计取舍](../design-tradeoffs.md)）。

## 可以带走的东西

这篇里的三处融合，都来自同一个问题：**推理本来就要做哪几遍？** LM head 要读一遍权重，那就让小网络的行搭这一遍；采样前要拷一遍 logits，那就让加法搭这一遍；采样器要扫一遍过滤后的 logits，那就在这一遍里只为幸存者取权重、抽样。在带宽受限的地方，新增一遍遍历几乎总是比任何聪明的算术都贵。

另一个可以带走的是容量原则。固定容量的数据结构里，容量不是越小越省；容量的意义在于让常见情况永远看不到复杂路径。选一个比实测最坏情况还大的数，往往比选一个“够用”的数更快。
