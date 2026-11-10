# 给推理引擎装一个“eBPF”：pLLM 为什么存在

这个系列有十二篇文章，讲的是 pLLM 里的几项技术各自是被什么真实约束逼出来的。这一篇是总起：先讲一个具体的问题，再退一步，说明我们认为它属于哪一类问题、用什么样的整体思路去回答，最后给出阅读的地图。

## 十几行代码，五堵墙

ESamp 是这个项目的起点。它的想法是：每个请求在生成的同时，在线训练一个 rank-4 的小网络，用第 0 层的 hidden 去预测最后一层的 hidden；采样时，把这个“浅层就能猜到”的预测从 logits 里减掉一部分，压低那些太容易被猜到的 token。在 HF 的 `generate` 里，这是十几行 PyTorch 加两个 forward hook。在 pLLM 里，它是 README 上的这几行：

```python
# check-docs: skip: excerpt from README.md
class ESampLite(pllm.Workload):
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    @pllm.guide_sampling(beta=0.5)
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)
```

可要把那十几行 HF 代码搬进 vLLM，会立刻撞上五堵墙。

第一堵是**连续批处理**。vLLM 每一步把几十上百个请求这一步要算的 token 首尾相接，拼成一张扁平的 `[tokens, hidden]` 张量。hook 拿到的就是这张张量，它不知道哪一行是谁的；而且行序由调度器决定，每步都在变，请求还可能被抢占、稍后从头重算。

第二堵是 **CUDA graph**。decode 一步只有几毫秒，CPU 一个一个地发射 kernel 跟不上 GPU，所以 vLLM 把整步录成 CUDA graph 回放。回放时 Python 不执行，hook 根本不会被调用。差别有多大：在 RTX 4090 上，原生 vLLM 0.32 跑 Qwen2.5-0.5B（8 个 prompt × n=16 = 128 行，256 token）时，关掉 graph 只剩开着时的 49%；7B 一步本来就要 21.7 ms，关掉也还有 98%。录制还有一条铁律：图里记的是指针，回放只认录制时的地址。

第三堵是**每请求的状态**。小网络的参数和 AdamW 的动量要随请求到来而分配、随请求结束而回收，还得在 CUDA graph 里地址固定；训练写参数时，下一步的前向可能正在读。

第四堵是**额外的计算拖慢吞吐**。训练串在主计算流上，每步多出几百微秒，而 0.5B 一步才 3 ms。想重叠，就得自己管 CUDA stream 和同步。

第五堵是**改注意力或 KV 要拆引擎**。注意力 kernel 和 KV 内存管理器紧紧咬合，想换个读法、换个存法，往往得 fork。

还有两条是背景约束。GPU 显存的分配顺序是：模型权重、CUDA graph 的显存池、扩展自己的状态，剩下的才归 KV cache；多占几百 MiB，并发就掉一截——有一次在 7B 上为了录更大的 graph，KV 能装的 token 从 76,144 掉到 3,872，吞吐只剩约 21%（历史测量）。再就是主机同步：一次 `.item()` 就让 CPU 停下来等 GPU 把队列跑空，而在这个项目里我们学到的是，同步少几次没有用，只有变成零才有用。

## 退一步：这是一个老问题

把这五堵墙放在一起看，它们没有一堵是算法问题。它们问的是同一件事：**怎样在一个你不拥有的、为极致性能而高度静态化的引擎里，安全地加一段有状态的用户程序？**

Linux 内核早就遇到过这个问题。想在网络栈里做包过滤、想追踪系统调用，最早的办法是写内核模块——一个空指针就能让整台机器挂掉；或者把数据拷到用户态处理，慢。eBPF 给出的答案，是一套很克制的四件套：**verifier** 在装载前证明程序不会搞坏内核；**JIT** 把通过验证的字节码编成本机码全速运行；**maps** 给程序提供跨调用、有生命周期的状态；**挂载点**（kprobe、tracepoint、XDP……）规定程序只能在哪些地方被触发。再加上 perf、bpftrace 这样的可观测性工具。

pLLM 的结构几乎是这四件套的镜像：

| eBPF | pLLM | 讲它的文章 |
|---|---|---|
| verifier | 声明 + Rust 规划 + CPU 上的契约检查（请求隔离、行置换、抢占重算、零同步） | [《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md) |
| JIT | 把方法的前向、反向、优化器录进 CUDA graph；挂载点的方法体用 Inductor 编译 | [《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)、[《图内挂载点》](05-in-graph-attach-points.md)、[《参数覆盖层》](06-parameter-overlays.md) |
| maps | PTTS：分页的、带版本的请求与实例状态 | [《PTTS》](01-ptts.md) |
| 挂载点 | 图内挂载点：残差、分支输出、投影、注意力、采样尾声 | [《图内挂载点》](05-in-graph-attach-points.md)、[《参数覆盖层》](06-parameter-overlays.md)、[《采样尾声融合》](07-sampler-epilogue.md)、[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md) |
| 可观测性 | `pllm.explain`（CPU 上的计划与成本账单）、trace、diagnostics | [《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md) |

（PTTS 是 Paged Tensor Transition System 的缩写。它的 P 指 Paged——分页、页表——和操作系统的虚拟内存、vLLM 的 PagedAttention 是同一个思想；和 pLLM 名字里的 p（programmable，可编程）没有关系，只是碰巧。）

类比也有边界，最好一开始就说清楚。eBPF 的 verifier 给的是**证明**；pLLM 对任意 PyTorch 给不出证明，它的契约检查是固定调度下的差分测试。eBPF 程序很小、指令集受限；pLLM 的程序可以在线训练一个网络、可以改写整层前向。eBPF 有稳定的 ABI；pLLM 依赖 vLLM 的内部结构，靠把两个支持版本（0.27.1 和 0.32.0）的差异关在一个兼容文件里过日子。

## 第二个统一的想法：一张小表

eBPF 的类比说的是“结构”。还有一个想法贯穿了几乎每一篇文章，说的是“机制”：**CPU 提前算好一张小表，GPU 按表执行。**

CUDA graph 只认录制时的地址，而连续批处理、请求的到来和离去、状态的版本，每一步都在变。这两件事怎么同时成立？答案是把所有会变的东西都变成一张地址固定的 int64 表里的数：这一步每一行属于哪个请求、读哪个版本的哪一页、写到哪一页、哪些行有效，甚至一个外部张量现在住在哪个地址。每一步，CPU 在锁页内存里填好这张表，一次异步拷贝上传；图里的 kernel 只认表的地址，表的内容每步换。`pllm/execution/tensor_executor.py` 里 `TableLayout` 的注释把这张表的四段写得很清楚：

```python
# check-docs: skip: excerpt from pllm/execution/tensor_executor.py
@dataclass(frozen=True, slots=True)
class TableLayout:
    """One resource's segments in the shared int64 table, uploaded in one copy per step.

    rows          [6, R]   per row: sampler row, sample index, state (S: padding sink), version, packed row, active
    sources       [F, S+1] source page slot per field and state (-1: none)
    destinations  [F, S]   destination page slot per field and state (-1: do not write)
    row inputs    [K, 3]   address and strides of each token or sampler tensor
    """
```

你会在[《行模型》](04-packed-rows.md)里看到这张表的 `rows` 段从哪里来，在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里看到它怎样让训练步录进 graph，在[《PTTS》](01-ptts.md)里看到 `sources` 和 `destinations` 怎样指向不同版本的页，在[《图内挂载点》](05-in-graph-attach-points.md)和[《参数覆盖层》](06-parameter-overlays.md)里看到一张表的**地址**本身被拿来当身份证。

从声明到执行，整条路径是这样的：带类型的声明 → 语义图（SemanticGraph）→ Rust 写的调度与表示决策 → 签名的物理计划 → Python 执行与 PTTS → 挂到 vLLM 上。今天 `pllm/` 下约 4 万行 Python，`rust/pllm_core/src` 下约 4,300 行 Rust。

这一切的前提是**声明代替 hook**。作者只在参数默认值上写标记——`pllm.hidden(0)`、`pllm.module("distiller")`——框架在 CPU 上用 meta 张量（只有形状、没有数据的张量）把方法追踪一遍，就知道它读什么、写什么、什么时候能跑、要多少显存。后面所有的自动化——放在哪个锚点、哪条 stream、留几份版本——都建立在这一次追踪上。

## 这不是什么

pLLM 不是一个算法库，也不替代 vLLM 已有的扩展点。vLLM 有 logits processor（只改采样前的 logits）、multi-LoRA（很多只读的 adapter 一起服务），也可以写自定义模型（要 fork 模型代码）。研究工具 TransformerLens、nnsight、pyvene 很好用，但它们面向单请求或静态 batch，没有连续批处理，也没有 CUDA graph。手改 vLLM 最快，代价是每个算法一份 fork，隔离和抢占的 bug 自己背。pLLM 想覆盖的是中间那一块：**有状态、在 graph 内、可组合**。

它也有明确做不到、或者做得更差的地方。最直接的是开销：同样的 RTX 4090、vLLM 0.32、128 行、256 token，ESamp 在 0.5B 上（一步约 3.0 ms）β=0 时只有原生的 91.9%，带引导时更低（早先 vLLM 0.27.1 上 20 轮测得 β=0.5 为 83.38%）；在 7B 上（一步约 21.7 ms）β=0 是 97.4%，β=0.5 是 96.3%。**这套东西在小模型上有 8%–17% 的可见开销；模型越大，相对开销越小，7B 上约 3%–4%。** 表的上传、旁路流的调度、几个零碎的 kernel，本身有几十到几百微秒的底噪；一步只有 3 ms 时，它们就看得见。

很多组合今天写不了：按注意力分数释放 KV（SnapKV、H2O）、接管外层解码循环（投机解码）、从头预训练——[设计取舍](../design-tradeoffs.md)一项项列了原因。依赖 vLLM 的内部结构，意味着每次升级都有维护成本。张量并行只支持一部分特性。写法上也有规矩：方法里不能 `.item()`，不能对张量值写 Python `if`，不能沿行归约；训练的结果默认晚一步才可见（`publish_lag=1`）。这些不是品味，每一条都能从上面那五堵墙推出来——这个系列的大半篇幅，就是在做这种推导。

我们的立场是：推理时的算法会越来越“脏”、越来越有状态——在线训练、记忆查表、改残差、改注意力、自适应深度。算法本身不是我们的工作；我们的工作是让它们在真实约束下跑得对、跑得快，并且在跑之前就能看见代价。

## 地图

推荐的阅读顺序如下，每篇一句话说它回答的那个具体问题：

1. [《行模型：在连续批处理里找回“每个请求”》](04-packed-rows.md)：扁平张量里的一行是谁的？为什么三个整数就够，以及当年我们在 vLLM 源码里怎么找到它们（附 vLLM attention metadata 的实测字段笔记）。
2. [《Python 只在录制时活着：固定地址、一张表、零同步》](03-whole-algorithm-in-cuda-graphs.md)：怎样把一整步在线训练录进 CUDA graph；为什么 hook 会在回放时消失；为什么编译训练步反而更慢。
3. [《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)：地址、重叠、导入、抢占这四个问题，为什么都由“带版本的分页状态”回答。
4. [《图内挂载点：在 CUDA graph 里改残差》](05-in-graph-attach-points.md)：残差加法藏在 norm 里，怎样在那里挂载；一个什么也不做的改写，怎样从 25% 追到 99.5%。
5. [《参数覆盖层：顺着 PyTorch 的抽象层往下走》](06-parameter-overlays.md)：每请求的权重增量和前向内训练，怎样由 vmap、grad、compile、make_fx、custom_op 各管一件事地叠起来。
6. [《自定义层循环与状态传播》](10-custom-layer-loops.md)：跳层和循环深度里，被跳过的层的 KV 怎么交代；为什么跳过 23% 的 token 不会更快。
7. [《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)：哪些布局能真的省显存、哪些只能少读，以及 SnapKV 为什么做不了。
8. [《复用上游：kernel、管理器与原语》](09-reusing-upstream-kernels.md)：一个“更对”的复用让吞吐掉了两成之后，我们学到的“像上游自己那样调用上游”。
9. [《采样尾声融合》](07-sampler-epilogue.md)：不新增任何一遍词表遍历，把采样引导塞进推理本来就有的那几遍。
10. [《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)：大表放主机内存时，为什么把线程块砍到四分之一反而更快。
11. [《自动放置与重叠》](11-automatic-placement-overlap.md)：任务放在时间线的哪里、哪条 stream；为什么重叠是零和的；以及我们为什么删掉了自己引入的多面体编译器。
12. [《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)：回到 verifier：证明不了的东西，怎样用“改掉别人、看看自己”去测。

如果你只关心某一件事：被“哪一行是谁的”困扰，读[《行模型》](04-packed-rows.md)；hook 在 CUDA graph 里不生效，读[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)；要给每个请求维护会变的参数，读[《PTTS》](01-ptts.md)和[《参数覆盖层》](06-parameter-overlays.md)；要改模型内部，读[《图内挂载点》](05-in-graph-attach-points.md)、[《自定义层循环与状态传播》](10-custom-layer-loops.md)、[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)；关心吞吐，读[《采样尾声融合》](07-sampler-epilogue.md)、[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)、[《自动放置与重叠》](11-automatic-placement-overlap.md)；想知道一个系统怎样做减法，读[《复用上游》](09-reusing-upstream-kernels.md)和[《自动放置与重叠》](11-automatic-placement-overlap.md)。

最后一句：性能总是想把系统变得静态——固定的地址、固定的形状、录好的图；算法总是想要动态——每个请求不同、每一步在变、随时学习。这个系列里的每一项技术，说到底都是在这两者之间找一个接缝：让会变的东西变成数据，让不变的东西保持不变。
