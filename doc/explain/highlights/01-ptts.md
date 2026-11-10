# PTTS：给每个请求一块会“长版本”的显存

ESamp 是这个项目的起点。它的想法不复杂：每个请求在生成的同时，在线训练一个 rank-4 的小网络（我们叫它 distiller），用第 0 层的 hidden 预测最后一层的 hidden，采样时再用这个预测去调整 logits。在 pLLM 里，“每个请求一个小网络加一个优化器”只是一句声明：

```python
# check-docs: skip: excerpt from README.md
distiller = pllm.RequestState(
    init=pllm.FromModule(
        lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
    ),
    optimizer=pllm.AdamW(1e-3),
)
```

这句声明背后，是一个我们叫做 PTTS 的东西：Paged Tensor Transition System。先把名字说清楚：PTTS 开头的 P 是 Paged，分页、页表的意思，和操作系统的虚拟内存、vLLM 的 PagedAttention 同一个思想；它和 pLLM 名字里的 p（programmable）没有关系，只是碰巧都姓 p。

这篇文章想讲的是一个观察：在一个 serving 引擎里给每个请求维护会变的状态，会遇到好几个看起来毫不相干的问题——CUDA graph 的地址、训练和推理的重叠、生成中途导入参数、抢占和取消。我们一个一个去修，修到后来才发现它们其实是同一个问题，而答案是同一个抽象：**带版本的、分页的状态**。

## 四个看起来不相干的问题

第一个问题来自 CUDA graph。如果在 HF 的 `generate` 循环里写 ESamp，你大概会用一个字典：请求来了，`states[request_id] = deepcopy(template)`；结束了，`del`。但 vLLM 的 decode 步是录成 CUDA graph 回放的：录制时 GPU 驱动记下一串 kernel 和它们的参数，其中包括每个张量的地址；之后每步只是原样重放，Python 不参与（[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。回放能改的只是地址里的内容，不能改地址本身。字典里 128 个小网络散落在显存各处，这一步参与计算的是哪些请求每步都在变，录好的图找不到“第 3 行那个请求的参数”。2025 年的旧 tLLM 用固定槽解决了它——预先分配一块地址固定的显存，请求借一个槽、走时还回去，当时叫 model-bank：“把多个请求映射到固定的 bank slots。每个 slot 有自己的 distiller 参数。这样可以批量训练、复用固定 buffer，并配合 CUDA graph replay。”

第二个问题来自重叠。训练不能串在主计算流上，否则每一步都要多等一段训练。pLLM 把训练放到一条旁路的 CUDA stream 上（GPU 上的另一条任务队列，可以和主流同时跑）。于是第 n 步的训练在旁路流上改某个槽的时候，主流上第 n+1 步的前向正在读同一个槽，读到的是一半新一半旧的参数。

第三个问题来自外部导入。有的状态由用户在 `llm.generate(..., batch=StateImport(...))` 时给定，比如一个引导向量。直接覆盖那个槽，正在生成的老请求就会在一句话中间被换掉参数。

第四个问题来自抢占和取消。KV cache 不够时，vLLM 会把一个请求赶出去，稍后把它的 prompt 和已生成的 token 从头重算；用户也可能在一次更新写到一半时取消请求。前者需要知道“重算时该用哪一份参数”，后者需要回到上一个完整的状态——可原地写过之后，上一个状态已经不在了。

这四个问题，我们最初是当作四个 bug 分别修的。后来退一步才看清楚：它们问的其实是同一件事——**在某个时刻，某个读者应该看到状态的哪一份？** 图要知道这一步该读哪一份的地址；主流要一份不会被旁路流改动的；老请求要它开始时的那份，新请求要新导入的那份；抢占重算要准入时的那份，取消要上一份。只要状态是“原地改的一块内存”，这个问题就没有答案，因为任何时刻都只有一份。

## 一个抽象：不可变的版本

答案是不要原地写。每份状态（比如一个请求的 distiller）是一串**不可变的版本**：写者把新内容写到新的地方，写完了把它发布成下一个版本；读者在开始读的时候锁定一个版本，读完之前这个版本既不会被改，也不会被回收。

这在系统软件里是老办法，换个透镜就能认出来。数据库的 MVCC 就是这么做的：读事务看一个快照，写事务写新版本，提交时原子地切换，读写互不阻塞。Linux 的 RCU 也是：写者复制、修改、原子地发布新指针，旧版本要等一个“宽限期”——发布之前进来的读者都退出——之后才回收。GPU 上的宽限期有一个特殊之处：CPU 上的函数返回了，不代表 GPU 上排队的 kernel 跑完了。所以 PTTS 用 **CUDA event** 来界定宽限期：一块内存只有在所有读写过它的 GPU 工作都完成之后，才能重新分配。

“分页”是让这个抽象付得起的那一半。多个版本意味着同一个请求的参数会同时存在好几份，如果每次写都把整个小网络复制一遍，代价就正比于状态大小。PTTS 借用了 PagedAttention 的间接寻址：一份状态由若干字段组成（第一层权重、偏置、AdamW 的一阶矩、二阶矩……），每个字段有一个自己的池，是一块地址固定的大张量 `[capacity, *field_shape]`，第一维就是槽。**一个版本只是一组槽号**，每个字段一个。写的时候只为要写的字段分配新槽，没写的字段和旧版本共享同一个物理槽——字段级的写时复制。

举个小例子。请求 A 的 v7 是 W1→0 号槽、b1→0 号槽；旁路流要更新 W1，于是 v8 是 W1→3 号槽、b1 还是 0 号槽。主流读 v7，旁路流写 v8 的 W1，两边碰的不是同一块内存；v8 发布之后，等 v7 的读者都结束（event 完成），0 号槽的 W1 才回到空闲池。

页的大小是我们有意识的选择。vLLM 的 KV cache 每 16 个 token 一页，因为 KV 是一个 token 一个 token往后追加的；而现有的 workload 更新状态时都是整个字段一起改（优化器一步要动所有参数），细分只会多出簿记、换不来少拷贝。所以 PTTS 的“页”是一个字段的整块，与其说像 KV 的页，不如说像一组预分配的 buffer。代价在后面讲。

## 同一个抽象，回答四个问题

有了版本，前面四个问题都变成了同一个问题的不同查询。

先看 CUDA graph，它要的是固定地址。版本是一组槽号，而槽都在地址固定的池里，于是“这一步第 i 行读哪个版本”就变成了一张小表里的几个整数：每步开始前 CPU 把槽号填进一张地址固定的 int64 表，一次异步拷贝上传，图里的 kernel 先查表、再按槽号去池里取数。版本切换只是改表的内容，图一次也不用重录（表的细节见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。

重叠要的是读写分离。主流读一个已发布的旧版本，旁路流写一个私有的新版本，自然不冲突。这也解释了一个初看奇怪的默认值：`publish_lag=1`，训练结果晚一步才可见。第 n 步采样之后才能算 loss、反向、AdamW；如果要求第 n+1 步的前向立刻用上第 n 步的结果（`publish_lag=0`），主流就得停下来等训练，两条流退化成串行。允许晚一步——第 n 步的结果第 n+2 步才可见——第 n+1 步的主流就读第 n−1 步发布的版本，谁也不等谁；训练自己始终在最新版本上迭代，一步梯度也不丢。`pllm.explain` 会把这件事写出来：

```text
loss     train           token.commit / side / captured   distiller (latest, 0)
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

代价是要同时留几份版本。留几份？只有一行（`pllm/execution/ptts/page_bundle_planning.py`）：

```python
# check-docs: skip: excerpt from pllm/execution/ptts/page_bundle_planning.py
def version_capacity(lineages: int, publish_lag: int, *, exact_decode: bool = False) -> int:
    """Current/private roots, retained readers and history, plus two rollover slots.

    Exact in-forward transitions retain no post-sampler history. GPU completion
    fences govern reuse, so outstanding GPU work needs no further reserved roots.
    """
    return (2 if exact_decode else 3 + publish_lag) * lineages + 2
```

每一项都对得上一个真实存在的读者或写者：每个请求一份当前版本、一份正在写的私有版本、一份采样之后训练要读的历史，再加 `publish_lag` 份被滞后的读者锁住的旧版本，外加两个翻转用的槽。docstring 最后一句最值得体会：**还没完成的 GPU 工作不需要额外的份数**。因为复用由 event 管着，event 没完成，槽就不会被重新分配，规划时根本不必为“GPU 还没跑完”预留空间。`max_num_seqs=128`、`publish_lag=1` 时是 4 × 128 + 2 = 514 份；用文档里的 Distill 例子在 Qwen2.5-7B（hidden 3584）上算，每份 516,176 字节，约 253 MiB；`publish_lag=0` 时是 386 份，约 190 MiB。

外部导入要的是在途隔离。导入就是发布一个新版本；这次 `generate` 新准入的请求锁定新版本，正在生成的请求继续读它开始时锁定的那个。没有一行专门为“导入”写的同步代码。

抢占和取消要的是回到过去。取消一次还没发布的写，就是把那个私有版本扔掉——`abort`。抢占时则要分情况，这正是版本让判断成为可能的地方。如果状态只影响采样、或者只读 hidden（ESamp 的 distiller 就是），重算时模型算出的 K/V 和原来一样，状态应该留着，只是重算的 token 不能再被训练一遍——行标签器记着每个请求见过的最大位置，比它小的 token 标成重算（[《行模型》](04-packed-rows.md)）。如果状态影响了模型本身——改权重、改残差、改注意力——用新版本重算会得到不同的 K/V，这时就把状态倒回请求准入时的版本，从那里重放。`pllm/execution/preemption.py` 的 docstring 把规则写得很清楚，中间那段尤其有意思：

```python
# check-docs: skip: excerpt from pllm/execution/preemption.py
def preemption_policy(attachment: BackendAttachment, sites: tuple[ResidualRewrite, ...]) -> Literal["keep", "reset"]:
    """Model-state effects replay from admission; sampler and hidden-only effects retain state.

    Adapters and activation patches whose state no task rewrites while requests run (it is frozen, or imported
    before each request's first forward) also retain it: vLLM's recompute reads the version the request pinned, so
    it rebuilds the same K/V. A reset also needs the plan's recompute_rejection to be empty before replay begins.
    """
    ...
    return "reset" if applied & _rewritten(graph) or custom or attachment.probe.custom_attention or adaptive else "keep"
```

一个只在请求开始前导入、运行中没人改写的适配器，请求锁住的就是它准入时的版本，vLLM 重算时读到的还是这个版本，K/V 自然一样，连倒回都不需要。没有版本，这件事无从判断；有了版本，它只是比较两个版本号。

同一个抽象还顺带回答了一个我们一开始没打算回答的问题：同一块 GPU 上的另一个进程，能不能读这份状态？能——借用某个精确版本的页（通过 CUDA IPC），原来的锁要等所有远程读者确认之后才释放。读者是谁、在哪个进程，对版本来说都一样。

## 账本：Python 管对象，Rust 管数字

抽象讲完了，实现里有一个我们很满意的分工。Python 只持有 GPU 对象——字段池的张量、CUDA event；所有的数字事实——哪个槽被谁持有、每个槽分配到第几代、哪些槽在等哪个 event——都在 Rust 写的账本里，两边只传数组。`rust/pllm_core/src/page_ledger.rs` 开头的注释把这条边界写成了四行：

```rust
//! Single-owner PTTS admission, transition, exact-pin and publication bookkeeping.
//!
//! Python retains tensors and completion events. This ledger owns every numeric fact and
//! returns only array snapshots, copy descriptors and event IDs for the owner to enqueue.
```

账本本身是几个平坦的数组（`rust/pllm_core/src/slot_ledger.rs`），`holders` 是持有计数，`generations` 是分配代次（让过期的引用响亮地失败，而不是悄悄指向别人的槽），`fences` 记着“哪个 event 完成之前哪些槽不能复用”：

```rust
#[derive(Default)]
pub(crate) struct Pool {
    pub width: usize,
    pub capacity: usize,
    holders: Vec<i64>,
    generations: Vec<i64>,
    fences: Vec<(i64, Vec<bool>)>,
}
```

没有锁，因为整个 arena 只被跑模型步的那一个线程碰。发布和取消几乎是同一个调用（`pllm/execution/ptts/page_bundle.py`），只差账本调用的最后一个布尔参数：

```python
# check-docs: skip: excerpt from pllm/execution/ptts/page_bundle.py
def publish(self, token: PageWrite) -> PagePublication:
    self._issued_write(token)
    completion = self._record(token.stream)
    completed = self._events.completed()
    retained = self._ledger.publish(token.operation_id, self._events.add(completion), completed, False)
    self._events.retain(retained)
    return PagePublication(token.operation_id, token.before_refs, token.after_refs, completion)


def abort(self, token: PageWrite) -> None:
    """End logical ownership now; future physical reuse waits for the queued writes."""
    self._issued_write(token)
    completion = self._record(token.stream)
    completed = self._events.completed()
    retained = self._ledger.publish(token.operation_id, self._events.add(completion), completed, True)
    self._events.retain(retained)
```

这种相似不是偷懒。取消也必须在写者的 stream 上记一个 event：取消立刻结束**逻辑上**的所有权，可已经排进 GPU 队列的写还在跑，物理上的复用照样要等它们完成。“CPU 上取消了”和“GPU 上停下了”是两件事，很多并发 bug 就出在把它们当成一件事。

2026 年 10 月我们对 PTTS 做过一次复盘，把它每一步做的事情一条条列出来，问“这到底在保证什么”。答案只有四件：读者在使用期间看到一个固定的版本；写者不覆盖别人还在读的页；每份状态同一时刻只有一个写者；页只在所有 GPU 读写完成后才复用。复盘文档接下来那句话后来成了我们审代码的标尺：**凡是不服务于这四条的，都可以删掉；服务于它们但不必每步做的，都可以惰性或异步做。** 于是发布时复制整张版本字典，变成了按下标给一个数组赋值；每次都扫描所有收据的回收，变成了分配失败时才做一次；运行时的令牌合法性检查，大多改成了 `assert`；规划期 Rust 已经验过的字段合法性，运行时不再验。复盘时测的是：GPU 拷贝每步约 6 MB、十几微秒，CPU 簿记却要约 1.4 ms——瓶颈在记账，不在搬数据。同一次复盘还促成了“把拷贝融进更新”：AdamW 一步要整体覆盖参数和两个动量，分页 AdamW 就直接从源槽读、往目标槽写（`pllm/kernels/adamw.py`），五个优化器字段都省掉了写时复制。

## 版本的价钱

一个抽象好不好，也要看它收多少钱。

最明显的是显存。为了重叠要多留版本，状态小的时候无所谓：Qwen2.5-7B、128 并发、rank 4 时，README 里那个精简版 ESamp 的 distiller 一共 514 份 × 458,768 字节，约 236 MB。状态大的时候就疼了：In-Place TTT（一个 4B 的社区检查点）每个请求的快权重一份约 714 MiB，单请求要留 4 份、约 2.8 GiB，4 个请求要 10 份。这些显存都是从 KV cache 那里拿走的。

重叠到底换来了多少？和步长成反比。今天的测量（RTX 4090，vLLM 0.32，8 个 prompt × n=16 = 128 行，256 token，3 轮中位数）：0.5B 原生 42,355 tok/s、一步约 3.0 ms，ESamp β=0 在 lag=1 时是原生的 91.9%，lag=0 时是 89.8%；7B 原生 5,894 tok/s、一步约 21.7 ms，lag=1 是 97.4%，lag=0 是 97.2%，几乎没有差别。7B 的旁路训练每步约 616 µs（其中分页 AdamW 约 191 µs），放在 21.7 ms 的一步里，串不串行都不太要紧了。（这组数还有一段插曲：2026 年 10 月刚实现重叠时，0.5B 上测得 lag=0 → lag=1 是 91.4% → 94.3%，vLLM 0.27.1，3 轮中位数；后来 20 轮的交叉测量给出 92.45%。我们以 20 轮为准，但把两个数都留着——在共享的 GPU 上，三轮中位数不够稳。）

页太粗是第二个价钱。一页是一个字段的整块，某个算法如果每步只改大矩阵的一行，写时复制依然要分配一整个新槽、拷贝其余的行。子字段分块我们想过，没做；真遇到稀疏更新的算法，会有明显的写放大。

记账也不是零。在 0.5B、β=0 的宿主路径 profile 里（RTX 4090，vLLM 0.27.1，2026 年 10 月），`prepare_transition` 每次调用约 181 µs，`publish` 约 82 µs，`release` 约 58 µs。0.5B 一步 3 ms 左右，这几百微秒看得见；7B 一步 21.7 ms，就被盖住了。

还有几件做不到的事。容量在规划期定死，运行中不会再 `cudaMalloc` 扩张——那会破坏 CUDA graph 的固定地址——超了就是准入失败或窗口背压，不会平滑伸缩。张量并行时状态在每个 rank 上完整复制而不分片，因为分片意味着读写状态要跨卡通信，而我们选择了“pLLM 不发任何集合通信”；代价是 TP 从 1 提到 4，权重显存降到四分之一，状态显存一点没少。跨进程的借用是只读的，但这个“只读”是同一个可信程序内部的约定，不是操作系统的写保护。和 S-LoRA、Punica 这类多 LoRA 服务比，PTTS 也不是在同一个问题上竞争：它们服务成千上万个只读的 adapter，问题是显存怎么共享、计算怎么批量化；PTTS 的问题是状态在服务过程中会变。分页思想相通，语义不同。

## 一点可以带走的东西

回头看，PTTS 的大部分复杂度都来自两个事实：CUDA graph 只认地址，GPU 上的“完成”和 CPU 上的“返回”不是一回事。把这两件事想清楚，版本、分页、event 栅栏几乎是被推出来的。

更值得带走的是那个转折：当几个看似无关的 bug——地址、竞争、导入、回滚——反复出现时，先别急着一个一个修，问一句“它们是不是在问同一个问题”。对我们来说，那个问题是“谁在什么时候看到哪一份”，而一旦状态有了不可变的版本，这个问题就从一种并发控制变成了一种记账。记账可能很繁琐，但它是确定的、可以写成数组的、可以拿去复盘的。
