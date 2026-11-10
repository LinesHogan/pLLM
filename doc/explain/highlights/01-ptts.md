# PTTS：给每个请求一块会“长版本”的显存

现代大模型推理引擎之所以能把吞吐推到极致，很大程度上仰赖于一个隐含的前提：除了不断向后追加的 KV Cache 之外，模型本身对每个请求是无状态的。成百上千个并发请求共用同一份只读的模型权重，每算完一步就丢掉中间激活，彼此之间没有任何可变数据的牵连。

但只要你试着往推理过程里引入一点跨步演化的算法——比如在生成每个 token 的同时，在线为当前请求训练一个轻量级的预测网络来引导采样（如在线蒸馏引导 ESamp），或者随着上下文推进实时更新每个请求专属的快权重（如 Test-Time Training），又或者在生成中途动态注入外部的引导向量——这个平静的无状态假设就瓦解了。你突然需要在高速运转的推理引擎内部，为上百个并发请求维护可读可写、每一步都在变化的状态张量。

在单请求的研究脚本里，这根本算不上一个问题。你顺手建一个 Python 字典 `states[req_id] = copy.deepcopy(template)`，再绑一个 `torch.optim.AdamW`，每步直接在原地更新权重（In-place Mutation），请求结束时 `del states[req_id]` 就完事了。可一旦把这个字典放进一台真正在线上跑满的推理引擎，你会接连撞上一串棘手的系统故障。

最先找上门的是 CUDA Graph。为了省掉 CPU 逐个发射算子的微秒级开销，生产引擎会把解码步录制成静态的计算图反复回放（见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。图在录制那一刻就把所有算子读写的显存指针焊死了，回放时只能修改指针指向的内存内容，不能修改指针本身。而字典里动态分配的请求张量散落在显存的各个角落，每一步哪些请求在批里、排在第几行都在变化，录好的图根本找不到当前这一行对应的权重。

要绕过 CUDA Graph 的地址限制，最直觉的做法是预先划出一块容量固定、地址永远不变的静态槽数组（Slot Bank）：每个槽位存放一份网络参数，请求进门时借一个槽位，生成结束时还回去。如果整个系统只在一条流上串行执行，静态槽数组确实够用。但在线训练包含反向传播与优化器更新，如果串行堵在模型前向的主计算流（`main` stream）上，每一步都要白白多等几百微秒。自然的解法是把训练放到一条旁路流（`side` stream）上，让第 $n$ 步的权重更新与第 $n+1$ 步的模型前向在 GPU 上并行执行。然而，只要你还停留在“原地修改同一个槽位”的模型里，双流并发就会立刻引发读写撕裂：当旁路流上的 `AdamW` 正在往 3 号槽位写第 $n$ 步的新权重和一阶、二阶动量时，主流上的第 $n+1$ 步前向已经冲进来读同一个 3 号槽位了。两边没有任何锁，主流读出来的是一半刚被新梯度冲掉、一半还留着上一步残骸的撕裂参数。

即便状态不是由旁路流在线训练出来的，而是由外部调用者在发起请求时动态推入的共享状态（比如激活引导里的方向向量 `llm.generate(..., batch=pllm.io.StateImport(...))`），原地修改同样会闯祸。假设老请求 A 的句子才生成到一半，新请求 B 带着一个新的引导向量进场并直接覆盖了那个固定槽位，正在说话的请求 A 就会在半句话中间被换掉参数，生成出前后割裂的文本。

更不用说在线服务里随时可能发生的取消与 KV Cache 抢占（Preemption）。当显存吃紧时，vLLM 的调度器会把某个正在生成的请求临时踢出显存，等稍后腾出空间，再把它的 prompt 连同已经生成的 token 从第 0 步开始重新送进模型做一遍预填充（prefill），把 KV Cache 重建出来。如果状态是在原地直接覆盖的，当一个请求在第 50 步被抢占、不得不退回第 0 步重算时，它第 0 步时的初始状态早就已经被覆盖得无影无踪了；如果拿着第 50 步训练出来的权重去重算第 0 步的 K 和 V，算出来的注意力与被踢走前完全对不上。

如果把这四件事当成四个孤立的工程问题来修——给图录制预留静态槽，给双流并发做双缓冲，给参数导入排等待队列，给抢占设计状态备份——你会发现每加一个补丁，它就会和另外三个补丁打架。

---

## 一、 把时间维度找回来：不可变版本与字段级分页

当一个系统里三四个看似无关的并发与恢复问题缠在一起时，往往意味着底层的状态抽象丢掉了某个关键维度。让我们停下来退后一步，把具体场景的细节剥掉，看看这四起故障到底在争抢什么：

图录制问的是，怎么用一个不变的物理基地址，去指代运行中不断变化的当前状态；双流并发问的是，当写者正在制造下一份状态时，读者怎么安全地读取上一份状态；在途导入问的是，当新请求换上新一份状态时，老请求怎么继续持有它进门时的那一份；而抢占与取消问的则是，当操作作废或时间倒流时，怎么退回更新前的那一份或准入时的初始状态。

这四个问题其实在问完全相同的一句话：**在时间线上的任意一刻，某一个读者究竟应该看见状态的哪一份快照？**

只要状态模型是“在原地修改同一块显存”，这个问题在数学上就无解。因为原地修改抹杀了时间维度——过去、现在和未提交的未来被粗暴地拍扁在了同一个物理地址里。同一时刻永远只有一份数据，读者和写者、老请求和新请求、正常推进和回滚重算，全都被迫在同一个内存格子里肉搏。

一旦看清这一点，解法根本不需要重新发明。数据库里的多版本并发控制（MVCC）与操作系统内核里的读-拷贝-更新（RCU），早就用同一个基础原语把这个问题彻底解决了。它的核心规则只有一句话：状态永远是不可变的（Immutable）。

在 pLLM 中，当你写下这样一句声明，要求引擎为每个并发请求维护一份在线训练的预测器状态时：

```python
# check-docs: skip: excerpt from README.md
distiller = pllm.RequestState(
    init=pllm.FromModule(
        lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
    ),
    optimizer=pllm.AdamW(1e-3),
)
```

接管这句声明的，是底层一套称为 **PTTS（Paged Tensor Transition System，分页张量状态迁移系统）** 的状态引擎——注意它开头的 P 是 Paged（分页与页表，与操作系统的虚拟内存、vLLM 的 PagedAttention 同源），和 pLLM 名字里代表 programmable 的 p 没有任何关系，只是碰巧撞了首字母。

在 PTTS 里，一份逻辑状态（比如某个请求的 `distiller`，我们称之为一条状态谱系 `Lineage`）不是一块会被反复涂改的内存，而是一串单调递增的不可变版本（Version）。写者绝不触碰任何已经发布的版本；每次更新都是一次状态迁移（Transition）——写者申请新的物理槽位，把新参数写进去，写完后原子地发布为下一个版本号 `StateRef(LayoutId, LineageId, VersionId)`；读者在执行窗口开始时钉住（`pin_exact`）某个已发布的版本，在它读完放手之前，这个版本对应的物理显存既不会被任何人修改，也绝不会被回收。

整套状态引擎只向系统承诺四条硬保证：

1. 读者在使用期间，看到的永远是一个固定不变的版本；
2. 写者绝不覆盖任何仍被读者持有的页；
3. 每个状态谱系在同一时刻最多只有一个活跃写者；
4. 一个物理页只有在触碰过它的所有 GPU 读写工作全部完成后，才能被复用。

但在 GPU 上落地这套多版本机制，会立刻撞上两个独特的物理约束：第一，一份状态可能包含多个张量字段，如果每次只改其中一个字段却要把整份状态复制一遍，显存带宽怎么付得起？第二，GPU 是异步执行的，CPU 上的 Python 函数返回了，只代表内核被塞进了硬件队列，远不代表 GPU 真正跑完了读写，RCU 里的“宽限期（Grace Period）”在显卡上怎么界定？

这就引出了 PTTS 的两个核心机制：字段级写时复制（Field-level Copy-on-Write）与 CUDA Event 回收栅栏。

```text
 Version Roots                    Physical Slot Pools (Fixed Base Addresses)
┌─────────────┐                  ┌──────┬──────┬──────┬──────┬──────┐
│ Req A   v7  │──┬──────────────►│Slot 0│Slot 1│Slot 2│Slot 3│ ...  │  Field 0: W1 [cap, 4, H]
└─────────────┘  │   ┌──────────►│ (v7) │      │      │ (v8) │      │
                 │   │           └──────┴──────┴──────┴──────┴──────┘
┌─────────────┐  │   │           ┌──────┬──────┬──────┬──────┬──────┐
│ Req A   v8  │──┼───┘   ┌──────►│Slot 0│Slot 1│Slot 2│Slot 3│ ...  │  Field 1: b1 [cap, 4]
└─────────────┘  └───────┴──────►│(v7/8)│      │      │      │      │
  (W1 updated, b1 shared)        └──────┴──────┴──────┴──────┴──────┘
```

如上图所示，一份状态由若干个字段（Field）组成——例如一份带 `AdamW` 的状态，包含展平的参数 `parameters`、梯度 `gradient`、步数 `step`、一阶矩 `exp_avg`、二阶矩 `exp_avg_sq` 五个字段；图里画的是一份有 `W1`、`b1` 两个字段的状态。在引擎启动的规划期，PTTS 为每个字段预先分配好一块基地址永远不变的连续大张量，形状是 `[capacity, *field_shape]`。这就是该字段的物理槽池（Slot Pool），第 0 维的下标就是槽号（Slot ID）。

在这个结构下，一个“版本（Version Root）”在物理上仅仅是一组整数槽号——每个字段一个整数。当请求 A 的状态处于版本 `v7` 时，它的 `W1` 在 0 号槽，`b1` 也在 0 号槽。如果接下来的任务只写 `W1`、不改 `b1`，在准备这次状态迁移（`prepare_transition`）时，PTTS 只为被修改的 `W1` 分配一个新的私有空闲槽（如 3 号槽），而未修改的 `b1` 直接沿用 `v7` 的 0 号槽。发布一个新版本的显存与拷贝成本，只正比于真正被修改的字段，而不是整份状态的大小（在 `handle.diagnostics().state_usage` 中，`logical_live_bytes` 按所有存活版本累加，而 `allocated_payload_bytes` 对共享的物理槽只算一次，两者的差值就是字段级共享省下的物理显存）。

这里在“页（Page）”的粒度上有一个有意识的取舍。vLLM 的 PagedAttention 是每 16 个 token 一页，因为 KV Cache 是随着生成逐个 token 往后生长的；而在 PTTS 里，一页就是一个字段的完整张量。之所以不在字段内部进一步切分小块，是因为无论是优化器步进还是激活更新，现有的推理期算法几乎都是整字段一起读写。把一个 `[4, 3584]` 的小矩阵再切碎，除了增加页表寻址负担，换不来实际的拷贝节省。从物理形态上看，它更接近一组带引用计数与版本号的预分配缓冲池。

至于 GPU 上的异步宽限期，则由 `torch.cuda.Event` 把关。当一个旧版本被新版本取代、或者一个读者调用 `release_pin` 释放它钉住的版本时，对应的物理槽位哪怕持有者计数（`holders`）归了零，也不能立刻分给别人覆盖——因为刚才那个读者或写者在 CUDA 流上发射的内核可能还在硬件队列里排队。PTTS 会在那条流上记录一个完成事件（Fence）。宿主端轮询发现该事件已经完成（`event.query() == True`），证明所有触碰过那块显存的 GPU 线程都已退场，槽位才直接回到就绪池；如果在极高负载下就绪槽位用尽，分配器也会把持有计数已归零、但仍挂着未完成事件的槽位先分出去，同时返回该事件 ID，强制接手的新流在 GPU 端先执行 `stream.wait_event`——保证旧的 GPU 读写彻底撤离之后，新的写入才会真正开始。

---

## 二、 四个难题如何退化为自然推论

检验一个底层原语是否切中要害，最好的标准不是看它本身有多复杂，而是看当初把你困住的那几个问题，在它面前是不是全都退化成了无需特判的推论。

回到最开始的第一个问题：CUDA Graph 怎么读写每一步都在变化的请求状态？既然所有字段的物理槽池都是在启动时一次性分配好的固定大张量，基地址从生到死不变，而一个版本又只是一串整数槽号，那么“当前批次里第 $i$ 行请求该读写哪份状态”，就彻底简化成了整数查表。每一步开始前，CPU 从账本里取出当前各行对应的源槽号与目标槽号，填进那张地址同样固定的共享 `int64` 索引表（即[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)中的 `TableLayout`），一次异步拷贝上传到 GPU；录在图里的内核先从表里读出整数 `slot`，再去池子里按 `pool[slot]` 寻址。甚至对于 `AdamW` 这样要同时更新参数、梯度、步数和一阶、二阶动量五个字段的操作，分页优化器内核（`pllm/kernels/adamw.py`）直接从旧版本的源槽号读取旧参数和动量，在寄存器里算完一步更新后，直接写进新版本的目标槽号——连写时复制的事先拷贝都一并省掉了。

再看第二个问题：双流并发下的读写竞争与可见性边界。有了不可变版本，主流上的前向读者钉住已发布的旧版本，旁路流上的训练写者往新分配的私有槽位里写新版本，两边在物理上触碰的根本不是同一块显存，读写撕裂自然消失了。但这里引出了一个更本质的流水线问题：旁路流写出来的新版本，主流什么时候才能看见？

如果你在 CPU 上运行 `pllm.explain` 查看在线蒸馏负载的执行计划，会看到这样三行输出：

```text
loss  train  token.commit / side / captured  distiller (latest, 0); one AdamW step per state per engine step; prompt rows train once per prefill chunk
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

注意 `predict` 后面的 `(boundary, 1)` 与默认配置 `publish_lag=1`：第 $n$ 步训练产生的新状态，默认要到第 $n+2$ 步才会被主流上的 `predict` 读到。

为什么不让第 $n+1$ 步立刻用上第 $n$ 步刚训出来的权重？把两条流的时间线画出来，物理约束就一目了然了：

```text
Main Stream:  [  Step n Forward & Sample  ] ──► [ Step n+1 Forward (reads v_{n-1}) ] ──► [ Step n+2 Forward (reads v_n) ]
                                           \                                            ▲
Side Stream:                                └──► [ Step n Train (reads v_{n-1} -> writes v_n) ]

Active Roots per Lineage during Step n+1 (publish_lag = 1, total = 4 per lineage):
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ 1. Current   │   │ 2. Private   │   │ 3. Retained  │   │ 4. History   │
  │ Published    │   │ Writing (v_n)│   │ Lag Reader   │   │ Post-Sampler │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

第 $n$ 步的训练任务 `loss` 必须等第 $n$ 步的采样完成（`token.commit`，临近整步末尾）之后才能在旁路流上启动。而几乎在同一微秒，主流已经跨过步边界，开始跑第 $n+1$ 步的模型前向和 `predict` 了。如果你要求 `publish_lag=0`——即第 $n+1$ 步的前向必须读到第 $n$ 步训练出的 `v_n`——主流在进入第 $n+1$ 步时就必须停下来，干等旁路流把反向传播和 `AdamW` 全部跑完，双流流水线当场退化成串行等待。

反过来，一旦允许 1 步的可见性滞后（`publish_lag=1`），当第 $n$ 步的训练还在旁路流上写 `v_n` 时，第 $n+1$ 步的主流根本不用等它，直接钉住第 $n-1$ 步早已发布好的 `v_{n-1}` 往前跑；等第 $n$ 步的训练在后台跑完并发布 `v_n`，第 $n+2$ 步的主流自然就能读到 `v_n`。与此同时，训练任务自己声明的是 `(latest, 0)`——第 $n+1$ 步的训练在旁路流上紧接在第 $n$ 步训练之后，直接在最新的 `v_n` 上继续迭代，一步梯度都不会丢（如果某个算法在数学上严格要求下一步立即可见，也可以显式声明 `publish_lag=0`，主动用吞吐换取零滞后）。

用一两步的可见性延迟换取双流并发，代价是同一时刻显存池里必须同时容纳多个存活版本。到底要预留多少个版本槽位，才能保证满负荷流水线绝不因为槽位耗尽而卡死？在 `pllm/execution/ptts/page_bundle_planning.py` 中，这个容量规划是一行精确的算术公式：

```python
# check-docs: skip: excerpt from pllm/execution/ptts/page_bundle_planning.py
def version_capacity(lineages: int, publish_lag: int, *, exact_decode: bool = False) -> int:
    """Current/private roots, retained readers and history, plus two rollover slots.

    Exact in-forward transitions retain no post-sampler history. GPU completion
    fences govern reuse, so outstanding GPU work needs no further reserved roots.
    """
    return (2 if exact_decode else 3 + publish_lag) * lineages + 2
```

公式里的每一项都对应着时间线上的一个具体角色：每个并发状态谱系（`lineages`）需要 `1` 份当前已发布根、`1` 份旁路流正在写入的私有根、`1` 份从主前向结束保留到采样后移交给训练任务的历史根，外加 `publish_lag` 份被滞后读者钉住的旧版本，最后加上 `2` 个用于步间指针切换周转的翻转槽（如果是像 In-Place TTT 那样在前向传播内部当场完成读写、不留采样后历史的精确更新 `exact_decode=True`，则收敛为 `2 * lineages + 2`；不可变的全局 `InstanceState` 则只需 `1` 份）。

值得注意的是 docstring 里的第二句话：**未完成的 GPU 工作不需要额外预留根。** 因为只要某个旧版本还被读者持有，它就已经算在 `(3 + publish_lag)` 的名额里了；而一旦持有者释放、进入等待 Event 完成的退役状态，分配器随时可以在必要时通过挂接 `stream.wait_event` 原地复用该槽位。逻辑名额与物理复用完全闭环。

至于第三个问题——外部参数导入与跨进程读取——在版本模型下甚至不需要任何专门的同步分支。当外部通过 `StateImport` 推入一份新的引导向量时，它只是在状态谱系上发布了一个新版本 `v_new`。新准入的请求在进门时调用 `pin_exact` 锁住 `v_new`；而更早就已经进场、正说到半句话的老请求，手里钉住的依然是它进门时的 `v_old`（除非用户显式声明 `refresh="step"` 要求每步换绑最新帧）。同理，通过 `RequestStateDirectory` 的 `key_fields` 配置，“每个请求独占一份状态”、“同一个 prompt 裂变出的 $n$ 条采样轨迹共享一份状态”与“按路由键跨请求保留状态”这三种作用域，底层走的也是完全同一套谱系映射表。甚至当状态由同一张 GPU 上的另一个操作系统进程写出、引擎这边零拷贝读取时（如独立解码视频帧、提取特征的生产者进程），也只是通过 CUDA IPC 借出某个精确版本的只读视图：读取方不过是又一个握着 `PagePin` 的读者，生产者那边的物理槽位会一直保留到本地 Event 完成且读取方发回完成凭证（ACK）之后才被回收。

最后看第四个问题：操作取消与 KV Cache 抢占重算。如果写者在更新中途报错或被取消，只需调用 `abort` 丢弃尚未发布的私有槽位，已发布的当前版本毫发无损。而当请求生成到第 50 步遭遇 vLLM 抢占驱逐、稍后从第 0 个 token 重新 prefill 时，它的状态究竟该保留在第 50 步的最新版本（`keep`），还是倒回准入时的初始版本重头再来（`reset`）？

这取决于一条清晰的物理分界线：**这份状态到底有没有改动模型算出来的 K 和 V？** 在 `pllm/execution/preemption.py` 中，判定逻辑只有短短几行：

```python
# check-docs: skip: excerpt from pllm/execution/preemption.py
def preemption_policy(attachment: BackendAttachment, sites: tuple[ResidualRewrite, ...]) -> Literal["keep", "reset"]:
    # ……
    graph = attachment.plan.semantic_graph
    adaptive = any(task.custom_forward is not None and task.custom_forward.adaptive for task in graph.tasks)
    applied = {
        binding.resource for binding in (*attachment.spec.overlay_bindings, *attachment.spec.activation_bindings)
    }
    custom = any(not rewrite.patch for rewrite in sites)
    return "reset" if applied & _rewritten(graph) or custom or attachment.probe.custom_attention or adaptive else "keep"
```

如果状态只读隐藏层、或者只影响最后的采样器（例如 ESamp 的 `distiller`），它从头到尾没有改动过 Transformer 内部的任何激活或权重。当 vLLM 把前 50 个 token 重新送进模型重建 KV Cache 时，模型算出来的 K 和 V 与被踢走前一模一样。因此，`distiller` 已经训练了 50 步的权重完全应该保留（`keep`），只需由行标签器记住每个请求已经处理过的最大位置游标（`frontier`），把重算的旧 token 过滤掉、不送进训练器重复训练即可（详见[《行模型》](04-packed-rows.md)）。

反之，如果状态直接参与了模型前向计算（如参数覆盖层 Overlay、激活补丁），且在请求运行期间会被内部任务改写（属于 `applied & _rewritten(graph)`），那么第 10 步的 K/V 是用 `v10` 算的，第 20 步是用 `v20` 算的。（自定义残差改写、自定义注意力和自适应跳层直接改写了前向结构，不论状态是否在运行期被改写，一律按这种情况处理。）拿着第 50 步的 `v50` 去重算第 0 步的 K/V 就会篡改历史。此时策略自动定为 `reset`：调用 `rewind_versions` 把该请求的状态指针拨回准入时锁定的初始版本，让状态跟着重算的 token 重新演进。而最微妙的交集在于：如果一个激活补丁或适配器虽然改了模型前向，但它在请求运行期间是冻结的（或者是在请求第一次前向之前通过 `StateImport` 导入的，不在 `_rewritten(graph)` 集合里），策略依然是 `keep`——因为请求从准入起钉住的就是同一个不可变版本，重算时读着它自然能重建出分毫不差的 K/V。

---

## 三、 账本的工程边界：Python 管设备对象，Rust 管数字事实

把“不可变版本 + 字段级槽池 + Event 栅栏”想清楚之后，工程实现上的核心问题浮出水面：在每步只有几毫秒的解码循环里，状态引擎要频繁执行准入分配、写时复制配对、版本发布、精确钉住以及 Event 回收，怎么让这套记账逻辑本身足够快，而且结构上不出错？

如果在热路径上用 Python 字典维护版本映射、每次发布都拷贝引用集合、每次操作都遍历清理所有收据，CPU 在宿主端折腾字典与对象的开销甚至会超过 GPU 在显存里真正拷贝状态页的时间。要消灭这笔记账税，需要立下两条工程纪律。

第一条纪律，是沿着前面那“四条硬保证”做减法：**凡是不服务于这四条保证的簿记动作，统统不做；凡是服务于这四条保证、但不必每一步都做的动作，统统改为按需惰性执行。**

例如，只要池子里的空闲槽位还够用，已完成的释放收据留在队列里根本不妨碍任何一条保证。因此在 `page_bundle.py` 的 `prepare_transition` 里，平时根本不主动扫描回收收据，只有当底层账本抛出 `_core.PageCapacity` 槽位不足异常时，才调用一次 `self.collect_completed()` 清理已完成的栅栏并重试。同样，字段 ID 是否合法、状态布局是否匹配这些静态事实，在启动前的规划期早就校验完毕了，运行时热路径上不再做任何防御性的重复检查。

第二条纪律，是在 Python 和 Rust 之间划定清晰的职责边界：**Python 只保管 GPU 设备对象（字段池的大张量与 `torch.cuda.Event` 实例），而所有的数字事实（哪个槽位被谁持有、每个槽位分配到了第几代、哪些槽位正挡在哪个 Event 栅栏后面）全部收归单线程的 Rust 账本（`rust/pllm_core/src/page_ledger.rs` 与 `slot_ledger.rs`），跨语言边界只传递扁平的 `int64` 数组、拷贝描述符和事件编号。**

看看 `slot_ledger.rs` 里管理一个物理池的 `Pool` 结构：

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

这里没有任何互斥锁（Mutex）——因为整个状态 Arena 严格归执行模型步的那一个宿主线程单线程拥有。`holders` 记录每个槽位的当前持有计数；`fences` 记录某个 Event ID 完成之前哪些槽位不可无条件复用；而 `generations`（分配代次）则是防止野指针别名的安全阀：每当一个槽位的持有计数归零，`release` 就把它的代次加一。如果上层代码因为 bug 拿着一个已经释放的旧 `PageSlot` 引用回来取张量，视图检查一比对 `self.generations[index] == generation` 不匹配，当场就会触发断言失败，绝不会悄无声息地读到已经分给另一个请求的脏数据。

在这层纯数字账本之上，Python 侧的 `PageBundleArena` 薄得只剩下设备动作入队。以 `pllm/execution/ptts/page_bundle.py` 中的 `publish`（发布新版本）和 `abort`（中止私有写）为例：

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

这两段代码几乎一模一样，只差传给 `self._ledger.publish` 的最后一个布尔参数（`False` 代表发布，`True` 代表中止）。为什么连取消一次写入（`abort`）都必须在 `token.stream` 上记录一个完成事件 `completion` 并交给账本立栅栏？

因为**“CPU 上决定取消”和“GPU 上已经停下”是两码事**。当 Python 调用 `abort` 的那一刻，这次状态迁移在逻辑上的所有权确实立即终结了，那个私有版本永远不会变成可见版本；但在调用 `abort` 之前，写者可能已经往 `token.stream` 的硬件队列里发射了几个写内存的 CUDA 内核，它们此刻正在 GPU 上飞奔。如果 `abort` 不打下 Event 栅栏就把私有槽位直接放回空闲池，下一微秒另一个流上的新请求分到同一个槽位，就会和尚未跑完的废弃写指令迎头相撞。因此，`abort` 和 `publish` 走的是完全相同的物理栅栏回收路径：逻辑上立刻作废，物理上老老实实等 GPU 队列排空。

在单张 NVIDIA RTX 4090 上对 Qwen2.5-0.5B（vLLM 0.27.1，128 并发，ESamp $\beta=0$）做的宿主端剖析显示，这套数组化账本每次调用的中位耗时约为：`prepare_transition` 181 µs，`publish` 82 µs，`release` 58 µs。放到一步耗时 21.7 ms 的 7B 模型上，这点 CPU 记账时间相对整步就很小了。

---

## 四、 物理定律的账单：在哪里会疼？

在系统设计里，没有任何抽象是免费的。当我们选择用“预分配池 + 字段级写时复制 + 多版本驻留”来换取无锁并发和 CUDA Graph 兼容时，代价体现在三个非常具体的地方。

第一笔账单是**显存容量直接与 KV Cache 竞争**。在 vLLM 中，显卡显存先装模型权重，再划出 CUDA Graph 捕获池和 pLLM 的状态池，最后剩下的空间才全部分给 KV Cache。而根据 `(3 + publish_lag) * lineages + 2` 的份数公式，为了支持 128 个并发请求在 `publish_lag=1` 下流水线重叠，每个可变请求状态必须预先留出 $4 \times 128 + 2 = 514$ 份版本根（root）的物理槽位。

当状态本身很轻量时，这笔开销微不足道。以 `Qwen/Qwen2.5-7B-Instruct`（BF16，hidden 维度 3584）上的在线蒸馏为例：pLLM 的自动精度分析（`dtype="auto"`）只把需要被优化器累加更新的参数字段提升为 `float32`（防止微小更新在 BF16 的 7 位尾数里因大数吃小数而丢失；同一规则也覆盖“读出再写回”的累加器——在 32k token、chunk 64 的跨块递推测试中，进位用 FP32 的相对误差为 $3.692 \times 10^{-4}$，而用 BF16 高达 $1.437 \times 10^{-1}$，相差三个数量级），让只读或直接覆盖的字段保持 `bfloat16`。这样算下来，一份完整的 `Distill` 状态（含 rank-4 权重与全套 `AdamW` 动量）是 516,176 字节（约 504 KiB），预留 514 份共占约 **253 MiB**（若设 `publish_lag=0`，则为 386 份、约 **190 MiB**；不带偏置的 `MiniESamp` 每份 458,768 字节，514 份约 **225 MiB**）。在 24 GiB 的显卡上，用两百多兆显存换取 128 路并发在线训练，代价完全可控。

但如果状态本身是个庞然大物，账单就会立刻变得沉重。在[《参数覆盖层》](06-parameter-overlays.md)将要讨论的 In-Place TTT（基于 4B 社区检查点）中，算法要把 6 层 MLP 的下投影矩阵变成每个请求在线更新的快权重。哪怕用了前向内精确更新（`exact_decode`）把份数公式压到了 $2N + 2$，单个请求的一份快权重根就重达 **714 MiB**。这意味着哪怕只跑 1 个请求（4 份根），状态池就要吃掉 **2.8 GiB** 显存；并发开到 4 个请求（10 份根），近 7 GiB 的显存就会从 KV Cache 池里被划走。此外，在多卡张量并行（TP > 1）下，为了守住“扩展层绝不自己插入跨卡集合通信”的原则，普通的请求状态（如 `distiller`）在每个 TP rank 上完整复制而不分片——扩充显卡数量可以切分基座模型权重，但不会减少这类状态在每张卡上的显存占用；只有参数覆盖层的字段会跟着被覆盖的投影层按特征轴切分。

第二笔账单是**模型规模对双流重叠收益的稀释**。前面我们解释了 `publish_lag=1` 如何让旁路训练与下一步的主流前向重叠。那么在真实的基准测试里，这步滞后到底能换来多少端到端吞吐？这完全取决于一步旁路训练占一步主干前向的时间比例。

下面是在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0（开启 CUDA Graph 与 FlashInfer 采样器）、128 并发序列（8 条 prompt $\times$ `n=16`）、生成 256 个 token（预热 1 轮后取 3 轮中位数）的标准设置下，分别在 `Qwen2.5-0.5B-Instruct` 与 `Qwen2.5-7B-Instruct` 上实测的 ESamp（rank 4）对比数据：

| 模型与配置 | 原生 vLLM 吞吐 (单步耗时) | `publish_lag=0` (串行等待) | `publish_lag=1` (跨步重叠) | 旁路训练每步 GPU 耗时 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B** ($\beta=0$) | 42,355 tok/s (`3.0 ms`) | 38,047 tok/s (**89.8%**) | 38,936 tok/s (**91.9%**) | ~178 µs (AdamW ~29 µs，0.27.1 剖析) |
| **Qwen2.5-7B** ($\beta=0$) | 5,894 tok/s (`21.7 ms`) | 5,731 tok/s (**97.2%**) | 5,738 tok/s (**97.4%**) | ~616 µs (AdamW ~191 µs) |
| **Qwen2.5-7B** ($\beta=0.1$) | 5,894 tok/s (`21.7 ms`) | 5,663 tok/s (**96.1%**) | 5,681 tok/s (**96.4%**) | ~616 µs + 引导采样开销 |

在 **0.5B** 模型上，原生解码一步只有 **3.0 ms**，旁路训练和宿主记账在时间轴上占比可观，把 `publish_lag` 从 `0` 切到 `1`，相对吞吐从 **89.8%** 提升到了 **91.9%**。然而换到 **7B** 模型时，原生解码一步拉长到了 **21.7 ms**，而 rank-4 小网络的旁路训练一步总共只需 **616 µs**（其中分页 AdamW 占 191 µs），仅占整步时间的 **2.8%**。在 21.7 ms 的分母面前，这 616 µs 无论是串行等待（97.2%）还是多花 63 MiB 显存用 `publish_lag=1` 藏进旁路流（97.4%），端到端吞吐的差别只有 0.2% 左右（关于 GPU 满载时双流重叠为何受制于硬件资源争用，详见[《自动放置与重叠》](11-automatic-placement-overlap.md)）。

第三笔账单来自**整字段成页与静态硬容量的刚性边界**。因为一页就是整个字段，如果某个算法持有一张巨大的状态表、每步却只稀疏地修改其中一两行，字段级写时复制依然会分配一整块新字段槽位并拷贝未修改的其余行，造成写放大。同时，因为所有槽池必须在录制 CUDA Graph 之前钉死物理基地址，PTTS 的容量在引擎启动时就是硬上限（Hard Ceiling），运行期绝不会动态调用 `cudaMalloc` 扩容；一旦并发请求加上滞后版本超出了预留槽位数，新请求或跨步窗口就会直接收到 `WindowBackpressure` 准入拒绝。此外，在事务语义上，未发布的私有写随时可以调用 `abort` 无损中止；但一旦 `publish` 成功返回，新版本就已经对系统其余部分可见，若此后的下游结算步骤发生故障，绝不能把这次已发布的迁移当成“未提交事务”再重放一遍。同理，跨进程的 `BorrowedPage` 只读借用也是同一可信程序内部的接口约定，而非操作系统页表级的硬件写保护。

如果把视野拉远，拿 PTTS 与 S-LoRA、Punica 这类多 LoRA 服务系统并排比较，就能看清两者的定位差异：S-LoRA 同样借用了操作系统的“分页（Paging）”概念，但它面对的是成百上千个**只读**的预训练 Adapter，它的分页是为了把不同 rank 的静态权重塞进统一内存池防止碎片，并用定制的异构批量 GEMM 算子合并计算；而 PTTS 解决的则是**状态在生成每一个 token 的过程中都在被实时改写、需要跨流并发隔离、需要版本滞后流水线、需要随时应对抢占回滚**的动态状态问题。

---

## 五、 结语

回过头看，PTTS 的整套设计几乎是被 GPU 的两条底层定律一步步推出来的：第一，CUDA Graph 只认固定不变的物理地址；第二，GPU 上的“读写执行完成”与 CPU 上的“函数调用返回”之间隔着一条异步鸿沟。

面对这两条定律，如果停留在“原地修改显存”的模型里，每一个新场景都会逼你打上一块更复杂的锁或双缓冲补丁。而一旦退后一步，把“状态”重新定义为固定物理槽池上生长出来的不可变版本序列，图的地址绑定、双流的读写隔离、在途请求的快照保持以及抢占取消的回滚，就全都变成了同一个数据结构上的简单查询。在复杂的并发系统里，把被抹掉的时间维度作为显式的版本重新请回账本，往往比发明更精巧的锁要管用得多。
