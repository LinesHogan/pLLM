# 给 GPU 显存加上 MVCC：当每个并发请求都有一份会变的权重

现代大模型推理引擎之所以能把吞吐推到极致，很大程度上仰赖于一个隐含的前提：除了不断向后追加的 KV Cache 之外，模型本身对每个请求是无状态的。成百上千个并发请求共用同一份只读的模型权重，每算完一步就丢掉中间激活，彼此之间没有任何可变数据的牵连。

但只要你试着往推理过程里引入一点“跨步演化”的算法——比如在生成每个 token 的同时，在线为当前请求微调一个轻量级的预测网络来引导采样（如在线蒸馏 ESamp），或者随着上下文推进实时更新每个请求专属的“快权重”（如 Test-Time Training），又或者在生成中途动态注入外部的引导向量——这个平静的无状态假设就瞬间瓦解了。你突然需要在高速运转的推理引擎内部，为上百个并发请求维护可读可写、每一步都在变化的状态张量。

在单请求的研究脚本里，这根本算不上一个问题。你顺手建一个 Python 字典 `states[req_id] = nn.Module(...)`，新请求进来时 `deepcopy` 一份模板并绑上 `AdamW`，每步直接在原地更新权重（in-place mutation），请求结束时 `del states[req_id]` 就完事了。可一旦把这个想法放进一台真正在线上跑满的推理引擎，你会接连撞上一串棘手的系统故障。

最先找上门的是 CUDA Graph。为了省掉 CPU 逐个发射算子的微秒级开销，生产引擎会把解码步录制成静态的计算图反复回放。图在录制那一刻就把所有算子读写的显存指针焊死了，而字典里动态分配的请求张量散落在显存的各个角落，每一步哪些请求在批次里、排在第几行都在变化，录好的图根本找不到当前这一行对应的权重（见[《What Color is Your Tensor?》](02-tensor-colors.md)）。你可能会想：那我预先划出一块地址固定的静态槽数组，每个活跃请求来了借一个槽位、走了还回去，不就能满足 CUDA Graph 的固定地址要求了吗？

如果只用一条流串行执行，静态槽数组确实够用。但在线训练包含反向传播与优化器更新，如果串行堵在模型前向的主计算流（`main` stream）上，每一步都要白白多等几百微秒。自然的解法是把训练放到一条旁路流（`side` stream）上，让第 $n$ 步的权重更新与第 $n+1$ 步的模型前向在 GPU 上并行跑。然而，只要你还停留在“原地修改同一个槽位”的模型里，双流并发就会立刻引发读写撕裂：当旁路流上的 `AdamW` 正在往某个槽位里写第 $n$ 步的新权重和动量时，主流上的第 $n+1$ 步前向已经冲进来读同一个槽位了。两边没有任何锁，主流读出来的是一半新梯度、一半旧残骸的半成品。

即便你的状态不是在线训练出来的，而是由外部调用者在发起请求时动态导入的共享状态（比如激活引导里的方向向量），原地修改同样会闯祸。假设老请求 A 的句子才生成到一半，新请求 B 带着一个新的引导向量进场并直接覆盖了那块固定显存，正在说话的请求 A 就会在半句话中间被换掉参数，吐出前后割裂的文本。更不用说在线服务里随时可能发生的 KV Cache 抢占（Preemption）：当显存被长文本撑满、调度器把某个跑到第 50 步的请求临时驱逐出显存，稍后又从第 0 个 token 开始重新预填充它的历史时，原地覆盖早就把第 0 步的初始状态抹得一干二净了；如果拿着第 50 步训出来的权重去重算第 0 步的 K 和 V，算出来的注意力与被踢走前完全对不上。

如果把这四件事当成四个孤立的工程 bug 去修——给图录制预留静态槽，给双流并发做双缓冲，给外部导入排等待队列，给抢占写状态备份——你会发现补丁越打越厚，而且每加一个机制就会和另外三个打架。

遇到这种情况，往往意味着我们在错误的抽象层面上打转。不妨停下来退后一步，把具体场景的细节剥掉，看看这四起故障在底层到底在争抢什么：图录制问的是，怎么用一个不变的物理地址去指代运行中不断演化的状态；双流并发问的是，当写者正在制造下一份状态时，读者怎么安全地读取上一份状态；而在途导入与抢占重算问的则是，当新状态已经产生，某个读者怎么继续持有它出发时的那一份，或者退回过去的那一份。

这四个看似无关的问题，其实在问完全相同的一句话：**在时间线上的任意一刻，某一个读者究竟应该看见状态的哪一份快照？**

只要你的状态模型是“在原地修改同一块显存”，这个问题在数学上就无解。因为原地修改抹杀了时间维度——过去、现在和尚未写完的未来，被粗暴地拍扁在了同一个物理地址里。

---

## 一、 显存里的 MVCC 与 RCU：不可变版本、字段级写时复制与事件栅栏

一旦看清病根在于“丢失了时间维度”，解法根本不需要我们重新发明。在计算机科学的另外两个分支里，前人早就用同一个基础原语把这个问题彻底解决了。

在数据库系统里，它叫 **MVCC（多版本并发控制，Multi-Version Concurrency Control）**：写事务绝不在原地覆盖数据行，而是分配并写入一个新的行版本；读事务在开始时拿走一个确定性的版本快照，读的永远是那个时刻之前已提交的数据，读写之间连一把锁都不用加。在操作系统内核里，它叫 **RCU（Read-Copy-Update）**：写者先分配新内存、把旧数据拷过去改好，然后原子地切换发布指针；已经在读旧指针的线程继续读它们的，等所有在发布前进入读临界区的读者全部退出——这段时间称为“宽限期（Grace Period）”——旧内存才会被安全回收。

如果把 MVCC、RCU 和推理引擎的分页思想拧在一起搬上 GPU 显存，会得到什么？在 pLLM 中，这套机制被称为 **PTTS（Paged Tensor Transition System，分页张量状态迁移系统）**——注意它开头的 P 是 Paged（分页与页表，和操作系统的虚拟内存、vLLM 的 PagedAttention 同源），与 pLLM 名字里代表 programmable 的 p 没有任何关系，只是碰巧撞了首字母。

它的核心规则只有一句话：状态永远是不可变的（Immutable）。

在 PTTS 里，一份逻辑状态（比如请求 A 的预测器参数，称为一条状态谱系 Lineage）不是一块会被反复涂改的内存，而是一串单调递增的不可变版本（Version）。这套机制向整个系统承诺且只承诺四条硬不变量：
1. 读者在使用期间，看到的永远是一个固定不变的版本（通过精确钉住 `pin` 保证）；
2. 写者绝不覆盖任何仍被读者持有的物理页；
3. 每个状态谱系在同一时刻最多只有一个活跃写者；
4. 一个物理页只有在触碰过它的所有 GPU 读写工作全部完成后，才能被重新分配复用。

要在 GPU 上兑现这四条承诺，必须解决两个硬件特有的物理问题：第一，一份状态可能包含多个张量，每次更新都把整份状态复制一遍怎么付得起？第二，GPU 是异步执行的，CPU 上的 Python 函数返回了，不代表 GPU 队列里的内核跑完了，RCU 的“宽限期”在显卡上怎么界定？

答案是**字段级写时复制（Field-level Copy-on-Write）**与 **CUDA Event 回收栅栏**。

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

如上图所示，一份状态由若干个字段（Field）组成。在引擎启动的规划期，系统为每个字段预先分配好一块基地址永远不变的连续大张量，形状是 `[capacity, *field_shape]`，这就是该字段的**物理槽池（Slot Pool）**，第 0 维的下标就是物理槽号（Slot ID）。

那么，一个“版本（Version Root）”在物理上到底是什么？它仅仅是一组整数槽号——每个字段一个整数。

假设请求 A 的状态处于版本 `v7`，它的字段 `W1` 在 0 号槽，`b1` 也在 0 号槽。现在某个任务只修改 `W1`、不碰 `b1`。在准备这次状态迁移（`prepare_transition`）时，账本只为被修改的 `W1` 分配一个新的空闲槽（如 3 号槽），而未修改的 `b1` 直接沿用 `v7` 的 0 号槽，引用计数加一。发布一个新版本的显存与拷贝成本，只正比于真正被改写的字段，而不是整份状态的大小。

你可能会问：为什么这里的“一页”直接等于整个字段的张量，而不像 PagedAttention 那样切成 16 个 token 一页的小碎片？因为 KV Cache 是随序列推进一个 token 接一个 token 往后追加的，必须切小页防内部碎片；而在状态更新与在线训练里，无论是矩阵投影还是优化器步，几乎都是整个参数字段一起读写。把一个 `[4, 3584]` 的小矩阵再切成碎块，除了徒增页表寻址开销，换不来半点拷贝节省。与其说它像操作系统里 4 KB 的虚拟内存小页，不如说它是一组带引用计数和版本号的预分配定长缓冲池。

至于异步 GPU 上的“宽限期”，则由 `torch.cuda.Event` 把关。当读者释放它钉住的旧版本，或者写者完成一次迁移时，对应的物理槽位哪怕逻辑持有计数（`holders`）归了零，也不能立刻分给别人用——因为刚才那条 CUDA Stream 上发射的读写内核可能还在硬件队列里排队。系统会在该流上记录一个非阻塞的 `torch.cuda.Event` 栅栏（Fence）。之后只有当宿主轮询发现该 Event 已经完成（`event.query() == True`），或者接手该槽位的下一条流在 GPU 侧显式等待了这个 Event（`stream.wait_event`），这块物理内存才允许被重新写入。

这一点在“操作中途取消（`abort`）”时表现得尤为明显。来看看 `pllm/execution/ptts/page_bundle.py` 里发布新版本与中止私有写的实现：

```python
# check-docs: skip: excerpt from pllm/execution/ptts/page_bundle.py
def publish(self, token: PageWrite) -> PagePublication:
    self._issued_write(token)
    completion = self._record(token.stream)
    completed = self._events.completed()
    retained = self._ledger.publish(token.operation_id, self._events.add(completion), completed, False)
    self._events.retain(retained)
    return PagePublication(token.operation_id, token.before_refs, token.after_refs, completion)
```

紧挨着它的 `abort` 与 `publish` 的函数体几乎一字不差，只差传给底层账本 `self._ledger.publish(...)` 的最后一个布尔参数（`False` 代表正常发布，`True` 代表作废）：

```python
# check-docs: skip: excerpt from pllm/execution/ptts/page_bundle.py
def abort(self, token: PageWrite) -> None:
    """End logical ownership now; future physical reuse waits for the queued writes."""
    self._issued_write(token)
    completion = self._record(token.stream)
    completed = self._events.completed()
    retained = self._ledger.publish(token.operation_id, self._events.add(completion), completed, True)
    self._events.retain(retained)
```

为什么连“作废一次写入”都非要在 `token.stream` 上录一个完成事件 `completion` 并传给账本立栅栏？因为在异步硬件上，“CPU 决定取消了”和“GPU 已经停手了”完全是两码事。当 Python 调用 `abort` 的那一刻，这次迁移在逻辑上的所有权确实当场终结了，那个私有版本永远不会变成可见版本；但在调用 `abort` 之前，写者可能已经往 `token.stream` 的队列里发射了三四个写显存的内核，它们此刻正在 GPU 上飞奔。如果在 `abort` 时不打下 Event 栅栏就直接把私有槽位放回空闲池，下一微秒另一个流上的新请求分到同一个槽位，就会和尚未跑完的废弃写指令迎头相撞。逻辑上立刻作废，物理上老老实实排队等栅栏完成——这就是异步显存回收的铁律。

---

## 二、 四个推论：看棘手难题如何自然消解

检验一个底层原语立得正不正，最好的办法是看当初把你逼入绝境的那几个问题，在它面前是不是全都退化成了无需特判的平庸推论（trivial corollaries）。

### 1. 静态图寻址与无拷贝优化器步

先看 CUDA Graph 的地址绑定问题。既然每个字段的物理槽池都是在启动时一次性分配的固定大张量，基地址永不改变，而一个版本又只是一串整数槽号，那么“在图里读写动态变化的版本”就彻底退化成了一次数组下标寻址。

每一步开始前，CPU 侧的账本把当前批次里各请求应读的源槽号（`sources`）和应写的目标槽号（`destinations`）填进那张固定地址的共享 `int64` 表里（详见[《What Color is Your Tensor?》](02-tensor-colors.md)），一次异步拷贝推上 GPU。录在 CUDA Graph 里的内核只认池子的固定基地址和表的固定地址，先从表里读出整数槽号 `slot`，再去池子里访问 `pool[slot]`。

更妙的是，对于像 `AdamW` 这样每次都要同时更新参数、步数和一阶、二阶动量四个字段的操作（梯度直接取自本步算出的结果，不存成字段），分页优化器内核（`pllm/kernels/adamw.py`）直接拿着 `sources` 槽号从旧版本物理槽里读出旧参数和动量，在寄存器里算完一步更新后，直接写进 `destinations` 指向的新版本物理槽——连写时复制事先拷一遍旧页的动作都一并省掉了。

### 2. 双流重叠、`publish_lag` 与一行容量公式

再看双流并发下的读写撕裂。有了不可变版本，主流上的前向读者钉住已发布的旧版本，旁路流上的训练写者往新分配的私有槽位里写新版本，两边物理上碰的根本不是同一块显存，读写冲突自然消失了。

但这里藏着一个更关键的流水线问题：**旁路流写出来的新版本，主流什么时候才能看见？**

在单步解码的 8 个全序时间锚点（`request.prepare`、`model.execute`、`state.publish`、`sampler.pre_draw`、`sampler.complete`、`token.commit`、`request.finish`、`next.prepare`）中，在线训练任务 `loss` 必须等第 $n$ 步的采样结果落定（`token.commit` 锚点，已临近一步的末尾）之后才能启动。它要在旁路流上跑完小网络前向、反向传播和 `AdamW` 更新。而几乎在 `token.commit` 的同一时刻，主流上的第 $n+1$ 步就已经跨过步边界，准备开始跑模型前向和采样引导 `predict` 了。

```text
Main Stream:  [  Step n Forward & Sample  ] ──► [ Step n+1 Forward (reads v_{n-1}) ] ──► [ Step n+2 Forward (reads v_n) ]
                                           \                                            ▲
Side Stream:                                └──► [ Step n Train (reads v_{n-1} -> writes v_n) ]

Simultaneously Held Versions per Lineage (history = 1, written = True -> 3 pages/lineage):
  ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
  │ 1. Step's Read Version │   │ 2. Lagged History (H)  │   │ 3. Private Write (W=1) │
  │  (pinned until commit) │   │ (includes latest pub)  │   │ (filled before publish)│
  └────────────────────────┘   └────────────────────────┘   └────────────────────────┘
```

如果你坚持要求 `publish_lag=0`——即第 $n+1$ 步的 `predict` 必须读到第 $n$ 步刚训出来的 `v_n`——主流在跨入第 $n+1$ 步时就必须踩下急刹车，原地干等旁路流把第 $n$ 步的训练全部跑完，两条流在时间上彻底退化成串行。

反过来，只要允许 **1 步的可见性滞后（默认的 `publish_lag=1`）**，整条流水线就彻底解耦了：
* 当第 $n$ 步的训练在旁路流上写入私有版本 `v_n` 时，第 $n+1$ 步的主流根本不等它，直接读取步边界上早已就绪的上一版 `v_{n-1}`（在 `pllm.explain` 的计划里显示为读取 `(boundary, 1)`）；
* 等第 $n$ 步的训练在后台跑完并发布 `v_n`，第 $n+2$ 步的主流自然就能读到 `v_n`；
* 与此同时，训练任务自己在旁路流上读的永远是 `(latest, 0)`——第 $n+1$ 步的训练紧跟在第 $n$ 步训练后面，直接在最新的 `v_n` 上继续迭代，一步梯度都不会丢。

用“可见性晚一步”换取“双流无锁并行”，账面上的代价是显存里同一时刻必须多留几份历史版本。到底要预留多少页，才能保证在满负荷流水线下绝不因为槽位耗尽而卡死？

在 `pllm/execution/ptts/page_bundle_planning.py` 里，这个容量规划被浓缩成了一行极其干净的算术：

```python
# check-docs: skip: excerpt from pllm/execution/ptts/page_bundle_planning.py
def version_capacity(lineages: int, history: int, *, written: bool) -> int:
    # ……
    return (1 + history + written) * lineages
```

这行 `(1 + history + written) * lineages` 里的每一项，都严丝合缝地对应着状态机里一个真实的物理角色（其中 `history` 是任务图中跨步滞后读取的最大步距，对应 `publish_lag`；`written` 在有任务改写该状态时为 `True` 即 `1`，只读时为 `0`）：
* **`1` 份当前步读取版**：当前步从开始到提交期间一直钉住（pin）正在使用的版本；
* **`history` 份滞后保留版**：采样后重叠机制为滞后读者保留的旧版本。注意一个很巧的细节：这批保留版本里最新的一份，恰恰就是当前最新发布的版本（latest published version），因此“最新发布版”根本不需要再单独多占一个槽位！而且当一个老请求生成结束退场时，它留在滞后窗口里的旧版本可以和刚刚进场、还没有任何历史包袱的新请求共享同一组座位；
* **`written`（1 或 0）份私有写版**：如果状态可写，迁移任务在发布前正在后台填充的那份私有槽位。

如果是全局共享的 `InstanceState`，只读时整个实例只需要 `1` 页；可写时按单条谱系计算（`1 + history + 1`），若还有模型内的 `Adapter` 或 `ActivationPatch` 在请求准入时锁住了它，而又有任务在请求运行期间每步改写它，由于每个先后进场的请求锁住的准入版本可能各不相同，再加上 `max_admitted_lineages` 页即可；如果它只由 `generate(batch=...)` 传入的主机批次改写（如 `StateImport`、`CausalTrainBatch`），这次写只发生在每次调用的第一步，请求锁住的只可能是写前或写后的那一版，不必再加（而由外部流式喂入的 `Fed` 状态则固定为 `3` 页：引擎读一页、一页排队、生产者写下一页）。

这行公式算的是“被读者读的字段”的页数。`AdamW` 的步数和一阶、二阶动量只有优化器自己读，而且只读它正在迁移的最新版本，所以旧版本的钉（滞后读者、准入请求）只握住参数字段，这三个字段每条谱系只需最新版加私有写版两页。

更值得注意的是这行公式里**没有**什么：它没有为“GPU 队列里还没跑完的在途读取内核”多留哪怕一页冗余。为什么？因为前面说的 CUDA Event 完成栅栏已经在管物理复用了！一页一旦在逻辑上被释放，下一次分配到它的写操作只需要在 GPU 流上等一下它的完成 Event，根本不需要靠多划显存来躲避未完成的内核。

### 3. 在途请求隔离与抢占回滚的分界线

有了不可变版本，前面提到的第三和第四起故障——在途参数导入与 KV 抢占回滚——同样不需要任何复杂的同步补丁。

当外部通过 `StateImport` 动态推入一份新的引导向量时，系统只是在状态谱系上做了一次迁移，发布了一个新版本 `v_new`。新准入的请求在进门时调用 `pin_exact` 锁住 `v_new`；而更早就已经进场、此刻正说到半句话的老请求，手里钉住的依然是它进门时的 `v_old`（除非你显式声明了 `refresh="step"` 要求每步跟随最新版）。新老请求在同一个批次里并肩跑过模型层，各自从同一个物理槽池的不同槽号里取出自己的向量，井水不犯河水。（甚至当同一张 GPU 上的另一个进程通过 CUDA IPC 只读借用某个精确版本的页时，对账本来说也只是多了一个握着 `PagePin` 的普通读者，等远端进程发回完成凭证后再过栅栏回收即可。）

真正值得细品的是 **KV Cache 抢占重算（Preemption）**。当一个请求生成到第 50 个 token 被 vLLM 临时踢出显存，稍后又从第 0 个 token 开始重新 prefill 这 50 个 token 时，它身上的 PTTS 状态到底应该**保留在第 50 步的最新版本（`keep`）**，还是**倒回它刚进门时的初始版本重头再来（`reset`）**？

如果没有多版本账本，你连问这个问题的资格都没有，因为初始版本早就没了。而在 PTTS 里，答案取决于一条极其清晰的物理分界线：**这份状态在请求运行期间是否被改写，且改写是否影响了模型算出来的 K 和 V？**

来看看 `pllm/execution/preemption.py` 里的判定函数，全篇只有寥寥数行：

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

顺着这行 `return` 看过去，所有算法被干净利落地一分为二：
* **只读隐藏层或只影响采样器的状态（如 ESamp 的 `distiller`）$\rightarrow$ `keep`**：在 ESamp 里，小网络只旁路读取第 0 层和最后一层的隐藏状态，只在最后采样器掷骰子时修正 logits，从头到尾没有改过 Transformer 内部的任何一个激活值或权重。当 vLLM 把前 50 个 token 重新送进模型重算 KV Cache 时，模型算出来的 K 和 V 与被踢走前一模一样！因此，小网络辛辛苦苦训了 50 步的权重完全应该保留（`keep`）；唯一要做的是别让那 50 个重算的旧 token 再被送进训练器训第二遍——这只需要行标签器用历史水位线 `frontier` 把重算行过滤掉即可（详见[《你其实不需要读 `attn_metadata`》](01-attn-metadata.md)）。
* **运行中被改写、且直接参与模型前向的状态（如在线更新的参数覆盖层）$\rightarrow$ `reset`**（自定义残差、自定义注意力与自适应跳层则不看状态是否被改写，一律 `reset`）：此时第 10 步的 K/V 是用版本 `v10` 算出来的，第 20 步的 K/V 是用 `v20` 算出来的。如果拿着第 50 步的 `v50` 去重算前 50 步的 K/V，历史就被篡改了。所以策略必须是 `reset`：调用 `rewind_versions` 把状态指针拨回准入时锁定的初始版本，让状态跟着重算的 token 一步步重新演进。
* **最巧妙的交集（`applied & _rewritten(graph)`）**：如果一个 `Adapter` 或 `ActivationPatch` 确实改写了模型内部激活，但它在请求运行期间从来没有被任何任务改写过（例如它是冻结的，或是在请求第一次前向之前通过 `StateImport` 导入的，不在 `_rewritten(graph)` 集合里），策略依然是 **`keep`**！因为请求从进门起钉住的就是那个不可变版本，运行中没人改过它，重算时读到的自然还是同一个版本，算出的 K/V 分毫不差，连倒回的动作都省了。

---

## 三、 控制面与数据面分离，以及多版本的物理账单

把“不可变版本 + 字段级槽池 + Event 栅栏”想清楚之后，工程落地还有一个现实挑战：在每步只有几毫秒的解码循环里，怎么让这套版本记账逻辑本身足够快，而且绝对抓得住野指针？

做法是在宿主端划定一道清晰的边界：**Python 只保管 GPU 设备对象（字段池的大张量、`torch.cuda.Event` 实例），而所有的数字事实（哪个槽被谁持有、每个槽分配到了第几代、哪些槽在等哪个 Event 编号）全部收归单线程无锁的 Rust 数组账本（`rust/pllm_core/src/slot_ledger.rs` 与 `page_ledger.rs`），跨语言边界只传递扁平的 `int64` 数组。**

看看 `slot_ledger.rs` 里物理池的账本结构：

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

这里连一把互斥锁都没有——因为整个状态 Arena 严格归执行模型步的那一个宿主线程单线程拥有。`holders` 是每个槽位的当前持有计数；`fences` 记录着“哪一个 Event ID 完成之前，哪些槽位被禁止复用”；而 `generations`（分配代次）则是用来抓过期引用的保险丝：每当一个物理槽位的持有计数归零、准备迎接下一次分配时，它的代次就加一。如果上层代码因为 bug 拿着一个已经释放的旧 `PageSlot(slot_id, generation)` 回来取张量，视图检查一比对代次不对，当场就会抛异常报错，绝不会悄无声息地读到已经分给另一个请求的脏数据。

同时，任何不直接服务于前述四条硬保证的日常开销都被砍掉或改为了**惰性触发**。例如，只要池子里还有空闲槽位，未决的释放收据留在那儿根本不妨碍任何一条保证；只有在 `prepare_transition` 真正遇到槽位不足（抛出 `_core.PageCapacity`）时，系统才会去轮询清理已完成的收据并重试。在单张 NVIDIA RTX 4090（24 GiB GDDR6X）上对 Qwen2.5-0.5B（128 并发，ESamp $\beta=0$，vLLM 0.27.1）做的宿主端剖析中，每次调用的中位耗时分别是：`prepare_transition` 约 **181 µs**、`publish` 约 **82 µs**、`release` 约 **58 µs**。相对单步耗时约 21.7 ms 的 7B 模型，这点微秒级的 CPU 记账开销就很小了。

当然，在系统工程里，没有任何一个抽象是免费的。当我们选择用“预分配池 + 字段级 COW + 多版本驻留”来换取无锁并发和 CUDA Graph 兼容时，必须支付三笔实打实的物理账单：

**第一笔账单是显存容量，而且它直接与 KV Cache 争抢空间。**
在推理引擎的显存食物链里，模型权重、CUDA Graph 捕获池与 PTTS 状态池先分走固定份额，最后剩下的空间才归 KV Cache。根据前面的容量公式 $(1 + H + W) \times N$，在 128 并发（`max_num_seqs=128`）、默认 `publish_lag=1`（即 $H=1, W=1$）下，每个被滞后读取的可写请求状态的参数字段需要预先留出 $3 \times 128 = \mathbf{384}$ 页物理槽位（若设为 `publish_lag=0`，则是 $2 \times 128 = \mathbf{256}$ 页；`AdamW` 的步数与动量字段始终是 256 页）。

这笔买卖划不划算，完全取决于你的状态有多大、以及用什么精度存。
为了防止微小的梯度更新在低精度浮点数里因为“大数吃小数”而丢失，pLLM 的自动精度分析（`dtype="auto"`）会将所有被优化器更新的参数和跨块累加器（carry）自动设为 `float32`，而只读或只写字段则跟随模型的 `bfloat16`（在 32k token、块长 64 的长序列递推测试中，FP32 carry 的相对误差仅为 $3.692 \times 10^{-4}$，而 BF16 高达 $1.437 \times 10^{-1}$，相差整整三个数量级）。

在单张 RTX 4090（vLLM 0.32.0，开启 CUDA Graph 与 FlashInfer 采样器，128 并发序列）下，我们用 `pllm.explain` 算一笔 `Qwen/Qwen2.5-7B-Instruct`（BF16，hidden 维度 3584）的真实显存账：
* **轻量状态极其便宜**：以在线蒸馏 `Distill`（rank-4 双层网络配 `AdamW`）为例，单份状态根（一个版本的四个字段）的大小是 $387,136$ 字节（约 378 KiB）。在 `publish_lag=1` 下参数字段预留 384 页、优化器字段各 256 页，总共只占约 **110 MiB**（$115,623,984$ 字节；若设 `publish_lag=0` 全部 256 页，则只需约 **95 MiB**）。在一张 24 GiB 的显卡上，花不到 200 MiB 显存买下 128 个并发请求的在线训练与无锁隔离，几乎毫无压力。
* **巨型状态则会迅速吃穿显存**：如果你跑的是像 In-Place TTT（4B 检查点，见[《How to Lose 61,440 Kernels in 10 Lines》](04-compiled-model-hooks.md)）那样把 6 层 MLP 的下投影矩阵本身变成每请求可写快权重的算法，单份状态根就重达 **714.0 MiB**！哪怕只跑 1 个请求（2 份根）也要占掉约 1.4 GiB，并发开到 4 个请求（8 份根）就会吃掉约 5.6 GiB 的 KV Cache 空间。此外，为了守住“扩展层绝不自己发射跨卡集合通信”的原则，除通过分片证明的覆盖层外，PTTS 状态池在多卡张量并行（TP > 1）下是按 rank 完整复制的，扩卡并不会减少状态池的每卡显存占用。

**第二笔账单是“整字段成页”与“硬容量上限”的刚性边界。**
因为 PTTS 的一页就是整个字段张量，如果你的算法拥有一张巨大的状态表、每步却只稀疏地修改其中一两行，写时复制依然会为你分配一整块新字段槽位并拷贝未修改的行，带来明显的写放大。同样，由于所有物理槽池必须在录制 CUDA Graph 之前钉死基地址，池容量在启动那一刻就是硬上限（Hard Ceiling），运行期绝不动态 `cudaMalloc` 扩容；一旦并发与滞后版本超限，准入就会直接触发 `WindowBackpressure` 拒绝，而不会像操作系统虚拟内存那样换页到磁盘。而且在语义上，未发布的私有写随时可以 `abort` 作废，但一旦 `publish` 成功返回，新版本就已对全系统可见，倘若后续步骤再出错，不能把已发布的迁移当成未提交事务重放。

**第三笔账单，则是对比只读分页系统时的定位差异。**
如果你熟悉 **S-LoRA** 或 **Punica** 这类多 LoRA 服务系统，会发现它们虽然也用“分页（Paging）”这个词，解的却是完全正交的另一个问题：S-LoRA 面对的是成百上千个**只读**的预训练 Adapter，它的分页是为了把不同 rank 的静态权重塞进统一内存池防外部碎片，并用定制的异构分组 GEMM 算子批量计算；而 PTTS 面对的是**在生成每一个 token 的过程中都在实时演变、需要主旁双流并发读写、需要跨步滞后流水线、需要随时应对抢占回滚的活状态**。至于 `publish_lag=1` 的双流重叠到底能在不同规模的模型上换来多少吞吐（例如在单步 3.0 ms 的 0.5B 模型上能从 89.8% 提升到 91.9%，而在单步 21.7 ms、训练仅占 616 µs 的 7B 模型上则是 97.2% 与 97.4% 的微小差别），我们留到[《You Probably Don't Need a Compiler IR》](09-stream-scheduling.md)里连同 GPU 流调度一起算总账。

---

回过头看，当我们在高并发推理引擎里被图的固定地址、双流的读写竞争、在途请求的干扰以及抢占回滚撞得焦头烂额时，真正救场的并不是什么精巧的锁或复杂的双缓冲状态机，而是退后一步，承认“状态不可变，每次更新只是发布一串指向固定池子的新槽号”。

一旦把时间作为不可变的版本显式请回系统里，原本充满时序不确定性的并发控制，就变成了一本确定性的、可以用几个平坦的 `int64` 数组算得清清楚楚的账本。记账当然要付显存和预留的价钱，但账本不会说谎，更不会在两条 CUDA Stream 微妙的微秒级交错里，悄悄算错一个比特。
