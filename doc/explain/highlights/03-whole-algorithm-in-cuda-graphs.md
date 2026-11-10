# Python 只在录制时活着：固定地址、一张表、零同步

写 PyTorch 的人有一个很舒服的心智模型：代码是一行一行执行的。`hidden = layer(x)` 跑一段前向，`loss.backward()` 算一遍梯度，中间想 `print(loss.item())` 就 `print`，想按某个张量的值走 `if` 就走。Python 是指挥，GPU 是乐队。

这个模型在 vLLM 的 decode 循环里是错的。稳态生成时，指挥 GPU 的不是 Python，而是一段录好的 CUDA graph；Python 只在录制的那一次活过，之后每一步都只是回放。这篇文章要做的事情，是在这个前提下，把一个在线学习算法的一整步——取出某个请求的参数、跑用户写的前向、算 loss、反向、AdamW、写回——装进 CUDA graph。我们会从一个请求、一步开始，一点一点加上现实的麻烦，看每个麻烦逼出了什么。

## 第零步：一个消失的 hook

故事从 2025 年的旧 tLLM 开始。ESamp 每一步要拿第 0 层的 hidden 去跑一个小网络（distiller），它的预测要在同一步的采样器里用上。如果你是 PyTorch 用户，第一反应一定是 forward hook：在第 0 层挂一个 `register_forward_hook`，hidden 一出来就顺手发起小网络的预计算。时机最早，代码也干净。

我们当年就是这么写的。为了确认时序，给链路加了 CUDA event 计时，日志里出来这么一行（旧 tLLM 优化记录第 10 步）：

```text
schedule_attempt_count=1, schedule_hit_count=0, fallback_count=30
```

调度尝试了一次，命中零次，回退了三十次。hook 里发起的预计算，在回放的 decode 步里基本没有跑到。

要理解为什么，得先说说 CUDA graph 是什么。PyTorch 里一句 `c = a + b`，背后是 CPU 经过 Python 解释器、PyTorch 的调度、CUDA 驱动，发射一个 kernel 到 GPU，每次要花几微秒的 CPU 时间。prefill 时一个 kernel 要算很久，这几微秒无所谓。decode 不一样：每个请求每一步只算一个 token，128 个请求加在一起，很多 kernel 在 GPU 上也只跑几微秒，而一个 Transformer 走完一次前向要上千个 kernel。CPU 一个一个地发射，GPU 就一直在等。

CUDA graph 的办法是录制：在预热时让 CPU 把这一整串 kernel 连同参数（包括每个张量的指针）和依赖关系录成一张图，之后每一步只发一次 `cudaGraphLaunch`，GPU 自己把整张图跑完。差别有多大？在 RTX 4090 上，原生 vLLM 0.32 跑 Qwen2.5-0.5B（8 个 prompt × n=16 = 128 行，256 token），开着 graph 是 42,355 tok/s，一步约 3.0 ms；关掉 graph 只剩 20,926 tok/s，是原来的 49%。7B 一步本来就要 21.7 ms，关掉 graph 也还有 98%（5,773 对 5,887）。步越短，graph 越是命根子。

现在谜底就清楚了。录制只有一次，录制时 hook 确实被调用了，但 GPU 只记下它发射的 kernel。“设一个 Python 标志”“把张量存进一个列表”“满足某个条件时发起一个异步任务”——这些全是 Python 层面的事，GPU 看不见，图里也就没有。回放时 Python 一行不跑，hook 里的调度自然消失了。

当时的修法是把触发点挪到被包装的 `model.compute_logits` 之后（第 11 步）：`fallback_count` 从 30 降到 0，烟测吞吐比第一次稳定越过 79%（0.8243）。今天 pLLM 在 vLLM 编译模型之前装 hook，第一件事仍然是这个（`pllm/backends/vllm/runner.py`）：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/runner.py
def install_model_hooks(self) -> None:
    """Install the logits hook and announced hidden taps before vLLM compiles the model."""
    model = self.decoder.model
    model.compute_logits = self._compute_logits
    ...
```

`compute_logits` 是个好地方：它被调用时 Transformer 的前向已经结束，采样还没开始。前向侧的工作在这里完成，结果正好赶上同一步的采样器（[《采样尾声融合》](07-sampler-epilogue.md)会接着讲这个边界）。

## 第一步：加上训练

hook 挪对了地方，预测就能跑了。可 ESamp 还要训练：每一步用这个请求刚算出的 hidden 更新它自己的小网络。先只想一个请求，用普通的 PyTorch 写，大概是这样（示意，不是 pLLM 的代码）：

```python
# check-docs: skip: illustrative sketch of one request's training step, not pLLM code
params = states[request_id]                  # 取出这个请求的参数
loss = (f(params, shallow) - deep).square().mean()
grads = torch.autograd.grad(loss, params)
adamw_step(params, grads, moments[request_id])
```

四行。可在 decode 步里，这四行会变成几十个小 kernel：gather 参数、前向、loss、归约、反向、AdamW 的几个逐元素操作、写回。2026 年 10 月，在通用捕获做出来之前，我们用 py-spy 看过 0.5B、128 并发的 ESamp：pLLM 每步多出约 4.6 ms 的宿主时间，最大的一块就是 CPU 在一个一个地发射这些训练 kernel，光反向就占训练的约 36%。0.5B 一步只有 3 ms 左右，这根本藏不住。

所以训练也必须录进 graph。录之前先把规矩说清楚。图里记的是指针，所以**地址要固定**：张量要是换了地方，回放时 GPU 还会去读旧地址。图里记的是某个固定大小的网格，所以**形状要固定**：可服务流量是动态的，这一步 128 个请求，下一步可能走了 3 个、又来了 5 个。还有，**不能有主机同步**：录制时 CPU 等 GPU 会破坏捕获，回放的热路径上也不能有。

第三条值得多说两句，因为它最容易被“就这一次”打折扣。在一次 0.5B、128 并发的 profile 里（RTX 4090，vLLM 0.27.1），pLLM 每步在 CPU 上的准备工作约 700 µs，藏在约 3 ms 的 GPU 前向后面；准备好的数据一次非阻塞拷贝送上去，约 6.24 µs，GPU 步与步之间只空闲 4–6 µs。只要热路径上出现一次 `.item()`，CPU 就得停下来等 GPU 把队列跑空，那 700 µs 立刻从“藏在后面”变成“挡在前面”。我们在这个项目里学到的经验是：**少几次同步什么也换不来，只有变成零才有用。**

带着这三条规矩，一个一个往上加。

## 第二步：两个请求，一张表

一个请求的时候，录制很简单：参数就在那儿，录下它的地址，回放即可。

两个请求呢？每个请求的参数在不同的地方，这一步该用哪个请求的，每步都可能不同。“不同的地址”和“地址固定”看起来是矛盾的。解法是多加一层间接：把所有请求的参数放进一块地址固定的大张量（一个池），每个请求占一个槽；图里只认这个池的地址。至于“第 i 行用第几号槽”，那是**数据**，每步写进一张同样地址固定的小表里，图里的 kernel 先查表、再按槽号去池里取。版本换了、请求换了，改的都是表里的数，图一次也不用重录。（池和槽号背后那套带版本的分页状态，是[《PTTS》](01-ptts.md)的主题。）

这张表的布局写在 `pllm/execution/tensor_executor.py` 里，注释本身就是规格说明：

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

四段：每行的信息（属于哪个状态、版本、在批里的位置、是否有效）；每个字段、每个状态从哪个槽读；往哪个槽写；以及外部张量的地址和步长（最后这一段我们留到第五步再说）。每一步，CPU 用 Rust 的 `_core.fill_table` 把数填进一块锁页（pinned）内存，然后发一次非阻塞拷贝。在 0.5B、β=0 的 ESamp 里，加上这张表之后每步主机到设备一共约 23,692 字节、6.24 µs；原生 vLLM 自己是 1,088 字节、2.50 µs（RTX 4090，vLLM 0.27.1 时期的 profile）。

（旁注：为什么要强调“锁页”？`pllm/execution/cuda_stream.py` 里 `upload` 的 docstring 说了一个很多人不知道的坑：`torch.tensor(values, device=cuda)` 是从可分页内存拷贝的，会让主机等着流跑完；先放进锁页内存再拷，这次拷贝就只是排在 GPU 已有工作的后面。一个看起来无害的构造函数，藏着一次同步。）

## 第三步：请求数在变，用一个哑状态

两个请求能做了，那 30 个呢？128 个呢？这一步 30 个、下一步 31 个呢？形状要固定，可请求数在变。

你可能会想到为每一种请求数各录一张图——vLLM 对模型本身就是按几档 batch 大小这么录的。可训练图要考虑的不只是行数，还有状态数，每一档都是一个私有的显存池，档位一多就放不下。pLLM 的做法更朴素：**按最大容量录一次**。每个任务在规划时就知道最多有多少行（R）、最多有多少个请求的状态（S），张量形状就按 R 和 S 固定。这一步只有 30 个请求？其余 98 行都指向一个额外的“哑状态”——第 S 号。它可以被读（读出来是零），在所有归约里权重是 0，而且永远没有写的目标。回头看上面那张表：`sources` 是 `[F, S+1]`，`destinations` 是 `[F, S]`，多出来的那一列就是给它的。

于是不管来的是 1 个还是 128 个请求，GPU 看到的形状都是 `[R, …]` 和 `[S, …]`，图从来不用重录。代价也很直白：空位也要算。我们后面再算这笔账。

## 第四步：选择留在 GPU 上

很多算法不是每一行都要训练：只训练 loss 超过阈值的行，只更新被选中的状态。最直接的写法是把掩码读回 CPU，`(mask > 0).nonzero()`，再决定更新谁。可那就是一次同步。

pLLM 的选择是：**没被选中的状态照样发布一个新版本，只是字段逐字节不变**。选择完全在 GPU 上算，CPU 根本不需要知道这一步谁被选中了。`_selection` 的 docstring 把这个取舍写得很清楚：

```python
# check-docs: skip: excerpt from pllm/execution/tensor_executor.py
def _selection(self, rows: FixedRows) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Which rows train, which states update and each state's last trained row (-1: none), on the device.

    `rows=` selects the trained rows; `update=` (or else the trained rows) selects the states whose optimizer
    fields change. Every other state publishes its fields unchanged, so selectors never sync the host.
    """
```

花一点冗余——发布一个内容没变的版本——换掉一次主机读回。在 decode 的时间尺度上，这几乎总是划算的。

## 第五步：别人的张量会搬家

到这里，凡是 pLLM 自己分配的东西都住在地址固定的池里。可有些输入不归我们管：采样器用的张量是 vLLM 分配的，某些步可能换一块内存。如果把这个张量当参数录进图，它一搬家，图就得重录。

这时候前面那张表的第四段就派上用场了：不把张量本身录进图，而是把它的**地址**当成一个整数写进表里，连同两个步长；然后让 kernel 自己去解引用。`pllm/kernels/page_copy.py` 里的这个 Triton kernel 是我们最喜欢的一段代码：

```python
# check-docs: skip: excerpt from pllm/kernels/page_copy.py
@triton.jit
def _gather_rows(
    descriptor: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    width: tl.constexpr,
) -> None:
    row = tl.program_id(0) // tl.cdiv(width, 256)
    column = (tl.program_id(0) % tl.cdiv(width, 256)) * 256 + tl.arange(0, 256)
    # Triton lowers the annotated Torch argument to a typed device pointer.
    address = tl.load(descriptor).to(tl.pointer_type(output.dtype.element_ty))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    stride0 = tl.load(descriptor + 1)
    stride1 = tl.load(descriptor + 2)
    index = tl.load(rows + row)
    value = tl.load(address + index * stride0 + column * stride1, column < width, 0)
    tl.store(output + row * width + column, value, column < width)
```

关键是 `address` 那一行：从 `descriptor` 读出一个 int64，`.to(tl.pointer_type(...))` 把它变成一个带类型的指针，接下来两行再读出两个步长，然后照常 `tl.load`。为什么这在 graph 里成立？因为图录下的是“从 `descriptor` 这个固定地址读一个整数，再顺着它去取数”这个**动作**。`descriptor` 本身是表的一部分，地址不变；它的**内容**每一步由 CPU 重写。外部张量搬到哪里都无所谓，变的只是表里的一个数。

写过 C 的人看到这里大概会会心一笑：这不就是一级间接寻址吗？是的。可在“张量就是张量”的 PyTorch 世界里，把 `data_ptr()` 当成普通整数塞进一张表、再在 GPU 上把它变回指针，确实有点离经叛道。我们觉得它漂亮，正是因为它简单：一行转换，就把“图的生命周期”和“外部内存的生命周期”拆开了。

## 第六步：真的去录

把这些拼起来，训练步由 `pllm/execution/step_capture.py` 里的 `StepCapture` 录制：地址不变就不重录。录制时有两个细节，都是只有真跑起来才会碰到的。

第一个，录制之前把所有写目标填成 −1：

```python
# check-docs: skip: excerpt from pllm/execution/step_capture.py
def _capture(
    self,
    body: Callable[[], None],
    table: torch.Tensor,
    destinations: torch.Tensor,
    buffers: tuple[torch.Tensor, ...],
) -> None:
    # Release the invalidated pool before allocating its replacement.
    self.graph = None
    # Inactive destinations make warmup and capture read-only with respect to the arena.
    destinations.fill_(-1)
    self.graph, _, pool = capture_graph(table.device, body)
    # Maximum-shape tracing funds capture at build, so exceeding it is a framework bug.
    self.resident_bytes = sum(value.untyped_storage().nbytes() for value in (table, *buffers)) + pool
    assert self.resident_bytes <= self.budget.resident_bytes, (self.resident_bytes, self.budget.resident_bytes)
```

录之前要预热几次，预热跑的是真的 kernel，包括 AdamW 往池里写。−1 表示“不写”，于是预热和录制对真实的状态池是只读的，录一次图不会弄脏任何请求的参数。最后那个 `assert` 也有讲究：图池的大小在构建期已经按最大形状推导过并预留了，录出来超了，那是框架算错了，不该让用户背锅。

第二个，录制期间关掉 Python 的循环垃圾回收（`capture_graph` 里的一段）：

```python
# check-docs: skip: excerpt from pllm/execution/step_capture.py
collecting = gc.isenabled()
gc.disable()
try:
    with torch.cuda.stream(stream):
        restore()
    # Cyclic collection can release tensors from another CUDA stream and invalidate capture.
    # Torch's context collects before starting; postpone automatic collection until capture ends.
    pool = torch.cuda.graph_pool_handle() if pool is None else pool
    with conditional_graph_pool(pool), torch.cuda.graph(graph, stream=stream, pool=pool):
        prepare()
        result = body()
finally:
    if collecting:
        gc.enable()
```

循环 GC 可能在任何时候醒来，释放某个属于别的 CUDA stream 的张量；要是正好发生在捕获窗口里，捕获就失效了。PyTorch 的 `torch.cuda.graph` 自己会在开始前回收一次，但拦不住中途再来一次。两行注释，讲清楚一个只在真实负载下偶尔出现的问题。

训练体本身就是普通的 eager PyTorch：按槽号从池里 gather 字段（哑槽填 0），在 `torch.enable_grad()` 下跑用户的方法、求梯度，再由优化器写回。FP32 的 AdamW 走一个分页的 Triton kernel，直接从源槽读、往目标槽写，连写时复制那一遍拷贝都省了。

效果如何？2026 年 10 月这套通用捕获落地时，0.5B 的 ESamp（共享的 RTX 4090，128 并发，rank 4）每步从 6.73 ms 降到 5.25 ms，提交 token 那一段从 3.32 ms 降到 1.56 ms，吞吐从 19,100 提到 23,829 tok/s，约 25%（基线取自当时的规格说明，不是同时重跑的；历史测量）。之后的 profile 里，每个 decode 步恰好回放一次训练图，从不重录。

还有个今昔对比可以讲。旧 tLLM 为了让训练能在 graph 里可靠回放，曾经把优化器从 AdamW 换成了 SGD（优化记录第 21 步：“capturable model-bank optimizer 改用 SGD……graph capture 不再依赖 AdamW state 初始化，replay 更可靠”）。今天 AdamW 的五个字段都住在固定的池里，那个妥协不需要了。

## 插曲：我们试过编译它，结果更慢

捕获做完之后，一个很自然的念头冒了出来：训练步里还有几十个零碎的 kernel，PyTorch 2 又主推 `torch.compile`，为什么不把整个训练体编译一下，让 Inductor 去融合？

我们试了两次（RTX 4090，vLLM 0.27.1，Qwen2.5-0.5B，128 行 × 256 token，rank 4，CUDA graph 开；当时 β=0 的基线是原生的 92.45%）。

第一次只编译作者写的 loss 前向和反向：7 轮中位数 36,013.5 tok/s，是同次原生参考的 92.77%，和不编译没有可测的差别。回头想并不奇怪：训练本身的计算量很小，发射开销已经被 graph 吃掉了，把两三个逐元素 kernel 合成一个，省下的时间淹没在噪声里。

第二次把整个被录制的执行体——gather、loss 归约、梯度路由、优化器准备——都交给编译器，录图仍由 StepCapture 负责。5 轮中位数 35,144.6 tok/s，只有原生参考（39,005.3）的 90.10%。更慢了。profile 显示编译器确实“立了功”：旁路流每步的 kernel 从 59 个降到 25 个；可旁路 GPU 时间却从约 178 µs 涨到约 257 µs。多出来的是一个新 kernel，`triton_poi_fused_clone_paged_adamw_slice_0`，每步约 39 µs；分页 AdamW 也从约 29 µs 涨到约 64 µs。

原因是 functionalization（函数式化）。AOTAutograd 在编译前会把所有原地修改改写成“产生新张量”的纯函数形式，这是它做图变换的前提。而我们的状态池恰恰是靠原地写页槽工作的。编译器看到对池切片的原地写，为了守住自己的语义，就把池复制了一份——名字里那个 `clone` 就是证据。我们又用 `torch.compiler.disable` 把优化器那一段排除在外，trace 里的复制还在，于是也放弃了。

这次失败的教训比成功更具体：对已经录进 graph 的一串小 kernel，编译器能省的发射开销已经没了，能融合的收益很小；而“函数式化”和“原地写页池”是两种相反的世界观，硬凑在一起会互相拖累。所以用户文档里直说：不要对 workload 方法手动套 `@torch.compile`。pLLM 在别处仍然用编译器——采样尾声的词表重加权、只读查表的整图融合、挂载点的方法体、参数覆盖层的梯度——那些地方都不碰状态池的原地写。

同一次调查还留下一个相关的负面结果：我们以为“采样器前面少等几次”能让训练和更多东西重叠，于是去掉了几个等待；交叉测量从 94.49% 掉到 92.73%。更多的重叠把训练挤进了带宽受限的采样阶段，两边互相拖慢。重叠不是免费的，[《自动放置与重叠》](11-automatic-placement-overlap.md)专门讲这件事。

## 这样做的代价

这套做法在几个地方明显比 eager 差，值得说清楚。

最直接的是空位也要算。R=128 而这一步只有 16 个请求时，其余 112 行照样跑完前向、反向和归约，只是权重为 0。低负载时这是纯粹的浪费，而能动态改形状的 eager 代码没有这个问题。每个任务还有一个私有的 graph 显存池，大小按最大形状保守推导，不会在空闲时还给 KV cache，预算不够就在启动时报错。第一次用到时要预热、录制，`capture_graph` 开头有一次 `stream.synchronize()`，第一个请求会感到一次抖动；hidden 缓冲换了存储，也会再录一次。

写法上的限制也是从这里来的。“loss 小于 0.01 就跳过”这种按内容分支的逻辑写不进录好的图，只能写成 `torch.where` 一类的掩码计算。方法必须逐行独立，每行返回一个 loss，不能自己对行求均值，也不能用 BatchNorm 或者跨行注意力——否则“空位权重为 0”这个前提就不成立了。这不是品味问题，是固定形状直接推出来的。另外，训练图还没有和模型图合并，相邻任务之间还有表的准备、结果拷贝、版本的锁定与释放。

规模决定了这套东西能做到多好。同样是 RTX 4090、vLLM 0.32、128 行、256 token、CUDA graph 开：0.5B 原生一步约 3.0 ms，ESamp β=0 在 `publish_lag=1` 时是原生的 91.9%（`publish_lag=0` 是 89.8%）；7B 原生一步约 21.7 ms，ESamp 是 97.4%（`publish_lag=0` 是 97.2%）。7B 的 profile 里，旁路流上的训练每步约 616 µs（分页 AdamW 约 191 µs），只占一步的 3% 左右；0.5B 一步只有 3 ms，表的准备、拷贝、图回放这些固定开销就显眼了。我们没有办法让一个额外的训练步在 3 ms 的 decode 里完全消失，只能让它不同步、不重录、尽量重叠。

## 可以带走的东西

回头看，每一步都是被同一个事实推出来的：Python 只活在录制时。地址要固定，于是有了池和表；形状要固定，于是有了哑状态和逐行的 loss；不能同步，于是选择留在 GPU 上、上传走锁页内存；外部张量会搬家，于是地址本身也成了表里的数据。

如果要从这里带走一个可以用在别处的想法，那就是：**把会变的东西变成数据，而不是变成结构。**图是死的，表是活的。凡是每一步会变的东西——哪些行、哪个版本、甚至一个张量住在哪里——都不要做进图的结构里，而是放进图要读的数据里。很多看起来需要“重新编译”或“重新录制”的问题，换个角度，就只是多写一张表。
