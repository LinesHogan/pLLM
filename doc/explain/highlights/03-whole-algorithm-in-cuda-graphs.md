# Python 只在录制时活着：固定地址、一张表、零同步

写惯了研究代码的 PyTorch 用户，脑子里大多有一个非常舒服的心智模型：代码是一行一行往下走的。

你写一句 `hidden = layer(x)`，GPU 就跑一层前向；你写一句 `loss.backward()`，GPU 就去算一遍梯度；中间你想看看数值对不对，随手塞一句 `print(loss.item())`，或者写个 `if loss < 0.01:` 跳过更新，一切都顺理成章。在这个世界里，Python 解释器站在时间线的最前头，代码执行到哪一行，底层的硬件就跟到哪一步。

如果你只跑单卡的离线脚本，这个心智模型没有任何问题。但当你试图往生产级推理引擎（比如 vLLM）的解码循环里加一点东西时，你会发现脚下的地板突然空了。

假设你想在解码的每一步里，从模型第 0 层取出刚算好的隐藏状态（`hidden`），喂给一个轻量级的小网络做一次前向预测，赶在采样器抽样之前把修正信号送过去。照着研究代码的习惯，最自然的做法是往第 0 层挂一个 PyTorch 的 `register_forward_hook`，在钩子里拿到激活值，顺手在宿主端发起一次旁路流上的预计算。

如果你真的这么写，并放进开启了 CUDA graph 的引擎里加上探针去跑，你会看到一个让人发愣的现象：在引擎刚启动预热的那一步，这个钩子确实执行了一次；可一旦进入后面连续几十步的稳态逐字生成（decode），钩子里的调度逻辑命中的次数是 `0`。模型在外头热火朝天地一个字接一个字往外蹦，而你挂在模型层上的那段 Python 函数，就像在预热结束的那一秒人间蒸发了一样，再也没有被调用过。

为什么明明挂在模型层上的 Python 代码会突然失效？如果我们要放进每个解码步的还不只是一次前向，而是一整套在线学习算法——从读取每个请求专属的权重，到算损失、反向传播求导，再用 `AdamW` 更新一阶和二阶动量写回显存——又该怎么让它在没有 Python 在场的世界里跑起来？

我们不妨先停下来算一笔硬件时间账，然后从最简单的单请求代码开始，一步一步把这个动态算法装进静态的图里。

---

## 一、 为什么解码步容不下 Python？

要弄明白钩子为什么会消失，得先看看 GPU 在自回归解码阶段过着什么样的日子。

在 PyTorch 默认的即时执行（Eager）模式下，每当你调用一个算子，CPU 都要依次走过 Python 解释器、PyTorch 算子分发器（Dispatcher）和 CUDA 驱动，最后向 GPU 的硬件队列发射（launch）一个内核。这套动作哪怕什么都不算，光在 CPU 上就要花掉好几微秒。

在处理长提示词的预填充（prefill）阶段，几千个 token 拼成一块大矩阵送进 GPU，一个矩阵乘内核一跑就是几毫秒，CPU 花几微秒发一道指令根本不算什么。但到了逐字生成的 decode 阶段，情况彻底反过来了：哪怕把 128 个并发请求拼成一批准入，每个请求每一步也只产生 1 个新 token。

我们可以拿一套具体的基准设置来量一量这个时间尺度：在单张 NVIDIA RTX 4090（24 GiB）上运行 vLLM 0.32.0（开启 CUDA graph 与 FlashInfer 采样器），跑 128 个并发序列（8 条提示词 × 每条采样 `n=16`）、每条生成 256 个 token。对于 `Qwen2.5-0.5B-Instruct`（BF16）这样轻快的模型，GPU 跑完一整步 24 层 Transformer 的前向加采样，总共只需要 **3.0 ms**（吞吐 42,355 tok/s）。

然而，在这短短的 3.0 ms 里，一座二十几层的模型前前后后要发射几百个细碎的 GPU 内核。几百个内核乘上每次几微秒的 CPU 发射开销，本身就是毫秒量级——也就是说，CPU 喊口令的时间跟 GPU 真正算数的时间几乎一样长。如果让 Python 在每一步里现场逐个发射算子，GPU 算完一个内核扭头一看，CPU 还在慢吞吞地准备下一个内核的调用栈，整张显卡一半的时间都在空转等待。在同样的 0.5B 基准下，只要关掉图模式退回 eager 执行，单步耗时就会拉长到 6.1 ms，吞吐直接跌到 20,926 tok/s，腰斩了一半（即便在单步计算长达 21.7 ms 的 `Qwen2.5-7B-Instruct` 上，肥厚的矩阵乘掩盖了大部分主干发射开销，使关掉图模式的原生吞吐仍能维持在 5,773 对 5,894 tok/s 的 98% 左右，一旦扩展逻辑自己要在每步里额外发射几十个零碎的训练小算子，CPU 发射瓶颈依然会暴露出来）。

```text
Eager 模式（每一步由 CPU 现场逐个发射内核）：
  CPU:  [Python + 发射 k1][Python + 发射 k2][Python + 发射 k3]...
                        \                 \                 \
  GPU:                   [k1] (空转等 CPU)  [k2] (空转等 CPU)  [k3]

CUDA Graph 模式（预热时录制一次，运行期单次指令重放）：
  录制 (1 次):  CPU 跑一遍 Python 与钩子 ──► 录下 GPU 内核序列与裸指针
  回放 (每步):  CPU: [ cudaGraphLaunch ]
               GPU: [k1][k2][k3][k4][k5][k6]... (首尾相接，Python 不再参与)
```

为了砍掉这笔发射税，推理引擎在启动预热时会动用 **CUDA Graph**。预热时，CPU 照常跑一遍模型的 Python `forward` 代码，但 GPU 并不真的去算业务数据，而是把这一路上收到的所有内核名称、线程网格尺寸、以及每个张量参数的**裸显存地址**，原封不动地记录成一张静态的执行图。等到真正服务请求时，引擎在稳态 decode 步直接绕过模型的 Python `forward` 函数，每一步只调一次极轻量的 `cudaGraphLaunch`，让 GPU 顺着录好的图把几百个内核一口气放完。

现在回头看开头那个“失踪”的钩子，原因就非常清楚了。

录制只在预热时发生过一次。在那唯一的一次执行里，钩子函数确实跑了，但录像机眼里只有发往 CUDA 驱动的 GPU 内核调用。至于你在钩子里写的 Python 变量修改、字典查询、或者在宿主端动态发起的任务调度，全都发生在 CPU 内存里，图上连半个比特都没记下来。到了随后的解码回放步，Python 解释器根本没有进场，藏在层钩子里的宿主逻辑自然也就永远停在了过去。

所以，凡是需要由宿主端协调的前向收尾工作，都不能埋在会被图录制跳过的模型内部层钩子里，而必须放在模型图回放刚刚结束、采样器还没启动的接缝处——也就是模型外层的 `compute_logits` 入口（同时只让模型层在图内把所需的隐藏状态顺手拷进固定地址的缓冲区）：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/runner.py
def install_model_hooks(self) -> None:
    """Install the logits hook and announced hidden taps before vLLM compiles the model."""
    model = self.decoder.model
    model.compute_logits = self._compute_logits
    # ……
```

把前向触发点挪到 `compute_logits` 解决了“找回控制权”的问题。但紧接着，更棘手的问题来了：我们接下来要跑的那套在线训练算法本身，该怎么执行？

---

## 二、 为什么在线训练也必须进图，且同步次数必须是“零”？

先看最简单的单请求情况。假设某一步解码拿到了浅层隐藏状态 `shallow` 和深层隐藏状态 `deep`，我们要用它们在线更新当前请求专属的小网络参数 `params`：

```python
# check-docs: skip: illustrative sketch of one request's training step, not pLLM code
params = states[request_id]
loss = (f(params, shallow) - deep).square().mean()
grads = torch.autograd.grad(loss, params)
adamw_step(params, grads, moments[request_id])
```

在 Python 里看，这只是短短四行代码。但在 GPU 底层，这四行代码会展开成五六十个零碎的小内核：收集参数、两层线性映射、相减、平方、求均值、反向传播的一连串梯度算子、再加上 `AdamW` 里更新一阶矩、二阶矩和权重的逐元素操作。

如果你让这五六十个小内核在 CUDA graph 外面以 eager 模式跑，刚才我们在模型主干上躲开的 CPU 发射瓶颈，就会在训练步上原样重演。在 0.5B 模型（vLLM 0.27.1，128 并发，rank 4）的剖析中，如果不把训练步录进图里，CPU 光是逐个发射这几十个训练小算子（其中反向传播一段就占了训练宿主耗时的 36%），每步就会平白多出几个毫秒的宿主开销——比 0.5B 模型一步 3.0 ms 的 GPU 前向时间还要长。

因此，**不能只让模型前向录图，整个在线训练步——从收集状态、前向计算、反向求导到优化器写回——也必须完整地录进一张 CUDA graph 里。**

而一旦决定把算法录进 CUDA graph，我们就必须同时满足它的三条硬件铁律：
1. **显存地址必须固定**：图里每个内核参数绑定的都是录制那一刻的物理显存指针；
2. **张量形状必须固定**：每个内核的网格尺寸与循环边界在录制时就已经钉死；
3. **零主机同步（Zero Host Synchronization）**：录制期间任何 CPU 等待 GPU 的操作（如 `.item()`）都会直接导致捕获失败；而在运行期的热路径上，哪怕图外残留一次同步，也会把异步流水线彻底打断。

在动手处理地址和形状之前，值得先算一笔关于“第三条铁律”的账。因为在写系统代码时，人们很容易产生一种直觉：如果每步有十几次 CPU-GPU 同步，我优化掉其中的九成、只留一两次 `.item()` 查一下状态，性能应该已经恢复九成了吧？

恰恰相反。在异步推理流水线里，**少几次同步几乎什么也换不来，只有当同步次数真正变成零的那一刻，吞吐才会发生跳变。**

原因在于 CPU 和 GPU 的流水线重叠方式。在健康的稳态解码步里，当 GPU 正在埋头执行第 $t$ 步的算子时，CPU 并没有闲着，它已经领先半步，在宿主端准备当前步稍后或下一步所需的元数据了。在 RTX 4090、0.5B 模型（128 并发，vLLM 0.27.1 时期）的剖析中，pLLM 每步在 CPU 上的准备工作总共需要约 **700 µs**，最后通过一次耗时仅 **6.24 µs** 的非阻塞拷贝上传到 GPU。只要没有任何阻塞，这 700 µs 的 CPU 工作完完全全藏在约 3 ms 的 GPU 执行阴影背后，GPU 在相邻两步之间的空闲间隙只有 **4–6 µs**。

```text
零同步流水线（700 µs 的 CPU 准备工作完全藏在 GPU 执行阴影后）：
  CPU:  [ 准备第 t 步 (700 µs) ][ 准备第 t+1 步 (700 µs) ]
                               \                        \
  GPU:   [   第 t-1 步 GPU   ]──►[    第 t 步 GPU (~3 ms)    ] (间隙仅 4-6 µs)

残留哪怕 1 次同步（CPU 被迫等队列排空，700 µs 直接暴露在关键路径上）：
  CPU:  ...(阻塞等 GPU)... ──► [ 准备下一步 (700 µs) ]
                             ▲                       \
  GPU:  [   第 t 步跑完   ]───┘ (GPU 空转干等 700 µs!)  ──► [ 第 t+1 步 ]
```

现在，假设你在整步代码的某个角落里留了哪怕**一次**主机同步（比如把一个布尔标志拷回 CPU 看一眼）。为了拿到这几个字节，CPU 必须立刻停下脚步，等待 GPU 把队列里积压的所有算子全部跑完。等 GPU 终于排空队列把结果递给 CPU，CPU 这才开始跑那 700 µs 的准备工作——而在这 700 µs 里，GPU 的任务队列是空的，整张显卡只能干坐着等 CPU。少五次同步还是留一次同步，那 700 µs 的宿主耗时都会结结实实地暴露在关键路径上；只有当最后一次同步被拔除（在一版较早的 0.5B、vLLM 0.27.1 图模式实现上，这一步让吞吐从约 11.6k 跃升至 19.1k tok/s），CPU 才能重新跑在 GPU 前头。

立下了“地址固定、形状固定、零同步”三条规矩，我们来看看怎样把每步都在变化的在线训练装进图里。

---

## 三、 逐步演进：把动态世界折叠进一张 `int64` 表

如果系统里永远只有 1 个请求，给训练录图不需要任何技巧：这个请求的参数就放在固定的显存地址上，把前向、反向和优化器跑一遍录下来，之后每步回放即可。

但只要进入真实的连续批处理环境，麻烦就会一件接一件找上门来。我们每次往系统里加一个真实约束，看看数据结构会如何跟着生长。

### 第一道坎：从 1 个请求到多个请求——用槽号代替指针

当系统里同时有多个请求时，请求 A 的权重在地址 `0x1000`，请求 B 的权重在地址 `0x2000`。在连续批处理里，这一步 batch 的第 0 行可能是请求 A，下一步请求 A 结束了，第 0 行换成了新来的请求 C。如果图里的内核直接绑定某个请求的张量地址，下一步换人时图就废了。

解法是系统设计里最经典的**间接寻址（Indirection）**。

我们不再为每个请求单独分配零散的张量，而是在启动时预先分配一块基地址永远不变的连续大张量——也就是**分页状态池（Slot Pool）**，形状为 `[capacity, *field_shape]`（关于这套池子如何通过不可变版本号管理并发与回滚，将在下一篇[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)中详细展开）。请求 A 住在 0 号槽，请求 B 住在 1 号槽。

录制 CUDA graph 时，图里的内核只绑定这个大池子的固定基地址；至于“当前这一步，第 $r$ 行应该去几号槽读旧参数（`sources`）、算完的新参数又该写进几号槽（`destinations`）”，我们把它做成一张同样地址固定的 `int64` 索引表。每一步开始前，CPU 把当步的槽号填进表里上传到 GPU，图里的内核先从表里读出整数槽号，再去池子里按槽号读写。请求怎么换、版本怎么变，改的都只是表里的整数。

这里顺便提一个容易踩中的底层细节：既然每步都要把 CPU 填好的小表拷到 GPU 上，能不能直接调 `torch.tensor(values, device="cuda")`？

不能。普通的 Python 列表或默认 CPU 张量分配在操作系统的**可分页内存（Pageable Memory）**里。由于操作系统随时可能把这页内存换出到磁盘，GPU 的硬件 DMA 引擎无法直接异步读取它——从可分页内存往 GPU 拷贝数据，会在底层迫使主机等待流。要想做到真正的零阻塞上传，数据必须先落进**锁页内存（Pinned Memory）**：

```python
# check-docs: skip: excerpt from pllm/execution/cuda_stream.py
def upload(
    values: Sequence[float] | Sequence[Sequence[float]] | npt.NDArray[np.int64],
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    # ……
    host = torch.tensor(values, dtype=dtype)
    return host.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else host.to(device)
```

而在每一步都要执行的 `TensorExecutor` 里，连每次临时调用 `.pin_memory()` 向缓存分配器申请内存的开销都可以省掉：`PinnedStaging` 维护了一组预先锁好页的主机缓冲区，每步由 Rust 编写的 `_core.fill_table` 直接往锁页数组的 NumPy 视图里填数，发起 `non_blocking=True` 拷贝后在流上记录一个 `cuda.Event`。下一步要用缓冲时，只调用非阻塞的 `event.query()` 看一眼上次拷完没有——拷完了就原地复用，还在路上就顺手添一块新缓冲，CPU 永远不停下来等流。

### 第二道坎：请求数量每步都在变——第 `S` 号哑状态（Padding Sink）

有了状态池和槽号表，固定数量请求的轮换解决了。但线上的并发请求数是随时起伏的：图是按最大容量（比如最大行数 $R=128$、最大状态数 $S=128$）录制的，如果某一步实际上只来了 30 个请求，剩下的 98 行空位怎么办？

对无状态的模型前向来说，可以按 `batch_size = 1, 2, 4, 8...` 分档录制一组图。但对有状态的训练任务来说，影响形状的既有行数 $R$ 又有状态数 $S$，而且每录一张图，CUDA 都要为这张图的中间激活与反向梯度保留一个私有的显存池（Graph Pool）。为训练任务也分档录一组图，会把宝贵的 KV cache 显存吃光。

怎么只按最大容量 $(R, S)$ 录**唯一的一张图**，却能安全地跑任意小于等于 $R$ 的请求数？

做法是在 $S$ 个真实状态槽位之外，故意把读取表做宽一列，变成 $S+1$ 列（下标 $0 \dots S$）。其中 $0 \dots S-1$ 给真实请求用，而多出来的第 $S$ 列专门用作吸收所有无效行的**“哑状态”（Padding Sink）**：

```text
每步索引表 (固定录制形状 R=128)           状态池寻址与哑状态隔离
┌───────────────────────────┐
│ Row 0   (active=1, st=0)  │─────────► State 0   [ 读 sources / 写 destinations ]
│ ...                       │           ...
│ Row 29  (active=1, st=29) │─────────► State 29  [ 读 sources / 写 destinations ]
├───────────────────────────┤
│ Row 30  (active=0, st=S)  │──┐
│ ...                       │  ├──────► State S (第 S 号哑状态 Padding Sink):
│ Row 127 (active=0, st=S)  │──┘        • sources [F, S+1]   -> 槽号填 -1 (读出全 0.0)
└───────────────────────────┘           • loss 与梯度归约     -> active=0 (权重为 0.0)
                                        • destinations [F,S] -> 只有 S 列，无第 S 列!
```

当这一步只有 30 个真实请求时，CPU 把第 $30 \dots 127$ 行的状态编号统一填成 `S`，并把它们的 `active` 标志填成 `0`：
1. **读取时**：`sources` 段的形状是 `[F, S+1]`，第 `S` 列的源槽号固定填 `-1`。负责从池里取参数的 `_gather_field` 看到槽号小于 0，直接把对应位置填成 `0.0`：
   ```python
   # check-docs: skip: excerpt from pllm/execution/tensor_executor.py
   def _gather_field(pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
       """Gather one field's state pages, filling padding slots with zero."""
       return pool.index_select(0, slots.clamp_min(0)).masked_fill(
           (slots < 0).reshape(-1, *(1 for _ in pool.shape[1:])), 0
       )
   ```
2. **求导与归约时**：虽然 GPU 里的内核依然算满了 128 行，但因为填充行的 `active=0`，在计算损失归约和按状态累加梯度时，它们贡献的权重严格为 `0`，绝不会污染任何真实请求的梯度。
3. **写回时**：`destinations` 段的形状是 `[F, S]`——比 `sources` 少了一列，压根没有第 `S` 列。优化器只更新前 $S$ 个真实状态，哑状态永远不会被写回。

### 第三道坎：动态条件筛选——不回传掩码，照发“原样版本”

很多在线学习算法并不是每一步对所有请求都无条件更新，而是允许通过 `rows=` 或 `update=` 选择器传入一个设备端的布尔掩码（例如只在满足特定条件的行上训练并更新权重）。

如果是在 eager 模式下，你大概会先把 GPU 上算出的布尔掩码拷回 CPU，看看哪几个状态被选中了，然后只为被选中的状态分配新版本页槽。但在图模式下，把掩码读回 CPU 就意味着一次主机同步，直接违反了第三条铁律。

如果 CPU 不去读这个掩码，负责在宿主端管理版本账本的 PTTS 怎么知道这一步哪几个状态变了呢？

答案是：**宿主端根本不需要知道。**

只要某一步批里有 30 个活跃状态，CPU 账本就干脆利落地假定这 30 个状态在本步**全部发布新版本**，统统为它们分配好目标槽位。真正的筛选完全留在 GPU 内部完成：

```python
# check-docs: skip: excerpt from pllm/execution/tensor_executor.py
def _selection(self, rows: FixedRows) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Which rows train, which states update and each state's last trained row (-1: none), on the device.

    `rows=` selects the trained rows; `update=` (or else the trained rows) selects the states whose optimizer
    fields change. Every other state publishes its fields unchanged, so selectors never sync the host.
    """
    policy, states = self.task.policy, rows.index.states
    trained = rows.active & self.inputs[policy.forward_rows] if policy.forward_rows else rows.active
    update_rows = self.inputs[policy.update_rows] if policy.update_rows in self.task.domains else None
    updates = None
    if policy.forward_rows or update_rows is not None:
        chosen = trained if update_rows is None else rows.active & update_rows
        count = states.new_zeros(rows.states + 1).index_add_(0, states, chosen.long())
        updates = count[: self.state_count] > 0
    # ……
    return trained, updates, last
```

通过 `index_add_`，GPU 自己在设备端算出每个状态在本步有没有被选中的行（`updates = count[: self.state_count] > 0`）。随后在写回目标槽时，对于 `updates[s]` 为真的状态，优化器写入更新后的新参数；对于 `updates[s]` 为假的状态，内核直接把它的旧字段**逐字节原样拷贝**进新槽位。用显存里一次极低成本的页内拷贝，换掉了整条流水线的零主机同步。

### 第四道坎：外部张量每步都在搬家——把裸指针当 `int64` 写进表里

到现在为止，凡是归扩展层自己管理的状态池和缓冲区，地址全都固定下来了。但训练任务有时还需要读取宿主引擎（vLLM）自己分配的张量——比如采样器传过来的张量或当前步的 token 输入。vLLM 在每一步都可能重新分配这些张量，它们的显存地址每步都在变。

如果把这些外部张量直接作为参数传进被录制的图里，图就会死死记住录制那一刻的旧地址，下一步回放时直接读飞。

怎么在一张连指针都被焊死的 CUDA graph 里，去读一个每步都在换地址的外部张量？

再加一层间接：既然图里的内核只认固定地址，那我们就**不要把那个外部张量本身录进图里，而是让 CPU 在每一步调一次 `tensor.data_ptr()`，把拿到的 64 位显存地址连同它的两个维度步长（strides）当作三个普通的 `int64` 整数，写进我们那张固定地址的表里。**

到了 GPU 上，Triton 内核 `_gather_rows` 从固定表（`descriptor`）里读出这三个整数，当场把第一个整数强转回设备指针去解引用：

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

看 `address = tl.load(descriptor).to(tl.pointer_type(...))` 这一行。对 CUDA graph 的录像机来说，它录下的内核参数是 `descriptor`——而 `descriptor` 是固定表里的一个切片，地址从头到尾都没变过。录像机记下的指令仅仅是“去这个固定地址读三个 64 位整数，再顺着第一个整数指的方向去搬数据”。至于那个整数每一步指向哪里，那是运行时流过的数据，不是图的结构。

```text
固定地址表中的描述符切片 (地址恒定)          每步动态重分配的外部张量
┌──────────────────────────────────┐
│ descriptor[0]: 0x7f8812340000    │ ──► .to(tl.pointer_type(...))
│ descriptor[1]: 3584 (stride 0)   │             │
│ descriptor[2]: 1    (stride 1)   │             ▼
└──────────────────────────────────┘     ┌───────────────────────────────┐
                                         │ 0x7f8812340000: [row 0][row 1]│
                                         └───────────────────────────────┘
```

现在，把这四步演进攒出来的零件合在一起，就是 `pllm/execution/tensor_executor.py` 里定义的四段式共享表 `TableLayout`：

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

    rows: int
    states: int
    fields: int
    row_inputs: int
```

整张表里的四个段落，不多不少正好对应刚才的四个问题：`rows` 交代每行的身份与有效掩码（见[《行模型》](04-packed-rows.md)），`sources` 和 `destinations` 交代状态池的读写槽位并用相差一列的宽度隔离第 `S` 号哑状态，`row inputs` 装下外部动态张量的裸地址与步长。

在 RTX 4090、0.5B 模型（128 并发，ESamp $\beta=0$，vLLM 0.27.1 时期）的剖析中，每一步上传这张共享表总共只需拷贝 **23,692 字节**，耗时 **6.24 µs**（同期原生 vLLM 自己每步上传元数据为 1,088 字节、2.50 µs）。每步花 6 微秒更新这二十几 KB 的 `int64` 整数，整张训练图就再也不需要因为请求进出、版本更替或外部张量搬家而重录。

---

## 四、 按下录音键时的两个底层陷阱

表和缓冲池都就位后，负责把执行体录进 CUDA graph 的是 `pllm/execution/step_capture.py` 里的 `StepCapture`。它的主体代码只有一百来行，但里面处理了两个只有真正把“带原地状态写入的图捕获”放进多流系统里才会撞上的陷阱。

第一个陷阱是：**录制前的预热（Warmup）会写坏线上状态。**

在正式捕获一张 CUDA graph 之前，常规做法是在私有流上先把函数预热几遍（`capture_graph` 里用 `for _ in range(3)` 跑三遍），让 cuBLAS、Triton 等底层库完成工作区初始化和内核加载。但别忘了，被录制的 `body` 末尾跟着一个会往 `destinations` 槽位里写新权重的优化器。如果在运行中途触发捕获时直接跑三遍预热，状态池里正在服务线上请求的真实参数就会被预热产生的垃圾梯度覆盖三遍。

看看 `StepCapture._capture` 是怎么用一行代码解决这个问题的：

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

在进入 `capture_graph` 之前，先调一句 `destinations.fill_(-1)`，把表里所有的目标写槽统统涂成 `-1`。

因为无论是分页拷贝内核 `_copy_pages` 还是分页 `AdamW` 内核，内部都有一条 `dst >= 0` 的掩码判断，看到目标槽号小于 0 就会跳过最终的存储指令。于是，三次预热外加一次录制，整整四遍跑下来，所有的算子和中间工作区都完成了分配与捕获，但对于状态池而言，这四遍执行完完全全是**只读的**。等录完图回到 `StepCapture.run`，再调用一次 `refill()` 把当步真正的目标槽号填回表里，然后执行 `self.graph.replay()`。（顺带看一眼最后那行 `assert`：图池占用的显存已在引擎启动的规划期用最大形状追踪并预留了预算，如果实际捕获超出了预留字节数，直接作为框架内部错误断言拦截，绝不静默侵占 KV cache 的空间。）

第二个陷阱藏在 `capture_graph` 的捕获窗口前后：

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

为什么在录制 GPU 图的时候，必须把 Python 的循环垃圾回收器关掉（`gc.disable()`）？

因为 Python 的循环 GC 是按对象分配计数随时自动触发的。当 CPU 正在 `with torch.cuda.graph(...):` 块里一行行执行 Python 代码、创建各种中间张量包装对象时，GC 随时可能因为对象分配达到阈值而在半路醒来，顺手回收一批早已不可达的循环引用对象。如果那批死对象里恰好包含一个曾经在**另一条 CUDA stream** 上分配或使用过的旧张量，它的 C++ 析构函数就会在此时向 CUDA 缓存分配器发出跨流释放通知——而正在录图的 CUDA 驱动看到捕获窗口里突然插进来一个跨流操作，会当场报错中止捕获。虽然 PyTorch 自己的 `torch.cuda.graph` 上下文会在进入捕获前主动调一次 `gc.collect()`，但那只能清理进入前的旧垃圾，拦不住捕获中途因为新建对象再次触发自动回收。在捕获窗口外围显式关掉自动 GC，才能彻底堵住这个偶发的崩溃源。

至于被录进图里的训练体 `execute_body` 本身，反倒不需要任何花哨的图编译器技巧：它就是普通的 eager PyTorch 代码。先用编译好的 `_cuda_gather` 按 `sources` 槽号从池里捞出参数，在 `torch.enable_grad()` 下跑一遍用户写的函数，调 `autograd` 求导，最后交给优化器写回。对于最常用的 FP32 `AdamW`，分页 Triton 内核（`pllm/kernels/adamw.py`）直接从 `sources` 槽位读取旧权重与一阶、二阶动量，在寄存器里算完更新后直接写进 `destinations` 的新槽位——连写时复制（Copy-on-Write）事先拷一遍旧页的步骤也一并省掉了。

在单卡 RTX 4090、0.5B 模型（vLLM 0.27.1，128 并发，rank 4，一版较早的实现）上对照：不捕获训练步时单步总耗时 6.73 ms（其中提交 token 触发训练的那一段 3.32 ms），捕获后降到 5.25 ms（那一段 1.56 ms），端到端吞吐从 19,100 tok/s 提升到 23,829 tok/s（+25%）；在一段 765 个稳态解码步的剖析轨迹里，每一步恰好触发 1 次训练图回放，重录制次数为 `0`。

---

## 五、 为什么不用 `torch.compile` 编译训练步？

看到这里，一个很自然的问题会冒出来：既然训练体里包含了前向、损失归约、反向求导等五六十个细碎的逐元素小内核，为什么我们只用 CUDA graph 把这些 eager 算子原样录下来，而不顺手给训练体套上一层 `@torch.compile`（Inductor 编译器），把几十个小内核融合成几个大内核呢？

直觉上，“算子融合 + 图捕获”应该比单纯的“未融合算子 + 图捕获”更快。但当我们在 RTX 4090、0.5B 模型（vLLM 0.27.1，128 并发，rank 4，ESamp $\beta=0$，未编译时吞吐为原生的 92.45%）上做完对照测量后，数据给出了相反的结论：

1. **只编译用户写的 loss 前向与反向函数**（外围的填表、收集与分页优化器保持不变）：7 轮中位数吞吐是原生的 **92.77%**，与不编译相比没有任何可分辨的差异。
   原因其实不难理解：当一段小算子序列已经被录进 CUDA graph 之后，原本占大头的 CPU 逐算子发射开销已经被 `cudaGraphLaunch` 消除了；而 rank 4 小网络的中间激活张量一共只有几百 KB，很可能本来就还留在 GPU 的 L2 cache 里，把两三个逐元素算子融合成一个内核省下的那点显存往返，在端到端时间上几乎听不见响。
2. **把整个被 `StepCapture` 捕获的 `execute_body`（含参数收集、损失归约、梯度路由与优化器）交给 `torch.compile` 编译后再录进图**：吞吐不仅没涨，反而倒退到了原生的 **90.10%**。

打开剖析器看旁路流的时间线，就能看清为什么“越编译越慢”：编译之后，旁路流每步发射的内核数量确实从 59 个大幅减少到了 25 个；但旁路流每步消耗的 GPU 总时间，却从原来的约 **178 µs** 暴涨到了约 **257 µs**。

多出来的近 80 µs 花在了哪里？时间线上多出了一个名叫 `triton_poi_fused_clone_paged_adamw_slice_0` 的内核，每步单枪匹马吃掉约 39 µs；同时，紧随其后的分页 `AdamW` 内核耗时也从约 29 µs 膨胀到了 64 µs。

名字里的那个 `clone` 道出了病根。PyTorch 编译器底层的 AOTAutograd 依赖一套叫作 **Functionalization（函数式化）** 的机制：在对计算图做变换与融合之前，编译器要求图里的所有操作都必须是纯函数（Pure Functions），不允许存在对输入张量的原地修改（In-place Mutation）。而我们的状态池与分页优化器，核心设计恰恰是“在一块预先分配的巨大张量池里，原地覆写指定的目标页槽”。

当编译器看到执行体试图原地修改状态池切片时，为了维护它内部的函数式不变量，它会自动插入一道防守性拷贝，把状态池切片**克隆（clone）一份副本**出来，在副本上算完再拷回去；即便用 `torch.compiler.disable` 把优化器调用单独圈出去，追踪图边界上的保护性复制依然存在。

这组对照实验揭示了一条很实用的边界：**“纯函数式图编译”与“固定地址池上的原地分页覆写”在底层假设上是冲突的。** 对于已经录进 CUDA graph 的短算子链，发射开销已死、中间数据已在缓存，编译器融合的边际收益极小，而函数式化引入的防御性克隆却会实打实地浪费显存带宽。因此，在 pLLM 里，逐行训练任务直接采用“固定容量 eager 自动微分 + CUDA graph 捕获”，而把 `torch.compile` 留给那些真正纯函数式、或需要跨算子消除大张量往返的场景——比如[《图内挂载点》](05-in-graph-attach-points.md)里的挂载点方法体、[《参数覆盖层》](06-parameter-overlays.md)里的块内梯度、以及[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)里的只读查表融合。

---

## 六、 固定形状的代价与不同规模下的账单

把所有动态变化都收进那张 `int64` 表之后，我们得到了一个在稳态解码期间零重录、零主机同步的静态训练图。但系统设计里没有白捡的便宜，用“按最大容量录一张图 + 填表间接寻址”换取静态性，必然要在另外几个地方付出代价。

第一个代价是**不满批时的空转浪费（Padding Waste）与常驻图池**。因为图是按任务声明的最大行数 $R$ 和最大状态数 $S$ 录制的，当线上并发不满（例如最大容量 128 行，但当前步只有 16 个活跃请求）时，指向第 $S$ 号哑状态的那 112 行虽然权重为 0、不会影响计算结果，但 GPU 里的矩阵乘和反向传播依然会实打实地算满 128 行。同时，每个捕获任务都独占着一块按最大形状分配的私有 CUDA graph 显存池，首次触发冷录制（或底层隐藏状态缓冲更换存储导致重录）时需要跑三次预热加一次流同步。此外，目前的训练图与 vLLM 自己的主模型图是分开录制的独立图，相邻任务之间仍保留着宿主侧填表、锁页上传、结果拷出以及版本锁定与释放的边界。

第二个代价是**对用户代码写法的硬约束**。既然底层永远是“$R$ 行一起跑，靠掩码屏蔽无效行”，用户写的训练方法就必须遵守两条铁律：
1. **不能写依赖张量数值的 Python 分支**：诸如 `if loss < 0.01:` 之类的语句在规划期用 `meta` 张量做形状追踪时就会被直接报错拦截（见[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)），任何条件逻辑都必须写成 `torch.where` 或通过 `rows=` / `update=` 选择器传入；
2. **第 0 维（行维度）必须严格逐行独立**：训练方法必须返回形状为 `[R]` 的逐行损失，由框架结合 `active` 掩码做最终归约。如果用户在方法体内部自己写了跨行操作（比如 `loss.mean(dim=0)` 或 `BatchNorm`），那 112 行指向哑状态的零就会把 16 行真实请求的梯度稀释掉。

那么，把这套机制放到不同规模的真实模型上，最终的端到端吞吐表现如何？我们可以对比同一张 RTX 4090（vLLM 0.32.0，128 并发，生成 256 token，ESamp $\beta=0$，rank 4）下 `Qwen2.5-0.5B` 与 `Qwen2.5-7B` 的实测数字：

| 模型规模（128 并发，RTX 4090） | 原生 vLLM 一步耗时与吞吐 | ESamp ($\beta=0$, `publish_lag=1`) | ESamp ($\beta=0$, `publish_lag=0`) | 旁路训练图单步 GPU 耗时 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-7B-Instruct** | ~21.7 ms (5,894 tok/s) | **97.4%** (5,738 tok/s) | **97.2%** (5,731 tok/s) | ~616 µs（其中分页 AdamW ~191 µs），约占一步的 **2.8%** |
| **Qwen2.5-0.5B-Instruct** | ~3.0 ms (42,355 tok/s) | **91.9%** (38,936 tok/s) | **89.8%** (38,047 tok/s) | ~178 µs（其中分页 AdamW ~29 µs，vLLM 0.27.1 时期剖析） |

对比这两行数据，固定开销与模型规模的比例关系非常清晰：
* 在 **7B 模型**上，主干网络跑一步解码需要 21.7 ms；而回放一整张训练图（含参数收集、前向、反向以及约 191 µs 的分页 `AdamW`）总共只需约 **616 µs**，不到主干一步耗时的 3%。因此，在默认的跨步重叠模式（`publish_lag=1`）下，端到端吞吐达到了原生 vLLM 的 **97.4%**；即便强制要求当步串行等待训练完成（`publish_lag=0`），吞吐依然有 **97.2%**。
* 而在一步只有 **3.0 ms** 的 **0.5B 小模型**上，虽然训练图本身的 GPU 回放只需一百多微秒，但任务之间填表、6 µs 的锁页拷贝以及宿主侧版本簿记的固定交接成本占比被放大了，使得 `publish_lag=1` 下的吞吐定格在原生的 **91.9%**（早先在 vLLM 0.27.1 上做 20 轮交叉测量时，$\beta=0$ 为 92.45%，加上采样引导 $\beta=0.5$ 后为 83.38%）。

回顾整个过程，从最初挂在模型层上的 Python 钩子在图回放时“人间蒸发”，到最后把包含自动微分与优化器更新的完整算法塞进一张永不重录的 CUDA graph，核心的工程动作其实只有一个：**把“控制流与指针的变化”降维成“固定地址表里的整数数据”**。

只要你还试图在每一步里靠 Python 去切换张量引用、按活跃行数动态裁剪形状、或者把设备上的掩码拉回主机做分支判断，你就会一直站在 CUDA graph 的对立面。反过来，一旦你用预分配的固定池接住状态，用第 `S` 号哑状态接住多余的空行，甚至把外部张量那随时搬家的 64 位内存地址也当成普通整数填进表里交给内核去解引用，Python 就可以安心地在预热录制结束那一刻退场了——因为剩下的所有动态性，都已经变成了那张每步 23 KB 的小表里安静流淌的数字。
