# What Color is Your Tensor? —— 当 Python 只在录制时活着，怎么在 CUDA Graph 里跑动态算法

如果你写过一段时间的 PyTorch 研究代码，你脑子里大概住着一个非常舒服的世界观：Python 是站在指挥台上的乐队指挥。

你写下一句 `hidden = layer(x)`，GPU 就拉一小节小提琴；你写下一句 `loss.backward()`，GPU 就吹一段圆号；要是演出中途你想确认声音对不对，随手塞一句 `if loss.item() < 0.01:`，或者往模型的第 0 层挂一个 `register_forward_hook`，整个乐队就会乖乖停下脚步，把刚算出来的数字递到你手里，等你看完再决定下一小节怎么演。

在这个世界里，所有的张量都是平起平坐的，你的 Python 代码永远活在当下。

直到有一天，你打算把一段在单卡脚本里跑得顺风顺水的算法，搬进像 vLLM 这样的生产级推理引擎里。

需求听起来再自然不过了：在逐字生成（decode）的每一步，你想从模型第 0 层和最后一层捞出隐藏状态，为当前批次里的每个请求在线训练一个轻量级的秩-4（rank-4）小网络：算出蒸馏 loss，反向传播求导，再用 `AdamW` 更新这个请求专属的权重与一阶、二阶动量。

你满怀信心地往模型层上挂好 PyTorch forward hook，写好训练逻辑，启动服务。然后你会接连撞上两件怪事：

第一件怪事是，你在模型层 hook 里写的那些 Python 调度逻辑，在引擎启动预热的那一步确实跑过一次；可一旦预热结束、真正的用户流量开始一个字接一个字往外蹦时，你的 Python 函数就像被人施了定身法一样，**再也没有被调用过哪怕一次**。

第二件怪事是，如果你把那五六十个零碎的在线训练算子（取权重、前向、相减、平方、反向求导、AdamW 动量更新）放在图外老老实实逐个执行，你会发现 GPU 大部分时间都在空转，CPU 却满头大汗——在单张 RTX 4090（24 GiB GDDR6X 显存）上跑 128 并发的 Qwen2.5-0.5B 模型（vLLM 0.27.1，训练步尚未进图的一版较早实现），原生 vLLM 一步 decode 约 3.3 毫秒，扩展层却要每步额外增加约 4.6 毫秒的主机耗时，最大头正是一个接一个地发射这堆训练小算子（其中反向传播一段约占训练的 36%）。

要想不让 CPU 拖垮 GPU，你的算法也必须钻进 **CUDA Graph** 里。而一旦你推开 CUDA Graph 的大门，你会发现自己走进了一个被严格划分成**两种颜色**的世界。

---

## 一、 蓝色张量与红色张量的五条铁律

2015 年，Bob Nystrom 写过一篇著名的文章叫《What Color is Your Function?》，讲的是同步函数（蓝）与异步函数（红）之间那道让人抓狂的隔离墙。

在开启了 CUDA Graph 的现代推理引擎里，张量（以及操作它们的函数）其实也被悄悄涂上了两种颜色。不妨把静态图内部的世界叫作**蓝色（Blue）**，把动态变化的运行时世界叫作**红色（Red）**。

为什么非要有蓝色世界不可？算一笔 CPU 发射账就明白了。

在 PyTorch 默认的即时执行（Eager）模式下，每执行一个算子，CPU 都要依次穿过 Python 解释器、PyTorch 分发器（Dispatcher）和 CUDA 驱动，最后往 GPU 队列里塞一个内核。哪怕算子本身什么都不算，这套广播体操在 CPU 上也要花掉好几微秒。在长文本预填充（prefill）时，一个矩阵乘在 GPU 上跑好几毫秒，几微秒的发射开销不值一提；但在逐字生成的 decode 阶段，每个请求每一步只算 1 个新 token。跑完一整座数十层的 Transformer 要发射几百个小算子，几百乘上几微秒就是毫秒级——CPU 喊口令的速度根本追不上 GPU 算数的速度。

在单张 RTX 4090、vLLM 0.32.0、128 并发序列（8 条 prompt × `n=16`，生成 256 个 token）的基准设置下，如果把 CUDA Graph 关掉退回 Eager 模式，单步耗时 21.7 ms 的 Qwen2.5-7B 还能靠肥厚的矩阵乘掩盖发射开销，保住 98% 的吞吐（5,773 vs 5,894 tok/s）；但在单步仅 3.0 ms 的 Qwen2.5-0.5B 上，关掉 CUDA Graph 会让吞吐当场从 42,355 tok/s 暴跌到 20,926 tok/s——**整整腰斩了一半（49%）**。

```text
红色世界 (Eager 模式 — CPU 现场指挥每一拍，GPU 频频停顿等口令):
  CPU:  [Py+发射 k1][Py+发射 k2][Py+发射 k3][Py+发射 k4]...
                   \           \           \
  GPU:              [k1] (空转)  [k2] (空转)  [k3] (空转)

蓝色世界 (CUDA Graph — 预热时录成静态磁带，运行期一键回放):
  录制 (仅1次):  CPU 跑一遍 Python 与钩子 ──► 录下固定地址与固定网格的指令图
  回放 (每一步): CPU: [ cudaGraphLaunch ] (全步只需调这一次！)
                GPU: [k1][k2][k3][k4][k5][k6]... (首尾相接狂奔，Python 已退场)
```

为了消灭这笔发射税，引擎在启动预热时会开启 CUDA Graph 捕获：CPU 照常跑一遍 Python 代码，但 GPU 并不急着算数，而是把这一路上收到的所有内核名称、线程网格尺寸、以及每个张量参数所在的**裸显存地址**，原封不动地录成一张静态的蓝色指令图。到了真正的线上解码期，CPU 每一步只调用一次极轻量的 `cudaGraphLaunch`，GPU 就顺着磁带自己把几百个内核一口气放完。

这听起来很美，但只要你想把自己的算法也放进这张图里，就必须签下石碑上的**五条颜色铁律**：

1. **蓝色函数的输入输出必须是蓝色的（地址与形状焊死）**：录进 CUDA Graph 的每一个算子，只认录制那一刻绑定的物理显存指针（`data_ptr`）和张量形状。如果在下一步里你用 `torch.empty()` 新分配了一块张量，蓝色图回放时会对新张量视而不见，依然蒙着眼睛去踩录制时的旧地址。
2. **偏偏真实世界的输入全是红色的（每步都在变）**：在线上服务里，每一步进来几个活跃请求是红的（人数在变）；连续批处理把哪个请求排在第几行是红的（座位在变）；每个请求的权重更新到了哪个版本槽位是红的；甚至外部组件（如 vLLM 采样器）递给你的张量也是红的——宿主随时可能释放旧显存、换一块新地址。
3. **蓝色函数内部看不见红色的 Python（录制后 Python 已死）**：这就解释了开头的第一件怪事。录制只在预热时发生一次。录像机眼里只有发往 GPU 的内核指令；至于你在 Python hook 里写的字典查找、列表追加、或者宿主端的 `if/else` 条件分支，全都留在 CPU 内存里，磁带上连半个比特都没录进去。回放时 vLLM 直接绕过模型的 Python `forward` 按下播放键，你的 Python 代码自然也就“死”在了预热结束的那一秒。
4. **红色代码一旦偷看蓝色张量的值，全线流水线当场急刹车（零主机同步）**：如果宿主端的红色 Python 代码试图去读一个蓝色 GPU 张量里的数值——哪怕只是写了一句看似无害的 `if loss.item() < 0.01:`、`assert not torch.isnan(x).any()` 或者 `.tolist()`——就会立刻触发**主机同步（Host Synchronization）**。
5. **纯函数编译器讨厌蓝色的原地写池子（Functionalization 冲突）**：如果你试图用讲究“纯函数无副作用”的编译器去直接包裹蓝色状态池的原地写入，编译器为了自保，会在图里偷偷克隆出一份副本。

其它几条都好理解，唯独**第四条铁律（零主机同步）**值得咱们停下来算一笔账。因为在写系统代码时，人们太容易产生一种侥幸心理：“我就拷一个布尔值回 CPU 看一眼，几微秒的事，能有多大影响？”

影响是毁灭性的：**少几次同步什么也换不来，只有把同步次数真正变成零，性能才会跳变。**

为什么？因为在健康的异步推理引擎里，CPU 和 GPU 是在跳双人舞：当 GPU 正在埋头苦算第 $t$ 步的 Transformer 前向时，CPU 并没有闲着，它已经领先半步，在后台紧锣密鼓地准备当前步稍后或下一步所需的元数据了。在一次 RTX 4090 上对 0.5B 模型（vLLM 0.27.1，128 并发，ESamp $\beta=0$）的剖析中，pLLM 每步在 CPU 上的全部准备工作大约要花 **700 µs**，最后通过一次耗时仅 **6.24 µs** 的非阻塞上传送进显卡。因为 CPU 提前干完了活，这 700 µs 完完全全藏在了约 3 ms 的 GPU 前向阴影背后，GPU 在两步之间的空闲间隙只有微不足道的 **4–6 µs**。

现在，假设你在代码里留了哪怕**一次** `.item()`。为了拿到这一个字节的结果，CPU 必须立刻踩下急刹车，原地阻塞，眼睁睁等着 GPU 把队列里积压的所有前向内核全部排空。等 GPU 终于停下来把数字递给 CPU，CPU 这才开始慢吞吞地跑那 700 µs 的准备工作——而此时此刻，整张显卡只能干坐着等 CPU！原本藏在阴影里的 700 µs 瞬间跳到台前，严严实实地堵在了关键路径上。在 pLLM 较早的一版实现上（0.5B，vLLM 0.27.1，128 并发），只有当热路径上残留的最后一次同步被彻底拔除时，图模式吞吐才从约 11,600 tok/s 一举跃升到约 19,100 tok/s。

> **旁注：一个看起来无害的 `torch.tensor` 是怎么偷走异步性的？**
> 就算你一个 `.item()` 都没写，在把 CPU 算好的索引传给 GPU 时，如果你随手写下 `torch.tensor(values, device="cuda")` 或者从普通 Python 列表做 `copy_(..., non_blocking=True)`，你的红色代码依然会偷偷触发同步等待！
> `pllm/execution/cuda_stream.py` 里的 `upload` 函数道破了这个很多 PyTorch 老手都会踩的暗坑：普通的 CPU 张量分配在操作系统的**可分页内存（Pageable Memory）**里。由于操作系统随时可能把这页内存换出到磁盘，GPU 的硬件 DMA 引擎根本不敢在后台直接去读它——CUDA 驱动为了安全，会在底层把拷贝变成同步等待主机流。真正的非阻塞上传，必须先踏进**锁页内存（Pinned Memory）**：
> ```python
> # check-docs: skip: excerpt from pllm/execution/cuda_stream.py
> def upload(
>     values: Sequence[float] | Sequence[Sequence[float]] | npt.NDArray[np.int64],
>     device: torch.device,
>     dtype: torch.dtype = torch.long,
> ) -> torch.Tensor:
>     # ……
>     host = torch.tensor(values, dtype=dtype)
>     return host.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else host.to(device)
> ```
> 在每步都要高频执行的 `TensorExecutor` 里，连每次调用 `.pin_memory()` 向主机缓存分配器借内存的开销都省了：我们用 `PinnedStaging` 维护几个预先锁好页的缓冲区，每步让 Rust 直接往锁页数组的 NumPy 视图里填数字，发完异步拷贝后录一个 `cuda.Event`。下一步要用时，只调非阻塞的 `event.query()` 问一句“上次拷完没？”——拷完了就原地复用，还在路上就再添一块新缓冲，绝不让 CPU 等半步。

---

## 二、 破局：造一张“外蓝内红”的 `int64` 转换表

现在，矛盾摆在了台面上：

为了不让五六十个训练小算子把 CPU 发射累死，我们必须把整个在线训练步——读取请求参数、跑前向、算 loss、反向传播求导、再用优化器更新写回——统统涂成**蓝色**，录进一张 CUDA Graph 里（宿主侧的前向收尾放在模型图回放刚结束、采样器还没开始的 `compute_logits` 接缝上；训练本身默认在 token 提交后发往旁路流）。

但外面的世界彻头彻尾是**红色**的：请求每步都在换座位、请求人数每步都在变、哪些请求要更新由动态条件决定、甚至连外部输入张量的显存地址都在变。

怎么让一个只认蓝色固定地址和固定形状的死板机器，去干一件每一步都在随机应变的红色差事？

答案是：**不要试图去改蓝色图的形状，而是造一个“外皮是蓝色、肚子里装红色数据”的染色转换器——一张物理地址永远固定、形状永远不变的 `int64` 整数表。**

咱们不妨从最简单的情况开始，把这四件红色的麻烦事一件一件塞进这张蓝表里。

### 第一件红事：请求的座位每步都在变

如果世界上自始至终只有 1 个请求，录制训练图易如反掌。但只要来了第 2 个请求，麻烦就来了：在连续批处理里，这一步 batch 的第 0 行是请求 A，下一步请求 A 结束了，第 0 行变成了新来的请求 C（见[《你其实不需要读 `attn_metadata`》](01-attn-metadata.md)）。

要让固定的蓝色指针指向变化的请求参数，经典的系统解法是**加一层间接寻址（Indirection）**。

我们预先向显卡申请一块基地址永远不变的蓝色大张量——一个**分页状态池（Slot Pool）**，形状是 `[capacity, ...]`。请求 A 分到 0 号槽，请求 B 分到 1 号槽，请求 C 分到 2 号槽（关于这套状态池如何在多流并发下通过写时复制生长出不可变版本，详见下一篇[《给 GPU 显存加上 MVCC》](03-gpu-mvcc.md)）。

接着，我们在显存里再划出一张地址同样永远固定的蓝色 `int64` 小表。每一步开始前，CPU 在锁页内存里把“当前步第 $r$ 行应该去状态池的几号源槽（`sources`）读参数、算完的新参数又该写进几号目标槽（`destinations`）”填好，一次异步拷贝推上显卡。

录在 CUDA Graph 里的蓝色内核，只认状态池的固定基地址和小表的固定地址：先去表里读出整数槽号 `slot`，再去池子里按 `pool[slot]` 读写。外面的请求怎么换座位、版本怎么升，变的都只是表里的几个整数，录好的图连一根毫毛都不用动。

### 第二件红事：请求的人数每步都在变（造一个“哑状态”当黑洞）

解决了换座位，紧接着就是第二个难题：线上的活跃请求数 $n$ 每一步都在波动。这一步有 30 个请求，下一步涌进来 15 个变成 45 个，再下一步只剩 12 个。

但铁律一规定：**蓝色张量的形状必须固定**。

你可能会想：“vLLM 录模型前向的时候，不是按 `batch_size = 1, 2, 4, 8, 16...` 分档位（Buckets）录了好几张图吗？我们也给训练按档位录十几张图不就行了？”

对无状态的模型前向来说，分档录图很划算；但对有状态的在线训练任务来说，影响形状的维度有两个：当前步的打包行数 $R$，以及同时活跃的请求状态数 $S$。更要命的是，每录一张图，CUDA 都会为这张图的中间激活值和反向梯度分配一个**私有的显存池（Graph Pool）**。如果按 $(R, S)$ 的组合分档录上一堆图，这些图池占用的常驻显存都要从宝贵的 KV Cache 里扣。

怎么只录**唯一的一张蓝色图**，却能应付从 $1$ 到 $R$ 的任意行数、从 $1$ 到 $S$ 的任意状态数？

我们的解法朴素得近乎耍赖：**永远按任务声明的最大容量（最大行数 $R$、最大状态数 $S$）录一张图；至于没坐满的空座位，全部指给一个不会喊疼的“假人”。**

具体来说，如果最大状态容量是 $S$，我们在规划读取槽号表 `sources` 时，故意多留出一列，做成 $S+1$ 列（下标 $0 \dots S$）。其中 $0 \dots S-1$ 是真正活着的请求状态，而第 $S$ 号则是专门设立的**“哑状态”（Padding Sink）**。

假设最大容量 $R=128, S=128$，而这一步红世界里只来了 30 个请求。CPU 在填蓝表时，把前 30 行指向它们各自真实的状态号（$0 \dots 29$），并把有效掩码 `active` 标为 `1`；而剩下的第 $30 \dots 127$ 行，统统指向第 $S$ 号哑状态，同时把 `active` 标为 `0`：

```text
蓝色索引表 (固定形状 R=128)              蓝色状态池与第 S 号哑状态 (Sink)
┌───────────────────────────┐
│ Row 0   (active=1, st=0)  │─────────► State 0   [ 读真实源槽 / 写真实目标槽 ]
│ Row 1   (active=1, st=1)  │─────────► State 1   [ 读真实源槽 / 写真实目标槽 ]
│ ...                       │           ...
│ Row 29  (active=1, st=29) │─────────► State 29  [ 读真实源槽 / 写真实目标槽 ]
├───────────────────────────┤
│ Row 30  (active=0, st=S)  │──┐
│ ...                       │  ├──────► State S (Padding Sink 假人黑洞):
│ Row 127 (active=0, st=S)  │──┘        • sources [F, S+1]   -> 槽号填 -1 (读出全 0.0)
└───────────────────────────┘           • loss 归约与求导     -> active 权重为 0.0 (零梯度)
                                        • destinations [F, S]-> 压根没有第 S 列！(永不写入)
```

顺着这张图看过去，你会发现这个“第 $S$ 号假人”把红蓝边界缝合得严丝合缝：
* **读取时（`sources` 形状为 `[F, S+1]`）**：哑状态对应的源槽号填的是 `-1`。负责从池子里捞参数的 `_gather_field` 一眼看到槽号 `< 0`，就会用 `masked_fill` 把那一行涂成全 `0.0`。假人拿到了一身全零的干净参数，绝不会在后续计算里触发 `NaN`。
* **求导与归约时**：虽然蓝色图里的前向和反向内核还是老老实实地跑满了全部 128 行，但因为空行的 `active=0`，在算 loss 归约和按状态累加梯度时，假人所在行的权重严格为 `0`。它算出来的任何数字，连一丁点儿都不会混进真实请求的梯度里。
* **写回时（`destinations` 形状为 `[F, S]`）**：注意看两个表的列数——目标槽表比源槽表**少了一列**，压根就没有第 $S$ 列！优化器只更新前 $S$ 个真实状态，哑状态永远没有写入目标。

靠着这个“读出来是零、权重是零、写回时隐形”的第 $S$ 号假人，无论外面的红色世界是 1 个请求还是 128 个请求，里面的蓝色 CUDA Graph 看到的永远是整整齐齐的 `[R, ...]` 和 `[S, ...]`。

### 第三件红事：按动态条件选择谁更新（把选择留在显卡里）

解决了可变人数，算法作者又提了一个更刁钻的红色需求：**按运行时的数值条件选择性训练**。比如只对满足特定掩码的行算梯度，或者只更新本步被动态条件选中的请求状态（通过 `rows=` 或 `update=` 选择器指定）。

如果是在 Eager 模式下，这太好写了：在 GPU 上算出布尔掩码 `mask`，调一句 `selected = states[mask].unique().tolist()` 拉回 CPU，然后 CPU 只为被选中的状态分配新版本。

停！看到 `.tolist()` 的一瞬间，铁律四的警报就拉响了——**那是主机同步！**

只要 CPU 想知道“这一步到底哪几个状态被动态条件选中了”，CPU 就必须停下来等 GPU 把前面的条件算完。可如果 CPU 不去读这个掩码，负责在 CPU 上管账本的页表管理器怎么知道这一步该给哪几个状态发布新版本呢？

答案很简单：**谁规定“没被选中的状态”就不能发布新版本了？**

如果这一步批里有 30 个活跃请求，CPU 根本不去打听 GPU 上的动态掩码算出了什么结果，而是干脆利落地假定**这 30 个活跃状态全部参与本步发布**，在 CPU 账本上统统给它们分配好新槽位、推进版本号，写进蓝表里。

真正的筛选，100% 留在蓝色的 GPU 图内部完成。来看看 `pllm/execution/tensor_executor.py` 里的 `_selection`：

```python
# check-docs: skip: excerpt from pllm/execution/tensor_executor.py
def _selection(self, rows: FixedRows) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    # ……
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

看那行 `index_add_`：GPU 自己在设备端统计出每个状态在这一步有几行被选中（`updates = count[: self.state_count] > 0`）。接着，当图里的优化器去写目标槽时：
* 如果某个状态的 `updates[s]` 是 `True`，分页 `AdamW` 内核就把更新后的新权重和动量写进新槽位；
* 如果某个状态的 `updates[s]` 是 `False`（说明它这一步其实没被选中），图里的内核就直接把它的旧字段**逐字节原封不动地拷进新槽位**！

这是一笔极其划算的系统交易：我们在显存里多做了一点微不足道的拷贝，在版本树上多挂了一个内容不变的新版本号，却彻底买断了“CPU 必须等 GPU 算完掩码”的红色同步传染。

### 第四件红事：当别人的张量每步都在搬家——把裸指针涂成蓝色整数

走到这一步，凡是归扩展层自己管的状态池、中间缓冲、结果槽，全都住进了预先盖好的蓝色固定公寓里，门牌号生老病死都不变。

但城外还有别人。

在线算法有时需要读取一些**不归我们分配**的外部张量——比如宿主 vLLM 采样器传过来的反馈张量，或者刚拼好的 token 输入。这些张量是彻头彻尾的**红色张量**：vLLM 并不知道我们在录自己的图，它在某些步心情好，就会把旧张量释放掉，换一块新分配的显存。

如果你直接把这个外部张量传进要录制的算子里，CUDA Graph 就会死死记住它录制那一刻的旧地址 `0x7f00...`。等到下一步，vLLM 给它搬了家（比如搬到了 `0x7f88...`），而录好的蓝色图还在按图索骥去敲旧门牌——不是读出一堆垃圾，就是当场非法内存访问崩溃。

怎么在一张连指针都被焊死的蓝色 CUDA Graph 里，去读一个每步都在搬家的红色外部张量？

如果你写过 C 语言，你大概已经想到那个漏洞了：在硬件底层，**指针（Pointer）不就是一个普通的 64 位整数吗？**

既然蓝色图只认固定地址，那我们就**不要把那个红色张量作为张量参数录进图里；而是让 CPU 每一步调一次 `tensor.data_ptr()`，把拿到的 64 位裸内存地址当成一个平平无奇的 `int64` 整数，写进我们那张固定地址的蓝色小表里！**

来看看 `pllm/kernels/page_copy.py` 里的 `_gather_rows`。这是整套机制里把“红转蓝”玩得最彻底的十行代码：

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

请盯住 `address = ...` 那一行。

CPU 每一步把外部张量的首地址 `data_ptr()` 连同它的两个维度步长（`stride(0)` 和 `stride(1)`），打包成三个 `int64` 整数，填进固定蓝表的 `descriptor` 段里。到了 GPU 上，Triton 的 `.to(tl.pointer_type(...))` 轻轻一挥魔杖，把这个从蓝表里读出来的 64 位整数，当场强转成了一个带类型的设备指针（Device Pointer），下一行直接用 `tl.load(address + ...)` 去解引用：

```text
固定地址的蓝色 int64 表 (显存地址 0xAAAA0000 —— 永不改变)
┌─────────────────────────────────────────────────────────┐
│ descriptor[0]: 0x7f8812340000  (来自 .data_ptr() 的整数) │
│ descriptor[1]: 3584            (int64 stride0)          │
│ descriptor[2]: 1               (int64 stride1)          │
└───────────────────────┬─────────────────────────────────┘
                        │ 1. tl.load(descriptor) 读出 int64 整数
                        │ 2. .to(tl.pointer_type(...)) 强转为设备指针 T*
                        ▼
每步都在搬家的红色外部张量 (本步刚搬到 0x7f8812340000)
┌─────────────────────────────────────────────────────────┐
│ [ row 0 ][ row 1 ][ row 2 ] ...                         │
└─────────────────────────────────────────────────────────┘
```

为什么脾气暴躁的 CUDA Graph 不会跟它急眼？

因为对录像机来说，它录下的内核参数是 `descriptor`——而 `descriptor` 是固定蓝表的切片，地址一万年不变，纯蓝！录像机记下的动作仅仅是：“去 `0xAAAA0000` 这个固定地址读三个整数，再顺着第一个整数指的方向去搬数据”。至于那个整数今天指向东街、明天指向西街，那是流过蓝色管道的数据，不是录制时的图结构。一行类型转换，就把“图的生命周期”和“外部张量的内存生命周期”彻底解耦了。

现在，把这四件红事的解法拼在一起，你就看懂了 `pllm/execution/tensor_executor.py` 里那张统领全局的蓝色转换表 `TableLayout`：

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

整张表没有一行多余的装饰：`rows` 管行身份与有效掩码，`sources` 和 `destinations` 管状态池的读写槽号与第 `S` 号哑状态，`row inputs` 装那 `K` 个随时搬家的外部张量裸地址。在前述 0.5B、128 并发的 ESamp（$\beta=0$）剖析中，把这四段全部算上，每一步从主机锁页内存拷进显卡的总字节数只有 **23,692 字节**，耗时 **6.24 µs**（作为参照，原生 vLLM 自己每步上传元数据是 1,088 字节、耗时 2.50 µs）。每步多花不到 4 微秒上传二十几 KB 的整数，就换来了整张计算图永远不用重录的特权。

---

## 三、 按下录音键时的两个魔鬼细节，以及为什么不套 `torch.compile`

万事俱备，只欠按下录音键了。负责把训练体录成蓝色图的是 `pllm/execution/step_capture.py` 里的 `StepCapture`。

按理说，调用 PyTorch 的 `with torch.cuda.graph(graph):` 只是十来行模板代码的事。但当你真的把这套包含状态池原地写入的训练体放进引擎里去录时，有两个只有亲手踩过才会懂的边界细节：

第一个细节：**预热（Warmup）会弄脏真数据**。
在正式录制 CUDA Graph 之前，需要先在一条私有流上把函数预热 3 遍（`for _ in range(3)`），让 cuBLAS 和 Triton 完成工作区分配与初始化。但别忘了，我们的训练体尾巴上挂着真刀真枪的 `AdamW` 优化器，会往 `destinations` 指向的池子槽位里写新权重！如果在某一步触发录制时直接跑 3 次预热，池子里正在服务线上请求的真实参数就会被预热的垃圾梯度覆盖 3 次。

看看 `StepCapture._capture` 是怎么用一行代码化解危机的：

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

在进入 `capture_graph` 之前，先调一句 `destinations.fill_(-1)`，把表里所有的目标写槽统统涂成 `-1`！因为无论是分页拷贝内核还是分页 `AdamW` 内核，内部都有 `dst >= 0` 的掩码守卫，看到 `-1` 就会跳过最终的存储指令。于是，3 次预热外加 1 次录制，整整 4 遍跑下来，所有的算子都分配好了显存池，但对于保存真实参数的状态池来说，这 4 遍执行**完完全全是只读的**。等录完图回到 `StepCapture.run`，再调一次 `refill()` 把当前步真正的目标槽号填回表里，然后从容地按下 `self.graph.replay()`。

第二个细节，藏在 `capture_graph` 录制窗口的前后：

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

为什么录个 GPU 的图，还要关掉 Python 的循环垃圾回收器（`gc.disable()`）？
因为 Python 的循环 GC 是按对象分配计数随时触发的。当代码在 `with torch.cuda.graph(...):` 里分配中间张量包装对象时，GC 可能突然在半路醒来，清理一批早已死掉的循环引用对象。如果那批死对象里恰好包着一个曾在**另一条 CUDA Stream** 上用过的旧张量，它的 C++ 析构函数就会在此时向 CUDA 缓存分配器发出跨流释放通知——而正在录图的 CUDA 驱动看到捕获窗口里突然插进一个别的流的动作，这次捕获就会失效。关掉自动 GC 直到录制结束，才彻底封死了这个幽灵故障。

### 铁律五的印证：为什么给训练步套上 `torch.compile` 反而更慢？

把整个训练步作为普通 Eager Autograd 代码录进 CUDA Graph 之后，一个非常自然的念头会冒出来：虽然 CPU 发射开销被 CUDA Graph 吃掉了，但训练图里毕竟还有五六十个零碎的逐元素小内核。要是我们顺手给训练体再套上一层 `@torch.compile`（Inductor），把碎算子融合成几个大内核，岂不是锦上添花？

我们在 RTX 4090、0.5B 模型（vLLM 0.27.1，128 行，rank 4，$\beta=0$，未编译时吞吐为原生的 **92.45%**）上老老实实做了两组对照实验：
1. **只用 `torch.compile` 编译用户写的 loss 前向与反向函数**（外围填表与分页优化器不变）：跑 7 轮取中位数，吞吐是原生的 **92.77%**——与不编译相比，差距完全落在测量噪声里。道理很直接：小算子序列录进 CUDA Graph 后，CPU 发射开销早已归零；而 rank-4 小网络的中间张量极小，很可能本来就还留在 GPU 的 L2 Cache 里，省下两三次中间读写根本听不见响。
2. **把整个被录制的 `execute_body`（含参数 gather、loss 归约、梯度路由与优化器）统统交给 `torch.compile`**：结果吞吐不仅没涨，反而掉到了原生的 **90.10%**！

打开 profiler 时间线一看，这就是前面说的**铁律五**。编译器确实很卖力地把旁路流每步发射的内核数从 59 个砍到了 25 个；但是，旁路流每步消耗的 GPU 总时间，却从原来的约 **178 µs** 暴涨到了约 **257 µs**！

多出来的近 80 微秒花在哪儿了？时间线上赫然多出了一个名叫 `triton_poi_fused_clone_paged_adamw_slice_0` 的陌生内核，每步白白吃掉约 **39 µs**；同时紧随其后的分页 `AdamW` 内核耗时也从约 29 µs 膨胀到了 64 µs。

罪魁祸首是 PyTorch 编译器底层的 **Functionalization（函数式化）** pass。AOTAutograd 在对计算图做变换之前，要求图里所有算子都是没有副作用的纯函数；而我们的蓝色状态池偏偏是围绕“在一块巨大的预分配固定张量池里原地读写指定页槽”设计的。当编译器看到代码试图往状态池切片里原地写数据时，为了捍卫纯函数语义，它在生成的代码里默默把状态池切片**克隆（clone）了一份副本**，改完再拷回去。“纯函数式图变换”与“原地写固定分页池”两种世界观迎头相撞，保护性克隆吃掉的显存带宽远远超过了算子融合省下的几微秒。（因此，我们只在真正属于纯函数计算的场景——如[图内挂载点与覆盖层梯度](04-compiled-model-hooks.md)、[采样引导重加权](07-free-flops.md)和[只读查表融合](08-bounded-gather.md)——里使用 `torch.compile`，而让分页状态步直接以原生算子序列录进 CUDA Graph。）

---

## 四、 账单与边界：当所有红色都变成表里的数字

当然，用“按最大容量录一张蓝图 + 第 `S` 号假人陪跑”来统一动态与静态两个世界，并不是没有代价的。

第一笔代价是**空位陪跑税（Padding Waste）与常驻图池**。因为按最大容量（比如 $R=128$）只录一张图，当线上处于低谷、某一步只有 16 个活跃请求时，剩下的 112 个假人虽然权重为 0，但 GPU 里的矩阵乘和反向传播并不会跳过它们；同时每个捕获任务都独占着一块按最大形状预留的私有 graph 显存池。此外，这种“112 个假人陪着 16 个真人一起跑”的机制，直接要求用户写的训练方法必须在第 0 维（行维度）上严格独立，绝不能在方法内部写 `loss.mean(dim=0)` 或 `BatchNorm`——否则 112 个假人算出的 `0.0` 就会把 16 个真人的梯度稀释得面目全非（我们会在第 10 篇[《`h - h.mean(dim=0)` 不会报错，但会让张三看见李四的提示词》](10-isolation-testing.md)里展示如何自动捉住这种错误）。

那么，把整套在线蒸馏训练（ESamp $\beta=0$，rank 4）装进蓝色 CUDA Graph 后，在真实的 128 并发基准下（单张 RTX 4090，vLLM 0.32.0）到底跑多快？这取决于你站在多大的模型面前：

| 基座模型（128 并发，RTX 4090） | 原生 vLLM 吞吐（单步耗时） | 开启在线训练（`publish_lag=1`） | 开启在线训练（`publish_lag=0`） | 旁路训练图每步 GPU 耗时 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-7B-Instruct** | 5,894 tok/s (`~21.7 ms`) | **5,738 tok/s (97.4%)** | **5,731 tok/s (97.2%)** | `~616 µs`（其中 AdamW `~191 µs`；616 µs 占整步 **2.8%**） |
| **Qwen2.5-0.5B-Instruct** | 42,355 tok/s (`~3.0 ms`) | **38,936 tok/s (91.9%)** | **38,047 tok/s (89.8%)** | `~178 µs`（其中 AdamW `~29 µs`，vLLM 0.27.1 剖析） |

在 7B 模型上，主干跑一步 decode 要 21.7 ms，而回放一整张包含前向、反向与分页 `AdamW` 的训练图总共只要 **616 µs**，不到主干一步的 3%，端到端吞吐稳稳停在原生的 **97.4%**；即便切到一步只有 3.0 ms 的 0.5B 小模型，在扣除了独立的表上传、结果拷出与版本交接等固定底噪后，依然保住了 **91.9%** 的吞吐；而且只要表与池的地址不变，稳态生成中就不必重录图。

回看这趟穿梭在红蓝两个世界之间的旅程，最核心的秘密其实就写在 `_gather_rows` 和 `TableLayout` 的那几行代码里：

当你不得不把一段充满动态变化的红色算法，塞进一个要求地址固定、形状固定、零主机同步的蓝色硬件容器时，千万不要试图在运行时去撬动容器的结构。退后一步，把所有每一步会变的东西——哪些请求在场、读哪一页版本、谁被选中更新、甚至一个外部张量此刻搬到了哪个内存地址——统统降维成一张固定表里的几个 64 位整数。

**图是死的，表是活的。** 当所有的红色动态性都被编码成蓝色表里的整数时，最死板的机器，也能跑最灵活的算法。
