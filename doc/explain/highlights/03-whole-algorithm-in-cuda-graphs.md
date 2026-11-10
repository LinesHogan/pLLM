# Python 只在录制时活着：固定地址、一张表、零同步

写惯了研究代码的 PyTorch 用户，脑子里大多住着一个非常舒服的心智模型：代码是一行一行往下走的。

你写一句 `hidden = layer(x)`，GPU 就跑一段前向；你写一句 `loss.backward()`，GPU 就去算一遍梯度；中间你想看看数值对不对，随手塞一句 `print(loss.item())`，或者写个 `if loss < 0.01:` 跳过更新，一切都顺理成章。在这个世界里，Python 是站在指挥台上的乐队指挥，手里的指挥棒指到哪儿，底层的 GPU 乐手就演奏到哪儿。

遗憾的是，当你走进生产级推理引擎（比如 vLLM）的稳态 decode 循环时，指挥台上是空的。

在稳态生成阶段，真正指挥 GPU 的不是 Python，而是一卷早就录好的磁带——一张 **CUDA graph**。你的 Python 函数只在引擎预热录制的那一次真正活过；等录制结束、真正的用户流量涌进来之后，每一步 decode 都只是在按一下复读机的播放键。

如果在这种环境里，你不仅想改一改前向特征，还想把一整套在线学习算法——从取出某个请求专属的权重参数，到跑完用户定义的前向、算出 loss、反向传播求导，再用 AdamW 更新动量并写回显存——原封不动地塞进每一个 decode 步里，会发生什么？

这篇文章就是我们一路踩坑、一路搭积木的手记。我们不妨坐下来，从最天真的第一行代码开始，看看每当你往现实世界迈出一步，CUDA graph 的铁律会怎样把你绊倒，又怎样逼着我们把动态的世界一点点折叠进一张静态的 `int64` 整数表里。

---

## 第零步：人间蒸发的 Hook

故事要从我们做早期版本的 ESamp（在线蒸馏引导采样）说起。

需求听起来非常简单：每一步生成时，我们要从模型第 0 层把刚算出来的 `hidden` 状态捞出来，喂给一个轻量级的小网络（distiller）做一次前向预测，赶在同一步的采样器抽样之前把引导信号送过去。

如果你熟悉 PyTorch，你的手指大概已经条件反射般地敲出 `register_forward_hook` 了。在第 0 层挂一个 forward hook，`hidden` 一出炉就顺手在 hook 里发起旁路流上的预计算——时机最早，还不侵入模型源码，多优雅！

我当年也是这么想的，而且高高兴兴地把代码写了出来。为了查清小网络到底是什么时候跑的，我在调度链路上加了 CUDA event 计时探针，终端吐出了这样一行日志：

```text
schedule_attempt_count=1, schedule_hit_count=0, fallback_count=30
```

尝试调度 1 次，命中 0 次，回退 30 次。

我们写在 layer hook 里的那段预计算调度代码，在整整 30 个稳态 decode 步里，就像是人间蒸发了一样，一次都没有执行过。

为什么一段明明挂在模型第 0 层上的 Python 函数，模型每步都在吐字，它却纹丝不动？要解开这个谜题，我们得先蹲下来看看 GPU 在 decode 阶段到底过着什么样的日子。

在 PyTorch 的默认模式（Eager 模式）下，每当你写下一句 `c = a + b`，CPU 都要依次走完 Python 解释器、PyTorch 算子分发器（Dispatcher）和 CUDA 驱动，最后向 GPU 的任务队列发射（launch）一个内核。这一套广播体操做下来，哪怕什么都不算，光在 CPU 上就要花掉 3 到 5 微秒。

在做长文本预填充（prefill）时，一个矩阵乘内核在 GPU 上一跑就是几毫秒，CPU 花几微秒发个令根本不算事。但到了逐字生成的 decode 阶段，画风突变：哪怕把 128 个并发请求拼成一批准入，每个请求每一步也只算 1 个新 token。算子在 GPU 上往往几微秒就一闪而过，而跑完一整座数十层的 Transformer，前前后后要发射几百个小内核。

算一笔小学算术就明白了：几百个内核 × 每次发射几微秒，就是毫秒量级——和 0.5B 模型一步 3 ms 的 GPU 时间是同一个数量级。如果让 CPU 一个接一个地现场指挥，GPU 算完一个内核扭头一看，CPU 还在慢吞吞地填下一个内核的发射表单——整张显卡一半以上的时间都在空转等 CPU。

```text
Eager Mode (CPU conducts every bar live):
  CPU:  [Py+Launch k1][Py+Launch k2][Py+Launch k3][Py+Launch k4]...
                     \             \             \
  GPU:                [k1] (idle...) [k2] (idle...) [k3] (idle...)

CUDA Graph Mode (Record once, replay the tape):
  Record (1x):  CPU runs Python hooks & launches ──► Captured into Graph Tape
  Replay (Nx):  CPU: [ cudaGraphLaunch ] (one call!)
                GPU: [k1][k2][k3][k4][k5][k6]... (back-to-back, zero Python!)
```

为了消灭这笔发射税，vLLM 在启动预热时动用了 **CUDA graph**。它就像一台自动钢琴的打孔录音机：预热时，CPU 正常跑一遍 Python 前向代码，但 GPU 并不急着干活，而是把这一路上收到的所有内核名称、网格尺寸、甚至每个张量参数的**裸显存地址**，原封不动地录成一张图。等到真正服务请求时，CPU 每一步只需要调一次极轻量的 `cudaGraphLaunch`，GPU 就会自己顺着磁带把上千个内核一口气放完。

> **实验基准说明**：除特别注明的历史剖析外，本文引用的性能数据均在统一环境下测得——单张 RTX 4090（24 GiB）、vLLM 0.32.0、Qwen2.5-7B-Instruct（BF16）、128 并发（8 个 prompt × `n=16`）、生成 256 token，开启 CUDA graph 与 FlashInfer 采样器。为了让你直观看到与模型规模无关的固定开销，我们会把同配置下的 Qwen2.5-0.5B 数据放在旁边作对照。

这台“自动钢琴”到底有多重要？在我们的 RTX 4090 基准上，跑 Qwen2.5-7B 时一步 decode 本身要算 21.7 ms，关掉 CUDA graph 吞吐依然有原生的 98%（5,773 对 5,887 tok/s），因为肥厚的矩阵乘把 CPU 发射开销掩盖了；但只要切到步长极短的 Qwen2.5-0.5B，开着 graph 一步只要 3.0 ms（42,355 tok/s），一旦关掉 graph 退回 eager，吞吐当场腰斩到 **49%**（20,926 tok/s）。模型越轻、显卡越快，CUDA graph 就越是推理引擎的命根子。

看懂了这台录音机，前面那行诡异日志的谜底就水落石出了。

录制只有预热那一次。在录制那一刻，我们挂在第 0 层的 Python hook 确实执行了——这大概也是为什么 `schedule_attempt_count` 是 `1`。但是，录像机眼里只有发送给 CUDA 驱动的 GPU 内核指令！至于你在 hook 里写的“修改一个 Python 布尔标志”、“把张量塞进一个 Python 列表”、或者“根据条件在宿主端发起一次异步任务调度”，这些全都是发生在 CPU 内存里的 Python 动作。GPU 看不见它们，磁带上自然连一个比特都没录进去。

到了后面 30 步回放时，vLLM 直接绕过了整个模型的 Python `forward` 函数，一把按下了 `cudaGraphLaunch`。你的 hook 连被调用的机会都没有，藏在里面的调度逻辑自然也就随着录制结束而“死”在了过去。

当时找到病根后，我们的急救办法是把触发点从模型内部的层 hook 里拔出来，挪到模型图回放刚刚结束、采样器还没开始的边界上——也就是把模型外层的 `model.compute_logits` 包装起来。这一挪，`fallback_count` 立刻从 30 降到了 0，端到端吞吐比（当年 0.5B 冒烟负载上与原生之比）第一次稳定越过了 79%（达到 82.4%）。直到今天，你在 `pllm/backends/vllm/runner.py` 里看 pLLM 挂载模型时的第一行动作，依然保留着当年的教训：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/runner.py
def install_model_hooks(self) -> None:
    """Install the logits hook and announced hidden taps before vLLM compiles the model."""
    model = self.decoder.model
    model.compute_logits = self._compute_logits
    # ……
```

`compute_logits` 是一个天然的避风港：当控制流走到这里时，Transformer 主干的图回放刚刚落地，各层顺手写进固定缓冲区的 `hidden` 已经就位，而采样器还没开始工作。

---

## 第一步：把在线训练也录进去，以及为什么“零同步”没有商量余地

把触发点挪到 `compute_logits` 之后，前向预测能跑了。但 ESamp 不光要预测，它每一步还要**在线训练**：拿当前请求刚刚算出来的深层与浅层 `hidden`，算一次蒸馏 loss，反向传播求出梯度，再用 AdamW 更新这个请求专属的小网络参数。

我们先别管多租户并发，只盯着 1 个请求看。如果用最直白的 PyTorch 写一步训练，代码大概长这样：

```python
# check-docs: skip: illustrative sketch of one request's training step, not pLLM code
params = states[request_id]                  # 取出这个请求的参数
loss = (f(params, shallow) - deep).square().mean()
grads = torch.autograd.grad(loss, params)
adamw_step(params, grads, moments[request_id])
```

短短四行，清晰明了。但当你把这四行放进 128 并发的 decode 循环里跑起来时，灾难降临了。

在 GPU 底层，这四行代码会碎裂成五六十个零碎的小算子：收集参数、几层线性映射与激活、相减、平方、求均值、反向传播的一连串梯度算子、再加上 AdamW 里更新一阶矩、二阶矩和权重的逐元素操作。在还没有把训练装进 CUDA graph 的早期阶段，我们用 `py-spy` 剖析过 0.5B 模型在 128 并发下的 ESamp：pLLM 每步凭空多出了约 **4.6 ms** 的 CPU 宿主耗时，里头最大的一块蛋糕，就是 CPU 正在满头大汗地把这几十个训练小算子一个接一个往 GPU 上发（光反向传播一段就占了训练 CPU 时间的 36%）。一步 decode 本来才 3 ms，CPU 发射训练算子倒要花 4.6 ms，这还怎么玩？

结论只有一个：**不能只让模型前向坐自动钢琴，整个在线训练步——前向、loss、求导、优化器写回——也必须整包录进一张 CUDA graph 里。**

可一旦你决定给训练步录图，你就必须向自动钢琴签下三份严苛的生死状：

1. **地址必须固定**：图里焊死的是物理显存指针。如果某个张量在下一步换了块内存，回放时 GPU 还会蒙着眼睛去踩旧地址，轻则读出脏数据，重则当场触发非法内存访问崩溃。
2. **形状必须固定**：每个内核发射多少个线程块（Grid Size）在录制那一刻就定死了。哪怕线上流量忽高忽低，这一步有 128 个请求、下一步走了 3 个又来了 5 个，图里的张量形状也不能变。
3. **绝对的零主机同步（Zero Host Synchronization）**：录制中途如果 CPU 停下来等 GPU（比如调了 `.item()`），CUDA 驱动会直接报错拒绝捕获；而在回放的热路径上，哪怕图外有一次同步，也会把异步流水线砸得粉碎。

这第三条规矩，值得我们停下来算一笔账。因为在写系统代码时，人们太容易产生一种侥幸心理：“我就拷一个标量布尔值回 CPU 看一眼，几微秒的事，能有多大影响？”

我们在做这个项目的过程里，用血泪买到的一条铁律是：**少几次同步什么也换不来，只有把同步彻底变成零，性能才会跳变。**

为什么？因为在健康的异步推理引擎里，CPU 和 GPU 是在跳双人舞：当 GPU 正在埋头苦算第 $t$ 步的 Transformer 前向时，CPU 并没有闲着，它已经领先半步，在后台紧锣密鼓地准备当前步稍后或下一步所需的元数据了。在一次 0.5B、128 并发的剖析中（vLLM 0.27.1 时期），pLLM 每步在 CPU 上的全部准备工作大约要花 **700 µs**，最后通过一次耗时仅 **6.24 µs** 的非阻塞上传送进显卡。因为 CPU 提前干完了活，这 700 µs 完完全全藏在了约 3 ms 的 GPU 前向阴影背后，GPU 在两步之间的空闲间隙只有微不足道的 **4–6 µs**。

现在，假设你在代码里留了哪怕**一次** `.item()` 或 `if tensor.any():`。为了拿到这一个比特的布尔值，CPU 必须立刻踩下急刹车，原地阻塞，眼睁睁等着 GPU 把队列里积压的所有前向内核全部跑空。等 GPU 终于停下来把结果递给 CPU，CPU 这才开始慢吞吞地跑那 700 µs 的准备工作——而此时此刻，整张价值上万美金的显卡只能干坐着等 CPU！原本完美藏在阴影里的 700 µs，因为一次小小的同步，瞬间从“幕后”跳到了“台前”，严严实实地堵在了关键路径上。当时我们把热路径上残留的最后几次同步彻底拔除后，0.5B 上的 ESamp 吞吐直接从一万出头跃升到了 19,100 tok/s。

好，记住这三条铁律：**地址固定、形状固定、零同步**。现在我们挽起袖子，试着把在线训练一步步塞进图里。

---

## 第二步：两个请求与第一张小表

先看最简单的情况：如果世界上自始至终只有 1 个请求，录制训练图简直易如反掌。这个请求的参数张量就静静地躺在显存的某个固定地址上，我们把前向、反向和 AdamW 跑一遍录下来，以后每步回放就行了。

但只要来了**第 2 个请求**，第一道坎就出现了。

请求 A 的参数住在地址 `0x1000`，请求 B 的参数住在地址 `0x2000`。在连续批处理里，这一步 batch 的第 0 行可能是请求 A，下一步请求 A 结束了，第 0 行变成了新来的请求 C（住在 `0x3000`）。“每一步要读写的参数地址都在变”和“CUDA graph 要求地址绝对固定”，迎头撞在了一起。

怎么让固定的指针指向变化的目标？

所有系统程序员都会脱口而出那个经典答案：**再加一层间接（Indirection）**。

我们不再为每个请求单独分配零散的张量，而是预先向显卡申请一块地址永远不变的大张量——一个**分页状态池（Slot Pool）**，形状是 `[capacity, ...]`。请求 A 分到 0 号槽，请求 B 分到 1 号槽，请求 C 分到 2 号槽。在录制 CUDA graph 时，图里的内核只认这个大池子的基地址；至于“当前这一步，batch 里的第 $i$ 行到底该去几号槽拿参数、算完的新参数又该写回几号槽”，我们不再把它当成图的**结构**，而是把它当成普通的 `int64` **整数数据**，每一步填进一张同样地址固定的小表里！（关于这套状态池如何在多流并发下通过写时复制生长出不可变版本，我们在下一篇[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)里会彻底拆开讲。）

这样一来，图里的动作就变成了永远不变的两步走：先从固定地址的索引表里读出槽号 `slot`，再去固定地址的池子里按 `pool[slot]` 读写。请求怎么换、版本怎么升，变的都只是表里的几个整数，录好的图连一根毫毛都不用动。

> **旁注：一个看起来无害的 `torch.tensor` 是怎么偷走异步性的？**
> 既然每步都要把 CPU 算好的槽号表传给 GPU，你可能会随手写下一句 `table_gpu.copy_(torch.tensor(values), non_blocking=True)`，或者直接调 `torch.tensor(values, device="cuda")`。看起来很清爽对吧？
> 但如果你打开 profiler，会抓狂地发现 CPU 居然在等 GPU！为什么 `non_blocking=True` 失灵了？
> `pllm/execution/cuda_stream.py` 里的 `upload` 函数揭开了这个很多 PyTorch 老手都会踩的暗坑：Python 列表或普通 `torch.tensor(values)` 分配在操作系统的**可分页内存（Pageable Memory）**里。由于操作系统随时可能把这页内存换出到磁盘，GPU 的 DMA 引擎根本不敢直接在后台异步去读它——CUDA 驱动为了安全，会在底层悄悄把这次拷贝降级成同步等待，或者先阻塞主机流！
> 真正的异步上传，必须先踏进**锁页内存（Pinned Memory）**：
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
> 在每步都要高频执行的 `TensorExecutor` 里，连每次调 `.pin_memory()` 向缓存分配器借内存的开销我们都嫌贵。所以我们写了 `PinnedStaging`：维护几个预先锁好页的缓冲区，每步让 Rust 写好的 `_core.fill_table` 直接往锁页数组的 NumPy 视图里填数字，发完异步拷贝后录一个 `cuda.Event`。下一步要用时，只调非阻塞的 `event.query()` 问一句“上次拷完没？”——拷完了就原地复用，还在路上就再添一块新缓冲，绝不让 CPU 停下来等半步。

---

## 第三步：请求数每步都在变？造一个“哑状态”当黑洞

有了固定池和槽号表，两个请求搞定了。但线上的真实流量不是恒定的：这一步 batch 里有 30 个请求，下一步涌进来 15 个变成 45 个，再下一步只剩 12 个。

而 CUDA graph 的第二条铁律正冷冷地盯着我们：**张量形状必须固定**。

你可能会想：“vLLM 录模型前向的时候，不是按 `batch_size = 1, 2, 4, 8, 16, 32...` 分档位（Buckets）录了好几张图吗？我们也给训练按档位录十几张图不就行了？”

对无状态的模型前向来说，分档录图很划算；但对有状态的在线训练任务来说，影响形状的维度有两个：当前步的打包行数 $R$，以及同时活跃的请求状态数 $S$。更要命的是，每录一张图，CUDA 都会为这张图的中间激活值和反向梯度分配一个**私有的显存池（Graph Pool）**。如果按 $(R, S)$ 的组合分档录上十几张图，光这些图池占用的常驻显存就会把 KV cache 挤爆。

怎么只录**一张图**，却能应付从 1 到 $R$ 的任意行数、从 1 到 $S$ 的任意状态数？

我们的解法朴素得近乎耍赖：**永远按任务声明的最大容量（最大行数 $R$、最大状态数 $S$）录唯一的一张图；至于没坐满的空座位，全部指给一个不会喊疼的“假人”。**

具体来说，如果最大状态容量是 $S$，我们在规划读取表时，会故意多留出一列，做成 $S+1$ 列（下标 $0 \dots S$）。其中 $0 \dots S-1$ 是真正活着的请求状态，而第 $S$ 号则是我们专门设立的**“哑状态”（Padding Sink）**。

假设最大容量 $R=128, S=128$，而这一步实际上只来了 30 个请求。CPU 在填表时，会把前 30 行指向它们各自真实的状态号（$0 \dots 29$），并把 `active` 掩码标为 `1`；而剩下的第 $30 \dots 127$ 行，统统指向第 $S$ 号哑状态，同时把 `active` 标为 `0`。

```text
Step Table (Fixed Shape R=128)         State Slot Pools & Sink
┌───────────────────────────┐
│ Row 0   (active=1, st=0)  │─────────► State 0   [ read slot / write slot ]
│ Row 1   (active=1, st=1)  │─────────► State 1   [ read slot / write slot ]
│ ...                       │           ...
│ Row 29  (active=1, st=29) │─────────► State 29  [ read slot / write slot ]
├───────────────────────────┤
│ Row 30  (active=0, st=S)  │──┐
│ ...                       │  ├──────► State S (Padding Sink):
│ Row 127 (active=0, st=S)  │──┘        • sources [F, S+1]  -> slot -1 (reads 0.0)
└───────────────────────────┘           • loss reduction    -> weight 0.0 (ignored)
                                        • destinations [F,S]-> NO column S! (never written)
```

顺着这张图看过去，你会发现这个“第 $S$ 号假人”设计得有多严丝合缝：
* **读取时（`sources` 形状为 `[F, S+1]`）**：哑状态对应的源槽号填的是 `-1`。负责从池子里捞参数的 `_gather_field` 一眼看到槽号 `< 0`，就会直接把那一行涂成全 `0.0`。假人拿到了一身全零的干净参数，不会触发任何 `NaN`。
* **求导与归约时**：虽然 GPU 上的前向和反向内核还是老老实实地跑满了全部 128 行，但因为空行的 `active=0`，在算 loss 归约和按状态累加梯度时，假人所在行的权重严格为 `0`。它算出来的任何数字，连一丁点儿都不会混进真实请求的梯度里。
* **写回时（`destinations` 形状为 `[F, S]`）**：注意到了吗？目标槽表比源槽表**少了一列**，压根就没有第 $S$ 列！优化器只更新前 $S$ 个真实状态，哑状态永远没有写入目标。

就这样，靠着一个“读出来是零、权重是零、写回时隐形”的第 $S$ 号哑状态，无论外面是 1 个请求还是 128 个请求，GPU 里的 CUDA graph 看到的永远是整整齐齐的 `[R, ...]` 和 `[S, ...]`。形状彻底焊死，图一次都不用重录。

---

## 第四步：谁该更新？把选择留在显卡里

解决了可变请求数，我们又迎来了算法作者常提的另一个需求：**按条件选择性训练**。

比如在很多在线学习算法里，不是每一行 token 都要触发参数更新——也许我们只想在预测熵大于某个阈值、或者满足特定掩码的行上训练（通过 `rows=` 或 `update=` 选择器指定）。

如果是在 Eager 模式下，这太好写了：在 GPU 上算出布尔掩码 `mask`，调一句 `selected_states = states[mask].unique().tolist()` 把选中的状态号拉回 CPU，然后 CPU 只为这些被选中的状态分配新版本、发起优化器更新。

停！听到 `.tolist()` 的一瞬间，你的警报器应该已经拉响了——**那是主机同步！**

只要 CPU 想知道“这一步到底哪几个状态被动态条件选中了”，CPU 就必须等 GPU 把前面的条件算出来。这直接违反了我们的第三条铁律。可如果 CPU 不去读这个掩码，负责在 CPU 上管理状态版本账本的 PTTS 怎么知道这一步该给哪几个状态发布新版本呢？

当时我们盯着这个问题想了很久，最后灵机一动：**谁规定“没被选中的状态”就不能发布新版本了？**

如果这一步批里有 30 个活跃请求，CPU 根本不去打听 GPU 上的掩码算出了什么结果，而是干脆利落地假定**这 30 个活跃状态全部参与了本步发布**，在 CPU 账本上统统给它们分配好新槽位、推进版本号。

真正的筛选，100% 留在 GPU 内部完成。来看看 `pllm/execution/tensor_executor.py` 里的 `_selection` 是怎么算的：

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

看那行 `index_add_`：GPU 自己在设备端统计出每个状态在这一步有几行被选中（`updates = count[: self.state_count] > 0`）。接着，当图里的优化器去写目标槽时：
* 如果某个状态的 `updates[s]` 是 `True`，AdamW 就把更新后的新权重和动量写进新槽位；
* 如果某个状态的 `updates[s]` 是 `False`（说明它这一步其实没被选中），图里的内核就直接把它的旧字段**逐字节原封不动地拷进新槽位**！

这是一笔极其划算的系统交易：我们在 GPU 显存里多做了一点点微不足道的拷贝，在版本树上多挂了一个内容没变的新版本号，却彻底买断了“CPU 必须等 GPU 算完掩码”的同步依赖。选择留在了设备端，CPU 继续蒙着眼睛大步往前跑。

---

## 第五步：当别人的张量搬家时——把裸指针塞进表里

走到这一步，我们的 CUDA graph 就像一座秩序井然的封闭小城：凡是归 pLLM 自己管的状态池、中间缓冲、结果槽，全都住进了预先盖好的固定公寓里，门牌号生老病死都不变。

但城外还有别人。

在线训练有时候需要读取一些**不归 pLLM 分配**的输入张量——比如采样器传过来的反馈张量，或者外部刚拼好的 token 输入。这些张量归宿主引擎 vLLM 管。vLLM 并不知道我们在偷偷录自己的训练图，它在某些步心情好，就会把旧张量释放掉，换一块新分配的显存。

如果你傻乎乎地把这个外部张量直接作为参数传进要录制的算子里，CUDA graph 就会死死记住它录制那一刻的地址 `0x7f00...`。等到下一步，vLLM 给它搬了家（比如搬到了 `0x7f88...`），而我们录好的图还在按图索骥去敲旧门牌——轰的一声，不是读出一堆垃圾，就是直接段错误崩溃。

怎么在一张连指针都被焊死的 CUDA graph 里，去读一个每步都在搬家的外部张量？

如果你写过 C 语言，你脑子里大概已经蹦出那个主意了：既然图只认固定地址，那我们就**不要把那个外部张量录进图里，而是由 CPU 每步调一次 `tensor.data_ptr()`，把拿到的 64 位内存地址当成一个平平无奇的 `int64` 整数，写进我们那张固定地址的小表里！**

来看看 `pllm/kernels/page_copy.py` 里的 `_gather_rows`。这是我在这个代码库里最偏爱的一段小魔法：

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
    address = tl.load(descriptor).to(tl.pointer_type(output.dtype.element_ty))
    stride0 = tl.load(descriptor + 1)
    stride1 = tl.load(descriptor + 2)
    index = tl.load(rows + row)
    value = tl.load(address + index * stride0 + column * stride1, column < width, 0)
    tl.store(output + row * width + column, value, column < width)
```

请盯住 `address = ...` 那一行。

在 PyTorch 的日常世界里，张量是神圣的高层对象，你几乎不会去碰它底层的裸指针。但在这里，CPU 每一步把外部张量的首地址 `data_ptr()` 连同它的两个维度步长（`stride(0)` 和 `stride(1)`），打包成三个 `int64` 整数，填进固定表的 `descriptor` 段里。到了 GPU 上，Triton 的 `.to(tl.pointer_type(...))` 轻轻一挥魔杖，把这个从表里读出来的 64 位整数，当场强转成了一个带类型的设备指针（Device Pointer），下一行直接用 `tl.load(address + ...)` 去解引用！

```text
Fixed Table in GPU HBM (Address 0xAAAA0000 - NEVER changes)
┌─────────────────────────────────────────────────────────┐
│ descriptor[0]: 0x7f8812340000  (int64 from .data_ptr()) │
│ descriptor[1]: 3584            (int64 stride0)          │
│ descriptor[2]: 1               (int64 stride1)          │
└───────────────────────┬─────────────────────────────────┘
                        │ 1. tl.load(descriptor) -> int64
                        │ 2. .to(tl.pointer_type(...)) -> T*
                        ▼
External vLLM Tensor (Moved to 0x7f8812340000 this step!)
┌─────────────────────────────────────────────────────────┐
│ [ row 0 ][ row 1 ][ row 2 ] ...                         │
└─────────────────────────────────────────────────────────┘
```

为什么脾气暴躁的 CUDA graph 不会跟它急眼？

因为对录像机来说，它录下的内核参数是 `descriptor`——而 `descriptor` 是我们固定表的切片，地址一万年不变！录像机记下的动作仅仅是：“去 `0xAAAA0000` 这个固定地址读三个整数，再顺着第一个整数指的方向去搬数据”。至于那个整数今天指向东街、明天指向西街，那是运行时流过的数据，不是录制时的图结构。一行类型转换，就把“图的生命周期”和“外部张量的内存生命周期”干净利落地解耦了。

现在，把我们在第二步、第三步和第五步里被逼出来的零件拼在一起，你就看懂了 `pllm/execution/tensor_executor.py` 里那张贯穿整个系统的核心数据结构——`TableLayout`：

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

没有一行是多余的装饰：
* `rows [6, R]` 交代了每一行的身份、对应哪个状态（空行指给第 `S` 号哑状态）以及 `active` 掩码；
* `sources [F, S+1]` 和 `destinations [F, S]` 把可变的状态版本映射成了固定的物理页槽，并用相差一列的维度彻底封死了哑状态的写入；
* `row inputs [K, 3]` 则装下了那 `K` 个随时可能搬家的外部张量的裸地址与步长。

整张表全都是紧凑的 `int64`。在 0.5B、128 并发的 ESamp（$\beta=0$）剖析中（vLLM 0.27.1 时期），把这四段全部算上，每一步从主机锁页内存拷进显卡的总字节数只有区区 **23,692 字节**，耗时 **6.24 µs**（作为参照，原生 vLLM 自己每步上传元数据是 1,088 字节、耗时 2.50 µs）。用每步多花 4 微秒上传二十几 KB 整数的代价，我们换来了整张计算图永远不用重录的特权。

---

## 第六步：按下录音键时的两个魔鬼细节

万事俱备，只欠按下录音键了。负责这件事的是 `pllm/execution/step_capture.py` 里的 `StepCapture`。

按理说，调用 PyTorch 的 `with torch.cuda.graph(graph):` 也就是十来行模板代码的事。但当你真的把这套包含状态池原地写入的训练体放进高并发引擎里去录时，有两个只有亲手踩过才会懂的魔鬼细节跳了出来。

第一个细节：**预热（Warmup）会弄脏真数据**。

在正式录制 CUDA graph 之前，PyTorch 建议在捕获前先在一条旁路流上把函数预热几遍（pLLM 的 `capture_graph` 用 `for _ in range(3)` 跑三遍），以便让 cuBLAS、Triton 等底层库完成工作区分配和算法初始化。但别忘了，我们的 `body()` 不是纯只读的前向，它的尾巴上挂着真刀真枪的 AdamW 优化器，会往 `destinations` 指向的池子槽位里写新权重！如果在某一步触发录制时直接跑三次预热，池子里正在服务线上请求的真实参数就会被预热的垃圾梯度覆盖三次。

看看 `StepCapture._capture` 是怎么用一行代码化解这个危机的：

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

在进入 `capture_graph` 之前，我们先调了一句 `destinations.fill_(-1)`，把表里所有的目标写槽统统涂成 `-1`！

因为不论是分页拷贝 kernel 还是分页 AdamW kernel，内部都有一条判断 `dst >= 0` 的掩码守卫，看到 `-1` 就会跳过最终的存储指令。于是，三次预热外加一次录制，整整四遍跑下来，所有的算子都得到了充分的预热，显存池也一分不差地分配好了，但对于保存真实请求参数的状态池来说，这四遍执行**完完全全是只读的**。等录完图回到 `StepCapture.run`，再调一次 `refill()` 把当前步真正的目标槽号填回表里，然后从容地按下 `self.graph.replay()`。（顺便看一眼最后那行 `assert`：图池需要吃多少显存，我们在构建期就已经用最大形状的虚拟追踪算好并预留了预算；如果录出来的实际字节数超过了预算，那是框架自己的 bug，当场断死，绝不偷偷去抢 KV cache 的显存。）

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

为什么录个 GPU 的图，还要去关 Python 的循环垃圾回收器（`gc.disable()`）？

因为 Python 的循环 GC 是按对象分配计数随时触发的。当你的代码正在 `with torch.cuda.graph(...):` 里面一行行往下走、分配各种中间张量对象时，Python 的 GC 极有可能突然在半路醒来，决定清理一批早已死掉的循环引用对象。如果那批死对象里恰好包着一个曾在**另一条 CUDA stream** 上用过的旧张量，它的 C++ 析构函数就会在此时此刻向 CUDA 缓存分配器发出跨流释放通知——而正在录图的 CUDA 驱动看到捕获窗口里突然插进来一个属于别的流的非法动作，会当场掀桌子报错：`operation not permitted when stream is capturing`！

PyTorch 自己的 `torch.cuda.graph` 上下文管理器虽然会在进入录制前主动调一次 `gc.collect()`，但那拦不住录制中途因为新建对象过多而再次触发自动回收。加上这几行 `gc.disable()`，那种偶发的捕获崩溃才终于销声匿迹。

至于被录进图里的训练体本身，反倒简单得让人松一口气：它就是最普通的 eager PyTorch 代码。按槽号从池里捞出参数，在 `torch.enable_grad()` 下跑完用户写的数学函数，调 `autograd` 求导，最后交给优化器写回。对于最常用的 FP32 AdamW，我们写了一个分页的 Triton kernel（`pllm/kernels/adamw.py`），直接拿着 `sources` 槽号去读旧权重和一阶/二阶动量，算完直接往 `destinations` 的新槽位里写——连写时复制（Copy-on-Write）事先拷一遍旧页的功夫都省掉了。

不妨对比一下今昔：在旧版系统（旧 tLLM）里，为了让训练放进 CUDA graph 后回放不出错，我们曾不得不向工程现实低头，把优化器从 AdamW 硬生生降级成了没有动量状态的 SGD。而今天，AdamW 的全部五个状态字段都稳稳当当地住在固定地址的池子里，借着一张 `int64` 小表进出自由，再也不需要牺牲算法精度去迁就系统了。

这套通用捕获机制在 0.5B（128 并发，rank 4）上合入主线那天，每步耗时直接从 6.73 ms 降到了 5.25 ms（其中提交 token 触发训练的那一段从 3.32 ms 砍到了 1.56 ms），吞吐从 19,100 tok/s 跳到了 23,829 tok/s，提升了整整 25%。在当时 765 步的稳态 decode 剖析轨迹里，每一步恰好只有一次训练图回放，重录制次数为 0。

---

## 插曲：为什么我们试过 `torch.compile`，结果反而更慢？

把整个训练步录进 CUDA graph 之后，一个非常自然的念头在我们脑子里冒了出来：

虽然发射开销被 CUDA graph 吃掉了，但训练图里面毕竟还有五六十个零碎的逐元素小 kernel。既然 PyTorch 2 有强大的 `torch.compile`（Inductor 编译器），我们为什么不顺手给训练体套上一层 `@torch.compile`，让编译器把这些碎算子融合成几个大 kernel 呢？岂不是锦上添花？

动手试一试总是比争论快。我们在 0.5B 模型（vLLM 0.27.1，128 行，rank 4，$\beta=0$，未编译时吞吐为原生的 92.45%）上老老实实做了两组对照实验。

**第一次实验**，我们只用 `torch.compile` 编译用户写的 loss 前向与反向函数，外围的填表和分页优化器保持原样。跑完 7 轮取中位数，吞吐是原生的 **92.77%**——和不编译相比，差距完全淹没在测量噪声里。

冷静下来想想，这其实一点也不奇怪：在小算子序列已经被录进 CUDA graph 的前提下，CPU 的发射开销早就归零了；而 rank 4 的小网络中间张量非常小，原本就全都泡在 GPU 的 L2 cache 里，把两三个逐元素操作融合成一个 kernel，省下的那几微秒显存读写根本听不见响。

“那如果步子迈大一点呢？”我们接着做了**第二次实验**：把整个被 `StepCapture` 录制的 `execute_body`——包括从池里 gather 参数、算 loss 归约、路由梯度、准备优化器输入——统统打包交给 `torch.compile`，然后再把编译出来的产物录进 CUDA graph。

结果出来了：吞吐不仅没涨，反而掉到了原生的 **90.10%**！

怎么会越编译越慢？我们打开 profiler 的时间线一看，顿时哭笑不得。

编译器确实很卖力地干了融合的活：旁路流上每步发射的 kernel 数量从 59 个大幅砍到了 25 个。但是，旁路流每步消耗的 GPU 总时间，却从原来的约 **178 µs** 暴涨到了约 **257 µs**！

多出来的近 80 微秒花在哪儿了？时间线上赫然躺着一个名叫 `triton_poi_fused_clone_paged_adamw_slice_0` 的陌生 kernel，每步白白吃掉约 39 µs；与此同时，紧随其后的分页 AdamW kernel 耗时也从约 29 µs 膨胀到了 64 µs。

看到名字里那个刺眼的 `clone`，真相大白了。

罪魁祸首是 PyTorch 编译器底层的 **Functionalization（函数式化）** 机制。AOTAutograd 在对计算图做变换和优化之前，有一个铁律：图里所有的算子都必须是纯函数（Pure Functions），绝不允许有原地修改（In-place Mutation）。而我们的状态池和分页优化器，偏偏是围绕着“在一块巨大的预分配固定张量池里原地读写指定页槽”来设计的！

当编译器看到我们的代码试图往状态池的切片里原地写数据时，为了捍卫它自己的纯函数语义，它在生成的代码里默默地把状态池切片**克隆（clone）了一份副本**，在副本上改完再拷回去。即使我们试着用 `torch.compiler.disable` 把优化器那一段圈出去，追踪图边界上的保护性复制依然阴魂不散。

这次失败的实验给我们上了一堂极好的系统课：**“纯函数式图变换”与“原地写固定分页池”是两种截然相反的世界观。** 对已经装进 CUDA graph 的小算子序列来说，编译器能省的发射开销已经没了，能抠出来的算子融合收益微乎其微，而两个世界观碰撞产生的防守性拷贝却会实打实地咬掉你的显存带宽。这也是为什么我们在文档里明确劝阻用户：不要在 `@pllm.train` 方法上手工套 `@torch.compile`——把编译留给真正能发挥威力的纯函数场景（比如《图内挂载点》的挂载点方法体、《参数覆盖层》的覆盖层梯度、采样器里的引导重加权和《有界的零拷贝 gather 与预取》里的只读查表融合）。

（顺便提一嘴，在那次性能深挖里我们还撞上了另一个反直觉的跟头：我们原以为把采样器前面的流等待去掉几个，让旁路训练跟更多的主干阶段重叠起来会更快；结果去掉等待后，吞吐反而从 94.49% 掉到了 92.73%——因为训练一脚踏进了正在满负荷扫词表的采样阶段，两边在窄窄的显存总线上撞车了。关于什么时候重叠才不是零和博弈，我们留到第 11 篇[《自动放置与重叠》](11-automatic-placement-overlap.md)再细算。）

---

## 天下没有免费的午餐：固定形状的代价与规模的天平

回顾这一路，我们靠着固定地址的状态池、第 $S$ 号哑状态、设备端掩码选择、以及把裸指针塞进 `int64` 表里的技巧，终于把一个活生生的在线学习算法塞进了死板的 CUDA graph 里，做到了稳态每步零次重录、零次主机同步。

但正如我们在系统工程里一贯信奉的：**天下没有免费的午餐，你省掉的每一次动态开销，账单一定会寄到别的地方去。**

第一张账单，叫作**空位陪跑税（Padding Waste）**。因为我们按最大容量（比如 $R=128$）只录一张图，当线上处于低谷期、某一步实际上只有 16 个活跃请求时，剩下的 112 行虽然指向了权重为 0 的哑状态，但 GPU 里的矩阵乘和反向传播并不会跳过它们——那 112 个假人依然要老老实实陪着跑完全程。同时，每个捕获任务都独占着一块按最大形状保守推导出来的私有 graph 显存池，哪怕当前没有请求在跑，这块显存也不会退还给 vLLM 的 KV cache（如果启动时预算不够，会在构建期直接报错拦下）。当第一次有请求进来触发冷录制（或者底层 `hidden` 缓冲换了存储重新录制）时，`capture_graph` 里的三次预热加一次 `stream.synchronize()` 也会让首个请求感受到一下微小的毛刺。

更深远的影响在于，这种“112 个假人陪着 16 个真人一起跑”的机制，直接给用户编写算法立下了铁律：

* 你不能在方法里写 `if loss < 0.01: return` 这种基于张量数值的 Python 分支（在规划期用 meta 张量追踪时，任何 `.item()` 或张量布尔判断都会被当场举红牌拒绝），所有的条件逻辑都必须改写成 `torch.where` 这样的逐元素掩码；
* 你的训练方法必须**在行维度（第 0 维）上严格独立**，返回形状为 `[R]` 的逐行 loss，把最终的 `mean` 或 `sum` 归约交给框架去做。你绝不能在自己的代码里写一句 `return loss.mean(dim=0)`，更不能在行与行之间用 BatchNorm 或跨行注意力——想想看，如果你自己在第 0 维上求了均值，那 112 个陪跑假人算出来的 `0.0` 就会把 16 个真人的梯度稀释得面目全非！

那么，把这笔账单算总账，这套架构在实际模型上到底能跑多快？这取决于你站在多大的模型面前：

| 模型规模（128 并发，RTX 4090） | 原生 vLLM 一步耗时 | ESamp ($\beta=0$, `publish_lag=1`) | ESamp ($\beta=0$, `publish_lag=0`) | 旁路训练 GPU 耗时 / 占比 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-7B-Instruct** | ~21.7 ms (5,887 tok/s) | **97.4%** (5,738 tok/s) | **97.2%** (5,731 tok/s) | ~616 µs（其中 AdamW ~191 µs），占一步的 **~2.8%** |
| **Qwen2.5-0.5B-Instruct** | ~3.0 ms (42,355 tok/s) | **91.9%** (38,936 tok/s) | **89.8%** (38,047 tok/s) | ~178 µs（其中 AdamW ~29 µs；vLLM 0.27.1 时期的剖析），固定调度与拷贝开销凸显 |

看这张对比表，物理规律一目了然。

在 7B 模型上，主干网络跑一步 decode 要花 21.7 ms；而旁路流上回放一整张训练图（包含参数收集、前向、反向和约 191 µs 的分页 AdamW），总共只花掉约 **616 µs**，不到主干一步的 3%。因此，开启允许滞后一步发布的 `publish_lag=1` 时，端到端吞吐高达原生的 **97.4%**；哪怕你强制要求 `publish_lag=0`（下一步前向必须串行等待当前步训练完成），吞吐依然有 **97.2%**。那大约 2.6% 的差额，基本就是训练内核本身实打实吃掉的那 616 µs 硬件算力与带宽。

而当我们切到一步只有 3.0 ms 的 0.5B 袖珍模型时，训练图本身的回放虽然只有一百多微秒，但由于我们的训练图目前还没有和 vLLM 的主模型图合并成同一张大图，相邻任务之间填表、锁页拷贝、结果拷出以及版本锁定与释放的固定交接开销（哪怕每次只有几微秒到十几微秒）就从海面下浮了出来，把吞吐定格在了 91.9%。我们没办法违反能量守恒定律让一个额外的在线训练步在 3 ms 的窗口里凭空消失，但我们做到了让它不阻塞主机、不打断图回放、安安静静地贴着硬件底线跑完。

如果从这趟跟“自动钢琴”斗智斗勇的经历里，只能带走一条可以在别的系统里复用的经验，我想就是这句话：

**图是死的，表是活的。**

当你不得不把一段充满动态变化的算法塞进一个要求地址固定、形状固定、零同步的底层硬件容器时，千万不要试图去动态修改容器的结构（比如频繁重编译或重录图）。退后一步，把所有每一步会变的东西——哪些请求在场、读哪一页版本、谁被选中更新、甚至一个外部张量此刻搬到了哪个内存地址——统统降维成一张固定表里的几个 64 位整数。当变化全部坍缩成数据时，最死板的机器，也能弹出最灵活的曲子。
