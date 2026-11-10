# 有界的零拷贝 gather 与预取

在推理时算法的花园里，有一类算法长得越来越像数据库系统：模型一边往前生成 token，一边拿当前 token 和前几个历史 token 算一个哈希，去一张硕大无比的外部查找表里捞出几行向量，过一道门控加进某层的残差流，或者直接叠到词表的 logits 上。DeepSeek 的 Engram 记忆层是这个路数，经典的 kNN-LM 与 n-gram 先验也是这个路数。

如果只看算法逻辑，代码干净得让人心情愉悦。在 pLLM 的示例 `examples/engram_lite.py` 里，核心的查表方法只有区区一行：

```python
# check-docs: skip: excerpt from examples/engram_lite.py
def lookup(
    self,
    ids: torch.Tensor = pllm.input_tokens(history=2),
    table: torch.Tensor = pllm.parameter("memory", "table"),
) -> torch.Tensor:
    """Prefetch two rows per model token, independently of the hidden state."""
    return table[ngram_keys(ids, self.buckets)].flatten(1)
```

把前两个历史 token 连同当前 token 算出两个哈希桶号（二元组和三元组），从 `table` 里把对应的两行切出来，拍平，收工。

真正让人睡不着觉的不是这一行代码，而是 `table` 的体型。在我们的查表记忆教程（示例 11）里，一张词表宽度的 n-gram 先验表，形状是 `2048 × 152064`（BF16），光躺在那儿就要占掉 **594 MiB**；而在真实的 Engram 场景下，记忆表动辄膨胀到几个 GiB 甚至更大。在线推理服务的显存是一场残酷的零和游戏——你往显存里塞进 1 GiB 的静态表，vLLM 的 KV cache 池就得吐出 1 GiB，能同时服务的并发请求数当场跟着缩水。

“那还不简单？”你大概会脱口而出，“主板上插着几百个 G 的 CPU 内存呢！把大表放到主机内存里，每一步需要哪几行，临时从 PCIe 总线上拉过来不就行了？”

想法完全正确。于是我们真的去写了一个最标准、最教科书式的 GPU gather kernel，放到后台的旁路 CUDA stream 上，满心欢喜地指望它在后台偷偷把数据搬过来，和前台的模型前向计算完美重叠。

结果它一跑起来，直接把前台的 Transformer 前向给饿死了。为了把后台搬数据的耗时藏起来，我们最后不得不做了一件极度反直觉的事：**把读取数据的 GPU 线程块数量狠狠砍到只剩 SM 数的四分之一。** 线程块越少，整个步反而跑得越快。

这篇文章讲的就是这根 PCIe 细水管引发的连环案：为什么现成的硬件 DMA 拷贝引擎在这里完全指望不上；为什么一个写法再正常不过的 kernel 会制造出一场硬件调度器级别的交通大拥堵；以及我们怎么在不改动用户那一行 `table[keys]` 的前提下，在底层完成这场偷梁换柱。

> **实验环境说明**：如无特别说明，本文引用的基准测试与剖析数据均在单张 NVIDIA RTX 4090（24 GiB，128 个 SM，PCIe 4.0 x16）、vLLM 0.32.0、PyTorch 2.13（CUDA 12.9）环境下测得。主模型为 `Qwen/Qwen2.5-7B-Instruct`（BF16），开启 CUDA Graph 与 FlashInfer 采样器；吞吐测试并发度为 128 行（8 个 prompt × `n=16`），每条生成 256 个 token，预热 1 轮后取 3 轮中位数。文中涉及 `Qwen2.5-0.5B-Instruct` 的数据均会显式标出，用于对比固定传输开销在小模型极短步长下的放大效应。

---

## 先算一笔过桥费

在动手写代码之前，我们先拿信封背面算一笔物理账。

假设线上正跑着 128 个并发请求。每一步 decode，每个请求要从那张 `2048 × 152064` 的 BF16 大表里查一行。一行有 152,064 个双字节浮点数，也就是 `304,128` 字节（约 297 KiB）；128 个请求加起来，每一步需要跨过 PCIe 搬运 **38.9 MB** 的数据。

我们的 RTX 4090 插在 PCIe 4.0 x16 插槽上，实测 GPU 直接跨总线读主机内存能跑到的有效带宽大约是 **23.7 GB/s**。用小学除法算一下：

$$\frac{38.9 \text{ MB}}{23.7 \text{ GB/s}} \approx 1.64 \text{ ms}$$

1.6 毫秒是什么概念？在 7B 模型上，跑完一整步 28 层 Transformer 的 decode 大约需要 **21.8 ms**，1.6 ms 大约占了整步时间的 7.5%。但如果你换成像 0.5B 这样的小模型，一步 decode 总共才 **3.0 ms** 出头！如果让这 1.6 ms 大摇大摆地横在关键路径上，GPU 光等数据过桥就要花掉半步的时间，小模型的吞吐当场就得腰斩。

好消息是，查表所依赖的输入仅仅是**刚刚采出来的 token ID**，它完全不需要等第 0 层或第 10 层的隐藏状态（`hidden`）算出来。从逻辑依赖上看，每一步最开始（`request.prepare` 阶段），查表就可以在一条独立的旁路流（side stream）上启动了。只要这 1.6 ms 的跨桥搬运能和主干流上 21.8 ms（哪怕是 0.5B 的 3 ms）的模型前向**同时发生**，它就应该能严丝合缝地躲进前向计算的阴影里，变成“免费”的午餐。

怎么把这 128 行从 CPU 内存搬上 GPU，还不打扰主干流？

---

## 为什么硬件拷贝引擎帮不上忙？

做系统工程的人听到“跨 PCIe 搬数据还要跟计算重叠”，肌肉记忆里的第一反应一定是 **DMA（直接内存访问）**。

现代 GPU 芯片上专门配有独立的硬件拷贝引擎（Copy Engines）。当你调用 `cudaMemcpyAsync`、带步长的 `cudaMemcpy2DAsync`、批量拷贝接口，或者像 vLLM 换入换出 KV 块那样调用 `swap_blocks` 时，干活的是拷贝引擎。它最大的美德就是**完全不占用 SM（流多处理器）**：SM 在那边安心算矩阵乘，拷贝引擎在这边闷头搬内存，两边井水不犯河水。

我最开始也想走这条阳关大道。但只看了一眼接口签名，心就凉了半截。

无论是哪种 `cudaMemcpy` 变体，都有一个铁打的前提：**要拷贝的源地址（或者源地址的指针数组），必须在 CPU 主机端是已知的。** 是 CPU 在组装 DMA 命令包时，亲手把源内存地址填进描述符里，再推给硬件队列。

可我们的行号（`keys`）在哪儿？在 **GPU 显存**里！

每一步的当前 token ID 是上一步采样器刚刚在 GPU 上吐出来的，历史 token 也是 vLLM 在 GPU 上维护的，哈希桶号 `ngram_keys(ids, buckets)` 更是在 GPU 上当场算出来的。CPU 从头到尾根本不知道这一步那 128 个请求到底抽中了哪 128 行。

如果你非要动用拷贝引擎，唯一的办法就是先把这 128 个整数从 GPU 拷回 CPU（一次 `device -> host` 的同步等待），让 CPU 算出 128 个主机内存指针，再由 CPU 发起异步拷贝。而在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里我们已经立下了死规矩：在步长以毫秒计的 decode 循环里，任何一次主机同步都是毁灭性的——它不仅会当场掐断 CPU 提前发射流水线的节奏，还会让整个方法彻底丧失录进 CUDA Graph 的资格（何况录进 CUDA Graph 里的 `memcpy` 节点，源地址参数在录制那一刻就已经焊死了，回放时根本改不了）。

拷贝引擎指望不上，路就只剩下了唯一的一条：**不拷贝了，让 GPU 的计算 kernel 自己伸手去主机内存里拿。**

要完成这个动作，我们需要凑齐两块拼图：
1. **锁页内存（Pinned Memory）与统一虚拟地址空间（UVA）**：普通的操作系统内存随时可能被换出到磁盘（pageable），GPU 硬件不敢直接碰。但只要我们调用 `.pin_memory()` 把那张大表钉死在物理内存里，现代 CUDA 驱动就会通过 UVA 给它分配一段 GPU 也能看懂的虚拟地址。
2. **一个“装作在 GPU 上”的视图张量**：PyTorch 默认不让你把一个 `cpu` 张量直接传给 CUDA kernel。好在 vLLM 的工具箱里刚好躺着一个现成的宝贝函数 `get_accelerator_view_from_cpu_tensor`。你喂给它一个锁页的 CPU 张量，它会还给你一个 `is_cuda == True` 的张量外壳——如果你打印它们俩的 `.data_ptr()`，会发现底层的 64 位内存地址分毫不差！它是纯粹的别名（alias），没有发生一个字节的拷贝。

拿到这个视图之后，GPU 线程只需要对着这个地址发出一道平平无奇的 `tl.load` 指令，SM 的内存单元发现地址落在主机区间，就会自动发起 PCIe 读事务，把字节跨越总线拉进 GPU 寄存器。这就是所谓的**零拷贝（Zero-Copy）**：没有显式的搬运工，读取本身就是传输。

行号留在 GPU 上，读取由 GPU 自己发起，CPU 全程连眼皮都不用抬一下，还能原封不动地录进 CUDA Graph。

看起来，我们已经赢了。直到我们写出了第一个 kernel。

---

## 一个再正常不过的 Kernel，一场意外的拥堵

为了把当时撞上的怪现象看得一清二楚，我们把问题从庞大的 vLLM 引擎里剥离出来，写了一个只有一百多行的独立玩具脚本。

玩具实验的布景很简单：
* 在锁页主机内存里放一张 `2048 × 151936` 的 BF16 大表（约 594 MiB），用 `get_accelerator_view_from_cpu_tensor` 造出它的 UVA 视图；
* 在 GPU 上随机生成 128 个行号（每步共需读取 `37.1 MiB`）；
* 用 4 组 $4096 \times 4096$ 的 BF16 大矩阵乘法（`x = x @ b`）拼成一个假的模型“前向（forward）”，在 RTX 4090 上刚好耗时约 **3.88 ms**；
* 把读表的 gather kernel 放到旁路 `side` 流上，把假前向放到主干 `main` 流上，两边同时开跑，用 CUDA Event 量一量到底能省下多少时间。

如果你写过一两周的 CUDA 或 Triton，让你写一个从二维表里按行号抽数据的 gather kernel，你闭着眼睛写出来的代码一定长成下面这个 `naive_rows` 的样子：

```python
# check-docs: skip: excerpt from toy_host_gather.py, a toy experiment outside the repository
@triton.jit
def naive_rows(source, rows, output, width: tl.constexpr, block: tl.constexpr):
    # 每个 program 只负责一个 (行, 列 tile)；grid 跟数据量成正比。
    tiles = tl.cdiv(width, block)
    row = tl.program_id(0) // tiles
    column = tl.program_id(0) % tiles * block + tl.arange(0, block)
    index = tl.load(rows + row).to(tl.int64)  # 行号在 GPU 上，kernel 内部自己取
    value = tl.load(source + index * width + column, column < width, 0)
    tl.store(output + row * width + column, value, column < width)
```

每个线程块（Triton 里叫一个 `program`）负责搬运某一行里的一个连续列块（`block = 2048` 个 BF16，即连续的 4 KiB 字节）。一行宽 151,936，切分成 $\lceil 151936 / 2048 \rceil = 75$ 个列块；一共要取 128 行，所以启动网格（grid）的大小是 $128 \times 75 = \mathbf{9,600}$ 个线程块。

数据有多少块，grid 就开多大——这是 GPU 编程入门第一课教的金科玉律。

现在，我们在它旁边写第二个版本 `bounded_rows`。算术逻辑连一个标点符号都不改，唯一的区别是：**我们把启动的网格大小死死卡在 GPU SM 总数的四分之一（在 128 个 SM 的 RTX 4090 上，也就是仅仅 32 个线程块）**，然后让这 32 个常驻的线程块用 `tl.range` 循环跨步领取那 9,600 个列块——也就是系统圈子里常说的持久化内核（Persistent Kernel）：

```python
# check-docs: skip: excerpt from toy_host_gather.py, a toy experiment outside the repository
@triton.jit
def bounded_rows(source, rows, output, count, width: tl.constexpr, block: tl.constexpr):
    # 同样的循环体，但 grid 固定；program 以 num_programs 为步长跨步领取 tile。
    tiles = tl.cdiv(width, block)
    for tile in tl.range(tl.program_id(0), count * tiles, tl.num_programs(0)):
        row = tile // tiles
        column = tile % tiles * block + tl.arange(0, block)
        index = tl.load(rows + row).to(tl.int64)
        value = tl.load(source + index * width + column, column < width, 0)
        tl.store(output + row * width + column, value, column < width)
```

把这两个只差了一行 `for` 循环的 kernel 放进同一张 RTX 4090 里跑一遍，终端上跳出来的实测数字会让你盯上半天：

```text
SM 数=128  naive grid=9600  bounded grid=32
forward = 4 组 4096^3 matmul；传输量 37.1 MiB

kernel      gather    GB/s  forward  concurrent  overlap saved
naive       2.38ms    16.3   3.88ms      6.19ms         0.07ms
bounded     1.64ms    23.7   3.88ms      4.00ms         1.53ms
```

我们先盯着 `naive` 那一行看。

单独跑假前向要 `3.88 ms`，单独跨 PCIe 捞数据要 `2.38 ms`。当我们把它们分装进两条独立的 CUDA stream、满怀期待地让它们“并发”执行时，总耗时是 **`6.19 ms`**——跟把它们头尾相接排队串行跑（$3.88 + 2.38 = 6.26\text{ ms}$）几乎没有任何区别！两条流并排跑，一共只省下了可怜兮兮的 **`0.07 ms`**。

怎么会这样？体系结构教科书上不是天天夸 GPU 是“隐藏内存延迟的魔术师”吗？

魔术师是有附加条款的。GPU 擅长隐藏延迟，指的是**已经住进同一个 SM 房间里的不同 warp 之间互相打掩护**：当 Warp A 发出一道读取指令、停下来等数据时，SM 的 warp 调度器可以在一纳秒内切到已经在房间里待命的 Warp B 去算乘法。

但看看我们的 `naive_rows` 干了什么：它在旁路流上一口气向硬件发射了 **9,600 个线程块（CTA）**。GPU 的硬件块调度器（Block Scheduler）是个极其实诚、不懂看人眼色的旅馆管理员——只要看到队列里有 9,600 个块在排队，它就会立刻把整张 4090 上全部 128 个 SM 的驻留槽位（resident slots）塞得满满当当。

```text
Naive Grid (9,600 blocks launched on side stream):
  ┌────────────┬────────────┬────────────┬─────┬────────────┐
  │   SM 0     │   SM 1     │   SM 2     │ ... │  SM 127    │
  │ [Gather]   │ [Gather]   │ [Gather]   │     │ [Gather]   │ ◄── All 128 SMs occupied!
  │ (waiting   │ (waiting   │ (waiting   │     │ (waiting   │     Tensor Cores sit idle
  │  on PCIe)  │  on PCIe)  │  on PCIe)  │     │  on PCIe)  │     waiting on 23.7 GB/s bus.
  └────────────┴────────────┴────────────┴─────┴────────────┘
        ▲
        └── Main Stream MatMul: "Can I get an SM?" ──► Hardware: "Sorry, house is full."

Bounded Grid (32 blocks = SMs / 4):
  ┌────────────┬─────┬────────────╦────────────┬────────────┬─────┬────────────┐
  │   SM 0     │ ... │   SM 31    ║   SM 32    │   SM 33    │ ... │  SM 127    │
  │ [Gather]   │     │ [Gather]   ║ [ MatMul ] │ [ MatMul ] │     │ [ MatMul ] │
  │ (loops     │     │ (saturates ║ (Main stream forward runs at full speed    │
  │  tiles)    │     │  PCIe bus) ║  across the remaining 96 SMs!)             │
  └────────────┴─────┴────────────╩────────────┴────────────┴─────┴────────────┘
```

一旦这群 `naive` 线程块住进了全部 128 个 SM，灾难就开始了：它们每一个人都在等那根只有 23.7 GB/s 的细细的 PCIe 总线。在 GPU 的时钟尺度上，等一次 PCIe 就像等一封跨洋平邮。在这漫长的等待期里，SM 里的 Tensor Core 算力单元完全闲着没事干，可**房间的床位却被这群等信的房客占死了**！由于 CUDA 的硬件块调度器是非抢占式的（non-preemptive），它绝不会因为一个线程块在干等外部总线就把它踢出 SM。于是，主干流上真正急着用 Tensor Core 算前向的矩阵乘 kernel 站在门外，发现 128 个房间全挂着“客满”的牌子，连门都进不去。“双流并发”在硬件层面硬生生退化成了先搬完数据、再算前向的排队串行。

打个更接地气的比方：厨房外头有一根细细的自来水管（PCIe），厨房里头有 128 个灶台（SM）。你为了接水快一点，派了 9,600 个帮工涌进厨房排队接水。水管的出水量一滴都没变多，有效带宽反而从 **23.7 GB/s 跌到了 16.3 GB/s**（`2.38 ms`）；更要命的是，这帮手里端着空盆的帮工把 128 个灶台前的地方站得水泄不通，真正要颠勺炒菜的大厨（模型前向）连厨房门都挤不进去。

再看 `bounded_rows` 那一行。

我们只派了 **32 个线程块**进场。首先，一个有趣的副产品出现了：哪怕只单独测 gather，32 个块也比 9,600 个块更快（`1.64 ms` 对 `2.38 ms`，带宽冲上 `23.7 GB/s`）——要喂饱一根 PCIe 4.0 总线，32 个块轮流发起读请求早就绰绰有余了。其次，当它和 `3.88 ms` 的前向一起跑时，总耗时只有 **`4.00 ms`**！也就是说，`1.64 ms` 的跨海搬运里有 **`1.53 ms`** 被彻彻底底地藏进了前向计算的阴影里，端到端只多付出了 `0.12 ms` 的代价。

原因就在上面的图里：32 个线程块最多只占走 32 个 SM 的一部分床位去等水管，剩下整整 **96 个 SM** 敞开大门留给主干流的矩阵乘。一边要的是 PCIe 总线带宽，另一边要的是 SM 里的 Tensor Core 算力——**只有当两条流索取的硬件资源互补、且谁也不把公共跑道堵死时，重叠才会真正发生。**

这可不是玩具脚本里才有的巧合。当初我们在 pLLM 的真实路径上（0.5B 模型，128 并发，查同一张宽表）抓 profile 时，看到了分毫不差的同一幕：不限网格大小的 UVA gather 一口气发射了 **37,984 个 CTA**，把 GPU 塞到 100% 占用率，结果 `1,753.7 µs` 的读表耗时里只有可怜的 `26.2 µs` 与主干前向发生了重叠；而一旦换成限制为 32 个 CTA 的有界持久化读法，`1,634.6 µs` 的读表时间里足足有 **99.14%** 都与主干流上的模型 kernel 严丝合缝地叠在了一起，每步的主机-设备空隙从 `1.75 ms` 直接砸到了 `0.24 ms`。

---

## 放进 pLLM：不改作者一个字的隐形手术

原理跑通了，接下来是工匠活：怎么把这个有界持久化 kernel 装进 pLLM，同时让写算法的研究者完全感知不到它的存在？

回头看文章开头，作者在 `lookup` 方法里写的只是最自然不过的 PyTorch 语法：`table[keys]`（或者 `torch.index_select`、`F.embedding`）。我们绝对不能要求作者为了把表放到主机内存，就去手动调用某个专门的主机 gather 函数——存储位置是系统配置，不应该污染算法的数学表达。

我们在底层搭了一条三级流水线：一个生产级算子、一层隐形拦截网、外加一次编译期预取调度。

### 第一级：支持动态视图的 `pllm::host_rows`

生产环境的 kernel 住在 `pllm/kernels/page_copy.py` 里。它和我们刚才写的玩具 `bounded_rows` 几乎是一个模子刻出来的，只为了应对真实的 PyTorch 世界多加了两处小细节：

```python
# check-docs: skip: excerpt from pllm/kernels/page_copy.py
@triton.jit
def _host_rows(
    source: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    height: int,
    width: tl.constexpr,
    stride0: int,
    stride1: int,
    count: int,
    block: tl.constexpr,
) -> None:
    # Row counts and strides vary per step and view; as runtime arguments they reuse one compiled kernel.
    tiles = tl.cdiv(width, block)
    for tile in tl.range(tl.program_id(0), count * tiles, tl.num_programs(0)):
        row = tile // tiles
        column = tile % tiles * block + tl.arange(0, block)
        index = tl.load(rows + row).to(tl.int64)
        index = tl.where(index < 0, index + height, index)
        value = tl.load(source + index * stride0 + column * stride1, column < width, 0)
        tl.store(output + row * width + column, value, column < width)
```

注意看多出来的那几行参数：
1. **负数下标回绕**：在 Python 和 PyTorch 里，用户随时可能写出 `table[-1]` 或产生负偏移。那行 `tl.where(index < 0, index + height, index)` 在 GPU 寄存器里顺手把负索引加上总行数 `height` 绕回正数区间。
2. **运行时步长（`stride0`, `stride1`）与行数（`count`）**：用户完全可能先对大表切一刀视图（比如隔列采样 `table[:, ::2][keys]`），这时候张量的内存不再连续，步长变了；而且不同步里的行数也可能变。通过把 `height`、`stride0`、`stride1`、`count` 作为普通运行时整数传入（只留 `width` 和 `block` 作为编译期常量 `tl.constexpr`），所有不同切片视图和行数的读取都能复用同一个编译好的 Triton 机器码，不会触发恼人的重编译。

接着，我们把它包进一个标准的 PyTorch 自定义算子 `pllm::host_rows` 里，“四分之一 SM”的紧箍咒就念在这里：

```python
# check-docs: skip: excerpt from pllm/kernels/page_copy.py
@torch.library.custom_op("pllm::host_rows", mutates_args=())
def host_rows(source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Read GPU-indexed pinned rows with a bounded grid, leaving SMs for the forward."""
    output = source.new_empty((*rows.shape, source.shape[1]))
    if output.numel():
        block = 2048
        tiles = rows.numel() * triton.cdiv(source.shape[1], block)
        sms = int(torch.cuda.get_device_properties(source.device).multi_processor_count)
        programs = min(tiles, max(1, sms // 4))
        # ……
    return output
```

紧接着这十几行下方，我们还给 `host_rows` 注册了两个关键的伴侣函数：一个是 `@host_rows.register_fake`（在 CPU 上做 `pllm.explain` 静态规划时，只返回输出形状 `(*rows.shape, source.shape[1])`，绝不去真碰锁页内存）；另一个是 `@host_rows.register_vmap`（当用户的代码被 `torch.vmap` 按请求矢量化包裹时，把映射维度上的多次查表自动拍扁折叠成对共享二维表的**单次** `host_rows` 调用）。有了这两件套，这个手写的 Triton 算子就能在 meta 追踪、`vmap` 批处理和 CUDA Graph 录制里畅行无阻。

### 第二级：用 `TorchFunctionMode` 实施偷梁换柱

算子准备好了，怎么让作者写的 `table[keys]` 自动跳进 `host_rows` 里？

PyTorch 提供了一个非常优雅的拦截机制，叫 `TorchFunctionMode`。它就像套在 Python 解释器外面的一个透明滤镜：只要在 `with HostGather(tables):` 的上下文管理器内部执行代码，每一次张量方法调用（哪怕是 `a[b]` 这样的语法糖，底层即 `torch.Tensor.__getitem__`）都会先经过我们的 `__torch_function__` 盘问。

来看看 `pllm/execution/torch_private.py` 里的拦截现场：

```python
# check-docs: skip: excerpt from pllm/execution/torch_private.py
source = args[0] if args else keywords.get("input", keywords.get("self"))
if any(source is table for table in self.tables):
    assert isinstance(source, torch.Tensor)
    if func in (torch.Tensor.__getitem__, torch.ops.aten.index.Tensor):
        indices = tree_leaves(args[1] if len(args) > 1 else keywords["indices"])
        if len(indices) == 1 and isinstance(indices[0], torch.Tensor):
            return host_rows(source, indices[0])
    if func in (torch.index_select, torch.Tensor.index_select, torch.ops.aten.index_select.default):
        indices = args[2] if len(args) > 2 else keywords["index"]
        assert isinstance(indices, torch.Tensor)
        return host_rows(source, indices)
```

逻辑清清楚楚：当被操作的张量 `source` 是我们登记在案的主机大表之一时：
* 如果调的是 `F.embedding` / `torch.embedding`（在这段代码上方几行），直接转交 `host_rows(source, indices)`；
* 如果调的是 `table[idx]`（单张量索引）或 `index_select`，同样当场截胡，换成 `host_rows(source, indices)`；
* 如果用户先做了一个不搬数据的视图变换（比如 `sub = table[:, ::2]`），后面几行还会聪明地把新生成的视图张量 `result` 也追加进 `self.tables` 名单里，确保下一步对 `sub[keys]` 的索引依然逃不出手掌心！

那如果用户不小心对一张主机大表写了 `table + 1.0` 或者 `h @ table` 呢？想都别想——把整张 594 MiB 甚至 3 GiB 的表每步跨 PCIe 全量拉过来做算术，会把整个推理引擎当场卡死。所以在规划阶段，还有一个配对的 `GatherTrace`（基于 `TorchDispatchMode`）守在门口：凡是对 `placement="host"` 的表做非按行读取的算子，或者忘了先 gather 就直接把表视图作为结果返回，规划器会当场抛出 `ValueError` 拒绝准入，同时顺手把每次合法 gather 产生的结果字节数累加起来，记入 `logical transfer bytes/row` 账本。

作者的代码连一个字母都不用改。把声明里的 `placement="device"` 改成 `placement="host"`，底层的执行就悄无声息地从普通显存寻址切换成了 32 线程块的有界 PCIe 零拷贝拉取。

### 第三级：提前发射与双流接力

只选对 kernel 还不够，还得在对的时刻扣动扳机。

```text
Step Timeline (with HostGather + Prefetch):

  Anchor:        [0: request.prepare]                       [1: model.execute]
                 ────────────────────                       ──────────────────
  Side Stream:   ┌───────────────────────────────────────┐
  (32 SMs max)   │ pllm::host_rows(table, ngram_keys)    │──► record(event)
                 └───────────────────────────────────────┘         │
                        ║  (Hidden behind forward)                 ▼
  Main Stream:   ┌──────╨────────┬────────┬─── ... ───┬────────┬──────────────┐
  (96+ SMs)      │ Embed + L0    │   L1   │           │ wait() │ L2 + Gate(e) │
                 └───────────────┴────────┴─── ... ───┴────────┴──────────────┘
```

正如这幅时间线画的那样，因为 `lookup` 方法的参数列表里只有 `pllm.input_tokens` 和只读的 `pllm.parameter`，没有任何对模型中间层 `hidden` 的依赖，pLLM 的调度器（详见下一篇[《自动放置与重叠》](11-automatic-placement-overlap.md)）一眼就能认出它是一个可以提前预取的任务，自动把它排到一步最开头的 `request.prepare` 锚点、放到 `side` 旁路流上发射，并一起录进 CUDA Graph。在 `pllm.explain` 的输出里，你会看到一行干净利落的确认：

```text
lookup  compute  request.prepare / side / captured
```

当旁路流上的 32 个线程块正顺着 PCIe 蚂蚁搬家时，主干流上的模型已经马不停蹄地从词嵌入跑过了第 0 层、第 1 层；直到真正消费查表结果的那一层（比如 `EngramLite` 的第 2 层残差挂载点，或者采样器前的 `add_logits`），主干流才会等待旁路流记录下的那个 CUDA Event，接过刚刚落地的张量继续往下走。

这也是为什么在使用主机表时，有一条铁律必须遵守：**查表必须写成一个不读 `hidden` 的独立方法，再由效果方法通过 `pllm.result("lookup")` 去拿结果。** 如果你图省事，在 `@pllm.custom_residual` 方法体内部直接去读主机表，那个方法可是嵌在主干流某两层 Transformer 之间同步执行的！在那里发起 PCIe 读取，主干流后面的所有层就只能原地罚站干等 1.6 ms，神仙也救不回来。

顺便提一句，把查表拆成“只读 token + 只读表”的独立方法，连 `placement="device"`（表在显存）都会跟着沾光：当 pLLM 的 `_lookup_math` 识别出这种纯查表任务时，会在绑定引擎阶段自动调用 Inductor 编译器，把里面的哈希取行（gather）、类型转换（`.float()`）、条件掩码（`torch.where`）和常数缩放融合成**单次显存表读取 kernel**，省掉了把 38.9 MB 的宽向量写回 HBM 显存再读出来做逐元素乘法的冤枉路（如果你需要严格逐比特对齐未融合的参考算术，也可以传 `allow_numerical_transforms=False` 把这项融合关掉）。

---

## 7B 上的真实账单，以及为什么放哪里必须由你签字

把这三件套全部合拢，我们在 RTX 4090 上用 `Qwen2.5-7B-Instruct` 跑一遍示例 11 的 594 MiB `NGramPrior` 大表（128 并发），看看最终的成绩单长什么样：

| 配置（7B，128 并发，594 MiB 表） | 吞吐（tok/s） | 相对原生 vLLM | 单步耗时 | pLLM 显存预留峰值（`explain`） |
| :--- | ---: | ---: | ---: | ---: |
| 原生 vLLM（无查表） | 5,877 | 100% | 21.8 ms | — |
| 表在显存（`placement="device"`） | 5,742 | **98%** | 22.3 ms | 1,540 MiB |
| 表在主机（`placement="host"`） | 5,615 | **96%** | 22.8 ms | **946 MiB** |

算一算这笔买卖有多划算：每一步 decode，我们实打实地跨过 PCIe 4.0 总线搬运了 **38.9 MB** 的数据（物理传输耗时约 `1.6 ms`）；但由于 32 线程块的有界拉取与 21.8 ms 的前向计算在双流上高度重叠，表放在主机内存比放在显存每步**只慢了约 `0.5 ms`**，整体吞吐依然咬在原生 vLLM 的 **96%**！与此同时，pLLM 在 GPU 上预留的显存峰值从 `1,540 MiB` 直接降到了 `946 MiB`，分毫不差地省出了整张表的 **594 MiB** 显存。在我们的自动化测试里，哪怕把 GPU 显存预算死死卡在 `1 GiB`，也能稳稳当当地挂上一张 **`3 GiB`** 的主机大表跑完全程。

看到这么漂亮的数字，你可能会问最后一个问题：“既然放主机内存这么香，为什么当我的显存预算不够装下大表时，pLLM 不干脆帮我自动把表溢出（offload）到主机内存，非要报错逼我亲手写 `placement="host"`？”

在系统设计里，这种“出了问题不报错、悄悄退回慢路径”的做法叫静默回退（Silent Fallback），而 pLLM 的核心哲学之一就是**拒绝静默回退（Fail-Closed）**。

原因很简单：跨 PCIe 搬数据的账单，在 7B 上好看，不代表在所有地方都好看。

我们刚才算过，128 行词表宽度的传输物理上要花 `1.6 ms`。在一步 `21.8 ms` 的 7B 模型上，前向这把大伞足够宽，能把 `1.6 ms` 遮得严严实实；可如果你跑的是一步只有 `3.0 ms` 的 0.5B 小模型，或者你的并发度冲到了 512 行（每步传输翻四倍到 `155 MB`，耗时超过 `6.5 ms`），PCIe 传输时间就会撑破前向的伞面直接暴露出来。反过来，如果你用的是 `EngramLite` 那种窄表（每行只有 32 个数，整张表才 4 MiB），每步跨 PCIe 只搬几千个字节，在 0.5B 上的实测里，`host`（相对第 2 层恒等残差基线的 `95.78%` / `94.94%`，对应 batch 1 / batch 8）和 `device`（`96.29%` / `95.86%`）的差距根本落在了测量噪声里，有界无界都无所谓。

除此之外，这条零拷贝通道还有几个明确的物理边界：
* **只服务只读的预训练表（`Pretrained`）**：在线训练的可变状态或模型模块需要高频读写，不能放上主机；
* **没有 GPU 侧热行缓存**：做推荐系统（如 DLRM）的同学都知道，大 embedding 表放主机时，工业界常在显存里配一层 LRU 热行缓存，因为自然语言的 n-gram 分布同样遵循齐夫定律（Zipf's law），少数高频词组会被反复命中。pLLM 目前没有做这一层缓存——每一步、每一行都老老实实走一次 PCIe。如果你的访问局部性极强，这无疑浪费了总线带宽；但代价是省下了一整套 GPU 端并发哈希表与缓存一致性维护的复杂度；
* **1/4 SM 是一个经验常数**：`sms // 4` 并没有根据 PCIe 5.0 或不同 GPU 架构做动态自适应。

既然“把表放到慢速内存里”是一笔代价极度依赖模型大小、行宽与并发数的昂贵交易，框架就绝不该在背后替你偷偷签字。

在 `examples/engram_lite.py` 的文档注释里，我们把这条规矩写得斩钉截铁：**“placement is the author's choice; pLLM never moves the table to satisfy a budget.”** 在启动 GPU 之前，你可以先在 CPU 上花两秒钟跑一次 `pllm.explain(workload, config)`，它会像老练的会计一样把账单摊在你面前（下面是示例 11 在 0.5B、`max_num_seqs=8` 下的一行）：

```text
table …… placement: host; host bytes: 622362638; logical transfer bytes/row: 303888; decode bytes/step: 2431104
```

你只需要拿 `并发数 × logical transfer bytes/row / 22 GB/s` 心算一下：如果算出来的毫秒数远远短于你模型的一步前向，那就放心大胆地写下 `placement="host"`；如果算出来比前向还长，你立刻就知道该去升级显存，而不是等到半夜线上服务吞吐暴跌时再去抓耳挠腮地查 profiler。

---

回顾这一路，最让我着迷的依然是 `host_rows` 里那行把网格砍到 `sms // 4` 的代码。

我们平时写 GPU 程序，总是被“占用率（Occupancy）越高越好、线程块开得越多越好”的教条推着走。但硬件从来不听教条，硬件只认瓶颈。当一群线程块的瓶颈根本不在计算单元、而是在等一根细细的外部水管时，把所有房间塞满不仅不能让水流变快，反而会把真正需要炉火的大厨堵在门外。

学会克制——只派刚好够用的 32 个线程块去守着总线，把剩下的天地留给计算——我们才真正实现了“鱼与熊掌兼得”。而在下一篇[《自动放置与重叠》](11-automatic-placement-overlap.md)里，我们会把视野从这一张表拉升到整个推理引擎的时间线，去看看当你的程序里有好几个任务都想跟模型前向重叠时，这套“资源互补才有真并发”的物理铁律还会给我们带来什么样更深刻的惊喜与教训。
