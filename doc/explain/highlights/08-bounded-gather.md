# 为什么 32 个线程块比 9,600 个更快？一次把 GPU 前向饿死的跨 PCIe 查表

有一类推理期算法长得越来越像数据库查询：模型一边往前生成 token，一边拿当前 token 和前几个历史 token 算一个哈希，去一张巨大的外部静态表里捞出几行向量，经过门控加进某层的残差流，或者直接作为先验偏置加到词表 logits 上。DeepSeek 的 Engram 记忆层是这个路数，经典的 kNN-LM 与 n-gram 先验也是这个路数。

单看算法本身，代码干净得只有一行。在 `examples/engram_lite.py` 里，从表里取两行二元组与三元组记忆向量的逻辑是这样的：

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

真正让人头疼的不是这一行索引操作，而是 `table` 的体积。在我们的查表记忆教程（示例 11 `NGramPrior`）里，一张与 `Qwen2.5-7B` 词表同宽的先验表，形状是 `2048 × 152064`（BF16），光静静地躺在那里就要占掉 **594 MiB**；而在真实的 Engram 场景下，记忆表还要大得多。在线推理服务的显存是一场零和博弈——往显存里塞进 1 GiB 的静态查找表，vLLM 的 KV cache 池就得吐出 1 GiB，能同时容纳的并发请求数当场跟着缩水。

既然主板上插着几百 GiB 的 CPU 内存，最自然的想法就是把这张只读大表放到主机内存里，每一步解码需要哪几行，临时从 PCIe 总线上拉过来。更妙的是，查表只依赖上一步已经生成的 token ID，完全不需要等待当前步的 Transformer 隐藏状态（`hidden`）。只要在每一步最开头把查表任务发到一条旁路 CUDA stream 上，让它在后台慢慢跨 PCIe 搬数据，前台的主流同时跑模型前向，两边并行重叠，这笔传输耗时不就白捡了吗？

想法无懈可击。但如果你真的按标准写法写出一个 GPU gather 内核放到旁路流上，打开性能分析器就会看到诡异的一幕：后台的查表内核一启动，直接把前台的模型前向矩阵乘给“饿死”了，两条流并发跑跟串行排队跑几乎一样慢。为了把传输耗时真正藏进前向阴影里，我们不得不做了一件反直觉的事——**把读取数据的 GPU 线程块（CTA）数量从几千个硬生生砍到 32 个（即 GPU SM 总数的四分之一）。**

为什么派出去干活的线程块越少，整个解码步反而跑得越快？在看代码之前，我们先拿信封背面算一笔硬件带宽账。

---

## 一、 594 MiB 的大表与 1.64 ms 的跨海大桥

假设我们在单张 NVIDIA RTX 4090（24 GiB GDDR6X 显存，128 个 SM，插在 PCIe 4.0 x16 插槽上）上运行 `Qwen/Qwen2.5-7B-Instruct`（BF16），并发度设为 128 行（8 个 prompt × `n=16`）。

每一步 decode，128 个并发序列要各自从那张 `2048 × 152064` 的 BF16 大表里查一行。每一行有 152,064 个双字节浮点数，即 `304,128` 字节（约 297 KiB）；128 行加起来，每一步需要跨过 PCIe 搬运的数据量是：

$$128 \times 304,128\text{ Bytes} \approx 38.9\text{ MB}$$

在这台机器的 PCIe 4.0 x16 插槽上，后文那个有界 gather 内核直接读取主机锁页内存的实测吞吐大约是 **23.7 GB/s**。做一道除法就能算出每步跨总线搬运要花的时间：

$$\frac{38.9\text{ MB}}{23.7\text{ GB/s}} \approx 1.64\text{ ms}$$

1.64 毫秒有多长？在同样的 RTX 4090、128 并发配置下（vLLM 0.32.0，开启 CUDA Graph 与 FlashInfer 采样器），`Qwen2.5-7B-Instruct` 跑完一整步 28 层的 decode 大约需要 **21.8 ms**（吞吐约 5,877 tok/s）；而如果是轻量级的 `Qwen2.5-0.5B-Instruct`，跑完一步 decode 总共只需要 **3.0 ms**（吞吐 42,355 tok/s）。

账面非常清楚：如果让这 1.64 ms 串行横在关键路径上，7B 模型每步要白白慢上近 8%，0.5B 小模型的吞吐更是会直接掉去三分之一以上。反过来，因为查表只依赖输入 token ID，在步内时间线的第一个锚点（`request.prepare`）就可以发往旁路流；只要这 1.64 ms 能和主干流上 21.8 ms（乃至 0.5B 的 3.0 ms）的模型前向同时推进，它就应该能被前向计算严丝合缝地遮住。

那么，怎么把这 128 行从 CPU 内存搬上 GPU，还不打扰主干流上的矩阵乘？

---

## 二、 为什么硬件 DMA 拷贝引擎指望不上？

听到“跨 PCIe 搬数据还要与计算重叠”，系统程序员的第一反应一定是 **DMA（直接内存访问）**。

现代 GPU 芯片上配有独立的硬件拷贝引擎（Copy Engines）。当你调用 `cudaMemcpyAsync`、`cudaMemcpy2DAsync`、`cudaMemcpyBatchAsync`，或者像 vLLM 在 CPU 与 GPU 之间换入换出 KV 缓存块那样调用 `swap_blocks` 时，干活的是专职的拷贝引擎。拷贝引擎最大的优点是**完全不占用 SM（流多处理器）**：SM 在主干流上全速算矩阵乘，拷贝引擎在旁路流上走 PCIe 搬字节，两边在硬件上互不干扰。

像 ZeRO-Infinity、FlexGen 或是 vLLM 自己的权重 CPU offload，之所以能把拷贝引擎用得风生水起，是因为它们搬运的是第 $l$ 层的模型权重或已知的 KV 块——**下一拍要搬哪块内存，CPU 在发令前就已经一清二楚了。**

但我们的查表场景撞上了一个死结：**要拷哪几行，CPU 根本不知道。**

每一步的当前 token ID 是上一步采样器刚刚在 GPU 上抽出来的，历史 token 是 GPU 维护的，哈希桶号 `ngram_keys(ids, buckets)` 更是在 GPU 显存里当场算出来的。在生成步的循环里，那 128 个行号自始至终都躺在 GPU 显存里。

而所有的 `cudaMemcpy` 变体都有一条铁律：**源内存地址（或源地址指针数组）必须在 CPU 主机端作为参数传给驱动。** 如果你非要用硬件拷贝引擎，唯一的办法就是先把 GPU 算出来的 128 个行号拷回 CPU，让 CPU 算出 128 个主机内存偏移地址，再由 CPU 组装 DMA 命令发回 GPU。

在步长只有几毫秒的推理引擎里，把数据从设备拷回主机意味着一次阻塞式的**主机同步（Host Synchronization）**。这不仅会当场掐断 CPU 提前发射内核的异步流水线，还会直接违反 CUDA Graph 的捕获规则（详见[《What Color is Your Tensor?》](02-tensor-colors.md)——更何况录进 CUDA Graph 里的 `memcpy` 节点，其源地址参数在录制那一刻就被焊死了，回放时根本无法按步修改）。

拷贝引擎这条路断了，就只剩下唯一的一条路：**不走 DMA 拷贝，让 GPU 的计算内核（Kernel）自己在显存里读出行号，直接伸手去主机内存里把数据拿过来。**

在 CUDA 里，这套机制叫作**锁页内存（Pinned Memory）上的统一虚拟地址空间（UVA）零拷贝读取**：
1. 普通的操作系统内存（pageable memory）随时可能被换出到磁盘，GPU 硬件无法直接寻址；但只要调用 `.pin_memory()` 将大表锁定在物理内存中，CUDA 驱动就会将其映射进统一虚拟地址空间。
2. 为了让 PyTorch 与 Triton 算子愿意接受这块主机内存，我们可以借用 vLLM 提供的工具函数 `get_accelerator_view_from_cpu_tensor(table)`。它不会拷贝哪怕一个字节，而是直接创建一个 `is_cuda == True`、但底层 `.data_ptr()` 与原主机张量完全相同的设备视图外壳（alias）。
3. 当 GPU 线程在内核里拿到 GPU 显存中的行号 `index`，对这个视图地址执行普通的 `tl.load` 指令时，SM 的加载存储单元发现目标虚拟地址落在主机端，便会自动发起跨 PCIe 的读事务，把字节拉进 GPU 寄存器，再由 `tl.store` 写进 GDDR6X 显存的输出缓冲区。

行号留在 GPU 上，地址由 GPU 自己算，全程零次主机同步，还能原封不动地录进 CUDA Graph。

看起来万事大吉。直到我们写出了第一个内核，并把它放进了双流并发测试里。

---

## 三、 百行复现实验：9,600 个线程块是怎么把前向饿死的？

为了把问题剥离得干干净净，我们写了一个不依赖整个推理引擎、只有一百多行的独立复现脚本 `tools/toy_host_gather.py`，在任何一台有 CUDA GPU 的机器上用 `python tools/toy_host_gather.py` 就能自己跑一遍。

实验的设置完全复刻了真实场景的尺寸：
* 在锁页主机内存里放一张 `2048 × 151936` 的 BF16 大表（约 594 MiB，对应 0.5B 词表宽度），用 `get_accelerator_view_from_cpu_tensor` 取其 UVA 视图，并断言 `view.data_ptr() == table.data_ptr()`；
* 在 GPU 上随机生成 128 个行号 `idx`（每次 gather 需跨 PCIe 读取 `37.1 MiB`）；
* 在主干流（`main_stream`）上，用 4 组 $4096 \times 4096$ 的 BF16 矩阵乘法串成一个模拟的“模型前向（forward）”——在 RTX 4090 上单独运行它恰好耗时 **3.88 ms**，并且会用满几乎所有的 SM 算力；
* 在旁路流（`side`）上同时启动主机查表内核，用 CUDA Event 测量两边单独跑的时间、并发跑的总时间，以及并发重叠真正省下的时间（`overlap saved = gather + forward - concurrent`）。

按照任何一本 GPU 编程教材的写法，从二维表里按行读取数据的 `naive_rows` 内核会写成这样：

```python
# check-docs: skip: excerpt from tools/toy_host_gather.py
@triton.jit
def naive_rows(
    source: torch.Tensor, rows: torch.Tensor, output: torch.Tensor, width: tl.constexpr, block: tl.constexpr
) -> None:
    # 每个 program 只负责一个 (行, 列 tile)；grid 跟数据量成正比。
    tiles = tl.cdiv(width, block)
    row = tl.program_id(0) // tiles
    column = tl.program_id(0) % tiles * block + tl.arange(0, block)
    index = tl.load(rows + row).to(tl.int64)  # 行号在 GPU 上，kernel 内部自己取
    value = tl.load(source + index * width + column, column < width, 0)
    tl.store(output + row * width + column, value, column < width)
```

为了保证内存合并访问，每个线程块（Triton 里的一个 `program`）负责读取某一行里的一个连续列分块（`block = 2048` 个 BF16 元素，即 4 KiB）。一行宽 151,936，需要 $\lceil 151936 / 2048 \rceil = 75$ 个列分块；一共要取 128 行，所以启动网格（grid）的大小设为 `128 * 75 = 9,600` 个线程块。数据有多少块，grid 就开多大——再标准不过了。

现在，我们在它旁边写第二个版本 `bounded_rows`。里面的算术与读写指令一行不改，唯一的区别是：**不管外面要读多少行，启动网格的大小死死固定为 GPU SM 总数的四分之一**（在 128 个 SM 的 RTX 4090 上，只启动 **32 个** 线程块），让这 32 个常驻的线程块用 `tl.range` 循环跨步领取那 9,600 个列分块——即持久化内核（Persistent Kernel）：

```python
# check-docs: skip: excerpt from tools/toy_host_gather.py
@triton.jit
def bounded_rows(
    source: torch.Tensor,
    rows: torch.Tensor,
    output: torch.Tensor,
    count: int,
    width: tl.constexpr,
    block: tl.constexpr,
) -> None:
    # 同样的循环体，但 grid 固定；program 以 num_programs 为步长跨步领取 tile。
    tiles = tl.cdiv(width, block)
    for tile in tl.range(tl.program_id(0), count * tiles, tl.num_programs(0)):
        row = tile // tiles
        column = tile % tiles * block + tl.arange(0, block)
        index = tl.load(rows + row).to(tl.int64)
        value = tl.load(source + index * width + column, column < width, 0)
        tl.store(output + row * width + column, value, column < width)
```

在 RTX 4090（PCIe 4.0 x16，PyTorch 2.13.0+cu129，15 轮中位数）上运行这个脚本，先与 CPU 参考结果做 `int16` 逐比特校验确认两者结果 100% 一致，随后终端打印出了这组数字：

```text
SM 数=128  naive grid=9600  bounded grid=32
forward = 4 组 4096^3 matmul；传输量 37.1 MiB

kernel      gather    GB/s  forward  concurrent  overlap saved
naive       2.38ms    16.3   3.88ms      6.19ms         0.07ms
bounded     1.64ms    23.7   3.88ms      4.00ms         1.53ms
```

先看第一行 `naive`。
单独跑矩阵乘前向要 `3.88 ms`，单独跑 `naive` 查表要 `2.38 ms`（有效带宽只有 16.3 GB/s）。当我们把它们放进两条不同的 CUDA stream 同时发射时，总耗时高达 **`6.19 ms`**——跟把它们头尾相接串行跑（$3.88 + 2.38 = 6.26\text{ ms}$）几乎没有任何区别！两条流并发，一共只省下了 **`0.07 ms`**。

再看第二行 `bounded`。
仅仅把线程块数量从 9,600 个砍到 32 个，不仅单独跑 gather 的时间从 `2.38 ms` 降到了 **`1.64 ms`**（吞吐升到 `23.7 GB/s`），而且当它与 `3.88 ms` 的矩阵乘并发执行时，总耗时直接落到了 **`4.00 ms`**！也就是说，`1.64 ms` 的跨 PCIe 传输里有 **`1.53 ms`（超过 93%）** 被彻彻底底地藏进了矩阵乘法的阴影里，端到端只多付出了 `0.12 ms`。

同样的现象在真实引擎的剖析轨迹里甚至更加夸张：在 Qwen2.5-0.5B（128 并发，示例 11 宽表，至少 5 轮中位数，一版较早的实现）的剖析中，不限制网格大小的 UVA gather 一口气发射了 **37,984 个 CTA**，把 GPU 推到 100% 占用率，结果 `1,753.7 µs` 的读表耗时里只有可怜的 `26.2 µs` 与主干前向发生了重叠；换成 32 个 CTA 的有界持久化内核后，`1,634.6 µs` 的读表时间里足足有 **99.14%** 与主干流上的模型内核重叠在了一起（与在前向末尾才发射无界读的写法相比，每步的主机-设备空隙从 `1.75 ms` 降到了 `0.24 ms`）。

为什么多派了三百倍的线程块，反而把并发彻底搞砸了？

---

## 四、 掀开硬件调度器：SM 驻留槽位与慢速总线的错配

要理解那消失的 1.5 毫秒，我们得区分 GPU 上的两层调度器：**SM 内部的 Warp 调度器**，以及 **GPU 全局的硬件块调度器（Block / CTA Scheduler）**。

平时我们说“GPU 擅长用高并发隐藏内存延迟”，指的是在**同一个 SM 内部**：当已经驻留进该 SM 的 Warp A 发出一条显存读取指令并停下来等待数据时，SM 的 Warp 调度器可以在单个周期内切换到同一个 SM 里的 Warp B 去使用 Tensor Core 算矩阵乘。但请注意前提——**Warp B 所在的线程块，必须已经分到了这个 SM 的房间（驻留槽位，Resident Slot）！**

看看当旁路流发射 `naive_rows`（9,600 个线程块）时，全局硬件块调度器干了什么：

```text
1. Naive Grid (9,600 blocks launched on side stream):
   ┌────────────┬────────────┬────────────┬─────┬────────────┐
   │   SM 0     │   SM 1     │   SM 2     │ ... │  SM 127    │
   │ [Gather]   │ [Gather]   │ [Gather]   │     │ [Gather]   │ ◄── All 128 SMs occupied!
   │ (waiting   │ (waiting   │ (waiting   │     │ (waiting   │     Tensor Cores sit idle
   │  on PCIe)  │  on PCIe)  │  on PCIe)  │     │  on PCIe)  │     waiting on 23.7 GB/s bus.
   └────────────┴────────────┴────────────┴─────┴────────────┘
         ▲
         └── Main Stream MatMul: "Can I get an SM?" ──► Block Scheduler: "House is full."

2. Bounded Grid (32 blocks = SMs / 4):
   ┌────────────┬─────┬────────────╦────────────┬────────────┬─────┬────────────┐
   │   SM 0     │ ... │   SM 31    ║   SM 32    │   SM 33    │ ... │  SM 127    │
   │ [Gather]   │     │ [Gather]   ║ [ MatMul ] │ [ MatMul ] │     │ [ MatMul ] │
   │ (loops     │     │ (saturates ║ (Main stream forward runs at full speed    │
   │  tiles)    │     │  PCIe bus) ║  across the remaining 96 SMs!)             │
   └────────────┴─────┴────────────╩────────────┴────────────┴─────┴────────────┘
```

CUDA 的硬件块调度器是非抢占式的（non-preemptive）。当旁路流一下子把 9,600 个线程块推入硬件队列时，块调度器看到有海量任务待命，便尽职尽责地把整张 RTX 4090 上全部 **128 个 SM** 的驻留槽位塞得满满当当。

一旦这些 `naive` 线程块住进了全部 128 个 SM，瓶颈立刻显现：它们执行的不是计算，而是跨过那根细细的 PCIe 总线去拉数据（零拷贝读取实测只有约 `23.7 GB/s`；相比之下，4090 板载 GDDR6X 显存的带宽是 `1,008 GB/s`，差了四十多倍）。在漫长的 PCIe 读事务等待期间，这 128 个 SM 里的算术单元和 Tensor Core 完全闲置；但由于线程块还没拿到数据执行完 `tl.store` 退出，它们死死占着 SM 的硬件槽位不退房。每当有一个块好不容易读完 4 KiB 退房，排队等候的剩下九千多个 `naive` 兄弟立刻补位抢占。结果就是，主干流上真正需要 Tensor Core 算前向的矩阵乘内核被堵在门外，等旁路流把 9,600 个块几乎排空了才挤进 SM——两条流的并发在物理硬件上退化成了串行（而且这种写法单独跑时，有效吞吐也只有 `16.3 GB/s`）。

这就好比厨房外头只有一根细细的自来水管（PCIe），厨房里头有 128 个猛火灶台（SM）。为了接水快一点，你派了 9,600 个帮工涌进厨房排队接水。水管的出水量不会因为人多而变大，但这群端着空盆等水滴的帮工把 128 个灶台前的位置站得水泄不通，真正要颠勺炒菜的大厨（模型前向）连厨房门都进不去。

而当我们把网格硬限制在 `sms // 4 = 32` 个持久化线程块时，秩序瞬间恢复了：
1. **32 个块就够把零拷贝读取推到实测的 `23.7 GB/s`**：每个块在循环里连续发起合并的 4 KiB 读请求；
2. **留出四分之三的江山给计算**：32 个块最多只触碰四分之一的 SM 资源，剩下至少 **96 个 SM** 完完全全空出来留给主干流上的模型前向。一边吃的是 PCIe I/O 带宽，另一边吃的是 SM 算力与板载 GDDR6X 带宽，两种互补的硬件资源这才真正并行了起来。

---

## 五、 不改作者一个字的隐形替换与 7B 真实账单

搞清楚了硬件机制，在框架里落地这套方案只需要解决一个工程问题：怎样让算法作者继续写最自然的 `table[keys]`（或 `F.embedding`、`index_select`），而在底层自动把它变成运行在旁路流上的 32 线程块有界拉取？

在 pLLM 中，这由三个薄薄的组件配合完成。

首先是位于 `pllm/kernels/page_copy.py` 的生产级算子 `_host_rows` 与 `pllm::host_rows`。相比刚才的玩具代码，它只多处理了两件真实世界里的小事——把 Python 允许的负数行下标绕回正数（`index + height`），以及把视图步长（`stride0`, `stride1`）和行数（`count`）作为运行时整数传入，使得对表做任意切片视图或不同批次大小都能复用同一个编译好的 Triton 内核：

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


@torch.library.custom_op("pllm::host_rows", mutates_args=())
def host_rows(source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Read GPU-indexed pinned rows with a bounded grid, leaving SMs for the forward."""
    output = source.new_empty((*rows.shape, source.shape[1]))
    if output.numel():
        block = 2048
        tiles = rows.numel() * triton.cdiv(source.shape[1], block)
        sms = int(torch.cuda.get_device_properties(source.device).multi_processor_count)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        programs = min(tiles, max(1, sms // 4))
        # ……
    return output
```

紧随其后，我们为 `host_rows` 注册了 `register_fake`（在 CPU 上做静态规划时不触碰锁页存储，只推导形状）和 `register_vmap`（将按请求 `vmap` 展开的查表自动折叠为对共享二维表的单次 `host_rows` 调用）。

接着，在执行作者的查表方法时，`pllm/execution/torch_private.py` 里的 `HostGather`（一个 `TorchFunctionMode` 上下文）会在解释器层拦截对主机表的所有索引操作——无论是 `table[keys]`、`index_select` 还是 `F.embedding`——无缝替换成 `host_rows(source, indices)`；如果作者先对表做了切片视图，该视图也会被自动加入跟踪列表。而在规划期，配对的 `GatherTrace` 则负责守门：如果有人不小心对一张几百兆的主机大表写了 `table + 1` 或矩阵乘，企图每步把整张表拉过 PCIe，规划器会当场抛异常拦下，并精确统计出每次合法行读取的传输字节数。

最后一步是**预取时机**。在 pLLM 的步内 8 锚点时间线上（详见下一篇[《You Probably Don't Need a Compiler IR：用 40 行最长路自动调度 CUDA Stream》](09-stream-scheduling.md)），当规划器看到 `lookup` 方法只依赖 `pllm.input_tokens` 和不可变参数、不依赖任何模型层的 `hidden` 时，就会自动将它排到一步最开头的 `request.prepare` 锚点，放进 `side` 旁路流并录进 CUDA Graph：

```text
Step Timeline (8 Anchors, HostGather Prefetch on Side Stream):

  Anchor:        [0: request.prepare]                       [1: model.execute]
                 ────────────────────                       ──────────────────
  Side Stream:   ┌───────────────────────────────────────┐
  (32 SMs max)   │ pllm::host_rows(table, ngram_keys)    │──► record(event)
                 └───────────────────────────────────────┘         │
                        ║  (93%+ hidden behind forward)            ▼
  Main Stream:   ┌──────╨────────┬────────┬─── ... ───┬────────┬──────────────┐
  (96+ SMs)      │ Embed + L0    │   L1   │           │ wait() │ Consume rows │
                 └───────────────┴────────┴─── ... ───┴────────┴──────────────┘
```

这也是为什么查表必须写成一个不读 `hidden` 的独立辅助方法，再由下游通过 `pllm.result("lookup")` 消费结果。如果你把 `table[keys]` 和读取某层 `hidden` 混写在同一个方法里，查表就只能被迫推迟到模型前向结束后的 `state.publish` 锚点（或者嵌在某两层 Transformer 之间的主干流上同步执行），那 1.64 ms 的 PCIe 传输就会失去掩护，彻底暴露在关键路径上。（顺带一提，把纯查表拆成独立方法，连表放在显存里的 `placement="device"` 也会受益：绑定引擎时 Inductor 会自动将取行、类型转换、掩码与缩放融合成单次显存读取，避免把 38.9 MB 的宽向量写回显存再读出来做逐元素运算。）

来看看这套机制在 `Qwen2.5-7B-Instruct`（RTX 4090，128 并发，生成 256 token，示例 11 的 594 MiB `NGramPrior` 宽表）上的实测账单：

| 存储位置配置（7B，128 并发，594 MiB 表） | 吞吐（tok/s） | 相对原生 vLLM | 单步耗时 | pLLM 显存预留峰值（`explain`） |
| :--- | ---: | ---: | ---: | ---: |
| 原生 vLLM（无查表） | 5,877 | 100% | 21.8 ms | — |
| 表在显存（`placement="device"`） | 5,742 | **98%** | 22.3 ms | 1,540 MiB |
| 表在主机（`placement="host"`） | 5,615 | **96%** | 22.8 ms | **946 MiB** |

在每一步 decode 都跨 PCIe 搬运 **38.9 MB** 数据的情况下，依靠 32 线程块的有界零拷贝与 21.8 ms 前向的重叠，表放主机内存只比放显存每步慢了约 **0.5 ms**，端到端吞吐达到原生 vLLM 的 **96%**；而 GPU 显存预留峰值从 `1,540 MiB` 降到了 `946 MiB`，分毫不差地省出了整张表的 **594 MiB** 显存（在功能测试中，哪怕将设备显存预算卡死在 `1 GiB`，也能正常挂载一张 **`3 GiB`** 的主机大表跑完全程）。

---

## 六、 什么时候藏不住？（以及为什么放哪里必须由你签字）

既然放在主机内存能在 7B 上跑出 96% 的吞吐、白省几百兆显存，为什么当显存预算不够时，框架不自动帮用户把表溢写（offload）到主机内存，非要报错逼着作者自己在代码里写下 `placement="host"`？

因为“把 PCIe 传输藏进前向阴影里”并不是一条无条件成立的魔法，它受制于一笔极其具体的算术边界：

$$T_{\text{PCIe}} = \frac{\text{并发数} \times \text{每行读取字节数}}{23.7\text{ GB/s}} < T_{\text{forward}}$$

在 7B 模型、128 并发下，$T_{\text{PCIe}} \approx 1.64\text{ ms}$，而遮风挡雨的前向大伞 $T_{\text{forward}} \approx 21.8\text{ ms}$，自然藏得严严实实。但只要改变等式两边的任何一个变量，账单就会立刻变脸：
* **当模型本身极小时**：如果换成一步前向只有约 `3 ms` 的 `Qwen2.5-0.5B`，在同样的 128 并发与 594 MiB 宽表下，就算有界拉取藏住了绝大部分传输，固定开销与尾部等待依然会让主机查表的吞吐落在原生的 **78%–79%** 左右（显存查表为 82%–86%；这两组数字来自两次较早的 0.5B 测量，不是与 7B 同一次运行）。
* **当并发度继续翻倍或每步查多行宽表时**：如果并发涨到 512 行，每步传输量翻四倍到 `155 MB`，光是过 PCIe 就要花掉 `6.5 ms` 以上；而如果在像 `EngramLite` 那样的窄表场景下（每行宽仅 32，整张表仅 4 MiB），每步跨 PCIe 只搬几 KB 到十几 KB，在 0.5B（vLLM 0.32，生成 256 token，3 轮中位数）上的实测里 `host`（相对第 2 层恒等残差基线的 `95.78%` / `94.94%`，对应 batch 1×16 / batch 8×16）与 `device`（`96.29%` / `95.86%`）的差异本来就小到落在测量噪声之内，有界无界都无所谓。
* **没有 GPU 侧热行缓存**：在推荐系统（如 DLRM）中，主机大表通常会配一层 GPU 显存内的 LRU 热行缓存，利用自然语言 n-gram 的齐夫分布（Zipf's law）省掉高频行的重复传输。为了保持状态池与图捕获的极简确定性，我们没有引入设备端动态哈希缓存——每一步、每一行都实打实地走一次 PCIe。

既然放主机的代价高度依赖你的模型尺寸、行宽与目标并发度，框架就绝不该在背后替你偷偷做决定。正如 `examples/engram_lite.py` 注释所写：*“placement is the author's choice; pLLM never moves the table to satisfy a budget.”* 在启动 GPU 之前，你可以先在 CPU 上调用 `pllm.explain`，它会直接把这笔传输账单打印在终端上（下面是示例 11 在 0.5B、`max_num_seqs=8` 下的一行输出）：

```text
table …… placement: host; host bytes: 622362638; logical transfer bytes/row: 303888; decode bytes/step: 2431104
```

拿 `decode bytes/step` 除以 `22 GB/s`，再和你模型的一步前向耗时比一下，藏不藏得住一目了然。

回过头看，那行只有十几个字符的 `programs = min(tiles, max(1, sms // 4))` 给我们留下的启示其实超越了 PCIe 查表本身：在写 GPU 内核时，我们总习惯于把网格开到最大、把 SM 占用率拉到 100%。但当你写的是一个跑在旁路流上、瓶颈在慢速外部 I/O 的后台内核时，“占满所有 SM”恰恰是最大的毒药。克制住后台任务的硬件胃口，只派刚好能喂饱总线的 32 个线程块去守着水管，把剩下的 96 个 SM 留给前台真正需要算力的矩阵乘法——**只有当两条流索取的硬件资源互补时，并发才不是自欺欺人的零和博弈。**
