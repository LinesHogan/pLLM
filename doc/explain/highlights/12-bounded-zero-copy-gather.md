# 有界的零拷贝 gather 与预取

在推理时算法里，有一类结构长得越来越像检索系统：模型每生成一个 token，就拿当前 token 和前几个历史 token 算一个哈希，去一张很大的外部查找表里取出几行向量，经过门控加进某层的残差流，或者直接作为先验偏置加到词表 logits 上。DeepSeek 的 Engram 记忆层是这个思路，经典的 kNN-LM 与 n-gram 先验也是这个思路。

如果只看算法逻辑，这类查表操作写起来非常简短。在 `examples/engram_lite.py` 里，核心的查表方法只有一行：

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

把前两个历史 token 连同当前 token 映射成两个哈希桶号（二元组和三元组），从 `table` 里切出对应的两行再拍平，算法部分就结束了。

真正的麻烦不在算术，而在 `table` 的体积。一张词表宽度的 n-gram 先验表（形状 `2048 × 152064`，BF16）静态大小是 594 MiB；而在真实的 Engram 场景里，记忆表很容易膨胀到几个 GiB。在线推理服务的显存是一场严格的零和博弈：模型权重和激活缓冲区占定之后，剩下的显存全部归 KV cache。往显存里塞进几百兆甚至上吉字节的静态表，KV cache 池就会等量缩水，单卡能同时承载的并发请求数也随之下降。

既然主机内存（Host DRAM）动辄有上百 GiB，最自然的想法就是把大表留在主机内存里：每一步解码只需要百来行数据，按需跨过 PCIe 总线拉取到 GPU 即可。而且查表只依赖输入 token ID，不依赖模型中间层的隐藏状态，理论上完全可以在每一步开头放到一条旁路 CUDA stream 上异步预取，让传输时间藏在 Transformer 的前向计算背后。

但如果你真的照这个直觉写一个标准的 GPU gather 内核放到旁路流上，会看到一个反常的现象：后台的查表不仅没有藏进前向计算里，反而把主流上的模型前向卡住了，两条流并发跑完的总时间几乎等于串行相加。要让重叠真正发生，我们需要做的恰恰相反——把读取数据的 GPU 线程块（CTA）数量砍到只剩流多处理器（SM）总数的四分之一。

这篇文章就从物理带宽的上限算起，一步步拆解这根 PCIe 总线上的性能账：为什么硬件自带的 DMA 拷贝引擎在这里派不上用场，为什么一个写法完全正常的内核会在硬件块调度器里引发拥堵，以及如何在不改动用户那一行 `table[keys]` 的前提下，在框架底层完成这套有界零拷贝预取。

---

## 一、 信封背面的物理极限账

在写任何内核代码之前，先算清楚硬件的物理上限（Speed of Light）。

假设在一张 NVIDIA RTX 4090（24 GiB，128 个 SM，插在 PCIe 4.0 x16 插槽上）上运行 128 个并发解码序列。每一步解码，每个序列需要从那张 `2048 × 152064` 的 BF16 先验表里查出一行。单行的宽度是 152,064 个双字节浮点数，即 304,128 字节（约 297 KiB）；128 行合在一起，每一步需要跨 PCIe 总线搬运的数据量是：

$$128 \times 304,128\text{ Bytes} \approx 38.9\text{ MB}$$

PCIe 4.0 x16 的理论单向峰值带宽是 32 GB/s，扣除协议包头和事务开销后，实测 GPU 内核直接读取主机内存（零拷贝 gather）能跑到的有效带宽大约在 22 到 24 GB/s 之间（下文玩具实验里有界写法实测为 23.7 GB/s）。按 23.7 GB/s 计算，把这 38.9 MB 数据搬过总线所需的最短物理时间是：

$$\frac{38.9\text{ MB}}{23.7\text{ GB/s}} \approx 1.64\text{ ms}$$

这 1.64 ms 处于什么量级？在同一张 RTX 4090 上（vLLM 0.32.0，开启 CUDA Graph 与 FlashInfer 采样器，128 并发序列），跑完一整步 28 层 `Qwen2.5-7B-Instruct`（BF16）的解码大约需要 21.8 ms；如果换作轻量级的 `Qwen2.5-0.5B-Instruct`，跑完一步解码总共只需要 3.0 ms。

对比这几个数字，结论很清楚：1.64 ms 的过桥时间绝不能串行暴露在主关键路径上，否则 0.5B 模型单步耗时直接增加一半以上，7B 模型也会白白损失近 8% 的吞吐。但反过来看，无论是 21.8 ms 还是 3.0 ms，主干前向的窗口都比 1.64 ms 更长。既然 `lookup` 只需要刚刚提交的 token ID，从时间线最开头的 `request.prepare` 阶段就可以启动读取；只要跨总线的 1.64 ms 能与主干流上的矩阵乘法并行推进，这笔传输开销在端到端时间上就应该接近于零。

问题在于：怎么把这 128 行数据从主机内存搬上 GPU，同时还不干扰正在全力运转的主干流？

---

## 二、 为什么硬件 DMA 拷贝引擎指望不上？

提到“跨 PCIe 搬运数据并与计算重叠”，系统程序员的第一反应通常是使用 GPU 上的硬件拷贝引擎（Copy Engine / DMA）。

现代 GPU 芯片除了拥有负责计算的 SM 之外，还配有独立的异步拷贝引擎。当我们调用 `cudaMemcpyAsync`、`cudaMemcpy2DAsync`、批量拷贝 `cudaMemcpyBatchAsync`，或者像 vLLM 换入换出 KV 缓存块那样调用 `swap_blocks` 时，干活的是 DMA 引擎。DMA 最大的优势在于它完全不占用 SM 的计算与寄存器资源——SM 可以 100% 留给主干流算矩阵乘，DMA 在后台自己走总线搬字节。

遗憾的是，在这个场景下，所有的 `cudaMemcpy` 变体都用不上。原因出在一条基本限制上：发起 DMA 拷贝时，要拷贝的源内存地址（或者源地址的指针数组）必须在 CPU 主机端是已知的。DMA 描述符是由 CPU 驱动组装并推入队列的。

而在连续批处理的解码循环里，要查哪 128 行，行号（`keys`）是在 GPU 显存里动态算出来的——当前 token ID 刚刚由上一步的 GPU 采样器生成，哈希运算 `ngram_keys(ids, buckets)` 也在 GPU 上执行，CPU 从头到尾并不知道这一步命中了表里的哪几行。

如果强行要用 DMA 拷贝引擎，CPU 就必须先把这 128 个整数行号从 GPU 显存拷回主机内存。这不仅意味着一次阻塞流水线的设备到主机同步（Device-to-Host Synchronization），而且正如我们在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里讨论过的，稳态解码步必须被录进静态的 CUDA Graph 反复回放——录制在图里的 `memcpy` 节点，其源地址参数在录制那一刻就已经焊死，回放时根本无法根据每步变化的 token 动态更改源地址。

既然不能把行号拉回 CPU，唯一的出路就是反过来：**让 GPU 上的计算内核自己拿着行号，直接跨过 PCIe 去主机内存里取数据。**

在 CUDA 的统一虚拟地址空间（Unified Virtual Addressing，UVA）下，只要主机上的大表分配在锁页内存（Pinned Memory，不可被操作系统换出到磁盘）中，GPU 内核就可以直接通过映射后的设备指针发起读取。vLLM 内部已经提供了一个现成的辅助函数 `get_accelerator_view_from_cpu_tensor`：给它传入一个 `.pin_memory()` 后的 CPU 张量，它会返回一个 `is_cuda == True`、但底层 `.data_ptr()` 物理首地址完全相同的张量视图。没有任何数据被提前复制，当 GPU 线程对这个视图地址执行加载指令（如 Triton 的 `tl.load`）时，SM 的内存单元会直接向 PCIe 控制器发出读事务，把那一行字节拉进 GPU 寄存器——这就是零拷贝（Zero-Copy）gather。

行号留在 GPU，寻址在 GPU 上完成，全程零主机同步，而且因为只是一个普通的计算内核，它可以毫无障碍地被录进 CUDA Graph。

看起来一切顺理成章。但当我们写下第一个直观的 gather 内核时，硬件调度器给我们上了一课。

---

## 三、 一个对照实验：9,600 个线程块是怎么卡死主干前向的？

为了把现象从庞大的推理引擎里单独剥离出来，我们可以用一个一百多行的独立微基准脚本 `toy_host_gather.py` 来复现这个问题。

实验的设置与真实负载保持一致：在锁页主机内存里放一张 `2048 × 151936` 的 BF16 表（约 594 MiB），通过 `get_accelerator_view_from_cpu_tensor` 获取它的 UVA 视图；在 GPU 上随机生成 128 个行号（每次读取 `128 × 151936 × 2` 字节，即 37.1 MiB）；同时在主干流上用 4 组 $4096 \times 4096$ 的 BF16 矩阵乘法构造一个耗时约 3.88 ms 的模拟“前向计算（forward）”。然后把读表的 gather 内核放在旁路 `side` 流上，与主干流并发执行，用 CUDA Event 测量两边单独跑和一起跑的耗时。

先看第一种最符合直觉的内核写法 `naive_rows`：

```python
# check-docs: skip: excerpt from toy_host_gather.py
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

这是教科书里最标准的二维并行切分：把每一行按 `block = 2048` 个元素（4 KiB 连续字节）切分成 $\lceil 151936 / 2048 \rceil = 75$ 个列块（tiles），128 行一共是 $128 \times 75 = 9,600$ 个列块。我们启动一个大小为 9,600 的一维网格（grid），每个 Triton program（即一个 CUDA 线程块 CTA）负责搬运其中一个 4 KiB 的列块。

再看第二种写法 `bounded_rows`：

```python
# check-docs: skip: excerpt from toy_host_gather.py
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

循环体内部的寻址和读写完全没变，唯一的改动是在外面套了一层 `tl.range` 跨步循环，并且把启动网格的大小限制为 `max(1, sms // 4)`——在拥有 128 个 SM 的 RTX 4090 上，总共只启动 **32 个线程块**，由这 32 个常驻线程块轮流把 9,600 个列块搬完（即持久化内核，Persistent Kernel）。

在同一张 RTX 4090（PCIe 4.0 x16，vLLM 0.32.0，PyTorch 2.13.0+cu129，预热 3 轮后取 15 轮中位数）上运行这个脚本，输出结果如下：

```text
SM 数=128  naive grid=9600  bounded grid=32
forward = 4 组 4096^3 matmul；传输量 37.1 MiB

kernel      gather    GB/s  forward  concurrent  overlap saved
naive       2.38ms    16.3   3.88ms      6.19ms         0.07ms
bounded     1.64ms    23.7   3.88ms      4.00ms         1.53ms
```

先看 `naive` 那一行。单独运行 gather 需要 2.38 ms，单独运行模拟前向需要 3.88 ms。把它们放到两条不同的 CUDA stream 上并发运行，总耗时是 **6.19 ms**——只比完全串行相加（$2.38 + 3.88 = 6.26\text{ ms}$）省下了 **0.07 ms**。两条流在硬件上几乎是排着队先后跑完的。

再看 `bounded` 那一行。仅仅把网格从 9,600 个块缩减到 32 个块，首先单独运行 gather 的时间从 2.38 ms 降到了 1.64 ms，有效带宽从 16.3 GB/s 升到了 23.7 GB/s；更重要的是，当它与 3.88 ms 的前向并发运行时，总耗时只有 **4.00 ms**。1.64 ms 的 PCIe 传输里有 **1.53 ms**（超过 93%）被藏在了前向矩阵乘法的背后，端到端只多付出了 0.12 ms。

为什么发射 9,600 个线程块反而会让双流并发彻底失效？

这要回到 GPU 硬件块调度器（Block / Work Distributor）的工作机制上。在 CUDA 中，将内核放进不同的非阻塞 stream，只是告诉硬件“这两批线程块之间没有先后依赖，可以同时派发”。至于它们能不能真的在同一个时刻一起跑，取决于 GPU 上的 128 个 SM 还有没有空闲的驻留槽位（Resident Slots）。

当旁路流发射 `naive_rows` 时，9,600 个线程块瞬间涌入硬件队列。块调度器看到有大量待执行的线程块，会立刻把整张显卡全部 128 个 SM 的驻留槽位塞满。

```text
Naive Grid (9,600 CTAs launched on side stream):
  ┌────────────┬────────────┬────────────┬─────┬────────────┐
  │   SM 0     │   SM 1     │   SM 2     │ ... │  SM 127    │
  │ [Gather]   │ [Gather]   │ [Gather]   │     │ [Gather]   │  <-- All 128 SMs occupied!
  │ (waiting   │ (waiting   │ (waiting   │     │ (waiting   │      Tensor Cores sit idle
  │  on PCIe)  │  on PCIe)  │  on PCIe)  │     │  on PCIe)  │      waiting on 23.7 GB/s bus.
  └────────────┴────────────┴────────────┴─────┴────────────┘
        ▲
        └── Main Stream MatMul waits at the door: no free SM slots until gather drains.

Bounded Grid (32 CTAs = SMs / 4):
  ┌────────────┬─────┬────────────╦────────────┬────────────┬─────┬────────────┐
  │   SM 0     │ ... │   SM 31    ║   SM 32    │   SM 33    │ ... │  SM 127    │
  │ [Gather]   │     │ [Gather]   ║ [ MatMul ] │ [ MatMul ] │     │ [ MatMul ] │
  │ (loops     │     │ (saturates ║ (Main stream forward runs across           │
  │  tiles)    │     │  PCIe bus) ║  the remaining 96 SMs concurrently!)       │
  └────────────┴─────┴────────────╩────────────┴────────────┴─────┴────────────┘
```

一旦这批线程块住进了全部 128 个 SM，问题就来了：它们发出的 `tl.load` 指向的是大洋彼岸的主机内存。受限于 23.7 GB/s 的 PCIe 带宽（约为这张卡 1,008 GB/s 显存（GDDR6X）带宽的四十分之一），这些线程块绝大部分时钟周期都处于停顿（Stall）状态，干等着字节从 PCIe 总线上一点点滴过来。

在等待期间，SM 内部的算术单元和 Tensor Core 完全是空闲的；但由于 GPU 的线程块调度是非抢占式的（Non-preemptive），一个线程块一旦被分配到 SM 上，除非它自己把指令执行完退出，否则硬件不会因为它在等内存就把它踢出 SM。而且每当某个块好不容易等来 4 KiB 数据写回并退出时，旁路队列里还有成千上万个同样要等 PCIe 的 `naive` 线程块排在前面争抢空位。结果就是：128 个 SM 的房间被一群“光占地方不干算术”的访存块占满（这种写法单独跑时有效带宽也只有 16.3 GB/s），而主干流上真正需要 Tensor Core 的矩阵乘法内核被堵在门外，只能等 9,600 个块几乎全部排空之后才能进场。

反观 `bounded_rows`：PCIe 4.0 x16 总线是一条很窄的管道，根本不需要 128 个 SM 同时去拉数据；仅仅 32 个线程块循环发起读请求，就已经足够把 23.7 GB/s 的总线喂饱了。通过把网格上限硬性卡在 `sms // 4`（32 个块），旁路流最多只会触碰四分之一的 SM，剩下四分之三的 SM（96 个）完好无损地留给了主干流的前向矩阵乘。一边吃的是稀缺的 PCIe 带宽，另一边吃的是充裕的 SM 算力与显存带宽——两边的瓶颈资源正交互补，真正的硬件重叠才得以发生。

在真实的推理引擎剖析中（RTX 4090，`Qwen2.5-0.5B`，128 并发，示例 11 的宽表设置，至少 5 轮中位数，一版较早的实现），这两种写法的表现与上面的玩具实验完全吻合：无界 UVA gather 一次发射 37,984 个 CTA，占满 100% 的设备槽位，在 1,753.7 µs 的读取时间里只有 26.2 µs 与主流重叠，提前发射反而让单步变慢到 5.939 ms；换成 32 个 CTA 的有界 gather 后，1,634.6 µs 的读取时间里有 **99.14%** 与主流的模型内核重叠执行。与在前向末尾才发射无界读的写法相比，每步的主机-设备空隙从 1.75 ms 降到了 0.24 ms，单步总耗时从 5.697 ms 降到 4.194 ms（端到端吞吐从 22,469 升至 30,522 tok/s，提升 35.8%）。

---

## 四、 放进框架底层：拦截、算子契约与预取调度

弄清楚了“必须用有界网格的持久化内核读取，并在每步开头提前发射”，接下来的工程问题是：怎样把这套机制装进系统，同时不让算法作者在代码里写任何底层的算子调用？

在文章开头的 `EngramLite` 里，作者声明表在主机内存的方式，仅仅是在初始化配置里写一个 `placement="host"`：

```python
# check-docs: skip: excerpt from examples/engram_lite.py
@property
def memory(self) -> pllm.InstanceState:
    """Only the gathered table uses the declared placement; projections stay on device."""
    return pllm.InstanceState(
        {"table": (self.buckets, self.width)}, init=pllm.Pretrained(self.path, placement=self.placement)
    )
```

而在 `lookup` 方法体内，作者写的依然是标准的 PyTorch 索引 `table[ngram_keys(ids, self.buckets)]`。从这一行普通索引到后台的 32 块持久化内核，中间靠三层机制衔接。

### 1. 生产级算子 `pllm::host_rows`

首先是位于 `pllm/kernels/page_copy.py` 中的生产内核 `_host_rows`。相比于前面的玩具代码，它需要多处理两件真实 PyTorch 代码里常见的需求：负数下标回绕（例如 `table[-1]`），以及对张量切片视图（非连续步长）的支持：

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

注意参数列表里的一个细节：只有列宽 `width` 和分块大小 `block` 被标为编译期常量 `tl.constexpr`，而总行数 `height`、两个维度的步长 `stride0`、`stride1` 以及当前步要取的行数 `count` 全部作为运行时整数传入。这样一来，无论用户是先对表做了列切片视图，还是每一步传入的索引形状不同，都能直接复用同一个编译好的 Triton 内核，不会触发运行期重编译。

外层的自定义算子 `pllm::host_rows` 负责计算 `min(tiles, max(1, sms // 4))` 的有界网格并发起调用：

```python
# check-docs: skip: excerpt from pllm/kernels/page_copy.py
@torch.library.custom_op("pllm::host_rows", mutates_args=())
def host_rows(source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Read GPU-indexed pinned rows with a bounded grid, leaving SMs for the forward."""
    output = source.new_empty((*rows.shape, source.shape[1]))
    if output.numel():
        block = 2048
        tiles = rows.numel() * triton.cdiv(source.shape[1], block)
        sms = int(torch.cuda.get_device_properties(source.device).multi_processor_count)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        programs = min(tiles, max(1, sms // 4))
```

为了让这个自定义算子无缝融入整个编译与规划体系（见[《参数覆盖层》](06-parameter-overlays.md)与[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)），我们紧接着为它注册了两个伴侣规则：

```python
# check-docs: skip: excerpt from pllm/kernels/page_copy.py
@host_rows.register_fake  # pyright: ignore[reportUnknownMemberType]
def host_rows_meta(source: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """The ordinary row-gather shape, without accessing pinned storage."""
    return source.new_empty((*rows.shape, source.shape[1]))


@host_rows.register_vmap  # pyright: ignore[reportUnknownMemberType]
def host_rows_vmap(
    info: VmapInfo, in_dims: tuple[int | None, ...], source: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Fold mapped indices into one gather of the shared, two-dimensional table."""
    source_dim, row_dim = in_dims
    assert source_dim is None and row_dim is not None
    return host_rows(source, rows.movedim(row_dim, 0)), 0
```

`register_fake` 让 CPU 侧的 `meta` 张量追踪在不触碰真实锁页内存的情况下推导出输出形状与显存占用；而 `register_vmap` 则保证当外层用 `torch.vmap` 将单请求代码向量化到一批请求时，多个请求的索引会被折叠成对共享二维表的**单次** `host_rows` 调用。

### 2. 用 `TorchFunctionMode` 透明替换索引操作

算子就位后，`pllm/execution/torch_private.py` 中的 `HostGather` 利用 PyTorch 的 `TorchFunctionMode`，在执行用户方法时拦截所有针对已声明主机表的读取操作：

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
    result = func(*args, **keywords)
    # Planning already restricts host-table operations to views and row reads.
    if isinstance(result, torch.Tensor):
        self.tables = (*self.tables, result)
    return result
```

只要被操作的张量 `source` 在主机表名单 `self.tables` 里，无论是 `table[indices]`、`index_select`，还是在这段代码上方同样被拦截的 `F.embedding` / `torch.embedding`，都会被直接重定向到 `host_rows(source, indices)`。如果用户先对表做了切片等视图变换（不触发实际读取），倒数第二行会把生成的视图张量 `result` 也加入 `self.tables` 追踪列表，保证后续对该视图的按行索引同样走有界 gather。

而如果用户不小心对一张主机大表写了全表算术（比如 `table + 1` 或矩阵乘 `x @ table`），或者直接把未 gather 的主机表视图作为方法返回值交出去，规划期配对的 `GatherTrace`（基于 `TorchDispatchMode`）会在构建阶段立即抛出 `ValueError` 拦截——主机表只允许做按行抽取，绝不允许把几百兆的原表在每步里全量拉过 PCIe。

### 3. 时间线上的预取与写法纪律

最后一步是发射时机。正如下一篇[《自动放置与重叠》](11-automatic-placement-overlap.md)将详细展示的，当规划器扫描语义图时，看到 `lookup` 只读取 `pllm.input_tokens` 和不可变参数（`immutable`），不依赖任何模型层的隐藏状态，同时具备进入 CUDA Graph 的能力，就会自动将它放置在单步最开头的 `request.prepare` 锚点、分配到 `side` 旁路流上执行：

```text
Step Timeline (HostGather on Side Stream + Forward on Main Stream):

  Anchor:        [0: request.prepare]                       [1: model.execute]
                 ────────────────────                       ──────────────────
  Side Stream:   ┌───────────────────────────────────────┐
  (<= 32 SMs)    │ pllm::host_rows(table, ngram_keys)    │──► record(event)
                 └───────────────────────────────────────┘         │
                        ║  (Overlaps with early layers)            ▼
  Main Stream:   ┌──────╨────────┬────────┬─── ... ───┬────────┬──────────────┐
  (96+ SMs)      │ Embed + L0    │   L1   │           │ wait() │ Consume rows │
                 └───────────────┴────────┴─── ... ───┴────────┴──────────────┘
```

在 `pllm.explain` 的输出报告里，这一调度决策会显示为一行确认：

```text
lookup  compute  request.prepare / side / captured
```

旁路流在 `request.prepare` 立即开始用 32 个线程块拉取主机行，完成后记录一个 CUDA Event；主干流同时向前推进词嵌入和前几层 Transformer，直到真正需要消费 `pllm.result("lookup")` 的那个站点（例如第 2 层残差挂载点，见[《图内挂载点》](05-in-graph-attach-points.md)，或采样器前的 `@pllm.add_logits`，见[《采样尾声融合》](07-sampler-epilogue.md)），主干流才通过 `wait_event` 等待该事件完成。

这也解释了一条重要的代码编写纪律：**为什么必须把主机查表写成一个不读 `hidden` 的独立辅助方法，再由下游通过 `pllm.result("lookup")` 引用它？**

如果你把 `table[keys]` 和 `pllm.hidden(0)` 写在同一个方法里，这个方法就必须等到模型前向跑完、隐藏状态在 `state.publish` 发布之后才能启动；更糟的是，如果你在 `@pllm.custom_residual` 这样的模型层内挂载点里直接去读主机表，那个挂载点是在主干流执行到某一层中间时同步调用的——这等于强迫整个主干流在半路停下来干等 1.64 ms 的 PCIe 传输，预取重叠也就无从谈起了。

顺带一提，把查表拆成“只读 token + 只读表”的独立方法，对于放在显存里的表（`placement="device"`）同样有好处：当 `pllm/execution/tensor_declaration.py` 中的 `_lookup_math` 识别出这种纯查表模式时，会自动用 Inductor 将取行（gather）、类型转换（cast）、条件掩码（`where`）和缩放融合成单次表读取内核，省掉把宽向量写回显存再读出来做逐元素运算的额外往返（在 0.5B、示例 11 宽表设置的一次测量中，这项融合配合提前发射将 `device` 吞吐从原生的 66% 提升到了 84%；若需严格保持未融合的浮点算术，也可设置 `allow_numerical_transforms=False` 将其关闭）。

---

## 五、 真实模型上的代价账单与边界

把整套机制放进完整的 vLLM 引擎后，我们在单张 RTX 4090（vLLM 0.32.0，CUDA Graph 与 FlashInfer 采样器开启，8 个 prompt × `n=16` 共 128 并发序列，生成 256 个 token，取预热后 3 轮中位数）上，对挂载了 594 MiB `NGramPrior` 宽表（通过 `@pllm.add_logits` 作用于词表分布）的 `Qwen2.5-7B-Instruct` 和 `Qwen2.5-0.5B-Instruct` 进行了测量：

| 模型与表放置位置（128 并发，594 MiB 表） | 端到端吞吐 (tok/s) | 相对各自原生 vLLM | 单步耗时 | pLLM 显存预留峰值 (`explain`) |
| :--- | ---: | ---: | ---: | ---: |
| **Qwen2.5-7B** 原生 vLLM（无查表） | 5,877 | 100% | 21.8 ms | — |
| **Qwen2.5-7B** 表在显存 (`placement="device"`) | 5,742 | **98%** | 22.3 ms | 1,540 MiB |
| **Qwen2.5-7B** 表在主机 (`placement="host"`) | 5,615 | **96%** | 22.8 ms | **946 MiB** |
| **Qwen2.5-0.5B** 原生 vLLM（无查表） | 39,584 | 100% | 3.2 ms | — |
| **Qwen2.5-0.5B** 表在显存 (`placement="device"`) | 34,083 | **86%** | 3.8 ms | 702 MiB |
| **Qwen2.5-0.5B** 表在主机 (`placement="host"`) | 31,445 | **79%** | 4.1 ms | **109 MiB** |

*(注：0.5B 的三行吞吐取自示例 11 改测 7B 之前的 0.5B 笔记本输出，不是与 7B 同一次运行；`performance.md` 记录过另一次 0.5B 测量，原生、device、host 分别为 38,821、31,750 [82%, 4.03 ms/步]、30,174 tok/s [78%, 4.24 ms/步]，相对差距相近。)*

把这张表里的数字拆开看，物理规律非常清晰：

在 7B 模型上，尽管每一步解码都要实打实地跨过 PCIe 搬运 38.9 MB 数据（单独传输需约 1.64 ms），但因为在 21.8 ms 的漫长前向背后做了有界重叠，把表从显存挪到主机内存每步**只多花了约 0.5 ms**，端到端吞吐依然保持在原生 vLLM 的 **96%**。与此同时，pLLM 在显卡上预留的峰值内存从 1,540 MiB 降到了 946 MiB（在 0.5B 上则是从 702 MiB 降到 109 MiB），完整省出了那张表本身的 **594 MiB** 显存。在功能测试中，即便把 GPU 侧的显存预算上限卡在 1 GiB，系统也能顺利挂载一张 **3 GiB** 的主机表跑完全程。

而对比不同尺寸的模型与不同宽度的表，也能清楚地看到这套机制的适用边界：

首先是**前向步长与并发批大小对传输的掩盖能力**。主机查表的每步传输字节数与“并发请求数 × 单行字节数”成正比。在 7B 模型（21.8 ms/步）甚至 128 并发的 0.5B 模型（约 3.2 ms/步）下，1.64 ms 的传输都能被前向计算有效掩盖（0.5B 上 `host` 比 `device` 每步也只慢约 0.2–0.3 ms）。但如果并发度继续成倍放大（例如翻四倍到 512 行，每步传输超过 150 MB、耗时超过 6 ms），在小模型上 PCIe 传输时间就会超过前向计算本身，超出部分将无法避免地暴露在关键路径上。

反过来，如果表本身是**窄行表**——比如 `EngramLite` 默认配置下宽度只有 32、总大小仅 4 MiB 的表——每步跨 PCIe 传输的数据量只有几 KB 到十几 KB。在 RTX 4090（vLLM 0.32，0.5B，生成 256 token，3 轮中位数）的实测中，这种窄表放在 `host`（相对第 2 层恒等残差基线吞吐为 95.78% / 94.94%，分别对应 batch 1×16 与 batch 8×16）与放在 `device`（96.29% / 95.86%）的差异完全落在测量噪声以内。此时 PCIe 根本不是瓶颈，有界网格与普通读取没有可测差异，只有当表的总容量大到显存装不下时才需要放上主机。

其次是与工业界其他卸载方案的取舍对比。像 ZeRO-Infinity、FlexGen 或 vLLM 自带的 `--cpu-offload-gb` 权重卸载，搬运的是固定的模型层权重或已知位置的 KV 块，地址在主机端完全已知，因此可以直接用 DMA 拷贝引擎做流水线预取；而这里的场景是“只有 GPU 知道这一步要读哪几行”。在同样面对 GPU 侧动态索引的推荐系统（如 DLRM 的超大 Embedding 表）中，常见的重型工程方案是在显存里维护一层 GPU 端 LRU 热行缓存（利用自然语言 n-gram 的齐夫分布局部性减少跨总线次数）。pLLM 没有引入 GPU 端缓存层和配套的并发一致性管理——每一步、每一行都直接走 PCIe 读取，且只支持只读的预训练表（`Pretrained`），不支持在主机上放置需要每步写入的可变状态；`sms // 4` 也是一个固定的经验分母，未按 PCIe 代际或算子负载做动态自适应。

正因为“把表放在主机内存”是一笔收益与代价高度依赖模型步长、行宽和并发度的交易，我们在设计上坚持了一条原则：**放置位置必须由作者显式指定，系统绝不在显存不足时自动把表静默卸载（Silent Offload）到主机内存。**

正如 `examples/engram_lite.py` 的文档字符串所写：`placement is the author's choice; pLLM never moves the table to satisfy a budget.` 在真正启动 GPU 之前，你可以先在 CPU 上运行 `pllm.explain(workload, config)`，它会直接列出主机表的字节账单（下面是示例 11 在 0.5B、`max_num_seqs=8` 下的输出）：

```text
table …… placement: host; host bytes: 622362638; logical transfer bytes/row: 303888; decode bytes/step: 2431104
```

拿到 `logical transfer bytes/row` 后，只需要用 `并发数 × 每行字节数 / 22 GB/s` 做一个简单的除法：如果算出来的毫秒数明显短于模型的一步前向时间，那么有界持久化 gather 就能把这段跨海传输稳稳藏在阴影里；如果算出来比前向还长，账单在启动前就已经摆在了桌面上。
