# 有界的零拷贝 gather 与预取

Engram 一类的记忆查表是那种“越来越脏”的推理时算法的典型：用当前 token 和前一两个 token 做个哈希，去一张很大的表里取几行向量，门控之后加到某一层的残差上。pLLM 的示例里有一个精简版（`examples/engram_lite.py`），查表的方法只有一行：

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

难的是表的大小。示例 11 里一张词表宽的 n-gram 先验，`2048 × 152064` 的 BF16，就有 594 MiB；真实的记忆表可以到 GiB 级。放在显存里，就是从 KV cache 那里拿走这么多空间。

那就放主机内存？这篇讲的就是这个看似简单的决定背后的事：GPU 怎样在“行号只有 GPU 知道”的前提下直接从主机内存取行；为什么一个写法再正常不过的 gather kernel 会把模型前向饿死；为什么把线程块砍到四分之一反而更快；以及为什么“放在主机”必须由作者亲手写下来。

## 先算账

128 个并发请求，每步每个请求取一行，一行 152,064 个 BF16，就是 304,128 字节；一步一共 38.9 MB。RTX 4090 的 PCIe 4.0 x16 实测约 23.7 GB/s，搬完要 1.6 ms 左右。7B 一步 decode 约 21.7 ms，1.6 ms 是 7%；0.5B 一步只有约 3 ms，1.6 ms 超过半步。如果这 1.6 ms 摆在关键路径上，小模型的吞吐就去了一半。

所以问题是：能不能让这 1.6 ms 和模型前向**同时**发生，藏在前向的阴影里？

## 拷贝引擎帮不上忙

做系统的人第一反应是 DMA：GPU 有专门的拷贝引擎，`cudaMemcpyAsync`、带跨步的 `cudaMemcpy2DAsync`、批量拷贝接口、vLLM 用来换入换出 KV 的 `swap_blocks`，都不占计算单元。可它们都有一个前提：源地址（或者指针数组）在**主机端**已知。我们的行号是 GPU 上用 token id 哈希出来的，CPU 不知道。要用拷贝引擎，就得先把 128 个行号读回 CPU——一次主机同步，正是 decode 循环里最不能做的事（[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。何况录进 CUDA graph 的 memcpy 节点，参数是录制时定死的。

剩下的路只有一条：让 GPU 的 kernel 自己去主机内存里读。这需要两样东西。一是锁页（pinned）内存：锁住的主机内存不会被操作系统换出，可以映射进 GPU 的统一虚拟地址空间（UVA）。二是一个指向它的“GPU 张量”。vLLM 有现成的工具 `get_accelerator_view_from_cpu_tensor`，给一个锁页的 CPU 张量造一个 CUDA 上的视图——它和原张量的 `data_ptr` 相同，是别名，不是拷贝。之后，GPU 上任何一条普通的 load 指令只要碰到这个地址，硬件就会自己经 PCIe 去取。这就是“零拷贝”：没有显式的拷贝，读本身就是传输。

行号在 GPU 上，读由 GPU 发起，CPU 全程不参与。看起来问题解决了。

## 一个正常的 kernel，一次意外的拥堵

为了看清楚接下来会发生什么，我们写了一个独立的小实验（不在仓库里，只用了 vLLM 的那个视图工具）：一张 594 MiB 的锁页表，128 个 GPU 上生成的行号，一个用四组 4096³ 矩阵乘标定到约 3.9 ms 的假“前向”；把 gather 放在旁路 stream 上，和主流上的前向一起跑，看能藏住多少。

先写最正常的 gather：每个线程块（program）负责一个（行，列块），列块 2,048 个 BF16，也就是 4 KiB 连续字节。

```python
# check-docs: skip: excerpt from the toy experiment (not in the repository)
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

128 行 × 每行 75 个列块，grid 是 9,600 个线程块。这是 GPU 编程最常见的写法：网格大小和数据量成正比。

再写一个“有界”的版本：同样的循环体，但 grid 固定为 SM 数的四分之一，每个 program 以整个网格的大小为步长，循环领取多个列块——所谓持久化 kernel。

```python
# check-docs: skip: excerpt from the toy experiment (not in the repository)
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

在一张 RTX 4090 上跑一次，输出原样如下：

```text
# toy_host_gather.py, run 2026-11-09 on local RTX 4090 (PCIe 4.0 x16), .venv-vllm032 (vLLM 0.32.0, torch 2.13.0+cu129), flock + taskset -c 0-15, one run
SM 数=128  naive grid=9600  bounded grid=32
forward = 4 组 4096^3 matmul；传输量 37.1 MiB

kernel      gather    GB/s  forward  concurrent  overlap saved
naive       2.38ms    16.3   3.88ms      6.19ms         0.07ms
bounded     1.64ms    23.7   3.88ms      4.00ms         1.53ms
```

各列的意思：`gather` 是单独跑 gather 的时间，`GB/s` 是它折算出的有效 PCIe 带宽；`forward` 是单独跑假前向的时间；`concurrent` 是两条 stream 同时跑的总时间；`overlap saved` 是 `forward + gather − concurrent`——完全串行时为 0，完全藏住时等于两者中较小的那个。

`naive` 那一行：前向 3.88 ms，gather 2.38 ms，一起跑要 6.19 ms，几乎就是两者之和，只省下 0.07 ms。

为什么？GPU 不是很擅长隐藏延迟吗？它擅长的是**已经驻留在 SM 上的** warp 之间互相掩护：一个 warp 在等内存，另一个 warp 就上去算。可 9,600 个线程块一发射，块调度器会尽可能把它们塞满全部 128 个 SM 的驻留槽位。每个块都在等 PCIe——在 GPU 的时间尺度上，那是很久——SM 的计算单元几乎闲着，**槽位却全被占了**。主流上的矩阵乘想发射，发现没有空位；调度器不会因为一个块在等总线就把它踢出去。于是“并发”退化成了串行。一个比喻：一根细水管，派一万个人排着队去接水，他们手里都空着，厨房却被站满了，真正要炒菜的大厨进不了门。

有界版本只有 32 个线程块。单独测，它的带宽反而更高（23.7 GB/s 对 16.3 GB/s）——喂饱一根 PCIe 本来就不需要几千个读者。和前向一起跑：4.00 ms，只比单独跑前向多 0.12 ms，省下了 1.53 ms。32 个块占着 32 个 SM 的一部分槽位等 PCIe，其余的 SM 空出来给矩阵乘。两边要的资源不一样——一边要总线，一边要算力——才藏得住。

在 pLLM 的真实路径上，我们设计这条路径时的 0.5B profile 记录给出的是同一个故事：不限线程块的读法发了 37,984 个 CTA，1,753.7 µs 的读表只有 26.2 µs 和前向重叠；换成有界读法后，读表时间的 99.14% 和主流的 kernel 重叠了（据当时的开发记录，原始 profile 已不在本机）。

## 放进 pLLM：一个算子、一层拦截、一个预取

生产版本的 kernel 在 `pllm/kernels/page_copy.py` 里，和玩具几乎一样，只多了两件事：行数和步长作为运行时参数（这样一个编译好的 kernel 可以复用于各种视图），以及负的行号按 Python 的语义回绕：

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

“有界”的那个界写在包它的 custom op 里，就一行：

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
        ...
    return output
```

`programs = min(tiles, max(1, sms // 4))`：4090 上是 32。这个 op 还注册了一个 fake 实现（规划时只给形状，不碰锁页存储）和一个 vmap 规则（把映射的索引折叠成一次 gather），所以它能被 meta 追踪、能被 vmap、能被录进 graph。

作者写的却是普通的 `table[keys]`。中间那一步替换由 `pllm/execution/torch_private.py` 里的 `HostGather` 完成——一个 `TorchFunctionMode`：在作者的方法执行期间，凡是对声明为 host 的表做 `embedding`、单张量索引 `__getitem__`、`index_select`，都被换成 `pllm::host_rows`：

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

对主机表的其他操作在规划时就会被拒绝：只允许按行读。作者的代码一个字不用改，`placement` 从 `"device"` 换成 `"host"`，读法就从普通的显存 gather 变成了有界的 PCIe 读。

最后是**什么时候**读。查表只依赖 token id 和只读的表，这两样在一步刚开始时就已经有了。调度器认出这种任务，把它放到 `request.prepare` 锚点、旁路 stream 上发射（[《自动放置与重叠》](11-automatic-placement-overlap.md)的 `select_prefetch`）；模型里消费它的那一层，只在用之前等一个 event（[《图内挂载点》](05-in-graph-attach-points.md)）。`explain` 里那一行是 `lookup  compute  request.prepare / side / captured`。这一切要成立，查表必须写成一个独立的方法：如果它和读 `pllm.hidden` 的计算写在一起，它就得等前向结束，传输就藏不住了。

设备表也从同一个识别里受益：只读 token、只读表的查表方法，Inductor 会把 gather、类型转换、`where`、缩放融合成一次表读（可以用 `allow_numerical_transforms=False` 关掉）。

## 7B 上的账单

示例 11（RTX 4090，vLLM 0.32，Qwen2.5-7B，128 并发 = 8 × n=16，256 token，594 MiB 的 `NGramPrior` 表）：

| | tok/s | 相对原生 | 约每步 |
|---|---:|---:|---:|
| 原生 vLLM | 5,877 | 100% | 21.8 ms |
| 表在显存（`placement="device"`） | 5,742 | 98% | 22.3 ms |
| 表在主机（`placement="host"`） | 5,615 | 96% | 22.8 ms |

每步 38.9 MB 跨过 PCIe（物理上约 1.6 ms），放主机比放显存每步只慢约 0.5 ms。显存这边，`explain` 报告 pLLM 预留的峰值从 1,540 MiB 降到 946 MiB，正好少了一整张表；测试里，3 GiB 的主机表在 1 GiB 的 GPU 预算下也能挂上。

## 放哪里，必须由作者决定

现在可以解释为什么 pLLM 永远不会因为显存不够就把表自动挪到主机。上面的账在 7B 上很好看；可同样的 1.6 ms 放在 0.5B 的 3 ms 一步里，是另一回事。跨 PCIe 的代价太大、太依赖模型和负载，这种取舍应该由作者写下来，框架负责把代价算清楚。示例的 docstring 写的就是这句话：“placement is the author's choice; pLLM never moves the table to satisfy a budget”。`explain` 会把账单印出来：每行逻辑上要搬多少字节（`logical transfer bytes/row`）、每步一共多少（`decode bytes/step`）、主机上占多少（`host bytes`）。估算规则也很简单：`并发数 × 每行字节 / 22 GB/s` 明显短于一次前向，就藏得住。

## 在哪里更差

这条路径只支持只读的预训练表（`Pretrained`）。可变的状态、模块、在线训练的参数都不能放主机。

没有热行缓存。推荐系统里放在主机的大 embedding 表，通常会在 GPU 上配一层热行缓存，因为访问往往高度集中；n-gram 的分布也很可能如此。pLLM 没有这一层——每一步、每一行都实打实地过一次 PCIe。如果你的访问局部性很强，这是在浪费总线。

收益只在宽行上。Engram 精简版的表很窄（每行 32 个数，整张表 4 MiB）：在 RTX 4090、vLLM 0.32、0.5B、n=16、256 token、3 轮的测量里，相对“第 2 层恒等残差”的基线，batch 1 时 host 是 95.78%、device 是 96.29%，batch 8 时 host 是 94.94%、device 是 95.86%——差别在噪声里。传输本来就小，有界不有界都无所谓。

四分之一是一个经验常数。`sms // 4` 是在 RTX 4090 上定下的，没有按 GPU 型号或前向的负载自适应。

写法上的纪律也来自这里：主机表不能被模型层内的方法直接读（那会在主流上某一层的中间等 PCIe，卡住之后的前向），只能通过一个独立的查表方法、再用 `pllm.result` 取结果；查表方法不能读 hidden。传输量和并发数成正比，batch 越大越难藏。

## 可以带走的东西

这篇最反直觉的地方，是“少用一点 GPU 反而更快”。我们习惯把 kernel 的 grid 开到和数据量一样大，让占用率越高越好；可当一个 kernel 的瓶颈根本不在 SM 上——它在等一根细细的 PCIe——占满 SM 只会把别人挤出去。并发能不能带来收益，取决于两边要的资源是否互补，而不是 stream 开了几条。

另一点是那张账单。最贵的决定——把东西放到另一种内存里——由人来做；机器做它最擅长的事：在你做决定之前，把每一步要搬多少字节算给你看。
