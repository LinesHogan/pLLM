# 参数覆盖层：顺着 PyTorch 的抽象层往下走

在单卡的 PyTorch 脚本里，给一个线性投影层加一点“私货”是全世界最容易的事。

如果你要加一个稠密的权重增量 $\Delta W$，一行矩阵乘就够了：`y = x @ (W + delta).T`。如果你要挂一对低秩适配器（LoRA），也不过是多写一次乘法：`y = x @ W.T + (x @ A.T) @ B.T`。在研究者的笔记本里，权重就摆在那里，形状规规矩矩，你想读就读，想改就改。

但只要你试着把这行代码搬进像 vLLM 这样的生产级推理引擎，平静的水面下立刻会浮出两座冰山。

第一座冰山是**成百上千个请求各怀心事**。像 Text-to-LoRA 这类方法会为每一个请求实时生成一份专属的 LoRA 权重；或者外部系统会给每个请求导入一份独立的稠密增量。当 128 个请求被连续批处理（continuous batching）拍扁并拼成同一个二维张量 `[num_tokens, hidden_size]`，冲进同一层 MLP 的 `down_proj` 时，第 0 行需要乘张三的增量，第 1 行需要乘李四的增量，第 2 到 33 行又属于正在做预填充（prefill）的王五。

第二座冰山更陡峭：**在模型前向传播的半路上直接训练权重**。

以 ByteDance Seed 提出的 In-Place TTT（*In-Place Test-Time Training*）为例。传统的 Transformer 把上下文记忆全部塞进 KV cache，而 In-Place TTT 干了一件很酷的事：它不往模型里加新层，而是直接把选定层 MLP 的最后一个投影矩阵 `down_proj` 当作可写的“快权重（fast weights）”。模型一边读入 token，每攒够一个固定大小的块（比如 1024 个 token），就当场算一次外积更新：

$$\Delta W = \eta \cdot V^\top Z$$

其中 $Z$ 是进入 `down_proj` 前的中间激活，$V$ 是由 MLP 输入经过一层一维卷积与线性投影算出的目标。最要命的是它的可见性铁律：**在同一趟前向传播里，后一个块必须立刻读到前一个块刚刚更新完的权重**。

如果你去翻 In-Place TTT 的官方开源仓库，它基于 Hugging Face 的 `generate` 循环写成，而在推理入口处静静地躺着这样一行断言：

`assert len(input_ids) == 1, "only support bs=1 for TTT inference"`

我完全理解官方作者为什么要写这行 `assert`。当每个请求都有自己的一套 570 MiB 快权重、每个请求的更新边界落在不同的位置、有的请求在 prefill 里一口气跨过四个块、有的请求在 decode 里刚好吐出凑满第 1024 个位置的那个 token，还要同时兼顾 CUDA Graph 录制与多卡张量并行（Tensor Parallelism，TP）时——如果你让算法作者自己去写批处理切片、页表寻址、图捕获与跨卡同步，那篇论文的代码大概永远也写不完。

我们希望算法作者写出来的代码是什么样的？翻开 `pllm/workloads/inplace_ttt.py`，核心的 `step` 方法长这样：

```python
# check-docs: skip: excerpt from pllm/workloads/inplace_ttt.py
@pllm.train_in_forward(
    visibility="exact",
    state="dense_fast_weights",
    targets=targets,
    weight="delta",
    chunk=chunk_size,
    tokens="all",
)
def step(
    self,
    delta: torch.Tensor = pllm.parameter("dense_fast_weights", "delta"),
    inputs: torch.Tensor = pllm.operand("input"),
    targets: torch.Tensor = pllm.operand(target_source),
    base: torch.Tensor = pllm.operand("weight"),
    conv: torch.Tensor = pllm.parameter("target_parameters", "conv"),
    proj: torch.Tensor = pllm.parameter("target_parameters", "proj"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve inside this chunk, read the incoming weight, then sum the positive outer products."""
    values = (
        torch.nn.functional.conv1d(targets.T.unsqueeze(0), conv, padding=2, groups=targets.shape[1]).squeeze(0).T
        @ proj
    )
    update = values.T @ inputs if self.dtype is torch.bfloat16 else values.float().T @ inputs.float()
    # ……
    update = update * self.config.learning_rate
    # ……
    return delta + update, inputs @ delta.to(inputs.dtype).T
```

仔细看这个函数的参数签名。它接收的只是“**一个请求**的快权重增量 `delta`”、“**一个块**的输入 `inputs` 与目标 `targets`”，返回的也只是“更新后的 `delta`”和“这一块输入在旧权重下读出的修正量 `inputs @ delta.T`”。

这里没有 batch 维，没有请求 ID，没有分页池的槽号，甚至感知不到自己脚下是一张显卡还是四张显卡。它纯粹得就像从论文算法框图里直接拓印下来的数学公式。

怎么在不改动作者哪怕一行公式的前提下，让这个“单请求、单块”的玩具般的函数，无缝跑在 128 并发、开启 CUDA Graph、甚至跨越多张 GPU 的 vLLM 引擎里？

答案不是去手搓几千行专用的 C++/CUDA 内核。事实上，**为了实现整套参数覆盖层（Parameter Overlays），我们没有写哪怕一个专用算子内核**。我们所做的，是顺着 PyTorch 自己的抽象阶梯一层一层往下走。每当我们撞上一堵系统工程的墙，恰好都能在 PyTorch 现有的工具箱里找到一件趁手的兵器：

```text
Author's Code:  step(delta, inputs, targets)   <-- Pure single-request, single-chunk math
  │
  ├─► Layer 1: torch.func.vmap                 <-- Maps 1 request to N concurrent requests
  ├─► Layer 2: Bounded index_select + bmm      <-- Amortizes 128-request grouped projection reads
  ├─► Layer 3: torch.library.custom_op         <-- Crosses into vLLM's compiled CUDA Graph via table.data_ptr()
  ├─► Layer 4: grad ──► vmap ──► torch.compile <-- Generates per-chunk backward without mixing BF16 reductions
  └─► Layer 5: make_fx (Global vs. Shard)      <-- Proves TP>1 updates need zero extra cross-GPU communication
```

让我们从只有一个请求的世界出发，每次往里加一个真实世界的麻烦，看看这五层抽象是怎么严丝合缝地咬合在一起的。

> **实验环境说明**：除特别注明外，文中所有性能与显存数据均在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0、CUDA Graph 开启下测得。In-Place TTT 实验使用社区 4B 检查点 `zhongweixie/inplace-ttt-qwen3-4b-32k-ttt`（36 层，第 0、6、12、18、24、30 层的 `down_proj` 为 `[2560, 9728]` 快权重，块长 1024，学习率 3.0），贪心解码生成 256 个 token，1 轮预热后取 3 轮中位数，吞吐定义为“生成 token 总数 / 含 prefill 的整次 `generate` 耗时”；官方基线使用 `ByteDance-Seed/In-Place-TTT` 仓库基于 Hugging Face `generate`（BF16 + FlashAttention 2）在同一张 RTX 4090 上按相同协议测得。量化底座反量化数据测自 A100（80 GiB）上的 14B GPTQ 模型。

---

## 第一关：从 1 个请求到一批请求——让 `vmap` 替作者操心

只有一个请求的时候，什么魔法都不用施展：拿这个请求的权重，调作者写的函数。

但当线上同时跑着两个、十个、一百个请求时呢？每个请求都有自己独立的一份参数（比如在状态池里各自占着一行展平的向量 `flat`），而模型传过来的激活张量也是按行打包好的。如果我们要求作者在写代码时自己去对齐“第 $i$ 行激活乘第 $r(i)$ 号请求的第 $k$ 个参数矩阵”，那不仅枯燥，而且几乎是一切多租户串味 bug 的温床（见[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）。

怎么让一段只认识“单行、单份参数”的代码，自动学会处理一整批请求？

在函数式编程里，这个操作叫 `map`；在 PyTorch 2 里，它叫 `torch.func.vmap`。当作者在方法里调用一个绑定在请求状态上的 `nn.Module` 小网络时，框架在底层用 `ModuleCall`（`pllm/execution/module_graph.py`）悄悄给它包了一层外壳：

```python
# check-docs: skip: excerpt from pllm/execution/module_graph.py
def _run(self, module: torch.nn.Module, prefix: str, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    def call(flat: torch.Tensor, *call_args: torch.Tensor) -> torch.Tensor:
        parameters = {prefix + name: view for name, view in _views(self._module, flat)}
        return functional_call(
            module, parameters, tuple(arg.to(flat.dtype) if arg.is_floating_point() else arg for arg in call_args)
        )

    def row(flat: torch.Tensor, *row_args: torch.Tensor) -> torch.Tensor:
        # The module sees a batch of one row, as it would in ordinary PyTorch code.
        return call(flat, *(arg.unsqueeze(0) for arg in row_args)).squeeze(0)

    return vmap(row)(self._flat, *args) if self._per_row else call(self._flat, *args)
```

我特别喜欢 `row` 里面的那行注释：“模块看到的是由一行组成的 batch，和在普通 PyTorch 代码里一模一样。”

看它是怎么工作的：`_views` 把属于单个请求的那条扁平参数向量 `flat` 切分并重塑回模块各个参数的原始形状，然后用 `functional_call` 把无状态的算子逻辑和这组参数拼在一起。在 `row` 函数里，我们给输入 `arg` 轻轻加上一个长度为 1 的伪 batch 维（`unsqueeze(0)`），算完再把它捏掉（`squeeze(0)`）。最后，`vmap(row)` 在最外层大笔一挥，沿着请求维把所有算子自动向量化。

作者以为自己在写 `bs=1` 的玩具代码，而底层发出去的已经是合并后的批量算子。

---

## 第二关：128 个请求挂在同一个投影上——用分块 `bmm` 摊薄读取

`vmap` 解决了同构函数的批量映射。但在连续批处理的模型前向内部，只读覆盖层（或者固定读起始状态的 `visibility="forward_pinned"`）还会遇到一个更骨感的几何难题。

想象一下，vLLM 正在跑一步混合了 chunked prefill 和 decode 的前向：128 个请求共享同一个 `down_proj` 层，但每个请求挂载着不同的权重增量（或者各自驻留在 PTTS 状态池的不同物理槽位里，见[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)）。其中请求 A 正在做 prefill，一口气占了 80 行；请求 B、C、D 正在做 decode，每人只占 1 行。

这就是臭名昭著的“批内分组矩阵乘（Grouped GEMM）”。如果你用 `for` 循环给每个请求单独发射一次小矩阵乘，128 个请求就是 128 个微小的 kernel，CPU 发射开销会直接把 GPU 拖垮；如果你为每一行输入都去状态池里单独拷一份权重矩阵出来做 `bmm`（Batch Matrix Multiplication），那么请求 A 的 80 行明明用的是同一份权重，你却把那块权重矩阵在显存里重复搬运了 80 遍！

怎么在不写定制 CUDA 内核的前提下，既不让 kernel 数量爆炸，又不做愚蠢的重复搬运？

我们在 `pllm/execution/parameter_projection.py` 里用了一个极其朴素的“定长分块装车”策略：

```text
Packed Hidden [Rows, H]         Bounded Blocks [Blocks, Count, H]       Pinned Pools
┌────────────────────┐          ┌─────────────────────────┐             ┌──────────┐
│ Req A (rows 0..31) │ ───────► │ Block 0: 32 rows (Req A)│  ×  Slot 4 ─┤ Matrix A │
├────────────────────┤          ├─────────────────────────┤             ├──────────┤
│ Req A (rows 32..49)│ ───────► │ Block 1: 18 rows + pad  │  ×  Slot 4 ─┤    ...   │
├────────────────────┤          ├─────────────────────────┤             ├──────────┤
│ Req B (row 50)     │ ───────► │ Block 2:  1 row  + pad  │  ×  Slot 9 ─┤ Matrix B │
└────────────────────┘          └─────────────────────────┘             └──────────┘
                                             │
                                             ▼  Single batched torch.bmm
                                Scattered back to [Rows, Out] via owners mask
```

规则只有两句话：我们把属于同一个请求的连续行切分成固定宽度 `count` 的小块（`blocks`）。如果当前步全是单行 decode（`rows <= requests`），每块宽度就是 `1`；只要混入了多行 prefill，每块宽度就设为常数 `ROWS_PER_BLOCK = 32`。每一个块只需要从分页池里抽取**一份**权重矩阵（或一对低秩的 A/B 因子），然后所有块一起跑一次标准的 `torch.bmm`：

```python
# check-docs: skip: excerpt from pllm/execution/parameter_projection.py
def projection_read(
    hidden: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    owners: torch.Tensor,
    *pools: torch.Tensor,
    operator: ParameterOverlayOperator,
) -> tuple[torch.Tensor, ...]:
    """Gather bounded blocks; each block reads one pinned matrix or factor pair."""
    blocks, count = rows.shape
    inputs = hidden.index_select(0, rows.flatten()).view(blocks, count, hidden.shape[-1])
    weights = tuple(pool.index_select(0, slots[:, index]) for index, pool in enumerate(pools))
    result = parameter_correction(inputs, weights, operator, batched=True).flatten(0, 1)
    return (result.index_select(0, owners.clamp_min(0)).masked_fill((owners < 0)[:, None], 0),)
```

这段代码总共只有五句话，用的全是 PyTorch 现成的 `index_select`、`bmm` 和 `masked_fill`：
1. 用 `rows` 索引把打散的隐藏状态 `hidden` 捞出来，排成规整的三维张量 `[blocks, count, hidden_size]`；
2. 用 `slots` 索引从状态页池 `pools` 里为每个块捞出一份权重；
3. 调 `parameter_correction(..., batched=True)` 做一次批量矩阵乘，把 prefill 时读取一次权重的带宽成本平摊给块内的 32 个 token；
4. 用 `owners` 索引把算好的修正量贴回原始的行位置，顺手把凑块用的无效填充位置（`owners < 0`）涂成 `0`。

为了能录进静态形状的图里，块数会按 2 的幂次分档（`capacities`）预先热身。

当然，天下没有免费的午餐，我们得老老实实把账算清楚：**如果你在生产环境里唯一的需求就是同时服务成百上千个只读的 LoRA，我们这套 `index_select + bmm` 肯定不如专门的分组矩阵乘快。** 像 S-LoRA、Punica 以及 vLLM 自带的 multi-LoRA，底层都有手写的分组 GEMM 内核（比如基于 cutlass 或 Triton 的 SGMV / BGMV），不仅省掉了显式的权重 gather 拷贝，还能直接处理不同 rank 混排和 QKV 融合。如果你只需要部署静态只读的 LoRA，请直接用 vLLM 原生的 multi-LoRA。

我们之所以选择用纯 ATen 算子搭这套分块读取，看中的是它的**正交性与通用性**：同一套五行的 `projection_read`，不改一字就能同时通吃稠密矩阵增量、低秩因子对和按注意力头的增量；更重要的是，它直接读取 PTTS 的版本化页池——这意味着它读到的权重可以是在线训练出来的、带版本的、甚至在生成中途随时更新的。

---

## 第三关：钻进 CUDA Graph——再用一次“地址即身份”

现在，我们的投影读取要把手伸进 decode 阶段了。

在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里我们已经领教过 CUDA Graph 的厉害：为了省掉 CPU 逐个发射 kernel 的微秒级开销，vLLM 会把整个模型的前向过程录制成一张静态图。录制一旦结束，Python 解释器就退场了；每次回放时，GPU 只会机械地按录制时的固定内存地址去读数据。

这给我们出了个难题：在每一步 decode 里，第 0 行可能换成了另一个请求，或者同一个请求的快权重刚刚升到了新版本（槽号从 `Slot 2` 变成了 `Slot 5`）。既然回放时不能向 CPU 发问，GPU 怎么知道当前步去哪个槽位拿权重？

答案依然是那张贯穿全系列的“固定地址 `int64` 小表”。在每一步发射图之前，CPU 把当前步算好的块槽号 `slots`、行号 `rows` 和归属映射 `owners` 统统打包写进一张地址永远不变的设备张量 `table` 里。

但要想让 vLLM 的编译器（Dynamo / Inductor）乖乖把我们的读取逻辑录进整模型图，还得跨过一道关卡：我们必须把它包装成一个不透明的自定义算子 `@torch.library.custom_op`。

这就撞上了我们在[《图内挂载点》](05-in-graph-attach-points.md)里遇到过的同一个窘境：PyTorch 的 `custom_op` 契约极其严格，它的参数列表只允许传张量（`Tensor`）和基本标量（`int`/`float`），绝不允许传自定义的 Python 对象（比如管理这层投影状态的 `ProjectionReader` 实例）。你可能会想：“那我给每个 `ProjectionReader` 分配一个整数编号 `reader_id`，当成标量传进去去全局字典里查不就行了？”

不行。因为 vLLM 的编译图（包括 AOT 编译缓存）会把传入的标量参数当成编译期常数直接烘焙（bake）进图里！如果你传整数 ID，编译器就会把它焊死在图里，在不同会话之间张冠李戴。

怎么只靠传张量，就能在算子内部认出“我是哪一层的哪个 `ProjectionReader`”？我们在 `pllm/backends/vllm/parameter_overlay.py` 里重施故技——**拿那张索引表张量 `table` 的显存首地址当身份证**：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/parameter_overlay.py
@torch.library.custom_op("pllm::pinned_projection", mutates_args={"output", "captured", "desired_capture", "pools"})
def pinned_projection(
    hidden: torch.Tensor,
    desired: torch.Tensor,
    pools: list[torch.Tensor],
    table: torch.Tensor,
    output: torch.Tensor,
    captured: torch.Tensor,
    desired_capture: torch.Tensor,
) -> None:
    # ……
    _READERS[table.data_ptr()].emit(hidden, desired, pools, table, output, captured, desired_capture)
```

每一个 `ProjectionReader` 在初始化时都会分配一张自己专属的 `table` 张量，并在关闭（`close()`）之前一直死死攥着它的所有权，绝不让 PyTorch 的显存分配器把这个地址回收复作他用。然后，它把 `self.table.data_ptr()`——也就是这块显存的 64 位整数地址——作为键，把自己注册进全局字典 `_READERS`。

当 `pinned_projection` 被调用时，对 Dynamo 编译器来说，`table` 只是一个再普通不过的运行时输入张量，图结构干干净净，没有任何标量特化；而在算子内部，`table.data_ptr()` 瞬间就能在 `_READERS` 里找回对应的 `ProjectionReader` 实例，发射出刚刚那套 `projection_read` 算子序列。

---

## 第四关：在前向里边走边训——`grad`、`vmap` 与 `compile` 的洋葱顺序

到目前为止，权重还只是被读取。现在我们要揭开最刺激的部分：**在前向传播的过程中直接训练这些权重。**

在 pLLM 里，前向内训练有两种写法。第一种像文章开头的 In-Place TTT，作者直接手写状态转移公式 `return delta + update, readout`；第二种更省心，作者连更新公式都不用推，只需要给状态配一个优化器（在 `RequestState` 上写 `optimizer=pllm.SGD(self.learning_rate)`），然后在 `@pllm.train_in_forward` 修饰的方法里只写出**每一行的损失函数（loss）**。

来看看 `pllm/workloads/prefill_adapters.py` 里的 `PrefillAdaptersWorkload`。它在阅读长 prompt 的过程中，每隔 128 个 token（一个 chunk）就在第 0、2、4、6、8 层的 `down_proj` 上实时训练一对低秩适配器 `a` 和 `b`，目标是让这层输出尽可能重构出指定目标 `target`，并在 prompt 读完时立刻把尾部不满一块的梯度也刷进去（`flush="prompt_end"`）：

```python
# check-docs: skip: excerpt from pllm/workloads/prefill_adapters.py
@pllm.train_in_forward(
    state="adapters",
    targets=TARGETS,
    operator="low_rank",
    visibility="exact",
    chunk=chunk,
    flush="prompt_end",
)
def learn(
    self,
    x: torch.Tensor = pllm.operand("input"),
    target: torch.Tensor = pllm.operand("target"),
    base: torch.Tensor = pllm.operand("weight"),
    a: torch.Tensor = pllm.parameter("adapters", "a"),
    b: torch.Tensor = pllm.parameter("adapters", "b"),
) -> torch.Tensor:
    """Return each row's reconstruction error with FP32 (or wider) error math."""
    prediction = x @ base.T + (x @ a.to(x.dtype).T) @ b.to(x.dtype).T
    dtype = torch.promote_types(prediction.dtype, torch.float32)
    return (prediction.to(dtype) - target.to(dtype)).square().mean(-1)
```

看，作者只写了三行求均方误差的前向数学。那反向求导呢？多请求批处理呢？算子融合编译呢？

全都在 `pllm/execution/overlay_task.py` 的 `overlay_gradient` 里，由三层 PyTorch 变换像剥洋葱一样套起来完成：

```python
# check-docs: skip: excerpt from pllm/execution/overlay_task.py
def overlay_gradient(function: Callable[..., torch.Tensor], *, batched: bool = False) -> Gradient:
    """Compose request mapping with differentiation before compiling; broadcast frozen weights."""

    def loss(parameters: Tensors, operands: Tensors, constants: Tensors, valid: torch.Tensor) -> torch.Tensor:
        rows = function(**parameters, **operands, **constants)
        # ……
        return torch.where(valid, rows.to(torch.promote_types(rows.dtype, torch.float32)), 0).sum()

    eager = torch.func.grad(loss)
    if batched:
        # Folding requests into the frozen-base GEMM changes CUDA's BF16 reduction.
        # Compile the chunks together while retaining each source GEMM's shape.
        eager = torch.vmap(eager, in_dims=(0, 0, None, 0), chunk_size=1)
    compiled = torch.compile(
        eager, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )
```

盯着这不到二十行代码看一会儿，里面藏着两个只有真正在 GPU 上踩过坑才会明白的细节。

第一个细节是**洋葱的顺序**：为什么必须是先 `torch.func.grad`，再 `torch.vmap`，最后 `torch.compile`？
因为 `torch.func.grad` 要求被求导的函数输出一个**标量（scalar）**。在单个请求内部，我们用 `valid` 掩码把属于当前块的有效行挑出来求和（`sum()`），变成一个标量 loss，这样 `grad(loss)` 就能干净地算出这一个请求的参数梯度。接着，外层的 `vmap` 把这个“单请求求梯度”的函数沿着请求维铺开，并且通过 `in_dims=(0, 0, None, 0)` 告诉 PyTorch：参数、输入操作数和有效掩码是每个请求各有一份的（沿第 0 维切开），唯独第三个参数 `constants`——也就是那块冻结的底座权重 `base`——是所有人共享的（`None`，直接广播，绝不复制！）。最后，最外层的 `torch.compile` 把展开后的整套前向 + AOTAutograd 反向图一口气编译融合。

第二个细节更加微妙，就在 `torch.vmap(..., chunk_size=1)` 那行注释里：**既然用了 `vmap`，为什么还要加一个看似自废武功的 `chunk_size=1`？**

我们在做逐请求与批处理的数值对齐测试时，曾抓到一个诡异的幽灵：在 `learn` 函数里有一句 `x @ base.T`，用来算冻结底座权重的输出。如果让 `vmap` 放开手脚把 $B$ 个请求的输入全折进同一次矩阵乘里，输入形状就会从 `[C, H]` 变成 `[B * C, H]`。按理说，矩阵乘的每一行是独立的，拼在一起算和分开算应该一模一样对吧？

在数学上是的，但在 GPU 的 cuBLAS 库里不是！当矩阵的 M 维（行数）从 `C` 变成 `B * C` 时，cuBLAS 的启发式算法会挑选另一个分块（tile）尺寸不同的内核，导致内部 BF16 乘累加的归约顺序发生细微改变。于是，同一个请求在单跑时和跟别人拼批跑时，算出来的梯度在最后几个比特上出现了漂移！加上 `chunk_size=1` 之后，`vmap` 在面对共享底座 `base` 的 GEMM 时会保持每个请求原本的 `[C, H]` 形状，而在外层依然被 `torch.compile` 放在同一个编译图里一起优化——既保住了逐请求比特级一致的铁律，又没丢掉编译融合。

### 一段插曲：那个被我们留在设计稿里的“最小割”方案

聊到这里，我还想跟你坦白一个我们在设计“跨步攒块训练”时放弃掉的漂亮点子。

在连续批处理里，一个块（比如 128 或 1024 个 token）往往不是在同一步里到齐的。可能第 1 步分块 prefill 来了 64 个 token，第 2 步又来了 64 个 token，到了 decode 阶段更是每步只蹦出 1 个 token。只有当攒满一整块（或者遇到 `flush="prompt_end"`）时，优化器才能真正迈出一步，更新权重。

当初在设计这个机制时，我们脑子里蹦出了一个极具编译器美感的方案：既然一块数据分好几步才到齐，那我们能不能把这几步看成**“一次被拉长到多步的前向 + 反向”**？

具体来说，PyTorch 的 AOTAutograd 内部有一个非常聪明的最小割重计算分割器（`min_cut_rematerialization_partition`）。当你把前向和反向的联合计算图交给它时，它会自动求一个最小割，决定前向算出来的哪些中间激活值得保存下来传给反向，哪些激活太占显存、不如留到反向时重算。我们当时设想：让这个最小割分割器来决定每一步到达的行需要保存哪些中间张量，而 pLLM 只负责把这些切口上的张量存进 PTTS 的状态页里，等攒满一块的那一步再统一跑反向图！

这个方案写进设计文档时，我们自己都觉得太酷了：“分割器决定存什么，PTTS 决定存在哪、存多久。”

但冷静下来一算账，我们决定**不做它**。

为什么？看看块内训练的数学结构就知道了：在一个块还没攒满之前，请求的快权重是**固定不变**的。既然权重在块内不变，那么“把每步到达的行存起来、等凑满一整块再一起求梯度”，和“**每步对刚刚到达的这几行直接求梯度，把算出来的梯度累加进 PTTS 的 `_pending_grad_*` 字段里，等攒满一块时直接拿累加好的总梯度做一次优化器更新**”，在数学上是 100% 严格等价的！

后者有一个大家耳熟能详的名字，叫**梯度累加（Gradient Accumulation）**。

如果走跨步最小割分割器的路线，我们需要动用 PyTorch 的私有内部 API，在代码库里多维护两三百条语句的图切分与跨步张量生命周期胶水；而走梯度累加路线，AOTAutograd 的最小割分割器依然在为我们工作，只不过它只需要安静地待在**每一步自己的 `torch.compile` 内部**决定保存还是重算，跨步留存的只有形状永远固定、跟参数一模一样的梯度张量。

“如无必要，勿增实体。”先用最简单且数学等价的做法把问题解决掉，把那个华丽的跨步分割器留在设计稿里——直到某天真的出现一个块内行与行之间在激活层互相耦合、无法拆成梯度累加的算法为止。

---

## 第五关：张量并行（TP > 1）——用 `make_fx` 双图同构当证明器

现在，让我们把难度调到最高档：如果模型太大，单张显卡装不下，我们用 `tensor_parallel_size=2` 或 `4` 把模型切开放在多张 GPU 上跑，覆盖层还能工作吗？

先看只读（或者读取当前生效权重）的那一半。以 MLP 的 `down_proj` 为例，在 vLLM 里它是一个行并行线性层（`RowParallelLinear`）：如果输入特征总宽度是 $H$，那么在 TP=2 下，0 号卡只持有左半边输入 $x_0$ 和左半边权重 $W_0$，1 号卡只持有右半边输入 $x_1$ 和右半边权重 $W_1$。每张卡各自算出部分和 $y_j = x_j W_j^\top$，然后 vLLM 调用一次跨卡的 `all-reduce` 把它们加起来，得到完整输出 $y = \sum_j x_j W_j^\top = x W^\top$。

这时候如果要加上请求专属的稠密增量 $\Delta W$（或者低秩增量 $B A$），我们需要额外加一次跨卡通信吗？

完全不需要！因为矩阵乘法在线性代数上对被收缩的特征维是天然可加的：

$$x (W + \Delta W)^\top = \sum_j x_j (W_j + \Delta W_j)^\top = \sum_j \left( x_j W_j^\top + x_j \Delta W_j^\top \right)$$

只要我们也把稠密增量 $\Delta W$（或者低秩因子里的输入侧矩阵 $A$）按输入特征维切成两半，让每张卡在 vLLM 发起那次 `all-reduce` **之前**，把自己算出来的修正量加到本卡的部分和上，修正量就能白搭一趟上游原有的 `all-reduce` 顺风车！全过程零次额外通信。

为了搭上这趟顺风车，`pllm/backends/vllm/parameter_overlay.py` 里的 `ProjectionHook` 做了一个非常精细的拦截：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/parameter_overlay.py
class ProjectionHook:
    # ……
    def __init__(self, module: LinearBase) -> None:
        self.module = module
        self.forward = module.forward
        self.apply = module.quant_method.apply if _row_parallel(module) else None
        # ……

    def base(self, hidden: torch.Tensor) -> LinearOutput:
        if self.apply is None:
            return self.forward(hidden)
        # Accumulating the correction in BF16 before the reduction adds a rounding
        # on every rank. Keep the partial sum wide and round once after vLLM sums it.
        return self.apply(self.module, hidden, self.bias).float(), None
```

在单卡（`TP=1`）时，它只是个普通的 `forward` 钩子；而在多卡行并行（`_row_parallel`）时，它真正伸手拦住的是里面的 `quant_method.apply`——也就是本卡刚算完局部 GEMM、还没进 `all-reduce` 的那一刻（外层 `forward` 只换成一层在求和之后做舍入的薄壳 `reduced`）。更讲究的是 `base` 里的那个 `.float()`：如果每张卡先把修正量加进 BF16 的部分和里再去 `all-reduce`，每张卡都会多引入一次 BF16 舍入误差；所以我们让局部和加上修正量之后保持宽裕的 FP32 穿过 vLLM 的 `all-reduce`，等所有卡加完了，最后再在 `reduced` 里做唯一一次向 BF16 的舍入。

读取搞定了，但真正让人头皮发麻的问题来了：**如果这个覆盖层包含在线更新（`@pllm.train_in_forward`），每张卡只拿着自己那一半输入特征，各自跑一遍作者写的更新函数，算出来的新权重还对吗？**

这完全取决于作者在更新函数里写了什么！

让我们对比刚刚见过的两个例子：
*   **例子 A（In-Place TTT 的外积更新）**：`update = values.T @ inputs`。这里目标 `values`（形状 `[C, Out]`）在每张卡上是完整复制的，而输入 `inputs`（形状 `[C, H/2]`）是按特征维切开的。矩阵乘收缩的是长度为 `C` 的块内行维度，特征维 `H/2` 只参与外积展开，互不干扰。0 号卡算左半边权重更新，1 号卡算右半边权重更新，两边拼起来和单卡算出来的分毫不差！
*   **例子 B（PrefillAdapters 的重构误差求导）**：`prediction = x @ base.T + (x @ a.T) @ b.T`，然后对 `(prediction - target).square()` 求导。注意看第一项：`x` 是切片的 `[C, H/2]`，`base` 也是切片的 `[Out, H/2]`，乘完之后得到了本卡的局部预测值。但要想算出真正的重构误差去更新低秩因子，你必须先把所有卡的局部预测值加起来（跨过被切开的 `H` 维求和），才能知道完整的 `prediction` 偏离了 `target` 多少！如果每张卡闷头只用自己那半截输入算误差求梯度，算出来的权重就是彻底错乱的，而且不会报任何运行时错误，只会让模型悄悄吐出胡言乱语。

我们绝不能允许系统悄悄算错。但怎么在加载模型之前，自动判断一段任意的 PyTorch 更新函数是不是“分片本地（shard-local）”的？

按照传统编译器的套路，你得写一部庞大的“张量分片传播静态分析器（Sharding Propagation Analyzer）”，为 PyTorch 成百上千个算子逐一编写分片维度推导规则。那将是一个几万行代码的大工程，而且永远有漏网之鱼。

我们在 `pllm/execution/overlay_parallel.py` 里想出了一个偷懒得多、却异常锋利的招数：**既然 PyTorch 的 `make_fx` 可以用不占显存的假张量（Meta / FakeTensor）把函数追踪成一张计算图，那我们干脆把作者的更新函数追踪两遍——第一遍喂给它单卡的完整形状（比如特征宽 $H$），第二遍喂给它切了一半的分片形状（特征宽 $H/2$）——然后把两张图并排摆在桌子上“找不同”！**

```text
Case 1: In-Place TTT Outer Product (ADMITTED)
  Global Trace (width = H)              Shard Trace (width = H/2)
  inputs [C, H]   values [C, Out]       inputs [C, H/2] (sharded axis=1)  values [C, Out]
         └───┬───┘                             └─────┬─────┘
             ▼ aten.mm                               ▼ aten.mm
       update [Out, H]                         update [Out, H/2]  <-- Axis 1 still sharded! OK!

Case 2: Reconstruction Error / Feature Norm (REJECTED)
  Global Trace (width = H)              Shard Trace (width = H/2)
  inputs [C, H]   base.T [H, Out]       inputs [C, H/2] (sharded axis=1)  base.T [H/2, Out]
         └───┬───┘                             └─────┬─────┘
             ▼ aten.mm                               ▼ aten.mm
       pred   [C, Out]                         pred   [C, Out]    <-- Sharded axis vanished! REJECT!
```

看上面这张对比图，真相是不是一目了然？

我们甚至不需要教分析器什么是外积、什么是内积。只要看每个节点输出张量在两张图里的形状对比就够了！在 `pllm/execution/overlay_parallel.py` 里，判定两个张量之间是否存在合法分片轴的函数 `_axis` 短得让人发笑：

```python
# check-docs: skip: excerpt from pllm/execution/overlay_parallel.py
def _axis(whole: torch.Tensor, local: torch.Tensor, size: int) -> int | None:
    axes = tuple(i for i, (a, b) in enumerate(zip(whole.shape, local.shape, strict=True)) if a != b)
    if not axes:
        return None
    if len(axes) != 1 or whole.shape[axes[0]] != local.shape[axes[0]] * size:
        raise ValueError("TP updates must preserve one contiguous feature shard")
    return axes[0]
```

拿着全局图 `whole` 和分片图 `local` 里对应的两个输出张量，比对它们的每一维：
*   如果所有维度都一样长（`not axes`），说明这个张量是全卡复制的（返回 `None`）；
*   如果有且仅有一个维度不同，且全局长度恰好等于分片长度乘以 TP 卡数 `size`，说明这个张量在该轴上是均匀切分的（返回分片轴下标 `axes[0]`）；
*   其他任何情况，当场判出局。

有了 `_axis`，`admit_update` 和 `_check_graphs` 的核心证明逻辑就水到渠成了：

```python
# check-docs: skip: excerpt from pllm/execution/overlay_parallel.py
def _check_graphs(whole: tuple[Node, ...], local: tuple[Node, ...], size: int) -> None:
    if len(whole) != len(local):
        raise _reject("shape-dependent control flow")
    axes: dict[Node, int | None] = {}
    # ……
    for a, b in zip(whole, local, strict=True):
        # ……
        axis = _axis(value, part, size)
        axes[a] = axis
        # ……
        inputs = tuple(node for node in a.all_input_nodes if axes.get(node) is not None)
        if not inputs and axis is None:
            # ……
            continue
        # ……
        if axis is None:
            raise _reject(label)
```

我们用 `make_fx(update, tracing_mode="fake")` 追出全局和分片两张图，先调 `eliminate_dead_code()` 剪掉只用于当前步读出、不影响新权重的旁支节点，然后把两张图的存活节点一对一拉链式（`zip`）对齐：
1. 如果两张图的节点数不一样，或者对应节点的算子对不上，说明代码里写了依赖张量宽度的 `if/else` 控制流——直接拒绝；
2. 如果里面出现了张量字面量常量（`get_attr`）或带随机种子的算子（`nondeterministic_seeded`），各卡可能生成不一致的随机数——直接拒绝；
3. **最核心的黄金铁律就在最后几行**：我们先找出当前节点所有带分片轴的输入 `inputs`。只要一个节点吃进了哪怕一个分片输入（`inputs` 非空），它的输出就**绝不能变成未分片状态**（`if axis is None: raise _reject(label)`）！

一旦像例子 B 那样，在分片轴上做了矩阵乘收缩或 `sum`/`mean` 归约，分片图上的输出形状就会变得和全局图一模一样，`axis` 瞬间变成 `None`，立刻触发 `_reject`，在模型加载前报错拦下：

`TP projection update is not shard-local at aten.mm.default; its update/gradient needs other ranks' shards or an unproved operation. Use a shard-local update or tensor_parallel_size=1`

这就是全篇我最偏爱的一段代码。总共两百行出头，它把一个看似需要深度程序分析的编译难题，巧妙地转化成了“用两种假形状跑出来的两张 FX 图是否同构、分片标记是否守恒”。虽然这套白名单很保守（目前在 `TP > 1` 下只放行通过该检验的前向内稠密或低秩更新；旁路训练的适配器、外部导入或带状态的通用模块适配器、以及读取共享可变 `InstanceState` 的适配器仍限制在 `TP = 1`），但凡是拿到准入通行证的函数，在多卡运行时都**不需要插入任何运行时检查或额外集合通信**。我们在 TP=2 下用 FP64 的 GPU 基准（oracle）做过严格验算：覆盖层在双卡上的均方根（RMS）误差不超过原生 TP=2 自身浮点误差的 105%，最大绝对误差不超过 1 个 BF16 ULP（最小精度单位）。

---

## 把 In-Place TTT 放上跑道：跨块可见性、吞吐与被吞掉的尾数

五层抽象全部就位。现在，让我们把开头那个社区训练的 4B In-Place TTT 检查点加载进来，看看这套机器在真实硬件上是怎么运转的。

在真正把 8 GB 的模型权重搬进显存之前，我们可以先在 CPU 上花零点几秒调一次 `pllm.explain`，给显存算一笔明白账。在这个 36 层的 4B 模型里，每隔 6 层（第 0、6、12、18、24、30 层）有一个形状为 `[2560, 9728]` 的 `down_proj` 快权重矩阵，6 层加起来一共是 `149,422,080` 个参数。按 FP32 存放，光这 6 个 `delta` 矩阵就要占约 570 MiB；再加上暂存未满 1024 块输入行的缓冲，每个请求的一份状态根（root）重达 **`714.0 MiB`**（而 6 层共享的卷积与投影参数 `target_parameters` 只占 `75.1 MiB`）。

因为 PTTS 采用不可变版本与写时复制，即使只跑 1 个请求，为了在跨块更新的那一刻同时容纳正在被读取的旧版本和正在写入的新版本，系统也需要预留 4 份根（`714.0 MiB × 4`）；当并发开到 4 个请求时，连同在途版本一共需要 10 份根。在 24 GiB 的 RTX 4090 上，我们需要把 `gpu_memory_utilization` 设为 `0.48`，给快权重池留出足够的呼吸空间。

### 亲眼看见第 1024 个 token 跨过边界

在测速度之前，我们先做个有趣的显微镜实验，验证一下 `visibility="exact"` 到底有没有兑现承诺：**当快权重的更新点恰好落在逐字生成的 decode 半路上时，究竟哪一个 token 读旧权重，哪一个 token 开始读新权重？**

我们拿每台机器上都有的 Python 标准库 `argparse.py` 源码当输入，故意截取前 **1023 个 token** 作为 prompt——比块长 1024 刚好少 1 个 token！然后让模型贪心解码连续生成 5 个 token。我们同时跑两组对比：实验组（`TTT`）使用正常的 `chunk_size=1024`；对照组（`不更新`）把 `chunk_size` 改成 `2048`（因为整条序列加起来只有 $1023 + 5 = 1028$ 个 token，永远凑不满 2048，所以对照组全程更新 0 次）。

看看两组在每一步生成 token 时的对数概率（logprob）：

| 生成序号 | 生成的 token | 算出它的位置下标 | `TTT` 的 logprob | `不更新` 的 logprob | 差值 | 发生了什么？ |
|---:|---|---:|---:|---:|---:|---|
| 1 | `' _'` | 1022 | `-0.4400` | `-0.4400` | `+0.0000` | 第 1023 个 token，未满一块，读初始权重 |
| 2 | `'get'` | 1023 | `-1.9058` | `-1.9058` | `+0.0000` | **第 1024 个 token，凑满第一块并触发更新，但它自己仍读旧权重！** |
| 3 | `'_default'` | 1024 | `-2.2565` | `-2.3191` | `+0.0626` | **第二块的第 1 个 token，立刻读到新权重！** |
| 4 | `'_action'` | 1025 | `-1.8538` | `-1.8214` | `-0.0324` | 继续读更新后的新权重 |
| 5 | `'(action'` | 1026 | `-1.7855` | `-1.7124` | `-0.0732` | 继续读更新后的新权重 |

盯着第 2 行和第 3 行的分界线看：位置 `1023`（全序列的第 1024 个 token）进来的那一刻，第一块正好凑满，6 个快权重层各自触发了一次外积更新（`update_count = 6`）。但是按照因果序，第 1024 个 token 自己前向传播时看到的仍然是更新前的旧权重，所以它算出的 logprob 和对照组分毫不差（`+0.0000`）。紧接着下一步，位置 `1024`（第二块的头一个 token）进来时，新版本的 PTTS 状态根已经原子切换完毕，logprob 立刻跳变了 `+0.0626`！

而且，这一切都是在开启了 CUDA Graph 的 decode 循环里自动发生的（单独计时纯 decode 阶段，把 `exact` 更新录进 CUDA Graph 让单请求吞吐从 eager 的 `57.59` 提升到 `59.51 tok/s`，4 请求从 `107.56` 提升到 `113.16 tok/s`）。

### 与官方实现的吞吐对比，以及为什么我们默认用 FP32

现在来看整次生成（含 prefill）的端到端吞吐对比：

| 实现与快权重精度 | 1 个请求，prompt 4096 | 1 个请求，prompt 16384 | 4 个请求，prompt 4096 |
|---|---:|---:|---:|
| 官方 HF 实现（BF16） | `54.79 tok/s` | `34.89 tok/s` | 不支持（仅限 `bs=1`） |
| pLLM，BF16 快权重 | `57.12 tok/s` | `40.64 tok/s` | 未测 |
| pLLM，FP32 快权重（默认） | `59.64 tok/s` | `42.27 tok/s` | `111.38 tok/s`（4 请求合计） |

先看同精度的公平对比（第二行 vs 第一行）：在完全复现官方 BF16 快权重舍入行为的前提下，pLLM 在 4k 长度下比官方快约 4%，在 16k 长度下快约 16%。而当并发增加到 4 个请求时，官方实现因为 `bs=1` 的硬限制无法运行，pLLM 的合计吞吐则达到了 **`111.38 tok/s`**，约为单请求的 1.9 倍。（顺便提一句诚实的代价：因为块与块之间存在严格的串行依赖，`visibility="exact"` 的 prefill 阶段必须以 eager 模式按块顺序推进；如果你的线上负载几乎全是超长 prompt 而只生成几个 token，这条路径并不会创造速度奇迹。此外，一旦改写了模型权重，相同前缀算出的 KV 就不再相同，因此覆盖层天然与 prefix caching 互斥；且使用了 `exact` 图捕获的引擎在关闭时暂不能归还全部显存，后续引擎需要在新子进程中启动。）

但看着上面那张表，你一定注意到了一个反直觉的现象：**为什么用 32 位浮点数（FP32）存快权重，跑得反而比 16 位的 BF16 还要快？而且为什么 pLLM 要把 FP32 设成默认模式？**

这就要掀开我们在开篇省略掉的那几行 `inplace_ttt.py` 代码了：

```python
# check-docs: skip: excerpt from pllm/workloads/inplace_ttt.py
if self.dtype is torch.bfloat16:
    weight = (base.float() + delta).to(torch.bfloat16)
    output = (inputs @ weight.T).float() - (inputs @ base.T).float()
    return (weight + update).float() - base.float(), output
return delta + update, inputs @ delta.to(inputs.dtype).T
```

原来，官方 Hugging Face 代码在加载模型时，是直接把 BF16 的 `down_proj` 权重矩阵复制了一份当作快权重，每次算完 $\Delta W$ 就直接往这个 BF16 矩阵上硬加（$W \leftarrow \text{BF16}(W + \Delta W)$），身边并没有保留一份 FP32 的主副本（master weight）。

在数值分析里，把一个微小的增量直接加到一个数值较大的低精度浮点数上，是一场灾难——**大数吃小数（swamping）**。

BF16 只有 7 位尾数（精度大约是 $2^{-7} \approx 0.78\%$）。我们专门拿这个检查点第 0 层的真实 `down_proj` 权重矩阵（一共 `24,903,680` 个元素）做过一次算术实验：在论文的正文评测设置（Track T2）里，每一次块的更新矩阵在乘学习率前会被范数裁剪到 `1e-5`。如果我们真的拿一个范数为 `1e-5` 的更新矩阵，直接用 BF16 加到第 0 层的权重上，猜猜会发生什么？

在全部 **`24,903,680`** 个元素里，只有可怜的 **`472`** 个元素的值发生了改变！其余 99.998% 的元素加完之后，因为增量跨不过 BF16 的最小精度台阶，直接被硬件舍入回了原值，整次更新实际生效的改变量只有预期值的约 **`0.97%`**！（当然，我们手头这个社区检查点遵循的是官方仓库默认的不裁剪设置，且学习率高达 3.0，单步更新比 `1e-5` 大得多；这只是一个展示浮点尾数效应的数值实验，而非模型生成质量评测。）

一旦我们把底座权重 `W` 和快权重增量 `delta` 解耦开来——底座 `W` 依然安静地待在 BF16（甚至是 GPTQ/Marlin 量化格式！在 A100 上跑 14B GPTQ 模型时，我们只需在反向求导读底座那一刻做一次瞬态反量化，用完即焚，单步训练耗时从 `1.117 s` 降到 `0.193 s` 且不涨一分钱峰值显存），而每个请求单独用 FP32 存放从零开始累积的增量 `delta`——那只“吃掉小更新的怪兽”就消失了。哪怕单次更新再微小，也能在 FP32 的 `delta` 里稳稳当当地积少成多。而且在前向计算时，FP32 模式只需要额外算一次 `inputs @ delta.to(inputs.dtype).T` 加到底座输出上；反倒是严格复现官方 BF16 舍入的兼容模式，为了在不污染共享底座的前提下模拟“先合成 BF16 权重再相乘”的舍入误差，不得不做两次矩阵乘再相减。这就是为什么 FP32 模式既保住了微小的梯度信号，跑起来反而还更轻快。

---

## 回头看那五级台阶

合上 `parameter_overlay` 的源码，最让我感到踏实的不是我们写了多少精妙的代码，而是我们**少写了多少代码**。

当面对“多租户隔离、分组读取、进 CUDA Graph、前向内分块求导、多卡零通信证明”这五道连环关卡时，人们很容易产生一种撸起袖子从零发明一套深度学习编译器的冲动。但实际上，PyTorch 社区这些年已经把“批处理向量化（`vmap`）”、“函数式自动微分（`grad`）”、“算子图捕获与融合（`compile`）”、“向编译器声明不透明边界（`custom_op`）”以及“符号化追踪计算图（`make_fx`）”分别锤炼成了坚固而正交的原语。

系统工程师真正的活儿，不是去重造这五块砖，而是看清每一堵物理约束背后的数学本质，把这五块砖按唯一正确的顺序（比如先 `grad`、再 `vmap(chunk_size=1)`、最后 `compile`）垒在一起；并且在诱惑面前保持克制——当普通的梯度累加在数学上已经等价时，微笑着把那个需要几百行私有 API 的跨步最小割分割器留在草稿纸上；当你需要证明一段任意代码在多卡下不会跨分片串味时，别急着去写几万行的静态分析器，试试拿两个不同宽度的假张量把代码跑两遍，让两张计算图自己开口说话。
