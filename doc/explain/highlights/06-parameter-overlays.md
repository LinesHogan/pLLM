# 参数覆盖层：顺着 PyTorch 的抽象层往下走

在单请求的 PyTorch 研究脚本里，修改一个线性层的权重是再直接不过的事。如果要给权重矩阵 $W$ 加上一个稠密增量 $\Delta W$，只需写下 `y = x @ (W + delta).T`；如果要挂接一对低秩适配器（LoRA），也不过是多写一次乘法：`y = x @ W.T + (x @ A.T) @ B.T`。研究代码的舒服之处在于，它默认世界上只有一个请求，张量形状规整，权重就放在显存里，随时可以读取或原地修改。

但一旦把这个需求放进生产级推理引擎（如 vLLM），问题就立刻变得棘手起来。

首先是**多租户并发下的异构读取**。在像 Text-to-LoRA 这类按输入实时生成适配器的算法里，或者当外部系统为不同会话导入各自的稠密权重增量时，同时在线的 128 个请求拥有 128 份彼此不同的参数。连续批处理（continuous batching）会把这 128 个请求的输入拍扁并首尾相接，拼成同一张二维张量 `[num_tokens, hidden_size]` 送进模型的线性层（见[《行模型》](04-packed-rows.md)）。当这张张量流经某一层 MLP 的 `down_proj` 时，第 0 行需要乘请求 A 的增量，第 1 行需要乘请求 B 的增量，而第 2 到 33 行可能属于正在做预填充（prefill）的请求 C。

更进一步的挑战是**在前向传播的同时在线更新权重**。以 In-Place TTT（*In-Place Test-Time Training*）为例，它没有往 Transformer 里引入新的序列混合层，而是直接把选定层 MLP 的下投影矩阵 `down_proj` 当作每个请求私有的“快权重（fast weights）”。模型在消费上下文的同时，每攒够一个固定长度的块（例如 `chunk_size = 1024` 个 token），就根据当前块的中间激活 $Z$ 与目标投影 $V$ 计算一次外积更新：

$$\Delta W \leftarrow \Delta W + \eta \cdot V^\top Z$$

这类算法对执行次序有一条硬性要求：**在同一趟前向传播中，后一个块必须立即读到前一个块刚刚更新完的权重**。

去翻看 In-Place TTT 的官方开源实现（基于 Hugging Face 的 `generate` 循环），会在推理入口处看到这样一行断言：

`assert len(input_ids) == 1, "only support bs=1 for TTT inference"`

这行断言非常诚实地反映了算法研究与服务系统之间的鸿沟。当每个请求各自维护着几百兆的快权重、各个请求的块边界落在不同位置、同一个批次里既有跨越多块的长 prefill 又有单步推进的 decode，并且还得兼容 CUDA Graph 静态录制与多卡张量并行（Tensor Parallelism，TP）时，如果让算法作者用裸 PyTorch 手写批处理切片、页表寻址、反向求导和跨卡同步，几行就能写完的数学更新公式会膨胀成数千行脆弱的工程代码。

在 pLLM 中，算法作者实际编写的 In-Place TTT 核心更新函数 `step` 是这样的：

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

观察 `step` 的签名：它接收的是**单个请求**的快权重增量 `delta`、**单个块**的输入 `inputs` 与目标 `targets`，返回的也只是更新后的 `delta` 以及当前块在旧权重下的修正输出 `inputs @ delta.T`。代码里没有 batch 维度，没有请求槽位索引，也没有任何关于 CUDA Graph 或多卡切分的逻辑。

怎样在不改动作者这一处单请求数学公式的前提下，让它在高并发、开启 CUDA Graph、甚至跨越多张 GPU 的推理引擎里正确高效地运行？

实现这套参数覆盖层（Parameter Overlays）并不需要从头编写专用的 C++/CUDA 内核——事实上，覆盖层的数学部分没有一个专用的手写内核，只复用了 PTTS 通用的分页拷贝内核（量化底座另需一个通用的反量化内核）。我们的做法是顺着 PyTorch 自身的抽象层往下走：把“单请求语义与多租户引擎”之间的差距拆解成五个正交的系统问题，让每一层 PyTorch 原生变换各自解决其中一个问题。

```text
Author's Math:  step(delta, inputs, targets)   <-- Written for 1 request, 1 chunk
  │
  ├─► 1. torch.func.vmap                 <-- Vectorizes single-request math across a batch
  ├─► 2. Bounded index_select + bmm      <-- Amortizes grouped weight reads in continuous batching
  ├─► 3. torch.library.custom_op         <-- Enters vLLM's compiled CUDA Graph via table.data_ptr()
  ├─► 4. grad ──► vmap ──► torch.compile <-- Generates per-chunk backward without mixing BF16 reductions
  └─► 5. make_fx (Global vs. Shard)      <-- Proves TP>1 updates require zero extra communication
```

下面我们从最简单的单请求前向出发，逐一引入真实服务环境里的五道约束，看每一层抽象为什么必须长成现在的样子。

---

## 一、 从单个请求到一批请求：用 `vmap` 隔离批处理维度

如果系统里永远只有一个请求，框架只需取出该请求的参数张量，直接调用作者写的函数即可。

一旦进入多请求并发环境，每个请求各自持有一份独立的参数，而模型传来的激活张量则是按行拼在一起的。如果把 batch 维度直接暴露给作者，要求作者自己去写批量矩阵乘或张量索引，不仅容易写错维度，还会带来隐患：一旦作者不小心在 batch 维（第 0 维）上做了归约或归一化，不同租户的数据就会互相污染（见[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）。

要让一段只针对“单行、单份参数”编写的代码自动扩展到一批请求，同时从构造上禁止跨请求的数据混合，最直接的工具是函数式向量化变换 `torch.func.vmap`。

当用户在负载中调用一个绑定在请求状态上的 `nn.Module` 时，`pllm/execution/module_graph.py` 中的 `ModuleCall` 会将模块调用包在如下变换中：

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

这里的逻辑分为三步：
1. 每个请求的参数在底层被展平存放在连续的一维切片 `flat` 中；`_views` 将这片一维内存重新切分并还原为模块各个权重张量的原始形状，交给 `torch.func.functional_call` 进行无状态调用。
2. 在内部函数 `row` 中，输入激活被临时加上一个长度为 1 的首维（`unsqueeze(0)`），执行完模块前向后再将该维度去掉（`squeeze(0)`）。因此，对模块内部的代码而言，它看到的永远是一个 `batch_size = 1` 的标准输入。
3. 最外层的 `vmap(row)` 沿着请求维度将该单行函数自动映射为批量算子。作者无需感知并发请求的存在，而底层发出的依然是合并后的张量运算。

---

## 二、 连续批处理下的分组投影：用定长分块 `bmm` 摊薄权重读取

`vmap` 解决了“每行输入对应一份独立参数”的同构调用。但在连续批处理的模型前向内部，只读覆盖层（或固定读取本步起始权重的 `visibility="forward_pinned"`）会遇到一个更具体的访存效率问题。

在推理引擎的同一个执行步里，不同请求占用的行数往往是不均匀的：正在做分块预填充（chunked prefill）的请求 A 可能一口气占了 80 行，而正在做解码（decode）的请求 B、C、D 各自只占 1 行。所有这些行共享同一个基座线性层，但每个请求的权重增量（稠密矩阵 $\Delta W$、低秩因子对 $A, B$ 或按头的增量）各自驻留在不同槽位里。这些槽位由底层的 **PTTS（Paged Tensor Transition System，分页张量状态迁移系统）** 统一管理（详见[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)；顺便说明一句，PTTS 开头的 P 指的是 Paged，即像操作系统虚拟内存与 vLLM 的 PagedAttention 那样做分页与页表管理，和 pLLM 名字里代表 programmable 的 p 没有关系）。

在 GPU 上执行这种“不同行乘不同权重”的计算，本质上是一个分组矩阵乘（Grouped GEMM）问题。如果采取最天真的两种做法，都会撞上硬件瓶颈：
* **做法一：按请求循环，为每个请求单独发射一次矩阵乘。** 当批次里有 128 个 decode 请求时，一次线性投影就会碎裂成 128 个极小的 kernel，CPU 发射开销与小网格的低占用率会严重拖慢执行。
* **做法二：为每一行输入都从状态池里单独 gather 一份权重，拼成 `[num_tokens, ...]` 做一次批量矩阵乘（`torch.bmm`）。** 这样虽然只有一个 kernel，但对于占了 80 行的 prefill 请求 A 来说，同一份权重矩阵会被重复从显存读取并拷贝 80 遍，白白烧掉显存带宽。

要在不编写定制 CUDA 内核的前提下兼顾这两头，关键在于让读取粒度与计算阶段对齐：`pllm/execution/parameter_projection.py` 采用了按请求定长分块（bounded blocks）的策略。

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

具体而言，我们将属于同一个请求的连续输入行切分为固定宽度 `count` 的块（`blocks`）。如果当前步全部是单行 decode（`rows <= requests`），块宽 `count` 就取 `1`；只要批次中混入了多行 prefill，块宽就固定为常数 `ROWS_PER_BLOCK = 32`（块内不足 32 行的尾部补齐并在写回时掩码清零）。这样，每个 32 行的块只需从状态页池里抽取**一次**权重，就把权重的显存读取成本平摊给了块内的 32 个 token：

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

这段实现完全由标准 ATen 原语（`index_select`、`bmm`、`masked_fill`）组成，为了配合静态图捕获，总块数 `blocks` 会在初始化时按 2 的幂次分档（`capacities`）预先编译与热身。

这里需要做一笔诚实的横向对比。如果你的生产场景只需要同时服务成百上千个**只读、静态**的 LoRA 适配器，像 S-LoRA、Punica 以及 vLLM 原生内置的 multi-LoRA 拥有专门手写的分组 GEMM 内核（如基于 Punica BGMV / SGMV 的定制算子），它们能够在算子内部直接根据指针数组寻址不同 rank 的因子并支持 QKV 融合，省去了显式的 `index_select` 权重聚集。在纯静态只读的 multi-LoRA 场景下，专用内核比我们这套 `index_select + bmm` 更快、功能也更完备。

`projection_read` 选择用短短五行标准算子来构建，换取的是**对状态形态与演化的通用性**：同一套代码不加修改即可同时支持稠密矩阵增量（`dense`）、低秩因子对（`low_rank`）和按注意力头的增量（`per_head`）；更重要的是，它的源张量直接来自版本化的分页状态池——这意味着被读取的权重可以是在线训练实时更新的、具有多版本并发隔离的活状态。

---

## 三、 跨越编译与 CUDA Graph 边界：用张量地址充当会话身份

有了分块读取逻辑，下一步是让它进入 vLLM 的稳态解码图。

正如我们在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)中所讨论的，vLLM 会使用 `torch.compile`（Dynamo）追踪模型前向，并将解码步录制成静态的 CUDA Graph。在图回放期间，Python 解释器完全不参与执行，GPU 内核只能访问录制时绑定好的固定显存地址。然而，在连续批处理中，每一步里哪些请求在场、每个请求当前应该读取哪个物理页槽（`slots`），是每步都在变化的。

解决地址固定的办法依然是“加一层间接表”：每个目标投影层挂载的 `ProjectionReader` 持有一张地址固定不变的 `int64` 设备索引表 `table`。在每一步图回放之前，CPU 将当前步算好的 `slots`、`rows` 和 `owners` 异步拷入 `table`；图内的算子只对着固定地址的 `table` 和固定地址的状态页池 `pools` 寻址。

但要把这段读取逻辑放进 `torch.compile` 追踪的模型图里而不引发图断裂（graph break），我们必须将它注册为一个不透明的 PyTorch 自定义算子 `@torch.library.custom_op`。这会立刻撞上我们在[《图内挂载点》](05-in-graph-attach-points.md)中也遇到过的同一个编译器约束：

`custom_op` 的函数签名只允许传递 `Tensor` 和基础标量（`int`, `float` 等），不能直接传入 Python 层的 `ProjectionReader` 对象。如果改成传一个整数编号 `reader_id: int` 去全局字典里查找呢？在单次脚本里可行，但在 vLLM 里不行——因为 Dynamo 编译出的计算图（包括持久化的编译缓存）会把传入的整数标量当作编译期常量直接烘焙（bake in）进生成的代码里，导致不同引擎会话或重绑定的实例在复用编译图时查到错误的整数 ID。

既然标量会被当成常量焊死，而输入 `Tensor` 不会被焊死，我们就可以利用张量自身携带的一个唯一整数：**它在 GPU 显存中的物理起始地址 `table.data_ptr()`**。

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

每个 `ProjectionReader` 在创建时分配自己专属的 `table` 张量，并在整个会话生命周期内持有它的引用（直到 `close()` 时才释放并注销）。只要张量不释放，CUDA 分配器就不会把这块物理地址分给其他张量。在初始化时，reader 以 `self.table.data_ptr()` 为键将自己登记到全局字典 `_READERS` 中；当 `pinned_projection` 在图内被调用时，编译器看到的 `table` 只是一个普通的运行时张量输入，而在算子内部，一次 `.data_ptr()` 查找就能准确找回对应的 `ProjectionReader` 实例。

---

## 四、 在前向传播中求导：`grad`、`vmap` 与 `compile` 的组合次序

解决了图内读取，接下来看更核心的问题：如果权重需要在模型前向传播的半路上按块（chunk）在线训练，反向传播和批量更新该怎么组织？

在 pLLM 中，前向内训练（`@pllm.train_in_forward`）支持两种声明方式。第一种如开篇的 In-Place TTT，作者直接写出从 `(旧权重, 块输入)` 到 `(新权重, 块读出)` 的显式更新算式；第二种则允许作者给状态绑定一个优化器（如 `optimizer=pllm.SGD(self.learning_rate)`），在方法体里只写前向的逐行损失函数（loss），把求导和参数更新全部交给框架。

例如在 `pllm/workloads/prefill_adapters.py` 的 `PrefillAdaptersWorkload` 中，模型在阅读长 prompt 时，每隔 128 个 token 就在第 0、2、4、6、8 层的 `down_proj` 上实时训练一对低秩因子 `a` 和 `b`，以最小化该层输出对目标的均方重构误差，并在 prompt 结束时将尾部不满一块的梯度也刷入权重（`flush="prompt_end"`）：

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

作者只写了三行计算误差的前向代码。为了把它变成一个能同时对多个请求的块求梯度并融合执行的内核，`pllm/execution/overlay_task.py` 中的 `overlay_gradient` 将三个 PyTorch 变换按严格的次序组合在一起：

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
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        eager, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )
```

这段不到二十行的代码里，有两个非常关键的技术细节：

第一，**为什么变换次序必须是 `grad` 在最里层、`vmap` 在中间、`torch.compile` 在最外层？**
`torch.func.grad` 要求被求导的函数必须返回一个标量（scalar）。在 `loss` 函数内部，我们针对单个请求的一个块，用 `valid` 掩码滤除填充行后求和（`.sum()`），得到该请求的标量损失，再由 `torch.func.grad(loss)` 生成单请求的参数梯度函数。随后，中间层的 `torch.vmap` 沿着请求维度将该梯度函数向量化，并通过 `in_dims=(0, 0, None, 0)` 指定：每个请求的参数 `parameters`、激活操作数 `operands` 与掩码 `valid` 按第 0 维切分，而第三个参数 `constants`（即模型冻结的基座权重 `base`）设为 `None`——直接跨请求广播共享，绝不在显存里复制基座权重。最后，最外层的 `torch.compile`（AOTAutograd + Inductor）将展开后的前向与反向图一起编译融合。

第二，**既然已经用了 `vmap` 做批处理，为什么要显式传一个 `chunk_size=1`？**
注意看代码里的注释。在 `learn` 函数中包含一项基座矩阵乘 `x @ base.T`。对于单个请求，输入 `x` 的形状是 `[C, H]`；如果让默认的 `vmap` 把 $B$ 个请求的输入沿着行维彻底折叠进同一个矩阵乘里，送给 cuBLAS 的左矩阵形状就会变成 `[B * C, H]`。
在实数代数上，矩阵乘的每一行互不相干，拼成 `[B * C, H]` 算和按 `[C, H]` 算完全等价；但在 GPU 的浮点数实现里并非如此。当 GEMM 的 $M$ 维（行数）从 $C$ 变成 $B \times C$ 时，cuBLAS 的启发式选择器往往会切换到另一个分块（tile）尺寸不同的内核，导致内部 BF16 乘累加的归约顺序发生改变。这会让同一个请求在“单独运行”和“与其他请求同批运行”时算出微小的梯度差异，破坏多租户引擎严格要求的批次大小不变性（batch invariance）。加上 `chunk_size=1` 后，每个请求针对共享基座 `base` 的 GEMM 保持了各自原本的 `[C, H]` 形状，而外层依然处在同一个 `torch.compile` 图内统一编译，在守住逐请求数值一致的同时消除了 Python 循环开销。

### 设计取舍：跨步分块时的“最小割重计算”与“梯度累加”

在把 `overlay_gradient` 放进连续批处理时，还有一个棘手的跨步生命周期问题：一个训练块（例如 `chunk = 128` 或 `1024` 个 token）通常不会在同一个引擎步里一次到齐。在分块预填充时，它可能分两步到达（每步 64 个 token）；到了解码阶段，更是每一步只到达 1 个新 token。只有当攒满整个 `chunk` 时，优化器才能真正执行一次权重更新。

那么，在块还没攒满的中间这几步里，系统应该在显存里保存什么？

一个从编译器视角看非常诱人的思路是：把跨越多步的一个 chunk 看作**“一次被拉长到多步的联合前向与反向”**。PyTorch 的 AOTAutograd 内部配有一个最小割重计算分割器（`min_cut_rematerialization_partition`），它能自动分析前向与反向图之间的张量依赖，求出一个最小割，决定哪些中间激活值得保存下来传给反向、哪些激活不如在反向时重算。如果让这个分割器对单行计算图做切分，决定每一步到达的 token 需要把哪些切口上的中间张量存进 PTTS 页池，等攒满整个 chunk 的那一步再统一运行反向传播，在概念上似乎极其优雅。

但只要审视一下块内训练的数学前提，就会发现有一条简单得多的路。

由于在同一个 chunk 尚未攒满之前，该请求对外可见的快权重是**固定不变**的，每一行输入对权重的损失函数都是独立可加的：

$$\nabla_\theta \sum_{i=1}^{C} \ell(x_i; \theta) = \sum_{i=1}^{C} \nabla_\theta \ell(x_i; \theta)$$

这意味着，“先把每一步到达的行的中间激活存起来、等攒满 $C$ 行再一起求一次导”，与**“每一步直接对刚刚到达的这几行求导、把算出的梯度累加进状态池的待提交梯度字段（`_pending_grad_*`）里、等攒满 $C$ 行时直接用累加好的总梯度走一步优化器”**，在数学上是严格等价的。后者就是最标准的**梯度累加（Gradient Accumulation）**。

如果走跨步激活分割的路线，不仅需要依赖 PyTorch 私有的内部图分割接口、多维护几百条语句的跨步激活页生命周期管理，而且当块很大（如 `chunk = 1024`）时，跨步保存几百上千行的中间激活会消耗大量显存。而采用梯度累加，AOTAutograd 的最小割分割器依然在工作，只不过它只需要安静地待在**每一步自己的 `torch.compile` 内部**处理当步的临时激活；跨步留存下来的，只有形状永远固定、与参数大小完全一致的梯度张量。当简单的数学等价已经能把问题彻底解掉时，没有必要引入更复杂的编译器机制。

---

## 五、 张量并行（TP > 1）：用 `make_fx` 双图同构证明“零通信”

当基座模型增大到单卡显存不够宽裕时，服务引擎会开启张量并行（如 `tensor_parallel_size = 2` 或 `4`），将线性层的权重矩阵切分到多张 GPU 上。此时参数覆盖层面临的问题是：如果我们在多卡上运行，读写和更新这些增量权重会不会引入昂贵的跨卡集合通信（`all-reduce` / `all-gather`）？

先看**读取**路径。以最常挂载覆盖层的 MLP `down_proj` 为例，它在 vLLM 中是一个行并行线性层（`RowParallelLinear`）：设输入特征总宽度为 $H$，在 TP=$P$ 下，第 $j$ 张卡只持有输入激活的第 $j$ 个特征分片 $x_j \in \mathbb{R}^{\text{rows} \times (H/P)}$ 和基座权重的对应列分片 $W_j \in \mathbb{R}^{\text{out} \times (H/P)}$。每张卡先独立算出局部矩阵乘 $x_j W_j^\top$，随后由 vLLM 发起一次跨卡 `all-reduce` 求和。

由于矩阵乘法在被收缩的特征维上天然满足分配律：

$$x(W + \Delta W)^\top = \sum_{j=0}^{P-1} x_j (W_j + \Delta W_j)^\top = \sum_{j=0}^{P-1} \left( x_j W_j^\top + x_j \Delta W_j^\top \right)$$

只要我们也把稠密增量 $\Delta W$（或低秩因子对 $B A$ 中右边的输入侧因子 $A$）沿着同样的特征维 $H$ 切分成 $\Delta W_j$，让每张卡在 vLLM 执行那次 `all-reduce` **之前**，把本地算出的修正量 $x_j \Delta W_j^\top$ 加到基座的局部和上，修正量就能直接搭上 vLLM 原本那次 `all-reduce` 的顺风车完成跨卡归约，不需要增加任何一次额外的跨卡通信！（对于像 `q_proj`、`k_proj`、`v_proj` 或 `gate_proj`、`up_proj` 这样的列并行线性层 `ColumnParallelLinear`，输入 $x$ 是全卡复制的，而输出特征维被切分，因此沿着输出维切分 $\Delta W$ 或低秩左因子 $B$ 同样天然免通信。）

为了把修正量精确塞进 `RowParallelLinear` 的 `all-reduce` 之前，同时不破坏多卡下的浮点精度，`pllm/backends/vllm/parameter_overlay.py` 中的 `ProjectionHook` 是这样拦截的：

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

这里有一个很容易被忽略的数值细节：在 `TP > 1` 的行并行层上，`ProjectionHook` 并没有在 `module.forward` 的出口（即 `all-reduce` 之后）才去加修正量，而是下沉一层，直接拦截了 `module.quant_method.apply`（局部 GEMM 刚算完、尚未进入 `all-reduce` 的位置）。更重要的是，如果每张卡先把局部修正量加进 BF16 的局部和里再去做跨卡归约，那么 $P$ 张卡就会各自多引入一次向 BF16 的舍入误差；因此 `base` 方法先用 `.float()` 将局部结果提升为 FP32，在 FP32 下叠加上覆盖层的修正量并直接穿过 vLLM 的 `all-reduce`，等所有卡求和完毕后，再在最外层做唯一一次向 BF16 的舍入。

读取路径搭上了顺风车，但真正的难题在于**在线更新（`@pllm.train_in_forward`）**：当每张卡手里只有输入特征的一个分片 $x_j$ 和权重的一个分片 $\Delta W_j$ 时，如果让每张卡各自独立运行一遍作者编写的更新函数，算出来的新权重分片还正确吗？

这完全取决于作者在更新函数里写了什么样的代数运算：
* **回顾开篇的 In-Place TTT 外积更新**：`update = values.T @ inputs`。在这里，由 MLP 输入算出的目标 `values`（形状 `[C, Out]`）在每张卡上是完整复制的，而 `inputs`（形状 `[C, H/P]`）是按特征维切分的。矩阵乘法收缩的是长度为 `C` 的块内行维度，特征维 `H/P` 只是外积的自由维度。因此，每张卡独立计算 `values.T @ inputs_shard`，得到的结果恰好就是完整外积矩阵的第 $j$ 个列分片，完全不需要与其他卡交换信息。
* **再看上一节的 `PrefillAdapters` 重构误差求导**：`prediction = x @ base.T + (x @ a.T) @ b.T`，随后对 `(prediction - target).square()` 求导。在这里，输入 `x`（`[C, H/P]`）与权重 `base`（`[Out, H/P]`）都是沿特征维切分的，矩阵乘 `x @ base.T` 收缩的恰恰是被切开的那个特征维！单张卡算出的只是局部部分和，如果不先跨卡 `all-reduce` 求和就直接在本地算重构误差并求梯度，每张卡算出的梯度就是彻底错误的。

作为一个拒绝静默算错（fail-closed）的系统，我们必须在加载模型之前自动判定：用户写的更新函数（或其自动生成的梯度图）究竟是不是“分片本地（shard-local）”的？

如果按照传统编译器的做法，我们需要编写一套完整的分片传播静态分析器，为成百上千个 ATen 算子逐一定制分片推导规则。但在 `pllm/execution/overlay_parallel.py` 中，我们利用 `make_fx` 将这个问题简化成了一次**双图同构检验**：

既然 `make_fx` 可以用不占物理内存的假张量（FakeTensor）把函数追踪成符号计算图，我们干脆把同一个更新函数追踪两遍——**第一遍喂给它单卡的全局完整形状（例如特征宽 $H$），第二遍喂给它切分后的单卡本地形状（例如特征宽 $H/P$）——然后把两张 FX 图并排放在一起，逐个节点比对它们的输出形状！**

```text
Case 1: In-Place TTT Outer Product (Shard-Local -> ADMITTED)
  Global Trace (width = H)              Shard Trace (width = H/P)
  inputs [C, H]   values [C, Out]       inputs [C, H/P] (sharded axis=1)  values [C, Out]
         └───┬───┘                             └─────┬─────┘
             ▼ aten.mm                               ▼ aten.mm
       update [Out, H]                         update [Out, H/P]  <-- Axis 1 still sharded! OK!

Case 2: Feature-Axis Contraction / Reduction (Needs All-Reduce -> REJECTED)
  Global Trace (width = H)              Shard Trace (width = H/P)
  inputs [C, H]   base.T [H, Out]       inputs [C, H/P] (sharded axis=1)  base.T [H/P, Out]
         └───┬───┘                             └─────┬─────┘
             ▼ aten.mm                               ▼ aten.mm
       pred   [C, Out]                         pred   [C, Out]    <-- Sharded axis vanished! REJECT!
```

看上面的对比图：我们根本不需要给算子写复杂的线性代数定理，分片张量自己在两张图上的形状变化就已经把答案招出来了。在 `overlay_parallel.py` 中，判断一个节点输出处于“全卡复制”还是“单轴均匀分片”状态的函数 `_axis` 只有寥寥数行：

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

拿着全局图 `whole` 和分片图 `local` 在同一节点处的输出张量形状逐维比对：如果所有维度长度完全相等（`not axes`），说明该张量是全卡复制的（返回 `None`）；如果恰好只有一个维度不同，且全局维度恰好是分片维度的 `size`（即 TP 卡数）倍，说明该张量在该轴上保持着均匀分片（返回该分片轴下标 `axes[0]`）；其余情况直接拒绝。

在此基础上，`_check_graphs` 对剪除死代码（`eliminate_dead_code()`，去掉仅用于当前步读出而不影响新权重的分支）后的两张图做拉链式遍历：

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

遍历时执行的核心判据非常清晰：
1. 两张图的节点数量与算子目标必须一一对应，否则说明存在依赖张量宽度的 Python 控制流，直接拒绝；
2. 禁止出现张量常量属性或带随机种子的非确定性算子；
3. **分片守恒检查**：找出当前节点所有带分片轴的输入节点 `inputs`。如果一个算子没有消费任何分片输入且输出也未分片（`not inputs and axis is None`），说明它是像计算 `values` 那样的纯复制计算，直接放行；但**只要一个算子消费了哪怕一个带分片轴的输入（`inputs` 非空），它的输出就必须继续保持分片状态**（一旦 `axis is None`，说明被切分的特征轴在算子内部被归约或收缩消掉了，立刻触发 `_reject`）！

当有人试图在 `TP > 1` 下运行前面那个需要跨卡求和的 `PrefillAdapters` 时，`admit_update` 会在模型加载前的规划阶段直接拦截，并指出具体的违规算子与修改方向：

`TP projection update is not shard-local at aten.mm.default; its update/gradient needs other ranks' shards or an unproved operation. Use a shard-local update or tensor_parallel_size=1`

通过比较同一个函数在全局形状与分片形状下的两张 FX 图，仅用两百行代码就完成了无通信分片更新的证明。凡是通过准入的稠密或低秩前向内更新，在 `TP > 1` 下均无需插入任何额外集合通信。在 TP=2 下以 FP64 GPU 参考实现作为基准（oracle）进行核对时，覆盖层输出的均方根（RMS）误差不超过原生 TP=2 自身浮点误差的 105%，最大绝对误差不超过 1 个 BF16 ULP。

---

## 六、 实测表现与浮点尾数的算术账

将这五层机制组合起来，我们在真实的 4B In-Place TTT 检查点上验证它的运行行为与性能。

以下 In-Place TTT 实验均在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0、CUDA Graph 开启下运行，使用社区开源的 4B 检查点 `zhongweixie/inplace-ttt-qwen3-4b-32k-ttt`（共 36 层，第 0、6、12、18、24、30 层的 `down_proj` 为 `[2560, 9728]` 的快权重，块大小 `chunk_size = 1024`，学习率 `3.0`），贪心解码生成 256 个 token，预热 1 轮后取 3 轮中位数；吞吐定义为“生成的 token 总数除以包含 prefill 在内的整次 `generate` 耗时”。官方基线使用 `ByteDance-Seed/In-Place-TTT` 仓库基于 Hugging Face `generate`（BF16 + FlashAttention 2）在同一张 RTX 4090 上按相同协议测得。

### 1. 显存账本与跨块精确可见性

在这个 4B 模型中，6 个目标层的 `[2560, 9728]` 快权重矩阵共有 `149,422,080` 个参数。若以 FP32 存放，仅这 6 个增量矩阵就要占约 570 MiB；加上暂存未满 1024 块输入行的缓冲，每个请求的一份状态根（root）占 **`714.0 MiB`**（6 层共享的只读卷积与投影权重 `target_parameters` 另占 `75.1 MiB`）。由于 PTTS 采用不可变版本与写时复制，单请求运行时需预留 4 份根，4 并发请求时需预留 10 份根，因此测试时将 vLLM 的 `gpu_memory_utilization` 设为 `0.48`，为快权重状态池留出固定显存空间。

为了直观检验 `visibility="exact"` 在解码过程中的边界行为，我们可以构造一个刚好卡在块边界上的输入：取 Python 标准库 `argparse.py` 源码的前 **1023 个 token** 作为 prompt（比块长 1024 少 1 个 token），然后贪心解码生成 5 个 token。实验组（`TTT`）使用正常的 `chunk_size = 1024`；对照组（`不更新`）将 `chunk_size` 设为 `2048`（整条序列共 $1023 + 5 = 1028$ 个 token，永远不会触发更新）。

对比两组在每一步生成的 token 及其对数概率（logprob）：

| 生成序号 | 生成的 token | 产出该词的位置下标 | `TTT` 的 logprob | `不更新` 的 logprob | 差值 | 权重状态与可见性 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `' _'` | 1022 | `-0.4400` | `-0.4400` | `+0.0000` | 第 1023 个位置，未满一块，读取初始权重 |
| 2 | `'get'` | 1023 | `-1.9058` | `-1.9058` | `+0.0000` | **第 1024 个位置：凑满第一块并触发更新，但其自身前向仍读旧权重** |
| 3 | `'_default'` | 1024 | `-2.2565` | `-2.3191` | `+0.0626` | **第二块的第 1 个位置：立即读到更新后的新权重** |
| 4 | `'_action'` | 1025 | `-1.8538` | `-1.8214` | `-0.0324` | 继续读取更新后的新权重 |
| 5 | `'(action'` | 1026 | `-1.7855` | `-1.7124` | `-0.0732` | 继续读取更新后的新权重 |

可以看到，当位置 `1023`（全序列的第 1024 个 token）进入模型时，第一块刚好凑满 1024 个 token，6 个快权重层各自触发了一次外积更新。但按照严格的因果语义，第 1024 个 token 自身的前向输出仍然由更新前的旧权重算出，因此它的 logprob 与不更新的对照组完全一致（差值为 `0.0000`）。到了紧接着的下一步（位置 `1024`，即第二块的第一个 token），PTTS 状态根已完成切换，模型立即读到新权重，logprob 随之发生 `+0.0626` 的改变。

### 2. 端到端吞吐与 BF16 尾数吞噬现象

在长上下文（4096 与 16384 prompt 长度）及多请求并发下，整次生成的吞吐对比如下：

| 实现与快权重精度 | 1 个请求，prompt 4096 | 1 个请求，prompt 16384 | 4 个请求，prompt 4096 |
|---|---:|---:|---:|
| 官方 HF 实现（BF16） | `54.79 tok/s` | `34.89 tok/s` | 不支持（仅限 `bs=1`） |
| pLLM，BF16 快权重 | `57.12 tok/s` | `40.64 tok/s` | 未测 |
| pLLM，FP32 快权重（默认） | `59.64 tok/s` | `42.27 tok/s` | `111.38 tok/s`（4 请求合计） |

在同为 BF16 快权重的设置下，pLLM 在 4k 长度下比官方实现快约 4%，在 16k 长度下快约 16%；当并发数增加到 4 个请求时，官方实现受限于 `bs=1` 无法运行，而 pLLM 的合计吞吐提升至 **`111.38 tok/s`**（另一组较早的、只对纯 decode 阶段计时的测量里——4096 prompt，BF16 官方快权重配 FP32 残差存储——把 `exact` 更新录进 CUDA Graph，使单请求纯解码吞吐从 eager 的 `57.59` 升至 `59.51 tok/s`，4 请求从 `107.56` 升至 `113.16 tok/s`）。

观察上表，有一个现象非常值得拿出来细看：**为什么将快权重设为 32 位浮点数（FP32）不仅成了 pLLM 的默认选项，而且跑起来反而比 16 位的 BF16 更快？**

答案就在前面省略的 `inplace_ttt.py` 末尾那几行分支代码里：

```python
# check-docs: skip: excerpt from pllm/workloads/inplace_ttt.py
if self.dtype is torch.bfloat16:
    weight = (base.float() + delta).to(torch.bfloat16)
    output = (inputs @ weight.T).float() - (inputs @ base.T).float()
    return (weight + update).float() - base.float(), output
return delta + update, inputs @ delta.to(inputs.dtype).T
```

在官方的 Hugging Face 实现中，推理代码直接把 BF16 的 `down_proj` 权重克隆一份作为快权重，每次算出微小的更新量 $\Delta W$ 后，直接在 BF16 精度下往完整的权重矩阵上硬加：$W \leftarrow \text{BF16}(W + \Delta W)$，没有维护 FP32 主副本（master weights）。

从数值分析的角度看，直接用 BF16 累加微小权重更新会遭遇严重的大数吃小数（mantissa swamping）。BF16 只有 7 位有效尾数，相对精度台阶约为 $2^{-7} \approx 0.78\%$。我们可以拿这个 4B 检查点第 0 层真实的 `down_proj` 权重矩阵（共 `24,903,680` 个元素）做一次简单的数值实验：原论文附录 C.2 说，对 Qwen3-4B 报告的所有评测，每个块的局部更新都按 Frobenius 范数裁剪到 `1e-5`。如果将一个范数为 `1e-5` 的更新矩阵直接在 BF16 下加到第 0 层的基座权重上，在全部 **`24,903,680`** 个元素中，只有 **`472`** 个元素的值真正发生了改变——其余 99.998% 的元素因为增量小于 BF16 的最低有效位（ULP），在加法完成后被直接舍入回了原值，整次更新实际留存下来的改变量只有预期值的约 **`0.97%`**。（注：该社区检查点本身采用的是官方仓库默认的不裁剪设置且学习率为 3.0，单步更新幅度比 `1e-5` 大得多，因此上述实验仅用于说明浮点尾数的精度边界，而非对该检查点生成质量的评测。）

一旦我们将“只读的基座权重 $W$”与“每个请求私有的快权重增量 $\Delta W$”解耦开来——基座 $W$ 保持原本的 BF16（甚至是量化格式：在 A100 80 GiB 上运行 14B GPTQ 量化模型时，训练求导只需在读取基座那一刻做一次用完即焚的瞬态反量化；用一个直接的 Triton 分块逆变换代替借 vLLM 单位矩阵乘的反量化后，单步训练耗时从 `1.117 s` 降至 `0.193 s`，峰值显存不变），而每个请求单独用 FP32 累加从零起步的增量 `delta`——微小更新被基座大数吞噬的问题就彻底消失了。同时，在计算前向修正时，FP32 模式只需做一次 `inputs @ delta.to(inputs.dtype).T`；反倒是严格模拟官方 BF16 舍入行为的分支，为了在不改动共享基座的前提下复现“先合成 BF16 权重再做矩阵乘”的舍入误差，不得不做两次矩阵乘再相减。这就是为什么 FP32 快权重既保住了数值精度，执行起来反而更加轻快。

---

## 七、 边界与代价

把每层抽象的工作原理摊开之后，这套参数覆盖层的设计边界与代价也就一清二楚了：

* **前缀缓存（Prefix Caching）互斥**：前缀缓存的前提是相同的 token 前缀永远算出相同的 KV Cache。一旦在线性层上挂载了随请求变化或随步更新的参数覆盖层，相同的 token 在不同请求或不同步里会算出不同的隐藏状态与 KV，因此覆盖层必须关闭或隔离前缀缓存。
* **`exact` 模式的串行预填充与图池驻留**：因为 `visibility="exact"` 要求后一个块必须读到前一个块更新后的权重，块与块之间存在严格的因果串行依赖，所以它的 prefill 阶段必须以 eager 模式按块顺序推进；如果你的负载全是超长 prompt 而只生成寥寥几个 token，这条路径无法并行重叠块间更新。此外，启用了 `exact` 解码图捕获的引擎在关闭时无法将私有图池显存全部归还给 CUDA，同一进程若要启动下一个大模型引擎需在独立子进程中运行。
* **多卡张量并行的准入范围**：目前 `TP > 1` 仅对通过了 `make_fx` 双图同构检验的前向内稠密或低秩更新开放；在旁路流上异步训练的适配器、外部动态导入或带状态的通用 `nn.Module` 适配器、以及读取共享可变 `InstanceState` 的适配器（后者还会使模型退回 eager 模式）目前均限制在 `TP = 1`。

回顾整个参数覆盖层的构建过程，最核心的体会其实只有一条：当我们需要把一段干净的“单请求、单块”研究代码搬进高度复杂的分布式推理引擎时，不必急着去手写面面俱到的底层大算子。PyTorch 已经把向量化批处理（`vmap`）、函数式自动求导（`grad`）、算子融合编译（`compile`）、编译器不透明边界（`custom_op`）和符号图追踪（`make_fx`）做成了正交的基础设施。看清每一步的物理约束与浮点脾气，把这几个现成的变换按正确的顺序组合起来，就能用最少的代码把黑板上的公式变成生产引擎里的高速路径。
