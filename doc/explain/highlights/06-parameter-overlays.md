# 参数覆盖层：顺着 PyTorch 的抽象层往下走

在 PyTorch 里给一个线性层加点东西，一行就够了：稠密增量是 `y = x @ (W + Δ).T`，低秩是 `y = x @ W.T + (x @ A.T) @ B.T`。

难的是 serving 引擎里的两个真实需求。第一个是**每个请求一份只读的权重增量**：Text-to-LoRA 一类的方法给每个请求生成一份 LoRA，或者外部给每个请求导入一份稠密增量，128 个请求各自的 A/B 不同，却挂在同一个 `down_proj` 上。第二个更难，是**前向内的在线训练**。In-Place TTT（ByteDance Seed 的论文 *In-Place Test-Time Training*）把选定层 MLP 的 down 投影当作“快权重”：prompt 每读满一块（这个检查点是 1024 个 token），就用 `ΔW = η · Vᵀ Z` 更新一次，而且**同一次前向里，后一块必须读到前一块更新后的权重**。官方实现基于 HF `generate`，源码里写着 `assert len(input_ids) == 1, "only support bs=1 for TTT inference"`。

先看结果。在 pLLM 里，In-Place TTT 的作者写的代码长这样（`pllm/workloads/inplace_ttt.py`）：

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
    if self.config.track == "T2":
        norm = update.float().square().sum().sqrt()
        update = update * (1e-5 / norm.clamp_min(1e-5)).to(update.dtype)
    update = update * self.config.learning_rate
    if self.dtype is torch.bfloat16:
        weight = (base.float() + delta).to(torch.bfloat16)
        output = (inputs @ weight.T).float() - (inputs @ base.T).float()
        return (weight + update).float() - base.float(), output
    return delta + update, inputs @ delta.to(inputs.dtype).T
```

参数是“一个请求的 delta、一块的输入与目标、冻结的底座权重”，返回“新的 delta 和这一块读到的修正”。没有 batch 维，没有请求维，没有页表，也不知道自己跑在一张卡还是四张卡上。读起来就是论文里的公式。

这篇讲框架怎样接住这样一个函数。我们会从“一个请求、一个投影”开始，每次只加一样现实的麻烦，每样麻烦都正好对应 PyTorch 里一层已有的抽象：`vmap`、`grad` 加 `compile`、`custom_op`、`make_fx`。

## 一个请求：vmap

只有一个请求的时候，什么都不用做：拿这个请求的 delta，按论文的公式算。作者写的就是这个情形。

两个请求呢？它们的 delta 不同，行也不同。我们不想让作者改写代码去处理“一批请求”——批处理正是 bug 最多的地方。PyTorch 已经有现成的答案：`torch.func.vmap` 把一个“处理一行”的函数自动变成“处理一批”的函数。用户方法里出现的模块（比如一个请求自己的小网络），框架用 `ModuleCall` 包一层（`pllm/execution/module_graph.py`）：

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

`row` 里那句注释就是全部想法：模块看到的是“一行组成的 batch”，和普通 PyTorch 代码一模一样。每一行的参数来自那一行所属请求的那份状态（展平成一个向量 `flat`），`vmap` 把单行函数变成整批。作者不需要知道还有别的请求。

## 一百二十八个请求：分组读

只读的覆盖层还有一个更具体的问题：128 个请求，同一个投影，每个请求乘自己的矩阵。这是一个“批内分组 GEMM”。每个请求单独一个小 GEMM，kernel 数就爆了；把所有人的矩阵拼起来，又不是一个规则的形状。

pLLM 的做法很朴素（`pllm/execution/parameter_projection.py`）：把行按“属于同一个请求”分块——decode 时每块 1 行，prefill 时每块最多 32 行——每块从页池里 gather 一份矩阵（或一对 A/B 因子），然后一次 `bmm`：

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

没有专门的 kernel：`index_select`、`bmm`、`masked_fill`，全是普通的 ATen 算子，块数按 2 的幂分几档预热录图。prefill 时 32 行一块，把一次 gather 一个矩阵的代价摊给 32 个 token。（旁注：这不是最快的做法。专门为多 LoRA 设计的分组 GEMM kernel 会更快，后面会老实说。我们选它，是因为它对稠密、低秩、按头的增量都一样，而且不用维护一个 kernel。）

## 进入 CUDA graph：custom op 与地址

decode 步是录成 CUDA graph 回放的（[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)），回放时不能问 CPU“这一行该读哪个请求的哪一版权重”。和[《图内挂载点》](05-in-graph-attach-points.md)一样，入口是一个 custom op：编译器把它当不透明的节点，录图时录下它里面的 kernel；每步会变的“行 → 槽号”写在一张地址固定的索引表里。而 op 怎么知道自己属于哪个读者？用那张表的地址（`pllm/backends/vllm/parameter_overlay.py`）：

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
    """Execute the same prepared tensor callable eagerly or during upstream capture.

    A custom op takes only tensors and scalars, and vLLM's compiled (and AOT-cached) model graph bakes scalars in,
    so the reader travels as `table`: a runtime graph input whose address finds the reader without specializing
    the graph on one, as model_site does for sessions. A pinned reader registers in ProjectionReader.__init__ and
    leaves in close(); it owns the table meanwhile, so the address cannot be reused while registered.
    """
    _READERS[table.data_ptr()].emit(hidden, desired, pools, table, output, captured, desired_capture)
```

custom op 只收张量和标量，而标量会被烘焙进编译好的图，所以“我是谁”只能藏在一个张量的地址里。docstring 里说得很直白：和挂载点用的是同一个技巧。

## 在前向里训练：grad、vmap、compile 的顺序

只读覆盖层到此为止。前向内训练还要求导。以 PrefillAdapters 为例：它在 prefill 阶段训练一组低秩适配器，作者只写每行的 loss。反向由 `overlay_gradient`（`pllm/execution/overlay_task.py`）生成：

```python
# check-docs: skip: excerpt from pllm/execution/overlay_task.py
def overlay_gradient(function: Callable[..., torch.Tensor], *, batched: bool = False) -> Gradient:
    """Compose request mapping with differentiation before compiling; broadcast frozen weights."""

    def loss(parameters: Tensors, operands: Tensors, constants: Tensors, valid: torch.Tensor) -> torch.Tensor:
        rows = function(**parameters, **operands, **constants)
        if rows.shape != valid.shape:
            raise ValueError("autograd overlays must return one loss per input row")
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

顺序是先 `grad`，再 `vmap`，最后 `compile`。中间那两行注释记着一件只有真跑过才会知道的事：如果让 `vmap` 把多个请求折进冻结底座的那次 GEMM，cuBLAS 会因为形状变了而换一种 BF16 归约顺序，结果就和逐请求算的不一样。`chunk_size=1` 让每块保持原来的 GEMM 形状，同时仍然一起编译。

chunk 训练有一个时间维度：每步来几行，攒满一块（或者 prompt 结束时 `flush="prompt_end"`）才更新一次。这里有一段插曲值得讲。2026 年 10 月下旬，一份设计稿提出了一个很漂亮的方案：把 chunk 训练看成“一次被拉长到多步的前向 + 反向”，让 AOTAutograd 的 `min_cut_rematerialization_partition`——一个按最小割决定“保存还是重算”的分割器——把联合图切成前向图和反向图；切口上的张量就是反向需要的东西。**分割器决定存什么**，pLLM 只决定存在哪（PTTS 的页）、存多久（到块的边界）。

漂亮，但 10 月 28 日的决定是先不做。块内的状态是不变的，所以“每步对到达的行求梯度、累加、块满了做一次优化器步”和“攒满一块再求梯度”在数学上等价——这就是普通的梯度累加。而分割器路线要用 PyTorch 的私有 API，估计多出 240–360 条语句。于是落地的是：每步编译一次 `torch.func.grad`，梯度累进 PTTS 的一个字段，块满时做一次优化器步；分割器仍然在用，只是只在**每一步内部**决定保存还是重算。设计稿上写的是“等有 workload 需要再说”。

我们觉得这是“如无必要，勿增实体”的一个现场示范：先找一个更简单、而且可证明等价的做法，把漂亮的那个留着。

## 张量并行：make_fx 当证明器

最后一层麻烦来自多卡。先看只读的情况。`down_proj` 在 TP 下是行并行的：rank j 持有输入特征的第 j 片，算 `y_j = x_j W_jᵀ`，然后 vLLM 用一次 all-reduce 求和，得到 `y = Σ y_j = x Wᵀ`。稠密增量也按输入特征切片，每个 rank 在自己的部分和上加 `x_j Δ_jᵀ`；低秩增量切 A、复制 B，加 `(x_j A_jᵀ) Bᵀ`。矩阵乘对输入特征维是可加的，所以 vLLM 原本那次 all-reduce 求完和，结果正好是 `x (W + Δ)ᵀ`。修正量搭了原有通信的便车，一次也没多。`ProjectionHook` 只拦截 `quant_method.apply`，切片、偏置、all-reduce 都留给上游：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/parameter_overlay.py
class ProjectionHook:
    """Put a down delta inside vLLM's GEMM, before its own row all-reduce.

    TP1 keeps the established forward hook. TP>1 intercepts only quant_method.apply;
    input slicing, rank-zero bias, reduction and return-bias handling stay upstream.
    """

    def __init__(self, module: LinearBase) -> None:
        self.module = module
        self.forward = module.forward
        self.apply = module.quant_method.apply if _row_parallel(module) else None
        self.callback: Callable[[torch.Tensor], LinearOutput] = self.forward
        self.bias: torch.Tensor | None = None

    def base(self, hidden: torch.Tensor) -> LinearOutput:
        if self.apply is None:
            return self.forward(hidden)
        # Accumulating the correction in BF16 before the reduction adds a rounding
        # on every rank. Keep the partial sum wide and round once after vLLM sums it.
        return self.apply(self.module, hidden, self.bias).float(), None
```

`base` 里那两行注释是一个数值细节：如果每个 rank 先把修正加成 BF16 再去 all-reduce，每个 rank 都多一次舍入。所以部分和保持 FP32 穿过 vLLM 的 all-reduce，求完和只舍入一次。

难的是**更新**。作者写的更新函数，在 TP 下每个 rank 只拿到自己那片输入，它算出的还对吗？如果更新里有一个在特征维上的求和，每个 rank 算出来的就是残缺的结果，模型会默默地输出错的东西。

写一个通用的“分片分析器”去分析任意 PyTorch 代码？那会是一个大项目，而且永远不完整。`admit_update`（`pllm/execution/overlay_parallel.py`）的做法简单得多：**同一个更新函数，用全局形状和 rank 本地形状各追踪一次，然后比较两张 FX 图**。模块 docstring 一句话说完：

```python
# check-docs: skip: excerpt from pllm/execution/overlay_parallel.py
"""Prove projection updates commute with feature slicing before backend installation.

Authoring calls this with global and rank-local fake tensors. Only a conservative
set of ATen operations is admitted; execution adds no checks or collectives.
"""
```

比较的规则不多。两张图的节点必须一一对应，否则说明有依赖形状的控制流。每个张量最多有一个轴被切开，而且是均匀、连续的切片。最关键的一条：只要一个节点吃进了分片的输入，它的输出也必须仍然是分片的——在分片轴上做了归约或混合的 `mm`、`sum`，输出就不再分片，于是被拒绝。逐元素运算、转置、连续轴上的 view、在非分片轴上的 `mm`/`sum` 被允许；随机数、张量常量、不认识的算子，一律拒绝。

拿两个例子试一试。In-Place TTT 的外积更新 `Vᵀ Z`：目标 V 是复制的，输入 Z 按特征分片，乘积沿输出维展开，仍然保持分片——通过。一个“重构误差”式的低秩更新，要先用完整的 down 投影预测输出，才能给两个因子求梯度——预测这一步在分片轴上做了收缩，输出不再分片——拒绝，报错告诉用户：“its update/gradient needs other ranks' shards … Use a shard-local update or tensor_parallel_size=1”。

这是我们在这一篇里最喜欢的部分。它把一个看起来需要“程序分析”的问题，换成了“两张图是否同构、切法是否一致”的问题。它很保守，只认一小撮算子；但通过的东西，都可以放心地在每个 rank 上独立执行，运行时不再有任何检查。数值上用一个 FP64 的 GPU oracle 来验证：覆盖层在 TP=2 上的 RMS 误差不得超过原生 TP=2 自身误差的 105%，最大绝对误差不超过一个 BF16 ULP；实测都在门限之内。

## In-Place TTT 跑起来怎样

同一张 RTX 4090，社区 4B 检查点（`zhongweixie/inplace-ttt-qwen3-4b-32k-ttt`；不是 7B），贪心解码 256 token，1 轮预热 + 3 轮中位数，吞吐按“生成 token 数 / 含 prefill 的整次 generate 时间”计。pLLM 用 vLLM 0.32、CUDA graph 开；官方实现用 HF `generate`、BF16、FlashAttention 2，同一天同一张卡：

| | 1 个请求，prompt 4096 | 1 个请求，prompt 16384 | 4 个请求，prompt 4096 |
|---|---:|---:|---:|
| 官方 HF 实现（BF16） | 54.79 tok/s | 34.89 | 不支持 |
| pLLM，BF16 快权重 | 57.12 | 40.64 | 未测 |
| pLLM，FP32 快权重（默认） | 59.64 | 42.27 | 111.38（合计） |

公平的比较是同精度那一行：BF16 模式复现了官方“把更新直接加到 BF16 权重上”的舍入，比官方快约 4%（4k）和 16%（16k）。FP32 模式更快一点，但那不是同精度比较——FP32 模式每次投影只多一个 `inputs @ delta`；BF16 模式为了复现官方的舍入，要先把底座和 delta 合成 BF16，再做两次矩阵乘取差。4 个请求合计 111.38 tok/s，约是单请求的 1.9 倍。

为什么默认 FP32？官方代码把更新直接加在 BF16 的 down 投影上。在这个检查点第 0 层的真实矩阵上做个演示：一个范数为 1e-5 的更新（论文评测时每次更新被裁剪到的大小）直接用 BF16 加上去，24,903,680 个元素里只有 472 个变了，实际生效的改变量只有预期的约 0.97%——小更新被 BF16 的尾数吃掉了。（这只是一个舍入效应的演示，不是模型质量实验；而且这个社区检查点并没有开裁剪，单次更新比 1e-5 大得多。）论文没有规定精度，所以 pLLM 默认用 FP32 存增量，同时保留 BF16 模式给想逐步对齐官方的人。

exact 的 decode 后来也录进了 CUDA graph：同一个检查点（4096 prompt + 256 token，用 CUDA event 只计纯 decode 时间，2026 年 11 月的测量），单请求从 eager 的 57.59 提到 59.51 tok/s（+3.3%），4 请求从 107.56 提到 113.16（+5.2%）。prefill 仍是 eager。

## 在哪里更差

只读多 LoRA，pLLM 大概率更慢。vLLM 自己的 multi-LoRA、S-LoRA、Punica 是为“很多只读 LoRA 同批服务”专门设计的，有专门的分组 GEMM kernel，还处理了不同 rank 混排、QKV 融合这些情况；pLLM 用的是 gather 加 `bmm`。如果你只需要部署很多只读 LoRA，用 vLLM 自带的 multi-LoRA。pLLM 的增量在别处：权重可以在线训练、带版本、能和别的方法组合，稠密、低秩、按头的增量用同一套机制。

TP 的支持是窄的：只有能通过上面那个分片检查的稠密或低秩更新支持 TP>1；旁路训练的适配器、导入的、从检查点或模块来的适配器仍然只能 TP=1；列并行投影（q/k/v、gate/up）是 2026 年 11 月才加入的；读共享可变 `InstanceState` 的 Adapter 会让模型退回 eager。exact 的 prefill 是 eager——块与块之间是真正的串行依赖，负载要是几乎全是超长 prefill，这条路径不会快。与 prefix caching 互斥：权重改了，相同前缀的 KV 也就不同了（[设计取舍](../design-tradeoffs.md)）。

显存很重。In-Place TTT 每个请求的快权重是 6 个 FP32 矩阵，约 570 MiB，加上未满块的输入缓冲，一份约 714 MiB；单请求要留 4 份，4 个请求要 10 份，在 24 GiB 的卡上要把 vLLM 的显存比例降到 0.48 才装得下。用了 exact 前向内训练的引擎，关闭时暂时不能归还全部显存，后续的引擎要在新的子进程里启动。

量化底座倒是能用：覆盖层把高精度修正加在量化算子的输出上；训练要读冻结权重时做一次瞬态反量化（14B GPTQ，A100，历史测量：每步 1.117 s → 0.193 s，峰值显存不变）。

## 可以带走的东西

这一篇里没有一个为覆盖层专门写的 kernel。我们用了 `vmap`、`grad`、`compile`、`custom_op`、`make_fx`，每一样都是 PyTorch 早就做好的变换，每一样只管一件事：批处理、求导、和编译器划界、看懂一段程序。我们做的，主要是想清楚它们该按什么顺序叠——先 `grad` 后 `vmap` 再 `compile`，而不是反过来——以及在哪里停下：那个更漂亮的分割器方案，到现在还在设计稿里。

如果要带走一个更具体的想法，那就是 TP 那一节的做法：当你需要证明一段任意代码满足某个性质时，先别急着写分析器，看看能不能把它变成“同一段代码在两种设定下跑出的两张图是否一致”。比较两张图，比理解一段程序容易得多。
