# 状态与记忆

> **先决条件**：[写一个负载](write-a-workload.md)；教程
> [02](../tutorials/02_request_adapters.ipynb)、[03](../tutorials/03_decode_training.ipynb)
> 演示了请求状态与实例状态。
>
> **本文回答**：我有值要跨步、跨请求保存，该怎么声明？该选哪种状态？

负载的数学在绑定后就被冻结签名，整个系统里唯一可变的东西是状态。所以
「我的值放哪」这个问题，其实就是「我的状态声明成什么样」。本文按你可能
想做的事组织：选作用域、让请求分组共享、读参数、跨步记忆，最后是两个
绕不开的问题——显存预算和生命周期。

## 我想让值活下去：先选作用域

状态只有一个真正要选的轴：这份值是**每个请求一份**，还是**整个引擎实例
一份**。常数（比如一个永不训练的 SAE 编码器）不算第三种作用域，它只是
任意一种状态加上 `optimizer=None`。

| 你想要 | 声明 |
| --- | --- |
| 每个请求（或每组请求）一份独立状态 | `pllm.RequestState(...)` |
| 整个引擎实例共享一份状态 | `pllm.InstanceState(...)` |
| 只读、永不训练的常数（如 SAE 编码器） | 任意一种 + `optimizer=None`（**冻结状态**） |

配方——一个实例作用域的冻结编码器：

```python
import torch
import pllm

class Probe(pllm.Workload):
    # 实例作用域 + 冻结：准入时初始化一次，之后任何任务（包括观测）可读，永不被更新
    encoder = pllm.InstanceState(
        {"weight": lambda m: (4, m.hidden_size)},   # 形状回调拿到模型事实
        dtype=torch.float32,
        initialize=lambda tensors, generator: tensors["weight"].copy_(torch.eye(4, tensors["weight"].shape[-1])),
        optimizer=None,                              # 可省略，默认即 None
    )
```

为什么这样能工作：冻结状态不是「另一种东西」，它就是状态的冻结形态——
准入时初始化、任何任务（包括观测）可读、可通过外部通道导入、永不更新。
没有优化器，就没有梯度/动量槽，字段布局本身就是全部存储。

形状可以是 lambda（拿模型事实 `m`），也可以是 `pllm.Field(...)`（overlay
场景按目标输入/输出宽度解析）。声明可以写在类上、实例上；形状依赖配置时，
写成带类型的 property。默认 `dtype="auto"` 按字段角色选择：优化器参数、chunk carry、
声明读取并写回同一字段的方法使用 FP32；其他字段使用模型 dtype。`pllm.Field.dtype`
覆盖该字段的默认值。`RequestState.dtype` 只作用于普通形状字段。
梯度与 loss 归约至少使用 FP32，不随存储精度降低；布局的 `dtype` 是解析后的具体值。
浮点导入值在边界转换为布局的 dtype；整数字段仍需匹配。只读字段若参与固定 FP32 的矩阵乘法，
请显式声明 `dtype=torch.float32` 或在数学函数中转换操作数。
FP32 相对 BF16 使该字段的 payload 翻倍，也增加投影读取的转换工作区。旧的紧预算
可能在规划或图捕获时报错；提高 state / capture / execution 预算，或显式选择 BF16。

## 我想让一组请求共享训练成果：路由键

`RequestState` 默认按请求 ID 隔离——每个请求一条独立 lineage，互不可见。
如果你心里想的是「同一租户/同一类 prompt 应该共享训练成果」，那要显式
声明路由键：

```python
session = pllm.RequestState({"proj": lambda m: (m.hidden_size, m.hidden_size)},
                            optimizer=pllm.AdamW(lr=1e-3),
                            key_fields=("route",))
```

```python
# check-docs: skip: requires a CUDA engine, prompts, params and application route_of
llm.generate(prompts, params, routes=tuple(route_of(p) for p in prompts))
```

注意第二行：路由值由**你的宿主代码**给每个 prompt 计算。框架从不解释它——
正则、判别器分数、租户 ID 都行。路由值相同的请求**加入同一 lineage**：
任何一个请求训练出的适配器对全部生效，无论在同一批次内还是跨步；不同
route 完全隔离。

两个当场要问的问题：

- **缺路由值会怎样？** 请求在准入时被拒——路由键是签名声明的一部分，
  不会临时改判。
- **lineage 什么时候消失？** route lineage 在最后一个请求退役后**休眠存活**，
  仍被其键 pin 住——所以适配器池在整个会话中存续，会话关闭时才退役。代价是
  休眠 lineage 仍占准入容量（`max_admitted_lineages` 约束它）：`max_num_seqs`
  要留出余量，否则新请求会因为容量耗尽而被拒。

## 我想读参数：实例字段与请求行

读取 `RequestState` 的 `pllm.parameter("state", "field")` 带**领先的行轴**——
混合批次里每行拿到自己 lineage 的版本。请求参数的矩阵运算使用按行形式：

```python
h = torch.ones(2, 4)
w = torch.eye(4).expand(2, 4, 4)
torch.bmm(h.unsqueeze(1), w).squeeze(1)     # (rows, H) @ (rows, H, H) → (rows, H)
```

读取 `InstanceState` 默认保持字段声明的形状；需要广播到当前操作行时，使用
`pllm.parameter("state", "field", per_row=True)`。视频消费者的实例特征字段
直接保持网格形状，编码器自己组织计算输入。冻结状态也遵循其作用域的读取规则。

读到的是哪个版本？**输入版本**：一步之内的所有读取看到的都是步开始时
pin 住的版本，永远不是「最新值」。`pllm.version(state)` 输入给出该版本号，常用
作「是否训练过」的门（`versions > 0`）。

## 我想跨步记点东西：carry 字段与 pllm.carry

优化器状态（参数 + Adam 槽）之外的普通记忆——上一轮的 carry、上一个请求
的终态——用 `carry=` 声明：

```python
head = pllm.RequestState(
    {"w": lambda m: (m.hidden_size, m.hidden_size)},
    optimizer=pllm.AdamW(lr=1e-3),
    carry={"carry": lambda m: (m.hidden_size,)},   # 无梯度/动量槽，不被 Adam 碰
)
```

写：把训练方法的输入包一层 `pllm.carry`，例如
`h: torch.Tensor = pllm.carry(pllm.hidden(0), state="head", field="carry")`——发布时
把该输入每个 lineage 最后一行的值写进 carry 字段。读：`pllm.parameter("head",
"carry")`，读到的是**上一步发布**的值（输入版本）。

**第一步读到什么？** 零。carry 页在准入时零初始化，这不是 bug——正是
「输入版本」语义的直接推论：第 N+1 步看到第 N 步发布的值。

这一个机制组合出教程里的三种循环：

- **跨步**：每步捕获、下步读取（教程 05 的「下一步目标」）；
- **跨请求**：route 键让 lineage 跨请求存活，memory 随之留存
  （教程 04 的会话循环、教程 05 的「上一请求终态」）；
- **步内多层**：不需要 memory——`@pllm.custom_forward` 直接声明
  （见 [Overlay 与重放](overlays-and-replay.md)）。

## 状态会不会爆显存：state_page_budget_bytes 预算

所有状态页都住在 PageBundleArena 里，arena 的容量由
`ExecutionOptions.state_page_budget_bytes` 设定。默认 `0`：取后端观测到的可用
容量。显式给值就是给 arena 划一个固定预算（基准就这么做，见
[基准解读](benchmarking.md)）。

这是 **arena 预算，不是进程总显存**——模型权重、KV 池、scratch、分配器
预留都不在内。别把「state 预算够用」读成「GPU 不会 OOM」。

**超预算在哪一步失败？** 两个边界，各有明确口径：

- **编译时**：显式预算超过后端观测到的可用内存，特化直接抛
  `ValueError`（"requested state capacity exceeds available backend
  memory"）；预算内但计划本身放不下时，Rust 规划拒绝不足的状态容量。
  失败发生在任何分配之前。
- **运行时**：活跃 lineage 的准入页耗尽，准入抛 `RuntimeError`（"PageBundle
  admission capacity is exhausted"），并计入诊断里的
  `admission_capacity_failures`。这个计数**只覆盖被拒的准入**——不含迁移
  失败，也不含非法输入。

路由用户特别注意：休眠 lineage 仍占准入容量（见上文路由键一节），所以
「route 基数 × 每 lineage 字节」才是你的稳态占用，不是并发请求数。

`ExecutionOptions.execution_peak_budget_bytes` 约束 pllm 的执行峰值：状态页、任务工作区与存活结果、anchor 缓冲以及 CUDA Graph 池预留。模型权重与 KV 池不在内。框架在 meta/fake
输入上执行实际张量步及前后向，按 512 字节追踪存储，不初始化 CUDA；负载不维护
字节公式。每个任务和模型的捕获池预算用 `cuda_graph_pool_budget_bytes`；它不是所有捕获池的合计上限。默认 `None` 按负载追踪任务预留，模型池默认 64 MiB；exact decode overlay 另按形状计入工作区。宿主有限训练声明 `max_input_rows`，
并为奖励等宿主选择提供最大形状样例；详见[资源事实](../dev/execution-internals.md#资源事实与内存预算)。

## 我想看状态到底占了多少：state_usage

`llm.diagnostics().state_usage` 给出状态页的冻结快照：预留字节、逻辑与物理载荷、填充和页所有权。
逻辑字节把每个保留的版本都计入，物理字节对写时复制共享的字段页只计一次，所以两者可以相差很大。
各字段的完整口径与状态的生命周期见[贡献者文档](../dev/state-internals.md)。

精确边界（零梯度语义、empty 选择、捕获 detach、外部导入通道）见
[创作契约](../reference/authoring-contract.md#训练与梯度)。

`llm.diagnostics().numeric_health` 返回当前存活的已发布字段的精确统计：resource、
版本 ref、field ID、L2 norm、max-abs 和 nonfinite_count。归约在设备上的现有快照进行，
只有查询时才读取主机标量，不增加每步或发布路径的同步。非有限的 norm/max-abs 为 `None`，
空字段二者为零。退役字段不保留历史统计；需在请求存活时查询。
