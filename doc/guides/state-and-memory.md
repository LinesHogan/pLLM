# 状态与记忆

> **先决条件**：[写一个负载](write-a-workload.md)；教程
> [03](../tutorials/03_request_lora.ipynb)、[04](../tutorials/04_loops.ipynb)
> 演示了本文的每个概念。
>
> **本文回答**：我有值要跨步、跨请求保存，该怎么声明？该选哪种状态？

## 先选作用域

| 你想要 | 声明 |
| --- | --- |
| 每个请求（或每组请求）一份独立状态 | `pllm.RequestState(...)` |
| 整个引擎实例共享一份状态 | `pllm.InstanceState(...)` |
| 只读、永不训练的常数（如 SAE 编码器） | 任意一种 + `optimizer=None`（**冻结状态**） |

```python
class Probe(pllm.Workload):
    # 实例作用域 + 冻结：准入时初始化一次，之后任何任务（包括观测）可读，永不被更新
    encoder = pllm.InstanceState(
        {"weight": lambda m: (4, m.hidden_size)},   # 形状回调拿到模型事实
        dtype=torch.float32,
        initialize=lambda tensors, generator: tensors["weight"].copy_(torch.eye(4)),
        optimizer=None,                              # 可省略，默认即 None
    )
```

形状可以是 lambda（拿模型事实 `m`）或 `pllm.Buffer(...)`（overlay 场景按目标
输入/输出宽度解析）。状态声明可以写在类上、实例上，或形状依赖配置时写成
带类型的 property。

## 隔离键：谁和谁共享

`RequestState` 默认按请求 ID 隔离。加 `keys=(pllm.RequestKeyField.ROUTE,)` 后，
由**你的宿主代码**给每个 prompt 算一个路由值：

```python
session = pllm.RequestState({"proj": lambda m: (m.hidden_size, m.hidden_size)},
                            optimizer=pllm.AdamW(lr=1e-3),
                            keys=(pllm.RequestKeyField.ROUTE,))

llm.generate(prompts, params, routes=(route_of(p) for p in prompts))
```

路由值相同的请求**加入同一 lineage**：训练成果共享；不同 route 完全隔离。
route lineage 在最后一个请求退役后休眠存活（仍占准入容量 —— `max_num_seqs`
要留出余量），会话关闭时退役。框架从不解释路由值：正则、判别器分数、租户 ID
都行。缺路由值的请求在准入时被拒。

## 参数怎么读：按行就位

`Parameter("state", "field")` 输入带**领先的行轴** —— 每行拿到自己 lineage
的版本。所以矩阵运算一律用按行形式：

```python
torch.bmm(h.unsqueeze(1), w).squeeze(1)     # (rows, H) @ (rows, H, H) → (rows, H)
```

不要裸 `matmul`/`conv1d`。冻结状态的读取也一样按行。输入版本语义：一步之内的
所有读取看到的是**步开始时 pin 住的版本**；`StateVersion` 输入给出该版本号，
常用作"是否训练过"的门（`versions > 0`）。

## memory 字段与 Capture：跨步记忆的极简形式

优化器状态（参数 + Adam 槽）之外的普通记忆，用 `memory=` 声明：

```python
head = pllm.RequestState(
    {"w": lambda m: (m.hidden_size, m.hidden_size)},
    optimizer=pllm.AdamW(lr=1e-3),
    memory={"carry": lambda m: (m.hidden_size,)},   # 无梯度/动量槽，不被 Adam 碰
)
```

写：在训练输入上加第三个注解标记 `pllm.Capture("head", "carry")` —— 发布时把
该输入每个 lineage 最后一行的值写进 memory。读：`Parameter("head", "carry")`，
读到的是**上一步发布**的值（输入版本）。页零初始化，第一步读到零。

这一个机制组合出教程里的三种循环：

- **跨步**：每步捕获、下步读取（教程 05 的"下一步目标"）；
- **跨请求**：route 键让 lineage 跨请求存活，memory 随之留存
  （教程 04 的会话循环、教程 05 的"上一请求终态"）；
- **步内多层**：不需要 memory —— `Recurrence` 或 overlay replay 直接声明
  （见 [Overlay 与重放](overlays-and-replay.md)）。

## 生命周期

状态不因 Python 对象析构而回收：PTTS 的 pin 与完成协议管理其寿命。
写入只在训练/overlay 迁移的私有副本上进行，发布是原子的；失败保留原根可重试。
观测与预测读到的是 pin 住的精确版本，永远不是"最新值"。

精确边界（零梯度语义、empty 选择、捕获 detach、外部导入通道）见
[创作契约](../reference/authoring-contract.md#训练与梯度)。
