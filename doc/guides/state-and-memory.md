# 状态与记忆

> **先决条件**：[写一个负载](write-a-workload.md)；教程
> [03](../tutorials/03_request_lora.ipynb)、[04](../tutorials/04_loops.ipynb)
> 演示了本文的每个概念。
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
class Probe(pllm.Workload):
    # 实例作用域 + 冻结：准入时初始化一次，之后任何任务（包括观测）可读，永不被更新
    encoder = pllm.InstanceState(
        {"weight": lambda m: (4, m.hidden_size)},   # 形状回调拿到模型事实
        dtype=torch.float32,
        initialize=lambda tensors, generator: tensors["weight"].copy_(torch.eye(4)),
        optimizer=None,                              # 可省略，默认即 None
    )
```

为什么这样能工作：冻结状态不是「另一种东西」，它就是状态的冻结形态——
准入时初始化、任何任务（包括观测）可读、可通过外部通道导入、永不更新。
没有优化器，就没有梯度/动量槽，字段布局本身就是全部存储。

形状可以是 lambda（拿模型事实 `m`），也可以是 `pllm.Buffer(...)`（overlay
场景按目标输入/输出宽度解析）。声明可以写在类上、实例上；形状依赖配置时，
写成带类型的 property。

## 我想让一组请求共享训练成果：路由键

`RequestState` 默认按请求 ID 隔离——每个请求一条独立 lineage，互不可见。
如果你心里想的是「同一租户/同一类 prompt 应该共享训练成果」，那要显式
声明路由键：

```python
session = pllm.RequestState({"proj": lambda m: (m.hidden_size, m.hidden_size)},
                            optimizer=pllm.AdamW(lr=1e-3),
                            keys=(pllm.RequestKeyField.ROUTE,))

llm.generate(prompts, params, routes=(route_of(p) for p in prompts))
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

## 我想读参数：一律按行

`Parameter("state", "field")` 输入带**领先的行轴**——混合批次里每行拿到
自己 lineage 的版本。所以矩阵运算一律用按行形式：

```python
torch.bmm(h.unsqueeze(1), w).squeeze(1)     # (rows, H) @ (rows, H, H) → (rows, H)
```

不要裸 `matmul`/`conv1d`：它们不感知行轴，混合 lineage 的批次会算错。
冻结状态的读取也一样按行。

读到的是哪个版本？**输入版本**：一步之内的所有读取看到的都是步开始时
pin 住的版本，永远不是「最新值」。`StateVersion` 输入给出该版本号，常用
作「是否训练过」的门（`versions > 0`）。

## 我想跨步记点东西：memory 字段与 Capture

优化器状态（参数 + Adam 槽）之外的普通记忆——上一轮的 carry、上一个请求
的终态——用 `memory=` 声明：

```python
head = pllm.RequestState(
    {"w": lambda m: (m.hidden_size, m.hidden_size)},
    optimizer=pllm.AdamW(lr=1e-3),
    memory={"carry": lambda m: (m.hidden_size,)},   # 无梯度/动量槽，不被 Adam 碰
)
```

写：在训练输入上加第三个注解标记 `pllm.Capture("head", "carry")`——发布时
把该输入每个 lineage 最后一行的值写进 memory。读：`Parameter("head",
"carry")`，读到的是**上一步发布**的值（输入版本）。

**第一步读到什么？** 零。memory 页在准入时零初始化，这不是 bug——正是
「输入版本」语义的直接推论：第 N+1 步看到第 N 步发布的值。

这一个机制组合出教程里的三种循环：

- **跨步**：每步捕获、下步读取（教程 05 的「下一步目标」）；
- **跨请求**：route 键让 lineage 跨请求存活，memory 随之留存
  （教程 04 的会话循环、教程 05 的「上一请求终态」）；
- **步内多层**：不需要 memory——`Recurrence` 或 overlay replay 直接声明
  （见 [Overlay 与重放](overlays-and-replay.md)）。

## 状态会不会爆显存：state_memory_bytes 预算

所有状态页都住在 PageBundleArena 里，arena 的容量由
`ExecutionOptions.state_memory_bytes` 设定。默认 `0`：取后端观测到的可用
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

## 我想看状态到底占了多少：state_usage

`handle.diagnostics().state_usage` 给出 arena 的冻结快照：预留、逻辑/物理
载荷、填充、页所有权与待收 receipt 等字段。它是从原始 arena 根、字段槽、
私有写与 receipt **推导**出来的视图，自己不保留这些所有者。

读它之前，先建立三个正确预期：

- **逻辑字节 ≠ 物理字节**。`logical_live_bytes` 把每个保留/私有版本都计
  入；`allocated_payload_bytes` 对 COW 共享的字段页只计一次。两者的差
  正是字段级 COW 省下的部分，不是账目错误。`reserved_bytes` 的填充包含
  所有容量对齐/余量。
- **页所有权类别不是 GPU 活动度量**。空闲、最新根、仅私有、仅退役是互斥
  的所有权分类（最新根优先于私有，再优先于退役引用）；被 pin 的版本数
  包括内部写/发布持有与待释放。待收 receipt 包括已结算但尚未收集的发布，
  不只是未完成的 GPU 工作。
- **快照本身什么都不做**。`usage()` 不收集、不同步、不选版本；它是调用
  时刻的冻结快照，不是采样峰值，也不是一份独立的分配台账。

这些口径的完整表述在[当前架构](../design/architecture.md)的「运行时边界」
一节。

## 生命周期：谁回收状态

### 跨进程借用与独立快照

短期消费同一 GPU 上的精确状态版本，可以使用低层 `PageBundleArena.send_field`
与 `receive_field`。发送者必须持有原 arena 签发的 `PagePin`，提供者必须在
原计划中声明 `state/shared_cuda@1`。接收者先验证自己持有的同一 `PlanBundle`，
再接收字段；签名、布局、设备和容量不能因跨进程而放宽。每个字段借用使用
独立完成通道。接收者不获得页的所有权，也不获得修改或发布权限。

`BorrowedPage` 返回 native tensor，只读是同一可信程序内的使用约定，不是
OS 写保护或进程身份认证。用完全部别名和依赖它们的计算后，接收者在原接收
stream 上 `close()`；它同步该 stream 后发回精确完成凭证。其他 stream 的
计算须先归并到该 stream。发送者 `release_pin` 只签发释放 receipt，原 arena
仍保留页，直到本地完成事件和全部远程读凭证满足；较新版本发布不会覆盖旧页。
有界容量耗尽时拒绝新写，不能驱逐尚未完成的远程读者。

`snapshot` 则复制出独立 tensor，适合要在 arena/worker 关闭后继续保留的结果。
普通辅助输入和独立 completion 输出沿用 `SharedTensor`；它们不是版本页授权。
页授权现在仅支持 CUDA 只读，CPU 本地 arena、CPU SharedTensor 和本地私有写
仍保留。没有远程私有写或 CPU 页授权接口。

坏凭证、断连或等待超时都不能作为完成：原 owner 保留相关资源。会话集成必须
停止新工作，终止全部参与进程后丢弃实例；当前低层 arena 不提供自动崩溃恢复。
进程退出码不能替代字段完成凭证，owner 崩溃后也不能依靠遗留 IPC mapping 继续服务。

状态不因 Python 对象析构而回收——PTTS 的 pin 与完成协议管理其寿命。写入
只在训练/overlay 迁移的私有副本上进行，发布是原子的；失败的发布保留原根，
可以重试。观测与预测读到的是 pin 住的精确版本，永远不是「最新值」。

**COW 什么时候真拷贝？** 字段级：一次迁移只给它要**写**的字段分配私有
页，未改动的字段与旧版本共享物理槽。所以「发布一个新版本」的成本正比于
被写字段，不是整个状态——这也是上面 `state_usage` 里逻辑与物理字节可以
相差很大的原因。

精确边界（零梯度语义、empty 选择、捕获 detach、外部导入通道）见
[创作契约](../reference/authoring-contract.md#训练与梯度)。
