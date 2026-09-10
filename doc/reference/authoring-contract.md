# 负载创作契约

本文是 PLLM 负载创作的**完整契约**：每一条能力边界与禁止事项都在这里。
如果你是第一次写负载，请先做[教程](../tutorials/00_quickstart.ipynb)和
[使用指南](../guides/write-a-workload.md)；要把需求交给编码代理，见
[代理指南](../guides/agent-workloads.md)。本文按主题组织，供查证用。

普通负载使用 `import pllm` 的普通张量方法。[Dummy](../../pllm/workloads/dummy.py)、
[OverRIDE](../../pllm/workloads/override.py)、[In-Place-TTT](../../pllm/workloads/inplace_ttt.py)
和 [ReFT](../../pllm/workloads/reft.py) 使用这个接口；[Recirculation](../../pllm/workloads/recirculation.py)
使用普通的 `Recurrence` 声明；[TTRL](../../pllm/workloads/ttrl.py) 使用
`CompletionWorkload`；FTTT 和 GTTA 使用结构化的 `transaction` 绑定；
[TLM](../../pllm/workloads/tlm/workload.py) 使用 `CausalAdapterWorkload`；
[ESamp](../../pllm/workloads/esamp_request.py) 使用普通具名参数、预测与 loss，
并在同一张图上做显式分页/捕获替换；Output LoRA 使用普通 `Batch` 输入与多步 SGD。
全部十一个内置负载都使用各自适用的共享创作入口，构造同一张原始图，
使用同一套执行所有者。独立的 [Huber 示例](../../examples/huber.py) 用普通
Tensor 方法表达请求内低秩学习与持久 AdamW，不需要改动核心。

## 张量方法与状态

```python
import torch
import pllm

class CenteredHidden(pllm.Workload):
    @pllm.on(pllm.Phase.OBSERVE)
    @pllm.hook(0)
    @pllm.export(name="centered_hidden", max_rows=2, max_cols=16)
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden - hidden.mean(dim=-1, keepdim=True)
```

`hook` 声明的是受支持的块/模型输出捕获，不是任意的 eager PyTorch 钩子。
单个普通 Tensor 参数继承该来源；多个来源用 `Annotated[Tensor, ...]` 标注，
来源标记有 `Hidden`、`Samples`、`BaseLogits`、`SampledTokens`、
`Parameter(state, field)`、`StateIndex(state)`、`StateVersion(state)` 和
`Result(operation)`。输入歧义在绑定时报错；参数名不做猜测。
注解必须是求值后的真实类型，绑定器从不求值字符串注解。

`InstanceState` 和 `RequestState` 声明单个 `shape=` 或具名字段、dtype、
优化器和可选初始化器。它们可以声明在类上、实例上，或形状依赖配置/模型维度时
声明为带类型的 property。当 `optimizer=None` 时字段就是布局本身：准入时初始化、
任何任务（包括观测）都可读、可通过外部通道导入、永不更新 —— 这是同一种状态的
冻结形态。请求状态默认按真实请求 ID 隔离。它不因 Python 对象析构而回收：
生命周期由原始 PTTS pin 与完成管理。实例参数就是声明的形状；请求参数会多出
一个操作行轴。普通实现显式打包这些视图 —— 这**不是原生分页 GEMM**。
输入与参数视图是借来的只读值；普通数学不得原地修改它们。
形状/dtype 声明不证明任意 Python 副作用是自由的。
优化器写入仍走原始私有版本。

## 训练与梯度

`@train(state=..., update=...)` 绑定一个标量 loss 和一个可选的状态资格选择器；
`@on(phase, rows=...)` 独立选择它的前向行。选择器是具名 Tensor 方法，
只使用该操作已注解输入的子集，返回其原始行轴上的布尔掩码，并成为同一签名图中的
普通任务 —— 不是执行期回调。旧的 `train(select=...)` 选项已移除；
OverRIDE 用 `on(..., rows=...)` 表达它原先的训练子集。
不带 `update` 时，更新前向行所代表的状态。显式的 update 掩码授权每个选中行的
原始状态（任一选中行授权共享状态一次），与前向或梯度资格相互独立。
空的 update 选择保留所有字段与版本；被授权的零梯度仍会推进 Adam 步数、
动量与衰减，包括空前向调用。标量函数自己拥有归约；绑定不会发明归一化。

`@train(state=..., reduction="state_mean")` 改为返回每个原始选中状态一个均值 loss。
绑定器对它们求**和**再微分，而不是跨状态求均值；loss 与优化器在原始图中
分阶段成为独立任务，保持声明的时机。这个形式目前要求无条件的
forward/update 域。`StateIndex` 给出每行在本次调用内的状态索引，用于归约；
`StateVersion` 给出该行的原始输入 pin 版本 —— 永远不是 latest 或优化器步数。
两者都不可微。显式的 `RequestState.keys` 仍是原始隔离键；省略则按请求 ID。
ESamp 保留其仅请求 ID 的后端限制。

状态还可以携带普通 **memory 字段**：`InstanceState(..., memory={"carry": shape})`
声明跟随训练迁移、但不参与优化器展开的每 lineage 张量（没有梯度/动量槽，
永不被 Adam 更新）。memory 读取（`Parameter(state, field)`）看到的是步开始时
pin 住的输入版本 —— 上一步发布的值 —— 所以训练操作消费的是它上一步写下的内容。
写入是按行捕获：训练输入上的第三个注解标记
`Annotated[Tensor, source, pllm.Capture(state, field)]` 在同一迁移与发布中，
把该输入每个 lineage 最后一个选中行的值写进它的 memory 字段。捕获目标必须是
任务自己的训练状态，要求已声明的 memory 字段，并且是 detach 的。页零初始化，
所以第一步读到的是零。跨请求共享由已有的宿主导出/导入通道或共享 route
lineage 组合而成；memory 本身永远不逃离其 lineage 的 pin。

`@differentiate(inputs=("features",), parameters=False)` 在该操作保留选中输入的
导数边，同时冻结参数导数。上游方法用 `parameters=True`，`@train` 通过声明的边
消费它们的 `Result` 输入。其他输入被 detach。这是单次调用内的一阶 autograd，
带一个优化器状态和一个训练终点 —— 不是穿透 vLLM 或跨越 replay/发布边界的
全模型反向传播。同一个共享状态可以在另一个操作可微；`parameters=False` 不会
永久冻结它的优化器。`train(state=...)` 授权该状态的更新，包括其既有的 Adam
步数/动量/衰减语义。显式标记的外部浮动输入的梯度以
`result.input_gradients[(operation, argument)]` 返回，按该操作的行序。
中间结果边直接传播导数，不需要单独的梯度 DTO。
`differentiate(input_rows="dx", parameter_rows="dw", ...)` 独立掩码声明的导数边，
不改变前向值。选择器使用原始行，不是前向压实后的位置。对共享状态，按行的
参数导数要求 `Parameter(state, field, per_row=True)` —— 一个逻辑
`[rows, *parameter_shape]` 视图，由 `bmm` 等按行数学消费。请求参数本就带这个
行轴。未分批的共享参数配参数梯度掩码会被拒绝；没有隐藏的形状转换或替代 backward。
前向选中的非 loss 方法保留行轴并接受空批次；其结果用零值恢复到原始行位置。
loss 方法只收到选中行，空前向集不调用。有界导出有各自的行限制，
不接受这些行选择器。采样 token 谓词只在采样完成之后、签名更新之前执行；
更早的反馈读取、有效果的辅助读取和更新后读取都会被拒绝。
`result.parameter_gradients` 把原始输入 `StateRef` 映射到扁平梯度，即使该状态
被排除在发布之外。多字段扁平化是显式的检查成本；
`result.parameter_gradient_fields` 按声明的字段序借用原始的自持有梯度张量，
不做 eager 的按请求拼接。捕获的结果在后续 replay 间保持独立存储。
没有发布的 loss 工作在成功完成时计数。所有导数都保持调用局部、一阶。
这是普通 eager Tensor 执行，不是稀疏 backward 或原生分页下沉：前向选择先于
请求参数打包，共享的按行参数保留其零步长视图。连续前向行借用输入/掩码切片；
不规则行打包。整行结果无需 scatter；部分结果用一个零填充目标恢复原位。
导数掩码作用于每个消费者一条独立 PyTorch 梯度边，不需要参数尺寸的前向副本，
也不改变另一个消费者的梯度。原始选择拥有其共享只读恒等索引。
结果张量共享本次调用的原始 `(task, output)` 存储。Rust 规划它们的最后使用 —
— 包括下游声明的梯度消费者 —— 并在原始物理任务上签名释放 ID。Python 只
应用这些释放；导出的值存续到完成。没有单独的 TensorFrame 结果缓存或 Python
活性规划器。压实可能同步，autograd 仍可能为 detach 的行保存中间量。
完整的 scratch/保存张量规划和选择性 backward 融合仍未完成；不作捕获或
零分配声称。完成/取消释放调用数据与保留的反向图；失败的结算只保留到原始
receipt 能结算为止。

## 采样引导、logits 与导出

`@guide_sampling(beta=...)` 返回过滤后的 hidden 预测，可由 `@on(..., rows=...)`
选择。绑定把它们关联到原始采样器链接（包括重排的链接）；不要求词表维度。
ESamp 用它做版本感知引导和普通的 Tensor loss/初始化；其五字段 Adam 布局与
初始化种子推进保持不变。

`@add_logits` 和 `@train` 把过滤前加法连接到真实采样反馈。
`@export` 拥有有界、按步长门控的 CPU 输出；`Config("config.layer")` 等属性路径
在装载时绑定不可变配置。普通负载不需要 `build`、输入字典、`prepare/output`、
字段 ID、digest 或显式发布节点。

局部函数替换就是 `dataclasses.replace(workload, transform=fn)`；见
[可运行示例](../../examples/centered_hidden.py)。用户不分配实现 ID。
冷绑定器签名代码、受检依赖、配置、初始化器与替换，并为绑定的实例快照
Python 函数。反射式数学或未受检的依赖需要显式原生绑定。
EngineCore 多进程按**引用**传输负载（类坐标 + 冻结字段 + 模块摘要），
worker 重新导入并重建，安装时其编译签名必须与驱动端签名计划逐字节相等；
模块级函数替换同样按引用走，签名校验兜底。函数体从不序列化。

只有 `pllm/usage.py` 允许冷创作反射/类型擦除。严格类型检查仍覆盖该模块，
运行时执行不检查签名。旧的 `TensorWorkload`、`TensorBuilder` 和
`TensorFunction` 公共 API 已移除。任意模块锚点、捕获和结构化宿主事务仍需
高级绑定。

## 请求 Overlay

`@on(Phase.MODEL_UPDATE)` 和 `@overlay(state=..., targets=..., weight=...)` 把
一个加性参数 overlay 与它的发布绑定在一起。声明不带优化器的 `RequestState`：
固定 `shape=` 使用默认的 `value` 字段；具名 `Buffer` 字段可以按每个 target
解析 `(input_size, output_size)`。Buffer dtype 默认取目标模型 dtype；显式整数
buffer 保留自己的 dtype。`Operand.INPUT/TARGET/WEIGHT` 命名捕获的张量；
`Parameter(state, field)` 命名私有 buffer 视图。只有这些被授权的 buffer 可以
被原地修改，操作数永远不行。方法返回
`TensorUpdate(output, changed=..., update_count=..., trained_rows=...)`。
可选的 `metrics` 属于算法，不属于状态或 replay 协议。

默认算子是 `DENSE_ADD`，应用 `weight` 字段。传
`operator=ParameterOverlayOperator.LOW_RANK_ADD` 和 `factors=("a", "b")` 加
有限 `scale`，则应用 `scale * b @ a`：down 因子是每个 target 的
`(rank, input_size)`，up 因子是 `(output_size, rank)`。算子选择与请求作用域
正交：同样的发布、pin 与隔离规则对两者都适用。targets 可以命名任何可定位的
投影；后端目前定位 MLP down 与 attention q/v 投影。

`prefill`/`decode` 决定更新资格。被禁用的阶段只应用既有 overlay。
`replay=ConditionalLayerReplay((22, 23), ...)` 命名一组签名的升序层集合：对选中
token 按序以发布后的状态重跑这些层；每个被重放的层覆盖自己当前的 KV 槽，
集合外的层保留正常前向的 KV —— 这是声明过的近似，不是静默近似。
每个被重放的层必须被签名的 overlay target 覆盖。replay 复用原始发布版本，
且只对 `changed` 的状态运行：仅尾部 buffer 变化不必触发 replay。
共享绑定拥有 target 覆盖、形状/设备检查、根选择与结果出处。结果的 `entries`
暴露原始请求/before/after 引用和按 target 的更新。普通效果支持请求作用域的
稠密与低秩 overlay，不支持激活或任意 replay 组合。

## 外部 Overlay

overlay 的字段值不必来自引擎内更新：任何宿主计算 —— 读请求上下文的
超网络、加载的 checkpoint —— 都可以产出它们。声明同样的 `RequestState`
（或 `InstanceState`）buffer 和一个 `Overlay` property，命名状态、targets、
算子、因子与 scale：

```python
@dataclass(frozen=True)
class ShineStyleLoRA(pllm.Workload):
    @property
    def factors(self) -> pllm.RequestState:
        return pllm.RequestState({
            "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
            "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
        })

    @property
    def adapter(self) -> pllm.Overlay:
        return pllm.Overlay("factors", targets, operator=ParameterOverlayOperator.LOW_RANK_ADD,
                            factors=("a", "b"), scale=0.5)
```

产出的张量作为 `StateTensorUpdate(values, keys)` 经
`generate(..., auxiliary=..., prompt_indices=...)` 送入；请求替换按原始 prompt
索引作键，且只导入到版本零的请求状态一次。共享绑定校验布局、拷入私有根并在
`request.prepare` 发布 —— 在请求的第一次前向之前 —— 所以混合批次中的每一行
读到的是自己请求的精确 pin，导入的零因子精确复现基座模型。应用、发布、隔离与
退役与优化器产出的 overlay 完全相同；只有产出地点不同。与激活检查点一样，
负载通过 `resource_facts` 声明导入的任务工作区（拷贝不需要自有张量）。

## 路由键

请求状态键是签名声明；键值可以由用户计算。给 `RequestState` 的 `keys` 加上
`RequestKeyField.ROUTE`，并传 `generate(..., routes=(...))` —— 每个 prompt 一个
路由值，由你自己的宿主代码计算（一个正则、一个判别器分数、一个租户 id；
框架从不解释它）。键重合的请求**加入同一 lineage**：任何一个请求训练出的
适配器对全部生效，无论在同一批次内还是跨步。不同 route 保持隔离。route
lineage 在其最后一个请求退役后仍然存活 —— 休眠，仍被其键 pin 住 —— 所以
适配器池在整个会话中存续；`max_admitted_lineages` 约束它，会话关闭时退役休眠
lineage。缺少 route 值的请求在准入时被拒。辅助批输入（`WeightedCEBatch`）按行
解析自己的键投影，所以带键训练在共享 lineage 上照常工作。

## 激活检查点

声明普通的 `InstanceState` 或 `RequestState` buffer 和一个 `Activation`
property，命名状态、层、首个/末个 prefill token 与支持的算子。
`LOW_RANK_AFFINE` 每层使用 weight/bias/rotation 字段；`ADD` 使用一个 FP32 向量。
Buffer 形状回调收到激活的输入/输出宽度。以原始 `pllm.StateTensorUpdate` 作为
`auxiliary=` 提供。共享绑定校验检查点、选择根、拷贝私有字段并在模型执行前发布。
实例替换可重复；请求替换使用原始 prompt 键，只导入到版本零的请求状态。
缺失的更新不发布，既有推理 pin 继续看到它们精确的旧版本。见维护中的 ReFT
负载；它不再实现 prepare/copy/publish 任务。

## 逐 Token 循环

在 `pllm.Workload` 上声明 `pllm.Recurrence(source_layer=..., destination_layer=...,
alpha=..., beta=..., ramp_tokens=..., enabled=...)` 作为类/实例属性或带类型的
property。负层索引对真实模型解析。`beta=None` 使用互补的目标权重；源在混合前
做范数匹配。这是既有的独立逐 token prefill 效果：正常 logits 保留，replay 更新
选中 token 的层 KV，decode 行只正常运行。不需要 RequestState、hidden 捕获、
外部训练输入、build 或 prepare 方法。把这个效果与普通训练/激活操作混合尚不支持，
在绑定时失败；纯宿主执行没有模型 replay 能力。

## 加权批量训练

[Output LoRA](../../pllm/workloads/lora.py) 声明具名字段状态（`SGD(lr)` 与
初始化器）、一个普通的 `@add_logits` 预测，以及一个 `AFTER_SAMPLE` 方法（
`@train(state=..., reset="reset", steps=Config(...),
reduction="weighted_state_mean_then_mean")`）。reset 方法接收具名局部参数张量和
一个生成器。reset 按训练批次进行，区别于准入初始化；两个函数都被冻结并与普通
数学一起签名。SGD 保留原始状态字段，没有 Adam 槽。

训练输入用 `Annotated[Tensor, Batch("hidden"|"base_logits"|"labels"|"weights")]`
和 `Parameter(state, field)`。它们消费原始 `WeightedCEBatch`，不是新的 DTO。
共享输入用 `pllm.WeightedCEBatch(hidden, logits, labels, weights)`；独立输入用
`pllm.WeightedCEBatch.for_prompts(hidden, logits, labels, weights, prompt_indices)`，
同样的索引传给 `generate`，可选 `sample_idx=0`。两者都借用所供张量，不拷贝。
绑定对齐原始键，拒绝出界输入，掩码非正权重/词表外标签。每个选中状态必须有
有效行。训练张量有领先的选中状态轴，共享状态时是单例。每个状态返回一个归一化
loss；绑定在每个 SGD 步微分它们的**均值**。这不是 ESamp 的独立按状态 Adam 归约。
缺批次意味着没有 reset、没有私有写、没有发布。成功训练在所有步之后发布一次；
取消/失败保留原始 pin 状态。归一化/reset 在原始图的 `__train_prepare` 任务中
于采样前运行，不在调用准备期。其结果边喂给 `train`；同一帧拥有归一化张量直到
消费后释放。LoRA 签名准备、预测与训练工作区以及预备/输出张量尺寸，特化与绑定
检查使用同一公式。签名事实不足则在绑定失败。公开提交、反馈传输与 worker STAGE
对原始辅助张量/描述符做预检，然后才做共享存储导出、映射或设备拷贝；dtype 来自
原始状态布局，标签为 int64。形状检查不查看张量内容，也不替代之后的有限值/选择
检查。reset 训练与激活导入完全覆盖其私有字段，所以它们的绑定用
`copy_fields=()` 跳过把旧字段拷进新页。默认 `copy_fields=None` 拷贝全部私有
字段；显式有序子集只拷那些字段。内置 Adam 绑定保留参数、步数与动量，但不保留
旧梯度 —— 更新会完全覆盖梯度（包括每个 TLM 因子组）。没有单独的 `replace`
执行模式。这里使用同一 arena、pin 与 receipt；调用方必须在发布前完全覆盖被省略
的私有字段，且不得读取其未初始化内容。私有页容量不变。这个入口目前支持 reset
SGD 与加权批次，不支持持久动量、任意外部 DTO、推理行选择器或跨步 autograd。
`SamplerWorkload` 与 `LoRAData` 已移除；没有平行的原生创作路径。

普通 `Workload` 还暴露带类型的 `implementations` 与 `operations` 钩子，用于对其
原始签名任务做显式原生替换 —— 不是另一张图/运行时。ESamp 与 LoRA 用它们接入
既有的捕获与分页内核。普通数学可以局部替换；不兼容的替换在选择特化内核时被拒
（`native_methods` 冻结参考代码/源码/效果指纹；`require_native_methods` 在绑定时
检查它们，包括对注解的原地编辑）。这不是把任意修改过的数学自动下沉到特化内核。
继承来的数学 `resource_facts` 公式是针对内置数学的。修改过的方法必须提供自己的
成本声明，而不是静默继承那个信封 —— 即使在普通张量路径上。检查在特化与绑定时
运行；显式成本覆盖不授权不兼容的原生内核。

## 高级数据流

自定义高级数据流：继承 `a.BackendWorkload[Data, Result]`，从 `declare` 返回原始
图/布局/附着，带原生 `bind_math` 与 `backend_requirements`。旧的 `a.Workload`
build 包装已移除。新的局部操作获得基于负载/名字的 ID，除非绑定了显式实现。
持久布局命名空间与实现标识不做机械重命名。

`Call["output"]` 命名原始任务输出。它建立依赖，不是 Python 执行顺序。
`When` 命名语义时机窗口；锚点与排序仍由 Rust 选择。输入、状态与输出名必须
描述数学函数的真实访问。不做函数体检查或自动追踪。带类型的 `Data` 记录保存
这些函数使用的调用局部中间量；声明一个端口不会自动提取或赋值 Python 属性。

Hidden 需求从声明推导。形状/dtype 回调只在模型事实可用后运行；需求发现不猜
维度、不分配参数。`state.read()` 是逻辑输入版本引用。
`state.version(publication)` 显式读生产者的版本，永远不是 latest。

## 编写原生数学

高级 `BackendWorkload.bind_math(facts, plan)` 返回原始 `MathBinding`。其
`FunctionMath` 提供初始化、输入行选择、准备与结果构造；负载不构造执行器。
结构化创作入口自己组装这个绑定。

原生函数接收带类型的 `Data` 与 `TensorBatch`。后者持有原始选择/出处与借来的
张量操作数，不是 arena 或发布 API。只读操作数按契约不可变，不得逃出本次调用。
只有显式选中的迁移根可写。流排序、捕获、私有分配、回滚与发布仍留在原始
执行层。

返回 `TaskOutput`，携带数学梯度、引导与有效工作量计数。空梯度元组跳过优化器
更新；零梯度元组仍推进步数、动量与权重衰减。多参数组时，梯度按状态优先、
然后声明的参数组序排列，一次发布覆盖所有组。直接迁移原地修改被授权的私有
视图，使用单独的 `b.publish`；replay 消费该原始发布与变化状态选择。
优化器支持的迁移在发布必须延迟时也用 `b.publish`，如 TLM 与 TTRL。
梯度跳过时没有私有分配或发布。

`InvocationOutput` 暴露原始引导与发布，允许在不拥有生命周期的情况下构造结果。
`FunctionMath` 对输入/数据/结果类型泛型：完成与宿主负载不转换成纯张量 DTO。
宿主事务继续使用既有宿主服务与 receipt 协议，不是第二个状态数据库或模型推理
回调。

结构化宿主事务用 `pllm.transaction(name, facts, config=config,
state_factory=make_state, service=service, update=update, query=query,
publish=publish)`。三个普通带类型函数接收同一原始上下文。共享绑定声明
update-before-query-before-result 依赖与宿主时机窗口；调度仍由 Rust 完成。
`publish` 只构造结果：原始协调器独自调用服务 commit 并结算其 receipt。
状态与服务对象按身份保留，保持其既有 prepare/snapshot/restore/output 与
begin/commit/abort 契约。不引入 TensorFrame 转换、拷贝的队列或第二个状态
所有者。这些是显式原生函数绑定，不是自动微分或冻结的 Tensor 方法；改动其
原生辅助实现需要重新绑定。见 [FTTT](../../pllm/workloads/fttt/workload.py) 与
[GTTA](../../pllm/workloads/gtta/workload.py)，它们不再手写图、端口/模式字典或
实现 ID。`declare_host` 仍提供设备检查与状态工厂。更一般的宿主数据流可用高级
构建器。原始宿主执行器只在 Rust 准入后调用 `state_factory`，所以有限预算拒绝
不会先分配状态。高级声明可以传其原始 `timeline`；它不能放松
`HostRuntimeFacts.memory_budget_bytes`。FTTT 以 `max_input_rows` 与
`memory_budget_bytes` 提供有限张量成本：参数/动量、回滚快照、活跃 hidden 输入、
反馈特征、训练/查询工作区与结果存储。输入基数在快照前检查。队列存储、
文本/judge/Python 分配与库 scratch 在这个按执行张量预算之外；GTTA 没有完整的
Python 堆成本声明。

`Observation` 把选中行或身份出处带到既有 trace。显式有界导出用 `export=True`；
可选的数学观测仍遵循原始 trace 选择。两者都不需要访问执行器或 arena。

## Completion 学习

继承 `pllm.CompletionWorkload[Data, Result]`，从原始 `CompletionInput` 做实例
作用域学习。`parameters` 声明为带具名形状、优化器与可选初始化的
`InstanceState`。`prepare` 计算算法的结构化上下文；`loss(data, parameters)`
接收具名张量视图并返回标量，或返回 `None` 跳过整个优化器更新。
`trained_rows` 报告所返回 loss 的有效工作量。上下文里只存 detach 的诊断。
`output(data, invocation)` 接收原始 `InvocationOutput`；它可以暴露原始发布与
状态引用而不拥有任何一个。

绑定处理输入形状/设备/就绪、Adam 字段、autograd、私有写与发布。不需要 build、
张量字段索引或手工梯度收集。具名参数共享一个实例根并保留各自的 Adam 字段。
零梯度仍推进优化器状态并发布；`None` 让参数、动量与步数不变。这里没有请求
状态、跨调用反向或自动捕获。初始化与 loss 使用冷张量依赖契约；结构化的
prepare/output 辅助仍是显式原生宿主数学。标量配置在绑定时快照（包括类属性），
之后的 monkey patch 只影响新绑定的执行。TTRL 是维护中的范例：多数投票、
平局规则、采样限制与重试/栅栏构造仍是它的数学，原始 completion 对象保留。

有界 TTRL 张量执行：把 `HostRuntimeFacts(device, hidden_size=...,
max_input_rows=..., memory_budget_bytes=...)` 传给同一公共工厂。内置公式包含
原始 hidden 缓冲、选中行、head/MSE/Adam 工作区与保留结果，都在原始
SchedulerTimeline 里。行上限在准备前检查；CUDA 输入记录其消费流。修改过的
prepare/loss/output 数学必须提供自己的 `resource_facts(facts)`，而不是继承那个
公式。返回的 timeline 不能擦除用户限制；缺失的任务/输出成本导致有限 Rust 准入
失败。没有预算时普通行为不变。这不是对 completion 文本、Python 元数据/历史或
外部库 scratch 的约束。宿主声明保留其原始 `HostRuntimeFacts`；高级调用方用
`binding.compile()`，不是编译期另一套事实。

## 因果适配器

继承 `pllm.CausalAdapterWorkload[Data, Result]`，带冻结的 `pllm.CausalModel`、
优化器和设为 `BEFORE_MODEL` 或 `AFTER_SAMPLE` 的 `phase`。它使用与 completion
学习相同的 prepare/loss/trained_rows/output 数学契约。`prepare` 接收原始
`CausalTrainBatch`，未提供训练输入时为 `None`。具名 loss 参数按模型的有序
A/B 因子对排列。共享绑定推导 Adam 字段与 LOW_RANK_ADD 附着，校验目标
形状/设备/dtype 与因果输入，并在声明的阶段原子发布。TLM 保留其 q/v 限制与
detach 的句子选择目标；负载中没有 TLM 专属的布局、梯度收集或发布实现。

基座模型与初始化模板必须保持不可变。它们按身份保留，不与负载配置一起深拷贝；
活跃因子是原始 PTTS 视图。模型指纹、解析出的优化器/scale 与 loss 进入签名声明。
meta 模型可以对照 worker 的真实设备事实做声明，但可执行数学需要该设备上
物化的模型。内置 TLM 定义支持封闭的 EngineCore 重建，无需初始化父 CUDA。
这个接口是实例作用域的，不增加请求适配器、捕获或 replay。
`CausalModel.resident_bytes` 声明 PTTS 之外的冻结参数/缓冲存储。`HfQvModel`
从唯一张量存储推导它，包括 meta 模型：tied 权重与借用的因子模板不重复计数。
原始后端探针把它加到每个 timeline 锚点，连同输入缓冲；worker 重建检查物化
需求。这是张量驻留，不是分配器开销或 backward/库 scratch。HF 调用显式禁用
KV、hidden 状态与注意力输出保留；因果目标只消费 logits。显式训练成本声明必须
包含 `update.input_shapes=((sequences, tokens),)`。两个轴都是容量界，在传输前
与因果数学前检查；与打包推理行无关。标准 HF/PEFT TLM 绑定仍没有完整的模型
backward scratch 成本声明：其无界模式不是有限内存保证，Rust 在有限计划中拒绝
缺失成本。离线 TLM 把冻结模型的 logits 归约为 detach 的句子 loss，再做适配
前向。只有这个向量存活，不是完整的 batch/token/词表张量。该目标属于 TLM 数学，
不属于共享原语目录。

关闭模型句柄会 detach 并关闭引擎，然后丢弃其 runner、执行与引擎引用。
调用方自有的模型与返回的张量保留自己的生命周期。失败的关闭使句柄不再接受
新工作，但保留引擎与任何未确认的共享输入作用域；`close()` 可以重试，只在
关闭成功后释放该作用域。保留已关闭的句柄本身不会保留已成功关闭的引擎。

## 原语与资源契约

后端绑定输入使用原始探针的 `max_packed_rows`：该上限计入所有打包的模型 token，
包括 prefill，不是选中的 decode 行或去重请求数。超容量输入在状态选择前被拒；
分阶段提交在被拒时取消其既有调用。vLLM 在打开步或捕获前检查。无状态后端
观测者同样强制签名的 hidden 宽度。辅助输入有针对原始任务输入界或检查点字段
布局与状态作用域的自己的预检。请求检查点输入可以包含超出同时 arena 容量的
排队请求；原始 arena 仍在其额度内准入每个活跃队列。缺失任务成本仍意味着无界
执行，而不是推断出的安全；形状/dtype 检查不证明任意用户数学的 scratch 成本。
纯宿主观测没有后端探针。

[PLLM 设计](../design/pllm-next-stage-design.md) 区分普通数学执行与显式高性能
原语绑定。`compute` 是不透明的原生操作；`a.ops` 放维护中的原语绑定。当前目录
包含门控预测/loss/更新实现候选与既有的低秩 loss/AdamW 捕获绑定。它不是通用
算子库，也不是任意 autograd 程序的自动下沉。

`MathContract` 挂在原始 SemanticTask 上并计入其签名。它分别声明输入梯度、
参数状态梯度、detach 输入、前向/输入梯度/参数梯度行域、更新状态域与归约。
域命名原始输入端口；原生数学在这些域内做显式过滤，迁移选择器标识实际可写
状态。Rust 拒绝不存在的端口/资源与矛盾的 detach/梯度声明。改动归约或需求会使
旧签名失效。这些检查不证明任意 Python 数学实现了其声明的导数或归约。

原生输入梯度必须显式计算并由声明的输出/数据边界携带；普通一阶方法用
`differentiate`。冻结参数梯度不意味着 detach 输入。共享 loss 归约与按请求归约
是不同算法，不是调度选择。Output LoRA 的独立路径使用按适配器加权归一化后再
跨适配器取均值；迁移时不得变成一个全局加权均值。

显式张量与分页数学现在共享 `pllm.TensorRows` / `pllm.PagedRows` 操作：
`linear`、`input_gradient`、`parameter_gradient`、`values`、`reduce` 与 `write`。
`TensorRows(packed_fields, row_state_indices)` 用显式张量打包；整数第二参数表示
连续等大行组，使用分批 GEMM 而不做逐 token 矩阵展开。负张量索引是闲置填充。
写要求调用方自有私有字段；`PagedRows` 继续要求原始 arena 签发的写能力。
两个对象都不拥有逻辑版本、发布或回收。ESamp 对两种实现使用同一个
残差/loss/反向表达式。Output LoRA 对张量/分页 eager 与捕获使用同一个显式
reset/加权 CE/多步 SGD 循环；其原先独立的 `weighted_lora.py` 实现已移除。
普通 Tensor 方法声明图与输入；这些显式原生替换绑定同一张图。

逻辑参数在绑定选择实现前一直留在原始图中。张量实现借用或打包视图；其捕获的
训练仍做 arena gather/scatter。**捕获不等于直接页表 GEMM。** ESamp 与共享/独立
Output LoRA 额外在配置中接受 `paged=True`。该签名实现直接在原始 FP32 矩阵池上
使用共享 Triton 分组内核，包括输入反向与逻辑状态参数梯度归约。需要 CUDA
（已在 SM89、PyTorch 2.13、Triton 3.7 验证）。支持任意正矩阵维度与跨步池切片；
使用 IEEE FP32 而非 TF32。偏置/范数向量仍用行读取。默认张量实现保持显式，
不是分页失败后的兜底。原先那个在 SM89 上把偏移同步到宿主的库分组调用已移除。
公开 trace 确认 ESamp loss 的分组在设备侧完成，无 DtoH。特化键使用布局/方向与
两档归约容量桶，不用请求身份、版本或单行数。小归约用直接 FMA，而不是填充到
完整点积 tile。长内维窄输出用共享的 split-inner 内核与 FP32 部分归约，直接读
同一授权槽/活跃映射。不使用负载专属 GPU 实现或逐 token 矩阵拷贝。

高级 `batch.paged(fields, row_state_indices)` 每字段返回一个视图，借用来自原始
精确 pin 或私有写的不可变调用局部映射操作数。`linear`、`input_gradient`、
`parameter_gradient` 与 `reduce` 共享这些映射；矩阵偏移/形状是逻辑字段切片，
不是物理槽选择。`parameter_gradient(inputs, gradient, selected)` 可以只返回选中
逻辑状态，按该设备索引序。索引必须属于原始映射；这不授予写权限。分页归约直接
写紧凑结果，不做全状态梯度再 gather。省略选择则保留完整逻辑状态结果。
`PagedRows.select_rows(fields, mask)` 用原始行上的设备布尔掩码限制计算。前向、
输入梯度与参数梯度可以分别喂不同选择；矩阵内核不加载被排除行的操作数（包括
NaN），其输出/梯度贡献为零。所有选中字段共享一个稳定过滤序。选择不能重新启用
被排除的行，也不能授予写权限：更新资格仍来自原始私有 PageWrite。这是显式原生
绑定，不是普通 `differentiate` 的自动下沉。过滤用一次前缀扫描启动，最多 16384
行；更大容量用固定尺寸 tile、块计数 cumsum 与 scatter。它拥有一个 int64 行序、
一个布尔活跃掩码与 int32 状态末端，大输入另有每 tile 一个 int32 计数；不 gather
任何张量操作数或参数。这些分配属于签名工作区/捕获驻留。空行、跨步掩码与
65537 行动态捕获都被覆盖。分页原语拒绝 `requires_grad` 操作数，而不是静默丢弃
普通 autograd 图。映射在释放/中止后或另一流上不可使用。eager 字段共享一个行
序、宿主计算的状态末端与每次借用一张字段槽表。全私有字段复用该表做写；混合
访问保留只读槽掩码。标量字段（包括 Adam 步数）用同样的页视图。ESamp 的捕获
loss 则从原始 arena 的稳定槽表推导设备映射，替换前有原始跨流排序，固定容量
填充被排除在 loss/梯度之外。捕获字段共享一个逻辑状态分组。矩阵前向与导数在
内核内直接消费原始行序，不做物理页排序、逆映射或显式操作数/输出行 gather。
页槽按逻辑状态索引，不按行和字段复制；矩阵内核直接解引用行/状态/槽。向量
偏置/范数读取仍 gather。ESamp 的显式 LayerNorm 导数调用同一原生 PyTorch
反向算子，不构造 autograd 磁带；原始输入、均值与逆标准差仍是必要 scratch。
普通用户函数保留正常 autograd 语义，不 gather 矩阵参数。
`PagedRows.adamw(fields, gradients, config)` 接受参数/梯度/步数/一阶动量/二阶
动量顺序的五个原始 FP32 字段和连续 `[states, parameters]` 梯度。只有原始私有槽
可被更新；过滤计算行不改变写权限。ESamp 分页 eager 与 Adam 捕获共享一个
步数/偏置内核和一个参数/梯度/动量内核。eager 仍构造启动槽表；其已知工作区按
每状态 96 字节取整（80 槽表字节、4 状态末端字节、8 校正字节），不含分配器
开销。捕获用其原始稳定槽表并计入图池预留。普通 eager AdamW 仍是原始 PyTorch
实现；库融合开关没有保持既定的归一化 ESamp 容差，未被选用。共享/独立 LoRA 也
支持分页写捕获：reset 与 SGD 通过 `write(values, selected, add=..., scale=...)`
直接写原始私有字段。`selected` 是原始授权状态索引的一个去重子集；值保持该序。
只读槽不可变。eager 与捕获训练共享同一数学与加权归约；捕获的缺席槽用 -1，永不
写入。写内核直接读选中状态与原始槽（包括跨步选择），没有中间选中槽张量。
缩放对替换写与加性写都适用。显式 LoRA 循环复用 log-softmax 存储做 CE 导数，
并把固定行归一化提升到 SGD 迭代之外。LoRA 在其既有捕获行状态输入中排除零权重
行；eager 过滤用已算出的有效行数，不全活跃批次不重组。直接 `arena.capture_kernel`
在返回前恢复调用方自有私有字段。框架 `KernelCapture.bind` 使用一次性准入页并在
捕获后中止它们，不分配字段备份。KernelCapture 输入工厂接受设备；它们必须在
meta 与 CUDA 上产出相同的去重缓冲形状/dtype。所有捕获都在分配任何捕获输入缓冲
前，对照原始签名驻留预算在 meta 上量尺寸。真实输入存储与字段 scratch 在冷准入前
再检查；图池预留在捕获后检查，被拒时回收冷页。真实 replay、跨流排序、发布与
中止仍归原始 arena，无字段 gather 缓冲。`resource_facts(facts, options)` 返回从
`facts.timeline` 派生的 SchedulerTimeline，带原始任务/值资源事实，不是另一张
成本图。`ExecutionOptions(memory_budget_bytes=...)` 约束该原始 AER 计划。原始
timeline、返回的资源事实与用户限制三者求交：任何声明都不能放松或擦除既有限制。
`None` 不加限制；零是零字节限制。有限规划要求每个任务与输出（包括零字节输出）
都有显式成本。该限制覆盖声明的状态、驻留、预留、工作区与存活结果字节，不覆盖
未声明的 vLLM 模型/KV、分配器或外部库成本。它区别于 arena 容量
（`state_memory_bytes`）与每任务捕获容量。ESamp 还声明普通张量打包/保存的
中间量、谓词、Adam 工作区与导出的 loss/梯度存储。ESamp 与 LoRA 在拒绝资金不足
的任务/输出事实时复用其声明公式，并在调用时强制签名的行/状态容量。Dummy、
OverRIDE、普通 In-Place-TTT 与 ReFT 在请求有限预算（或任务成本已存在）时额外
提供内置任务/输出成本。无界普通用法不变。它们的 worker 绑定从原始事实重算最小
成本，连重新签名过的资金不足声称也拒绝。Dummy 在有界导出前计入全宽不规则行
打包；OverRIDE 覆盖有界 head/CE/Adam 张量；In-Place-TTT 覆盖尾部、块输出与稠密
更新，强制每 target 总操作数行数；ReFT 在拷贝字段前检查预算更新的有限内容。
条件 In-Place-TTT 与 Recirculation 模型 replay、完整 HF backward 与外部库
scratch 没有内置完整成本声明。这些路径的有限使用需要显式声明；这些公式都不是
总 GPU 内存保证。自定义声明为其改动的数学拥有成本声称；签名图和继承检查都不
证明任意 autograd/库 scratch 的界。原生源/目标打包在消费任务内进行；预测只打包
合格采样行。训练结果是导出的图值，所以 AER 保留其存储到完成，不止到优化器消费。
连续 hidden 行借用原始快照（包括跨步列）；不规则行仍打包。vLLM 为每个资源创建
一个自持有的 FP32 快照，并跨原始 timeline 预留其最大容量；worker 重建保留该
timeline 并拒绝更低的快照预留。就绪排序让生产先于消费，同时记录消费流防止
分配器在读取完成前复用。既有后端需求从原始图投影采样阶段/反馈。vLLM 在采样
锚点预留已知投影/转换/行运算缓冲，并自持反馈 logits/token 到完成。探针与附着
安装用真实采样容量与填充头宽检查同样的最小值；安装从原始签名图读取需求。
整行加法/重加权避免 gather/拷回；部分行保留精确索引与有限支撑语义。这些预留
不覆盖上游采样器/GEMM scratch 或模型/KV 池。特化还从原始附着、字段形状与
replay 任务推导效果缓冲预留。保守信封跨 timeline 保留稠密操作数快照、低秩/激活
中间量、replay 进位/当前槽 KV 备份与正常输出；资金不足的重新签名计划在钩子前
被安装拒绝。Qwen2 适配器直接更新其新鲜投影输出，从不写状态页或残差输入。
连续低秩/replay 行借用视图；稠密训练捕获仍自持其输入。逐 token replay 跨层共享
一个不可变序列张量，保留正常输出不克隆。其逐层 KV 备份仍在正常 token 之前；
条件 replay 备份成功的正常 token。加权 CE 准备借用全部有效的普通输入；无效行
需要掩码拷贝，推理模式张量需要自持普通存储供 autograd。普通请求参数读取在
所有选中行使用同一原始状态时借用广播视图（包括单例/空选择）。混合状态仍打包；
backward 把重复行贡献求和进同一原始导数叶。这些叶别名 pin 住的参数，不是第二份
参数分配。连续采样反馈行同样借用其自持反馈缓冲；重排行打包。这些输入对用户
数学只读。结算的上下文释放引导/反馈；返回的结果保留其原始张量，失败结算保留
其重试数据。在唯一普通训练终点，导数叶与保留的结果图引用在成功后释放，包括
无训练分支。导出的张量存储、计算的梯度与分阶段优化器更新被保留；这依赖既有
要求：每条微分路径都到达该终点。这些保守已知张量信封不是库 scratch 保证，也不
约束任意自定义 autograd。不支持的捕获在特化时被拒。

语义执行、直接分页访问与捕获/融合保持为彼此独立的能力声称。新优化原语以跨
可复用场景的测量为准，改实现前先固定独立参考。

## 全负载覆盖

| 负载 | 声明与数学需求 |
| --- | --- |
| Dummy | 无状态；选中 hidden 行、结果与有界观测 |
| OverRIDE | 实例头；过滤前加法、真实采样反馈、条件 Adam 更新 |
| ESamp | 请求键；两个 hidden 输入、版本感知引导、按状态均值 loss、eager/捕获更新 |
| In-Place-TTT | 多 target 与尾部字段；直接迁移、发布、可选新版本 replay |
| TLM | 检查点/模型事实、因果输入、分组 q/v 字段、原子发布、before-model 离线 vs after-sample 在线时机 |
| Output LoRA | 实例/请求作用域、原始键、外部加权 CE、独立训练/更新选择、SGD 与捕获 |
| ReFT | 无优化器外部字段替换、发布、激活目标与选中行 |
| Recirculation | 无状态逐 token replay；有序 prefill、原始 KV/RNG 与正常输出 |
| FTTT | completion/judge 输入、update-before-predict、原始事务、receipt 与重试输出 |
| GTTA | 无张量字段的宿主状态；FIFO 轨迹、update-before-query、abort/requeue 与 commit |
| TTRL | 结构化 completion/reward 输入、条件梯度、发布与组合重试/栅栏结果 |

`Builder.input` 命名既有带类型服务端口；无字段的 `state` 描述不分配张量 arena
的宿主状态。宿主图声明继续进入既有 `TensorHostBinding` 或
`TransactionHostBinding`。Overlay/激活绑定与两种原始 replay 效果保持彼此独立。
可表达不等于支持任意组合：后端能力检查仍拒绝不支持的组合。表中所有条目都执行
通过这个接口构造的声明。它们既有的数学 oracle 独立于构建器；本次迁移不蕴含
任意跨负载组合。

## 证据与边界

人类作者与编码代理使用同一契约和维护中的源码范例。不要为了让优化通过对比而
改动独立参考。对输出、梯度、优化器状态、无更新、隔离、取消与结算扩展现有证据；
不要重建负载/运行时矩阵。公开基准使用同一工厂。

显式选择的观测与内核计时使用原始有界 trace。不需要反射钩子、重复的性能数据库、
运行时代理服务或生成代码下载。EngineCore 按引用传输负载并要求签名逐字节复核，
从不做任意函数序列化。

证据 47 对照冻结的 PyTorch 参考检查独立 Huber 适配，包括 CPU/CUDA 梯度、全部
Adam 字段与生命周期。它只声称普通 eager。其余的优化器融合、完整资源审计与
发布/性能研究不被该实验或已完成的共享映射替换所蕴含。
