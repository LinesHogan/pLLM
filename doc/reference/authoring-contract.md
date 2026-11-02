# 负载创作契约

本文是 PLLM 负载创作的**完整契约**：每一条能力边界与禁止事项都在这里。
这是一份**参考文档**，不是教程：条款按主题组织、供查证用，不按学习路径展开。
如果你是第一次写负载，请先做[教程](../tutorials/00_quickstart.ipynb)和
[使用指南](../guides/write-a-workload.md)；要把需求交给编码代理，见
[代理指南](../guides/agent-workloads.md)。

## 如何使用这份契约

按你正在做的事定位小节。每节开头有一段引子，说明这节约束什么、为什么存在、
违反会怎样；引子只负责导航，条款本身才是契约。术语以[词汇表](glossary.md)为准。

| 你在做的事 | 读这几节 |
| --- | --- |
| 写普通张量负载（Dummy / OverRIDE / Huber 风格） | [张量方法与状态](#张量方法与状态)、[训练与梯度](#训练与梯度) |
| 采样引导、logits 加法、有界导出 | [采样引导、logits 与导出](#采样引导logits-与导出) |
| 参数 overlay、更新后条件 replay（In-Place-TTT 风格） | [请求 Overlay](#请求-overlay) |
| 用宿主计算（超网络 / checkpoint）产出 overlay 或激活字段 | [外部 Overlay](#外部-overlay)、[激活检查点](#激活检查点) |
| 让多个请求按路由键共享适配器 | [路由键](#路由键) |
| 逐 token prefill 循环（Recirculation） | [逐 Token 循环](#逐-token-循环) |
| 外部加权批量训练、多步 SGD（Output LoRA 风格） | [加权批量训练](#加权批量训练) |
| 自定义数据流、写原生数学、结构化宿主事务 | [高级数据流](#高级数据流)、[编写原生数学](#编写原生数学) |
| completion / reward 驱动学习（TTRL） | [Completion 学习](#completion-学习) |
| 因果批训练（TLM） | [因果批训练](#因果批训练) |
| 改运行时、加原语、声明资源成本、排查有限准入失败 | [原语与资源契约](#原语与资源契约) |
| 排查签名不匹配 / worker 重建失败 | [采样引导、logits 与导出](#采样引导logits-与导出)末尾的冷绑定器段落 |
| 查某个内置负载用了哪些声明 | [全负载覆盖](#全负载覆盖) |
| 确认某项优化是否已有证据支撑 | [证据与边界](#证据与边界) |

**内置负载与共享入口。** 普通负载使用 `import pllm` 的普通张量方法。
[Dummy](../../pllm/workloads/dummy.py)、
[OverRIDE](../../pllm/workloads/override.py)、[In-Place-TTT](../../pllm/workloads/inplace_ttt.py)
和 [ReFT](../../pllm/workloads/reft.py) 使用这个接口；[Recirculation](../../pllm/workloads/recirculation.py)
使用普通的 `Recurrence` 声明；[TTRL](../../pllm/workloads/ttrl.py) 使用普通 `Workload`
的宿主路径（奖励投票是宿主函数 `ttrl_vote`），FTTT 的编辑也是（`FTTTFeedback` 作反馈回调）；
GTTA 是普通宿主类，其 `run` 作反馈回调；
[TLM](../../pllm/workloads/tlm/workload.py) 使用普通 `Workload` 的因果批输入与模型端口；
[ESamp](../../pllm/workloads/esamp.py) 使用普通具名参数、预测与 loss，
并在同一张图上做显式分页/捕获替换；Output LoRA 使用普通 `Batch` 输入与多步 SGD。
全部十一个内置负载都使用各自适用的共享创作入口，构造同一张原始图，
使用同一套执行所有者。独立的 [Huber 示例](../../examples/huber.py) 用普通
Tensor 方法表达请求内低秩学习与持久 AdamW，不需要改动核心。

## 张量方法与状态

本节约束最普通的负载形态：`import pllm` 的 Tensor 方法如何声明输入来源、
如何声明状态、这些值的生命周期与可写性。它存在的原因是：普通数学拿到的
输入与参数视图都是借来的只读值——把它们当普通 PyTorch 张量原地修改、
或假设它们随 Python 对象存活，就越过了签名边界。违反本节的典型后果是
绑定失败（输入歧义、字符串注解）或读写了不属于你的版本。
写 Dummy / OverRIDE / Huber 风格负载时先读本节。

```python
import torch
import pllm

class CenteredHidden(pllm.Workload):
    @pllm.export(name="centered_hidden", max_rows=2, max_cols=16)
    def forward(self, hidden: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return hidden - hidden.mean(dim=-1, keepdim=True)
```

`pllm.hidden(layer)` 声明的是受支持的块/模型输出捕获，不是任意的 eager PyTorch 钩子。
每个输入的来源写成参数默认值：`pllm.hidden`、`pllm.residual`、`pllm.layer_logits`、
`pllm.sample_index`、`pllm.base_logits`、
`pllm.sampled_tokens`、`pllm.parameter(state, field)`、`pllm.module(state)`、
`pllm.index(state)`、`pllm.version(state)`、`pllm.result(operation)` 与
`pllm.batch(field)`；参数名不做猜测。

`InstanceState` 和 `RequestState` 声明单个 `shape=` 或具名字段、dtype、
优化器和可选初始化器。它们可以声明在类上、实例上，或形状依赖配置/模型维度时
声明为带类型的 property。当 `optimizer=None` 时字段就是布局本身：准入时初始化、
任何任务（包括观测）都可读、可通过外部通道导入、永不更新 —— 这是同一种状态的
冻结形态。

请求状态默认按真实请求 ID 隔离。它不因 Python 对象析构而回收：
生命周期由原始 PTTS pin 与完成管理。

`pllm.hidden(layer)` 是第 `layer` 层新鲜的 MLP 输出（不含残差）；`pllm.residual(layer)`
是该层之后的残差流——embedding 加上之前每个子层的输出，也就是下一个子层 norm 读入的和；
`pllm.residual(layer, after="attention")` 是该层 attention 与 MLP 之间的残差流。
两者都按请求取一行（`tokens="decode"` 或 `"prefill"`），在 vLLM 的 CUDA graph 内捕获。
`pllm.layer_logits(layers)` 把模型自己的最终 norm 与 LM head 用到这些层之后的残差流上
（logit lens），返回 FP32 `[rows, len(layers), vocab]`；最后一层就是模型自己的 logits。
它覆盖采样器抽取的每个 token：prompt 的最后一个 token 与每个 decode token。层 logits
在 LM head 处用一个独立 GEMM 计算，模型自身的 logits 保持与 vLLM 逐位相同；
内置的 `pllm.workloads.DoLaWorkload` 用它实现 DoLa。

实例参数就是声明的形状；请求参数会多出一个操作行轴。普通实现显式打包这些
视图 —— 这**不是原生分页 GEMM**。输入与参数视图是借来的只读值；普通数学
不得原地修改它们。形状/dtype 声明不证明任意 Python 副作用是自由的。
优化器写入仍走原始私有版本。

## 训练与梯度

本节约束训练声明：`@train` 如何选择状态与更新资格、`@differentiate` 如何
保留导数边、梯度如何归约、结果如何存储与释放。核心原则是「声明即授权」：
只有被 update 掩码选中的原始状态会被写，其余一律只读；归约方式由声明决定，
绑定不会发明归一化。最容易踩的坑：空 update 与零梯度仍会推进 Adam 步数；
`StateVersion` 永远不是 latest。本节所有导数规则都是调用局部、一阶的，
不是穿透 vLLM 或跨越 replay/发布边界的全模型反向传播。

### `@train`：状态资格与更新掩码

优化器只声明在状态上（`RequestState`/`InstanceState(optimizer=...)`），装饰器不带
优化器。训练方式由方法读取的输入决定，不由优化器类型决定：没有 `pllm.batch(...)`
字段时，按请求在运行中的行上训练（`pllm.AdamW`）；读取 `WeightedRowBatch` 或
`CausalTrainBatch` 的字段（或 `pllm.causal_model`）时，对宿主批次做一次 AdamW 步；
读取 `WeightedCEBatch` 的字段（`base_logits`，或 `labels` 与 `hidden` 同用）时，
按状态从 `reset` 起做 `steps` 次 SGD（`pllm.SGD`）。字段属于第一个包含全部所读字段的
批次类型，顺序为 `WeightedRowBatch`、`CausalTrainBatch`、`WeightedCEBatch`。
状态的优化器与输入不符时，声明期报错并说明应声明哪种优化器。

`@train(state=..., update=...)` 绑定一个标量 loss 和一个可选的状态资格选择器；
`@on(phase, rows=...)` 独立选择它的前向行。选择器是具名 Tensor 方法，
只使用该操作已注解输入的子集，返回其原始行轴上的布尔掩码，并成为同一签名图中的
普通任务 —— 不是执行期回调。旧的 `train(select=...)` 选项已移除；
OverRIDE 用 `on(..., rows=...)` 表达它原先的训练子集。
不带 `update` 时，更新前向行所代表的状态。显式的 update 掩码授权每个选中行的
原始状态（任一选中行授权共享状态一次），与前向或梯度资格相互独立。
空的 update 选择保留所有字段与版本；被授权的零梯度仍会推进 Adam 步数、
动量与衰减，包括空前向调用。标量函数自己拥有归约；绑定不会发明归一化。

### 按状态归约与 `StateIndex` / `StateVersion`

`@train(state=..., reduction="state_mean")` 改为返回每个原始选中状态一个均值 loss。
绑定器对它们求**和**再微分，而不是跨状态求均值；loss 与优化器在原始图中
分阶段成为独立任务，保持声明的时机。这个形式目前要求无条件的
forward/update 域。`StateIndex` 给出每行在本次调用内的状态索引，用于归约；
`StateVersion` 给出该行的原始输入 pin 版本 —— 永远不是 latest 或优化器步数。
两者都不可微。显式的 `RequestState.keys` 仍是原始隔离键；省略则按请求 ID。
ESamp 保留其仅请求 ID 的后端限制。

### Memory 字段

状态还可以携带普通 **carry 字段**：`InstanceState(..., carry={"carry": shape})`
声明跟随训练迁移、但不参与优化器展开的每 lineage 张量（没有梯度/动量槽，
永不被 Adam 更新）。memory 读取（`Parameter(state, field)`）看到的是步开始时
pin 住的输入版本 —— 上一步发布的值 —— 所以训练操作消费的是它上一步写下的内容。
写入是按行捕获：训练输入上的第三个注解标记
`Annotated[Tensor, source, pllm.carry(state, field)]` 在同一迁移与发布中，
把该输入每个 lineage 最后一个选中行的值写进它的 carry 字段。捕获目标必须是
任务自己的训练状态，要求已声明的 carry 字段，并且是 detach 的。页零初始化，
所以第一步读到的是零。跨请求共享由已有的宿主导出/导入通道或共享 route
lineage 组合而成；memory 本身永远不逃离其 lineage 的 pin。

### `@differentiate` 与导数边

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

### 结果存储、释放与活性

结果张量共享本次调用的原始 `(task, output)` 存储。Rust 规划它们的最后使用 —
— 包括下游声明的梯度消费者 —— 并在原始物理任务上签名释放 ID。Python 只
应用这些释放；导出的值存续到完成。没有单独的 TensorFrame 结果缓存或 Python
活性规划器。压实可能同步，autograd 仍可能为 detach 的行保存中间量。
完整的 scratch/保存张量规划和选择性 backward 融合仍未完成；不作捕获或
零分配声称。完成/取消释放调用数据与保留的反向图；失败的结算只保留到原始
receipt 能结算为止。

## 采样引导、logits 与导出

本节约束负载与采样器/导出的接口，以及冷绑定器的签名与传输规则。前半部分
（`@guide_sampling`、`@add_logits`、`@export`、`Attr`）是写负载时用的声明面；
后半部分（函数替换、指纹、冻结、EngineCore 按引用传输）决定「改数学」的代价：
改动函数体或受检依赖会重新签名，多进程 worker 重建时签名必须与驱动端逐字节
相等。排查签名不匹配、函数替换不生效、worker 重建失败时回到本节末尾。

`@guide_sampling(beta=...)` 返回过滤后的 hidden 预测，可由 `@on(..., rows=...)`
选择。绑定把它们关联到原始采样器链接（包括重排的链接）；不要求词表维度。
ESamp 用它做版本感知引导和普通的 Tensor loss/初始化；其五字段 Adam 布局与
初始化种子推进保持不变。

`@add_logits` 在过滤前把结果加到 logits 上，不需要 `@train`；与 `@train` 同在时，
训练方法可读真实采样反馈（`pllm.base_logits`、`pllm.sampled_tokens`）。
`@export` 拥有有界、按步长门控的 CPU 输出；`Attr("config.layer")` 等属性路径
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

## 效果组合

每个效果各自声明它的位置与状态读写，框架按位置排序；只有真实冲突被拒绝，
错误信息说明冲突是什么。可以并存的例子：多个 `@pllm.train`（同一种训练方式，
各训各的状态）、`@pllm.export` 与训练或写入、`@pllm.add_logits` /
`@pllm.guide_sampling` 与 Adapter、ActivationPatch、`@pllm.train_in_forward`。
当前仍被拒绝的组合及原因：

| 组合 | 原因 |
| --- | --- |
| 两个采样器方法（`@add_logits` 与 `@guide_sampling`，或两个同类） | 采样器每步只应用一个方法；合并成一个方法 |
| 请求行训练与宿主批训练 | 每个负载只资助一种训练方式 |
| `@custom_forward` 与读 hidden 行的方法 | custom forward 可能每个 token 跑某层多次或不跑，hidden 行没有唯一值 |
| `@custom_forward` 与 `pllm.model.Adapter` | Adapter 的投影钩子按模型自己的行索引 |
| `@train_in_forward` 与 `pllm.model.Adapter` | 两者共用同一组投影钩子 |
| 带 feed 的状态与其他状态或方法 | feed 的条件输入组只容纳它的写链与读取它的模型 |
| 同一子层上的两个 `@custom_residual` 方法 | 一个方法拥有一个子层的残差更新 |

## 定制残差（`@pllm.custom_residual`）

`@pllm.custom_residual(site=...)` 让一个方法拥有某些层某个位置的残差更新。原生分支
（attention 或 MLP、KV 写入、MoE 路由）总是照常运行，方法只决定流怎样更新：

| `site` | 方法收到 | 返回 | 替换的原生行为 |
| --- | --- | --- | --- |
| `"attention"` / `"mlp"` | 子层前的流 `x` 与分支输出 `y` | 子层后的流 | `x + y` |
| `"both"` | 同上，按前向顺序作用于两个子层 | 同上 | 同上 |
| `"value"` | 全注意力层的 value 行 `[rows, kv_heads, head_dim]` | 替换后的 value（attention 读取并写入 KV cache） | `v` |
| `"final"` | 进入最终 norm 的流 | 一条流 `[rows, hidden]` | 流本身 |

`layers=` 选层（默认所有层）；`read="method"` 命名一个 `(x, ...) -> 分支输入` 的方法，子层归一化它
而不是流；`streams=n` 让 n 条残差流从 embedding 一直走到 `"final"` 方法把它们合成一条（超连接）。
位置张量之后的参数读取 `pllm.parameter`、`pllm.module`、`pllm.depth()`（层号；`"both"` 时为
`2 * layer`，MLP 再加 1）、`pllm.value(layer)` 与 `pllm.residual(layer)`（同一 token 在本次前向中
更早位置的值）。张量为模型 dtype，需要时在方法内用 float32 计算。实例状态按本步最新版本读取，
请求状态按每行所属请求的版本读取。框架用模型自己的融合 add+norm kernel 归一化方法写出的流，
所以恒等方法与原模型逐位相同。定制残差目前以 eager 运行（`cuda_graphs="auto"` 自动选择 eager），
需要 `tensor_parallel_size=1`，且要求关闭前缀缓存。它与 `@pllm.custom_forward` 组合：
custom forward 运行的每一层也经过这些方法。

## 请求 Overlay

本节约束由引擎内更新产出的加性参数 overlay：状态怎么声明、哪些 buffer 可以
被原地修改、更新何时发布、replay 重跑哪些层。它存在的原因是 overlay 直接
作用于模型执行，所以可写集合被显式收窄——只有被授权的 buffer 可以原地修改，
操作数永远不行。一个训练 overlay 可以与同一 workload 的一个 `custom_forward` 组合。
写 In-Place-TTT 风格负载时读本节。

`@on(Phase.IN_FORWARD)` 和 `@train_in_forward(state=..., targets=..., weight=...)` 把
一个加性参数 overlay 与它的发布绑定在一起。状态不带优化器时，方法自己算出下一个
值；状态声明 `optimizer=pllm.SGD(lr)` 时，方法返回每行一个 loss，框架按 `chunk=`
累积梯度并做 SGD（`flush="prompt_end"` 让 prompt 末尾的短 chunk 也训练）。
`@train_in_forward` 不接受 `optimizer=`。固定 `shape=` 使用默认的 `value` 字段；具名 `pllm.Field` 字段可以按每个 target
解析 `(in_features, out_features)`。`pllm.Field` 默认以 FP32 存储并累积状态，独立于
模型 dtype；字段自身的 `dtype=` 覆盖默认值（不会继承 `RequestState.dtype`）。
`dtype=None` 显式选择模型 dtype，整数字段 保留自己的 dtype。
Dense / low-rank overlay 在投影读取时转换到输入 dtype，"per_head" 始终要求 FP32；
FP64 可用于数值 oracle。SGD 的梯度累积至少使用 FP32，显式选择模型 dtype 时也一样。
自定义更新函数也必须在自己的投影运算中显式转换操作数。`Operand.INPUT/TARGET/WEIGHT` 命名捕获的张量；
`Parameter(state, field)` 命名私有 buffer 视图。只有这些被授权的 buffer 可以
被原地修改，操作数永远不行。方法返回
`UpdateOutput(output, changed=..., update_count=..., trained_rows=...)`。
可选的 `metrics` 属于算法，不属于状态或 replay 协议。

默认算子是 `"dense"`，应用 `weight` 字段。传
`operator="low_rank"` 和 `factors=("a", "b")` 加
有限 `scale`，则应用 `scale * b @ a`：down 因子是每个 target 的
`(rank, input_size)`，up 因子是 `(output_size, rank)`。算子选择与请求作用域
正交：同样的发布、pin 与隔离规则对两者都适用。targets 可以命名任何可定位的
投影；后端目前定位 MLP down 与 attention q/v 投影。

`prefill`/`decode` 决定更新资格。被禁用的阶段只应用既有 overlay。
`@pllm.custom_forward(tokens="prefill" | "decode" | "all")` 内的 `layers(...)` 捕获每次
投影的操作数；`layers.apply_updates()` 按 target 拼接自上个边界以来的捕获，并对
当前私有字段执行 transition。后续调用读取更新后的私有版本，尾部 buffer 变化
同样保留。空边界不改变状态或可见版本；末尾未消费的捕获被丢弃。每个步骤仅分配
一个私有版本，并由已有 publication 发布一次。未选中的行保留普通 overlay 语义。
入口遵循作者的可见性策略；边界后的读取关系由签名路由源决定。
返回的 residual stream 决定采样。有界 Python for 循环在追踪时展开；当前拒绝
traced while_loop，并提示使用 bounded for-loop。
共享绑定拥有 target 覆盖、形状/设备检查、根选择与结果出处。结果的 `entries`
暴露请求/before/after 引用和按 target 的设备计数与 metrics。

## 外部 Overlay

`pllm.model.Adapter` 消费状态，不指定生产者。普通 `@pllm.train`、tuple-valued
`@pllm.write(state=..., fields=(...))`、初始化及 `imported=True` 均可独立生产字段。
`hidden(layer, tokens="prefill")` 读取完整 prompt 的最后一行，默认 tokens 为 decode。
写入默认 BEFORE_FORWARD；当前 forward 的 hidden 只能用于 AFTER_FORWARD/AFTER_SAMPLE。
详见[生产者与消费者](../guides/overlays-and-replay.md#独立声明状态生产者与模型消费者)。

以下示例的字段来自引擎之外（超网络、加载的 checkpoint）。
它与[请求 Overlay](#请求-overlay) 的差别只有产出地点：应用、发布、隔离与
退役规则完全相同。关键点：导入只发生一次（只进版本零的请求状态），按原始
prompt 索引作键；之后混合批次中的每一行读到的是自己请求的精确 pin。

overlay 的字段值不必来自引擎内更新：任何宿主计算 —— 读请求上下文的
超网络、加载的 checkpoint —— 都可以产出它们。声明同样的 `RequestState`
（或 `InstanceState`）buffer 和一个 `Adapter` property，命名状态、targets、
算子、因子与 scale：

```python
@dataclass(frozen=True)
class ShineStyleLoRA(pllm.Workload):
    @property
    def factors(self) -> pllm.RequestState:
        return pllm.RequestState({
            "a": pllm.Field(lambda in_features, out_features: (2, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 2)),
        }, imported=True)

    @property
    def adapter(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("factors", targets, operator="low_rank",
                            factors=("a", "b"), scale=0.5)
```

产出的张量作为 `StateImport(values, keys)` 经
`generate(..., batch=..., prompt_indices=...)` 送入；请求替换按原始 prompt
索引作键，且只导入到版本零的请求状态一次。共享绑定校验布局、拷入私有根并在
`request.prepare` 发布 —— 在请求的第一次前向之前 —— 所以混合批次中的每一行
读到的是自己请求的精确 pin，导入的零因子精确复现基座模型。应用、发布、隔离与
退役与优化器产出的 overlay 完全相同；只有产出地点不同。与激活检查点一样，
框架从字段布局追踪有限值检查的工作区；拷贝直接写入私有页，不需要自有张量。

## 路由键

本节约束请求状态的隔离键如何由用户计算：route 值相同的请求加入同一 lineage，
不同 route 保持隔离。框架从不解释 route 值——它是你的宿主代码的产物。
两个容易忽略的后果：休眠 lineage 仍然存活并占用 `max_admitted_lineages`
额度；缺少 route 值的请求在准入时被拒。

请求状态键是签名声明；键值可以由用户计算。给 `RequestState` 的 `key_fields` 加上
`"route"`，并传 `generate(..., routes=(...))` —— 每个 prompt 一个
路由值，由你自己的宿主代码计算（一个正则、一个判别器分数、一个租户 id；
框架从不解释它）。键重合的请求**加入同一 lineage**：任何一个请求训练出的
适配器对全部生效，无论在同一批次内还是跨步。不同 route 保持隔离。route
lineage 在其最后一个请求退役后仍然存活 —— 休眠，仍被其键 pin 住 —— 所以
适配器池在整个会话中存续；`max_admitted_lineages` 约束它，会话关闭时退役休眠
lineage。缺少 route 值的请求在准入时被拒。辅助批输入（`WeightedCEBatch`）按行
解析自己的键投影，所以带键训练在共享 lineage 上照常工作。

## 激活检查点

本节约束激活修改（ReFT 风格）：以外部 `StateImport` 作为 `batch=`
替换检查点字段。它与外部 overlay 同族：共享绑定负责校验检查点、选择根、
拷贝私有字段并在模型执行前发布；缺失的更新不发布，既有推理 pin 继续看到
它们精确的旧版本。

声明普通的 `InstanceState` 或 `RequestState` buffer 和一个 `ActivationPatch`
property，命名状态、层、首个/末个 prefill token 与支持的算子。
`LOW_RANK_AFFINE` 每层使用 weight/bias/rotation 字段；`ADD` 使用一个 FP32 向量。
Field 形状回调收到激活的输入/输出宽度。以原始 `pllm.io.StateImport` 作为
`batch=` 提供。共享绑定校验检查点、选择根、拷贝私有字段并在模型执行前发布。
实例替换可重复；请求替换使用原始 prompt 键，只导入到版本零的请求状态。
缺失的更新不发布，既有推理 pin 继续看到它们精确的旧版本。见维护中的 ReFT
负载；它不再实现 prepare/copy/publish 任务。

## 逐 Token 循环

本节约束 `Recurrence` —— 既有的独立逐 token prefill 效果。它自成一体：
不需要 RequestState、hidden 捕获、外部训练输入、build 或 prepare 方法；
代价是不能与普通训练/激活操作混合（绑定时失败），纯宿主执行没有模型
replay 能力。

在 `pllm.Workload` 上声明 `pllm.model.Recurrence(source_layer=..., destination_layer=...,
alpha=..., beta=..., ramp_tokens=..., enabled=...)` 作为类/实例属性或带类型的
property。负层索引对真实模型解析。`beta=None` 使用互补的目标权重；源在混合前
做范数匹配。这是既有的独立逐 token prefill 效果：正常 logits 保留，replay 更新
选中 token 的层 KV，decode 行只正常运行。不需要 RequestState、hidden 捕获、
外部训练输入、build 或 prepare 方法。把这个效果与普通训练/激活操作混合尚不支持，
在绑定时失败；纯宿主执行没有模型 replay 能力。

## 加权批量训练

本节约束 Output LoRA 形态的负载：具名 SGD 状态、外部 `WeightedCEBatch` 输入、
reset 与多步训练。这条路径把「外部数据喂入 + 实例/请求共享状态 + 多步 SGD」
组合在一起，边界多于普通训练：缺批次的含义、发布时机、`copy_fields` 语义、
最大输入形状与资源推导都在这里。改 LoRA 数学或接捕获/分页内核之前，
先读本节末尾的原生替换规则。

[Output LoRA](../../pllm/workloads/lora.py) 声明具名字段状态（`SGD(lr)` 与
初始化器）、一个普通的 `@add_logits` 预测，以及一个读取 `WeightedCEBatch` 字段的
`AFTER_SAMPLE` 方法（`@train(state=..., reset="reset", steps=Attr(...))`）。reset 方法接收具名局部参数张量和
一个生成器。reset 按训练批次进行，区别于准入初始化；两个函数都被冻结并与普通
数学一起签名。SGD 保留原始状态字段，没有 Adam 槽。

训练输入用 `Annotated[Tensor, Batch("hidden"|"base_logits"|"labels"|"weights")]`
和 `Parameter(state, field)`。它们消费原始 `WeightedCEBatch`，不是新的 DTO。
共享输入用 `pllm.host.WeightedCEBatch(hidden, logits, labels, weights)`；独立输入用
`pllm.host.WeightedCEBatch.for_prompts(hidden, logits, labels, weights, prompt_indices)`，
同样的索引传给 `generate`，可选 `sample_index=0`。两者都借用所供张量，不拷贝。
绑定对齐原始键，拒绝出界输入，掩码非正权重/词表外标签。每个选中状态必须有
有效行。训练张量有领先的选中状态轴，共享状态时是单例。每个状态返回一个归一化
loss；绑定在每个 SGD 步微分它们的**均值**。这不是 ESamp 的独立按状态 Adam 归约。

缺批次意味着没有 reset、没有私有写、没有发布。成功训练在所有步之后发布一次；
取消/失败保留原始 pin 状态。归一化/reset 在原始图的 `__train_prepare` 任务中
于采样前运行，不在调用准备期。其结果边喂给 `train`；同一帧拥有归一化张量直到
消费后释放。框架在最大训练形状上追踪准备、预测、SGD 前后向及保留结果，并将推导的
任务/输出资源事实写入签名时间轴。公开提交、反馈传输与 worker STAGE
对原始辅助张量/描述符做预检，然后才做共享存储导出、映射或设备拷贝；dtype 来自
原始状态布局，标签为 int64。形状检查不查看张量内容，也不替代之后的有限值/选择
检查。reset 训练与激活导入完全覆盖其私有字段，所以它们的绑定用
`copy_fields=()` 跳过把旧字段拷进新页。默认 `copy_fields=None` 拷贝全部私有
字段；显式有序子集只拷那些字段。内置 Adam 绑定保留参数、步数与动量，但不保留
旧梯度（包括每个 TLM 因子组）。发布前必须写满省略复制的字段，不得读未初始化值。
此入口支持 reset SGD 与加权批次；私有页容量及 arena/pin/receipt 所有权不变。

`implementations` 与 `operations` 可为签名任务选择原生实现；实现仍须保持所声明的
数学。修改普通方法会重新推导成本，不需要资源公式或方法锁。

## 高级数据流

本节约束绕过结构化入口、直接继承 `BackendWorkload` 的高级创作。读它的前提是
前面各节的结构化入口表达不了你的数据流。这里的声明直接构成语义图：`Call` /
`When` 命名依赖与时机窗口，绑定不做函数体检查也不自动追踪——所以声明必须
描述数学函数的真实访问，撒谎不会被当场发现，但会产生错误的计划。

自定义高级数据流：继承 `a.BackendWorkload[Data, Result]`，从 `declaration` 返回持有原始
图/布局/附着的 `TensorBinding`，带原生 `bind_math` 与 `backend_requirements`。旧的 `a.Workload`
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

本节约束原生数学绑定（`bind_math` / `FunctionMath` / `TaskOutput`）。这是负载数学与
执行层的接缝：流排序、捕获、私有分配、回滚与发布仍留在原始执行层，原生函数只拿到
借来的操作数与显式授权的迁移根。空梯度与零梯度的区别、预算与成本声明的失败模式
都在这里。给 ESamp / LoRA 的捕获内核写数学时读本节。

高级 `BackendWorkload.bind_math(facts, plan, declaration)` 把规划过的声明绑成原始 `MathBinding`。其
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

宿主循环只有一种写法：反馈回调 `CompletionInput -> 反馈`（带 `retry_controls`、
`update_barriers`、`update_count`）传给 `generate_feedback(feedback=...)`。需要框架管理的
张量状态写成普通 `pllm.Workload`，用 `pllm.host.make_host(HostConfig(...), workload)` 编译，
回调里 `step`。[FTTT](../../pllm/workloads/fttt/workload.py) 的低秩编辑就是这样的
`FTTTWorkload`（AdamW 写在状态上）；`FTTTFeedback(service, HostConfig(...))` 用 `make_host` 编译并准入它
（`feedback.edit` 是普通执行句柄，可取 `diagnostics()`），`feedback(batch)` 评判、入队，
在失败行上 `step` 一次，再用 `StepResult.parameters` 里更新后的编辑算出重试要加的
增量（`step()` 只处理已排队的反馈）；失败的一步清空队列，已经发布的编辑保留。
[GTTA](../../pllm/workloads/gtta/workload.py) 是普通宿主类：`GTTAWorkload.run(request)`
从 `GTTAEnvironment` 的 FIFO 队列取出事件，用 Python 字典中的 FP64 EMA 更新动力学记忆，再回答查询。
分数、稳定排名和历史格式保持不变；`GTTAResult` 直接返回 `processed` 与 `query`。
查询失败时，已消费的事件和已更新的记忆保留。私有 snapshot/restore、版本计数、transaction ID、intent、receipt 与两阶段结算已移除。

`Observation` 把选中行或身份出处带到既有 trace。显式有界导出用 `export=True`；
可选的数学观测仍遵循原始 trace 选择。两者都不需要访问执行器或 arena。

## Completion 学习

本节说明 TTRL 这类“宿主侧算奖励、框架做优化器更新”的负载。它们是普通
`pllm.Workload`，只是走宿主路径：`pllm.host.make_host(HostConfig(...), workload)`。
没有请求状态、没有跨调用反向、没有自动捕获。

奖励投票是宿主字符串逻辑，不是张量数学，所以是一个普通函数：
`ttrl_vote(completion_input, config)` 返回 `TTRLResult`（`reward_outcomes`、
`retry_controls`、`update_barriers`，以及训练步 `step`）。调用方先算它，再
`execution.step(rewards.step, stream)`；CUDA 上在传给 `step` 的同一条流里调用它。
`generate_feedback` 的 `feedback` 回调可以先 `step` 再返回 `rewards`，它带
`update_count`。

训练步携带 `WeightedRowBatch(hidden, targets, weights)`（`pllm.host`）：`hidden` 是
`[行数, hidden_size]`，`weights` 是同 dtype 的 `[行数]`，权重为 0 的行不参与；`targets` 是
`[行数, *HostConfig.target_shape]`——默认 `target_shape=()` 即每行一个值（TTRL），FTTT 每行
一个向量（反馈特征加奖励，`FTTTFeedback` 自己设置 `target_shape`）。
训练方法用 `@pllm.train(state=...)`，通过 `pllm.batch("hidden")`、`pllm.batch("targets")`、
`pllm.batch("weights")` 与 `pllm.parameter(state, field)` 取输入，返回
`pllm.UpdateOutput(loss, changed, update_count, trained_rows)`、标量 loss（每行都训练），或返回 `None` 跳过整个
更新。`changed` 是设备张量：空选择（权重全 0）时发布不变的值，优化器字段保持原样，
训练路径没有主机同步。宿主训练步的 `StepResult.parameters` 按字段名给出这次更新后的
状态（在步所在流上的独立拷贝，计入资源测量），回调在同一步里就能用更新后的值。

参数是带 `pllm.AdamW` 的普通 `InstanceState`；具名形状只是同一个打包 AdamW 组的视图
（AdamW 逐元素，所以与独立参数逐位一致）。标量配置在绑定时快照（包括类属性），
之后的 monkey patch 只影响新绑定的执行。TTRL 是维护中的范例：多数投票、
平局规则、采样限制与重试/栅栏构造是 `ttrl_vote` 的数学，原始 completion 对象保留。

有界宿主执行使用 `HostConfig(max_input_rows=..., memory_budget_bytes=...)`。框架用
最大形状的 `WeightedRowBatch`（含 `target_shape`）执行实际 loss/update 路径来推导资源，不要求负载申报字节；
超过 `max_input_rows` 的输入在 `step` 前被拒绝。

## 因果批训练

[TLM](../../pllm/workloads/tlm/workload.py) 用普通 `pllm.Workload` 训练所有层的 q/v
低秩因子，推理通过 `pllm.model.Adapter` 读取同一状态。使用时只提供配置与训练批：

```python
import torch
import pllm
from vllm import SamplingParams
from pllm.host import CausalTrainBatch
from pllm.workloads.tlm import TLMConfig, TLMMode, TLMWorkload

workload = TLMWorkload(TLMConfig(mode=TLMMode.ONLINE), rank=8, alpha=16)
ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
# TLM 的 overlay 读 InstanceState，默认的 cuda_graphs="auto" 因此让模型以 eager 运行。
config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct")
with pllm.make_llm(config, workload=workload) as llm:
    outputs = llm.generate(
        ["The sky is"], SamplingParams(max_tokens=2),
        batch=CausalTrainBatch(ids, ids),
    )
```

`pllm.batch("input_ids")`、`pllm.batch("labels")`、`pllm.batch("attention_mask")`
把 `CausalTrainBatch` 的字段注入训练方法。ids 与 labels 是同形的 int64
`[序列数, token 数]` 张量，至少一个序列、每个序列至少两个 token；可选 mask
同形，省略时为 `None`。TLM 在 loss 内移动标签以预测下一个 token，忽略
`ignore_index` 标记的标签。`generate(..., batch=batch)` 每次调用只消费一次训练批。

TLM 的 `InstanceState` 用 a/b `pllm.Field` 声明因子，形状回调接收目标投影的输入/输出
宽度；`Adapter("factors", targets, operator="low_rank", scale=alpha/rank)`
声明如何应用它们。Adapter 的 `Projection(layer=None, ...)` 表示所有层，
模型层数确定后展开；显式层目标须有序且唯一。

训练方法声明 `model: pllm.model.CausalModel = pllm.causal_model("adapter")`，其中
`"adapter"` 是 Adapter 声明的名称。`model(ids, attention_mask=mask)` 返回带该
Adapter 的 logits，梯度流向其因子；`model.base(ids, attention_mask=mask)` 返回
冻结基座的 logits。负载不持有或传输活模型。
`@pllm.on(pllm.Attr("phase"))` 在绑定时读取负载的 `phase` 属性并解析为 `Phase`；
也可用 `Attr("config.phase")` 读取嵌套配置。TLM 的 ONLINE 选择
`AFTER_SAMPLE`，用适配后的 logits 选句；OFFLINE 选择 `BEFORE_FORWARD`，用
`torch.no_grad()` 下的基座 logits 选句，再训练选中的句子。

因果训练当前要求实例 AdamW 状态、BF16 基座（投影可量化，嵌入与 LM head 不可）、单设备（TP1），返回标量 loss
或 `UpdateOutput`。直接执行可用 `StepInput(batch=batch)`；返回的
`StepResult.value` 是报告值，`loss` 是优化目标，`publication` 给出发布前后的版本，
`metrics` 是算法自定义的设备张量映射。TLM 的 `metrics["selected_count"]` 是选中的
句子数，执行时不把它同步到宿主。

**空选择仍发布一个新版本**：参数、梯度、Adam 步数与动量保持原值；loss 计数增加一，
逻辑 update 与 trained_rows 计数不增加，`selected_count` 为设备侧的零。
这与没有训练输入、或 `lamb=0` 不同：后二者不运行训练、不发布，`value` 和 `loss`
为 `None`，metrics 为空。不要用版本是否变化来判断这一步是否训练了句子。

有限训练预算需要提供实测成本及 `update.input_shapes=((sequences, tokens),)`，
两个轴都是容量上限；默认无界模式不保证有限训练峰值。

关闭模型句柄会 detach 并关闭引擎，然后丢弃其 runner、执行与引擎引用。
调用方自有的模型与返回的张量保留自己的生命周期。失败的关闭使句柄不再接受
新工作，但保留引擎与任何未确认的共享输入作用域；`close()` 可以重试，只在
关闭成功后释放该作用域。保留已关闭的句柄本身不会保留已成功关闭的引擎。

## 原语与资源契约

本节约束框架最底层的三件事：输入容量与预检、显式原生原语
（`TensorRows` / `PagedRows` / 捕获）、资源事实与内存预算。它存在的原因是
性能路径的每一条捷径都有对应的显式声明义务：**捕获不等于直接页表 GEMM**；
缺失任务成本意味着无界执行，而不是推断出的安全；原始 timeline、资源事实与
用户限制三者求交，任何声明都不能放松既有限制。改运行时、加原语、写成本
声明、排查有限准入失败时读本节；只写普通负载可以跳过。

### 输入容量与预检

后端绑定输入使用原始探针的 `max_packed_rows`：该上限计入所有打包的模型 token，
包括 prefill，不是选中的 decode 行或去重请求数。超容量输入在状态选择前被拒；
分阶段提交在被拒时取消其既有调用。vLLM 在打开步或捕获前检查。无状态后端
观测者同样强制签名的 hidden 宽度。辅助输入有针对原始任务输入界或检查点字段
布局与状态作用域的自己的预检。请求检查点输入可以包含超出同时 arena 容量的
排队请求；原始 arena 仍在其额度内准入每个活跃队列。缺失任务成本仍意味着无界
执行，而不是推断出的安全；形状/dtype 检查不证明任意用户数学的 scratch 成本。
纯宿主观测没有后端探针。

### 原语目录与 `MathContract`

PLLM 区分普通数学执行与显式高性能
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

### `TensorRows` / `PagedRows`：显式张量与分页操作

显式张量与分页数学现在共享 `pllm.TensorRows` / `pllm.PagedRows` 操作：
`linear`、`input_gradient`、`parameter_gradient`、`values`、`reduce` 与 `write`。
`TensorRows(packed_fields, row_state_indices)` 用显式张量打包；整数第二参数表示
连续等大行组，使用分批 GEMM 而不做逐 token 矩阵展开。负张量索引是闲置填充。
写要求调用方自有私有字段；`PagedRows` 继续要求原始 arena 签发的写能力。
两个对象都不拥有逻辑版本、发布或回收。ESamp 对两种实现使用同一个
残差/loss/反向表达式。Output LoRA 对张量/分页 eager 与捕获使用同一个显式
reset/加权 CE/多步 SGD 循环；其原先独立的 `weighted_lora.py` 实现已移除。
普通 Tensor 方法声明图与输入；这些显式原生替换绑定同一张图。

### 分页捕获实现（`paged=True`）

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

### 分页视图、选择与写

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

### 张量步、分页更新与捕获

框架用同一个 StepCapture body 执行普通张量步、形状追踪与 CUDA 图捕获。
它读行/状态表、gather 字段、执行前后向，再更新授权的私有页。FP32 CUDA
AdamW 直接更新页池；CPU 和其他 dtype 共享 PyTorch 优化器路径。缺席槽为 -1，
永不写入；发布、跨流排序、回滚与页生命周期仍归 PTTS。

捕获使用固定输入上限和私有图池；绑定前检查表容量，捕获后核对真实池大小与
`capture_memory_bytes`。负载不计算槽表、校正张量或优化器临时空间的字节公式。

### 资源事实与内存预算

框架的 `resource_facts(facts, options)` 在最大输入形状上执行已绑定的任务，
不另写一套数学或字节公式。它追踪真实的 StepCapture/TensorTask、加权训练、
原生 overlay 与区域扫描；Triton 内核使用 Torch 注册的 fake 实现。存储别名共享
计数，autograd 保存值和返回后保留值持续计入；分配按 512 字节对齐，并计入大块不拆分余量。
输入页另行计账，捕获池仍使用 `capture_memory_bytes`，规划不初始化 CUDA。

任务对象提供自己的资源投影，发布只改变已有页的所有权元数据。并发生产者声明
可执行 `TensorComputation` 与输入形状，框架将其峰值保留到每个 worker anchor。
In-Place-TTT 的区域源码留在负载，TensorScan 负责区域预算和恢复生命周期。
宿主有限训练需要 `max_input_rows`；有宿主选择条件时提供最大形状样例。

`memory_budget_bytes` 约束 AER 中的状态、驻留、工作区和结果，不等于整个进程
显存或 arena 的 `state_memory_bytes`。原时间轴与用户预算求交，不能放松限制；
`None` 不加限制，零就是零字节。未声明的模型/KV、分配器及外部库内存仍不在此界内。

输入与导出结果仍遵循原有生命周期：连续行可借用视图，不规则行会打包；
输入对用户数学只读，结果与梯度保留到消费者完成。普通训练终点释放 autograd
图及导数叶，无训练分支也一样；失败结算保留重试所需的数据。

vLLM 继续按真实容量预留 hidden 快照、采样反馈和效果/replay 缓冲；新推导不重复
计入这些输入。就绪排序和消费流记录防止读完前复用。采样器、GEMM 与模型/KV 的
外部工作区不因此获得新保证。直接分页访问、语义执行和捕获/融合仍是独立能力。

## 全负载覆盖

本节是速查表：十一个内置负载各自使用了哪些声明与数学需求。用它快速定位
「我要的效果有没有现成范例」。注意可表达不等于支持任意组合：后端能力检查
仍拒绝不支持的组合。

| 负载 | 声明与数学需求 |
| --- | --- |
| Dummy | 无状态；选中 hidden 行、结果与有界观测 |
| OverRIDE | 实例头；过滤前加法、真实采样反馈、条件 Adam 更新 |
| ESamp | 请求键；两个 hidden 输入、版本感知引导、按状态均值 loss、eager/捕获更新 |
| In-Place-TTT | 多 target 与尾部字段；直接迁移、发布、可选新版本 replay |
| TLM | 普通 Workload；因果批输入、可微模型端口、全层 q/v Adapter、设备侧选择与发布、在线/离线阶段 |
| Output LoRA | 实例/请求作用域、原始键、外部加权 CE、独立训练/更新选择、SGD 与捕获 |
| ReFT | 无优化器外部字段替换、发布、激活目标与选中行 |
| Recirculation | 无状态逐 token replay；有序 prefill、原始 KV/RNG 与正常输出 |
| FTTT | 宿主 judge 与有界队列、向量目标 `WeightedRowBatch`、AdamW 编辑、用 `parameters` 做 update-before-predict 与重试输出 |
| GTTA | Python 字典中的动力学记忆；FIFO 轨迹、update-before-query，查询失败保留更新 |
| TTRL | 宿主 `ttrl_vote` 投票、`WeightedRowBatch` 加权梯度、空选择不变发布与重试/栅栏结果 |

`Builder.input` 命名既有带类型服务端口；无字段的 `state` 描述不分配张量 arena
的宿主状态。宿主图由 `make_host`（`Workload.compile_host`）规划并绑定。Adapter/激活绑定与两种原始 replay 效果保持彼此独立。
可表达不等于支持任意组合：后端能力检查仍拒绝不支持的组合。表中所有条目都执行
通过这个接口构造的声明。它们既有的数学 oracle 独立于构建器；本次迁移不蕴含
任意跨负载组合。

## 证据与边界

本节约束证据纪律：哪些声称已有编号证据支撑，哪些不被既有实验蕴含。
改优化实现前先固定独立参考；对输出、梯度、优化器状态、无更新、隔离、
取消与结算扩展现有证据，不要重建负载/运行时矩阵。

人类作者与编码代理使用同一契约和维护中的源码范例。不要为了让优化通过对比而
改动独立参考。对输出、梯度、优化器状态、无更新、隔离、取消与结算扩展现有证据；
不要重建负载/运行时矩阵。公开基准使用同一工厂。

显式选择的观测与内核计时使用原始有界 trace。不需要反射钩子、重复的性能数据库、
运行时代理服务或生成代码下载。EngineCore 按引用传输负载并要求签名逐字节复核，
从不做任意函数序列化。

证据 47 对照冻结的 PyTorch 参考检查独立 Huber 适配，包括 CPU/CUDA 梯度、全部
Adam 字段与生命周期。它只声称普通 eager。其余的优化器融合、完整资源审计与
发布/性能研究不被该实验或已完成的共享映射替换所蕴含。
