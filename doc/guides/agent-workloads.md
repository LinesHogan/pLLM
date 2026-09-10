# 给编码代理的 Workload 交付规程

本文是写给 AI 编程助手的操作规程：用户把本文、算法描述和相关代码交给你，由你在
开发期产出一个 PLLM workload。PLLM 运行时不调用 agent——你交付的是普通源码，
框架只按这份源码的签名执行。因此本文的每一条规程都服务于同一件事：
**你写下的语义就是将被执行的语义，没有人在运行时替你兜底。**

## 第一步：让用户把语义说完整

框架负责执行边界、调度与状态真相，所以它不需要用户描述机制，只需要用户作出
**算法选择**。你拿到需求后的第一件事是按下表核对四类信息是否齐全；缺任何一项，
你后面的实现就只能在语义上猜测，而猜出来的语义会被原样执行。

| 必要信息 | 用户需要作出的算法选择 |
| --- | --- |
| 数学 | 要算什么？给公式、朴素 PyTorch 实现或论文中的具体算法；输入输出的含义、形状规则、初始化和超参数是什么？有训练时说明 loss、归约、优化器及更新次数。 |
| 状态与生命周期 | 哪些值需要跨步保存？每个请求独立，还是同一实例共享？何时重置？跨请求／重试是否复用？共享参数是聚合后更新一次，还是按确定顺序多次更新？ |
| 时机与锚点 | 在哪一层、prefill／decode 的哪个阶段计算？采样前还是实际采样后？按 token、chunk 还是完成的回答触发？新参数从哪一步开始生效？ |
| 数据流与作用 | 数据来自模型激活、实际采样 token、外部标签还是 judge？结果用于观察、修改 logits、更新参数／激活、replay 还是重试？哪些请求／行参与？ |

没有状态、训练或外部动作的维度，让用户明确写“无”，而不是留白。三类容易
被含糊带过的语义必须追问到底：

- **训练**：补充 forward、输入梯度、参数梯度、允许更新的状态各自的选择条件，
  以及 loss 按 token、请求还是共享状态归一化。“所有行参与，无输入梯度”是
  合法答案。注意区分三对概念：“不更新”与“零梯度仍执行 optimizer”不同
  （后者仍推进 Adam 步数、动量与衰减）；冻结参数也不一定截断输入梯度。
- **replay**：说明次数／停止条件、读取哪个更新版本、KV/RNG 和正常输出应
  如何变化。
- **host 动作**：说明成功、失败、取消及重试是否允许重复执行外部动作。

另外让用户给出已有的运行约束：模型／checkpoint、设备、dtype、典型
batch／序列长度。交付顺序上，建议先交付普通正确实现，验证正确后再考虑
PLLM 的加速技术（内置的 advanced 精细操作）；但自动页表总是推荐开启——
它对 CUDA graph 兼容性好，而 CUDA graph 的加速总是比较大的。

以下事情用户不需要提供，你也不应向他们索要：图节点 ID、digest、物理 slot、
版本号分配、receipt、手工调度顺序、重复的 batch DTO——这些由共享框架负责；
模型维度可由模型事实得到，不必手填。若用户要求有限预算，你需核对支持范围并
提供适用的成本声明；**不能把缺失的成本数据当作零成本**。

## 第二步：按五条规则实现

### 1. 先确定语义，再选择入口

原 reference 是验收基准：它固定之后，你再同步改写它，任何比较都失去意义。
所以先读仓库 `AGENTS.md`、用户需求与 reference，确认语义无歧义后才动手。
阅读范围只需覆盖下表相关的契约部分和一个最接近的范例，不要求通读迁移历史
或所有 workload。实现过程中发现必要的算法变更，单独与用户确认，不混入
“顺便修复”。

### 2. 先用普通公共接口

普通公共接口（`import pllm`、普通 Tensor 数学、装饰器、具名状态、注解输入）
就是完整契约，表达能力不逊于内部通道，而内部通道绕过的正是框架的签名与复核
机制。局部函数替换可参考 `dataclasses.replace` 示例；不要运行时 monkey-patch
引擎或修改已经绑定的函数——这会让签名与实际行为脱钩。注解须为实际类型：
当前 binder 不求值字符串注解。

### 3. 沿用同一执行边界

框架的状态真相只有一份（Arena 的版本/pin/发布纪律）。你维护的第二份账本
——自己挑物理页、猜 latest、另存版本／生命周期记录——必然与它在边界处
冲突，且冲突在 borrow 过期前不可见。因此：逻辑输入只读，不原地修改借来的
参数，不保存跨调用的借用 tensor；沿用原图、签定计划、PTTS 状态和 receipt。
高级入口同样遵守这一条，不能为统一外观把 host 算法硬塞进 Tensor。

### 4. 优化是显式、按需的第二步

普通 eager、原生分页和 capture 是三种不同能力，签名与成本各异，不能互相
冒充。正确顺序是先复用已有 `TensorRows`／`PagedRows` 原语（它们不拥有版本
或发布权限），仅在确有优化需求时测量相关瓶颈，再决定引入显式绑定；不默认
写专用 kernel，也不在报告中隐瞒打包、scratch 或库成本。

### 5. 做与变更相称的检查并交付

检查的目的是抓住你这类变更真正会引入的错误，不是每次重跑全套发布验收。
用 strict 类型检查发现不可能分支和错误注解；用独立数学基准检查相关输出、
梯度、optimizer 字段和归约。涉及状态时覆盖无更新、请求隔离及相关取消／
结算语义，复用已有夹具。

### 按算法选择契约和范例

| 需求 | 契约入口 | 维护中的范例 |
| --- | --- | --- |
| hidden 观察、普通数学／函数替换 | [Tensor methods](../reference/authoring-contract.md#张量方法与状态) | [Dummy](../../pllm/workloads/dummy.py)、[局部替换](../../examples/centered_hidden.py) |
| 请求独立训练、采样引导 | [状态与梯度](../reference/authoring-contract.md#张量方法与状态) | [Huber](../../examples/huber.py)、[ESamp](../../pllm/workloads/esamp_request.py) |
| 实例共享头、实际采样反馈与选择性更新 | [Tensor methods](../reference/authoring-contract.md#张量方法与状态) | [OverRIDE](../../pllm/workloads/override.py) |
| 共享／独立 LoRA、外部 batch、多步 SGD | [Weighted batch training](../reference/authoring-contract.md#加权批量训练) | [Output LoRA](../../pllm/workloads/lora.py) |
| 参数 overlay、更新后条件 replay | [Request overlays](../reference/authoring-contract.md#请求-overlay) | [In-Place-TTT](../../pllm/workloads/inplace_ttt.py) |
| 激活修改／外部字段发布 | [Activation checkpoints](../reference/authoring-contract.md#激活检查点) | [ReFT](../../pllm/workloads/reft.py) |
| 逐 token prefill replay | [Tokenwise recurrence](../reference/authoring-contract.md#逐-token-循环) | [Recirculation](../../pllm/workloads/recirculation.py) |
| 完整因果模型训练适配 | [Causal adapters](../reference/authoring-contract.md#因果适配器) | [TLM](../../pllm/workloads/tlm/workload.py) |
| completion／reward 驱动训练 | [Completion learning](../reference/authoring-contract.md#completion-学习) | [TTRL](../../pllm/workloads/ttrl.py) |
| judge、环境查询、队列与事务 | [Write mathematics：host 事务](../reference/authoring-contract.md#编写原生数学) | [FTTT](../../pllm/workloads/fttt/workload.py)、[GTTA](../../pllm/workloads/gtta/workload.py) |
| 显式分页／反向／capture 优化 | [原语与资源契约](../reference/authoring-contract.md#原语与资源契约) | ESamp、Output LoRA 中的显式绑定 |

范例的 imports 反映它自己的资源绑定层级。**不要因为范例含高级资源绑定，
就把它的全部内部 imports 复制进一个普通 workload**——那会引入你的算法
并未声明的绑定层级，也让读者无法分辨哪些是契约、哪些是范例的优化选择。

## 当前支持边界

以[编写契约](../reference/authoring-contract.md)为完整依据。交付前必须
对照的硬边界：

- 普通训练当前仅支持**单优化器状态／单训练终点**。
- `state_mean` 归约目前要求**无条件的 forward/update 域**。
- **replay、模型锚点和组合有后端限制**：可表达不等于支持任意组合，后端
  能力检查会拒绝不支持的组合。

关于执行模式：自定义 Python workload 默认走进程内公共工厂。EngineCore
多进程下 workload 按引用跨进程传输（以 `pllm/contracts/workload_reference.py`
与 `pllm/worker.py` 为准）：

- workload 必须是**可导入的冻结 dataclass 类**（无字段亦可），字段值必须是
  封闭可传输的；
- **函数体从不序列化**——跨进程传递的是类坐标（module + qualname）、模块
  源码 digest 与封闭字段编码，worker 导入同一模块重建实例并重新绑定；
- driver 随后**逐字节复核** worker 重建出的编译签名是否等于自己签定的计划，
  任何代码漂移都响亮失败，而不是静默跑不同的数学。

因此多进程交付的正确做法是让双方从相同源码启动，而不是在签名与 spawn 之间
编辑模块。

## 遇到不支持的要求时

静默改算法或 fallback 到别的实现，等于替用户做了一个他们不知情的语义决定
——而框架会忠实地执行这个被改过的语义。所以规程只有一条：**给出具体缺口
（缺哪个能力、卡在哪条边界），请用户选择**（缩减需求、换算法，或扩展框架），
自己不静默改算法、不静默 fallback。

## 最小交付

- workload 源码，以及通过 `make_llm(..., workload=...)` 使用它的最小调用；
  host 算法使用 `HostRuntimeFacts`。调用示例沿用 [README](../../README.md)
  和对应范例。
- reference 来源／版本、你作出的语义选择，以及尚需用户决定的事项。
- 实际运行的针对性检查、Python 环境、结果及未验证范围。**不能把 CPU 通过
  说成 CUDA 通过**——设备与执行模式属于结论的一部分。
- 支持的设备／执行模式、资源限制。只有测过等算法、等有效工作量的对照，
  才报告性能收益。

验证入口见[验证指南](validation.md)，性能解释见[测量指南](benchmarking.md)。
例如修改 Huber 时，可在仓库根目录用
`.venv/bin/python -m pytest -q test/workloads/test_huber.py` 检查已有
reference 对照；这不是所有新算法的通用验收。类型检查入口是
`.venv/bin/python tools/check_typing.py`。其他项目使用各自的环境和相关
检查，不照搬本机路径。

若这次交付属于设计文档中的独立适配研究，再随实际工作记录澄清、重写、核心
修改与失败原因；没有记录就标“未知”——不要事后推断次数，也不把研究记录
变成普通用户的接入门槛。
