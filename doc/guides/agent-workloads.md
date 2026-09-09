# 让你的 Agent 编写 Workload

把本文、算法描述和相关代码交给你自己的 coding agent 即可。agent 在开发期产出普通源码，PLLM 运行时不调用 agent。

## 用户需要提供什么

用户只需要用自然语言说明以下四件事。没有状态、训练或外部动作时写“无”。

| 必要信息 | 用户需要作出的算法选择 |
| --- | --- |
| 数学 | 要算什么？给公式、朴素 PyTorch 实现或论文中的具体算法；输入输出的含义、形状规则、初始化和超参数是什么？有训练时说明 loss、归约、优化器及更新次数。 |
| 状态与生命周期 | 哪些值需要跨步保存？每个请求独立，还是同一实例共享？何时重置？跨请求／重试是否复用？共享参数是聚合后更新一次，还是按确定顺序多次更新？ |
| 时机与锚点 | 在哪一层、prefill／decode 的哪个阶段计算？采样前还是实际采样后？按 token、chunk 还是完成的回答触发？新参数从哪一步开始生效？ |
| 数据流与作用 | 数据来自模型激活、实际采样 token、外部标签还是 judge？结果用于观察、修改 logits、更新参数／激活、replay 还是重试？哪些请求／行参与？ |

涉及训练时，补充 forward、输入梯度、参数梯度、允许更新的状态各自的选择条件，以及
loss 按 token、请求还是共享状态归一化。可以说“所有行参与，无输入梯度”。
“不更新”与“零梯度仍执行 optimizer”不同；冻结参数也不一定截断输入梯度。
涉及 replay 时说明次数／停止条件、读取哪个更新版本、KV/RNG 和正常输出应如何变化；
涉及 host 动作时说明成功、失败、取消及重试是否允许重复执行外部动作。

另外给出已有的运行约束：模型／checkpoint、设备、dtype、典型 batch／序列长度，
建议有限交付普通正确实现，验证正确后再考虑使用pLLM中的加速技术，即使用内置的advanced精细操作。但是自动页表总是推荐开启的，这对于cuda graph兼容性好，而cuda graph的加速总是比较大的。

用户不需要提供图节点 ID、digest、物理 slot、版本号分配、receipt、手工调度顺序或
重复的 batch DTO；这些由共享框架负责。模型维度可由模型事实得到，不必手填。
若要求有限预算，agent 需核对支持范围并提供适用的成本声明，不能把缺失成本当零。

## Agent 应如何实现

1. **先确定语义，再选择入口。** 阅读仓库 `AGENTS.md`、上面的需求与 reference；
   只读下表相关的契约部分和一个最接近的范例，不要求通读迁移历史或所有 workload。
   原 reference 固定后，不得为了通过比较而同步改写它；必要的算法变更单独确认。
2. **先用普通公共接口。** 优先 `import pllm`、普通 Tensor 数学、装饰器、具名状态和
   注解输入。局部函数替换可参考 `dataclasses.replace` 示例；不要运行时 monkey-patch
   引擎或修改已经绑定的函数。注解须为实际类型，当前 binder 不求值字符串注解。
3. **沿用同一执行边界。** 逻辑输入只读，不原地修改借来的参数，不保存跨调用的借用
   tensor。沿用原图、签定计划、PTTS 状态和 receipt；不自己挑物理页、猜 latest 或维护
   第二份版本／生命周期账本。高级入口也遵守这一点，不能为统一外观把 host 算法硬塞进 Tensor。
4. **优化是显式、按需的第二步。** 先复用已有 `TensorRows`／`PagedRows` 原语；它们不拥有
   版本或发布权限。普通 eager、原生分页和 capture 是三种不同能力，不能互相冒充。
   仅在有优化需求时测量相关瓶颈，不默认写专用 kernel，不隐瞒打包、scratch 或库成本。
5. **做与变更相称的检查并交付。** 用 strict 类型检查发现不可能分支和错误注解；
   用独立数学基准检查相关输出、梯度、optimizer 字段和归约。涉及状态时覆盖无更新、
   请求隔离及相关取消／结算语义，复用已有夹具；不是要求每次重跑全套发布验收。

### 按算法选择契约和范例

| 需求 | 契约入口 | 维护中的范例 |
| --- | --- | --- |
| hidden 观察、普通数学／函数替换 | [Tensor methods](authoring.md#tensor-methods-and-state) | [Dummy](../../pllm/workloads/dummy.py)、[局部替换](../../examples/centered_hidden.py) |
| 请求独立训练、采样引导 | [状态与梯度](authoring.md#tensor-methods-and-state) | [Huber](../../examples/huber.py)、[ESamp](../../pllm/workloads/esamp_request.py) |
| 实例共享头、实际采样反馈与选择性更新 | [Tensor methods](authoring.md#tensor-methods-and-state) | [OverRIDE](../../pllm/workloads/override.py) |
| 共享／独立 LoRA、外部 batch、多步 SGD | [Weighted batch training](authoring.md#weighted-batch-training) | [Output LoRA](../../pllm/workloads/lora.py) |
| 参数 overlay、更新后条件 replay | [Request overlays](authoring.md#request-overlays) | [In-Place-TTT](../../pllm/workloads/inplace_ttt.py) |
| 激活修改／外部字段发布 | [Activation checkpoints](authoring.md#activation-checkpoints) | [ReFT](../../pllm/workloads/reft.py) |
| 逐 token prefill replay | [Tokenwise recurrence](authoring.md#tokenwise-recurrence) | [Recirculation](../../pllm/workloads/recirculation.py) |
| 完整因果模型训练适配 | [Causal adapters](authoring.md#causal-adapters) | [TLM](../../pllm/workloads/tlm/workload.py) |
| completion／reward 驱动训练 | [Completion learning](authoring.md#completion-learning) | [TTRL](../../pllm/workloads/ttrl.py) |
| judge、环境查询、队列与事务 | [Write mathematics：host 事务](authoring.md#write-mathematics) | [FTTT](../../pllm/workloads/fttt/workload.py)、[GTTA](../../pllm/workloads/gtta/workload.py) |
| 显式分页／反向／capture 优化 | [原语与资源契约](authoring.md#gradients-and-pllm-primitives) | ESamp、Output LoRA 中的显式绑定 |

不要因为范例含高级资源绑定，就把它的全部内部 imports 复制进一个普通 workload。
当前普通训练仅支持单优化器状态／单训练终点，`state_mean` 要求无条件 forward/update；
replay、模型锚点和组合有后端限制。自定义 Python workload 默认走进程内公共工厂，
EngineCore 不序列化任意用户函数。遇到不支持的要求，给出具体缺口并请用户选择，
不静默改算法或 fallback。完整边界见[编写契约](authoring.md)。

## Agent 的最小交付

- workload 源码，以及通过 `make_llm(..., workload=...)` 使用它的最小调用；host 算法
  使用 `HostRuntimeFacts`。调用示例沿用 [README](../../README.md) 和对应范例。
- reference 来源／版本、作出的语义选择，以及尚需用户决定的事项。
- 实际运行的针对性检查、Python 环境、结果及未验证范围；不能把 CPU 通过说成 CUDA 通过。
- 支持的设备／执行模式、资源限制；只有测过等算法、等有效工作量，才报告性能收益。

验证入口见[验证指南](validation.md)，性能解释见[测量指南](benchmarking.md)。例如修改
Huber 时，可在仓库根目录用 `.venv/bin/python -m pytest -q test/workloads/test_huber.py`
检查已有 reference 对照；这不是所有新算法的通用验收。类型检查入口是
`.venv/bin/python tools/check_typing.py`。其他项目使用各自的环境和相关检查，不照搬本机路径。
若用于设计文档中的独立适配研究，再随实际工作记录澄清、重写、核心修改与失败原因；
没有记录就标“未知”，不要事后推断次数，也不把研究记录变成普通用户的接入门槛。
