# 检查正确性

写好负载之后、相信它给出的任何数字之前，先确认它在真实服务的调度下算的就是你想算的东西。本页讲 pLLM 提供的三层检查。

---

## 为什么需要检查正确性

在单请求的常规 PyTorch 调试中，一个写错维度的算法往往不会抛出任何异常。例如，把特征归一化误写为 `h - h.mean(dim=0)`，只要 batch 中只有一个请求，第 0 维的长度就是 1，输出与正确的 `h - h.mean(dim=-1)` 完全一致，单元测试也能顺利通过。

但在 vLLM 连续批处理（continuous batching）的服务调度下，运行时会将多个并发请求的 token 动态打包进同一个张量批次：
- **请求行混淆**：第 0 维动态排布着不同请求的行。一旦错误地沿 `dim=0` 做 `mean`、`sum`、`softmax` 或按行排序，一个请求的激活值就会渗入另一个请求，导致生成内容无故损坏。
- **只读输入被原地篡改**：方法的输入张量与模型参数是执行器共享缓冲区的只读视图。若在方法体内执行 `w.add_(1)` 或 `h.mul_(2)` 等原地修改，其他并发任务或后续调度步读到的就是被污染的数据。
- **状态版本读错**：在线训练与采样引导存在时序依赖，`publish_lag` 明确限定了状态更新对后续计算可见的步数窗口。如果方法读到了尚未提交或超出许可范围的状态版本，计算就会失去确定性。

为了在接触真实 GPU 流量前排除这些隐蔽缺陷，pLLM 提供了三层互为补充的检查机制：
1. **CPU 契约检查（`pllm.testing.check`）**：数秒内用真实的规划器与执行器在 CPU 上运行多套固定的三请求调度，验证请求隔离、行重排、分块、抢占重放与版本可见性。
2. **GPU 引擎检查（`pllm.testing.check_on_gpu`）**：在真实 GPU 引擎中跑满 8 步连续解码，触发强制 KV cache 逐出与重算，检验 CUDA graph 捕获、无宿主同步以及自定义算子的设备等价性。
3. **GPU 分布偏移度量（`pllm.testing.check_distribution`）**：以基座模型自生成的 token 做 teacher forcing，度量全词表双向 KL 散度与 TV 距离的均值及 P99 尾部，拦截破坏罕见 token 分布的隐藏崩溃。

---

## 第一层：在 CPU 上检查契约（pllm.testing.check）

`pllm.testing.check` 不需要 GPU，也不加载模型权重。它只读取模型的 `config.json` 元数据，在 CPU 上构建真实的规划器和执行器，让三个构造好的请求经历行重排、扰动输入、混合 prefill/decode、提示词分块以及抢占重放等多种服务调度变体，对比每份请求状态副本的输出是否严格一致。

### 快速上手与零强度对照（neutral=）

最划算的正确性检验是提供一个把干预强度设为 0 的对照实例 `neutral=`（例如将引导系数 `beta` 设为 `0.0`）。如果一个干预算法在强度为 0 时依然改动了模型 logits，或者依然声明了活跃的采样器阶段，说明算法的状态绑定或逻辑分支存在泄漏。

```python
from dataclasses import dataclass
import torch
from torch import nn
import pllm
from pllm import testing

@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    # 声明每请求状态：每个请求拥有一份独立的小型网络与 AdamW 优化器副本
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        # 每行独立计算 loss，返回一维张量 [rows]，由 pLLM 负责按状态归约
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(self, shallow: torch.Tensor = pllm.hidden(0),
                f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow.float())

config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
result = testing.check(Distill(), config, neutral=Distill(beta=0.0))
print(result)
```

输出如下：

```text
status  contract / reason
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    publish_lag
PASS    preempted
PASS    neutral
```

### 契约检查项全表

`pllm.testing.check` 逐项验证以下契约：

| 契约项 | 触发失败的具体场景 |
| --- | --- |
| `compile` | 负载无法通过规划器准入（如方法缺少效果装饰器、未给输入参数指定数据源标记、优化器与训练作用域不匹配）。 |
| `row permutation`、`request isolation` | 某个请求的输出取决于其在批次中所处的行位置，或者受同批其他请求的输入影响（例如沿 `dim=0` 归约，或在方法间残留了 Python 属性状态）。 |
| `mixed prefill/decode`、`chunking / tokens=` | 当长提示词被切分成多个 chunk 处理，或预填充行与解码行混合打包在同一步时，同一 token 算出的结果发生漂移。 |
| `preempted` | 请求因显存压力被抢占后，重新回放其历史 token 无法精确复现先前的状态张量。 |
| `publish_lag` | 方法读取了尚未提交或超出 `publish_lag` 允许范围的状态版本。 |
| `neutral` | 传入 `neutral=` 后算法仍对基座模型产生了影响：`@pllm.add_logits` 返回了非零修正，或 `@pllm.guide_sampling` 未将有效系数设为 0。 |

### 如何阅读 PASS 与 SKIP

`print(result)` 输出的每一行状态代表一个契约的判定结果：
- `PASS`：该契约在多套 CPU 调度变体下得到了精确验证。
- `SKIP`：明确列出该项契约在 CPU 上跳过的具体原因。

**为什么部分项目会被 SKIP**：在模型站点上挂载的执行逻辑（如 `pllm.model.ActivationPatch`、`pllm.model.Adapter`、`@pllm.custom_residual` 的层内钩子、`@pllm.custom_forward` 的跨层展开等）需要加载基座模型权重和真实的 KV cache。CPU 检查只负责验证它们的操作数运算（operand math）与状态转移逻辑，不会在 CPU 上模拟整套模型的前向推理，因此这些模型站点应用会明确标记为 `SKIP: model-site execution: ...`，需要通过第二层的 GPU 检查来覆盖。

### 结构化的 CheckResult 与 ContractError

- **`CheckResult`**：检查通过时返回的不可变对象，包含 `passed`（通过的契约名称元组）与 `skipped`（跳过项元组，每项包含 `property` 和 `reason`）。通过 `dataclasses.asdict(result)` 可以将其转换为标准的 Python 字典，直接用于 CI 流程的 JSON 归档与断言。
- **`ContractError`**：继承自 `AssertionError`。只要任何一项契约未通过，检查立即抛出该异常。其 `failures` 属性是一个元组，列出了所有检测到的差异，并明确指出是哪个用户方法跨行混淆，还是框架内部的路由差异。

### 捕获只读篡改与行混淆

如果用户代码对输入的参数执行了原地写入（如 `w.add_(1)`），或者沿行维度做了归约（如 `h.mean(dim=0)`），`pllm.testing.check` 会立即捕获并报错：

```python
class BadRowMix(pllm.Workload):
    @pllm.export(name="probe", every=1, max_rows=8, max_cols=32)
    def probe(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        # 错误：沿 dim=0 归约跨越了不同的请求行
        return h - h.mean(dim=0, keepdim=True)

try:
    testing.check(BadRowMix(), config)
except testing.ContractError as err:
    print(f"捕获到契约违背，共 {len(err.failures)} 处失败：")
    for failure in err.failures:
        print(" -", failure)
```

在这个例子中，`testing.check` 会直接捕获异常并指明方法 `probe` 混淆了行：`Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0)`。

---

## 第二层：在 GPU 真实引擎上检查（pllm.testing.check_on_gpu）

CPU 检查通过是算法正确的前提，但不是充分条件。模型站点上的实际修改（如权重注入、残差改写）、CUDA graph 的静态执行序列、以及真实的显存分页分配，都必须在 GPU 上才能得到完整检验。

`pllm.testing.check_on_gpu` 会加载真实的 vLLM 引擎，执行以下严格测试：
1. **8 步平稳解码（steady decode）**：在无任何 GPU 到 CPU 宿主同步（host sync）的前提下，验证连续 8 步解码完全在 CUDA graph 内平稳运行。
2. **强制 KV cache 逐出与重算**：人为制造显存配额耗尽，强制触发 vLLM 原生的序列抢占、逐出和上下文重算，验证请求状态副本在极端显存压力下能否无损恢复。
3. **零强度逐位对齐（bitwise neutral）**：配合 `neutral=`，验证贪心解码与设定了 seed 的随机采样生成的 token ID 与纯净基座模型（plain model）完全逐位一致。
4. **自定义算子设备等价性**：如果负载引入了自定义算子，它会把 CPU 检查脚本中的输入回放到 CUDA 上执行，并验证输出与 CPU 参考实现是否在数据类型容差内保持一致。
5. **因果批更新等价性**：对于因果批训练，打乱输入序列与权重的顺序，验证 AdamW 优化器更新后的权重保持一致。

### 运行环境与参数约束

调用 `pllm.testing.check_on_gpu` 必须遵循以下环境约束：
- **独立的全新进程**：由于 vLLM 初始化、显存池锁定与 CUDA graph 捕获会深度绑定当前 Python 进程状态，GPU 检查必须在一个全新的子进程中独立运行，不能与其它大型任务复用进程。
- **单进程与单设备配置**：必须设置 `multiprocessing=False`，且张量并行、流水线并行及数据并行度必须全部为 1（`tensor_parallel_size=1` 等）。检查套件直接探测进程内的执行引擎，不支持跨 worker 进程编排。
- **容量下限**：必须设置 `max_num_seqs >= 2` 且 `max_model_len >= 32`。固定测试脚本需要同时运行多个并发请求并触发抢占回放，容量过小会导致调度脚本无法加载。

```python
# check-docs: skip: check_on_gpu 需要在独立的包含 GPU 的进程中运行真实引擎
gpu_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    max_model_len=128,
    multiprocessing=False,
    gpu_memory_utilization=0.3,
)
gpu_result = testing.check_on_gpu(Distill(), gpu_config, neutral=Distill(beta=0.0))
print(gpu_result)
```

---

## 第三层：度量分布偏移（pllm.testing.check_distribution）

在 `neutral=` 下通过了逐位一致性检查，说明干预被完全关闭时算法是干净的；但当干预强度大于 0（`beta > 0`）时，你怎么知道算法没有悄悄毁掉模型的输出分布？

在许多算法实验中，平均指标（如平均交叉熵或整体 BLEU）看起来很正常，但个别罕见的 token 位置由于 logits 异常放大产生了严重的数值爆炸。平均值会掩盖这种极端的异常点，而 P99 和最大值能直接暴露出危险的长尾。

`pllm.testing.check_distribution` 采用 **Teacher Forcing** 在 GPU 上度量全词表输出分布的漂移程度：
1. **生成参考序列**：基座模型基于预置的测试 prompt（默认两条序列提示词）自生成固定步数（默认 `tokens=16`）的续写。
2. **强制对齐前向**：负载在开启状态更新与采样引导的情况下，强制以这些 token 作为输入前向推进，确保两个引擎处于完全相同的上下文和批次几何构型中。
3. **全词表 FP32 距离统计**：直接在设备端基于未截断的原始 logits 计算全词表 `log_softmax`，度量三种距离：
   - `KL(plain || workload)`：基座模型到负载模型的 KL 散度（单位为 nats）。
   - `KL(workload || plain)`：负载模型到基座模型的反向 KL 散度。
   - `TV`：全词表总变差距离（Total Variation，即 L1 概率距离的一半）。
4. **统计摘要与极值定位**：为每种距离返回 `DistanceSummary`，包含 `mean`（均值）、nearest-rank `p99`（99 分位点）、`max`（最大值）以及最严重的几个 token 位置（`worst`，标注 prompt 序号与 token 偏移）。

### 度量与断言

- **纯度量模式**：不传阈值参数时，该函数作为纯观测工具运行，返回完整的 `DistributionResult` 数据，绝不会抛出 `ContractError`。
- **阈值断言模式**：传入 `max_mean_kl` 和 `max_p99_kl` 时，这两个阈值会同时约束两个方向的 KL 散度。如果超过阈值或出现 NaN，`result.passed` 将变为 `False`，并在 `result.failures` 中记录违背详情，但依然保留所有统计数字供分析。
- **它度量什么与不度量什么**：它度量的是在给定测试序列上全词表概率分布的位移幅度；它**不度量**下游任务的生成质量，也不代表未测试的长文本分布。

```python
# check-docs: skip: check_distribution 需要在包含 GPU 的进程中加载基座模型与采样器
dist_result = testing.check_distribution(
    Distill(),
    gpu_config,
    max_mean_kl=0.01,
    max_p99_kl=0.1,
)
print(dist_result)
# beta=0.5 的 Distill 会明显推开分布：在一张 RTX 4090 上实测 KL(workload || plain) 均值约 0.049、P99 约 0.20，两个阈值都没通过
print(f"通过: {dist_result.passed}")
```

---

## 结果汇报规范

在团队讨论、论文撰写或算法评估中，汇报正确性检查结果必须严格遵循以下规范：

> [!IMPORTANT]
> **必须明确写明哪些检查在 CPU 上完成，哪些检查在 GPU 上完成，以及具体的 Python 与硬件环境。**
> 
> 禁止笼统宣称“负载已通过测试”。CPU 契约检查通过仅证明调度与状态隔离在数学上是封闭的，但由于 CPU 明确跳过了模型站点上的权重注入（`ActivationPatch`、`Adapter`、`custom_residual`）与实际 KV cache 显存管理，**CPU 通过绝不等于 GPU 通过**。

汇报的推荐表述范式：
- “在 CPU 环境下通过 `pllm.testing.check`（包含 `neutral=` 对照，通过了 compile、row permutation、request isolation、publish_lag 与 preempted 契约）。”
- “在单卡 <GPU 型号>、<Python 环境>（`multiprocessing=False`）下通过 `pllm.testing.check_on_gpu`（8 步 steady decode 无 host sync，KV cache 强制逐出与重算通过，`neutral=` 贪心采样逐位一致）。”
- “用 `pllm.testing.check_distribution` 在默认提示词上测得：全词表 KL(plain || workload) 均值 <数值> nats，P99 <数值> nats。”

---

## 无模型单元测试：底层单步辅助工具

当你只想为方法里的某个复杂张量运算编写纯单元测试，连模型的 `config.json` 都不想读取时，可以使用 `pllm.testing` 提供的底层编译与单步驱动工具：

| 工具接口 | 功能说明 |
| --- | --- |
| `pllm.testing.backend_facts` | 构造在进程内模拟执行的基础事实（隐藏层宽度、挂载的隐藏层插桩、显存上限等），代替 vLLM 探测的元数据。 |
| `pllm.testing.hidden_binding(layer, layer_count)` | 生成与 `pllm.hidden(layer)` 对应的输入插桩绑定。 |
| `pllm.testing.final_hidden_binding(layer_count)` | 生成与 `pllm.final_hidden()` 对应的最终隐藏层插桩绑定。 |
| `pllm.testing.compile_execution(workload, facts)` | 编译负载并分配其实例状态，返回执行器句柄。 |
| `pllm.testing.RequestIdentity(request_id, prompt_index, sample_index)` | 构造代表单个请求的身份元数据对象。 |
| `pllm.testing.step_input(identities, hidden=...)` | 手工组装一步打包好的输入数据（每行对应一个请求）。 |
| `pllm.testing.run_step(execution, value)` | 驱动执行器执行该步的所有锚点任务并产出结果。 |

以下是一个完整的纯 CPU 单元测试示例：

```python
class Centered(pllm.Workload):
    @pllm.export(name="centered", max_rows=2, max_cols=4)
    def center(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return h - h.mean(dim=-1, keepdim=True)


binding = testing.hidden_binding(0, layer_count=1)
execution = testing.compile_execution(Centered(), testing.backend_facts(4, (binding,), layer_count=1))
rows = (testing.RequestIdentity("a", 0, 0),)
execution.admit(rows)
h = torch.randn(1, 4)
testing.run_step(execution, testing.step_input(rows, {binding: h}))
exported = next(event.tensor() for event in execution.trace.snapshot().events if event.tensor_name == "centered")
torch.testing.assert_close(exported, h - h.mean(dim=-1, keepdim=True))
execution.retire(rows, wait=True)
```

---

## 常见报错

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[testing.check 的 ContractError](../reference/errors.md#testingcheck-的-contracterror)一节。其余构建时的报错按主题列在[报错目录](../reference/errors.md)。

---

## 自测

<details>
<summary>1. 为什么一个包含了 @pllm.custom_residual 的负载在 CPU 上通过 testing.check 后，输出表格中仍会显示 SKIP model-site execution？</summary>

因为 `@pllm.custom_residual`、`ActivationPatch` 以及 `Adapter` 等模型站点修改需要注入到 Transformer 骨干网络的层间前向计算中。这依赖于真实的模型权重、注意力内核与 GPU KV cache 显存。CPU 契约检查只负责使用真实规划器验证方法体内部的操作数运算、状态更新时序与跨行隔离契约，无法在 CPU 上完成整套大模型推理，因此会将实际层内应用标记为 `SKIP`，需要通过 `testing.check_on_gpu` 在真实 GPU 引擎上补齐测试。
</details>

<details>
<summary>2. 为什么 testing.check_on_gpu 强制要求 multiprocessing=False，并且必须运行在独立的全新进程中？</summary>

vLLM 底座引擎在初始化时会分配独占的 GPU 显存池并进行 CUDA graph 的预热捕获。如果与先前的脚本复用同一进程，先前的显存碎片和 CUDA context 会破坏图捕获环境。同时，`check_on_gpu` 需要直接探查进程内部引擎的执行器与张量状态，无法穿透多进程 worker 的通信管道，因此必须在独立的单进程（`multiprocessing=False`）环境中运行。
</details>

<details>
<summary>3. 既然有了 neutral= 逐位对齐检查，为什么还需要 check_distribution？</summary>

`neutral=` 检查的是算法完全关闭（强度为 0）时的基线等价性；而 `check_distribution` 检验的是算法正常开启（强度大于 0）时的数值健康度。许多算法在平均意义下工作良好，但在少数罕见的 token 位置会因为 logits 极端偏移而产生数值发散。`check_distribution` 通过计算全词表的双向 KL 散度与 TV 距离，并提取 P99 和最大异常点，能够精准捕捉这种被平均指标掩盖的长尾数值崩溃。
</details>

---

## 相关文档

- [看成本：pllm.explain](explain.md)：在 CPU 上静态查看显存预留、CUDA graph 捕获状态与计算重叠计划。
- [负载的组成](workload-anatomy.md)：深入理解三类声明、方法体编写规则与版本可见性。
- [自写算子](custom-ops.md)：如何为负载引入 Triton 算子并注册 CPU 参考实现以支持契约检查。
- [排查报错](../howto/troubleshooting.md)：完整的报错信息目录与排错指引。
- [API 参考](../reference/api.md#检查正确性)：`pllm.testing` 与全部公开接口的签名和说明。
