# 排查报错

编写或运行 pLLM 算法负载（`pllm.Workload`）时，如果遇到程序抛出异常，或者吞吐下降、显存不足、数值产生未预期偏移，本页提供快速定位和修复的具体步骤。

---

## 1. 快速分诊流程：先 CPU 后 GPU

当负载运行出现问题时，不要直接在多卡集群或大模型上盲目调试。完整的排查路线是：**先在 CPU 上用 `pllm.explain` 和 `testing.check` 几秒钟复现并定位，再到 GPU 上用 `check_on_gpu` 与 `diagnostics` 排查运行时状态**。

```mermaid
flowchart TD
    A["编写/修改负载代码"] --> B["第一步：CPU 静态检查<br/>pllm.explain(workload, config)"]
    B -->|抛出 ValueError / TypeError 等| BX["准入拒绝：检查输入源、装饰器、优化器或显存预算配置"]
    B -->|正常返回计划报告| C["第二步：CPU 合约检查<br/>testing.check(workload, config, neutral=...)"]
    C -->|抛出 ContractError| CX["逻辑缺陷：排查跨行归约、原地修改、版本时机或中性基线"]
    C -->|通过 CPU 合约| D["第三步：GPU 真机检查<br/>testing.check_on_gpu(workload, config)"]
    D -->|抛出异常| DX["底层执行：排查 CUDA Graph 捕获、主机同步或抢占重算"]
    D -->|通过 GPU 检查| E["第四步：运行期监控<br/>llm.diagnostics() / trace_snapshot()"]
    E -->|无异常但吞吐低或数值偏移| EX["对照无异常症状排查表优化算子与时序"]
```

### 第一步：在 CPU 上运行 `pllm.explain`（检查准入与计划）

`pllm.explain` 与 `pllm.make_llm` 共享完全一致的准入校验（admission check）和显存规划逻辑。它只需模型配置文件（`config.json`），在 CPU 上使用 `meta` 张量完成形状追踪，不加载任何模型权重，不初始化 CUDA，几秒内即可完成。

如果代码存在以下问题，`pllm.explain` 会立即抛出与真实启动完全相同的异常：
- 方法参数缺少输入源默认值（如缺少 `= pllm.hidden(0)`）；
- 方法未挂载任何效果装饰器（未指明结果去哪）；
- 声明了多个 `@pllm.guide_sampling`；
- 在线训练方法 `@pllm.train` 对应的状态缺少优化器声明；
- 导出张量尺寸超出单行 32 KiB 限制；
- 显存预算不足（见返回报告中的 `budget_errors`）。

即便没有抛出异常，也可检查返回的计划报告：`report.eager_reason` 是否为空（非空表示模型前向会退回 eager 模式）、各方法的 `anchor` 与 `lane`（`side` 旁路流还是 `main` 主流）是否符合预期。详见 [看成本：pllm.explain](../guides/explain.md)。

### 第二步：在 CPU 上运行 `pllm.testing.check`（检查算法契约）

当 `pllm.explain` 正常通过后，下一步在 CPU 上运行 `pllm.testing.check`。它使用固定的三个合成请求，经由真实规划器与执行器测试请求调度，检验你的张量数学是否满足框架契约：
- **请求行混淆（Row Mixing）**：是否在第 0 维误用了 `mean(0)`、`sum(0)` 或跨行排序，破坏了并发请求间的隔离性；
- **原地修改借用视图（In-place Mutation）**：是否直接对输入张量或参数执行了 `add_()`、`mul_()` 等原地写入；
- **版本可见性时机（publish_lag）**：训练更新与前向读取的版本是否符合时序预期；
- **中性基线（Neutral）**：当把干预强度设为 0 时（传入 `neutral=Workload(beta=0.0)`），干预输出是否严格为 0。

若违反契约，会抛出 `pllm.testing.ContractError`。详见 [检查正确性](../guides/checking.md)。

### 第三步：在 GPU 上运行 `pllm.testing.check_on_gpu`（验证图捕获与真机执行）

CPU 检查不执行模型内部的真实前向。对于挂载在模型内部的效果（如 `pllm.model.ActivationPatch`、`@pllm.custom_residual`、`@pllm.custom_attention`、`pllm.model.Adapter`），需要在 GPU 上使用 `pllm.testing.check_on_gpu` 进一步验证：
- 稳态解码是否被完整捕获进 CUDA Graph；
- 解码步骤中是否存在未预期的主机同步（Host sync，要求解码期间 0 次 CPU 同步）；
- 发生显存不足或抢占重算时，状态能否正确重算。

### 第四步：运行时状态诊断 `llm.diagnostics()` 与 `llm.trace_snapshot()`

在真实服务运行期间，若发现吞吐低于预期或怀疑数值异常，可查询运行时诊断信息：
- `llm.diagnostics().eager_reason`：为空字符串表示运行在整图 CUDA Graph 模式；若非空，会明确点名退回 eager 的原因；
- `llm.diagnostics().numeric_health`：检查状态张量中是否存在 NaN 或 Inf；
- `llm.diagnostics().state_usage`：监控逻辑存活字节与物理分配字节；
- `llm.trace_snapshot()`：查看主干流与旁路流各任务的耗时分布，以及是否有导出丢弃（`export_dropped`）。

---

## 最小诊断示例

下面是利用 `Qwen/Qwen2.5-0.5B-Instruct` 在 CPU 上运行 `pllm.explain` 与 `testing.check` 的完整可运行示例：

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing
from pllm.workloads.esamp import ESampConfig, ESampWorkload

# 1. 声明模型配置（隐藏层 896，24 层，词表 151936）
config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)

# 2. 准备待检查的工作负载及其强度为 0 的中性基线
workload = ESampWorkload(ESampConfig(rank=4, intervention_beta=0.5))
neutral_workload = ESampWorkload(ESampConfig(rank=4, intervention_beta=0.0))

# 3. 第一步：在 CPU 上查看执行计划与显存预算（快速捕获准入错误与 eager 风险）
report = pllm.explain(workload, config)
assert report.eager_reason == "", f"未进入 CUDA Graph: {report.eager_reason}"

# 4. 第二步：在 CPU 上验证请求隔离、版本时滞与中性基线
check_result = testing.check(workload, config, neutral=neutral_workload)
print("CPU 检查通过项:", check_result.passed)
```

若要在 GPU 上进一步检验 CUDA Graph 捕获与运行时状态，可补充以下步骤：

```python
# check-docs: skip: requires a CUDA engine and GPU
# 第三步：在 GPU 上验证模型内部算子与 0 主机同步
print(testing.check_on_gpu(workload, config, neutral=neutral_workload))

# 第四步：运行时查看引擎诊断信息
with pllm.make_llm(config, workload=workload) as llm:
    diag = llm.diagnostics()
    assert diag.eager_reason == "", f"模型退回 eager: {diag.eager_reason}"
    assert all(field.nonfinite_count == 0 for field in diag.numeric_health), "状态里出现了 NaN/Inf"
```

---

## 常见报错速查（报错 → 原因 → 改法）

每条报错的原文、原因和改法按主题列在[报错目录](../reference/errors.md)；`...` 代表随你的代码变化的部分。

---

## 运行无异常但行为不合预期（症状排查）

很多时候代码并未抛出异常，但吞吐大幅下降、显存超标或数值产生微小偏移。下表汇总此类常见症状的物理根因与针对性调优策略。关于进一步的性能调优技巧，详见 [让负载更快](faster.md)。

| 症状表现 | 物理机制与原因 | 排查与解决办法 |
| :--- | :--- | :--- |
| **`eager_reason` 非空<br>（模型退回 eager）** | 某些声明暂不支持整图捕获（如将 `pllm.model.Adapter` 挂在跨请求共享的 `InstanceState` 上）；或方法包含不可追踪操作；或使用 `@pllm.custom_forward` 时整模型图被切片图替代。退回 eager 的代价取决于步长：RTX 4090、128 并发下，0.5B 降至图模式的约 49%，7B 只慢约 2%。 | 查看 `pllm.explain` 或 `llm.diagnostics().eager_reason` 指明的原因；若需整图吞吐，将 Adapter 改为每请求独立的 `RequestState`；在配置中设置 `ModelConfig(cuda_graphs=True)`，使任何导致 eager 的写法在启动时直接报错，避免静默降级。 |
| **吞吐远低于原生 vLLM<br>（例如只有一两成）** | 常见于以下原因：<br>1. 动态注意力（如 `kv.attend(q, topk=...)`）虽减少了 KV 读取，但未经融合的打分与 gather 算子开销远超节省的带宽（如示例 10 中 `QuestRead` 在 7B 上吞吐仅为原生的 13%）；<br>2. 方法内含隐式 CPU 同步；<br>3. `placement="host"` 主机大表查表方法混读了 `pllm.hidden`，导致查表无法提前到步首 `request.prepare / side` 与前向重叠（跨 PCIe 搬运 38.9 MB 约需 1.6 ms，暴露在外直接拖慢步时）；<br>4. 声明了多层 `pllm.layer_logits`，触发多次全词表 LM Head 矩阵乘（GEMM）。 | 1. 要省 KV 显存用静态布局（`keep="window"` / `source=k`），要提速就把打分写成融合 kernel（见示例 09、10）；<br>2. 移除所有主机同步代码；<br>3. 将主机大表查表拆成仅依赖 `pllm.input_tokens` 和不可变状态的独立方法，经 `pllm.result` 传递，确保调度在 `side` 旁路流上重叠；<br>4. 用 `@pllm.guide_sampling` 替代多层 `layer_logits`，共用 LM Head。详见 [让负载更快](faster.md)。 |
| **强度为 0 时输出与原生 vLLM 存在微小差异** | 带 `@pllm.custom_residual` 或 `ActivationPatch` 时，vLLM 在 CUDA graph 里捕获模型原本的算子，不做 inductor 融合；默认的原生 vLLM 用融合后的算子，bf16 舍入略有不同。 | 这是正常的舍入差异。和 eager 模式的原生 vLLM 对比（应逐位相同），或运行 `pllm.testing.check_on_gpu`，它会建一个设置相同的原生引擎做逐位对照。 |
| **导出数据丢失行或不完整<br>（`dropped_rows > 0`）** | `@pllm.export` 具有**零阻塞保证**：为不阻塞 GPU 计算流，采用有界锁页环形缓冲区（队列上限 256 个事件，每个 $\le 32\text{ KiB}$）。当消费拉取不及时队列满时，默认直接丢弃（`overflow="drop"`，记入 `export_dropped`）；当单步有效行数超过 `max_rows`，或单次 `generate` 传入的 prompt 数超过 `max_rows` 时，超出部分会被截断（记入 `event.dropped_rows`）。 | 调大 `@pllm.export(max_rows=...)`，确保覆盖最大并发序列数或分块大小；调大导出间隔 `every=...`（如每 4 步导出一次）；确保消费进程及时读取；单次 `generate` 传入的 prompt 数量不要超过 `max_rows`。 |
| **`make_llm` 报 OOM 或图捕获预算报错点名字节数** | pLLM 为确保在线训练与状态发布完全进 CUDA Graph，会在初始化时预留固定显存池。若默认预算叠加 vLLM 权重与 KV cache 超出了物理显存容量，就会导致分配失败。`pllm.explain` 会在 `budget_errors` 中列出超标字节数。 | 对照 `pllm.explain` 的 `memory` 与 `state` 行调整 `ExecutionOptions` 参数：<br>- `state_page_budget_bytes`：状态页显存预算（调小以限制状态显存，或将大只读表设为 `placement="host"`）；<br>- `execution_peak_budget_bytes`：执行峰值预算（包含任务工作区与重叠缓冲区）；<br>- `cuda_graph_pool_budget_bytes`：CUDA Graph 显存池预算（若使用复杂切片或多个自定义方法捕获显存不足，需适当调大）；<br>- `max_host_training_rows`：主机批训练最大并行行数（调小以降低显存峰值）；<br>适当调低 `gpu_memory_utilization` 或 `max_num_seqs`。 |
| **抢占时抛出异常<br>（Preemption Rejection）** | 当显存紧张触发抢占、vLLM 试图逐出并重算序列历史时，仅当请求状态以 `request_id` 为键（私有独立副本）、且无跨请求可变实例状态、无外部导入（`Imported`）或主机外部输入时，框架才能够无害重算历史；任何其他带有覆盖（overlay）或可变共享状态的算法，由于历史无法复原，框架会拒绝重算并抛出异常。 | 提高 `gpu_memory_utilization`（如从 0.85 提至 0.90）为 KV cache 预留更多显存，从根源上避免触发抢占；调低并发序列数 `max_num_seqs`；避免在算法中依赖跨请求的可变 `InstanceState`。详见 [性能与显存](../explain/performance.md)。 |
| **旁路训练（`side`）没有和前向重叠** | `execution_peak_budget_bytes` 介于 `explain` 的 `memory` 行给出的两个阈值之间时，pLLM 把旁路任务改成主 stream 内联执行，lane 从 `side` 变为 `main`；`publish_lag=0` 也会让训练与 forward 串行。 | 看 `explain` 的 `member` 行和 `timing` 行；保持默认 `publish_lag=1`，或把预算调到重叠阈值以上。见[让负载更快](faster.md)第 3 节。 |
| **长序列递推或在线训练出现 NaN / 发散** | 累加器、carry 或 `pllm.scan` 的系数是低精度（BF16）时，舍入误差随步数累积。 | 看 `explain` 的 `state` 行里各字段的 `dtype`；用 `pllm.Field(..., dtype=torch.float32)` 显式指定，`pllm.scan` 的系数用 FP32；运行时查 `llm.diagnostics().numeric_health`。 |
| **采样器里抛出没有说明的 `AssertionError`** | `@pllm.guide_sampling` 返回的宽度不等于 hidden 宽度（Qwen2.5-0.5B 为 896）。`@pllm.add_logits` 的词表宽度在构建时就检查，guide_sampling 的宽度目前不在 CPU 上检查。 | 核对 `@pllm.guide_sampling` 返回值的最后一维等于模型的 `hidden_size`。 |
| **宿主批训练某几轮特别慢** | 训练批次的形状（序列数、长度）第一次出现时，pLLM 要为它编译训练步。 | 形状稳定后不再重编；测吞吐时用固定形状的批次并排除预热轮（见[公平地测吞吐](measure-throughput.md)）。 |
| **单进程内创建多个引擎导致崩溃或显存冲突** | 每个 pLLM 引擎会在 GPU 端独占分配图显存池、状态页和 vLLM KV cache。前一个引擎若未完全释放，显存将被锁死，导致后续引擎无法分配连续显存块。 | 使用上下文管理器 `with pllm.make_llm(...) as llm:`，退出时自动释放资源；或显式调用 `llm.close()` 释放 GPU 显存与后台 Worker 进程后再创建新引擎。用了 `@pllm.train_in_forward(visibility="exact")` 的引擎关闭时暂不能归还全部显存，之后的引擎在新的子进程里启动。 |

---

## 自测小问题

1. **问题**：在实现每请求在线学习时，你的方法被 `@pllm.train(state="adapter")` 装饰，并在函数体最后写了 `return loss.mean()`。在 CPU 上运行 `testing.check` 时抛出了 `training methods return one loss per row; pLLM reduces the selected rows per state`。为什么框架禁止用户自己在方法里调用 `.mean()`？
   <details><summary>答案</summary>
   因为连续批处理中每个 step 拼入了多个并发请求，框架需要获得每行独立的标量损失 <code>[rows]</code>，才能知道每个请求的私有状态对应的梯度，进而执行逐请求的掩码过滤与优化器更新。如果用户自己在函数内调用 <code>.mean()</code> 归约成单个标量，不同请求的损失就混合在了一起，框架无法还原各请求独立的损失值。跨行归约应当交由 <code>@pllm.train(reduction="mean"|"sum")</code> 框架自动处理。
   </details>

2. **问题**：你给模型添加了一个 `placement="host"` 的超大查表状态，并在同一个方法里既通过 `pllm.input_tokens` 查表，又通过 `pllm.final_hidden()` 读取模型最后一层激活做门控。为什么在 GPU 上运行后吞吐明显下降？
   <details><summary>答案</summary>
   因为该方法同时读取了 <code>final_hidden</code>，它必须等到整个模型前向全部计算完毕后才能启动执行。这导致跨 PCIe 搬运主机大表数据的耗时（[性能与显存](../explain/performance.md)一页中 128 行、7B 词表宽的查表约 1.6 ms/步）完全暴露在模型前向之外，无法在后台与前向重叠。正确的改法是将主机查表拆写为一个只读 <code>input_tokens</code> 的无效果方法（规划器会将其调度在步首的 <code>request.prepare / side</code> 旁路流上，利用 1/4 SM 在后台等 PCIe 搬运），再由门控方法通过 <code>pllm.result(...)</code> 读取查表结果，从而让大表搬运与主干前向深度异步重叠。
   </details>

---

## 接下来

- 查看全部错误消息清单与异常类型：[报错目录](../reference/errors.md)
- 了解如何在 CPU 与 GPU 上系统性验证契约：[检查正确性](../guides/checking.md)
- 在不启动 GPU 的前提下分析执行图与显存预算：[看成本：pllm.explain](../guides/explain.md)
- 了解避免 eager 退化与最大化重叠的优化技巧：[让负载更快](faster.md)
