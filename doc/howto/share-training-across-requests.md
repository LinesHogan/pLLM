# 让一组请求共享训练成果

默认情况下，`pllm.RequestState` 按请求严格隔离：每个请求在准入（admission）时获得一份独立私有的状态副本，随请求解码独立更新，并在请求完成退役后释放显存。

但在许多算法中，状态应当被**一组相关的请求共同拥有并持续演化**：
- **同 Prompt 的多个采样（Best-of-N / 自我一致性）**：一次生成抽取 $N$ 条候选文本，所有采样在解码时共同训练同一个在线适配器（LoRA）或探针。
- **跨调用的多轮会话或多租户服务**：同一个会话（Session）、同一个租户（Tenant）或同一类业务主题（如数学、代码）的请求，跨越多次 `llm.generate` 调用持续更新并复用同一个适配器。
- **全局实例共享**：整个推理引擎的所有请求彻底共享唯一一份全局状态。

`pllm` 通过在状态中声明路由键 `key_fields` 来表达请求之间的分组与生命周期。本指南介绍如何配置不同的共享粒度、规划容量，以及避免组训练中的常见陷阱。

---

## 1. 同一 Prompt 的多个采样共享：`key_fields=("prompt_index",)`

### 问题与动机

在并行采样（例如设置 `SamplingParams(n=4)`）时，同一个 Prompt 会分叉出 4 个并行解码的兄弟采样。如果使用默认的 `RequestState`，每个采样都会分配一份独立的适配器副本，各自闭门造车，无法汇聚群体探索的信息。

如果希望同一次生成中所有采样探索到的知识共同更新同一个适配器，需要让它们共享同一份状态副本。

### 声明方式

在声明 `RequestState` 时，将 `key_fields` 显式设为 `("prompt_index",)`：

```python
import torch

import pllm

adapter = pllm.RequestState(
    {"weight": lambda m: (m.hidden_size, m.hidden_size)},
    optimizer=pllm.AdamW(lr=1e-3),
    key_fields=("prompt_index",),
)
```

在执行 `llm.generate(prompts, sampling_params)` 时，`pllm` 会自动为每个请求标记其所属 Prompt 的序号 `prompt_index`。所有具有相同 `prompt_index` 的采样行，在连续批处理中都对齐并操作同一份物理状态。

### 共享组训练的归约建议：为什么必须选 `reduction="sum"`

在编写 `@pllm.train` 方法时，逐请求训练的方法针对当前批次的每个请求行返回一个标量损失，由框架在状态内部进行归约：

```python
@pllm.train(state="adapter", reduction="sum")
def learn(
    self,
    h: torch.Tensor = pllm.hidden(0),
    w: torch.Tensor = pllm.parameter("adapter", "weight"),
) -> torch.Tensor:
    # 针对当前批次中属于本状态的每一行计算损失
    return (torch.bmm(h.unsqueeze(1), w).squeeze(1) - h).square().mean(-1)
```

**为什么共享 `key_fields` 组训练时强烈建议使用 `reduction="sum"` 而不是默认的 `reduction="mean"`？**

- **默认 `reduction="mean"` 的隐患**：`"mean"` 会将本步该状态选中的 $n_s$ 个活跃行的损失求均值（即除以 $n_s$）。在一组并发的兄弟采样中，由于各采样的生成长度不同（遇到 EOS 的时间不同），样本通常会在不同的 decode 步提前停止退出。如果使用 `"mean"`，随着部分样本完成，分母 $n_s$ 逐渐变小（例如从 4 变到 1）。这会导致**剩余未结束样本中每个 token 贡献的梯度被动放大数倍**，破坏了训练步长的平稳性。
- **`reduction="sum"` 的保障**：`"sum"` 直接将本步选中的所有行损失求和。每个存活的 token 对梯度的贡献只取决于它本身的损失大小，无论此时恰好有 4 个兄弟样本在并发生成，还是只剩最后 1 个样本在独跑，每个 token 贡献的梯度权重始终恒定。

关于在线损失归约的更多数学细节与优化器交互，参见 [在线训练](../guides/training.md)。

---

## 2. 跨请求与跨调用共享：业务路由键 `key_fields=("route",)`

### 问题与动机

许多在线学习算法需要超越单次 `generate` 的范围：例如长驻对话中的多轮会话（同一 session_id 持续更新对话记忆）、按用户租户划分的轻量级专家（同一 tenant_id 复用专属 LoRA），或是按领域自动路由的分类器（math 任务训练数学适配器，code 任务训练代码适配器）。

此时，状态的分组逻辑由外部宿主业务决定，并且状态必须在单次 `generate` 结束后继续存活。

### 声明与调用

#### 1. 在状态中声明路由键

将 `key_fields` 声明为 `("route",)`：

```python
adapter = pllm.RequestState(
    {"weight": lambda m: (m.hidden_size, m.hidden_size)},
    optimizer=pllm.AdamW(lr=1e-3),
    key_fields=("route",),
)
```

#### 2. 在宿主代码中传入路由值

在调用 `llm.generate` 时，通过 `routes` 参数为每个 prompt 传入一个路由字符串：

```python
# check-docs: skip: requires a CUDA engine and GPU generate
from vllm import SamplingParams

prompts = [
    "请推导微积分基本定理",
    "写一个快速排序算法",
    "请证明勾股定理",
]


def route_of(prompt: str) -> str:
    # 宿主业务代码根据分类、正则或租户计算路由键，pllm 从不解释其含义
    if "定理" in prompt:
        return "math"
    return "code"


routes = tuple(route_of(p) for p in prompts)
# routes 为 ("math", "code", "math")

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

with pllm.make_llm(CONFIG, workload=workload) as llm:
    # 第一次调用：第 0 和第 2 条请求共享 "math" 副本，第 1 条请求使用独立的 "code" 副本
    outputs_1 = llm.generate(prompts, sampling_params, routes=routes)

    # 第二次调用：再次传入包含 "math" 路由的请求，直接复用此前在显存池中已训练好的副本
    follow_up_prompts = ["请解释柯西不等式的几何意义"]
    outputs_2 = llm.generate(follow_up_prompts, sampling_params, routes=("math",))
```

### 路由值的本质与隔离语义

- **由宿主代码全权计算**：路由值是一个普通的 Python 字符串，可以由正则表达式、外部判别器模型打分、租户 ID 或会话 ID 产生。
- **`pllm` 从不解释路由内容**：框架不关心字符串的具体含义，仅将其作为底层状态寻址的不透明标识。
- **共享与隔离**：
  - **相同 route 的请求加入同一份状态副本**：无论这些请求是在同一个批次中并发到达，还是跨越多次 `generate` 调用先后到达，它们读取并更新的都是同一个物理状态副本。其中任何一条请求产生的参数更新，对该 route 下的所有请求均即时生效。
  - **不同 route 之间绝对隔离**：`"math"` 和 `"code"` 拥有完全独立的参数张量与优化器槽位，互不干扰。

---

## 3. 路由状态的生命周期与容量规划

### 为什么路由状态在请求结束后不会消失

- **普通请求状态**（`key_fields=("request_id",)`）：随着请求生成完毕并退役，其状态副本被立即释放，显存归还给内存池。
- **路由状态**（`key_fields=("route",)`）：当某一个 route 的最后一个活跃请求退役后，其状态副本**不会被释放**，而是以**休眠（dormant）**状态存活在引擎显存池中，依然与该路由键保持绑定。当后续的 `generate` 调用再次传入相同的 route 值时，引擎直接唤醒该副本并继续累积训练，直到整个引擎关闭（`llm.close()`）。

### 容量规划：为 `max_num_seqs` 预留余量

休眠的状态副本虽然没有活跃请求在执行前向，但**仍然占用引擎的准入容量额度**。

在构建引擎时，`pllm.ModelConfig` 的 `max_num_seqs` 参数规定了引擎能够同时维持的状态副本容量上限。在采用路由键时，你的稳态资源占用由以下公式决定：

$$\text{稳态容量占用} = \text{活跃与休眠的 route 基数} \times \text{每份状态副本的字节数}$$

**必须为 `max_num_seqs` 预留足够余量**：如果系统只按瞬时最大并发请求数配置 `max_num_seqs`，随着不同 route 的请求先后到达并退役，休眠的路由副本会逐渐占满准入额度。当新的 route 到达时，由于没有空闲的状态槽位，请求会在准入阶段被直接拒绝。因此，在规划配置时，应将 `max_num_seqs` 设定为能够容纳预期的不同 route 总数加上并发请求缓冲。

---

## 4. 全局完全共享：何时使用 `InstanceState`

如果你的算法语义是“整个推理引擎中的所有请求，彻底共享唯一一份物理参数”（即无需按 prompt 或业务 route 分组），不要使用 `RequestState`。

直接使用 `pllm.InstanceState`：

```python
global_table = pllm.InstanceState(
    {"table": lambda m: (10000, m.hidden_size)},
    init=pllm.Zeros(),
)
```

### 四种状态共享粒度对比

| 声明方式 | 共享范围 | 存活生命周期 | 适用场景 |
| :--- | :--- | :--- | :--- |
| `RequestState(key_fields=("request_id",))`（默认） | 单请求严格私有 | 请求完成即退役释放 | 单请求独立 LoRA、单请求在线蒸馏 |
| `RequestState(key_fields=("prompt_index",))` | 同一 Prompt 的多个采样共享 | 该 Prompt 的全部采样完成后释放 | Best-of-N 组探索、采样间协同引导 |
| `RequestState(key_fields=("route",))` | 具有相同业务路由值的请求共享 | 请求完成后休眠存活，直到引擎关闭 | 多轮对话会话、分租户适配器、跨调用持续学习 |
| `InstanceState(...)` | 整个引擎实例内所有请求共享 | 引擎创建时初始化，直到引擎关闭 | 全局冻结探针、跨请求大型词表、模型级全局统计量 |

详细的状态作用域与字段声明机制参见 [状态](../guides/state.md)。

---

## 5. 完整可运行示例：声明与成本检查

下面是一个完整的、可在 CPU 上直接运行的算法声明与执行计划检查代码。算法声明了一个按路由键共享的在线适配器，并在 decode 阶段采用 `reduction="sum"` 最小化重构损失：

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class SharedRouteLoRA(pllm.Workload):
    # 存什么：按业务路由键共享的适配器权重与 AdamW 优化器
    adapter = pllm.RequestState(
        {"weight": lambda m: (m.hidden_size, m.hidden_size)},
        optimizer=pllm.AdamW(lr=1e-3),
        key_fields=("route",),
    )

    # 结果去哪：解码阶段在线训练，sum 归约保持各 token 贡献权重恒定
    @pllm.train(state="adapter", reduction="sum")
    def learn(
        self,
        h: torch.Tensor = pllm.hidden(0),
        w: torch.Tensor = pllm.parameter("adapter", "weight"),
    ) -> torch.Tensor:
        # [rows, 1, H] @ [rows, H, H] -> [rows, H]
        reconstructed = torch.bmm(h.unsqueeze(1), w).squeeze(1)
        return (reconstructed - h).square().mean(-1)


# 在 CPU 上分析规划，不占用 GPU 显存
workload = SharedRouteLoRA()
plan = pllm.explain(workload, CONFIG)

print(f"CUDA Graph 模式: {plan.eager_reason or 'CUDA graphs: no effect needs eager'}")
print(f"状态名称: {plan.states[0].name}, 作用域: {plan.states[0].scope}")
```

通过 [`pllm.explain`](../guides/explain.md)，你可以在启动真实 GPU 服务之前，清晰检查状态副本的物理占用与 CUDA Graph 捕获状态。

---

## 6. 检查共享是否生效

当负载编写完毕后，如何验证共享与隔离机制是否如预期工作？推荐以下两种自检方法：

### 1. 零状态确定性检查（Zero-state Determinism）

验证状态通道是否正确绑定的最简捷手段是利用初始零状态的确定性：
1. 向引擎同时发送两个**全新且不同**的路由请求（例如 `"route_a"` 与 `"route_b"`）。
2. 在输入 prompt 完全相同的情况下，由于两个路由的初始参数均为零，它们在**第 1 步计算出的训练损失（loss）应当完全相等**。
3. 当 `"route_a"` 经过数步训练更新后，再次向引擎发送一个 `"route_a"` 请求和一个从未见过的全新 `"route_c"` 请求。此时 `"route_a"` 的首步损失应当明显偏离初始值，而 `"route_c"` 仍呈现出标准的零状态损失。若 `"route_a"` 与 `"route_c"` 仍然相同，说明更新未成功发布或路由值未正确映射。

### 2. 利用 `llm.diagnostics()` 监控运行时状态

通过 `llm.diagnostics()` 可以实时获取引擎内部的计数指标：
- `trained_rows`：实际参与优化器更新的总行数。
- `update_count`：优化器执行权值更新的总步数。
- `loss_count`：计算出的损失次数。
- `active_states`：引擎当前分配并保持的状态副本总数。可以通过该数值验证具有相同路由的请求是否复用了同一个状态副本，而不是重复创建新副本。

关于诊断对象的全部字段说明，参见 [explain 与 diagnostics 字段](../reference/diagnostics.md)。

---

## 7. 常见报错与排查

当使用路由键时，最容易遇到的错误通常源自状态声明与调用传参不一致。

### 为什么缺少路由值的请求会被直接拒绝？

当状态声明了 `key_fields=("route",)` 时，如果一条请求没有提供路由值，`pllm` **会在准入（admission）阶段直接报错拒绝该请求**。

**原因**：路由键是状态声明的一部分，请求用哪份副本完全由它决定。没有路由值的请求不知道该用哪份；悄悄给它一份默认副本，会把不相干的请求混进同一份训练状态（例如两个租户共用了一个适配器），所以准入时直接拒绝。

### 常见错误目录

下表列出了与路由键相关的核心报错信息及其排查方式：

| 报错信息 | 触发原因 | 解决办法 |
| :--- | :--- | :--- |
| `routes= needs a request state keyed by "route" and one value per prompt` | 调用 `llm.generate(prompts, ..., routes=...)` 时，负载未声明 `key_fields=("route",)`，或者 `routes` 的长度与 `prompts` 的数量不相等，或者包含了空字符串。 | 检查状态声明是否包含 `key_fields=("route",)`，并确保为输入的每个 prompt 都提供了一个非空的有效路由字符串。 |
| `a route-keyed request state requires per-prompt routes` | 负载中声明了带有 `"route"` 的 `RequestState`，但在调用 `llm.generate` 时未传入 `routes=` 参数。 | 在宿主代码中通过 `routes=tuple(route_of(p) for p in prompts)` 传入路由值。 |
| `RequestIdentity ... is missing a field required by ...` | 底层请求准入时缺少声明的键字段值（例如声明了 `"route"` 却传入了空字符串，或声明了 `"prompt_index"` 却传入了负数）。 | 检查请求身份构造或宿主路由生成函数，确保所有声明的键字段均具有合法非空值。 |
| `request key fields must be closed, sorted, and unique` | 声明 `key_fields` 时字段顺序不符合规范，或者包含重复字段。规范顺序必须严格按照 `("request_id", "prompt_index", "sample_index", "route")` 的子序列排列。 | 调整字段顺序，例如写为 `key_fields=("prompt_index",)` 或 `key_fields=("route",)`。 |
| `key_fields=...: use names from ...` | 在 `RequestState` 的 `key_fields` 中传入了未知的键名称。 | 仅使用支持的合法字段名：`"request_id"`、`"prompt_index"`、`"sample_index"` 或 `"route"`。 |
| `batch, sample_index, prompt_indices and routes need workload= in make_llm` | 在调用 `pllm.make_llm` 时未传入 `workload=`（即作为原生 vLLM 实例启动），但在调用 `generate` 时传入了 `routes=` 等专属于负载的参数。 | 在启动引擎时显式传入负载：`pllm.make_llm(CONFIG, workload=workload)`。 |

更多报错排查指南参见 [排查报错](troubleshooting.md) 与 [报错目录](../reference/errors.md)。

---

## 自测

1. 为什么在声明了 `key_fields=("route",)` 后，即使并发请求数很小，`max_num_seqs` 依然不能设得太小？
<details>
<summary>答案</summary>

因为带有业务路由键的状态副本在最后一个请求退役后不会被释放，而是以休眠状态继续存活并占用准入容量（`max_num_seqs` 额度）。如果系统访问过的 route 种类很多，休眠副本会逐渐耗尽配额，导致携带新路由到达的请求无法准入。
</details>

2. 在对同一 Prompt 的多个兄弟采样进行在线组训练时，为什么损失归约推荐用 `reduction="sum"` 而不是默认的 `reduction="mean"`？
<details>
<summary>答案</summary>

兄弟采样通常由于生成长度不同而在不同时间步退出。若使用 `"mean"`，当部分样本结束退出后，活跃行数分母 $n_s$ 变小，会导致剩余未结束样本中每个 token 贡献的梯度被动放大数倍；而 `"sum"` 不受当前活跃行数变化的影响，保证每个 token 贡献恒定的梯度。
</details>

3. 能否在宿主代码中把 `routes` 设为整数类型（如 `routes=(1, 2, 1)`）？
<details>
<summary>答案</summary>

不能。`routes` 要求传入由字符串组成的元组（每个 prompt 一个非空字符串）。如果需要根据整数 ID 分组，请将其转换为字符串传入，如 `routes=("tenant_1", "tenant_2", "tenant_1")`。
</details>

---

## 下一步

- 了解在解码循环中训练状态与设置梯度边 → [在线训练](../guides/training.md)
- 深入掌握状态的作用域、字段与初值来源 → [状态](../guides/state.md)
- 启动前在 CPU 上查看显存分配与捕获状态 → [看成本：pllm.explain](../guides/explain.md)
- 排查更多运行时与准入错误 → [排查报错](troubleshooting.md)
