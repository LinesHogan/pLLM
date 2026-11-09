# 上生产：多进程、张量并行与量化底座

在 Notebook 或本地脚本中验证完算法之后，将负载（`pllm.Workload`）推向生产服务通常面临三类工程需求：将推理引擎隔离进独立子进程（`multiprocessing=True`）以服务在线流量、在多张 GPU 上开启张量并行（Tensor Parallelism）以承载更大模型，以及在量化底座（FP8、AWQ、GPTQ）上保持高效运行。

本文介绍这三项能力的配置方法、约束条件及其背后的运行机制。

---

## 开启多进程：EngineCore 隔离与序列化要求

在本地开发调试时，负载与引擎默认运行在同一个 Python 进程中（`multiprocessing=False`）。在生产部署时，为了将请求接入与模型执行解耦、避免 Python 全局解释器锁（GIL）干扰并发处理，应通过 `ModelConfig(multiprocessing=True)` 将执行引擎放入独立的 EngineCore worker 进程中。

驱动端（driver）通过 `pickle` 将负载对象序列化传递给 EngineCore worker。

```python
from dataclasses import dataclass

import torch
from torch import nn
import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class StreamMonitor(pllm.Workload):
    @pllm.export(name="norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


# 在 CPU 上检查规划和调度契约
report = pllm.explain(StreamMonitor(), CONFIG)
check_result = testing.check(StreamMonitor(), CONFIG)
```

要在生产服务中以多进程模式启动引擎：

```python
# check-docs: skip: starts a GPU engine in multiprocessing mode
from vllm import SamplingParams

from my_project.workloads import StreamMonitor  # 多进程时负载类要从可导入的 .py 模块里来

mp_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    multiprocessing=True,
)

with pllm.make_llm(mp_config, workload=StreamMonitor()) as llm:
    outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=8))
```

在多进程模式下，负载代码必须遵守以下约束：

1. **负载类与模块级函数必须位于可导入的 `.py` 模块中**  
   **原因**：Python 的 `pickle` 协议序列化类和函数时仅记录其导入路径（模块名与限定类名），而不包含函数体字节码。worker 子进程启动后会按路径重新 `import` 所在模块。如果类定义在 `__main__`、Notebook 单元格或函数局部作用域内，worker 子进程无法在文件系统中找到对应的模块路径，反序列化会直接失败。
2. **负载字段必须支持 pickle 序列化（禁止 lambda 与闭包）**  
   **原因**：标准 `pickle` 无法序列化动态构造的闭包环境与匿名函数（lambda）。负载实例的字段（超参数配置）应当使用原生数据类型（如 int、float、str、tuple 等）。
3. **模块必须具备导入安全性（import-safe）**  
   **原因**：worker 子进程被拉起时，在初始化 CUDA 运行时与 GPU 显存之前就会在顶层 `import` 你的模块。如果在模块顶层执行了分配 GPU 张量或触碰 CUDA 上下文的代码，子进程会在设备上下文就绪前崩溃。
4. **两端计划逐字段校验（驱动端与 worker 必须运行相同源码）**  
   **原因**：worker 反序列化出负载后，会在本地重新推导执行计划，并核对任务语义图与张量布局是否与驱动端完全一致。如果部署时两端源码不一致或预训练文件被篡改，引擎会在安装时明确报错拒绝运行，防止因两端计算逻辑漂移而产生未知的数学错误。
5. **模型对象不可作为负载字段**  
   **原因**：基座模型权重通常达数 GB 至数十 GB，且已经由 worker 底层加载并驻留在 GPU 显存中。负载字段不能持有重复的模型权重。如果算法需要在方法内调用基座模型（如因果语言模型训练），应在方法内使用 `pllm.causal_model("adapter")`，它会直接复用 worker 中已加载的基座模型。

<details>
<summary>自测：为什么把模型对象存入负载字段会在多进程模式下失败？</summary>

因为基座模型占用大量显存且与特定的 CUDA 设备上下文绑定，无法通过 `pickle` 跨进程安全传输；此外 EngineCore worker 已经拥有加载好的基座模型，重复传递不仅浪费资源，还会引发反序列化错误。方法内应当使用 `pllm.causal_model` 访问已加载的模型。
</details>

---

## 多卡张量并行（TP > 1）

当单张卡显存不足以容纳整个模型权重，或者需要更高显存带宽时，可以通过 `ModelConfig(tensor_parallel_size=2, ...)` 开启多卡张量并行（Tensor Parallelism）。

在张量并行下，必须使用 vLLM 0.32.0 与 `runner="mrv2"`：

```python
# check-docs: skip: starts a multi-GPU engine with tensor parallelism
from vllm import SamplingParams

tp_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    tensor_parallel_size=2,
    runner="mrv2",
)

with pllm.make_llm(tp_config, workload=StreamMonitor()) as llm:
    outputs = llm.generate(["Tensor parallel run"], SamplingParams(max_tokens=8))
```

### 运行机制：零跨卡集合通信与错峰发布

张量并行下，vLLM 在每一层注意力与 MLP 之后都必须进行一次全卡同步（Custom All-Reduce）。如果 pllm 自身也引入跨卡集合通信，或者旁路流上的训练计算与 vLLM 的 All-Reduce 产生总线带宽竞争，所有卡都会发生严重停顿。

为此，pllm 采取两项设计：
1. **零 pllm 跨卡集合通信**：隐藏状态读取（`pllm.hidden`、`pllm.residual`、`pllm.final_hidden`）、采样引导（`@pllm.guide_sampling`）、Logits 调整（`@pllm.add_logits`）和请求级状态在每个 TP rank 上各存一份完全相同的副本，各卡独立执行相同的确定性计算，彼此不交换数据。
2. **流优先级与错峰发布**：将张量并行推理主流设为高优先级（`-1`），旁路训练流设为普通优先级（`0`），并让状态更新在采样前写入、在 token 提交时发布，避免旁路计算与 All-Reduce 撞车。

### TP > 1 的支持边界

| 特性分类 | 在 TP > 1 下的表现 | 说明与原因 |
| :--- | :--- | :--- |
| **隐藏状态与采样** | 完全支持 | `pllm.hidden`、`pllm.residual`、`pllm.final_hidden`、`@pllm.guide_sampling`、`@pllm.add_logits` 在各卡本地副本上确定性计算。 |
| **请求状态在线训练** | 完全支持 | 例如 ESamp 等请求状态训练，各卡独立更新本地副本，无跨卡通信开销。 |
| **自定义前向** | 完全支持 | `@pllm.custom_forward` 在各卡本地分片上推进层执行。 |
| **适配器权重分片** | 支持 dense / low-rank | `pllm.model.Adapter` 的 dense 与 low-rank 增量与投影分片对齐：q/k/v 和 gate/up 投影按输出行切分权重与因子 B；down 投影按输入列切分权重与因子 A；另一因子各卡复制。由 `@pllm.train_in_forward` 在本地分片上更新。 |
| **导出（`@pllm.export`）** | 仅由 rank 0 交付 | 所有 rank 上的计算结果完全一致，因此导出事件只由 rank 0 送回 CPU，避免重复传输。 |
| **必须设为 TP=1 的特性** | 直接拒绝报错 | `operator="per_head"` 的适配器增量、带优化器或非 `Zeros` / `Computed` 初始化的适配器状态、经 `pllm.causal_model` 的整模型训练、外部状态投喂（`pllm.Fed`）、`@pllm.custom_residual`、`@pllm.custom_attention`（单卡只能看到本地分配的头，无法做全局注意力重写），以及 `pllm.testing.check_on_gpu`（须先在 TP=1 下验证）。 |

> [!IMPORTANT]
> **TP 不会减少单卡上请求状态的显存占用**：`pllm.explain` 在报告显存时会提示 `reservations are per rank`。张量并行平摊的是底座模型的权重显存，但 pllm 的请求状态在每张卡上均保留完整副本，因此请求状态占用的显存不会随卡数增加而减少。

<details>
<summary>自测：开启 TP=2 后，为什么单卡上的请求状态显存没有减半？</summary>

因为张量并行切分的是底座模型的投影层权重，而 pllm 为了保证零跨卡通信与确定性执行，在每个 TP rank 上都独立维护一份请求状态副本。因此单卡上的请求状态显存需求保持不变。
</details>

---

## 量化底座：FP8、AWQ 与 GPTQ

在生产部署中，服务通常使用 FP8、AWQ 或 GPTQ 量化权重以削减显存占用并提升访存受限阶段的 decode 吞吐。

pllm 将量化底座作为一等公民支持：
1. **模型准入规则相同**：量化检查点遵循与稠密模型相同的家族准入规则（如 Qwen2/2.5、Llama、Qwen3 等，详见[支持矩阵](../reference/support-matrix.md)）。
2. **LoRA 类增量不反量化底座**：通过 `pllm.model.Adapter` 或 `@pllm.train_in_forward` 施加的权重增量，就像 vLLM 原生对量化模型应用 LoRA 一样，在量化矩阵乘法算子的输出上直接累加高精度（BF16 或 FP32）的修正项 `(x @ A.T) @ B.T`，底座权重全程保持量化状态，不需要在显存中展开为稠密矩阵。
3. **整模型训练按需反量化**：经 `pllm.causal_model` 训练时，采用类似 QLoRA 的机制：前向使用量化算子计算，反向传播求导时临时对参与计算的权重进行反量化，兼顾显存节省与导数计算。
4. **物化稠密权重**：在 `@pllm.train_in_forward` 中如果声明读取 `pllm.operand("weight")`，pllm 会在显存中物化出一份稠密的浮点权重副本供方法使用。

---

## 引擎生命周期与资源清理

在长期运行的服务进程或批处理流水线中，可能需要在一个 Python 进程内先后创建并销毁多个引擎实例。务必使用上下文管理器管理引擎生命周期：

```python
# check-docs: skip: demonstrates engine lifecycle
from vllm import SamplingParams

config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)

with pllm.make_llm(config, workload=StreamMonitor()) as llm:
    llm.generate(["Prompt 1"], SamplingParams(max_tokens=8))

# 离开 with 代码块会自动调用 llm.close()
```

离开 `with`（或调用 `llm.close()`）时，引擎关闭并释放它的编译状态和 CUDA graph 内存，所以同一个进程里可以反复创建、关闭引擎。

### GPU 契约检查的环境要求

`pllm.testing.check_on_gpu` 在内部会构建真实引擎并验证八步稳态解码、强制 KV 驱逐与重算等调度契约。由于它需要完全控制底层引擎的生命周期，必须在独立的进程中调用，且要求：
- `multiprocessing=False`；
- `max_num_seqs >= 2`；
- `max_model_len >= 32`。

```python
# check-docs: skip: check_on_gpu requires a dedicated GPU process
testing.check_on_gpu(
    StreamMonitor(),
    pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4, max_model_len=64),
)
```

详见[检查正确性](../guides/checking.md)。

---

## 显存旋钮与请求抢占（Preemption）

在生产环境中，单卡 GPU 显存主要由四部分构成：
$$\text{总显存} = \text{底座模型权重} + \text{CUDA Graph 内存池} + \text{pllm 状态与执行工作区} + \text{vLLM KV Cache 块}$$

模型权重、CUDA Graph 内存池与 pllm 的状态显存优先分走固定份额，剩下的显存全部归 vLLM 的分页 KV Cache 使用。

### 核心显存旋钮

在 `pllm.ModelConfig` 与 `pllm.ExecutionOptions` 中配置显存边界：

```python
# 调整执行预算与调度限制
service_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=16,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=0.7,
    execution=pllm.ExecutionOptions(
        state_page_budget_bytes=0,
        cuda_graph_pool_budget_bytes=128 * 1024 * 1024,
    ),
)
service_plan = pllm.explain(StreamMonitor(), service_config)
```

- **`gpu_memory_utilization`**（默认 0.5）：分配给推理引擎的显存比例。调高该值可为 KV Cache 提供更多显存，但在大并发或大模型下需防止 GPU OOM。
- **`max_num_seqs`**（默认 8）：单步允许并发处理的序列数上限。它直接决定了 `RequestState` 为并发请求预先分配的槽位数。
- **`max_num_batched_tokens`**（默认 512）：单步内并发请求累计的最大 token 数量（包含 prefill 与 decode）。
- **`state_page_budget_bytes`**（默认 0）：限制请求状态分配的设备页面显存上限；设为 0 表示使用框架观测到的空闲容量。
- **`execution_peak_budget_bytes`**（默认 None）：限制状态页、任务工作区与中间输出的总峰值显存。
- **`cuda_graph_pool_budget_bytes`**（默认 None）：任务与模型的 CUDA Graph 私有内存池上限（默认模型池为 64 MiB）。
- **`max_host_training_rows`**（默认 1）：每步每个状态由主机投喂训练的最大行数（请求解码行训练单独统计）。

所有执行选项的详细参数见[配置项](../reference/config.md)。在上线前，务必使用 `pllm.explain`（详见[看成本：pllm.explain](../guides/explain.md)）在 CPU 上核对显存预留峰值。

### 请求抢占与重算（Preemption & Recompute）

当并发请求激增导致 KV Cache 块耗尽时，vLLM 调度器会主动抢占（preempt）正在生成的请求，驱逐其 KV Cache 块，待显存空闲后再重新计算其历史。

被抢占的请求重新调度时，vLLM 会重算它的历史。这对负载意味着什么，`pllm.explain` 的 `preempt` 行已经写明：

- `state keep`：请求状态在抢占期间保留，重算历史不影响它；
- `state reset`：状态要回到起点、随历史重算重新推导。请求状态按 `request_id` 区分、没有可变的实例状态、没有外部 feed 或导入时，重算能复现原来的状态，引擎直接重算；否则 `preempt` 行写出不能重算的原因，真的发生抢占时报错：
  `vLLM preempted a request and would recompute its history; ... To avoid preemption, give vLLM more KV cache (gpu_memory_utilization) or fewer concurrent sequences (max_num_seqs)`

**防范措施**：上线前看 `preempt` 行；属于不能重算的负载时，提高 `gpu_memory_utilization` 给 KV cache 留足空间，或调小 `max_num_seqs`，避免触发抢占。

---

## 前向改写与 Prefix Caching 的互斥

vLLM 提供了前缀缓存功能（Prefix Caching，`enable_prefix_caching=True`），能够对相同前缀的 prompt 复用已有 KV Cache 块，从而显著降低长上下文 prefill 的计算耗时。

在 `ModelConfig` 中，`enable_prefix_caching` 默认为 `False`。**凡是改写模型前向计算的负载，必须保持该选项关闭，若显式开启会在准入时报错拒绝。**

### 互斥原因

前缀缓存的工作原理是对 prompt token 序列计算哈希值，当发现匹配时直接复用缓存中的 KV 块。

当负载改写了模型内部计算时（使用了 `ActivationPatch`、`Adapter`、`@pllm.custom_residual` 或 `@pllm.custom_attention`），生成的 KV 张量不再单纯由 prompt 文本决定，还会受到当前请求独有的适配器权重、激活补丁或动态算子影响。如果复用前缀缓存，后续请求就会读到未经本请求修改的旧 KV 数据，导致计算结果与算法语义不一致。因此，改写模型站点的负载严格互斥前缀缓存。

---

## 常见报错与排查

在部署到多进程、多卡张量并行或调整生产配置时，若遇到准入拒绝或运行时异常，对照下表排查：

| 报错信息 | 产生原因与排查方法 |
| :--- | :--- |
| `worker processes cannot send ... defined in __main__ or a notebook; move it into an importable .py module, or use tensor_parallel_size=1 with multiprocessing=False` | **原因**：设置了 `multiprocessing=True`，但负载类定义在 `__main__`、Notebook 单元格或函数内部，子进程无法通过模块路径导入该类。<br>**排查**：将负载类移入一个在 `sys.path` 中的 `.py` 文件中，并通过 `from your_module import YourWorkload` 导入；或在单进程调试时设为 `multiprocessing=False`。 |
| `worker processes receive the workload by pickling (...); hold module-level functions instead of lambdas or closures, or use tensor_parallel_size=1 with multiprocessing=False` | **原因**：负载实例字段或配置中包含了无法被 `pickle` 序列化的 lambda 匿名函数或闭包对象。<br>**排查**：将配置改写为顶级具名函数、常量或原生数据类型，避免在类属性中捕获闭包。 |
| `the EngineCore worker imported a workload (code or pretrained weights) that differs from the driver's; run both processes from the same installed sources and do not rewrite pretrained files while the engine starts` | **原因**：驱动端与 worker 子进程导入的源码或预训练权重不一致，导致推导出的执行计划产生分歧。<br>**排查**：确保所有进程运行自同一套安装好的源码，且不要在引擎启动过程中修改 `.py` 文件或覆写预训练权重文件。 |
| `multi-GPU needs vLLM 0.32.0 and runner="mrv2"; otherwise use tensor_parallel_size=1` | **原因**：配置了 `tensor_parallel_size > 1`，但运行环境不是 vLLM 0.32.0，或未配置 `runner="mrv2"`。<br>**排查**：升级环境至 vLLM 0.32.0 并在 `ModelConfig` 中设置 `runner="mrv2"`；在 vLLM 0.27.1 下请保持 `tensor_parallel_size=1`。 |
| `@pllm.custom_attention needs tensor_parallel_size=1: each rank sees only its heads` | **原因**：在张量并行模式下声明了 `@pllm.custom_attention`。<br>**排查**：多卡下注意力头被切分，单卡无法看到全量注意力头。自定义注意力读取目前需在 `tensor_parallel_size=1` 下运行。 |
| `disable prefix caching (enable_prefix_caching=False) when a workload rewrites model sites; cached KV would not match` | **原因**：负载包含残差改写或注意力改写，同时开启了 `enable_prefix_caching=True`。<br>**排查**：保持 `ModelConfig(enable_prefix_caching=False)`。模型改写使得 KV Cache 与请求状态相关联，无法跨请求按 prompt 文本复用。 |
| `vLLM preempted a request and would recompute its history; ... To avoid preemption, give vLLM more KV cache (gpu_memory_utilization) or fewer concurrent sequences (max_num_seqs)` | **原因**：KV Cache 耗尽触发抢占，但负载的状态转移不可重放（如包含可变实例状态或外部投喂），无法安全重算历史。<br>**排查**：调高 `gpu_memory_utilization` 扩充 KV Cache 空间，或调小 `max_num_seqs` 降低并发以避免发生抢占。 |

关于更多报错排查，参考[报错目录](../reference/errors.md)。
