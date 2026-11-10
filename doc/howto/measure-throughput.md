# 公平地测吞吐

在为大语言模型设计推理期算法时（如激活引导、每请求 LoRA、decode 期在线训练 ESamp 或查表内存），我们不仅关心算法带来的算法质量收益，更关心它相对原生基准究竟带来了多少吞吐开销。然而，大模型推理的吞吐测定极其容易受到各种隐匿因素的干扰：早停导致的 token 总量缩水、Python 发射开销带来的虚假落差、预热冷启动干扰、CUPTI 插桩的驱动污染，或者把不同的显存口径混为一谈。

这是一份面向算法研究员与工程师的操作指南。我们将建立一套无偏的基准测试标准，介绍 pLLM 提供的两组测试工具——通用的 `pllm.benchmark.throughput` 与严格对齐的 `pllm.benchmark.compare`，讲解内置 CLI 基准脚本的参数与用法，以及如何正确解读输出的吞吐、显存与性能分析数据。

---

## 1. 公平对比的六条法则与背后的原因

想要诚实地回答“算法让 vLLM 慢了多少”，核心原则是**等效工作量（Equal Work）**与**同等运行环境（Identical Environment）**。以下六条规则缺一不可：

### 1.1 完全相同的输入 Prompt 与并发批次
- **规则**：对照组与实验组必须使用完全相同的 Prompt 文本列表与批次大小（`batch`）。
- **原因**：不同的 Prompt 长度与分词结果对应完全不同的 Prefill 阶段算力；并发序列数（`batch × sampling_n`）直接决定了连续批处理张量在每步解码时的行数。若两边输入的序列长度不同，测出的吞吐差异会被 Prefill 的算力波动所掩盖。

### 1.2 强制 `ignore_eos=True` 与一致的采样参数
- **规则**：采样参数必须设置 `ignore_eos=True`，并保持 `n`（采样候选数）、`temperature`、`top_p` 与 `seed` 完全一致。
- **原因**：自回归生成的耗时与生成的 token 数量直接成正比。若不设置 `ignore_eos=True`，某个算法可能因为输出退化或倾向于输出简短回答而更早遇到 EOS 终止符；一个只生成了 50 个 token 就提早停机的请求，算出来的“每秒 token 数”不仅毫无参考价值，还会荒谬地把算法缺陷包装成“速度更快”。`ignore_eos=True` 强制所有请求生成满指定的 `tokens` 长度，保证两边执行的解码步数与生成的 token 总数绝对相等。此外，相同的采样参数保证了采样器内部的计算复杂度与状态分配完全对齐。

### 1.3 双方均开启 CUDA Graph 并保持引擎参数一致
- **规则**：实验组与原生对照组必须具有完全一致的 `ModelConfig`：**双方均必须开启 CUDA Graph**（若使用 eager 模式，则双方均必须在 `--eager` 下运行），保持相同的 `max_num_seqs`、`gpu_memory_utilization`，并且**禁用前缀缓存（prefix caching off）**。
- **原因**：
  1. 原生 vLLM 在解码期默认开启 CUDA Graph，通过预录制的图消除逐步发射算子的 CPU 开销。若将未捕获 CUDA Graph 的算法与开了图的原生 vLLM 对比，测出的主要不是算法本身的算力，而是 Python CPU 发射与 GPU 执行的巨大基础落差（在解码阶段，仅 CPU 发射开销就足以使吞吐下降约一半，详见 [性能与显存](../explain/performance.md)）。
  2. `gpu_memory_utilization`（基准统一默认为 0.25）与 `max_num_seqs` 决定了留给 KV Cache 和状态池的可用显存；显存余量不同会导致调度器规划出不同的批次行为。
  3. 前缀缓存（Prefix Caching）会在后续轮次复用 KV Cache 并跳过部分 Prefill 计算，破坏轮次之间的算力一致性。

### 1.4 排除预热轮次（Warmup）
- **规则**：在计时前必须至少执行 1 轮完整的未计时预热（`warmup=1`）。
- **原因**：首轮执行承担了 Python 模块懒加载、JIT 编译、PyTorch 显存池初始化以及 CUDA Graph 的图捕获（Capture）等一次性开销。这些开销属于冷启动初始化成本，绝不能摊销进自回归解码的稳态吞吐中。

### 1.5 测定多轮并取中位数（Median of Rounds）
- **规则**：预热后至少运行 3 轮稳态生成（`rounds=3`），取各轮吞吐的中位数（`median(round_rates)`）。
- **原因**：单次计时极易受到操作系统线程调度、后台垃圾回收、驱动上下文切换或硬件时钟瞬态抖动的干扰。多轮中位数能过滤异常值，提供可复现的稳态指标。

### 1.6 计时整次 `generate` 并显式同步设备
- **规则**：计时区间必须完整覆盖包含 Prefill 和全部 Decode 的整次 `llm.generate` 调用，并在计时终点调用 `torch.cuda.synchronize()`。吞吐定义为该次调用实际生成的输出 token 总数除以总秒数。
- **原因**：GPU 算子是向 CUDA 流异步提交的。如果不在 `generate` 结束后调用 `torch.cuda.synchronize()`，测得的只是 CPU 发射指令的时间，而非 GPU 真正完成计算的时间。同时，端到端完整请求是服务场景的核心考量，不应随意截取片断。

### 1.7 报告数字必须附带完整配置环境
- **规则**：任何吞吐数字必须与测试环境并列陈列：GPU 型号、模型全称（如 `Qwen/Qwen2.5-0.5B-Instruct`）、精度（如 bfloat16）、vLLM 与 PyTorch 版本、批次（`batch`）、并发采样数（`sampling_n`）、生成 token 数（`tokens`）以及 CUDA Graph 状态。
- **原因**：脱离并发批次、token 长度和硬件规格的绝对吞吐数字毫无物理意义，也无法被他人复现。

<details>
<summary>自测：为什么不能用动态停止（不设 ignore_eos）来测吞吐？</summary>
答：自回归生成的步数决定了计算量。如果不设 <code>ignore_eos=True</code>，生成停顿较早的差模型或较短回答会提前退出，导致实际生成的 token 数量变少；此时若直接拿“生成 token 数 / 耗时”比较，不仅无法反映真实的自回归步进速率，还会让生成质量差（过早截断）的算法由于步数少而虚假地显现出更高的数字。
</details>

---

## 2. 测量任意引擎的通用工具：`pllm.benchmark.throughput`

### 适用场景与工作原理

如果你已经有一个初始化好的模型句柄（无论是否挂载了 `Workload`，或者单纯想评估原生 vLLM 在特定 Prompt 下的性能），可以使用轻量级的 `pllm.benchmark.throughput`（接口说明见 [agent 参考](../agent/reference.md#measure-throughput)）。

```text
pllm.benchmark.throughput(llm, prompts, sampling, *, warmup=1, rounds=3) -> float
```

它接收引擎句柄、Prompts 列表与 `SamplingParams`：
1. 内部先运行 `warmup` 轮未计时的 `llm.generate`，使 CUDA Graph 录制和显存分配充分完成；
2. 执行 `torch.cuda.synchronize()` 确保设备处于空闲稳态；
3. 循环执行 `rounds` 轮完整的 `generate` 调用，在调用前后用 `perf_counter` 计时并在每次调用后同步设备；
4. 统计每轮实际生成的输出 token 数，计算各轮的 `tokens / elapsed`，返回多轮速率的**中位数**。

### 代码示例：对比原生与挂载算法的吞吐

我们首先在 CPU 上确认环境配置与选项：

```python
from dataclasses import dataclass

import torch
from torch import nn
from vllm import SamplingParams

import pllm
from pllm.benchmark import BenchmarkOptions, BenchmarkResult, compare, throughput

# 检查默认基准选项的引擎配置
options = BenchmarkOptions(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    batch=8,
    tokens=256,
    rounds=3,
    warmup=1,
)
config = options.model_config()
assert config.max_num_seqs == 128  # 8 个 prompt × 每个 16 个采样候选
assert config.cuda_graphs is True
```

接着在 GPU 上对比原生 vLLM 与一个自定义算法负载（例如 [03 decode 时训练](../tutorials/03_decode_training.ipynb) 中的在线蒸馏算法）的吞吐（需在配备 GPU 的环境下运行）：

```python
# check-docs: skip: requires a CUDA engine and GPU model
# 定义测试所用的标准并发输入与对齐采样参数
prompts = [f"Explain the mathematical idea numbered {i}:" for i in range(8)]
sampling = SamplingParams(
    n=16,
    max_tokens=256,
    temperature=0.8,
    top_p=0.95,
    ignore_eos=True,
    seed=7,
)

# 1. 测定原生 vLLM 基线速率
with pllm.make_llm(config) as plain_llm:
    plain_rate = throughput(plain_llm, prompts, sampling, warmup=1, rounds=3)

# 2. 测定挂载算法负载后的速率
with pllm.make_llm(config, workload=my_workload) as guided_llm:
    guided_rate = throughput(guided_llm, prompts, sampling, warmup=1, rounds=3)

print(f"原生 vLLM 速率   : {plain_rate:6.1f} token/s")
print(f"挂载算法后速率   : {guided_rate:6.1f} token/s")
print(f"相对吞吐保留比例 : {guided_rate / plain_rate:.1%}")
```

> [!TIP]
> `throughput` 测量的是模型前向与生成的完整端到端速率。如果你的算法涉及宿主进程中的训练协调或复杂的外部数据反馈，请配合使用 `llm.trace_snapshot()` 观察宿主侧任务与设备任务的具体耗时，详见 [宿主循环与批训练](../guides/host-loops.md)。

---

## 3. 标准化对齐基准：`pllm.benchmark.compare`

### 自动化等效工作量对比

当你需要正式发布算法性能报告或向他人汇报时，手动写对照组容易出现配置漂移。`pllm.benchmark.compare` 封装了标准的对齐对比流程：

```text
pllm.benchmark.compare(
    workload,
    options,
    *,
    prepare=None,
    max_host_training_rows=1,
    require_training=False,
    prompt_indices=None,
    alternatives=(),
) -> tuple[BenchmarkResult, ...]
```

`compare` 的自动化流程包括：
1. **自动测定原生基线**：在与算法完全相同的 prompts、sampling 和 CUDA Graph 配置下启动原生 vLLM，测出 `vanilla_tokens_per_second` 并记录在每个结果中；
2. **测量各个算法臂**：依次构建主算法臂以及 `alternatives` 中传入的备选实现变体；
3. **等效工作量契约检查**：验证各臂实际生成的 token 总数严格等于 `batch × sampling_n × tokens × rounds`，并且在运行结束时检查没有未释放的状态副本或资源泄漏；
4. **统一格式输出**：打印标准 JSON 报告并返回 `BenchmarkResult` 元组。

### 关键参数规则：`require_training` 的使用边界

在使用 `compare` 时，必须特别注意 `require_training` 参数：

- **仅限在线训练类负载设为 True**：当算法包含在线训练（即声明了带有 `@pllm.train` 的方法，如 ESamp 或每请求 LoRA）时，必须设置 `require_training=True`。框架会严格断言 `min(report.loss_count, report.update_count, report.trained_rows) > 0`，确保反向传播和优化器更新在 GPU 上真正执行了，防止训练逻辑被静默跳过；
- **非训练负载必须保持 False（默认）**：对于仅读取模型、激活引导（`@pllm.guide_sampling`）、探针导出（`@pllm.export`）或残差改写的算法，**不要**开启 `require_training=True`。因为这些算法没有训练算子，训练计数必然为 0，开启该标志会导致基准断言失败抛出异常。

### 代码示例：运行标准比较

```python
# check-docs: skip: requires a CUDA engine to run compare
options = BenchmarkOptions(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    batch=8,
    tokens=256,
    rounds=3,
    warmup=1,
    gpu_memory_utilization=0.25,
)

# 非训练负载保持 require_training=False（默认）
results = compare(my_workload, options, require_training=False)

for res in results:
    ratio = res.tokens_per_second / res.vanilla_tokens_per_second
    print(f"实现臂 [{res.implementation}]:")
    print(f"  稳态吞吐 : {res.tokens_per_second:.1f} token/s")
    print(f"  原生对比 : {res.vanilla_tokens_per_second:.1f} token/s（相对比例 {ratio:.1%}）")
    print(f"  各轮速率 : {res.round_rates}")
    print(f"  规划显存 : {res.planned_peak_memory_bytes / 1024**2:.1f} MiB")
    print(f"  实测净增 : {(res.peak_allocated_bytes - res.baseline_allocated_bytes) / 1024**2:.1f} MiB")
```

---

## 4. 使用内置 CLI 脚本测定标准算法场景

pLLM 仓库在 `tools/` 目录下提供了数个维护中的基准测试场景脚本。它们直接封装了 `BenchmarkOptions.parse`，无需编写任何胶水代码即可通过命令行运行：

- **`tools/benchmark_esamp.py`**：测试 ESamp（解码期在线训练小网络并进行采样引导）。默认 rank 4，支持 `--beta` 调节引导强度，显式开启 `require_training=True`；
- **`tools/benchmark_lora.py`**：测试 Output LoRA。支持 `--scope instance|request|both`（实例共享或每请求私有）、`--rank`、`--max-training-rows`、`--training-fraction` 与 `--state-groups`；
- **`tools/benchmark_engram.py`**：测试 Engram 式 N-gram 查表内存算法。对比大表分别放置在主机内存（`--placement host`）与显存（`--placement device`）时的解码性能损耗。

### 运行命令示例

在终端中直接运行内置基准：

```bash
# 运行 ESamp 基准，指定批次 8、生成 256 token、测定 3 轮
python -m tools.benchmark_esamp --batch 8 --tokens 256 --rounds 3

# 运行 Output LoRA 基准，测试每请求私有状态
python -m tools.benchmark_lora --batch 8 --tokens 256 --rounds 3 --scope request

# 在 24 GiB 的卡上测 7B：显式选择基座模型的显存比例；捕获与状态页自动定容
python -m tools.benchmark_esamp --model Qwen/Qwen2.5-7B-Instruct --gpu-memory-utilization 0.80
```

默认按规划得到的需求为张量任务捕获池定容，状态页使用后端可用容量；原生模型图的池由 vLLM 管理。预算参数只在需要显式限制资源时设置，过小的上限会报错。explain 仍报告任务的捕获预留，排除 vLLM 的模型图与激活池。

### 公共命令行参数一览

所有内置基准脚本共享相同的参数集：

| 参数 | 类型与默认值 | 物理含义与考量 |
| :--- | :--- | :--- |
| `--model` | `str`，默认 `"Qwen/Qwen2.5-0.5B-Instruct"` | 基座模型路径或 Hugging Face 模型名。 |
| `--batch` | `int`，默认 `8` | 并发 Prompt 数量。 |
| `--tokens` | `int`，默认 `256` | 每个序列生成的目标 token 数量。 |
| `--rounds` | `int`，默认 `3` | 稳态计时的轮数。 |
| `--warmup` | `int`，默认 `1` | 排除在外的预热轮数。 |
| `--sampling-n` | `int`，默认 `16` | 每个 Prompt 生成的独立候选序列数。总并发序列数为 `batch × sampling_n`（默认 128）。 |
| `--gpu-memory-utilization` | `float`，默认 `0.25` | vLLM 显存分配比例上限。留出充足显存供算法状态与工作区使用。 |
| `--eager` | `bool` 标志 | 强制所有算法臂与原生基线关闭 CUDA Graph，以 eager 模式运行。 |
| `--cuda-graph-pool-budget-mib` | `int`，可选，默认自动定容 | 每个张量任务及模型捕获池的显存上限（MiB）。不足时会明确报错拒绝。 |
| `--state-page-budget-mib` | `int`，默认 `0`（后端可用容量） | 设备端状态页的容量上限（MiB）。使计时与重放环境可复现。 |
| `--execution-peak-budget-mib` | `int`，可选 | 限制规划期的执行峰值显存（含状态、工作区与临时张量，不含模型/KV）。 |
| `--guidance-capacity` | `int` 或 `None`，默认 `4096` | 采样引导仅针对前 K 个保留词；设为 None 则执行全词表稠密引导。 |
| `--profile` | `Path`，可选 | 导出 Chrome trace（`.json`）的目录。详见第 6 节。 |

### 内置对齐设置（Aligned Throughput Settings）

内置脚本固化了一套严密的推理超参数，避免任何隐式变量污染对比：
- **精度**：统一使用 `bfloat16`；
- **EOS 机制**：必须设置 `ignore_eos=True`；
- **缓存策略**：强制关闭前缀缓存（Prefix Caching）；
- **采样设置**：温度 `temperature=0.8`、`top_p=0.95`、`top_k=-1`、随机种子 `seed=7`；
- **模型长度限制**：`max_model_len = max(512, tokens + 64)`；
- **批处理 token 规模**：`max_num_batched_tokens = max(512, batch * sampling_n * 64)`。

修改这些参数即意味着定义了一个新的测试形状，报告时不可与标准形状混为一谈。

---

## 5. 解读基准输出：吞吐、显存与状态使用

基准测试运行后，控制台会输出包含结构化指标的 JSON 行。正确解读这些字段是避免误判的关键：

### 5.1 核心吞吐指标

- **`tokens_per_second`**：主要汇报指标。它是各轮吞吐的中位数。
- **`round_rates`**：每轮实测速率的元组。
  - *排查提示*：如果各轮速率波动超过 5%，通常表明 GPU 受到宿主内存交换、显存颠簸或多进程竞争干扰。
- **`vanilla_tokens_per_second`**：在完全相同硬件、Prompts 与采样配置下原生 vLLM 的中位数速率。
- **`vs_vanilla`**：即 `tokens_per_second / vanilla_tokens_per_second`，直观反映挂载算法后的吞吐保留百分比（例如 0.92 代表吞吐为原生的 92%，开销为 8%）。

### 5.2 显存口径对比：规划上限与分配器实测不可互换

报告中会并列出现两组显存数字，它们的物理定义截然不同：

1. **`planned_peak_memory_bytes`（静态规划峰值）**：
   - 这是编译器在 CPU 静态规划阶段（通过 [`pllm.explain`](../guides/explain.md)）根据算法声明推导出的**保守理论上限**。
   - 它精确涵盖了所有声明的请求状态页、张量工作区、捕获池预留与存活中间变量；它**不包含**底座模型自身权重、KV Cache 块以及未受控的外部库内存。
2. **`peak_allocated_bytes` 与 `peak_extra_allocated_bytes`（分配器实测峰值）**：
   - 这是 PyTorch 的 CUDA 显存分配器在运行时实际记录的显存分配水位。
   - `baseline_allocated_bytes` 是预热完成后的基线显存；`peak_extra_allocated_bytes = peak_allocated_bytes - baseline_allocated_bytes` 代表执行算法引入的净增显存峰值。

> [!CAUTION]
> **保守规划界与分配器实测值绝对不可互换！**
> 规划界是静态证明与安全配额，而分配器实测值受 PyTorch 底层内存池复用策略、内存碎片与内存对齐影响。在撰写论文或技术报告时，绝不能将规划界声称为“实测显存”，也不能将单次运行的实测净增当作所有形状下的理论保证。

### 5.3 状态使用与训练计数

- **`state_usage`**：轮末状态页的物理快照，反映状态显存的物理分配与利用率（包括已保留页数、逻辑字节、物理字节、内存填充与页所有权）。通过它可观察是否存在严重的状态显存碎片。
- **`loss_count`、`update_count`、`trained_rows`**：对于在线训练算法，分别记录在测量轮次中实际执行的 Loss 计算次数、优化器更新步数与累计训练行数。必须确保它们大于 0 且符合数学期望。

---

## 6. 深入分析耗时：为什么 `--profile` 必须在计时后独立重放

当你需要用性能分析器（Profiler）排查内核瓶颈时，可以直接在 CLI 脚本后加上 `--profile DIRECTORY` 参数。pLLM 在执行 Profiling 时采用了一种特殊的**两阶段机制**：

1. **第一阶段：纯净计时**：所有算法臂首先在没有挂载任何 Profiler 的干净环境下跑完所有轮次，输出基准吞吐；
2. **第二阶段：独立重放与插桩**：重新初始化模型引擎并完成预热，在完全相同的输入与测量轮次下挂载 PyTorch Profiler，并把每个臂的 Chrome trace 存到指定目录下以臂名命名的 `.json` 文件。

### 为什么不能边计时边做性能分析？

这一设计的原因是 **CUPTI 驱动污染**：
- PyTorch Profiler 依赖 NVIDIA 的 CUPTI（CUDA Profiling Tools Interface）底层驱动库。
- CUPTI 插桩在 Profiler 的上下文退出之后，仍可能让后续的 CUDA API 调用变慢。
- 所以先在同一进程里做 Profiling 再测吞吐，测出的吞吐会偏低；边插桩边计时，分析本身的开销也会混进吞吐数字。

因此，pLLM 始终让纯净计时在前、插桩分析在后。**在编写自己的测试脚本时，也绝对不要在同一个 Python 进程内跑完 Profiler 后再次启动稳态计时比较**。

导出的 trace 文件可用 Chrome 浏览器（访问 `chrome://tracing`）或 Perfetto 打开。报告会输出 GPU 真实活跃时间（`gpu_activity_microseconds`）、内核执行总数（`kernel_count`）、CUDA 图发射次数（`cuda_graph_launch_api_count`）以及内存拷贝字节数。注意：trace 中的 API 时间包含主机同步等待，不是纯 CPU 提交时间；重叠执行的内核时间是算术相加的，不等于物理墙钟时间。

---

## 7. 实用加速技巧：何时启用 `VLLM_USE_FLASHINFER_SAMPLER`

### 采样器瓶颈的成因

在标准基准设置中，每个 Prompt 会采样 16 个候选（`sampling_n=16`），8 个 Prompt 构成了 128 条并发请求的大批次。此外，许多算法（如 ESamp、DoLa 或对比解码）会修改 logits 或挂载采样引导（`@pllm.guide_sampling`）。

在面对大批次并发与自定义 logits 偏置时，vLLM 原生的采样内核可能会花费可观的时间处理 CPU 调度、logits 拷贝与 GPU 归约。若采样内核本身耗时过大，整个生成的吞吐会被采样器拖慢，反而掩盖了算法模型前向本身的真实性能。

### 开启 FlashInfer Sampler

如果运行环境中安装了 FlashInfer，建议在运行基准测试前通过环境变量开启融合采样加速：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
```

启用后，vLLM 用 FlashInfer 的采样算子代替默认实现。它会改变两边（原生与负载）的采样耗时，所以两边要在同一设置下测，并在报告里写明是否打开。

---

## 8. 常见报错与排查

在配置与执行吞吐基准时，常见错误通常源于参数配置超出物理限制或违反了测试契约：

| 报错信息 | 触发原因 | 正确做法 |
| :--- | :--- | :--- |
| `positive batch, tokens, rounds, warmup and capture/state capacities required` | `BenchmarkOptions` 中传入了小于 1 的批次、token 数、轮次、预热数或显存预算。 | 检查传入参数，确保 `--batch`、`--tokens`、`--rounds`、`--warmup` 及各预算参数均为正整数。 |
| `workload needs cuda_graph_pool_budget_bytes >= ... for its conservative task capture reservations; got ...; leave it unset to size from the workload` | 手动指定的 `--cuda-graph-pool-budget-mib` 小于算法任务捕获所需的保守显存。 | 调大该参数至报错提示的数值以上，或在构造 `ExecutionOptions` 时不要显式设小该上限。 |
| `ExecutionOptions.state_page_budget_bytes=... exceeds the ... bytes free on the device; lower it, or lower gpu_memory_utilization ...` | 设定的状态页显存预算超过了设备上的实际剩余可用显存。 | 调小 `--state-page-budget-mib`，或者适当降低 `--gpu-memory-utilization` 以便 vLLM 为系统留出更多可用显存。 |
| `raise execution_peak_budget_bytes to fund the custom forward's workspace and graphs` | 设定的 `--execution-peak-budget-mib` 无法容纳改写 forward 的工作区、图保留与中间值。 | 增大该参数的显存上限，或留空不作人为约束。 |

此外，如果 `compare` 抛出 `AssertionError`：
- 若错误发生在断言 `min(report.loss_count, report.update_count, report.trained_rows) > 0`，说明你在非训练负载中误传了 `require_training=True`。请移除非训练负载中的该参数；
- 若错误发生在输出 token 校验，说明模型因未设置 `ignore_eos=True` 而提前退出了某些请求，破坏了等效工作量契约。

---

## 下一步

- [看成本：pllm.explain](../guides/explain.md)：在不上 GPU 的情况下，在 CPU 上推导算法的显存峰值与 CUDA Graph 捕获状态；
- [让负载更快](faster.md)：优化状态布局、算子融合与并行流，进一步缩减算法开销；
- [让一组请求共享训练成果](share-training-across-requests.md)：探索按 Prompt 或路由键共享状态时的吞吐特征；
- [性能与显存](../explain/performance.md)：深入理解 CUDA Graph 回放、显存规划与连续批处理的底层机制。
