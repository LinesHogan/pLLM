# pllm 性能指南：为什么几行算法会拖慢解码，以及框架替你省下了什么

写推理时算法（在线训练、记忆查表、残差改写、引导采样、稀疏注意力、自适应深度）时，作者最常遇到的困惑是：**算法本身只有几行 PyTorch 张量运算，放进自回归 decode 循环后却慢了一大截，甚至直接撑爆显存。**

这篇指南帮你建立一套清晰的判断直觉：在自回归解码阶段，性能到底卡在哪里、`pllm` 在底层自动替你做了哪些优化、哪些地方需要你做显式取舍，以及怎么用工具一眼看穿瓶颈。

---

## 一、 建立直觉：自回归 Decode 的三个物理瓶颈

在自回归 decode 阶段，模型每步只为每个请求生成 **1 个 token**。这决定了它的物理特性与预填充（prefill）或离线训练截然不同：矩阵乘（GEMM）的算术强度（每读一字节权重能做多少次乘加）约等于有效 batch 行数。在小到中等 batch 下，**GPU 算力几乎是过剩的，真正的瓶颈只有三个**：

1. **GPU 时间线有没有断掉（气泡）**：CPU 每步能不能把整串算子一口气喂给 GPU？只要有一次 CPU 等 GPU（host sync）或者退回逐算子发射（eager），GPU 就会大面积空转。
2. **每步多搬了多少字节（显存带宽与通信）**：权重、状态、词表 logits 和主机内存查表每步要读写多少字节？
3. **给 KV Cache 剩了多少空间（显存容量）**：模型权重、CUDA Graph 私有内存池和 `pllm` 的状态预留先分走固定份额，**剩下的显存才归 vLLM 的 KV cache**。多预留几百 MiB，并发上限就会断崖式下跌。

### 先记住一组“标尺数字”

以单卡 RTX 4090、Qwen2.5-0.5B BF16、128 条并发序列（$8 \times 16$）为例（[教程 11](../tutorials/11_lookup_memory.ipynb) 的设置），原生 vLLM 跑一步 decode 约 **$3.30\text{ ms}$**（吞吐约 $38,821\text{ tok/s}$）。在这个尺度下：

* **关掉 CUDA Graph（退回 eager 逐算子发射）**：哪怕不加任何自定义算法，原生 vLLM 吞吐也会直接掉到图模式的 **45%**（[教程 07](../tutorials/07_custom_residual.ipynb)、[教程 12](../tutorials/12_adaptive_depth.ipynb)）。
* **多读一次词表权重（HBM 带宽）**：Qwen2.5-0.5B 的 LM head 权重形状为 $896 \times 151936$（BF16），约 **$272\text{ MB}$**。128 行做一次投影实测约 **$0.38\text{ ms}$**（几乎全花在从显存读这块权重上），占整步耗时的十分之一以上。原生采样器每步扫词表也不便宜：temperature 缩放约 $120\ \mu\text{s}$、top-k/top-p 过滤约 $360\ \mu\text{s}$、Gumbel 采样约 $95\ \mu\text{s}$。
* **跨 PCIe 搬一次宽表（Host $\leftrightarrow$ GPU）**：PCIe 4.0 x16 有效带宽约 **$22.4\text{ GB/s}$**（仅为 4090 显存带宽 $\sim 1000\text{ GB/s}$ 的四十五分之一）。128 个请求每步查一行宽表（共 **$38.9\text{ MB}$**），光在 PCIe 上传输就要 **$1.7\text{ ms}$**——超过半步前向时间。
* **多卡 All-Reduce 被旁路任务撞上（卡间链路）**：在 4 张 A100 上跑 7B 模型，如果旁路训练每步和 vLLM 的 Custom All-Reduce 重叠 **$193\text{–}205\ \mu\text{s}$**，吞吐会从单卡的 93.79% 掉到 **87.07%**。

围绕这三个物理瓶颈，`pllm` 的全部底层优化可以归纳为**三条原则**：
1. **保住时间线**：零每步主机同步，把前向改写和旁路训练统统录进 CUDA Graph，把 CPU 簿记藏进 GPU 执行阴影里。
2. **省下显存容量**：用分页状态池、按角色自动选精度和真实代码追踪，把显存留给 KV cache。
3. **少搬字节、聪明重叠**：能合并读权重的绝不读两遍；必须搬运的，只在 GPU 对应资源空闲时塞进旁路 stream 重叠执行。

下面我们先讲这三条全局原则，再按**五种具体算法场景**带你算清每一步的账单。

---

## 二、 全局机制：`pllm` 在底层替你守住的三条原则

### 原则 1：消灭时间线气泡（零主机同步 + 全链路 CUDA Graph + 极薄 CPU 簿记）

#### 1) 稳态每步 0 次主机同步（硬契约 E1）
在 Python 代码里不小心写下一句 `.item()`、`if tensor.any():` 或 `.cpu()`，CPU 就会停下来等 GPU 排空流水线。`pllm` 将“稳态 decode 每步 0 次阻塞式主机同步”作为强制契约：
* **静态检查**：`pllm.explain` 在形状追踪阶段统计同步次数（输出 `host 0 syncs in traced workload math`）。
* **GPU 实跑验证**：[`pllm.testing.check_on_gpu`](validation.md) 在开启 `torch.cuda.set_sync_debug_mode("error")` 的情况下挂载所有执行入口，实跑至少 8 个稳态 decode 步，遇到任何隐式同步立即抛错。
* **实测表现**：在 S-4090 剖析中，128 次事件等待全部来自 vLLM 自身的 `async_utils.get_output`，**`pllm` 产生的 Event/Stream 主机等待为 0，decode 窗口内 `cudaDeviceSynchronize` 空闲时间为 0**。
* **设备端条件跳转（E4）**：遇到自适应深度分桶、计数 scan 完成判断、前向内精确 decode 分支等需要动态判断的场景，`pllm` 使用 CUDA Graph 内的设备端条件节点（`CUDA IF nodes`，需 CUDA 12.4+ 驱动路径）在 GPU 上直接跳转，绝不把条件拷回 CPU。

#### 2) 把自定义逻辑录进 CUDA Graph（E2、E3、E6、E8）
**CUDA Graph** 会把一串 GPU kernel 的启动参数与依赖预先录制（capture）成一张图，后续每步只用一次极轻量的回放（replay）发射整串计算，但要求图内张量地址固定。为了不让自定义逻辑打碎图模式，`pllm` 提供了三层捕获机制：

* **任务与在线训练整包捕获 `StepCapture`（E2）**：在线训练（如 `ESamp` 的 `loss`）包含收集参数、前向算 loss、掩码归约、`backward()`、写状态和 `paged_adamw` 等几十个小 kernel。`StepCapture` 按签名的最大行数和状态容量把它们录进同一张图；输入使用固定 tap 缓冲，未满 batch 的空行指向不会污染真实状态的哑槽位（sink state）。因此即便每步活跃请求集合在变，也**不需要重录图**。
  * *实测*：128 并发下开启 `StepCapture` 后，步时从 $6.73\text{ ms}$ 降到 $5.25\text{ ms}$，`commit_token` 从 $3.32\text{ ms}$ 降到 $1.56\text{ ms}$，吞吐从 $19,100$ 升至 **$23,829\text{ tok/s}$（+25%）**；765 步剖析中每步恰好发射 1 次训练图，**0 次重捕获**。
  * *边界*：要求按行独立的数学与逐行 loss；动态训练选择器会被拒绝并退回 eager。
* **模型层内残差改写捕进 vLLM 模型图（E3）**：`@pllm.custom_residual`（[教程 07](../tutorials/07_custom_residual.ipynb)）发生在每一层内部。`pllm` 将钩子降级为具名算子 `pllm::model_site`，用设备端索引读写私有状态页，使 vLLM 录制模型图时能把残差改写一并录进去（要求 TP=1 且关闭 prefix caching；捕获的是源算子而非 Inductor 融合算子）。
  * *实测（RTX 4090，$8 \times n=4 \times 256$，进图前 eager $\rightarrow$ 进图后，占原生图吞吐比例）*：
    * 空改写（identity）：$25.3\% \rightarrow \mathbf{65.8\%}$
    * Value Residual：$34.4\% \rightarrow \mathbf{85.2\%}$（教程 07 现为 **85%**）
    * Hyper-Connections（$n=4$）：$19.2\% \rightarrow \mathbf{66.6\%}$（教程 07 现为 **65%**）
    * 状态化记忆子层：未进图时 $11.6\%$ $\rightarrow$ 教程 07（1 GiB 捕获预算）进图后 **41%**（对比原生 eager 为 45%）。
* **自定义算子进图（E8）**：作者通过 `torch.library.custom_op` 注册并补齐 fake/autograd/vmap 规则的自定义 Triton 或 PyTorch 算子（[教程 09](../tutorials/09_custom_kernels.ipynb)）可直接捕进图与旁路通道。实测 Triton 算子达原生吞吐的 **95%**（$37,518\text{ tok/s}$），PyTorch 算子达 **96%**（$37,976\text{ tok/s}$，原生 $39,623\text{ tok/s}$）。
* **准入控制 `cuda_graphs="auto"`（E6）**：默认只要支持就进图，不支持时退回 eager 并在 `diagnostics().eager_reason` 和 `explain` 的 `forward` 行给出明确原因（如 `cuda_graphs=False`、MRv1、`ActivationPatch`、改用分段 wave 图的 `custom_forward`、共享可变 `InstanceState` 的 Adapter）；若设为 `cuda_graphs=True`，遇到任何 eager 原因会在启动时直接报错。

#### 3) 把主机 CPU 簿记压到 $700\ \mu\text{s}$ 以内，藏进 GPU 步时（G1、G2、C3、C4）
在异步调度下，CPU 准备第 $n+1$ 步元数据时，GPU 正在跑第 $n$ 步（约 $3.30\text{ ms}$）。只要 CPU 每步开销不超过 GPU 步时，就不会露出步间气泡：
* **单次锁页表上传（C3）**：每步所需的槽位 ID、掩码、行数、采样器地址与步幅全部由 NumPy 填入**同一张锁页（pinned）`int64` 表**，每个捕获任务只做一次 H2D 拷贝。在 S-4090（`ESamp` $\beta=0$）剖析中，`pllm` 每步 H2D 仅 $23,692\text{ B}$ / **$6.24\ \mu\text{s}$**（原生 vLLM 为 $1,088\text{ B}$ / $2.50\ \mu\text{s}$），D2H 保持 $1,542\text{ B}$/步不变。
* **削减 Python 簿记（G1）**：固定设备表视图、用位掩码（slot mask）替代原本每次完成栅栏都要做的 Python 槽位 ID 集合排序（仅此一项在 256 步里省下 **$43\text{ ms}$** CPU 时间）、推迟创建 trace 事件、跨流直接移交引用计数（pin）、用数组支撑 `StateRefs`。综合优化后（S-4090，`gpu_memory_utilization=0.5`），`ESamp` 在 vLLM 0.27 上 $\beta=0 / 0.5$ 从 $91.73\% / 76.50\%$ 升至 **$92.78\% / 83.74\%$**；在 vLLM 0.32 上从 $89.44\% / 69.23\%$ 升至 **$92.37\% / 81.61\%$**。
* **当前 CPU 耗时拆解（G2）**：S-4090 下 `pllm` 每步主机耗时总计约 **$700\ \mu\text{s}$**（中位数：`encode_mrv2_rows` $122\ \mu\text{s}$、`prepare_input` $174\ \mu\text{s}$、`HiddenTaps.prepare` $110\ \mu\text{s}$、`StateExecution.begin` $79\ \mu\text{s}$、`prepare_transition` $181\ \mu\text{s}$、`publish` $82\ \mu\text{s}$、`release` $58\ \mu\text{s}$，合入 `commit_token` 均值 $720\ \mu\text{s}$/步；外加 GC $58\text{–}66\ \mu\text{s}$/步）。这 $700\ \mu\text{s}$ 几乎完全被 $3.3\text{ ms}$ 的 GPU 前向掩盖，**`pllm` 造成的步间 GPU 空闲仅约 $4\text{–}6\ \mu\text{s}$/步**（把这部分簿记迁往 Rust 在计划中，为 GPU 步时更短的场景留出余量）。
* **有界异步导出 `@pllm.export`（C4）**：把设备张量导回主机做观测时，使用 `non_blocking=True` 拷入有界锁页缓冲并记录事件，后台异步回调消费；**默认在缓冲满时丢弃（`overflow="drop"`）而不阻塞 GPU/CPU**（丢包数记入 `export_dropped`；TP 侧计数器快照异步 D2H 仅 $\sim 2.2\ \mu\text{s}$/步，零主机等待）。

---

### 原则 2：守住显存容量（分页状态池 + 自动精度 + 真实代码追踪）

任何多预留的显存都会直接从 vLLM 的 KV cache 里扣。最惨痛的教训是早期曾无条件按调度器最大 token 数（8192）强行录制模型图（负面结果 **N15**）：在 RTX 4090 上跑 7B 模型（`gpu_memory_utilization=0.85`）时，vLLM 0.27 直接 OOM，0.32 的可用 KV cache 从原生的 76,144 个 token 暴跌到 **3,872 个 token**，吞吐只剩约 **21%**。修复为常规捕获尺寸（**A7**，2048-token 对照下 $\beta=0 / 0.5$ 恢复至 $\sim 97\% / 94\%$）并引入以下机制后，每一字节预留都受到严格控制：

#### 1) 分页状态池 PTTS（A1）与字段级写时复制（A2）
* **按页分配与字段级 COW**：**PTTS**（分页张量状态）用固定容量的物理页管理 `RequestState` 和 `InstanceState`，靠引用计数（pin）与完成栅栏（completion fence）回收。当状态版本更新时，**只为被修改的字段分配新物理页，未修改字段直接与旧版本共享物理槽位**（A2，可通过 `diagnostics().state_usage` 对比 `logical_live_bytes` 与实际分配的 `allocated_payload_bytes`）。
* **精确版本槽位公式（`roots`）**：为了支持前向读取旧版本的同时旁路写入新版本，每个状态资源需要的槽位数由精确公式决定（其中 `publish_lag` 为发布滞后步数，`lineages` 为并发请求槽位数）：
  * 普通请求状态：$\text{roots} = (3 + \text{publish\_lag}) \times \text{lineages} + 2$
  * 前向内精确更新（`visibility="exact"` decode）：$\text{roots} = 2 \times \text{lineages} + 2$
  * 实例状态：最多按单 lineage 公式取槽位，不可变实例状态仅需 `1` 个 root。
* **实测数字（`explain@5cdc9785`）**：单向量引导负载 `Steering`（1 个 896 维向量）收紧实例容量并配合自动精度后，预留从 `514 roots × 3,584 B = 1,842,188 B` 降到 **`6 roots × 1,792 B = 10,766 B`**；128 并发序列、rank 4 的 `ESamp` 请求状态预留为 `514 roots × 200,784 B = 103,203,036 B`（约 $98.4\text{ MiB}$）。
* **边界**：超出 `ExecutionOptions.state_memory_bytes` 在绑定时报错 `ValueError`；运行时若休眠路由分支占满槽位，抛 `RuntimeError("PageBundle admission capacity is exhausted")` 并累加 `admission_capacity_failures`。

#### 2) 按声明角色自动选择存储精度（A3）
默认 `dtype="auto"` 时，`pllm` 静态分析每个字段的读写角色（规约、梯度累加与优化器内部计算始终保持 $\ge$ FP32）：
* **必须用 FP32 的角色**：被优化器更新的参数，以及在方法里“读出再写回”的累加器/跨块进位（carry）。在 32k tokens、chunk 64 的长序列递推测试中，若 carry 用 BF16，相对误差会累积到 **$1.437 \times 10^{-1}$**；用 FP32 则只有 **$3.692 \times 10^{-4}$**（低三个数量级）。
* **默认用模型精度（如 BF16）的角色**：只读或只写字段。在 RTX 4090（$8 \times 16 \times 256$）上，自动精度让 ReFT ADD 每请求状态从 $3.50\text{ KiB}$ 降到 **$1.75\text{ KiB}$**，FactorAdapter 从 $22.00\text{ KiB}$ 降到 **$11.00\text{ KiB}$**（常驻显存分别省 $1.82\text{ MiB}$ 和 $4.59\text{ MiB}$）。反之，若把 $9.00\text{ MiB}$/请求的稠密状态强行设为 FP32（$17.32\text{ MiB}$/请求），不仅显存翻倍，吞吐还会从 $3,156\text{ tok/s}$ 掉到 **$2,724\text{ tok/s}$（86.31%）**。
* **边界**：可在 `RequestState`/`InstanceState` 或 `pllm.Field(..., dtype=...)` 显式覆盖；若在冻结字段上直接做仅支持 FP32 的矩阵乘，需显式声明 `dtype=torch.float32` 或在计算中转换。

#### 3) 真实任务代码追踪（A4）、预算自动降级（A5）与显式捕获预算（A6）
* **CPU 追踪替代手写估算（A4）**：`pllm` 在 CPU 上用 meta/fake 张量对真实任务体（前向、反向、优化器、scan）做存储生命周期与别名追踪（含 autograd 保存的张量、512 B 对齐和 1 MiB 分配器尾余量，不初始化 CUDA），由 Rust 规划器按精确区间算出峰值，并优先选择 `[峰值显存最小, 任务数最少]` 的实现。替换手写公式后，`ESamp` 的 `predict` 工作区从 $177.6\text{ MB}$ 降到 **$27.8\text{ MB}$**；`OverRIDE` 训练工作区从 $1,056\text{ MB}$ 降到 **$267\text{ MB}$**（实测预热后 CUDA 峰值 $264,242,176\text{ B} \le$ 推导上限 $267,269,120\text{ B}$）。
  * *边界*：不覆盖无界自定义 autograd、第三方库临时缓冲、首次 cuBLAS 初始化（约 $8.5\text{ MB}$）及 vLLM 自身的模型/KV 显存。
* **预算不足时自动退回内联（A5）**：旁路重叠（`side` lane）需要同时保留两步缓冲。若超出 `ExecutionOptions.memory_budget_bytes`，Rust 规划器会自动将任务退回主通道（`inline`）执行；再低于内联下限才报错并指明需调整的预算。例如 `ESamp`（$\beta=0.5$，128 序列）旁路计划峰值为 $1,091,375,392\text{ B}$（约 $1041\text{ MiB}$），低于此值退回内联，低于 $1,020,590,368\text{ B}$（约 $973\text{ MiB}$）规划失败。
* **私有图捕获池预算（A6）**：每个 `StepCapture` 任务拥有受 `resident_bytes` 限制的私有图内存池；模型图默认池为 $64\text{ MiB}$（LAuReL 实占 $46\text{ MiB}$、value residual $22\text{ MiB}$、HC $26\text{ MiB}$）；`custom_forward` 按实际推导（32 并发 $748\text{ MiB}$，128 并发 $1,392\text{ MiB}$）。可通过 `capture_memory_bytes` 调整，**预算不足直接报错，绝不静默退回 eager**（注：`explain` 是否估算 `custom_forward` 捕获池在文档与代码记录间尚待核实）。

#### 4) 其他容量与生命周期保护（速览）
* **引擎关闭确定性释放（A10）**：`with make_llm(...)` 退出时按序清理 vLLM、cuBLAS 工作区、CUDA Graph 条件池引用与编译缓存并调用 `empty_cache`。修复前连续建两个 4B 捕获引擎会残留约 $14.4\text{ GiB}$；修复后两次关闭均回到 **$21.42\text{ GiB}$** 空闲（起始 $21.53\text{ GiB}$；注：带捕获的精确前向更新关闭时仍未归还全部显存）。
* **共享 hidden tap 缓冲（A12）与跨消费者复用（A13）**：同一 `(layer, phase)` 的激活只建一份 tap 缓冲供所有方法共享（如 `loss` 与 `predict` 共享 `460,800 B`）；sampler adapter 消费的 token 辅助计算只算一次（A13，128 行 decode 每步省下 $303,872\text{ B/行} = 38,895,616\text{ B}$）。
* **有界诊断与观测内存（A14）**：`numeric_health` 在设备端分块（FP64）归约快照，仅查询时拷回标量（594 MiB 表的健康检查额外显存从 $5.51\text{ GiB}$ 降到 **$19.0\text{ MiB}$**；64 桶下主机校验 RSS 从 $9,181,812\text{ KiB}$ 降到 **$1,935,532\text{ KiB}$**）；trace 队列上限 256 事件 $\times 32\text{ KiB}$。
* **KV CPU 卸载（A15）、感知负载的 Prefix Cache（A16）、同卡零拷贝投喂（A17）**：`ModelConfig(kv_offload=...)` 支持经 vLLM 连接器卸载 prompt KV（拒绝 LMCache）；`enable_prefix_caching=True` 在块哈希中混入负载命名空间，仅在数学一致时复用（与 `custom_residual`/`custom_attention` 互斥）；`InstanceState(feed=...)` + `llm.feed()` 通过 CUDA IPC 页授权实现同 GPU 只读状态零拷贝借用（`data_ptr` 相同，无第二份池）。

---

### 原则 3：重叠的真相——旁路 Stream 什么时候真能提速？

把任务放到旁路 CUDA stream（`lane: side`）并不等于多了一张显卡。同一张 GPU 的 SM（流多处理器）和显存带宽是固定的：**只有当主 stream 没有把某项硬件资源吃满时，旁路任务塞进来才叫“免费重叠”；如果主 stream 已经把显存带宽或 SM 吃满了，硬把旁路任务挤进来只是在抢同一块蛋糕（零和博弈）。**

两项实测把这条边界划得一清二楚：

* **什么时候真能提速（跨步滞后发布 `publish_lag=1`，F1）**：
  如果要求第 $n$ 步采完样立刻算完梯度、第 $n+1$ 步前向必须读到最新权重（`publish_lag=0` 严格串行），每步前向开头都得干等反向传播和 AdamW 跑完。`pllm` 默认设置 `publish_lag=1`：第 $n$ 步 token 确定后，经极短拷贝栅栏就把训练放到旁路 stream 上跑；第 $n+1$ 步前向直接读第 $n-1$ 步的旧版本，新版本在第 $n+2$ 步生效。这样旁路训练就能填进主干流在步间、小算子和非满载区段的空隙里。
  * *实测*：在 128 并发 RTX 4090 测试中，将 `publish_lag` 从 0 改为 1 后，`ESamp` 在 $\beta=0$ 下吞吐从原生的 $91.4\%$ 升至 **$94.3\%$**，$\beta=0.1$ 下从 $73.7\%$ 升至 **$84.1\%$**（1,245 个旁路 kernel 中有 780 个与主干重叠；且 $\beta=0$ 下 `lag=0` 与 `lag=1` 输出位级完全一致）。对 `custom_forward` 或 `visibility="exact"` 任务自动禁用。
* **什么时候是零和博弈（负面结果 N6）**：
  在 `ESamp`（S-4090，$\beta=0$）剖析中，旁路训练还有 $70.33\ \mu\text{s}$ 露在主干外面。当我们尝试去掉两个采样器等待事件、强行提高重叠度时，露在外面的旁路时间确实降到了 $8.57\ \mu\text{s}$/步，但**端到端吞吐反而从 94.49% 掉到了 92.73%**（把训练输入拷贝提前到采样前的试验也同样从 94.3% 降到 92.9%）。原因很简单：训练并没有消失，它只是从步间移到了**正在满负荷扫词表的采样阶段**，跟采样器争抢显存带宽，把采样拖慢了。这也揭示了 `esamp94` 剖析的核心结论：**$\beta=0$ 时比原生 vLLM 慢的那 $\sim 7.5\%$，主要是 GPU 上实打实的训练图回放（约 $100\text{–}140\ \mu\text{s}$/步）与主干流的资源争抢，而非主机气泡。**

---

## 三、 按算法对号入座：五大典型场景的账单与优化

知道全局原则后，你可以直接找到自己正在写的算法类型，看清它的开销账单和 `pllm` 的专项优化。

### 场景 1：在线训练与引导采样（以 `ESamp`、`DoLa`、量化训练为例）

#### 1) 算清引导采样的 $401\ \mu\text{s}$ 账单（B2、B3）
在 [教程 03](../tutorials/03_decode_training.ipynb) 的 `ESamp`（S-4090 设置）中，只做旁路在线训练（$\beta=0$）能跑到原生 vLLM 的 **92.48%**（20 轮中位数为 **92.45%**，$36,113\text{ tok/s}$；教程 03 的 `MiniESamp` 为 **95%**）；而开启引导采样（`@pllm.guide_sampling`，$\beta=0.5$）后为 **83.99%**（20 轮中位数 **83.38%**，$32,614\text{ tok/s}$；`MiniESamp` 为 **84%**），每步比 $\beta=0$ 多花 **$401\ \mu\text{s}$**。这 $401\ \mu\text{s}$ 花在了哪？

1. **LM head 矩阵乘从 128 行翻倍到 256 行：$+118\ \mu\text{s}$（机制 B2：共用 LM-head GEMM）**
   引导采样需要为每个请求的引导分支也算一份 151,936 词表的 logits（`explain` 报告 `projection FLOP: 34,850,471,936`）。为什么工作量翻倍只多了 $118\ \mu\text{s}$，远小于单独再跑一次 128 行的 LM head（约 $380\ \mu\text{s}$）？因为 `pllm` 把过滤后的引导隐状态直接拼在 vLLM 原有的 `compute_logits` 调用末尾，把矩阵乘从 $[128, 896] \times [896, 151936]$ 扩成 $[256, 896] \times [896, 151936]$，**每步省下了再读一遍 $272.27\text{ MB}$ LM head 权重的显存带宽**（$\beta=0$ 或无引导行时不追加；cuBLAS tile 变化带来的舍入差异在 Qwen 0.5B 上 $\le 0.0625$，不超过 vLLM 自身随 batch 变化的幅度）。
   * *对比反例*：读取中间层词表分布的 `layer_logits`（如 [教程 06](../tutorials/06_reading_inside.ipynb) 的 `DoLa`，32 请求 $\times n=4$）默认使用独立 GEMM，每步必须多读一次或多次 LM head 权重，在 0.5B 上吞吐仅为原生的 **25%**。
2. **词表宽度重加权：$+187\ \mu\text{s}$（机制 B3：融合重加权）**
   按 $(1+\beta)\,\text{logits} - \beta\,\text{pred}$ 混合主干与引导 logits 时，`pllm` 用单个 `torch.compile` 内核完成，避免在 $[128, 151936]$ 大张量上做 7 次逐元素往返读写。单次逻辑流量固定为 $194,478,080\text{ B}$（约 $195\text{ MB}$），$187\ \mu\text{s}$ 已贴近 4090 的显存带宽物理极限（注：尝试把重加权与更外层采样算术进一步融合会产生 $1.9 \times 10^{-6}$ 的偏差，未通过精度门槛被拒，见 **N14**）。
3. **两次额外小 CUDA Graph 回放：约 $+56\ \mu\text{s}$**；**过滤（filter）：$+8\ \mu\text{s}$**。

> **模型规模效应**：在 0.5B 小模型上，多一次词表投影和加权占到步时的 $\sim 10\%$；但在 7B 模型上，主干每步要读十几 GB 权重，同样的词表开销占比极小（RTX 4090 上的 7B 模型在 2048-token 捕获下 $\beta=0.5$ 可达原生的 **94%**）。

#### 2) 压缩旁路训练的显存读写（B1、B5、B9）
* **捕获训练内的分页 AdamW（B1）**：`paged_adamw` 直接从源物理槽位读取参数和一阶/二阶矩并写回目标槽位，省去对 5 个优化器字段做写时复制（COW）的整页搬运。在 S-4090（$\beta=0$）每步 $177.6\ \mu\text{s}$ 的旁路 GPU 时间里，AdamW 仅占 **$29.3\ \mu\text{s}$**。
* **削减冗余梯度与拷贝流量（B9）**：`FieldView` 跳过全宽梯度分配、增大页拷贝 tile、单次参数切分避免重复 padding、目标已知时省略掩码保留 kernel、仅消费 token 的方法跳过梯度快照。在 A100 微基准上：模块前向/损失/梯度从 $0.544 \rightarrow \mathbf{0.207\text{ ms}}$，字段查表 $187.45 \rightarrow \mathbf{134.25\ \mu\text{s}}$，掩码 gather $109.83 \rightarrow \mathbf{50.97\ \mu\text{s}}$，页拷贝 $24.67 \rightarrow \mathbf{4.79\ \mu\text{s}}$。
* **行级掩码裁剪（B5）**：`add_logits(rows=...)` 只在受影响的采样行上加偏置；非残差消费 `input_tokens` 的方法自动只看采样行。

#### 3) 量化底座上的在线训练：瞬态反量化（A11 vs N9）
在 GPTQ/Marlin 等量化模型上做在线训练或 overlay 读取时，若预先把反量化后的稠密权重缓存下来（负面结果 **N9**），虽然在 A100 上跑 14B GPTQ 能把步时从 $1.122\text{ s}$ 降到 $0.176\text{ s}$，但会多占 **$24.63\text{ GiB}$** 常驻显存。`pllm` 改用 Marlin A16 Triton 逆变换做**瞬态反量化（A11）**，稠密权重绝不活过单次反向传播或读取：14B GPTQ TLM（A100）步时从 $1.117\text{ s}$ 降到 **$0.193\text{ s}$**（4090 上 14B qkv 反量化从 $2.535\text{ ms}$ 降到 **$0.104\text{ ms}$**），**峰值显存完全不变**，且 loss/logits/梯度位级一致（注：目前为脚本验证，尚无合入仓库的专项回归测试）。

---

### 场景 2：外部大表与词表级查表（以 `Engram` 为例）

像 [教程 11](../tutorials/11_lookup_memory.ipynb) 中 $2048 \times 151936$ 的 BF16 查找表，光表本身就占 **$594\text{ MiB}$**，每查一行要读 $303,872\text{ B}$，128 个并发请求每步共需读取 $38,895,616\text{ B}$（约 **$38.9\text{ MB}$**）。

#### 1) 第一步：Inductor 整图融合省掉显存往返（B4）
针对只依赖 `input_tokens` 的不可微查表任务（如 gather 出宽向量后紧接 `cast`、`where` 掩码和缩放），`pllm` 在绑定引擎时用 Inductor `fullgraph` 将其编译并预热为**单次表读取 kernel**，省去把宽向量写回显存再读出来做逐元素运算的往返流量。
* *实测*：配合提前到 `request.prepare` 发射（E7），16-token 剖析中 D2D 拷贝调用从 36 次减半到 18 次，CUDA 耗时从 $84.671\text{ ms}$ 降到 $71.754\text{ ms}$；设备端表吞吐从 $26,133$ 升至 **$32,881\text{ tok/s}$（原生 vLLM 的 66% $\rightarrow$ 84%）**，主机端表（在下面 C2 优化前）从 $12,557$ 升至 **$23,044\text{ tok/s}$（32% $\rightarrow$ 59%）**。若需严格位级一致，可通过 `allow_numerical_transforms=False` 关闭。

#### 2) 第二步：显存装不下时放主机内存 `placement="host"`（A9）
若显存紧张，作者可显式声明 `InstanceState(pretrained=..., placement="host")`（因涉及容量与 PCIe 带宽的权衡，`pllm` 不做后台自动迁移；仅限只读预训练表的按行视图读取，不可直接在模型站点内读取）。
* *容量收益*：教程 11 中把 $594\text{ MiB}$ 的表从 `"device"` 改为 `"host"` 后，`pllm` 负责的设备端显存峰值从 $736,523,358\text{ B}$（约 $702\text{ MiB}$）直降到 **$114,193,488\text{ B}$（约 $109\text{ MiB}$）**；测试中可在仅 $1\text{ GiB}$ 的设备端预算下运行 $3\text{ GiB}$ 的主机表。

#### 3) 第三步：如何把 $1.7\text{ ms}$ 的 PCIe 传输藏进 $3.3\text{ ms}$ 的模型前向里？（C1、C2、E7 vs N8）
表放到主机锁页内存后，128 行每步 $38.9\text{ MB}$ 必须跨过 PCIe 4.0 x16（$\sim 22.4\text{ GB/s}$），光物理传输就要 **$1.7\text{ ms}$**。想把它和模型前向重叠，两条直觉路都走不通（负面结果 **N8**）：
1. **不能用独立 DMA 拷贝引擎（Copy Engine）**：要查哪一行（token ID）是在 GPU 上算出来的，`cudaMemcpy2DAsync` 或块交换 API 都需要 CPU 知道源地址，且图捕获下 memcpy 参数固定；拷回索引会违反零主机同步原则（E1）。因此必须由 GPU SM 通过 **UVA**（统一虚拟寻址）按设备索引直接去主机内存拉取（C1）。
2. **不能直接把普通 UVA gather 提前到旁路 stream 发射**：普通 gather kernel 按数据量开网格，会启动 **37,984 个线程块（CTA）**，把 RTX 4090 的 128 个 SM 占到 **100% 满载**。虽然这些线程块绝大部分时间只是在傻等慢速的 PCIe，但它们占死了 SM 槽位，导致主 stream 上的模型前向排不上队。实测单独提前发射反而让步时从 $5.697\text{ ms}$ 恶化到 **$5.939\text{ ms}$**，$1,753.7\ \mu\text{s}$ 的读表只有 **$26.2\ \mu\text{s}$** 与主干重叠。

**`pllm` 的解法（机制 C2 + E7）**：
跑满 $22.4\text{ GB/s}$ 的 PCIe 根本不需要多少 SM。`pllm` 通过 `HostGather` 拦截对主机表的索引操作，改调自定义常驻（persistent）Triton 算子 `pllm::host_rows`，**把线程块数量硬限制在 GPU SM 总数的四分之一**（128 个 SM 的 4090 上只开 **32 个 CTA $\times 128$ 线程**），并由 Rust 规划器自动把只依赖 `input_tokens` 和不可变状态的查表任务提前到整步最开头的 `request.prepare` 旁路 stream 发射（E7）。
这样一来，32 个线程块在后台等 PCIe，**剩下四分之三的 SM 完全空出来跑主干模型前向**，剖析中读表与主干前向的重叠率达到 **99.14%**：
* **实测对比（教程 11 在 `main` `ea7ff796` 上重跑，RTX 4090，vLLM 0.27.1）**：
  * 原生 vLLM：**$38,821\text{ tok/s}$**（约 $3.30\text{ ms}$/步）
  * 表在 `device`：**$31,750\text{ tok/s}$**（原生的 **82%**，约 $4.03\text{ ms}$/步）
  * 表在 `host`：**$30,174\text{ tok/s}$**（原生的 **78%**，约 $4.24\text{ ms}$/步）
  可以看到，尽管每步跨 PCIe 实打实搬运了 $38.9\text{ MB}$（物理耗时 $1.7\text{ ms}$），**表放 `host` 比放 `device` 每步只慢了约 $0.21\text{ ms}$**（5 轮中位数剖析中，限制 1/4 SM 让步时从 $5.697\text{ ms}$ 降到 $4.194\text{ ms}$，吞吐从 $22,469$ 升至 $30,522\text{ tok/s}$，提升 **35.8%**）。
* **怎么自己估算能不能藏住**：看 `pllm.explain` 输出的 `logical transfer bytes/row`（或 `decode bytes/step`），用 $\text{PCIe 耗时} \approx \frac{\text{活跃行数} \times \text{bytes/row}}{22\text{ GB/s}}$ 与单步前向时间对比。只要明显短于一次前向且任务只依赖 `input_tokens`/不可变状态，绝大部分传输都能被藏住。

---

### 场景 3：自定义注意力与 KV Cache 瘦身（`@pllm.custom_attention`）

在写自定义注意力（[教程 10](../tutorials/10_custom_attention.ipynb)）时，首先要分清你的机制是在**真省 KV 显存容量**，还是**只省读取带宽**：

| 方式 | 机制与写法 | 省显存容量吗？ | 省读取带宽吗？ | 实测吞吐与容量效果 | 适用边界 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **静态几何布局**<br>（A8 + B8） | `keep="window"`（滑动窗口）、`sink=True`（保留 prompt + 窗口）、`keep="none", source=k`（复用低层第 $k$ 层 KV） | **真省容量**：直接映射为 vLLM 原生 `SlidingWindowSpec`、`SinkSpec`（R-SWA）和跨层共享组，物理块真回收 | **真省带宽**：直接调用 vLLM 原生 FA2 与上游 Triton `unified_attention`（位级一致） | **容量**：教程 10（`max_model_len=2048`）单请求 KV 从 $24.0\text{ MiB}$ 降到 **$16.6\text{ MiB}$**（偶数层窗口 256）和 **$12.0\text{ MiB}$**（共享一半层，最大并发达 717.6）；固定长度（8+112）下奇数层共享省 **50.00%**、2/24 层窗口省 **7.29%**、sink 省 **6.25%**。<br>**吞吐**：教程 10 中 `NativeRead` **97%**、`LocalGlobal`（sink）**97%**、`CrossLayerShared` **95%**（对比旧路径手算 FP32 sink 仅 $8,867\text{ tok/s}$ = 22.6%，换原生 R-SWA 后升至 $36,458\text{ tok/s}$，拷贝从 $561.4 \rightarrow 2.0\text{ ms}$/轮，见 N16）。 | 仅限 MRv2、TP=1、FA2、非量化 KV、关闭 prefix caching；不可与 `PositionState` 或 `custom_forward` 同用。极小窗口如 `(3,0)` 下 FA2 耗时 $91.47\ \mu\text{s}$ 反慢于全量 $16.56\ \mu\text{s}$，窗口吞吐为全量的 89.3%/90.7%（N7）。 |
| **稀疏 Top-K 读取**<br>（B7） | `PagedKV.attend(q, topk=...)` 按块打分挑 Top-K 块读取 | **不省容量**（所有历史 KV 仍在池中） | **省逻辑读取字节**：`explain` 显示 `QuestRead <= 458,752` B/token（全量 `<= 1,048,576` B/token） | **当前限制**：因打分与 gather 走 PyTorch 算子，教程 10 中 `QuestRead` 吞吐仅 $2,727\text{ tok/s}$，即原生的 **7%**（早期另一组 `cattn` 基准为 75%）。 | vLLM 原生高效 DSA Top-K 路径硬性要求 **SM90+**（Hopper），在 RTX 4090（SM89）和 A100（SM80）上不可用，调用 `pllm.attention(layer, "topk")` 会报错（N3）。 |

* **为什么不支持按分数动态淘汰物理 KV 块（如 SnapKV / H2O，负面结果 N1）**：vLLM 对每个请求只维护单一的 `num_computed_tokens` 标量，位置编码与物理槽位映射均由它线性推导，且块表（block table）只能追加、不能在中间挖洞，因此无法在不破坏块管理器的前提下淘汰中间 token 回收物理块。
* **为什么放弃了 LastKV 式的压缩远端记忆（M7，负面结果 N2）**：vLLM 的 `HiddenStateCacheSpec` 在 $8,192\text{ B}$ 的物理页里只装 $2,048\text{ B}$ 有效内容（零显存节省），深层可见条目数会随 prefill 分块大小（64/512/1024）在 960/896/768 之间波动，且单请求下 `near+far+merge` 每层耗时 **$30.31\ \mu\text{s}$**，远慢于全量 FA2 的 **$11.88\ \mu\text{s}$**。

---

### 场景 4：前向内快权重更新（In-Place TTT）与自适应深度（`@pllm.custom_forward`）

#### 1) 线性递推的分块并行前缀扫描 `pllm.scan`（F2）
在 chunk 内做形如 $S_t = A_t S_{t-1} + B_t$ 的线性递推时（如线性注意力或快权重更新），串行循环既慢又填不满 SM。Rust 规划器会自动识别仿射递推结构，在工作区预算允许时改用有序 / Hillis-Steele / 树形并行前缀扫描（支持 `tuning="measured"` 在绑定时实测挑选最快实现）：
* *实测（早期分支微基准）*：单头 $64 \times 64$ 状态扫描耗时从串行的 $1.4387\text{ ms}$ 降到 **$0.2190\text{ ms}$（6.57 倍加速）**，14 头 $64 \times 64$ 为 **1.85 倍**，完整状态转移从 $5.8786\text{ ms}$ 降到 **$1.7585\text{ ms}$（3.34 倍）**。当状态维度很大（如 $d=896$）导致树形扫描不划算时，形状守卫会自动保留串行循环。
* *边界*：系数需为 FP32；CPU、部分扫描或需微分的调用保持串行；可通过 `allow_numerical_transforms=False` 禁用并行重排。

#### 2) 前向内精确更新捕获 `train_in_forward(visibility="exact")`（E5 vs N12）
在默认数值容差模式（`allow_numerical_transforms`）下，`pllm` 会将 `visibility="exact"` 的 decode 阶段捕获进图：
* *实测*：RTX 4090 上跑 In-Place TTT 4B 社区权重（4096 prompt + 256 生成），捕获后 bs1 从 $57.59 \rightarrow \mathbf{59.51\text{ tok/s}}$（+3.3%），bs4 从 $107.56 \rightarrow \mathbf{113.16\text{ tok/s}}$（+5.2%）。在 [教程 08](../tutorials/08_inplace_ttt.ipynb) 中，bs1 在 4k / 16k 上达到 **$61.61 / 42.91\text{ tok/s}$**（优于官方 HF 实现的 $54.8 / 34.9\text{ tok/s}$），且支持官方无法运行的 bs4（**$123.76\text{ tok/s}$**；快权重状态占 $714.0\text{ MiB} \times 4\text{ roots}$）。
* *边界（N12）*：prefill 仍为 eager；若开启严格位级一致模式（strict bitwise），捕获反而比 eager 更慢（bs1/bs4 仅 $48.84 / 78.66\text{ tok/s}$）；关闭引擎时暂不归还全部显存。

#### 3) 自适应深度跳层 `layers(..., active=mask)`：为什么只有“整批全跳”才省读权重？（B6）
在 `@pllm.custom_forward`（[教程 12](../tutorials/12_adaptive_depth.ipynb)）中让部分 token 跳过中间层时，`pllm` 会用设备端条件节点把活跃行稳定压实（compact）到 2 的幂次桶里执行，被跳过的行只在首次跳过时用原生算子补齐 norm/QKV/RoPE 以维持 KV 连续（约占单层开销的 7%）。
* **核心成本模型**：decode 阶段的矩阵乘是**权重带宽受限**的（例如 7B 模型单层权重达 $466\text{ MB}$）。哪怕 128 个 token 里只有 1 个活跃，GPU 也必须把整层几百 MB 的权重读进 SM 才能算这 1 行。因此，**部分跳过只能按不活跃行数比例省下注意力 KV 读取流量；只有当整批 token 在该层全部跳过（落入 `bucket 0`）时，才真正省下读该层权重的带宽**。
* *实测（教程 12，S-tut $n=4$，第 16–19 层可跳）*：从不跳过为原生图吞吐的 **93%**；阈值 0.85（约 14% token 跳过）为 **90%**；全部跳过（落入 bucket 0）达 **102%**（对比原生 eager 仅 **45%**）。仅对 decode token 生效，不支持 expert-choice 路由，且不释放被跳层的 KV 显存。

---

### 场景 5：多卡张量并行（TP > 1，机制 D1）

在多卡张量并行（Tensor Parallelism，仅支持 vLLM 0.32 MRv2）下，vLLM 每算完一层注意力或 MLP 都要做一次全卡 Custom All-Reduce。如果旁路流上的训练 kernel 刚好在同一时刻抢占显存带宽和 SM，就会拖慢全卡同步：
* **机制 D1（小状态按 rank 复制 + 零 `pllm` 集合通信 + 流优先级错峰）**：
  1. **`pllm` 绝不自己发起集合通信**：hidden tap、路由计算、`PositionState` 和请求级小状态在每个 TP rank 上各存一份完全相同的副本，独立执行相同的确定性逻辑（各 TP 规模下 `pllm` 与原生 vLLM 的 logits 均方根误差为 0），`@pllm.export` 仅在 rank 0 触发。
  2. **流优先级与发布时机对齐**：将 TP 推理主 stream 设为高优先级（`-1`）、旁路 stream 设为 `0`，并让各 rank 的自动状态更新在采样前写入私有状态、在 token 提交时发布，消除旁路训练与 vLLM Custom All-Reduce 的重叠争抢。
* **实测（A100-SXM4-80GB，Qwen2.5-7B，`ESamp` $\beta=0$，rank 4，$8 \times 16 \times 256$）**：消除训练与 All-Reduce 重叠后（重叠时间从 $205/193\ \mu\text{s}$ 降为 **$0\ \mu\text{s}$/步**）：
  * TP1：原生吞吐的 $93.79\% \rightarrow \mathbf{95.84\%}$
  * TP2：$89.09\% \rightarrow \mathbf{95.63\%}$
  * TP4：$87.07\% \rightarrow \mathbf{97.60\%}$（仅设置流优先级一项在 TP4 上即带来 $14,936 \rightarrow 15,031\text{ tok/s}$）。
* **适用边界**：因为状态按 rank 复制而非分片，**TP 不会减少单卡上 `pllm` 状态或预留缓冲的显存占用**（`explain` 会提示 `parallel ... reservations are per rank`）。以下涉及跨分片语义或每层改写的能力在 `tensor_parallel_size > 1` 时会被直接拒绝：`custom_residual`、`custom_attention`、LoRA 式权重改写、全模型训练、状态投喂（`feed`）以及 `check_on_gpu`（后者需在 TP=1 下运行）。

---

## 四、 实操工作流：写代码时怎么看成本、设预算、查瓶颈（H1–H5、G3）

`pllm` 把上述所有机制都暴露在静态解释（[《写代码时就看到成本》](explain.md)）、显式预算、契约校验（[《验证与发布检查》](validation.md)）和无同步运行时诊断中。多个效果组合时，框架会按挂载点自动排序，只对真正的读写冲突报错并点名冲突双方（H5）。

推荐按以下三步排查性能与显存问题：

### 第一步：不占 GPU，先跑一行 `pllm.explain(workload, config)`（G3、H1）
`pllm.explain` 只读模型的 `config.json` 并在 CPU 上用 meta 张量推导，**不初始化 CUDA、不调 `mem_get_info`、不打开权重文件**（以 `MiniESamp` 为例，进程内首次调用耗时 **$2.9\text{ s}$**，重复调用 **$< 0.1\text{ s}$**）。重点看四样东西：
1. **有没有退回 eager / 有没有主机同步**：看 `forward` 行（是否进图或给出 eager 原因）、`member` 行是否有 `captured`，以及 `host 0 syncs in traced workload math`。
2. **显存峰值与降级阈值**：看 `state` 行的 `roots × bytes/root = reserved` 和字段 `dtype`；看 `memory peak ... funded by pllm` 以及紧接着的 `below X tasks run inline; below Y planning fails`。
3. **每步多搬了多少字节 / 多少 FLOP**：看 `projection FLOP`、`sampler N bytes`、`optimizer ... bytes`、每层 `KV bytes/token`，以及主机表的 `logical transfer bytes/row` 和 `decode bytes/step`。
4. **挂在哪个锚点、走哪条通道**：看 `member` 行是 `request.prepare` 还是 `token.commit`，是 `main` 还是 `side`，以及 `timing publish_lag=1 ... readable at n + 2`。

### 第二步：用预算参数提前拦截，用 `check` / `check_on_gpu` 验契约（H2、H3）
* **设预算快速失败（H2）**：通过 `ExecutionOptions` 的 `state_memory_bytes`、`memory_budget_bytes`、`capture_memory_bytes`、`max_training_rows` 设好上限，超预算时在分配显存前直接报错并写入 `report.budget_errors`。
* **跑契约测试（H3）**：在 CPU 上跑 `pllm.testing.check` 验证编译、`publish_lag`、可见性、抢占、行隔离与组件独立性；在 GPU 上跑 `check_on_gpu` 验证推导峰值与真实加载探针一致（`explain/probe`）、8 步稳态 decode 零同步、抢占重放、中性位级一致与序列置换不变性。

### 第三步：运行时看无同步诊断 `llm.diagnostics()` 与 `trace_snapshot()`（H4）

遇到具体症状时，直接查下表对应的字段：

| 你遇到的症状或问题 | 先看 `pllm.explain`（静态） | 可调的预算 / 配置开关 | 运行时查 `diagnostics()` / `trace_snapshot()` |
| :--- | :--- | :--- | :--- |
| **启动时报显存不足，或 KV 并发太低** | `state` 行（`roots`、`bytes/root`、`dtype`）；`memory peak`；`capture reservation`；每层 KV 准入容量 | 大表改 `placement="host"`；注意力改 `keep="window"` / `source=k`；调 `state_memory_bytes`、`memory_budget_bytes`、`capture_memory_bytes`、`max_training_rows` | `state_usage`（逻辑 vs 物理分配）、`admission_capacity_failures`、`kv_cache_blocks`、`kv_cache_max_concurrency` |
| **步时明显变慢，怀疑退回了 eager 或有 CPU 同步** | `forward` 行、`member` 行是否 `captured`、`host 0 syncs`、`timing` 条件节点说明 | 设 `cuda_graphs=True` 强制拒绝任何会导致 eager 的特性；跑 `check_on_gpu` 抓隐式同步 | `diagnostics().eager_reason` |
| **开启引导采样或额外计算后吞吐下降** | `projection FLOP`、`sampler N bytes`、`optimizer ... bytes`、每层 `KV bytes/token`、自定义 `op` 字节下界 | 设 `max_training_rows` 限制单步训练行数；用 `rows=` 掩码缩小作用范围；避免不必要的 `layer_logits` | `trace_snapshot()` 中的 `device_task_times`、`kernel_calls` |
| **大表放 `host` 后想确认 PCIe 传输有没有藏住** | `placement: host; host bytes`、`logical transfer bytes/row`、`decode bytes/step`；确认任务锚点为 `request.prepare / side` | 确保查表任务只依赖 `input_tokens` 和不可变状态（不依赖中间层 hidden） | 对比 `device_task_times` 与步时 |
| **旁路任务（`side`）没有带来预期的重叠加速** | 看 `member` 的 `anchor / lane` 与每步 FLOP/字节数（判断是等待型还是带宽/算力争抢型） | 调低 `memory_budget_bytes` 可对比 `side` 与自动退回 `inline` 的行为 | `trace_snapshot()` 中主干流与旁路流的时间分布、`host_task_times` |
| **怀疑长序列或训练出现数值溢出/NaN** | 查看 `state` 行各字段解析出的 `dtype`（累加器/carry 应为 `float32`） | 在字段上显式指定 `dtype=torch.float32` | `diagnostics().numeric_health`（设备端分块归约，无每步同步） |

---

## 五、 避坑清单：那些试过但走不通的路（负面结果汇总 N1–N16）

许多看似顺理成章的优化思路，在自回归 decode 和 vLLM 架构下其实没有收益甚至会起反作用。动手改代码前可以先对照这份实测避坑清单：

1. **关于 `torch.compile` 与算子融合的坑**
   * **直接 `torch.compile` 作者手写的 loss 前反向（N4）**：在 S-4090 的 `ESamp`（$\beta=0$）7 轮测试中吞吐为 **92.77%**，相比不编译没有任何可测收益。
   * **把整个捕获训练执行器丢进 `torch.compile`（N5）**：虽然旁路算子活动数从 59 降到 25/步，但 PyTorch 的函数化（functionalization）为了消除原地写入，把整个分页状态池克隆了一份（多出约 $39\ \mu\text{s}$ 的 `clone` 内核，`paged_adamw` 从 $29\ \mu\text{s}$ 恶化到 $64\ \mu\text{s}$），旁路 GPU 总耗时从 $\sim 178\ \mu\text{s}$ 涨到 $\sim 257\ \mu\text{s}$，端到端吞吐掉到 **90.10%**。
   * **过度融合引导采样的外层算术（N14）**：产生 $1.9 \times 10^{-6}$ 的数值偏差，未通过精度门槛被拒（保留位级一致的 B3）。
2. **关于重叠、调度与主机微优化的坑**
   * **在 GPU 已满载时强行提高旁路重叠度（N6）**：去掉采样器等待事件把露在主干外的旁路时间从 $70.33\ \mu\text{s}$ 压到 $8.57\ \mu\text{s}$，但训练挤进了带宽受限的采样阶段，总吞吐反而从 **94.49% 掉到 92.73%**。
   * **在 GPU 受限时死磕主机端与裁剪微优化（N11）**：在 `esamp94` 剖析中尝试过按活跃行/状态裁剪（少 14 个内核/步、省 39 MiB 预留，吞吐 91.70%/93.17%）、无环 `TensorBatch` 减 GC（GC 从 $52.8 \rightarrow 10.4\text{ ms}$，但吞吐 $36,522.9$ vs $36,529.2\text{ tok/s}$ 无变化）、`NamedTuple` 记录、写流优先级、CPU 绑核、复用完成选择等，**均无端到端可复现收益**——因为瓶颈在 GPU 上那 $100\text{–}140\ \mu\text{s}$/步的真实训练图回放。
   * **直接打补丁改 vLLM 0.32 调度器内部（N13）**：仅带来约 **1%** 收益，却引入了极脆弱的跨版本内部耦合，未合入。
   * **前向内精确 decode 更新的严格位级捕获（N12）**：增加约 600 条语句，容差模式下仅快 **+3.3% / +5.2%**，而严格位级模式下比 eager 还慢（$48.84 / 78.66\text{ tok/s}$）。
3. **关于 KV Cache 与注意力的坑**
   * **按注意力分数动态淘汰 KV 显存（SnapKV/H2O，N1）**：vLLM 块表只追加不挖洞且依赖单一 `num_computed_tokens` 标量，无法干净地回收中间物理块。
   * **LastKV 式的压缩远端记忆（M7，N2）**：`HiddenStateCacheSpec` 在 $8,192\text{ B}$ 页里只装 $2,048\text{ B}$（零节省），深层可用条目随 prefill 分块波动，且单请求每层耗时 $30.31\ \mu\text{s}$ 远慢于全量 FA2 的 $11.88\ \mu\text{s}$。
   * **在 4090 / A100 上调 vLLM 原生 DSA Top-K（N3）**：硬性要求 SM90+（Hopper），SM89/SM80 不可用。
   * **极小窗口下复用 FA2 `window_size`（N7）**：窗口 `(3, 0)` 耗时 $91.47\ \mu\text{s}$，比全量注意力（$16.56\ \mu\text{s}$）慢 5.5 倍。
   * **用 gather 出的 FP32 张量手算 `sink` 注意力（N16）**：吞吐仅 $8,867\text{ tok/s}$（全量的 22.6%），已被原生 R-SWA（$36,458\text{ tok/s}$，B8）取代。
4. **关于显存与 PCIe 传输的坑**
   * **用 DMA 拷贝引擎或无限制 UVA 内核搬主机表（N8）**：设备索引使 DMA 无法在不做 CPU 同步的前提下知道源地址；无限制 UVA gather 启动 37,984 个 CTA 占满 100% SM 饿死模型前向（催生了限制 1/4 SM 的 C2）。
   * **为量化模型在线训练缓存反量化后的稠密权重（N9）**：14B GPTQ 虽快 6.38 倍但多占 **$24.63\text{ GiB}$** 常驻显存（已被零额外峰值显存的瞬态反量化 A11 取代；注：bitsandbytes NF4/FP4/INT8 仅在草稿中探针测试过，且已从 vLLM 0.32 移除，见 N10）。
   * **无条件按调度器最大 token 数录制模型图（N15）**：7B 模型上挤空 KV cache（76,144 $\rightarrow$ 3,872 tokens，吞吐跌至 ~21%），已由 A7 修复。

---

## 附录：测量设置、比值口径与待复测说明

本文引用的所有数字均来自对齐的基准配置或教程 notebook 实测输出：

### 1. 标准硬件与对齐设置
* **S-4090（主力微基准与剖析设置）**：单卡 NVIDIA RTX 4090（128 SM，24 GiB），`.venv`（vLLM 0.27.1，PyTorch 2.13+cu129），通过 `flock` 独占 GPU 并用 `taskset -c 0-15` 绑核。模型为 `Qwen/Qwen2.5-0.5B-Instruct`（BF16），batch $8 \times \text{sampling\_n}=16$（128 条并发序列），生成 256 tokens，开启 `ignore_eos`，`temperature=0.8`、`top_p=0.95`、`top_k=-1`，关闭 prefix caching，开启 CUDA graphs，1 轮 warmup。**所有百分比均为同一次会话中相对于原生开启 CUDA graph 的 vLLM 吞吐比值**（参见 [《基准解读》](benchmarking.md)）。
* **S-tut（教程 notebook 输出设置）**：取自 `doc/tutorials/` 下各 `.ipynb` 实测输出（见 [《能力地图》](capability-map.md)），硬件为 RTX 4090，模型为 `Qwen/Qwen2.5-0.5B-Instruct`，8 条 prompt $\times n=16$（教程 06、07、12 为 $n=4$），生成 256 tokens，1 轮 warmup，**正式运行 2 轮取较好一轮（best of 2 rounds）** 与同一 notebook 内原生图模式 vLLM 求比值。其中教程 11 的主机/设备查表数字已更新为在 `main`（`ea7ff796`，RTX 4090，vLLM 0.27.1）上重跑的结果。
* **多卡与大模型专项设置**：TP（D1）、梯度/拷贝微基准（B9）及 14B GPTQ 瞬态反量化（A11/N9）在 NVIDIA A100-SXM4-80GB 上测得（`Qwen/Qwen2.5-7B-Instruct` 或 14B GPTQ）；In-Place TTT（E5、教程 08）在 RTX 4090 上使用 4B 社区权重测得（4096 或 16k prompt + 256 生成）。
* **静态计划数字（`explain@5cdc9785`）**：文中引用的 `pllm.explain` 字节数、槽位数与 FLOP 均在 `main` `5cdc9785` 上以 `max_num_seqs=128` 实跑 CPU 规划得出。

### 2. 版本与待复测说明（Gaps）
* **来自较早提交或分支、尚未在最新 `HEAD` 复测的专项数字**：`StepCapture` 前后对比（E2，测于 2026-10-18）、`publish_lag` 前后对比（F1，测于 2026-10-19）、真实代码追踪替代手写公式峰值对比（A4，测于 2026-10-19）、`pllm.scan` 并行前缀扫描微基准（F2，测于 2026-10-23 早期分支），以及受限主机查表 `pllm::host_rows` 的 5 轮剖析数据（C2，测于合入前的提交 `065b034e`）。
* **尚未单独拆出端到端吞吐数字的机制**：字段级写时复制（A2）、旁路退回内联的吞吐代价（A5）、共享 tap 缓冲（A12）、KV 卸载（A15）、感知负载的 prefix cache（A16）、同卡零拷贝投喂（A17）、分页 AdamW 独立加速比（B1）、共用与拆分 LM-head GEMM 的端到端吞吐差（B2）、行掩码裁剪（B5）、异步导出开销（C4），以及 `explain` 静态估算与实测硬件计数器的偏差。它们目前作为保障容量安全、位级正确性或组合能力的机制工作。
