# 性能与显存：代价从哪里来

算法本身只有几行张量运算，放进 decode 循环后却可能慢一截、甚至撑爆显存。这一页解释 decode 里什么花时间、什么占显存，
以及 pLLM 的各种机制和各种写法各要付出什么。怎么改写负载见[让负载更快](../howto/faster.md)，怎么测见
[公平地测吞吐](../howto/measure-throughput.md)，症状对照见[排查报错](../howto/troubleshooting.md)，怎么读 `pllm.explain`
的账单见[看成本：pllm.explain](../guides/explain.md)。

## 1. 心智模型：自回归 Decode 的三个物理瓶颈

在预填充（prefill）或离线训练中，GPU 一次处理几千个 token，瓶颈通常是算力（FLOPs）。但在自回归 decode 阶段，模型每一步只为每个并发请求生成 **1 个 token**。矩阵乘（GEMM）的算术强度（每从显存读一字节权重能做多少次乘加）约等于有效 batch 行数。在小到中等 batch 下，**GPU 算力几乎是过剩的，真正的瓶颈只有三个**：

1. **GPU 时间线有没有断掉（气泡）**：CPU 每步能不能把整串算子一口气喂给 GPU？只要有一次 CPU 等待 GPU（host sync），或者退回逐算子发射（eager 模式），GPU 就会大面积空转。
2. **每步多搬了多少字节（显存带宽与总线通信）**：权重、请求状态、词表级 logits 和主机内存查表每步要读写多少字节？
3. **给 KV Cache 剩了多少空间（显存容量）**：模型权重、CUDA Graph 私有内存池和 pLLM 的状态预留先分走固定份额，**剩下的显存才归 vLLM 的 KV cache**。多预留几百 MiB，并发上限就会断崖式下跌。

### 一组值得记住的“标尺数字”

在评估自己的 workload 之前，先看一组基准数字。除非另有说明，本文的数字都在以下设置下测得：**单卡 RTX 4090（24 GiB）、`Qwen/Qwen2.5-7B-Instruct`（BF16）、vLLM 0.32.0、128 条并发序列（8 条 prompt $\times$ 16 个采样分支）、每条生成 256 个 token（忽略 EOS）、开启 CUDA Graph、FlashInfer 采样器；吞吐用 `pllm.benchmark.throughput` 测，预热 1 轮后取 3 轮的中位数**。百分比都是相对同样设置下原生 vLLM（开启 CUDA Graph）吞吐的比值；每步耗时按“128 行 ÷ 吞吐”折算（含 prefill 的摊销）。另有设置的数字（0.5B 对照、多卡 A100、In-Place TTT 的 4B 检查点、较早的剖析）在出现处单独写明；每个数字的完整设置和测量日期记录在[实测记录](../reference/measurements.toml)里。

在这个设置下，原生 vLLM 跑一步 decode 约 **$21.7\text{ ms}$**（吞吐约 **$5,894\text{ tok/s}$**）。同样设置换成 0.5B 模型，一步只有约 $3.0\text{ ms}$（$42,355\text{ tok/s}$）。下面的很多代价在两种规模上差别很大：

| 操作或现象 | 实测耗时 / 吞吐代价 | 物理原因 |
| :--- | :--- | :--- |
| **退回 Eager 模式**（关掉 CUDA Graph） | 7B 上吞吐是图模式的 **98%**（$5,773\text{ tok/s}$）；同样设置的 0.5B 上掉到 **49%**（$20,926\text{ tok/s}$） | 逐算子从 Python 发射有固定的 CPU 开销。7B 一步要从显存读约 15 GB 权重，GPU 执行比发射慢，CPU 开销藏得住；0.5B 一步只有 3 ms，CPU 发射跟不上 GPU |
| **多算一次词表投影**（LM Head GEMM） | 约 **$+1.35\text{ ms}$ / 步**（整步的约 6%） | 7B 的 LM Head 权重为 $3584 \times 152064$（BF16），重达 **$1.09\text{ GB}$**，耗时几乎全花在从显存读这块权重上（微基准：128 行一次投影 $1349\ \mu\text{s}$；0.5B 的 $272\text{ MB}$ 权重为 $359\ \mu\text{s}$，约占 0.5B 一步的 12%） |
| **原生采样器扫一遍词表** | 7B 剖析中（`ESamp` $\beta=0$，128 行）temperature 约 **$125\ \mu\text{s}$**，top-k/top-p 约 **$283\ \mu\text{s}$**，Gumbel 采样约 **$78\ \mu\text{s}$**，合计约整步的 2% | $128 \times 152064$ 的词表张量在显存里反复读写；这部分只和行数、词表宽度有关，与模型大小无关，所以在一步只有 3 ms 的 0.5B 上占比大得多 |
| **跨 PCIe 搬一次宽表**（Host $\rightarrow$ GPU） | 物理传输约 **$1.6\text{ ms}$ / 步**（7B 一步的约 7%；0.5B 上超过半步） | 4090 上实测 PCIe 4.0 x16 读主机内存约 **$23.7\text{ GB/s}$**（显存带宽约 $1000\text{ GB/s}$ 的 1/40）；128 行词表宽的查表共需搬运 **$38.9\text{ MB}$** |

围绕这三个瓶颈，pLLM 在底层替你做了三件事：
1. **保住时间线**：稳态 decode 每步 0 次主机同步，把前向改写和旁路训练录进 CUDA Graph，并把每步的 CPU 端准备工作（0.5B 剖析中约 **$700\ \mu\text{s}$**，与模型大小基本无关）藏进 GPU 执行阴影里。
2. **省下显存容量**：用分页状态池、按角色自动选精度和 CPU 静态代码追踪，把显存留给 KV cache。
3. **少搬字节、聪明重叠**：能合并读权重的绝不读两遍；必须搬运或训练的，只在 GPU 对应硬件资源空闲时塞进旁路 stream 重叠执行。

---

## 2. 时间线：同步、CUDA graph 与旁路重叠

在 4090 上跑 7B 模型，一步 decode 约 $21.7\text{ ms}$；同样的 128 行换成 0.5B，一步只有约 $3.0\text{ ms}$。想在加了自定义算法后依然贴近原生吞吐，核心就是不让 GPU 停下来等 CPU，也不让自定义小算子打碎 CUDA Graph。步越短，这两件事越要紧。

### 2.1 主机同步为什么致命

在异步执行模型下，CPU 总是领先 GPU 一步准备元数据。在一次 0.5B、128 并发的剖析中（vLLM 0.27.1），pLLM 每步在 CPU 上的准备工作总计约 **$700\ \mu\text{s}$**，所需的槽位、掩码等元数据通过一张锁页 `int64` 表一次上传（H2D 约 **$6.24\ \mu\text{s}$**），完全藏在约 $3.3\text{ ms}$ 的 GPU 前向阴影里，造成的步间 GPU 空闲仅约 **$4\text{–}6\ \mu\text{s}$**。这部分 CPU 工作与模型大小基本无关，7B 一步约 $21.7\text{ ms}$，更藏得住。

一旦在方法里写下 `.item()`、`.tolist()`、`.cpu()`、`print(tensor)` 或 `if tensor.any():`，CPU 就必须停下来等 GPU 把前面的所有算子跑完，流水线被彻底排空；而方法又运行在 CUDA Graph 和旁路 stream 里，这样的代码根本无法录制。所以 pLLM 不让它跑起来：规划时用 meta 张量追踪你的方法，遇到这类操作直接报错，`pllm.explain` 和 `make_llm` 都过不去。改法（掩码与 `torch.where`）见[方法体里写什么](../guides/workload-anatomy.md#方法体里写什么)。

**机制与验证**：
* pLLM 保证稳态 decode 每步 0 次阻塞式主机同步，`check_on_gpu` 会检查这一点。在 RTX 4090 的 128 并发剖析中（0.5B），pLLM 产生的 Event/Stream 主机等待为 **0**，decode 窗口内 `cudaDeviceSynchronize` 空闲时间为 **0**。
* 遇到自适应深度分桶、计数 scan 完成判断等必须动态跳转的内部场景，pLLM 使用 CUDA Graph 内的设备端条件节点（CUDA IF nodes，需 CUDA 12.4+ 驱动路径）直接在 GPU 上跳转，不把条件拷回 CPU。
* `explain` 只能追踪普通方法里的数学。模型层内和注意力里的计算要用 `pllm.testing.check_on_gpu(workload, config)` 验证：它在 `torch.cuda.set_sync_debug_mode("error")` 下实跑至少 8 个稳态 decode 步，任何隐式同步都会立即报错。

### 2.2 哪些计算在 CUDA graph 里

CUDA Graph 会把一串 GPU kernel 的启动参数预先录制成一张图，之后每步只用一次极轻量的回放（replay）发射整串计算。pLLM 在三个层面维持图模式：
1. **任务与在线训练整包捕获**：在线训练（如 `@pllm.train`）包含前向、loss、掩码归约、`backward()`、写状态和分页 `AdamW` 等几十个小 kernel。pLLM 按最大行数和状态容量把它们录进同一张图；输入使用固定缓冲，未满 batch 的空行指向不会污染真实状态的哑槽位（sink state）。因此即便每步活跃请求集合在变，也**不需要重录图**：稳态下每个 decode 步只回放 1 次训练图，0 次重捕获。
2. **模型层内残差改写进图**：`@pllm.custom_residual` 会被编译为具名算子并直接录进 vLLM 的模型图里（详见 [4.4 节](#44-模型层内残差改写)）。
3. **自定义算子进图**：用 `torch.library.custom_op` 注册并附上 `register_fake` 的 Triton 或 PyTorch 算子，只要直接调用该 op 对象（不要调 `torch.ops.<namespace>`），就能被捕进图与旁路流（见[示例 09](../examples/09_custom_kernels.ipynb)）。示例 09 在 7B、128 并发下实测：在第 12 层挂一个自定义 Triton 算子，吞吐为原生 vLLM 的 **98%**（$5,760$ 对 $5,872\text{ tok/s}$），同样数学的 PyTorch 写法也是 **98%**（$5,755\text{ tok/s}$）。

`ActivationPatch` 通过 `custom_residual` 机制在 CUDA Graph 内执行。会让模型前向退回 eager（或替换整图）的只有两种写法：
* 共享可变 `InstanceState` 的 `pllm.model.Adapter`：退回 eager。
* `@pllm.custom_forward`：vLLM 的整模型图会被替换为 pLLM 按层切片捕获的分段图（`eager_reason` 会注明 `@pllm.custom_forward`）。

### 2.3 旁路 stream 重叠与 `publish_lag`：什么时候真能提速

当你在 `pllm.explain` 中看到某个任务（比如 `@pllm.train`）的 lane 是 `side` 时，意味着它在旁路 CUDA stream 上执行。但你需要建立一个关键直觉：**旁路 stream 并不是多了一张显卡。同一张 GPU 的 SM 和显存带宽是固定的——只有当主 stream 没有把某项硬件资源吃满时，旁路任务塞进来才叫“免费重叠”；如果主 stream 已经把带宽吃满了，旁路任务挤进来只是在抢同一块蛋糕。**

#### 为什么默认 `publish_lag=1` 能提速？

如果要求第 $n$ 步采完样立刻算完梯度、第 $n+1$ 步前向开头必须读到最新权重（即 `publish_lag=0`），那么每步前向都必须干等反向传播和 AdamW 跑完。

pLLM 默认采用 `publish_lag=1`：
* 第 $n$ 步 token 确定后，经过极短的拷贝栅栏，训练任务立刻放到 `side` stream 上异步执行；
* 第 $n+1$ 步前向直接读取第 $n-1$ 步的旧版本状态，不等待第 $n$ 步的训练；
* 第 $n$ 步算出的新权重在第 $n+2$ 步生效。

这样，旁路训练就能填进主干流在步间、小算子和非满载区段的空隙里。省下多少，取决于训练在整步里占多大比例。用 `ESamp`（rank 4）实测：
* **7B（本文默认设置）**：只训练不引导（$\beta=0$）时，`publish_lag=1` 为原生 vLLM 吞吐的 **97.4%**，`publish_lag=0` 为 **97.2%**；训练 + 引导采样（$\beta=0.1$）时分别为 **96.4%** 和 **96.1%**。差别在测量噪声附近：每步的训练只有几百微秒，相对 21.7 ms 的一步很小，串行等它也只多花这么一点。
* **0.5B（同样设置，只换模型）**：只训练不引导时，`publish_lag=1` 为 **91.9%**，`publish_lag=0` 为 **89.8%**。一步只有 3 ms 时，串行等待训练的代价才看得出来。

设 `ExecutionOptions(publish_lag=0)` 后，`explain` 中训练任务的 lane 从 `side` 变为 `main`，`timing` 行写着 `publish_lag=0 (you set): updates at step n become readable at n + 1; updates serialize before the next forward`；状态参数字段的页数也随之变少（见 [3.1 节](#31-分页状态池与每字段页数公式pages)）。

重叠不是越多越好。`ESamp`（$\beta=0$）开启 `publish_lag=1` 后，7B 上吞吐为原生的 97.4%。剩下的差距和旁路训练每步约 $616\ \mu\text{s}$ 的 GPU 时间（约整步的 2.8%）相当：训练与主干流共用同一块 GPU，而不是 CPU 调度没做好。强行提高重叠度（让训练挤进正在满负荷扫词表的采样阶段）只会让训练去和采样器抢显存带宽。同理，pLLM 已经把训练整包录进 CUDA graph，对方法再套 `@torch.compile` 没有可测的收益。

---

## 3. 显存：分页状态、精度与预算

在 vLLM 中，显存分配的顺序是：**模型权重 + CUDA Graph 捕获池 + pLLM 状态与任务工作区 $\rightarrow$ 剩下的才分给 KV Cache**。多预留几百 MiB，可用的 KV cache 和并发上限就随之下降，所以 pLLM 让下面每一字节预留都在 `explain` 里可见、可控。

### 3.1 分页状态池与每字段页数公式（`pages`）

当你声明 `pllm.RequestState` 或 `pllm.InstanceState` 时，pLLM 用一个分页的状态池管理显存：每个字段一个池，池里每页放这个字段的一份值；`explain` 的 `state` 行里 `x N pages` 是每个池预留的页数。状态更新时，框架执行**字段级写时复制（COW）**：只为这次更新改写的字段分配新页，没改的字段继续与旧版本共用同一页。

带 `optimizer=pllm.AdamW(...)` 的状态有四个字段：展平的参数 `parameters`、步数 `step`、一阶矩 `exp_avg`、二阶矩 `exp_avg_sq`（梯度不存成字段，优化器直接用这一步算出的梯度）。`Distill` 和 `ESamp` 的训练每步都改写这四个字段，所以训练产生的每个新版本都要四个字段各一页新页。为什么参数字段的页数会比 `max_num_seqs` 大？因为每个请求槽位在同一时刻可能要留住下面几份版本，每一份都有它的用处：

| 份数 | 什么时候需要 |
| :--- | :--- |
| 1 | 这一步读的版本：从这一步开始一直留到这一步提交，主干流和旁路流都在读它。 |
| $H$ | 落后读者要读的旧版本。只有当某个方法读落后的版本时才需要，`explain` 的 `state reads` 里会出现 `(boundary, H)`，$H$ 等于 `publish_lag`；例如第 n 步的训练还在旁路流写新版本时，第 n+1 步的 `predict` 读的是更早的版本。这几份里最新的一份就是当前已发布的版本，所以“最新版本”不用再单独占一份。 |
| 1 | 这一步训练正在写的新版本；没有方法写这个状态时不需要。 |

所以请求级状态被读者读的字段（参数、carry 字段）的公式是 $\text{pages} = (1 + H + W) \times \text{max\_num\_seqs}$，其中 $W$ 在有方法写这个状态时为 1、否则为 0。请求结束后，它留在落后读者那里的旧版本和接替它槽位的新请求共用这几份：新请求还没有旧版本可留。还在 GPU 上运行的读操作不需要额外的份数：一页被释放后，下一次写它会先在 GPU 上等前面的读完成，而不是另占一页。

AdamW 的 `step`、`exp_avg`、`exp_avg_sq` 只有优化器自己读，而且只读它正在更新的最新版本；旧版本（落后读者或请求准入时留住的版本）不保留这三个字段。所以它们只需要最新版本和正在写的新版本：$\text{pages} = 2 \times \text{max\_num\_seqs}$（实例状态为 2）。

* **读落后版本的可写 `RequestState`**：例如[负载的组成](../guides/workload-anatomy.md)里的 `Distill`（7B，hidden 3584，`predict` 读 `(boundary, 1)`）：`max_num_seqs=128`、默认 `publish_lag=1` 时，参数字段 $3 \times 128 = \mathbf{384}$ 页，三个优化器字段各 $2 \times 128 = 256$ 页，一个版本 $387,136\text{ B}$，总预留 $115,623,984\text{ B} \approx \mathbf{110\text{ MiB}}$。改成 `publish_lag=0` 后四个字段都是 256 页（约 95 MiB）。
* **只读最新版本的可写 `RequestState`**：例如 `ESamp`（$\beta=0$，只有 `loss` 读 `(latest, 0)`）、前向内精确更新（`@pllm.train_in_forward(visibility="exact")`），以及模型前向里读状态的适配器：$H=0$，每个字段 $2 \times \text{max\_num\_seqs}$ 页，与 `publish_lag` 无关。
* **只读 `RequestState`**：$\text{pages} = (1 + H) \times \text{max\_num\_seqs}$。
* **`InstanceState`**：可写实例状态按一个槽位计算，即参数字段 $1 + H + 1$ 页（例如 `OverRIDE` 在 `publish_lag=1` 时为 3，优化器字段各 2 页）。如果 `pllm.model.Adapter` 或 `pllm.model.ActivationPatch` 在请求准入时绑定读取它（每个请求一直读自己准入时的版本），且存在请求运行期间写入的任务——无论是步级写入方法（`@pllm.write`、`@pllm.train`）还是在线控制面更新（`import_state()`、`train_batch()` 与按批导入 `StateImport`）——在途请求留住的准入快照可能各不相同，因此读取端字段额外预留 `max_num_seqs` 页：$1 + H + 1 + \text{max\_num\_seqs}$（例如 [看成本：pllm.explain](../guides/explain.md) 中的 `Adapter` / `Imported` 场景在 `max_num_seqs=128` 时每个字段预留 $2 + 128 = 130$ 页）。不可变（`Zeros` / `Computed` / `Pretrained` 且无写入或导入任务）的实例状态只需 **1 页**；由 `pllm.Fed` 供给的状态固定 **3 页**（引擎读一份、一份排队、生产者写下一份）。

### 3.2 自动精度：什么时候用 BF16，什么时候必须用 FP32

声明状态字段时，默认 `dtype="auto"`。pLLM 会静态分析每个字段在 workload 中的读写角色，自动平衡显存占用与数值稳定性（规约、梯度累加与优化器内部计算始终保持 $\ge$ FP32）：

| 字段在方法中的角色 | `dtype="auto"` 自动选择的精度 | 原因与实测影响 |
| :--- | :--- | :--- |
| **被优化器更新的参数**<br>或 **“读出再写回”的累加器/跨块进位（carry）** | `torch.float32` | 防止微小更新被吞噬或长序列误差发散。在 32k tokens（chunk 64）递推测试中，carry 用 BF16 的相对误差高达 **$1.437 \times 10^{-1}$**，而用 FP32 仅为 **$3.692 \times 10^{-4}$**（低三个数量级） |
| **只读字段**（如静态引导向量、冻结参数）<br>或 **只写字段** | 跟随模型精度（如 `torch.bfloat16`） | 省下一半状态显存与读写带宽。按 `pllm.explain`（7B，128 并发），自动精度让 ReFT（`operator="add"`，一层）每请求状态从 FP32 的 $14\text{ KiB}$ 降到 **$7\text{ KiB}$**，FactorAdapter（rank 2，q/v 投影）从 $88\text{ KiB}$ 降到 **$44\text{ KiB}$**。反之，把大状态强行设为 FP32，状态的显存和每步读写字节都翻倍 |

需要时用 `pllm.Field(..., dtype=torch.float32)` 显式指定，写法见[状态](../guides/state.md)。

### 3.3 工作区与预算

pLLM 在绑定引擎时，会对你的真实方法代码（含 autograd 保存的张量、优化器和 scan）做生命周期追踪，算出精确的峰值工作区与捕获池预留（例如 7B、128 并发下，`ESamp`（rank 4）的 `predict` 在采样阶段预留约 **$133\text{ MiB}$**、旁路训练约 **$306\text{ MiB}$**；默认设置的 `OverRIDE` 旁路训练约 **$3.5\text{ GiB}$**）。

`pllm.ExecutionOptions` 上的四个旋钮分别限制这些预留：
1. **`execution_peak_budget_bytes`（两级优雅降级）**：包含状态页、任务工作区与存活结果、anchor 缓冲和捕获池预留；不含模型权重与 KV 池。旁路重叠（`side` lane）需要同时保留两步缓冲，因此峰值更高。以 7B、128 并发的 `ESamp`（rank 4，$\beta=0.5$）为例，旁路重叠计划峰值约为 **$1,775\text{ MiB}$**，内联（inline）下限约为 **$1,525\text{ MiB}$**。
   * 如果你设置的 `execution_peak_budget_bytes` 介于两者之间，规划器**不会报错，而是自动把旁路任务退回主 stream 内联执行**（`explain` 里该任务的 lane 从 `side` 变为 `main`），用一点重叠吞吐换取显存装得下。
   * 如果低于内联下限，规划失败，`explain(...).budget_errors` 会写明峰值需要多少字节、预算只有多少，并建议调小 `max_num_batched_tokens`、调大 `execution_peak_budget_bytes`，或把只读大表声明为 `placement="host"`。
2. **`state_page_budget_bytes`**：状态总预留（各字段页数 × 每页字节之和）超过该值时，`explain` 把缺口写进 `budget_errors`（建议调小 `max_num_seqs` 或调大预算），`make_llm` 在绑定阶段抛出 `ValueError`。默认 `0` 表示用后端实际可用的显存。
3. **`cuda_graph_pool_budget_bytes`**：控制每个任务/模型的 CUDA Graph 私有内存池预算，不是所有池的合计上限。默认 `None` 按负载追踪任务预留（模型图默认池为 **$64\text{ MiB}$**，exact decode overlay 另按形状计入工作区）。模型越大，改写后的模型图越大：7B 上哪怕只在一层挂一个 `@pllm.custom_residual`，模型图也要约 $84\text{ MiB}$，超过默认值；状态化 `@pllm.custom_residual` 子层（示例 07，8 并发）约 $1.9\text{ GiB}$，示例 12 的 `@pllm.custom_forward`（32 并发）约 $143\text{ MiB}$。这份预算会计入 pLLM 的显存峰值，按报错给出的字节数留一点余量即可。**捕获预算不足时会直接报错并给出所需字节数，绝不静默退回 eager。**
4. **`max_host_training_rows`**：只用于主机批训练（`llm.generate(..., batch=...)` 传入的 `WeightedCEBatch`、`CausalTrainBatch`），表示每个状态每步最多训练多少行，训练工作区按它规划；默认 `1`，批次超过时会报错提示调大。它不限制 decode 时 `@pllm.train` 在请求行上的训练。

**量化模型上的在线训练**：在 GPTQ/Marlin 量化底座上做在线训练时，pLLM 使用瞬态反量化：反量化后的稠密权重只活过一次反向传播，用完即释放，**峰值显存不增加**（14B GPTQ 在 A100 上每个训练步 **$0.193\text{ s}$**；4090 上 14B 的 QKV 反量化仅 **$0.104\text{ ms}$**）。预先缓存反量化权重虽然每步略快（$0.176\text{ s}$），却要多占 $24.63\text{ GiB}$ 常驻显存。

---

## 4. 各类写法的代价

### 4.1 采样阶段：logits 加法、引导与逐层 logits

框架把工作负载的 logits 操作并入 LM Head 和采样阶段，统一称为 **sampler epilogue**。`@pllm.add_logits` 的多个贡献按声明顺序加入，每次仍按输入 logits 的 dtype 舍入；`rows=` 只影响选中行，再照常执行 vLLM 的惩罚、掩码、温度和过滤。在需要处理 logits 的步骤中，框架把这些加法与已有的 FP32 采样输入拷贝合成一次遍历；跳过处理的步骤保留 LM Head 的 dtype，省去每个贡献的 dtype 转换、独立加法和行子集的全量拷贝；随后可继续组合 `@pllm.guide_sampling`。逐步路径不读取设备值到主机。请求 raw logprobs 时仍保留独立原始分布。

`explain` 将这次共享拷贝的输入读取、FP32 写回和一个内核归到第一个解码加法，后续加法只计各自贡献的读取字节；行选择按最大全批大小估计。这是逻辑流量估计，不是延迟预测。

#### 默认有界融合引导：容量为什么是 4096

`ExecutionOptions.guidance_capacity` 默认是 **4096**。在 TP=1、未量化且未缩放或变换的 LM Head 上，框架在 vLLM 完成过滤后只投影保留下来的 token，并把引导与抽样融合；其他 LM Head 会自动选择精确的 dense 路径。`guidance_capacity=None` 可显式选择精确 dense，`pllm.explain` 会报告实际路径、回退原因以及相应的 FLOP/字节估计。

**容量应让常见行不进入 overflow 路径，因为 overflow 行还需要 top-K 选择。** 保留下来的有限 logits 不超过 K 时，内核直接压紧、投影、抽样，无需排序。超过 K 时才选出 logits 最高的 K 个 token 进行引导，其余 tail 保持未引导并参与抽样（相同 logit 按 token ID 决胜）。若上游跳过了全词表行的过滤，且 top-1 概率达到 `top_p < 1`，则使用真正的单 token nucleus。

本文默认设置（7B、128 并发、`ESamp` rank 4、`beta=0.5`）下，各容量的吞吐如下（每个容量一次独立启动，3 轮中位数）：

| guidance capacity | 吞吐（tok/s） | 相对原生 vLLM |
|---|---:|---:|
| 256 | 5,684 | 96.4% |
| 1024 | 5,684 | 96.4% |
| **4096（默认）** | **5,678** | **96.3%** |
| `None`（精确 dense） | 5,533 | 93.9% |

在 7B 上，256 到 4096 的差别在测量噪声内，有界路径都比精确 dense 快约 2.6%。一步只有 3 ms 的 0.5B 上差别更明显：较早的一次 0.5B 测量（同样 128 并发、`beta=0.5`，七轮中位数）里，K=256 比 4096 慢约 3%，dense 慢约 6%。

在 0.5B 上的四组实测语料共 652,800 行中，最大的真实 nucleus 为 **2,785**，因此 K=4096 的真实支持集全部走简单路径；相对 dense 引导的策略误差 p99.9 TV 约 **$6\times10^{-14}$**，共享 Gumbel 抽样一致率为 **100%**。这些是该工作负载的测量结果，并非其他模型、过滤参数或 beta 的精度保证；有界路径在 overflow 时只引导 top-K，尖峰全词表行也会使用上述 nucleus 规则。需要精确 dense 行为时设置 `guidance_capacity=None`。

Dense 路径仍需额外的全词表 LM Head 投影和重加权：旧版 RTX 4090 测量中（0.5B），开启引导比 `beta=0` 每步多约 $401\ \mu\text{s}$，其中投影约 $118\ \mu\text{s}$、重加权约 $187\ \mu\text{s}$；7B 的 LM Head 是 0.5B 的 4 倍大，同一投影约 $1.35\text{ ms}$（见第 1 节的微基准）。这些 dense 成本不代表默认有界融合路径；有界路径的 `explain` 投影与流量数字是容量上界，不能把更小的上界直接当作更高的吞吐。

旁路训练的账也可以从剖析里算清：7B、128 行、`ESamp`（rank 4，`beta=0`）每步的旁路 GPU 时间约 $616\ \mu\text{s}$（约整步的 2.8%），其中分页 `AdamW` 约 **$191\ \mu\text{s}$**——它直接从源物理槽位读取参数和一阶/二阶矩并写回目标槽位，省去了对 5 个优化器状态字段做写时复制的整页搬运。

`pllm.layer_logits`（如 `DoLa` 读取多个中间层的词表分布）走独立的 GEMM，每多读一层就要多读一遍 LM Head 权重（7B 为 $1.09\text{ GB}$）。在 7B 上（32 个并发请求 = 8 $\times n=4$），`DoLa`（9 层）的吞吐为原生的 **66%**（见[示例 06](../examples/06_reading_inside.ipynb)）；按 `explain` 的估计，它让每个 token 的计算变成底座的 1.7 倍，而在 0.5B 上是 3.5 倍，小模型上代价更重。反过来，模型越大，引导采样的相对代价越小：一次全词表投影在 0.5B 上约占一步的 12%（$359\ \mu\text{s}$ / $3.0\text{ ms}$），在 7B 上约占 6%（$1.35\text{ ms}$ / $21.7\text{ ms}$）。本文设置下，`ESamp`（$\beta=0.5$）吞吐为原生 vLLM 的 **96.3%**（$\beta=0$ 为 **97.4%**）。

### 4.2 外部大表与词表级查表

以 7B 词表宽度的 $2048 \times 152064$（BF16）查找表为例，表本身占 **$594\text{ MiB}$**。128 个并发请求每步各查一行（每行 $304,128\text{ B}$），每步共需读取 **$38.9\text{ MB}$**。pLLM 对这种查表做了两层优化：

1. **表在显存（`placement="device"`）：Inductor 整图融合**
   对于只依赖 `pllm.input_tokens` 和不可变状态的不可微查表方法，pLLM 在绑定引擎时会自动将其中的 gather、类型转换、掩码（`where`）和缩放编译融合为**单次表读取 kernel**，省去把 $38.9\text{ MB}$ 宽向量写回显存再读出来做逐元素运算的往返流量，并把它提前到一步最开头的 `request.prepare` 发射。在 4090 上跑 7B（示例 11 的 `NGramPrior`，128 并发），设备端查表吞吐为原生 vLLM 的 **98%**（**$5,742\text{ tok/s}$**，约 $22.3\text{ ms}$/步）。融合可能改变舍入；需要严格位级一致时用 `ExecutionOptions(allow_numerical_transforms=False)` 关闭。
2. **表在主机内存（`placement="host"`）：受限 SM 的旁路 UVA 拉取**
   把 $594\text{ MiB}$ 的表改为 `placement="host"` 后，pLLM 在 GPU 上预留的显存峰值（`explain`，7B，128 并发，上面的 `LookupPrior`）从 **$1,464\text{ MiB}$ 降到 $870\text{ MiB}$**，正好少了整张表（$1\text{ GiB}$ 的 GPU 预算下也能挂载 $3\text{ GiB}$ 的主机表）。
   但每步 $38.9\text{ MB}$ 跨过 PCIe 4.0 x16（4090 上实测约 $23.7\text{ GB/s}$）需要实打实的 **$1.6\text{ ms}$**。如果直接用普通 GPU kernel 去主机内存拉数据，它会启动几万个线程块占满 4090 的全部 128 个 SM——虽然这些线程块只是在等慢速的 PCIe，但主干流的模型前向根本排不上队。在设计这条路径时的 0.5B 剖析中，$1,753.7\ \mu\text{s}$ 的读表只有 $26.2\ \mu\text{s}$ 与主干重叠；一个独立的小实验（4090，37.1 MiB，3.9 ms 的矩阵乘“前向”）里，不限线程块的读法与前向并发时只藏住了 $0.07\text{ ms}$。
   **pLLM 的做法**：跑满 PCIe 根本不需要多少 SM。pLLM 拦截对主机表的按行读取，改用自定义常驻 Triton 算子，**把线程块数量限制在 GPU SM 总数的四分之一**（4090 上只用 32 个线程块），并在一步最开头的 `request.prepare / side` 发射。这样，1/4 的 SM 在后台等 PCIe，剩下 **3/4 的 SM 空出来跑主干模型前向**：0.5B 剖析中读表时间的 **99.14%** 与主干 kernel 重叠；上面那个小实验里，32 个线程块读 37.1 MiB 用 $1.64\text{ ms}$（$23.7\text{ GB/s}$），与前向并发时总共只比单独跑前向多 $0.12\text{ ms}$。

| 配置（RTX 4090，7B，128 并发，594 MiB 表，示例 11 的 `NGramPrior`） | 吞吐（tok/s） | 相对原生 vLLM 比例 | 单步耗时 | pLLM 显存峰值（`explain`） |
| :--- | :--- | :--- | :--- | :--- |
| 原生 vLLM（无查表） | $5,877$ | 100% | $21.8\text{ ms}$ | — |
| 查表：`placement="device"` | $5,742$ | **98%** | $22.3\text{ ms}$ | $1,540\text{ MiB}$ |
| 查表：`placement="host"` | $5,615$ | **96%** | $22.8\text{ ms}$ | **$946\text{ MiB}$** |

可以看到，尽管每步跨 PCIe 搬运了 $38.9\text{ MB}$（物理耗时约 $1.6\text{ ms}$），**表放 `host` 比放 `device` 每步只慢了约 $0.5\text{ ms}$**，却省下了整张表的 $594\text{ MiB}$ 显存。在 7B 上，查表本身的代价也小：同样的读表和加法在 21.7 ms 的一步里只占 2%–4%；换成一步只有 3 ms 的小模型，同样的字节占比会大得多。

主机查表只有排在步首的旁路 stream 上才藏得住：查表写成只读 `pllm.input_tokens` 和只读表的独立方法，效果方法经 `pllm.result` 读它的结果；混读 `pllm.hidden` 或写进模型层内的方法，传输就暴露在 forward 之外（写法见[让负载更快](../howto/faster.md)）。粗估能否藏住：每步传输字节 ÷ 约 22 GB/s，明显短于一次 forward 就基本藏得住。pLLM 不会为了装下预算自动把表挪到主机内存，这个取舍由你显式声明。

### 4.3 定制 attention：省容量还是只省读取

写 `@pllm.custom_attention` 时，最重要的一条分界线是你的写法能否映射到底层 vLLM 的物理块管理器和原生 FlashAttention-2 / Triton 内核：

| 特性维度 | 静态几何布局（`keep="window"` / `sink=True` / `keep="none", source=k`） | 动态稀疏读取（`kv.attend(q, topk=...)`） |
| :--- | :--- | :--- |
| **省 KV 显存容量吗？** | **真省容量**：直接映射为 vLLM 原生 `SlidingWindowSpec`、`SinkSpec` 和跨层共享组，滑出窗口的物理块会被真正回收给池子 | **不省容量**：所有历史 token 的 KV 仍必须保留在显存中供打分和挑选 |
| **省读取带宽吗？** | **真省带宽**：直接调用原生 FA2 / FlashInfer 内核 | **省逻辑读取字节**（`explain` 中示例 10 的 `QuestRead` 在 7B 上每层每 token 最多读 $1,835,008\text{ B}$，全量为 $4,194,304\text{ B}$），但打分和 gather 走 PyTorch 算子 |
| **单请求 KV 显存上限**（`explain`）<br>（7B，示例 10：`max_model_len=2048`，128 token 分块 prefill） | 全量 **$112.0\text{ MiB}$** $\rightarrow$ 偶数层窗口 256 降至 **$70.4\text{ MiB}$**；共享一半层降至 **$56.0\text{ MiB}$**（0.5B 上为 24.0 $\rightarrow$ 16.6 / 12.0 MiB，可同时容纳的 2048-token 请求从 358 个增至 518 / 716 个）。`sink=True` 保留整个 prompt，上限不变，实际节省取决于生成长度 | 不变（7B 仍为 **$112.0\text{ MiB}$**） |
| **实测吞吐**<br>（示例 10：RTX 4090，7B，128 并发，每步最多 128 token） | 原生读取（`NativeRead`）、偶数层窗口（`LocalGlobal`）、跨层共享（`CrossLayerShared`）均为原生的 **99%** | 示例 10 中 `QuestRead` 仅为原生的 **13%**（$649$ 对 $4,855\text{ tok/s}$），因为 PyTorch 级打分与 gather 开销远大于省下的读取 |

窗口首先改变模型能看见的历史，并回收窗口外的物理块；它能否加速，还取决于内核是否让同组 query 头共享 KV。Qwen2.5-0.5B 有 14 个 query 头、2 个 KV 头，Qwen2.5-7B 有 28 个 query 头、4 个 KV 头，每组都是 7 个 query 头。FA2 的窗口路径会关闭 decode 的 GQA 打包：同一次读的 grid 从按 KV 头组织变成按 query 头组织，重复处理 KV，窗口更小也未必更快。Ampere/Ada 上，普通存储窗口使用 vLLM 的 FlashInfer decode/prefill wrappers 和 `window_left`；全量、prompt-sink、共享层读取及临时改变窗口或头数的读取保留各自适用的路径；要求 batch invariant、头维不在 FlashInfer 支持范围内或 query 头维不连续时保留 FA2。

FlashInfer 的 softmax 分母累加 BF16 舍入后的概率，LSE 再由底数 2 转成自然对数；FP32 返回类型不代表与 FA2 位相同。窗口回收的独立参考计算中，输出相对 FP32 的最大绝对误差约 0.0079，LSE 约 0.0017。同一组 query 上，FlashInfer 与 FA2 对 FP64 输出的最大误差都是约 0.0076；LSE 与按该 BF16 分母计算的 FP64 参考差异不超过 0.00005。若算法对很小的 LSE 差异敏感，应把这个舍入量级纳入质量验证。

按注意力分数动态淘汰物理 KV 块（SnapKV、H2O 一类）做不到：vLLM 对每个请求只维护一个 `num_computed_tokens` 计数，位置编码与物理槽位都由它线性推出，块表只能追加、不能在中间挖洞。要省显存用静态布局，只想少读用 `kv.attend(q, topk=...)`。布局改的是模型数学：未经相应训练的模型换成窗口或跨层共享，回答质量可能明显变差，先在自己的任务上验证质量。

### 4.4 模型层内残差改写

`pllm.model.ActivationPatch` 通过同一套 `custom_residual` 模型站点在 CUDA Graph 内修改层输出。
而 `@pllm.custom_residual` 会被 pLLM 降级为具名算子 `pllm::model_site`，并用设备端索引读写私有状态页，使 vLLM 在录制整模型 CUDA Graph 时**把你的残差改写一并录进图里**。

示例 07 在 RTX 4090、7B、生成 256 token 下的实测吞吐占原生图模式比例：
* **Value Residual**（改写 27 层的 value，32 个并发请求）：**99%**
* **Hyper-Connections（$n=4$，56 个子层，32 个并发请求）**：**93%**
* **带状态记忆子层**（每个请求一份 $3584 \times 3584$ 的 FP32 矩阵，约 49 MiB，所以只开 8 个并发请求；模型图约需 1.9 GiB 捕获池）：同样 8 个并发请求的原生图模式的 **75%**。

这一组设置下原生 vLLM 的 eager 模式也有图模式的 99%（7B 一步约 19 ms），所以 7B 上看不出“进图”本身挽回了多少；早先在 0.5B 上，同样的改写不进图只有图模式的 12%–34%。

挂了 `custom_residual` 的模型图里，vLLM 录的是原始算子而不是 Inductor 融合后的算子，所以即使是恒等改写（`x + y`）也有代价：7B、128 并发下，只在第 12 层挂恒等改写时吞吐为原生图模式的 **97%**，每层的 MLP 都挂也是 **97%**。这点差距来自整张模型图失去 Inductor 融合，和挂几层关系不大；改写本身多出的计算另算，所以仍只在真正需要改写的层上挂载（用 `layers=...` 指定）。评估改写本身的代价时，可以和 eager 模式的原生 vLLM 对比，或直接用 `check_on_gpu`（它会构建匹配的原生引擎）。

### 4.5 前向内训练、线性扫描与自适应深度

**线性递推 `pllm.scan`。**在 chunk 内做线性递推时，如果用 Python 循环串行步进，GPU 的 SM 根本填不满。用 `pllm.scan` 写递推后，pLLM 的规划器会自动识别形如 $S_t = A_t S_{t-1} + B_t$ 的仿射递推，在显存工作区预算允许且系数为 FP32 时，替换为**并行前缀扫描（Parallel Prefix Scan）**：
* **微基准**（只测扫描本身，与模型大小无关）：单头 $64 \times 64$ 状态扫描从串行的 $1.439\text{ ms}$ 降到 **$0.219\text{ ms}$（6.57 倍）**；14 头 $64 \times 64$ 为 **1.85 倍**；完整状态转移从 $5.879\text{ ms}$ 降到 **$1.759\text{ ms}$（3.34 倍）**。
* **自动回退**：当状态维度很大（如 $d=896$）导致树形扫描不划算时，形状守卫会保留串行循环；CPU 上、部分扫描或需要求导的调用也保持串行。你也可以设 `ExecutionOptions(tuning="measured")` 让引擎在绑定时实测挑选最快实现，或用 `allow_numerical_transforms=False` 禁止并行重排。

**前向内精确更新（`@pllm.train_in_forward(visibility="exact")`）。**默认情况下，`@pllm.train_in_forward` 使用 `visibility="forward_pinned"`（同一次前向的所有 chunk 读同一个起始状态）。对于 In-Place TTT 等要求“同一前向内后一个 chunk 必须读到前一个 chunk 更新后的快权重”的算法，需声明 `visibility="exact"`：
* **自动降级与实测表现**：pLLM 会把 `visibility="exact"` 的 decode 阶段捕获进 CUDA Graph（prefill 仍为 eager），并用设备端条件节点（CUDA IF）包裹只在凑满整块时才触发的快权重更新。同时，pLLM 通过静态追踪方法体的计算图自动完成三项底层优化：① 在未凑满整块的非边界 decode 步直接共享只读快权重页（0 字节写时复制），只为尾行缓冲分配私有页；② 原地借用来自 `ModelCheckpoint` 的只读共享参数而不重复拷贝；③ 识别形如 `inputs @ delta.to(inputs.dtype).T` 的线性读出并降级为直接从分页状态池流式读取的内核（自动跳过未满批的空闲槽位）。示例 08 在 RTX 4090 上跑 In-Place TTT 的 4B 社区检查点（vLLM 0.32.0，CUDA Graph 开启，FP32 快权重，贪心生成 256 token，预热 1 轮后 3 轮中位数）：1 个请求、4096 token 的 prompt 达 **$73.02\text{ tok/s}$**（16384 token 的 prompt 为 **$47.97\text{ tok/s}$**）；官方 HF 实现在同一张卡、同样协议下为 $54.79 / 34.89\text{ tok/s}$。4 个请求一起跑合计 **$171.56\text{ tok/s}$**（官方实现硬编码 `bs=1` 只支持 1 个请求）。

**自适应深度（`@pllm.custom_forward` + `active=`）。** 被跳过的层做的是状态传播：一行在某一段层上 `active=False` 时，它的残差流原样穿过这段层（等价于把这几层当作恒等映射）。但后面的 token 在每一层做 attention 时，仍然要读到这个位置在这一层的 K/V，所以 pLLM 会为这行补写被跳过层的 K/V：拿它停下时的隐藏状态，经过每个被跳过层自己的输入 norm、K/V 投影和 RoPE 算出 K/V 写进 KV cache（第一次跳过时补写一次，约占一层计算量的 7%；之后再跳过同一层时保留已写入的值）。这就是 CALM 提出的“状态传播”：它给出的 K/V，正是“这些层输出等于输入”这个假设下该位置本应有的 K/V，误差只局限在被跳过的 token 自己的深层状态上。

其他做法为什么不作默认：直接复制上一层的 K/V 不对（不同层的 K/V 投影在不同子空间里）；“留空洞”（被跳过的 token 不进这一层的 attention，Mixture-of-Depths 的做法）只适用于按这种方式训练过的模型，而且要求每层维护各自的 KV 长度，vLLM 的 KV 管理做不到；“推迟补算真实深层”（FREE 的做法）会让每个请求每步处理的 token 数变得不固定，和固定形状的 CUDA graph 冲突。

跳层能省多少：
* **为什么 128 个 token 里跳过 50% 并不会让这几层快一倍？**
  因为 decode 阶段的矩阵乘是**权重带宽受限**的（例如 7B 模型单层权重达 $466\text{ MB}$）。哪怕 128 行里只有 1 行 `active=True`，GPU 也必须把这一层几百 MB 的权重完整读进 SM 才能算这 1 行！被跳过的行只省下了它自己的注意力 KV 读取流量（且首次被跳过时仍需用原生算子补齐约占单层开销 7% 的 norm/QKV/RoPE，以维持后续 KV 连续）。 <!-- check-docs: not measured -->
* **什么时候才能省下读权重的带宽？**
  pLLM 使用 CUDA Graph 条件节点将活跃行压实（compact）到 2 的幂次桶里执行。**只有当整个 batch 在该层全部跳过（活跃行数为 0，落入 `bucket 0`）时，GPU 才会通过条件节点直接跳过整层算子，真正省下读权重的带宽。**
* **实测数字（示例 12：RTX 4090，7B，32 个并发请求 = 8 条 prompt $\times n=4$，256 token，28 层中的第 16–19 层可跳）**：
  * 从不跳过：原生图模式吞吐的 **99%**（原生 eager 在这个设置下也是 99%：7B 一步约 19 ms，发射开销藏得住）
  * 约 23% token 跳过（阈值 0.85）：原生图吞吐的 **98%**
  * 整批全部跳过（落入 bucket 0）：原生图吞吐的 **112%**，和少读 4/28 层权重相符。
* **上下文更长、batch 更大时情况不同。** 上面说的是短上下文的 decode：时间主要花在读权重上。上下文很长时（例如几 k 到几十 k），每行要读自己全部历史的 K/V，这部分可能超过读一层权重的量；被跳过的行不做这一层的 attention，这部分读取会按跳过的行数成比例省下（原理如此，尚未实测）。batch 很大或在 prefill 那样受算力限制时，压紧活跃行直接省下矩阵乘的计算量。

prompt token 为什么不跳层：它唯一的作用是给后面所有 token 提供 K/V，深层 K/V 被近似后影响整段回答，还会经 prefix cache 和 n 个样本扩散出去（Ouro 论文中 prefill 用近似 K/V 会掉 10 个以上 GSM8K 点，decode 几乎不掉）；prompt 又是整块并行计算的，压紧后的行不连续，省下的矩阵乘大多被额外的 attention 开销抵消。所以 prompt 默认按完整深度运行。

---

## 5. 多卡张量并行（TP > 1）

当模型单卡装不下或需要更高带宽时，可以通过 `ModelConfig(tensor_parallel_size=...)` 开启多卡张量并行（需 vLLM 0.32.0）。

在张量并行下，vLLM 每算完一层注意力或 MLP 都要做一次全卡同步（Custom All-Reduce）。如果：① pLLM 自己也发起跨卡集合通信，或者 ② 旁路流上的在线训练刚好在某张卡上和 vLLM 的 All-Reduce 撞在一起抢带宽，全卡就会一起被卡住。

因此，pLLM 在 TP 下采用两条机制：
1. **零 pLLM 集合通信**：hidden tap、路由计算、`PositionState` 和请求级小状态在每个 TP rank 上各存一份完全相同的副本，各卡独立执行完全相同的确定性逻辑（实测各 TP 规模下与原生 vLLM 的 logits 均方根差为 0），`@pllm.export` 只在 rank 0 交付。
2. **流优先级与发布错峰**：将 TP 推理主 stream 设为高优先级（`-1`）、旁路训练 stream 设为普通优先级（`0`），并让自动状态更新在采样前写入私有状态、在 token 提交时发布，使**旁路训练与 vLLM Custom All-Reduce 的重叠时间降为 $0\ \mu\text{s}$/步**。

**实测表现（A100-SXM4-80GB，`Qwen/Qwen2.5-7B-Instruct`，`ESamp` $\beta=0$，128 并发）**：
* **TP = 1**：原生 vLLM 吞吐的 **95.84%**
* **TP = 2**：原生 vLLM 吞吐的 **95.63%**
* **TP = 4**：原生 vLLM 吞吐的 **97.60%**

TP 不会减少单卡上 pLLM 状态的显存占用：状态按 rank 复制而不是分片（`pllm.explain` 会提示 `reservations are per rank`），TP 省下的是底座权重。哪些功能能多卡见[支持矩阵](../reference/support-matrix.md)。

---

## 自测

1. **问题**：既然 `lane: side` 可以在后台流上跑，如果把一个每步要读写几百 MB 显存的重度任务塞进 `side` 流，能不能白嫖时间？
   <details><summary>点击查看答案</summary>
   不能。Decode 阶段主干模型本身就是显存带宽受限的。如果旁路任务也在大量读写显存，它会和主干的投影或采样器争抢同一份显存带宽，导致主干变慢。只有当旁路任务占用的资源与主干互补（例如只用少量 SM 等待慢速 PCIe，或者填补步间空隙）时，重叠才能带来净收益。
   </details>

2. **问题**：你写了一个查表方法，既读取了 `pllm.input_tokens()` 去 `placement="host"` 的大表里查向量，又在同一个方法里读取了 `pllm.final_hidden()` 做门控。为什么加上这个表之后每步突然多出了 $1\text{–}2\text{ ms}$？ <!-- check-docs: not measured -->
   <details><summary>点击查看答案</summary>
   因为同一个方法里读取了 <code>pllm.final_hidden()</code>，这个方法就必须等到模型前向跑完之后才能启动（<code>explain</code> 中它的 anchor/lane 是 <code>state.publish / main</code>），无法在一步最开头的 <code>request.prepare / side</code> 发射。结果约 1.6 ms 的 PCIe 传输完全暴露在模型前向之外。解决办法是把只依赖 <code>input_tokens</code> 的主机查表拆成独立的方法，用 <code>pllm.result</code> 把结果交给读 <code>final_hidden</code> 的效果方法，让规划器能把查表提前到步首与前向重叠。
   </details>

3. **问题**：在 7B 模型上，你用 `@pllm.custom_forward(tokens="decode")` 实现了早退（Early Exit），发现在 128 并发下有 30% 的 token 跳过了后 4 层，但端到端吞吐几乎没有提升。这是为什么？
   <details><summary>点击查看答案</summary>
   Decode 阶段的层计算是权重带宽受限的。只要 128 个并发 token 里还有 70%（甚至哪怕只有 1 个 token）需要跑后 4 层，GPU 就必须把这 4 层的全部权重从显存读进 SM。部分跳过只省下了那 30% token 的 KV cache 读取流量；只有当整个 batch 在这 4 层全部跳过（落入 <code>bucket 0</code>）时，才会省下读权重的带宽。
   </details>
