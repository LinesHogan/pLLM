# 给推理引擎装一个“eBPF”：pLLM 为什么存在

在单请求的研究环境里，验证一个推理时（inference-time）算法的想法通常只需要几十行代码。

以在线蒸馏引导采样（ESamp）为例：大模型在自回归生成时，浅层往往已经能预测出高频的套话，而更深层的语义要到最后几层才会浮现。如果在生成每个 token 的同时，为当前请求在线训练一个秩为 4（rank-4）的双层小网络，用第 0 层的隐藏状态（hidden state）去拟合最后一层的隐藏状态；到了采样前，把这个小网络“浅层就能猜到”的预测从模型的词表得分（logits）里扣掉一部分，就能压低平庸词的概率。

在 Hugging Face 的 `generate` 循环里写出这个原型并不费事：定义一个 `nn.Sequential`，往第 0 层和最后一层各挂一个 PyTorch 的 `register_forward_hook`，每步生成后调一次 `loss.backward()` 和 `optimizer.step()`，不到二十行代码就能跑通。

但在单请求脚本里跑通，距离在生产级推理引擎（如 vLLM）里上线，中间隔着一道巨大的工程鸿沟。在 pLLM 中，同样的算法写成声明式负载依然只有十几行：

```python
# check-docs: skip: excerpt from README.md
class ESampLite(pllm.Workload):
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    @pllm.guide_sampling(beta=0.5)
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)
```

这段代码表面上只写了张量算术，但如果你不用扩展层，而是直接把原先那二十行带 forward hook 和优化器的 PyTorch 代码塞进 vLLM，它会立刻撞上现代推理引擎赖以生存的五条底层约束。

---

## 一、 从研究脚本到生产引擎的五道屏障

研究代码默认自己独占整张 GPU、每次只处理结构规整的一批请求；而服务引擎的目标是在延迟上限内榨干每一分显存带宽和算力。这两种假设在五个地方直接冲突。

第一个冲突来自连续批处理（Continuous Batching）与扁平张量。在研究脚本里，张量是三维的 `[batch_size, seq_len, hidden_size]`，第 0 号请求永远住在 `hidden[0]`。但在服务引擎里，为了不让短请求干等长请求，调度器在每一步（engine step）都会把几十个正在逐字生成的解码（decode）请求（每个请求 1 个 token）和几个正在分段处理的预填充（chunked prefill）请求（每个请求几十到几百个 token）首尾相接，拍扁成一张紧缩的二维张量 `[num_tokens, hidden_size]`。

```text
研究脚本里的张量 (Hugging Face):             服务引擎里的张量 (vLLM Continuous Batching):
┌───────────────────────────┐               ┌───────────────────────────────────┐
│ Req A: [t0, t1, t2, pad]  │               │ Row 0:    Req B (decode, tok #41) │
├───────────────────────────┤               │ Row 1..4: Req C (prefill chunk)   │
│ Req B: [t0, t1, pad, pad] │      ──►      │ Row 5:    Req A (decode, tok #12) │
└───────────────────────────┘               │ Row 6..7: Graph Padding           │
  形状规整 [B, S, H]，位置固定                 └───────────────────────────────────┘
                                              扁平张量 [N, H]，每步洗牌，末尾带填充
```

当你的 forward hook 在第 0 层被调用时，拿到手的只有这张二维张量，没有任何标记告诉你哪一行属于哪个请求、哪几行是还没算完的半截 prompt、哪几行又是为了凑齐硬件图尺寸塞在末尾的填充行（padding）。到了下一步，老请求退场、新请求插入，或者某个请求因为显存不足被临时抢占（preempt）驱逐、稍后从头重算，整张表的行号顺序又会彻底洗牌。

第二个冲突来自 CUDA Graph。就算你在 Python 里弄清了每一行是谁，一旦进入稳态 decode 阶段，你会发现挂在模型层上的 Python hook 根本不会被执行。原因在于延迟账单：decode 阶段每个请求每步只推进 1 个 token，GPU 算子执行时间极短。在单张 NVIDIA RTX 4090（24 GiB）上运行 Qwen2.5-0.5B-Instruct（BF16，8 条 prompt × 每条采样 $n=16$ 共 128 个并发序列，生成 256 个 token，vLLM 0.32.0 开启 CUDA Graph 与 FlashInfer 采样器，预热 1 轮后取 3 轮中位数；把模型换成 7B，就是本系列默认的标准测试台），跑完一整步 24 层前向加采样总共只需要 3.0 ms（吞吐 42,355 tok/s）。如果让 Python 解释器在每一步里逐个发射几百个 GPU 内核，每次发射的几微秒 CPU 开销加起来比 GPU 干活的时间还长。

因此，引擎会在启动时把整个 decode 步录制成一张静态的 CUDA Graph：预热录制时跑一遍 Python 代码，把沿途发射的内核与它们读写的显存指针全部焊死在图里；线上服务时，CPU 每步只发一条重放（replay）指令，GPU 自己顺着图跑完，Python 解释器完全不参与层的执行。如果为了运行动态 Python hook 而强行关掉 CUDA Graph 退回 eager 模式，0.5B 模型单步耗时就会从 3.0 ms 拉长到 6.1 ms，吞吐跌至 20,926 tok/s，直接腰斩到图模式的 49%。在同样配置下，单步耗时 21.7 ms 的 Qwen2.5-7B-Instruct 虽然因为矩阵乘够重，关掉模型图后原生吞吐仍能保持在 5,773 tok/s（图模式为 5,894 tok/s，约 98%），但扩展任务自己的前向、反向求导和 `AdamW` 更新包含几十个细碎算子，若不录进 CUDA Graph，同样会被 CPU 发射开销拖垮。而一旦录进图，所有算子就只认录制那一刻绑定的固定显存地址。

第三个冲突是每请求状态的生命周期与并发隔离。在 ESamp 中，128 个并发请求各自拥有一套独立的 `distiller` 权重与 `AdamW` 的一阶、二阶动量。这些状态必须在请求到达时初始化、请求结束时立即回收，同时它们的物理地址还必须对 CUDA Graph 保持固定。更棘手的是多流并发：如果把第 $n$ 步的在线训练放到旁路 CUDA 流（side stream）上异步执行，好让它与主流（main stream）上第 $n+1$ 步的模型前向重叠，那么当旁路流的优化器正在改写某个请求的权重时，主流可能同时正在读取同一份权重。如果采用原地修改（in-place mutation），主流就会读到写了一半的撕裂数据。

第四个冲突是显存预算与主机同步（host synchronization）的零和博弈。一张显卡的显存要先装下模型权重，再划出 CUDA Graph 的静态内存池和扩展状态缓冲区，最后剩下的空间才留给 KV Cache。扩展层多占几百兆显存，能同时服务的并发序列数就会跟着缩水——例如在 24 GiB 的 RTX 4090 上运行 7B 模型时，如果把模型图的捕获上限拉大到 8,192 个 token，图内存池会挤占大量空间，使可用 KV Cache 容量从原生 vLLM 的 76,144 个 token 跌到 3,872 个，端到端吞吐只剩原生的约 21%（`gpu_memory_utilization=0.85`，vLLM 0.32）。在时间线上同样如此：哪怕整步里只有一次 `.item()` 或基于张量数值的 Python `if` 判断，CPU 都必须停下来等待 GPU 把队列里的内核全部排空。异步流水线里，减少几次同步往往看不到效果，只有把每步的主机同步彻底清零（在一版较早的 ESamp 图模式实现上——0.5B、vLLM 0.27.1、128 并发——清零同步直接让吞吐从 11.6k 跳升至 19.1k tok/s），CPU 的准备工作才能真正藏进 GPU 的计算阴影里。

第五个冲突在于模型内部结构的耦合。当算法不再满足于在模型外围读隐藏状态，而是要改写某层的残差连接（如 Hyper-Connections）、在投影层叠加每请求在线更新的快权重（如 In-Place TTT）、让部分 token 在深层提前跳过（Early Exit），或者将特定层的注意力改为滑动窗口与跨层共享时，它会发现模型前向早已被 `torch.compile`（Dynamo 与 Inductor）编成了计算图，残差加法被融进了下一层的 `fused_add_rms_norm` 内核，而注意力层与底层的块表管理器紧密咬合。不拆改引擎源码，这些位置连挂载的入口都没有。

---

## 二、 为什么说这是推理引擎里的“eBPF 问题”

把这五道屏障放在一起看，会发现它们没有一个是机器学习问题，全都是典型的系统工程问题。

过去几年里，大模型推理遵循的是极简的扩展定律（Scaling Laws）：除分词器外，整个系统只有一个匀质、无状态的 Transformer 主干。但随着硬件单芯片算力与显存带宽的增长放缓，而实际应用对降低单 token 推理成本的要求越来越苛刻，推理侧的算法正在明显走向复杂化与有状态化：有的挂接数 GiB 的外部 n-gram 查找表（如 Engram），有的在上下文推进时做测试时训练（TTT），有的动态路由或跳过层，有的在线训练轻量预测器干预采样。

这些五花八门的新算法在落地时，面对的其实是同一个系统命题：**在一个为了极致性能而高度静态化、同时服务上百个租户的底层引擎里，怎样允许外部开发者安全、高效地挂载一段带状态的自定义程序，而不必每次都把引擎源码拆开重装？**

操作系统内核在 eBPF 出现之前遇到过几乎一模一样的困境。

在 eBPF 出现之前，如果想在 Linux 内核的网络路径或系统调用入口加入超出简单过滤的自定义处理逻辑，开发者只有两个选择：要么把数据全部拷贝到用户态处理，写法自由安全，但上下文切换与拷贝开销会让高吞吐网络直接瘫痪；要么编写 Linux 内核模块（Kernel Module），直接挂进内核函数指针拿到全速性能，但代价是任何一个越界指针或未释放的锁都会导致整机崩溃（Kernel Panic），而且内核内部结构一旦升级，模块就得跟着重写。

今天的大模型推理生态正处在这个节点上。在 Hugging Face、TransformerLens 或 nnsight 里用 Python hook 写原型，就像在用户态写脚本，灵活但没有连续批处理与 CUDA Graph，无法直接用于高并发服务；而为了上线去 fork 一份 vLLM 源码、直接在 `GPUModelRunner` 和底层算子里动刀，就是在写内核模块——每个新算法拉一个分支，抢占回滚、多租户隔离和显存泄漏的坑全靠自己填，上游版本一升级就面临大面积冲突。至于引擎自带的轻量接口，如只能改最终词表分数的 `LogitsProcessor`，或形态固定的只读 Multi-LoRA，则覆盖不了“在模型内部读写、跨步维护可变状态、在图内在线更新”的需求。

Linux 走出这个死胡同靠的是 eBPF。它没有向用户开放任意修改内核的权限，而是提供了一套受限但正交的机制：装载前的静态验证器（Verifier）、将字节码编译为原生指令的 JIT、内核统一托管的键值/数组存储（BPF Maps）、明确的内核挂载点（Attach Points），以及配套的资源观测工具（如 `bpftool`）。

pLLM 的整体架构正是沿着同样的思路设计的。它把扩展推理引擎所需的能力拆成了对应的五个部分：

| Linux eBPF 组件 | pLLM 中的对应机制 | 承担的系统职责 | 对应篇章 |
| :--- | :--- | :--- | :--- |
| **Verifier**（装载前验证） | **声明反射 + Rust 规划 + `pllm.testing.check`** | 用 `meta` 张量静态拦截 `.item()` 同步与非法依赖；在 CPU 上用确定性调度做蜕变测试，检验请求隔离、行置换与抢占重算一致性 | [《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md) |
| **JIT**（原生全速执行） | **StepCapture 整步录图 + Inductor 编译** | 把用户的张量前向、反向传播与优化器更新录进 CUDA Graph；图内挂载点方法体用 Inductor 编译，末尾加法交还原生融合算子 | [《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)<br>[《图内挂载点》](05-in-graph-attach-points.md)<br>[《参数覆盖层》](06-parameter-overlays.md) |
| **BPF Maps**（受管状态存储） | **PTTS（分页张量状态迁移系统）** | 为请求或实例提供基地址固定、带不可变版本号、支持字段级写时复制（COW）、页引用计数与跨流事件栅栏的分页显存池 | [《PTTS》](01-ptts.md) |
| **Attach Points**（挂载点） | **时间线锚点 + 模型内部挂载点** | 步内划分 8 个有序时间轴锚点与主/旁路流；模型内挂载残差流、线性投影、注意力 KV 与采样尾声 | [《图内挂载点》](05-in-graph-attach-points.md)、[《参数覆盖层》](06-parameter-overlays.md)、[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)、[《采样尾声融合》](07-sampler-epilogue.md) |
| **Observability**（可观测性） | **`pllm.explain` + 异步 Trace** | 不接触 GPU 即可在 CPU 上输出流调度计划、状态版本份数与精确到字节的显存预留账单 | [《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md) |

这里顺带说明一下命名：表中 PTTS（Paged Tensor Transition System）开头的 P 是 Paged，即分页与页表，与操作系统的虚拟内存、vLLM 的 PagedAttention 同源；它和 pLLM 名字里代表 programmable 的 p 没有关系，只是碰巧首字母相同。

当然，这个类比有它明确的边界。首先，eBPF 的验证器面对的是受限字节码，在装载前静态地证明程序满足一组安全性质（必定终止、内存访问不越界）；而 pLLM 面对的是任意 PyTorch 算子组成的 Python 函数，无法从数学上形式化证明一段张量代码绝不跨行串味，因此我们用的是“静态形状/同步拦截 + 固定调度下的差分测试”——它能高效抓出常见的跨请求污染与抢占 bug，但不是数学证明。其次，eBPF 探针通常只有几百条指令，而 pLLM 的任务可以在旁路流上跑完整的反向求导与优化器，或是跨 PCIe 从几百兆的主机查找表里按行预取数据。最后，Linux 的 tracepoint 有相对稳定的社区契约，而 vLLM 内部结构演进极快；pLLM 将宿主引擎（如 vLLM 0.27.1 与 0.32.0）的内部接口差异集中在 `pllm/backends/vllm/compat.py` 这一处隔离层处理（另有少数几处按它导出的版本开关分支），以此向算法作者提供稳定的声明接口。

---

## 三、 两个核心机制：声明代替回调，以及“一张 `int64` 小表”

要把动态的多租户算法装进静态的引擎，pLLM 在底层反复用到两个基本设计。后面的十二篇文章几乎都是这两个设计在不同场景下的展开。

### 1. 用声明与一次虚拟追踪代替运行时 Hook

传统的 PyTorch hook 之所以难以优化，是因为它是一个黑盒运行时回调：不到函数执行的那一微秒，引擎根本不知道这个 hook 要读哪一层、会写哪块内存、里面有没有藏着 `.item()` 同步。

回看开头的 `ESampLite`，pLLM 要求作者把“控制面信息”全部提到函数签名和类属性上：状态的作用域（`RequestState` 还是 `InstanceState`）与优化器写在类属性里；读取什么数据写在参数默认值上（如 `pllm.hidden(0)`、`pllm.module("distiller")`）；产生什么效果写在装饰器上（如 `@pllm.train`、`@pllm.guide_sampling`）。方法体内部则只留纯粹的张量数学，默认第 0 维永远是对齐好的活跃请求行。

因为输入、输出和状态依赖全部在签名上明牌，系统不需要等到 GPU 运行时才去猜测代码的行为。在引擎启动前（或者在纯 CPU 上运行 `pllm.explain` 时），框架只需构造一批只有形状、没有物理内存的 PyTorch `meta` 张量，把用户的方法空跑一遍。

这次毫无硬件成本的虚拟追踪一次性完成了四件事：如果代码里写了 `.item()` 或基于张量值的 Python 分支，因为 `meta` 张量没有数值，追踪会当场报错拦截；同时，框架顺着计算图提取出所有中间张量的生命周期、显存峰值和算子计算量，生成一张带类型的语义图（`SemanticGraph`）。随后，语义图交由约 4,300 行 Rust 编写的规划器（`rust/pllm_core/src`）求解数据依赖、分配时间轴锚点与 CUDA 流、核算显存预算并签发物理执行计划（`PhysicalPlan`），最终交给约 4 万行 Python 的运行时与 PTTS 状态池执行。

### 2. CPU 每步算好一张 `int64` 小表，GPU 按表寻址

有了执行计划，接下来的核心矛盾是：CUDA Graph 要求所有张量地址与形状固定不变，而连续批处理里的请求行号、每个请求当前读写的参数版本槽位、乃至宿主引擎分配的外部张量地址，每一步都在变。

解法是加一层间接寻址（indirection）：把所有每步变化的关系，全部编码成一张地址固定的 `int64` 整数表里的数据。

在 pLLM 中，所有每请求状态都预先分配在基地址固定的分页显存池里；同时，显存里还留有一张基地址同样固定的共享 `int64` 表。`pllm/execution/tensor_executor.py` 中的 `TableLayout` 定义了这张表的四段结构：

```python
# check-docs: skip: excerpt from pllm/execution/tensor_executor.py
@dataclass(frozen=True, slots=True)
class TableLayout:
    """One resource's segments in the shared int64 table, uploaded in one copy per step.

    rows          [6, R]   per row: sampler row, sample index, state (S: padding sink), version, packed row, active
    sources       [F, S+1] source page slot per field and state (-1: none)
    destinations  [F, S]   destination page slot per field and state (-1: do not write)
    row inputs    [K, 3]   address and strides of each token or sampler tensor
    """
```

每一步开始前，CPU 在锁页内存（pinned memory）里填好这四段整数，通过一次仅耗时几微秒的异步拷贝推上 GPU；录在 CUDA Graph 里的内核则只认这张表的固定地址：

```text
CPU (每步在锁页内存中填好整数)                         GPU (地址固定的 CUDA Graph 回放)
┌──────────────────────────────────────────┐          ┌──────────────────────────────────────────┐
│ Shared int64 Table (固定设备地址)         │  1次异步  │ 录好的内核只认表的固定地址：              │
│ ──────────────────────────────────────── │  H2D拷贝  │                                          │
│ 1. rows [6, R]:                          │ ───────► │ • 查 rows + sources:                     │
│    行 r 属于哪个请求状态? 在扁平 batch    │          │   去状态池的哪个物理页槽读当前版本参数？    │
│    的哪一行? 是否有效(active)? (空行指到S)│          │   (填充空行自动路由到全 0 的哑状态槽 S)     │
│ 2. sources [F, S+1]:                     │          │ • 查 destinations:                       │
│    每个请求的第 f 个字段，这步读哪一页？    │          │   算完的新权重与动量写进哪个新分配的页槽？ │
│ 3. destinations [F, S]:                  │          │ • 查 row inputs:                         │
│    写状态的方法，这步把新版本写到哪一页？   │          │   把 64 位整数转回设备指针 tl.pointer_type │
│ 4. row inputs [K, 3]:                    │          │   去读宿主引擎每步动态分配的外部张量        │
│    外部张量这步的裸地址 data_ptr 与步长     │          │                                          │
└──────────────────────────────────────────┘          └──────────────────────────────────────────┘
```

这四段整数各自解决了一个动态性难题。当线上请求数少于图录制的最大容量 $R$ 时，多余的空行在 `rows` 段里被指向第 $S$ 号“哑状态（padding sink）”，其源槽号为 `-1`（读出全 0）、`active` 掩码为 `0`，而 `destinations` 段压根没有第 $S$ 列，从而在不重录图的前提下保证了空行陪跑绝不污染真实梯度。当主流前向与旁路训练并发运行时，`sources` 段让主流读取上一步已发布的旧页槽，`destinations` 段让旁路流写入新分配的空闲页槽，两边在同一个池子的不同页槽上无锁并行。甚至当采样器或宿主引擎传来的外部张量每步更换显存地址时，CPU 只需调 `tensor.data_ptr()` 把它的 64 位裸地址和步长当成普通整数填进 `row inputs` 段，GPU 上的 Triton 内核读出整数后再强转回设备指针解引用，就绕过了图的地址绑定。

这张 `int64` 表作为母题贯穿了整个系列：你会在[《行模型》](04-packed-rows.md)里看到 `rows` 段如何从调度器的三个整数算出，在[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里看到四段表如何支撑整步训练录图，在[《PTTS》](01-ptts.md)里看到 `sources` 与 `destinations` 背部的版本账本，并在[《图内挂载点》](05-in-graph-attach-points.md)和[《参数覆盖层》](06-parameter-overlays.md)里看到表的显存地址本身被用作穿过 `torch.compile` 边界的会话标识。

---

## 四、 一步解码的时间线与真实成本

将声明式任务放到一张时间线上，就能看清后面十二篇文章在单步执行里各自所处的位置。pLLM 将引擎的每一步划分为 8 个有序的时间轴锚点（anchors），并在主流（`main`）与旁路流（`side`）两条车道上调度任务（下图画出其中主要的 6 个）：

```text
时间轴锚点:  request.prepare    model.execute        state.publish    sampler.pre_draw  token.commit      next.prepare
                  │                  │                    │                 │                 │                 │
                  ▼                  ▼                    ▼                 ▼                 ▼                 ▼
第 n 步   main:  ├── [01:算行标签] ──┼── [04:图内改残差] ──┼── [predict] ────┼── [09:采样尾声] ─┤                 │
(主流)           │   (填表上传H2D)   │   [05:参数覆盖层]   │   (读 lag=1 版本)│   (三遍扫描融合)  │                 │
                  │                  │   [06:自定义层循环] │                 │                 │                 │
                  │                  │   [07:自定义注意力] │                 │                 │                 │
                  │                  │                    │                 │                 │                 │
           side:  ├── [10:主机查表] ──┼────────────────────┘                 │                 ├── [02:整步训练] ──┼──────┐
(旁路流)         │   (1/4 SM 限流， │                                      │                 │   (loss反向+AdamW │      │
                  │    与前向重叠)    │                                      │                 │    写新页槽，与   │      │
                  │                                                         │                 │    n+1步前向重叠) │      │
第 n+1 步 main:  │                                                         │                 ├── [准备 n+1] ───┼── [模型前向 n+1] ──►
                                                                                                                ▲      │
                                                                                               [03:PTTS 在边界发布版本]─┘
                                                                                   [11:自动放置求解锚点窗口与跨步 lag=1]
```

在一步开头的 `request.prepare`，CPU 根据输入批次算出每行标签并上传 `int64` 表（[第 1 篇](04-packed-rows.md)）；若有只读的主机内存大表查询，旁路流此时立即启动一个仅占用四分之一 SM 的持久化内核，与主流的模型前向重叠拉取数据（[第 10 篇](12-bounded-zero-copy-gather.md)）。进入 `model.execute` 后，模型顺着 CUDA Graph 执行，图内挂载点在此改写残差（[第 4 篇](05-in-graph-attach-points.md)）、叠加参数覆盖层（[第 5 篇](06-parameter-overlays.md)）、按掩码跳过深层（[第 6 篇](10-custom-layer-loops.md)）或切换注意力与 KV 布局（[第 7 篇](08-programmable-attention-kv.md)、[第 8 篇](09-reusing-upstream-kernels.md)）。前向到达 `state.publish` 与 `sampler.pre_draw` 后，引导方法算出预测并融合进采样器原有的词表扫描尾声中（[第 9 篇](07-sampler-epilogue.md)）。最后在 `token.commit` 采样落定，在线训练任务在旁路流上启动（[第 2 篇](03-whole-algorithm-in-cuda-graphs.md)），一边算反向求导与 `AdamW`，主流一边进入第 $n+1$ 步。

为了让第 $n$ 步的旁路训练能与第 $n+1$ 步的主流前向真正并行，系统默认采用 `publish_lag=1`（[第 3 篇](01-ptts.md)、[第 11 篇](11-automatic-placement-overlap.md)）：第 $n$ 步训练产生的新权重版本，默认在第 $n+2$ 步对其他读取方法可见，第 $n+1$ 步的前向直接读取第 $n-1$ 步已发布的不可变版本（训练任务自身则始终在最新参数上连续迭代）。如果算法在数学上要求零滞后，也可以显式指定 `publish_lag=0`，用串行等待换取严格步序。

### 成本与边界

这套机制的代价体现在两个方面：固定调度开销与表达边界。

首先是固定开销。每步在主机端更新页表账本、上传几千字节的 `int64` 表、记录流事件以及运行收集/分发内核，大约需要几十到几百微秒。这笔开销与模型层数无关，因此在小模型上显著，在大模型上则被分母摊薄：
* 在 **Qwen2.5-7B-Instruct** 上（单步耗时 21.7 ms，原生吞吐 5,894 tok/s），运行 ESamp 仅在线训练（$\beta=0$）时吞吐为 **5,738 tok/s（原生的 97.4%）**，同时开启训练与引导采样（$\beta=0.5$）时为 **5,678 tok/s（原生的 96.3%）**。在多卡 A100-SXM4-80GB 环境下测试 7B ESamp（$\beta=0$，rank 4，128 并发序列 × 256 token），TP=1、2、4 下的吞吐分别为原生的 95.84%、95.63% 和 97.60%。
* 在 **Qwen2.5-0.5B-Instruct** 上（单步耗时仅 3.0 ms，原生吞吐 42,355 tok/s），几百微秒的底噪占比被放大：vLLM 0.32.0 下运行 ESamp $\beta=0$ 的吞吐为 **38,936 tok/s（原生的 91.9%）**；在早先基于 vLLM 0.27.1 的 20 轮中位数测量中，$\beta=0$ 为原生的 92.45%，加上采样引导（$\beta=0.5$）后为 **83.38%**。

简言之，这套架构在 0.5B 这类极轻的小模型上仍有 8%–17% 的可见开销；到了 7B 规模，相对开销收敛到 3%–4% 左右。

其次是表达边界。为了守住多租户隔离、CUDA Graph 静态形状与宿主引擎的记账不变量，有一类算法在 pLLM 中是明确写不了的（完整列表见[《设计取舍》](../design-tradeoffs.md)）：不能编写跨行归约（如 `h - h.mean(dim=0)`）或批内竞争路由（Expert-Choice MoD），因为那会让一个租户的输出取决于同批拼在一起的其他租户；不能实现像 SnapKV、H2O 那样按运行时注意力分数释放序列中间某个物理 KV 块的算法，因为 vLLM 底层的块表只支持末尾追加与连续位置计数；也不能接管外层解码循环实现投机解码（Speculative Decoding）的树分支回滚，或从头预训练模型。

遇到不支持的写法或显存预算不足时，pLLM 的原则是闭合失败（fail-closed）：在规划期直接报错并说明物理原因（默认 `cuda_graphs="auto"` 下，少数无法进图的写法会以 eager 运行并在 `pllm.explain` 的 `forward` 行注明原因；若设 `cuda_graphs=True` 则直接报错拒绝），绝不静默把放不下的显存表溢写到 CPU 内存。

---

## 五、 系列导览

接下来的十二篇文章按照从外围到模型内部、从单算子机制到全局调度验证的顺序排列。每一篇都围绕一个具体的系统问题展开：

1. **[《行模型：在连续批处理里找回“每个请求”》](04-packed-rows.md)**：面对拍扁的 `[num_tokens, hidden]` 张量，为什么不必去解析 GPU 上复杂的注意力元数据结构体，只靠调度器入口的三个整数与一条位置水位线就能推导出所有行的身份与抢占重算状态（附 vLLM 注意力元数据实测解剖笔记）。
2. **[《Python 只在录制时活着：固定地址、一张表、零同步》](03-whole-algorithm-in-cuda-graphs.md)**：当 Python 钩子在 CUDA Graph 回放时失效，如何从 1 个请求出发，一步步引入固定状态池、哑状态槽（sink）与设备端指针解引用，把整步反向求导与 `AdamW` 录进图里；以及为什么给这套训练体套上 `torch.compile` 反而会因为函数式化克隆变慢。
3. **[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)**：图地址固定、双流读写并发、在途导入更新、抢占回滚——这四个问题为什么本质上都在呼唤同一个不可变版本与字段级写时复制（COW）原语。
4. **[《图内挂载点：在 CUDA graph 里改残差》](05-in-graph-attach-points.md)**：当残差加法被上游吞进下一层的 `fused_add_rms_norm` 时，如何骑在归一化层上挂载代码；以及如何通过剖析器找出恒等改写里多出的 61,440 个内核，用 `make_fx` 剪掉末尾加法交还原生融合算子，让 0.5B 上恒等改写的吞吐回到原生的 99.5%。
5. **[《参数覆盖层：顺着 PyTorch 的抽象层往下走》](06-parameter-overlays.md)**：不写专用 CUDA 内核，如何将 `vmap`、分块 `bmm`、`custom_op`、`grad`、`compile` 与 `make_fx` 组合起来，让论文里只支持 `batch_size=1` 的在线快权重更新跑在多租户与多卡张量并行下，并用全局/分片双图同构证明多卡更新零通信。
6. **[《自定义层循环与状态传播》](10-custom-layer-loops.md)**：当某个 token 在第 15 层决定跳过后续四层，如何用状态传播（借用原生注意力模块前半截补写 KV）防止后续 token 注意力踩空，并用 CUDA 条件节点与 2 的幂次桶装进图里；以及为什么在短上下文 decode 中跳过 23% 的 token 并不会变快。
7. **[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)**：算清“少读一半 KV”与“少占一半显存”的区别，剖析 vLLM v1 的三条记账不变量，解释为什么按分动态驱逐中间块在不拆引擎的前提下不可行，以及如何用零行新内存管理代码将窗口、保留前缀窗口（R-SWA）与跨层共享伪装成上游原生层。
8. **[《复用上游：kernel、管理器与原语》](09-reusing-upstream-kernels.md)**：为什么把滑动窗口切换到标准的 FlashAttention-2 `window_size` 接口，会在 0.5B 上让单次读取慢上 5 倍、整机吞吐掉到 78.8%？从 GQA 头打包与 FlashInfer 规划管线出发，讨论为什么扩展层真正该复用的是上游的执行路径而非孤立函数。
9. **[《采样尾声融合》](07-sampler-epilogue.md)**：面对 1.09 GB 的 `lm_head` 权重与近 80 MB 的词表 logits，如何把偏置加法与后过滤引导塞进原有的三遍遍历里，只为过滤后平均剩下的 4.83 个幸存词拉取权重行，并用 `uint64` 的 IEEE 754 位翻转实现单指令排序决胜。
10. **[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)**：当几百兆的查找表放在主机内存、每步需跨 PCIe 拉取 38.9 MB 数据时，为什么发射 9,600 个线程块的后台 gather 内核会占满全部 SM 把主干矩阵乘堵死，而把网格限制到四分之一 SM（32 个块）反而能打满 23.7 GB/s 总线并藏住九成以上的传输耗时。
11. **[《自动放置与重叠》](11-automatic-placement-overlap.md)**：为什么多面体编译器模型对推理步调度是错位的重武器，如何将单步锚点与跨步常数滞后化简为一致依赖图上的 40 行 Bellman-Ford 最长路松弛，以及为什么在满载的 GPU 上只有资源互补或错开多卡通信障才能拿到真正的重叠收益。
12. **[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)**：回到 eBPF 验证器的起点——当形式化证明对任意 PyTorch 代码不可行时，如何把连续批处理的调度本身表示为静态三元组数据，用加盐差分测试（`salt = 1_000_000`）在装载前验证多租户隔离与抢占不变性。
