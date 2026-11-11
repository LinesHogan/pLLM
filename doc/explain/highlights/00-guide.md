# 给 vLLM 装一个“eBPF”：在不拆引擎的前提下运行推理期算法

> 2026-11-10 · pLLM 提交 `be3569b0` · 测量环境：除文中另注外为单张 RTX 4090（24 GiB）、vLLM 0.32.0、Qwen2.5-7B-Instruct，8 条 prompt × 16 个样本、生成 256 token。
> 这是一篇有日期的文章，不随代码更新；当前的用法以[用户指南](../../README.md#用户指南)为准，当前的实测数字以[性能与显存](../performance.md)为准。

在研究脚本里验证一个推理期算法，通常只需要十几行 PyTorch。

以在线蒸馏引导采样（ESamp）为例：在生成的每一步，都为当前请求在线训练一个极小的秩-4（rank-4）双层网络，用第 0 层的隐藏状态去拟合最后一层的隐藏状态；到了采样前，再把这个小网络的浅层预测从词表 logits 里对减一部分，压低那些“浅层就能猜到”的 token。

在 Hugging Face 的 `generate` 循环里，挂两个 `register_forward_hook`、每步调一次 `loss.backward()` 和 `optimizer.step()`，十几行代码就能跑通。在 pLLM 中，它的声明同样只有十几行：

```python
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

但如果你不用扩展层，而是直接把那十几行原生 PyTorch hook 塞进生产级的 vLLM 引擎，这十几行代码会立刻撞上五道系统边界：

第一，连续批处理（Continuous Batching）抹掉了请求的边界。为了最大化吞吐，vLLM 把几十个正在逐字解码（decode）的请求和正在分块预填充（chunked prefill）的请求首尾相接，拍扁成一张紧致的二维张量 `[num_tokens, hidden_size]`。你的 hook 拿到这张张量时，既不知道哪一行属于哪个请求，也不知道末尾哪几行是凑尺寸的 padding；更何况每一步都有老请求退出、新请求插队、甚至有请求因显存不足被临时抢占（Preempt）踢出并在稍后重算，行号每步都在洗牌。

第二，CUDA Graph 让运行期的 Python 回调失效。在单张 NVIDIA RTX 4090（24 GiB）上，跑 128 个并发序列（8 条 prompt $\times$ `n=16`，生成 256 个 token，vLLM 0.32.0，开启 FlashInfer 采样器——除特别注明外，这也是本系列共享的基准测试设置，默认模型是 `Qwen2.5-7B-Instruct`）时，`Qwen2.5-0.5B-Instruct`（BF16）跑完一步 decode 只需 **3.0 ms**（42,355 tok/s）。如果关掉 CUDA Graph 退回由 Python 逐个发射算子的 eager 模式，CPU 发射开销会让单步耗时翻倍，吞吐直接腰斩到 **49%**（20,926 tok/s）。为了省掉这笔 CPU 发射税，vLLM 在预热时把解码步录制成静态的 CUDA Graph，线上回放时 Python 解释器根本不参与层的执行，且图内所有算子只认录制那一刻绑定的固定显存指针。

第三，每请求的可变状态无处安放。128 个并发请求各自拥有独立的权重与 `AdamW` 一阶、二阶动量，既要随请求进出动态分配回收，又要在 CUDA Graph 里保持基地址固定；如果把在线训练放到旁路 CUDA stream 上与下一步前向并发执行，原地修改还会导致主流读到写了一半的撕裂参数。

第四，额外的显存与流时间是零和的。显卡的 GDDR6X 显存要先扣除模型权重、CUDA Graph 静态池和扩展状态，剩下的才分给决定并发上限的 KV Cache；而在执行流上，哪怕整步里只有一次 `.item()` 或张量布尔判断触发主机同步，CPU 领先 GPU 半步的异步发射流水线就会被当场排空。

第五，模型内部结构已被编译器封死。如果你想改写残差流（如 Hyper-Connections）、给投影层叠加每请求快权重（如 In-Place TTT）、动态跳过部分深层（Early Exit）或更改 KV 缓存保留策略，你会发现模型前向已被 `torch.compile` 编成计算图，残差加法被融进了下一层的 `fused_add_rms_norm` 内核，注意力则与底层的块表管理器咬合在一起。

---

## 一、 eBPF 之前的 Linux 内核困境与 eBPF 契约

退后一步看，这五道障碍里没有一道是机器学习问题。它们全都在问同一个经典系统工程问题：**面对一个为了压榨硬件性能而高度静态化、同时服务上百个租户的底层引擎，怎样允许外部开发者在关键路径上安全、高速地运行一段有状态的自定义程序，而不必每次都把引擎拆开重改？**

在 eBPF 出现之前，Linux 内核在网络包处理与系统追踪上走过完全相同的死胡同。想在内核里做超出简单过滤的自定义处理，开发者只有两个糟糕的选择：要么把数据包拷回用户态处理——写起来自由安全，但上下文切换与拷贝开销会让吞吐暴跌；要么自己写一个 Linux 内核模块（Kernel Module）直接挂进内核函数指针——速度够快，但一个越界指针或并发竞态就能让整台机器 Kernel Panic，且内核内部结构一升级，模块就得推倒重修。

今天的大模型推理生态正处在这个节点上。在 Hugging Face、TransformerLens 或 nnsight 里写 hook，就像在用户态写脚本，灵活但无法承受高并发生产流量；而直接 `git clone` 一份 vLLM 源码去改 `GPUModelRunner` 和底层算子，就是在写内核模块——每个新算法维护一个私有 fork，抢占回滚、多租户隔离与显存碎片的雷全得自己排，上游一升级就面临大面积冲突。

Linux 走出这个二选一困境的方案是 **eBPF**。eBPF 没有给用户开放任意捅破内核的裸指针权限，而是立下了一套由验证器、JIT 编译器、受管 Map、标准挂载点与观测工具组成的契约。pLLM 的架构正是这套契约在 GPU 推理引擎中的投影：

| Linux eBPF 组件 | pLLM 中的对应机制 | 解决的核心约束 |
| :--- | :--- | :--- |
| **Verifier**（装载前验证） | **声明反射 + `meta` 追踪 + `pllm.testing.check`** | 静态拦截 `.item()` 同步与非法时序依赖；在 CPU 上用合成调度做加盐差分测试，验证请求隔离与抢占重算一致性 |
| **JIT**（原生代码执行） | **`StepCapture` 整步录图 + Inductor 编译** | 将用户的前向、反向求导与优化器更新整包录进 CUDA Graph；图内挂载点单独编译并把末尾加法归还给上游融合算子 |
| **BPF Maps**（受管状态存储） | **PTTS（分页张量状态迁移系统）** | 在固定基地址的物理槽池上提供不可变版本号、字段级写时复制（COW）与基于 CUDA Event 的异步回收 |
| **Attach Points**（挂载点） | **8 个步内时间轴锚点 + 模型内部挂载点** | 宏观上划分步内锚点与主/旁路流；微观上挂载残差流、线性投影、注意力 KV 与采样器尾声 |
| **Observability**（可观测性） | **`pllm.explain` + 运行期诊断** | 不接触 GPU 即可在 CPU 上算出流调度计划、状态版本份数与精确到字节的显存预算账单 |

这里顺带说明一句命名：PTTS 是 Paged Tensor Transition System 的缩写，它开头的 P 指的是 **Paged**（即像操作系统虚拟内存与 vLLM 的 PagedAttention 那样通过页表做分页映射），与 pLLM 名字里代表 programmable 的 p 没有任何关系。

当然，工程类比不能代替诚实的边界。与 Linux eBPF 相比，GPU 推理扩展层有三处本质差异：
1. **验证的性质不同**：eBPF verifier 面对的是受限字节码，能在装载前证明一组安全性质；而 PyTorch 算子库过于庞大，无法用静态分析证明任意张量运算“绝不在第 0 维上混行”。因此 pLLM 采用“静态形状/同步拦截 + 确定性三元组调度下的差分蜕变测试”来守住隔离底线。
2. **计算体量不同**：eBPF 程序通常短小；而推理期负载可以在旁路流上跑完整的反向传播与 `AdamW`，或从数百兆的主机查找表里跨 PCIe 预取行。
3. **宿主 ABI 的演化速度不同**：vLLM 内部结构演进极快，pLLM 选择将 vLLM 0.27.1 与 0.32.0 的内部差异集中关在 `pllm/backends/vllm/compat.py` 这一个隔离层内（另有少数几处按它导出的版本开关分支），向上提供稳定的声明接口。

一步 decode 怎样切成有序的锚点、工作怎样分到主流与旁路流、`publish_lag` 为什么让训练能与下一步前向重叠，见[核心概念](../core-concepts.md#3-时间线锚点阶段窗口与主旁路流)的第 3、4 节；固定 `int64` 路由表见[第 2 篇](02-tensor-colors.md)，这套机制的固定开销见[性能与显存](../performance.md)。

---

## 二、 系列导读：十个独立的系统问题

本系列的后续十篇文章，每一篇都围绕一个在改造 vLLM 或编写高性能推理扩展时会撞上的具体技术问题展开。它们各自独立成篇，你可以按顺序通读，也可以直接挑最关心的痛点切入：

1. **[《你其实不需要读 `attn_metadata`：用三个整数算清 vLLM 的打包行布局》](01-attn-metadata.md)**
   面对拍扁的 `[num_tokens, hidden]` 张量，为什么不必去碰 GPU 上复杂的 `attn_metadata`，仅凭 `prepare_attn` 入口处的三个 CPU 标量整数和一条水位线就能还原全部行身份并过滤抢占重算？（文末附 vLLM `attn_metadata` 五步实测解剖图鉴。）
2. **[《What Color is Your Tensor? —— 当 Python 只在录制时活着，怎么在 CUDA Graph 里跑动态算法》](02-tensor-colors.md)**
   当图捕获焊死了显存地址与形状，如何通过固定地址的 `int64` 表、吸收空位的“哑状态”以及在 Triton 里把 64 位整数强转回设备指针，将整步反向传播与 `AdamW` 装进静态图？以及为什么给原地写池子的代码套 `torch.compile` 反而会因克隆而变慢？
3. **[《给 GPU 显存加上 MVCC：当每个并发请求都有一份会变的权重》](03-gpu-mvcc.md)**
   图地址绑定、双流读写撕裂、在途参数导入与 KV 抢占回滚，为什么本质上都是因为“原地修改抹掉了时间维度”？看数据库 MVCC 与内核 RCU 如何在固定显存池上通过字段级写时复制与 CUDA Event 栅栏落地。
4. **[《How to Lose 61,440 Kernels in 10 Lines：在 `torch.compile` 图里改残差与训权重》](04-compiled-model-hooks.md)**
   一次从 65.8% 到 99.5% 的性能破案：为什么一行 `return x + y` 会在生成中多发 61,440 个内核？如何用 10 行 `make_fx` 代码剪掉末尾加法还给 `fused_add_rms_norm`，并用 `grad → vmap → compile` 与双图同构证明实现多卡 TP 零通信的前向内训练？
5. **[《Falsehoods Programmers Believe About Early Exit：为什么在 vLLM 里跳过 23% 的 Token 一点也不快》](05-early-exit.md)**
   逐条推翻关于动态跳层的五个常见错觉：为什么跳层不能在 KV Cache 里留白？如何一行新内核不写、借用原生 Attention 前半截补写 KV？以及在受限于 466 MB/层权重读取带宽的短上下文 decode 中，为什么部分跳层省不下时间？
6. **[《稀疏注意力在 vLLM 里省不下显存（以及为什么调 `window_size` 反而慢了 5 倍）》](06-sparse-attention-kv.md)**
   为什么“每步少读一半 KV”省不下一兆物理显存？剖析 vLLM 块管理器的三条记账不变量（解释为何 SnapKV 式动态挖洞走不通），展示如何用零行分配代码伪装成原生静态布局，并算清 GQA 7:1 打包带宽账，揭示为何裸调 FA2 窗口函数会慢 5.5 倍而必须接入 FlashInfer 规划管线。
7. **[《Free FLOPs and Where to Find Them：利用 Memory-Bound 解码尾声顺手做完采样引导》](07-free-flops.md)**
   在被 1.09 GB `lm_head` 与 80 MB 词表 logits 锁死带宽的解码末尾，去哪里找免费算力？看看如何把偏置折进必经拷贝、只为 `top-p` 幸存的平均 4.83 个词取权重行、用 5 行 IEEE 754 位翻转把双条件排序压进单个 `uint64`，以及为什么暂存容量选 4,096 比 256 更快。
8. **[《为什么 32 个线程块比 9,600 个更快？一次把 GPU 前向饿死的跨 PCIe 查表》](08-bounded-gather.md)**
   当 594 MiB 的查找表放在主机内存、每步跨 PCIe 零拷贝拉取 38.9 MB 时，为什么发射 9,600 个线程块会把 128 个 SM 的驻留槽位全部占死、让主干矩阵乘寸步难行，而把网格砍到 SM 数的四分之一（32 个块）反而能藏住 93% 以上的传输耗时？
9. **[《You Probably Don't Need a Compiler IR：用 40 行最长路自动调度 CUDA Stream》](09-stream-scheduling.md)**
   为什么我们引进了 45 万行代码的 ISL 多面体库，最后又亲手删掉它、换成了 8 个时间锚点上的 40 行 Bellman-Ford 最长路？兼谈 GPU 多流重叠的物理真相：为什么单卡早发训练会与采样器撞车，而在多卡 TP 下把训练错峰放到 All-Reduce 之外，能让吞吐从 87.07% 跳到 97.60%？
10. **[《`h - h.mean(dim=0)` 不会报错，但会让张三看见李四的提示词：连续批处理的隔离测试》](10-isolation-testing.md)**
    当静态分析无法证明任意 PyTorch 代码没有跨行串味时，如何把连续批处理的调度表示成几行静态的整数三元组，并在纯 CPU 上给邻居请求的随机种子加一百万盐（`salt = 1_000_000`），在不跑大模型、不写模拟器的情况下几秒内揪出多租户污染？
