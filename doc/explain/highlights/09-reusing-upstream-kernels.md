# 复用上游：kernel、管理器与原语

每一个构建在大型底层系统之上的扩展框架——无论是 Linux 内核上的模块、PostgreSQL 的插件，还是大模型推理引擎上的可编程扩展层——迟早都会迎来同一次灵魂拷问：**当宿主系统的某个组件不够顺手时，我们要不要自己重写一个？**

诱惑总是极其具体的，而且每一次都披着“为了性能”或“为了代码干净”的正当外衣。上游的注意力算子不返回我们需要的 `log-sum-exp`（LSE）统计量？自己写一个 Triton kernel 吧，也就两百行；上游的分配器不认识我们的状态布局？自己写一套页表吧；上游的编译器融合改动了最后一个比特的舍入？干脆接管它的整张计算图吧。

单独看每一次决定，自己动手总是更快、更痛快，跑出来的微基准数字往往也更漂亮。但当你把十几个这样的决定加在一起，某天早上打开代码库，你会突然惊出一身冷汗：你本来想做一个轻量级的扩展层（就像推理引擎里的 eBPF），结果却在宿主引擎旁边平行地造出了半个山寨版的 vLLM。每一个手写的 kernel 都需要你单独维护数值基准（oracle），每一条绕开上游的平行路径都会在下一次宿主引擎升级时摔得粉身碎骨。

我们在代码库里做过一次彻底的清创手术。在那次手术里，一个看似最“政治正确”的改动差点把我们的吞吐砸穿；也正是那次翻车，逼着我弄明白了一件看似平淡、实则决定了扩展层生死的系统工程法则：**在底层系统里，“调用上游的一个函数”往往只是伪复用；真正的复用，是接入上游自己的那条执行路径（Execution Path）。**

我们先从那次翻车说起。

> **实验环境说明**：如无特别说明，本文引用的 7B 生产实测数据均在单卡 RTX 4090（24 GiB）、vLLM 0.32.0（PyTorch 2.13，CUDA Graph 开启）、Qwen2.5-7B-Instruct（BF16，28 个 Query 头 / 4 个 KV 头，头维 128）下测得，预热 1 轮后取 3 轮中位数。在剖析单层注意力内核的微秒级回归时，我们使用同卡上的 Qwen2.5-0.5B-Instruct（BF16，14 个 Query 头 / 2 个 KV 头，头维 64，vLLM 0.27.1，8 个 prompt × n=16 共 128 条并发序列，生成 256 token，`max_model_len=2048`）作为“显微镜”——0.5B 模型单步耗时仅约 3 ms（7B 约 21.7 ms），没有庞大的 MLP 权重读取打掩护，任何内核级的效率退化都会在整机吞吐上暴露无遗。

---

## 一、 一个“更标准”的改动，为什么让读取慢了五倍？

在[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)里，我们允许用户声明某些层只保留一个固定长度的滑动窗口 KV（例如偶数层只看最近 256 个 token，奇数层看全量历史，即 `LocalGlobal`）。

最初实现滑动窗口读取时，我们直接拿 vLLM 的原始 Triton kernel 跑一个不分段的单段读，再配一个自己写的收尾 kernel 补出 LSE，跑得飞快。但后来做代码审计时，我看着这套东西觉得很刺眼：到目前为止，到处造轮子并不符合 pLLM 的设计思想。底层依赖的 FlashAttention-2（FA2）明明原生就有一个 `window_size` 参数，专门用来做滑动窗口注意力；既然上游已经有现成的标准接口，我们为什么还要自己维护一套 Triton 代码？

于是，我们做了一个看起来不能更正确的决定：删掉这条 Triton 窗口读取路径和我们自己写的输出归一化 kernel，直接调用上游 FA2 的 `flash_attn_varlen_func`，把 `window_size=[window - 1, 0]` 传进去。

然后我们跑了一遍基准测试，结果让人大跌眼镜。

在一个仅仅把 24 层里的 2 层改成窗口、且窗口大小只有 4 的极简测试里，原本应该“不慢于全量注意力”的窗口配置，整机吞吐竟然掉到了全量注意力的 **89.3%**（vLLM 0.27.1）和 **90.7%**（vLLM 0.32.0）。换到真正的局部/全局交替布局 `LocalGlobal`（偶数层窗口 256）上，整机吞吐更是从原生的 39,120 tok/s 直接摔到了 **30,846 tok/s（78.8%）**，平白无故丢掉了超过两成的性能！

慢着——滑动窗口明明让 GPU **少读**了历史 KV，怎么会越少读越慢？

我把这一层单独抽出来，在 128 行并发、序列长 256 的条件下跑了一组 1,000 次 CUDA Graph 回放的微基准。终端上跳出来的两个数字，把问题照得清清楚楚：
* 让 FA2 读全部 256 个历史 token，一次调用只要 **`16.56 µs`**；
* 给同一个 FA2 函数传进 `window_size=(3, 0)`，让它只读最近 4 个 token，耗时却暴涨到了 **`91.47 µs`**！

少读了 98% 的历史数据，执行时间反而慢了 **5.5 倍**。在 `LocalGlobal` 的稳态运行中（3,024 次窗口读平均），单次 FA2 窗口读同样高达 **`86.43 µs`**。

为什么会这样？要看穿这个反直觉的现象，我们得先算一笔硬件带宽账。

在自回归解码（decode）阶段，每条序列每次只进 1 个新 token。这时候算力（FLOPs）严重过剩，注意力层的唯一瓶颈是**显存带宽（Memory Bandwidth）**——也就是把历史 K 和 V 从 HBM 显存搬进芯片寄存器的速度。

而现代大模型普遍采用 **GQA（Grouped-Query Attention，分组查询注意力）**。拿 Qwen2.5 系列来说，无论是 0.5B（14 个 Query 头、2 个 KV 头）还是 7B（28 个 Query 头、4 个 KV 头），Query 头与 KV 头的比例都是 **7 : 1**。也就是说，每 7 个 Query 头共享完全相同的一组 K 和 V。

这意味着在计算带宽账单时，有效载荷必须按 **KV 头数** 而不是 Query 头数来算。以 0.5B 模型、128 行并发、历史长度 256、头维 64、BF16（2 字节）为例，这一步需要搬运的 K/V 有效载荷是：

$$128 \times 256 \times 2\text{ (K与V)} \times 2\text{ (KV头)} \times 64 \times 2\text{ Bytes} = 16\text{ MiB}$$

在 RTX 4090 标称 `1,008 GB/s` 的显存带宽下，把这 16 MiB 数据完整搬运一遍的物理极限时间（Roofline）大约是 **`16.6 µs`**（历史长度为 128 时则是 8 MiB、约 `8.3 µs`）。你看，FA2 全量读取测出来的 `16.56 µs`，几乎严丝合缝地贴在这条硬件物理极限线上！

它是怎么做到的？秘诀在于 **GQA 打包（Packed GQA）**。一个优秀的 decode kernel 会把共享同一组 KV 的 7 个 Query 头打包进同一个线程块（CTA）里：从显存里把 K 和 V 读进片上内存一次，7 个 Query 头轮流或并行跟它做点积，榨干每一个字节的价值。

但当我们打开 profiler 检查 FA2 走 `window_size` 窗口路径时的内核发射参数，真相大白了：

```text
1. Packed GQA Decode (Grid organized by 2 KV Heads — 1x KV traffic):
   ┌───────────────────────────┐
   │  KV Head 0 Pages (in HBM) │ ──(Read Once)──► [ CTA: Q0, Q1, Q2, Q3, Q4, Q5, Q6 ]
   └───────────────────────────┘

2. Unpacked Window Decode (Grid organized by 14 Query Heads — up to 7x KV requests):
   ┌───────────────────────────┐    ┌──► [ CTA 0: Q0 ]
   │                           │────┼──► [ CTA 1: Q1 ]
   │  KV Head 0 Pages (in HBM) │────┼──►   ...          (7 independent CTAs fetching
   │                           │────┴──► [ CTA 6: Q6 ]   the exact same K/V pages!)
   └───────────────────────────┘
```

在我们测试的这个形状下，一旦开启滑动窗口，FA2 的 decode 路径就关掉了 GQA 打包！它的线程网格（grid）从按 **2 个 KV 头** 组织，退化成了按 **14 个 Query 头** 组织。原本只需要读 1 次的 K/V 页，现在被 7 个各自为政的线程块分别去请求一遍（至于有多少请求被 L2 Cache 救回来、有多少真正打到了显存，全看运气和调度时序），再加上分块与窗口边界判定的额外开销，直接把一个 `16 µs` 的内核拖成了 `86 µs` 的泥潭。

---

## 二、 第三条路：不要把上游当函数库，要走上游自己的路

面对这个 78.8% 的惨淡数字，摆在我们面前的有两条显而易见的路。

第一条路是认怂退回去：把原来那条 Triton 单段读（vLLM 的原始 Triton kernel 加我们自己写的收尾 kernel）捡回来。在完全相同的基准下，它单次读取只要 **`16.23 µs`**（本体 `14.17 µs` + 收尾 `2.06 µs`），整机吞吐高达 **`98.4%`**（38,654 对 39,281 tok/s）。性能无可挑剔，但代价是我们得永远背着那个自己写的收尾 kernel，和一条绕开上游注意力后端、直接去调它内部 kernel 的路。

第二条路是换一个上游算子库。我们知道 vLLM 里除了 FA2，还集成了 **FlashInfer**。在 Ampere 和 Ada 架构（SM80/SM89，如 A100 和 RTX 4090）上，FlashInfer 的 tensor-core decode 内核使用 `16×128` 的 Q/KV 分块和 4 个 warp，即使在开启滑动窗口（`window_left`）时，依然牢牢保持着按 KV 头组织的打包网格（在这个测试的图里，grid 是规规矩矩的 `[183, 1, 2]`）。

“那还不简单？”你可能会说，“把调用 `flash_attn_varlen_func` 的那一行，改成调用 FlashInfer 的 decode 函数不就完了？”

如果你真的去试一下，你会发现自己掉进了另一个深坑。

FA2 之所以讨人喜欢，是因为它在接口上更像一个无状态的纯函数：你把 Q、K、V 和块表塞进去，它把结果吐出来。但 FlashInfer 不是一个无状态的函数，它是一个**两阶段的有状态管线**。在每一轮发射 GPU 内核之前，FlashInfer 必须在 CPU 上先执行一次 `plan`（规划）：根据当前批次里每条序列的实际长度，计算负载平衡，把长序列的 KV 切分成若干段（split-KV），并生成对应的页索引与工作区偏移。更棘手的是，为了放进 CUDA Graph 录制与回放，这个 `plan` 所写往的页索引暂存区（page-index staging）和 split-KV 归约缓冲区，**内存地址必须死死固定**，在每次回放前由主机端原地刷新。

看明白这里的悖论了吗？如果我们在 pLLM 里直接去 import FlashInfer 的底层 API，我们就必须自己接管页索引转换、自己算 split-KV 规划、自己分配并管理一套适配 CUDA Graph 的固定地址缓冲区——**为了复用一个上游 kernel，我们竟然得自己重写半个 vLLM 的注意力后端！**

就在这时候，我退后一步，重新打量了一眼宿主引擎：慢着，vLLM 自己平时是怎么跑 FlashInfer 的？

vLLM 的 `vllm.v1.attention.backends.flashinfer` 模块里，早就有一整套千锤百炼的 `FlashInferMetadataBuilder` 和 `FlashInferImpl`！页索引的暂存、split-KV 的规划、CUDA Graph 固定缓冲区的生命周期，乃至 394 MiB 共享工作区（workspace）的分配，vLLM 自己早就替 FlashInfer 安排得明明白白了。

我们之前之所以觉得难，是因为我们把上游当成了一个静态的“算子函数库（Function Library）”，总想越过 vLLM 的执行管线，自己在半路抓一个裸函数来调。如果我们换一种思路——**不直接调 FlashInfer，而是像 vLLM 自己那样去调 FlashInfer**呢？

```text
❌ Function-Level Reuse (Bypassing Host Pipeline):
   pLLM Hook ──(Manual plan? Manual graph buffers?)──► Raw FlashInfer / FA2 Kernel

✅ Path-Level Reuse (Riding Host's Native Pipeline):
                                ┌──► FlashInferMetadataBuilder ──► Planned Wrapper (Stored Window)
   Native Page Mapping ─────────┤
   (WindowMetadataBuilder)      └──► FlashAttentionMetadataBuilder ──► FA2 Metadata (Alt Reads)
```

看上面这张图。在 `pllm/backends/vllm/flashinfer_attention.py` 里，我们没有写任何一行规划或缓冲区管理代码，而是直接继承了 vLLM 原生的 Builder：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/flashinfer_attention.py
@dataclass
class WindowMetadata(FlashInferMetadata):
    """Native FlashInfer execution plus FA2 metadata for authored alternative reads."""

    flash: FlashAttentionMetadata


class WindowMetadataBuilder(FlashInferMetadataBuilder):
    """Let both upstream builders consume the same native page mappings."""

    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert vllm_config.speculative_config is None
        self.flash = FlashAttentionMetadataBuilder(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> WindowMetadata:
        infer = super().build(common_prefix_len, common_attn_metadata, fast_build)
        flash = self.flash.build(common_prefix_len, common_attn_metadata, fast_build)
        return WindowMetadata(**vars(infer), flash=flash)
```

这段不到二十行的代码，把“搭顺风车”做到了极致。当一个层被声明为滑动窗口存储（且运行在 Ampere/Ada 架构上）时，`install_window` 会在模型初始化阶段把这一层的后端悄悄替换成 `WindowBackend`，并换上原生的 `FlashInferImpl`。到了每一步推理开始前，vLLM 的模型 runner 照常调用 `WindowMetadataBuilder.build`，把同一份原生页映射同时喂给两个上游 Builder：
1. 主干的 `FlashInferMetadataBuilder` 负责生成已经做好 split 规划、绑好 CUDA Graph 缓冲区的 `infer` 元数据，专门服务于窗口层的高速读取；
2. 旁边的 `FlashAttentionMetadataBuilder` 顺手生成一份 `flash` 元数据备用——万一用户在自定义注意力代码里心血来潮，想临时换一个不同的窗口大小或裁剪头数来做二次读取，随时可以回退到 FA2。

而真正执行读取的 `window_attention`，只剩下了对上游预规划包装器（wrapper）的直接调用，外加最后一行小小的数学换底：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/flashinfer_attention.py
def window_attention(
    metadata: WindowMetadata, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # ……
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    count = metadata.num_decode_tokens
    if metadata.decode is not None:
        assert isinstance(metadata.decode, FIDecode)
        metadata.decode.wrapper.run(
            q[:count], (keys, values), out=output[:count], lse=lse[:count], return_lse=True
        )
    if metadata.prefill is not None:
        assert isinstance(metadata.prefill, FIPrefill)
        assert isinstance(metadata.prefill.wrapper, BatchPrefillWithPagedKVCacheWrapper)
        metadata.prefill.wrapper.run(
            q[count:], (keys, values), out=output[count:], lse=lse[count:], return_lse=True
        )
    # FlashInfer reports base-2 LSE; authored reads expose natural logarithms.
    return output, lse.mul_(math.log(2))
```

为什么要乘一个 `math.log(2)`？因为 FlashInfer 内部为了用硬件指令加速指数运算，计算并返回的是以 2 为底的 $\text{LSE}_2$，而 pLLM 暴露给算法作者的契约是自然对数 $\text{LSE}_e$。在结尾原地乘上 $\ln 2$（耗时仅 `0.97 µs`），就把两边的语义严丝合缝地对齐了。

把三条路径放在同一个 0.5B 显微镜基准下同台竞技，账单如下：

| 窗口读取路径 | 原生 vLLM (tok/s) | LocalGlobal (tok/s) | 相对原生 | 单次窗口读耗时 |
|---|---:|---:|---:|---:|
| 1. 直接调 FA2 `window_size` | 39,120 | 30,846 | **78.8%** | 86.43 µs |
| 2. 复用 vLLM 的 FlashInfer 计划 | 39,069 | 38,050 | **97.4%** | 18.04 µs |
| 3. 原 Triton 单段读（实验对照） | 39,281 | 38,654 | **98.4%** | 16.23 µs |

拆开 FlashInfer 的 `18.04 µs`，其中注意力读取本体是 `14.84 µs`（和自写 Triton 的 `14.17 µs` 几乎打平），Graph 模式下 split-KV 的状态归约合并占了 `2.23 µs`，自然对数换底占了 `0.97 µs`。

注意看最后两行：原来那条 Triton 单段读其实比 FlashInfer 还要快约 11%，在 0.5B 整机吞吐上也高出 1 个百分点（98.4% 对 97.4%）。如果这是一场刷榜比赛，我们毫无疑问会把那条 Triton 路径捡回来。但作为一个长期演进的系统，我们毫不犹豫地没有捡它，保留了 FlashInfer。因为那 1% 的差距在真实的大模型上根本看不见——今天在 Qwen2.5-7B（vLLM 0.32，128 并发）上，走这套 FlashInfer 路径的 `LocalGlobal` 跑出了原生吞吐的 **100%**（同时把单请求 KV 显存上限从 112.0 MiB 砍到了 70.4 MiB）——而它为我们省下的，是一整套跨硬件、跨版本的内核维护包袱。

---

## 三、 当法则立住之后：四处看似无关的设计，其实是同一件事

一旦我们把这条法则提炼出来——**复用的基本单位不是一个孤立的函数，而是宿主系统自己的那条调用与生命周期路径**——回头再看 pLLM 的代码库，你会发现一件特别过瘾的事：全系统里另外四处处于不同模块、解决不同难题的核心设计，突然像多米诺骨牌一样，全都变成了这条法则的直接推论。

### 1. KV 内存管理器：零行分配代码的“伪装术”
在上一篇[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)里，我们要支持滑动窗口、保留 Prompt 的窗口（R-SWA）以及跨层共享 KV。一个冲动的系统程序员第一反应往往是：“我要接管 vLLM 的块分配器，自己写一套支持异构策略的页表管理器。”

但如果你去走上游自己的路呢？vLLM v1 的 `KVCacheManager` 本来就已经实现了 `SlidingWindowManager`、`RSWAManager` 和跨层共享目标（`kv_sharing_target_layer_name`）。于是 pLLM **一行内存分配代码都没写**。我们只做了一件事：在 vLLM 启动显存探测和构建管理器之前，走进去把每一层的属性“伪装”成 vLLM 原本就认识的那三种层，并把挑剔的 `SinkSpec` 排到字典最前面（`sink_first`），防止它被全量注意力的分组逻辑吞并。随后的物理块分配、滑出窗口后的 `null_block` 回收、抢占时的引用计数，甚至连准入控制里的 `max_admission_blocks_per_request`，百分之百由 vLLM 的原生管道自己跑完。

### 2. 残差融合 Norm：把加法“还给”上游
在[《图内挂载点：在 CUDA graph 里改残差》](05-in-graph-attach-points.md)里，用户写了一段 `@pllm.custom_residual` 来修改残差流，比如最简单的 `return x + y`。

为什么朴素的 hook 实现慢得要命？因为准入的现代模型（Qwen、Llama、DeepSeek）根本没有独立的残差加法算子，它们全都把“上一层的输出加回残差”融合进了下一层开头的 `fused_add_rms_norm` 里。如果你老老实实按照用户写的 `x + y` 发射一个独立的加法 kernel，算出和之后再交给原生的 `rms_norm`，你不仅多发了 kernel，还白白增加了一次显存读写往返。

怎么走原生模型自己走的路？我们没有去写一个“融合了用户代码与 RMSNorm”的新 kernel，而是用 PyTorch 的 `make_fx` 把用户的方法追踪成一张计算图。如果发现图的最后一步恰好是一个形状和类型匹配的 `aten.add`，我们就拿剪刀把这个加法节点从图里剪掉，把它的左右两个操作数原封不动地**还给**上游的 `fused_add_rms_norm`！连同清理掉端口上多余的拷贝和填零，这一招让 0.5B 模型每步 decode 图发射的 kernel 数直接从 628 个锐减到 388 个（8,192 token 生成的总 kernel 数从 166,277 个精准降回原生源算子的 104,837 个），恒等改写的吞吐从 89.8% 一步跨到了 **99.5%**。

### 3. 抽样原语：在自己的 kernel 里调别人的心脏
到了[《采样尾声融合》](07-sampler-epilogue.md)，为了实现有界引导采样（只对 top-p 过滤后剩下的平均约 4.8 个幸存者做 LM head 点积），我们确实不得不自己写一个 Triton kernel——因为上游没有任何入口支持“只挑出几个幸存的词表行做矩阵乘再重加权”。

但哪怕在这个必须手写的 kernel 内部，到了最后一步“根据重加权后的 logits 掷骰子抽样”时，我们依然停下了手，没有自己去写随机数生成和 Gumbel 噪声。我们在 Triton 代码里直接 import 并调用了 vLLM 原生的 `@triton.jit` 函数 `gumbel_block_argmax`。结果是什么？请求级的随机种子、token 位置偏移、FP32/FP64 精度分支以及贪心退化行为，天然与原生采样器严丝合缝。在初次实现时的 163,200 个引导行配对测试中，有界路径与稠密引导抽出的 token 达到了 **100% 逐个一致**。这是路径复用最细粒度的形态：就算外壳必须自己造，核心原语也要嵌入上游的齿轮。

### 4. 编译后端选择：反向复用更底层的上游
最后一个推论最反直觉，却最能体现这条原则的纵深。当模型挂上自定义注意力或残差改写时，pLLM 会要求 vLLM 使用 `"eager"` 编译后端来录制 CUDA Graph——注意，这绝不是退回慢吞吞的无图模式，而是让 CUDA Graph **直接录制模型原本的源 CUDA 算子**，绕开 PyTorch Inductor 的二次算子融合。

为什么放着现成的 Inductor 不用？这看起来像是在“拒绝上游工具”。但我们在 7B 模型上核对数值时发现，哪怕 FA2 注意力读取本身与原生逐元素相等，Inductor 面对“挂了自定义算子的图”和“干净的原生图”时，选择融合 `add` 与 `RMSNorm` 的边界也会发生变化。融合边界一变，浮点加法的结合顺序和舍入就变了，“中性配置下与原生 vLLM 逐位一致（Bitwise Neutrality）”的铁律就会被打破。

关掉 Inductor 融合、改录源算子，表面上是放弃了编译器的聪明才智（为此我们在 0.5B 上付出了约 1.9% 的固定吞吐代价；7B 上挂钩子的总开销约 1%–3%），实质上却是在**复用更底层、更确定的那个上游路径**：原生参考基准跑的是哪几个 C++/CUDA 源算子，我们就录一模一样的源算子。一旦走上同一条底层路径，7B 模型在反复测试下的 logits 和输出 token ID 便做到了分毫不差的逐位相同。

---

## 四、 解剖室里的成本账单与“清创审计表”

把原则讲得再漂亮，落到真实的系统工程里，也必须回答两个冷酷的问题：**怎么防止团队在日常开发中悄悄违背它？以及，坚持这条原则到底要付出什么代价？**

对第一个问题，我们的做法是把原则变成一张可执行的**代码清创审计表**。我们把 `pllm/` 下所有的 Python 模块和每一个手写的 Triton kernel 全部拉出来过堂，强制每一项手写实现回答三个问题：**pLLM 的实现在哪里？上游的对应物在哪里？上游缺了什么才逼得我们自己写（或者什么时候可以删掉）？**

下面是这张审计表（以及后来按它做的改动）里的几个切片：

| pLLM 中的实现模块 | 上游对应物与缺口审计结论 |
|---|---|
| 存储窗口读取（`flashinfer_attention.py`） | **已切换为上游路径**：直接借用 vLLM 的 `FlashInferMetadataBuilder` 与预规划 wrapper，仅保留自然对数换底与双 builder 组装。 |
| 全量 / 变参注意力读取（`custom_attention.py`） | **保留极薄适配层**：调用上游 `flash_attn_varlen_func` 同时获取输出与 LSE；待上游后端 `forward` 直接暴露 LSE 后即可删除。 |
| R-SWA 提示词窗口读取（`rswa_attention.py`） | **带便签的临时补丁**：输出直接走上游 `unified_attention`；因上游丢弃了 2-D 分支的 `M/L` 统计量，暂多发一次不分段 kernel 补算 LSE，上游暴露 LSE 即删。 |
| 查询行到请求的映射（`execution/attention.py`） | **已删除手写逻辑**：用 PyTorch 原生的 `torch.searchsorted` 替换掉原先手写的二次比较。 |
| 描述符指针的行收集（`kernels/page_copy.py`） | **批准保留**：上游 `index_select` 在录图时焊死输入张量地址；此处必须在 GPU 上解引用每步动态变化的外部张量地址（见《Python 只在录制时活着》）。 |
| 分页状态 AdamW（`kernels/adamw.py`） | **收缩至最小特化**：稠密更新已全部委托给 `torch.optim`；仅保留“从旧版本物理槽读、向新版本物理槽掩码写”的 PTTS 分页特化内核。 |

我非常推荐在任何扩展型项目里建立这样一张表。它给了代码库一条极硬的纪律：**手写 kernel 不是原罪，但每一个手写 kernel 都必须持有“上游缺口证明”才能存活。写不出上游缺什么的代码，就应该被删掉。**

而表格里的第三行（`rswa_attention.py`），恰好展示了当“上游真的有缺口”时，体面的工程妥协应该长什么样。

在保留完整 Prompt 加滑动窗口（R-SWA）的场景下，vLLM 使用的是它自己的 Triton `unified_attention` 启动器。它算出来的注意力输出完全正确，但它有一个让我们头疼的缺口：**它只返回 `output`，不返回 `LSE`！** 具体来说，在 vLLM 0.27.1 和 0.32.0 中，`unified_attention` 的 2-D 分支（包括所有 prefill）算完就直接把中间的最大值 `M` 和分母和 `L` 扔掉了；3-D 分支也只暴露出分段 decode 的临时草稿，归约后同样不返回 LSE。

遇到这种情况，你会怎么干？最容易犯的错误是把上游的 `unified_attention` 整个复制粘贴到自己的仓库里，改两行代码把 `M` 和 `L` 吐出来。那样干虽然省了一次内核发射，但从此你就得替上游维护那个几百行的巨型 Triton 内核了。

来看看我们在 `pllm/backends/vllm/rswa_attention.py` 里是怎么做的：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/rswa_attention.py
def rswa_attention(
    kv: PagedKV, q: torch.Tensor, scale: float, output: torch.Tensor, max_query_len: int, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # ……
    unified_attention(
        q=q,
        k=kv.key_cache,
        v=kv.value_cache,
        out=output,
        # ……
        rswa_prefix_lens=kv.prompt_lens,
        rswa_window=kv.storage_window,
    )
    _lse_statistics(kv, q, scale, partial, maximum, denominator)
    return output, maximum + denominator.log()
```

我们没有复制也没有修改上游的内核。真正的注意力输出 `output`，依然百分之百交给上游原装的 `unified_attention` 去算；紧接着在下一行，我们调用 `_lse_statistics`，用不分段的模式再调一次上游同一个 `kernel_unified_attention`，仅仅为了把被丢掉的 `maximum` 和 `denominator` 捞回来拼出 LSE。

多跑一次 kernel 优雅吗？一点也不优雅，它在 R-SWA 路径上实打实地重复了一次注意力读取。但它是诚实且有界的：它没有污染上游代码，没有引入私有的注意力算子分叉，并且把“为什么多跑一次”固化成了代码里一张随时可以撕掉的便签——只要哪天上游的 `unified_attention` 加上了返回 LSE 的选项，我们只需删掉 `_lse_statistics` 这一行，一切就归于圆满。

这就引出了第二个问题：除了像 R-SWA 这样为上游缺口买单，当我们选择全面复用上游路径时，我们还付出了哪些隐性成本？

天下没有免费的午餐。当你决定坐上宿主的顺风车，你就必须签下三份长期契约：

1. **接受上游的数值脾气**。复用 FlashInfer，意味着你必须接受它内部用 BF16 舍入后的概率来累加 softmax 分母，再从底数 2 转成自然对数。虽然 `window_attention` 返回的 LSE 类型是 `float32`，但它和 FA2 算出来的 `float32` 并不是逐比特相同的。我们在同一组 query 上对比过 FP64 参考基准：FlashInfer 与 FA2 的输出最大绝对误差都在 `0.0076` 左右；LSE 上，FA2 与 FP64 参考只差约 `0.00003`，FlashInfer 差约 `0.0013`，这正是 BF16 分母的舍入——与按同样 BF16 分母算出的 FP64 参考相比，它的差异不超过 `0.00005`（在窗口回收的独立参考计算中，输出相对 FP32 的最大绝对误差约 `0.0079`，LSE 约 `0.0017`）。对绝大多数算法来说这完全等价，但如果你的算法对 LSE 千分之一量级的差异敏感，你就必须把这个舍入量级算进验证容差里。
2. **接受上游的资源胃口**。vLLM 的 FlashInfer 后端在初始化时，会默认在显卡上划走一块 **`394 MiB`** 的共享工作区（workspace buffer）。这是上游管线自带的门票钱，不是扩展层自己能精打细算省下来的。而且我们的硬件切换规则目前也很保守：只在做过充分测试的 Ampere/Ada（capability 8.x）架构上将存储窗口切换到 FlashInfer；在 Hopper（SM90）架构或开启 `VLLM_BATCH_INVARIANT` 时，依然保留 FA2，而 Hopper 上的窗口表现在我们的单卡 4090 环境里尚未实测。
3. **替上游的跨版本演化看门**。走上游的路，意味着当上游修路改道时，你得跟着打方向盘。举个最真实的例子：在 vLLM 0.27.1 里，`FlashInferMetadataBuilder` 在构建计划时如果不做干预，会触发一次不必要的主机与设备间拷贝；到了 vLLM 0.32.0，上游自己优化掉了这次拷贝，同时还修改了查询 KV 缓存布局（`HND` vs `NHD`）的工具函数签名。为了同时兼容这两个版本，我们在 `flashinfer_attention.py` 开头写了一个小小的版本隔离阀：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/flashinfer_attention.py
if VLLM_027:
    # ……
    class FlashInferMetadataBuilder(VllmFlashInferMetadataBuilder):
        """Give 0.27's planner the exact host lengths that 0.32 already uses without a CUDA copy."""

        def build(
            self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
        ) -> FlashInferMetadata:
            # MRv2's host bound is exact outside speculative decoding, which pLLM does not enable.
            if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
                common_attn_metadata = replace(
                    common_attn_metadata, _seq_lens_cpu=common_attn_metadata.seq_lens_cpu_upper_bound
                )
            return super().build(common_prefix_len, common_attn_metadata, fast_build)
```

有人看到这段 `if VLLM_027:` 可能会抱怨：“你看，复用上游还是要写版本兼容代码，多麻烦！”

但请算一笔对比账：如果自己维护一套手写的注意力 kernel、自己的页表规划和自己的图缓冲区，当 vLLM 从 0.27 升级到 0.32 时，你要面对的是几千行底层接口的全面对撞；而现在，所有的版本差异被死死关在了这十几行适配器里。

---

## 五、 结语：扩展层的生存之道

回头看，从原来的 Triton 路径（98.4%），到盲目调 FA2 `window_size` 掉进坑里（78.8%），再到接入 vLLM 的 FlashInfer 规划管线回到高点（0.5B 上 97.4%，7B 上 100%），这短短一圈折腾，把一个系统扩展层最核心的生存法则浓缩在了几行代码里。

每个工程师都知道“不要重复造轮子（Don't Reinvent the Wheel）”。但在复杂的现代推理引擎里，真正的陷阱往往不是从零造一个轮子，而是**从上游卸下一个轮子，然后自己焊一套车轴去转它**。

当你只调用上游的一个裸函数，却绕开了它选后端、建 `plan`、管 CUDA Graph 缓冲区、融合算子和分配物理块的原生管线时，你看似复用了代码，实则背叛了架构。反过来，当你学会顺着宿主系统自己的毛孔呼吸——用它预期的 Spec 喂给它的内存管理器，把剪下来的加法操作数还给它的融合 Norm，在它规划好的注意力包装器上直接按运行键——上游社区成百上千位工程师日夜打磨的每一分优化，都会自动流淌进你的扩展层里。
