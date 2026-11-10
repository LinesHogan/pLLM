# 复用上游：kernel、管理器与原语

在 Qwen2.5-0.5B 的单层注意力上跑一组微基准测试，你会看到两个乍看之下违反直觉的数字。

测试条件很单纯：单张 RTX 4090，128 条并发解码序列，每条序列有 256 个历史 token，14 个 Query 头共享 2 个 KV 头（头维 64，BF16），放在 CUDA Graph 里反复回放 1,000 次。如果调用 FlashAttention-2（FA2）的 `flash_attn_varlen_func` 读取全部 256 个历史位置，一次调用耗时 **16.56 µs**。如果给同一个函数传进 `window_size=(3, 0)`，让它只读取最近 4 个位置——读取的数据量减少了 98% 以上——一次调用的耗时却涨到了 **91.47 µs**。

少读了几乎全部历史，耗时反而慢了五倍半。

如果把视野放大到整机吞吐，同样的现象依然存在。在[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)讨论过的 `LocalGlobal` 布局里，模型共有 24 层，其中偶数层只保留最近 256 个 token 的滑动窗口，奇数层保留全量历史（同样在 RTX 4090、vLLM 0.27.1、PyTorch 2.13 下测试，8 个 prompt × `n=16` 共 128 条并发序列，生成 256 个 token，`max_model_len=2048`）。当窗口层直接通过 FA2 的 `window_size=[window - 1, 0]` 读取时，3,024 次稳态窗口读的平均耗时是 **86.43 µs**，整机吞吐从原生全量注意力的 39,120 tok/s 跌到了 **30,846 tok/s（78.8%）**。即便只在 24 层里把 2 层改成长度为 4 的极小窗口，整机吞吐也只有全量注意力的 **89.3%**（vLLM 0.27.1）和 **90.7%**（vLLM 0.32.0）。

为什么调用上游最标准的滑动窗口接口，会得到一个比全量注意力慢得多的结果？

顺着这个问题往下查，不仅能看清分组查询注意力（GQA）在解码阶段的硬件带宽账，还会触及一个在构建系统扩展层时反复出现的核心问题：**当我们在宿主系统之上编写扩展时，“复用上游”到底意味着什么？**

---

## 一、 先算清 GQA 解码的带宽账

在自回归解码（decode）阶段，每条序列每一步只进入 1 个新 token。此时算术计算量极小，注意力层的速度几乎完全受限于显存带宽——把历史 Key 和 Value 从 HBM 显存搬进芯片寄存器的时间，决定了内核的下限。

现代大模型普遍采用分组查询注意力（GQA）。在 Qwen2.5-0.5B 中，每层有 14 个 Query 头，但只有 2 个 KV 头；在 Qwen2.5-7B 中，每层有 28 个 Query 头、4 个 KV 头。两者的比例都是 **7 : 1**，即每 7 个 Query 头共享完全相同的一份 Key 和 Value。

因此，计算解码注意力的理想显存流量时，有效载荷必须按 **KV 头数** 而不是 Query 头数来算。以前面 0.5B 的 128 并发设置为例，每步需要读取的 KV 字节数是：

$$128 \times \min(\text{历史长度}, 256) \times 2\text{ (K与V)} \times 2\text{ (KV头)} \times 64 \times 2\text{ Bytes}$$

当历史长度为 128 时，有效载荷是 8 MiB；当历史长度达到 256 时，有效载荷是 16 MiB。按 RTX 4090 标称的 1,008 GB/s 显存带宽折算，搬运 8 MiB 和 16 MiB 的理论极限时间（Roofline）分别是 **8.3 µs** 和 **16.6 µs**。

回头看刚刚测出的数字：FA2 全量读取 256 个历史位置的实测耗时是 `16.56 µs`，几乎贴在这条硬件带宽极限线上。

它能贴住极限，靠的是 **GQA 头打包（Packed GQA）**。在解码阶段，同一组里的 7 个 Query 头要读的既然是同一批 KV 页，内核就会把这 7 个 Query 头打包进同一个线程块（CTA）里：从显存把 KV 页加载进片上内存一次，7 个 Query 头在寄存器和共享内存里共同消费这份数据，避免重复访问显存。

但在我们测试的这个形状下，一旦给 FA2 传进滑动窗口参数，解码路径就关掉了 GQA 打包。从 profiler 捕获的内核发射网格（Grid）可以直接看到这一变化：

```text
1. 打包 GQA 解码（Grid 按 2 个 KV 头组织，每页 KV 逻辑上只请求 1 次）：
   ┌───────────────────────────┐
   │  KV Head 0 物理页 (HBM)    │ ──(读取 1 次)──► [ CTA: Q0, Q1, Q2, Q3, Q4, Q5, Q6 ]
   └───────────────────────────┘

2. 未打包的窗口解码（Grid 退化为按 14 个 Query 头组织，逻辑请求重复 7 次）：
   ┌───────────────────────────┐    ┌──► [ CTA 0: Q0 ]
   │                           │────┼──► [ CTA 1: Q1 ]
   │  KV Head 0 物理页 (HBM)    │────┼──►   ...          (7 个独立线程块各自发起读取，
   │                           │────┴──► [ CTA 6: Q6 ]   实际显存流量取决于缓存命中)
   └───────────────────────────┘
```

当线程网格从按 2 个 KV 头组织退化为按 14 个 Query 头组织时，原本只需请求 1 次的 KV 页，在逻辑上被 7 个各自独立的线程块重复请求了 7 次（至于其中有多少被 L2 缓存吸收、有多少真正落到显存控制器上，取决于运行时的调度时序），再加上窗口边界判断与分块调度的固定开销，单次读取耗时便从 16 µs 膨胀到了 86 µs 以上。这说明：**在 GQA 模型上，内核有没有保持 Query 头打包，比窗口少读了多少历史位置重要得多。**

---

## 二、 两种直觉解法，与第三条路

弄清了慢的原因，接下来该怎么选？通常会有两种直觉反应。

第一种反应是“自己动手写一个”。既然上游的 FA2 窗口路径关掉了 GQA 打包，那我们干脆写一个专门服务于单段窗口读取的 Triton 内核（或者直接调底层的原始 Triton 算子再自己配一个收尾内核来归一化并输出 `log-sum-exp`）。

在同一个 0.5B `LocalGlobal` 基准上，这条自写单段 Triton 对照路径（使用 `16×32` 分块，grid 为 `[192, 2, 1]`）跑得非常快：单次读取总共只需 **16.23 µs**（其中注意力读取本体 14.17 µs，LSE 收尾内核 2.06 µs），整机吞吐达到 **38,654 tok/s**，是同期原生全量基线（39,281 tok/s）的 **98.4%**。

单看跑分，这件事似乎已经结束了。但如果站在系统长期演进的角度看，每一次像这样绕开宿主后端、自己维护一套平行算子，都是在背负隐性债务。在一个构建于推理引擎之上的可编程扩展层里，如果注意力自己写一套、归一化自己写一套、采样抽样自己写一套，扩展层很快就会膨胀成半个山寨版的宿主引擎。每一个自写内核都需要独立的数值基准（Oracle）来验证精度，都会在遇到新硬件架构或上游内部接口调整时成为维护负担。

那么第二种反应呢？“换用上游的另一个算子库——FlashInfer。”

vLLM 本身就集成了 FlashInfer。在 Ampere 和 Ada 架构（SM80/SM89，如 A100 和 RTX 4090）上，FlashInfer 的 tensor-core 解码内核使用 `16×128` 的 Q/KV 分块和 4 个 warp，即便开启了滑动窗口（`window_left`），它的线程网格依然按 2 个 KV 头组织（在我们的测试图里，grid 是规规矩矩的 `[183, 1, 2]`）。

但如果你试图在自己的扩展代码里直接 `import flashinfer` 并调用它的底层函数，马上就会撞上另一堵墙。

FA2 在调用方式上接近一个无状态函数：传入张量和块表，直接拿到输出。而 FlashInfer 是一个**两阶段的有状态管线**。在每一轮发射 GPU 内核之前，它必须在 CPU 上先执行一次 `plan`（规划）：根据当前批次每条序列的实际长度切分 KV 工作量（split-KV），生成页索引与归约偏移。更关键的是，为了让解码步能够被录制进 CUDA Graph 反复回放，规划阶段写入的页索引暂存区（page-index staging）和 split-KV 中间缓冲区，**物理地址必须在录制时完全固定**，并在每次回放前由宿主端原地更新。

如果扩展层自己去调 FlashInfer 的底层接口，就意味着扩展层必须自己管页索引转换、自己算 split-KV 规划、自己分配并维护一套适配 CUDA Graph 的固定地址缓冲池。为了复用一个内核，反而要在外围把宿主引擎的后端管线重写一遍。

这就引出了真正走得通的第三条路：**不要把宿主系统当成一堆孤立函数的集合，而要直接接入宿主系统自己的那条执行路径。**

vLLM 的 `vllm.v1.attention.backends.flashinfer` 模块里，本来就有现成的 `FlashInferMetadataBuilder` 和 `FlashInferImpl`。页索引暂存、split-KV 规划、CUDA Graph 缓冲区以及共享工作区，vLLM 早就替 FlashInfer 管好了。我们不需要在旁边另起炉灶，只需要让滑动窗口层直接坐上 vLLM 自己的这条管线。

```text
孤立的函数调用（绕开宿主管线，不得不自己重建外围状态）：
  扩展层钩子 ──(自己做页索引转换、split 规划与图缓冲管理)──► 底层算子函数

路径级复用（直接接入宿主的原生构建与执行管线）：
                                ┌──► FlashInferMetadataBuilder ──► 预规划 Wrapper (存储窗口读)
  vLLM 原生页映射 ──────────────┤
  (WindowMetadataBuilder)       └──► FlashAttentionMetadataBuilder ──► FA2 Metadata (供变参读备用)
```

在 `pllm/backends/vllm/flashinfer_attention.py` 中，整套机制只用了几十行代码。首先，当某个注意力层被声明为滑动窗口存储（且运行在计算能力 8.x 的 Ampere/Ada 显卡上、未开启 batch-invariant、头维受支持）时，`install_window` 会在引擎初始化阶段将该层的后端切换为 `WindowBackend`，并换上原生的 `FlashInferImpl`。

接着，`WindowMetadataBuilder` 继承自原生的 `FlashInferMetadataBuilder`，让两个上游构建器同时消费同一份原生页映射：

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

为什么要在构建 FlashInfer 元数据的同时，顺手构建一份 `flash`（FA2）元数据？因为用户在编写 `@pllm.custom_attention` 方法时，除了按默认存储窗口读取之外，还可能在代码里临时指定一个不同的窗口大小、或者只挑选一部分注意力头来做二次读取。主路径用预规划好的 FlashInfer 保证速度，而旁路变参读取随时可以回退到灵活的 FA2，两边共享同一套底层页表，互不干扰。

到了真正执行读取的 `window_attention`，代码就只剩下对上游预规划包装器（wrapper）的直接调用：

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
        metadata.decode.wrapper.run(  # pyright: ignore[reportUnknownMemberType]
            q[:count], (keys, values), out=output[:count], lse=lse[:count], return_lse=True
        )
    if metadata.prefill is not None:
        assert isinstance(metadata.prefill, FIPrefill)
        assert isinstance(metadata.prefill.wrapper, BatchPrefillWithPagedKVCacheWrapper)
        metadata.prefill.wrapper.run(  # pyright: ignore[reportUnknownMemberType]
            q[count:], (keys, values), out=output[count:], lse=lse[count:], return_lse=True
        )
    # FlashInfer reports base-2 LSE; authored reads expose natural logarithms.
    return output, lse.mul_(math.log(2))
```

最后一行 `lse.mul_(math.log(2))` 处理了一个细微的语义对齐：FlashInfer 内部为了配合硬件的以 2 为底指数指令，返回的 `lse` 是以 2 为底的对数；而 pLLM 暴露给算法作者的 `kv.attend` 契约统一使用自然对数。原地乘上 $\ln 2$，就把两边接上了。

把这三种方案放在同一个 0.5B `LocalGlobal` 基准下对比，账面非常清楚：

| 窗口读取方案 | 原生全量 vLLM (tok/s) | LocalGlobal (tok/s) | 相对原生 | 单次窗口读耗时 |
| :--- | ---: | ---: | ---: | ---: |
| 直接调用 FA2 `window_size` | 39,120 | 30,846 | 78.8% | 86.43 µs |
| 接入 vLLM 的 FlashInfer 计划 | 39,069 | 38,050 | 97.4% | 18.04 µs |
| 专用 Triton 单段读（实验对照） | 39,281 | 38,654 | 98.4% | 16.23 µs |

*(注：测于单卡 RTX 4090，vLLM 0.27.1，PyTorch 2.13，Qwen2.5-0.5B-Instruct，128 并发序列生成 256 token，偶数层窗口 256，预热 1 轮后取 3 轮中位数；在同一批硬件状态的 rebased 样本下，FA2 与 FlashInfer 的相对吞吐分别为 83.2% 与 98.8%。换用单步计算更重的 Qwen2.5-7B-Instruct、vLLM 0.32.0 时，走这套 FlashInfer 路径的 `LocalGlobal` 实测吞吐为原生的 100%。)*

把 FlashInfer 的 `18.04 µs` 拆开看：注意力读取本体占 `14.84 µs`（与专用 Triton 的 `14.17 µs` 几乎一致），图模式下 split-KV 的状态归约合并占 `2.23 µs`，乘 $\ln 2$ 换底占 `0.97 µs`。

对比表的最后两行是最值得玩味的地方：专用 Triton 单段读其实比 FlashInfer 还要快约 11%，在 0.5B 整机吞吐上也高出 1 个百分点（98.4% 对 97.4%）。为什么放着 98.4% 不用，而要选 97.4% 的 FlashInfer？因为那 `2.23 µs` 的 split-KV 合并开销在 7B 这样规模的模型上已经完全淹没在二十多毫秒的单步耗时里（整机吞吐 100%），而放弃自写内核换来的，是整条路径的规划、图缓冲和内核演进全部由上游原生管线兜底。

---

## 三、 同一个原则在不同模块里的投射

一旦看清“复用应当是路径级的，而不是孤立函数级的”，回头审视整个推理扩展层的设计，会发现其他几个核心模块里看似各异的做法，其实都在遵循同一条逻辑。

第一个例子是 **KV 缓存的物理块管理**（见[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)）。当用户声明某几层使用滑动窗口、某几层使用保留完整 Prompt 的窗口（R-SWA）、某几层跨层共享 KV 时，扩展层自己写了几行物理块分配与回收代码？一行都没写。因为 vLLM v1 的 `KVCacheManager` 本来就已经有 `SlidingWindowManager`、`RSWAManager` 和 `kv_sharing_target_layer_name`。扩展层做的唯一一件事，是在引擎探测显存和初始化管理器之前，把每一层伪装成 vLLM 原本就认识的规格（Spec），并调整字典顺序防止 R-SWA 子类被全量注意力组吞并。随后的块分配、滑出窗口后的 `null_block` 占位回收、抢占清理以及 `pllm.explain` 里的准入容量计算，全部走的都是 vLLM 自己的管理路径。

第二个例子是 **图内残差改写与融合归一化**（见[《图内挂载点：在 CUDA graph 里改残差》](05-in-graph-attach-points.md)）。现代模型为了省显存往返，把残差加法融进了下一层开头的 `fused_add_rms_norm` 里。如果用户写了 `@pllm.custom_residual`（例如 `return x + 0.95 * y`），朴素的做法是在钩子里先跑完这个加法、写出张量，再调归一化——这会在一次 8,192 token 的 0.5B 生成里平白多发射 12,288 个独立的 BF16 加法内核（连同端口拷贝与填零共多出 61,440 个内核）。我们的解法不是去手写一个“把用户逻辑和 RMSNorm 揉在一起”的新内核，而是用 `make_fx` 检查用户的计算图：只要最后一步是形状和类型匹配的 `aten.add`，就直接把图末尾的加法节点剪掉，把左右两个操作数交还给上游原生的 `fused_add_rms_norm` 去一并完成相加与归一化。单步解码图发射的内核数由此从 628 个精准降回原生源算子的 388 个，恒等改写的吞吐也从 89.8% 回到了 99.5%。

第三个例子是 **采样尾声里的随机抽样原语**（见[《采样尾声融合》](07-sampler-epilogue.md)）。在实现有界后过滤引导采样时，上游确实没有任何接口支持“只挑出 top-p 过滤后幸存的几个词表行做 LM head 点积并重加权”，因此外层的扫描与点积内核必须自己写。但到了内核最末尾、需要根据重加权后的 logits 注入 Gumbel 噪声并选出最大值的那一刻，我们没有自己写随机数生成器和比较逻辑，而是在 Triton 内核里直接调用了 vLLM 原生的 `@triton.jit` 函数 `gumbel_block_argmax`。请求级的随机种子、位置偏移、FP32/FP64 噪声精度以及贪心退化规则，依然由上游的那几行核心代码决定。在 163,200 个引导行的配对测试中，有界路径与全词表稠密引导抽出的 token 达到了 100% 逐个一致。

第四个例子最为微妙：**为了复用更底层的算子语义，有时反而要主动关掉上层的自动优化。** 当模型挂上自定义注意力或残差改写时，pLLM 会将 vLLM 录制 CUDA Graph 时的编译后端设为 `"eager"`——这里的 `"eager"` 不是退回无图模式，而是让 CUDA Graph 直接捕获模型原本的手写 C++/CUDA 源算子，关闭 PyTorch Inductor 的二次算子融合。为什么？因为在 7B 模型上我们观察到，即使自定义注意力里调用的 FA2 读取结果与原生逐元素完全相等，图的拓扑变化也会让 Inductor 改变周围 `add` 与 `RMSNorm` 的融合边界，进而改变 BF16 浮点加法的归约顺序与尾数舍入。关掉 Inductor 融合（在 0.5B 上大约付出 1.9% 的固定吞吐代价，7B 上挂载钩子的总开销在 1%–3% 之间），换来的是被录进图里的每一个底层源算子都与原生参考路径完全一致，从而守住了“中性参数下输出与原生逐位相同（Bitwise Neutrality）”的契约。

---

## 四、 给每一个手写实现立一张“缺口清单”

在工程实践中，“能用上游就用上游”很容易沦为一句空洞的口号。因为具体写代码的时候，自己顺手写一个百来行的辅助函数或 Triton 内核总是比去通读上游源码要快得多。

要让这条原则真正落地，最有效的办法是建立一张**缺口审计表（Reuse Audit）**：把代码库里所有的自定义模块和手写内核列出来，强制每一项回答一个问题——**上游的对应物在哪里？它缺了什么具体能力，才迫使我们保留这段代码？**

下面是我们对核心执行与后端模块做清点时的几行记录：

| 模块与功能 | 上游对应物 | 审计结论与保留理由 |
| :--- | :--- | :--- |
| **存储窗口读取**<br>`flashinfer_attention.py` | vLLM `FlashInferMetadataBuilder` / `FlashInferImpl` | **完全接入上游路径**。只保留双 builder 组装与底数 2 到自然对数的原地换底。 |
| **全量与变参注意力读**<br>`custom_attention.py` | `vllm_flash_attn.flash_attn_varlen_func` | **极薄封装**。因上游注意力层 `forward` 只返回 `output` 不暴露 `LSE`，需直接调该函数同时取回 `(out, softmax_lse)`。 |
| **R-SWA 提示词窗口读**<br>`rswa_attention.py` | vLLM `triton_unified_attention` | **复用输出 + 独立补算 LSE**。上游启动器丢弃了 LSE 统计量，暂多发一次不分段内核回收 `M/L`，上游暴露 LSE 后即可删除。 |
| **查询行到请求映射**<br>`execution/attention.py` | `torch.searchsorted` | **直接用上游原语**。对 `query_start` 做设备端二分查找，不手写比较逻辑。 |
| **外部动态张量行收集**<br>`kernels/page_copy.py` | `torch.index_select` | **必须保留**。原生算子在录图时焊死输入张量基地址；此处需从固定表读出 `int64` 地址并在 GPU 上转回指针解引用（见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。 |
| **分页状态 AdamW**<br>`kernels/adamw.py` | `torch.optim.AdamW` | **只保留最小特化**。普通稠密更新交还 `torch.optim`；仅保留为 **PTTS**（Paged Tensor Transition System，开头的 P 指与虚拟内存同源的 Paged 分页，和 pLLM 里代表 programmable 的 p 无关）从旧版本物理槽读、向新版本物理槽掩码写的分页内核。 |

这张表给代码库立下了一条清晰的纪律：**手写内核必须持有“上游缺口证明”才能存在。**

表里的第三项（`rswa_attention.py`）是一个很好的例子，展示了当上游确实存在缺口时，应该如何划定边界。

在保留完整 Prompt 加滑动窗口（R-SWA）的布局下，vLLM 使用的是它的 Triton `unified_attention` 启动器。这个启动器算出的注意力输出 `output` 是完全正确的，但在 vLLM 0.27.1 和 0.32.0 中，它只返回 `output`，不返回 `LSE`：它的 2-D 分支（包含全部预填充阶段）在内核结束时直接丢弃了每行的最大值 `M` 与指数和 `L`；它的 3-D 分支虽然在分段解码时写了临时缓冲区，但经过 `reduce_segments` 归约后同样不向外返回 LSE。

面对这个缺口，一种诱人的做法是把上游几百行的 `unified_attention` 复制进自己的仓库，改几行代码把 `M` 和 `L` 接出来。那样做虽然省掉了一次内核调用，却意味着我们要从此替上游维护一整套复杂的 Triton 注意力内核。

我们在 `pllm/backends/vllm/rswa_attention.py` 里选择了另一种做法：

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

真正的注意力输出 `output`，依然原封不动地交给上游的 `unified_attention` 去生成；紧接着，`_lse_statistics` 用不分段的模式（`NUM_SEGMENTS_PER_SEQ=1, IS_3D=True`）再调用一次上游同一个 `kernel_unified_attention`，只为了把 `maximum` 和 `denominator` 取出来算出 `maximum + denominator.log()`。

多发射一次内核当然有代价，它在 R-SWA 读取路径上实打实地多读了一遍缓存。但这个代价是局部的、透明的，而且没有复制一行上游内核源码。模块顶部的注释明确记下了这笔账：只要未来某一天上游给 `unified_attention` 加上了可选的 LSE 返回值，删掉 `_lse_statistics` 这一行即可。

---

## 五、 复用的长期账单

当然，接入上游路径绝不意味着“零成本”。当你决定把扩展层架在宿主系统的原生管线上时，你省下了维护私有内核的精力，但必须承担另外三项长期的工程代价。

**第一项代价，是接受上游不同后端之间的数值差异。**
虽然 `window_attention` 和 FA2 返回的 LSE 张量类型都是 `float32`，但它们并不是逐位相同的。FlashInfer 在内部累加 softmax 分母时，使用的是舍入到 BF16 后的概率值，最后再从底数 2 的对数转换回自然对数。
我们在同一组 Query 上与 FP64 参考基准做过精确比对：在注意力输出 `output` 上，FlashInfer 与 FA2 相对于 FP64 参考的最大绝对误差都在 **`0.0076`** 左右；而在 `LSE` 上，FA2 与标准 FP64 参考相差约 `0.00003`，FlashInfer 则相差约 `0.0013`——这正是 BF16 分母舍入带来的差异（如果拿同样按 BF16 分母累加的 FP64 参考去和 FlashInfer 比对，两者的 LSE 差异不超过 **`0.00005`**；在窗口物理块回收的独立 FP32 参考验算中，输出最大绝对误差约为 `0.0079`，LSE 约为 `0.0017`）。对于绝大多数算法而言，千分之一量级的 LSE 差异完全在噪声范围内；但如果某个算法对极小的对数分母漂移极其敏感，就必须在验证容差中显式考虑这一数值特性。

**第二项代价，是接受上游管线的资源预留与平台边界。**
接入 vLLM 的 FlashInfer 管线，意味着在引擎启动时，vLLM 会按其默认配置分配一块 **394 MiB** 的共享工作区（Workspace Buffer）。同时，我们的切换条件也必须迁就上游后端的覆盖范围：目前 `install_window` 只在经过充分验证的 Ampere 与 Ada 架构（计算能力 8.x）上启用 FlashInfer 窗口路径；在 Hopper（SM90）架构、开启 `VLLM_BATCH_INVARIANT`、头维不在支持列表内、或者自定义读取改变了头数与连续性时，系统仍会保留 FA2 路径（Hopper 架构上的窗口表现尚未在我们的单卡 4090 环境中实测）。

**第三项代价，是替上游的跨版本演进看门。**
只要你复用了宿主的内部构建器，当宿主升级版本时，你就必须跟进它的内部签名变化。比如从 vLLM 0.27.1 升级到 0.32.0 时，`FlashInferMetadataBuilder` 就有两处细微的变动：一是查询 KV 缓存布局（`HND` 还是 `NHD`）的辅助函数从进程级的 `get_kv_cache_layout()` 变成了挂在元数据上的 `get_flashinfer_layout_string(metadata.kv_cache_layout)`；二是在 0.27.1 中，构建器如果不显式拿到 CPU 侧的序列长度，就会触发一次不必要的设备到主机同步拷贝，而 0.32.0 已经原生消除了这次拷贝。

因此，在 `flashinfer_attention.py` 的开头，有一段专门用来抹平这两个版本差异的适配代码：

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

有人看到这类版本分支，可能会觉得“复用上游也挺麻烦的”。但把这十几行适配代码和“自己维护一套跨版本的分页注意力内核、split-KV 规划器与 CUDA Graph 内存池”放在天平上称一称，孰轻孰重一目了然。把版本差异集中在薄薄的一层隔离舱里，正是为了让上面的扩展逻辑和下面的硬件内核都能安心做自己最擅长的事。

---

## 六、 结语

回顾这三种读法——98.4%（专用 Triton）、78.8%（直接调 FA2 窗口）与 97.4%（0.5B）/ 100%（7B，接入 vLLM 的 FlashInfer 计划）——它们留下的教训非常朴素。

在成熟的底层系统之上做扩展开发，“不要重复造轮子”并不是一句简单的“去调上游的 API”。因为高性能系统里的核心组件——无论是注意力内核、分块归约还是显存分配器——从来都不是孤立存在的纯函数。它们依赖特定的线程块打包方式、特定的宿主端预先规划（Plan）、以及特定的静态图内存绑定。

当你只把上游当成函数库、随手抽一个函数出来裸调时，你往往会掉进性能陷阱；而当你试图绕开上游自己手写全套内核时，你又背上了本不该由扩展层承担的维护包袱。真正的复用，是看懂宿主系统自己的数据流与生命周期，把扩展逻辑挂进宿主已经铺好的那条高速轨道上。
