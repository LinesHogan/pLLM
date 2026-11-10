# 复用上游：kernel、管理器与原语

在一个成熟的大型运行时上做扩展——Linux 内核、PostgreSQL、vLLM——迟早会听到一个声音：自己写一个吧。上游的分配器太保守，自己写一个；上游的算子不返回你要的中间量，自己写一个 kernel；上游的编译器做了你不想要的融合，干脆接管整张图。每一次都有很好的理由，写出来也常常更快。

2026 年 11 月 4 日，pLLM 的作者看着代码库里越来越多的手写 kernel 和平行路径，写下了一句话：“到处造轮子不符合 pllm 的设计思想。”

这篇文章想说的是，这句话背后不只是一种洁癖，而是一个可以操作的原则；而且在 pLLM 里，看起来毫不相干的几处设计——注意力 kernel 的选择、KV 内存管理、残差的融合、采样的随机数、编译后端的选择——其实都是同一个原则在不同地方的样子。我们先讲一个它差点失败的例子。

## 一个“更对”的改动，让吞吐掉了两成

pLLM 允许用户声明某些层只保留一个滑动窗口的 KV（[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)）。窗口读最初是 pLLM 自己写的 Triton kernel。11 月 4 日，按照“能用上游就用上游”的原则，这部分换成了 FA2 原生的 `window_size` 参数——FA2 本来就支持滑动窗口，何必自己维护一个 kernel？同一次改动还删掉了一个 pLLM 自己写的输出归一化 kernel。

改动“更干净、更标准”，结果却不好。在一个只有 2/24 层是窗口、窗口只有 4 的设置下（RTX 4090，0.5B，8 × n=16，256 token），窗口配置只有全量的 0.893（vLLM 0.27.1）和 0.907（vLLM 0.32）——目标本来是“窗口不慢于全量”。抽出来做微基准（128 行，14 个 query 头 / 2 个 KV 头，头维 64，序列长 256，1000 次 graph 回放）：FA2 全量读 16.56 µs，`window_size=(3,0)` 只读最近 4 个 token，却要 91.47 µs。读得更少，慢了五倍多。到了真正的局部 / 全局交替布局（`LocalGlobal`，偶数层窗口 256），FA2 窗口路径只有原生的 78.8%（30,846 对 39,120 tok/s），一次窗口读 86.43 µs。

为什么？decode 的注意力是带宽受限的，账要按 KV 头算。Qwen2.5-0.5B 有 14 个 query 头、2 个 KV 头，每 7 个 query 头共享一组 K/V。好的 decode kernel 会把这 7 个头“打包”在一起，同一份 K/V 读一次、服务 7 个头。我们观察到的是：FA2 走窗口路径时，grid 从按 KV 头组织变成了按 query 头组织（pLLM 的 FlashInfer 模块 docstring 写的是“FA2 disables its packed GQA decode for windows”）。同一份 K/V 被 7 个线程块各读一遍，能不能命中缓存决定了真实代价。按有效载荷算，128 行、历史 256 时只有 16 MiB，在 4090 标称 1,008 GB/s 下理想时间约 16.6 µs——FA2 全量读几乎就贴在这条线上，而窗口读的 86 µs 显然不能只用载荷解释。

## 第三条路：像上游自己那样调用上游

面对这个结果，最自然的反应是退回去：自己写的 Triton 单段读在同样设置下有 98.4%（38,654 对 39,281 tok/s），一次读 16.23 µs。可那样就又回到了“每个 kernel 自己维护、自己验证数值、自己跟版本”的老路。

我们找到的是第三条路。vLLM 里除了 FA2 还有 FlashInfer，它的 tensor-core decode 用 16×128 的 Q/KV tile、4 个 warp，并且保持按 KV 头组织（在这个设置的 graph 里，grid 是 `[183, 1, 2]`）。可 FlashInfer 不是一个无状态的函数：decode 前要先“plan”——按长度把 KV 切成若干段——在 CUDA graph 模式下，plan 和它的缓冲区还要地址固定。如果 pLLM 自己去调 FlashInfer，就得自己管 plan、管缓冲区、管 graph，这又是一种造轮子。

关键在于：vLLM 自己已经为 FlashInfer 后端做好了这一切——页索引的暂存、split 的规划、graph 的缓冲区，都在它的 metadata builder 里。所以 pLLM 做的不是“调用 FlashInfer”，而是**像 vLLM 自己那样调用 FlashInfer**：借用 vLLM 的 builder，只在旁边多做两件小事（`pllm/backends/vllm/flashinfer_attention.py`）：

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

一份原生的页映射，同时喂给两个上游 builder：FlashInfer 的那份用来做存储窗口的读，FA2 的那份留给作者另行指定窗口或头数的读。另一件小事是 LSE 的底数：FlashInfer 报的是以 2 为底的 log-sum-exp，而 pLLM 给作者的是自然对数，所以 `window_attention` 最后一行是 `lse.mul_(math.log(2))`，注释写着 “FlashInfer reports base-2 LSE; authored reads expose natural logarithms”。

结果（同一设置：RTX 4090，vLLM 0.27.1，0.5B，`max_model_len` 2048，8 × n=16，256 token，1 轮预热 + 3 轮中位数；每次窗口读的时间来自 3,024 次稳态读）：

| 窗口读 | 原生 vLLM | LocalGlobal | 相对原生 | 每次窗口读 |
|---|---:|---:|---:|---:|
| FA2 `window_size` | 39,120 | 30,846 | 78.8% | 86.43 µs |
| vLLM 的 FlashInfer 计划 | 39,069 | 38,050 | 97.4% | 18.04 µs |
| 自写 Triton 单段读（对照） | 39,281 | 38,654 | 98.4% | 16.23 µs |

FlashInfer 的 18.04 µs 里，读本身是 14.84 µs，graph 模式下 split-KV 的状态合并 2.23 µs，换底 0.97 µs。它比自写的 Triton 慢约 11%，整机差 1–2 个百分点。我们保留的是 FlashInfer。今天在 Qwen2.5-7B 上（vLLM 0.32，128 并发），`LocalGlobal` 是原生的 100%。

这个选择值得说清楚它放弃了什么。复用意味着接受上游的数值语义：FlashInfer 的 softmax 分母累加的是 BF16 舍入后的概率，FP32 的返回类型不代表和 FA2 逐位相同；同一组 query 上，两者对 FP64 参考的输出最大误差都约 0.0076，LSE 的差异不超过 0.00005。复用意味着接受上游的资源：这个设置下 vLLM 默认分配一块 394 MiB 的共享工作区。复用也意味着跟随上游的版本差异：0.27 的 FlashInfer builder 需要 pLLM 把主机端的长度上界交给它当精确长度（避免一次 CUDA 拷贝），0.32 已经原生如此；两个版本的 KV 布局查询方式也不一样。这类差异都关在一个文件里，但都要有人看着。

有一处我们没能完全复用。保留 prompt 的窗口（R-SWA）走 vLLM 的 Triton `unified_attention`，可这个 launcher 不返回 LSE。`pllm/backends/vllm/rswa_attention.py` 的模块 docstring 把缺口写成了可以检验的事实：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/rswa_attention.py
"""Read R-SWA output through vLLM's launcher and recover its otherwise discarded LSE.

Custom attention uses FA2/FlashInfer for full and ordinary window reads. vLLM 0.27.1/0.32.0
TritonAttentionImpl.forward calls unified_attention, which returns only output.
Its 2-D branch discards M/L, including all prefill; its 3-D branch exposes only
split-decode scratch. Neither the backend nor reduce_segments returns LSE.
Until upstream exposes LSE, the extra unsplit launch below recovers M/L only;
vLLM's ordinary launcher remains responsible for the returned attention output.
"""
```

输出仍由 vLLM 的 launcher 负责，pLLM 只多跑一次不分段的原始 kernel 来补回 LSE 需要的统计量。当时的报告写得很老实：要去掉这一次额外的 launch，需要上游提供可选的 LSE 输出；“it is not claimed eliminated”。这就是复用失败时应有的样子：把上游缺什么写成一行事实，把自己补的那一小段写成“上游补上就删”的便签。

## 同一个原则，换个地方

“像上游自己那样调用上游”在 pLLM 里不止出现在注意力 kernel 上。退一步看，好几处看起来各不相同的设计，其实是同一件事。

先看 KV 内存管理器。想让某些层只存一个窗口、或者借用别的层的 KV，第一反应是写一个自己的页表分配器。pLLM 一行分配代码也没写：它只在 vLLM 建管理器之前，把每一层改写成 vLLM 已经会管的那种层——窗口层、R-SWA 层、跨层共享——之后的分配、回收、抢占全由原生管理器负责，连准入容量都调用 vLLM 自己的 `max_admission_blocks_per_request`。需要动 vLLM 的地方只有一处：它给 KV spec 分组时只问第一个 spec，于是 pLLM 把 R-SWA 的 spec 排到前面（`sink_first`），并让它只肯和同窗口的层同组（细节见[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)）。这和 FlashInfer 那件事一模一样：不绕开上游的机制，而是用它预期的方式喂给它。

再看融合的 norm。用户改写残差连接时，最朴素的实现会自己做一次加法，再交给下一个 norm；可准入的模型家族都把这个加法融合在下一个 norm 的 `fused_add_rms_norm` 里。pLLM 的做法是用 `make_fx` 看懂用户的方法，如果它最后是一个加法，就把加法从图里删掉，把两个操作数**还给**原生的融合 kernel。在 0.5B 上，这一步（连同删掉端口上多余的拷贝和填零）让一个什么也不做的改写从原生的 89.8% 到了约 99.5%，decode 图每步的 kernel 从 628 个降到 388 个（[《图内挂载点》](05-in-graph-attach-points.md)）。我们没有写一个“改写 + 归一化”的新 kernel，而是让上游已有的 kernel 多干了一点。

抽样原语也是一样。有界的采样引导需要一个自己的 Triton kernel——上游有过滤、有 Gumbel-max，却没有“只为幸存者取 LM head 行再抽样”的入口。但 kernel 的最后一步，抽样本身，我们没有自己写：它调用 vLLM 的 `gumbel_block_argmax`（`pllm/backends/vllm/guided_sampling.py` 的模块 docstring：“This kernel keeps its filter output and calls its Gumbel helper, including the request seed, token position, precision and greedy behavior”）。于是种子语义天然一致：初次实现时的配对比较里，有界路径和 dense 引导在 163,200 个引导行上的抽样全部相同（[《采样尾声融合》](07-sampler-epilogue.md)）。自己写的 kernel 里调用上游的原语，是这个原则最细粒度的形态。

最后一个例子是编译后端，方向相反，也最容易被误解。挂了自定义注意力或残差改写的模型，pLLM 让 vLLM 用 `"eager"` 编译后端录图——录模型的源 CUDA 算子，而不是 Inductor 重新融合的 kernel。2026 年 11 月我们在 7B 上发现，即使 FA2 的读和原生逐元素相同，Inductor 对“挂了 hook 的图”和“原生参考图”融合 add/RMSNorm 的方式也不同，舍入就不同，中性设置下的逐位一致就保不住；换成源算子之后，7B 上反复对比的 logits 和 token id 都逐位一致（[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)）。这看起来是“少用了一个上游工具”，其实是在复用更底层的那个上游：原生参考跑的是这些源算子，我们就跑同样的源算子。代价是放弃融合，0.5B 上约 1.9%，7B 上约 3%。

把这些放在一起看，原则可以说得更精确一点：**复用的单位不是一个函数，而是上游自己的那条调用路径。**调用 FA2 的 `window_size` 也是复用，可它绕过了 vLLM 自己选后端、建 plan、管 graph 的那条路；借用 vLLM 的 builder，才是走上游自己走的路。同样，自己做加法也“用到了” norm，可只有把加法还给 `fused_add_rms_norm`，才是原生模型自己走的路。

## 审计：每个手写实现都要回答“上游哪里有”

这个原则落到工程上，是一次逐项的审计。2026 年 11 月 4 日的那份复用审计，把 `pllm/` 下所有 Python 模块和所有手写 Triton kernel 列成一张表，每一行回答三个问题：pLLM 的实现在哪里、上游对应的东西在哪里、该保留还是替换。挑几行：

| pLLM 实现 | 结论 |
|---|---|
| 全量 / 窗口读的 launcher 参数（`custom_attention.py`） | 保留薄适配层：上游 FA2 计算输出和 LSE；等后端 forward 暴露 LSE 就删 |
| R-SWA 的原始统计量 launch（`rswa_attention.py`） | 等上游加 LSE 输出就替换：2-D 分支丢弃 M/L，当前的回退重复了注意力计算 |
| 查询到请求的映射（`execution/attention.py`） | 用上游的 `searchsorted` 替换二次比较 |
| 分页 AdamW（`kernels/adamw.py`） | 只保留“源槽读、目标槽写”的分页特化；torch 的优化器没有掩码间接写的接口，dense 情形已委托给 torch |
| 描述符指针的行 gather（`kernels/page_copy.py`） | 保留：`index_select` 在录图时绑定张量地址，而这里要在设备上读出当前地址 |
| 每请求 dense/LoRA 投影（`parameter_projection.py`） | 只读静态 LoRA 可以考虑 Punica；PTTS 版本、按头算子和可微训练是 Punica 的推理接口没有的 |

这张表本身就是方法论：手写的东西不是不能有，但每一个都要能说出“上游缺什么”。说不出来的，就应该删。

## 复用的代价，和它做不到的事

复用不是免费的，上面每一处都付了钱。跟随上游的数值语义：FlashInfer 的 BF16 分母、LSE 的换底，“返回 FP32”不等于“和 FA2 逐位相同”，对 LSE 很敏感的算法要把这个量级算进质量验证。接受上游的资源：394 MiB 的共享工作区，不是 pLLM 自己能省掉的。跟随上游的版本：0.27 和 0.32 的 builder 差异关在一个文件里，可每一次升级都要有人看。还有性能：在我们测的形状上，自写的 Triton 单段读比 FlashInfer 快约 11%，整机快 1–2 个百分点，我们放弃了它。

也有没测到的地方。窗口 kernel 的选择只在 Ampere/Ada（capability 8.x）上切换到 FlashInfer，Hopper 上保留 FA2，那边的表现我们没有测过；FlashInfer 在极小窗口上的表现也没有单独测过。R-SWA 那一次为 LSE 多跑的 kernel 还在。逐位中性的代价——放弃 Inductor 融合的那几个百分点——是持续的，只要挂了模型 hook 就一直在付。

## 可以带走的东西

“不要造轮子”是一句谁都会说的话，难的是它在具体情况下指什么。我们从 FA2 那次失败里学到的是：调用上游的某个函数，不一定是在复用上游；**复用的是上游自己的那条路径**——它怎么选后端、怎么建 plan、怎么管 graph、怎么融合 norm、怎么抽样。沿着那条路走，上游所有的优化和维护都自然流到你这边；绕开它，哪怕只调了一个上游函数，你也在和上游平行地维护一条自己的路。

另一个可以带走的是那张审计表的格式：每一个手写实现旁边写上“上游的对应物在哪里、缺什么、什么时候可以删”。写不出这三栏的代码，大概就不该存在。
