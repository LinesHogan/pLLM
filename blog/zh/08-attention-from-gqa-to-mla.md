# 少读 98% 的历史反而慢 5 倍，换上 DeepSeek 又罢工：从显存布局重看自定义注意力

> **TL;DR**：照着单机脚本里 `[batch, heads, seq, dim]` 展开张量设计的自定义注意力，放进生产引擎会连续撞上两堵墙：在分组查询注意力（GQA）模型上裸调标准 `window_size`，会因拆散 7:1 头部共享读取反而比读全量历史慢出 5.5 倍；切到采用多头潜在注意力（MLA）的 DeepSeek V2/V3 上，物理缓存里压根不存在独立的 `Key` 和 `Value`。本文复盘如何用一页纸的带宽极限算术定位 GQA 网格退化，又如何从矩阵乘法的维度复杂度出发推导出潜空间矩阵吸收，最终把两套截然不同的物理存储收进同一个读取抽象。

做长上下文探测、局部/全局交替注意力或者块级稀疏筛选时，研究员在单机脚本里习惯直接伸手去拿 `Key` 和 `Value` 张量，或者给注意力函数传一个 `window_size` 参数。

但只要把这套习惯带进生产级 vLLM 引擎，没走两步就会遇上两件让人摸不着头脑的事：第一件发生在常规的 Qwen 或 Llama 模型上，传进官方滑动窗口参数让算子只读最近 4 个词，单层耗时非但没降，反而比老老实实读完 256 个词**慢了 5.5 倍**；第二件发生在模型切换到 DeepSeek V2 或 V3 时，自定义注意力直接没法跑了——因为物理分页缓存里根本没有 `Key` 和 `Value` 这两张表。

如果孤立地看，一件像是算子库的性能怪癖，另一件像是新模型的兼容缺口；可要是顺着底层显存布局把账排开，会发现它们讲的是同一个系统建模教训：**上层注意力接口一旦绑死了“物理展开的多头键值张量”，只要碰上真实的硬件打包优化或压缩缓存就会失灵。**

---

先复盘第一件事是怎么排查水落石出的：**为什么少读 98% 的数据，耗时反而从 16.6 微秒涨到了 91.5 微秒？**

排查单层算子性能有一个实用的前置技巧：**选对显微镜尺寸**。如果在 7B 或 70B 大模型上测单层注意力，每步二十多毫秒的稠密线性层会把几十微秒的算子退化遮得严严实实；换到单步总耗时只有约 3 毫秒的 `Qwen2.5-0.5B`（每层 14 个 Query 头配 2 个 KV 头，头维 64，128 行并发解码，历史缓存长 256），微小的退化也会立刻在端到端吞吐上显形。

我们把这个形状下的单层解码注意力单独拎出来回放 1,000 次计时：
- 调上游 FlashAttention-2（FA2）读满全部 256 个历史词，单次耗时 **16.6 微秒**；
- 给同一个函数传进 `window_size=(3, 0)` 只读最近 4 个词，单次耗时涨到了 **91.5 微秒**。

先别急着一头扎进几千行 C++ 模板源码里大海捞针。碰到反常耗时，第一步永远是拿信封背面算一下**硬件物理极限（Roofline）**，看看基准那 `16.6 微秒` 到底处在什么水平。

逐字解码每步只进 1 个新词，算力大幅过剩，耗时纯粹由显存搬运量决定。注意 `Qwen2.5-0.5B`（以及 `7B`）采用的是 **7 比 1 的分组查询注意力（GQA）**：14 个 Query 头每 7 个一组，共享完全相同的 1 个 KV 头。在理想情况下，每组历史 K 和 V 只需要从显存读进芯片 **1 次**：

$$\text{有效搬运量} = 128\text{ 行} \times 256\text{ 词} \times 2\text{ 个 KV 头} \times 2\text{ (K与V)} \times 64\text{ 维} \times 2\text{ Bytes} = 16\text{ MiB}$$

除以测试卡约 `1,000 GB/s` 的显存带宽，理论物理极限正好是 **`16 微秒`**！

这道简单的除法立刻给出了破案方向：全量读取测出 `16.6 微秒`，说明它几乎贴在了硬件极限上——而要想贴住极限，内核就**必然**把共享同一组 KV 的 7 个 Query 头打包进了同一个线程块（CTA）里共用片上内存。反过来，只读 4 个词却慢到了 `91.5 微秒`，5.5 倍的倒退幅度正好跟 **7:1 的头共享比例**撞在同一个量级上！

带着这个猜想打开剖析器看一眼两个内核的启动网格（Grid）参数，病因一抓一个准：

```text
1. 全量解码 (按 2 个 KV 头组织网格 —— GQA 打包生效，耗时 16.6 µs):
   ┌──────────────────────┐
   │ 显存中的 KV Head 0 页 │ ──(读取 1 次)──► [ 单个 CTA 内共享: Q0, Q1, Q2, Q3, Q4, Q5, Q6 ]
   └──────────────────────┘

2. 裸调 FA2 窗口解码 (退化为按 14 个 Query 头组织网格 —— 打包失效，耗时 91.5 µs):
   ┌──────────────────────┐   ┌────────────► [ CTA 0: Q0 ]
   │ 显存中的 KV Head 0 页 │───┼────────────► [ CTA 1: Q1 ]  ... (7 个独立块重复拉取同一页缓存)
   └──────────────────────┘   └────────────► [ CTA 6: Q6 ]
```

在这个解码形状下，一旦开启滑动窗口，FA2 的解码路径丢掉了 GQA 头部打包，网格从按 **2 个 KV 头**派发退化成了按 **14 个 Query 头**各自派发，省下的序列长度全赔进了重复访存和窗口边界判定里。

确认了根因是“窗口模式下丢了 KV 头打包”，修法就是换用开启窗口依然按 KV 头组织网格的 FlashInfer 解码内核。但动手集成时如果直接 `import flashinfer` 去调底层裸函数，又会撞上静态图约束：FlashInfer 是两阶段有状态管线，每步执行前得先在 CPU 上跑一步 `plan` 算好分块并写进固定地址的工作区，直接调裸函数就得自己重写半套图缓冲区管理。

更省心的做法是顺着宿主的架构借力：vLLM 内部本来就有现成的 `FlashInferMetadataBuilder` 在管这套固定工作区，只需写十来行代码继承它、把同一份原生页映射喂进去，单次窗口读取当即落回 **18.0 微秒**，整机吞吐恢复到原生的 97% 以上。

顺着宿主借力做滑窗物理显存回收时，还能学到一招排查大型框架配置失效的思路。vLLM 内置了实用的 `RSWASpec`（保留完整提示词前缀 + 最近滑动窗口，滑出中间段的物理页在页表里填上 `null_block` 退回空闲池）。可当初我们只把模型里部分层设成 `RSWASpec`、第 0 层仍保留全量注意力 `FullAttentionSpec` 时，运行期这几层居然一页显存都没回收。

碰到“局部配置被静默忽略”，直接去查宿主合并配置组的入口函数，十行代码就暴露了一个面向对象继承陷阱：

```text
类继承关系：   AttentionSpec ──► FullAttentionSpec ──► RSWASpec (子类!)

vLLM 分组逻辑： 取字典第 0 项 (FullAttentionSpec) 询问：“后面所有层都是 isinstance(..., FullAttentionSpec) 吗？”
               ──► 返回 True！挑剔的窗口子类被宽容的全量父类直接吞并进了全量保留组。
```

看清继承树之后，不用改上游一行源码：派生一个收紧相等条件的子类，再在把配置字典交给引擎前排个序（`sink_first`），把挑剔的 `RSWASpec` 子类挪到字典最前头让它先发言，原生的物理块回收器就正常接管了。

---

理顺了常规 GQA 模型的页表与头部打包，再看切换到 DeepSeek V2 / V3 时撞上的第二堵墙：**当物理缓存里根本不存在 `Key` 和 `Value` 时，怎么建模自定义注意力？**

先把两种架构的物理分页缓存并排画出来对比：

```text
标准 GQA 缓存页 (如 Qwen / Llama):
  每个位置按头分开存： [ KV Head 0: K, V ] [ KV Head 1: K, V ] ...  (存在明确的 K、V 张量)

DeepSeek MLA 压缩潜空间缓存页 (V2 / V3):
  所有注意力头只共享一条低秩向量： [ 共享压缩潜变量 c_KV (如 512 维) | 共享旋转编码 k_rope (64 维) ]
  逻辑还原公式 (共 H 个头)：       K_h^nope = c_KV @ W_UK[h],    V_h = c_KV @ W_UV[h]
```

MLA 之所以能把长上下文显存占用砍掉一大截，正是因为它只存那一条所有头共享的 $c_{KV}$。这时候如果上层自定义注意力接口非要向底层索取展开后的 `[位置数, 头数 H, 头维]` 键值张量，底层就只能在每次读取时把历史 $L$ 个位置的 $c_{KV}$ 全部乘上 $W_{UK}, W_{UV}$ 物化出来——历史长度 $L$ 动辄几千上万，头数 $H$ 在大模型上有上百个，这一乘就把 MLA 省下来的显存带宽全数倒贴了回去。

怎么在不物化 K 和 V 的前提下，算出任意窗口或稀疏 `topk` 位置的精确注意力？

不用发明新算子，直接拿**矩阵维度的计算复杂度**来引导代数变形——这其实跟写 LoRA 时先算 `x @ A` 降维再乘 `B`、以及 DeepSeek-V2 论文里推导推理加速是同一个思路：**利用矩阵乘法结合律交换括号顺序，把昂贵的按头投影从长度为 $L$ 的历史序列挪到长度为 $1$ 的当前查询与输出上。**

对照下面的维度流向图看两次括号移动：

```text
解码单步：当前查询长度 = 1，历史缓存长度 = L (L >> 1)，注意力头数 = H

【打分阶段】
  朴素顺序： q_nope [1, H, 128]  ·  (c_KV [L, 512] @ W_UK [H, 512, 128])^T   ──► 展开 O(L · H) 显存读写!
  结合律后： (q_nope [1, H, 128] @ W_UK^T [H, 128, 512])  ·  c_KV^T [512, L] ──► 仅入口做 O(1 · H) 投影!

【加权求和阶段】
  朴素顺序： sum_t  weights [1, H, L]  ·  (c_KV [L, 512] @ W_UV [H, 512, 128]) ──► 展开 O(L · H) 显存读写!
  结合律后： (weights [1, H, L] @ c_KV [L, 512])  @  W_UV [H, 512, 128]        ──► 仅出口做 O(1 · H) 投影!
```

看清维度之后，代码几乎是照着图抄出来的。配合分块在线 `logaddexp` 归并（每扫过一个 128 词的压缩缓存块就动态更新局部最大值与配分函数，保证任意长序列数值稳定），完整的流式潜空间注意力用普通 PyTorch 写出来只有十来行：

```python
def paged_latent_attention(q_nope, q_pe, latent_pages, rope_pages, w_uk_t, w_uv, scale, tile=128):
    # 1. 入口吸收：只在长度为 1 的当前查询上乘一次 W_UK^T，变到 512 维潜空间
    q_abs = torch.einsum("rhp,hpl->rhl", q_nope.float(), w_uk_t.float())
    acc = torch.zeros_like(q_abs)
    lse = q_abs.new_full(q_abs.shape[:2], -float("inf"))

    # 2. 直接在压缩潜变量页上分块流式打分与累加（或按 topk 索引抽取），零 K/V 物化
    for c_tile, r_tile, valid_mask in iter_compressed_tiles(latent_pages, rope_pages, tile):
        scores = (torch.einsum("rhl,rkl->rhk", q_abs, c_tile)
                  + torch.einsum("rhd,rkd->rhk", q_pe.float(), r_tile)) * scale
        scores = scores.masked_fill(~valid_mask, -float("inf"))
        tile_lse = torch.logsumexp(scores, dim=-1)
        merged = torch.logaddexp(lse, tile_lse)
        finite = torch.where(merged.isneginf(), 0.0, merged)
        weights = (scores - tile_lse[..., None]).exp().masked_fill(~valid_mask, 0.0)
        acc = (acc * (lse - finite).exp()[..., None]
               + torch.einsum("rhk,rkl->rhl", weights, c_tile) * (tile_lse - finite).exp()[..., None])
        lse = merged

    # 3. 出口吸收：历史全部归并完后，只在最终结果上每头乘一次 W_UV 还原输出维度
    return torch.einsum("rhl,hlv->rhv", acc, w_uv.float()).to(q_nope.dtype), lse
```

---

把这两段排障与推导合在一起，就能看明白一个耐用的自定义注意力抽象该怎么划界：**不要把底层的物理缓存数组直接暴露给算法作者，而要抽象出与存储格式无关的“注意力动作（Capability）”。**

在我们架在 vLLM 上的开源可编程层 **[pLLM](https://github.com/LinesHogan/pLLM)**（`pllm`）里，`@pllm.custom_attention` 传给用户方法的是一个屏蔽了物理布局的 `PagedKV` 对象：

```python
@dataclass(frozen=True)
class WindowProbe(pllm.Workload):
    @pllm.custom_attention(layers=(0,))
    def monitor(self, q: torch.Tensor, kv: pllm.PagedKV):
        out, full_lse = kv.attend(q)
        _, recent_lse = kv.attend(q, window=64)
        return out, recent_lse - full_lse  # 同一份代码，跨 Qwen/Llama (GQA) 与 DeepSeek (MLA) 通用
```

同一段八行探针代码，挂在 Qwen 或 Llama 上自动走预规划好的 FlashInfer GQA 打包内核，换到 DeepSeek V2/V3 上自动切入潜空间矩阵吸收，两边都能直接录进 CUDA Graph。

当然，前面那张维度流向图也顺便把这套抽象的代价边界标在了明面上：一方面，启用 vLLM 的 FlashInfer 规划管线需要预先留出约 `394 MiB` 的共享工作区显存；另一方面，仔细看 MLA 结合律前后的复杂度对比——矩阵吸收之所以划算，靠的是历史长度 $L \gg 1$。如果你的稀疏挑选算法极端激进，每步 `topk` 只挑两三个历史位置（$L \approx 2$），那么在入口和出口固定支付的那两次全头矩阵投影（$W_{UK}^\top$ 与 $W_{UV}$），算力开销反而会超过直接解压那两三个位置。先画清数据维度流向图，什么时候省钱、什么时候亏本，不用上机跑心里就有一本明白账了。
