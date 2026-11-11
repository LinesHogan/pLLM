# 你其实不需要读 `attn_metadata`：用三个整数算清 vLLM 的打包行布局

> 2026-11-10 · pLLM 提交 `4d571723` · 测量环境：除文中另注外为单张 RTX 4090（24 GiB）、vLLM 0.32.0、Qwen2.5-7B-Instruct，8 条 prompt × 16 个样本、生成 256 token。
> 这是一篇有日期的文章，不随代码更新；当前的用法以[用户指南](../../README.md#用户指南)为准，当前的实测数字以[性能与显存](../performance.md)为准。

如果你在单卡研究脚本里写过 PyTorch 模型钩子（hook），你眼里的 batch 一定是个规规矩矩的三维盒子：`[batch_size, seq_len, hidden_size]`。`x[0]` 永远属于第一个请求，`x[1]` 永远属于第二个请求，门牌号清清楚楚。

但当你把同样的钩子挂进 vLLM 的第 0 层，打印出传进来的 `hidden` 张量形状时，那个三维盒子不见了。你手里只有一个拍扁的二维张量：`[64, 896]`（下文探针用的是隐藏维 896 的 `Qwen2.5-0.5B-Instruct`）。

```text
研究脚本里的张量 [B, S, H]：              vLLM 传给模型层的张量 [num_tokens, H]：
┌───────────────────────────┐            ┌─────────────────────────────┐
│ Req 0_0 (decode): [tok]   │            │ Row 0:      [ ??? ]         │
├───────────────────────────┤            │ Row 1:      [ ??? ]         │
│ Req 1_0 (decode): [tok]   │    ───►    │ Row 2:      [ ??? ]         │
├───────────────────────────┤            │ Row 3..63:  [ ??? ]         │
│ Req 2 (prefill):  [t0..60]│            │ (特定步末尾还可能跟着 padding)│
└───────────────────────────┘            └─────────────────────────────┘
```

这是现代推理引擎的**连续批处理（Continuous Batching）**与**分块预填充（Chunked Prefill）**在起作用。为了不让短请求干等长请求，调度器在每一步（engine step）为每个活跃请求决定本步要算几个 token：正在逐字生成的请求通常算 1 个 token（decode），刚进门的长文档请求则切出一截来算（prefill chunk）。接着，引擎把所有请求在本步要算的 token 首尾相接，拼成一条紧致的二维长条送进 Transformer；为了命中预先录好的 CUDA Graph 尺寸，有时还会在张量末尾补上几行凑数的 padding。到了下一步，旧请求退场、新请求插队，所有人在张量里的行号又会彻底洗牌。

如果你只是用 vLLM 生成文本，这层拼接完全是透明的。但只要你想在推理过程中维护一点“与具体请求绑定”的状态——比如为每个请求在线微调一个轻量级的引导网络（如 ESamp）、维护每请求的外部记忆、或者按请求收集隐藏层特征——你在动手算任何数学之前，都必须先搞清楚一件事：

**我手里的这一行，到底属于哪个请求？它是正在消化提示词（prefill），还是在生成新词（decode）？**

这篇文章要讲的就是怎么解开这道题。很多人在改 vLLM 时，第一反应是一头扎进注意力层的元数据结构体 `attn_metadata` 里去扒字段；但我们会拿一个 64 行的真实批次摆在桌面上，一行一行算给你看：为什么你其实根本不需要碰 `attn_metadata`，只靠装车前 CPU 上的**三个标量整数**和一条水位线，就能用初等加减法把所有行的来龙去脉算得一清二楚。

至于那个让无数开发者抓狂的 `attn_metadata`，我们也用探针把它在各个版本和后端下的真实长相完整扒了下来，整理在[文末附录](#附vllm-attn_metadata-字段实测解剖笔记)里，供需要深入 vLLM 底层的朋友备查。

---

## 为什么去扒 `attn_metadata` 是一条弯路？

当你在模型第 0 层拿到拍扁的 `[num_tokens, hidden]` 张量，想要找回每一行的归属时，开源社区里常见的现成做法是走 `LogitsProcessor`。

但看一眼时间线就知道这条路走不通：`LogitsProcessor` 要等整座数十层的 Transformer 全部跑完、连最后那层庞大的 `lm_head` 词表投影都乘完了，才会被引擎调用。如果你打算把第 0 层的隐藏状态捞出来，放到旁路 CUDA Stream 上与模型后续层的前向计算并行做在线训练或预取，等到 `LogitsProcessor` 才认出哪一行是谁，黄花菜都凉了。

要在模型前向开始之前（或者在第 0 层当场）拿到行拆包信息，顺着调用栈很容易盯上注意力层。毕竟，Transformer 的线性层和归一化层都是逐行独立计算的，唯独注意力（Attention）层绝对不能让张三的 token 去读李四的 KV Cache——注意力层手里一定攥着拆包的“户口本”。

翻开 vLLM 的注意力调用链路，底层计算被封装在 C++/CUDA 内核里，Python 层唯一能看到的入口参数就是一个叫 **`attn_metadata`** 的结构体。

可一旦你试图靠解析 `attn_metadata` 来还原行身份，就会接连踩进两个泥潭：

第一，**它的张量在 GPU 上，且格式随注意力后端和运行模式剧烈变化**。`attn_metadata` 是专门为底层 CUDA 注意力内核量身定制的参数包。如果在模型前向途中把里面的 `query_start_loc`、`seq_lens` 等张量拷回 CPU 解析，再把挑出来的行号传回 GPU，就会引入昂贵的主机-设备同步（Host-Device Synchronization），还会直接撞碎要求地址固定的 CUDA Graph 录制。哪怕你费尽周折设计一套 CPU 与 GPU 协同的轮询缓冲区，还会发现 FlashAttention 后端和 FlashInfer 后端的字段定义截然不同，甚至连“哪些请求算 prefill、哪些算 decode”、“总 token 数包不包含图 padding 行”在不同模式下都有两套口径（附录会展示这些暗坑）。

第二，**怎么确认自己没有在某个边界条件下挑错一行？** 在混排了分块预填充、多样本并行采样（`n > 1`）和图 padding 的扁平张量里，哪怕你的拆包下标算术在某个角落偏了一行，代码也不会报错崩溃，只会悄无声息地拿李四的激活去更新张三的状态。

要验证拆包逻辑到底对不对，最可靠的办法是建立一套不依赖任何内部假设的“金标准对照协议”（出自验证脚本 `tllm/legacy/verify_v1_decode_rows_runtime.py` 的文档字符串）：

```python
"""Verify vLLM v1 decode-row localization using first-layer hidden MSE.

Gold standard:
- Run prompts one-by-one (greedy)
- Capture first transformer layer output hidden for decode tokens only

Candidate:
- Run the same prompts in one parallel batch (greedy)
- Use v1 logits_indices to select rows from first-layer hidden

Pass condition:
- Per prompt, per decode step MSE(gold, parallel) <= mse_tol
"""
```

这个验证思路在任何批处理系统里都值得借用：先把一批测试请求**一个接一个单独跑**（`batch_size = 1`，贪心解码）。当引擎里只有一个请求时，根本不存在混排，第 0 层输出的 decode 隐藏状态毫无歧义；把它存下来作为金标准（Gold standard），再把同一批请求**同时扔进引擎并行混排**，用你的拆包逻辑挑出各请求在每一步的对应行，逐请求、逐步比对两边的均方误差（MSE）。只要有一处对不上，就说明对行布局的理解还有死角。

那么，有没有办法彻底绕开下游复杂的 `attn_metadata`，在包裹被拍扁装进 GPU 之前就把账算清？

---

## 桌面手算：装车前的三个整数推导一切

让我们退后一步，去看看调度器刚刚排好班、还没开始构造 `attn_metadata` 的那一刻。

下面这个场景取自我们在 vLLM 0.32.0 上的真实运行探针（加载 `Qwen2.5-0.5B-Instruct`，故意将单步最大 token 数设为 `max_num_batched_tokens = 64` 以触发分块预填充）。此时引擎里同时跑着三个用户请求：
* **请求 0**：提示词（prompt）长 5 个 token，但用户设置了并行采样 `n = 2`，因此 vLLM 在内部将它裂变为两个独立的子请求 `0_0` 和 `1_0`；
* **请求 1**：一段长 2 个 token 的短提示词；
* **请求 2**：一段长达 150 个 token 的长提示词。因为每步最多只能算 64 个 token，它必须被切成好几块、分好几步算完。

现在我们把镜头定格在运行的**第 2 步**。在这一步里，前三个短请求都已经算完了各自的 prompt，正在一步吐 1 个字地做 decode；而请求 2 在第 1 步刚算完前 52 个 token，这一步正接着算后续的 61 个 token。

送进模型的扁平张量一共 `1 + 1 + 1 + 61 = 64` 行：

```text
扁平张量行号     所属请求     本步行数
Row 0           请求 0_0     1 个 token
Row 1           请求 1_0     1 个 token
Row 2           请求 1       1 个 token
Row 3..63       请求 2       61 个 token
```

面对这 64 行，我们需要搞清楚四件事：
1. **每一行属于哪个请求身份？**
2. **每一行处于什么阶段？**（是正在读提示词的 prefill token，还是已经开始生成的 decode token？）
3. **哪一行是提示词刚刚读完、需要触发首次采样的“prompt 结束行”？**（注意请求 2 在本步占了 Row 3 到 Row 63，它的最后一行 Row 63 该不该触发采样？）
4. **哪些行是因为 KV Cache 显存不足被抢占（preempt）驱逐后，从头重算的旧历史？**

要回答这四个问题，我们根本不需要看 GPU 上的任何张量。因为在 CPU 的 `InputBatch` 里，每个被调度的请求本来就整齐地躺着**三个标量整数**：

* `scheduled`：本步调度器决定为该请求计算几个 token；
* `prompt`：该请求原始提示词的总长度；
* `computed`：在今天这一步开始**之前**，历史上已经为该请求计算过几个 token。

把第 2 步这四个请求的三个整数列在桌面上，就是这样一张只有 12 个数字的小表：

```text
请求      scheduled (本步算几个)   prompt (提示词总长)   computed (此前已算几个)
0_0              1                       5                      5
1_0              1                       5                      5
1                1                       2                      2
2               61                     150                     52
```

现在，顺着这张表从上往下走一遍，对于某个请求在本步被调度的第 `token` 个位置（`0 <= token < scheduled`），它在整条序列里的**绝对逻辑位置（`offset`）**就是一道加法：

$$\text{offset} = \text{computed} + \text{token}$$

一旦算出 `offset`，前面所有的疑问瞬间就解开了：

1. **看请求 `0_0`（Row 0）**：本步算 1 个词（`token = 0`），它的 `offset = 5 + 0 = 5`。拿去跟它的 `prompt` 长度（`5`）比一下：`offset >= prompt`（`5 >= 5`）。因为 5 个提示词的下标是 `0, 1, 2, 3, 4`，下标 `5` 说明提示词早就翻篇了——**Row 0 是如假包换的 decode 生成行**。同理，Row 1（请求 `1_0`，`5 >= 5`）和 Row 2（请求 `1`，`2 >= 2`）也全都是最新的 decode 行。
2. **看请求 `2`（Row 3..63）**：本步算 61 个词（`token` 从 `0` 到 `60`），此前已算 `computed = 52` 个。那么这 61 行对应的 `offset` 就是从 `52 + 0 = 52` 一路排到 `52 + 60 = 112`。拿去跟它的 `prompt` 长度（`150`）比一下：全都严格小于 `150`！所以 Row 3 到 Row 63 **全部处于 prefill 提示词阶段**。
3. **Row 63 是不是请求 `2` 的“prompt 结束行”？** 所谓提示词刚刚算完，充要条件只有一个：当前 token 恰好落在了提示词的最后一个位置上，即 **`offset == prompt - 1`**（对请求 `2` 来说就是 `149`）。而在第 2 步里，Row 63 的 `offset` 才走到 `112`，离 `149` 还差着 37 个词呢！
   这一点极其关键：如果你去翻这一步 vLLM 内部的采样索引 `logits_indices`（见文末附录第 2 步），你会震惊地发现里面赫然写着 `[0, 1, 2, 63]`——vLLM 照样把还未算完提示词的 Row 63 送进 `lm_head` 算了一次 logits，然后再把结果丢弃。如果你盲目相信 `logits_indices` 来判断预填充是否结束，就会在分块预填充的半山腰错误地触发状态更新或特征导出；而用 `offset == prompt - 1` 这道减法，只有在**第 3 步**（此时 `computed = 113, scheduled = 37`，最后一个词 `offset = 113 + 36 = 149`）时，才会唯一、精确地命中真正的 prompt 结束行。

---

## 加一条水位线 `frontier`：一眼认出抢占重算

现在还剩第四个问题：**KV Cache 抢占重算（Preemption Replay）**。

在高并发线上服务里，显存预算总是紧巴巴的。当几个长生成请求把物理 KV 块吃光时，vLLM V1 的调度器会临时驱逐某个正在生成的请求，释放它的 KV 块；等过几步显存宽裕了，再把这个请求的原始 prompt 连同它被踢走前已经生成的 token 拼在一起，**从位置 0 开始重新做一遍 prefill**，把 KV Cache 重建出来。

对于无状态的纯推理来说，重算一遍毫无副作用。但如果你的扩展逻辑是有状态的——比如每读过一个新 token 就把特征导出给外部，或者像在线蒸馏那样用每个 token 的隐藏状态去训练请求专属的权重（且策略是在抢占时保留已训好的权重 `keep`）——如果不知道这是一次“回放重算”，你就会把历史 token 当成新数据再训练一遍、再导出一次！

怎么在不改动调度器的前提下，把这些“炒冷饭”的重算行精准筛掉？

我们只需要在每个请求进入系统时，为它多记一个标量整数：**`frontier`（历史最高水位线）**——表示这个请求自打进门以来，已经推进到过的最大逻辑位置 `computed + scheduled`。

```text
请求 2 的生命周期位置轴（prompt = 150）：

正常算完第 2 步后，水位线推到 113：
  0                52                   113                150
  ├─────────────────┼────────────────────┤                  │
  │   Step 1 (52)   │    Step 2 (+61)    ▲                  │
  └─────────────────┴────────────────────┴──────────────────┘
                                     frontier = 113

遭遇 KV 抢占！稍后从 0 重新 prefill 前 64 个词 (computed = 0, scheduled = 64)：
  0                         64          113                150
  ├──────────────────────────┼───────────┤                  │
  │ offset 0..63 < frontier! │           ▲                  │
  │  ──► 标记为 RECOMPUTED    │       frontier = 113         │
  └──────────────────────────┴──────────────────────────────┘
```

看上面的数轴：请求 2 正常跑完第 2 步后，它的 `frontier` 从第 1 步留下的 `52` 更新成了 `max(52, 52 + 61) = 113`。假设此时显存告急，请求 2 被驱逐；几步之后它回来重算第一块（`computed = 0, scheduled = 64`）。当我们遍历这 64 行时，算出来的 `offset` 是 `0..63`——**它们全都严格小于 `frontier`（`113`）！**

只要看到 `offset < frontier`，我们就立刻知道：“这一行以前已经见过面了，现在只是在重算 KV。”直接给它打上 `RECOMPUTED` 标记，不让它进入本步的新鲜行（`fresh`）、`prefill` 或 `decode` 集合，直到重算的游标重新跨过 `113` 那条水位线为止。

你看，连续批处理就像一辆把所有人包裹拆散混装的快递货车。与其跳上已经开动的货车，去拆解给底层 CUDA 内核看的复杂二进制包（`attn_metadata`），不如直接在发车前的调度室桌子上，看一眼那张印着 `(scheduled, prompt, computed)` 的发货单。三个整数加一条水位线，连 GPU 的一根毫毛都不用碰，所有包裹的归属就已经明明白白了。

---

## 选最稳的挂载接缝，用 30 行 Rust 一次算完

想通了这套算术，代码该挂在 vLLM 的哪个位置？

在给宿主系统写扩展时，有一条很重要的经验：**不要去钩那个信息最全、但签名随版本频繁变动的内部大函数；要去钩那个处于数据流上游、签名最简单且跨版本最稳定的窄接口。**

在 vLLM 的新版模型执行器（MRv2 `GPUModelRunner`）里，每一步当 `prepare_inputs` 刚刚在 CPU 上组装好 `InputBatch`、但**还没有**开始构建 `attn_metadata` 和发射模型前向之前，都会调用一次 `prepare_attn(self, input_batch: InputBatch)`。在 pLLM 支持的两个 vLLM 版本（0.27.1 与 0.32.0）上，它的函数签名分毫不差。在我们的开源扩展框架 pLLM 中，行标签的计算就挂在这里（`pllm/backends/vllm/runner.py`）：

```python
def prepare_attn(self, input_batch: InputBatch) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    # ……
    session = self.workload_session
    if session is not None:
        # ……
        session.prepare_input(
            encode_mrv2_rows(
                input_batch,
                self.req_states.prompt_len.np[input_batch.idx_mapping_np],
                session.row_labeler,
                step_id=session.next_step_id,
                keep=session.keep_replayed_state,
            )
        )
    return super().prepare_attn(input_batch)
```

`encode_mrv2_rows` 从 `input_batch` 里借出 `num_scheduled_tokens`、`prompt_lengths`、`num_computed_tokens_np` 以及每个请求在持久表里的整数槽位号（请求字符串 ID 在准入时就已编码成整数，热路径上绝不碰 Python 字符串；同时用断言守住 `num_draft_tokens == 0` 等调度前提，并只取真实请求数 `num_reqs`，从而天然滤除尾部的图 padding 行），直接交给 `rust/pllm_core/src/packed_rows.rs` 里的 `RowLabeler`。

来看看 Rust 里的核心循环——它几乎就是我们刚才桌面手算过程的逐字翻译：

```rust
        for (request, (((&count, &length), &done), &id)) in scheduled
            .iter()
            .zip(prompt)
            .zip(computed)
            .zip(ids)
            .enumerate()
        {
            let frontier = self.frontiers[id as usize].unwrap();
            for token in 0..count {
                let packed = start + token as usize;
                let offset = i64::from(done) + i64::from(token);
                let replay = keep && offset < frontier;
                // ...
                if !replay {
                    fresh.push(packed as i64);
                }
                if !replay && offset == i64::from(length) - 1 {
                    prefill.push(packed as i64);
                }
                if !replay && offset >= i64::from(length) && token == count - 1 {
                    decode.push(packed as i64);
                }
            }
            start += count as usize;
            links.push(start as i64 - 1);
            if keep && count > 0 {
                self.frontiers[id as usize] =
                    Some(frontier.max(i64::from(done) + i64::from(count)));
            }
        }
```

这段不到 30 行的循环只读取只读的 NumPy 数组切片，完全不碰 PyTorch 张量或 CUDA 上下文，因此在进入计算时直接通过 `py.allow_threads(...)` 释放了 Python 的 GIL。一次线性扫描下来，它同时产出：
1. 一张 7 列的完整标签矩阵（打包行号 `PACKED`、批内请求序号 `ORDINAL`、逻辑位置 `OFFSET`、阶段 `PHASE`、身份 `IDENTITY`、提示词长 `PROMPT_LENGTH`、重算标记 `RECOMPUTED`）；
2. 四组预先筛好的活跃行下标集合：本步最新的解码行 `decode`、恰好完成提示词的 `prefill`、两者之并 `both`、以及所有非重算的新鲜 token `fresh`；
3. 每个请求在本步的最后一行 `links`（用于对齐采样器行），以及将打包行号映射回槽内名次（rank）的查找表。

把行标签抽象成只吃 `(scheduled, prompt, computed)` 的纯函数，还带来了一个意想不到的工程红利：**生产环境与离线 CPU 测试可以用同一套代码**。在[第 10 篇：连续批处理的隔离测试](10-isolation-testing.md)里你会看到，当我们在没有 GPU 的笔记本上测试用户算法是否满足“分块不变性”与“抢占重放不变性”时，根本不需要写一套模拟 vLLM 的假调度器，只需喂给同一个 `RowLabeler` 几行手写的三元组数据就够了。

---

## 身份粒度、多卡同步与真实开销

当底层用三个整数把每一行的来龙去脉算清楚后，上层的所有扩展接口都可以围绕**“第 0 维永远是对齐好的有效行 `[rows, ...]`”**来构建：读取隐藏状态默认拿到 `decode` 集合（每请求 1 行），需要在提示词读完瞬间做汇总的拿到 `prefill` 集合（哪怕长提示词被切成十块，也只在最后一块完成的那一步出现唯一一次），而全量特征导出则拿到剔除了抢占重算的 `fresh` 集合。算出来的行索引和状态槽号会被打包进一张固定地址的 `int64` 小表异步上传到 GPU，驱动 CUDA Graph 内的算子（详见[第 02 篇：What Color is Your Tensor?](02-tensor-colors.md)与[第 03 篇：给 GPU 显存加上 MVCC](03-gpu-mvcc.md)）。

在实际系统中，这套行模型还有三笔值得算清楚的账：

**第一，把“行的身份”定义准，本身就能省下大笔算力。**
考虑一个常见的并行采样场景：8 条并发 prompt，每条要求采样 `n = 16` 个候选回答，批次里一共 `8 × 16 = 128` 个子请求行。如果机械地按子请求 ID 绑定状态，在线训练每一步都要更新 128 份独立的参数；但实际上，同一个 prompt 裂变出的 16 条采样分支共享着完全相同的提示词起点。由于请求身份在准入时就编码了 `(request_id, prompt_index, sample_index, route)`，只需声明按 `key_fields=("prompt_index",)` 路由，同一个 prompt 下的 16 行就会自动归并到同一份状态上：16 行各自照常算出损失，再按状态求均值（`reduction="mean"`），每份状态每步只做一次优化器更新，需要维护和更新的状态从 128 份缩减到 8 份。

**第二，多卡张量并行（TP > 1）下的零通信对齐。**
当模型切分在多张 GPU 上运行（如 TP = 2 或 TP = 4）时，需要跨卡同步行标签吗？完全不需要。因为在 vLLM 中，所有 TP rank 在每一步都会收到完全相同的调度指令和 `InputBatch` 顺序，每个 worker 进程上的 `RowLabeler` 各自独立跑一遍上面那 30 行确定性整数算术，算出来的行标签天然逐字节一致（文末附录在两张 A100 上实测了两卡收到的 `InputBatch`，探针打印的字段完全相同）。

**第三，CPU 准备开销与调度边界。**
虽然省掉了昂贵的 GPU-CPU 张量同步，但每一步在 CPU 上借用 NumPy 列、跑一遍 Rust 标签器并填充路由表，依然有固定的宿主端耗时。在单张 NVIDIA RTX 4090（24 GiB GDDR6X）上对 `Qwen2.5-0.5B-Instruct` 跑 ESamp（128 并发序列，$\beta = 0$，开启 CUDA Graph，vLLM 0.27.1）做宿主侧剖析，每次调用的中位耗时分别是：`encode_mrv2_rows` 约 **122 µs**，会话 `prepare_input` 约 **174 µs**，准备隐藏层抓取槽位 `HiddenTaps.prepare` 约 **110 µs**。
这段耗时只取决于批次里的请求数与行数，与模型大小无关。在单步 GPU 前向只有 **3.0 ms**（原生吞吐 42,355 tok/s）的 0.5B 小模型上，这四百多微秒的主机工作在剖析时间线上清晰可见；而当切换到单步耗时 **21.7 ms**（原生吞吐 5,894 tok/s，vLLM 0.32.0，CUDA Graph 与 FlashInfer 采样器开启）的 `Qwen2.5-7B-Instruct` 时，几百微秒的 CPU 准备工作相对整步就微不足道了。

当然，这套极简算术也有明确的适用边界：它建立在“每个处于生成阶段的请求在本步最多产出 1 个最新 decode 行、且每个请求在本步恰好对应 1 个采样位（`cu_num_logits_np == 0..count`）”的前提之上。一旦开启**投机解码（Speculative Decoding）**，单个请求在一步内会同时塞入多个草稿（draft）token 并产生多个采样位，三个整数的简单映射就不再成立（此时入口处的 `assert input_batch.num_draft_tokens == 0` 会直接报错拦截）。同样，如果你挂载的逻辑会因请求而异地改写模型内部的残差流或 KV 值，那么“相同 token 前缀必然产生相同 KV”的假设被打破，就必须关闭或隔离 vLLM 的前缀缓存（Prefix Caching）。

---

## 附：vLLM `attn_metadata` 字段实测解剖笔记

虽然我们通过退回上游的 `prepare_attn` 避开了直接解析 `attn_metadata`，但在很多深入定制注意力内核或调试 vLLM V1 引擎的场景里，你依然免不了要和 `attn_metadata` 打交道。偏偏这个结构体的运行态行为极少有公开文档说明。

为此，我们写了一段轻量探针脚本包住 MRv2 runner 的 `model_state.prepare_attn`，在 **vLLM 0.27.1 与 0.32.0** 两个版本、**FlashAttention 与 FlashInfer** 两个注意力后端、以及单卡 RTX 4090 与双卡 A100（TP = 2）上实测了它的每一个字段，整理成这份速查图鉴。下面的快照和速查表都由仓库里的 `tools/probe_attention_metadata.py` 打印的结果整理而来；升级 vLLM 之后，用 `python tools/probe_attention_metadata.py --backend FLASH_ATTN`（换成 `--backend FLASHINFER` 看另一个后端，双卡加 `--tp 2`）重跑一遍，就能核对这些字段含义是否仍然成立。

### 1. 探针实验设置

开启新版执行器（`VLLM_USE_V2_MODEL_RUNNER=1`），加载 `Qwen2.5-0.5B-Instruct`，设置 `max_num_seqs = 8`、`max_num_batched_tokens = 64`（强制触发分块 prefill），关闭 prefix caching，开启 CUDA Graph。同时提交三个请求：
* **请求 0**：`"Hello, my name is"`（5 个 token），并行采样 `n = 2`，各生成 5 个 token（内部裂变为 `0_0` 和 `1_0`）；
* **请求 1**：`"Paris is"`（2 个 token），生成 5 个 token；
* **请求 2**：一段 150 个 token 的长提示词，生成 2 个 token。

### 2. 前五个真实步的逐帧快照

下表为 vLLM 0.32.0 在 FlashAttention 后端下跑出的前 5 步真实快照（0.27.1 除 `idx_mapping` 持久槽号的具体数值及整数位宽外完全一致）。表中 `computed` 即 `num_computed_tokens`，`prefilling` 即 `is_prefilling`：

| 步 | 图模式 | 批内请求排列顺序 | scheduled | computed | query_start_loc | seq_lens | logits_indices | prefilling |
|---:|---|---|---|---|---|---|---|---|
| 1 | NONE (eager) | `1, 0_0, 1_0, 2` | `2, 5, 5, 52` | `0, 0, 0, 0` | `0, 2, 7, 12, 64` | `2, 5, 5, 52` | `1, 6, 11, 63` | `T, T, T, T` |
| 2 | NONE (eager) | `0_0, 1_0, 1, 2` | `1, 1, 1, 61` | `5, 5, 2, 52` | `0, 1, 2, 3, 64` | `6, 6, 3, 113` | `0, 1, 2, 63` | `F, F, F, T` |
| 3 | NONE (eager) | `0_0, 1_0, 1, 2` | `1, 1, 1, 37` | `6, 6, 3, 113` | `0, 1, 2, 3, 40` | `7, 7, 4, 150` | `0, 1, 2, 39` | `F, F, F, T` |
| 4 | FULL (graph) | `0_0, 1_0, 1, 2` | `1, 1, 1, 1` | `7, 7, 4, 150` | `0, 1, 2, 3, 4` | `8, 8, 5, 151` | `0, 1, 2, 3` | `F, F, F, F` |
| 5 | FULL (graph) | `0_0, 1_0, 1, (pad)` | `1, 1, 1` | `8, 8, 5` | `0, 1, 2, 3, 3` | `9, 9, 6, 0` | `0, 1, 2` | `F, F, F` |

对照这 5 行数字，可以直观看到六个光读静态源码极易踩坑的细节：

1. **批内请求顺序是被 `sort_batch_req_ids` 重排过的**：
   第 1 步四个请求都在做 prefill，它们并不是按提交顺序排的，而是按本步 `scheduled` 长度从小到大升序排列（`2, 5, 5, 52`）。到了第 2 步，本步长度为 1 的三个 decode 请求（`0_0, 1_0, 1`）被排在最前面，长度为 61 的分块 prefill 请求 `2` 排在最后——因为底层切分 decode 与 prefill 的逻辑强依赖“短 query（decode）排在最前”的布局。
2. **`logits_indices` 包含尚未完成分块 prefill 的中间行**：
   看第 1 步和第 2 步的 `logits_indices`：请求 `2` 的 150 个提示词才算到第 52 和第 113 个，但它的末尾行 `63` 依然出现在 `logits_indices` 里参与了词表投影。直到第 3 步算完最后 37 个词（`seq_lens` 达到 `150`），`logits_indices` 里的 `39` 才是真正有效的首次采样行。
3. **`seq_lens` 是“算完本步之后”的 KV 总长**：
   在每一步里，`seq_lens` 严格等于 `computed + scheduled`，即本步注意力内核需要扫描的有效 KV 长度。
4. **“谁是 prefill、谁是 decode”在两个后端里有不同口径**：
   公共底座里的 `CommonAttentionMetadata.is_prefilling` 按**真实生命周期**判定（`num_computed_tokens < num_prompt_tokens`）。但切换到 FlashInfer 后端时，`FlashInferMetadata` 里的 `num_decodes` 和 `num_prefills` 是按**本步 query 长度**判定的（凡本步长度 $\le$ 阈值 1 的统统归入 decode）。如果一个长提示词分块到最后一步恰好只剩 1 个 prompt token，FlashInfer 会把它当成 decode 处理（short extend），而 `is_prefilling` 仍为 `True`。
5. **`FULL` 图模式下，名叫“实际 token 数”的 `num_actual_tokens` 包含了图 padding！**
   看第 5 步：请求 `2` 已生成完毕退场，场上只剩 3 个真实请求、共 3 个真实 token（`InputBatch.num_reqs = 3, num_tokens = 3`）。为了命中尺寸为 4 的 `FULL` CUDA Graph，批次被补齐到了 4（`num_reqs_after_padding = 4, num_tokens_after_padding = 4`）。此时 `query_start_loc` 末尾重复了一次 `3`（`[0, 1, 2, 3, 3]`，表示第 4 个幽灵请求长度为 0），`seq_lens` 末尾补 `0`，`slot_mapping` 末尾填 `-1`（指示内核跳过 KV 写入）；然而 `attn_metadata.num_actual_tokens` 的值却是 **`4`**！源码里该字段旁留有一行注释：`TODO(lucas): rename to num_tokens since it may be padded and this is misleading`。在 MRv2 的整图模式下，切勿用 `num_actual_tokens` 过滤有效行，必须读 `InputBatch.num_tokens`。
6. **`n > 1` 采样在未开前缀缓存时的物理槽位分布**：
   `slot_mapping` 记录每个 token 写入分页 KV 池的物理槽位（`物理块号 × 块大小 16 + 块内偏移`）。第 1 步中，请求 `1` 写入槽位 `48, 49`，而请求 `0` 裂变出的 `0_0` 和 `1_0` 分别写入 `16..20` 和 `32..36`——关闭前缀缓存时，两个分支各自独立计算并存储了相同的 5 个提示词 KV。

### 3. 核心结构体字段速查表

**公共底座 `CommonAttentionMetadata`**（每步构造一份）：

| 字段名 | 实测含义与使用注意 |
| :--- | :--- |
| `query_start_loc` / `query_start_loc_cpu` | 形状 `(num_reqs + 1,)` 的 `int32` 张量（GPU/CPU 各一），记录每个请求在扁平批次中的起始行偏移（前缀和）。`FULL` 图模式下包含末尾长度为 0 的 padding 段。 |
| `seq_lens` | 形状 `(num_reqs,)` 的 GPU `int32` 张量，值为 `computed + scheduled`；padding 请求处填 `0`。 |
| `num_reqs` | 请求数；在 `FULL` 图模式下为**补齐后**的请求数（如第 5 步为 `4` 而非 `3`）。 |
| `num_actual_tokens` | 扁平张量总行数；**`FULL` 图模式下包含尾部 padding 行**，真实有效行数请读 `InputBatch.num_tokens`。 |
| `max_query_len` / `max_seq_len` | 本步各请求的最大单步 query 长度与最大 KV 长度。 |
| `block_table_tensor` | `[num_reqs, max_blocks]` 的 GPU `int32` 张量，记录每个请求分配到的物理 KV 页表块号。 |
| `slot_mapping` | 长度为 `num_actual_tokens` 的 GPU `int64` 张量，记录本步各 token 在 KV 池中的物理写入槽位；padding 行固定为 `-1`。 |
| `positions` | 长度为 `num_actual_tokens` 的 GPU `int64` 张量，即各 token 在各自序列中的逻辑位置 `offset`。 |
| `is_prefilling` | 形状 `(真实请求数,)` 的 CPU `bool` 张量，按 `computed < prompt` 标记真实生命周期阶段。 |
| `seq_lens_cpu_upper_bound` | `seq_lens` 在 CPU 侧的上界拷贝（无投机解码时即精确值）。 |
| `req_idx`（仅 vLLM 0.32） | `int64` NumPy 数组，记录每个真实请求在 runner 持久表中的槽位下标，等同于 `InputBatch.idx_mapping_np`（该数组在 0.27.1 中为 `int32`，0.32.0 中升级为 `int64`）。 |

**各注意力后端专属 Metadata 的差异**：
* **`FlashAttentionMetadata`**：除继承公共底座的核心张量与 FA3 调度元数据外，需特别注意其内部的 `num_decode_reqs`、`num_prefill_reqs`、`num_decode_tokens`、`num_prefill_tokens` 在该后端下**恒为 `0`**，切勿用来判断阶段！（此外 0.27.1 中的 `use_cascade` / `common_prefix_len` 在 0.32.0 中被删掉了，而 R-SWA 相关的 `rswa_*` 字段两版本都有。）
* **`FlashInferMetadata`**：结构体中不再保留 `seq_lens` 和 `block_table`，因为它们在 build 阶段已被直接编译（plan）进 `prefill` 和 `decode` 两个 FlashInfer wrapper 对象内部；其 `num_decodes / num_decode_tokens / num_prefills / num_prefill_tokens` 反映了按 query 长度切分的真实数目。

### 4. 多卡并行（TP / DP / DCP）下的表现

在两张 A100-SXM4-80GB 上对 **张量并行（TP = 2）** 进行的全组合实测表明：在每一步，Rank 0 与 Rank 1 上探针打印出的 `InputBatch` 与 `attn_metadata` 字段数值都**完全相同**，两卡的 KV Cache 形状也相同；与单卡相比，唯一的区别是每张卡的 KV Cache 张量只持有一半 KV 头（如 0.5B 模型的 2 个 KV 头变为每卡 1 个）。

而根据 vLLM 0.32 源码，在其他并行模式下（pLLM 目前仅支持 TP）：**数据并行（DP）** 的各个 rank 拥有各自独立的调度器与批次，但会协商一个共同的补齐后 token 数以对齐协同算子，因此尾部 padding 行可能多于单卡；**解码上下文并行（DCP）** 则将历史 KV 沿序列维切分到不同卡上，通过 `dcp_local_seq_lens` 记录本卡持有的局部 KV 长度，而 `seq_lens` 仍代表全局总长度。
