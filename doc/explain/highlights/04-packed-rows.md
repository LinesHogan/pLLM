# 行模型：在连续批处理里找回“每个请求”

如果你习惯了在单卡研究脚本里写 PyTorch 代码，你眼里的 batch 通常是三维的：`[batch_size, seq_len, hidden_size]`。第 0 号请求永远住在 `x[0]`，第 1 号请求永远住在 `x[1]`。哪怕不同请求的长度不一样，无非是在短请求后面补一串 `<pad>`，谁的激活值在哪一行、哪一列，从头到尾固定不变。

但当你往 vLLM 这样的生产级推理引擎里挂一个 forward hook，在第 0 层把传入的 `hidden` 张量打印出来时，第三个维度不见了。你拿到的是一张只有两个维度的扁平张量，比如 `[64, 896]`。

这是连续批处理（continuous batching）与分块预填充（chunked prefill）工作时的真实画面。在线服务不会等所有请求凑齐再一起出发，也不会让一个刚进来的长文档独占整张显卡。每一步（step），调度器都会决定为场上每个活跃请求处理几个 token：正在逐字生成的请求（decode）通常算 1 个 token；刚进门的短请求可能一次算完整段提示词（prefill）；而动辄几千字的长提示词则会被切成小块，分好几步消化完。接着，引擎把所有人这一步要算的 token 首尾相接，拍扁成一张 `[num_tokens, hidden_size]` 的二维长条送进模型。

我们不妨直接拿一个真实的批次摆在桌面上看。下面这组数据取自我们在 vLLM 0.32.0 上对 `Qwen2.5-0.5B-Instruct`（hidden 维度 896）的实测探针（设 `max_num_batched_tokens=64` 以触发分块预填充，探针设置见文末附录）。此时系统里同时跑着三个用户请求：

* 请求 `0`：提示词长 5 个 token，但调用方设置了并行采样 `n=2`，要同时生成两条回答，于是引擎在内部将它拆成了两个独立的子请求 `0_0` 和 `1_0`；
* 请求 `1`：一段只有 2 个 token 的短提示词；
* 请求 `2`：一段长达 150 个 token 的长提示词。因为每步最多只能处理 64 个 token，它在第 1 步先算完了前 52 个 token，在当前这一步（第 2 步）接着算接下来的 61 个 token。

在第 2 步进入模型第 0 层时，那张 `[64, 896]` 的张量内部其实是这样排布的：

```text
二维张量 hidden [64, 896] 的真实内部切片（第 2 步）：
┌──────────────────────────────────────────────────────────┐
│ Row 0:      请求 0_0 的第 1 个生成词 (decode, 位置 5)     │
│ Row 1:      请求 1_0 的第 1 个生成词 (decode, 位置 5)     │
│ Row 2:      请求 1   的第 1 个生成词 (decode, 位置 2)     │
│ Row 3..63:  请求 2   的提示词中间段  (prefill, 位置 52..112)│
└──────────────────────────────────────────────────────────┘
（若在某一步总 token 数凑不够 CUDA Graph 录制尺寸，末尾还会补上几行幽灵 padding）
```

如果你只是拿模型纯生成文本，这种拼图细节被引擎藏在底下，你完全不必理会。但只要你想在推理的同时维护与请求绑定的状态——比如为每个请求在线训练一个轻量级的蒸馏网络（ESamp），或者为每个请求更新一份专属的快权重——你在写下第一行数学算子之前，就必须先对着这张扁平张量回答四个问题：

第一，第 `r` 行到底属于哪个请求？到了下一步，有人生成结束退出了、有人新加进来，调度器重新洗牌，同一个请求又跑到了哪一行？
第二，这一行当前处在什么阶段？请求 `0_0` 的这 1 个 token，究竟是提示词的一部分，还是已经进入生成阶段的 decode token？
第三，谁在这一步需要进采样器抽样？比如请求 `2` 在这一步占了第 3 到 63 行，它的最后一行（行 63）该不该拿去触发第一次采样？
第四，这一行是不是“炒冷饭”？在线服务里，当 KV 缓存被长文本撑满时，调度器会把某个正在生成的请求临时驱逐（preempt）；等过几步显存宽裕了，再把它的提示词连同之前已经生成的 token 从头当成 prefill 重算一遍。对无状态推理来说，重算一遍毫无副作用；但对有状态的在线学习来说，如果不把这些重算的历史 token 识别出来剔除掉，同一段文本就会被训练器喂两遍。

这篇文章要讨论的，就是怎样在不破坏 CUDA Graph、也不引入 CPU-GPU 同步等待的前提下，干净地解开这四个问题。最终的答案非常简单：每个请求只需要三个已经在 CPU 上的整数，外加一个记录历史水位线的整数。但在展开这个三整数算术之前，我们先看看如果顺着直觉往模型内部去找答案，会撞上哪些暗礁。

---

## 为什么不能等 `LogitsProcessor`，也不必去扒 `attn_metadata`

当你想在推理引擎里知道“哪一行对应哪个请求”时，最容易找到的公开接口是 `LogitsProcessor`。社区里许多外挂在 vLLM 上的引导或监控工具走的都是这条路。

但如果你的算法需要和模型前向并行，这条路从一开始就走不通。看一眼一步执行的时间线就明白了：`LogitsProcessor` 要等整座数十层的 Transformer 全部跑完、连最后那层庞大的 `lm_head` 词表投影都乘完了，才会被引擎调用。如果你只想在第 0 层把隐藏状态捞出来，放到旁路 CUDA stream 上做预计算或在线训练，好让它与后面几十层的模型前向重叠（overlap）起来，等到 `LogitsProcessor` 才拿到行号拆包信息，黄花菜早就凉了。

既然等不到最后，自然的念头就是顺着调用栈往前找：模型内部一定有人知道每一行是谁——至少 Attention 层知道！否则注意力算子怎么做到不让张三的 token 偷看李四的 KV 缓存？

顺着这个思路摸进模型执行器（Model Runner），你会发现 vLLM 的注意力计算封装在底层的 C++/CUDA 内核里，Python 层看不见拆包过程；而在传给注意力层的参数里，唯一装着批次结构信息的，是一个叫 `attn_metadata`（Attention Metadata）的对象。

从 `attn_metadata` 里确实能抠出行布局，但这是一条布满荆棘的弯路（我们在文末[附录](#附vllm-attention-metadata-字段解剖笔记)里整理了在 vLLM 0.27.1 与 0.32.0 上用探针实测出的 `attn_metadata` 字段行为，供需要研究底层内核的读者参考）。为什么说它是弯路？

首先是**口径的泥潭**。`attn_metadata` 不是为上层扩展设计的通用账本，而是为特定 CUDA 内核准备的启动参数，不同注意力后端各有一套脾气。比如在 FlashAttention 后端的 `FlashAttentionMetadata` 里，名叫 `num_decode_tokens` 和 `num_prefill_tokens` 的字段恒为 `0`；而在新版 runner（MRv2）的整图（`FULL` CUDA Graph）模式下，名叫“实际 token 数”的 `num_actual_tokens` 竟然把末尾用来凑数的幽灵 padding 行也算了进去！更隐蔽的是代表采样位置的 `logits_indices`：在前面那个第 2 步的例子里，请求 `2` 的 150 个提示词才算到第 113 个，离算完还差得远，但它的本步最后一行（行 63）依然会出现在 `logits_indices` 里——如果你盲目相信出现在 `logits_indices` 里的行都已经完成了预填充，就会在半山腰上错误地触发采样引导或状态导出。

比口径更致命的是**物理位置**。`attn_metadata` 里的关键切分字段（如 `query_start_loc`、`seq_lens`）主要是一组已经在 GPU 显存里的张量。如果你在模型前向传播的半路上，把这些 GPU 张量拷回 CPU 解析出“第 3 行属于请求 A”，再把挑出来的行索引传回 GPU 去切分 `hidden`，这一来一回就会触发阻塞式的 CPU-GPU 同步（Host-Device Synchronization）。在单张 RTX 4090 上用 vLLM 0.32.0 跑 Qwen2.5-0.5B（BF16，128 并发序列，开启 CUDA Graph）时，一步完整的 decode 连同采样总共只有 3.0 ms，几次同步等待再加上被同步打断的异步发射流水线，足以让吞吐掉下一大截；要想不阻塞，你就得被迫去写一套复杂的 CPU/GPU 协同解析与图内轮询缓冲区。

### 怎样确认自己真的读懂了黑盒系统的行布局？

在离开这段探索之前，有一件工具层面的经验值得单拎出来。当你面对一个复杂的黑盒引擎，写了一套自以为正确的“行拆包（unpack）”逻辑时，你怎么确认自己在分块预填充、混排、重排序和图 padding 的各种边界条件下，真的没有偏掉哪怕一行？

毕竟，哪怕你的索引算术在某个角落错位了 1 行，把李四的隐藏状态拿去更新了张三的参数，代码也绝不会抛出越界异常，损失函数甚至还会看起来正常下降。

验证这类行定位逻辑，有一个不依赖宿主系统任何内部假设的“金标准”实验协议（出自验证脚本 `tllm/legacy/verify_v1_decode_rows_runtime.py` 的文档字符串）：

```python
# check-docs: skip: excerpt from tllm/legacy/verify_v1_decode_rows_runtime.py in the old tLLM repository
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

这个协议的思想非常朴素：
1. **金标准（Gold standard）**：把一批测试 prompt **一个接一个单独**喂给引擎（使用 greedy 贪心解码固定生成轨迹）。当整个引擎里只有 1 个请求在跑时，根本不存在多请求混排，“哪一行是谁”毫无歧义。把每个请求在每一步的第 0 层输出抓下来存为基准。
2. **候选方案（Candidate）**：把完全相同的这批 prompt **同时扔进引擎并行跑**，让连续批处理调度器自由洗牌、混排分块，然后用你写的拆包逻辑从第 0 层挑出每个请求在每一步对应的行。
3. **判据**：逐个请求、逐个生成步比对两边的隐藏状态均方误差（MSE）。只要有一处对不上，就说明你对宿主行布局的理解还有死角。

用“单跑无歧义”去校验“批跑能拆包”，是检验一切多租户批处理定位逻辑最可靠的试纸。

---

## 装车前的发货单：三个整数推导一切

现在让我们退后一步。为什么去扒 `attn_metadata` 会那么痛苦？因为那时数据已经被打包成了面向底层硬件内核的形态——就像快递包裹已经被胡乱塞进了货车车厢，你跳上开动的货车去挨个拆包装纸，自然费力不讨好。

可如果我们在包裹装车之前，往调度室的桌子上看一眼呢？

在每一轮模型前向开始之前，CPU 上的调度器本来就已经决定好了这一步要处理哪些请求。对于本步被调度的每一个请求，CPU 手里现成握着三个最基础的整数：

* `scheduled`：这一步调度器决定为它算几个 token；
* `prompt`：它的原始提示词（prompt）总共有多长；
* `computed`：在今天这一步开始**之前**，历史上已经为它算过了几个 token。

让我们回到文章开头那个第 2 步的真实批次，把这三个整数列成一张小表：

```text
请求     scheduled    prompt    computed
0_0          1           5          5
1_0          1           5          5
1            1           2          2
2           61         150         52
```

不需要碰任何一个 GPU 张量，也不需要管底层用的是 FlashAttention 还是 FlashInfer，我们只拿这三个数做两步加减法。

首先，因为所有请求的 token 是按顺序首尾相接拼进扁平张量的，只要把 `scheduled` 累加起来，就能知道每个请求占据了哪几行：请求 `0_0` 占行 `0`，请求 `1_0` 占行 `1`，请求 `1` 占行 `2`，请求 `2` 占行 `3..63`。

其次，对于某个请求在本步被调度的第 `i` 个 token（`0 <= i < scheduled`），它在整条序列生命周期里的绝对逻辑位置（即 RoPE 位置编码的下标 `offset`）就是：

$$\text{offset} = \text{computed} + i$$

一旦算出 `offset`，前面那四个棘手的问题在初等算术面前瞬间迎刃而解：

1. **它是提示词还是生成词？** 直接比大小：`offset < prompt` 就是提示词阶段（prefill），`offset >= prompt` 就是生成阶段（decode）。看请求 `0_0`：它的 `computed = 5`，本步 `i = 0`，所以 `offset = 5`。拿去和 `prompt = 5` 一比：`5 >= 5`，说明前 5 个提示词（下标 `0..4`）早就翻篇了，行 `0` 是实打实的 decode 行。请求 `1_0`（`5 >= 5`）和请求 `1`（`2 >= 2`）同理。而请求 `2` 的 61 个 token，`offset` 从 `52` 排到 `112`，全都小于 `150`，所以行 `3..63` 全是提示词。
2. **哪一行是“提示词刚刚算完”的采样行？** 充要条件只有一个：`offset == prompt - 1`。在第 2 步里，请求 `2` 算到最后一行（行 63）时 `offset` 才到 `112`，不等于 `149`（`150 - 1`）。所以不管内部 `logits_indices` 怎么填，算术直接告诉我们：请求 `2` 的提示词还没完，行 63 绝不是预填充结束行。等到下一步（第 3 步），请求 `2` 的 `computed` 变成 `113`，调度器排上剩余的 `scheduled = 37` 个词，`offset` 从 `113` 走到 `149`——唯有在 `i = 36` 那一行，等式 `offset == 149` 成立。哪怕一段长提示词被切成了十块、跨十步算完，`offset == prompt - 1` 在整个生命周期里也只会精准地成立唯一一次。
3. **哪一行是本步的“最新 decode 行”？** 满足 `offset >= prompt` 且是该请求在本步的最后一个 token（`i == scheduled - 1`）。
4. **怎么识别被抢占驱逐后的重算（Preemption Replay）？**

为了抓出重算，我们只需要在会话里为每个请求多记一个整数：**`frontier`（前沿水位线）**——代表这个请求自打进入引擎以来，我们的有状态逻辑**曾经见过的最大位置**。

```text
请求 2 的生命周期位置轴（prompt = 150）：

正常推进完第 2 步后，水位线推到 113：
  0                52                   113                150
  ├─────────────────┼────────────────────┤                  │
  │   第 1 步算完    │     第 2 步算完     ▲                  │
  └─────────────────┴────────────────────┴──────────────────┘
                                     frontier = 113

若此时遭遇 KV 驱逐，稍后从 0 开始重算前 64 个词 (computed=0, scheduled=64)：
  0                         64          113                150
  ├──────────────────────────┼───────────┤                  │
  │ offset 0..63 < frontier! │           ▲                  │
  │  ──► 标记为 RECOMPUTED    │       frontier = 113         │
  └──────────────────────────┴──────────────────────────────┘
```

看上面的数轴：请求 `2` 正常跑完前两步后，`frontier` 更新到了 `113`。假设此时显存吃紧，它的 KV 块被驱逐回收；几步之后显存宽裕了，调度器把它重新排上来从头算起，此时传入的 `computed` 归零为 `0`，本步 `scheduled = 64`。当我们遍历这 64 个 token 时，算出来的 `offset` 是 `0..63`——它们全都严格小于 `frontier`（`113`）。

只要看到 `offset < frontier`，标签器当场就能判定：“这几行是为了重建 KV 缓存而回放的旧历史，不是新推进的 token。”于是把它们的 `RECOMPUTED` 标志置为 `1`，自动从在线训练和激活导出的选中集合里剔除，直到重算进度重新跨过 `113` 这条水位线为止。（这对应的正是保留已训练状态的 `keep` 抢占策略；至于什么时候该保留状态、什么时候该把状态也一起倒回初始版本 `reset`，我们在[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)里会专门讨论。）

---

## 选最稳定的接缝，用一次遍历喂饱生产与测试

算术想清楚了，接下来的工程问题是：这段代码应该挂在 vLLM 的哪个位置？

在扩展一个快速演进的宿主系统时，选挂载点有一条很实在的经验：**不要去钩那个包含信息最多、但返回结构复杂的私有函数，要去钩那个参数最简单、在各个版本间签名最稳定的交界点。**

如果去钩模型执行器里构建全部输入元数据的内部函数（如 MRv1 的 `_prepare_inputs`），它往往返回一个包含五六个元素的长元组，vLLM 每隔一两个小版本增删一个内部特性，元组长度就会变一次，逼着你在外面写一堆按 `len(out) >= 6` 分支的脆弱胶水。

相反，在新版模型执行器（MRv2 runner）里，每一步当 `prepare_inputs` 刚刚把 CPU 侧的 `InputBatch` 填好、但还没有开始构造复杂的注意力元数据时，vLLM 都会调用一次 `prepare_attn(self, input_batch: InputBatch)`。在 pLLM 支持的每个 vLLM 版本（0.27.1 与 0.32.0）上，它的签名分毫不差。我们就把行标签的发布放在这里（`pllm/backends/vllm/runner.py`）：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/runner.py
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

在这个时刻动手，模型前向还没开始，CUDA Graph 还没按播放键。`encode_mrv2_rows`（位于 `pllm/backends/vllm/packed_rows.py`）只需做几项基本的契约断言（如确认 `input_batch.num_draft_tokens == 0` 即未开启投机解码，确认每个请求恰好对应一个采样位，并只取真实请求数 `num_reqs` 从而天然滤除尾部的图 padding 行），然后直接把 `num_scheduled_tokens`、`prompt_lengths`、`num_computed_tokens_np` 这三列现成的 CPU NumPy 数组借给底层的 Rust 标签器。（旧版 MRv1 runner 在 `pllm/backends/vllm/mrv1.py` 中通过薄适配层传入同样的三列数组。）

至于请求的字符串 ID、属于第几个 prompt、第几个采样分支等身份信息（`RequestIdentity`），则在请求准入（`admit`）和进入持久槽（`add_requests`）时就已经编码成了紧凑的整数槽号 `labeler.slots`。因此在每步高频执行的循环里，跨过 Python/Rust 边界的只有纯数字切片，连一个字符串都不碰。

来看看 `rust/pllm_core/src/packed_rows.rs` 里的核心遍历逻辑：

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

这段循环就是上一节纸面加减法的直接翻译。因为全程只读写只读借用的 NumPy 内存切片、完全不触碰 PyTorch 张量或 CUDA 上下文，外层在调用它时直接通过 `py.allow_threads(...)` 释放了 Python GIL。一次线性扫描下来，它会同时生成：
* 一张 7 列的完整标签矩阵（记录每个 token 的打包行号 `PACKED`、请求在批内的序号 `ORDINAL`、逻辑位置 `OFFSET`、阶段 `PHASE`、身份索引 `IDENTITY`、提示词长 `PROMPT_LENGTH` 以及重算标志 `RECOMPUTED`）；
* 四组预筛选好的活跃行列表：最新解码行 `decode`、提示词完成行 `prefill`、两者的并集 `both`、以及全部非重算的新算行 `fresh`；
* 每组选中行对应的“打包行号 → 槽内名次（rank）”映射表（供模型内的隐藏层探针直接按名次把激活写进固定缓冲区），以及连接每个请求最后一行与采样器行的 `links`。

把行标签器的输入收敛成 `(scheduled, prompt, computed)` 三个整数，还带来了一个意想不到的工程红利：**离线测试再也不需要写“模拟调度器”了。**

在[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)中你会看到，pLLM 的 CPU 契约检查器 `pllm.testing.check` 需要在不启动 GPU 的情况下，检验用户代码在分块预填充、混排、行乱序和抢占重算下是否保持结果不变。它之所以能确定性地跑完这些复杂场景，正是因为测试框架直接把几串手写的 `(请求, scheduled, computed)` 三元组喂给了同一个 `RowLabeler`。生产环境跑哪几十行 Rust 算术，离线契约检查就跑哪几十行算术。

---

## 当“行”成为所有 API 的第一维，以及身份粒度的性能账

底层的标签器把账算清之后，“行（Row）”就成了用户编写算法时唯一需要面对的批次维度。

在 pLLM 里，无论你写的是读取隐藏层、改写残差、在线更新参数还是修正采样分布，方法收到的每一个按请求组织的张量，**第 0 维永远是对齐好的活跃行 `[rows, ...]`**。你写的函数只需要沿着特征维（`dim=-1`）做算术，至于这一步的 `rows` 到底是从扁平张量的哪几行抽出来的、对应哪个请求的状态版本，全由底层根据你在声明时指定的视角自动路由：

```text
InputBatch 三整数 (CPU)        RowLabeler (Rust)                用户方法收到的第 0 维 [rows, ...]
┌──────────────────┐         ┌──────────────────┐  decode  ──► pllm.hidden (默认 tokens="decode")
│ scheduled        │         │ 7 列标签矩阵      │  prefill ──► pllm.hidden (tokens="prefill")
│ prompt_len       ├────────►│ + 4 组选中行索引  │  fresh   ──► pllm.hidden (tokens="all")
│ computed         │         │ + 槽内名次映射表  │  links   ──► @pllm.add_logits / guide_sampling
└──────────────────┘         └──────────────────┘
```

不仅如此，把“行的身份（`RequestIdentity`）”显式建模出来，有时候还能直接省下大笔算力。

考虑一个常见的采样设置：用户发来 8 条 prompt，每条要求并行采样 `n=16` 个候选回答，引擎里一共跑着 `8 × 16 = 128` 个子请求行。如果在线学习算法机械地按“每个子请求一份状态”去绑定，每一步就得为全部 128 行做反向传播和参数更新。但同一个 prompt 展开的 16 个分支，起点的提示词上下文是完全相同的。如果你在声明 `RequestState` 时加上一行：

```python
# check-docs: skip: single keyword argument, not a standalone snippet
key_fields=("prompt_index",)
```

行标签器就会把同一个 `prompt_index` 下的 16 个采样分支路由到同一份状态谱系上：16 行各自照常算出损失，再按状态求均值（`reduction="mean"`），每份状态每步只做一次优化器更新，需要维护和更新的状态立刻从 128 份缩减到 8 份。

而在多卡张量并行（TP=2 或 TP=4）下，因为 vLLM 的所有 TP worker 在每一步都会收到完全相同的调度指令，这套纯确定性的整数算术可以在每张卡的 CPU 进程上各自独立跑完，算出的行标签与名次映射天然一致，不需要一比特的跨卡通信。

---

## 成本账单与表达边界

最后，我们来看看这套行模型在运行时要付出的代价，以及它的边界在哪里。

第一笔账是**每一步的宿主端 CPU 准备耗时**。虽然我们避开了从 GPU 往回拷张量的同步停顿，但每一步在 CPU 上借用 NumPy 数组、跑一遍 Rust 标签器、再把算好的行索引和状态槽号填进锁页内存准备异步上传（即[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)里那张共享 `int64` 表的 `rows [6, R]` 段），仍然需要消耗 CPU 时间。

在单张 NVIDIA RTX 4090（24 GiB）上，对 128 并发序列（8 个 prompt × `n=16`）、生成 256 个 token 的 ESamp 负载（$\beta=0$，开启 CUDA Graph，vLLM 0.27.1）做宿主侧剖析：在 `Qwen2.5-0.5B-Instruct` 上，每次调用的中位耗时约为 `encode_mrv2_rows` **122 µs**、会话 `prepare_input` **174 µs**、隐藏层抓取准备 `HiddenTaps.prepare` **110 µs**。由于这段耗时只取决于批次里的请求数和行数，与模型参数量无关：在一步 decode 只有 **3.0 ms**（原生吞吐 42,355 tok/s）的 0.5B 小模型上，这四百多微秒的 CPU 工作量虽然大部分能被异步流水线藏在 GPU 背后，但依然占了不可忽视的比重；而当切换到同配置下的 **`Qwen2.5-7B-Instruct`**（BF16，vLLM 0.32.0，开启 FlashInfer 采样器）时，GPU 跑一步 decode 需要 **21.7 ms**（原生吞吐 5,894 tok/s），这几百微秒的宿主准备时间相对整步就微不足道了。

第二道边界来自**对宿主调度不变量的依赖**。尽管 `prepare_attn` 的函数签名很稳，但我们依然读取了 `InputBatch` 的内部列，并依赖两条关键假设：
* **不开启投机解码（Speculative Decoding）**：投机解码会让一个处于 decode 阶段的请求在同一步里塞入多个草稿 token、产生多个采样位，打破“每请求每步最多一个最新 decode 行、恰好一个采样位”的前提。因此遇到投机解码时，pLLM 会在入口处直接报错拒绝，而不是带着错位的行号勉强运行。
* **前缀缓存（Prefix Caching）与模型内改写互斥**：vLLM 的前缀缓存假设“只要两段请求的 token 前缀相同，算出来的 KV 缓存就一定相同”。当你的工作负载只是在模型外围读取隐藏态或训练独立小网络时，前缀缓存可以正常开启；但只要你挂载了按请求改写残差、权重覆盖层或自定义注意力的逻辑，相同的 token 在不同请求手里会算出不同的 KV，此时必须关闭前缀缓存。
* **禁止跨行归约**：因为张量的第 0 维代表同一步里拼在一起的不同请求，用户方法里绝对不能写 `h - h.mean(dim=0)` 或 `BatchNorm` 这类跨第 0 维的算术——至于万一你不小心写了，我们怎么在装入 GPU 之前自动把它揪出来，留到第 12 篇[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)再细讲。

---

## 附：vLLM attention metadata 字段解剖笔记

虽然今天的 pLLM 已经在更上游的 `prepare_attn` 入口直接用三个整数解决了行定位，完全不再读取 `attn_metadata`，但对于想要研究 vLLM 内部连续批处理拼装、分页 KV 寻址与 CUDA Graph 配合机制的开发者来说，`attn_metadata` 依然是最直观的切片。

下面是用一段轻量探针包住 MRv2 runner 的 `model_state.prepare_attn`，在 **vLLM 0.27.1 与 0.32.0** 两个版本上实测核对出的运行态记录。

### 1. 探针实验设置

开启 MRv2 runner（`VLLM_USE_V2_MODEL_RUNNER=1`），加载 `Qwen2.5-0.5B-Instruct`，设 `max_num_seqs=8`、`max_num_batched_tokens=64`（逼出分块 prefill），关闭 prefix caching，开启 CUDA Graph。同时提交三个请求：
* **请求 0**：`"Hello, my name is"`（5 个 token），设 `n=2` 并行生成 5 个 token，内部裂变为 `0_0` 与 `1_0`；
* **请求 1**：`"Paris is"`（2 个 token），生成 5 个 token；
* **请求 2**：150 个 token 的长提示词，生成 2 个 token。

单卡实验在 RTX 4090 上完成（覆盖 0.27.1 / 0.32.0 × FlashAttention / FlashInfer）；多卡张量并行（TP=2）在两张 A100-SXM4-80GB 上完成。

### 2. 前五个真实步的逐帧打印

以下是 vLLM 0.32.0 在 FlashAttention 后端下的前五步实测值（0.27.1 除 `idx_mapping` 的具体槽号与整数位宽外完全一致；表中 `computed` 即 `num_computed_tokens`，`prefilling` 即 `is_prefilling`）：

| 步 | 图模式 | 行序（请求） | scheduled | computed | query_start_loc | seq_lens | logits_indices | prefilling |
|---|---|---|---|---|---|---|---|---|
| 1 | NONE (eager) | `1, 0_0, 1_0, 2` | `2, 5, 5, 52` | `0, 0, 0, 0` | `0, 2, 7, 12, 64` | `2, 5, 5, 52` | `1, 6, 11, 63` | `T, T, T, T` |
| 2 | NONE (eager) | `0_0, 1_0, 1, 2` | `1, 1, 1, 61` | `5, 5, 2, 52` | `0, 1, 2, 3, 64` | `6, 6, 3, 113` | `0, 1, 2, 63` | `F, F, F, T` |
| 3 | NONE (eager) | `0_0, 1_0, 1, 2` | `1, 1, 1, 37` | `6, 6, 3, 113` | `0, 1, 2, 3, 40` | `7, 7, 4, 150` | `0, 1, 2, 39` | `F, F, F, T` |
| 4 | FULL (graph) | `0_0, 1_0, 1, 2` | `1, 1, 1, 1` | `7, 7, 4, 150` | `0, 1, 2, 3, 4` | `8, 8, 5, 151` | `0, 1, 2, 3` | `F, F, F, F` |
| 5 | FULL (graph) | `0_0, 1_0, 1, (pad)` | `1, 1, 1` | `8, 8, 5` | `0, 1, 2, 3, 3` | `9, 9, 6, 0` | `0, 1, 2` | `F, F, F` |

对照这五步数字，可以直接验证六个光靠静态读代码容易误判的细节：

1. **批内行序会被主动重排**。第 1 步四个请求按本步 `scheduled` 从小到大排成了 `2, 5, 5, 52`；第 2 步三个单 token 的 decode 请求被排在最前面，分块 prefill 的请求 `2` 排在最后。这是 MRv2 的 `sort_batch_req_ids` 所为，目的是满足注意力后端 `split_decodes_and_prefills` 对“短查询排在最前”的布局要求。
2. **`logits_indices` 包含尚未完成分块 prefill 的中间末尾行**。第 1、2 步里，请求 `2` 的提示词还没算完，但它的本步末尾行 `63` 照样在 `logits_indices` 里参与了 LM head 投影，直到第 3 步的 `39`（对应 `offset = 149`）才是真正的提示词结束采样位。
3. **`seq_lens` 是算完本步之后的有效 KV 总长**，严格等于 `computed + scheduled`。
4. **两种截然不同的“prefill / decode”划分口径**。`CommonAttentionMetadata.is_prefilling` 按请求的真实生命周期（`computed < prompt`）判定；而 FlashInfer 后端的 `num_decodes / num_prefills` 则按**本步 query 长度是否不超过阈值（1）**判定——如果一个分块 prefill 的长提示词在最后一步恰好只剩 1 个 token，FlashInfer 会把它归入 decode（short extend），而 `is_prefilling` 仍为 `True`。
5. **`FULL` 图模式下 `num_actual_tokens` 包含了 padding 行**。看第 5 步：请求 `2` 已退役，场上只剩 3 个真实请求（`InputBatch.num_tokens = 3`），但为了对齐尺寸为 4 的 CUDA Graph，`num_tokens_after_padding = 4`。此时 `query_start_loc` 末尾重复了 `3`，`seq_lens` 补了 `0`，`slot_mapping` 填了 `-1`，但 `num_actual_tokens` 的值却是 `4`（vLLM 源码此处有一行 `TODO(lucas): rename to num_tokens since it may be padded and this is misleading`）。在 MRv2 下过滤 padding 必须看 `InputBatch.num_tokens`。
6. **`n > 1` 在关闭前缀缓存时的物理槽位分配**。第 1 步中 `slot_mapping` 显示请求 `0` 的两个分支 `0_0` 和 `1_0` 分别写入了物理槽 `16..20` 和 `32..36`，各自独立计算并存储了相同的 5 个提示词 token。

### 3. 核心结构体字段速查表

**`CommonAttentionMetadata`**（各注意力后端共享的公共输入结构）：

| 字段名 | 实测含义与使用注意 |
|---|---|
| `query_start_loc` / `query_start_loc_cpu` | 形状 `(num_reqs + 1,)` 的 `int32` 张量（GPU/CPU 各一），记录每个请求在扁平批次中的起始行偏移（前缀和）；`FULL` 图模式下包含长度为 0 的尾部补齐段。 |
| `seq_lens` | 形状 `(num_reqs,)` 的 GPU `int32` 张量，记录本步完成后各请求的 KV 长度（`computed + scheduled`）；补齐请求处为 `0`。 |
| `num_reqs` | 请求槽位数；`FULL` 图模式下为补齐后的数量（如第 5 步为 `4`）。 |
| `num_actual_tokens` | 扁平批次行数；**MRv2 的 `FULL` 图模式下包含尾部 padding 行**，真实 token 数应取 `InputBatch.num_tokens`。 |
| `max_query_len` / `max_seq_len` | 本步最大单请求查询长度与最大 KV 长度。 |
| `block_table_tensor` | `[num_reqs, max_blocks]` 的 GPU `int32` 张量，记录各请求的物理 KV 块号表。 |
| `slot_mapping` | 长度为 `num_actual_tokens` 的 GPU `int64` 张量，记录本步各 token 写入 KV 池的物理槽位；padding 行处固定为 `-1`。 |
| `positions` | 长度为 `num_actual_tokens` 的 GPU `int64` 张量，即各 token 在序列内的逻辑位置 `offset`。 |
| `is_prefilling` | 形状 `(真实请求数,)` 的 CPU `bool` 张量，按 `computed < prompt` 标记真实阶段。 |
| `seq_lens_cpu_upper_bound` | `seq_lens` 在 CPU 侧的上界视图（无投机解码时即精确值）。 |
| `req_idx`（仅 vLLM 0.32） | `int64` NumPy 数组，记录真实请求在 runner 持久表中的槽位，同 `InputBatch.idx_mapping_np`。 |
| `dcp_local_seq_lens` 等 | 开启 Decode Context Parallel 时本卡持有的局部 KV 长度；未开启时为 `None`。 |
| `logits_indices_padded` / `num_logits_indices` | 仅特定 fast-prefill 路径使用；常规路径下恒为 `None`。 |

**后端专属 Metadata 与 `InputBatch` 差异**：
* **`FlashAttentionMetadata`**：包含公共底座子集及 FA3 调度元数据 `scheduler_metadata / max_num_splits`；注意其内部的 `num_decode_reqs / num_prefill_reqs / num_decode_tokens / num_prefill_tokens` 在该后端下**恒为 0**。0.27.1 中的级联注意力字段（`use_cascade` 等）在 0.32.0 中被删掉了。
* **`FlashInferMetadata`**：不含 `seq_lens` 与 `block_table`（已在 `build` 阶段编译进 `prefill` / `decode` 包装器对象内部）；其 `num_decodes / num_decode_tokens / num_prefills / num_prefill_tokens` 按本步查询长度切分并实际生效。
* **`InputBatch`（pLLM 在 MRv2 中借用的源结构）**：提供 CPU 侧的 `num_scheduled_tokens`、`num_computed_tokens_np`、`idx_mapping_np`（0.27.1 为 `int32`，0.32.0 为 `int64`）、`cu_num_logits_np`、`num_reqs / num_reqs_after_padding`、`num_tokens / num_tokens_after_padding` 以及 GPU 侧的 `logits_indices`，配合 `req_states.prompt_len` 即可直接完成全部行标签推导。

### 4. 多卡并行下的行为

在两张 A100-SXM4-80GB 上对 **张量并行（TP=2）** 的实测（覆盖 0.27.1 与 0.32.0 × FlashAttention 与 FlashInfer）表明：每一步 Rank 0 与 Rank 1 上探针打印出的 `InputBatch` 及 `attn_metadata` 字段数值完全一致，两卡的 KV 缓存形状也相同；与单卡相比，唯一区别是每张卡的 KV 缓存张量只持有半数 KV 头（如 0.5B 的 2 个 KV 头变为每卡 1 个）。至于 pLLM 当前未支持的另外两种并行模式（据 vLLM 0.32 源码）：数据并行（DP）下各 rank 独立调度，但会协商共同的补齐后 token 数从而增加尾部 padding 行；Decode Context Parallel（DCP）下历史 KV 沿序列维切分，`dcp_local_seq_lens` 记录本卡局部长度而 `seq_lens` 仍为全局长度。
