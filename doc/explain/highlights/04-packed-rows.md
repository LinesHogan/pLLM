# 行模型：在连续批处理里找回“每个请求”

在单卡上写 PyTorch，batch 是一件平常事：把 B 条序列补齐成一样长，堆成 `[B, S, hidden]`，`x[i]` 就是第 i 条样本。像一排编了号的抽屉，谁的东西在哪个抽屉，一目了然。

vLLM 这样的 serving 引擎没有这排抽屉。它用**连续批处理**（continuous batching）：请求随时来、随时走；每一步，调度器为每个活跃请求决定这一步算几个 token——正在生成的请求通常算 1 个（decode），刚来的请求可能一次算完整段 prompt（prefill），开了分块预填充（chunked prefill）的话只算其中一块——然后把这些 token **首尾相接，拼成一张扁平的** `[num_tokens, hidden]` 送进模型。第 0 行可能是某个长 prompt 的一块，第 37 行可能是另一个请求刚生成的 token；为了凑上录好的 CUDA graph 尺寸，末尾还可能补几行没有意义的 padding。行序由调度器决定，每一步都在变。

如果只是用模型生成文字，这些都不用管。可只要你想在生成的同时做点有状态的事——给每个请求在线训练一个小网络、给每个请求维护一份记忆——就得先回答一个问题：**这一行是谁的？**

这篇讲我们怎样回答它。答案最后落在每个请求的三个整数上，简单得有点不好意思；但在走到那里之前，我们在 vLLM 的源码里绕了很久。先讲绕路的故事，再拿一个真实的 batch，把这三个数能推出的东西一行一行算出来。

## 在 vLLM 源码里找“第几行是谁”

2025 年做 ESamp 的时候，需求很具体：每个 decode 步，拿到第 0 层和最后一层的 hidden，还要知道每一行属于哪个请求，才能用那个请求自己的 distiller 去训练。forward hook 拿到的只是一张光秃秃的 `[num_tokens, hidden]`。

当时开源项目里能拿到拆包信息——哪一行属于哪个请求——的做法，都要等到 logits processor 那一步。可 logits processor 是在整个前向、连同最后的 LM head 都算完之后才被调用的。等到那时才知道谁是谁就太晚了：训练、读 hidden 这些额外的工作，没法和主计算重叠，只能排在后面干等。

我们不能等那么晚，只好往前追。attention 一定知道每个请求的 token 在哪里——否则它没法让每个请求只看自己的 KV——于是我们追进 runner，看 attention 是怎么拆包的。结果撞上一堵墙：vLLM 的 attention 是它自己的 C++/CUDA kernel，拆包的细节都在 kernel 里，Python 层看不见。我们注意到，attention 的输入里可能装着这份信息的只有一个结构：**attention metadata**。解包的信息一定是从这里来的。

这个结构的格式相当复杂，当时网上也没有任何解读。更麻烦的是它是动态的，光读代码很难确定：prefill 和 decode 混在一起时，哪些段是 prefill、哪些是 decode？有 padding 时，哪些字段把 padding 算进去了？多卡时又会怎么变？我们打了许多断点，一步一步地看，最后才弄明白。（今天在 vLLM 0.27.1 和 0.32 上重新实测的字段含义，整理在[文末附录](#附vllm-attention-metadata-字段笔记)里，给后来的人省点时间。）

弄懂格式之后，又冒出一个问题。attention metadata 是一个装着 GPU 张量引用的结构体，要是把它拷回 CPU 解析，算出“第 3 行是请求 A”，再把行号送回 GPU 去挑行，这一来一回就是几次同步和等待，而 decode 一步只有几毫秒。所以当时的方案只能和 GPU 缓冲区、CUDA graph 一起设计：不等 metadata 构造完，从构造它时用到的信息就开始解析；CPU 和 GPU 上都有解析，不把所有工作堆在 CPU 上；ESamp 当时的实现是一个**图内的轮询缓冲区**——CPU 定期、异步地把解析所需的信息放进固定的 GPU 缓冲区，GPU 在录好的图里完成解包。

在后来公开的旧 tLLM 代码里还看得到这套设计的样子。`prepare_decode_localization` 从 `input_batch` 取 `req_ids`、`num_prompt_tokens`、`num_computed_tokens_cpu`，用 `num_computed_tokens >= num_prompt_tokens` 判断一个请求是否真的在 decode，用 attention metadata 的 `num_actual_tokens` 排除 padding 行，再把选出的行号写进几块固定的 GPU 暂存区（`decode_row_idx`、`decode_valid_mask`）。超出容量时的报错是“Increase graph_scratch_rows”——那几块缓冲区就是给图里的 kernel 读的。

### 怎样确认自己真的读懂了

逆向一个大系统，最危险的是“以为自己懂了”。当时我们写过一个验证脚本，它的 docstring 就是一份实验协议（旧 tLLM 的 `verify_v1_decode_rows_runtime.py`）：

```python
# check-docs: skip: excerpt from the 2025 tLLM verify_v1_decode_rows_runtime.py
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

金标准是一个 prompt 一个 prompt 单独跑，这时候“第 0 行是谁”毫无歧义；候选是同样的 prompt 一起成批跑，用我们以为正确的方式挑行；任何一个 prompt、任何一个 decode 步的第一层 hidden 对不上，就说明我们还没懂。我们很喜欢这个协议：它不依赖对 vLLM 内部的任何理解，只依赖“单独跑时没有歧义”这一个事实。想验证你对别人系统的理解，这几乎是一个通用的配方。

### 留下的伤疤

旧 tLLM 的生产入口包在 `GPUModelRunner._prepare_inputs` 的返回值上。这是个私有函数，返回元组的长度在不同 vLLM 版本里不一样，于是有了三个适配器（`<0.10`、`0.10`、`0.11+`），基类里按长度去猜：

```python
# check-docs: skip: excerpt from the 2025 tLLM adapters/base.py
if len(out) >= 6:
    return PrepareInputsView(
        attn_metadata=out[0],
        logits_indices=out[1],
        spec_decode_common=out[4],
        num_scheduled_tokens_np=out[3],
    )

if len(out) >= 2:
    attn_metadata = out[0]
    logits_indices = out[1]
    spec_decode_common = None
```

这就是“手工追结构体”的代价：理解是对的，代码却寄生在一个随时会变的私有接口上。

## 一个真实的 batch

把绕路的故事放一放，看看我们到底要从一个 batch 里认出什么。下面是附录里实测的一步（vLLM 0.32，Qwen2.5-0.5B，`max_num_batched_tokens=64`）。这一步里有四个请求：请求 0 要两个样本（n=2），vLLM 把它拆成两个子请求 `0_0` 和 `1_0`，各自的 prompt 都是 5 个 token；请求 1 的 prompt 是 2 个 token；请求 2 是一个 150 token 的长 prompt，因为每步最多 64 个 token，只能分块算。这一步 vLLM 拼出来的扁平张量有 64 行：

```text
行 0      请求 0_0  （1 个 token）
行 1      请求 1_0  （1 个 token）
行 2      请求 1    （1 个 token）
行 3..63  请求 2    （61 个 token）
```

这四段各是什么？请求 0_0 是 prompt 的一部分还是生成的？请求 2 的 61 个 token 里，有没有哪一行该拿去采样？光看这张表是答不出来的。

再加上现实里的几件事，问题会更难。为了命中录好的 CUDA graph 尺寸，batch 可能被补齐，多出来的行是垃圾。一个请求可能被抢占：KV cache 不够时，vLLM 把它赶出去，过一会儿把它的 prompt 和已经生成的 token **从头重算**一遍，于是同样的内容会再一次以“prefill”的身份经过模型——对有状态的算法来说，这些重算的 token 不能再被当成新数据，否则同一段内容会被训练两遍。还有 CUDA graph 本身：行标签必须在前向开始之前算好、写进地址固定的缓冲区，前向中途不能回头问 CPU（见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。

## 三个数

答案是：对这一步被调度的每个请求，只需要三个整数。

- `scheduled`：本步为它算几个 token；
- `prompt`：它的 prompt 有多长；
- `computed`：此前已经为它算过几个 token。

这三个数在调度器决定这一步的时候就已经在 CPU 上了。把它们填到上面那个 batch 里：

```text
请求   scheduled  prompt  computed
0_0        1         5        5
1_0        1         5        5
1          1         2        2
2         61       150       52
```

现在可以一行一行地算。请求在本步的第 i 个 token，在请求里的位置是 `offset = computed + i`。请求 0_0 只有一个 token，offset 是 5；5 ≥ prompt（5），所以它是**生成的 token**——更具体地说，是这个请求最新的 decode 行（它是本段的最后一个）。请求 1_0、请求 1 同理，都是 decode 行。请求 2 的 61 个 token，offset 从 52 到 112，全都小于 150，**都是 prompt**；其中有没有“prompt 的最后一个 token”，也就是采样第一个生成 token 的那一行？条件是 `offset == prompt - 1`，即 149，这一步还没到。

下一步，请求 2 再算 37 个（offset 113 到 149），最后一个 token 的 offset 正好是 149：prompt 结束行在这一步出现，而且只出现这一次。分块预填充不需要任何额外的跟踪，算术自己保证了这一点。

还剩抢占。为每个请求再记一个整数 `frontier`——它见过的最大位置。假如请求 2 在第 100 个 token 之后被抢占，重新调度回来时 `computed` 从 0 开始，可 offset 小于 frontier 的那些 token 我们都见过了：标成“重算”，不再交给训练。

一个比喻：快递车里的包裹被随便堆在一起，但每个包裹上贴着“哪家、第几件”。我们之前做的，是跳上车去拆包裹看里面是什么；而这三个数是发车前就打印好的发货单。分拣根本不需要拆包。

事后看，这简单得让人不好意思。我们花了很多时间去理解 attention metadata，最后用上的只是构造它之前就已经存在的三个数。

## 今天的做法

### 入口：选签名最稳定的那个调用点

MRv2 runner（vLLM 新版的模型 runner）每步在构建好 `InputBatch` 之后调用 `prepare_attn`，之后才去构建 attention metadata。pLLM 覆盖的就是它（`pllm/backends/vllm/runner.py`）：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/runner.py
def prepare_attn(self, input_batch: InputBatch) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Publish the step's rows to the session.

    vLLM calls this once per real step, right after prepare_inputs built the InputBatch; unlike
    prepare_inputs its signature is the same on every supported vLLM release.
    """
    session = self.workload_session
    if session is not None:
        if session.input_tokens is not None:
            session.prepare_tokens(
                self.req_states.all_token_ids.gpu,
                tuple(int(index) for index in input_batch.idx_mapping_np),
                input_batch.query_start_loc,
                input_batch.positions,
                input_batch.num_tokens,
            )
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

docstring 里写着选它的理由：它的签名在 pLLM 支持的每个 vLLM 版本（0.27.1 和 0.32.0）上都一样。2026 年 10 月支持 vLLM 0.32 的那次提交说明里也写着：“rows are published from prepare_attn, whose signature is the same on both releases”。信息最多的调用点不一定最好；**最不容易变的那个**才是。（旧的 MRv1 runner 走 `pllm/backends/vllm/mrv1.py`，最后调用的是同一个标签器。）

这也应验了当年那条经验：从“构造 metadata 时的信息”开始解析。今天走得更远——pLLM 根本不读 attention metadata，三个数都来自 `InputBatch` 和请求状态的 CPU 列。

### 把假设写成断言

`encode_mrv2_rows`（`pllm/backends/vllm/packed_rows.py`）把这几列借过来交给标签器：

```python
# check-docs: skip: excerpt from pllm/backends/vllm/packed_rows.py
def encode_mrv2_rows(
    input_batch: InputBatch,
    prompt_lengths: np.ndarray,
    labeler: RowLabeler,
    *,
    step_id: int,
    keep: bool,
) -> PackedSamplerRowBatch:
    """Label actual prompt positions, including generated history in recomputed prefills."""
    count = input_batch.num_reqs
    assert count >= 1 and input_batch.num_reqs_after_padding >= count
    assert input_batch.num_tokens_after_padding >= input_batch.num_tokens
    assert input_batch.num_draft_tokens == 0 and input_batch.num_draft_tokens_per_req is None
    assert np.array_equal(input_batch.cu_num_logits_np, np.arange(count + 1))
    logits = input_batch.logits_indices
    assert logits.ndim == 1 and logits.device.type == "cuda" and logits.numel() == count
    return labeler.prepare(
        input_batch.num_scheduled_tokens,
        prompt_lengths,
        input_batch.num_computed_tokens_np,
        labeler.slots[input_batch.idx_mapping_np],
        step_id,
        keep,
    )
```

那几条 `assert` 是这段代码里最有价值的部分。它们把标签器依赖的假设写了出来：没有投机解码的草稿 token；`cu_num_logits` 是 `0..count`，也就是每个请求恰好一个采样行；`logits_indices` 在 GPU 上。vLLM 哪天改了调度协议，我们会在第一步就知道，而不是默默地挑错行。

请求的身份（请求 ID、第几个 prompt、第几个样本、路由键）在准入时编成整数，绑到 vLLM 的持久槽位上；每步只传槽号数组，不传字符串。

### 标签器：上一节那几行算术的直译

真正的标签在 Rust 里（`rust/pllm_core/src/packed_rows.rs`）。标签矩阵有七列：

```rust
// Rows of the label matrix; PackedPhaseRowBatch names them in Python.
pub(crate) const PACKED: usize = 0;
const ORDINAL: usize = 1;
const OFFSET: usize = 2;
const PHASE: usize = 3;
pub(crate) const IDENTITY: usize = 4;
const PROMPT_LENGTH: usize = 5;
const RECOMPUTED: usize = 6;
const COLUMNS: usize = 7;
```

核心的循环，和我们刚才在那个 batch 上手算的过程一模一样：

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

`offset`、`replay`、`offset == length - 1`、`offset >= length && token == count - 1`——每一行都能在上一节找到对应的那句话。一次遍历得到四种选择：最新 decode 行、prompt 结束行、两者之并、本步所有新算的 token；`links` 记下每个请求的最后一行，也就是它的采样行。frontier 只在 `keep` 策略下推进，被标成重算的 token 不会进入任何一种选择。（一个小细节：Python 调这个函数时，Rust 用 `py.allow_threads` 释放了 GIL——它只碰 NumPy 列，不碰张量，也不碰 CUDA。）

这个标签器还有第二个用户：CPU 上的契约检查（[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）。检查用的调度表就是一串 `(请求, scheduled, computed)`，喂给同一个 Rust 标签器。测试和生产走同一份实现，而不是另写一个“模拟 vLLM”的东西。

### 行流向哪里

有了标签，“行”就成了 pLLM 所有用户 API 的第 0 维。用户方法收到的每个张量，第 0 维都是行，只是不同的输入来源，“一行”的含义不同：默认的 `tokens="decode"` 是每个 decode 请求的最新 token；`tokens="prefill"` 是本步刚完成 prompt 的请求的最后一个 prompt token（分块时只出现一次）；`@pllm.add_logits`、`@pllm.guide_sampling` 是本步每个要采样的 token；`tokens="all"`（仅 `@pllm.export`）是本步新算的每个 token，抢占重算的只导出一次。这些全由同一个标签器派生：hidden tap 用名次映射把选中的行写进固定的槽（[《图内挂载点》](05-in-graph-attach-points.md)）；状态选择用行的身份找到它的版本（[《PTTS》](01-ptts.md)）；采样器用采样行（[《采样尾声融合》](07-sampler-epilogue.md)）。

行的身份分清楚，有时候直接就是性能。旧 tLLM 时代有一次大跳跃（优化记录第 20 步；7B，8 个 prompt × n=16，256 token，β=0，min_p 0.05，历史测量）：吞吐比从约 79% 提到约 95.4%（原生约 5,048 tok/s，开训练后约 4,817 tok/s）。改动的原话是：“对 V1 `n>1` prompt expansion，不再训练每个 expanded sample，而是每个 prompt/slot 训练一行”，同时修了 graph capture。今天对应的写法是 `key_fields=("prompt_index",)`：同一个 prompt 的样本共享一份状态。128 行变成 8 行，训练量少了一个数量级。

多卡时，每个 rank 拿到同样的调度结果，标签在每个 rank 上独立、确定地算出来，不需要任何通信（附录里有 TP=2 的实测）。

## 在哪里更差

行模型仍然依赖 vLLM 的内部类。入口换成了签名稳定的 `prepare_attn`，读的却还是 `vllm.v1.worker.gpu.input_batch.InputBatch`（MRv1 是 `gpu_input_batch.InputBatch`）的列；vLLM 改调度协议，这里就得改。最现实的例子是投机解码：一个请求一步可能有多个草稿 token、多个采样行，“每个请求恰好一个采样行”的假设就不成立了。今天遇到这样的 batch 会直接断言失败——pLLM 不支持投机解码。

每步在 CPU 上算标签、准备 hidden tap 也要时间。在 0.5B、β=0 的宿主 profile 里（RTX 4090，vLLM 0.27.1），每次调用的中位数是：`encode_mrv2_rows` 约 122 µs，`prepare_input` 约 174 µs，`HiddenTaps.prepare` 约 110 µs。0.5B 一步约 3 ms，这几百微秒大多藏在 GPU 的阴影里，但在 profile 里看得见；7B 一步约 21.7 ms，就不重要了。

vLLM 的 prefix caching 假设“相同的 token 前缀产生相同的 KV”。用户一旦改写前向——改残差、改权重、改注意力——这个假设就破了，所以改写前向和 prefix caching 互斥（[设计取舍](../design-tradeoffs.md)）。

最后是写法上的限制：行是不同请求的，用户代码不能沿第 0 维归约。`h - h.mean(0)`、BatchNorm、行间 softmax，都会让一个请求的输出依赖于“这一步恰好和谁拼在一起”。

## 可以带走的东西

这个故事的转折点不在于终于读懂了 attention metadata，而在于发现**不需要读它**：我们要的信息，在那个结构被构造之前就已经以三个整数的形式存在了。往别人的系统里接东西时，最诱人的往往是信息最多的那个结构，可它也最复杂、最容易变；更好的接缝通常在更上游，那里的信息更少、更干净，接口也更稳定。

当然，如果没有那段打断点的日子，我们大概也认不出这三个数就是全部。读懂别人的系统和找到最小的依赖，并不矛盾——前者常常是后者的前提。

## 附：vLLM attention metadata 字段笔记

ESamp 时代我们靠打断点摸清了 attention metadata。今天的 pLLM 已经不读它，但它仍然是理解 vLLM 怎么拼 batch 的最好窗口。下面是在 vLLM 0.27.1 和 0.32.0 上重新实测的结果，给后来想在 vLLM 里“找行”的人。

### 怎么测的

用一个 30 行的探针包住 MRv2 runner 里构建 metadata 的 `model_state.prepare_attn`，每个真实步打印 `InputBatch` 和第一层的 attention metadata。设置：`VLLM_USE_V2_MODEL_RUNNER=1`，Qwen2.5-0.5B-Instruct，`max_num_seqs=8`，`max_num_batched_tokens=64`（逼出分块 prefill），prefix caching 关，CUDA graph 开。三个请求同时提交：

- 请求 0：`"Hello, my name is"`（5 个 token），n=2，生成 5 个 token；vLLM 把它拆成两个子请求，ID 形如 `0_0-…`、`1_0-…`；
- 请求 1：`"Paris is"`（2 个 token），生成 5 个 token；
- 请求 2：一个 150 token 的长 prompt，生成 2 个 token。

单卡在 RTX 4090 上测（两个 vLLM 版本 × FlashAttention、FlashInfer 两个后端）；TP=2 在两张 A100-SXM4-80GB 上测。

### 一步一步看

下面是 vLLM 0.32.0、FlashAttention 后端的五个真实步（0.27.1 除了 `idx_mapping` 的数值和 dtype 外完全相同）。`computed` 是 `num_computed_tokens`；`prefilling` 是 `is_prefilling`。

| 步 | 图模式 | 行序（请求） | scheduled | computed | query_start_loc | seq_lens | logits_indices | prefilling |
|---|---|---|---|---|---|---|---|---|
| 1 | eager | 1, 0_0, 1_0, 2 | 2, 5, 5, 52 | 0, 0, 0, 0 | 0, 2, 7, 12, 64 | 2, 5, 5, 52 | 1, 6, 11, 63 | T, T, T, T |
| 2 | eager | 0_0, 1_0, 1, 2 | 1, 1, 1, 61 | 5, 5, 2, 52 | 0, 1, 2, 3, 64 | 6, 6, 3, 113 | 0, 1, 2, 63 | F, F, F, T |
| 3 | eager | 0_0, 1_0, 1, 2 | 1, 1, 1, 37 | 6, 6, 3, 113 | 0, 1, 2, 3, 40 | 7, 7, 4, 150 | 0, 1, 2, 39 | F, F, F, T |
| 4 | FULL | 0_0, 1_0, 1, 2 | 1, 1, 1, 1 | 7, 7, 4, 150 | 0, 1, 2, 3, 4 | 8, 8, 5, 151 | 0, 1, 2, 3 | F, F, F, F |
| 5 | FULL | 0_0, 1_0, 1, (pad) | 1, 1, 1 | 8, 8, 5 | 0, 1, 2, 3, 3 | 9, 9, 6, 0 | 0, 1, 2 | F, F, F |

几件从表里直接能看到、光读代码不容易确定的事：

1. **行序是排过的。** 第 1 步四个请求都在 prefill，按 token 数从小到大排（2、5、5、52）；第 2 步起 decode 请求排在最前。这是 MRv2 的 `sort_batch_req_ids`：先放本步 token 数等于 decode 长度（1）的请求，其余按 token 数升序。FlashInfer 一类后端依赖“decode 在前”来切分 batch。
2. **`logits_indices` 包含还没算完 prompt 的请求。** 第 1、2 步里，请求 2 的 prompt 还没算完（52/150、113/150），可它的最后一行（63）仍在 `logits_indices` 里：vLLM 照样为它算 logits、采样，然后丢掉结果。所以“`logits_indices` 里的行”不等于“有意义的采样行”。第 3 步的 39 才是请求 2 真正的 prompt 结束行（offset 149 = 150 − 1）。这正是 pLLM 用 `offset == prompt - 1` 而不是 `logits_indices` 来判断 prompt 结束的原因。
3. **`seq_lens` 是“算完本步之后”的长度**（`computed + scheduled`），也就是本步 attention 要看的 KV 长度，不是 prompt 长度。
4. **哪些是 prefill、哪些是 decode，有两种不同的说法。** `is_prefilling` 按阶段判断（`num_computed_tokens < num_prompt_tokens`），和旧 tLLM 的 `num_computed_tokens >= num_prompt_tokens` 判 decode 是同一回事。FlashInfer 的 `num_decodes / num_prefills` 却是按**本步的 query 长度**判断：长度不超过 `reorder_batch_threshold`（这里是 1）的都算 decode。第 2 步 FlashInfer 报告 `num_decodes=3, num_decode_tokens=3, num_prefills=1, num_prefill_tokens=61`，和 `is_prefilling` 一致；但一个只剩 1 个 token 没算的 prompt 块，在 FlashInfer 眼里是 “decode”，在 `is_prefilling` 眼里仍是 prefill（vLLM 源码里把这叫 short extend，默认按 decode 处理）。要知道“真正的阶段”，看 `is_prefilling` 或自己用三个数算。
5. **padding 在 FULL 图模式下会进入 metadata。** 第 5 步只有 3 个真实请求、3 个 token，被补到 4：`InputBatch.num_reqs=3, num_reqs_after_padding=4, num_tokens=3, num_tokens_after_padding=4`；metadata 里 `query_start_loc` 末尾重复（`0,1,2,3,3`，padding 请求的 query 长度为 0），`seq_lens` 补 0，`slot_mapping` 的 padding 位置是 −1（不写 KV），而 **`num_actual_tokens` 是 4——包含 padding**。vLLM 源码里那个字段旁边就有一句 TODO：“rename to num_tokens since it may be padded and this is misleading”。旧 tLLM 时代（MRv1）用 `num_actual_tokens` 排除 padding 是对的；在 MRv2 的 FULL 图模式下要用 `InputBatch.num_tokens`。eager 和 piecewise 模式用的是未补齐的大小。
6. **`slot_mapping` 是每个 token 写 KV 的物理位置**（块号 × 块大小 + 块内偏移）。第 1 步请求 1 的两个 token 写到 48、49，请求 0 的两个子请求各自写一份（16–20、32–36）——prefix caching 关着时，n=2 的两个样本各自算一遍同一个 prompt。

### 字段速查

**`CommonAttentionMetadata`**（所有后端共享，每步一份）：

| 字段 | 含义（实测） |
|---|---|
| `query_start_loc`、`query_start_loc_cpu` | 每个请求在扁平 batch 里的起点，长度 `num_reqs + 1`；GPU 和 CPU 各一份。FULL 图模式下包含 padding 请求（长度为 0 的段） |
| `seq_lens` | 算完本步后每个请求的 KV 长度（`computed + scheduled`）；padding 请求为 0 |
| `num_reqs` | 请求数；FULL 模式下是补齐后的数 |
| `num_actual_tokens` | token 总数；**FULL 模式下包含 padding** |
| `max_query_len`、`max_seq_len` | 本步最长的 query 和 KV 长度（后者可能是上界） |
| `block_table_tensor` | `[num_reqs, max_blocks]`，每个请求的 KV 块号 |
| `slot_mapping` | 每个 token 写 KV 的槽位；padding 为 −1 |
| `positions` | 每个 token 在请求里的位置（也就是 `offset`） |
| `is_prefilling` | 每个请求是否仍在 prompt 阶段（CPU 张量） |
| `seq_lens_cpu_upper_bound` | `seq_lens` 的 CPU 上界（没有投机解码时就是精确值） |
| `req_idx`（仅 0.32） | 每行请求在 runner 请求表里的下标，等于 `InputBatch.idx_mapping` |
| `dcp_local_seq_lens` 等 | decode context parallel 时每个 rank 本地的 KV 长度；未开启时为 None |
| `logits_indices_padded`、`num_logits_indices` | 只给 fast-prefill 一类特殊 builder 用；普通路径为 None |

**FlashAttention 后端的 `FlashAttentionMetadata`**：基本是上表的子集（`num_actual_tokens`、`max_query_len`、`query_start_loc`、`max_seq_len`、`seq_lens`、`block_table`、`slot_mapping`、`causal`），外加 FA3 用的 `scheduler_metadata` / `max_num_splits`（FA2 下为空）。`num_decode_reqs` 等字段在这个后端上一直是 0，不要拿来判断阶段。0.27.1 还有级联注意力的 `use_cascade`、`common_prefix_len`、`prefix_kv_lens`、`suffix_kv_lens` 和 `sliding_window`，0.32 去掉了这些。

**FlashInfer 后端的 `FlashInferMetadata`**：没有 `seq_lens` 和 `block_table`——它们在构建时已经被“plan”进了 FlashInfer 的 prefill / decode wrapper 对象里。剩下的是 `num_actual_tokens`、`slot_mapping`、`num_decodes / num_decode_tokens / num_prefills / num_prefill_tokens`（按 query 长度切分，decode 在前）、`prefill` / `decode`（两个 wrapper）、两个 q 的 dtype；0.32 多了 `kv_cache_layout`，0.27.1 多了级联用的 `use_cascade` / `cascade_wrapper`。

**`InputBatch`（MRv2）里 pLLM 实际使用的列**：`num_scheduled_tokens`、`num_computed_tokens_np`、`idx_mapping_np`（行 → 持久请求槽；0.27.1 是 int32，0.32 是 int64，同一批请求的槽号也不同）、`cu_num_logits_np`、`logits_indices`、`num_reqs / num_reqs_after_padding`、`num_tokens / num_tokens_after_padding`，加上 `req_states.prompt_len`。三个数都在里面。

### 多卡时

**张量并行（TP=2，两张 A100-SXM4-80GB，vLLM 0.27.1 与 0.32.0、FlashAttention 与 FlashInfer 四种组合都测了）。** 两个 rank 上每一步的 `InputBatch` 和 attention metadata 逐字段相同；唯一的区别在 KV cache 张量本身：每个 rank 只存一半的 KV 头（0.5B 的 2 个 KV 头变成每卡 1 个）。所以行标签可以在每个 rank 上独立算，不需要通信——pLLM 正是这么做的。

**以下来自读 vLLM 0.32 的源码，没有实测：** 数据并行（DP）时每个 DP rank 有自己的调度器和自己的 batch，彼此的行毫无关系；为了让各 rank 能用同样的 CUDA graph，它们会商定一个共同的补齐后 token 数，所以 padding 可能比单卡时多。decode context parallel（DCP）时，KV 按 rank 切分，`dcp_local_seq_lens` 给出每个 rank 本地的 KV 长度，而 `seq_lens` 仍是全局长度。prefill context parallel（PCP）走另一条 `pcp_manager.prepare_attn` 路径。pLLM 今天只支持 TP。
