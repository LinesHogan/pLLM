# 行模型：在连续批处理里找回“每个请求”

如果你只在单卡上写过 PyTorch 研究代码，你眼里的 batch 大概像一排整整齐齐的储物柜：张量的形状是 `[B, S, hidden]`，`x[0]` 永远属于第一个请求，`x[1]` 永远属于第二个请求。谁的东西锁在哪个柜子里，一目了然。

但在 vLLM 这样的生产级推理引擎里，当你满怀信心地在模型的第 0 层挂上一个 forward hook，打印出拿到的 `hidden` 形状时，那排储物柜不见了。你手里只有一个光秃秃的二维张量：`[64, 3584]`。

```text
你以为的 Batch（整齐的储物柜）：        vLLM 塞给 Hook 的张量（拍扁的传送带）：
┌───────────────────────────┐          ┌───────────────────────────┐
│ Req 0_0 (decode): [tok]   │          │ Row 0:     [ ??? ]        │
├───────────────────────────┤          │ Row 1:     [ ??? ]        │
│ Req 1_0 (decode): [tok]   │   ───►   │ Row 2:     [ ??? ]        │
├───────────────────────────┤          │ Row 3..60: [ ??? ]        │
│ Req 2 (prefill):  [t0..58]│          │ Row 61..63:[ 幽灵 Padding ]│
└───────────────────────────┘          └───────────────────────────┘
```

这 64 行里，哪一行是刚刚进来、正在做分块预填充（chunked prefill）的长文档？哪一行是已经聊了十轮、正在吐第 11 个字的请求？末尾甚至还藏着几行为了凑齐 CUDA graph 录制尺寸而塞进来的幽灵 padding。更要命的是，等千分之几秒后下一步再进来时，调度器洗了一次牌，所有人的行号全变了。

这是因为现代服务引擎用的是**连续批处理（continuous batching）**。请求随时到达、随时离开；每一步，调度器为每个活跃请求决定这一步算几个 token——正在生成的请求通常算 1 个（decode），刚来的请求可能一次算完整段 prompt（prefill），开了分块预填充的话只算其中一截——然后把所有人的 token 首尾相接，拍扁成一张 `[num_tokens, hidden]` 的长条送进模型。

如果你只是拿模型生成文字，这些细节都被引擎藏得好好的，你根本不用管。可只要你想在推理的同时干点有状态的事——比如像 ESamp 那样给每个请求在线训练一个轻量级蒸馏器（distiller），或者给每个请求维护一份外部记忆——你在写下第一行算子之前，就必须先回答一个灵魂问题：**我手里的这一行，到底是谁的？**

这篇文章讲的就是我们怎样回答这个问题。最终的答案落在了每个请求的三个整数上，简单得甚至让人有点不好意思；但在走到那一步之前，我们曾在 vLLM 的 C++ 内核与动态结构体构成的地下城里迷路了很久。我们先回到当年的案发现场看看那段弯路，然后再拿一个真实的 batch，把那三个整数能推演出的整个世界一行一行算出来。

> **实验与测量环境说明**：除特别注明外，本文引用的性能数据均在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0（开启 CUDA graph 与 FlashInfer 采样器）下测得，基座模型为 `Qwen/Qwen2.5-7B-Instruct`（BF16，隐藏维 3584），标准负载为 8 个并发 prompt × `n=16` 展开 = 128 个并行解码行，每请求生成 256 个 token（原生基线吞吐 5,894 tok/s，单步耗时 21.7 ms）。文中仅在对比固定宿主开销与展示底层探针时使用同配置的 `Qwen2.5-0.5B-Instruct`（单步耗时 3.0 ms，原生基线 42,355 tok/s）。

---

## 一场没有地图的地下城探险

我们刚在早期原型系统（当时叫 tLLM）里做 ESamp 时，需求非常具体：在每一个 decode 步，我们需要从第 0 层和最后一层把隐藏状态（hidden states）捞出来，并且必须知道每一行对应哪个请求，才能拿那个请求专属的 distiller 权重去算梯度、更新参数。

当时开源社区里倒是有别的项目能分清“哪一行是谁”，但无一例外，它们走的都是 `LogitsProcessor` 接口。

为什么我们不能走那条路？看一眼时间线就明白了：`LogitsProcessor` 要等整座 Transformer 大山全部跑完、连最后那层庞大的 LM head 词表投影都乘完了，才会被引擎调用。如果我们等到那时候才拿到“拆包（unpack）”信息，黄花菜都凉了——本来可以放在旁路 CUDA stream 上、跟模型前向计算并行重叠（overlap）的在线训练和状态读取，只能傻乎乎地排在采样器前头干瞪眼。

我们等不起那么晚，只能顺着调用栈往前追。

我们的直觉是：模型内部一定有人知道每一行是谁——至少 Attention 层知道！否则它怎么做到不让张三的 token 偷看李四的 KV cache？

顺着这个念头，我们一路摸进了 vLLM 的模型 runner，想看看注意力层是怎么把那张扁平张量拆开的。结果迎头撞上了第一堵墙：vLLM 的注意力计算全部封装在它自己的 C++/CUDA 内核里，拆包的细节藏在编译好的二进制文件里，在 Python 层完全隐形。

我们把注意力层的 Python 调用入口翻了个底朝天，发现传进内核的参数里，唯一可能装着这本“户口本”的，是一个叫 **`attn_metadata`（attention metadata）** 的结构体。拆包的信息，一定就藏在它里面。

但当我们扒开 `attn_metadata` 时，才发现自己跳进了一个没有地图的地下城。这个结构体的格式极其复杂，当时网上找不到哪怕一篇解析它的文章。更让人头疼的是，它完全是动态变化的，光靠静态读代码根本猜不出它在运行态的真实长相：
* 当一个 batch 里同时混着 prefill 和 decode 请求时，里面哪几段是 prefill、哪几段是 decode？
* 当 CUDA graph 为了对齐尺寸在末尾补了 padding 时，哪些计数把幽灵行算进去了，哪些没算？
* 当开启多卡并行时，这些字段又会怎么变？

那段时间我们几乎是在靠打断点度日。我们在 runner 里设下埋伏，构造各种刁钻的混排请求，一步一步单步调试、打印张量，硬生生靠肉眼把各个字段的脾气摸清楚了（为了不让后来的人再吃一遍这份苦，我把今天在 vLLM 0.27.1 和 0.32.0 上重新实测核对的字段解剖笔记完整放在了[文末附录](#附vllm-attention-metadata-字段解剖笔记)里）。

然而，好不容易读懂了 `attn_metadata`，第二堵墙紧跟着就立在了面前。

`attn_metadata` 是一个塞满了 GPU 张量引用的结构体。如果我们在前向传播途中把它里面的张量拷回 CPU，在 Python 里解析出“第 3 行属于请求 A”，再把挑出来的行号传回 GPU 去切分 `hidden`——这一来一回就是好几次主机与设备间的同步（Host-Device Synchronization）。在 GPU 上，一次同步等待动辄几十到上百微秒，而小模型一步 decode 总共也就 3 毫秒，再这么折腾几下，吞吐直接就废了。

被逼无奈，当时的方案只能和 GPU 缓冲区、CUDA graph 绑在一起做协同设计：
1. **抢在构造完成前动手**：不等 `attn_metadata` 完全造好，直接从构造它时用到的源头信息开始解析；
2. **CPU 与 GPU 分工解析**：绝不把所有解析逻辑都堆在 CPU 上，避免把 GPU 张量往回拉；
3. **图内轮询缓冲区（in-graph polling buffer）**：在 ESamp 当时的实现里，我们在显存里划出几块地址死死固定的暂存区（`decode_row_idx`、`decode_valid_mask`）。CPU 定期、异步地把解析出来的必要行号填进这块缓冲区；而在录好的 CUDA graph 内部，GPU 内核每一步直接去固定地址轮询读取，在设备端自己完成拆包。

在后来公开的旧版 tLLM 代码库里，还能看到当年这套精密又沉重的机关：如果活跃行数超过了预留缓冲区的大小，系统就会抛出一句老用户很眼熟的报错——`"Increase graph_scratch_rows"`。

### 怎么确认自己不是“以为懂了”？

逆向工程别人家的庞大系统，最可怕的不是看不懂，而是**“自以为看懂了”**。万一在某个奇葩的分块或混排边界上，我们挑行的算术偏了一行，拿李四的 hidden 训练了张三的 distiller，代码不会崩溃，甚至 loss 还会看似正常地下降，但算法逻辑已经彻底烂掉了。

为了睡个安稳觉，我们专门写了一个运行时验证脚本 `verify_v1_decode_rows_runtime.py`。它开头的几行注释，就是我们用来拷问自己的实验协议：

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

这个协议的逻辑极其朴素：
* **金标准（Gold standard）**：把一批 prompt **一个接一个单独喂给引擎**（采用 greedy 贪心解码保证确定性）。当引擎里只有一个请求时，根本不存在混排，“第 0 行是谁”毫无歧义。我们把每个请求在每个 decode 步的第 0 层输出抓下来存好。
* **候选方案（Candidate）**：把同一批 prompt **同时扔进引擎并行跑**，让连续批处理调度器随便洗牌，然后用我们写的拆包逻辑去第 0 层挑出每个请求对应的行。
* **生死线**：逐个请求、逐个 decode 步比对两边的隐藏状态均方误差（MSE）。哪怕有一个请求在某一步对不上，就说明我们还没真正搞懂 vLLM 的行布局。

我一直很喜欢这个小协议。它不依赖你对宿主系统内部的任何假设，只依赖“单请求运行时没有歧义”这一个铁打的事实。每当你要在一个复杂的黑盒系统里验证自己的定位逻辑时，这几乎是一张百试百灵的试纸。

### 寄生在私有接口上的伤疤

靠着断点调试和金标准验证，ESamp 跑起来了。但“手工追内部结构体”很快让我们尝到了苦头。

旧版 tLLM 的生产入口，是挂在 `GPUModelRunner._prepare_inputs` 这个私有函数的返回值上的。糟糕的是，这个函数返回一个长长的 Python 元组，而随着 vLLM 在 `<0.10`、`0.10`、`0.11+` 一路演进，这个元组里的元素个数一直在变！于是我们的代码里被迫长出了三个版本适配器，基类里甚至出现了这样一段靠数元组长度来猜字段的凄凉代码：

```python
# check-docs: skip: excerpt from tllm/runtime/vllm_patch/adapters/base.py in the old tLLM repository
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

看到 `if len(out) >= 6:` 了吗？这就是深入敌后、扒内部结构体留下的伤疤。我们对机制的理解是对的，但代码却脆弱地寄生在一个随时会变形的私有元组上。

在做 pLLM 的重构时，看着这段代码，我们心里只有一个念头：一定有更干净的办法。我们到底需要从 vLLM 里拿什么？

---

## 顿悟：发货单在装车前就印好了

让我们把历史包袱先放一放，拿显微镜看一个真实的 batch。

下面这个场景取自我们在 vLLM 0.32 上的实测探针（设 `max_num_batched_tokens=64` 以触发分块预填充）。系统里同时跑着三个用户发来的请求：
* **请求 0**：prompt 长 5 个 token，但用户要求并行采样两个回答（`n=2`），于是 vLLM 在内部把它裂变成了两个独立的子请求 `0_0` 和 `1_0`；
* **请求 1**：一段只有 2 个 token 的短 prompt；
* **请求 2**：一段长达 150 个 token 的长文档。因为每步最多只能算 64 个 token，它必须被切成好几块，分好几步算完。

在运行的**第 2 步**，前三个短请求都已经算完了各自的 prompt，正在一步吐 1 个字地做 decode；而请求 2 在第 1 步刚算完前 52 个 token，这一步正接着算接下来的 61 个 token。

此时，vLLM 拼出来送进模型的扁平张量一共 64 行：

```text
行 0       请求 0_0  （1 个 token）
行 1       请求 1_0  （1 个 token）
行 2       请求 1    （1 个 token）
行 3..63   请求 2    （61 个 token）
```

面对这张 64 行的表，如果我们要支撑起一个通用扩展层，我们需要回答四个问题：
1. **谁是谁？** 每一行对应哪个请求身份？
2. **什么阶段？** 请求 `0_0` 的这 1 个 token，究竟是 prompt 的一部分，还是已经进入生成阶段的 decode token？
3. **谁该去采样？** 请求 2 在这一步足足占了 61 行（行 3 到行 63），它的最后一行（行 63）该不该拿去给采样器抽第一个生成词？
4. **是不是炒冷饭？** 在真实服务里，如果显存 KV cache 突然不够用了，vLLM 会把某个请求临时**抢占（preempt）**踢出显存；等过一会儿腾出地方了，再把它的 prompt 连同之前已经生成的 token **从头到尾重新当成 prefill 算一遍**。对无状态的纯推理来说，重算一遍毫无影响；但对有状态的在线学习来说，如果不把这些重算的 token 揪出来屏蔽掉，同一段文本就会被你的训练器喂两遍！

当年我们为了搞清这四件事，一头扎进了 `attn_metadata` 的张量迷宫里。但某一天当我退后一步，看着调度器在发车前刚刚填好的那几列普通 CPU 数组时，我突然愣住了。

要回答上面所有问题，我们根本不需要碰 `attn_metadata` 里的任何一个 GPU 张量。对于这一步被调度的每一个请求，我们只需要**三个已经在 CPU 上的整数**：

* `scheduled`：本步调度器决定为它算几个 token；
* `prompt`：它的原始 prompt 总共有多长；
* `computed`：在今天这一步开始**之前**，历史上已经为它算过几个 token。

让我们把这三个数字填进刚才那个第 2 步的 batch 里，看看会发生什么奇迹：

```text
请求     scheduled    prompt    computed
0_0          1           5          5
1_0          1           5          5
1            1           2          2
2           61         150         52
```

现在，拿出一支笔，跟我们一起做小学加减法。

对于某个请求在本步被调度的第 `i` 个 token（`0 <= i < scheduled`），它在整条序列生命周期里的绝对逻辑位置（也就是 RoPE 位置编码的下标）是：

$$\text{offset} = \text{computed} + i$$

有了 `offset`，所有的谜底像多米诺骨牌一样哗啦啦全倒了：

* **看请求 `0_0`**：它本步只算 1 个 token（`i = 0`），所以 `offset = 5 + 0 = 5`。拿它和 `prompt`（5）比一下：`5 >= 5`！这意味着前 5 个 prompt token（下标 `0..4`）早就翻篇了，这一行如假包换，是**生成阶段的 decode 行**。请求 `1_0`（`5 >= 5`）和请求 `1`（`2 >= 2`）完全同理。
* **看请求 `2`**：它本步算 61 个 token（`i` 从 `0` 到 `60`），所以它们的 `offset` 从 `52` 一路排到 `112`。拿去和它的 `prompt`（150）比一下：全都小于 150！所以这 61 行**统统都是 prompt**。
* **那么请求 `2` 在这一步有“prompt 结束行”（用来触发第一次采样的那一行）吗？** 所谓 prompt 刚刚结束，充要条件只有一个：`offset == prompt - 1`（即 `149`）。这一步算到头才到 `112`，根本没到 `149`！哪怕 vLLM 的内部采样索引把第 63 行塞了进去，我们的算术也一眼就能看穿：这只是分块预填充的半山腰，绝不能触发采样或导出。
* **到了下一步（第 3 步）**：请求 `2` 的 `computed` 变成了 `113`，调度器把剩下的 `scheduled = 37` 个 prompt token 排了上来。这 37 个 token 的 `offset` 从 `113` 走到 `149`。当 `i = 36` 时，`offset` 稳稳当当地撞上了 `149 == 150 - 1`！Prompt 结束行在这一步、且只在这一步唯一地诞生了一次。分块预填充根本不需要任何复杂的跨步状态机，初等算术自己就保证了它的唯一性。

那最后一个难题——**抢占重算（Preemption Replay）**呢？

只需要给每个请求再加一个整数：**`frontier`（前沿水位线）**，用来记录这个请求自打进入引擎以来，我们的有状态逻辑**曾经见过的最大位置**。

```text
请求 2 的生命周期游标（prompt = 150）：

正常推进到 113：
  0                52                   113                150
  ├─────────────────┼────────────────────┤                  │
  │   Step 1 算完   │     Step 2 算完    ▲                  │
  └─────────────────┴────────────────────┴──────────────────┘
                                     frontier = 113

遭遇 KV 抢占！Step k 从 0 开始重算前 64 个 token (computed=0, scheduled=64)：
  0                         64          113                150
  ├──────────────────────────┼───────────┤                  │
  │ offset 0..63 < frontier! │           ▲                  │
  │  ──► 全部标为 RECOMPUTED │       frontier = 113         │
  └──────────────────────────┴──────────────────────────────┘
```

看上面的时间轴：正常跑完前两步后，请求 2 的 `frontier` 推到了 `113`。假设此时显存告急，请求 2 被无情踢走；几步之后它被重新捞回来从头重算，`computed` 瞬间归零，本步 `scheduled = 64`。当我们遍历这 64 个 token 时，算出来的 `offset` 是 `0..63`——它们全都严格小于 `frontier`（`113`）！

看到 `offset < frontier`，标签器立刻明白：“这几行是旧账重算，不是新知识。”于是给它们盖上 `RECOMPUTED` 的戳，把它们从训练和导出的名单里直接剔除，直到重算的游标重新跨过 `113` 那条水位线为止。

打个比方：连续批处理就像一辆快递货车，不同人家的包裹被胡乱塞进车厢里。当年我们干的事，是跳上已经开动的货车，挨个去扒包裹的包装纸（解析 GPU 上的 `attn_metadata`），想看清里面装的是什么；而今天我们恍然大悟——**在包裹被装上车之前，调度室的桌子上本来就放着一张印好的纸质发货单（`scheduled`、`prompt`、`computed`）。**

照着发货单贴好标签再装车，分拣连一个包裹都不用拆。事后回头看，这个道理简单得让人想拍大腿；可如果没有当年在 `attn_metadata` 里打断点的那段日子，我们大概也没法一眼认出：原来这三个数字，就已经是全部了。

---

## 今天的做法：找最稳的接缝，写最响的断言

一旦想通了“三个数加一个水位线决定一切”，代码的实现就从一场跟 C++ 内核斗智斗勇的泥潭战，变成了一件清爽愉快的工程活。

### 1. 选最不容易变的那个调用点

还记得旧版 tLLM 挂在 `_prepare_inputs` 上、被返回值元组长度折磨得死去活来的惨状吗？在给 vLLM 的新版 model runner（MRv2）写 pLLM 的子类（`pllm/backends/vllm/runner.py`）时，我们立下一条规矩：**绝不去钩信息最多、但签名飘忽不定的内部函数；我们要找那个签名最简单、在所有支持的 vLLM 版本上都纹丝不动的接缝。**

在 MRv2 runner 里，每一步当 `prepare_inputs` 刚刚把 `InputBatch` 填好、但**还没有**开始构造复杂的 `attn_metadata` 之前，vLLM 都会调用一次 `prepare_attn(self, input_batch: InputBatch)`。在 pLLM 支持的每一个 vLLM 版本（无论是 0.27.1 还是 0.32.0）上，它的签名都长得一模一样。我们就守在这里：

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

注意这个时机有多妙：此时模型的前向计算还没开始，CUDA graph 还没发射，`attn_metadata` 甚至还没生出来。我们直接从 `input_batch` 和 `req_states` 的 NumPy 数组里借出那三个整数，当场把本步所有行的标签算好、发布给会话（`session.prepare_input`），然后深藏功与名，调用 `super().prepare_attn(input_batch)` 让 vLLM 干它自己的活去。（对于旧版的 MRv1 runner，我们在 `pllm/backends/vllm/mrv1.py` 里做了薄薄一层适配，底层调的是完全相同的原生标签器。）

### 2. 把脆弱的假设变成响亮的 `assert`

既然我们不再靠猜元组长度过日子，那万一将来某天 vLLM 升级了调度协议，我们怎么保证自己不会悄无声息地挑错行？

答案写在 `pllm/backends/vllm/packed_rows.py` 的 `encode_mrv2_rows` 里——把我们对宿主引擎的所有假设，全都写成毫不留情的运行时断言：

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

在很多代码库里，`assert` 只是防御性的点缀；但在这段代码里，那五条 `assert` 是最有价值的契约文档：
* `num_draft_tokens == 0`：明确宣告我们假设没有投机解码（speculative decoding）的草稿 token；
* `np.array_equal(input_batch.cu_num_logits_np, np.arange(count + 1))`：要求累积采样行数严格是 `0, 1, 2, ..., count`，即**每个请求在这一步恰好对应一个采样位**；
* `input_batch.num_tokens_after_padding >= input_batch.num_tokens`：明确区分真实 token 数和为了凑齐 CUDA graph 补出来的幽灵 padding 行，而传给 `labeler.prepare` 的 `num_scheduled_tokens` 只包含真实请求，天然把尾部的 padding 拒之门外。

至于请求的身份（`RequestIdentity`，包含请求字符串 ID、第几个 prompt、第几个采样样本、路由键），我们在请求刚获准进入引擎（`admit`）时，就把它编码成一串紧凑的 `int64` 整数，并在 `add_requests` 时绑定到 vLLM 的持久请求槽位（`labeler.slots`）上。这样到了每一步的高频循环里，我们只传递纯数字的槽号数组 `labeler.slots[input_batch.idx_mapping_np]`，连一个 Python 字符串都不碰。

### 3. Rust 里的 30 行算术：一次遍历，两处复用

顺着 `labeler.prepare` 往下走，真正的贴标签工作落在了 `rust/pllm_core/src/packed_rows.rs` 里。

对于本步全部的真实 token，Rust 标签器会生成一张 7 列的 `int64` 标签矩阵：

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

而生成这张矩阵的核心循环，几乎就是我们上一节在草稿纸上手算过程的逐字翻译：

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

停下来欣赏一下这短短几十行代码。前面我们在纸上推导的每一条规则，在这里都有一行分毫不差的对映：
* `let offset = i64::from(done) + i64::from(token);` —— 算出当前 token 在请求生命周期里的绝对位置；
* `let replay = keep && offset < frontier;` —— 在保留状态的抢占策略（`keep`）下，一眼揪出回放重算的旧 token；
* `if !replay && offset == i64::from(length) - 1` —— 挑出唯一的那一行“prompt 结束行”塞进 `prefill`；
* `if !replay && offset >= i64::from(length) && token == count - 1` —— 挑出每个请求本步最新的生成行塞进 `decode`；
* `links.push(start as i64 - 1)` —— 记下每个请求本步的最后一行，作为采样器行与打包行之间的桥梁。

由于这段逻辑只读取只读的 NumPy 切片，完全不碰 PyTorch 张量或 CUDA 上下文，我们在进入 `self.label(...)` 时直接用 `py.allow_threads(...)` 释放了 Python 的 GIL。一次线性扫描，同时吐出 7 列完整标签、4 组预筛选好的活跃行集合（`decode`、`prefill`、两者之并 `both`、以及所有非重算的新算行 `fresh`），外加每一组行对应的“打包行号 → 槽内名次（rank）”映射表。

更妙的是，这个 Rust 标签器不仅服务于线上的 vLLM runner，它还有第二个每天都在高频调用它的用户：**CPU 契约检查器 `pllm.testing.check`**（见本系列第 12 篇《在 serving 引擎里信任一段用户程序》）。当我们在 CPU 上验证用户的代码是否满足“分块不变性”或“抢占重算不变性”时，测试框架并不是去另写一套拙劣的“vLLM 模拟器”，而是直接把一串 `(请求, scheduled, computed)` 喂给同一个 `RowLabeler`。生产环境跑什么算术，离线验证就跑什么算术。

---

## 当“行”成为所有 API 的第一维

一旦底层的标签器把账算清，“行（Row）”就顺理成章地升格成了 pLLM 编程模型里的一等公民。

在 pLLM 中，无论你写的是读取隐藏状态、修改残差流、在线训练参数，还是干预采样器，你收到的每一个按请求组织的张量，其**第 0 维永远是行（`[rows, ...]`）**。你再也不需要关心 continuous batching 怎么拼张量，也不用关心 CUDA graph 在尾巴上塞了多少行幽灵 padding——pLLM 已经在幕后用刚才算出的名次映射和身份索引，把属于每一行的激活值与属于同一个请求的状态版本（见第 3 篇《PTTS》）严丝合缝地对齐了。

```text
InputBatch (CPU)               RowLabeler (Rust)                用户代码收到的第 0 维 [rows, ...]
┌──────────────┐              ┌──────────────────┐             ┌────────────────────────────────┐
│ scheduled    │              │ 7 列标签矩阵      │  decode ──► │ pllm.hidden (tokens="decode")  │
│ prompt_len   ├──prepare()──►│ + 4 组选中行索引  ├──prefill──► │ pllm.hidden (tokens="prefill") │
│ computed     │              │ + 槽内名次映射表  ├──fresh───►  │ @pllm.export (tokens="all")    │
│ idx_mapping  │              │ + 采样行 links    ├──links───►  │ @pllm.add_logits / guide_...   │
└──────────────┘              └──────────────────┘             └────────────────────────────────┘
```

根据你在声明时选择的入口不同，“一行”会自动映射为那四组集合里的对应切片：
* **默认激活读取（`tokens="decode"`）**：每个处于生成阶段的请求各占 1 行（即刚才挑出的 `decode` 集合）；
* **预填充完成（`tokens="prefill"`）**：本步刚刚走完整个 prompt 的请求各占 1 行（即 `prefill` 集合，哪怕长文档被切成了十块，也只在最后一块完成的那一步出现唯一一次）；
* **采样器干预（`@pllm.add_logits`、`@pllm.guide_sampling`）**：本步需要抽样的每个请求各占 1 行；
* **全量导出（`@pllm.export` 设 `tokens="all"`）**：本步所有新算出来的 token 各占 1 行（即 `fresh` 集合，抢占重算的旧 token 会被 `frontier` 自动拦在门外，绝不重复导出）。

### 把身份分清楚，有时候直接就是性能

你可能会觉得，给行贴标签、绑身份只是一件为了“语义正确”而做的苦力活。但在真实的推理系统里，**对“行身份”的精确建模，有时候本身就能换来惊人的性能跃升。**

回到我们当年优化 ESamp 的历史现场。在我们的标准基准测试里，有 8 个并发 prompt，每个 prompt 要求并行采样 `n=16` 条输出序列（共 `8 × 16 = 128` 行）。早期版本的实现非常老实：既然 vLLM 把 `n=16` 拆成了 128 个子请求，那我们就每一步对这 128 行全部跑一遍训练。当时在 7B 模型上（8 个 prompt × `n=16`，256 token，`β=0`，`min_p=0.05`，历史测量），开启在线训练后的吞吐只有原生 vLLM 的约 **79%**。

后来我们突然反应过来：同一个 prompt 裂变出来的 16 条采样轨迹，共享的是完全相同的上下文起点！如果我们不是按子请求、而是按原始 prompt 来绑定状态，每个 prompt 只抽 1 行代表去更新 distiller，会怎么样？

那次改动（配合修复 graph capture，以及把可捕获的优化器换成 SGD）直接让 7B 上的吞吐比从约 **79% 飙升到了 95.4%**（原生约 5,048 tok/s，开启训练后达 4,817 tok/s）。在今天的 pLLM 里，这不再需要你去改底层引擎，只需要在声明状态时写下一行配置：

```python
key_fields=("prompt_index",)
```

短短一行声明，就把同一个 prompt 下的 16 个子请求路由到了同一份状态上，要维护和更新的状态从 128 份缩减到了 8 份。

再比如在多卡张量并行（Tensor Parallelism，如 TP=2 或 TP=4）下，由于 vLLM 的所有 TP rank 在每一步都会收到完全相同的调度指令，我们的 Rust 标签器可以在每张卡的 worker 进程上**各自独立、确定性地算出完全一致的行标签**，连一比特的跨卡通信都不需要（文末附录展示了我们在两张 A100 上实测 TP=2 的逐字段比对）。

---

## 天下没有免费的午餐：账单与边界

当然，把行模型立在 `prepare_attn` 和 `InputBatch` 之上，并不是没有代价的。作为系统设计者，我们得把账单摊开来算清楚。

**第一笔账是每一步的 CPU 准备耗时。**
虽然我们彻底告别了把 GPU 张量拷回 CPU 的同步噩梦，但每一步在 CPU 上借用 NumPy 列、跑一遍 Rust 标签器、再把算好的行索引和名次映射填进即将拷往 GPU 的固定表里，仍然需要花掉几百微秒的主机时间。在 RTX 4090 的宿主侧 profile 中（0.5B 模型，`β=0`），每次调用的中位数耗时大约是：`encode_mrv2_rows` 占 **122 µs**，会话 `prepare_input` 占 **174 µs**，准备隐藏层抓取槽位 `HiddenTaps.prepare` 占 **110 µs**。
这点时间贵不贵？这取决于你的模型有多大。这段 CPU 工作量只跟批次里的行数有关，跟模型大小无关。在 0.5B 这样的小模型上，GPU 跑完一步前向总共只要 **3.0 ms**，这四百多微秒的 CPU 准备工作虽然大部分能藏在 GPU 执行的阴影里，但在 profiler 的时间线上依然清晰可见；而当你换到 **7B 模型**时，一步 decode 长达 **21.7 ms**，这几百微秒的固定开销就彻底淹没在浩瀚的矩阵乘法背后，连一丝涟漪都看不见了。

**第二道边界是对宿主调度协议的依赖。**
尽管我们挑了签名最稳定的 `prepare_attn`，但 `encode_mrv2_rows` 读取的依然是 vLLM 内部 `InputBatch` 的属性列。只要 vLLM 的调度假设发生结构性改变，我们的断言就会立刻拉响警报。最典型的例子就是**投机解码（Speculative Decoding）**：一旦开启投机解码，一个处于生成阶段的请求在同一步里会同时塞进多个草稿（draft）token，也会产生多个采样位置，“每个请求在本步最多一个最新 decode 行、恰好一个采样行”的算术前提当场破裂。因此，今天的 pLLM 遇到投机解码会直接报错，请你关掉它；`assert input_batch.num_draft_tokens == 0` 是最后一道保险。

**第三道边界来自前缀缓存（Prefix Caching）与跨行操作。**
vLLM 的前缀缓存默认一条铁律：“只要两个请求的 token 前缀一模一样，它们算出来的 KV cache 就一定一模一样。”当你的程序只是旁路读取隐藏状态或训练自己的小网络时，这条铁律安然无恙；但只要你在模型内部挂载了会因请求而异地改写残差流、权重或注意力的逻辑，相同的 token 前缀在不同请求手里就会算出不同的 KV，前缀缓存的假设便不复存在——此时 pLLM 会强制要求关闭或隔离前缀缓存。
同样地，因为张量的第 0 维现在代表“恰好在这一步被拼在一起的不同租户”，你在写模型方法时必须守住一条铁律：**永远只沿着特征维（`dim=-1`）做归约，绝不能跨行（`dim=0`）归约。** 一句随手写下的 `h - h.mean(dim=0)` 或 `BatchNorm`，就会把张三和李四的命运死死绑在一起（至于我们怎么在你不小心写出这种代码时自动抓住你，那是第 12 篇的故事了）。

---

回顾从当年初探 ESamp 到今天 pLLM 的行模型，最让我感慨的转折点，并不是我们终于靠打断点破译了 `attn_metadata` 的全部密码，而是我们最终意识到**根本不需要去读它**。

当你试图往一个庞大、高速演进的底层系统里挂载扩展时，最容易犯的错误，就是被那个“看起来信息最全、离计算最近”的复杂结构体吸引过去。但它往往也是耦合最深、变化最剧烈的泥潭。更好的接缝通常藏在更上游的地方——在那里，数据还没有被打包成面向特定硬件内核的复杂形态，信息更少、更朴素，接口反而稳如磐石。

用三个在装车前就印好的整数，替代对一整车包裹的拆解——这是我们在 vLLM 里交过学费后，学到的最划算的一课。

---

## 附：vLLM attention metadata 字段解剖笔记

当年做 ESamp 时，我们曾因为找不到任何关于 vLLM v1 `attn_metadata` 的开源文档而吃尽苦头，只能靠打断点一点点摸索。虽然今天的 pLLM 已经完全不再读取 `attn_metadata`，但对于任何想要深入理解 vLLM 是如何拼装连续批处理、如何管理分页 KV 以及如何配合 CUDA graph 的系统开发者来说，它依然是最好的观察窗口。

我用一段几十行的探针脚本包住了 MRv2 runner 里构建元数据的 `model_state.prepare_attn`，在 **vLLM 0.27.1 和 0.32.0** 两个版本上重新实测了各个字段的运行态长相，整理成这份速查笔记，希望能帮后来在 vLLM 源码里探险的朋友省下几个通宵。

### 1. 探针实验设置

开启新版 runner（`VLLM_USE_V2_MODEL_RUNNER=1`），加载 `Qwen2.5-0.5B-Instruct`，设 `max_num_seqs=8`、`max_num_batched_tokens=64`（故意设小以逼出分块 prefill），关闭 prefix caching，开启 CUDA graph。同时向引擎提交三个请求：
* **请求 0**：`"Hello, my name is"`（5 个 token），设并行采样 `n=2`，各生成 5 个 token；vLLM 内部将它裂变为两个子请求，ID 形如 `0_0-…` 和 `1_0-…`；
* **请求 1**：`"Paris is"`（2 个 token），生成 5 个 token；
* **请求 2**：一段 150 个 token 的长 prompt，生成 2 个 token。

单卡实验在 RTX 4090 上完成（覆盖 0.27.1 / 0.32.0 两个版本 × FlashAttention / FlashInfer 两个注意力后端）；多卡张量并行（TP=2）在两张 A100-SXM4-80GB 上完成。

### 2. 五个真实步的逐帧解剖

下面是 vLLM 0.32.0 在 FlashAttention 后端下跑出的前五个真实步（0.27.1 除 `idx_mapping` 的具体槽号数值和整数位宽外完全一致）。表中 `computed` 即 `num_computed_tokens`，`prefilling` 即 `is_prefilling`：

| 步 | 图模式 | 行序（请求） | scheduled | computed | query_start_loc | seq_lens | logits_indices | prefilling |
|---|---|---|---|---|---|---|---|---|
| 1 | NONE (eager) | `1, 0_0, 1_0, 2` | `2, 5, 5, 52` | `0, 0, 0, 0` | `0, 2, 7, 12, 64` | `2, 5, 5, 52` | `1, 6, 11, 63` | `T, T, T, T` |
| 2 | NONE (eager) | `0_0, 1_0, 1, 2` | `1, 1, 1, 61` | `5, 5, 2, 52` | `0, 1, 2, 3, 64` | `6, 6, 3, 113` | `0, 1, 2, 63` | `F, F, F, T` |
| 3 | NONE (eager) | `0_0, 1_0, 1, 2` | `1, 1, 1, 37` | `6, 6, 3, 113` | `0, 1, 2, 3, 40` | `7, 7, 4, 150` | `0, 1, 2, 39` | `F, F, F, T` |
| 4 | FULL (graph) | `0_0, 1_0, 1, 2` | `1, 1, 1, 1` | `7, 7, 4, 150` | `0, 1, 2, 3, 4` | `8, 8, 5, 151` | `0, 1, 2, 3` | `F, F, F, F` |
| 5 | FULL (graph) | `0_0, 1_0, 1, (pad)` | `1, 1, 1` | `8, 8, 5` | `0, 1, 2, 3, 3` | `9, 9, 6, 0` | `0, 1, 2` | `F, F, F` |

盯着这张表看五分钟，你能直接看穿六件光靠静态读代码极难察觉的真相：

1. **行序是被悄悄重排过的。**
   看第 1 步：四个请求全在做 prefill，它们在扁平张量里的顺序并不是按提交顺序排的，而是按本步调度的 token 数**从小到大升序排列**（`2, 5, 5, 52`）。到了第 2 步，进入 decode 的三个请求（`0_0, 1_0, 1`）被整齐地排在了最前面，还在做分块 prefill 的请求 `2` 被扔在最后。这是 MRv2 里 `sort_batch_req_ids` 干的活：先把本步 token 数等于 decode 长度（1）的请求排在最前，其余按 token 数升序排列。源码注释写得明白：后端切分 decode 与 prefill 的 `split_decodes_and_prefills` 依赖“decode 类请求排在最前”的布局。
2. **`logits_indices` 里混着“假采样行”！**
   看第 1 步和第 2 步：请求 `2` 的 150 个 prompt token 才算到第 52 个和第 113 个，离算完差得远呢，但它的最后一行（行 63）依然赫然出现在 `logits_indices`（`[1, 6, 11, 63]` 和 `[0, 1, 2, 63]`）里！换句话说，vLLM 照样把第 63 行送进 LM head 算出了词表 logits、甚至跑了采样，最后再把结果丢掉。直到第 3 步，请求 `2` 算完了最后的 37 个 token（总进度达到 150），此时的 `logits_indices` 里的 `39`（对应 `offset = 149`）才真正是有意义的 prompt 结束采样行。**千万不要以为出现在 `logits_indices` 里的行就代表 prompt 已经算完了**——这正是 pLLM 坚持用 `offset == prompt - 1` 判定预填充结束的根本原因。
3. **`seq_lens` 代表“算完本步之后”的序列总长。**
   在每一行里，`seq_lens` 严格等于 `computed + scheduled`，也就是本步注意力内核需要扫描的 KV 有效总长度，而不是原始 prompt 长度。
4. **到底谁算 prefill、谁算 decode？内部有两种截然不同的口径。**
   `CommonAttentionMetadata.is_prefilling` 是按**真实生命周期阶段**判定的（即 `num_computed_tokens < num_prompt_tokens`），这与旧版 tLLM 以及今天 pLLM 的口径完全一致。但如果你切到 FlashInfer 后端去看 `FlashInferMetadata` 里的 `num_decodes` 和 `num_prefills`，你会发现它是按**本步的 query 长度**判定的：凡是本步调度长度不超过 `reorder_batch_threshold`（此处为 1）的请求，统统被归入 decode！在第 2 步里，两边恰好一致（`num_decodes=3, num_prefills=1`）；但如果某个分块 prefill 的长请求在最后一步恰好只剩 1 个 prompt token 没算，FlashInfer 会把它当成 decode 处理（源码里称为 short extend），而 `is_prefilling` 依然是 `True`。想判断算法的真实阶段，永远看 `is_prefilling` 或自己用三个数算。
5. **在 `FULL` CUDA graph 模式下，`num_actual_tokens` 居然包含幽灵 padding！**
   看第 5 步：请求 `2` 已经生成完 2 个 token 退役了，场上只剩 3 个活跃请求、共 3 个真实 token。但为了命中尺寸为 4 的 `FULL` CUDA graph，batch 被补齐到了 4：`InputBatch.num_reqs=3, num_reqs_after_padding=4, num_tokens=3, num_tokens_after_padding=4`。此时看 `attn_metadata`：`query_start_loc` 末尾重复了一次 `3`（`[0, 1, 2, 3, 3]`，表示第 4 个幽灵请求长度为 0），`seq_lens` 末尾补了 `0`，`slot_mapping` 末尾填了 `-1`（告诉内核不要写 KV cache），但名叫“实际 token 数”的 **`num_actual_tokens` 却是 `4`——它把 padding 行算进去了！** 翻开 vLLM 源码，你会看到那个字段旁边尴尬地躺着一行注释：`TODO(lucas): rename to num_tokens since it may be padded and this is misleading`。在旧的 MRv1 时代用 `num_actual_tokens` 过滤 padding 是对的，但在 MRv2 的 `FULL` 图模式下，必须用 `InputBatch.num_tokens`。
6. **`slot_mapping` 揭示了 `n>1` 的物理写入位置。**
   `slot_mapping` 记录每个 token 写入分页 KV cache 的物理槽位（`物理块号 × 块大小 + 块内偏移`）。在第 1 步里，请求 1 的两个 token 写进了槽位 `48, 49`；而请求 0 裂变出的两个子请求 `0_0` 和 `1_0`，分别写进了 `16..20` 和 `32..36`——当关闭 prefix caching 时，`n=2` 的两个样本会各自独立地把相同的 5 个 prompt token 算一遍、存两份。

### 3. 核心结构体字段速查表

**`CommonAttentionMetadata`**（所有注意力后端共享的公共底座，每步构造一份）：

| 字段名 | 实测含义与踩坑提示 |
|---|---|
| `query_start_loc` / `query_start_loc_cpu` | 形状 `(num_reqs + 1,)` 的 `int32` 张量（GPU 与 CPU 各一份），记录每个请求在扁平 batch 里的起始行号（前缀和）。`FULL` 图模式下包含长度为 0 的 padding 请求段。 |
| `seq_lens` | 形状 `(num_reqs,)` 的 GPU `int32` 张量，记录算完本步后每个请求的 KV 长度（`computed + scheduled`）；padding 请求处填 `0`。 |
| `num_reqs` | 请求数量；注意在 `FULL` 图模式下是补齐后的请求数（如第 5 步为 `4` 而非 `3`）。 |
| `num_actual_tokens` | 扁平张量的总行数；**在 `FULL` 图模式下包含尾部的 padding 行**，切勿在 MRv2 下拿它当真实 token 数。 |
| `max_query_len` / `max_seq_len` | 本步所有请求中最长的单步 query 长度与最大 KV 长度（后者在某些后端可能是上界）。 |
| `block_table_tensor` | `[num_reqs, max_blocks]` 的 GPU `int32` 张量，记录每个请求分配到的物理 KV 页表块号。 |
| `slot_mapping` | 长度为 `num_actual_tokens` 的 GPU `int64` 张量，记录本步每个 token 在 KV 池中的物理写入槽位；padding 行处固定为 `-1`。 |
| `positions` | 长度为 `num_actual_tokens` 的 GPU `int64` 张量，即每个 token 在各自序列中的逻辑位置 `offset`。 |
| `is_prefilling` | 形状 `(真实请求数,)` 的 CPU `bool` 张量，标记每个真实请求是否仍处于 prompt 预填充阶段。 |
| `seq_lens_cpu_upper_bound` | `seq_lens` 在 CPU 侧的上界拷贝（无投机解码时即精确值）。 |
| `req_idx`（仅 vLLM 0.32） | `int64` NumPy 数组，记录每个真实请求在 runner 持久请求表中的槽位下标，等同于 `InputBatch.idx_mapping_np`。 |
| `dcp_local_seq_lens` 等 | 开启 Decode Context Parallel 时每个 rank 本地的 KV 长度；未开启时为 `None`。 |
| `logits_indices_padded` / `num_logits_indices` | 仅在 fast-prefill 等特殊 builder 路径下使用；常规路径下恒为 `None`。 |

**后端专属 Metadata 的差异：**
* **`FlashAttentionMetadata`（FlashAttention 后端）**：基本是公共底座的子集（含 `num_actual_tokens`、`max_query_len`、`query_start_loc`、`max_seq_len`、`seq_lens`、`block_table`、`slot_mapping`、`causal`），外加供 FA3 调度的 `scheduler_metadata` / `max_num_splits`。**特别注意**：该结构体里的 `num_decode_reqs`、`num_prefill_reqs`、`num_decode_tokens`、`num_prefill_tokens` 在该后端下**恒为 0**，千万不要拿它们来判断阶段！此外，0.27.1 中用于级联注意力的 `use_cascade`、`common_prefix_len` 等字段在 0.32 中被删掉了（`rswa_*` 字段两个版本都有）。
* **`FlashInferMetadata`（FlashInfer 后端）**：结构体里直接去掉了 `seq_lens` 和 `block_table`——因为它们在构建阶段就已经被提前编译（plan）进了 FlashInfer 的 `prefill` 和 `decode` 两个 C++ wrapper 对象内部。留下的字段里，`num_decodes / num_decode_tokens / num_prefills / num_prefill_tokens` 是真正在工作的（按本步 query 长度切分，要求 decode 行严格排在前面）。
* **`InputBatch`（MRv2 中 pLLM 真正借用的源头）**：包含 `num_scheduled_tokens`、`num_computed_tokens_np`、`idx_mapping_np`（在 0.27.1 中为 `int32`，在 0.32.0 中升级为 `int64`）、`cu_num_logits_np`、`logits_indices`、`num_reqs / num_reqs_after_padding`、`num_tokens / num_tokens_after_padding`，配合 `req_states.prompt_len` 即可拿到全部三个核心整数。

### 4. 多卡并行时会发生什么？

我们在两张 A100-SXM4-80GB 上对 **张量并行（TP=2）** 进行了全组合实测（覆盖 vLLM 0.27.1 与 0.32.0 × FlashAttention 与 FlashInfer）。结论非常干净：**在每一步，Rank 0 和 Rank 1 上的 `InputBatch` 与 `attn_metadata` 的每一个字段、探针打印出的每一个张量数值都完全相同。** 和单卡相比，唯一的物理区别在于 KV cache 张量本身的形状——每张卡只持有总 KV 头数的一半（例如 0.5B 模型的 2 个 KV 头变成了每卡 1 个）。因此，行标签在每个 TP rank 上各自独立计算是 100% 确定且对齐的，无需任何跨卡同步。

至于其他并行模式（基于 vLLM 0.32 源码分析，pLLM 目前仅支持 TP）：在**数据并行（DP）**下，每个 DP rank 拥有独立的调度器和完全不相干的 batch，但为了能协同执行步调一致的算子，各 rank 会协商一个共同的补齐后 token 数，因此尾部的 padding 行可能会比单卡时更多；在 **Decode Context Parallel（DCP）**下，历史 KV 沿序列维切分到不同卡上，`dcp_local_seq_lens` 会记录本卡持有的局部 KV 长度，而 `seq_lens` 依然保持全局总长度。
