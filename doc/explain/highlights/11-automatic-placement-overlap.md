# 自动放置与重叠

我们曾在代码库里建起了一座宏伟的编译器大教堂：把多面体编译器的标准基石 ISL（Integer Set Library）连同它的高精度整数库——一共 1,530 个文件、457,770 行 vendor 代码（大部分是 C）——完整地引进了仓库，又在上面用 Rust 搭起了一整套 Region IR 中端，配齐了符号依赖分析、调度空间搜索、流划分、生命周期保留分析与容量规划。

那是一台精密、体面、学术品味拉满的机器。我们认真地写了实现、补齐了测试、撰写了架构文档。

然后，我们做了一次诚实的盘点，亲手把这 457,770 行 vendor 代码彻底删光了。一同退役的还有 19 个 Rust 模块（净减 7,171 行）、44 个 Python 实现文件（生产代码从 18,320 条 AST 语句减到 14,172 条，净减 4,148 条；测试净减 2,033 条语句）。取而代之的，是 Rust 调度器里一段只有 40 行的 Bellman-Ford 最长路松弛，外加一个基于 PyTorch FX 的轻量级仿射代数识别器。

为什么一个旨在“自动调度与流重叠”的系统，会先走进多面体编译器的迷宫，最后又用 40 行基础算法换掉四十多万行工业级代码？在回答这个问题之前，让我们先退回起点，看看把一段用户程序塞进高速运转的推理引擎时，我们真正面对的物理问题到底是什么。

> **实验环境说明**：除特别注明的跨机多卡或历史对照实验外，本文引用的基准测量均在单张 NVIDIA RTX 4090（24 GiB）、vLLM 0.32.0（开启 CUDA Graph 与 FlashInfer 采样器）、8 个 prompt × `n=16`（共 128 条并发序列）、生成 256 个 token 的设置下完成，取预热后 3 轮端到端吞吐的中位数。主测量模型为 `Qwen/Qwen2.5-7B-Instruct`（BF16，原生图模式步耗时约 21.7 ms，吞吐 5,894 tok/s）；需要放大展示微秒级固定调度开销时，会对照同配置下的 `Qwen/Qwen2.5-0.5B-Instruct`（原生图模式步耗时约 3.0 ms，吞吐 42,355 tok/s）。

---

## 一、 三个问题，其实是一个问题

当你在 Hugging Face 的单步循环里写研究代码时，“一段逻辑什么时候跑”根本不是个问题——你写在哪一行，Python 解释器就执行到哪一行。但在连续批处理的服务引擎里，一旦你允许用户挂载自定义的状态读写、在线训练与外部查表，每一次生成步（decode step）都会迎头撞上三个问题：

1. **它在一步里的什么时候跑（When）？**
2. **它跑在哪条 CUDA stream 上（Where）？**
3. **它写出的新状态，什么时候对其他计算可见（Visibility）？**

拿贯穿本系列的在线蒸馏负载 ESamp 来说。每走完一步前向，它要拿第 0 层和最后一层的隐藏状态（hidden states）去训练一个小网络（算损失、反向传播、跑分页 AdamW 更新权重）；而在采样器掷骰子之前，它又要拿这个小网络的预测去修正词表分布（[《采样尾声融合》](07-sampler-epilogue.md)）。再比如像 Engram 那样的稀疏记忆模型，需要在进入模型消费层之前，按输入 token 的 n-gram 从一张大表里捞出对应的嵌入向量（[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)）。

如果把在线训练老老实实串在主 CUDA stream 上，逻辑最简单，但主干前向每一步都要干等反向传播和优化器跑完。在 0.5B 模型上，一步前向总共才 3.0 ms，多等几百微秒就意味着吞吐肉眼可见地往下掉。那如果把训练放到另一条旁路 stream（`side` stream）上异步跑呢？下一步的模型前向马上就要开始了，那时候旁路上的 AdamW 还没跑完，前向到底该读哪个版本的权重（[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)）？同样地，查表任务只依赖一步开头的输入 token ID，能不能在前向开始之前就发到旁路流上预取？当然能，但它必须在模型内部第一层消费它的挂载点之前到达。

起初，我们是用手写的启发式规则来回答这三个问题的：在采样后阶段写一段重叠特判，在签名校验里硬编码“哪些组合允许异步”，在 Python 执行器里塞几个放置标志位。但每接入一种新算法——比如把优化器更新换成超网络生成每请求的 LoRA 权重，或者让一个滞后一步的训练结果喂给下一步的覆盖层——我们就得去核心分支里再打一块特判补丁。

当你手写了第三条特判规则时，任何系统工程师心里的警钟都会敲响：**把具体 workload 的习惯固化成框架里的特判分支，这条路走不远。** 我们需要一个有原则的自动方法，让用户只需要用贴近原生 PyTorch 的方式声明“我读什么输入、写什么状态、作用在模型的哪个阶段”，剩下的放置、分流与版本边界，全部由规划器根据数据依赖自动推导出来。

慢着——在我们一头扎进算法选型之前，先把这三个问题并排画在白板上仔细看一眼：

* “什么时候跑”，问的是任务 $u$ 在单步时间轴上的坐标区间；
* “哪条流”，问的是任务 $u$ 和消费它结果的任务 $v$ 在时间轴上隔了多远、中间有没有足够的算力空隙可以并行；
* “什么时候可见”，问的是写者任务 $u$ 和读者任务 $v$ 在跨步时间轴上相距几个生成步（step distance）。

看出来了吗？**这三个看似独立的问题，在数学上是完全同构的。它们都在问同一件事：每个任务在一条带有跨步循环的时间线上，处于什么几何位置。**

只要把这条时间线的代数结构刻画清楚，三个问题就会同时迎刃而解。

---

## 二、 多面体大教堂的幻觉与一次冷酷的盘点

当时为了追求“最通用、最有理论根基的自动调度”，我们翻开了高级编译器教材。在编译器领域，处理循环迭代与数据依赖的皇冠明珠是**多面体模型（Polyhedral Model）**。

多面体模型的思想非常漂亮：它把循环的每一次迭代看作高维整数空间 $\mathbb{Z}^n$ 里的一个格点，把读写依赖看作格点之间的仿射关系（Affine Relations），然后动用整数线性规划（ILP）求解器，在多面体空间里自动搜索合法的循环分块（tiling）、歪斜（skewing）、融合与多流并行切面。这也是 Pluto、PPCG、LLVM Polly 等经典编译器的底层引擎。

我当时想：既然一步接一步的生成循环也是循环，层与层之间的遍历也是循环，那把整个推理过程建成一个 Region IR，交给多面体基础库 ISL 去求最优调度，岂不是一劳永逸？于是，我们干脆利落地把 45 万行 ISL 源码请进了仓库。

这台大机器运转了两个月。直到有一天，我看着日渐膨胀的代码库，决定停下来做一次彻底的审计。我只问了一个简单得近乎残酷的问题：

**在所有我们已经测出真实吞吐收益的优化里，到底有哪几分加速是靠 ISL 算出来的？**

把代码一层层剥开后，盘点的结果让我出了一身冷汗：

1. **第一项真实收益：把状态递推变成并行前缀扫描（Parallel Prefix Scan）。** 在 In-Place TTT 和 GatedDelta 这类线性状态转移负载上，把逐 token 的串行递推改写成关联前缀扫描，在微基准测试里（历史测量）拿到了显著加速（单头 $64 \times 64$ 状态转移从 1.44 ms 降到 0.22 ms，快了 $6.57\times$；14 头快了 $1.85\times$；完整转移从 5.88 ms 降到 1.76 ms，快了 $3.34\times$）。但这套优化是怎么认出一段代码可以做前缀扫描的？去翻当时的 `affine_scan.rs`，里面是一个纯 Rust 写的表达式树模式匹配器，**调用 ISL 的代码是零行**——它只是在最外层套了一个 Region IR 的壳子。
2. **第二项真实收益：ESamp 的旁路训练与主干前向重叠。** 当时的可见性分析模块确实调用了 ISL，但它是怎么调的呢？它把一个整数滞后量 `k`（即 `publish_lag`）格式化成一段 ISL 集合字符串，传进 C 库里去求解不等式 `0 <= b < t` 是否成立——翻译成人话，它动用了 45 万行的整数规划求解器，去算了一道小学一年级算术题：**整数 `lag` 是不是大于等于 1！**
3. **那些真正重度调用 ISL 的模块呢？** 区域分块、自动多流划分、生命周期保留分析、多面体容量分析——在所有的生产级 workload 上，它们要么每次搜索完只吐出一个平庸的恒等调度（Identity Schedule），要么干脆在死代码路径上从来没被触发过。

更糟糕的是，当用户真的触碰到暴露在外的多面体入口时，这台大机器直接翻车了。当时有一个 `ModelScan` 接口，会把请求序列展开成 ISL 整数集程序去求解。我们在 RTX 4090、0.5B 模型（eager 模式，prefill 长度 1024，分块 128，作用在 5 个目标层）上跑了一次实测：
* 1 条序列：耗时 **146 ms**；
* 2 条序列：耗时 **293 ms**；
* 4 条序列：直接抛出 `ISL quota exceeded`（求解步数超限）；
* 8 条序列：当场崩溃报 `LibISLError`。

作为对比，在完全相同的 prefill 规模下，一条不经过 ISL、直接在模型前向内部做精确矩阵更新的路径，跑完只需要 **19.6 ms**（两者数学定义略有不同，但量级差了整整一个世代）。而在层循环与自适应深度（[《自定义层循环与状态传播》](10-custom-layer-loops.md)）那一侧，当我们把逐 token 的 Region IR 执行器撕掉、换成直接批量调用 vLLM 原生层之后，recirculation 的 decode 步耗时从 26.23 ms 骤降到 3.26 ms，layer loop 的 prefill 更是从荒唐的 3,550.81 ms 降到了 14.76 ms。

为什么会这样？多面体模型明明是编译器领域的杰作，为什么在这里变成了一个沉重而脆弱的累赘？

答案在于**问题的几何形状完全错了**。多面体模型是为“任意维度的密集仿射循环嵌套”（比如三层矩阵乘或图像卷积 stencil）设计的；但在一个大模型服务引擎里，外层是连续批处理动态洗牌的请求，内层是被 CUDA Graph 焊死的巨型融合算子。留给扩展层去调度的，根本不是什么五维空间里的多面体格点，而是**一条一维时间轴上的几个到几十个粗粒度任务，外加固定步长的跨步状态引用**。

拿一把用来切高维晶体的激光手术刀去切一根一维的香肠，你不仅切得不会更快，而且每次出了 bug，你的调用栈都要穿过四十多万行 vendor 代码的茫茫黑盒。

想明白这一点后，我们做出了那个决定：**把 ISL 和 Region IR 全部删掉。** 但删除有一个不可妥协的前提——我们绝不退回“为每个 workload 写死特判规则”的旧路。没有预见到的用户递推依然要被自动加速，任务的放置窗口与显存账本依然要由规划器有原则地自动推导。

只不过这一次，我们要用刚刚好匹配问题几何形状的最小工具。

---

## 三、 一维时间线、一致依赖与 40 行最长路

让我们把多面体编译器的复杂术语全部忘掉，重新在白板上画一步 decode 的真实面貌。

在 pLLM 里，引擎的每一个生成步（Step $t$）都被切分成 $M=7$ 个严格全序的时间轴**锚点（Anchors）**。每两个相邻锚点之间的空隙，就是一个声明阶段（`pllm.Phase`）：

```text
Step t Timeline (M = 7 Ordered Anchors):

  0:request     1:model       2:state       3:sampler      4:sampler     5:token      6:next
   .prepare     .execute      .publish      .pre_draw      .complete     .commit     .prepare
──────┼───────────┼─────────────┼──────────────┼───────────────┼─────────────┼───────────┼──────►
      │           │             │              │               │             │           │
   [Tokens]   [Forward]     [Hiddens]      [Guidance/      [Sampled]     [Sampled    [Retire &
    Ready      Running       Published      Bias Due]        Done         Tokens]     Next Step]
```

在这条时间线上，每一类数据都有它确切的“出生时刻”：
* 输入 token（`pllm.input_tokens`）在 `0: request.prepare` 就已经就绪；
* 模型各层的隐藏状态（`pllm.hidden`、`pllm.residual`）要等前向跑完，在 `2: state.publish` 才诞生；
* 采样引导与 logits 偏置必须在 `3: sampler.pre_draw` 之前交卷；
* 而本步抽出的新 token（`pllm.sampled_tokens`）要等到 `5: token.commit` 才真正落袋为安。

因此，任何一个任务 $u$ 都有一个初始的合法运行区间 $[a_u^{\min}, a_u^{\max})$：最早不能早于它所声明的阶段起点和它所有输入数据的出生锚点，最晚不能晚于它声明的阶段终点。

那么，任务与任务之间的依赖边长什么样？

如果把第 $t$ 步的第 $a$ 个锚点映射成一条全局一维时间轴上的整数刻度 $T = t \cdot M + a$，你会惊讶地发现，整个系统里所有的依赖关系只有极其规整的两种：

1. **同一步内的数据依赖（Step Distance $k = 0$）**：任务 $v$ 读取了任务 $u$ 在本步算出的结果，要求 $v$ 的锚点不早于 $u$ 加上最小间隔 $d$（即 $a_v - a_u \ge d$）。
2. **跨步的状态版本依赖（Step Distance $k \ge 1$）**：任务 $v$ 在第 $t$ 步读取了任务 $u$ 在第 $t-k$ 步写入的状态版本（也就是 `publish_lag = k`）。换算到全局整数刻度上，这意味着：
   $$(t \cdot M + a_v) - ((t - k) \cdot M + a_u) \ge d$$

把左右两边的 $t \cdot M$ 消掉，步数变量 $t$ 瞬间消失了！不等式坍缩成了一个与循环步数完全无关的常数差分约束：

$$a_v - a_u \ge d - k \cdot M$$

```text
Unrolled Steps (Infinite Graph):              Reduced Uniform Graph (Finite, Weight = d - k*M):

Step t-1:  [Task u @ a_u]                                   weight: d - 0*M = d
                 │                                       ┌───────────────────────┐
                 │ (lag k = 1, sep = d)                  │                       ▼
                 ▼                                  ┌─────────┐             ┌─────────┐
Step t:    [Task v @ a_v] ──(k=0, d)──► [Task w]    │ Task u  │             │ Task v  │
                                                    └─────────┘             └─────────┘
                                                         ▲                       │
                                                         └───────────────────────┘
                                                            weight: d - 1*M
```

早在 1967 年，Karp、Miller 和 Winograd 在研究一致递推方程（Uniform Recurrence Equations）时就已经证明过这类结构的性质：当循环依赖的距离向量全是常数时（即**一致依赖，Uniform Dependence**），无限展开的跨步依赖图可以直接折叠成一张只有有限个节点的**约化依赖图（Reduced Dependence Graph）**。在这张小图上，求所有任务合法的 `[最早锚点, 最晚锚点]` 窗口，根本不需要什么整数线性规划求解器——它等价于求图上的**最长路（Longest Path）**。

看看 `rust/pllm_core/src/planning/scheduler.rs` 里的 `propagate` 函数。当年 45 万行的 ISL 调度推导，今天完完整整地浓缩在了这 40 行 Rust 代码里：

```rust
/// Reduced uniform dependence graph: u(t) -> v(t+k) has weight d-k*M.
fn propagate(
    bounds: &[(u32, u32)],
    edges: &[(usize, usize, i64)],
) -> Result<Vec<(u32, u32)>, String> {
    let mut lo: Vec<_> = bounds.iter().map(|b| i64::from(b.0)).collect();
    let mut hi: Vec<_> = bounds.iter().map(|b| i64::from(b.1) - 1).collect();
    for pass in 0..=bounds.len() {
        let mut changed = false;
        for &(u, v, weight) in edges {
            let lower = lo[u] + weight;
            let upper = hi[v] - weight;
            if lower > lo[v] {
                lo[v] = lower;
                changed = true;
            }
            if upper < hi[u] {
                hi[u] = upper;
                changed = true;
            }
        }
        if lo.iter().zip(&hi).any(|(a, b)| a > b) {
            return Err(
                "a task reads a value (sampled tokens, hidden states or another task's \
                 result) that exists only after its phase ends; declare it in a later phase"
                    .into(),
            );
        }
        if !changed {
            return Ok(lo
                .iter()
                .zip(&hi)
                .map(|(&a, &b)| (a as u32, b as u32 + 1))
                .collect());
        }
        if pass == bounds.len() {
            return Err("placement constraints contain a contradictory cycle".into());
        }
    }
    unreachable!()
}
```

我非常喜欢这 40 行代码。第一行文档注释 `u(t) -> v(t+k) has weight d-k*M` 就是全部的数学理论；中间是一个教科书般干净的双向 Bellman-Ford 松弛——前驱的下界 `lo[u] + weight` 往后推着后继的下界 `lo[v]`，后继的上界 `hi[v] - weight` 往前压着前驱的上界 `hi[u]`。

更妙的是它失败的方式。多面体求解器失败时，往往只会吐出一个冰冷的“无解（unbounded / empty set）”或配额超限；而这个 40 行的最长路松弛只有两种失败出口，每一种都精确对应着一句用户立刻能照着修代码的工程诊断：
1. **某个任务的下界越过了上界（`lo > hi`）**：说明你在某个阶段声明了一个任务，但它去读了一个要等该阶段结束之后才诞生的值（比如在采样前的方法里去读 `sampled_tokens`）。报错直接告诉你：`declare it in a later phase`。
2. **松弛了 $|V|$ 轮之后还在变（`pass == bounds.len()`）**：根据 Bellman-Ford 定理，图里存在正权环——也就是说，你的跨步读写约束首尾相咬，形成了一个时间穿越的死锁环（`placement constraints contain a contradictory cycle`）。

一旦 `propagate` 算出了每个任务的合法窗口，接下来的 `place` 函数就顺理成章了：

```rust
fn place(
    graph: &SemanticGraph,
    timeline: &Timeline,
    index: &GraphIndex,
    state: Option<&StateRepresentationDecision>,
    implementations: &[&ImplementationCandidate],
) -> Result<Solution, String> {
    let mut order = index.order.clone();
    order.sort_by_key(|&id| index.windows[id].0);
    // ...
    select_lanes(graph, timeline, &index.windows, &mut tasks);
    select_prefetch(graph, &mut tasks);
    plan_memory(graph, timeline, index, state, &tasks).or_else(|_| {
        // Memory fallback preserves the selected logical read boundaries.
        for task in &mut tasks {
            task.lane = Lane::Main;
        }
        plan_memory(graph, timeline, index, state, &tasks)
    })
}
```

先把每个任务放到它合法窗口的最早锚点 `windows[id].0` 上；然后调用 `select_lanes` 和 `select_prefetch` 分配执行车道（`Main` 还是 `Side`）；最后调用 `plan_memory` 沿着时间线模拟一遍临时张量与工作区的生老病死，核对显存峰值是否超出预算。

注意看那个 `.or_else(|_| ...)`。把任务放到旁路流上异步跨越步边界时，它的输入、中间激活与状态版本必须活得更久，占用的显存峰值自然更高。如果 `plan_memory` 发现旁路重叠的峰值撑破了显存预算怎么办？它会自动把所有任务退回 `Lane::Main` 串行执行，再算一次显存——**但严格保留已经选定的逻辑读边界（`read_origin` 与 `read_distance`）**。也就是说，显存吃紧时你只会退化回串行吞吐，而程序的数学语义（读哪一步的版本）连一个比特都不会变！如果退回主流依然装不下，规划器才会拒绝启动，并明确报出所需的字节数。（顺便说一句，由于我们的 24 GiB 测试卡预算充裕，这条自动回退主流的兜底路径在吞吐上具体会折损多少，目前还没有专门的压测数字。）

在这里，我们要保持绝对的诚实：**在这套新调度器里，真正“自动求解”的是锚点窗口推导、版本冲突检测与显存生命周期核算；至于“哪个任务上旁路流（`select_lanes` / `select_prefetch`）”，今天依然是几条写明的工程规则。**

为什么不用代价模型去自动搜索多流划分？因为服务引擎里的物理规律太清晰了，写成明确的条件判断比黑盒搜索更稳健：
* **状态写者上旁路（`select_lanes`）**：只有当存在跨步滞后（`publish_lag >= 1` 或模型内读取）、没有自定义前向或显式可见性覆盖、第一个写状态的任务锚点在 `model.execute` 之后、且后续任务不喂给采样器也不读宿主服务时，规划器才会把从第一个写状态任务开始的后缀全部切到 `Lane::Side`。如果 `publish_lag=0`，所有更新必须在本步结束前立即可见，旁路训练自然关闭，全部留在主流串行。
* **只读查表预取（`select_prefetch`）**：如果一个任务只读取 `input_tokens` 和不可变状态（`immutable`），并且具备进入 CUDA Graph 的能力（`compute/cuda_graph@1`），它就会被放到 `Lane::Side`，在一步开头的 `request.prepare` 立即发射，与当前步的模型前向并行。
* **同一锚点内的多流列表调度（List Scheduling）**：如果同一个锚点上挂了五六个互不依赖的旁路任务，要不要给它们开多条旁路流？我们选择**不做**，直到真有生产 workload 需要它为止。

还有一条设计边界，是我们被真实测量狠狠教育之后主动收缩的。在实现这套放置调度器的第一个版本时，我们曾觉得“锚点越细越通用”，于是在 eager 后端（关闭 CUDA Graph 时）上为模型的每一层都注册了细粒度的执行锚点，允许调度器把任务精确塞进“第 $l$ 层之后”。结果一跑基准测试（RTX 4090，0.5B，batch $8 \times 32$，eager），几个状态生成器的相对原生吞吐全面倒退：超网络（hypernet）从 0.722 掉到 0.656，混合专家（mixture）从 0.723 掉到 0.667，AdamW 从 0.554 掉到 0.505。原因很简单：在 Python 里做逐层 hook 检查与旁路分发，平白无故给每轮增加了约 14% 的宿主 CPU 开销（同一分支里的显存预留修正倒是把超网络的预留从 797 MB 压到了 24 MB，这部分保留了下来）。

面对这组数字，我们立刻修改了铁律：**调度器只能把任务放在后端显式声明的执行点上；eager 后端只提供粗粒度锚点，逐层分发代码直接删除，连配置开关都不留。** 当美妙的编译器原则撞上残酷的硬件开销时，原则必须向测量低头。

---

## 四、 认识代数，不认识名字：任务内部的自动加速

搞定了任务之间的放置与分流，那任务**内部**的计算呢？删掉 Region IR 之前，我们承诺过一条底线：“即便是一个我们从未见过的用户自定义递推，只要它在数学上可并行，框架依然要能自动加速它，绝不能靠认 workload 的类名来硬编码。”

今天承担这份职责的，是 `pllm/execution/scan_analysis.py`。它的设计哲学可以用十个字概括：**认识代数结构，不认识名字。**

当用户写下一个沿序列方向逐位置更新的递推函数 $S_t = \text{step}(S_{t-1}, x_t)$ 时，我们不再把它翻译成什么自定义的 Region IR，而是直接用 PyTorch 原生的 `make_fx` 把这个单步函数追踪成一张符号计算图（FX Graph），然后对状态变量（carry $S$）做一次轻量级的符号线性度分析：

1. **剥离无关子图（Carry-independent hoisting）**：计算图里凡是不依赖上一时刻状态 $S_{t-1}$ 的节点（比如对输入 $x_t$ 做的线性投影、激活函数或归一化），全部被抽取成一张独立的 `coefficients` 子图，直接交给 `torch.func.vmap` 在整个窗口长度上一次性批量算完（并用公共子表达式消除 `fx_graph_cse` 去重）。
2. **识别仿射作用代数（Affine Action Algebra）**：剩下的、真正依赖 $S_{t-1}$ 的核心更新子图，如果能被化简成仿射递推形式：
   $$S_t = \mathcal{L}_t(S_{t-1}) + B_t$$
   分析器就会根据线性算子 $\mathcal{L}_t$ 的代数种类，直接把递推分派到预先写好的固定容量前缀扫描（Prefix Scan）内核上；如果用户在递推里写了非线性的状态自乘或无法化简的控制流，分析器就抛出 `Unsupported`，退回老老实实的按步串行执行。

这个分析器一共只认四种线性算子结构，在 `scan_analysis.py` 里定义得清清楚楚：

```python
# check-docs: skip: excerpt from pllm/execution/scan_analysis.py
Kind = Literal["scalar", "diagonal", "left", "right"]
# ……
@dataclass(frozen=True)
class Linear:
    """One scalar, pointwise or one-sided matrix action, optionally on transposed state."""

    kind: Kind = "scalar"
    coefficient: Coefficient = 1.0
    transposed: bool = False


@dataclass(frozen=True)
class Affine:
    """A symbolic linear map and its independent offset; None denotes an exact zero."""

    linear: Linear | None
    bias: Node | None
```

你看，`scan_analysis.py` 根本不知道世界上有什么叫 In-Place TTT、什么叫 GatedDelta 或线性注意力。它只问一个纯粹的代数问题：你对状态 $S$ 的变换，到底是乘一个标量（`scalar`）、按元素乘一个对角掩码（`diagonal`）、左乘一个矩阵（`left`），还是右乘一个矩阵（`right`）？只要落在这四种仿射作用构成的闭合代数里，结合律（Associativity）就天然成立，原本必须串行走 $L$ 步的递推，就会自动变成 $\mathcal{O}(\log L)$ 步的树状前缀归约（比如 `pllm/kernels/additive_scan.py` 和 `right_matrix_scan.py`）。

这也帮我们彻底理清了 pLLM 与 PyTorch 原生编译栈的职责边界：
* **pLLM 只管“任务之间”与“跨步代数”**：任务放在哪个锚点、走哪条 stream、读哪一步的 PTTS 版本、前向里哪些中间激活要留给反向训练、以及把跨时间步的仿射递推改写成批量前缀扫描；
* **`torch.compile` 管“单个任务内部”**：算子融合、寄存器分配、以及 AOTAutograd 在单步内部的激活保存与重算。

不再自己造编译器中端去和 `torch.compile` 抢活干，代码量大幅瘦身。而且，在删掉 ISL 之前的复核中我们还抓到了一个发人深省的细节：早期基于测量自动挑出的某种前缀扫描变形，在 In-Place TTT $64 \times 64$ 状态上跑出来的速度，居然只有参考实现的 $0.707\times$（慢了近 30%）！这提醒我们，“自动优化”本身也必须受到制衡。在重构后的新选择器里，我们立下一条规矩：永远把未改写的参考实现留在候选池里同台测速，并且必须在计划报告里如实交代最终选了哪一种实现。

---

## 五、 为什么在满载的 GPU 上，盲目重叠是零和博弈？

现在，规划器已经能用 40 行最长路算出放置窗口，也能把写状态的训练任务自动切到旁路流 `Lane::Side` 上了。

这时候，很多刚接触 GPU 编程的同学会掉进一个直觉陷阱：**人们总喜欢把 CUDA stream 想象成 CPU 上的线程，觉得只要开一条 `side` stream、把在线训练扔过去跟下一步的模型前向“并发执行”，训练的时间就白嫖到了。**

这在物理上是错的。

开启一条新的 CUDA stream，并不会让你的机箱里凭空多长出一张显卡。主流和旁路流上的所有 kernel，最终都要挤在同一张 GPU 的同一组流多处理器（SM）上，争抢同一条显存总线（HBM Bandwidth）。在 decode 阶段，大模型的主干前向和采样器本来就是高度显存带宽受限（Memory-Bandwidth Bound）的——当主流已经把显存带宽吃满时，旁路流上的训练任务横插一脚冲进来，它并不是在“利用空闲时间”，而是在**和主流抢同一块蛋糕**。

为了看清这场零和博弈（甚至常常是负和博弈），不妨看看我们在演进过程中亲历的三组实测。

**第一组实测：早点启动旁路训练，吞吐反而掉了。**
在 0.5B 模型上打磨 ESamp（$\beta=0$）的旁路重叠时，我们曾用 profiler 看到：每次前向还是有大约 $70.33\ \mu\text{s}$ 的旁路训练时间露在主流外面。当时我想：“为什么不在采样器跑完之前，就把训练所需的输入提前拷走、撤掉采样前的等待栅栏，让训练早点在旁路流上跑起来呢？”
我们动手改了。改完再看 profiler trace，数字漂亮极了——露在主流外面的旁路时间从 $70.33\ \mu\text{s}$ 暴降到了 **$8.57\ \mu\text{s}$**，几乎实现了 100% 的时间重叠！
但当我们切回端到端基准测试看总吞吐时，傻眼了：相对原生 vLLM 的吞吐不仅没涨，反而从 **94.49% 跌到了 92.73%**（早期另一组去掉采样栅栏的试验也同样从 94.3% 跌到了 92.9%）。

为什么“重叠得更完美”，总耗时反而更长？画一张时间线资源图就全明白了：

```text
Case A: Unfenced Early Side Launch (Collides with Bandwidth-Bound Sampler!)
Main Stream:  [   Model Forward   ] ──► [    Sampler (Sweeping 152k Vocab!)    ]
Side Stream:                        ──► [ Train Copy + Backward + AdamW ]
Hardware Bus:                           █████████████████████████████████  <-- Severe HBM Contention!
                                        (Both streams stall waiting on memory bus -> Slower step!)

Case B: Fenced Launch + TP Priority Staggering (Complementary Windows)
Main Stream:  [ Layer k Compute ] ──► [ Custom All-Reduce (NVLink/PCIe) ] ──► [ Layer k+1 ]
Side Stream:                          [     Replicated Local Train      ]
Hardware Bus:                         (Main waits on interconnect; Side uses local SMs/HBM!)
```

看上面的 Case A：采样器在干什么？它在扫过 `[128, 152064]` 的巨型词表张量做温度缩放、top-k/top-p 过滤或 Gumbel 抽样（[《采样尾声融合》](07-sampler-epilogue.md)），那是一步里显存带宽压榨得最狠的时刻。你把训练提前放出来，训练任务没有消失，它只是一头撞进了正在全力扫词表的采样器怀里，把原本流畅的词表读取撞得支离破碎。

**第二组实测：模型变大后，`publish_lag=1` 的收益会被分母摊薄。**
理解了“重叠不消灭工作量，只填补缝隙”，你就能秒懂为什么 `publish_lag`（用一两步的权重陈旧度换取跨步重叠，和分布式训练里 PipeDream 的权重暂存、异步 SGD 的陈旧梯度同源）在不同尺寸的模型上表现截然不同。
在今天的标准设置下（RTX 4090，vLLM 0.32，128 并发），跑 ESamp（$\beta=0$，rank 4）：
* 在 **0.5B 模型**上，一步只有 3.0 ms。开启 `publish_lag=1`（允许旁路重叠）时吞吐是原生的 **91.9%**，而设为 `publish_lag=0`（强制在主流上串行等待训练完成）时掉到 **89.8%**——2.1 个百分点的差距清晰可见。
* 但在 **7B 模型**上，一步前向长达 **21.7 ms**，而旁路流上的在线训练每步只需要约 **$616\ \mu\text{s}$**（其中分页 AdamW 约 $191\ \mu\text{s}$），只占整步时间的 2.8%。此时 `publish_lag=1` 的吞吐是原生的 **97.4%**，`publish_lag=0` 也有 **97.2%**（开启引导采样 $\beta=0.1$ 时则分别是 **96.4%** 和 **96.1%**）。当分母变成 21.7 ms 时，那几百微秒的训练无论串行还是并行，都已经退化成了测量噪声边缘的微小涟漪。

**第三组实测：什么时候重叠才能真正拿到大收益？当两边的硬件瓶颈互补时！**
既然同抢显存带宽是零和博弈，那什么情况下旁路流才是真白嫖？答案是：**当一条流在等某样东西，而另一条流恰好不用那样东西的时候。**

我们在本系列里见过两个绝佳的正面教材。第一个是在 [《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md) 里：旁路流只占满 1/4 的 SM（32 个线程块），慢悠悠地等着 23.7 GB/s 的 PCIe 总线从主机内存搬数据；剩下的 3/4 SM 和绝大部分 HBM 带宽全留给主流跑矩阵乘，两边各取所需——在一次 0.5B 剖析里，99% 的读表时间都藏在了主干 kernel 背后。

第二个正面教材发生在多卡张量并行（TP > 1）上。在单卡（TP=1）上，我们曾拒绝过一个“在采样前提前把训练更新写进私有状态、到 `token.commit` 再发布”的改动，因为单卡上它毫无收益。但到了 4 张 A100-SXM4-80GB 跑 7B 模型的张量并行环境里，每一层 Transformer 算完注意力或 MLP，主干流都必须停下来做一次跨卡的全归约集合通信（vLLM 的 Custom All-Reduce）。如果某张卡上的旁路训练恰好在此时抢占了 SM 或显存带宽，四张卡就会在同步障（Barrier）前上演木桶效应——全卡一起干等最慢的那张卡！

怎么破局？我们在 `pllm/worker.py` 里立下了只有寥寥几行的硬件调度纪律：

```python
# check-docs: skip: excerpt from pllm/worker.py
class GPUWorker(compat.Worker):
    def _make_model_runner(self) -> WorkloadModelRunner | Mrv1Runner:
        if self.parallel_config.tensor_parallel_size > 1:
            # Prioritize TP collectives over the replicated training lane, whose stream has priority 0.
            stream = torch.cuda.Stream(device=self.device, priority=-1)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            torch.cuda.set_stream(stream)
```

我们把 TP 推理主流的 CUDA stream 优先级设为高优（`priority=-1`），旁路训练流保持普通优先级（`0`），再配合“采样前写私有状态、提交时发布”的时间错峰。看看在 A100-SXM4-80GB（Qwen2.5-7B，ESamp $\beta=0$，rank 4，128 并发）上发生了什么：旁路训练与 vLLM Custom All-Reduce 的冲突重叠时间，从每步 **193–205 µs 直接降到了 0 µs**！端到端吞吐随之全线起飞：
* **TP = 1**：从 93.79% 提升到 **95.84%**；
* **TP = 2**：从 89.09% 提升到 **95.63%**；
* **TP = 4**：从 **87.07% 跃升到 97.60%**！

这就是硬件系统的真实脾气。不存在什么抽象世界里稳赚不赔的“并发箭头”；只有当你摸透了 SM 槽位、HBM 带宽与跨卡通信障的物理脉搏，把旁路任务精确塞进主流让出的硬件空隙里，重叠才会兑现成真金白银的吞吐。

---

## 六、 苦涩的教训与主动约束复杂度

回顾这段“请进 45 万行多面体编译器、又用 40 行最长路替掉它”的经历，真正让我反复咀嚼的，其实不是 Bellman-Ford 算法本身，而是**我们为什么那么难删掉一台自己亲手造出来的机器**。

强化学习先驱 Richard Sutton 在他那篇著名的短文《苦涩的教训（The Bitter Lesson）》里写过一句直戳人心的话：“花在某一条路线上的时间，就是没有花在另一条路线上的时间；人们在心理上会对自己已经投入心血的路线产生执念。”当你已经把 ISL 集成进构建链、调通了跨语言的内存管理、写完了几千行 Region IR 和配套测试时，承认“这套漂亮的多面体引擎在生产路径上其实没起作用”，需要克服极大的沉没成本心理。

那份冷静的代码盘点救了我们。它逼着我们把“这台机器在学术上很漂亮”和“这台机器在真实 workload 上到底干了什么”彻底剥离开来。

今天，大模型推理的世界正在发生一场肉眼可见的剧变。早年那种“除了分词器之外，从头到尾只有一个纯洁、匀质、没有任何人类先验的 Transformer 主干”的田园时代正在终结。因为硬件算力与显存带宽的摩尔定律双双放缓，而全世界对降低单 token 推理成本的渴求却依然按指数级狂奔，算法研究者们正在把越来越多“肮脏”、繁杂、充满结构先验与动态状态的机制塞进推理过程里——从跨 PCIe 的 n-gram 巨型记忆查表（Engram），到测试时在线微调（In-Place TTT），再到超网络动态生成每请求 LoRA、稀疏注意力索引与自适应跳层。

推理时的算法只会越来越杂、越来越脏、越来越有状态。

**恰恰是因为上层的算法注定要走向繁杂，承载它们的底层扩展底座，才必须拼命保持极简、内聚与正交。**

如果在面对每一个新算法时，我们都往引擎内核里塞一套专属的回调模板，或者反过来走向另一个极端——为了七个锚点的一维调度问题去背负四十多万行通用多面体编译器的包袱，那么用不了半年，这个底座就会在自身的重量下坍塌。

删掉那四十多万行代码之后，pLLM 的核心边界变得前所未有地清晰：
* 向上的用户接口，只索取算法不可省却的本质信息——状态的作用域与形状、挂载的时机阶段、数据的输入输出与纯正的 PyTorch 数学计算；
* 中间的规划器，用 40 行最长路看住一维时间线上的窗口与显存账本，用符号 FX 分析识别那四种仿射递推代数；
* 向下的执行层，把算子融合交给 `torch.compile`，把主干执行和显存管理交还给 vLLM 原生路径。

在软件工程里，写出几万行精巧的代码固然需要本事；但在看清问题的最小几何形状之后，敢于挥刀砍掉自己写的四十多万行代码、让系统退回那 40 行一眼就能看穿的最长路——这或许才是我们在构建 pLLM 的旅途中，学到的最珍贵的一课。
