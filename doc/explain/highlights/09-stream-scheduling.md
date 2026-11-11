# You Probably Don't Need a Compiler IR：用 40 行最长路自动调度 CUDA Stream

> 2026-11-11 · pLLM 提交 `1f9e03d6` · 测量环境：除文中另注外为单张 RTX 4090（24 GiB）、vLLM 0.32.0、Qwen2.5-7B-Instruct，8 条 prompt × 16 个样本、生成 256 token。
> 这是一篇有日期的文章，不随代码更新；当前的用法以[用户指南](../../README.md#用户指南)为准，当前的实测数字以[性能与显存](../performance.md)为准。

当你只写一个十几行的单请求 PyTorch 脚本时，“哪段代码在什么时候跑、跑在哪条 CUDA Stream 上”根本算不上一个问题。Python 解释器从上往下走，你在第 5 行写查表，它就在第 5 行查表；你在第 12 行写 `loss.backward()`，它就在第 12 行求导。

但只要你试着把这些推理期算法放进像 vLLM 这样的连续批处理服务引擎，让它们跟着每一个解码步（decode step）一起运转，三个系统调度问题就会立刻找上门来：

1. **在一步内部的什么时候跑（When）？**
2. **跑在主计算流（`main` stream）还是旁路流（`side` stream）上（Where）？**
3. **它写出的新状态，在哪一步对其他任务可见（Visibility）？**

拿一个结合了在线蒸馏（ESamp）与外部记忆查表（Engram）的场景来说。在每一个解码步里，模型跑完前向之后，在线训练任务需要拿第 0 层和最后一层的隐藏状态（hidden states）去算损失、做反向传播、并用 AdamW 更新每个请求专属的小网络权重；而在采样器掷骰子之前，引导任务又要拿这个小网络的预测去修正词表 logits（见[《Free FLOPs and Where to Find Them》](07-free-flops.md)）。与此同时，查表任务需要根据刚输入的 token ID 去几百兆的记忆表里捞出嵌入向量，供模型中间层消费（见[《为什么 32 个线程块比 9,600 个更快？》](08-bounded-gather.md)）。

如果把所有任务老老实实串在主 CUDA Stream 上，逻辑最简单，但主干前向每一步都要干等反向传播和优化器跑完。在单步耗时只有 3.0 ms 的 0.5B 模型上，多等几百微秒就意味着吞吐肉眼可见地往下掉。那如果把训练放到一条旁路 CUDA Stream 上异步跑呢？下一步的模型前向马上就要开始了，此时旁路上的优化器可能还没跑完，主干前向到底该读哪个版本的权重（见[《给 GPU 显存加上 MVCC》](03-gpu-mvcc.md)）？同样地，查表任务只依赖一步最开头的输入 token ID，完全不需要等隐藏状态算出来——它能不能在模型前向开始之前就发到旁路流上预取，赶在消费它的模型层之前抵达？

解决这三个问题，最容易想到的办法是手写一套启发式特判规则：如果看到是在线训练，就塞进采样后的钩子里上旁路流；如果看到是只读查表，就塞进前向前的钩子里。但每当有人写出一种新组合——比如用超网络在步内生成每请求的 LoRA 权重，或者让一个滞后一步更新的状态去喂给下一轮的线性覆盖层——你就得去引擎核心分支里再打一块特判补丁。

当你打到第三块补丁时，你就会想要一套有原则的自动调度机制：让写算法的人只用贴近原生 PyTorch 的方式声明“我读什么输入、写什么状态”，剩下的执行时机、流分配与跨步可见性，全部由规划器根据数据依赖自动推导出来。

这篇文章讲的就是如何实现这套自动推导，以及为什么为了解这道题，你大概率**不需要**自己造一套复杂的编译器中端（Compiler IR），更不需要请进重型的多面体整数规划器——认清问题的几何形状之后，40 行基础图算法就能给出精确解；而在硬件执行层面上，我们还会用几组实测数据看看，为什么“把任务扔上旁路 CUDA Stream”常常是一场零和博弈。

---

## 一、 三个问题，在数学上是同一道题

在动手挑选算法之前，不妨先把上面那三个问题并排画在白板上，看看它们在底层到底在算什么：

* “什么时候跑”，问的是任务 $u$ 在单步时间轴上的合法坐标区间；
* “跑在哪条流”，问的是任务 $u$ 的生产时刻与消费方任务 $v$ 的使用时刻之间隔了多远、中间有没有主干计算的空隙可以重叠；
* “什么时候可见”，问的是写状态任务 $u$ 与读状态任务 $v$ 在跨步时间轴上相距几个生成步（step distance）。

这三个看似独立的问题，在数学结构上是完全同构的：**它们都在求解一个任务在一条带有跨步循环的时间线上所处的几何位置。**

那么，这条时间线到底长什么样？

如果翻开高级编译器教材，看到“循环迭代”和“跨步数据依赖”这两个词，教科书会立刻把你引向编译器领域的重型武器——**多面体模型（Polyhedral Model）**。

多面体模型的思想很优雅：它把嵌套循环的每一次迭代看作高维整数空间 $\mathbb{Z}^n$ 里的一个格点，把读写依赖看作格点之间的仿射关系（Affine Relations），然后动用整数线性规划（ILP）求解器，在多面体空间里自动搜索合法的循环分块（tiling）、歪斜（skewing）、融合与多流并行切面。这也是 Pluto、PPCG、LLVM Polly 等经典编译器的核心引擎。

面对推理引擎里一步接一步的生成循环、以及模型内部一层接一层的遍历，把整个计算过程抽象成一套自定义的区域中间表示（Region IR），再交给多面体编译器的标准基石库 **ISL（Integer Set Library）** 去求最优调度，听起来简直顺理成章。

事实上，我们不仅认真考虑过这条路，而且真的把整套多面体引擎造了出来：将包含 1,530 个文件、457,770 行代码（主要是 C 语言）的 ISL 连同其高精度整数库完整引进了仓库，并在上面用 Rust 搭起了一整套 Region IR 中端，配齐了符号依赖流分析、分块与流搜索、生命周期保留分析与容量规划。

那是一台结构工整、理论完备的大机器。但当我们冷静下来，对生产路径上的每一分真实加速做了一次彻底的代码盘点时，却看到了三个让人清醒的事实：

第一，在像 In-Place TTT 和 GatedDelta 这类包含状态递推的负载上，将逐位置串行递推改写为并行前缀扫描（Parallel Prefix Scan）确实带来了可观的加速（在单张 RTX 4090 的微基准测试中，单头 $64 \times 64$ 状态转移耗时从 1.44 ms 降到 0.22 ms，提速 $6.57\times$；14 头提速 $1.85\times$；完整转移从 5.88 ms 降到 1.76 ms，提速 $3.34\times$）。但掀开当时负责识别这种递推的 `affine_scan.rs` 一看，里面是一个纯 Rust 写的表达式树模式匹配器，**调用 ISL 的代码是零行**——它只是在最外层套了一个 Region IR 的壳子。

第二，在跨步可见性分析（即判断在线训练写出的新权重能否与下一步前向重叠）中，代码确实调用了 ISL。但它是怎么调的呢？它把整数滞后量 `k`（即 `publish_lag`）格式化成一段 ISL 集合字符串，传进 C 库里去求解不等式 `0 <= b < t` 是否成立。换句话说，它动用了 45 万行的整数线性规划求解器，去算了一道小学算术题：**整数 `lag` 是不是大于等于 1。**

第三，那些真正重度调用 ISL 的多面体分块、多流划分与保留分析模块，在真实的推理负载上要么只吐出平庸的恒等调度（Identity Schedule），要么干脆不可达；而当把多序列请求真正展开进 ISL 整数集程序去求解网格调度时，高维格点的组合爆炸立刻显现了出来——在 RTX 4090 上对 Qwen2.5-0.5B（eager 模式，预填充长度 1024，块大小 128，作用在 5 个目标层）做实测，1 条序列耗时 **146 ms**，2 条序列耗时 **293 ms**，到 4 条序列时直接抛出 `ISL quota exceeded`（求解步数超配额），8 条序列当场触发 `LibISLError`（作为量级参照，在同一预填充规模下，不经过多面体展开、直接在模型前向内部做精确矩阵更新的路径仅需 **19.6 ms**）。

为什么在传统编译器里无往不利的多面体模型，放进推理引擎会水土不服？

原因在于**我们看错了问题的几何形状**。多面体模型是为高维、密集的仿射循环嵌套（比如三层矩阵乘法或图像卷积 stencil）设计的。但在现代大模型服务引擎里，外层是连续批处理动态洗牌的请求，内层是被 CUDA Graph 焊死的巨型算子；留给扩展层在步内调度的，根本不是什么五维空间里的多面体格点，而是**一条一维时间轴上的几个固定阶段锚点，外加常数步长的跨步状态引用**。

拿一把用来切高维晶体的激光手术刀去切一根一维的香肠，不仅切得不会更快，反而让每一次故障排查都要穿过四十多万行外部 C 代码的黑盒。看清这一点后，我们把 45.7 万行 ISL 代码连同整套 Region IR 全部从仓库里删掉了，换成了真正贴合这个几何形状的最小算法。

---

## 二、 8 个锚点、一致依赖与 40 行最长路

让我们把复杂的编译器术语放在一边，在纸上画出推理引擎一步解码的真实时间线。

在我们的扩展框架 pLLM（见[《给 vLLM 装一个“eBPF”》](00-guide.md)）中，引擎的每一个生成步（Step $t$）都被划分为 $M=8$ 个严格全序的时间轴**锚点（Anchors）**。每个执行阶段覆盖其中一段锚点区间（一段可以跨好几个锚点）：

```text
Step t Timeline (M = 8 Ordered Anchors):

  0:request    1:model     2:state     3:sampler    4:sampler     5:token     6:request   7:next
   .prepare    .execute    .publish    .pre_draw    .complete     .commit      .finish    .prepare
──────┼──────────┼───────────┼───────────┼────────────┼─────────────┼───────────┼───────────┼──────►
      │          │           │           │            │             │           │           │
   [Tokens]  [Forward]   [Hiddens]   [Guidance/   [Sampled]     [Sampled    [Retire     [Next Step
    Ready     Running    Published    Bias Due]     Done         Tokens]     Finished]   Boundary]
```

在这 8 个刻度（编号 $0, 1, \dots, 7$）上，每一类数据都有它确切的“出生时刻”：
* 请求槽位与历史输入 token（`pllm.input_tokens`）在一步最开头的 `0: request.prepare` 就已经就绪；
* 基座模型在 `1: model.execute` 运行前向传播，各层的隐藏状态（`pllm.hidden`）要等到前向跑完后的 `2: state.publish` 才正式发布；
* 采样前的 logits 偏置与引导预测必须在 `3: sampler.pre_draw` 之前交卷；
* 采样器在 `4: sampler.complete` 完成抽样，而在 `5: token.commit` 提交本步最终确定的新 token（`pllm.sampled_tokens`）；
* 最后在 `6: request.finish` 与 `7: next.prepare` 之间回收已结束的请求，跨入下一步。

因此，任何一个扩展任务 $u$，根据它声明的作用阶段和它所读取的输入源，都会天然带有一个初始的合法运行区间 $[a_u^{\min}, a_u^{\max})$：最早不能早于它所有输入数据的出生锚点，最晚不能晚于它声明阶段的截止锚点。

现在来看看任务与任务之间的依赖边。如果把第 $t$ 步的第 $a$ 个锚点映射成一条全局一维时间轴上的整数刻度 $T = t \cdot M + a$，整个系统里所有的依赖关系只有极其规整的两种：

第一种是**同一步内的数据依赖（Step Distance $k = 0$）**：任务 $v$ 读取了任务 $u$ 在本步算出的结果（或者在同一版本链上先后读写状态），要求任务 $v$ 的锚点不早于任务 $u$ 加上最小间隔 $d$（即 $a_v - a_u \ge d$）。

第二种是**跨步的状态版本依赖（Step Distance $k \ge 1$）**：任务 $v$ 在第 $t$ 步读取了任务 $u$ 在第 $t-k$ 步写入的状态版本（即 `publish_lag = k`）。换算到全局整数刻度上，这意味着：

$$(t \cdot M + a_v) - ((t - k) \cdot M + a_u) \ge d$$

把不等式左右两边的 $t \cdot M$ 消掉，循环步数变量 $t$ 当场就消失了。不等式化简成了一个与步数完全无关的初等差分约束：

$$a_v - a_u \ge d - k \cdot M$$

```text
Unrolled Steps (Infinite Timeline):            Reduced Uniform Graph (Finite, Weight = d - k*M):

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

早在 1967 年，Karp、Miller 和 Winograd 在研究一致递推方程（Uniform Recurrence Equations）时就已经透彻地分析过这种结构：当循环依赖的跨步距离全是常数时，这种结构被称为**一致依赖（Uniform Dependence）**。对于一致依赖，无限展开的跨步时间线可以直接折叠成一张只有有限个节点的**约化依赖图（Reduced Dependence Graph）**——任务 $u$ 在第 $t$ 步指向任务 $v$ 在第 $t+k$ 步的依赖边，只需赋予权重 $d - k \cdot M$。

在这张只有几个到十几个节点的小图上，求解所有任务合法的 `[最早锚点, 最晚锚点]` 窗口，根本不需要整数线性规划求解器，它就是图论中最基础的**最长路（Longest Path）/ 差分约束问题**。

来看看 `rust/pllm_core/src/planning/scheduler.rs` 里的 `propagate` 函数。调度窗口的推导，完完整整地落在这 40 行 Rust 代码里：

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

第一行文档注释 `u(t) -> v(t+k) has weight d-k*M` 就是全部的理论基础；函数体是一个双向的 Bellman-Ford 松弛循环——前驱的下界 `lo[u] + weight` 往后推着后继的下界 `lo[v]`，后继的上界 `hi[v] - weight` 往前压着前驱的上界 `hi[u]`。

这段代码最讨人喜欢的地方，在于它失败时的表现。通用的整数规划求解器在无解时，往往只会报出一个晦涩的“集合为空（empty set）”；而这个 40 行的松弛循环只有两个错误出口，每一个都直接对应着一句用户能立刻照着改代码的工程诊断：

1. **某个任务的下界越过了上界（`lo > hi`）**：说明你在某个阶段声明了一个方法，却在参数里去读一个要等该阶段结束之后才诞生的值（比如在采样前运行的 `@pllm.add_logits` 里去读采样后才存在的 `pllm.sampled_tokens`）。报错直接提示你把它移到更晚的阶段。
2. **松弛了 $|V|$ 轮之后依然在变（`pass == bounds.len()`）**：根据 Bellman-Ford 定理，图里存在正权环——也就是说，你的跨步读写约束首尾相咬，形成了一个时间上的死锁环。

一旦 `propagate` 收紧了每个任务的合法窗口，接下来的 `place` 函数就把每个任务安放在它合法窗口的最早锚点 `windows[id].0` 上，随后执行流分配（`select_lanes` 与 `select_prefetch`）和显存生命周期核算（`plan_memory`）：

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

注意最后那个 `.or_else(|_| ...)` 兜底。把任务放到旁路流（`Lane::Side`）跨步异步执行时，它所依赖的输入张量、中间激活与状态版本都必须跨步保留更久，显存峰值自然更高。如果 `plan_memory` 沿着时间线模拟生命周期时发现峰值超出了显存预算，规划器会自动把所有任务退回 `Lane::Main` 串行执行再算一次——但**严格保留已经选定的逻辑读边界（`read_origin` 与 `read_distance`）**。也就是说，显存紧张只会让你退回串行吞吐，而算法读哪一步版本的数学语义连一个比特都不会变；如果退回主流依然超出预算，规划器就会在启动前直接报错并打印出精确的所需字节数。

这里需要坦诚交代“自动”与“规则”的边界：在这套调度器里，窗口推导、版本冲突检测与显存生命周期核算是完全自动求解的；但“哪个任务走旁路流（`select_lanes` / `select_prefetch`）”依然是几条写明的物理规则——
* 当一个任务只读取 `input_tokens` 和只读状态、且可被 CUDA Graph 捕获时，`select_prefetch` 会将它放到 `Lane::Side` 并在一步开头的 `request.prepare` 发射，与当前步的模型前向重叠；
* 当存在至少一步的状态可见性滞后（`publish_lag >= 1` 或模型内读取）、没有自定义层循环或显式可见性覆盖、第一个写状态的任务锚点在 `model.execute` 之后、且后续任务不喂给采样器也不依赖宿主服务时，`select_lanes` 会把从第一个写状态任务开始的后缀切到 `Lane::Side`。如果 `publish_lag=0`，所有更新必须在当前步结束前立即可见，任务就全部留在主流串行。

甚至连“锚点应该切多细”，也受真实测量约束。锚点越细看起来越通用：在关闭 CUDA Graph 的 eager 后端上为模型的每一层都注册细粒度的执行锚点，调度器就能把任务精确塞进“第 $l$ 层之后”。但每一层都要在 Python 里做钩子检查与分发；在一次较早的测量里（单张 RTX 4090，Qwen2.5-0.5B，并发批次 $8 \times 32$，eager 模式），几个状态生成器的相对原生吞吐因此全面倒退：超网络（hypernet）从 0.722 掉到 0.656，混合专家（mixture）从 0.723 掉到 0.667，AdamW 从 0.554 掉到 0.505。所以调度器只把任务放在后端显式提供的粗粒度锚点上。

---

## 三、 认识代数，不认识名字：任务内部的自动加速

搞定了任务之间的放置与分流，那么任务**内部**的计算呢？

在删掉 Region IR 时，我们给自己立下了一条硬要求：对于用户编写的沿序列方向逐位置更新的递推函数 $S_t = \text{step}(S_{t-1}, x_t)$，即便是一个框架从未见过的新算法，只要它在数学上可并行，依然要能被自动改写为并行前缀扫描（Prefix Scan），绝不能靠识别 workload 的类名来硬编码。

要做到这件事，同样不需要自己发明一套编译器 IR。PyTorch 自带的符号追踪工具 `make_fx` 已经能把单步函数变成一张干干净净的 FX 计算图。我们在 `pllm/execution/scan_analysis.py` 里所做的，只是对追踪出的状态变量（carry $S$）做一次轻量级的符号线性度分析：

1. **剥离状态无关子图**：计算图里凡是不依赖上一时刻状态 $S_{t-1}$ 的节点（比如对输入 $x_t$ 做的线性投影、激活或归一化），全部被抽离成一张独立的 `coefficients` 子图，直接用 `torch.func.vmap` 在整个序列窗口上一次性批量算完（并配合公共子表达式消除 `fx_graph_cse` 去重）。
2. **识别闭合的仿射作用代数**：剩下的、真正依赖 $S_{t-1}$ 的核心更新子图，如果能被化简成仿射递推形式：
   $$S_t = \mathcal{L}_t(S_{t-1}) + B_t$$
   分析器就会根据线性算子 $\mathcal{L}_t$ 的代数类型，把递推自动分派到现成的固定容量前缀扫描内核（如 `pllm/kernels/additive_scan.py` 和 `right_matrix_scan.py`）；如果用户在递推里写了非线性的状态自乘或无法化简的控制流，分析器就退回按步串行执行。

这套分析器一共只认四种线性算子形态，在 `scan_analysis.py` 里定义得一目了然：

```python
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

分析器根本不知道世界上有什么叫 In-Place TTT、什么叫 GatedDelta。它只检查状态 $S$ 上的变换是不是乘一个标量（`scalar`）、按元素乘一个对角掩码（`diagonal`）、左乘一个矩阵（`left`）或右乘一个矩阵（`right`）。只要落在这四种仿射作用构成的代数里，结合律就天然成立，原本必须串行走 $L$ 步的递推就能自动折成 $\mathcal{O}(\log L)$ 步的树状归约。（而且为了防止自动改写弄巧成拙——并行扫描并不总比串行快——选择器会始终把原始参考实现留在候选池里同台实测比对，并在计划报告中如实交代最终选中的方案。）

这也彻底划清了扩展层与 PyTorch 原生编译栈的分工：**pLLM 只管“任务之间”的锚点放置、流分配、版本边界与跨步代数改写；而“单个任务内部”的算子融合、寄存器分配与单步重算分割，全部交给 `torch.compile`。**

---

## 四、 为什么把任务扔上旁路 CUDA Stream 常常是零和博弈？

现在，规划器已经能用 40 行最长路算出放置窗口，也能把在线训练自动切到旁路流 `Lane::Side` 上了。

这时候，很多写 GPU 程序的工程师容易产生一种乐观的错觉：把 CUDA Stream 当成独立 CPU 核心上的线程，觉得只要新建一条 `side` stream、把在线训练扔过去跟下一步的模型前向“并发执行”，训练耗时就能白白省下来。

遗憾的是，硬件不是这么工作的。

新建一条 CUDA Stream，并不会让你的机箱里多长出一张显卡。主流和旁路流上的所有内核，最终都要挤在同一张 GPU 的同一组流多处理器（SM）上，争抢同一条显存总线（例如 RTX 4090 上的 GDDR6X 显存带宽，或 A100 上的 HBM2e 显存带宽）。在逐字生成的 decode 阶段，大模型的主干前向和采样器本来就是高度受限于显存带宽的——当主流已经把显存总线吃满时，旁路流上的训练任务挤进来，并不是在“利用空闲时间”，而是在**和主流抢同一块带宽蛋糕**。

下面三组实测数据，把这场硬件上的零和博弈（有时甚至是负和博弈）展现得清清楚楚。

> **测量环境说明**：除明确标出的多卡 A100 实验与早期剖析外，以下基准数据均在单张 NVIDIA RTX 4090（24 GiB GDDR6X）、vLLM 0.32.0（开启 CUDA Graph 与 FlashInfer 采样器）、并发负载 128 条序列（8 个 prompt × `n=16`）、每条生成 256 个 token 的标准设置下测得，取预热后 3 轮中位数。主模型为 `Qwen/Qwen2.5-7B-Instruct`（BF16，原生图模式单步耗时约 21.7 ms，吞吐 5,894 tok/s），小模型对照为同配置的 `Qwen/Qwen2.5-0.5B-Instruct`（原生图模式单步耗时约 3.0 ms，吞吐 42,355 tok/s）。

### 1. 提早启动旁路训练，为什么反而更慢？

在 RTX 4090 上用 Qwen2.5-0.5B 跑在线蒸馏 ESamp（$\beta=0$，128 并发，vLLM 0.27.1，一版较早的实现）时，性能剖析器显示：每一步还是有大约 $70.33\ \mu\text{s}$ 的旁路训练时间露在主流前向的外面。

一个非常符合直觉的想法冒了出来：为什么非要等采样器抽完词才启动训练呢？如果在采样器运行之前，就提前把训练所需的输入拷好、撤掉采样前的等待栅栏，让旁路训练早点跑起来，岂不是能重叠得更彻底？

这样做之后，剖析器的时间线漂亮极了——露在主流外面的旁路时间从 $70.33\ \mu\text{s}$ 骤降到了 **$8.57\ \mu\text{s}$**，时间轴上几乎实现了严丝合缝的 100% 重叠。

但切回端到端基准测试一看，相对原生 vLLM 的吞吐不仅没涨，反而从 **94.49% 掉到了 92.73%**（另一组同样提前拷贝训练输入并撤掉采样栅栏的测量，也从保留栅栏时的 94.3% 跌到了 92.9%）。

为什么“重叠得更充分”，总耗时反而变长了？把两条流在硬件总线上的动作画出来就明白了：

```text
1. Unfenced Early Side Launch (Collides with Bandwidth-Bound Sampler):
Main Stream:  [   Model Forward   ] ──► [    Sampler (Sweeping 152k Vocab!)    ]
Side Stream:                        ──► [ Train Copy + Backward + AdamW ]
Memory Bus:                             █████████████████████████████████  <-- Severe Bus Contention!
                                        (Both streams stall waiting on memory bus -> Slower step!)

2. Fenced Launch + TP Priority Staggering (Complementary Windows):
Main Stream:  [ Layer k Compute ] ──► [ Custom All-Reduce (Interconnect) ] ──► [ Layer k+1 ]
Side Stream:                          [     Replicated Local Train       ]
Hardware:                             (Main waits on cross-GPU link; Side uses local SMs/Memory!)
```

看看第一种情况：采样器在那个时刻正在干什么？它正在扫过 `[128, 152064]` 的巨型词表张量做温度缩放、top-k/top-p 过滤和 Gumbel 抽样（见[《Free FLOPs and Where to Find Them》](07-free-flops.md)），那是一步里显存带宽压榨得最猛烈的时段。撤掉栅栏把训练提前放出来，训练的工作量并没有凭空消失，它只是一头撞进了正在全力扫词表的采样器怀里，两边在显存总线上互相踩踏，把原本流畅的采样步拖慢了。

### 2. 模型尺寸如何稀释跨步重叠的收益？

理解了“旁路流不消灭工作量，只填补空隙”，你就能明白为什么用一步状态可见性滞后（`publish_lag=1`，与流水线并行里的 PipeDream 权重暂存和异步 SGD 陈旧度同源）换取的跨步重叠，在小模型和大模型上看起来截然不同。

看看同一张 RTX 4090 上，0.5B 与 7B 跑 ESamp（rank 4，128 并发）的实测对比：

| 模型与配置 | 原生 vLLM 吞吐 (单步耗时) | `publish_lag=0` (主流串行等待) | `publish_lag=1` (旁路跨步重叠) | 旁路训练每步 GPU 耗时 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B** ($\beta=0$) | 42,355 tok/s (`3.0 ms`) | 38,047 tok/s (**89.8%**) | 38,936 tok/s (**91.9%**) | ~178 µs (AdamW ~29 µs；vLLM 0.27.1 时期剖析) |
| **Qwen2.5-7B** ($\beta=0$) | 5,894 tok/s (`21.7 ms`) | 5,731 tok/s (**97.2%**) | 5,738 tok/s (**97.4%**) | ~616 µs (AdamW ~191 µs) |
| **Qwen2.5-7B** ($\beta=0.1$) | 5,894 tok/s (`21.7 ms`) | 5,663 tok/s (**96.1%**) | 5,681 tok/s (**96.4%**) | ~616 µs + 引导采样开销 |

在 **0.5B** 模型上，原生 decode 一步只有短短的 **3.0 ms**（较早在 vLLM 0.27.1 上的另一轮测量里，1,245 个旁路内核中有 780 个与主流重叠，$\beta=0$ 从 91.4% 升至 94.3%——更稳的 20 轮交叉测量是 92.45%——$\beta=0.1$ 从 73.7% 升至 84.1%）。在本系列统一的 vLLM 0.32 基准上，把 `publish_lag` 从 `0` 切到 `1`，吞吐从 **89.8%** 升到了 **91.9%**，2.1 个百分点的提升清晰可见。

但当你换到 **7B** 模型时，主干跑一步前向拉长到了 **21.7 ms**，而 rank-4 小网络的旁路训练一步总共只需要约 **616 µs**，只占整步时间的 **2.8%**。当分母变成 21.7 ms 时，这 616 µs 无论你是串行等它（97.2%），还是放到旁路流上重叠（97.4%），端到端吞吐的差别只有 0.2%，几乎落在了测量噪声的边缘。

### 3. 什么时候重叠才是真白嫖？当两边索取的资源互补时

既然同抢一条显存总线是零和博弈，那什么情况下多流并发才能兑现真金白银的加速？答案是：**当一条流在等某样硬件资源，而另一条流恰好不用那样资源的时候。**

第一个典型的正例，是我们在上一篇[《为什么 32 个线程块比 9,600 个更快？》](08-bounded-gather.md)里看到的跨 PCIe 主机查表：旁路流被硬限制在 32 个线程块（占 1/4 的 SM），慢悠悠地等着 23.7 GB/s 的 PCIe 总线从 CPU 内存搬数据；剩下的 3/4 SM 和显存带宽全留给主流跑矩阵乘。一边吃 PCIe 带宽，一边吃 SM 算力与本地显存，两边各取所需，99% 的读表耗时就这样藏进了主干前向的阴影里。

第二个正例发生在**多卡张量并行（Tensor Parallelism，TP > 1）**上。

在单卡（TP=1）上，“在采样前提前把训练更新写进私有状态槽位、等到 `token.commit` 再发布”的做法没有空隙可钻，测不出收益，所以单卡上不这么放。但到了 4 张 A100-SXM4-80GB 上跑 Qwen2.5-7B 的张量并行时，情况完全变了：每一层 Transformer 算完注意力或 MLP，主干流都必须停下来做一次跨卡的全归约集合通信（vLLM 的 Custom All-Reduce）。此时此刻，如果某一张卡上的旁路训练恰好抢占了 SM 或显存带宽，拖慢了这张卡进入集合通信的脚步，其余三张卡就必须在同步障（Barrier）前干等它一个——单卡上的微小抖动，通过跨卡同步障被放大成了全卡停摆。

怎么把旁路训练塞进跨卡通信的空隙里，同时绝不让它挡住集合通信的路？我们在 `pllm/worker.py` 里加上了短短几行流优先级纪律：

```python
class GPUWorker(compat.Worker):
    def _make_model_runner(self) -> WorkloadModelRunner:
        if self.parallel_config.tensor_parallel_size > 1:
            # Prioritize TP collectives over the replicated training lane, whose stream has priority 0.
            stream = torch.cuda.Stream(device=self.device, priority=-1)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            torch.cuda.set_stream(stream)
```

我们把 TP 推理主流的 CUDA Stream 优先级设为高优（`priority=-1`），而复制执行的旁路训练流保持默认优先级（`0`）。不过单独这一条只带来小幅提升（TP=4 下从 14,936 到 15,031 tok/s），主要收益来自“采样前写私有状态、提交时发布”的时间错峰放置。

在 A100-SXM4-80GB 上跑 Qwen2.5-7B（ESamp $\beta=0$，rank 4，128 并发序列，生成 256 token）实测：旁路训练与 vLLM Custom All-Reduce 的冲突重叠时间，直接从每步 **193–205 µs 降到了 0 µs**。连同同一批改动里去掉的冗余字段拷贝与梯度填充（单卡的提升来自这一部分），端到端吞吐随之全线回升：
* **TP = 1**：从 93.79% 提升至 **95.84%**；
* **TP = 2**：从 89.09% 提升至 **95.63%**；
* **TP = 4**：从 **87.07% 跃升至 97.60%**。

这就是硬件系统的真实脾气。在白板上画两条并行的箭头很容易，但只有当你摸清了 SM 槽位、显存总线与跨卡通信障的物理脉搏，让旁路任务避开主流的锋芒、钻进主流等待外部链路的空隙里，并发才会真正变成加速。

---

## 五、 主动约束系统的复杂度

回顾从“引进 45 万行多面体编译器与自定义 Region IR”到“退回 40 行 Bellman-Ford 最长路与 FX 代数识别”的这一路，最难的一步其实不是写出那 40 行代码，而是下定决心删掉那台已经写好、调通、并且写满了测试的大机器。

强化学习先驱 Richard Sutton 在《苦涩的教训（The Bitter Lesson）》里写过一句话：“花在某一条路线上的时间，就是没有花在另一条路线上的时间；人们在心理上会对自己已经投入心血的路线产生执念。”在过去几年里，大模型本身是《苦涩的教训》的最佳践行者——除了分词器之外，中间是一个纯洁、匀质的 Transformer 主干，靠堆算力和数据解决一切。但随着单晶体管成本下降放缓、显存带宽受限，而全行业对降低单 token 推理成本的要求又在倒逼，推理期的算法正在变得越来越繁杂、越来越有状态：有人外挂跨 PCIe 的 n-gram 记忆大表（Engram），有人在上下文里实时微调快权重（In-Place TTT），有人动态跳层，有人在线蒸馏引导采样。

恰恰是因为上层的推理算法注定要走向繁杂与多样，承载它们的底层系统底座，才更要拼命克制自己的复杂度。

当你要为一个流式系统设计自动调度器时，先别急着去搬教科书里最通用的重型编译器框架。先拿一张纸，把你的时间线和依赖边画出来看一看：如果它只是一条一维时间轴上的 8 个有序锚点，而所有的跨步状态引用都只是常数步长的平移，那么请相信 1967 年的图论——40 行最长路松弛，就已经是你所需要的全部编译器了。
