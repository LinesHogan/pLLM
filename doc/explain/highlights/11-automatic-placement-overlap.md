# 自动放置与重叠

在单步研究脚本里写推理期扩展时，“一段代码什么时候跑”从来不是个问题。你把它写在第 0 层后面，Python 解释器就在第 0 层跑完时调用它；你把它写在 `loss.backward()` 后面，优化器就紧跟着更新参数。控制流就是你代码的行号。

一旦进入连续批处理的服务引擎，这个简单的世界就解体了。在每一个解码步（decode step）里，模型主干、采样器、外部记忆查表和在线学习任务同时在场。哪怕只是挂载一个像在线蒸馏（ESamp）这样只有十几行数学逻辑的算法——每走完一步前向，拿第 0 层与最后一层的隐藏状态去训练一个轻量级预测器（计算损失、反向传播、执行分页 `AdamW` 更新），同时在采样器掷骰子前用这个预测器修正词表分布（见[《采样尾声融合》](07-sampler-epilogue.md)）——系统也必须在启动前回答三个问题：

第一，每个扩展任务在单步内部的**什么时刻运行**？查表任务（见[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)）只依赖输入 token ID，能不能在模型前向开始前就提前发射？而读取深层隐藏状态的训练任务，最早要等到什么时候才能拿到数据？

第二，它运行在**哪条 CUDA stream** 上？如果把反向传播和优化器更新串行放在主计算流（`main` stream）上，主干网络每一步都要干等几百微秒；如果把它挪到旁路流（`side` stream）上异步执行，它和主流之间在哪里同步？

第三，它写出的新状态，**什么时候对其他计算可见**？当第 $n$ 步的训练还在旁路流上运行时，第 $n+1$ 步的模型前向可能已经开始了，此时前向预测到底该读哪一个版本的权重（见[《PTTS：给每个请求一块会“长版本”的显存》](01-ptts.md)）？

应付这三个问题，最顺手的权宜之计是在执行器里写启发式特判：给采样后的训练写一段重叠分支，给预取写一个标志位，在签名检查里列一张允许异步的组合白名单。但特判规则一旦超过两三条，就会迅速纠缠成一团乱麻。每当有人写出一种没见过的新组合——比如用超网络在步间生成每个请求的低秩权重，或者把滞后一步的状态喂给模型内部的覆盖层——你就得再去修改调度分支。

我们需要的不是越来越多的特判，而是一个有原则的自动推导机制：算法作者只声明自己读什么输入、写什么状态、作用在哪个阶段，剩下的执行时机、流划分与版本边界，由规划器根据数据依赖自动算出来。

但要想“自动算出来”，首先得看清这个问题在数学上到底长什么样。

---

## 一、 问题的几何形状：为什么多面体模型用错了地方

当你把上面那三个问题——何时运行、走哪条流、哪一步可见——并排画在纸上时，会发现它们并不是三个独立的问题：

* “何时运行”，问的是任务 $u$ 在单步时间轴上的合法坐标区间；
* “走哪条流”，问的是任务 $u$ 与消费它结果的任务 $v$ 在时间轴上隔了多远，中间有没有算力空隙可以并行；
* “何时可见”，问的是写者任务 $u$ 与读者任务 $v$ 在跨步时间轴上相隔几个生成步（step distance）。

它们在问同一件事：**每个任务在一条跨步循环的时间线上，处于什么几何位置。**

一旦提到“循环迭代空间里的依赖分析与自动调度”，熟悉编译原理的人脑子里往往会跳出一个现成的重武器：**多面体模型（Polyhedral Model）**。

多面体模型是编译器领域处理嵌套循环的经典工具（如 Pluto、PPCG、LLVM Polly 以及底层的整数集库 ISL）。它的思想很漂亮：把 $n$ 层嵌套循环的每一次迭代看作高维整数空间 $\mathbb{Z}^n$ 里的一个格点，把数组读写依赖表示为格点之间的仿射关系（Affine Relations），然后动用整数线性规划（ILP）求解器，在高维空间里搜索合法的循环分块（tiling）、歪斜（skewing）与并行超平面。

既然推理是一步接一步的循环，层与层之间也是循环，把整个推理扩展过程建模成多面体区域（Region IR），交给整数线性规划去自动求最优调度，听起来似乎顺理成章。

但如果你真的沿着这条路走下去，就会发现这门大炮完全打错了地方。

多面体模型之所以需要动用复杂的整数集操作与线性规划，是因为它面对的是**高维、密集、带复杂仿射下标的循环嵌套**（例如矩阵乘法或三维图像卷积 stencil，内层循环的边界依赖于外层循环变量）。而在一个大模型推理服务引擎里，外层是连续批处理动态洗牌的并发请求，内层是被 CUDA Graph 和原生算子焊死的大块计算。留给扩展层去调度的，根本不是什么高维空间里的多面体格点，而是**一条一维时间轴上的几个到几十个粗粒度任务，外加常数步长的跨步状态引用**。

如果强行把高维多面体求解器塞进这个场景，会发生两件尴尬的事。

第一件尴尬的事是**杀鸡用牛刀，甚至根本切不动肉**。对于跨步状态依赖（比如第 $t$ 步读取第 $t-k$ 步写入的状态），多面体分析无非是把整数滞后量 $k$ 格式化成一串集合约束，放进求解器里去判断不等式 $0 \le b < t$ 是否成立——这本质上只是在检验一维整数 $k \ge 1$。而那些真正重度依赖多面体变换的区域分块与多流划分，在真实的推理负载上面对的只是寥寥几个任务节点，每次大费周章搜索完，吐出来的不过是一个平庸的恒等调度（Identity Schedule）。更糟的是，如果把请求的序列块展开成整数集程序交给多面体求解器（如 ISL）去直接规划，求解器的组合爆炸会立刻暴露出来：在单张 RTX 4090 上对 Qwen2.5-0.5B（eager 模式，预填充长度 1024，块大小 128，5 个目标层）进行测试，1 条序列需要 146 ms，2 条序列涨到 293 ms，到 4 条序列时就会直接耗尽求解配额报错退出；而同样预填充规模下，直接在模型前向内部做精确矩阵更新只需要 19.6 ms。

第二件尴尬的事发生在**执行粒度**上。多面体与区域中端很容易诱导人写出“逐 token 或逐层切片解释执行”的调度器。但正如我们在[《自定义层循环与状态传播》](10-custom-layer-loops.md)里看到的，GPU 的吞吐命脉在于把整批激活聚在一起喂给原生大算子；任何试图把控制流拆碎到逐 token 级别的做法，都会让解码步耗时暴涨数倍。同样地，如果你为了让自动调度器拥有最大的自由度，在非图模式（eager）下为模型的每一层都注册细粒度的任务挂载钩子，允许调度器把任务塞进任意两层之间，那么单单是在宿主端逐层检查钩子与分发旁路任务，就会凭空吃掉可观的 CPU 时间——在 RTX 4090、Qwen2.5-0.5B（并发 $8 \times 32$，eager 模式）的实测中，引入逐层分发钩子会让超网络（hypernet）、混合专家（mixture）和 AdamW 三类状态生成器的相对原生吞吐分别从 72.2%、72.3%、55.4% 倒退到 65.6%、66.7% 和 50.5%。

硬件的测量结果非常诚实：调度器不应该自己创造细碎的执行点，而只能把任务放在后端显式声明的粗粒度阶段边界上。

既然执行点是有限的粗粒度锚点，跨步依赖又是固定步长的平移，那么一旦剥掉多面体编译器的繁复外壳，这个问题真正的数学核心到底是什么？

---

## 二、 一维时间线、一致依赖与 40 行最长路

让我们在纸上画出一步解码的真实结构。

在 pLLM 中，引擎的每一个生成步（Step $t$）由后端提供 $M=8$ 个严格全序的时间轴**锚点（Anchors）**，每个声明阶段（`pllm.Phase`）对应其中一段锚点区间：

```text
Step t Timeline (M = 8 Ordered Anchors):

  0:request     1:model       2:state       3:sampler      4:sampler     5:token      6:request    7:next
   .prepare     .execute      .publish      .pre_draw      .complete     .commit      .finish      .prepare
──────┼───────────┼─────────────┼──────────────┼───────────────┼─────────────┼────────────┼───────────┼──────►
      │           │             │              │               │             │            │           │
   [Tokens]   [Forward]     [Hiddens]      [Guidance/      [Sampled]     [Sampled     [Retire]    [Next Step]
    Ready      Running       Published      Bias Due]        Done         Tokens]
```

在这条时间线上，每一种输入数据都有它确定的诞生时刻：输入 token（`pllm.input_tokens`）在 `0: request.prepare` 就已经就绪；模型各层的隐藏状态（`pllm.hidden`）要等前向完成并在 `2: state.publish` 发布后才能读取；采样器反馈（`SamplerFeedback`）要等到 `4: sampler.complete`。结合方法自己声明的阶段窗口，每个任务 $u$ 在单步内都有一个初始的合法锚点区间 $[a_u^{\min}, a_u^{\max})$。

现在来看任务之间的依赖边。如果我们把第 $t$ 步的第 $a$ 个锚点映射为全局一维时间轴上的整数坐标 $T = t \cdot M + a$，整个系统里的依赖关系就只有两种：

第一种是**步内依赖（Step Distance $k = 0$）**：任务 $v$ 读取了任务 $u$ 在同一步产出的结果，或者通过状态版本的静态单赋值（SSA）顺序排在 $u$ 之后，要求两者的锚点满足 $a_v - a_u \ge d$（$d$ 为最小间隔，通常为 $0$ 或 $1$）。

第二种是**跨步依赖（Step Distance $k \ge 1$）**：任务 $v$ 在第 $t$ 步读取了任务 $u$ 在第 $t-k$ 步写入的状态版本（即 `publish_lag = k`）。在全局时间轴上，这意味着第 $t$ 步的 $v$ 必须晚于第 $t-k$ 步的 $u$ 至少 $d$ 个锚点：

$$(t \cdot M + a_v) - ((t - k) \cdot M + a_u) \ge d$$

把式子左右两边的 $t \cdot M$ 消掉，循环步数变量 $t$ 就彻底消失了，只剩下一个与步数无关的初等差分不等式：

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

在并行计算理论中，这种“所有依赖向量均为常数平移”的结构有一个经典的名字，叫**一致依赖（Uniform Dependence）**，Karp、Miller 和 Winograd 早在 1967 年研究一致递推方程时就已经透彻地分析过它。对于一致依赖，无限展开的跨步循环图可以直接折叠成一张只有有限个任务节点的**约化依赖图（Reduced Dependence Graph）**，每条从 $u(t)$ 指向 $v(t+k)$ 的有向边只带一个常数权重 $w = d - k \cdot M$。

在这张只有几个到几十个节点的小图上，求所有任务合法的 `[最早锚点, 最晚锚点]` 窗口，根本不需要整数线性规划求解器——它就是一个标准的**图上最长路（Longest Path）问题**，用教科书里的 Bellman-Ford 松弛算法就能直接算出精确解。

在 `rust/pllm_core/src/planning/scheduler.rs` 中，整个放置窗口的推导就是下面这 40 行函数 `propagate`：

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

这段代码几乎不需要额外解释：每次迭代沿着有向边做双向松弛，前驱的下界 `lo[u] + weight` 向后推高后继的最早可执行点 `lo[v]`，后继的上界 `hi[v] - weight` 向前压低前驱的最晚可执行点 `hi[u]`。

更自然的是它的两条错误返回路径。当数据结构和算法真正贴合问题本质时，算法失败的出口往往直接对应着物理世界里的错误原因：
* 如果某一步松弛后发现某个任务的下界越过了上界（`lo > hi`），说明这个任务读取了一个在它声明的阶段结束之后才诞生的值（比如在采样前的方法里试图读取采样后才生成的 `sampled_tokens`），报错信息直接提示用户将该任务声明到更晚的阶段；
* 如果松弛满 $|V|$ 轮之后边界仍在变化（`pass == bounds.len()`），根据 Bellman-Ford 定理，图中存在正权环——也就是跨步读写约束首尾相咬，形成了一个时间上的死锁环。

算出了每个任务的合法窗口 `windows` 之后，接下来的 `place` 函数负责把它们落位到具体的锚点与 CUDA stream 车道（`Lane::Main` 或 `Lane::Side`），并核算显存峰值：

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

这里有两处设计取舍值得挑明。

第一处是 `.or_else` 里的显存兜底策略。把任务放到旁路流上跨步异步执行，意味着它的输入缓冲、中间激活与状态版本必须跨越步边界存活更长时间，占用的显存峰值高于串行执行。如果 `plan_memory` 沿着时间线模拟张量生命周期后，发现旁路重叠的峰值超出了用户的显存预算，规划器会自动把所有任务退回 `Lane::Main` 串行执行再算一次——但注意那行注释：**退回主流时，严格保留已经选定的逻辑读边界（`read_origin` 与 `read_distance`）**。也就是说，显存吃紧只会让执行退化回串行，而绝不会偷偷改变算法读哪一步状态版本的数学语义；如果退回主流后预算依然不够，规划直接报错并给出所需的精确字节数。

第二处是诚实区分“什么是自动推导的”与“什么是明确规则”。在这套规划器里，放置窗口、状态版本危险检测（见[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）与显存峰值核算是由图算法自动求解的；而“哪些任务切到旁路流”（`select_lanes` 与 `select_prefetch`），靠的是几条直接反映硬件规律的确定性规则：
* 对于只读预取（`select_prefetch`）：只要一个任务只依赖 `input_tokens` 和不可变状态（`immutable`），并且具备进入 CUDA Graph 的能力，就把它放到 `Lane::Side`，在一步开头的 `request.prepare` 立即发射，与当前步的模型前向重叠；
* 对于状态更新（`select_lanes`）：只有当允许跨步滞后（`publish_lag >= 1` 或模型内读取）、没有自定义层循环或显式可见性覆盖、第一个写状态的任务落在 `model.execute` 之后、且后续任务不喂给采样器也不读取宿主服务时，才把从第一个写状态任务开始的后缀切到 `Lane::Side`。如果用户指定 `publish_lag=0` 要求更新立即可见，所有任务就老老实实留在主流串行。至于同一锚点内多个独立任务的复杂多流列表调度（List Scheduling），在没有真实负载需要它之前，我们干脆不做。

---

## 三、 认识代数，不认识名字：任务内部的递推加速

用 40 行最长路解决掉“任务之间”的放置与分流后，还剩下一半问题：**任务内部的计算怎么自动加速？**

在测试时训练（如 In-Place TTT）或线性注意力变体（如 GatedDelta）中，算法作者经常会写一个沿着序列方向逐位置更新状态 $S$ 的单步递推函数 $S_t = \text{step}(S_{t-1}, x_t)$。如果一个序列块有 128 或 1024 个 token，在 GPU 上用 Python `for` 循环一步步串行跑完，延迟会高得无法接受。

要自动加速这类递推，最忌讳的是在框架里靠识别 workload 的类名去硬编码替换。即便面对一段从未见过的用户自定义递推，只要它在数学上满足结合律，系统也应该能自动将它改写为并行前缀扫描（Parallel Prefix Scan）。

这件事同样不需要自定义一套庞大的编译器中间表示。在 `pllm/execution/scan_analysis.py` 中，我们直接用 PyTorch 原生的 `make_fx` 把用户的单步函数追踪成一张符号计算图（FX Graph），然后对状态变量（carry $S$）做一次轻量级的符号线性度分析：

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

分析器做的事情非常纯粹：**它不认识任何算法的名字，只认识代数结构。**

拿到单步 FX 图后，分析器先将所有不依赖上一时刻状态 $S_{t-1}$ 的节点（例如对输入 $x_t$ 做的线性投影、激活变换或归一化）剥离出来，组成一张独立的 `coefficients` 子图，直接交给 `torch.func.vmap` 在整个序列窗口上一次性批量算完（并配合公共子表达式消除 `fx_graph_cse` 去重）。

接着看剩下的核心状态更新部分。如果新状态 $S_t$ 可以被符号化简为仿射形式 $S_t = \mathcal{L}_t(S_{t-1}) + B_t$，且线性算子 $\mathcal{L}_t$ 属于上面 `Kind` 定义的四种闭合作用之一——标量缩放（`scalar`）、逐元素对角掩码（`diagonal`）、左乘矩阵（`left`）或右乘矩阵（`right`）——那么结合律就天然成立，递推会被自动分派到对应的固定容量前缀扫描内核（如 `additive_scan.py` 或 `right_matrix_scan.py`）；如果用户在递推里写了非线性的状态自乘，分析器就抛出 `Unsupported`，退回按步串行执行。在单张 RTX 4090 的微基准测试中（较早的一次测量，只测扫描本身），将 $64 \times 64$ 状态转移从逐位置串行改写为仿射前缀扫描，单头耗时从 1.44 ms 降至 0.22 ms（$6.57\times$），14 头耗时降低 $1.85\times$，完整状态转移从 5.88 ms 降至 1.76 ms（$3.34\times$）。（同时，为了防止基于测量的自动调优在个别形状下选出比未改写参考实现更慢的变形，选择器始终将参考实现留在候选池中同台比对，并在计划报告中如实记录最终入选的实现。）

这样一来，pLLM 与 PyTorch 原生编译栈的分工就变得异常清晰：**pLLM 只管“任务之间”的锚点放置、流车道与跨步代数识别；而单个任务内部的算子融合与寄存器分配，全部交给 `torch.compile`。**

---

## 四、 为什么在忙碌的 GPU 上，盲目重叠是零和博弈？

现在，规划器已经能算准放置窗口，也能把写状态的训练任务自动切到旁路流 `Lane::Side` 上了。但在这里，我们必须戳破一个在 GPU 编程里非常普遍的直觉误区。

很多人习惯把 CUDA stream 想象成多核 CPU 上的独立线程，觉得只要新建一条 `side` stream，把在线训练扔过去跟下一步的模型前向“并发执行”，训练的时间就被免费隐藏了。

在物理硬件上，这是不可能的。

多开一条 CUDA stream，并不会让你的服务器里多长出一张显卡。主流和旁路流上的所有内核，最终都要挤在同一张 GPU 的同一组流多处理器（SM）上，争抢同一条显存总线（HBM Bandwidth）。在逐字生成的解码阶段，大模型的主干前向和采样器本来就是高度受限于显存带宽的——当主流已经把显存带宽吃满时，旁路流上的训练任务挤进来，它并不是在“利用空闲时间”，而是在**和主流抢同一块蛋糕**。

下面三组在真实硬件上测得的数据，把这场资源竞争的真相展示得清清楚楚。

### 1. 撤掉采样前的栅栏，为什么“重叠度更高”反而更慢？

在单张 RTX 4090 上剖析 Qwen2.5-0.5B（128 并发，ESamp $\beta=0$，vLLM 0.27.1 时期测量）时，我们观察到每次前向仍有约 $70.33\ \mu\text{s}$ 的旁路训练时间露在主流之外。一个很自然的冲动是：既然训练只需要浅层和深层的隐藏状态，为什么不在采样器开始之前就把训练输入拷走、撤掉采样前的流等待栅栏，让旁路训练尽早跑起来呢？

如果你真的这么改，打开 profiler 的时间线一看，会觉得非常成功：露在主流外面的旁路时间从 $70.33\ \mu\text{s}$ 大幅缩减到了 **$8.57\ \mu\text{s}$**，几乎完全重叠进了主流内部。

但再看端到端的总吞吐，结果却倒退了：相对原生 vLLM 的吞吐从保留栅栏时的 **94.49% 掉到了 92.73%**（在另一组同样提前拷贝训练输入并撤掉采样栅栏的对照试验中，吞吐也同样从 94.3% 掉到了 92.9%）。

```text
Case A: Unfenced Early Side Launch (Collides with Bandwidth-Bound Sampler)
Main Stream:  [   Model Forward   ] ──► [    Sampler (Sweeping 152k Vocab!)    ]
Side Stream:                        ──► [ Train Copy + Backward + AdamW ]
Hardware Bus:                           █████████████████████████████████  <-- HBM Bus Contention!

Case B: Fenced Launch + TP Priority Staggering (Complementary Hardware Windows)
Main Stream:  [ Layer k Compute ] ──► [ Custom All-Reduce (NVLink/PCIe) ] ──► [ Layer k+1 ]
Side Stream:                          [     Replicated Local Train      ]
Hardware Bus:                         (Main waits on interconnect; Side uses local SMs/HBM!)
```

原因就在上面的图里。采样器正在扫描 `[128, 151936]` 的庞大词表张量做温度缩放、top-k/top-p 过滤和 Gumbel 抽样，这是一步之中显存带宽吃得最紧的时刻。撤掉栅栏让训练提前冲进来，训练的工作量并没有消失，它只是去和正在全力扫词表的采样器争抢显存总线，把两边都拖慢了。

### 2. 当模型变大时，重叠的相对收益如何被分母稀释？

像流水线并行（如 PipeDream 的权重暂存）或异步 SGD 一样，pLLM 的 `publish_lag=1` 本质上是用“一到两步的参数陈旧度”去换取跨步重叠的机会（第 $n$ 步的训练更新在第 $n+2$ 步对其他读取者可见，从而允许第 $n$ 步训练与第 $n+1$ 步前向并行）。但这笔交易在端到端吞吐上能换回多少百分比，完全取决于**一步训练占一步前向的比例**。

看看在同一张 RTX 4090（vLLM 0.32.0，开启 CUDA Graph 与 FlashInfer 采样器，8 个 prompt $\times$ `n=16` = 128 并发序列，生成 256 token，3 轮中位数）上，0.5B 与 7B 两个模型的实测对比：

| 模型与 ESamp 配置 | 原生 vLLM 吞吐（单步耗时） | `publish_lag=0`（主流串行等待） | `publish_lag=1`（旁路跨步重叠） | 旁路训练每步 GPU 耗时 |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B** ($\beta=0$) | 42,355 tok/s (`3.0 ms`) | 38,047 tok/s (**89.8%**) | 38,936 tok/s (**91.9%**) | ~178 µs（AdamW ~29 µs，0.27.1 剖析） |
| **Qwen2.5-7B** ($\beta=0$) | 5,894 tok/s (`21.7 ms`) | 5,731 tok/s (**97.2%**) | 5,738 tok/s (**97.4%**) | ~616 µs（AdamW ~191 µs） |
| **Qwen2.5-7B** ($\beta=0.1$) | 5,894 tok/s (`21.7 ms`) | 5,663 tok/s (**96.1%**) | 5,681 tok/s (**96.4%**) | ~616 µs + 引导采样开销 |

在 **0.5B** 模型上，一步解码总共只有 3.0 ms。此时几百微秒的训练与交接开销占比较大，开启 `publish_lag=1` 可以看到吞吐从 **89.8%** 提升到 **91.9%**（在早先 vLLM 0.27.1 的同配置共享卡测量中，开启重叠使 $\beta=0$ 从 91.4% 升至 94.3%、20 轮复测为 92.45%，$\beta=0.1$ 从 73.7% 升至 84.1%，1,245 个旁路内核中有 780 个与主流内核在时间上重叠）。

而一旦切换到 **7B** 模型，主干一步解码拉长到了 **21.7 ms**，rank-4 蒸馏网络的旁路训练一步耗时约 **616 µs**，只占主干步长的 **2.8%**。在这 21.7 ms 的分母面前，616 µs 的训练无论是串行等待（97.2%）还是放到旁路流重叠（97.4%），端到端吞吐的差距都收敛到了 0.2% 左右。

### 3. 什么时候重叠才是真红利？当两边的硬件瓶颈互补时

既然争抢同一条显存总线是零和博弈，那么旁路流在什么时候才能真正创造净收益？答案是：**当主流正在等待某种外部资源、把本地的 SM 或显存带宽让出来的时候。**

我们在[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)里见过第一个例子：旁路流限制在 1/4 的 SM（32 个线程块）上等待 23.7 GB/s 的 PCIe 总线拉取主机内存，把 3/4 的 SM 和显存带宽留给主流跑矩阵乘，两边资源互补，成功藏住了九成以上的 PCIe 传输耗时。

第二个极具代表性的例子发生在**多卡张量并行（Tensor Parallelism，TP > 1）**下。

在单卡（TP=1）时，“在采样前提前把训练更新写进私有状态槽、等到 `token.commit` 再发布”的做法并没有带来端到端收益。但到了 4 张 A100-SXM4-80GB 运行 Qwen2.5-7B 的多卡环境里，情况完全变了：Transformer 的每一层做完注意力或 MLP 之后，主流都必须停下来执行一次跨卡集合通信（vLLM 的 Custom All-Reduce）。如果某一张卡上的旁路训练恰好在这个时刻抢占了 SM，四张卡就会在同步障（Barrier）前发生木桶效应——所有卡一起停下来干等最慢的那张卡。

解法的主体是改放置：多卡下把自动状态更新挪到采样之前计算、写进私有状态，到 `token.commit` 才发布，让旁路训练不再和下一步前向里的 Custom All-Reduce 撞车。`pllm/worker.py` 再给多卡 worker 加上一条执行流优先级作为配套：

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

把多卡推理主流设为高优先级（`priority = -1`），旁路训练流保持默认优先级（`0`）——单独这一条只带来小幅提升（TP=4 下从 14,936 到 15,031 tok/s），主要收益来自上面“采样前写私有状态、提交时发布”的错峰放置。在 A100-SXM4-80GB 上实测 Qwen2.5-7B（ESamp $\beta=0$，rank 4，8 个 prompt $\times$ `n=16` = 128 并发，生成 256 token），旁路训练与 Custom All-Reduce 的冲突重叠时间直接从每步 **193–205 µs 降到了 0 µs**；连同同一批改动里去掉的冗余字段拷贝与梯度填充（单卡的提升来自这一部分），相对原生吞吐随之回升：
* **TP = 1**：从 93.79% 提升至 **95.84%**；
* **TP = 2**：从 89.09% 提升至 **95.63%**；
* **TP = 4**：从 **87.07% 跃升至 97.60%**。

---

## 五、 约束系统自己的复杂度

回顾自动放置与重叠的设计，最核心的教益其实在于对“问题规模与工具复杂度”的诚实校准。

理查德·萨顿（Richard Sutton）的《苦涩的教训（The Bitter Lesson）》说，长远看，能随算力扩展的通用方法总会胜过依赖人类先验的方法；大模型过去几年的成功，正是“抛弃人类先验、用统一的架构堆叠算力”这条路线的写照。然而那条定律成立的前提是单位算力成本持续按指数级下降。随着制程微缩放缓、显存带宽成为硬瓶颈，为了把单 token 的推理成本继续往下压，推理侧的算法正在不可避免地走向另一个方向——引入各种充满结构先验与状态的异构组件：从跨 PCIe 的巨型 n-gram 记忆查表（Engram），到测试时快权重更新（In-Place TTT），再到稀疏注意力与在线蒸馏引导。

上层的推理期算法注定会越来越繁杂、越来越有状态。**恰恰是因为上层算法在变复杂，承载它们的底层系统底座才更要克制自己的复杂度。**

面对“把几个扩展任务放进推理步”这样一个一维时间轴上的常数平移依赖问题，如果去引入一套面向任意高维循环嵌套的多面体编译基础设施，系统就会背上沉重的黑盒包袱；而一旦认清它只是“一致依赖（Uniform Dependence）”，整个放置与窗口验证问题就会退化为 `scheduler.rs` 里那 40 行一眼就能看穿的 Bellman-Ford 最长路松弛。

把复杂的理论留给真正需要它的高维空间，把任务内部的算子融合交给 `torch.compile`，用 40 行最长路管好任务之间的时间窗口与显存账本，再顺着 GPU 真实的带宽与通信空隙去安排双流重叠——看清问题的最小几何形状，永远是系统设计里最省力气的一步。
