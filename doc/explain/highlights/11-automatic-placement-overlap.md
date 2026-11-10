# 自动放置与重叠

给推理引擎加一段用户程序，最后都会落到三个问题上：它在一步里的**什么时候**跑，跑在**哪条 CUDA stream** 上，它写的状态**什么时候**对别人可见。这篇讲 pLLM 怎样自动回答这三个问题，以及一段我们自己觉得最值得讲的经历：为了回答它们，我们引入过一整套多面体编译器，后来又把它删掉了。

## 一个问题，三种问法

还是从 ESamp 说起。它每一步有三件额外的事。前向结束后，用第 0 层和最后一层的 hidden 训练一个小网络：反向、AdamW。采样之前，用这个小网络的预测去引导采样。还有一些 workload 要在前向开始前查一张大表，比如按 token 的 n-gram 去取向量（[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)）。

把训练串在主流上最简单，可每一步就多等几百微秒：在 RTX 4090、vLLM 0.32、8 个 prompt × n=16 = 128 行、256 token 的设置下，0.5B 一步只有约 3.0 ms，这点时间相当显眼。放到旁路 stream 上（GPU 上的另一条任务队列）呢？下一步的前向读到的是哪个版本的参数，得有人说清楚。查表只依赖 token id，能不能在前向开始之前就发出去？能，但它必须在模型里消费它的那一层之前完成。

pLLM 最早是靠手写规则回答这些问题的：采样后重叠的特判、签名里写死的“哪些情况允许重叠”、Python 侧的放置标志。每来一个新算法，就要加一条规则。作者想要的是一个**有原则的自动方法**。

退一步看，这三个问题其实是同一个问题：**每个任务在时间线上的位置**。“什么时候跑”是它在一步里的位置；“哪条 stream”取决于它和后面的人隔了多远；“什么时候可见”是它的写和别人的读在时间线上相距几步。只要把时间线说清楚，三个问题就有了共同的语言。

## 时间线：锚点和常数平移

一步 decode 在 pLLM 里被切成一串有序的**锚点**：`request.prepare`（这一步开始，输入 token 就绪）→ `model.execute`（前向）→ `state.publish`（前向结束，各层 hidden 就绪）→ `sampler.pre_draw`（logits 偏置和采样引导要在这之前就绪）→ `sampler.complete` → `token.commit`（采出的 token 确定）→ `next.prepare`（[核心概念](../core-concepts.md)）。每个值都在某个锚点“诞生”：hidden 在 `state.publish` 之后才有，采出的 token 在 `sampler.complete` 之后才有。于是每个任务都有一个允许运行的窗口：最早是它所有输入都诞生之后，最晚是它的第一个消费者需要它之前。

一步之内，锚点是全序的。跨步的依赖也很规整：读 `publish_lag=k` 的状态，就是依赖第 t−k 步的写者——**永远是一个常数平移**。在并行计算的文献里，这种“依赖向量全是常数”的结构叫一致依赖（uniform dependence），Karp、Miller 和 Winograd 在 1967 年研究一致递推方程时就讨论过它。对一致依赖，合法的调度窗口可以由约化依赖图上的最长路求出来。

这句话在后来救了我们。但我们是绕了一大圈才回到它的。

## 一台很大的机器

2026 年 8 月到 9 月，为了得到“有原则的自动方法”，我们走了教科书的路：多面体模型。它把循环的每一次迭代看成多维整数空间里的一个点，把依赖看成点与点之间的仿射关系，然后用整数规划去求合法的分块、并行和重排。ISL（Integer Set Library）是这类编译器（Pluto、PPCG、Polly）的基础库。我们先用 ISL 的 must-source 数据流分析推导仿射值依赖，然后把 ISL 和它的整数库作为 vendor 依赖引进来，接着落地了一个 region IR 中端：符号依赖分析、ISL 调度候选、流划分、保留分析、容量分析，都在里面。

这台机器是认真做的：写了、测了、写了文档。两个月后，我们做了一次盘点，问一个简单的问题：**真正带来加速的东西，到底靠不靠它？**

答案让人不太舒服。真正测到收益的两项优化——仿射递推变并行前缀扫描（历史测量：单头 64×64 的状态转移 6.57×，14 头 1.85×），和 ESamp 的旁路重叠——发现它们的逻辑都不靠 ISL。识别仿射递推的是一个纯 Rust 的表达式模式匹配器，ISL 调用零行，只是外面套了一个 region 的壳；重叠的可见性判断，是把一个整数滞后 k 格式化成 ISL 字符串，去判断 `0 <= b < t`——也就是 `lag >= 1`。反过来，真正重度调用 ISL 的分块、流划分、保留、容量这些模块，在生产 workload 上要么只产出平凡的恒等调度，要么根本走不到。有一次提交的说明写得很直白：每次编译额外做的那次 ISL 调度搜索，产出的字段“Python never consumed”。（这份盘点是一份二手调研；后来的删除计划又在当时的代码上逐项核对了每一个 region/ISL 的使用者。）

还有一个用户能直接碰到的入口，测出来更糟。一个把请求展开进 ISL 程序的 ModelScan 接口（RTX 4090，0.5B，eager，prefill 1024 token，chunk 128，5 个目标层）：1 个序列 146 ms，2 个 293 ms，4 个序列报“ISL quota exceeded”，8 个直接 `LibISLError`。同样的 prefill 下，一条不经过 ISL 的前向内训练路径是 19.6 ms（两者数学不完全相同，看量级）。另一边，逐 token 的 region 执行在被换成批量调用 vLLM 原生层之后，层循环一类的程序快了一个到两个数量级：recirculation 的 decode 从每步 26.23 ms 到 3.26 ms，layer_loop 的 prefill 从 3,550.81 ms 到 14.76 ms（0.5B，历史测量，见[《自定义层循环与状态传播》](10-custom-layer-loops.md)）。

## 删掉它

2026 年 10 月 26 日，作者批准了删除，同时提了一个要求：**必须保留一个有原则的自动方法**——没预见到的用户递推仍然要被加速，而且不能靠逐个 workload 的规则。删除于是分成三步：先用“锚点 + 一致依赖”的放置调度器替掉所有放置逻辑；再用一个 FX 上的仿射分析替掉 region 里的递推识别；最后，10 月 27 日，把 region 与 ISL 整个退役。

那次退役删掉了 vendor 下 1,530 个文件、457,770 行代码；Rust 核心净减 7,171 行，19 个模块退役；Python 生产代码从 18,320 条语句减到 14,172 条（净减 4,148），44 个实现文件退役；测试净减 2,033 条语句。

我们想把这件事讲清楚，是因为它最难的部分不是技术。Sutton 在《苦涩的教训》里有一句话：“花在一条路线上的时间，就是没花在另一条路线上的时间；人们在心理上会投入到某一条路线中。”一台已经写好、测过、写了文档的机器，最难的是承认它没有在解决真正的问题。盘点给了我们勇气：它把“这台机器很漂亮”和“这台机器在生产里做了什么”分开了。多面体模型为任意的仿射循环嵌套而设计，而 serving 引擎里的调度问题——几个到几十个任务，一维的时间线，常数的跨步依赖——根本用不上它的大部分能力。为一个问题引入比它大得多的工具，代价是每一次调试都要穿过那个工具。

删掉之后，“自动”靠什么？两样很小的东西。

## 四十行最长路

第一样是放置。前面说过，对一致依赖，合法窗口由最长路给出。`rust/pllm_core/src/planning/scheduler.rs` 里的 `propagate` 就是这件事的全部：

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

第一行注释就是整个理论：跨步的边 `u(t) → v(t+k)` 在约化图上的权重是 `d − k·M`，M 是一步里的锚点数——把“晚 k 步”折算成“晚 k·M 个锚点”。然后是 Bellman-Ford 式的松弛：前驱的下界推着后继的下界，后继的上界压着前驱的上界。失败只有两种方式，各对应一句用户能照着改的话：某个任务的下界超过了上界，说明它要读的东西在它能运行时还不存在——“declare it in a later phase”；松弛了 n 轮还在变，说明约束里有矛盾的环。窗口的初值来自方法声明的阶段和后端提供的锚点；依赖边来自结果输入、显式的边，以及状态版本的 SSA 依赖（[《在 serving 引擎里信任一段用户程序》](02-trusting-user-programs.md)）。

窗口有了，再选具体的位置和车道。`place` 把这件事写得很短：

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

每个任务放在窗口的最早处；然后选车道、选预取；最后按选定的放置扫一遍显存峰值。放不下怎么办？`or_else` 那几行：所有任务回到主流再试一次——但保留已经选定的逻辑读边界，也就是版本语义不变，只是失去重叠。再放不下，就规划失败，报出需要的字节数。

这里要诚实：**自动的是窗口推导和显存核算，车道选择仍然是规则。** `select_lanes` 只在存在滞后（`publish_lag ≥ 1` 或者有模型内的读）、没有 custom_forward 或显式可见性、写者的锚点在模型执行之后等条件都满足时，才把“第一个写状态的任务”及其后的任务放上旁路流；`select_prefetch` 只把“只读 token、只读不可变状态、能进 graph”的任务放到旁路流上、在 `request.prepare` 发射。这几条规则都有物理理由，但它们是规则。同一锚点内多个独立任务的多流调度，计划里写的是“等有 workload 需要再做”。

还有一条规则是被测量逼出来的。放置调度器的第一个实现在 eager 后端上加了逐层的挂点，让任务可以放在“第 l 层之后”。测出来是回退：几个 producer 示例相对原生从 0.722 到 0.656、0.723 到 0.667、0.554 到 0.505（RTX 4090，0.5B，batch 8 × 32，eager），逐层 hook 加旁路分发让每轮多了约 14% 的宿主 CPU 时间。决定是：**调度器只能把任务放在后端声明的执行点上**；eager 后端只声明粗粒度的点，逐层分发删掉，不留开关。原则让位于测量。

## 认识代数，不认识名字

第二样是任务内部的加速。作者的要求是“没预见到的递推也要被加速”。替代 region 的，是 `pllm/execution/scan_analysis.py` 里的一个 FX 分析：用 `make_fx` 追踪作者写的单步函数 `S' = step(S, x)`，对 carry 做线性分析——与 carry 无关的子图抽出来，用 `vmap` 在窗口上批量算；carry 的更新如果是 `S' = L(S) + B` 的形式，就按 L 的种类分派到现成的前缀扫描 kernel；不是，就老老实实串行。它认识的种类只有四种：

```python
# check-docs: skip: excerpt from pllm/execution/scan_analysis.py
Kind = Literal["scalar", "diagonal", "left", "right"]
...
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

这就是“认识代数结构，而不是认识 workload 的名字”：分析器不知道什么是 In-Place TTT、什么是 GatedDelta，它只知道一个更新是不是对状态的标量、逐元素、左乘或右乘的仿射作用。一个从没见过的递推，只要落在这四种里，就会被加速。分工也随之清楚了：2026 年 10 月 28 日定下的原则是，**pLLM 管任务之间**——放在哪、哪条车道、读哪个版本、主前向里哪些值要留给训练；**torch.compile 管任务内部**——融合、AOTAutograd 的保存与重算。

删掉之后的测量也提醒我们“自动”本身要被检查。删除前的计划在当时的 HEAD 上发现，按测量自动选出的扫描形式在 InPlaceTTT 64×64 上反而比参考实现慢（参考 / 选中 = 0.707×）。于是新的选择器要求永远保留参考实现作为候选，并报告它选了什么。

## 重叠是零和的

最后说车道。很多人把 CUDA stream 想成线程：开一条新流，把工作扔过去，就白赚了并行。可两条流共享的是同一组 SM、同一条显存总线。主流已经吃满某种资源的时候，旁路任务挤进来只是在分同一块蛋糕。

我们有三次测量能说明这件事。第一次：2026 年 10 月实现重叠时试过一个更激进的版本——在采样之前就拷走训练输入、去掉采样栅栏，让训练和采样完全重叠。结果 92.9%，不如保留栅栏的 94.3%（0.5B，vLLM 0.27.1，3 轮中位数）；trace 显示训练在和采样抢资源。第二次：后来一次调查去掉了采样器前的两个等待，让训练更早上旁路流，“旁路在主流之外”的时间从每次前向 70.33 µs 降到 8.57 µs——可吞吐从 94.49% 掉到 92.73%。训练没有消失，它只是跑去和正在扫词表的采样器抢显存带宽。第三次是张量并行：TP 下每层都有一次 all-reduce，旁路训练撞上它，全卡一起等。我们在 TP1 上拒绝过“提前发布私有训练”的改动，在 TP>1 上它却有用——因为它把训练和 vLLM 的 custom all-reduce 错开了：再加上把推理流的优先级设成 −1、训练流设成 0，重叠时间从 193–205 µs 降到 0，TP=4 从 87.07% 到 97.60%，TP=2 从 89.09% 到 95.63%，TP=1 从 93.79% 到 95.84%（A100-SXM4-80GB，7B，ESamp β=0，rank 4，8 × 16 × 256）。

所以正确的心智模型是：**重叠只在两边资源互补时有收益。**[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)里有一个正面的例子：1/4 的 SM 在等 PCIe，3/4 的 SM 在跑前向，两边要的东西不一样，才藏得住。

`publish_lag` 本身也可以换个透镜看：PipeDream 的权重暂存、异步 SGD 的陈旧梯度，都是“用有界的陈旧换重叠”。它值多少，和步长成反比。今天的测量（RTX 4090，vLLM 0.32，128 行，256 token，3 轮中位数）：0.5B 上 ESamp β=0 在 lag=1 时是原生的 91.9%，lag=0 时 89.8%；7B 上分别是 97.4% 和 97.2%——一步 21.7 ms 里，旁路训练每步约 616 µs，串不串行都不太要紧了。

## 删掉之后，少了什么

删除是有代价的，而且应该说出来。region 级的分块、多流、保留分析这些能力不再存在了；作者最初设想的“自动调度”，今天实现的只是其中一层：放置窗口、显存核算、递推识别。车道选择是规则，不是代价模型；同一锚点内的多流调度没有做；FX 分析只认四种线性形态，非线性的递推只能串行；Rust 里的实现选择搜索，目前还没有 workload 真的在练它。`publish_lag=0` 时没有旁路训练；显存紧时自动回到主流，这条退路的吞吐代价我们还没有测过；eager 后端只有粗粒度的锚点可用。

## 可以带走的东西

这篇里真正的转折，是认出问题的形状：一维的时间线、常数的跨步依赖。认出来之后，四十行最长路就够了，四十多万行的 vendor 代码就成了负担。很多时候，“有原则的自动方法”不是去找最强大的通用工具，而是先弄清自己的问题属于哪个最小的类。

另一个可以带走的，是删除本身。一台写好、测过、写了文档的机器，删掉它需要一次诚实的盘点：把“它很漂亮”和“它在生产里做了什么”分开。推理时的算法只会越来越杂、越来越有状态；正因为如此，承载它们的底座更应该保持小——小到一个新来的人能读懂它，小到出了问题时，我们知道该去哪里找。
