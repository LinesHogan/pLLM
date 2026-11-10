# 在 serving 引擎里信任一段用户程序

pLLM 一直拿 eBPF 来类比自己：用户在外面写一小段程序，装进 vLLM 这个“内核”里全速运行，每段程序可以有自己的状态。前面的文章讲的都是“怎么让它跑得快”。这篇讲类比里最难的那一块：eBPF 之所以敢让用户程序进内核，是因为装载前有一个 verifier 证明它不会搞坏内核。那么，一段研究者随手写的 PyTorch 方法，凭什么能被放进一个每步只有几毫秒、同时服务上百个请求的引擎？

先把结论说在前面，免得类比被读得太满：对任意的 PyTorch 代码，我们**证明不了**任何有意思的语义性质。我们能做的是三件更朴素的事——把程序关进一个已知的形状，在装载前检查结构上的错误，再对少数几条语义性质做有针对性的差分测试。其中只有最后一件算得上新东西。

## 三个不报错的 bug

先看几个在研究脚本里完全正常、放进 serving 引擎就出事的写法。

第一个是一个“激活中心化”的方法：

```python
# check-docs: skip: a one-line method body, not a complete program
return h - h.mean(dim=0, keepdim=True)
```

在 HF 里单请求调试时，第 0 维长度是 1，这行代码看不出任何问题，单元测试也能过。可是 vLLM 用连续批处理：每一步把所有请求这一步要算的 token 首尾相接，拼成一张扁平的 `[tokens, hidden]` 张量（[《行模型》](04-packed-rows.md)讲过这张表怎么拼）。第 0 维是不同请求的行。于是一个请求的输出，取决于这一步恰好和谁拼在一起。它不报错，不崩溃，数值也像模像样，只是结果随负载悄悄漂移。

第二个是一行调试用的断言：

```python
# check-docs: skip: a one-line method body, not a complete program
assert not torch.isnan(loss).any()
```

`assert` 要把一个 GPU 上的张量变成 Python 的 `bool`，CPU 只能停下来，等 GPU 把排在前面的所有工作做完、把结果拷回来。这叫一次主机同步。在 RTX 4090 上，7B 模型一步 decode 约 21.7 ms，0.5B 只有约 3.0 ms（128 个并发请求，vLLM 0.32）；CPU 本来领先 GPU 一步在准备下一步的数据，一次同步就把这点领先全部打掉。更麻烦的是，decode 步是录成 CUDA graph 回放的，回放时 Python 根本不执行，这行断言要么录不进去，要么只在录制时跑过一次（[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。

第三个藏得更深。KV cache 不够时，vLLM 会抢占一个请求，过一会儿把它的 prompt 和已生成的 token 从头重算一遍。一个在线训练的算法如果不知道这件事，就会把同一段 prompt 再训练一遍。

退一步看，这三个 bug 其实是同一种 bug：一个请求的结果，依赖了它不该依赖的东西——同一批里还有谁，这一步发生了什么，它之前有没有被赶出去过。在 HF 的单请求世界里，这些东西根本不存在，所以没人会想到去防；在 serving 引擎里，它们每一步都在变。换句话说，我们真正想要的性质不是“这段代码对不对”，而是“**这个请求的结果，对调度的变化保持不变**”。

这句话是整篇的钥匙。它也说明了为什么运行时检查帮不上忙：错误只在特定的组合下出现，而每步只有几毫秒，任何检查要么引入同步，要么拖慢所有请求，何况你也不知道该在哪一步、对哪种组合去查。检查只能在装载之前做，最好在 CPU 上做——前提是我们能把“调度”本身写成可以随意摆布的数据。后面会看到，这正是 pLLM 能做、而一般的插件系统做不到的地方。

## 先把程序关进一个形状

eBPF 程序之所以能被分析，首先是因为它只能调用白名单里的 helper。pLLM 的对应物是声明：一个方法读什么，写在参数的默认值上；状态写成类属性；效果写成装饰器。用户文档里那个在线蒸馏的例子（`doc/explain/performance.md`）：

```python
# check-docs: skip: excerpt from doc/explain/performance.md
@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        init=pllm.FromModule(lambda m: nn.Sequential(
            nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size)
        )),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)
```

框架不需要猜这个方法读什么：`pllm.hidden(0)` 说它要第 0 层的 hidden，`pllm.module("distiller")` 说它要这个请求自己的那份小网络，`@pllm.train` 说它会更新状态。类是冻结的 dataclass，写 `self.count += 1` 直接报错——方法并不是“每步被 Python 调用一次的函数”，可变的 Python 字段在回放时根本不会变。

这一层本身是普通的工程。反射只发生在一条很窄的边界上（`pllm/usage.py` 和 `pllm/workload.py`），过了这条边界，用户程序就变成了带类型的声明和一张语义图，后面没有人再去看 Python 对象。但后面的一切都依赖它：没有声明，规划器就不知道依赖关系，CPU 检查也不知道该合成什么样的输入。

## 结构上的错误，在规划时就能抓到

语义图交给 Rust 写的规划器。它做的检查都是编译课程里的老算法，我们不把它们当新东西讲，但有两点值得一看：报错的写法，以及一个顺手得到的“同步检测器”。

最直接的是依赖成环。两个方法如果通过结果或者状态读写互相等待，拓扑排序就排不完（`rust/pllm_core/src/planning/scheduler.rs`）：

```rust
    let mut order = Vec::with_capacity(graph.tasks.len());
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for successor in &successors[index] {
            indegree[*successor] -= 1;
            if indegree[*successor] == 0 {
                ready.insert(*successor);
            }
        }
    }
    if order.len() != graph.tasks.len() {
        let waiting: Vec<_> = (graph.tasks.iter().zip(&indegree))
            .filter(|(_, &degree)| degree != 0)
            .map(|(task, _)| task.symbol.as_str())
            .collect();
        return Err(format!(
            "methods {waiting:?} wait on each other through results or state reads and writes; \
             remove one of the dependencies in the cycle"
        ));
    }
```

这是教科书上的 Kahn 算法。我们想指出的只有两处：`ready` 是一个 `BTreeSet`，`pop_first` 每次取编号最小的就绪任务，所以同一张图永远排出同一个顺序，计划是输入的确定函数；报错不说“检测到环”，而是列出卡住的方法，告诉用户“拿掉环里的一条依赖”。

第二类是读到未来的值。每个任务有一个允许运行的时间窗口（一步里的一段锚点区间），依赖边带着“至少晚几个锚点”的权重。规划器把窗口沿着边来回收紧；收紧到下界超过上界，就说明某个任务要读的东西在它能运行的时候还不存在——比如一个挂在前向之前的方法，却声明要读这一步采样出的 token。同一个文件里的 `propagate` 这样报告：

```rust
        if lo.iter().zip(&hi).any(|(a, b)| a > b) {
            return Err(
                "a task reads a value (sampled tokens, hidden states or another task's \
                 result) that exists only after its phase ends; declare it in a later phase"
                    .into(),
            );
        }
```

第三类是两个写者写同一个版本。pLLM 的状态带版本（[《PTTS》](01-ptts.md)），规划器把它当 SSA 看：每次写产生一个新版本。如果两个方法都从同一个版本出发写同一份状态，谁的结果算数就取决于两条 CUDA stream 谁跑得快——这在 GPU 上是不可接受的不确定性（`rust/pllm_core/src/planning/version_hazards.rs`）：

```rust
            if let Some((other, _)) = commits.insert(origin(effect), (writer, commit)) {
                return Err(format!(
                    "{} and {} both write state {:?} from the same version; let one method write it",
                    graph.tasks[other].symbol, graph.tasks[writer].symbol, resource.resource
                ));
            }
```

还有一个检查是顺手得到的。规划之前，框架用 PyTorch 的 meta 张量——只有形状和 dtype、没有数据的张量——把每个方法追踪一遍。任何需要真实数值的操作，`.item()`、对张量值写 `if`，在 meta 张量上都没法执行，追踪当场失败。也就是说，同一次形状追踪顺带成了一个零成本的同步检测器。框架把 PyTorch 内部的异常翻译成用户能照着改的话（`pllm/execution/torch_private.py`）：

```python
# check-docs: skip: excerpt from pllm/execution/torch_private.py
@contextmanager
def fixed_shapes() -> Generator[None]:
    """Explain why data-dependent control flow and custom-op outputs cannot be planned or captured."""
    try:
        yield
    except GuardOnDataDependentSymNode as error:
        raise ValueError(
            "methods run inside CUDA graphs, so Python cannot branch on tensor values (if, while or break on a "
            "tensor); run a fixed number of steps and select rows with torch.where, or in a custom_forward stop "
            "rows with layers(..., active=mask)"
        ) from error
```

同一次追踪还给出了中间张量什么时候诞生、什么时候死，以及 FLOP 和字节数的估计。一次 CPU 追踪，四样东西。

可是到这里为止，开头那个 `h.mean(dim=0)` 我们一样也抓不到。它形状合法，没有同步，没有环。它错在语义上：一个请求的输出依赖了别人的数据。

## 改掉别人，看看自己

对语义性质我们没有 verifier，但有一个老办法：蜕变测试（metamorphic testing）。很多时候我们不知道“正确输出”是什么，却知道某些变换下输出必须不变。测一个排序函数，你不必知道排好的数组长什么样，只要知道“把输入打乱，输出不变”。

serving 引擎里的用户方法也有这样的不变式。对任何一个请求来说，三个请求一起跑和它单独跑，它的输出必须一样；按 0-1-2 的顺序拼和按 2-1-0 拼，必须一样；把其他所有请求的输入都改掉，它的输出也必须一样；它在 decode、别人在做分块 prefill 的混排调度下，结果要和常规调度一样；它中途被抢占、再全量或分块地重放历史，结果要和没被抢占时一样；两个 workload 组合在一起，每个组件的输出要和它单独运行时逐位相同；把干预强度设为零，就要和原模型一样。

`pllm.testing.check` 做的就是这件事。它不加载模型权重、不用 GPU，只读模型的 `config.json`，用**真实的**规划器和执行器，在 CPU 上按几套固定调度把同一个 workload 跑几遍，然后两两比较。

第一个问题是：没有模型，方法要读的 `pllm.hidden(0)` 从哪来？答案是合成，而且每个值只由“请求、位置、通道”决定（`pllm/testing_contracts.py`）：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
def _random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _token_values(index: int, offsets: range, width: int, channel: int, perturbed: int | None) -> torch.Tensor:
    salt = 1_000_000 if perturbed is not None and index != perturbed else 0
    return torch.stack(tuple(_random((width,), salt + 10000 * index + 100 * offset + channel) for offset in offsets))
```

确定性保证了“同一个请求在不同调度下拿到同样的输入”，比较才有意义。而 `salt` 那一行，就是整个隔离检查的全部机关：测请求 `perturbed` 的时候，**除它以外**所有请求的随机种子都加上一百万，输入面目全非；它自己的输入一个比特都没变。如果它的输出变了，那一定是有谁跨行读了数据。

我们很喜欢这一行。要测“我有没有受别人影响”，最直接的办法不是去分析代码里有没有跨行的操作——那需要理解任意的 PyTorch——而是把别人全部换掉，看看自己变没变。比较失败的时候，框架还能分清是谁的错：如果某个方法拿到的这个请求的输入相同、输出却不同，是这个方法混了行；如果输入在进方法之前就已经不同，那是 pLLM 自己把行送错了，报错会请你把它当作框架的 bug 提交。

调度表本身也有讲究。三个请求的 prompt 长度分别是 5、1、9，每一步的调度写成一串三元组 `(请求, 本步算几个 token, 之前已经算了几个)`。混排的那一套是这样的（`_schedule`）：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
if mixed:
    return (
        ((0, 4, 0), (1, 1, 0)),
        ((2, 4, 0), (1, 1, 1), (0, 1, 4)),
        ((0, 1, 5), (1, 1, 2), (2, 4, 4)),
        ((2, 1, 8), (0, 1, 6), (1, 1, 3)),
        ((1, 1, 4), (2, 1, 9), (0, 1, 7)),
        ((0, 1, 8), (2, 1, 10)),
        ((2, 1, 11),),
        ((2, 1, 12),),
    )
```

第一步，请求 0 的 5 个 prompt token 只算了 4 个，请求 1 的唯一一个 prompt token 算完；第二步，请求 2 开始分块 prefill，请求 1 已经在 decode，请求 0 补完它最后一个 prompt token……行序也每步打乱。这里就是前面说的“把调度写成数据”。这三个数，恰恰是 vLLM 每步交给 pLLM 行标签器的东西（为什么三个数就够，见[《行模型》](04-packed-rows.md)）。CPU 检查把这张表喂给生产里用的同一个 Rust 标签器，于是测试里的行打包、选择、映射和 GPU 上走的是同一份实现，而不是另写一个“模拟 vLLM”的假调度器。（要诚实地补一句：抢占重放时“哪些 token 已经被消费过”的标记，CPU 检查用的是一个 Python 写的参考实现，和生产里 Rust 的那份是分开写的。）

有些东西 CPU 上跑不了，最重要的一类是模型内部的方法：改残差、覆盖参数、自定义注意力。CPU 上没有模型，它们就没有地方挂。框架不假装通过，而是在结果里写一行明确的跳过（“CPU decoder does not execute residual hooks, overlays, spans or PositionState”）。跳过不是通过。

## GPU 上补的那一部分

CPU 抓不到的，交给 `pllm.testing.check_on_gpu` 在真实引擎上补。

零同步的检查用的是 PyTorch 自带的 `torch.cuda.set_sync_debug_mode("error")`：稳态 decode 的每个框架入口都在这个模式下跑，任何隐式同步直接抛异常。可是这个开关有死角，显式的 `Event.synchronize()` 它看不见。2026 年 10 月我们对 0.5B 的 ESamp（RTX 4090，vLLM 0.27.1，128 个请求 × 256 token）另外做过一次审计，逐个追到直接的调用者：整次生成有 128 次主机等待，全部来自 vLLM 自己取输出的 `get_output`，pLLM 自己的 event/stream 等待是 0 次。调试开关只能当第一道筛子，“零”要靠审计调用者来确认。

抢占在 CPU 上只是调度表里的一种排列。在 GPU 上，我们让抢占真的发生一次：包一层 vLLM 真实的 KV 分配器，让它对某个请求失败一次（`pllm/testing_gpu.py`）：

```python
# check-docs: skip: excerpt from pllm/testing_gpu.py
class _Eviction:
    """Make the real KV allocator fail once, then let vLLM evict and recompute normally."""

    def __init__(self) -> None:
        self.request: Request | None = None
        self.requests: dict[str, Request] = {}

    def wrap(
        self, allocate: Callable[Concatenate[Request, _Allocation], KVCacheBlocks | None]
    ) -> Callable[Concatenate[Request, _Allocation], KVCacheBlocks | None]:
        def allocation(request: Request, *args: _Allocation.args, **kwargs: _Allocation.kwargs) -> KVCacheBlocks | None:
            self.requests[request.request_id] = request
            if self.request is None and request.num_output_tokens >= 4:
                self.request = request
                return None
            return allocate(request, *args, **kwargs)

        return allocation
```

分配器一返回 `None`，vLLM 就按它自己的逻辑驱逐、稍后重算；我们比较这个请求最后的 token 和没被驱逐时是否一致。这比自己写一个“模拟抢占”可靠得多，因为走的是 vLLM 真正会走的那条路。

最后是中性逐位：把干预强度设为零，必须和原生 vLLM 在同样的图模式下逐位相同。这条看起来最简单，其实最苛刻——[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)里讲了它在 7B 上逼我们放弃一部分编译器融合的事。

## 把代价摆上台面

eBPF 有 perf 和 bpftrace 让人看见程序在内核里花了多少；pLLM 的对应物是 `pllm.explain(workload, config)`。它同样只在 CPU 上跑，对 Qwen2.5-7B 的配置，本机上第一次调用约 0.2 s，同一个进程里再调一次约 0.1 s。下面是 Distill 在 7B 上的节选：

```text
forward  CUDA graphs: no effect needs eager
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0)
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
tap      model.layers.0[0], decode: 1837056 bytes shared by loss, predict
state  scope  roots  bytes/root  reserved bytes [exact; state_page_budget_bytes]
distiller  request  514  516176  265314524  field 0: float32, field 1: float32, ...
memory   peak 1512728864 bytes funded by pLLM [per-rank tensor reservations; excludes GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1512728864 bytes tasks run inline; below 1333743904 bytes planning fails
host     0 syncs in traced workload math
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

每一行回答一个“框架替你做了什么决定”的问题：模型前向保没保住 CUDA graph，每个方法挂在一步里的哪个时刻、跑在主流还是旁路流、读哪个版本的状态，状态留了几份、每份多大，显存峰值和两条门槛——预算低于第一条，旁路任务放弃重叠、退回主流；低于第二条，规划直接失败。

背后的原则是不静默回退。昂贵的取舍——把表放到主机内存、关掉 CUDA graph、放弃重叠——要么由作者显式声明，要么在规划时失败并说明原因，而不是悄悄换一条慢路径。唯一的例外是 `cuda_graphs="auto"` 时模型可以退回 eager，原因写在 `forward` 那一行里。

显存的推导用的也是那次 meta 追踪。2026 年 10 月，我们把各个 workload 手写的显存公式全部换成追踪推导：ESamp `predict` 的工作区预留从手写的 177.6 MB 降到推导的 27.8 MB，OverRIDE 的训练从 1,056 MB 降到 267 MB，而后者在 GPU 上实测的峰值是 264,242,176 字节，落在推导值 267,269,120 字节以内（历史测量）。手写公式的问题不只是不准，而是它和代码是两份东西，代码一改它就过时；推导和代码至少是同一份。

## 这套检查做不到什么

eBPF 的 verifier 给的是证明：在它接受的程序上，某些坏事一定不会发生。pLLM 给的是固定调度下的差分测试。这两者差得很远，我们不想让类比掩盖这一点。

最大的缺口恰恰在最容易出错的地方：模型内部的方法——改残差、改权重、改注意力——在 CPU 上全部跳过，只能上 GPU 去查，而这类方法最容易写错维度、最容易破坏逐位中性。调度也只有固定的三个请求、几套手写的排列，不是 Hypothesis 那样随机生成的属性测试，没测过四个以上的请求，也没测过极端长度。合成的输入是随机正态的，擅长发现“行混了”这类结构错误，却会掩盖依赖特定数值的问题，比如接近零时的除法、极端 logits 下的溢出，那要靠 `pllm.testing.check_distribution` 在真实模型上去量。`explain` 的 FLOP 和字节估计还没有和实测系统地校准过，它也不含模型权重、KV cache、vLLM 的激活和依赖 GPU 的 BLAS 工作区——能不能装下，要到 `make_llm` 时才知道。还有一点：fail-closed 有时显得保守，[设计取舍](../design-tradeoffs.md)里列了一串今天写不了的组合，其中有些也许是安全的，只是我们还没有办法检查，于是先拒绝了。

## 一点可以带走的东西

如果说这一篇有一个统一的抽象，那就是**把调度当成数据**。生产里，vLLM 每步交给我们的是一串“请求、本步几个 token、之前算了几个”；测试里，我们自己写几串这样的三元组，打乱它、混排它、在中间插一次抢占，喂给同一个标签器、同一个规划器、同一个执行器。一旦调度是数据，“结果对调度不变”就从一句愿望变成了一组可以在 CPU 上几秒钟跑完的比较；“改掉别人，看看自己”也只是生成输入时多加的一个盐。

这个想法在别处也用得上：凡是“同一份计算要服务很多租户”的系统——数据库的多租户、共享一个模型的多个请求、一个调度器上的多个作业——隔离性往往很难证明，却很容易测试，只要你能把“租户们如何交错”写成数据，然后把除了被测租户以外的一切都换掉。

剩下的——声明、拓扑排序、区间传播、报错的措辞——都是普通的工程。它们之所以值得写进这个系列，是因为每一条都被同一个问题逼着：在一个我们不拥有的引擎里，怎样提前知道一段我们没写过的程序会不会出事。而对这个问题，我们诚实的回答是：证明不了，但可以把它放进足够多、足够刁钻的世界里跑一遍，看它有没有变。
