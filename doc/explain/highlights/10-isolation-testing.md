# `h - h.mean(dim=0)` 不会报错，但会让张三看见李四的提示词：连续批处理的隔离测试

在单请求的研究脚本里调试一个“激活中心化”钩子时，你很容易随手写下这一行代码：

```python
# check-docs: skip: a one-line method body, not a complete program
return h - h.mean(dim=0, keepdim=True)
```

在单请求环境下，张量第 0 维的长度永远是 1。这行代码不抛异常、不触发形状告警，梯度能正常回传，单元测试一片绿灯。但只要把这行代码原封不动地放进生产级的 vLLM 引擎里，一场静默的线上事故就已经发生了。

在连续批处理（continuous batching）引擎里，每一步送进模型的并不是规整的 `[batch, seq, hidden]` 三维立方体，而是一张把所有活跃请求首尾相接拍扁的二维张量 `[tokens, hidden]`（见[《你其实不需要读 `attn_metadata`》](01-attn-metadata.md)）。第 0 维上并排坐着的，是同一毫秒恰好被调度器拼进同一个批次的不同用户。当你在 `dim=0` 上求均值时，张三收到的回答，就开始取决于这一步李四和王五恰好向模型提了什么问题。没有崩溃堆栈，没有显存越界，生成的文本甚至看起来依然通顺——只是输出结果随着线上并发邻居的变化而悄无声息地漂移。

这是往推理引擎里挂载自定义代码时最让人头疼的现实：**最致命的 bug 从来不报错。**

除了跨行串味，还有两类同样安静的陷阱。第一类是调试在线学习算法时顺手写下的防御性断言：

```python
# check-docs: skip: a one-line method body, not a complete program
assert not torch.isnan(loss).any()
```

为了算出这个 Python 的 `bool` 值，CPU 必须立刻停下脚步，等待 GPU 把队列里积压的所有内核全部跑空，再把一个字节的结果跨过 PCIe 总线传回主机内存。这叫作一次主机同步（host synchronization）。在异步流水线里，CPU 本来领先 GPU 半步在准备下一步的元数据；这一下急刹车直接把流水线排空。更糟的是，稳态解码步是被录成 CUDA Graph 反复回放的——回放时 Python 解释器根本不在场，这行断言要么让图捕获当场报错，要么只在预热录制那一步跑过一次，之后彻底沦为摆设（见[《What Color is Your Tensor?》](02-tensor-colors.md)）。

第二类陷阱藏在显存压力之下：当 GPU 的 KV Cache 被长文本撑满时，vLLM 的调度器会临时抢占（preempt）一个正在生成的请求，释放它的物理块；等过几步显存宽裕了，再把这个请求的提示词连同已生成的 token 从头重算一遍。如果一个维护着每请求状态的在线训练算法不知道“这是一次重放”，它就会把同一段历史文本再训练第二遍，把优化器动量和权重悄悄训歪。

当我们把外部用户编写的扩展程序放进一个每步只有几毫秒到二十来毫秒、同时服务上百个租户的推理引擎时，怎么敢信任它？

在系统软件领域，人们很容易想到 Linux 内核的 **eBPF**：用户程序在装入内核之前，先过一道静态验证器（verifier），从数学上证明它不会死循环、不会越界访问。那么，面对一段研究者用 PyTorch 写的任意张量函数，我们能不能也写一个静态验证器？

先把最诚实的底牌亮在桌面上：**对一段任意的 PyTorch 代码，我们证明不了任何有意思的语义性质。**

PyTorch 不是只有十几个寄存器的受限字节码，背后也没有一个能穷举所有张量索引与广播路径的抽象解释器。如果有人声称能用静态分析证明一段任意的 PyTorch 方法“绝不会让第 $i$ 行的输出受第 $j$ 行输入的影响”，他一定是在哄你。

既然静态证明走不通，而在几毫秒一步的 GPU 热路径上插运行时检查又会拖垮吞吐，我们还能怎么办？

退后一步，把前面那三个不报错的幽灵 bug 摆在一起看。主机同步属于结构问题，下一节的静态关口就能拦住；剩下的跨请求串味、抢占重复训练，再加上同属一类的分块切分敏感，这几个看似风马牛不相及的工程事故，破坏的其实是**同一条不变式（Invariant）**：

> **任何一个请求的最终结果，都只应该取决于它自己的输入历史，而对底层引擎“如何调度它、和谁拼在一起跑”保持绝对不变。**

一旦这样重新定义了问题，解法就从“程序证明”变成了“差分测试”。我们不需要知道一段新算法在数学上的标准答案是多少，我们只需要在把程序装进 GPU **之前**，用三道关口在纯 CPU 上把它筛一遍：

```text
User Workload (Python class)
  │
  ├─► [Gate 1: Declarative Signature]   Freeze Python state; explicit input & state ports
  │
  ├─► [Gate 2: Meta Trace & DAG Plan]   1x Meta-Tensor Trace ──► Shapes + Lifetimes + 0 Host Syncs
  │                                     Rust Graph Planner   ──► Phase Windows + SSA Write Hazards
  │
  └─► [Gate 3: Schedule as Data]        Real Planner / Executor / Row-Labeler on CPU
                                        Perturb neighbors (salt = 1_000_000) across 3-tuple schedules
```

前两道关用最小代价拦截结构与同步错误；而第三道关——也是本文真正想分享的核心方法——**把连续批处理的“调度”本身剥离成可以任意摆布的数据，在 CPU 上给邻居请求加盐下毒，用几秒钟的差分运行把所有跨行污染与重放错乱逼出原形。**

---

## 第一道关：把输入端口明牌，为“无模型测试”铺路

为什么在一般的深度学习插件系统里，测试一段钩子代码非得启动整座大模型不可？

因为传统插件给你的是一个 `def on_step(engine, batch):` 回调，把半个引擎的内部对象直接裸露给你。你在函数体里顺着对象属性摸到了哪一层激活、改了哪个 Python 列表、什么时候偷偷调了一次 `.item()`，外层框架完全无从知晓。要测它，就只能把十几 GB 的模型权重真正加载进显存。

在 pLLM 中，我们用声明式签名替掉了开放式回调。一个方法需要读什么模型激活、读写哪份状态、在什么时机生效，全部写在函数参数的默认值与装饰器上。以在线蒸馏负载 `Distill` 为例：

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

不用运行 `loss` 的函数体，框架只扫一眼签名就知道：它只从模型读取第 0 层和最后一层的隐藏状态（`pllm.hidden`），按请求对齐 `distiller` 参数模块，并在采样落定后触发一次反向传播与优化器更新。同时，这个例子把类声明成冻结的 dataclass（`frozen=True`），`self.step += 1` 这类在 CUDA Graph 回放时会失效的写法会直接报错；pLLM 在入口处签名方法代码时，也会拒绝 `setattr()`、`globals()` 这类藏起可变状态的写法——跨步演化的状态只该放在受管的状态池里。

这层设计本身只是规规矩矩的软件工程：Python 的动态反射被锁死在冷启动入口（`pllm/usage.py` 与 `pllm/workload.py`），一旦跨过入口，所有方法就凝固成了带类型的 `TensorDeclaration` 与 `SemanticGraph`。

但正是这个显式的端口契约，买来了后面所有脱离 GPU 测试的前提：**既然方法只通过声明好的端口读取 `pllm.hidden(0)`，我们在测试时就根本不需要加载真正的 Transformer 权重——只要在 CPU 上按契约造出形状一致的合成张量喂进端口，就能让整套用户代码和框架调度器真刀真枪地跑起来。**

---

## 第二道关：一次 `meta` 张量追踪一石四鸟，外加静态时序检查

拿到语义图之后，程序在见到真数据之前，会先在 CPU 上过两台轻量级扫描仪。

第一台扫描仪是 PyTorch 的 **`meta` 张量追踪**。如果你不熟悉 `meta` 设备（或 FakeTensor），可以把它理解成一张“只有长宽高和数据类型标签、但没有分配哪怕一字节物理内存”的空头支票。当你把形状为 `[128, 3584]` 的 `meta` 张量喂进上面的 `loss` 方法时，PyTorch 的算子分发器会照常推导每一步矩阵乘与加法的输出形状。

通常人们只拿它做维度校验。但实际上，在同一次几乎零成本的 CPU 空跑里，我们可以顺手捞回四样东西：
1. **端口形状与类型校验**：比如采样引导方法返回的到底是不是 `[rows, hidden_size]`；
2. **精确的显存生命周期**：记录每个中间张量在哪一个算子诞生、在哪一个算子之后引用归零，直接算出工作区峰值；
3. **计算与访存量估算**：通过 `TorchDispatchMode` 拦截算子，统计 FLOPs、读写字节与内核数；
4. **一个免费的“主机同步检测器”**。

为什么形状追踪能抓主机同步？想想看，无论是 `.item()`、`.tolist()`，还是刚才那句 `assert not torch.isnan(loss).any()`，又或是 `if h.norm() > 10.0:`，它们在底层都要求 CPU 去读取张量里的**真实数值**。而 `meta` 张量里根本没有数值！只要代码里藏着哪怕一处隐式同步或依赖张量数值的 Python 分支，追踪就会当场触发 PyTorch 的符号异常。

我们在 `pllm/execution/torch_private.py` 里唯一多做的，是把这个晦涩的内部异常拦下来，翻译成直接告诉用户怎么改代码的报错信息：

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

过了 `meta` 追踪，语义图进入 `rust/pllm_core/src/planning/` 下的 Rust 规划器。这里跑的是标准的编译期图检查：
* **确定性环检测**：用基于有序 `BTreeSet` 的拓扑排序检查任务依赖，一旦成环，明确列出互相等待的方法名；
* **阶段窗口检查**：引擎的每一步被划分为 **8 个严格有序的时间锚点**（从 `request.prepare` 到 `next.prepare`，详见[《You Probably Don't Need a Compiler IR》](09-stream-scheduling.md)）。如果一个挂在采样前的方法试图读取采样后才诞生的 `pllm.sampled_tokens()`，窗口传播会发现最早可执行锚点越过了最晚锚点（`lo > hi`），直接报错指出：`a task reads a value (sampled tokens, hidden states or another task's result) that exists only after its phase ends; declare it in a later phase`；
* **状态的 SSA 写写冲突**：在[《给 GPU 显存加上 MVCC》](03-gpu-mvcc.md)中我们讲过，我们的 **PTTS**（Paged Tensor Transition System，分页张量状态迁移系统——注意它开头的 P 是 Paged 分页与页表，与 pLLM 里代表 programmable 的 p 没有任何关系）把每次状态更新都看作从旧版本派生新版本的静态单赋值（SSA）迁移。如果两个方法同时声明从同一个源版本出发去写同一份状态，运行时谁先谁后就成了并发竞态，规划器会在编译期直接通过 `version_hazards.rs` 拒之门外：

```rust
            if let Some((other, _)) = commits.insert(origin(effect), (writer, commit)) {
                return Err(format!(
                    "{} and {} both write state {:?} from the same version; let one method write it",
                    graph.tasks[other].symbol, graph.tasks[writer].symbol, resource.resource
                ));
            }
```

这些静态检查很管用，能挡住绝大多数结构性失误。

但是，回头看看文章开头那行真正的罪魁祸首：

```python
# check-docs: skip: a one-line method body, not a complete program
return h - h.mean(dim=0, keepdim=True)
```

它能大摇大摆地走过前两道关。它的输出形状完全合法（因为写了 `keepdim=True`），没有半点主机同步，也没有任何依赖环或版本冲突。静态检查对它束手无策，因为它错在**多租户的运行时语义**上。

---

## 第三道关：给邻居的随机种子加一百万盐

怎么在不跑真实大模型、也不写复杂符号证明器的前提下，自动抓出 `h.mean(dim=0)` 这类跨行污染？

如果我们换一个黑盒视角来问：**“第 1 号请求没有偷看同批其他请求的数据”，在输入与输出的关系上到底意味着什么？**

它意味着一个极其朴素的等价关系（软件测试文献里常把这类测试称为蜕变测试，Metamorphic Testing）：**如果我保持第 1 号请求自己的输入比特纹丝不动，而把同一批次里除它以外所有其他请求的输入全部换成另一组随机数，第 1 号请求算出来的输出应该连一个比特都不变。**

同理，所有的多租户与连续批处理正确性，都可以写成这类“变换调度、结果不变”的等价法则：
* **行置换不变（Row Permutation）**：请求 `0, 1, 2` 各自单独跑，和拼成一批按 `0-1-2` 跑、或者倒过来按 `2-1-0` 跑，每个请求在每个 token 位置上的输出与最终状态必须完全一致；
* **租户隔离（Request Isolation）**：涂改同批邻居的输入，目标请求的输出必须不变；
* **分块与混排不变（Chunking & Mixing）**：一条长提示词无论是一口气算完预填充，还是被切成几块、跟其他正在解码（decode）的请求混在同一批里交错算完，结果必须一样；
* **抢占重放不变（Preemption Replay）**：请求算到一半被驱逐，随后无论是整段重放还是分块重放历史，最终状态与输出必须和从未被抢占时完全一致；
* **组件独立与中性等价**：组合两个 workload 时各自结果与单跑逐位相同；把干预强度（如 `beta=0`）关掉时退化回原生模型。

`pllm.testing.check` 干的就是这件事。它只读取模型的 `config.json`（拿到 `hidden_size`、层数等几个整数），在纯 CPU 上花几秒钟把你的程序放进上面这些故意刁难的平行宇宙里各跑一遍，然后逐个位置比对输出。

要把这件事做进几秒钟里，全靠两个小巧的工程设计。

### 1. 一行 `salt = 1_000_000` 的污点追踪

首先，既然不加载模型权重，喂给用户方法的合成激活值 `pllm.hidden(0)` 怎么造？

如果随手调 `torch.randn(batch_rows, width)` 生成整批随机数，那么“单独跑请求 1”和“三个请求合起来跑”时，随机数生成器的步进偏移就变了——请求 1 在两次运行里收到的输入本身就不一样，根本没法比对输出。

因此，合成激活值必须是**关于 `（请求编号 index, token 在该请求内的位置 offset, 特征通道 channel）` 的确定性纯函数**。来看看 `pllm/testing_contracts.py` 里的 `_token_values`，一共只有寥寥数行：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
def _random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _token_values(index: int, offsets: range, width: int, channel: int, perturbed: int | None) -> torch.Tensor:
    salt = 1_000_000 if perturbed is not None and index != perturbed else 0
    return torch.stack(tuple(_random((width,), salt + 10000 * index + 100 * offset + channel) for offset in offsets))
```

先看最后一行里的种子算术：`10000 * index + 100 * offset + channel`。无论第 `index` 号请求在当前步被排在扁平张量的第 0 行还是第 5 行，也无论它的提示词被切成了几块，只要是这个请求的第 `offset` 个位置，它生成的随机向量就永远一模一样。这就为所有的跨调度比对打下了地基。

现在，盯住上面那一行：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
salt = 1_000_000 if perturbed is not None and index != perturbed else 0
```

这就是整个“多租户隔离验证”的全部机关。

当我们想检验第 1 号请求（设 `perturbed = 1`）有没有被邻居污染时，我们把 `0, 1, 2` 三个请求放在同一个批次里跑两遍：

```text
Run 1 (Baseline, perturbed = None):
  Batch dim 0:  [ Req 0 (seed: 0xxxx) ]  [ Req 1 (seed: 1xxxx) ]  [ Req 2 (seed: 2xxxx) ]
                           │                        │                        │
                           ▼                        ▼                        ▼
                        Out 0                    Out 1                    Out 2

Run 2 (Perturb neighbors of Req 1, perturbed = 1):
  Batch dim 0:  [ Req 0 (seed: +1,000,000) ] [ Req 1 (seed: 1xxxx) ] [ Req 2 (seed: +1,000,000) ]
                           │                        │                        │
                           ▼                        ▼                        ▼
                        Out 0'                   Out 1'                   Out 2'
                                                    ║
                                  Must match Out 1! (Else: Row Mixing!)
```

第一遍（`perturbed = None`），所有请求用正常种子跑，记下请求 1 的输出 `Out 1`。
第二遍（`perturbed = 1`），凡是 `index != 1` 的邻居请求，种子瞬间被加上了一百万（`1_000_000`），它们的输入张量变得面目全非；唯独请求 1 自己，因为 `index == perturbed`，`salt` 依然是 `0`，它收到的每一个输入比特都和第一遍分毫不差。

这时候再去比对 `Out 1` 和 `Out 1'`。哪怕用户在代码最深处的某个角落写了 `h.mean(dim=0)`、`F.softmax(x, dim=0)` 或是跨行索引切片，邻居那一百万盐带来的数值剧变就会顺着第 0 维传染给请求 1，被比对器当场抓获。

更进一步，一旦抓到 `Out 1` 变了，到底是用户写的某个方法把行混在了一起，还是底层的框架自己在路由行号时出了 bug？因为测试器在每个方法调用前后都记录了按请求拆开的输入与输出切片，`_fault` 函数只需做一次极其简单的比对就能精准定责（`pllm/testing_contracts.py`）：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
if changed and equal_inputs:
    return (
        f"method `{key[2]}` mixes rows: equal inputs for this request produced different outputs. "
        "Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0)."
    )
if changed:
    return (
        f"pLLM misrouted rows before `{key[2]}`: its per-request inputs already differ. "
        "Report this message as a framework bug."
    )
```

如果进入这个方法之前，请求 1 的输入还一模一样（`equal_inputs`），出来之后却变了——那是你的方法把不同请求混在了一起，报错信息会直接指出方法名，并提醒你把 `dim=0` 改成 `dim=-1`；反之，如果进方法之前输入就已经不一样了——那是框架自己的路由 bug。

### 2. 三元组“调度即数据”：为什么不需要写引擎模拟器？

做系统级测试最怕的一件事，就是为了在 CPU 上测 GPU 的逻辑，专门去写一套“模拟调度器（Mock Scheduler）”。一旦测试走的是模拟器、线上走的是真引擎，你测出来的就只是模拟器自己的逻辑，生产代码里的边界 bug 照样漏网。

为什么 `pllm.testing.check` 不需要写模拟器？

因为在[《你其实不需要读 `attn_metadata`》](01-attn-metadata.md)里，我们已经把 vLLM 调度器在每一步传达给扩展层的排班信息，提炼成了每个请求最基础的**三个整数**：`（请求编号, 本步调度几个 token, 在本步之前已经算过几个 token）`（提示词总长在请求准入时已知）。

既然生产环境里的 Rust 行标签器（`RowLabeler`）只认这三个整数，那么在测试里，**“连续批处理的复杂调度”本身就退化成了一串写死在代码里的整数三元组！**

来看看 `pllm/testing_contracts.py` 里测试“分块预填充与解码混排”的那张调度表。测试里固定有三个请求：请求 `0`（提示词长 5）、请求 `1`（提示词长 1）、请求 `2`（提示词长 9）。代码里的 `_schedule` 是这样写的：

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

把前四步画在时间轴上，就能看清这几行数字里埋了多少刁钻的边界条件：

```text
Requests:  Req 0 (prompt = 5)      Req 1 (prompt = 1)      Req 2 (prompt = 9)

Step 0:    (0, 4, 0) [prefill 0..4] (1, 1, 0) [prefill end!]  -- not arrived --
           └───────► Packed batch order: [ Req 0 (4 rows), Req 1 (1 row) ]

Step 1:    (2, 4, 0) [prefill 0..4] (1, 1, 1) [decode #1]   (0, 1, 4) [prefill end!]
           └───────► Packed batch order: [ Req 2 (4 rows), Req 1 (1 row), Req 0 (1 row) ]

Step 2:    (0, 1, 5) [decode #1]    (1, 1, 2) [decode #2]   (2, 4, 4) [prefill 4..8]
           └───────► Packed batch order: [ Req 0 (1 row),  Req 1 (1 row), Req 2 (4 rows) ]

Step 3:    (2, 1, 8) [prefill end!] (0, 1, 6) [decode #2]   (1, 1, 3) [decode #3]
           └───────► Packed batch order: [ Req 2 (1 row),  Req 0 (1 row), Req 1 (1 row) ]
```

你看这四步里发生了什么：
* **Step 0**：请求 0 的 5 个提示词被切了一刀，这一步只算前 4 个 `(0, 4, 0)`——还没算完，所以声明了 `tokens="prefill"`（提示词结束时触发）的方法在这一步绝不能为请求 0 触发；而请求 1 只有 1 个词 `(1, 1, 0)`，当场完成预填充。
* **Step 1**：长提示词的请求 2 进场了，先算前 4 个词 `(2, 4, 0)`；请求 1 已经迈入解码阶段 `(1, 1, 1)`；请求 0 终于补完了它提示词的第 5 个词 `(0, 1, 4)`——而且注意，这一步三个请求在扁平张量里的行顺序被故意洗成了 `2 -> 1 -> 0`。
* **Step 2 & 3**：请求 0 和 1 继续解码，而请求 2 的 9 个提示词足足跨了三步（`4 + 4 + 1`）才在 Step 3 的 `(2, 1, 8)` 完成预填充，同时每一步的行排列顺序都在颠倒洗牌。
* 而在测试抢占重放的分支里（用的是不混排的那张调度表），调度表会在请求推进到半途时，突然插入一段 `computed` 归零的三元组（例如请求 1 本该走 `(1, 1, 3)` 的那一步变成整段重算的 `(1, 4, 0)`，或拆成 `(1, 1, 0)`、`(1, 1, 1)`、`(1, 2, 2)` 分块重算），模拟请求被踢出显存后重新预填充的全过程。

这几串静态三元组被直接喂给生产环境的同一个 Rust 行标签器入口（`label_rows`）、同一个规划器和同一个状态执行器（唯一的细微差别是：抢占重放时过滤已消费历史位置的 `retained_rows` 游标，在 CPU 检查器里走了一个几行的 Python 参考实现，而线上走原生标签器）。没有模拟器，测试跑的就是真管道本身。

---

## 诚实的边界，以及真卡上的最后一道防线

把这套 CPU 契约检查吹得再天花乱坠，我们也必须给它泼一盆冷水，把它的盲区明明白白地摆在台面上。

首先，**差分测试不是数学证明**。`pllm.testing.check` 只是在 3 个请求、提示词长度为 `(5, 1, 9)` 的几套固定调度表上跑过了等价性比对。它不是随机生成上万种拓扑的模糊测试（Fuzzing）——如果你的代码里写了一个极其古怪的条件，非得在并发请求数超过 10 个、或者序列长度恰好满 1,024 时才触发跨行操作，这几套三元组是抓不住它的。

其次，**温和的正态随机数试不出数值病态**。`_token_values` 生成的是标准正态分布 `torch.randn`，非常适合抓“行混了”、“位置偏了”、“重放多训了一次”这类结构与路由错误；但如果你的算法在遇到真实大模型里的极端稀疏激活、接近 0 的分母或极大 logits 时会算出 `NaN`，正态合成数据是试不出来的（数值分布偏移需要靠 `pllm.testing.check_distribution` 在真模型上测量）。

最重要的是第三点：**凡是深入模型层内部的手术，无模型的 CPU 检查器根本跑不了。**

如果你的 workload 用到了残差流改写（`@pllm.custom_residual`）、参数覆盖层（`Adapter`）、自定义注意力或自适应跳层（见第 [04](04-compiled-model-hooks.md)、[05](05-early-exit.md)、[06](06-sparse-attention-kv.md) 篇），这些钩子是直接嵌在 vLLM 的 Transformer 层与 C++ 注意力算子中间执行的。CPU 上既然没有加载真模型，自然也就没有这些层。遇到这类组件，CPU 检查器绝不会不懂装懂地打上 `PASS`，而是会在报告里印上一行醒目的 `SKIP`：

```text
SKIP    model-site execution: CPU decoder does not execute residual hooks, overlays, spans or PositionState
```

**跳过就是跳过，绝不伪装成通过。**

凡是被 CPU 标为 `SKIP` 的模型内钩子，以及只有真实硬件才能检验的物理性质，必须交给 `pllm.testing.check_on_gpu(workload, config)` 在真实的 vLLM 引擎上补齐。在真卡上，除了把上面的单跑 vs 批跑、行逆序、分块混排和中性等价（关掉干预时与原生 vLLM 对比：模型内核相同时逐比特比对，用了 Inductor 融合的钩子则比较 teacher-forced 的全词表分布）实打实地跑在模型上之外，还有两件小事值得单拎出来说说。

第一件是**稳态解码的零同步抓捕**。`SynchronizationCheck` 会在每个批次宽度跑过两步预热后，把所有的锚点入口和模型内钩子包进 PyTorch 的 `torch.cuda.set_sync_debug_mode("error")` 上下文里——只要稳态解码期间有任何隐式的主机等待（如 `.item()` 或同步拷贝），当场抛异常报错。

但这里有一个很多人不知道的工具盲区：**PyTorch 的 `set_sync_debug_mode` 只能抓住拷贝和张量取值触发的隐式同步，却完全看不见代码里显式调用的 `Event.synchronize()` 或 `Stream.synchronize()`！** 为了确认框架在热路径上真的做到了零主机等待，我们在单张 RTX 4090（24 GiB GDDR6X）上跑 Qwen2.5-0.5B 的 ESamp 负载（vLLM 0.27.1，8 条提示词 × `n=16` 共 128 个并发序列，生成 256 个 token）时，专门对底层调用栈做过一次完整审计：整场生成跑下来一共发生了 128 次主机等待，**100% 来自 vLLM 自己取回异步输出的 `get_output`**，而扩展层自己产生的 Event 或 Stream 等待是严格的 **0 次**。调试开关可以当日常护栏，但真正的“零同步”得靠调用栈审计来兜底。

第二件是**如何在真卡上稳定触发一次真实的 KV 抢占**。在 CPU 测试里，抢占只是我们在三元组列表里人工塞入的一段重放；到了 GPU 上，我们要检验 vLLM 的真实调度器在显存见底踢人时，会不会和我们的状态版本管理配合失误。

怎么让一个正常的短测试百分百撞上一次真实的 KV 驱逐？去改 vLLM 的调度器源码太重了，而把显卡显存故意塞到只剩几兆又极不稳定。来看看 `pllm/testing_gpu.py` 里的 `_Eviction` 是怎么用十行代码四两拨千斤的：

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

盯住那个 `return None`。我们轻轻包了一层 vLLM 真实的 `kv_cache_manager.allocate_slots`。当第一个已经生成了至少 4 个 token 的请求向分配器伸手要新的 KV 块时，包装函数故意对它两手一摊返回 `None`（谎称“显存块用光了”），随后立刻恢复正常分配。

vLLM 的调度器收到这一个 `None`，就会一五一十地走完它线上的真实抢占路径：把这个倒霉的请求踢出运行队列、释放它已占有的全部 KV 块、根据我们的抢占策略保留或重置它的 PTTS 状态版本，等后续步再把它排回来从第 0 个词重算。跑完之后，测试器比对这个被踢过的请求最终生成的 token 序列和终态参数，验证它与从未被驱逐的基准运行分毫不差。一行 `return None`，不改宿主一行源码，就把整条真实的容错恢复链路彻底验了一遍。

---

## 结语：把调度变成数据

回顾这三道关口，最值得从推理引擎带去其他多租户系统（无论是数据库、流处理引擎还是在线特征服务）的，其实就是那个简单的念头：**把调度变成数据（Schedule as Data）。**

在任何支持动态批处理与资源抢占的系统里，多租户隔离与调度无关性永远是最难测的——因为 bug 不在单条请求的数学逻辑里，而在不同请求交错拼装的缝隙里。如果在设计系统内部边界时，能把“调度器在每一步做了什么决定”剥离成一个极简的纯数据结构（就像那三个整数组成的三元组），让下游的状态管理、行路由和算子执行全都只认这串数据，两件好事就会同时发生：

第一，你再也不需要维护一套会撒谎的 Mock 模拟器，几行手写的三元组列表就能在 CPU 上驱动真实的生产执行管道；
第二，原本虚无缥缈的“多租户隔离”属性，瞬间变成了一个可以在几秒钟内反复执行的差分实验：给邻居的随机种子加上一百万盐，然后看看自己变没变。
