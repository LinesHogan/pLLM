# 在 serving 引擎里信任一段用户程序

如果你在 Hugging Face 的单请求脚本里写一个“激活中心化”的钩子，你大概会随手写下这一行代码：

```python
# check-docs: skip: a one-line method body, not a complete program
return h - h.mean(dim=0, keepdim=True)
```

在单请求调试时，张量的第 0 维长度永远是 1。这行代码不报错、不崩溃，梯度能正常回传，单元测试一片绿灯。但是，一旦你把这行代码原封不动地搬进生产级的 vLLM 引擎里，灾难就会在毫无察觉中发生。

在连续批处理（continuous batching）引擎里，每一步进入模型的并不是整整齐齐的 `[batch, seq, hidden]` 立方体，而是一张把所有活跃请求首尾相接拍扁的二维张量 `[tokens, hidden]`（见[《行模型》](04-packed-rows.md)）。第 0 维上排着的，是同一步里恰好拼在一起的不同租户。当你在 `dim=0` 上求均值时，张三得到的回答，就开始取决于这一毫秒李四和王五恰好问了什么问题。没有异常堆栈，没有显存越界，生成的文本甚至看起来依然通顺——只是结果随着线上负载的波动悄无声息地漂移。

这还只是第一种死法。再看第二种：你在调试在线学习算法时，出于好习惯加了一行防御性断言：

```python
# check-docs: skip: a one-line method body, not a complete program
assert not torch.isnan(loss).any()
```

为了算出这个 Python 的 `bool`，CPU 必须立刻刹住车，等待 GPU 把队列里排着的所有算子全部跑完，再把一个字节的结果跨过 PCIe 总线拷回主机内存。这叫作一次**主机同步（host synchronization）**。在异步流水线里，CPU 本来领先 GPU 一步在紧锣密鼓地准备下一步的元数据；这一下急刹车，直接把流水线排空。更要命的是，稳态 decode 步是被录成 CUDA graph 反复回放的——回放时 Python 解释器根本不在场，这行断言要么让图录制当场炸裂，要么只在录制那一步跑过一次，之后彻底沦为摆设（见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。

还有第三种更隐蔽的死法：当 GPU 的 KV cache 被长文本撑满时，vLLM 的调度器会不动声色地抢占（preempt）一个正在生成的请求，释放它的物理块；等过几步显存宽裕了，再把这个请求的 prompt 连同已经生成的 token 从头重算一遍。如果一个维护着每请求状态的在线训练算法不知道“这是一次重放”，它就会傻乎乎地把同一段历史再训练一遍，把优化器动量和权重彻底算歪。

> **本文实验基线**：如无特别说明，文中的性能、显存与规划耗时数据均在单张 NVIDIA RTX 4090（24 GiB）、`Qwen/Qwen2.5-7B-Instruct`（BF16）、vLLM 0.32.0、128 并发序列（8 条 prompt × `n=16`）、生成 256 个 token、开启 CUDA graph 的标准设置下测得；小模型 `Qwen2.5-0.5B` 仅在展示固定开销占比的对照处显式标出。

在这个系列的总起篇里，我们曾拿 Linux 内核里的 **eBPF** 来类比 pLLM：允许外部用户写一小段带状态的程序，安全、高速地运行在底层引擎的关键路径上。前面的文章讲的都是怎么让这段程序“跑得和原生一样快”。而在这篇收官之作里，我想正面回答那个被留到最后、也最让人睡不着觉的问题：

**eBPF 之所以敢让用户代码进内核，是因为装载前有一个静态验证器（verifier）从数学上证明它不会搞垮内核；那么，面对一段研究者随手写的 PyTorch 方法，我们凭什么敢把它放进一个每步只有二十来毫秒、同时服务上百个租户的推理引擎里？**

先把最诚实的底牌亮在桌面上，免得你把 eBPF 的类比读得太满：**对一段任意的 PyTorch 代码，我们证明不了任何有意思的语义性质。**

PyTorch 不是受限的 64 位寄存器字节码，背后也没有一个可以穷举所有执行路径的抽象解释器。如果有人声称能用形式化方法证明一段任意的张量计算“绝不会让第 $i$ 行的输出污染第 $j$ 行”，他一定是在骗你。

那么我们能做什么？

退后一步，把前面那三个不报错的幽灵 bug 摆在一起看。你会突然发现，跨请求串行、分块切分错误、抢占重复训练，这三个看似风马牛不相及的工程事故，在数学上其实违背的是**同一条不变式（Invariant）**：

**任何一个请求的最终结果，都只应该取决于它自己的输入历史，而对底层引擎“如何调度它”保持绝对不变。**

一旦你这样重新定义了问题，通往解法的路就豁然开朗了。我们为什么非要去静态证明一段代码“对不对”？在运行时检查当然指望不上——7B 模型一步 decode 约 21.7 ms，0.5B 模型更是只有约 3.0 ms，在每一步里插检查探针要么触发同步、要么拖垮吞吐，何况许多 bug 只有在“请求 A 恰好看完第 4 个词、请求 B 刚好在 decode、请求 C 刚被抢占恢复”的刁钻交错下才会露头。

但如果我们在把程序装进 GPU **之前**，先在 CPU 上用三道漏斗把它筛一遍呢？

```text
User Workload (Python class)
  │
  ├─► [Gate 1: Declarative Cage]   Freeze Python state; reflect typed ports & effects
  │
  ├─► [Gate 2: Meta Trace & Plan]  1x Meta-Tensor Trace ──► Shapes + Memory Lifetimes
  │                                                     ──► 0 Host Syncs + FLOP/Bytes
  │                                Rust Graph Planner   ──► DAG Cycles + Phase Windows
  │                                                     ──► SSA State Version Hazards
  │
  └─► [Gate 3: Schedule as Data]   Real Planner/Executor/Row-Labeler on CPU
                                   Differential / Metamorphic runs over 3-tuple schedules
```

第一道关，把程序关进一个已知的声明式形状；第二道关，在规划期用一次虚拟张量追踪和图算法抓出所有结构与时序错误；第三道关——也是整套机制里真正称得上新东西的部分——**把“调度”本身变成可以随意捏造的数据，在 CPU 上对这几条核心语义不变式做蜕变测试（Metamorphic Testing）**。

让我们顺着这三道关往下走。

---

## 第一道关：先把程序关进一个形状

eBPF 验证器之所以能开展工作，前提是程序不能随便去捅内核里的任意内存地址，它只能调用白名单里的 helper 函数。

在一般的深度学习插件系统里，框架给你一个 `def on_step(engine, batch):` 钩子，把半个引擎的内部对象裸露给你。你在里面改了哪个列表、读了哪层激活、什么时候偷偷做了一次 `.item()`，框架完全是瞎子。

在 pLLM 里，我们的对应物是**声明式契约**。一个方法需要读什么，直接写在函数参数的默认值上；需要维护什么状态，写成类属性；对引擎产生什么副作用，写成装饰器。看看在线蒸馏负载 `Distill` 的写法（`doc/explain/performance.md`）：

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

不用运行 `loss` 的函数体，框架只扫一眼签名就知道：它需要从模型第 0 层和最后一层引出激活缓冲（`pllm.hidden`），需要按请求行对齐 `distiller` 的参数模块，并且会在采样结束后触发一次针对 `distiller` 的反向传播与优化器步。同时，跨步的东西只能声明成状态：pLLM 在编译时把负载连同它依赖的函数冻结成一份脱离作者环境的签名副本，想在方法里写 `self.step_count += 1` 来记步数是行不通的——在 CUDA graph 的世界里，Python 方法只在录制时跑一次，把跨步状态存在 Python 对象属性里是典型的自欺欺人。

老实说，这一层本身只是规规矩矩的软件工程。Python 的动态反射被严格锁死在极窄的冷启动入口（`pllm/usage.py` 与 `pllm/workload.py`）；一旦跨过这道边界，所有的 Python 函数与装饰器就彻底凝固成了带类型的 `TensorDeclaration` 和 `SemanticGraph`。后面的规划器、显存分析器和契约测试器，谁也不会再去碰动态的 Python 对象。

但正是这个略显死板的“牢笼”，买来了后面两道关成立的前提：**因为所有的输入端口都已在签名上明牌，我们才有可能在不加载十几 GB 模型权重的情况下，在 CPU 上凭空喂给它假的张量。**

---

## 第二道关：一次 Meta 追踪一石四鸟，外加规划期的结构铁律

拿到语义图之后，程序在真正见到 GPU 之前，要先过两台静态扫描仪：一台在 Python 里跑 PyTorch 的 `meta` 张量追踪，另一台在 Rust 里跑依赖与时序规划。

先说那个让我每次想起来都觉得划算的买卖：**`meta` 张量的一石四鸟**。

如果你不熟悉 PyTorch 的 `meta` 设备（或者 FakeTensor），可以把它想象成一张“只有长宽高和数据类型标签、但没有分配哪怕一字节物理内存”的空支票。当你把形状为 `[128, 3584]` 的 `meta` 张量喂进用户的 `loss` 方法时，PyTorch 的算子分发器会照常推导每一步矩阵乘和加法的输出形状。

在常规印象里，这只是用来检查维度匹不匹配的。但在 pLLM 里，我们在同一次 CPU 追踪里顺手捞回了四样东西：
1. **端口形状校验**：比如 `@pllm.guide_sampling` 返回的到底是不是 `[rows, hidden_size]`；
2. **精确的显存生命周期**：每个中间张量在哪一个算子诞生、在哪一个算子之后引用归零，直接算出工作区峰值；
3. **FLOP、读写字节与算子计数**：通过 `TorchDispatchMode` 拦截算子，估算这一步的计算量与访存量；
4. **一个零成本的“主机同步检测器”**！

为什么形状追踪能抓主机同步？想想看，`.item()`、`.tolist()`，或者对张量的值写一句 `if h.norm() > 10.0:`，它们在底层都需要去读取张量的**真实数值**。而 `meta` 张量里根本没有数值！只要用户的代码里藏着哪怕一处隐式同步或数据依赖的 Python 分支，追踪就会当场撞墙。

我们唯一多做的一步，是把 PyTorch 内部晦涩的符号追踪异常拦下来，翻译成人类能看懂并直接照着改的报错建议（`pllm/execution/torch_private.py`）：

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

过了 `meta` 追踪，语义图被送进 `rust/pllm_core/src/planning/` 下的 Rust 规划器。这里跑的都是编译原理课本上的经典图算法，我不打算贴几十行标准拓扑排序的代码来耽误你的时间，只点出三个专门针对推理引擎的检查设计：

第一，**确定性的环检测**。在 `scheduler.rs` 的 Kahn 拓扑排序里，就绪队列用的是有序的 `BTreeSet` 而不是哈希集合，每次总是 `pop_first` 取出编号最小的任务——这保证了同一份声明无论在哪台机器上编译，排出的顺序永远确定。一旦发现入度没消完的环，它不会冷冰冰地甩一句 `"cycle detected"`，而是把还在互相等待的方法名全揪出来：`"methods ... wait on each other through results or state reads and writes; remove one of the dependencies in the cycle"`。

第二，**防止“穿越到未来”读数据**。引擎的一步被切成了 7 个严格有序的锚点（见[《自动放置与重叠》](11-automatic-placement-overlap.md)）。如果你写了一个方法挂在采样器之前（比如 `@pllm.add_logits`），却在参数里声明要读这一步采样完才诞生的 `pllm.sampled_tokens()`，规划器在沿着依赖边收紧每个任务的 `[最早锚点, 最晚锚点]` 区间时，就会发现下界超过了上界（`scheduler.rs`）：

```rust
        if lo.iter().zip(&hi).any(|(a, b)| a > b) {
            return Err(
                "a task reads a value (sampled tokens, hidden states or another task's \
                 result) that exists only after its phase ends; declare it in a later phase"
                    .into(),
            );
        }
```

第三，**状态的 SSA 写写冲突**。在[《PTTS》](01-ptts.md)里我们讲过，pLLM 把每份状态都看作静态单赋值（SSA）形式的版本链：每次写入都是从一个旧版本派生出一个新版本。如果两个方法同时声明从同一个源版本出发去写同一份状态，那到底谁的更新算数，就纯粹取决于运行时两条 CUDA stream 谁跑得更快。这种看天吃饭的并发竞争，在编译期就会被 `version_hazards.rs` 直接枪毙：

```rust
            if let Some((other, _)) = commits.insert(origin(effect), (writer, commit)) {
                return Err(format!(
                    "{} and {} both write state {:?} from the same version; let one method write it",
                    graph.tasks[other].symbol, graph.tasks[writer].symbol, resource.resource
                ));
            }
```

请注意这几条报错信息的共同癖好：它们从不只说“什么坏了”，最后一句话永远是祈使句——**告诉你该怎么改代码**。

---

## 第三道关：把调度变成数据，改掉别人看看自己

到目前为止，我们拦住了可变 Python 属性，拦住了 `.item()` 同步，拦住了依赖死锁、时间穿越和并发写冲突。

但是，回头看看文章开头那个最阴险的罪魁祸首：

```python
# check-docs: skip: a one-line method body, not a complete program
return h - h.mean(dim=0, keepdim=True)
```

它能大摇大摆地走过前两道关！它的输出形状完全合法（因为 `keepdim=True`），没有半点主机同步，也没有任何依赖环。静态检查对它束手无策，因为它错在**多租户语义**上。

怎么在不跑真实大模型、也不写复杂的张量符号证明器的前提下，抓出这类语义 bug？

测试领域有一个老派而优雅的思想，叫**蜕变测试（Metamorphic Testing）**。当你测试一个复杂的数值函数时，你往往不知道对于某个输入 $x$，它的标准答案 $f(x)$ 到底是多少；但你非常清楚，当你对输入施加某种保持语义的变换 $T$ 时，输出必须满足特定的等价关系。

放在多租户推理引擎里，我们对用户程序要求的全部语义正确性，恰恰就是一组关于“调度变换”的蜕变关系：
1. **行置换不变（Row Permutation）**：请求 0、1、2 三个各跑各的（单跑），和三个拼成一批按 `0-1-2` 跑、或者倒过来按 `2-1-0` 跑，每个请求在每个 token 位置上拿到的结果必须分毫不差；
2. **租户隔离（Request Isolation）**：同一批里，如果我把除你以外所有人的输入全部涂改掉，你自己的输出连一个比特都不准变；
3. **混排与分块不变（Mixed Prefill/Decode & Chunking）**：无论一条长 prompt 是一口气算完，还是被切成几块、跟别人的 decode 步混在一个 batch 里交错算完，结果必须一样；
4. **抢占重放不变（Preemption Replay）**：请求算到一半被驱逐，之后无论是全量重放历史还是分块重放历史，最终状态与输出必须和从未被抢占时完全一致；
5. **组件独立与中性等价（Component Independence & Neutrality）**：把两个 workload 组合在一起，各自的结果必须和单独运行时逐位相同；把干预强度（如 `beta=0`）关掉，必须退化回原生模型。

`pllm.testing.check` 干的就是这件事。它不加载权重，不占 GPU，只读取模型的 `config.json`，在 CPU 上用几秒钟把你的 workload 在上面这些刁钻的宇宙里各跑一遍，然后拿放大镜逐个位置比对输出。

这里有两个让我爱不释手的工程巧思。让我们把代码掀开，凑近了看。

### 巧思一：一行 `salt = 1_000_000`，给邻居下毒

首先，既然不加载模型权重，用户方法要读的 `pllm.hidden(0)` 从哪儿来？

如果随便调个 `torch.randn` 生成一批随机数，那“单独跑请求 1”和“三个请求合起来跑”时，随机数生成器的步进顺序就不一样了，请求 1 在两次运行里拿到的输入本身就不同，根本没法比对输出。

所以，合成的激活值必须是**由 `（请求编号, token 在该请求内的位置, 特征通道）` 唯一确定的纯函数**。来看看 `pllm/testing_contracts.py` 里生成测试激活的 `_token_values`，一共只有五六行：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
def _random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _token_values(index: int, offsets: range, width: int, channel: int, perturbed: int | None) -> torch.Tensor:
    salt = 1_000_000 if perturbed is not None and index != perturbed else 0
    return torch.stack(tuple(_random((width,), salt + 10000 * index + 100 * offset + channel) for offset in offsets))
```

先看最后一行里的种子算术：`10000 * index + 100 * offset + channel`。不管请求 `index` 在这一步被排在张量的第 0 行还是第 5 行，也不管它的 prompt 被切成了几块，只要是第 `index` 号请求的第 `offset` 个 token，它拿到的随机种子就永远锁死在同一个整数上！这就为所有的跨调度比对打下了地基。

现在，请把视线移到上面那一行：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
salt = 1_000_000 if perturbed is not None and index != perturbed else 0
```

这就是整个“多租户隔离验证”的全部机关。

当我们想检验“第 1 号请求（`perturbed = 1`）有没有偷看别人的数据”时，我们不需要去解析用户的 AST，也不需要追踪复杂的张量广播。我们只需要把 `0, 1, 2` 三个请求放在一起跑两遍：

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

第一遍（`perturbed = None`），大家用正常的种子跑，记下请求 1 的输出 `Out 1`。
第二遍（`perturbed = 1`），`index != 1` 的所有邻居请求，种子瞬间被加上了一百万（`1_000_000`），它们的输入张量变得面目全非；唯独请求 1 自己，因为 `index == perturbed`，`salt` 依然是 `0`，它收到的每一个输入比特都和第一遍分毫不差！

这时候再去比对 `Out 1` 和 `Out 1'`。哪怕用户的代码里在最深层的某个角落写了一句 `h.mean(dim=0)`、`F.softmax(x, dim=0)` 或是跨行切片，邻居那一百万盐带来的数值剧变就会立刻顺着第 0 维传染给请求 1，被比对器当场抓获。

更妙的是，一旦抓到不一致，究竟是用户的方法写错了，还是 pLLM 框架自己把行号路由错了？因为测试器记录了每个方法调用前后的按请求切片，`_fault` 函数只需做一次简单的归因判定（`pllm/testing_contracts.py`）：

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

如果进这个方法之前，请求 1 的输入还一模一样，出来之后却变了——那是你的方法把行混了，请把 `dim=0` 改成 `dim=-1`；如果进方法之前输入就已经不对了——那是我们框架自己的 bug，报错信息会直接请你去提 issue。

### 巧思二：三元组调度表——不写模拟器的秘密

做系统测试最怕的一件事，就是为了在 CPU 上测 GPU 的逻辑，专门写一套“模拟调度器（Mock Scheduler）”。一旦测试走的是模拟器、生产走的是真引擎，你测出来的就只是模拟器自己的逻辑，生产代码里的 bug 照样漏网。

为什么 `pllm.testing.check` 不需要写模拟器？

因为在[《行模型》](04-packed-rows.md)里，我们已经把 vLLM 调度器每一步告诉 pLLM 的信息，提炼成了每个请求最精简的**三个整数**：`（这一步算几个 token, prompt 有多长, 在这一步之前已经算完了几个 token）`。

既然生产环境里的 Rust 行标签器只吃这三个整数（`label_rows` 是它给 CPU 检查用的入口），而测试里三个请求的 prompt 长度又是固定的，那在测试里，**“调度”本身就退化成了一串写死在代码里的整数三元组！**

来看看 `pllm/testing_contracts.py` 里测试“分块 prefill 与 decode 混排”的那张调度表。我们有三个请求：请求 `0`（prompt 长 5）、请求 `1`（prompt 长 1）、请求 `2`（prompt 长 9）。代码里的 `_schedule` 是这样写的：

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

光看括号可能有点眼花，我们把前四步展开成时间轴上的画面，你就能一眼看出这短短几行元组里埋了多少机关：

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

你看：
* **Step 0**：请求 0 的 5 个 prompt 词被切了一刀，只算前 4 个 `(0, 4, 0)`（还没算完，所以 `tokens="prefill"` 的方法在这步绝不能触发！）；而请求 1 只有 1 个词 `(1, 1, 0)`，当场完成预填充。
* **Step 1**：长长的请求 2 杀进来了，先算前 4 个词 `(2, 4, 0)`；请求 1 已经进入 decode 阶段 `(1, 1, 1)`；请求 0 终于补完了它 prompt 的第 5 个词 `(0, 1, 4)`——而且注意，这一步三个请求在 batch 里的排列顺序故意洗成了 `2 -> 1 -> 0`！
* **Step 2 & 3**：请求 0 和 1 在继续 decode，而请求 2 的 9 个 prompt 词足足跨了三步（`4 + 4 + 1`）才在 Step 3 的 `(2, 1, 8)` 算完，同时每一步的行序都在颠倒洗牌。

把这几串三元组直接喂给生产环境用的那个 `label_rows`、同一个 Rust 规划器和同一个状态执行器，CPU 测试就走完了和线上 GPU 完全相同的打包、筛选、状态版本推进与路由代码！（唯一要诚实交代的例外是：抢占重放时标记“哪些历史 token 已经被消费过”的 `retained_rows` 游标过滤，CPU 检查器里用的是一个几行的 Python 参考实现，而生产环境走的是原生行标签器。）

当然，CPU 上没有加载真正的 Transformer 层。如果你的 workload 用到了深入模型骨髓的操作——比如改写残差流（`@pllm.custom_residual`）、参数覆盖层（`Adapter`）或自适应深度——CPU 契约检查器不会不懂装懂地打上 `PASS`，而是会在报告里清清楚楚地印上一行 `SKIP`：

```text
SKIP    model-site execution: CPU decoder does not execute residual hooks, overlays, spans or PositionState
```

**跳过就是跳过，绝不伪装成通过。** 这正是我们把验证拆成 CPU 与 GPU 两半的原因。

---

## 第四道关：上 GPU 真刀真枪补齐死角，再把账单摊在台面上

凡是 CPU 跳过的地方，以及只有真硬件才能检验的物理性质，全部交给 `pllm.testing.check_on_gpu(workload, config)` 在真实的 vLLM 引擎上补齐。

在 GPU 上，我们重点盯防三件事。

第一件是**稳态 decode 的零同步验证**。`SynchronizationCheck` 会在每个打包宽度跑过两步预热之后，把 pLLM 所有的锚点入口和模型挂载点都包进 PyTorch 自带的 `torch.cuda.set_sync_debug_mode("error")` 上下文里——只要稳态解码期间有任何隐式的主机等待，当场抛异常报错。

但我要提醒你一个我们自己踩过的坑：**PyTorch 的这个调试开关是有盲区的。** 它能抓住拷贝和 `.item()` 触发的隐式同步，却看不见代码里显式调用的 `Event.synchronize()`！为了确认 pLLM 真的做到了绝对的零等待，我们早先在 vLLM 0.27.1、0.5B 的 ESamp 负载（128 并发 × 256 token）上专门做过一次底层调用栈审计，把整个生成过程里的每一次同步都追到了直接调用方：整场跑下来一共发生了 128 次主机等待，**100% 来自 vLLM 自己每步用来取回采样结果的 `get_output`**，而 pLLM 自己产生的 Event 或 Stream 主机等待是严格的 **0 次**。工具开关只能当第一道筛子，真正的“零”得靠对调用链的冷酷审计来兜底。

第二件是**真实的 KV 驱逐抢占**。在 CPU 上，抢占只是我们在三元组列表里人工插入的一段重放；到了 GPU 上，我们要看 vLLM 的真实调度器在显存见底时会不会和 pLLM 配合失误。怎么让一个正常的测试稳定触发一次真实的抢占？我们没有去改 vLLM 的调度器，而是用 `_Eviction` 轻轻包了一层 vLLM 真实的 `kv_cache_manager.allocate_slots`（`pllm/testing_gpu.py`）：

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

盯住那个 `return None`。当第一个请求刚生成满 4 个 token、向分配器伸手要新 KV 块时，我们的包装函数故意对它两手一摊说“没内存了”；之后立刻恢复正常。vLLM 的调度器收到这一个 `None`，就会按它线上真实的代码路径把这个请求踢出去、回收它的块，等后面的步再把它排回来重算。跑完之后，我们比对这个倒霉请求最终吐出的 token 序列和它的终态参数，看跟从未被驱逐的基准运行是否一致。一行 `return None`，撬动了宿主引擎整条真实的容错链路。

第三件是**中性逐位等价（Neutral Bitwise Equivalence）**：当你把干预强度设为 0 时，挂载了 pLLM 的引擎吐出的每一个 token，必须和同样图模式下的原生 vLLM **逐比特相同**。这条契约听起来像是废话，但在 GPU 浮点算术里其实苛刻到了极点——在[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)里你已经看到，为了守住 7B 模型上这最后一位尾数的逐位一致，我们甚至不惜让自定义注意力主动退出了上游的一条编译器融合路径。

最后，当功能正确性全部过关，还有一个系统工程师无法回避的问题：**这段用户程序装进去之后，到底要花多少显存、走哪条流、会不会把引擎拖慢？**

eBPF 有 `bpftool` 和 `bpftrace`；在 pLLM 里，对应的工具是 `pllm.explain(workload, config)`。它同样在纯 CPU 上运行，读取 `Qwen2.5-7B-Instruct` 的配置，第一次调用约 0.2 秒，同进程内再次调用约 0.1 秒，直接给你打印出一份精确到字节的物理执行账单。下面是前面那个 `Distill` 在 7B、128 并发下的账单节选：

```text
forward  CUDA graphs: no effect needs eager
……
member  effect  anchor / lane / graph  state reads (origin, distance)
……
loss  train  token.commit / side / captured  distiller (latest, 0); one AdamW step per state per engine step; prompt rows train once per prefill chunk
……
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
……
tap      model.layers.0[0], decode: 1837056 bytes shared by loss, predict
……
state  scope  roots  bytes/root  reserved bytes [exact; state_page_budget_bytes]
distiller  request  514  516176  265314524  field 0: float32, field 1: float32, ……
memory   peak 1512728864 bytes funded by pLLM [per-rank tensor reservations; excludes GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 1512728864 bytes tasks run inline; below 1333743904 bytes planning fails
……
host     0 syncs in traced workload math
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

你不需要去猜框架在背后耍了什么聪明：
* `forward` 告诉你模型主干稳稳保住了 CUDA graph；
* `member` 告诉你 `loss` 被排在了 `token.commit` 锚点的 `side`（旁路流）上且已被录进图，而 `predict` 在 `state.publish` 的主流上读取滞后一步的快照 `(boundary, 1)`；
* `tap` 告诉你 `loss` 和 `predict` 都要读第 0 层的 decode 激活，框架只建了一份约 1.84 MB（`1,837,056` 字节）的共享缓冲；
* `state` 与 `memory` 把状态池的 514 个版本根（共约 265 MB）和 1.51 GB 的运行峰值摆在明面上，连同两条冷酷的预算红线：如果显存预算压到 `1,512,728,864` 字节以下，旁路训练就会放弃并发重叠、退回主流串行执行；如果压到 `1,333,743,904` 字节以下，规划直接报错拒绝启动。

顺便提一句，账单上如此精准的显存数字，靠的也是前面那次 `meta` 张量追踪。早期我们曾试过为每个算法手写显存估算公式，结果不仅维护起来苦不堪言，而且为了防溢出总是写得极度保守。改用 `meta` 追踪自动推导张量生命周期后，ESamp `predict` 的工作区预留从手写公式的 177.6 MB 一口气降到了 27.8 MB；OverRIDE 的训练预留从 1,056 MB 降到了 267 MB——而在 GPU 上实测出来的真实峰值是 `264,242,176` 字节，严丝合缝地卡在推导上限 `267,269,120` 字节之内（历史测量）。手写的公式永远和代码隔着一层肚皮，代码一改公式就撒谎；而追踪出来的数字，和真正要跑的代码永远是同一份。

在这份账单背后，藏着 pLLM 贯穿始终的一条设计哲学：**拒绝静默回退（Fail-Closed）**。

在研究性质的库里，当显存不够或者算子不兼容时，框架喜欢做好人：悄悄帮你关掉 CUDA graph 退回 eager，或者悄悄把放不下的大表溢写到 CPU 内存。但在高并发生产服务里，“做好人”就是埋地雷——在 0.5B 模型上关掉 CUDA graph 会让吞吐当场腰斩到 49%（7B 模型一步本身够长，也依然会损失约 2%），把表挪到主机内存则会平白引入每步一毫秒以上的 PCIe 传输。因此，除 `cuda_graphs="auto"` 会在 `forward` 行显式写出退回 eager 的原因外，pLLM 绝不替你悄悄切换到慢路径。昂贵的妥协——无论是 `placement="host"`、`publish_lag=0` 还是调低显存预算放弃双流重叠——必须由你在代码里亲手签字画押（详见[《设计取舍》](../design-tradeoffs.md)）。

---

## 诚实的边界，以及“调度即数据”

写到这里，我们需要给这套验证体系泼一盆清醒的冷水。

首先，**差分测试不是数学证明**。eBPF 的 verifier 在放行一段字节码时，是在所有可能的输入路径上给出了安全保证；而 `pllm.testing.check` 只是在 3 个请求、长度为 `(5, 1, 9)` 的几套手写固定调度表上跑过了蜕变测试。它不是像 QuickCheck 或 Hypothesis 那样随机生成成千上万种拓扑的属性测试——如果你的代码里写了一个极其古怪的 bug，非得在并发请求数超过 3 个、或者序列长度恰好是 1024 时才触发，CPU 上的这几套三元组是抓不住它的。

其次，**正态合成输入会掩盖数值病态**。`_token_values` 生成的是标准正态分布 `torch.randn`，它非常适合用来抓“行混了”、“位置错位了”这类结构与路由错误；但如果你的算法在遇到极端稀疏激活、接近 0 的分母、或者极大 logits 时会发生 `NaN` 溢出，温和的正态随机数根本试不出来。这类数值分布问题，得靠 `pllm.testing.check_distribution` 在真模型上量一量输出分布被挪动了多少（它量的是分布的移动，不是任务质量）。

再者，**最危险的深水区恰恰最依赖 GPU**。像 `@pllm.custom_residual`、参数覆盖层这类直接插进模型层内部的手术（自定义注意力在 CPU 上也只查得了方法本身的数学和分页 KV 调度），最容易写错维度或破坏逐位中性，偏偏它们在无模型的 CPU 检查里只能 `SKIP`，必须等上了 GPU 跑 `check_on_gpu` 才能现形。同样，`pllm.explain` 的算子计数与 FLOP 估计还没有和硬件性能计数器做过系统校准，它的显存账单也不包含基座模型权重、vLLM 自身的 KV cache 以及依赖具体 GPU 型号的 cuBLAS 工作区——整台机器到底装不装得下，最终要到 `pllm.make_llm` 在显卡上真刀真枪分配完才能拍板。

但即便有这些边界，每当我回顾 `pllm.testing.check` 的设计时，依然觉得其中有一个超越推理引擎本身的架构思想值得你带走。

那就是：**把调度变成数据（Schedule as Data）。**

在构建多租户系统——无论是共享同一个大模型权重的几十个推理请求、共享同一个数据库实例的几百个事务，还是共享同一个流处理引擎的多个作业——的时候，最难测、也最让人头疼的永远是**隔离性（Isolation）与调度无关性（Schedule Invariance）**。你很难用静态分析去证明租户 A 的操作绝不会透过某块共享缓冲或某次批处理污染租户 B。

但如果你在设计系统内部接口的那一天，就把“调度器在每一步做了什么决定”剥离成一个极简的纯数据结构（就像我们的 `(req_id, scheduled_tokens, computed_tokens)` 三元组），让下游所有的状态管理、行路由和算子执行都只认这串数据，奇迹就发生了：

1. 你不再需要为了测试去启动庞大的底层引擎，也不需要维护一套会撒谎的 Mock 模拟器——测试代码和生产代码吃的是同一种三元组，走的是同一条执行管道；
2. 复杂的系统级不变式（分块、混排、乱序、抢占恢复），瞬间坍缩成了几行静态三元组列表之间的两两比对；
3. 而看起来高深莫测的“多租户隔离验证”，最终只需要在生成合成输入时，给邻居们的随机种子加上一百万盐（`salt = 1_000_000`），然后看看自己变没变。

这大概就是整个 pLLM 系列想要传达的工程态度。面对一个为了榨干硬件而被焊死得密不透风的现代推理引擎，我们没有魔法去证明任意用户代码的绝对正确；但通过划定声明式的边界、复用同一个最小原语、把物理代价和调度交错统统摊在阳光下变成可检验的数据，我们依然可以让一段新算法既跑出裸金属般的速度，又不必每天提心吊胆地去猜它什么时候会悄悄算错。
