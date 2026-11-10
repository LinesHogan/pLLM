# 在 serving 引擎里信任一段用户程序

如果你在单请求的研究脚本里写一个“激活中心化”的钩子，很可能会随手写下这一行代码：

```python
# check-docs: skip: a one-line method body, not a complete program
return h - h.mean(dim=0, keepdim=True)
```

在单请求调试时，张量的第 0 维长度永远是 1。这行代码不报错、不崩溃，梯度能正常回传，单元测试一片绿灯。但是，一旦把这行代码原封不动地放进生产级的 vLLM 引擎，麻烦就会在毫无察觉中发生。

在连续批处理（continuous batching）引擎里，每一步进入模型的并不是规整的 `[batch, seq, hidden]` 立方体，而是一张把所有活跃请求首尾相接拍扁的二维张量 `[tokens, hidden]`（见[《行模型》](04-packed-rows.md)）。第 0 维上排着的，是同一步里恰好拼在一起的不同租户。当你在 `dim=0` 上求均值时，张三得到的回答，就开始取决于这一毫秒李四和王五恰好问了什么问题。没有异常堆栈，没有显存越界，生成的文本看起来依然通顺——只是结果随着线上负载的波动悄无声息地漂移。

在系统软件里，最难对付的从来不是当场崩溃的 bug，而是这种不报错、不挂起、只让计算结果静默偏离预期语义的错误（silent wrong-code / semantic bugs）。

除了跨行串味，这类静默错误在推理引擎里还有几种常见的形态。比如你在调试在线学习算法时，出于防御性编程的习惯加了一行断言：

```python
# check-docs: skip: a one-line method body, not a complete program
assert not torch.isnan(loss).any()
```

为了算出这个 Python 的 `bool`，CPU 必须立刻停下来，等待 GPU 把队列里排着的所有算子全部跑完，再把一个字节的结果跨过 PCIe 总线拷回主机内存。这叫作一次主机同步（host synchronization）。在异步流水线里，CPU 本来领先 GPU 一步在准备下一步的元数据；这一下急刹车直接把流水线排空。更麻烦的是，稳态解码步是被录成 CUDA graph 反复回放的——回放时 Python 解释器根本不在场，这行断言要么让图录制当场失败，要么只在录制那一步执行一次，之后彻底沦为摆设（见[《Python 只在录制时活着》](03-whole-algorithm-in-cuda-graphs.md)）。

再比如，当 GPU 的 KV cache 被长文本撑满时，vLLM 的调度器会临时抢占（preempt）一个正在生成的请求，释放它的物理块；等过几步显存宽裕了，再把这个请求的 prompt 连同已经生成的 token 从头重算一遍。如果一个维护着每请求状态的在线训练算法不知道“这是一次重放”，它就会把同一段历史再训练一遍，把优化器动量和权重算歪。

在这个系列的总起篇里，我们曾拿 Linux 内核里的 eBPF 来类比 pLLM：允许外部开发者写一小段带状态的程序，安全、高速地运行在底层引擎的关键路径上。前面的文章讨论的都是怎么让这段程序跑得和原生一样快。而在本篇里，我们要正面回答那个绕不开的安全性问题：

eBPF 之所以敢让用户代码进内核，是因为装载前有一个静态验证器（verifier）从数学上证明它不会越界、不会死循环；那么，面对一段研究者用 PyTorch 写的张量方法，我们凭什么敢把它放进一个每步只有几毫秒到二十几毫秒、同时服务上百个租户的推理引擎里？

先把最诚实的底牌亮在桌面上，免得把类比读得太满：**对一段任意的 PyTorch 代码，我们证明不了任何有意思的语义性质。**

PyTorch 不是受限的 64 位寄存器字节码，背后也没有一个可以穷举所有执行路径的抽象解释器。如果有人声称能用静态分析或形式化方法证明一段任意的 PyTorch 函数“绝不会让第 $i$ 行的输出污染第 $j$ 行”，那一定是不切实际的。

既然证明不了，那我们能做什么？

退后一步看前面的跨请求求均值、抢占后重复训练，再加上同一类的分块切分敏感（同一条 prompt 切成几块算，结果就变了）——会发现它们虽然发生在不同模块，违背的却是同一条语义不变式（invariant）：

**任何一个请求的最终结果，都只应该取决于它自己的输入历史，而对底层引擎“如何调度它”保持绝对不变。**

一旦把问题定义为验证这条不变式，思路就清晰了。在运行时做动态检查当然指望不上——在单张 NVIDIA RTX 4090（24 GiB）上运行 vLLM 0.32.0、开启 CUDA graph 与 FlashInfer 采样器、以 128 个并发序列（8 条 prompt × `n=16`）生成 256 个 token 时，`Qwen2.5-7B-Instruct`（BF16）一步解码只有约 21.7 ms，而 `Qwen2.5-0.5B-Instruct` 更是只有约 3.0 ms。在每一步里安插检查探针要么触发同步、要么拖垮吞吐；何况许多 bug 只有在“请求 A 恰好看完第 4 个词、请求 B 刚好在解码、请求 C 刚被抢占恢复”的特定交错下才会露头。

但如果我们在把程序装进 GPU 之前，先在 CPU 上用三道关口把它筛一遍呢？

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

第一道关，用声明式接口把程序关进一个已知的形状；第二道关，在规划期用一次虚拟张量追踪和图算法抓出所有结构与时序错误；第三道关——也是整套机制里真正值得细聊的部分——**把“调度”本身变成可以随意摆布的数据，在 CPU 上用真实的规划器与执行器对语义不变式做蜕变测试（metamorphic testing）**。

---

## 第一道关：先把程序关进一个形状

eBPF 验证器之所以能工作，前提是程序不能随意读写内核里的任意内存地址，它只能通过固定的上下文结构体访问数据，并调用白名单里的辅助函数。

在传统的深度学习插件系统里，框架通常会提供一个类似 `def on_step(engine, batch):` 的回调钩子，把半个引擎的内部对象直接暴露给用户。用户代码在里面改了哪个列表、读了哪层激活、在哪个分支里偷偷做了一次 `.item()`，框架在运行前一无所知。

在 pLLM 里，对应的方式是声明式契约。一个方法需要读什么，直接写在函数参数的默认值上；需要维护什么状态，写成类属性；对引擎产生什么副作用，写成装饰器。以在线蒸馏负载 `Distill` 为例（出自 `doc/explain/performance.md`）：

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

不需要执行 `loss` 的函数体，框架只扫描一眼函数签名就能确定：它需要从模型第 0 层和最后一层引出激活缓冲（`pllm.hidden`），需要按请求行对齐 `distiller` 的参数模块，并且会在采样结束后触发一次针对 `distiller` 的反向传播与优化器更新。同时，跨步存活的数据只能声明成受管状态：pLLM 在编译时把负载连同它依赖的函数冻结成一份脱离作者环境的签名副本，想在方法里写 `self.step_count += 1` 来记步数是行不通的——在 CUDA graph 的世界里，Python 方法只在录制时跑一次，把跨步状态存在 Python 对象属性里本来就会失效。

平心而论，这一层本身只是常规的软件工程。Python 的动态反射被严格限制在极窄的冷启动入口（`pllm/usage.py` 与 `pllm/workload.py`）；一旦跨过这道边界，所有的 Python 函数与装饰器就转化成了带类型的 `TensorDeclaration` 和 `SemanticGraph`。后面的规划器、显存分析器和契约测试器，都不再接触动态的 Python 对象。

但这个受限的声明式接口，是后面两道关能够成立的前提：**正因为所有的输入端口都已在签名上明牌，我们才有可能在不加载十几 GB 模型权重的情况下，在 CPU 上用合成张量或虚拟张量去驱动它。**

---

## 第二道关：Meta 追踪与规划期的结构检查

拿到语义图之后，程序在真正接触 GPU 之前，要先经过两轮静态扫描：一轮在 Python 里跑 PyTorch 的 `meta` 张量追踪，另一轮在 Rust 里跑依赖与时序规划。

先看 `meta` 张量追踪。PyTorch 的 `meta` 设备（以及 `FakeTensor`）提供了一种“只有维度和数据类型、不分配任何物理内存”的虚拟张量。当你把形状为 `[128, 3584]` 的虚拟张量喂进用户的 `loss` 方法时，PyTorch 的算子分发器会照常推导每一步矩阵乘和加法的输出形状。

通常人们只用它来检查张量维度能不能对齐。但既然已经把函数追踪了一遍，我们可以在同一次 CPU 空跑里顺手把四件事情一起做完：

第一是校验输出端口的形状与类型（例如 `@pllm.guide_sampling` 返回的是不是 `[rows, hidden_size]`）；第二是通过 `TorchDispatchMode` 拦截算子，估算这一步的 FLOPs、读写字节与源算子数量；第三是追踪精确的显存生命周期——每个中间张量的底层存储（`UntypedStorage`）在哪一个算子诞生、在哪一个算子之后引用归零，从而直接算出任务的工作区峰值与 CUDA graph 捕获池预留量。

用同一份算子追踪来推导显存，比手写静态估算公式要可靠得多。手写的显存公式不仅维护负担重，而且为了防溢出往往写得过度保守，代码一改公式就会失真；以一组 0.5B 尺寸（hidden 896、64 个打包行、8 个状态谱系）的对照为例：按手写公式，rank 128 的 ESamp `predict` 要预留 177.6 MB 工作区，而用 `meta` 张量生命周期追踪推导只需 27.8 MB；另一个在线训练负载 OverRIDE 的 `train` 则是 1,056 MB 对 267 MB——在 GPU 上实测出的预热后运行峰值是 `264,242,176` 字节，刚好卡在追踪推导出的 `267,269,120` 字节上限之内（首次调用另有约 8.5 MB 的 cuBLAS 初始化，不在任务张量的账上）。

第四件顺手做完的事最实用：**它天然就是一个零成本的“主机同步与动态控制流检测器”。**

原因很简单：`.item()`、`.tolist()`，或者对张量的值写一句 `if h.norm() > 10.0:`，在底层都需要去读取张量的具体数值。而虚拟张量里根本没有数值。只要用户的代码里藏着哪怕一处隐式同步或依赖张量数值的 Python 分支，追踪就会当场抛出异常。框架在 `pllm/execution/torch_private.py` 里所做的，只是把 PyTorch 内部晦涩的符号追踪异常拦下来，翻译成直接指出修改办法的报错信息：

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

通过了虚拟张量追踪后，语义图被送进 `rust/pllm_core/src/planning/` 下的 Rust 规划器。这里运行的都是标准的图算法，但有三个专门针对推理引擎语义的结构检查值得留意。

第一个是**确定性的依赖环检测**。在 `scheduler.rs` 的 Kahn 拓扑排序中，就绪队列使用的是有序的 `BTreeSet` 而不是哈希集合，每次总是 `pop_first` 取出编号最小的任务，保证同一份声明在任何机器上编译出的拓扑序完全确定。如果发现有任务入度未能归零，规划器会把相互等待的方法名全部列出来，告诉作者 `"methods ... wait on each other through results or state reads and writes; remove one of the dependencies in the cycle"`。

第二个是**防止读取尚未诞生的值**。推理引擎的一步被划分为 8 个严格有序的时间轴锚点（见[《自动放置与重叠》](11-automatic-placement-overlap.md)）。如果你写了一个挂在采样器之前的方法（比如 `@pllm.add_logits`），却在参数里声明要读取本步采样完成后才产生的 `pllm.sampled_tokens()`，规划器在沿着依赖边收紧每个任务的 `[最早锚点, 最晚锚点]` 窗口时，就会发现下界超过了上界（`scheduler.rs`）：

```rust
        if lo.iter().zip(&hi).any(|(a, b)| a > b) {
            return Err(
                "a task reads a value (sampled tokens, hidden states or another task's \
                 result) that exists only after its phase ends; declare it in a later phase"
                    .into(),
            );
        }
```

第三个是**状态版本的写写冲突检测**。在 pLLM 底层的 PTTS（Paged Tensor Transition System，分页张量状态迁移系统——注意它开头的 P 指的是与虚拟内存和 PagedAttention 同源的 Paged，即分页与页表，和 pLLM 里代表 programmable 的 p 没有关系）中，每份状态都被看作静态单赋值（SSA）形式的版本链：每次写入都是从一个旧版本派生出一个新版本（见[《PTTS》](01-ptts.md)）。如果两个方法同时声明从同一个源版本出发去写同一份状态，那最终谁的更新生效就取决于运行时两条流谁跑得快。这种并发竞争会在编译期被 `version_hazards.rs` 直接拦下：

```rust
            if let Some((other, _)) = commits.insert(origin(effect), (writer, commit)) {
                return Err(format!(
                    "{} and {} both write state {:?} from the same version; let one method write it",
                    graph.tasks[other].symbol, graph.tasks[writer].symbol, resource.resource
                ));
            }
```

不难看出这几处报错信息的共同原则：不仅指出违反了哪条约束，最后一句话总是明确给出修复动作。

---

## 第三道关：把调度变成数据，在 CPU 上做蜕变测试

走完前两道关，我们拦住了 `setattr()` 这类藏起可变状态的写法，拦住了 `.item()` 同步，也拦住了依赖死锁、时序倒置和并发写冲突。

但回过头看文章开头那行惹祸的代码：

```python
# check-docs: skip: a one-line method body, not a complete program
return h - h.mean(dim=0, keepdim=True)
```

它能毫无阻碍地通过前两道关。它的输出形状完全合法（因为加了 `keepdim=True`），没有触发任何主机同步，也没有构成依赖环。静态检查对它无能为力，因为它的错误出在**多租户的运行时语义**上。

怎么在不启动真实大模型、也不编写复杂张量符号证明器的前提下，自动抓出这类 bug？

在软件测试领域，面对“没有标准答案（no test oracle）”的复杂数值程序，最有效的武器是**差分测试（differential testing）与蜕变测试（metamorphic testing）**。对于任意给定的输入 $x$，我们确实不知道用户编写的函数 $f(x)$ 应该算出什么具体数值；但我们非常清楚，当对输入和执行环境施加某种保持语义的变换 $T$ 时，两次运行的结果必须满足严格的等价关系。

放在多租户推理引擎里，我们对用户程序要求的语义正确性，刚好可以完整地表达为一组关于“调度变换”的蜕变关系：

* **行置换不变（Row Permutation）**：请求 0、1、2 三个各自单独跑，与三个拼成一批按 `(0, 1, 2)` 跑、或者倒过来按 `(2, 1, 0)` 跑，每个请求在每个 token 位置上拿到的输出与状态必须一致；
* **租户隔离（Request Isolation）**：在同一批里，如果把除目标请求之外所有邻居请求的输入全部换成别的数值，目标请求自己的输出不能发生任何改变；
* **混排与分块不变（Mixed Prefill/Decode & Chunking）**：无论一条长 prompt 是一次性算完，还是被切成几块、与别的请求的解码步混在同一个批次里交错算完，结果必须一致；
* **抢占重放不变（Preemption Replay）**：请求算到中途被驱逐，之后无论是全量重放历史还是分块重放历史，最终状态与输出必须和从未被抢占时一致；
* **组件独立与中性等价（Component Independence & Neutrality）**：把多个 workload 组合在一起运行时，每个子组件的结果必须与它单独运行时逐位相同；当把干预强度（如 `beta=0`）关掉时，必须退化回不产生副作用的中性状态。

`pllm.testing.check` 做的就是这件事。它不需要加载几十 GB 的模型权重，也不占用 GPU，只读取模型的 `config.json` 形状信息，在 CPU 上用几秒钟把用户的 workload 放在上述几个不同的调度宇宙里各跑一遍，再逐个请求、逐个 token 位置比对所有输出。

要让这套检查既快又准，核心在于解决两个问题：合成输入怎么构造？调度器怎么模拟？

### 1. 用 `salt = 1_000_000` 测试租户隔离

既然不加载真模型，用户方法读取的 `pllm.hidden(0)` 从哪里来？

如果只是随手调用 `torch.randn` 生成整批随机数，那么“单独跑请求 1”和“三个请求合起来跑”时，随机数生成器的推进顺序就不一样了，请求 1 在两次运行里拿到的输入本身就不同，输出自然无法比对。因此，合成的激活值必须是由 `(请求编号, token 在该请求内的位置, 特征通道)` 唯一确定的纯函数。

来看看 `pllm/testing_contracts.py` 里生成测试激活的 `_token_values`，只有短短几行：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
def _random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _token_values(index: int, offsets: range, width: int, channel: int, perturbed: int | None) -> torch.Tensor:
    salt = 1_000_000 if perturbed is not None and index != perturbed else 0
    return torch.stack(tuple(_random((width,), salt + 10000 * index + 100 * offset + channel) for offset in offsets))
```

先看最后一行的种子算术：`10000 * index + 100 * offset + channel`。无论请求 `index` 在当前步被排在批次的第 0 行还是第 5 行，也无论它的 prompt 被切成了几块，只要是第 `index` 号请求的第 `offset` 个位置，它生成的随机向量永远完全相同。这就让跨调度的逐位置比对成为了可能。

再看上面那一行：

```python
# check-docs: skip: excerpt from pllm/testing_contracts.py
salt = 1_000_000 if perturbed is not None and index != perturbed else 0
```

这一行代码就完成了整套“多租户隔离验证”。

为了检验第 1 号请求（令 `perturbed = 1`）有没有受到同批邻居的干扰，我们不需要去分析用户的 AST，也不需要追踪复杂的张量广播规则，只需要把 `0, 1, 2` 三个请求放在一起跑两遍：

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

第一遍（`perturbed = None`），所有请求用正常种子运行，记录下请求 1 的所有输出与状态 `Out 1`。
第二遍（`perturbed = 1`），所有 `index != 1` 的邻居请求，种子都被加上了 `1_000_000`，它们的输入张量数值彻底改变；唯独请求 1 自己，因为 `index == perturbed`，`salt` 依然为 `0`，它收到的每一个输入比特都与第一遍完全相同。

这时候去比对 `Out 1` 和 `Out 1'`。只要用户的代码里在任何地方写了 `h.mean(dim=0)`、`F.softmax(x, dim=0)` 或是跨行索引，邻居因为那一百万盐而产生的数值变化就会立刻顺着第 0 维传给请求 1，被比对器抓个正着。

而且，一旦发现某一步的输出变了，究竟是用户的方法写错了，还是 pLLM 框架自己把行号路由错了？因为测试器在运行过程中记录了每个方法调用前后的按请求输入与输出切片，`_fault` 函数可以直接做出归因判定（`pllm/testing_contracts.py`）：

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

如果进入该方法之前，请求 1 的输入还与基准运行完全一致，出来之后却变了——那是这个方法把不同请求的行混在了一起，报错会提示将归约维度从 `dim=0` 改为 `dim=-1`；如果在进入方法之前输入就已经不一致了——那是框架自身的路由 bug，报错信息会直接要求提交框架 issue。

### 2. 三元组调度表：为什么不需要写模拟器

做系统测试时，一个常见的陷阱是为了在 CPU 上测试 GPU 逻辑，专门去写一套“模拟调度器（mock scheduler）”。一旦测试走的是模拟器、生产环境走的是真引擎，两边迟早会出现行为偏差，测试通过了也无法代表生产代码安全。

`pllm.testing.check` 之所以不需要写模拟调度器，得益于我们在[《行模型》](04-packed-rows.md)里建立的那个窄接口：vLLM 调度器在每一步向 pLLM 传递的批次信息，被完全收敛成了每个请求的三个整数——`（本步调度的 token 数, 原始 prompt 长度, 在本步之前已计算的 token 数）`。

既然生产环境里的 Rust 行标签器只接收这三个整数（`label_rows` 是它暴露给 CPU 检查的入口），而测试脚本里三个请求 `r2, r0, r1` 的 prompt 长度又是固定的 `(5, 1, 9)`，那么在测试里，**一整套连续批处理调度就退化成了一串静态的整数三元组 `(请求下标, 本步 token 数, 已算 token 数)`**。

来看看 `pllm/testing_contracts.py` 里用来测试“分块 prefill 与 decode 混排”的调度表 `_schedule`：

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

把前四步展开到时间轴上，就能看清这几行数字是如何故意刁难被测程序的：

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

在 Step 0，请求 0 的 5 个 prompt token 被切开，只算前 4 个 `(0, 4, 0)`——因为还没算完 prompt，声明了 `tokens="prefill"` 的方法在这一步绝不能为请求 0 触发；而请求 1 只有 1 个 prompt token `(1, 1, 0)`，当场完成预填充。到了 Step 1，长度为 9 的请求 2 进来先算前 4 个词 `(2, 4, 0)`，请求 1 已经进入第一个解码步 `(1, 1, 1)`，请求 0 则补完了它 prompt 的最后一个词 `(0, 1, 4)`——同时，这三个请求在扁平批次里的排列顺序被故意打乱成了 `2 -> 1 -> 0`。而在随后的步骤里，请求 2 的 9 个 prompt 词跨越三步（`4 + 4 + 1`）才在 Step 3 完成预填充，每一步的批次行序都在重新洗牌。

把这几行三元组直接喂给生产环境使用的同一个 `label_rows`、同一个 Rust 规划器和同一个状态执行器，CPU 测试就走完了与线上 GPU 相同的行打包、阶段筛选、状态版本推进与路由逻辑。（唯一的例外是抢占重放时用于标记“哪些历史位置已被消费过”的 `retained_rows` 游标过滤：CPU 检查器里使用的是一个十几行的 Python 参考实现，而生产环境由原生行标签器直接完成。）

当然，CPU 上并没有真正加载 Transformer 的权重层。如果一个 workload 用到了深入模型内部的操作——比如残差流改写（`@pllm.custom_residual`）、参数覆盖层（`Adapter`）或带状态的层间循环——CPU 解码器不会去执行那些模型层，也不会不懂装懂地打出 `PASS`，而是会在返回的 `CheckResult` 里明确记录一条 `SKIP`：

```text
SKIP    model-site execution: CPU decoder does not execute residual hooks, overlays, spans or PositionState
```

同样，如果 workload 包含自定义的 C++/Triton 算子但没有注册 CPU 参考实现，`check` 也会跳过 CPU 调度测试并给出具体的注册方法提示：`no CPU kernel for ...: add ....register_kernel("cpu", ...)`。跳过就是跳过，绝不伪装成通过。

---

## 第四道关：GPU 真机验证与 `explain` 成本账单

凡是 CPU 检查显式标为 `SKIP` 的模型内操作，以及只有真实硬件才能检验的物理性质，都交由 `pllm.testing.check_on_gpu(workload, config)` 在真实的 vLLM 引擎上补齐。

在 GPU 上，验证重点集中在三件事上。

第一件是**稳态解码期的零主机同步验证**。`SynchronizationCheck` 会在每个打包宽度跑过两步预热之后，把 pLLM 所有的锚点执行入口、输入准备和模型挂载点都包进 PyTorch 的 `torch.cuda.set_sync_debug_mode("error")` 上下文里——只要稳态解码期间在受检路径上发生任何隐式的主机等待（如 `.item()` 或同步拷贝），就会当场抛出异常并打印调用栈。

不过需要注意一个技术细节：PyTorch 的 `set_sync_debug_mode` 只能拦截由张量拷贝或标量读取触发的隐式同步，**它抓不到代码里显式调用的 `Event.synchronize()` 或 `Stream.synchronize()`**。因此，调试模式开关只能作为第一道防线，要确认系统真正实现零等待，还必须对底层调用链做直接审计。在单张 RTX 4090、vLLM 0.27.1、`Qwen2.5-0.5B` 的 ESamp 负载（128 并发序列 × 生成 256 个 token）上对整场生成过程做同步调用栈追踪时，全部 128 次主机等待 100% 来自 vLLM 自身取回异步输出的 `get_output`，而 pLLM 自己产生的 Event 或 Stream 主机等待为 0 次。

第二件是**真实的 KV 驱逐与抢占恢复**。在 CPU 检查里，抢占只是三元组列表里人工编写的一段重放；到了 GPU 上，我们需要检验当 vLLM 的真实调度器在显存见底并踢出请求时，状态回滚或游标保留是否真的能对齐。为了在不修改 vLLM 源码的前提下稳定触发一次真实的抢占，`pllm/testing_gpu.py` 里的 `_Eviction` 临时包装了 vLLM 真实的 `kv_cache_manager.allocate_slots`：

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

当第一个已生成至少 4 个 token 的请求向块管理器申请新的 KV 槽位时，包装函数故意返回一次 `None`（模拟显存耗尽），随后立即恢复正常分配。vLLM 的调度器收到这个 `None` 后，会走它线上真实的抢占代码路径将该请求驱逐出局、释放其 KV 块，并在后续步骤中重新调度预填充。测试器同时固定住每个 token 位置录得的隐藏层或投影输入（以排除基础模型在不同批次宽度下的浮点归约尾数差异），在生成结束后严格比对被驱逐请求最终生成的 token 序列与终态参数是否与未驱逐的基准运行一致。

第三件是**中性设置的逐位等价（Neutral Bitwise Equivalence）**：当传入 `neutral=`（例如把引导强度设为 `beta=0`）时，挂载了 pLLM 的引擎在贪心解码和固定随机种子采样下吐出的每一个 token ID，必须与相同图模式下的原生 vLLM 逐比特完全一致（见[《可编程注意力与 KV 布局》](08-programmable-attention-kv.md)）。

当功能与隔离性检查都通过后，剩下的问题是：**这段用户程序装进引擎之后，到底要花多少显存、走哪条流、会不会因为某个配置而退回慢路径？**

对应的可观测性工具是 `pllm.explain(workload, config)`。它同样在纯 CPU 上运行，读取模型配置即可生成报告（在 CPU 上解析 `Qwen2.5-7B-Instruct` 时首次调用约 0.2 秒，同进程内再次调用约 0.1 秒，均不含导入 pLLM 本身的时间）。下面是前文那个 `Distill` 负载在 7B、128 并发下的账单节选：

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

这份报告把规划器在幕后做出的所有物理决定都摆在了明面上：
* `forward` 确认模型主干保持了 CUDA graph，没有效果迫使模型退回 eager；
* `member` 显示 `loss` 被放置在 `token.commit` 锚点的 `side`（旁路流）上且已被捕获进图，而 `predict` 在 `state.publish` 的主流上读取滞后一步的快照 `(boundary, 1)`；
* `tap` 表明 `loss` 和 `predict` 都需要读取第 0 层的解码激活，框架只分配了一份约 1.84 MB（`1,837,056` 字节）的共享缓冲；
* `state` 与 `memory` 列出了状态池预留的 514 个版本根（约 265 MB）以及约 1.51 GB 的运行峰值，并给出了两条明确的预算水位线：如果显存预算低于 `1,512,728,864` 字节，旁路训练将放弃并发重叠、退回主流内联执行；如果低于 `1,333,743,904` 字节，规划直接报错拒绝启动。

在这份账单背后，是 pLLM 贯穿始终的一条设计取舍：**闭合失败（Fail-Closed）**。

在研究性质的框架里，当显存不够或者某个算子不支持捕获时，系统往往倾向于“替用户兜底”：悄悄关掉 CUDA graph 退回 eager 模式，或者悄悄把放不下的大表溢写到 CPU 主机内存。但在高并发服务引擎里，静默回退往往意味着隐蔽的线上性能雪崩——在 RTX 4090 上跑 128 并发时，关掉 CUDA graph 会让 0.5B 模型的吞吐直接从 42,355 tok/s 腰斩到 20,926 tok/s（49%），即便在单步计算较重的 7B 模型上也会从 5,894 tok/s 降到 5,773 tok/s（约 98%）；而把查找表挪到主机内存则会引入每步毫秒级的 PCIe 传输（见[《有界的零拷贝 gather 与预取》](12-bounded-zero-copy-gather.md)）。因此，除了默认的 `cuda_graphs="auto"` 会在 `forward` 行显式打印退回 eager 的原因外，框架绝不替用户悄悄切换到慢路径。任何昂贵的妥协——无论是 `placement="host"`、`publish_lag=0` 还是调低预算放弃双流重叠——都必须由作者在配置里显式写出（详见[《设计取舍》](../design-tradeoffs.md)）。

---

## 诚实的边界，以及“调度即数据”

聊完这四道关，我们需要客观地划清这套检查体系的能力边界。

首先，**差分测试不是数学证明**。eBPF 的 verifier 在放行一段字节码时，是在所有可能的输入路径上给出了安全保证；而 `pllm.testing.check` 只是在 3 个请求、长度为 `(5, 1, 9)` 的几套固定调度表上运行了蜕变测试。它甚至不是像 QuickCheck 或 Hypothesis 那样随机生成成千上万种调度拓扑的属性测试（property-based testing）——如果一段代码里藏着一个只在并发请求数超过 4 个、或者序列长度超过 1024 时才会触发的边界 bug，CPU 上的这几套固定三元组是抓不住它的。

其次，**标准正态合成输入会掩盖数值病态**。`_token_values` 生成的是温和的 `torch.randn` 标准正态分布随机数，它非常适合用来抓“行混了”、“位置错位了”、“抢占多训了一遍”这类结构与路由错误；但如果用户的算法在遇到极端稀疏激活、接近 0 的分母、或者极大 logits 时会发生 `NaN` 溢出，正态随机数很难试出来。同理，如果用户调用了不透明的自定义算子（custom op），由于其 `register_fake` 实现只做形状推导，静态追踪无法证明其内部 C++/Triton 内核真的没有主机同步，因此 `explain` 会如实将其标记为 `unverified`。

再者，**最容易出错的模型内手术恰恰最依赖 GPU 检查**。像 `@pllm.custom_residual`、参数覆盖层这类直接嵌入模型层内部的修改，在无模型的 CPU 检查里只能跳过（或仅检查方法自身的算术），必须上真卡跑 `check_on_gpu` 才能完成端到端验证；而在 `tensor_parallel_size > 1` 时，CPU 检查器不会启动多进程 TP worker，跨卡一致性同样需要多 GPU 环境验证。此外，`pllm.explain` 给出的算子计数与 FLOPs 估算尚未与硬件性能计数器做过系统性的偏差校准，它的显存账单也只统计 pLLM 自身管理的张量，不包含基座模型权重、vLLM 的 KV cache 以及依赖具体 GPU 型号的 cuBLAS 工作区——整台机器最终能不能装下，要到 `pllm.make_llm` 在显卡上真正完成分配才能揭晓。

尽管有这些明确的局限，回头看这套验证机制的设计，里面依然有一个超越推理引擎本身的工程思想值得带走：**把调度变成数据（Schedule as Data）。**

在构建任何多租户系统——无论是共享同一个大模型权重的几十个推理请求、共享同一个存储引擎的并发事务，还是共享同一个流处理拓扑的多个作业——时，最难测试的永远是隔离性与调度无关性。我们很难靠静态分析去证明租户 A 的操作绝不会透过某块共享缓冲或某次批处理重排去影响租户 B。

但只要我们在设计系统内部接口时，把“调度器在每一步做出了什么决定”剥离成一个极简的纯数据结构（在这里，就是每个请求的 `(req_id, scheduled_tokens, computed_tokens)` 三元组），并让下游所有的状态管理、行路由和算子执行都只认这串数据，测试的形态就会发生根本性的改变：

我们不再需要为了测试去启动庞大的底层引擎，也不需要维护一套容易失真的模拟器——测试代码和生产代码吃的是同一种三元组，走的是同一条执行管道。复杂的系统级交错（分块预填充、混排洗牌、抢占重算），直接化简成了几行静态三元组列表之间的两两比对；而原本棘手的多租户隔离验证，最终只需要在生成合成输入时给邻居的随机种子加上一百万盐（`salt = 1_000_000`），再看一眼自己的输出变没变。
