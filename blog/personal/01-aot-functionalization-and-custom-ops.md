# 为什么自定义算子在即时模式下写进了数据，套上 `torch.compile` 却变回全零？从一次静默丢数看懂 PyTorch AOT 函数化

> **TL;DR**：在为推理引擎的层间归一化边界挂载自定义算子时，即时（Eager）模式下写入特征池的 `[7, 7, 7, 7]` 一旦经过 `torch.compile` 的 AOT 函数化（Functionalization）变换，就会无声无息地变回 `[0, 0, 0, 0]`。根源在于我们在 `custom_op` 签名里把特征池声明进了 `mutates_args`，算子内部却绕开形参、顺着全局字典里的原始对象引用直接写了显存——编译器在函数化收尾阶段拿“未被形参触碰过的全零克隆副本”反向覆盖了真实池子。理解 AOTAutograd 把原地变异翻译成纯函数 SSA 图的完整机制后，不仅能用十五行 CPU 最小探针在毫秒级抓出这类契约违约，还能顺手修掉动态跳层下“预填层号覆盖实际未访问层”的溯源漏洞。

事情起于一次看似稳妥的算子封装。

在连续批处理推理引擎里，为了在不打断 `torch.compile` 整图编译的前提下，往 Transformer 层间的 `RMSNorm` 边界塞进自定义的残差改写与激活探针（Tap），惯常做法是把这段 Python 调度逻辑包成一个不透明的 PyTorch 自定义算子 `pllm::model_site`。由于 `torch.compile` 会把整数标量当成编译期常量烘死进图里，我们不能传 Python 对象 ID，于是借用了第 0 个只读输入张量的首地址 `reads[0].data_ptr()` 当作运行期查找键，去全局注册表 `_SITES` 里取回当前会话的钩子对象执行。

为了守规矩，注册算子时我们还特意把所有可能被原地修改的缓冲区——包括激活探针用来收集隐藏状态的显存池 `tap_pool`——老老实实填进了 `writes` 列表，并在装饰器上标明了 `mutates_args={"outputs", "writes"}`：

```python
@torch.library.custom_op("pllm::model_site", mutates_args={"outputs", "writes"})
def model_site(
    inputs: list[Tensor], outputs: list[Tensor], reads: list[Tensor], writes: list[Tensor], hook: int
) -> None:
    # 顺着第 0 个只读张量的物理地址找回会话对象，执行第 hook 号边界逻辑
    _SITES[reads[0].data_ptr()].hooks[hook].emit(inputs, outputs)
```

在即时执行（Eager）和纯 CUDA Graph 录制下，这段代码表现得挑不出毛病：残差算出来是 `[7, 7, 7, 7]`，探针池 `tap_pool` 的目标槽位里也准确存进了 `[7, 7, 7, 7]`。

诡异的一幕出现在启用 Inductor / AOT 编译路径时。只要用 `torch.compile(..., backend="aot_eager", fullgraph=True)` 把这个边界包起来跑一遍：前向返回的 `residual` 依然是 `[7, 7, 7, 7]`，没有任何异常抛出，但回头去查 `tap_pool` 里的同一个槽位，里面躺着的却是整整齐齐的 `[0, 0, 0, 0]`。

明明在 `emit` 里亲手写进去的数，算子一返回，怎么就被人抹平了？

---

要看穿这个“写完即焚”的现场，得先补上 PyTorch 2 编译器栈里承上启下的一环：**为什么图编译器非做“函数化（Functionalization）”不可，以及它是怎么对付自定义原地算子的。**

平时写 PyTorch 代码，我们习惯了命令式的原地修改（In-place Mutation）：`pool.index_copy_(0, slots, values)`、`x.add_(1)`，甚至两个不同形状的视图指向同一块底层存储（Storage Aliasing）。这种写法对省显存有利，但对图编译器（如 Inductor）却是噩梦——编译器要做算子融合、死代码消除和指令重排，前提是整张计算图满足**单静态赋值（SSA，Single Static Assignment）**：每条边代表一个不可变的值，算子之间只有显式的数据流依赖，绝不存在“算子 B 偷偷改了算子 A 输入张量的底层内存”这种隐式副作用。

负责把充满原地修改的 PyTorch 程序翻译成纯函数 SSA 图的组件，就是 AOTAutograd 里的 **Functionalization（函数化器）**。

当它在追踪期遇到一个声明了 `mutates_args={"outputs", "writes"}` 的自定义算子时，它既不敢拆开你的黑盒算子看源码，又必须向后端编译器交出一个“无副作用的纯函数节点”。为了两头交差，PyTorch 的 `auto_functionalized` 包装器会在幕后自动插入一套三步转换：

1. **克隆替身（Clone）**：在调用你的自定义算子之前，把 `outputs` 和 `writes` 里的每一个张量各自克隆一份全新的临时副本（`writes_clone = [w.clone() for w in writes]`），把克隆副本作为形参传进你的算子实现；
2. **黑盒执行（Execute）**：运行你的算子函数体，期望你的代码把所有原地修改都写在这批克隆副本上，并把修改后的克隆副本作为纯函数节点的输出张量返回；
3. **尾声回写（Epilogue Copy-back）**：当需要把编译图的结果同步回外部持久状态（例如模型的 buffer 或外部显存池）时，生成一条显式的 ` writes_orig.copy_(writes_clone)`，用克隆副本里的最新内容覆盖原始张量。

把这三步画到时间线上，刚才那个灵异事件的原因就一目了然了：

```text
【即时模式 Eager】
  tap_pool (初值 0) ──────────────────▶ Tap.copy 顺着 self.pool 引用直接写入 [7,7,7,7] ──▶ 结果: [7,7,7,7]

【AOT Functionalization 模式下的“诚实声明 + 旁路偷写”】
  tap_pool (初值 0) ─┬─ 1. 编译器克隆 ─▶ writes_clone[0] (初值 0) ──▶ 传入 model_site(..., writes_clone)
                     │                                                       │
                     │                     (model_site 忽略了形参 writes_clone！)
                     │                                                       │
                     ├─ 2. 旁路直接写 ◀── _SITES[ptr].hooks[h].emit 顺着注册表里的 self.pool 写进 [7,7,7,7]
                     │     (此时真实 tap_pool 确实短暂变成了 [7,7,7,7])
                     │                                                       │
                     └─ 3. 尾声回写   ◀── tap_pool.copy_(writes_clone[0]) 拿未动过的全 0 克隆把 [7,7,7,7] 抹回 [0,0,0,0]！
```

问题恰好出在“规矩只守了一半”上。

`outputs` 之所以没丢数据，是因为 `emit(inputs, outputs)` 内部确实往传入的形参 `outputs` 上写了数据——克隆副本被改成了 `[7, 7, 7, 7]`，尾声拷回时自然也是 `[7, 7, 7, 7]`。而对于探针池 `tap_pool`，我们在外围 `run_hook` 里把它塞进了 `writes` 列表向编译器“自首”说我会改它，进了 `emit` 却根本没碰形参 `writes`，而是顺着 Python 闭包里缓存的 `self.pool.tensors[0]` 直接改了原张量。

在函数化器眼里：形参 `writes_clone[0]` 进去是全零、出来还是全零，说明你这步计算算出来的新状态就是全零；于是它在图结尾忠实地执行了一句 `tap_pool.copy_(writes_clone[0])`，亲手把你刚从旁路写进去的 `[7, 7, 7, 7]` 擦得干干净净。

更隐蔽的是，如果你反过来“干脆不在 `writes` 里声明 `tap_pool`”呢？那叫欺骗编译器：编译器会以为 `tap_pool` 从未被修改，后续若在同一张图里读取它，重排优化器可以直接把读取提前到 `model_site` 之前，或者干脆把未产生输出依赖的节点当死代码剪掉。

---

看懂了函数化器的脾气，修法就不再是给 `_SITES` 打补丁，而是重新审视职责边界：**探针拷贝根本没必要躲进 `_SITES` 黑盒里。**

之所以要有 `model_site` 这个靠指针查表的黑盒算子，唯一原因是自定义的残差改写方法里可能挂着任意复杂的 Python 闭包状态；而激活探针（Tap）要干的事单纯得多——拿到边界算完的 `(normed, residual)` 两个标准张量后，按行索引表把命中的行拷进显存池。

只要把 `self._copy_taps(normed, residual)` 从 `emit` 内部挪到外层的 `graph_forward` 里，紧挨在 `run_hook` 之后直接调用：

```text
  branch, stream ──▶ pllm::model_site (只负责算出 normed, residual，不再声明也不触碰 tap_pool)
                           │
                           ▼
                     normed, residual ──▶ pllm::routed_tap_copy(..., pool, slot_layers, layer_id)
                                          (显式声明 mutates_args=("pool", "slot_layers")，形参即实参！)
```

这样一拆，两条算子各司其职：`model_site` 的 `outputs` 走标准形参回写；而紧随其后的 `routed_tap_copy` 把真实的 `pool` 和溯源数组 `slot_layers` 直接作为张量形参传进去，并在内核里原地写入这组形参。AOT 函数化克隆它们、内核改写克隆体、尾声再拷回原池，整条 SSA 数据流闭环自洽，一行绕路代码都不用写。

---

顺着这次调整，我们还顺手拔掉了动态切层里另一个同源的设计隐患：**溯源层号到底该在“准备阶段”写，还是在“物理命中阶段”写？**

在支持动态切层导出（`@pllm.export`）时，每条导出的激活行都要附带一个整数标签 `event.layers[r]`，告诉下游“这一行激活究竟采自模型的第几层（若被 `-1` 旁路关闭则记为 `-1`）”。

第一版实现里，我们在前向开始前的 `bind_routes()` 阶段，一读出控制状态里的目标层号（比如第 `2` 层），就顺手把 `2` 提前填进了与显存槽绑定的 `_slot_layers[dst]` 里。在常规的 24 层顺序前向下，这完全正确：既然控制状态选了第 `2` 层，待会儿跑到第 `2` 层时探针必然会触发并把激活写进槽里。

但当它与自定义前向循环 `@pllm.custom_forward`（例如提前退出的自适应深度，某一步只跑了 `layers(0, 1)` 就直接返回）组合在一起时，矛盾就暴露了：

```python
@pllm.custom_forward(tokens="decode")
def early_exit(self, layers: pllm.model.Layers) -> torch.Tensor:
    return layers(0, 1)[0]  # 本步只执行第 0 层，压根没走到第 2 层！
```

控制状态说“我想看第 2 层”，但自定义前向在第 0 层结束时就跳出了循环，第 2 层的模块钩子压根没执行过。如果按第一版在 `bind_routes` 里提前预填 `_slot_layers`，下游消费者就会收到一行**数值全零（初始清零值）、标签却赫然写着来自第 2 层**的假数据；更糟的是，如果用户调用的是 `route.layer_logits()` 且候选层包含最后一层，后续的最终层 logits 替换逻辑看到标签等于 `L - 1`，还会把第 0 层的输出当成最后一层的 logits 塞进去。

把溯源写入从 `bind_routes` 下沉到刚才那个 `routed_tap_copy` 内核内部，问题迎刃而解：

```python
@triton.jit
def _routed_tap_copy_kernel(
    output_ptr, src_rows_ptr, dst_slots_ptr, row_layers_ptr,
    pool_ptr, slot_layers_ptr, out_stride0, out_stride1, pool_stride,
    width, layer_id, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    dst = tl.load(dst_slots_ptr + row)
    target_layer = tl.load(row_layers_ptr + row)
    if (dst < 0) | (target_layer != layer_id):
        return
    src = tl.load(src_rows_ptr + row)
    if src < 0:
        return
    if tl.program_id(1) == 0:
        # 只有物理层钩子真正跑到这里并拷贝数据时，才由第 0 列块落盘真实层号
        tl.store(slot_layers_ptr + dst.to(tl.int64), tl.full((), layer_id, dtype=tl.int64))
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    values = tl.load(output_ptr + src.to(tl.int64) * out_stride0 + cols * out_stride1, mask=mask, other=0.0)
    tl.store(pool_ptr + dst.to(tl.int64) * pool_stride + cols, values, mask=mask)
```

准备阶段一律把 `_slot_layers` 初始化成 `-1`；只有当某层的物理钩子真正被执行（包括自适应跳层显式广播冻结流的 `frozen()` 钩子）且 `target_layer == layer_id` 时，由第 0 号列线程块在拷贝激活的同时把 `layer_id` 写进 `slot_layers[dst]`。没跑到的层，激活是零，标签是 `-1`，事实与元数据在同一个原子时刻产生，再也不会脱节。

---

回头复盘这次故障，为什么我们的第一轮 GPU 端到端测试没能第一时间抓出 `model_site` 的函数化写回漏洞？

答案藏在引擎的编译器分派策略里：在之前的测试组合中，只要负载同时开启了自定义残差或投影适配器，引擎为了规避嵌套追踪限制会自动切到 `backend="eager"` 配合原生 CUDA Graph 录制——在那个分支下不经过 Inductor 的 AOT 函数化，旁路写碰巧能活下来；一旦用户单独使用纯激活补丁加探针，引擎切回 `backend="inductor"`，静默清零就会浮出水面。

这也给我们留下了一条低成本的日常开发习惯：**每写完一个带 `mutates_args` 的 `torch.library.custom_op`，不必急着起几 GB 的大模型测 GPU 图，先在纯 CPU 单测里套一层 `torch.compile(fn, backend="aot_eager", fullgraph=True)` 跑一行非零输入。** `aot_eager` 不需要显卡，也不用等 C++ 内核编译，0.1 秒内就能把 AOTAutograd 的克隆与回写契约完整走一遍——任何“声明了形参变异、实际却走闭包旁路写内存”的坏味道，在这层探针下都会立刻现形。

这套探针与动态路由实现目前跑在开源项目 **[pLLM](https://github.com/LinesHogan/pLLM)**（`pllm.model.LayerRoute`）中。需要留意它的代价边界：把 `routed_tap_copy` 作为独立自定义算子放在归一化边界外，意味着候选集合 `over` 里的每一层在 CUDA Graph 中都会留下一个轻量内核节点；虽然未命中层在 L2 读完两个 `int64` 就会直接退出，不碰 HBM 里的隐藏态池，但内核本身的图内发射开销仍与 `|over|` 成正比，适合在少数几个目标层之间切换，而不宜无差别挂满几十层。
