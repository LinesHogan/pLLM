# 不重录 CUDA Graph 怎么在运行时动态切层？对比条件子图、掩码求值与 L2 早退 Triton 内核

> **TL;DR**：在线推理服务录好 CUDA Graph 后，想把激活干预或 LoRA 的作用位置从第 8 层切到第 20 层，或者让同批次的不同请求各自命中不同层，直接重录整套图会带来数秒停服；而在候选层全挂上朴素 `torch.where` 掩码又会让未命中层白跑全量显存读写与矩阵乘。对比过 CUDA 条件子图的私有工作区显存账之后，我们最终采用 **L2 标量早退 Triton 内核 + 借用边界输出端口原地累加** 的方案，让未命中候选层在 ~1 微秒内零 HBM 读写退出。文末复盘为什么大秩回退路径上一句不起眼的 `if bool(mask.any()):` 平时测不出来、上线调参却会炸毁图捕获。

做模型内部可解释性实验或者在线激活引导（Steering）时，常会遇到这样一个需求：7B 或 70B 的模型已经在显卡上跑起来了，分档位的 CUDA Graph 也花十几秒钟全部预热捕获完毕；这时候你想把读取残差探针或注入干预向量的位置从第 8 层换到第 20 层，甚至希望同一个连续批次里的请求 A 走第 4 层的 LoRA、请求 B 走第 18 层的 LoRA、请求 C 传 `-1` 走原版基座模型。

如果把层号当成静态结构参数，每换一次层就得拆掉旧钩子、重新触发 Dynamo 编译并把 `[1, 2, 4, 8, ..., 256]` 十来个批次档位的 CUDA Graph 全部重录一遍。离线脚本里等十秒尚可忍受，放到交互式探索或在线服务里，每次切层卡顿十秒基本没法用。

既然不想重录图，自然就得提前把候选层集合 `over=(4, 12, 20, ...)` 的钩子全埋进图里，运行时靠状态池里的一个 `0-d int64` 标量控制哪一层真正干活。

想法听起来简单，但只要算一算显存和带宽账，“怎么让没选中的那批候选层闭嘴”就成了真问题。假如有人想在 28 层模型上做逐层扫描，把 `over=tuple(range(28))` 全挂上 LoRA：每一步每行其实只有 1 层需要修正，另外 27 层全是陪跑。如果陪跑层的代价压不下去，动态切层的吞吐就会退化成“同时挂了 28 层静态适配器”。

---

在冻结的 CUDA Graph 里实现“命中才算、未命中跳过”，手头主要有三条技术路线可选。我们把三条路线的硬件账挨个摆在桌面上比了一遍。

**第一条路：用 CUDA 12.4+ 的设备端条件图节点（Conditional CUDA Graph / `device_branch`）。**  
从直觉上看，这最像教科书标准答案——在每个候选层放一个硬件级 `IF (active_count > 0)` 条件节点，只有本步真有请求命中第 $l$ 层时，GPU 硬件调度器才展开执行子图里的矩阵乘。

但它有一个在多档位推理引擎里吃不消的隐性显存开销：**每个条件子图分支内部的算子库（如 cuBLAS）会在各自的分支流上独立申请私有工作区（Workspace）显存池。**

```text
【路线 A：CUDA 条件子图】
  档位 1, 2, 4, ..., 256 (共 9 档) × 28 个候选层 IF 子图
  ──▶ 每个子图分支一旦调 cuBLAS BMM，都会在条件流内存池里圈走数 MiB 私有工作区
  ──▶ 256 MiB 的图显存预算（cuda_graph_pool_budget）尚未跑完预热就会直接耗尽报错
```

更何况，在异构连续批处理里，哪怕批次中只有 1 个请求命中第 $l$ 层，整层 `IF` 条件依然为真，子图还是得照常进去区分哪几行有效、哪几行无效。单靠层级的布尔分支，既省不下异构批次的行级过滤，还会把原本留给 KV Cache 的几百 MiB 显存赔进去。

**第二条路：朴素全算后掩码（`torch.where(mask, h + delta, h)`）。**  
这是写起来最快、也最容易通过功能测试的写法：每层照常按槽位收集权重因子 `A, B` 或仿射矩阵 `W, b, R`，照常跑完批量矩阵乘算出 `delta`，最后用一行 `torch.where` 把未命中行的增量抹掉。

它不占额外工作区，也能录进 CUDA Graph，但带宽账全盘亏空——未命中的 27 层照样要把全部权重块和隐藏状态从高带宽显存（HBM）搬进寄存器算完再丢掉。在排查第一版实现时，我们还顺带发现静态适配器路径里藏着一句无条件拷贝 `captured[:len(hidden)].copy_(hidden)`：原本那是为了给“边前向边求导训练”保留前向输入激活的，结果纯推理模式的适配器也跟着每层白拷一遍 `[rows, hidden]` 的激活进显存。

**第三条路：基于 L2 缓存标量判定的线程块早退（CTA Early-Exit）Triton 内核 + 借用已有输出端口原地累加。**  
回过头看 GPU 的存储层次：每一步前向开始前，我们把各行对应的目标层号比对结果写进一张只有几百字节的 `int64` 归属表 `owners[rows]`（未命中行记为 `-1`）。这张小表在整趟前向期间常驻片上 L2 缓存。

如果我们不用 cuBLAS，而是写一个按 `(row, col_block)` 划分网格的轻量 Triton 内核，让每个线程块（CTA）一进门先花几百纳秒去 L2 里读一个 8 字节整数 `owner = tl.load(owners_ptr + row)`，发现 `owner < 0` 立刻 `return` 退出呢？

```text
【路线 B：朴素掩码求值（未命中层照付 100% HBM 带宽）】
  HBM(hidden) + HBM(A, B 权重池) ──▶ 全量加载进 SM 算完 BMM ──▶ torch.where 丢弃结果 ──▶ 写回 HBM

【路线 C：L2 标量早退 + 原地累加（未命中层 0 字节 HBM 读写）】
  L2 Cache(owners[row] == -1) ──▶ CTA 第一条指令判定 < 0 ──▶ 直接 return 退出（耗时 ~1 µs，不碰 HBM！）
  L2 Cache(owners[row] >=  0) ──▶ 按 owner // block_rows 算槽位 ──▶ 读 HBM(hidden, A, B) ──▶ 原地加进输出切片
```

算清楚这笔账后，还要小心最后一个带宽漏勺：**千万别给激活补丁分配新的输出张量。**

哪怕你在内核里判断 `if active == 0: return` 跳过了权重池读取，只要算子签名设计成返回新张量的 `out = masked_patch_add(hidden, table, pool)`，那未命中行的 `hidden` 怎么去到 `out` 里？内核还是得老老实实把整行 `hidden` 从 HBM 读出来再原封不动写进新分配的 `out` 里——27 个未命中层照样白跑 27 遍全宽激活拷贝！

怎么把这遍无用拷贝也省掉？去看模型层间边界的调用上下文：在 `RMSNorm` 融合残差边界上，外层本来就已经准备好了两个现成的输出端口 `(residual, normed)`，并先把本层分支输出 `branch` 拷进了 `normed` 端口等待归一化。既然 `normed` 里已经躺着现成的值，补丁算子只需改成原地修改该端口的 `masked_patch_add_into(normed, table, pool)`：

```python
@triton.jit
def _masked_patch_add_into_kernel(
    target_ptr, table_ptr, pool_ptr,
    table_stride, target_stride, pool_stride, width, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # 先查常驻 L2 的掩码行；未命中时连 target_ptr 和 pool_ptr 的地址都不解引用，直接退出
    if tl.load(table_ptr + row) == 0:
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < width
    slot = tl.load(table_ptr + table_stride + row)
    dst_ptrs = target_ptr + row.to(tl.int64) * target_stride + cols
    h = tl.load(dst_ptrs, mask=col_mask, other=0.0)
    delta = tl.load(pool_ptr + slot.to(tl.int64) * pool_stride + cols, mask=col_mask, other=0.0)
    tl.store(dst_ptrs, h + delta.to(target_ptr.dtype.element_ty), mask=col_mask)
```

同理，在低秩 LoRA 适配器上，我们预先分配一块 `[max_rows, rank]` 的 `float32` 中间缓冲 `routed_mid`，用两阶段早退内核（第一阶段 `[rows, rank]` 算下投影，第二阶段 `[rows, cdiv(out_features, 256)]` 算上投影并原地加进 `output[:, offset:offset+width]`）替代通用分块 `bmm`。顺带还有一个意外收获：由于内核直接用 `owner // block_rows` 反查请求的参数槽号，当外层跑在自定义前向 `@pllm.custom_forward` 的抽样重排子批次（Wave，`call_rows != None`）上时，只需先做一次 `owners.index_select(0, call_rows)`，同一套早退内核连一行代码都不用改就能直接复用。

---

这套早退内核写完后，代码审查揪出了一个极具代表性的潜伏炸弹——它藏在我们平时根本不会测到的“大秩回退分支”里。

由于两阶段低秩内核把中间激活存在寄存器和 `routed_mid` 里，我们给快路径设了一个上限 `rank <= 256`；当用户声明的秩超过 `256`（或者张量是双精度 `float64`、或者最后一维步长非连续）时，代码回退到常规的 PyTorch 算子路径。为了“顺手省点未命中时的计算”，我们在回退分支里随手写了一句和 CPU 分支一模一样的判断：

```python
# 潜伏在 rank > 256 回退路径里的图捕获炸弹：
if target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
    mask = table[0, :rows].bool()
    if bool(mask.any()):  # <-- 平时 rank=4 从不走这里；一旦 rank=260，捕获 CUDA Graph 立刻崩溃！
        ...
```

为什么这句话在平时所有 `rank = 4`、`rank = 8` 的 GPU 测试里全都蒙混过关了？

因为常规测试全都命中了上面的 Triton 快路径，这行回退代码在 GPU 上一次都没执行过；而在纯 CPU 测试里，`bool(mask.any())` 读的是主机内存，合法且飞快。直到某天有人把低秩仿射的秩设成 `260` 并在 GPU 上启动引擎，预热录制 CUDA Graph 走到这行 `bool(mask.any())`，PyTorch 试图调用 `cudaStreamSynchronize` 把 GPU 上的布尔规约结果拉回 CPU 转成 Python `bool`——CUDA 驱动立刻抛出 `cudaErrorStreamCaptureUnsupported: operation not permitted when stream is capturing`。哪怕那一步所有候选层全都未命中（`mask` 全是 `False`），照样必崩无疑。

排查这类问题可以提炼出一条铁律：**在编写任何可能进入 CUDA Graph 录制范围的算子时，CPU 分支与 CUDA 回退分支必须分开审视——CUDA 分支（哪怕是慢速兜底路径）严禁出现任何把 Tensor 转成 Python `bool` / `int` 的控制流（`if mask.any():`、`int(x.item())`、`nonzero()`），一律改写为无分支的设备端 `torch.where(mask[:, None], updated, target)`，并且必须专门配一条越过快路径阈值的 GPU 图捕获单测（如 `rank = 260`）。**

```python
if not target.is_cuda or target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
    mask = table[0, :rows].bool()
    if not target.is_cuda and not bool(mask.any()):
        # 只有纯 CPU 路径才允许用 Python bool 短路跳过计算！
        return
    # CUDA 兜底路径保持纯设备端张量掩码，零主机同步，保证任何秩都能安全捕获进 CUDA Graph
    ...
    target.copy_(torch.where(mask[:, None], target + delta, target))
    return
```

---

把未命中层的 HBM 读写清零之后，我们在单张 RTX 4090 上用 `Qwen2.5-7B-Instruct`（`batch=8, n=16` 共 128 并发解码行，以及 `rows=1` 单流解码）实测了真实吞吐代价：

- 在 4 个候选层（`over=(4, 12, 20, 26)`）上同时挂动态激活引导 + 异步残差导出，并在轮次间实时调用 `import_state` 切层：128 并发下 TPOT 为 `24.6 ms`（静态单层为 `24.4 ms`，原版 vLLM 为 `23.2 ms`），相比静态单层仅多出 `0.2 ms`；
- 即便在最极端的 28 层全挂低秩 LoRA（`over=tuple(range(28))`，共 56 个 `q_proj` / `v_proj` 投影钩子，每步动态激活其中 1 层）场景下，得益于 27 个未命中层的 L2 早退与跳过无用输入拷贝，整套分档位 CUDA Graph 总共只占 `0.30 GiB` 显存，128 并发下 TPOT 为 `24.8 ms`（静态单层 LoRA 为 `24.0 ms`，27 个陪跑层合计仅增加 `0.8 ms`）。

这套早退内核与路由表机制目前集成在开源项目 **[pLLM](https://github.com/LinesHogan/pLLM)** 的 `pllm.model.LayerRoute` 中。最后也必须把它的物理边界摊开讲明：L2 早退省掉的是参数池读取与矩阵乘，但并没有让图节点凭空消失——第一，每个候选层每次前向仍要付约 1 微秒的内核网格发射与 L2 标量检查开销；第二，对于挂在层间归一化上的激活补丁，外层自定义边界算子仍需把残差流和分支输出端口传递给原生归一化核；第三，`per_head` 算子与 `rank > 256` 的超大秩兜底路径仍走全算后掩码。正因如此，我们在 API 设计上拒绝把普通静态层号参数悄悄做成全局动态，而是要求用户显式写出 `LayerRoute(..., over=(...))`——你需要切哪几层，就只为那几层的门控买单。
