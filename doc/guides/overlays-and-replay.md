# Overlay 与重放

> **先决条件**：[状态与记忆](state-and-memory.md)；教程
> [03](../tutorials/03_request_lora.ipynb)（overlay 全家桶）、
> [04](../tutorials/04_loops.ipynb)（层间重放与再循环）。

生产里常见的两类需求：推理过程中修改模型的**有效权重**（每请求一副 LoRA
式补丁、在线 TTT），或者让某些层在更新发布后**立即重跑**。这篇指南回答怎么
声明这两件事，以及一个常见的后续问题：补丁的字段值不来自引擎内训练行不行？
（行，见下文「外部生产的 overlay」。）

## 引擎内训练的 overlay

**要解决的问题**：请求在生成的同时，根据它自己的推理轨迹更新一副私有补丁，
后续 token 立刻在"基座权重 + 补丁"的有效权重下运行——且不碰共享的基座权重。

`@pllm.overlay` 把一个 `MODEL_UPDATE` 任务变成"读操作数 → 写私有因子 →
应用有效权重"的完整循环：

```python
targets = (pllm.ParameterTarget(9, ParameterModuleKind.MLP,
                                ParameterRole.DOWN_PROJECTION_WEIGHT),)

class MyLoRA(pllm.Workload):
    factors = pllm.RequestState({
        "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),    # (rank, 输入宽)
        "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),   # (输出宽, rank)
    }, dtype=torch.bfloat16)

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(state="factors", targets=targets,
                  operator=pllm.ParameterOverlayOperator.LOW_RANK_ADD,
                  factors=("a", "b"), scale=1.0, decode=True)
    def update(self, x: Annotated[torch.Tensor, pllm.Operand.INPUT],
               y: Annotated[torch.Tensor, pllm.Operand.TARGET],
               base: Annotated[torch.Tensor, pllm.Operand.WEIGHT],
               a: Annotated[torch.Tensor, pllm.Parameter("factors", "a")],
               b: Annotated[torch.Tensor, pllm.Parameter("factors", "b")]) -> TensorUpdate:
        output = x @ (base + b @ a).T                    # 有效权重 = 基座 + scale·b@a
        a.add_((0.1 * y[:, :a.shape[0]].T @ x).to(a))    # 只有 a/b 可原地写
        b.add_((0.1 * (y.T @ y)[:, :b.shape[1]]).to(b))
        return TensorUpdate(output, changed=True, update_count=1, trained_rows=x.shape[0])
```

**overlay 会不会改到共享权重？** 不会。`Operand.INPUT/TARGET/WEIGHT` 是捕获的
张量，永远只读；能原地修改的只有 `Parameter` 命名的 buffer——那是这条 lineage
私有的因子存储。基座权重只是被借用来计算有效权重，补丁落在私有页上，按版本
发布。

**为什么 buffer 必须零初始化、不允许自定义初始化器？** 零初始化的 overlay
精确等于没有补丁——导入的零因子会精确复现基座模型，准入与导入都建立在这条
基准线上。代价是一个退化陷阱值得
记住：零初始化下，如果因子互相依赖对方的增量（`a += f(b)`、`b += g(a)`），
两者会永远停在零。至少让一个因子的增量只依赖操作数——上例中 `a` 的增量只用
`x` 和 `y`。

**`prefill`/`decode` 控制什么？** 更新资格（默认 `prefill=True、decode=False`）。
被禁用的阶段不更新，但仍会应用既有 overlay——它读的是已发布的版本。

**稠密还是低秩？** 稠密补丁用默认的 `DENSE_ADD` 和一个固定 `shape=` 的
`value` 字段；低秩用 `LOW_RANK_ADD` + `factors`，应用 `scale · b @ a`。
算子选择与请求作用域正交：同样的发布、pin 与隔离规则对两者都适用。

两个边界提前说清：后端目前定位 MLP down 与 attention q/v 投影；overlay
不能用共享前缀缓存（绑定时就拒绝，不是运行时才出错）。

## 重放：让更新立即生效

**要解决的问题**：默认情况下，decode 步的更新要等下一步才可见。如果算法
（比如 In-Place-TTT）要求更新**当步**生效，就需要重放。

`replay=pllm.ConditionalLayerReplay((5, 8))` 声明一组升序层：对选中 token，
用发布后的状态按序重跑这些层。每个被重放的层覆盖自己的当前 KV 槽，集合外
的层保留正常前向的 KV——混合 KV 是一种**声明过的近似**：它写进了签名，不是
静默发生的。

**replay 和重新生成差在哪？** 重放不重解码：logits 与 RNG 原样保留，只有
选中 token 在签名层集合上的层输出被重算，KV 原位覆盖。它是一个有界、带收敛
判据的层区间重跑，不是第二次生成。

两条纪律保证这个近似可控。其一，每个被重放的层必须被签名 overlay 的 target
覆盖——没有补丁的层没有重放的意义，绑定直接拒绝。其二，replay 复用原始的
发布版本，且只对 `changed=True` 的状态运行：只动了尾部 buffer 的发布不会触发
重放。教程 04 的"层循环"一节有"重放 vs 不重放"的并排对照，可以直观看差别。

**只想逐 token 循环 hidden，不想动权重呢？** 那是另一条路：在负载上声明一个
独立的 `pllm.Recurrence(source_layer=-1, destination_layer=0, alpha=1.0,
beta=None, ramp_tokens=0)` 属性即可，无需任何状态（教程 04 的"步循环"一节）。
代价是它必须独占负载：与普通训练/激活操作混用会在绑定时报错。

## 外部生产的 overlay（SHINE 式）

**要解决的问题**：补丁字段不想在引擎里训练——它来自超网络、加载的
checkpoint，或任何一段宿主计算。

做法和引擎内训练几乎一样：声明同样的 buffer，再声明一个 `Overlay` property
命名状态、targets、算子、因子与 scale；把产出的张量经
`generate(..., auxiliary=pllm.StateTensorUpdate(values, keys), prompt_indices=...)`
送入：

```python
@property
def adapter(self) -> pllm.Overlay:
    return pllm.Overlay("factors", targets,
                        operator=pllm.ParameterOverlayOperator.LOW_RANK_ADD,
                        factors=("a", "b"), scale=0.5)
```

**这和引擎内训练的 overlay 有什么差别？** 只有生产地点不同。导入在
`request.prepare` 发布——即请求的第一次前向之前——且只导入到版本零的请求状态
一次；导入的零因子精确复现基座模型。之后的应用、发布、隔离、退役与引擎内
训练的 overlay 完全相同。优化器、超网络、checkpoint 是同一状态的三种生产
方式，消费端不区分。

## 激活检查点（ReFT 式）

**要解决的问题**：要改的不是权重，而是激活。

声明 buffer 加一个 `pllm.Activation` property（命名状态、层、首/末 prefill
token、算子），同样经 `auxiliary=` 送入。`LOW_RANK_AFFINE` 每层用
weight/bias/rotation 字段，`ADD` 用一个 FP32 向量。见维护中的
[ReFT](../../pllm/workloads/reft.py)。

## 组合边界：失败关闭

overlay、激活与两种 replay 是彼此独立的声明。**可表达不等于支持任意组合**：
后端能力检查拒绝不支持的组合，拒绝发生在绑定或安装时——失败关闭，不是静默
降级。共享前缀缓存与 overlay/重放不兼容、Recurrence 不能与普通训练混用，
都是这一纪律的实例。

精确语义（TensorUpdate 字段、changed 与发布、copy_fields、容量声明）见
[创作契约](../reference/authoring-contract.md#请求-overlay)。
