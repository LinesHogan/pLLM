# Overlay 与重放

> **先决条件**：[状态与记忆](state-and-memory.md)；教程
> [03](../tutorials/03_request_lora.ipynb)（overlay 全家桶）、
> [04](../tutorials/04_loops.ipynb)（层间重放与再循环）。
>
> **本文回答**：要在推理中修改模型的有效权重（LoRA 式补丁），或让某些层
> 重跑，该怎么声明？字段值不来自引擎内训练行不行？

## 引擎内训练的 overlay

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

关键规则：

- **操作数只读，因子可写**。`Operand.INPUT/TARGET/WEIGHT` 是捕获的张量；
  能原地修改的只有 `Parameter` 命名的 buffer。
- **Buffer 必须零初始化**，不允许自定义初始化器。注意退化陷阱：零初始化下
  因子互相依赖的增量（`a += f(b)`、`b += g(a)`）永远停在零 —— 至少让
  一个因子的增量不依赖因子（上例 `a` 的增量只用操作数）。
- **`prefill`/`decode` 决定更新资格**（默认 prefill=True、decode=False）。
  被禁用的阶段只应用既有 overlay，不更新。
- 稠密补丁用默认 `DENSE_ADD` 和固定 `shape=` 的 `value` 字段；低秩用
  `LOW_RANK_ADD` + `factors`。算子选择与请求作用域正交。
- 后端目前定位 MLP down 与 attention q/v 投影。Overlay 不能用共享前缀缓存。

## 重放：让更新立即生效

默认情况下 decode 步的更新下一步才可见。`replay=pllm.ConditionalLayerReplay((5, 8))`
声明一组升序层：对选中 token 用发布后的状态按序重跑这些层，被重放的层覆盖
自己的当前 KV 槽，集合外的层保留正常前向 KV —— 这是**声明过的近似**。
每个被重放的层必须被 overlay target 覆盖；replay 只对 `changed=True` 的状态
运行。教程 04 有"重放 vs 不重放"的并排对照。

逐 token 的 hidden 再循环是另一条路：声明独立的
`pllm.Recurrence(source_layer=-1, destination_layer=0, alpha=1.0, beta=None,
ramp_tokens=0)` 属性即可，无需状态（教程 04 第 3 节）。它不能与普通训练/激活
操作混用（绑定时报错）。

## 外部生产的 overlay（SHINE 式）

字段值不必来自引擎内更新 —— 超网络、加载的 checkpoint、任何宿主计算都行。
声明同样的 buffer 和一个 `Overlay` property，把产出的张量经
`generate(..., auxiliary=pllm.StateTensorUpdate(values, keys), prompt_indices=...)`
送入：

```python
@property
def adapter(self) -> pllm.Overlay:
    return pllm.Overlay("factors", targets,
                        operator=pllm.ParameterOverlayOperator.LOW_RANK_ADD,
                        factors=("a", "b"), scale=0.5)
```

导入在 `request.prepare` 发布（请求的第一次前向之前），只导入到版本零的
请求状态一次；导入的零因子精确复现基座模型。应用、发布、隔离、退役与引擎内
训练的 overlay 完全相同 —— **只有生产地点不同**。优化器、超网络、checkpoint
是同一状态的三种生产方式。

## 激活检查点（ReFT 式）

改激活而不是权重：声明 buffer + `pllm.Activation` property（命名状态、层、
首/末 prefill token、算子），同样经 `auxiliary=` 送入。`LOW_RANK_AFFINE` 每层
用 weight/bias/rotation 字段，`ADD` 用一个 FP32 向量。见维护中的
[ReFT](../../pllm/workloads/reft.py)。

精确语义（TensorUpdate 字段、changed 与发布、copy_fields、容量声明）见
[创作契约](../reference/authoring-contract.md#请求-overlay)。
