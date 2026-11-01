# Overlay 与重放

> **先决条件**：[状态与记忆](state-and-memory.md)；教程
> [03](../tutorials/03_request_lora.ipynb)（overlay 全家桶）、
> [04](../tutorials/04_loops.ipynb)（层间重放与再循环）。

生产里常见的两类需求：推理过程中修改模型的**有效权重**（每请求一副 LoRA
式补丁、在线 TTT），或者让某些层在更新发布后**立即重跑**。这篇指南回答怎么
声明这两件事，以及一个常见的后续问题：补丁的字段值不来自引擎内训练行不行？
（行，见下文「外部生产的 overlay」。）

`pllm.Field` 默认使用 FP32 保存累积状态。Dense / low-rank 投影读取时转换到模型输入 dtype；
"per_head" 保持 FP32。`pllm.Field(shape, dtype=torch.bfloat16)` 可显式节省一半字段显存，
代价是小更新可能被舍入掉；`dtype=None` 显式跟随模型 dtype。
`InPlaceTTTWorkload(..., dtype=...)` 与 `PrefillAdaptersWorkload(dtype=...)` 均可配置状态精度，默认 FP32。

## 引擎内训练的 overlay

**要解决的问题**：请求在生成的同时，根据它自己的推理轨迹更新一副私有补丁，
后续 token 立刻在"基座权重 + 补丁"的有效权重下运行——且不碰共享的基座权重。

`@pllm.train_in_forward` 把一个 `IN_FORWARD` 任务变成"读操作数 → 写私有因子 →
应用有效权重"的完整循环：

```python
import torch
import pllm

targets = (pllm.model.Projection("down_proj", 9),)

class MyLoRA(pllm.Workload):
    factors = pllm.RequestState({
        "a": pllm.Field(lambda in_features, out_features: (2, in_features)),    # (rank, 输入宽)
        "b": pllm.Field(lambda in_features, out_features: (out_features, 2)),   # (输出宽, rank)
    })

    @pllm.train_in_forward(state="factors", targets=targets,
                  operator="low_rank",
                  factors=("a", "b"), scale=1.0, tokens="all")
    def update(
        self,
        x: torch.Tensor = pllm.operand("input"),
        y: torch.Tensor = pllm.operand("target"),
        a: torch.Tensor = pllm.parameter("factors", "a"),
        b: torch.Tensor = pllm.parameter("factors", "b"),
    ) -> pllm.UpdateOutput:
        residual = (x @ a.to(x.dtype).T) @ b.to(x.dtype).T                         # 只返回补丁的增量，不含基座投影
        a.add_((0.1 * y[:, :a.shape[0]].T @ x).to(a))    # 只有 a/b 可原地写
        b.add_((0.1 * (y.T @ y)[:, :b.shape[1]]).to(b))
        return pllm.UpdateOutput(
            residual,
            residual.new_ones((), dtype=torch.bool),                     # changed
            residual.new_ones((), dtype=torch.int64),                    # update_count
            residual.new_full((), x.shape[0], dtype=torch.int64),        # trained_rows
        )
```

返回的 `UpdateOutput` 里，`residual` 是补丁在基座投影之上**加上**的那部分（框架再乘
`scale` 并加到基座输出上）；三个计数都是与输出同设备的 0 维张量，不是 Python
数字，这样一步之内不需要主机同步。默认在 `IN_FORWARD` 时机运行。

**overlay 会不会改到共享权重？** 不会。`pllm.operand("input"/"target"/"weight")`
是捕获的张量，永远只读；能原地修改的只有 `pllm.parameter` 命名的 buffer——那是
这条 lineage 私有的因子存储。基座权重只是被借用来计算有效权重，补丁落在私有页上，按版本
发布。量化模型上，`pllm.operand("weight")` 是 vLLM 反量化出的稠密副本：只有读取它的 overlay
才会让该投影多常驻一份模型精度的权重。

**想让框架替你求梯度？** 给状态声明 `optimizer=pllm.SGD(lr)` 并传 `chunk=`：
方法改为返回每行一个 loss，框架在每个 chunk 结束时用 SGD 更新补丁
（`flush="prompt_end"` 让 prompt 末尾不满一个 chunk 的行也参与）。优化器只写在
状态上，`@pllm.train_in_forward` 本身不带优化器，参见
[Prefill adapters](../../pllm/workloads/prefill_adapters.py)。

`initialize=` 可以初始化逻辑字段。普通 LoRA 训练通常随机初始化 A、将 B 置零，
这样初始模型增量为零，第一次优化器更新仍能学习 B。

**`tokens` 控制什么？** 更新资格（默认 `tokens="prefill"`，可选 `"decode"`、`"all"`）。
被禁用的阶段不更新，但仍会应用既有 overlay——它读的是已发布的版本。

**稠密还是低秩？** 稠密补丁用默认的 `"dense"` 和一个固定 `shape=` 的
`value` 字段；低秩用 `"low_rank"` + `factors`，应用 `scale · b @ a`。
算子选择与请求作用域正交：同样的发布、pin 与隔离规则对两者都适用。

两个边界提前说清：后端目前定位 MLP down 与 attention q/v 投影；overlay
不能用共享前缀缓存（绑定时就拒绝，不是运行时才出错）。

## 层路由：让更新当步生效

在同一 workload 上组合一个训练 overlay 与 `@pllm.custom_forward(tokens="decode")`。
`layers(...)` 读取当前可见版本并捕获操作数；`layers.apply_updates()` 将上个边界后
同一 target 的全部操作数按访问顺序拼接，执行 overlay transition。后续层调用读取
该 token 的私有字段。空边界不改变可见版本；最后一个边界之后的捕获被丢弃。
整个步骤复用同一个私有根，结束时由原 publication 发布，包括只更新尾部 buffer 的情况。

```python
@pllm.custom_forward(tokens="decode")
def settle(self, layers: pllm.Layers) -> torch.Tensor:
    taps = layers()
    layers.apply_updates()
    for _ in range(2):
        layers(start=22, stop=24, hidden=taps[21])
    return taps[-1]
```

返回值决定采样；示例保留第一遍输出。层调用覆盖当前 token 对应层的 KV，历史槽
保持不变。未选中的行遵循 overlay 的普通行为。入口读取遵循 `forward_pinned`
或 `"exact"`；显式边界之后的私有读取由路由源程序表达。
目前支持有界 Python `for` 循环；被追踪的 `while_loop` 会提示改用有界 for-loop。
逐 token 的 hidden 循环可直接使用独立 `custom_forward`，无需 overlay。

## 独立声明状态生产者与模型消费者

`pllm.model.Adapter` 只读 `state=` 指定的状态。`initialize=`、普通 `@pllm.train`、
`@pllm.write` 和显式 `imported=True` 可以生产同一份状态，消费声明不用改变。
作用域可以是 request 或 instance；一个资源每次 invocation 只允许一个修改生产者。
没有生产者时读取初始化的版本。AdamW 因子是其扁平参数字段的视图，不复制一份模型因子。

维护中的三个短示例见 [state_producers.py](../../pllm/workloads/state_producers.py)：

```python
from pllm.workloads.state_producers import PromptFactorsWorkload, PromptMixtureWorkload, DecodeFactorsWorkload

workload = PromptFactorsWorkload(rank=2)                 # 共享超网络 → 每请求 q/v 因子
workload = PromptMixtureWorkload(rank=2, components=3)  # 判别器 → K 个 LoRA 的精确加权和
workload = DecodeFactorsWorkload(rank=2)               # hidden 目标 → AdamW → 下次前向
```

三者继承同一个 `consume` property。形状由模型的 q/v 投影决定，包括 GQA 的不同输出宽度。
混合 LoRA 使用拼接 A 与加权 B，得到 `sum(p[k] * B[k] @ A[k])`，有效 rank 为 K×rank。
AdamW 示例的损失只训练旁路预测，不对 vLLM attention 或历史 KV 求梯度。

```python
@pllm.on(pllm.Phase.AFTER_FORWARD)
@pllm.write(state="factors", fields=("a_0", "b_0", "a_1", "b_1"))
def produce(self, h=pllm.hidden(8, tokens="prefill"), net=pllm.module("network")):
    return make_factor_tuple(net(h))  # 四个 tensor，前导轴对应本任务的请求行
```

`make_factor_tuple` 是用户自己的张量数学；完整可运行拆分见上面的示例。
写入总是返回 tuple，包括只有一个字段时的 `(value,)`；字段作为一个版本原子发布。
目标展开名称是 `a_0/b_0/a_1/b_1`。InstanceState 写入没有请求轴。
替换优化器管理的参数保留其 moments/step；不会隐式重置优化器。

`hidden(..., tokens="prefill")` 只取**完整 prompt 的最后一个 token**，每个合格请求一行；
部分 prompt span 不发布。默认仍是 decode。各任务独立选择 phase，结果消费者继承生产者的行域；
没有显式 join 时不能混合两种 phase。`write` 默认 BEFORE_FORWARD，读取本次 hidden 的写入须使用
AFTER_FORWARD 或 AFTER_SAMPLE。前向入口 pin 在整个 forward 内不变；prefill 结果只影响第一次 decode
forward，不改变已完成的 prompt forward 或由它采样的第一个生成 token。

These examples use eager execution with TP=PP=1. See
[Supported models](../../README.md#支持的模型与限制) for
architectures and the exact-class admission rule: overlays and replay need a
Qwen2, Llama or Qwen3 model, quantized or not. Q/V factor sizes come from
the selected projections; query width need not equal hidden width. The AdamW
example predicts the hidden row's mean at every projection output coordinate.
GatedDelta input reconstruction needs equal input and output widths; on
Qwen3-0.6B, its target is the value projection.

Graph parity uses identical padded model shapes for eager execution and capture.
Different prefill widths can change BF16 hidden inputs before a producer runs;
subsequent factor updates and near-tied greedy tokens can therefore differ.
Replay checks compare carry, normalization of the materialized tap and KV with native vLLM
at the same shape and paged history. The FP32 diagnosis establishes the numerical
error separately; cross-shape token equality is not the execution contract.

Prefix reuse and speculative decoding are unsupported here. When vLLM preempts a
request and recomputes its history, a workload whose request state is keyed by request id,
with no mutable instance state and no external feeds, imports or host inputs, simply
recomputes; any other overlay or activation workload raises and asks for more KV cache
(`gpu_memory_utilization`) or fewer concurrent sequences (`max_num_seqs`). Old KV retains the factors used when it was computed; updates
do not recompute the prefix. A separate SHINE context pass and checkpoint or
memory-token integration still need implementation. Finite-value checks may
synchronize CUDA.

## 外部生产的 overlay（SHINE 式）

**要解决的问题**：补丁字段不想在引擎里训练——它来自超网络、加载的
checkpoint，或任何一段宿主计算。

声明状态时设置 `imported=True`，例如 `pllm.RequestState(fields, imported=True)`；再声明一个 `Adapter` property
命名状态、targets、算子、因子与 scale；把产出的张量经
`generate(..., batch=pllm.state.StateImport(values, keys), prompt_indices=...)`
送入：

```python
@property
def adapter(self) -> pllm.model.Adapter:
    return pllm.model.Adapter("factors", targets,
                        operator="low_rank",
                        factors=("a", "b"), scale=0.5)
```

**这和引擎内训练的 overlay 有什么差别？** 只有生产地点不同。导入在
`request.prepare` 发布——即请求的第一次前向之前——且只导入到版本零的请求状态
一次；导入的零因子精确复现基座模型。之后的应用、发布、隔离、退役与引擎内
训练的 overlay 完全相同。优化器、超网络、checkpoint 是同一状态的三种生产
方式，消费端不区分。

## 激活检查点（ReFT 式）

**要解决的问题**：要改的不是权重，而是激活。

声明 `imported=True` 的 buffer 状态加一个 `pllm.model.ActivationPatch` property（命名状态、层、首/末 prefill
token、算子），同样经 `batch=` 送入。`LOW_RANK_AFFINE` 每层用
weight/bias/rotation 字段，`ADD` 用一个 FP32 向量。见维护中的
[ReFT](../../pllm/workloads/reft.py)。

`first_tokens/last_tokens` 仍只选择 prefill 行。逐 decode 通过三个 keyword-only
字段声明：`decode=True`、
`refresh="step"`、`on_missing="hold"`
（默认分别为 `False`、`"admission"`、`"stop"`）。`decode=False` 携带非默认刷新/缺席策略会在声明期拒绝。
INHERIT 保留原作用域的版本选择，STEP 每步读取本 invocation 新发布的 exact ref；
只有签署且成功验证的 HOLD 分支可以在没有新 publication 时读已成功的输入版本。
单独声明 HOLD 不会给任意缺根图添加保持能力，完整生命周期见
[状态与记忆](state-and-memory.md#没有新特征时签署的-hold--stop)。

ActivationPatch 可与 `custom_forward` 组合：注入在 DecoderSource 层输出之后、tap 之前，
normal/replay 使用同一 ref 和选择位；低秩算子用 replay 新 hidden 重算 delta，
ADD 不在已注入的结果上重复累加。HOLD 的 source 数学与正常输入相同，只改变
签署的版本来源；borrowed 输入生产组全部关闭，模型 READ 仍精确绑定 inject。
公共作者入口只放行状态/任务符号精确匹配签署 activation 链的组合；额外状态或
普通训练不能借此绕过 standalone 限制。REQUEST 组合保留原请求行，INSTANCE
仍通过实例目录取值。

干预遵循 forward-only KV：层后注入不改变该层自身已计算的 KV，可能影响更深层
和后续 token；旧 token 的 KV 不回改。同 invocation replay 仍重放本步版本，
不能换成较新版本。逐 decode 注入与 vLLM 抢占/历史重算互斥，ActivationPatch 与
prefix caching 仍拒绝组合；没有按注入版本 cache_salt 键控或历史版本重建。

## 组合边界：失败关闭

一个 `custom_forward` 可以独立执行，或与一个训练 overlay、一个 ActivationPatch 分别
组合。多个 overlay、外部导入 overlay 加路由、三方组合不在当前支持范围内。
后端在绑定或安装时拒绝不支持的组合。共享前缀缓存仍与 overlay/层路由不兼容。

精确语义（UpdateOutput 字段、changed 与发布、copy_fields、容量声明）见
[创作契约](../reference/authoring-contract.md#请求-overlay)。
