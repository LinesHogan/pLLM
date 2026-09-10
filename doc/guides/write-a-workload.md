# 写一个负载

> **先决条件**：做完[教程 00](../tutorials/00_quickstart.ipynb)，知道 Workload
> 是"声明"、Rust 签名、托管运行时执行这个三段式。
>
> **本文回答**：我要让自己的数学在模型推理的某个时机运行，最少要写什么？

## 最小骨架

一个负载就是一个 `pllm.Workload` 子类，方法上的装饰器回答三个问题：

```python
import torch
import pllm

class MyProbe(pllm.Workload):

    @pllm.on(pllm.Phase.OBSERVE)          # 什么时候运行：观测时机
    @pllm.hook(0)                          # 数据从哪来：第 0 层块的输出
    @pllm.export(name="probe", every=256, max_cols=16)   # 结果去哪：有界导出
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden - hidden.mean(dim=-1, keepdim=True)
```

- **什么时候** —— `@pllm.on(phase)`。`Phase` 是签名时间轴上的窗口
  （`OBSERVE`、`BEFORE_SAMPLE`、`AFTER_SAMPLE`、`MODEL_UPDATE` 等），
  不是运行时钩子。
- **数据从哪来** —— 参数注解。单个普通 `torch.Tensor` 参数继承 `hook` 的来源；
  多来源用 `Annotated` 标记：`Hidden`、`Samples`、`Parameter(state, field)`、
  `StateIndex`、`StateVersion`、`Result` 等。来源歧义在**绑定时**报错，不靠猜。
- **结果去哪** —— 效果装饰器（`export`、`guide_sampling`、`add_logits`、
  `train`）。没有声明效果的数学不会被调度。

写好后直接交给工厂：

```python
with pllm.make_llm(ModelConfig("Qwen/Qwen2.5-0.5B-Instruct"), workload=MyProbe()) as llm:
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=8))
```

## 三条容易踩的规矩

1. **输入是借来的只读视图。** 不要原地修改输入张量；普通数学只返回新值。
   能写的只有你自己声明的状态（见[状态与记忆](state-and-memory.md)）。
2. **注解必须是真实类型对象。** 绑定器从不求值字符串注解；
   `from __future__ import annotations` 会把注解变成字符串，别在负载文件里用。
3. **声明即签名。** 冷绑定器会签名你的函数体、依赖与配置，并为实例快照它们；
   之后改动源码不影响已绑定的执行。反射式数学或未受检依赖需要显式原生绑定。
   要换一小段数学，用 `dataclasses.replace(workload, transform=fn)`
   （[可运行示例](../../examples/centered_hidden.py)），不要 monkey-patch 引擎。

## 选择器：让部分行参与

`@pllm.on(phase, rows="ready")` 里 `rows` 指向同类的一个方法，它用该操作
已声明输入的子集算出布尔掩码：

```python
def ready(self, versions: Annotated[torch.Tensor, pllm.StateVersion("head")]) -> torch.Tensor:
    return versions > 0
```

选择器本身也是签名图里的普通任务，不是执行期回调。它用**原始行**计，
不是前向压实后的位置。教程 05 用这个做"训练过一步才开始引导"的门。

## 下一步

- 要跨步/跨请求存东西 → [状态与记忆](state-and-memory.md)
- 要在 decode 途中训练、看 loss → [训练与观测](training-and-observation.md)
- 要改模型参数（LoRA 式） → [Overlay 与重放](overlays-and-replay.md)
- 每条声明的精确边界 → [创作契约](../reference/authoring-contract.md)
