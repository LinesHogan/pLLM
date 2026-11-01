# 写一个负载

> **先决条件**：做完[教程 00](../tutorials/00_quickstart.ipynb)，知道 Workload
> 是"声明"、Rust 签名、托管运行时执行这个三段式。
>
> **本文回答**：我想让自己的数学在模型推理的某个时机运行，最少要写什么？

## 最小骨架：一个类回答三个问题

先直接看成品。一个负载就是一个 `pllm.Workload` 子类，每个方法回答三个问题：

```python
import torch
import pllm

class MyProbe(pllm.Workload):

    @pllm.export(name="probe", every=256, max_cols=16)   # 结果去哪：有界导出（默认在观测时机运行）
    def forward(
        self,
        hidden: torch.Tensor = pllm.hidden(0),           # 数据从哪来：第 0 层块的输出
    ) -> torch.Tensor:
        return hidden - hidden.mean(dim=-1, keepdim=True)
```

这就是全部——没有 register、没有 hook 句柄、没有运行时回调。逐个问题看：

**什么时候运行**：每个效果有默认时机——导出在 `OBSERVE`，采样引导在
`BEFORE_SAMPLE`，训练在采样之后。要改时机，或方法没有效果时，加
`@pllm.on(phase)`。`Phase` 是时间轴上的窗口（`OBSERVE`、`BEFORE_SAMPLE`、
`AFTER_SAMPLE`、`MODEL_UPDATE` 等），不是运行时钩子：你在声明"这段数学属于
这个窗口"，调度由引擎决定。

**数据从哪来**，由参数默认值回答：`pllm.hidden(layer)`、`pllm.samples()`、
`pllm.parameter(state, field)`、`pllm.module(state)`、`pllm.index(state)`、
`pllm.version(state)`、`pllm.result(operation)` 等。每个张量参数都要写明来源，
类型注解只说明方法体拿到的值是什么。

**结果去哪**，由效果装饰器回答：`export`、`guide_sampling`、`add_logits`、
`train`。注意没有声明效果的数学不会被调度——返回值本身不构成效果，你必须
说清楚它去哪。

写好后交给工厂就行，不需要任何注册步骤：

```python
with pllm.make_llm(ModelConfig("Qwen/Qwen2.5-0.5B-Instruct"), workload=MyProbe()) as llm:
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=8))
```

`make_llm` 拿到负载后会冷绑定：编译签名、生成计划、交给托管运行时。你的
`forward` 从这一刻起就是被签名过的图节点，而不是一个随时会被调用的 Python
函数。

## 三条容易踩的规矩

骨架跑通之后，绝大多数绑定期报错都能归到下面三条。它们看起来苛刻，但每一条
都对应"声明即签名"这个核心约定。

**输入是借来的只读视图。** 不要原地修改输入张量；普通数学只返回新值。如果你
想"写点什么留到下一步"，那是状态，要写在自己声明的状态字段里
（见[状态与记忆](state-and-memory.md)），而不是写进别人的激活。

**状态 property 要写返回类型。** 形状依赖配置的状态写成 property 时，必须标注
`-> pllm.RequestState`（或 `-> pllm.InstanceState`），框架靠这个返回类型认出
状态声明；`from __future__ import annotations` 下的字符串注解同样可以。漏写时
`@pllm.train` 会报 "'adapter' is not a declared state"。

**声明即签名。** 冷绑定器会签名你的函数体、依赖与配置，并为实例快照它们；之后
改动源码不影响已绑定的执行。反射式数学或未受检依赖需要显式原生绑定。所以"我
只想换一小段数学"的正确做法不是 monkey-patch 引擎，而是用
`dataclasses.replace(workload, transform=fn)` 换掉对应字段
（[可运行示例](../../examples/centered_hidden.py)）——换掉之后重新走一遍签名，
语义依然封闭。

## 选择器：让部分行参与

常见的下一个问题是：我不想每一步都处理所有行，比如"只对训练过一步的请求做
引导"。做法是给效果装饰器（`guide_sampling`、`add_logits`、`train` 等）加
`rows="方法名"`，它指向同类里一个返回布尔掩码的方法：

```python
def ready(self, versions: torch.Tensor = pllm.version("head")) -> torch.Tensor:
    return versions > 0

@pllm.guide_sampling(beta=0.2, rows="ready")
def predict(
    self,
    h: torch.Tensor = pllm.hidden(0),
    w: torch.Tensor = pllm.parameter("head", "w"),
) -> torch.Tensor:
    return torch.bmm(h.unsqueeze(1), w).squeeze(1)
```

`predict` 只在掩码为真的行上引导采样。用在 `@pllm.train` 上时，方法仍在每一行
上运行，掩码只决定哪些行的 loss 计入。

两个容易误解的点。其一，选择器不是执行期回调——它本身也是签名图里的普通
任务，和别的方法一样被签名、被调度。其二，掩码用**原始行**计，不是前向
压实后的位置；你面对的是声明层面的行，不是内部打包后的索引。教程 05 正是
用这个做"训练过一步才开始引导"的门。

## 生产部署：EngineCore 多进程

到目前为止负载和引擎跑在同一进程里——这是默认行为（`multiprocessing=False`），
notebook 和随手脚本都走这条路。上生产时，`ModelConfig(multiprocessing=True)`
会把负载送进 EngineCore 子进程。这时第一个问题自然是：函数怎么跨进程？

答案是 **pickle + 安装时复核**：

- 驱动端用 pickle 编码负载，类和模块级函数按导入路径传递，不传函数体；
- worker 进程 `import` 同一个模块、反序列化出负载、重新声明一次；
- 安装时，worker 的声明必须与驱动端的计划一致，不一致就拒绝安装。

所以"两个进程里跑的是不是同一份数学"是一个校验问题：两端源码不一致，安装时
就会报错，而不是悄悄跑出不同的结果。

代价是两条纪律。好消息是违反了会立刻报错，且报错会指明具体缺哪条：

1. **类必须可导入**：写在 `.py` 模块里。`__main__`、notebook cell、函数
   局部定义的类都无法被 spawn 出来的 worker import，留在同进程模式即可。
2. **可以被 pickle**：workload 用 pickle 送到 worker，类和模块级函数按导入路径传递。
   字段里夹了 lambda 或闭包会报错。TLM 只传 `config/rank/alpha`；训练方法通过
   `pllm.causal_model("adapter")` 使用 worker 已加载的模型，不把模型放进负载字段。

最后一条隐性要求：模块必须 import-safe。spawn 会让 worker 重新导入你的模块，
顶层不能有 GPU 工作——否则它在 import 阶段就炸了，而不是在你以为的初始化
阶段。

## 下一步

- 要跨步/跨请求存东西 → [状态与记忆](state-and-memory.md)
- 要在 decode 途中训练、看 loss → [训练与观测](training-and-observation.md)
- 要改模型参数（LoRA 式） → [Overlay 与重放](overlays-and-replay.md)
- 每条声明的精确边界 → [创作契约](../reference/authoring-contract.md)
