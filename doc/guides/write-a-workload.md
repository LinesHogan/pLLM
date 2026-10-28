# 写一个负载

> **先决条件**：做完[教程 00](../tutorials/00_quickstart.ipynb)，知道 Workload
> 是"声明"、Rust 签名、托管运行时执行这个三段式。
>
> **本文回答**：我想让自己的数学在模型推理的某个时机运行，最少要写什么？

## 最小骨架：一个类回答三个问题

先直接看成品。一个负载就是一个 `pllm.Workload` 子类，方法上的三个装饰器
各回答一个问题：

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

这就是全部——没有 register、没有 hook 句柄、没有运行时回调。逐个问题看：

**什么时候运行**，由 `@pllm.on(phase)` 回答。`Phase` 是签名时间轴上的窗口
（`OBSERVE`、`BEFORE_SAMPLE`、`AFTER_SAMPLE`、`MODEL_UPDATE` 等），不是运行时
钩子：你在声明"这段数学属于这个窗口"，调度由引擎决定。

**数据从哪来**，由参数注解回答。这里 `hidden` 是唯一的普通 `torch.Tensor`
参数，它自动继承 `hook` 声明的来源。一旦需要多个来源，就要用 `Annotated`
显式标记：`Hidden`、`Samples`、`Parameter(state, field)`、`StateIndex`、
`StateVersion`、`Result` 等。来源有歧义会在**绑定时**报错，而不是静默猜一个
——这是刻意的设计，错要错在签名之前。

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

**注解必须是真实类型对象。** 绑定器从不求值字符串注解。这意味着
`from __future__ import annotations` 不能在负载文件里用——它会把所有注解变成
字符串，绑定器会直接拒绝。看到 "input source requires Annotated" 之类的报错，
先检查文件头有没有这一行。

**声明即签名。** 冷绑定器会签名你的函数体、依赖与配置，并为实例快照它们；之后
改动源码不影响已绑定的执行。反射式数学或未受检依赖需要显式原生绑定。所以"我
只想换一小段数学"的正确做法不是 monkey-patch 引擎，而是用
`dataclasses.replace(workload, transform=fn)` 换掉对应字段
（[可运行示例](../../examples/centered_hidden.py)）——换掉之后重新走一遍签名，
语义依然封闭。

## 选择器：让部分行参与

常见的下一个问题是：我不想每一步都处理所有行，比如"只对训练过一步的请求做
引导"。做法是给 `@pllm.on` 加一个 `rows` 参数，它指向同类里的一个方法，用
该操作已声明输入的子集算出一个布尔掩码：

```python
def ready(self, versions: Annotated[torch.Tensor, pllm.StateVersion("head")]) -> torch.Tensor:
    return versions > 0
```

把这个方法名传给 `rows="ready"`，对应的操作就只在掩码为真的行上生效。

两个容易误解的点。其一，选择器不是执行期回调——它本身也是签名图里的普通
任务，和别的方法一样被签名、被调度。其二，掩码用**原始行**计，不是前向
压实后的位置；你面对的是声明层面的行，不是内部打包后的索引。教程 05 正是
用这个做"训练过一步才开始引导"的门。

## 生产部署：EngineCore 多进程

到目前为止负载和引擎跑在同一进程里——这是默认行为（`multiprocessing=False`），
notebook 和随手脚本都走这条路。上生产时，`ModelConfig(multiprocessing=True)`
会把负载送进 EngineCore 子进程。这时第一个问题自然是：函数怎么跨进程？

答案是**按引用传输 + 签名复核**，函数体从不被序列化：

- 驱动端把负载编码为类坐标（模块 + 限定名）、模块源码摘要和冻结字段值；
- worker 进程 `import` 同一个模块、用字段值重建实例、重新冷绑定；
- 安装时，worker 重建出的编译签名必须与驱动端的签名计划**逐字节相等**，
  不等就拒绝安装。

所以"两个进程里跑的是不是同一份数学"是一个校验问题，不是信任问题：worker
不相信驱动端传来的任何代码，它只相信自己从同一源码重建后逐字节对上的签名。

代价是两条纪律。好消息是违反了会立刻报错，且报错会指明具体缺哪条：

1. **类必须可导入**：写在 `.py` 模块里。`__main__`、notebook cell、函数
   局部定义的类都无法被 spawn 出来的 worker import，留在同进程模式即可。
2. **可以被 pickle**：workload 用 pickle 送到 worker，类和模块级函数按导入路径传递。
   字段里夹了 lambda 或闭包会报错；不该按字节传的对象（如 TLM 的 meta 模型模板）
   用 `__reduce__` 只传配置，由 worker 自己加载。

最后一条隐性要求：模块必须 import-safe。spawn 会让 worker 重新导入你的模块，
顶层不能有 GPU 工作——否则它在 import 阶段就炸了，而不是在你以为的初始化
阶段。

## 下一步

- 要跨步/跨请求存东西 → [状态与记忆](state-and-memory.md)
- 要在 decode 途中训练、看 loss → [训练与观测](training-and-observation.md)
- 要改模型参数（LoRA 式） → [Overlay 与重放](overlays-and-replay.md)
- 每条声明的精确边界 → [创作契约](../reference/authoring-contract.md)
