# 写代码时就看到成本

本文面向已经读过教程 03（ESamp，每请求在线训练）的读者：只需一行 `pllm.explain(workload, config)`，不用加载权重、不用占用 GPU，几秒钟就能在 CPU 上看到负载会不会进 CUDA graph、计算排在哪、pllm 要预留多少显存，以及预算不够时该改哪个参数。

## 为什么需要它

以前要知道一个负载会不会退回 eager 模式、要占多少显存，只能真正启动引擎：加载权重、分配 KV cache、捕获 CUDA graph，然后看 `llm.diagnostics().eager_reason` 或者等报错。

`pllm.explain` 用同一套声明、同一套准入规则和规划器，在 CPU 上把计划算出来给你看。它只读取模型配置（`config.json`）并在 CPU 上追踪张量形状，不加载模型权重，不初始化 CUDA，也不分配 GPU 显存。以教程 03 的 `MiniESamp` 为例，在 CPU 上首次调用耗时 2.9 s（包含读取配置），同一进程里第二次调用不到 0.1 s。

## 一个能进 CUDA graph 的负载：教程 03 的 MiniESamp

如果你想知道写好的在线训练负载能不能跑进 CUDA graph、每步计算怎么重叠，可以直接对教程 03 里的 `MiniESamp` 类调用 `pllm.explain`（类的定义见教程 03）：

```python
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import pllm

class Distiller(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.up(F.silu(self.down(x)))
        return F.layer_norm(y, y.shape[-1:])

@dataclass(frozen=True)
class MiniESamp(pllm.Workload):
    beta: float
    rank: int = 4

    @property
    def distiller(self) -> pllm.RequestState:
        return pllm.RequestState(module=lambda m: Distiller(m.hidden_size, self.rank), optimizer=pllm.AdamW(1e-3))

    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow) - deep).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(self, shallow: torch.Tensor = pllm.hidden(0), f: Distiller = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow)

config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_model_len=512, max_num_seqs=128,
                          max_num_batched_tokens=8192)
print(pllm.explain(MiniESamp(beta=0.5), config))
```

输出如下（节选，省略了 `kernels`、`timing` 与 `note` 的大部分行）：

```text
model    Qwen2ForCausalLM, hidden 896, 24 layers, vocab 151936, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  CUDA graphs: no effect needs eager
base     987922432 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0); one AdamW step per state per engine step; prompt rows train once per prefill chunk
  estimate / step (128 rows): 4587520 matmul FLOP + 0 projection FLOP; state read/write 14680576/14680576 bytes, optimizer 29361152 bytes (included), sampler 0 bytes
predict  guide_sampling  model.execute / main / captured  distiller (boundary, 1)
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
  estimate / step (128 rows): 1835008 matmul FLOP + 939524096 projection FLOP; state read/write 3670016/0 bytes, optimizer 0 bytes (included), sampler 1261406208 bytes
tap      model.layers.0[0], decode: 460800 bytes shared by loss, predict
tap      model.layers.23[0], decode: 460800 bytes shared by loss
state  scope  roots  bytes/root  reserved bytes [exact; state_page_budget_bytes]
distiller  request  514  114704  58957916  field 0: float32, ...
memory   peak 964919904 bytes funded by pllm [per-rank tensor reservations; excludes GPU-dependent BLAS workspace; execution_peak_budget_bytes]
         below 964919904 bytes tasks run inline; below 919036000 bytes planning fails
preempt  state keep; replay rejection: not applicable to kept state
host     0 syncs in traced workload math
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

按行对照就能读懂这份执行计划：

- **`forward`**：模型前向能整段走 CUDA graph，没有哪个效果要求 eager。
- **`base`**：基座模型每个 token 的稠密投影与 LM head FLOP 估计，不含对 KV 历史的 attention。
- **`member`**：你写的每个方法被安排在哪。一个方法可能对应多行，每行是它的一段计算（比如 `predict` 出现了两行）：
  - `anchor`（什么时候跑）：`token.commit` 表示这一步的 token 确定之后；`model.execute` 表示模型前向时；`state.publish` 表示状态发布新版本时。
  - `lane`（在哪条 CUDA stream）：`side` 是旁路 stream，和下一步 forward 重叠；`main` 是主 stream。
  - `graph`：`captured` 表示进 CUDA graph。
  - `state reads (origin, distance)`：读哪份状态、读到的版本离最新差几步。`loss` 读最新的（距离 0），`predict` 读的版本落后 1（这就是教程 03 里说的“第 N 步的更新在第 N+2 步被读到”带来的重叠）。所以对 `MiniESamp` 来说，`loss` 训练在旁路 stream、进 graph；`predict` 在主 stream、进 graph。
  - `estimate / step`：这段计算每步（按 `max_num_seqs` 行算）的 FLOP、状态读写字节和采样器字节估计。`predict` 的 `projection FLOP` 与 `sampler` 字节来自用 LM head 投影预测并引导采样。
- **`tap`**：从模型里取出的 hidden（第 0 层、第 23 层）和预留的缓冲大小（各 460800 字节）；第 0 层的那份由 `loss` 和 `predict` 共用。
- **`state`**：`distiller` 是请求级状态，预留 514 份（`roots`），每份 114704 字节，共 58957916 字节（约 56 MiB）；`[exact]` 表示这是精确值。514 是为 128 个并发请求和在途的版本预留的份数——比并发数多，因为训练中的新版本和正在被读的旧版本要同时存在。
- **`memory`**：由 `execution_peak_budget_bytes` 约束的 pllm 执行峰值（含状态页、任务工作区与结果、图池和 anchor 预留），为 964919904 字节（约 920 MiB）。第二行说明：预算低于 964919904 字节时，旁路任务改成在主 stream 上内联运行（不再重叠）；低于 919036000 字节时规划失败。
- **`host`**：负载的张量数学里没有 GPU 到 CPU 的同步（`0 syncs`）。
- **`timing`**：时机设置及其后果；`(default)` 表示你没改过默认值。
- **`note`**：这份报告不知道的事（见下文的限制说明）。

## 一个会退回 eager 的负载：共享的 LoRA

有些写法会让模型前向退回 eager 模式。例如把一份 LoRA 声明成所有请求共享的 `InstanceState`，再用 `pllm.model.Adapter` 加到第 12 层的 q/v 投影上。在启动引擎前用 `pllm.explain` 检查它，就能提前看到原因：

```python
class SharedLoRA(pllm.Workload):
    lora = pllm.InstanceState({"a": pllm.Field(lambda inputs, outputs: (4, inputs)),
                               "b": pllm.Field(lambda inputs, outputs: (outputs, 4))}, imported=True)
    apply = pllm.model.Adapter("lora", (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12)),
                               operator="low_rank")


print(pllm.explain(SharedLoRA(), config))
```

输出的前几行如下（`timing` 与 `note` 行与上面相同）：

```text
model    Qwen2ForCausalLM, hidden 896, 24 layers, vocab 151936, bfloat16 [exact, config.json]
parallel tensor_parallel_size=1; reservations are per rank
forward  eager: CUDA-graph overlays do not support adapters backed by shared InstanceState
base     987922432 FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]
member  effect  anchor / lane / graph  state reads (origin, distance)
apply  Adapter  model site or inactive method
  forward eager because apply: CUDA-graph overlays do not support adapters backed by shared InstanceState
state  scope  roots  bytes/root  reserved bytes [exact; state_page_budget_bytes]
lora  instance  6  22528  135224  field 0: bfloat16, ...
preempt  state reset; replay rejection: use request-owned keys containing request_id and remove mutable instance state
host     0 syncs in traced workload math
```

这里重点看两处变化：

- **`forward eager`**：读共享 `InstanceState` 的 Adapter 目前不在 CUDA graph 里运行，整个模型前向会用 eager 模式；报告点名是 `apply` 这个声明导致的。启动引擎后 `llm.diagnostics().eager_reason` 会给出同样的原因，但现在你在启动前就看到了。把同一份 LoRA 声明成 `RequestState`（每个请求一份），前向就留在 CUDA graph 里。
- **`model site or inactive method`**：Adapter 直接作用在模型的投影上，不是独立排进时间线的任务，所以没有 `anchor` 和 `lane`。

把两个负载放在一起对比：

| 负载 | 模型前向与执行方式 |
| --- | --- |
| `MiniESamp` | CUDA graph，训练在旁路 stream 上与后续前向重叠 |
| `SharedLoRA` | eager forward |

写之前就能权衡：要 CUDA graph 的吞吐，就避开会触发 eager 的写法；需要这些写法时，你知道代价在哪。

## 预算不够时，它告诉你改什么

如果你给负载设了显存预算，想知道当前配置能不能装下、装不下该调哪个参数，可以直接查看报告的 `budget_errors`：

```python
small = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_model_len=512, max_num_seqs=128,
                         max_num_batched_tokens=8192,
                         execution=pllm.ExecutionOptions(execution_peak_budget_bytes=64 << 20))
print(pllm.explain(MiniESamp(beta=0.5), small).budget_errors)
```

输出如下：

```text
('pllm execution needs 919036000 bytes at its peak but only 67108864 fit; lower max_num_batched_tokens or raise ExecutionOptions.execution_peak_budget_bytes; reduce device tables or explicitly declare placement="host" for read-only lookup tables',)
```

`budget_errors` 为空表示声明的预算装得下；不为空时，报告仍会给出不带这个限制的计划，并附上和 `make_llm` 相同的报错信息。如果是其他准入错误（例如不支持的组合），`pllm.explain` 会直接抛出异常，报错信息与创建引擎时相同。

## 它不知道的事

`pllm.explain` 在 CPU 上运行，有以下边界：

- 能不能装进你的 GPU，要到 `make_llm` 时才确定：报告里的显存不含模型权重、KV cache、vLLM 自己的激活和碎片。
- 投影和 scan 用到的 BLAS 工作区大小取决于 GPU，报告不计入；受影响的峰值是下界。
- FLOP 和逻辑访存字节是按形状算出的估计（自写算子给出下界）；没有吞吐或延迟估计——吞吐还是要实测（教程 03 的做法）。
- 同步次数只覆盖在 CPU 上追踪过形状的负载数学；作用在模型内部的计算（激活补丁、`custom_residual` 等）标为 `unverified`。
- 配置读取目前支持稠密 Qwen2、Qwen3 和 Llama 的投影形状；其他家族的投影准入在 `make_llm` 时判断。

## 给程序/代理用

如果你在写自动化脚本或让代理自动检查负载，不需要解析文本。`pllm.explain` 返回的对象可以用 `dataclasses.asdict(report)` 转成能够 `json.dumps` 的字典，包含 `eager_reason`、`members`（每个方法的任务）、`states`、`peak_memory_bytes`、`budget_errors`、`exports` 等字段；`workload_digest` 是负载声明、方法代码和取值的签名（相同即同一个负载），`flops_per_token_estimate` 是基座与各任务合计的每 token FLOP 估计：

```python
import json
from dataclasses import asdict

report = pllm.explain(MiniESamp(beta=0.5), config)
print(report.eager_reason == "", [state.reserved_bytes for state in report.states])
print(len(json.dumps(asdict(report))) > 0)
```

输出为 `True [58957916]` 和 `True`。
