# 在线服务：AsyncVllmLlm 与 pllm serve

`pllm.make_llm` 与 `pllm.make_async_llm` 共享同一套控制面能力：无论是离线评测、按批次导入外部张量，还是在线并发流式请求与实时参数更新，都可以组合使用。当服务需要持续接纳新到达的并发请求、按 token 流式推送、在客户端断连时立刻取消请求，或直接暴露 OpenAI `/v1/chat/completions` 接口时，使用本页的两个在线入口：

- **OpenAI 兼容 HTTP 服务**：命令行 `pllm serve`（或 `pllm.backends.vllm.serving.serve_openai_llm`）
- **异步 Python 引擎句柄**：`pllm.make_async_llm`（返回 `AsyncVllmLlm`）

## 前提条件

1. **负载放在可导入的 Python 模块中**，字段可被序列化（与多进程规则相同，见 [上生产：多进程、张量并行与量化底座](production.md#开启多进程enginecore-隔离与序列化要求)）。
2. **`multiprocessing="auto"` 默认隔离运行**：`ModelConfig` 默认 `multiprocessing="auto"`，传给 `pllm.make_async_llm` 与 `pllm serve` 时会自动启动独立的 EngineCore 子进程（无需再手动传 `multiprocessing=True`）。
3. **支持在途并发与动态更新**：`init=pllm.Imported()`、`init=pllm.Fed(...)`、`pllm.causal_model` 与 `pllm.batch` 宿主批次训练均可直接用于在线并发流；运行中的请求会读取自己准入时的状态版本，不会与后续的 `import_state()`、`train_batch()` 或 `feed()` 冲突。

---

## 第一步：编写负载并在 CPU 上检查

下面以一个按用户会话（`route`）隔离并跨轮次持续演化的在线蒸馏负载为例。相同会话下的请求共享同一份 `distiller` 状态（因此使用 `reduction="sum"`，原因见 [让一组请求共享训练成果](share-training-across-requests.md#共享组训练的归约建议为什么必须选-reductionsum)）：

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing


@dataclass(frozen=True)
class SessionDistiller(pllm.Workload):
    beta: float = 0.5
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
        key_fields=("route",),
    )

    @pllm.train(state="distiller", reduction="sum")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


# 启动 GPU 服务前，在 CPU 上核对 CUDA graph 状态与多请求契约
async_cfg = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=16)
plan = pllm.explain(SessionDistiller(), async_cfg)
assert plan.eager_reason == ""
print(testing.check(SessionDistiller(), async_cfg, neutral=SessionDistiller(beta=0.0)))
```

---

## 第二步：用 `pllm serve` 启动 OpenAI 兼容服务

将任意可导入的负载类或实例（格式为 `模块路径:类名或对象名`）挂载到 HTTP 端口：

```bash
pllm serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --workload my_project.workloads:SessionDistiller \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
```

使用标准 OpenAI SDK 流式调用。对于声明了 `key_fields=("route",)` 的负载，在 `extra_body` 中通过 `vllm_xargs={"pllm_route": ...}` 传入会话 ID，同一会话的后续轮次会自动接续上一轮训练出的状态：

```python
# check-docs: skip: connects to a running OpenAI server
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "用两句话解释分页注意力。"}],
    max_tokens=64,
    stream=True,
    extra_body={"vllm_xargs": {"pllm_route": "session-42"}},
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

> [!IMPORTANT]
> 按 `"route"` 分组的请求状态在当前轮结束后会转为休眠保留、供下一轮复用，仍占用 `max_num_seqs` 的状态槽位配额。上线前请按预期并存的会话基数留足 `max_num_seqs`，详见 [让一组请求共享训练成果](share-training-across-requests.md#3-路由状态的生命周期与容量规划)。

若负载仅使用默认的 `key_fields=("request_id",)`（如内置的 `pllm.workloads:DoLaWorkload` 或 `pllm.workloads:ESampWorkload`），客户端无需传 `pllm_route`，每条请求在完成或断连取消时立即归还状态页。

---

## 第三步：在异步 Python 服务中嵌入 `pllm.make_async_llm`

需要将流式推理嵌入自定义 FastAPI 路由、WebSocket 服务或异步任务编排时，直接使用 `pllm.make_async_llm`：

```python
# check-docs: skip: runs GPU async engine
import asyncio
from vllm import SamplingParams
import pllm
from my_project.workloads import SessionDistiller

config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=16,
    gpu_memory_utilization=0.5,
)


async def main() -> None:
    async with pllm.make_async_llm(config, workload=SessionDistiller()) as llm:
        async def ask(prompt: str, route: str) -> str:
            last = None
            async for chunk in llm.stream(prompt, SamplingParams(max_tokens=32), route=route):
                last = chunk
            return "" if last is None else last.outputs[0].text

        reply_a, reply_b = await asyncio.gather(
            ask("Explain gradient descent:", route="user-a"),
            ask("Explain KV caching:", route="user-b"),
        )
        print(reply_a, reply_b)


asyncio.run(main())
```

当调用方提前跳出 `async for ... in llm.stream(...)`（或上层协程因客户端断连被取消）时，`stream()` 会在退出时自动向引擎发出 `abort(request_id)`，释放该请求占用的 KV 块与非持久请求状态。对于不需要逐段流式消费的场景，可直接 `await llm.generate(prompt, params, route=...)`。

---

## 确认运行符合预期

在服务启动后或集成测试中，调用 `await llm.diagnostics()` 检查三项指标（各字段含义见 [explain 与 diagnostics 字段](../reference/diagnostics.md)）：

1. **`diag.eager_reason == ""`**：确认模型前向运行在 CUDA graph 中，没有静默退回 eager。
2. **`diag.inference_pins == 0` 与 `diag.pending_state_releases == 0`（无在途请求时）**：确认已完成和已取消的请求均已排空回收，没有状态页泄漏。
3. **`diag.active_states`**：对 `key_fields=("request_id",)` 的负载，空闲时应回到 `0`；对 `key_fields=("route",)` 的负载，应等于访问过的不同路由数。

---

## 在线模式支持范围与边界

| 写法 | 在线服务（`make_async_llm` / `pllm serve`） | 备注 |
| --- | --- | --- |
| hidden / 残差流 / 逐层 logits 读取、`@pllm.export` | ✓ | 异步有界导出，不阻塞主流 |
| `@pllm.add_logits`、`@pllm.guide_sampling` | ✓ | 按在途请求逐行生效；支持无负载推理与在线回调 |
| `RequestState`（含 `request_id`、`prompt_index`、`route`）、`@pllm.write`、`@pllm.train`、`@pllm.train_in_forward` | ✓ | 请求独立到达与退出，状态页自动分配与回收 |
| `ActivationPatch`、`Adapter`、`@pllm.custom_residual`、`@pllm.custom_attention`、`@pllm.custom_forward` | ✓ | `RequestState` 与共享 `InstanceState` 均保留在图内（`@pllm.custom_forward` 使用切片图） |
| `init=pllm.Imported()`、`init=pllm.Fed(...)`、`pllm.causal_model`、`pllm.batch` 宿主批次 | ✓（`make_async_llm` 控制面） | 支持在途 `import_state()`、`feed()` 与 `train_batch()`（也可按单次调用随请求附带 `state_import`）；与模型能力边界同见[支持矩阵](../reference/support-matrix.md) |

在线准入与运行期限制的报错原文、原因和改法，见报错目录的 [请求与运行时](../reference/errors.md#请求与运行时)。
