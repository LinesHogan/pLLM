# 快速上手

## 环境要求

- Python ≥ 3.11、带 CUDA 的 PyTorch、vLLM v1（验证版本 `vllm==0.27.1`）。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
pip install -e ".[vllm027]"
```

## 运行 starter

```bash
python starter.py --max-new-tokens 32
```

这会以 `Qwen/Qwen2.5-7B-Instruct` 运行 ESamp（内置 test-time training consumer），并行生成 16 个回答，并打印自适应统计：训练 loss 计数、sampler 干预计数、port 发布命中数。

常用参数：

- `--seed-mode shared`（默认）保留 FlashInfer sampler 后端；`--seed-mode per-request` 给每个请求独立的 `seed + i` 流，代价是失去该后端。
- `--enable-distiller-intervention / --no-enable-distiller-intervention` 开关 sampler 引导。
- `--implementation program|legacy` 选择编译路径或手工调优基线。

## 在自己的脚本中使用

```python
from vllm import SamplingParams
from tllm import make_llm, register_consumer
from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig

register_consumer(ESampConsumer(ESampConsumerConfig()))
llm = make_llm(model_name="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.8, n=8))
```

- `make_llm(...)` 包装 `vllm.LLM` 并安装 tLLM 的 v1 runtime 钩子（多进程模式下入口插件会在每个 vLLM 进程中重建它们）。
- `register_consumer(...)` 挂载 consumer 实例；从配置构建可用 `tllm.consumers.esamp.activate(ESampConsumerConfig(...))`。

## 排错

- **CUDA Graph 没生效？** program 路径在 CUDA 上自动捕获整个训练步；CPU 按设计保持 eager。
- **启动时报 `PG###`？** 那是编译器拒绝不支持的结构 —— 见[错误码表](write-a-consumer.md#错误码)。这些是 fail-closed 的：任何降级都会显式报错，不会静默变慢。
- **吞吐 sanity check**：benchmark 在 tok/s 之外还报告功能计数（loss 数、干预状态数），坏掉的 no-op consumer 无法伪装成快。

下一步：[编写自己的 consumer](write-a-consumer.md)。
