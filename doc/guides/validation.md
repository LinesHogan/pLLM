# 验证与发布检查

这份指南回答三个问题：改完代码后该跑什么、什么时候必须跑 GPU、以及发布之前
还差什么。所有命令从仓库根目录运行，使用
`/workspace/pllm/.venv/bin/python`；在 worktree 里工作时也用同一个环境。

先建立一个心态：实现期间按改动边界挑选检查，不需要每次保存都全量重跑。本文
列出的完整清单是发布门禁，不是编辑回路。

## 我刚改完代码，先跑什么？

绝大多数改动的第一道防线是 CPU 测试加 Rust 测试。它们验证签名数学、状态
语义与执行边界——也就是仓库里不依赖 GPU 的那部分契约：

```bash
python -m maturin develop --offline
python -m pytest -q
cargo test --manifest-path rust/pllm_core/Cargo.toml
```

`maturin develop --offline` 重建原生扩展，离线构建依赖已缓存的 Rust 依赖；
如果它在全新机器上失败，先看 Cargo 缓存是否就位，而不是改网络配置。
pytest 或 cargo test 失败时，失败的测试名直接指向被破坏的证据契约——按
测试对应的证据编号修，不要绕过断言。

## 我动了模块边界或类型，还要加什么？

仓库有三条结构性门禁，改导入关系、执行区域归属或类型注解时必跑：

```bash
python tools/check_import_layers.py
python tools/check_region_authority.py
python tools/check_typing.py
```

导入门禁要求零运行时违规，包括函数局部导入——把 import 挪进函数体里「优化」
会被直接拦下。没有 baseline/rebase 工作流：依赖规则直接在检查器里评审，
不接受「先记下来以后修」。类型检查用一份严格配置，覆盖维护中的实现、基准
场景与示例，要求零错误零警告。

## 检查自己写的 workload

```python
from pllm import testing

result = testing.check(workload, model_config, neutral=disabled_workload)
print(result)
```

`neutral` 可省略。检查只在 CPU 上运行：读取模型配置，用固定的三个请求比较独立、合批、
倒序、其他请求输入扰动，以及混合 prefill/decode 和分块 prompt。它复用 runner 的行标签和
真实 planner/executor，比较方法输出、状态、采样干预和已交付的 export；浮点比较使用 FP32
`torch.testing.assert_close` 默认容差。checkpoint-backed state 还需要配置指向的本地权重。

失败抛出 `testing.ContractError`，列出所有差异，并区分方法混合请求行和框架错写目的行。
成功返回不可变的 `CheckResult(passed, skipped)`；`dataclasses.asdict(result)` 可转成 JSON。
共享训练后的依赖输出、export 的限行规则、host batch/feed 和需要模型的效果都有明确的
skip 原因。CPU 检查不验证 preemption、publish_lag/visibility 或模型里的实际应用；这些仍需 GPU 证据。

## CPU 全过了，能说明 CUDA 没问题吗？

不能。在仅 CPU 主机上被跳过的 CUDA 测试不算 GPU 证据；跳过不是通过。动了
执行路径、训练核心或后端附着，就必须在 GPU 主机上补跑边界场景。

前提：一台兼容的 CUDA 主机，装有 vLLM 0.27.1 和
Qwen/Qwen2.5-0.5B-Instruct checkpoint。顺序运行：

```bash
python test/gpu/vllm_esamp_smoke.py
python test/gpu/vllm_mrv1_smoke.py
python test/gpu/vllm_inplace_ttt_smoke.py
python test/gpu/vllm_tlm_smoke.py --mode online
python test/gpu/vllm_tlm_smoke.py --mode offline
python test/gpu/vllm_reft_smoke.py
python test/gpu/vllm_route_updates_smoke.py
python test/gpu/vllm_override_smoke.py
python test/gpu/vllm_recirculation_smoke.py
python test/gpu/esamp_cuda_graph_replay_smoke.py
python test/gpu/fttt_cuda_smoke.py
python test/gpu/enginecore_smoke.py
python test/gpu/enginecore_smoke.py --mrv1-captured
```

TLM 的同一个 smoke 接受 `--model`，始终使用 BF16。默认仍为
`Qwen/Qwen2.5-0.5B-Instruct`、ONLINE；常规完整验证只运行这个默认模型。
显存利用率设为 0.8，以容纳 7B 基座。
缓存好下列四个 checkpoint 后，可顺序覆盖两种训练时机：

```bash
for model in Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct \
             Qwen/Qwen3-0.6B NousResearch/Llama-2-7b-chat-hf; do
  for mode in online offline; do
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 10; done
    taskset -c 0-15 python test/gpu/vllm_tlm_smoke.py --model "$model" --mode "$mode" || exit 1
  done
done
```

每次检查一次训练发布、版本递增、非零因子、后续生成使用已发布因子，以及关闭后
状态释放；逐层融合 QKV 输出与独立 PyTorch 线性运算对照（`rtol=atol=0.02`）。

这些场景验证的不是「进程没崩」，而是行为契约：第一个场景在一个进程内先跑
纯推理再跑附着的 ESamp；训练路径必须有非零 loss/update 计数；结束时请求
状态必须清理干净；回退计数必须为零。任何一个不满足，即使输出看起来正常
也算失败——检查时先看这三项计数，再看输出内容。

注意 `esamp_cuda_graph_replay_smoke.py` 的定位：CUDA Graph 场景覆盖的是
训练核心，不是 vLLM 捕获模式的支持矩阵。它通过不代表某种捕获拓扑被支持。

## 我要发布，还差什么？

先构建发行产物并跑证据 39：

```bash
python -m maturin build --release
python -m maturin sdist
```

证据 39 由 pytest 收集：它构建 sdist，从解出的源码归档构建优化发布 wheel
并装进隔离目录，验证 Python/原生模块出处，执行 Rust 编解码，并检查张量
依赖、许可证打包与已退役包/插件的缺席。

但要明确：wheel 构建成功本身不授权发布。打 tag 或发布之前，必须完成
整体验收：全部选中的
39 项原始证据契约、组合证据 40/41 与采样反馈证据 42、共享/独立 LoRA 证据
43/44、性能对比 45/46 与独立适配证据 47——合计 26 个 pytest、8 个 Rust、
13 条运行时命令与 3 个基准对比，共 50 次选中执行——外加运行时边界与 30k
源码上限。原先的 22k–30k 区间是估计，不是要填满的最小尺寸。

两个常见误读，提前说清：不能从短正确性冒烟推断吞吐或不支持拓扑的声称；
通过这一门禁本身也不授权提交、打 tag 或发布——它是必要条件，不是许可。
