# 验证与发布检查

从仓库根目录运行。本检出使用
`/workspace/pllm/.venv/bin/python`，在 worktree 中工作时也用同一个。
实现期间按改动边界选择检查；完整清单是发布门禁，不是每次编辑后都要重跑。

## CPU、Rust 与静态检查

```bash
python -m maturin develop --offline
python -m pytest -q
cargo test --manifest-path rust/pllm_core/Cargo.toml
python tools/check_import_layers.py
python tools/check_region_authority.py
python tools/check_typing.py
```

这些命令使用激活的仓库虚拟环境。离线原生构建需要缓存的 Rust 依赖。
一份严格配置覆盖维护中的实现、基准场景与示例，要求零错误零警告。
在仅 CPU 主机上被跳过的 CUDA 测试不算 GPU 证据。导入门禁要求零运行时
违规，包括函数局部导入。没有 baseline/rebase 工作流；依赖规则直接在检查器
里评审。

## GPU 边界

在有 vLLM 0.27.1 与 Qwen/Qwen2.5-0.5B-Instruct checkpoint 的兼容 CUDA 主机上
顺序运行：

```bash
python test/gpu/vllm_esamp_smoke.py
python test/gpu/vllm_mrv1_smoke.py
python test/gpu/vllm_inplace_ttt_smoke.py
python test/gpu/vllm_tlm_smoke.py --mode online
python test/gpu/vllm_tlm_smoke.py --mode offline
python test/gpu/vllm_reft_smoke.py
python test/gpu/vllm_conditional_replay_smoke.py
python test/gpu/vllm_override_smoke.py
python test/gpu/vllm_recirculation_smoke.py
python test/gpu/esamp_cuda_graph_replay_smoke.py
python test/gpu/fttt_cuda_smoke.py
python test/gpu/enginecore_smoke.py
python test/gpu/enginecore_smoke.py --mrv1-captured
```

第一个场景在一个进程内先跑纯推理再跑附着的 ESamp。训练必须有非零
loss/update 计数，请求状态必须清理干净，回退必须为零。CUDA Graph 场景覆盖
训练核心，不是 vLLM 捕获模式的支持矩阵。

## 发布门禁

证据 39 由 pytest 收集：构建 sdist，从解出的源码归档构建优化发布 wheel 并
装进隔离目录。它验证 Python/原生模块出处，执行 Rust 编解码并检查张量依赖、
许可证打包与已退役包/插件的缺席。

从仓库根目录构建发行产物：

```bash
python -m maturin build --release
python -m maturin sdist
```

wheel 成功本身不授权发布。打 tag 或发布前，完成
[整体验收](../design/architecture-consolidation-10k-plan.md)：全部选中的 39 项
原始证据契约、组合证据 40/41 与采样反馈证据 42、共享/独立 LoRA 证据 43/44、
性能对比 45/46 与独立适配证据 47（26 个 pytest、8 个 Rust、13 条运行时命令与
3 个基准对比；50 次选中执行），外加运行时边界与 30k 源码上限。原先的
22k–30k 区间是估计，不是要填满的最小尺寸。不能从短正确性冒烟推断吞吐或
不支持拓扑的声称。这个门禁不授权提交、打 tag 或发布。
