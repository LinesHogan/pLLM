# tLLM 软件架构规划

> 日期：2026-08-17 · 基线：vLLM 0.27.1+cu129 · 状态：8.17_todo 十项目标已全部落地后的现状盘点 + 下一步规划

## 1. 参照系：vLLM 为什么"职责清晰"

vLLM 的关键路径是严格分层的，每一层只和相邻层说话：

| 层 | 形态 | 职责 |
|---|---|---|
| 调度/API 层 | Rust（tokenizer/agg）+ Python（EngineCore、Scheduler） | 请求生命周期、批量调度、KV 分配 |
| Runner 层 | Python（MRv1 `gpu_model_runner` / MRv2 `gpu.model_runner`） | 组 batch、驱动 forward、持有 ExecuteModelState |
| 采样层 | Python 编排 + Triton kernel（`min_p.py`、`gumbel.py`，伪装成 .py 的 GPU 代码） | logits 处理与采样，host-sync-free |
| 算子层 | CUDA（attention/moe/quant）+ Triton | 计算密集型 kernel |

对 tLLM 的启示有三条：

1. **层与层之间是窄接口**——Scheduler 不知道 sampler 怎么实现，sampler 不知道 attention 用哪个 backend。tLLM 的对应物必须是 *类型化契约*（Port / Contract），而不是共享全局字典。
2. **热路径必须是 host-sync-free 的 GPU 代码**——vLLM 把 min-p / gumbel 都下沉成 Triton；tLLM 的对应物是 `sampler_bridge/kernels/`。任何每步执行的 Python/host 逻辑都是架构债。
3. **探测与适配集中在一处**——vLLM 用 `Platform`/`AttentionBackend` 注册表隔离版本与硬件差异；tLLM 的对应物是 `backends/vllm/detect.py` + driver + adapter 三层。

## 2. 现状：重构后的分层地图

```
┌─────────────────────────────────────────────────────────────────┐
│ 用户入口                                                          │
│   tllm.make_llm (façade) → TllmRuntime 实例 (runtime/runtime.py)  │
├─────────────────────────────────────────────────────────────────┤
│ Consumer 插件（算法层，不知道 vLLM 内部结构）                        │
│   consumers/esamp · devlist/{inplace_ttt,override,gtta,ttrl,     │
│   fttt,tlm} · dummy                                              │
│   声明 BaseConsumer.flows() → 订阅 Port、声明 GraphSafety          │
├─────────────────────────────────────────────────────────────────┤
│ Producer 插件（数据层，一等公民）                                    │
│   producers/{base,registry,residual} + execution_plan.py         │
│   ExecutionPlanCompiler：启动期编译 DAG，缺 provider/冲突/环 → 报错 │
├─────────────────────────────────────────────────────────────────┤
│ 契约层（稳定窄接口，唯一允许跨层引用的类型）                          │
│   contracts/{port_bundle,subscription,runtime_context,gpu_stage,  │
│   hidden_batch,request_meta_view,test_time_*}                    │
│   ports/*（PortKind/PortSpec/PortId + 18 个具体 port 定义）         │
├─────────────────────────────────────────────────────────────────┤
│ Runtime 核心（编排，不再堆积实现）                                   │
│   residual_runtime.py  — 薄 façade + hook 核心契约                 │
│   state.py             — 配置与可变状态 (RUNTIME)                   │
│   resources.py         — GPU 预计算缓冲区/缓存/流                   │
│   stats.py             — 统计容器与采集器                           │
│   lifecycle.py         — patch 安装与 make_llm                     │
│   step_session.py      — 跨 _prepare_inputs/execute_model/         │
│                          sample_tokens 的步级会话                   │
│   graph_policy.py      — EAGER/PIECEWISE/FULL 形式化策略           │
├─────────────────────────────────────────────────────────────────┤
│ Sampler 干预通道（热路径，host-sync-free）                           │
│   sampler_bridge/bridge.py      — 后端分发                          │
│   sampler_bridge/backends/      — torch 参考实现（可读性优先）       │
│   sampler_bridge/kernels/       — Triton kernel（性能优先）：       │
│     finite_compact_triton · minp_compact_triton                  │
│     （tile 并行、atomic 追加、seeded gumbel，零 host sync）         │
├─────────────────────────────────────────────────────────────────┤
│ Backend 绑定层（唯一允许 import vllm 内部的地方）                    │
│   backends/vllm/detect.py   — 结构探测，未知组合 fail fast          │
│   backends/vllm/v027_mrv{1,2}.py — driver（安装类 hook）            │
│   backends/vllm/sampler_mrv{1,2}.py — 采样拦截器                    │
│   runtime/vllm_patch/adapters/v0{10,11,27,72}.py — 版本适配器       │
└─────────────────────────────────────────────────────────────────┘
```

与 vLLM 分层对照：

| vLLM | tLLM 对应 | 差距 |
|---|---|---|
| Rust 调度核心 | 无（vLLM 自己的调度对我们够用） | 不需要 |
| CUDA 算子 | 无自有 CUDA；热路径用 Triton | 见 §4 规划 K1 |
| Triton kernel（.py 形态） | `sampler_bridge/kernels/`（3 个 kernel 模块） | 覆盖面小，需扩到 ESamp 训练路径 |
| Python 编排 | runtime/* + consumers/* | 已对齐 |
| 版本/平台探测 | `backends/vllm/detect.py` + adapter | 已对齐 |

## 3. 已确立的架构红线（十项目标沉淀下来的不变量）

1. **窄接口跨层**：consumer/producer 只见 contracts 与 ports，禁止 import `vllm.*` 或 `tllm.backends.*`；backend 层是唯一允许触碰 vLLM 内部的地方。
2. **核心无堆积**：`residual_runtime.py` 只是 façade + hook 契约载体（`GPUModelRunner`/`_ORIG_*`/`_PATCH_INSTALLED` 必须留在该模块对象上，因为 hook 按模块寻址）；实现必须下沉到 state/resources/stats/lifecycle。
3. **热路径零 host sync**：采样干预路径禁止 `nonzero()`/`.item()`/`.tolist()`；候选集操作用 atomic compaction kernel；survivor 数只能通过"有限 logit"语义推导，不能读回 host。
4. **启动期 fail fast**：ExecutionPlanCompiler 在 `make_llm` 时检查端口供需/冲突/环/graph-safety；运行时不再静默降级（结构不符只允许 warn+skip，禁止 crash）。
5. **graph 安全是声明式能力**：每个 producer/consumer 声明 `GraphSafety`（EAGER_ONLY / CAPTURE_SAFE / REPLAY_STATEFUL），`graph_policy` 在启动期校验，而不是靠 `_runner_uses_compilation_or_cudagraph` 这类事后猜测。
6. **实例作用域优先**：`TllmRuntime` 是推荐入口；模块级单例只保留为兼容 façade，runner→runtime 路由表（`bind_runner_runtime`）是后续多实例的接缝。

## 4. 下一步规划（按优先级）

### K1. Triton kernel 层扩容（对标 vLLM 的"Triton 伪装 py"层）

现状只有采样干预的 3 个 kernel 模块。ESamp 训练路径（fwd+MSE+backward）目前是纯 torch，每步约 99 GFLOP 量级的工作如果能融合还有空间：

- **K1a. Fused gather-project**：candidate gather + lm_head 行投影 + reweight 单 kernel 化（现在 = gather + bmm + 逐元素三段）。
- **K1b. Fused MSE + low-rank backward**：model-bank 训练的 fwd/loss/grad 融合，减少小 batch decode 步的 kernel 发射数（配合已有的 train CUDA graph 开关 `model_bank_train_cudagraph`）。
- **K1c. kernel 正确性基座**：每个 Triton kernel 必须有一个"逐位对齐的 torch 参考实现 + 随机化单测"（沿用 minp_compact 对 `_min_p_kernel` bit-exact 镜像的先例）。

### K2. 实例化收尾（PR5 的已知尾巴）

`runtime/runtime.py` 头部注释已写明：类 hook 每进程只装一次，当前所有 hook 派发到活跃 runtime。要支持同进程多 runtime：

- 把 hook 派发从"活跃单例"改成 `bind_runner_runtime` 查表（接缝已留好）。
- `state.RUNTIME` 全局单例降级为"默认 runtime 的 state"，stats/lifecycle 已参数化，改动面可控。
- 验收：单进程内两个不同配置的 `TllmRuntime` 各起一个 1.5B engine 互不串扰。

### K3. Producer 迁移收尾

`execution_plan.py` 目前只对已迁移的 port kind 做强制校验，legacy 硬编码 producer 仍在服务剩余 kind。规划：把 `runtime/ports/residual_bindings.py` 之后的所有硬编码路径迁入 `producers/`，然后打开全量 fail-fast。

### K4. 多卡契约（ShardSpec）

8.17_todo 附录已给出方向：所有 Port 带 ShardSpec，TP 下低秩 ESamp、PP/DP/EP 的 port 语义。当前全部实现都是单卡假设（atomic compaction、model bank slot 分配、precompute buffer 都是 per-device）。**在进入多卡前，先把 K1/K2 做完**——多卡会放大任何分层不清的代价。

### K5. 文档与可见性

- 本文件 + `docs/goals_10_implementation.md` 作为架构入口文档；
- `consumers/esamp/OPTIMIZATION_HISTORY.md` 继续记录性能决策（本次 sync-free 化的完整因果链已在其中）；
- 每个新 port/kernel 必须附"输入/输出/不变量"三行注释（沿 goals 文档格式）。

## 5. 不做清单（同样重要）

- **不引入 Rust/C++ 扩展**：tLLM 的热路径体量用 Triton 足够；引入编译型扩展会破坏 `pip install -e .` 的零构建体验，收益不成比例。
- **不复制 vLLM 调度器**：请求调度/KV 管理完全交给 vLLM；tLLM 只做 test-time 计算侧的插件。
- **不在 consumer 里出现 vllm import**：一经发现视为架构回归，code review 直接打回。
