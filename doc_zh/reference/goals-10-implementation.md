# 十项目标：实现方式与输入输出

> 日期：2026-08-17 · 对应 docs/8.17_todo.md 的六个 PR 重构 + ESamp 迁移 + 两个追加目标（模块化、性能）。
> 每个目标按「实现方式 / 输入 / 输出 / 验证」四段描述。

## 目标 1：BackendDriver 边界 + 版本探测（PR1）

**实现方式**：新建 `tllm/backends/vllm/` 作为唯一允许触碰 vLLM 内部的层。`detect.py` 做结构探测（runner class、方法签名、min-p 形态），版本字符串只用于初选候选族，结构不符即 fail fast；`v027_mrv1.py` / `v027_mrv2.py` / `legacy_mrv1.py` 三个 driver 各自封装自己版本的类 hook 安装。旧的全局 patch 代码迁入 legacy driver 作为过渡。

- **输入**：已安装的 vLLM 包（运行时探测其结构）。
- **输出**：`install_backend(core=...)` 完成后，`core`（residual_runtime 模块）上的 `GPUModelRunner` / `_ORIG_*` 被绑定，对应 driver 的 hook 就位；未知结构抛 `UnsupportedBackendError`。
- **验证**：`test/test_vllm_adapter_selection_unit.py`、`test/test_sampler_patch_unit.py`。

## 目标 2：v0.27 MRv1 driver 可运行 + ESamp 迁移（PR2）

**实现方式**：适配 vLLM 0.25+ 的三处结构变化：`_prepare_inputs(scheduler_output)` 新签名与 `(logits_indices, spec_decode_metadata)` 返回值；一个逻辑 step 横跨 `execute_model` + `sample_tokens` 两段（新增 sample_tokens.pre/post 生命周期，step 在 sample_tokens.post 才结束）；min-p 变成 vLLM 原生 argmax-invariant logits processor（删除旧 InputBatch patch，survivor 改为"过滤后有限 logit"语义）。`sampler_mrv1.py` 处理 `Sampler.sample` 新的 `(sampled, processed_logprobs)` 返回。

- **输入**：vLLM 0.27 MRv1 runner、ESamp consumer 配置。
- **输出**：MRv1 下 ESamp 训练 + 干预全流程可运行；无干预时与 raw vLLM 逐位一致。
- **验证**：`test/gpu/parity_harness.py --runner mrv1`（bit-exact）、ratio benchmark MRv1 臂。

## 目标 3：v0.27 MRv2 driver + StepSession + 采样拦截器（PR3）

**实现方式**：`runtime/step_session.py` 引入显式步级会话（CREATED→INPUTS_READY→FORWARD_DONE→SAMPLING→COMPLETED/FAILED），把 MRv2 散落在 `execute_model_state` 上的瞬态引用（InputBatch、hidden_states、slot_mapping）收敛成单一日志对象，上层不再关心 runner 家族。`sampler_mrv2.py` 用**实例级**委托拦截器替换 `runner.sampler`（不全局 patch Sampler 类），干预逻辑组织成四级 fallback 链（fused Triton → 候选集 torch → 全词表 reweight → 精确 dense）。kernel-warmup batch 在 decode localization 中跳过（3d7862d）。

- **输入**：vLLM 0.27 MRv2 runner（默认路径）、StepSession 生命周期事件。
- **输出**：MRv2 下与 MRv1 语义一致的 StepSession；采样干预只在受影响行生效，无干预时零开销直通。
- **验证**：parity harness MRv2 臂 bit-exact；`test/gpu/esamp_intervention_smoke.py`（β=0.5 下 31/32 命中）。

## 目标 4：Producer 一等插件化 + ExecutionPlanCompiler（PR4）

**实现方式**：`producers/base.py` 定义 Producer/BoundProducer 协议与 HookPoint；`producers/registry.py` 的 registry 保留同一 port 的全部候选 producer（不再后来者静默覆盖），`ExecutionPlanCompiler` 在启动期把供需声明编译成有序 DAG，并对「无 provider / 同优先级冲突 / 依赖环 / graph-safety 不满足」四类错误 fail fast。prefill/decode residual producer 迁入新接口。`ports/` 下 18 个 port 定义 + `PortId(namespace,name,version)` 构成类型化供需语言。

- **输入**：consumer 的 `flows()` 订阅声明 + 已注册 producer 集合。
- **输出**：`ExecutionPlan`（有序 DAG）；配置错误在 `make_llm` 时抛异常而不是运行时静默错。
- **验证**：`test/test_producer_plan_unit.py`、`test/test_port_catalog_unit.py`。

## 目标 5：TllmRuntime 实例化 + ESamp 配置剥离（PR5）

**实现方式**：`runtime/runtime.py` 新增 `TllmRuntime` 实例入口（`TllmRuntime(consumers=[...]).make_llm(...)`）；模块级 `make_llm`/`register_consumer` 降级为默认全局 runtime 的薄 façade；`bind_runner_runtime` 的 WeakKeyDictionary 留作将来 runner→runtime 路由接缝。ESamp 的配置（beta、hidden_dim、model bank、stream mode…）收敛到 `consumers/esamp/config.py` 的 `ESampConsumerConfig`，通用 `configure_runtime` 只保留 runtime 级旋钮。

- **输入**：构造期注入的 consumers/producers、vLLM engine 参数。
- **输出**：配置隔离的 runtime 实例；旧模块级 API 行为不变（兼容）。
- **验证**：`test/test_tllm_runtime_instance_unit.py`；全量 707 单测。

## 目标 6：CUDA graph 形式化能力 + speculative decoding 映射（PR6）

**实现方式**：`runtime/graph_policy.py` 把临时判断升级为显式契约：`GraphMode`（EAGER/PIECEWISE/FULL）解析、hook 启动许可判定、启动期对编译好的 execution plan 做模式校验。producer/consumer 声明 `GraphSafety`（EAGER_ONLY/CAPTURE_SAFE/REPLAY_STATEFUL），replay 期必须执行的工作只允许操作地址稳定的 persistent buffer。spec decode 下多 logits row → request/sample 建立严格映射（不再启发式猜测）。

- **输入**：runner 的 compilation_config / use_cuda_graph、plan 中各节点的 GraphSafety 声明。
- **输出**：启动期校验通过或报错；spec-decode batch 的每行正确归属到 (prompt_idx, sample_idx)。
- **验证**：`test/test_graph_policy_and_spec_mapping_unit.py`、`test/gpu/spec_decode_smoke.py`。

## 目标 7：其余 devlist consumer 保持可运行

**实现方式**：inplace_ttt / override / gtta / ttrl / fttt / tlm 六个 consumer 保证三件事：import 与注册不炸、`flows()` 契约满足新 dispatch 编译、CPU 单测全绿。GPU 路径在结构不符时明确 warn+skip 降级而非 crash；严格正确率本轮不追求（ESamp 是唯一完成 GPU 对齐的 consumer）。

- **输入**：各 consumer 的现有配置与 CPU 侧测试夹具。
- **输出**：六个 consumer 在新架构下可注册、可编译 plan、可降级运行。
- **验证**：`test/consumers/` 下各 consumer 单测 + `test_consumer_*_contracts_unit.py`。

## 目标 8：依赖声明更新

**实现方式**：`requirements-minimal.txt` 锁定 vllm 0.27.1（记录 cu129 wheel 源），torch 2.13+cu129 经 uv 的索引 pinning 安装；pyproject 声明同步。提交 13af7af。

- **输入**：目标 vLLM/torch 版本组合。
- **输出**：`uv sync` 一条命令复现环境；文档记录 CUDA 12.9 wheel 来源。
- **验证**：干净环境重建后 parity harness 通过。

## 目标 9：runtime 核心模块化拆分（追加目标①：易用性/可扩展性/简洁优雅）

**实现方式**：把 923 行的 `residual_runtime.py` 拆成专职模块，façade 只保留 hook 核心契约载体（hook 按模块寻址，故 `GPUModelRunner`/`_ORIG_*`/`_PATCH_INSTALLED` 必须留在原模块对象上）：

| 新模块 | 内容 | 规则 |
|---|---|---|
| `state.py` | ResidualRuntimeConfig/State + RUNTIME | 不得 import residual_runtime |
| `resources.py` | SamplerPrecompute 缓冲区/缓存/流 | 纯资源持有，无 hook 逻辑 |
| `stats.py` | 统计容器 + 采集器 | 显式传 state/consumer，不读全局 |
| `lifecycle.py` | patch 安装 + make_llm | core 模块显式传参 |

`__all__` 全量 re-export 保持旧 import 可用；`sampler_provider.py` 改为从 `resources` 直接导入作示范。提交 7c881b4。

- **输入**：原 residual_runtime.py 的实现代码与对外符号清单。
- **输出**：479 行 façade + 四个专职模块；所有旧 import 路径行为不变。
- **验证**：707 单测全绿；拆分后重跑双 runner ratio 无回退（见目标 10）。

## 目标 10：基准验证 ≥ README 旧版改进比例（追加目标②：高效率）

**实现方式**：新建 `test/gpu/esamp_ratio_benchmark.py`（raw/tllm 双臂、双 runner、best-of-3 轮次抗共享 GPU 噪声，模型 Qwen2.5-1.5B-Instruct，batch 8×16 样本×128 token）。性能工作的因果链：训练侧由 dual-stream 完全隐藏（training-only ratio 1.003）→ 干预侧瓶颈定位到每步 `nonzero()` host sync 抽干发射流水线（10.4ms/step）→ 改为 sync-free atomic compaction Triton kernel → fused minp+compact tile 并行 kernel → fused reweight+seeded-gumbel+argmax 单 kernel。候选路径还省掉了 baseline 的全词表 fp32 softmax/copy，因此可以反超 raw。

- **输入**：raw baseline JSON + tllm 臂 JSON（`run` 子命令产出）；`ratio` 子命令低于 0.9878 时退出码 1。
- **输出**：MRv1 **1.0076**（17936 vs 17801 tok/s）、MRv2 **1.0245**（20668 vs 20174 tok/s），均超过 README 旧版 0.9878；干预激活时候选路径 telemetry（candidate_sample_count 等）随 JSON 输出。
- **验证**：提交 7505738（优化）与重构后复测；parity 双臂 bit-exact 保证「快」没有牺牲「对」。
