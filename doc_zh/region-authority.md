# Region Authority v1

本文冻结 ExecutionRegion v1 边界。`tools/check_region_authority.py` 是对应的可执行门禁，
只扫描通用生产 Region 文件；测试和兼容 adapter 不会被误判为 Region core。

## 权责表

| 问题 | 权威 | Python 可以做 | Python 不得做 |
| --- | --- | --- | --- |
| 依赖、anchor window、Region 划分、materialization/lifetime、fence、内存 | Rust `compile_schedule` / `validate_schedule` | 传输 JSON、呈现诊断 | 重切/重排 Region，发明 boundary/lifetime |
| adapter、fullgraph、动态/静态 shape、autograd、fallback | Rust `RegionCompileCommand` | 按 enum 调用注册 adapter | 根据 task/callable/options 选择策略 |
| artifact/decision identity | Rust 已签名 decision | 绑定精确 key | 用本地事实重算 key |
| Physical DAG | 签名 Region record/physical descriptor | 绑定 Tensor/buffer/anchor/跨 Region fence | 读取 task membership、`compile_safe`、原始 `compile_policy` 后重决策 |
| typed extern | Rust implementation identity/capability | 解析精确注册 stager | 解析 implementation 字符串或选择 workload stager |
| state version | Rust typed effect/version publish | 审计观察到的 transition | 重排 effect 或伪造 publish |

`compile_safe` 是静态 capability evidence。改变 `CompileOptions.compile_policy` 时，只能改变
签名 compile command、artifact key、decision signature，不能改变 capability evidence。

## 稳定接口

| 接口 | 必须稳定的字段 | 失败行为 |
| --- | --- | --- |
| `ScheduleDecision` v3 / `ExecutionRegion` | id、anchor、依赖、materialization、effect、capture/stream/fallback、已签名 side wait、artifact key、`compile_command` | 未知、缺失、未签名、过期 v2 或非规范字段拒绝 |
| `RegionCompileCommand` | adapter、fullgraph、dynamic、有序 shape specialization、autograd、fallback | 不支持或编译失败直接报错，不重试 eager |
| `RegionCallableArtifact` | Region id/key、有序输入输出 key、精确 typed binding/effect、本地 callable、compiled marker、command | callable/key/binding 缺失或不一致时在执行前拒绝 |
| `BackendRegionBinderSlot` | 签名 implementation id/event context；一次 `stage` 后必须执行一次 `invoke`、`clear` 或 `abort` | 重复、过期或缺失 stage 拒绝；callback 失败后不重试 |
| `StateVersionBinding` / `StateVersionRegistry` | Rust state descriptor、本地 version snapshot、typed read/write/transition/publish effect | binding coverage 或 slot identity 不一致时拒绝；每个签名 mutation 必须观察到一次 version 增量 |
| physical primitive | 签名 plan/node/fence/buffer descriptor | 未知 node 或 command 不一致时拒绝 |

FFI 保持为 `compile_schedule(graph, timeline, options)` 和
`validate_schedule(graph, timeline, options, decision)`；callable、Tensor、stream、backend model object
都不能越过该边界。

## Fail-Closed 矩阵

| 条件 | 必须结果 |
| --- | --- |
| Region/compile-command 字段未知或遗漏 | decode/validation 时拒绝 |
| command、artifact key、shape specialization 或 signature 被篡改 | binding 前拒绝 |
| callable、stager、capability、residual binding 或静态资源缺失 | forward 前或签名 event 边界拒绝 |
| `torch.compile` 失败 | 传播错误，绝不重建 eager 路径 |
| shape miss | 请求 coordinator 新 specialization epoch，不改写当前 Region |
| Region 内 materialization 或 per-task fence | schedule/physical validator 拒绝 |
| state/version 不一致 | 拒绝 step，不能伪造 version |
| native declaration 请求 legacy-only 能力 | 拒绝选择，禁止一个 step 混合 authority |

## Native Workload 迁移模板

1. 用 `tllm.workload` 声明 typed input/output/effect/timing/affine 数据。
2. 先添加失败的 schedule-contract 测试，覆盖 Region boundary、effect、materialization、capability failure。
3. 断言 schema v2、完整 Region 覆盖、每个 compiled Region 都有 Rust-signed command。
4. Python 只绑定 declaration、Tensor/buffer、sink、residual path、精确 backend stager；不能创建 task plan、stream placement、event schedule。
5. 验证 state/version、request 隔离和 artifact coverage，再跑 GPU 正确性。
6. 一个 workload 必须完整 native 或完整 legacy adapter，禁止部分接管。

检查：每个 task 恰好属于一个 Region；跨 Region value 都有签名 lifetime；内部 value 不 materialize；
compiled Region 有 command/artifact；extern Region 有显式 null command/stager identity；资源缺失测试必须
fail closed。

## Typed Recirculation Core v1

Recirculation prefill 使用 Rust 签名的单请求、精确行数 stage artifact family。Core v1 将冷编译
family 上限设为 16 个 prompt row（或后端更小的 `max_prefill_tokens`）；观察到更长 span 时必须以
`specialization-required` 失败。扩展 family 必须启动新的 coordinator cold epoch，绝不能在 hook 内
编译或悄悄回退 eager。

每个 row specialization 只有一个签名 artifact。fusion legality、operation order、grouping kind 和
`release_after` 均由 Rust 派生；native request 不带 serial/wavefront 选择器。serial 实现只保留为数值
oracle 和测试 fixture。

## 诊断协议

`NativeWorkloadExecution.read_resource_diagnostics()` 返回 worker status 使用的稳定本地资源视图。
外层 key 是 workload id；内层把非空诊断名映射为有限 JSON 数值（`int` 或 `float`，拒绝 bool）。
诊断只用于观察，不参与调度、artifact identity、cache selection 或签名。reader 返回非法值时，status
请求直接失败，不能将其规范化或用作调度提示。MP status 在 immutable plan/artifact identity 旁返回同一份
经校验的 per-worker mapping。

## 冻结验证

Core v1 checkpoint 于 2026-08-25 使用
`/workspace/pllm/.venv/bin/python` 验证：Python 3.11.11、pytest 9.0.3、
torch 2.13.0+cu129、vLLM 0.27.1、maturin 1.14.1、Rust 1.97.1。

```bash
cargo fmt --manifest-path rust/tllm_core/Cargo.toml -- --check
cargo check --manifest-path rust/tllm_core/Cargo.toml
env LD_LIBRARY_PATH=/home/user/.local/share/uv/python/cpython-3.11.11-linux-x86_64-gnu/lib \
  cargo test --manifest-path rust/tllm_core/Cargo.toml --lib
/workspace/pllm/.venv/bin/python -m pytest -q
PATH=.venv/bin:$PATH /workspace/pllm/.venv/bin/python tools/check_typing.py
/workspace/pllm/.venv/bin/python tools/check_region_authority.py
```

结果：Rust `46 passed`；Python `1067 passed, 1079 subtests passed`；strict typing `0 errors`；
basic typing 相对 baseline 176 为 `163 errors`；authority gate 覆盖 20 个生产文件并通过。冻结 schema
示例包括：`test/workload/golden/schedule_v3_region.json` 冻结最大 compiled Region，
`test/workload/golden/schedule_v3_complex_region.json` 冻结 typed extern staging、state effect、
materialization 与 fence。schedule、binder/state、recurrence、worker reconstruction 与 authority
boundary 回归分别位于
`test/workload/test_schedule_contract_unit.py`、`test/runtime/test_backend_region_binders_unit.py`、
`test/runtime/test_native_workload_execution_unit.py`、`test/core/test_typed_recurrence_plan_unit.py` 和
`test/runtime/test_execution_region_core_v1_boundary_unit.py`。

新构建的 release wheel 已从独立解包目录成功 import，并运行 schema-v2
`compile_schedule`/`validate_schedule` static-isl smoke；扩展只依赖 libc、libgcc 和 loader，没有
动态链接 `libisl`。

在 `CUDA_VISIBLE_DEVICES=0`、离线 model cache、`--gpu-mem 0.15` 下，严格串行的正确性 smoke
全部通过：Region CUDA Graph、native Dummy MRv1/MRv2、native Recirculation MRv1/MRv2、native
ESamp in-process MRv1/MRv2、native ESamp EngineCore MP MRv1/MRv2。ESamp 的
loss/update/state/intervention 证据非零且 fallback 为零；Recirculation 观察到 typed layer replay 且
fallback 为零。本次冻结按要求没有运行吞吐 benchmark。

## 第二部分：Legacy 清单

| Surface | 迁移完成前允许角色 |
| --- | --- |
| `tllm.program` `@flow` / `@extern` | 兼容 declaration/lowering lane |
| `tllm.consumers.*` | 完整 workload adapter 或 benchmark baseline |
| legacy ESamp program/手工训练 | 显式迁移与 paired-benchmark baseline |
| Override、TTRL、FTTT、TLM、In-Place-TTT、GTTA | native workload authority；无 legacy consumer 或 Program 实现 |
| consumer stream/event/plan store/recurrence scheduler/shadow snapshot | legacy detail，不得成为 Region physical authority |
| vLLM model 专用 adapter | backend quarantine，只暴露 typed capability/binder contract |

删除 legacy 项需要单独迁移评审，保留数值/state 语义与完整签名 Region executor。
