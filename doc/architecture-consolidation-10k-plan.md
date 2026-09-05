# tLLM 架构合并、删除与精简计划

## 文档地位

2026-09-06 实施交接见[剩余工作清单](architecture-consolidation-handoff.md)：
代码检查点 `4cd162f` 后的工作区，SOURCE_LOC 27,961，终态 39 项证据尚欠 27 项。
旧 runtime/ports/common 和公共 hook/plugin 入口已删除；行数达标不代表功能验收完成。
历史 worktree 已从 30 个降至 3 个；过时文档与重复打包配置已清理，sdist→wheel 隔离验证通过。
下文基线与估算保留历史口径；当前完成情况以交接清单和 ledger 的最新检查点为准。

本文是从 2026-09-01 起执行的减法计划。它不把 10k 当作倒填配额，而是先根据当前
代码的可达性、重复职责和待完成功能估算可以删除的范围，再持续向 10k 靠近。

优先原则：

1. 不再以增加兼容层换取迁移进度；
2. 每个迁移 tranche 必须净删除生产代码；
3. workload 切换公共入口时，在同一 tranche 删除旧 authority；
4. 功能完整指公共语义和真实运行边界完整，不指保留旧内部结构；
5. 测试按真实 evidence item 计数，整个仓库不得超过 50 项；
6. 10k 是长期压缩方向，不是牺牲类型、功能或重新制造巨型文件的硬指标。

[`doc/research-aer-paged-state.md`](research-aer-paged-state.md) 仍是 PTTS/AER 的唯一
设计权威。该未跟踪研究文档不得因本计划被修改、加入或提交。

## 1. 当前基线

基线为提交 `12551b8`，统计采用受 Git 跟踪文件的物理行数。

| 范围 | 文件数 | 物理行数/测试数 |
| --- | ---: | ---: |
| 第一方 `.py`/`.rs`，排除 `test/` 与 `typings/` | 427 | 132,361 行 |
| 第一方 `tllm/_core.pyi` | 1 | 849 行 |
| **生产及支撑源码合计** | — | **133,210 行** |
| `test/` Python | 206 | 65,073 行 |
| Python `test_*` 函数 | 167 个文件 | 1,281 个函数 |
| Rust `#[test]`/`#[tokio::test]` | — | 54 项 |

参数化 collection 和独立 GPU smoke 尚未计入 1,281，因此当前真实测试项多于 1,335。

上述基线已于 2026-09-01 按第 3 节口径实测复核：pytest 实际 collect 1,349 项（加
Rust 54 项，合计 1,403）；`schedule.rs` 恰含 38 个 test；新栈
（canonical/contracts/canonical_runtime/canonical_vllm）对旧栈 import 为零，
clean-room 隔离成立，第 2.1 节的整块删除在机械上可执行。

当前 clean-room 相关代码已经达到 27,742 行：

| 范围 | 当前行数 |
| --- | ---: |
| `tllm/canonical/` | 16,580 |
| `tllm/contracts/` | 4,303 |
| `tllm/canonical_runtime/` | 2,253 |
| `tllm/canonical_vllm/` | 1,376 |
| `rust/tllm_core/src/canonical/` | 3,230 |

其中 canonical workload 已经出现明显重复：

| workload | 当前行数/文件 |
| --- | ---: |
| FTTT | 2,634 / 10 |
| GTTA | 1,180 / 9 |
| In-Place-TTT | 2,623 / 9 |
| TLM foundation | 231 / 2 |
| flat Dummy/ESamp | 2,868 / 9 |

GTTA 和 In-Place-TTT 已删除旧 consumer，仍然显著净增长。因此这不是普通双栈过渡，
而是 CPU/CUDA 状态机、plan DTO、application/session/execution 和 backend attachment 被
重复实现。

## 2. 能删多少：基于当前仓库的估算

### 2.1 可整块删除的旧 authority

以下是当前目录的实测值，不是目标倒推：

| 最终退役集合 | 当前行数 | 删除依据 |
| --- | ---: | --- |
| `runtime/workgraph/schedule/workload/program/consumers/kit/ports` | 73,578 | 最终全部由 canonical graph/execution/typed services 取代 |
| Rust 非 canonical 部分 | 18,715 | 主要是 `schedule.rs` 9,000、`workgraph.rs` 5,147、`recurrence.rs` 2,321 及旧 placement/plan |
| **旧栈 footprint** | **92,293** | `lib.rs` 等极小入口会改写，数量级不受影响 |

这批删除完成后，133,210 行会机械下降到约 40,900 行。旧 consumer 中真正独立的 tensor
数学、reward、environment 或 loader 会被抽到新 owner，因此 92,293 是 gross deletion，
不是最终 net deletion。

### 2.2 删除旧栈前后仍需新增的真实能力

当前 canonical 尚未包含完整的：

- AER regionization 和其 correctness oracle；
- closed typed MP transport；
- workload-independent profiling；
- TLM 完整 online/offline training/publication；
- ReFT、Recirculation、OverRIDE、TTRL 的最终 compact declaration/math；
- ESamp 完整 request PTTS/sampler/capture 终态。

从现有旧 consumer 中可识别的独立数学和 host implementation，加上上述尚缺核心，预计
还需 5,000–7,500 行。3,500 的下限要求独立数学全部复用且 AER/MP/profiling 极度紧凑，
按现有证据不现实，不再列入估算。也就是说，单纯删旧目录后，实际待精简的新栈规模会在
约 45,900–48,400 行，而不是 40,900 行。

### 2.3 当前可以点名的重复代码

下面是可以通过现有文件直接定位的重复，不依赖未来猜测：

| 重复集合 | 当前 footprint | 合并判断 |
| --- | ---: | --- |
| FTTT application/session/execution/tasks/workload/init | 1,580 | 数学、judge 和 host payload 之外应由 generic executor/host transaction 接管 |
| GTTA application/session/execution/tasks/workload/init | 765 | state/environment 保留，编排删除 |
| In-Place CPU/CUDA execution/tasks + application/workload/init | 2,164 | CPU/CUDA 只保留 provider kernel，不保留两套 transition protocol |
| ESamp request CPU/CUDA execution/tasks/control | 2,181 | 合并成同一 signed transition 和 provider implementation |
| `canonical_runtime` deployment/session/runner/worker/llm | 2,253 | 两条 workload 硬编码链（session/deployment/runtime/runner import ESamp，`parameter_overlay_*` import InPlaceTTT），是「不新增 workload-specific runtime」的现行违例；整条删除，由一个真 generic executor/backend seam 取代 |
| **可见重复编排合计** | **8,943** | generic replacement 仍需一份，因此不是全部净删除 |

这 8,943 行合并后仍需约 1,500–3,000 行 generic execution/attachment 代码，可信净删除为
约 6,000–7,500 行。

另外还有三组可压缩对象：

- Python/Rust backend、parameter-overlay、sampler attachment 的 envelope、digest、codec 和
  validation 重复，预计净删 1,000–2,000 行；其中实测 Python 侧有 15 个 `validate_*`
  函数（state_representation/backend_attachment/parameter_overlay_attachment/
  tensor_layout/scheduler/dense_page_validation）复制 Rust signed validation，删除由
  Tranche B item 5 执行，worker tamper 检测改调 Rust codec 验证；
- `tllm/backends` 与 `canonical_vllm` 当前合计 7,500 行，保留 MRv1/MRv2 与 best-effort
  旧版本后，预计能合并到 4,000–6,000 行，净删 1,500–3,500 行；
- common/producers、workflows、tools、examples 与顶层入口当前约 6,000 行（仓库中不
  存在独立的 facade 目录，原计划中的 facade 一项并入此组），旧调用者退役后预计保留
  2,000–3,500 行，净删 2,500–4,000 行；其中
  `workflows/benchmarks/per_request_esamp_benchmark.py` 一个文件 1,274 行，超过所有
  生产模块，点名瘦身或删除。

### 2.4 可信终态范围

按必须保留的职责自底向上估算：

| 终态职责 | 当前证据 | 预计终态 |
| --- | --- | ---: |
| Rust graph/interner/AER/scheduler/representation/FFI | canonical Rust 已有 3,230 行，但 AER 未完成；attachment 可去重 | 3,500–5,000 |
| Python typed schema/authoring/closed codec | contracts + canonical top-level 当前约 7,000 行，存在重复 validation/DTO | 2,500–4,000 |
| PTTS/generic execution/host/MP/profiling | PTTS + canonical runtime 当前约 6,300 行，MP/profiling 未完成 | 4,500–7,000 |
| 十个 workload 的声明、独立数学、loader | 当前六个 canonical workload 约 9,500 行，四个仍需抽取 | 4,000–6,500 |
| vLLM/backend adapters | old + canonical adapter 当前约 7,500 行 | 4,000–6,000 |
| 工具、workflows、必要 benchmark/release support | 当前约 6,000 行 | 2,000–3,500 |
| **总计外沿** | — | **20,500–32,000** |

考虑各项共享后的相关性，当前最可信的工程区间是 **22,000–30,000 行**，中心估计约
**26,000 行**。对应生产及支撑源码净删除约 **103,000–111,000 行（77%–83%）**。

因此：

- 30k 是第一轮完整迁移的失败线：超过它说明上述重复尚未真正合并；
- 22k–30k 是当前有代码证据支持的验收区间，不是固定配额；其中 22k 外沿依赖第 2.2
  节已判定不现实的新增下限，现实落点是 26k–30k；
- 达到该区间后继续逐 owner 做第二轮 reachability/duplication 审计；
- 10k 仍是长期方向，但当前没有证据证明在保留十个 workload、AER/PTTS、typed MP、
  profiling 和旧 vLLM best-effort adapter 的同时能达到 10k；
- 不得为了命中 10k 删除功能、弱化类型、把代码藏到测试/工具，或把职责压回巨型文件。

## 3. 不可规避的计量口径

`SOURCE_LOC` 统计所有第一方、受 Git 跟踪、手写的 `.py`、`.pyi` 和 `.rs`，排除：

- `test/`；
- 只描述第三方 API 的 `typings/`；
- 构建产物、缓存、文档和冻结 golden。

`tllm/`、Rust、tools、example、benchmark、backend adapter 和第一方 stub 全部计入。
不得通过移动目录、字符串模板、smoke 或 typing stub 规避统计。透明命令为：

```bash
git ls-files \
  | rg '\.(py|pyi|rs)$' \
  | rg -v '^(test|typings)/' \
  | xargs wc -l
```

每个 tranche 同时记录：新增、删除、净变化、移动而未改变的行数、抽取保留的独立数学，
以及与第 2 节估算的偏差。不能只报告最终总数。

## 4. 终态架构

```text
typed authoring descriptors
        |
canonical SemanticGraph
        |
Rust dense IDs / AER / scheduler / representation selection
        |
signed PhysicalPlan（机械投影）
        |
one generic typed Python execution layer
   |              |               |
PTTS provider   host services   closed MP wire
CPU/CUDA            |               |
   +----------------+---------------+
                    |
             thin backend adapter
```

硬约束：

- workload 只声明数学、数据流、effect、state 和 timing constraint；
- workload 不声明 lane、priority、provider、fusion 或手写执行顺序；
- Rust 是调度、regionization、placement 和 representation 的唯一 authority；
- PhysicalPlan 只投影 Rust 已签名决策，worker 不重选、不猜测、不 fallback；
- 新旧共存只在最外层 routing boundary，最终 routing boundary 也删除旧分支；
- Rust 拥有 graph/dense IDs/signed validation；Python 动态解析只在 Python–Rust codec；
- Python 其他目录不出现 `Any`、宽 `object`、`cast`、反射发现或字符串 type path；
- 运行时类型守卫只存在于 FFI decode 和公共入口边界；typed-internal 层禁止防御性
  isinstance/`type(x) is not` 守卫——strict Pyright 已保证调用方类型，实测约 80 处
  此类守卫（集中在 session/host_service/deployment/hidden_capture）随所属模块重构删除；
- contract DTO 只携带 typed 数据与 effect 声明，不携带 Callable/callback，也不定义
  「存参数 + 转发一次」的浅 wrapper（现行违例：`contracts/typed_replay.py` 的
  `replay: Callable`、`LayerReplayCall`、`FinalNormCall`，随 Recirculation 切换为
  typed row-selection + effect 声明后删除）；
- CPU/CUDA 共享 task/transition/state contract，仅 provider kernel 不同；
- request/global overlay 复用一个 versioned state capability；
- host service 复用 typed transaction/receipt，不按 workload 建 session；
- 最终不存在 Program、BaseConsumer、旧 WorkGraph 或 residual global workload authority。

大文件按 authority 拆分，而不是按行数机械切片。`schedule.rs` 等旧巨型文件冻结后删除，
不在原文件内大爆炸重写。任何新模块必须拥有不可合并的独立职责；只转发一次调用、复制
参数或重导出符号的浅 wrapper 不得存在。

## 5. 测试清理

### 5.1 计数规则

`TEST_COUNT` 是以下总和：

- pytest 实际 collected items，参数化 case 分别计数；
- `cargo test -- --list` 列出的 Rust tests；
- pytest 外 correctness smoke；
- benchmark 中兼任 correctness gate 的模式。

skipped、xfail、改名为脚本或藏进 helper 仍计数。静态 typing/import/LOC gate 不计为行为
测试。最终硬约束是 **`TEST_COUNT <= 50`**；当前明确选择 39 项终态证据，为尚未发现的
真实 runtime boundary 留 11 项余量。余量不能预先用装饰性测试填满。

### 5.2 预计删除量

当前已有至少 1,335 个声明测试和 65,073 行 Python tests。压到最多 50 项意味着删除至少
96% 的行为测试项。按保留完整 lifecycle smoke 的实际长度估计，测试/oracle 最终约
3,000–7,000 行，可净删约 58,000–62,000 行。行数是 forecast，50 项才是硬门。

### 5.3 明确保留的 39 项证据

这里列的是终态 evidence item，不是原样保留 39 个现有文件。一个现有文件中的数十个
case 会被合并为一个观察稳定边界的 lifecycle/oracle；路径会随旧栈删除而变化。

#### Rust canonical core：8 项

1. `layout_and_dense_ids_are_canonical_and_tamper_evident`：TensorLayout、LayoutId、dense
   interner 的确定性及篡改拒绝。
2. `semantic_graph_closed_codec_rejects_schema_drift`：SemanticGraph closed codec、签名、
   unknown/missing/duplicate fields。
3. `scheduler_matches_exhaustive_small_dag_oracle`：通用 scheduler 与 exhaustive 小 DAG
   oracle 一致。
4. `aer_matches_bruteforce_partition_oracle`：AER regionization 与 brute-force partition
   oracle 一致。
5. `representation_selection_and_capacity_are_signed_once`：representation/provider 只在
   Rust 选择一次，capacity facts 进入 signed decision。
6. `physical_plan_is_mechanical_and_reconstruction_is_exact`：PhysicalPlan 机械投影，
   reconstruction 拒绝篡改。
7. `versioned_state_hazards_and_timing_match_oracle`：RAW/WAR/WAW、state version 和 timing
   constraints 的小图 oracle。
8. `distributed_placement_transfer_matches_small_topology_oracle`：只保留一个最小
   distributed placement/transfer topology oracle。

现有 `rust/tllm_core/src/canonical/{tensor_layout,scheduler,state_representation}.rs` 的三个
测试分别作为第 1、3、5 项基础。旧 `schedule.rs` 的 38 个测试不保留；其中仍有效的
算法性质只允许并入上述 oracle。

#### PTTS：6 项

9. `cpu_request_ptts_admit_cow_publish_pin_retire`：CPU request state 的 admission、COW、
   atomic publication、exact pin、retire 完整生命周期。
10. `multi_lineage_admission_and_publication_roll_back_atomically`：capacity/factory/transition
    失败不得产生部分 admission 或 publication。
11. `global_multifactor_adamw_matches_oracle_and_publishes_one_version`：TLM A/B factor、
    gradient、step、moments 的数值 oracle 和 multi-root publication。
12. `generation_receipt_blocks_aba_and_double_settlement`：generational identity、issued
    receipt、slot reuse 和 one-shot settlement。
13. `cuda_ptts_waits_for_exact_fence_before_reclaim`：CUDA COW、pin release、completion fence
    和 page reclaim。
14. `cuda_graph_pagebundle_keeps_stable_addresses_across_versions`：capture/replay 期间
    PageBundle 地址稳定而 logical version 正确变化。

第 9 项以 `test/core/test_canonical_dense_state_producer_unit.py` 为基础；第 13 项以
`test/gpu/test_canonical_cuda_dense_page_smoke.py` 为基础。旧 request-slot、paged-state、
parameter-overlay lifecycle 测试只抽取未覆盖反例，随后删除。

#### Python–Rust/runtime boundary：5 项

15. `signed_attachment_binds_exact_tasks_fields_stages_and_rejects_tamper`：合并 backend、
    sampler 和 parameter-overlay attachment 的 exact binding/tamper proof。
16. `closed_mp_codec_round_trips_registered_payload_and_shared_tensors`：closed typed payload
    与共享 tensor 往返，明确证明无 type-path import/reflection。
17. `mp_transaction_commit_abort_and_quorum_failure_are_atomic`：prepare/commit/abort、worker
    quorum 和 fail-stop 在一个生命周期中证明。
18. `worker_reconstructs_exact_signed_plan_and_rejects_local_drift`：worker 只重建及验证，
    不进行 local selection/fallback。
19. `typed_host_transaction_abort_requeue_and_postpublish_settlement`：generic host service 的
    pre-commit abort/requeue 和 irreversible publication 后 settlement。

`test_canonical_backend_attachment_unit.py` 与
`test_parameter_overlay_attachment_unit.py` 合并为第 15 项，不再各自维护一套数百行 DTO
fixture。旧 MP control/port transport/worker reconstruction 测试只抽取第 16–18 项需要的
边界反例。

#### 每个 workload 一个语义 oracle：10 项

20. `dummy_runs_graph_sign_plan_execute`：最小 SemanticGraph 到 execution 路径。
21. `fttt_matches_edit_oracle_and_settles_judge_retry`：edit 数值、judge、retry、receipt。
22. `gtta_updates_environment_and_rolls_back_fifo`：environment FIFO、update/query、abort
    requeue。
23. `inplace_ttt_matches_multitarget_oracle_and_cleans_versions`：exact toy、多 target、
    publication、cleanup。
24. `tlm_matches_objective_online_offline_graph_and_qv_publication`：官方 objective、q/v cold
    facts、online/offline timing 和 LoRA publication。
25. `esamp_predicts_before_training_and_isolates_request_versions`：固定 seed、request
    isolation、sampler guidance、版本和 cleanup。
26. `reft_matches_loreft_math_and_publish_skip_semantics`：LoReFT 数学、publish 和无更新分支。
27. `recirculation_selects_rows_and_replays_typed_effect`：row selection、layer replay 和版本
    lifecycle。
28. `override_matches_reweight_adam_and_final_noop`：reweight、Adam update 和 final no-op。
29. `ttrl_matches_reward_update_retry_and_control_quorum`：reward selection、conditional update、
    retry/control output。

现有六个 canonical workload unit 测试（其中 TLM 为
`test/core/test_canonical_tlm_foundation_unit.py`）是第 20–25 项的基础。ReFT、
Recirculation、OverRIDE、TTRL 只从旧 consumer tests 冻结数学 oracle/golden；不保留旧
runtime/class hierarchy assertions。

#### 真实程序–运行时边界：10 项

30. `public_mrv2_esamp_trains_pins_sampler_and_cleans_request`：公共入口上的真实 hidden
    capture、sampler intervention、PTTS、非零训练和 cleanup。
31. `mrv1_adapter_captures_exact_rows_and_sampler_links`：只验证 MRv1 与 MRv2 真正不同的
    adapter path，不复制完整 workload matrix。
32. `public_vllm_dense_overlay_pins_two_request_versions`：In-Place-TTT DENSE_ADD、两个请求、
    exact version pin/release。
33. `public_vllm_low_rank_overlay_pins_and_releases_global_version`：TLM LOW_RANK_ADD、global
    publication、admission pin 和 cleanup。
34. `public_vllm_reft_applies_activation_patch`：真实 activation patch adapter。
35. `public_vllm_recirculation_replays_signed_layer_rows`：真实 typed layer replay adapter。
36. `enginecore_mp_executes_closed_plan_and_returns_quorum`：真实 child worker、closed wire、
    exact plan 和 quorum。
37. `cuda_graph_replay_preserves_training_and_zero_fallback`：capture/replay 地址与非零
    loss/update，fallback 为零。
38. `fttt_cuda_host_receipt_crosses_stream_boundary`：typed host receipt 与 CUDA stream/fence
    边界。
39. `isolated_wheel_imports_and_executes_rust_codec`：fresh wheel、隔离目录 import 和 Rust
    codec execution。

当前 `canonical_vllm_esamp_smoke.py`、`canonical_vllm_inplace_ttt_smoke.py` 和
`canonical_fttt_cuda_smoke.py` 分别是第 30、32、38 项基础。旧 MP/TLM/ReFT/
Recirculation smoke 只能抽取场景，必须改为新公共入口后才可进入 survivor list。

GTTA 是 CPU host-service workload，Dummy 不拥有 backend，二者不增加 GPU smoke。
OverRIDE/TTRL 若最终只消费已经被上述证据覆盖的 generic native task/typed port，也不各自
复制 vLLM smoke；只有发现新的、不可由通用边界覆盖的 adapter mechanism 时，才能使用
11 项余量之一。

### 5.4 明确删除范围

在对应 oracle/golden 抽取完成、生产入口切换后，按 workload 切换提交逐个删除
（见 Tranche A/C），而不是一次性删除：

- `test/runtime/`；
- `test/workgraph/`；
- `test/program/`；
- `test/workload/`；
- `test/consumers/`；
- 不在第 30–39 项中的旧 GPU smoke；
- import-layer/region-authority gate 的 meta-tests，CI 直接执行静态 gate；
- Python 重复 Rust validator 的 tests；
- typed-internal 层的防御性运行时类型守卫及其测试（strict Pyright 之外的装饰性
  防御，含 `reportUnnecessaryIsInstance` 抑制配 "untyped callers" 注释的模式）；
- dataclass、enum、`repr`、getter、re-export 和简单 mapping tests；
- 只断言 mock 调用次数、兼容 wrapper、registry 或旧 class hierarchy 的 tests；
- MRv1/MRv2 × in-process/MP × CPU/CUDA × capture 的机械矩阵；
- 没有 numerical/state oracle 或 loss/update/version/cleanup 证据的 smoke。

第 39 项完成后，survivor manifest 必须列出每一项的稳定 observable contract、运行环境、
是否需要 GPU/MP，以及它替代了哪些旧测试文件。训练证据必须有
`loss_count > 0`、`update_count > 0`；所有运行证据必须有 `fallback_count == 0`。

## 6. 实施 tranches 与实际减量

### Tranche A：测试清单、oracle 抽取与死路径测试删除

1. 从 public API、CLI、entry point、动态 import、文档和 benchmark 建 capability/reachability
   清单。
2. 为十个 workload 只列 observable behavior，不列旧内部类和调用顺序。
3. 先产出旧 consumer 的逐文件 keep/kill manifest，把独立数学、loader、judge、
   environment 的唯一新 owner 定下来；第 2.2 节的抽取预算以这份 manifest 记账。
4. 从旧测试提取最小 oracle/golden，形成第 5.3 节明确列出的 39 项 survivor。
5. 删除生产路径已死亡的测试（已退役 workload 的旧测试、meta-test、装饰性测试）、
   fixture、mock helper 和只为旧测试存在的生产 seam。
6. 仍在服役的旧 workload（ESamp、FTTT、ReFT、Recirculation、OverRIDE、TTRL、TLM）
   的旧测试保留到其 canonical 切换提交，随旧实现原子删除；不得先删测试让活路径
   裸奔。
7. 增加一个统一 test/LOC ledger，同时跟踪 TEST_COUNT 与待删保留项；不为 workload
   建 wrapper。

验收：keep/kill manifest 完整，survivor 每项均有高价值证据覆盖；死路径测试删除
完毕；没有被改名隐藏的 smoke；本 tranche 生产代码不净增长。`TEST_COUNT <= 50`
自 Tranche E（旧栈删除）完成起强制，此前只要求单调下降且 ledger 可解释每一项的
去留。

### Tranche B：压缩已经切换的 canonical verticals

1. 合并 In-Place-TTT 和 ESamp 的 CPU/CUDA execution/tasks，差异只留在 provider kernel。
2. 删除 FTTT、GTTA、In-Place-TTT 的 application/session/execution 编排，改由 generic
   executor/host transaction 消费 typed workload。
3. 合并 parameter-overlay 的 deployment/session/runner/worker/llm 转发链。
4. 合并 backend/parameter/sampler attachment 的 envelope/digest/codec/validation plumbing，
   但不建立万能 callback bag。
5. Rust 只验证一次；Python 不复制 semantic validation；PhysicalPlan 不重新选择。
6. PTTS CPU/CUDA publication 共用 lifecycle/ledger。

基于当前保留数学/state/host 内容，预计而非强制倒填的结果：

| workload | 当前 | 合并后可信范围 | 预计净删 |
| --- | ---: | ---: | ---: |
| FTTT | 2,634 | 400–700 | 1,900–2,200 |
| GTTA | 1,180 | 300–500 | 680–880 |
| In-Place-TTT | 2,623 | 400–700 | 1,900–2,200 |
| TLM foundation | 231 | 完整实现 600–900 | 新增 370–670，但删除旧 TLM consumer 抵消 |

加上 `canonical_runtime` 合并，本 tranche 预期净删约 5,000–7,000 行。偏离范围时必须说明
是哪一项真实语义或上游 backend seam 造成，而不是通过补 wrapper 达标。

### Tranche C：剩余 workload 切换即删除

1. TLM 只增加 loader/objective/graph，复用 global LOW_RANK_ADD、version pin 和 generic
   executor；online/offline 由 timing edges 表达。
2. ESamp 复用 request PTTS/sampler/generic execution，不迁移旧 init，使用随机初始化。
3. ReFT/Recirculation 只保留 activation/replay 数学与 typed effect。
4. OverRIDE/TTRL 只保留数学、state transition 和 typed output，不迁移 Program lowering。
5. Dummy 只证明最小 graph/sign/plan/execute 路径。

全部十个 workload 的可信终态是 4,000–6,500 行，其中 ESamp/TLM/TTRL 明显比 Dummy
复杂，不能再用平均 200 行的倒填预算。每个 workload 必须在同一提交完成：最小证据、
公共入口切换、旧实现删除、旧测试删除和动态引用清理。净减少按整个 tranche 验收
（第 8 节规则 1），不按单次切换提交验收：实测旧 consumer 中 TLM 仅 491 行、ReFT
321 行、OverRIDE 235 行，其新声明加独立数学可能接近甚至超过旧 consumer 行数，差额
由本 tranche 内其他 workload 的删除抵消；不允许为凑单提交净减少而压缩功能或拆假
提交。

### Tranche D：vLLM adapter 解耦重写（旧栈删除的阻塞项）

实测 `tllm/backends` 20 个文件中 16 个（5,528/6,124 行）import 旧
runtime/workgraph/program。这批 adapter 不脱离旧 authority，第 2.1 节的整块删除就
无法执行。本 tranche 把 adapter 重写为 typed backend quarantine：

1. adapter 不拥有 workload/调度/版本选择/fallback authority；
2. MRv1/MRv2 与旧版本结构共享 probe 和 translation，不为每个版本复制
   runner/session；
3. adapter 只消费 canonical typed seam，对旧栈 import 归零；
4. adapter 实际保留规模根据 probe 后记账，预计 4,000–6,000 行，不预设 1,500 行这种
   无依据上限；这是重写而非保留，工作量按 seam 记账，不允许把旧 runtime 代码搬进
   adapter 充数。

验收：`tllm/backends` 对旧栈 import 为零；MRv1/MRv2 公共入口 smoke 通过。本
tranche 允许因解耦 seam 出现的小幅净增长，但必须点名归因到具体 seam。

### Tranche E：整体删除旧技术栈

最后一个调用者归零（含 Tranche D 的 adapter 解耦）后，直接删除第 2.1 节的 73,578
行 Python 旧 authority 和约 18,700 行 Rust 旧 authority，不在原巨型文件中重构。

独立 tensor 数学、tokenizer、environment、judge、checkpoint 和 numerical oracle 先移动到
唯一新 owner；移动后原文件立即删除，不留 compat import。

验收：生产树中 `Program`、`BaseConsumer`、`DispatchPlan`、`ExecutionPlan`、
`consumer_compat`、旧 WorkGraph 和 residual workload authority 引用为零；
`TEST_COUNT <= 50` 自本 tranche 完成起强制。

### Tranche F：只在精简核心上补齐终态设计

- AER 替换 provisional scheduler，不与旧 WorkGraph 并存；
- typed authoring descriptors 替换 string selector，不保留双 facade；
- closed typed MP codec 替换 type-path/reflection transport；
- profiling surface 替换散落 counters/hooks，不建立 workload callback；
- PageBundle/generational arena 替换 provisional dense_page 家族（13 文件 / 4,062 行，
  按 device×concern 机械切片，含 types-only 浅模块）；替换前 dense_page 冻结禁投，
  不做中间态拆分或精致化重构；
- representation/provider facts 只在 specialization/physical facts，不进入 SemanticGraph。

新增能力必须在同一 tranche 删除其替代对象。若 tranche 净增长，说明 replacement boundary
尚未选对，应重新合并而不是先堆代码。

### Tranche G：最终审计和第二轮压缩

1. 从 public API、wheel、CLI、entry point、dynamic string 和 benchmark 做 reachability 审计。
2. 删除旧内部文档和失效 module path；Git 历史承担归档。
3. 全量运行不超过 50 项的证据集。
4. 生产源码必须进入 22k–30k 的当前可信区间；超过 30k 表示第一轮合并未完成。
5. 对剩余每个 owner 记录必要职责和替代成本，再决定能否从约 26k 继续压向 20k、15k、
   最终接近 10k；没有代码证据时不提前承诺下一档。
6. benchmark 只在 workload 切换门和最终发布门运行。

## 7. 功能与架构验收

### 7.1 唯一执行路径

- 所有 workload 通过同一 typed public factory；
- 每个请求只有 `SemanticGraph -> Rust decision -> PhysicalPlan -> execution`；
- workload 中不存在 lane/priority/fusion/provider/手写顺序；
- active core 不导入任何退役包；
- strict Pyright 0 error/0 warning，动态类型只在 Python–Rust codec；
- 无运行时反射、保护性 fallback、callback bag 或第二调度 authority。

### 7.2 PTTS/AER

- `StateRef(LayoutId, LineageId, VersionId)` 基于 frozen typed layout namespace；
- capacity 在 admission-time entitlement，耗尽 fail closed；
- multi-root publication 原子，inference view immutable；
- request/global version exact pin/release/cleanup，generational identity 阻断 ABA；
- CUDA completion 由 issued receipt/fence 证明；
- AER/scheduler 与小图 exhaustive oracle 一致；
- PhysicalPlan 是 signed decision 的机械投影，worker reconstruction 检测 tamper。

### 7.3 workload observable semantics

| 能力组 | 必须观察到的证据 |
| --- | --- |
| Dummy/simple | graph/sign/plan/execute 完整路径 |
| ESamp | request PTTS、非零 loss/update、sampler pin、cleanup、零 fallback |
| FTTT/GTTA | typed host transaction、receipt、abort/retry 或 queue settlement |
| In-Place-TTT | request dense overlay、版本隔离、两请求 cleanup |
| TLM | global LOW_RANK_ADD、online/offline timing、admission pin、immutable publication |
| ReFT/Recirculation | activation patch/replay typed effect 和生命周期 |
| OverRIDE/TTRL | 数学更新、state version、reward/control、closed-loop settlement |

GPU/MP 只为真正不同的程序–运行时边界保留代表性 smoke，不以未测试组合声称支持。

### 7.4 数量验收

- 生产及支撑源码从 133,210 行降到 22,000–30,000 行；超过 30,000 不通过；
- 实际净删约 103,000–111,000 行，最终数字以 reachability 和功能 manifest 为准；
- `TEST_COUNT <= 50`，且没有脚本化规避；
- 旧 Python/Rust authority 整块删除；
- import-layer/reachability 0 violation；
- strict Pyright 0/0、`cargo check`、wheel/import probe 和代表性 backend smoke 通过；
- 训练证据有非零 loss/update，所有执行证据 fallback 为零。

所有 pytest 使用 `/workspace/pllm/.venv/bin/python` 并报告环境。纯文档 tranche 不运行
行为测试。

## 8. 每个提交的硬规则

1. 除纯文档和 Tranche D 已点名归因的解耦 seam 外，每个 tranche 的生产代码必须净减少；
2. 不允许先加新实现、下次再删旧实现；route switch 与删除原子完成；
3. 不为 `StateStorage.FIXED_BATCHED`、Program、BaseConsumer 或旧 runtime 建 bridge；
4. 不新增 workload-specific scheduler/session/runtime；
5. 不先写 `Any`/`object`/`cast` 再等待审查返工；
6. review 同时检查 authority/import 和装饰性测试；
7. 估算偏差必须归因到可点名的功能、owner 或 backend seam；
8. 不用浅 wrapper、代码搬家或巨型文件压缩数字；
9. benchmark 只在切换门和发布门运行。

## 9. 完成定义

第一轮重构完成必须同时满足：

1. 旧生产 authority 从仓库删除；
2. 十个 workload 的公共功能由唯一 native path 提供；
3. Rust 自动调度、PTTS、typed host service、closed MP 和 backend 边界完整；
4. 生产源码落入当前有证据支持的 22k–30k 区间；
5. 全仓最多 50 项高价值测试；
6. 不靠 compatibility wrapper、猜测、fallback 或重复 DTO 维持功能。

完成第一轮后继续以实际 owner/reachability 证据向 10k 压缩。10k 是方向，不是允许拍脑袋
删除功能或倒填层级预算的理由。
