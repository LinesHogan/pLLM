# 架构精简交接：剩余实施工作

核查日期：2026-09-06。代码检查点：`e03ea16`，分支 `main`。
工作目录：`/workspace/pllm`。
本文补充[原计划](architecture-consolidation-10k-plan.md)，不修改原计划的历史基线、
功能要求或验收标准；逐提交减量见 [ledger](architecture-consolidation-ledger.md)。

## 当前结论

旧栈大删除已取得实质进展，第一轮重构尚未完成。后续主线是继续删除残留 authority、
合并已有执行路径，并补齐被删除旧实现所承担的公共功能和必要证据。
不能把旧目录消失、静态检查通过或测试少于 50 项视为功能迁移完成。

| 当前事实 | 核查结果 |
| --- | --- |
| SOURCE_LOC | 18,676；相对 133,210 净减 114,534（约 86.0%）；低于预测区间不代表功能补齐 |
| 距第一轮 30k 上限 | 当前余量 11,324 行；补齐缺失能力仍须与新栈压缩配套，不为凑下限增加代码 |
| 本轮源码差分 | 相对 `ccc211e` 新增 645 / 删除 2,967 / 净减 2,322；包含新 backend_program 全部 87 行；没有整文件原样移动；独立 AdamW/fast-weight 数学不变 |
| 已删除旧包 | consumers、kit、program、workgraph、schedule、workload、producers、canonical_runtime、runtime、ports、common、core、util |
| 已删除 Rust 旧 authority | workgraph、schedule、recurrence、execution_plan、placement，以及失去调用者的 shape_specialization/decode_localization/sampler_plan |
| 尚存主要合并对象 | canonical 编排、canonical_vllm 与 backend probe/replay primitives、typed_replay 旧契约；overlay_state 已删 |
| 留存证据基础 | 4 个 Rust tests + 12 个 pytest items + 4 个独立 GPU smoke，共 20 项 |
| 终态证据债务 | 原 39 项中 19 项待补，加新增组合 40/41，共 21 项待补 |

SOURCE_LOC 包含 Rust 文件内测试行，减量不全等于生产行为删除。移动不作为减量成果。
当前目录体积用于选择整块处理边界，不能直接当作可删除量：

| 范围 | 行数 |
| --- | ---: |
| tllm/canonical | 8,523 |
| tllm/backends + canonical_vllm | 1,633 + 1,916 |
| tllm/contracts | 2,053 |
| Rust | 3,434 |
| tools / 顶层入口及 stub | 861 / 256 |

## 下一轮实施顺序与完成条件

### 1. Residual 入口和旧状态岛已删除（D/E 本轮完成的切片）

`make_llm` 的 plain 分支现在直接构造 vLLM，保留引擎参数；不再安装全局 hook、
导出 bootstrap 或注入 worker extension。旧 runtime、MP、ports、common、旧 driver 安装器
与失效测试/gate 引用同批删除。实际 wheel 已验证不含退役包和 general plugin。

后续不再重复审计这些已删除目录。当时保留到 backend 的三份文件是 `graph_mode.py` 79 行、
`overlay_state.py` 486 行、`specialization.py` 52 行，合计 617 行原样移动，不计净删。
`e03ea16` 已删除 overlay_state、versioned_parameter_overlay、dense_parameter_overlay
三个无公共调用者的文件（1,556 行），以及旧 protocol 中的 store/handle/lease 能力声明。
旧 tensor CPU materialization、backend version cache 和 latest/lease 账本没有迁入新模块。
MRv1/旧版本结构 probe、typed replay callback 仍待接入共享边界；TLM/global 公共路径仍欠。

### 2. 完成 generic execution 与 backend 的实质合并（B/D）

- FTTT/GTTA 已删除各自 execution、receipt、completion、compilation/options 包装；
  `compile` 直接返回同一个 `HostExecution`，共享 signed task 执行、快照回滚和 receipt 结算。
  FTTT/GTTA 的数学状态各自保留，不被复制进通用 Context/service locator。
  `ccc211e` 删除 ESamp/In-Place 各自 execution、handle、active/sampler lifecycle 和 diagnostics。
  两者的 CPU/CUDA admission、signed anchor、sampler、cancel、retire、settlement 统一由
  `StateExecution` 持有；workload 只保留初始化、输入约束、数学 task 和结果。
  invocation 引用原始 PhysicalPlan 游标；request directory 引用原始 SemanticState；
  context 持有原始 PagePin、PageWrite、PagePublication 和 PageRelease，不复制版本根。
  发布前失败清理私有写入；发布后错误携带原始 invocation/receipt，重试 finish 或取消结算
  不再训练、不再次释放同一 pin。admission 发布前失败恢复容量并允许重试。
  无法证明 CUDA 完成时仍 fail-stop 保留资源；不声称任意设备故障均可恢复。
  In-Place 直接消费原始 CapturedParameterOperand，删除 OperandPair→StepInput→Batch→CUDA
  wrapper 转换链、重复 phase 和 ParameterOverlayPublication；发布凭据只由 arena 签发。
  backend 删除 predicted/sampled/prepared/transitioned 镜像阶段，以原始游标和 sampler 状态推进；
  `e03ea16` 进一步删除两个 typed active/deployment 分支，backend runtime 不再 import workload。
  ESamp readiness 改为预测结果产生后签发，只由共享 context 持有；延迟写入反例已先失败后通过。
- SemanticGraph/plan 的 17 个 Python 重复校验已删除；Rust 保持唯一语义校验，FFI codec
  保留封闭类型解析。没有为 AER/MP/replay 新建各自的图或状态 DTO 家族。
  Rust 现在从结果输入、附加 ordering edge 和 timing 派生唯一 GraphIndex；FTTT、GTTA、
  In-Place、ESamp 的重复 producer→consumer 边已删除。Python FFI 从共享 dataclass 机械编码，
  删除第二套 wire DTO 及其逐字段复制，只补封闭 input variant tag。
- 附件输入收敛为原始 `request/probe/plan` 引用，plan 持有原始 tensor-layout request/decision；
  request 不再复制 probe facts、plan signatures 或 request-key schema。sampler 独立附件模块和
  wire DTO 删除，FFI 机械编码同一结构；Rust 校验 layout 身份、字段存在性和阶段/任务绑定。
  `e03ea16` 的 deployment 仅引用原始 StateExecution 和 signed attachment；不复制 compilation
  或 overlay mapping。schema 3 签入原始 ParameterOverlayFieldBinding 的 target/operator/fields，
  修复同 field ID 更换层后签名不变的问题。Rust 拒绝当前尚不支持的 hidden/sampler/overlay 组合。
- 公共 factory 消费 BackendWorkload；共享 BackendCompileFacts、TensorStepInput 和同一 timeline，
  runtime 793 行且不再 import ESamp/In-Place。hidden capture 统一按原始签名 resource 捕获；
  sampler beta 属于原始 guidance。新 workload 仍需实现真正的数学/声明；这不是十项公共支持完成。
- dense backend 直接消费原始 packed rows、StateTaskContext、PagePin 和签名 field binding，
  删除 ParameterOverlayPinnedTensor/RowAssignment/StepPlan 及 field_rows 重新打包链。
  In-Place 数学也借用原页，不再先 snapshot 全根。borrow_field 是原 pin 生命周期内的只读借用；
  tensor 不提供硬件只读保护，native consumer 不得原地修改或在 release 后持有别名。
  CPU/CUDA 证据覆盖 COW 页共享、旧版本不变、伪造/释放 pin 拒绝和延迟读取后回收。
- 把 workload 编排收进同一 typed executor/host transaction；CPU/CUDA 共享 transition，
  provider 只承载 kernel。压缩 FTTT/GTTA、ESamp、In-Place 的剩余 execution/task 链。
- 合并 `backends` 与 `canonical_vllm` 的 probe、capture、sampler、overlay mechanics，
  去除 attachment/codec/DTO 的残余重复。Python 不重做 Rust semantic selection/validation。
- `contracts/typed_replay.py` 等 callback 型契约随 typed effect 切换删除；
  typed-internal 防御性类型检查随所属 owner 重构清掉。

完成条件：公共 factory 消费统一 typed declaration；新 workload 不要求向 runtime 添加
专属分支/session。MRv1/MRv2 保留实际不同的 adapter 边界，不能用 MRv2 结果代替 MRv1 验证。

### 3. 补齐十个 workload 的公共功能与语义 oracle（C）

| Workload | 现状与待办 | Evidence IDs |
| --- | --- | --- |
| Dummy | graph→sign→plan→execute、stride 与 owned export oracle 已补；统一 factory 尚待 B | 20 |
| FTTT | 已切共享 host execution；独立 Adam/edit、回滚、retry/stop/accept oracle 与 CUDA receipt 通过 | 21, 38 |
| GTTA | canonical host 路径的精确 EMA、FIFO abort/requeue、版本回滚 oracle 已补；统一 factory 待 B | 22 |
| In-Place-TTT | 共享 StateExecution 和 generic backend session；原始捕获记录及页借用、独立数值、取消、发布前回滚与发布后结算重试通过 | 23, 32 |
| ESamp | 共享 StateExecution 和 generic backend session；独立数学、admission 回滚、取消、发布后结算、跨流 guidance readiness 与清理通过；MRv1 待完成 | 25, 30, 31, 37 |
| TLM | 仅 loader/cold facts 与 objective kernel 基础；缺完整 online/offline graph、global LOW_RANK_ADD publication/pin | 11, 24, 33 |
| ReFT | 旧 owner 已删，canonical owner 缺失；补 LoReFT 数学、publish/skip 和 activation patch | 26, 34 |
| Recirculation | 旧 recurrence 岛已删，canonical owner 缺失；补 row selection、typed replay effect | 27, 35 |
| OverRIDE | canonical owner 缺失；补 reweight/Adam/state transition/final no-op | 28 |
| TTRL | canonical owner 缺失；补 reward/update/retry/control quorum 与 settlement | 29 |

缺失项是功能债务。需要时从删除前的 Git 历史提取独立数学、loader、golden，明确唯一
新 owner；不得恢复 consumer/class hierarchy。每项以可观察结果验收，已有源文件不等于
公共支持。新增能力须与该 tranche 的替代对象删除配套，净变化按原计划记账。

### 4. 补齐终态核心，直接替换 provisional owner（F，与第 2/3 项按依赖交错实施）

- Rust AER regionization、完整版本 hazard、distributed placement/transfer 仍待实现。
  第 3 项已升级为独立枚举：三任务的 64 种有向边集合 × 216 种窗口组合，验证最早可行放置、
  cycle、数据流依赖和顺序。冲突 effect 若无因果路径或不相交窗口证明则拒绝；这不是
  exact-version RAW/WAR/WAW oracle。第 4、6、7、8 项仍欠，不能以索引或最早放置冒充 AER。
  后续直接推进版本/effect、resource facts 与真正的联合 partition/placement，不先精修索引。
- `e996998` 已用一个 PageBundleArena 替换旧 PTTS 的 13 个模块，含 CPU/CUDA provider、
  双 publication coordinator、独立 global directory 和旧 ledger。request 目录只维护 key→lineage
  与 exclusive lease，不再分配版本或保存 latest。ESamp/In-Place 直接持有同一 arena、原始 pin
  与 publication；没有再加 workload storage/kernel wrapper，也不各自维护 pending-release 列表。
  每字段独立 pool，部分字段 COW 共享未修改 slot；唯一 arena 原子切换多资源版本根。
  `519d4ed` 修正字段 slot 对齐：pool 基址与步长均满足 layout，padding 计入同一签名容量；
  独立对齐反例在修改前失败、修改后通过。不是额外的 page/capacity authority。
  issued completion 与 one-shot settlement 共同约束回收，未结算 publication 保留前后版本；
  CUDA abort 等待已提交私有写入，无法建立完成证明时 fail-stop 保留资源，不声称可恢复。
  附件和 arena 重建共用 Rust `CanonicalPlanBundle::validate`，检查原始 layout/plan 身份与 epoch。
  已补 9/10/12；13 已迁移并通过真实跨流等待，10 含不同布局的 request/instance 原子发布。
  仍欠 11 的 TLM 多因子数学/发布，以及 14 的跨版本 CUDA Graph page-table 间接寻址。
  当前 pool 地址稳定不等于 capture-stable version selection，provider 仍声明 `capture_stable=False`。
  不因通用 instance arena 测试通过就声称 TLM/global overlay 公共路径完成；旧 backend overlay
  store 已删，TLM 必须直接接此 arena，不能恢复另一份 global version store。
  closed MP、其余布局泛化、容量指标与 AER 资源联合规划继续共用此边界。
- 旧 `runtime/mp/port_wire.py` 的反射 transport 已删除；closed typed MP 尚未实现。
  旧 workload RPC 被删除不等于新的 worker reconstruction、prepare/commit/abort、quorum
  已实现；补共享 tensor 生命周期和真实 EngineCore child 路径，对应 16–18、36。
- generic host transaction 的 abort/requeue/post-publish settlement（19）已通过；
  发布后失败返回原始 receipt 继续结算，不重跑 host effect。signed attachment 统一证据（15）已补；
  它证明字段属于签名 layout 及原始选择不可篡改，不声称推导了 workload 字段的数学角色。
- typed authoring descriptors、统一 workload-independent profiling 仍待完成；
  profiling 消费通用执行事件，替换散落 counters/hooks，不新增 workload callback。

PTTS/AER 设计继续以 `doc/research-aer-paged-state.md` 为唯一权威。
该研究文档不在本次交接修改、加入或提交范围。按实际 workload 依赖切 replacement tranche，
不要先并行堆出一套完整新框架再等待删除旧实现。

### 5. 补齐证据、发布验证与最终压缩（A/G）

- `test/evidence_manifest.toml` 的 active IDs 是 `1,2,3,5,9,10,12,13,15,19,20,21,22,23,25,30,32,37,38,39`。
  pending IDs 是 `4,6–8,11,14,16–18,24,26–29,31,33–36,40,41`，含新增的条件 replay 组合路径。
- 21 项随真实边界实现补成紧凑 oracle/smoke；现有基础复核是否达到命名所要求的全部
  数值/版本/cleanup contract。禁止为凑 39 写 DTO、mock 调用次数或旧内部结构测试。
- manifest 最终逐项列出实际文件/命令、observable contract、环境、GPU/MP 要求及替代的
  旧测试来源。训练必须非零 loss/update，执行必须 zero fallback。
- 第 39 项已验证 sdist→从归档构建 wheel→隔离安装/import→Rust codec 执行，并核对
  Torch 依赖、LICENSE 和退役包/plugin 缺席。根 `pyproject.toml` 是唯一 Python 打包配置；
  删除 Rust 子目录重复配置和 `requirements-minimal.txt`，支持 ABI 限定 vLLM 0.27.1。
- 全量行为证据不超过 50，原 39 项及组合 40/41 完整；SOURCE_LOC 进入 22k–30k；strict typing 0/0、
  import/authority gates、Rust build、代表性 backend 边界和发布验证通过后才算第一轮完成。
  再逐 owner 审计向 10k 压缩，不承诺缺乏代码依据的工期或最终行数。

### 6. 仓库与发布分支整理（G，已执行首轮）

- 30 个 worktree 降至 1 个：仅保留主目录 `/workspace/pllm`，`main` 已快进至
  `7d53caf`。辅助 checkout 均以非 force 的 `git worktree remove` 移除，所有分支/提交保留。
- 原 main 的 5 项修改和 LoRA 的 31 项修改先完整保存为下述快照，再整理 checkout。
  main 的旧 Program/WorkGraph 删除已由合并代码覆盖；LoRA 数学与进程训练能力仍待吸收，
  不因归档就视为功能整合完成。研究原稿保持原样、未跟踪。
- 删除 17 份旧 consumer/Program/PTTS 中间态说明及失效中文镜像，重写 README、文档索引、
  当前架构、验证和 benchmark 指南；Git 历史承担归档。不再维护旧 API 的翻译副本。
- 已补齐原元数据声明的 Apache-2.0 正文。README 明示实际支持范围与开发状态；
  原历史 benchmark harness 已删，旧 n=16 吞吐不能作为当前 n=1 公共路径的支持证明。
- 主目录残留的旧 `tllm.egg-info` 会让 vLLM 发现已删除的 general plugin；已移至
  `/tmp/tllm-stale-metadata.sUqZvf/tllm.egg-info`，可恢复。当前仅发现 `.venv` 的无插件发行元数据，
  plugin discovery 和真实 GPU smoke 已复验。补充设计原文纳入文档索引，研究原稿保持未跟踪。
- 未发布、未打 tag、未推送。功能/架构证据未齐和 LoRA 语义整合仍是发布阻碍。
- 两处改动已保存为可恢复快照：`archive/consolidation-main-20260906` (`865ad4c`)、
  `archive/consolidation-lora-20260906` (`c5b3589`)，包括 LoRA 的四个未跟踪源码/测试文件。
  LoRA 原分支另以 `781034e` 提交同一快照树后移除 checkout。grouped kernel、weighted-CE/shared update、
  进程训练边界须按 PTTS/closed MP 语义吸收，不能复制旧 registry、反射或 fallback。

### 7. 新设计的组合路径与共享边界（未完成）

执行原计划第 10 节已确认的范围：同一 SemanticGraph、同一版本根、同一 issued receipt，
不按 AER/PTTS/replay/MP/trace 分别建立镜像 authority。需要完成有界条件区、明确 KV/RNG
effect、统一有界 trace，以及独立 PyTorch oracle 和真实 Qwen2 最后一层 replay 场景（40/41）。
全面性能研究和独立变体实验不纳入本次门槛。第一轮完整后仍须进行 owner 第二轮压缩。

## 验证记录与复现

`e03ea16`：仓库 `/workspace/pllm/.venv/bin/python` 在沙箱外运行完整 pytest
12 passed（26.66s），含真实 CUDA 页借用/延迟回收和隔离 sdist→wheel；Rust 4 passed，
strict/basic typing 0/0、import 0/0、authority 扫描 7 个生产文件通过。
页借用最终版本的公共 In-Place GPU smoke：update/trained_rows 6、apply_rows 8、
updated_rows 2、active_lineages/pending_releases/fallback 0。本批较早的 generic session/schema 3
版本已通过四个 GPU smoke：ESamp loss/update 2、In-Place 6、CUDA Graph 4 且地址稳定、
FTTT 1 且 version 1；最终 dense-only 借用和死代码删除后未重跑其余三个 smoke。
target 签名和未支持 hidden/overlay 组合的反例先失败后通过；页借用 oracle 从缺失接口
失败后通过 COW、原地址和 issued pin 生命周期检查。未运行吞吐 benchmark、未发布。
下一项应补 TLM online/offline/global LOW_RANK_ADD 和原 arena admission pin，而非重做
已完成的 generic backend；AER、MP、条件 replay 和其余 workload 仍为明确功能债务。

`ccc211e`：仓库 `/workspace/pllm/.venv/bin/python` 在沙箱外运行完整 pytest
12 passed（22.16s），含 CUDA PTTS 和隔离 sdist→wheel；Rust 4 passed，strict/basic typing
0 errors/0 warnings，import/authority gates 通过，authority 扫描 5 个生产文件。
四个 GPU smoke 均复跑：公共 ESamp loss/update 2；公共 In-Place update/trained_rows 6、
apply_rows 8、updated_rows 2；CUDA Graph loss/update 4、地址稳定；FTTT loss/update 1、
version 1。全部 fallback 为 0、清理通过。admission 容量泄漏、发布后 finish 重试和
sampler 提前读取三个反例在修复前失败、修复后通过。沙箱内 CUDA 不可见的那次运行
为 11 passed/1 skipped，不作为完整 GPU 通过记录。没有运行吞吐 benchmark 或发布。

`519d4ed` PageBundle replacement：仓库 `/workspace/pllm/.venv/bin/python` 运行完整
pytest 12 passed（27.24s），含隔离 sdist→wheel；Rust 4 passed，strict/basic typing 0/0，
import/authority gates 通过。四个 GPU smoke 均通过：ESamp loss/update 2，In-Place
update/trained_rows 6、apply_rows 8，CUDA Graph loss/update 4 且地址稳定，FTTT
loss/update 1、version 1。fallback 均为 0，清理通过。没有运行吞吐 benchmark 或发布。
证据 14 的跨版本 captured page-table 选择仍未实现，不能用第 37 项训练 capture 替代。

`4cd162f` 已验证：pytest 3 passed（CUDA PTTS、Dummy oracle、隔离 wheel）；cargo test 3 passed。
import-layer 0/0；region authority 扫描实际 canonical execution/host/sampler 和 replay，
共 4 个文件；strict/basic Pyright 均 0 errors/0 warnings，basic baseline 已随删除降为 0。
公共 plain→ESamp 同进程 smoke 通过（loss/update 各 2、active_lineages=0、fallback=0）；
In-Place-TTT 双请求 smoke 通过（update=6、dense_apply_rows=8、active_lineages=0、fallback=0）。
CUDA graph replay 为 loss/update 各 4、stable addresses、fallback=0；FTTT 为 loss/update
各 1、version=1、fallback=0。当时 10 项均已执行通过。

`19c59d5` 新增 GTTA、In-Place CPU oracle 并强化分发包证据后，完整 pytest 5 passed；
import-layer 0/0、region authority gate、strict/basic Pyright 均 0 errors/0 warnings。
FTTT CUDA 重新执行通过：loss/update 各 1、version=1、delta_max=0.0127174、fallback=0。
其余三个 GPU smoke 与 Rust 核心本轮未改、未重复执行，沿用 `4cd162f` 证据。
本轮未运行吞吐 benchmark。

当前共享 host/Rust validation tranche：完整 pytest 7 passed、Rust 4 passed、strict/basic
typing 0/0、import/authority gates 通过。四个 GPU smoke 均复跑：ESamp loss/update 2，
In-Place update 6，CUDA Graph loss/update 4，FTTT loss/update 1；清理正确且 fallback 全为 0。
FTTT CUDA 初始化也以 issued readiness event 跨流排序，不依赖编译耗时碰巧掩盖竞态。
合入主目录并从主目录重建 native extension 后，再次运行完整 pytest：7 passed（28.25s）。
Python 包和 `_core.abi3.so` 的加载路径均已核对为主目录，不依赖被删除 worktree。

`bea197d`：完整 pytest 8 passed（22.81s）、Rust 4 passed、strict/basic typing 0/0，
import/authority gates 通过。第 23 项发布失败注入在替换前后均通过，第 25 项补独立数学与
版本 oracle。In-Place 真 MRv2 GPU smoke 复跑通过：update/trained_rows 各 6，apply_rows 8，
updated_rows 2，active_lineages/pending_releases/fallback 均为 0；第二次运行确认旧插件错误消失。
另外三个 GPU smoke 本次未改其执行路径，沿用 `7d53caf` 记录；没有运行吞吐 benchmark。

`86e1326`：主目录重建 native extension 后，完整 pytest 8 passed（26.94s）、Rust 4 passed
（含 13,824 组独立枚举），strict/basic typing 0/0、import/authority gates 通过。新 oracle
在旧的显式重复边要求上先失败，再经生产实现修改通过。四个 GPU smoke 全部复跑：
ESamp loss/update 2、In-Place update/trained_rows 6、CUDA Graph loss/update 4 且地址稳定、
FTTT loss/update 1 且 version 1；所有执行 fallback 为 0，清理检查通过。没有运行吞吐 benchmark。

`4f7e370`：ESamp task/context 合并后，完整 pytest 8 passed（27.33s），strict/basic
typing 0/0、import/authority gates 通过。第 25 项取消与发布前失败注入在替换前后均通过；
第 37 项每次 CUDA prediction 后先取消再重试，loss/update 各 4、地址稳定、fallback 0。
公共 plain→ESamp 真 MRv2 smoke 复跑：loss/update 各 2、active_lineages/fallback 均 0。
Rust 和 In-Place/FTTT 本次未改其执行路径，未复跑，沿用 `86e1326` 记录。
逻辑目录与物理 provider 的 authority 拆分仍待 PageBundle 替换；此次没有补全跨根原子
结算或 CUDA 提交失败恢复，也未精修 provisional dense_page。没有运行吞吐 benchmark。

`9dfa7e7`：ESamp lifecycle 合并后，完整 pytest 8 passed（28.63s）、Rust 4 passed，
strict/basic typing 0/0、import/authority gates 通过。prediction/loss 后取消场景替换前后
均通过；新增在途 sampler retire 反例在缺少跨流等待时失败，接入真实 completion event 后
通过。CUDA Graph loss/update 各 4 且地址稳定；公共 ESamp loss/update 各 2；公共 In-Place
update/trained_rows 各 6、apply_rows 8。live lineage、pending release 和 fallback 检查通过。
FTTT CUDA 未改、未复跑，沿用 `86e1326`。未运行吞吐 benchmark；PTTS 内部拆分、
跨根 settlement 和 CUDA 提交 fail-stop 语义未在本次替换中宣称完成。

`e57caa6`：In-Place lifecycle 合并后，完整 pytest 8 passed（24.23s）、Rust 4 passed，
strict/basic typing 0/0、import/authority gates 通过。第 23 项新增分阶段更新在旧 CPU
execution 缺少接口处先失败，合并后通过取消重试、sampler gate、重复结算拒绝及独立数值。
公共 In-Place MRv2 smoke：update/trained_rows 各 6、apply_rows 8、updated_rows 2，
active_lineages/pending_releases/fallback 均 0；公共 plain→ESamp loss/update 各 2，
active_lineages/fallback 均 0。CUDA Graph/FTTT 路径本次未改、未复跑，分别沿用
`9dfa7e7`/`86e1326`。未运行吞吐 benchmark；未声称完成发布后故障恢复。

`70a82d2`：附件字段越界反例在旧 Rust 编译器接受字段 999 时先失败，改为共享 plan/layout
验证后通过。第 15 项用真实 ESamp/In-Place 编译覆盖 exact task/hidden/field/stage、跨 layout
错配、mechanical projection、reconstruction/local drift 和 closed schema，替代原附件 fixture。
完整 pytest 9 passed（26.10s）、Rust 4 passed，strict/basic typing 0/0、import/authority gates
通过。四个 GPU smoke 均复跑：公共 ESamp loss/update 各 2，In-Place update/trained_rows 各 6、
apply_rows 8、updated_rows 2，CUDA Graph loss/update 各 4 且地址稳定，FTTT loss/update 各 1、
version 1；清理和零 fallback 检查通过。未运行吞吐 benchmark。没有改 provisional PTTS、
没有新增镜像图/版本账本；generic factory/session、AER/PageBundle/MP/组合 replay 仍未完成。

所有 Python 使用 `/workspace/pllm/.venv/bin/python`。canonical vLLM 当前代码要求
`0.27.1`，且限定 in-process/eager 等边界；本轮实测 vLLM runtime 为 0.27.1，发行元数据
为 0.27.1+cu129，Rust extension 已用仓库根目录的 maturin develop 重建。
运行 GPU 证据前仍需检查环境和设备，不能把历史 smoke 日志视作其他版本的支持范围。

```bash
cd /workspace/pllm
git status --short --branch
git ls-files | rg '\.(py|pyi|rs)$' | rg -v '^(test|typings)/' | xargs wc -l
/workspace/pllm/.venv/bin/python tools/check_import_layers.py
/workspace/pllm/.venv/bin/python tools/check_region_authority.py
/workspace/pllm/.venv/bin/python tools/check_typing.py
# 实施产生行为变更时，按对应 evidence 执行；以下 collection 不含四个独立 smoke。
/workspace/pllm/.venv/bin/python -m pytest --collect-only -q
cargo test --manifest-path rust/tllm_core/Cargo.toml -- --list
```

交接执行偏好：主代理掌握大局并审查是否真正减掉重复 authority，避免细枝末节精修。
用户最新要求不建议再启动 subagent；如确需使用，优先 gpt-6。
