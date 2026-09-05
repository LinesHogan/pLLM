# 架构精简交接：剩余实施工作

核查日期：2026-09-06。代码检查点：`4cd162f` 后的工作区变更，分支 `tranche-b-atomic`。
工作目录：`/workspace/pllm/.worktrees/tranche-b-atomic`。
本文补充[原计划](architecture-consolidation-10k-plan.md)，不修改原计划的历史基线、
功能要求或验收标准；逐提交减量见 [ledger](architecture-consolidation-ledger.md)。

## 当前结论

旧栈大删除已取得实质进展，第一轮重构尚未完成。后续主线是继续删除残留 authority、
合并已有执行路径，并补齐被删除旧实现所承担的公共功能和必要证据。
不能把旧目录消失、静态检查通过或测试少于 50 项视为功能迁移完成。

| 当前事实 | 核查结果 |
| --- | --- |
| SOURCE_LOC | 27,961；相对 133,210 净减 105,249（约 79.0%） |
| 距第一轮 30k 上限 | 当前余量 2,039 行；补齐缺失能力仍须与新栈压缩配套 |
| 本轮源码差分 | 相对 `4cd162f` 新增 10 / 删除 22 / 净减 12；无移动、无数学抽取 |
| 已删除旧包 | consumers、kit、program、workgraph、schedule、workload、producers、canonical_runtime、runtime、ports、common、core、util |
| 已删除 Rust 旧 authority | workgraph、schedule、recurrence、execution_plan、placement，以及失去调用者的 shape_specialization/decode_localization/sampler_plan |
| 尚存主要合并对象 | canonical 编排、canonical_vllm 与 backend primitives、overlay_state 和 typed_replay 旧契约 |
| 留存证据基础 | 3 个 Rust tests + 5 个 pytest items + 4 个独立 GPU smoke，共 12 项 |
| 终态证据债务 | 39 项目标中 27 项待补；留存基础也须复核完整 observable contract |

SOURCE_LOC 包含 Rust 文件内测试行，减量不全等于生产行为删除。移动不作为减量成果。
当前目录体积用于选择整块处理边界，不能直接当作可删除量：

| 范围 | 行数 |
| --- | ---: |
| tllm/canonical | 15,011 |
| tllm/backends + canonical_vllm | 3,360 + 2,853 |
| tllm/contracts | 2,478 |
| Rust | 3,174 |
| tools / 顶层入口及 stub | 857 / 228 |

## 下一轮实施顺序与完成条件

### 1. Residual 入口和旧状态岛已删除（D/E 本轮完成的切片）

`make_llm` 的 plain 分支现在直接构造 vLLM，保留引擎参数；不再安装全局 hook、
导出 bootstrap 或注入 worker extension。旧 runtime、MP、ports、common、旧 driver 安装器
与失效测试/gate 引用同批删除。实际 wheel 已验证不含退役包和 general plugin。

后续不再重复审计这些已删除目录。保留到 backend 的三份文件是 `graph_mode.py` 79 行、
`overlay_state.py` 486 行、`specialization.py` 52 行，合计 617 行原样移动，不计净删。
其中 overlay state/version store 和 typed replay callback 仍是待统一对象，不能因搬到
backend 就宣称唯一版本 authority 已达标。MRv1/旧版本结构 probe、replay/overlay primitives
留待下一项接入统一 typed seam；旧 driver 删除不代表新 MRv1 支持已完成。

### 2. 完成 generic execution 与 backend 的实质合并（B/D）

- `canonical/execution.py` 的 `TaskGraphExecution.run` 已接管 FTTT、GTTA、In-Place CPU
  的同步执行循环，按 signed task 顺序执行。CUDA 外部 anchor 游标保留；仅完成 B 的小切片，
  workload transaction/session 和 CPU/CUDA transition 合并仍未完成。
- `canonical_vllm/runtime.py` 仍有 1,494 行，直接 import ESamp/In-Place-TTT，
  deployment/execution 类型为两者 union，公共 factory 仅接受这两种 workload。
  `canonical_runtime` 目录虽然已删，workload-specific 编排职责仍有残留。
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
| FTTT | 数学/judge/host owner 存在；收敛 generic transaction，补 retry/receipt oracle | 21, 38 |
| GTTA | canonical host 路径的精确 EMA、FIFO abort/requeue、版本回滚 oracle 已补；统一 factory 待 B | 22 |
| In-Place-TTT | 多 target tail/chunk 数值、原子发布、两请求版本隔离/清理 oracle 已补；CPU/CUDA 合并待 B | 23, 32 |
| ESamp | 有 request CUDA/sampler 路径；补 request isolation、先预测后训练及 MRv1 证据 | 25, 30, 31, 37 |
| TLM | 仅 loader/cold facts 与 objective kernel 基础；缺完整 online/offline graph、global LOW_RANK_ADD publication/pin | 11, 24, 33 |
| ReFT | 旧 owner 已删，canonical owner 缺失；补 LoReFT 数学、publish/skip 和 activation patch | 26, 34 |
| Recirculation | 旧 recurrence 岛已删，canonical owner 缺失；补 row selection、typed replay effect | 27, 35 |
| OverRIDE | canonical owner 缺失；补 reweight/Adam/state transition/final no-op | 28 |
| TTRL | canonical owner 缺失；补 reward/update/retry/control quorum 与 settlement | 29 |

缺失项是功能债务。需要时从删除前的 Git 历史提取独立数学、loader、golden，明确唯一
新 owner；不得恢复 consumer/class hierarchy。每项以可观察结果验收，已有源文件不等于
公共支持。新增能力须与该 tranche 的替代对象删除配套，净变化按原计划记账。

### 4. 补齐终态核心，直接替换 provisional owner（F，与第 2/3 项按依赖交错实施）

- Rust AER regionization、state hazard/timing、distributed placement/transfer；当前
  canonical scheduler 是基础，不能据此声称完整 AER 已实现。对应 2、4、6、7、8。
- PageBundle/generational arena 替换 provisional `canonical/ptts/dense_page*`；
  dense_page 冻结禁投，不先做中间态精修。覆盖 admission/COW/multi-root atomic publication、
  ABA/receipt、exact pin/reclaim、CUDA stable address；对应 9–14。
- 旧 `runtime/mp/port_wire.py` 的反射 transport 已删除；closed typed MP 尚未实现。
  旧 workload RPC 被删除不等于新的 worker reconstruction、prepare/commit/abort、quorum
  已实现；补共享 tensor 生命周期和真实 EngineCore child 路径，对应 16–18、36。
- 完成 generic host transaction 的 abort/requeue/post-publish settlement（19），
  统一 signed attachment exact binding/tamper 证据（15）。
- typed authoring descriptors、统一 workload-independent profiling 仍待完成；
  profiling 消费通用执行事件，替换散落 counters/hooks，不新增 workload callback。

PTTS/AER 设计继续以 `doc/research-aer-paged-state.md` 为唯一权威。
该研究文档不在本次交接修改、加入或提交范围。按实际 workload 依赖切 replacement tranche，
不要先并行堆出一套完整新框架再等待删除旧实现。

### 5. 补齐证据、发布验证与最终压缩（A/G）

- `test/evidence_manifest.toml` 的 active IDs 是 `1,3,5,13,20,22,23,30,32,37,38,39`。
  pending IDs 是 `2,4,6–12,14–19,21,24–29,31,33–36`。这些编号对应原计划第 5.3 节。
- 27 项随真实边界实现补成紧凑 oracle/smoke；现有基础复核是否达到命名所要求的全部
  数值/版本/cleanup contract。禁止为凑 39 写 DTO、mock 调用次数或旧内部结构测试。
- manifest 最终逐项列出实际文件/命令、observable contract、环境、GPU/MP 要求及替代的
  旧测试来源。训练必须非零 loss/update，执行必须 zero fallback。
- 第 39 项已验证 sdist→从归档构建 wheel→隔离安装/import→Rust codec 执行，并核对
  Torch 依赖、LICENSE 和退役包/plugin 缺席。根 `pyproject.toml` 是唯一 Python 打包配置；
  删除 Rust 子目录重复配置和 `requirements-minimal.txt`，支持 ABI 限定 vLLM 0.27.1。
- 全量行为证据不超过 50，39 项目标完整；SOURCE_LOC 进入 22k–30k；strict typing 0/0、
  import/authority gates、Rust build、代表性 backend 边界和发布验证通过后才算第一轮完成。
  再逐 owner 审计向 10k 压缩，不承诺缺乏代码依据的工期或最终行数。

### 6. 仓库与发布分支整理（G，已执行首轮）

- 30 个 worktree 降至 3 个：27 个干净历史 checkout 已用非 force 的 `git worktree remove`
  移除，所有分支/提交保留，可从原分支重建。没有删除 dirty checkout 或用户改动。
- 保留主目录 `/workspace/pllm`：`main` 上有 5 个用户修改文件及未跟踪研究文档。
  保留 `.worktrees/shared-test-time-lora-runtime`：31 项未提交变更。
  第三个是本实施目录。发布前仍须逐项归属并整合这两处改动，不能直接覆盖或 reset。
- 删除 17 份旧 consumer/Program/PTTS 中间态说明及失效中文镜像，重写 README、文档索引、
  当前架构、验证和 benchmark 指南；Git 历史承担归档。不再维护旧 API 的翻译副本。
- 已补齐原元数据声明的 Apache-2.0 正文。README 明示实际支持范围与开发状态；
  原历史 benchmark harness 已删，旧 n=16 吞吐不能作为当前 n=1 公共路径的支持证明。
- 未发布、未打 tag、未推送。功能/架构证据未齐，以及 dirty 分支整合，仍是发布阻碍。

## 验证记录与复现

`4cd162f` 已验证：pytest 3 passed（CUDA PTTS、Dummy oracle、隔离 wheel）；cargo test 3 passed。
import-layer 0/0；region authority 扫描实际 canonical execution/host/sampler 和 replay，
共 4 个文件；strict/basic Pyright 均 0 errors/0 warnings，basic baseline 已随删除降为 0。
公共 plain→ESamp 同进程 smoke 通过（loss/update 各 2、active_lineages=0、fallback=0）；
In-Place-TTT 双请求 smoke 通过（update=6、dense_apply_rows=8、active_lineages=0、fallback=0）。
CUDA graph replay 为 loss/update 各 4、stable addresses、fallback=0；FTTT 为 loss/update
各 1、version=1、fallback=0。当时 10 项均已执行通过。

本轮新增 GTTA、In-Place CPU oracle 并强化分发包证据后，完整 pytest 5 passed；
import-layer 0/0、region authority gate、strict/basic Pyright 均 0 errors/0 warnings。
FTTT CUDA 重新执行通过：loss/update 各 1、version=1、delta_max=0.0127174、fallback=0。
其余三个 GPU smoke 与 Rust 核心本轮未改、未重复执行，沿用 `4cd162f` 证据。
本轮未运行吞吐 benchmark。

所有 Python 使用 `/workspace/pllm/.venv/bin/python`。canonical vLLM 当前代码要求
`0.27.1`，且限定 in-process/eager 等边界；本轮实测 vLLM runtime 为 0.27.1，发行元数据
为 0.27.1+cu129，Rust extension 已用仓库根目录的 maturin develop 重建。
运行 GPU 证据前仍需检查环境和设备，不能把历史 smoke 日志视作其他版本的支持范围。

```bash
cd /workspace/pllm/.worktrees/tranche-b-atomic
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
