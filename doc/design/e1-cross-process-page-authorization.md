# E1 上下文包：跨进程页授权的真实负载

日期：2026-10-04。本文是 E1 任务的完整交接：任务定义、复杂性分析、目标、硬约束与代码导航。
读者是承接该任务的工程协作者（GPT6），默认不熟悉本仓库历史。

## 任务定义

`PageBundleArena.send_field / send_write_field / receive_field` 是一条完整的跨进程页授权链路：
持有方把已签署页的一个字段以只读（或一次私有写）方式导出给 worker 进程，worker 依据自己持有的
签署计划（PlanBundle）校验签名、所有者、访问级别与血缘后获得 `BorrowedPage`。机制本身已实现并有
CPU/CUDA 两侧测试，但**目前只有测试在调用它**——没有任何真实负载消费这条链路。

E1 的任务：让一个真实负载以文档化最佳实践的方式使用这条链路，使其从"只有测试消费的机制"
变成"有真实消费者、用法被文档固化的能力"。

反向出口同样合法：如果充分探索后确认不存在诚实的消费者（所有候选集成点都是在为用而用），
则交付物改为"带证据的删除提案"。优雅是选择不做什么；保留机制必须有真实负载作证。

## 现状事实

- `send_field`（`pllm/execution/ptts/page_bundle.py:799`）：导出一个已 pin 字段的只读授权。
- `send_write_field`（`:810`）：一次写授权恰好授予一个私有字段——"A field may be exported only
  once in a write, preventing overlapping grants"；导出前必须丢弃本地别名。原写持有完成法定人数
  （completion quorum）与原子提交/中止（`:464`, `:814`）。
- `_export`（`:828`）：要求已签署共享存储能力 `state/shared_{device}@1`；每次远程访问强制
  独立的完成通道；CUDA 走 `CudaIpcStorage.export`（`pllm/execution/shared_tensor.py:32`），
  CPU 走共享文件 + `send_handle`。
- `receive_field`（`:851`，staticmethod）：对照 worker 自己的 PlanBundle 校验
  compilation_signature / owner_id / access / lineage，返回 `BorrowedPage`。
- 多进程编解码是封闭载荷注册表（`test_closed_mp_codec_round_trips_registered_payload_and_shared_tensors`），
  spawn 上下文；事务提交/中止/法定人数失败的原子性有专项测试
  （`test_mp_transaction_commit_abort_and_quorum_failure_are_atomic`）。
- 现有调用方：`test/boundary/test_shared_page.py`（CPU）、`test/gpu/test_page_bundle_cuda.py`
  （CUDA，:397/:460）。无生产调用方。
- worker 侧已有跨进程重建协议：`transport_payload / transport_rebuild`（类坐标 + 冻结字段），
  worker 校验签署计划签名并拒绝本地漂移（`test/boundary/test_worker_reconstruction.py`）。
  今天 worker 拿到的是计划与载荷，页数据本身不经过授权链路。

## 复杂性在哪（system codesign）

1. **跨进程所有权与生命周期**：pin/release 跨越进程边界后，持有者崩溃、worker 崩溃、授权
   描述符先于页释放到达等交错都必须有定义良好的语义。惰性页回收与远程完成法定人数
   （quorum）相互耦合：释放时机不能早于最后一个远程消费者确认完成。
2. **CUDA IPC**：`CudaIpcStorage.export` 的句柄生命周期、设备上下文、流语义与页 pin 的
   耦合；接收方 import 后的释放顺序不能留泄漏或悬垂映射。
3. **签署授权边界**：worker 只能消费已签署额度。这条边界必须保持精确——集成时不得为了
   接通而放宽签名校验、能力声明或血缘检查；任何"先放行后补校验"都是不可接受的。
4. **vLLM/EngineCore 集成点**：现有跨进程路径（worker 重建、canonical quorum 检查见
   `pllm/backends/vllm/runtime.py:926,1259`）是单 worker 拓扑。E1 需要在真实负载里找到
   页数据跨进程流动的诚实用途（例如训练/更新类负载的参数页或优化器状态页在 host 与
   worker 间的授权共享），而不是为测试链路再发明一个测试形状的场景。
5. **多进程编解码**：封闭载荷注册表意味着新增跨进程消息类型必须注册进编解码器，且
   spawn 上下文下的可 pickle 性、句柄传递（`send_handle/recv_handle`）都要过注册表。

## 目标（合理范围内的高目标）

1. 一个真实负载（首选：更新/训练类负载，如 inplace_ttt 或 learning 路径的跨进程状态共享；
   次选：EngineCore worker 的页获取路径）通过 `send_field/receive_field` 完成页授权，
   且该用法成为文档化最佳实践（什么时候该用、怎么签、怎么收、生命周期谁负责）。
2. 授权链路的不变量在真实负载下保持：一写一授权、导出前丢弃本地别名、完成法定人数、
   原子提交/中止、worker 侧校验不放宽。
3. 失败语义可演示：法定人数不满足时无部分可见状态；worker 持漂移计划时被拒。
4. 若走反向出口：删除提案须列出所有候选集成点的考察记录与排除理由。

## 硬约束

- **验收门**（全绿才算完成）：完整 Python 套件 + release wheel 测试（`test/release/test_wheel.py`）
  + `cargo test`（21 项）+ typing/layers/authority 三个门禁 + 两个 GPU 冒烟
  （`test/gpu/enginecore_smoke.py` 要求 fallback=0、issued_streams=2；`vllm_recirculation_smoke.py`）。
- **测试纪律**：全局原则是测试行数净减少；E1 教学负载是显式例外，但新增测试必须教机制
  而不是堆断言对冲特判。不得为通过测试而在生产代码里加特判；每个保留的特判 `if` 须注明
  触发条件并指认覆盖它的既有测试，否则删除或改断言。
- **`legal()` 必须保持精确**；ISL 20M op 配额永不放宽。
- 提交只留本地，永不 push。
- 文档同步：`doc/design/implementation-status.md` 是唯一进度入口，完成后更新它。
- 环境：venv 无 pip，用 `uv pip --python .venv/bin/python`；Rust 扩展重建用
  `env -u CONDA_PREFIX .venv/bin/python -m maturin develop`（**不要**加 `--manifest-path`，
  否则会装出一个顶层 `pllm_core` 包而运行时仍加载旧的 `pllm/_core.abi3.so`）。

## 代码导航

| 主题 | 位置 |
| --- | --- |
| 授权链路本体 | `pllm/execution/ptts/page_bundle.py:799-870`（send_field / send_write_field / _export / _send_export / receive_field） |
| 法定人数与原子提交 | `pllm/execution/ptts/page_bundle.py:464, 814` |
| CUDA IPC 存储 | `pllm/execution/shared_tensor.py:32`（CudaIpcStorage） |
| 能力声明 | `state/shared_{device}@1`（contracts/state_representation.py 及签署路径） |
| CPU 消费测试 | `test/boundary/test_shared_page.py`（含 mp 编解码、事务原子性） |
| CUDA 消费测试 | `test/gpu/test_page_bundle_cuda.py:397,460` |
| worker 重建协议 | `test/boundary/test_worker_reconstruction.py`（transport_payload / transport_rebuild / 签名校验） |
| EngineCore 集成 | `pllm/backends/vllm/runtime.py:926,1259`；冒烟 `test/gpu/enginecore_smoke.py` |
| 候选负载 | `pllm/workloads/inplace_ttt.py`、`pllm/execution/learning.py` |
