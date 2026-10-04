# E3 上下文包：模型与算法迁移（管线驱动）

日期：2026-10-04。本文是路线第 3 步的完整交接：P5 裁决记录、任务分解、阶段顺序、
硬约束与第一阶段任务卡。读者是承接该任务的工程协作者（GPT6）。

## P5 裁决（2026-10-04，用户定案）

用户确认了管线驱动的第 3 步方案（提案 A），要点：

- **生产管线是驱动负载**：编码域视觉编码器——消费 AV1 码流与 NVDEC 中间信息
  （CUDA 13.2 起 H264 可提取 MV 等），输出 hidden state 注入残差流。第 3 步的
  通用化能力以这条管线为第一个真实负载验收，不凭空造通用性。
- **四块全部保留**：通用模型前端、完整异构层、全局 prefill chunk×layer 在线更新、
  剩余训练路径，都是 pLLM 应提供的能力，不裁剪。
- **追加第一项：逐 decode 注入**。用户明确：每个 decode step 可以干预 LLM 是
  pLLM 应有的能力。现状（代码事实）：`Activation` 的 `first_tokens/last_tokens`
  窗口只选 prefill 行（`pllm/backends/vllm/activation.py:1`），decode token 从不
  被注入；请求的 inject 版本在准入时 pin 定、整请求冻结。逐 decode = 两个可独立
  决策的扩展：(a) decode 行进入 patch 窗口；(b) decode 期间注入版本按步刷新。
- **验收口径**：路线表第 3 行原有完成条件（原模型计算保留；后续 chunk 读取正确
  权重版本；KV、更新次数、RNG、取消和 worker 重建全部有端到端证据）+ 逐块端到端
  证据（见各阶段任务卡）。

## 大图：E3 在路线图中的位置

路线表（implementation-status.md"接下来按什么顺序完成整体功能"）第 3 步 =
模型与算法迁移。E3 是它的工程主体。E1（跨进程只读页授权，已关账：
752b4c5/0b35c9e/27f25b5/989d263）为它提供地基：真实的跨进程只读页授权消费者、
所有权分区、pin/ACK 生命周期纪律。E2 已取消。E4（跨请求合批与峰值感知调度）
仍为条件立项，不在 E3 预埋接口。

**环境裁决（2026-10-04，用户定案）**：NVDEC H264 中间信息需要 CUDA 13.2+，
本机驱动 575.57.08（最高 CUDA 12.9）、toolkit 12.4、torch CUDA 12.9，升级要重装
大量环境。**用户决定：不升级，E3-β 用软解 mock**——ffmpeg 软解 + 软件提取的中间
信号（E1 的 `frame_features` 块统计正是这个形状）模拟 NVDEC 中间信息，管线的
工程形态（真实视频文件、持续流、有界积压、背压、逐 decode 接入）全部做真，
只有提取器是 mock。CUDA 13.2 硬件路径等环境自然升级时再立项；**不为它预埋
提取器接口**——届时替换提取函数即可，接口由真实硬件需求定义。

## 交接协议（沿用 E1，逐字不变的部分不重复）

- 禁区不变：`pllm/usage.py`、`pllm/execution/freeze.py`、`tools/`、
  `rust/pllm_core/src/planning/region_dependence.rs` 及区域调度核心。
  **E3-α 极可能需要作者面扩展**（如何声明 decode 窗口与版本刷新）——这是
  usage.py 的作者签名变更，GPT6 只提方案，由 Claude 裁决并动手。
- 设计歧义回来问，不猜；接口形状一律先裁决后实现；GPT6 不 commit/push。
- 每阶段：设计轮（探索+提案）→ Claude 裁决 → 实现 → 本机验收，阶段间停下等审。

## 任务分解与阶段顺序

| 阶段 | 内容 | 为什么在这个位置 |
| --- | --- | --- |
| E3-α | 逐 decode 注入语义 | 管线的直接缺口；设计最重（作者面+版本纪律+KV 语义），先行定调 |
| E3-β | 摄取管线工程化（软解 mock：ffmpeg 软解 + 软件中间信号 → 特征页；持续流、背压、逐 decode 接入） | 不等 CUDA 13.2；α 的语义先定，管线按它接入 |
| E3-γ | 通用模型前端：第二个真实模型族只提供源图+资源绑定 | 管线编码器+第二 LLM 族构成两个真实数据点 |
| E3-δ | 全局 prefill chunk×layer 在线更新 | 路线表原有项，独立验收 |
| E3-ε | 完整异构层 + 剩余训练路径 | 收尾项；异构层定义 = 层类型不再假设齐次 decoder block（视觉编码器层与 LLM 层共存于同一签署计划），用户已确认 |

每阶段的详细任务卡（现状事实/代码导航/验收命令）在该阶段启动时由 Claude 补入本文。

## 阶段任务卡：E3-α 逐 decode 注入语义

### 任务定义

让负载作者能声明 decode 行的注入窗口，并让注入值在 decode 期间按签署纪律
刷新版本。两个子项可以分别裁决：(a) decode 行进入 patch 窗口；(b) 版本按步刷新。

设计轮必须正面回答的问题（探索后给 Claude 裁决，不许猜）：

1. **作者面形状**：decode 窗口怎么声明？扩展 `Activation`（新字段）还是新声明
   类型？窗口语义（绝对位置/相对尾部/全部 decode 行）各自的可组合性与
   签名稳定性。usage.py 改动方案由 Claude 执行。
2. **版本刷新纪律**：decode 期间新版本何时 pin、何时 ACK，与步生命周期、
   sampler、cancel、tokenwise replay 的组合语义。replay 重算 KV 时注入版本
   是否重放？这是真问题，列出选项与后果。
3. **KV 向前干预语义**：decode 时注入 token t 只影响 t 及之后的前向，已缓存
   KV 不回改——作为显式声明语义。前缀缓存交互：注入使 KV 依赖请求外部状态，
   共享前缀的 KV 必须按注入版本键控或拒绝共享。
4. **失败形状**：特征流中断/迟到时 decode 怎么办。候选：响亮停实例（E1 纪律）
   / 冻结最后版本继续（新语义，需用户裁决）/ 其他。列出，不预选。
5. **与 E1 绑定的组合**：逐 decode 刷新复用 BorrowedStateRoot/stage_readonly
   生命周期到什么程度？每步 staged 输入的形状是否变化？

### 目标与验收

- 作者签名扩展最小、语义声明完整（窗口、版本纪律、KV 向前干预写进契约文档）。
- 端到端证据：真实负载下同一请求 decode 期间按步刷新的注入版本生效；
  KV、更新次数、RNG、取消、replay 组合全部有证据；fallback=0。
- 门禁不变：完整 Python 套件 + release wheel + cargo + typing/layers/authority
  + 两个 GPU 冒烟顺序执行。
- 测试纪律不变：净减少原则，教学例外逐文件申报，断言教机制。

### 代码导航（起步点，GPT6 须自行复核）

| 主题 | 位置 |
| --- | --- |
| prefill patch 窗口现状 | `pllm/backends/vllm/activation.py`（`rows.prefill_rows`、first/last_tokens） |
| Activation 作者声明 | `pllm/usage.py:162`（禁区，提案不改） |
| publish 与 inference pin 时序 | `pllm/backends/vllm/runtime.py:371-389`（VllmSession.prepare_input） |
| 借用根生命周期 | `pllm/execution/ptts/page_borrow.py`、`state_execution.py`（stage_readonly/begin/_release） |
| tokenwise replay | `pllm/execution/tokenwise_region.py`、`pllm/backends/vllm/replay.py` |
| 前缀缓存键 | vLLM 侧 prefix caching 与 pLLM 请求键的交点（设计轮探索） |
| E1 视频会话（消费形状参照） | `pllm/workloads/video.py`、`examples/video_session.py`、`test/gpu/test_video_session.py` |

### 明确不在 E3-α

真实 NVDEC/AV1 摄取（E3-β）、第二模型族（E3-γ）、chunk×layer 在线更新（E3-δ）。
不为它们预埋接口。演示负载可以沿用 E1 的近似特征（块统计四字段），
不等待真实管线。
