# 架构详解

这篇文档解释 tLLM 的内部架构：数据怎么从 vLLM 里流出来，经过哪些组件，最后到达 consumer。

如果你只想写 consumer，不需要读完这篇。先看 [写你的第一个 Consumer](../getting-started/write-your-first-consumer.md)，遇到不懂的概念再来查。这篇文档主要面向想理解 runtime 工作原理的开发者。

tLLM 的核心设计是把"提取数据"和"使用数据"拆开：

- **Producer** 只管从 packed tensor 中定位并提取需要的行
- **Consumer** 只管拿到这些行之后做什么
- **Runtime** 做桥接：装 hook、组装 bundle、调度 feedback

下面先解释这个设计要解决什么问题，再展开每个组件。

## 从问题出发

假设你想在 vLLM 推理过程中捕获每一层 transformer block 的 hidden state，做分析，然后可选写回影响下一层。

你会遇到三个问题。

### 问题 1：vLLM 把多个请求打包成一个 dense tensor

4 个请求同时跑，vLLM 拼成 `[total_tokens, hidden_size]`。你需要知道"第 3 行属于哪个请求、是不是 decode 阶段"。这叫 **localization**。

### 问题 2：捕获时机和 vLLM 版本强耦合

`_prepare_inputs`、`execute_model`、layer forward 在不同版本里接口会变。把代码写死在 forward 里，每次升级 vLLM 都要重写。

### 问题 3：训练逻辑和推理混在一起

在 forward 里直接启动 ESamp distiller backward，会阻塞主推理线程，吞吐崩掉。需要异步执行，但又必须和主 stream 正确同步。

---

## tLLM 的做法

把"提取数据"和"使用数据"拆开：

- **Producer** 只管一件事：在正确时机从 packed tensor 中定位并提取需要的行
- **Consumer** 只管一件事：拿到这些行之后做什么分析、训练或反馈
- **Runtime** 做桥接：装 hook、组装 bundle、调度 feedback

当前内置的 ESamp 就遵循这个边界：
- Runtime 负责 capture / localization / bundle assembly
- `ESampConsumer` 负责消费 bundle
- `ESampTrainEngine` 负责训练流水线、model-bank、shared-graph

Producer 的改动只影响"怎么找到正确的行"。Consumer 的改动只影响"拿到行之后做什么"。两者独立演进。

---

## 语言与编译器边界

tLLM 的声明模型既是一个小型数据流语言，也是一条编译流水线。正确边界不是按
Python/Rust 文件数量划分，而是按“声明、决策、执行”划分：

| 层 | 负责 | 不负责 |
|----|------|--------|
| Python typed front-end | `ConsumerFlow`、producer descriptor、plugin spec、结构化 port address、typed consumer input view | 跨 flow 的调度决策 |
| Rust compiler mid-end | 启动期校验、provider 选择、依赖 DAG、合法 hook placement、fusion eligibility 元数据 | Tensor、CUDA stream/event、vLLM 对象、plugin 实例 |
| Python runtime executor | vLLM adapter、Tensor/CUDA 物化、bundle 组装、延迟队列、stream/event 启动 | 每 token 重新决定已编译计划 |

FFI 只传 frozen、基于属性的 descriptor，不传嵌套 magic dict 或 Torch 对象。动态结构
探测只保留在外部 API 和 plugin 边界；边界值完成一次校验后，下游只通过 typed
attribute 访问。`PortBundle.entries` 继续作为兼容与扩展 API，内置 consumer 则在入口
处一次性降低为 typed input view。

Producer 对象是声明，不是小型 executor。它描述 outputs、requires、hook points 和
graph safety；真正从 vLLM Tensor 物化 port 的工作仍由 backend driver 负责。

---

## DAG 与自动 placement

启动时，tLLM 把 flow 的读写降低为带 tag 的结构化 `PortAddress`。Rust core 为同一
consumer 的状态顺序、RAW/WAR hazard 和显式排序的 WAW hazard 建边；稳定 Kahn
拓扑排序负责拒绝环并生成执行顺序。

随后 placement 根据 mobility、declared/earliest hook、graph mode、backend 已物化
hook 以及前驱选择计算合法区间。当前算法是受约束的贪心 list scheduling：按拓扑序
依次为每个 flow 最大化以下目标：

1. capture eligibility bonus；
2. 配置或默认的 hook `bubble_score`；
3. 与 declared hook 的接近程度；
4. 稳定 hook 顺序，作为确定性 tie-break。

这不是资源约束项目调度器。当前成本模型没有任务时长、SM/HBM/copy-engine 容量或
传输成本，因此 tLLM 不声称全局最优。fusion 输出也只是 eligibility 元数据：group
必须在拓扑上连续，实际 selected hook 必须等于已物化的 fusion hook，并且使用稳定
lease buffer 与静态 row bucket。kernel 和图优化仍交给 Torch/vLLM 编译系统。

后续算法应由实际优化机会和已有成本数据决定：

| 优化机会 | 合适算法 |
|----------|----------|
| 依赖合法性与稳定顺序 | Kahn；只有需要更丰富 SCC 诊断时才引入 Tarjan |
| 最早/最晚合法 hook 区间 | ASAP/ALAP 约束传播 |
| 有时长/资源 profile 后的在线 deadline | critical-path 或 EDF 驱动的 list scheduling + admission control |
| capture-safe fusion region | 拓扑凸区域构造 |
| 顺序固定后的稳定 buffer 复用 | liveness interval + linear scan/interval coloring |
| 带共享依赖的多 provider 选择 | AND-OR selection；仅在图规模需要时使用 bounded search 或 ILP |

CP-SAT 或穷举适合做小图 oracle；在真实资源/成本模型出现之前引入求解器，只会对
虚构数字做更精确的优化。

---

## 核心抽象：Port

Runtime 和 Consumer 之间不共享 vLLM 内部对象，而是通过正式的数据接口交互。

当前定义的 port 类型：

| Port | 含义 | 例子 |
|------|------|------|
| `residual_stream` | 某层某位置的 hidden state | `ResidualStream.read(layer=0, site="block_output")` |
| `request_meta` | Request identity 信息 | `RequestMeta.read()` |
| `cpu_export` | 异步导出到 CPU | `CpuExport.write(channel="my_analysis")` |
| `logits` | 采样前的 logits | `Logits.read()` |
| `kv_cache` | KV cache 状态 | `KVCache.read(layer=12)` |

Consumer 通过 `ConsumerFlow` 声明自己要读写哪些 port：

```python
ConsumerFlow(
    reads=(
        ResidualStream.read(layer=0, site="block_output", phase="decode"),
        RequestMeta.read(),
    ),
    writes=(CpuExport.write(channel="debug"),),
    window="background",
)
```

Runtime 负责：
- 在 consumer 声明的 layer 上装 hook
- 每步从 packed tensor 中收集这些 port 的数据
- 按 `bundle_key` 聚合成一个 `PortBundle`
- 在合适的 window 调用 `consume_bundle(bundle, ctx)`

---

## 一次 step 内发生了什么

以 decode 为例，从一个 Consumer 的视角看一次 step：

**Step 1：vLLM 准备输入**

`_prepare_inputs` 被调用。Runtime 在这里：
- 快照当前 step 的 request 顺序和 metadata
- 计算 decode localization：哪些行属于 decode phase、对应哪个 prompt

**Step 2：Layer forward**

vLLM 执行模型推理。在 consumer 声明的 layer 上，runtime 的 hook 被触发：
- 用固定 GPU buffer 做 graph-safe gather：从 packed hidden 中选出 decode 行
- 同时 stash prefill 行（如果启用了 prefill producer）

**Step 3：Bundle 组装**

`execute_model` 返回后，runtime 从所有 hook 结果中：
- 提取 consumer 需要的 port 数据
- 按 `bundle_key` 聚合
- 调用 `consumer.consume_bundle(bundle, ctx)`

**Step 4：Feedback（可选）**

如果 consumer 需要 step 末尾动作，或者实现了 `on_step_end()`：
- 在 step 末尾触发 feedback
- ESamp 在这里排队 delayed backward 到 side stream

**Step 5：Capture export（可选）**

如果启用了 capture 模式：
- 把 localized hidden rows 导出到 CPU 存储
- 用于后续的 MSE 对齐验证

---

## Localization

Producer 的核心工作，分 decode 和 prefill 两种。

### Decode localization

vLLM 的 packed tensor 中，每个请求在 decode phase 只占**一行**（当前正在生成的那个 token）。

Producer 的做法：
1. 读取 `logits_indices[i]` —— 第 i 个请求的采样行在 packed tensor 中的位置
2. 筛选出 `is_decode_req=True` 的请求
3. 把这些行索引写入固定 GPU buffer（`decode_row_idx`）
4. 在 hook 中用 `index_select(..., out=decode_hidden_rows_buffer)` gather 到固定 buffer
5. 用 `valid_mask` 标记有效行

为什么用固定 buffer？因为 CUDA Graph 回放时不能每步分配新 tensor。

### Prefill localization

Prefill 不同：每个请求可能对应**一段连续的多行**。

Producer 的做法：
1. 计算 `prefill_len = min(scheduled, prompt_len - computed)`
2. 记录每个请求在 packed tensor 中的 `[start, start + prefill_len)` 范围
3. 用动态 row index tensor 取出这些行
4. 导出到 CPU capture 存储

Prefill 目前走 eager-first 路径，与 decode 的 graph-safe 路径解耦。

---

## Consumer 怎么接入

最简单的方式：继承 `BaseConsumer`，实现三个方法。

```python
class MyConsumer(BaseConsumer):
    @property
    def consumer_id(self):
        return "my_consumer"

    def flows(self):
        return [
            ConsumerFlow(
                reads=(ResidualStream.read(layer=0, site="block_output", phase="decode"),),
                window="background",
            )
        ]

    def consume_bundle(self, bundle, ctx):
        hidden = bundle.entries["hidden"]
        # 你的处理逻辑
```

如果需要 step 末尾动作（drain 队列、启动 backward），再实现 `on_step_end(ctx)`。

完整的扩展教程见 [写你的第一个 Consumer](../getting-started/write-your-first-consumer.md)。

---

## 相关文档

- [术语表](../reference/glossary.md) — 所有核心概念速查
- [代码结构](../reference/project-structure.md) — 改代码时该看哪些文件
