# 术语表

文档和 `pllm.explain` 输出里出现的词，一个词一个意思。每条后面的链接指向讲它的页面。

## 写负载

- **负载（workload）**：`pllm.Workload` 的子类，只放声明和张量数学。构建引擎时被读一次，之后由 pLLM 安排执行；
  它的方法不是运行时回调。见[负载的组成](../guides/workload-anatomy.md)。
- **声明**：负载回答"读什么、存什么、结果去哪"的三种写法：参数默认值、状态属性、效果装饰器或模型改写属性。
- **输入标记**：写在参数默认值里、说明数据从哪来的值，如 `pllm.hidden(0)`、`pllm.parameter("s", "w")`。
  类型注解只说明方法体拿到什么。见[读取模型内部](../guides/reading.md)。
- **效果**：说明方法的结果去哪的装饰器：`@pllm.export`、`@pllm.add_logits`、`@pllm.guide_sampling`、`@pllm.write`、
  `@pllm.train` 等。见[结果去哪](../guides/results.md)。
- **模型站点**：直接作用在模型 forward 里的声明：`pllm.model.ActivationPatch`、`pllm.model.Adapter`、
  `@pllm.train_in_forward`、`@pllm.custom_residual`、`@pllm.custom_attention`。CPU 上的检查不执行它们。
- **设置方法**：交给装饰器或输入标记、代替固定值的方法，如 `beta=strength` 配 `def strength(self) -> float`。
  pLLM 对每个负载实例调用一次。
- **行（row）**：方法收到的张量第 0 维的一行：一个运行中的请求（或采样器要抽的一个 token、一步算出的一个 token）。
  行的顺序由调度器决定。见[核心概念](../explain/core-concepts.md)。
- **选择器（`rows=`）**：返回 `bool [rows]` 掩码的方法，交给效果的 `rows=`，只让选中的行生效。
- **组件**：作为 dataclass 字段放进另一个负载的负载；名字带字段名前缀，状态互相独立。见[组合负载](../guides/composition.md)。
- **自写算子**：用 `torch.library.custom_op` 包装、在方法里调用的 kernel。见[自写算子](../guides/custom-ops.md)。

## 状态

- **状态**：负载里唯一可变的东西：一组有名字的张量字段，由 pLLM 分配显存、按版本更新。见[状态](../guides/state.md)。
- **请求状态 / 实例状态**：`pllm.RequestState` 每个请求（或每组请求）一份；`pllm.InstanceState` 整个引擎一份。
- **`key_fields`**：决定哪些请求共用一份请求状态：每个请求、同一 prompt 的样本，或同一路由值（`routes=`）。
  见[让一组请求共享训练成果](../howto/share-training-across-requests.md)。
- **字段**：状态里的一个张量。`shape=` 声明一个叫 `"value"` 的字段；`pllm.Field` 给出按投影宽度算的形状或明确的精度。
- **初值来源（`init=`）**：状态的起始值从哪来：`pllm.Zeros`、`pllm.Computed`、`pllm.Pretrained`、
  `pllm.ModelCheckpoint`、`pllm.FromModule`、`pllm.Fed`、`pllm.Imported`。见[加载训练好的权重](../howto/load-trained-weights.md)。
- **冻结状态**：没有优化器的状态：到处可读，只能由导入或写入改变。
- **版本**：状态每次更新都发布一个新版本；读的一方读已发布的版本，在途的请求保留开始时的版本。
  `pllm.version(state)` 是每行的状态已更新的次数。
- **`publish_lag`**：第 n 步算出的更新从第 n + 1 + `publish_lag` 步起被读到；默认 1，换来训练和下一步 forward 的重叠。
- **carry 字段**：`carry=` 声明、随训练更新一起发布但不被优化器更新的字段，由 `pllm.carry` 写入。
- **位置状态（`pllm.PositionState`）**：每个 KV 块一个 hidden 宽的向量，只在 `@pllm.custom_forward` 里用
  `layers.load` / `layers.save` 读写。
- **放置（`placement=`）**：只读的大表放在显存（`"device"`）还是锁页主机内存（`"host"`）。

## 训练

- **请求行训练**：`@pllm.train` 方法读运行中请求的行、每行返回一个 loss，pLLM 按状态归约、反向、更新。
  见[在线训练](../guides/training.md)。
- **归约（`reduction=`）**：同一份状态在一步里有多行时，loss 取平均（`"mean"`）还是求和（`"sum"`）。
- **宿主批次**：宿主代码拼好、经 `generate(batch=...)` 送进来的训练批次：`WeightedRowBatch`、`CausalTrainBatch`、
  `WeightedCEBatch`。见[宿主循环与批训练](../guides/host-loops.md)。
- **因果模型（`pllm.causal_model`）**：训练方法里可调用的整模型，带上指定的 `Adapter`，梯度流向适配器的状态。
- **`HostTrainer`**：`pllm.host.make_host` 返回的对象，在没有模型的情况下用宿主批次训练一个负载的状态。
- **前向内训练（`@pllm.train_in_forward`）**：在 forward 运行时，用投影的输入输出训练一份请求状态，并把它的增量
  加到投影上（TTT、In-Place TTT 一类）。见[改模型](../guides/model-edits.md)。

## 改模型与 forward

- **激活补丁（`ActivationPatch`）**：从状态读值，改某些层的输出（引导向量、ReFT）。
- **Adapter**：把状态当作权重增量加到指定投影上（LoRA 一类），不管状态由谁写。
- **定制残差（`@pllm.custom_residual`）**：接管某些层某个位置的残差更新、value 行或最终的流。
  见[改 forward 结构](../guides/forward-structure.md)。
- **状态子层**：带 `state=`、`chunk=` 的定制残差，每个请求一份按块更新的携带量（快权重记忆）。
- **定制 attention（`@pllm.custom_attention`）**：替换某些层读 KV 的方式；可以同时声明静态 **KV 布局**
  （滑动窗口、prompt 加窗口、跨层共享）。
- **custom forward（`@pllm.custom_forward`）**：用 Python 编排每个 token 跑哪些层、跑几遍、何时停。

## 执行与检查

- **阶段（`pllm.Phase`）**：一步里的时间窗口：`BEFORE_FORWARD`、`IN_FORWARD`、`AFTER_FORWARD`、`BEFORE_SAMPLE`、
  `AFTER_SAMPLE`，以及由 pLLM 决定的 `AUTOMATIC`。每个效果有默认阶段，`@pllm.on(phase)` 可以改。
- **锚点**：`pllm.explain` 打印的、一步里的位置名：`request.prepare`、`model.execute`、`state.publish`、
  `sampler.pre_draw`、`token.commit` 等。
- **主 stream / 旁路 stream（lane）**：`main` 与 forward 串行；`side` 与下一步 forward 重叠。
- **eager 原因**：模型 forward 不在 CUDA graph 里的原因（`eager_reason`），空表示在。
- **中性设置（`neutral=`）**：强度为零的同一个负载；`testing.check` 要求它不改变采样。
- **契约检查**：`pllm.testing.check`（CPU）和 `check_on_gpu`（GPU）检查的性质：行的顺序、请求隔离、分块、抢占重算、
  版本可见性、中性设置。见[检查正确性](../guides/checking.md)。
- **预算**：`ExecutionOptions` 里限制 pLLM 显存的三个值：`state_page_budget_bytes`、`execution_peak_budget_bytes`、
  `cuda_graph_pool_budget_bytes`。见[配置项](config.md)。
