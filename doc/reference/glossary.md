# 词汇表

一个术语一个含义；签名与执行边界使用的就是这些词。

## 创作

- **负载（Workload）** —— 带类型张量方法与声明效果的冻结类（`pllm.Workload`）。编译一次；永远不是运行时钩子。
- **阶段 / 锚点窗口（Phase / anchor window）** —— 两个时间轴锚点
  （`request.prepare`、`model.execute`、`state.publish`、`sampler.pre_draw`、
  `token.commit`、`next.prepare`）之间的签名区间，操作在其中运行。
  `Phase` 是 `When` 窗口的 DSL 拼写。
- **来源标记（Source marker）** —— 命名输入来源的 `Annotated[Tensor, ...]`
  参数：`Hidden`、`Samples`、`Batch`、`Parameter`、`StateIndex`、
  `StateVersion`、`Result`、`BaseLogits`、`SampledTokens`。
- **Hidden / BLOCK_OUTPUT** —— `pllm.hidden(L)` 读取第 L 个 decoder 层的 MLP 输出，
  位于残差加法之前；它不是残差流（残差流见 `pllm.residual`）。每个请求读取最新 decode token 或 prompt 最后一个 token。
- **残差流读取（pllm.residual）** —— 某层之后（或 attention 与 MLP 之间）的残差流：
  下一个子层的 norm 读入的和。steering 向量、SAE 与 logit lens 读的是它；`pllm.hidden(L)`
  只是第 L 层 MLP 在加入残差流之前的输出。
- **层 logits（pllm.layer_logits）** —— 模型自己的最终 norm 与 LM head 作用于若干层之后的
  残差流（logit lens），FP32 `[rows, layers, vocab]`，覆盖采样器抽取的每个 token。
- **效果组合（Effect composition）** —— 一个负载里的效果各自声明位置与状态读写、
  并排生效；只有真实冲突（同一位置的两个所有者、同一状态的两个写者等）被拒绝。
- **因果批（Causal batch）** —— `pllm.host.CausalTrainBatch` 提供同形的
  `input_ids`、`labels` 和可选 `attention_mask`，训练方法用 `pllm.batch(...)` 读取。
- **因果模型端口（Causal model port）** —— `pllm.causal_model("adapter")` 注入
  `pllm.model.CausalModel`；`model(...)` 应用命名 Adapter，`model.base(...)` 只用冻结基座。
- **属性引用（pllm.Attr）** —— `pllm.Attr("phase")` 在绑定时读取负载属性；
  `@pllm.on(...)` 可据此决定阶段，如 TLM 的在线/离线训练时机。
- **张量结果（StepResult）** —— 一次执行的报告值、loss、发布与观测；
  `metrics` 保存算法提供的设备张量，TLM 的 `selected_count` 是选中的句子数；
  宿主训练步的 `parameters` 是更新后的状态字段。
- **携带标记（pllm.carry）** —— 训练输入上的第三注解参数
  `pllm.carry(source, state=, field=)`：发布时，每个 lineage 的 carry 字段收到该输入
  最后一个选中行的值。
- **指纹（Fingerprint）** —— 所创作数学的冷结构签名（marshaled 字节码加
  传递闭包的全局/自由变量），不求值注解、不执行代码。改动数学会重新签名；
  装载从不重新执行。
- **冻结（Freeze）** —— 对所创作负载做的 detach 深拷贝，函数值被递归冻结，
  之后对创作环境的改动无法漂移已签名的行为。

## 状态

- **状态（State）** —— 唯一可变的东西：声明布局下的具名、带版本张量字段。
  三种生产方式、消费完全相同：优化器更新、超网络/外部导入、checkpoint 装载。
  无优化器时状态是冻结的：准入时初始化、到处可读（观测也行）、永不更新。
- **Lineage** —— 一个逻辑状态的版本链。实例作用域一条；请求作用域每个准入键
  一条。执行从不合并或拆分 lineage。
- **Memory 字段** —— 跟随训练迁移但不参与优化器展开的每 lineage 普通字段
  （无梯度/动量槽）。读的是输入版本：第 N+1 步看到第 N 步发布的值。
  页在准入时零初始化。
- **版本（Version）** —— lineage 的不可变根。读取 pin 住一个版本；写入产生
  私有候选；发布使其原子可见。
- **输入版本（Input version）** —— 步开始时 pin 住的版本；这一步里所有输入
  版本读看到的都是它 —— 跨步 memory 语义因此零成本。

## PTTS（分页张量状态）

- **页（Page）** —— 字段存储的分配单位；槽携带代次（generation），过期引用
  响亮失败而不是错位别名。
- **Pin** —— 对发布版本的精确持有。回收是精确的：页只在引用它的每个 pin 与
  发布都释放后才归还。
- **发布（Publication）** —— 使私有迁移原子可见的动作，产出与 `before_refs`
  配对的 `after_refs`。失败的发布保留根并允许重试。TLM 空选择仍发布新版本，
  字段与优化器状态不变，逻辑更新计数不增加；缺少训练输入或 `lamb=0` 则不发布。
- **迁移（Transition）** —— 一次调用的私有写作用域：一个写集，一次发布。
- **Arena** —— 版本/代次/完成的唯一所有者（`PageBundleArena`）。不存在第二份
  状态真相。

## 执行

- **语义图（Semantic graph）** —— 签名的任务/状态/效果声明；规划器的唯一输入。
- **物理计划（Physical plan）** —— 图到锚点、页与缓冲的签名下沉。
- **回退（Fallback）** —— 签名原语之外的执行。其计数永远断言为零。
- **Adapter** —— 模型执行时从状态字段应用的签名参数效果（稠密或低秩），
  与字段的生产方式无关。`Projection(layer=None, ...)` 表示所有层；
  TLM 用一个 Adapter 让训练与推理读取同一组 q/v 因子。
- **重放（Replay）** —— `custom_forward` 定义逐 token 的层调用与返回 residual；
  `recirculation` 是使用该接口的负载。组合一个训练 overlay 时，
  `layers.apply_updates()` 消费捕获并改变本 token 后续读取的私有字段。
  执行使用 `tokenwise_region.py` 和 vLLM 的稳定 token CUDA graph；状态仍由原
  transition 与 publication 拥有。
- **参考调度（Reference schedule）** —— 按程序原始语句顺序排列的那份
  schedule。它没有独立公开符号：构造在 Rust `region_dependence.rs::analyze`
  内部（`reference`），是容量与合法性的基线，也是测量路径的对照组。
- **Route lineage** —— 由用户计算的路由键匹配的请求共享的 lineage，可休眠
  存续；会话关闭时显式退役。
- **证据（Evidence）** —— 编号的、记入 manifest 的可观测契约，带一条能证明
  它的命令。测试扩展现有证据；新 ID 需要真正新的可观测边界。
