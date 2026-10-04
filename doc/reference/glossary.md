# 词汇表

一个术语一个含义；签名与执行边界使用的就是这些词。

## 创作

- **负载（Workload）** —— 带类型张量方法与声明效果的冻结类（`pllm.Workload`、
  `CompletionWorkload`、`CausalAdapterWorkload`）。编译一次；永远不是运行时钩子。
- **阶段 / 锚点窗口（Phase / anchor window）** —— 两个时间轴锚点
  （`request.prepare`、`model.execute`、`state.publish`、`sampler.pre_draw`、
  `token.commit`、`next.prepare`）之间的签名区间，操作在其中运行。
  `Phase` 是 `When` 窗口的 DSL 拼写。
- **来源标记（Source marker）** —— 命名输入来源的 `Annotated[Tensor, ...]`
  参数：`Hidden`、`Samples`、`Batch`、`Parameter`、`StateIndex`、
  `StateVersion`、`Result`、`BaseLogits`、`SampledTokens`。
- **捕获标记（Capture marker）** —— 训练输入上的第三注解参数
  `Capture(state, field)`：发布时，每个 lineage 的 memory 字段收到该输入
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
  配对的 `after_refs`。失败的发布保留根并允许重试。
- **迁移（Transition）** —— 一次调用的私有写作用域：一个写集，一次发布。
- **Arena** —— 版本/代次/完成的唯一所有者（`PageBundleArena`）。不存在第二份
  状态真相。

## 执行

- **语义图（Semantic graph）** —— 签名的任务/状态/效果声明；规划器的唯一输入。
- **物理计划（Physical plan）** —— 图到锚点、页与缓冲的签名下沉。
- **回退（Fallback）** —— 签名原语之外的执行。其计数永远断言为零。
- **Overlay** —— 模型执行时从状态字段应用的签名参数效果（稠密或低秩），
  与字段的生产方式无关。
- **重放（Replay）** —— 签名重跑一个层区间（`ConditionalLayerReplay`）或逐
  token 循环，由同一发布纪律内的逐层 KV 备份/恢复支撑。同一概念的四个
  名字：`recirculation` 是场景/负载名（`pllm/workloads/recirculation.py`），
  `Recurrence` 是负载作者的声明 DSL（`pllm/execution/replay.py`），
  `TokenwiseLayerReplay` / tokenwise replay 是逐 token 循环的执行机制
  （`pllm/execution/tokenwise_region.py`、`pllm/backends/vllm/replay.py`）。
- **参考调度（Reference schedule）** —— 按程序原始语句顺序排列的那份
  schedule。它没有独立公开符号：构造在 Rust `region_dependence.rs::analyze`
  内部（`reference`），是容量与合法性的基线，也是测量路径的对照组。
- **Route lineage** —— 由用户计算的路由键匹配的请求共享的 lineage，可休眠
  存续；会话关闭时显式退役。
- **证据（Evidence）** —— 编号的、记入 manifest 的可观测契约，带一条能证明
  它的命令。测试扩展现有证据；新 ID 需要真正新的可观测边界。
