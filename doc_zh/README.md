# tLLM 文档

tLLM 是 vLLM 的 test-time intervention 层：把算法声明为一个小型数据流程序，编译器从声明推导快速执行。

## 阅读顺序

1. [快速上手](quickstart.md) — 安装、跑 starter、挂载 consumer。
2. [编写 consumer](write-a-consumer.md) — 程序模型：`@flow`、`@extern`、state bank，以及编译器推导什么。
3. [架构](architecture.md) — 分层结构、import 门禁、程序如何被编译与执行。
4. [ESamp 参考](esamp.md) — 内置的 test-time training consumer。

## 一段话心智模型

你写一个 `torch.nn.Module`，用 `@flow(inputs/outputs/state)` 和 `@extern(inputs/after/state/deadline)` 标注它的方法。`tllm.program.compile_program` 把每个 flow 切成 torch.export 图，把 optimizer 降级为函数式 transition，并在编译期用你的 Python 函数体做 oracle 校验。生成的 consumer 在 vLLM 生成过程中运行：状态写入走训练 lane，只读预测走 reader lane，整个 step 被捕获为 CUDA Graph —— 全部从你声明的依赖边推导。

如果程序用到语言不支持的结构，编译会带稳定错误码（`PG###`）显式失败。没有静默 fallback。

English docs: [doc/README.md](../doc/README.md)。
