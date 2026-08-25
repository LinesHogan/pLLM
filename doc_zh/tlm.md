# TLM q/v LoRA

内置 TLM consumer 遵循上游 TLM 仓库的实现族：默认在每一层 `q_proj`、
`v_proj` 上挂 PEFT LoRA，rank 8、alpha 16、dropout 0、学习率 `5e-5`。

它与 fast weight 的职责不同。TLM 在 Hugging Face/PEFT side model 上执行完整
causal-LM forward/backward，训练持久化的低秩参数，再把不可变 LoRA 版本交给
vLLM 原生 LoRA manager。请求在准入时固定当时的最新版本，直到完成都不会中途
换权重；训练更新只影响之后进入的请求。

fast weight 则是在 inference forward 内提供按请求隔离、短生命周期的 effective
weight；更新规则、tail 状态和生效时序属于 overlay provider。可复用基建只包括模块
解析、能力准入、逐请求上下文与 wrapper 生命周期。TLM 已有 vLLM 原生 LoRA 路径，
因此不应套用 fast-weight overlay。

安装可选依赖 `pip install -e '.[tlm]'`，用匹配的基础 causal LM 调用
`bind_hf_model()`，并让 vLLM 启用原生 LoRA 且 `max_lora_rank` 不小于配置 rank。
当前发布路径是进程内的；多进程部署还必须在请求准入前把不可变版本传给每个 worker。
