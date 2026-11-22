# 支持矩阵

本页按模型、功能、vLLM 版本和并行方式列出 pLLM 支持什么。每条限制后面写了原因；不支持的算法族及其原因见
[能力地图](../guides/capability-map.md) 和 [设计取舍](../explain/design-tradeoffs.md)，换模型的具体步骤见
[在新模型家族上使用](../howto/new-model-families.md)。

## vLLM 版本

| 项目 | 支持 | 说明 |
| --- | --- | --- |
| vLLM 版本 | 0.27 – 0.34 | 覆盖 `vllm>=0.27.0,<0.35`。其中 `0.27.1` 与 `0.32.0` 在本仓库跑过完整 GPU 回归；`0.33`–`0.34` 适配了上游接口变更（依赖 PyTorch 2.14 / CUDA 13 驱动） |
| model runner | vLLM 的 MRv2 | pLLM 自己选定 MRv2（在 0.27 等版本自动设置 `VLLM_USE_V2_MODEL_RUNNER=1`），不支持 MRv1 |
| 张量并行 | vLLM 0.32.0 及以上 | 见下文[张量并行](#张量并行) |
| 流水线并行、数据并行 | 不支持 | 必须为 1 |
| 在线服务 | `pllm.make_async_llm` 与 `pllm serve` | `ModelConfig(multiprocessing="auto")` 默认自动为异步句柄与多卡隔离 EngineCore 子进程；支持连续准入、流式 SSE、断连取消、`pllm_route` 会话路由，以及在途 `import_state()` / `train_batch()` / `feed()` 与无负载运行，见 [在线服务指南](../howto/online-serving.md) |

## 模型与功能

pLLM 按 vLLM 的**具体模型类**准入，共覆盖 15 类解码器架构（16 个原生模型类，见 [在新模型家族上使用](../howto/new-model-families.md)），任意未核对子类不予准入。

| 功能 | Qwen2 / 2.5、Llama、Qwen3、Mistral、Phi-3、GLM-4、Gemma 2 / 3 | Mixtral、Qwen2-MoE、Qwen3-MoE | Qwen3.5（Gated DeltaNet 混合，含 MoE 版） | DeepSeek V2 / V3 | gpt-oss |
| --- | --- | --- | --- | --- | --- |
| 读取与导出（`pllm.hidden`、`residual`、`final_hidden`、`layer_logits`）、探针 | ✓ | ✓ | ✓ | ✓ | ✓ |
| `@pllm.add_logits`、`@pllm.guide_sampling`、请求状态训练（ESamp 一类） | ✓（Gemma 2 需稠密引导 ¹） | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.ActivationPatch` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.Adapter`、`@pllm.train_in_forward` | ✓ 全部投影 | q/k/v ² | 全注意力层的 v、稠密 MLP ³ | — ⁵ | — ⁵ |
| `@pllm.custom_residual` | ✓ | ✓（`site="value"` 除外） | ✓（`site="value"` 仅全注意力层） | ✓（`site="value"` 除外） | ✓（`site="value"` 除外） |
| `@pllm.custom_attention`、静态 KV 布局 | ✓ | ✓ | — ⁴ | ✓（MLA 压缩潜变量 ⁶） | — ⁵ |
| `@pllm.custom_forward`、`pllm.PositionState` | ✓ | ✓ | — ⁴ | — ⁵ | — ⁵ |
| 整模型因果训练（`pllm.causal_model`：TLM、TTRL） | ✓（Gemma 2 除外 ⁷） | — ⁷ | — ⁷ | — ⁷ | — ⁷ |
| 只在宿主上训练（`pllm.host.make_host`） | 与模型无关 | 与模型无关 | 与模型无关 | 与模型无关 | 与模型无关 |

1. Gemma 2 的 `final_logit_softcapping` 需要全词表精确引导，不适用有限 top-K 引导；GLM-4 与 Gemma 2 / 3 的四层归一化结构已精确对齐原生分支顺序。
2. 稀疏 MoE 层的 MLP 把 token 路由给专家，无统一稠密投影，因此 `Mixtral`、`Qwen2-MoE` 与 `Qwen3-MoE` 路由层开放 attention `q/k/v`（`Qwen2-MoE` 与 `Qwen3-MoE` 的非路由稠密层同样开放 MLP 增量）；`site="value"` 依赖稠密 `v_proj` 切片。
3. Gated DeltaNet 层没有 q/k/v 投影；Qwen3.5 的全注意力层把每个头的 query 与输出门交错存放，单独改 query 会改到门，所以只开放全注意力层 `v` 投影与稠密 MLP。
4. Gated DeltaNet 层的递归状态不在分页 KV cache 里，pLLM 无法只靠 KV 页重放或改读这些层。
5. DeepSeek V2/V3 的 MLA 无独立 V 投影且 MoE 按专家拆分；gpt-oss 使用带 sink 的滑动窗口 attention 与 MXFP4 专家投影，二者均开放只读、`ActivationPatch` 与非 `value` 的 `@pllm.custom_residual` 改写，不开放 `Adapter` 与整模型训练。
6. DeepSeek V2/V3 上的 `@pllm.custom_attention` 直接对压缩潜变量页（`kv_lora_rank` + `qk_rope_head_dim`）通过矩阵吸收（`W_UK_T` / `W_UV`）计算注意力，仅支持 `keep="all"`，不支持静态 `window` 缓冲。
7. 整模型训练的反向按这 7 类稠密架构实现，并要求 BF16 底座、embedding 与 LM head 不量化；Gemma 2 绑定共享 embedding，没有独立 `lm_head`。

Kimi K2.x 的 INT4 文本塔可以切成六层的 DeepSeek V3 切片做结构测试，功能边界同 DeepSeek V3，见
[在新模型家族上使用](../howto/new-model-families.md)。

## CUDA graph 与 eager

默认 `ModelConfig(cuda_graphs="auto")`：能进 CUDA graph 就进，不能进时模型以 eager 运行，原因写在 `pllm.explain` 的
`forward` 行和 `llm.diagnostics().eager_reason` 里；`cuda_graphs=True` 时这些情况直接报错。

| 写法 | 模型 forward |
| --- | --- |
| 读取与导出、采样效果、请求状态训练（反向与 AdamW 一起捕获）、`@pllm.write` | vLLM 的 CUDA graph |
| `ActivationPatch`、基于 `RequestState` 或 `InstanceState` 的 `Adapter`、`@pllm.custom_residual`、`@pllm.custom_attention`、按 token 历史查表 | vLLM 的 CUDA graph（vLLM 捕获模型原本的算子与覆盖读表，不做 inductor 融合） |
| `@pllm.custom_forward` | 不用 vLLM 的整模型 graph；pLLM 按层段调度并捕获自己的切片图 |
| `cuda_graphs=False` | eager |

## 张量并行

需要 vLLM 0.32.0 及以上，设 `ModelConfig(tensor_parallel_size=N)`（`multiprocessing="auto"` 会自动启用子进程隔离）。

| 可以多卡 | 需要 `tensor_parallel_size=1` |
| --- | --- |
| hidden 读取与导出（只从 rank 0 交付）、采样引导、请求状态训练（ESamp、输出层 LoRA 一类）、`@pllm.custom_forward`、`ActivationPatch` | per-head 适配器增量、带优化器 / 侧链读写或使用 `FromModule` / `ModelCheckpoint` / `Fed` 初始化的 `Adapter` |
| `q/k/v`、`gate/up`、`down` 上基于 `Zeros`/`Computed`/`Pretrained`/`Imported` 的 dense/low-rank 增量（含分片 `@pllm.train_in_forward` 更新） | 整模型因果训练（`pllm.causal_model`）、状态 feed（`pllm.Fed`） |
| 无 `pllm.value(...)` / `pllm.input_tokens` 依赖的 `@pllm.custom_residual`；直接改读本卡本地头 KV 的 `@pllm.custom_attention` | 给 `@pllm.custom_residual` 喂 `pllm.value(...)` / `pllm.input_tokens`，或使用 `pllm.attention(...)` 事实表 / `window` 缓冲的 `@pllm.custom_attention` |

`pllm.explain` 报告的显存预留是每张卡的数。

## 量化与 attention 后端

| 项目 | 支持 | 说明 |
| --- | --- | --- |
| 量化底座 | GPTQ、AWQ、FP8 检查点，按同一家族的规则准入 | `Adapter` 一类改写像 vLLM 对量化模型做 LoRA 一样，在量化算子的输出上加一份高精度修正；整模型训练在反向时临时反量化（QLoRA 的做法）；读 `pllm.operand("weight")` 的方法会让该投影多常驻一份反量化后的权重 |
| 量化 KV（`kv_cache_dtype`） | 与 `@pllm.custom_attention` 不兼容 | 定制读法直接读 KV 页（含 MLA 潜变量页），需要未量化的 K/V |
| attention 后端 | `@pllm.custom_attention` 需要 FlashAttention 2（或 DeepSeek V2/V3 的 MLA 后端） | 其他功能不限后端 |
| 前缀缓存 | 只读、无状态 `@pllm.custom_residual`，以及仅读取不可变 `InstanceState`（非 `Imported`/`Fed`、无写入方法）的 `Adapter` 与 `last_tokens=0` `ActivationPatch` 可开启 | 读取 `RequestState`、可变或外部更新 `InstanceState`、`last_tokens > 0`、`@pllm.train_in_forward` 或 `@pllm.custom_attention` 时必须关闭：缓存里的 KV 依赖请求私有或动态变化的状态 |
