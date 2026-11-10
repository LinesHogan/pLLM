# 支持矩阵

本页按模型、功能、vLLM 版本和并行方式列出 pLLM 支持什么。每条限制后面写了原因；不支持的算法族及其原因见
[能力地图](../guides/capability-map.md) 和 [设计取舍](../explain/design-tradeoffs.md)，换模型的具体步骤见
[在新模型家族上使用](../howto/new-model-families.md)。

## vLLM 版本与 runner

| 项目 | 支持 | 说明 |
| --- | --- | --- |
| vLLM 版本 | 0.27.1、0.32.0 | 其他版本在导入 `pllm` 时报错：pLLM 接入 vLLM 的模型和调度器内部，只对这两个版本做过对齐 |
| runner | `"mrv2"`（默认）、`"mrv1"` | `"mrv1"` 下负载一律以 eager 运行；`pllm.input_tokens`、`PositionState`、`@pllm.custom_forward`、静态 KV 布局需要 `"mrv2"` |
| 张量并行 | vLLM 0.32.0 + `"mrv2"` | 见下文[张量并行](#张量并行) |
| 流水线并行、数据并行 | 不支持 | 必须为 1 |

## 模型与功能

pLLM 按 vLLM 的**具体模型类**准入，子类不算：Phi3、Mistral 这类近亲可能改变投影布局或 logits，pLLM 只对列出的类核对过
投影怎么切、KV 怎么存、层怎么重跑。

| 功能 | Qwen2 / Qwen2.5、Llama、Qwen3（稠密） | Qwen3-MoE | Qwen3.5（Gated DeltaNet 混合，含 MoE 版） | DeepSeek V2 / V3、gpt-oss |
| --- | --- | --- | --- | --- |
| 读取与导出（`pllm.hidden`、`residual`、`final_hidden`、`layer_logits`）、探针 | ✓ | ✓ | ✓ | ✓ |
| `@pllm.add_logits`、`@pllm.guide_sampling`、请求状态训练（ESamp 一类） | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.ActivationPatch` | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.Adapter`、`@pllm.train_in_forward` | ✓ 全部投影 | q/k/v ¹ | 全注意力层的 v、稠密 MLP ² | — ⁴ |
| `@pllm.custom_residual` | ✓ | ✓ | ✓（`site="value"` 只在全注意力层） | — ⁴ |
| `@pllm.custom_attention`、静态 KV 布局 | ✓ | ✓ | — ³ | — ⁴ |
| `@pllm.custom_forward`、`pllm.PositionState` | ✓ | ✓ | — ³ | — ⁴ |
| 整模型因果训练（`pllm.causal_model`：TLM、TTRL） | ✓ ⁵ | — ⁵ | — ⁵ | — ⁵ |
| 只在宿主上训练（`pllm.host.make_host`） | 与模型无关 | 与模型无关 | 与模型无关 | 与模型无关 |

1. Qwen3-MoE 的 MLP 把 token 路由给专家，没有一个统一的稠密 MLP 投影可以加增量，所以只能改 attention 的投影。
2. Gated DeltaNet 层没有 q/k/v 投影；Qwen3.5 的全注意力层把每个头的 query 与输出门交错存放，单独改 query 会改到门，
   所以只开放 v 投影。
3. Gated DeltaNet 层的递归状态不在分页 KV cache 里，pLLM 无法只靠 KV 页重放或改读这些层。
4. DeepSeek V2/V3 用 MLA（没有单独的 V 投影，KV cache 存的是压缩后的潜变量）和按专家拆分的 MoE 投影；gpt-oss 用带
   sink 的滑动窗口 attention 和 MXFP4 的专家投影。pLLM 没有为这些布局实现改写，只开放只读功能。
5. 整模型训练的反向是 pLLM 按稠密 Qwen2、Llama、Qwen3 的结构写的，并要求 BF16 底座、embedding 与 LM head 不量化。

Kimi K2.x 的 INT4 文本塔可以切成六层的 DeepSeek V3 切片做结构测试，功能边界同 DeepSeek V3，见
[在新模型家族上使用](../howto/new-model-families.md)。

## CUDA graph 与 eager

默认 `ModelConfig(cuda_graphs="auto")`：能进 CUDA graph 就进，不能进时模型以 eager 运行，原因写在 `pllm.explain` 的
`forward` 行和 `llm.diagnostics().eager_reason` 里；`cuda_graphs=True` 时这些情况直接报错。

| 写法 | 模型 forward |
| --- | --- |
| 读取与导出、采样效果、请求状态训练（反向与 AdamW 一起捕获）、`@pllm.write` | vLLM 的 CUDA graph |
| `ActivationPatch`、请求状态上的 `Adapter`、`@pllm.custom_residual`、`@pllm.custom_attention`、按 token 历史查表 | vLLM 的 CUDA graph（vLLM 捕获模型原本的算子，不做 inductor 融合） |
| `@pllm.custom_forward` | 不用 vLLM 的整模型 graph；pLLM 按层段调度并捕获自己的 graph |
| 读共享 `InstanceState` 的 `Adapter` | eager |
| `runner="mrv1"`、`cuda_graphs=False` | eager |

## 张量并行

需要 vLLM 0.32.0 和 `runner="mrv2"`，设 `ModelConfig(tensor_parallel_size=N)`。

| 可以多卡 | 需要 `tensor_parallel_size=1` |
| --- | --- |
| hidden 读取与导出（只从 rank 0 交付）、采样引导、请求状态训练（ESamp、输出层 LoRA 一类）、`@pllm.custom_forward` | per-head 改写、带优化器或从外部来源初始化（导入、checkpoint、文件、模块）的 `Adapter` 状态 |
| q/k/v、gate/up、down 上的 dense / low-rank 增量，按投影的分片切开，由 `@pllm.train_in_forward` 用本卡分片的数据更新 | 整模型因果训练、状态 feed（`pllm.Fed`） |
| | `@pllm.custom_residual`（以及给它喂 `pllm.input_tokens` 的方法）、`@pllm.custom_attention`（每张卡只看到自己的头） |

`pllm.explain` 报告的显存预留是每张卡的数。

## 量化与 attention 后端

| 项目 | 支持 | 说明 |
| --- | --- | --- |
| 量化底座 | GPTQ、AWQ、FP8 检查点，按同一家族的规则准入 | `Adapter` 一类改写像 vLLM 对量化模型做 LoRA 一样，在量化算子的输出上加一份高精度修正；整模型训练在反向时临时反量化（QLoRA 的做法）；读 `pllm.operand("weight")` 的方法会让该投影多常驻一份反量化后的权重 |
| 量化 KV（`kv_cache_dtype`） | 与 `@pllm.custom_attention` 不兼容 | 定制读法直接读 KV 页，需要未量化的 K/V |
| attention 后端 | `@pllm.custom_attention` 需要 FlashAttention 2 | 其他功能不限后端 |
| 前缀缓存 | 只读负载可以打开 | 改写模型站点（`custom_residual`、`custom_attention`、overlay、激活补丁）时必须关闭：缓存里的 KV 是在别的改写下算出的，复用会让结果不对 |
