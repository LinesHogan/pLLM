# 在新模型家族上使用

> **先决条件**：熟悉 `pllm.Workload` 的基本声明（输入、状态与效果），做过[教程 00](../tutorials/00_quickstart.ipynb)。
>
> **本文解决**：如何将写好的负载从默认的 Qwen2.5 迁移到其他模型家族（Qwen3、Qwen3-MoE、Qwen3.5、Llama、DeepSeek V2/V3、gpt-oss、量化检查点或超大切片），掌握准入规则与限制原因，并在不占用 GPU 的情况下完成快速排查。

---

## 1. 准入规则：基于 vLLM 具体模型类准入

当你在 Qwen2.5 上验证完一个算法负载并准备迁移到其他开源模型时，遇到的第一个问题通常是：**pLLM 怎么判定一个模型能不能跑？**

pLLM 的底层运行时直接按 vLLM 的**具体模型类**进行准入判断，**不支持未核对的任意子类**。

- **为什么按具体模型类准入**：不同变体或第三方派生子类可能改变注意力投影的物理布局（例如 QKV 的打包、切片划分、交错门控顺序）、子层归一化位置（如 GLM-4 与 Gemma 2 / 3 的四处归一化）或 LM Head 绑定与 logits 缩放逻辑（如 Gemma 2 的共享 embedding 与 `final_logit_softcapping`）。pLLM 会核对每一类原生模型内部的残差挂载点、投影输出分片与权重布局；因此仅准入以下已核对的原生 vLLM 模型类。

目前已准入 15 类架构（16 个原生模型类），按能力分为五组：

| 架构类别 | 准入的具体模型类 | 适用模型示例 |
| --- | --- | --- |
| **全功能稠密** | `Qwen2ForCausalLM`、`LlamaForCausalLM`、`Qwen3ForCausalLM`、`MistralForCausalLM`、`Phi3ForCausalLM`、`Glm4ForCausalLM`、`Gemma2ForCausalLM`、`Gemma3ForCausalLM` | Qwen2 / Qwen2.5、Llama 2 / 3、Qwen3（稠密版）、Mistral、Phi-3、GLM-4、Gemma 2 / 3 |
| **稀疏 KV-MoE** | `MixtralForCausalLM`、`Qwen2MoeForCausalLM`、`Qwen3MoeForCausalLM` | Mixtral、Qwen2-MoE、Qwen3-MoE |
| **混合架构** | `Qwen3_5ForCausalLM`、`Qwen3_5MoeForCausalLM` | Qwen3.5 稠密版与 MoE 版（Gated DeltaNet + 全注意力） |
| **潜变量 MLA** | `DeepseekV2ForCausalLM`、`DeepseekV3ForCausalLM` | DeepSeek V2 / V3（支持潜变量 `@pllm.custom_attention` 与非 `value` 子层残差） |
| **滑动窗融合** | `GptOssForCausalLM` | gpt-oss（支持只读、`ActivationPatch` 与非 `value` 子层残差） |

如果传入未准入的架构或派生子类，引擎初始化时会直接拒绝并明确提示。

---

## 2. 快速检查：在 CPU 上用 `pllm.explain` 预检

大模型检查点动辄数十 GB。如果每次更换模型都要先下载权重、占满 GPU 显存启动引擎才能发现架构不兼容，调试成本极高。

你可以直接在 CPU 上调用 `pllm.explain(workload, config)` 快速预检。它只读取检查点目录下的 `config.json`，在 CPU 上推导形状并执行准入校验，抛出的准入异常与启动真实 GPU 引擎（`pllm.make_llm`）完全一致：

```python
import torch
import pllm
from pllm import testing

class ObserveResidual(pllm.Workload):
    @pllm.export(name="residual_norm", max_rows=8, max_cols=1)
    def norm(
        self,
        r: torch.Tensor = pllm.residual(0),
    ) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)

config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
report = pllm.explain(ObserveResidual(), config)
assert report.budget_errors == ()
assert testing.check(ObserveResidual(), config).passed
```

### explain 的边界与须知

使用 `pllm.explain` 预检新模型时，需要注意以下边界：

1. **投影与结构解析范围**：`pllm.explain` 在 CPU 上即可根据模型配置（`config.json`）解析全部 8 类稠密解码器、3 类稀疏 KV-MoE 解码器（`Mixtral` / `Qwen2-MoE` / `Qwen3-MoE`）、`Qwen3.5` 混合架构以及 `DeepSeek V2/V3`（含 MLA 潜变量尺寸）的投影与注意力布局。
2. **模型类准入时机**：只要配置里声明了已核对的 `architectures` 模型类，`pllm.explain` 就会立即校验对应的功能支持矩阵；真正的模型权重布局、量化算子（如 gpt-oss 的 MXFP4 专家或自定义量化配置）与可用 KV Cache 容量会在调用 `pllm.make_llm` / `pllm.make_async_llm` 时由引擎核实。
3. **真实显存占用**：`pllm.explain` 给出的是负载自身状态与工作区的显存需求，不包含模型权重、KV 缓存以及 vLLM 自身的显存占用。模型能否装进显卡仍需在启动引擎时确定。

更多关于成本解析的细节，见[看成本：pllm.explain](../guides/explain.md)。

---

## 3. 各模型家族的功能边界与原因

不同模型家族在注意力机制、KV 存储和前馈网络结构上的差异，决定了它们在 pLLM 中的能力边界。各模型的详细特性对照见[支持矩阵](../reference/support-matrix.md)；本节总结各家族的能力边界及背后的设计原因。

### 稠密模型（Qwen2 / Qwen2.5、Llama、Qwen3、Mistral、Phi-3、GLM-4、Gemma 2 / 3）

- **可用功能**：全部核心功能。包括 hidden/残差流读取与导出、探针、激活补丁（`pllm.model.ActivationPatch`）、DoLa / logit lens、采样引导（`@pllm.guide_sampling`）、logits 修正（`@pllm.add_logits`）、全部投影的 LoRA / 权重改写（`pllm.model.Adapter`）、decode 时在线训练（`@pllm.train`）、循环深度与 forward 重放（`@pllm.custom_forward`）、`@pllm.custom_residual`、`@pllm.custom_attention`，以及除 Gemma 2 之外的整模型因果训练（`pllm.causal_model`）。
- **原因与差异细节**：这类模型采用标准多头或分组查询注意力，KV 全部保存在分页缓存中，MLP 均为稠密门控/前馈结构，投影切片明确；GLM-4 与 Gemma 2/3 使用四层归一化分支布局（已在残差位点与因果训练前馈分支中对齐）。Gemma 2 因词表嵌入与输出权重共享且带有 `final_logit_softcapping`，不开放 `pllm.causal_model` 整模型反向并需使用全词表精确采样引导。

### 稀疏 KV-MoE（Mixtral、Qwen2-MoE、Qwen3-MoE）

- **可用功能**：只读与采样引导功能全部可用；`@pllm.custom_forward`、`@pllm.custom_attention`、除 `site="value"` 外的 `@pllm.custom_residual` 均可用；`pllm.model.Adapter` 可挂载到全部层的注意力 `q_proj`、`k_proj`、`v_proj`，以及 `Qwen2-MoE` / `Qwen3-MoE` 中非路由稠密层的 MLP 投影。
- **受限功能与原因**：
  - **MoE 专家层不支持 MLP `Adapter`**：路由层将 token 动态分发给不同专家，不存在单一稠密 MLP 投影矩阵，因此无法挂载统一的专家层增量（会报错并提示改用注意力的 query/key/value 投影）。
  - **不开放 `site="value"` 残差与整模型因果训练**：`site="value"` 依赖独立稠密 V 投影切片约定，而因果反向传播暂不覆盖跨路由专家的梯度更新。

### Qwen3.5（Gated DeltaNet 混合注意力，含 MoE 版）

- **可用功能**：只读与采样引导全部可用；全注意力层的 `@pllm.custom_residual`（全部 site）以及所有层上的非 `site="value"` `@pllm.custom_residual` 可用；`Adapter` 可用于全注意力层的 `v_proj` 与稠密 MLP 投影。
- **受限功能与原因**：
  - **不支持 `@pllm.custom_forward` 与 `@pllm.custom_attention`**：Gated DeltaNet 层维护按 token 递推更新的循环隐藏状态，独立于分页 KV 缓存，无法仅靠 KV 页重放或替代读出。
  - **DeltaNet 层没有标准 Q/K/V 投影**，全注意力层则将 query 与输出门（`attn_output_gate`）交错打包，因此仅开放 `v_proj` 与稠密 MLP。
  - **不支持整模型因果训练**：混合循环状态的反向传播暂未纳入支持。

### DeepSeek V2 / V3（MLA + MoE）与 gpt-oss

- **可用功能**：只读、`@pllm.add_logits` / `@pllm.guide_sampling`、ESamp、`ActivationPatch`、不依赖 `site="value"` 的子层 `@pllm.custom_residual`（如 `stream` / `mlp` / `attention`）；此外，**DeepSeek V2 / V3 开放压缩潜空间 `@pllm.custom_attention`**。
- **技术说明与受限原因**：
  - **DeepSeek V2 / V3 的潜变量 `@pllm.custom_attention`**：MLA 将 KV 压缩为潜变量 `c_kv` 与解耦旋转位置编码部分存储于分页缓存，无独立 `v_proj`。pLLM 在执行自定义注意力时直接以压缩潜变量为存储单元，并通过吸收矩阵 `W_UK_T` / `W_UV` 在不显式还原全量 KV 缓存的情况下完成分数与加权和计算（要求 `keep="all"` 且不支持静态 `window` 缓冲）。
  - **不支持 `Adapter`、整模型因果训练与 `@pllm.custom_forward`**：DeepSeek 的专家层按路由拆分、gpt-oss 使用带 sink 的滑动窗口 attention 与 MXFP4 专家投影，均未开放参数覆盖与整模型训练。

各算法与模型能力的系统梳理，见[能力地图](../guides/capability-map.md)。

---

## 4. 量化底座模型

在实际部署中，许多模型是以 GPTQ、AWQ 或 FP8 检查点的形式提供的。在 pLLM 中，**量化底座是一等公民**：

- **准入规则一致**：量化检查点按照其所属模型家族的相同规则准入。例如，AWQ 格式的 Qwen2.5 具备与浮点 Qwen2.5 完全相同的能力。
- **权重改写机制（LoRA / Adapter）**：在量化基座上做 LoRA 或参数覆盖时，pLLM 遵循与 vLLM 原生量化 LoRA 相同的机制：在底层量化算子（如 Marlin、Scaled MM 等）的输出张量上，直接累加一份高精度（BF16/FP16）的低秩修正量：

  $$\text{output} = \text{quant\_linear}(x) + \Delta W(x)$$

  这种机制**不需要反量化基座权重，也不需要占用额外的稠密未量化权重显存**。
- **整模型训练机制（QLoRA 式反向传播）**：若在量化模型上执行整模型因果训练（例如 TLM），pLLM 采用类似 QLoRA 的机制：前向传播利用高效量化算子，在反向传播计算梯度时，根据量化 scale 与 zero-point 临时反量化权重参与求导，兼顾显存节省与导数计算。

```python
# check-docs: skip: quantized model execution requires CUDA
from vllm import SamplingParams

quant_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct-AWQ",  # AWQ 量化检查点，按 Qwen2 家族准入
    max_num_seqs=8,
)

with pllm.make_llm(quant_config, workload=ObserveResidual()) as llm:
    outputs = llm.generate(["测试量化底座"], SamplingParams(max_tokens=8))
```

---

## 5. 超大模型结构验证：model_slice 与 Kimi INT4 文本切片

研究或验证算法时，经常需要确认负载在千亿参数大模型（如 DeepSeek V3 或 Kimi K2.5 的 MLA 结构、384 路由专家）上的张量通路与结构兼容性。然而，单张显卡或常规服务器通常无法装下完整模型（动辄需要数百 GB 显存）。

pLLM 提供了 `tools/model_slice.py` 工具。它利用 HTTP Range 请求按字节范围直接读取远程 Hugging Face 仓库（或本地共享目录）中的权重与 safetensors 索引，仅提取所选解码器层并生成局部切片夹具，无需下载完整权重。

### Kimi INT4 文本切片配方

`tools/model_slice.py` 支持 Kimi K2-Thinking、K2.5–K2.7 的 compressed-tensors 对称 INT4 group-32 权重；不做反量化（注意：K2-Instruct 的 FP8 提取仍不支持）。

默认选取源层 `[0, 1, 30, 31, 59, 60]`（首、中、尾各两层，共 6 层），保留 `first_k_dense_replace=1`、MLA 结构与全部 384 个路由专家，去掉视觉塔与 MTP。**它是断开的六层数值夹具，用于验证执行图与张量通路，不能代表完整模型的生成质量。**

切片命令如下（建议先加 `--plan-only` 检查大小与依赖）：

```bash
python tools/model_slice.py \
  --repo moonshotai/Kimi-K2.5 \
  --revision 4d01dfe0332d63057c186e0b262165819efb6611 \
  --text-only --output model-slices/moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611
```

提取出的张量 payload 约为 50.98 GiB（54,735,407,616 字节），另有元数据与分词器。输出配置展开为 `DeepseekV3ForCausalLM`，并去掉了 `language_model.*` 前缀与视觉/MTP 包装。

### 分词器（Tokenizer）关键约定

Tokenizer 必须使用该版本的 `tokenization_kimi.py`、`tiktoken.model`、`tokenizer_config.json`，以及 tokenizer 导入的 `tool_declaration_ts.py`。切片工具会自动将它们连同 `LICENSE`、chat template 与 generation config 保留在切片目录中。

切片的模型配置去掉了 `auto_map`，模型本身直接走 vLLM 原生的 DeepSeek V3 实现，只有分词器需要远程代码许可。在配置时**必须显式指定 `trust_remote_code=True`**。不要省掉该参数而依赖 Transformers 的自动 tokenizer 转换：当前环境下 Transformers 转换对 `"I'm"` 会编码为两个 token `[40, 14404]`，而 Moonshot 原生实现编码为单个 token `[17172]`。

### 开源许可说明（License Note）

K2.5 采用 Modified MIT License，允许本地使用，要求保留版权与许可信息；其额外条款要求超过指定月活/月收入门槛的商业产品显示 “Kimi K2.5”。完整条款随切片目录保留。

### 在 pLLM 中加载切片

单张 A100 GPU 即可运行该切片的 INT4 Marlin MoE 内核，显存需求约 51 GiB：

```python
# check-docs: skip: Kimi slice execution requires CUDA and local slice checkpoint
from vllm import SamplingParams

slice_config = pllm.ModelConfig(
    "model-slices/moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611",
    trust_remote_code=True,
    cuda_graphs=False,
)

with pllm.make_llm(slice_config, workload=ObserveResidual()) as llm:
    outputs = llm.generate(["测试 Kimi 切片"], SamplingParams(max_tokens=8))
```

切片沿用 DeepSeek V3 的能力边界：支持 hidden taps、ESamp、采样引导、`ActivationPatch`、非 `value` 的 `@pllm.custom_residual` 以及潜变量 `@pllm.custom_attention`；参数 overlay、整模型因果训练和 `PositionState` 仍会被拒绝。

---

## 6. 常见准入报错与排查

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[引擎设置、模型与并行](../reference/errors.md#引擎设置模型与并行)、[改模型与 forward 结构](../reference/errors.md#改模型与-forward-结构)几节。

---

## 7. 自测

1. 为什么在 Qwen3-MoE 模型上，给注意力层的 `v_proj` 挂载 LoRA 可以正常运行，而给 MLP 的 `down_proj` 挂载 LoRA 会被直接拒绝？
<details>
<summary>答案</summary>

Qwen3-MoE 保留了与 Qwen3 相同的注意力结构，注意力层的 `v_proj` 是规整的单一稠密矩阵，可以切出清晰的低秩修正。而其 MLP 层是稀疏专家层，token 会被路由门控分发到许多独立的小专家网络中，不存在单一的稠密投影矩阵，因此 pLLM 拒绝在此挂载单一的 MLP LoRA 改写。
</details>

2. 为什么在加载 Kimi K2.5 切片时必须显式指定 `trust_remote_code=True`，而模型本身却能使用 vLLM 原生的 DeepSeek V3 算子运行？
<details>
<summary>答案</summary>

切片工具生成切片配置时去掉了模型的 `auto_map`，模型骨干网络直接复用 vLLM 内置的原生 `DeepseekV3ForCausalLM` 实现，因此模型前向本身不需要远程代码。但 Kimi 的分词逻辑依赖自定义的 `tokenization_kimi.py`，如果省略 `trust_remote_code=True`，Transformers 会自动回退到通用分词转换，导致词元切分不一致（例如将 `"I'm"` 错误切分为两个 token）。
</details>

---

## 8. 下一步去哪

- 要查阅所有模型家族与具体算法特性的完整对应表 → [支持矩阵](../reference/support-matrix.md)
- 要了解不同推理算法的设计取舍与支持现状 → [能力地图](../guides/capability-map.md)
- 要在 CPU 上精确估算显存占用与执行计划 → [看成本：pllm.explain](../guides/explain.md)
- 要排查更多模型加载与运行时报错 → [排查报错](troubleshooting.md)
