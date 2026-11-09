# 在新模型家族上使用

> **先决条件**：熟悉 `pllm.Workload` 的基本声明（输入、状态与效果），做过[教程 00](../tutorials/00_quickstart.ipynb)。
>
> **本文解决**：如何将写好的负载从默认的 Qwen2.5 迁移到其他模型家族（Qwen3、Qwen3-MoE、Qwen3.5、Llama、DeepSeek V2/V3、gpt-oss、量化检查点或超大切片），掌握准入规则与限制原因，并在不占用 GPU 的情况下完成快速排查。

---

## 1. 准入规则：基于 vLLM 具体模型类准入

当你在 Qwen2.5 上验证完一个算法负载并准备迁移到其他开源模型时，遇到的第一个问题通常是：**pllm 怎么判定一个模型能不能跑？**

pllm 的底层运行时直接按 vLLM 的**具体模型类**进行准入判断，**不支持模型子类**。

- **为什么不支持子类（动机）**：像 Phi-3、Mistral 等近亲变体或第三方派生子类，可能会改变注意力投影的物理布局（例如 QKV 的打包与切片划分规则、交错顺序）或 LM Head / logits 计算与归一化逻辑。pllm 需要精确绑定每一层内部残差流与子层归一化（LayerNorm / RMSNorm）的挂载点、投影输出分片和 logits 计算位置；如果允许任意子类准入，一旦底层张量切片假设失效，就会引发静默的数值错误或未定义行为。因此，必须使用原生 vLLM 模型类。

目前已准入的原生模型类分为四类：

| 架构类别 | 准入的具体模型类 | 适用模型示例 |
| --- | --- | --- |
| **全功能稠密** | `Qwen2ForCausalLM`、`LlamaForCausalLM`、`Qwen3ForCausalLM` | Qwen2 / Qwen2.5、Llama 2 / 3、Qwen3（稠密版） |
| **稀疏 MoE** | `Qwen3MoeForCausalLM` | Qwen3-MoE |
| **混合架构** | `Qwen3_5ForCausalLM`、`Qwen3_5MoeForCausalLM` | Qwen3.5 稠密版与 MoE 版（Gated DeltaNet + 全注意力） |
| **只读与引导** | `DeepseekV2ForCausalLM`、`DeepseekV3ForCausalLM`、`GptOssForCausalLM` | DeepSeek V2 / V3、gpt-oss |

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

1. **投影形状推导范围**：`pllm.explain` 在 CPU 上的配置解析目前覆盖**稠密 Qwen2、Qwen3 和 Llama** 的投影形状计算。
2. **非稠密模型的投影准入时机**：对于其他模型家族（例如 MoE 专家层或混合架构下的非标准层），由于各层投影排布需要在加载真实模块时解析，其具体的投影级准入（projection admission）是在调用 `pllm.make_llm` 创建引擎时判定的。若在 `explain` 中对这些模型推导参数改写（overlay）的投影形状，会提示使用稠密配置进行 CPU 检查。
3. **真实显存占用**：`pllm.explain` 给出的是负载自身状态与工作区的显存需求，不包含模型权重、KV 缓存以及 vLLM 自身的显存占用。模型能否装进显卡仍需在启动引擎时确定。

更多关于成本解析的细节，见[看成本：pllm.explain](../guides/explain.md)。

---

## 3. 各模型家族的功能边界与原因

不同模型家族在注意力机制、KV 存储和前馈网络结构上的差异，决定了它们在 pllm 中的能力边界。各模型的详细特性对照见[支持矩阵](../reference/support-matrix.md)；本节总结各家族的能力边界及背后的设计原因。

### 稠密模型（Qwen2 / Qwen2.5、Llama、Qwen3 稠密版）

- **可用功能**：全部功能。包括 hidden/残差流读取与导出、探针、激活补丁（`pllm.model.ActivationPatch`）、DoLa / logit lens、采样引导（`@pllm.guide_sampling`）、logits 修正（`@pllm.add_logits`）、每请求 LoRA / 权重改写（`pllm.model.Adapter`）、decode 时在线训练（`@pllm.train`）、整模型因果训练、循环深度与 forward 重放（`@pllm.custom_forward`）、`@pllm.custom_residual` 以及 `@pllm.custom_attention`。
- **原因**：这类模型采用标准的多头注意力（MHA）或分组查询注意力（GQA），KV 全部保存在分页缓存中，且 MLP 均为规整的稠密 SwiGLU 结构。子层归一化与残差连接位置统一，投影切片明确。

### Qwen3-MoE

- **可用功能**：只读功能全部可用；`@pllm.custom_forward` 可用；`@pllm.custom_attention` 可用；LoRA 类参数改写可用于注意力的 `q_proj` 和 `v_proj`。
- **受限功能与原因**：
  - **MoE 专家层不支持 LoRA 类改写**：MoE 路由将 token 动态分发给不同的专家，每个专家是一个独立的权重子集，不存在单一的稠密投影矩阵。因此无法直接对 `gate_proj` / `up_proj` / `down_proj` 挂载统一的低秩修正。若尝试改写会报错并建议改写 attention 的 query 或 value 投影。
  - **不支持整模型因果训练**：因果反向传播目前不支持跨路由专家的梯度更新。

### Qwen3.5（Gated DeltaNet 混合注意力，含 MoE 版）

- **可用功能**：只读功能全部可用；在全注意力（full-attention）层上，LoRA 类改写可用于 `v_proj`，以及稠密 MLP 的 `gate_proj` / `up_proj` / `down_proj`。
- **受限功能与原因**：
  - **不支持 `@pllm.custom_forward` 与 `@pllm.custom_attention`**：Gated DeltaNet 层维护的是按 token 递推更新的循环隐藏状态（recurrent state），该状态独立于 vLLM 的分页 KV 缓存。`custom_forward` 和 `custom_attention` 依赖从分页 KV 缓存中读取历史上下文并重放计算，无法重新生成或回溯 DeltaNet 的循环状态。
  - **DeltaNet 层没有 Q/K/V 投影**：DeltaNet 层属于线性循环注意力，没有常规自注意力的 `q_proj` / `k_proj` / `v_proj` 线性层，不能挂载注意力参数改写。
  - **全注意力层的输出门控交错**：Qwen3.5 的全注意力层带有输出门（`attn_output_gate`），其 QKV 权重矩阵中将每个头的 query 向量与 output gate 向量交织打包，无法简单切出纯 query 的线性投影；因此 `q_proj` 改写被拒绝，应改写 `v_proj`。
  - **不支持整模型因果训练**：混合循环状态的反向传播暂未纳入支持。

### DeepSeek V2 / V3（MLA + MoE）与 gpt-oss

- **可用功能**：只读与采样阶段功能。包括 hidden taps（`pllm.hidden`、残差流读取与导出）、线性探针、ESamp（在 decode 阶段基于 hidden 训练请求私有状态）、采样引导（`@pllm.guide_sampling`）、logits 修正（`@pllm.add_logits`）以及激活补丁（`pllm.model.ActivationPatch`）。
- **受限功能与原因**：
  - **不支持参数改写、整模型因果训练与 forward 重放**：DeepSeek 采用多头潜在注意力（MLA），将键值压缩为低维潜在向量存储在潜在 KV 缓存中（没有单独的 V 投影，且 KV 是压缩表示），无法切出独立的 V 投影或重放标准 KV；同时 MLP 采用细粒度路由专家，缺乏单一稠密投影矩阵。gpt-oss 则采用带滑动窗口的 sink attention 与每专家 MXFP4 量化投影。因此这类模型被归入只读与引导架构。

各算法与模型能力的系统梳理，见[能力地图](../guides/capability-map.md)。

---

## 4. 量化底座模型

在实际部署中，许多模型是以 GPTQ、AWQ 或 FP8 检查点的形式提供的。在 pllm 中，**量化底座是一等公民**：

- **准入规则一致**：量化检查点按照其所属模型家族的相同规则准入。例如，AWQ 格式的 Qwen2.5 具备与浮点 Qwen2.5 完全相同的能力。
- **权重改写机制（LoRA / Adapter）**：在量化基座上做 LoRA 或参数覆盖时，pllm 遵循与 vLLM 原生量化 LoRA 相同的机制：在底层量化算子（如 Marlin、Scaled MM 等）的输出张量上，直接累加一份高精度（BF16/FP16）的低秩修正量：

  $$\text{output} = \text{quant\_linear}(x) + \Delta W(x)$$

  这种机制**不需要反量化基座权重，也不需要占用额外的稠密未量化权重显存**。
- **整模型训练机制（QLoRA 式反向传播）**：若在量化模型上执行整模型因果训练（例如 TLM），pllm 采用类似 QLoRA 的机制：前向传播利用高效量化算子，在反向传播计算梯度时，根据量化 scale 与 zero-point 临时反量化权重参与求导，兼顾显存节省与导数计算。

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

pllm 提供了 `tools/model_slice.py` 工具。它利用 HTTP Range 请求按字节范围直接读取远程 Hugging Face 仓库（或本地共享目录）中的权重与 safetensors 索引，仅提取所选解码器层并生成局部切片夹具，无需下载完整权重。

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

### 在 pllm 中加载切片

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

切片沿用 DeepSeek V3 的只读能力边界：可以使用 hidden taps、ESamp 以及采样引导；参数 overlay、整模型因果训练和 PositionState 等写入特性会被拒绝。

---

## 6. 常见准入报错与排查

更换模型家族时，如果负载超出了目标架构的能力边界，pllm 会抛出对应的校验异常。对照下表排查原因与改法：

| 你看到的报错 | 原因与应对方式 |
| --- | --- |
| `choose a Qwen2, Llama, Qwen3, Qwen3Moe, Qwen3_5, Qwen3_5Moe, DeepseekV2, DeepseekV3 or GptOss ForCausalLM model for vLLM workloads` | 传入了未在准入名单中的模型架构。pllm 需要与具体的 vLLM 架构对齐内部残差流与子层归一化位置，目前请选择列表中的模型架构。 |
| `pllm runs vLLM's own model classes, not subclasses of them` | 传入了原生类的派生子类。派生子类可能改变投影布局或 logits 计算，pllm 拒绝子类准入；请使用原生模型类。 |
| `... needs a Qwen2, Llama, Qwen3, Qwen3-MoE or Qwen3.5 model (...); hidden taps, ESamp, sampler stages and activation patches still run on this model` | 试图在只读模型（如 DeepSeek V2/V3 或 gpt-oss）上运行 LoRA 权重改写、整模型因果训练或 forward 重放。这些模型采用压缩的潜在 KV（MLA）或多专家稀疏路由，缺乏单一稠密投影矩阵；请将负载限制在只读观测、ESamp、激活补丁或采样引导阶段。 |
| `... recomputes layer state from KV pages, but ... keeps Gated DeltaNet recurrent state outside them; use a Qwen2, Llama, Qwen3 or Qwen3-MoE model` | 在 Qwen3.5 混合模型上使用了 `@pllm.custom_forward` 或 `@pllm.custom_attention`。Gated DeltaNet 层的循环隐藏状态独立于分页 KV 缓存，无法从 KV 中重算历史状态；重放 forward 或自定义 attention 请使用全注意力稠密模型或 Qwen3-MoE。 |
| `layer ... routes tokens to experts; MLP overlays need a dense MLP, so overlay the attention query or value projection instead` | 试图在 MoE 模型的专家层上挂载 LoRA 改写。每个 token 被动态路由到不同专家，不存在单一稠密的 MLP 矩阵；请将参数改写挂载到注意力的 query 或 value 投影上。 |
| `layer ... is a Gated DeltaNet layer without q/k/v projections; choose a full-attention layer (see layer_types in the model config)` | 试图在 Qwen3.5 的 Gated DeltaNet 层上挂载 q/k/v 投影的 LoRA 改写。该层属于线性循环注意力，没有标准的 q/k/v 投影矩阵；请将改写目标设在全注意力（full_attention）层上。 |
| `gated attention interleaves each head's query rows with its output-gate rows; overlay the value projection instead` | 试图在 Qwen3.5 的门控全注意力层上挂载 `q_proj` 改写。该层将每个头的 query 行与 output-gate 行交错打包，无法直接切出独立的 query 投影；请改写 `v_proj`。 |
| `projection shapes for this architecture are decided at make_llm; use a dense Qwen2, Qwen3 or Llama config for CPU checks` | 在 CPU 上调用 `pllm.explain` 且负载包含参数改写时，使用了非稠密架构的配置。CPU 阶段的配置解析目前覆盖稠密 Qwen2、Qwen3 和 Llama；非稠密架构的投影排布推导在 `make_llm` 时判定，在 CPU 做预检时请使用稠密配置。 |

---

## 7. 自测

1. 为什么在 Qwen3-MoE 模型上，给注意力层的 `v_proj` 挂载 LoRA 可以正常运行，而给 MLP 的 `down_proj` 挂载 LoRA 会被直接拒绝？
<details>
<summary>答案</summary>

Qwen3-MoE 保留了与 Qwen3 相同的注意力结构，注意力层的 `v_proj` 是规整的单一稠密矩阵，可以切出清晰的低秩修正。而其 MLP 层是稀疏专家层，token 会被路由门控分发到许多独立的小专家网络中，不存在单一的稠密投影矩阵，因此 pllm 拒绝在此挂载单一的 MLP LoRA 改写。
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
