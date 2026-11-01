# Kimi INT4 文本切片

`tools/model_slice.py` 支持 Kimi K2-Thinking、K2.5–K2.7 的 compressed-tensors
对称 INT4 group-32 权重；不做反量化。K2-Instruct 的 FP8 提取仍不支持。
默认选取源层 `[0,1,30,31,59,60]`，保留 `first_k_dense_replace=1`、MLA 与全部
384 个路由专家，去掉视觉塔与 MTP。它是断开的六层数值夹具，不能代表完整模型的生成质量。

固定版本的 K2.5 示例（先加 `--plan-only` 检查大小与依赖）：

```bash
python tools/model_slice.py \
  --repo moonshotai/Kimi-K2.5 \
  --revision 4d01dfe0332d63057c186e0b262165819efb6611 \
  --text-only --output /home/zengdeveloper/pllm/model-slices/moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611
```

张量 payload 为 54,735,407,616 bytes（50.98 GiB），另有元数据与 tokenizer。
工具按字节范围读取权重，生成新 safetensors 头、索引和 `slice-manifest.json`；
manifest 记录来源版本、层映射、偏移、dtype 与实际读取字节数。输出的
`language_model.*` 前缀被去掉，配置展开为 `DeepseekV3ForCausalLM`，量化配置与
ignore 规则随之保留/去除文本包装前缀。含具体源层编号的量化规则仍拒绝提取。

Tokenizer 必须使用该版本的 `tokenization_kimi.py`、`tiktoken.model`、
`tokenizer_config.json`，以及 tokenizer 导入的 `tool_declaration_ts.py`。
工具将它们连同 `LICENSE`、chat template 与 generation config 放在切片目录里，
不把 Moonshot 代码复制进 pllm 包，也不下载或执行 Moonshot 模型代码。
K2.5 的 Modified MIT License 允许本地使用，要求保留版权和许可；其额外条款要求
超过指定月活/月收入门槛的商业产品显示 “Kimi K2.5”。完整条款随切片保留。

使用 vLLM 0.32 的 `LLM(model=path, trust_remote_code=True, enforce_eager=True)`，
或 pllm 的 `ModelConfig(path, trust_remote_code=True, cuda_graphs=False)`。
切片的模型配置已移除 `auto_map`，所以只有 tokenizer 需要远程代码许可；模型使用
vLLM 原生 DeepSeek V3 实现。不要省掉该许可而依赖自动 tokenizer 转换：本环境的
Transformers 转换对 `I'm` 得到 `[40,14404]`，Moonshot 实现得到 `[17172]`。

A100 可走 INT4 Marlin MoE，保持其余 BF16 权重。使用 TP=1、短上下文并为约 51 GiB
权重预留内存。pllm 沿用 DeepSeek V3 的只读能力边界：hidden taps、ESamp 与采样器
阶段可用；参数 overlay、因果训练/TLM、层路由与 PositionState 被拒绝。
