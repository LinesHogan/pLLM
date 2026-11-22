# Using New Model Families

> **Prerequisites**: Familiarity with basic `pllm.Workload` declarations (inputs, state, and effects); completed [Tutorial 00](../tutorials/00_quickstart.ipynb).
>
> **What this solves**: How to port a working workload from the default Qwen2.5 to other model families (Qwen3, Qwen3-MoE, Qwen3.5, Llama, DeepSeek V2/V3, gpt-oss, quantized checkpoints, or ultra-large model slices), understand admission rules and capability boundaries, and run quick diagnostics without occupying a GPU.

---

## 1. Admission Rules: Exact vLLM Model Class Admission

Once your workload is verified on Qwen2.5 and you want to run it on another open-weight model, the first question is: **how does pLLM decide whether a model is supported?**

pLLM's runtime admits models strictly by vLLM's **exact concrete model class**—**unverified custom subclasses are not admitted**.

- **Why exact class admission**: Variants or third-party subclasses can change the physical layout of attention projections (such as packed QKV splits, shard boundaries, or interleaved gating), sublayer normalization placement (such as the four norms per block in GLM-4 and Gemma 2 / 3), or LM Head tying and logit scaling (such as Gemma 2's tied embeddings and `final_logit_softcapping`). pLLM verifies the residual hook sites, projection output shards, and weight layouts for each native model class individually, so only the verified native vLLM classes listed below are admitted.

Currently 15 architecture families (16 native model classes) are admitted, grouped into five capability tiers:

| Architecture Tier | Admitted Concrete Model Classes | Representative Models |
| :--- | --- | --- |
| **Full-Feature Dense** | `Qwen2ForCausalLM`, `LlamaForCausalLM`, `Qwen3ForCausalLM`, `MistralForCausalLM`, `Phi3ForCausalLM`, `Glm4ForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM` | Qwen2 / Qwen2.5, Llama 2 / 3, Qwen3 (dense), Mistral, Phi-3, GLM-4, Gemma 2 / 3 |
| **Sparse KV-MoE** | `MixtralForCausalLM`, `Qwen2MoeForCausalLM`, `Qwen3MoeForCausalLM` | Mixtral, Qwen2-MoE, Qwen3-MoE |
| **Hybrid Architecture** | `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM` | Qwen3.5 dense and MoE (Gated DeltaNet + full attention) |
| **Latent MLA** | `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM` | DeepSeek V2 / V3 (supports latent `@pllm.custom_attention` and non-`value` sublayer residuals) |
| **Sliding-Window Fused** | `GptOssForCausalLM` | gpt-oss (supports read-only taps, `ActivationPatch`, and non-`value` sublayer residuals) |

Passing an unadmitted architecture or derived subclass is rejected immediately at engine initialization with a clear diagnostic message.

---

## 2. Quick Pre-Flight Check: Validate on CPU with `pllm.explain`

Large model checkpoints routinely exceed tens of gigabytes. Downloading full weights and filling GPU VRAM just to discover an architecture incompatibility wastes time and compute.

Instead, run `pllm.explain(workload, config)` directly on CPU. It reads only the checkpoint's `config.json`, traces shapes on CPU, and runs the exact same admission checks as launching the real GPU engine (`pllm.make_llm`):

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

### Boundaries and Caveats of `explain`

Keep three boundaries in mind when pre-checking a new model with `pllm.explain`:

1. **Projection and layout resolution scope**: From `config.json` alone on CPU, `pllm.explain` resolves projection and attention layouts for all 8 dense decoders, all 3 sparse KV-MoE decoders (`Mixtral` / `Qwen2-MoE` / `Qwen3-MoE`), `Qwen3.5` hybrid models, and `DeepSeek V2/V3` (including MLA latent dimensions).
2. **When admission runs**: As long as the config declares an admitted `architectures` class, `pllm.explain` checks the corresponding capability matrix immediately. Actual weight layouts, quantized kernels (such as gpt-oss MXFP4 experts or custom quantization configs), and available KV Cache capacity are verified by the engine when calling `pllm.make_llm` / `pllm.make_async_llm`.
3. **True VRAM footprint**: `pllm.explain` reports the workload's own state and workspace memory needs; it excludes model weights, KV Cache, and vLLM's internal allocations. Whether the full model fits in GPU memory is determined at engine startup.

For details on interpreting cost reports, see [Reading Costs: pllm.explain](../guides/explain.md).

---

## 3. Capability Boundaries by Model Family (and Why)

Architectural differences in attention, KV storage, and MLP structure determine what each model family supports in pLLM. For the complete feature table, see the [Support Matrix](../reference/support-matrix.md); this section summarizes each family's boundaries and the design reasons behind them.

### Dense Models (Qwen2 / Qwen2.5, Llama, Qwen3, Mistral, Phi-3, GLM-4, Gemma 2 / 3)

- **Supported**: All core features—hidden/residual taps and exports, probes, `pllm.model.ActivationPatch`, DoLa / logit lens, `@pllm.guide_sampling`, `@pllm.add_logits`, `pllm.model.Adapter` on all projections, decode-time online training (`@pllm.train`), looped depth and custom forwards (`@pllm.custom_forward`), `@pllm.custom_residual`, `@pllm.custom_attention`, and whole-model causal training (`pllm.causal_model`, except Gemma 2).
- **Details and edge cases**: These models use standard MHA or GQA with all K/V stored in paged KV blocks, dense gated/feed-forward MLPs, and clean projection slices. GLM-4 and Gemma 2/3 use four-norm sublayer layouts (already aligned in residual sites and causal-training feed-forward branches). Gemma 2 ties word embeddings to the output head and applies `final_logit_softcapping`, so it disables whole-model `pllm.causal_model` backward passes and uses exact full-vocabulary guided sampling.

### Sparse KV-MoE (Mixtral, Qwen2-MoE, Qwen3-MoE)

- **Supported**: All read-only taps and guided sampling; `@pllm.custom_forward`, `@pllm.custom_attention`, and `@pllm.custom_residual` for all sites except `site="value"`; `pllm.model.Adapter` on attention `q_proj`, `k_proj`, and `v_proj` across all layers, plus non-routed dense MLP projections in `Qwen2-MoE` / `Qwen3-MoE`.
- **Restrictions and reasons**:
  - **No MLP `Adapter` on MoE expert layers**: Routed layers dispatch tokens dynamically across multiple expert networks and have no single dense MLP projection matrix, so uniform per-layer MLP overlays are rejected (with an error suggesting attention query/key/value projections instead).
  - **No `site="value"` residuals or whole-model causal training**: `site="value"` relies on dense standalone V-projection slicing conventions, and causal backward passes do not yet cover routed expert gradients.

### Qwen3.5 (Gated DeltaNet Hybrid Attention, including MoE)

- **Supported**: All read-only taps and guided sampling; `@pllm.custom_residual` (all sites on full-attention layers, and non-`site="value"` on all layers); `Adapter` on `v_proj` of full-attention layers and dense MLP projections.
- **Restrictions and reasons**:
  - **No `@pllm.custom_forward` or `@pllm.custom_attention`**: Gated DeltaNet layers maintain recurrent hidden state updated token-by-token outside the paged KV cache, so layers cannot be replayed or read out solely from KV pages.
  - **DeltaNet layers lack standard Q/K/V projections**, and full-attention layers interleave query rows with output gates (`attn_output_gate`), so only `v_proj` and dense MLPs are exposed for `Adapter`.
  - **No whole-model causal training**: Backpropagation through hybrid recurrent states is not yet supported.

### DeepSeek V2 / V3 (MLA + MoE) and gpt-oss

- **Supported**: Read-only taps, `@pllm.add_logits` / `@pllm.guide_sampling`, ESamp, `ActivationPatch`, and non-`site="value"` sublayer `@pllm.custom_residual` (`stream` / `mlp` / `attention`); additionally, **DeepSeek V2 / V3 supports compressed-latent `@pllm.custom_attention`**.
- **Technical notes and restrictions**:
  - **Latent `@pllm.custom_attention` on DeepSeek V2 / V3**: MLA stores compressed latent `c_kv` plus decoupled RoPE keys in the paged cache and has no separate `v_proj`. pLLM runs custom attention directly over compressed latent pages using the absorbed projection matrices `W_UK_T` / `W_UV` to compute scores and weighted sums without materializing full KV caches (requires `keep="all"`; static `window` buffers are not supported).
  - **No `Adapter`, whole-model causal training, or `@pllm.custom_forward`**: DeepSeek splits expert weights across routed experts, and gpt-oss uses sink sliding-window attention with MXFP4 expert projections, so parameter overlays and whole-model training are not exposed.

For a systematic overview of algorithm patterns vs. model capabilities, see the [Capability Map](../guides/capability-map.md).

---

## 4. Quantized Base Models

Many deployed checkpoints use GPTQ, AWQ, or FP8 quantization. In pLLM, **quantized base models are first-class citizens**:

- **Identical admission rules**: A quantized checkpoint is admitted under the exact same rules as its base model family. For instance, an AWQ Qwen2.5 checkpoint supports the full Qwen2 capability set.
- **Weight overlay mechanism (LoRA / `Adapter`)**: When applying LoRA or parameter overlays on a quantized base, pLLM uses the same mechanism as vLLM's native quantized LoRA—accumulating a high-precision (BF16/FP16) low-rank correction directly onto the output of the quantized kernel (Marlin, Scaled MM, etc.):

  $$\text{output} = \text{quant\_linear}(x) + \Delta W(x)$$

  This **requires no dequantization of base weights and consumes no extra dense base-weight VRAM**.
- **Whole-model training mechanism (QLoRA-style backward)**: When running whole-model causal training (e.g., TLM) on a quantized model, pLLM uses a QLoRA-style workflow: the forward pass runs efficient quantized kernels, and the backward pass temporarily dequantizes weights using their quantization scales and zero-points to compute gradients—preserving VRAM savings while enabling differentiation.

```python
# check-docs: skip: quantized model execution requires CUDA
from vllm import SamplingParams

quant_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-7B-Instruct-AWQ",  # AWQ checkpoint, admitted under the Qwen2 family
    max_num_seqs=8,
)

with pllm.make_llm(quant_config, workload=ObserveResidual()) as llm:
    outputs = llm.generate(["Testing quantized base model"], SamplingParams(max_tokens=8))
```

---

## 5. Structural Verification on Ultra-Large Models: `model_slice` and the Kimi INT4 Text Slice

When researching or validating algorithms, you often need to confirm tensor paths and structural compatibility on 100B+ parameter models (such as DeepSeek V3 or Kimi K2.5's MLA layout with 384 routed experts). However, single GPUs or standard dev machines cannot fit hundreds of gigabytes of weights.

pLLM provides `tools/model_slice.py`, which uses HTTP Range requests to fetch only selected decoder layers and safetensors index metadata from a remote Hugging Face repo (or local directory), creating a compact local slice fixture without downloading the full checkpoint.

### Kimi INT4 Text Slice Recipe

`tools/model_slice.py` supports compressed-tensors symmetric INT4 group-32 weights for Kimi K2-Thinking and K2.5–K2.7 without dequantization (note: FP8 extraction for K2-Instruct is not supported).

By default, it extracts source layers `[0, 1, 30, 31, 59, 60]` (first two, middle two, last two—6 layers total), preserving `first_k_dense_replace=1`, the MLA structure, and all 384 routed experts while dropping the vision tower and MTP. **This is a disconnected 6-layer numerical fixture for validating execution graphs and tensor shapes—it does not reflect the full model's generation quality.**

Slice command (run with `--plan-only` first to inspect size and dependencies):

```bash
python tools/model_slice.py \
  --repo moonshotai/Kimi-K2.5 \
  --revision 4d01dfe0332d63057c186e0b262165819efb6611 \
  --text-only --output model-slices/moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611
```

The extracted tensor payload is ~50.98 GiB (54,735,407,616 bytes), plus metadata and tokenizer files. The output config is unwrapped to `DeepseekV3ForCausalLM` with `language_model.*` prefixes and vision/MTP wrappers stripped.

### Critical Tokenizer Convention

The tokenizer must use the pinned `tokenization_kimi.py`, `tiktoken.model`, `tokenizer_config.json`, and `tool_declaration_ts.py` imported by the tokenizer. The slice tool automatically preserves these alongside `LICENSE`, the chat template, and generation config in the output directory.

Because the slice config strips `auto_map`, the model itself runs directly on vLLM's native DeepSeek V3 implementation; only the tokenizer requires remote code permission. You **must pass `trust_remote_code=True`** explicitly—do not omit it and fall back to Transformers' automatic tokenizer conversion, which in current environments splits `"I'm"` into two tokens `[40, 14404]` instead of Moonshot's single token `[17172]`.

### License Note

K2.5 is released under the Modified MIT License, which permits local use while requiring retention of copyright and license notices and includes an attribution clause ("Kimi K2.5") for commercial products exceeding specified MAU/revenue thresholds. Full terms are preserved in the slice directory.

### Loading the Slice in pLLM

A single A100 GPU (~51 GiB VRAM) can run this slice using INT4 Marlin MoE kernels:

```python
# check-docs: skip: Kimi slice execution requires CUDA and local slice checkpoint
from vllm import SamplingParams

slice_config = pllm.ModelConfig(
    "model-slices/moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611",
    trust_remote_code=True,
    cuda_graphs=False,
)

with pllm.make_llm(slice_config, workload=ObserveResidual()) as llm:
    outputs = llm.generate(["Testing Kimi slice"], SamplingParams(max_tokens=8))
```

The slice inherits DeepSeek V3's capability boundaries: hidden taps, ESamp, guided sampling, `ActivationPatch`, non-`value` `@pllm.custom_residual`, and latent `@pllm.custom_attention` are supported, while parameter overlays, whole-model causal training, and `PositionState` remain rejected.

---

## 6. Common Admission Errors and Fixes

For exact rejection messages, causes, and fixes related to this page, see the [Engine settings, models and parallelism](../reference/errors.md#engine-settings-models-and-parallelism) and [Model sites and custom forwards](../reference/errors.md#model-sites-and-custom-forwards) sections of the Error Catalog.

---

## 7. Self-Check

1. On Qwen3-MoE, why does attaching LoRA to attention `v_proj` work while attaching LoRA to MLP `down_proj` is rejected?
<details>
<summary>Answer</summary>

Qwen3-MoE keeps the same attention layout as dense Qwen3, so attention <code>v_proj</code> is a single uniform dense matrix where a clean low-rank delta applies. Its MLP layers, however, are sparse Mixture-of-Experts layers that route tokens dynamically to many separate expert networks, with no single dense projection matrix to overlay.
</details>

2. When loading the Kimi K2.5 slice, why is `trust_remote_code=True` mandatory even though the model runs on vLLM's native DeepSeek V3 kernels?
<details>
<summary>Answer</summary>

The slice tool strips <code>auto_map</code> from the model config so the backbone runs directly on vLLM's built-in <code>DeepseekV3ForCausalLM</code> implementation, requiring no remote model code. However, Kimi's tokenization depends on its custom <code>tokenization_kimi.py</code>; omitting <code>trust_remote_code=True</code> causes Transformers to fall back to a generic tokenizer conversion that splits tokens differently (e.g., splitting <code>"I'm"</code> into two tokens).
</details>

---

## 8. Where to Next

- Full table of model families vs. supported features → [Support Matrix](../reference/support-matrix.md)
- Algorithm design trade-offs and capability coverage → [Capability Map](../guides/capability-map.md)
- Estimate memory and execution plans on CPU → [Reading Costs: pllm.explain](../guides/explain.md)
- Diagnose model loading and runtime issues → [Troubleshooting](troubleshooting.md)
