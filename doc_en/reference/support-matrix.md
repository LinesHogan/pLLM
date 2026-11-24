# Support Matrix

This page lists what pLLM supports by model family, feature, vLLM version, and parallelism mode. Each restriction includes its reason; for unsupported algorithm families and why, see the [Capability Map](../guides/capability-map.md) and [Design Tradeoffs](../explain/design-tradeoffs.md), and for steps to onboard another model family, see [Using New Model Families](../howto/new-model-families.md).

## vLLM Versions

| Item | Support | Notes |
| --- | --- | --- |
| vLLM versions | 0.27 – 0.34 | Covers `vllm>=0.27.0,<0.35`. `0.27.1` and `0.32.0` run full GPU regression suites in this repository; `0.33`–`0.34` adapt to upstream interface changes (requiring PyTorch 2.14 / CUDA 13 driver) |
| Model runner | vLLM MRv2 | pLLM selects MRv2 automatically (setting `VLLM_USE_V2_MODEL_RUNNER=1` on versions such as 0.27); MRv1 is not supported |
| Tensor parallelism | vLLM 0.32.0 and later | See [Tensor Parallelism](#tensor-parallelism) below |
| Pipeline & data parallelism | Not supported | Must be `1` |
| Online serving | `pllm.make_async_llm` and `pllm serve` | `ModelConfig(multiprocessing="auto")` automatically isolates `EngineCore` in a subprocess for async handles and multi-GPU runs; supports continuous admission, SSE streaming, disconnect cancellation, `pllm_route` session routing, in-flight `import_state()` / `train_batch()` / `feed()`, and no-workload operation. See [Online Serving Guide](../howto/online-serving.md) |

## Models and Features

pLLM admits models by vLLM's **concrete model class**, covering 15 decoder architecture families (16 native model classes; see [Using New Model Families](../howto/new-model-families.md)). Unverified subclasses are rejected at admission.

| Feature | Qwen2 / 2.5, Llama, Qwen3, Mistral, Phi-3, GLM-4, Gemma 2 / 3 | Mixtral, Qwen2-MoE, Qwen3-MoE | Qwen3.5 (Gated DeltaNet hybrid, incl. MoE) | DeepSeek V2 / V3 | gpt-oss |
| --- | --- | --- | --- | --- | --- |
| Reads & exports (`pllm.hidden`, `residual`, `final_hidden`, `layer_logits`), probes | ✓ | ✓ | ✓ | ✓ | ✓ |
| `@pllm.add_logits`, `@pllm.guide_sampling`, request-state training (e.g., ESamp) | ✓ (Gemma 2 requires dense guidance ¹) | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.ActivationPatch` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pllm.model.Adapter`, `@pllm.train_in_forward` | ✓ All projections (`q/k/v/o`, `gate/up/down`) | q/k/v/o ² | Full-attention k/v/o, dense MLP ³ | — ⁵ | — ⁵ |
| `pllm.model.SharedLayers` | ✓ Unquantized dense | — | — | — | — |
| `@pllm.custom_residual` | ✓ | ✓ (except `site="value"`) | ✓ (`site="value"` full-attention layers only) | ✓ (except `site="value"`) | ✓ (except `site="value"`) |
| `@pllm.custom_attention`, static KV layouts | ✓ | ✓ | — ⁴ | ✓ (MLA compressed latents ⁶) | — ⁵ |
| `@pllm.custom_forward`, `pllm.PositionState` | ✓ | ✓ | — ⁴ | — ⁵ | — ⁵ |
| Full-model causal training (`pllm.causal_model`: TLM, TTRL) | ✓ (except Gemma 2 ⁷) | — ⁷ | — ⁷ | — ⁷ | — ⁷ |
| Host-only training (`pllm.host.make_host`) | Model-agnostic | Model-agnostic | Model-agnostic | Model-agnostic | Model-agnostic |

1. Gemma 2's `final_logit_softcapping` requires exact full-vocabulary guidance rather than bounded top-K guidance; GLM-4 and Gemma 2 / 3 four-norm structures are aligned with native branch ordering.
2. Sparse MoE MLP layers route tokens to experts without a single unified dense projection, so routed layers in `Mixtral`, `Qwen2-MoE`, and `Qwen3-MoE` expose attention `q/k/v` (unrouted dense layers in `Qwen2-MoE` and `Qwen3-MoE` also expose MLP deltas); `site="value"` relies on a dense `v_proj` slice.
3. Gated DeltaNet layers have no q/k/v projections; Qwen3.5's full-attention layers interleave per-head queries with output gates, so modifying query alone would alter the gate—hence only full-attention `v` projections and dense MLPs are exposed.
4. Gated DeltaNet recurrent state does not reside in the paged KV cache, so pLLM cannot replay or re-read those layers from KV pages alone.
5. DeepSeek V2/V3 MLA has no standalone V projection and splits MoE by expert; gpt-oss uses sink-enabled sliding-window attention and MXFP4 expert projections. Both expose read-only ops, `ActivationPatch`, and non-`value` `@pllm.custom_residual` rewrites, but not `Adapter` or full-model training.
6. `@pllm.custom_attention` on DeepSeek V2/V3 computes attention directly over compressed latent pages (`kv_lora_rank` + `qk_rope_head_dim`) via matrix absorption (`W_UK_T` / `W_UV`), supporting `keep="all"` only (no static `window` buffer).
7. Full-model training backward passes are implemented for these 7 dense architectures and require a BF16 base model with unquantized embeddings and LM head; Gemma 2 ties shared embeddings without a separate `lm_head`.

Kimi K2.x INT4 text towers can be sliced into a 6-layer DeepSeek V3 test slice for structural testing with the same feature boundaries as DeepSeek V3; see [Using New Model Families](../howto/new-model-families.md).

## CUDA Graphs and Eager

By default, `ModelConfig(cuda_graphs="auto")` uses CUDA graphs whenever possible and falls back to eager execution otherwise, recording the reason in `pllm.explain`'s `forward` line and `llm.diagnostics().eager_reason`; `cuda_graphs=True` raises an error in those cases.

| Pattern | Model Forward |
| --- | --- |
| Reads & exports, sampling effects, request-state training (backward captured alongside AdamW), `@pllm.write` | vLLM CUDA graph |
| `ActivationPatch`, `Adapter` backed by `RequestState` or `InstanceState`, `@pllm.custom_residual`, `@pllm.custom_attention`, token-history lookups | vLLM CUDA graph (vLLM captures the base model ops and overlay reads without Inductor fusion) |
| `@pllm.custom_forward` | Does not use vLLM's full-model graph; pLLM schedules layer segments and captures its own sliced graphs |
| `cuda_graphs=False` | Eager |

## Tensor Parallelism

Requires vLLM 0.32.0 or later with `ModelConfig(tensor_parallel_size=N)` (`multiprocessing="auto"` enables subprocess isolation automatically).

| Supported Multi-GPU | Requires `tensor_parallel_size=1` |
| --- | --- |
| Hidden reads & exports (delivered from rank 0 only), sampling guidance, request-state training (ESamp, output-layer LoRA, etc.), `@pllm.custom_forward`, `ActivationPatch` | `per_head` adapter deltas, custom projection operators (`operator=fn`), `Adapter` with optimizer / side-stream reads-writes or initialized via `FromModule` / `ModelCheckpoint` / `Fed` |
| Dense/low-rank deltas on `q/k/v/o`, `gate/up`, `down` backed by `Zeros`/`Computed`/`Pretrained`/`Imported` (including sharded `@pllm.train_in_forward` updates) | Full-model causal training (`pllm.causal_model`), state feeding (`pllm.Fed`) |
| `@pllm.custom_residual` without `pllm.value(...)` / `pllm.input_tokens` dependencies; `@pllm.custom_attention` directly reading local-rank head KV | `@pllm.custom_residual` fed `pllm.value(...)` / `pllm.input_tokens`, or `@pllm.custom_attention` using `pllm.attention(...)` fact tables / `window` buffers |

Memory reservations reported by `pllm.explain` are per GPU.

## Quantization and Attention Backends

| Item | Support | Notes |
| --- | --- | --- |
| Quantized base models | GPTQ, AWQ, and FP8 checkpoints, admitted under the same family rules | `Adapter`-style edits add a high-precision correction to the quantized op's output, matching vLLM's LoRA behavior on quantized models; full-model training dequantizes transiently during backward (QLoRA-style); methods reading `pllm.operand("weight")` keep an additional dequantized weight copy resident |
| Quantized KV (`kv_cache_dtype`) | Incompatible with `@pllm.custom_attention` | Custom reads directly read KV pages (including MLA latent pages) and require unquantized K/V |
| Attention backend | `@pllm.custom_attention` requires FlashAttention 2 (or the MLA backend for DeepSeek V2/V3) | Other features are backend-agnostic |
| Prefix caching | Supported for read-only workloads, stateless `@pllm.custom_residual`, and `Adapter` / `last_tokens=0` `ActivationPatch` reading only immutable `InstanceState` (non-`Imported`/`Fed`, no writer methods) | Must be disabled when reading `RequestState`, mutable or externally updated `InstanceState`, `last_tokens > 0`, `@pllm.train_in_forward`, or `@pllm.custom_attention`: cached KV depends on request-private or dynamically changing state |
