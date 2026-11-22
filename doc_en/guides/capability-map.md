# Capability Map: Which Inference-Time Algorithms Can You Write in pLLM Today?

Inference-time LLM algorithms are growing increasingly complex: lookup memory (such as DeepSeek's Engram, which indexes a large table by n-gram), sparse-attention indexers and cross-layer reuse of their selections (such as DeepSeek's DSA and IndexCache), compressed or shared KV caches, looping and adaptive depth, and inference-time training (such as TTT and In-Place TTT).
This page lists each algorithm family one by one: whether you can implement it in pLLM today, which APIs to use, and which tutorial covers it; for algorithms that are not supported, it states the reason directly. All measured numbers below come from the tutorials and examples on an RTX 4090, with exact setups recorded in each notebook.

The **Status** column in the tables uses three primary labels:

- **Supported**: Implementable on supported models using existing public APIs, with a runnable example in the tutorials (or noted as having no example).
- **Partial**: Part of the algorithm family can be implemented and part cannot; the table specifies which part.
- **Unsupported / Out of scope**: Cannot be implemented today, or falls outside pLLM's scope (for example, pretraining a new architecture); the table explains why.

## Reading or Computing at a Forward Site

If you want to read internal activations, modify the residual stream, or insert custom tensor math and custom operators during the model's forward pass, start with this section.

| Algorithm (Examples) | How to Write Today / Why Not | Status | Tutorial |
|---|---|---|---|
| Activation steering (ActAdd, CAA), ReFT / LoReFT | `pllm.model.ActivationPatch`; built-in `pllm.workloads.ReFTWorkload`. Activation patches run inside the model's CUDA graph | Supported | [00](../tutorials/00_quickstart.ipynb) |
| Linear probes, SAE monitoring | `pllm.hidden` / `pllm.residual` + `@pllm.export`; the model stays in CUDA graph | Supported | [01](../tutorials/01_observe_probe.ipynb) |
| Logit lens, DoLa | `pllm.layer_logits`; built-in `DoLaWorkload` | Supported | [06](../examples/06_reading_inside.ipynb) |
| Value residual, hyper-connections (inference architecture for hyper-connections / mHC), LAuReL | `@pllm.custom_residual(site="value" / streams=n / site="both")`, running inside the CUDA graph; weights for these structures require checkpoints trained with them | Supported | [07](../examples/07_custom_residual.ipynb) |
| Steps requiring custom kernels (hashing, scoring, scans) | Wrap with `torch.library.custom_op` and call directly inside workload methods; `explain` reports FLOPs/bytes and `check_on_gpu` compares the kernel against a CPU reference | Supported | [09](../examples/09_custom_kernels.ipynb) |

## Attention and KV

When contexts grow long or you need visibility into attention distributions, common needs include tracking attention statistics, reading only a subset of KV blocks, or reducing KV memory footprint via sliding windows and cross-layer sharing.

| Algorithm (Examples) | How to Write Today / Why Not | Status | Tutorial |
|---|---|---|---|
| Attention statistics / monitoring (fraction of attention on recent positions) | `@pllm.custom_attention` returning `aux`, exported via `pllm.attention(layer, "aux")`; outputs are bitwise identical to native vLLM | Supported | [10](../examples/10_custom_attention.ipynb) |
| Lookback Lens style (fraction of attention on the prompt) | Requires knowing each row's prompt length; `PagedKV` exposes this only under `sink=True` layout (`kv.sink_lens`) | Partial | — |
| Training-free sparse reads (Quest page selection, fixed patterns like retrieval heads) | `kv.attend(q, topk=...)`, with per-layer and per-head budgets; saves memory reads but not KV memory capacity; PyTorch-based scoring has low throughput (13% of native on 7B in Example 10), so speeding it up requires a custom kernel | Supported | [10](../examples/10_custom_attention.ipynb), [09](../examples/09_custom_kernels.ipynb) |
| Score-based KV eviction (H2O, SnapKV, PyramidKV) | Read-only selection is supported (selecting positions by score with per-layer/per-head budgets); freeing physical KV memory by score is not supported—see [Design Tradeoffs](../explain/design-tradeoffs.md) | Partial | [10](../examples/10_custom_attention.ipynb) |
| Sliding window, alternating local/global layers | `keep="window", window=N`, where old blocks are freed by vLLM's sliding-window mechanism; in Example 10, a 256-token window on even layers reduces the 7B per-request KV ceiling at 2048 tokens from 112.0 MiB to 70.4 MiB | Supported | [10](../examples/10_custom_attention.ipynb) |
| Keep full prompt plus recent window (R-SWA) | `sink=True`: retains the entire prompt plus the most recent `N` generated tokens; worst-case ceiling is unchanged, while memory is saved during long generations | Supported | [10](../examples/10_custom_attention.ipynb) |
| StreamingLLM (first few sink tokens + recent window) | The read pattern can be written with `topk` to attend only to the initial positions and recent window; however, memory can only be freed at R-SWA granularity (full prompt + window), not down to just the initial few tokens | Partial | — |
| Cross-layer KV sharing (CLA, YOCO style, sourcing from lower layers) | `keep="none", source=k`: this layer allocates no KV cache and reads layer `k` below; Example 10 halves the KV ceiling when sharing on odd layers; models must be trained for this shared layout | Supported (execution) | [10](../examples/10_custom_attention.ipynb) |
| KV sharing sourcing from deeper layers (LCKV style) | The source layer has not been computed yet for the current token; the error message suggests using `kv="peek"` with `@pllm.custom_forward` instead | Partial | [04](../tutorials/04_custom_forward.ipynb) |
| Compressed distant memory (Activation Beacon, DeepSeek V4-style compressed KV, LastKV) | Not currently supported; see [Design Tradeoffs](../explain/design-tradeoffs.md) | Unsupported | — |
| Trained sparse attention (DSA, NSA, MoBA) | Part of the model architecture itself; vLLM supports DSA natively, but requires sm90+ GPUs (H100 class) | Out of scope | — |
| Cross-layer reuse of sparse attention selections (IndexCache, DeepSeek V4.1 indexer source layer) | Requires a DSA model, and vLLM's DSA path requires sm90+; `pllm.attention(layer, "topk")` is reserved for then and currently errors stating it is not yet supported | Unsupported (requires hardware) | — |
| KV quantization (KIVI style) | Managed by vLLM's `kv_cache_dtype`; `@pllm.custom_attention` requires unquantized KV | Out of scope | — |

## Lookup Memory

If you want the model to query external parameter tables during inference (for example, looking up an n-gram table over recent tokens and adding the result into the residual stream or logits), you can combine state declarations with residual or logit interfaces.

| Algorithm (Examples) | How to Write Today / Why Not | Status | Tutorial |
|---|---|---|---|
| Engram-style n-gram memory (attached to an existing model) | `pllm.input_tokens(history=n)` reads token history, `InstanceState(init=pllm.Pretrained(...))` holds the table with your choice of `placement="device"/"host"`, and `@pllm.custom_residual` adds retrieved vectors into the residual stream; the table must be trained elsewhere—pLLM does not train or update the table | Supported | [11](../examples/11_lookup_memory.ipynb) |
| N-gram language model prior (adding context-based scores to logits) | Same table lookup connected to `@pllm.add_logits` | Supported | [11](../examples/11_lookup_memory.ipynb) |
| Memory layers / product-key memory | Keys computed from hidden states with top-k selection and rows gathered from a large value table can be written as tensor math inside `custom_residual` (large tables can live in host memory); no example in this repository and unverified | Should work (unverified) | — |
| Built-in model Engram (DeepSeek V4.1, etc.) | Implemented by vLLM's model code; running these models in pLLM falls under model admission and requires sm90+ | Out of scope | — |

## Depth and Looping

When an algorithm changes layer execution order—such as looping a block of layers multiple times or dynamically skipping later layers based on the current token state—you can express it with a custom forward schedule and per-row activation masks.

| Algorithm (Examples) | How to Write Today / Why Not | Status | Tutorial |
|---|---|---|---|
| Looped depth (Ouro, Huginn style) | Loop `layers(...)` inside `@pllm.custom_forward`; every pass writes KV (Ouro's "last-pass reuse"); keeping separate KV per pass during prompt prefill is not currently expressible | Supported | [04](../tutorials/04_custom_forward.ipynb) |
| Early exit, per-token layer skipping (CALM saturation criteria, causal-routing MoD, token-choice MoR) | `layers(..., active=mask)`; pLLM populates KV for skipped rows, decisions stay on GPU with zero blocking sync per step; decode tokens only | Supported | [12](../examples/12_adaptive_depth.ipynb) |
| Expert-choice MoR (top-k across the batch), original MoD (skipped tokens omit KV) | The former makes a token's depth depend on other requests in the same batch, which pLLM intentionally disallows; the latter requires holes in the KV cache, sharing the same constraint as score-based eviction | Unsupported | — |
| Early exit for prompt tokens | `active=` currently supports only `tokens="decode"` | Unsupported | — |
| Continuous thought / token-level recurrence (Coconut style) | `custom_forward` + `pllm.PositionState` carries state from one position to the next; meaningful results require a model trained for it | Partial | [04](../tutorials/04_custom_forward.ipynb) |

## State and Inference-Time Learning

If an algorithm needs mutable state over the lifetime of each request or the whole engine instance—including gradient updates, fast-weight updates, or host-side feedback loops during generation—use the interfaces below.

| Algorithm (Examples) | How to Write Today / Why Not | Status | Tutorial |
|---|---|---|---|
| Per-request LoRA (Text-to-LoRA hypernetworks), decode-time online training (ESamp), test-time learning (TLM) | `pllm.RequestState` / `pllm.InstanceState` + `@pllm.train`; backward passes and optimizer updates run inside the CUDA graph | Supported | [02](../tutorials/02_request_adapters.ipynb), [03](../tutorials/03_decode_training.ipynb) |
| TTT, Titans, fast-weight memory, In-Place TTT | `@pllm.train_in_forward`, `@pllm.custom_residual` with `state=` | Supported | [07](../examples/07_custom_residual.ipynb), [08](../examples/08_inplace_ttt.ipynb) |
| Linear attention / Gated DeltaNet hybrid models | Qwen3.5 is admitted (see the [Support Matrix](../reference/support-matrix.md) for feature scope); adding a chunked recurrent sublayer onto an existing model uses `custom_residual` with `state=` and `chunk=` | Partial (model-dependent) | [07](../examples/07_custom_residual.ipynb) |
| Test-time reinforcement learning (TTRL) | Host loop: generation, majority voting, and training inside the same engine | Supported | [05](../tutorials/05_host_feedback.ipynb) |

## Decoding Control, Architecture & Training, and Agents

This final section covers boundaries around decoding-loop control, pretraining architectural changes, and outer agent-driven algorithm search.

| Algorithm (Examples) | How to Write Today / Why Not | Status | Tutorial |
|---|---|---|---|
| MTP, speculative decoding, rollback-capable decoding | Requires controlling the decoding loop itself, which belongs to a separate roadmap effort | Unsupported | — |
| Pretraining from scratch, modifying MoE routing, brand-new mixers | Requires new model architectures and pretraining | Out of scope | — |
| Agent-driven algorithm search (AlphaEvolve-style loops) | pLLM provides `explain`, `check`, and contracts; the search loop itself is demonstrated in `examples/algorithm_search` (score differences in the demo run are within noise) | Example | [13](../examples/13_algorithm_search.ipynb) |

## Built-in Workloads

Implementations in `pllm.workloads` go through the exact same compilation path as user-authored workloads. You can run them directly or use them as authoring references:

| Workload | Algorithm | Primitives Used |
|---|---|---|
| `ESampWorkload` | ESamp: per-request online shallow-to-deep predictor guiding sampling | `RequestState` + `@pllm.train` + `@pllm.guide_sampling` |
| `OverRIDEWorkload` | OverRIDE: instance-shared head updated conditionally on actual sampling feedback | `InstanceState` + `@pllm.add_logits` + `@pllm.train` |
| `OutputLoRAWorkload` | Output-layer LoRA with external weighted cross-entropy batches and multi-step SGD | `pllm.host.WeightedCEBatch` + `steps=` / `reset=` |
| `ReFTWorkload` | ReFT / LoReFT activation intervention | `pllm.model.ActivationPatch` + `pllm.Imported()` |
| `DoLaWorkload` | DoLa contrastive decoding | `pllm.layer_logits` + `@pllm.add_logits` |
| `TLMWorkload` | Test-Time Learning (TLM), all-layer q/v LoRA | `pllm.host.CausalTrainBatch` + `pllm.causal_model` + `Adapter` |
| `InPlaceTTTWorkload`, `PrefillAdaptersWorkload` | In-Place TTT fast weights, prompt-trained adapters | `@pllm.train_in_forward` |
| `PromptFactorsWorkload`, `PromptMixtureWorkload`, `DecodeFactorsWorkload` | Hypernetwork-written LoRA, LoRA mixtures, decode-time trained factors | One `Adapter` reading state written by different producers |
| `RecirculationWorkload`, `JacobiRecirculationWorkload`, `StepLoopWorkload` | Cross-layer recirculation, Jacobi-style prompt iteration, carrying hidden state from one position to the next | `@pllm.custom_forward`, `pllm.PositionState` |
| `RewardHeadWorkload`, `RewardHeadFeedback`, `majority_vote` | Majority-vote reward head | `pllm.host.make_host` + `llm.generate_feedback` |
| `FTTTWorkload` | Host-feedback-driven low-rank edits | `pllm.host.WeightedRowBatch` + `HostTrainer` |
| `DummyWorkload` | Minimal observation workload | `pllm.hidden` + `@pllm.export` |

## Why Some Algorithms Cannot Be Written

For why score-based KV memory eviction, compressed distant memory, and DSA top-k reuse are unsupported today—and what alternatives exist—see [Design Tradeoffs](../explain/design-tradeoffs.md).

## Shared Constraints

When using the capabilities above, keep these shared boundaries and design principles in mind:

- Prefix caching must be disabled (`enable_prefix_caching=False`) only when reading `RequestState`, mutable or externally fed `InstanceState`, `@pllm.train_in_forward`, `ActivationPatch` with `last_tokens > 0`, or `@pllm.custom_attention`; stateless `@pllm.custom_residual` and static rewrites reading immutable `InstanceState` can keep it enabled.
- `@pllm.custom_attention` requires unquantized KV (on dense and KV-MoE models it uses FlashAttention 2; on DeepSeek V2/V3 MLA it absorbs `W_UK_T` / `W_UV` directly to read the `kv_lora_rank` + `qk_rope_head_dim` latent cache and supports only `keep="all"`); reading `pllm.attention(...)` facts requires `tensor_parallel_size=1`.
- Static KV layouts cannot be combined with `PositionState` or `@pllm.custom_forward`.
- Tensor parallelism is available on vLLM 0.32.0 and above; see the [Support Matrix](../reference/support-matrix.md) for per-feature multi-GPU coverage.
- Design principle: expensive choices are declared explicitly by you rather than guessed by pLLM—whether a large table sits in GPU memory or host memory, how KV is stored, and how many steps later training updates become visible (`publish_lag`) are all stated in your declarations, and `pllm.explain` itemizes the cost. Defaults only save you from small decisions.
- For which resource each capability saves (VRAM capacity, memory bandwidth, PCIe, inter-GPU communication, timeline bubbles, SMs, host CPU), measured results, and limits, see [Performance and Memory](../explain/performance.md).
