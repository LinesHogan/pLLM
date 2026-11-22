# Design Tradeoffs: Why Some Algorithms Cannot Be Written Today

In a standalone PyTorch script or Hugging Face `generate` loop, a researcher can prototype a new idea in minutes with a few lines of `register_forward_hook`, an inline conditional branch, or `.item()`. Move that same idea into a high-throughput serving engine built on **continuous batching** and **CUDA Graph replay**—such as vLLM—and you quickly run into hard physical and architectural constraints.

pLLM is not a black-box library of pre-packaged algorithms. It is a **programmable stateful substrate** bridging frontier inference-time modeling algorithms and modern serving engines. The substrate handles continuous-batch packing, static CUDA Graph capture, versioned snapshot publishing, and memory lifetime planning so that algorithms compatible with high-throughput execution run at near-native speed.

When an algorithm's assumptions fundamentally collide with those physical mechanisms (fixed tensor shapes, paged block tables, asynchronous stream execution, and strict cross-request isolation), pLLM chooses **not to silently compromise or degrade**; instead, it exposes the system boundary and mathematical reason clearly. This page documents 11 intentional design tradeoffs and limitations, each organized as **Problem to Solve $\to$ pLLM's Choice $\to$ Why $\to$ Alternatives**.

---

## 1. Score-Based KV Eviction & Contiguous KV Caches (H2O, SnapKV)

### Problem to Solve

In long-context generation, physical KV cache memory grows linearly with sequence length. Score-based dynamic eviction methods—such as H2O, SnapKV, and PyramidKV—track cumulative attention scores across historical tokens during autoregressive generation and physically free KV entries for low-scoring "unimportant" tokens, keeping only initial sink tokens and key heavy hitters to slash VRAM usage.

### pLLM's Choice

pLLM does not support dynamically freeing physical KV cache memory based on attention scores.

### Why

vLLM's underlying memory model (PagedAttention) rests on one core invariant: **each request's KV cache is a logically contiguous sequence starting at position 0**.

1. **Scalar length counter & slot mapping**: The engine tracks a single integer per request recording how many tokens have been computed. Which physical page and slot each layer writes to for a given token—as well as its absolute Rotary Position Embedding (RoPE) position—is derived directly from this monotonically increasing counter. Block tables only append new pages at the end; there is no mechanism to punch out slots in the middle.
2. **Holes break paged indexing**: Evicting scattered mid-sequence tokens by attention score creates "holes" in the logical sequence. Supporting holes would require rewriting PagedAttention's block-lookup and stride arithmetic from scratch, and because eviction decisions depend on runtime attention scores on the current step, slot layouts could no longer be planned ahead of time.
3. **Breaks CUDA Graph capture**: Reclaiming mid-sequence blocks and compacting block tables on every decode step requires blocking GPU-to-CPU metadata syncs or invalidates the static tensor pointers and memory layouts that CUDA Graphs depend on, destroying graph-mode throughput.

### Alternatives

- **Sparse reads (saves compute and memory bandwidth, not VRAM capacity)**: Inside [`@pllm.custom_attention`](../guides/forward-structure.md), call `kv.attend(q, topk=...)` or write custom scoring logic (such as Quest). Within each layer and head, only the top-$K$ highest-scoring positions participate in attention while the rest are skipped. This cuts attention FLOPs and memory traffic, runs entirely on GPU, and stays inside CUDA Graphs; see [10 Custom Attention](../examples/10_custom_attention.ipynb).
- **Rule-driven regular physical reclamation (actually saves VRAM)**: To lower per-request peak KV memory, use a static KV layout aligned with block-table boundaries: sliding window (`keep="window", window=N`), prompt-sink plus sliding window (`sink=True`, i.e., R-SWA), or cross-layer sharing where odd layers reuse even layers' KV (`keep="none", source=k`). These reclaim memory at clean block boundaries without punching holes inside blocks.

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing


@dataclass(frozen=True)
class LocalGlobalAttention(pllm.Workload):
    """Even layers keep only the latest 256 tokens of KV (sliding window); odd layers keep full context."""

    window_size: int = 256

    def window(self) -> int:
        return self.window_size

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=window)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q, window=self.window_size)[0]


CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=4)
report = pllm.explain(LocalGlobalAttention(), CONFIG)
```

---

## 2. Compressed Remote Memory & Page Padding Waste (Activation Beacon)

### Problem to Solve

Activation Beacon and related KV compression methods condense distant history (hundreds or thousands of past tokens) into a compact set of summary tokens or compressed entries to balance long-context recall against bounded VRAM.

### pLLM's Choice

pLLM does not currently integrate compressed remote memory into its standard runtime pipeline.

### Why

Deeply integrating compressed remote memory into a continuous-batching engine faces three fundamental physical and algorithmic obstacles:

1. **Page padding waste**: vLLM allocates KV memory in fixed-size pages (blocks of typically 16 or 32 tokens). Storing heavily compressed entries in paged blocks (e.g., compressing 16 tokens down to 1 entry) still pads out whole pages, erasing most of the expected memory savings while introducing severe internal fragmentation.
2. **Chunk-dependent state non-determinism**: When compressed summaries depend on deep hidden states and `enable_chunked_prefill` is active, whether a given position's compressed entry is ready on the current step depends on how the scheduler happened to slice the prompt under current concurrency pressure. Varying chunk boundaries cause compressed features to drift across runs for the exact same input, violating inference determinism.
3. **Two-pass attention overhead**: Running one attention kernel over the full-resolution local window and a second pass over compressed remote entries, then merging them in Python/Triton, incurs kernel-launch and memory-reordering overhead that far exceeds a single native dense read (in a single-request single-layer microbenchmark, split two-pass read + merge took ~30.31 µs versus 11.88 µs for a native full read).

### Alternatives

- **Fixed-layout reduction**: Use sliding windows (`keep="window"`) or cross-layer sharing (`keep="none", source=k`) to reclaim whole physical KV pages with zero two-pass scheduling overhead.
- **Lookup-prior injection**: For long-range background memory or cross-document priors, use token-history lookup memory (such as the Engram architecture in [11 Lookup Memory](../examples/11_lookup_memory.ipynb)). Store large pretrained tables in host DRAM via `pllm.Pretrained(..., placement="host")`, prefetch them asynchronously over PCIe during forward, and inject them into the residual stream without perturbing the KV cache pipeline.

---

## 3. DSA Top-K Index Reuse & `sm90` Hardware Requirements

### Problem to Solve

In DeepSeek Sparse Attention (DSA, e.g., DeepSeek V3/V4), a lightweight indexer scores query–key affinity ahead of attention and selects the top-$k$ blocks or tokens (`IndexCache`). Researchers often want to read that indexer's selection and reuse it across subsequent layers without recomputing scores.

### pLLM's Choice

pLLM does not currently support reading and reusing native DSA top-$k$ index selections across layers.

### Why

DSA is baked into the model architecture itself, so readable indexer selections exist only on models trained with DSA (such as DeepSeek V3.2 / V4). vLLM's attention kernels for those models require `sm90` or newer GPUs (e.g., H100); on `sm80` (A100) and `sm89` (RTX 4090), those models cannot run at all, leaving no native top-$k$ selection to reuse. Today `pllm.attention(layer, ...)` exposes only `"aux"` and `"lse"`.

### Alternatives

- On `sm80`/`sm89` hardware, implement custom attention filtering inside `@pllm.custom_attention` using PyTorch or Triton ops—combining [09 Custom Kernels](../examples/09_custom_kernels.ipynb) with explicit selection via `kv.attend(q, topk=...)` from [10 Custom Attention](../examples/10_custom_attention.ipynb).
- To reduce compute across layers, use cross-layer KV sharing (`keep="none", source=k`).

---

## 4. Batch-Competitive Routing (MoR Expert-Choice) & Vanilla MoD

### Problem to Solve

Mixture-of-Depths (MoD) and Mixture-of-Recursions (MoR) allocate varying compute depth per token:
1. **MoR expert-choice routing**: Each layer ranks tokens across the entire batch and lets only the top-$k$ tokens execute that layer;
2. **Vanilla MoD**: Skipped tokens neither enter the layer's computation nor write any KV entries for that layer.

### pLLM's Choice

pLLM explicitly rejects batch-competitive expert-choice routing and vanilla MoD's no-KV-write skipping. Instead, pLLM supports per-row independent **token-choice adaptive depth** and **early exit**.

### Why

1. **Request isolation must never be broken**: In continuous batching, a single 2D tensor packs completely unrelated requests from different clients. Competing for top-$k$ slots across the batch makes whether a token runs a layer depend on which other requests happen to be co-batched at that millisecond. As concurrency or queue order fluctuates, the same prompt generates different text, breaking idempotency and determinism (`pllm.testing.check` catches this immediately with `ContractError: method ... mixes rows`).
2. **KV sequences cannot have holes**: In vanilla MoD, if token $t$ skips layer $L$ without writing KV, layer $L$'s KV cache has a missing slot at position $t$. When token $t+1$ attends at layer $L$, causal masking and paged block tables expect a contiguous history; missing positions break causal alignment just like score-based KV eviction.

### Alternatives

- **Token-choice adaptive routing**: Each token decides independently whether to run a layer from its own hidden state or local router (e.g., a sigmoid threshold), without competing against other requests.
- **GPU-side KV backfill & in-graph dispatch**: Pass a boolean mask `active=mask` (shape `[rows, 1]`) inside `@pllm.custom_forward`. Inactive rows skip the layer forward while pLLM automatically computes approximate K/V on GPU using the skipped layer's norm and projections so subsequent tokens always see a contiguous KV cache. Branching executes entirely on GPU via CUDA Graph conditional nodes with zero GPU-to-CPU syncs. See [12 Adaptive Depth](../examples/12_adaptive_depth.ipynb).

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing


@dataclass(frozen=True)
class SkipMiddle(pllm.Workload):
    """Token-choice adaptive depth: decide layer skipping per row during decode only."""

    threshold: float = 2.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h12 = layers(0, 12)[11]
        h16 = layers(12, 16, hidden=h12)[15]
        moved = (
            torch.nn.functional.cosine_similarity(h16.float(), h12.float(), dim=-1).unsqueeze(-1)
            <= self.threshold
        )
        h20 = layers(16, 20, hidden=h16, active=moved)[19]
        return layers(20, hidden=h20)[-1]


check = pllm.testing.check(SkipMiddle(), CONFIG)
```

---

## 5. Early-Exit Layer Skipping Is Restricted to Decode Tokens

### Problem to Solve

In adaptive-depth algorithms (such as LayerSkip or CALM), developers sometimes want to extend early exit to the prefill stage so certain prompt tokens exit at shallow layers to reduce Time to First Token (TTFT).

### pLLM's Choice

The dynamic skip mask `active=` in `@pllm.custom_forward` is strictly restricted to `tokens="decode"`. Passing `active=` under `tokens="all"` or during prefill raises a build-time error:

```text
Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. Prompt rows run the full model by default; add an unmasked tokens="prefill" method only to change prompt computation (prefer over="span").
```

### Why

When a row skips later layers, pLLM backfills approximate K/V for the skipped layers so subsequent tokens still have entries to read at those positions:
- During decode, only that single newly generated position is skipped;
- Every token in the prompt forms the shared context read by all future generated tokens. Skipping layers during prefill replaces the entire context's upper-layer K/V with approximations, compounding error across every subsequent decode step.

Prompts therefore run the full model by default, and `active=` is allowed only with `tokens="decode"`.

### Alternatives

- Keep prompts running through all layers to build an accurate KV cache foundation, and apply `@pllm.custom_forward(tokens="decode")` with `layers(..., active=mask)` during autoregressive decode.
- If you genuinely need to alter prefill topology (e.g., looped depth or span-level forward passes), declare an unmasked forward method or a span-specific forward (`over="span"`).

---

## 6. MTP, Speculative Decoding & Outer Decode-Loop Control

### Problem to Solve

Multi-Token Prediction (MTP), speculative decoding (draft-model speculation or Medusa multi-head verification), and tree-based rollback speculation aim to predict multiple future tokens per step or roll back generation progress and KV cache positions when draft tokens are rejected.

### pLLM's Choice

pLLM does not support taking over the outer decode scheduling loop (draft–verify loops, KV cache rollback, or dynamic tree-branch expansion).

### Why

pLLM is designed as a **programmable extension substrate inside and around a single-step forward pass**. Its lifecycle covers a single engine step: request admission, forward execution, hidden-state reads/writes, residual rewrites, attention interception, pre/post-sampling hooks, and online backward passes.

Speculative decoding and rollback algorithms fundamentally **alter the outer scheduler's control state machine**: managing draft-model streams, pruning candidate trees, decrementing token position counters, and transactionally rolling back physical KV block tables. Those responsibilities live in the serving scheduler and batching engine's outer loop, outside a single-step modeling substrate.

### Alternatives

- To accelerate production throughput via speculative decoding, use vLLM's built-in native speculative decoding support.
- To reshape candidate probabilities or incorporate intermediate-layer predictions on the current step, use `@pllm.add_logits` for additive vocabulary biases or `@pllm.guide_sampling` for contrastive reweighting.
- For multi-round generation, candidate resampling, or verification feedback, drive batched generation and updates from a host feedback loop (see [05 Host Feedback Loops](../tutorials/05_host_feedback.ipynb)).

---

## 7. Pretraining New Architectures from Scratch Is Out of Scope

### Problem to Solve

A researcher wants to invent an entirely new neural architecture from scratch—for example, a non-attention token mixer, a redesigned MoE router dispatch system, or pretraining a randomly initialized model from zero.

### pLLM's Choice

Pretraining new architectures from scratch is outside pLLM's scope.

### Why

pLLM exists to solve the engineering challenges of deploying **inference-time and test-time algorithms** (activation steering, online fine-tuning, adaptive compute, fast-weight memory, lookup residuals, etc.) inside a production serving engine.

Pretraining a base model from scratch requires distributed cluster orchestration (Megatron-LM, FSDP, DeepSpeed), streaming petabyte-scale datasets, and fault-tolerant checkpointing—a completely separate engineering stack. Every abstraction in pLLM centers on serving and adapting existing pretrained checkpoints.

### Alternatives

- **Attach new sublayers onto a supported base**: Use [`@pllm.custom_residual`](../guides/forward-structure.md), `pllm.model.ActivationPatch`, and `pllm.model.Adapter` to mount fine-tuned or incrementally trained modules onto a supported base model.
- **Load newly admitted hybrid architectures**: Pretrained hybrid architectures from the community (such as Qwen3.5 mixing linear and full attention) work out of the box in pLLM once admitted to the support matrix.
- **Online test-time learning**: Test-time training (TTT / ESamp / TLM) is first-class in pLLM via `@pllm.train` and `@pllm.train_in_forward`; see [Online Training](../guides/training.md).

---

## 8. Admission by Exact Native Model Class

### Problem to Solve

A developer passes a custom model wrapper, a derived subclass (such as a class inheriting from `Qwen2ForCausalLM`), or an unregistered class from a related family (such as Phi-3 or Mistral variants).

### pLLM's Choice

pLLM validates admission strictly by exact native vLLM model class (`type(model) in architectures`), rejecting any unregistered subclass:

```text
choose a native vLLM model class; workload admission does not accept model subclasses
```

### Why

pLLM's efficiency and zero-copy interception rely on strict structural invariants of the admitted model classes:

1. **Residual interception contract**: Intercepting per-layer activations relies on admitted families following a uniform sublayer convention—specifically, $2L+1$ LayerNorm / RMSNorm modules that accept `(branch, residual)` and return `(normalized, sum)`. Any unadmitted subclass that changes norm signatures or call order breaks residual mounting.
2. **Projection naming and layout**: `pllm.model.Adapter` and weight overlays bind directly to exact projection attributes (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, etc.). Custom subclasses frequently fuse projections (e.g., packing QKV into one matrix), alter tensor alignment, or change the LM Head. Without strict allowlist admission, mismatched tensor slicing would crash CUDA Graph capture or silently read/write the wrong memory offsets.

### Alternatives

- Use an admitted native vLLM model class from the [Support Matrix](../reference/support-matrix.md) (e.g., `Qwen2ForCausalLM`, `LlamaForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`).
- If your checkpoint is structurally compatible with an admitted family, ensure the `architectures` field in `config.json` names the standard native vLLM class.

---

## 9. Fail-Closed Guarantees & Explicit Expensive Tradeoffs

### Problem to Solve

In complex experiments or deployments, when memory runs tight, graph capture hits an obstacle, or component dependencies collide, traditional ML frameworks often resort to "silent fallbacks"—quietly disabling CUDA graphs in favor of slow eager execution, spilling tensors to CPU RAM, or silently dropping updates.

### pLLM's Choice

pLLM enforces a **fail-closed** policy: any semantic ambiguity, undeclared budget overflow, or incompatible combination raises an explicit error at build time (`pllm.explain` or `pllm.make_llm`) instead of silently switching execution paths. The sole automatic fallback is CUDA graph capture: under the default `cuda_graphs="auto"`, certain patterns requiring custom sliced scheduling (such as `@pllm.custom_forward`, where pLLM captures per-layer-slice graphs itself) let vLLM skip the whole-model graph, with the reason reported on `explain`'s `forward` line and in `eager_reason`; setting `cuda_graphs=True` makes that a hard error too. Expensive system tradeoffs must be declared explicitly in code:
- **Storage placement**: large lookup tables in GPU VRAM (`placement="device"`) vs. host DRAM (`placement="host"`);
- **Physical KV layout**: sacrificing positions to save memory (`keep="window"`, `sink=True`, or `keep="none", source=k`);
- **Training visibility lag**: strict blocking visibility (`publish_lag=0`) vs. one-step delayed publishing for full async overlap with the next forward (`publish_lag=1`).

### Why

In high-concurrency serving, silent fallbacks are catastrophic:
- Silently spilling a large lookup table to host RAM over PCIe slows every step by an order of magnitude, cratering production throughput with no obvious root cause;
- Silently dropping CUDA Graphs turns small-model or low-concurrency decode throughput down by half (on an RTX 4090 at concurrency 128, 0.5B drops to 49% of graph throughput; 7B steps are longer and lose ~2%; see [Performance & Memory](performance.md));
- Silently delaying or dropping online weight updates degrades output quality irreproducibly, leaving researchers unable to tell whether their algorithm is flawed or the runtime cut corners.

### Alternatives

- Run `pllm.explain(workload, config)` on CPU before launching the GPU engine to inspect the full execution plan and memory bill (see [Inspecting Costs: `pllm.explain`](../guides/explain.md)).
- Configure expensive tradeoffs explicitly for your workload: pass `ExecutionOptions(publish_lag=0)` when strict step-by-step update visibility is required, or set `placement="host"` / static KV layouts when VRAM is tight.

---

## 10. At Most One `@pllm.guide_sampling` Method

### Problem to Solve

In a composed workload (for instance, combining two algorithm submodules), a developer attaches `@pllm.guide_sampling` to two different methods so both models or feature streams can guide the final sampling distribution.

### pLLM's Choice

A workload (including all nested components) may declare at most one `@pllm.guide_sampling` method. Declaring two or more fails at compile time:

```text
... and ... both declare @pllm.guide_sampling; keep at most one guidance method
```

### Why

This differs fundamentally from `@pllm.add_logits`:
- `@pllm.add_logits` simply adds a bias vector to unnormalized pre-filter logits. Addition is commutative and associative ($A + B = B + A$), so applying multiple additive biases in any order yields the exact same result—pLLM allows any number of `@pllm.add_logits` methods to coexist.
- `@pllm.guide_sampling` performs **nonlinear distribution reweighting** inside the sampler after top-k / top-p truncation (via the contrastive guidance rule `(1 + beta) * logits - beta * pred_logits`). Given two distinct hidden-state predictions, should they be averaged in feature space, multiplied in probability space, or cascaded through two separate truncations? Each choice defines a different algorithm, and pLLM refuses to guess.

### Alternatives

- Combine multiple guidance signals explicitly inside a single method—read the layers or modules you need, apply your weighting, interpolation, or gating math, and return one unified tensor;
- For simple additive vocabulary score adjustments, use stackable `@pllm.add_logits` methods instead.

```python
@dataclass(frozen=True)
class VocabularyBiases(pllm.Workload):
    """Multiple @pllm.add_logits methods can coexist because vocabulary bias addition is commutative."""

    @pllm.add_logits()
    def shallow_bias(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return torch.zeros((h.shape[0], 151936), device=h.device, dtype=torch.float32)

    @pllm.add_logits()
    def deep_bias(self, h: torch.Tensor = pllm.hidden(-1)) -> torch.Tensor:
        return torch.zeros((h.shape[0], 151936), device=h.device, dtype=torch.float32)


bias_report = pllm.explain(VocabularyBiases(), CONFIG)
```

---

## 11. Prefix Caching Must Be Disabled When Rewriting Forward Execution

### Problem to Solve

In multi-turn chat or long system-prompt workloads, a developer wants to enable prefix caching (`enable_prefix_caching=True`) alongside `@pllm.custom_residual`, `@pllm.custom_attention`, parameter overlays (`pllm.model.Adapter`), or activation patches (`pllm.model.ActivationPatch`).

### pLLM's Choice

Whenever a workload rewrites internal model computation sites, prefix caching must stay disabled (`enable_prefix_caching=False`, the default). Enabling prefix caching alongside forward rewrites is rejected at admission time:

```text
disable prefix caching (enable_prefix_caching=False) when a workload rewrites model sites; cached KV would not match
```
or
```text
disable prefix caching when using parameter overlays or activation patches
```

### Why

Prefix caching assumes that "identical input token sequences on the same model always produce bitwise-identical KV caches," allowing cached KV pages to be reused across requests solely by hashing token IDs.

When your workload dynamically modifies forward computation:
1. Activation patches or residual rewrites make the KV tensors written at each layer depend not only on token IDs, but also on the request's dynamic state, online-trained weight version, or injected steering vector;
2. With prefix caching enabled, a later request with the same prompt would silently reuse stale KV pages computed under a different request's state, corrupting the forward pass and breaking mathematical correctness.

### Alternatives

- Keep `ModelConfig(enable_prefix_caching=False)` (pLLM's default).
- Pure read-only observation workloads (extracting probe features with `@pllm.export`) and pure vocabulary-bias workloads (`@pllm.add_logits`) do not alter in-layer forward activations or KV generation, so prefix caching may be safely enabled with them.

---

See the [Error Catalog](../reference/errors.md) for the exact error messages and fixes corresponding to these boundaries.

---

## Algorithms Are Not pLLM's Job: pLLM Provides the Substrate

When evaluating whether an algorithm can run efficiently inside a serving engine, the key mental model is separating **algorithm math** from **system substrate**:

- **Algorithm math belongs to you**: projection formulas, hidden-state interpolations, loss functions, and even custom Triton kernels are yours to define freely.
- **System substrate belongs to pLLM and vLLM**: tensor memory layouts, static CUDA Graph capture, continuous-batch packing, and strict cross-request memory isolation form the foundation that guarantees high throughput, low latency, and deterministic outputs.

When an algorithm hits a boundary today because it conflicts with physical or serving constraints, that boundary marks where the algorithm's assumptions need to align with high-throughput concurrent inference. Understanding these tradeoffs helps you write valid `pllm.Workload`s and design new algorithms with serving performance in mind from day one.

---

## Self-Check

1. **Why does pLLM allow multiple `@pllm.add_logits` methods in a workload, yet at most one `@pllm.guide_sampling` method?**
   <details><summary>Answer</summary>
   <code>@pllm.add_logits</code> simply adds bias vectors to pre-filter vocabulary logits; addition is commutative and associative ($A + B = B + A$), so the result is uniquely defined regardless of order. By contrast, <code>@pllm.guide_sampling</code> performs nonlinear contrastive reweighting ($(1 + \beta)\text{logits} - \beta\,\text{pred\_logits}$) on a distribution already truncated by top-k / top-p filtering. Combining two predictions—averaging features, summing log-probabilities, or cascading truncations—represents distinct algorithmic semantics, which pLLM refuses to guess.
   </details>

2. **If a researcher wants to drop unimportant prompt tokens to reduce Time to First Token (TTFT), why can't they pass an `active=` mask for prompt rows in `@pllm.custom_forward`?**
   <details><summary>Answer</summary>
   When a row skips a layer via early exit, pLLM backfills approximate K/V entries using that layer's projections on the row's current hidden state so the KV cache stays contiguous. Prompts span tens to thousands of tokens; skipping layers during prefill replaces whole stretches of upper-layer context KV with synthetic approximations. Over hundreds of subsequent decode steps, every generated token would attend over distorted prompt KV, rapidly degrading generation quality.
   </details>

3. **Why is enabling Prefix Caching mutually exclusive with `@pllm.custom_residual` or parameter overlays?**
   <details><summary>Answer</summary>
   Prefix caching assumes that identical prompt token sequences always produce identical KV caches and indexes reusable pages solely by token-ID hashes. When <code>@pllm.custom_residual</code> or parameter overlays alter forward computation, the resulting KV values depend on dynamic per-request state, online-trained weight versions, or injected steering vectors. Enabling prefix caching would cause new requests with matching prompts to reuse stale KV pages produced under another request's state, violating mathematical correctness.
   </details>

---

## Where to Go Next

- Browse the full feature feasibility checklist: [Capability Map](../guides/capability-map.md)
- Review core physical mechanisms and the step timeline: [Core Concepts](core-concepts.md)
- Inspect measured throughput and memory overhead across mechanisms: [Performance & Memory](performance.md)
- Learn how to write valid custom-attention reads: [Customizing Forward Structure](../guides/forward-structure.md) and [10 Custom Attention](../examples/10_custom_attention.ipynb)
- Learn how to implement token-choice adaptive depth: [12 Adaptive Depth](../examples/12_adaptive_depth.ipynb)
- Look up full error messages and fixes: [Error Catalog](../reference/errors.md)
