# The Second Half of the Bitter Lesson: Why Commercial Inference Is Returning to Dirty Priors, and Why Engine Forks Break the Economics of Search

> **TL;DR**: Richard Sutton's *Bitter Lesson* rests on the premise that hardware cost per unit of computation falls exponentially. With per-transistor cost flattening out and compute-to-bandwidth ratios exceeding 800:1, training and inference economics have diverged—pushing commercial inference back toward N-gram lookup tables, cross-layer index reuse, and online fast weights. Yet Sutton's second warning was that human specialist effort does not scale with compute. Today coding agents can turn local prior design into scalable search, except stateless serving engines charge a three-week engine-fork tax per idea that stalls the loop.

For most of the past decade, Richard Sutton's *Bitter Lesson* operated as deep learning's house creed: domain priors and intricate local machinery eventually lose to uniform architectures backed by more compute.

While scaling curves ran steep, reality cooperated. Aside from a slightly awkward tokenizer at both ends, a production LLM was a uniform stack of identical Transformer blocks. Veterans of 1990s NLP who spent years curating lexicons and bigram feature tables became routine cautionary tales.

Read recent inference-architecture papers and open-source releases, however, and a quieter shift stands out. Presentations still praise architectural minimalism, while deployed models wire multi-hundred-megabyte bigram and trigram hash tables into residual streams, reuse sparse-attention indices across adjacent layers, or update rank-4 fast weights online during decode. The statistical feature-engineering tricks ushered out the front door twenty years ago have slipped back in through the loading dock wearing a badge that reads "sparse memory layer."

Try benchmarking concurrent traffic on most of those paper repositories, though, and the entry script greets you with:

```python
assert len(input_ids) == 1, "only support bs=1 for inference"
```

When a systems-heavy lab avoids `bs=1`, you usually get the opposite burden: a sprawling `third_party/vllm-custom` tree frozen at a six-month-old upstream commit, carrying thousands of lines of edits across the scheduler and `GPUModelRunner` that nobody dares rebase. Where inference algorithms are heading and what serving engines assume are pulling in opposite directions.

---

Why are messy priors returning? Look at silicon unit economics rather than aesthetic taste.

Sutton stated the physical premise of *The Bitter Lesson* up front: general-purpose methods win because Moore's law—or its broader generalization—keeps driving down cost per unit of computation exponentially. So long as waiting eighteen months hands you dramatically cheaper FLOPs, spending researcher months tuning local structure is a losing bet.

That premise has stalled. Foundry node names keep marching down to two nanometers and one nanometer in marketing copy, yet those labels detached from physical gate pitch years ago; rising lithographic complexity and fab capex have flattened—and on some nodes inverted—cost per transistor. Narrower number formats have also picked most of the easy gains, and nobody can quantize past one bit. Memory has been tighter still: raw DRAM density inches forward slowly while advanced packaging carries most bandwidth gains, stretching chip peak compute versus memory bandwidth past **800 to 1**. Every byte fetched from HBM or GDDR demands roughly eight hundred arithmetic operations just to keep compute units occupied.

That imbalance has pulled **training** and **inference** onto diverging economic tracks:

```text
Training economics:  User volume grows ──► Labs pool capital into 10x clusters ──► Uniform scaling still pays
Inference economics: Token price sets margin ──► Annual price cuts required ──► Trade structural purity for fewer bytes moved
```

Viewed along the inference track, recent architectures are all making variations of the same trade—**spending structural priors or local state to save memory bandwidth**:
- **Exploiting discrete language priors to skip dense routers**: Mapping continuous activations back to sparse buckets normally requires dense router GEMMs. DeepSeek's Engram leans on two human priors instead: human civilization already discretized reality into words, and natural language concentrates heavily on bigram and trigram collocations—so hashing recent tokens straight into a static table fetches vectors with zero router matmuls (with a real catch: strong memorization paired with weak compute often drives training loss down while validation loss rises on multi-epoch runs).
- **Exploiting cross-layer similarity to amortize indexers**: When sparse-attention indexers eat up bandwidth savings, Zhipu's IndexCache and DeepSeek's CSA2 lean on the empirical prior that adjacent layers attend to nearly identical salient tokens, reusing selected indices across blocks.
- **Exploiting online local state on long-tail prompts**: Rather than doubling frozen base weights, test-time adaptation attaches compact per-request fast weights or predictors (In-Place TTT, online self-distillation ESamp, hypernetwork-generated LoRAs) updated on the fly.

Discrete lookups and online patches rarely look like a single textbook equation, because discrete optimization rarely does.

---

The sharper twist sits in the second half of Sutton's essay.

Why did Sutton argue against baking human knowledge into algorithms? He gave three operational reasons: time spent on one path is time not spent on the other; researchers form psychological commitments to their machinery; and human-knowledge methods complicate systems in ways that hinder general compute scaling. In plain economic terms, Sutton's second bottleneck was that **human specialist labor does not scale with compute**.

Five years ago, that critique was decisive—hand-crafting one stateful inference trick took two PhD students a semester. Today, coding agents and evolutionary search scripts are changing that side of the equation. Handing an agent a preprint and generating twenty cross-layer routing or self-distillation variants takes an afternoon. Designing local priors is turning into the very mechanism Sutton championed: **scalable search**.

Why, then, is the ecosystem still bottlenecked on `bs=1` scripts and abandoned `vllm-custom` forks?

Because the bottleneck moved from inventing the math to surviving the serving engine. Production engines like vLLM extract an order-of-magnitude throughput advantage from one strict invariant: **aside from an append-only paged KV cache, the model is stateless and read-only across concurrent requests.**

```text
Research-script model (bs = 1, mutable):
  One sequence owns the model ──► Python hooks fire per layer ──► Weights mutate via param.add_() in place

Serving-engine model (100+ concurrent sequences, static & read-only):
  Tokens flatten into 2D [tokens, H] ──► Fused ops freeze inside CUDA Graphs ──► Lock-free multi-stream pipeline
```

Drop per-request mutable state into that pipeline and both standard options break down: staying in a single-stream Hugging Face loop forfeits continuous batching and CUDA Graphs, serializing concurrent callers in a queue; forking vLLM recovers hardware efficiency at the price of three weeks of systems plumbing—reconstructing row ownership from flattened batches, reserving constant base pointers for CUDA Graphs, preventing torn reads across streams, rewinding state on KV preemption, and aligning tensor-parallel shards. Pay a three-week engine-surgery tax per candidate and automated search stalls at the starting line.

---

Cutting that switching cost from three weeks back to half an hour—without accumulating paper-specific special cases—follows the same separation of concerns Linux used when eBPF replaced out-of-tree kernel modules. Leave upstream vLLM untouched, and factor stateful inference into four orthogonal declarations: **who owns the state** (per-request or instance-shared), **how transitions evolve it** (stepwise optimizers, in-forward chunk recurrences, hypernetworks, or live imports), **where the model consumes it** (residual seams, paged or latent attention, projection overlays, or sampler epilogues), and **how many steps later updates publish**.

We built **[pLLM](https://github.com/LinesHogan/pLLM)** (`pllm`, an open-source programmability layer running atop unmodified vLLM) to test that boundary. Expressing online self-distillation sampling (ESamp)—where each concurrent request trains its own rank-4 MLP via `AdamW` every decode step while suppressing tokens already predictable from layer 0—takes a dozen lines of plain PyTorch:

```python
from dataclasses import dataclass
import torch
from torch import nn
import pllm


@dataclass(frozen=True)
class ESampLite(pllm.Workload):
    beta: float = 0.5
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.cfg("beta"))
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())
```

Block-table routing, versioned slot pools, side-stream CUDA Graph capture for backward + `AdamW`, and post-filter survivor dot products stay inside the runtime. On a 7B model across 128 concurrent sequences, online training plus guidance sustains **over 96%** of native vLLM throughput; porting a community 4B In-Place TTT checkpoint onto the same primitives beats the official single-stream loop for one caller, doubles throughput across four concurrent callers, and serves live SSE traffic behind `pllm serve` unchanged.

Respecting a host engine's invariants without forking it still enforces hard boundaries. Captured graphs and multi-tenant isolation rule out data-dependent Python branches and cross-row batch reductions; algorithms like H2O or SnapKV that demand score-based mid-sequence eviction of physical KV blocks conflict directly with vLLM's scalar progress counter and append-only block tables, so admission fails closed up front. Hardware ratios remain non-stationary too—if 3D-stacked DRAM narrows the compute-to-bandwidth gap years from now, half of today's discrete priors may retire again. Infrastructure does not need to predict eternal winners; letting ideas live or die on real concurrent traffic instead of dying inside three-week engine forks is work enough.
