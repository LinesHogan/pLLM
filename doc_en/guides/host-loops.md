# Host Loops and Batch Training

> **Prerequisites**: [State](state.md); Tutorial [05 Feedback Loops](../tutorials/05_host_feedback.ipynb) (TTRL, host batch training).
>
> **What this guide covers**: How to organize "generate → score → train" closed loops on the Python host side (such as TTRL, GRPO, and reward-head fitting), how to feed external batches into the engine, and how to unit-test host state without a model engine.

## Motivation: Why Host Batch Training and Feedback Loops?

In many test-time reinforcement learning (TTRL, GRPO), test-time training (TTT), or adaptive search algorithms, updates cannot happen in place during a single token's forward pass:
1. **Dependence on global feedback or complete sequences**: majority voting must wait until every sampled answer in a group finishes before counting votes for a pseudo-label; GRPO needs rewards across all candidate sequences in a group to compute advantages and group-level normalization.
2. **Training through the full model or external heads**: an algorithm may need to backpropagate a causal language modeling loss through the model to update LoRA factors, or simply fit a lightweight reward head.

Maintaining two separate systems—an inference engine and a training framework—forces cross-process serialization, transfer, and weight synchronization on every round, adding major engineering complexity and overhead.

In pLLM, host code and the engine share the same state and VRAM:
- The host Python loop generates completions, computes scores/rewards, and constructs batch objects;
- The workload declares methods that read host-supplied batch fields (`pllm.batch(...)`);
- The engine automatically runs an optimizer step at the start of the next `generate(..., batch=...)` call—immediately using the updated parameters for generation—or steps host-only training independently via `pllm.host.make_host`.

---

## Core Mechanism: Reading Host Batches with `pllm.batch`

Inside a `@pllm.train`-decorated method, declare parameter defaults with `pllm.batch("field_name")` to inject batches sent from the host.

### Three Batch Types

Which `pllm.batch` fields a method reads determines the batch type it consumes and the optimizer bound to it:

| Batch Type | Fields | Bound Optimizer & Step Rule | Use Cases |
| :--- | :--- | :--- | :--- |
| `pllm.host.WeightedRowBatch` | `hidden`, `targets`, `weights` | `pllm.AdamW`, one step per batch | Reward head training, feature regression, FTTT and other hidden-vector fitting (can run purely on host) |
| `pllm.host.CausalTrainBatch` | `input_ids`, `labels`, optional `attention_mask`, optional `weights` | `pllm.AdamW`, one step per batch | Causal LM training (e.g., TTRL, online GRPO, TLM) backpropagating end-to-end through the model into LoRA factors |
| `pllm.host.WeightedCEBatch` | `hidden`, `base_logits`, `labels`, `weights` | `pllm.SGD`, runs `steps=` steps with `reset=` | Weighted cross-entropy batch training (e.g., Output LoRA), supporting per-prompt batches (`WeightedCEBatch.for_prompts`) |

> [!IMPORTANT]
> Every `pllm.batch` field read by a single training method must belong to the same batch type; mixing fields across batch types is not allowed.

### Causal Language Model Port: `pllm.causal_model`

When an algorithm needs end-to-end autoregressive training through the language model itself, declare a parameter:
```text
model: pllm.model.CausalModel = pllm.causal_model("adapter_name")
```
where `"adapter_name"` is the attribute name of a `pllm.model.Adapter` declared on the workload. The injected `CausalModel` provides:
- `model(input_ids, attention_mask=None)`: returns logits with the adapter applied; gradients flow directly into the adapter's bound state factors.
- `model.base(input_ids, attention_mask=None)`: returns logits from the frozen base model without the adapter (useful for off-policy importance weights or baseline scoring).
- `model.token_logprobs(input_ids, labels, attention_mask=None, ignore_index=-100)`: computes chunked next-token log-probabilities at every position, returning a float32 tensor of shape `[num_seqs, seq_len - 1]`.
  - **Why chunked**: autoregressive vocabularies are often huge (e.g., 150k tokens). Materializing a full `[num_seqs, seq_len, vocab_size]` logits tensor before cross-entropy can instantly consume gigabytes of VRAM and trigger OOM. `token_logprobs` chunks the LM head projection internally to avoid materializing full-vocabulary logits for the whole batch. Positions whose label equals `ignore_index` (default `-100`) receive log-probability `0`.

---

## Loss Reduction Contract: Reduce Yourself and Return a Scalar

In ordinary per-request training, methods return a per-row loss (`[rows]`) and the framework reduces it via `reduction="mean"` or `"sum"`. In host batch training, the rule is completely different:

**Host batch training methods must perform their own loss reduction and return a scalar directly (or `pllm.UpdateOutput`). Passing `reduction=` to `@pllm.train` on a host batch method is rejected at declaration time.**

### Why Host Batch Training Disallows External `reduction=`

In per-request training, how many request rows appear in each engine step is decided dynamically by the continuous-batching scheduler, so user code cannot know the scheduling layout ahead of time and delegates reduction to the framework.

A host batch, by contrast, consists of complete sequences assembled by your own Python host code. In sequence-level RL and alignment algorithms, how sequence lengths $|o_i|$ and group size $G$ are normalized is part of the algorithm's core mathematics and cannot be expressed during token-by-token generation:

| Algorithm | Scalar Loss Reduction Formula | Normalization Behavior and Meaning |
| :--- | :--- | :--- |
| **GRPO** | $\frac{1}{G}\sum_{i=1}^{G} A_i \frac{1}{\|o_i\|} \sum_{t} \ell_{i,t}$ | Averages each sequence internally by its own generated length $\|o_i\|$, then averages across the $G$ candidates in the group. Every sequence gets equal weight regardless of length. |
| **DAPO** | $\frac{\sum_{i=1}^{G} \sum_{t} A_i \ell_{i,t}}{\sum_{i=1}^{G} \|o_i\|}$ | Sums all token losses across all sequences in the group and divides by the total token count. Longer sequences contribute proportionally more to the total gradient. |
| **RLOO / REINFORCE** | $\frac{1}{G}\sum_{i=1}^{G} A_i \sum_{t} \ell_{i,t}$ | Sums token losses within each sequence and averages across the $G$ candidate sequences in the group. |
| **Dr.GRPO** | $\frac{1}{G}\sum_{i=1}^{G} A_i \frac{1}{L_{\max}} \sum_{t} \ell_{i,t}$ | Replaces dynamic sequence length $\|o_i\|$ with a fixed maximum length constant $L_{\max}$, eliminating implicit length-normalization bias. |

Because only the algorithm author knows which reduction formula applies—and those formulas depend on full sequence lengths and advantage weights—the framework requires the method to compute the reduction explicitly and return a single scalar.

---

## Feeding Batches into the Engine: `llm.generate(batch=...)`

In a typical "generate → vote/score → update" loop, training batches are passed into `llm.generate`:

```text
outputs = llm.generate(prompts, sampling, batch=batch)
```

1. **Update timing and visibility**: when `batch` is provided, the engine executes the training method and takes one optimizer step at the start of that `generate` call. Once state (such as LoRA parameters) updates, all tokens generated in that call immediately use the updated parameters.
2. **VRAM budget: `max_host_training_rows`**:
   - Backpropagating through the full model requires substantial temporary activation memory.
   - At engine initialization, pLLM statically reserves this workspace based on `ExecutionOptions(max_host_training_rows=N)` so generation never hits unexpected OOM during a training backward pass.
   - Passing a batch whose row count exceeds `max_host_training_rows` is rejected before execution.
3. **First-shape compilation cost**:
   - The first time a previously unseen batch shape arrives (e.g., a new sequence count or max token length), pLLM compiles the training graph for that shape, making that round noticeably slower.
   - Once batch shapes stabilize, subsequent training steps reuse the compiled graph at full speed.

### Constraints and Reasons for Causal Batch Training

Using `CausalTrainBatch` with `pllm.causal_model` enforces four requirements:
1. **State scope**: must update an `InstanceState` configured with `pllm.AdamW`. Why: causal batch training runs forward and backward across the whole model, updating weight deltas shared by all requests.
2. **Base model dtype**: the base model must be BF16 (`bfloat16`), and neither `embed_tokens` nor `lm_head` may be quantized. Why: autograd must flow in BF16 through token embeddings and the LM head, and quantized layers do not support floating-point backpropagation directly.
3. **Single GPU**: requires `tensor_parallel_size=1`. Why: full-model causal forward and backward currently execute within a single GPU context; cross-rank backward and gradient aggregation under tensor parallelism are not supported yet.
4. **Low-rank initialization**: when using a LoRA adapter, `init=pllm.Computed(...)` must be provided (e.g., random $A$, zero $B$). Why: if both $A$ and $B$ start at zero, initial gradients for both are identically zero and the optimizer can never update the LoRA weights.

---

## Pure Host State: `pllm.host.make_host` Without a Model Engine

If a state is read and trained solely by host code (such as a reward head scoring final hidden vectors or an external classifier), it does not need a GPU vLLM engine at all.

You can compile the workload with `pllm.host.make_host` into a `HostTrainer` that supports single-step debugging and execution even on CPU.

### Runnable Example: `RewardHead` on CPU

The code below declares a reward-head workload and runs AdamW training on `WeightedRowBatch` on CPU via `make_host`:

```python
from dataclasses import dataclass
import torch
import pllm
from pllm.host import HostConfig, HostTrainer, TrainStep, WeightedRowBatch, make_host

@dataclass(frozen=True)
class RewardHead(pllm.Workload):
    head = pllm.InstanceState({"w": pllm.Field(lambda width, _: (width,))}, optimizer=pllm.AdamW(1e-2))

    @pllm.train(state="head")
    def fit(
        self,
        hidden: torch.Tensor = pllm.batch("hidden"),
        targets: torch.Tensor = pllm.batch("targets"),
        weights: torch.Tensor = pllm.batch("weights"),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> torch.Tensor:
        error = (hidden @ w - targets).square()
        return (weights * error).sum() / weights.sum().clamp_min(1)


with make_host(HostConfig(torch.device("cpu"), hidden_size=896), RewardHead()) as head:
    batch = WeightedRowBatch(torch.randn(4, 896), torch.ones(4), torch.ones(4))
    step = head.train(batch)
    print(step.loss is not None, tuple(step.parameters["w"].shape), head.version)
```

- `make_host(config, workload)`: compiles the workload and initializes its instance state.
- `head.train(batch)`: runs one training step and returns a `TrainStep` (containing scalar `loss` and the updated `parameters` dictionary).
- `head.version`: the state's published version counter (incrementing from 0).
- `head.diagnostics()`: returns training diagnostics (`loss_count`, `update_count`, `trained_rows`, etc.).
- `head.close()` or a `with` context manager: releases resources on exit.

---

## Defining Causal Host Training and Inspecting Costs (`pllm.explain`)

Before launching causal batch training on GPU, you can use `pllm.explain` on CPU to inspect the execution plan and memory overhead of a TTRL workload:

```python
import math
from collections.abc import Mapping

def lora_init(fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    for name, value in fields.items():
        if name.startswith("a_"):
            torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
        else:
            value.zero_()


@dataclass(frozen=True)
class TTRL(pllm.Workload):
    rank: int = 8
    learning_rate: float = 1e-4

    @property
    def lora(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            optimizer=pllm.AdamW(self.learning_rate),
            init=pllm.Computed(lora_init),
        )

    @property
    def adapter(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj"), pllm.model.Projection("v_proj"))
        return pllm.model.Adapter("lora", targets, operator="low_rank", scale=2.0)

    @pllm.train(state="lora")
    def learn(
        self,
        input_ids: torch.Tensor = pllm.batch("input_ids"),
        labels: torch.Tensor = pllm.batch("labels"),
        attention_mask: torch.Tensor | None = pllm.batch("attention_mask"),
        model: pllm.model.CausalModel = pllm.causal_model("adapter"),
    ) -> torch.Tensor:
        logprobs = model.token_logprobs(input_ids, labels, attention_mask=attention_mask)
        return -logprobs.sum() / labels[:, 1:].ne(-100).sum()


config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=640,
    max_num_seqs=8,
    execution=pllm.ExecutionOptions(max_host_training_rows=8),
)
report = pllm.explain(TTRL(), config)
print(report.states[0].name)
```

`pllm.explain` runs entirely on CPU without loading real weights or allocating CUDA memory, letting you validate workload structure and memory budgets while writing code.

---

## Building Closed Loops: Retry Loops and `llm.generate_feedback`

pLLM offers two ways to structure host loops:
1. **Explicit global loop (`llm.generate(batch=...)`)**: re-generates samples for all prompts each round, computes global rewards on the host to assemble a `CausalTrainBatch`, and updates model weights at the start of the next generation round (e.g., TTRL / GRPO).
2. **Fine-grained retry loop (`llm.generate_feedback(...)`)**: targets workflows that retry only failed samples across rounds (such as math self-consistency retries or FTTT), with per-sample retries and synchronization barriers.

### Fine-Grained Retry Mechanism: `generate_feedback`

The calling pattern for `llm.generate_feedback` is:

```python
# check-docs: skip: GPU engine and CUDA are required
from vllm import SamplingParams

from pllm.workloads import DummyWorkload
from pllm.workloads.reward_head import RewardHeadConfig, RewardHeadFeedback, RewardHeadWorkload

gpu_config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)
prompts = ["What is 2 + 3? Put the final answer in \\boxed{}.", "What is 7 * 6? Put the final answer in \\boxed{}."]
sampling = SamplingParams(n=4, max_tokens=64, temperature=0.8, seed=0)  # 4 samples per prompt for voting
head_workload = RewardHeadWorkload(RewardHeadConfig(rank=16, learning_rate=1e-3))
host_cfg = HostConfig(torch.device("cuda"), hidden_size=896)

# generate_feedback collects each completion's final hidden state, so the engine needs a workload;
# observation-only DummyWorkload is enough
with pllm.make_llm(gpu_config, workload=DummyWorkload()) as llm, make_host(host_cfg, head_workload) as head_trainer:
    feedback_handler = RewardHeadFeedback(head_trainer, head_workload.config)
    for completion_input, vote_result in llm.generate_feedback(
        prompts,
        sampling,
        feedback=feedback_handler,
        group_id="vote_group_1",
        max_rounds=3,
    ):
        print(f"Samples in round: {len(completion_input.completions.entries)}, updates: {vote_result.update_count}")
```

- **Arguments**:
  - `prompts`: list of prompts to evaluate.
  - `sampling`: `SamplingParams` sampling `n` completions per prompt. Retries happen per sample: any sample marked `RETRY` is re-sampled individually (`n=1`) in the next round while keeping its sample index slot within the prompt—the $j$-th sample of round $k$ has index `k*n + j`. When `seed` is set, sample $i$ uses `seed + i`, so retries draw fresh samples.
  - `feedback`: user callback of type `Callable[[CompletionInput], FeedbackT | None]`.
  - `group_id`: group identifier for this batch of requests.
  - `max_rounds`: maximum number of iteration rounds.
- **Callback input `CompletionInput`**:
  - `completions` (`CompletionBatch`): all `Completion` objects from the current round (each carrying `identity`, `text`, `token_ids`, `finish_reason`, etc.).
  - `hidden`: final-position hidden activation vector for each generated completion (`torch.Tensor` of shape `[num_samples, hidden_size]` in float32).
- **Callback return value**: the returned object must expose these attributes (returning `None` terminates the loop early):
  - `retry_controls` (`tuple[RetryControl, ...]`): per-sample scheduling decisions. Each `RetryControl` has `action` (`RetryAction.ACCEPT`, `RetryAction.RETRY`, or `RetryAction.STOP`), `reason` (`ControlReason`), and `retry_after_update` (bool indicating whether the retry must wait for an update).
  - `update_barriers` (`tuple[UpdateBarrier, ...]`): synchronization barriers declaring a set of `request_ids` and a state (`BarrierState.READY` or `BLOCKED`); update-dependent requests are re-sampled only when the barrier is ready and `update_count > 0`.
  - `update_count` (int): how many parameter updates occurred in this round.
- **Built-in helpers**:
  - `majority_vote(value: CompletionInput, config: RewardHeadConfig)`: extracts `\boxed{}` math answers, runs majority voting, marks agreeing samples `ACCEPT` and disagreeing ones `RETRY`, and builds `UpdateBarrier` plus a `WeightedRowBatch` for training the reward head.
  - `pllm.workloads.RewardHeadFeedback(head, config)`: pre-packaged feedback callback that runs `majority_vote`, trains the reward head via `head.train(vote.batch)`, and returns the vote outcome.

### Choosing Between `generate(batch=)` and `generate_feedback`

- **Use `generate(batch=)`**: when each round re-evaluates the current policy across all prompts (e.g., online RL, TTRL, GRPO), re-sampling every request each round and training adapters inside the model forward pass (such as LoRA).
- **Use `generate_feedback`**: when solving a concrete batch of tasks (e.g., self-correcting reasoning chains or verifier-guided retries) where any prompt with a valid answer exits immediately and only failed samples retry, or when the base model stays frozen and only a host-side reward head or scorer updates.

---

## Empty Selection Semantics

In real feedback loops, an entire batch of candidates may fail filtering (for instance, unanimous agreement yields zero GRPO advantages, or no sample produces a valid answer).

In that case, the training method can handle empty selection by:
1. Returning `UpdateOutput(loss, changed, update_count, trained_rows)` with `changed = False` (as a 0-d device bool tensor) and `update_count = 0`; or
2. Using `torch.where(total_weights > 0, loss, 0.0)` so the loss is zero when total weights are zero.

**Guarantee**: when the selection is empty (`changed` is false), pLLM automatically publishes the unchanged state parameters and leaves optimizer internals (moments, step counter) untouched. Crucially, `changed` is passed as a device tensor, so **no CPU-GPU synchronization wait occurs**, preserving pipeline throughput.

---

## Complete GPU Loop Example (TTRL Majority Voting)

Below is a full GPU loop running TTRL (condensed from [Tutorial 05](../tutorials/05_host_feedback.ipynb)):

```python
# check-docs: skip: GPU engine and CUDA are required
from collections import Counter
import re
from vllm import SamplingParams
from pllm.host import CausalTrainBatch

def get_boxed(text: str) -> str | None:
    found = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return found[-1].strip() if found else None

def prepare_batch(outputs, prompts, tokenizer) -> CausalTrainBatch | None:
    candidates = []
    for prompt, output in zip(prompts, outputs):
        answers = [get_boxed(s.text) for s in output.outputs]
        counts = Counter(a for a in answers if a is not None)
        if not counts:
            continue
        majority, _ = counts.most_common(1)[0]
        sample = next(s for s, a in zip(output.outputs, answers) if a == majority)
        candidates.append((tokenizer(prompt).input_ids, list(sample.token_ids)))
    if not candidates:
        return None
    width = max(len(p) + len(a) for p, a in candidates)
    ids = torch.full((len(candidates), width), tokenizer.pad_token_id, dtype=torch.int64)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for row, (p, a) in enumerate(candidates):
        ids[row, : len(p) + len(a)] = torch.tensor(p + a)
        labels[row, len(p) : len(p) + len(a)] = torch.tensor(a)
        mask[row, : len(p) + len(a)] = 1
    return CausalTrainBatch(ids.cuda(), labels.cuda(), mask.cuda())

gpu_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=640,
    max_num_seqs=128,
    max_num_batched_tokens=2048,
    gpu_memory_utilization=0.25,
    seed=0,
    execution=pllm.ExecutionOptions(max_host_training_rows=8),
)
sampling = SamplingParams(n=8, temperature=0.8, top_p=0.95, max_tokens=256, seed=0)

with pllm.make_llm(gpu_config, workload=TTRL()) as llm:
    batch = None
    for round_index in range(5):
        outputs = llm.generate(prompts, sampling, batch=batch)
        batch = prepare_batch(outputs, prompts, tokenizer)
        print(f"Round {round_index} complete, total updates: {llm.diagnostics().update_count}")
```

---

## Common Errors and Troubleshooting

For exact error messages, root causes, and fixes when declarations on this page are rejected, see [State and training](../reference/errors.md#state-and-training) and [Memory budgets](../reference/errors.md#memory-budgets) in the Error Catalog.

---

## Self-Check

1. Why must `max_host_training_rows` be declared upfront in `ExecutionOptions` when launching the engine?

<details><summary>Answer</summary>

Backpropagating through the whole model temporarily consumes large amounts of activation and workspace VRAM. pLLM statically reserves this VRAM at engine startup based on the declared maximum row count so generation never hits unexpected OOM when allocating large training workspaces mid-run. Batches exceeding that row count are rejected immediately.
</details>

2. In a TTRL or GRPO loop, why do certain rounds take noticeably longer than others?

<details><summary>Answer</summary>

Whenever the incoming training batch shape changes (e.g., number of sequences or maximum token length), pLLM compiles a training computation graph for that new shape, incurring a one-time compilation cost on that round. Once batch shapes stabilize, later steps reuse the compiled graph and return to normal speed.
</details>

3. Why cannot `@pllm.train` use `reduction="mean"` on host batch training, forcing you to divide manually inside the method?

<details><summary>Answer</summary>

Different sequence-level RL and alignment algorithms (such as GRPO, DAPO, RLOO, and Dr.GRPO) normalize sequence lengths and candidate groups differently (per-sequence length, total batch tokens, or fixed max length). That math can only be expressed explicitly by the algorithm author on the host batch; a single generic reduction parameter cannot capture all of them.
</details>

4. When should you use `llm.generate(batch=...)` versus `llm.generate_feedback(...)`?

<details><summary>Answer</summary>

Use `generate(batch=...)` when the algorithm re-samples all prompts every round and trains adapters inside the model forward (such as TTRL or online RL); use `generate_feedback(...)` when performing fine-grained partial retries (retrying only samples that failed verification, or keeping the model frozen while fitting a host-side reward head).
</details>

---

## Related Docs

- Tutorial: [05 Feedback Loops: TTRL](../tutorials/05_host_feedback.ipynb)
- Guides: [Online Training](training.md) · [State](state.md) · [Workload Anatomy](workload-anatomy.md) · [Seeing Costs: pllm.explain](explain.md)
- Reference: [Configuration](../reference/config.md) · [Core Concepts](../explain/core-concepts.md)
