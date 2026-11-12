---
name: pllm-workload
description: Write, inspect and check a pllm.Workload, an inference-time LLM algorithm that runs inside vLLM (activation steering, probes and SAE monitoring, logit edits such as DoLa, decode-time training such as ESamp, per-request adapters, residual rewrites, custom attention reads and KV layouts, token-history lookups, custom kernels, recurrent depth and early exit, host feedback loops such as TTRL). Use when asked to implement or port such an algorithm with pLLM, or to fix a pLLM rejection or ContractError.
---

# Writing a pLLM workload

pLLM runs inference-time algorithms inside vLLM (0.27.1 / 0.32.0) under continuous batching and CUDA
graphs. You declare what to read, what state to own and what the result does, and write the tensor math;
pLLM handles per-request state and memory, CUDA graph capture, overlap with the forward, and optimizer
updates.

It fits algorithms that read or add a computation at a point of the forward (steering, probes and SAEs,
ReFT, logit edits, logit lens and DoLa, residual rewrites), change how a layer reads its KV or how KV is stored
(attention monitors, Quest-style top-K reads, sliding-window and cross-layer KV layouts), look up tables keyed by
token history (n-gram priors, Engram-style memory), keep state that evolves with tokens per request or per engine
(generated LoRA, decode-time training such as ESamp, test-time learning, fast weights), rerun or skip layers per
token (recurrent depth, early exit), or control a generate-score-update loop (TTRL). Kernels of your own enter as
`torch.library.custom_op`. Out of scope: pretraining, new model architectures, controlling the decode loop
itself (speculative decoding, MTP), and freeing KV memory by score (H2O, SnapKV: read-only selection works).

## Where to look

| File | Use it for |
| --- | --- |
| this file | how a workload is built, which declaration fits which need, the workflow and how to report |
| [reference.md](reference.md) | every public name grouped by task, each section with a runnable example; generated signatures at the end |
| [pitfalls.md](pitfalls.md) | an error message or `ContractError` and its fix; anti-patterns; silent traps; preemption; shapes and dtypes; known issues |
| [rules.md](rules.md) | the invariants and why they exist; refused combinations; support matrix (models, CUDA graphs, TP); the expensive choices you must write explicitly |
| [examples.md](examples.md) | the nearest maintained example for a task, and which declarations it uses |

## How a workload is put together

A workload is a subclass of `pllm.Workload` holding only declarations and tensor math. pLLM reads the
declarations once, when the engine is built, and plans where each method runs; methods are not hooks that
Python calls at run time. The example shows the three kinds of declaration; the table after it lists the
names of each kind.

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing


@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    # What it owns: every request gets its own small network and AdamW state.
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # What the result does: a loss to minimize (decorator); what it reads: markers as defaults.
    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)  # [rows]

    # A setting read from the instance is a method the decorator names. Writing `beta=beta` would pass the
    # class default 0.5 forever (pitfalls.md, "A dataclass field named in the class body").
    def strength(self) -> float:
        return self.beta

    # A hidden-state prediction that guides sampling with strength beta.
    @pllm.guide_sampling(beta=strength)
    def predict(self, shallow: torch.Tensor = pllm.hidden(0),
                f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow.float())


CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
```

| Declares | Written as | Names |
| --- | --- | --- |
| what a method reads | a parameter default | `pllm.hidden`, `pllm.residual`, `pllm.final_hidden`, `pllm.layer_logits`, `pllm.input_tokens`, `pllm.parameter`, `pllm.module`, `pllm.version`, `pllm.index`, `pllm.sample_index`, `pllm.result`, `pllm.base_logits`, `pllm.sampled_tokens`, `pllm.batch`, `pllm.operand`, `pllm.causal_model`, `pllm.depth`, `pllm.value`, `pllm.attention`, `pllm.carry` |
| what the workload owns | a class attribute, or a property annotated with its type | `pllm.RequestState` (one copy per request), `pllm.InstanceState` (one shared copy), `pllm.PositionState`; starting values `init=pllm.Zeros()` / `Computed` / `Pretrained` / `ModelCheckpoint` / `FromModule` / `Fed` / `Imported`; optimizers `pllm.AdamW`, `pllm.SGD` |
| what a result does | a method decorator, or a model-site attribute | `@pllm.export`, `@pllm.add_logits`, `@pllm.guide_sampling`, `@pllm.train`, `@pllm.write`, `@pllm.train_in_forward`, `@pllm.custom_residual`, `@pllm.custom_attention`, `@pllm.custom_forward`; `pllm.model.ActivationPatch`, `pllm.model.Adapter` |

Everything else is your tensor math on row tensors: dimension 0 holds one row per request (or per sampled
token), packed in an order you do not control, so compute along `dim=-1`. Inputs are read-only. A kernel of
your own enters that math as a `torch.library.custom_op` with a `register_fake` (example 09); call the op
object, not `torch.ops.<namespace>`. Method code may use only `torch`, `math`, `pllm`, `triton`, `typing`,
`collections.abc`, builtins and your own module-level classes and functions ([rules.md](rules.md) says why).

### Composing workloads

A workload can hold other workloads as dataclass fields. Each field is an independent component: its
methods, states and export names get the field name as prefix (`monitor.norm` in explain), components cannot
read each other's state, and pLLM rejects real conflicts between them by name (for example two
`@pllm.guide_sampling` methods). `testing.check` adds a component-independence contract.

```python
from dataclasses import field


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


@dataclass(frozen=True)
class Both(pllm.Workload):
    distill: Distill = field(default_factory=Distill)
    monitor: StreamNorm = field(default_factory=StreamNorm)


print([member.name for member in pllm.explain(Both(), CONFIG).members])
```

## Which declaration for which need

Each rule names the choice, the tempting wrong one, and the reason.

| You need | Use | Not | Why |
| --- | --- | --- | --- |
| the stream a steering vector, SAE or probe was trained on (Hugging Face `hidden_states[l + 1]`) | `pllm.residual(l)` | `pllm.hidden(l)` | `pllm.hidden` is only the layer's MLP output before it is added to the stream. |
| the state the LM head reads (`last_hidden_state`) | `pllm.final_hidden()` | `pllm.residual(-1)` | The final norm is applied only in `final_hidden`. |
| the model's logits to adjust before sampling | `pllm.layer_logits((-1,))` | `pllm.base_logits()` | `base_logits` exists only after the draw. |
| to observe values without changing outputs | `@pllm.export` + `on_export` | `print`, `@pllm.write`, an `ActivationPatch` | Exports are bounded, asynchronous and keep CUDA graphs; nothing else leaves the GPU without a sync. |
| one row per prompt (prompt end), or every token | `tokens="prefill"`, or `tokens="all"` (exports only) | the default `"decode"` | The default reads only each request's latest decode token. |
| memory per request | `pllm.RequestState` | dicts keyed by request id, `self.x` | Rows arrive packed and permuted, maybe in another process; pLLM admits, pages and frees request state. |
| state shared by the `n` samples of a prompt, or by requests with a user key | `key_fields=("prompt_index",)`, or `("route",)` with `generate(routes=...)` | instance state plus bookkeeping | The key decides which requests share one copy; route copies survive between calls. |
| frozen weights or tables | `InstanceState(init=pllm.Pretrained(path))`, `pllm.FromModule(factory, pretrained=path)`, `pllm.ModelCheckpoint(...)` | a tensor or module on the workload | The workload is hashed and shipped by import; state is placed and budgeted by pLLM. |
| values the host computes per `generate` call | `init=pllm.Imported()` + `generate(batch=pllm.io.StateImport(...))` | a workload field changed between calls | Declarations are fixed when the engine is built. |
| a continuous producer (video, sensors) | `init=pllm.Fed("ordered")` + `llm.feed(name)` | repeated imports | Feeds publish with backpressure from another process. |
| to add a vector or a low-rank edit to a layer's output from state | `pllm.model.ActivationPatch` | `@pllm.custom_residual` | The patch is declarative and runs in CUDA graphs; use a custom residual only when the update is your own function of `x` and `y`. |
| a weight delta on projections (LoRA) | `pllm.model.Adapter` on a `RequestState` | an Adapter on an `InstanceState` (unless shared is the point) | Instance-state adapters make the forward eager. |
| to train per request from activations during decode | `@pllm.train` returning `[rows]`, state with `optimizer=pllm.AdamW(lr)` | your own optimizer step in `@pllm.write` | pLLM masks, reduces, backpropagates, steps and publishes inside CUDA graphs on a side stream. |
| your own update rule (not a gradient step) | `@pllm.write(state=, fields=)` | in-place writes to parameters | Writes publish a new version atomically. |
| to train weights while the prompt streams through | `@pllm.train_in_forward(chunk=, ...)` | `@pllm.train` | Only it updates within the forward, so later chunks of the same prompt see the update. |
| to train on data the host assembles | `@pllm.train` reading `pllm.batch(...)`; `pllm.host.make_host` when only host code reads the state | request-row training | The fields read pick the batch type and optimizer. |
| whole-model LoRA training through the LM (TLM, GRPO) | `pllm.causal_model(adapter)` + `pllm.host.CausalTrainBatch`; `model.token_logprobs(...)` | materializing `[sequences, tokens, vocab]` logits | `token_logprobs` runs the LM head in blocks. |
| to push sampling away from a hidden prediction | `@pllm.guide_sampling(beta=...)` | `@pllm.add_logits` with your own LM-head product | pLLM applies the LM head only to surviving tokens and fuses it with the sampler. |
| guidance or logit edits on some rows only | `rows="selector"` on the sampler method | `torch.where` with a zero result | Unselected rows sample exactly as the plain model; a zero guidance prediction is not zero guidance. |
| the tokens before the current one | `pllm.input_tokens(history=n)` | accumulating tokens in state | History is gathered from vLLM's own token table. |
| to rerun or skip layers per token | `@pllm.custom_forward` (`active=` for early exit) | a custom residual | Only the custom forward schedules layer slices. |
| a different residual update or value rows | `@pllm.custom_residual` | `@pllm.custom_forward` | A custom residual keeps vLLM's model CUDA graphs. |
| a different attention read or KV layout | `@pllm.custom_attention` | reading K/V in a custom forward | It keeps vLLM's projections and KV writes and gives you paged K/V. |
| the drawn token for training | `pllm.sampled_tokens()` in `@pllm.train` next to an `@pllm.add_logits` | reading it in an export | Feedback is delivered only to that training ([rules.md](rules.md)); the host gets tokens from `RequestOutput`. |
| strict ordering of updates | `ExecutionOptions(publish_lag=0)` | the default `1` | Only when the algorithm needs it: training then no longer overlaps the next forward. |

## Workflow

### 1. Pin down the algorithm with the user

pLLM executes exactly the semantics written in the workload; a guess about an unclear requirement is
executed as written. Settle four things before writing code; an explicit "none" counts as an answer:

- **Math**: formula or reference code, shapes, initialization, hyperparameters; for training the loss, its
  reduction (per token, per request, per shared state), the optimizer and how often it steps.
- **State and lifetime**: what persists across steps; per request, per prompt (shared by its `n` samples),
  per route or per engine; when it resets; whether retries reuse it.
- **Timing**: which layer and token positions (prompt end, every decode token, inside the forward); before or
  after sampling; from which step an update must be visible.
- **Data flow**: where inputs come from (activations, sampled tokens, external labels, a judge) and where
  results go (observation, logits, sampling, state, retry); which requests or rows take part.

Ask further where answers are often vague. Training: which rows contribute to the forward, to input
gradients, to parameter gradients and to the update ("all rows, no input gradients" is a valid answer);
"no update" differs from "zero gradient" (AdamW still advances its step count, momentum and decay).
Replays and retries: how many, what stops them, which state version they read. Host actions: whether an
external action may run twice on retry. Also get the run constraints: model or checkpoint, GPU, dtype,
typical batch and sequence lengths.

Do not ask the user for things pLLM derives: model widths, memory layout, scheduling order, CUDA streams.
If the algorithm needs something pLLM cannot express, name the gap (which capability, which rule) and let
the user choose: reduce the requirement, change the algorithm, or extend pLLM. Never change the algorithm
silently; pLLM will faithfully run the changed version.

### 2. Write the workload

Put it in an importable `.py` module (needed for `ModelConfig(multiprocessing=True)` and tensor
parallelism). Start from the nearest example in [examples.md](examples.md) and the matching section of
[reference.md](reference.md). Make strengths and switches fields of a frozen dataclass, read through methods,
so a neutral setting (strength 0) exists for step 4. Write the plain correct version first; optimize
(custom kernels, host placement, lag) only after it checks out and only with the user's agreement.

### 3. Explain the plan on CPU

```python
print(pllm.explain(Distill(), CONFIG))
```

`pllm.explain` plans the workload on CPU from the model's `config.json`, without loading weights or touching
the GPU, and raises the same declaration errors as `make_llm`. Output for `Distill` (abridged):

```text
forward  CUDA graphs: no effect needs eager
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0); one AdamW step per state per engine step; prompt rows train once per prefill chunk
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
state  scope  bytes/root  reserved bytes [exact; state_page_budget_bytes]
distiller  request  96832  1807536  field 0: float32 x 24 pages, field 1: float32 x 16 pages, ...; source: module init; ...
memory   peak 60451428 bytes funded by pLLM [per-rank tensor reservations; ...]
preempt  state keep; replay rejection: not applicable to kept state
host     0 syncs in traced workload math
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

Read: `forward` says whether the model keeps CUDA graphs and, if not, which member makes it eager; each
`member` row is one piece of a method with its anchor (when), lane (`side` overlaps the next forward) and
whether it is captured; `state` is what pLLM reserves (pages per field: older versions keep only the fields readers read, so AdamW's
step and moments need fewer pages than the parameters); `memory` is pLLM's own peak (weights and KV
excluded); `preempt` says what happens when vLLM evicts a request ([pitfalls.md](pitfalls.md#preemption-and-recompute));
`host` counts GPU-to-CPU syncs in your math (must be 0); `timing` lists settings that change when values
become visible. Budget problems are in `report.budget_errors`. `explain` does not estimate throughput or
whether the engine fits on the GPU.

### 4. Check the contracts on CPU

```python
print(testing.check(Distill(), CONFIG, neutral=Distill(beta=0.0)))
```

```text
status  contract / reason
PASS    tensor_parallel_size=1: admission
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    publish_lag
PASS    preempted
PASS    neutral
SKIP    timing-dependent values: publish_lag counts the steps the workload runs: ...
```

`pllm.testing.check` runs three requests through fixed schedules on CPU with the real planner and executor
and compares their results across schedules. A failure raises `pllm.testing.ContractError` listing every
discrepancy; [pitfalls.md](pitfalls.md#contract-check-errors) has the fixes.

| Contract | It fails when |
| --- | --- |
| row permutation, request isolation | a request's result depends on where its row sits or on other requests' rows |
| mixed prefill/decode, chunking / tokens= | a result changes when prompts are split into chunks or mixed with decode rows |
| preempted | replaying a preempted request does not reproduce its state (SKIP names why replay is not allowed) |
| publish_lag | a method reads a state version other than the one `publish_lag` allows |
| neutral | `neutral=` (your workload with strength 0) still changes sampling: `@pllm.add_logits` must return exactly zero, `@pllm.guide_sampling` must have `beta=0` |

`SKIP` lines name what CPU cannot check (for example model-site execution); those need step 6.

### 5. Run it on the GPU

```python gpu
from vllm import SamplingParams

with pllm.make_llm(pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3),
                   workload=Distill()) as llm:
    outputs = llm.generate(["Explain gradient descent:"], SamplingParams(n=4, max_tokens=32, seed=0))
    report = llm.diagnostics()
print(repr(report.eager_reason), report.update_count, report.trained_rows)
```

`eager_reason` is empty when the model runs in CUDA graphs; `update_count` and `trained_rows` show that
training happened. Always use `with` (or `close()`): it releases the engine's GPU memory and compile state.

### 6. Check the contracts on the GPU

```python gpu
print(testing.check_on_gpu(Distill(), pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8,
                                                       gpu_memory_utilization=0.3), neutral=Distill(beta=0.0)))
```

It loads real engines: eight steady decode steps without host syncs, forced KV eviction and recompute, with
`neutral=` greedy and seeded sampling bitwise equal to the plain model, and for custom ops the GPU kernel
against its CPU reference. Run it in a fresh process.

`testing.check_distribution(workload, config, max_mean_kl=0.01, max_p99_kl=0.1)` bounds how far a workload
moves the plain model's full-vocabulary distribution under teacher forcing: both KL directions and TV (mean,
P99, max, worst positions), `passed`, `failures`. Omit thresholds to only measure. P99 catches rare broken
positions a mean hides; this measures distribution movement, not task quality.

## What to deliver and how to report

- The workload module and the smallest call that uses it (`make_llm(..., workload=...)`; for host loops the
  feedback callable passed to `generate_feedback`; for host-only training `pllm.host.make_host`).
- The semantic choices you made, the source of the reference algorithm, and what the user still has to decide.
- The checks that ran, where and in which Python environment: CPU (`explain`, `testing.check`) and GPU
  (`make_llm`, `check_on_gpu`, `check_distribution`) separately. A CPU pass is not a GPU pass.
- Settings and limits: CUDA graphs or eager (`eager_reason`), preemption policy, TP and model support,
  every expensive choice from [rules.md](rules.md#expensive-choices-you-write-explicitly) you made.
- Performance only from an equal-work comparison with plain vLLM (`pllm.benchmark.throughput` or
  `pllm.benchmark.compare`), with the settings measured.
