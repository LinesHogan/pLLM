---
name: pllm-workload
description: Write, inspect and check a pllm.Workload, an inference-time LLM algorithm that runs inside vLLM (activation steering, probes and SAE monitoring, logit edits such as DoLa, decode-time training such as ESamp, per-request adapters, residual rewrites, recurrent depth, host feedback loops such as TTRL). Use when asked to implement or port such an algorithm with pllm, or to fix a pllm rejection or ContractError.
---

# Writing a pllm workload

pllm runs inference-time algorithms inside vLLM (0.27.1 / 0.32.0) under continuous batching and CUDA
graphs. You declare what to read, what state to own and what the result does, and write the tensor math;
pllm handles per-request state and memory, CUDA graph capture, overlap with the forward, and optimizer
updates.

It fits algorithms that read or add a computation at a point of the forward (steering, probes and SAEs,
ReFT, logit edits, logit lens and DoLa, residual rewrites), keep state that evolves with tokens per request
or per engine (generated LoRA, decode-time training such as ESamp, test-time learning, fast weights), rerun
layers per token (recurrent depth, early exit), or control a generate-score-update loop (TTRL). Pretraining
from scratch, new operators and new model architectures are out of scope.

Read [reference.md](reference.md) for every name, grouped by task. The repository's README lists the
supported models and the tutorials (`doc/tutorials/`), one per algorithm family.

## How a workload is put together

A workload is a subclass of `pllm.Workload` holding only declarations and tensor math. pllm reads the
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
        module=lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size)),
        optimizer=pllm.AdamW(1e-3),
    )

    # What the result does: a loss to minimize (decorator); what it reads: markers as defaults.
    @pllm.train(state="distiller", reduction="state_mean")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)  # [rows]

    # A hidden-state prediction that guides sampling with strength beta.
    @pllm.guide_sampling(beta=pllm.Attr("beta"))
    def predict(self, shallow: torch.Tensor = pllm.hidden(0),
                f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow.float())


CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
```

The three kinds of declaration:

| Declares | Written as | Names |
| --- | --- | --- |
| what a method reads | a parameter default | `pllm.hidden`, `pllm.residual`, `pllm.final_hidden`, `pllm.layer_logits`, `pllm.parameter`, `pllm.module`, `pllm.version`, `pllm.index`, `pllm.sample_index`, `pllm.result`, `pllm.base_logits`, `pllm.sampled_tokens`, `pllm.batch`, `pllm.operand`, `pllm.causal_model`, `pllm.depth`, `pllm.value` |
| what the workload owns | a class attribute or a property annotated with the state type | `pllm.RequestState` (one copy per request), `pllm.InstanceState` (one shared copy), `pllm.PositionState`; optimizers `pllm.AdamW`, `pllm.SGD` |
| what a result does | a method decorator, or a model-site attribute | `@pllm.export`, `@pllm.add_logits`, `@pllm.guide_sampling`, `@pllm.train`, `@pllm.write`, `@pllm.train_in_forward`, `@pllm.custom_residual`, `@pllm.custom_forward`; `pllm.model.ActivationPatch`, `pllm.model.Adapter` |

Everything else is your tensor math on row tensors: dimension 0 holds one row per request (or per sampled
token), packed in an order you do not control.

## Workflow

### 1. Pin down the algorithm with the user

pllm executes exactly the semantics written in the workload; a guess about an unclear requirement is
executed as written. Settle four things with the user before writing code.

Ask until each is answered (an explicit "none" counts):

- **Math**: formula or reference code, shapes, initialization, hyperparameters; for training the loss, its
  reduction, the optimizer and how often it steps.
- **State and lifetime**: what persists across steps; per request, per prompt (shared by its `n` samples),
  per route or per engine; when it resets.
- **Timing**: which layer and token positions (prompt end, every decode token, inside the forward); before or
  after sampling; from which step an update must be visible.
- **Data flow**: where inputs come from (activations, sampled tokens, external tensors, a judge) and where
  results go (observation, logits, sampling, state, retry).

If the algorithm needs something pllm cannot express, name the gap and let the user choose; do not change
the algorithm silently.

### 2. Write the workload

Put it in an importable `.py` module (needed for `ModelConfig(multiprocessing=True)`). Find the task in
[reference.md](reference.md) and start from its example. Make strengths and switches fields of a frozen
dataclass so a neutral setting (strength 0) exists for step 4.

### 3. Explain the plan on CPU

```python
print(pllm.explain(Distill(), CONFIG))
```

`pllm.explain` plans the workload on CPU from the model's `config.json`, without loading weights or touching
the GPU. It shows whether the model keeps CUDA graphs, where each method runs and how much memory pllm
reserves. It does not estimate throughput or decide whether the engine fits on the GPU.

Output for `Distill` (abridged):

```text
forward  CUDA graphs: no effect needs eager
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0)
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
state  scope  roots  bytes/root  reserved bytes [exact]
distiller  request  34  129104  4389596  field 0: float32, field 1: float32, ...
host     0 syncs in traced workload math; exports: none
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

Read: `forward` says whether the model keeps CUDA graphs and, if not, which member makes it eager; each
`member` row is one piece of a method with its anchor (when), lane (`side` overlaps the next forward) and
whether it is captured; `state` is what pllm reserves; `host` counts GPU-to-CPU syncs in your math (should
be 0); `timing` lists the settings that change when values become visible. `explain` raises the same
admission errors as `make_llm`; budget problems are in `report.budget_errors`.

### 4. Check the contracts on CPU

```python
print(testing.check(Distill(), CONFIG, neutral=Distill(beta=0.0)))
```

```text
status  contract / reason
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    publish_lag
PASS    preempted
PASS    neutral
```

`pllm.testing.check` runs three requests through fixed schedules on CPU with the real planner and executor,
and compares their results across schedules. The table lists what each contract catches.

| Contract | It fails when |
| --- | --- |
| row permutation, request isolation | a request's result depends on where its row sits or on other requests' rows |
| mixed prefill/decode, chunking / tokens= | a result changes when prompts are split into chunks or mixed with decode rows |
| preempted | replaying a preempted request's history does not reproduce its state |
| publish_lag | a method reads a state version other than the one `publish_lag` allows |
| neutral | `neutral=` (your workload with its strength set to zero) still changes sampling: `@pllm.add_logits` must return exactly zero, `@pllm.guide_sampling` must have `beta=0` |

`SKIP` lines name what CPU cannot check (for example model-site execution); those need step 6. A failure
raises `pllm.testing.ContractError` listing every discrepancy.

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
training happened.

### 6. Check the contracts on the GPU

```python gpu
print(testing.check_on_gpu(Distill(), pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8,
                                                       gpu_memory_utilization=0.3), neutral=Distill(beta=0.0)))
```

It loads real engines: eight steady decode steps, forced KV eviction and recompute, and with `neutral=`
greedy and seeded sampling must equal the plain model bitwise. Run it in a fresh process.

When you report, say which checks ran on CPU and which on GPU, and in which Python environment. A CPU pass
is not a GPU pass.

## Rejections and how to fix them

Messages are quoted from pllm; `...` marks text that depends on your code.

| Message | Fix |
| --- | --- |
| `x: give every input a source as its default, e.g. ...` | Give every parameter after `self` a marker default: `x: torch.Tensor = pllm.hidden(0)`. |
| `the workload has no effect on the model: add @pllm.train, ...` | Add the effect that says where the result goes; a method without an effect decorator only runs as a `rows=` selector. |
| `a method takes one pllm decorator of each kind; split stacked effects into one method per effect` | Use one method per effect; share math through a helper function or `pllm.result(name)`. |
| `... both send a result to vLLM's sampler, which applies one @pllm.add_logits or @pllm.guide_sampling method per step; combine them into one method` | Keep one sampler method per workload. |
| `property 'x' needs a return annotation; ...` | Annotate state properties `-> pllm.RequestState` / `-> pllm.InstanceState`, other properties with their type. |
| `pllm.parameter('s', 'f') names no declared field; declared: ...` | Use a declared state and field name; a `shape=` state has the single field `"value"`, a `module=` state is read with `pllm.module(s)`. |
| `@pllm.train reads rows of running requests (no pllm.batch fields), which trains with pllm.AdamW(lr); declare 's' with optimizer=pllm.AdamW(lr)` | Put the optimizer the message names on the state. |
| `request training updates every selected state: remove rows= and update_rows= from @pllm.train` | Training on request rows updates every running request; select rows on the sampler method instead (`rows=` on `@pllm.guide_sampling`). |
| `... pllm funds one training mode per workload, so move one of them to a separate workload` | Do not mix request-row training and host-batch training in one workload. |
| `a task reads a value (sampled tokens, hidden states or another task's result) that exists only after its phase ends; declare it in a later phase` | `pllm.base_logits()` and `pllm.sampled_tokens()` exist only after sampling: read logits before sampling with `pllm.layer_logits((-1,))`, or move the method to `@pllm.train` / `@pllm.on(pllm.Phase.AFTER_SAMPLE)`. |
| `Tensor.item() cannot be called on meta tensors` with the note `methods run inside CUDA graphs: use tensor operations such as torch.where, not .item() or Python branches on tensor values` | Remove `.item()`, `.tolist()`, `print(tensor)` and `if tensor ...:` from methods; use masks and `torch.where`. |
| `... reads hidden rows (pllm.hidden, pllm.residual, pllm.layer_logits or pllm.final_hidden), which a @pllm.custom_forward does not produce: ...` | Move the reading method to a separate workload, or compute what you need inside the custom forward. |
| `active= currently needs @pllm.custom_forward(tokens="decode"); keep prompt layers unmasked` | Use `tokens="decode"` for early exit. |
| `activation patches run outside CUDA graphs; set ModelConfig(cuda_graphs="auto") or cuda_graphs=False` | The workload needs eager execution; do not force `cuda_graphs=True`. |
| `worker processes cannot send X defined in __main__ or a notebook; move it into an importable .py module, ...` | Move the class to a module, or keep `multiprocessing=False`. |
| ContractError `method 'm' mixes rows: equal inputs for this request produced different outputs. Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0).` | The method computes across rows (`mean(0)`, `sum(0)`, batch norm, sorting rows) or keeps Python state between calls (`self.count += 1`). Make each row depend only on its own inputs. |
| ContractError `pllm misrouted rows ... Report this message as a framework bug.` | Rule out your code first: in-place writes to inputs or parameters (`w.add_(1)`), or request-state training with `reduction="mean"` (its gradient depends on the batch; use `reduction="state_mean"`). |
| ContractError `[neutral] ...: add_logits must return exactly zero` / `[neutral] guide_sampling still declares a sampler stage; set its beta to zero.` | Make the strength setting multiply the whole result, and pass the workload with strength 0 as `neutral=`. |

## Escape hatches and what they cost

| Setting | Effect | Cost |
| --- | --- | --- |
| `ExecutionOptions(publish_lag=0)` | An update computed at step n is read from step n + 1 (default `1`: from n + 2). | Training no longer overlaps the next forward. |
| `@pllm.train_in_forward(visibility="exact")` | Later chunks of the same forward read the updates of earlier chunks (default `"forward_pinned"`: the whole forward reads the state it started with). | With `tokens="decode"` or `"all"` the forward runs eagerly. |
| `tokens=` on `pllm.hidden` / `pllm.residual` / `pllm.final_hidden` | `"prefill"` reads each prompt's last token once its prefill completes (default `"decode"`: the latest decode token). | None; it selects different rows. |
| `tokens=` on `@pllm.custom_forward` / `@pllm.train_in_forward` | Which tokens the method handles: `"prefill"`, `"decode"` or `"all"`. | Rows outside the selection run the plain model. |
| `@pllm.on(phase)` | Moves a method to another phase. | Overrides automatic timing; `explain` lists it. |
| `@pllm.custom_forward` | Your program over layer slices: loops, early exit, KV peeks. | The forward runs eagerly; no hidden-row readers or `pllm.model.Adapter` in the same workload; `tensor_parallel_size=1`. |
| `@pllm.custom_residual` | Your residual update at a sublayer, value rows or the final stream. | The forward runs eagerly; `tensor_parallel_size=1`; prefix caching off. |
| `pllm.model.ActivationPatch` | Edits layer outputs from a state. | The forward runs eagerly. |

`explain` prints every non-default setting in its `timing` lines.

## Anti-patterns

The patterns below either fail when the workload is built or, worse, run and give wrong results for some
requests; each item says what to do instead.

- **Mutating borrowed views.** `h.mul_(2)`, `w.add_(1)` or `out=` into an input. Inputs and parameters are
  shared with other methods and with the framework's state versions. `testing.check` does not always catch
  an input written in place; a written parameter shows up as "pllm misrouted rows". Return new tensors;
  change state only through `@pllm.write`, `@pllm.train` or imports.
- **Host syncs in methods.** `.item()`, `.tolist()`, `.cpu()`, `print(tensor)`, `float(tensor)`. Methods run
  inside CUDA graphs and on side streams; a sync stalls the engine or fails tracing. Use `@pllm.export` to
  see values.
- **Python control flow on tensor values.** `if h.norm() > 1:` or `while not converged:`. Use masks and
  `torch.where`. Python `for` loops over workload settings (`range(self.loops)`) are fine; they unroll.
- **Framework identity in the workload.** Request ids, slot numbers or step counters in Python attributes,
  dicts keyed by request, `self.x += 1`, hooks on vLLM modules, monkey-patching the engine. Per-request
  memory is a `pllm.RequestState`; progress is `pllm.version(state)`; observation is `@pllm.export`.
- **Computing across rows.** `mean(0)`, `softmax(dim=0)`, row sorting or batch statistics mix requests.
- **Request-state training with `reduction="mean"`.** The mean runs over every request's rows, so one
  request's update depends on its batch neighbours; use `reduction="state_mean"`.
- **Wrong shapes for sampler methods.** `@pllm.add_logits` must return `[rows, vocab]`, `@pllm.guide_sampling`
  `[rows, hidden]`. The CPU check does not verify the vocabulary width; on the GPU a wrong width fails with a
  bare `AssertionError` in the sampler.
