# Diagnosing a pLLM workload: messages, traps and known issues

Look up what went wrong here. Each entry says what you see, why it happens and what to change. The rules
behind these messages, with their reasons, are in [rules.md](rules.md); every public name is in
[reference.md](reference.md); the workflow is in [SKILL.md](SKILL.md).

Sections:

- [Error messages and fixes](#error-messages-and-fixes): raised by `make_llm`, `pllm.explain`,
  `pllm.testing.check` or a running engine, grouped by what they are about.
- [Contract check errors](#contract-check-errors): the failures `pllm.testing.check` reports as `ContractError`.
- [Anti-patterns](#anti-patterns): code that pLLM rejects late or that runs with wrong results.
- [Silent traps](#silent-traps-code-that-runs-but-does-something-else): code that runs without any message
  and does something other than what you meant.
- [Preemption and recompute](#preemption-and-recompute): what happens when vLLM evicts a request.
- [Shape, dtype and row conventions](#shape-dtype-and-row-conventions): what every input holds and every
  effect must return.
- [Known issues](#known-issues): framework behaviour that is wrong or unhelpful today, with workarounds.

The examples on this page run on CPU (`tools/check_docs.py` executes them) and share this setup:

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
```

## Error messages and fixes

Messages are quoted from pllm's source; `...` marks text that depends on your code (method, state and field
names, sizes). Most declaration messages carry a note `while declaring <Class>.<method>` naming the place.
`pllm.explain` raises the same declaration and engine-setting errors as `make_llm`, on CPU and without
weights, so run it first.

### Method inputs and effects

| Message | Why | Fix |
| --- | --- | --- |
| `...: give every input a source as its default, e.g. ...` | pLLM decides what a method reads from its parameter defaults; the annotation is only for you. | Give every parameter after `self` a marker default: `h: torch.Tensor = pllm.hidden(0)`. |
| `the workload has no effect on the model: add @pllm.train, ...` | Nothing schedules a method whose result goes nowhere. | Add the decorator that says where the result goes. An undecorated method only runs as a `rows=` selector or as a `pllm.result` producer for a decorated one. |
| `a method takes one pLLM decorator of each kind; split stacked effects into one method per effect` | One method has one destination. | One method per effect; share math through a module-level helper function or `pllm.result(name)`. `@pllm.on` and `@pllm.differentiate` may sit next to an effect decorator. |
| `... both declare @pllm.guide_sampling; keep at most one guidance method` | The sampler applies one guidance per step (components included). | Keep one `@pllm.guide_sampling`. Several `@pllm.add_logits` methods may coexist; their sum applies before guidance. |
| `... reads pllm.result(...)...; results can only flow one way within a step, remove one of the reads` | Results of one step form a graph without cycles. | Break the cycle; pass a value to the next step through state (`@pllm.write`) instead. |
| `tensor export needs positive every, max_rows and max_cols, with max_cols * 8 <= ... bytes (32 KiB), so one row fits an export event` | One export event carries whole rows of 8-byte values, at most 32 KiB. | Lower `max_cols` to at most 4096; split wider rows into several exports. |
| `property ... needs a return annotation; ...` | pLLM reads declarations from annotations without calling the property. | Annotate state properties `-> pllm.RequestState` / `-> pllm.InstanceState`, model-site properties `-> pllm.model.Adapter` / `-> pllm.model.ActivationPatch`, other properties with their type. |
| `pllm.parameter(..., ...) names no declared field; declared: ...` | The field name is not one the state declares. | Use a listed name. A `shape=` state has the single field `"value"`; an `init=pllm.FromModule(...)` state is read with `pllm.module(state)`. |
| `...: pllm.module needs a state declared with init=pllm.FromModule(...)` | Only a module state can be called like a module. | Declare `init=pllm.FromModule(lambda m: nn.Linear(...))`, or read fields with `pllm.parameter`. |

### Phases: when a value exists

| Message | Why | Fix |
| --- | --- | --- |
| `sampler method ... reads pllm...() through ..., which is available only in Phase.AFTER_SAMPLE.` | `pllm.base_logits()` and `pllm.sampled_tokens()` exist only after the sampler has drawn. | Before sampling read `pllm.layer_logits((-1,))` (the model's logits) or `pllm.final_hidden()`. |
| `...: sampled tokens are readable only beside @pllm.add_logits, by the train operation or by read-only operations that feed it` | Sampler feedback is delivered only to the training method of a workload that also adds logits (the OverRIDE pattern), and to read-only helpers feeding that training. | Add the `@pllm.add_logits` method the feedback trains, or take tokens from `RequestOutput` on the host. A training method that does not need the drawn token reads `pllm.final_hidden()` / `pllm.layer_logits((-1,))` instead. |
| `...: an export and the helpers it reads run read-only in Phase.AFTER_FORWARD on hidden rows, state and results; read the value from a helper without state writes or phase overrides` | Exports run after the forward; they cannot read sampler feedback and their helpers cannot write or move phase. | Export hidden rows, state or results only; sampled token ids are in each `RequestOutput`. |
| `a task reads a value (sampled tokens, hidden states or another task's result) that exists only after its phase ends; declare it in a later phase` | A `@pllm.on(phase)` placed the method before its input exists. | Remove the `@pllm.on`, or move it to a later phase. |
| `tensor inputs need one phase; hidden inputs become available after model forward` | `@pllm.on(Phase.BEFORE_FORWARD)` on a method that reads this step's hidden rows. | Remove the `@pllm.on`; effects have correct default phases. |

### State and training

| Message | Why | Fix |
| --- | --- | --- |
| `@pllm.train reads ..., which trains with ...; declare ... with optimizer=...` | The batch fields a training method reads pick the training mode, and each mode has one optimizer. | Put the optimizer the message names on the state: request rows and `WeightedRowBatch` / `CausalTrainBatch` use `pllm.AdamW(lr)`, `WeightedCEBatch` uses `pllm.SGD(lr)`. |
| `rows= and update_rows= on request training require a model effect without sampler guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state, so remove rows= and update_rows=` | Training next to guidance (or alone) updates every running request. | Remove `rows=` from `@pllm.train`; select rows on the sampler method instead (`@pllm.guide_sampling(rows="method")`). |
| `... pLLM funds one training mode per workload, so move one of them to a separate workload` | Request-row training and host-batch training reserve different memory and run at different times. | Put them in two workloads. |
| `host-batch training returns one scalar and owns its reduction; remove reduction=` | A host batch's loss is already one number. | Remove `reduction=`; reduce inside the method (`(weights * error).sum() / weights.sum().clamp_min(1)`). |
| `training on WeightedRowBatch or CausalTrainBatch fields takes one AdamW step over the whole batch: remove rows=, update_rows=, steps=, reset=, @pllm.differentiate and pllm.carry` | These batches always train as one whole batch. | Remove the listed settings. |
| `steps= and reset= apply to training on WeightedCEBatch fields (pllm.batch); remove them to train on request rows` | Repeated SGD steps from a reset point exist only for `WeightedCEBatch`. | Remove `steps=` and `reset=`. |
| `... and ... both write state ... from the same version; let one method write it` | Two writers of one state in one step would each overwrite the other's version. | One writer per state: one `@pllm.write` method (writing all its fields) or one `@pllm.train`, not both. |
| `...: add init=pllm.Computed(...) (e.g. random A, zero B); with both low_rank factors at zero every gradient is zero and the adapter never learns` | Gradients of `b @ a` vanish when both factors are zero. | Initialize A randomly and B at zero with `init=pllm.Computed(function)` (see reference.md, Train inside the forward). |
| `placement="host" needs a read-only InstanceState of gathered fields; use placement="device" for mutable state` | Host tables are read in bounded gathers that overlap the forward; they cannot be written. | Use `placement="host"` only for frozen pretrained tables of an `InstanceState`. |
| `...: gather host rows in a helper and read pllm.result in the model site; or use placement="device"` | A model site runs in the middle of a layer, where a PCIe read would stall the forward. | Gather in an undecorated helper that reads only `pllm.input_tokens` and the table; the model site reads `pllm.result("helper")`. |
| `...: a workload cannot hold a module or tensor; save its weights and declare ...` | A workload is a declaration hashed into its identity; tensors belong in state that pLLM places and pages. | Save the tensor to safetensors and declare state with `init=pllm.Pretrained(path)` or `init=pllm.FromModule(factory, pretrained=path)`, or import it with `init=pllm.Imported()`. |
| `...: workload methods may use only torch, math, pllm, triton, typing, collections.abc and builtins modules; ...` | Method code is frozen and hashed; other modules (numpy, your utilities) cannot be traced or shipped to workers. | Compute such values outside and pass them as workload fields (plain numbers, tuples, strings). |
| `...: workload methods may reference torch and pllm classes and the author's nn.Module subclasses, enums and dataclasses` | A method referenced a mutable global or class attribute (for example a counter). | Keep per-request memory in `pllm.RequestState`, progress in `pllm.version(state)`; settings in frozen dataclass fields. |

### Model sites and custom forwards

| Message | Why | Fix |
| --- | --- | --- |
| `... reads block or model outputs, which a @pllm.custom_forward does not produce; use pllm.residual or pllm.layer_logits to read each layer's last visited stream` | A custom forward may run a layer several times or not at all, so a layer's MLP output has no single value. | Read `pllm.residual(layer)` or `pllm.layer_logits(...)`, or compute what you need inside the custom forward. |
| `Use @pllm.custom_forward(tokens="decode") for active=: ...` | Skipping layers for prompt tokens would leave approximate K/V that every later token reads. | Use `tokens="decode"` for early exit; prompts run the full model. |
| `index a tap inside the Layers slice that produced it` | `layers(...)` returns outputs indexed by absolute layer; `out[-1]` is the model's last layer, which a partial slice did not run. | Index the slice's own layers: `layers(stop=12)[11]`. `[-1]` is right only when the slice runs to the end. |
| `layer ... already has a custom residual (...); one method owns a sublayer's residual update` | One method decides how a sublayer's stream updates. | Merge the two methods into one. |
| `... advances its memory once per token, but a @pllm.custom_forward may run its layer several times or not at all per token; drop state= or the custom forward` | A stateful sublayer needs exactly one visit per token. | Use two workloads, or carry the memory in `pllm.PositionState` inside the custom forward. |
| `static KV layouts cannot accompany PositionState or @pllm.custom_forward (including kv_copy)` | A layout splits the KV cache into several groups; these features need one group. | Drop the layout (`keep=`/`window=`/`source=`) or the custom forward. |
| `declare one @pllm.train_in_forward method per workload; ... already is one` | One method drives every projection hook of a workload. | Train all targets in one method (targets is a tuple). |
| `a @pllm.train_in_forward method drives every projection hook of its workload, so it does not yet combine with a pllm.model.Adapter; ...` | The training method owns the projection hooks the adapter would also use. | Train the adapter's state in the `@pllm.train_in_forward` method itself, or use two workloads. |
| `... reads RequestState ..., but @pllm.custom_attention runs for every scheduled token with instance-wide inputs; ...` | Attention reads have no per-request state. | Read an `InstanceState` and select per request inside the math. |
| `pllm.model.Adapter, pllm.model.ActivationPatch and @pllm.custom_forward act inside a model forward; compile this workload with make_llm, not make_host` | `make_host` compiles without a model. | Use `make_llm`, or keep only `pllm.batch` training for `make_host`. |
| `... must return the residual for ... as [rows, ...] in the input dtype or float32 on the input's device; got ...` | A `@pllm.train_in_forward` method without an optimizer returns the correction added to the projection's output (`pllm.UpdateOutput(output, ...)`, or `(next_weight, output)` with `chunk=`). pLLM runs every overlay method once at declaration on maximum-shape operands and checks the result there, not on every step. | Return `output` shaped `[rows, out_features]` of the target, in the input dtype or float32. |
| `... must return changed (bool) and update_count and trained_rows (int64) as 0-d tensors on the input's device` | The counters stay on the device so no step synchronizes with the host. | Build them as 0-d device tensors (`x.new_ones((), dtype=torch.bool)`), or return `pllm.UpdateOutput.unchanged(output)`. |
| `@pllm.add_logits method ... must return [rows, vocab_size] = ...; got ...` | The result is added to the full-vocabulary logits. | Return `[rows, vocab]`: read `pllm.layer_logits((-1,))` and return `logits[:, -1]`-shaped tensors, or project hidden states to the vocabulary. |
| `@pllm.guide_sampling method ... must return a hidden-state prediction [rows, hidden_size] = ...; got ...` | pLLM projects the prediction with the model's LM head and pushes sampling away from it. | Return `[rows, hidden_size]` (e.g. a prediction of `pllm.final_hidden()`); to add logits directly, use `@pllm.add_logits`. |

### Engine settings and models

| Message | Why | Fix |
| --- | --- | --- |
| `disable prefix caching (enable_prefix_caching=False) with pllm.model.Adapter, pllm.model.ActivationPatch, @pllm.train_in_forward, @pllm.custom_residual or @pllm.custom_attention: ...` | vLLM keys cached K/V by tokens, not by state, so a cached prefix would hold K/V computed under another request's state. | Keep `enable_prefix_caching=False` (the default). |
| `...; set ModelConfig(cuda_graphs="auto") or cuda_graphs=False` | `cuda_graphs=True` demands vLLM's whole-model graphs, and the named feature runs the forward without them. | Use `cuda_graphs="auto"` (the default); `diagnostics().eager_reason` then names the cause. |
| `@pllm.custom_attention needs tensor_parallel_size=1: each rank sees only its heads` | Your read function sees all heads at once. | Use `tensor_parallel_size=1`. |
| `multi-GPU needs vLLM 0.32.0 and runner="mrv2"; otherwise use tensor_parallel_size=1` | Tensor parallelism exists only on the MRv2 runner of vLLM 0.32.0. | Use the 0.32.0 environment with the default `runner="mrv2"`, or one GPU. |
| `... runs on one GPU only; set tensor_parallel_size=1` | `causal_model` training, state feeds and `@pllm.custom_residual` run on one GPU only. | Use `tensor_parallel_size=1`. |
| `TP Adapter fields start from pllm.Zeros or pllm.Computed and take shard-local @pllm.train_in_forward updates; optimizers and other init= sources need tensor_parallel_size=1` | Under TP each rank holds its shard of the delta; only shard-local updates keep shards consistent. | With TP, start adapter state from `Zeros`/`Computed` and update it only in `@pllm.train_in_forward`. |
| `TP projection update is not shard-local at ...; ...` | The update math needs other ranks' shards (for example training replicated A from column-sharded outputs). | Rewrite the update to use only the local shard, or use `tensor_parallel_size=1`. |
| `pllm.input_tokens needs runner="mrv2"` | Token history is gathered by the MRv2 runner. | Keep the default `runner="mrv2"`. |
| `choose a Qwen2, Llama, Qwen3, Qwen3Moe, Qwen3_5, Qwen3_5Moe, DeepseekV2, DeepseekV3 or GptOss ForCausalLM model for vLLM workloads` | pLLM admits exact vLLM model classes whose layer structure it knows. | Use a listed family (see [rules.md, Support matrix](rules.md#support-matrix)). |
| `... needs a Qwen2, Llama, Qwen3, Qwen3-MoE or Qwen3.5 model (...); hidden taps, ESamp, sampler stages and activation patches still run on this model` | DeepSeek V2/V3 and gpt-oss have MLA or MoE projections pLLM cannot rewrite. | Use read-only features on these models, or a dense family. |
| `worker processes cannot send ... defined in __main__ or a notebook; move it into an importable .py module, ...` | Workers rebuild the workload by importing its class. | Put the class in a module, or keep `multiprocessing=False` and `tensor_parallel_size=1`. |
| `check_on_gpu needs max_num_seqs >= 2 and max_model_len >= 32 for its fixed script` | The GPU check runs a fixed multi-request script. | Raise those settings for the check. |
| `pLLM execution needs ... bytes at its peak but only ... fit; lower ...` | The planned execution peak exceeds `execution_peak_budget_bytes`. Reported in `pllm.explain(...).budget_errors` and raised by `make_llm`. | Follow the message: lower `max_num_batched_tokens` or `max_num_seqs`, raise the budget, or declare `placement="host"` for read-only tables. |

### At run time

| Message | Why | Fix |
| --- | --- | --- |
| `vLLM preempted a request and would recompute its history; ...` | The workload changes the model's computation from state that a replay cannot rebuild (see [Preemption and recompute](#preemption-and-recompute)). | Give vLLM more KV cache (`gpu_memory_utilization`) or fewer concurrent sequences (`max_num_seqs`), or remove what the message names. |
| `state feeds need tensor_parallel_size=1 and multiprocessing=False; pass the writer to a spawned producer` | `llm.feed(...)` hands out an in-process handle. | Use one in-process engine; run the producer in a spawned process. |
| `batch, sample_index, prompt_indices and routes need workload= in make_llm` | A plain engine has nothing to feed. | Pass `workload=` to `make_llm`. |
| `completions carry each sample's final hidden row, which only a workload collects; ...` | `generate_completions` and `generate_feedback` hand each sample's final hidden row to the host, and only an installed workload's runner captures it. | Pass `workload=pllm.workloads.DummyWorkload()` to `make_llm` when nothing should train. |
| PyTorch's "Tensor.item() cannot be called on meta tensors" with the note `while shape tracing method ...; methods run inside CUDA graphs: use tensor operations such as torch.where, not .item() or Python branches on tensor values` | pLLM traces methods on shape-only tensors to plan memory and capture graphs; host reads have no value there and would stall the engine. | Remove `.item()`, `.tolist()`, `.cpu()`, `float(tensor)`, `print(tensor)` and `if tensor ...:`; use masks and `torch.where`. |

## Contract check errors

`pllm.testing.check` raises `pllm.testing.ContractError`; its `failures` list every discrepancy with a
`[contract]` prefix and the request, token and method involved.

| Message | Why | Fix |
| --- | --- | --- |
| `method ... mixes rows: equal inputs for this request produced different outputs. Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0).` | The method computes across rows (`mean(0)`, `softmax(dim=0)`, batch norm, sorting rows) so one request's result depends on others. | Make every row depend only on its own inputs. |
| `pLLM misrouted rows before ...: its per-request inputs already differ.` | Usually your code: tensor math that mixes different states' rows, or an index computed from `pllm.index` used across states. | Rule out your code first; if the method is row-local, report the message as a framework bug. |
| `method ... mutates borrowed input ... in place; inputs and parameters are read-only views.` | Inputs and parameters are shared with other methods and with pLLM's state versions. | Use out-of-place math (`h * 2`, not `h.mul_(2)`), or `clone()` first; change state only through `@pllm.write` or `@pllm.train`. |
| `[neutral] ...: add_logits must return exactly zero` | With strength 0 the workload still changes the logits. | Multiply the whole result by the strength setting and pass the strength-0 workload as `neutral=`. |
| `[neutral] guide_sampling still declares a sampler stage; set its beta to zero.` | The neutral workload still guides. Often the [setting trap](#silent-traps-code-that-runs-but-does-something-else): `beta=beta` reads the class default. | Make `beta` a method of the workload (`beta=strength`). |

## Anti-patterns

Each of these either fails when the workload is built or, worse, runs and gives wrong results for some
requests.

- **Mutating borrowed views.** `h.mul_(2)`, `w.add_(1)`, `out=` into an input. Why wrong: inputs and
  parameters are views shared with other methods and with pLLM's state versions. `testing.check` reports
  `mutates borrowed input`. Instead return new tensors; change state only through `@pllm.write`,
  `@pllm.train` or imports.
- **Host syncs in methods.** `.item()`, `.tolist()`, `.cpu()`, `print(tensor)`, `float(tensor)`. Why
  wrong: methods run inside CUDA graphs and on side streams; a sync stalls the engine or fails tracing.
  Instead `@pllm.export` the value and read it in `on_export`.
- **Python control flow on tensor values.** `if h.norm() > 1:`, `while not converged:`. Why wrong: the
  method is traced once and replayed; a branch is frozen at trace time. Instead use masks and `torch.where`;
  Python `for` loops over workload settings (`range(self.loops)`) are fine because they unroll.
- **Framework identity in the workload.** Request ids, slot numbers or step counters in Python attributes,
  dicts keyed by request, `self.x += 1`, hooks on vLLM modules, monkey-patching the engine. Why wrong:
  methods are frozen copies that run on packed rows in an order you do not control, possibly in another
  process. Instead per-request memory is a `pllm.RequestState`, progress is `pllm.version(state)`,
  observation is `@pllm.export`.
- **Computing across rows.** `mean(0)`, `softmax(dim=0)`, row sorting, batch statistics. Why wrong:
  dimension 0 packs unrelated requests. Instead reduce along `dim=-1`.
- **Reducing request-row losses yourself.** `loss.mean()` in a `@pllm.train` method on request rows. Why
  wrong: pLLM masks padding and reduces per state; a scalar loses that. Instead return `[rows]`; the
  default `reduction="mean"` averages each state's selected rows, `reduction="sum"` sums them. Host-batch
  methods return one scalar and own their reduction.
- **Wrong shapes for sampler methods.** `@pllm.add_logits` must return `[rows, vocab]` (checked when the
  workload is built); `@pllm.guide_sampling` returns `[rows, hidden]`.
- **Reading `pllm.base_logits()` to adjust logits.** It exists only after sampling. Read
  `pllm.layer_logits((-1,))` (float32 `[rows, 1, vocab]`) before sampling.
- **Copying a built-in workload's imports.** Some files in `pllm/workloads/` use internal modules
  (`pllm.execution`, `pllm.contracts`) for optimizations. Why wrong: internals change without notice and
  bypass the checks public declarations get. Instead copy only the public declarations (see
  [examples.md](examples.md)).
- **Changing the algorithm to fit pLLM.** Dropping a term, changing a reduction or an update schedule
  because a declaration is awkward. Why wrong: pLLM executes exactly what you write; a silent change is a
  different algorithm. Instead name the gap and let the user choose.

## Silent traps: code that runs but does something else

### A dataclass field named in the class body passes the class default

Inside a dataclass body, a field name is an ordinary class variable holding the field's default. Passing it
to a decorator or marker (`max_rows=rows`, `beta=beta`, `pllm.hidden(layer)`) passes that default, once,
when the class is defined; instances built with another value are ignored without any message. Settings
that depend on the instance must be functions of the workload, usually a method defined above the
decorator:

```python
@dataclass(frozen=True)
class Wrong(pllm.Workload):
    rows: int = 2

    @pllm.export(name="x", max_rows=rows, max_cols=1)  # reads the class default 2
    def f(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return h[:, :1].float()


@dataclass(frozen=True)
class Right(pllm.Workload):
    rows: int = 2

    def row_limit(self) -> int:
        return self.rows

    @pllm.export(name="x", max_rows=row_limit, max_cols=1)  # reads the instance
    def f(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return h[:, :1].float()


print([export.rows for export in pllm.explain(Wrong(rows=6), CONFIG).exports])  # [2]
print([export.rows for export in pllm.explain(Right(rows=6), CONFIG).exports])  # [6]
```

The same holds for `beta=`, `layers=`, `targets=`, `chunk=` and the arguments of `pllm.hidden`,
`pllm.residual`, `pllm.layer_logits`, `pllm.sample_index` and `pllm.value`. `lambda self: self.beta` also
works; a method is preferred because the IDE follows it and pyright checks its return type. Tensor math
inside a method body reads `self.beta` normally.

### Other quiet surprises

- **`pllm.hidden(layer)` is not the residual stream.** It is the layer's MLP output before it is added.
  Steering vectors, SAEs, probes trained on Hugging Face `hidden_states` and the logit lens read
  `pllm.residual(layer)` (Hugging Face `hidden_states[layer + 1]`).
- **Default rows are decode rows.** `tokens="decode"` gives one row per running request: its latest
  decode token. Prompt tokens are not seen; read the prompt's last token with `tokens="prefill"`, every
  computed token with `tokens="all"` (exports only).
- **Exports drop rows beyond `max_rows` per step.** Prompts that finish prefill in the same step share one
  step's `max_rows`; the rest are counted in `event.dropped_rows`. Pass at most `max_rows` prompts per
  `generate` call when every prompt-end row matters.
- **Updates become visible two steps later.** With the default `publish_lag=1`, an update computed at step n
  is read from step n + 2; the training method itself reads the newest version. Use
  `ExecutionOptions(publish_lag=0)` only when the algorithm needs it (training then stops overlapping the
  next forward).
- **A neutral run is not bitwise plain vLLM unless the graph modes match.** Rewrites inside the model graph
  (`@pllm.custom_residual`, `ActivationPatch`) make vLLM capture source operators instead of
  inductor-fused ones; compare against a matching vanilla engine (`check_on_gpu` builds one).
- **`pllm.Field` shapes receive projection widths only for Adapter states.** For other states the callable
  gets `(hidden_size, hidden_size)`; `lambda width, _: (width,)` is the usual hidden-width vector.
- **`pllm.ModelCheckpoint` takes its tensors away from the model.** vLLM skips every listed tensor when it
  loads the weights, so the model runs without them and nothing reports it. List only tensors the model
  does not use itself; never list one it needs, such as `model.embed_tokens.weight` or a layer's weights.
- **Instance state read by an Adapter makes the forward eager.** `diagnostics().eager_reason` reports
  `CUDA-graph overlays do not support adapters backed by shared InstanceState`; per-request adapter state
  stays in CUDA graphs.

## Preemption and recompute

When vLLM runs out of KV cache it preempts a request, frees its KV and later recomputes its whole history.
pLLM chooses one of two policies per workload; `pllm.explain` prints it as the `preempt` line and
`testing.check` exercises it (contract `preempted`).

| Policy | Workloads | What happens on preemption |
| --- | --- | --- |
| `keep` | Effects that do not change the model's computation: hidden reads, exports, `@pllm.write`, `@pllm.train`, `@pllm.add_logits`, `@pllm.guide_sampling`, custom forwards without `active=`. Also `pllm.model.Adapter` and `pllm.model.ActivationPatch` on state no method writes or trains (frozen, or `init=pllm.Imported()`). | Request state is kept; the recompute reads the state version the request was admitted with, so its KV matches. |
| `reset` | Effects that change K/V or the residual stream from state that changes while requests run: `pllm.model.Adapter` and `pllm.model.ActivationPatch` on written or trained state, `@pllm.custom_residual`, `@pllm.custom_attention`, `@pllm.custom_forward` with `active=`, `@pllm.train_in_forward`. | pLLM retires the request's state and replays the request from admission. This is only correct when replay rebuilds the same state, so pLLM checks eligibility when it plans. |

A `reset` workload is replay-eligible when all of these hold; otherwise explain prints the first violated
rule as `replay rejection`, `testing.check` SKIPs `preempted` with that reason, and a real preemption raises
`vLLM preempted a request and would recompute its history; ...`:

| Rule (as explain prints it) | Why replay needs it |
| --- | --- |
| use request-owned keys containing request_id and remove mutable instance state | Shared or mutable state may have moved on since the request started. |
| remove external feeds, imports and host inputs from the request state computation | Imported values and host batches are not stored for replay. |
| remove sampler feedback dependencies | Replay does not redraw tokens. |
| remove reads of pllm.version and pllm.index | A state's version and pool slot differ between the first run and the replay. |
| use over="token" in @pllm.custom_forward instead of over="span" | Span programs depend on how the prompt was chunked. |
| use visibility="exact" with zero publish lag and no cross-step dependencies | Lagged reads depend on step timing, which replay does not reproduce. |

Size the engine so a rejected workload is not preempted (raise `gpu_memory_utilization`, lower
`max_num_seqs` or `max_model_len`), and say so to the user. `diagnostics().kv_cache_max_concurrency` tells
how many full-length requests fit. A steering workload whose vector is `init=pllm.Imported()` (tutorial 00)
is `keep` and survives preemption.

## Shape, dtype and row conventions

Rows: dimension 0 of every row tensor holds one row per request (per token for the sampler and for
`tokens="all"`), packed in an order you do not control. Measured with `pllm.explain` on Qwen2.5-0.5B
(hidden 896, vocab 151936) and `max_num_seqs=8`.

| Input | Shape | Dtype | Rows |
| --- | --- | --- | --- |
| `pllm.hidden(l)`, `pllm.residual(l)`, `pllm.final_hidden()` | `[rows, hidden]` | float32 in tensor methods (taps are promoted); model dtype inside `@pllm.custom_residual` | `tokens="decode"`: latest decode token per running request; `"prefill"`: last prompt token of requests whose prefill completed this step; `"all"`: every token computed this step (exports only) |
| `pllm.layer_logits(layers)` | `[rows, len(layers), vocab]` | float32 | one per token the sampler draws: a prompt's last token and each decode token |
| `pllm.base_logits()` | `[rows, vocab]` | float32 | sampled rows, after sampling only |
| `pllm.sampled_tokens()` | `[rows]` | int64 | sampled rows, after sampling only |
| `pllm.input_tokens(history=n)` | `[rows, n + 1]` | int64, `-1` before position 0 | current token first, then lookbacks |
| `pllm.version(s)`, `pllm.index(s)`, `pllm.sample_index()` | `[rows]` | int64 | the reading method's rows |
| `pllm.parameter(request_state, f)` | `[rows, *field_shape]` | storage dtype | each row its own request's copy |
| `pllm.parameter(instance_state, f)` | `field_shape` | storage dtype | no row axis |
| `pllm.parameter(instance_state, f, per_row=True)` | `[rows, *field_shape]` | storage dtype | broadcast view of the shared field |
| `pllm.batch(field)` | as passed to `generate(batch=...)` | as passed | host batch rows |
| site tensors of `@pllm.custom_residual` | `[rows, hidden]`, `"value"`: `[rows, kv_heads, head_dim]`, `streams=n`: `[rows, n, hidden]` | model dtype | every scheduled token |
| `q` of `@pllm.custom_attention` | `[rows, heads, head_dim]` (after RoPE) | model dtype | every scheduled token |

Storage dtype `"auto"` is float32 for states with an optimizer and for fields a method rewrites from their
old value; otherwise the model dtype. Give `dtype=torch.float32` (state or `pllm.Field`) for precision. A
model-dtype parameter times a float32 hidden row promotes to float32; model sites must return the site's
dtype (`.to(x.dtype)`).

Measured: with a bfloat16 Qwen2.5-0.5B, `pllm.hidden`, `pllm.residual` and `pllm.final_hidden` reach an
`@pllm.export` method as float32 on the GPU and in `pllm.explain`, while the site tensors and `pllm.residual`
inputs of a `@pllm.custom_residual` method are bfloat16.

| Effect | Must return |
| --- | --- |
| `@pllm.export` | `[rows, cols]`; columns beyond `max_cols` are cut; `event.tensor()` keeps the dtype |
| `@pllm.add_logits` | `[rows, vocab]` floating; added before temperature, top-k and top-p |
| `@pllm.guide_sampling` | `[rows, hidden]`; pLLM applies the LM head |
| `@pllm.train` on request rows | `[rows]` per-row loss (or `pllm.UpdateOutput`) |
| `@pllm.train` on a host batch | a scalar |
| `@pllm.write(fields=...)` | a tuple with one tensor per named field, request fields with the row axis |
| `@pllm.custom_residual` | the site's shape and dtype (stateful sublayers: `(stream, next_carry)`) |
| `@pllm.custom_attention` | q's shape and dtype, optionally `(attention, aux)` with aux `[rows, ...]` |
| `@pllm.custom_forward` | the residual stream `[rows, hidden]` the final norm reads |
| `@pllm.train_in_forward` | with `chunk=` and an SGD state: a `[rows]` loss; with `chunk=` and no optimizer: `(next_weight, output)`, the chunk's new weight field and the `[rows, out_features]` correction (see `pllm/workloads/inplace_ttt.py`); without `chunk=`: `pllm.UpdateOutput(output, changed, update_count, trained_rows)` with 0-d device counters |

## Known issues

Current framework behaviour that is wrong or unhelpful; each has a workaround. None is open at the moment.
