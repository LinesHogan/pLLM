# pLLM reference for coding agents

Every public name in `pllm`, `pllm.model`, `pllm.host`, `pllm.io`, `pllm.testing` and `pllm.benchmark`, grouped by the task you
are doing. Each section lists the names with a one-line meaning and a minimal example; the full signatures,
generated from the code, are at the end ([All signatures](#all-signatures)). The workflow and which declaration
fits which need are in [SKILL.md](SKILL.md); error messages, traps, preemption and the full shape and dtype
table in [pitfalls.md](pitfalls.md); invariants, refused combinations, the support matrix and the expensive
choices in [rules.md](rules.md); the nearest example per task in [examples.md](examples.md). Where a
hand-written default below differs from the generated signature, the signature is right.

Every Python block in this file runs: blocks fenced `python` on CPU in order, blocks fenced `python gpu` on a
GPU (`tools/check_docs.py --gpu`). They share this setup:

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
```

**Conventions used everywhere.**

- A method's inputs are declared by parameter defaults (`h: torch.Tensor = pllm.hidden(0)`); every parameter
  after `self` needs one. The annotation only says what the body receives.
- Row tensors have one row per request on dimension 0 (for `pllm.layer_logits` and sampler methods: one row
  per token the sampler draws). Rows of different requests are packed in an order you do not control:
  compute along the feature dimension (`dim=-1`), never across rows (`dim=0`).
- Hidden rows (`pllm.hidden`, `pllm.residual`, `pllm.final_hidden`) reach tensor methods as float32, and
  `pllm.layer_logits` is float32; tensors inside model sites (`@pllm.custom_residual`, `@pllm.custom_attention`)
  and state fields without an optimizer are in the model dtype (bfloat16 by default). Cast explicitly where
  the dtype matters (`.float()`, `.to(x.dtype)`). [pitfalls.md](pitfalls.md#shape-dtype-and-row-conventions) lists the shape
  and dtype of every input and the return shape of every effect.
- Inputs are read-only borrowed views. Return new tensors; change state only through the effects below.
- A setting that depends on the instance is a method of the workload passed by name (`beta=strength`), never a
  dataclass field named in the class body, which passes the class default.

## Read a hidden state, the residual stream or per-layer logits

| Name | Meaning |
| --- | --- |
| `pllm.hidden(layer=0, *, tokens="decode")` | The MLP output of decoder layer `layer` before it is added to the residual stream, `[rows, hidden]`. Negative layers count from the end. |
| `pllm.residual(layer, *, after="layer", tokens="decode")` | The residual stream after layer `layer` (Hugging Face `hidden_states[layer + 1]`), `[rows, hidden]`. `after="attention"` reads it between the layer's attention and its MLP. Steering vectors, SAEs and the logit lens read this. |
| `pllm.final_hidden(*, tokens="decode")` | The hidden state after the final norm, which the LM head reads (Hugging Face `last_hidden_state`), `[rows, hidden]`. |
| `pllm.layer_logits(layers)` | The final norm and LM head applied to the residual stream after each of `layers`: float32 `[rows, len(layers), vocab]` (the logit lens). Layer `-1` gives the model's own logits. One row per token the sampler draws: a prompt's last token and every decode token. |

`tokens="decode"` gives one row per running request: its latest decode token. `tokens="prefill"` gives one
row per request whose prompt just finished prefilling: its last prompt token. `tokens="all"` gives one row per
token the step computes fresh, prompt chunks included; only `@pllm.export` methods (without `pllm.result`
inputs) may read these rows, and each event lists the rows' token positions. All four inputs run inside
CUDA graphs. A reading method needs an effect (usually `@pllm.export`, next section) or nothing schedules it.

`pllm.input_tokens(history=n)` reads int64 `[rows, n + 1]` current and preceding request token IDs, with `-1` before position zero. Helpers consumed by `@pllm.custom_residual` use every scheduled row through `pllm.result`; other readers use sampled rows.

```python
class Lens(pllm.Workload):
    @pllm.export(name="lens", max_rows=8, max_cols=3)
    def top1(self, logits: torch.Tensor = pllm.layer_logits((6, 12, -1))) -> torch.Tensor:
        return logits.argmax(dim=-1).float()  # [rows, 3]: the most likely token at layers 6, 12 and the last

    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)  # [rows, 1]


print(testing.check(Lens(), CONFIG).passed)
```

## Export observations to the host

| Name | Meaning |
| --- | --- |
| `@pllm.export(*, name="value", every=1, max_rows=1, max_cols=16, enabled=True, overflow="drop")` | Deliver the method's `[rows, cols]` result to `on_export`: per step at most `max_rows` rows (the rest of that step's rows are dropped and counted in `event.dropped_rows`) and the first `max_cols` columns, every `every` steps, split into events of whole rows of at most 32 KiB each. `max_cols * 8` must be at most 32768 bytes, so one row fits an event. Runs after the forward by default. `overflow="block"` waits on a full delivery queue (256 events) instead of dropping the event. `pllm.explain` prints each export's rows, bytes and events per step. |
| `pllm.Workload.on_export(self, event)` | Override to receive each export as a `pllm.io.TraceEvent`, in the process that runs the model, asynchronously; `close()` (or leaving `with`) drains delivery. Never call the engine from it. |
| `pllm.io.TraceEvent` | One trace record. For exports: `tensor_name` (the export name), `identities` (one `pllm.testing.RequestIdentity` per row: `prompt_index` is the prompt's position in the `generate` call, or the value from `generate(..., prompt_indices=...)`; `sample_index`; `request_id` is pLLM's internal id, not vLLM's `RequestOutput.request_id`), `positions` (each row's token position in its request), `step_id`, `dropped_rows`, `tensor()` (an independent CPU tensor). |
| `pllm.io.TraceSnapshot` | `llm.trace_snapshot()`: the retained events (`events`), drop counters (`export_dropped`: events dropped on a full queue) and host/device task timings. |
| `pllm.io.TraceSelection` | `llm.select_trace(TraceSelection(...))` opts into extra tensor/gradient snapshots and kernel profiling; empty filters match everything. |

```python
from collections import defaultdict

seen: dict[int, list[float]] = defaultdict(list)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        for identity, row in zip(event.identities, event.tensor().tolist()):
            seen[identity.prompt_index].append(row[0])
```

```python gpu
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)
GREEDY = SamplingParams(max_tokens=8, temperature=0.0)

with pllm.make_llm(GPU_CONFIG, workload=StreamNorm()) as llm:
    llm.generate(["The capital of France is", "One plus one is"], GREEDY)
    assert llm.diagnostics().eager_reason == ""  # reading the residual stream keeps CUDA graphs
print({prompt: len(norms) for prompt, norms in seen.items()})  # 7: the first of 8 tokens comes from the prompt
```

A full hidden-width row is 896 columns for Qwen2.5-0.5B (7168 bytes), so four rows share one 32 KiB event and
a step with more rows arrives as several events. To capture one full row per prompt at the prompt's last token,
read with `tokens="prefill"` and pass at most `max_rows` prompts per `generate` call: prompts that finish prefill
in the same step share that step's `max_rows`.

```python
prompt_ends: dict[int, torch.Tensor] = {}


class PromptEnd(pllm.Workload):
    @pllm.export(name="prompt_end", max_rows=4, max_cols=896)
    def last_token(self, r: torch.Tensor = pllm.residual(12, tokens="prefill")) -> torch.Tensor:
        return r.float()  # [rows, 896]: one row per prompt whose prefill just finished

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        assert event.dropped_rows == 0
        for identity, row in zip(event.identities, event.tensor()):
            prompt_ends[identity.prompt_index] = row
```

```python gpu
texts = ["I love this", "I hate this", "This is wonderful", "This is terrible", "I feel happy", "I feel sad"]
with pllm.make_llm(GPU_CONFIG, workload=PromptEnd()) as llm:
    for start in range(0, len(texts), 4):  # at most max_rows prompts per call
        chunk = texts[start : start + 4]
        # prompt_indices numbers the prompts across calls; identity.prompt_index reports it back.
        llm.generate(chunk, SamplingParams(max_tokens=1), prompt_indices=tuple(range(start, start + len(chunk))))
# Leaving `with` drained on_export, so every row has arrived.
rows = torch.stack([prompt_ends[index] for index in range(len(texts))])
print(rows.shape)  # torch.Size([6, 896])
```

## Own state per request or per instance

| Name | Meaning |
| --- | --- |
| `pllm.RequestState(fields=..., shape=None, dtype="auto", key_fields=("request_id",), *, init=pllm.Zeros(), optimizer=None, carry=..., ...)` | Tensors that every request owns a private copy of, admitted with the request and freed when it finishes. Read as row-aligned tensors: a field of shape `S` reads as `[rows, *S]`. |
| `pllm.InstanceState(...)` | Tensors shared by all requests of the engine; same arguments minus `key_fields`. A field reads with its declared shape, without a row axis. |
| `pllm.Field(shape, dtype="auto")` | A field whose shape is `lambda in_features, out_features: (...)`: the widths of each target projection when a `pllm.model.Adapter` reads the state, otherwise `(hidden_size, hidden_size)`. Also gives one field an explicit `dtype`. |
| `pllm.Dimensions` | What a plain shape callable receives: `hidden_size`, `vocab_size`, `layer_count`, `target_shapes`, `dtype`. Write `{"bias": lambda m: (m.vocab_size,)}`. |
| `pllm.parameter(state, field, *, per_row=False)` | Read one field. `per_row=True` gives an instance field a broadcast row axis for row-wise math. |
| `pllm.module(state)` | Read a state declared with `init=pllm.FromModule(lambda m: nn.Module(...))` as a callable module; for request state each row runs with its own parameters. |
| `pllm.version(state)` | int64 `[rows]`: how many updates each row's state has received (0 before the first). |
| `pllm.index(state)` | int64 `[rows]`: each row's slot in the state pool. |
| `pllm.PositionState(initial=0.0)` | One hidden-width vector per KV cache block, read and written only by `layers.load` / `layers.save` inside `@pllm.custom_forward`. |

Declaring fields: `shape=(n,)` declares one field named `"value"`; `fields={"w": shape, ...}` declares named
fields, each shape a tuple, a `pllm.Dimensions` callable or a `pllm.Field`; `init=pllm.FromModule(...)` takes
the fields from the module's parameters. Initial values come from the one `init=` source (every request starts
from the same values):

| `init=` | Starting values |
| --- | --- |
| `pllm.Zeros()` | Every field at zero (the default). |
| `pllm.Computed(function)` | `function(fields, generator)` fills the named field tensors in place, once, with a seeded generator. |
| `pllm.Pretrained(path, placement="device")` | Tensors named like the fields from a safetensors file; the file's hash joins the workload's identity. `placement="host"` keeps a read-only table that methods only gather rows from in pinned host memory; `"device"` never falls back to host. |
| `pllm.ModelCheckpoint({field: tensor_name})` | The served model's checkpoint tensors (with `layers=(...)` the names may contain `{layer}`); `InstanceState` without optimizer, read-only. vLLM skips the listed tensors when it loads the model, so list only tensors the model class does not use (e.g. In-Place TTT's `ttt_conv` and `ttt_proj`), never one it needs such as `model.embed_tokens.weight`. |
| `pllm.FromModule(factory, pretrained="")` | The parameters of `factory(dimensions)`, an `nn.Module`, as the state's one field, initialized as the module does or from a state_dict file. |
| `pllm.Fed("ordered" or "latest")` | Published by a producer process through `llm.feed(state)`; one per workload. |
| `pllm.Imported()` | Passed to `generate(batch=pllm.io.StateImport(...))`. |

 Storage dtype `"auto"`: float32 when the state has an optimizer or a method rewrites a
field from its old value, the model dtype otherwise. Declare state as a class attribute, or as a property
annotated `-> pllm.RequestState` / `-> pllm.InstanceState` when its shape depends on workload fields.

`key_fields` decides which requests share one copy: `("request_id",)` (default, each request),
`("prompt_index",)` (the `n` samples of one prompt share it), `("route",)` (requests with equal
`generate(..., routes=(...))` values share it, and it survives between calls).

```python
@dataclass(frozen=True)
class Probe(pllm.Workload):
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank)))

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(self, h: torch.Tensor = pllm.final_hidden(), head: nn.Module = pllm.module("head")) -> torch.Tensor:
        return head(h.float())  # [rows, rank], each row through its own request's copy


print(testing.check(Probe(), CONFIG).passed)
```

## Write state from your own math

| Name | Meaning |
| --- | --- |
| `@pllm.write(*, state, fields)` | Return a tuple of new values, one per named field, and publish them together as the state's next version. Request fields have the leading row axis. Takes no `rows=`. |
| `@pllm.on(phase)` | Run the method in `phase` (see [Choose when a method runs](#choose-when-a-method-runs-and-which-rows-count)). |

A write that reads the field it rewrites (an accumulator) stores that field in float32. The value a method
reads is the version published before this step, so a running statistic updates once per decode step.

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.write(state="stats", fields=("mean",))
    def update(self, h: torch.Tensor = pllm.final_hidden(),
               mean: torch.Tensor = pllm.parameter("stats", "mean")) -> tuple[torch.Tensor]:
        return (0.9 * mean + 0.1 * h.float(),)

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(self, mean: torch.Tensor = pllm.parameter("stats", "mean")) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)


print(testing.check(RunningMean(), CONFIG).passed)
```

## Train during decode

| Name | Meaning |
| --- | --- |
| `@pllm.train(*, state, reduction="mean", rows="", update_rows="", steps=1, reset="")` | Minimize the returned loss with the optimizer declared on `state`; the framework masks padding, reduces, backpropagates, applies the optimizer and publishes the new version. Request-row training returns a `[rows]` loss. The default `reduction="mean"` averages each state's selected rows in this step; `"sum"` sums them. Host-batch training returns one scalar and owns its reduction; remove non-default `reduction=`. |
| `pllm.AdamW(lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)` | The optimizer for training on request rows and on `WeightedRowBatch` / `CausalTrainBatch` host batches. Declared on the state: `optimizer=pllm.AdamW(1e-3)`. |
| `pllm.SGD(lr)` | The optimizer for `WeightedCEBatch` training (`steps=`, `reset=`) and for `@pllm.train_in_forward`. |
| `@pllm.differentiate(*, inputs=(), parameters=False, input_rows="", parameter_rows="")` | Keep gradient edges through a non-loss method: `inputs` names its arguments whose gradients flow, `parameters=True` lets its parameters train too. A training method reads its result with `pllm.result(name)` and keeps the edge with its own `@pllm.differentiate(inputs=(...))`; without `@pllm.on` the method runs in the phase of the method that reads it. |
| `pllm.result(operation)` | The value another method of this workload returned in the same step. |
| `pllm.UpdateOutput(output, changed, update_count, trained_rows, metrics={})` | An optional richer return of a training method: the loss plus 0-d device counters (no host sync). |
| `pllm.carry(source, *, state, field)` | Wrap an input marker of a training method: when the update publishes, the input's value is also written to `state`'s `carry=` field (per request the last selected row wins). |

Training on request rows runs after sampling, on a side CUDA stream that overlaps the next forward, and is
captured in CUDA graphs. With the default `publish_lag=1`, the update computed at step n is read from step
n + 2. The training method reads the newest version; other methods read the published one.

```python
@dataclass(frozen=True)
class ShallowDistiller(pllm.Workload):
    beta: float = 0.5
    rank: int = 4

    @property
    def distiller(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(
                lambda m: nn.Sequential(nn.Linear(m.hidden_size, self.rank), nn.Linear(self.rank, m.hidden_size))
            ),
            optimizer=pllm.AdamW(1e-3),
        )

    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    @pllm.train(state="distiller")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(self, shallow: torch.Tensor = pllm.hidden(0),
                f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return f(shallow.float())


print(testing.check(ShallowDistiller(), CONFIG, neutral=ShallowDistiller(beta=0.0)).passed)
```

```python gpu
sampling = SamplingParams(n=4, max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
with pllm.make_llm(GPU_CONFIG, workload=ShallowDistiller()) as llm:
    llm.generate(["Explain gradient descent:"], sampling)
    report = llm.diagnostics()
print(report.update_count > 0, report.trained_rows > 0, repr(report.eager_reason))
```

## Add to the logits

| Name | Meaning |
| --- | --- |
| `@pllm.add_logits(*, rows="")` | Return float `[rows, vocab]`; pLLM adds it to the model's logits before temperature, top-k and top-p. One row per token the sampler draws. |
| `pllm.sample_index(limit=None)` | int64 `[rows]`: each row's sample number within its prompt (0 for the first of `n`). With `limit`, `generate` rejects requests whose sample index is `limit` or more. |
| `pllm.base_logits()` | The model's logits before any workload adjustment. Readable only after sampling (in `@pllm.train` or `@pllm.on(pllm.Phase.AFTER_SAMPLE)`). |
| `pllm.sampled_tokens()` | The tokens drawn this step. Readable only after sampling. |

Read the logits you want to adjust with `pllm.layer_logits((-1,))`, not `pllm.base_logits()`. Several
`@pllm.add_logits` methods may coexist; their sums apply before guidance. A workload has at most
one `@pllm.guide_sampling` method, components included. For the `neutral=` check, give the workload
a strength setting whose value 0 makes the method return exactly zero.

```python
@dataclass(frozen=True)
class LogitBias(pllm.Workload):
    strength: float = 1.0
    bias = pllm.InstanceState({"b": lambda m: (m.vocab_size,)}, init=pllm.Imported())

    @pllm.add_logits()
    def add(self, sample: torch.Tensor = pllm.sample_index(),
            b: torch.Tensor = pllm.parameter("bias", "b")) -> torch.Tensor:
        return self.strength * b.float().expand(sample.shape[0], -1)  # [rows, vocab]


print(testing.check(LogitBias(), CONFIG, neutral=LogitBias(strength=0.0)).passed)
```

## Guide sampling with a hidden-state prediction

| Name | Meaning |
| --- | --- |
| `@pllm.guide_sampling(*, beta, rows="")` | Return a hidden-width prediction `[rows, hidden]`. pLLM projects it with the LM head and samples from `(1 + beta) * logits - beta * prediction_logits` over the tokens that survive top-k / top-p, pushing sampling away from what the prediction already explains. `beta=0` declares no sampler work. |

`rows="method"` names a selector (see the next section); unselected rows sample unchanged. See
`ShallowDistiller` in [Train during decode](#train-during-decode).

## Choose when a method runs and which rows count

| Name | Meaning |
| --- | --- |
| `pllm.Phase` | Windows of one engine step: `BEFORE_FORWARD`, `IN_FORWARD`, `AFTER_FORWARD`, `BEFORE_SAMPLE`, `AFTER_SAMPLE`, `AUTOMATIC`. |
| `@pllm.on(phase)` | Run the method in `phase`. Effects have defaults (export: after the forward; sampler methods: before sampling; request training and writes: automatic), so most methods need no `@pllm.on`. |
| `rows="method"` | On `@pllm.add_logits`, `@pllm.guide_sampling` and `@pllm.train`: names a method with marker inputs that returns a bool `[rows]` mask; only those rows apply the effect (for training: count in the loss). Request-row training alone or next to `@pllm.guide_sampling` updates every request and rejects `rows=`. |
| Settings from the instance | A decorator or marker argument may be a function of the workload instead of a value, called once when the workload is built: usually a method defined above, `beta=strength` with `def strength(self) -> float: return self.beta`, or `layer_logits(lens)`; `lambda self: self.beta` also works. The IDE follows the method and pyright checks its return type. Naming a dataclass field in the class body (`beta=beta`) reads its class default, not the instance. |

Each method takes at most one decorator of each kind. A value can be read only in or after the phase that
produces it: sampled tokens and `pllm.base_logits()` after sampling, the current step's hidden rows after the
forward.

## Edit activations or weights from a state

| Name | Meaning |
| --- | --- |
| `pllm.model.ActivationPatch(state, layers, first_tokens=0, last_tokens=2, operator="low_rank_affine", *, decode=False, refresh="admission", on_missing="stop")` | Edit the output of `layers` (the residual stream after them) from a state, whoever writes it. `operator="add"` adds one vector field per layer; `"low_rank_affine"` reads three fields per layer (ReFT). Applies to the first `first_tokens` and last `last_tokens` prompt tokens, and with `decode=True` to every decode token. `refresh="step"` reads the newest version every decode step. Runs inside the model's CUDA graphs. |
| `pllm.model.Adapter(state, targets, weight="value", operator="dense", factors=("a", "b"), scale=1.0)` | Apply a state as a weight delta at the target projections, whoever trains it (PEFT-style). `operator="low_rank"` applies `scale * b @ a` from fields named by `factors`; `"dense"` and `"per_head"` read the field `weight`. A trainable `"low_rank"` state needs a non-zero `init=` (e.g. `pllm.Computed` with random A, zero B): with both factors at zero every gradient is zero, so it is rejected. An Adapter on an `InstanceState` makes the model forward eager; on a `RequestState` it stays in CUDA graphs. Inside `@pllm.custom_forward` the delta applies at every projection call, including repeated layers. |
| `pllm.model.Projection(name, layer=None)` | A linear projection by its Hugging Face module name: `"q_proj"`, `"k_proj"`, `"v_proj"`, `"gate_proj"`, `"up_proj"` or `"down_proj"`; `layer=None` means every layer. |

Declare both as class attributes or typed properties. A `pllm.Field` shape callable of a state an `Adapter`
reads receives each target's `(in_features, out_features)`; the state then has one copy of each field per
target, named `a_0`, `b_0`, `a_1`, ... in target order.

With `tensor_parallel_size > 1` (vLLM 0.32), `"dense"` and `"low_rank"` deltas follow each projection's
shards: q/k/v and gate/up partition dense weights and B by output rows, down partitions dense weights and A by
input columns; the other factor is replicated. The state starts from `pllm.Zeros` or `pllm.Computed` and only
`@pllm.train_in_forward` may update it, using the local shard (training replicated A from column-sharded outputs,
or B from row-sharded inputs, is rejected). `"per_head"` deltas and adapter states with an optimizer or another
`init=` source need `tensor_parallel_size=1`.

```python
class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


@dataclass(frozen=True)
class PromptLoRA(pllm.Workload):
    rank: int = 4

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Imported(),
        )

    @property
    def apply(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12))
        return pllm.model.Adapter("lora", targets, operator="low_rank")


print(pllm.explain(Steering(), CONFIG).eager_reason)
```

## Import or feed state from outside

| Name | Meaning |
| --- | --- |
| `pllm.io.StateImport(values, keys=())` | Tensors for an `init=pllm.Imported()` state, in field order, passed as `llm.generate(..., batch=StateImport(...))`. An instance state gets a new version that this call's requests read; later calls without `batch=` keep it. For request state the values carry a leading prompt axis and are written once, before each request's first forward. |
| `pllm.io.load_tensor_fields(path, fields)` | Load named tensors from a weights-only checkpoint file, in the order of `fields`. |
| `pllm.io.checkpoint_directory(model, revision=None)`, `pllm.io.checkpoint_tensors(checkpoint, names)` | Read base-model tensors (e.g. `model.embed_tokens.weight`) on CPU: resolve the downloaded model snapshot, then load the named tensors from it or from one safetensors file. |
| `pllm.io.Feed` | `llm.feed(state)` returns it for an `InstanceState(init=pllm.Fed("ordered" or "latest"))` filled by a producer process: `publish(**fields)`, `missing()`, `end()`, `close()`. See `examples/video_session.py`. |

```python gpu
del seen  # the StreamNorm example above is done
vector = torch.nn.functional.normalize(torch.randn(896), dim=0) * 8  # Qwen2.5-0.5B: hidden size 896

with pllm.make_llm(GPU_CONFIG, workload=Steering()) as llm:
    for strength in (0.0, 1.0):
        out = llm.generate(["I think that today"], GREEDY, batch=pllm.io.StateImport((strength * vector,)))
        print(strength, repr(out[0].outputs[0].text))
    print(repr(llm.diagnostics().eager_reason))  # '': activation patches run inside CUDA graphs
```

## Replace an attention read or the KV layout

| Name | Meaning |
| --- | --- |
| `@pllm.custom_attention(*, layers=None, keep="all", window=0, source=0, sink=False)` | Replace the attention read of `layers` with `f(self, q, kv, ...markers)`; vLLM's q/k/v projections and KV writes stay. `q` is `[rows, heads, head_dim]` after RoPE. Return attention with q's shape and dtype, optionally `(attention, aux)` with `aux` shaped `[rows, ...]`. One method per layer. |
| `pllm.model.PagedKV` | The method's `kv`: read-only paged K/V already holding the current tokens. `kv.attend(q, *, window=0, topk=None)` returns attention `[rows, heads, head_dim]` and float32 LSE `[rows, heads]`: the native causal read without arguments, the last `window` positions (current included) with `window=`, and only the given logical positions with `topk=` (int `[rows, k]` or `[rows, heads, k]`, padded with -1, no repeats within a head). Also `positions()`, `requests()`, `sink_lens()`, `window_start()`. |
| `pllm.attention(layer, fact)` | `fact="aux"`: the aux an earlier layer's custom attention returned for the same token; `"lse"`: the LSE of its first `kv.attend` call. Inside attention methods it reads strictly earlier layers; in tensor methods (e.g. an export) it reads sampled rows. |

Layouts change what is stored, not only what is read: `keep="window", window=N` keeps the last N tokens of K/V
(vLLM's sliding-window spec frees older blocks); adding `sink=True` keeps the whole prompt plus the window;
`keep="none", source=k` allocates no KV at this layer and reads layer `layer - k`'s. Layouts limit every read,
including `topk=`. Top-K selection saves reads, not memory; freeing memory by score is unsupported.
`pllm.explain` prints a `KV layer` line per layer with the per-request bound. Methods may read
`InstanceState` parameters or modules, `pllm.depth()` and earlier `pllm.attention(...)` facts, and call pure
custom ops. Requirements: `tensor_parallel_size=1`, unquantized KV, plain RoPE attention, prefix caching off,
dense Qwen2/Llama/Qwen3 or Qwen3-MoE; layouts also exclude `pllm.PositionState` and
`@pllm.custom_forward`. Full reads use FlashAttention 2; stored windows use vLLM's FlashInfer plans on
Ampere/Ada (they keep grouped-query decode, which FA2's window path drops); prompt-sink reads use vLLM's
Triton path.

```python
@dataclass(frozen=True)
class RecentWindow(pllm.Workload):
    window: int = 64

    def span(self) -> int:
        return self.window

    @pllm.custom_attention(layers=(0, 2, 4), keep="window", window=span)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        attention, _lse = kv.attend(q, window=self.window)
        return attention


LONG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, max_model_len=4096)
report = pllm.explain(RecentWindow(), LONG)
print(repr(report.eager_reason))  # '': custom attention runs inside the model's CUDA graphs
for capacity in report.kv_capacity[:2]:
    print(capacity.layout.layer, capacity.layout.description, capacity.bytes)  # 0 window 64 565248 / 1 all 2097152
```

The bound includes the tokens one step may schedule (`max_num_batched_tokens`), so a window saves memory only
when `max_model_len` is well above `window + max_num_batched_tokens`.

## Rewrite a residual site

| Name | Meaning |
| --- | --- |
| `@pllm.custom_residual(*, site, layers=None, read="", streams=1, state="", chunk=None)` | Own the residual update at `site` of `layers` (default every layer). `site="attention"` / `"mlp"` / `"both"`: `f(x, y, ...) -> stream` replaces `x + y` (x: the stream before the sublayer, y: its branch output); `"value"`: maps a full-attention layer's value rows `[rows, kv_heads, head_dim]`; `"final"`: maps the stream entering the final norm. `streams=n` carries `[rows, n, hidden]` streams (hyper-connections) until a `"final"` method collapses them. With `state=` (a one-field `RequestState`) and `chunk=`, the method is a stateful sublayer returning `(stream, next carry)`. |
| `pllm.depth()` | Inside a `custom_residual` method: the layer it runs at, a Python int (`2 * layer`, plus 1 for the MLP, with `site="both"`). |
| `pllm.value(layer)` | Inside a `custom_residual` method: the value rows layer `layer` computed for the same token in this forward. |

The site tensors come first, without defaults; later inputs may be `pllm.parameter`, `pllm.module`,
`pllm.depth()`, `pllm.value(layer)` and `pllm.residual(layer)` (the same token, earlier in this forward).
Tensors are in the model dtype. A method returning the plain update (`x + y`, `v`) reproduces the model
bitwise. Costs: the method runs inside the model's CUDA graphs, where vLLM captures the model's own operators
instead of inductor-fused ones, so compare a neutral run with eager vLLM (or use `check_on_gpu`, which builds a
matching vanilla engine); `tensor_parallel_size=1`; prefix caching off. Stateful sublayers (`state=`) need a larger
`ExecutionOptions(cuda_graph_pool_budget_bytes=...)`; the error names the bytes required.

```python
@dataclass(frozen=True)
class ValueResidual(pllm.Workload):
    mixing: float = 0.0

    @pllm.custom_residual(site="value", layers=tuple(range(1, 24)))
    def mix(self, v: torch.Tensor, first: torch.Tensor = pllm.value(0)) -> torch.Tensor:
        return (v.float() + self.mixing * (first.float() - v.float())).to(v.dtype)


print(pllm.explain(ValueResidual(), CONFIG).eager_reason)
```

## Run layers in a loop or exit early

| Name | Meaning |
| --- | --- |
| `@pllm.custom_forward(*, tokens="all", over="token")` | Replace the forward of the selected tokens (`"prefill"`, `"decode"`, `"all"`) with the method `f(self, layers, ...) -> stream`, returning the residual stream the final norm reads. Inputs after `layers` may be `pllm.parameter` / `pllm.module` of an `InstanceState`; the body reads plain settings from `self`. |
| `pllm.model.Layers` | The method's `layers` argument. `layers(start=0, stop=None, hidden=None, *, kv="commit", active=None)` runs layers `start..stop-1` and returns their outputs; `kv="peek"` attends without writing the KV cache; `active` (bool `[rows, 1]`, only with `tokens="decode"`) freezes the rows that are False (early exit). Also `embeddings`, `generated` (bool `[rows, 1]`), `load(name)` / `save(name, value)` for a `pllm.PositionState`, `apply_updates()` for `@pllm.train_in_forward`. |
| `pllm.model.LayerOutputs` | What `layers(...)` returns: the post-layer residual streams of the layers that slice ran, indexed by absolute model layer, negative from the model's end. `out[-1]` is the model's last layer, so it exists only when the slice runs to the end; `layers(stop=12)[11]` reads a partial slice (`layers(stop=12)[-1]` raises `index a tap inside the Layers slice that produced it`). |
| `pllm.scan(step, *, carry, over, chunk, invariants=(), valid_rows=None)` | Run `step(carry, *slices, *invariants) -> (carry, readout)` over `over` in chunks of `chunk` rows, in order, returning the final carry and the readouts. |

pLLM schedules a custom forward's layer slices itself and captures its own CUDA graphs for them, so vLLM's
whole-model graph is not used and `eager_reason` names `@pllm.custom_forward`. Methods reading `pllm.hidden` or
`pllm.final_hidden` are rejected; `pllm.residual` and `pllm.layer_logits` read each layer's last visited stream.
A `pllm.model.Adapter` in the same workload applies its delta at every projection call the program makes. Tensor parallelism needs vLLM 0.32.0. The slice graphs may need a larger
`ExecutionOptions(cuda_graph_pool_budget_bytes=...)` than the default; explain does not estimate it, and the error names the
bytes required. Bounded Python `for` loops over workload settings are unrolled when traced.

```python
@dataclass(frozen=True)
class LatentLoop(pllm.Workload):
    start: int = 20
    loops: int = 2
    tol: float = 0.05

    @pllm.custom_forward(tokens="decode")  # active= needs tokens="decode"; prompts run the plain forward
    def think(self, layers: pllm.model.Layers) -> torch.Tensor:
        hidden = layers()[-1]
        active = torch.ones_like(hidden[:, :1], dtype=torch.bool)
        for _ in range(self.loops):
            again = layers(start=self.start, hidden=hidden, active=active)[-1]
            change = (again - hidden).float().norm(dim=-1, keepdim=True) / hidden.float().norm(dim=-1, keepdim=True)
            hidden = torch.where(active, again, hidden)
            active = active & (change > self.tol)  # rows that converged stop looping
        return hidden


print(testing.check(LatentLoop(), CONFIG).passed)
```

## Train inside the forward

| Name | Meaning |
| --- | --- |
| `@pllm.train_in_forward(*, state, targets, weight="value", operator="dense", factors=("a", "b"), scale=1.0, tokens="prefill", visibility="forward_pinned", chunk=None, flush=None)` | Train a `RequestState` from the operands of `targets` while the forward runs, and add the state's delta at those projections (as `pllm.model.Adapter` would). With `chunk=` and `optimizer=pllm.SGD(lr)` on the state, return one loss per row; gradients accumulate over `chunk` rows and SGD steps once per chunk. `flush="prompt_end"` also trains the short last chunk of a prompt. |
| `pllm.operand(name)` | Inside `train_in_forward`: `"input"` (the projection's input rows), `"target"` (the MLP input of the target's layer), `"weight"` (the base weight), `"prompt_end"` (a scalar bool: this chunk ends the prompt). |
| `pllm.UpdateOutput.unchanged(output)` | An update result that changes nothing, with zero device counters. |

`visibility="forward_pinned"` (default) makes the whole forward read the state it started with;
`"exact"` lets later chunks of the same forward read the updates of earlier chunks. `tokens` picks whose
operands train: `"prefill"` (default), `"decode"` or `"all"`.

```python
TARGETS = tuple(pllm.model.Projection("down_proj", layer) for layer in (0, 2))


def start_small(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["a"].normal_(std=0.02, generator=generator)
    fields["b"].zero_()


@dataclass(frozen=True)
class PromptAdapters(pllm.Workload):
    rank: int = 4

    @property
    def adapters(self) -> pllm.RequestState:
        return pllm.RequestState({"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                                  "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))},
                                 init=pllm.Computed(start_small), optimizer=pllm.SGD(1e-3))

    @pllm.train_in_forward(state="adapters", targets=TARGETS, operator="low_rank", chunk=16, flush="prompt_end")
    def learn(self, x: torch.Tensor = pllm.operand("input"), target: torch.Tensor = pllm.operand("target"),
              base: torch.Tensor = pllm.operand("weight"), a: torch.Tensor = pllm.parameter("adapters", "a"),
              b: torch.Tensor = pllm.parameter("adapters", "b")) -> torch.Tensor:
        prediction = x @ base.T + (x @ a.to(x.dtype).T) @ b.to(x.dtype).T
        return (prediction.float() - target.float()).square().mean(-1)


print(testing.check(PromptAdapters(), CONFIG).passed)
```

## Train on host batches or run a host feedback loop

A training method that reads `pllm.batch(...)` fields trains on a batch you pass to
`llm.generate(..., batch=...)` (or to `HostTrainer.train` for state only host code reads) instead of on request rows.

| Name | Meaning |
| --- | --- |
| `pllm.batch(field)` | One field of the host batch. The fields a `@pllm.train` method reads pick the batch type and the optimizer. |
| `pllm.host.WeightedRowBatch(hidden, targets, weights)` | `[rows, hidden]` features, one target and one weight per row (weight 0 skips a row); one AdamW step per batch. |
| `pllm.host.CausalTrainBatch(input_ids, labels, attention_mask=None, weights=None)` | int64 `[sequences, tokens]` ids and labels for causal LM training through `pllm.causal_model`, with optional floating `weights` (one per sequence, read with `pllm.batch("weights")`, e.g. GRPO advantages); one AdamW step per batch. |
| `pllm.host.WeightedCEBatch(hidden, base_logits, labels, weights)` | Weighted cross-entropy rows; trains with `pllm.SGD`, `steps=` steps per state after the `reset=` method restores it. `WeightedCEBatch.for_prompts(..., prompt_indices)` keys one batch per prompt. |
| `pllm.causal_model(adapter)` | A `pllm.model.CausalModel` with the named `pllm.model.Adapter` applied: `model(ids, attention_mask=...)` returns logits whose gradients reach the adapter's state; `model.base` is the same model without the adapter. `model.token_logprobs(ids, labels, *, attention_mask=None, ignore_index=-100)` returns at least-FP32 `[sequences, tokens - 1]` log-probabilities of `labels[:, 1:]`, running the LM head in blocks so the `[sequences, tokens, vocab]` logits never exist at once (positions labelled `ignore_index` read 0). |
| `pllm.host.make_host(config, workload)` | Compile a batch-training workload without a model and return its `pllm.host.HostTrainer`: `train(batch)` per batch, `version`, `diagnostics()`, `close()` (or use it in `with`). |
| `pllm.host.HostConfig(device, hidden_size=0, target_shape=(), seed=0, state_page_budget_bytes=..., max_input_rows=None, execution_peak_budget_bytes=None)` | What `make_host` compiles against: device, hidden width of the batches and the shape of each row's target. |
| `pllm.host.TrainStep` | What `train` returns: `loss` and `parameters` (the updated fields by name). |
| `pllm.host.CompletionInput` | What a feedback callable receives: `completions` (a `CompletionBatch`) and `hidden` (each completion's final hidden row). |
| `pllm.host.Completion`, `pllm.host.CompletionBatch` | One sampled completion (`identity`, `output_index`, `group_id`, `token_ids`, `text`, `finish_reason`), and a batch of them (`entries`). |
| `pllm.host.FinishReason` | `STOP`, `LENGTH` or `ERROR`. |
| `pllm.host.RetryControl(completion, action, reason, retry_after_update)` | The callable's decision for one completion; `action` is a `pllm.host.RetryAction` (`ACCEPT`, `RETRY`, `STOP`), `reason` a `pllm.host.ControlReason`. |
| `pllm.host.UpdateBarrier(barrier_id, group_id, state, request_ids, ready_count, reason)` | A group of requests that retry together once an update is ready; `state` is a `pllm.host.BarrierState` (`READY`, `BLOCKED`). |
| `pllm.host.RewardOutcome(completion, reward, is_correct, feedback_text, judge, label, skip_reason)` | A scored completion, for the callable's own result. |

`llm.generate_feedback(prompts, sampling, *, feedback, group_id, max_rounds)` generates, calls
`feedback(completion_input)`, and regenerates the completions the result marks `RETRY`, for at
most `max_rounds` rounds. Each retried sample runs alone (`n=1`) and keeps its place among its prompt's `n`
samples: sample j has sample index `k*n + j` in round k. It yields `(completion_input, result)` per round. The result is any object with
`retry_controls`, `update_barriers` and `update_count`; returning `None` ends the loop. See
`pllm/workloads/reward_head.py` and tutorial 05.

```python
@dataclass(frozen=True)
class RewardHead(pllm.Workload):
    head = pllm.InstanceState({"w": pllm.Field(lambda width, _: (width,))}, optimizer=pllm.AdamW(1e-2))

    @pllm.train(state="head")
    def fit(self, hidden: torch.Tensor = pllm.batch("hidden"), targets: torch.Tensor = pllm.batch("targets"),
            weights: torch.Tensor = pllm.batch("weights"),
            w: torch.Tensor = pllm.parameter("head", "w")) -> torch.Tensor:
        error = (hidden @ w - targets).square()
        return (weights * error).sum() / weights.sum().clamp_min(1)


with pllm.host.make_host(pllm.host.HostConfig(torch.device("cpu"), hidden_size=896), RewardHead()) as head:
    batch = pllm.host.WeightedRowBatch(torch.randn(4, 896), torch.ones(4), torch.ones(4))
    step = head.train(batch)
    print(step.loss is not None, tuple(step.parameters["w"].shape), head.version)
```

## Configure the engine and execution

| Name | Meaning |
| --- | --- |
| `pllm.make_llm(config, *, workload=None)` | A vLLM engine running `workload`; use as a context manager. Methods: `generate(prompts, sampling_params, *, batch=None, sample_index=0, prompt_indices=None, routes=None)`, `generate_completions`, `generate_feedback`, `diagnostics()` (`eager_reason`, `update_count`, `trained_rows`, `loss_count`, ...), `trace_snapshot()`, `select_trace(...)`, `feed(name)`, `close()`. Without a workload it is plain vLLM. `close()` (or leaving `with`) also releases the engine's compile state, so engines can be created and closed repeatedly in one process. |
| `pllm.ModelConfig(model, dtype="bfloat16", max_model_len=512, max_num_seqs=8, max_num_batched_tokens=512, gpu_memory_utilization=0.5, seed=0, ..., multiprocessing=False, execution=ExecutionOptions(), cuda_graphs="auto", kv_offload=None, tensor_parallel_size=1, ...)` | The model and engine. `cuda_graphs="auto"` uses CUDA graphs when the workload allows it (else eager, reason in `diagnostics().eager_reason`); `True` requires them. `multiprocessing=True` needs the workload class in an importable module. |
| `pllm.ExecutionOptions(max_host_training_rows=1, cuda_graph_pool_budget_bytes=None, state_page_budget_bytes=0, execution_peak_budget_bytes=None, allow_numerical_transforms=True, tuning="heuristic", publish_lag=1)` | Framework policy. `publish_lag=0`: updates at step n are read at n + 1 and training no longer overlaps the next forward. `state_page_budget_bytes` bounds the state's device pages (0 uses observed free capacity); `execution_peak_budget_bytes` bounds the pLLM execution peak including those pages, task workspace/live outputs and graph/anchor reservations (None is unlimited), excluding model weights/KV. `cuda_graph_pool_budget_bytes` budgets each task/model CUDA-graph pool (None sizes tasks from tracing and lets vLLM manage model capture). `max_host_training_rows` bounds host-fed training rows per state per step, default 1; decode request rows are separate. |
| `pllm.KVOffload(connector="OffloadingConnector", cpu_bytes=536870912, prompt_only=False)` | Offload complete KV pages to CPU memory (`ModelConfig(kv_offload=...)`). |
| `pllm.Workload` | The base class. Subclass it (a frozen dataclass when it has settings); put declarations and tensor math in it, nothing else. |

### The engine handle

`pllm.make_llm` returns a handle; use it as a context manager. Its methods:

| Method | Meaning |
| --- | --- |
| `generate(prompts, sampling_params, *, batch=None, sample_index=0, prompt_indices=None, routes=None, use_tqdm=False)` | vLLM's `generate` with the workload running: one `RequestOutput` per prompt holding its `n` samples. `batch` feeds `pllm.batch` training or a `pllm.io.StateImport`; `sample_index` numbers each prompt's first sample (with a `seed`, sample index i samples with `seed + i`); `prompt_indices` labels prompts across calls (`identity.prompt_index` in exports); `routes` gives one value per prompt for `key_fields=("route",)` states. |
| `generate_completions(prompts, sampling_params, *, group_id, ...)` | Like `generate`, returning a `pllm.host.CompletionInput`: every sample as a `Completion` plus its final hidden row. |
| `generate_feedback(prompts, sampling_params, *, feedback, group_id, max_rounds, ...)` | The host loop of [Train on host batches](#train-on-host-batches-or-run-a-host-feedback-loop): generate, call `feedback`, retry. |
| `diagnostics()` | Exact counters read from the workers (see below). |
| `trace_snapshot()`, `select_trace(selection)` | Retained trace events (exports, timings) and opt-in extra snapshots or kernel profiling. |
| `feed(name)` | The `pllm.io.Feed` of a fed state (in-process engines only). |
| `hidden_size`, `vocab_size`, `device` | Model facts, for building host batches. |
| `close()` | Shut the engine down and release its GPU memory and compile state; leaving `with` calls it. |

`diagnostics()` fields: `eager_reason` (empty when the model runs in CUDA graphs), `update_count`,
`trained_rows`, `loss_count` (training progress), `active_states` and `inference_pins` (live request states and
the versions running steps hold), `state_usage` (pLLM's state pages: `reserved_bytes`, `logical_live_bytes`,
`admission_capacity_failures`, ...), `numeric_health` (norm, max-abs and non-finite count of each live
published field), `hold_count` / `last_hold` (steps that held an external input, such as a state feed, instead of applying it),
`kv_cache_blocks` and `kv_cache_max_concurrency` (vLLM's KV allocation and how many `max_model_len`
requests fit, per TP rank).

### ModelConfig fields

| Field | Default | Meaning |
| --- | --- | --- |
| `model`, `revision`, `trust_remote_code` | — | Hugging Face model id or path, as vLLM takes them. |
| `dtype` | `"bfloat16"` | Model dtype (`"float16"`, `"float32"`). |
| `max_model_len`, `max_num_seqs`, `max_num_batched_tokens` | `512`, `8`, `512` | vLLM's length, concurrency and per-step token limits; pLLM sizes its state and taps from them. |
| `gpu_memory_utilization` | `0.5` | Fraction of GPU memory vLLM takes for weights, activations and KV cache; pLLM's state pages and workspaces use free memory outside it, so leave room. |
| `seed` | `0` | Seeds sampling and `pllm.Computed` initializers. |
| `enable_prefix_caching`, `enable_chunked_prefill` | `False`, `False` | vLLM features; see [rules.md](rules.md#engine-settings-that-interact-with-workloads). |
| `generated_kv` | `"isolate"` | With prefix caching and a workload that treats prompt and generated tokens differently (patches, adapters, some custom forwards): `"isolate"` keys blocks of generated tokens apart from prompt blocks with the same tokens; `"reuse"` lets them match. |
| `multiprocessing` | `False` | Run vLLM's EngineCore in its own process; the workload must be picklable. |
| `execution` | `ExecutionOptions()` | pLLM's policy (next table). |
| `cuda_graphs` | `"auto"` | `"auto"`: graphs when the workload allows; `True`: require them; `False`: eager. |
| `kv_offload` | `None` | `pllm.KVOffload(...)` to offload complete KV pages to CPU memory. |
| `tensor_parallel_size` | `1` | Single-node tensor parallelism (vLLM 0.32.0); `pipeline_parallel_size` and `data_parallel_size` must stay 1. |

| `ExecutionOptions` field | Default | Meaning |
| --- | --- | --- |
| `publish_lag` | `1` | Steps hidden from readers: an update at step n is read from n + 1 + `publish_lag`. `0` stops training from overlapping the next forward. |
| `state_page_budget_bytes` | `0` | Device bytes for state pages; 0 uses the observed free capacity. |
| `execution_peak_budget_bytes` | `None` | Cap on pLLM's execution peak (pages, task workspace, results, graph and anchor reservations); excludes weights and KV. |
| `cuda_graph_pool_budget_bytes` | `None` | Budget of each task or model CUDA-graph pool; None sizes tasks from tracing and lets vLLM manage model capture. |
| `max_host_training_rows` | `1` | Host-fed training rows per state per step. |
| `guidance_capacity` | `4096` | Guide the highest K post-filter survivors per row (power of two, 32 to 4096); None is exact dense guidance. |
| `allow_numerical_transforms` | `True` | Let pLLM pick numerically equivalent implementations. |
| `tuning` | `"heuristic"` | `"measured"` times alternative implementations where several exist. |

## Inspect the plan and check contracts

| Name | Meaning |
| --- | --- |
| `pllm.explain(workload, config)` | Plan on CPU from `config.json` without weights or CUDA: per method its timing anchor, CUDA stream lane and graph capture; why the forward is eager; state bytes; peak memory pLLM reserves; host syncs in your math; timing settings. `str()` prints a table; `dataclasses.asdict()` gives JSON-ready data (`eager_reason`, `members`, `states`, `peak_memory_bytes`, `budget_errors`, `exports`, ...). `flops_per_token_estimate` is the base forward's dense projection and LM-head FLOPs per token plus every task's matmul and projection FLOPs per row (attention over the KV history excluded; `None` when the base is unknown). |
| `pllm.testing.check(workload, config, *, neutral=None)` | Run fixed three-request schedules on CPU through the real planner and executor: row permutation, request isolation, mixed prefill/decode, chunked prompts, preemption replay, publish_lag versions, and with `neutral=` that the disabled workload adds exactly zero. Returns a `CheckResult`; raises `ContractError`. |
| `pllm.testing.check_on_gpu(workload, config, *, neutral=None)` | Load real engines in a child process with batch-invariant kernels: planning, eight steady decode steps, each prompt alone versus all batched, forced KV eviction and recompute, and with `neutral=` greedy and seeded sampling bitwise equal to the plain model. Needs `multiprocessing=False`, `max_num_seqs >= 2`, `max_model_len >= 32`. |
| `pllm.testing.check_distribution(workload, config, *, prompts=None, tokens=16, temperature=0, max_mean_kl=None, max_p99_kl=None, worst=5)` | Teacher-force the plain model's continuation through the workload with matching graph/compile settings. Full-vocabulary FP32 KL in both directions and TV; thresholds bound both KL directions. Returns numbers and pass/fail; no thresholds measures only. |
| `pllm.testing.DistributionResult` | `kl_plain_to_workload`, `kl_workload_to_plain`, `tv`, `token_count`, `run_seconds`, `elapsed_seconds`, `passed`, `failures`. |
| `pllm.testing.DistanceSummary` | `mean`, nearest-rank `p99`, `max`, and `worst` positions. KL uses nats; TV is half the L1 probability distance. |
| `pllm.testing.TokenDistance` | `prompt_index`, predicted `token_offset` in prompt + continuation (both zero-based), and `value`. |
| `pllm.testing.CheckResult` | `passed` (names of verified properties) and `skipped` (each with `property` and `reason`); `str()` prints a table. |
| `pllm.testing.ContractError` | An `AssertionError` whose `failures` lists every discrepancy, naming the method that mixed rows or the framework routing that differs. |


`ExplainResult` fields, for programmatic checks: `eager_reason`; `members` (per method: `name`, `effect`,
`tasks` with `anchor`, `lane`, `captured`, `reads`, and `training`); `states` (`name`, `scope`, `pages` per field,
`bytes_per_root`, `reserved_bytes`, `fields`, `source`, `placement`, `host_bytes`, `transfer_bytes_per_row`);
`exports` (`name`, `rows`, `source_rows`, `columns`, `bytes`, `events`, `overflow`); `peak_memory_bytes`;
`budget_errors`; `preemption` (`"keep"` or `"reset"`) and `recompute_rejection`; `workload_host_syncs`;
`escape_hatches` (the `timing` lines); `kv_capacity` and `attention` (custom attention layers); `custom_ops`;
`flops_per_token_estimate`.

```python
report = pllm.explain(ShallowDistiller(), CONFIG)
print(repr(report.eager_reason), [(state.name, state.reserved_bytes) for state in report.states])
print(testing.check(Lens(), CONFIG))
```

## Measure throughput

`pllm.benchmark` times the public engine path; compare a workload with plain vLLM on equal work before claiming
a speedup.

| Name | Meaning |
| --- | --- |
| `pllm.benchmark.throughput(llm, prompts, sampling, *, warmup=1, rounds=3)` | Median output tokens per second over `rounds` whole `llm.generate` calls (prefill included) after `warmup` untimed calls. Works for any engine, with or without a workload. |
| `pllm.benchmark.compare(workload, options, *, prepare=None, max_host_training_rows=1, require_training=False, prompt_indices=None, alternatives=())` | The aligned comparison the `tools/benchmark_*.py` scripts run: builds plain vLLM and each arm with the same prompts and sampling (ignore-EOS, `sampling_n` samples per prompt) and checks equal work. Returns one `BenchmarkResult` per arm. |
| `pllm.benchmark.BenchmarkOptions(model=..., batch=8, tokens=256, rounds=3, warmup=1, gpu_memory_utilization=0.25, ..., sampling_n=16, cuda_graphs=True)` | The shape and memory settings `compare` uses. |
| `pllm.benchmark.BenchmarkResult` | `tokens_per_second` (median round rate), `round_rates`, `vanilla_tokens_per_second` (plain vLLM on the same prompts and sampling, in the arm's CUDA-graph mode), `loss_count`, `update_count`, `trained_rows`, peak memory and `state_usage`. |

```python gpu
from pllm.benchmark import throughput

prompts = ["Explain gradient descent:"] * 4
timed = SamplingParams(max_tokens=32, temperature=0.8, seed=0, ignore_eos=True)
with pllm.make_llm(GPU_CONFIG) as plain:
    vanilla = throughput(plain, prompts, timed, rounds=1)
with pllm.make_llm(GPU_CONFIG, workload=ShallowDistiller()) as llm:
    guided = throughput(llm, prompts, timed, rounds=1)
print(f"{guided / vanilla:.2f} of plain vLLM")
```

## Step a workload by hand in tests

For unit tests of your own math without a model, compile against stand-in facts and feed one packed step.

| Name | Meaning |
| --- | --- |
| `pllm.testing.backend_facts(hidden_size, hidden=(), *, device="cpu", capabilities=..., pages=16, max_lineages=2, memory_bytes=10000, max_rows=None, dtype=torch.float32, vocab_size=0, layer_count=0, targets={})` | Facts of an eager in-process model in place of what the vLLM runner probes. `hidden` lists the taps the workload reads (`hidden_binding`); `device` holds the state pages, `pages` versions per state; `capabilities` defaults to everything a CPU simulation runs; `max_rows` defaults to `2 * max_lineages`; `targets` maps each `pllm.model.Projection` an overlay targets to its `(in_features, out_features)`; `vocab_size` and `layer_count` are needed by workloads that read logits or layers. |
| `pllm.testing.hidden_binding(layer, layer_count)` | The tap `pllm.hidden(layer)` reads, for `backend_facts(hidden=...)` and `step_input(hidden=...)`. |
| `pllm.testing.final_hidden_binding(layer_count)` | The tap `pllm.final_hidden()` reads. |
| `pllm.testing.compile_execution(workload, facts, stream=None)` | Compile `workload` and admit its instance state; admit requests with `execution.admit(identities)`. |
| `pllm.testing.RequestIdentity(request_id, prompt_index, sample_index, route="")` | Who a row belongs to. |
| `pllm.testing.step_input(identities, hidden=None, *, step_id=0, tokens=None, sampler_order=None, batch=None, ...)` | One packed step, row i belonging to `identities[i]`. |
| `pllm.testing.scheduled_step(identities, scheduled, prompt, computed, hidden, *, step_id=0, operands=())` | A CPU step built from scheduler token counts, keeping only the rows the runner taps. |
| `pllm.testing.run_step(execution, value, *, feedback=None, stream=None)` | Execute every anchor of one step; returns the step result. |
| `pllm.testing.run_guided_step(execution, value, *, feedback=None, stream=None)` | Like `run_step`, also returning the sampler guidance produced before sampling. |
| `pllm.testing.SamplerFeedback(sampler_rows, base_logits, sampled_tokens)` | Sampler output to deliver to methods that read it. |

```python
class Centered(pllm.Workload):
    @pllm.export(name="centered", max_rows=2, max_cols=4)
    def center(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return h - h.mean(dim=-1, keepdim=True)


binding = testing.hidden_binding(0, layer_count=1)
execution = testing.compile_execution(Centered(), testing.backend_facts(4, (binding,), layer_count=1))
rows = (testing.RequestIdentity("a", 0, 0),)
execution.admit(rows)
h = torch.randn(1, 4)
testing.run_step(execution, testing.step_input(rows, {binding: h}))
exported = next(event.tensor() for event in execution.trace.snapshot().events if event.tensor_name == "centered")
torch.testing.assert_close(exported, h - h.mean(dim=-1, keepdim=True))
execution.retire(rows, wait=True)
```

## All signatures

<!-- signatures: generated by tools/check_docs.py --write-generated; do not edit -->
### pllm

```text
pllm.make_llm(config: 'ModelConfig', *, workload: 'Workload | None' = None) -> 'VllmLlm'
    A context-managed vLLM engine; execution options apply when a workload is installed.
pllm.explain(workload: 'Workload', config: 'ModelConfig') -> 'ExplainResult'
    Plan using model/quantization metadata and shape-only tracing; never load weights or allocate CUDA storage.
pllm.scan(step: collections.abc.Callable[..., tuple[torch.Tensor, torch.Tensor]], *, carry: torch.Tensor, over: tuple[torch.Tensor, ...], chunk: int, invariants: tuple[torch.Tensor, ...] = (), valid_rows: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]
    Run ``step(carry, *slices, *invariants)`` in chunk order, returning exclusive readouts.
pllm.custom_forward(*, tokens: 'Tokens' = 'all', over: 'Over' = 'token') -> 'Callable[[F], F]'
    Replace the forward of the selected tokens with this method's program over model layer slices and tensor math.
pllm.custom_residual(*, site: 'Site', layers: 'tuple[int, ...] | Callable[[Any], tuple[int, ...]] | None' = None, read: 'str' = '', streams: 'int | Callable[[Any], int]' = 1, state: 'str' = '', chunk: 'int | Callable[[Any], int] | None' = None) -> 'Callable[[F], F]'
    Own the residual update at `site` of `layers` (default: every layer) with this method.
pllm.custom_attention(*, layers: 'tuple[int, ...] | Callable[[Any], tuple[int, ...]] | None' = None, keep: 'Keep' = 'all', window: 'int | Callable[[Any], int]' = 0, source: 'int | Callable[[Any], int]' = 0, sink: 'bool | Callable[[Any], bool]' = False) -> 'Callable[[F], F]'
    Replace dense FlashAttention reads, retaining native projections and KV writes.
pllm.attention(layer: 'int', fact: 'Fact') -> 'Any'
    Read sampled attention aux or the first kv.attend call's float32 LSE; inside attention, use earlier layers.
pllm.depth() -> 'Any'
    Inside a @pllm.custom_residual method, the index of the place it runs at, a Python int: the layer, or 2 * layer (+ 1 for the MLP) with site="both".
pllm.value(layer: 'int | Callable[[Any], int]') -> 'Any'
    Inside a @pllm.custom_residual method, the value rows [rows, kv_heads, head_dim] layer `layer` computed for the same token in this forward (after its own value rewrite).
pllm.ModelConfig(model: str, dtype: Literal['bfloat16', 'float16', 'float32'] = 'bfloat16', max_model_len: int = 512, max_num_seqs: int = 8, max_num_batched_tokens: int = 512, gpu_memory_utilization: float = 0.5, kv_cache_memory_bytes: int | None = None, seed: int = 0, trust_remote_code: bool = False, enable_prefix_caching: bool = False, enable_chunked_prefill: bool = False, generated_kv: Literal['reuse', 'isolate'] = 'isolate', revision: str | None = None, multiprocessing: bool = False, execution: pllm.config.ExecutionOptions = ..., cuda_graphs: Union[Literal['auto'], bool] = 'auto', kv_offload: pllm.config.KVOffload | None = None, tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1, data_parallel_size: int = 1)
pllm.ExecutionOptions(max_host_training_rows: int = 1, cuda_graph_pool_budget_bytes: int | None = None, state_page_budget_bytes: int = 0, execution_peak_budget_bytes: int | None = None, allow_numerical_transforms: bool = True, tuning: Literal['heuristic', 'measured'] = 'heuristic', publish_lag: int = 1, guidance_capacity: int | None = 4096)
    Framework execution policy; lag counts post-sampler steps hidden from readers.
pllm.KVOffload(connector: Literal['OffloadingConnector', 'SimpleCPUOffloadConnector'] = 'OffloadingConnector', cpu_bytes: int = 536870912, prompt_only: bool = False)
    Batched same-engine CPU offload of complete allocation pages.
pllm.Workload()
    Python methods and effects compiled once, with independent sub-workloads in dataclass fields.
pllm.Phase: BEFORE_FORWARD | AFTER_FORWARD | BEFORE_SAMPLE | AFTER_SAMPLE | IN_FORWARD | AUTOMATIC
    The windows of one engine step a method can run in (@pllm.on); each effect has a default window.
pllm.on(phase: 'Phase | Callable[[Any], Phase]') -> 'Callable[[F], F]'
    Run the method in `phase`; needed for methods without an effect or to move an effect's phase.
pllm.write(*, state: 'str', fields: 'tuple[str, ...]') -> 'Callable[[F], F]'
    Atomically publish a tuple of computed fields; request values have a leading request axis.
pllm.export(*, name: 'str | Callable[[Any], str]' = 'value', every: 'int | Callable[[Any], int]' = 1, max_rows: 'int | Callable[[Any], int]' = 1, max_cols: 'int | Callable[[Any], int]' = 16, enabled: 'bool | Callable[[Any], bool]' = True, overflow: "Literal['drop', 'block']" = 'drop') -> 'Callable[[F], F]'
    Deliver bounded observations of executed forward rows to on_export.
pllm.add_logits(*, rows: 'str' = '') -> 'Callable[[F], F]'
    Add the returned tensor to the logits before sampling.
pllm.guide_sampling(*, beta: 'float | Callable[[Any], float]', rows: 'str' = '') -> 'Callable[[F], F]'
    Push sampling away from the returned hidden-state prediction for positive `beta`.
pllm.train(*, state: 'str', update_rows: 'str' = '', reduction: 'TrainReduction' = 'mean', steps: 'int | Callable[[Any], int]' = 1, reset: 'str' = '', rows: 'str' = '') -> 'Callable[[F], F]'
    Minimize the returned loss with `state`'s optimizer; updates publish after sampling by default.
pllm.differentiate(*, inputs: 'tuple[str, ...]' = (), parameters: 'bool' = False, input_rows: 'str' = '', parameter_rows: 'str' = '') -> 'Callable[[F], F]'
    Keep derivative edges; parameters may stay frozen while inputs differentiate.
pllm.train_in_forward(*, state: 'str', targets: 'tuple[Projection, ...] | Callable[[Any], tuple[Projection, ...]]', weight: 'str' = 'value', operator: 'AdapterOperator | Callable[[Any], AdapterOperator]' = 'dense', factors: 'tuple[str, str] | Callable[[Any], tuple[str, str]]' = ('a', 'b'), scale: 'float | Callable[[Any], float]' = 1.0, tokens: 'Tokens | Callable[[Any], Tokens]' = 'prefill', visibility: 'VisibilityKind | Callable[[Any], VisibilityKind]' = 'forward_pinned', chunk: 'int | Callable[[Any], int] | None' = None, flush: "Literal['prompt_end'] | None" = None) -> 'Callable[[F], F]'
    Train a request state inside the forward and add its residual at each target projection.
pllm.hidden(layer: 'int | Callable[[Any], int]' = 0, *, tokens: "Literal['decode', 'prefill', 'all']" = 'decode') -> 'Any'
    The MLP output before the residual add in decoder layer `layer` (negative from the end).
pllm.residual(layer: 'int | Callable[[Any], int]', *, after: "Literal['layer', 'attention']" = 'layer', tokens: "Literal['decode', 'prefill', 'all']" = 'decode') -> 'Any'
    The residual stream after decoder layer `layer` (negative from the end): the embedding plus every earlier sublayer's output, which the next sublayer normalizes. This is what steering vectors, sparse autoencoders and the logit lens read; pllm.hidden(layer) is only the layer's MLP output before it is added to this stream. after="attention" reads the stream between the layer's attention and its MLP. Rows are chosen by `tokens` as in pllm.hidden.
pllm.layer_logits(layers: 'tuple[int, ...] | Callable[[Any], tuple[int, ...]]') -> 'Any'
    The model's final norm and LM head applied to the residual stream after each of `layers` (negative from the end), as float32 [rows, len(layers), vocab]: the logit lens. The last layer's logits are the model's own. One row for each token the sampler draws: a prompt's last token and every decode token.
pllm.final_hidden(*, tokens: "Literal['decode', 'prefill', 'all']" = 'decode') -> 'Any'
    The hidden state after the model's final norm, which the LM head reads (Hugging Face's last_hidden_state); rows are chosen by `tokens` as in pllm.hidden.
pllm.parameter(state: 'str', field: 'str', *, per_row: 'bool' = False) -> 'Any'
    One tensor field of a declared state, aligned to the operation's rows for request state.
pllm.module(state: 'str') -> 'Any'
    A state declared with init=pllm.FromModule(...), called like the module; request state uses each row's own parameters.
pllm.version(state: 'str') -> 'Any'
    How many updates each row's state has received (0 before the first update).
pllm.index(state: 'str') -> 'Any'
    Each row's slot in the state pool, for indexing pooled tensors.
pllm.input_tokens(*, history: 'int' = 0) -> 'Any'
    Int64 [rows, history + 1] token ids: current token first, then lookbacks; -1 before the request starts.
pllm.sample_index(limit: 'int | Callable[[Any], int] | None' = None) -> 'Any'
    The sample index of each row's request: 0 for a prompt's first sample.
pllm.result(operation: 'str') -> 'Any'
    The value another operation of this workload returned in the same step.
pllm.base_logits() -> 'Any'
    The model's logits before any workload adjustment.
pllm.sampled_tokens() -> 'Any'
    The tokens the sampler drew this step.
pllm.operand(name: 'OperandName | Callable[[Any], OperandName]') -> 'Any'
    A projection input/target/weight, input token embedding, or scalar prompt-end bool.
pllm.carry(source: 'InputMarker', *, state: 'str', field: 'str') -> 'Any'
    Read `source` and, when the training update publishes, write its value into the carried `state.field`.
pllm.causal_model(adapter: 'str') -> 'Any'
    The causal LM with the named pllm.model.Adapter applied, as a pllm.model.CausalModel; .base skips it.
pllm.batch(field: "Literal['hidden', 'base_logits', 'labels', 'weights', 'targets', 'input_ids', 'attention_mask']") -> 'Any'
    One field of the host batch passed to generate.
pllm.InstanceState(fields: 'Mapping[str, Shape | Field]' = <factory>, shape: 'Shape | None' = None, dtype: "torch.dtype | Literal['auto']" = 'auto', *, init: 'StateInit' = Zeros(), optimizer: 'AdamWConfig | SGD | None' = None, alignment: 'int' = 16, carry: 'Mapping[str, Shape]' = <factory>, layers: 'tuple[int, ...]' = ())
    State shared by requests. `init` says where its starting values come from (default pllm.Zeros()): pllm.Computed(function), pllm.Pretrained(path), pllm.ModelCheckpoint(names), pllm.FromModule(factory), pllm.Fed(order) or pllm.Imported(). Values convert to the declared storage dtype, including trainable states.
pllm.RequestState(fields: 'Mapping[str, Shape | Field]' = <factory>, shape: 'Shape | None' = None, dtype: "torch.dtype | Literal['auto']" = 'auto', key_fields: 'tuple[KeyField, ...]' = ('request_id',), *, init: 'StateInit' = Zeros(), optimizer: 'AdamWConfig | SGD | None' = None, alignment: 'int' = 16, carry: 'Mapping[str, Shape]' = <factory>, layers: 'tuple[int, ...]' = ())
    Per-request parameters, read as tensors aligned to the current operation's rows.
pllm.Zeros()
    Every field starts at zero (the default).
pllm.Computed(function: 'Initializer')
    `function(fields, generator)` fills the named field tensors in place, once per build with a seeded generator; every admission copies the result.
pllm.Pretrained(path: 'str', placement: "Literal['device', 'host']" = 'device')
    Tensors from a safetensors file: a plain state loads the tensors named like its fields, a module state (pllm.FromModule(..., pretrained=path)) a strict state_dict. Values convert to the declared storage dtype.
pllm.ModelCheckpoint(names: 'Mapping[str, str]')
    Tensors of the served model's own checkpoint, one name per field (with `layers=`, a pattern formatted with each `{layer}`); vLLM skips them when it loads the model. The state is read-only.
pllm.FromModule(factory: 'ModuleFactory', pretrained: 'str' = '')
    The parameters of an `nn.Module` built from the model's dimensions become the state's one field, initialized as the module initializes them, or from a safetensors state_dict at `pretrained`. pllm.module(state) calls it.
pllm.Fed(order: "Literal['ordered', 'latest']")
    Filled by a producer process (VllmLlm.feed): "ordered" consumes every publication in order, "latest" only the newest. A workload declares at most one fed state.
pllm.Imported()
    Supplied by generate(batch=pllm.io.StateImport(...)) and applied before the request's first forward: one row per prompt for a RequestState, the field tensors for an InstanceState.
pllm.PositionState(initial: float = 0.0)
    One model-hidden vector per KV allocation block, initialized at token zero.
pllm.AdamW(lr: 'float', betas: 'tuple[float, float]' = (0.9, 0.999), eps: 'float' = 1e-08, weight_decay: 'float' = 0.0)
pllm.SGD(lr: 'float')
pllm.Dimensions(hidden_size: 'int', vocab_size: 'int', targets: 'tuple[ProjectionShape, ...]' = (), dtype: 'TensorDType' = <TensorDType.FLOAT32: 'float32'>, layer_count: 'int' = 0, tensor_parallel_size: 'int' = 1)
    Global model widths for workload declarations, and the probed shape of every projection the workload's adapters and @pllm.train_in_forward target.
pllm.Field(shape: 'tuple[int, ...] | Callable[[int, int], tuple[int, ...]]', dtype: "torch.dtype | Literal['auto']" = 'auto')
    A state field with an explicit dtype, or a shape computed from a projection's (in_features, out_features); a field read by an adapter gets one copy per target projection. Without a dtype it takes its state's dtype=; with neither, storage follows the declared role: accumulators use FP32, other fields use model precision.
pllm.UpdateOutput(output: 'torch.Tensor', changed: 'torch.Tensor', update_count: 'torch.Tensor', trained_rows: 'torch.Tensor', metrics: 'Mapping[str, torch.Tensor]' = <factory>)
    What a training or overlay method returns: its output and 0-d device counters.
```

### pllm.model

```text
pllm.model.ActivationPatch(state: 'str', layers: 'tuple[int, ...]', first_tokens: 'int' = 0, last_tokens: 'int' = 2, operator: "Literal['low_rank_affine', 'add']" = 'low_rank_affine', *, decode: 'bool' = False, refresh: "Literal['admission', 'step']" = 'admission', on_missing: "Literal['stop', 'hold']" = 'stop')
    Edit the hidden state at `layers` from a state, whoever writes it (ReFT-style): the first and last prompt tokens, and with decode=True every decode token.
pllm.model.Adapter(state: 'str', targets: 'tuple[Projection, ...]', weight: 'str' = 'value', operator: 'AdapterOperator' = 'dense', factors: 'tuple[str, str]' = ('a', 'b'), scale: 'float' = 1.0)
    Apply a state as a weight delta at the target projections, whoever trains it (a PEFT-style adapter).
pllm.model.CausalModel(forward: 'CausalForward | None', factors: 'tuple[torch.Tensor, ...] | None')
    The method's callable model argument; `base` is the same model without the declared overlay.
pllm.model.LayerOutputs(values: tuple[torch.Tensor, ...], start: int, layer_count: int)
    Post-layer residual streams indexed by absolute layer, negative from the model end.
pllm.model.Layers(embeddings: torch.Tensor, position: torch.Tensor, layer_count: int, generated: torch.Tensor, entries: dict[str, torch.Tensor], over: Literal['token', 'span'] = 'token', namespace: str = '')
    Public model slices and position state used inside custom_forward methods.
pllm.model.Projection(name: 'ProjectionName', layer: 'int | None' = None)
    One linear projection of the base model by its Hugging Face module name; `layer=None` means every layer.
pllm.model.PagedKV(key_cache: 'torch.Tensor', value_cache: 'torch.Tensor', block_table: 'torch.Tensor', seq_lens: 'torch.Tensor', query_start: 'torch.Tensor', query_count: 'int', read_attention: 'AttentionRead', storage_window: 'int' = 0, prompt_lens: 'torch.Tensor | None' = None)
    Borrowed paged K/V, already including the current tokens; never changes cache storage.
```

### pllm.host

```text
pllm.host.make_host(config: 'HostConfig', workload: 'Workload') -> 'HostTrainer'
    Compile `workload` without a model and admit its instance state; `train(batch)` steps it on one host batch.
pllm.host.HostConfig(device: 'torch.device', hidden_size: 'int' = 0, target_shape: 'tuple[int, ...]' = (), seed: 'int' = 0, state_page_budget_bytes: 'int' = 67108864, max_input_rows: 'int | None' = None, execution_peak_budget_bytes: 'int | None' = None)
    What pllm.host.make_host compiles a workload against: the device, the width of hidden rows in host batches and the shape of each row's target (() for one value per row), the seed of initial state values, and the state and per-step memory limits (max_input_rows bounds a batch).
pllm.host.HostTrainer(execution: 'StateExecution')
    A workload's instance state, trained on host batches without a model.
pllm.host.TrainStep(loss: 'torch.Tensor | None', parameters: 'Mapping[str, torch.Tensor]')
    One host training step: its loss (None when the train method skipped the update) and the trained state's fields after the step, by name, as owned copies ready on the caller's current stream.
pllm.host.CompletionInput(completions: 'CompletionBatch', hidden: 'torch.Tensor')
    The completions of one generate call and each one's final hidden row (float32, one row per completion), shared by host consumers; generate has finished writing `hidden` when it returns.
pllm.host.WeightedCEBatch(hidden: ~T, base_logits: ~T, labels: ~T, weights: ~T, keys: tuple[pllm.contracts.request_state.RequestStateKey, ...] = ())
    Shared [rows, ...] or independently keyed [adapters, rows, ...] weighted CE.
pllm.host.WeightedRowBatch(hidden: ~T, targets: ~T, weights: ~T)
    [rows, hidden_size] features with one target (a value or a fixed-shape tensor) and one weight per row; weight 0 skips a row.
pllm.host.CausalTrainBatch(input_ids: ~T, labels: ~T, attention_mask: Optional[~T] = None, weights: Optional[~T] = None)
    Token sequences for host training; optional floating weights have one value per sequence.
pllm.host.BarrierState: READY | BLOCKED
pllm.host.Completion(identity: pllm.contracts.request_identity.RequestIdentity, output_index: int, group_id: str, token_ids: tuple[int, ...], text: str, finish_reason: pllm.contracts.feedback.FinishReason)
pllm.host.CompletionBatch(entries: tuple[pllm.contracts.feedback.Completion, ...])
pllm.host.ControlReason: CORRECT_OUTPUT | POSITIVE_FEEDBACK | INSTANCE_STOPPED | FEEDBACK_UPDATE | RETRY_LIMIT | FEEDBACK_QUEUE_FULL | MAJORITY_ANSWER | NON_MAJORITY_ANSWER | MISSING_ANSWER | REWARD_READY | NO_REWARD
pllm.host.FinishReason: STOP | LENGTH | ERROR
pllm.host.RetryAction: ACCEPT | RETRY | STOP
pllm.host.RetryControl(completion: pllm.contracts.feedback.Completion, action: pllm.contracts.feedback.RetryAction, reason: pllm.contracts.feedback.ControlReason, retry_after_update: bool)
pllm.host.RewardOutcome(completion: pllm.contracts.feedback.Completion, reward: float | None, is_correct: bool | None, feedback_text: str, judge: str, label: str, skip_reason: str)
pllm.host.UpdateBarrier(barrier_id: str, group_id: str, state: pllm.contracts.feedback.BarrierState, request_ids: tuple[str, ...], ready_count: int, reason: pllm.contracts.feedback.ControlReason)
```

### pllm.io

```text
pllm.io.Feed(plan: 'PlanBundle', binding: 'FeedBinding', control: 'Connection', channels: 'Channels', peer: 'int', device: 'torch.device')
    Publish complete tensor fields into a declared state, with bounded backpressure.
pllm.io.StateImport(values: tuple[~T, ...])
    External tensors in field order, visible from the importing call's first step.
pllm.io.TraceEvent(sequence: 'int', host_time_ns: 'int', kind: 'TraceEventKind', invocation_id: 'int', task_id: 'TaskId | None' = None, identities: 'tuple[RequestIdentity, ...]' = (), refs: 'tuple[StateRef, ...]' = (), positions: 'tuple[int, ...]' = (), operation_id: 'int | None' = None, pin_id: 'int | None' = None, release_id: 'int | None' = None, row: 'PackedPhaseRow | None' = None, iteration: 'int | None' = None, relative_l2: 'float | None' = None, device_ordinal: 'int | None' = None, device_duration_ns: 'int | None' = None, tensor_name: 'str | None' = None, tensor_dtype: 'TensorDType | None' = None, tensor_shape: 'tuple[int, ...]' = (), tensor_data: 'str | None' = None, step_id: 'int | None' = None, source_shape: 'tuple[int, ...]' = (), dropped_rows: 'int' = 0, kernel_name: 'str | None' = None)
pllm.io.TraceSelection(tasks: 'tuple[TaskId, ...]' = (), identities: 'tuple[RequestIdentity, ...]' = (), iterations: 'tuple[int, ...]' = (), names: 'tuple[str, ...]' = (), tensors: 'bool' = False, gradients: 'bool' = False, states: 'bool' = False, max_events: 'int' = 0, max_tensor_bytes: 'int' = 32768, kernel_calls: 'int' = 0)
    Explicit observations only; empty filters match all, disabled flags retain nothing.
pllm.io.TraceSnapshot(execution_id: 'str', process_id: 'int', events: 'tuple[TraceEvent, ...]', dropped: 'int', capacity: 'int', pending_device_timings: 'int' = 0, export_dropped: 'int' = 0, schema_version: 'int' = 10)
pllm.io.checkpoint_directory(model: str, revision: str | None = None) -> str
    Resolve the local snapshot that the model loader has already downloaded.
pllm.io.checkpoint_tensors(checkpoint: str, names: tuple[str, ...], *, state: str = '') -> dict[str, torch.Tensor]
    Read named tensors from one safetensors file or a sharded model snapshot.
pllm.io.load_tensor_fields(path: 'Path', fields: 'tuple[str, ...]') -> 'tuple[torch.Tensor, ...]'
    Decode a weights-only checkpoint into the caller's explicit tensor field order.
```

### pllm.testing

```text
pllm.testing.CheckResult(passed: 'tuple[str, ...]', skipped: 'tuple[CheckSkip, ...]')
    Verified properties and explicit skips; dataclasses.asdict produces JSON-ready data.
pllm.testing.DistanceSummary(mean: 'float', p99: 'float', max: 'float', worst: 'tuple[TokenDistance, ...]')
    Token-weighted mean, nearest-rank P99, maximum, and largest distances with their positions.
pllm.testing.DistributionResult(kl_plain_to_workload: 'DistanceSummary', kl_workload_to_plain: 'DistanceSummary', tv: 'DistanceSummary', token_count: 'int', run_seconds: 'float', elapsed_seconds: 'float', failures: 'tuple[str, ...]')
    Exact diagnostic values and threshold failures; dataclasses.asdict is JSON-ready.
pllm.testing.TokenDistance(prompt_index: 'int', token_offset: 'int', value: 'float')
    A distance at a zero-based prompt index and predicted token offset in its full sequence.
pllm.testing.check_distribution(workload: 'Workload', config: 'ModelConfig', *, prompts: 'Sequence[str] | None' = None, tokens: 'int' = 16, temperature: 'float' = 0, max_mean_kl: 'float | None' = None, max_p99_kl: 'float | None' = None, max_mean_tv: 'float | None' = None, max_p99_tv: 'float | None' = None, worst: 'int' = 5) -> 'DistributionResult'
    Measure how far a workload moves the plain model's output distribution on CUDA.
pllm.testing.ContractError(failures: 'tuple[str, ...]')
    All observed contract failures, with the responsible method or framework routing named.
pllm.testing.check(workload: 'Workload', config: 'ModelConfig', *, neutral: 'Workload | None' = None) -> 'CheckResult'
    Check fixed request schedules on CPU using config shapes and the real planner/executor.
pllm.testing.check_on_gpu(workload: 'Workload', config: 'ModelConfig', *, neutral: 'Workload | None' = None) -> 'CheckResult'
    Load real engines to check planning, eight steady decode anchors, schedule independence and neutral sampling.
pllm.testing.RequestIdentity(request_id: 'str', prompt_index: 'int', sample_index: 'int', route: 'str' = '')
pllm.testing.SamplerFeedback(sampler_rows: 'PackedSamplerRowBatch', base_logits: 'torch.Tensor', sampled_tokens: 'torch.Tensor')
pllm.testing.backend_facts(hidden_size: 'int', hidden: 'tuple[PackedHiddenResourceBinding, ...]' = (), *, device: 'str' = 'cpu', capabilities: 'tuple[str, ...]' = ..., pages: 'int' = 0, max_lineages: 'int' = 2, memory_bytes: 'int' = 10000, max_rows: 'int | None' = None, dtype: 'torch.dtype' = torch.float32, vocab_size: 'int' = 0, layer_count: 'int' = 0, targets: 'Mapping[Projection, tuple[int, int]]' = {}) -> 'BackendCompileFacts'
    Facts for an eager in-process model, in place of what the vLLM runner probes.
pllm.testing.compile_execution(workload: 'Workload', facts: 'BackendCompileFacts', stream: 'torch.cuda.Stream | None' = None) -> 'StateExecution'
    Compile `workload` against `facts` and admit its instance state, as the vLLM runner does on install.
pllm.testing.final_hidden_binding(layer_count: 'int') -> 'PackedHiddenResourceBinding'
    The tap `pllm.final_hidden()` reads, named as the vLLM runner names it.
pllm.testing.hidden_binding(layer: 'int', layer_count: 'int') -> 'PackedHiddenResourceBinding'
    The tap `pllm.hidden(layer)` reads, named as the vLLM runner names it.
pllm.testing.run_guided_step(execution: 'StateExecution', value: 'StepInput', *, feedback: 'SamplerFeedback | None' = None, stream: 'torch.cuda.Stream | None' = None) -> 'tuple[tuple[PackedSamplerGuidanceBatch, ...], TensorOutput]'
    Like `run_step`, also returning additions in declaration order, followed by any guidance stage.
pllm.testing.run_step(execution: 'StateExecution', value: 'StepInput', *, feedback: 'SamplerFeedback | None' = None, stream: 'torch.cuda.Stream | None' = None) -> 'TensorOutput'
    Execute every anchor of one step, completing the sampler with `feedback` when the workload reads it.
pllm.testing.step_input(identities: 'Sequence[RequestIdentity]', hidden: 'Mapping[PackedHiddenResourceBinding, torch.Tensor] | None' = None, *, step_id: 'int' = 0, token_offset: 'int' = 0, tokens: "Sequence[Literal['prefill', 'decode']] | None" = None, sampler_order: 'Sequence[int] | None' = None, batch: 'HostBatch[torch.Tensor] | None' = None, operands: 'tuple[ProjectionOperands, ...]' = (), stream: 'torch.cuda.Stream | None' = None) -> 'StepInput'
    One packed step: row i is the one token of request `identities[i]` at `token_offset`; it decodes unless `tokens` says "prefill", which makes it the last token of the request's prompt.
pllm.testing.scheduled_step(identities: 'tuple[RequestIdentity, ...]', scheduled: 'list[int]', prompt: 'list[int]', computed: 'list[int]', hidden: 'Mapping[AttachmentHiddenResource, torch.Tensor]', *, step_id: 'int' = 0, operands: 'tuple[ProjectionOperands, ...]' = ()) -> 'StepInput'
    Build a CPU step from scheduler counts, selecting only the runners' tapped rows.
```

### pllm.benchmark

```text
pllm.benchmark.BenchmarkOptions(model: 'str' = 'Qwen/Qwen2.5-0.5B-Instruct', batch: 'int' = 8, tokens: 'int' = 256, rounds: 'int' = 3, warmup: 'int' = 1, gpu_memory_utilization: 'float' = 0.25, profile: 'Path | None' = None, cuda_graph_pool_budget_bytes: 'int | None' = None, state_page_budget_bytes: 'int' = 0, execution_peak_budget_bytes: 'int | None' = None, sampling_n: 'int' = 16, cuda_graphs: 'bool' = True, guidance_capacity: 'int | None' = 4096)
    Aligned benchmark settings; capture and state pages size themselves unless explicitly capped.
pllm.benchmark.BenchmarkResult(implementation: 'str', cold_seconds: 'float', round_seconds: 'tuple[float, ...]', output_tokens: 'int', loss_count: 'int', update_count: 'int', trained_rows: 'int', peak_allocated_bytes: 'int', baseline_allocated_bytes: 'int', plan_digest: 'str', task_implementations: 'tuple[tuple[str, str], ...]', planned_peak_memory_bytes: 'int', round_process_cpu_seconds: 'tuple[float, ...]', state_usage: 'PageBundleUsage', vanilla_tokens_per_second: 'float')
pllm.benchmark.compare(workload: 'Workload', options: 'BenchmarkOptions', *, prepare: 'Callable[[VllmLlm], HostBatch[torch.Tensor]] | None' = None, max_host_training_rows: 'int' = 1, require_training: 'bool' = False, prompt_indices: 'tuple[int, ...] | None' = None, alternatives: 'tuple[tuple[str, Workload], ...]' = ()) -> 'tuple[BenchmarkResult, ...]'
    Measure ordinary model workloads against plain vLLM; optimized arms and nonzero training are opt-in.
pllm.benchmark.throughput(llm: 'VllmLlm', prompts: 'list[str]', sampling: 'SamplingParams', *, warmup: 'int' = 1, rounds: 'int' = 3) -> 'float'
    Median output tokens per second over `rounds` generate calls after `warmup` untimed ones.
```
<!-- end of generated signatures -->
