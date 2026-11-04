# pllm reference for coding agents

Every public name in `pllm`, `pllm.model`, `pllm.host`, `pllm.io` and `pllm.testing`, grouped by the task you
are doing. Each section lists the names with a one-line meaning and a minimal example; the full signatures,
generated from the code, are at the end ([All signatures](#all-signatures)). The workflow and the rules that
cause rejections are in [SKILL.md](SKILL.md).

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
- Hidden-width tensors are in the model dtype (bfloat16 by default); `pllm.layer_logits` is float32. Cast with
  `.float()` when you need precision.
- Inputs are read-only borrowed views. Return new tensors; change state only through the effects below.

## Read a hidden state, the residual stream or per-layer logits

| Name | Meaning |
| --- | --- |
| `pllm.hidden(layer=0, *, tokens="decode")` | The MLP output of decoder layer `layer` before it is added to the residual stream, `[rows, hidden]`. Negative layers count from the end. |
| `pllm.residual(layer, *, after="layer", tokens="decode")` | The residual stream after layer `layer` (Hugging Face `hidden_states[layer + 1]`), `[rows, hidden]`. `after="attention"` reads it between the layer's attention and its MLP. Steering vectors, SAEs and the logit lens read this. |
| `pllm.final_hidden(*, tokens="decode")` | The hidden state after the final norm, which the LM head reads (Hugging Face `last_hidden_state`), `[rows, hidden]`. |
| `pllm.layer_logits(layers)` | The final norm and LM head applied to the residual stream after each of `layers`: float32 `[rows, len(layers), vocab]` (the logit lens). Layer `-1` gives the model's own logits. One row per token the sampler draws: a prompt's last token and every decode token. |

`tokens="decode"` gives one row per running request: its latest decode token. `tokens="prefill"` gives one
row per request whose prompt just finished prefilling: its last prompt token. All four inputs run inside
CUDA graphs. A reading method needs an effect (usually `@pllm.export`, next section) or nothing schedules it.

`pllm.input_tokens(history=n)` reads int64 `[rows, n + 1]` current and preceding request token IDs, with `-1` before position zero; MRv2 is required. Helpers consumed by `@pllm.custom_residual` use every scheduled row through `pllm.result`; other readers use sampled rows.

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
| `@pllm.export(*, name="value", every=1, max_rows=1, max_cols=16, enabled=True, overflow="drop")` | Deliver the method's `[rows, cols]` result to `on_export`: per step at most `max_rows` rows (the rest of that step's rows are dropped and counted in `event.dropped_rows`) and the first `max_cols` columns, every `every` steps. `max_rows * max_cols * 8` must be at most 32768 bytes. Runs after the forward by default. `overflow="block"` waits on a full delivery queue instead of dropping the event. |
| `pllm.Workload.on_export(self, event)` | Override to receive each export as a `pllm.io.TraceEvent`, in the process that runs the model, asynchronously; `close()` (or leaving `with`) drains delivery. Never call the engine from it. |
| `pllm.io.TraceEvent` | One trace record. For exports: `tensor_name` (the export name), `identities` (one `pllm.testing.RequestIdentity` per row: `prompt_index` is the prompt's position in the `generate` call, or the value from `generate(..., prompt_indices=...)`; `sample_index`; `request_id` is pllm's internal id, not vLLM's `RequestOutput.request_id`), `step_id`, `dropped_rows`, `tensor()` (an independent CPU tensor). |
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

A full hidden-width row is 896 columns for Qwen2.5-0.5B, so at most 4 rows fit one export
(4 * 896 * 8 = 28672 bytes). To capture one full row per prompt at the prompt's last token, read with
`tokens="prefill"` and pass at most `max_rows` prompts per `generate` call: prompts that finish prefill in
the same step share one export.

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
| `pllm.RequestState(fields=..., initialize=None, shape=None, dtype="auto", key_fields=("request_id",), *, optimizer=None, module=None, imported=False, carry=..., ...)` | Tensors that every request owns a private copy of, admitted with the request and freed when it finishes. Read as row-aligned tensors: a field of shape `S` reads as `[rows, *S]`. |
| `pllm.InstanceState(...)` | Tensors shared by all requests of the engine; same arguments minus `key_fields`. A field reads with its declared shape, without a row axis. |
| `pllm.Field(shape, dtype="auto")` | A field whose shape is `lambda in_features, out_features: (...)`: the widths of each target projection when a `pllm.model.Adapter` reads the state, otherwise `(hidden_size, hidden_size)`. Also gives one field an explicit `dtype`. |
| `pllm.Dimensions` | What a plain shape callable receives: `hidden_size`, `vocab_size`, `layer_count`, `target_shapes`, `dtype`. Write `{"bias": lambda m: (m.vocab_size,)}`. |
| `pllm.parameter(state, field, *, per_row=False)` | Read one field. `per_row=True` gives an instance field a broadcast row axis for row-wise math. |
| `pllm.module(state)` | Read a state declared with `module=lambda m: nn.Module(...)` as a callable module; for request state each row runs with its own parameters. |
| `pllm.version(state)` | int64 `[rows]`: how many updates each row's state has received (0 before the first). |
| `pllm.index(state)` | int64 `[rows]`: each row's slot in the state pool. |
| `pllm.PositionState(initial=0.0)` | One hidden-width vector per KV cache block, read and written only by `layers.load` / `layers.save` inside `@pllm.custom_forward`. |

Declaring fields: `shape=(n,)` declares one field named `"value"`; `fields={"w": shape, ...}` declares named
fields, each shape a tuple, a `pllm.Dimensions` callable or a `pllm.Field`; `module=` takes the fields from
the module's parameters. Initial values: `initialize(fields, generator)` fills the fields in place once
(every request then starts from the same values), `module=` uses the module's own initialization, otherwise
fields start at zero. Storage dtype `"auto"`: float32 when the state has an optimizer or a method rewrites a
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
        return pllm.RequestState(module=lambda m: nn.Linear(m.hidden_size, self.rank))

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
| `@pllm.train(*, state, reduction="mean", rows="", update_rows="", steps=1, reset="")` | Minimize the returned loss with the optimizer declared on `state`; the framework masks padding, averages, backpropagates, applies the optimizer and publishes the new version. Return a `[rows]` loss (or a scalar). `reduction="state_mean"` averages each state's rows, then sums over states. |
| `pllm.AdamW(lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)` | The optimizer for training on request rows and on `WeightedRowBatch` / `CausalTrainBatch` host batches. Declared on the state: `optimizer=pllm.AdamW(1e-3)`. |
| `pllm.SGD(lr)` | The optimizer for `WeightedCEBatch` training (`steps=`, `reset=`) and for `@pllm.train_in_forward`. |
| `@pllm.differentiate(*, inputs=(), parameters=False, input_rows="", parameter_rows="")` | Keep gradient edges through a non-loss method: `inputs` names its arguments whose gradients flow, `parameters=True` lets its parameters train too. A training method reads its result with `pllm.result(name)`. |
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
            module=lambda m: nn.Sequential(nn.Linear(m.hidden_size, self.rank), nn.Linear(self.rank, m.hidden_size)),
            optimizer=pllm.AdamW(1e-3),
        )

    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    @pllm.train(state="distiller", reduction="state_mean")
    def loss(self, shallow: torch.Tensor = pllm.hidden(0), deep: torch.Tensor = pllm.hidden(-1),
             f: nn.Module = pllm.module("distiller")) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.Attr("beta"), rows="trained")
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
| `@pllm.add_logits(*, rows="")` | Return float `[rows, vocab]`; pllm adds it to the model's logits before temperature, top-k and top-p. One row per token the sampler draws. |
| `pllm.sample_index(limit=None)` | int64 `[rows]`: each row's sample number within its prompt (0 for the first of `n`). With `limit`, `generate` rejects requests whose sample index is `limit` or more. |
| `pllm.base_logits()` | The model's logits before any workload adjustment. Readable only after sampling (in `@pllm.train` or `@pllm.on(pllm.Phase.AFTER_SAMPLE)`). |
| `pllm.sampled_tokens()` | The tokens drawn this step. Readable only after sampling. |

Read the logits you want to adjust with `pllm.layer_logits((-1,))`, not `pllm.base_logits()`. A workload has
at most one `@pllm.add_logits` or `@pllm.guide_sampling` method. For the `neutral=` check, give the workload
a strength setting whose value 0 makes the method return exactly zero.

```python
@dataclass(frozen=True)
class LogitBias(pllm.Workload):
    strength: float = 1.0
    bias = pllm.InstanceState({"b": lambda m: (m.vocab_size,)}, imported=True)

    @pllm.add_logits()
    def add(self, sample: torch.Tensor = pllm.sample_index(),
            b: torch.Tensor = pllm.parameter("bias", "b")) -> torch.Tensor:
        return self.strength * b.float().expand(sample.shape[0], -1)  # [rows, vocab]


print(testing.check(LogitBias(), CONFIG, neutral=LogitBias(strength=0.0)).passed)
```

## Guide sampling with a hidden-state prediction

| Name | Meaning |
| --- | --- |
| `@pllm.guide_sampling(*, beta, rows="")` | Return a hidden-width prediction `[rows, hidden]`. pllm projects it with the LM head and samples from `(1 + beta) * logits - beta * prediction_logits` over the tokens that survive top-k / top-p, pushing sampling away from what the prediction already explains. `beta=0` declares no sampler work. |

`rows="method"` names a selector (see the next section); unselected rows sample unchanged. See
`ShallowDistiller` in [Train during decode](#train-during-decode).

## Choose when a method runs and which rows count

| Name | Meaning |
| --- | --- |
| `pllm.Phase` | Windows of one engine step: `BEFORE_FORWARD`, `IN_FORWARD`, `AFTER_FORWARD`, `BEFORE_SAMPLE`, `AFTER_SAMPLE`, `AUTOMATIC`. |
| `@pllm.on(phase)` | Run the method in `phase`. Effects have defaults (export: after the forward; sampler methods: before sampling; request training and writes: automatic), so most methods need no `@pllm.on`. |
| `rows="method"` | On `@pllm.add_logits`, `@pllm.guide_sampling` and `@pllm.train`: names a method with marker inputs that returns a bool `[rows]` mask; only those rows apply the effect (for training: count in the loss). Request-row training alone or next to `@pllm.guide_sampling` updates every request and rejects `rows=`. |
| `pllm.Attr(name)` | A dotted path to a workload attribute, read once when the workload is built: `beta=pllm.Attr("beta")`, `layer_logits(pllm.Attr("lens"))`. Use it to make decorator arguments and marker arguments configurable. |

Each method takes at most one decorator of each kind. A value can be read only in or after the phase that
produces it: sampled tokens and `pllm.base_logits()` after sampling, the current step's hidden rows after the
forward.

## Edit activations or weights from a state

| Name | Meaning |
| --- | --- |
| `pllm.model.ActivationPatch(state, layers, first_tokens=0, last_tokens=2, operator="low_rank_affine", *, decode=False, refresh="admission", on_missing="stop")` | Edit the output of `layers` (the residual stream after them) from a state, whoever writes it. `operator="add"` adds one vector field per layer; `"low_rank_affine"` reads three fields per layer (ReFT). Applies to the first `first_tokens` and last `last_tokens` prompt tokens, and with `decode=True` to every decode token. `refresh="step"` reads the newest version every decode step. Runs eagerly (no CUDA graphs). |
| `pllm.model.Adapter(state, targets, weight="value", operator="dense", factors=("a", "b"), scale=1.0)` | Apply a state as a weight delta at the target projections, whoever trains it (PEFT-style). `operator="low_rank"` applies `scale * b @ a` from fields named by `factors`; `"dense"` and `"per_head"` read the field `weight`. |
| `pllm.model.Projection(name, layer=None)` | A linear projection by its Hugging Face module name (`"q_proj"`, `"v_proj"`, `"down_proj"`, ...); `layer=None` means every layer. |

Declare both as class attributes or typed properties. A `pllm.Field` shape callable of a state an `Adapter`
reads receives each target's `(in_features, out_features)`; the state then has one copy of each field per
target, named `a_0`, `b_0`, `a_1`, ... in target order.

```python
class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, imported=True)
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


@dataclass(frozen=True)
class PromptLoRA(pllm.Workload):
    rank: int = 4

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState({"a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                                  "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank))}, imported=True)

    @property
    def apply(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12))
        return pllm.model.Adapter("lora", targets, operator="low_rank")


print(pllm.explain(Steering(), CONFIG).eager_reason)
```

## Import or feed state from outside

| Name | Meaning |
| --- | --- |
| `pllm.io.StateImport(values, keys=())` | Tensors for an `imported=True` state, in field order, passed as `llm.generate(..., batch=StateImport(...))`. An instance state gets a new version that this call's requests read; later calls without `batch=` keep it. For request state the values carry a leading prompt axis and are written once, before each request's first forward. |
| `imported=True` | On `RequestState` / `InstanceState`: the values come from `StateImport`, not from the workload. |
| `checkpoint={field: tensor_name}` | On an `InstanceState` without optimizer: load the fields from the model checkpoint's tensors (with `layers=(...)` the names may contain `{layer}`). |
| `pllm.io.load_tensor_fields(path, fields)` | Load named tensors from a weights-only checkpoint file, in the order of `fields`. |
| `pllm.io.Feed` | `llm.feed(state)` returns it for an `InstanceState(feed="ordered" or "latest")` filled by a producer process: `publish(**fields)`, `missing()`, `end()`, `close()`. See `examples/video_session.py`. |

```python gpu
del seen  # the StreamNorm example above is done
vector = torch.nn.functional.normalize(torch.randn(896), dim=0) * 8  # Qwen2.5-0.5B: hidden size 896

with pllm.make_llm(GPU_CONFIG, workload=Steering()) as llm:
    for strength in (0.0, 1.0):
        out = llm.generate(["I think that today"], GREEDY, batch=pllm.io.StateImport((strength * vector,)))
        print(strength, repr(out[0].outputs[0].text))
    print(repr(llm.diagnostics().eager_reason))  # activation patches run eagerly
```

## Rewrite a residual site

| Name | Meaning |
| --- | --- |
| `@pllm.custom_residual(*, site, layers=None, read="", streams=1, state="", chunk=None)` | Own the residual update at `site` of `layers` (default every layer). `site="attention"` / `"mlp"` / `"both"`: `f(x, y, ...) -> stream` replaces `x + y` (x: the stream before the sublayer, y: its branch output); `"value"`: maps a full-attention layer's value rows `[rows, kv_heads, head_dim]`; `"final"`: maps the stream entering the final norm. `streams=n` carries `[rows, n, hidden]` streams (hyper-connections) until a `"final"` method collapses them. With `state=` (a one-field `RequestState`) and `chunk=`, the method is a stateful sublayer returning `(stream, next carry)`. |
| `pllm.depth()` | Inside a `custom_residual` method: the layer it runs at, a Python int (`2 * layer`, plus 1 for the MLP, with `site="both"`). |
| `pllm.value(layer)` | Inside a `custom_residual` method: the value rows layer `layer` computed for the same token in this forward. |

The site tensors come first, without defaults; later inputs may be `pllm.parameter`, `pllm.module`,
`pllm.depth()`, `pllm.value(layer)` and `pllm.residual(layer)` (the same token, earlier in this forward).
Tensors are in the model dtype. A method returning the plain update (`x + y`, `v`) reproduces the model
bitwise. Costs: the forward runs eagerly, `tensor_parallel_size=1`, prefix caching off.

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
| `@pllm.custom_forward(*, tokens="all", over="token")` | Replace the forward of the selected tokens (`"prefill"`, `"decode"`, `"all"`) with the method `f(self, layers, ...) -> stream`, returning the residual stream the final norm reads. Inputs after `layers` may be `pllm.parameter` / `pllm.module` of an `InstanceState`, or `pllm.Attr`. |
| `pllm.model.Layers` | The method's `layers` argument. `layers(start=0, stop=None, hidden=None, *, kv="commit", active=None)` runs layers `start..stop-1` and returns their outputs; `kv="peek"` attends without writing the KV cache; `active` (bool `[rows, 1]`, only with `tokens="decode"`) freezes the rows that are False (early exit). Also `embeddings`, `generated` (bool `[rows, 1]`), `load(name)` / `save(name, value)` for a `pllm.PositionState`, `apply_updates()` for `@pllm.train_in_forward`. |
| `pllm.model.LayerOutputs` | What `layers(...)` returns: the post-layer residual streams, indexed by absolute layer (`out[-1]` is the last layer run). |
| `pllm.scan(step, *, carry, over, chunk, invariants=(), valid_rows=None)` | Run `step(carry, *slices, *invariants) -> (carry, readout)` over `over` in chunks of `chunk` rows, in order, returning the final carry and the readouts. |

A custom forward runs eagerly, cannot be combined with methods reading hidden rows (`pllm.hidden`,
`pllm.residual`, `pllm.layer_logits`, `pllm.final_hidden`) or with `pllm.model.Adapter`, and needs
`tensor_parallel_size=1`. Bounded Python `for` loops over workload settings are unrolled when traced.

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
                                 initialize=start_small, optimizer=pllm.SGD(1e-3))

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
`llm.generate(..., batch=...)` (or to a `make_host` execution) instead of on request rows.

| Name | Meaning |
| --- | --- |
| `pllm.batch(field)` | One field of the host batch. The fields a `@pllm.train` method reads pick the batch type and the optimizer. |
| `pllm.host.WeightedRowBatch(hidden, targets, weights)` | `[rows, hidden]` features, one target and one weight per row (weight 0 skips a row); one AdamW step per batch. |
| `pllm.host.CausalTrainBatch(input_ids, labels, attention_mask=None)` | int64 `[sequences, tokens]` ids and labels for causal LM training through `pllm.causal_model`; one AdamW step per batch. |
| `pllm.host.WeightedCEBatch(hidden, base_logits, labels, weights)` | Weighted cross-entropy rows; trains with `pllm.SGD`, `steps=` steps per state after the `reset=` method restores it. `WeightedCEBatch.for_prompts(..., prompt_indices)` keys one batch per prompt. |
| `pllm.causal_model(adapter)` | A `pllm.model.CausalModel` with the named `pllm.model.Adapter` applied: `model(ids, attention_mask=...)` returns logits whose gradients reach the adapter's state; `model.base(...)` skips it. |
| `pllm.host.make_host(config, workload)` | Compile a batch-training workload without a model. `admit(())`, then `step(StepInput.of_batch(batch))` per batch (returns a `StepResult`), then `retire(())`. |
| `pllm.host.HostConfig(device, hidden_size=0, target_shape=(), seed=0, state_memory_bytes=..., max_input_rows=None, memory_budget_bytes=None)` | What `make_host` compiles against: device, hidden width of the batches and the shape of each row's target. |
| `pllm.host.StepInput` | One step's input; for host steps build it with `StepInput.of_batch(batch)`. |
| `pllm.host.StepResult` | A host step's result: `value`, `loss`, `metrics`, and `parameters` (the updated fields by name). |
| `pllm.host.CompletionInput` | What a feedback callable receives: `completions` (a `CompletionBatch`) and `hidden` (each completion's final hidden row). |
| `pllm.host.Completion`, `pllm.host.CompletionBatch` | One sampled completion (`identity`, `output_index`, `group_id`, `token_ids`, `text`, `finish_reason`), and a batch of them (`entries`). |
| `pllm.host.FinishReason` | `STOP`, `LENGTH` or `ERROR`. |
| `pllm.host.RetryControl(completion, action, reason, retry_after_update)` | The callable's decision for one completion; `action` is a `pllm.host.RetryAction` (`ACCEPT`, `RETRY`, `STOP`), `reason` a `pllm.host.ControlReason`. |
| `pllm.host.UpdateBarrier(barrier_id, group_id, state, request_ids, ready_count, reason)` | A group of requests that retry together once an update is ready; `state` is a `pllm.host.BarrierState` (`READY`, `BLOCKED`). |
| `pllm.host.RewardOutcome(completion, reward, is_correct, feedback_text, judge, label, skip_reason)` | A scored completion, for the callable's own result. |

`llm.generate_feedback(prompts, sampling, *, feedback, group_id, max_rounds)` generates, calls
`feedback(completion_input)`, and retries the prompts whose completions the result marks `RETRY`, for at
most `max_rounds` rounds; it yields `(completion_input, result)` per round. The result is any object with
`retry_controls`, `update_barriers` and `update_count`; returning `None` ends the loop. See
`pllm/workloads/ttrl.py` and tutorial 05.

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


execution = pllm.host.make_host(pllm.host.HostConfig(torch.device("cpu"), hidden_size=896), RewardHead())
execution.admit(())
batch = pllm.host.WeightedRowBatch(torch.randn(4, 896), torch.ones(4), torch.ones(4))
result = execution.step(pllm.host.StepInput.of_batch(batch))
print(result.loss is not None, tuple(result.parameters["w"].shape))
execution.retire(())
```

## Configure the engine and execution

| Name | Meaning |
| --- | --- |
| `pllm.make_llm(config, *, workload=None)` | A vLLM engine running `workload`; use as a context manager. Methods: `generate(prompts, sampling_params, *, batch=None, sample_index=0, prompt_indices=None, routes=None)`, `generate_completions`, `generate_feedback`, `diagnostics()` (`eager_reason`, `update_count`, `trained_rows`, `loss_count`, ...), `trace_snapshot()`, `select_trace(...)`, `feed(name)`, `close()`. Without a workload it is plain vLLM. |
| `pllm.ModelConfig(model, dtype="bfloat16", max_model_len=512, max_num_seqs=8, max_num_batched_tokens=512, gpu_memory_utilization=0.5, seed=0, ..., multiprocessing=False, execution=ExecutionOptions(), cuda_graphs="auto", kv_offload=None, tensor_parallel_size=1, ...)` | The model and engine. `cuda_graphs="auto"` uses CUDA graphs when the workload allows it (else eager, reason in `diagnostics().eager_reason`); `True` requires them. `multiprocessing=True` needs the workload class in an importable module. |
| `pllm.ExecutionOptions(max_training_rows=1, capture_memory_bytes=None, state_memory_bytes=0, memory_budget_bytes=None, allow_numerical_transforms=True, tuning="heuristic", publish_lag=1)` | Framework policy. `publish_lag=0`: updates at step n are read at n + 1 and training no longer overlaps the next forward. Memory knobs bound what pllm reserves. |
| `pllm.KVOffload(connector="OffloadingConnector", cpu_bytes=536870912, prompt_only=False)` | Offload complete KV pages to CPU memory (`ModelConfig(kv_offload=...)`). |
| `pllm.Workload` | The base class. Subclass it (a frozen dataclass when it has settings); put declarations and tensor math in it, nothing else. |

## Inspect the plan and check contracts

| Name | Meaning |
| --- | --- |
| `pllm.explain(workload, config)` | Plan on CPU from `config.json` without weights or CUDA: per method its timing anchor, CUDA stream lane and graph capture; why the forward is eager; state bytes; peak memory pllm reserves; host syncs in your math; timing settings. `str()` prints a table; `dataclasses.asdict()` gives JSON-ready data (`eager_reason`, `members`, `states`, `peak_memory_bytes`, `budget_errors`, ...). |
| `pllm.testing.check(workload, config, *, neutral=None)` | Run fixed three-request schedules on CPU through the real planner and executor: row permutation, request isolation, mixed prefill/decode, chunked prompts, preemption replay, publish_lag versions, and with `neutral=` that the disabled workload adds exactly zero. Returns a `CheckResult`; raises `ContractError`. |
| `pllm.testing.check_on_gpu(workload, config, *, neutral=None)` | Load real engines: planning, eight steady decode steps, forced KV eviction and recompute, and with `neutral=` greedy and seeded sampling bitwise equal to the plain model. Run it in its own process with `multiprocessing=False`, `max_num_seqs >= 2`, `max_model_len >= 32`. |
| `pllm.testing.CheckResult` | `passed` (names of verified properties) and `skipped` (each with `property` and `reason`); `str()` prints a table. |
| `pllm.testing.ContractError` | An `AssertionError` whose `failures` lists every discrepancy, naming the method that mixed rows or the framework routing that differs. |

```python
report = pllm.explain(ShallowDistiller(), CONFIG)
print(repr(report.eager_reason), [(state.name, state.reserved_bytes) for state in report.states])
print(testing.check(Lens(), CONFIG))
```

## Step a workload by hand in tests

For unit tests of your own math without a model, compile against stand-in facts and feed one packed step.

| Name | Meaning |
| --- | --- |
| `pllm.testing.backend_facts(hidden_size, hidden=(), ..., *, max_lineages=2, memory_bytes=10000, max_rows=None, dtype=..., vocab_size=0, layer_count=0, targets=())` | Facts of an eager in-process model in place of what the vLLM runner probes. |
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

<!-- signatures: generated by tools/check_docs.py --write-signatures; do not edit -->
### pllm

```text
pllm.make_llm(config: 'ModelConfig', *, workload: 'BackendWorkload[D, R] | None' = None) -> 'VllmLlm[D, R] | VllmLlm[None, None]'
    A context-managed vLLM engine for `config`, optionally running `workload` inside it.
pllm.explain(workload: 'Workload', config: 'ModelConfig') -> 'ExplainResult'
    Plan using config.json and shape-only tensor tracing; never load base-model weights or allocate CUDA storage.
pllm.scan(step: collections.abc.Callable[..., tuple[torch.Tensor, torch.Tensor]], *, carry: torch.Tensor, over: tuple[torch.Tensor, ...], chunk: int, invariants: tuple[torch.Tensor, ...] = (), valid_rows: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]
    Run ``step(carry, *slices, *invariants)`` in chunk order, returning exclusive readouts.
pllm.custom_forward(*, tokens: 'Tokens' = 'all', over: 'Over' = 'token') -> 'Callable[[F], F]'
    Replace the forward of the selected tokens with this method's program over model layer slices and tensor math.
pllm.custom_residual(*, site: 'Site', layers: 'tuple[int, ...] | Attr | None' = None, read: 'str' = '', streams: 'int | Attr' = 1, state: 'str' = '', chunk: 'int | Attr | None' = None) -> 'Callable[[F], F]'
    Own the residual update at `site` of `layers` (default: every layer) with this method.
pllm.depth() -> 'Any'
    Inside a @pllm.custom_residual method, the index of the place it runs at, a Python int: the layer, or 2 * layer (+ 1 for the MLP) with site="both".
pllm.value(layer: 'int | Attr') -> 'Any'
    Inside a @pllm.custom_residual method, the value rows [rows, kv_heads, head_dim] layer `layer` computed for the same token in this forward (after its own value rewrite).
pllm.ModelConfig(model: str, dtype: Literal['bfloat16', 'float16', 'float32'] = 'bfloat16', max_model_len: int = 512, max_num_seqs: int = 8, max_num_batched_tokens: int = 512, gpu_memory_utilization: float = 0.5, seed: int = 0, trust_remote_code: bool = False, enable_prefix_caching: bool = False, enable_chunked_prefill: bool = False, generated_kv: Literal['reuse', 'isolate'] = 'isolate', revision: str | None = None, multiprocessing: bool = False, runner: Literal['mrv1', 'mrv2'] = 'mrv2', execution: pllm.config.ExecutionOptions = ..., cuda_graphs: Union[Literal['auto'], bool] = 'auto', kv_offload: pllm.config.KVOffload | None = None, tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1, data_parallel_size: int = 1)
pllm.ExecutionOptions(max_training_rows: int = 1, capture_memory_bytes: int | None = None, state_memory_bytes: int = 0, memory_budget_bytes: int | None = None, allow_numerical_transforms: bool = True, tuning: Literal['heuristic', 'measured'] = 'heuristic', publish_lag: int = 1)
    Framework execution policy; lag counts post-sampler steps hidden from readers.
pllm.KVOffload(connector: Literal['OffloadingConnector', 'SimpleCPUOffloadConnector'] = 'OffloadingConnector', cpu_bytes: int = 536870912, prompt_only: bool = False)
    Batched same-engine CPU offload of complete allocation pages.
pllm.Workload()
    Python methods and effects compiled once, with independent sub-workloads in dataclass fields.
pllm.Phase: BEFORE_FORWARD | AFTER_FORWARD | BEFORE_SAMPLE | AFTER_SAMPLE | IN_FORWARD | AUTOMATIC
pllm.on(phase: 'Phase | Attr') -> 'Callable[[F], F]'
    Run the method in `phase`; needed for methods without an effect or to move an effect's phase.
pllm.write(*, state: 'str', fields: 'tuple[str, ...]') -> 'Callable[[F], F]'
    Atomically publish a tuple of computed fields; request values have a leading request axis.
pllm.export(*, name: 'str | Attr' = 'value', every: 'int | Attr' = 1, max_rows: 'int | Attr' = 1, max_cols: 'int | Attr' = 16, enabled: 'bool | Attr' = True, overflow: "Literal['drop', 'block']" = 'drop') -> 'Callable[[F], F]'
    Deliver bounded observations of executed forward rows to on_export.
pllm.add_logits(*, rows: 'str' = '') -> 'Callable[[F], F]'
    Add the returned tensor to the logits before sampling.
pllm.guide_sampling(*, beta: 'float | Attr', rows: 'str' = '') -> 'Callable[[F], F]'
    Push sampling away from the returned hidden-state prediction for positive `beta`.
pllm.train(*, state: 'str', update_rows: 'str' = '', reduction: 'TrainReduction' = 'state_mean', steps: 'int | Attr' = 1, reset: 'str' = '', rows: 'str' = '') -> 'Callable[[F], F]'
    Minimize the returned loss with `state`'s optimizer; updates publish after sampling by default.
pllm.differentiate(*, inputs: 'tuple[str, ...]' = (), parameters: 'bool' = False, input_rows: 'str' = '', parameter_rows: 'str' = '') -> 'Callable[[F], F]'
    Keep derivative edges; parameters may stay frozen while inputs differentiate.
pllm.train_in_forward(*, state: 'str', targets: 'tuple[Projection, ...] | Attr', weight: 'str' = 'value', operator: 'AdapterOperator | Attr' = 'dense', factors: 'tuple[str, str] | Attr' = ('a', 'b'), scale: 'float | Attr' = 1.0, tokens: 'Tokens | Attr' = 'prefill', visibility: "Literal['forward_pinned', 'exact'] | Attr" = 'forward_pinned', chunk: 'int | Attr | None' = None, flush: "Literal['prompt_end'] | None" = None) -> 'Callable[[F], F]'
    Train a request state inside the forward and add its residual at each target projection.
pllm.Attr(name: 'str')
    A reference to a workload attribute by dotted path, e.g. pllm.Attr("config.chunk_size"), resolved once when the workload is built.
pllm.hidden(layer: 'int | Attr' = 0, *, tokens: "Literal['decode', 'prefill']" = 'decode') -> 'Any'
    The MLP output before the residual add in decoder layer `layer` (negative from the end).
pllm.residual(layer: 'int | Attr', *, after: "Literal['layer', 'attention']" = 'layer', tokens: "Literal['decode', 'prefill']" = 'decode') -> 'Any'
    The residual stream after decoder layer `layer` (negative from the end): the embedding plus every earlier sublayer's output, which the next sublayer normalizes. This is what steering vectors, sparse autoencoders and the logit lens read; pllm.hidden(layer) is only the layer's MLP output before it is added to this stream. after="attention" reads the stream between the layer's attention and its MLP. One row per request, chosen as in pllm.hidden.
pllm.layer_logits(layers: 'tuple[int, ...] | Attr') -> 'Any'
    The model's final norm and LM head applied to the residual stream after each of `layers` (negative from the end), as float32 [rows, len(layers), vocab]: the logit lens. The last layer's logits are the model's own. One row for each token the sampler draws: a prompt's last token and every decode token.
pllm.final_hidden(*, tokens: "Literal['decode', 'prefill']" = 'decode') -> 'Any'
    The hidden state after the model's final norm, which the LM head reads (Hugging Face's last_hidden_state); one row per request, chosen as in pllm.hidden.
pllm.parameter(state: 'str', field: 'str', *, per_row: 'bool' = False) -> 'Any'
    One tensor field of a declared state, aligned to the operation's rows for request state.
pllm.module(state: 'str') -> 'Any'
    A state declared with `module=`, called like the module; request state uses each row's own parameters.
pllm.version(state: 'str') -> 'Any'
    How many updates each row's state has received (0 before the first update).
pllm.index(state: 'str') -> 'Any'
    Each row's slot in the state pool, for indexing pooled tensors.
pllm.input_tokens(*, history: 'int' = 0) -> 'Any'
    Int64 [rows, history + 1] token ids: current token first, then lookbacks; -1 before the request starts.
pllm.sample_index(limit: 'int | Attr | None' = None) -> 'Any'
    The sample index of each row's request: 0 for a prompt's first sample.
pllm.result(operation: 'str') -> 'Any'
    The value another operation of this workload returned in the same step.
pllm.base_logits() -> 'Any'
    The model's logits before any workload adjustment.
pllm.sampled_tokens() -> 'Any'
    The tokens the sampler drew this step.
pllm.operand(name: 'OperandName | Attr') -> 'Any'
    A projection input/target/weight, input token embedding, or scalar prompt-end bool.
pllm.carry(source: '_Source', *, state: 'str', field: 'str') -> 'Any'
    Read `source` and, when the training update publishes, write its value into the carried `state.field`.
pllm.causal_model(adapter: 'str') -> 'Any'
    The causal LM with the named pllm.model.Adapter applied, as a pllm.model.CausalModel; .base skips it.
pllm.batch(field: "Literal['hidden', 'base_logits', 'labels', 'weights', 'targets', 'input_ids', 'attention_mask']") -> 'Any'
    One field of the host batch passed to generate.
pllm.InstanceState(fields: 'Mapping[str, Shape | Field]' = <factory>, initialize: 'Callable[[Tensors, torch.Generator], None] | None' = None, shape: 'Shape | None' = None, dtype: "torch.dtype | Literal['auto'] | None" = 'auto', *, optimizer: 'AdamWConfig | SGD | None' = None, alignment: 'int' = 16, feed: "Literal['ordered', 'latest'] | None" = None, carry: 'Mapping[str, Shape]' = <factory>, module: 'ModuleFactory | None' = None, imported: 'bool' = False, pretrained: 'str' = '', placement: "Literal['device', 'host']" = 'device', checkpoint: 'Mapping[str, str]' = <factory>, layers: 'tuple[int, ...]' = ())
    State shared by requests; pretrained= supplies safetensors starting values instead of initialize=.
pllm.RequestState(fields: 'Mapping[str, Shape | Field]' = <factory>, initialize: 'Callable[[Tensors, torch.Generator], None] | None' = None, shape: 'Shape | None' = None, dtype: "torch.dtype | Literal['auto'] | None" = 'auto', key_fields: 'tuple[KeyField, ...]' = ('request_id',), *, optimizer: 'AdamWConfig | SGD | None' = None, alignment: 'int' = 16, feed: "Literal['ordered', 'latest'] | None" = None, carry: 'Mapping[str, Shape]' = <factory>, module: 'ModuleFactory | None' = None, imported: 'bool' = False, pretrained: 'str' = '', placement: "Literal['device', 'host']" = 'device', checkpoint: 'Mapping[str, str]' = <factory>, layers: 'tuple[int, ...]' = ())
    Per-request parameters, read as tensors aligned to the current operation's rows.
pllm.PositionState(initial: float = 0.0)
    One model-hidden vector per KV allocation block, initialized at token zero.
pllm.AdamW(lr: 'float', betas: 'tuple[float, float]' = (0.9, 0.999), eps: 'float' = 1e-08, weight_decay: 'float' = 0.0)
pllm.SGD(lr: 'float')
pllm.Dimensions(hidden_size: 'int', vocab_size: 'int', target_shapes: 'tuple[tuple[int, int], ...]' = (), dtype: 'TensorDType' = <TensorDType.FLOAT32: 'float32'>, layer_count: 'int' = 0)
pllm.Field(shape: 'tuple[int, ...] | Callable[[int, int], tuple[int, ...]]', dtype: "torch.dtype | Literal['auto'] | None" = 'auto')
    A state field with an explicit dtype, or a shape computed from a projection's (in_features, out_features); a field read by an adapter gets one copy per target projection. Storage follows its declared role by default: accumulators use FP32, other fields use model precision. An explicit dtype overrides this; None follows the model.
pllm.UpdateOutput(output: 'torch.Tensor', changed: 'torch.Tensor', update_count: 'torch.Tensor', trained_rows: 'torch.Tensor', metrics: 'Mapping[str, torch.Tensor]' = <factory>)
    What a training or overlay method returns: its output and 0-d device counters.
```

### pllm.model

```text
pllm.model.ActivationPatch(state: 'str', layers: 'tuple[int, ...]', first_tokens: 'int' = 0, last_tokens: 'int' = 2, operator: "Literal['low_rank_affine', 'add']" = 'low_rank_affine', *, decode: 'bool' = False, refresh: "Literal['admission', 'step']" = 'admission', on_missing: "Literal['stop', 'hold']" = 'stop')
    Edit the hidden state at `layers` from a state, whoever writes it (ReFT-style): the first and last prompt tokens, and with decode=True every decode token.
pllm.model.Adapter(state: 'str', targets: 'tuple[Projection, ...]', weight: 'str' = 'value', operator: 'AdapterOperator' = 'dense', factors: 'tuple[str, str]' = ('a', 'b'), scale: 'float' = 1.0)
    Apply a state as a weight delta at the target projections, whoever trains it (a PEFT-style adapter).
pllm.model.CausalModel(forward: 'CausalForward | None', factors: 'tuple[torch.Tensor, ...]')
    The method's callable model argument and its overlay-free base forward.
pllm.model.LayerOutputs(values: tuple[torch.Tensor, ...], start: int, layer_count: int)
    Post-layer residual streams indexed by absolute layer, negative from the model end.
pllm.model.Layers(embeddings: torch.Tensor, position: torch.Tensor, layer_count: int, generated: torch.Tensor, entries: dict[str, torch.Tensor], over: Literal['token', 'span'] = 'token', namespace: str = '')
    Public model slices and position state used inside custom_forward methods.
pllm.model.Projection(name: 'ProjectionName', layer: 'int | None' = None)
    One linear projection of the base model by its Hugging Face module name; `layer=None` means every layer.
```

### pllm.host

```text
pllm.host.make_host(config: 'HostConfig', workload: 'Workload') -> 'StateExecution[StepInput, TensorFrame, TensorOutput]'
    Compile `workload` without a model; each `step(StepInput.of_batch(batch))` trains on one host batch and returns a StepResult.
pllm.host.HostConfig(device: 'torch.device', hidden_size: 'int' = 0, target_shape: 'tuple[int, ...]' = (), seed: 'int' = 0, state_memory_bytes: 'int' = 67108864, max_input_rows: 'int | None' = None, memory_budget_bytes: 'int | None' = None)
    What pllm.host.make_host compiles a workload against: the device, the width of hidden rows in host batches and the shape of each row's target (() for one value per row), the seed of initial state values, and the state and per-step memory limits (max_input_rows bounds a batch).
pllm.host.StepInput(rows: 'PackedPhaseRowBatch', sampler_rows: 'PackedSamplerRowBatch', hidden: 'HiddenRowBatch | None' = None, operands: 'tuple[ProjectionOperands, ...]' = (), batch: 'HostBatch[torch.Tensor] | None' = None, input_tokens: 'torch.Tensor | None' = None, hidden_readiness: 'CudaStreamReadiness | None' = None, batch_readiness: 'CudaStreamReadiness | None' = None, import_rows: 'Mapping[str, int]' = <factory>, overlay_updates: 'Mapping[tuple[RequestIdentity, Projection], UpdateOutput]' = <factory>)
pllm.host.StepResult(value: 'torch.Tensor | None', invocation: 'InvocationOutput', entries: 'tuple[OverlayEntry, ...]' = (), input_gradients: 'Mapping[tuple[str, str], torch.Tensor]' = <factory>, parameter_gradient_fields: 'Mapping[StateRef, tuple[torch.Tensor, ...]]' = <factory>, metrics: 'Mapping[str, torch.Tensor]' = <factory>, parameters: 'Mapping[str, torch.Tensor]' = <factory>)
pllm.host.CompletionInput(completions: 'CompletionBatch', hidden: 'torch.Tensor', readiness: 'CudaStreamReadiness | None' = None)
    One owned completion/hidden boundary, shared by host consumers.
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
pllm.io.TraceEvent(sequence: 'int', host_time_ns: 'int', kind: 'TraceEventKind', invocation_id: 'int', task_id: 'TaskId | None' = None, identities: 'tuple[RequestIdentity, ...]' = (), refs: 'tuple[StateRef, ...]' = (), operation_id: 'int | None' = None, pin_id: 'int | None' = None, release_id: 'int | None' = None, row: 'PackedPhaseRow | None' = None, iteration: 'int | None' = None, relative_l2: 'float | None' = None, device_ordinal: 'int | None' = None, device_duration_ns: 'int | None' = None, tensor_name: 'str | None' = None, tensor_dtype: 'TensorDType | None' = None, tensor_shape: 'tuple[int, ...]' = (), tensor_data: 'str | None' = None, step_id: 'int | None' = None, source_shape: 'tuple[int, ...]' = (), dropped_rows: 'int' = 0, kernel_name: 'str | None' = None)
pllm.io.TraceSelection(tasks: 'tuple[TaskId, ...]' = (), identities: 'tuple[RequestIdentity, ...]' = (), iterations: 'tuple[int, ...]' = (), names: 'tuple[str, ...]' = (), tensors: 'bool' = False, gradients: 'bool' = False, max_events: 'int' = 0, max_tensor_bytes: 'int' = 32768, kernel_calls: 'int' = 0)
    Explicit observations only; empty filters match all, disabled flags retain nothing.
pllm.io.TraceSnapshot(execution_id: 'str', process_id: 'int', events: 'tuple[TraceEvent, ...]', dropped: 'int', capacity: 'int', pending_device_timings: 'int' = 0, export_dropped: 'int' = 0, schema_version: 'int' = 9)
pllm.io.load_tensor_fields(path: 'Path', fields: 'tuple[str, ...]') -> 'tuple[torch.Tensor, ...]'
    Decode a weights-only checkpoint into the caller's explicit tensor field order.
```

### pllm.testing

```text
pllm.testing.CheckResult(passed: 'tuple[str, ...]', skipped: 'tuple[CheckSkip, ...]')
    Verified properties and explicit skips; dataclasses.asdict produces JSON-ready data.
pllm.testing.ContractError(failures: 'tuple[str, ...]')
    All observed contract failures, with the responsible method or framework routing named.
pllm.testing.check(workload: 'Workload', config: 'ModelConfig', *, neutral: 'Workload | None' = None) -> 'CheckResult'
    Check fixed request schedules on CPU using config shapes and the real planner/executor.
pllm.testing.check_on_gpu(workload: 'Workload', config: 'ModelConfig', *, neutral: 'Workload | None' = None) -> 'CheckResult'
    Load real engines to check planning, eight steady decode anchors, replay and neutral sampling.
pllm.testing.RequestIdentity(request_id: 'str', prompt_index: 'int', sample_index: 'int', route: 'str' = '')
pllm.testing.SamplerFeedback(sampler_rows: 'PackedSamplerRowBatch', base_logits: 'torch.Tensor', sampled_tokens: 'torch.Tensor')
pllm.testing.backend_facts(hidden_size: 'int', hidden: 'tuple[PackedHiddenResourceBinding, ...]' = (), timeline: 'SchedulerTimeline' = ..., provider: 'PageBundleProviderFacts | None' = ..., *, max_lineages: 'int' = 2, memory_bytes: 'int' = 10000, max_rows: 'int | None' = None, dtype: 'TensorDType' = <TensorDType.FLOAT32: 'float32'>, vocab_size: 'int' = 0, layer_count: 'int' = 0, targets: 'tuple[ProjectionShape, ...]' = ()) -> 'BackendCompileFacts'
    Facts for an eager in-process model, in place of what the vLLM runner probes.
pllm.testing.compile_execution(workload: 'BackendWorkload[DataT, ResultT]', facts: 'BackendCompileFacts', stream: 'torch.cuda.Stream | None' = None) -> 'StateExecution[StepInput, DataT, ResultT]'
    Compile `workload` against `facts` and admit its instance state, as the vLLM runner does on install.
pllm.testing.final_hidden_binding(layer_count: 'int') -> 'PackedHiddenResourceBinding'
    The tap `pllm.final_hidden()` reads, named as the vLLM runner names it.
pllm.testing.hidden_binding(layer: 'int', layer_count: 'int') -> 'PackedHiddenResourceBinding'
    The tap `pllm.hidden(layer)` reads, named as the vLLM runner names it.
pllm.testing.run_guided_step(execution: 'StateExecution[StepInput, DataT, ResultT]', value: 'StepInput', *, feedback: 'SamplerFeedback | None' = None, stream: 'torch.cuda.Stream | None' = None) -> 'tuple[tuple[PackedSamplerGuidanceBatch, ...], ResultT]'
    Like `run_step`, also returning additions in declaration order, followed by any guidance stage.
pllm.testing.run_step(execution: 'StateExecution[StepInput, DataT, ResultT]', value: 'StepInput', *, feedback: 'SamplerFeedback | None' = None, stream: 'torch.cuda.Stream | None' = None) -> 'ResultT'
    Execute every anchor of one step, completing the sampler with `feedback` when the workload reads it.
pllm.testing.step_input(identities: 'Sequence[RequestIdentity]', hidden: 'Mapping[PackedHiddenResourceBinding, torch.Tensor] | None' = None, *, step_id: 'int' = 0, token_offset: 'int' = 0, tokens: "Sequence[Literal['prefill', 'decode']] | None" = None, sampler_order: 'Sequence[int] | None' = None, batch: 'HostBatch[torch.Tensor] | None' = None, operands: 'tuple[ProjectionOperands, ...]' = (), stream: 'torch.cuda.Stream | None' = None) -> 'StepInput'
    One packed step: row i belongs to `identities[i]` and decodes unless `tokens` says otherwise.
pllm.testing.scheduled_step(identities: 'tuple[RequestIdentity, ...]', scheduled: 'list[int]', prompt: 'list[int]', computed: 'list[int]', hidden: 'Mapping[AttachmentHiddenResource, torch.Tensor]', *, step_id: 'int' = 0, operands: 'tuple[ProjectionOperands, ...]' = ()) -> 'StepInput'
    Build a CPU step from scheduler counts, selecting only the runners' tapped rows.
```
<!-- end of generated signatures -->
