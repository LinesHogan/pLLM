# Rules for pLLM workloads and why they exist

pLLM reads a workload's declarations once, when the engine is built, and from them plans where every
method runs, how much memory it reserves, what it captures in CUDA graphs and when state versions become
visible. The rules below are what that planning relies on. Each states its reason so you can tell which
change satisfies it. The messages pLLM raises when a rule is broken, and their fixes, are in
[pitfalls.md](pitfalls.md).

## Invariants

| Rule | Reason |
| --- | --- |
| Every parameter after `self` has a marker default (`pllm.hidden(0)`, `pllm.parameter("s", "w")`, ...). | pLLM decides inputs from defaults, never from parameter names or annotations, so nothing is guessed. |
| A method's rows are independent: compute along `dim=-1`, never across `dim=0`. | Dimension 0 packs unrelated requests in an order that changes every step; a row may only depend on its own request. `testing.check` permutes and mixes rows to catch violations. |
| Methods never synchronize with the host (`.item()`, `.tolist()`, `.cpu()`, `print(tensor)`, Python `if` on tensor values). | Methods are shape-traced to plan memory, captured in CUDA graphs and run on side streams that overlap the forward; a sync either fails tracing or stalls the engine. |
| Inputs and parameters are read-only. | They are views shared with other methods and with pLLM's state versions; an in-place write corrupts what others read. |
| State changes only through effects: `@pllm.write`, `@pllm.train`, `@pllm.train_in_forward`, `init=` sources, imports and feeds. | pLLM versions every state, publishes new versions atomically and may replay them; a write it does not know about breaks versions and replay. |
| A workload holds declarations and settings only: plain values, enums, dataclasses, other workloads. No tensors, modules, open files or mutable globals. | The workload is deep-copied at make_llm and rebuilt in worker processes; tensors belong in state that pLLM places and pages. |
| A setting that depends on the instance is read with `pllm.cfg("field")` or a function of the workload (`beta=strength` with `def strength(self) -> float`). | Decorators run when the class is defined; a bare field name in the class body passes its class default (see [pitfalls.md](pitfalls.md#a-dataclass-field-named-in-the-class-body-passes-the-class-default)). |
| A method has at most one decorator of each kind, and one effect. | One method has one destination; the plan gives each a single anchor. |
| At most one `@pllm.guide_sampling` per workload, components included; any number of `@pllm.add_logits`. | The sampler applies one guidance per step; logit additions are summed before it. |
| One training mode per workload: request rows, or host batches. | The two reserve different memory and run at different times of the step. |
| One writer per state per step. | Two writers starting from the same version would overwrite each other's result. |
| Request-row training returns `[rows]`; pLLM masks, reduces (`reduction="mean"` or `"sum"`) and steps the optimizer. Host-batch training returns one scalar. | pLLM knows which rows are padding and which state each row belongs to; a pre-reduced scalar loses that. |
| Results flow one way within a step (`pllm.result` has no cycles). | Tasks of a step are ordered by their reads. |
| Sampler feedback (`pllm.sampled_tokens()`, `pllm.base_logits()`) is read only by the `@pllm.train` method of a workload that also has `@pllm.add_logits`, or by read-only helpers feeding that training. | Feedback exists only after the draw; pLLM delivers it to the training that closes an add-logits loop and to nothing else. |
| Components (workloads held as dataclass fields) are independent: their names get the field prefix (`probe.norm`) and they cannot read each other's state. | Composition must not create hidden coupling; to share state, put both methods in one workload. |
| `publish_lag` is one engine-wide setting; a component that requires a lag must match it. | There is one timeline of state versions per engine. |
| At most one `init=pllm.Fed(...)` state, one `init=pllm.Imported()` state and one `@pllm.train_in_forward` method per workload. | A feed and a `StateImport` each fill one state per call; one method drives all projection hooks. |
| Declarations are fixed when the engine is built; methods are not hooks Python calls at run time. | pLLM plans placement, memory and capture once; changing a setting means a new workload and a new engine. |

## Combinations that are refused

| Combination | Why | Do instead |
| --- | --- | --- |
| Two `@pllm.guide_sampling` methods (also across components) | One guidance per sampler step. | Keep one; turn the other into `@pllm.add_logits`. |
| Request-row training and host-batch training in one workload | Different memory reservations and step timing. | Two workloads. |
| `@pllm.train` with `rows=` / `update_rows=` next to `@pllm.guide_sampling`, or with no model effect | Such training updates every running request. | Put `rows=` on the guidance method. |
| `@pllm.custom_forward` and methods reading `pllm.hidden` / `pllm.final_hidden` | A custom forward may run a layer several times or skip it, so its MLP output has no single value. | Read `pllm.residual(layer)` or `pllm.layer_logits(...)`: they read each layer's last visited stream. |
| `active=` in a custom forward with `tokens="prefill"` or `"all"` | Skipped prompt tokens would leave approximate K/V that every later token reads. | `tokens="decode"`. |
| `@pllm.train_in_forward` and `pllm.model.Adapter` | Both drive the same projection hooks. | Train the adapter's state in the `@pllm.train_in_forward` method, or use two workloads. |
| Two `@pllm.custom_residual` methods on the same sublayer | One method owns a sublayer's residual update. | Merge them. |
| A stateful sublayer (`@pllm.custom_residual(state=..., chunk=...)`) and `@pllm.custom_forward` | The memory advances once per token, but a custom forward visits a layer a variable number of times. | Two workloads, or `pllm.PositionState` inside the custom forward. |
| Static KV layouts (`keep="window"` / `"none"`) and `pllm.PositionState` or `@pllm.custom_forward` | A layout splits the KV cache into several groups; these need one. | Drop one of them. |
| A fed state (`init=pllm.Fed(...)`) and other states or methods | The fed input group holds only its writer and the model effect reading it: a step without a new value skips the write. | Feed one state; one `@pllm.write` turns it into the state an `ActivationPatch` reads (see `pllm/workloads/video.py`). |
| `@pllm.custom_attention` reading a `RequestState`, or with `rows=` / `@pllm.on` / `@pllm.differentiate` | Attention runs for every scheduled token inside the forward, with instance-wide inputs only. | Read `InstanceState` parameters or modules, `pllm.depth()` and earlier `pllm.attention(...)` facts; select per request inside the math. |
| `streams=n` with `pllm.residual`, `pllm.layer_logits` or `@pllm.custom_forward` | Those read one stream. | Read `pllm.residual` inside a custom residual method. |
| Model-site effects (`Adapter`, `ActivationPatch`, `SharedLayers`, `@pllm.custom_forward`, `@pllm.custom_residual`, `@pllm.train_in_forward`) under `pllm.host.make_host` | `make_host` compiles without a model. | `make_llm`. |
| Two `@pllm.write` (or a `@pllm.train` and a `@pllm.write`) on one state | Two writers from one version. | One writer per state. |

## Support matrix

### Models

pLLM admits exact verified vLLM decoder classes (not arbitrary subclasses) across 15 architectures:

| Family | Supported | Not supported, and why |
| --- | --- | --- |
| Dense decoders (`Qwen2`, `Llama`, `Qwen3`, `Mistral`, `Phi3`, `Glm4`, `Gemma2`, `Gemma3`) | Everything (including `CausalTrainBatch` on dense `Qwen2`, `Llama`, `Qwen3`, `Mistral`, `Phi3`, `Glm4`, `Gemma3`). | Tied `Gemma2ForCausalLM` lacks a separate `lm_head` and `final_logit_softcapping` requires exact dense guidance instead of bounded top-K guidance. |
| KV-MoE (`Mixtral`, `Qwen2Moe`, `Qwen3Moe`) | Reads, exports, sampler effects, request training, `ActivationPatch`, sublayer `@pllm.custom_residual`, `@pllm.custom_forward`, `PositionState`, `@pllm.custom_attention`; `Adapter` / `@pllm.train_in_forward` on `q/k/v_proj`. | Adapters on MLP projections (tokens route across experts); `causal_model` training (implemented for dense decoders only). |
| Hybrids (`Qwen3_5`, `Qwen3_5Moe`: Gated DeltaNet + full attention) | Reads, exports, sampler effects, request training, `ActivationPatch`, `@pllm.custom_residual` (sublayer + `site="value"` on full-attention layers); `Adapter` / `@pllm.train_in_forward` on full-attention `k/v_proj` and dense MLP projections. | `@pllm.custom_forward`, `PositionState`, static KV layouts, and `@pllm.custom_attention` (DeltaNet state lives outside KV pages); `q_proj` overlays (interleaved output gate); DeltaNet linear-attention layers have no `q/k/v_proj`. |
| `DeepseekV2`, `DeepseekV3` (MLA + MoE) | Hidden taps, exports, sampler stages, request training (ESamp), `ActivationPatch`, non-`"value"` `@pllm.custom_residual`, and `@pllm.custom_attention` (`keep="all"`, absorbed reads in compressed latent space). | `Adapter`, `@pllm.train_in_forward`, `causal_model`, `site="value"`, `@pllm.custom_forward`, and sliding-window/shared KV layouts (MLA uses compressed latent KV pages and per-expert MoE projections). |
| `GptOss` (sink attention + MXFP4 MoE) | Hidden taps, exports, sampler stages, request training (ESamp), `ActivationPatch`, and non-`"value"` `@pllm.custom_residual`. | `Adapter`, `@pllm.train_in_forward`, `causal_model`, `@pllm.custom_attention`, and `@pllm.custom_forward`. |

Quantized checkpoints (GPTQ, AWQ, FP8) follow their family's rules: adapters correct the quantized
projection's output, and reruns call vLLM's quantized modules. `causal_model` training needs an unquantized
BF16 dense model whose embedding and LM head are not quantized.

### Engine versions

- vLLM 0.27 through 0.34 (with `.venv` on 0.27.1 and `.venv-vllm032` on 0.32.0 as live GPU baselines; 0.33–0.34 require CUDA 13 driver >= 580 for PyTorch 2.14+cu130); other releases refuse to start.
- pLLM always runs vLLM's MRv2 model runner (it sets `VLLM_USE_V2_MODEL_RUNNER=1` itself); MRv1 is unsupported.
- `pllm.make_llm`, `pllm.make_async_llm`, and `pllm serve` share one unified control plane: live `import_state(..., route=...)`, per-request `stream(..., state_import=...)`, `train_batch(HostBatch)` (including `CausalTrainBatch`), cross-process/TP `feed(name)`, `generate_completions` / `generate_feedback` (also with `workload=None`), and route-keyed persistence all work across sync and async handles. `ModelConfig.multiprocessing="auto"` selects `False` for sync `TP=1` and `True` for `make_async_llm`, `pllm serve`, or `TP > 1`.

### CUDA graphs

`ModelConfig(cuda_graphs="auto")` (default) runs vLLM's model graphs unless the workload needs eager;
`diagnostics().eager_reason` and `pllm.explain(...).eager_reason` name the cause. Workload math itself
(training with backward and optimizer, sampler stages, writes) is captured separately.

| Feature | Model forward |
| --- | --- |
| Hidden reads, `pllm.layer_logits`, exports, `@pllm.write`, `@pllm.train`, sampler effects | CUDA graphs |
| `pllm.model.ActivationPatch`, `@pllm.custom_residual`, `@pllm.custom_attention`, token-history lookups | CUDA graphs (vLLM captures source operators instead of inductor-fused ones) |
| `pllm.model.Adapter` on a `RequestState` or an `InstanceState` | CUDA graphs |
| `@pllm.custom_forward` | pLLM captures its own graphs per layer slice: `@pllm.custom_forward schedules model slices and owns their wave graphs` |
| `cuda_graphs=False` | eager |

### Tensor parallelism (`tensor_parallel_size > 1`)

Needs vLLM 0.32.0 or newer; single node; `pipeline_parallel_size` and `data_parallel_size`
stay 1.

| Works with TP | Needs `tensor_parallel_size=1` |
| --- | --- |
| Hidden reads, exports (delivered from rank 0), sampler effects, request-state training, `@pllm.custom_forward` | `causal_model` training |
| `Adapter` `"dense"` / `"low_rank"` deltas on `q/k/v/o`, `gate/up`, `down` (`Zeros`, `Computed`, `Pretrained`, or `Imported`) updated via shard-local `@pllm.train_in_forward` or `StateImport` / `import_state` | `"per_head"` deltas and custom projection operators (`operator=fn`); adapters updated by `@pllm.train` / `@pllm.write` side tasks; reading a sharded adapter field anywhere outside its projection |
| Non-`"value"` `@pllm.custom_residual` (including `pllm.input_tokens` helpers), `keep="all"` `@pllm.custom_attention` (without cross-head `pllm.attention(...)` facts), and `pllm.Fed` streams (`FanoutFeed`) | `@pllm.custom_residual(site="value")`, `@pllm.custom_attention` reading cross-head `pllm.attention(...)` facts, and sliding-window/shared KV layouts |

q/k/v and gate/up shard dense weights and B by output rows; o_proj and down_proj shard dense weights and A by input
columns; the other factor is replicated. An update that needs other ranks' shards is rejected.

## Expensive choices you write explicitly

Defaults only save you small decisions. A choice that changes memory, bandwidth or timing a lot is written
in your declaration, so it is visible in review and `pllm.explain` prints its cost. Never change one of
these to make something pass without telling the user.

| Choice | Default | What it costs | Where explain shows it |
| --- | --- | --- | --- |
| `pllm.Pretrained(path, placement="host")` | `"device"` (never falls back to host) | Rows cross PCIe each step (overlapped with the forward when the lookup reads only tokens and the table); only gathers (`table[idx]`, `index_select`, `embedding`, `vmap`) are allowed | `state` line: placement, host bytes, transfer bytes per row and per step |
| KV layout `@pllm.custom_attention(keep="window", window=N, sink=..., source=k)` | `keep="all"` | Changes what the model can attend to; frees memory only for the layout, not for `topk=` selections | `KV layer` lines |
| `ExecutionOptions(publish_lag=0)` | `1` (update at step n readable at n + 2) | Training no longer overlaps the next forward | `timing publish_lag` |
| `@pllm.train_in_forward(visibility="exact")` | `"forward_pinned"` | Later chunks wait for earlier chunks' updates | `timing` line |
| `tokens="all"` on hidden reads | `"decode"` | Sizes the tap by `max_num_batched_tokens` and exports many rows | `tap` and `export` lines |
| `@pllm.custom_forward` | none | vLLM's whole-model graph is replaced by per-slice graphs; may need a larger `cuda_graph_pool_budget_bytes`, which explain does not estimate | `forward eager because ...` |
| `ExecutionOptions(guidance_capacity=None)` | `4096` | Exact dense guidance over the full vocabulary instead of the top 4096 survivors | `timing guidance_capacity` |
| `ExecutionOptions(max_host_training_rows=n)` | `1` | Host-fed training rows per state per step; more memory | `state` and `memory` lines |
| Memory budgets: `state_page_budget_bytes`, `execution_peak_budget_bytes`, `cuda_graph_pool_budget_bytes` | observed free capacity / unlimited / traced | Caps pLLM's own reservations (not weights or KV); planning fails with a message when they do not fit | `state`, `memory`, `budget` lines |
| `ModelConfig(cuda_graphs=False)` | `"auto"` | Whole engine eager | `forward` line |
| `ModelConfig(tensor_parallel_size=n)` | `1` | Feature limits above | `parallel` line |

## Engine settings that interact with workloads

| Setting | Default | Rule | Reason |
| --- | --- | --- | --- |
| `enable_prefix_caching` | `False` | Safe for stateless model hooks and immutable `InstanceState` hooks with `ActivationPatch(last_tokens=0)`. Rejected for `RequestState`, mutable/`Imported`/`Fed` `InstanceState`s, `last_tokens > 0`, `@pllm.train_in_forward`, and `@pllm.custom_attention`. | Cached prompt K/V are keyed by tokens (plus workload namespace and generated-boundary isolation), not per-request or mutable state values. |
| `enable_chunked_prefill` | `False` | Results must not depend on how prompts are split into steps (`testing.check` verifies it), so either value is safe. | vLLM decides the split from its token budget. |
| `max_num_seqs`, `max_model_len`, `gpu_memory_utilization` | `8`, `512`, `0.5` | Size so vLLM does not preempt workloads whose replay is rejected. | See [Preemption and recompute](pitfalls.md#preemption-and-recompute). |
| `multiprocessing` | `"auto"` | `"auto"` uses `False` for sync `TP=1` and `True` for `make_async_llm`, `pllm serve`, or `TP > 1`. With worker processes, workload fields must be picklable (no locks, open files or live handles). | Workers rebuild the workload from cloudpickle bytes (classes from `__main__` or a notebook travel by value, importable ones by import path) and plan it themselves; editing imported modules between make_llm and the worker's start is your responsibility. |
| `dtype` | `"bfloat16"` | Hidden rows and model-dtype state follow it. | Inference precision; compute sensitive math in float32 inside the method. |
| `seed` | `0` | Seeds `pllm.Computed` initializers and sampling. | Reproducible runs and checks. |
