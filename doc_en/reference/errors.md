# Error Catalog

Every diagnostic and admission rejection message emitted by pLLM, paired with why it occurs and how to fix it.

## Method inputs and declarations

| Message | Why | Fix |
| --- | --- | --- |
| `...: give every input a source as its default, e.g. ...` | pLLM decides what a method reads from its parameter defaults; the annotation is only for you. | Give every parameter after `self` a marker default: `h: torch.Tensor = pllm.hidden(0)`. |
| `the workload has no effect on the model: add @pllm.train, ...` | Nothing schedules a method whose result goes nowhere. | Add the decorator that says where the result goes. An undecorated method only runs as a `rows=` selector or as a `pllm.result` producer for a decorated one. |
| `a method takes one pLLM decorator of each kind; split stacked effects into one method per effect` | One method has one destination; a second decorator would overwrite the first. | One method per effect; share math through a module-level helper function or `pllm.result(name)`. `@pllm.on` and `@pllm.differentiate` may sit next to an effect decorator. |
| `property ... needs a return annotation; annotate state declarations with -> pllm.RequestState or -> pllm.InstanceState, and other properties with their return type` | pLLM reads declarations from annotations without calling the property. | Annotate state properties `-> pllm.RequestState` / `-> pllm.InstanceState`, model-site properties `-> pllm.model.Adapter` / `-> pllm.model.ActivationPatch`, other properties with their type. |
| `... is not a declared state of ...` | A method names a state the workload does not declare. | Check the spelling; a state whose shape depends on settings is a typed property. |
| `pllm.parameter(..., ...) names no declared field; declared: ...` | The field name is not one the state declares. | Use a listed name. A `shape=` state has the single field `"value"`; an `init=pllm.FromModule(...)` state is read with `pllm.module(state)`. |
| `...: pllm.module needs a state declared with init=pllm.FromModule(...)` | Only a module state can be called like a module. | Declare `init=pllm.FromModule(lambda m: nn.Linear(...))`, or read fields with `pllm.parameter`. |
| `... names no method of ...` | A `rows=`, `pllm.result` or setting refers to a method that does not exist. | Check the method name. |
| `... and ... both declare ...; compose them as fields instead of inheriting from both` | With multiple inheritance, declarations of the same name overwrite each other. | Hold both workloads as dataclass fields (components). |
| `...: components are independent; to share state, put both methods in one workload` | A component must behave as it does alone, so it cannot read another component's state. | Put the methods that share state in one workload. |
| `...: a workload cannot hold a module or tensor; save its weights and declare ...` | A workload is a declaration copied at make_llm and rebuilt in worker processes; tensors belong in state that pLLM places and pages. | Save the tensor to safetensors and declare state with `init=pllm.Pretrained(path)` or `init=pllm.FromModule(factory, pretrained=path)`, or import it with `init=pllm.Imported()`. |
| `key_fields=...: use names from ...` | Request state is keyed by a fixed set of request facts. | Use `"request_id"`, `"prompt_index"`, `"sample_index"` or `"route"`. |
| `request key fields must be closed, sorted, and unique` | `key_fields` lists each name once, in the order `request_id`, `prompt_index`, `sample_index`, `route`. | Reorder or deduplicate, e.g. `key_fields=("prompt_index",)`. |
| `pllm.residual(after=...): use "layer" or "attention"` | The residual stream is readable after a whole layer or after its attention. | Pass `after="layer"` (the default) or `after="attention"`. |
| `pllm.input_tokens(history=) needs a nonnegative integer` | `history` counts tokens before the current one. | Pass `history=0` or more. |
| `pllm.attention needs a nonnegative layer and fact='aux' or 'lse'` | Attention facts exist per layer and are either the auxiliary output or the log-sum-exp. | Pass a layer index `>= 0` and `fact="aux"` or `fact="lse"`. |
| `... reads tokens="all" rows; only @pllm.export methods read them, and without pllm.result inputs. Export the rows, or read tokens="prefill" or "decode"` | `tokens="all"` gives one row per computed token, which no other effect can align with requests. | Export those rows, or read `tokens="decode"` / `tokens="prefill"`. |
| `pllm.cfg(...): expected a dotted Python attribute path such as 'layer' or 'config.layer'` | `pllm.cfg` resolves a dotted attribute path on the bound workload instance. | Pass one or more Python identifiers joined by dots, e.g. `pllm.cfg("beta")` or `pllm.cfg("config.rank")`. |

## Phases: when a value exists

| Message | Why | Fix |
| --- | --- | --- |
| `a task reads a value (sampled tokens, hidden states or another task's result) that exists only after its phase ends; declare it in a later phase` | A `@pllm.on(phase)` or the effect's default phase places the method before its input exists. | Before sampling read `pllm.layer_logits((-1,))`; read sampler results in `@pllm.train` or `@pllm.on(pllm.Phase.AFTER_SAMPLE)`. |
| `sampler method ... reads pllm...() through ..., which is available only in Phase.AFTER_SAMPLE.` | `pllm.base_logits()` and `pllm.sampled_tokens()` exist only after the sampler has drawn. | Before sampling read `pllm.layer_logits((-1,))` (the model's logits) or `pllm.final_hidden()`. |
| `...: sampled tokens are readable only beside @pllm.add_logits, by the train operation or by read-only operations that feed it` | Sampler feedback goes only to the training method of a workload that also adds logits, and to read-only helpers feeding that training. | Add the `@pllm.add_logits` method the feedback trains, or take tokens from `RequestOutput` on the host. Training that does not need the drawn token reads `pllm.final_hidden()` or `pllm.layer_logits((-1,))`. |
| `...: an export and the helpers it reads run read-only in Phase.AFTER_FORWARD on hidden rows, state and results; read the value from a helper without state writes or phase overrides` | Exports run after the forward; they cannot read sampler feedback and their helpers cannot write or move phase. | Export hidden rows, state or results only; sampled token ids are in each `RequestOutput`. |
| `tensor inputs need one phase; hidden inputs become available after model forward` | `@pllm.on(Phase.BEFORE_FORWARD)` on a method that reads this step's hidden rows. | Remove the `@pllm.on`; effects have correct default phases. |
| `...; results can only flow one way within a step, remove one of the reads` | Results of one step form a graph without cycles: two methods read each other's result. | Break the cycle; pass a value to the next step through state (`@pllm.write`) instead. |

## Method bodies

| Message | Why | Fix |
| --- | --- | --- |
| `while shape tracing method ...; methods run inside CUDA graphs: use tensor operations such as torch.where, not .item() or Python branches on tensor values` | PyTorch raises "Tensor.item() cannot be called on meta tensors" and pLLM adds this note. pLLM traces methods on shape-only tensors to plan memory and capture graphs; host reads have no value there and would stall the engine. | Remove `.item()`, `.tolist()`, `.cpu()`, `float(tensor)`, `print(tensor)` and `if tensor ...:`; use masks and `torch.where`; watch values with `@pllm.export`. |
| `methods run inside CUDA graphs, so outputs need shapes fixed by the inputs; return a fixed-size tensor and a tensor mask instead of ...` | An output whose shape depends on the data (dropping rows by a condition, `ctx.new_dynamic_size()`) cannot be captured in a graph. | Return a fixed-size tensor and a mask. |
| `methods run inside CUDA graphs, so Python cannot branch on tensor values (if, while or break on a tensor); ...` | An `if`, `while` or `break` on a tensor (often in a `@pllm.custom_forward` loop) makes each step compute something different. | Loop a fixed number of times and select rows with `torch.where`; in a `custom_forward` use `layers(..., active=mask)` so finished rows stay unchanged (needs `tokens="decode"`). |
| `... cannot compile as one full graph: ...` | The method contains an operation that cannot be compiled as one graph. | Use traceable tensor operations; give your own operators a fake implementation; or set `ExecutionOptions(allow_numerical_transforms=False)`. |

## Effects and results

| Message | Why | Fix |
| --- | --- | --- |
| `... both declare @pllm.guide_sampling; keep at most one guidance method` | The sampler applies one guidance per step (components included); how to combine two predictions is the algorithm's choice. | Keep one `@pllm.guide_sampling` that combines the predictions. Several `@pllm.add_logits` methods may coexist; their sum applies before guidance. |
| `@pllm.add_logits method ... must return [rows, vocab_size] = ...; got ...` | The result is added to the full-vocabulary logits. | Return `[rows, vocab]`: read `pllm.layer_logits((-1,))` and return `logits[:, -1]`-shaped tensors, or project hidden states to the vocabulary. |
| `@pllm.guide_sampling method ... must return a hidden-state prediction [rows, hidden_size] = ...; got ...` | pLLM projects the prediction with the model's LM head and pushes sampling away from it. | Return `[rows, hidden_size]` (e.g. a prediction of `pllm.final_hidden()`); to add logits directly, use `@pllm.add_logits`. |
| `tensor export needs positive every, max_rows and max_cols, with max_cols * 8 <= ... bytes (32 KiB), so one row fits an export event` | One export event carries whole rows of 8-byte values, at most 32 KiB. | Lower `max_cols` to at most 4096; compress wider rows in the method (e.g. top-k ids) or split them into several exports. |
| `export overflow must be 'drop' or 'block'` | A full export queue either drops events or blocks. | Pass `"drop"` (the default) or `"block"`. |
| `@pllm.write needs distinct field names and no rows=; it writes every row` | A write publishes a whole new version of the state. | Keep rows you do not want to change with `torch.where`. |
| `write must return a tuple matching its declared fields` | `@pllm.write` returns one tensor per declared field. | Return a tuple, even for one field: `return (new_value,)`. |
| `... and ... both write state ... from the same version; let one method write it` | Two writers of one state in one step would each overwrite the other's version. | One writer per state: one `@pllm.write` method (writing all its fields) or one `@pllm.train`, not both. |
| `StateImport supplies one imported resource per workload; ... are imported` | One `StateImport` fills one state per call. | Merge the imported states into one state with several fields. |
| `... requires publish_lag=...; the engine uses publish_lag=...; all components must use the engine-wide lag` | There is one timeline of state versions per engine. | Use one `publish_lag` for the engine and every component. |

## State and training

| Message | Why | Fix |
| --- | --- | --- |
| `@pllm.train reads ..., which trains with ...; declare ... with optimizer=...` | The batch fields a training method reads pick the training mode, and each mode has one optimizer. | Put the optimizer the message names on the state: request rows and `WeightedRowBatch` / `CausalTrainBatch` use `pllm.AdamW(lr)`, `WeightedCEBatch` uses `pllm.SGD(lr)`. |
| `training methods return one loss per row; pLLM reduces the selected rows per state` | Only pLLM knows which rows of a step are padding and which state each row belongs to. | Return a `[rows]` loss; do not `.mean()` it. `reduction="mean"` (default) or `"sum"` reduces per state. |
| `rows= and update_rows= on request training require a model effect without sampler guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state, so remove rows= and update_rows=` | Training next to guidance (or alone) updates and publishes every running request's state each step. | Remove `rows=` from `@pllm.train`; select rows on the sampler method (`@pllm.guide_sampling(rows="method")`), or use `reduction="sum"` with zero weights. |
| `... trains on ... but ... on ...; pLLM funds one training mode per workload, so move one of them to a separate workload` | Request-row training and host-batch training reserve different memory and run at different times. | Put them in two workloads; state only the host reads can train with `pllm.host.make_host`. |
| `carry fields ride on the training transition; give the state an optimizer` | Carry fields are published together with a training update. | Give the state an optimizer. |
| `pllm.carry writes into the training method's own state` | A carry is published with the state being trained. | Use a carry field of the trained state. |
| `every @pllm.differentiate result must feed a @pllm.train loss` | A differentiable result exists only to carry gradients into a training loss. | Read the result in a `@pllm.train` method through `pllm.result`, or remove `@pllm.differentiate`. |
| `the @pllm.train loss of ... must differentiate only its own state` | Gradients flow back into a state other than the one the method trains. | Check which differentiable results the loss reads; each training method updates only its own `state=`. |
| `...: add init=pllm.Computed(...) (e.g. random A, zero B); with both low_rank factors at zero every gradient is zero and the adapter never learns` | Gradients of `b @ a` vanish when both factors are zero. | Initialize A randomly and B at zero with `init=pllm.Computed(function)`. |
| `host-batch training returns one scalar and owns its reduction; remove reduction=` | A host batch's lengths, weights and normalization are known only to you. | Remove `reduction=`; reduce inside the method (`(weights * error).sum() / weights.sum().clamp_min(1)`). |
| `training on WeightedRowBatch or CausalTrainBatch fields takes one AdamW step over the whole batch: remove rows=, update_rows=, steps=, reset=, @pllm.differentiate and pllm.carry` | These batches always train as one whole batch, one step per batch. | Remove the listed settings; skip samples with weight 0. |
| `steps= and reset= apply to training on WeightedCEBatch fields (pllm.batch); remove them to train on request rows` | Repeated SGD steps from a reset point exist only for `WeightedCEBatch`. | Remove `steps=` and `reset=`. |
| `training on WeightedCEBatch fields restarts each state before its SGD steps: give @pllm.train(reset=...) naming a method that sets the starting values` | Each batch runs several SGD steps from the same starting point. | Write a method that returns the starting values and name it in `reset=`. |
| `pllm.batch fields ... must all come from one batch: WeightedRowBatch ..., CausalTrainBatch ... or WeightedCEBatch ...` | A training method reads one kind of host batch. | Read fields of one batch kind only. |
| `causal training needs a BF16 model whose embedding and LM head are not quantized` | Whole-model backward is written for BF16 and differentiates the embedding and LM head. | Use a BF16 base whose embedding and LM head are not quantized. |

## Loading and placing state

| Message | Why | Fix |
| --- | --- | --- |
| `pllm.Pretrained(...) for state ... does not exist; save it with safetensors.torch.save_file before make_llm` | The file named by `pllm.Pretrained` is not on disk. | Save it with `safetensors.torch.save_file` before `pllm.explain` or `make_llm`. |
| `pllm.Pretrained needs a .safetensors file; save it with safetensors.torch.save_file` | pLLM loads pretrained state only from safetensors files. | Convert `.pt` / `.bin` files with `safetensors.torch.save_file`. |
| `... has no tensor ... for state ...; it holds .... Name the fields like the saved tensors, or save the tensors under the field names` | The tensor names in the file differ from the state's field names. | Rename the fields or save the tensors under the field names. |
| `tensor ... in checkpoint ... has shape ...; adjust the declared field shape to match it` | The saved tensor's shape differs from the declared field shape. | Change the declared shape, or save a tensor of the declared shape. |
| `... cannot load module state ...: ...; save the file with safetensors.torch.save_file(module.state_dict(), path) from the same module class` | `pllm.FromModule` loads strictly; the file's keys or shapes do not match the module the factory builds. | Save `module.state_dict()` of the same module class. |
| `...: a pllm.ModelCheckpoint state is read-only; write to a separate state` | Base-model tensors are shared by every request and never change. | Write deltas or new values into a separate `RequestState` or `InstanceState`. |
| `...: declare one fed state per workload` | One feed fills one state per call. | Merge the fed values into one `init=pllm.Fed(...)` state with several fields. |
| `placement="host" needs a read-only InstanceState of gathered fields; use placement="device" for mutable state` | Host tables are read in bounded gathers that overlap the forward; they cannot be written. | Use `placement="host"` only for frozen pretrained tables of an `InstanceState`. |
| `...: gather host rows in a helper and read pllm.result in the model site; or use placement="device"` | A model site runs in the middle of a layer, where a PCIe read would stall the forward. | Gather in an undecorated helper that reads only `pllm.input_tokens` and the table; the model site reads `pllm.result("helper")`. |

## Model sites and custom forwards

| Message | Why | Fix |
| --- | --- | --- |
| `... reads block or model outputs, which a @pllm.custom_forward does not produce; use pllm.residual or pllm.layer_logits to read each layer's last visited stream` | A custom forward may run a layer several times or not at all, so a layer's MLP output has no single value. | Read `pllm.residual(layer)` or `pllm.layer_logits(...)`, or compute what you need inside the custom forward. |
| `Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. ...` | Skipping layers for prompt tokens would leave approximate K/V that every later token reads. | Use `tokens="decode"` for early exit; prompts run the full model. |
| `@pllm.custom_forward methods take (self, layers, marker inputs...) and no other pLLM decorator; use layers.save instead of @write` | A custom forward is an effect of its own. | Save position state with `layers.save`. |
| `...: use pllm.parameter or pllm.module after layers; read plain settings from self` | Parameters after `layers` are state reads. | Read plain settings from `self`. |
| `index a tap inside the Layers slice that produced it` | `layers(...)` returns outputs indexed by absolute layer; `out[-1]` is the model's last layer, which a partial slice did not run. | Index the slice's own layers: `layers(stop=12)[11]`. `[-1]` is right only when the slice runs to the end. |
| `PositionState is read and written by layers.load/save inside a @pllm.custom_forward method` | Position state means something only inside a custom forward. | Add a custom forward, or remove the `PositionState`. |
| `layer ... already has a custom residual (...); one method owns a sublayer's residual update` | One method decides how a sublayer's stream updates. | Merge the two methods into one. |
| `@pllm.custom_residual rewrites every row of its sites: remove rows= and @pllm.differentiate` | A residual rewrite acts on every row of its layer. | Select rows with a mask inside the method. |
| `@pllm.custom_residual(state=...) needs one attention or MLP site of one layer, chunk=, and a pllm.RequestState with one field, its carry, and no optimizer` | A stateful sublayer keeps one carried memory per request and updates it chunk by chunk. | Complete the declaration as the message lists. |
| `... advances its memory once per token, but a @pllm.custom_forward may run its layer several times or not at all per token; drop state= or the custom forward` | A stateful sublayer needs exactly one visit per token. | Use two workloads, or carry the memory in `pllm.PositionState` inside the custom forward. |
| `...: take the site tensors ... first, without defaults, then markers` | pLLM passes the site tensors positionally, in order. | Write `x, y` (or `v`) first, then inputs with marker defaults. |
| `...: take q and kv first, without defaults, then markers` | pLLM passes the query and the paged KV positionally. | Write `q, kv` first, then inputs with marker defaults. |
| `...: return attention shaped [rows, heads, v_head_dim] with model dtype` | The output feeds the model's own output projection (`v_head_dim` equals `head_dim` on dense GQA/MHA and the value head width on MLA). | Return a tensor shaped `[rows, heads, v_head_dim]` in `q.dtype`. |
| `MLA @pllm.custom_attention supports keep='all' with window=/topk= read selection; remove sliding-window or shared KV layout declarations` | MLA pages store compressed latent vectors whose block allocator uses full-history retention. | Keep `keep="all"` (the default) on MLA models and pass `window=` or `topk=` to `kv.attend(...)`. |
| `keep="none" needs explicit layers and source >= 1 pointing to an earlier layer; use @pllm.custom_forward with kv="peek" for top-layer sources` | Deeper layers have not run yet for the current token. | Point `source` at an earlier layer, or use a custom forward. |
| `kv.attend topk needs integer positions [rows, k] or [rows, heads, k], padded with -1` | Top-K entries are logical token positions. | Pass integer positions of that shape; pad missing ones with -1. |
| `static KV layouts cannot accompany PositionState or @pllm.custom_forward (including kv_copy)` | A layout splits the KV cache into several groups; these features need one group. | Drop the layout (`keep=` / `window=` / `source=`) or the custom forward. |
| `... reads RequestState ..., but @pllm.custom_attention runs for every scheduled token with instance-wide inputs; ...` | Attention reads have no per-request state. | Read an `InstanceState` and select per request inside the math. |
| `declare one @pllm.train_in_forward method per workload; ... already is one` | One method drives every projection hook of a workload. | Train all targets in one method (`targets` is a tuple). |
| `a @pllm.train_in_forward method drives every projection hook of its workload, so it does not yet combine with a pllm.model.Adapter; ...` | The training method owns the projection hooks the adapter would also use. | Train the adapter's state in the `@pllm.train_in_forward` method itself, or use two workloads. |
| `... must return the residual for ... as [rows, ...] in the input dtype or float32 on the input's device; got ...` | A `@pllm.train_in_forward` method without an optimizer returns the correction added to the projection's output; pLLM checks it once at declaration on maximum-shape operands. | Return `output` shaped `[rows, out_features]` of the target, in the input dtype or float32. |
| `... must return changed (bool) and update_count and trained_rows (int64) as 0-d tensors on the input's device` | The counters stay on the device so no step synchronizes with the host. | Build them as 0-d device tensors (`x.new_ones((), dtype=torch.bool)`), or return `pllm.UpdateOutput.unchanged(output)`. |
| `layer ... uses ...; MLP overlays need a dense MLP. Choose q_proj, k_proj or v_proj` | An MoE layer routes tokens across expert weights and has no single dense MLP projection. | Target `q_proj`, `k_proj`, or `v_proj` on that layer instead. |
| `layer ... is a Gated DeltaNet layer without q/k/v projections; choose a full-attention layer (see layer_types in the model config)` | Qwen3.5 linear-attention layers have no standard q/k/v projections. | Target a full-attention layer (`layer_types` in the model config). |
| `gated attention interleaves each head's query rows with its output-gate rows; overlay the value projection instead` | Qwen3.5 gated attention packs each head's query rows with its gate rows, so there is no separate query projection. | Rewrite `v_proj`. |
| `per_head adapters need tensor_parallel_size=1; use dense or low_rank deltas under TP` | Per-head deltas are not split along the tensor-parallel shards. | Use `"dense"` or `"low_rank"` under TP, or `tensor_parallel_size=1`. |
| `pllm.model.Adapter, pllm.model.ActivationPatch and @pllm.custom_forward act inside a model forward; compile this workload with make_llm, not make_host` | `make_host` compiles without a model. | Use `make_llm`, or keep only `pllm.batch` training for `make_host`. |
| `pllm.layer_logits reads the model's LM head; compile this workload with make_llm` | `make_host` compiles without a model, so there is no LM head. | Use `make_llm`. |

## Custom operators

| Message | Why | Fix |
| --- | --- | --- |
| `... mutates ...; kernels called from workload math must return new tensors (mutates_args=()); change state only through pLLM effects` | Workload inputs are shared read-only views, so an operator must not write into them. | Declare `mutates_args=()` and return new tensors; change state through `@pllm.write` or `@pllm.train`. |
| `... has no fake implementation; add @....register_fake` | pLLM plans memory from shapes, which it gets from the fake implementation. | Register one with `@op.register_fake` returning an empty tensor of the output's shape and dtype. |
| `call the op object you defined with torch.library.custom_op (from mykernels import gated_add), not torch.ops.<namespace>.<op>, so worker processes receive the op with its registration; use torch.ops.aten or torch.ops.prims only for built-in operators` | The op object carries the operator's definition to worker processes; a `torch.ops` name finds only what the current process registered. | Import and call the op object. |
| `... trains through ..., which has no autograd formula; add ....register_autograd(...)` | A `@pllm.train` loss backpropagates through an operator without a gradient. | Register one with `op.register_autograd(...)`. |
| `... runs in a per-token cell under torch.vmap; register a batching rule (register_vmap) or call it with over="span"` | A per-token custom forward runs under `torch.vmap`, which needs a batching rule for the operator. | Register one with `op.register_vmap(...)`, or use `over="span"`. |

## Engine settings, models and parallelism

| Message | Why | Fix |
| --- | --- | --- |
| `pLLM runs on vLLM 0.27 through 0.34; install a supported 0.27..0.34 release instead of vLLM ...` | pLLM integrates directly with vLLM's V1 worker, runner and scheduler internals within this tested range. | Install a vLLM release between 0.27.0 and 0.34.0 (with matching PyTorch / CUDA driver). |
| `...: choose one of the supported vLLM decoders: ...` | pLLM admits exact vLLM decoder classes whose norm, residual, KV, and projection structure it has verified. | Use a listed decoder family (see [Support Matrix](support-matrix.md)). |
| `pLLM runs vLLM's own model classes, not subclasses of them` | A subclass may change the projection layout or the logits. | Use the listed model class itself. |
| `... needs separate KV projections (...); hidden taps, sampler stages, activation patches and sublayer residuals still run on this model` | DeepSeek V2/V3 (MLA + MoE) and gpt-oss (sink attention + MXFP4 MoE) have no standard sliceable q/k/v or dense MLP projections. | Use hidden reads, sampler stages, activation patches, sublayer `@pllm.custom_residual`, or MLA `@pllm.custom_attention` (on DeepSeek V2/V3), or switch to a dense / KV-MoE model. |
| `... recomputes layer state from KV pages, but ... keeps Gated DeltaNet recurrent state outside them; choose a decoder with KV attention in every layer` | Qwen3.5 keeps linear-attention recurrent state outside paged KV, so layers cannot be replayed from KV pages alone. | Use a dense or KV-MoE decoder (such as Qwen2, Llama, Mistral, Phi3, Glm4, Gemma2/3, Mixtral, Qwen2-MoE, Qwen3, or Qwen3-MoE). |
| `projection shapes for this architecture are decided at make_llm; use a dense or KV MoE config for CPU checks` | CPU config-only shape derivation covers dense, KV-MoE, and Qwen3.5 hybrid configs. | Use a dense or KV-MoE config of matching dimensions for CPU checks, and verify the target checkpoint with `make_llm`. |
| `disable prefix caching (enable_prefix_caching=False): ...` | Prefix caching is safe only when model hooks are stateless or read immutable `InstanceState`s with `last_tokens == 0`; per-request or mutable state, `last_tokens > 0`, `@pllm.train_in_forward`, and `@pllm.custom_attention` produce request-dependent K/V. | Keep `enable_prefix_caching=False` (the default), or make the model-read state an immutable `InstanceState` (`Pretrained`, `Zeros`, `Computed`, `FromModule`, `ModelCheckpoint`) with `last_tokens=0`. |
| `...; set ModelConfig(cuda_graphs="auto") or cuda_graphs=False` | `cuda_graphs=True` demands vLLM's whole-model graphs, and the named feature runs the forward without them. | Remove that feature, or use `cuda_graphs="auto"` (the default); `diagnostics().eager_reason` then names the cause. |
| `@pllm.custom_attention needs vLLM's FlashAttention 2 backend; set VLLM_ATTENTION_BACKEND=FLASH_ATTN and VLLM_FLASH_ATTN_VERSION=2` | Custom attention is built on vLLM's FlashAttention 2 backend. | Set the two environment variables the message names. |
| `@pllm.custom_attention needs unquantized KV; set kv_cache_dtype=auto` | Your read function reads K/V in the model dtype. | Keep `kv_cache_dtype="auto"`. |
| `multi-GPU needs vLLM 0.32.0 or newer; otherwise use tensor_parallel_size=1` | pLLM supports tensor parallelism starting in vLLM 0.32.0. | Upgrade to vLLM 0.32.0 or newer, or set `tensor_parallel_size=1`. |
| `... runs on one GPU only; set tensor_parallel_size=1` | `causal_model` training, `@pllm.custom_residual(site="value")`, and `@pllm.custom_attention` with cross-head `pllm.attention(...)` facts or non-default KV layouts require a single GPU. | Use `tensor_parallel_size=1` (plain `keep="all"` `@pllm.custom_attention`, sublayer `@pllm.custom_residual`, and `Fed` states work with TP > 1). |
| `TP Adapter updates need shard-local @pllm.train_in_forward; side-lane updates need tensor_parallel_size=1` | Under TP each rank holds its shard of the delta (`Zeros`, `Computed`, `Pretrained`, or `Imported`); online gradient/write updates on side streams do not synchronize shards across GPUs. | Update TP adapters via shard-local `@pllm.train_in_forward` or `import_state` / `StateImport`, or use `tensor_parallel_size=1` for `@pllm.train` / `@pllm.write` updates. |
| `...: TP sharded fields are projection-local; other readers need tensor_parallel_size=1` | A sharded adapter field holds only the current GPU's slice, so ordinary tensor methods cannot read the full tensor. | Read the state only through `pllm.model.Adapter` / `@pllm.train_in_forward`, or set `tensor_parallel_size=1`. |
| `TP projection update is not shard-local at ...; ...` | The update math needs other ranks' shards (for example training replicated A from column-sharded outputs). | Rewrite the update to use only the local shard, or use `tensor_parallel_size=1`. |
| `other processes receive the workload by pickling (...); hold plain values, tensors and functions instead of locks, open files or live handles, or use tensor_parallel_size=1 with multiprocessing=False` | A workload field (or something a function in it captures) holds a lock, open file or live handle, which cannot be pickled. | Hold plain values, tensors and functions; open handles inside the method instead. |
| `check_on_gpu needs max_num_seqs >= 2 and max_model_len >= 32 for its fixed script` | The GPU check runs a fixed multi-request script. | Raise those settings for the check. |

## Memory budgets

| Message | Why | Fix |
| --- | --- | --- |
| `pLLM execution needs ... bytes at its peak but only ... fit; lower ...` | The planned execution peak exceeds `execution_peak_budget_bytes`. Reported in `pllm.explain(...).budget_errors` and raised by `make_llm`. | Follow the message: lower `max_num_batched_tokens` or `max_num_seqs`, raise the budget, or declare `placement="host"` for read-only tables. |
| `ExecutionOptions.state_page_budget_bytes=... exceeds the ... bytes free on the device; lower it, or lower gpu_memory_utilization ...` | The state-page budget is larger than the memory left on the device. | Lower the budget, or lower `gpu_memory_utilization` so vLLM leaves more memory free. |
| `workload needs cuda_graph_pool_budget_bytes >= ... for its conservative task capture reservations; got ...; leave it unset to size from the workload` | The CUDA graph pool budget is smaller than the workload's task captures need. | Leave `cuda_graph_pool_budget_bytes` unset, or raise it above the number in the message. |
| `custom-forward CUDA graphs need ... bytes; raise ExecutionOptions.cuda_graph_pool_budget_bytes above ...` | The layer-slice graphs of a custom forward need more capture memory than the budget. | Raise `cuda_graph_pool_budget_bytes` to the number in the message. |
| `raise execution_peak_budget_bytes to fund the custom forward's workspace and graphs` | `execution_peak_budget_bytes` cannot hold the custom forward's workspace, graphs and intermediate values. | Raise the budget, or leave it unset. |
| `training on CausalTrainBatch/WeightedRowBatch inputs cannot be funded under ExecutionOptions.execution_peak_budget_bytes yet; leave execution_peak_budget_bytes unset` | The peak of host-batch training is not yet counted in this budget. | Leave `execution_peak_budget_bytes` unset. |
| `...: host batch shapes ... exceed the planned ...; raise ExecutionOptions(max_host_training_rows=N) for more training rows per state, or HostConfig(max_input_rows=N) for a host workload` | pLLM reserves training memory for the declared maximum batch size. | Raise `ExecutionOptions(max_host_training_rows=N)`, or `HostConfig(max_input_rows=N)` for `make_host`. |
| `positive batch, tokens, rounds, warmup and capture/state capacities required` | A `BenchmarkOptions` count or budget is below 1. | Pass positive `--batch`, `--tokens`, `--rounds`, `--warmup` and budgets. |

## Requests and run time

| Message | Why | Fix |
| --- | --- | --- |
| `vLLM preempted a request and would recompute its history; ...` | The workload changes the model's computation from state that a replay cannot rebuild (see [Production Guide](../howto/production.md)). | Give vLLM more KV cache (`gpu_memory_utilization`) or fewer concurrent sequences (`max_num_seqs`), or remove what the message names. |
| `routes= needs a request state keyed by "route" and one value per prompt` | `routes=` picks a state copy per prompt, so it needs a route-keyed state and one non-empty route per prompt. | Declare `key_fields=("route",)` and pass one route per prompt. |
| `a route-keyed request state requires per-prompt routes` | A request without a route does not know which state copy to use. | Pass `routes=` to `generate`. |
| `RequestIdentity ... is missing a field required by ...` | A request lacks a value for a declared key field (an empty route, a negative prompt index). | Give every request a valid value for each key field. |
| `batch, sample_index, prompt_indices and routes need workload= in make_llm` | A plain engine has nothing to feed. | Pass `workload=` to `make_llm`. |
| `online serving runs vLLM's async multiprocess engine; omit multiprocessing=False or set True` | `make_async_llm` and `pllm serve` use vLLM's async worker process (`ModelConfig.multiprocessing="auto"` enables this automatically). | Leave `multiprocessing="auto"` (the default) or set `multiprocessing=True`. |
| `import_state needs a nonempty route and RequestState keyed only by route` | Out-of-band `llm.import_state(..., route=...)` targets either an `InstanceState` (no route) or a `RequestState` keyed solely by `"route"`. | Declare `RequestState(..., key_fields=("route",))` and pass `route="..."`, or pass per-request imports via `stream(..., state_import=...)` / `generate(..., batch=...)`. |
| `train_batch needs a workload with a host batch training method` | `llm.train_batch(batch)` runs a `@pllm.train` method declared with `pllm.batch(...)` inputs. | Declare a `@pllm.train` method reading `pllm.batch(...)` fields (for `CausalTrainBatch`, `WeightedRowBatch`, or unkeyed `WeightedCEBatch`). |
| `online request lacks pLLM identity metadata; submit through AsyncVllmLlm or pllm serve` | Worker execution requires the online identity metadata stamped by `make_async_llm` or `pllm serve`. | Submit requests through `pllm.make_async_llm` or `pllm serve` (passing session keys via `extra_body={"vllm_xargs": {"pllm_route": ...}}`). |

## Contract check errors

`pllm.testing.check` raises `pllm.testing.ContractError`; its `failures` list every discrepancy with a `[contract]` prefix and the request, token and method involved.

| Message | Why | Fix |
| --- | --- | --- |
| `method ... mixes rows: equal inputs for this request produced different outputs. Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0).` | The method computes across rows (`mean(0)`, `softmax(dim=0)`, batch norm, sorting rows), or keeps mutable state on `self`, so one request's result depends on others. | Make every row depend only on its own inputs: reduce along `dim=-1`. |
| `method ... mutates borrowed input ... in place; inputs and parameters are read-only views. ...` | Inputs and parameters are shared with other methods and with pLLM's state versions. | Use out-of-place math (`h * 2`, not `h.mul_(2)`), or `clone()` first; change state only through `@pllm.write` or `@pllm.train`. |
| `[neutral] ...: add_logits must return exactly zero` | With strength 0 the workload still changes the logits. | Multiply the whole result by the strength setting and pass the strength-0 workload as `neutral=`. |
| `[neutral] guide_sampling still declares a sampler stage; set its beta to zero.` | The neutral workload still guides (`beta=beta` reads the class default instead of an instance setting; see [Workload Anatomy](../guides/workload-anatomy.md)). | Make `beta` a method of the workload (`beta=strength`). |
| `pLLM misrouted rows before ...: its per-request inputs already differ.` | Usually your code: tensor math that mixes different states' rows, or an index computed from `pllm.index` used across states. | Rule out your code first; if the method is row-local, report the message as a framework bug. |
| `[device equivalence] ...: ...` | `check_on_gpu` found the GPU kernel's results outside the tolerance of its CPU reference for that dtype. | Compare the kernel with the reference: accumulation precision (float32 intermediates) and indexing. |
