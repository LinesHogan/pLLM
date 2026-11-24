# Nearest example for each task

Start a new workload from the closest maintained example instead of from scratch: it already follows the
[rules](rules.md) and passes the checks. Find the task below, open the example, and copy only its public
declarations (`pllm.*` names). Every example listed here is checked by `tools/check_docs.py` (symbols and
links) and, for `pllm/workloads/` and `examples/`, by the repository's tests.

Where each kind of example lives:

- **reference.md sections**: the smallest runnable example per declaration, executed on CPU (and some on
  the GPU) by `tools/check_docs.py`. Copy these first.
- **`examples/*.py`**: standalone scripts for one algorithm, with a `__main__` that runs it.
- **`pllm/workloads/`**: maintained algorithms (importable as `pllm.workloads.<Name>`), tested on GPU.
  `pllm/workloads/fttt/workload.py` imports one internal helper (`pllm.execution.cuda_stream.upload`); do
  not copy internal imports, because internals change without notice and skip the checks public
  declarations get.
- **`doc/tutorials/*.ipynb` (00–05) and `doc/examples/*.ipynb` (06–13)**: Chinese notebooks with measured outputs, one per algorithm family; the code
  cells are English and show the whole loop (declare, explain, run, measure).

## By task

| Task | Nearest example | Declarations it uses |
| --- | --- | --- |
| Observe a hidden state, the residual stream or the logit lens | [reference.md: Read a hidden state](reference.md#read-a-hidden-state-the-residual-stream-or-per-layer-logits) (`Lens`); `pllm/workloads/dummy.py`; example 06 | `pllm.residual`, `pllm.layer_logits`, `pllm.hidden`, `@pllm.export` |
| Export rows to the host (probe features, SAE codes) | [reference.md: Export observations](reference.md#export-observations-to-the-host) (`StreamNorm`, `PromptEnd`); `examples/sae_export.py`; tutorial 01 | `@pllm.export`, `on_export`, `pllm.io.TraceEvent`, `tokens="prefill"` / `"all"`, `pllm.FromModule` |
| Replace one function of a built-in without subclassing | `examples/centered_hidden.py` | `dataclasses.replace` on a workload field holding a module-level function |
| Steer activations with a vector (CAA, ActAdd) | [reference.md: Edit activations](reference.md#edit-activations-or-weights-from-a-state) (`Steering`); tutorial 00 | `pllm.model.ActivationPatch(operator="add")`, `pllm.InstanceState(init=pllm.Imported())`, `pllm.io.StateImport` |
| ReFT / LoReFT from a checkpoint | `pllm/workloads/reft.py` (`ReFTWorkload`) | `pllm.model.ActivationPatch(operator="low_rank_affine")`, `pllm.io.load_tensor_fields`, `pllm.io.StateImport` |
| Per-request LoRA written from the prompt (Text-to-LoRA style) | tutorial 02; `pllm/workloads/state_producers.py` (`PromptFactorsWorkload`, `PromptMixtureWorkload`) | `pllm.model.Adapter`, `pllm.RequestState`, `@pllm.write`, `pllm.hidden(..., tokens="prefill")`, `pllm.FromModule` |
| Per-request LoRA imported from outside | [reference.md: Edit activations](reference.md#edit-activations-or-weights-from-a-state) (`PromptLoRA`) | `pllm.model.Adapter(operator="low_rank")`, `pllm.Field`, `init=pllm.Imported()` |
| Train a small network per request during decode and guide sampling (ESamp) | [SKILL.md](SKILL.md) (`Distill`); [reference.md: Train during decode](reference.md#train-during-decode) (`ShallowDistiller`); `pllm/workloads/esamp.py`; tutorial 03 | `pllm.RequestState(init=pllm.FromModule(...), optimizer=pllm.AdamW(...))`, `@pllm.train`, `@pllm.guide_sampling(rows=...)`, `pllm.version` |
| Request-local training with a hand-written loss and initializer | `examples/huber.py` | `pllm.Computed`, `pllm.AdamW`, `@pllm.train`, `@pllm.guide_sampling`, `pllm.version` |
| Train decode-time factors that an Adapter applies | `pllm/workloads/state_producers.py` (`DecodeFactorsWorkload`) | `@pllm.train`, `pllm.model.Adapter`, `pllm.Computed` |
| Add to the logits (bias, prior, contrast) | [reference.md: Add to the logits](reference.md#add-to-the-logits) (`LogitBias`); `pllm/workloads/dola.py` (DoLa); example 06 | `@pllm.add_logits`, `pllm.layer_logits`, `pllm.sample_index` |
| Logit head trained on the drawn tokens (sampler feedback) | `pllm/workloads/override.py` (`OverRIDEWorkload`) | `@pllm.add_logits`, `@pllm.train` reading `pllm.base_logits()` / `pllm.sampled_tokens()`, `pllm.InstanceState`, `pllm.sample_index` |
| Output LoRA trained on weighted cross-entropy batches (shared or per request key) | `pllm/workloads/lora.py` (`OutputLoRAWorkload`) | `@pllm.add_logits`, `@pllm.train(steps=, reset=)`, `pllm.batch`, `pllm.host.WeightedCEBatch`, `pllm.SGD`, `pllm.Dimensions` |
| A running statistic per request | [reference.md: Write state](reference.md#write-state-from-your-own-math) (`RunningMean`) | `@pllm.write`, `pllm.parameter`, `pllm.Field` |
| Rewrite a residual connection or value rows (value residual, hyper-connections, LAuReL) | [reference.md: Rewrite a residual site](reference.md#rewrite-a-residual-site) (`ValueResidual`); example 07; `examples/algorithm_search/candidate.py` | `@pllm.custom_residual(site=..., streams=...)`, `pllm.value`, `pllm.depth` |
| Fast-weight memory sublayer (TTT / Titans style) | example 07 (memory sublayer) | `@pllm.custom_residual(state=..., chunk=...)`, one-field `pllm.RequestState` |
| Train adapters inside the forward while reading the prompt | [reference.md: Train inside the forward](reference.md#train-inside-the-forward) (`PromptAdapters`); `pllm/workloads/prefill_adapters.py` | `@pllm.train_in_forward(chunk=, flush="prompt_end")`, `pllm.operand`, `pllm.SGD`, `pllm.Computed` |
| In-Place TTT (fast weights updated per chunk, also during decode) | `pllm/workloads/inplace_ttt.py` (`InPlaceTTTWorkload`); example 08 | `@pllm.train_in_forward(visibility="exact")`, `pllm.ModelCheckpoint`, `pllm.UpdateOutput` |
| Delta-rule fast weights per request | `pllm/workloads/gated_delta.py` | `@pllm.train_in_forward`, `pllm.operand`, `pllm.Field` |
| Rerun layers (recurrent depth, Ouro / Huginn style) | [reference.md: Run layers in a loop](reference.md#run-layers-in-a-loop-or-exit-early) (`LatentLoop`); `pllm/workloads/recirculation.py`; tutorial 04 | `@pllm.custom_forward`, `pllm.model.Layers`, `pllm.model.LayerOutputs` |
| Early exit / per-token layer skipping | example 12; `examples/algorithm_search/early_exit.py` | `@pllm.custom_forward(tokens="decode")`, `layers(..., active=mask)` |
| Carry state from one position to the next (Coconut style) | `pllm/workloads/step_loop.py` | `@pllm.custom_forward`, `pllm.PositionState`, `layers.load` / `layers.save` |
| Custom attention read: monitor, top-K selection, sliding window, cross-layer KV | [reference.md: Replace an attention read](reference.md#replace-an-attention-read-or-the-kv-layout) (`RecentWindow`); example 10 | `@pllm.custom_attention(keep=, window=, sink=, source=)`, `pllm.model.PagedKV`, `pllm.attention` |
| Your own kernel in workload math | example 09 | `torch.library.custom_op` with `register_fake`, called inside a method |
| Look up a table by token history (n-gram prior, Engram-style memory) | `examples/ngram_prior.py`, `examples/engram_lite.py`; example 11 | `pllm.input_tokens`, `pllm.Pretrained(placement=)`, `pllm.result`, `@pllm.add_logits` / `@pllm.custom_residual`, `pllm.io.checkpoint_tensors` |
| Generate, score, train, retry (TTRL / majority vote) | `pllm/workloads/reward_head.py`; tutorial 05 | `llm.generate_feedback`, `pllm.host.make_host`, `pllm.host.WeightedRowBatch`, `pllm.host.RetryControl` |
| Whole-model LoRA training on generated sequences (TLM, GRPO) | `pllm/workloads/tlm/workload.py` (`TLMWorkload`); tutorial 05 | `pllm.causal_model`, `pllm.model.CausalModel.token_logprobs`, `pllm.host.CausalTrainBatch`, `pllm.model.Adapter` |
| Judge feedback, a retry queue and a trained edit (FTTT) | `pllm/workloads/fttt/` | `pllm.host.make_host`, `pllm.batch`, `pllm.host.RetryControl`, `pllm.host.UpdateBarrier` |
| Feed state from a producer process (video, sensors) | `examples/video_session.py`, `pllm/workloads/video.py` | `pllm.InstanceState(init=pllm.Fed("ordered"))`, `@pllm.write`, `pllm.model.ActivationPatch`, `llm.feed` |
| Compose two workloads | [SKILL.md: Composition](SKILL.md#composing-workloads) | dataclass fields holding workloads |
| Unit-test your math without a model | [reference.md: Step a workload by hand](reference.md#step-a-workload-by-hand-in-tests) (`Centered`) | `pllm.testing.compile_execution`, `step_input`, `run_step` |
| Compare throughput with plain vLLM | [reference.md: Measure throughput](reference.md#measure-throughput); `tools/benchmark_esamp.py` | `pllm.benchmark.throughput`, `pllm.benchmark.compare` |
| Let an agent search over workload variants | `examples/algorithm_search/`; example 13 | `pllm.explain`, `pllm.testing.check`, `pllm.benchmark` |
| Migrate existing HF / PyTorch / vLLM hooks | [migration.md](migration.md) | Pre-flight audit table + CPU-verified translations |
| Online streaming or OpenAI HTTP serving | [SKILL.md](SKILL.md); [migration.md: Runtime entry points](migration.md#3-wire-into-the-right-runtime-entry-point); [doc/howto/online-serving.md](../howto/online-serving.md) | `pllm.make_async_llm`, `pllm serve`, `extra_body={"vllm_xargs": {"pllm_route": ...}}` |

## Built-in workloads

All run through the same public path as your code. Every class below is exported from `pllm.workloads` except
`GatedDeltaWorkload` (`pllm.workloads.gated_delta`) and `VideoWorkload` (`pllm.workloads.video`).

| Class | File | Algorithm | Main declarations |
| --- | --- | --- | --- |
| `DummyWorkload` | `dummy.py` | bounded hidden export with a replaceable function | `@pllm.export`, `pllm.hidden` |
| `DoLaWorkload` | `dola.py` | DoLa layer contrast | `@pllm.add_logits`, `pllm.layer_logits` |
| `ESampWorkload` | `esamp.py` | ESamp: per-request distiller guiding sampling | `@pllm.train`, `@pllm.guide_sampling`, `pllm.FromModule` |
| `OverRIDEWorkload` | `override.py` | iterative logit heads trained on sampler feedback | `@pllm.add_logits`, `@pllm.train`, `pllm.sampled_tokens` |
| `OutputLoRAWorkload` | `lora.py` | output LoRA trained by SGD on weighted CE batches | `@pllm.add_logits`, `pllm.host.WeightedCEBatch` |
| `ReFTWorkload` | `reft.py` | LoReFT / constant activation edits | `pllm.model.ActivationPatch`, `pllm.io.StateImport` |
| `PromptFactorsWorkload`, `PromptMixtureWorkload`, `DecodeFactorsWorkload` | `state_producers.py` | adapter factors from prompt networks or decode AdamW | `pllm.model.Adapter`, `@pllm.write`, `@pllm.train` |
| `PrefillAdaptersWorkload` | `prefill_adapters.py` | adapters learned chunk by chunk during prefill | `@pllm.train_in_forward` |
| `InPlaceTTTWorkload` | `inplace_ttt.py` | In-Place TTT | `@pllm.train_in_forward`, `pllm.ModelCheckpoint` |
| `RecirculationWorkload`, `JacobiRecirculationWorkload` | `recirculation.py` | rerun upper layers per token | `@pllm.custom_forward` |
| `StepLoopWorkload` | `step_loop.py` | carry pre-norm residual into layer zero | `@pllm.custom_forward`, `pllm.PositionState` |
| `RewardHeadWorkload` | `reward_head.py` | majority-vote reward head | `pllm.host.make_host`, `pllm.host.WeightedRowBatch` |
| `TLMWorkload` | `tlm/workload.py` | test-time learning with a whole-model LoRA | `pllm.causal_model`, `pllm.host.CausalTrainBatch` |
| `FTTTWorkload` | `fttt/workload.py` | feedback test-time training with a judge | `pllm.host.make_host`, host retry loop |
| `GatedDeltaWorkload` | `gated_delta.py` | gated delta-rule fast weights per request | `@pllm.train_in_forward`, `pllm.operand` |
| `VideoWorkload` | `video.py` | video features fed by a producer process | `pllm.Fed`, `@pllm.write`, `pllm.model.ActivationPatch` |
| `ConcertinaWorkload` | `concertina.py` | Concertina folded layers with shared linear weights and gated step LoRAs | `pllm.model.SharedLayers`, `pllm.model.Adapter(operator="gated_low_rank")`, `pllm.Pretrained` |
