# Devlist Alignment Matrix

Date: 2026-05-01

This matrix is the consumer-facing closure record for the projects listed in
`docs/devlist.md`. It summarizes mechanism coverage, direct upstream feasibility,
toy/source parity, cached-model evidence, runtime evidence, and the required
nonzero work counter for each implementation.

| Project | tLLM alignment status | Upstream run result | Correctness evidence | Runtime / model evidence | Nonzero work counter | Scope exception |
| --- | --- | --- | --- | --- | --- | --- |
| OverRIDE | Mechanism aligned; runtime sampler smoke is opt-in | Direct upstream runner blocked by vLLM/API and Hydra/OmegaConf mismatch | Source-extracted logits-processor parity and sampler-provider unit tests cover previous-head residual subtraction, current-head `q_logits_next`, row-id mapping, and sampled-token CE update | `test_override_real_decode_smoke.py` provides a cached Qwen vLLM smoke guarded by `TLLM_OVERRIDE_RUN_CACHED_MODEL_SMOKE=1`; routine CI keeps dependency-light unit coverage | `loss_count`, `update_count`, `reweight_count`, `modified_row_count` | Full upstream diverse-decoding patch stack remains external because it targets a different vLLM surface and heavier config stack |
| In-Place TTT | Exact MLP fast-weight runtime path aligned behind explicit opt-in; legacy low-rank consumer remains separate | Direct upstream import blocked by missing/lightweight-incompatible dependencies and Transformers API mismatch | Official-shaped Qwen MLP parity covers `ttt_proj`, zero-init `ttt_conv`, effective `past_w`, tail caching, and pre-update emission order | Guarded cached Qwen smokes cover wrapper install, request/sample slots for batch 4 and 8, estimated state bytes, and generated-token change | `update_count`, `loss_count`, `trained_rows`, slot count, estimated bytes | Full checkpoints with upstream-trained `ttt_conv`/`ttt_proj` weights are not bundled; generic end-to-end prefill `ConsumerFlow` delivery is still a documented integration boundary; low-rank mode intentionally fails fast |
| TTRL | tLLM-native majority reward and lightweight GRPO trainer handoff aligned; distributed actor update stays optional | Upstream utility parity is source-extracted with local stubs where full verl stack is too heavy | Majority labels, selected reward rows, grouped rewards, and toy GRPO bridge behavior are tested | Cached Qwen rollout smoke feeds real generations into majority reward and adapter update | `reward_count`, `update_count`, `loss_count`, staged update count where bridge is used | Full verl PPO/GRPO actor update is an optional external integration, not core |
| FTTT | Feedback trace, feedback-LM batch, trainer bridge, LoRA staging, and retry/update barrier aligned | Direct upstream import/run blocked by Python-version-sensitive syntax and missing dataset/trainer dependencies | Trace -> judge -> feedback record, `feedback_lm` labels, no-update retry blocking, and request-scoped LoRA staging are tested | `test_fttt_real_model_smoke.py` provides a cached tiny-model smoke guarded by `TLLM_FTTT_RUN_CACHED_MODEL_SMOKE=1`; unit tests cover fake trainer/service handoff by default | `feedback_count`, `update_count`, `loss_count`, `staged_update_count` | Full PEFT/OpTune feedback-token optimization is optional external trainer work and is isolated from core imports |
| GTTA | BFCL-shaped minimal dynamics ingestion, action ranking, prompt-context injection, and local executor aligned | WebArena/BFCL imports are blocked by evaluation/browser dependencies | Minimal BFCL fixtures cover function-call correctness, wrong function, wrong argument, state-dependent behavior, and dynamics reranking | Cached tiny-model prompt-context smoke proves dynamics injection changes model logits; fake capture runner verifies prompt text injection | `update_count`, processed trajectory counters | WebArena/browser automation and full BFCL leaderboard runners remain external evaluation layers |
| TLM | TTL formula, causal-shift selection, online/offline ordering, and LoRA handoff aligned at contract boundary | Upstream LLaMA-Factory TTL import blocked by missing `accelerate`, `datasets`, PEFT, and installed package shape | Shifted causal logits, CE threshold selection, exponential weighting, offline base-logit selection, and online/offline state machine are tested | Cached tiny-model smoke stages a request-scoped LoRA update from real LM logits/hidden states | `selected_count`, `update_count`, `loss_count`, staged update count | Full LLaMA-Factory/PEFT optimization of target model modules remains optional external integration; autonomous runtime tensor collection, declared flow wiring, and real vLLM LoRA apply smoke are still outside the core closure |

## Dependency Isolation

tLLM core consumer imports remain dependency-light. Heavy upstream stacks such as
`verl`, PEFT, LLaMA-Factory, BFCL, WebArena, browsergym, and agentlab are not
required to import `tllm`, `tllm.consumers`, `tllm.contracts`, or `tllm.runtime`.
Optional trainer/evaluation integrations must stay behind explicit factories or
opt-in tests and should fail with clear missing-dependency errors.

## Throughput Evidence

Runtime-hot paths expose opt-in cached-model smokes or benchmark hooks rather
than unconditional CI workloads:

- OverRIDE: guarded cached Qwen vLLM decode smoke compares disabled and enabled
  decode on one workload and asserts nonzero `loss_count` and `update_count`.
- In-Place TTT: guarded cached Qwen request/sample smokes record batch 4 and 8
  state size and nonzero per-slot fast-weight updates; repeatable throughput
  comparison remains workload/hardware dependent.
- TLM/FTTT/TTRL: runtime impact is represented as trainer/LoRA handoff and retry
  barrier cost, not as a direct decode-throughput claim unless the caller applies
  the resulting update during generation.
- GTTA: prompt-context injection affects prompt construction and logits; WebArena
  task-throughput is outside core.

No algorithm is marked aligned solely because tests pass. Each aligned row above
has mechanism evidence plus an explicit nonzero work counter, with scope
exceptions called out where the upstream project requires heavyweight trainer or
evaluation stacks outside tLLM core.
