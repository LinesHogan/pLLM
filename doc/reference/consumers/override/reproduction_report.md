# OverRIDE Reproduction Report

## Scope

This report evaluates whether the tLLM OverRIDE consumer reproduces the purpose
and core mechanism of **Diverse Text Decoding via Iterative Reweighting** without
vendoring or depending on the upstream repository.

Reviewed local files:

- `tllm/consumers/override/consumer.py`
- `tllm/consumers/override/sampler_provider.py`
- `test/consumers/override/test_override_consumer_unit.py`
- `test/consumers/override/test_override_sampler_provider_unit.py`
- `test/consumers/override/test_override_toy_parity_unit.py`
- `test/test_sampler_runtime_view_unit.py`
- `test/test_sampler_patch_unit.py`
- `doc/reference/consumers/override/index.md`
- `docs/devlist.md`
- `docs/plans/2026-04-26-devlist-consumers-design.md`

Reviewed upstream files:

- `<upstream-repo>/modeling.py`
- `<upstream-repo>/gpu_model_runner.py`
- `<upstream-repo>/engine.py`
- `<upstream-repo>/main.py`
- `<upstream-repo>/README.md`
- `<upstream-repo>/config/default.yaml`

## Upstream Mechanism Summary

The upstream implementation patches vLLM rather than providing a standalone
library API. Its main mechanism is in `OverRIDELogitsProcessor`:

- It creates LoRA-like low-rank reweighting heads with `A: hidden -> rank` and
  `B: rank -> vocab`.
- For decoding iteration `i`, it uses head `i - 1` to predict residual logits
  and samples from:

  ```text
  reweighted_logits = p_logits - lambda * residual_logits
  ```

- It also computes the next trainable logits:

  ```text
  q_logits_next = logits + current_head(hidden_states)
  ```

- After sampling, the patched `GPUModelRunner` calls
  `update_reweighting_head(q_logits_next, sampled_token_ids, q_indices)`.
- The update loss is cross entropy between `q_logits_next` and the token sampled
  from the reweighted distribution.
- Request ids encode the iteration index. The upstream logits processor groups
  rows by that parsed index, so different rows in one vLLM batch can use
  different reweighting heads.
- Runtime row ids are treated as stable integer row identities across reweight
  and sampled-token update calls. Tensor row-id inputs are accepted only at the
  API boundary and immediately normalized to the canonical `tuple[int, ...]`
  representation used by the pending cache. Python row ids must already be
  provided as `tuple[int, ...]`, and canonical row ids must be unique; mutable,
  ambiguous, or duplicate identities fail at the boundary instead of being
  interpreted internally.

The intent is therefore not just to perturb logits once, but to iteratively
train small residual heads from previously sampled tokens and then penalize
directions already explored by earlier iterations.

## tLLM Implementation Mapping

The local consumer captures the mechanism-level core:

- `OverRIDEReweightEngine` owns a sequence of low-rank `LowRankAdapter` heads.
- `reweight_logits(hidden, logits, iteration=i)` uses the previous head and
  computes:

  ```text
  logits - lambd * previous_head(hidden)
  ```

- The same call returns `q_logits_next = logits + current_head(hidden)` when a
  current head exists.
- `update_from_sampled_tokens(q_logits_next, sampled_token_ids)` applies
  cross-entropy training to the current head.
- The implementation is native to tLLM/PyTorch and does not import upstream
  OverRIDE code.
- The disabled path returns the original logits unchanged.
- Stats track reweighting, modified rows, losses, updates, and last loss.
- `OverRIDESamplerModifierProvider` connects the batch primitive to the tLLM
  sampler patch. It receives `SamplerStepView` logits, source hidden states, and
  request ids before sampling; returns a dense pre-filter sampler state for the
  reweighted logits; and receives sampled token ids after sampling to train the
  cached `q_logits_next` rows for the same decode rows.
- The sampler view keeps `request_ids` even when prompt/sample metadata is
  provided as device tensors, so OverRIDE can still resolve iteration heads.
- The provider binds post-sample updates to the prepared engine step and row
  count, rejecting stale-step or shape-mismatched callbacks before updating.

This is a faithful reproduction of the purpose and central algorithmic formula.
It is not an end-to-end vLLM patch reproduction, and that appears intentional
from the local docs.

## Executed Checks

Commands are shown with placeholders to avoid local machine paths.

```bash
<repo-env-python> -m pytest -q test/consumers/override/
```

Result:

```text
22 passed, 7 subtests passed
```

Sampler patch callback check:

```bash
<repo-env-python> -m pytest -q test/test_sampler_patch_unit.py
```

Result:

```text
6 passed
```

Environment check:

```bash
<repo-env-python> - <<'PY'
import importlib.metadata as md
for pkg in ["vllm", "torch", "transformers", "hydra-core", "omegaconf"]:
    ...
PY
```

Relevant result:

```text
vllm 0.10.1.1
torch 2.7.1
transformers 4.55.2
hydra-core NOT_INSTALLED
omegaconf NOT_INSTALLED
```

Upstream full-run feasibility check:

```bash
PYTHONPATH=<upstream-repo> <repo-env-python> - <<'PY'
import gpu_model_runner
PY
```

Result:

```text
ImportError: cannot import name 'MultipleOf' from 'vllm.attention.backends.abstract'
```

The upstream README and requirements target vLLM `0.11.1`; the repo environment
has vLLM `0.10.1.1` and is missing Hydra/OmegaConf. A real upstream run is
therefore not a lightweight local check: it requires changing the vLLM patch
surface and installing extra upstream runtime dependencies before the runner can
even import. I did not install upstream dependencies, download models, or change
the environment, per the dependency policy.

## Fixed Toy Parity Coverage

The local test suite now includes a fixed toy reference in
`test/consumers/override/test_override_toy_parity_unit.py`. The reference is a
small PyTorch implementation of the upstream sampler-side math, not an import
from the upstream repository. That keeps the test lightweight while still
pinning the mechanism that mattered in the earlier comparison.

Inputs:

- hidden shape: `(2, 2)`
- logits shape: `(2, 3)`
- fixed `lambda` values for both residual and update checks
- `num_iteration = 3`
- rank: `2`
- current iteration: `2`

The fixed toy tests cover:

- residual reweighting as `logits - lambda * residual`;
- `q_logits_next = logits + current_head(hidden)` with 0.0 max absolute
  difference from the toy reference;
- one sampled-token cross-entropy update with 0.0 max absolute difference in the
  low-rank head weights after the optimizer step;
- row id boundaries: non-integer Python ids, non-canonical Python containers,
  duplicate tuple/tensor ids, bool/float tensor ids, and non-1D tensor ids are
  rejected;
- row id lifecycle: tensor row ids are normalized to stable integer cache keys,
  so mutating the original tensor after `reweight_logits_batch()` does not alter
  which cached rows are updated.

This codifies the previous toy parity result in local tests without depending
on the upstream package, vLLM patch version, Hydra/OmegaConf, or a downloaded
model.

## Cached Real-Model Tensor Comparison

Because the upstream vLLM runner could not be run in the current environment, I
used a cached small causal LM through Transformers in offline/local-files mode.
The model produced real hidden states and logits; those tensors were then fed to
both the upstream logits processor and the tLLM consumer with identical copied
head weights.

Command shape:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=<upstream-repo> <repo-env-python> - <<'PY'
# load cached tiny causal LM with local_files_only=True
# compute final-token hidden/logits for two prompts
# compare upstream logits processor with tLLM consumer
PY
```

Result:

```text
model sshleifer/tiny-gpt2
hidden_shape (2, 2)
logits_shape (2, 50257)
sampled [5087, 16046]
max_reweighted_absdiff 0.0
max_q_next_absdiff 0.0
max_updated_head_absdiff 0.0
local_stats OverRIDEStats(reweight_count=1, modified_row_count=2, loss_count=1, update_count=1, last_loss=10.713310241699219)
```

This is not a full upstream decode run, but it uses real model tensors and
verifies that the tLLM consumer matches the upstream reweight/update math on the
sampler-facing data shape.

## Alignment Conclusion

Mechanism-level verdict: **aligned**.

The tLLM consumer correctly reproduces the core OverRIDE mechanism:

- low-rank residual-logit heads,
- previous-iteration residual subtraction,
- current-iteration `q_logits_next`,
- sampled-token cross-entropy update,
- unique canonical `tuple[int, ...]` row ids resolving to the same cached update
  rows,
- no upstream dependency.

Runtime decode verdict: **guarded tLLM smoke available**.

The consumer now exposes the right primitive operations and a minimal tLLM
sampler-provider loop: request-id iteration parsing, batched grouping by
iteration, cached `q_logits_next`, pre-sample reweighting, and post-sample
sampled-token update by the same internal decode-row ids. This closes the local
sampler adapter gap for fake sampler tests.

The closure update adds `test_override_real_decode_smoke.py`, guarded by
`TLLM_OVERRIDE_RUN_CACHED_MODEL_SMOKE=1`, to run cached Qwen vLLM decode with
OverRIDE disabled and enabled on the same workload. The smoke asserts real
decode outputs, nonzero `loss_count` and `update_count`, and records a same-run
throughput ratio. Routine unit coverage remains dependency-light; the guarded
smoke is the evidence path for CUDA/local-cache sessions.

## Gaps And Risks

- `flows()` returns empty, so the current consumer is a manual logits/sampler
  primitive plus sampler provider rather than a hidden-row bundle consumer.
- The sampler provider requires one request id per decode row. The fake sampler
  tests cover this contract, and the guarded cached-model smoke is the opt-in
  check for active runtime metadata shape.
- The local update path does not use `ignore_index=-100`; upstream does. This
  should not matter for ordinary sampled token ids, but it matters if masked
  targets are ever passed through the same API.
- Initialization is not byte-identical: upstream initializes the down projection
  with normal std `0.02`; local `LowRankAdapter` uses a hidden-size-scaled std.
  Since the output projection is zero-initialized in both implementations, the
  initial residual is still zero, and the tested copied-weight mechanism is
  identical.
- Upstream creates `num_iteration` heads but only returns heads for iterations
  `1 <= i < num_iteration`; local creates `num_iterations - 1` heads. This is
  functionally equivalent for the used iteration range, but not structurally
  identical.
- Full upstream reproduction could not be executed in the current environment
  without changing dependencies: the upstream runner targets a newer vLLM and
  requires packages not present in the repo environment.

Overall, this is a correct tLLM-native reproduction of OverRIDE's core
algorithmic idea, with the main remaining risk being runtime integration rather
than the reweighting/update math.
