# OverRIDE Reproduction Report

## Scope

This report evaluates whether the tLLM OverRIDE consumer reproduces the purpose
and core mechanism of **Diverse Text Decoding via Iterative Reweighting** without
vendoring or depending on the upstream repository.

Reviewed local files:

- `tllm/consumers/override/consumer.py`
- `test/consumers/override/test_override_consumer_unit.py`
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
  representation used by the pending cache.

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

This is a faithful reproduction of the purpose and central algorithmic formula.
It is not an end-to-end vLLM patch reproduction, and that appears intentional
from the local docs.

## Executed Checks

Commands are shown with placeholders to avoid local machine paths.

```bash
<repo-env-python> -m pytest -q test/consumers/override/test_override_consumer_unit.py
```

Result:

```text
12 passed
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
has vLLM `0.10.1.1` and is missing Hydra/OmegaConf. I did not install upstream
dependencies or change the environment, per the dependency policy.

## Toy Mechanism Comparison

I imported `<upstream-repo>/modeling.py` and compared
`OverRIDELogitsProcessor._compute_reweighted_logits` against the tLLM consumer
on identical small tensors and identical manually copied head weights.

Inputs:

- hidden shape: `(2, 2)`
- logits shape: `(2, 3)`
- `lambda = 0.5`
- `num_iteration = 3`
- rank: `2`
- current iteration: `2`

Result:

```text
max_reweighted_absdiff 0.0
max_q_next_absdiff 0.0
up_reweighted [[9.0, 17.0, 27.5], [0.75, 3.0, 3.25]]
local_reweighted [[9.0, 17.0, 27.5], [0.75, 3.0, 3.25]]
up_update_norm 0.12247449159622192
local_update_norm 0.12247449159622192
max_updated_head_absdiff 0.0
```

This confirms exact agreement for the residual subtraction, next-head logits,
and one optimizer update when the two implementations start from the same
weights.

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
- canonical integer row ids resolving to the same cached update rows,
- no upstream dependency.

End-to-end diverse decoding verdict: **not yet fully integrated**.

The consumer exposes the right primitive operations and now includes the
batch-row bridge needed by a sampler/runtime hook: request-id iteration parsing,
batched grouping by iteration, cached `q_logits_next`, and sampled-token update
by row id. It still does not itself patch vLLM or perform sampling.

## Gaps And Risks

- `flows()` returns empty, so the current consumer is a manual logits/sampler
  primitive rather than an automatically wired runtime consumer.
- The batch bridge can parse iteration groups from request ids or accept
  explicit iteration ids, but tLLM still lacks an automatic vLLM sampler/runtime
  hook that feeds those groups through decoding.
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
