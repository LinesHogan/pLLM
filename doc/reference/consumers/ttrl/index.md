# TTRL-Inspired Consumer

This consumer is a tLLM-native rewrite of TTRL's lightweight test-time reward
idea. It does not import verl or the reference training stack.

## Supported Core

The first version implements two local pieces:

- a lightweight verifier interface, with a default boxed normalized exact
  verifier,
- boxed-answer-only majority-vote reward construction over multiple completions
  for the same prompt,
- upstream-style vote/sample separation: vote over `n_votes_per_prompt`
  completions, then keep the first `n_samples_per_prompt` indices per prompt
  for training,
- structured vote results with majority label, majority ratio, per-sample
  normalized label, and scalar reward,
- a toy generation trace to reward outcome loop that turns completed generation
  traces into `RewardOutcomeBatch` entries,
- a no-model closed-loop helper that builds rewards, runs the local toy update,
  and produces retry scheduling metadata tied to the updated adapter version,
- a small low-rank score adapter trained from hidden rows and scalar rewards,
- a lightweight GRPO batch builder that packages majority rewards, selected
  training indices, group ids, optional current logprobs, and optional reference
  logprobs into the shared test-time trainer contract,
- pure PyTorch toy helpers for per-group reward normalization and clipped-ratio
  loss on small logprob tensors.

Majority voting is boxed-only. The consumer extracts the last `\boxed{...}`
answer, normalizes that boxed content, and ignores unboxed completions for
judging. Unboxed completions are marked with `missing-boxed-answer` in
structured results and reward outcomes; they are not treated as bare-text
answers. Tie handling is deterministic: if boxed labels tie, the first boxed
label seen for that prompt is treated as the winner.

The trace-to-outcome path is intentionally light. It maps
`GenerationTraceBatch` identity fields (`request_id`, `prompt_idx`,
`sample_idx`, `output_index`, `group_id`) into `RewardOutcomeBatch`, assigns
`reward=1.0` to samples matching the boxed majority, `reward=0.0` to boxed
minority samples, and `reward=None` plus `skip_reason=missing-boxed-answer` to
unboxed samples.

The verifier is intentionally small and replaceable. `BoxedNormalizedVerifier`
is the default and does not import `math_verify`, sympy latex plugins, verl,
Ray, Hydra, or a full math grader. A future math-equivalence verifier can plug
into the same `label_for_completion()` contract, but it is not a runtime
dependency of this toy path.

The toy closed-loop helper is meant for local evidence, not policy training:
`run_ttrl_toy_closed_loop()` maps `GenerationTraceBatch` to rewards, selects
training rewards, calls `update_from_rewards()`, and returns a
`TTRLClosedLoopIntent` whose retry schedule carries the updated
`adapter_version_id`. A fake scheduler can use that version id to prove that a
minority retry would be issued against the post-update adapter state.

This remains a toy alignment path. It does not implement the full TTRL LoRA
actor loop, PPO/GRPO updates, policy feedback into generation, or a full
math-equivalence grader.

## Lightweight GRPO Bridge

`build_ttrl_grpo_training_batch()` converts a completed `GenerationTraceBatch`
into a `TestTimeTrainingBatch` with `objective_kind="grpo"`. Each example keeps
the request, prompt, sample, output, token ids, scalar majority reward, group id,
and selected training index. Current policy logprobs can be stored in the
contract `logprobs` field; reference policy logprobs are optional metadata.

`TTRLConsumer.submit_grpo_training_batch()` submits that batch through the shared
`TestTimeTrainerBridge` protocol and can stage returned request-scoped
`TestTimeLoRAUpdate` values through `TestTimeLoRAService`. This is only a bridge
surface: tLLM core does not ship a distributed GRPO actor trainer.

`ttrl_grpo_group_normalized_advantages()` and
`ttrl_grpo_clipped_ratio_loss()` are small PyTorch helpers for unit-testable toy
math. They do not claim parity with verl, Ray, FSDP, or a production RL trainer.

verl integration is optional. Core TTRL imports do not import `verl`; a future
integration should live outside the core package and provide the trainer bridge
protocol. `create_optional_verl_grpo_bridge()` is a placeholder that reports the
external bridge is unavailable instead of importing heavy dependencies.

## Usage

```python
from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig, TTRLSample
from tllm.consumers.ttrl import build_ttrl_grpo_training_batch, build_ttrl_reward_outcome_batch, build_ttrl_reward_result
from tllm.consumers.ttrl import run_ttrl_toy_closed_loop
from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

consumer = TTRLConsumer(TTRLConsumerConfig(hidden_size=896, rank=16))
samples = [
    TTRLSample(prompt_idx=0, sample_idx=0, completion=r"Trace A. \boxed{4}"),
    TTRLSample(prompt_idx=0, sample_idx=1, completion=r"Trace B. \boxed{4}"),
    TTRLSample(prompt_idx=0, sample_idx=2, completion=r"Trace C. \boxed{5}"),
]
rewards = consumer.rewards_for_samples(samples)
consumer.update_from_rewards(hidden_rows, rewards)

result = build_ttrl_reward_result(samples)
assert result.majority_by_prompt[0].label == "4"
assert result.majority_by_prompt[0].ratio == 2 / 3
assert result.rewards == [1.0, 1.0, 0.0]

trace = GenerationTraceBatch(
    entries=(
        CompletionTraceEntry(
            request_id="req-a",
            prompt_idx=0,
            sample_idx=0,
            output_index=0,
            token_ids=(101, 102),
            text=r"Trace A. \boxed{4}",
            finish_reason="stop",
            group_id="round-a",
        ),
    )
)
outcomes = build_ttrl_reward_outcome_batch(trace)
assert outcomes.entries[0].judge == "ttrl_majority:boxed_normalized_exact"

grpo_batch = build_ttrl_grpo_training_batch(trace)
assert grpo_batch.examples[0].objective_kind == "grpo"
```

For a no-model closed-loop toy:

```python
import torch
from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig, run_ttrl_toy_closed_loop

consumer = TTRLConsumer(TTRLConsumerConfig(hidden_size=4, rank=2))
intent = run_ttrl_toy_closed_loop(consumer, trace, torch.randn(len(trace.entries), 4))
assert intent.training_rewards() == [1.0]
assert intent.training_row_count == 1
assert intent.adapter_version_id == "ttrl:update:1"
```

For upstream-style vote/sample separation:

```python
from tllm.consumers.ttrl import select_ttrl_training_indices

indices = select_ttrl_training_indices(
    num_prompts=2,
    n_votes_per_prompt=4,
    n_samples_per_prompt=2,
)
assert indices == [0, 1, 4, 5]
```

## Counters

Use `read_stats()` to inspect:

- `update_count`
- `loss_count`
- `reward_count`
- `last_loss`

An enabled smoke run should report `reward_count > 0` and `update_count > 0`.
