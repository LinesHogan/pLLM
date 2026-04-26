# GTTA Reproduction Report

## Scope

- tLLM revision inspected: `21b76bd`
- Upstream GTTA revision inspected: `24ba304`
- tLLM files inspected: `tllm/consumers/devlist/gtta.py`, `test/devlist/gtta/test_gtta_consumer_unit.py`, `doc/reference/gtta-consumer.md`, `docs/devlist.md`, and the devlist consumer design docs.
- Upstream files inspected: `README.md`, `WebArena/README.md`, `BFCLv3/README.md`, `WebArena/src/run_exploration.py`, `WebArena/src/run_agent.py`, `WebArena/src/agent/agentlab_agent.py`, `WebArena/src/agent/prompts/prompt_constructor.py`, `WebArena/src/process_into_jsons.py`, `BFCLv3/berkeley-function-call-leaderboard/bfcl/model_handler/base_handler.py`, `BFCLv3/berkeley-function-call-leaderboard/bfcl/model_handler/utils.py`, and `vllm/model_runner.py`.

## Upstream Mechanism Summary

Upstream GTTA is not a single small consumer. It has two adaptation paths:

1. **Dynamics Grounding (DG), non-parametric adaptation**
   - Exploration runs collect agent/environment trajectories.
   - After each action, WebArena code compares the previous and current browser observation and asks an LM summarizer to produce environment dynamics records with `initial_state`, `action`, and `environmental_dynamics`.
   - Those records are filtered outside the hot path, then injected into task-time prompts so the policy can predict action consequences.
   - BFCL uses the same idea for function-call environments: load precomputed environment dynamics by environment name and append them to the system prompt.

2. **Syntactic Alignment (SA), parametric adaptation**
   - Upstream patches vLLM's `model_runner.py`.
   - During prompt processing, it maintains a `ptuning_params` vector, computes token-level loss on the prompt, optionally masks user tokens, and performs Adam updates.
   - The adapted vector is added to hidden states before logits are computed.

The WebArena, browsergym, bgym, agentlab, BFCL, notebooks, local website services, and browser automation pieces are environment/evaluation layers. They are necessary to reproduce the upstream benchmark workflow, but they are not inherently required for a tLLM consumer that only ingests already-produced trajectory feedback.

## tLLM Mapping

The current tLLM GTTA implementation is a lightweight feedback ingester:

- `TrajectoryFeedback` validates `task_id`, `action`, `observation`, and parseable reward.
- `GTTAConsumer.enqueue_trajectory()` stores feedback in a bounded queue and reports drops.
- `GTTAConsumer.process_queued_trajectories()` drains queued records and updates a per-action score with optional momentum.
- `GTTAConsumer.score_action()` exposes the local score table.
- `GTTAConsumer.read_stats()` reports queued, dropped, processed, and updated counts.
- The module imports only stdlib plus tLLM consumer contracts; it does not import WebArena, browsergym, bgym, agentlab, or BFCL.

This matches the narrow tLLM design target from the devlist docs: accept external action/observation/reward records and keep a local action or rerank score table without vendoring the benchmark stack.

## Reproduction Feasibility

I did not run an upstream WebArena or BFCL mini task. The current environment is not set up for a lightweight upstream run:

- Direct dependency probe with the repo venv showed `browsergym`, `bgym`, `agentlab`, and `bfcl` are missing.
- A minimal WebArena `run_agent` import using the upstream source path failed with `ModuleNotFoundError: No module named 'beartype'`.
- Upstream WebArena scripts require local WebArena service URLs, browsergym/agentlab, and usually a self-hosted or API-backed model.
- Upstream DG extraction/filtering also relies on notebooks and environment traces.
- Upstream SA requires replacing vLLM internals and uses a hard-coded Qwen tokenizer in the patched runner.

Because installing or launching those stacks would be heavier than this evaluation target, I used tLLM unit tests plus a toy trajectory case to validate the implemented subset.

## Checks Run

Commands are summarized with placeholders to avoid machine-specific paths.

1. GTTA unit tests:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 <repo-venv>/bin/python -m pytest -q -p no:cacheprovider test/devlist/gtta/test_gtta_consumer_unit.py
   ```

   Result: `7 passed`.

2. Toy trajectory check:

   - Config: `max_queue_size=3`, `score_momentum=0.25`.
   - Inputs: positive reward for `click search`, zero reward for `click ad`, another positive reward for `click search`, and one overflow record.
   - Result:
     - accepted flags: `[True, True, True, False]`
     - after partial drain: `queued_trajectories=1`, `dropped_trajectories=1`, `processed_trajectories=2`, `update_count=2`
     - final stats: `queued_trajectories=0`, `dropped_trajectories=1`, `processed_trajectories=3`, `update_count=3`
     - final scores: `click search=0.5625`, `click ad=0.0`, `type query=0.0`
     - selected score order placed `click search` first.

3. Dependency probe:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 <repo-venv>/bin/python - <<'PY'
   import importlib.util
   for name in ["browsergym", "bgym", "agentlab", "bfcl", "vllm"]:
       print(name, importlib.util.find_spec(name) is not None)
   PY
   ```

   Result: environment/evaluation dependencies were missing; `vllm` was available.

4. Upstream WebArena import probe:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<upstream>/WebArena/src <repo-venv>/bin/python - <<'PY'
   import run_agent
   PY
   ```

   Result: failed at missing `beartype`, before reaching browsergym service setup.

## Alignment Conclusion

**Against the tLLM devlist target:** the implementation is aligned. It preserves the environment boundary, avoids large dependencies, validates and queues feedback, updates action scores in the expected direction, and exposes counters that prove work happened.

**Against full upstream GTTA:** this is only an inspired subset, not a full reproduction. The following upstream mechanisms are not reproduced:

- no environment dynamics extraction from observation deltas;
- no dynamics filtering pipeline;
- no task-time prompt construction or prompt injection;
- no BFCL environment-dynamics loading by environment/class;
- no WebArena/browsergym action execution loop;
- no candidate-action pairwise judging, rejection tuning, or q-judge/reranker dataset generation;
- no SA parametric hidden-state/prompt-adapter update inside vLLM;
- no policy/model update path connected to real generation.

So the correct classification is: **tLLM GTTA is a dependency-light trajectory feedback consumer that implements a useful local subset of GTTA's feedback idea, but it is not a faithful reproduction of the full GTTA algorithmic stack.**

## Gaps And Risks

- `observation` is stored but not used in scoring, so the score table cannot learn state-conditioned dynamics.
- Scores are keyed only by raw action text; identical action strings in different states/tasks are conflated.
- `task_id` is validated but not used for per-task isolation.
- There is no consumer flow, sampler/reranker integration, or policy intervention; callers must manually use `score_action()`.
- The reward update is scalar exponential smoothing, not upstream DG prompt grounding, q-judge preference learning, or SA parameter adaptation.
- There is no real-model comparison because the current implementation has no model path to compare against upstream GTTA's prompt or vLLM adaptation behavior.

## Recommendation

Keep the current implementation documented as **GTTA-inspired**. If the goal is closer reproduction, the next minimum step should be state/task-aware dynamics records plus a policy-facing selection API, still without importing WebArena or BFCL. A faithful full reproduction would require a separate design for DG prompt injection and SA hidden-state adaptation rather than expanding this simple score-table consumer silently.
