# GTTA Reproduction Report

## Scope

- tLLM revision inspected: local `consumer-gtta-prompt-smoke-report` branch
  after cached prompt-context smoke and fake capture-runner verification
- Upstream GTTA revision inspected: `24ba304`
- tLLM files inspected: `tllm/consumers/gtta/consumer.py`,
  `test/consumers/gtta/test_gtta_consumer_unit.py`,
  `test/consumers/gtta/test_gtta_dynamics_toy_unit.py`,
  `tllm/runtime/dynamics_store.py`,
  `tllm/ports/prompt_context.py`, `tllm/ports/trajectory_event.py`,
  `doc/reference/consumers/gtta/index.md`, `docs/devlist.md`, and the
  devlist consumer design docs.
- Upstream files inspected: `README.md`, `WebArena/README.md`,
  `BFCLv3/README.md`, `WebArena/src/run_exploration.py`,
  `WebArena/src/run_agent.py`, `WebArena/src/agent/agentlab_agent.py`,
  `WebArena/src/agent/prompts/prompt_constructor.py`,
  `WebArena/src/agent/prompts/jsons/p_state_changelog_env_dynamics_v1.json`,
  `WebArena/src/agent/prompts/jsons/p_cot_exploration_with_history_persona_env_dynamics_v1.json`,
  `WebArena/env_dynamics/env_dynamics_webarena_gpt4.1_v1/all_env_dynamics_cleaned.json`,
  `WebArena/src/process_into_jsons.py`,
  `BFCLv3/berkeley-function-call-leaderboard/bfcl/model_handler/base_handler.py`,
  `BFCLv3/berkeley-function-call-leaderboard/bfcl/model_handler/utils.py`,
  `BFCLv3/berkeley-function-call-leaderboard/bfcl/constants/env_dynamics_prompt.py`,
  and `vllm/model_runner.py`.

## Upstream Mechanism Summary

Upstream GTTA is not a single small consumer. It has two adaptation paths:

1. **Dynamics Grounding (DG), non-parametric adaptation**
   - Exploration runs collect agent/environment trajectories.
   - After each action, WebArena code compares the previous and current browser observation and asks an LM summarizer to produce environment dynamics records with `initial_state`, `action`, and `environmental_dynamics`.
   - The exploration prompt receives the accumulated dynamics list as context
     so it can avoid redundant actions and choose actions that explore new
     environment transitions.
   - Those records are filtered outside the hot path, then injected into
     task-time prompts so the policy can predict action consequences.
   - BFCL uses the same idea for function-call environments: load precomputed
     environment dynamics by environment name and append JSON dynamics records
     to the system prompt.

2. **Syntactic Alignment (SA), parametric adaptation**
   - Upstream patches vLLM's `model_runner.py`.
   - During prompt processing, it maintains a `ptuning_params` vector, computes token-level loss on the prompt, optionally masks user tokens, and performs Adam updates.
   - The adapted vector is added to hidden states before logits are computed.

The WebArena, browsergym, bgym, agentlab, BFCL, notebooks, local website services, and browser automation pieces are environment/evaluation layers. They are necessary to reproduce the upstream benchmark workflow, but they are not inherently required for a tLLM consumer that only ingests already-produced trajectory feedback.

## tLLM Mapping

The current tLLM GTTA implementation is a lightweight trajectory feedback and
candidate-ranking ingester:

- `TrajectoryFeedback` validates `task_id`, `action`, and parseable reward.
- `GTTAConsumer.enqueue_trajectory()` stores feedback in a bounded queue and reports drops.
- `GTTAConsumer.enqueue_trajectory_event()` accepts the runtime
  `TrajectoryEvent` dataclass and uses its explicit `task_id`,
  `state_signature`, `previous_observation`, `action`, `next_observation`, and
  `reward` fields.
- `GTTAConsumer.process_queued_trajectories()` drains queued records and updates
  task/state/action-scoped scores with optional momentum.
- `GTTAConsumer.score_action()` exposes the local score table.
- `GTTAConsumer.rank_candidate_actions()` ranks candidate actions for a given
  task and state signature.
- `GTTAConsumer.export_dynamics_records()` exports copied processed dynamics
  records.
- `GTTAConsumer.query_dynamics_records()` filters copied dynamics records by
  task, state signature or observation, and action.
- `GTTAConsumer.render_ranked_dynamics_snippets()` creates deterministic
  reward-ranked dynamics text for future prompt context.
- `GTTAConsumer.build_prompt_context_entry()` wraps those snippets in the
  runtime `PromptContextEntry` contract.
- `GTTAConsumer.render_environment_dynamics_context()` creates deterministic
  newline-delimited JSON with upstream-shaped `initial_state`, `action_taken`,
  and `environmental_dynamics` fields, plus task/state/reward metadata.
- `GTTAConsumer.build_environment_dynamics_prompt_context_entry()` wraps that
  JSONL context in the runtime `PromptContextEntry` contract.
- `GTTAConsumer.read_stats()` reports queued, dropped, processed, and updated counts.
- The module imports only stdlib plus tLLM consumer/runtime contracts; it does
  not import WebArena, browsergym, bgym, agentlab, or BFCL.

This matches the narrow tLLM design target from the devlist docs: accept external action/observation/reward records and keep a local action or rerank score table without vendoring the benchmark stack.

The consumer-local toy coverage now fixes the narrow adoption subset:
trajectory ranking, processed dynamics accumulation/export, filtered dynamics
lookup, explicit runtime event ingestion, copy-safe query/export,
deterministic snippet generation, upstream-shaped environment-dynamics JSONL,
fake capture-runner prompt-context injection, and opt-in cached-model evidence
that prompt context changes real logits. The lookup is deliberately an
in-process helper over already ingested feedback; it uses the existing runtime
contracts instead of inventing a new runtime port.

## Runtime Contract Alignment

Task 6 runtime contracts now exist outside this consumer:

- `TrajectoryEventPort` describes write-side post-step event ingress.
- `PromptContext` describes write-side pre-generate prompt context.
- `DynamicsStore` stores `TrajectoryEventBatch` and `PromptContextBatch`, offers
  task/state/action query, deterministic text export, and prompt-context
  application before generation.

This consumer adopts those contracts where it has a local role. It accepts
`TrajectoryEvent` objects without loose dictionary coercion, keeps the explicit
runtime `state_signature`, and can produce a `PromptContextEntry` containing
ranked dynamics snippets or upstream-shaped environment-dynamics JSONL. It does
not modify runtime/ports and does not emulate missing runtime hooks inside the
consumer.

## Reproduction Feasibility

I did not run an upstream WebArena or BFCL mini task. The current environment is not set up for a lightweight upstream run:

- Direct dependency probe with the repo venv showed `browsergym`, `bgym`,
  `agentlab`, `beartype`, and `bfcl` are missing.
- A minimal WebArena `run_agent` import using the upstream source path failed with `ModuleNotFoundError: No module named 'beartype'`.
- Upstream WebArena scripts require local WebArena service URLs, browsergym/agentlab, and usually a self-hosted or API-backed model.
- Upstream DG extraction/filtering also relies on notebooks and environment traces.
- Upstream SA requires replacing vLLM internals and uses a hard-coded Qwen tokenizer in the patched runner.

Because installing or launching those stacks would be heavier than this evaluation target, I used tLLM unit tests plus a toy trajectory case to validate the implemented subset.

## Checks Run

Commands are summarized with placeholders to avoid machine-specific paths.

1. GTTA consumer target tests:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 <repo-venv>/bin/python -m pytest -q -p no:cacheprovider test/consumers/gtta
   ```

   Result: `12 passed`.

   Current result after prompt smoke refresh: `17 passed, 1 skipped`.

2. Consumer-local dynamics toy coverage:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 <repo-venv>/bin/python -m pytest -q -p no:cacheprovider test/consumers/gtta/test_gtta_dynamics_toy_unit.py
   ```

   Result before adoption coverage: `2 passed`.

   Current result after prompt smoke refresh: `7 passed, 1 skipped`.

   The toy case ingests three feedback transitions sharing an action string
   across different tasks and states. The consumer then:

   - returns only the matching dynamics record for a requested
     task/state/action;
   - returns copied query results so callers cannot mutate the retained store;
   - keeps candidate ranking scoped to the requested task and state;
   - emits upstream-shaped JSONL dynamics context with deterministic ordering.
   - proves a fake capture runner receives the `PromptContextEntry`-injected
     prompt text and records the expected prompt index in the generation trace.

3. Toy trajectory check:

   - Config: `max_queue_size=3`, `score_momentum=0.25`.
   - Inputs: positive reward for `click search`, zero reward for `click ad`, another positive reward for `click search`, and one overflow record.
   - Result:
     - accepted flags: `[True, True, True, False]`
     - after partial drain: `queued_trajectories=1`, `dropped_trajectories=1`, `processed_trajectories=2`, `update_count=2`
     - final stats: `queued_trajectories=0`, `dropped_trajectories=1`, `processed_trajectories=3`, `update_count=3`
     - final scores are scoped by task and state signature, so the same action
       string can have different values in different states/tasks.
     - selected score order placed `click search` first.

4. Dependency probe:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 <repo-venv>/bin/python - <<'PY'
   import importlib.util
   for name in ["browsergym", "bgym", "agentlab", "bfcl", "vllm"]:
       print(name, importlib.util.find_spec(name) is not None)
   PY
   ```

   Result: environment/evaluation dependencies were missing; `vllm` was available.

5. Upstream WebArena import probe:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<upstream>/WebArena/src <repo-venv>/bin/python - <<'PY'
   import run_agent
   PY
   ```

   Result: failed at missing `beartype`, before reaching browsergym service setup.

6. Cached tiny-model prompt-context smoke:

   ```bash
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TLLM_GTTA_RUN_CACHED_MODEL_SMOKE=1 \
     PYTHONDONTWRITEBYTECODE=1 <repo-venv>/bin/python -m pytest -q -p no:cacheprovider \
     test/consumers/gtta/test_gtta_dynamics_toy_unit.py::GTTACachedModelSmokeTest::test_prompt_context_changes_tiny_gpt2_logits
   ```

   Current local-cache result: `1 passed`.

   This uses a cached `sshleifer/tiny-gpt2` model with offline flags. The
   consumer writes a `PromptContextEntry`, the runtime prompt-context helper
   injects it into the prompt, and the model's next-token logits differ from the
   base prompt logits.

## Alignment Conclusion

**Against the tLLM devlist target:** the implementation is aligned. It preserves the environment boundary, avoids large dependencies, validates and queues feedback, updates task/state/action-scoped scores in the expected direction, exposes counters that prove work happened, and provides a local candidate-ranking API.

**Against Task 6 runtime contracts:** the consumer side is now aligned at the
contract boundary. It can ingest runtime `TrajectoryEvent` records and can
format ranked dynamics or upstream-shaped environment-dynamics JSONL as runtime
`PromptContextEntry` text. The runtime prompt-context and trajectory-event ports
already exist, so the remaining work is integration, not a new consumer-local
contract.

**Against full upstream GTTA:** this is only an inspired subset, not a full reproduction. The following upstream mechanisms are not reproduced:

- no environment dynamics extraction from observation deltas;
- no dynamics filtering pipeline;
- no closed-loop task-time prompt construction driven by policy/runtime hooks;
- no BFCL environment-dynamics loading by environment/class;
- no WebArena/browsergym action execution loop;
- no candidate-action pairwise judging, rejection tuning, or q-judge/reranker dataset generation;
- no SA parametric hidden-state/prompt-adapter update inside vLLM;
- no policy/model update path connected to real generation.

So the correct classification is: **tLLM GTTA is a dependency-light trajectory feedback consumer that implements a useful local subset of GTTA's feedback idea, but it is not a faithful reproduction of the full GTTA algorithmic stack.**

## Gaps And Risks

- State/task-aware dynamics records, candidate ranking, and deterministic prompt
  snippets/JSONL context exist locally, but they are not connected to a
  closed-loop tLLM policy. Callers must manually choose when to write trajectory
  events, when to build prompt context, and when to apply it.
- The reward update is scalar exponential smoothing, not upstream DG prompt grounding, q-judge preference learning, or SA parameter adaptation.
- The cached tiny-model smoke proves injected context changes logits, but it is
  not a WebArena/BFCL policy evaluation and does not measure task success.

## Recommendation

Keep the current implementation documented as **GTTA-inspired**. If the goal is
closer reproduction, the next integration work should use the runtime contracts
already present and add these hooks:

- environment or generation-loop wiring that writes `TrajectoryEvent` objects
  through the runtime trajectory-event port into this consumer or shared store;
- DG prompt-injection plumbing that selects consumer dynamics snippets and
  writes `PromptContextEntry` objects through the prompt-context port;
- sampler, reranker, or policy-facing use of the ranking API;
- a separate SA hidden prompt-adapter or hidden-state update design.

A faithful full reproduction should not be hidden inside this simple score-table
consumer.
