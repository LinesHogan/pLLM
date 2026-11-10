# Agent authoring regression suite

This suite measures whether users developing with agents can write a workload from
pLLM's published documentation. The default author is codex with `gpt-6.1-sol`.
This is an **easy to write** regression, not a model quality or throughput benchmark.

Run from a checkout whose framework/docs match the baseline revision:

```sh
/workspace/pllm/.venv/bin/python tools/ai_eval/run.py \
  --revision main --output tools/ai_eval/results/codex-main-new
```

Requirements: Linux user namespaces, `bwrap`, the authenticated codex CLI at
`/home/user/.local/bin/codex`, and the repo virtualenv. `--agent codex|kimi` selects
the author; `--model` overrides its default (`gpt-6.1-sol` for codex,
`gpt-6.1-sol` for Kimi). Kimi uses `/home/user/.codex/auth.json`.
Inspect each CLI's `--help` when updating versions. codex 1.4.12 uses
`-p ... --model ... --output-format stream-json --dangerously-skip-permissions
--add-dir /workspace`; permissions are skipped only inside bubblewrap. Kimi uses
`--prompt ... --output-format stream-json`, without `--auto`.
Both run non-interactively. Two tasks run concurrently by default (`--jobs 2`).
Each task gets a fresh session, no evaluator feedback, and a
900-second default authoring limit. A timeout is exit 124; any saved submission
is still scored. Infrastructure failures remain explicit, not documentation
failures. `--tasks` selects a subset; `--gpu` adds CUDA tensor comparisons under
`flock /tmp/pllm-gpu.lock taskset -c 0-15`. No engine is started.

## What the author sees

`git archive REV README.md doc/ ':(exclude)doc/dev'` makes a clean documentation snapshot without the
contributor docs, which describe pLLM's internals rather than how to write a workload. The runner
adds exactly one spec as `TASK.md`. A bubblewrap mount allowlist exposes that
workspace, fresh CLI home, operating-system binaries/libraries and the selected CLI
executable. It hides the entire host checkout, `.git`, venv, `test/`, `examples/`,
all `pllm/` (including **every** `pllm/workloads` algorithm), and this harness's
references, tests, scores and other submissions. Thus neither reading source nor
Python introspection can reveal an algorithm. The sandbox has no importable `pllm`
or Torch runtime. Doc tutorial code/output remains available because it is part
of the documented authoring surface. Broken links to hidden examples stay broken.

Only the selected CLI's authentication files are bound read-only into a fresh home,
without prior sessions, user/project skills, history or logs. codex reads
`~/.codex/auth.json`; no other host codex configuration
is exposed. Kimi receives its config.toml, device_id, region, credentials and oauth
paths when present. No credential contents are printed or copied into results.
Temporary home and workspace are removed in `finally`;
only submission, notes and transcript are retained. The CLI needs network for its
model service; network is not a security boundary, and the spec forbids external
sources. This evaluates cooperative authors, not adversarial agents. Python
submissions execute in a separate trusted scorer process with a 180-second
limit. Run this suite only with an author/model you trust to produce executable
Python. The isolation test checks actual mount visibility, not prompt compliance.

## Tasks and acceptance

Eight abstracts in `specs/` fix tensor math, initialization, state scope,
lifecycle, token phase, timing, constructor defaults and a small submission
interface. CAA steers the MLP residual; ITI sums **already output-projected** head
directions at the attention residual. Prompt LoRA uses a deterministic miniature
hypernetwork. ESamp uses request AdamW; TTT uses framework-managed chunks; adaptive
depth uses three bounded tail passes and monotone `active=` masks; TTRL votes on
normalized answer strings and trains through a real host execution.

`reference.py` holds executable reference workloads. `oracles.py` independently
computes acceptance math: rowwise float64 DoLa, explicit factor construction,
per-request linear predictions/gradients, PyTorch optimizer state, gradient-derived
memory updates, an independent fake decoder recurrence, and grouped voting plus
repeated real host steps. References and oracles are never mounted for the author.
Seeds/cases are fixed and versioned; “hidden” means hidden during authoring, not
secret from repository maintainers.

Each task earns one point for each gate (maximum 4):

| Gate | Evidence |
|---|---|
| imports/declares | Import, real Workload, requested effects/inputs/state scopes/optimizer/sites |
| `pllm.testing.check` | Existing CPU contract checker, with every explicit skip saved |
| CPU oracle | Independent math, plus real optimizer steps for ESamp/TTRL |
| `pllm.explain` | Successful plan; record graph/eager tier and eager reason |

Graph and eager both earn the explain point: the tier is a diagnostic, not a
performance claim. A `check` return with skips earns its contract point; it does
**not** establish skipped model execution. The CPU oracle does not load model
weights, run native attention, or prove model-site/KV integration; optional GPU
mode also compares tensor math only. Full-model GPU integration belongs to the
repo's established verification groups. No new GPU tests are introduced here.

`results.json` and the generated `README.md` contain the table, author wall time,
author turn count and per-task JSONL transcript links. Kimi turns are complete
assistant messages including tool calls; codex turns count completed
`step_update` events with `state=DONE` and `step_type=agent_response`, including
tool calls. Streamed deltas, tool replies and the terminal summary do not count.
Completed turns remain countable when a session times out before its summary. The transcript is the raw
combined CLI stream; incidental non-JSON diagnostics are preserved, and the turn
counter skips those lines. HTTP 429, quota, authentication and network errors are
**INFRA**, with a null
score rather than zero. One fresh attempt retries after a 10-second backoff;
`transcript-1.jsonl` and `transcript-2.jsonl` preserve both attempts, and
`transcript.jsonl` contains the final attempt. A successful provider recovery is
scored normally. Turns and wall time sum attempts, excluding backoff and scoring.
Records use `author_turns` and `author_version`, with the CLI name in `agent`. Each
task also retains `submission.py`, `score.json` with full tracebacks/check skips,
`scoring.log`, and `explain.txt` when planning succeeds. Generated submissions are
immutable evidence and excluded from formatting/lint traversal.

Review failures using transcript + scorer evidence, then record a primary
category and any secondary cause in `review.md` and task results:

- **docs gap**: required public usage is missing, stale, or only linked to hidden source.
- **API confusion**: available documented APIs were misunderstood or the requested math was misimplemented.
- **framework bug**: documented, correctly declared behavior fails in the planner/runtime; reproduce against the hidden reference before attributing this.

The report table shows INFRA separately; do not attribute provider failures to
documentation or API quality. Keep timeouts/harness failures explicitly unclassified.
Never infer a framework bug merely from an exception string or a `check` skip.
Do not change the reference to fit an author's output. Baselines stay pinned to
the recorded commit; preserve transcripts when comparing later revisions.

## Validate the harness

```sh
CUDA_VISIBLE_DEVICES= /workspace/pllm/.venv/bin/python -m pytest \
  tools/ai_eval/test_harness.py -q
/workspace/pllm/.venv/bin/python -m pyright \
  --pythonpath /workspace/pllm/.venv/bin/python tools/ai_eval/*.py
```

Tests verify all eight references, reject plausible incorrect math, and probe the
real author namespace. They use no GPU/model service. `tools/verify_groups.py`
classifies harness-only changes as needing no extra GPU feature group; the
standard static, CPU and GPU core gates still apply.
