# Algorithm search

This example searches workload math for Qwen2.5-0.5B-Instruct on the first 200
GSM8K test questions. It makes no quality claim beyond that slice. It adds no
search API to pLLM. Agent proposals are **not reproducible**; the pinned greedy
scores are reproducible.

From the repository root, using its venv (bubblewrap and the selected authenticated
agent CLI must already be installed):

```bash
/workspace/pllm/.venv/bin/python examples/algorithm_search/search.py \
  --rounds 5 --output outputs/search-1
/workspace/pllm/.venv/bin/python examples/algorithm_search/search.py \
  --replay outputs/search-1/ledger.jsonl --output outputs/replay-1
/workspace/pllm/.venv/bin/python examples/algorithm_search/search.py \
  --replay outputs/search-1/ledger.jsonl --output outputs/replay-2
```

Each output directory must be new. The default author is ai_eval's codex runner
with `gpt-6.1-sol`; `--agent kimi` remains selectable. `--model` overrides
the selected runner's default author model. Each author gets 900 seconds, as in
ai_eval; `--timeout` overrides that limit. The three initial seeds
are vanilla (an ordinary residual sum), the built-in DoLa, and a bounded
`active=` early-exit loop. Each round sees the committed README/doc snapshot,
task, seeds, current best candidate and complete leaderboard feedback. The
existing ai_eval bubblewrap sandbox excludes the repository and scorer. Allowed
credentials are copied into its private home so CLI OAuth refresh locks can be
written. Provider failures are recorded as `INFRA`, rather than failed algorithm scores.

To score a standalone, self-contained candidate defining `WORKLOAD` and `NEUTRAL`:

```bash
/workspace/pllm/.venv/bin/python examples/algorithm_search/score.py \
  examples/algorithm_search/dola.py --output outputs/dola-score.json
```

The output parent must exist. CPU explain and `pllm.testing.check` gate GPU work.
GPU scoring takes `/tmp/pllm-gpu.lock` and CPU affinity 0-15, runs
`check_on_gpu`, measures greedy accuracy, then calls `pllm.benchmark.compare`.
Every engine is scoped to its context manager. Use the local RTX 4090; the
standing pool remains available for the repository's full verifier.

Throughput uses bfloat16, batch 8, 256 tokens, 16 completions, warmup 1, three
rounds, seed 7, ignore-EOS, temperature 0.8, top-p 0.95, top-k -1 and prefix
caching off. GPU memory utilization is 0.7; model/task capture capacity is 8 GiB
(a planning limit, sized to admit the DoLa seed's conservative trace reservation).
Graphs are on where admitted;
early-exit custom forwards require eager model execution. Their throughput floor
still uses a separately measured vanilla **graph** reference. Accuracy uses
greedy generation with up to 512 tokens, model length 2048 and batch 8, with a
fixed zero-shot Qwen chat prompt. The last number in the response is normalized
and compared exactly to the GSM8K answer.

The ledger embeds candidate source, pLLM's signed declaration fingerprint, git
SHA, pinned model/dataset commits and dataset content hash, question IDs, seeds,
runtime versions, agent CLI version/model, explain and check text, GPU type,
answers and three-round timing samples. Highest accuracy passing all contracts
and the 90% throughput floor wins; ties use lower explain FLOPs. Identical
fingerprints reuse the score and do not add a ledger row. Each attempt retains
its own score.json and transcript, including duplicate provenance. The final
candidate.py in the output directory is the best passing source.

Replay imports embedded sources in fresh processes without an agent, reruns all
gates, requires identical fingerprints and greedy answers, and reports reference
and replay throughput with all round samples. Compare timings under matching
hardware and load: three rounds within one engine do not estimate noise between
fresh processes. Failed candidates are re-scored too; records without a
candidate source (such as provider failures) are skipped. Dataset download uses only the
standard library; model weights must be available to vLLM at the pinned revision.
