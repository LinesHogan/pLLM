Submit one small, concrete candidate.py edit for this round, then finish. The
external scorer evaluates it after you exit and returns feedback to the next round.
This sandbox contains only docs and seed files: torch/pllm packages, repository
source and GPUs are unavailable. Use the supplied API docs; do not search for
packages, install dependencies or run inference here.

Improve Qwen/Qwen2.5-0.5B-Instruct accuracy on the pinned first 200 GSM8K test
questions while retaining at least 90% of vanilla throughput. Edit candidate.py
and define WORKLOAD and NEUTRAL as pllm.Workload instances. NEUTRAL must disable
every effect. Use only the documented pLLM API and torch math; do not read files,
fetch data, alter scoring, or depend on other workspace files from candidate.py.

Seeds: candidate.py is the current best; dola.py uses the DoLa built-in;
early_exit.py uses an active= loop over the last two layers. You may replace the
algorithm or tune it. The leaderboard contains previous accuracy, throughput,
explain and check feedback. A failed contract or throughput below 90% disqualifies
a candidate. Highest accuracy wins; ties go to lower explain FLOPs. Vanilla is a
valid candidate. Each proposal is scored once per source.

Quality uses greedy generation, up to 512 tokens, a fixed zero-shot chat prompt,
and exact numeric match of the last number in the response. Throughput uses the
aligned batch=8, tokens=256, sampling_n=16, warmup=1, rounds=3 settings. Model
graphs are enabled when supported; early exits require eager model execution.
The throughput floor always uses the vanilla graph reference. Submit only
candidate.py; execution and scoring happen outside this sandbox.
