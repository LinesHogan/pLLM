# Benchmark Interpretation

There is no maintained throughput benchmark for the current canonical public
path yet. The retired consumer/Program benchmark scripts and their results
remain in Git history. The current GPU smokes establish correctness only.

## Comparable Runs

Compare arms with identical model, prompts, request count, token budget,
sampling parameters, runner, topology, memory budget, capture mode and warmup.
A training arm with zero loss/update or nonzero fallback is invalid.

Historical aligned ESamp experiments used Qwen2.5-0.5B or 7B, bfloat16,
batch 8, sampling n=16, at least 256 generated tokens, max_model_len >=512,
temperature 0.8, top_p 0.95, top_k -1, ignored EOS and disabled prefix caching.
The current public attachment accepts n=1; it does not support this entire
historical workload. Small correctness-run token rates are not comparable to
the historical multi-request throughput figures.

Reintroduce an aligned benchmark only at the relevant workload cutover or
release gate, after the required functionality and evidence exist. Use random
ESamp initialization as required by the consolidation plan. Do not restore the
deleted legacy runtime as a permanent benchmark baseline.

Use the repository virtualenv. Record the actual vLLM/PyTorch/CUDA versions,
sampler implementation and any upstream generator fallback. After abnormal
termination inspect GPU process ownership before retrying; never terminate
unrelated workers.
