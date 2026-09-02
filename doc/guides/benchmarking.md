# Benchmarking

Use `/workspace/pllm/.venv` for every runtime and pytest command. The
current verified environment is vLLM 0.27.1; historical vLLM 0.10.1.1 logs
are not an interchangeable runtime baseline.

## Comparison rules

- Compare throughput only between arms with the same model, prompts, expanded
  request count, decode length, sampling parameters, runner, MP topology,
  memory limits, compile/capture mode, and warmup/round counts.
- A training arm is invalid when `loss_count == 0`, its expected update density
  is missing, or fallback is nonzero. Higher throughput cannot compensate for
  missing work.
- The small `side_train`-style smokes are ratio/debug checks. Their absolute
  token rate is not comparable to the aligned per-request workload.
- After an abnormal exit, inspect `nvidia-smi` for a stale
  `/workspace/pllm/.venv/bin/python` or EngineCore worker before trusting a
  later OOM. Never terminate unrelated GPU processes.

## Aligned ESamp Workload

Use `test/gpu/esamp_ratio_benchmark.py` for the maintained raw/capture-only/
legacy/native ESamp ratio harness. A comparable 0.5B shape uses:

```text
model=Qwen/Qwen2.5-0.5B-Instruct
dtype=bfloat16
benchmark_batch_size=8
sampling_n=16
benchmark_max_new_tokens=256
max_model_len>=512
max_num_seqs=128
max_num_batched_tokens=4096
benchmark_warmup_rounds>=1
benchmark_disable_prefix_caching=true
benchmark_ignore_eos=true
sampling_temperature=0.8
sampling_top_p=0.95
sampling_top_k=-1
distiller_lr=1e-3
model_bank_flush_interval=1
model_bank_initializer=svd
model_bank_initializer_svd_method=ffn_fast_svd
trajectory_topk=1
```

Set `VLLM_USE_FLASHINFER_SAMPLER=1` when available. On seeded requests vLLM
may still report that FlashInfer does not support per-request generators and
fall back to PyTorch sampling; record that as part of the arm identity. Do not
disable the compile cache on older installations.

`test/gpu/esamp_ratio_benchmark.py` is the stricter
raw/capture-only/legacy/native ratio harness. It fixes 128 active sequences
and 4096 batched tokens and requires sampler, capture, training-density, and
scheduler-shape evidence. The `native` arm exercises shared-state
`ESampWorkload`; it is not per-request or model-bank evidence.

One 2026-08-31 diagnostic pair on the verified vLLM 0.27.1 environment used
the 0.5B model, MRv2 in-process, batch `8 × n16`, 128 output tokens, one
warmup, and one measured round. Raw measured `41,495.6 tok/s`; native measured
`42,973.9 tok/s`, ratio `1.0356`. Native recorded exactly 127 losses, 127
intervention samples, 127 Triton candidate-kernel uses, 512 Region executions,
509 replays, and zero fallback. This one pair is shared-state diagnostic
evidence, not the formal five-pair native-vs-legacy retirement gate.

## Memory precondition

vLLM computes requested GPU memory from total device memory, not currently
free memory. On a 24,564 MiB device, `gpu_memory_utilization=0.5` requests
about 12,282 MiB before ESamp attribution is possible. If less is free, the
raw arm is expected to fail the same startup check. A 0.45 run can diagnose
phases but cannot stand in for a 0.5 acceptance run.

Pin `max_num_seqs` and `max_num_batched_tokens`: vLLM 0.27 defaults can be
larger than the aligned 128/4096 shape and change profile/CUDA-Graph memory.
Record initial free/requested memory, available KV cache, graph estimate,
failure phase, allocated/reserved memory, loss/update counts, and capture
counters for every arm.

Historical context: aligned per-request experiments commonly reached 4k+
tokens/s; one 0.5B smoke reached 11,672 tokens/s. These numbers are diagnostic
context only, never a substitute for a same-run ratio.
