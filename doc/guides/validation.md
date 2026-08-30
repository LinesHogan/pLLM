# Native Workload Validation

Validate legacy and native implementations in independent processes with the
same backend, inputs, seed, and configuration. Freeze only a minimal numerical
oracle or golden before deleting the old production implementation; do not
keep a runnable dual stack indefinitely.

For deterministic paths compare tensors, tokens, losses, update counts, and
state versions exactly or under a declared tolerance. For stochastic sampling,
compare token-by-token under a fixed RNG; when that is impossible, use a
predeclared distribution-level test. Also verify retry/barrier/trajectory,
version pinning, request cleanup, and epoch replacement.

GPU coverage must include the supported MRv1/MRv2 and in-process/EngineCore-MP
cells, plus eager/CUDA Graph and distributed boundaries when the workload
claims those capabilities. Training evidence requires positive loss and update
counts. Every fallback count must be zero.

## ESamp deletion gate

Legacy ESamp and its Program baseline remain reachable until native ESamp has:

1. numerical and lifecycle parity for shared, per-request, and model-bank
   state;
2. compact capture, sampler intervention, request cleanup, and claimed CUDA
   Graph coverage;
3. an aligned native-vs-legacy run from the workload in
   [benchmarking.md](benchmarking.md), with real training work and zero
   fallback;
4. throughput ratio `>= 0.98`. A result below `0.95` blocks deletion outright.

The existing `docs/p5_paired_benchmark_report.json` measures Program against
legacy and records a ratio near 0.7968 with `gate_pass=false`; it is evidence
that deletion remains blocked, not native takeover evidence. Native ESamp's
current eager GPU smokes establish shared-state correctness only.
