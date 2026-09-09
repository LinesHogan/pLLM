# Validation and Release Checks

Run from the repository root. This checkout uses
`/workspace/pllm/.venv/bin/python`, including when working in a worktree.
During implementation, select checks for the changed boundary; the complete
list is a release gate, not a mandatory rerun after every edit.

## CPU, Rust and Static Checks

```bash
python -m maturin develop --offline
python -m pytest -q
cargo test --manifest-path rust/pllm_core/Cargo.toml
python tools/check_import_layers.py
python tools/check_region_authority.py
python tools/check_typing.py
```

Use the activated repository virtualenv for these commands. The offline native
build requires cached Rust dependencies. One strict config covers the maintained
implementation, benchmark scenarios and examples, with zero errors/warnings.
CUDA tests skipped on a CPU-only host are not GPU evidence.
The import gate requires zero runtime violations, including function-local imports.
There is no baseline/rebase workflow; dependency rules are reviewed directly in the checker.

## GPU Boundaries

Run sequentially on a compatible CUDA host with vLLM 0.27.1 and the
Qwen/Qwen2.5-0.5B-Instruct checkpoint available:

```bash
python test/gpu/vllm_esamp_smoke.py
python test/gpu/vllm_mrv1_smoke.py
python test/gpu/vllm_inplace_ttt_smoke.py
python test/gpu/vllm_tlm_smoke.py --mode online
python test/gpu/vllm_tlm_smoke.py --mode offline
python test/gpu/vllm_reft_smoke.py
python test/gpu/vllm_conditional_replay_smoke.py
python test/gpu/vllm_override_smoke.py
python test/gpu/vllm_recirculation_smoke.py
python test/gpu/esamp_cuda_graph_replay_smoke.py
python test/gpu/fttt_cuda_smoke.py
python test/gpu/enginecore_smoke.py
python test/gpu/enginecore_smoke.py --mrv1-captured
```

The first scenario runs plain inference followed by attached ESamp in one
process. Training must have nonzero loss/update counts, request state must be
cleaned up and fallback must be zero. The CUDA Graph scenario covers the
training core, not a vLLM capture-mode support matrix.

## Release Gate

Evidence 39, collected by pytest, builds an sdist, builds an optimized release wheel from that
extracted source archive and installs it in an isolated directory. It verifies
Python/native-module provenance, executes the Rust codec and checks tensor
dependencies, license packaging and absence of retired packages/plugins.

Build distribution artifacts from the repository root:

```bash
python -m maturin build --release
python -m maturin sdist
```

Wheel success alone does not authorize a release. Before tagging or publishing,
complete the [consolidation acceptance](../architecture-consolidation-10k-plan.md), all selected
39 original evidence contracts, composition evidence 40/41 and sampler-feedback
evidence 42, shared/independent LoRA evidence 43/44, and performance comparisons
45/46 and independent adaptation evidence 47 (26 pytest, 8 Rust, 13 runtime commands and 3 benchmark comparisons;
50 selected executions), plus the runtime boundaries and 30k source
ceiling. The original 22k–30k interval was an estimate, not a minimum size to fill.
No throughput or unsupported topology claim may be inferred from a short
correctness smoke. This gate does not authorize commits, tagging or publishing.
