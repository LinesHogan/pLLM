# Validation and Release Checks

Run from the repository root. This checkout uses
`/workspace/pllm/.venv/bin/python`, including when working in a worktree.

## CPU, Rust and Static Checks

```bash
python -m maturin develop --offline
python -m pytest -q
cargo test --manifest-path rust/tllm_core/Cargo.toml
python tools/check_import_layers.py
python tools/check_region_authority.py
python tools/check_typing.py
```

Use the activated repository virtualenv for these commands. The offline native
build requires cached Rust dependencies. Strict and basic typing both have a
zero-error baseline. CUDA tests skipped on a CPU-only host are not GPU evidence.

## GPU Boundaries

Run sequentially on a compatible CUDA host with vLLM 0.27.1 and the
Qwen/Qwen2.5-0.5B-Instruct checkpoint available:

```bash
python test/gpu/canonical_vllm_esamp_smoke.py
python test/gpu/canonical_vllm_inplace_ttt_smoke.py
python test/gpu/canonical_esamp_cuda_graph_replay_smoke.py
python test/gpu/canonical_fttt_cuda_smoke.py
```

The first scenario runs plain inference followed by attached ESamp in one
process. Training must have nonzero loss/update counts, request state must be
cleaned up and fallback must be zero. The CUDA Graph scenario covers the
training core, not a vLLM capture-mode support matrix.

## Release Gate

Evidence 39, collected by pytest, builds an sdist, builds a wheel from that
extracted source archive and installs it in an isolated directory. It verifies
Python/native-module provenance, executes the Rust codec and checks tensor
dependencies, license packaging and absence of retired packages/plugins.

Build distribution artifacts from the repository root:

```bash
python -m maturin build --release
python -m maturin sdist
```

Wheel success alone does not authorize a release. Before tagging or publishing,
complete the [handoff](../architecture-consolidation-handoff.md), all selected
39 evidence contracts (maximum 50), supported runtime boundaries, and the
22k–30k source gate. Consolidate the release branch without discarding pending
worktree changes. No throughput or unsupported topology claim may be inferred
from a short correctness smoke.
