# tLLM Documentation

tLLM is an auxiliary-workload compiler for vLLM. New work is declared with
`tllm.workload`; its task graph is fine-grained for analysis, while signed
`ExecutionRegion`s are the coarse-grained physical execution units. Rust makes
and signs that decision; Python mechanically binds the signed plan.
Program/Consumer lanes, streams, and events remain documented as migration
compatibility, not as the optimization model.

## Reading order

1. [Auxiliary workload compiler](workload-compiler.md) — the current frontend, signed Region contracts, and migration boundary.
2. [Region authority v1](region-authority.md) — frozen Rust/Python authority, fail-closed matrix, and migration checklist.
3. [Quickstart](quickstart.md) — native ESamp orientation and the retained Program example.
4. [Write a consumer](write-a-consumer.md) — `tllm.workload` guidance plus the legacy Program compatibility API.
5. [Architecture](architecture.md) — the Region execution stack and compatibility layers.
6. [ESamp reference](esamp.md) — the built-in test-time training consumer.
7. [Native In-Place TTT](inplace-ttt.md) — signed exact fast-weight tasks and shared immutable overlays.
8. [TLM q/v LoRA](tlm.md) — full HF/PEFT side training and immutable versions through vLLM's native LoRA path.

## The one-paragraph mental model

For a native workload, declare `@workload` and `@task` nodes with typed inputs,
outputs, effects, timing, and optional affine access. Rust validates the full
graph and signs Region boundaries, bindings, materializations, lifetimes, and
capture policy. The legacy `@flow`/`@extern` Program lane remains available as
an adapter while workloads migrate; it is not used to make new physical
scheduling decisions.

Unsupported native or compatibility declarations fail closed; there is no
silent fallback.

With `torch==2.13.0+cu129` and `vllm==0.27.1`, GPU correctness covers exact
eager in-process native Dummy, native Recirculation, and native ESamp on
MRv1/MRv2. Recirculation also passed EngineCore MP on both runners, replaying
two requests and seven prompt rows with zero fallback. The
latest ESamp short smoke recorded six losses, six updates, state version six,
and zero fallbacks. Native ESamp exact eager Regions also passed EngineCore MP
correctness on both runners: loss, update, state-version, and intervention
counters were nonzero, with zero fallbacks and zero replays. This is not a
throughput claim. Model-backed native piecewise CUDA Graph execution,
Recirculation capture/hook-retirement integration, and shape-miss epoch
switching are not yet claims.

中文文档见 [doc_zh/README.md](../doc_zh/README.md)。
