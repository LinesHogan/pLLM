# Documentation

The guides describe the current implementation. Retired API guides and migration
reports are not maintained alongside the current architecture.

- [Installation and ESamp example](../README.md)
- [In-Place-TTT dense overlays](inplace-ttt.md)
- [Current architecture and ownership](architecture.md)
- [Ask your agent to write a workload: inputs and workflow](guides/agent-workloads.md)
- [Workload authoring contract](guides/authoring.md)
- [Validation and release checks](guides/validation.md)
- [Benchmark interpretation](guides/benchmarking.md)
- [Consolidation plan](architecture-consolidation-10k-plan.md)
- [Programmable inference design and shared boundaries](programmable-inference-next-design.md)
- [Shared high-performance primitive design](pllm-next-stage-design.md)

The consolidation plan records acceptance and exclusions. The executable evidence
list lives in `test/evidence_manifest.toml`; there is no second progress ledger.
