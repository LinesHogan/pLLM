# Architecture consolidation ledger

This ledger records the implementation state of
`architecture-consolidation-10k-plan.md`.  It is deliberately separate from
the AER/PTTS research note, which is not an implementation artifact.

## Baseline

| Fact | Value |
| --- | ---: |
| Commit | `12551b8` |
| Production/support source | 133,210 lines |
| Pytest collection | 1,349 items |
| Rust tests | 54 items |
| Total behavioral evidence | 1,403 items |

All Python commands use `/workspace/pllm/.venv/bin/python`.

## Current checkpoint

| Fact | Value | Cumulative change from `12551b8` | Change in FTTT cutover |
| --- | ---: | ---: | ---: |
| Production/support source | 131,973 lines | -1,237 lines | -912 lines |
| Pytest collection | 1,333 items | -16 items | -16 items |

The FTTT authority-retirement slice added no counted source. It deleted 909
production lines from `tllm/consumers/fttt` and three counted support lines
from static gates, for `+0 / -912 / net -912` under `SOURCE_LOC`. It also
deleted 1,026 test lines from the superseded unit and GPU smoke; evidence 21
and 38 remain as the canonical CPU and CUDA survivors. Documentation and JSON
config changes are excluded from `SOURCE_LOC` and are not used to offset these
deletions.

## Tranche accounting

| Tranche | Added | Deleted | Moved unchanged | Net | State |
| --- | ---: | ---: | ---: | ---: | --- |
| A — ledger and survivor extraction | — | — | — | — | active |
| B — canonical vertical compression | 249 | 574 | 0 | -325 | active |
| C — workload cutovers | 0 | 912 | 0 | -912 | active: FTTT retired |
| D — backend quarantine and closed MP | — | — | — | — | pending |
| E — legacy authority removal | — | — | — | — | pending |
| F — AER/PTTS final core | — | — | — | — | pending |
| G — reachability and final compression | — | — | — | — | pending |

Update this table from tracked physical line counts at each tranche boundary.
Moves are recorded separately and never counted as deletion.

## Production keep/move/delete authority

| Current owner | Disposition | Unique content that survives | Final owner |
| --- | --- | --- | --- |
| `tllm/canonical`, `tllm/contracts` | keep and compress | semantic descriptors, tensor math, FFI DTOs | canonical graph/execution core |
| `tllm/canonical_runtime` | delete in B | no workload-specific orchestration | generic executor/backend seam |
| `tllm/canonical_vllm` | merge in D | vLLM probes and physical adapter mechanics | `tllm/backends/vllm` |
| `tllm/runtime` | delete in E | MP/backend facts only | closed MP and generic backend owners |
| `tllm/workgraph`, `tllm/schedule`, `tllm/workload`, `tllm/program` | delete in E | independent math/loaders only | canonical descriptors/workloads |
| `tllm/consumers` | delete by workload in C/E | math, loaders, judge, environment | one canonical workload owner each |
| `tllm/kit`, `tllm/ports` | delete in E | none after typed execution cutover | canonical contracts |
| Rust non-canonical graph/schedule | delete in E | no duplicated authority | `rust/tllm_core/src/canonical` |
| provisional `canonical/ptts/dense_page*` | replace in F | lifecycle invariants and CUDA kernels | PageBundle/generational arena |

## Workload cutover manifest

| Workload | Unique content to retain | Old owner removal gate | Evidence IDs |
| --- | --- | --- | --- |
| Dummy | minimum graph/sign/plan/execute declaration | generic public factory | 20 |
| FTTT | edit math, judge, feedback retry | legacy owner removed; canonical host transaction and receipt only | 21, 38 survivors |
| GTTA | environment state/update/query/FIFO | generic host transaction | 22 |
| In-Place-TTT | dense fast-weight math and layout | generic request PTTS/overlay | 23, 32 |
| TLM | loader, objective, q/v LoRA math | global PTTS and online/offline timing | 24, 33 |
| ESamp | request math, sampler guidance, random init | request PTTS and public vLLM path | 25, 30, 31, 37 |
| ReFT | loader and LoReFT activation math | signed activation-patch attachment | 26, 34 |
| Recirculation | row selection and replay math | typed replay effect | 27, 35 |
| OverRIDE | reweight and Adam math | generic sampler/state transition | 28 |
| TTRL | verifier, rewards and update math | typed host/control quorum | 29 |

The missing historical `esamp-base-consumer` worktree tests are behavior
sources, not paths to recreate.  Their base-consumer, hidden-storage,
per-request-training and benchmark-selection contracts are absorbed into
evidence 15, 25 and 30 plus the ESamp aligned cutover benchmark.

## Hard gates

- Every cutover commit adds or updates its survivor evidence before removing
  the live legacy implementation and tests.
- Production source is non-increasing in A, decreases in B/C/E/F, and may grow
  in D only for an identified backend/MP seam.
- From E onward the total collected behavioral evidence is at most 50.
- Final source is 22,000–30,000 lines; all legacy imports and named authorities
  are absent; strict Pyright is 0 errors/0 warnings.
- Training evidence reports positive loss/update counts and every runtime
  evidence item reports zero fallback.
