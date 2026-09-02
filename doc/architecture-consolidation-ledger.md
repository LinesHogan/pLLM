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

| Fact | Value | Cumulative change from `12551b8` | Change since FTTT checkpoint |
| --- | ---: | ---: | ---: |
| Commit | `3c91d04` | — | after `753bdfd` |
| Production/support source | 126,745 lines | -6,465 lines | -5,228 lines |
| Pytest collection | 1,295 items | -54 items | -38 items |

The FTTT authority-retirement slice added no counted source. It deleted 909
production lines from `tllm/consumers/fttt` and three counted support lines
from static gates, for `+0 / -912 / net -912` under `SOURCE_LOC`. It also
deleted 1,026 test lines from the superseded unit and GPU smoke; evidence 21
and 38 remain as the canonical CPU and CUDA survivors. Documentation and JSON
config changes are excluded from `SOURCE_LOC` and are not used to offset these
deletions.

The backend import-frontier cleanup in `d59920f` removed the deprecated
`tllm/backends/vllm/typed_recurrence.py` re-export and its region-authority gate
entries.  It also removed orphan `tools/getattr_probe_baseline.json`, which was
a static-gate artifact with no live consumer.  Counted source changed by
`+0 / -21 / net -21`; the JSON deletion is excluded from `SOURCE_LOC` and does
not change `TEST_COUNT`.  This is recorded as narrow frontier cleanup, not as
completion of the Tranche D backend seam rewrite.

The canonical vertical-compression checkpoint now includes `753bdfd`
(`+3,868 / -8,972 / net -5,104` under `SOURCE_LOC`) and `a822028`
(`+296 / -398 / net -102`) on top of the previous Tranche B ledger entry
(`+249 / -574 / net -325`).  The `b122512` plus `b66448c` route/typing gate
repair contributes only `+5 / -6 / net -1` under `SOURCE_LOC`; it is recorded
as public boundary cleanup, not a separate architecture deletion.  Current
`SOURCE_LOC` was recalculated at `d59920f` with the transparent command from
the plan and matches the cumulative delta: `133,210 - 6,465 = 126,745`.

Tranche A static meta-test cleanup deleted 19 pytest items from import-layer,
region-authority, native import-boundary and getattr-baseline wrappers.  These
were pytest shells around static gates or import discipline, not numerical,
state, runtime, loss/update/version or cleanup oracles.  The corresponding
static gates remain directly executable and do not count as behavioral evidence.
The follow-up decorative legacy test cleanup in `3c91d04` deleted five more
test files and reduced pytest collection by 11 items, from 1,306 to 1,295.  The
deleted files covered import-boundary subprocess checks, MRv1/MRv2 shadow
compatibility matrices, registry/plugin shells, enum/catalog mappings and
facade identity.  They did not contain numerical, state, loss/update, version,
cleanup, GPU or survivor evidence.

## Tranche accounting

| Tranche | Added | Deleted | Moved unchanged | Net | State |
| --- | ---: | ---: | ---: | ---: | --- |
| A — ledger and survivor extraction | — | — | — | — | active: static/decorative test cleanup; pytest collection -54 |
| B — canonical vertical compression | 4,418 | 9,950 | 0 | -5,532 | active: canonical runtime attachments, ESamp task merge and boundary gate repair |
| C — workload cutovers | 0 | 912 | 0 | -912 | active: FTTT retired |
| D — backend quarantine and closed MP | 0 | 21 | 0 | -21 | active: backend import frontier cleanup; seam rewrite still pending |
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
