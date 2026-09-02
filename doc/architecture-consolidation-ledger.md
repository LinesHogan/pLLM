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

| Fact | Value | Cumulative change from `12551b8` | Change since `71afa2e` checkpoint |
| --- | ---: | ---: | ---: |
| Commit | `3da233b` | — | after `71afa2e` |
| Counted production/support source | 96,589 lines | -36,621 lines | -26,658 lines |
| Pytest collection | 25 items | -1,324 items | -1,260 items |
| Rust tests | 3 items | -51 items | -51 items |
| Standalone correctness smokes | 4 items | baseline not fully enumerated | old non-survivor smokes deleted |
| Visible `TEST_COUNT` | 32 items | now below the 50-item ceiling | — |

The FTTT authority-retirement slice added no counted source. It deleted 909
production lines from `tllm/consumers/fttt` and three counted support lines
from static gates, for `+0 / -912 / net -912` under `SOURCE_LOC`. It also
deleted 1,026 test lines from the superseded unit and GPU smoke; evidence 21
and 38 remain as the canonical CPU and CUDA survivors. Documentation and JSON
config changes are excluded from `SOURCE_LOC` and are not used to offset these
deletions.

The backend import-frontier cleanups in `d59920f`, `3c6ba2c` and `99f7067`
removed the deprecated `tllm/backends/vllm/typed_recurrence.py` re-export, its
region-authority gate entries, the old-runtime `packed_hidden_snapshot` module,
three vLLM packed-hidden old-stack imports, the old-runtime `step_session`
module path, and the inert `prepare_model_region_step` compatibility hook.  The
packed-hidden buffer is now private to the runner-owned vLLM capture seam, and
the vLLM 0.27 step session is backend-private and strict-typed; signed public
DTOs remain in `tllm.contracts.packed_hidden`.  These commits also removed
orphan `tools/getattr_probe_baseline.json`, which was a static-gate artifact
with no live consumer.  Counted source changed by `+187 / -262 / net -75`; the
JSON deletion is excluded from `SOURCE_LOC`, and the packed-hidden internal
service test deletion reduced pytest collection by 10 items.  The remaining
backend old-stack reference scan reports 63 hits.  This is recorded as narrow
frontier cleanup, not as completion of the Tranche D backend seam rewrite.

The canonical vertical-compression checkpoint now includes `753bdfd`
(`+3,868 / -8,972 / net -5,104` under `SOURCE_LOC`) and `a822028`
(`+296 / -398 / net -102`) on top of the previous Tranche B ledger entry
(`+249 / -574 / net -325`).  The `b122512` plus `b66448c` route/typing gate
repair contributes only `+5 / -6 / net -1` under `SOURCE_LOC`; it is recorded
as public boundary cleanup, not a separate architecture deletion.  The
`41ad50b` re-export cleanup removed the unused
`tllm/canonical/workloads/__init__.py` and `tllm/contracts/__init__.py` export
catalogs for `+0 / -279 / net -279`; callers import concrete submodules
directly.  `SOURCE_LOC` was recalculated at `99f7067` with the transparent
command from the plan and matched the cumulative delta:
`133,210 - 6,798 = 126,412`.  Subsequent counted source deletion totals
`-3,165`: `3de5390` moved backend observable events to the backend protocol
seam for `+23 / -25 / net -2`, `19e3531` removed duplicated Python attachment
validation for `+0 / -201 / net -201`, `5adf50c` removed the vLLM WorkGraph
cold handoff for `+65 / -220 / net -155`, and `71afa2e` deleted Rust inline
test modules for `+0 / -2,807 / net -2,807`.  The new current `SOURCE_LOC`
is `123,247`.  The Rust inline test deletion is counted by the plan's
file-based `SOURCE_LOC` command but is recorded here as test evidence cleanup,
not production behavior removal.

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

The survivor-evidence collapse in `bda926d` and `71afa2e` deleted the old
WorkGraph, Program, runtime, workload, validator-matrix, decorative consumer,
orphan oracle-helper and non-survivor GPU smoke tests.  The commit deliberately
removed old native GPU scripts that imported `tllm.consumers` or old runtime
instead of treating renamed scripts as free tests.  It also pruned Rust tests
to the three canonical core bases explicitly named by section 5.3.  After the
collapse, full pytest collection is 25 items, `cargo test -- --list` reports
3 items, and the only standalone correctness smokes left under `test/gpu/` are
the four canonical/public survivor bases:
`canonical_esamp_cuda_graph_replay_smoke.py`, `canonical_fttt_cuda_smoke.py`,
`canonical_vllm_esamp_smoke.py`, and
`canonical_vllm_inplace_ttt_smoke.py`.

The old-authority deletion checkpoint `3da233b` removed the tracked
`tllm/program` package, the tracked `tllm/workgraph` package, Rust
`workgraph.rs`, ESamp's old Program/BaseConsumer path, old adaptive,
collective, deferred and transfer scheduling modules, stale repro workflows
and the old verification tool wrapper.  Recurrence declarations were moved
unchanged into `tllm.contracts.recurrence`, because Recirculation still has a
live typed recurrence semantic survivor; `GraphMode` moved into
`tllm.runtime.graph_policy`.  `tllm/runtime/ports/residual_bundle_dispatch.py`
was reduced to the direct residual-bundle owner
`tllm/runtime/ports/residual_bundle.py`; deferred/adaptive WorkGraph dispatch
was not restored.  The same checkpoint also removed stale `_core.pyi` and
`tllm.core` exports for Rust WorkGraph validators that no longer exist in the
Rust extension.  Under the plan's counted-source command this commit is
`+201 / -26,859 / net -26,658`, with 988 moved lines recorded separately by
Git rename accounting.  Import probes report `None` for `tllm.workgraph`,
`tllm.program`, `tllm.runtime.scheduling.deferred`, and
`tllm.runtime.ports.residual_bundle_dispatch`; `tllm.contracts.recurrence`
is now the recurrence owner.

## Tranche accounting

| Tranche | Added | Deleted | Moved unchanged | Net | State |
| --- | ---: | ---: | ---: | ---: | --- |
| A — ledger and survivor extraction | 0 | 2,807 | 0 | -2,807 | active: survivor collapse done; pytest 25, Rust 3, standalone smokes 4 |
| B — canonical vertical compression | 4,418 | 10,430 | 0 | -6,012 | active: canonical runtime attachments, ESamp task merge, duplicated Python validation pruning, boundary gate repair and re-export cleanup |
| C — workload cutovers | 0 | 912 | 0 | -912 | active: FTTT retired |
| D — backend quarantine and closed MP | 275 | 507 | 0 | -232 | active: backend event seam and vLLM WorkGraph cold handoff removed; seam rewrite still pending |
| E — legacy authority removal | 201 | 26,859 | 988 | -26,658 | active: Program and old WorkGraph packages removed; runtime shadow/deferred/adaptive/distributed residuals pruned |
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
| `tllm/runtime` | prune in E/F | MP/backend facts, native safe points, stats | closed MP and generic backend owners |
| `tllm/workgraph` | deleted in `3da233b` | finite recurrence declarations moved | `tllm.contracts.recurrence` |
| `tllm/program` | deleted in `3da233b` | none | canonical descriptors/workloads |
| `tllm/schedule`, `tllm/workload` | delete remaining old authority in E/F | independent math/loaders only | canonical descriptors/workloads |
| `tllm/consumers` | delete by workload in C/E | math, loaders, judge, environment | one canonical workload owner each |
| `tllm/kit`, `tllm/ports` | delete in E | none after typed execution cutover | canonical contracts |
| Rust non-canonical graph/schedule | prune in E/F | WorkGraph removed; remaining schedule/recurrence are pending audit | `rust/tllm_core/src/canonical` |
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
