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

| Fact | Value | Cumulative change from `12551b8` | Change since `e03816d` checkpoint |
| --- | ---: | ---: | ---: |
| Commit | `ed1c487` | — | after `e03816d` |
| Counted production/support source | 82,143 lines | -51,067 lines | -9,183 lines |
| Pytest collection | 1 item | -1,348 items | -10 items |
| Rust tests | 3 items | -51 items | -51 items |
| Standalone correctness smokes | 4 items | baseline not fully enumerated | old non-survivor smokes deleted |
| Visible `TEST_COUNT` | 8 items | now below the 50-item ceiling | -10 items |

The visible `TEST_COUNT` is not the terminal survivor suite.  It is the
currently executable survivor base after old-stack tests were deleted:
evidence IDs 1, 3, 5, 13, 30, 32, 37 and 38.  The manifest now records
31 pending survivor evidence items.  Those pending items are implementation
debt tied to their cutover tranche; they must become compact observable
oracles or public runtime smokes before the corresponding canonical boundary
is claimed complete, and must not be recreated as legacy wrapper,
class-hierarchy, registry or mock-call-count tests.

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

The stale consumer-test prune in `7b01440` deleted the remaining behavioral
tests under `test/consumers/`.  Those files still asserted legacy consumer
package shape, mock-patched the residual runtime, or duplicated native ESamp
request-state coverage already represented by the canonical ESamp evidence.
This reduced pytest collection from 25 to 11 items and removed 1,238 tracked
test/config lines.  ReFT, OverRIDE and TTRL are intentionally no longer
protected by old consumer-path tests; when their production owners are cut over,
their survivor evidence must be added as compact canonical workload oracles,
not restored as `test/consumers` compatibility tests.

The consumer-compat and recurrence-shim prune in `0fde70c` removed ESamp's
old `BaseConsumer`/Program runtime island, deleted the consumer
base/plugin/registry re-export modules, removed the compatibility-only Rust
`compile_recurrence_schedule` FFI and `CompiledRecurrenceSchedule` stub, and
closed the `tllm.schedule` top-level coordinator/compiler re-export door.
Runtime schedule bridges now point at the actual owner modules and the
import-layer baseline records those exact bridges instead of allowing the
root facade.  Under `SOURCE_LOC`, this checkpoint is
`+63 / -2,849 / net -2,786`; `SOURCE_LOC` is now 93,803.  Probe imports for
`tllm.consumers.base`, `tllm.consumers.registry`,
`tllm.consumers.plugin`, `tllm.consumers.esamp.config`,
`tllm.consumers.esamp.engine`,
`tllm.consumers.esamp.sampler_provider` and
`tllm.consumers.esamp.initializers` all return `None`.

The legacy consumer dispatch deletion in `e03816d` removed the remaining
`BaseConsumer` object authority, `DispatchPlan`, `consumer_compat`,
consumer plugin/spec reconstruction, consumer registry, fusion executor,
HiddenBatch event bridge, residual-bundle flow dispatch and post-generate
consumer dispatch.  `TllmRuntime` no longer accepts `consumers=` and the
top-level `tllm.register_consumer`/`clear_consumers` façade is gone.  MP
bootstrap now carries runtime config only; workers no longer reconstruct
consumer specs.  Production/tools grep for `BaseConsumer`, `DispatchPlan`,
`consumer_compat`, `consumer_registry`, `consumer_plugins`,
`declaration_adapter`, `register_dispatch_consumer` and
`set_runtime_consumer` is empty.  Probe imports for the removed runtime
modules return `None`.  Under `SOURCE_LOC`, this checkpoint is
`+80 / -2,557 / net -2,477`; `SOURCE_LOC` is now 91,326.

The duplicate-test and retired event-hook cleanup in `e47fbaf` deleted the
remaining duplicate Python attachment/state validator tests
`test_parameter_overlay_attachment_unit.py` and
`test_state_representation_plan_unit.py`, removed the old workload public
typing fixture plus its ConfigKind negative pseudo-test, and dropped the
tracked prompt debug list.  Pytest collection is now 9 items.  The same commit
removed the no-op legacy load-model event dispatch and collapsed
`runtime/vllm_patch/common_hooks.py` to a single native-event advancement
helper; sampler pre-draw no longer constructs an unused predecessor-event
fence for a retired consumer receiver.  Production/tools grep for the deleted
test names, old typing fixture, `dispatch_legacy_runtime_event` and
`build_runtime_context` is empty.  Under `SOURCE_LOC`, this checkpoint is
`+11 / -300 / net -289`; `SOURCE_LOC` is now 91,037.

The coupled core-unit evidence prune in `18e1686` deleted the remaining
`test/core` pytest files and the `tllm.consumers.dummy` compatibility
re-export.  The deleted Python unit tests were tied to provisional DTO shapes,
fixture construction and internal workload class paths; their successor
evidence must be rebuilt only as compact public-path survivor checks named in
section 5.3, not restored as per-module unit coverage.  Pytest collection is
now the single CUDA dense-page survivor base; the four standalone GPU smokes
remain outside pytest.  Grep for `tllm.consumers.dummy`, `test/core`,
`test_canonical`, `BaseConsumer`, `DispatchPlan` and
`dispatch_legacy_runtime_event` is empty across production, tests, tools and
examples.  Under `SOURCE_LOC`, this checkpoint is
`+1 / -12 / net -11`; `SOURCE_LOC` is now 91,026.

The producer and placement planner deletion in `8a790ba` removed the retired
producer package, runtime producer executor, Python execution/placement planner
wrappers, the Rust `execution_plan` and `placement` FFI modules, and their
remaining `_core.pyi`/`tllm.core` exports.  The vLLM hooks now prepare only
native decode metadata; the old execute-model producer dispatch hook and its
empty replacement were deleted rather than left as a no-op compatibility seam.
The import-layer rules no longer contain a producer layer.  Grep for
`tllm.producers`, `ProducerRegistry`, `producer_executor`,
`runtime.placement`, `runtime.execution_plan`, `compile_execution_plan` and
`compile_placement_schedule` is empty across production, tests, tools,
examples and typing configs.  Under `SOURCE_LOC`, this checkpoint is
`+19 / -2,559 / net -2,540`; `SOURCE_LOC` is now 88,486.

The consumer and kit deletion in `b3af0d4` removed the tracked `tllm/consumers`
and `tllm/kit` packages, including the old ESamp/TLM/ReFT/Recirculation/
OverRIDE/TTRL workload class hierarchy, compatibility export catalogs, old
ReFT vLLM patch, shared alignment helpers and authoring-kit wrappers.  The
remaining public/runtime tree no longer imports `tllm.consumers` or `tllm.kit`;
survivor workload semantics must be rebuilt as compact canonical workload
owners and public-path evidence, not restored under the deleted packages.
Under `SOURCE_LOC`, this checkpoint is `+0 / -5,732 / net -5,732`;
`SOURCE_LOC` is now 82,754.

The gate cleanup in `76a0cf9` removed the deleted consumers/kit layer rules,
strict typing includes, old consumer warning allowances and the recirculation
consumer-specific region-authority visitor.  Static gates now fail on new
warnings without preserving deleted consumer baselines, and the region gate
tracks 16 live production files instead of missing retired files.  Basic typing
baseline is rebased to 108 errors.  Under `SOURCE_LOC`, this checkpoint is
`+7 / -170 / net -163`; `SOURCE_LOC` is now 82,591.

The public workload facade retirement in `ed1c487` removed the lazy top-level
`TllmRuntime`/`default_runtime` export, closed the legacy
`make_llm(workloads=...)` branch to non-canonical workloads, deleted the
`tllm.workload` and `tllm.schedule` root re-export catalogs, and removed
vLLM/MP fail-stop hooks whose only consumer was the retired native workload
coordinator.  Remaining old workload/schedule references are direct owner
module imports from the still-live legacy runtime execution island, not root
facades.  Under `SOURCE_LOC`, this checkpoint is `+31 / -479 / net -448`;
`SOURCE_LOC` is now 82,143.

## Tranche accounting

| Tranche | Added | Deleted | Moved unchanged | Net | State |
| --- | ---: | ---: | ---: | ---: | --- |
| A — ledger and survivor extraction | 7 | 2,997 | 0 | -2,990 | active: stale consumer/duplicate/core unit tests pruned; pytest 1, Rust 3, standalone smokes 4 |
| B — canonical vertical compression | 4,418 | 10,430 | 0 | -6,012 | active: canonical runtime attachments, ESamp task merge, duplicated Python validation pruning, boundary gate repair and re-export cleanup |
| C — workload cutovers | 0 | 912 | 0 | -912 | active: FTTT retired |
| D — backend quarantine and closed MP | 286 | 787 | 0 | -501 | active: backend event seam, vLLM WorkGraph cold handoff and retired legacy hook payload removed; seam rewrite still pending |
| E — legacy authority removal | 395 | 41,047 | 988 | -40,652 | active: Program, old WorkGraph, consumers/kit, producer/placement planners, legacy workload facades, consumer compat island, BaseConsumer dispatch chain, dummy consumer re-export and recurrence FFI shim removed |
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
| `tllm/consumers` | deleted in `b3af0d4` | survivor semantics to be rebuilt only in canonical owners | one canonical workload owner each |
| `tllm/kit` | deleted in `b3af0d4` | none after typed execution cutover | canonical public factories |
| `tllm/ports` | delete in E/F | none after typed execution cutover | canonical contracts |
| Rust non-canonical graph/schedule | prune in E/F | WorkGraph, execution_plan and placement removed; remaining schedule/recurrence are pending audit | `rust/tllm_core/src/canonical` |
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
