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

Verified on 2026-09-06. See the [implementation handoff](architecture-consolidation-handoff.md)
for outstanding architecture, functionality and evidence work. The narrative
entries below are chronological history, not additional current checkpoints.

| Fact | Value | Cumulative change from `12551b8` | Change since `70a82d2` checkpoint |
| --- | ---: | ---: | ---: |
| Commit | `519d4ed` on `main` | — | after `70a82d2` |
| Counted production/support source | 21,789 lines | -111,421 lines | -3,550 lines |
| Pytest collection | 12 items | -1,337 items | +3 items |
| Rust tests | 4 items | -50 items | unchanged |
| Standalone correctness smokes | 4 items | baseline not fully enumerated | old non-survivor smokes deleted |
| Visible `TEST_COUNT` | 20 items | below the 50-item ceiling; not yet the final evidence set | +3 items |

The visible `TEST_COUNT` is not the terminal survivor suite.  It is the
currently executable survivor base after old-stack tests were deleted:
evidence IDs 1, 2, 3, 5, 9, 10, 12, 13, 15, 19, 20, 21, 22, 23, 25, 30, 32, 37, 38 and 39.
The agreed programmable-inference composition adds target IDs 40/41 to the
original 39, leaving 21 pending survivor evidence items. Those items are implementation
debt tied to their cutover tranche; they must become compact observable
oracles or public runtime smokes before the corresponding canonical boundary
is claimed complete, and must not be recreated as legacy wrapper,
class-hierarchy, registry or mock-call-count tests.

### Shared PageBundle replacement after `70a82d2`

Commits `e996998` and `519d4ed`: combined counted source `+1,011 / -4,561 / net -3,550`, including all new
arena/planning/math source. No complete files moved unchanged. AdamW formulas
were extracted from the CPU/CUDA providers into 118 lines operating on private
tensors; common Rust plan validation moved out of attachment into the shared
bundle and is not an independent deletion achievement.
The first commit was `+990 / -4,561 / net -3,571`; the alignment correction was
`+31 / -10 / net +21` within this replacement tranche. It pads field strides
and pool bases to the declared layout alignment, with all padding in signed
capacity; a real second-slot misalignment counterexample failed before the fix.

Deleted 13 old PTTS modules and the CPU/CUDA storage/coordinator wrappers.
One arena owns logical versions, generational field pools, multi-resource COW,
publication, pins, completion and reclaim. The request directory retains only
key-to-lineage selection and exclusive leases. ESamp and In-Place share the
original arena/pin/publication, without mirrored release lists or gradient tokens.
Attachment and arena reconstruction use the same native plan-bundle validator.

Evidence 9/10/12 are now active; 13 uses the same arena on CUDA. The heterogeneous
request/instance test covers partial initialization failure, private-write abort,
atomic publication and complete reclaim. An unsettled-publication retirement
counterexample first failed, then passed after receipts retained the associated
roots until settlement and device completion. This does not complete TLM (11),
captured version indirection (14), the generic backend, AER, MP or composition.
Provider capture stability remains false. Source below the original estimated
22k floor is not a reason to inflate code; missing capabilities still need work.

Repository-venv verification: full pytest 12 passed in 27.24s including the
isolated sdist/wheel gate; Rust 4 passed, strict/basic typing 0/0 and import/authority
gates passed. All four standalone GPU smokes passed: ESamp loss/update 2,
In-Place updates/trained rows 6 and apply rows 8, CUDA Graph loss/update 4 with
stable addresses, FTTT loss/update 1 and version 1. Cleanup passed and all
fallback counts were zero. No throughput benchmark, release or research-note edits.

### Shared attachment boundary after `e57caa6`

Commit `70a82d2`: counted source `+330 / -1,005 / net -675`; the 24-line
task-binding function moved unchanged. Independent tensor math did not change.
Attachment inputs now retain the original request, probe and plan; the plan
retains original tensor-layout facts. Removed copied probe/signature/key facts,
wire DTOs, the sampler attachment module, deployment field copies and compilation
digest copies. Python attachment guards were removed; Rust is the validator.
Request/decision/physical attachment schema is now 2; probe remains schema 1.

Evidence 15 first exposed acceptance of overlay field 999 outside the layout.
It now covers both actual workload compilations, signed layout identity, field
membership, task/hidden/stage binding, mechanical projection, reconstruction
drift and closed schema. Legal alternate field/module choices change signatures;
the test does not claim inference of algorithm-specific field roles. There are
17 active and 24 pending items. Generic backend sessions/factory, final PTTS,
AER, MP and programmable composition remain unfinished.

Repository-venv verification: full pytest 9 passed in 26.10s, Rust 4 passed,
strict/basic typing 0/0 and import/authority gates passed. All four standalone
GPU smokes ran: ESamp loss/update 2; In-Place updates/trained rows 6, apply rows 8;
CUDA Graph loss/update 4 with stable addresses; FTTT loss/update 1 and version 1.
Cleanup and zero-fallback checks passed. No benchmark, release or research-note edits.

### Shared In-Place execution after `9dfa7e7`

Commit `e57caa6`: counted source `+294 / -523 / net -229`; no file moves or
independent math-kernel changes. CPU/CUDA use one execution and handle type;
synchronous steps reuse the staged pin/submit/sampler/finish lifecycle.
The active invocation holds one context and its original publication receipt
through settlement. Mirrored executed/result states and the separate CUDA
execution were deleted without compatibility aliases. Fail-stop retains the
original invocation resources; recovery and post-publication settlement remain
pending. Physical providers/coordinators are existing objects, not copied ledgers.

Evidence 23 first failed on the old CPU execution's missing staged entry,
then passed with synchronous/staged updates matching the independent
multi-target oracle, pre-capture cancellation/retry, sampler completion gating,
consumed-handle rejection and prepublication rollback. Counts remain 16 active
and 25 pending. Generic execution across workloads, backend sessions and final
PTTS ownership remain incomplete; provisional dense_page was not refactored.

Repository-venv verification: pytest 8 passed in 24.23s, Rust 4 passed,
strict/basic typing 0/0, import/authority gates passed. Real In-Place MRv2
reported 6 updates/trained rows, 8 apply rows, zero live lineages/releases
and zero fallback. Public plain-to-ESamp also passed with 2 losses/updates,
zero live lineages and zero fallback. CUDA Graph/FTTT paths are unchanged and were not rerun;
their last records remain at `9dfa7e7`/`86e1326`. No benchmark or release.

### Shared ESamp execution after `4f7e370`

Commit `9dfa7e7`: counted source `+276 / -650 / net -374`; no file moves or
independent math-kernel changes. CPU/CUDA now use one execution class, handle,
active-invocation table and admission/anchor/cancel/retire/settlement path.
The old CUDA execution and handle names have no compatibility aliases.
Physical admission, pin, capture-pool and stream operations retain their
device dispatch. Gradient tensors stay in the shared context; a CUDA gradient
token exists only inside physical submission, removing device-dependent
context generics and the execution's separate gradient-token cleanup.

Evidence 25/37 cover cancellation after prediction and loss before and after
replacement. Evidence 37 also retires a request while a sampler read is in
flight: omitting the sampler-to-execution event wait fails the test; retaining
the actual completion dependency passes with zero live state/releases.
The test count remains 16, with 25 final evidence items pending. In-Place
lifecycle, cross-workload generic execution/backend sessions, PageBundle and
cross-root settlement remain unfinished; provisional PTTS was not polished.

Repository-venv validation: pytest 8 passed in 28.63s; Rust 4 passed;
strict/basic typing 0/0 and import/authority gates passed. CUDA Graph has
4 losses/updates and stable addresses, public ESamp has 2 losses/updates,
and public In-Place has 6 updates/trained rows and 8 apply rows. Cleanup and
zero-fallback checks passed. FTTT CUDA is unchanged and not rerun; its last
record remains at `86e1326`. No throughput benchmark or release.

### Shared ESamp task state after `86e1326`

Commit `4f7e370`: counted source `+152 / -279 / net -127`; no unchanged
moves or independent tensor math changes. Six CPU/CUDA tasks became three
shared tasks with one gradient-typed context and one logical transition path.
Device kernels only bind existing storage, gradients, pins and streams.
The CUDA invocation and context reference the same kernel object.
`ESampPublished` was deleted; context holds the original shared
`RequestStatePublicationReceipt`, not a workload-specific receipt projection.

Evidence 25 now checks cancellation after prediction and injected failure
after AdamW but before publication: original roots and capacity are preserved,
and retry still matches the independent oracle. Evidence 37 exercises CUDA
cancellation before each successful capture/replay step. These additions
passed before and after replacement; active/pending counts remain 16/25.
Execution lifecycles and backend sessions remain separate. This change does
not complete PageBundle, generational identity or cross-root settlement;
provisional dense_page internals were left unchanged.

Validation using the repository venv: full pytest 8 passed in 27.33s;
CUDA Graph loss/update 4, stable addresses and zero fallback; strict/basic
typing 0/0 and import/authority gates passed. Public plain-to-ESamp MRv2 smoke
also passed with loss/update 2, zero live lineages and zero fallback.
Rust and In-Place/FTTT paths
are unchanged and were not rerun; their last evidence remains at `86e1326`.
No throughput benchmark or release.

### Shared Graph Index and Codec after `1a9eca6`

Commit `86e1326`: counted source `+268 / -365 / net -97`. Rust inline evidence
accounts for 155 net added lines; nine timing-lookup lines moved unchanged
within the Rust scheduler. No independent tensor math changed or moved.
The old explicit-result-edge requirement and four workloads' duplicate edge
declarations are gone. One derived index supplies dependencies, windows and
topological order, without copying semantic tasks/state/effects into another
graph. Conflicting effects require a causal path or disjoint windows; task
names no longer silently decide an otherwise ambiguous state order.

Python FFI now mechanically serializes the shared immutable descriptors with
the standard dataclass/JSON APIs, adding only closed input-variant tags.
The parallel wire DTO family and manual per-field encoding were deleted;
Rust remains the semantic validator and Python retains closed FFI decoding.

Evidence 3 now enumerates all 64 directed edge subsets and 216 timing-window
assignments on three tasks against an independent placement/order oracle,
including cycles, implicit dataflow, duplicate dependencies and effect-order
counterexamples. It failed before implementation and passes afterward. The
visible evidence count remains 16; AER partition/placement, exact-version
hazards and distributed placement remain missing, not implied by this test.

Validation from main after rebuilding the native extension: repository-venv
pytest 8 passed in 26.94s, Rust 4 passed, strict/basic typing 0/0, import/authority
gates passed. All four GPU smokes rerun: ESamp loss/update 2, In-Place updates 6,
CUDA Graph loss/update 4 with stable addresses, FTTT loss/update 1/version 1.
Cleanup and zero-fallback checks passed. No throughput benchmark or release.

### Shared In-Place task state after `7d53caf`

Commit `bea197d`: counted source `+201 / -523 / net -322`, including six
implementation-ID declaration lines moved unchanged. Independent tensor math
was neither moved nor changed. CPU/CUDA now use the same task implementations,
invocation context, transition/result states and publication receipt type.
The 292-line CUDA task module and its mirrored semantic states were deleted.
Device bindings retain only provider/pin/stream access and actual storage I/O;
they do not own another version map. The CUDA submission boundary remains
fail-stop, and context retains the original issued receipt, not a conversion.
Provisional dense_page internals were not refactored. Execution lifecycles,
ESamp task merging and final PageBundle ownership remain unfinished.

Evidence 23 now injects prepublication failure and proves unchanged roots and
clean retry against the independent multi-target oracle; it passed both before
and after replacement. New evidence 25 independently checks ESamp seeded
initialization, gated residual/layernorm, AdamW gradients/moments/steps,
per-state loss averaging, sample isolation, immutable prediction guidance,
sampler-before-training ordering and cleanup. This is one added evidence item,
not a device/workload matrix.

Validation: repository-venv pytest 8 passed in 22.81s, Rust 4 passed, typing
strict/basic 0/0, import/authority gates passed. Real In-Place MRv2 smoke passed:
6 updates, 6 trained rows, 8 apply rows, 2 updated rows, zero live lineages,
pending releases and fallback. Other standalone GPU smoke records remain at
`7d53caf`; no throughput benchmark was run.

An ignored root `tllm.egg-info` still advertised the retired general plugin.
It was moved intact to `/tmp/tllm-stale-metadata.sUqZvf/tllm.egg-info`;
metadata discovery and a second real GPU run confirmed the stale plugin error
is gone. This generated-artifact cleanup receives no source deletion credit.

### Shared host boundary and Rust validation after `19c59d5`

Counted source: `+457 / -1,255 / net -798`. This includes 100 unchanged lines
of GTTA tasks moved next to their state owner; that move contributes zero net
deletion. Independent tensor math is unchanged. Rust inline evidence adds 51
counted lines, included in the source budget rather than hidden as test cleanup.

Both workload execution modules, private receipt/completion state machines and
compilation/options wrappers are gone. FTTT and GTTA compile directly to the
same `HostExecution`: the signed plan, rollback snapshot, issued receipt and
CUDA readiness/completion have one owner. Publication failure restores state;
post-publication settlement failure carries the exact receipt, never a copied
message or retry of the host effect. Evidence 19/21 proves these contracts and
matches independent PyTorch Adam/edit math through abort and retry settlement.

Seventeen Python graph/plan validators (213 lines) were deleted. The shared
immutable descriptors remain data; Rust validates graph semantics, and the FFI
codec retains closed type parsing. Rust evidence 2 covers schema drift and
semantic digest changes. No separate AER/replay/MP graph family was introduced.

Validation: repository-venv pytest 7 passed; Rust 4 passed; strict/basic typing
0/0, import/authority gates passed. All four existing GPU smokes rerun with
nonzero training and zero fallback. No throughput benchmark or release.
Remaining B/F work is substantive: unified ESamp/In-Place transitions,
backend attachment, PageBundle/AER, closed MP and conditional control remain
unfinished. Snapshot branches preserve original dirty main and LoRA changes;
the research draft is neither modified nor committed.

Root integration: `main` fast-forwarded to `7d53caf`; all auxiliary worktrees
were removed without force after preserving original edits in
`archive/consolidation-main-20260906` (`865ad4c`) and
`archive/consolidation-lora-20260906` (`c5b3589`). The LoRA branch also retains
the same snapshot tree at `781034e`; its semantics are not yet integrated.
The native extension was rebuilt from the main directory and its import origin
checked. Main-directory pytest: 7 passed in 28.25s using the repository venv.
No source changes or deletion credit are attributed to this checkout cleanup.

### Release-preparation slice after `4cd162f`

The previous residual-runtime retirement is committed in `4cd162f`:
`+111 / -21,856 / net -21,745`, with 617 unchanged moved lines separately
accounted for. This slice is primarily G documentation/distribution cleanup,
not completion of the B generic executor migration. Its counted-source change
is `+10 / -22 / net -12`: three workload-local synchronous anchor loops now
use `TaskGraphExecution.run`, preserving the signed order. No source was moved
and no independent mathematics was extracted or changed.

Seventeen obsolete documents were removed; the maintained entry guides were
rewritten around real supported boundaries. The redundant Rust Python-package
configuration and minimal requirements file were removed. The root project
metadata now owns tensor/runtime/test dependencies and the existing Apache-2.0
license is included in distributions. None of these non-source deletions offsets
the source budget. Evidence 39 now builds a wheel from a fresh extracted sdist,
then checks isolated import/codec execution, dependency and license metadata.

Evidence 22 adds the GTTA FIFO/EMA rollback oracle; evidence 23 adds the
multi-target In-Place chunk-update/publication/isolation/retirement oracle.
Full pytest: 5 passed in the repository venv. Import and authority gates passed;
strict/basic typing: 0 errors, 0 warnings. FTTT CUDA: loss/update 1, version 1,
zero fallback. Rust and the other three GPU smokes retain the unchanged
`4cd162f` results; no throughput benchmark was run.

Worktrees decreased from 30 to 3 using non-force removal of 27 clean historical
checkouts. Branches and commits remain recoverable. Dirty main (five modified
files and the protected research note) and shared-test-time-lora-runtime
(31 changes) remain untouched. Their integration and the outstanding functional
contracts are release blockers; this checkpoint is neither tagged nor published.

The FTTT authority-retirement slice added no counted source. It deleted 909
production lines from `tllm/consumers/fttt` and three counted support lines
from static gates, for `+0 / -912 / net -912` under `SOURCE_LOC`. It also
deleted 1,026 test lines from the superseded unit and GPU smoke. At the current
checkpoint evidence 21 is pending and evidence 38 retains its CUDA smoke base. Documentation and JSON
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

The native workload/schedule island deletion in this checkpoint removed the
old `_core.compile_schedule` Rust scheduler, `tllm.runtime.runtime`,
`tllm.schedule`, `tllm.workload`, native workload execution/plugins/bindings/
sinks, and the MP worker/control-plane workload-decision RPC surface.  The
remaining worker/control plane exposes only residual vLLM liveness, request
mapping, stats, synchronization and enable flags; import-layer baseline is now
0/0 and the region-authority gate no longer lists deleted old authorities.
Under `SOURCE_LOC`, this checkpoint is `+125 / -27,036 / net -26,911`;
`SOURCE_LOC` is now 55,232.

The recurrence island deletion removed the orphan finite-recurrence contracts,
typed recurrence executor, Rust `recurrence.rs` FFI/classes, `_core.pyi`
typed-recurrence stubs, and MRv1 backend-region/packed-hidden capture hooks
whose old signed Region consumer had been deleted.  Recirculation survivor
semantics must be rebuilt as the compact canonical typed replay effect named
by evidence 27/35, not by restoring this recurrence API.  Under `SOURCE_LOC`,
this checkpoint is `+2 / -3,296 / net -3,294`; `SOURCE_LOC` is now 51,938.

The WorkGraph/backend-region cleanup removed the residual WorkGraph stats
executor, runtime stats DTO, backend-region binder, old packed-hidden capture,
old packed-phase row staging, and the remaining backend timeline/model-region
hooks tied to the retired native scheduler.  The MP stats surface no longer
publishes a `workgraph` category, and import-layer rules no longer contain a
deleted WorkGraph package tier.  Under `SOURCE_LOC`, this checkpoint is
`+7 / -2,223 / net -2,216`; `SOURCE_LOC` is now 49,718.

## Tranche accounting

The 2026-09-06 residual retirement slice removed the remaining old runtime,
ports, common, core re-export and util packages; orphan slot/paged-state and
physical-plan stores; old driver installers and sampler hooks; MP bootstrap,
plugin and control plane; and unused Rust shape/decode/sampler FFI. Plain
`make_llm` now constructs vLLM directly with typed engine options, while the
canonical route still attaches the supported workloads. No compatibility
entry point or no-op installer was left behind.

This coordinated D/E slice is `+111 / -21,856 / net -21,745` counted source.
Three independent backend helpers moved byte-for-byte: graph mode (79),
overlay state (486), and specialization facts (52), totaling 617 lines.
These moves are not deletion. The retained overlay version store and replay
callback contracts still need consolidation with canonical PTTS/effects;
deleting the runtime directory does not certify a single final state authority.
Retained structural probes do not constitute verified MRv1/legacy public support.

Evidence 39 now builds and installs a fresh non-editable wheel in an isolated
directory, verifies package/native-module provenance and removal of the old
plugin/packages, and executes the Rust layout codec. Evidence 20 restores a
compact Dummy graph/plan/owned-export oracle from the historical numerical
scenario without its legacy consumer import. Neither adds production helpers.
Evidence 30 now exercises plain inference followed by canonical ESamp in the
same process, covering the public routing change without another test item.

Validation uses `/workspace/pllm/.venv/bin/python`: pytest 3 passed, Rust
tests 3 passed, import-layer 0/0, region authority OK (4 real execution/binding
files), strict and basic typing both 0 errors/0 warnings. The basic baseline
fell from 108 errors to 0 through deletion. The root-project editable wheel
was rebuilt; installed metadata has no tLLM general plugin. Real public GPU
smokes passed: plain -> ESamp (loss/update 2, active lineages 0, fallback 0),
In-Place-TTT (updates 6, dense apply rows 8, active lineages 0, fallback 0).
The remaining standalone CUDA evidence also passed: graph replay reported
4 losses and 4 updates with stable addresses and zero fallback; FTTT reported
1 loss, 1 update, state version 1 and zero fallback. All 10 currently retained
evidence items executed successfully; this does not settle the 29 pending items.
No throughput benchmark was run. SOURCE_LOC now meets the numeric window,
but missing workloads, MRv1, AER/PTTS, closed MP and evidence remain open.

| Tranche | Added | Deleted | Moved unchanged | Net | State |
| --- | ---: | ---: | ---: | ---: | --- |
| A — ledger and survivor extraction | 7 | 2,997 | 0 | -2,990 | active: stale consumer/duplicate/core unit tests pruned; pytest 1, Rust 3, standalone smokes 4 |
| B — canonical vertical compression | 4,418 | 10,430 | 0 | -6,012 | active: canonical runtime attachments, ESamp task merge, duplicated Python validation pruning, boundary gate repair and re-export cleanup |
| C — workload cutovers | 0 | 912 | 0 | -912 | active: FTTT retired |
| D — backend quarantine and closed MP | 286 | 787 | 0 | -501 | active: backend event seam, vLLM WorkGraph cold handoff and retired legacy hook payload removed; seam rewrite still pending |
| E — legacy authority removal | 529 | 73,602 | 988 | -73,073 | active: Program, old WorkGraph, consumers/kit, producer/placement planners, legacy workload facades, native workload/schedule island, old Rust scheduler/recurrence island, backend-region/packed-hidden old hooks, consumer compat island, BaseConsumer dispatch chain and dummy consumer re-export removed |
| D/E — residual retirement continuation | 111 | 21,856 | 617 | -21,745 | old packages and public hook/plugin path deleted; retained backend state/effects still require canonical consolidation |
| F — AER/PTTS final core | 1,011 | 4,561 | 0 complete files | -3,550 | active: shared PageBundle replacement; AER, MP, TLM and captured version selection remain pending |
| G — reachability and final compression | — | — | — | — | pending |

Update this table from tracked physical line counts at each tranche boundary.
Moves are recorded separately and never counted as deletion.

## Production keep/move/delete authority

| Current owner | Disposition | Unique content that survives | Final owner |
| --- | --- | --- | --- |
| `tllm/canonical`, `tllm/contracts` | keep and compress | semantic descriptors, tensor math, FFI DTOs | canonical graph/execution core |
| `tllm/canonical_runtime` | directory deleted; B remains incomplete | workload-specific branching still exists in canonical_vllm/runtime.py | generic executor/backend seam |
| `tllm/canonical_vllm` | merge in D | vLLM probes and physical adapter mechanics | `tllm/backends/vllm` |
| `tllm/runtime` | deleted in current worktree | 617 lines of backend helper content moved; closed MP remains missing | backend primitives, then canonical state/effect consolidation |
| `tllm/workgraph` | deleted in `3da233b`; temporary recurrence owner subsequently deleted in `07a64d8` | row-selection/replay semantics still to implement | canonical typed replay effect |
| `tllm/program` | deleted in `3da233b` | none | canonical descriptors/workloads |
| `tllm/schedule`, `tllm/workload` | deleted in `19e0389` | missing workload semantics remain implementation debt | canonical descriptors/workloads |
| `tllm/consumers` | deleted in `b3af0d4` | survivor semantics to be rebuilt only in canonical owners | one canonical workload owner each |
| `tllm/kit` | deleted in `b3af0d4` | none after typed execution cutover | canonical public factories |
| `tllm/ports`, `tllm/common`, `tllm/core`, `tllm/util` | deleted in current worktree | plain engine construction moved into public factory; obsolete helpers removed | canonical contracts and direct public factory |
| Rust non-canonical graph/schedule | old graph/schedule/recurrence/placement/execution_plan and shape/decode/sampler removed | prefill row planning and batch-phase classification still consumed by adapters | unique backend fact helpers |
| provisional `canonical/ptts/dense_page*` | deleted in `e996998` | one CPU/CUDA lifecycle and independent AdamW math | shared PageBundleArena; TLM/captured indirection still pending |

## Workload cutover manifest

| Workload | Unique content to retain | Old owner removal gate | Evidence IDs |
| --- | --- | --- | --- |
| Dummy | minimum graph/sign/plan/execute declaration | workload compile/owned export oracle restored; generic public factory pending | 20 active |
| FTTT | edit math, judge, feedback retry | legacy owner removed; shared host transaction and receipt | 21, 38 active |
| GTTA | environment state/update/query/FIFO | generic host transaction | 22 |
| In-Place-TTT | dense fast-weight math and layout | shared device execution/task/context complete; generic backend sessions/PTTS pending | 23, 32 active |
| TLM | loader, objective, q/v LoRA math | global PTTS and online/offline timing | 24, 33 |
| ESamp | request math, sampler guidance, random init | shared device execution/task/context complete; generic backend sessions and MRv1 pending | 25, 30, 37 active; 31 pending |
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
