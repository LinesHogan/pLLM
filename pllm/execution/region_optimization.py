"""Cold candidate measurement, native AER selection, and local rewrite replay.

This tuner handles exported regions, audited pure kernels and readonly tensor borrows, with
static tensor shapes and bounded loop/bank size parameters. It measures supplied
examples on private frame inputs and the original owner's readonly versions.
Payload limits cover the modeled tensors only;
backend workspace, allocator overhead and original owners need separate budgets.
"""

from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from itertools import combinations
import platform
from time import perf_counter_ns
from typing import Literal, Protocol

import torch

from pllm import _core
from pllm.contracts.region_program import BorrowedResource, Call, If, Map, Program, Region, Scan, TensorType, While, bind_sizes
from pllm.contracts.region_outlining import CallOutlineEvidence, call_outline_sites
from pllm.contracts.region_imports import BorrowedTensorImport, CompiledTensorImport, ExportedTensorImport, FunctionalKernelImport, NativeKernelImport, RegionTensorImport
from pllm.contracts.region_streams import RegionStreamPlan
from pllm.contracts.tensor_access import TensorAccessGraph
from pllm.contracts.region_selection import (
    RegionCandidate, RegionCandidateAssessment, RegionPayloadLimit, RegionSelection, RegionSelectionRequest,
    RegionSelectionPolicy,
    RegionTuningSize,
)
from pllm.contracts.tensor_rewrite import TensorRewrite
from pllm.contracts.tensor_compilation import TensorCompiler, UnsupportedTensorKernel
from pllm.execution.codec import boundary_digest, decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell, warn_numerical_transforms
from pllm.execution.region_compilation import analyze_region_program, explore_region_schedules, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionInstructionExecution, RegionExecutionResult
from pllm.execution.region_native import (
    BorrowedTensorPrimitive, CompiledPrimitive, RegionFunction, TensorSource,
    is_original_region_source, is_original_tensor_source, original_region_sources,
)
from pllm.execution.region_outlining import outline_region_calls
from pllm.execution.region_storage import import_region_functions
from pllm.execution.region_values import Data
from pllm.execution.region_resources import RegionResources
from pllm.execution.region_streams import RegionStreamBindings, explore_region_streams
from pllm.execution.tensor_rewrites import explore_tensor_combinations, replay_tensor_rewrites
from pllm.execution.tensor_access import import_tensor_access_graph
from pllm.execution.tensor_compilation import DEFAULT_TENSOR_COMPILER, compile_tensor_cell, supports_tensor_compilation


class RegionOptimizationOptions(Protocol):
    """The authoring options satisfy this interface without an upward import."""

    @property
    def allow_numerical_transforms(self) -> bool: ...

    @property
    def schedule_policy(self) -> Literal["auto", "serial"]: ...


def assess_region_candidates(program: Program, request: RegionSelectionRequest) -> tuple[RegionCandidateAssessment, ...]:
    return decode_boundary(_core.assess_region_candidates(encode_boundary(program), encode_boundary(request)),
                           tuple[RegionCandidateAssessment, ...])


def select_region_candidate(program: Program, request: RegionSelectionRequest) -> RegionSelection:
    return decode_boundary(_core.select_region_candidate(encode_boundary(program), encode_boundary(request)), RegionSelection)


def validate_region_selection(program: Program, request: RegionSelectionRequest, selection: RegionSelection) -> None:
    _core.validate_region_selection(encode_boundary(program), encode_boundary(request), encode_boundary(selection))


def _replay(sources: Mapping[str, TensorSource], records: tuple[TensorRewrite, ...]) -> dict[str, RegionFunction]:
    cells: dict[str, RegionFunction] = dict(sources)
    grouped: dict[str, list[TensorRewrite]] = {}
    for record in records:
        if record.source_digest not in sources:
            raise ValueError("region rewrite has no original source function")
        grouped.setdefault(record.source_digest, []).append(record)
    for name, chain in grouped.items():
        source = sources[name]
        if not isinstance(source, ExportedCell):
            raise ValueError("tensor rewrites cannot change original resource accessors or opaque kernels")
        cells[name] = replay_tensor_rewrites(source, tuple(chain))
    return cells


def _executor(program: Program, candidate: RegionCandidate, cells: Mapping[str, RegionFunction],
              payload_limits: tuple[RegionPayloadLimit, ...]) -> RegionInstructionExecution:
    return RegionInstructionExecution(program, lower_region_schedule(program, candidate.schedule_map),
        lower_region_bindings(program, candidate.schedule_map), cells, stream_plan=candidate.stream_plan,
        payload_limits=payload_limits)


def _replay_outlines(program: Program, sources: Mapping[str, TensorSource], records: tuple[CallOutlineEvidence, ...]
                    ) -> tuple[Program, dict[str, TensorSource]]:
    current, cells = program, dict(sources)
    for record in records:
        result = outline_region_calls(current, cells, record.site)
        if import_tensor_access_graph(result.cell) != record.access:
            raise ValueError("outlined tensor baseline differs from locally replayed original calls")
        current = result.outlined.program
        cells[result.cell.digest] = result.cell
    return current, cells


def _replay_compilation(cells: Mapping[str, RegionFunction], imports: tuple[RegionTensorImport, ...], compiler: TensorCompiler) -> dict[str, RegionFunction]:
    result = dict(cells)
    for imported in imports:
        if not isinstance(imported, CompiledTensorImport):
            continue
        name = imported.source_access.source_digest
        source = cells.get(name)
        if not isinstance(source, ExportedCell):
            raise ValueError("compiled candidate has no locally replayed source graph")
        compiled = compile_tensor_cell(source, compiler=compiler)
        if compiled.implementation != imported:
            raise ValueError("compiled candidate differs from locally replayed backend code or allocation facts")
        result[name] = compiled
    return result


@dataclass(frozen=True, slots=True)
class SelectedRegion:
    request: RegionSelectionRequest
    selection: RegionSelection
    rewrites: tuple[TensorRewrite, ...]
    _execution: RegionInstructionExecution

    @property
    def stream_plan(self) -> RegionStreamPlan | None:
        return self._execution.stream_plan

    def run(self, inputs: Mapping[str, Data], *, sizes: Mapping[str, int] = {}, pages: RegionResources | None = None,
            streams: RegionStreamBindings | None = None) -> RegionExecutionResult:
        if streams is not None:
            streams.validate_available()
            if any(d.device != str(streams.owner.device) for d in self.selection.candidates[self.selection.selected].peak_bytes):
                raise ValueError("selected execution belongs to another borrowed stream device")
            if pages is not None:
                pages.validate_stream(streams.owner)
        # Available auxiliary lanes may remain idle when the measured serial
        # candidate wins. The owner stream is the same for every alternative.
        with nullcontext() if streams is None else torch.cuda.stream(streams.owner):
            return self._execution.run(inputs, sizes=sizes, pages=pages,
                streams=streams if self._execution.stream_plan is not None else None)


def bind_selected_schedule(program: Program, cells: Mapping[str, RegionFunction], request: RegionSelectionRequest,
                           selection: RegionSelection, *, options: RegionOptimizationOptions = RegionSelectionPolicy()) -> SelectedRegion:
    """Bind unchanged native/exported imports to a verified measured schedule.

    Resource capabilities still arrive from the original invocation at run time.
    Payload limits cover the declared subtotal; native/library internals and
    device completion still require the original backend's reservations.
    """
    validate_region_selection(program, request, selection)
    candidate = request.candidates[selection.selected]
    imports = import_region_functions(program, cells)
    if imports != request.source_imports or candidate.imports != imports or candidate.outlines:
        raise ValueError("schedule-only binding requires unchanged original imports")
    if any(isinstance(cell, CompiledPrimitive) or isinstance(cell, ExportedCell) and cell.transformations for cell in cells.values()):
        raise ValueError("transformed cells require mathematical selection replay")
    if request.allow_numerical_transforms != options.allow_numerical_transforms:
        raise ValueError("region selection differs from the local numerical policy")
    if options.schedule_policy == "serial" and (candidate.stream_plan is not None or
        selection.candidates[selection.selected].capacity.schedule_map != selection.candidates[0].capacity.schedule_map):
        raise ValueError("region selection differs from the local serial schedule policy")
    return SelectedRegion(request, selection, (), _executor(program, candidate, cells, request.payload_limits))


def _check_tensor_sources(program: Program, sources: Mapping[str, TensorSource]) -> None:
    if any(not isinstance(resource, BorrowedResource) for resource in program.resources):
        raise ValueError("tensor tuning requires a pure program or original readonly tensor borrows")
    if any(not is_original_tensor_source(cell) for cell in sources.values()):
        raise ValueError("tensor tuning requires original graphs, readonly accessors or audited pure kernels")


def _candidate_sources(program: Program, sources: Mapping[str, TensorSource], request: RegionSelectionRequest,
                       index: int, assessment: RegionCandidateAssessment, records: tuple[TensorRewrite, ...],
                       options: RegionOptimizationOptions, compiler: TensorCompiler
                       ) -> tuple[Program, dict[str, RegionFunction]]:
    if any(not is_original_region_source(cell) for cell in sources.values()):
        raise ValueError("region binding requires original source implementations")
    if request.allow_numerical_transforms != options.allow_numerical_transforms:
        raise ValueError("region selection differs from the local numerical policy")
    if import_region_functions(program, sources) != request.source_imports:
        raise ValueError("region selection source imports differ from local graphs")
    candidate = request.candidates[index]
    derived, originals = _replay_outlines(program, sources, candidate.outlines)
    if options.schedule_policy == "serial":
        # Source analysis already provides the canonical reference. Routing
        # accessors are needed by execution, not by this policy comparison.
        reference = analyze_region_program(derived, optimize=False).reference_schedule
        if candidate.stream_plan is not None or assessment.capacity.schedule_map != reference:
            raise ValueError("region selection differs from the local serial schedule policy")
    cells = _replay_compilation(_replay(originals, records), candidate.imports, compiler)
    if import_region_functions(derived, cells) != candidate.imports:
        raise ValueError("selected region imports differ from locally replayed mathematical evidence")
    return derived, cells


def bind_region_candidate(program: Program, sources: Mapping[str, TensorSource], request: RegionSelectionRequest,
                          index: int, records: tuple[TensorRewrite, ...], *,
                          options: RegionOptimizationOptions = RegionSelectionPolicy(),
                          compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER) -> RegionInstructionExecution:
    """Prepare a funded trial from original sources without making a selection.

    No sample times are required. The caller supplies isolated invocation
    resources for each trial and measures completed execution. This binder
    verifies the same source replay, numerical policy and capacity as final
    installation, but neither executes effects nor emits a selection warning.
    """
    if type(index) is not int or not 0 <= index < len(request.candidates):
        raise ValueError("region trial has no such candidate")
    assessment = assess_region_candidates(program, request)[index]
    if not assessment.feasible:
        raise ValueError("region trial exceeds its payload or numerical policy")
    derived, cells = _candidate_sources(program, sources, request, index, assessment, records, options, compiler)
    return _executor(derived, request.candidates[index], cells, request.payload_limits)


def bind_selected_region(program: Program, sources: Mapping[str, TensorSource], request: RegionSelectionRequest,
                         selection: RegionSelection, records: tuple[TensorRewrite, ...], *,
                         options: RegionOptimizationOptions = RegionSelectionPolicy(),
                         compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER) -> SelectedRegion:
    """Rebuild selected tensor changes while retaining original effectful calls.

    External/debug entry with full re-verification; production finish paths
    reuse the measured executor instead. Only exported tensor graphs may be
    rewritten or compiled; native effects and accessors keep their original
    implementations. Binding never evaluates a source. The caller provides
    the original invocation's resources at run.
    """
    validate_region_selection(program, request, selection)
    candidate = request.candidates[selection.selected]
    derived, cells = _candidate_sources(program, sources, request, selection.selected,
        selection.candidates[selection.selected], records, options, compiler)
    execution = _executor(derived, candidate, cells, request.payload_limits)
    changed = selection.candidates[selection.selected].changed_functions
    transforms = tuple(t for name, cell in cells.items() if name in changed and isinstance(cell, (ExportedCell, CompiledPrimitive))
                       for t in cell.transformations)
    warn_numerical_transforms(selection.signature, transforms)
    return SelectedRegion(request, selection, records, execution)


def _synchronize(devices: tuple[str, ...]) -> None:
    for device in devices:
        if torch.device(device).type == "cuda":
            torch.cuda.current_stream(device).synchronize()


def _device_context(device: str) -> tuple[str, str, tuple[int, ...]]:
    if torch.device(device).type == "cuda":
        return device, torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device)
    return device, platform.processor() or platform.machine(), ()


def _measure(execution: RegionInstructionExecution, inputs: Mapping[str, Data], devices: tuple[str, ...],
             sizes: Mapping[str, int], streams: RegionStreamBindings | None = None,
             pages: RegionResources | None = None) -> int:
    scope = torch.cuda.stream(streams.owner) if streams is not None else pages.scope() if pages is not None else nullcontext()
    with scope:
        _synchronize(devices)
        start = perf_counter_ns()
        result = execution.run(inputs, sizes=sizes, streams=streams if execution.stream_plan is not None else None, pages=pages)
        _synchronize(devices)
        elapsed = perf_counter_ns() - start
        del result
        return max(1, elapsed)


@dataclass(frozen=True, slots=True)
class PreparedRegionCandidate:
    """Worker-local source derivation; contains no invocation or measurement."""

    program: Program
    candidate: RegionCandidate
    rewrites: tuple[TensorRewrite, ...]
    cells: Mapping[str, RegionFunction]


def _physical_candidates(family: Iterator[PreparedRegionCandidate], lanes: int | None) -> Iterator[PreparedRegionCandidate]:
    """Give each mathematics/backend combination serial and concurrent choices."""
    seen: set[str] = set()
    for proposal in family:
        yield proposal
        if lanes is None:
            continue
        key = boundary_digest(proposal.program, proposal.candidate.imports)
        if key in seen:
            continue
        seen.add(key)
        # This explores the shared schedule catalog once for this import set;
        # later serial schedules must not regenerate the same physical plans.
        catalog = explore_region_streams(proposal.program, proposal.cells, lanes=lanes, max_candidates=2, max_pipeline_depth=2)
        for plan in catalog.candidates:
            yield replace(proposal, candidate=replace(proposal.candidate, schedule_map=plan.phase_schedule, stream_plan=plan))


def _called_functions(region: Region) -> set[str]:
    names: set[str] = set()
    for op in region.block.operations:
        if isinstance(op, Call):
            names.add(op.function)
        children = ((op.body,) if isinstance(op, (Scan, Map)) else
                    (op.then_region, op.else_region) if isinstance(op, If) else
                    (op.condition, op.body) if isinstance(op, While) else ())
        for child in children:
            names.update(_called_functions(child))
    return names


def _compiled_variants(cells: Mapping[str, RegionFunction], called: set[str],
                       cache: dict[TensorAccessGraph, CompiledPrimitive], compiler: TensorCompiler) -> Iterator[dict[str, RegionFunction]]:
    """Explore individual lowerings and their joint use without greedy pruning.

    The enclosing candidate budget stops this iterator. Compilation is shared
    across schedules; a combination changes only function bindings, so its
    lifetimes, capacity and completed time are derived by the same region path.
    """
    compiled: dict[str, CompiledPrimitive] = {}
    for name in sorted(called):
        source = cells[name]
        if not isinstance(source, ExportedCell) or not supports_tensor_compilation(source, compiler=compiler):
            continue
        access = import_tensor_access_graph(source)
        implementation = cache.get(access)
        if implementation is None:
            try:
                implementation = compile_tensor_cell(source, compiler=compiler)
            except UnsupportedTensorKernel:
                continue
            cache[access] = implementation
        compiled[name] = implementation
        yield {**cells, name: implementation}
    if len(compiled) > 1:
        # Try the full combination before intermediate subsets can consume
        # the remaining budget. Single-function wins are not a prerequisite.
        for count in (len(compiled), *range(2, len(compiled))):
            for names in combinations(compiled, count):
                yield {**cells, **{name: compiled[name] for name in names}}


def _candidate_family(program: Program, sources: Mapping[str, TensorSource], outlines: tuple[CallOutlineEvidence, ...],
                      options: RegionOptimizationOptions, max_candidates: int, max_rewrite_depth: int,
                      compiled_cache: dict[TensorAccessGraph, CompiledPrimitive], compiler: TensorCompiler,
                      schedules: tuple[str, ...] | None = None) -> Iterator[PreparedRegionCandidate]:
    if schedules is None:
        catalog = explore_region_schedules(program,
            max_candidates=min(max_candidates, 8) if options.schedule_policy == "auto" else 1)
        schedules = tuple(candidate.schedule_map for candidate in catalog.candidates)
    variants: list[tuple[TensorRewrite, ...]] = [()]
    if options.allow_numerical_transforms:
        called = _called_functions(program.entry)
        variants.extend(explore_tensor_combinations(tuple(cell for name, cell in sources.items()
            if name in called and isinstance(cell, ExportedCell)),
            max_depth=max_rewrite_depth, max_candidates=max_candidates - 1))
    produced = 0
    # Share each family's budget between schedules and mathematics.
    for diagonal in range(len(schedules) + len(variants) - 1):
        for schedule_index in range(max(0, diagonal - len(variants) + 1), min(diagonal + 1, len(schedules))):
            records = variants[diagonal - schedule_index]
            cells = _replay(sources, records)
            yield PreparedRegionCandidate(program, RegionCandidate(schedules[schedule_index], import_region_functions(program, cells), (), outlines),
                             records, cells)
            produced += 1
            if produced == max_candidates:
                return
            if options.allow_numerical_transforms:
                for variant in _compiled_variants(cells, _called_functions(program.entry), compiled_cache, compiler):
                    yield PreparedRegionCandidate(program, RegionCandidate(schedules[schedule_index],
                        import_region_functions(program, variant), (), outlines), records, variant)
                    produced += 1
                    if produced == max_candidates:
                        return


def _outline_candidates(program: Program, sources: Mapping[str, TensorSource], limit: int
                        ) -> Iterator[tuple[Program, dict[str, TensorSource], tuple[CallOutlineEvidence, ...]]]:
    if limit == 0:
        return
    available = frozenset(name for name, cell in sources.items() if isinstance(cell, ExportedCell))
    for index, site in enumerate(call_outline_sites(program, functions=available)):
        if index == limit:
            break
        result = outline_region_calls(program, sources, site)
        originals = {**sources, result.cell.digest: result.cell}
        records = (CallOutlineEvidence(site, import_tensor_access_graph(result.cell)),)
        yield result.outlined.program, originals, records


def prepare_region_implementations(program: Program, cells: Mapping[str, RegionFunction], *,
                                   options: RegionOptimizationOptions = RegionSelectionPolicy(),
                                   max_variants: int = 4, max_rewrite_depth: int = 2,
                                   compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER) -> Iterator[PreparedRegionCandidate]:
    """Derive local implementations without searching schedules or running effects.

    The original reference is already in the caller's schedule catalog. Each
    returned candidate uses its own reference order; unchanged source structure
    can also reuse driver schedules after rebuilding their storage/stream facts.
    Outlined graphs require their own schedule. Trial isolation stays with the
    owning engine, and final installation replays this local source evidence.
    """
    if (type(max_variants) is not int or max_variants < 0
        or type(max_rewrite_depth) is not int or max_rewrite_depth < 0):
        raise ValueError("implementation preparation requires nonnegative search bounds")
    if not max_variants:
        return
    sources = original_region_sources(cells)
    cache: dict[TensorAccessGraph, CompiledPrimitive] = {}
    def implementation_family(source: Program, local: Mapping[str, TensorSource], records: tuple[CallOutlineEvidence, ...],
                              limit: int) -> Iterator[PreparedRegionCandidate]:
        # Do not analyze a derived full program until its family actually gets
        # a catalog slot. A bounded search can leave later families untouched.
        reference = analyze_region_program(source, optimize=False).reference_schedule
        yield from _candidate_family(source, local, records, options, limit,
            max_rewrite_depth, cache, compiler, (reference,))
    original = implementation_family(program, sources, (), max_variants + 1)
    next(original)  # The unchanged source is already the caller's candidate zero.
    families = deque((original,))
    families.extend(implementation_family(p, local, records, max_variants)
        for p, local, records in _outline_candidates(program, sources, min(max_variants, 8)))
    produced = 0
    while families and produced < max_variants:
        family = families.popleft()
        candidate = next(family, None)
        if candidate is not None:
            yield candidate
            produced += 1
            families.append(family)


def optimize_region(program: Program, sources: Mapping[str, TensorSource], inputs: Mapping[str, Data], *,
                    payload_limits: tuple[RegionPayloadLimit, ...], options: RegionOptimizationOptions = RegionSelectionPolicy(),
                    warmup_rounds: int = 1, measurement_rounds: int = 3, max_candidates: int = 32,
                    sizes: Mapping[str, int] = {}, max_rewrite_depth: int = 2,
                    streams: RegionStreamBindings | None = None, pages: RegionResources | None = None, compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER) -> SelectedRegion:
    """Compare bounded outlining/ISL/reassociation candidates on this workload.

    Selection uses median completed wall time, including Python issue and frame
    copies. Candidate rounds are interleaved. Numerical policy authorizes private
    evaluation; warnings are emitted only when the final chosen variant changes
    evaluation order. The sample sizes govern timing only; legality and payload
    bounds cover the full declared parameter range. This is local tuning, not
    a global optimality claim or a runtime dispatch by sample size. Rewrite
    depth limits total steps across all functions in each region candidate.
    """
    _check_tensor_sources(program, sources)
    sizes = bind_sizes(program, sizes)
    if pages is not None:
        pages.validate_readonly_tuning(program, sizes)
        pages.validate_tensor_borrows({name: source.implementation for name, source in sources.items()
            if isinstance(source, BorrowedTensorPrimitive)})
    elif program.resources:
        raise ValueError("tensor tuning requires the original readonly resource bindings")
    if (type(warmup_rounds) is not int or warmup_rounds < 0 or type(measurement_rounds) is not int
        or measurement_rounds < 1 or type(max_candidates) is not int or max_candidates < 1
        or type(max_rewrite_depth) is not int or max_rewrite_depth < 0):
        raise ValueError("region tuning needs nonnegative warmup/rewrite depth and positive measurement/candidate limits")
    source_imports = import_region_functions(program, sources)
    devices = tuple(sorted({storage.device for imported in source_imports if isinstance(imported, ExportedTensorImport)
                            for storage in imported.access.storages} |
                           {type_.device for imported in source_imports if isinstance(imported, BorrowedTensorImport)
                            for type_ in imported.declaration.outputs if isinstance(type_, TensorType)} |
                           {imported.device for imported in source_imports if isinstance(imported, (NativeKernelImport, FunctionalKernelImport))}))
    if streams is not None:
        streams.validate_available()
        if pages is not None:
            pages.validate_stream(streams.owner)
        if any(device != str(streams.owner.device) for device in devices):
            raise ValueError("tuning imports must use the borrowed stream device")
    lanes = len(streams.auxiliary) + 1 if streams is not None and options.schedule_policy == "auto" else None
    compiled_cache: dict[TensorAccessGraph, CompiledPrimitive] = {}
    families = deque((_physical_candidates(
        _candidate_family(program, sources, (), options, max_candidates, max_rewrite_depth, compiled_cache, compiler), lanes),))
    families.extend(_physical_candidates(_candidate_family(p, cells, records, options, max_candidates, max_rewrite_depth, compiled_cache, compiler), lanes)
                    for p, cells, records in _outline_candidates(program, sources, min(max_candidates - 1, 8)))
    proposals: list[PreparedRegionCandidate] = []
    # Round robin gives original and outlined programs a share of the catalog.
    # Within each family, schedules and mathematics are enumerated diagonally.
    while families and len(proposals) < max_candidates:
        family = families.popleft()
        proposal = next(family, None)
        if proposal is not None:
            proposals.append(proposal)
            families.append(family)
    candidates = [proposal.candidate for proposal in proposals]
    context = encode_boundary(("completed-wall-time-ns@2", torch.__version__, platform.platform(),
                               tuple(_device_context(d) for d in devices), torch.get_num_threads(),
                               warmup_rounds, measurement_rounds, 1 if lanes is None else lanes))
    request = RegionSelectionRequest(source_imports, tuple(candidates), payload_limits,
                                      options.allow_numerical_transforms, context,
                                      tuple(RegionTuningSize(name, value) for name, value in sorted(sizes.items())))
    assessments = assess_region_candidates(program, request)
    executors = {i: _executor(p.program, p.candidate, p.cells, request.payload_limits)
                 for i, (p, a) in enumerate(zip(proposals, assessments, strict=True)) if a.feasible}
    samples: dict[int, list[int]] = {i: [] for i in executors}
    for _ in range(warmup_rounds):
        for execution in executors.values():
            _measure(execution, inputs, devices, sizes, streams, pages)
    for _ in range(measurement_rounds):
        for index, execution in executors.items():
            samples[index].append(_measure(execution, inputs, devices, sizes, streams, pages))
    request = replace(request, candidates=tuple(replace(c, elapsed_ns=tuple(samples.get(i, ())))
                                                for i, c in enumerate(candidates)))
    selection = select_region_candidate(program, request)
    # The winner was measured, so its executor is already built from the
    # replayed sources: install it instead of replaying the selection bind.
    proposal = proposals[selection.selected]
    execution = executors[selection.selected]
    changed = selection.candidates[selection.selected].changed_functions
    transforms = tuple(t for name, cell in proposal.cells.items()
        if name in changed and isinstance(cell, (ExportedCell, CompiledPrimitive)) for t in cell.transformations)
    warn_numerical_transforms(selection.signature, transforms)
    return SelectedRegion(request, selection, proposal.rewrites, execution)
