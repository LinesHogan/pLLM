"""Prepare local region implementations, admit them, and bind once.

Static preparation returns an executor. Explicit tuning additionally returns
its complete measurement report. Neither path owns invocation state.
"""

from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from itertools import islice
from time import perf_counter_ns
from typing import Literal, Protocol

import torch

from pllm import _core
from pllm.contracts.region_program import BorrowedResource, Call, If, Map, Program, Region, Scan, TensorType, While, bind_sizes
from pllm.contracts.region_outlining import CallOutlineEvidence, call_outline_sites
from pllm.contracts.region_retention import RetainedAllocation
from pllm.contracts.region_proposal import RegionScheduleProposal
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
from pllm.execution.region_execution import RegionInstructionExecution
from pllm.execution.region_native import (
    CompiledPrimitive, RegionFunction, TensorSource,
    is_original_tensor_source, original_region_sources,
)
from pllm.execution.region_outlining import outline_region_calls
from pllm.execution.region_storage import import_region_functions, analyze_region_payload
from pllm.execution.region_values import Data
from pllm.execution.region_resources import RegionResources
from pllm.execution.region_streams import RegionStreamBindings, plan_region_schedule, arrange_region_schedule
from pllm.execution.tensor_rewrites import explore_tensor_combinations, replay_tensor_rewrites
from pllm.execution.tensor_access import import_tensor_access_graph
from pllm.execution.tensor_compilation import DEFAULT_TENSOR_COMPILER, compile_tensor_cell, supports_tensor_compilation


class RegionOptimizationOptions(Protocol):
    """The authoring options satisfy this interface without an upward import."""

    @property
    def allow_numerical_transforms(self) -> bool: ...

    @property
    def schedule_policy(self) -> Literal["auto", "serial"]: ...

    @property
    def tuning(self) -> Literal["heuristic", "measured"]: ...


@dataclass(frozen=True, slots=True)
class SelectedRegion:
    """Complete output of explicit tuning; execution itself has no tuning state."""
    execution: RegionInstructionExecution
    request: RegionSelectionRequest
    selection: RegionSelection
    rewrites: tuple[TensorRewrite, ...]


def assess_region_candidates(program: Program, request: RegionSelectionRequest) -> tuple[RegionCandidateAssessment, ...]:
    return decode_boundary(_core.assess_region_candidates(encode_boundary(program), encode_boundary(request)),
                           tuple[RegionCandidateAssessment, ...])


def select_region_candidate(program: Program, request: RegionSelectionRequest) -> RegionSelection:
    return decode_boundary(_core.select_region_candidate(encode_boundary(program), encode_boundary(request)), RegionSelection)


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


def _synchronize(devices: tuple[str, ...]) -> None:
    for device in devices:
        if torch.device(device).type == "cuda":
            torch.cuda.current_stream(device).synchronize()


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

    def arranged(self, proposal: RegionScheduleProposal) -> "PreparedRegionCandidate":
        return replace(self, candidate=replace(self.candidate, schedule_map=proposal.payload.capacity.schedule_map,
                                               stream_plan=proposal.stream_plan))


def bind_region_candidate(prepared: PreparedRegionCandidate, limits: tuple[RegionPayloadLimit, ...],
                          assessment: RegionCandidateAssessment) -> RegionInstructionExecution:
    """Bind a local, admitted compiler result; no callable is loaded from a wire proposal."""
    program, candidate = prepared.program, prepared.candidate
    if not assessment.feasible or any(not any(r.device == limit.device and r.bytes >= limit.bytes for r in limits)
                                      for limit in assessment.peak_bytes):
        raise ValueError("prepared execution exceeds its admitted payload or numerical policy")
    if import_region_functions(program, prepared.cells) != candidate.imports:
        raise ValueError("local prepared implementation differs from its admitted imports")
    return RegionInstructionExecution(program, lower_region_schedule(program, candidate.schedule_map),
        lower_region_bindings(program, candidate.schedule_map), prepared.cells,
        stream_plan=candidate.stream_plan, payload_limits=limits)


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


def _candidate_family(program: Program, sources: Mapping[str, TensorSource], outlines: tuple[CallOutlineEvidence, ...],
                      options: RegionOptimizationOptions, cache: dict[TensorAccessGraph, CompiledPrimitive],
                      compiler: TensorCompiler, rewrites: tuple[TensorRewrite, ...] = ()) -> Iterator[PreparedRegionCandidate]:
    """An eager graph and one complete compilation, without subset search."""
    reference = analyze_region_program(program, optimize=False).reference_schedule
    cells = _replay(sources, rewrites)
    yield PreparedRegionCandidate(program, RegionCandidate(reference, import_region_functions(program, cells), (), outlines), rewrites, cells)
    if not options.allow_numerical_transforms:
        return
    compiled = dict(cells)
    for name in sorted(_called_functions(program.entry)):
        source = cells[name]
        if not isinstance(source, ExportedCell) or not supports_tensor_compilation(source, compiler=compiler):
            continue
        access = import_tensor_access_graph(source)
        if access not in cache:
            try:
                cache[access] = compile_tensor_cell(source, compiler=compiler)
            except UnsupportedTensorKernel:
                continue
        compiled[name] = cache[access]
    if compiled != cells:
        yield PreparedRegionCandidate(program, RegionCandidate(reference, import_region_functions(program, compiled), (), outlines), rewrites, compiled)


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
    """Prepare a linear shortlist; source analysis and rewrite rules are unchanged."""
    if type(max_variants) is not int or max_variants < 0 or type(max_rewrite_depth) is not int or max_rewrite_depth < 0:
        raise ValueError("implementation preparation requires nonnegative search bounds")
    if not max_variants:
        return
    sources = original_region_sources(cells)
    cache: dict[TensorAccessGraph, CompiledPrimitive] = {}
    def variants() -> Iterator[PreparedRegionCandidate]:
        original = _candidate_family(program, sources, (), options, cache, compiler)
        next(original)  # The reference is already supplied by the caller.
        yield from original
        outlined = next(_outline_candidates(program, sources, 1), None)
        derived, local, records = (program, dict(sources), ()) if outlined is None else outlined
        if outlined is not None:
            yield from _candidate_family(derived, local, records, options, cache, compiler)
        if options.tuning == "measured" and options.allow_numerical_transforms:
            called = _called_functions(derived.entry)
            paths = explore_tensor_combinations(tuple(cell for name, cell in local.items()
                if name in called and isinstance(cell, ExportedCell)), max_depth=max_rewrite_depth, max_candidates=16)
            for rewrites in sorted(paths, key=len, reverse=True)[:2]:
                # Budget truncation drops the tail, so the full rewrite+compile
                # combination must precede its uncompiled reference: cutting
                # intermediates is recoverable, cutting the combination is not.
                family = tuple(_candidate_family(derived, local, records, options, cache, compiler, rewrites))
                yield from reversed(family)
    yield from islice(variants(), max_variants)


def prepare_region_candidates(program: Program, sources: Mapping[str, RegionFunction], *,
                              limits: tuple[RegionPayloadLimit, ...], options: RegionOptimizationOptions,
                              parallel: bool, sizes: Mapping[str, int], retained: tuple[RetainedAllocation, ...] = (),
                              max_candidates: int = 8, max_rewrite_depth: int = 2,
                              compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER, catalog: tuple[RegionScheduleProposal, ...] = ()) -> tuple[tuple[PreparedRegionCandidate, ...], RegionSelectionRequest, tuple[RegionCandidateAssessment, ...]]:
    """Shared cold preparation. Source analysis and rule implementations are unchanged."""
    if type(max_candidates) is not int or max_candidates < 1 or type(max_rewrite_depth) is not int or max_rewrite_depth < 0:
        raise ValueError("region preparation requires positive candidate and nonnegative rewrite limits")
    measured = options.tuning == "measured"
    imports = import_region_functions(program, sources)
    if catalog:
        proposals = [PreparedRegionCandidate(program, RegionCandidate(p.payload.capacity.schedule_map, imports, (),
            stream_plan=p.stream_plan), (), sources) for p in catalog]
    else:
        reference = analyze_region_program(program, optimize=False).reference_schedule
        reference = analyze_region_payload(program, reference, imports, retained).capacity.schedule_map
        proposals = [PreparedRegionCandidate(program, RegionCandidate(reference, imports, ()), (), sources)]
        default = plan_region_schedule(program, imports, payload_limits=limits, policy=options.schedule_policy, parallel=parallel, retained=retained)
        if default is not None and max_candidates > 1:
            proposed = proposals[0].arranged(default)
            if proposed.candidate != proposals[0].candidate:
                proposals.append(proposed)
        if measured and options.schedule_policy == "auto" and max_candidates > len(proposals):
            for schedule in explore_region_schedules(program, max_candidates=4).candidates:
                if len(proposals) == min(4, max_candidates):
                    break
                arranged = arrange_region_schedule(program, schedule.schedule_map, imports,
                    parallel=parallel, retained=retained, payload_limits=limits)
                candidate = RegionCandidate(arranged.payload.capacity.schedule_map, imports, (), stream_plan=arranged.stream_plan)
                if all(candidate != p.candidate for p in proposals):
                    proposals.append(PreparedRegionCandidate(program, candidate, (), sources))
    base = next((p.candidate for p in proposals if p.candidate.stream_plan is not None), proposals[min(1, len(proposals) - 1)].candidate)
    order = base.schedule_map if base.stream_plan is None else base.stream_plan.schedule_map
    for prepared in prepare_region_implementations(program, sources, options=options,
            max_variants=min(4 if measured else 3, max(0, min(max_candidates, 8) - len(proposals))),
            max_rewrite_depth=max_rewrite_depth, compiler=compiler):
        physical = (arrange_region_schedule(program, order, prepared.candidate.imports, parallel=parallel, retained=retained, payload_limits=limits)
            if prepared.program == program else plan_region_schedule(prepared.program, prepared.candidate.imports,
                policy=options.schedule_policy, parallel=parallel, retained=retained, payload_limits=limits))
        candidate = prepared if physical is None else prepared.arranged(physical)
        if all(candidate.candidate != p.candidate for p in proposals):
            proposals.append(candidate)
    request = RegionSelectionRequest(imports, tuple(p.candidate for p in proposals), limits,
        options.allow_numerical_transforms, "completed-wall-time-ns" if measured else "static preparation",
        tuple(RegionTuningSize(name, value) for name, value in sorted(sizes.items())), retained)
    assessments = assess_region_candidates(program, request)
    if not any(a.feasible for a in assessments):
        raise ValueError("no region candidate satisfies policy and tensor payload limits")
    return tuple(proposals), request, assessments


def warn_region_transforms(execution: RegionInstructionExecution, assessment: RegionCandidateAssessment, identity: str) -> None:
    transforms = tuple(t for name, cell in execution.cells.items()
        if name in assessment.changed_functions and isinstance(cell, (ExportedCell, CompiledPrimitive)) for t in cell.transformations)
    warn_numerical_transforms(identity, transforms)


def prepare_region_execution(program: Program, sources: Mapping[str, RegionFunction], *,
                             payload_limits: tuple[RegionPayloadLimit, ...], options: RegionOptimizationOptions,
                             parallel: bool = False, retained: tuple[RetainedAllocation, ...] = (),
                             compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER) -> RegionInstructionExecution:
    """Static boundary: no examples, measurements, or optional report fields."""
    if options.tuning != "heuristic":
        raise ValueError("static preparation requires heuristic policy; measured tuning needs explicit examples")
    proposals, _, assessments = prepare_region_candidates(program, sources, limits=payload_limits, options=options,
        parallel=parallel, retained=retained, max_rewrite_depth=0, compiler=compiler, sizes={p.name: p.minimum for p in program.parameters})
    index = max(i for i, a in enumerate(assessments) if a.feasible)
    chosen = proposals[index]
    execution = bind_region_candidate(chosen, payload_limits, assessments[index])
    warn_region_transforms(execution, assessments[index], boundary_digest(program, chosen.candidate))
    return execution


def optimize_region(program: Program, sources: Mapping[str, TensorSource], inputs: Mapping[str, Data], *,
                    payload_limits: tuple[RegionPayloadLimit, ...], options: RegionOptimizationOptions = RegionSelectionPolicy(tuning="measured"),
                    warmup_rounds: int = 1, measurement_rounds: int = 3, max_candidates: int = 8,
                    sizes: Mapping[str, int] = {}, max_rewrite_depth: int = 2,
                    streams: RegionStreamBindings | None = None, pages: RegionResources | None = None,
                    compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER) -> SelectedRegion:
    """Explicit tuning boundary: measure private examples, return a complete report."""
    if any(not isinstance(resource, BorrowedResource) for resource in program.resources):
        raise ValueError("tensor tuning requires a pure program or original readonly tensor borrows")
    if any(not is_original_tensor_source(cell) for cell in sources.values()):
        raise ValueError("tensor tuning requires original graphs, readonly accessors or audited pure kernels")
    sizes = bind_sizes(program, sizes)
    if options.tuning != "measured" or type(warmup_rounds) is not int or warmup_rounds < 0 or type(measurement_rounds) is not int or measurement_rounds < 1:
        raise ValueError("measured tuning requires its policy and positive measurement rounds")
    if pages is not None:
        pages.validate_readonly_tuning(program, sizes)
    elif program.resources:
        raise ValueError("tensor tuning requires the original readonly resource bindings")
    devices = tuple(sorted({t.device for f in program.functions for t in (*f.inputs, *f.outputs) if isinstance(t, TensorType)}))
    if streams is not None:
        streams.validate_available()
        if any(device != str(streams.owner.device) for device in devices):
            raise ValueError("tuning imports must use the borrowed stream device")
    proposals, request, assessments = prepare_region_candidates(program, sources, limits=payload_limits, options=options,
        parallel=streams is not None, sizes=sizes, max_candidates=max_candidates, max_rewrite_depth=max_rewrite_depth, compiler=compiler)
    executors = {i: bind_region_candidate(p, payload_limits, assessments[i])
        for i, p in enumerate(proposals) if assessments[i].feasible}
    samples: dict[int, list[int]] = {i: [] for i in executors}
    for round_ in range(warmup_rounds + measurement_rounds):
        for i, execution in executors.items():
            elapsed = _measure(execution, inputs, devices, sizes, streams, pages)
            if round_ >= warmup_rounds:
                samples[i].append(elapsed)
    request = replace(request, candidates=tuple(replace(c, elapsed_ns=tuple(samples.get(i, ())))
        for i, c in enumerate(request.candidates)))
    selection = select_region_candidate(program, request)
    execution = executors[selection.selected]
    warn_region_transforms(execution, assessments[selection.selected], selection.signature)
    return SelectedRegion(execution, request, selection, proposals[selection.selected].rewrites)
