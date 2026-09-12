"""Cold native trials using the original execution, request owners and trace."""

from contextlib import ExitStack
from dataclasses import replace
from typing import Generic, TypeVar

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_selection import RegionCandidate, RegionSelection, RegionSelectionRequest, RegionTuningSize
from pllm.contracts.semantic_graph import ImplementationId
from pllm.contracts.tensor_rewrite import TensorRewrite
from pllm.execution.region_optimization import prepare_region_implementations, select_region_candidate
from pllm.execution.codec import boundary_digest, preserve_torch_rng
from pllm.execution.region_storage import analyze_region_payload
from pllm.execution.region_streams import lower_region_streams
from pllm.execution.replay import TokenwiseReplayTask
from pllm.execution.state_execution import StateExecution
from pllm.execution.tokenwise_region import TokenwiseRegion

InputT = TypeVar("InputT")
DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")


class TokenwiseTuning(Generic[InputT, DataT, ResultT]):
    def __init__(self, execution: StateExecution[InputT, DataT, ResultT], identity: ImplementationId,
                 proposals: tuple[RegionScheduleProposal, ...], *, stream_lanes: int = 1) -> None:
        self.execution = execution
        self.original = TokenwiseReplayTask[DataT].from_implementation(execution.implementation(identity))
        if self.original.policy.schedule_policy == "serial" and len(proposals) != 1:
            raise ValueError("serial native policy retains only its reference schedule")
        budget = next(fact.workspace_bytes for fact in execution.arena.plan.timeline.task_resources
                      if fact.task == self.original.task.symbol)
        self.source = replace(self.original, region=self.original.region.with_catalog(proposals, budget, stream_lanes=stream_lanes))
        self._scope = ExitStack()
        self._sizes: tuple[RegionTuningSize, ...] | None = None
        self._visited: set[int] = set()
        self._candidate: int | None = None
        self._measurements: dict[int, TokenwiseRegion] = {}
        self._candidates: tuple[RegionCandidate, ...] = ()
        self._rewrites: tuple[tuple[TensorRewrite, ...], ...] = ()
        self._closed = False
        try:
            self._scope.enter_context(execution.trace.private())
            devices = [] if execution.device.type == "cpu" else [execution.device.index or 0]
            self._scope.enter_context(preserve_torch_rng(devices))
            self._prepare_implementations()
        except BaseException:
            self._scope.close()
            raise

    @property
    def count(self) -> int:
        return len(self._candidates)

    def _prepare_implementations(self) -> None:
        region = self.source.region
        candidates = [RegionCandidate(proposal.payload.capacity.schedule_map, region.imports, (),
            stream_plan=proposal.stream_plan) for proposal in region.candidate_proposals]
        records: list[tuple[TensorRewrite, ...]] = [() for _ in candidates]
        seen = {boundary_digest(candidate) for candidate in candidates}
        budgets = {limit.device: limit.bytes for limit in region.payload_limits}
        parallel = next((proposal.stream_plan for proposal in region.candidate_proposals
            if proposal.stream_plan is not None), None)
        for prepared in prepare_region_implementations(region.program, region.functions, options=self.source.policy):
            alternatives = [prepared.candidate]
            # A changed implementation invalidates the old allocation witness.
            # Reuse only the schedule coordinates; lower physical evidence again.
            if parallel is not None and prepared.program == region.program:
                plan = lower_region_streams(prepared.program, parallel.schedule_map, prepared.candidate.imports,
                    axis=parallel.axis, lanes=parallel.lanes, retained=region.retained,
                    pipeline_depth=parallel.pipeline_depth)
                alternatives.append(replace(prepared.candidate, schedule_map=plan.phase_schedule, stream_plan=plan))
            for candidate in alternatives:
                payload = (analyze_region_payload(prepared.program, candidate.schedule_map, candidate.imports, region.retained)
                    if candidate.stream_plan is None else candidate.stream_plan.payload)
                proposal = RegionScheduleProposal(payload, candidate.stream_plan)
                if any(limit.device not in budgets or limit.bytes > budgets[limit.device] for limit in proposal.peak_bytes):
                    continue
                candidate = replace(candidate, schedule_map=payload.capacity.schedule_map)
                identity = boundary_digest(candidate)
                if identity not in seen:
                    seen.add(identity)
                    candidates.append(candidate)
                    records.append(prepared.rewrites)
        self._candidates, self._rewrites = tuple(candidates), tuple(records)

    def _request(self, candidates: tuple[RegionCandidate, ...]) -> RegionSelectionRequest:
        region = self.source.region
        # Unmeasured preparation uses the valid empty TokenwiseRegion domain.
        # Capacity still covers every parameter value. Once rows are observed,
        # the final measured request records their actual sizes instead.
        sizes = self._sizes if self._sizes is not None else tuple(
            RegionTuningSize(parameter.name, parameter.minimum)
            for parameter in sorted(region.program.parameters, key=lambda value: value.name))
        return RegionSelectionRequest(region.imports, candidates, region.payload_limits,
            self.source.policy.allow_numerical_transforms,
            "vLLM original-engine private prefill; completed generate wall time" if self._sizes is not None
            else "unmeasured worker preparation; full parameter-domain capacity",
            sizes, region.retained)

    def candidate(self, index: int) -> None:
        if self._closed:
            raise RuntimeError("native tuning scope has already closed")
        # Interleaved trials revisit candidates. Keep only owner-free lowering;
        # each execution still creates its own frame and invocation bindings.
        if type(index) is not int or not 0 <= index < self.count:
            raise ValueError("native tuning has no such funded candidate")
        region = self._measurements.get(index)
        if region is None:
            if index < len(self.source.region.candidate_proposals):
                region = self.source.region.measurement_schedule(index)
            else:
                request = self._request((self._candidates[0], self._candidates[index]))
                region = self.source.region.measurement_candidate(request, 1, self._rewrites[index], self.source.policy)
            # This worker command completes before the driver starts timing.
            # Source planning stays lazy, but trials must have local lowering
            # ready before entering the measured invocation.
            _ = region.execution
            self._measurements[index] = region
        self.execution.rebind_implementation(replace(self.source, region=region))
        self._candidate = index

    def record_rows(self, rows: tuple[PackedPhaseRow, ...]) -> None:
        if self._candidate is None or not rows or any(row.phase is not HiddenPhase.PREFILL for row in rows):
            raise ValueError("native tuning requires a prepared prefill-only sample")
        sizes: dict[str, int] = {}
        for request in range(self.source.region.domain.requests):
            selected = tuple(row for row in rows if row.request_ordinal == request)
            sizes[f"tokens{request}"] = len(selected)
            sizes[f"prefix{request}"] = selected[0].token_offset if selected else 0
        observed = tuple(RegionTuningSize(name, value) for name, value in sorted(sizes.items()))
        if self._sizes is not None and observed != self._sizes:
            raise ValueError("native tuning candidates must measure the same packed workload")
        self._sizes = observed
        self._visited.add(self._candidate)

    def finish(self, samples: tuple[tuple[int, ...], ...]) -> RegionSelection:
        if (self._closed or self._sizes is None or self._visited != set(range(self.count))
            or len(samples) != self.count or any(not values for values in samples)):
            raise ValueError("native tuning requires measurements for every funded candidate")
        region = self.source.region
        request = self._request(tuple(replace(candidate, elapsed_ns=elapsed)
            for candidate, elapsed in zip(self._candidates, samples, strict=True)))
        selection = select_region_candidate(region.program, request)
        selected = self.source.with_schedule(request, selection, self._rewrites[selection.selected])
        self.execution.rebind_implementation(selected)
        self._close()
        return selection

    def abort(self) -> None:
        if not self._closed:
            self.execution.rebind_implementation(self.original)
            self._close()

    def _close(self) -> None:
        try:
            if self.execution.device.type == "cuda":
                torch.cuda.synchronize(self.execution.device)
        finally:
            try:
                self._scope.close()
            finally:
                self._measurements.clear()
                self._closed = True
