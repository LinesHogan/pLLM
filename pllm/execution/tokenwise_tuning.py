"""Cold native trials using the original execution, request owners and trace."""

from contextlib import ExitStack
from dataclasses import replace
from typing import Generic, TypeVar

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_selection import RegionSelection, RegionTuningSize
from pllm.contracts.semantic_graph import ImplementationId
from pllm.execution.region_optimization import prepare_region_candidates, select_region_candidate, warn_region_transforms, bind_region_candidate
from pllm.execution.codec import preserve_torch_rng
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
        self._visited: set[int] = set()
        self._candidate: int | None = None
        self._measurements: dict[int, TokenwiseRegion] = {}
        self._closed = False
        try:
            self._scope.enter_context(execution.trace.private())
            devices = [] if execution.device.type == "cpu" else [execution.device.index or 0]
            self._scope.enter_context(preserve_torch_rng(devices))
            region = self.source.region
            prepared, request, assessments = prepare_region_candidates(region.program, region.functions, limits=region.payload_limits,
                options=self.source.policy, parallel=stream_lanes > 1, retained=region.retained,
                sizes={p.name: p.minimum for p in region.program.parameters}, catalog=region.candidate_proposals)
            self._prepared = tuple(p for p, a in zip(prepared, assessments, strict=True) if a.feasible)
            self._assessments = tuple(a for a in assessments if a.feasible)
            self._request = replace(request, candidates=tuple(p.candidate for p in self._prepared))
        except BaseException:
            self._scope.close()
            raise

    @property
    def count(self) -> int:
        return len(self._prepared)

    def candidate(self, index: int) -> None:
        if self._closed:
            raise RuntimeError("native tuning scope has already closed")
        # Interleaved trials revisit candidates. Keep only owner-free lowering;
        # each execution still creates its own frame and invocation bindings.
        if type(index) is not int or not 0 <= index < self.count:
            raise ValueError("native tuning has no such funded candidate")
        if index not in self._measurements:
            bound = bind_region_candidate(self._prepared[index], self.source.region.payload_limits, self._assessments[index])
            self._measurements[index] = self.source.region.with_execution(bound)
        self.execution.rebind_implementation(replace(self.source, region=self._measurements[index]))
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
        if self._visited and observed != self._request.tuning_sizes:
            raise ValueError("native tuning candidates must measure the same packed workload")
        self._request = replace(self._request, tuning_sizes=observed, measurement_context="original-engine private prefill; completed generate wall time")
        self._visited.add(self._candidate)

    def finish(self, samples: tuple[tuple[int, ...], ...]) -> RegionSelection:
        if (self._closed or self._visited != set(range(self.count))
            or len(samples) != self.count or any(not values for values in samples)):
            raise ValueError("native tuning requires measurements for every funded candidate")
        region = self.source.region
        request = replace(self._request, candidates=tuple(replace(candidate, elapsed_ns=elapsed)
            for candidate, elapsed in zip(self._request.candidates, samples, strict=True)))
        selection = select_region_candidate(region.program, request)
        # The visited-set check above guarantees the winner was measured, so its
        # executor is already bound: install it instead of replaying the bind.
        bound = self._measurements[selection.selected].execution
        warn_region_transforms(bound, selection.candidates[selection.selected], selection.signature)
        final = self._measurements[selection.selected]
        self.execution.rebind_implementation(replace(self.source, region=final))
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
