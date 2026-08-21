"""Phase 1a compatibility compiler over the shared WorkGraph descriptors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence, cast

from tllm.consumers.base import BaseConsumer
from tllm.producers.registry import ExecutionPlan
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.workgraph.anchors import shadow_backend_snapshot
from tllm.workgraph.explain import explain_workgraph
from tllm.workgraph.frontend import CapturedFlow
from tllm.workgraph.hierarchy import (
    CaptureAttachment,
    CaptureHierarchy,
    apply_attachment_live_ranges,
    build_hierarchy,
)
from tllm.workgraph.ir import (
    BackendSnapshot,
    CompiledWorkGraph,
    WorkGraphDiagnostic,
    WorkGraphDiagnosticCode,
)
from tllm.workgraph.lowering import (
    RuntimePlanBinding,
    WorkGraphError,
    lower_plans,
)
from tllm.workgraph.reconcile import PlanDiff, reconcile
from tllm.workgraph.signature import structural_signature
from tllm.workgraph.trace import (
    TraceComparison,
    compare_event_traces,
    legacy_event_trace,
    shadow_event_trace,
)


@dataclass(frozen=True, slots=True)
class ShadowParityPlan:
    compiled: CompiledWorkGraph
    binding: RuntimePlanBinding
    hierarchy: CaptureHierarchy | None = None
    trace: TraceComparison | None = None
    legacy_boundaries: tuple[str, ...] = ()
    lowering_notes: tuple[str, ...] = ()
    diagnostics: tuple[WorkGraphDiagnostic, ...] = ()
    diff: PlanDiff | None = None

    @property
    def backend(self) -> BackendSnapshot | None:
        return self.compiled.semantic.backend

    @property
    def semantic(self):
        return self.compiled.semantic

    @property
    def data(self):
        return self.compiled.data_centric

    @property
    def epoch(self) -> int:
        return self.compiled.epoch

    @property
    def signature(self) -> str:
        return self.compiled.signature

    def explain(self) -> dict[str, object]:
        result = explain_workgraph(self.compiled, self.diff).to_dict()
        result["binding"] = {
            "dispatch_flow_ids": list(self.binding.dispatch_flow_ids),
            "dispatch_hooks": [list(item) for item in self.binding.dispatch_hooks],
            "dispatch_fusion_groups": [list(group) for group in self.binding.dispatch_fusion_groups],
            "execution_producer_ids": list(self.binding.execution_producer_ids),
            "producer_hooks": [
                [producer_id, list(hooks)]
                for producer_id, hooks in self.binding.producer_hooks
            ],
            "execution_port_sources": [list(item) for item in self.binding.execution_port_sources],
            "capture_resident_producer_ids": list(self.binding.capture_resident_producer_ids),
        }
        result["legacy_boundaries"] = list(self.legacy_boundaries)
        result["lowering_notes"] = list(self.lowering_notes)
        result["typed_diagnostics"] = [
            {
                "code": diagnostic.code.value,
                "message": diagnostic.message,
                "detail": diagnostic.detail,
            }
            for diagnostic in self.diagnostics
        ]
        result["capture_hierarchy"] = (
            None if self.hierarchy is None else self.hierarchy.to_dict()
        )
        result["event_trace"] = None if self.trace is None else self.trace.to_dict()
        return result


def _current_signature(
    graph: object,
    data: object,
    binding: RuntimePlanBinding,
    hierarchy: CaptureHierarchy | None,
) -> str:
    """Hash the IR plus the authoritative runtime-plan binding facts."""

    return structural_signature((graph, data, binding, hierarchy))


def compile_shadow_parity(
    *,
    consumers: Sequence[BaseConsumer[object]],
    dispatch_plan: DispatchPlan,
    execution_plan: ExecutionPlan,
    backend: BackendSnapshot,
    previous: ShadowParityPlan | None = None,
    strict_native: bool = False,
    captures: tuple[CapturedFlow, ...] = (),
    attachments: tuple[CaptureAttachment, ...] = (),
) -> ShadowParityPlan:
    """Build a shadow IR and verify its binding to existing runtime plans."""
    if backend.graph_mode not in {"eager", "piecewise", "full"}:
        raise WorkGraphError(
            "WG120_UNSUPPORTED_GRAPH_MODE",
            "Phase 1a supports eager/piecewise; full is legacy-only",
        )
    graph, data, binding, legacy, notes = lower_plans(
        consumers=consumers,
        dispatch_plan=dispatch_plan,
        execution_plan=execution_plan,
        backend=backend,
        strict_native=strict_native,
    )
    hierarchy: CaptureHierarchy | None = None
    if captures or attachments:
        hierarchy = build_hierarchy(
            captures=captures,
            attachments=attachments,
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            semantic=graph,
        )
        data = apply_attachment_live_ranges(data, hierarchy, execution_plan)
    expected_flows = tuple(target.flow_id for target in dispatch_plan.flow_targets())
    if binding.dispatch_flow_ids != expected_flows:
        raise WorkGraphError("WG200_DISPATCH_PARITY", "IR dispatch flow order differs from DispatchPlan")
    expected_dispatch_hooks = tuple(
        (target.flow_id, target.placement_hook) for target in dispatch_plan.flow_targets()
    )
    if binding.dispatch_hooks != expected_dispatch_hooks:
        raise WorkGraphError("WG205_DISPATCH_HOOK_PARITY", "IR dispatch hooks differ from DispatchPlan")
    expected_producers = tuple(planned.entry.producer_id for planned in execution_plan.producers)
    if binding.execution_producer_ids != expected_producers:
        raise WorkGraphError("WG201_EXECUTION_PARITY", "IR producer order differs from ExecutionPlan")
    expected_producer_hooks = tuple(
        (planned.entry.producer_id, tuple(sorted(hook.value for hook in planned.emits_at)))
        for planned in execution_plan.producers
    )
    if binding.producer_hooks != expected_producer_hooks:
        raise WorkGraphError("WG206_PRODUCER_HOOK_PARITY", "IR producer hooks differ from ExecutionPlan")
    expected_fusion_groups = tuple(dispatch_plan.fusion_groups)
    if binding.dispatch_fusion_groups != expected_fusion_groups:
        raise WorkGraphError("WG203_FUSION_PARITY", "IR fusion groups differ from DispatchPlan")
    expected_port_sources = tuple(
        sorted((str(port), str(producer_id)) for port, producer_id in execution_plan.port_sources.items())
    )
    if binding.execution_port_sources != expected_port_sources:
        raise WorkGraphError("WG204_PORT_SOURCE_PARITY", "IR port sources differ from ExecutionPlan")
    expected_capture_resident = tuple(execution_plan.capture_resident_producer_ids)
    if binding.capture_resident_producer_ids != expected_capture_resident:
        raise WorkGraphError(
            "WG207_CAPTURE_RESIDENT_PARITY",
            "IR capture-resident producer ids differ from ExecutionPlan",
        )
    trace = compare_event_traces(
        legacy_event_trace(dispatch_plan, execution_plan),
        shadow_event_trace(binding),
    )
    if not trace.equivalent:
        raise WorkGraphError("WG208_TRACE_PARITY", trace.mismatches[0].message)
    if data.moves:
        raise WorkGraphError("WG202_UNEXPECTED_MOVE", "Phase 1a Data-Centric IR must not contain Move")

    diagnostics: tuple[WorkGraphDiagnostic, ...] = ()
    if backend.graph_mode not in {"eager", "piecewise"}:
        notes = (*notes, f"WG120_LEGACY_GRAPH_MODE:{backend.graph_mode}")
        diagnostics = (
            WorkGraphDiagnostic(
                code=WorkGraphDiagnosticCode.UNSUPPORTED_GRAPH_MODE,
                message="Phase 1a WorkGraph does not execute in full graph mode",
                detail="legacy DispatchPlan/ExecutionPlan dispatch remains authoritative",
            ),
        )

    candidate = CompiledWorkGraph(
        semantic=graph,
        data_centric=data,
        structural_signature=_current_signature(graph, data, binding, hierarchy),
        diagnostics=tuple((*legacy, *notes)),
    )
    if backend.graph_mode in {"eager", "piecewise"}:
        try:
            from tllm import core as _core

            cast(Any, _core.validate_workgraph)(graph, data)
            if hierarchy is not None:
                cast(Any, _core.validate_capture_hierarchy)(hierarchy, graph)
        except (TypeError, ValueError) as exc:
            raise WorkGraphError(
                "WG300_RUST_VALIDATION",
                "Rust WorkGraph descriptor validation failed",
            ) from exc
    previous_compiled = None if previous is None else previous.compiled
    diff = reconcile(previous_compiled, candidate)
    compiled = replace(candidate, epoch=diff.epoch)
    return ShadowParityPlan(
        compiled=compiled,
        binding=binding,
        hierarchy=hierarchy,
        trace=trace,
        legacy_boundaries=legacy,
        lowering_notes=notes,
        diagnostics=diagnostics,
        diff=diff,
    )


def compile_shadow_snapshot(
    *,
    consumers: Sequence[BaseConsumer[object]],
    dispatch_plan: DispatchPlan,
    execution_plan: ExecutionPlan,
    previous: ShadowParityPlan | None = None,
    captures: tuple[CapturedFlow, ...] = (),
    attachments: tuple[CaptureAttachment, ...] = (),
) -> ShadowParityPlan:
    """Compile startup metadata before a concrete vLLM runner is selected."""
    return compile_shadow_parity(
        consumers=consumers,
        dispatch_plan=dispatch_plan,
        execution_plan=execution_plan,
        backend=shadow_backend_snapshot(),
        previous=previous,
        strict_native=False,
        captures=captures,
        attachments=attachments,
    )


__all__ = ["ShadowParityPlan", "PlanDiff", "compile_shadow_parity", "compile_shadow_snapshot"]
