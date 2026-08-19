"""Cold-path event equivalence oracle for the Phase 1 compatibility plan.

The trace is deliberately a pure descriptor.  It records the stable facts
that can change runtime ordering or binding, while leaving all execution
objects owned by the legacy runtime and the eventual physical executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from tllm.producers.registry import ExecutionPlan
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.workgraph.ir import RuntimePlanBinding


class TraceOrigin(StrEnum):
    LEGACY = "legacy"
    SHADOW = "shadow"


class TraceEventKind(StrEnum):
    PRODUCER_HOOK = "producer_hook"
    CONSUMER_FLOW_HOOK = "consumer_flow_hook"
    FUSION_MEMBERSHIP = "fusion_membership"
    PORT_SOURCE_BINDING = "port_source_binding"
    CAPTURE_RESIDENCY = "capture_residency"


def _require_nonempty(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One stable event fact at a canonical trace ordinal."""

    ordinal: int
    kind: TraceEventKind
    subject: str
    details: tuple[str, ...] = ()
    declared_subject: str = ""

    def __post_init__(self) -> None:
        ordinal = cast(object, self.ordinal)
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise TypeError("TraceEvent.ordinal must be int")
        if ordinal < 0:
            raise ValueError("TraceEvent.ordinal must be non-negative")
        if not isinstance(cast(object, self.kind), TraceEventKind):
            raise TypeError("TraceEvent.kind must be TraceEventKind")
        object.__setattr__(self, "subject", _require_nonempty(self.subject, "TraceEvent.subject"))
        details = cast(object, self.details)
        if not isinstance(details, tuple) or any(
            not isinstance(detail, str) or not detail
            for detail in cast(tuple[object, ...], details)
        ):
            raise TypeError("TraceEvent.details must be tuple[str, ...] of non-empty strings")
        declared_subject = self.declared_subject or self.subject
        object.__setattr__(self, "declared_subject", _require_nonempty(
            declared_subject,
            "TraceEvent.declared_subject",
        ))

    @property
    def canonical(self) -> str:
        details = ",".join(self.details)
        if self.declared_subject == self.subject:
            return f"{self.kind.value}:{self.subject}[{details}]"
        return (
            f"{self.kind.value}:{self.subject}"
            f"{{declared={self.declared_subject}}}[{details}]"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "kind": self.kind.value,
            "subject": self.subject,
            "details": list(self.details),
            "declared_subject": self.declared_subject,
        }


@dataclass(frozen=True, slots=True)
class EventTrace:
    """A complete ordered trace from one plan representation."""

    origin: TraceOrigin
    events: tuple[TraceEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.origin), TraceOrigin):
            raise TypeError("EventTrace.origin must be TraceOrigin")
        events = cast(object, self.events)
        if not isinstance(events, tuple) or any(
            not isinstance(event, TraceEvent)
            for event in cast(tuple[object, ...], events)
        ):
            raise TypeError("EventTrace.events must be tuple[TraceEvent, ...]")
        for ordinal, event in enumerate(self.events):
            if event.ordinal != ordinal:
                raise ValueError("EventTrace event ordinals must be contiguous")

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin.value,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True, slots=True)
class TraceMismatch:
    """One event ordinal where legacy and shadow facts differ."""

    ordinal: int
    legacy: TraceEvent | None
    shadow: TraceEvent | None

    def __post_init__(self) -> None:
        ordinal = cast(object, self.ordinal)
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise TypeError("TraceMismatch.ordinal must be int")
        if ordinal < 0:
            raise ValueError("TraceMismatch.ordinal must be non-negative")
        if self.legacy is None and self.shadow is None:
            raise ValueError("TraceMismatch must retain a legacy or shadow event")
        if self.legacy is not None and not isinstance(cast(object, self.legacy), TraceEvent):
            raise TypeError("TraceMismatch.legacy must be TraceEvent or None")
        if self.shadow is not None and not isinstance(cast(object, self.shadow), TraceEvent):
            raise TypeError("TraceMismatch.shadow must be TraceEvent or None")

    @property
    def message(self) -> str:
        legacy = "<missing>" if self.legacy is None else self.legacy.canonical
        shadow = "<missing>" if self.shadow is None else self.shadow.canonical
        return f"trace mismatch ordinal={self.ordinal}: legacy={legacy}; shadow={shadow}"

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "legacy": None if self.legacy is None else self.legacy.to_dict(),
            "shadow": None if self.shadow is None else self.shadow.to_dict(),
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class TraceComparison:
    legacy: EventTrace
    shadow: EventTrace
    mismatches: tuple[TraceMismatch, ...]

    def __post_init__(self) -> None:
        if self.legacy.origin != TraceOrigin.LEGACY:
            raise ValueError("TraceComparison.legacy must have legacy origin")
        if self.shadow.origin != TraceOrigin.SHADOW:
            raise ValueError("TraceComparison.shadow must have shadow origin")
        mismatches = cast(object, self.mismatches)
        if not isinstance(mismatches, tuple) or any(
            not isinstance(mismatch, TraceMismatch)
            for mismatch in cast(tuple[object, ...], mismatches)
        ):
            raise TypeError("TraceComparison.mismatches must be tuple[TraceMismatch, ...]")

    @property
    def equivalent(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, object]:
        return {
            "legacy": self.legacy.to_dict(),
            "shadow": self.shadow.to_dict(),
            "equivalent": self.equivalent,
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
        }


def _event(
    ordinal: int,
    kind: TraceEventKind,
    subject: str,
    details: tuple[str, ...] = (),
    *,
    declared_subject: str = "",
) -> TraceEvent:
    return TraceEvent(
        ordinal=ordinal,
        kind=kind,
        subject=subject,
        details=details,
        declared_subject=declared_subject,
    )


def legacy_event_trace(dispatch_plan: DispatchPlan, execution_plan: ExecutionPlan) -> EventTrace:
    """Lower authoritative runtime plans into their stable event sequence."""

    events: list[TraceEvent] = []
    for planned in execution_plan.producers:
        producer_id = planned.entry.producer_id
        for hook in sorted(hook.value for hook in planned.emits_at):
            events.append(
                _event(
                    len(events),
                    TraceEventKind.PRODUCER_HOOK,
                    producer_id,
                    (hook,),
                )
            )
    for target in dispatch_plan.flow_targets():
        events.append(
            _event(
                len(events),
                TraceEventKind.CONSUMER_FLOW_HOOK,
                target.flow_id,
                (target.placement_hook,),
            )
        )
    for group_index, group in enumerate(dispatch_plan.fusion_groups):
        events.append(
            _event(
                len(events),
                TraceEventKind.FUSION_MEMBERSHIP,
                f"group/{group_index}",
                tuple(group),
            )
        )
    for port, producer_id in sorted(
        (str(port), str(producer_id))
        for port, producer_id in execution_plan.port_sources.items()
    ):
        events.append(
            _event(
                len(events),
                TraceEventKind.PORT_SOURCE_BINDING,
                port,
                (producer_id,),
            )
        )
    for producer_id in execution_plan.capture_resident_producer_ids:
        events.append(
            _event(
                len(events),
                TraceEventKind.CAPTURE_RESIDENCY,
                producer_id,
            )
        )
    return EventTrace(TraceOrigin.LEGACY, tuple(events))


def _indexed_subjects(
    subjects: tuple[str, ...],
    declarations: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    count = max(len(subjects), len(declarations))
    result: list[tuple[str, str, tuple[str, ...]]] = []
    for index in range(count):
        subject = subjects[index] if index < len(subjects) else "<missing>"
        if index < len(declarations):
            declared_subject, details = declarations[index]
        else:
            declared_subject, details = "<missing>", ()
        result.append((subject, declared_subject, details))
    return tuple(result)


def _indexed_hooks(
    subjects: tuple[str, ...],
    declarations: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str, str], ...]:
    count = max(len(subjects), len(declarations))
    result: list[tuple[str, str, str]] = []
    for index in range(count):
        subject = subjects[index] if index < len(subjects) else "<missing>"
        if index < len(declarations):
            declared_subject, hook = declarations[index]
        else:
            declared_subject, hook = "<missing>", "<missing>"
        result.append((subject, declared_subject, hook))
    return tuple(result)


def shadow_event_trace(binding: RuntimePlanBinding) -> EventTrace:
    """Expand a typed RuntimePlanBinding into the same canonical events."""

    events: list[TraceEvent] = []
    producer_declarations = _indexed_subjects(
        binding.execution_producer_ids,
        binding.producer_hooks,
    )
    for producer_id, declared_id, hooks in producer_declarations:
        for hook in hooks:
            events.append(
                _event(
                    len(events),
                    TraceEventKind.PRODUCER_HOOK,
                    producer_id,
                    (hook,),
                    declared_subject=declared_id,
                )
            )
    flow_declarations = _indexed_hooks(
        binding.dispatch_flow_ids,
        binding.dispatch_hooks,
    )
    for flow_id, declared_id, hook in flow_declarations:
        events.append(
            _event(
                len(events),
                TraceEventKind.CONSUMER_FLOW_HOOK,
                flow_id,
                (hook,),
                declared_subject=declared_id,
            )
        )
    for group_index, group in enumerate(binding.dispatch_fusion_groups):
        events.append(
            _event(
                len(events),
                TraceEventKind.FUSION_MEMBERSHIP,
                f"group/{group_index}",
                tuple(group),
            )
        )
    for port, producer_id in binding.execution_port_sources:
        events.append(
            _event(
                len(events),
                TraceEventKind.PORT_SOURCE_BINDING,
                port,
                (producer_id,),
            )
        )
    for producer_id in binding.capture_resident_producer_ids:
        events.append(
            _event(
                len(events),
                TraceEventKind.CAPTURE_RESIDENCY,
                producer_id,
            )
        )
    return EventTrace(TraceOrigin.SHADOW, tuple(events))


def compare_event_traces(legacy: EventTrace, shadow: EventTrace) -> TraceComparison:
    """Compare every ordinal and retain both typed events for each mismatch."""

    if legacy.origin != TraceOrigin.LEGACY:
        raise ValueError("legacy trace must have legacy origin")
    if shadow.origin != TraceOrigin.SHADOW:
        raise ValueError("shadow trace must have shadow origin")
    mismatch_count = max(len(legacy.events), len(shadow.events))
    mismatches: list[TraceMismatch] = []
    for ordinal in range(mismatch_count):
        legacy_event = legacy.events[ordinal] if ordinal < len(legacy.events) else None
        shadow_event = shadow.events[ordinal] if ordinal < len(shadow.events) else None
        if legacy_event != shadow_event:
            mismatches.append(TraceMismatch(ordinal, legacy_event, shadow_event))
    return TraceComparison(legacy, shadow, tuple(mismatches))


__all__ = [
    "EventTrace",
    "TraceComparison",
    "TraceEvent",
    "TraceEventKind",
    "TraceMismatch",
    "TraceOrigin",
    "compare_event_traces",
    "legacy_event_trace",
    "shadow_event_trace",
]
