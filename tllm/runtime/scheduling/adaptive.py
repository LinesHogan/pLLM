"""Mutable adaptive admission runtime driven by frozen admission plans.

The admission control plane owns candidate actions, bucket identities, budgets,
deadlines, and the placement move universe.  This module owns only mutable
runtime facts: integer EWMAs and current admission reservations.  It never
creates or edits a ``Move`` or ``ScheduledMove``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeVar, cast

from tllm.workgraph.ir import NodeKey
from tllm.workgraph.placement import MoveKey, PlacementPlan, ValueKey
from tllm.workgraph.admission import (
    AdmissionMode,
    CandidateActionDesc,
    CandidateActionKey,
    CandidateActionKind,
    CandidateMoveSetDesc,
    IntegerEWMAProfile,
    MoveSetKey,
    EWMAProfileKey,
    InterferenceBudgetDesc,
    InterferenceBudgetKey,
    AdmissionControlPlan,
    ShapeBucketDesc,
    ShapeBucketKey,
    WorkItemDesc,
    WorkItemKey,
    WorkloadBucketDesc,
    compile_admission_plan,
)


class AdaptiveMode(StrEnum):
    STATIC = "static"
    ADAPTIVE = "adaptive"


class AdaptiveDecisionKind(StrEnum):
    RUN = "run"
    KEEP = "keep"
    DROP = "drop"
    FALLBACK = "fallback"


class AdaptiveTransferStatus(StrEnum):
    READY = "ready"
    PENDING = "pending"


_T = TypeVar("_T")
_RuntimeKey = CandidateActionKey | EWMAProfileKey | InterferenceBudgetKey


def _instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _runtime_key(value: object, label: str) -> _RuntimeKey:
    if not isinstance(value, (CandidateActionKey, EWMAProfileKey, InterferenceBudgetKey)):
        raise TypeError(
            f"{label} must be CandidateActionKey, EWMAProfileKey, or InterferenceBudgetKey"
        )
    return cast(_RuntimeKey, value)


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    """One typed runtime decision for a WorkItemDesc."""

    work_key: WorkItemKey
    action: AdaptiveDecisionKind
    reason: str
    plan_epoch: int = 1
    action_key: CandidateActionKey | None = None
    candidate: CandidateActionDesc | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_key", _instance(self.work_key, WorkItemKey, "decision work_key"))
        if not isinstance(self.action, AdaptiveDecisionKind):
            try:
                object.__setattr__(self, "action", AdaptiveDecisionKind(str(self.action)))
            except ValueError as exc:
                raise ValueError("decision action must be run, keep, drop, or fallback") from exc
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise TypeError("decision reason must be non-empty str")
        object.__setattr__(self, "plan_epoch", _integer(self.plan_epoch, "decision plan_epoch", minimum=1))
        if self.action_key is not None:
            object.__setattr__(self, "action_key", _instance(self.action_key, CandidateActionKey, "decision action_key"))
        if self.candidate is not None:
            object.__setattr__(self, "candidate", _instance(self.candidate, CandidateActionDesc, "decision candidate"))

    @property
    def work_id(self) -> str:
        return self.work_key.value

    @property
    def candidate_key(self) -> CandidateActionKey | None:
        return self.action_key

    @property
    def decision(self) -> AdaptiveDecisionKind:
        return self.action

    @property
    def kind(self) -> AdaptiveDecisionKind:
        return self.action

    @property
    def selected(self) -> bool:
        return self.action in (AdaptiveDecisionKind.RUN, AdaptiveDecisionKind.KEEP)


@dataclass(frozen=True, slots=True)
class AdmissionContext:
    """Python-owned per-step resource facts used by adaptive admission."""

    step: int = 0
    available_buffers: int | None = None
    buffer_capacity: int | None = None
    buffer_used: int = 0
    deadline_step: int | None = None
    deadline_slack: int | None = None
    interference: Mapping[InterferenceBudgetKey, int] = field(default_factory=dict)
    interference_limit: int | Mapping[InterferenceBudgetKey, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _integer(self.step, "admission step"))
        for name, value in (
            ("available_buffers", self.available_buffers),
            ("buffer_capacity", self.buffer_capacity),
            ("buffer_used", self.buffer_used),
        ):
            if value is not None:
                object.__setattr__(self, name, _integer(value, f"admission {name}"))
        for name, value in (
            ("deadline_step", self.deadline_step),
            ("deadline_slack", self.deadline_slack),
        ):
            if value is not None:
                object.__setattr__(self, name, _integer(value, f"admission {name}"))
        if not isinstance(self.interference, Mapping):
            raise TypeError("admission interference must be a mapping")
        if any(not isinstance(key, InterferenceBudgetKey) for key in self.interference):
            raise TypeError("admission interference keys must be InterferenceBudgetKey")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.interference.values()):
            raise TypeError("admission interference values must be non-negative ints")
        limit = self.interference_limit
        if limit is not None and not isinstance(limit, (int, Mapping)):
            raise TypeError("admission interference_limit must be int, mapping, or None")
        if isinstance(limit, int):
            _integer(limit, "admission interference_limit")
        elif isinstance(limit, Mapping):
            if any(not isinstance(key, InterferenceBudgetKey) for key in limit):
                raise TypeError("admission interference_limit keys must be InterferenceBudgetKey")
            for value in limit.values():
                _integer(value, "admission interference_limit value")


@dataclass(slots=True)
class IntegerEWMA:
    """Mutable integer-only exponentially weighted moving average."""

    alpha_num: int = 1
    alpha_den: int = 2
    values: dict[_RuntimeKey, int] = field(default_factory=dict)
    profiles: dict[EWMAProfileKey, IntegerEWMAProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.alpha_num = _integer(self.alpha_num, "EWMA alpha_num", minimum=1)
        self.alpha_den = _integer(self.alpha_den, "EWMA alpha_den", minimum=1)
        if self.alpha_num > self.alpha_den:
            raise ValueError("EWMA alpha_num must not exceed alpha_den")
        for key, value in tuple(self.values.items()):
            _runtime_key(key, "EWMA key")
            self.values[key] = _integer(value, "EWMA value")
        configured = tuple(self.profiles.values())
        self.profiles.clear()
        self.configure_profiles(configured)

    def configure_profiles(
        self,
        profiles: Sequence[IntegerEWMAProfile],
    ) -> None:
        typed = tuple(
            _instance(profile, IntegerEWMAProfile, "EWMA profile")
            for profile in profiles
        )
        if len({profile.key for profile in typed}) != len(typed):
            raise ValueError("EWMA profiles must use unique keys")
        self.profiles = {profile.key: profile for profile in typed}
        for profile in typed:
            self.values[profile.key] = profile.estimate

    def clear(self) -> None:
        self.values.clear()
        self.profiles.clear()

    def update(self, key: _RuntimeKey, observation: int) -> int:
        key = _runtime_key(key, "EWMA key")
        sample = _integer(observation, "EWMA observation")
        if isinstance(key, EWMAProfileKey) and key in self.profiles:
            updated = self.profiles[key].observe(sample)
            self.profiles[key] = updated
            self.values[key] = updated.estimate
            return updated.estimate
        previous = self.values.get(key)
        if previous is None:
            estimate = sample
        else:
            weighted = ((self.alpha_den - self.alpha_num) * previous) + self.alpha_num * sample
            estimate = (weighted + self.alpha_den // 2) // self.alpha_den
        self.values[key] = int(max(0, estimate))
        return self.values[key]

    def get(self, key: _RuntimeKey, default: int | None = None) -> int | None:
        return self.values.get(_runtime_key(key, "EWMA key"), default)

    def snapshot(self) -> Mapping[_RuntimeKey, int]:
        return MappingProxyType(dict(self.values))


@dataclass(slots=True)
class RuntimeAdmissionState:
    """Mutable reservations and EWMAs retained only by the runtime."""

    reserved_buffers: int = 0
    reserved_budgets: dict[InterferenceBudgetKey, int] = field(default_factory=dict)
    latency_ewma: IntegerEWMA = field(default_factory=IntegerEWMA)
    interference_ewma: IntegerEWMA = field(default_factory=IntegerEWMA)

    def reset(self) -> None:
        self.reserved_buffers = 0
        self.reserved_budgets.clear()


@dataclass(frozen=True, slots=True)
class AdaptiveTransferBinding:
    """Bind one approved MoveSet route to explicit bundle payload entries."""

    move_set: MoveSetKey
    producer: NodeKey
    consumer: NodeKey
    value: ValueKey
    source_entry: str
    destination_entry: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "move_set",
            _instance(self.move_set, MoveSetKey, "AdaptiveTransferBinding.move_set"),
        )
        object.__setattr__(
            self,
            "producer",
            _instance(self.producer, NodeKey, "AdaptiveTransferBinding.producer"),
        )
        object.__setattr__(
            self,
            "consumer",
            _instance(self.consumer, NodeKey, "AdaptiveTransferBinding.consumer"),
        )
        object.__setattr__(
            self,
            "value",
            _instance(self.value, ValueKey, "AdaptiveTransferBinding.value"),
        )
        for name, value in (
            ("source_entry", self.source_entry),
            ("destination_entry", self.destination_entry),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
                or "0x" in value.lower()
                or "<" in value
                or ">" in value
            ):
                raise ValueError(f"AdaptiveTransferBinding.{name} must be stable text")
            object.__setattr__(self, name, value.strip())
        if self.source_entry == self.destination_entry:
            raise ValueError("adaptive transfer source and destination entries must differ")

    @property
    def route_key(self) -> tuple[NodeKey, NodeKey, ValueKey]:
        return self.producer, self.consumer, self.value


@dataclass(frozen=True, slots=True)
class AdaptiveFlowBinding:
    """Explicitly migrate one legacy flow id to one typed work descriptor."""

    flow_id: str
    work: WorkItemDesc
    transfers: tuple[AdaptiveTransferBinding, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.flow_id, str)
            or not self.flow_id.strip()
            or "0x" in self.flow_id.lower()
            or "<" in self.flow_id
            or ">" in self.flow_id
        ):
            raise ValueError("adaptive flow_id must be a stable non-empty string")
        object.__setattr__(self, "flow_id", self.flow_id.strip())
        object.__setattr__(
            self,
            "work",
            _instance(self.work, WorkItemDesc, "AdaptiveFlowBinding.work"),
        )
        if not isinstance(self.transfers, tuple) or any(
            not isinstance(transfer, AdaptiveTransferBinding)
            for transfer in self.transfers
        ):
            raise TypeError(
                "AdaptiveFlowBinding.transfers must be tuple[AdaptiveTransferBinding, ...]"
            )
        move_sets = tuple(transfer.move_set for transfer in self.transfers)
        if len(set(move_sets)) != len(move_sets):
            raise ValueError(
                "AdaptiveFlowBinding supports one explicit route per MoveSet"
            )
        object.__setattr__(
            self,
            "transfers",
            tuple(sorted(self.transfers, key=lambda transfer: transfer.move_set)),
        )

    def transfer_for(
        self,
        move_set: MoveSetKey,
    ) -> AdaptiveTransferBinding | None:
        return next(
            (
                transfer
                for transfer in self.transfers
                if transfer.move_set == move_set
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class AdaptiveRuntimeBinding:
    """Python-owned cold compiler inputs and explicit legacy-flow bindings."""

    move_sets: tuple[CandidateMoveSetDesc, ...]
    workload_buckets: tuple[WorkloadBucketDesc, ...]
    shape_buckets: tuple[ShapeBucketDesc, ...]
    budgets: tuple[InterferenceBudgetDesc, ...]
    actions: tuple[CandidateActionDesc, ...]
    unknown_bucket_fallback: CandidateActionKey
    flow_bindings: tuple[AdaptiveFlowBinding, ...]
    profiles: tuple[IntegerEWMAProfile, ...] = ()
    buffer_capacity: int = 0
    mode: AdmissionMode = AdmissionMode.STATIC
    densification_limit: int = 64

    def __post_init__(self) -> None:
        typed_fields = (
            ("move_sets", self.move_sets, CandidateMoveSetDesc),
            ("workload_buckets", self.workload_buckets, WorkloadBucketDesc),
            ("shape_buckets", self.shape_buckets, ShapeBucketDesc),
            ("budgets", self.budgets, InterferenceBudgetDesc),
            ("actions", self.actions, CandidateActionDesc),
            ("profiles", self.profiles, IntegerEWMAProfile),
            ("flow_bindings", self.flow_bindings, AdaptiveFlowBinding),
        )
        for label, values, expected in typed_fields:
            if not isinstance(values, tuple) or any(
                not isinstance(item, expected) for item in values
            ):
                raise TypeError(
                    f"AdaptiveRuntimeBinding.{label} must be tuple[{expected.__name__}, ...]"
                )
        if len({binding.flow_id for binding in self.flow_bindings}) != len(
            self.flow_bindings
        ):
            raise ValueError("adaptive flow bindings must use unique flow ids")
        if len({binding.work.key for binding in self.flow_bindings}) != len(
            self.flow_bindings
        ):
            raise ValueError("adaptive flow bindings must use unique work keys")
        _instance(
            self.unknown_bucket_fallback,
            CandidateActionKey,
            "AdaptiveRuntimeBinding.unknown_bucket_fallback",
        )
        _integer(self.buffer_capacity, "AdaptiveRuntimeBinding.buffer_capacity")
        _integer(
            self.densification_limit,
            "AdaptiveRuntimeBinding.densification_limit",
            minimum=1,
        )
        if not isinstance(self.mode, AdmissionMode):
            try:
                object.__setattr__(self, "mode", AdmissionMode(self.mode))
            except ValueError as exc:
                raise ValueError("adaptive runtime mode must be static or adaptive") from exc

    def compile(
        self,
        placement_plan: PlacementPlan,
    ) -> tuple[
        AdmissionControlPlan,
        AdaptiveAdmissionScheduler,
        tuple[AdaptiveFlowBinding, ...],
    ]:
        plan = compile_admission_plan(
            placement_plan=placement_plan,
            move_sets=self.move_sets,
            workload_buckets=self.workload_buckets,
            shape_buckets=self.shape_buckets,
            budgets=self.budgets,
            actions=self.actions,
            profiles=self.profiles,
            unknown_bucket_fallback=self.unknown_bucket_fallback,
            buffer_capacity=self.buffer_capacity,
            mode=self.mode,
            densification_limit=self.densification_limit,
        )
        fallback = next(
            action
            for action in plan.actions
            if action.key == plan.unknown_bucket_fallback
        )
        if any(
            binding.work.semantic_key != fallback.semantic_key
            for binding in self.flow_bindings
        ):
            raise ValueError(
                "adaptive flow semantics must match the authoritative unknown-bucket fallback"
            )
        validated_bindings = validate_adaptive_flow_bindings(
            plan,
            self.flow_bindings,
        )
        return plan, AdaptiveAdmissionScheduler(plan), validated_bindings


@dataclass(frozen=True, slots=True)
class PreparedAdmissionPlan:
    """Immutable runtime index over one AdmissionControlPlan."""

    plan: AdmissionControlPlan
    actions: tuple[CandidateActionDesc, ...]
    action_by_key: Mapping[CandidateActionKey, CandidateActionDesc]
    fallback_action: CandidateActionDesc
    plan_epoch: int
    buffer_capacity: int
    budgets: Mapping[InterferenceBudgetKey, InterferenceBudgetDesc]
    move_sets: Mapping[MoveSetKey, CandidateMoveSetDesc]
    move_keys: tuple[MoveKey, ...]
    shape_buckets: tuple[ShapeBucketDesc, ...]
    workload_buckets: tuple[WorkloadBucketDesc, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_by_key", MappingProxyType(dict(self.action_by_key)))
        object.__setattr__(self, "budgets", MappingProxyType(dict(self.budgets)))
        object.__setattr__(self, "move_sets", MappingProxyType(dict(self.move_sets)))
        object.__setattr__(self, "plan_epoch", _integer(self.plan_epoch, "plan epoch", minimum=1))
        object.__setattr__(self, "buffer_capacity", _integer(self.buffer_capacity, "buffer capacity"))


def prepare_admission_plan(plan: AdmissionControlPlan) -> PreparedAdmissionPlan:
    """Prepare one frozen control plan without modifying it."""

    plan = _instance(plan, AdmissionControlPlan, "phase-5 plan")
    actions = tuple(plan.actions)
    action_by_key = {action.key: action for action in actions}
    fallback = action_by_key.get(plan.unknown_bucket_fallback)
    if fallback is None:
        raise ValueError("phase-5 fallback is not an approved action")
    budgets = {budget.key: budget for budget in plan.budgets}
    move_sets = {move_set.key: move_set for move_set in plan.move_sets}
    return PreparedAdmissionPlan(
        plan=plan,
        actions=actions,
        action_by_key=action_by_key,
        fallback_action=fallback,
        plan_epoch=plan.plan_epoch,
        buffer_capacity=plan.buffer_capacity,
        budgets=budgets,
        move_sets=move_sets,
        move_keys=tuple(move.key for move in plan.scheduled_moves),
        shape_buckets=tuple(plan.shape_buckets),
        workload_buckets=tuple(plan.workload_buckets),
    )


def _shape_bucket(prepared: PreparedAdmissionPlan, shape: tuple[int, ...]) -> ShapeBucketKey | None:
    matches = tuple(bucket.key for bucket in prepared.shape_buckets if bucket.matches(shape))
    return min(matches) if matches else None


def _approved_actions(prepared: PreparedAdmissionPlan, item: WorkItemDesc) -> tuple[CandidateActionDesc, ...]:
    shape_key = _shape_bucket(prepared, item.shape)
    workload_key = item.workload_bucket
    workload_keys = {bucket.key for bucket in prepared.workload_buckets}
    if workload_key is None or workload_key not in workload_keys or shape_key is None:
        return (_semantic_fallback(prepared, item, unknown_bucket=True),)
    must_complete = _must_complete(prepared, item)
    actions = tuple(
        action
        for action in prepared.actions
        if not action.unknown_bucket
        and action.workload_bucket == workload_key
        and (action.shape_bucket is None or action.shape_bucket == shape_key)
        and action.semantic_key == item.semantic_key
        and (not must_complete or action.kind != CandidateActionKind.SHED)
    )
    if not actions:
        return (_semantic_fallback(prepared, item, unknown_bucket=False),)
    return tuple(sorted(actions, key=lambda action: (-action.priority, action.key)))


def _semantic_fallback(
    prepared: PreparedAdmissionPlan,
    item: WorkItemDesc,
    *,
    unknown_bucket: bool,
) -> CandidateActionDesc:
    fallback = prepared.fallback_action
    if fallback.semantic_key != item.semantic_key:
        boundary = "unknown bucket" if unknown_bucket else "work item"
        raise RuntimeError(
            f"{boundary} has no semantic-preserving authoritative fallback"
        )
    return fallback


def _must_complete(prepared: PreparedAdmissionPlan, item: WorkItemDesc) -> bool:
    workload_required = any(
        bucket.key == item.workload_bucket and bucket.required_by_default
        for bucket in prepared.workload_buckets
    )
    shape_key = _shape_bucket(prepared, item.shape)
    action_required = any(
        action.required
        and action.semantic_key == item.semantic_key
        and (
            (
                item.workload_bucket is not None
                and action.workload_bucket == item.workload_bucket
                and (action.shape_bucket is None or action.shape_bucket == shape_key)
            )
            or (
                (item.workload_bucket is None or shape_key is None)
                and action.key == prepared.fallback_action.key
            )
        )
        for action in prepared.actions
    )
    return (
        item.required
        or workload_required
        or action_required
        or not item.allow_shedding
    )


def _action_is_approved(
    prepared: PreparedAdmissionPlan,
    item: WorkItemDesc,
    candidate: CandidateActionDesc,
) -> bool:
    primary = _approved_actions(prepared, item)
    if candidate in primary:
        return True
    return any(
        action.fallback_key == candidate.key
        and candidate.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK
        and candidate.semantic_key == item.semantic_key
        for action in primary
    )


def validate_adaptive_flow_bindings(
    plan: AdmissionControlPlan,
    bindings: tuple[AdaptiveFlowBinding, ...],
) -> tuple[AdaptiveFlowBinding, ...]:
    """Require explicit exact runtime bindings for every selectable MoveSet."""

    plan = _instance(plan, AdmissionControlPlan, "adaptive binding plan")
    if not isinstance(bindings, tuple) or any(
        not isinstance(binding, AdaptiveFlowBinding) for binding in bindings
    ):
        raise TypeError("adaptive flow bindings must be AdaptiveFlowBinding tuple")
    prepared = prepare_admission_plan(plan)
    action_by_key = prepared.action_by_key
    move_sets = prepared.move_sets
    for binding in bindings:
        actions = list(_approved_actions(prepared, binding.work))
        for action in tuple(actions):
            if action.fallback_key is not None:
                fallback = action_by_key[action.fallback_key]
                if fallback not in actions:
                    actions.append(fallback)
        required_move_sets = {
            action.move_set
            for action in actions
            if action.move_set is not None and move_sets[action.move_set].moves
        }
        configured_move_sets = {transfer.move_set for transfer in binding.transfers}
        if configured_move_sets != required_move_sets:
            raise ValueError(
                "adaptive flow transfers must exactly cover selectable non-empty MoveSets"
            )
        for move_set_key in required_move_sets:
            move_set = move_sets[move_set_key]
            routes = {
                (scheduled.producer, scheduled.consumer, scheduled.move.value)
                for scheduled in move_set.moves
            }
            if None in {
                endpoint
                for scheduled in move_set.moves
                for endpoint in (scheduled.producer, scheduled.consumer)
            }:
                raise ValueError(
                    "adaptive production MoveSet requires producer/consumer bindings"
                )
            if len(routes) != 1:
                raise ValueError(
                    "adaptive production MoveSet currently supports one transfer route"
                )
            transfer = binding.transfer_for(move_set_key)
            assert transfer is not None
            if transfer.route_key != next(iter(routes)):
                raise ValueError(
                    "adaptive transfer binding differs from its frozen MoveSet route"
                )
    return tuple(bindings)


class AdaptiveAdmissionScheduler:
    """Admit compiler-approved actions from one prepared immutable plan."""

    __slots__ = (
        "mode",
        "prepared",
        "state",
        "latency_ewma",
        "interference_ewma",
        "interference_limit",
        "_last_decisions",
    )

    def __init__(
        self,
        plan: AdmissionControlPlan | None = None,
        *,
        mode: AdaptiveMode | AdmissionMode | str | None = None,
        alpha_num: int = 1,
        alpha_den: int = 2,
        interference_limit: int | Mapping[InterferenceBudgetKey, int] | None = None,
    ) -> None:
        if mode is None:
            mode = plan.mode if plan is not None else AdmissionMode.ADAPTIVE
        mode_value = mode.value if isinstance(mode, (AdaptiveMode, AdmissionMode)) else mode
        try:
            self.mode = AdaptiveMode(mode_value)
        except ValueError as exc:
            raise ValueError("phase-5 mode must be static or adaptive") from exc
        self.prepared: PreparedAdmissionPlan | None = None
        self.state = RuntimeAdmissionState(
            latency_ewma=IntegerEWMA(alpha_num, alpha_den),
            interference_ewma=IntegerEWMA(alpha_num, alpha_den),
        )
        self.latency_ewma = self.state.latency_ewma
        self.interference_ewma = self.state.interference_ewma
        self.interference_limit = interference_limit
        self._validate_interference_limit(interference_limit)
        self._last_decisions: tuple[AdaptiveDecision, ...] = ()
        if plan is not None:
            self.prepare(plan)

    @staticmethod
    def _validate_interference_limit(value: int | Mapping[InterferenceBudgetKey, int] | None) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, Mapping)):
            raise TypeError("interference_limit must be int, mapping, or None")
        if isinstance(value, int):
            _integer(value, "interference_limit")
            return
        if any(not isinstance(key, InterferenceBudgetKey) for key in value):
            raise TypeError("interference_limit keys must be InterferenceBudgetKey")
        for item in value.values():
            _integer(item, "interference_limit value")

    def prepare(self, plan: AdmissionControlPlan) -> PreparedAdmissionPlan:
        self.prepared = prepare_admission_plan(plan)
        self.state.reset()
        self.latency_ewma.clear()
        self.interference_ewma.clear()
        self.interference_ewma.configure_profiles(plan.profiles)
        self._last_decisions = ()
        return self.prepared

    def observe(self, key: _RuntimeKey, observation: int, *, interference: int | None = None) -> int | tuple[int, int]:
        """Update integer latency and optional interference observations."""

        latency = self.latency_ewma.update(key, observation)
        if interference is None:
            return latency
        return latency, self.interference_ewma.update(key, interference)

    @property
    def ewma(self) -> Mapping[_RuntimeKey, int]:
        return self.latency_ewma.snapshot()

    @property
    def interference(self) -> Mapping[_RuntimeKey, int]:
        return self.interference_ewma.snapshot()

    def observe_interference(self, key: _RuntimeKey, observation: int) -> int:
        return self.interference_ewma.update(key, observation)

    def decide(
        self,
        item: WorkItemDesc,
        *,
        context: AdmissionContext | None = None,
        action: CandidateActionDesc | CandidateActionKey | None = None,
        reserve: bool = False,
        allow_defer: bool = False,
    ) -> AdaptiveDecision:
        prepared = self._require_prepared()
        item = _instance(item, WorkItemDesc, "phase-5 work item")
        facts = context if context is not None else AdmissionContext()
        facts = _instance(facts, AdmissionContext, "admission context")
        must_complete = _must_complete(prepared, item)
        if action is None:
            actions = _approved_actions(prepared, item)
        else:
            if isinstance(action, CandidateActionDesc):
                candidate = action
            else:
                action_key = _instance(action, CandidateActionKey, "candidate action key")
                candidate = prepared.action_by_key.get(action_key)
                if candidate is None:
                    raise ValueError("candidate action is not in the prepared plan")
            canonical = prepared.action_by_key.get(candidate.key)
            if canonical is None:
                raise ValueError("candidate action is not in the prepared plan")
            if candidate != canonical:
                raise ValueError("candidate action differs from the frozen prepared action")
            if not _action_is_approved(prepared, item, canonical):
                raise ValueError("candidate action is not approved for this work item")
            actions = (canonical,)

        last_reason = "admission rejected"
        approved_shed: CandidateActionDesc | None = None
        fallbacks: list[CandidateActionDesc] = []
        for candidate in actions:
            if candidate.kind == CandidateActionKind.SHED:
                if approved_shed is None:
                    approved_shed = candidate
                continue
            reason = self._admission_reason(candidate, item, facts)
            if reason is None:
                if reserve:
                    self._reserve(candidate)
                if candidate.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK:
                    fallback_reason = (
                        "unknown bucket"
                        if candidate.unknown_bucket
                        else "authoritative fallback"
                    )
                    return self._emit(
                        item.key,
                        AdaptiveDecisionKind.FALLBACK,
                        fallback_reason,
                        candidate,
                    )
                return self._emit(
                    item.key,
                    AdaptiveDecisionKind.RUN,
                    "required work" if must_complete else "admitted",
                    candidate,
                )
            last_reason = reason
            fallback = self._fallback_for(candidate)
            if fallback is not None and fallback not in fallbacks:
                fallbacks.append(fallback)

        if must_complete:
            for fallback in fallbacks:
                reason = self._admission_reason(fallback, item, facts)
                if reason is not None:
                    last_reason = reason
                    continue
                if reserve:
                    self._reserve(fallback)
                return self._emit(
                    item.key,
                    AdaptiveDecisionKind.FALLBACK,
                    "required fallback",
                    fallback,
                )

        if allow_defer:
            return self._emit(
                item.key,
                AdaptiveDecisionKind.KEEP,
                last_reason,
                actions[-1] if actions else None,
            )
        if not must_complete and approved_shed is not None:
            return self._emit(
                item.key,
                AdaptiveDecisionKind.DROP,
                last_reason,
                approved_shed,
            )

        if not must_complete:
            for fallback in fallbacks:
                reason = self._admission_reason(fallback, item, facts)
                if reason is not None:
                    last_reason = reason
                    continue
                if reserve:
                    self._reserve(fallback)
                return self._emit(
                    item.key,
                    AdaptiveDecisionKind.FALLBACK,
                    "shedding not approved",
                    fallback,
                )

        raise RuntimeError(
            f"no feasible compiler-approved action for {item.key}: {last_reason}"
        )

    def decide_all(
        self,
        items: Sequence[WorkItemDesc],
        *,
        context: AdmissionContext | None = None,
        reserve: bool = True,
    ) -> tuple[AdaptiveDecision, ...]:
        typed = tuple(_instance(item, WorkItemDesc, "phase-5 work item") for item in items)
        if len({item.key for item in typed}) != len(typed):
            raise ValueError("admission work item keys must be unique")
        prepared = self._require_prepared()
        ordered = tuple(
            sorted(
                typed,
                key=lambda item: (not _must_complete(prepared, item), item.key),
            )
        )
        self.state.reset()
        decisions = tuple(self.decide(item, context=context, reserve=reserve) for item in ordered)
        self._last_decisions = decisions
        return decisions

    def release(self, *, buffers: int = 0, budget: Mapping[InterferenceBudgetKey, int] | None = None) -> None:
        buffers = _integer(buffers, "released buffers")
        if buffers > self.state.reserved_buffers:
            raise ValueError("released buffers exceed reservation")
        self.state.reserved_buffers -= buffers
        if budget is None:
            return
        for key, amount in budget.items():
            _instance(key, InterferenceBudgetKey, "released budget key")
            amount = _integer(amount, "released budget amount")
            current = self.state.reserved_budgets.get(key, 0)
            if amount > current:
                raise ValueError("released budget exceeds reservation")
            if amount == current:
                self.state.reserved_budgets.pop(key, None)
            else:
                self.state.reserved_budgets[key] = current - amount

    def reset_reservations(self) -> None:
        self.state.reset()

    def _emit(
        self,
        work_key: WorkItemKey,
        action: AdaptiveDecisionKind,
        reason: str,
        candidate: CandidateActionDesc | None,
    ) -> AdaptiveDecision:
        decision = AdaptiveDecision(
            work_key=work_key,
            action=action,
            reason=reason,
            plan_epoch=self._require_prepared().plan_epoch,
            action_key=None if candidate is None else candidate.key,
            candidate=candidate,
        )
        self._last_decisions = (decision,)
        return decision

    def _fallback_for(self, action: CandidateActionDesc) -> CandidateActionDesc | None:
        if action.fallback_key is None:
            return None
        return self._require_prepared().action_by_key.get(action.fallback_key)

    def _duration(self, action: CandidateActionDesc) -> int:
        duration = action.duration
        move_set = self._require_prepared().move_sets.get(action.move_set) if action.move_set is not None else None
        if move_set is not None and move_set.moves:
            duration = max(
                duration,
                max(move.end_at - action.ready_at for move in move_set.moves),
            )
        if self.mode == AdaptiveMode.ADAPTIVE:
            estimate = self.latency_ewma.get(action.key)
            if estimate is not None:
                duration = max(duration, estimate)
        return duration

    def _admission_reason(self, action: CandidateActionDesc, item: WorkItemDesc, facts: AdmissionContext) -> str | None:
        prepared = self._require_prepared()
        available = facts.available_buffers
        if available is None:
            capacity = prepared.buffer_capacity if facts.buffer_capacity is None else facts.buffer_capacity
            available = max(0, capacity - facts.buffer_used)
        if available - self.state.reserved_buffers < action.buffer_credits:
            return "buffer admission"

        duration = self._duration(action)
        end = action.ready_at + duration
        if end > action.deadline:
            return "deadline admission"
        if item.deadline is not None and action.deadline > item.deadline:
            return "deadline admission"
        if facts.deadline_slack is not None and duration > facts.deadline_slack:
            return "deadline admission"
        if facts.deadline_step is not None and facts.step + duration > facts.deadline_step:
            return "deadline admission"

        for use in action.budget_uses:
            budget = prepared.budgets.get(use.key)
            if budget is None:
                return "interference admission"
            observed = facts.interference.get(use.key)
            if observed is None and self.mode == AdaptiveMode.ADAPTIVE and budget.profile is not None:
                observed = self.interference_ewma.get(budget.profile)
            if observed is None and self.mode == AdaptiveMode.ADAPTIVE:
                observed = self.interference_ewma.get(use.key, 0) or 0
            if observed is None:
                observed = 0
            limit = facts.interference_limit
            limit_value = limit if isinstance(limit, int) else None if limit is None else limit.get(use.key)
            if limit_value is None:
                configured = self.interference_limit
                limit_value = configured if isinstance(configured, int) else None if configured is None else configured.get(use.key)
            if limit_value is not None and observed > limit_value:
                return "interference admission"
            reserved = self.state.reserved_budgets.get(use.key, 0)
            if observed + reserved + use.amount > budget.capacity:
                return "interference admission"
        return None

    def _reserve(self, action: CandidateActionDesc) -> None:
        self.state.reserved_buffers += action.buffer_credits
        for use in action.budget_uses:
            self.state.reserved_budgets[use.key] = self.state.reserved_budgets.get(use.key, 0) + use.amount

    def _require_prepared(self) -> PreparedAdmissionPlan:
        if self.prepared is None:
            raise RuntimeError("phase-5 scheduler has no prepared plan")
        return self.prepared


__all__ = [
    "AdaptiveFlowBinding",
    "AdaptiveRuntimeBinding",
    "AdaptiveTransferBinding",
    "AdaptiveTransferStatus",
    "AdaptiveDecision",
    "AdaptiveDecisionKind",
    "AdaptiveMode",
    "AdmissionContext",
    "IntegerEWMA",
    "AdaptiveAdmissionScheduler",
    "PreparedAdmissionPlan",
    "RuntimeAdmissionState",
    "prepare_admission_plan",
    "validate_adaptive_flow_bindings",
]
