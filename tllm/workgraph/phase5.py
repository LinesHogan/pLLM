"""Pure adaptive admission contracts for WorkGraph Phase 5.

Phase 5 is a control-plane boundary.  The descriptors in this module contain
only stable keys, integers, tuples, and Phase 3's frozen ``ScheduledMove``
values.  Runtime tensors, timers, queues, and callbacks deliberately do not
enter the plan.  Adaptive code can select one of the compiler-approved
actions, but cannot change an action's semantic identity, deadline, buffer
credit, or move set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import TypeVar, cast

from tllm.workgraph.placement import MoveKey, PlacementPlan, ScheduledMove


_T = TypeVar("_T")
_EnumT = TypeVar("_EnumT", bound=StrEnum)
_DEFAULT_DENSIFICATION_LIMIT = 64


def _placement_signature(plan: PlacementPlan) -> str:
    # Imported lazily because signature.py has a closed-world Phase 5 branch.
    from tllm.workgraph.signature import structural_signature

    return structural_signature(plan)


def _instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


def _enum(value: object, expected: type[_EnumT], label: str) -> _EnumT:
    if isinstance(value, expected):
        return value
    if isinstance(value, str):
        try:
            return expected(value)
        except ValueError as exc:
            raise ValueError(f"{label} is unsupported") from exc
    raise TypeError(f"{label} must be {expected.__name__}")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if not value or "0x" in value.lower() or "<" in value or ">" in value:
        raise ValueError(f"{label} must be a stable non-empty string")
    return value


def _key_segment(
    value: object,
    label: str,
    *,
    reserved: tuple[str, ...] = (":", "@", "/"),
) -> str:
    text = _text(value, label)
    if any(delimiter in text for delimiter in reserved):
        raise ValueError(f"{label} contains a reserved key delimiter")
    return text


def _tuple_of(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return cast(tuple[_T, ...], items)


def _integers(value: object, label: str, *, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple[int, ...]")
    return tuple(
        _integer(item, f"{label} item", minimum=minimum)
        for item in cast(tuple[object, ...], value)
    )


@dataclass(frozen=True, slots=True, order=True)
class EWMAProfileKey:
    """Stable identity for one integer EWMA profile."""

    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace",
            _key_segment(self.namespace, "EWMAProfileKey.namespace"),
        )
        object.__setattr__(
            self,
            "name",
            _key_segment(self.name, "EWMAProfileKey.name"),
        )
        object.__setattr__(
            self,
            "version",
            _integer(self.version, "EWMAProfileKey.version", minimum=1),
        )

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"


@dataclass(frozen=True, slots=True, order=True)
class FixedRatio:
    """A positive rational number represented without floating point."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        numerator = _integer(self.numerator, "FixedRatio.numerator", minimum=1)
        denominator = _integer(
            self.denominator,
            "FixedRatio.denominator",
            minimum=1,
        )
        if numerator > denominator:
            raise ValueError("FixedRatio.numerator must not exceed denominator")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)


@dataclass(frozen=True, slots=True)
class IntegerEWMAProfile:
    """An immutable integer EWMA with a compiler-fixed update ratio."""

    key: EWMAProfileKey
    estimate: int
    alpha: FixedRatio = FixedRatio(1, 2)
    samples: int = 0
    minimum: int = 0
    maximum: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _instance(self.key, EWMAProfileKey, "IntegerEWMAProfile.key"))
        estimate = _integer(self.estimate, "IntegerEWMAProfile.estimate")
        alpha = _instance(self.alpha, FixedRatio, "IntegerEWMAProfile.alpha")
        samples = _integer(self.samples, "IntegerEWMAProfile.samples")
        minimum = _integer(self.minimum, "IntegerEWMAProfile.minimum")
        maximum = self.maximum
        if maximum is not None:
            maximum = _integer(maximum, "IntegerEWMAProfile.maximum")
            if maximum < minimum:
                raise ValueError("IntegerEWMAProfile.maximum must be >= minimum")
        if estimate < minimum or (maximum is not None and estimate > maximum):
            raise ValueError("IntegerEWMAProfile.estimate is outside its bounds")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def value(self) -> int:
        """Compatibility spelling for callers that call an EWMA a value."""

        return self.estimate

    @property
    def alpha_num(self) -> int:
        return self.alpha.numerator

    @property
    def alpha_den(self) -> int:
        return self.alpha.denominator

    def observe(self, sample: int) -> IntegerEWMAProfile:
        sample = _integer(sample, "IntegerEWMAProfile.sample")
        if sample < self.minimum or (
            self.maximum is not None and sample > self.maximum
        ):
            raise ValueError("IntegerEWMAProfile.sample is outside its bounds")
        ratio = self.alpha
        weight_old = ratio.denominator - ratio.numerator
        weighted = weight_old * self.estimate + ratio.numerator * sample
        estimate = (weighted + ratio.denominator // 2) // ratio.denominator
        estimate = max(self.minimum, estimate)
        if self.maximum is not None:
            estimate = min(self.maximum, estimate)
        return IntegerEWMAProfile(
            key=self.key,
            estimate=estimate,
            alpha=ratio,
            samples=self.samples + 1,
            minimum=self.minimum,
            maximum=self.maximum,
        )

    update = observe


@dataclass(frozen=True, slots=True, order=True)
class WorkloadBucketKey:
    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _key_segment(self.namespace, "WorkloadBucketKey.namespace"))
        object.__setattr__(self, "name", _key_segment(self.name, "WorkloadBucketKey.name"))
        object.__setattr__(self, "version", _integer(self.version, "WorkloadBucketKey.version", minimum=1))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"


@dataclass(frozen=True, slots=True, order=True)
class ShapeBucketKey:
    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _key_segment(self.namespace, "ShapeBucketKey.namespace"))
        object.__setattr__(self, "name", _key_segment(self.name, "ShapeBucketKey.name"))
        object.__setattr__(self, "version", _integer(self.version, "ShapeBucketKey.version", minimum=1))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class ShapeBucketDesc:
    """A static shape bucket, either exact or bounded by integer dimensions."""

    key: ShapeBucketKey
    shape: tuple[int, ...] = ()
    lower_bounds: tuple[int, ...] = ()
    upper_bounds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        key = _instance(self.key, ShapeBucketKey, "ShapeBucketDesc.key")
        shape = _integers(self.shape, "ShapeBucketDesc.shape", minimum=1)
        lower = _integers(self.lower_bounds, "ShapeBucketDesc.lower_bounds", minimum=1)
        upper = _integers(self.upper_bounds, "ShapeBucketDesc.upper_bounds", minimum=1)
        if shape and (lower or upper):
            raise ValueError("ShapeBucketDesc.shape cannot combine with bounds")
        if not shape and (not lower or not upper):
            raise ValueError("ShapeBucketDesc requires shape or both bounds")
        if lower and (len(lower) != len(upper) or any(a > b for a, b in zip(lower, upper))):
            raise ValueError("ShapeBucketDesc bounds must have equal ordered dimensions")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)

    def matches(self, shape: tuple[int, ...]) -> bool:
        shape = _integers(shape, "ShapeBucketDesc.match shape", minimum=1)
        if self.shape:
            return shape == self.shape
        return len(shape) == len(self.lower_bounds) and all(
            lower <= value <= upper
            for value, lower, upper in zip(shape, self.lower_bounds, self.upper_bounds)
        )


@dataclass(frozen=True, slots=True)
class WorkloadBucketDesc:
    key: WorkloadBucketKey
    shape_buckets: tuple[ShapeBucketKey, ...] = ()
    required_by_default: bool = False

    def __post_init__(self) -> None:
        key = _instance(self.key, WorkloadBucketKey, "WorkloadBucketDesc.key")
        shapes = _tuple_of(self.shape_buckets, ShapeBucketKey, "WorkloadBucketDesc.shape_buckets")
        if len(set(shapes)) != len(shapes):
            raise ValueError("WorkloadBucketDesc.shape_buckets must be unique")
        if not isinstance(cast(object, self.required_by_default), bool):
            raise TypeError("WorkloadBucketDesc.required_by_default must be bool")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "shape_buckets", tuple(sorted(shapes)))


@dataclass(frozen=True, slots=True, order=True)
class InterferenceBudgetKey:
    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _key_segment(self.namespace, "InterferenceBudgetKey.namespace"))
        object.__setattr__(self, "name", _key_segment(self.name, "InterferenceBudgetKey.name"))
        object.__setattr__(self, "version", _integer(self.version, "InterferenceBudgetKey.version", minimum=1))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class InterferenceBudgetDesc:
    key: InterferenceBudgetKey
    capacity: int
    profile: EWMAProfileKey | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _instance(self.key, InterferenceBudgetKey, "InterferenceBudgetDesc.key"))
        object.__setattr__(self, "capacity", _integer(self.capacity, "InterferenceBudgetDesc.capacity"))
        if self.profile is not None:
            object.__setattr__(self, "profile", _instance(self.profile, EWMAProfileKey, "InterferenceBudgetDesc.profile"))


@dataclass(frozen=True, slots=True)
class BudgetUseDesc:
    key: InterferenceBudgetKey
    amount: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _instance(self.key, InterferenceBudgetKey, "BudgetUseDesc.key"))
        object.__setattr__(self, "amount", _integer(self.amount, "BudgetUseDesc.amount", minimum=1))


@dataclass(frozen=True, slots=True, order=True)
class MoveSetKey:
    namespace: str
    name: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _key_segment(self.namespace, "MoveSetKey.namespace"))
        object.__setattr__(self, "name", _key_segment(self.name, "MoveSetKey.name"))
        object.__setattr__(self, "ordinal", _integer(self.ordinal, "MoveSetKey.ordinal"))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}/{self.ordinal}"


@dataclass(frozen=True, slots=True)
class CandidateMoveSetDesc:
    """A dependency-closed subset of one Phase 3 scheduled-move universe."""

    key: MoveSetKey
    moves: tuple[ScheduledMove, ...] = ()

    def __post_init__(self) -> None:
        key = _instance(self.key, MoveSetKey, "CandidateMoveSetDesc.key")
        moves = _tuple_of(self.moves, ScheduledMove, "CandidateMoveSetDesc.moves")
        keys = tuple(item.key for item in moves)
        if len(set(keys)) != len(keys):
            raise ValueError("CandidateMoveSetDesc moves must have unique keys")
        key_set = set(keys)
        if any(dependency not in key_set for item in moves for dependency in item.depends_on):
            raise ValueError("CandidateMoveSetDesc moves must be dependency-closed")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "moves", tuple(sorted(moves, key=lambda item: item.key)))

    @property
    def move_keys(self) -> tuple[MoveKey, ...]:
        return tuple(item.key for item in self.moves)


@dataclass(frozen=True, slots=True, order=True)
class CandidateActionKey:
    namespace: str
    name: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _key_segment(self.namespace, "CandidateActionKey.namespace"))
        object.__setattr__(self, "name", _key_segment(self.name, "CandidateActionKey.name"))
        object.__setattr__(self, "ordinal", _integer(self.ordinal, "CandidateActionKey.ordinal"))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}/{self.ordinal}"


class CandidateActionKind(StrEnum):
    EXECUTE = "execute"
    AUTHORITATIVE_FALLBACK = "authoritative_fallback"
    SHED = "shed"


class Phase5CompilationError(RuntimeError):
    """Raised when Phase 5 cannot anchor to a valid Phase 3 placement plan."""


class AdmissionMode(StrEnum):
    STATIC = "static"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True, slots=True)
class CandidateActionDesc:
    """One compiler-approved action for a workload/shape bucket."""

    key: CandidateActionKey
    kind: CandidateActionKind
    semantic_key: str = "default"
    workload_bucket: WorkloadBucketKey | None = None
    shape_bucket: ShapeBucketKey | None = None
    move_set: MoveSetKey | None = None
    fallback_key: CandidateActionKey | None = None
    deadline: int = 1
    ready_at: int = 0
    duration: int = 1
    buffer_credits: int = 0
    budget_uses: tuple[BudgetUseDesc, ...] = ()
    priority: int = 0
    utility: int = 0
    required: bool = False
    unknown_bucket: bool = False
    authoritative: bool = False

    def __post_init__(self) -> None:
        key = _instance(self.key, CandidateActionKey, "CandidateActionDesc.key")
        kind = _enum(cast(object, self.kind), CandidateActionKind, "CandidateActionDesc.kind")
        semantic = _key_segment(self.semantic_key, "CandidateActionDesc.semantic_key")
        workload = self.workload_bucket
        shape = self.shape_bucket
        move_set = self.move_set
        fallback = self.fallback_key
        if workload is not None:
            workload = _instance(workload, WorkloadBucketKey, "CandidateActionDesc.workload_bucket")
        if shape is not None:
            shape = _instance(shape, ShapeBucketKey, "CandidateActionDesc.shape_bucket")
        if move_set is not None:
            move_set = _instance(move_set, MoveSetKey, "CandidateActionDesc.move_set")
        if fallback is not None:
            fallback = _instance(fallback, CandidateActionKey, "CandidateActionDesc.fallback_key")
        deadline = _integer(self.deadline, "CandidateActionDesc.deadline", minimum=1)
        ready_at = _integer(self.ready_at, "CandidateActionDesc.ready_at")
        duration = _integer(self.duration, "CandidateActionDesc.duration")
        credits = _integer(self.buffer_credits, "CandidateActionDesc.buffer_credits")
        priority = _integer(self.priority, "CandidateActionDesc.priority")
        utility = _integer(self.utility, "CandidateActionDesc.utility")
        uses = _tuple_of(self.budget_uses, BudgetUseDesc, "CandidateActionDesc.budget_uses")
        if len({item.key for item in uses}) != len(uses):
            raise ValueError("CandidateActionDesc budget uses must have unique keys")
        if not isinstance(cast(object, self.required), bool):
            raise TypeError("CandidateActionDesc.required must be bool")
        if not isinstance(cast(object, self.unknown_bucket), bool):
            raise TypeError("CandidateActionDesc.unknown_bucket must be bool")
        if not isinstance(cast(object, self.authoritative), bool):
            raise TypeError("CandidateActionDesc.authoritative must be bool")
        if kind == CandidateActionKind.SHED:
            if self.required:
                raise ValueError("required work cannot use a shedding action")
            if move_set is not None or duration != 0 or credits != 0 or uses:
                raise ValueError("shedding action cannot reserve work or resources")
            authoritative = False
        elif duration < 1:
            raise ValueError("execute/fallback actions require positive duration")
        else:
            authoritative = self.authoritative
        if kind == CandidateActionKind.AUTHORITATIVE_FALLBACK:
            authoritative = True
        elif authoritative:
            raise ValueError("only authoritative fallback actions may be authoritative")
        if self.unknown_bucket and kind != CandidateActionKind.AUTHORITATIVE_FALLBACK:
            raise ValueError("unknown_bucket actions must be authoritative fallbacks")
        if self.unknown_bucket and (workload is not None or shape is not None):
            raise ValueError("unknown_bucket fallback must not target a known bucket")
        if kind == CandidateActionKind.AUTHORITATIVE_FALLBACK and fallback is not None:
            raise ValueError("an authoritative fallback cannot require another fallback")
        if fallback == key:
            raise ValueError("CandidateActionDesc fallback_key cannot be self")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "semantic_key", semantic)
        object.__setattr__(self, "workload_bucket", workload)
        object.__setattr__(self, "shape_bucket", shape)
        object.__setattr__(self, "move_set", move_set)
        object.__setattr__(self, "fallback_key", fallback)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "ready_at", ready_at)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "buffer_credits", credits)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "utility", utility)
        object.__setattr__(self, "budget_uses", tuple(sorted(uses, key=lambda item: item.key)))
        object.__setattr__(self, "authoritative", authoritative)


@dataclass(frozen=True, slots=True, order=True)
class WorkItemKey:
    namespace: str
    name: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _key_segment(self.namespace, "WorkItemKey.namespace"))
        object.__setattr__(self, "name", _key_segment(self.name, "WorkItemKey.name"))
        object.__setattr__(self, "ordinal", _integer(self.ordinal, "WorkItemKey.ordinal"))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}/{self.ordinal}"


@dataclass(frozen=True, slots=True)
class WorkItemDesc:
    key: WorkItemKey
    shape: tuple[int, ...]
    workload_bucket: WorkloadBucketKey | None = None
    semantic_key: str = "default"
    required: bool = False
    allow_shedding: bool = True
    deadline: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _instance(self.key, WorkItemKey, "WorkItemDesc.key"))
        shape = _integers(self.shape, "WorkItemDesc.shape", minimum=1)
        if not shape:
            raise ValueError("WorkItemDesc.shape must be non-empty")
        object.__setattr__(self, "shape", shape)
        if self.workload_bucket is not None:
            object.__setattr__(self, "workload_bucket", _instance(self.workload_bucket, WorkloadBucketKey, "WorkItemDesc.workload_bucket"))
        object.__setattr__(self, "semantic_key", _key_segment(self.semantic_key, "WorkItemDesc.semantic_key"))
        if not isinstance(cast(object, self.required), bool) or not isinstance(cast(object, self.allow_shedding), bool):
            raise TypeError("WorkItemDesc required and allow_shedding must be bool")
        if self.required and self.allow_shedding:
            object.__setattr__(self, "allow_shedding", False)
        if self.deadline is not None:
            object.__setattr__(self, "deadline", _integer(self.deadline, "WorkItemDesc.deadline", minimum=1))


@dataclass(frozen=True, slots=True)
class Phase5ControlPlan:
    """Complete immutable Phase 5 compiler output."""

    plan_epoch: int
    placement_plan: PlacementPlan
    scheduled_moves: tuple[ScheduledMove, ...]
    move_sets: tuple[CandidateMoveSetDesc, ...]
    workload_buckets: tuple[WorkloadBucketDesc, ...]
    shape_buckets: tuple[ShapeBucketDesc, ...]
    budgets: tuple[InterferenceBudgetDesc, ...]
    actions: tuple[CandidateActionDesc, ...]
    unknown_bucket_fallback: CandidateActionKey
    buffer_capacity: int = 0
    mode: AdmissionMode = AdmissionMode.STATIC
    schema_version: int = 1
    profiles: tuple[IntegerEWMAProfile, ...] = ()
    placement_signature: str = ""
    densification_limit: int = _DEFAULT_DENSIFICATION_LIMIT

    def __post_init__(self) -> None:
        moves = _tuple_of(self.scheduled_moves, ScheduledMove, "Phase5ControlPlan.scheduled_moves")
        move_keys = tuple(item.key for item in moves)
        if len(set(move_keys)) != len(move_keys):
            raise ValueError("Phase5ControlPlan scheduled move keys must be unique")
        move_sets = _tuple_of(self.move_sets, CandidateMoveSetDesc, "Phase5ControlPlan.move_sets")
        workload = _tuple_of(self.workload_buckets, WorkloadBucketDesc, "Phase5ControlPlan.workload_buckets")
        shapes = _tuple_of(self.shape_buckets, ShapeBucketDesc, "Phase5ControlPlan.shape_buckets")
        budgets = _tuple_of(self.budgets, InterferenceBudgetDesc, "Phase5ControlPlan.budgets")
        actions = _tuple_of(self.actions, CandidateActionDesc, "Phase5ControlPlan.actions")
        profiles = _tuple_of(self.profiles, IntegerEWMAProfile, "Phase5ControlPlan.profiles")
        epoch = _integer(self.plan_epoch, "Phase5ControlPlan.plan_epoch", minimum=1)
        placement_plan = _instance(
            self.placement_plan,
            PlacementPlan,
            "Phase5ControlPlan.placement_plan",
        )
        placement_signature = self.placement_signature
        if epoch != placement_plan.plan_epoch:
            raise ValueError("Phase5ControlPlan epoch differs from its PlacementPlan")
        if tuple(moves) != placement_plan.moves:
            raise ValueError("Phase5ControlPlan moves must come from its PlacementPlan")
        expected_signature = _placement_signature(placement_plan)
        if placement_signature and placement_signature != expected_signature:
            raise ValueError("Phase5ControlPlan placement signature differs from its PlacementPlan")
        placement_signature = expected_signature
        densification_limit = _integer(
            self.densification_limit,
            "Phase5ControlPlan.densification_limit",
            minimum=1,
        )
        if self.schema_version != 1:
            raise ValueError("Phase5ControlPlan.schema_version must be 1")
        if len({item.key for item in move_sets}) != len(move_sets):
            raise ValueError("Phase5ControlPlan move-set keys must be unique")
        if len(move_sets) > densification_limit or len(actions) > densification_limit:
            raise ValueError("Phase5ControlPlan exceeds its densification limit")
        move_by_key = {item.key: item for item in moves}
        for move_set in move_sets:
            for item in move_set.moves:
                anchored = move_by_key.get(item.key)
                if anchored is None:
                    raise ValueError("candidate Move set references a non-Phase-3 Move")
                if anchored != item:
                    raise ValueError("candidate Move set rewrites a Phase 3 Move")
            selected = set(move_set.move_keys)
            if any(dep not in selected for item in move_set.moves for dep in item.depends_on):
                raise ValueError("candidate Move sets must be dependency-closed")
        workload_keys = {item.key for item in workload}
        shape_keys = {item.key for item in shapes}
        budget_keys = {item.key for item in budgets}
        profile_keys = {item.key for item in profiles}
        move_set_keys = {item.key for item in move_sets}
        if (
            len(workload_keys) != len(workload)
            or len(shape_keys) != len(shapes)
            or len(budget_keys) != len(budgets)
            or len(profile_keys) != len(profiles)
        ):
            raise ValueError("Phase5ControlPlan descriptor keys must be unique")
        if any(budget.profile is not None and budget.profile not in profile_keys for budget in budgets):
            raise ValueError("interference budget references an unknown EWMA profile")
        shape_by_key = {item.key: item for item in shapes}
        for bucket in workload:
            if any(shape_key not in shape_keys for shape_key in bucket.shape_buckets):
                raise ValueError("workload bucket references an unknown shape bucket")
        action_keys = {item.key for item in actions}
        if len(action_keys) != len(actions):
            raise ValueError("Phase5ControlPlan action keys must be unique")
        fallback = _instance(self.unknown_bucket_fallback, CandidateActionKey, "Phase5ControlPlan.unknown_bucket_fallback")
        if fallback not in action_keys:
            raise ValueError("unknown_bucket_fallback is not an approved action")
        fallback_action = next(item for item in actions if item.key == fallback)
        if fallback_action.kind != CandidateActionKind.AUTHORITATIVE_FALLBACK or not fallback_action.unknown_bucket:
            raise ValueError("unknown_bucket_fallback must be an unknown-bucket authoritative fallback")
        capacity = _integer(self.buffer_capacity, "Phase5ControlPlan.buffer_capacity")
        object.__setattr__(self, "mode", _enum(cast(object, self.mode), AdmissionMode, "Phase5ControlPlan.mode"))
        for action in actions:
            if action.workload_bucket is not None and action.workload_bucket not in workload_keys:
                raise ValueError("action references an unknown workload bucket")
            if action.shape_bucket is not None and action.shape_bucket not in shape_keys:
                raise ValueError("action references an unknown shape bucket")
            if action.move_set is not None and action.move_set not in move_set_keys:
                raise ValueError("action references an unknown Move set")
            if any(use.key not in budget_keys for use in action.budget_uses):
                raise ValueError("action references an unknown interference budget")
            if action.buffer_credits > capacity:
                raise ValueError("action buffer credits exceed compiler-approved capacity")
            if action.shape_bucket is not None and action.workload_bucket is not None:
                workload_desc = next(item for item in workload if item.key == action.workload_bucket)
                if workload_desc.shape_buckets and action.shape_bucket not in workload_desc.shape_buckets:
                    raise ValueError("action shape bucket is not approved for its workload bucket")
            if action.fallback_key is not None:
                if action.fallback_key not in action_keys:
                    raise ValueError("action fallback references an unknown action")
                fallback_candidate = next(item for item in actions if item.key == action.fallback_key)
                if fallback_candidate.kind != CandidateActionKind.AUTHORITATIVE_FALLBACK:
                    raise ValueError("required action fallback must be authoritative")
                if fallback_candidate.semantic_key != action.semantic_key:
                    raise ValueError("fallback changes action semantics")
            if (
                action.kind == CandidateActionKind.EXECUTE
                and action.fallback_key is None
            ):
                raise ValueError(
                    "every execute action requires an authoritative fallback"
                )
        object.__setattr__(self, "scheduled_moves", tuple(sorted(moves, key=lambda item: item.key)))
        object.__setattr__(self, "move_sets", tuple(sorted(move_sets, key=lambda item: item.key)))
        object.__setattr__(self, "workload_buckets", tuple(sorted(workload, key=lambda item: item.key)))
        object.__setattr__(self, "shape_buckets", tuple(sorted(shapes, key=lambda item: item.key)))
        object.__setattr__(self, "budgets", tuple(sorted(budgets, key=lambda item: item.key)))
        object.__setattr__(self, "actions", tuple(sorted(actions, key=lambda item: item.key)))
        object.__setattr__(self, "profiles", tuple(sorted(profiles, key=lambda item: item.key)))
        object.__setattr__(self, "unknown_bucket_fallback", fallback)
        object.__setattr__(self, "buffer_capacity", capacity)
        object.__setattr__(self, "shape_buckets", tuple(shape_by_key[key] for key in sorted(shape_by_key)))
        object.__setattr__(self, "plan_epoch", epoch)
        object.__setattr__(self, "placement_plan", placement_plan)
        object.__setattr__(self, "placement_signature", placement_signature)
        object.__setattr__(self, "densification_limit", densification_limit)


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    selected: tuple[tuple[WorkItemKey, CandidateActionKey], ...]
    shed: tuple[WorkItemKey, ...]
    objective: tuple[int, ...]
    feasible: bool = True

    def __post_init__(self) -> None:
        selected_raw = cast(object, self.selected)
        if not isinstance(selected_raw, tuple):
            raise TypeError("AdmissionResult.selected must be tuple")
        selected_items: list[tuple[WorkItemKey, CandidateActionKey]] = []
        for raw_item in cast(tuple[object, ...], selected_raw):
            if not isinstance(raw_item, tuple):
                raise TypeError("AdmissionResult.selected entries must be pairs")
            pair = cast(tuple[object, ...], raw_item)
            if len(pair) != 2:
                raise TypeError("AdmissionResult.selected entries must be pairs")
            work_key = _instance(pair[0], WorkItemKey, "AdmissionResult.selected work key")
            action_key = _instance(pair[1], CandidateActionKey, "AdmissionResult.selected action key")
            selected_items.append((work_key, action_key))
        object.__setattr__(self, "selected", tuple(sorted(selected_items, key=lambda item: item[0])))
        object.__setattr__(self, "shed", tuple(sorted(_tuple_of(self.shed, WorkItemKey, "AdmissionResult.shed"))))
        if not isinstance(cast(object, self.feasible), bool):
            raise TypeError("AdmissionResult.feasible must be bool")


def validate_candidate_move_sets(
    scheduled_moves: tuple[ScheduledMove, ...],
    move_sets: tuple[CandidateMoveSetDesc, ...],
) -> tuple[CandidateMoveSetDesc, ...]:
    """Validate and return dependency-closed candidate Move sets."""

    scheduled = _tuple_of(scheduled_moves, ScheduledMove, "validate_candidate_move_sets.scheduled_moves")
    candidates = _tuple_of(move_sets, CandidateMoveSetDesc, "validate_candidate_move_sets.move_sets")
    universe = {item.key: item for item in scheduled}
    for candidate in candidates:
        for item in candidate.moves:
            anchored = universe.get(item.key)
            if anchored is None:
                raise ValueError("candidate Move set is not a subset of scheduled Phase 3 Moves")
            if anchored != item:
                raise ValueError("candidate Move set rewrites a scheduled Phase 3 Move")
        selected = set(candidate.move_keys)
        if any(dep not in selected for item in candidate.moves for dep in item.depends_on):
            raise ValueError("candidate Move set is not dependency-closed")
    return tuple(sorted(candidates, key=lambda item: item.key))


def _shape_bucket(plan: Phase5ControlPlan, shape: tuple[int, ...]) -> ShapeBucketKey | None:
    matching = [bucket.key for bucket in plan.shape_buckets if bucket.matches(shape)]
    return min(matching) if matching else None


def _must_complete(plan: Phase5ControlPlan, item: WorkItemDesc) -> bool:
    workload_required = any(
        bucket.key == item.workload_bucket and bucket.required_by_default
        for bucket in plan.workload_buckets
    )
    shape_key = _shape_bucket(plan, item.shape)
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
                and action.key == plan.unknown_bucket_fallback
            )
        )
        for action in plan.actions
    )
    return (
        item.required
        or workload_required
        or action_required
        or not item.allow_shedding
    )


def approved_actions(plan: Phase5ControlPlan, item: WorkItemDesc) -> tuple[CandidateActionDesc, ...]:
    """Resolve a request to compiler-approved actions, failing closed on misses."""

    plan = _instance(plan, Phase5ControlPlan, "approved_actions.plan")
    item = _instance(item, WorkItemDesc, "approved_actions.item")
    shape_key = _shape_bucket(plan, item.shape)
    known_workload = item.workload_bucket in {bucket.key for bucket in plan.workload_buckets}
    must_complete = _must_complete(plan, item)
    fallback = next(
        action
        for action in plan.actions
        if action.key == plan.unknown_bucket_fallback
    )
    if item.workload_bucket is None or not known_workload or shape_key is None:
        if fallback.semantic_key != item.semantic_key:
            raise RuntimeError(
                "unknown bucket has no semantic-preserving authoritative fallback"
            )
        return (fallback,)
    candidates = tuple(
        action
        for action in plan.actions
        if not action.unknown_bucket
        and action.workload_bucket == item.workload_bucket
        and (action.shape_bucket is None or action.shape_bucket == shape_key)
        and action.semantic_key == item.semantic_key
        and (not must_complete or action.kind != CandidateActionKind.SHED)
    )
    if must_complete:
        candidates = tuple(
            action
            for action in candidates
            if action.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK
            or action.fallback_key is not None
        )
    if not candidates:
        if fallback.semantic_key != item.semantic_key:
            raise RuntimeError(
                "work item has no semantic-preserving authoritative fallback"
            )
        return (fallback,)
    approved = list(candidates)
    candidate_keys = {action.key for action in approved}
    for action in candidates:
        if action.fallback_key is None or action.fallback_key in candidate_keys:
            continue
        fallback = next(candidate for candidate in plan.actions if candidate.key == action.fallback_key)
        if fallback.semantic_key == item.semantic_key:
            approved.append(fallback)
            candidate_keys.add(fallback.key)
    return tuple(sorted(approved, key=lambda action: (-action.priority, action.key)))


def _can_shed(plan: Phase5ControlPlan, item: WorkItemDesc) -> bool:
    if _must_complete(plan, item):
        return False
    return any(
        action.kind == CandidateActionKind.SHED
        for action in approved_actions(plan, item)
    )


@dataclass(frozen=True, slots=True)
class _EvaluatedSelection:
    selected: tuple[tuple[WorkItemDesc, CandidateActionDesc], ...]
    shed: tuple[WorkItemDesc, ...]
    objective: tuple[int, ...]


def _evaluate_selection(
    plan: Phase5ControlPlan,
    selection: tuple[tuple[WorkItemDesc, CandidateActionDesc], ...],
    *,
    items: tuple[WorkItemDesc, ...],
) -> _EvaluatedSelection | None:
    if len(selection) > plan.densification_limit:
        return None
    selected_keys = {item.key for item, _ in selection}
    if len(selected_keys) != len(selection):
        return None
    budget_totals = {budget.key: 0 for budget in plan.budgets}
    credits = 0
    move_keys: set[MoveKey] = set()
    for item, action in selection:
        if action.kind == CandidateActionKind.SHED:
            return None
        if item.semantic_key != action.semantic_key:
            return None
        if item.deadline is not None and action.deadline > item.deadline:
            return None
        end = action.ready_at + action.duration
        if action.move_set is not None:
            move_set = next(candidate for candidate in plan.move_sets if candidate.key == action.move_set)
            end = max(end, max((move.end_at for move in move_set.moves), default=end))
            move_keys.update(move_set.move_keys)
        if end > action.deadline:
            return None
        credits += action.buffer_credits
        for use in action.budget_uses:
            budget_totals[use.key] += use.amount
    if credits > plan.buffer_capacity:
        return None
    if any(budget_totals[key] > budget.capacity for key, budget in ((budget.key, budget) for budget in plan.budgets)):
        return None
    selected_keys = {item.key for item, _ in selection}
    shed = tuple(item for item in items if item.key not in selected_keys)
    if any(not _can_shed(plan, item) for item in shed):
        return None
    required_count = sum(1 for item, _ in selection if _must_complete(plan, item))
    optional_count = sum(1 for item, _ in selection if not _must_complete(plan, item))
    fallback_count = sum(1 for _, action in selection if action.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK)
    utility = sum(action.utility for _, action in selection)
    duration = sum(action.duration for _, action in selection)
    # The final key tuple makes ties independent of input dictionary/set order.
    key_order = tuple(action.key.value for _, action in sorted(selection, key=lambda pair: pair[0].key))
    objective = (
        required_count,
        optional_count,
        utility,
        -fallback_count,
        -duration,
        -len(move_keys),
        *tuple(-ord(char) for key in key_order for char in key),
    )
    return _EvaluatedSelection(selection, shed, objective)


def _result(evaluated: _EvaluatedSelection) -> AdmissionResult:
    return AdmissionResult(
        selected=tuple((item.key, action.key) for item, action in evaluated.selected),
        shed=tuple(item.key for item in evaluated.shed),
        objective=evaluated.objective,
    )


def greedy_admission(
    plan: Phase5ControlPlan,
    items: Sequence[WorkItemDesc],
) -> AdmissionResult:
    """Deterministically admit approved actions in required-first order."""

    plan = _instance(plan, Phase5ControlPlan, "greedy_admission.plan")
    typed = _tuple_of(tuple(items), WorkItemDesc, "greedy_admission.items")
    if len({item.key for item in typed}) != len(typed):
        raise ValueError("admission work item keys must be unique")
    ordered = tuple(
        sorted(typed, key=lambda item: (not _must_complete(plan, item), item.key))
    )
    chosen: list[tuple[WorkItemDesc, CandidateActionDesc]] = []
    for item in ordered:
        options = approved_actions(plan, item)
        options = tuple(
            option
            for option in options
            if _must_complete(plan, item)
            or option.kind != CandidateActionKind.SHED
            or item.allow_shedding
        )
        if _must_complete(plan, item) and all(
            option.kind == CandidateActionKind.SHED for option in options
        ):
            return AdmissionResult((), (), (), feasible=False)
        accepted = False
        for option in options:
            considered = tuple(selected_item for selected_item, _ in chosen) + (item,)
            evaluated = _evaluate_selection(
                plan,
                tuple(chosen + [(item, option)]),
                items=considered,
            )
            if evaluated is not None:
                chosen.append((item, option))
                accepted = True
                break
        if not accepted and _must_complete(plan, item):
            return AdmissionResult((), (), (), feasible=False)
    evaluated = _evaluate_selection(plan, tuple(chosen), items=typed)
    if evaluated is None:
        return AdmissionResult((), (), (), feasible=False)
    return _result(evaluated)


def exhaustive_admission_oracle(
    plan: Phase5ControlPlan,
    items: Sequence[WorkItemDesc],
    *,
    max_items: int = 10,
) -> AdmissionResult:
    """Exhaustive small-graph oracle using the greedy feasibility/objective."""

    plan = _instance(plan, Phase5ControlPlan, "exhaustive_admission_oracle.plan")
    typed = _tuple_of(tuple(items), WorkItemDesc, "exhaustive_admission_oracle.items")
    max_items = _integer(max_items, "exhaustive_admission_oracle.max_items", minimum=1)
    if len(typed) > max_items:
        raise ValueError("exhaustive admission is bounded to a small graph")
    if len({item.key for item in typed}) != len(typed):
        raise ValueError("admission work item keys must be unique")
    options: list[tuple[CandidateActionDesc | None, ...]] = []
    for item in typed:
        candidates = approved_actions(plan, item)
        if _can_shed(plan, item):
            candidates = candidates + (None,)
        options.append(candidates)
    best: _EvaluatedSelection | None = None
    for choices in product(*options):
        selection = tuple(
            (item, action)
            for item, action in zip(typed, choices)
            if action is not None
        )
        evaluated = _evaluate_selection(plan, selection, items=typed)
        if evaluated is not None and (best is None or evaluated.objective > best.objective):
            best = evaluated
    if best is None:
        return AdmissionResult((), (), (), feasible=False)
    return _result(best)


def static_admission(plan: Phase5ControlPlan, items: Sequence[WorkItemDesc]) -> AdmissionResult:
    """Use the deterministic compiler-approved order, with no profile adaptation."""

    plan = _instance(plan, Phase5ControlPlan, "static_admission.plan")
    return greedy_admission(plan, items)


def admit_greedy(plan: Phase5ControlPlan, items: Sequence[WorkItemDesc]) -> AdmissionResult:
    return greedy_admission(plan, items)


def admit_exhaustive_oracle(
    plan: Phase5ControlPlan,
    items: Sequence[WorkItemDesc],
    *,
    max_items: int = 10,
) -> AdmissionResult:
    return exhaustive_admission_oracle(plan, items, max_items=max_items)


def compile_phase5_plan(
    *,
    placement_plan: PlacementPlan,
    move_sets: tuple[CandidateMoveSetDesc, ...],
    workload_buckets: tuple[WorkloadBucketDesc, ...],
    shape_buckets: tuple[ShapeBucketDesc, ...],
    budgets: tuple[InterferenceBudgetDesc, ...],
    actions: tuple[CandidateActionDesc, ...],
    profiles: tuple[IntegerEWMAProfile, ...] = (),
    unknown_bucket_fallback: CandidateActionKey,
    buffer_capacity: int = 0,
    mode: AdmissionMode = AdmissionMode.STATIC,
    densification_limit: int = _DEFAULT_DENSIFICATION_LIMIT,
) -> Phase5ControlPlan:
    """Compile and validate the pure Phase 5 control-plane plan."""

    placement_plan = _instance(placement_plan, PlacementPlan, "compile_phase5_plan.placement_plan")
    try:
        from tllm import core as _core

        _core.validate_placement_plan(placement_plan)
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        raise Phase5CompilationError(
            "Rust placement-plan descriptor validation failed"
        ) from exc
    scheduled_moves = placement_plan.moves
    validate_candidate_move_sets(scheduled_moves, move_sets)
    result = Phase5ControlPlan(
        plan_epoch=placement_plan.plan_epoch,
        placement_plan=placement_plan,
        scheduled_moves=scheduled_moves,
        move_sets=move_sets,
        workload_buckets=workload_buckets,
        shape_buckets=shape_buckets,
        budgets=budgets,
        actions=actions,
        profiles=profiles,
        placement_signature=_placement_signature(placement_plan),
        densification_limit=densification_limit,
        unknown_bucket_fallback=unknown_bucket_fallback,
        buffer_capacity=buffer_capacity,
        mode=mode,
    )
    try:
        _core.validate_adaptive_flow_plan(result)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Phase5CompilationError(
            "Rust adaptive-plan descriptor validation failed"
        ) from exc
    return result


__all__ = [
    "AdmissionMode",
    "AdmissionResult",
    "BudgetUseDesc",
    "CandidateActionDesc",
    "CandidateActionKey",
    "CandidateActionKind",
    "CandidateMoveSetDesc",
    "EWMAProfileKey",
    "FixedRatio",
    "IntegerEWMAProfile",
    "InterferenceBudgetDesc",
    "InterferenceBudgetKey",
    "MoveSetKey",
    "Phase5CompilationError",
    "Phase5ControlPlan",
    "ShapeBucketDesc",
    "ShapeBucketKey",
    "WorkItemDesc",
    "WorkItemKey",
    "WorkloadBucketDesc",
    "WorkloadBucketKey",
    "admit_exhaustive_oracle",
    "admit_greedy",
    "approved_actions",
    "compile_phase5_plan",
    "exhaustive_admission_oracle",
    "greedy_admission",
    "static_admission",
    "validate_candidate_move_sets",
]
