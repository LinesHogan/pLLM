"""Pure-data IR for auxiliary workload declarations.

The classes in this module are safe to serialize, hash, and send through the
compiler control plane. Runtime callables, tensors, streams, and backend
objects deliberately live outside this IR.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from typing import Any, Mapping, cast

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite


class StateScope(StrEnum):
    REQUEST = "request"
    REPLICA = "replica"
    SHARD = "shard"
    GLOBAL = "global"


class ApproximationPolicy(StrEnum):
    EXACT = "exact"
    JACOBI = "jacobi"


class EffectKind(StrEnum):
    READ = "read"
    CURRENT = "current"
    TRANSITION = "transition"
    NEXT = "next"


class WorkloadEdgeKind(StrEnum):
    DATA = "data"
    EFFECT = "effect"
    CONTROL = "control"


class ImplementationKind(StrEnum):
    CAPTURED = "captured"
    EXTERN = "extern"


class ValueSourceKind(StrEnum):
    HIDDEN = "hidden"
    RESULT = "result"
    PROVIDED = "provided"
    STEP_ID = "step_id"


class HiddenLayerKind(StrEnum):
    INDEX = "index"
    PATH = "path"
    WILDCARD = "wildcard"


class ValueTargetKind(StrEnum):
    SINK = "sink"
    TEMPORARY = "temporary"


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _nonnegative(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _str_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{label} must be tuple[non-empty str, ...]")
    return cast(tuple[str, ...], value)


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    kind: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _nonempty(self.kind, "OptimizerSpec.kind"))
        if not isinstance(self.options, tuple):
            raise TypeError("OptimizerSpec.options must be tuple")
        normalized: list[tuple[str, str]] = []
        for option in self.options:
            if not isinstance(option, tuple) or len(option) != 2:
                raise TypeError("OptimizerSpec.options must contain key/value tuples")
            normalized.append(
                (_nonempty(option[0], "optimizer option key"), _nonempty(option[1], "optimizer option value"))
            )
        object.__setattr__(self, "options", tuple(normalized))


@dataclass(frozen=True, slots=True)
class StateIR:
    name: str
    scope: StateScope
    request_key: tuple[str, ...]
    parameters: tuple[str, ...]
    optimizer: OptimizerSpec | None
    transition: str = "optimizer"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "StateIR.name"))
        object.__setattr__(self, "scope", StateScope(self.scope))
        object.__setattr__(self, "request_key", _str_tuple(self.request_key, "StateIR.request_key"))
        object.__setattr__(self, "parameters", _str_tuple(self.parameters, "StateIR.parameters"))
        if self.scope is StateScope.REQUEST and not self.request_key:
            raise ValueError("REQUEST StateIR requires a request_key")
        if self.scope is not StateScope.REQUEST and self.request_key:
            raise ValueError("only REQUEST StateIR may declare a request_key")
        if self.optimizer is not None and not isinstance(self.optimizer, OptimizerSpec):
            raise TypeError("StateIR.optimizer must be OptimizerSpec or None")
        object.__setattr__(self, "transition", _nonempty(self.transition, "StateIR.transition"))


@dataclass(frozen=True, slots=True)
class TaskInputIR:
    name: str
    kind: ValueSourceKind
    source: str
    output: str = ""
    layer: str = ""
    layer_kind: str = ""
    site: HiddenSite = HiddenSite.NONE
    phase: HiddenPhase = HiddenPhase.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "TaskInputIR.name"))
        object.__setattr__(self, "kind", ValueSourceKind(self.kind))
        object.__setattr__(self, "source", _nonempty(self.source, "TaskInputIR.source"))
        object.__setattr__(self, "site", HiddenSite(self.site))
        object.__setattr__(self, "phase", HiddenPhase(self.phase))
        if self.kind is ValueSourceKind.HIDDEN:
            object.__setattr__(self, "layer_kind", HiddenLayerKind(self.layer_kind).value)
            if self.site is HiddenSite.NONE or self.phase is HiddenPhase.NONE:
                raise ValueError("hidden TaskInputIR requires site and phase")
        elif self.layer_kind:
            raise ValueError("only hidden TaskInputIR may declare layer_kind")
        elif self.site is not HiddenSite.NONE or self.phase is not HiddenPhase.NONE:
            raise ValueError("only hidden TaskInputIR may declare a locator")
        if self.kind is ValueSourceKind.RESULT and not self.output:
            raise ValueError("result TaskInputIR requires an output name")
        if self.kind is ValueSourceKind.STEP_ID and self.source != "step_id":
            raise ValueError("step_id TaskInputIR requires the built-in step_id source")


@dataclass(frozen=True, slots=True)
class TaskOutputIR:
    name: str
    kind: ValueTargetKind
    target: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "TaskOutputIR.name"))
        object.__setattr__(self, "kind", ValueTargetKind(self.kind))
        object.__setattr__(self, "target", _nonempty(self.target, "TaskOutputIR.target"))


@dataclass(frozen=True, slots=True)
class EffectIR:
    kind: EffectKind
    state: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EffectKind(self.kind))
        object.__setattr__(self, "state", _nonempty(self.state, "EffectIR.state"))


@dataclass(frozen=True, slots=True)
class TimingIR:
    after: str
    before: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "after", _nonempty(self.after, "TimingIR.after"))
        object.__setattr__(self, "before", _nonempty(self.before, "TimingIR.before"))
        if self.after == self.before:
            raise ValueError("TimingIR requires distinct boundaries")


@dataclass(frozen=True, slots=True)
class AffineExprIR:
    coefficients: tuple[tuple[str, int], ...] = ()
    constant: int = 0


@dataclass(frozen=True, slots=True)
class AffineAccessIR:
    value: str
    indices: tuple[AffineExprIR, ...]
    mode: str = "read"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _nonempty(self.value, "AffineAccessIR.value"))
        if self.mode not in {"read", "write", "read_write"}:
            raise ValueError("AffineAccessIR.mode must be read, write, or read_write")


@dataclass(frozen=True, slots=True)
class DomainIR:
    name: str = "scalar"
    indices: tuple[str, ...] = ()
    bounds: tuple[tuple[str, str, str], ...] = ()
    parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "DomainIR.name"))
        object.__setattr__(self, "indices", _str_tuple(self.indices, "DomainIR.indices"))
        object.__setattr__(self, "parameters", _str_tuple(self.parameters, "DomainIR.parameters"))
        known = set(self.indices)
        normalized: list[tuple[str, str, str]] = []
        for bound in self.bounds:
            if not isinstance(bound, tuple) or len(bound) != 3:
                raise TypeError("DomainIR.bounds must contain (index, lower, upper) tuples")
            index, lower, upper = (str(part) for part in bound)
            if index not in known:
                raise ValueError(f"DomainIR bound references unknown index {index!r}")
            normalized.append((index, lower, upper))
        object.__setattr__(self, "bounds", tuple(normalized))


@dataclass(frozen=True, slots=True)
class ImplementationIR:
    kind: ImplementationKind
    key: str
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ImplementationKind(self.kind))
        object.__setattr__(self, "key", _nonempty(self.key, "ImplementationIR.key"))
        object.__setattr__(self, "input_types", _str_tuple(self.input_types, "ImplementationIR.input_types"))
        object.__setattr__(self, "output_types", _str_tuple(self.output_types, "ImplementationIR.output_types"))
        object.__setattr__(self, "capabilities", _str_tuple(self.capabilities, "ImplementationIR.capabilities"))


@dataclass(frozen=True, slots=True)
class TaskIR:
    name: str
    ordinal: int
    inputs: tuple[TaskInputIR, ...]
    outputs: tuple[TaskOutputIR, ...]
    effects: tuple[EffectIR, ...]
    timing: TimingIR
    implementations: tuple[ImplementationIR, ...]
    domain: DomainIR = DomainIR()
    accesses: tuple[AffineAccessIR, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "TaskIR.name"))
        object.__setattr__(self, "ordinal", _nonnegative(self.ordinal, "TaskIR.ordinal"))
        if not self.implementations:
            raise ValueError("TaskIR requires at least one implementation")


@dataclass(frozen=True, slots=True)
class ValueIR:
    value_id: str
    producer: str
    output: str
    consumers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkloadEdge:
    source: str
    target: str
    kind: WorkloadEdgeKind
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _nonempty(self.source, "WorkloadEdge.source"))
        object.__setattr__(self, "target", _nonempty(self.target, "WorkloadEdge.target"))
        object.__setattr__(self, "kind", WorkloadEdgeKind(self.kind))


def canonical_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_value(getattr(value, field.name))
            for field in fields(value)
            if field.name != "semantic_key"
        }
    raise TypeError(f"workload IR cannot serialize {type(value).__module__}.{type(value).__qualname__}")


def canonical_json(value: object) -> str:
    return json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class WorkloadIR:
    workload_id: str
    states: tuple[StateIR, ...]
    tasks: tuple[TaskIR, ...]
    values: tuple[ValueIR, ...]
    edges: tuple[WorkloadEdge, ...]
    schedule: tuple[str, ...]
    approximation: ApproximationPolicy = ApproximationPolicy.EXACT
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "workload_id", _nonempty(self.workload_id, "WorkloadIR.workload_id"))
        object.__setattr__(self, "approximation", ApproximationPolicy(self.approximation))
        if self.schema_version != 1:
            raise ValueError(f"unsupported WorkloadIR.schema_version {self.schema_version}")
        names = tuple(task.name for task in self.tasks)
        if len(names) != len(set(names)):
            raise ValueError("WorkloadIR contains duplicate task names")
        if set(self.schedule) != set(names) or len(self.schedule) != len(names):
            raise ValueError("WorkloadIR.schedule must contain every task exactly once")

    @property
    def semantic_key(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    def to_json(self) -> str:
        return canonical_json(self)


__all__ = [
    "AffineAccessIR",
    "AffineExprIR",
    "ApproximationPolicy",
    "DomainIR",
    "EffectIR",
    "EffectKind",
    "HiddenLayerKind",
    "ImplementationIR",
    "ImplementationKind",
    "OptimizerSpec",
    "StateIR",
    "StateScope",
    "TaskIR",
    "TaskInputIR",
    "TaskOutputIR",
    "TimingIR",
    "ValueIR",
    "ValueSourceKind",
    "ValueTargetKind",
    "WorkloadEdge",
    "WorkloadEdgeKind",
    "WorkloadIR",
    "canonical_json",
    "canonical_value",
]
