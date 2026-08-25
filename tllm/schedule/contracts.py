"""Pure-data contracts at the scheduler FFI boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from typing import Mapping

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite


def _name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _bytes(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _canonical(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            if field.name != "signature"
        }
    raise TypeError(f"schedule metadata cannot serialize {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def structural_key(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _require_json_fields(
    raw: object,
    label: str,
    required: tuple[str, ...],
    *,
    allowed: tuple[str, ...] | None = None,
) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    missing = tuple(field_name for field_name in required if field_name not in raw)
    if missing:
        raise ValueError(
            f"{label} schema v2 is missing required fields: {', '.join(missing)}"
        )
    if allowed is not None:
        unknown = tuple(sorted(set(raw).difference(allowed)))
        if unknown:
            raise ValueError(
                f"{label} schema v2 has unknown fields: {', '.join(unknown)}"
            )
    return raw


def _json_array(raw: object, label: str) -> tuple[object, ...]:
    """Narrow one decoded JSON array without accepting arbitrary iterables."""
    if not isinstance(raw, list):
        raise TypeError(f"{label} must be a JSON array")
    return tuple(raw)


def _json_string(raw: object, label: str) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"{label} must be a JSON string")
    return raw


def _json_bool(raw: object, label: str) -> bool:
    if not isinstance(raw, bool):
        raise TypeError(f"{label} must be a JSON boolean")
    return raw


def _json_optional_int(raw: object, label: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise TypeError(f"{label} must be a JSON integer or null")
    return raw


def _json_int(raw: object, label: str) -> int:
    value = _json_optional_int(raw, label)
    if value is None:
        raise TypeError(f"{label} must be a JSON integer")
    return value


def _json_strings(raw: object, label: str) -> tuple[str, ...]:
    return tuple(
        _json_string(item, f"{label}[{position}]")
        for position, item in enumerate(_json_array(raw, label))
    )


@dataclass(frozen=True, slots=True)
class TimelineAnchor:
    anchor_id: str
    ordinal: int
    observable: bool = True
    graph_boundary: bool = True
    visible_values: tuple[str, ...] = ()
    backend_events: tuple["BackendObservableEvent", ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_id", _name(self.anchor_id, "TimelineAnchor.anchor_id"))
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool) or self.ordinal < 0:
            raise ValueError("TimelineAnchor.ordinal must be a non-negative int")
        if not isinstance(self.observable, bool) or not isinstance(self.graph_boundary, bool):
            raise TypeError("TimelineAnchor flags must be bool")
        if self.graph_boundary and not self.observable:
            raise ValueError("a graph boundary must be observable")
        if not isinstance(self.visible_values, tuple) or any(
            not isinstance(value, str) or not value for value in self.visible_values
        ):
            raise TypeError("TimelineAnchor.visible_values must be tuple[str, ...]")
        if self.backend_events is not None:
            if not isinstance(self.backend_events, tuple):
                raise TypeError("TimelineAnchor.backend_events must be tuple or None")
            normalized = tuple(
                BackendObservableEvent(event) for event in self.backend_events
            )
            if len(normalized) != len(set(normalized)):
                raise ValueError("TimelineAnchor.backend_events must be unique")
            object.__setattr__(self, "backend_events", normalized)


@dataclass(frozen=True, slots=True)
class ResourceReservation:
    memory_bytes: int = 0
    compute_units: int = 0
    bandwidth_units: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_bytes", _bytes(self.memory_bytes, "ResourceReservation.memory_bytes"))
        object.__setattr__(self, "compute_units", _bytes(self.compute_units, "ResourceReservation.compute_units"))
        object.__setattr__(self, "bandwidth_units", _bytes(self.bandwidth_units, "ResourceReservation.bandwidth_units"))


@dataclass(frozen=True, slots=True)
class MainWorkloadRegion:
    region_id: str
    start: str
    end: str
    pinned: bool = True
    reservation: ResourceReservation = ResourceReservation()
    capture_region: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _name(self.region_id, "MainWorkloadRegion.region_id"))
        object.__setattr__(self, "start", _name(self.start, "MainWorkloadRegion.start"))
        object.__setattr__(self, "end", _name(self.end, "MainWorkloadRegion.end"))
        if self.start == self.end:
            raise ValueError("MainWorkloadRegion requires distinct boundaries")
        if self.pinned is not True:
            raise ValueError("main workload regions must be pinned")
        if not isinstance(self.reservation, ResourceReservation):
            raise TypeError("MainWorkloadRegion.reservation must be ResourceReservation")


@dataclass(frozen=True, slots=True)
class MainWorkloadTimeline:
    backend_id: str
    backend_version: str
    runner_family: str
    graph_mode: str
    anchors: tuple[TimelineAnchor, ...]
    regions: tuple[MainWorkloadRegion, ...]
    static_headroom_bytes: int
    capabilities: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field_name in ("backend_id", "backend_version", "runner_family", "graph_mode"):
            object.__setattr__(self, field_name, _name(getattr(self, field_name), f"MainWorkloadTimeline.{field_name}"))
        if self.runner_family not in {"mrv1", "mrv2"}:
            raise ValueError("MainWorkloadTimeline.runner_family must be mrv1 or mrv2")
        if self.graph_mode not in {"eager", "piecewise", "full"}:
            raise ValueError("MainWorkloadTimeline.graph_mode must be eager, piecewise, or full")
        if not isinstance(self.anchors, tuple) or any(
            not isinstance(anchor, TimelineAnchor) for anchor in self.anchors
        ):
            raise TypeError("MainWorkloadTimeline.anchors must be tuple[TimelineAnchor, ...]")
        if not isinstance(self.regions, tuple) or any(
            not isinstance(region, MainWorkloadRegion) for region in self.regions
        ):
            raise TypeError("MainWorkloadTimeline.regions must be tuple[MainWorkloadRegion, ...]")
        if not self.anchors:
            raise ValueError("MainWorkloadTimeline requires anchors")
        anchor_ids = tuple(anchor.anchor_id for anchor in self.anchors)
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("MainWorkloadTimeline anchor ids must be unique")
        if tuple(anchor.ordinal for anchor in self.anchors) != tuple(range(len(self.anchors))):
            raise ValueError("MainWorkloadTimeline anchors must have contiguous canonical ordinals")
        index = {anchor.anchor_id: anchor.ordinal for anchor in self.anchors}
        region_ids: set[str] = set()
        for region in self.regions:
            if region.region_id in region_ids:
                raise ValueError("MainWorkloadTimeline region ids must be unique")
            region_ids.add(region.region_id)
            if region.start not in index or region.end not in index:
                raise ValueError(f"region {region.region_id!r} references an unknown anchor")
            if index[region.start] >= index[region.end]:
                raise ValueError(f"region {region.region_id!r} has an invalid interval")
        object.__setattr__(self, "static_headroom_bytes", _bytes(self.static_headroom_bytes, "MainWorkloadTimeline.static_headroom_bytes"))
        if not isinstance(self.capabilities, tuple) or any(
            not isinstance(capability, str) or not capability
            for capability in self.capabilities
        ):
            raise TypeError("MainWorkloadTimeline.capabilities must be tuple[str, ...]")
        if tuple(sorted(set(self.capabilities))) != self.capabilities:
            raise ValueError("MainWorkloadTimeline.capabilities must be unique and sorted")
        if self.schema_version not in {1, 2}:
            raise ValueError("unsupported MainWorkloadTimeline schema version")
        if self.schema_version == 2 and any(
            anchor.backend_events is None for anchor in self.anchors
        ):
            raise ValueError(
                "MainWorkloadTimeline schema v2 anchors require backend_events"
            )

    @property
    def timeline_key(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    def to_json(self) -> str:
        payload = _canonical(self)
        assert isinstance(payload, dict)
        if self.schema_version == 1:
            anchors = payload["anchors"]
            assert isinstance(anchors, list)
            for anchor in anchors:
                assert isinstance(anchor, dict)
                anchor.pop("backend_events")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, payload: str) -> "MainWorkloadTimeline":
        raw = json.loads(payload)
        if not isinstance(raw, Mapping):
            raise TypeError("MainWorkloadTimeline must be a JSON object")
        schema_version = raw.get("schema_version", 1)
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise TypeError("MainWorkloadTimeline.schema_version must be int")
        return cls(
            backend_id=raw["backend_id"],
            backend_version=raw["backend_version"],
            runner_family=raw["runner_family"],
            graph_mode=raw["graph_mode"],
            anchors=tuple(
                TimelineAnchor(
                    anchor_id=anchor["anchor_id"],
                    ordinal=anchor["ordinal"],
                    observable=anchor["observable"],
                    graph_boundary=anchor["graph_boundary"],
                    visible_values=tuple(anchor.get("visible_values", ())),
                    backend_events=(
                        None
                        if "backend_events" not in anchor
                        else tuple(
                            BackendObservableEvent(event)
                            for event in _json_array(
                                anchor["backend_events"],
                                "TimelineAnchor.backend_events",
                            )
                        )
                    ),
                )
                for anchor in raw["anchors"]
            ),
            regions=tuple(
                MainWorkloadRegion(
                    region_id=region["region_id"],
                    start=region["start"],
                    end=region["end"],
                    pinned=region["pinned"],
                    reservation=ResourceReservation(**region["reservation"]),
                    capture_region=region["capture_region"],
                )
                for region in raw["regions"]
            ),
            static_headroom_bytes=raw["static_headroom_bytes"],
            capabilities=tuple(raw.get("capabilities", ())),
            schema_version=schema_version,
        )


class ScheduleEdgeKind(StrEnum):
    DATA = "data"
    EFFECT = "effect"
    CONTROL = "control"
    ANCHOR = "anchor"


class BackendObservableEvent(StrEnum):
    """Closed backend hook events admitted by a signed timeline."""

    PREPARE_INPUTS_PRE = "prepare_inputs.pre"
    PREPARE_INPUTS_POST = "prepare_inputs.post"
    EXECUTE_MODEL_PRE = "execute_model.pre"
    EXECUTE_MODEL_POST = "execute_model.post"
    SAMPLE_TOKENS_PRE = "sample_tokens.pre"
    SAMPLER_PRE_DRAW = "sampler.pre_draw"
    SAMPLE_TOKENS_POST = "sample_tokens.post"


class ScheduleEffectKind(StrEnum):
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    STATE_TRANSITION = "state_transition"
    VERSION_PUBLISH = "version_publish"
    EXTERN_IO = "extern_io"


class ScheduleInputKind(StrEnum):
    HIDDEN = "hidden"
    RESULT = "result"
    PROVIDED = "provided"
    STEP_ID = "step_id"


class ScheduleOutputKind(StrEnum):
    SINK = "sink"
    TEMPORARY = "temporary"


class RegionInputKind(StrEnum):
    HIDDEN = "hidden"
    PROVIDED = "provided"
    MATERIALIZED = "materialized"
    STEP_ID = "step_id"


class HiddenLayerKind(StrEnum):
    INDEX = "index"
    PATH = "path"
    WILDCARD = "wildcard"
    NONE = "none"


class RegionOutputKind(StrEnum):
    SINK = "sink"
    MATERIALIZED = "materialized"


class RegionImplementationKind(StrEnum):
    """Closed execution choice emitted by Rust for one ExecutionRegion."""

    COMPILED = "compiled"
    EXTERN = "extern"


class RegionCompilerAdapter(StrEnum):
    """Closed Rust-selected adapter for a normal ExecutionRegion."""

    TORCH_COMPILE = "torch_compile"
    EAGER = "eager"


class RegionAutogradPolicy(StrEnum):
    """Closed gradient-context command carried by a Region artifact."""

    ENABLE_GRAD = "enable_grad"


class RegionCaptureMode(StrEnum):
    """Closed capture command emitted and signed by the Rust planner."""

    DISABLED = "disabled"
    REQUIRE = "require"


class RegionStreamClass(StrEnum):
    """Closed physical stream command emitted and signed by the Rust planner."""

    DEFAULT = "default"


class RegionFallbackPolicy(StrEnum):
    """Closed fallback command emitted and signed by the Rust planner."""

    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class RegionCompileCommand:
    """Rust-signed compiler policy; Python may only dispatch this record."""

    adapter: RegionCompilerAdapter
    fullgraph: bool
    dynamic: bool
    shape_specialization: tuple[tuple[str, str], ...]
    autograd_policy: RegionAutogradPolicy
    fallback_policy: RegionFallbackPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter", RegionCompilerAdapter(self.adapter))
        if not isinstance(self.fullgraph, bool) or not isinstance(self.dynamic, bool):
            raise TypeError("RegionCompileCommand.fullgraph and dynamic must be bool")
        if self.adapter is RegionCompilerAdapter.TORCH_COMPILE:
            if not self.fullgraph:
                raise ValueError("torch_compile RegionCompileCommand requires fullgraph")
        elif self.fullgraph:
            raise ValueError("eager RegionCompileCommand cannot request fullgraph")
        if self.dynamic:
            raise ValueError("RegionCompileCommand only supports static shapes")
        if not isinstance(self.shape_specialization, tuple):
            raise TypeError("RegionCompileCommand.shape_specialization must be tuple")
        normalized: list[tuple[str, str]] = []
        for item in self.shape_specialization:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "RegionCompileCommand.shape_specialization must contain key/value tuples"
                )
            normalized.append(
                (
                    _name(item[0], "RegionCompileCommand shape key"),
                    _name(item[1], "RegionCompileCommand shape value"),
                )
            )
        if tuple(normalized) != tuple(sorted(normalized)) or len(
            {key for key, _value in normalized}
        ) != len(normalized):
            raise ValueError(
                "RegionCompileCommand.shape_specialization must have sorted unique keys"
            )
        object.__setattr__(self, "shape_specialization", tuple(normalized))
        object.__setattr__(
            self, "autograd_policy", RegionAutogradPolicy(self.autograd_policy)
        )
        object.__setattr__(
            self, "fallback_policy", RegionFallbackPolicy(self.fallback_policy)
        )



@dataclass(frozen=True, slots=True)
class ScheduleEffect:
    kind: ScheduleEffectKind
    resource: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ScheduleEffectKind(self.kind))
        object.__setattr__(
            self, "resource", _name(self.resource, "ScheduleEffect.resource")
        )


@dataclass(frozen=True, slots=True)
class ScheduleInput:
    name: str
    kind: ScheduleInputKind
    source: str
    output: str = ""
    layer: str = ""
    layer_kind: HiddenLayerKind = HiddenLayerKind.NONE
    site: HiddenSite = HiddenSite.NONE
    phase: HiddenPhase = HiddenPhase.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "ScheduleInput.name"))
        object.__setattr__(self, "kind", ScheduleInputKind(self.kind))
        object.__setattr__(self, "source", _name(self.source, "ScheduleInput.source"))
        object.__setattr__(self, "layer_kind", HiddenLayerKind(self.layer_kind))
        object.__setattr__(self, "site", HiddenSite(self.site))
        object.__setattr__(self, "phase", HiddenPhase(self.phase))
        if self.kind is ScheduleInputKind.HIDDEN:
            if self.layer_kind is HiddenLayerKind.NONE:
                raise ValueError("hidden ScheduleInput requires layer_kind")
            if self.site is HiddenSite.NONE or self.phase is HiddenPhase.NONE:
                raise ValueError("hidden ScheduleInput requires site and phase")
        elif self.layer_kind is not HiddenLayerKind.NONE:
            raise ValueError("only hidden ScheduleInput may declare layer_kind")
        elif self.site is not HiddenSite.NONE or self.phase is not HiddenPhase.NONE:
            raise ValueError("only hidden ScheduleInput may declare a locator")
        if self.kind is ScheduleInputKind.RESULT and not self.output:
            raise ValueError("result ScheduleInput requires an output name")
        if self.kind is ScheduleInputKind.STEP_ID and self.source != "step_id":
            raise ValueError("step_id ScheduleInput requires the built-in step_id source")


@dataclass(frozen=True, slots=True)
class ScheduleOutput:
    name: str
    kind: ScheduleOutputKind
    target: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "ScheduleOutput.name"))
        object.__setattr__(self, "kind", ScheduleOutputKind(self.kind))
        object.__setattr__(self, "target", _name(self.target, "ScheduleOutput.target"))


@dataclass(frozen=True, slots=True)
class TaskTiming:
    after: str
    before: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "after", _name(self.after, "TaskTiming.after"))
        object.__setattr__(self, "before", _name(self.before, "TaskTiming.before"))
        if self.after == self.before:
            raise ValueError("TaskTiming requires distinct anchors")


@dataclass(frozen=True, slots=True)
class ScheduleAffineExpr:
    coefficients: tuple[tuple[str, int], ...] = ()
    constant: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.coefficients, tuple):
            raise TypeError("ScheduleAffineExpr.coefficients must be tuple")
        normalized: list[tuple[str, int]] = []
        for coefficient in self.coefficients:
            if not isinstance(coefficient, tuple) or len(coefficient) != 2:
                raise TypeError(
                    "ScheduleAffineExpr.coefficients must contain (name, value) tuples"
                )
            name, value = coefficient
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("ScheduleAffineExpr coefficient values must be int")
            if value == 0:
                raise ValueError("ScheduleAffineExpr coefficients must be non-zero")
            normalized.append((_name(name, "ScheduleAffineExpr coefficient name"), value))
        if len({name for name, _value in normalized}) != len(normalized):
            raise ValueError("ScheduleAffineExpr coefficient names must be unique")
        if tuple(sorted(normalized)) != tuple(normalized):
            raise ValueError("ScheduleAffineExpr coefficients must be sorted")
        if not isinstance(self.constant, int) or isinstance(self.constant, bool):
            raise TypeError("ScheduleAffineExpr.constant must be int")


@dataclass(frozen=True, slots=True)
class ScheduleAffineAccess:
    value: str
    indices: tuple[ScheduleAffineExpr, ...]
    mode: str = "read"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _name(self.value, "ScheduleAffineAccess.value"))
        if not isinstance(self.indices, tuple) or any(
            not isinstance(index, ScheduleAffineExpr) for index in self.indices
        ):
            raise TypeError(
                "ScheduleAffineAccess.indices must be tuple[ScheduleAffineExpr, ...]"
            )
        if self.mode not in {"read", "write", "read_write"}:
            raise ValueError(
                "ScheduleAffineAccess.mode must be read, write, or read_write"
            )


@dataclass(frozen=True, slots=True)
class ScheduleDomain:
    name: str = "scalar"
    indices: tuple[str, ...] = ()
    bounds: tuple[tuple[str, str, str], ...] = ()
    parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "ScheduleDomain.name"))
        if not isinstance(self.indices, tuple) or any(
            not isinstance(index, str) or not index for index in self.indices
        ):
            raise TypeError("ScheduleDomain.indices must be tuple[str, ...]")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("ScheduleDomain.indices must be unique")
        if not isinstance(self.parameters, tuple) or any(
            not isinstance(parameter, str) or not parameter
            for parameter in self.parameters
        ):
            raise TypeError("ScheduleDomain.parameters must be tuple[str, ...]")
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError("ScheduleDomain.parameters must be unique")
        if not isinstance(self.bounds, tuple):
            raise TypeError("ScheduleDomain.bounds must be tuple")
        known = set(self.indices)
        bounded: set[str] = set()
        for bound in self.bounds:
            if not isinstance(bound, tuple) or len(bound) != 3:
                raise TypeError(
                    "ScheduleDomain.bounds must contain (index, lower, upper) tuples"
                )
            index, lower, upper = bound
            if index not in known:
                raise ValueError(
                    f"ScheduleDomain bound references unknown index {index!r}"
                )
            if index in bounded:
                raise ValueError(f"ScheduleDomain has duplicate bound for {index!r}")
            bounded.add(index)
            _name(lower, "ScheduleDomain lower bound")
            _name(upper, "ScheduleDomain upper bound")
        if self.indices and bounded != known:
            raise ValueError("ScheduleDomain requires one bound for every index")
        if not self.indices and (self.bounds or self.parameters):
            raise ValueError("scalar ScheduleDomain cannot declare bounds or parameters")


@dataclass(frozen=True, slots=True)
class ScheduleTask:
    task_id: str
    timing: TaskTiming
    temporary_bytes: int = 0
    implementation_ids: tuple[str, ...] = ("python:capture@1",)
    implementation_kinds: tuple[str, ...] = ("captured",)
    # V2 retains raw per-candidate capability evidence so Rust can derive the
    # backend-facing requirement set instead of trusting Python lowering.
    implementation_capabilities: tuple[tuple[str, ...], ...] = ()
    required_capabilities: tuple[str, ...] = ()
    capture_safe: bool = False
    effects: tuple[ScheduleEffect, ...] = ()
    domain: ScheduleDomain = ScheduleDomain()
    accesses: tuple[ScheduleAffineAccess, ...] = ()
    inputs: tuple[ScheduleInput, ...] = ()
    outputs: tuple[ScheduleOutput, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _name(self.task_id, "ScheduleTask.task_id"))
        if not isinstance(self.timing, TaskTiming):
            raise TypeError("ScheduleTask.timing must be TaskTiming")
        object.__setattr__(self, "temporary_bytes", _bytes(self.temporary_bytes, "ScheduleTask.temporary_bytes"))
        if not self.implementation_ids or any(not isinstance(value, str) or not value for value in self.implementation_ids):
            raise TypeError("ScheduleTask.implementation_ids must be non-empty tuple[str, ...]")
        if not isinstance(self.implementation_ids, tuple):
            raise TypeError("ScheduleTask.implementation_ids must be tuple[str, ...]")
        if not isinstance(self.implementation_kinds, tuple) or any(
            value not in {"captured", "extern"}
            for value in self.implementation_kinds
        ):
            raise TypeError(
                "ScheduleTask.implementation_kinds must be tuple[captured | extern, ...]"
            )
        if len(self.implementation_kinds) != len(self.implementation_ids):
            raise ValueError(
                "ScheduleTask implementation ids and kinds must have equal length"
            )
        if not isinstance(self.implementation_capabilities, tuple) or any(
            not isinstance(capabilities, tuple)
            or any(not isinstance(value, str) or not value for value in capabilities)
            or tuple(sorted(set(capabilities))) != capabilities
            for capabilities in self.implementation_capabilities
        ):
            raise TypeError(
                "ScheduleTask.implementation_capabilities must contain sorted unique capability tuples"
            )
        if self.implementation_capabilities and (
            len(self.implementation_capabilities) != len(self.implementation_ids)
        ):
            raise ValueError(
                "ScheduleTask implementation capabilities must align with implementation ids"
            )
        if not isinstance(self.required_capabilities, tuple) or any(
            not isinstance(value, str) or not value for value in self.required_capabilities
        ):
            raise TypeError("ScheduleTask.required_capabilities must be tuple[str, ...]")
        if not isinstance(self.capture_safe, bool):
            raise TypeError("ScheduleTask.capture_safe must be bool")
        if not isinstance(self.effects, tuple) or any(
            not isinstance(effect, ScheduleEffect) for effect in self.effects
        ):
            raise TypeError("ScheduleTask.effects must be tuple[ScheduleEffect, ...]")
        effect_keys = tuple((effect.kind, effect.resource) for effect in self.effects)
        if len(effect_keys) != len(set(effect_keys)):
            raise ValueError("ScheduleTask effects must be unique")
        if not isinstance(self.domain, ScheduleDomain):
            raise TypeError("ScheduleTask.domain must be ScheduleDomain")
        if not isinstance(self.accesses, tuple) or any(
            not isinstance(access, ScheduleAffineAccess) for access in self.accesses
        ):
            raise TypeError(
                "ScheduleTask.accesses must be tuple[ScheduleAffineAccess, ...]"
            )
        if not isinstance(self.inputs, tuple) or any(
            not isinstance(task_input, ScheduleInput) for task_input in self.inputs
        ):
            raise TypeError("ScheduleTask.inputs must be tuple[ScheduleInput, ...]")
        if not isinstance(self.outputs, tuple) or any(
            not isinstance(output, ScheduleOutput) for output in self.outputs
        ):
            raise TypeError("ScheduleTask.outputs must be tuple[ScheduleOutput, ...]")
        if len({task_input.name for task_input in self.inputs}) != len(self.inputs):
            raise ValueError("ScheduleTask input names must be unique")
        if len({output.name for output in self.outputs}) != len(self.outputs):
            raise ValueError("ScheduleTask output names must be unique")
        if self.accesses and not self.domain.indices:
            raise ValueError("affine accesses require a non-scalar ScheduleDomain")
        if any(len(access.indices) != len(self.domain.indices) for access in self.accesses):
            raise ValueError("affine access rank must equal ScheduleDomain rank")
        known_indices = set(self.domain.indices)
        known_parameters = set(self.domain.parameters)
        for access in self.accesses:
            referenced = {
                name
                for expression in access.indices
                for name, _coefficient in expression.coefficients
            }
            unknown = referenced.difference(known_indices, known_parameters)
            if unknown:
                raise ValueError(
                    "affine access references unknown symbols: "
                    + ", ".join(sorted(unknown))
                )


@dataclass(frozen=True, slots=True)
class ScheduleEdge:
    source: str
    target: str
    kind: ScheduleEdgeKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _name(self.source, "ScheduleEdge.source"))
        object.__setattr__(self, "target", _name(self.target, "ScheduleEdge.target"))
        object.__setattr__(self, "kind", ScheduleEdgeKind(self.kind))
        if self.source == self.target:
            raise ValueError("ScheduleEdge may not be a self edge")


@dataclass(frozen=True, slots=True)
class ScheduleValue:
    value_id: str
    version: int = 1
    producer: str = ""
    producer_output: str = ""
    consumers: tuple[str, ...] = ()
    size_bytes: int = 0
    alias_set: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_id", _name(self.value_id, "ScheduleValue.value_id"))
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("ScheduleValue.version must be a positive int")
        if not isinstance(self.producer, str):
            raise TypeError("ScheduleValue.producer must be str")
        if not isinstance(self.consumers, tuple) or any(
            not isinstance(consumer, str) or not consumer for consumer in self.consumers
        ):
            raise TypeError("ScheduleValue.consumers must be tuple[str, ...]")
        object.__setattr__(self, "size_bytes", _bytes(self.size_bytes, "ScheduleValue.size_bytes"))
        if not isinstance(self.alias_set, str):
            raise TypeError("ScheduleValue.alias_set must be str")


@dataclass(frozen=True, slots=True)
class ScheduleState:
    resource: str
    scope: str
    request_key: tuple[str, ...] = ()
    transition: str = "optimizer"

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource", _name(self.resource, "ScheduleState.resource"))
        if self.scope not in {"request", "replica", "shard", "global"}:
            raise ValueError("ScheduleState.scope is invalid")
        if not isinstance(self.request_key, tuple) or any(
            not isinstance(field, str) or not field for field in self.request_key
        ):
            raise TypeError("ScheduleState.request_key must be tuple[str, ...]")
        if (self.scope == "request") != bool(self.request_key):
            raise ValueError("only request ScheduleState requires a request_key")
        object.__setattr__(
            self, "transition", _name(self.transition, "ScheduleState.transition")
        )


@dataclass(frozen=True, slots=True)
class ScheduleGraph:
    workload_key: str
    tasks: tuple[ScheduleTask, ...]
    edges: tuple[ScheduleEdge, ...] = ()
    values: tuple[ScheduleValue, ...] = ()
    states: tuple[ScheduleState, ...] = ()
    persistent_state_bytes: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "workload_key", _name(self.workload_key, "ScheduleGraph.workload_key"))
        if not isinstance(self.tasks, tuple) or any(
            not isinstance(task, ScheduleTask) for task in self.tasks
        ):
            raise TypeError("ScheduleGraph.tasks must be tuple[ScheduleTask, ...]")
        if not isinstance(self.edges, tuple) or any(
            not isinstance(edge, ScheduleEdge) for edge in self.edges
        ):
            raise TypeError("ScheduleGraph.edges must be tuple[ScheduleEdge, ...]")
        if not isinstance(self.values, tuple) or any(
            not isinstance(value, ScheduleValue) for value in self.values
        ):
            raise TypeError("ScheduleGraph.values must be tuple[ScheduleValue, ...]")
        if not isinstance(self.states, tuple) or any(
            not isinstance(state, ScheduleState) for state in self.states
        ):
            raise TypeError("ScheduleGraph.states must be tuple[ScheduleState, ...]")
        resources = tuple(state.resource for state in self.states)
        if len(resources) != len(set(resources)):
            raise ValueError("ScheduleGraph state resources must be unique")
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("ScheduleGraph task ids must be unique")
        known = set(task_ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("ScheduleGraph edge references an unknown task")
        value_ids = tuple((value.value_id, value.version) for value in self.values)
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("ScheduleGraph value versions must be unique")
        for value in self.values:
            if value.producer and value.producer not in known:
                raise ValueError("ScheduleGraph value references an unknown producer")
            if any(consumer not in known for consumer in value.consumers):
                raise ValueError("ScheduleGraph value references an unknown consumer")
        object.__setattr__(self, "persistent_state_bytes", _bytes(self.persistent_state_bytes, "ScheduleGraph.persistent_state_bytes"))
        if self.schema_version not in {1, 2}:
            raise ValueError("unsupported ScheduleGraph schema version")
        if self.schema_version == 2 and any(
            len(task.implementation_capabilities) != len(task.implementation_ids)
            for task in self.tasks
        ):
            raise ValueError(
                "ScheduleGraph schema v2 tasks require per-implementation capabilities"
            )

    @property
    def graph_key(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    def to_json(self) -> str:
        payload = _canonical(self)
        assert isinstance(payload, dict)
        if self.schema_version == 1:
            tasks = payload["tasks"]
            assert isinstance(tasks, list)
            for task in tasks:
                assert isinstance(task, dict)
                task.pop("implementation_capabilities")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, payload: str) -> "ScheduleGraph":
        raw = json.loads(payload)
        tasks = []
        for task in raw["tasks"]:
            domain_raw = task.get("domain", {})
            accesses = tuple(
                ScheduleAffineAccess(
                    value=access["value"],
                    indices=tuple(
                        ScheduleAffineExpr(
                            coefficients=tuple(
                                (str(name), int(value))
                                for name, value in expression["coefficients"]
                            ),
                            constant=expression["constant"],
                        )
                        for expression in access["indices"]
                    ),
                    mode=access["mode"],
                )
                for access in task.get("accesses", ())
            )
            tasks.append(
                ScheduleTask(
                    task_id=task["task_id"],
                    timing=TaskTiming(**task["timing"]),
                    temporary_bytes=task["temporary_bytes"],
                    implementation_ids=tuple(task["implementation_ids"]),
                    implementation_kinds=tuple(
                        task.get(
                            "implementation_kinds",
                            ("captured",) * len(task["implementation_ids"]),
                        )
                    ),
                    implementation_capabilities=tuple(
                        tuple(capabilities)
                        for capabilities in task.get("implementation_capabilities", ())
                    ),
                    required_capabilities=tuple(task["required_capabilities"]),
                    capture_safe=task["capture_safe"],
                    effects=tuple(
                        ScheduleEffect(
                            kind=ScheduleEffectKind(effect["kind"]),
                            resource=effect["resource"],
                        )
                        for effect in task.get("effects", ())
                    ),
                    domain=ScheduleDomain(
                        name=domain_raw.get("name", "scalar"),
                        indices=tuple(domain_raw.get("indices", ())),
                        bounds=tuple(
                            (str(bound[0]), str(bound[1]), str(bound[2]))
                            for bound in domain_raw.get("bounds", ())
                        ),
                        parameters=tuple(domain_raw.get("parameters", ())),
                    ),
                    accesses=accesses,
                    inputs=tuple(
                        ScheduleInput(
                            name=task_input["name"],
                            kind=ScheduleInputKind(task_input["kind"]),
                            source=task_input["source"],
                            output=task_input.get("output", ""),
                            layer=task_input.get("layer", ""),
                            layer_kind=task_input.get("layer_kind", "none"),
                            site=task_input.get("site", ""),
                            phase=task_input.get("phase", ""),
                        )
                        for task_input in task.get("inputs", ())
                    ),
                    outputs=tuple(
                        ScheduleOutput(
                            name=output["name"],
                            kind=ScheduleOutputKind(output["kind"]),
                            target=output["target"],
                        )
                        for output in task.get("outputs", ())
                    ),
                )
            )
        return cls(
            workload_key=raw["workload_key"],
            tasks=tuple(tasks),
            edges=tuple(
                ScheduleEdge(
                    source=edge["source"],
                    target=edge["target"],
                    kind=ScheduleEdgeKind(edge["kind"]),
                )
                for edge in raw.get("edges", ())
            ),
            values=tuple(
                ScheduleValue(
                    value_id=value["value_id"],
                    version=value["version"],
                    producer=value["producer"],
                    producer_output=value.get("producer_output", ""),
                    consumers=tuple(value["consumers"]),
                    size_bytes=value["size_bytes"],
                    alias_set=value["alias_set"],
                )
                for value in raw.get("values", ())
            ),
            states=tuple(
                ScheduleState(
                    resource=state["resource"],
                    scope=state["scope"],
                    request_key=tuple(state.get("request_key", ())),
                    transition=state.get("transition", "optimizer"),
                )
                for state in raw.get("states", ())
            ),
            persistent_state_bytes=raw["persistent_state_bytes"],
            schema_version=raw.get("schema_version", 1),
        )


@dataclass(frozen=True, slots=True)
class ResidualResourceBinding:
    """One signed module path for an indexed hidden-activation locator."""

    layer_index: int
    site: HiddenSite
    phase: HiddenPhase
    module_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer_index, int) or isinstance(self.layer_index, bool):
            raise TypeError("ResidualResourceBinding.layer_index must be a signed int")
        object.__setattr__(self, "site", HiddenSite(self.site))
        object.__setattr__(self, "phase", HiddenPhase(self.phase))
        if self.site is HiddenSite.NONE or self.phase is HiddenPhase.NONE:
            raise ValueError("ResidualResourceBinding requires a typed locator")
        object.__setattr__(
            self,
            "module_path",
            _name(self.module_path, "ResidualResourceBinding.module_path"),
        )

    @property
    def locator_key(self) -> tuple[int, str, str]:
        return (self.layer_index, self.site.value, self.phase.value)


@dataclass(frozen=True, slots=True)
class CompileOptions:
    memory_budget_bytes: int = 0
    allow_serial_fallback: bool = True
    shape_specialization: tuple[tuple[str, str], ...] = ()
    residual_resource_manifest: tuple[ResidualResourceBinding, ...] = ()
    # Opaque to Python after construction. Rust parses and compares this
    # closed JSON manifest before local worker resources are bound.
    static_binding_manifest: str = "[]"
    compiler_fingerprint: str = ""
    compile_policy: str = "torch_compile"
    capture_policy: str = "auto"
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_budget_bytes", _bytes(self.memory_budget_bytes, "CompileOptions.memory_budget_bytes"))
        if not isinstance(self.allow_serial_fallback, bool):
            raise TypeError("CompileOptions.allow_serial_fallback must be bool")
        if not isinstance(self.shape_specialization, tuple):
            raise TypeError("CompileOptions.shape_specialization must be tuple")
        normalized: list[tuple[str, str]] = []
        for item in self.shape_specialization:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "CompileOptions.shape_specialization must contain key/value tuples"
                )
            normalized.append(
                (
                    _name(item[0], "CompileOptions shape key"),
                    _name(item[1], "CompileOptions shape value"),
                )
            )
        if tuple(sorted(normalized)) != tuple(normalized) or len(
            {key for key, _value in normalized}
        ) != len(normalized):
            raise ValueError(
                "CompileOptions.shape_specialization must have sorted unique keys"
            )
        if not isinstance(self.residual_resource_manifest, tuple) or any(
            not isinstance(binding, ResidualResourceBinding)
            for binding in self.residual_resource_manifest
        ):
            raise TypeError(
                "CompileOptions.residual_resource_manifest must be "
                "tuple[ResidualResourceBinding, ...]"
            )
        locator_keys = tuple(
            binding.locator_key for binding in self.residual_resource_manifest
        )
        if tuple(sorted(locator_keys)) != locator_keys or len(set(locator_keys)) != len(
            locator_keys
        ):
            raise ValueError(
                "CompileOptions.residual_resource_manifest must have sorted unique locators"
            )
        if not isinstance(self.static_binding_manifest, str):
            raise TypeError("CompileOptions.static_binding_manifest must be str")
        if not isinstance(self.compiler_fingerprint, str):
            raise TypeError("CompileOptions.compiler_fingerprint must be str")
        if self.compiler_fingerprint and (
            len(self.compiler_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.compiler_fingerprint
            )
        ):
            raise ValueError(
                "CompileOptions.compiler_fingerprint must be lowercase SHA-256"
            )
        if self.compile_policy not in {"torch_compile", "eager"}:
            raise ValueError(
                "CompileOptions.compile_policy must be torch_compile or eager"
            )
        if self.capture_policy not in {"auto", "disabled", "require"}:
            raise ValueError(
                "CompileOptions.capture_policy must be auto, disabled, or require"
            )
        if self.schema_version != 1:
            raise ValueError("unsupported CompileOptions schema version")

    def to_json(self) -> str:
        return canonical_json(self)

    @classmethod
    def from_json(cls, payload: str) -> "CompileOptions":
        raw = json.loads(payload)
        raw["shape_specialization"] = tuple(
            (str(key), str(value))
            for key, value in raw.get("shape_specialization", ())
        )
        raw["residual_resource_manifest"] = tuple(
            ResidualResourceBinding(
                layer_index=binding["layer_index"],
                site=HiddenSite(binding["site"]),
                phase=HiddenPhase(binding["phase"]),
                module_path=binding["module_path"],
            )
            for binding in raw.get("residual_resource_manifest", ())
        )
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class TaskPlacement:
    task_id: str
    after_anchor: str
    implementation_id: str
    stream: str = "serial"


@dataclass(frozen=True, slots=True)
class ScheduleFence:
    fence_id: str
    source: str
    target: str
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fence_id", _name(self.fence_id, "ScheduleFence.fence_id"))
        object.__setattr__(self, "source", _name(self.source, "ScheduleFence.source"))
        object.__setattr__(self, "target", _name(self.target, "ScheduleFence.target"))
        if self.source == self.target:
            raise ValueError("ScheduleFence source and target must differ")
        if self.kind not in {"ready", "completion", "retire"}:
            raise ValueError("ScheduleFence.kind must be ready, completion, or retire")


@dataclass(frozen=True, slots=True)
class AffineRegionSchedule:
    tasks: tuple[str, ...]
    schedule: str
    dependences: str

    def __post_init__(self) -> None:
        if not self.tasks or any(
            not isinstance(task, str) or not task for task in self.tasks
        ):
            raise TypeError("AffineRegionSchedule.tasks must be non-empty tuple[str, ...]")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("AffineRegionSchedule.tasks must be unique")
        object.__setattr__(
            self, "schedule", _name(self.schedule, "AffineRegionSchedule.schedule")
        )
        object.__setattr__(
            self,
            "dependences",
            _name(self.dependences, "AffineRegionSchedule.dependences"),
        )


@dataclass(frozen=True, slots=True)
class ScheduleTree:
    kind: str
    tasks: tuple[str, ...]
    affine_regions: tuple[AffineRegionSchedule, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"sequence", "hybrid"}:
            raise ValueError("ScheduleTree.kind must be sequence or hybrid")
        if not isinstance(self.tasks, tuple) or any(
            not isinstance(task, str) or not task for task in self.tasks
        ):
            raise TypeError("ScheduleTree.tasks must be tuple[str, ...]")
        if not isinstance(self.affine_regions, tuple) or any(
            not isinstance(region, AffineRegionSchedule)
            for region in self.affine_regions
        ):
            raise TypeError(
                "ScheduleTree.affine_regions must be tuple[AffineRegionSchedule, ...]"
            )
        flattened = tuple(task for region in self.affine_regions for task in region.tasks)
        if len(set(flattened)) != len(flattened):
            raise ValueError("a task may belong to only one affine schedule region")
        if any(task not in self.tasks for task in flattened):
            raise ValueError("affine schedule region references an unknown task")
        if (self.kind == "hybrid") != bool(self.affine_regions):
            raise ValueError("hybrid ScheduleTree requires affine regions")


@dataclass(frozen=True, slots=True)
class MemoryAllocation:
    task_id: str
    size_bytes: int
    live_start: str
    live_end: str


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    budget_bytes: int
    persistent_bytes: int
    peak_bytes: int
    allocations: tuple[MemoryAllocation, ...]


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    peak_memory_bytes: int
    transfer_bytes: int
    main_path_extension_ns: int
    compute_cost: int
    materialization_bytes: int = 0
    capture_boundary_count: int = 0
    launch_group_count: int = 0
    cross_region_fence_count: int = 0
    critical_path_extension: int = 0


@dataclass(frozen=True, slots=True)
class MaterializedValue:
    materialization_id: str
    value_id: str
    version: int
    producer_region: str
    consumer_region: str
    size_bytes: int
    live_start: str
    live_end: str

    def __post_init__(self) -> None:
        for field_name in (
            "materialization_id",
            "value_id",
            "producer_region",
            "consumer_region",
            "live_start",
            "live_end",
        ):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), f"MaterializedValue.{field_name}"),
            )
        if self.producer_region == self.consumer_region:
            raise ValueError("MaterializedValue must cross an ExecutionRegion boundary")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("MaterializedValue.version must be a positive int")
        object.__setattr__(
            self, "size_bytes", _bytes(self.size_bytes, "MaterializedValue.size_bytes")
        )


@dataclass(frozen=True, slots=True)
class RegionPort:
    task_id: str
    port_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _name(self.task_id, "RegionPort.task_id"))
        object.__setattr__(
            self, "port_name", _name(self.port_name, "RegionPort.port_name")
        )


@dataclass(frozen=True, slots=True)
class RegionInputBinding:
    binding_id: str
    kind: RegionInputKind
    source: str
    consumers: tuple[RegionPort, ...]
    layer: str = ""
    layer_index: int | None = None
    layer_kind: HiddenLayerKind = HiddenLayerKind.NONE
    site: HiddenSite = HiddenSite.NONE
    phase: HiddenPhase = HiddenPhase.NONE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _name(self.binding_id, "RegionInputBinding.binding_id")
        )
        object.__setattr__(self, "kind", RegionInputKind(self.kind))
        object.__setattr__(self, "layer_kind", HiddenLayerKind(self.layer_kind))
        object.__setattr__(self, "site", HiddenSite(self.site))
        object.__setattr__(self, "phase", HiddenPhase(self.phase))
        if self.kind is RegionInputKind.HIDDEN:
            if self.layer_kind is HiddenLayerKind.NONE:
                raise ValueError("hidden RegionInputBinding requires layer_kind")
            if self.site is HiddenSite.NONE or self.phase is HiddenPhase.NONE:
                raise ValueError("hidden RegionInputBinding requires site and phase")
        elif self.layer_kind is not HiddenLayerKind.NONE:
            raise ValueError("only hidden RegionInputBinding may declare layer_kind")
        elif self.site is not HiddenSite.NONE or self.phase is not HiddenPhase.NONE:
            raise ValueError("only hidden RegionInputBinding may declare a locator")
        if self.layer_kind is HiddenLayerKind.INDEX:
            if (
                not isinstance(self.layer_index, int)
                or isinstance(self.layer_index, bool)
            ):
                raise ValueError(
                    "indexed RegionInputBinding requires a signed layer_index"
                )
        elif self.layer_index is not None:
            raise ValueError("only indexed RegionInputBinding may declare layer_index")
        object.__setattr__(
            self, "source", _name(self.source, "RegionInputBinding.source")
        )
        if not isinstance(self.consumers, tuple) or not self.consumers or any(
            not isinstance(port, RegionPort) for port in self.consumers
        ):
            raise TypeError(
                "RegionInputBinding.consumers must be a non-empty tuple[RegionPort, ...]"
            )
        if len(set(self.consumers)) != len(self.consumers):
            raise ValueError("RegionInputBinding consumers must be unique")


@dataclass(frozen=True, slots=True)
class RegionOutputBinding:
    binding_id: str
    kind: RegionOutputKind
    target: str
    producer: RegionPort
    value_binding_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _name(self.binding_id, "RegionOutputBinding.binding_id")
        )
        object.__setattr__(self, "kind", RegionOutputKind(self.kind))
        object.__setattr__(
            self, "target", _name(self.target, "RegionOutputBinding.target")
        )
        if not isinstance(self.producer, RegionPort):
            raise TypeError("RegionOutputBinding.producer must be RegionPort")
        object.__setattr__(
            self,
            "value_binding_id",
            _name(self.value_binding_id, "RegionOutputBinding.value_binding_id"),
        )


@dataclass(frozen=True, slots=True)
class RegionTaskPortBinding:
    port_name: str
    value_binding_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "port_name", _name(self.port_name, "RegionTaskPortBinding.port_name")
        )
        object.__setattr__(
            self,
            "value_binding_id",
            _name(
                self.value_binding_id,
                "RegionTaskPortBinding.value_binding_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class RegionTaskInvocation:
    task_id: str
    inputs: tuple[RegionTaskPortBinding, ...] = ()
    outputs: tuple[RegionTaskPortBinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_id", _name(self.task_id, "RegionTaskInvocation.task_id")
        )
        for field_name in ("inputs", "outputs"):
            bindings = getattr(self, field_name)
            if not isinstance(bindings, tuple) or any(
                not isinstance(binding, RegionTaskPortBinding)
                for binding in bindings
            ):
                raise TypeError(
                    f"RegionTaskInvocation.{field_name} must be "
                    "tuple[RegionTaskPortBinding, ...]"
                )
            port_names = tuple(binding.port_name for binding in bindings)
            if len(port_names) != len(set(port_names)):
                raise ValueError(
                    f"RegionTaskInvocation.{field_name} port names must be unique"
                )


@dataclass(frozen=True, slots=True)
class ExecutionRegion:
    region_id: str
    tasks: tuple[str, ...]
    after_anchor: str
    before_anchor: str
    placement_anchor: str
    implementation_kind: RegionImplementationKind
    implementation_ids: tuple[str, ...]
    compile_command: RegionCompileCommand | None = None
    required_capabilities: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    materialized_inputs: tuple[str, ...] = ()
    materialized_outputs: tuple[str, ...] = ()
    input_bindings: tuple[RegionInputBinding, ...] = ()
    output_bindings: tuple[RegionOutputBinding, ...] = ()
    task_invocations: tuple[RegionTaskInvocation, ...] = ()
    effects: tuple[ScheduleEffect, ...] = ()
    compile_safe: bool = True
    capture_safe: bool = False
    capture_mode: RegionCaptureMode = RegionCaptureMode.DISABLED
    stream: RegionStreamClass = RegionStreamClass.DEFAULT
    fallback_policy: RegionFallbackPolicy = RegionFallbackPolicy.REJECT
    artifact_key: str = ""
    serial_fallback: bool = False
    affine_schedule: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "region_id",
            "after_anchor",
            "before_anchor",
            "placement_anchor",
        ):
            object.__setattr__(
                self,
                field_name,
                _name(getattr(self, field_name), f"ExecutionRegion.{field_name}"),
            )
        if self.after_anchor == self.before_anchor:
            raise ValueError("ExecutionRegion requires a non-empty anchor window")
        if not isinstance(self.tasks, tuple) or not self.tasks or any(
            not isinstance(task, str) or not task for task in self.tasks
        ):
            raise TypeError("ExecutionRegion.tasks must be non-empty tuple[str, ...]")
        if len(self.tasks) != len(set(self.tasks)):
            raise ValueError("ExecutionRegion.tasks must be unique")
        try:
            implementation_kind = RegionImplementationKind(self.implementation_kind)
        except ValueError as exc:
            raise ValueError(
                "ExecutionRegion implementation must be compiled or extern"
            ) from exc
        object.__setattr__(self, "implementation_kind", implementation_kind)
        if self.compile_command is not None and not isinstance(
            self.compile_command, RegionCompileCommand
        ):
            raise TypeError(
                "ExecutionRegion.compile_command must be RegionCompileCommand or None"
            )
        if (
            implementation_kind is RegionImplementationKind.EXTERN
            and self.compile_command is not None
        ):
            raise ValueError("extern ExecutionRegion cannot have a compile command")
        if not isinstance(self.implementation_ids, tuple) or any(
            not isinstance(item, str) or not item for item in self.implementation_ids
        ):
            raise TypeError("ExecutionRegion.implementation_ids must be tuple[str, ...]")
        if len(self.implementation_ids) != len(self.tasks):
            raise ValueError("ExecutionRegion requires one implementation per task")
        if (
            self.implementation_kind is RegionImplementationKind.EXTERN
            and len(self.tasks) != 1
        ):
            raise ValueError("extern ExecutionRegion must be a singleton")
        for field_name in (
            "required_capabilities",
            "dependencies",
            "materialized_inputs",
            "materialized_outputs",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise TypeError(f"ExecutionRegion.{field_name} must be tuple[str, ...]")
            if len(values) != len(set(values)):
                raise ValueError(f"ExecutionRegion.{field_name} must be unique")
        if not isinstance(self.input_bindings, tuple) or any(
            not isinstance(binding, RegionInputBinding)
            for binding in self.input_bindings
        ):
            raise TypeError(
                "ExecutionRegion.input_bindings must be tuple[RegionInputBinding, ...]"
            )
        if not isinstance(self.output_bindings, tuple) or any(
            not isinstance(binding, RegionOutputBinding)
            for binding in self.output_bindings
        ):
            raise TypeError(
                "ExecutionRegion.output_bindings must be tuple[RegionOutputBinding, ...]"
            )
        if not isinstance(self.task_invocations, tuple) or any(
            not isinstance(invocation, RegionTaskInvocation)
            for invocation in self.task_invocations
        ):
            raise TypeError(
                "ExecutionRegion.task_invocations must be "
                "tuple[RegionTaskInvocation, ...]"
            )
        if tuple(invocation.task_id for invocation in self.task_invocations) != self.tasks:
            raise ValueError(
                "ExecutionRegion.task_invocations must cover Region tasks in order"
            )
        if not isinstance(self.effects, tuple) or any(
            not isinstance(effect, ScheduleEffect) for effect in self.effects
        ):
            raise TypeError("ExecutionRegion.effects must be tuple[ScheduleEffect, ...]")
        binding_ids = tuple(
            binding.binding_id
            for binding in (*self.input_bindings, *self.output_bindings)
        )
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("ExecutionRegion binding ids must be unique")
        if tuple(sorted(self.required_capabilities)) != self.required_capabilities:
            raise ValueError("ExecutionRegion.required_capabilities must be sorted")
        for field_name in ("compile_safe", "capture_safe", "serial_fallback"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"ExecutionRegion.{field_name} must be bool")
        if (
            self.implementation_kind is RegionImplementationKind.EXTERN
            and self.compile_safe
        ):
            raise ValueError("extern ExecutionRegion cannot be compile safe")
        if self.capture_safe and not self.compile_safe:
            raise ValueError("capture-safe ExecutionRegion must be compile safe")
        try:
            capture_mode = RegionCaptureMode(self.capture_mode)
        except ValueError as exc:
            raise ValueError(
                "ExecutionRegion.capture_mode must be disabled or require"
            ) from exc
        object.__setattr__(self, "capture_mode", capture_mode)
        if capture_mode is RegionCaptureMode.REQUIRE and not self.capture_safe:
            raise ValueError("capture-required ExecutionRegion must be capture safe")
        try:
            stream = RegionStreamClass(self.stream)
        except ValueError as exc:
            raise ValueError("ExecutionRegion.stream must be default") from exc
        object.__setattr__(self, "stream", stream)
        try:
            fallback_policy = RegionFallbackPolicy(self.fallback_policy)
        except ValueError as exc:
            raise ValueError("ExecutionRegion.fallback_policy must be reject") from exc
        object.__setattr__(self, "fallback_policy", fallback_policy)
        if self.compile_command is not None and (
            self.compile_command.fallback_policy is not fallback_policy
        ):
            raise ValueError(
                "ExecutionRegion compile command fallback policy must match Region fallback"
            )
        if (
            len(self.artifact_key) != 64
            or any(character not in "0123456789abcdef" for character in self.artifact_key)
        ):
            raise ValueError("ExecutionRegion.artifact_key must be lowercase SHA-256")
        if not isinstance(self.affine_schedule, str):
            raise TypeError("ExecutionRegion.affine_schedule must be str")


@dataclass(frozen=True, slots=True)
class BackendRegionBinderStage:
    """One signed backend binding staged at an observable backend event."""

    backend_event: BackendObservableEvent
    region_id: str
    implementation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_event", BackendObservableEvent(self.backend_event)
        )
        object.__setattr__(
            self, "region_id", _name(self.region_id, "BackendRegionBinderStage.region_id")
        )
        object.__setattr__(
            self,
            "implementation_id",
            _name(
                self.implementation_id,
                "BackendRegionBinderStage.implementation_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeAdvancePlan:
    """Signed backend-event advancement for one observable timeline anchor."""

    backend_event: BackendObservableEvent
    anchor_id: str
    region_ids: tuple[str, ...]
    backend_region_binder_stages: tuple[BackendRegionBinderStage, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_event", BackendObservableEvent(self.backend_event)
        )
        object.__setattr__(
            self, "anchor_id", _name(self.anchor_id, "RuntimeAdvancePlan.anchor_id")
        )
        if not isinstance(self.region_ids, tuple) or any(
            not isinstance(region_id, str) or not region_id
            for region_id in self.region_ids
        ):
            raise TypeError("RuntimeAdvancePlan.region_ids must be tuple[str, ...]")
        if len(self.region_ids) != len(set(self.region_ids)):
            raise ValueError("RuntimeAdvancePlan.region_ids must be unique")
        if not isinstance(self.backend_region_binder_stages, tuple) or any(
            not isinstance(stage, BackendRegionBinderStage)
            for stage in self.backend_region_binder_stages
        ):
            raise TypeError(
                "RuntimeAdvancePlan.backend_region_binder_stages must be "
                "tuple[BackendRegionBinderStage, ...]"
            )
        if any(
            stage.backend_event is not self.backend_event
            or stage.region_id not in self.region_ids
            for stage in self.backend_region_binder_stages
        ):
            raise ValueError(
                "RuntimeAdvancePlan binder stages must match its event and regions"
            )
        stage_ids = tuple(
            (stage.region_id, stage.implementation_id)
            for stage in self.backend_region_binder_stages
        )
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("RuntimeAdvancePlan backend binder stages must be unique")


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    graph_key: str
    timeline_key: str
    execution_order: tuple[str, ...]
    placements: tuple[TaskPlacement, ...]
    schedule_tree: ScheduleTree
    memory_plan: MemoryPlan
    execution_regions: tuple[ExecutionRegion, ...] = ()
    runtime_advance_plan: tuple[RuntimeAdvancePlan, ...] = ()
    completion_anchor: str = ""
    materializations: tuple[MaterializedValue, ...] = ()
    moves: tuple[object, ...] = ()
    fences: tuple[ScheduleFence, ...] = ()
    capture_partitions: tuple[tuple[str, ...], ...] = ()
    fallback: str = "serial"
    cost: CostBreakdown = CostBreakdown(0, 0, 0, 0)
    explanations: tuple[str, ...] = ()
    signature: str = ""
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError("unsupported ScheduleDecision schema version")
        if not isinstance(self.execution_regions, tuple) or any(
            not isinstance(region, ExecutionRegion) for region in self.execution_regions
        ):
            raise TypeError(
                "ScheduleDecision.execution_regions must be tuple[ExecutionRegion, ...]"
            )
        if not isinstance(self.materializations, tuple) or any(
            not isinstance(value, MaterializedValue) for value in self.materializations
        ):
            raise TypeError(
                "ScheduleDecision.materializations must be tuple[MaterializedValue, ...]"
            )
        if self.schema_version == 1:
            if (
                self.execution_regions
                or self.runtime_advance_plan
                or self.completion_anchor
                or self.materializations
            ):
                raise ValueError("legacy ScheduleDecision cannot contain ExecutionRegion data")
        else:
            if not isinstance(self.runtime_advance_plan, tuple) or any(
                not isinstance(advance, RuntimeAdvancePlan)
                for advance in self.runtime_advance_plan
            ):
                raise TypeError(
                    "ScheduleDecision.runtime_advance_plan must be "
                    "tuple[RuntimeAdvancePlan, ...]"
                )
            object.__setattr__(
                self,
                "completion_anchor",
                _name(self.completion_anchor, "ScheduleDecision.completion_anchor"),
            )
            region_ids = tuple(region.region_id for region in self.execution_regions)
            if len(region_ids) != len(set(region_ids)):
                raise ValueError("ScheduleDecision region ids must be unique")
            flattened = tuple(
                task for region in self.execution_regions for task in region.tasks
            )
            if flattened != self.execution_order:
                raise ValueError(
                    "ScheduleDecision regions must cover execution_order exactly once"
                )
            positions = {
                region_id: position for position, region_id in enumerate(region_ids)
            }
            for position, region in enumerate(self.execution_regions):
                if (
                    region.implementation_kind is RegionImplementationKind.COMPILED
                ) != (region.compile_command is not None):
                    raise ValueError(
                        "compiled ExecutionRegion requires a compile command and extern Region forbids one"
                    )
                if any(
                    dependency not in positions or positions[dependency] >= position
                    for dependency in region.dependencies
                ):
                    raise ValueError(
                        "ExecutionRegion dependencies must reference earlier regions"
                    )
            materialization_ids = tuple(
                value.materialization_id for value in self.materializations
            )
            if len(materialization_ids) != len(set(materialization_ids)):
                raise ValueError("ScheduleDecision materialization ids must be unique")
            for value in self.materializations:
                if (
                    value.producer_region not in positions
                    or value.consumer_region not in positions
                    or positions[value.producer_region]
                    >= positions[value.consumer_region]
                ):
                    raise ValueError(
                        "ScheduleDecision materialization references invalid regions"
                    )
            binder_identities: set[tuple[BackendObservableEvent, str]] = set()
            for advance in self.runtime_advance_plan:
                if any(region_id not in positions for region_id in advance.region_ids):
                    raise ValueError(
                        "RuntimeAdvancePlan references an unknown ExecutionRegion"
                    )
                for stage in advance.backend_region_binder_stages:
                    region = self.execution_regions[positions[stage.region_id]]
                    if (
                        region.implementation_kind is not RegionImplementationKind.EXTERN
                        or "backend/region_binder" not in region.required_capabilities
                        or stage.implementation_id not in region.implementation_ids
                    ):
                        raise ValueError(
                            "RuntimeAdvancePlan binder stage does not match its "
                            "signed backend binder Region"
                        )
                    identity = (stage.backend_event, stage.implementation_id)
                    if identity in binder_identities:
                        raise ValueError(
                            "ScheduleDecision backend binder identities must be unique per event"
                        )
                    binder_identities.add(identity)
            if self.capture_partitions:
                raise ValueError(
                    "ScheduleDecision schema v2 cannot contain capture_partitions"
                )
        if not isinstance(self.fences, tuple) or any(
            not isinstance(fence, ScheduleFence) for fence in self.fences
        ):
            raise TypeError("ScheduleDecision.fences must be tuple[ScheduleFence, ...]")
        if len({fence.fence_id for fence in self.fences}) != len(self.fences):
            raise ValueError("ScheduleDecision fence ids must be unique")
        if self.signature and (
            len(self.signature) != 64
            or any(character not in "0123456789abcdef" for character in self.signature)
        ):
            raise ValueError("ScheduleDecision.signature must be lowercase SHA-256")

    def to_json(self) -> str:
        payload = _canonical(self)
        assert isinstance(payload, dict)
        if self.schema_version == 1:
            payload.pop("execution_regions")
            payload.pop("runtime_advance_plan")
            payload.pop("completion_anchor")
            payload.pop("materializations")
            cost = payload["cost"]
            assert isinstance(cost, dict)
            for key in (
                "materialization_bytes",
                "capture_boundary_count",
                "launch_group_count",
                "cross_region_fence_count",
                "critical_path_extension",
            ):
                cost.pop(key)
        payload["signature"] = self.signature
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, payload: str) -> "ScheduleDecision":
        raw = json.loads(payload)
        if not isinstance(raw, Mapping):
            raise TypeError("ScheduleDecision must be a JSON object")
        if "schema_version" not in raw:
            raise ValueError("ScheduleDecision requires an explicit schema_version")
        schema_version = raw["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise TypeError("ScheduleDecision.schema_version must be int")

        is_v2 = schema_version == 2
        if is_v2:
            _require_json_fields(
                raw,
                "ScheduleDecision",
                (
                    "execution_regions",
                    "runtime_advance_plan",
                    "completion_anchor",
                    "materializations",
                ),
            )
        region_payloads = _json_array(
            raw["execution_regions"]
            if is_v2
            else raw.get("execution_regions", []),
            "ScheduleDecision.execution_regions",
        )
        materialization_payloads = _json_array(
            raw["materializations"] if is_v2 else raw.get("materializations", []),
            "ScheduleDecision.materializations",
        )
        runtime_advance_payloads = _json_array(
            raw["runtime_advance_plan"]
            if is_v2
            else raw.get("runtime_advance_plan", []),
            "ScheduleDecision.runtime_advance_plan",
        )

        region_fields = (
            "region_id",
            "tasks",
            "after_anchor",
            "before_anchor",
            "placement_anchor",
            "implementation_kind",
            "implementation_ids",
            "compile_command",
            "required_capabilities",
            "dependencies",
            "materialized_inputs",
            "materialized_outputs",
            "input_bindings",
            "output_bindings",
            "task_invocations",
            "effects",
            "compile_safe",
            "capture_safe",
            "capture_mode",
            "stream",
            "fallback_policy",
            "artifact_key",
            "serial_fallback",
            "affine_schedule",
        )
        input_binding_fields = (
            "binding_id",
            "kind",
            "source",
            "consumers",
            "layer",
            "layer_index",
            "layer_kind",
            "site",
            "phase",
        )
        output_binding_fields = (
            "binding_id",
            "kind",
            "target",
            "producer",
            "value_binding_id",
        )
        port_fields = ("task_id", "port_name")
        invocation_fields = ("task_id", "inputs", "outputs")
        task_port_fields = ("port_name", "value_binding_id")
        compile_command_fields = (
            "adapter",
            "fullgraph",
            "dynamic",
            "shape_specialization",
            "autograd_policy",
            "fallback_policy",
        )
        effect_fields = ("kind", "resource")
        binder_stage_fields = ("backend_event", "region_id", "implementation_id")
        runtime_advance_fields = (
            "backend_event",
            "anchor_id",
            "region_ids",
            "backend_region_binder_stages",
        )
        decision_fields = (
            "graph_key",
            "timeline_key",
            "execution_order",
            "placements",
            "schedule_tree",
            "memory_plan",
            "execution_regions",
            "runtime_advance_plan",
            "completion_anchor",
            "materializations",
            "moves",
            "fences",
            "capture_partitions",
            "fallback",
            "cost",
            "explanations",
            "signature",
            "schema_version",
        )
        placement_fields = (
            "task_id",
            "after_anchor",
            "implementation_id",
            "stream",
        )
        schedule_tree_fields = ("kind", "tasks", "affine_regions")
        affine_region_fields = ("tasks", "schedule", "dependences")
        memory_plan_fields = (
            "budget_bytes",
            "persistent_bytes",
            "peak_bytes",
            "allocations",
        )
        allocation_fields = ("task_id", "size_bytes", "live_start", "live_end")
        fence_fields = ("fence_id", "source", "target", "kind")
        cost_fields = (
            "peak_memory_bytes",
            "transfer_bytes",
            "main_path_extension_ns",
            "compute_cost",
            "materialization_bytes",
            "capture_boundary_count",
            "launch_group_count",
            "cross_region_fence_count",
            "critical_path_extension",
        )
        materialization_fields = (
            "materialization_id",
            "value_id",
            "version",
            "producer_region",
            "consumer_region",
            "size_bytes",
            "live_start",
            "live_end",
        )

        def require_v2_object(
            raw_value: object, label: str, field_names: tuple[str, ...]
        ) -> Mapping[str, object]:
            return _require_json_fields(
                raw_value,
                label,
                field_names,
                allowed=field_names,
            )

        if is_v2:
            require_v2_object(raw, "ScheduleDecision", decision_fields)
            for position, placement in enumerate(
                _json_array(raw["placements"], "ScheduleDecision.placements")
            ):
                require_v2_object(
                    placement,
                    f"TaskPlacement[{position}]",
                    placement_fields,
                )
            tree_raw = require_v2_object(
                raw["schedule_tree"], "ScheduleTree", schedule_tree_fields
            )
            for position, affine_region in enumerate(
                _json_array(tree_raw["affine_regions"], "ScheduleTree.affine_regions")
            ):
                require_v2_object(
                    affine_region,
                    f"AffineRegionSchedule[{position}]",
                    affine_region_fields,
                )
            memory_raw = require_v2_object(
                raw["memory_plan"], "MemoryPlan", memory_plan_fields
            )
            for position, allocation in enumerate(
                _json_array(memory_raw["allocations"], "MemoryPlan.allocations")
            ):
                require_v2_object(
                    allocation,
                    f"MemoryAllocation[{position}]",
                    allocation_fields,
                )
            for position, fence in enumerate(
                _json_array(raw["fences"], "ScheduleDecision.fences")
            ):
                require_v2_object(fence, f"ScheduleFence[{position}]", fence_fields)
            require_v2_object(raw["cost"], "CostBreakdown", cost_fields)
            for position, materialization in enumerate(materialization_payloads):
                require_v2_object(
                    materialization,
                    f"MaterializedValue[{position}]",
                    materialization_fields,
                )

        regions = tuple(
            require_v2_object(region, f"ExecutionRegion[{position}]", region_fields)
            if is_v2
            else _require_json_fields(region, f"ExecutionRegion[{position}]", ())
            for position, region in enumerate(region_payloads)
        )

        runtime_advances = tuple(
            require_v2_object(
                advance,
                f"RuntimeAdvancePlan[{position}]",
                runtime_advance_fields,
            )
            if is_v2
            else _require_json_fields(advance, f"RuntimeAdvancePlan[{position}]", ())
            for position, advance in enumerate(runtime_advance_payloads)
        )

        if is_v2:
            for position, advance in enumerate(runtime_advances):
                for stage_position, stage in enumerate(
                    _json_array(
                        advance["backend_region_binder_stages"],
                        "RuntimeAdvancePlan"
                        f"[{position}].backend_region_binder_stages",
                    )
                ):
                    require_v2_object(
                        stage,
                        "RuntimeAdvancePlan"
                        f"[{position}].backend_region_binder_stages[{stage_position}]",
                        binder_stage_fields,
                    )
            for position, region in enumerate(regions):
                for binding_position, binding_raw in enumerate(
                    _json_array(
                        region["input_bindings"],
                        f"ExecutionRegion[{position}].input_bindings",
                    )
                ):
                    binding = require_v2_object(
                        binding_raw,
                        f"ExecutionRegion[{position}].input_bindings[{binding_position}]",
                        input_binding_fields,
                    )
                    for port_position, port in enumerate(
                        _json_array(
                            binding["consumers"],
                            "ExecutionRegion"
                            f"[{position}].input_bindings[{binding_position}].consumers",
                        )
                    ):
                        require_v2_object(
                            port,
                            "ExecutionRegion"
                            f"[{position}].input_bindings[{binding_position}]"
                            f".consumers[{port_position}]",
                            port_fields,
                        )
                for binding_position, binding_raw in enumerate(
                    _json_array(
                        region["output_bindings"],
                        f"ExecutionRegion[{position}].output_bindings",
                    )
                ):
                    binding = require_v2_object(
                        binding_raw,
                        f"ExecutionRegion[{position}].output_bindings[{binding_position}]",
                        output_binding_fields,
                    )
                    require_v2_object(
                        binding["producer"],
                        "ExecutionRegion"
                        f"[{position}].output_bindings[{binding_position}].producer",
                        port_fields,
                    )
                for invocation_position, invocation_raw in enumerate(
                    _json_array(
                        region["task_invocations"],
                        f"ExecutionRegion[{position}].task_invocations",
                    )
                ):
                    invocation = require_v2_object(
                        invocation_raw,
                        "ExecutionRegion"
                        f"[{position}].task_invocations[{invocation_position}]",
                        invocation_fields,
                    )
                    for direction in ("inputs", "outputs"):
                        for binding_position, binding in enumerate(
                            _json_array(
                                invocation[direction],
                                "ExecutionRegion"
                                f"[{position}].task_invocations[{invocation_position}]"
                                f".{direction}",
                            )
                        ):
                            require_v2_object(
                                binding,
                                "ExecutionRegion"
                                f"[{position}].task_invocations[{invocation_position}]"
                                f".{direction}[{binding_position}]",
                                task_port_fields,
                            )
                for effect_position, effect in enumerate(
                    _json_array(
                        region["effects"], f"ExecutionRegion[{position}].effects"
                    )
                ):
                    require_v2_object(
                        effect,
                        f"ExecutionRegion[{position}].effects[{effect_position}]",
                        effect_fields,
                    )

        def region_value(
            region: Mapping[str, object], field_name: str, legacy_default: object
        ) -> object:
            return region[field_name] if is_v2 else region.get(field_name, legacy_default)

        def nested_value(
            raw_value: Mapping[str, object], field_name: str, legacy_default: object
        ) -> object:
            return raw_value[field_name] if is_v2 else raw_value.get(field_name, legacy_default)

        def region_array(
            region: Mapping[str, object], field_name: str
        ) -> tuple[object, ...]:
            return _json_array(
                region_value(region, field_name, []),
                f"ExecutionRegion.{field_name}",
            )

        def nested_array(
            raw_value: Mapping[str, object], field_name: str
        ) -> tuple[object, ...]:
            return _json_array(
                nested_value(raw_value, field_name, []),
                f"ExecutionRegion.{field_name}",
            )

        def region_port(raw_value: object, label: str) -> RegionPort:
            port = _require_json_fields(raw_value, label, port_fields)
            return RegionPort(
                task_id=_json_string(port["task_id"], f"{label}.task_id"),
                port_name=_json_string(port["port_name"], f"{label}.port_name"),
            )

        def input_binding(raw_value: object, label: str) -> RegionInputBinding:
            binding = _require_json_fields(
                raw_value,
                label,
                ("binding_id", "kind", "source", "consumers"),
            )
            return RegionInputBinding(
                binding_id=_json_string(binding["binding_id"], f"{label}.binding_id"),
                kind=RegionInputKind(_json_string(binding["kind"], f"{label}.kind")),
                source=_json_string(binding["source"], f"{label}.source"),
                consumers=tuple(
                    region_port(
                        port, f"{label}.consumers[{position}]"
                    )
                    for position, port in enumerate(
                        _json_array(binding["consumers"], f"{label}.consumers")
                    )
                ),
                layer=_json_string(
                    nested_value(binding, "layer", ""), f"{label}.layer"
                ),
                layer_index=_json_optional_int(
                    nested_value(binding, "layer_index", None),
                    f"{label}.layer_index",
                ),
                layer_kind=HiddenLayerKind(
                    _json_string(
                        nested_value(binding, "layer_kind", "none"),
                        f"{label}.layer_kind",
                    )
                ),
                site=HiddenSite(
                    _json_string(
                        nested_value(binding, "site", ""), f"{label}.site"
                    )
                ),
                phase=HiddenPhase(
                    _json_string(
                        nested_value(binding, "phase", ""), f"{label}.phase"
                    )
                ),
            )

        def output_binding(raw_value: object, label: str) -> RegionOutputBinding:
            binding = _require_json_fields(raw_value, label, output_binding_fields)
            return RegionOutputBinding(
                binding_id=_json_string(binding["binding_id"], f"{label}.binding_id"),
                kind=RegionOutputKind(_json_string(binding["kind"], f"{label}.kind")),
                target=_json_string(binding["target"], f"{label}.target"),
                producer=region_port(binding["producer"], f"{label}.producer"),
                value_binding_id=_json_string(
                    binding["value_binding_id"], f"{label}.value_binding_id"
                ),
            )

        def task_port_binding(raw_value: object, label: str) -> RegionTaskPortBinding:
            binding = _require_json_fields(raw_value, label, task_port_fields)
            return RegionTaskPortBinding(
                port_name=_json_string(binding["port_name"], f"{label}.port_name"),
                value_binding_id=_json_string(
                    binding["value_binding_id"], f"{label}.value_binding_id"
                ),
            )

        def task_invocation(raw_value: object, label: str) -> RegionTaskInvocation:
            invocation = _require_json_fields(raw_value, label, ("task_id",))
            return RegionTaskInvocation(
                task_id=_json_string(invocation["task_id"], f"{label}.task_id"),
                inputs=tuple(
                    task_port_binding(port, f"{label}.inputs[{position}]")
                    for position, port in enumerate(nested_array(invocation, "inputs"))
                ),
                outputs=tuple(
                    task_port_binding(port, f"{label}.outputs[{position}]")
                    for position, port in enumerate(nested_array(invocation, "outputs"))
                ),
            )

        def schedule_effect(raw_value: object, label: str) -> ScheduleEffect:
            effect = _require_json_fields(raw_value, label, effect_fields)
            return ScheduleEffect(
                kind=ScheduleEffectKind(_json_string(effect["kind"], f"{label}.kind")),
                resource=_json_string(effect["resource"], f"{label}.resource"),
            )

        def region_capture_mode(raw_value: object, label: str) -> RegionCaptureMode:
            try:
                return RegionCaptureMode(_json_string(raw_value, label))
            except ValueError as exc:
                raise ValueError(
                    "ExecutionRegion.capture_mode must be disabled or require"
                ) from exc

        def region_stream(raw_value: object, label: str) -> RegionStreamClass:
            try:
                return RegionStreamClass(_json_string(raw_value, label))
            except ValueError as exc:
                raise ValueError("ExecutionRegion.stream must be default") from exc

        def region_fallback_policy(
            raw_value: object, label: str
        ) -> RegionFallbackPolicy:
            try:
                return RegionFallbackPolicy(_json_string(raw_value, label))
            except ValueError as exc:
                raise ValueError(
                    "ExecutionRegion.fallback_policy must be reject"
                ) from exc

        def region_compile_command(
            raw_value: object, label: str
        ) -> RegionCompileCommand | None:
            if raw_value is None:
                return None
            command = _require_json_fields(
                raw_value, label, compile_command_fields, allowed=compile_command_fields
            )
            shape_specialization: list[tuple[str, str]] = []
            for position, pair in enumerate(
                _json_array(command["shape_specialization"], f"{label}.shape_specialization")
            ):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise TypeError(
                        f"{label}.shape_specialization[{position}] must be a JSON pair"
                    )
                shape_specialization.append(
                    (
                        _json_string(
                            pair[0],
                            f"{label}.shape_specialization[{position}][0]",
                        ),
                        _json_string(
                            pair[1],
                            f"{label}.shape_specialization[{position}][1]",
                        ),
                    )
                )
            return RegionCompileCommand(
                adapter=RegionCompilerAdapter(
                    _json_string(command["adapter"], f"{label}.adapter")
                ),
                fullgraph=_json_bool(command["fullgraph"], f"{label}.fullgraph"),
                dynamic=_json_bool(command["dynamic"], f"{label}.dynamic"),
                shape_specialization=tuple(shape_specialization),
                autograd_policy=RegionAutogradPolicy(
                    _json_string(command["autograd_policy"], f"{label}.autograd_policy")
                ),
                fallback_policy=RegionFallbackPolicy(
                    _json_string(command["fallback_policy"], f"{label}.fallback_policy")
                ),
            )

        def runtime_advance(
            raw_value: Mapping[str, object], position: int
        ) -> RuntimeAdvancePlan:
            label = f"RuntimeAdvancePlan[{position}]"
            return RuntimeAdvancePlan(
                backend_event=BackendObservableEvent(
                    _json_string(raw_value["backend_event"], f"{label}.backend_event")
                ),
                anchor_id=_json_string(raw_value["anchor_id"], f"{label}.anchor_id"),
                region_ids=_json_strings(raw_value["region_ids"], f"{label}.region_ids"),
                backend_region_binder_stages=tuple(
                    BackendRegionBinderStage(
                        backend_event=BackendObservableEvent(
                            _json_string(
                                _require_json_fields(
                                    stage,
                                    f"{label}.backend_region_binder_stages[{stage_position}]",
                                    binder_stage_fields,
                                )["backend_event"],
                                f"{label}.backend_region_binder_stages[{stage_position}].backend_event",
                            )
                        ),
                        region_id=_json_string(
                            _require_json_fields(
                                stage,
                                f"{label}.backend_region_binder_stages[{stage_position}]",
                                binder_stage_fields,
                            )["region_id"],
                            f"{label}.backend_region_binder_stages[{stage_position}].region_id",
                        ),
                        implementation_id=_json_string(
                            _require_json_fields(
                                stage,
                                f"{label}.backend_region_binder_stages[{stage_position}]",
                                binder_stage_fields,
                            )["implementation_id"],
                            f"{label}.backend_region_binder_stages[{stage_position}].implementation_id",
                        ),
                    )
                    for stage_position, stage in enumerate(
                        _json_array(
                            raw_value["backend_region_binder_stages"],
                            f"{label}.backend_region_binder_stages",
                        )
                    )
                ),
            )

        def execution_region(
            region: Mapping[str, object], position: int
        ) -> ExecutionRegion:
            label = f"ExecutionRegion[{position}]"
            return ExecutionRegion(
                region_id=_json_string(region["region_id"], f"{label}.region_id"),
                tasks=_json_strings(region["tasks"], f"{label}.tasks"),
                after_anchor=_json_string(
                    region["after_anchor"], f"{label}.after_anchor"
                ),
                before_anchor=_json_string(
                    region["before_anchor"], f"{label}.before_anchor"
                ),
                placement_anchor=_json_string(
                    region["placement_anchor"], f"{label}.placement_anchor"
                ),
                implementation_kind=RegionImplementationKind(
                    _json_string(region["implementation_kind"], f"{label}.implementation_kind")
                ),
                implementation_ids=_json_strings(
                    region["implementation_ids"], f"{label}.implementation_ids"
                ),
                compile_command=region_compile_command(
                    region_value(region, "compile_command", None),
                    f"{label}.compile_command",
                ),
                required_capabilities=_json_strings(
                    region_value(region, "required_capabilities", []),
                    f"{label}.required_capabilities",
                ),
                dependencies=_json_strings(
                    region_value(region, "dependencies", []), f"{label}.dependencies"
                ),
                materialized_inputs=_json_strings(
                    region_value(region, "materialized_inputs", []),
                    f"{label}.materialized_inputs",
                ),
                materialized_outputs=_json_strings(
                    region_value(region, "materialized_outputs", []),
                    f"{label}.materialized_outputs",
                ),
                input_bindings=tuple(
                    input_binding(binding, f"{label}.input_bindings[{binding_position}]")
                    for binding_position, binding in enumerate(
                        region_array(region, "input_bindings")
                    )
                ),
                output_bindings=tuple(
                    output_binding(binding, f"{label}.output_bindings[{binding_position}]")
                    for binding_position, binding in enumerate(
                        region_array(region, "output_bindings")
                    )
                ),
                task_invocations=tuple(
                    task_invocation(
                        invocation, f"{label}.task_invocations[{invocation_position}]"
                    )
                    for invocation_position, invocation in enumerate(
                        region_array(region, "task_invocations")
                    )
                ),
                effects=tuple(
                    schedule_effect(effect, f"{label}.effects[{effect_position}]")
                    for effect_position, effect in enumerate(region_array(region, "effects"))
                ),
                compile_safe=_json_bool(
                    region_value(region, "compile_safe", True), f"{label}.compile_safe"
                ),
                capture_safe=_json_bool(
                    region_value(region, "capture_safe", False), f"{label}.capture_safe"
                ),
                capture_mode=region_capture_mode(
                    region_value(
                        region, "capture_mode", RegionCaptureMode.DISABLED.value
                    ),
                    f"{label}.capture_mode",
                ),
                stream=region_stream(
                    region_value(region, "stream", RegionStreamClass.DEFAULT.value),
                    f"{label}.stream",
                ),
                fallback_policy=region_fallback_policy(
                    region_value(
                        region,
                        "fallback_policy",
                        RegionFallbackPolicy.REJECT.value,
                    ),
                    f"{label}.fallback_policy",
                ),
                artifact_key=_json_string(
                    region["artifact_key"], f"{label}.artifact_key"
                ),
                serial_fallback=_json_bool(
                    region_value(region, "serial_fallback", False),
                    f"{label}.serial_fallback",
                ),
                affine_schedule=_json_string(
                    region_value(region, "affine_schedule", ""),
                    f"{label}.affine_schedule",
                ),
            )

        def materialized_value(raw_value: object, position: int) -> MaterializedValue:
            label = f"MaterializedValue[{position}]"
            value = _require_json_fields(
                raw_value,
                label,
                (
                    "materialization_id",
                    "value_id",
                    "version",
                    "producer_region",
                    "consumer_region",
                    "size_bytes",
                    "live_start",
                    "live_end",
                ),
            )
            return MaterializedValue(
                materialization_id=_json_string(
                    value["materialization_id"], f"{label}.materialization_id"
                ),
                value_id=_json_string(value["value_id"], f"{label}.value_id"),
                version=_json_int(value["version"], f"{label}.version"),
                producer_region=_json_string(
                    value["producer_region"], f"{label}.producer_region"
                ),
                consumer_region=_json_string(
                    value["consumer_region"], f"{label}.consumer_region"
                ),
                size_bytes=_json_int(value["size_bytes"], f"{label}.size_bytes"),
                live_start=_json_string(value["live_start"], f"{label}.live_start"),
                live_end=_json_string(value["live_end"], f"{label}.live_end"),
            )

        placements = tuple(TaskPlacement(**item) for item in raw["placements"])
        fences = tuple(ScheduleFence(**item) for item in raw.get("fences", ()))
        tree = ScheduleTree(
            kind=raw["schedule_tree"]["kind"],
            tasks=tuple(raw["schedule_tree"]["tasks"]),
            affine_regions=tuple(
                AffineRegionSchedule(
                    tasks=tuple(region["tasks"]),
                    schedule=region["schedule"],
                    dependences=region["dependences"],
                )
                for region in raw["schedule_tree"].get("affine_regions", ())
            ),
        )
        memory_raw = raw["memory_plan"]
        memory = MemoryPlan(
            budget_bytes=memory_raw["budget_bytes"],
            persistent_bytes=memory_raw["persistent_bytes"],
            peak_bytes=memory_raw["peak_bytes"],
            allocations=tuple(MemoryAllocation(**item) for item in memory_raw["allocations"]),
        )
        cost = CostBreakdown(**raw["cost"])
        return cls(
            graph_key=raw["graph_key"],
            timeline_key=raw["timeline_key"],
            execution_order=tuple(raw["execution_order"]),
            placements=placements,
            schedule_tree=tree,
            memory_plan=memory,
            execution_regions=tuple(
                execution_region(region, position)
                for position, region in enumerate(regions)
            ),
            runtime_advance_plan=tuple(
                runtime_advance(advance, position)
                for position, advance in enumerate(runtime_advances)
            ),
            completion_anchor=(
                _json_string(raw["completion_anchor"], "ScheduleDecision.completion_anchor")
                if is_v2
                else _json_string(raw.get("completion_anchor", ""), "ScheduleDecision.completion_anchor")
            ),
            materializations=tuple(
                materialized_value(value, position)
                for position, value in enumerate(materialization_payloads)
            ),
            moves=tuple(raw.get("moves", ())),
            fences=fences,
            capture_partitions=tuple(tuple(group) for group in raw.get("capture_partitions", ())),
            fallback=raw["fallback"],
            cost=cost,
            explanations=tuple(raw.get("explanations", ())),
            signature=raw["signature"],
            schema_version=schema_version,
        )


__all__ = [
    "AffineRegionSchedule",
    "BackendObservableEvent",
    "BackendRegionBinderStage",
    "CompileOptions",
    "CostBreakdown",
    "ExecutionRegion",
    "HiddenPhase",
    "HiddenLayerKind",
    "HiddenSite",
    "MainWorkloadRegion",
    "MainWorkloadTimeline",
    "MemoryAllocation",
    "MemoryPlan",
    "MaterializedValue",
    "RegionInputBinding",
    "RegionInputKind",
    "RegionImplementationKind",
    "RegionCompilerAdapter",
    "RegionAutogradPolicy",
    "RegionCompileCommand",
    "RegionCaptureMode",
    "RegionStreamClass",
    "RegionFallbackPolicy",
    "RegionOutputBinding",
    "RegionOutputKind",
    "RegionPort",
    "RegionTaskInvocation",
    "RegionTaskPortBinding",
    "ResidualResourceBinding",
    "ResourceReservation",
    "RuntimeAdvancePlan",
    "ScheduleAffineAccess",
    "ScheduleAffineExpr",
    "ScheduleDecision",
    "ScheduleDomain",
    "ScheduleEdge",
    "ScheduleEdgeKind",
    "ScheduleEffect",
    "ScheduleEffectKind",
    "ScheduleInput",
    "ScheduleInputKind",
    "ScheduleOutput",
    "ScheduleOutputKind",
    "ScheduleState",
    "ScheduleFence",
    "ScheduleGraph",
    "ScheduleTask",
    "ScheduleTree",
    "ScheduleValue",
    "TaskPlacement",
    "TaskTiming",
    "TimelineAnchor",
    "canonical_json",
    "structural_key",
]
