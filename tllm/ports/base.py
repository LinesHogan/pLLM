#!/usr/bin/env python3
"""Core public port contracts for consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from enum import StrEnum
from typing import Literal, Sequence


class PortKind(str, Enum):
    RESIDUAL_STREAM = "residual_stream"
    KV_CACHE = "kv_cache"
    LOGITS = "logits"
    SAMPLER = "sampler"
    TOKEN_TARGET = "token_target"
    REQUEST_META = "request_meta"
    CPU_EXPORT = "cpu_export"
    GENERATION_TRACE = "generation_trace"
    REWARD_OUTCOME = "reward_outcome"
    RETRY_CONTROL = "retry_control"
    UPDATE_BARRIER = "update_barrier"
    PROMPT_CONTEXT = "prompt_context"
    TRAJECTORY_EVENT = "trajectory_event"
    MLP_ACTIVATION = "mlp_activation"
    HIDDEN_TARGET = "hidden_target"
    MODULE_WEIGHT_VIEW = "module_weight_view"


class FlowWindow(StrEnum):
    BACKGROUND = "background"
    SAME_STEP = "same_step"
    NEXT_STEP = "next_step"
    OUT_OF_BAND = "out_of_band"


class FlowDelivery(StrEnum):
    BUNDLE = "bundle"
    DEVICE_LEASE = "device_lease"


class BufferOwnership(StrEnum):
    BORROWED = "borrowed"
    RUNTIME_LEASE = "runtime_lease"


class RowSelection(StrEnum):
    ALL = "none"
    FIRST_PER_PROMPT = "first_per_prompt"


class FlowFusion(StrEnum):
    NONE = "none"
    SIDE_STREAM_CUDAGRAPH = "side_stream_cudagraph"


class WriteMode(StrEnum):
    """How a consumer write mutates its target port.

    ``WRAPPER`` is the escape hatch: the consumer installs its own forward
    wrapper through shared runtime infra (e.g. fast-weight overlays) instead
    of staging a value through the write channel.
    """

    REPLACE = "replace"
    ADD = "add"
    APPEND = "append"
    WRAPPER = "wrapper"


class WriteRows(StrEnum):
    """Which rows of a structured port a write touches."""

    ALL = "all"
    DECODE_ONLY = "decode_only"
    FIRST_PER_PROMPT = "first_per_prompt"
    PROVIDER_SELECTED = "provider_selected"


class TimingMobility(StrEnum):
    """Whether a flow's work may be placed at a different point than declared.

    Consumers declare mobility; the placement compiler (Rust core) decides
    actual placement. The correctness invariant: a consumer observes
    identical (port, version) values regardless of where it is placed.
    """

    PINNED = "pinned"  # must run at the declared hook point
    DEFERRABLE = "deferrable"  # may run later, bounded by deadline/max_defer_steps
    ADVANCEABLE = "advanceable"  # may run earlier (e.g. precompute at prepare time)
    FREE = "free"  # any point satisfying the deadline


class TimingDeadline(StrEnum):
    """The latest point a movable flow's work must complete."""

    STEP_END = "step_end"
    NEXT_PREPARE = "next_prepare"
    GENERATE_END = "generate_end"


class VersionObservation(StrEnum):
    """Which port version a flow observes when placed away from its window."""

    AT_DECLARED_WINDOW = "at_declared_window"  # the version at the declared hook (runtime snapshots when moved)
    LATEST_AT_EXECUTION = "latest_at_execution"  # whatever version is current when the flow runs
    STALE_OK = "stale_ok"  # any version within max_defer_steps of current


@dataclass(frozen=True)
class TimingContract:
    """A flow's declared timing semantics. Defaults preserve pinned behavior."""

    mobility: TimingMobility = TimingMobility.PINNED
    deadline: TimingDeadline = TimingDeadline.STEP_END
    max_defer_steps: int = 0
    observes_version: VersionObservation = VersionObservation.AT_DECLARED_WINDOW

    def __post_init__(self) -> None:
        try:
            mobility = TimingMobility(str(self.mobility))
        except ValueError as exc:
            raise ValueError(f"unsupported timing mobility `{self.mobility}`") from exc
        try:
            deadline = TimingDeadline(str(self.deadline))
        except ValueError as exc:
            raise ValueError(f"unsupported timing deadline `{self.deadline}`") from exc
        try:
            observes_version = VersionObservation(str(self.observes_version))
        except ValueError as exc:
            raise ValueError(f"unsupported version observation `{self.observes_version}`") from exc
        object.__setattr__(self, "mobility", mobility)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "observes_version", observes_version)
        if not isinstance(self.max_defer_steps, int) or self.max_defer_steps < 0:  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise ValueError("TimingContract.max_defer_steps must be a non-negative int")
        if mobility == TimingMobility.PINNED:
            if self.max_defer_steps != 0:
                raise ValueError("pinned mobility requires max_defer_steps=0")
            if observes_version != VersionObservation.AT_DECLARED_WINDOW:
                raise ValueError("pinned mobility requires observes_version='at_declared_window'")
        if mobility in (TimingMobility.DEFERRABLE, TimingMobility.FREE) and self.max_defer_steps < 1:
            raise ValueError(f"{mobility} mobility requires max_defer_steps>=1")
        if mobility == TimingMobility.ADVANCEABLE and observes_version == VersionObservation.AT_DECLARED_WINDOW:
            # Running earlier than the declared window can never observe the
            # version that window will produce.
            raise ValueError("advanceable mobility cannot observe 'at_declared_window'")
        if mobility in (TimingMobility.DEFERRABLE, TimingMobility.FREE) and deadline == TimingDeadline.STEP_END:
            # Deferral is across steps; a step_end deadline admits no movement.
            raise ValueError(f"{mobility} mobility is contradictory with deadline='step_end'")


Window = FlowWindow | Literal["background", "same_step", "next_step", "out_of_band"]
DeliveryMode = FlowDelivery | Literal["bundle", "device_lease"]
LeaseLifetime = Literal["consume_call", "step"]
RowCompaction = RowSelection | Literal["none", "first_per_prompt"]
FusionKind = FlowFusion | Literal["none", "side_stream_cudagraph"]
WriteModeName = WriteMode | Literal["replace", "add", "append", "wrapper"]
WriteRowSelection = WriteRows | Literal["all", "decode_only", "first_per_prompt", "provider_selected"]
Timing = TimingContract | Literal["pinned", "deferrable", "advanceable", "free"]


@dataclass(frozen=True)
class Locator:
    """Base logical locator for a public runtime port."""


@dataclass(frozen=True)
class PortRead:
    kind: PortKind
    locator: Locator | None = None
    role: str = ""


@dataclass(frozen=True)
class PortWrite:
    kind: PortKind
    locator: Locator | None = None
    role: str = ""
    mode: str = ""
    rows: WriteRowSelection = WriteRows.ALL
    order: int = 0
    fusion_candidate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.role, str):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard
            raise TypeError("write role must be str")
        object.__setattr__(self, "role", self.role.strip())
        if self.mode:
            if not isinstance(self.mode, (str, WriteMode)):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
                raise TypeError("write mode must be str")
            try:
                object.__setattr__(self, "mode", str(WriteMode(str(self.mode))))
            except ValueError as exc:
                raise ValueError(f"unsupported write mode `{self.mode}`") from exc
        try:
            object.__setattr__(self, "rows", WriteRows(str(self.rows)))
        except ValueError as exc:
            raise ValueError(f"unsupported write rows `{self.rows}`") from exc
        if not isinstance(self.order, int):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise ValueError("PortWrite.order must be an int")
        if not isinstance(self.fusion_candidate, bool):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise ValueError("PortWrite.fusion_candidate must be a bool")
        if self.fusion_candidate and self.mode not in (WriteMode.REPLACE, WriteMode.ADD):
            raise ValueError("fusion_candidate writes require mode='replace' or mode='add'")


@dataclass(frozen=True)
class ConsumerFlow:
    reads: Sequence[PortRead]
    writes: Sequence[PortWrite]
    window: Window
    bundle_key: tuple[str, ...] = field(default_factory=tuple)
    dispatch_every_n_steps: int = 1
    max_bundle_rows: int = 0
    delivery: DeliveryMode = "bundle"
    ownership: BufferOwnership = BufferOwnership.BORROWED
    lease_lifetime: LeaseLifetime = "consume_call"
    row_compaction: RowCompaction = "none"
    fusion: FusionKind = FlowFusion.NONE
    timing: Timing = TimingContract()

    @property
    def bundle_fields(self) -> tuple[str, ...]:
        return tuple(read.role.strip() or read.kind.value for read in self.reads)

    @property
    def output_fields(self) -> tuple[str, ...]:
        return tuple(write.role.strip() or f"{write.kind.value}_write" for write in self.writes)

    def __post_init__(self) -> None:
        try:
            window = FlowWindow(str(self.window))
        except ValueError as exc:
            raise ValueError(f"unsupported flow window `{self.window}`") from exc
        try:
            delivery = FlowDelivery(str(self.delivery))
        except ValueError as exc:
            raise ValueError(f"unsupported delivery mode `{self.delivery}`") from exc
        try:
            ownership = BufferOwnership(str(self.ownership))
        except ValueError as exc:
            raise ValueError(f"unsupported buffer ownership `{self.ownership}`") from exc
        lease_lifetime = str(self.lease_lifetime)
        if lease_lifetime not in ("consume_call", "step"):
            raise ValueError(
                f"unsupported lease_lifetime `{self.lease_lifetime}`; expected 'consume_call' or 'step'"
            )
        try:
            row_compaction = RowSelection(str(self.row_compaction))
        except ValueError as exc:
            raise ValueError(f"unsupported row_compaction mode `{self.row_compaction}`") from exc
        try:
            fusion = FlowFusion(str(self.fusion))
        except ValueError as exc:
            raise ValueError(f"unsupported flow fusion kind `{self.fusion}`") from exc
        timing = self.timing if isinstance(self.timing, TimingContract) else TimingContract(mobility=TimingMobility(str(self.timing)))
        object.__setattr__(self, "timing", timing)
        reads = tuple(self.reads)
        writes = tuple(self.writes)
        bundle_key = tuple(self.bundle_key)
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "bundle_key", bundle_key)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "delivery", delivery)
        object.__setattr__(self, "ownership", ownership)
        object.__setattr__(self, "lease_lifetime", lease_lifetime)
        object.__setattr__(self, "row_compaction", row_compaction)
        object.__setattr__(self, "fusion", fusion)
        valid_bundle_key_fields = {
            "engine_step_id",
            "generation_group_id",
            "phase",
            "request_id",
            "sample_idx",
        }
        seen_key_fields: set[str] = set()
        for field_name in bundle_key:
            if field_name not in valid_bundle_key_fields:
                raise ValueError(f"unknown bundle_key field `{field_name}`")
            if field_name in seen_key_fields:
                raise ValueError(f"duplicate bundle_key field `{field_name}`")
            seen_key_fields.add(field_name)
        seen_bundle_fields: set[str] = set()
        seen_read_addresses: list[PortRead] = []
        for read, field_name in zip(reads, self.bundle_fields):
            if field_name in seen_bundle_fields:
                raise ValueError(f"duplicate bundle field `{field_name}`; assign distinct PortRead.role values")
            seen_bundle_fields.add(field_name)
            if any(
                previous.kind == read.kind and previous.locator == read.locator
                for previous in seen_read_addresses
            ):
                raise ValueError(
                    f"duplicate flow read for port kind `{read.kind.value}` and locator {read.locator!r}"
                )
            seen_read_addresses.append(read)
        output_fields = self.output_fields
        if len(set(output_fields)) != len(output_fields):
            raise ValueError("duplicate flow output role; assign distinct PortWrite.role values")
        step_scope_key = ("engine_step_id", "phase")
        if delivery == FlowDelivery.DEVICE_LEASE:
            if ownership != BufferOwnership.RUNTIME_LEASE:
                raise ValueError("device_lease delivery requires ownership='runtime_lease'")
            if bundle_key != step_scope_key:
                raise ValueError("device_lease delivery currently requires bundle_key=('engine_step_id', 'phase')")
        if row_compaction == RowSelection.FIRST_PER_PROMPT and bundle_key != step_scope_key:
            raise ValueError("first_per_prompt row compaction currently requires bundle_key=('engine_step_id', 'phase')")
        if timing.mobility != TimingMobility.PINNED and delivery == FlowDelivery.DEVICE_LEASE:
            if ownership != BufferOwnership.RUNTIME_LEASE or lease_lifetime != "step":
                raise ValueError(
                    "timing-movable device_lease delivery requires ownership='runtime_lease' "
                    "and lease_lifetime='step'"
                )
        if timing.mobility != TimingMobility.PINNED and bundle_key != step_scope_key:
            raise ValueError(
                "timing-movable flow currently requires bundle_key=('engine_step_id', 'phase')"
            )
        if any(write.fusion_candidate for write in self.writes):
            if delivery != FlowDelivery.DEVICE_LEASE or ownership != BufferOwnership.RUNTIME_LEASE:
                raise ValueError(
                    "fusion_candidate writes require stable device_lease delivery with ownership='runtime_lease'"
                )
            if self.max_bundle_rows <= 0:
                raise ValueError("fusion_candidate writes require max_bundle_rows>0 for a static shape bucket")
        if fusion != FlowFusion.NONE:
            if delivery != FlowDelivery.DEVICE_LEASE or ownership != BufferOwnership.RUNTIME_LEASE:
                raise ValueError(
                    "flow fusion requires stable device_lease delivery with ownership='runtime_lease'"
                )
            if self.max_bundle_rows <= 0:
                raise ValueError("flow fusion requires max_bundle_rows>0 for a static shape bucket")
