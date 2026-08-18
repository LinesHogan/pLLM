"""Type stub for the Rust extension module ``tllm._core`` (pyo3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

class PlanPortSpecDescriptor(Protocol):
    @property
    def port(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def writable(self) -> bool: ...
    @property
    def write_modes(self) -> Sequence[str]: ...
    @property
    def phases(self) -> Sequence[str]: ...

class PlanProducerDescriptor(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def priority(self) -> int: ...
    @property
    def outputs(self) -> Sequence[PlanPortSpecDescriptor]: ...
    @property
    def requires(self) -> Sequence[str]: ...
    @property
    def hook_points(self) -> Sequence[str]: ...
    @property
    def graph_safety(self) -> str: ...

class PlanConsumerReadDescriptor(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def role(self) -> str: ...
    @property
    def phase(self) -> str: ...

class PlanConsumerWriteDescriptor(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def mode(self) -> str: ...
    @property
    def phase(self) -> str: ...

class PlanConsumerFlowDescriptor(Protocol):
    @property
    def reads(self) -> Sequence[PlanConsumerReadDescriptor]: ...
    @property
    def writes(self) -> Sequence[PlanConsumerWriteDescriptor]: ...
    @property
    def bundle_key(self) -> Sequence[str]: ...

class PlanConsumerDescriptor(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def flows(self) -> Sequence[PlanConsumerFlowDescriptor]: ...

class LocatorFieldDescriptor(Protocol):
    @property
    def path(self) -> Sequence[str]: ...
    @property
    def value_type(self) -> str: ...
    @property
    def value(self) -> str: ...

class PortAddressDescriptor(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def locator_type(self) -> str: ...
    @property
    def locator_fields(self) -> Sequence[LocatorFieldDescriptor]: ...

class PlacementWriteDescriptor(Protocol):
    @property
    def port(self) -> PortAddressDescriptor: ...
    @property
    def order(self) -> int: ...
    @property
    def mode(self) -> str: ...
    @property
    def fusion_candidate(self) -> bool: ...

class VersionedStateResourceDescriptor(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def namespace(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> int: ...

class PlacementResourceWriteDescriptor(Protocol):
    @property
    def resource(self) -> VersionedStateResourceDescriptor: ...

class PlacementFlowDescriptor(Protocol):
    @property
    def flow_id(self) -> str: ...
    @property
    def consumer_id(self) -> str: ...
    @property
    def declared_hook(self) -> str: ...
    @property
    def mobility(self) -> str | None: ...
    @property
    def deadline(self) -> str | None: ...
    @property
    def max_defer_steps(self) -> int | None: ...
    @property
    def observes_version(self) -> str | None: ...
    @property
    def reads(self) -> Sequence[PortAddressDescriptor]: ...
    @property
    def writes(self) -> Sequence[PlacementWriteDescriptor]: ...
    @property
    def delivery(self) -> str: ...
    @property
    def ownership(self) -> str: ...
    @property
    def lease_lifetime(self) -> str: ...
    @property
    def max_bundle_rows(self) -> int: ...
    @property
    def effect_hook(self) -> str | None: ...
    @property
    def earliest_hook(self) -> str | None: ...
    @property
    def allowed_hooks(self) -> Sequence[str]: ...
    @property
    def fusion_kind(self) -> str: ...
    @property
    def resource_reads(self) -> Sequence[VersionedStateResourceDescriptor]: ...
    @property
    def resource_writes(self) -> Sequence[PlacementResourceWriteDescriptor]: ...

class PlacementHookCostDescriptor(Protocol):
    @property
    def hook(self) -> str: ...
    @property
    def bubble_score(self) -> int: ...

class DecodeLocalization:
    positions: list[int]
    row_idx: list[int]
    contiguous: bool

class DecodePositions:
    positions: list[int]
    contiguous: bool

class RowPartition:
    affected_greedy: list[int]
    affected_random: list[int]
    unaffected_greedy: list[int]
    unaffected_random: list[int]

class SamplerPlan:
    path: str
    use_candidate_kernel: bool
    dense_inplace_applied: bool
    base: str

class CompiledPlan:
    ordered_producer_ids: list[str]
    port_sources: list[tuple[str, str]]
    capture_resident_producer_ids: list[str]

class PlacementSchedule:
    placements: list[tuple[str, str]]
    execution_order: list[str]
    fusion_groups: list[list[str]]
    fusion_rejections: list[tuple[str, str]]

def compile_placement_schedule(
    flows: Sequence[PlacementFlowDescriptor],
    graph_mode: str,
    hook_costs: Sequence[PlacementHookCostDescriptor] | None = None,
) -> PlacementSchedule: ...

def localize_decode_rows(
    is_decode: list[bool],
    logits_indices: list[int],
    req_count: int,
    num_actual_tokens: int,
    max_decode_rows: int,
) -> DecodeLocalization: ...

def localize_decode_positions(
    is_decode: list[bool],
    n_logits: int,
    req_count: int,
    max_decode_rows: int,
) -> DecodePositions: ...

def plan_sampler_step(
    backend: str,
    temperature_mode: str,
    n_rows: int,
    n_affected: int,
    n_greedy: int,
    greedy_sampled_available: bool,
    candidate_eligible: bool,
) -> SamplerPlan: ...

def partition_rows(
    n_rows: int,
    affected_rows: list[int],
    greedy_rows: list[int],
    greedy_sampled_available: bool,
) -> RowPartition: ...

def compile_execution_plan(
    producers: Sequence[PlanProducerDescriptor],
    consumers: Sequence[PlanConsumerDescriptor],
    enforced_kinds: list[str] | None = ...,
    supported_hooks: list[str] | None = ...,
    backend: tuple[str, str] | None = ...,
) -> CompiledPlan: ...
