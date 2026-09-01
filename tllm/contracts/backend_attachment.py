"""Closed backend attachment facts bound to canonical physical plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.request_identity import (
    RequestKeyField,
    canonical_request_key_fields,
)
from tllm.contracts.semantic_graph import (
    AnchorId,
    ImplementationId,
    TaskId,
    PhysicalPlan,
    ScheduleDecision,
    SchedulerTimeline,
    SemanticGraph,
)
from tllm.contracts.state_representation import (
    StateDeviceKind,
    StatePhysicalPlan,
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import TensorDType


class AttachmentTopology(StrEnum):
    IN_PROCESS = "in_process"


class AttachmentKind(StrEnum):
    REQUEST_SAMPLER_TRAINING = "request_sampler_training"


class AttachmentGraphMode(StrEnum):
    EAGER = "eager"


class AttachmentSamplerStage(StrEnum):
    PACKED_POST_FILTER_EXACT = "packed_post_filter_exact"


class AttachmentCleanupStage(StrEnum):
    REQUEST_FINISH = "request_finish"


class StreamDependency(StrEnum):
    HIDDEN_PRODUCER_TO_COMPUTE = "hidden_producer_to_compute"
    COMPUTE_TO_SAMPLER = "compute_to_sampler"
    SAMPLER_TO_TRANSITION = "sampler_to_transition"
    TRANSITION_TO_RELEASE = "transition_to_release"


class AttachmentTaskRole(StrEnum):
    COMPUTE = "compute"
    TRANSITION = "transition"


@dataclass(frozen=True, slots=True)
class AttachmentHiddenResource:
    binding: PackedHiddenResourceBinding
    phase: HiddenPhase

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not PackedHiddenResourceBinding
            or type(self.phase) is not HiddenPhase
            or self.phase is HiddenPhase.NONE
        ):
            raise ValueError("attachment hidden resource is not closed")

    @property
    def resource_key(self) -> tuple[str, str, str]:
        return self.binding.locator_key


@dataclass(frozen=True, slots=True)
class AttachmentTaskBinding:
    role: AttachmentTaskRole
    task_id: TaskId
    implementation_id: ImplementationId
    anchor_id: AnchorId

    def __post_init__(self) -> None:
        if (
            type(self.role) is not AttachmentTaskRole
            or type(self.task_id) is not int
            or type(self.implementation_id) is not str
            or not self.implementation_id
            or type(self.anchor_id) is not int
            or int(self.task_id) < 0
            or int(self.anchor_id) < 0
        ):
            raise ValueError("attachment task binding is invalid")


@dataclass(frozen=True, slots=True)
class BackendPlacementProbe:
    adapter_id: str
    backend_abi_digest: str
    runner_fingerprint: str
    model_fingerprint: str
    topology: AttachmentTopology
    graph_mode: AttachmentGraphMode
    device_kind: StateDeviceKind
    device_ordinal: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    tensor_dtype: TensorDType
    hidden_size: int
    max_packed_rows: int
    hidden_resources: tuple[AttachmentHiddenResource, ...]
    capabilities: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        digests = (
            self.backend_abi_digest,
            self.runner_fingerprint,
            self.model_fingerprint,
        )
        resource_keys = tuple(
            resource.resource_key for resource in self.hidden_resources
        )
        if (
            type(self.adapter_id) is not str
            or not self.adapter_id
            or any(
                type(value) is not str
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in digests
            )
            or type(self.topology) is not AttachmentTopology
            or type(self.graph_mode) is not AttachmentGraphMode
            or type(self.device_kind) is not StateDeviceKind
            or type(self.device_ordinal) is not int
            or self.device_ordinal < 0
            or (
                self.device_kind is StateDeviceKind.CPU
                and self.device_ordinal != 0
            )
            or type(self.tensor_parallel_size) is not int
            or type(self.pipeline_parallel_size) is not int
            or self.tensor_parallel_size != 1
            or self.pipeline_parallel_size != 1
            or type(self.tensor_dtype) is not TensorDType
            or type(self.hidden_size) is not int
            or type(self.max_packed_rows) is not int
            or self.hidden_size < 1
            or self.max_packed_rows < 1
            or type(self.hidden_resources) is not tuple
            or any(
                type(resource) is not AttachmentHiddenResource
                for resource in self.hidden_resources
            )
            or not self.hidden_resources
            or tuple(sorted(resource_keys)) != resource_keys
            or len(set(resource_keys)) != len(resource_keys)
            or type(self.capabilities) is not tuple
            or any(type(capability) is not str for capability in self.capabilities)
            or tuple(sorted(self.capabilities)) != self.capabilities
            or len(set(self.capabilities)) != len(self.capabilities)
            or any(not capability for capability in self.capabilities)
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("backend placement probe is not canonical")


@dataclass(frozen=True, slots=True)
class BackendAttachmentRequest:
    adapter_id: str
    attachment_kind: AttachmentKind
    backend_abi_digest: str
    runner_fingerprint: str
    model_fingerprint: str
    topology: AttachmentTopology
    graph_mode: AttachmentGraphMode
    device_kind: StateDeviceKind
    device_ordinal: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    tensor_dtype: TensorDType
    hidden_size: int
    max_packed_rows: int
    packed_row_schema_version: int
    hidden_resources: tuple[AttachmentHiddenResource, ...]
    task_bindings: tuple[AttachmentTaskBinding, ...]
    sampler_anchor_id: AnchorId
    sampler_anchor_symbol: str
    stream_dependencies: tuple[StreamDependency, ...]
    sampler_stage: AttachmentSamplerStage
    cleanup_stage: AttachmentCleanupStage
    request_key_fields: tuple[RequestKeyField, ...]
    state_resource: str
    plan_epoch: int
    schedule_signature: str
    state_request_digest: str
    state_representation_signature: str
    execution_parameters_digest: str
    capabilities: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        digest_values = (
            self.backend_abi_digest,
            self.runner_fingerprint,
            self.model_fingerprint,
            self.schedule_signature,
            self.state_request_digest,
            self.state_representation_signature,
            self.execution_parameters_digest,
        )
        if any(type(value) is not str for value in digest_values):
            raise TypeError("attachment digests must be exact strings")
        for value in digest_values:
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("attachment digests must be lowercase SHA-256")
        canonical_request_key_fields(self.request_key_fields)
        resource_keys = tuple(
            resource.resource_key for resource in self.hidden_resources
        )
        canonical_dependencies = tuple(
            dependency
            for dependency in (
                StreamDependency.HIDDEN_PRODUCER_TO_COMPUTE,
                StreamDependency.COMPUTE_TO_SAMPLER,
                StreamDependency.SAMPLER_TO_TRANSITION,
                StreamDependency.TRANSITION_TO_RELEASE,
            )
            if dependency in self.stream_dependencies
        )
        binding_roles = tuple(binding.role for binding in self.task_bindings)
        if (
            type(self.adapter_id) is not str
            or not self.adapter_id
            or type(self.attachment_kind) is not AttachmentKind
            or type(self.topology) is not AttachmentTopology
            or type(self.graph_mode) is not AttachmentGraphMode
            or type(self.device_kind) is not StateDeviceKind
            or type(self.device_ordinal) is not int
            or type(self.tensor_parallel_size) is not int
            or type(self.pipeline_parallel_size) is not int
            or type(self.tensor_dtype) is not TensorDType
            or type(self.hidden_size) is not int
            or type(self.max_packed_rows) is not int
            or type(self.packed_row_schema_version) is not int
            or type(self.hidden_resources) is not tuple
            or any(
                type(resource) is not AttachmentHiddenResource
                for resource in self.hidden_resources
            )
            or not self.hidden_resources
            or tuple(sorted(resource_keys)) != resource_keys
            or len(set(resource_keys)) != len(resource_keys)
            or type(self.task_bindings) is not tuple
            or any(
                type(binding) is not AttachmentTaskBinding
                for binding in self.task_bindings
            )
            or binding_roles
            != (AttachmentTaskRole.COMPUTE, AttachmentTaskRole.TRANSITION)
            or type(self.sampler_anchor_id) is not int
            or int(self.sampler_anchor_id) < 0
            or type(self.sampler_anchor_symbol) is not str
            or not self.sampler_anchor_symbol
            or type(self.stream_dependencies) is not tuple
            or any(
                type(dependency) is not StreamDependency
                for dependency in self.stream_dependencies
            )
            or self.stream_dependencies != canonical_dependencies
            or (
                self.attachment_kind
                is AttachmentKind.REQUEST_SAMPLER_TRAINING
                and self.stream_dependencies
                != (
                    StreamDependency.HIDDEN_PRODUCER_TO_COMPUTE,
                    StreamDependency.COMPUTE_TO_SAMPLER,
                    StreamDependency.SAMPLER_TO_TRANSITION,
                    StreamDependency.TRANSITION_TO_RELEASE,
                )
            )
            or type(self.sampler_stage) is not AttachmentSamplerStage
            or type(self.cleanup_stage) is not AttachmentCleanupStage
            or type(self.state_resource) is not str
            or not self.state_resource
            or self.device_ordinal < 0
            or self.tensor_parallel_size != 1
            or self.pipeline_parallel_size != 1
            or self.hidden_size < 1
            or self.max_packed_rows < 1
            or self.packed_row_schema_version != 1
            or type(self.plan_epoch) is not int
            or self.plan_epoch < 1
            or type(self.capabilities) is not tuple
            or any(type(capability) is not str for capability in self.capabilities)
            or tuple(sorted(self.capabilities)) != self.capabilities
            or len(set(self.capabilities)) != len(self.capabilities)
            or any(not capability for capability in self.capabilities)
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("backend attachment request is not canonical")


@dataclass(frozen=True, slots=True)
class BackendAttachmentCompilerInputs:
    request: BackendAttachmentRequest
    probe: BackendPlacementProbe
    graph: SemanticGraph
    timeline: SchedulerTimeline
    schedule_decision: ScheduleDecision
    schedule_physical: PhysicalPlan
    state_request: StateRepresentationRequest
    state_decision: StateRepresentationDecision
    state_physical: StatePhysicalPlan

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BackendAttachmentRequest
            or type(self.probe) is not BackendPlacementProbe
            or type(self.graph) is not SemanticGraph
            or type(self.timeline) is not SchedulerTimeline
            or type(self.schedule_decision) is not ScheduleDecision
            or type(self.schedule_physical) is not PhysicalPlan
            or type(self.state_request) is not StateRepresentationRequest
            or type(self.state_decision) is not StateRepresentationDecision
            or type(self.state_physical) is not StatePhysicalPlan
        ):
            raise TypeError("backend attachment compiler inputs must be exact plans")


@dataclass(frozen=True, slots=True)
class BackendAttachmentDecision:
    request_digest: str
    signature: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if any(
            type(value) is not str
            for value in (self.request_digest, self.signature)
        ):
            raise TypeError("attachment decision digests must be strings")
        for value in (self.request_digest, self.signature):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("attachment decision digests must be SHA-256")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("attachment decision schema is unsupported")


@dataclass(frozen=True, slots=True)
class BackendAttachmentPhysicalPlan:
    plan_epoch: int
    adapter_id: str
    attachment_signature: str
    schedule_signature: str
    state_representation_signature: str
    device_kind: StateDeviceKind
    device_ordinal: int
    topology: AttachmentTopology
    graph_mode: AttachmentGraphMode
    bundle_signature: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        signature_values = (
            self.attachment_signature,
            self.schedule_signature,
            self.state_representation_signature,
            self.bundle_signature,
        )
        if any(type(value) is not str for value in signature_values):
            raise TypeError("physical attachment signatures must be strings")
        for value in signature_values:
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("physical attachment signatures must be SHA-256")
        if (
            type(self.plan_epoch) is not int
            or self.plan_epoch < 1
            or type(self.adapter_id) is not str
            or not self.adapter_id
            or type(self.device_kind) is not StateDeviceKind
            or type(self.device_ordinal) is not int
            or self.device_ordinal < 0
            or type(self.topology) is not AttachmentTopology
            or type(self.graph_mode) is not AttachmentGraphMode
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("physical attachment plan is invalid")


__all__ = [
    "AttachmentCleanupStage",
    "AttachmentGraphMode",
    "AttachmentKind",
    "AttachmentHiddenResource",
    "AttachmentSamplerStage",
    "AttachmentTaskBinding",
    "AttachmentTaskRole",
    "AttachmentTopology",
    "BackendAttachmentDecision",
    "BackendAttachmentCompilerInputs",
    "BackendAttachmentPhysicalPlan",
    "BackendAttachmentRequest",
    "BackendPlacementProbe",
    "StreamDependency",
]
