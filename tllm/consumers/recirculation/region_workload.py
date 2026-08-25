"""Native Region authority for finite exact Recirculation executions.

The workload deliberately owns the whole token/layer recurrence.  It does not
adapt ``RecirculationConsumer`` and therefore cannot leave model-region hooks
and a Region plan as competing schedulers for the same request.

The current backend timeline has no prefill row/model-region binding.  A
caller must consequently publish an explicit :class:`RecirculationExecutionRequest`;
attempting to use this workload as an implicit vLLM prefill feature fails
closed at the provider boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Mapping, cast

import torch

import tllm._core as _core
from tllm.runtime.typed_recurrence import (
    BackendRegionAdapter,
    PrimitiveAdapterRegistry,
    TypedRecurrenceExecutor,
)
from tllm.contracts.backend_capability import BackendCapability
from tllm.consumers.recirculation.config import RecirculationConfig
from tllm.consumers.recirculation.reference import ReferenceDecoder
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    WorkloadWorkerSpec,
)
from tllm.workgraph.recurrence import RecurrenceGraph
from tllm.workload import (
    AffineAccess,
    Before,
    Domain,
    ExternImplementation,
    Index,
    provided,
    sink,
    step_id,
    task,
    workload,
)

_REQUEST_INPUT = "recirculation_request"
_RESULT_SINK = "recirculation_result"
_REFERENCE_IMPLEMENTATION_ID = "tllm.recirculation:reference_replay@1"
_REFERENCE_LAYER_PRIMITIVE = "tllm.recirculation:reference_layer@1"
_REFERENCE_MIX_PRIMITIVE = "tllm.recirculation:mix@1"
_REFERENCE_NORMAL_RESOURCE = "tllm.recirculation:reference_normal_resource@1"
_REFERENCE_RECURRENT_RESOURCE = "tllm.recirculation:reference_recurrent_resource@1"
_REFERENCE_MIX_RESOURCE = "tllm.recirculation:reference_mix_resource@1"
_REFERENCE_NORMAL_OUTPUT = "tllm.recirculation:reference_normal_output@1"
_REFERENCE_RECURRENT_OUTPUT = "tllm.recirculation:reference_recurrent_output@1"
_ARTIFACT_WIRE_KEY = "stage_artifact_bundle"


@dataclass(frozen=True, slots=True)
class RecirculationExecutionRequest:
    """One explicit finite recurrence execution owned by this Region."""

    decoder: ReferenceDecoder
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple) or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in self.token_ids
        ):
            raise TypeError("token_ids must be tuple[int, ...]")
        if not self.token_ids:
            raise ValueError("exact Recirculation requires at least one token")
        try:
            layer_count = self.decoder.layer_count
        except AttributeError as exc:
            raise TypeError("decoder must expose an integer layer_count") from exc
        if not isinstance(layer_count, int):
            raise TypeError("decoder must expose an integer layer_count")


@dataclass(frozen=True, slots=True)
class RecirculationExecutionResult:
    """Observable normal logits and the replayed states for one request."""

    logits: tuple[torch.Tensor, ...]
    recurrent_states: tuple[torch.Tensor, ...]
    schedule_kind: str
    schedule_groups: tuple[tuple[str, ...], ...]
    step_id: int


def _normal(token: int, layer: int) -> str:
    return f"t{token}/normal/l{layer}"


def _recurrent(token: int, layer: int) -> str:
    return f"t{token}/recurrent/l{layer}"


def _mix(token: int) -> str:
    return f"t{token}/mix"


def _mix_tensors(
    config: RecirculationConfig,
    source: torch.Tensor,
    destination: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Keep Region execution numerically identical to consumer mix semantics."""

    if source.shape != destination.shape:
        raise RuntimeError("Recirculation source and destination shapes must match")
    if source.ndim < 2:
        raise RuntimeError("Recirculation residuals must have a hidden dimension")
    flat_positions = positions[0] if positions.ndim == 2 else positions
    if flat_positions.ndim != 1 or int(flat_positions.shape[0]) != int(source.shape[0]):
        raise RuntimeError("Recirculation positions must align with residual rows")
    source_float = source.float()
    destination_float = destination.float()
    source_norm = torch.linalg.vector_norm(source_float, dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    destination_norm = torch.linalg.vector_norm(destination_float, dim=-1, keepdim=True)
    normalized = source_float * (destination_norm / source_norm)
    alpha: float | torch.Tensor = float(config.alpha)
    if config.ramp_tokens:
        ramp = flat_positions.to(dtype=torch.float32).clamp(
            min=0, max=int(config.ramp_tokens)
        )
        alpha = (ramp / int(config.ramp_tokens)).unsqueeze(-1) * float(config.alpha)
    beta = (1.0 - alpha) if config.beta is None else float(config.beta)
    return (beta * destination_float + alpha * normalized).to(destination.dtype)


def _operand(kind: str, field: str, value: str) -> dict[str, str]:
    return {"kind": kind, field: value}


def _finite_recurrence_spec(
    graph: RecurrenceGraph,
    config: RecirculationConfig,
    *,
    layer_count: int,
) -> str:
    """Declare algorithm dataflow; Rust alone groups it and derives liveness."""

    source_layer, destination_layer = config.resolved_layers(layer_count)
    operations: list[dict[str, object]] = []
    for op in graph.ops:
        point = op.point
        if point.pass_name == "normal":
            inputs = (
                [_operand("seed", "binding_id", f"token/{point.token}")]
                if point.layer == 0
                else [
                    _operand(
                        "region_value",
                        "value_id",
                        _normal(point.token, point.layer - 1),
                    )
                ]
            )
            inputs.append(
                _operand("backend_resource", "resource_id", f"decoder/{op.op_id}")
            )
            primitive_key = _REFERENCE_LAYER_PRIMITIVE
        elif point.pass_name == "recurrent":
            previous = (
                _mix(point.token)
                if point.layer == destination_layer + 1
                else _recurrent(point.token, point.layer - 1)
            )
            inputs = [
                _operand("region_value", "value_id", previous),
                _operand("backend_resource", "resource_id", f"decoder/{op.op_id}"),
            ]
            primitive_key = _REFERENCE_LAYER_PRIMITIVE
        elif point.pass_name == "mix":
            inputs = [
                _operand(
                    "region_value",
                    "value_id",
                    _normal(point.token, source_layer),
                ),
                _operand(
                    "region_value",
                    "value_id",
                    _normal(point.token, destination_layer),
                ),
                _operand("seed", "binding_id", f"position/{point.token}"),
                _operand("backend_resource", "resource_id", f"config/{op.op_id}"),
            ]
            primitive_key = _REFERENCE_MIX_PRIMITIVE
        else:
            raise RuntimeError(
                f"Recirculation declaration has unsupported pass {point.pass_name!r}"
            )
        operations.append(
            {
                "operation_id": op.op_id,
                "primitive_key": primitive_key,
                "implementation_id": _REFERENCE_IMPLEMENTATION_ID,
                "operands": inputs,
                "results": [op.op_id],
            }
        )
    retained = [
        op.op_id
        for op in graph.ops
        if (op.point.pass_name == "normal" and op.point.layer == layer_count - 1)
        or (
            op.point.pass_name == "recurrent"
            and op.point.layer == layer_count - 1
        )
    ]
    spec = {
        "schema_version": 1,
        "region_id": "recirculation.reference",
        "implementation_id": _REFERENCE_IMPLEMENTATION_ID,
        "row_admission": {
            "allowed_phases": ["prefill"],
            "mixed_batch": "reject",
        },
        # This is an implementation capability declaration, not a runtime
        # selection. Rust validates each dependence then derives the one
        # grouping recorded in the signed artifact.
        "fusion_policy": "max_rows=2;internalized_edges=kv_overwrite_read",
        "operations": operations,
        "dependencies": [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind.value,
            }
            for edge in graph.edges
        ],
        "retained_values": retained,
    }
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def _stage_shape_specialization(row_count: int) -> list[list[str]]:
    return [
        ["stage/request_count", "1"],
        ["stage/row_count", str(row_count)],
    ]


def _reference_stage_artifact_spec(
    config: RecirculationConfig,
    *,
    row_count: int,
    layer_count: int,
) -> str:
    """Cold-only algorithm declaration for one signed reference stage."""

    graph = build_recirculation_graph(
        config, token_count=row_count, layer_count=layer_count
    )
    recurrence = json.loads(
        _finite_recurrence_spec(
            graph, config, layer_count=layer_count
        )
    )
    resource_bindings: list[dict[str, object]] = []
    seed_bindings: list[dict[str, object]] = []
    output_bindings: list[dict[str, object]] = []
    for op in graph.ops:
        point = op.point
        if point.pass_name == "normal":
            if point.layer == 0:
                seed_bindings.append(
                    {
                        "binding_id": f"token/{point.token}",
                        "kind": "tllm.recirculation:reference_token@1",
                        "coordinates": [point.token],
                    }
                )
            resource_bindings.append(
                {
                    "resource_id": f"decoder/{op.op_id}",
                    "kind": _REFERENCE_NORMAL_RESOURCE,
                    "coordinates": [point.token, point.layer],
                }
            )
        elif point.pass_name == "recurrent":
            resource_bindings.append(
                {
                    "resource_id": f"decoder/{op.op_id}",
                    "kind": _REFERENCE_RECURRENT_RESOURCE,
                    "coordinates": [point.token, point.layer],
                }
            )
        elif point.pass_name == "mix":
            seed_bindings.append(
                {
                    "binding_id": f"position/{point.token}",
                    "kind": "tllm.recirculation:reference_position@1",
                    "coordinates": [point.token],
                }
            )
            resource_bindings.append(
                {
                    "resource_id": f"config/{op.op_id}",
                    "kind": _REFERENCE_MIX_RESOURCE,
                    "coordinates": [point.token],
                }
            )
    for token in range(row_count):
        output_bindings.extend(
            (
                {
                    "output_id": f"normal/{token}",
                    "value_id": _normal(token, layer_count - 1),
                    "kind": _REFERENCE_NORMAL_OUTPUT,
                    "coordinates": [token],
                },
                {
                    "output_id": f"recurrent/{token}",
                    "value_id": _recurrent(token, layer_count - 1),
                    "kind": _REFERENCE_RECURRENT_OUTPUT,
                    "coordinates": [token],
                },
            )
        )
    return json.dumps(
        {
            "schema_version": 1,
            "shape_specialization": _stage_shape_specialization(row_count),
            "row_count": row_count,
            "request_count": 1,
            "recurrence": recurrence,
            "seed_bindings": sorted(seed_bindings, key=lambda item: str(item["binding_id"])),
            "resource_bindings": sorted(
                resource_bindings, key=lambda item: str(item["resource_id"])
            ),
            "output_bindings": sorted(
                output_bindings, key=lambda item: str(item["output_id"])
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _reference_registry_from_artifact(
    artifact: _core.TypedRecurrenceStageArtifact,
) -> PrimitiveAdapterRegistry:
    """Bind closed consumer descriptors without rebuilding the recurrence graph."""

    resources = {
        binding.resource_id: (binding.kind, tuple(binding.coordinates))
        for binding in artifact.resource_bindings
    }
    descriptors: dict[str, tuple[str, tuple[int, ...]]] = {}
    primitive_keys: set[str] = set()
    for group in artifact.plan.groups:
        primitive_keys.add(group.primitive_key)
        for operation in group.operations:
            resource_ids = [
                operand.resource_id
                for operand in operation.operands
                if operand.kind == _core.OperandKind.BackendResource
            ]
            if len(resource_ids) != 1 or resource_ids[0] is None:
                raise RuntimeError(
                    "signed reference stage operation must have exactly one resource descriptor"
                )
            try:
                descriptor = resources[resource_ids[0]]
            except KeyError as exc:
                raise RuntimeError(
                    "signed reference stage operation references an unknown resource descriptor"
                ) from exc
            if operation.operation_id in descriptors:
                raise RuntimeError("signed reference stage repeats an operation descriptor")
            descriptors[operation.operation_id] = descriptor

    expected_primitives = {_REFERENCE_LAYER_PRIMITIVE, _REFERENCE_MIX_PRIMITIVE}
    expected_resources = {
        _REFERENCE_NORMAL_RESOURCE,
        _REFERENCE_RECURRENT_RESOURCE,
        _REFERENCE_MIX_RESOURCE,
    }
    if primitive_keys != expected_primitives:
        raise RuntimeError("signed reference stage has an unsupported primitive set")
    if any(kind not in expected_resources for kind, _coordinates in descriptors.values()):
        raise RuntimeError("signed reference stage has an unsupported resource descriptor")

    def descriptor_for(operation: object) -> tuple[str, tuple[int, ...]]:
        operation_id = operation.operation_id  # type: ignore[attr-defined]
        try:
            return descriptors[operation_id]
        except KeyError as exc:
            raise RuntimeError("Rust stage operation has no cold-bound descriptor") from exc

    def normal(
        operands: tuple[object, ...], token: int, layer: int
    ) -> tuple[object, ...]:
        decoder = cast(ReferenceDecoder, operands[-1])
        hidden = decoder.embed(cast(int, operands[0]), token) if layer == 0 else cast(
            torch.Tensor, operands[0]
        )
        return (decoder.run_layer(layer, hidden, position=token, overwrite_kv=False),)

    def recurrent(
        operands: tuple[object, ...], token: int, layer: int
    ) -> tuple[object, ...]:
        decoder = cast(ReferenceDecoder, operands[-1])
        return (
            decoder.run_layer(
                layer,
                cast(torch.Tensor, operands[0]),
                position=token,
                overwrite_kv=True,
            ),
        )

    layer_handlers: dict[
        str, Callable[[tuple[object, ...]], tuple[object, ...]]
    ] = {}
    for operation_id, (kind, coordinates) in descriptors.items():
        if kind == _REFERENCE_NORMAL_RESOURCE:
            token, layer_index = coordinates
            layer_handlers[operation_id] = (
                lambda operands, token=token, layer_index=layer_index: normal(
                    operands, token, layer_index
                )
            )
        elif kind == _REFERENCE_RECURRENT_RESOURCE:
            token, layer_index = coordinates
            layer_handlers[operation_id] = (
                lambda operands, token=token, layer_index=layer_index: recurrent(
                    operands, token, layer_index
                )
            )

    def layer(operation: object, operands: tuple[object, ...]) -> tuple[object, ...]:
        try:
            handler = layer_handlers[operation.operation_id]  # type: ignore[attr-defined]
        except KeyError as exc:
            raise RuntimeError("Rust stage layer operation has no cold-bound handler") from exc
        return handler(operands)

    def mix(operation: object, operands: tuple[object, ...]) -> tuple[object, ...]:
        kind, (token,) = descriptor_for(operation)
        if kind != _REFERENCE_MIX_RESOURCE:
            raise RuntimeError("Rust stage mix primitive and descriptor kind disagree")
        source = cast(torch.Tensor, operands[0])
        destination = cast(torch.Tensor, operands[1])
        position = cast(int, operands[2])
        if position != token:
            raise RuntimeError("signed reference mix position differs from its descriptor")
        config = cast(RecirculationConfig, operands[3])
        positions = torch.tensor([position], device=destination.device)
        return (_mix_tensors(config, source, destination, positions),)

    return PrimitiveAdapterRegistry(
        (
            BackendRegionAdapter(
                _REFERENCE_IMPLEMENTATION_ID,
                _REFERENCE_LAYER_PRIMITIVE,
                layer,
            ),
            BackendRegionAdapter(
                _REFERENCE_IMPLEMENTATION_ID, _REFERENCE_MIX_PRIMITIVE, mix
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class _ReferenceStageArtifacts:
    wire: str
    bundle: _core.TypedRecurrenceStageArtifactBundle
    layer_count: int
    max_row_count: int
    registries: Mapping[str, PrimitiveAdapterRegistry]

    @classmethod
    def compile(
        cls,
        config: RecirculationConfig,
        *,
        max_row_count: int,
        layer_count: int,
    ) -> "_ReferenceStageArtifacts":
        if max_row_count < 1 or layer_count < 1:
            raise ValueError("reference stage cold specialization must be positive")

        artifacts = [
            {
                "row_count": row_count,
                "artifact_json": _core.compile_typed_recurrence_stage_artifact_json(
                    _reference_stage_artifact_spec(
                        config,
                        row_count=row_count,
                        layer_count=layer_count,
                    )
                ),
            }
            for row_count in range(1, max_row_count + 1)
        ]
        wire = _core.compile_typed_recurrence_stage_artifact_bundle_json(
            json.dumps(
                {"schema_version": 1, "artifacts": artifacts},
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        return cls.load(
            wire=wire,
            layer_count=layer_count,
        )

    @classmethod
    def load(
        cls,
        *,
        wire: str,
        layer_count: int,
    ) -> "_ReferenceStageArtifacts":
        bundle = _core.load_typed_recurrence_stage_artifact_bundle(wire)
        if layer_count < 1:
            raise ValueError("reference stage artifact layer_count must be positive")
        registries: dict[str, PrimitiveAdapterRegistry] = {}
        for entry in bundle.artifacts:
            artifact = entry.artifact
            if artifact.request_count != 1:
                raise RuntimeError("reference stage artifact must specialize one request")
            registries[artifact.signature] = _reference_registry_from_artifact(artifact)
        return cls(
            wire=wire,
            bundle=bundle,
            layer_count=layer_count,
            max_row_count=len(bundle.artifacts),
            registries=MappingProxyType(registries),
        )

    def validate_runtime_specialization(
        self, *, max_row_count: int, layer_count: int
    ) -> None:
        if max_row_count != self.max_row_count or layer_count != self.layer_count:
            raise RuntimeError("reference stage artifact specialization does not match runtime facts")
        for row_count in range(1, self.max_row_count + 1):
            bound = _core.bind_typed_recurrence_stage_bundle_for_request_rows(
                self.bundle, row_count
            )
            if bound.stage.signature not in self.registries:
                raise RuntimeError(
                    "reference stage artifact has no cold-bound primitive registry"
                )

    def require(
        self, row_count: int
    ) -> tuple[_core.TypedBoundRecurrenceStage, PrimitiveAdapterRegistry]:
        try:
            bound = _core.bind_typed_recurrence_stage_bundle_for_request_rows(
                self.bundle, row_count
            ).stage
            return bound, self.registries[bound.signature]
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "specialization-required: exact Recirculation has no signed stage artifact "
                f"for row_count={row_count}"
            ) from exc


def run_exact_recirculation_region(
    request: RecirculationExecutionRequest,
    config: RecirculationConfig,
    *,
    step_id: int,
    artifacts: _ReferenceStageArtifacts,
) -> RecirculationExecutionResult:
    """Bind and execute one immutable Rust stage without runtime planning."""

    if not isinstance(request, RecirculationExecutionRequest):
        raise TypeError("request must be RecirculationExecutionRequest")
    if not isinstance(config, RecirculationConfig):
        raise TypeError("config must be RecirculationConfig")
    if not config.enabled or config.jacobi is not None:
        raise RuntimeError("exact Recirculation requires an enabled non-Jacobi config")
    if request.decoder.layer_count != artifacts.layer_count:
        raise RuntimeError("exact Recirculation request decoder does not match cold specialization")
    count = len(request.token_ids)
    bound, registry = artifacts.require(count)
    values = TypedRecurrenceExecutor(registry).execute(
        bound.plan,
        seed_values={
            **{f"token/{token}": token_id for token, token_id in enumerate(request.token_ids)},
            **{f"position/{token}": token for token in range(count)},
        },
        backend_resources={
            binding.resource_id: (
                request.decoder
                if binding.kind
                in {_REFERENCE_NORMAL_RESOURCE, _REFERENCE_RECURRENT_RESOURCE}
                else config
            )
            for binding in bound.resource_bindings
        },
    )
    normals: dict[int, torch.Tensor] = {}
    recurrents: dict[int, torch.Tensor] = {}
    for binding in bound.output_bindings:
        if len(binding.coordinates) != 1:
            raise RuntimeError("signed reference stage output requires one token coordinate")
        token = binding.coordinates[0]
        value = cast(torch.Tensor, values[binding.value_id])
        if binding.kind == _REFERENCE_NORMAL_OUTPUT:
            normals[token] = value
        elif binding.kind == _REFERENCE_RECURRENT_OUTPUT:
            recurrents[token] = value
        else:
            raise RuntimeError("signed reference stage has an unsupported output descriptor")
    if set(normals) != set(range(count)) or set(recurrents) != set(range(count)):
        raise RuntimeError("signed reference stage output descriptors do not cover request rows")
    groups = tuple(
        tuple(operation.operation_id for operation in group.operations)
        for group in bound.plan.groups
    )
    return RecirculationExecutionResult(
        logits=tuple(request.decoder.project_logits(normals[token]) for token in range(count)),
        recurrent_states=tuple(recurrents[token].detach().clone() for token in range(count)),
        schedule_kind=(
            "serial"
            if bound.plan.grouping_kind == _core.ExecutionGroupingKind.Serial
            else "wavefront"
        ),
        schedule_groups=groups,
        step_id=int(step_id),
    )


def build_recirculation_graph(
    config: RecirculationConfig,
    *,
    token_count: int,
    layer_count: int,
) -> RecurrenceGraph:
    """Import lazily to retain the public graph implementation as the source."""

    from tllm.consumers.recirculation.graph import build_recirculation_graph as build

    return build(config, token_count=token_count, layer_count=layer_count)


class RecirculationRegionResources:
    """Explicit request/sink bindings for the native Recirculation Region."""

    def __init__(self, workload_instance: "RecirculationRegionWorkload") -> None:
        self._workload = workload_instance
        self._request: RecirculationExecutionRequest | None = None
        self._result: RecirculationExecutionResult | None = None

    def publish_request(self, request: RecirculationExecutionRequest) -> None:
        if not isinstance(request, RecirculationExecutionRequest):
            raise TypeError("request must be RecirculationExecutionRequest")
        self._request = request

    def request_for_step(self, _step_id: int) -> RecirculationExecutionRequest:
        request = self._request
        if request is None:
            raise RuntimeError(
                "Recirculation Region requires an explicit finite execution request; "
                "native vLLM prefill binding is not implemented"
            )
        return request

    def publish_result(self, value: object, step_id: int) -> None:
        if not isinstance(value, RecirculationExecutionResult):
            raise TypeError("Recirculation Region sink requires RecirculationExecutionResult")
        if value.step_id != int(step_id):
            raise RuntimeError("Recirculation Region result step id mismatch")
        self._result = value
        self._request = None

    def result(self) -> RecirculationExecutionResult | None:
        return self._result


_TOKEN = Index("token")
_LAYER = Index("layer")
_RECURRENCE_DOMAIN = Domain(
    "recirculation",
    indices=(_TOKEN, _LAYER),
    bounds=(("token", 0, "token_count"), ("layer", 0, "layer_count")),
    parameters=("token_count", "layer_count"),
)


@workload(id="recirculation.region")
class RecirculationRegionWorkload(NativeWorkload):
    """No-adapter Region declaration for exact finite Recirculation."""

    def __init__(
        self,
        config: RecirculationConfig,
        *,
        stage_artifact_bundle: str | None = None,
        artifact_layer_count: int | None = None,
    ) -> None:
        if not isinstance(config, RecirculationConfig):
            raise TypeError("config must be RecirculationConfig")
        if not config.enabled:
            raise ValueError("Recirculation Region requires enabled=True")
        if config.jacobi is not None:
            raise ValueError("Recirculation Region supports exact execution only")
        if stage_artifact_bundle is not None:
            if type(artifact_layer_count) is not int or artifact_layer_count < 1:
                raise ValueError(
                    "signed exact Recirculation artifacts require artifact_layer_count"
                )
            self._artifacts: _ReferenceStageArtifacts | None = _ReferenceStageArtifacts.load(
                wire=stage_artifact_bundle,
                layer_count=artifact_layer_count,
            )
        else:
            self._artifacts = None
        self.config = config
        self._resources = RecirculationRegionResources(self)
        self._native_resources = self._make_native_resources()

    def _make_native_resources(self) -> NativeWorkloadResources:
        artifacts = self._artifacts
        worker_config = _config_to_wire(
            self.config,
            stage_artifact_bundle=None if artifacts is None else artifacts.wire,
            artifact_layer_count=(None if artifacts is None else artifacts.layer_count),
        )
        return NativeWorkloadResources(
            workload=self,
            provided_inputs={_REQUEST_INPUT: self._resources.request_for_step},
            output_sinks={_RESULT_SINK: self._resources.publish_result},
            specialization_key={
                "algorithm": "recirculation-exact-region-v2",
                "source_layer": str(self.config.source_layer),
                "destination_layer": str(self.config.destination_layer),
                "alpha": float(self.config.alpha).hex(),
                "beta": "default"
                if self.config.beta is None
                else float(self.config.beta).hex(),
                "ramp_tokens": str(self.config.ramp_tokens),
                "stage_artifact_bundle": (
                    "cold-specialization-required"
                    if artifacts is None
                    else artifacts.bundle.signature
                ),
            },
            cold_specializer=self._cold_specialize,
            worker_spec=WorkloadWorkerSpec(
                factory=(
                    "tllm.consumers.recirculation.region_workload:"
                    "reconstruct_recirculation_region_workload"
                ),
                config=worker_config,
            ),
        )

    def _cold_specialize(self, spec: object) -> None:
        from tllm.runtime.specialization import RuntimeSpecializationSpec

        if not isinstance(spec, RuntimeSpecializationSpec):
            raise TypeError(
                "exact Recirculation cold specialization requires RuntimeSpecializationSpec"
            )
        if spec.max_prefill_tokens is None or spec.model_layer_count is None:
            raise RuntimeError(
                "exact Recirculation cold specialization requires max_prefill_tokens "
                "and model_layer_count"
            )
        artifacts = self._artifacts
        if artifacts is None:
            artifacts = _ReferenceStageArtifacts.compile(
                self.config,
                max_row_count=spec.max_prefill_tokens,
                layer_count=spec.model_layer_count,
            )
            self._artifacts = artifacts
        artifacts.validate_runtime_specialization(
            max_row_count=spec.max_prefill_tokens,
            layer_count=spec.model_layer_count,
        )
        self._native_resources = self._make_native_resources()

    @property
    def resources(self) -> RecirculationRegionResources:
        return self._resources

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._native_resources

    @task(
        inputs={"request": provided(_REQUEST_INPUT), "engine_step_id": step_id()},
        outputs={"result": sink(_RESULT_SINK)},
        timing=Before("step.end"),
        implementation=ExternImplementation(
            namespace="tllm.recirculation",
            name="exact_recurrence",
            version="1",
            input_types=("RecirculationExecutionRequest", "int"),
            output_types=("RecirculationExecutionResult",),
            capabilities=(
                "eager_only",
                BackendCapability.RECIRCULATION_FINITE_REQUEST.value,
            ),
        ),
        domain=_RECURRENCE_DOMAIN,
        accesses=(
            AffineAccess("recirculation_kv", (_TOKEN, _LAYER), "write"),
            AffineAccess("recirculation_kv", (_TOKEN - 1, _LAYER), "read"),
        ),
    )
    def execute_exact(
        self,
        request: RecirculationExecutionRequest,
        engine_step_id: int,
    ) -> RecirculationExecutionResult:
        artifacts = self._artifacts
        if artifacts is None:
            raise RuntimeError(
                "specialization-required: exact Recirculation has no signed cold stage artifacts"
            )
        return run_exact_recirculation_region(
            request,
            self.config,
            step_id=engine_step_id,
            artifacts=artifacts,
        )


def build_recirculation_region_workload(
    config: RecirculationConfig,
    *,
    stage_artifact_bundle: str | None = None,
    artifact_layer_count: int | None = None,
) -> RecirculationRegionWorkload:
    return RecirculationRegionWorkload(
        config,
        stage_artifact_bundle=stage_artifact_bundle,
        artifact_layer_count=artifact_layer_count,
    )


def _config_to_wire(
    config: RecirculationConfig,
    *,
    stage_artifact_bundle: str | None = None,
    artifact_layer_count: int | None = None,
) -> Mapping[str, object]:
    if config.jacobi is not None:
        raise ValueError("exact Recirculation Region cannot serialize Jacobi config")
    payload: dict[str, object] = {
            "source_layer": config.source_layer,
            "destination_layer": config.destination_layer,
            "alpha": config.alpha,
            "beta": config.beta,
            "ramp_tokens": config.ramp_tokens,
            "consumer_id": config.consumer_id,
            "enabled": config.enabled,
    }
    if stage_artifact_bundle is not None:
        if artifact_layer_count is None:
            raise ValueError("exact Recirculation wire requires a signed artifact bundle")
        payload[_ARTIFACT_WIRE_KEY] = stage_artifact_bundle
        payload["artifact_layer_count"] = artifact_layer_count
    return MappingProxyType(payload)


def reconstruct_recirculation_region_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    from tllm.common.dataclass_json import dataclass_from_dict

    payload = dict(raw)
    wire = payload.pop(_ARTIFACT_WIRE_KEY, None)
    artifact_layer_count = payload.pop("artifact_layer_count", None)
    if not isinstance(wire, str):
        raise RuntimeError(
            "worker exact Recirculation reconstruction requires a signed cold stage artifact"
        )
    config = dataclass_from_dict(RecirculationConfig, payload)
    return build_recirculation_region_workload(
        config,
        stage_artifact_bundle=wire,
        artifact_layer_count=cast(int, artifact_layer_count),
    ).native_workload_resources()


__all__ = [
    "RecirculationExecutionRequest",
    "RecirculationExecutionResult",
    "RecirculationRegionResources",
    "RecirculationRegionWorkload",
    "build_recirculation_region_workload",
    "reconstruct_recirculation_region_workload",
    "run_exact_recirculation_region",
]
