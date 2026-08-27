"""Native vLLM prefill Region for exact Recirculation.

The consumer declares its finite recurrence and owns the resulting slot
stager. Rust compiles that declaration into typed groups and liveness; this
module binds the signed operands to the backend's typed replay resources
(reached only through ``tllm.contracts.typed_replay`` vocabulary and the
``BackendDriver`` protocol) and the consumer's exact mix primitive. No
legacy recurrence plan participates in this path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Callable, Mapping, cast

import tllm._core as _core
import torch

from tllm.backends.protocol import runner_binding_for
from tllm.contracts.backend_capability import BackendCapability
from tllm.contracts.recirculation import RECIRCULATION_PREFILL_BINDER_IMPLEMENTATION_ID
from tllm.contracts.typed_replay import (
    VLLM_FINAL_NORM_PRIMITIVE,
    VLLM_LAYER_REPLAY_PRIMITIVE,
    FinalNormCall,
    LayerReplayCall,
    LayerReplayState,
    TypedLayerReplayHandle,
    TypedLayerReplayRequest,
    TypedLayerReplayStep,
)
from tllm.consumers.recirculation.config import RecirculationConfig
from tllm.consumers.recirculation.graph import build_recirculation_graph
from tllm.runtime.backend_region_binders import (
    BackendRegionBinderSlot,
    BackendRegionStageContext,
)
from tllm.runtime.recirculation import mix_recirculation_tensors
from tllm.runtime.typed_recurrence import (
    PrimitiveAdapter,
    PrimitiveAdapterRegistry,
    TypedRecurrenceExecutor,
)
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
    step_id,
    task,
    workload,
)

PREFILL_BINDER_IMPLEMENTATION = ExternImplementation(
    namespace="tllm.recirculation",
    name="bind_prefill",
    version="1",
    input_types=("TypedRecurrenceStage", "int"),
    output_types=(),
    capabilities=(
        "eager_only",
        BackendCapability.BACKEND_REGION_BINDER.value,
        BackendCapability.RECIRCULATION_PREFILL_BINDING.value,
    ),
)
PREFILL_BINDER_IMPLEMENTATION_ID = RECIRCULATION_PREFILL_BINDER_IMPLEMENTATION_ID
if PREFILL_BINDER_IMPLEMENTATION.key != PREFILL_BINDER_IMPLEMENTATION_ID:
    raise RuntimeError("Recirculation prefill binder identity disagrees with its contract")

_PLAN_INPUT = "typed_recurrence_stage"
_TOKEN = Index("token")
_LAYER = Index("layer")
_DOMAIN = Domain(
    "recirculation_prefill",
    indices=(_TOKEN, _LAYER),
    bounds=(("token", 0, "token_count"), ("layer", 0, "layer_count")),
    parameters=("token_count", "layer_count"),
)
_RECIRCULATION_MIX_PRIMITIVE = "tllm.recirculation:typed_mix@1"
_SEED_STATE_BINDING_KIND = "tllm.vllm:seed-state@1"
_LAYER_REPLAY_BINDING_KIND = "tllm.vllm:layer-replay@1"
_MIX_BINDING_KIND = "tllm.recirculation:mix@1"
_FINAL_NORM_BINDING_KIND = "tllm.vllm:final-norm@1"
_NORMALIZED_OUTPUT_BINDING_KIND = "tllm.vllm:normalized-output@1"
_STAGE_ARTIFACT_BUNDLE_WIRE_KEY = "typed_stage_artifact_bundle"
_STAGE_ARTIFACT_BUNDLE_SCHEMA_VERSION = 1
# Core v1 intentionally compiles a small, exact static family. A prefill span
# beyond this bound requests a new cold epoch instead of compiling in a model
# hook or silently using a different shape.
MAX_COLD_STAGE_ARTIFACT_ROWS = 16


def _normal(token: int, layer: int) -> str:
    return f"t{token}/normal/l{layer}"


def _recurrent(token: int, layer: int) -> str:
    return f"t{token}/recurrent/l{layer}"


def _mix(token: int) -> str:
    return f"t{token}/mix"


def _output(token: int) -> str:
    return f"recirculation/output/{token}"


def _operand(kind: str, field: str, value: str) -> dict[str, str]:
    return {"kind": kind, field: value}


def _finite_recurrence_spec(
    graph: RecurrenceGraph,
    config: RecirculationConfig,
    *,
    region_id: str,
    layer_count: int,
    fusion_policy: str,
) -> dict[str, object]:
    """Declare one exact Recirculation Region without selecting its schedule."""

    source_layer, destination_layer = config.resolved_layers(layer_count)
    operations: list[dict[str, object]] = []
    for op in graph.ops:
        point = op.point
        call_resource = f"vllm_call/{op.op_id}"
        if point.pass_name == "normal":
            inputs = (
                [_operand("seed", "binding_id", f"seed/{op.op_id}")]
                if point.layer == 0
                else [
                    _operand(
                        "region_value",
                        "value_id",
                        _normal(point.token, point.layer - 1),
                    )
                ]
            )
            inputs.append(_operand("backend_resource", "resource_id", call_resource))
            primitive_key = VLLM_LAYER_REPLAY_PRIMITIVE
        elif point.pass_name == "recurrent":
            previous = (
                _mix(point.token)
                if point.layer == destination_layer + 1
                else _recurrent(point.token, point.layer - 1)
            )
            inputs = [
                _operand("region_value", "value_id", previous),
                _operand("backend_resource", "resource_id", call_resource),
            ]
            primitive_key = VLLM_LAYER_REPLAY_PRIMITIVE
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
                _operand(
                    "backend_resource",
                    "resource_id",
                    f"recirculation_mix/{op.op_id}",
                ),
            ]
            primitive_key = _RECIRCULATION_MIX_PRIMITIVE
        else:
            raise RuntimeError(
                f"Recirculation declaration has unsupported pass {point.pass_name!r}"
            )
        operations.append(
            {
                "operation_id": op.op_id,
                "primitive_key": primitive_key,
                "implementation_id": PREFILL_BINDER_IMPLEMENTATION_ID,
                "operands": inputs,
                "results": [op.op_id],
            }
        )
    for token in range(max((op.point.token for op in graph.ops), default=-1) + 1):
        final_normal = _normal(token, layer_count - 1)
        operations.append(
            {
                "operation_id": f"vllm_norm/{token}",
                "primitive_key": VLLM_FINAL_NORM_PRIMITIVE,
                "implementation_id": PREFILL_BINDER_IMPLEMENTATION_ID,
                "operands": [
                    _operand("region_value", "value_id", final_normal),
                    _operand("backend_resource", "resource_id", "vllm_final_norm"),
                ],
                "results": [_output(token)],
            }
        )
    token_count = max((op.point.token for op in graph.ops), default=-1) + 1
    # The final recurrent state has no consumer outside this Region.  Retain
    # only the normalized rows that replace the model forward output.
    retained = [_output(token) for token in range(token_count)]
    dependencies = [
        {"source": edge.source, "target": edge.target, "kind": edge.kind.value}
        for edge in graph.edges
    ]
    dependencies.extend(
        {
            "source": _normal(token, layer_count - 1),
            "target": f"vllm_norm/{token}",
            "kind": "data",
        }
        for token in range(token_count)
    )
    return {
        "schema_version": 1,
        "region_id": region_id,
        "implementation_id": PREFILL_BINDER_IMPLEMENTATION_ID,
        "row_admission": {
            "allowed_phases": ["prefill"],
            "mixed_batch": "reject",
        },
        # This first artifact epoch deliberately uses a serial Region policy.
        # Future cold specializations may sign a stronger backend policy, but
        # a hook never picks or rewrites a policy for an already signed plan.
        "fusion_policy": fusion_policy,
        "operations": operations,
        "dependencies": dependencies,
        "retained_values": retained,
    }


def _stage_shape_specialization(*, row_count: int) -> list[list[str]]:
    """Return the exact static shape identity for one request-local stage."""

    return [
        ["stage/request_count", "1"],
        ["stage/row_count", str(row_count)],
    ]


def _stage_artifact_spec(
    config: RecirculationConfig,
    *,
    row_count: int,
    layer_count: int,
) -> str:
    """Build a cold-only declaration for one exact prefill length.

    This factory is the consumer-owned algorithm declaration boundary. Rust
    validates, schedules, derives liveness, and signs the resulting artifact.
    It is intentionally never called from a vLLM hook.
    """

    graph = build_recirculation_graph(
        config,
        token_count=row_count,
        layer_count=layer_count,
    )
    recurrence = _finite_recurrence_spec(
        graph,
        config,
        region_id=f"recirculation.vllm-prefill/rows-{row_count}",
        layer_count=layer_count,
        fusion_policy="none",
    )
    seed_bindings: list[dict[str, object]] = []
    resource_bindings: list[dict[str, object]] = [
        {
            "resource_id": "vllm_final_norm",
            "kind": _FINAL_NORM_BINDING_KIND,
            "coordinates": [],
        }
    ]
    output_bindings: list[dict[str, object]] = []
    for op in graph.ops:
        point = op.point
        if point.pass_name == "normal" and point.layer == 0:
            seed_bindings.append(
                {
                    "binding_id": f"seed/{op.op_id}",
                    "kind": _SEED_STATE_BINDING_KIND,
                    "coordinates": [point.token],
                }
            )
        if point.pass_name in {"normal", "recurrent"}:
            pass_ordinal = 0 if point.pass_name == "normal" else 1
            resource_bindings.append(
                {
                    "resource_id": f"vllm_call/{op.op_id}",
                    "kind": _LAYER_REPLAY_BINDING_KIND,
                    # A token/layer pair may have both normal and recurrent
                    # calls. The exact signed pass ordinal keeps those
                    # physical replay resources unambiguous at bind time.
                    "coordinates": [point.token, point.layer, pass_ordinal],
                }
            )
        elif point.pass_name == "mix":
            resource_bindings.append(
                {
                    "resource_id": f"recirculation_mix/{op.op_id}",
                    "kind": _MIX_BINDING_KIND,
                    "coordinates": [point.token],
                }
            )
    for token in range(row_count):
        output_bindings.append(
            {
                "output_id": f"result/{token}",
                "value_id": _output(token),
                "kind": _NORMALIZED_OUTPUT_BINDING_KIND,
                "coordinates": [token],
            }
        )
    return json.dumps(
        {
            "schema_version": 1,
            "shape_specialization": _stage_shape_specialization(row_count=row_count),
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


@dataclass(frozen=True, slots=True)
class _RecirculationMixCall:
    config: RecirculationConfig
    token_position: int

    def execute(
        self, source: LayerReplayState, destination: LayerReplayState
    ) -> LayerReplayState:
        mixed = mix_recirculation_tensors(
            self.config,
            source.stream(),
            destination.stream(),
            destination.hidden.new_tensor([self.token_position]),
        )
        return LayerReplayState(mixed, None)


def _primitive_registry() -> PrimitiveAdapterRegistry:
    """Combine backend replay primitives with the consumer's exact mix.

    The replay/norm trampolines dispatch to backend-owned bound call
    resources (``LayerReplayCall``/``FinalNormCall`` over the typed replay
    handle installed through the driver protocol); the consumer owns only
    registry composition and its own mix primitive.
    """

    def replay(_operation: object, operands: tuple[object, ...]) -> tuple[object, ...]:
        state, call = operands
        return (call.execute(state),)  # type: ignore[union-attr]

    def norm(_operation: object, operands: tuple[object, ...]) -> tuple[object, ...]:
        state, call = operands
        return (call.execute(state),)  # type: ignore[union-attr]

    def mix(_operation: object, operands: tuple[object, ...]) -> tuple[object, ...]:
        source, destination, call = operands
        return (call.execute(source, destination),)  # type: ignore[union-attr]

    return PrimitiveAdapterRegistry(
        (
            PrimitiveAdapter(
                PREFILL_BINDER_IMPLEMENTATION_ID,
                VLLM_LAYER_REPLAY_PRIMITIVE,
                replay,
            ),
            PrimitiveAdapter(
                PREFILL_BINDER_IMPLEMENTATION_ID,
                VLLM_FINAL_NORM_PRIMITIVE,
                norm,
            ),
            PrimitiveAdapter(
                PREFILL_BINDER_IMPLEMENTATION_ID,
                _RECIRCULATION_MIX_PRIMITIVE,
                mix,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class _RequestBindingContext:
    handle: TypedLayerReplayHandle
    config: RecirculationConfig
    request_ordinal: int
    packed_rows: tuple[int, ...]
    token_positions: tuple[int, ...]


_DescriptorKey = tuple[str, tuple[int, ...]]
_SeedResolver = Callable[[_RequestBindingContext], object]
_ResourceResolver = Callable[[_RequestBindingContext], object]
_OutputResolver = Callable[[Mapping[str, object], str], tuple[int, object]]


def _descriptor_key(kind: str, coordinates: list[int]) -> _DescriptorKey:
    return kind, tuple(coordinates)


def _one_coordinate(coordinates: tuple[int, ...], *, label: str) -> int:
    if len(coordinates) != 1:
        raise RuntimeError(f"signed {label} descriptor requires exactly one coordinate")
    return coordinates[0]


def _two_coordinates(coordinates: tuple[int, ...], *, label: str) -> tuple[int, int]:
    if len(coordinates) != 2:
        raise RuntimeError(f"signed {label} descriptor requires exactly two coordinates")
    return coordinates[0], coordinates[1]


def _layer_replay_coordinates(coordinates: tuple[int, ...]) -> tuple[int, int]:
    if len(coordinates) != 3:
        raise RuntimeError(
            "signed vLLM layer-replay descriptor requires token, layer, and pass coordinates"
        )
    token, layer, pass_ordinal = coordinates
    if pass_ordinal not in {0, 1}:
        raise RuntimeError(
            "signed vLLM layer-replay descriptor has an unknown pass coordinate"
        )
    return token, layer


def _seed_state_resolver(coordinates: tuple[int, ...]) -> _SeedResolver:
    token = _one_coordinate(coordinates, label="vLLM seed-state")
    return lambda context: context.handle.seed_state(context.packed_rows[token])


def _layer_replay_resolver(coordinates: tuple[int, ...]) -> _ResourceResolver:
    token, layer = _layer_replay_coordinates(coordinates)
    return lambda context: LayerReplayCall(
        handle=context.handle,
        layer=layer,
        request_ordinal=context.request_ordinal,
        packed_row=context.packed_rows[token],
        token_position=context.token_positions[token],
    )


def _mix_resolver(coordinates: tuple[int, ...]) -> _ResourceResolver:
    token = _one_coordinate(coordinates, label="Recirculation mix")
    return lambda context: _RecirculationMixCall(
        config=context.config,
        token_position=context.token_positions[token],
    )


def _final_norm_resolver(coordinates: tuple[int, ...]) -> _ResourceResolver:
    if coordinates:
        raise RuntimeError("signed vLLM final-norm descriptor must not have coordinates")
    return lambda context: FinalNormCall(context.handle)


def _normalized_output_resolver(coordinates: tuple[int, ...]) -> _OutputResolver:
    token = _one_coordinate(coordinates, label="vLLM normalized-output")
    return lambda values, value_id: (token, values[value_id])


_SEED_DESCRIPTOR_FACTORIES: Mapping[str, Callable[[tuple[int, ...]], _SeedResolver]] = (
    MappingProxyType({_SEED_STATE_BINDING_KIND: _seed_state_resolver})
)
_RESOURCE_DESCRIPTOR_FACTORIES: Mapping[
    str, Callable[[tuple[int, ...]], _ResourceResolver]
] = MappingProxyType(
    {
        _LAYER_REPLAY_BINDING_KIND: _layer_replay_resolver,
        _MIX_BINDING_KIND: _mix_resolver,
        _FINAL_NORM_BINDING_KIND: _final_norm_resolver,
    }
)
_OUTPUT_DESCRIPTOR_FACTORIES: Mapping[
    str, Callable[[tuple[int, ...]], _OutputResolver]
] = MappingProxyType({_NORMALIZED_OUTPUT_BINDING_KIND: _normalized_output_resolver})


def _compile_seed_descriptor_registry(
    bindings: list[_core.TypedStageBinding],
) -> Mapping[_DescriptorKey, _SeedResolver]:
    resolved: dict[_DescriptorKey, _SeedResolver] = {}
    for binding in bindings:
        key = _descriptor_key(binding.kind, binding.coordinates)
        try:
            factory = _SEED_DESCRIPTOR_FACTORIES[binding.kind]
        except KeyError as exc:
            raise RuntimeError(
                "signed typed Recirculation artifact has no seed descriptor adapter "
                f"for {binding.kind!r}"
            ) from exc
        if key in resolved:
            raise RuntimeError(
                f"signed typed Recirculation artifact repeats seed descriptor {key!r}"
            )
        resolved[key] = factory(key[1])
    return MappingProxyType(resolved)


def _compile_resource_descriptor_registry(
    bindings: list[_core.TypedStageResourceBinding],
) -> Mapping[_DescriptorKey, _ResourceResolver]:
    resolved: dict[_DescriptorKey, _ResourceResolver] = {}
    for binding in bindings:
        key = _descriptor_key(binding.kind, binding.coordinates)
        try:
            factory = _RESOURCE_DESCRIPTOR_FACTORIES[binding.kind]
        except KeyError as exc:
            raise RuntimeError(
                "signed typed Recirculation artifact has no resource descriptor adapter "
                f"for {binding.kind!r}"
            ) from exc
        if key in resolved:
            raise RuntimeError(
                f"signed typed Recirculation artifact repeats resource descriptor {key!r}"
            )
        resolved[key] = factory(key[1])
    return MappingProxyType(resolved)


def _compile_output_descriptor_registry(
    bindings: list[_core.TypedStageOutputBinding],
) -> Mapping[_DescriptorKey, _OutputResolver]:
    resolved: dict[_DescriptorKey, _OutputResolver] = {}
    for binding in bindings:
        key = _descriptor_key(binding.kind, binding.coordinates)
        try:
            factory = _OUTPUT_DESCRIPTOR_FACTORIES[binding.kind]
        except KeyError as exc:
            raise RuntimeError(
                "signed typed Recirculation artifact has no output descriptor adapter "
                f"for {binding.kind!r}"
            ) from exc
        if key in resolved:
            raise RuntimeError(
                f"signed typed Recirculation artifact repeats output descriptor {key!r}"
            )
        resolved[key] = factory(key[1])
    return MappingProxyType(resolved)


@dataclass(frozen=True, slots=True)
class _StageDescriptorRegistry:
    seeds: Mapping[_DescriptorKey, _SeedResolver]
    resources: Mapping[_DescriptorKey, _ResourceResolver]
    outputs: Mapping[_DescriptorKey, _OutputResolver]

    @classmethod
    def from_artifact(
        cls, artifact: _core.TypedRecurrenceStageArtifact
    ) -> "_StageDescriptorRegistry":
        return cls(
            seeds=_compile_seed_descriptor_registry(artifact.seed_bindings),
            resources=_compile_resource_descriptor_registry(artifact.resource_bindings),
            outputs=_compile_output_descriptor_registry(artifact.output_bindings),
        )

    def seed_values(
        self,
        bindings: list[_core.TypedStageBinding],
        context: _RequestBindingContext,
    ) -> Mapping[str, object]:
        values: dict[str, object] = {}
        for binding in bindings:
            try:
                resolver = self.seeds[_descriptor_key(binding.kind, binding.coordinates)]
            except KeyError as exc:
                raise RuntimeError("bound typed Recirculation seed descriptor is not signed") from exc
            values[binding.binding_id] = resolver(context)
        return values

    def resource_values(
        self,
        bindings: list[_core.TypedStageResourceBinding],
        context: _RequestBindingContext,
    ) -> Mapping[str, object]:
        values: dict[str, object] = {}
        for binding in bindings:
            try:
                resolver = self.resources[
                    _descriptor_key(binding.kind, binding.coordinates)
                ]
            except KeyError as exc:
                raise RuntimeError(
                    "bound typed Recirculation resource descriptor is not signed"
                ) from exc
            values[binding.resource_id] = resolver(context)
        return values

    def output_values(
        self,
        bindings: list[_core.TypedStageOutputBinding],
        values: Mapping[str, object],
    ) -> Mapping[int, object]:
        results: dict[int, object] = {}
        for binding in bindings:
            try:
                resolver = self.outputs[_descriptor_key(binding.kind, binding.coordinates)]
            except KeyError as exc:
                raise RuntimeError(
                    "bound typed Recirculation output descriptor is not signed"
                ) from exc
            token, value = resolver(values, binding.value_id)
            if token in results:
                raise RuntimeError("bound typed Recirculation output token is not unique")
            results[token] = value
        return results


@dataclass(frozen=True, slots=True)
class _TypedStageArtifactBundle:
    wire: str
    signature: str
    bundle: _core.TypedRecurrenceStageArtifactBundle
    max_row_count: int
    descriptors_by_signature: Mapping[str, _StageDescriptorRegistry]

    @classmethod
    def compile(
        cls,
        config: RecirculationConfig,
        *,
        max_row_count: int,
        layer_count: int,
    ) -> "_TypedStageArtifactBundle":
        artifact_wires = {
            row_count: _core.compile_typed_recurrence_stage_artifact_json(
                _stage_artifact_spec(
                    config,
                    row_count=row_count,
                    layer_count=layer_count,
                )
            )
            for row_count in range(1, max_row_count + 1)
        }
        return cls.load(
            _core.compile_typed_recurrence_stage_artifact_bundle_json(
                json.dumps(
                    {
                        "schema_version": _STAGE_ARTIFACT_BUNDLE_SCHEMA_VERSION,
                        "artifacts": [
                            {
                                "row_count": row_count,
                                "artifact_json": artifact_json,
                            }
                            for row_count, artifact_json in artifact_wires.items()
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        )

    @classmethod
    def load(cls, wire: str) -> "_TypedStageArtifactBundle":
        bundle = _core.load_typed_recurrence_stage_artifact_bundle(wire)
        max_row_count = max(entry.row_count for entry in bundle.artifacts)
        descriptors_by_signature = {
            entry.artifact.signature: _StageDescriptorRegistry.from_artifact(
                entry.artifact
            )
            for entry in bundle.artifacts
        }
        return cls(
            wire=wire,
            signature=bundle.signature,
            bundle=bundle,
            max_row_count=max_row_count,
            descriptors_by_signature=MappingProxyType(descriptors_by_signature),
        )

    def validate_runtime_specialization(
        self,
        *,
        max_row_count: int,
        layer_count: int,
        config: RecirculationConfig,
    ) -> None:
        # The signed artifacts already contain the recurrence graph and its
        # layer coordinates. Rebuilding the declaration here would be a hidden
        # worker-side compilation path, so Rust bind validates only fixed facts.
        config.resolved_layers(layer_count)
        if max_row_count != self.max_row_count:
            raise RuntimeError(
                "typed Recirculation signed artifact capacity does not match "
                "the cold specialization"
            )
        for row_count in range(1, self.max_row_count + 1):
            artifact = _core.typed_recurrence_stage_artifact_for_rows(
                self.bundle, row_count
            )
            if artifact.signature not in self.descriptors_by_signature:
                raise RuntimeError(
                    "typed Recirculation artifact has no cold-bound descriptor registry"
                )


@dataclass(frozen=True, slots=True)
class _TypedRecirculationRequest:
    request_id: str
    request_ordinal: int
    packed_rows: tuple[int, ...]
    token_positions: tuple[int, ...]
    bound_stage: _core.TypedBoundRecurrenceStage
    descriptors: _StageDescriptorRegistry
    config: RecirculationConfig

    def replay(self, handle: TypedLayerReplayHandle) -> Mapping[int, torch.Tensor]:
        context = _RequestBindingContext(
            handle=handle,
            config=self.config,
            request_ordinal=self.request_ordinal,
            packed_rows=self.packed_rows,
            token_positions=self.token_positions,
        )
        values = TypedRecurrenceExecutor(_primitive_registry()).execute(
            self.bound_stage.plan,
            seed_values=self.descriptors.seed_values(
                self.bound_stage.seed_bindings, context
            ),
            backend_resources=self.descriptors.resource_values(
                self.bound_stage.resource_bindings, context
            ),
        )
        return cast(
            Mapping[int, torch.Tensor],
            self.descriptors.output_values(self.bound_stage.output_bindings, values),
        )


@dataclass(frozen=True, slots=True)
class _TypedRecirculationStage:
    total_rows: int
    requests: tuple[_TypedRecirculationRequest, ...]

    def to_replay_step(self) -> TypedLayerReplayStep | None:
        if self.total_rows == 0:
            return None
        return TypedLayerReplayStep(
            total_rows=self.total_rows,
            requests=tuple(
                TypedLayerReplayRequest(
                    request_id=request.request_id,
                    packed_rows=request.packed_rows,
                    replay=request.replay,
                )
                for request in self.requests
            ),
        )


def _stage_typed_recirculation(
    context: BackendRegionStageContext,
    config: RecirculationConfig,
    artifacts: _TypedStageArtifactBundle | None,
) -> None:
    """Bind existing Rust artifacts to one exact vLLM row plan."""

    if not isinstance(context, BackendRegionStageContext):
        raise TypeError("vLLM typed Recirculation staging requires BackendRegionStageContext")
    if artifacts is None:
        raise RuntimeError(
            "specialization-required: typed Recirculation has no cold stage artifact bundle"
        )
    # The concrete replay machinery is backend-owned; this binder reaches it
    # only through the runner binding's driver protocol.
    binding = runner_binding_for(context.runner)
    if binding is None or not binding.active:
        raise RuntimeError("typed Recirculation staging requires an active runner binding")
    driver = binding.driver
    handle = driver.install_typed_layer_replay(runtime=context.runtime, runner=context.runner)
    try:
        columns = driver.fast_weight_packed_row_plan_inputs(
            runner=context.runner,
            num_scheduled_tokens=context.num_scheduled_tokens,
            input_batch=context.input_batch,
        )
        row_plan = _core.plan_packed_prefill_rows(
            list(columns.scheduled_tokens),
            list(columns.prompt_tokens),
            list(columns.computed_tokens),
        )
        admission = _core.admit_typed_recurrence_stage_bundle_for_packed_plan(
            artifacts.bundle, row_plan
        )
        if admission == _core.TypedStagePlanAdmission.Noop:
            stage = _TypedRecirculationStage(total_rows=0, requests=())
        elif admission == _core.TypedStagePlanAdmission.Execute:
            requests: list[_TypedRecirculationRequest] = []
            for span in row_plan.request_spans:
                try:
                    packed_binding = (
                        _core.bind_typed_recurrence_stage_bundle_for_packed_span(
                            artifacts.bundle,
                            row_plan,
                            int(span.request_ordinal),
                        )
                    )
                except _core.TypedRecurrenceSpecializationRequired as exc:
                    raise RuntimeError(
                        "specialization-required: typed Recirculation has no signed "
                        "artifact for the observed packed span"
                    ) from exc
                except ValueError as exc:
                    raise RuntimeError(
                        f"Rust typed Recirculation packed-span admission rejected: {exc}"
                    ) from exc
                observed = packed_binding.observed_span
                bound_stage = packed_binding.stage
                ordinal = int(observed.request_ordinal)
                request_id = columns.request_ids[ordinal]
                if not isinstance(request_id, str):
                    raise RuntimeError(
                        "typed Recirculation requires string vLLM request identifiers"
                    )
                try:
                    descriptors = artifacts.descriptors_by_signature[
                        bound_stage.signature
                    ]
                except KeyError as exc:
                    raise RuntimeError(
                        "Rust-bound typed Recirculation stage has no signed descriptor registry"
                    ) from exc
                requests.append(
                    _TypedRecirculationRequest(
                        request_id=request_id,
                        request_ordinal=ordinal,
                        packed_rows=tuple(
                            range(
                                int(observed.packed_row_start),
                                int(observed.packed_row_end),
                            )
                        ),
                        token_positions=tuple(
                            range(
                                int(observed.token_offset_start),
                                int(observed.token_offset_end),
                            )
                        ),
                        bound_stage=bound_stage,
                        descriptors=descriptors,
                        config=config,
                    )
                )
            stage = _TypedRecirculationStage(
                total_rows=int(row_plan.total_rows),
                requests=tuple(requests),
            )
        else:
            raise RuntimeError("Rust typed recurrence returned an unknown stage admission")

        def bind(
            staged: object,
            _inputs: Mapping[str, object],
            _step_id: int,
        ) -> None:
            if staged is not stage:
                raise RuntimeError("typed Recirculation binder received a different staged plan")
            replay_step = cast(_TypedRecirculationStage, staged).to_replay_step()
            if replay_step is not None:
                handle.prepare(replay_step)

        context.slot.stage(stage, bind, context.step_id)
    except Exception:
        driver.restore_typed_layer_replay(runtime=context.runtime)
        raise


@workload(id="recirculation.vllm-prefill")
class VllmRecirculationRegionWorkload(NativeWorkload):
    """Bind a Rust-compiled typed Region before one vLLM model forward."""

    def __init__(
        self,
        config: RecirculationConfig,
        *,
        stage_artifact_bundle_json: str | None = None,
    ) -> None:
        if not isinstance(config, RecirculationConfig):
            raise TypeError("config must be RecirculationConfig")
        if not config.enabled:
            raise ValueError("vLLM Recirculation Region requires enabled=True")
        if config.jacobi is not None:
            raise ValueError("vLLM Recirculation Region supports exact execution only")
        self.config = config
        self._stage_artifacts = (
            None
            if stage_artifact_bundle_json is None
            else _TypedStageArtifactBundle.load(stage_artifact_bundle_json)
        )

        def stage(context: BackendRegionStageContext) -> None:
            _stage_typed_recirculation(context, self.config, self._stage_artifacts)

        self._binder = BackendRegionBinderSlot(
            PREFILL_BINDER_IMPLEMENTATION_ID,
            backend_stager=stage,
        )
        self._native_resources = self._make_native_resources()

    def _make_native_resources(self) -> NativeWorkloadResources:
        artifacts = self._stage_artifacts
        worker_config = _config_to_wire(
            self.config,
            stage_artifact_bundle_json=None if artifacts is None else artifacts.wire,
        )
        specialization_key = {
            "algorithm": "recirculation-vllm-typed-region-v3",
            "source_layer": str(self.config.source_layer),
            "destination_layer": str(self.config.destination_layer),
            "alpha": float(self.config.alpha).hex(),
            "beta": "default" if self.config.beta is None else float(self.config.beta).hex(),
            "ramp_tokens": str(self.config.ramp_tokens),
            "stage_artifact_bundle": (
                "cold-specialization-required"
                if artifacts is None
                else artifacts.signature
            ),
            "stage_artifact_max_rows": (
                "cold-specialization-required"
                if artifacts is None
                else str(artifacts.max_row_count)
            ),
        }
        return NativeWorkloadResources(
            workload=self,
            provided_inputs={_PLAN_INPUT: self._binder.plan_for_step},
            cold_specializer=self._cold_specialize,
            specialization_key=specialization_key,
            backend_region_binders={
                PREFILL_BINDER_IMPLEMENTATION_ID: self._binder
            },
            worker_spec=WorkloadWorkerSpec(
                factory=(
                    "tllm.consumers.recirculation.vllm_region_workload:"
                    "reconstruct_vllm_recirculation_region_workload"
                ),
                config=worker_config,
            ),
        )

    def _cold_specialize(self, spec: object) -> None:
        """Compile in the coordinator, or load/validate in a worker only."""

        from tllm.runtime.specialization import RuntimeSpecializationSpec

        if not isinstance(spec, RuntimeSpecializationSpec):
            raise TypeError("typed Recirculation cold specialization requires RuntimeSpecializationSpec")
        layer_count = spec.model_layer_count
        if layer_count is None:
            raise RuntimeError(
                "typed Recirculation cold specialization requires model_layer_count"
            )
        max_prefill_tokens = spec.max_prefill_tokens
        if max_prefill_tokens is None:
            raise RuntimeError(
                "typed Recirculation cold specialization requires max_prefill_tokens"
            )
        artifact_max_rows = min(max_prefill_tokens, MAX_COLD_STAGE_ARTIFACT_ROWS)
        artifacts = self._stage_artifacts
        if artifacts is None:
            artifacts = _TypedStageArtifactBundle.compile(
                self.config,
                max_row_count=artifact_max_rows,
                layer_count=layer_count,
            )
            self._stage_artifacts = artifacts
        artifacts.validate_runtime_specialization(
            max_row_count=artifact_max_rows,
            layer_count=layer_count,
            config=self.config,
        )
        self._native_resources = self._make_native_resources()

    @property
    def binder(self) -> BackendRegionBinderSlot:
        return self._binder

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._native_resources

    @task(
        inputs={"plan": provided(_PLAN_INPUT), "engine_step_id": step_id()},
        timing=Before("model.execute"),
        implementation=PREFILL_BINDER_IMPLEMENTATION,
        domain=_DOMAIN,
        accesses=(
            AffineAccess("vllm_kv", (_TOKEN, _LAYER), "write"),
            AffineAccess("vllm_kv", (_TOKEN - 1, _LAYER), "read"),
        ),
    )
    def bind_prefill(self, plan: object, engine_step_id: int) -> None:
        if plan is not self._binder.plan_for_step(engine_step_id):
            raise RuntimeError("typed Recirculation Region received a different staged plan")
        self._binder.invoke(engine_step_id)


def build_vllm_recirculation_region_workload(
    config: RecirculationConfig,
    *,
    stage_artifact_bundle_json: str | None = None,
) -> VllmRecirculationRegionWorkload:
    return VllmRecirculationRegionWorkload(
        config,
        stage_artifact_bundle_json=stage_artifact_bundle_json,
    )


def _config_to_wire(
    config: RecirculationConfig,
    *,
    stage_artifact_bundle_json: str | None,
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "source_layer": config.source_layer,
        "destination_layer": config.destination_layer,
        "alpha": config.alpha,
        "beta": config.beta,
        "ramp_tokens": config.ramp_tokens,
        "consumer_id": config.consumer_id,
        "enabled": config.enabled,
    }
    if stage_artifact_bundle_json is not None:
        payload[_STAGE_ARTIFACT_BUNDLE_WIRE_KEY] = stage_artifact_bundle_json
    return MappingProxyType(payload)


def reconstruct_vllm_recirculation_region_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    from tllm.common.dataclass_json import dataclass_from_dict

    payload = dict(raw)
    artifact_bundle = payload.pop(_STAGE_ARTIFACT_BUNDLE_WIRE_KEY, None)
    if not isinstance(artifact_bundle, str):
        raise RuntimeError(
            "worker typed Recirculation reconstruction requires a signed cold artifact bundle"
        )
    config = dataclass_from_dict(RecirculationConfig, payload)
    return build_vllm_recirculation_region_workload(
        config,
        stage_artifact_bundle_json=artifact_bundle,
    ).native_workload_resources()


__all__ = [
    "MAX_COLD_STAGE_ARTIFACT_ROWS",
    "PREFILL_BINDER_IMPLEMENTATION",
    "PREFILL_BINDER_IMPLEMENTATION_ID",
    "VllmRecirculationRegionWorkload",
    "build_vllm_recirculation_region_workload",
    "reconstruct_vllm_recirculation_region_workload",
]
