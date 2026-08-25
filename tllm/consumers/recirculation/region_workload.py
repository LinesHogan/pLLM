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

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import torch

from tllm.contracts.backend_capability import BackendCapability
from tllm.consumers.recirculation.config import RecirculationConfig
from tllm.consumers.recirculation.reference import ReferenceDecoder
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    WorkloadWorkerSpec,
)
from tllm.workgraph.recurrence import (
    DependenceKind,
    OrderedRowFusionCapability,
    RecurrenceGraph,
    RecurrenceOp,
    RecurrenceSchedule,
)
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
_MODE = Literal["serial", "wavefront"]


@dataclass(frozen=True, slots=True)
class RecirculationExecutionRequest:
    """One explicit finite recurrence execution owned by this Region."""

    decoder: ReferenceDecoder
    token_ids: tuple[int, ...]
    mode: _MODE = "serial"

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple) or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in self.token_ids
        ):
            raise TypeError("token_ids must be tuple[int, ...]")
        if not self.token_ids:
            raise ValueError("exact Recirculation requires at least one token")
        if self.mode not in {"serial", "wavefront"}:
            raise ValueError("mode must be 'serial' or 'wavefront'")
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


def _execute_op(
    *,
    op: RecurrenceOp,
    request: RecirculationExecutionRequest,
    config: RecirculationConfig,
    values: dict[str, torch.Tensor],
    logits: dict[int, torch.Tensor],
    recurrent_states: dict[int, torch.Tensor],
) -> None:
    """Execute one finite graph operation in an already-proven schedule order."""

    decoder = request.decoder
    source_layer, destination_layer = config.resolved_layers(decoder.layer_count)
    point = op.point
    token = point.token
    if point.pass_name == "normal":
        if point.layer == 0:
            hidden = decoder.embed(request.token_ids[token], token)
        else:
            hidden = values[_normal(token, point.layer - 1)]
        value = decoder.run_layer(
            point.layer,
            hidden,
            position=token,
            overwrite_kv=False,
        )
        values[op.op_id] = value
        if point.layer == decoder.layer_count - 1:
            logits[token] = decoder.project_logits(value)
        return
    if point.pass_name == "mix":
        source = values[_normal(token, source_layer)]
        destination = values[_normal(token, destination_layer)]
        positions = torch.tensor([token], device=destination.device)
        values[op.op_id] = _mix_tensors(config, source, destination, positions)
        return
    if point.pass_name == "recurrent":
        prior = (
            values[_mix(token)]
            if point.layer == destination_layer + 1
            else values[_recurrent(token, point.layer - 1)]
        )
        value = decoder.run_layer(
            point.layer,
            prior,
            position=token,
            overwrite_kv=True,
        )
        values[op.op_id] = value
        if point.layer == decoder.layer_count - 1:
            recurrent_states[token] = value.detach().clone()
        return
    raise RuntimeError(f"unsupported Recirculation recurrence pass {point.pass_name!r}")


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


def run_exact_recirculation_region(
    request: RecirculationExecutionRequest,
    config: RecirculationConfig,
    *,
    step_id: int,
) -> RecirculationExecutionResult:
    """Run the compiler-owned serial or ordered-row exact finite schedule."""

    if not isinstance(request, RecirculationExecutionRequest):
        raise TypeError("request must be RecirculationExecutionRequest")
    if not isinstance(config, RecirculationConfig):
        raise TypeError("config must be RecirculationConfig")
    if not config.enabled:
        raise RuntimeError("disabled Recirculation cannot create an exact Region")
    if config.jacobi is not None:
        raise RuntimeError("exact Recirculation Region does not accept Jacobi approximation")
    graph = build_recirculation_graph(
        config,
        token_count=len(request.token_ids),
        layer_count=request.decoder.layer_count,
    )
    schedule = _compile_exact_schedule(graph, mode=request.mode)
    by_id = {op.op_id: op for op in graph.ops}
    values: dict[str, torch.Tensor] = {}
    logits: dict[int, torch.Tensor] = {}
    recurrent_states: dict[int, torch.Tensor] = {}
    for group in schedule.groups:
        # A CPU reference executes an ordered row group source-to-target.  A
        # backend may fuse it only after internalizing the same dependence.
        for op_id in group:
            _execute_op(
                op=by_id[op_id],
                request=request,
                config=config,
                values=values,
                logits=logits,
                recurrent_states=recurrent_states,
            )
    count = len(request.token_ids)
    if len(logits) != count or len(recurrent_states) != count:
        raise RuntimeError("exact Recirculation schedule did not complete every token")
    return RecirculationExecutionResult(
        logits=tuple(logits[index] for index in range(count)),
        recurrent_states=tuple(recurrent_states[index] for index in range(count)),
        schedule_kind=schedule.kind,
        schedule_groups=schedule.groups,
        step_id=int(step_id),
    )


def _compile_exact_schedule(
    graph: RecurrenceGraph,
    *,
    mode: _MODE,
) -> RecurrenceSchedule:
    """Use the Rust recurrence compiler for both serial and wavefront modes."""

    if mode == "serial":
        internalized: tuple[str, ...] = ()
        max_rows = 1
    else:
        fusion = OrderedRowFusionCapability(
            frozenset({DependenceKind.KV_OVERWRITE_READ}), max_rows=2
        )
        internalized = tuple(edge.value for edge in fusion.internalized_edges)
        max_rows = fusion.max_rows
    from tllm import _core

    compiled = _core.compile_recurrence_schedule(
        graph.ops,
        graph.edges,
        internalized,
        max_rows,
    )
    schedule = RecurrenceSchedule(
        groups=tuple(tuple(str(op_id) for op_id in group) for group in compiled.groups),
        kind=str(compiled.kind),
        fusion_rejections=tuple(
            (str(source), str(target), str(reason))
            for source, target, reason in compiled.fusion_rejections
        ),
    )
    expected_kind = "serial" if mode == "serial" else "wavefront"
    if schedule.kind != expected_kind:
        raise RuntimeError(
            f"{mode} Recirculation request compiled as {schedule.kind!r}, expected {expected_kind!r}"
        )
    return schedule


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

    def __init__(self, config: RecirculationConfig) -> None:
        if not isinstance(config, RecirculationConfig):
            raise TypeError("config must be RecirculationConfig")
        if not config.enabled:
            raise ValueError("Recirculation Region requires enabled=True")
        if config.jacobi is not None:
            raise ValueError("Recirculation Region supports exact execution only")
        self.config = config
        self._resources = RecirculationRegionResources(self)
        self._native_resources = NativeWorkloadResources(
            workload=self,
            provided_inputs={_REQUEST_INPUT: self._resources.request_for_step},
            output_sinks={_RESULT_SINK: self._resources.publish_result},
            specialization_key={
                "algorithm": "recirculation-exact-region-v1",
                "source_layer": str(config.source_layer),
                "destination_layer": str(config.destination_layer),
                "alpha": float(config.alpha).hex(),
                "beta": "default" if config.beta is None else float(config.beta).hex(),
                "ramp_tokens": str(config.ramp_tokens),
            },
            worker_spec=WorkloadWorkerSpec(
                factory=(
                    "tllm.consumers.recirculation.region_workload:"
                    "reconstruct_recirculation_region_workload"
                ),
                config=_config_to_wire(config),
            ),
        )

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
        return run_exact_recirculation_region(
            request,
            self.config,
            step_id=engine_step_id,
        )


def build_recirculation_region_workload(
    config: RecirculationConfig,
) -> RecirculationRegionWorkload:
    return RecirculationRegionWorkload(config)


def _config_to_wire(config: RecirculationConfig) -> Mapping[str, object]:
    if config.jacobi is not None:
        raise ValueError("exact Recirculation Region cannot serialize Jacobi config")
    return MappingProxyType(
        {
            "source_layer": config.source_layer,
            "destination_layer": config.destination_layer,
            "alpha": config.alpha,
            "beta": config.beta,
            "ramp_tokens": config.ramp_tokens,
            "consumer_id": config.consumer_id,
            "enabled": config.enabled,
        }
    )


def reconstruct_recirculation_region_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    from tllm.common.dataclass_json import dataclass_from_dict

    config = dataclass_from_dict(RecirculationConfig, raw)
    return build_recirculation_region_workload(config).native_workload_resources()


__all__ = [
    "RecirculationExecutionRequest",
    "RecirculationExecutionResult",
    "RecirculationRegionResources",
    "RecirculationRegionWorkload",
    "build_recirculation_region_workload",
    "reconstruct_recirculation_region_workload",
    "run_exact_recirculation_region",
]
