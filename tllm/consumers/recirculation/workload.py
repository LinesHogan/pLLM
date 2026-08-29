"""Exact Recirculation workload: one config and one signed backend task."""

from __future__ import annotations

from typing import ClassVar, Protocol, cast

from tllm.kit.backend import WorkloadBackendAdapter
from tllm.kit.workload import Workload
from tllm.contracts.backend_capability import BackendCapability
from tllm.workload import (
    AffineAccess,
    Before,
    ConfigField,
    Domain,
    Index,
    NumericConstraint,
    WorkloadConfig,
    config_field,
    extern,
    provided,
    step_id,
    task,
    workload,
)

from tllm.consumers.recirculation.native_backend import (
    MAX_COLD_STAGE_ARTIFACT_ROWS,
    PREFILL_BINDER_IMPLEMENTATION_ID,
    RecirculationBackend,
)

_PLAN_INPUT = "typed_recurrence_stage"
_TOKEN = Index("token")
_LAYER = Index("layer")
_RECIRCULATION_ENABLED = ConfigField(True)


class RecirculationStage(Protocol):
    pass


class RecirculationConfig(WorkloadConfig):
    source_layer: int = config_field(default=-1)
    destination_layer: int = config_field(default=0)
    alpha: float = config_field(
        default=0.15, constraint=NumericConstraint.UNIT_INTERVAL
    )
    beta: float = config_field(
        default=0.85, constraint=NumericConstraint.UNIT_INTERVAL
    )
    ramp_tokens: int = config_field(
        default=0, constraint=NumericConstraint.NONNEGATIVE
    )
    enabled: bool = cast(bool, _RECIRCULATION_ENABLED)

    def resolved_layers(self, layer_count: int) -> tuple[int, int]:
        source = self.source_layer if self.source_layer >= 0 else layer_count + self.source_layer
        destination = (
            self.destination_layer
            if self.destination_layer >= 0
            else layer_count + self.destination_layer
        )
        if not 0 <= destination < source < layer_count:
            raise ValueError(
                "Recirculation layers require destination < source within the model"
            )
        return source, destination


class RecirculationWorkload(Workload[RecirculationConfig]):
    backend_type: ClassVar[type[WorkloadBackendAdapter]] = RecirculationBackend
    backend: RecirculationBackend

    @task(
        inputs={"plan": provided(_PLAN_INPUT), "step_id": step_id()},
        timing=Before("model.execute"),
        implementation=extern(
            PREFILL_BINDER_IMPLEMENTATION_ID,
            capabilities=(
                "eager_only",
                BackendCapability.BACKEND_REGION_BINDER.value,
                BackendCapability.TYPED_LAYER_REPLAY.value,
            ),
        ),
        domain=Domain(
            "recirculation_prefill",
            indices=(_TOKEN, _LAYER),
            bounds=(("token", 0, "token_count"), ("layer", 0, "layer_count")),
            parameters=("token_count", "layer_count"),
        ),
        accesses=(
            AffineAccess("vllm_kv", (_TOKEN, _LAYER), "write"),
            AffineAccess("vllm_kv", (_TOKEN - 1, _LAYER), "read"),
        ),
    )
    def bind_prefill(self, plan: RecirculationStage, step_id: int) -> None:
        del plan
        self.backend.invoke(step_id)


workload(
    RecirculationWorkload,
    id="recirculation",
    config=RecirculationConfig,
    enabled=_RECIRCULATION_ENABLED,
)


__all__ = [
    "MAX_COLD_STAGE_ARTIFACT_ROWS",
    "RecirculationConfig",
    "RecirculationWorkload",
]
