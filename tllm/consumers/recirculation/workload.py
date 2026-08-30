"""Exact Recirculation workload: one config and one signed backend task."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

from tllm.kit.backend import WorkloadBackendAdapter
from tllm.kit.workload import Workload
from tllm.contracts.backend_capability import BackendCapability
from tllm.workload import (
    AffineAccess,
    Before,
    ConfigField,
    ConfigKind,
    Domain,
    Index,
    NumericConstraint,
    WorkloadConfig,
    config_field,
    declared_config_field,
    extern,
    step_id,
    task,
    workload,
)

from tllm.consumers.recirculation.native_backend import (
    MAX_COLD_STAGE_ARTIFACT_ROWS,
    PREFILL_BINDER_IMPLEMENTATION_ID,
    RecirculationBackend,
)

_TOKEN = Index("token")
_LAYER = Index("layer")
_RECIRCULATION_ENABLED = ConfigField.boolean(True)


class RecirculationConfig(WorkloadConfig):
    source_layer: int = config_field(default=-1, kind=ConfigKind.INT)
    destination_layer: int = config_field(default=0, kind=ConfigKind.INT)
    alpha: float = config_field(
        default=0.15, kind=ConfigKind.FLOAT, constraint=NumericConstraint.UNIT_INTERVAL
    )
    beta: float = config_field(
        default=0.85, kind=ConfigKind.FLOAT, constraint=NumericConstraint.UNIT_INTERVAL
    )
    ramp_tokens: int = config_field(
        default=0, kind=ConfigKind.INT, constraint=NumericConstraint.NONNEGATIVE
    )
    enabled: bool = declared_config_field(_RECIRCULATION_ENABLED)

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
        inputs={"step_id": step_id()},
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
    def bind_prefill(self, step_id: int) -> None:
        self.backend.invoke(step_id)

    def _diagnostics(self) -> Mapping[str, int]:
        return MappingProxyType(
            {**super()._diagnostics(), **self.backend.diagnostics()}
        )


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
