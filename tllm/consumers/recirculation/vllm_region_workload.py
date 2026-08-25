"""Native vLLM prefill binding Region for exact Recirculation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import tllm._core as _core
from tllm.contracts.backend_capability import BackendCapability
from tllm.contracts.recirculation import (
    RECIRCULATION_PREFILL_BINDER_IMPLEMENTATION_ID,
    RecirculationPrefillBindingConfig,
)
from tllm.consumers.recirculation.config import RecirculationConfig
from tllm.runtime.backend_region_binders import (
    BackendRegionBinderSlot,
)
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    WorkloadWorkerSpec,
)
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
    input_types=("RecirculationPrefillPlan", "int"),
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

_PLAN_INPUT = "recirculation_prefill_plan"
_TOKEN = Index("token")
_LAYER = Index("layer")
_DOMAIN = Domain(
    "recirculation_prefill",
    indices=(_TOKEN, _LAYER),
    bounds=(("token", 0, "token_count"), ("layer", 0, "layer_count")),
    parameters=("token_count", "layer_count"),
)


@workload(id="recirculation.vllm-prefill")
class VllmRecirculationRegionWorkload(NativeWorkload):
    """Bind one Rust-validated packed prefill plan before model execution."""

    def __init__(self, config: RecirculationConfig) -> None:
        if not isinstance(config, RecirculationConfig):
            raise TypeError("config must be RecirculationConfig")
        if not config.enabled:
            raise ValueError("vLLM Recirculation Region requires enabled=True")
        if config.jacobi is not None:
            raise ValueError("vLLM Recirculation Region supports exact execution only")
        self.config = config
        self.binding_config = RecirculationPrefillBindingConfig(
            source_layer=config.source_layer,
            destination_layer=config.destination_layer,
            alpha=config.alpha,
            beta=config.beta,
            ramp_tokens=config.ramp_tokens,
        )
        self._binder = BackendRegionBinderSlot(
            PREFILL_BINDER_IMPLEMENTATION_ID,
            provided_inputs={"config": lambda _step_id: self.binding_config},
        )
        self._native_resources = NativeWorkloadResources(
            workload=self,
            provided_inputs={_PLAN_INPUT: self._binder.plan_for_step},
            specialization_key={
                "algorithm": "recirculation-vllm-prefill-region-v1",
                "source_layer": str(config.source_layer),
                "destination_layer": str(config.destination_layer),
                "alpha": float(config.alpha).hex(),
                "beta": "default" if config.beta is None else float(config.beta).hex(),
                "ramp_tokens": str(config.ramp_tokens),
            },
            backend_region_binders={
                PREFILL_BINDER_IMPLEMENTATION_ID: self._binder
            },
            worker_spec=WorkloadWorkerSpec(
                factory=(
                    "tllm.consumers.recirculation.vllm_region_workload:"
                    "reconstruct_vllm_recirculation_region_workload"
                ),
                config=_config_to_wire(config),
            ),
        )

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
    def bind_prefill(
        self, plan: _core.RecirculationPrefillPlan, engine_step_id: int
    ) -> None:
        if plan is not self._binder.plan_for_step(engine_step_id):
            raise RuntimeError("Recirculation Region received a different staged plan")
        self._binder.invoke(engine_step_id)


def build_vllm_recirculation_region_workload(
    config: RecirculationConfig,
) -> VllmRecirculationRegionWorkload:
    return VllmRecirculationRegionWorkload(config)


def _config_to_wire(config: RecirculationConfig) -> Mapping[str, object]:
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


def reconstruct_vllm_recirculation_region_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    from tllm.common.dataclass_json import dataclass_from_dict

    config = dataclass_from_dict(RecirculationConfig, raw)
    return build_vllm_recirculation_region_workload(config).native_workload_resources()


__all__ = [
    "PREFILL_BINDER_IMPLEMENTATION",
    "PREFILL_BINDER_IMPLEMENTATION_ID",
    "VllmRecirculationRegionWorkload",
    "build_vllm_recirculation_region_workload",
    "reconstruct_vllm_recirculation_region_workload",
]
