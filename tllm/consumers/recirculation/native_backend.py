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
from types import MappingProxyType
from typing import Mapping, cast

import tllm._core as _core
import torch

from tllm.backends.protocol import runner_binding_for
from tllm.contracts.typed_replay import (
    TypedLayerReplayHandle,
    TypedLayerReplayRequest,
    TypedLayerReplayStep,
)
from tllm.consumers.recirculation.spec import RecirculationSettings
from tllm.consumers.recirculation.stage_binding import (
    RecirculationBindingProvider,
)
from tllm.consumers.recirculation.stage_definition import (
    PREFILL_BINDER_IMPLEMENTATION_ID,
    build_stage_definitions,
)
from tllm.runtime.backend_region_binders import (
    BackendRegionBinderSlot,
    BackendRegionStageContext,
)
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.kit.backend import BackendRuntimeBinding
from tllm.runtime.workload_bindings import RuntimePortRegistry
from tllm.workload import WorkloadConfig

_PLAN_INPUT = "typed_recurrence_stage"
_STAGE_ARTIFACT_BUNDLE_WIRE_KEY = "typed_stage_artifact_bundle"
# Core v1 intentionally compiles a small, exact static family. A prefill span
# beyond this bound requests a new cold epoch instead of compiling in a model
# hook or silently using a different shape.
MAX_COLD_STAGE_ARTIFACT_ROWS = 16


@dataclass(frozen=True, slots=True)
class _TypedRecirculationRequest:
    request_id: str
    request_ordinal: int
    packed_rows: tuple[int, ...]
    token_positions: tuple[int, ...]
    bound_stage: _core.TypedBoundRecurrenceStage
    config: RecirculationSettings

    def replay(self, handle: TypedLayerReplayHandle) -> Mapping[int, torch.Tensor]:
        return RecirculationBindingProvider(
            handle=handle,
            config=self.config,
            request_ordinal=self.request_ordinal,
            packed_rows=self.packed_rows,
            token_positions=self.token_positions,
        ).execute(self.bound_stage)


@dataclass(frozen=True, slots=True)
class _NoopRecirculationStage:
    def prepare(self, handle: TypedLayerReplayHandle) -> None:
        del handle


@dataclass(frozen=True, slots=True)
class _ExecuteRecirculationStage:
    total_rows: int
    requests: tuple[_TypedRecirculationRequest, ...]

    def prepare(self, handle: TypedLayerReplayHandle) -> None:
        handle.prepare(TypedLayerReplayStep(
            total_rows=self.total_rows,
            requests=tuple(
                TypedLayerReplayRequest(
                    request_id=request.request_id,
                    packed_rows=request.packed_rows,
                    replay=request.replay,
                )
                for request in self.requests
            ),
        ))


_RecirculationStage = _NoopRecirculationStage | _ExecuteRecirculationStage


def _stage_typed_recirculation(
    context: BackendRegionStageContext,
    config: RecirculationSettings,
    artifacts: _core.TypedRecurrenceStageArtifactBundle,
) -> None:
    """Bind existing Rust artifacts to one exact vLLM row plan."""

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
        try:
            batch = artifacts.bind_packed_plan(row_plan)
        except _core.TypedRecurrenceSpecializationRequired as exc:
            raise RuntimeError(
                "specialization-required: typed Recirculation has no signed "
                "artifact for the observed packed span"
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                f"Rust typed Recirculation packed-plan admission rejected: {exc}"
            ) from exc
        if batch.admission == _core.TypedStagePlanAdmission.Noop:
            stage: _RecirculationStage = _NoopRecirculationStage()
        elif batch.admission == _core.TypedStagePlanAdmission.Execute:
            requests: list[_TypedRecirculationRequest] = []
            for packed_binding in batch.requests:
                observed = packed_binding.observed_span
                bound_stage = packed_binding.stage
                ordinal = int(observed.request_ordinal)
                request_id = columns.request_ids[ordinal]
                if not isinstance(request_id, str):
                    raise RuntimeError(
                        "typed Recirculation requires string vLLM request identifiers"
                    )
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
                        config=config,
                    )
                )
            stage = _ExecuteRecirculationStage(
                total_rows=batch.total_rows,
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
            cast(_RecirculationStage, staged).prepare(handle)

        context.slot.stage(stage, bind, context.step_id)
    except Exception:
        driver.restore_typed_layer_replay(runtime=context.runtime)
        raise


@dataclass(frozen=True, slots=True)
class _ColdBackend:
    pass


@dataclass(frozen=True, slots=True)
class _ReadyBackend:
    artifacts: _core.TypedRecurrenceStageArtifactBundle


class RecirculationBackend:
    """Consumer-owned adapter from signed recurrence artifacts to vLLM replay."""

    def __init__(self, config: RecirculationSettings) -> None:
        self.config = config
        self._state: _ColdBackend | _ReadyBackend = _ColdBackend()
        self.binder = BackendRegionBinderSlot(
            PREFILL_BINDER_IMPLEMENTATION_ID,
            backend_stager=self._stage,
        )

    @property
    def requires_specialization(self) -> bool:
        return True

    @classmethod
    def create(cls, config: WorkloadConfig) -> "RecirculationBackend":
        return cls(cast(RecirculationSettings, config))

    @classmethod
    def reconstruct(
        cls, config: WorkloadConfig, wire: Mapping[str, object]
    ) -> "RecirculationBackend":
        artifact_wire = wire[_STAGE_ARTIFACT_BUNDLE_WIRE_KEY]
        if not isinstance(artifact_wire, str):
            raise TypeError("Recirculation artifact bundle must be a string")
        return cls.from_artifacts(
            cast(RecirculationSettings, config), artifact_wire
        )

    @classmethod
    def from_artifacts(
        cls, config: RecirculationSettings, wire: str
    ) -> "RecirculationBackend":
        backend = cls(config)
        backend._state = _ReadyBackend(
            _core.TypedRecurrenceStageArtifactBundle.from_wire(wire)
        )
        return backend

    @property
    def artifacts(self) -> _core.TypedRecurrenceStageArtifactBundle:
        match self._state:
            case _ReadyBackend(artifacts=artifacts):
                return artifacts
            case _ColdBackend():
                raise RuntimeError(
                    "specialization-required: Recirculation has no stage artifacts"
                )

    def _stage(self, context: BackendRegionStageContext) -> None:
        _stage_typed_recirculation(context, self.config, self.artifacts)

    def specialize(self, spec: RuntimeSpecializationSpec) -> None:
        layer_count = spec.model_layer_count
        max_prefill_tokens = spec.max_prefill_tokens
        if layer_count is None or max_prefill_tokens is None:
            raise RuntimeError(
                "Recirculation requires model_layer_count and max_prefill_tokens"
            )
        max_rows = min(max_prefill_tokens, MAX_COLD_STAGE_ARTIFACT_ROWS)
        match self._state:
            case _ColdBackend():
                artifacts = _core.TypedRecurrenceStageArtifactBundle.compile(
                    build_stage_definitions(
                        self.config,
                        max_row_count=max_rows,
                        layer_count=layer_count,
                    )
                )
                self._state = _ReadyBackend(artifacts)
            case _ReadyBackend(artifacts=artifacts):
                pass
        self.config.resolved_layers(layer_count)
        if artifacts.max_row_count != max_rows:
            raise RuntimeError(
                "typed Recirculation signed artifact capacity does not match "
                "the cold specialization"
            )

    def specialization_key(self) -> Mapping[str, str]:
        match self._state:
            case _ColdBackend():
                signature = "cold-specialization-required"
                rows = "cold-specialization-required"
            case _ReadyBackend(artifacts=artifacts):
                signature = artifacts.signature
                rows = str(artifacts.max_row_count)
        return MappingProxyType(
            {"stage_artifact_bundle": signature, "stage_artifact_max_rows": rows}
        )

    def worker_wire(self) -> Mapping[str, object]:
        match self._state:
            case _ColdBackend():
                return MappingProxyType({})
            case _ReadyBackend(artifacts=artifacts):
                return MappingProxyType(
                    {_STAGE_ARTIFACT_BUNDLE_WIRE_KEY: artifacts.to_wire()}
                )

    def runtime_binding(self) -> BackendRuntimeBinding:
        return BackendRuntimeBinding(
            ports=RuntimePortRegistry(inputs={_PLAN_INPUT: self.plan_for_step}),
            binders={PREFILL_BINDER_IMPLEMENTATION_ID: self.binder},
        )

    def plan_for_step(self, step_id: int) -> object:
        return self.binder.plan_for_step(step_id)

    def invoke(self, step_id: int) -> None:
        self.binder.invoke(step_id)


__all__ = [
    "MAX_COLD_STAGE_ARTIFACT_ROWS",
    "PREFILL_BINDER_IMPLEMENTATION_ID",
    "RecirculationBackend",
]
