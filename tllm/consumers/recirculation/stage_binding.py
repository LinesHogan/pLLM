"""Consumer-owned binding from signed stage coordinates to replay values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import tllm._core as _core
import torch

from tllm.contracts.typed_replay import (
    VLLM_FINAL_NORM_PRIMITIVE,
    VLLM_LAYER_REPLAY_PRIMITIVE,
    FinalNormCall,
    LayerReplayCall,
    LayerReplayState,
    TypedLayerReplayHandle,
)
from tllm.consumers.recirculation.mix import mix_recirculation_tensors
from tllm.consumers.recirculation.spec import RecirculationSettings
from tllm.consumers.recirculation.stage_definition import (
    FINAL_NORM_BINDING_KIND,
    LAYER_REPLAY_BINDING_KIND,
    MIX_BINDING_KIND,
    NORMALIZED_OUTPUT_BINDING_KIND,
    PREFILL_BINDER_IMPLEMENTATION_ID,
    RECIRCULATION_MIX_PRIMITIVE,
    SEED_STATE_BINDING_KIND,
)
from tllm.runtime.typed_recurrence import (
    PrimitiveAdapter,
    PrimitiveAdapterRegistry,
    TypedRecurrenceExecutor,
)


@dataclass(frozen=True, slots=True)
class _RecirculationMixCall:
    config: RecirculationSettings
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
                RECIRCULATION_MIX_PRIMITIVE,
                mix,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class RecirculationBindingProvider:
    handle: TypedLayerReplayHandle
    config: RecirculationSettings
    request_ordinal: int
    packed_rows: tuple[int, ...]
    token_positions: tuple[int, ...]

    def execute(
        self, stage: _core.TypedBoundRecurrenceStage
    ) -> Mapping[int, torch.Tensor]:
        values = TypedRecurrenceExecutor(_primitive_registry()).execute(
            stage.plan,
            seed_values=self._seed_values(stage.seed_bindings),
            backend_resources=self._resource_values(stage.resource_bindings),
        )
        return cast(
            Mapping[int, torch.Tensor],
            self._output_values(stage.output_bindings, values),
        )

    def _seed_values(
        self, bindings: list[_core.TypedStageBinding]
    ) -> Mapping[str, object]:
        values: dict[str, object] = {}
        for binding in bindings:
            if binding.kind != SEED_STATE_BINDING_KIND:
                raise RuntimeError(
                    f"unknown Recirculation seed kind {binding.kind!r}"
                )
            token = binding.coordinates[0]
            values[binding.binding_id] = self.handle.seed_state(
                self.packed_rows[token]
            )
        return values

    def _resource_values(
        self, bindings: list[_core.TypedStageResourceBinding]
    ) -> Mapping[str, object]:
        values: dict[str, object] = {}
        for binding in bindings:
            coordinates = binding.coordinates
            if binding.kind == LAYER_REPLAY_BINDING_KIND:
                token, layer, _pass_ordinal = coordinates
                value: object = LayerReplayCall(
                    handle=self.handle,
                    layer=layer,
                    request_ordinal=self.request_ordinal,
                    packed_row=self.packed_rows[token],
                    token_position=self.token_positions[token],
                )
            elif binding.kind == MIX_BINDING_KIND:
                token = coordinates[0]
                value = _RecirculationMixCall(
                    config=self.config,
                    token_position=self.token_positions[token],
                )
            elif binding.kind == FINAL_NORM_BINDING_KIND:
                value = FinalNormCall(self.handle)
            else:
                raise RuntimeError(
                    f"unknown Recirculation resource kind {binding.kind!r}"
                )
            values[binding.resource_id] = value
        return values

    @staticmethod
    def _output_values(
        bindings: list[_core.TypedStageOutputBinding],
        values: Mapping[str, object],
    ) -> Mapping[int, object]:
        results: dict[int, object] = {}
        for binding in bindings:
            if binding.kind != NORMALIZED_OUTPUT_BINDING_KIND:
                raise RuntimeError(
                    f"unknown Recirculation output kind {binding.kind!r}"
                )
            results[binding.coordinates[0]] = values[binding.value_id]
        return results


__all__ = ["RecirculationBindingProvider"]
