"""Closed runtime view of one Rust-signed parameter-overlay attachment."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.contracts.parameter_overlay import ParameterTarget
from tllm.contracts.parameter_overlay_attachment import (
    ParameterOverlayAttachmentBinding,
    ParameterOverlayAttachmentPhysicalPlan,
    ParameterOverlayStage,
    ParameterOverlayStageAnchor,
    ParameterOverlayTaskBinding,
    ParameterOverlayTaskRole,
)


@dataclass(frozen=True, slots=True)
class ParameterOverlayRuntimePlan:
    physical: ParameterOverlayAttachmentPhysicalPlan

    @property
    def bindings(self) -> tuple[ParameterOverlayAttachmentBinding, ...]:
        return self.physical.bindings

    @property
    def targets(self) -> tuple[ParameterTarget, ...]:
        return tuple(binding.target for binding in self.physical.bindings)

    def stage(self, role: ParameterOverlayStage) -> ParameterOverlayStageAnchor:
        return self.physical.stage_anchors[tuple(ParameterOverlayStage).index(role)]

    def task(
        self,
        role: ParameterOverlayTaskRole,
    ) -> ParameterOverlayTaskBinding:
        return self.physical.task_bindings[
            tuple(ParameterOverlayTaskRole).index(role)
        ]


__all__ = ["ParameterOverlayRuntimePlan"]
