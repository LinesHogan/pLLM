"""Video feature consumption through ordinary Parameter ports and Activation ADD.

The producer's block statistics approximate codec-side signals from real frames;
they are not parsed AV1 QP/segmentation or NVDEC motion vectors. The small fixed
Fourier encoder demonstrates conditioning, not a trained vision-language model.
"""

from dataclasses import dataclass, replace
from math import isfinite, pi
from typing import Annotated, ClassVar

import torch
from torch.nn import functional as F
import pllm
from pllm.authoring import BackendCompileFacts, BackendRequirements, BackendWorkload, MathBinding, PlanBundle, TensorStepInput
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import ActivationPatchBinding, BackendAttachmentSpec
from pllm.contracts.semantic_graph import SchedulerTimeline, SemanticGraph, StateEffectKind, TaskResourceFact, TaskResultInputSource, TaskSymbol, ValueResourceFact, WorkloadId
from pllm.contracts.tensor_layout import LayoutFieldId, TensorLayoutRequest
from pllm.execution.codec import boundary_digest
from pllm.execution.declaration import Field, When, native
from pllm.execution.tensor_binding import TensorBinding, TensorFrame, TensorOutput
from pllm.execution.tensor_program import TaskOutput, TensorBatch


FEATURES = ("qp", "residual_energy", "motion_magnitude", "segmentation")
_ENCODE = TaskSymbol("encode")
_DELTA = LayoutFieldId(0)
QP = Annotated[torch.Tensor, pllm.Parameter("features", "qp")]
Residual = Annotated[torch.Tensor, pllm.Parameter("features", "residual_energy")]
Motion = Annotated[torch.Tensor, pllm.Parameter("features", "motion_magnitude")]
Segments = Annotated[torch.Tensor, pllm.Parameter("features", "segmentation")]


def frame_features(frame: torch.Tensor, previous: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, ...]:
    """Four FP32 block maps from aligned 8x8-pixel grayscale blocks.

    QP proxy: 63 times normalized local contrast. Residual: minimum prediction
    MSE over a one-pixel motion search. Motion: winning displacement in pixels.
    Segments: texture/activity bits (0..3). Center wins motion-search ties.
    """
    shape = (8 * grid_size, 8 * grid_size)
    if (grid_size < 1 or tuple(frame.shape) != shape or previous.shape != frame.shape
        or frame.dtype != torch.uint8 or previous.dtype != frame.dtype or frame.device != previous.device):
        raise ValueError("video features require aligned uint8 grayscale frames at the declared block grid")
    current, prior = frame.float() / 255, previous.float() / 255
    blocks = current.unfold(0, 8, 8).unfold(1, 8, 8)
    variance = blocks.var(dim=(-1, -2), correction=0)
    padded = F.pad(prior[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    shifts = ((0, 0), *((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0)))
    errors = torch.stack(tuple((blocks - padded[1 + y:1 + y + shape[0], 1 + x:1 + x + shape[1]]
        .unfold(0, 8, 8).unfold(1, 8, 8)).square().mean(dim=(-1, -2)) for y, x in shifts))
    energy, choice = errors.min(dim=0)
    magnitudes = current.new_tensor(tuple((x * x + y * y) ** 0.5 for y, x in shifts))
    return (torch.round(126 * variance.sqrt()).clamp(0, 63), energy, magnitudes[choice],
        (variance > 0.02).float() + 2 * (energy > 0.01).float())


@dataclass(frozen=True)
class _VideoEncoder(pllm.Workload):
    name = "video.encoder"
    grid_size: int
    hidden_size: int
    gain: float

    @property
    def features(self) -> pllm.InstanceState:
        return pllm.InstanceState({name: (self.grid_size, self.grid_size) for name in FEATURES},
            namespace="video.features@1", dtype=torch.float32)

    @pllm.on(pllm.Phase.BEFORE_MODEL)
    def encode(self, qp: QP, residual: Residual, motion: Motion, segmentation: Segments) -> torch.Tensor:
        signal = torch.cat((qp.flatten() / 63, residual.flatten().sqrt(),
            motion.flatten() / (2 ** 0.5), segmentation.flatten() / 3))
        positions = torch.arange(signal.numel(), dtype=torch.float32, device=signal.device) + 0.5
        frequencies = torch.arange(self.hidden_size, dtype=torch.float32, device=signal.device) + 1
        basis = torch.cos((pi / signal.numel()) * frequencies[:, None] * positions[None, :])
        return self.gain * torch.tanh((basis @ signal) / signal.numel() ** 0.5)


def _write_inject(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
    hidden = batch.results[_ENCODE, "value"]
    if not torch.isfinite(hidden).all():
        raise ValueError("video encoder produced a non-finite injection")
    return TaskOutput(field_writes=((_DELTA, hidden.unsqueeze(0)),), changed=(0,), update_count=1)


@dataclass(frozen=True)
class VideoWorkload(BackendWorkload[TensorFrame, TensorOutput]):
    """One borrowed feature root and one entirely local per-step injection root."""

    name: ClassVar[str] = "video.session"
    grid_size: int = 8
    layer: int = 0
    gain: float = 0.1
    page_capacity: int = 3
    recurrence: pllm.Recurrence | None = None

    def __post_init__(self) -> None:
        if self.grid_size < 1 or self.layer < 0 or self.page_capacity < 2 or not isfinite(self.gain) or self.gain <= 0:
            raise ValueError("video session requires a positive grid/gain, a layer and current/private page capacity")

    @property
    def activation(self) -> pllm.Activation:
        return pllm.Activation("inject", (self.layer,), first_tokens=1, last_tokens=1,
            operator=pllm.ActivationPatchOperator.ADD, decode=pllm.DecodeWindow.ALL,
            decode_refresh=pllm.DecodeRefresh.STEP, on_missing=pllm.MissingPolicy.HOLD)

    def backend_requirements(self) -> BackendRequirements:
        return BackendRequirements(activation_layers=self.activation.layers)

    def specialize(self, facts: BackendCompileFacts, options: ExecutionOptions) -> BackendCompileFacts:
        return super().specialize(replace(facts, providers=tuple(replace(provider,
            page_capacity=self.page_capacity, shared_memory=True) for provider in facts.providers)), options)

    def resource_facts(self, facts: BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline:
        timeline = super().resource_facts(facts, options)
        width, signals = facts.probe.hidden_size, 4 * self.grid_size ** 2
        # Producer scratch may overlap any worker anchor; it is not worker task mathematics.
        producer = 4 * (24 * (8 * self.grid_size) ** 2 + 16 * self.grid_size ** 2)
        return replace(timeline, reserved_bytes=tuple(value + producer for value in (
            timeline.reserved_bytes or (0,) * len(timeline.anchors))),
            task_resources=(TaskResourceFact(_ENCODE, 4 * (3 * width * signals + 8 * width + 16 * signals)),
                TaskResourceFact(TaskSymbol("publish"), 0), TaskResourceFact(TaskSymbol("write_inject"), 16 * width)),
            value_resources=(ValueResourceFact(_ENCODE, "value", 4 * width),
                ValueResourceFact(TaskSymbol("publish"), "publication", 0),
                ValueResourceFact(TaskSymbol("write_inject"), "updated", 0)))

    def _binding(self, facts: BackendCompileFacts) -> tuple[_VideoEncoder, TensorBinding]:
        encoder = _VideoEncoder(self.grid_size, facts.probe.hidden_size, self.gain)
        b = encoder.declaration(facts)
        inject = b.state("inject", (Field("delta", (facts.probe.hidden_size,)),), namespace="video.inject@1")
        activation = self.activation
        update = b.compute("write_inject", inputs={"hidden": TaskResultInputSource(_ENCODE, "value")},
            outputs=("updated",), changes=inject, effect=StateEffectKind.TRANSITION, when=When.BEFORE_MODEL,
            function=_write_inject, implementations=(native("video.inject:write@1", "transition/full_root@1"),))
        publication = b.publish(update, inject, when=When.BEFORE_MODEL)
        b.overlay(inject, update, publication, activations=(ActivationPatchBinding(self.layer, (_DELTA,),
            activation.first_tokens, activation.last_tokens, activation.operator,
            activation.decode, activation.decode_refresh, activation.on_missing),))
        if self.recurrence is not None:
            b.recurrence(self.recurrence)
        b.graph = replace(b.graph, workload_id=WorkloadId(self.name), execution_parameters_digest=boundary_digest(b.graph.execution_parameters_digest,
            self, activation, ("features",), "video.inject:write@1"))
        return encoder, b

    def declare(self, facts: BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        return self._binding(facts)[1].finish()

    def bind_math(self, facts: BackendCompileFacts, plan: PlanBundle) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        encoder, b = self._binding(facts)
        if b.graph != plan.semantic_graph or b.layouts != plan.tensor_layout_request:
            raise ValueError("video binding differs from the original signed declaration")
        self.require_resources(plan, self.minimum_resources(facts, plan))
        return replace(encoder.operations(facts, plan, b), borrowed_resources=("features",))
