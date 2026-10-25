"""Video declarations and Fourier encoding used by the feed example.

Features arrive from a producer; the model consumes a local activation state.
"""

from dataclasses import dataclass
from math import isfinite, pi
from typing import ClassVar

import torch
from torch.nn import functional as F

import pllm

FEATURES = ("qp", "residual_energy", "motion_magnitude", "segmentation")


def injection_shape(inputs: int, outputs: int) -> tuple[int, ...]:
    """Use the model's hidden width for its additive activation field."""
    return (inputs,)


@dataclass(frozen=True)
class VideoWorkload(pllm.Workload):
    """Encode fed feature maps into one local injection on each supplied step."""

    name: ClassVar[str] = "video.session"
    grid_size: int = 8
    layer: int = 0
    gain: float = 0.1
    recurrence: pllm.model.Recurrence | None = None

    def __post_init__(self) -> None:
        if self.grid_size < 1 or self.layer < 0 or not isfinite(self.gain) or self.gain <= 0:
            raise ValueError("VideoWorkload needs a positive grid/gain and a nonnegative layer")

    @property
    def features(self) -> pllm.InstanceState:
        """Declare the four read-only feature fields supplied by the producer."""
        return pllm.InstanceState(
            {name: (self.grid_size, self.grid_size) for name in FEATURES},
            namespace="video.features@1",
            dtype=torch.float32,
            feed=pllm.state.FeedOrder.ORDERED,
        )

    @property
    def inject(self) -> pllm.InstanceState:
        """Declare the local state written by the encoder and read by Activation."""
        return pllm.InstanceState(
            {"delta": pllm.Buffer(injection_shape, dtype=torch.float32)},
            namespace="video.inject@1",
        )

    @property
    def activation(self) -> pllm.model.Activation:
        """Apply the latest settled injection to the selected prefill/decode rows."""
        return pllm.model.Activation(
            "inject",
            (self.layer,),
            first_tokens=1,
            last_tokens=1,
            operator=pllm.model.ActivationPatchOperator.ADD,
            decode=pllm.model.DecodeWindow.ALL,
            decode_refresh=pllm.model.DecodeRefresh.STEP,
            on_missing=pllm.model.MissingPolicy.HOLD,
        )

    @pllm.write(state="inject", field="delta")
    def encode(
        self,
        qp: torch.Tensor = pllm.parameter("features", "qp"),
        residual: torch.Tensor = pllm.parameter("features", "residual_energy"),
        motion: torch.Tensor = pllm.parameter("features", "motion_magnitude"),
        segmentation: torch.Tensor = pllm.parameter("features", "segmentation"),
        template: torch.Tensor = pllm.parameter("inject", "delta"),
    ) -> torch.Tensor:
        """Project block statistics onto a fixed Fourier basis at model hidden width."""
        signal = torch.cat(
            (qp.flatten() / 63, residual.flatten().sqrt(), motion.flatten() / (2**0.5), segmentation.flatten() / 3)
        )
        positions = torch.arange(signal.numel(), dtype=torch.float32, device=signal.device) + 0.5
        frequencies = torch.arange(template.numel(), dtype=torch.float32, device=signal.device) + 1
        basis = torch.cos((pi / signal.numel()) * frequencies[:, None] * positions[None, :])
        return self.gain * torch.tanh((basis @ signal) / signal.numel() ** 0.5)


def frame_features(frame: torch.Tensor, previous: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, ...]:
    """Four FP32 block maps from aligned 8x8-pixel grayscale blocks.

    QP proxy: 63 times normalized local contrast. Residual: minimum prediction
    MSE over a one-pixel motion search. Motion: winning displacement in pixels.
    Segments: texture/activity bits (0..3). Center wins motion-search ties.
    """
    shape = (8 * grid_size, 8 * grid_size)
    if (
        grid_size < 1
        or tuple(frame.shape) != shape
        or previous.shape != frame.shape
        or frame.dtype != torch.uint8
        or previous.dtype != frame.dtype
        or frame.device != previous.device
    ):
        raise ValueError("video features require aligned uint8 grayscale frames at the declared block grid")
    current, prior = frame.float() / 255, previous.float() / 255
    blocks = current.unfold(0, 8, 8).unfold(1, 8, 8)
    variance = blocks.var(dim=(-1, -2), correction=0)
    padded = F.pad(prior[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    shifts = ((0, 0), *((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0)))
    errors = torch.stack(
        tuple(
            (blocks - padded[1 + y : 1 + y + shape[0], 1 + x : 1 + x + shape[1]].unfold(0, 8, 8).unfold(1, 8, 8))
            .square()
            .mean(dim=(-1, -2))
            for y, x in shifts
        )
    )
    energy, choice = errors.min(dim=0)
    magnitudes = current.new_tensor(tuple((x * x + y * y) ** 0.5 for y, x in shifts))
    return (
        torch.round(126 * variance.sqrt()).clamp(0, 63),
        energy,
        magnitudes[choice],
        (variance > 0.02).float() + 2 * (energy > 0.01).float(),
    )
