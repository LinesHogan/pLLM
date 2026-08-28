"""Explicit per-step values exposed to signed sampler input ports."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.runtime.sampler_bridge.types import SamplerStepView


@dataclass(frozen=True, slots=True)
class _Waiting:
    pass


@dataclass(frozen=True, slots=True)
class _BeforeDraw:
    view: SamplerStepView


@dataclass(frozen=True, slots=True)
class _AfterDraw:
    view: SamplerStepView
    tokens: torch.Tensor


class SamplerStepContext:
    def __init__(self) -> None:
        self._state: _Waiting | _BeforeDraw | _AfterDraw = _Waiting()

    def prepare(self, view: SamplerStepView) -> None:
        self._state = _BeforeDraw(view)

    def sampled(self, view: SamplerStepView, tokens: torch.Tensor) -> None:
        self._state = _AfterDraw(view, tokens.reshape(-1).to(torch.long))

    def logits(self, _step_id: int) -> torch.Tensor:
        match self._state:
            case _BeforeDraw(view=view) | _AfterDraw(view=view):
                return view.logits
            case _Waiting():
                raise RuntimeError("sampler logits are not prepared")

    def tokens(self, _step_id: int) -> torch.Tensor:
        match self._state:
            case _AfterDraw(tokens=tokens):
                return tokens
            case _:
                raise RuntimeError("sampled tokens are not committed")

    def request_ids(self, _step_id: int) -> tuple[str, ...]:
        from tllm.runtime import residual_runtime

        runtime = residual_runtime.RUNTIME
        return tuple(runtime.decode_request_ids[: runtime.decode_count])


__all__ = ["SamplerStepContext"]
