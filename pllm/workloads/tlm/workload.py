"""TLM selection mathematics on the shared causal low-rank adapter binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import torch
from torch.nn import functional as F

import pllm


def _sentence_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    shifted = labels[..., 1:].contiguous()
    valid = shifted.ne(ignore_index)
    losses = F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
        shifted.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(shifted)
    return (losses * valid).sum(-1) / valid.sum(-1).clamp_min(1)


class TLMMode(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class TLMConfig:
    mode: TLMMode = TLMMode.ONLINE
    optimizer: pllm.AdamW = pllm.AdamW(5e-5, (0.9, 0.999), 1e-8, 0.01)
    threshold: float = 3.0
    lamb: float = 0.1
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if not isfinite(self.threshold) or not isfinite(self.lamb) or self.lamb < 0:
            raise ValueError("TLMConfig needs finite threshold and finite, nonnegative lamb")


@dataclass(frozen=True, slots=True)
class TLMResult:
    loss: torch.Tensor
    selected_count: int
    publication: pllm.state.PagePublication | None


@dataclass(slots=True)
class TLMData:
    batch: pllm.host.CausalTrainBatch[torch.Tensor] | None
    loss: torch.Tensor
    selected_count: torch.Tensor


@dataclass(frozen=True, slots=True)
class TLMWorkload(pllm.host.CausalAdapterWorkload[TLMData, TLMResult]):
    name = "tlm.qv_lora"
    namespace = "canonical.qv_adamw@1"
    config: TLMConfig
    model: pllm.host.CausalModel | pllm.host.CausalAdapter = pllm.host.CausalAdapter()

    def __post_init__(self) -> None:
        if isinstance(self.model, pllm.host.CausalAdapter):
            return
        if any(
            item.target.module is not pllm.model.ParameterModuleKind.ATTENTION
            or item.target.role
            not in (pllm.model.ParameterRole.QUERY_PROJECTION_WEIGHT, pllm.model.ParameterRole.VALUE_PROJECTION_WEIGHT)
            for item in self.model.targets
        ):
            raise ValueError("TLM factor topology must name q/v targets")

    @property
    def optimizer(self) -> pllm.AdamW:
        return self.config.optimizer

    @property
    def phase(self) -> pllm.Phase:
        return pllm.Phase.BEFORE_MODEL if self.config.mode is TLMMode.OFFLINE else pllm.Phase.AFTER_SAMPLE

    def prepare(self, value: pllm.host.CausalTrainBatch[torch.Tensor] | None) -> TLMData:
        assert not isinstance(self.model, pllm.host.CausalAdapter)
        return TLMData(
            value,
            torch.zeros((), device=self.model.device),
            torch.zeros((), device=self.model.device, dtype=torch.int64),
        )

    def trained_rows(self, data: TLMData) -> int:
        return int(data.selected_count.item())

    def output(self, data: TLMData, result: pllm.state.InvocationOutput) -> TLMResult:
        return TLMResult(data.loss, self.trained_rows(data), result.publication)

    def loss(self, data: TLMData, parameters: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
        model, config = self.model, self.config
        assert not isinstance(model, pllm.host.CausalAdapter)
        if data.batch is None:
            return None
        selection = None
        if config.mode.value == "offline":
            with torch.no_grad():
                selection = _sentence_cross_entropy(
                    model.logits(data.batch, None), data.batch.labels, config.ignore_index
                )
        sentence = _sentence_cross_entropy(
            model.logits(data.batch, tuple(parameters.values())), data.batch.labels, config.ignore_index
        )
        selection = sentence.detach() if selection is None else selection
        selected = selection > config.threshold
        weighted = sentence * (config.lamb * torch.exp(selection - config.threshold)) * selected.to(sentence.dtype)
        count = selected.sum()
        loss = torch.where(count == 0, weighted.mean(), weighted.sum() / count.clamp_min(1))
        data.loss, data.selected_count = loss.detach(), count
        return loss if config.lamb != 0 else None
