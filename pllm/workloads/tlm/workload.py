"""TLM sequence selection and loss over ordinary instance buffers and a causal model port."""

import math
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


@dataclass(frozen=True)
class TLMWorkload(pllm.Workload):
    """All-layer q/v LoRA trained from host-supplied causal sequence batches."""

    name = "tlm.qv_lora"
    config: TLMConfig
    rank: int = 8
    alpha: int = 16

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("TLM requires positive rank and alpha")

    @property
    def phase(self) -> pllm.Phase:
        return pllm.Phase.BEFORE_MODEL if self.config.mode is TLMMode.OFFLINE else pllm.Phase.AFTER_SAMPLE

    @property
    def factors(self) -> pllm.InstanceState:
        """Keep each projection's factors as separately shaped AdamW parameters."""
        return pllm.InstanceState(
            {
                "a": pllm.Buffer(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Buffer(lambda inputs, outputs: (outputs, self.rank)),
            },
            optimizer=self.config.optimizer,
            initialize=self.initialize,
            namespace="canonical.qv_adamw@1",
        )

    @property
    def adapter(self) -> pllm.model.Overlay:
        """Resolve all-layer q/v geometry after probing the model."""
        return pllm.model.Overlay(
            "factors",
            tuple(
                pllm.model.ParameterTarget(None, pllm.model.ParameterModuleKind.ATTENTION, role)
                for role in (
                    pllm.model.ParameterRole.QUERY_PROJECTION_WEIGHT,
                    pllm.model.ParameterRole.VALUE_PROJECTION_WEIGHT,
                )
            ),
            operator=pllm.model.ParameterOverlayOperator.LOW_RANK_ADD,
            scale=self.alpha / self.rank,
        )

    def initialize(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        """Seed Kaiming-uniform A and zero B in target order, as in the original TLM."""
        for name, value in fields.items():
            torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator) if name.startswith(
                "a_"
            ) else value.zero_()

    @pllm.on(pllm.Config("phase"))
    @pllm.train(state="factors")
    def update(
        self,
        input_ids: torch.Tensor = pllm.batch("input_ids"),
        labels: torch.Tensor = pllm.batch("labels"),
        attention_mask: torch.Tensor | None = pllm.batch("attention_mask"),
        model: pllm.model.CausalCall = pllm.causal_model("adapter"),
    ) -> pllm.TensorUpdate:
        """Select with the adapted model online, or the no-grad base offline."""
        config = self.config
        selection = None
        if config.mode.value == "offline":
            with torch.no_grad():
                selection = _sentence_cross_entropy(
                    model.base(input_ids, attention_mask=attention_mask), labels, config.ignore_index
                )
        sentence = _sentence_cross_entropy(model(input_ids, attention_mask=attention_mask), labels, config.ignore_index)
        selection = sentence.detach() if selection is None else selection
        selected = selection > config.threshold
        weighted = sentence * (config.lamb * torch.exp(selection - config.threshold)) * selected.to(sentence.dtype)
        count = selected.sum()
        loss = torch.where(count == 0, weighted.mean(), weighted.sum() / count.clamp_min(1))
        changed = (count > 0) & (config.lamb != 0)
        return pllm.TensorUpdate(loss, changed, changed.to(torch.int64), count, {"selected_count": count})
