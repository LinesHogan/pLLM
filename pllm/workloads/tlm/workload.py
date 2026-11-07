"""TLM sequence selection and loss over ordinary instance buffers and a causal model port."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import torch

import pllm


def _sentence_cross_entropy(
    model: pllm.model.CausalModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    ignore_index: int,
) -> torch.Tensor:
    logprobs = model.token_logprobs(input_ids, labels, attention_mask=attention_mask, ignore_index=ignore_index)
    return -logprobs.sum(-1) / labels[:, 1:].ne(ignore_index).sum(-1).clamp_min(1)


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

    config: TLMConfig
    rank: int = 8
    alpha: int = 16

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("TLM requires positive rank and alpha")

    @property
    def phase(self) -> pllm.Phase:
        return pllm.Phase.BEFORE_FORWARD if self.config.mode is TLMMode.OFFLINE else pllm.Phase.AFTER_SAMPLE

    @property
    def factors(self) -> pllm.InstanceState:
        """Declare projection-shaped views of the packed AdamW state."""
        return pllm.InstanceState(
            {
                "a": pllm.Field(lambda in_features, out_features: (self.rank, in_features)),
                "b": pllm.Field(lambda in_features, out_features: (out_features, self.rank)),
            },
            optimizer=self.config.optimizer,
            initialize=self.initialize,
        )

    @property
    def adapter(self) -> pllm.model.Adapter:
        """Resolve all-layer q/v geometry after probing the model."""
        return pllm.model.Adapter(
            "factors",
            (pllm.model.Projection("q_proj"), pllm.model.Projection("v_proj")),
            operator="low_rank",
            scale=self.alpha / self.rank,
        )

    def initialize(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        """Seed Kaiming-uniform A and zero B in target order, as in the original TLM."""
        for name, value in fields.items():
            if name.startswith("a_"):
                torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
            else:
                value.zero_()

    @pllm.on(pllm.Attr("phase"))
    @pllm.train(state="factors")
    def update(
        self,
        input_ids: torch.Tensor = pllm.batch("input_ids"),
        labels: torch.Tensor = pllm.batch("labels"),
        attention_mask: torch.Tensor | None = pllm.batch("attention_mask"),
        model: pllm.model.CausalModel = pllm.causal_model("adapter"),
    ) -> pllm.UpdateOutput | None:
        """Select online with adapted logits, offline with base logits.

        Empty selections publish unchanged fields; selected_count remains a device tensor.
        A disabled loss skips the forward and produces no selection metric.
        """
        config = self.config
        if config.lamb == 0:
            return None
        selection = None
        if config.mode is TLMMode.OFFLINE:
            with torch.no_grad():
                selection = _sentence_cross_entropy(model.base, input_ids, labels, attention_mask, config.ignore_index)
        sentence = _sentence_cross_entropy(model, input_ids, labels, attention_mask, config.ignore_index)
        selection = sentence.detach() if selection is None else selection
        selected = selection > config.threshold
        weighted = sentence * (config.lamb * torch.exp(selection - config.threshold)) * selected.to(sentence.dtype)
        count = selected.sum()
        loss = torch.where(count == 0, weighted.mean(), weighted.sum() / count.clamp_min(1))
        changed = count > 0
        return pllm.UpdateOutput(loss, changed, changed.to(torch.int64), count, {"selected_count": count})
