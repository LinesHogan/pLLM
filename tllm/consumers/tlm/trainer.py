#!/usr/bin/env python3
"""Official-family TLM side training on internal q/v LoRA parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import torch
from torch.nn import functional as F

from tllm.consumers.tlm.config import TLMConsumerConfig
from tllm.runtime.test_time_lora import LoRAAdapterWeights


@runtime_checkable
class _CausalLMOutput(Protocol):
    """The subset of a Hugging Face causal-LM output used by TLM."""

    logits: torch.Tensor


@runtime_checkable
class _LoRAProjection(Protocol):
    """A PEFT LoRA projection whose trainable weight is exportable."""

    weight: torch.Tensor


@runtime_checkable
class _LoRAProjectionModules(Protocol):
    """The string-indexed subset of ``torch.nn.ModuleDict`` used by PEFT."""

    def __contains__(self, key: object) -> bool: ...

    def __getitem__(self, key: str) -> _LoRAProjection: ...


@runtime_checkable
class _PEFTLoRALayer(Protocol):
    """Stable PEFT LoRA fields needed to publish q/v adapter weights."""

    lora_A: _LoRAProjectionModules
    lora_B: _LoRAProjectionModules
    lora_alpha: Mapping[str, float | int]
    active_adapter: str | Sequence[str]


@dataclass(frozen=True, slots=True)
class TLMTrainBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.labels.ndim != 2:
            raise ValueError("TLM input_ids and labels must be [batch, sequence]")
        if self.input_ids.shape != self.labels.shape:
            raise ValueError("TLM input_ids and labels shapes must match")
        if self.attention_mask is not None and self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("TLM attention_mask shape must match input_ids")


@dataclass(frozen=True, slots=True)
class TLMTrainResult:
    loss: float
    selected_count: int
    target_modules: Mapping[str, LoRAAdapterWeights]


def sentence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    """Official TLM sentence CE: mean next-token NLL over valid labels."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("TLM logits must be [batch, sequence, vocab]")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(int(ignore_index))
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=int(ignore_index),
    ).reshape_as(shift_labels)
    return (token_loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)


def official_tlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    selection_logits: torch.Tensor | None,
    threshold: float,
    lamb: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized equivalent of the upstream one-hot KL objective."""

    sentence_ce = sentence_cross_entropy(
        logits if selection_logits is None else selection_logits,
        labels,
        ignore_index=ignore_index,
    ).detach()
    sentence_kl = sentence_cross_entropy(logits, labels, ignore_index=ignore_index)
    selected = sentence_ce > float(threshold)
    coefficients = float(lamb) * torch.exp(sentence_ce - float(threshold))
    weighted = sentence_kl * coefficients * selected.to(sentence_kl.dtype)
    loss = (
        weighted.mean()
        if int(selected.sum().item()) == 0
        else weighted.sum() / selected.sum()
    )
    return loss, selected


def _canonical_peft_module_name(name: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


class HFPEFTTLMTrainer:
    """Train PEFT LoRA on a Hugging Face causal LM; vLLM never backpropagates."""

    def __init__(self, model: torch.nn.Module, config: TLMConsumerConfig) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import PreTrainedModel
        except ImportError as exc:
            raise RuntimeError(
                "TLM q/v LoRA training requires the optional 'tlm' dependency (peft)"
            ) from exc
        if not isinstance(model, PreTrainedModel):
            raise RuntimeError("TLM requires a Hugging Face PreTrainedModel")
        hf_model = cast(PreTrainedModel, model)
        target_modules = tuple(str(value) for value in config.target_modules)
        suffixes = {name.rsplit(".", 1)[-1] for name, _ in hf_model.named_modules()}
        missing = sorted(set(target_modules) - suffixes)
        if missing:
            raise RuntimeError(f"HF model is missing TLM LoRA targets: {missing}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(config.rank),
            lora_alpha=int(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            target_modules=list(target_modules),
            bias="none",
        )
        self.model = get_peft_model(hf_model, lora_config)
        self.config = config
        self.target_modules = target_modules
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("PEFT produced no trainable q/v LoRA parameters")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config.learning_rate),
        )

    def _model_logits(self, batch: TLMTrainBatch) -> torch.Tensor:
        outputs: object = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
        if not isinstance(outputs, _CausalLMOutput):
            raise RuntimeError("HF causal LM did not return logits")
        logits = outputs.logits
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("HF causal LM did not return tensor logits")
        return logits

    def _weights(self) -> dict[str, LoRAAdapterWeights]:
        result: dict[str, LoRAAdapterWeights] = {}
        found_suffixes: set[str] = set()
        for name, module in self.model.named_modules():
            suffix = name.rsplit(".", 1)[-1]
            if suffix not in self.target_modules:
                continue
            if not isinstance(module, _PEFTLoRALayer):
                continue
            active = module.active_adapter
            if isinstance(active, str):
                adapter_name = active
            else:
                if (
                    not isinstance(active, Sequence)
                    or not active
                    or not isinstance(active[0], str)
                ):
                    raise RuntimeError("PEFT did not expose an active TLM adapter")
                adapter_name = active[0]
            if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                raise RuntimeError("PEFT did not expose complete TLM adapter weights")
            a_layer = module.lora_A[adapter_name]
            b_layer = module.lora_B[adapter_name]
            if not isinstance(module.lora_alpha, Mapping):
                raise RuntimeError("PEFT did not expose TLM adapter scales")
            alpha = module.lora_alpha.get(adapter_name)
            if (
                not isinstance(a_layer, _LoRAProjection)
                or not isinstance(b_layer, _LoRAProjection)
                or not isinstance(alpha, (float, int))
            ):
                raise RuntimeError("PEFT did not expose complete TLM adapter weights")
            a_weight = a_layer.weight
            b_weight = b_layer.weight
            if not isinstance(a_weight, torch.Tensor) or not isinstance(b_weight, torch.Tensor):
                raise RuntimeError("PEFT did not expose tensor TLM adapter weights")
            result[_canonical_peft_module_name(name)] = LoRAAdapterWeights(
                a_weight.detach(),
                b_weight.detach(),
                float(alpha),
            )
            found_suffixes.add(suffix)
        missing = sorted(set(self.target_modules) - found_suffixes)
        if missing:
            raise RuntimeError(f"PEFT did not expose trained TLM targets: {missing}")
        return result

    def train_batch(self, batch: TLMTrainBatch, *, mode: str) -> TLMTrainResult:
        if mode not in ("online", "offline"):
            raise ValueError("TLM mode must be online or offline")
        self.model.train()
        selection_logits: torch.Tensor | None = None
        if mode == "offline":
            with torch.no_grad(), self.model.disable_adapter():
                selection_logits = self._model_logits(batch)
        logits = self._model_logits(batch)
        loss, selected = official_tlm_loss(
            logits,
            batch.labels,
            selection_logits=selection_logits,
            threshold=float(self.config.threshold),
            lamb=float(self.config.lamb),
            ignore_index=int(self.config.ignore_index),
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return TLMTrainResult(
            loss=float(loss.detach()),
            selected_count=int(selected.sum().item()),
            target_modules=self._weights(),
        )


__all__ = [
    "HFPEFTTLMTrainer",
    "TLMTrainBatch",
    "TLMTrainResult",
    "official_tlm_loss",
    "sentence_cross_entropy",
]
