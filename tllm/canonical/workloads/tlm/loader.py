"""Cold HF/PEFT q/v LoRA specialization facts for the canonical core."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

import torch

from tllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)
from tllm.contracts.tensor_layout import TensorDType

if TYPE_CHECKING:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


@dataclass(frozen=True, slots=True)
class HfPeftQvLoaderConfig:
    """Closed cold inputs for loading one q/v LoRA model family."""

    model_id: str
    revision: str
    local_files_only: bool
    trust_remote_code: bool
    rank: int
    alpha: int
    dropout: float
    tensor_dtype: TensorDType

    def __post_init__(self) -> None:
        if (
            not self.model_id
            or not self.revision
            or self.rank < 1
            or self.alpha < 1
            or not 0.0 <= self.dropout < 1.0
            or self.tensor_dtype
            not in (
                TensorDType.FLOAT16,
                TensorDType.BFLOAT16,
                TensorDType.FLOAT32,
            )
        ):
            raise ValueError("HF/PEFT q/v loader configuration is invalid")


@dataclass(frozen=True, slots=True, order=True)
class QvLoraTargetFacts:
    """One resolved frozen projection shape for low-rank state layout."""

    target: ParameterTarget
    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        if (
            self.target.module is not ParameterModuleKind.ATTENTION
            or self.target.role
            not in (
                ParameterRole.QUERY_PROJECTION_WEIGHT,
                ParameterRole.VALUE_PROJECTION_WEIGHT,
            )
            or self.input_size < 1
            or self.output_size < 1
        ):
            raise ValueError("q/v LoRA target facts are invalid")


@dataclass(frozen=True, slots=True)
class QvLoraSpecialization:
    """No-hot-state result of a cold HF/PEFT q/v adaptation probe."""

    model_fingerprint: str
    targets: tuple[QvLoraTargetFacts, ...]
    rank: int
    alpha: int
    dropout: float
    tensor_dtype: TensorDType

    def __post_init__(self) -> None:
        if (
            len(self.model_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.model_fingerprint
            )
            or not self.targets
            or tuple(sorted(self.targets)) != self.targets
            or len(self.targets) != len(set(self.targets))
            or self.rank < 1
            or self.alpha < 1
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("q/v LoRA specialization is invalid")


class HfPeftQvLoader:
    """Explicit cold adapter: load, adapt, inspect, then discard hot ownership."""

    @staticmethod
    def load(config: HfPeftQvLoaderConfig) -> QvLoraSpecialization:
        """Load one Qwen2 family model and validate PEFT q/v adaptation cold."""

        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        model = Qwen2ForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=_torch_dtype(config.tensor_dtype),
        )
        return HfPeftQvLoader.inspect(model, config)

    @staticmethod
    def inspect(
        model: "Qwen2ForCausalLM",
        config: HfPeftQvLoaderConfig,
    ) -> QvLoraSpecialization:
        """Freeze facts from an already-loaded exact Qwen2 cold model."""

        from peft import LoraConfig, TaskType, get_peft_model

        targets = _qv_target_facts(model)
        get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            ),
        )
        return QvLoraSpecialization(
            _fingerprint(config, targets),
            targets,
            config.rank,
            config.alpha,
            config.dropout,
            config.tensor_dtype,
        )


def _qv_target_facts(
    model: "Qwen2ForCausalLM",
) -> tuple[QvLoraTargetFacts, ...]:
    """Resolve Qwen2's explicit q/v projection topology without reflection."""

    targets: list[QvLoraTargetFacts] = []
    for layer_index, layer in enumerate(model.model.layers):
        for role, projection in (
            (ParameterRole.QUERY_PROJECTION_WEIGHT, layer.self_attn.q_proj),
            (ParameterRole.VALUE_PROJECTION_WEIGHT, layer.self_attn.v_proj),
        ):
            targets.append(
                QvLoraTargetFacts(
                    ParameterTarget(layer_index, ParameterModuleKind.ATTENTION, role),
                    int(projection.weight.shape[1]),
                    int(projection.weight.shape[0]),
                )
            )
    return tuple(sorted(targets))


def _torch_dtype(dtype: TensorDType) -> torch.dtype:
    match dtype:
        case TensorDType.FLOAT16:
            return torch.float16
        case TensorDType.BFLOAT16:
            return torch.bfloat16
        case TensorDType.FLOAT32:
            return torch.float32
        case _:
            raise ValueError("cold q/v loader requires a floating training dtype")


def _fingerprint(
    config: HfPeftQvLoaderConfig,
    targets: tuple[QvLoraTargetFacts, ...],
) -> str:
    fields = (
        "canonical.hf_peft_qv_loader@1",
        f"model_id={config.model_id}",
        f"revision={config.revision}",
        f"local_files_only={int(config.local_files_only)}",
        f"trust_remote_code={int(config.trust_remote_code)}",
        f"rank={config.rank}",
        f"alpha={config.alpha}",
        f"dropout={config.dropout.hex()}",
        f"tensor_dtype={config.tensor_dtype.value}",
        *(
            "target="
            + str(item.target.layer)
            + ":"
            + item.target.role.value
            + f":{item.input_size}:{item.output_size}"
            for item in targets
        ),
    )
    return sha256("\x00".join(fields).encode("utf-8")).hexdigest()


__all__ = [
    "HfPeftQvLoader",
    "HfPeftQvLoaderConfig",
    "QvLoraSpecialization",
    "QvLoraTargetFacts",
]
