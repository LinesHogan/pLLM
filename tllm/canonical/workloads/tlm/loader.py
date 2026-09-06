"""Cold HF/PEFT topology adapter; all live factors are supplied by the arena."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from tllm.canonical.backend_program import ParameterTargetShape
from tllm.canonical.ptts.page_bundle import tensor_dtype
from tllm.canonical.workloads.tlm.workload import TLMTrainBatch
from tllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from tllm.contracts.tensor_layout import TensorDType


@dataclass(frozen=True, slots=True)
class HfQvConfig:
    model_id: str
    revision: str
    local_files_only: bool
    trust_remote_code: bool
    rank: int
    alpha: int
    dropout: float
    tensor_dtype: TensorDType

    def __post_init__(self) -> None:
        if (not self.model_id or not self.revision or self.rank < 1 or self.alpha < 1
            or self.dropout != 0.0 or self.tensor_dtype not in (
                TensorDType.FLOAT16, TensorDType.BFLOAT16, TensorDType.FLOAT32, TensorDType.FLOAT64)):
            raise ValueError("HF q/v loader requires a frozen floating layout and dropout=0; RNG effects are not yet supported")


class HfQvModel:
    """Frozen base and initialization template, never another mutable factor store."""

    def __init__(self, base: Qwen2ForCausalLM, config: HfQvConfig) -> None:
        from peft import LoraConfig, PeftModelForCausalLM, TaskType, get_peft_model
        self.tensor_dtype = config.tensor_dtype
        self.device = next(base.parameters()).device
        targets: list[ParameterTargetShape] = []
        names: list[str] = []
        for index, layer in enumerate(base.model.layers):
            for role, name, projection in (
                (ParameterRole.QUERY_PROJECTION_WEIGHT, "q_proj", layer.self_attn.q_proj),
                (ParameterRole.VALUE_PROJECTION_WEIGHT, "v_proj", layer.self_attn.v_proj),
            ):
                if projection.weight.dtype != tensor_dtype(config.tensor_dtype):
                    raise ValueError("HF projection dtype differs from the frozen tensor layout")
                targets.append(ParameterTargetShape(ParameterTarget(index, ParameterModuleKind.ATTENTION, role),
                    int(projection.weight.shape[1]), int(projection.weight.shape[0])))
                names.extend(f"base_model.model.model.layers.{index}.self_attn.{name}.lora_{factor}.default.weight"
                    for factor in ("A", "B"))
        adapted = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
            target_modules=["q_proj", "v_proj"], bias="none"))
        if not isinstance(adapted, PeftModelForCausalLM):
            raise TypeError("PEFT did not construct the supported causal model")
        self.model = adapted.to(dtype=tensor_dtype(config.tensor_dtype)).eval().requires_grad_(False)
        self.targets = tuple(targets)
        self._names = tuple(names)
        parameters = dict(self.model.named_parameters())
        self.initial_factors = tuple(parameters[name].detach() for name in names)
        self.fingerprint = sha256(json.dumps((asdict(config), tuple(asdict(target) for target in targets)),
            sort_keys=True).encode()).hexdigest()

    def logits(self, batch: TLMTrainBatch, factors: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        from transformers.modeling_outputs import CausalLMOutputWithPast
        arguments = dict(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
            use_cache=False, return_dict=True)
        if factors is None:
            with self.model.disable_adapter():
                output = self.model.forward(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                    use_cache=False, return_dict=True)
        else:
            # functional_call temporarily binds the original leaf views and restores
            # the frozen template on success or failure. It does not copy factors.
            output = torch.func.functional_call(self.model, dict(zip(self._names, factors, strict=True)),
                (), arguments)
        if not isinstance(output, CausalLMOutputWithPast) or output.logits is None:
            raise TypeError("HF causal model did not produce dense logits")
        return output.logits


    @classmethod
    def load(cls, config: HfQvConfig) -> HfQvModel:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        return cls(Qwen2ForCausalLM.from_pretrained(config.model_id, revision=config.revision,
            local_files_only=config.local_files_only, trust_remote_code=config.trust_remote_code,
            torch_dtype=tensor_dtype(config.tensor_dtype)), config)
