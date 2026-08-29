"""Native HF/PEFT TLM workload with immutable q/v LoRA publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch
from torch.nn import functional as F

from tllm.kit.workload import Workload
from tllm.runtime.parameter_overlay import (
    OverlayChannel,
    OverlayLineage,
    OverlayPayload,
    OverlayPublication,
    OverlayRepresentation,
    OverlayVersionRef,
    OverlayVersionStore,
)
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.workload_bindings import LocalRuntimeServices
from tllm.workload import (
    AdamWSpec,
    Between,
    ConfigField,
    ConfigKind,
    HostOutput,
    NumericConstraint,
    ServiceState,
    State,
    StagedPort,
    TrainableState,
    WorkloadConfig,
    config_field,
    extern,
    host_output,
    maybe_transition,
    optimize,
    read,
    result,
    staged,
    task,
    temporary,
    when,
    workload,
)

if TYPE_CHECKING:
    from peft import PeftModelForCausalLM
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast


class _CausalLMFactory(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        *,
        revision: str,
        local_files_only: bool,
        trust_remote_code: bool,
        torch_dtype: torch.dtype,
    ) -> "PreTrainedModel": ...


TLMMode = Literal["online", "offline"]
_TLM_ENABLED = ConfigField.boolean(True)
_TLM_MODE = ConfigField.choice("online", ("online", "offline"))
_TLM_LEARNING_RATE = ConfigField.floating(5e-5, NumericConstraint.POSITIVE)


class TLMConfig(WorkloadConfig):
    enabled: bool = cast(bool, _TLM_ENABLED)
    mode: TLMMode = cast(TLMMode, _TLM_MODE)
    model_id: str = config_field(
        default="Qwen/Qwen2.5-0.5B-Instruct", kind=ConfigKind.STR
    )
    revision: str = config_field(default="main", kind=ConfigKind.STR)
    local_files_only: bool = config_field(default=True, kind=ConfigKind.BOOL)
    trust_remote_code: bool = config_field(default=False, kind=ConfigKind.BOOL)
    rank: int = config_field(
        default=8, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    lora_alpha: int = config_field(
        default=16, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    lora_dropout: float = config_field(
        default=0.0,
        kind=ConfigKind.FLOAT,
        constraint=NumericConstraint.UNIT_INTERVAL,
    )
    learning_rate: float = cast(float, _TLM_LEARNING_RATE)
    threshold: float = config_field(default=3.0, kind=ConfigKind.FLOAT)
    lamb: float = config_field(
        default=0.1,
        kind=ConfigKind.FLOAT,
        constraint=NumericConstraint.NONNEGATIVE,
    )
    ignore_index: int = config_field(default=-100, kind=ConfigKind.INT)


@dataclass(frozen=True, slots=True)
class TLMTrainBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TLMPublication:
    version: OverlayVersionRef
    selected_count: int


def sentence_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
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
    selection = logits if selection_logits is None else selection_logits
    sentence_ce = sentence_cross_entropy(
        selection, labels, ignore_index=ignore_index
    ).detach()
    sentence_kl = sentence_cross_entropy(logits, labels, ignore_index=ignore_index)
    selected = sentence_ce > threshold
    coefficients = lamb * torch.exp(sentence_ce - threshold)
    weighted = sentence_kl * coefficients * selected.to(sentence_kl.dtype)
    selected_count = selected.sum()
    loss = torch.where(
        selected_count == 0,
        weighted.mean(),
        weighted.sum() / selected_count.clamp_min(1),
    )
    return loss, selected


class TLMTrainingModel(torch.nn.Module):
    """Cold typed service around one concrete PEFT causal-LM model."""

    def __init__(self, model: "PeftModelForCausalLM", lora_alpha: int) -> None:
        super().__init__()
        self.model = model
        self.lora_alpha = lora_alpha
        self.overlay_parameters: dict[str, torch.nn.Parameter] = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @classmethod
    def load(
        cls, config: TLMConfig, spec: RuntimeSpecializationSpec
    ) -> "TLMTrainingModel":
        from transformers import AutoModelForCausalLM

        factory = cast(_CausalLMFactory, AutoModelForCausalLM)
        base = factory.from_pretrained(
            config.model_id,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=spec.dtype,
        )
        return cls.adapt(base, config, spec)

    @classmethod
    def adapt(
        cls,
        base: "PreTrainedModel",
        config: TLMConfig,
        spec: RuntimeSpecializationSpec,
    ) -> "TLMTrainingModel":
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = cast("PeftModelForCausalLM", get_peft_model(base, peft_config))
        model.to(device=spec.device, dtype=spec.dtype)
        return cls(model, config.lora_alpha)

    def logits(self, batch: TLMTrainBatch) -> torch.Tensor:
        output = cast(
            "CausalLMOutputWithPast",
            self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
                return_dict=True,
            ),
        )
        logits = output.logits
        if logits is None:
            raise RuntimeError("causal LM output has no logits")
        return logits

    def online_objective(
        self, batch: TLMTrainBatch, config: TLMConfig
    ) -> tuple[torch.Tensor, int]:
        loss, selected = official_tlm_loss(
            self.logits(batch),
            batch.labels,
            selection_logits=None,
            threshold=config.threshold,
            lamb=config.lamb,
            ignore_index=config.ignore_index,
        )
        return loss, int(selected.sum().item())

    def offline_objective(
        self, batch: TLMTrainBatch, config: TLMConfig
    ) -> tuple[torch.Tensor, int]:
        with torch.no_grad(), self.model.disable_adapter():
            selection_logits = self.logits(batch)
        loss, selected = official_tlm_loss(
            self.logits(batch),
            batch.labels,
            selection_logits=selection_logits,
            threshold=config.threshold,
            lamb=config.lamb,
            ignore_index=config.ignore_index,
        )
        return loss, int(selected.sum().item())

    def overlay_payload(self) -> OverlayPayload:
        return OverlayPayload(
            OverlayRepresentation.LOW_RANK_ADD,
            {
                **self.overlay_parameters,
                "__tllm_lora_alpha__": torch.tensor(
                    self.lora_alpha,
                    dtype=torch.float32,
                ),
            },
        )


class TLMOverlayState(torch.nn.Module):
    def __init__(self, store: OverlayVersionStore) -> None:
        super().__init__()
        self.store = store


class TLMWorkload(Workload[TLMConfig]):
    trainer: TrainableState[TLMTrainingModel] = State[TLMTrainingModel].replica(
        optimizer=AdamWSpec.configured(_TLM_LEARNING_RATE)
    )
    overlay_versions: ServiceState[TLMOverlayState] = State[TLMOverlayState].service(
        transition="overlay_publish"
    )
    batches: StagedPort[Mapping[str, torch.Tensor]] = StagedPort()
    publications: HostOutput[TLMPublication] = HostOutput()

    def __init__(self, config: TLMConfig = TLMConfig()) -> None:
        self.selected_count = 0
        self._overlay_store = OverlayVersionStore()
        super().__init__(config)

    @trainer.initialize
    def init_trainer(self, spec: RuntimeSpecializationSpec) -> TLMTrainingModel:
        return TLMTrainingModel.load(self.config, spec)

    @overlay_versions.initialize
    def init_overlay(
        self, _spec: RuntimeSpecializationSpec
    ) -> TLMOverlayState:
        return TLMOverlayState(self._overlay_store)

    def parameter_overlay_bindings(
        self,
    ) -> MappingProxyType[OverlayChannel, OverlayVersionStore]:
        return MappingProxyType(
            {OverlayChannel("tlm", "qv-lora"): self._overlay_store}
        )

    @task(
        inputs={"batch": staged(batches)},
        outputs={
            "loss": optimize(
                trainer, at=Between("token.commit", "step.end")
            ),
            "selected": temporary(),
            "batch_out": temporary(),
        },
        timing=Between("token.commit", "step.end"),
        implementation=extern("tllm.tlm:online_qv_lora_loss@1"),
        when=when(_TLM_MODE, "online"),
    )
    def train_online(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int, TLMTrainBatch]:
        attention = batch["attention_mask"]
        typed_batch = TLMTrainBatch(
            batch["input_ids"],
            batch["labels"],
            None if attention.numel() == 0 else attention,
        )
        loss, selected = self.trainer.online_objective(typed_batch, self.config)
        return loss, selected, typed_batch

    @task(
        inputs={
            "batch": result("train_online", "batch_out"),
            "selected": result("train_online", "selected"),
        },
        outputs={"payload": temporary(), "selected_out": temporary()},
        effects=(read(trainer),),
        timing=Between("token.commit", "step.end"),
        implementation=extern("tllm.tlm:snapshot_qv_lora@1"),
        when=when(_TLM_MODE, "online"),
    )
    def snapshot_online(
        self, batch: TLMTrainBatch, selected: int
    ) -> tuple[OverlayPayload, int]:
        del batch
        return self.trainer.overlay_payload(), selected

    @task(
        inputs={
            "payload": result("snapshot_online", "payload"),
            "selected": result("snapshot_online", "selected_out"),
        },
        outputs={"publication": host_output(publications)},
        effects=(read(trainer), maybe_transition(overlay_versions)),
        timing=Between("token.commit", "step.end"),
        implementation=extern("tllm.parameter_overlay:publish_lora@1"),
        when=when(_TLM_MODE, "online"),
    )
    def publish_online(
        self, payload: OverlayPayload, selected: int
    ) -> TLMPublication:
        return self._publish(payload, selected, "online")

    @task(
        inputs={"batch": staged(batches)},
        outputs={
            "loss": optimize(
                trainer, at=Between("prepare.pre", "prepare.post")
            ),
            "selected": temporary(),
            "batch_out": temporary(),
        },
        timing=Between("prepare.pre", "prepare.post"),
        implementation=extern("tllm.tlm:offline_qv_lora_loss@1"),
        when=when(_TLM_MODE, "offline"),
    )
    def train_offline(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int, TLMTrainBatch]:
        attention = batch["attention_mask"]
        typed_batch = TLMTrainBatch(
            batch["input_ids"],
            batch["labels"],
            None if attention.numel() == 0 else attention,
        )
        loss, selected = self.trainer.offline_objective(typed_batch, self.config)
        return loss, selected, typed_batch

    @task(
        inputs={
            "batch": result("train_offline", "batch_out"),
            "selected": result("train_offline", "selected"),
        },
        outputs={"payload": temporary(), "selected_out": temporary()},
        effects=(read(trainer),),
        timing=Between("prepare.pre", "prepare.post"),
        implementation=extern("tllm.tlm:snapshot_qv_lora@1"),
        when=when(_TLM_MODE, "offline"),
    )
    def snapshot_offline(
        self, batch: TLMTrainBatch, selected: int
    ) -> tuple[OverlayPayload, int]:
        del batch
        return self.trainer.overlay_payload(), selected

    @task(
        inputs={
            "payload": result("snapshot_offline", "payload"),
            "selected": result("snapshot_offline", "selected_out"),
        },
        outputs={"publication": host_output(publications)},
        effects=(read(trainer), maybe_transition(overlay_versions)),
        timing=Between("prepare.pre", "prepare.post"),
        implementation=extern("tllm.parameter_overlay:publish_lora@1"),
        when=when(_TLM_MODE, "offline"),
    )
    def publish_offline(
        self, payload: OverlayPayload, selected: int
    ) -> TLMPublication:
        return self._publish(payload, selected, "offline")

    def _publish(
        self, payload: OverlayPayload, selected: int, mode: TLMMode
    ) -> TLMPublication:
        publication = OverlayPublication(
            OverlayLineage(OverlayChannel("tlm", "qv-lora")),
            epoch=1,
            payload=payload,
        )
        transaction = f"tlm:{mode}:{self.state_version_for(TLMWorkload.trainer)}"
        store = self.overlay_versions.store
        store.stage(transaction, publication)
        version = store.commit(transaction)
        self.advance_state(TLMWorkload.overlay_versions)
        self.selected_count += selected
        return TLMPublication(version.ref, selected)

    def runtime_services(self) -> LocalRuntimeServices:
        return LocalRuntimeServices(
            cold_specialize=self.cold_specialize,
            diagnostics=lambda: MappingProxyType(
                {
                    "selected_count": self.selected_count,
                    **self._accounting.stats(),
                }
            ),
            stats=self._accounting.read,
            set_enabled=self.set_enabled,
            is_enabled=lambda: self.config.enabled,
            synchronize=lambda: None,
        )


workload(TLMWorkload, id="tlm", config=TLMConfig, enabled=_TLM_ENABLED)


__all__ = [
    "TLMConfig",
    "TLMMode",
    "TLMPublication",
    "TLMTrainBatch",
    "TLMTrainingModel",
    "TLMWorkload",
    "official_tlm_loss",
    "sentence_cross_entropy",
]
