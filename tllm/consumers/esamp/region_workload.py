"""Native, replica-scoped ESamp Region workload.

This module owns only the ESamp algorithm and its explicit process-local
resources.  Runtime integration binds the declared ``guidance`` sink and
calls :meth:`begin_step` before materializing the Region; it never needs a
legacy consumer or an on-demand sampler model forward.

The process-local lifecycle (specialization identity, resource assembly,
cold-specialization guards, optimizer construction, update finalization,
worker reconstruction) is provided by :class:`tllm.kit.RegionAlgorithm`;
what remains here is the ESamp algorithm itself: row metadata, guidance
publication, the sampler bridge, and the two signed tasks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.lm_head import resolve_lm_head
from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.kit import RegionAlgorithm, RegionResourcesBase
from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.workload_plugins import NativeWorkloadResources
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    ExternImplementation,
    State,
    StateScope,
    hidden,
    read,
    result,
    sink,
    task,
    temporary,
    transition,
    workload,
)


@dataclass(frozen=True, slots=True)
class ESampRegionStepMetadata:
    """Immutable row identity attached to a guidance sink publication."""

    request_ids: tuple[str, ...]
    prompt_idxs: tuple[int, ...]
    sample_idxs: tuple[int, ...]
    phase: str
    engine_step_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_ids", tuple(str(value) for value in self.request_ids))
        object.__setattr__(self, "prompt_idxs", tuple(int(value) for value in self.prompt_idxs))
        object.__setattr__(self, "sample_idxs", tuple(int(value) for value in self.sample_idxs))
        object.__setattr__(self, "phase", str(self.phase))
        object.__setattr__(self, "engine_step_id", int(self.engine_step_id))
        if len({len(self.request_ids), len(self.prompt_idxs), len(self.sample_idxs)}) != 1:
            raise ValueError("ESamp Region metadata row fields must have equal length")

    @property
    def rows(self) -> int:
        return len(self.request_ids)

    def isolated_copy(self) -> "ESampRegionStepMetadata":
        return ESampRegionStepMetadata(
            request_ids=self.request_ids,
            prompt_idxs=self.prompt_idxs,
            sample_idxs=self.sample_idxs,
            phase=self.phase,
            engine_step_id=self.engine_step_id,
        )


@dataclass(frozen=True, slots=True)
class ESampRegionGuidance:
    """One immutable same-step guidance publication."""

    step_id: int
    prediction: torch.Tensor
    metadata: ESampRegionStepMetadata

    @property
    def rows(self) -> int:
        return int(self.prediction.shape[0])


class ESampRegionResources(RegionResourcesBase):
    """Guidance publication storage plus the kit-owned stats bookkeeping."""

    def __init__(self, workload: "ESampRegionWorkload") -> None:
        super().__init__(workload)
        self._algorithm: "ESampRegionWorkload" = workload
        self._guidance: ESampRegionGuidance | None = None

    def publish_guidance(
        self,
        *,
        step_id: int,
        prediction: torch.Tensor,
        metadata: ESampRegionStepMetadata,
    ) -> ESampRegionGuidance:
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 2:
            raise ValueError("ESamp Region guidance must be a rank-2 Tensor")
        if int(step_id) != metadata.engine_step_id:
            raise RuntimeError("ESamp Region guidance step and metadata step differ")
        if int(prediction.shape[0]) != metadata.rows:
            raise RuntimeError("ESamp Region guidance rows and metadata rows differ")
        record = ESampRegionGuidance(
            step_id=int(step_id),
            prediction=prediction.detach().clone(),
            metadata=metadata.isolated_copy(),
        )
        self._guidance = record
        return record

    def output_sink(self) -> ESampRegionGuidance | None:
        """Return the most recent explicit guidance sink publication."""
        return self._guidance

    def guidance_for_step(self, step_id: int) -> ESampRegionGuidance | None:
        record = self._guidance
        return record if record is not None and record.step_id == int(step_id) else None

    def sampler_provider(self) -> "ESampRegionSamplerProvider":
        return cast("ESampRegionSamplerProvider", self._algorithm.sampler_provider)

    def extra_stats(self) -> Mapping[str, int]:
        guidance = self._guidance
        return {"guidance_rows": 0 if guidance is None else guidance.rows}


class ESampRegionSamplerProvider:
    """Sampler bridge which consumes, but never recomputes, Region guidance."""

    def __init__(self, workload: "ESampRegionWorkload") -> None:
        self._workload = workload

    def is_active(self) -> bool:
        config = self._workload.config
        return (
            bool(config.enable_esamp_training)
            and bool(config.enable_distiller_intervention)
            and float(config.distiller_beta) != 0.0
        )

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        if not self.is_active():
            return None
        record = self._workload.resources.guidance_for_step(view.engine_step_id)
        if record is None:
            return None
        rows = int(view.decode_count)
        if rows < 0 or record.rows != rows:
            raise RuntimeError(
                "ESamp Region guidance rows do not match sampler decode rows"
            )
        expected = record.metadata
        observed = ESampRegionStepMetadata(
            request_ids=tuple(view.request_ids[:rows]),
            prompt_idxs=tuple(view.prompt_idxs[:rows]),
            sample_idxs=tuple(view.sample_idxs[:rows]),
            phase=view.phase,
            engine_step_id=view.engine_step_id,
        )
        if observed != expected:
            raise RuntimeError("ESamp Region guidance metadata does not match sampler step")
        if record.prediction.device != view.device:
            raise RuntimeError("ESamp Region guidance device does not match sampler step")
        lm_head = resolve_lm_head(view.model)
        return CandidateModifierState(
            beta=float(self._workload.config.distiller_beta),
            backend=self._workload.config.distiller_sampler_backend,
            affected_row_ids=torch.arange(rows, device=view.device, dtype=torch.long),
            pred_hidden=record.prediction,
            lm_head_weight=lm_head.weight,
            lm_head_bias=lm_head.bias,
        )


def validate_region_config(config: ESampConsumerConfig) -> None:
    """Reject every legacy-only ESamp execution mode before Region creation."""
    if not isinstance(config, ESampConsumerConfig):
        raise TypeError("config must be ESampConsumerConfig")
    prefix = "ESamp implementation='region'"
    if config.implementation != "region":
        raise ValueError(f"{prefix} requires implementation='region'")
    unsupported = (
        (config.per_request_models, "per_request_models=True"),
        (config.per_request_model_bank, "per_request_model_bank=True"),
        (int(config.model_bank_slots) != 0, "model_bank_slots"),
        (config.model_bank_initializer is not None, "model_bank_initializer"),
        (config.model_bank_train_cudagraph, "model_bank_train_cudagraph=True"),
        (config.model_bank_forward_backend != "torch", "model_bank_forward_backend"),
        (config.trace_per_request_losses, "trace_per_request_losses=True"),
        (config.compact_capture_lane, "compact_capture_lane=True"),
        (str(config.training_row_selection) != "none", "training_row_selection"),
    )
    for enabled, label in unsupported:
        if enabled:
            raise ValueError(f"{prefix} does not support {label}")
    if config.distiller_sampler_backend != "post_filter_exact":
        raise ValueError(
            f"{prefix} supports only distiller_sampler_backend='post_filter_exact'"
        )


_DISTILLER = State(
    scope=StateScope.REPLICA,
    parameters="distiller",
    optimizer=AdamWSpec(lr=1e-3),
)


@workload(id="esamp.region")
class ESampRegionWorkload(RegionAlgorithm[ESampConsumerConfig]):
    """Replica-state ESamp algorithm ready for native Region materialization."""

    algorithm = "esamp-exact-region-v1"
    config_type = ESampConsumerConfig
    specialization_fields = {
        "distiller_hidden_dim": "distiller_hidden_dim",
        "distiller_lr": "distiller_lr",
        "output_layernorm": "model_bank_use_output_layernorm",
        "training_enabled": "enable_esamp_training",
        "intervention_enabled": "enable_distiller_intervention",
        "distiller_beta": "distiller_beta",
        "sampler_backend": "distiller_sampler_backend",
        "source_layer_path": "source_layer_path",
        "target_layer_path": "target_layer_path",
    }
    resources_type = ESampRegionResources
    sampler_provider_type = ESampRegionSamplerProvider
    enabled_field = "enable_esamp_training"
    worker_factory = "tllm.consumers.esamp.region_workload:reconstruct_esamp_region_workload"

    distiller = _DISTILLER

    def validate_config(self, config: ESampConsumerConfig) -> None:
        validate_region_config(config)

    def output_sinks(self) -> Mapping[str, object]:
        return {"guidance": self._publish_guidance_sink}

    def residual_paths(self) -> Mapping[object, str]:
        return {
            ResidualLocator(0, "block_output", "decode"): str(self.config.source_layer_path),
            ResidualLocator(-1, "block_output", "decode"): str(self.config.target_layer_path),
        }

    def extra_resource_fields(self) -> Mapping[str, object]:
        return {"per_request_stats_reader": lambda _sync: {}}

    # -- cold specialization (kit-owned guards; this hook is the algorithm) ----

    def init_parameters(
        self, spec: RuntimeSpecializationSpec
    ) -> Mapping[str, torch.nn.Module]:
        rank = max(1, min(int(self.config.distiller_hidden_dim), int(spec.hidden_size)))
        return {
            "distiller": LowRankGatedResidualModel(
                hidden_size=int(spec.hidden_size),
                rank=rank,
                use_output_layernorm=bool(self.config.model_bank_use_output_layernorm),
                device=spec.device,
                dtype=spec.dtype,
            )
        }

    def optimizer_overrides(self) -> Mapping[str, object]:
        # The signed optimizer lr is nominal; the physical lr is pinned per
        # instance by the specialization key.
        return {"lr": float(self.config.distiller_lr)}

    # -- resource exposure ------------------------------------------------------

    @property
    def resources(self) -> ESampRegionResources:
        return cast(ESampRegionResources, self._resources)

    @property
    def model(self) -> LowRankGatedResidualModel | None:
        parameters = self._parameters
        if parameters is None:
            return None
        return cast(LowRankGatedResidualModel, parameters["distiller"])

    def _require_model(self) -> LowRankGatedResidualModel:
        model = self.model
        if model is None:
            raise RuntimeError("ESamp Region must be cold_specialize()d before execution")
        return model

    # -- step bookkeeping --------------------------------------------------------

    def begin_step(self, *, step_id: int, metadata: ESampRegionStepMetadata) -> None:
        if int(step_id) != metadata.engine_step_id:
            raise ValueError("ESamp Region step_id must match metadata.engine_step_id")
        if self._active_step is not None:
            raise RuntimeError("ESamp Region already has an active step")
        self._active_step = (int(step_id), metadata.isolated_copy())

    def _end_step(self) -> None:
        self._active_step = None

    _active_step: tuple[int, ESampRegionStepMetadata] | None = None

    def _publish_guidance_sink(self, value: object, step_id: int) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("ESamp Region guidance sink requires a Tensor")
        from tllm.runtime import residual_runtime

        runtime = residual_runtime.RUNTIME
        rows = int(value.shape[0])
        request_ids = tuple(str(item) for item in runtime.decode_request_ids[:rows])
        prompt_idxs = tuple(int(item) for item in runtime.decode_prompt_idxs[:rows])
        sample_idxs = tuple(int(item) for item in runtime.decode_sample_idxs[:rows])
        metadata = ESampRegionStepMetadata(
            request_ids=request_ids,
            prompt_idxs=prompt_idxs,
            sample_idxs=sample_idxs,
            phase="decode",
            engine_step_id=int(step_id),
        )
        if metadata.rows != rows:
            raise RuntimeError(
                "ESamp Region guidance sink has incomplete decode row metadata"
            )
        self.resources.publish_guidance(
            step_id=int(step_id),
            prediction=value,
            metadata=metadata,
        )

    # -- signed tasks ------------------------------------------------------------

    @task(
        inputs={
            "source": hidden(0, "block_output", "decode"),
            "target": hidden(-1, "block_output", "decode"),
        },
        outputs={"guidance": sink("guidance"), "loss": temporary()},
        effects=(read(_DISTILLER),),
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def predict_and_loss(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self._require_model()
        if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
            raise ValueError("ESamp Region source and target must be matching rank-2 tensors")
        with torch.inference_mode(False):
            with torch.enable_grad():
                source_for_grad = source.clone()
                target_for_loss = target.clone()
                prediction = model(source_for_grad)
                loss = F.mse_loss(prediction.float(), target_for_loss.float())
        return prediction.detach(), loss

    @task(
        inputs={"loss": result("predict_and_loss", "loss")},
        effects=(transition(_DISTILLER),),
        timing=Before("next.prepare"),
        implementation=ExternImplementation(
            namespace="tllm.esamp",
            name="adamw_transition",
            version="1",
            input_types=("Tensor",),
            output_types=("None",),
        ),
    )
    def update(self, loss: torch.Tensor) -> None:
        self.step_update(loss)

    # -- manual drive (standalone parity/testing) --------------------------------

    def run_predict_and_loss(
        self,
        *,
        step_id: int,
        source: torch.Tensor,
        target: torch.Tensor,
        metadata: ESampRegionStepMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.begin_step(step_id=step_id, metadata=metadata)
        try:
            prediction, loss = self.predict_and_loss(source, target)
            self.resources.publish_guidance(
                step_id=step_id,
                prediction=prediction,
                metadata=metadata,
            )
            return prediction, loss
        finally:
            self._end_step()

    def run_update(self, loss: torch.Tensor) -> None:
        self.update(loss)


def build_esamp_region_workload(config: ESampConsumerConfig) -> ESampRegionWorkload:
    return ESampRegionWorkload(config)


def reconstruct_esamp_region_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    return ESampRegionWorkload.reconstruct(raw)


__all__ = [
    "ESampRegionGuidance",
    "ESampRegionResources",
    "ESampRegionSamplerProvider",
    "ESampRegionStepMetadata",
    "ESampRegionWorkload",
    "build_esamp_region_workload",
    "reconstruct_esamp_region_workload",
    "validate_region_config",
]
