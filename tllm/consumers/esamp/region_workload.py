"""Native, replica-scoped ESamp Region workload.

This module owns only the ESamp algorithm and its explicit process-local
resources.  Runtime integration binds the declared ``guidance`` sink and
calls :meth:`begin_step` before materializing the Region; it never needs a
legacy consumer or an on-demand sampler model forward.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.lm_head import resolve_lm_head
from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.contracts.consumer_stats import TrainingStats
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    WorkloadWorkerSpec,
)
from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.ports.residual_stream import ResidualLocator
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

_STATE_NAME = "distiller"


def _specialization_key(config: ESampConsumerConfig) -> Mapping[str, str]:
    """Return the immutable compile-time ESamp configuration identity."""

    return {
        "algorithm": "esamp-exact-region-v1",
        "distiller_hidden_dim": str(config.distiller_hidden_dim),
        "distiller_lr": float(config.distiller_lr).hex(),
        "output_layernorm": str(bool(config.model_bank_use_output_layernorm)),
        "training_enabled": str(bool(config.enable_esamp_training)),
        "intervention_enabled": str(bool(config.enable_distiller_intervention)),
        "distiller_beta": float(config.distiller_beta).hex(),
        "sampler_backend": str(config.distiller_sampler_backend),
        "source_layer_path": str(config.source_layer_path),
        "target_layer_path": str(config.target_layer_path),
    }


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


class ESampRegionResources:
    """Explicit local bindings required by the native ESamp Region.

    Resource ownership is intentionally visible through this single object:
    the runtime can bind its sink, audit the replica version, and obtain the
    sampler provider without reaching into implementation-private storage.
    """

    def __init__(self, workload: "ESampRegionWorkload") -> None:
        self._workload = workload
        self._guidance: ESampRegionGuidance | None = None
        self._loss_count = 0
        self._update_count = 0
        self._loss_sum: torch.Tensor | None = None

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

    def state_snapshot(self) -> Mapping[str, int]:
        return MappingProxyType({"replica": self._workload.state_version})

    def runtime_state_snapshot(self, _step_id: int) -> Mapping[Hashable, int]:
        snapshot: dict[Hashable, int] = {None: self._workload.state_version}
        return MappingProxyType(snapshot)

    def sampler_provider(self) -> "ESampRegionSamplerProvider":
        return self._workload._sampler_provider

    def stats(self) -> Mapping[str, int]:
        guidance = self._guidance
        return MappingProxyType(
            {
                "guidance_rows": 0 if guidance is None else guidance.rows,
                "loss_count": self._loss_count,
                "update_count": self._update_count,
            }
        )

    def read_and_reset_stats(self, sync: bool = True) -> TrainingStats:
        loss_sum = self._loss_sum
        if loss_sum is None or self._loss_count <= 0:
            return TrainingStats(loss_avg=0.0, loss_count=0)
        if sync and loss_sum.device.type == "cuda":
            torch.cuda.current_stream(loss_sum.device).synchronize()
        result = TrainingStats(
            loss_avg=float(loss_sum.item()) / self._loss_count,
            loss_count=self._loss_count,
        )
        self._loss_sum = None
        self._loss_count = 0
        return result

    def _record_update(self, loss: torch.Tensor) -> None:
        detached = loss.detach().float()
        self._loss_sum = (
            detached.clone()
            if self._loss_sum is None
            else self._loss_sum + detached
        )
        self._loss_count += 1
        self._update_count += 1


class ESampRegionSamplerProvider(SamplerModifierProvider):
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
class ESampRegionWorkload(NativeWorkload):
    """Replica-state ESamp algorithm ready for native Region materialization."""

    distiller = _DISTILLER

    def __init__(self, config: ESampConsumerConfig) -> None:
        validate_region_config(config)
        self.config = config
        enabled_snapshot = bool(config.enable_esamp_training)
        self._model: LowRankGatedResidualModel | None = None
        self._optimizer: torch.optim.AdamW | None = None
        self._specialization: RuntimeSpecializationSpec | None = None
        self._state_version = 0
        self._active_step: tuple[int, ESampRegionStepMetadata] | None = None
        self._resources = ESampRegionResources(self)
        self._sampler_provider = ESampRegionSamplerProvider(self)
        self._native_resources = NativeWorkloadResources(
            workload=self,
            output_sinks={"guidance": self._publish_guidance_sink},
            state_snapshots={_STATE_NAME: self._resources.runtime_state_snapshot},
            cold_specializer=self.cold_specialize,
            sampler_provider=self._sampler_provider,
            diagnostics_reader=self._resources.stats,
            stats_reader=self._resources.read_and_reset_stats,
            per_request_stats_reader=lambda _sync: {},
            enabled_setter=self.set_enabled,
            # This predicate belongs to this immutable resource snapshot. A
            # later enable transition creates a new signed Region epoch.
            enabled_predicate=lambda enabled=enabled_snapshot: enabled,
            synchronizer=lambda: None,
            specialization_key=_specialization_key(config),
            residual_paths={
                ResidualLocator(0, "block_output", "decode"): str(
                    config.source_layer_path
                ),
                ResidualLocator(-1, "block_output", "decode"): str(
                    config.target_layer_path
                ),
            },
            worker_spec=WorkloadWorkerSpec(
                factory=(
                    "tllm.consumers.esamp.region_workload:"
                    "reconstruct_esamp_region_workload"
                ),
                config=_config_to_wire(config),
            ),
        )

    @property
    def model(self) -> LowRankGatedResidualModel | None:
        return self._model

    @property
    def optimizer(self) -> torch.optim.AdamW | None:
        return self._optimizer

    @property
    def state_version(self) -> int:
        return self._state_version

    @property
    def resources(self) -> ESampRegionResources:
        return self._resources

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._native_resources

    def set_enabled(self, enabled: bool) -> None:
        self.config.enable_esamp_training = bool(enabled)
        enabled_snapshot = bool(self.config.enable_esamp_training)
        worker_spec = self._native_resources.worker_spec
        if worker_spec is None:
            raise RuntimeError("ESamp Region workload requires a worker spec")
        # Worker specs and resource keys are frozen snapshots.  A changed
        # enable state therefore gets a new snapshot for the next epoch.
        self._native_resources = replace(
            self._native_resources,
            specialization_key=_specialization_key(self.config),
            enabled_predicate=lambda enabled=enabled_snapshot: enabled,
            worker_spec=WorkloadWorkerSpec(
                factory=worker_spec.factory,
                config=_config_to_wire(self.config),
            ),
        )

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
        self._resources.publish_guidance(
            step_id=int(step_id),
            prediction=value,
            metadata=metadata,
        )

    def cold_specialize(self, spec: RuntimeSpecializationSpec) -> None:
        """Allocate trainable model and optimizer outside ambient inference mode."""
        if not isinstance(spec, RuntimeSpecializationSpec):
            raise TypeError("spec must be RuntimeSpecializationSpec")
        previous = self._specialization
        if previous is not None:
            if previous != spec:
                raise RuntimeError(
                    "ESamp Region is already specialized for another static shape, "
                    "dtype, or device"
                )
            return
        rank = max(1, min(int(self.config.distiller_hidden_dim), int(spec.hidden_size)))
        with torch.inference_mode(False):
            with torch.enable_grad():
                model = LowRankGatedResidualModel(
                    hidden_size=int(spec.hidden_size),
                    rank=rank,
                    use_output_layernorm=bool(self.config.model_bank_use_output_layernorm),
                    device=spec.device,
                    dtype=spec.dtype,
                )
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=float(self.config.distiller_lr), foreach=False
                )
        self._model = model
        self._optimizer = optimizer
        self._specialization = spec

    def begin_step(self, *, step_id: int, metadata: ESampRegionStepMetadata) -> None:
        if int(step_id) != metadata.engine_step_id:
            raise ValueError("ESamp Region step_id must match metadata.engine_step_id")
        if self._active_step is not None:
            raise RuntimeError("ESamp Region already has an active step")
        self._active_step = (int(step_id), metadata.isolated_copy())

    def _end_step(self) -> None:
        self._active_step = None

    def _require_model(self) -> tuple[LowRankGatedResidualModel, torch.optim.AdamW]:
        if self._model is None or self._optimizer is None:
            raise RuntimeError("ESamp Region must be cold_specialize()d before execution")
        return self._model, self._optimizer

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
        model, _ = self._require_model()
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
        _, optimizer = self._require_model()
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            raise ValueError("ESamp Region update requires a differentiable loss tensor")
        with torch.inference_mode(False):
            with torch.enable_grad():
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self._state_version += 1
        self._resources._record_update(loss)

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
            self._resources.publish_guidance(
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


def _config_to_wire(config: ESampConsumerConfig) -> Mapping[str, object]:
    from tllm.common.dataclass_json import dataclass_to_jsonable

    raw = dataclass_to_jsonable(config)
    if not isinstance(raw, dict):
        raise TypeError("ESamp Region config did not encode as a JSON object")
    return raw


def reconstruct_esamp_region_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    from tllm.common.dataclass_json import dataclass_from_dict

    config = dataclass_from_dict(ESampConsumerConfig, dict(raw))
    workload_instance = build_esamp_region_workload(config)
    return workload_instance.native_workload_resources()


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
