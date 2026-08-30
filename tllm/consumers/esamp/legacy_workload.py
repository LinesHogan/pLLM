"""Legacy ESamp declaration and migration dispatcher."""

from __future__ import annotations

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.register import build as build_legacy_consumer
from tllm.ports.consumer import BaseConsumer
from tllm.runtime.request_trainable_state import RequestStateSelectionBatch
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    State,
    StateScope,
    RequestKeyField,
    WorkloadAuthority,
    hidden,
    read,
    request_key,
    result,
    sink,
    task,
    temporary,
    transition,
    workload,
)


def build_esamp_workload(config: ESampConsumerConfig | None = None) -> object:
    """Build a config-specialized declaration backed by the current runtime."""

    resolved = config or ESampConsumerConfig()
    if resolved.implementation == "native":
        from tllm.consumers.esamp.workload import (
            ESampConfig,
            ESampWorkload,
        )

        unsupported: list[str] = []
        if resolved.consumer_id != "esamp":
            unsupported.append("consumer_id")
        if resolved.graph_scratch_rows != 0:
            unsupported.append("graph_scratch_rows")
        if not resolved.enable_workgraph_execution:
            unsupported.append("enable_workgraph_execution")
        if resolved.per_request_model_bank:
            unsupported.append("per_request_model_bank")
        if resolved.model_bank_slots != 0:
            unsupported.append("model_bank_slots")
        if resolved.model_bank_flush_interval != 1:
            unsupported.append("model_bank_flush_interval")
        if resolved.model_bank_rank != 64:
            unsupported.append("model_bank_rank")
        if resolved.model_bank_initializer is not None:
            unsupported.append("model_bank_initializer")
        if resolved.model_bank_train_cudagraph:
            unsupported.append("model_bank_train_cudagraph")
        if resolved.model_bank_forward_backend != "torch":
            unsupported.append("model_bank_forward_backend")
        if resolved.adaptation_pipeline_slots != 4:
            unsupported.append("adaptation_pipeline_slots")
        if resolved.adaptation_stream_mode != "dual":
            unsupported.append("adaptation_stream_mode")
        if resolved.adaptation_stream_priority != 0:
            unsupported.append("adaptation_stream_priority")
        if resolved.trace_per_request_losses:
            unsupported.append("trace_per_request_losses")
        if resolved.trace_interval != 1:
            unsupported.append("trace_interval")
        if resolved.trace_max_points != 0:
            unsupported.append("trace_max_points")
        if resolved.compact_capture_lane:
            unsupported.append("compact_capture_lane")
        if resolved.distiller_sampler_backend != "post_filter_exact":
            unsupported.append("distiller_sampler_backend")
        if (
            resolved.per_request_models
            and resolved.enable_distiller_intervention
        ):
            unsupported.append("enable_distiller_intervention")
        if (
            resolved.per_request_models
            and resolved.training_row_selection.value != "none"
        ):
            unsupported.append("training_row_selection")
        if (
            resolved.per_request_models
            and not resolved.model_bank_use_output_layernorm
        ):
            unsupported.append("model_bank_use_output_layernorm")
        if unsupported:
            raise ValueError(
                "implementation='native' does not support legacy-only "
                f"option(s): {', '.join(unsupported)}"
            )
        return ESampWorkload(
            ESampConfig(
                source_layer_path=str(resolved.source_layer_path),
                target_layer_path=str(resolved.target_layer_path),
                enabled=bool(resolved.enable_esamp_training),
                distiller_hidden_dim=int(resolved.distiller_hidden_dim),
                distiller_lr=float(resolved.distiller_lr),
                output_layernorm=bool(resolved.model_bank_use_output_layernorm),
                intervention=bool(resolved.enable_distiller_intervention),
                distiller_beta=float(resolved.distiller_beta),
                sampler_backend="post_filter_exact",
                training_row_selection=resolved.training_row_selection.value,
                per_request_models=bool(resolved.per_request_models),
            )
        )
    identifier = str(resolved.consumer_id)
    legacy = build_legacy_consumer(resolved)

    def legacy_consumer(_instance: object) -> BaseConsumer[object]:
        return legacy

    if not resolved.enable_esamp_training:
        @workload(
            id=identifier,
            authority=WorkloadAuthority.LEGACY_ADAPTER,
            legacy_consumer=legacy_consumer,
        )
        class DisabledESampWorkload:
            pass

        return DisabledESampWorkload()

    per_request = bool(resolved.per_request_models)
    distiller = State(
        scope=StateScope.REQUEST if per_request else StateScope.REPLICA,
        key=(
            request_key(RequestKeyField.PROMPT_IDX)
            if per_request
            else None
        ),
        parameters="distiller",
        optimizer=AdamWSpec.fixed(float(resolved.distiller_lr)),
    )
    if per_request:
        selection = distiller.selection()
        prediction_inputs = {
            "source": hidden(resolved.source_layer_path, "block_output", "decode"),
            "target": hidden(resolved.target_layer_path, "block_output", "decode"),
            "selection": selection,
        }
        prediction_effect = read(selection)
        update_inputs = {
            "loss": result("predict_and_loss", "loss"),
            "selection": selection,
        }
        update_effect = transition(selection)
    else:
        prediction_inputs = {
            "source": hidden(resolved.source_layer_path, "block_output", "decode"),
            "target": hidden(resolved.target_layer_path, "block_output", "decode"),
        }
        prediction_effect = read(distiller)
        update_inputs = {"loss": result("predict_and_loss", "loss")}
        update_effect = transition(distiller)

    @workload(
        id=identifier,
        authority=WorkloadAuthority.LEGACY_ADAPTER,
        legacy_consumer=legacy_consumer,
    )
    class ESampWorkloadDeclaration:
        state = distiller

        @task(
            inputs=prediction_inputs,
            outputs={"prediction": sink("guidance"), "loss": temporary()},
            effects=(prediction_effect,),
            timing=Between("model.execute", "sampler.pre_draw"),
        )
        def predict_and_loss(
            self,
            source: object,
            target: object,
            selection: RequestStateSelectionBatch | None = None,
        ) -> tuple[object, object]:
            _ = selection
            raise RuntimeError("ESamp workload is declaration-only during legacy takeover")

        @task(
            inputs=update_inputs,
            effects=(update_effect,),
            timing=Before("next.prepare"),
        )
        def update(
            self,
            loss: object,
            selection: RequestStateSelectionBatch | None = None,
        ) -> None:
            _ = selection
            raise RuntimeError("ESamp workload is declaration-only during legacy takeover")

    return ESampWorkloadDeclaration()


__all__ = ["build_esamp_workload"]
