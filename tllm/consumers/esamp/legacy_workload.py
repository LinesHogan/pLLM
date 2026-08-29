"""Legacy ESamp declaration and migration dispatcher."""

from __future__ import annotations

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.register import build as build_legacy_consumer
from tllm.ports.consumer import BaseConsumer
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    State,
    StateScope,
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

        if resolved.distiller_sampler_backend != "post_filter_exact":
            raise ValueError(
                "implementation='native' supports only "
                "distiller_sampler_backend='post_filter_exact'"
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
            )
        )
    identifier = str(resolved.consumer_id)
    legacy = build_legacy_consumer(resolved)

    if not resolved.enable_esamp_training:
        @workload(id=identifier, authority=WorkloadAuthority.LEGACY_ADAPTER)
        class DisabledESampWorkload:
            @property
            def legacy_consumer(self) -> BaseConsumer:
                return legacy

        return DisabledESampWorkload()

    per_request = bool(resolved.per_request_models)
    distiller = State(
        scope=StateScope.REQUEST if per_request else StateScope.REPLICA,
        key=request_key("request_id", "sample_idx") if per_request else None,
        parameters="distiller",
        optimizer=AdamWSpec(lr=float(resolved.distiller_lr)),
    )

    @workload(id=identifier, authority=WorkloadAuthority.LEGACY_ADAPTER)
    class ESampWorkloadDeclaration:
        state = distiller

        @task(
            inputs={
                "source": hidden(resolved.source_layer_path, "block_output", "decode"),
                "target": hidden(resolved.target_layer_path, "block_output", "decode"),
            },
            outputs={"prediction": sink("guidance"), "loss": temporary()},
            effects=(read(state),),
            timing=Between("model.execute", "sampler.pre_draw"),
        )
        def predict_and_loss(
            self, source: object, target: object
        ) -> tuple[object, object]:
            raise RuntimeError("ESamp workload is declaration-only during legacy takeover")

        @task(
            inputs={"loss": result("predict_and_loss", "loss")},
            effects=(transition(state),),
            timing=Before("next.prepare"),
        )
        def update(self, loss: object) -> None:
            raise RuntimeError("ESamp workload is declaration-only during legacy takeover")

        @property
        def legacy_consumer(self) -> BaseConsumer:
            return legacy

    return ESampWorkloadDeclaration()


__all__ = ["build_esamp_workload"]
