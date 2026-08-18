#!/usr/bin/env python3
"""Runtime binding for compiler-selected producer materializers.

The Rust core selects and orders producer descriptors.  This module binds
that pure-data plan to Tensor/CUDA implementations once, then exposes the
three residual stages used by the vLLM adapters.  Per-step code never chooses
a producer by port kind or Python type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, cast

import torch

from tllm.contracts.runtime_context import RunnerLike
from tllm.producers.base import HookPoint
from tllm.producers.registry import ExecutionPlan, PlannedProducer
from tllm.runtime.stats import PlacementStats


class ProducerRuntimeLike(Protocol):
    placement_stats: PlacementStats


@dataclass(frozen=True, slots=True)
class PrepareInputsPostContext:
    runner: object
    localize_decode: Callable[[], object]
    phase: str = "decode"
    runtime: ProducerRuntimeLike | None = None


@dataclass(frozen=True, slots=True)
class LayerPostContext:
    runtime: ProducerRuntimeLike | None
    runner: object
    layer_path: str
    hidden: torch.Tensor
    phase: str = "decode"


@dataclass(frozen=True, slots=True)
class ExecuteModelPostContext:
    runner: object
    publish_decode: Callable[[], object]
    phase: str = "decode"
    runtime: ProducerRuntimeLike | None = None


class ProducerMaterializer:
    """Base class for a runtime implementation of one producer descriptor."""

    producer_id: str

    def prepare_inputs_post(self, ctx: PrepareInputsPostContext) -> None:
        del ctx

    def layer_post(self, ctx: LayerPostContext) -> None:
        del ctx

    def execute_model_post(self, ctx: ExecuteModelPostContext) -> None:
        del ctx


class ProducerMaterializerRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, ProducerMaterializer] = {}

    def register(self, materializer: ProducerMaterializer) -> None:
        producer_id = str(materializer.producer_id).strip()
        if not producer_id:
            raise TypeError("producer materializer must declare a non-empty producer_id")
        if producer_id in self._by_id:
            raise RuntimeError(f"duplicate runtime materializer for producer {producer_id!r}")
        self._by_id[producer_id] = materializer

    def get(self, producer_id: str) -> ProducerMaterializer | None:
        return self._by_id.get(str(producer_id))


def _supports_hook(materializer: ProducerMaterializer, hook: HookPoint) -> bool:
    materializer_type = type(materializer)
    if hook == HookPoint.PREPARE_INPUTS_POST:
        return materializer_type.prepare_inputs_post is not ProducerMaterializer.prepare_inputs_post
    if hook == HookPoint.LAYER_POST:
        return materializer_type.layer_post is not ProducerMaterializer.layer_post
    if hook == HookPoint.EXECUTE_MODEL_POST:
        return materializer_type.execute_model_post is not ProducerMaterializer.execute_model_post
    return False


def _producer_accepts_phase(producer: PlannedProducer, phase: str) -> bool:
    declared = {
        declared_phase
        for spec in producer.entry.producer.outputs
        for declared_phase in spec.phases
    }
    return not declared or str(phase) in declared


class ProducerExecutor:
    """Executable, topology-ordered view of a compiled producer plan."""

    def __init__(
        self,
        *,
        plan: ExecutionPlan,
        stages: dict[HookPoint, tuple[tuple[PlannedProducer, ProducerMaterializer], ...]],
    ) -> None:
        self.plan = plan
        self._stages = stages
        self.capture_resident_producer_ids = tuple(plan.capture_resident_producer_ids)

    @classmethod
    def bind(
        cls,
        *,
        plan: ExecutionPlan,
        registry: ProducerMaterializerRegistry,
    ) -> "ProducerExecutor":
        stages: dict[HookPoint, list[tuple[PlannedProducer, ProducerMaterializer]]] = {}
        for producer in plan.producers:
            producer_id = producer.entry.producer_id
            materializer = registry.get(producer_id)
            if materializer is None:
                raise RuntimeError(
                    f"compiled producer {producer_id!r} has no runtime materializer"
                )
            for hook in producer.emits_at:
                if not _supports_hook(materializer, hook):
                    raise RuntimeError(
                        f"runtime materializer {producer_id!r} has no handler for selected hook {hook.value!r}"
                    )
                stages.setdefault(hook, []).append((producer, materializer))
        return cls(
            plan=plan,
            stages={hook: tuple(bound) for hook, bound in stages.items()},
        )

    def _run(
        self,
        hook: HookPoint,
        ctx: PrepareInputsPostContext | LayerPostContext | ExecuteModelPostContext,
    ) -> None:
        phase = str(ctx.phase)
        for producer, materializer in self._stages.get(hook, ()):
            if _producer_accepts_phase(producer, phase):
                if hook == HookPoint.PREPARE_INPUTS_POST:
                    assert isinstance(ctx, PrepareInputsPostContext)
                    materializer.prepare_inputs_post(ctx)
                elif hook == HookPoint.LAYER_POST:
                    assert isinstance(ctx, LayerPostContext)
                    materializer.layer_post(ctx)
                elif hook == HookPoint.EXECUTE_MODEL_POST:
                    assert isinstance(ctx, ExecuteModelPostContext)
                    materializer.execute_model_post(ctx)
                runtime = ctx.runtime
                if runtime is not None:
                    stats = runtime.placement_stats
                    stats.producer_stage_dispatch_count += 1
                    if (
                        hook == HookPoint.LAYER_POST
                        and producer.entry.producer_id
                        in self.capture_resident_producer_ids
                    ):
                        stats.capture_resident_stage_dispatch_count += 1

    def run_prepare_inputs_post(self, ctx: PrepareInputsPostContext) -> None:
        self._run(HookPoint.PREPARE_INPUTS_POST, ctx)

    def run_layer_post(self, ctx: LayerPostContext) -> None:
        self._run(HookPoint.LAYER_POST, ctx)

    def run_execute_model_post(self, ctx: ExecuteModelPostContext) -> None:
        self._run(HookPoint.EXECUTE_MODEL_POST, ctx)


class DecodeResidualMaterializer(ProducerMaterializer):
    producer_id = "tllm.residual.decode"

    def prepare_inputs_post(self, ctx: PrepareInputsPostContext) -> None:
        ctx.localize_decode()

    def layer_post(self, ctx: LayerPostContext) -> None:
        from tllm.runtime.ports.residual_capture_hooks import materialize_decode_residual

        if ctx.runtime is None:
            raise RuntimeError("decode residual materializer requires runtime state")
        materialize_decode_residual(
            runtime=ctx.runtime,
            runner=cast(RunnerLike, ctx.runner),
            layer_path=ctx.layer_path,
            hidden=ctx.hidden,
        )

    def execute_model_post(self, ctx: ExecuteModelPostContext) -> None:
        ctx.publish_decode()


class PrefillResidualMaterializer(ProducerMaterializer):
    producer_id = "tllm.residual.prefill"

    def prepare_inputs_post(self, ctx: PrepareInputsPostContext) -> None:
        ctx.localize_decode()

    def execute_model_post(self, ctx: ExecuteModelPostContext) -> None:
        ctx.publish_decode()


def default_producer_materializers() -> ProducerMaterializerRegistry:
    registry = ProducerMaterializerRegistry()
    registry.register(DecodeResidualMaterializer())
    registry.register(PrefillResidualMaterializer())
    return registry
