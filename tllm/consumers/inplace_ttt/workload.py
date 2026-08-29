"""Native exact In-Place-TTT fast-weight workload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import cast

import torch

from tllm.contracts.request_identity import RequestIdentity
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
    Before,
    Between,
    ConfigField,
    ConfigKind,
    HostOutput,
    NumericConstraint,
    ServiceState,
    State,
    StagedPort,
    WorkloadConfig,
    config_field,
    extern,
    host_output,
    maybe_transition,
    read,
    result,
    staged,
    task,
    temporary,
    workload,
)


class InPlaceTTTPhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


_INPLACE_ENABLED = ConfigField.boolean(True)


class InPlaceTTTConfig(WorkloadConfig):
    enabled: bool = cast(bool, _INPLACE_ENABLED)
    layers: tuple[int, ...] = config_field(
        default=(0,), kind=ConfigKind.INT_TUPLE
    )
    chunk_size: int = config_field(
        default=4096, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    learning_rate: float = config_field(
        default=1e-3,
        kind=ConfigKind.FLOAT,
        constraint=NumericConstraint.POSITIVE,
    )
    update_prefill: bool = config_field(default=True, kind=ConfigKind.BOOL)
    update_decode: bool = config_field(default=False, kind=ConfigKind.BOOL)


@dataclass(frozen=True, slots=True)
class InPlaceTTTKey:
    request: RequestIdentity
    layer: int


@dataclass(frozen=True, slots=True)
class InPlaceTTTSlot:
    tail_hidden: torch.Tensor
    tail_target: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True, slots=True)
class InPlaceTTTBatch:
    key: InPlaceTTTKey
    phase: InPlaceTTTPhase
    hidden: torch.Tensor
    target: torch.Tensor
    base_weight: torch.Tensor


@dataclass(frozen=True, slots=True)
class InPlaceTTTMetadata:
    key: InPlaceTTTKey
    phase: InPlaceTTTPhase


@dataclass(frozen=True, slots=True)
class InPlaceTTTStep:
    key: InPlaceTTTKey
    output: torch.Tensor
    weight: torch.Tensor
    update_count: int
    trained_rows: int
    tail_rows: int
    state_changed: bool


@dataclass(frozen=True, slots=True)
class InPlaceTTTPublication:
    request: RequestIdentity
    version: OverlayVersionRef | None
    update_count: int


@dataclass(frozen=True, slots=True)
class InPlaceTTTFinishBatch:
    requests: tuple[RequestIdentity, ...]


@dataclass(frozen=True, slots=True)
class InPlaceTTTCleanupReceipt:
    cleared_slots: int
    released_versions: int


class InPlaceTTTFastWeightState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slots: dict[InPlaceTTTKey, InPlaceTTTSlot] = {}

    def execute(
        self,
        batch: InPlaceTTTBatch,
        *,
        chunk_size: int,
        learning_rate: float,
        update_enabled: bool,
    ) -> InPlaceTTTStep:
        slot = self.slots.get(batch.key)
        weight = (
            batch.base_weight.detach().clone()
            if slot is None
            else slot.weight.to(batch.base_weight).clone()
        )
        hidden = batch.hidden
        target = batch.target
        if update_enabled and slot is not None and slot.tail_hidden.shape[0]:
            hidden = torch.cat((slot.tail_hidden.to(hidden), hidden), dim=0)
            target = torch.cat((slot.tail_target.to(target), target), dim=0)
        if not update_enabled:
            output = batch.hidden @ weight.to(batch.hidden).transpose(0, 1)
            tail_rows = 0 if slot is None else slot.tail_hidden.shape[0]
            return InPlaceTTTStep(
                batch.key, output, weight, 0, 0, tail_rows, False
            )

        full_rows = (hidden.shape[0] // chunk_size) * chunk_size
        outputs: list[torch.Tensor] = []
        updates = 0
        for start in range(0, full_rows, chunk_size):
            chunk_hidden = hidden[start : start + chunk_size]
            chunk_target = target[start : start + chunk_size]
            outputs.append(chunk_hidden @ weight.to(chunk_hidden).transpose(0, 1))
            weight = weight + learning_rate * (
                chunk_target.transpose(0, 1) @ chunk_hidden
            ).to(weight)
            updates += 1
        tail_hidden = hidden[full_rows:]
        tail_target = target[full_rows:]
        if tail_hidden.shape[0]:
            outputs.append(tail_hidden @ weight.to(tail_hidden).transpose(0, 1))
        output_all = torch.cat(outputs, dim=0)
        output = output_all[-batch.hidden.shape[0] :]
        self.slots[batch.key] = InPlaceTTTSlot(
            tail_hidden.detach().clone(),
            tail_target.detach().clone(),
            weight.detach().clone(),
        )
        return InPlaceTTTStep(
            key=batch.key,
            output=output,
            weight=weight,
            update_count=updates,
            trained_rows=full_rows,
            tail_rows=tail_hidden.shape[0],
            state_changed=True,
        )

    def clear_requests(self, requests: frozenset[RequestIdentity]) -> int:
        keys = tuple(key for key in self.slots if key.request in requests)
        for key in keys:
            self.slots.pop(key)
        return len(keys)


class InPlaceTTTOverlayState(torch.nn.Module):
    def __init__(self, store: OverlayVersionStore) -> None:
        super().__init__()
        self.store = store


class InPlaceTTTWorkload(Workload[InPlaceTTTConfig]):
    fast_weights: ServiceState[InPlaceTTTFastWeightState] = State[
        InPlaceTTTFastWeightState
    ].service(transition="fast_weight_update")
    overlay_versions: ServiceState[InPlaceTTTOverlayState] = State[
        InPlaceTTTOverlayState
    ].service(transition="overlay_publish")
    batch_tensors: StagedPort[Mapping[str, torch.Tensor]] = StagedPort()
    batch_metadata: StagedPort[InPlaceTTTMetadata] = StagedPort()
    finished_requests: StagedPort[InPlaceTTTFinishBatch] = StagedPort()
    steps: HostOutput[InPlaceTTTStep] = HostOutput()
    publications: HostOutput[InPlaceTTTPublication] = HostOutput()
    cleanup: HostOutput[InPlaceTTTCleanupReceipt] = HostOutput()

    def __init__(self, config: InPlaceTTTConfig = InPlaceTTTConfig()) -> None:
        self.update_count = 0
        self.trained_rows = 0
        self._overlay_store = OverlayVersionStore()
        super().__init__(config)

    @fast_weights.initialize
    def init_fast_weights(
        self, _spec: RuntimeSpecializationSpec
    ) -> InPlaceTTTFastWeightState:
        return InPlaceTTTFastWeightState()

    @overlay_versions.initialize
    def init_overlay_versions(
        self, _spec: RuntimeSpecializationSpec
    ) -> InPlaceTTTOverlayState:
        return InPlaceTTTOverlayState(self._overlay_store)

    def parameter_overlay_bindings(
        self,
    ) -> MappingProxyType[OverlayChannel, OverlayVersionStore]:
        return MappingProxyType(
            {
                OverlayChannel("inplace_ttt", f"layer-{layer}-down-proj"): self._overlay_store
                for layer in self.config.layers
            }
        )

    @task(
        inputs={
            "tensors": staged(batch_tensors),
            "metadata": staged(batch_metadata),
        },
        outputs={"step": host_output(steps)},
        effects=(maybe_transition(fast_weights),),
        timing=Between("model.execute", "step.end"),
        implementation=extern("tllm.inplace_ttt:exact_fast_weight@1"),
    )
    def update(
        self,
        tensors: Mapping[str, torch.Tensor],
        metadata: InPlaceTTTMetadata,
    ) -> InPlaceTTTStep:
        batch = InPlaceTTTBatch(
            metadata.key,
            metadata.phase,
            tensors["hidden"],
            tensors["target"],
            tensors["base_weight"],
        )
        update_enabled = (
            self.config.update_prefill
            if batch.phase is InPlaceTTTPhase.PREFILL
            else self.config.update_decode
        )
        step = self.fast_weights.execute(
            batch,
            chunk_size=self.config.chunk_size,
            learning_rate=self.config.learning_rate,
            update_enabled=update_enabled,
        )
        if step.state_changed:
            self.advance_state(InPlaceTTTWorkload.fast_weights)
        self.update_count += step.update_count
        self.trained_rows += step.trained_rows
        return step

    @task(
        inputs={"step": result("update", "step")},
        outputs={"publication": host_output(publications)},
        effects=(read(fast_weights), maybe_transition(overlay_versions)),
        timing=Before("next.prepare"),
        implementation=extern("tllm.parameter_overlay:publish_exact@1"),
    )
    def publish(self, step: InPlaceTTTStep) -> InPlaceTTTPublication:
        if not step.update_count:
            return InPlaceTTTPublication(step.key.request, None, 0)
        lineage = OverlayLineage(
            OverlayChannel("inplace_ttt", f"layer-{step.key.layer}-down-proj"),
            step.key.request,
        )
        publication = OverlayPublication(
            lineage,
            epoch=1,
            payload=OverlayPayload(
                OverlayRepresentation.DENSE_REPLACE, {"weight": step.weight}
            ),
        )
        store = self.overlay_versions.store
        transaction = (
            f"{step.key.request.request_id}:{step.key.request.sample_idx}:"
            f"{step.key.layer}:{self.update_count}"
        )
        store.stage(transaction, publication)
        version = store.commit(transaction)
        store.advance_request(step.key.request, version.ref)
        self.advance_state(InPlaceTTTWorkload.overlay_versions)
        return InPlaceTTTPublication(
            step.key.request, version.ref, step.update_count
        )

    @task(
        inputs={"finished": staged(finished_requests)},
        outputs={"requests": temporary()},
        timing=Before("next.prepare"),
        implementation=extern("tllm.parameter_overlay:release_requests@1"),
    )
    def finish(self, finished: InPlaceTTTFinishBatch) -> tuple[RequestIdentity, ...]:
        return finished.requests

    @task(
        inputs={"requests": result("finish", "requests")},
        outputs={"cleared": temporary()},
        effects=(maybe_transition(fast_weights),),
        timing=Before("next.prepare"),
        implementation=extern("tllm.inplace_ttt:clear_fast_weights@1"),
    )
    def finish_fast_weights(self, requests: tuple[RequestIdentity, ...]) -> int:
        request_set = frozenset(requests)
        cleared = self.fast_weights.clear_requests(request_set)
        if cleared:
            self.advance_state(InPlaceTTTWorkload.fast_weights)
        return cleared

    @task(
        inputs={
            "requests": result("finish", "requests"),
            "cleared": result("finish_fast_weights", "cleared"),
        },
        outputs={"receipt": host_output(cleanup)},
        effects=(maybe_transition(overlay_versions),),
        timing=Before("next.prepare"),
        implementation=extern("tllm.parameter_overlay:release_leases@1"),
    )
    def finish_overlay(
        self, requests: tuple[RequestIdentity, ...], cleared: int
    ) -> InPlaceTTTCleanupReceipt:
        released = sum(
            self.overlay_versions.store.release(request) for request in requests
        )
        if released:
            self.advance_state(InPlaceTTTWorkload.overlay_versions)
        return InPlaceTTTCleanupReceipt(cleared, released)

    def runtime_services(self) -> LocalRuntimeServices:
        return LocalRuntimeServices(
            cold_specialize=self.cold_specialize,
            diagnostics=lambda: MappingProxyType(
                {
                    "update_count": self.update_count,
                    "trained_rows": self.trained_rows,
                }
            ),
            stats=self._accounting.read,
            set_enabled=self.set_enabled,
            is_enabled=lambda: self.config.enabled,
            synchronize=lambda: None,
        )


workload(
    InPlaceTTTWorkload,
    id="inplace_ttt",
    config=InPlaceTTTConfig,
    enabled=_INPLACE_ENABLED,
)


__all__ = [
    "InPlaceTTTBatch",
    "InPlaceTTTCleanupReceipt",
    "InPlaceTTTConfig",
    "InPlaceTTTFastWeightState",
    "InPlaceTTTFinishBatch",
    "InPlaceTTTKey",
    "InPlaceTTTMetadata",
    "InPlaceTTTPhase",
    "InPlaceTTTPublication",
    "InPlaceTTTSlot",
    "InPlaceTTTStep",
    "InPlaceTTTWorkload",
]
