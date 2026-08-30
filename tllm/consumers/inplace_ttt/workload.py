"""Native exact In-Place-TTT fast-weight workload."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
import torch

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
)
from tllm.kit.workload import Workload
from tllm.runtime.parameter_overlay import (
    DenseOperandCaptureBatch,
    OverlayChannel,
    OverlayLeaseKey,
    OverlayLineage,
    OverlayPayload,
    OverlayPublication,
    OverlayRepresentation,
    OverlayRowPhase,
    OverlayTarget,
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
    StagedLifecycle,
    StagedPort,
    WorkloadConfig,
    config_field,
    declared_config_field,
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
    enabled: bool = declared_config_field(_INPLACE_ENABLED)
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
class InPlaceTTTStepBundle:
    steps: tuple[InPlaceTTTStep, ...]


@dataclass(frozen=True, slots=True)
class InPlaceTTTPublicationBundle:
    publications: tuple[InPlaceTTTPublication, ...]


@dataclass(frozen=True, slots=True)
class _CapturedUpdateGroup:
    request: RequestIdentity
    phase: InPlaceTTTPhase
    update_enabled: bool
    packed_rows: tuple[int, ...]


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
    captured_operands: StagedPort[DenseOperandCaptureBatch] = StagedPort(
        lifecycle=StagedLifecycle.DENSE_OPERAND_CAPTURED
    )
    finished_requests: StagedPort[RequestFinishedBatch] = StagedPort(
        lifecycle=StagedLifecycle.REQUEST_FINISHED
    )
    captured_steps: HostOutput[InPlaceTTTStepBundle] = HostOutput()
    captured_publications: HostOutput[InPlaceTTTPublicationBundle] = HostOutput()
    cleanup: HostOutput[InPlaceTTTCleanupReceipt] = HostOutput()

    def __init__(self, config: InPlaceTTTConfig = InPlaceTTTConfig()) -> None:
        self.update_count = 0
        self.trained_rows = 0
        self.fast_weight_slot_count = 0
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
                OverlayChannel(
                    "inplace_ttt",
                    f"layer-{layer}-down-proj",
                    OverlayTarget(layer, "mlp", "down_proj.weight"),
                    capture_operands=True,
                ): self._overlay_store
                for layer in self.config.layers
            }
        )

    @task(
        inputs={"batch": staged(captured_operands)},
        outputs={"steps": host_output(captured_steps)},
        effects=(maybe_transition(fast_weights),),
        timing=Between("model.execute", "step.end"),
        implementation=extern(
            "tllm.inplace_ttt:captured_fast_weight@1",
            capabilities=(
                "eager_only",
                "parameter_overlay/operand_capture/mlp_down_proj",
                "parameter_overlay/tensor_parallel_size/1",
            ),
        ),
    )
    def update_captured(
        self, batch: DenseOperandCaptureBatch
    ) -> InPlaceTTTStepBundle:
        steps: list[InPlaceTTTStep] = []
        state_changed = False
        for capture in batch.captures:
            if capture.base_weight.target != capture.target or any(
                rows.target != capture.target for rows in capture.rows
            ):
                raise RuntimeError(
                    "captured operands do not match their declared target"
                )
            groups: list[_CapturedUpdateGroup] = []
            for rows in capture.rows:
                phase = (
                    InPlaceTTTPhase.PREFILL
                    if rows.phase is OverlayRowPhase.PREFILL
                    else InPlaceTTTPhase.DECODE
                )
                update_enabled = (
                    self.config.update_prefill
                    if phase is InPlaceTTTPhase.PREFILL
                    else self.config.update_decode
                )
                if (
                    groups
                    and groups[-1].request == rows.request
                    and groups[-1].update_enabled == update_enabled
                    and groups[-1].packed_rows[-1] + 1
                    == rows.packed_rows[0]
                ):
                    previous = groups[-1]
                    groups[-1] = _CapturedUpdateGroup(
                        request=previous.request,
                        phase=previous.phase,
                        update_enabled=previous.update_enabled,
                        packed_rows=(
                            previous.packed_rows + rows.packed_rows
                        ),
                    )
                else:
                    groups.append(
                        _CapturedUpdateGroup(
                            request=rows.request,
                            phase=phase,
                            update_enabled=update_enabled,
                            packed_rows=rows.packed_rows,
                        )
                    )
            for group in groups:
                row_index = torch.tensor(
                    group.packed_rows,
                    device=capture.down_proj_input.device,
                    dtype=torch.long,
                )
                step = self.fast_weights.execute(
                    InPlaceTTTBatch(
                        key=InPlaceTTTKey(
                            group.request, capture.target.layer
                        ),
                        phase=group.phase,
                        hidden=capture.down_proj_input.index_select(
                            0, row_index
                        ),
                        target=capture.mlp_input.index_select(0, row_index),
                        base_weight=capture.base_weight.tensor,
                    ),
                    chunk_size=self.config.chunk_size,
                    learning_rate=self.config.learning_rate,
                    update_enabled=group.update_enabled,
                )
                steps.append(step)
                state_changed = state_changed or step.state_changed
                self.update_count += step.update_count
                self.trained_rows += step.trained_rows
        if state_changed:
            self.advance_state(InPlaceTTTWorkload.fast_weights)
        self.fast_weight_slot_count = len(self.fast_weights.slots)
        return InPlaceTTTStepBundle(tuple(steps))

    def _publish_step(
        self, step: InPlaceTTTStep, *, ordinal: int
    ) -> InPlaceTTTPublication:
        if not step.update_count:
            return InPlaceTTTPublication(step.key.request, None, 0)
        lineage = OverlayLineage(
            OverlayChannel(
                "inplace_ttt",
                f"layer-{step.key.layer}-down-proj",
                OverlayTarget(step.key.layer, "mlp", "down_proj.weight"),
                capture_operands=True,
            ),
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
            f"{step.key.layer}:{self.update_count}:{ordinal}"
        )
        store.stage(transaction, publication)
        version = store.commit(transaction)
        store.advance_request(
            OverlayLeaseKey(step.key.request, lineage), version.ref
        )
        return InPlaceTTTPublication(
            step.key.request, version.ref, step.update_count
        )

    @task(
        inputs={"bundle": result("update_captured", "steps")},
        outputs={"publications": host_output(captured_publications)},
        effects=(read(fast_weights), maybe_transition(overlay_versions)),
        timing=Before("next.prepare"),
        implementation=extern(
            "tllm.parameter_overlay:publish_captured_exact@1",
            capabilities=(
                "eager_only",
                "parameter_overlay/representation/dense_replace",
                "parameter_overlay/scope/request",
                "parameter_overlay/phase/prefill",
                "parameter_overlay/phase/decode",
                "parameter_overlay/pinning/explicit_request_advance",
                "parameter_overlay/mixed_versions/true",
                "parameter_overlay/tensor_parallel_size/1",
                "parameter_overlay/operand_capture/mlp_down_proj",
            ),
        ),
    )
    def publish_captured(
        self, bundle: InPlaceTTTStepBundle
    ) -> InPlaceTTTPublicationBundle:
        publications = tuple(
            self._publish_step(step, ordinal=ordinal)
            for ordinal, step in enumerate(bundle.steps)
        )
        if any(publication.version is not None for publication in publications):
            self.advance_state(InPlaceTTTWorkload.overlay_versions)
        return InPlaceTTTPublicationBundle(publications)

    @task(
        inputs={"finished": staged(finished_requests)},
        outputs={"requests": temporary()},
        timing=Before("next.prepare"),
        implementation=extern("tllm.parameter_overlay:release_requests@1"),
    )
    def finish(self, finished: RequestFinishedBatch) -> tuple[RequestIdentity, ...]:
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
        self.fast_weight_slot_count = len(self.fast_weights.slots)
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
        retired_versions = sum(
            self.overlay_versions.store.retire_request(request).retired_versions
            for request in requests
        )
        if retired_versions:
            self.advance_state(InPlaceTTTWorkload.overlay_versions)
        return InPlaceTTTCleanupReceipt(cleared, retired_versions)

    def runtime_services(self) -> LocalRuntimeServices:
        return LocalRuntimeServices(
            cold_specialize=self.cold_specialize,
            diagnostics=lambda: MappingProxyType(
                {
                    "update_count": self.update_count,
                    "trained_rows": self.trained_rows,
                    "dense_apply_rows": self._overlay_store.backend_applied_rows,
                    "fast_weight_slots": self.fast_weight_slot_count,
                    "overlay_bytes": self._overlay_store.byte_size,
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
    "InPlaceTTTKey",
    "InPlaceTTTPhase",
    "InPlaceTTTPublication",
    "InPlaceTTTPublicationBundle",
    "InPlaceTTTSlot",
    "InPlaceTTTStep",
    "InPlaceTTTStepBundle",
    "InPlaceTTTWorkload",
]
