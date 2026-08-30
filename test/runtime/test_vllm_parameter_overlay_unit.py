"""Generic low-rank overlay translation into vLLM's adapter manager."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
import torch
from vllm.lora.lora_model import LoRAModel
from vllm.lora.request import LoRARequest

from tllm.backends.protocol import OverlayBackendState
from tllm.backends.vllm.versioned_parameter_overlay import (
    VLLMParameterOverlayController,
    maybe_bind_publisher,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.runtime.parameter_overlay import (
    OverlayChannel,
    OverlayLeaseKey,
    OverlayLineage,
    OverlayPayload,
    OverlayPublication,
    OverlayRepresentation,
    OverlayVersionStore,
)


class _AdapterManager:
    def __init__(self) -> None:
        self.models: dict[int, LoRAModel] = {}
        self.fail_after_remove: int | None = None

    def list_adapters(self) -> dict[int, LoRAModel]:
        return dict(self.models)

    def add_adapter(self, model: LoRAModel) -> bool:
        self.models[model.id] = model
        return True

    def remove_adapter(self, lora_id: int) -> bool:
        self.models.pop(lora_id)
        if lora_id == self.fail_after_remove:
            raise RuntimeError("injected remove failure")
        return True


@dataclass(slots=True)
class _LoRAManager:
    _adapter_manager: _AdapterManager


@dataclass(slots=True)
class _LoRAConfig:
    max_lora_rank: int


@dataclass(slots=True)
class _RequestStates:
    req_id_to_index: Mapping[str, int]


@dataclass(slots=True)
class _LoRAState:
    requests: dict[str, LoRARequest] = field(default_factory=dict[str, LoRARequest])

    def add_request(
        self, request_id: str, request_index: int, request: LoRARequest
    ) -> None:
        del request_index
        self.requests[request_id] = request


@dataclass(slots=True)
class _RequestLoRAMapping:
    values: dict[int, int] = field(default_factory=dict[int, int])

    def __setitem__(self, index: int, value: int) -> None:
        self.values[index] = value


@dataclass(slots=True)
class _InputBatch:
    req_id_to_index: Mapping[str, int] = field(default_factory=dict[str, int])
    request_lora_mapping: _RequestLoRAMapping = field(
        default_factory=_RequestLoRAMapping
    )
    lora_id_to_request_ids: dict[int, set[str]] = field(
        default_factory=dict[int, set[str]]
    )
    lora_id_to_lora_request: dict[int, LoRARequest] = field(
        default_factory=dict[int, LoRARequest]
    )


@dataclass(slots=True)
class _Runner:
    lora_manager: _LoRAManager
    lora_config: _LoRAConfig
    req_states: _RequestStates
    lora_state: _LoRAState
    input_batch: _InputBatch


@dataclass(slots=True)
class _Execution:
    parameter_overlays: Mapping[OverlayChannel, OverlayVersionStore]


@dataclass(slots=True)
class _Runtime:
    reqid_to_promptidx: Mapping[str, int]
    reqid_to_sampleidx: Mapping[str, int]
    workload_execution: _Execution


@dataclass(slots=True)
class _OverlayHarness:
    manager: _AdapterManager
    runner: _Runner
    runtime: _Runtime
    state: OverlayBackendState

    @classmethod
    def create(
        cls, stores: Mapping[OverlayChannel, OverlayVersionStore]
    ) -> _OverlayHarness:
        manager = _AdapterManager()
        runner = _Runner(
            _LoRAManager(manager),
            _LoRAConfig(max_lora_rank=8),
            _RequestStates({}),
            _LoRAState(),
            _InputBatch(),
        )
        return cls(
            manager=manager,
            runner=runner,
            runtime=_Runtime(
                reqid_to_promptidx={"active": 0, "new": 1},
                reqid_to_sampleidx={"active": 0, "new": 0},
                workload_execution=_Execution(stores),
            ),
            state=OverlayBackendState(),
        )

    def bind(self) -> VLLMParameterOverlayController:
        controller = maybe_bind_publisher(
            runtime=self.runtime,
            runner=self.runner,
            state=self.state,
        )
        assert controller is not None
        return controller


def _payload(value: float) -> OverlayPayload:
    prefix = "base_model.model.model.layers.0.self_attn"
    return OverlayPayload(
        OverlayRepresentation.LOW_RANK_ADD,
        {
            f"{prefix}.q_proj.lora_A.default.weight": torch.full((2, 4), value),
            f"{prefix}.q_proj.lora_B.default.weight": torch.full((4, 2), value),
            f"{prefix}.v_proj.lora_A.default.weight": torch.full((2, 4), value),
            f"{prefix}.v_proj.lora_B.default.weight": torch.full((4, 2), value),
            "__tllm_lora_alpha__": torch.tensor(4.0),
        },
    )


def _store(value: float) -> tuple[OverlayChannel, OverlayVersionStore]:
    channel = OverlayChannel("tlm", "qv-lora")
    store = OverlayVersionStore()
    store.stage(
        "tx:1", OverlayPublication(OverlayLineage(channel), 1, _payload(value))
    )
    store.commit("tx:1")
    return channel, store


def test_controller_materializes_and_keeps_request_version_stable() -> None:
    channel, store = _store(1.0)
    harness = _OverlayHarness.create({channel: store})
    controller = harness.bind()

    assert controller.sync_versions() == 1
    controller.pin_request("active")
    first = store.pinned(
        OverlayLeaseKey(
            RequestIdentity("active", 0, 0), OverlayLineage(channel)
        )
    )
    assert first is not None
    store.stage(
        "tx:2",
        OverlayPublication(OverlayLineage(channel), 1, _payload(2.0)),
    )
    store.commit("tx:2")
    assert controller.sync_versions() == 1

    controller.pin_request("active")
    controller.pin_request("new")
    active = store.pinned(
        OverlayLeaseKey(
            RequestIdentity("active", 0, 0), OverlayLineage(channel)
        )
    )
    new = store.pinned(
        OverlayLeaseKey(
            RequestIdentity("new", 1, 0), OverlayLineage(channel)
        )
    )
    assert active == first
    assert new is not None and new.ref != first.ref
    model = harness.manager.models[next(iter(harness.manager.models))]
    assert set(model.loras) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert controller.release_finished(("active",)) == 1


def test_epoch_rebind_releases_old_leases_removes_adapters_and_allocates_new_ids() -> None:
    channel, first_store = _store(1.0)
    _, second_store = _store(2.0)
    harness = _OverlayHarness.create({channel: first_store})
    first = harness.bind()
    assert first.sync_versions() == 1
    first.pin_request("active")
    first_id = next(iter(harness.manager.models))

    harness.runtime.workload_execution = _Execution({channel: second_store})
    second = harness.bind()
    assert second is not first
    assert first_store.pinned(
        OverlayLeaseKey(
            RequestIdentity("active", 0, 0), OverlayLineage(channel)
        )
    ) is None
    assert harness.manager.models == {}

    assert second.sync_versions() == 1
    second.pin_request("active")
    assert next(iter(harness.manager.models)) != first_id


def test_close_removes_remaining_adapters_after_one_remove_failure() -> None:
    channel, store = _store(1.0)
    harness = _OverlayHarness.create({channel: store})
    controller = harness.bind()
    assert controller.sync_versions() == 1
    controller.pin_request("active")
    first_id = next(iter(harness.manager.models))
    store.stage(
        "tx:2",
        OverlayPublication(OverlayLineage(channel), 1, _payload(2.0)),
    )
    store.commit("tx:2")
    assert controller.sync_versions() == 1
    harness.manager.fail_after_remove = first_id

    with pytest.raises(RuntimeError, match="could not remove"):
        controller.close()

    assert harness.manager.models == {}
    assert store.pinned(
        OverlayLeaseKey(
            RequestIdentity("active", 0, 0), OverlayLineage(channel)
        )
    ) is None
