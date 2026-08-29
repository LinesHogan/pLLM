"""Generic low-rank overlay translation into vLLM's adapter manager."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tllm.backends.vllm.versioned_parameter_overlay import (
    VLLMParameterOverlayController,
)
from tllm.runtime.parameter_overlay import (
    OverlayChannel,
    OverlayLineage,
    OverlayPayload,
    OverlayPublication,
    OverlayRepresentation,
    OverlayVersionStore,
)


class _AdapterManager:
    def __init__(self) -> None:
        self.models: dict[int, object] = {}

    def list_adapters(self) -> dict[int, object]:
        return dict(self.models)

    def add_adapter(self, model: object) -> bool:
        self.models[model.id] = model
        return True


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


def test_controller_materializes_and_keeps_request_version_stable() -> None:
    channel = OverlayChannel("tlm", "qv-lora")
    lineage = OverlayLineage(channel)
    store = OverlayVersionStore()
    store.stage("tx:1", OverlayPublication(lineage, 1, _payload(1.0)))
    store.commit("tx:1")
    manager = _AdapterManager()
    runtime = SimpleNamespace(
        reqid_to_promptidx={"active": 0, "new": 1},
        reqid_to_sampleidx={"active": 0, "new": 0},
    )
    runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_adapter_manager=manager),
        lora_config=SimpleNamespace(max_lora_rank=8),
    )
    controller = VLLMParameterOverlayController(
        runtime=runtime,
        runner=runner,
        stores={channel: store},
    )

    assert controller.sync_versions() == 1
    active = controller.pin_request("active")
    store.stage("tx:2", OverlayPublication(lineage, 1, _payload(2.0)))
    store.commit("tx:2")
    assert controller.sync_versions() == 1

    assert controller.pin_request("active").lora_int_id == active.lora_int_id
    assert controller.pin_request("new").lora_int_id != active.lora_int_id
    model = manager.models[active.lora_int_id]
    assert set(model.loras) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert controller.release_finished(("active",)) == 1
