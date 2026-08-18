#!/usr/bin/env python3
"""Unit tests for the test-time LoRA runtime service."""

from __future__ import annotations

import unittest

import torch


class _FakeLoRALayer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None, float]] = []

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor, bias=None, scaling: float = 1.0) -> None:
        self.calls.append((int(index), lora_a.clone(), lora_b.clone(), None if bias is None else bias.clone(), float(scaling)))


class _FakeLoRAManager:
    def __init__(self) -> None:
        self.modules = {"model.layers.0.self_attn.q_proj": _FakeLoRALayer()}
        self.active: dict[int, object] = {}

    def activate_adapter(self, lora_id: int, request: object) -> int:
        self.active[int(lora_id)] = request
        return int(lora_id)


class TestTimeLoRAServiceUnitTest(unittest.TestCase):
    def test_stages_adapter_slot_and_updates_target_module(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time.lora import TestTimeLoRAService

        manager = _FakeLoRAManager()
        service = TestTimeLoRAService(manager=manager, max_rank=4)
        update = TestTimeLoRAUpdate(
            adapter_key="req-7",
            lora_int_id=1,
            rank=2,
            scope="request",
            request_ids=("request-a",),
            target_modules={
                "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                    lora_a=torch.ones(3, 2),
                    lora_b=torch.full((2, 4), 2.0),
                    alpha=8.0,
                )
            },
        )

        service.stage_update(update)
        service.commit_staged_updates()

        layer = manager.modules["model.layers.0.self_attn.q_proj"]
        self.assertEqual(len(layer.calls), 1)
        slot, lora_a, lora_b, bias, scaling = layer.calls[0]
        self.assertEqual(slot, 1)
        self.assertTrue(torch.equal(lora_a, torch.ones(3, 2)))
        self.assertTrue(torch.equal(lora_b, torch.full((2, 4), 2.0)))
        self.assertIsNone(bias)
        self.assertEqual(scaling, 4.0)
        self.assertEqual(service.adapter_for_request("request-a"), 1)


if __name__ == "__main__":
    unittest.main()
