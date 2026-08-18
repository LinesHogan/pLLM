#!/usr/bin/env python3
"""Unit tests for shared test-time trainer bridge contracts."""

from __future__ import annotations

import unittest

import torch


class _FakeLoRALayer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, torch.Tensor, torch.Tensor, float]] = []

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        scaling: float = 1.0,
    ) -> None:
        if bias is not None:
            raise AssertionError("test fixture does not expect bias")
        self.calls.append((int(index), lora_a.clone(), lora_b.clone(), float(scaling)))


class _FakeLoRAManager:
    def __init__(self) -> None:
        self.modules = {"model.layers.0.self_attn.q_proj": _FakeLoRALayer()}
        self.active: dict[int, object] = {}

    def activate_adapter(self, lora_id: int, update: object) -> int:
        self.active[int(lora_id)] = update
        return int(lora_id)


def _example(request_id: str = "request-a", sample_id: str = "sample-a"):
    from tllm.contracts.test_time_training import TestTimeTrainingExample

    return TestTimeTrainingExample(
        request_id=request_id,
        prompt_id="prompt-a",
        sample_id=sample_id,
        output_id="output-a",
        objective_kind="feedback_lm",
        token_ids=(11, 12, 13),
        labels=(-100, 12, 13),
        rewards=(1.0,),
        logprobs=(-0.2, -0.1, -0.3),
        metadata={"group_id": "group-a"},
    )


def _request(
    scope: str = "request",
    request_ids: tuple[str, ...] = ("request-a",),
    target_modules: tuple[str, ...] = ("model.layers.0.self_attn.q_proj",),
):
    from tllm.contracts.test_time_training import TestTimeTrainerRequest, TestTimeTrainingBatch

    return TestTimeTrainerRequest(
        algorithm_id="fttt",
        adapter_key="adapter-a",
        target_modules=target_modules,
        rank=1,
        scope=scope,  # type: ignore[arg-type]
        request_ids=request_ids,
        examples=TestTimeTrainingBatch(examples=(_example(),)),
    )


def _update(
    *,
    adapter_key: str = "adapter-a",
    rank: int = 1,
    scope: str = "request",
    request_ids: tuple[str, ...] = ("request-a",),
    target_module: str = "model.layers.0.self_attn.q_proj",
):
    from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights

    return TestTimeLoRAUpdate(
        adapter_key=adapter_key,
        lora_int_id=1,
        rank=rank,
        scope=scope,  # type: ignore[arg-type]
        request_ids=request_ids,
        target_modules={
            target_module: TestTimeLoRAWeights(
                lora_a=torch.ones(3, rank),
                lora_b=torch.full((rank, 4), 2.0),
                alpha=4.0,
            )
        },
    )


class TestTimeTrainerBridgeUnitTest(unittest.TestCase):
    def test_fake_bridge_stages_returned_updates_to_lora_service(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.test_time.trainer import FakeTestTimeTrainerBridge

        service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1)
        result = TestTimeTrainerResult(updates=(_update(),), metrics={"loss": 0.25})
        bridge = FakeTestTimeTrainerBridge(result=result, lora_service=service)

        returned = bridge.train(_request())
        committed = service.commit_staged_updates()

        self.assertIs(returned, result)
        self.assertEqual(tuple(bridge.requests), (_request(),))
        self.assertEqual(committed, 1)
        self.assertEqual(service.adapter_for_request("request-a"), 1)
        layer = service.modules["model.layers.0.self_attn.q_proj"]
        self.assertEqual(len(layer.calls), 1)
        slot, lora_a, lora_b, scaling = layer.calls[0]
        self.assertEqual(slot, 1)
        self.assertTrue(torch.equal(lora_a, torch.ones(3, 1)))
        self.assertTrue(torch.equal(lora_b, torch.full((1, 4), 2.0)))
        self.assertEqual(scaling, 4.0)


if __name__ == "__main__":
    unittest.main()
