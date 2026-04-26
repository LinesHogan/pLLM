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


class _DuckUpdate:
    adapter_key = "duck"
    lora_int_id = 1
    rank = 1
    scope = "request"
    request_ids = ("request-a",)
    target_modules = {
        "model.layers.0.self_attn.q_proj": object(),
    }


class TestTimeLoRAServiceUnitTest(unittest.TestCase):
    def test_contract_package_exports_lora_contracts(self) -> None:
        from tllm import contracts

        self.assertEqual(contracts.TestTimeLoRAUpdate.__name__, "TestTimeLoRAUpdate")
        self.assertEqual(contracts.TestTimeLoRAWeights.__name__, "TestTimeLoRAWeights")

    def test_stages_adapter_slot_and_updates_target_module(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time_lora import TestTimeLoRAService

        manager = _FakeLoRAManager()
        service = TestTimeLoRAService(manager=manager, max_rank=4, max_adapters=2)
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

    def test_rejects_unknown_target_and_rank_over_capacity(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time_lora import TestTimeLoRAService

        service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1, max_adapters=2)
        update = TestTimeLoRAUpdate(
            adapter_key="bad",
            lora_int_id=1,
            rank=2,
            scope="request",
            request_ids=("request-a",),
            target_modules={
                "missing": TestTimeLoRAWeights(
                    lora_a=torch.ones(3, 2),
                    lora_b=torch.ones(2, 4),
                )
            },
        )

        with self.assertRaisesRegex(RuntimeError, "rank exceeds"):
            service.stage_update(update)

        update_rank_ok = TestTimeLoRAUpdate(
            adapter_key="bad",
            lora_int_id=1,
            rank=1,
            scope="request",
            request_ids=("request-a",),
            target_modules={
                "missing": TestTimeLoRAWeights(
                    lora_a=torch.ones(3, 1),
                    lora_b=torch.ones(1, 4),
                )
            },
        )
        with self.assertRaisesRegex(RuntimeError, "unknown LoRA target module"):
            service.stage_update(update_rank_ok)

    def test_rejects_ambiguous_request_scope_without_request_ids(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights

        with self.assertRaisesRegex(ValueError, "request-scoped LoRA update requires request_ids"):
            TestTimeLoRAUpdate(
                adapter_key="bad",
                lora_int_id=1,
                rank=1,
                scope="request",
                target_modules={
                    "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                        lora_a=torch.ones(3, 1),
                        lora_b=torch.ones(1, 4),
                    )
                },
            )

    def test_rejects_implicit_contract_type_coercions(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRATarget, TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time_lora import TestTimeLoRAService

        weights = TestTimeLoRAWeights(lora_a=torch.ones(3, 1), lora_b=torch.ones(1, 4))

        with self.assertRaisesRegex(TypeError, "module_name must be str"):
            TestTimeLoRATarget(module_name=123)  # type: ignore[arg-type]

        with self.assertRaisesRegex(TypeError, "packed_members must contain str"):
            TestTimeLoRATarget(module_name="q_proj", packed_members=("q", 7))  # type: ignore[arg-type]

        with self.assertRaisesRegex(TypeError, "lora_int_id must be int"):
            TestTimeLoRAUpdate(
                adapter_key="bad",
                lora_int_id=1.2,  # type: ignore[arg-type]
                rank=1,
                scope="request",
                request_ids=("request-a",),
                target_modules={"model.layers.0.self_attn.q_proj": weights},
            )

        with self.assertRaisesRegex(TypeError, "rank must be int"):
            TestTimeLoRAUpdate(
                adapter_key="bad",
                lora_int_id=1,
                rank="1",  # type: ignore[arg-type]
                scope="request",
                request_ids=("request-a",),
                target_modules={"model.layers.0.self_attn.q_proj": weights},
            )

        with self.assertRaisesRegex(TypeError, "request_ids must contain str"):
            TestTimeLoRAUpdate(
                adapter_key="bad",
                lora_int_id=1,
                rank=1,
                scope="request",
                request_ids=(7,),  # type: ignore[arg-type]
                target_modules={"model.layers.0.self_attn.q_proj": weights},
            )

        with self.assertRaisesRegex(TypeError, "adapter_key must be str"):
            TestTimeLoRAUpdate(
                adapter_key=8,  # type: ignore[arg-type]
                lora_int_id=1,
                rank=1,
                scope="request",
                request_ids=("request-a",),
                target_modules={"model.layers.0.self_attn.q_proj": weights},
            )

        with self.assertRaisesRegex(TypeError, "target module name must be str"):
            TestTimeLoRAUpdate(
                adapter_key="bad",
                lora_int_id=1,
                rank=1,
                scope="request",
                request_ids=("request-a",),
                target_modules={9: weights},  # type: ignore[dict-item]
            )

        with self.assertRaisesRegex(TypeError, "step_id must be int"):
            TestTimeLoRAUpdate(
                adapter_key="bad",
                lora_int_id=1,
                rank=1,
                scope="request",
                request_ids=("request-a",),
                step_id=True,  # type: ignore[arg-type]
                target_modules={"model.layers.0.self_attn.q_proj": weights},
            )

        with self.assertRaisesRegex(TypeError, "alpha must be real"):
            TestTimeLoRAWeights(lora_a=torch.ones(3, 1), lora_b=torch.ones(1, 4), alpha=True)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "adapter_key must not contain surrounding whitespace"):
            TestTimeLoRAUpdate(
                adapter_key=" adapter ",
                lora_int_id=1,
                rank=1,
                scope="request",
                request_ids=("request-a",),
                target_modules={"model.layers.0.self_attn.q_proj": weights},
            )

        with self.assertRaisesRegex(ValueError, "request_id must not contain surrounding whitespace"):
            TestTimeLoRAUpdate(
                adapter_key="bad",
                lora_int_id=1,
                rank=1,
                scope="request",
                request_ids=(" request-a ",),
                target_modules={"model.layers.0.self_attn.q_proj": weights},
            )

        with self.assertRaisesRegex(ValueError, "target module name must not contain surrounding whitespace"):
            TestTimeLoRAUpdate(
                adapter_key="bad",
                lora_int_id=1,
                rank=1,
                scope="request",
                request_ids=("request-a",),
                target_modules={" model.layers.0.self_attn.q_proj ": weights},
            )

        with self.assertRaisesRegex(TypeError, "request_id must be str"):
            TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1, max_adapters=2).adapter_for_request(7)  # type: ignore[arg-type]

        with self.assertRaisesRegex(TypeError, "max_rank must be int"):
            TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1.5, max_adapters=2)  # type: ignore[arg-type]

        with self.assertRaisesRegex(TypeError, "max_adapters must be int"):
            TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1, max_adapters=True)  # type: ignore[arg-type]

    def test_rejects_duck_typed_updates_at_service_boundary(self) -> None:
        from tllm.runtime.test_time_lora import TestTimeLoRAService

        service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1, max_adapters=2)

        with self.assertRaisesRegex(TypeError, "update must be TestTimeLoRAUpdate"):
            service.stage_update(_DuckUpdate())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
