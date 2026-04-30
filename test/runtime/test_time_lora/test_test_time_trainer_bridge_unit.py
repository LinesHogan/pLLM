#!/usr/bin/env python3
"""Unit tests for shared test-time trainer bridge contracts."""

from __future__ import annotations

import sys
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
    def test_training_example_requires_strict_identity_and_token_shapes(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainingExample

        example = _example()

        self.assertEqual(example.request_id, "request-a")
        self.assertEqual(example.prompt_id, "prompt-a")
        self.assertEqual(example.sample_id, "sample-a")
        self.assertEqual(example.output_id, "output-a")
        self.assertEqual(example.token_ids, (11, 12, 13))
        self.assertEqual(example.labels, (-100, 12, 13))
        self.assertEqual(example.rewards, (1.0,))
        self.assertEqual(example.logprobs, (-0.2, -0.1, -0.3))
        with self.assertRaises(TypeError):
            example.metadata["new"] = "blocked"  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "request_id must not contain surrounding whitespace"):
            TestTimeTrainingExample(
                request_id=" request-a ",
                prompt_id="prompt-a",
                sample_id="sample-a",
                output_id="output-a",
                objective_kind="feedback_lm",
                token_ids=(1,),
            )

        with self.assertRaisesRegex(TypeError, "token_ids must contain int"):
            TestTimeTrainingExample(
                request_id="request-a",
                prompt_id="prompt-a",
                sample_id="sample-a",
                output_id="output-a",
                objective_kind="feedback_lm",
                token_ids=(1, True),  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(ValueError, "labels length must match token_ids"):
            TestTimeTrainingExample(
                request_id="request-a",
                prompt_id="prompt-a",
                sample_id="sample-a",
                output_id="output-a",
                objective_kind="feedback_lm",
                token_ids=(1, 2),
                labels=(1,),
            )

        with self.assertRaisesRegex(ValueError, "objective_kind"):
            TestTimeTrainingExample(
                request_id="request-a",
                prompt_id="prompt-a",
                sample_id="sample-a",
                output_id="output-a",
                objective_kind="not-real",  # type: ignore[arg-type]
                token_ids=(1,),
            )

    def test_batch_and_request_scope_are_strict(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerRequest, TestTimeTrainingBatch

        batch = TestTimeTrainingBatch(examples=(_example(),))

        self.assertEqual(batch.request_ids, ("request-a",))
        with self.assertRaises(TypeError):
            batch.metadata["new"] = "blocked"  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "duplicate training example identity"):
            TestTimeTrainingBatch(examples=(_example(), _example()))

        request = TestTimeTrainerRequest(
            algorithm_id="fttt",
            adapter_key="adapter-a",
            target_modules=("model.layers.0.self_attn.q_proj",),
            rank=1,
            scope="request",
            request_ids=("request-a",),
            examples=batch,
        )
        self.assertEqual(request.scope, "request")
        self.assertEqual(request.request_ids, ("request-a",))
        self.assertIs(request.examples.examples[0], batch.examples[0])

        with self.assertRaisesRegex(ValueError, "request-scoped trainer request requires request_ids"):
            TestTimeTrainerRequest(
                algorithm_id="fttt",
                adapter_key="adapter-a",
                target_modules=("model.layers.0.self_attn.q_proj",),
                rank=1,
                scope="request",
                request_ids=(),
                examples=batch,
            )

        with self.assertRaisesRegex(ValueError, "global trainer request must not include request_ids"):
            TestTimeTrainerRequest(
                algorithm_id="fttt",
                adapter_key="adapter-a",
                target_modules=("model.layers.0.self_attn.q_proj",),
                rank=1,
                scope="global",
                request_ids=("request-a",),
                examples=batch,
            )

        with self.assertRaisesRegex(ValueError, "examples request_ids must match"):
            TestTimeTrainerRequest(
                algorithm_id="fttt",
                adapter_key="adapter-a",
                target_modules=("model.layers.0.self_attn.q_proj",),
                rank=1,
                scope="request",
                request_ids=("request-b",),
                examples=batch,
            )

        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            TestTimeTrainerRequest(
                algorithm_id="fttt",
                adapter_key="adapter-a",
                target_modules=("model.layers.0.self_attn.q_proj",),
                rank=1,
                scope="request",
                request_ids=("request-a", "request-a"),
                examples=batch,
            )

        with self.assertRaisesRegex(ValueError, "examples request_ids must match"):
            TestTimeTrainerRequest(
                algorithm_id="fttt",
                adapter_key="adapter-a",
                target_modules=("model.layers.0.self_attn.q_proj",),
                rank=1,
                scope="request",
                request_ids=("request-a", "request-b"),
                examples=batch,
            )

        multi_request_batch = TestTimeTrainingBatch(examples=(_example("request-a"), _example("request-b", "sample-b")))
        with self.assertRaisesRegex(ValueError, "examples request_ids must match"):
            TestTimeTrainerRequest(
                algorithm_id="fttt",
                adapter_key="adapter-a",
                target_modules=("model.layers.0.self_attn.q_proj",),
                rank=1,
                scope="request",
                request_ids=("request-b", "request-a"),
                examples=multi_request_batch,
            )

        with self.assertRaisesRegex(ValueError, "target_modules"):
            TestTimeTrainerRequest(
                algorithm_id="fttt",
                adapter_key="adapter-a",
                target_modules=(),
                rank=1,
                scope="request",
                request_ids=("request-a",),
                examples=batch,
            )

    def test_trainer_result_preserves_update_objects_and_metrics(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerResult

        update = _update()
        result = TestTimeTrainerResult(
            updates=(update,),
            metrics={"loss": 0.25, "trained_examples": 1},
        )

        self.assertIs(result.updates[0], update)
        self.assertEqual(result.metrics["loss"], 0.25)
        with self.assertRaises(TypeError):
            result.metrics["extra"] = 1.0  # type: ignore[index]

        with self.assertRaisesRegex(TypeError, "updates must contain TestTimeLoRAUpdate"):
            TestTimeTrainerResult(updates=(object(),))  # type: ignore[arg-type]

        with self.assertRaisesRegex(TypeError, "metrics values must be real"):
            TestTimeTrainerResult(metrics={"loss": True})  # type: ignore[arg-type]

    def test_fake_bridge_stages_returned_updates_to_lora_service(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.test_time_lora import TestTimeLoRAService
        from tllm.runtime.test_time_trainer import FakeTestTimeTrainerBridge

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

    def test_stage_trainer_result_updates_rejects_updates_outside_original_request(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.test_time_lora import TestTimeLoRAService
        from tllm.runtime.test_time_trainer import stage_trainer_result_updates

        request = _request()

        incompatible_updates = [
            ("trainer result update adapter_key does not match request", _update(adapter_key="adapter-b")),
            ("trainer result update scope does not match request", _update(scope="global", request_ids=())),
            ("trainer result update rank does not match request", _update(rank=2)),
            ("trainer result update request_ids must match request", _update(request_ids=("request-b",))),
            ("trainer result update request_ids must match request", _update(request_ids=("request-a", "request-b"))),
            (
                "trainer result update target_modules must be within request target_modules",
                _update(target_module="model.layers.1.self_attn.q_proj"),
            ),
        ]

        for message, update in incompatible_updates:
            service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=2)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                stage_trainer_result_updates(request, TestTimeTrainerResult(updates=(update,)), service)
            self.assertEqual(service.commit_staged_updates(), 0)

    def test_stage_trainer_result_updates_is_atomic_when_later_update_is_invalid(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.test_time_lora import TestTimeLoRAService
        from tllm.runtime.test_time_trainer import stage_trainer_result_updates

        service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1)
        result = TestTimeTrainerResult(
            updates=(
                _update(),
                _update(target_module="model.layers.1.self_attn.q_proj"),
            )
        )

        with self.assertRaisesRegex(ValueError, "trainer result update target_modules must be within request target_modules"):
            stage_trainer_result_updates(_request(), result, service)
        self.assertEqual(service.commit_staged_updates(), 0)

    def test_stage_trainer_result_updates_is_atomic_when_service_validation_fails(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.test_time_lora import TestTimeLoRAService
        from tllm.runtime.test_time_trainer import stage_trainer_result_updates

        service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1)
        missing_service_module = "model.layers.1.self_attn.q_proj"
        request = _request(
            target_modules=(
                "model.layers.0.self_attn.q_proj",
                missing_service_module,
            )
        )
        result = TestTimeTrainerResult(
            updates=(
                _update(),
                _update(target_module=missing_service_module),
            )
        )

        with self.assertRaisesRegex(RuntimeError, "unknown LoRA target module"):
            stage_trainer_result_updates(request, result, service)
        self.assertEqual(service.commit_staged_updates(), 0)

    def test_in_process_bridge_validates_and_stages_callable_result(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.test_time_lora import TestTimeLoRAService
        from tllm.runtime.test_time_trainer import InProcessTestTimeTrainerBridge

        service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1)
        seen_requests = []

        def train(request):
            seen_requests.append(request)
            return TestTimeTrainerResult(updates=(_update(),), metrics={"loss": 0.5})

        bridge = InProcessTestTimeTrainerBridge(train, lora_service=service)
        result = bridge.train(_request())

        self.assertEqual(result.metrics["loss"], 0.5)
        self.assertEqual(seen_requests, [_request()])
        self.assertEqual(service.commit_staged_updates(), 1)

        broken_bridge = InProcessTestTimeTrainerBridge(lambda request: object())
        with self.assertRaisesRegex(TypeError, "trainer callable must return TestTimeTrainerResult"):
            broken_bridge.train(_request())

    def test_bridges_reject_incompatible_trainer_updates_before_staging(self) -> None:
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.test_time_lora import TestTimeLoRAService
        from tllm.runtime.test_time_trainer import FakeTestTimeTrainerBridge, InProcessTestTimeTrainerBridge

        request = _request()

        fake_service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1)
        fake_bridge = FakeTestTimeTrainerBridge(
            result=TestTimeTrainerResult(updates=(_update(request_ids=("request-b",)),)),
            lora_service=fake_service,
        )
        with self.assertRaisesRegex(ValueError, "trainer result update request_ids must match request"):
            fake_bridge.train(request)
        self.assertEqual(fake_service.commit_staged_updates(), 0)

        in_process_service = TestTimeLoRAService(manager=_FakeLoRAManager(), max_rank=1)
        in_process_bridge = InProcessTestTimeTrainerBridge(
            lambda request: TestTimeTrainerResult(updates=(_update(adapter_key="adapter-b"),)),
            lora_service=in_process_service,
        )
        with self.assertRaisesRegex(ValueError, "trainer result update adapter_key does not match request"):
            in_process_bridge.train(request)
        self.assertEqual(in_process_service.commit_staged_updates(), 0)

    def test_core_trainer_bridge_import_has_no_heavy_trainer_imports(self) -> None:
        heavy_modules = {"verl", "peft", "llamafactory", "llama_factory", "bfcl"}
        before = set(sys.modules)

        import tllm.contracts  # noqa: F401
        import tllm.contracts.test_time_training  # noqa: F401
        import tllm.runtime.test_time_trainer  # noqa: F401

        newly_imported_heavy = {name for name in heavy_modules if name not in before and name in sys.modules}
        self.assertEqual(newly_imported_heavy, set())


if __name__ == "__main__":
    unittest.main()
