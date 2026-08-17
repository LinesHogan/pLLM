#!/usr/bin/env python3
"""Unit tests for vLLM-facing test-time LoRA runtime helpers."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from types import ModuleType
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


class _FakeVLLMLoRALayer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: torch.Tensor | None,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.calls.append(
            (
                int(index),
                lora_a.clone(),
                lora_b.clone(),
                None if embeddings_tensor is None else embeddings_tensor.clone(),
                None if bias is None else bias.clone(),
            )
        )


class _FakeAdapterManager:
    def __init__(self) -> None:
        self.modules = {"model.layers.0.self_attn.q_proj": _FakeVLLMLoRALayer()}
        self.lora_slots = 2
        self.adapter_slots = 2
        self.lora_index_to_id: list[int | None] = [None, None]
        self.registered: dict[int, object] = {}
        self.activated: list[int] = []

    def list_adapters(self) -> dict[int, object]:
        return dict(self.registered)

    def remove_adapter(self, adapter_id: int) -> bool:
        adapter_id = int(adapter_id)
        try:
            slot_index = self.lora_index_to_id.index(adapter_id)
        except ValueError:
            pass
        else:
            self.lora_index_to_id[slot_index] = None
        return self.registered.pop(adapter_id, None) is not None

    def add_adapter(self, adapter: object) -> bool:
        adapter_id = int(getattr(adapter, "id"))
        self.registered[adapter_id] = adapter
        return True

    def activate_adapter(self, adapter_id: int) -> bool:
        adapter_id = int(adapter_id)
        self.activated.append(adapter_id)
        try:
            slot_index = self.lora_index_to_id.index(adapter_id)
        except ValueError:
            slot_index = self.lora_index_to_id.index(None)
            self.lora_index_to_id[slot_index] = adapter_id
        adapter = self.registered[adapter_id]
        lora = adapter.loras["model.layers.0.self_attn.q_proj"]
        lora.optimize()
        self.modules["model.layers.0.self_attn.q_proj"].set_lora(
            slot_index,
            lora.lora_a,
            lora.lora_b,
            lora.embeddings_tensor,
            lora.bias,
        )
        return True


class _FakeWorkerManager:
    def __init__(self) -> None:
        self._adapter_manager = _FakeAdapterManager()


class _FakeLoRALayerWeights:
    def __init__(
        self,
        *,
        module_name: str,
        rank: int,
        lora_alpha: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        embeddings_tensor: torch.Tensor | None = None,
        scaling: float = 1.0,
    ) -> None:
        self.module_name = module_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.bias = bias
        self.embeddings_tensor = embeddings_tensor
        self.scaling = scaling

    def optimize(self):
        if self.scaling != 1.0:
            self.lora_b.mul_(self.scaling)
            self.scaling = 1.0
        return self


class _FakeLoRAModel:
    def __init__(self, lora_id: int, rank: int, loras: dict[str, object]) -> None:
        self.id = int(lora_id)
        self.rank = int(rank)
        self.loras = loras


class _FakeLoRARequest:
    def __init__(self, *, lora_name: str, lora_int_id: int, lora_path: str) -> None:
        self.lora_name = lora_name
        self.lora_int_id = int(lora_int_id)
        self.lora_path = lora_path


@contextmanager
def _fake_vllm_lora_modules():
    lora_module = ModuleType("vllm.lora.lora")
    lora_module.LoRALayerWeights = _FakeLoRALayerWeights
    models_module = ModuleType("vllm.lora.models")
    models_module.LoRAModel = _FakeLoRAModel
    request_module = ModuleType("vllm.lora.request")
    request_module.LoRARequest = _FakeLoRARequest
    with mock.patch.dict(
        sys.modules,
        {
            "vllm.lora.lora": lora_module,
            "vllm.lora.models": models_module,
            "vllm.lora.request": request_module,
        },
    ):
        yield


class VLLMTestTimeLoRARuntimeUnitTest(unittest.TestCase):
    def test_vllm_manager_commits_in_memory_lora_without_disk_load(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.vllm_patch.test_time_lora_hooks import VLLMInMemoryLoRAManager

        manager = VLLMInMemoryLoRAManager(_FakeWorkerManager())
        service = TestTimeLoRAService(manager=manager, max_rank=4)
        lora_a_src = torch.ones(3, 2)
        lora_b_src = torch.full((2, 4), 2.0)
        update = TestTimeLoRAUpdate(
            adapter_key="adapter-a",
            lora_int_id=9,
            rank=2,
            scope="request",
            request_ids=("req-a",),
            target_modules={
                "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                    lora_a=lora_a_src,
                    lora_b=lora_b_src,
                    alpha=8.0,
                )
            },
        )

        service.stage_update(update)
        with _fake_vllm_lora_modules():
            self.assertEqual(service.commit_staged_updates(), 1)
            input_batch = SimpleNamespace(
                req_id_to_index={"req-a": 0},
                request_lora_mapping=np.zeros((1,), dtype=np.int32),
                lora_id_to_request_ids={},
                lora_id_to_lora_request={},
            )
            self.assertEqual(
                __import__("tllm.runtime.vllm_patch.test_time_lora_hooks", fromlist=["apply_request_lora_mapping"]).apply_request_lora_mapping(
                    service=service,
                    input_batch=input_batch,
                ),
                1,
            )

        adapter_manager = manager.adapter_manager
        self.assertIn(9, adapter_manager.registered)
        self.assertEqual(adapter_manager.activated, [])
        layer = adapter_manager.modules["model.layers.0.self_attn.q_proj"]
        self.assertEqual(len(layer.calls), 0)
        adapter_manager.activate_adapter(9)
        self.assertEqual(len(layer.calls), 1)
        slot, lora_a, lora_b, embeddings, bias = layer.calls[0]
        self.assertEqual(slot, 0)
        self.assertTrue(torch.equal(lora_a, torch.ones(3, 2)))
        self.assertTrue(torch.equal(lora_b, torch.full((2, 4), 8.0)))
        self.assertIsNone(embeddings)
        self.assertIsNone(bias)
        lora_a_src.fill_(9.0)
        lora_b_src.fill_(9.0)
        registered_lora = adapter_manager.registered[9].loras["model.layers.0.self_attn.q_proj"]
        self.assertTrue(torch.equal(registered_lora.lora_a, torch.ones(3, 2)))
        self.assertTrue(torch.equal(registered_lora.lora_b, torch.full((2, 4), 8.0)))
        self.assertEqual(registered_lora.lora_a.device.type, "cpu")
        self.assertEqual(registered_lora.lora_b.device.type, "cpu")

    def test_vllm_layer_weights_are_cpu_contiguous_snapshots(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAWeights
        from tllm.runtime.vllm_patch.test_time_lora_hooks import _to_vllm_layer_weights

        lora_a_src = torch.arange(12, dtype=torch.float32).reshape(2, 6).t()
        lora_b_src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        weights = TestTimeLoRAWeights(
            lora_a=lora_a_src,
            lora_b=lora_b_src,
            bias=torch.ones(4),
            embeddings_tensor=torch.ones(1, 4),
            alpha=4.0,
        )

        layer = _to_vllm_layer_weights("model.layers.0.self_attn.q_proj", 2, weights, _FakeLoRALayerWeights)

        self.assertEqual(layer.lora_a.device.type, "cpu")
        self.assertEqual(layer.lora_b.device.type, "cpu")
        self.assertTrue(layer.lora_a.is_contiguous())
        self.assertTrue(layer.lora_b.is_contiguous())
        self.assertTrue(torch.equal(layer.lora_a, lora_a_src.contiguous()))
        self.assertTrue(torch.equal(layer.lora_b, lora_b_src * 2.0))
        lora_a_src.fill_(99.0)
        lora_b_src.fill_(99.0)
        self.assertFalse(torch.equal(layer.lora_a, lora_a_src.contiguous()))
        self.assertFalse(torch.equal(layer.lora_b, lora_b_src))

    def test_vllm_manager_defers_registration_until_request_is_active(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.vllm_patch.test_time_lora_hooks import VLLMInMemoryLoRAManager, apply_request_lora_mapping

        manager = VLLMInMemoryLoRAManager(_FakeWorkerManager())
        service = TestTimeLoRAService(manager=manager, max_rank=4)
        update = TestTimeLoRAUpdate(
            adapter_key="future-adapter",
            lora_int_id=3,
            rank=1,
            scope="request",
            request_ids=("future-req",),
            target_modules={
                "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                    lora_a=torch.ones(3, 1),
                    lora_b=torch.ones(1, 4),
                )
            },
        )

        with _fake_vllm_lora_modules():
            service.stage_update(update)
            service.commit_staged_updates()
            self.assertEqual(manager.adapter_manager.registered, {})
            inactive_batch = SimpleNamespace(
                req_id_to_index={"other-req": 0},
                request_lora_mapping=np.zeros((1,), dtype=np.int32),
                lora_id_to_request_ids={},
                lora_id_to_lora_request={},
            )
            self.assertEqual(apply_request_lora_mapping(service=service, input_batch=inactive_batch), 0)
            self.assertEqual(manager.adapter_manager.registered, {})

            active_batch = SimpleNamespace(
                req_id_to_index={"future-req": 0},
                request_lora_mapping=np.zeros((1,), dtype=np.int32),
                lora_id_to_request_ids={},
                lora_id_to_lora_request={},
            )
            self.assertEqual(apply_request_lora_mapping(service=service, input_batch=active_batch), 1)
            self.assertIn(3, manager.adapter_manager.registered)
            self.assertEqual(manager.adapter_manager.activated, [])

    def test_vllm_repeated_update_replaces_registered_adapter_weights(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.vllm_patch.test_time_lora_hooks import VLLMInMemoryLoRAManager, apply_request_lora_mapping

        manager = VLLMInMemoryLoRAManager(_FakeWorkerManager())
        service = TestTimeLoRAService(manager=manager, max_rank=4)
        input_batch = SimpleNamespace(
            req_id_to_index={"req-a": 0},
            request_lora_mapping=np.zeros((1,), dtype=np.int32),
            lora_id_to_request_ids={},
            lora_id_to_lora_request={},
        )

        def _update(value: float) -> TestTimeLoRAUpdate:
            return TestTimeLoRAUpdate(
                adapter_key="adapter-a",
                lora_int_id=1,
                rank=1,
                scope="request",
                request_ids=("req-a",),
                target_modules={
                    "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                        lora_a=torch.ones(3, 1),
                        lora_b=torch.full((1, 4), value),
                    )
                },
            )

        with _fake_vllm_lora_modules():
            service.stage_update(_update(1.0))
            service.commit_staged_updates()
            apply_request_lora_mapping(service=service, input_batch=input_batch)
            manager.adapter_manager.activate_adapter(1)
            service.stage_update(_update(9.0))
            service.commit_staged_updates()
            apply_request_lora_mapping(service=service, input_batch=input_batch)
            manager.adapter_manager.activate_adapter(1)

        layer = manager.adapter_manager.modules["model.layers.0.self_attn.q_proj"]
        self.assertEqual(len(layer.calls), 2)
        self.assertTrue(torch.equal(layer.calls[0][2], torch.full((1, 4), 1.0)))
        self.assertTrue(torch.equal(layer.calls[1][2], torch.full((1, 4), 9.0)))

    def test_vllm_materialize_does_not_preempt_full_active_slot_before_native_reconciliation(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.vllm_patch.test_time_lora_hooks import VLLMInMemoryLoRAManager, apply_request_lora_mapping

        worker = _FakeWorkerManager()
        worker._adapter_manager.lora_index_to_id = [1]
        worker._adapter_manager.lora_slots = 1
        worker._adapter_manager.adapter_slots = 1
        worker._adapter_manager.registered[1] = _FakeLoRAModel(1, 1, {})
        manager = VLLMInMemoryLoRAManager(worker)
        service = TestTimeLoRAService(manager=manager, max_rank=4)
        update = TestTimeLoRAUpdate(
            adapter_key="adapter-b",
            lora_int_id=2,
            rank=1,
            scope="request",
            request_ids=("req-b",),
            target_modules={
                "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                    lora_a=torch.ones(3, 1),
                    lora_b=torch.ones(1, 4),
                )
            },
        )
        input_batch = SimpleNamespace(
            req_id_to_index={"req-b": 0},
            request_lora_mapping=np.zeros((1,), dtype=np.int32),
            lora_id_to_request_ids={},
            lora_id_to_lora_request={},
        )

        with _fake_vllm_lora_modules():
            service.stage_update(update)
            service.commit_staged_updates()
            self.assertEqual(apply_request_lora_mapping(service=service, input_batch=input_batch), 1)

        self.assertIn(2, worker._adapter_manager.registered)
        self.assertEqual(worker._adapter_manager.lora_index_to_id, [1])
        self.assertEqual(worker._adapter_manager.activated, [])

    def test_vllm_request_mapping_updates_existing_input_batch_rows(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.vllm_patch.test_time_lora_hooks import VLLMInMemoryLoRAManager, apply_request_lora_mapping

        manager = VLLMInMemoryLoRAManager(_FakeWorkerManager())
        service = TestTimeLoRAService(manager=manager, max_rank=4)
        update = TestTimeLoRAUpdate(
            adapter_key="adapter-a",
            lora_int_id=1,
            rank=1,
            scope="request",
            request_ids=("req-a", "missing"),
            target_modules={
                "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                    lora_a=torch.ones(3, 1),
                    lora_b=torch.ones(1, 4),
                )
            },
        )
        service.stage_update(update)
        with _fake_vllm_lora_modules():
            service.commit_staged_updates()
        input_batch = SimpleNamespace(
            req_id_to_index={"req-a": 0, "req-b": 1},
            request_lora_mapping=np.zeros((2,), dtype=np.int32),
            lora_id_to_request_ids={},
            lora_id_to_lora_request={},
        )

        applied = apply_request_lora_mapping(service=service, input_batch=input_batch)

        self.assertEqual(applied, 1)
        self.assertEqual(input_batch.request_lora_mapping.tolist(), [1, 0])
        self.assertEqual(input_batch.lora_id_to_request_ids, {1: {"req-a"}})
        self.assertEqual(input_batch.lora_id_to_lora_request[1].lora_name, "adapter-a")

    def test_vllm_request_mapping_removes_stale_lora_registry_entries(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.vllm_patch.test_time_lora_hooks import VLLMInMemoryLoRAManager, apply_request_lora_mapping

        manager = VLLMInMemoryLoRAManager(_FakeWorkerManager())
        service = TestTimeLoRAService(manager=manager, max_rank=4)
        weights = TestTimeLoRAWeights(lora_a=torch.ones(3, 1), lora_b=torch.ones(1, 4))
        first = TestTimeLoRAUpdate(
            adapter_key="adapter-a",
            lora_int_id=1,
            rank=1,
            scope="request",
            request_ids=("req-a",),
            target_modules={"model.layers.0.self_attn.q_proj": weights},
        )
        second = TestTimeLoRAUpdate(
            adapter_key="adapter-b",
            lora_int_id=2,
            rank=1,
            scope="request",
            request_ids=("req-a",),
            target_modules={"model.layers.0.self_attn.q_proj": weights},
        )
        input_batch = SimpleNamespace(
            req_id_to_index={"req-a": 0},
            request_lora_mapping=np.zeros((1,), dtype=np.int32),
            lora_id_to_request_ids={},
            lora_id_to_lora_request={},
        )

        with _fake_vllm_lora_modules():
            service.stage_update(first)
            service.commit_staged_updates()
            apply_request_lora_mapping(service=service, input_batch=input_batch)
            service.stage_update(second)
            service.commit_staged_updates()
            apply_request_lora_mapping(service=service, input_batch=input_batch)

        self.assertEqual(input_batch.request_lora_mapping.tolist(), [2])
        self.assertEqual(input_batch.lora_id_to_request_ids, {2: {"req-a"}})
        self.assertNotIn(1, input_batch.lora_id_to_lora_request)
        self.assertEqual(input_batch.lora_id_to_lora_request[2].lora_name, "adapter-b")

    def test_port_hooks_bind_and_apply_test_time_lora_service(self) -> None:
        from tllm.runtime.vllm_patch import port_runtime_hooks

        class _Core:
            def __init__(self) -> None:
                self.RUNTIME = SimpleNamespace(
                    launch_consumer_from_hooks=True,
                    event_step_id=0,
                    decode_count=0,
                    decode_prompt_idxs=[],
                    decode_sample_idxs=[],
                    decode_request_ids=[],
                    decode_prompt_idx_tensor=None,
                    decode_sample_idx_tensor=None,
                    decode_compact_count=0,
                    decode_compact_row_ids=(),
                )
                self._ORIG_LOAD_MODEL = lambda runner, *a, **k: "load_ok"
                self._ORIG_PREPARE_INPUTS = lambda runner, so: object()

        core = _Core()
        runner = object()

        with mock.patch.object(port_runtime_hooks.active_targets, "runtime_has_active_targets", return_value=True), mock.patch.object(
            port_runtime_hooks._common_hooks, "dispatch_runtime_event"
        ), mock.patch.object(
            port_runtime_hooks, "setup_runtime_hooks_if_active"
        ), mock.patch.object(
            port_runtime_hooks._sampler_patch, "maybe_prepare_sampler_decode_step"
        ), mock.patch.object(
            port_runtime_hooks._test_time_lora, "maybe_bind_test_time_lora_service"
        ) as p_bind, mock.patch.object(
            port_runtime_hooks._test_time_lora, "maybe_commit_and_apply_request_mapping"
        ) as p_apply:
            self.assertEqual(port_runtime_hooks.wrapped_load_model(core=core, runner=runner, args=tuple(), kwargs={}), "load_ok")
            port_runtime_hooks.wrapped_prepare_inputs(core=core, runner=runner, scheduler_output=object())

        p_bind.assert_called_once_with(runtime=core.RUNTIME, runner=runner)
        p_apply.assert_called_once_with(runtime=core.RUNTIME, runner=runner)

    def test_port_hooks_support_lora_only_runtime_without_residual_targets(self) -> None:
        from tllm.runtime.vllm_patch import port_runtime_hooks

        class _Core:
            def __init__(self) -> None:
                self.RUNTIME = SimpleNamespace(
                    test_time_lora_enabled=True,
                    test_time_lora_service=object(),
                )
                self._ORIG_LOAD_MODEL = lambda runner, *a, **k: "load_ok"
                self._ORIG_PREPARE_INPUTS = lambda runner, so: "prepared"

        core = _Core()
        runner = object()

        with mock.patch.object(port_runtime_hooks.active_targets, "runtime_has_active_targets", return_value=False), mock.patch.object(
            port_runtime_hooks._common_hooks, "dispatch_runtime_event"
        ) as p_dispatch, mock.patch.object(
            port_runtime_hooks, "setup_runtime_hooks_if_active"
        ) as p_setup, mock.patch.object(
            port_runtime_hooks, "prepare_decode_localization"
        ) as p_localize, mock.patch.object(
            port_runtime_hooks._test_time_lora, "maybe_bind_test_time_lora_service"
        ) as p_bind, mock.patch.object(
            port_runtime_hooks._test_time_lora, "maybe_commit_and_apply_request_mapping"
        ) as p_apply:
            self.assertEqual(port_runtime_hooks.wrapped_load_model(core=core, runner=runner, args=tuple(), kwargs={}), "load_ok")
            self.assertEqual(port_runtime_hooks.wrapped_prepare_inputs(core=core, runner=runner, scheduler_output=object()), "prepared")

        p_bind.assert_called_once_with(runtime=core.RUNTIME, runner=runner)
        p_apply.assert_called_once_with(runtime=core.RUNTIME, runner=runner)
        p_setup.assert_not_called()
        p_localize.assert_not_called()
        p_dispatch.assert_not_called()

    def test_commit_apply_records_runtime_counts(self) -> None:
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.vllm_patch.test_time_lora_hooks import VLLMInMemoryLoRAManager, maybe_commit_and_apply_request_mapping

        manager = VLLMInMemoryLoRAManager(_FakeWorkerManager())
        service = TestTimeLoRAService(manager=manager, max_rank=4)
        runtime = SimpleNamespace(test_time_lora_service=service)
        update = TestTimeLoRAUpdate(
            adapter_key="adapter-a",
            lora_int_id=1,
            rank=1,
            scope="request",
            request_ids=("req-a",),
            target_modules={
                "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                    lora_a=torch.ones(3, 1),
                    lora_b=torch.ones(1, 4),
                )
            },
        )
        input_batch = SimpleNamespace(
            req_id_to_index={"req-a": 0},
            request_lora_mapping=np.zeros((1,), dtype=np.int32),
            lora_id_to_request_ids={},
            lora_id_to_lora_request={},
        )
        runner = SimpleNamespace(input_batch=input_batch)

        service.stage_update(update)
        with _fake_vllm_lora_modules():
            self.assertEqual(maybe_commit_and_apply_request_mapping(runtime=runtime, runner=runner), 2)

        self.assertEqual(runtime.test_time_lora_last_committed, 1)
        self.assertEqual(runtime.test_time_lora_last_applied, 1)
        self.assertEqual(runtime.test_time_lora_total_committed, 1)
        self.assertEqual(runtime.test_time_lora_total_applied, 1)


class VLLMTestTimeLoRACachedModelSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("TLLM_TEST_TIME_LORA_RUN_CACHED_MODEL_SMOKE") == "1" and torch.cuda.is_available(),
        "set TLLM_TEST_TIME_LORA_RUN_CACHED_MODEL_SMOKE=1 and provide CUDA to run cached-model vLLM smoke",
    )
    def test_in_memory_lora_update_changes_qwen_generation(self) -> None:
        from vllm import LLM, SamplingParams

        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime import residual_runtime as runtime

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        runtime.set_test_time_lora_enabled(True)
        runtime.install_runner_patch()
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="bfloat16",
            gpu_memory_utilization=0.45,
            max_model_len=64,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=8,
            max_cpu_loras=1,
            trust_remote_code=True,
        )
        service = runtime.RUNTIME.test_time_lora_service
        self.assertIsNotNone(service)
        assert service is not None
        target = "model.layers.0.mlp.down_proj"
        module = service.modules[target]
        rank = 8
        input_size = int(getattr(module, "input_size"))
        output_size = int(getattr(module, "output_size"))
        params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
        prompt = "Q: 1+1=\nA:"

        base = llm.generate([prompt], params)[0].outputs[0]
        original_add_request = llm.llm_engine.add_request

        def _stage_update_for_request(request_id, prompt_arg, params_arg, *args, **kwargs):
            torch.manual_seed(123)
            service.stage_update(
                TestTimeLoRAUpdate(
                    adapter_key="smoke-adapter",
                    lora_int_id=1,
                    rank=rank,
                    scope="request",
                    request_ids=(str(request_id),),
                    target_modules={
                        target: TestTimeLoRAWeights(
                            lora_a=torch.randn(input_size, rank, dtype=torch.float32) * 3.0,
                            lora_b=torch.randn(rank, output_size, dtype=torch.float32) * 3.0,
                            alpha=float(rank),
                        )
                    },
                    step_id=1,
                )
            )
            return original_add_request(request_id, prompt_arg, params_arg, *args, **kwargs)

        llm.llm_engine.add_request = _stage_update_for_request
        try:
            adapted = llm.generate([prompt], params)[0].outputs[0]
        finally:
            llm.llm_engine.add_request = original_add_request

        self.assertNotEqual(list(base.token_ids), list(adapted.token_ids))
        self.assertGreater(int(getattr(runtime.RUNTIME, "test_time_lora_total_committed", 0) or 0), 0)
        self.assertGreater(int(getattr(runtime.RUNTIME, "test_time_lora_total_applied", 0) or 0), 0)


if __name__ == "__main__":
    unittest.main()
