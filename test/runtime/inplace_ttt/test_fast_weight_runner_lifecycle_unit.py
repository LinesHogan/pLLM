#!/usr/bin/env python3
"""Unit tests for In-Place TTT fast-weight runner lifecycle integration."""

from __future__ import annotations

from types import SimpleNamespace
import os
import unittest

import numpy as np
import torch


class _FakeDownProj:
    def __init__(self) -> None:
        self.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.bias = None

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(hidden, self.weight), None


class _FakeGateUpProj:
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden, None


class _FakeMLP:
    def __init__(self) -> None:
        self.gate_up_proj = _FakeGateUpProj()
        self.act_fn = torch.nn.Identity()
        self.down_proj = _FakeDownProj()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj.forward(hidden)
        activated = self.act_fn(gate_up)
        output, _ = self.down_proj.forward(activated)
        return output


class _FakeLayer:
    def __init__(self) -> None:
        self.mlp = _FakeMLP()


class _FakeModel:
    def __init__(self, layer_count: int = 2) -> None:
        self.model = SimpleNamespace(layers=[_FakeLayer() for _ in range(layer_count)])


class _FakeInputBatch:
    def __init__(self, *, computed: int, prompt: int) -> None:
        self.req_ids = ["req-a"]
        self.num_reqs = 1
        self.req_id_to_index = {"req-a": 0}
        self.num_prompt_tokens = [prompt]
        self.num_computed_tokens_cpu = [computed]


class InPlaceTTTFastWeightRunnerLifecycleUnitTest(unittest.TestCase):
    def _runtime(self) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(
                inplace_ttt_fast_weight_enabled=True,
                inplace_ttt_fast_weight_layers=(0,),
                inplace_ttt_fast_weight_chunk_size=4,
                inplace_ttt_fast_weight_lr=0.5,
                inplace_ttt_fast_weight_mode="exact",
                inplace_ttt_fast_weight_decode_update_enabled=False,
                inplace_ttt_fast_weight_key_mode="static",
                inplace_ttt_fast_weight_static_request_id="static-req",
                inplace_ttt_fast_weight_fail_on_unsupported=True,
            ),
            inplace_ttt_fast_weight_phase="prefill",
            inplace_ttt_fast_weight_handles={},
            inplace_ttt_fast_weight_hook_spec=None,
        )

    def test_lifecycle_installs_static_key_wrapper_once_and_restores(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey
        from tllm.runtime.inplace_ttt_fast_weight_runtime import (
            maybe_install_inplace_ttt_fast_weight_wrappers,
            restore_inplace_ttt_fast_weight_wrappers,
        )

        runtime = self._runtime()
        runner = SimpleNamespace(model=_FakeModel())
        mlp = runner.model.model.layers[0].mlp
        base_probe = mlp.forward(torch.tensor([[3.0, 0.0]]))

        installed = maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)
        installed_again = maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)
        first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        second = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        mlp.forward(first)
        mlp.forward(second)
        key = FastWeightKey(request_id="static-req", sample_idx=0, layer=0)
        stats = runtime.inplace_ttt_fast_weight_state.stats(key)

        self.assertEqual(installed, 1)
        self.assertEqual(installed_again, 0)
        self.assertEqual(set(runtime.inplace_ttt_fast_weight_handles), {"model.layers.0.mlp"})
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.trained_rows, 4)
        self.assertEqual(restore_inplace_ttt_fast_weight_wrappers(runtime), 1)
        restored_probe = mlp.forward(torch.tensor([[3.0, 0.0]]))
        self.assertTrue(torch.equal(restored_probe, base_probe))

    def test_low_rank_runtime_mode_fails_fast_before_installing_wrapper(self) -> None:
        from tllm.runtime.inplace_ttt_fast_weight_runtime import maybe_install_inplace_ttt_fast_weight_wrappers

        runtime = self._runtime()
        runtime.config.inplace_ttt_fast_weight_mode = "low_rank"
        runner = SimpleNamespace(model=_FakeModel())

        with self.assertRaisesRegex(NotImplementedError, "low_rank"):
            maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)
        self.assertEqual(runtime.inplace_ttt_fast_weight_handles, {})

    def test_low_rank_runtime_mode_restores_existing_exact_wrapper_before_failing(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey
        from tllm.runtime.inplace_ttt_fast_weight_runtime import maybe_install_inplace_ttt_fast_weight_wrappers

        runtime = self._runtime()
        runner = SimpleNamespace(model=_FakeModel())
        mlp = runner.model.model.layers[0].mlp

        self.assertEqual(maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner), 1)
        mlp.forward(torch.ones(4, 2))
        key = FastWeightKey(request_id="static-req", sample_idx=0, layer=0)
        self.assertEqual(runtime.inplace_ttt_fast_weight_state.stats(key).update_count, 1)

        runtime.config.inplace_ttt_fast_weight_mode = "low_rank"
        with self.assertRaisesRegex(NotImplementedError, "low_rank"):
            maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)

        self.assertEqual(runtime.inplace_ttt_fast_weight_handles, {})
        self.assertIsNone(runtime.inplace_ttt_fast_weight_hook_spec)
        mlp.forward(torch.ones(4, 2))
        self.assertEqual(runtime.inplace_ttt_fast_weight_state.stats(key).update_count, 1)

    def test_runtime_configuration_rejects_unknown_fast_weight_mode(self) -> None:
        from tllm.runtime import residual_runtime

        runtime = residual_runtime.ResidualRuntimeState()
        old_runtime = residual_runtime.RUNTIME
        try:
            residual_runtime.RUNTIME = runtime
            with self.assertRaisesRegex(ValueError, "exact.*low_rank"):
                residual_runtime.set_inplace_ttt_fast_weight_enabled(True, mode="compressed")
        finally:
            residual_runtime.RUNTIME = old_runtime

    def test_disabling_fast_weight_runtime_clears_request_sample_token_metadata(self) -> None:
        from tllm.runtime import residual_runtime
        from tllm.runtime.token_batch_metadata import PackedTokenRow, RequestSampleKey, RuntimeTokenBatchMeta, RuntimeTokenRow

        runtime = residual_runtime.ResidualRuntimeState()
        old_runtime = residual_runtime.RUNTIME
        runtime.inplace_ttt_current_token_batch_meta = RuntimeTokenBatchMeta(
            rows=(
                RuntimeTokenRow(
                    key=RequestSampleKey(request_id="stale-req", prompt_idx=0, sample_idx=0),
                    row=PackedTokenRow(packed_row_idx=0, logical_row_idx=0, token_offset=0),
                    phase="prefill",
                    engine_step_id=1,
                ),
            ),
            row_layout="packed",
        )
        runtime.inplace_ttt_fast_weight_phase = "decode"
        try:
            residual_runtime.RUNTIME = runtime
            residual_runtime.set_inplace_ttt_fast_weight_enabled(False)
        finally:
            residual_runtime.RUNTIME = old_runtime

        self.assertIsNone(runtime.inplace_ttt_current_token_batch_meta)
        self.assertEqual(runtime.inplace_ttt_fast_weight_phase, "prefill")

    def test_configured_layers_are_isolated_by_layer_key(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey
        from tllm.runtime.inplace_ttt_fast_weight_runtime import maybe_install_inplace_ttt_fast_weight_wrappers

        runtime = self._runtime()
        runtime.config.inplace_ttt_fast_weight_layers = (1,)
        runner = SimpleNamespace(model=_FakeModel())

        maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)
        runner.model.model.layers[0].mlp.forward(torch.ones(4, 2))
        runner.model.model.layers[1].mlp.forward(torch.ones(4, 2))

        state = runtime.inplace_ttt_fast_weight_state
        self.assertEqual(state.stats(FastWeightKey(request_id="static-req", sample_idx=0, layer=0)).update_count, 0)
        self.assertEqual(state.stats(FastWeightKey(request_id="static-req", sample_idx=0, layer=1)).update_count, 1)

    def test_unsupported_mlp_can_be_skipped_without_installing_handles(self) -> None:
        from tllm.runtime.inplace_ttt_fast_weight_runtime import maybe_install_inplace_ttt_fast_weight_wrappers

        runtime = self._runtime()
        runtime.config.inplace_ttt_fast_weight_fail_on_unsupported = False
        runner = SimpleNamespace(model=_FakeModel())
        runner.model.model.layers[0].mlp.down_proj.bias = torch.ones(2)

        installed = maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)

        self.assertEqual(installed, 0)
        self.assertEqual(runtime.inplace_ttt_fast_weight_installed_count, 0)
        self.assertEqual(runtime.inplace_ttt_fast_weight_skipped_count, 1)
        self.assertEqual(runtime.inplace_ttt_fast_weight_handles, {})

    def test_phase_updates_from_runner_input_batch(self) -> None:
        from tllm.runtime.inplace_ttt_fast_weight_runtime import update_inplace_ttt_fast_weight_phase_from_runner

        runtime = self._runtime()
        runner = SimpleNamespace(input_batch=_FakeInputBatch(computed=0, prompt=8))
        self.assertEqual(update_inplace_ttt_fast_weight_phase_from_runner(runtime, runner), "prefill")

        runner.input_batch = _FakeInputBatch(computed=8, prompt=8)
        self.assertEqual(update_inplace_ttt_fast_weight_phase_from_runner(runtime, runner), "decode")

    def test_current_token_metadata_tracks_packed_prefill_rows_for_batch_requests(self) -> None:
        from tllm.runtime.inplace_ttt_fast_weight_runtime import update_inplace_ttt_fast_weight_metadata_from_runner

        runtime = self._runtime()
        runtime.event_step_id = 7
        runtime.reqid_to_promptidx = {"req-a": 10, "req-b": 11}
        runtime.reqid_to_sampleidx = {"req-a": 0, "req-b": 1}
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["req-a", "req-b"],
                num_reqs=2,
                req_id_to_index={"req-a": 0, "req-b": 1},
                num_prompt_tokens=[4, 5],
                num_computed_tokens_cpu=[1, 2],
            )
        )

        phase = update_inplace_ttt_fast_weight_metadata_from_runner(
            runtime,
            runner,
            num_scheduled_tokens_np=[3, 3],
        )
        meta = runtime.inplace_ttt_current_token_batch_meta

        self.assertEqual(phase, "prefill")
        self.assertEqual([row.key.request_id for row in meta.rows], ["req-a", "req-a", "req-a", "req-b", "req-b", "req-b"])
        self.assertEqual([row.key.prompt_idx for row in meta.rows], [10, 10, 10, 11, 11, 11])
        self.assertEqual([row.key.sample_idx for row in meta.rows], [0, 0, 0, 1, 1, 1])
        self.assertEqual([row.row.packed_row_idx for row in meta.rows], [0, 1, 2, 3, 4, 5])
        self.assertEqual([row.row.token_offset for row in meta.rows], [1, 2, 3, 2, 3, 4])

    def test_current_token_metadata_accepts_numpy_count_arrays_from_vllm(self) -> None:
        from tllm.runtime.inplace_ttt_fast_weight_runtime import update_inplace_ttt_fast_weight_metadata_from_runner

        runtime = self._runtime()
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=[0, 1],
                num_reqs=2,
                req_id_to_index={0: 0, 1: 1},
                num_prompt_tokens=np.asarray([4, 4]),
                num_computed_tokens_cpu=np.asarray([0, 0]),
            )
        )

        update_inplace_ttt_fast_weight_metadata_from_runner(
            runtime,
            runner,
            num_scheduled_tokens_np=np.asarray([4, 4]),
        )

        self.assertEqual(len(runtime.inplace_ttt_current_token_batch_meta.rows), 8)

    def test_static_key_mode_rejects_multi_request_runner_batch(self) -> None:
        from tllm.runtime.inplace_ttt_fast_weight_runtime import require_static_key_single_request

        runtime = self._runtime()
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["req-a", "req-b"],
                num_reqs=2,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "requires a single active request"):
            require_static_key_single_request(runtime, runner)

    def test_request_sample_key_mode_groups_rows_by_current_runtime_metadata(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey
        from tllm.runtime.inplace_ttt_fast_weight_runtime import (
            maybe_install_inplace_ttt_fast_weight_wrappers,
            require_static_key_single_request,
        )
        from tllm.runtime.token_batch_metadata import (
            PackedTokenRow,
            RequestSampleKey,
            RuntimeTokenBatchMeta,
            RuntimeTokenRow,
        )

        runtime = self._runtime()
        runtime.config.inplace_ttt_fast_weight_key_mode = "request_sample"
        runtime.inplace_ttt_current_token_batch_meta = RuntimeTokenBatchMeta(
            rows=(
                RuntimeTokenRow(
                    key=RequestSampleKey(request_id="req-a", prompt_idx=0, sample_idx=0),
                    row=PackedTokenRow(packed_row_idx=0, logical_row_idx=0, token_offset=0),
                    phase="prefill",
                    engine_step_id=1,
                ),
                RuntimeTokenRow(
                    key=RequestSampleKey(request_id="req-a", prompt_idx=0, sample_idx=0),
                    row=PackedTokenRow(packed_row_idx=1, logical_row_idx=1, token_offset=1),
                    phase="prefill",
                    engine_step_id=1,
                ),
                RuntimeTokenRow(
                    key=RequestSampleKey(request_id="req-b", prompt_idx=1, sample_idx=0),
                    row=PackedTokenRow(packed_row_idx=2, logical_row_idx=2, token_offset=0),
                    phase="prefill",
                    engine_step_id=1,
                ),
                RuntimeTokenRow(
                    key=RequestSampleKey(request_id="req-b", prompt_idx=1, sample_idx=0),
                    row=PackedTokenRow(packed_row_idx=3, logical_row_idx=3, token_offset=1),
                    phase="prefill",
                    engine_step_id=1,
                ),
            )
        )
        runner = SimpleNamespace(model=_FakeModel(), input_batch=SimpleNamespace(req_ids=["req-a", "req-b"], num_reqs=2))

        require_static_key_single_request(runtime, runner)
        installed = maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)
        runner.model.model.layers[0].mlp.forward(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
        )

        state = runtime.inplace_ttt_fast_weight_state
        key_a = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        key_b = FastWeightKey(request_id="req-b", sample_idx=0, layer=0)
        self.assertEqual(installed, 1)
        self.assertEqual(state.stats(key_a).update_count, 0)
        self.assertEqual(state.stats(key_b).update_count, 0)
        self.assertEqual(state.stats(key_a).tail_rows, 2)
        self.assertEqual(state.stats(key_b).tail_rows, 2)

    def test_runtime_disable_restores_installed_wrapper(self) -> None:
        from tllm.runtime import residual_runtime
        from tllm.runtime.inplace_ttt_fast_weight_runtime import maybe_install_inplace_ttt_fast_weight_wrappers

        runtime = residual_runtime.ResidualRuntimeState()
        runner = SimpleNamespace(model=_FakeModel())
        runtime.config.inplace_ttt_fast_weight_enabled = True
        runtime.config.inplace_ttt_fast_weight_layers = (0,)
        runtime.config.inplace_ttt_fast_weight_chunk_size = 4
        runtime.config.inplace_ttt_fast_weight_lr = 0.5
        mlp = runner.model.model.layers[0].mlp
        base_probe = mlp.forward(torch.tensor([[3.0, 0.0]]))

        maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)
        old_runtime = residual_runtime.RUNTIME
        try:
            residual_runtime.RUNTIME = runtime
            residual_runtime.set_inplace_ttt_fast_weight_enabled(False)
        finally:
            residual_runtime.RUNTIME = old_runtime
        restored_probe = mlp.forward(torch.tensor([[3.0, 0.0]]))

        self.assertFalse(runtime.config.inplace_ttt_fast_weight_enabled)
        self.assertEqual(runtime.inplace_ttt_fast_weight_handles, {})
        self.assertTrue(torch.equal(restored_probe, base_probe))

    def test_configure_runtime_disable_restores_installed_wrapper(self) -> None:
        from tllm.runtime import residual_runtime
        from tllm.runtime.inplace_ttt_fast_weight_runtime import maybe_install_inplace_ttt_fast_weight_wrappers

        runtime = residual_runtime.ResidualRuntimeState()
        runner = SimpleNamespace(model=_FakeModel())
        runtime.config.inplace_ttt_fast_weight_enabled = True
        runtime.config.inplace_ttt_fast_weight_layers = (0,)
        mlp = runner.model.model.layers[0].mlp
        base_probe = mlp.forward(torch.tensor([[3.0, 0.0]]))
        maybe_install_inplace_ttt_fast_weight_wrappers(runtime, runner)
        old_runtime = residual_runtime.RUNTIME
        try:
            residual_runtime.RUNTIME = runtime
            residual_runtime.configure_runtime(
                graph_scratch_rows=1,
                tap_layer_paths=(),
                source_layer_path="model.model.layers[0].input_layernorm",
                target_layer_path="model.model.layers[-1].input_layernorm",
                enable_esamp_training=False,
                distiller_hidden_dim=2,
                distiller_lr=1e-3,
                inplace_ttt_fast_weight_enabled=False,
            )
        finally:
            residual_runtime.RUNTIME = old_runtime
        restored_probe = mlp.forward(torch.tensor([[3.0, 0.0]]))

        self.assertFalse(runtime.config.inplace_ttt_fast_weight_enabled)
        self.assertEqual(runtime.inplace_ttt_fast_weight_handles, {})
        self.assertTrue(torch.equal(restored_probe, base_probe))


class InPlaceTTTFastWeightRunnerCachedModelSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("TLLM_INPLACE_TTT_RUN_CACHED_MODEL_SMOKE") == "1" and torch.cuda.is_available(),
        "set TLLM_INPLACE_TTT_RUN_CACHED_MODEL_SMOKE=1 and provide CUDA to run cached-model vLLM smoke",
    )
    def test_load_model_installs_qwen_mlp_wrapper_without_manual_model_patch(self) -> None:
        from vllm import LLM

        from tllm.runtime import residual_runtime
        from tllm.runtime.fast_weight_state import FastWeightKey, module_weight_view
        from tllm.runtime.inplace_ttt_fast_weight_runtime import restore_inplace_ttt_fast_weight_wrappers

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        residual_runtime.set_runtime_consumer(None)
        residual_runtime.clear_dispatch_consumers()
        residual_runtime.set_inplace_ttt_fast_weight_enabled(
            True,
            layers=(0,),
            chunk_size=4,
            learning_rate=0.5,
            fail_on_unsupported=True,
        )
        residual_runtime.install_runner_patch()
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="bfloat16",
            gpu_memory_utilization=0.45,
            max_model_len=64,
            enforce_eager=True,
            enable_prefix_caching=False,
            trust_remote_code=True,
        )
        runtime = residual_runtime.RUNTIME
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        mlp = runner.model.model.layers[0].mlp
        weight = module_weight_view(mlp.down_proj)
        hidden_size = int(weight.shape[0])
        torch.manual_seed(23)
        first = torch.randn((2, hidden_size), device=weight.device, dtype=weight.dtype)
        second = torch.randn((2, hidden_size), device=weight.device, dtype=weight.dtype)
        try:
            self.assertEqual(runtime.inplace_ttt_fast_weight_installed_count, 1)
            mlp.forward(first)
            mlp.forward(second)
            key = FastWeightKey(request_id="__tllm_inplace_ttt_static__", sample_idx=0, layer=0)
            stats = runtime.inplace_ttt_fast_weight_state.stats(key)
            self.assertGreater(stats.update_count, 0)
            self.assertEqual(stats.tail_rows, 0)
        finally:
            restore_inplace_ttt_fast_weight_wrappers(runtime)
            residual_runtime.set_inplace_ttt_fast_weight_enabled(False)

    def _run_request_sample_key_mode_cached_qwen_batch(self, batch_size: int) -> None:
        from vllm import LLM, SamplingParams

        from tllm.runtime import residual_runtime
        from tllm.runtime.inplace_ttt_fast_weight_runtime import restore_inplace_ttt_fast_weight_wrappers

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        residual_runtime.set_runtime_consumer(None)
        residual_runtime.clear_dispatch_consumers()
        residual_runtime.set_inplace_ttt_fast_weight_enabled(
            True,
            layers=(0,),
            chunk_size=2,
            learning_rate=0.5,
            key_mode="request_sample",
            max_slots=max(8, int(batch_size)),
            fail_on_unsupported=True,
        )
        residual_runtime.install_runner_patch()
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="bfloat16",
            gpu_memory_utilization=0.45,
            max_model_len=64,
            enforce_eager=True,
            enable_prefix_caching=False,
            trust_remote_code=True,
        )
        params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        prompts = [f"Q: {i}+{i}=\nA:" for i in range(int(batch_size))]
        try:
            outputs = llm.generate(prompts, params)
            runtime = residual_runtime.RUNTIME
            state = runtime.inplace_ttt_fast_weight_state
            stats = [state.stats(key) for key in list(state._stats)]

            self.assertEqual(len(outputs), int(batch_size))
            self.assertEqual(runtime.inplace_ttt_fast_weight_installed_count, 1)
            self.assertEqual(state.slot_count, int(batch_size))
            self.assertGreater(state.estimated_bytes, 0)
            self.assertEqual(sum(1 for item in stats if item.update_count > 0), int(batch_size))
            self.assertGreaterEqual(sum(item.trained_rows for item in stats), int(batch_size) * 8)
        finally:
            restore_inplace_ttt_fast_weight_wrappers(residual_runtime.RUNTIME)
            residual_runtime.set_inplace_ttt_fast_weight_enabled(False)

    @unittest.skipUnless(
        os.getenv("TLLM_INPLACE_TTT_RUN_REQUEST_SAMPLE_BATCH_SMOKE") == "1" and torch.cuda.is_available(),
        "set TLLM_INPLACE_TTT_RUN_REQUEST_SAMPLE_BATCH_SMOKE=1 and provide CUDA to run request-sample batch smoke",
    )
    def test_request_sample_key_mode_runs_cached_qwen_batch4(self) -> None:
        self._run_request_sample_key_mode_cached_qwen_batch(4)

    @unittest.skipUnless(
        os.getenv("TLLM_INPLACE_TTT_RUN_REQUEST_SAMPLE_BATCH_SMOKE") == "1" and torch.cuda.is_available(),
        "set TLLM_INPLACE_TTT_RUN_REQUEST_SAMPLE_BATCH_SMOKE=1 and provide CUDA to run request-sample batch smoke",
    )
    def test_request_sample_key_mode_runs_cached_qwen_batch8(self) -> None:
        self._run_request_sample_key_mode_cached_qwen_batch(8)


if __name__ == "__main__":
    unittest.main()
