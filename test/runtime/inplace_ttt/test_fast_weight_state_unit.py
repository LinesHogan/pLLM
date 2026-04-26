#!/usr/bin/env python3
"""Unit tests for In-Place TTT fast-weight runtime state."""

from __future__ import annotations

import os
import unittest

import torch


class _FakeDownProj:
    def __init__(self) -> None:
        self.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.bias = None

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(hidden, self.weight), None


class _FakeBiasedDownProj(_FakeDownProj):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.tensor([0.25, -0.25])

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.linear(hidden, self.weight, self.bias), self.bias


class _FakeGateUpProj:
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden, None


class _FakeQwenLikeMLP:
    def __init__(self) -> None:
        self.gate_up_proj = _FakeGateUpProj()
        self.act_fn = torch.nn.Identity()
        self.down_proj = _FakeDownProj()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj.forward(hidden)
        activated = self.act_fn(gate_up)
        output, _ = self.down_proj.forward(activated)
        return output


class _FakeBiasedQwenLikeMLP(_FakeQwenLikeMLP):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = _FakeBiasedDownProj()


class FastWeightStateUnitTest(unittest.TestCase):
    def test_fast_weight_key_rejects_implicit_identity_coercion(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey

        with self.assertRaisesRegex(TypeError, "request_id must be str"):
            FastWeightKey(request_id=1, sample_idx=0, layer=0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "sample_idx must be int"):
            FastWeightKey(request_id="req-a", sample_idx=True, layer=0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "target_module must be non-empty"):
            FastWeightKey(request_id="req-a", sample_idx=0, layer=0, target_module="")

    def test_fast_weight_state_stores_tail_and_weight_clone_safely(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey, FastWeightState

        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=2, layer=3)
        h = torch.tensor([[1.0, 2.0]])
        t = torch.tensor([[0.5, -0.5]])
        w = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

        state.write(key=key, past_h=h, past_t=t, past_w=w)
        h.zero_()
        t.zero_()
        w.zero_()

        slot = state.read(key)

        self.assertTrue(torch.equal(slot.past_h, torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.equal(slot.past_t, torch.tensor([[0.5, -0.5]])))
        self.assertTrue(torch.equal(slot.past_w, torch.tensor([[0.1, 0.2], [0.3, 0.4]])))

    def test_fast_weight_state_rejects_shape_mismatch(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey, FastWeightState

        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)

        with self.assertRaisesRegex(RuntimeError, "past_h and past_t must have matching row count"):
            state.write(key=key, past_h=torch.zeros(1, 2), past_t=torch.zeros(2, 2), past_w=torch.zeros(2, 2))
        with self.assertRaisesRegex(RuntimeError, "past_h width must match"):
            state.write(key=key, past_h=torch.zeros(1, 3), past_t=torch.zeros(1, 2), past_w=torch.zeros(2, 2))
        with self.assertRaisesRegex(RuntimeError, "past_t width must match"):
            state.write(key=key, past_h=torch.zeros(1, 2), past_t=torch.zeros(1, 3), past_w=torch.zeros(2, 2))
        with self.assertRaisesRegex(RuntimeError, "past_w must be 2D"):
            state.write(key=key, past_h=torch.zeros(1, 2), past_t=torch.zeros(1, 2), past_w=torch.zeros(2))

    def test_apply_down_proj_fast_weight_uses_effective_past_weight(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey, FastWeightState, apply_down_proj_fast_weight

        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        hidden = torch.tensor([[2.0, 3.0]])
        base_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        effective_weight = torch.tensor([[1.5, 0.0], [0.0, 0.5]])
        state.write(key=key, past_h=torch.zeros(1, 2), past_t=torch.zeros(1, 2), past_w=effective_weight)

        projected = apply_down_proj_fast_weight(hidden=hidden, base_weight=base_weight, state=state, key=key)

        self.assertTrue(torch.equal(projected, torch.tensor([[3.0, 1.5]])))

    def test_apply_down_proj_without_slot_uses_base_weight(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey, FastWeightState, apply_down_proj_fast_weight

        projected = apply_down_proj_fast_weight(
            hidden=torch.tensor([[2.0, 3.0]]),
            base_weight=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            state=FastWeightState(),
            key=FastWeightKey(request_id="req-a", sample_idx=0, layer=0),
        )

        self.assertTrue(torch.equal(projected, torch.tensor([[2.0, 3.0]])))

    def test_down_proj_overlay_wrapper_changes_output_and_restores_forward(self) -> None:
        from tllm.runtime.fast_weight_state import (
            FastWeightKey,
            FastWeightState,
            install_down_proj_fast_weight_overlay,
        )

        module = _FakeDownProj()
        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        hidden = torch.tensor([[2.0, 3.0]])

        handle = install_down_proj_fast_weight_overlay(
            module=module,
            state=state,
            key=key,
            require_single_tp=False,
        )
        base, _ = module.forward(hidden)
        self.assertTrue(torch.equal(base, torch.tensor([[2.0, 3.0]])))

        state.write(
            key=key,
            past_h=torch.zeros(1, 2),
            past_t=torch.zeros(1, 2),
            past_w=torch.tensor([[1.5, 0.0], [0.0, 0.5]]),
        )
        overlaid, _ = module.forward(hidden)
        self.assertTrue(torch.equal(overlaid, torch.tensor([[3.0, 1.5]])))

        handle.restore()
        restored, _ = module.forward(hidden)
        self.assertTrue(torch.equal(restored, torch.tensor([[2.0, 3.0]])))

    def test_down_proj_overlay_rejects_shape_mismatch(self) -> None:
        from tllm.runtime.fast_weight_state import (
            FastWeightKey,
            FastWeightState,
            install_down_proj_fast_weight_overlay,
        )

        module = _FakeDownProj()
        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        handle = install_down_proj_fast_weight_overlay(
            module=module,
            state=state,
            key=key,
            require_single_tp=False,
        )
        state.write(
            key=key,
            past_h=torch.zeros(1, 3),
            past_t=torch.zeros(1, 3),
            past_w=torch.zeros(3, 3),
        )

        with self.assertRaisesRegex(RuntimeError, "overlay shape must match"):
            module.forward(torch.zeros(1, 2))
        handle.restore()

    def test_mlp_wrapper_caches_prefill_tail_without_update(self) -> None:
        from tllm.runtime.fast_weight_state import (
            FastWeightKey,
            FastWeightState,
            install_mlp_fast_weight_wrapper,
        )

        mlp = _FakeQwenLikeMLP()
        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        handle = install_mlp_fast_weight_wrapper(
            module=mlp,
            state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.5,
            require_single_tp=False,
        )
        hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        output = mlp.forward(hidden)
        slot = state.read(key)
        stats = state.stats(key)

        self.assertTrue(torch.equal(output, hidden))
        self.assertEqual(int(slot.past_h.shape[0]), 2)
        self.assertTrue(torch.equal(slot.past_w, mlp.down_proj.weight))
        self.assertEqual(stats.update_count, 0)
        self.assertEqual(stats.trained_rows, 0)
        self.assertEqual(stats.tail_rows, 2)
        handle.restore()

    def test_mlp_wrapper_updates_when_cached_tail_reaches_full_chunk(self) -> None:
        from tllm.runtime.fast_weight_state import (
            FastWeightKey,
            FastWeightState,
            install_mlp_fast_weight_wrapper,
        )

        mlp = _FakeQwenLikeMLP()
        mlp.down_proj.weight.zero_()
        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        handle = install_mlp_fast_weight_wrapper(
            module=mlp,
            state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.5,
            require_single_tp=False,
        )

        first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        second = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        mlp.forward(first)
        output = mlp.forward(second)
        slot = state.read(key)
        stats = state.stats(key)

        self.assertEqual(tuple(output.shape), tuple(second.shape))
        self.assertEqual(int(slot.past_h.shape[0]), 0)
        self.assertFalse(torch.equal(slot.past_w, mlp.down_proj.weight))
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.trained_rows, 4)
        self.assertEqual(stats.tail_rows, 0)
        handle.restore()

    def test_mlp_wrapper_decode_default_applies_without_training_or_tail_cache(self) -> None:
        from tllm.runtime.fast_weight_state import (
            FastWeightKey,
            FastWeightState,
            install_mlp_fast_weight_wrapper,
        )

        phase = "prefill"
        mlp = _FakeQwenLikeMLP()
        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        handle = install_mlp_fast_weight_wrapper(
            module=mlp,
            state=state,
            key=key,
            chunk_size=2,
            learning_rate=0.5,
            phase_getter=lambda: phase,
            require_single_tp=False,
        )

        mlp.forward(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        trained = state.stats(key)
        before_decode = state.read(key)
        phase = "decode"
        output = mlp.forward(torch.tensor([[3.0, 0.0]]))
        after_decode = state.read(key)
        decoded = state.stats(key)

        expected = torch.nn.functional.linear(torch.tensor([[3.0, 0.0]]), before_decode.past_w)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(decoded.update_count, trained.update_count)
        self.assertEqual(decoded.trained_rows, trained.trained_rows)
        self.assertTrue(torch.equal(after_decode.past_h, before_decode.past_h))
        self.assertTrue(torch.equal(after_decode.past_t, before_decode.past_t))
        self.assertTrue(torch.equal(after_decode.past_w, before_decode.past_w))
        self.assertGreater(decoded.apply_count, trained.apply_count)
        handle.restore()

    def test_mlp_wrapper_rejects_biased_down_proj_in_limited_scope(self) -> None:
        from tllm.runtime.fast_weight_state import (
            FastWeightKey,
            FastWeightState,
            install_mlp_fast_weight_wrapper,
        )

        with self.assertRaisesRegex(RuntimeError, "requires biasless down_proj"):
            install_mlp_fast_weight_wrapper(
                module=_FakeBiasedQwenLikeMLP(),
                state=FastWeightState(),
                key=FastWeightKey(request_id="req-a", sample_idx=0, layer=0),
                chunk_size=2,
                learning_rate=0.5,
                require_single_tp=False,
            )


class InPlaceTTTCachedModelSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("TLLM_INPLACE_TTT_RUN_CACHED_MODEL_SMOKE") == "1" and torch.cuda.is_available(),
        "set TLLM_INPLACE_TTT_RUN_CACHED_MODEL_SMOKE=1 and provide CUDA to run cached-model vLLM smoke",
    )
    def test_down_proj_effective_weight_overlay_changes_qwen_generation(self) -> None:
        from vllm import LLM, SamplingParams

        from tllm.runtime import residual_runtime
        from tllm.runtime.fast_weight_state import (
            FastWeightKey,
            FastWeightState,
            install_down_proj_fast_weight_overlay,
            module_weight_view,
        )

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        mlp = runner.model.model.layers[0].mlp
        down_proj = mlp.down_proj
        base_weight = module_weight_view(down_proj)
        state = FastWeightState()
        key = FastWeightKey(request_id="smoke", sample_idx=0, layer=0)
        params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
        prompt = "Q: 1+1=\nA:"

        base = llm.generate([prompt], params)[0].outputs[0]
        torch.manual_seed(11)
        effective_weight = torch.randn_like(base_weight, dtype=torch.float32).to(base_weight.device, dtype=base_weight.dtype) * 0.25
        state.write(
            key=key,
            past_h=torch.zeros(1, int(base_weight.shape[1])),
            past_t=torch.zeros(1, int(base_weight.shape[0])),
            past_w=effective_weight,
        )
        handle = install_down_proj_fast_weight_overlay(
            module=down_proj,
            state=state,
            key=key,
            require_single_tp=True,
        )
        try:
            adapted = llm.generate([prompt], params)[0].outputs[0]
        finally:
            handle.restore()

        self.assertNotEqual(list(base.token_ids), list(adapted.token_ids))

    @unittest.skipUnless(
        os.getenv("TLLM_INPLACE_TTT_RUN_CACHED_MODEL_SMOKE") == "1" and torch.cuda.is_available(),
        "set TLLM_INPLACE_TTT_RUN_CACHED_MODEL_SMOKE=1 and provide CUDA to run cached-model vLLM smoke",
    )
    def test_qwen_mlp_wrapper_tracks_tail_updates_and_changes_output(self) -> None:
        from vllm import LLM

        from tllm.runtime import residual_runtime
        from tllm.runtime.fast_weight_state import (
            FastWeightKey,
            FastWeightState,
            install_mlp_fast_weight_wrapper,
            module_weight_view,
        )

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        mlp = runner.model.model.layers[0].mlp
        weight = module_weight_view(mlp.down_proj)
        hidden_size = int(weight.shape[0])
        state = FastWeightState()
        key = FastWeightKey(request_id="smoke", sample_idx=0, layer=0)
        handle = install_mlp_fast_weight_wrapper(
            module=mlp,
            state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.5,
            require_single_tp=True,
        )
        torch.manual_seed(17)
        first = torch.randn((2, hidden_size), device=weight.device, dtype=weight.dtype)
        second = torch.randn((2, hidden_size), device=weight.device, dtype=weight.dtype)
        probe = torch.randn((1, hidden_size), device=weight.device, dtype=weight.dtype)
        try:
            mlp.forward(first)
            tail_stats = state.stats(key)
            self.assertEqual(tail_stats.update_count, 0)
            self.assertEqual(tail_stats.tail_rows, 2)

            mlp.forward(second)
            update_stats = state.stats(key)
            self.assertGreater(update_stats.update_count, 0)
            self.assertEqual(update_stats.tail_rows, 0)
            adapted_probe = mlp.forward(probe)
        finally:
            handle.restore()
        base_probe = mlp.forward(probe)

        self.assertFalse(torch.allclose(adapted_probe, base_probe))


if __name__ == "__main__":
    unittest.main()
