#!/usr/bin/env python3
"""Unit tests for In-Place TTT fast-weight runtime state."""

from __future__ import annotations

import unittest

import torch


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

        with self.assertRaisesRegex(RuntimeError, "past_h and past_t must have matching shape"):
            state.write(key=key, past_h=torch.zeros(1, 2), past_t=torch.zeros(2, 2), past_w=torch.zeros(2, 2))
        with self.assertRaisesRegex(RuntimeError, "past_w must be 2D"):
            state.write(key=key, past_h=torch.zeros(1, 2), past_t=torch.zeros(1, 2), past_w=torch.zeros(2))

    def test_apply_down_proj_fast_weight_overlay_changes_projection(self) -> None:
        from tllm.runtime.fast_weight_state import FastWeightKey, FastWeightState, apply_down_proj_fast_weight

        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        hidden = torch.tensor([[2.0, 3.0]])
        base_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        delta = torch.tensor([[0.5, 0.0], [0.0, -0.5]])
        state.write(key=key, past_h=torch.zeros(1, 2), past_t=torch.zeros(1, 2), past_w=delta)

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


if __name__ == "__main__":
    unittest.main()
