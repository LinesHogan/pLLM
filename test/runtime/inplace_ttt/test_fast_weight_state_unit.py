#!/usr/bin/env python3
"""Unit tests for In-Place TTT fast-weight runtime state."""

from __future__ import annotations

import unittest

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


class FastWeightStateUnitTest(unittest.TestCase):
    def test_fast_weight_state_stores_tail_and_weight_clone_safely(self) -> None:
        from tllm.runtime.fast_weight.state import FastWeightKey, FastWeightState

        state = FastWeightState()
        key = FastWeightKey(request_id="req-a", sample_idx=2, layer=3)
        h = torch.tensor([[1.0, 2.0]])
        t = torch.tensor([[0.5, -0.5]])
        w = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

        state.write(key=key, past_h=h, past_t=t, past_w=w)
        self.assertEqual(state.keys(), (key,))
        self.assertEqual(state.bank.keys(), (key,))
        self.assertEqual(state.bank.get_slot(key).version, 1)
        h.zero_()
        t.zero_()
        w.zero_()

        slot = state.read(key)

        self.assertTrue(torch.equal(slot.past_h, torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.equal(slot.past_t, torch.tensor([[0.5, -0.5]])))
        self.assertTrue(torch.equal(slot.past_w, torch.tensor([[0.1, 0.2], [0.3, 0.4]])))

    def test_mlp_wrapper_updates_when_cached_tail_reaches_full_chunk(self) -> None:
        from tllm.runtime.fast_weight.state import (
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

    def test_request_keyed_rows_keep_tail_weight_and_stats_isolated(self) -> None:
        from tllm.runtime.fast_weight.state import (
            FastWeightKey,
            FastWeightState,
            install_mlp_fast_weight_wrapper,
        )

        mlp = _FakeQwenLikeMLP()
        mlp.down_proj.weight.zero_()
        state = FastWeightState()
        req_a = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        req_b = FastWeightKey(request_id="req-b", sample_idx=1, layer=0)
        row_keys = (req_a, req_a, req_b, req_b)
        handle = install_mlp_fast_weight_wrapper(
            module=mlp,
            state=state,
            key=req_a,
            row_key_provider=lambda rows: row_keys if rows == 4 else None,
            chunk_size=3,
            learning_rate=0.5,
            require_single_tp=False,
        )

        mlp.forward(torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]))

        slot_a = state.read(req_a)
        slot_b = state.read(req_b)
        self.assertEqual(int(slot_a.past_h.shape[0]), 2)
        self.assertEqual(int(slot_b.past_h.shape[0]), 2)
        self.assertTrue(torch.equal(slot_a.past_h[:, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(slot_b.past_h[:, 0], torch.zeros(2)))
        self.assertEqual(state.stats(req_a).tail_rows, 2)
        self.assertEqual(state.stats(req_b).tail_rows, 2)

        state.clear(req_a)
        self.assertIsNone(state.get(req_a))
        self.assertIsNotNone(state.get(req_b))
        self.assertEqual(state.stats(req_b).tail_rows, 2)
        handle.restore()

    def test_mixed_packed_step_updates_prefill_without_updating_decode_request(self) -> None:
        from tllm.runtime.fast_weight.state import (
            FastWeightKey,
            FastWeightState,
            install_mlp_fast_weight_wrapper,
        )

        mlp = _FakeQwenLikeMLP()
        mlp.down_proj.weight.zero_()
        state = FastWeightState()
        req_a = FastWeightKey(request_id="req-a", sample_idx=0, layer=0)
        req_b = FastWeightKey(request_id="req-b", sample_idx=0, layer=0)
        handle = install_mlp_fast_weight_wrapper(
            module=mlp,
            state=state,
            key=req_a,
            row_key_provider=lambda rows: (req_a, req_a, req_b, req_b),
            row_phase_provider=lambda rows: ("prefill", "prefill", "decode", "decode"),
            chunk_size=2,
            learning_rate=0.5,
            decode_update_enabled=False,
            require_single_tp=False,
        )

        output = mlp.forward(
            torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        )

        self.assertEqual(state.stats(req_a).update_count, 1)
        self.assertEqual(state.stats(req_b).update_count, 0)
        self.assertIsNotNone(state.get(req_a))
        self.assertIsNone(state.get(req_b))
        self.assertTrue(torch.equal(output[2:], torch.zeros_like(output[2:])))
        handle.restore()

    def test_request_keyed_wrapper_fails_when_metadata_is_missing(self) -> None:
        from tllm.runtime.fast_weight.state import (
            FastWeightKey,
            FastWeightState,
            install_mlp_fast_weight_wrapper,
        )

        mlp = _FakeQwenLikeMLP()
        handle = install_mlp_fast_weight_wrapper(
            module=mlp,
            state=FastWeightState(),
            key=FastWeightKey(request_id="req-a", sample_idx=0, layer=0),
            row_key_provider=lambda rows: None,
            chunk_size=2,
            learning_rate=0.5,
            require_single_tp=False,
        )
        with self.assertRaisesRegex(RuntimeError, "requires packed row metadata"):
            mlp.forward(torch.ones(1, 2))
        handle.restore()


if __name__ == "__main__":
    unittest.main()
