#!/usr/bin/env python3
"""Official-shaped In-Place TTT parity tests for runtime fast weights."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class _ReferenceState:
    tail_x: torch.Tensor | None = None
    tail_t: torch.Tensor | None = None
    past_w: torch.Tensor | None = None


@dataclass(frozen=True)
class _ReferenceStep:
    output: torch.Tensor
    state: _ReferenceState
    activated_tail: torch.Tensor
    update_count: int
    trained_rows: int


class _UpstreamShapedToyMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 3,
        intermediate_size: int = 5,
        use_ttt_proj: bool = False,
        zero_ttt_conv: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=dtype)
        self.ttt_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=dtype) if use_ttt_proj else None
        self.ttt_conv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=5,
            padding=2,
            groups=self.hidden_size,
            bias=False,
            dtype=dtype,
        )
        self._init_weights(zero_ttt_conv=zero_ttt_conv, dtype=dtype)

    def _init_weights(self, *, zero_ttt_conv: bool, dtype: torch.dtype) -> None:
        with torch.no_grad():
            self.gate_proj.weight.copy_(
                torch.tensor(
                    [
                        [0.20, -0.10, 0.05],
                        [-0.30, 0.40, 0.10],
                        [0.15, 0.25, -0.20],
                        [0.05, -0.35, 0.45],
                        [0.30, 0.10, -0.25],
                    ],
                    dtype=dtype,
                )
            )
            self.up_proj.weight.copy_(
                torch.tensor(
                    [
                        [0.10, 0.25, -0.15],
                        [0.35, -0.05, 0.20],
                        [-0.25, 0.15, 0.30],
                        [0.40, 0.05, -0.10],
                        [-0.05, 0.30, 0.25],
                    ],
                    dtype=dtype,
                )
            )
            self.down_proj.weight.copy_(
                torch.tensor(
                    [
                        [0.20, -0.30, 0.10, 0.40, -0.20],
                        [-0.10, 0.25, 0.35, -0.15, 0.05],
                        [0.30, 0.10, -0.25, 0.20, 0.15],
                    ],
                    dtype=dtype,
                )
            )
            if self.ttt_proj is not None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(42)
                self.ttt_proj.weight.copy_(
                    torch.randn(
                        self.hidden_size,
                        self.hidden_size,
                        generator=generator,
                        dtype=dtype,
                    )
                    * 0.02
                )
            if zero_ttt_conv:
                self.ttt_conv.weight.zero_()
            else:
                self.ttt_conv.weight.copy_(
                    torch.tensor(
                        [
                            [[[0.05, -0.10, 0.20, 0.10, -0.05]]],
                            [[[-0.15, 0.05, 0.10, -0.05, 0.20]]],
                            [[[0.10, 0.15, -0.20, 0.05, 0.10]]],
                        ],
                        dtype=dtype,
                    ).reshape_as(self.ttt_conv.weight)
                )

    def activated(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x)) * self.up_proj(x)

    def _pad_and_chunk(self, rows: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
        chunk = max(1, int(chunk_size))
        remainder = int(rows.shape[0]) % chunk
        if remainder:
            padding = rows.new_zeros((chunk - remainder, int(rows.shape[1])))
            rows = torch.cat([rows, padding], dim=0)
        return rows.view(1, -1, chunk, int(rows.shape[1]))

    def official_step(
        self,
        *,
        x: torch.Tensor,
        target: torch.Tensor,
        state: _ReferenceState | None,
        chunk_size: int,
        learning_rate: float,
        use_effective_conv: bool,
    ) -> _ReferenceStep:
        current_state = state or _ReferenceState()
        current_w = (
            self.down_proj.weight.detach().clone()
            if current_state.past_w is None
            else current_state.past_w.to(device=x.device, dtype=x.dtype).clone()
        )
        if current_state.tail_x is None:
            present_x = x
            present_t = target
        else:
            present_x = torch.cat([current_state.tail_x.to(device=x.device, dtype=x.dtype), x], dim=0)
            present_t = torch.cat([current_state.tail_t.to(device=target.device, dtype=target.dtype), target], dim=0)

        chunk = max(1, int(chunk_size))
        update_count = 0
        trained_rows = 0
        if int(present_x.shape[0]) < chunk:
            output = self.activated(x).matmul(current_w.t())
        else:
            h_padded = self._pad_and_chunk(self.activated(present_x), chunk_size=chunk)
            t_padded = self._pad_and_chunk(present_t, chunk_size=chunk)
            conv_t = t_padded
            if use_effective_conv:
                batch_chunks, width, rows = (
                    int(t_padded.shape[0] * t_padded.shape[1]),
                    int(t_padded.shape[3]),
                    int(t_padded.shape[2]),
                )
                conv_t = (
                    self.ttt_conv(t_padded.transpose(-1, -2).reshape(batch_chunks, width, rows))
                    .transpose(-1, -2)
                    .reshape_as(t_padded)
                )
            chunk_outputs = torch.zeros_like(conv_t)
            chunk_num = int(h_padded.shape[1])
            for chunk_idx in range(chunk_num):
                current_h = h_padded[0, chunk_idx]
                current_t = conv_t[0, chunk_idx]
                chunk_outputs[0, chunk_idx] = current_h.matmul(current_w.t())
                is_full_chunk = int(present_x.shape[0]) % chunk == 0 or chunk_idx != chunk_num - 1
                if is_full_chunk:
                    if self.ttt_proj is None:
                        dw = torch.einsum("ch,cd->dh", current_h, current_t)
                    else:
                        dw = torch.einsum("ch,cd,de->eh", current_h, current_t, self.ttt_proj.weight)
                    current_w = current_w + (dw * float(learning_rate)).to(device=current_w.device, dtype=current_w.dtype)
                    update_count += 1
                    trained_rows += chunk
            all_output = chunk_outputs.reshape(1, -1, int(self.hidden_size))[:, : int(present_x.shape[0]), :][0]
            output = all_output[-int(x.shape[0]) :]

        tail_rows = int(present_x.shape[0]) % chunk
        tail_x = present_x[-tail_rows:].detach().clone() if tail_rows else None
        tail_t = present_t[-tail_rows:].detach().clone() if tail_rows else None
        activated_tail = self.activated(tail_x) if tail_x is not None else self.activated(present_x)[:0]
        return _ReferenceStep(
            output=output,
            state=_ReferenceState(tail_x=tail_x, tail_t=tail_t, past_w=current_w.detach().clone()),
            activated_tail=activated_tail.detach().clone(),
            update_count=update_count,
            trained_rows=trained_rows,
        )


class InPlaceTTTOfficialParityUnitTest(unittest.TestCase):
    def _run_tllm_step(
        self,
        *,
        toy: _UpstreamShapedToyMLP,
        x: torch.Tensor,
        target: torch.Tensor,
        state: object,
        key: object,
        chunk_size: int,
        learning_rate: float,
        use_effective_conv: bool,
    ):
        from tllm.runtime.fast_weight.state import run_down_proj_fast_weight_step

        return run_down_proj_fast_weight_step(
            hidden=toy.activated(x),
            target=target,
            base_weight=toy.down_proj.weight,
            state=state,
            key=key,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
            ttt_proj_weight=None if toy.ttt_proj is None else toy.ttt_proj.weight,
            ttt_conv_weight=toy.ttt_conv.weight if use_effective_conv else None,
        )

    def _assert_step_matches_reference(
        self,
        *,
        toy: _UpstreamShapedToyMLP,
        x: torch.Tensor,
        target: torch.Tensor,
        ref_state: _ReferenceState | None,
        tllm_state: object,
        key: object,
        chunk_size: int,
        learning_rate: float,
        use_effective_conv: bool,
    ) -> _ReferenceStep:
        ref_step = toy.official_step(
            x=x,
            target=target,
            state=ref_state,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
            use_effective_conv=use_effective_conv,
        )
        tllm_step = self._run_tllm_step(
            toy=toy,
            x=x,
            target=target,
            state=tllm_state,
            key=key,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
            use_effective_conv=use_effective_conv,
        )
        slot = tllm_state.read(key)
        self.assertTrue(torch.allclose(tllm_step.output, ref_step.output, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(slot.past_w, ref_step.state.past_w, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(slot.past_h, ref_step.activated_tail, atol=1e-12, rtol=1e-12))
        expected_tail_t = ref_step.state.tail_t if ref_step.state.tail_t is not None else target[:0]
        self.assertTrue(torch.allclose(slot.past_t, expected_tail_t, atol=1e-12, rtol=1e-12))
        self.assertEqual(tllm_step.update_count, ref_step.update_count)
        self.assertEqual(tllm_step.trained_rows, ref_step.trained_rows)
        return ref_step

    def test_no_projection_without_effective_conv_matches_upstream_shape(self) -> None:
        from tllm.runtime.fast_weight.state import FastWeightKey, FastWeightState

        toy = _UpstreamShapedToyMLP(use_ttt_proj=False)
        state = FastWeightState()
        key = FastWeightKey(request_id="req", sample_idx=0, layer=0)
        x = torch.tensor(
            [
                [0.25, -0.50, 0.75],
                [1.00, 0.20, -0.10],
                [-0.30, 0.40, 0.60],
                [0.70, -0.20, 0.30],
            ],
            dtype=torch.float64,
        )
        target = torch.tensor(
            [
                [0.10, -0.20, 0.30],
                [0.35, 0.25, -0.15],
                [-0.45, 0.15, 0.20],
                [0.05, -0.35, 0.40],
            ],
            dtype=torch.float64,
        )

        self._assert_step_matches_reference(
            toy=toy,
            x=x,
            target=target,
            ref_state=None,
            tllm_state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.3,
            use_effective_conv=False,
        )

    def test_official_random_projection_and_zero_conv_matches_no_weight_change(self) -> None:
        from tllm.runtime.fast_weight.state import FastWeightKey, FastWeightState

        toy = _UpstreamShapedToyMLP(use_ttt_proj=True, zero_ttt_conv=True)
        state = FastWeightState()
        key = FastWeightKey(request_id="req", sample_idx=0, layer=0)
        x = torch.tensor(
            [
                [0.30, -0.10, 0.20],
                [-0.25, 0.60, 0.15],
                [0.45, 0.35, -0.40],
                [0.10, -0.55, 0.50],
            ],
            dtype=torch.float64,
        )
        target = torch.tensor(
            [
                [0.25, -0.15, 0.45],
                [0.05, 0.30, -0.25],
                [-0.35, 0.40, 0.10],
                [0.50, -0.20, -0.10],
            ],
            dtype=torch.float64,
        )

        ref_step = self._assert_step_matches_reference(
            toy=toy,
            x=x,
            target=target,
            ref_state=None,
            tllm_state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.3,
            use_effective_conv=True,
        )

        self.assertTrue(torch.allclose(ref_step.state.past_w, toy.down_proj.weight, atol=1e-12, rtol=1e-12))

    def test_nonzero_conv_ablation_matches_upstream_projection_update(self) -> None:
        from tllm.runtime.fast_weight.state import FastWeightKey, FastWeightState

        toy = _UpstreamShapedToyMLP(use_ttt_proj=True, zero_ttt_conv=False)
        state = FastWeightState()
        key = FastWeightKey(request_id="req", sample_idx=0, layer=0)
        x = torch.tensor(
            [
                [0.20, 0.10, -0.30],
                [0.40, -0.20, 0.50],
                [-0.10, 0.30, 0.25],
                [0.60, -0.45, 0.15],
            ],
            dtype=torch.float64,
        )
        target = torch.tensor(
            [
                [0.30, -0.25, 0.15],
                [-0.05, 0.45, 0.35],
                [0.20, 0.10, -0.40],
                [0.55, -0.15, 0.05],
            ],
            dtype=torch.float64,
        )

        ref_step = self._assert_step_matches_reference(
            toy=toy,
            x=x,
            target=target,
            ref_state=None,
            tllm_state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.3,
            use_effective_conv=True,
        )

        self.assertFalse(torch.allclose(ref_step.state.past_w, toy.down_proj.weight, atol=1e-12, rtol=1e-12))

    def test_below_chunk_rows_only_update_tail_cache(self) -> None:
        from tllm.runtime.fast_weight.state import FastWeightKey, FastWeightState

        toy = _UpstreamShapedToyMLP(use_ttt_proj=False)
        state = FastWeightState()
        key = FastWeightKey(request_id="req", sample_idx=0, layer=0)
        x = torch.tensor([[0.15, -0.20, 0.35], [0.45, 0.10, -0.25], [-0.30, 0.55, 0.20]], dtype=torch.float64)
        target = torch.tensor([[0.10, 0.20, -0.15], [0.35, -0.30, 0.25], [-0.05, 0.40, 0.30]], dtype=torch.float64)

        ref_step = self._assert_step_matches_reference(
            toy=toy,
            x=x,
            target=target,
            ref_state=None,
            tllm_state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.3,
            use_effective_conv=False,
        )

        self.assertEqual(ref_step.update_count, 0)
        self.assertTrue(torch.allclose(ref_step.state.past_w, toy.down_proj.weight, atol=1e-12, rtol=1e-12))
        self.assertEqual(int(state.read(key).past_h.shape[0]), 3)

    def test_tail_plus_next_rows_form_chunk_and_leave_new_tail(self) -> None:
        from tllm.runtime.fast_weight.state import FastWeightKey, FastWeightState

        toy = _UpstreamShapedToyMLP(use_ttt_proj=False)
        state = FastWeightState()
        key = FastWeightKey(request_id="req", sample_idx=0, layer=0)
        first_x = torch.tensor([[0.10, -0.35, 0.25], [0.30, 0.45, -0.15]], dtype=torch.float64)
        first_t = torch.tensor([[0.25, -0.10, 0.15], [-0.20, 0.35, 0.30]], dtype=torch.float64)
        first_ref = self._assert_step_matches_reference(
            toy=toy,
            x=first_x,
            target=first_t,
            ref_state=None,
            tllm_state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.3,
            use_effective_conv=False,
        )
        second_x = torch.tensor([[0.50, -0.05, 0.40], [-0.15, 0.20, 0.55], [0.35, 0.10, -0.30]], dtype=torch.float64)
        second_t = torch.tensor([[0.05, 0.45, -0.25], [0.40, -0.20, 0.10], [-0.30, 0.15, 0.35]], dtype=torch.float64)

        second_ref = self._assert_step_matches_reference(
            toy=toy,
            x=second_x,
            target=second_t,
            ref_state=first_ref.state,
            tllm_state=state,
            key=key,
            chunk_size=4,
            learning_rate=0.3,
            use_effective_conv=False,
        )

        slot = state.read(key)
        self.assertEqual(second_ref.update_count, 1)
        self.assertEqual(int(slot.past_h.shape[0]), 1)
        self.assertEqual(int(slot.past_t.shape[0]), 1)
        self.assertFalse(torch.allclose(slot.past_w, toy.down_proj.weight, atol=1e-12, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
