#!/usr/bin/env python3
"""Unit tests for the staged residual write channel."""

from __future__ import annotations

import unittest

import torch

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, WriteMode, WriteRows
from tllm.ports.residual_stream import ResidualStream
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports import residual_write
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey


def _lease(*, mode: WriteMode = WriteMode.REPLACE, order: int = 0, writer: str = "w0") -> ResidualWriteLease:
    return ResidualWriteLease(
        key=WriteLeaseKey(layer_path="layers.0", writer_id=writer, order=order),
        mode=mode,
        rows=WriteRows.DECODE_ONLY,
        max_rows=4,
        hidden_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class _WriterConsumer(BaseConsumer):
    def __init__(self, consumer_id: str, *, order: int, mode: str = "replace") -> None:
        self._consumer_id = consumer_id
        self._order = order
        self._mode = mode

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def flows(self):
        return (
            ConsumerFlow(
                reads=(),
                writes=(
                    ResidualStream.write(
                        layer=0,
                        site="block_output",
                        phase="decode",
                        mode=self._mode,
                        order=self._order,
                    ),
                ),
                window="out_of_band",
            ),
        )


class ResidualWriteChannelUnitTest(unittest.TestCase):
    def test_replace_applies_then_clears(self) -> None:
        lease = _lease(mode=WriteMode.REPLACE)
        lease.stage(delta=torch.full((2, 3), 9.0), row_idx=torch.tensor([0, 2]))
        tensor = torch.zeros((4, 3))
        residual_write.apply_pending_writes({lease.key: lease}, "layers.0", tensor)
        self.assertTrue(torch.equal(tensor[0], torch.full((3,), 9.0)))
        self.assertTrue(torch.equal(tensor[2], torch.full((3,), 9.0)))
        self.assertTrue(torch.equal(tensor[1], torch.zeros(3)))
        self.assertEqual(lease.count, 0)

    def test_add_accumulates(self) -> None:
        lease = _lease(mode=WriteMode.ADD)
        lease.stage(delta=torch.ones((1, 3)), row_idx=torch.tensor([1]))
        tensor = torch.full((4, 3), 5.0)
        residual_write.apply_pending_writes({lease.key: lease}, "layers.0", tensor)
        self.assertTrue(torch.equal(tensor[1], torch.full((3,), 6.0)))

    def test_apply_does_not_depend_on_host_count(self) -> None:
        lease = _lease(mode=WriteMode.REPLACE)
        lease.stage(delta=torch.full((1, 3), 7.0), row_idx=torch.tensor([2]))
        # CUDA graph replay does not re-enter Python. A captured write must be
        # controlled by address-stable device state, not this host integer.
        lease.count = 0
        tensor = torch.zeros((4, 3))

        residual_write.apply_pending_writes({lease.key: lease}, "layers.0", tensor)

        self.assertTrue(torch.equal(tensor[2], torch.full((3,), 7.0)))

    def test_same_locator_same_order_conflicts(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting residual writes"):
            DispatchPlan.build([_WriterConsumer("c0", order=0), _WriterConsumer("c1", order=0)])

    def test_wrapper_mode_has_no_lease(self) -> None:
        with self.assertRaisesRegex(ValueError, "replace/add"):
            _lease(mode=WriteMode.WRAPPER)


if __name__ == "__main__":
    unittest.main()
