#!/usr/bin/env python3
"""CUDA-graph captured train step tests (plan Phase 5).

The capture path is CUDA-only; these tests skip on CPU. They verify that a
captured step matches the eager functional transition at capture time (the
runtime also self-checks with PG146), that repeated replays track a
hand-rolled AdamW reference over multiple steps, and that varying batch rows
capture distinct graphs that share one slot.
"""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.consumers.esamp.program_consumer import ESampProgramConsumer

_HIDDEN = 64
_RANK = 16
_LR = 1e-2


def _bundle(source: torch.Tensor, target: torch.Tensor, step_id: int) -> PortBundle:
    rows = int(source.shape[0])
    return PortBundle(
        key=BundleKey(
            engine_step_id=step_id, phase="decode", request_id="req", sample_idx=0
        ),
        entries={
            "device_lease": DeviceTensorLease(
                entries={"source": source, "target": target}, active_rows=rows
            ),
            "request_meta": RowBatchMeta(
                request_ids=tuple(f"req-{i}" for i in range(rows)),
                prompt_idxs=tuple(0 for _ in range(rows)),
                sample_idxs=tuple(0 for _ in range(rows)),
                phase="decode",
                engine_step_id=step_id,
            ),
        },
    )


@unittest.skipUnless(torch.cuda.is_available(), "capture path is CUDA-only")
class CapturedTrainStepTest(unittest.TestCase):
    def _consumer(self) -> ESampProgramConsumer:
        return ESampProgramConsumer(
            ESampConsumerConfig(distiller_hidden_dim=_RANK, distiller_lr=_LR)
        )

    def test_capture_then_replay(self) -> None:
        device = torch.device("cuda:0")
        consumer = self._consumer()
        torch.manual_seed(0)
        for step in range(1, 6):
            source = torch.randn(8, _HIDDEN, device=device, dtype=torch.bfloat16)
            target = torch.randn(8, _HIDDEN, device=device, dtype=torch.bfloat16)
            consumer.consume_flow(0, _bundle(source, target, step), None)
        compiled = consumer._compiled
        assert compiled is not None
        self.assertEqual(compiled.graph_capture_count, 1)
        self.assertEqual(compiled.graph_replay_count, 4)
        self.assertEqual(compiled.loss_count, 5)
        self.assertEqual(compiled.update_count, 5)
        slot = compiled.bank("distiller").slot_for(None)
        self.assertEqual(slot.step, 5)

    def test_replays_track_adamw_reference(self) -> None:
        device = torch.device("cuda:0")
        dtype = torch.float32
        consumer = self._consumer()
        batches = [
            (
                torch.randn(8, _HIDDEN, device=device, dtype=dtype),
                torch.randn(8, _HIDDEN, device=device, dtype=dtype),
            )
            for _ in range(4)
        ]
        # Seed-replay: the program model is built lazily at the first consume.
        torch.manual_seed(0)
        consumer.consume_flow(0, _bundle(*batches[0], 1), None)
        torch.manual_seed(0)
        reference = LowRankGatedResidualModel(
            hidden_size=_HIDDEN,
            rank=_RANK,
            use_output_layernorm=True,
            device=device,
            dtype=dtype,
        )
        optimizer = torch.optim.AdamW(reference.parameters(), lr=_LR, foreach=False)
        import torch.nn.functional as F

        source0, target0 = batches[0]
        pred = reference(source0)
        loss = F.mse_loss(pred.float(), target0.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for step, (source, target) in enumerate(batches[1:], start=2):
            consumer.consume_flow(0, _bundle(source, target, step), None)
            pred = reference(source)
            loss = F.mse_loss(pred.float(), target.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        compiled = consumer._compiled
        assert compiled is not None
        # The captured updates ran on the training lane; sync before reading.
        torch.cuda.synchronize()
        slot = compiled.bank("distiller").slot_for(None)
        reference_state = reference.state_dict()
        for name, param in slot.params.items():
            reference_param = reference_state[name.removeprefix("model.")]
            self.assertTrue(
                torch.allclose(param, reference_param, rtol=2e-2, atol=2e-3),
                f"parameter {name} diverged from the AdamW reference after "
                f"{len(batches)} captured steps: "
                f"max abs diff {(param - reference_param).abs().max().item()}",
            )

    def test_varying_rows_capture_distinct_graphs(self) -> None:
        device = torch.device("cuda:0")
        consumer = self._consumer()
        torch.manual_seed(0)
        for step, rows in enumerate((8, 8, 4, 8, 4), start=1):
            source = torch.randn(rows, _HIDDEN, device=device, dtype=torch.bfloat16)
            target = torch.randn(rows, _HIDDEN, device=device, dtype=torch.bfloat16)
            consumer.consume_flow(0, _bundle(source, target, step), None)
        compiled = consumer._compiled
        assert compiled is not None
        self.assertEqual(compiled.graph_capture_count, 2)
        self.assertEqual(compiled.graph_replay_count, 3)
        slot = compiled.bank("distiller").slot_for(None)
        self.assertEqual(slot.step, 5)


if __name__ == "__main__":
    unittest.main()
