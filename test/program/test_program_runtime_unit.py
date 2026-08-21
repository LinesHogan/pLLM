#!/usr/bin/env python3
"""Program runtime tests (plan Phase 3).

Covers compile_program (oracle-checked functional transitions), the
generated consumer's consume_flow dispatch, REPLICA and REQUEST state banks
(slot grouping, no cross-prompt contamination, one update per prompt per
batch), version/fence observability, sink publication, and fail-closed
behavior (PG145 oracle mismatch, PG160 unsupported optimizer).
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.ports.base import TimingDeadline
from tllm.program import ProgramSink, StateSpec, compile_program
from tllm.ports.consumer import BaseConsumer
from tllm.workgraph.frontend import (
    StateScope,
    extern,
    flow,
    hidden,
    local,
    request_prompt_idx,
    result,
    sink,
)
from tllm.workgraph.lowering import WorkGraphError


class _TinyDistiller(torch.nn.Module):
    def __init__(self, lr: float = 0.1) -> None:
        super().__init__()
        self.model = torch.nn.Linear(4, 4)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="decode"),
            "target": hidden(layer=-1, site="block_output", phase="decode"),
        },
        outputs={"prediction": sink("guidance"), "loss": local()},
        state="distiller",
    )
    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(source)
        return prediction, F.mse_loss(prediction.float(), target.detach().float())

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="distiller",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


class _RecordingSink:
    def __init__(self) -> None:
        self.published: list[torch.Tensor] = []

    @property
    def name(self) -> str:
        return "guidance"

    def publish(self, value: torch.Tensor, meta: RowBatchMeta | None) -> None:
        del meta
        self.published.append(value.detach().clone())


def _bundle(
    source: torch.Tensor,
    target: torch.Tensor,
    prompt_idxs: tuple[int, ...],
    *,
    step_id: int,
) -> PortBundle:
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
                prompt_idxs=prompt_idxs,
                sample_idxs=tuple(0 for _ in range(rows)),
                phase="decode",
                engine_step_id=step_id,
            ),
        },
    )


def _examples(rows: int = 2) -> dict[str, tuple[torch.Tensor, ...]]:
    return {"forward": (torch.randn(rows, 4), torch.randn(rows, 4))}


def _compile_replica(instance: _TinyDistiller, sink_impl: _RecordingSink):
    return compile_program(
        instance,
        states={"distiller": StateSpec(scope=StateScope.REPLICA, optimizer=torch.optim.SGD)},
        examples=_examples(),
        sinks=(sink_impl,),
    )


class CompileProgramTest(unittest.TestCase):
    def test_compile_runs_oracle_and_builds_consumer(self) -> None:
        sink_impl = _RecordingSink()
        consumer = _compile_replica(_TinyDistiller(), sink_impl)
        self.assertEqual(consumer.consumer_id, "program:_TinyDistiller")

    def test_missing_example_fails_pg130(self) -> None:
        with self.assertRaises(WorkGraphError) as ctx:
            compile_program(
                _TinyDistiller(),
                states={"distiller": StateSpec(scope=StateScope.REPLICA)},
                examples={},
                sinks=(_RecordingSink(),),
            )
        self.assertEqual(ctx.exception.code, "PG130")

    def test_unregistered_sink_fails_pg100(self) -> None:
        with self.assertRaises(WorkGraphError) as ctx:
            compile_program(
                _TinyDistiller(),
                states={"distiller": StateSpec(scope=StateScope.REPLICA)},
                examples=_examples(),
            )
        self.assertEqual(ctx.exception.code, "PG100")

    def test_unsupported_optimizer_fails_pg160(self) -> None:
        class RMS(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Linear(4, 4)
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.1)

            forward = _TinyDistiller.forward
            backward = _TinyDistiller.backward

        with self.assertRaises(WorkGraphError) as ctx:
            compile_program(
                RMS(),
                states={"distiller": StateSpec(scope=StateScope.REPLICA)},
                examples=_examples(),
                sinks=(_RecordingSink(),),
            )
        self.assertEqual(ctx.exception.code, "PG160")

    def test_oracle_mismatch_fails_pg145(self) -> None:
        class Cheater(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Linear(4, 4)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

            forward = _TinyDistiller.forward

            @extern(
                inputs={"loss": result("forward", "loss")},
                after=("forward",),
                state="distiller",
                deadline=TimingDeadline.NEXT_PREPARE,
            )
            def backward(self, loss: torch.Tensor) -> None:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    for param in self.model.parameters():
                        param.mul_(2.0)

        with self.assertRaises(WorkGraphError) as ctx:
            compile_program(
                Cheater(),
                states={"distiller": StateSpec(scope=StateScope.REPLICA)},
                examples=_examples(),
                sinks=(_RecordingSink(),),
            )
        self.assertEqual(ctx.exception.code, "PG145")


class ReplicaRuntimeTest(unittest.TestCase):
    def test_training_updates_state_and_sink_sees_prediction(self) -> None:
        torch.manual_seed(0)
        instance = _TinyDistiller(lr=0.5)
        sink_impl = _RecordingSink()
        consumer = _compile_replica(instance, sink_impl)
        source = torch.randn(4, 4)
        target = torch.randn(4, 4)

        initial_loss = F.mse_loss(
            instance.model(source).float(), target.float()
        ).item()
        consumer.consume_flow(0, _bundle(source, target, (0, 0, 0, 0), step_id=1), None)

        self.assertEqual(consumer.update_count, 1)
        self.assertEqual(consumer.loss_count, 1)
        self.assertNotEqual(consumer.loss_sum, 0.0)
        self.assertEqual(len(sink_impl.published), 1)
        self.assertTrue(
            torch.allclose(sink_impl.published[0], instance.model(source))
        )

        bank = consumer.bank("distiller")
        slot = bank.slot_for(None)
        self.assertEqual(slot.version, 1)
        self.assertEqual(consumer.read_versions[("distiller", 0)], 0)

        # The update descends: same batch after the step has lower loss.
        updated_loss = F.mse_loss(
            (source @ slot.params["model.weight"].t() + slot.params["model.bias"]).float(),
            target.float(),
        ).item()
        self.assertLess(updated_loss, initial_loss)

    def test_version_fence_visible_across_steps(self) -> None:
        torch.manual_seed(0)
        instance = _TinyDistiller(lr=0.5)
        sink_impl = _RecordingSink()
        consumer = _compile_replica(instance, sink_impl)
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)

        consumer.consume_flow(0, _bundle(source, target, (0, 0), step_id=1), None)
        consumer.consume_flow(0, _bundle(source, target, (0, 0), step_id=2), None)
        # Second read observed version 1 (published by the first update).
        self.assertEqual(consumer.read_versions[("distiller", 0)], 1)
        self.assertEqual(consumer.update_count, 2)
        # The second prediction used the updated parameters.
        slot = consumer.bank("distiller").slot_for(None)
        expected = (
            source @ slot.params["model.weight"].t() + slot.params["model.bias"]
        )
        # slot is now at version 2; prediction for step 2 used version 1, so
        # recompute the version-1 prediction indirectly: the two published
        # predictions must differ.
        self.assertFalse(
            torch.allclose(sink_impl.published[0], sink_impl.published[1])
        )
        del expected


class RequestRuntimeTest(unittest.TestCase):
    def _compile_request(self, instance: _TinyDistiller, sink_impl: _RecordingSink):
        return compile_program(
            instance,
            states={
                "distiller": StateSpec(
                    scope=StateScope.REQUEST,
                    key=request_prompt_idx(),
                    optimizer=torch.optim.SGD,
                )
            },
            examples=_examples(),
            sinks=(sink_impl,),
        )

    def test_duplicate_prompt_updates_once_per_batch(self) -> None:
        torch.manual_seed(0)
        instance = _TinyDistiller(lr=0.3)
        consumer = self._compile_request(instance, _RecordingSink())
        source = torch.randn(4, 4)
        target = torch.randn(4, 4)
        consumer.consume_flow(0, _bundle(source, target, (7, 7, 9, 7), step_id=1), None)
        # Prompt 7 appears three times but is updated once.
        self.assertEqual(consumer.update_count, 2)
        bank = consumer.bank("distiller")
        self.assertEqual(bank.slot_for(7).step, 1)
        self.assertEqual(bank.slot_for(9).step, 1)

    def test_prompts_do_not_contaminate_each_other(self) -> None:
        torch.manual_seed(0)
        instance = _TinyDistiller(lr=0.5)
        consumer = self._compile_request(instance, _RecordingSink())
        source = torch.randn(2, 4)
        target_a = torch.randn(2, 4)
        target_b = -target_a
        # Prompt 1 sees target_a twice; prompt 2 sees target_b twice.
        rows_source = torch.cat([source, source], dim=0)
        rows_target = torch.cat([target_a, target_b], dim=0)
        consumer.consume_flow(
            0, _bundle(rows_source, rows_target, (1, 1, 2, 2), step_id=1), None
        )
        bank = consumer.bank("distiller")
        self.assertFalse(
            torch.allclose(
                bank.slot_for(1).params["model.weight"],
                bank.slot_for(2).params["model.weight"],
            )
        )

    def test_negative_prompt_idx_fails_pg180(self) -> None:
        torch.manual_seed(0)
        instance = _TinyDistiller()
        consumer = self._compile_request(instance, _RecordingSink())
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)
        with self.assertRaises(WorkGraphError) as ctx:
            consumer.consume_flow(
                0, _bundle(source, target, (0, -1), step_id=1), None
            )
        self.assertEqual(ctx.exception.code, "PG180")


class LegacyCompatTest(unittest.TestCase):
    def test_default_consume_flow_delegates_to_consume_bundle(self) -> None:
        calls: list[tuple[object, object]] = []

        class Legacy(BaseConsumer[object]):
            config = object()

            @property
            def consumer_id(self) -> str:
                return "legacy"

            def consume_bundle(self, bundle: PortBundle, ctx: object) -> None:
                calls.append((bundle, ctx))

        legacy = Legacy()
        bundle = _bundle(torch.randn(1, 4), torch.randn(1, 4), (0,), step_id=1)
        legacy.consume_flow(3, bundle, None)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], bundle)


if __name__ == "__main__":
    unittest.main()
