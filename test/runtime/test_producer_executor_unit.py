#!/usr/bin/env python3
"""Runtime contracts for plan-driven producer materialization."""

from __future__ import annotations

import unittest

import torch

from tllm.backends.protocol import GraphSafety
from tllm.ports.base import ConsumerFlow, PortKind
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.ports.residual_stream import ResidualStream
from tllm.producers.base import HookPoint
from tllm.producers.registry import ProducerRegistry, compile_execution_plan
from tllm.runtime.producer_executor import (
    ExecuteModelPostContext,
    LayerPostContext,
    PrepareInputsPostContext,
    ProducerExecutor,
    ProducerMaterializer,
    ProducerMaterializerRegistry,
)


class _Consumer:
    consumer_id = "test.consumer"

    def flows(self):
        return (
            ConsumerFlow(
                reads=(
                    ResidualStream.read(
                        layer=0,
                        site="block_output",
                        phase="decode",
                    ),
                ),
                writes=(),
                window="same_step",
            ),
        )


class _Producer:
    producer_id = "test.decode"
    outputs = (
        PortSpec(
            id=PortId("test", "decode", 1),
            kind=PortKind.RESIDUAL_STREAM,
            payload_type="Tensor",
            phases=("decode",),
        ),
    )
    requires = ()
    hook_points = frozenset(
        {
            HookPoint.PREPARE_INPUTS_POST,
            HookPoint.LAYER_POST,
            HookPoint.EXECUTE_MODEL_POST,
        }
    )
    graph_safety = GraphSafety.CAPTURE_SAFE


class _RecordingMaterializer(ProducerMaterializer):
    producer_id = "test.decode"

    def __init__(self) -> None:
        self.events: list[str] = []

    def prepare_inputs_post(self, ctx: PrepareInputsPostContext) -> None:
        self.events.append("prepare_inputs.post")
        ctx.localize_decode()

    def layer_post(self, ctx: LayerPostContext) -> None:
        self.events.append(f"layer.post:{ctx.layer_path}:{int(ctx.hidden.sum().item())}")

    def execute_model_post(self, ctx: ExecuteModelPostContext) -> None:
        self.events.append("execute_model.post")
        ctx.publish_decode()


def _plan():
    registry = ProducerRegistry()
    registry.register(_Producer())
    return compile_execution_plan(
        registry=registry,
        consumers=[_Consumer()],
        enforced_kinds=frozenset({PortKind.RESIDUAL_STREAM}),
    )


class ProducerExecutorUnitTest(unittest.TestCase):
    def test_compiled_plan_drives_all_materialization_stages(self) -> None:
        materializer = _RecordingMaterializer()
        registry = ProducerMaterializerRegistry()
        registry.register(materializer)
        executor = ProducerExecutor.bind(plan=_plan(), registry=registry)
        callbacks: list[str] = []

        executor.run_prepare_inputs_post(
            PrepareInputsPostContext(
                runner=object(),
                localize_decode=lambda: callbacks.append("localized"),
            )
        )
        executor.run_layer_post(
            LayerPostContext(
                runtime=None,
                runner=object(),
                layer_path="layers.0",
                hidden=torch.tensor([[1.0, 2.0]]),
            )
        )
        executor.run_execute_model_post(
            ExecuteModelPostContext(
                runner=object(),
                publish_decode=lambda: callbacks.append("published"),
            )
        )

        self.assertEqual(
            materializer.events,
            [
                "prepare_inputs.post",
                "layer.post:layers.0:3",
                "execute_model.post",
            ],
        )
        self.assertEqual(callbacks, ["localized", "published"])
        self.assertEqual(executor.capture_resident_producer_ids, ("test.decode",))

    def test_binding_rejects_selected_producer_without_materializer(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no runtime materializer"):
            ProducerExecutor.bind(
                plan=_plan(),
                registry=ProducerMaterializerRegistry(),
            )


if __name__ == "__main__":
    unittest.main()
