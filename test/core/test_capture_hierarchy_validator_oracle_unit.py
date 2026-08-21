from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Callable
import unittest

import torch

from tllm import _core
from tllm.backends.protocol import GraphSafety
from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, PortKind, PortRead
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.producers.base import HookPoint
from tllm.producers.registry import ExecutionPlan, PlannedProducer, ProducerEntry
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.workgraph import (
    BackendSnapshot,
    CaptureAttachment,
    PortBindingDesc,
    PortBindingDirection,
    NodeKey,
    build_hierarchy,
    compile_shadow_parity,
    flow,
)
from test.core.oracles.capture_hierarchy_validator import validate_capture_hierarchy


class _Consumer(BaseConsumer[object]):
    def __init__(self) -> None:
        self.config = object()

    @property
    def consumer_id(self) -> str:
        return "oracle-consumer"

    def flows(self) -> tuple[ConsumerFlow, ...]:
        return (
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.HIDDEN_TARGET, role="oracle-role"),),
                writes=(),
                window="same_step",
            ),
        )


class _Producer:
    producer_id = "oracle-producer"
    output_port = PortId("oracle", "hidden")
    outputs = (
        PortSpec(
            id=output_port,
            kind=PortKind.HIDDEN_TARGET,
            payload_type="torch.Tensor",
        ),
    )
    requires: tuple[PortId, ...] = ()
    hook_points = frozenset({HookPoint.EXECUTE_MODEL_POST})
    graph_safety = GraphSafety.CAPTURE_SAFE


_BINDING = PortBindingDesc(
    port=_Producer.output_port,
    direction=PortBindingDirection.INPUT,
    value_path=(0,),
)


@flow(bindings=(_BINDING,))
def _oracle_flow(value: torch.Tensor) -> torch.Tensor:
    return value + torch.rand_like(value)


def _plans() -> tuple[_Consumer, DispatchPlan, ExecutionPlan]:
    consumer = _Consumer()
    dispatch = DispatchPlan.build([consumer])
    producer = _Producer()
    planned = PlannedProducer(
        entry=ProducerEntry(producer=producer),
        emits_at=producer.hook_points,
        output_ports=(producer.output_port,),
        depends_on=(),
    )
    execution = ExecutionPlan(
        producers=(planned,),
        port_sources={producer.output_port: producer.producer_id},
    )
    return consumer, dispatch, execution


def _fixture() -> tuple[object, object]:
    consumer, dispatch, execution = _plans()
    captured = _oracle_flow.capture(torch.ones(2))
    attachment = CaptureAttachment(
        node_key=NodeKey.consumer_flow(consumer.consumer_id, 0),
        artifact_key=captured.artifact_key,
        bindings=(_BINDING,),
    )
    hierarchy = build_hierarchy(
        captures=(captured,),
        attachments=(attachment,),
        dispatch_plan=dispatch,
        execution_plan=execution,
    )
    plan = compile_shadow_parity(
        consumers=(consumer,),
        dispatch_plan=dispatch,
        execution_plan=execution,
        backend=BackendSnapshot(
            backend_id="vllm-oracle",
            runner_family="mrv2",
            graph_mode="eager",
            backend_version="0.27.1",
        ),
    )
    return hierarchy, plan.semantic


def _proxy(value: object, **changes: object) -> SimpleNamespace:
    if dataclasses.is_dataclass(value):
        fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    else:
        fields = dict(vars(value))
    fields.update(changes)
    return SimpleNamespace(**fields)


def _hierarchy_with_root(hierarchy: object, root: object, *, attachments: object | None = None, data_centric: object | None = None) -> object:
    return _proxy(
        hierarchy,
        roots=(root,),
        attachments=hierarchy.attachments if attachments is None else attachments,
        data_centric=hierarchy.data_centric if data_centric is None else data_centric,
    )


def _root_parts(hierarchy: object) -> tuple[object, object, object, object]:
    root = hierarchy.roots[0]
    task = root.children[0]
    leaf = task.children[0]
    return root, task, leaf, leaf.leaf_artifact


def _corruptions(hierarchy: object, semantic: object) -> dict[str, tuple[object, object]]:
    root, task, leaf, artifact = _root_parts(hierarchy)
    attachment = hierarchy.attachments[0]
    corruptions: dict[str, tuple[object, object]] = {}

    corruptions["schema_version"] = (_proxy(hierarchy, schema_version=1), semantic)

    bad_artifact = _proxy(root.artifact_key, digest="A" * 64)
    bad_root = _proxy(root, artifact_key=bad_artifact, key=_proxy(root.key, artifact_key=bad_artifact))
    bad_attachment = _proxy(attachment, artifact_key=bad_artifact)
    corruptions["artifact_identity"] = (
        _hierarchy_with_root(hierarchy, bad_root, attachments=(bad_attachment,)),
        semantic,
    )

    corruptions["root_region_identity"] = (
        _hierarchy_with_root(hierarchy, _proxy(root, key=_proxy(root.key, kind="task"))),
        semantic,
    )

    bad_operator = _proxy(artifact.operators[0], operator_kind="unsupported")
    bad_artifact_desc = _proxy(artifact, operators=(bad_operator, *artifact.operators[1:]))
    bad_leaf = _proxy(leaf, leaf_artifact=bad_artifact_desc)
    bad_task = _proxy(task, children=(bad_leaf,))
    corruptions["typed_operator_fact"] = (
        _hierarchy_with_root(hierarchy, _proxy(root, children=(bad_task,))),
        semantic,
    )

    bad_task = _proxy(task, effects=("opaque",))
    corruptions["task_fact_parity"] = (
        _hierarchy_with_root(hierarchy, _proxy(root, children=(bad_task,))),
        semantic,
    )

    bad_leaf = _proxy(leaf, operator_ordinals=())
    bad_task = _proxy(task, operator_ordinals=(), children=(bad_leaf,))
    corruptions["operator_coverage"] = (
        _hierarchy_with_root(hierarchy, _proxy(root, children=(bad_task,))),
        semantic,
    )

    bad_value = _proxy(root.values[0], shape=())
    corruptions["typed_value_fact"] = (
        _hierarchy_with_root(hierarchy, _proxy(root, values=(bad_value, *root.values[1:]))),
        semantic,
    )

    corruptions["root_effect_order"] = (
        _hierarchy_with_root(hierarchy, _proxy(root, effects=tuple(reversed(root.effects)))),
        semantic,
    )

    bad_endpoint = _proxy(attachment, node_key=NodeKey.consumer_flow("unknown", 0))
    corruptions["attachment_endpoint"] = (
        _hierarchy_with_root(hierarchy, root, attachments=(bad_endpoint,)),
        semantic,
    )

    bad_binding = _proxy(attachment.bindings[0], port=PortId("oracle", "unknown"))
    corruptions["attachment_binding"] = (
        _hierarchy_with_root(hierarchy, root, attachments=(_proxy(attachment, bindings=(bad_binding,)),)),
        semantic,
    )

    bad_lease = _proxy(hierarchy.data_centric.leases[0], materialized=True)
    bad_data = _proxy(hierarchy.data_centric, leases=(bad_lease, *hierarchy.data_centric.leases[1:]))
    corruptions["materialized_lease"] = (_hierarchy_with_root(hierarchy, root, data_centric=bad_data), semantic)

    bad_data = _proxy(hierarchy.data_centric, moves=("move",))
    corruptions["move_descriptor"] = (_hierarchy_with_root(hierarchy, root, data_centric=bad_data), semantic)

    bad_node = _proxy(
        semantic.nodes[0],
        node_key=NodeKey(namespace="capture_operator", owner="aten.add", ordinal=0),
    )
    corruptions["aten_schedule_node"] = (
        hierarchy,
        _proxy(semantic, nodes=(*semantic.nodes, bad_node)),
    )
    return corruptions


def _outcome(callable_: Callable[[], object]) -> bool:
    try:
        callable_()
    except BaseException:  # Differential tests normalize Python/Rust exception classes.
        return False
    return True


class CaptureHierarchyValidatorOracleTest(unittest.TestCase):
    def test_valid_hierarchy_has_identical_python_and_rust_outcome(self) -> None:
        hierarchy, semantic = _fixture()
        python_result = validate_capture_hierarchy(hierarchy, semantic)
        rust_result = _core.validate_capture_hierarchy(hierarchy, semantic)
        self.assertEqual(python_result, tuple(rust_result.topological_order))

    def test_single_field_corruptions_have_identical_python_and_rust_outcomes(self) -> None:
        hierarchy, semantic = _fixture()
        for name, (broken_hierarchy, broken_semantic) in _corruptions(hierarchy, semantic).items():
            with self.subTest(corruption=name):
                self.assertFalse(_outcome(lambda: validate_capture_hierarchy(broken_hierarchy, broken_semantic)))
                self.assertFalse(_outcome(lambda: _core.validate_capture_hierarchy(broken_hierarchy, broken_semantic)))


if __name__ == "__main__":
    unittest.main()
