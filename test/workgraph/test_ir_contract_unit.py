from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

from tllm.ports.address import LocatorFieldDesc, PortAddressDesc
from tllm.ports.base import PortKind, PortRead
from tllm.ports.port_id import PortId
from tllm.workgraph.ir import (
    AccessDesc,
    AccessMode,
    AnchorDesc,
    BackendSnapshot,
    CompiledWorkGraph,
    DataCentricIR,
    EdgeKind,
    ExecutionNodeDesc,
    ExecutorKey,
    LeaseDesc,
    MoveDesc,
    NodeKey,
    ResidenceDesc,
    SemanticGraph,
    SemanticNodeDesc,
    TimingDesc,
    ValueDesc,
)
from tllm.workgraph.signature import canonical_json, structural_signature
from tllm.workgraph.lowering import _address_value


class WorkGraphIRContractTest(unittest.TestCase):
    def _graph(self) -> CompiledWorkGraph:
        key = NodeKey.consumer_flow("demo", 0)
        value = ValueDesc("hidden", port=PortId("demo", "hidden"))
        read = AccessDesc(key, value, AccessMode.READ)
        node = SemanticNodeDesc(key, reads=(read,))
        semantic = SemanticGraph(nodes=(node,), values=(value,))
        execution = ExecutionNodeDesc(key, accesses=(read,))
        data = DataCentricIR(nodes=(execution,), values=(value,), accesses=(read,))
        return CompiledWorkGraph(semantic=semantic, data_centric=data, epoch=1)

    def test_stable_keys_are_structural(self) -> None:
        self.assertEqual(str(NodeKey.consumer_flow("consumer", 3)), "consumer_flow/consumer/3")
        self.assertEqual(str(NodeKey.producer("provider")), "producer/provider/0")
        self.assertEqual(
            str(NodeKey.anchor("vllm", "mrv2", "piecewise", "execute_model.post")),
            "anchor/vllm/mrv2/piecewise/execute_model.post",
        )
        for descriptor in (NodeKey("x", "y"), AnchorDesc("vllm", "mrv2", "eager", "execute_model.post")):
            self.assertTrue(dataclasses.is_dataclass(descriptor))
            self.assertTrue(getattr(type(descriptor), "__dataclass_params__").frozen)
            self.assertTrue(hasattr(type(descriptor), "__slots__"))

    def test_port_refs_are_typed_and_lease_is_unmaterialized(self) -> None:
        address = PortAddressDesc.opaque("hidden")
        value = ValueDesc("hidden", port=address)
        self.assertIsInstance(value.port, PortAddressDesc)
        self.assertFalse(LeaseDesc(value).materialized)
        with self.assertRaises(ValueError):
            LeaseDesc(value, materialized=True)

    def test_address_value_identity_uses_structured_digest_not_display_label(self) -> None:
        first = PortAddressDesc(
            kind="residual_stream",
            locator_type="test.Locator",
            locator_fields=(
                LocatorFieldDesc(
                    path=("site",),
                    value_type="str",
                    value="block_output,phase:str=decode",
                ),
            ),
        )
        second = PortAddressDesc(
            kind="residual_stream",
            locator_type="test.Locator",
            locator_fields=(
                LocatorFieldDesc(path=("site",), value_type="str", value="block_output"),
                LocatorFieldDesc(path=("phase",), value_type="str", value="decode"),
            ),
        )
        self.assertEqual(first.label, second.label)
        self.assertNotEqual(first.identity, second.identity)

        with mock.patch(
            "tllm.workgraph.lowering.port_address_desc",
            side_effect=(first, second),
        ):
            left = _address_value(PortRead(kind=PortKind.RESIDUAL_STREAM))
            right = _address_value(PortRead(kind=PortKind.RESIDUAL_STREAM))
        self.assertNotEqual(left.key, right.key)
        self.assertEqual(left.port, first)
        self.assertEqual(right.port, second)

    def test_signature_rejects_untyped_runtime_objects(self) -> None:
        graph = self._graph()
        self.assertEqual(structural_signature(graph), structural_signature(graph))
        later_epoch = dataclasses.replace(graph, epoch=99)
        self.assertEqual(structural_signature(graph), structural_signature(later_epoch))
        with self.assertRaises(TypeError):
            canonical_json({"payload": object()})
        with self.assertRaises(TypeError):
            structural_signature(object())
        with self.assertRaises(TypeError):
            structural_signature(ValueDesc("hidden", port=object()))  # type: ignore[arg-type]

    def test_public_structural_fields_reject_implicit_stringification(self) -> None:
        opaque = object()
        with self.assertRaises(TypeError):
            NodeKey(opaque, "owner")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ExecutorKey("executor", opaque)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BackendSnapshot(
                backend_id=opaque,  # type: ignore[arg-type]
                runner_family="mrv1",
                backend_version="0.27.1",
            )
        with self.assertRaises(TypeError):
            AnchorDesc("vllm", "mrv1", "eager", opaque)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ValueDesc(opaque)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ValueDesc("hidden", shape=["batch"])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            TimingDesc(mobility=opaque)  # type: ignore[arg-type]

    def test_descriptor_enums_reject_magic_values(self) -> None:
        key = NodeKey.consumer_flow("demo", 0)
        with self.assertRaises(ValueError):
            SemanticNodeDesc(key, kind="unknown")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AccessDesc(key, ValueDesc("hidden"), "unknown")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AccessDesc(
                key,
                ValueDesc("hidden", port=PortId("demo", "hidden")),
                AccessMode.READ,
                port=PortId("other", "hidden"),
            )

    def test_ir_schema_version_is_explicit_and_validated(self) -> None:
        key = NodeKey.consumer_flow("demo", 0)
        with self.assertRaises(ValueError):
            SemanticGraph(nodes=(SemanticNodeDesc(key),), schema_version=2)
        with self.assertRaises(ValueError):
            DataCentricIR(nodes=(ExecutionNodeDesc(key),), schema_version=2)
        self.assertEqual(self._graph().semantic.schema_version, 1)
        self.assertEqual(self._graph().data_centric.schema_version, 1)

    def test_graph_rejects_duplicate_dangling_and_cyclic_edges(self) -> None:
        first = NodeKey.consumer_flow("demo", 0)
        second = NodeKey.consumer_flow("demo", 1)
        node_a = SemanticNodeDesc(first)
        node_b = SemanticNodeDesc(second)
        with self.assertRaises(ValueError):
            SemanticGraph(nodes=(node_a, node_a))
        with self.assertRaises(ValueError):
            SemanticGraph(nodes=(node_a,), edges=((first, second, EdgeKind.DATA),))
        with self.assertRaises(ValueError):
            SemanticGraph(
                nodes=(node_a, node_b),
                edges=(
                    (first, second, EdgeKind.DATA),
                    (second, first, EdgeKind.CONTROL),
                ),
            )

    def test_phase1a_data_ir_rejects_moves(self) -> None:
        key = NodeKey.consumer_flow("demo", 0)
        value = ValueDesc("hidden")
        source = ResidenceDesc(value, device="cuda")
        target = ResidenceDesc(value, device="cpu")
        move = MoveDesc(value, source, target)
        with self.assertRaises(ValueError):
            DataCentricIR(
                nodes=(ExecutionNodeDesc(key),),
                values=(value,),
                residences=(source, target),
                moves=(move,),
            )
        with self.assertRaises(ValueError):
            DataCentricIR(nodes=(ExecutionNodeDesc(key, move_keys=(value.key,)),))
        with self.assertRaises(ValueError):
            DataCentricIR(
                nodes=(ExecutionNodeDesc(key),),
                values=(value, value),
                residences=(source, source),
                leases=(LeaseDesc(value), LeaseDesc(value)),
            )

    def test_anchor_edge_shape_and_compiled_projection_are_fail_closed(self) -> None:
        anchor = AnchorDesc("vllm", "mrv1", "eager", "execute_model.post")
        anchor_key = anchor.node_key
        compute_key = NodeKey.consumer_flow("demo", 0)
        anchor_node = SemanticNodeDesc(anchor_key, kind="anchor", anchor=anchor)
        compute_node = SemanticNodeDesc(compute_key)
        with self.assertRaises(ValueError):
            SemanticGraph(
                nodes=(anchor_node, compute_node),
                edges=((compute_key, anchor_key, EdgeKind.ANCHOR),),
            )
        semantic = SemanticGraph(nodes=(compute_node,))
        data = DataCentricIR(nodes=(ExecutionNodeDesc(compute_key, kind="provider"),))
        with self.assertRaises(ValueError):
            CompiledWorkGraph(semantic=semantic, data_centric=data)


if __name__ == "__main__":
    unittest.main()
