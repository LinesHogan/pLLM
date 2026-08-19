from __future__ import annotations

import json
import unittest

from tllm.workgraph.explain import explain_workgraph
from tllm.workgraph.ir import CompiledWorkGraph, DataCentricIR, ExecutionNodeDesc, NodeKey, SemanticGraph, SemanticNodeDesc
from tllm.workgraph.reconcile import reconcile


class WorkGraphExplainTest(unittest.TestCase):
    def test_explain_is_json_safe_and_does_not_expose_private_objects(self) -> None:
        key = NodeKey.consumer_flow("demo", 0)
        node = SemanticNodeDesc(key, source="consumer_flow:demo#0")
        graph = CompiledWorkGraph(
            semantic=SemanticGraph(nodes=(node,)),
            data_centric=DataCentricIR(nodes=(ExecutionNodeDesc(key),)),
            epoch=4,
        )
        report = explain_workgraph(graph, reconcile(graph, graph))
        encoded = report.to_json()
        decoded = json.loads(encoded)
        self.assertEqual(decoded["epoch"], 4)
        self.assertEqual(decoded["nodes"][0]["node_key"], "consumer_flow/demo/0")
        self.assertEqual(decoded["reconciliation"]["epoch"], 4)
        self.assertNotIn("Tensor", encoded)
        self.assertNotIn("0x", encoded)


if __name__ == "__main__":
    unittest.main()
