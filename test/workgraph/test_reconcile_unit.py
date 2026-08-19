from __future__ import annotations

import unittest

from tllm.workgraph.ir import (
    CompiledWorkGraph,
    DataCentricIR,
    ExecutionNodeDesc,
    NodeKey,
    SemanticGraph,
    SemanticNodeDesc,
)
from tllm.workgraph.reconcile import reconcile


def _plan(kind: str = "compute", *, epoch: int = 1) -> CompiledWorkGraph:
    key = NodeKey.consumer_flow("demo", 0)
    node = SemanticNodeDesc(key, kind=kind)
    semantic = SemanticGraph(nodes=(node,))
    data = DataCentricIR(nodes=(ExecutionNodeDesc(key, kind=kind),))
    return CompiledWorkGraph(semantic, data, epoch=epoch)


class WorkGraphReconcileTest(unittest.TestCase):
    def test_same_signature_keeps_epoch(self) -> None:
        first = _plan(epoch=7)
        second = _plan(epoch=7)
        diff = reconcile(first, second)
        self.assertFalse(diff.changed)
        self.assertEqual(diff.epoch, 7)
        self.assertEqual(tuple(diff.keep_regions[0].node_keys), (NodeKey.consumer_flow("demo", 0),))
        self.assertEqual(diff.rebuild_regions, ())

    def test_structural_change_increments_epoch(self) -> None:
        diff = reconcile(_plan(epoch=2), _plan("barrier", epoch=2))
        self.assertTrue(diff.changed)
        self.assertEqual(diff.epoch, 3)
        self.assertEqual(diff.keep_regions, ())
        self.assertEqual(diff.rebuild_regions[0].node_keys, (NodeKey.consumer_flow("demo", 0),))


if __name__ == "__main__":
    unittest.main()
