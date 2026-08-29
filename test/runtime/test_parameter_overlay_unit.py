"""Generic immutable overlay publication and lease contracts."""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.request_identity import RequestIdentity
from tllm.runtime.parameter_overlay import (
    OverlayChannel,
    OverlayLineage,
    OverlayPayload,
    OverlayPublication,
    OverlayRepresentation,
    OverlayVersionStore,
)


class ParameterOverlayUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lineage = OverlayLineage(OverlayChannel("test", "weights"))

    def test_publication_owns_input_and_materialization_is_not_mutable_state(self) -> None:
        source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        payload = OverlayPayload(OverlayRepresentation.DENSE_REPLACE, {"w": source})
        source.add_(100)
        store = OverlayVersionStore()
        store.stage("tx:1", OverlayPublication(self.lineage, 3, payload))
        version = store.commit("tx:1")

        first = store.materialize(version.ref)
        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        torch.testing.assert_close(first["w"], expected)
        first["w"].add_(50)
        torch.testing.assert_close(
            store.materialize(version.ref)["w"], expected
        )

    def test_request_pin_is_stable_across_new_publication_and_released_version_collects(self) -> None:
        store = OverlayVersionStore()
        store.stage(
            "tx:1",
            OverlayPublication(
                self.lineage,
                1,
                OverlayPayload(OverlayRepresentation.LOW_RANK_ADD, {"a": torch.ones(1)}),
            ),
        )
        first = store.commit("tx:1")
        request = RequestIdentity("req", 4, 0)
        lease = store.pin_latest(request, self.lineage)
        assert lease is not None

        store.stage(
            "tx:2",
            OverlayPublication(
                self.lineage,
                1,
                OverlayPayload(OverlayRepresentation.LOW_RANK_ADD, {"a": torch.zeros(1)}),
            ),
        )
        second = store.commit("tx:2")

        self.assertEqual(store.pinned(request), lease)
        self.assertEqual(store.pin_latest(request, self.lineage), lease)
        self.assertEqual(tuple(item.ref for item in store.versions(self.lineage)), (first.ref, second.ref))
        self.assertTrue(store.release(request))
        self.assertEqual(tuple(item.ref for item in store.versions(self.lineage)), (second.ref,))

    def test_abort_and_failed_capacity_leave_latest_unchanged(self) -> None:
        store = OverlayVersionStore(max_versions=1)
        store.stage(
            "tx:1",
            OverlayPublication(
                self.lineage,
                1,
                OverlayPayload(OverlayRepresentation.DENSE_REPLACE, {"w": torch.ones(1)}),
            ),
        )
        first = store.commit("tx:1")
        store.pin_exact(RequestIdentity("req", 0, 0), first.ref)
        store.stage(
            "tx:2",
            OverlayPublication(
                self.lineage,
                1,
                OverlayPayload(OverlayRepresentation.DENSE_REPLACE, {"w": torch.zeros(1)}),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            store.commit("tx:2")
        self.assertEqual(store.latest(self.lineage), first)
        store.abort("tx:2")


if __name__ == "__main__":
    unittest.main()
