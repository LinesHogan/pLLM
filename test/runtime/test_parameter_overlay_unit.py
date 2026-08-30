"""Generic immutable overlay publication and lease contracts."""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.request_identity import RequestIdentity
from tllm.runtime.parameter_overlay import (
    OverlayChannel,
    OverlayLeaseKey,
    OverlayLineage,
    OverlayPayload,
    OverlayPublication,
    OverlayRepresentation,
    OverlayVersion,
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
        key = OverlayLeaseKey(request, self.lineage)
        lease = store.pin_latest(key)
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

        self.assertEqual(store.pinned(key), lease)
        self.assertEqual(store.pin_latest(key), lease)
        self.assertEqual(tuple(item.ref for item in store.versions(self.lineage)), (first.ref, second.ref))
        self.assertTrue(store.release(key))
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
        store.pin_exact(
            OverlayLeaseKey(RequestIdentity("req", 0, 0), self.lineage),
            first.ref,
        )
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

    def test_one_request_keeps_independent_leases_for_multiple_lineages(self) -> None:
        request = RequestIdentity("multi", 2, 0)
        first_lineage = OverlayLineage(
            OverlayChannel("test", "layer-0"), request
        )
        second_lineage = OverlayLineage(
            OverlayChannel("test", "layer-1"), request
        )
        store = OverlayVersionStore()
        store.stage(
            "layer-0:1",
            OverlayPublication(
                first_lineage,
                1,
                OverlayPayload(
                    OverlayRepresentation.DENSE_REPLACE,
                    {"weight": torch.ones(1)},
                ),
            ),
        )
        first = store.commit("layer-0:1")
        store.stage(
            "layer-1:1",
            OverlayPublication(
                second_lineage,
                1,
                OverlayPayload(
                    OverlayRepresentation.DENSE_REPLACE,
                    {"weight": torch.full((1,), 2.0)},
                ),
            ),
        )
        second = store.commit("layer-1:1")
        first_key = OverlayLeaseKey(request, first_lineage)
        second_key = OverlayLeaseKey(request, second_lineage)

        first_lease = store.advance_request(first_key, first.ref)
        second_lease = store.advance_request(second_key, second.ref)

        self.assertEqual(store.pinned(first_key), first_lease)
        self.assertEqual(store.pinned(second_key), second_lease)
        self.assertEqual(store.byte_size, first.byte_size + second.byte_size)

    def test_request_retirement_collects_request_lineages_but_keeps_global(self) -> None:
        request = RequestIdentity("finished", 3, 1)
        global_lineage = OverlayLineage(OverlayChannel("test", "global"))
        request_lineages = (
            OverlayLineage(OverlayChannel("test", "layer-0"), request),
            OverlayLineage(OverlayChannel("test", "layer-1"), request),
        )
        store = OverlayVersionStore()

        def publish(
            transaction: str, lineage: OverlayLineage, value: float
        ) -> OverlayVersion:
            store.stage(
                transaction,
                OverlayPublication(
                    lineage,
                    1,
                    OverlayPayload(
                        OverlayRepresentation.DENSE_REPLACE,
                        {"weight": torch.full((1,), value)},
                    ),
                ),
            )
            return store.commit(transaction)

        first_global_version = publish("global:1", global_lineage, 1.0)
        global_key = OverlayLeaseKey(request, global_lineage)
        global_lease = store.pin_latest(global_key)
        assert global_lease is not None
        self.assertEqual(global_lease.ref, first_global_version.ref)
        latest_global_version = publish("global:2", global_lineage, 4.0)
        request_versions = (
            publish("layer-0:1", request_lineages[0], 2.0),
            publish("layer-1:1", request_lineages[1], 3.0),
        )
        request_keys = tuple(
            OverlayLeaseKey(request, lineage) for lineage in request_lineages
        )
        for key, version in zip(request_keys, request_versions, strict=True):
            store.advance_request(key, version.ref)

        retirement = store.retire_request(request)

        self.assertEqual(retirement.released_leases, 3)
        self.assertEqual(retirement.retired_lineages, 2)
        self.assertEqual(retirement.retired_versions, 2)
        self.assertEqual(
            retirement.retired_bytes,
            sum(version.byte_size for version in request_versions),
        )
        self.assertEqual(store.latest(global_lineage), latest_global_version)
        self.assertEqual(
            tuple(version.ref for version in store.versions(global_lineage)),
            (latest_global_version.ref,),
        )
        self.assertEqual(store.byte_size, latest_global_version.byte_size)
        self.assertIsNone(store.pinned(global_key))
        for key, lineage in zip(request_keys, request_lineages, strict=True):
            self.assertIsNone(store.pinned(key))
            self.assertIsNone(store.latest(lineage))
            self.assertEqual(store.versions(lineage), ())
        self.assertEqual(store.lineages(), (global_lineage,))


if __name__ == "__main__":
    unittest.main()
