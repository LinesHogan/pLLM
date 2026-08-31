"""Exact oracle for entitled, atomic immutable state-root publication."""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId
from tllm.runtime.paged_state import (
    FrozenTensorField,
    FrozenTensorLayout,
    PageBundle,
    PagedStateVersionArena,
    StateCapacityEntitlement,
    TensorFieldId,
)
from tllm.runtime.request_slot_adapter import NoCompletionFence


class PagedStateUnitTest(unittest.TestCase):
    def test_entitlement_atomic_publication_pin_reclaim_and_aba(self) -> None:
        layout = FrozenTensorLayout(
            LayoutId(7),
            2,
            (
                FrozenTensorField(TensorFieldId(0), (2,), torch.float32),
                FrozenTensorField(TensorFieldId(1), (2,), torch.float32),
            ),
        )
        entitlement = StateCapacityEntitlement(
            max_admitted_lineages=2,
            max_live_versions_per_lineage=3,
        )
        with self.assertRaisesRegex(ValueError, "capacity entitlement"):
            PagedStateVersionArena(
                PageBundle(layout, capacity=11, device=torch.device("cpu")),
                bundle_epoch=3,
                capacity_entitlement=entitlement,
            )

        bundle = PageBundle(layout, capacity=12, device=torch.device("cpu"))
        arena = PagedStateVersionArena(
            bundle,
            bundle_epoch=3,
            capacity_entitlement=entitlement,
        )
        first_lineage = LineageId(1)
        second_lineage = LineageId(2)
        first_v0 = StateRef(layout.layout_id, first_lineage, VersionId(0))
        second_v0 = StateRef(layout.layout_id, second_lineage, VersionId(0))

        wrong_layout = StateRef(LayoutId(8), second_lineage, VersionId(0))
        with self.assertRaisesRegex(ValueError, "another layout"):
            arena.prepare_admission_batch((first_v0, wrong_layout))
        self.assertEqual(arena.free_pages, 12)

        admission = arena.prepare_admission_batch((first_v0, second_v0))
        for ref in (first_v0, second_v0):
            with self.assertRaisesRegex(RuntimeError, "unknown or reclaimed"):
                arena.root(ref)
        for root_index, write in enumerate(admission.writes):
            for virtual_page, page in enumerate(write.target_pages):
                page[0].fill_(float(root_index * 20 + virtual_page + 1))
                page[1].fill_(float(root_index * 20 + virtual_page + 11))
        first_root, _ = arena.publish_batch(
            admission,
            NoCompletionFence(),
        )
        self.assertEqual(arena.free_pages, 8)

        third_v0 = StateRef(layout.layout_id, LineageId(3), VersionId(0))
        with self.assertRaisesRegex(RuntimeError, "lineage entitlement"):
            arena.prepare_admission_batch((third_v0,))
        self.assertEqual(arena.free_pages, 8)

        pinned_v0 = arena.pin_exact_synchronous((first_v0, second_v0))
        first_v1 = StateRef(layout.layout_id, first_lineage, VersionId(1))
        second_v1 = StateRef(layout.layout_id, second_lineage, VersionId(1))
        aborted = arena.prepare_transition_batch_synchronous(
            (first_v0, second_v0),
            (first_v1, second_v1),
        )
        arena.abort_unsubmitted_batch(aborted)
        for ref in (first_v1, second_v1):
            with self.assertRaisesRegex(RuntimeError, "unknown or reclaimed"):
                arena.root(ref)
        self.assertEqual(arena.free_pages, 8)

        first_transition = arena.prepare_transition_batch_synchronous(
            (first_v0,),
            (first_v1,),
        )
        (first_write,) = first_transition.writes
        for target_page, source_page in zip(
            first_write.target_pages,
            first_write.source_pages,
            strict=True,
        ):
            for target, source in zip(target_page, source_page, strict=True):
                target.copy_(source)
            target_page[0].add_(100.0)
            target_page[1].add_(200.0)
        (first_v1_root,) = arena.publish_batch(
            first_transition,
            NoCompletionFence(),
        )

        weight_pool, moment_pool = bundle.capture_operands
        torch.testing.assert_close(
            weight_pool[first_root.page_ids[0]],
            torch.full((2,), 1.0),
        )
        torch.testing.assert_close(
            moment_pool[first_root.page_ids[1]],
            torch.full((2,), 12.0),
        )
        torch.testing.assert_close(
            weight_pool[first_v1_root.page_ids[0]],
            torch.full((2,), 101.0),
        )
        torch.testing.assert_close(
            moment_pool[first_v1_root.page_ids[1]],
            torch.full((2,), 212.0),
        )

        pinned_v1 = arena.pin_exact_synchronous((first_v1,))
        first_v2 = StateRef(layout.layout_id, first_lineage, VersionId(2))
        first_v2_batch = arena.prepare_transition_batch_synchronous(
            (first_v1,),
            (first_v2,),
        )
        (first_v2_write,) = first_v2_batch.writes
        for target_page, source_page in zip(
            first_v2_write.target_pages,
            first_v2_write.source_pages,
            strict=True,
        ):
            for target, source in zip(target_page, source_page, strict=True):
                target.copy_(source)
            target_page[0].add_(1.0)
        arena.publish_batch(first_v2_batch, NoCompletionFence())

        first_v3 = StateRef(layout.layout_id, first_lineage, VersionId(3))
        with self.assertRaisesRegex(RuntimeError, "version entitlement"):
            arena.prepare_transition_batch_synchronous((first_v2,), (first_v3,))
        self.assertEqual(arena.root(first_v2).ref, first_v2)

        arena.release_pin(pinned_v0, NoCompletionFence())
        first_v3_batch = arena.prepare_transition_batch_synchronous(
            (first_v2,),
            (first_v3,),
        )
        (first_v3_write,) = first_v3_batch.writes
        for target_page, source_page in zip(
            first_v3_write.target_pages,
            first_v3_write.source_pages,
            strict=True,
        ):
            for target, source in zip(target_page, source_page, strict=True):
                target.copy_(source)
        arena.publish_batch(first_v3_batch, NoCompletionFence())
        arena.release_pin(pinned_v1, NoCompletionFence())

        arena.retire(first_v3)
        arena.retire(second_v0)
        self.assertEqual(arena.free_pages, 12)
        with self.assertRaisesRegex(RuntimeError, "unknown or reclaimed"):
            arena.root(first_v0)

        reused_admission = arena.prepare_admission_batch((third_v0,))
        for page in reused_admission.writes[0].target_pages:
            for tensor in page:
                tensor.zero_()
        arena.publish_batch(reused_admission, NoCompletionFence())
        arena.retire(third_v0)
        self.assertEqual(arena.free_pages, 12)


if __name__ == "__main__":
    unittest.main()
