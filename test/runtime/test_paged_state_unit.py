"""Exact oracle for immutable paged-state roots and generational reclaim."""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId
from tllm.runtime.paged_state import (
    FrozenTensorField,
    FrozenTensorLayout,
    PageBundle,
    PagedStateVersionArena,
    TensorFieldId,
)
from tllm.runtime.request_slot_adapter import NoCompletionFence


class PagedStateUnitTest(unittest.TestCase):
    def test_pinned_root_capacity_reclaim_and_aba(self) -> None:
        layout = FrozenTensorLayout(
            LayoutId(7),
            2,
            (
                FrozenTensorField(
                    TensorFieldId(0),
                    (2,),
                    torch.float32,
                ),
                FrozenTensorField(
                    TensorFieldId(1),
                    (2,),
                    torch.float32,
                ),
            ),
        )
        bundle = PageBundle(
            layout,
            capacity=6,
            device=torch.device("cpu"),
        )
        arena = PagedStateVersionArena(bundle, bundle_epoch=3)
        first_lineage = LineageId(1)
        second_lineage = LineageId(2)
        first_v0 = StateRef(layout.layout_id, first_lineage, VersionId(0))
        second_v0 = StateRef(layout.layout_id, second_lineage, VersionId(0))

        first_admission = arena.prepare_admission(first_v0)
        for virtual_page, page in enumerate(first_admission.target_pages):
            page[0].fill_(float(virtual_page + 1))
            page[1].fill_(float(virtual_page + 11))
        first_root = arena.publish(first_admission, NoCompletionFence())

        second_admission = arena.prepare_admission(second_v0)
        for virtual_page, page in enumerate(second_admission.target_pages):
            page[0].fill_(float(virtual_page + 21))
            page[1].fill_(float(virtual_page + 31))
        arena.publish(second_admission, NoCompletionFence())
        self.assertEqual(arena.free_pages, 2)

        pinned_v0 = arena.pin_exact_synchronous((first_v0, second_v0))
        first_v1 = StateRef(layout.layout_id, first_lineage, VersionId(1))
        aborted_transition = arena.prepare_transition_synchronous(
            first_v0,
            first_v1,
        )
        arena.abort_unsubmitted(aborted_transition)
        self.assertEqual(arena.free_pages, 2)
        self.assertEqual(arena.root(first_v0), first_root)
        first_transition = arena.prepare_transition_synchronous(
            first_v0,
            first_v1,
        )
        for target_page, source_page in zip(
            first_transition.target_pages,
            first_transition.source_pages,
            strict=True,
        ):
            for target, source in zip(
                target_page,
                source_page,
                strict=True,
            ):
                target.copy_(source)
            target_page[0].add_(100.0)
            target_page[1].add_(200.0)
        first_v1_root = arena.publish(
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
        self.assertEqual(arena.free_pages, 0)

        first_v2 = StateRef(layout.layout_id, first_lineage, VersionId(2))
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            arena.prepare_transition_synchronous(first_v1, first_v2)
        self.assertEqual(arena.root(first_v1), first_v1_root)

        arena.release_pin(pinned_v0, NoCompletionFence())
        self.assertEqual(arena.free_pages, 2)
        second_transition = arena.prepare_transition_synchronous(
            first_v1,
            first_v2,
        )
        for target_page, source_page in zip(
            second_transition.target_pages,
            second_transition.source_pages,
            strict=True,
        ):
            for target, source in zip(
                target_page,
                source_page,
                strict=True,
            ):
                target.copy_(source)
            target_page[0].add_(1.0)
        arena.publish(second_transition, NoCompletionFence())

        arena.retire(first_v2)
        arena.retire(second_v0)
        self.assertEqual(arena.free_pages, 6)
        with self.assertRaisesRegex(RuntimeError, "unknown or reclaimed"):
            arena.root(first_v0)

        reused_lineage = LineageId(3)
        reused_v0 = StateRef(
            layout.layout_id,
            reused_lineage,
            VersionId(0),
        )
        reused_admission = arena.prepare_admission(reused_v0)
        for page in reused_admission.target_pages:
            for tensor in page:
                tensor.zero_()
        arena.publish(reused_admission, NoCompletionFence())
        with self.assertRaisesRegex(RuntimeError, "unknown or reclaimed"):
            arena.root(first_v0)
        arena.retire(reused_v0)
        self.assertEqual(arena.free_pages, 6)


if __name__ == "__main__":
    unittest.main()
