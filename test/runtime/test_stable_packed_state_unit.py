"""Conformance oracle for fixed and paged stable-state view producers."""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.request_identity import (
    RequestIdentity,
    RequestKeyField,
)
from tllm.contracts.request_state import StateSlotRef
from tllm.runtime.paged_state import (
    FrozenTensorField,
    FrozenTensorLayout,
    PageBundle,
    PagedStateVersionArena,
    StateLayoutId,
    StateResourceId,
    StateVersionLineage,
    StateVersionRef,
    TensorFieldId,
)
from tllm.runtime.request_slot_arena import RequestSlotArena
from tllm.runtime.request_slot_adapter import NoCompletionFence
from tllm.runtime.stable_packed_state import (
    FrozenPackedTensorField,
    FrozenPackedTensorLayout,
    PackedLayoutId,
    StablePackedStateViewArena,
)


def _tensor_only_consumer(
    packed_source: torch.Tensor,
    first_row_ids: torch.Tensor,
    lease_valid_mask: torch.Tensor,
    root_page_ids: torch.Tensor,
    state_pool: torch.Tensor,
) -> torch.Tensor:
    source_rows = packed_source.index_select(0, first_row_ids)
    state_rows = state_pool.index_select(0, root_page_ids[:, 0])
    return (
        source_rows + state_rows
    ) * lease_valid_mask.unsqueeze(1).to(dtype=source_rows.dtype)


class StablePackedStateUnitTest(unittest.TestCase):
    def test_fixed_and_fragmented_roots_share_one_view_consumer(self) -> None:
        resource_text = "test.request_state"
        resource = StateResourceId(resource_text)
        logical = RequestSlotArena(
            resource=resource_text,
            key_fields=(RequestKeyField.REQUEST_ID,),
            capacity=2,
            epoch=3,
            device=torch.device("cpu"),
        )
        request_a = RequestIdentity("a", 0, 0)
        request_b = RequestIdentity("b", 1, 0)
        admission = logical.plan_admission(
            (request_a, request_a, request_b),
            (0, 2, 3),
        )
        acquisition = logical.commit_admission(admission)
        logical.consume_receipt(acquisition.admission)
        selection = acquisition.selection

        state_layout = FrozenTensorLayout(
            StateLayoutId(4),
            1,
            (
                FrozenTensorField(
                    TensorFieldId(0),
                    (2,),
                    torch.float32,
                ),
            ),
        )
        fixed_roots = PagedStateVersionArena(
            PageBundle(
                state_layout,
                capacity=3,
                device=torch.device("cpu"),
            ),
            bundle_epoch=3,
        )
        fragmented_roots = PagedStateVersionArena(
            PageBundle(
                state_layout,
                capacity=4,
                device=torch.device("cpu"),
            ),
            bundle_epoch=3,
        )
        refs = tuple(
            StateVersionRef(
                StateVersionLineage(resource, 3, lease.slot),
                version,
            )
            for lease, version in zip(
                selection.leases,
                selection.before_versions,
                strict=True,
            )
        )
        for index, ref in enumerate(refs):
            fixed_write = fixed_roots.prepare_admission(ref)
            fixed_write.target_pages[0][0].fill_(float(index + 1))
            fixed_roots.publish(fixed_write, NoCompletionFence())

        dummy_ref = StateVersionRef(
            StateVersionLineage(
                resource,
                3,
                StateSlotRef(99, 1),
            ),
            0,
        )
        dummy_write = fragmented_roots.prepare_admission(dummy_ref)
        dummy_write.target_pages[0][0].fill_(99.0)
        fragmented_roots.publish(dummy_write, NoCompletionFence())
        for index, ref in enumerate(refs):
            fragmented_write = fragmented_roots.prepare_admission(ref)
            fragmented_write.target_pages[0][0].fill_(float(index + 1))
            fragmented_roots.publish(
                fragmented_write,
                NoCompletionFence(),
            )

        packed_layout = FrozenPackedTensorLayout(
            PackedLayoutId(8),
            row_capacity=4,
            lease_capacity=3,
            fields=(
                FrozenPackedTensorField(
                    TensorFieldId(0),
                    (2,),
                    torch.float32,
                ),
                FrozenPackedTensorField(
                    TensorFieldId(1),
                    (2,),
                    torch.float32,
                ),
            ),
        )
        fixed_views = StablePackedStateViewArena(
            fixed_roots,
            resource=resource,
            state_epoch=3,
            view_epoch=5,
            layout=packed_layout,
            buffer_count=1,
        )
        fragmented_views = StablePackedStateViewArena(
            fragmented_roots,
            resource=resource,
            state_epoch=3,
            view_epoch=5,
            layout=packed_layout,
            buffer_count=1,
        )
        source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        target = source + 20.0
        fixed_view = fixed_views.create_synchronous(
            selection,
            logical.selection_versions(selection),
            (source, target),
        )
        fragmented_view = fragmented_views.create_synchronous(
            selection,
            logical.selection_versions(selection),
            (source, target),
        )

        torch.testing.assert_close(
            fixed_view.packed_tensors.tensors[0],
            torch.tensor(
                (
                    (0.0, 1.0),
                    (4.0, 5.0),
                    (6.0, 7.0),
                    (0.0, 0.0),
                )
            ),
        )
        torch.testing.assert_close(
            fixed_view.row_slot_ids,
            torch.tensor((0, 0, 1, 0)),
        )
        torch.testing.assert_close(
            fixed_view.lease_slot_ids,
            torch.tensor((0, 1, 0)),
        )
        torch.testing.assert_close(
            fixed_view.first_row_ids,
            torch.tensor((0, 2, 0)),
        )
        torch.testing.assert_close(
            fixed_view.valid_masks.rows,
            torch.tensor((True, True, True, False)),
        )
        torch.testing.assert_close(
            fixed_view.valid_masks.leases,
            torch.tensor((True, True, False)),
        )
        torch.testing.assert_close(
            fixed_view.version_epoch.page_ids,
            torch.tensor(((0,), (1,), (0,))),
        )
        torch.testing.assert_close(
            fragmented_view.version_epoch.page_ids,
            torch.tensor(((1,), (2,), (0,))),
        )

        fixed_result = _tensor_only_consumer(
            fixed_view.packed_tensors.tensors[0],
            fixed_view.first_row_ids,
            fixed_view.valid_masks.leases,
            fixed_view.version_epoch.page_ids,
            fixed_view.version_epoch.state_operands[0],
        )
        fragmented_result = _tensor_only_consumer(
            fragmented_view.packed_tensors.tensors[0],
            fragmented_view.first_row_ids,
            fragmented_view.valid_masks.leases,
            fragmented_view.version_epoch.page_ids,
            fragmented_view.version_epoch.state_operands[0],
        )
        torch.testing.assert_close(fixed_result, fragmented_result)

        stable_address = fixed_view.packed_tensors.tensors[0].data_ptr()
        fixed_views.complete_synchronous(fixed_view.completion_fence)
        fragmented_views.complete_synchronous(
            fragmented_view.completion_fence
        )
        replay_view = fixed_views.create_synchronous(
            selection,
            logical.selection_versions(selection),
            (source + 1.0, target + 1.0),
        )
        self.assertEqual(
            replay_view.packed_tensors.tensors[0].data_ptr(),
            stable_address,
        )
        fixed_views.complete_synchronous(replay_view.completion_fence)

        for ref in refs:
            fixed_roots.retire(ref)
            fragmented_roots.retire(ref)
        fragmented_roots.retire(dummy_ref)
        logical.release_selection(selection)


if __name__ == "__main__":
    unittest.main()
