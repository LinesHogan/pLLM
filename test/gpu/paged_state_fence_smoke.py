#!/usr/bin/env python3
"""One CUDA oracle for stable-view readiness and completion-gated reclaim."""

from __future__ import annotations

import json

import torch

from tllm.contracts.request_identity import (
    RequestIdentity,
    RequestKeyField,
)
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
from tllm.runtime.request_slot_adapter import CudaCompletionFence
from tllm.runtime.stable_packed_state import (
    FrozenPackedTensorField,
    FrozenPackedTensorLayout,
    PackedLayoutId,
    StablePackedStateViewArena,
)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("paged-state fence smoke requires CUDA")
    device = torch.device("cuda:0")
    resource_text = "test.cuda_state"
    resource = StateResourceId(resource_text)
    logical = RequestSlotArena(
        resource=resource_text,
        key_fields=(RequestKeyField.REQUEST_ID,),
        capacity=1,
        epoch=5,
        device=device,
    )
    admission_plan = logical.plan_admission(
        (RequestIdentity("cuda", 0, 0),),
        (0,),
    )
    acquisition = logical.commit_admission(admission_plan)
    logical.consume_receipt(acquisition.admission)
    selection = acquisition.selection
    lease = selection.leases[0]
    lineage = StateVersionLineage(resource, 5, lease.slot)
    version_zero = StateVersionRef(lineage, 0)
    version_one = StateVersionRef(lineage, 1)

    state_layout = FrozenTensorLayout(
        StateLayoutId(11),
        1,
        (
            FrozenTensorField(
                TensorFieldId(0),
                (1,),
                torch.float32,
            ),
        ),
    )
    bundle = PageBundle(state_layout, capacity=2, device=device)
    roots = PagedStateVersionArena(bundle, bundle_epoch=5)
    packed_layout = FrozenPackedTensorLayout(
        PackedLayoutId(12),
        row_capacity=1,
        lease_capacity=1,
        fields=(
            FrozenPackedTensorField(
                TensorFieldId(0),
                (1,),
                torch.float32,
            ),
        ),
    )
    views = StablePackedStateViewArena(
        roots,
        resource=resource,
        state_epoch=5,
        view_epoch=6,
        layout=packed_layout,
        buffer_count=1,
    )
    writer = torch.cuda.Stream(device=device)
    reader = torch.cuda.Stream(device=device)

    root_admission = roots.prepare_admission(version_zero)
    root_zero_ready = torch.cuda.Event()
    with torch.no_grad(), torch.cuda.stream(writer):
        root_admission.target_pages[0][0].fill_(3.0)
        root_zero_ready.record(writer)
    roots.publish(
        root_admission,
        CudaCompletionFence(root_zero_ready),
    )
    packed_source = torch.full(
        (1, 1),
        7.0,
        dtype=torch.float32,
        device=device,
    )
    view = views.create_cuda(selection, (packed_source,), writer)

    transition = roots.prepare_transition_cuda(
        version_zero,
        version_one,
        writer,
    )
    root_one_ready = torch.cuda.Event()
    with torch.no_grad(), torch.cuda.stream(writer):
        transition.target_pages[0][0].copy_(
            transition.source_pages[0][0]
        )
        transition.target_pages[0][0].add_(1.0)
        root_one_ready.record(writer)
    roots.publish(
        transition,
        CudaCompletionFence(root_one_ready),
    )

    views.order_cuda(view, reader)
    observed = torch.empty((1,), dtype=torch.float32, device=device)
    state_pool = view.version_epoch.state_operands[0]
    with torch.cuda.stream(reader):
        selected_state = state_pool.index_select(
            0,
            view.version_epoch.page_ids[:, 0],
        )
        observed.copy_(
            view.packed_tensors.tensors[0][0] + selected_state[0]
        )
    views.complete_cuda(view.completion_fence, reader)
    free_before_completion: int = roots.free_pages
    buffers_before_completion: int = views.free_buffers
    if buffers_before_completion != 0 or free_before_completion != 0:
        raise AssertionError("view/root storage reclaimed before completion")

    reader.synchronize()
    root_one_ready.synchronize()
    roots.collect_completed()
    views.collect_completed()
    free_after_completion: int = roots.free_pages
    buffers_after_completion: int = views.free_buffers
    if buffers_after_completion != 1 or free_after_completion != 1:
        raise AssertionError("completed view/root storage was not reclaimed")
    if float(observed.item()) != 10.0:
        raise AssertionError("reader observed a mixed state version")
    roots.retire(version_one)
    logical.release_selection(selection)
    if roots.free_pages != 2:
        raise AssertionError("latest root did not retire cleanly")

    print(
        json.dumps(
            {
                "completion_gated": True,
                "free_pages": roots.free_pages,
                "observed": float(observed.item()),
                "view_buffers": views.free_buffers,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
