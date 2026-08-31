#!/usr/bin/env python3
"""One CUDA oracle for root readiness and completion-gated reclaim."""

from __future__ import annotations

import json

import torch

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
from tllm.runtime.request_slot_adapter import CudaCompletionFence


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("paged-state fence smoke requires CUDA")
    device = torch.device("cuda:0")
    layout = FrozenTensorLayout(
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
    bundle = PageBundle(layout, capacity=2, device=device)
    arena = PagedStateVersionArena(bundle, bundle_epoch=5)
    lineage = StateVersionLineage(
        StateResourceId("test.cuda_state"),
        5,
        StateSlotRef(0, 1),
    )
    version_zero = StateVersionRef(lineage, 0)
    version_one = StateVersionRef(lineage, 1)
    writer = torch.cuda.Stream(device=device)
    reader = torch.cuda.Stream(device=device)

    delay_left = torch.ones(
        (4096, 4096),
        dtype=torch.float16,
        device=device,
    )
    delay_right = torch.ones_like(delay_left)
    delay_output = torch.empty_like(delay_left)
    torch.mm(delay_left, delay_right, out=delay_output)
    torch.cuda.synchronize(device)

    admission = arena.prepare_admission(version_zero)
    root_zero_ready = torch.cuda.Event()
    with torch.no_grad(), torch.cuda.stream(writer):
        admission.target_pages[0][0].fill_(3.0)
        root_zero_ready.record(writer)
    root_zero = arena.publish(
        admission,
        CudaCompletionFence(root_zero_ready),
    )
    pin = arena.pin_exact_cuda((version_zero,), reader)

    transition = arena.prepare_transition_cuda(
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
    arena.publish(
        transition,
        CudaCompletionFence(root_one_ready),
    )

    observed = torch.empty((1,), dtype=torch.float32, device=device)
    completion = torch.cuda.Event()
    state_pool = bundle.capture_operands[0]
    with torch.cuda.stream(reader):
        observed.copy_(state_pool[root_zero.page_ids[0]])
        torch.mm(delay_left, delay_right, out=delay_output)
        completion.record(reader)
    arena.release_pin(pin, CudaCompletionFence(completion))
    if completion.query():
        raise AssertionError("completion fence unexpectedly finished synchronously")
    free_before_completion: int = arena.free_pages
    if free_before_completion != 0:
        raise AssertionError("old root reclaimed before reader completion")

    completion.synchronize()
    root_one_ready.synchronize()
    arena.collect_completed()
    free_after_completion: int = arena.free_pages
    if free_after_completion != 1:
        raise AssertionError("completed old root was not reclaimed")
    if float(observed.item()) != 3.0:
        raise AssertionError("reader observed a mutated version-zero root")
    arena.retire(version_one)
    free_after_retire: int = arena.free_pages
    if free_after_retire != 2:
        raise AssertionError("latest root did not retire cleanly")

    print(
        json.dumps(
            {
                "completion_gated": True,
                "old_value": float(observed.item()),
                "free_pages": arena.free_pages,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
