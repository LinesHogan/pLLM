"""Evidence 13: actual CUDA PageBundle COW and exact event-gated reclamation."""

from dataclasses import replace

import pytest
import torch

from tllm.canonical.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw
from tllm.canonical.ptts.page_bundle import PageBundleArena
from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts
from tllm.canonical.workloads.esamp_request import ESampRequestWorkload
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions, ESampRequestConfig, ESampRequestPackedHiddenBindings,
    ESampRequestSpecialization,
)
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.semantic_graph import AnchorSymbol, SchedulerTimeline, TimelineAnchor
from tllm.contracts.state_representation import StateDeviceKind
from tllm.contracts.tensor_layout import LayoutFieldId, TensorDType


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ptts_waits_for_exact_fence_before_reclaim() -> None:
    bindings = ESampRequestPackedHiddenBindings(*(PackedHiddenResourceBinding(str(layer),
        HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]") for layer in (0, -1)))
    compiled = ESampRequestWorkload(ESampRequestConfig(rank=2)).compile(
        ESampRequestSpecialization(4, TensorDType.FLOAT32, 7, bindings, StateDeviceKind.CUDA, 0),
        ESampRequestCompileOptions(SchedulerTimeline(tuple(TimelineAnchor(AnchorSymbol(name), index)
            for index, name in enumerate(("model.execute", "sampler.pre_draw", "token.commit", "next.prepare"))),
            ("compute/cuda@1", "state/dense_page_cuda@1")),
            (PageBundleProviderFacts("test.page_bundle@1", 3, StateDeviceKind.CUDA, 0),), 2, 10000),
    )
    arena = PageBundleArena(compiled.plan_bundle)
    device = torch.device("cuda", 0)
    writer, reader = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    write = arena.prepare_admission(("request_distiller",) * 2, writer)
    arena.abort(write)
    assert arena.free_pages == 6
    write = arena.prepare_admission(("request_distiller",) * 2, writer)
    roots = arena.writable_roots(write)
    with torch.cuda.stream(writer):
        initial = tuple(torch.arange(root[0].numel(), device=device, dtype=torch.float32) / 10 + index
            for index, root in enumerate(roots))
        for root, value in zip(roots, initial, strict=True):
            root[0].copy_(value)
    admitted = arena.publish(write)
    arena.settle(admitted)
    pin = arena.pin_exact(admitted.after_refs, reader)
    borrowed = arena.borrow_field(pin, admitted.after_refs[0], LayoutFieldId(0))
    original = arena.snapshot(pin, admitted.after_refs[0])
    update = arena.prepare_transition(admitted.after_refs, writer)
    fields = AdamWFields(*(LayoutFieldId(index) for index in range(5)))
    with torch.cuda.stream(writer):
        gradients = tuple(torch.full_like(value, gradient) for value, gradient in zip(initial, (0.25, -0.5), strict=True))
        apply_adamw(arena.writable_roots(update), fields, AdamWConfig(0.01, (0.9, 0.999), 1e-8, 0.0), gradients)
    publication = arena.publish(update)
    arena.settle(publication)
    updated = arena.pin_exact(publication.after_refs, reader)
    result = arena.snapshot(updated, publication.after_refs[0])
    # Old roots must remain live while their original consumer stream still runs.
    with torch.cuda.stream(reader):
        torch.cuda._sleep(100_000_000)
        consumed = borrowed.clone()
    release = arena.release_pin(pin)
    assert arena.pending_pin_releases == 1
    assert arena.free_pages == 2
    with pytest.raises(RuntimeError, match="no new reads"):
        arena.snapshot(pin, admitted.after_refs[0])
    with pytest.raises(RuntimeError, match="issued"):
        arena.wait_release(replace(release))
    arena.wait_release(release)
    assert arena.free_pages == 4
    arena.require_stable()
    torch.testing.assert_close(original[0], initial[0])
    torch.testing.assert_close(consumed, initial[0])
    torch.testing.assert_close(result[0], initial[0] - 0.01)
    assert result[2].item() == 1
    arena.retire(publication.after_refs)
    arena.wait_release(arena.release_pin(updated))
    assert arena.free_pages == 6 and arena.active_lineages == 0
