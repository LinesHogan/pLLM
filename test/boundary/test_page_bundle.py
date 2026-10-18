"""PTTS evidence 9/10/12: state values, atomic failure and issued generations."""

from dataclasses import replace

import pytest
import torch

from pllm.authoring import BackendCompileFacts, ParameterTargetShape
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from pllm.contracts.semantic_graph import SemanticState, StateResource, StateScope
from pllm.contracts.state_representation import StateDeviceKind, StateLayoutRequirement, StateRepresentationRequest
from pllm.contracts.tensor_layout import LayoutFieldId, StateTensorLayout, TensorDType, TensorLayoutRequest
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.scheduler import compile_state_schedule, intern_tensor_layouts
from pllm.workloads.inplace_ttt import (
    InPlaceTTTConfig,
    InPlaceTTTWorkload,
)


def _arena(*, heterogeneous: bool = False) -> PageBundleArena:
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    facts = BackendCompileFacts(
        BackendPlacementProbe(
            "test.page_bundle@1",
            "a" * 64,
            "b" * 64,
            "c" * 64,
            AttachmentTopology.IN_PROCESS,
            AttachmentGraphMode.EAGER,
            StateDeviceKind.CPU,
            0,
            1,
            1,
            TensorDType.FLOAT32,
            3,
            8,
            (),
            (),
        ),
        RuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",)),
        (PageBundleProviderFacts("test.page_bundle@1", 6),),
        2,
        4096,
        0,
        (ParameterTargetShape(target, 3, 2),),
    )
    plan = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2)).plan_backend(facts).inputs.plan
    if heterogeneous:
        (original,) = plan.tensor_layout_request.layouts
        layout = replace(
            original.layout,
            namespace="test.global_factors@1",
            fields=(replace(original.layout.fields[0], shape=(2, 5)),),
        )
        layouts = TensorLayoutRequest((original, StateTensorLayout("global_factors", layout)))
        bindings = intern_tensor_layouts(layouts)
        graph = replace(
            plan.semantic_graph,
            states=(
                *plan.semantic_graph.states,
                SemanticState(StateResource("global_factors"), StateScope.INSTANCE, ()),
            ),
        )
        facts = PageBundleProviderFacts("test.page_bundle@1", 6)
        candidates = tuple(
            facts.materialize_plan(
                resource=entry.resource, layout=entry.layout, binding=binding, max_admitted_lineages=2
            )
            for entry, binding in zip(layouts.layouts, bindings.bindings, strict=True)
        )
        request = StateRepresentationRequest(
            4096,
            tuple(
                StateLayoutRequirement(
                    binding.resource,
                    binding.layout_id,
                    binding.layout_namespace,
                    binding.layout_digest,
                    2,
                )
                for binding in bindings.bindings
            ),
            candidates,
        )
        decision, state_physical, schedule, physical = compile_state_schedule(graph, plan.timeline, request)
        plan = replace(
            plan,
            semantic_graph=graph,
            schedule_decision=schedule,
            schedule_physical=physical,
            state_request=request,
            state_decision=decision,
            state_physical=state_physical,
            tensor_layout_request=layouts,
            tensor_layout_decision=bindings,
        )
    return PageBundleArena(plan)


def test_cpu_request_ptts_admit_cow_publish_pin_retire() -> None:
    arena = _arena()
    write = arena.prepare_admission(("dense_fast_weights", "dense_fast_weights"))
    field_count = len(arena.plan.tensor_layout_request.layouts[0].layout.fields)
    prepared_usage = arena.usage()
    assert prepared_usage.private_versions == 2 and prepared_usage.private_field_pages == 2 * field_count
    assert prepared_usage.live_field_pages == prepared_usage.retiring_field_pages == 0
    assert prepared_usage.free_field_pages == 4 * field_count
    for index in range(2):
        for field in arena.plan.tensor_layout_request.layouts[0].layout.fields:
            assert arena.write_field(write, index, field.field_id).data_ptr() % field.alignment_bytes == 0
        arena.write_field(write, index, LayoutFieldId(0)).fill_(index + 1)
    admitted = arena.publish(write)
    a, b = admitted.after_refs
    arena.settle(admitted)
    pin = arena.pin_exact((a, b))
    borrowed = arena.borrow_field(pin, a, LayoutFieldId(0))
    assert arena.borrow_field(pin, a, LayoutFieldId(0)).data_ptr() == borrowed.data_ptr()
    with pytest.raises(RuntimeError, match="issued"):
        arena.borrow_field(replace(pin), a, LayoutFieldId(0))
    old_a, old_b = arena.snapshot(pin, a), arena.snapshot(pin, b)
    payload = sum(value.numel() * value.element_size() for value in old_a)
    usage = arena.usage()
    assert usage.logical_live_bytes == usage.allocated_payload_bytes == 2 * payload
    assert usage.reserved_bytes == sum(value.reserved_bytes for value in arena.plan.state_physical.representations)
    assert usage.reserved_padding_bytes == usage.reserved_bytes - 6 * payload
    assert usage.pinned_versions == 2 and usage.pending_receipts == 1 and usage.private_versions == 0
    free = arena.free_pages
    assert arena.usage().pending_receipts == 0 and usage.pending_receipts == 1
    for copied in (
        (LayoutFieldId(-1),),
        (LayoutFieldId(4),),
        (LayoutFieldId(0), LayoutFieldId(0)),
        (LayoutFieldId(1), LayoutFieldId(0)),
    ):
        with pytest.raises(ValueError, match="copied fields"):
            arena.prepare_transition((a,), copy_fields=copied)
    with pytest.raises(ValueError, match="copied fields"):
        arena.prepare_transition((a,), fields=((LayoutFieldId(0),),), copy_fields=(LayoutFieldId(1),))
    assert arena.free_pages == free
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        partial = arena.prepare_transition((a,), copy_fields=(LayoutFieldId(0),))
    assert sum(event.name == "aten::copy_" for event in profile.events()) == 1
    torch.testing.assert_close(arena.write_field(partial, 0, LayoutFieldId(0)), old_a[0])
    for index in range(1, len(old_a)):
        arena.write_field(partial, 0, LayoutFieldId(index)).fill_(99)
    arena.abort(partial)
    assert arena.free_pages == free
    update = arena.prepare_transition((a,), fields=((LayoutFieldId(0),),))
    arena.write_field(update, 0, LayoutFieldId(0)).add_(4)
    with pytest.raises(RuntimeError, match="writable"):
        arena.write_field(update, 0, LayoutFieldId(1))
    receipt = arena.publish(update)
    (new_a,) = receipt.after_refs
    assert new_a.version_id == 1
    assert arena.latest(a.lineage_id) == new_a
    assert arena.latest(b.lineage_id) == b
    assert arena.root(a).pages[0] != arena.root(new_a).pages[0]
    assert arena.root(a).pages[1:] == arena.root(new_a).pages[1:]
    new_pin = arena.pin_exact((new_a,))
    usage = arena.usage()
    assert usage.logical_live_bytes == 3 * payload
    assert usage.allocated_payload_bytes == 2 * payload + old_a[0].numel() * old_a[0].element_size()
    assert usage.live_field_pages == 2 * field_count and usage.retiring_field_pages == 1
    assert usage.retiring_versions == 1 and usage.pinned_versions == 3 and usage.pending_receipts == 1
    assert arena.borrow_field(new_pin, new_a, LayoutFieldId(0)).data_ptr() != borrowed.data_ptr()
    assert (
        arena.borrow_field(new_pin, new_a, LayoutFieldId(1)).data_ptr()
        == arena.borrow_field(pin, a, LayoutFieldId(1)).data_ptr()
    )
    with pytest.raises(RuntimeError, match="held"):
        arena.borrow_field(pin, new_a, LayoutFieldId(0))
    torch.testing.assert_close(borrowed, old_a[0])
    torch.testing.assert_close(arena.snapshot(new_pin, new_a)[0], old_a[0] + 4)
    torch.testing.assert_close(arena.snapshot(pin, a)[0], old_a[0])
    torch.testing.assert_close(arena.snapshot(pin, b)[0], old_b[0])
    arena.settle(receipt)
    arena.retire((new_a, b))
    assert arena.active_lineages == 2
    # Retired roots retain storage, not the signed admission entitlement.
    replacement = arena.publish(arena.prepare_admission(("dense_fast_weights",)))
    arena.settle(replacement)
    assert arena.active_lineages == 3
    with pytest.raises(RuntimeError, match="admission capacity"):
        arena.prepare_admission(("dense_fast_weights",) * 2)
    with pytest.raises(RuntimeError, match="live owners"):
        arena.prepare_admission(("dense_fast_weights",))
    assert arena.usage().admission_capacity_failures == 2
    torch.testing.assert_close(arena.borrow_field(pin, a, LayoutFieldId(0)), old_a[0])
    arena.wait_release(arena.release_pin(pin))
    with pytest.raises(RuntimeError, match="released"):
        arena.borrow_field(pin, a, LayoutFieldId(0))
    arena.wait_release(arena.release_pin(new_pin))
    arena.retire(replacement.after_refs)
    assert arena.active_lineages == 0
    assert arena.free_pages == 6
    usage = arena.usage()
    assert usage.logical_live_bytes == usage.allocated_payload_bytes == 0
    assert usage.free_field_pages == 6 * field_count
    assert usage.live_field_pages == usage.private_field_pages == usage.retiring_field_pages == 0
    assert usage.pinned_versions == usage.pending_receipts == 0

    # One hot lineage borrows the idle lineage's unused version capacity.
    admitted = arena.publish(arena.prepare_admission(("dense_fast_weights",) * 2))
    arena.settle(admitted)
    current, idle = admitted.after_refs
    addresses = arena.pool_addresses
    retained = []
    for value in range(5):
        held = arena.pin_exact((current,))
        retained.append(held)
        torch.testing.assert_close(
            arena.borrow_field(held, current, LayoutFieldId(0)), torch.full_like(old_a[0], value)
        )
        if value < 4:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                update = arena.prepare_transition((current,), fields=((LayoutFieldId(0),),), copy_fields=())
            assert not any(event.name == "aten::copy_" for event in profile.events())
            arena.write_field(update, 0, LayoutFieldId(0)).fill_(value + 1)
            published = arena.publish(update)
            arena.settle(published)
            (current,) = published.after_refs
    assert current.version_id == 4 and arena.free_pages == 0
    with pytest.raises(RuntimeError, match="live owners"):
        arena.prepare_transition((current,), fields=((LayoutFieldId(0),),))
    for value, held in enumerate(retained):
        torch.testing.assert_close(
            arena.borrow_field(held, held.refs[0], LayoutFieldId(0)), torch.full_like(old_a[0], value)
        )
    arena.wait_release(arena.release_pin(retained.pop(0)))
    update = arena.prepare_transition((current,), fields=((LayoutFieldId(0),),))
    arena.abort(update)
    assert arena.pool_addresses == addresses and arena.free_pages == 1
    for held in retained:
        arena.wait_release(arena.release_pin(held))
    arena.retire((current, idle))
    assert arena.active_lineages == 0 and arena.free_pages == 6


def test_multi_lineage_admission_and_publication_roll_back_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    arena = _arena(heterogeneous=True)
    resources = ("dense_fast_weights", "global_factors")
    capacity = arena.free_pages
    with pytest.raises(RuntimeError, match="capacity"):
        arena.prepare_admission(("dense_fast_weights",) * 3)
    assert arena.free_pages == capacity and arena.active_lineages == 0
    # Initialization fails after the first resource has already been written.
    original_zero = torch.Tensor.zero_

    def fail_second_resource(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape == (2, 5):
            raise RuntimeError("injected initialization failure")
        return original_zero(tensor)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "zero_", fail_second_resource)
        with pytest.raises(RuntimeError, match="initialization"):
            arena.prepare_admission(resources)
    assert arena.free_pages == capacity and arena.active_lineages == 0
    write = arena.prepare_admission(resources)
    arena.write_field(write, 0, LayoutFieldId(0)).fill_(7)
    with pytest.raises(RuntimeError, match="size"):
        arena.write_field(write, 1, LayoutFieldId(0)).copy_(torch.ones(99))
    arena.abort(write)
    assert arena.free_pages == capacity and arena.active_lineages == 0
    write = arena.prepare_admission(resources)
    admitted = arena.publish(write)
    arena.settle(admitted)
    refs = admitted.after_refs
    assert refs[0].layout_id != refs[1].layout_id
    pin = arena.pin_exact(refs)
    before = tuple(arena.snapshot(pin, ref) for ref in refs)
    update = arena.prepare_transition(refs)
    arena.write_field(update, 0, LayoutFieldId(0)).fill_(99)
    arena.abort(update)
    assert tuple(arena.latest(ref.lineage_id) for ref in refs) == refs
    for ref, expected in zip(refs, before, strict=True):
        for actual, value in zip(arena.snapshot(pin, ref), expected, strict=True):
            torch.testing.assert_close(actual, value)
    with pytest.raises(RuntimeError, match="consumed"):
        arena.publish(update)
    update = arena.prepare_transition(refs)
    for index in range(2):
        arena.write_field(update, index, LayoutFieldId(0)).fill_(index + 5)
    published = arena.publish(update)
    assert tuple(arena.latest(ref.lineage_id) for ref in refs) == published.after_refs
    new_pin = arena.pin_exact(published.after_refs)
    for index, ref in enumerate(published.after_refs):
        actual = arena.snapshot(new_pin, ref)[0]
        torch.testing.assert_close(actual, torch.full_like(actual, index + 5))
    arena.wait_release(arena.release_pin(new_pin))
    arena.settle(published)
    arena.wait_release(arena.release_pin(pin))
    arena.retire(published.after_refs)
    assert arena.free_pages == capacity and arena.active_lineages == 0
    with pytest.raises(ValueError, match="epochs"):
        PageBundleArena(replace(arena.plan, state_physical=replace(arena.plan.state_physical, plan_epoch=2)))
    with pytest.raises(ValueError, match="catalog"):
        PageBundleArena(arena.plan, local_candidates=arena.plan.state_request.candidates[:-1])
    with pytest.raises(ValueError):
        PageBundleArena(
            replace(
                arena.plan,
                state_physical=replace(
                    arena.plan.state_physical, reserved_bytes=arena.plan.state_physical.reserved_bytes + 1
                ),
            )
        )


def test_generation_receipt_blocks_aba_and_double_settlement() -> None:
    arena = _arena()
    write = arena.prepare_admission(("dense_fast_weights",))
    with pytest.raises(RuntimeError, match="issued"):
        arena.publish(replace(write))
    publication = arena.publish(write)
    (old_ref,) = publication.after_refs
    pages = arena.root(old_ref).pages
    with pytest.raises(RuntimeError, match="issued"):
        arena.settle(replace(publication))
    arena.settle(publication)
    with pytest.raises(RuntimeError, match="settled"):
        arena.settle(publication)
    pin = arena.pin_exact((old_ref,))
    with pytest.raises(RuntimeError, match="issued"):
        arena.release_pin(replace(pin))
    release = arena.release_pin(pin)
    arena.wait_release(release)
    arena.retire((old_ref,))
    replacement = arena.publish(arena.prepare_admission(("dense_fast_weights",)))
    (ref,) = replacement.after_refs
    assert ref.lineage_id != old_ref.lineage_id
    reused = arena.root(ref).pages
    assert tuple(page.slot_id for page in reused) == tuple(page.slot_id for page in pages)
    assert all(new.generation > old.generation for old, new in zip(pages, reused, strict=True))
    with pytest.raises(RuntimeError, match="reclaimed"):
        arena.root(old_ref)
    with pytest.raises(RuntimeError, match="released"):
        arena.release_pin(pin)
    with pytest.raises(RuntimeError, match="issued"):
        arena.wait_release(replace(release))
    arena.wait_release(release)
    assert arena.latest(ref.lineage_id) == ref
    arena.settle(replacement)
    arena.retire((ref,))
    assert arena.free_pages == 6 and arena.active_lineages == 0
    unclosed = arena.publish(arena.prepare_admission(("dense_fast_weights",)))
    arena.retire(unclosed.after_refs)
    assert arena.active_lineages == 1 and arena.free_pages == 5
    arena.settle(unclosed)
    assert arena.active_lineages == 0 and arena.free_pages == 6
