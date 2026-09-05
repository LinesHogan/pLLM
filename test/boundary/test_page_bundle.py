"""PTTS evidence 9/10/12: state values, atomic failure and issued generations."""

from dataclasses import replace

import pytest
import torch

from tllm.canonical.ptts.page_bundle import PageBundleArena
from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts
from tllm.canonical.runtime_facts import CanonicalRuntimeFacts
from tllm.canonical.scheduler import compile_schedule, lower_physical_plan
from tllm.canonical.state_representation import compile_state_representation, lower_state_physical_plan
from tllm.canonical.tensor_layout import intern_tensor_layouts
from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTCompileOptions, InPlaceTTTConfig, InPlaceTTTSpecialization,
    InPlaceTTTTargetSpecialization, InPlaceTTTWorkload,
)
from tllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from tllm.contracts.semantic_graph import StateResource, SemanticState, StateScope
from tllm.contracts.state_representation import StateLayoutRequirement, StateRepresentationRequest
from tllm.contracts.tensor_layout import LayoutFieldId, StateTensorLayout, TensorDType, TensorLayoutRequest


def _arena(*, heterogeneous: bool = False) -> PageBundleArena:
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    compiled = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2)).compile(
        InPlaceTTTSpecialization((InPlaceTTTTargetSpecialization(target, 3, 2),),
            TensorDType.FLOAT32, torch.device("cpu")),
        InPlaceTTTCompileOptions(
            CanonicalRuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",)),
            (PageBundleProviderFacts("test.page_bundle@1", 3),), 2, 4096,
        ),
    )
    plan = compiled.plan_bundle
    if heterogeneous:
        original, = plan.tensor_layout_request.layouts
        layout = replace(original.layout, namespace="test.global_factors@1", fields=(
            replace(original.layout.fields[0], shape=(2, 5)),
        ))
        layouts = TensorLayoutRequest((original, StateTensorLayout("global_factors", layout)))
        bindings = intern_tensor_layouts(layouts)
        graph = replace(plan.semantic_graph, states=(*plan.semantic_graph.states,
            SemanticState(StateResource("global_factors"), StateScope.INSTANCE, ())))
        schedule = compile_schedule(graph, plan.timeline)
        facts = PageBundleProviderFacts("test.page_bundle@1", 3)
        candidates = tuple(facts.materialize_plan(resource=entry.resource, layout=entry.layout,
            binding=binding, max_admitted_lineages=2)
            for entry, binding in zip(layouts.layouts, bindings.bindings, strict=True))
        request = StateRepresentationRequest(4096, tuple(StateLayoutRequirement(
            binding.resource, binding.layout_id, binding.layout_namespace, binding.layout_digest, 2,
        ) for binding in bindings.bindings), candidates)
        decision = compile_state_representation(request)
        plan = replace(plan, semantic_graph=graph, schedule_decision=schedule,
            schedule_physical=lower_physical_plan(graph, plan.timeline, schedule, plan_epoch=1),
            state_request=request, state_decision=decision,
            state_physical=lower_state_physical_plan(request, decision, plan_epoch=1),
            tensor_layout_request=layouts, tensor_layout_decision=bindings)
    return PageBundleArena(plan)


def test_cpu_request_ptts_admit_cow_publish_pin_retire() -> None:
    arena = _arena()
    write = arena.prepare_admission(("dense_fast_weights", "dense_fast_weights"))
    for index in range(2):
        for field in arena.plan.tensor_layout_request.layouts[0].layout.fields:
            assert arena.write_field(write, index, field.field_id).data_ptr() % field.alignment_bytes == 0
        arena.write_field(write, index, LayoutFieldId(0)).fill_(index + 1)
    admitted = arena.publish(write)
    a, b = admitted.after_refs
    arena.settle(admitted)
    pin = arena.pin_exact((a, b))
    old_a, old_b = arena.snapshot(pin, a), arena.snapshot(pin, b)
    update = arena.prepare_transition((a,), fields=((LayoutFieldId(0),),))
    arena.write_field(update, 0, LayoutFieldId(0)).add_(4)
    with pytest.raises(RuntimeError, match="writable"):
        arena.write_field(update, 0, LayoutFieldId(1))
    receipt = arena.publish(update)
    new_a, = receipt.after_refs
    assert new_a.version_id == 1
    assert arena.latest(a.lineage_id) == new_a
    assert arena.latest(b.lineage_id) == b
    assert arena.root(a).pages[0] != arena.root(new_a).pages[0]
    assert arena.root(a).pages[1:] == arena.root(new_a).pages[1:]
    new_pin = arena.pin_exact((new_a,))
    torch.testing.assert_close(arena.snapshot(new_pin, new_a)[0], old_a[0] + 4)
    torch.testing.assert_close(arena.snapshot(pin, a)[0], old_a[0])
    torch.testing.assert_close(arena.snapshot(pin, b)[0], old_b[0])
    arena.settle(receipt)
    arena.retire((new_a, b))
    assert arena.active_lineages == 2
    arena.wait_release(arena.release_pin(pin))
    arena.wait_release(arena.release_pin(new_pin))
    assert arena.active_lineages == 0
    assert arena.free_pages == 6


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
    with pytest.raises(ValueError, match="layout"):
        arena.copy_roots(write, ((torch.ones(99),), (torch.ones(99),)))
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


def test_generation_receipt_blocks_aba_and_double_settlement() -> None:
    arena = _arena()
    write = arena.prepare_admission(("dense_fast_weights",))
    with pytest.raises(RuntimeError, match="issued"):
        arena.publish(replace(write))
    publication = arena.publish(write)
    old_ref, = publication.after_refs
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
    ref, = replacement.after_refs
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
