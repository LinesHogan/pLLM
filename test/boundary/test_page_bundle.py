"""PTTS evidence 9/10/12: copy-on-write versions, exact pins, atomic failure and reclamation."""

import numpy as np
import pytest
import torch

from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.state_refs import StateRefs
from test.state_fixtures import page_arena

F0 = LayoutFieldId(0)


def _admit(arena, count: int = 2):
    write = arena.prepare_admission(("dense_fast_weights",) * count)
    for index in range(count):
        arena.write_field(write, index, F0).fill_(index + 1)
    receipt = arena.publish(write)
    return receipt.after_refs


def test_copy_on_write_transition_keeps_the_pinned_version_unchanged() -> None:
    arena = page_arena()
    a, b = _admit(arena)
    pin = arena.pin_exact(StateRefs.of((a, b)))
    (old_a0,) = arena.snapshot(pin, a)[:1]
    update = arena.prepare_transition(StateRefs.of((a,)), fields=((F0,),))
    arena.write_field(update, 0, F0).add_(4)
    with pytest.raises(AssertionError):
        arena.write_field(update, 0, LayoutFieldId(1))
    receipt = arena.publish(update)
    (new_a,) = receipt.after_refs
    assert new_a.version_id == 1
    assert arena.latest(a.lineage_id) == new_a and arena.latest(b.lineage_id) == b
    new_pin = arena.pin_exact(StateRefs.of((new_a,)))
    resource, slots = arena.slot_matrix(pin, pin.refs)
    assert resource == "dense_fast_weights"
    selected = pin.refs.take(np.array([1, 0, 1]))
    np.testing.assert_array_equal(arena.slot_matrix(pin, selected)[1], slots[[1, 0, 1]])
    with pytest.raises(KeyError):
        arena.slot_matrix(pin, new_pin.refs)
    both = arena.pin_exact(StateRefs.of((a, new_a)))
    _, versions = arena.slot_matrix(both, both.refs.take(np.array([1, 0])))
    assert versions[0, 0] != versions[1, 0]
    arena.wait_release(arena.release_pin(both))

    # Unwritten fields share storage; the written field got fresh pages.
    assert (
        arena.borrow_field(new_pin, new_a, LayoutFieldId(1)).data_ptr()
        == arena.borrow_field(pin, a, LayoutFieldId(1)).data_ptr()
    )
    assert arena.borrow_field(new_pin, new_a, F0).data_ptr() != arena.borrow_field(pin, a, F0).data_ptr()
    # An exact pin sees only its own version.
    with pytest.raises(KeyError):
        arena.borrow_field(pin, new_a, F0)
    torch.testing.assert_close(arena.snapshot(pin, a)[0], old_a0)
    torch.testing.assert_close(arena.snapshot(new_pin, new_a)[0], old_a0 + 4)


def test_retired_roots_hold_storage_until_pages_are_free() -> None:
    arena = page_arena()
    a, b = _admit(arena)
    pin = arena.pin_exact(StateRefs.of((a, b)))
    arena.retire(StateRefs.of((a, b)))
    assert arena.active_lineages == 2  # retired roots retain storage, not the admission entitlement
    replacement = arena.publish(arena.prepare_admission(("dense_fast_weights",)))
    with pytest.raises(RuntimeError):
        arena.prepare_admission(("dense_fast_weights",) * 2)
    arena.wait_release(arena.release_pin(pin))
    arena.retire(replacement.after_refs)
    assert arena.active_lineages == 0 and arena.free_pages == 6


def test_versions_stack_until_pool_exhaustion_and_pins_keep_their_version() -> None:
    arena = page_arena()
    write = arena.prepare_admission(("dense_fast_weights",) * 2)
    for index in range(2):
        arena.write_field(write, index, F0).fill_(0)
    published = arena.publish(write)
    current, idle = published.after_refs
    retained = []
    for value in range(5):
        held = arena.pin_exact(StateRefs.of((current,)))
        retained.append(held)
        borrowed = arena.borrow_field(held, current, F0)
        torch.testing.assert_close(borrowed, torch.full_like(borrowed, float(value)))
        if value < 4:
            update = arena.prepare_transition(StateRefs.of((current,)), fields=((F0,),), copy_fields=())
            arena.write_field(update, 0, F0).fill_(value + 1)
            published = arena.publish(update)
            (current,) = published.after_refs
    assert current.version_id == 4 and arena.free_pages == 0
    with pytest.raises(RuntimeError):
        arena.prepare_transition(StateRefs.of((current,)), fields=((F0,),))
    for value, held in enumerate(retained):
        borrowed = arena.borrow_field(held, held.refs[0], F0)
        torch.testing.assert_close(borrowed, torch.full_like(borrowed, float(value)))
    arena.wait_release(arena.release_pin(retained.pop(0)))
    update = arena.prepare_transition(StateRefs.of((current,)), fields=((F0,),))
    arena.abort(update)
    assert arena.free_pages == 1
    for held in retained:
        arena.wait_release(arena.release_pin(held))
    arena.retire(StateRefs.of((current, idle)))
    assert arena.active_lineages == 0 and arena.free_pages == 6


def test_multi_lineage_admission_failures_roll_back_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    arena = page_arena(heterogeneous=True)
    capacity = arena.free_pages
    with pytest.raises(RuntimeError):
        arena.prepare_admission(("dense_fast_weights",) * 3)
    assert arena.free_pages == capacity and arena.active_lineages == 0
    original_fill = torch.Tensor.index_fill_

    def fail_second_resource(tensor: torch.Tensor, *args: object) -> torch.Tensor:
        if tensor.shape[1:] == (2, 5):
            raise RuntimeError("injected initialization failure")
        return original_fill(tensor, *args)

    # Inject a torch failure so admission dies after the first resource is written.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "index_fill_", fail_second_resource)
        with pytest.raises(RuntimeError):
            arena.prepare_admission(("dense_fast_weights", "global_factors"))
    assert arena.free_pages == capacity and arena.active_lineages == 0


def test_multi_resource_update_is_atomic_and_write_handles_are_single_use() -> None:
    arena = page_arena(heterogeneous=True)
    admitted = arena.publish(arena.prepare_admission(("dense_fast_weights", "global_factors")))
    refs = admitted.after_refs
    pin = arena.pin_exact(refs)
    before = tuple(arena.snapshot(pin, ref) for ref in refs)
    update = arena.prepare_transition(refs)
    arena.write_field(update, 0, F0).fill_(99)
    arena.abort(update)
    assert tuple(arena.latest(ref.lineage_id) for ref in refs) == tuple(refs)
    for ref, expected in zip(refs, before, strict=True):
        for actual, value in zip(arena.snapshot(pin, ref), expected, strict=True):
            torch.testing.assert_close(actual, value)
    with pytest.raises(AssertionError):
        arena.publish(update)
    update = arena.prepare_transition(refs)
    for index in range(2):
        arena.write_field(update, index, F0).fill_(index + 5)
    published = arena.publish(update)
    assert tuple(arena.latest(ref.lineage_id) for ref in refs) == tuple(published.after_refs)
    new_pin = arena.pin_exact(published.after_refs)
    for index, ref in enumerate(published.after_refs):
        actual = arena.snapshot(new_pin, ref)[0]
        torch.testing.assert_close(actual, torch.full_like(actual, index + 5))
    arena.wait_release(arena.release_pin(new_pin))
    arena.wait_release(arena.release_pin(pin))
    arena.retire(published.after_refs)
    assert arena.active_lineages == 0
