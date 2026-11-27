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
    assert arena.active_states == 2  # retired roots retain storage, not the admission budget
    replacement = arena.publish(arena.prepare_admission(("dense_fast_weights",)))
    with pytest.raises(RuntimeError):
        arena.prepare_admission(("dense_fast_weights",) * 2)
    arena.wait_release(arena.release_pin(pin))
    arena.retire(replacement.after_refs)
    assert arena.active_states == 0 and arena.free_pages == 6


def test_versions_stack_until_pool_exhaustion_and_pins_keep_their_version() -> None:
    arena = page_arena()
    write = arena.prepare_admission(("dense_fast_weights",) * 2)
    for index in range(2):
        arena.write_field(write, index, F0).fill_(0)
    published = arena.publish(write)
    current, idle = published.after_refs
    retained = []
    sequence = []
    for value in range(5):
        held = arena.pin_exact(StateRefs.of((current,)))
        retained.append(held)
        sequence.append(
            (
                published.operation_id,
                held.pin_id,
                current.version_id,
                int(arena.slot_matrix(held, held.refs)[1][0, 0]),
                arena.root(current).pages[0].generation,
            )
        )
        borrowed = arena.borrow_field(held, current, F0)
        torch.testing.assert_close(borrowed, torch.full_like(borrowed, float(value)))
        if value < 4:
            update = arena.prepare_transition(StateRefs.of((current,)), fields=((F0,),), copy_fields=())
            arena.write_field(update, 0, F0).fill_(value + 1)
            published = arena.publish(update)
            (current,) = published.after_refs
    # Recorded from the Python ledger: exact publication/pin/version/slot/generation order.
    assert sequence == [(1, 1, 0, 0, 0), (2, 2, 1, 2, 0), (3, 3, 2, 3, 0), (4, 4, 3, 4, 0), (5, 5, 4, 5, 0)]
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
    assert arena.active_states == 0 and arena.free_pages == 6


def test_multi_lineage_admission_failures_roll_back_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    arena = page_arena(heterogeneous=True)
    capacity = arena.free_pages
    with pytest.raises(RuntimeError):
        arena.prepare_admission(("dense_fast_weights",) * 3)
    assert arena.free_pages == capacity and arena.active_states == 0
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
    assert arena.free_pages == capacity and arena.active_states == 0


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
    assert arena.active_states == 0


def test_deferred_overwrite_preserves_pins_abort_and_unvisited_fields() -> None:
    arena = page_arena()
    refs = _admit(arena, 1)
    pin = arena.pin_exact(refs)
    original = arena.snapshot(pin, refs[0])
    resource, before = arena.slot_matrix(pin, refs)
    for abort in (True, False):
        write = arena.prepare_transition(refs, copy_fields="defer")
        _, after = arena.slot_matrix(write, write.after_refs)
        deferred = write.deferred
        assert deferred is not None
        src, dst = int(before[0, 0]), int(after[0, 0])
        assert src != dst
        assert deferred.slots(resource, 0, dst, False) == (src, -1)
        assert deferred.slots(resource, 0, dst, True) == (src, dst)
        arena.field_pool(resource, F0)[dst].fill_(23)
        assert deferred.slots(resource, 0, dst, True) == (dst, dst)
        # Poison an unvisited destination: publishing a partial projection must inherit its source.
        arena.field_pool(resource, LayoutFieldId(1))[after[0, 1]].fill_(99)
        if abort:
            arena.abort(write)
            assert arena.latest(refs[0].lineage_id) == refs[0]
        else:
            receipt = arena.publish(write)
            current = arena.pin_exact(receipt.after_refs)
            actual = arena.snapshot(current, current.refs[0])
            torch.testing.assert_close(actual[0], torch.full_like(actual[0], 23))
            torch.testing.assert_close(actual[1:], original[1:])
            arena.wait_release(arena.release_pin(current))
            arena.retire(receipt.after_refs)
        torch.testing.assert_close(arena.snapshot(pin, pin.refs[0]), original)
    arena.wait_release(arena.release_pin(pin))


def test_clock_rotates_across_fenced_slots_and_skips_pins() -> None:
    from pllm import _core

    ledger = _core.SlotLedger([3])
    events = np.empty(0, dtype=np.int64)

    def allocate(count):
        slots, _, waits, _ = ledger.allocate(0, count, events)
        return slots.tolist(), waits.tolist()

    def release(slot, event):
        ledger.release(np.array([[slot]], dtype=np.int64), event, events)

    for event, expected in enumerate((0, 1, 2, 0), 1):
        assert allocate(0) == ([], [])
        assert allocate(1) == ([expected], [1] if event == 4 else [])
        release(expected, event)
    assert allocate(2) == ([1, 2], [2, 3])
    ledger.hold(np.array([[1]], dtype=np.int64))
    release(1, 0)
    release(2, 5)
    # Slot 1 stays pinned while the hand wraps across it.
    assert allocate(1) == ([0], [4])
    release(0, 6)
    assert allocate(1) == ([2], [5])
    release(2, 0)
    # Hand is at 0, fenced. The later ready slot wins even in a mixed batch.
    assert allocate(2) == ([2, 0], [6])
    release(0, 0)
    release(1, 0)
    assert allocate(1) == ([1], [])
