"""Transactional invariants of the PTTS-0 request-slot authority."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    RequestKeyField,
)
from tllm.runtime.request_slot_arena import RequestSlotArena


def test_aborted_admission_publishes_nothing_and_can_retry() -> None:
    arena = RequestSlotArena(
        resource="state",
        key_fields=(RequestKeyField.REQUEST_ID,),
        capacity=2,
        epoch=1,
        device=torch.device("cpu"),
    )
    identities = (RequestIdentity("first", 0, 0),)
    plan = arena.plan_admission(identities, (0,))
    assert arena.versions() == ()
    arena.abort_admission(plan, poison=False)

    retry = arena.plan_admission(identities, (0,))
    acquired = arena.commit_admission(retry)
    arena.consume_receipt(acquired.admission)
    assert tuple(version.version for version in arena.versions()) == (0,)
    arena.release_selection(acquired.selection)


def test_pending_transition_excludes_effects_and_abort_preserves_version() -> None:
    arena = RequestSlotArena(
        resource="state",
        key_fields=(RequestKeyField.REQUEST_ID,),
        capacity=1,
        epoch=1,
        device=torch.device("cpu"),
    )
    admission = arena.plan_admission(
        (RequestIdentity("request", 0, 0),),
        (0,),
    )
    acquired = arena.commit_admission(admission)
    arena.consume_receipt(acquired.admission)

    pending = arena.plan_transition(acquired.selection)
    with pytest.raises(RuntimeError, match="pending plan"):
        arena.read_receipt(acquired.selection)
    assert arena.versions()[0].version == 0
    arena.abort_transition(pending, poison=False)

    retry = arena.plan_transition(acquired.selection)
    receipt = arena.commit_transition(retry)
    arena.consume_receipt(receipt)
    assert arena.versions()[0].version == 1


def test_selection_tensors_are_exact_physical_provenance() -> None:
    arena = RequestSlotArena(
        resource="state",
        key_fields=(RequestKeyField.REQUEST_ID,),
        capacity=1,
        epoch=1,
        device=torch.device("cpu"),
    )
    admission = arena.plan_admission(
        (RequestIdentity("request", 0, 0),),
        (0,),
    )
    acquired = arena.commit_admission(admission)
    arena.consume_receipt(acquired.admission)

    wrong_dtype = replace(
        acquired.selection,
        device_slot_ids=acquired.selection.device_slot_ids.float(),
    )
    with pytest.raises(RuntimeError, match="tensors differ"):
        arena.read_receipt(wrong_dtype)
    wrong_unique_slot = replace(
        acquired.selection,
        device_lease_slot_ids=torch.tensor((1,), dtype=torch.long),
    )
    with pytest.raises(RuntimeError, match="tensors differ"):
        arena.read_receipt(wrong_unique_slot)
    wrong_shape = replace(
        acquired.selection,
        valid_mask=torch.tensor(True),
    )
    with pytest.raises(RuntimeError, match="tensors differ"):
        arena.read_receipt(wrong_shape)


def test_pin_blocks_retirement_and_reuse_advances_generation() -> None:
    arena = RequestSlotArena(
        resource="state",
        key_fields=(RequestKeyField.REQUEST_ID,),
        capacity=1,
        epoch=1,
        device=torch.device("cpu"),
    )
    identity = RequestIdentity("first", 0, 0)
    admission = arena.plan_admission((identity,), (0,))
    acquired = arena.commit_admission(admission)
    arena.consume_receipt(acquired.admission)
    pin_plan = arena.plan_pin(acquired.selection)
    pin = arena.commit_pin(pin_plan)
    assert pin.versions[0].lineage_id == acquired.selection.leases[0].lineage_id
    assert pin.versions[0].version == 0

    transition = arena.plan_transition(acquired.selection)
    transition_receipt = arena.commit_transition(transition)
    arena.consume_receipt(transition_receipt)
    assert arena.versions()[0].version == 1
    arena.validate_pin(pin)
    arena.release_selection(acquired.selection)

    with pytest.raises(RuntimeError, match="active leases, pins, or receipts"):
        arena.retire(RequestFinishedBatch((identity,)))
    arena.release_pin(pin)
    retired = arena.retire(RequestFinishedBatch((identity,)))
    old_slot = retired.slots[0].slot
    old_lineage = retired.slots[0].lineage_id
    arena.consume_receipt(retired)

    replacement = arena.plan_admission(
        (RequestIdentity("second", 1, 0),),
        (0,),
    )
    reacquired = arena.commit_admission(replacement)
    arena.consume_receipt(reacquired.admission)
    new_slot = reacquired.selection.leases[0].slot
    new_lineage = reacquired.selection.leases[0].lineage_id
    assert new_slot.slot_id == old_slot.slot_id
    assert new_slot.generation == old_slot.generation + 1
    assert new_lineage != old_lineage


def test_poison_is_a_fail_stop_for_effects_and_cleanup() -> None:
    arena = RequestSlotArena(
        resource="state",
        key_fields=(RequestKeyField.REQUEST_ID,),
        capacity=1,
        epoch=1,
        device=torch.device("cpu"),
    )
    admission = arena.plan_admission(
        (RequestIdentity("request", 0, 0),),
        (0,),
    )
    acquired = arena.commit_admission(admission)
    arena.consume_receipt(acquired.admission)
    pending = arena.plan_transition(acquired.selection)
    arena.abort_transition(pending, poison=True)

    with pytest.raises(RuntimeError, match="poisoned"):
        arena.read_receipt(acquired.selection)
    with pytest.raises(RuntimeError, match="poisoned"):
        arena.release_selection(acquired.selection)
