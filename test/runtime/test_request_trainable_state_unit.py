"""Lifecycle and numerical oracle for generic request-scoped trainable state."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    RequestKeyField,
)
from tllm.runtime.request_trainable_state import (
    RequestIdentityBatch,
    RequestStateRuntimeService,
    RequestStateSelectionBatch,
    RequestTrainableStateStore,
    StateEffectOperation,
)
from tllm.runtime.workload_execution import StateVersionRegistry
from tllm.schedule.contracts import (
    RegionEffect,
    ScheduleEffectKind,
    ScheduleState,
)


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0]]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight


def _model() -> _ScalarModel:
    return _ScalarModel()


def _optimizer(module: _ScalarModel) -> torch.optim.Optimizer:
    return torch.optim.SGD(module.parameters(), lr=0.1)


def _store(
    capacity: int = 2,
    *,
    resource: str = "distiller",
) -> RequestTrainableStateStore[_ScalarModel]:
    return RequestTrainableStateStore(
        resource=resource,
        key_fields=(RequestKeyField.REQUEST_ID,),
        capacity=capacity,
        epoch=3,
        device=torch.device("cpu"),
        module_factory=_model,
        optimizer_factory=_optimizer,
    )


def _losses(
    store: RequestTrainableStateStore[_ScalarModel],
    selection: RequestStateSelectionBatch,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.nn.functional.mse_loss(
            group.module(source.index_select(0, group.row_indices)),
            target.index_select(0, group.row_indices),
        )
        for group in store.groups(selection)
    )


def test_parallel_aliases_share_one_transition_and_other_requests_are_isolated() -> None:
    store = _store()
    identities = (
        RequestIdentity("parent-a", 0, 0),
        RequestIdentity("parent-a", 0, 1),
        RequestIdentity("parent-b", 1, 0),
    )
    acquired = store.acquire(identities)
    store.consume_receipt(acquired.admission)

    assert len(acquired.selection.leases) == 2
    torch.testing.assert_close(
        acquired.selection.device_slot_ids,
        torch.tensor([0, 0, 1], dtype=torch.long),
    )
    assert torch.equal(
        acquired.selection.valid_mask,
        torch.ones((3,), dtype=torch.bool),
    )
    assert acquired.admission.operation is StateEffectOperation.ADMIT
    assert len(acquired.admission.slots) == 2

    source = torch.tensor([[1.0], [2.0], [4.0]])
    target = torch.zeros_like(source)
    before = tuple(
        group.module.weight.detach().clone()
        for group in store.groups(acquired.selection)
    )
    receipt = store.optimize(
        acquired.selection,
        _losses(store, acquired.selection, source, target),
    )

    assert receipt.operation is StateEffectOperation.TRANSITION
    assert tuple((item.before_version, item.after_version) for item in receipt.slots) == (
        (0, 1),
        (0, 1),
    )
    store.consume_receipt(receipt)
    after = tuple(
        group.module.weight.detach().clone()
        for group in store.groups(acquired.selection)
    )
    assert not torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])
    assert tuple(version.version for version in store.versions()) == (1, 1)


def test_capacity_failure_is_explicit_and_does_not_evict_active_state() -> None:
    store = _store(capacity=1)
    first = store.acquire((RequestIdentity("a", 0, 0),))
    store.consume_receipt(first.admission)

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        store.acquire((RequestIdentity("b", 1, 0),))

    assert tuple(version.key.request_id for version in store.versions()) == ("a",)
    store.release_selection(first.selection)


def test_finish_waits_for_selection_release_and_slot_generation_blocks_aba() -> None:
    store = _store(capacity=1)
    first_identity = RequestIdentity("a", 0, 0)
    first = store.acquire((first_identity,))
    store.consume_receipt(first.admission)

    with pytest.raises(RuntimeError, match="active leases, pins, or receipts"):
        store.retire(RequestFinishedBatch((first_identity,)))

    store.release_selection(first.selection)
    retired = store.retire(RequestFinishedBatch((first_identity,)))
    assert retired.operation is StateEffectOperation.RETIRE
    assert len(retired.slots) == 1
    store.consume_receipt(retired)

    second_identity = RequestIdentity("b", 1, 0)
    second = store.acquire((second_identity,))
    store.consume_receipt(second.admission)
    assert second.selection.leases[0].slot.slot_id == first.selection.leases[0].slot.slot_id
    assert (
        second.selection.leases[0].slot.generation
        == first.selection.leases[0].slot.generation + 1
    )
    with pytest.raises(RuntimeError, match="no longer active"):
        store.read_receipt(first.selection)
    store.release_selection(second.selection)


def test_shared_alias_retires_only_after_last_identity_and_reuse_resets_parameters() -> None:
    store = _store(capacity=1)
    first = RequestIdentity("parent", 0, 0)
    second = RequestIdentity("parent", 0, 1)
    acquired = store.acquire((first, second))
    store.consume_receipt(acquired.admission)
    source = torch.tensor([[1.0], [2.0]])
    target = torch.zeros_like(source)
    transition_receipt = store.optimize(
        acquired.selection,
        _losses(store, acquired.selection, source, target),
    )
    store.consume_receipt(transition_receipt)
    trained = store.groups(acquired.selection)[0].module.weight.detach().clone()
    assert not torch.equal(trained, torch.tensor([[1.0]]))
    store.release_selection(acquired.selection)

    partial = store.retire(RequestFinishedBatch((first,)))
    assert partial.slots == ()
    assert len(store.versions()) == 1
    store.consume_receipt(partial)
    final = store.retire(RequestFinishedBatch((second,)))
    assert len(final.slots) == 1
    assert store.versions() == ()
    store.consume_receipt(final)

    replacement = store.acquire((RequestIdentity("replacement", 2, 0),))
    store.consume_receipt(replacement.admission)
    reset = store.groups(replacement.selection)[0].module.weight.detach()
    torch.testing.assert_close(reset, torch.tensor([[1.0]]))
    store.release_selection(replacement.selection)


def test_epoch_replacement_requires_empty_lifecycle() -> None:
    store = _store(capacity=1)
    identity = RequestIdentity("active", 0, 0)
    acquired = store.acquire((identity,))
    with pytest.raises(RuntimeError, match="requests are active"):
        store.replace_epoch(4)
    store.consume_receipt(acquired.admission)
    store.release_selection(acquired.selection)
    retirement = store.retire(RequestFinishedBatch((identity,)))
    store.consume_receipt(retirement)
    store.replace_epoch(4)
    assert store.epoch == 4


def test_selection_rows_and_device_materialization_are_tamper_evident() -> None:
    store = _store(capacity=1)
    acquired = store.acquire((RequestIdentity("active", 0, 0),))
    forged_rows = replace(acquired.selection, rows=())
    with pytest.raises(RuntimeError, match="leases differ"):
        store.read_receipt(forged_rows)

    acquired.selection.device_slot_ids.fill_(9)
    with pytest.raises(RuntimeError, match="tensors differ"):
        store.read_receipt(acquired.selection)


def test_all_losses_are_validated_before_any_slot_changes() -> None:
    store = _store(capacity=2)
    acquired = store.acquire(
        (
            RequestIdentity("first", 0, 0),
            RequestIdentity("second", 1, 0),
        )
    )
    store.consume_receipt(acquired.admission)
    valid = torch.tensor(1.0, requires_grad=True)
    invalid = torch.ones((2,), requires_grad=True)

    with pytest.raises(ValueError, match="loss must be scalar"):
        store.optimize(acquired.selection, (valid, invalid))

    assert tuple(version.version for version in store.versions()) == (0, 0)


def test_module_factory_cannot_alias_physical_slots() -> None:
    shared = _ScalarModel()

    def shared_model() -> _ScalarModel:
        return shared

    with pytest.raises(RuntimeError, match="slots must be distinct"):
        RequestTrainableStateStore(
            resource="distiller",
            key_fields=(RequestKeyField.REQUEST_ID,),
            capacity=2,
            epoch=1,
            device=torch.device("cpu"),
            module_factory=shared_model,
            optimizer_factory=_optimizer,
        )


def test_async_pin_blocks_retirement_after_selection_release() -> None:
    store = _store(capacity=1)
    identity = RequestIdentity("active", 0, 0)
    acquired = store.acquire((identity,))
    store.consume_receipt(acquired.admission)
    pin = store.pin_synchronous(acquired.selection)
    store.release_selection(acquired.selection)

    with pytest.raises(RuntimeError, match="active leases, pins, or receipts"):
        store.retire(RequestFinishedBatch((identity,)))

    store.release_async_pin(pin)
    retirement = store.retire(RequestFinishedBatch((identity,)))
    assert len(retirement.slots) == 1
    store.consume_receipt(retirement)


def test_receipts_are_store_issued_and_consumed_once() -> None:
    store = _store(capacity=1)
    acquired = store.acquire((RequestIdentity("active", 0, 0),))
    store.consume_receipt(acquired.admission)
    receipt = store.read_receipt(acquired.selection)
    forged = replace(receipt, selection_id=receipt.selection_id + 1)

    with pytest.raises(RuntimeError, match="differs from issued"):
        store.consume_receipt(forged)
    store.consume_receipt(receipt)
    with pytest.raises(RuntimeError, match="unknown or consumed"):
        store.consume_receipt(receipt)


def test_retirement_receipt_pins_generation_until_audit_consumes_it() -> None:
    store = _store(capacity=1)
    identity = RequestIdentity("first", 0, 0)
    acquired = store.acquire((identity,))
    store.consume_receipt(acquired.admission)
    store.release_selection(acquired.selection)
    retirement = store.retire(RequestFinishedBatch((identity,)))

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        store.acquire((RequestIdentity("second", 1, 0),))

    store.consume_receipt(retirement)
    replacement = store.acquire((RequestIdentity("second", 1, 0),))
    assert (
        replacement.selection.leases[0].slot.generation
        == retirement.slots[0].slot.generation + 1
    )


def test_runtime_service_reuses_duplicate_rows_and_retains_failed_audit_receipts() -> None:
    service = RequestStateRuntimeService(_store())
    first = RequestIdentity("first", 0, 0)
    service.stage_identities(11, RequestIdentityBatch((first, first)))
    selection = service.selection_for_step(11)
    assert tuple(row.lease.slot for row in selection.rows) == (
        selection.leases[0].slot,
        selection.leases[0].slot,
    )
    assert len(selection.leases) == 1

    transition = service.before(
        StateEffectOperation.TRANSITION,
        selection,
        11,
    )
    source = torch.tensor([[1.0], [2.0]])
    target = torch.zeros_like(source)
    losses = tuple(
        torch.nn.functional.mse_loss(
            group.module(source.index_select(0, group.row_indices)),
            target.index_select(0, group.row_indices),
        )
        for group in service.groups(selection)
    )
    service.optimize(selection, losses)
    forged = replace(
        transition,
        selected_before=(
            replace(
                transition.selected_before[0],
                before_version=transition.selected_before[0].before_version + 1,
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="before/after effect"):
        service.after(forged)
    with pytest.raises(RuntimeError, match="unaudited transition receipt"):
        service.finish_step(11)
    service.after(transition)
    service.finish_step(11)

    second = RequestIdentity("second", 1, 0)
    service.stage_identities(12, RequestIdentityBatch((second,)))
    second_selection = service.selection_for_step(12)
    transition = service.before(
        StateEffectOperation.TRANSITION,
        second_selection,
        12,
    )
    group = service.groups(second_selection)[0]
    loss = torch.nn.functional.mse_loss(
        group.module(torch.tensor([[3.0]])), torch.zeros((1, 1))
    )
    service.optimize(second_selection, (loss,))
    service.after(transition)
    read = service.before(StateEffectOperation.READ, second_selection, 12)
    service.after(read)
    assert tuple(version.version for version in service.versions()) == (1, 1)
    service.finish_step(12)

    retirement = service.retire_requests(RequestFinishedBatch((first, second)))
    assert retirement.operation is StateEffectOperation.RETIRE
    assert len(retirement.slots) == 2


def test_runtime_identity_batch_rejects_unresolved_rows() -> None:
    with pytest.raises(ValueError, match="fully resolved"):
        RequestIdentityBatch((RequestIdentity("unknown", -1, 0),))


def test_qualified_execution_services_share_task_receipts_across_two_workloads() -> None:
    left_task_service = RequestStateRuntimeService(_store(resource="left"))
    right_task_service = RequestStateRuntimeService(_store(resource="right"))
    left_execution_service = left_task_service.for_signed_resource("alpha/left")
    right_execution_service = right_task_service.for_signed_resource("beta/right")
    registry = {
        left_execution_service.resource: left_execution_service,
        right_execution_service.resource: right_execution_service,
    }
    assert tuple(registry) == ("alpha/left", "beta/right")

    for task_service, execution_service, identity in (
        (
            left_task_service,
            left_execution_service,
            RequestIdentity("left", 0, 0),
        ),
        (
            right_task_service,
            right_execution_service,
            RequestIdentity("right", 1, 0),
        ),
    ):
        execution_service.stage_identities(3, RequestIdentityBatch((identity,)))
        selection = execution_service.selection_for_step(3)
        audit = execution_service.before(
            StateEffectOperation.TRANSITION,
            selection,
            3,
        )
        group = task_service.groups(selection)[0]
        loss = torch.nn.functional.mse_loss(
            group.module(torch.tensor([[1.0]])), torch.zeros((1, 1))
        )
        task_service.optimize(selection, (loss,))
        execution_service.after(audit)
        execution_service.finish_step(3)
        assert task_service.versions()[0].version == 1


def test_runtime_service_retires_a_staged_step_without_a_selection() -> None:
    service = RequestStateRuntimeService(_store())
    identity = RequestIdentity("disabled", 0, 0)
    service.stage_identities(7, RequestIdentityBatch((identity,)))
    service.finish_step(7)

    receipt = service.retire_requests(RequestFinishedBatch((identity,)))
    assert receipt.operation is StateEffectOperation.RETIRE
    assert receipt.slots == ()


def test_request_effect_cannot_fall_back_to_legacy_snapshot_audit() -> None:
    state = ScheduleState(
        resource="distiller",
        scope="request",
        request_key=(RequestKeyField.REQUEST_ID,),
    )
    effect = RegionEffect(
        kind=ScheduleEffectKind.STATE_READ,
        resource="distiller",
        selection_binding_id="selection:distiller",
    )

    with pytest.raises(RuntimeError, match="typed request-state audit service"):
        StateVersionRegistry((state,), {}, (effect,))
