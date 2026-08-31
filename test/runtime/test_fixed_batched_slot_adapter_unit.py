"""Exact CPU oracle for fixed batched request-slot storage."""

from __future__ import annotations

import pytest
import torch
from torch.optim.adamw import adamw

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    RequestKeyField,
)
from tllm.contracts.request_state import (
    RequestIdentityBatch,
    StateEffectOperation,
)
from tllm.runtime.fixed_batched_slot_adapter import (
    FixedBatchedSlotAdapter,
    FixedBatchedSlotLayout,
    FixedBatchedSlotModule,
)
from tllm.runtime.paged_state import (
    StateResourceId,
    StateVersionLineage,
    StateVersionRef,
    StateVersionRootPin,
    TensorFieldId,
)
from tllm.runtime.request_trainable_state import (
    RequestStateRuntimeService,
    RequestTrainableStateStore,
)
from tllm.runtime.request_slot_arena import RequestSlotArena
from tllm.runtime.request_slot_adapter import NoCompletionFence
from tllm.runtime.stable_packed_state import (
    FrozenPackedTensorField,
    FrozenPackedTensorLayout,
    PackedLayoutId,
)


class _ScalarBank(FixedBatchedSlotModule):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((capacity, 1))
        )
        self.bias: torch.Tensor
        self.register_buffer("bias", torch.empty((capacity, 1)))

    def fixed_batched_slot_layout(self) -> FixedBatchedSlotLayout:
        return FixedBatchedSlotLayout(
            capacity=int(self.weight.shape[0]),
            slot_parameters=(self.weight,),
            slot_buffers=(self.bias,),
        )

    def install_fixed_batched_slot_layout(
        self,
        layout: FixedBatchedSlotLayout,
    ) -> None:
        match layout.slot_parameters, layout.slot_buffers:
            case (weight,), (bias,):
                self.weight = weight
                self.bias = bias
            case _:
                raise ValueError("scalar bank received the wrong page layout")

    def forward(
        self,
        slot_ids: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        return source * self.weight.index_select(0, slot_ids) + self.bias.index_select(
            0,
            slot_ids,
        )


def test_fixed_batched_adamw_matches_independent_slots_and_aba_reset() -> None:
    bank = _ScalarBank(capacity=3)
    with torch.no_grad():
        bank.weight.copy_(torch.tensor([[1.0], [-0.5], [3.0]]))
        bank.bias.copy_(torch.tensor([[0.25], [-0.75], [2.0]]))
    initial_weight = bank.weight.detach().clone()
    initial_bias = bank.bias.detach().clone()
    adapter = FixedBatchedSlotAdapter(
        bank,
        resource=StateResourceId("state"),
        epoch=1,
        lr=0.03,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=0.02,
    )
    arena = RequestSlotArena(
        resource="state",
        key_fields=(RequestKeyField.REQUEST_ID,),
        capacity=3,
        epoch=1,
        device=torch.device("cpu"),
    )
    first = RequestIdentity("first", 0, 0)
    first_alias = RequestIdentity("first", 0, 1)
    second = RequestIdentity("second", 1, 0)
    admission = arena.plan_admission((first, first_alias, second), (0, 1, 2))
    preparation = adapter.prepare_admission(admission)
    adapter.commit_admission(admission, preparation)
    acquired = arena.commit_admission(admission)
    arena.consume_receipt(acquired.admission)

    baseline_weights = tuple(
        torch.nn.Parameter(initial_weight[slot_id].clone())
        for slot_id in range(2)
    )
    baseline_exp_avgs = tuple(torch.zeros_like(weight) for weight in baseline_weights)
    baseline_exp_avg_sqs = tuple(
        torch.zeros_like(weight) for weight in baseline_weights
    )
    baseline_steps = tuple(torch.zeros(()) for _weight in baseline_weights)
    old_root_pins: list[StateVersionRootPin] = []
    old_root_pages: tuple[int, ...] = ()
    old_root_weights: tuple[torch.Tensor, ...] = ()
    old_root_steps: tuple[torch.Tensor, ...] = ()
    old_root_exp_avgs: tuple[torch.Tensor, ...] = ()
    old_root_exp_avg_sqs: tuple[torch.Tensor, ...] = ()
    for update, (source, target) in enumerate((
        (
            torch.tensor([[1.0], [2.0], [-3.0]]),
            torch.tensor([[0.0], [1.0], [2.0]]),
        ),
        (
            torch.tensor([[0.5], [-1.0], [4.0]]),
            torch.tensor([[1.0], [0.0], [-2.0]]),
        ),
    ), start=1):
        selected = adapter.selected(
            acquired.selection,
            arena.selection_versions(acquired.selection),
        )
        prediction = adapter.bank(selected.row_slot_ids, source)
        losses = (
            torch.nn.functional.mse_loss(prediction[:2], target[:2]),
            torch.nn.functional.mse_loss(prediction[2:], target[2:]),
        )
        if update == 1:
            current_versions = arena.selection_versions(acquired.selection)
            old_refs = tuple(
                StateVersionRef(
                    StateVersionLineage(
                        StateResourceId("state"),
                        1,
                        version.slot,
                    ),
                    version.version,
                )
                for version in current_versions
            )
            old_pin = adapter.version_roots.pin_exact_synchronous(old_refs)
            old_root_pins.append(old_pin)
            old_root_pages = tuple(root.page_ids[0] for root in old_pin.roots)
            old_root_weights = tuple(
                bank.weight[page_id].detach().clone()
                for page_id in old_root_pages
            )
            old_root_steps = tuple(
                adapter.optimizer.states[0].step[page_id].clone()
                for page_id in old_root_pages
            )
            old_root_exp_avgs = tuple(
                adapter.optimizer.states[0].exp_avg[page_id].clone()
                for page_id in old_root_pages
            )
            old_root_exp_avg_sqs = tuple(
                adapter.optimizer.states[0].exp_avg_sq[page_id].clone()
                for page_id in old_root_pages
            )
        transition = arena.plan_transition(acquired.selection)
        acquired.selection.device_lease_slot_ids.fill_(2)
        adapter.execute_transition(transition, losses)
        if update == 1:
            for page_id, expected in zip(
                old_root_pages,
                old_root_weights,
                strict=True,
            ):
                torch.testing.assert_close(bank.weight[page_id], expected)
            for page_id, step, exp_avg, exp_avg_sq in zip(
                old_root_pages,
                old_root_steps,
                old_root_exp_avgs,
                old_root_exp_avg_sqs,
                strict=True,
            ):
                torch.testing.assert_close(
                    adapter.optimizer.states[0].step[page_id],
                    step,
                )
                torch.testing.assert_close(
                    adapter.optimizer.states[0].exp_avg[page_id],
                    exp_avg,
                )
                torch.testing.assert_close(
                    adapter.optimizer.states[0].exp_avg_sq[page_id],
                    exp_avg_sq,
                )
            adapter.version_roots.release_pin(
                old_root_pins.pop(),
                NoCompletionFence(),
            )
        acquired.selection.device_lease_slot_ids.copy_(
            torch.tensor((0, 1), dtype=torch.long)
        )
        receipt = arena.commit_transition(transition)
        assert tuple(
            (effect.before_version, effect.after_version)
            for effect in receipt.slots
        ) == ((update - 1, update), (update - 1, update))
        arena.consume_receipt(receipt)

        baseline_losses = (
            torch.nn.functional.mse_loss(
                source[:2] * baseline_weights[0] + initial_bias[0],
                target[:2],
            ),
            torch.nn.functional.mse_loss(
                source[2:] * baseline_weights[1] + initial_bias[1],
                target[2:],
            ),
        )
        for actual, expected in zip(losses, baseline_losses, strict=True):
            torch.testing.assert_close(actual, expected)
        torch.autograd.backward(baseline_losses)
        baseline_gradients = tuple(
            weight.grad
            for weight in baseline_weights
        )
        match baseline_gradients:
            case (torch.Tensor() as first_gradient, torch.Tensor() as second_gradient):
                with torch.no_grad():
                    adamw(
                        params=list(baseline_weights),
                        grads=[first_gradient, second_gradient],
                        exp_avgs=list(baseline_exp_avgs),
                        exp_avg_sqs=list(baseline_exp_avg_sqs),
                        max_exp_avg_sqs=[],
                        state_steps=list(baseline_steps),
                        foreach=False,
                        capturable=False,
                        differentiable=False,
                        fused=False,
                        amsgrad=False,
                        beta1=0.8,
                        beta2=0.9,
                        lr=0.03,
                        weight_decay=0.02,
                        eps=1e-6,
                        maximize=False,
                    )
            case _:
                raise AssertionError("independent AdamW must receive gradients")
        for weight in baseline_weights:
            weight.grad = torch.zeros_like(weight)

    latest = adapter.selected(
        acquired.selection,
        arena.selection_versions(acquired.selection),
    )
    latest_page_ids = tuple(
        int(latest.unique_slot_ids[index].item())
        for index in range(len(acquired.selection.leases))
    )
    for page_id, (weight, exp_avg, exp_avg_sq, step) in zip(
        latest_page_ids,
        zip(
            baseline_weights,
            baseline_exp_avgs,
            baseline_exp_avg_sqs,
            baseline_steps,
            strict=True,
        ),
        strict=True,
    ):
        torch.testing.assert_close(bank.weight[page_id], weight)
        torch.testing.assert_close(
            adapter.optimizer.states[0].exp_avg[page_id],
            exp_avg,
        )
        torch.testing.assert_close(
            adapter.optimizer.states[0].exp_avg_sq[page_id],
            exp_avg_sq,
        )
        torch.testing.assert_close(
            adapter.optimizer.states[0].step[page_id],
            step,
        )
    assert tuple(version.version for version in arena.versions()) == (2, 2)
    adapter.complete_reads_synchronous(acquired.selection)

    with torch.no_grad():
        bank.bias[latest_page_ids[0]].fill_(7.0)
    arena.release_selection(acquired.selection)
    retired = arena.retire(RequestFinishedBatch((first, first_alias)))
    adapter.retire(retired)
    arena.consume_receipt(retired)
    replacement = arena.plan_admission((RequestIdentity("replacement", 2, 0),), (0,))
    adapter.commit_admission(replacement, adapter.prepare_admission(replacement))
    reacquired = arena.commit_admission(replacement)
    arena.consume_receipt(reacquired.admission)

    assert reacquired.selection.leases[0].slot.slot_id == 0
    assert reacquired.selection.leases[0].slot.generation == 2
    replacement_view = adapter.selected(
        reacquired.selection,
        arena.selection_versions(reacquired.selection),
    )
    replacement_page = int(replacement_view.unique_slot_ids[0].item())
    torch.testing.assert_close(bank.weight[replacement_page], initial_weight[0])
    torch.testing.assert_close(bank.bias[replacement_page], initial_bias[0])
    torch.testing.assert_close(
        adapter.optimizer.states[0].exp_avg[replacement_page],
        torch.zeros(1),
    )
    torch.testing.assert_close(
        adapter.optimizer.states[0].exp_avg_sq[replacement_page],
        torch.zeros(1),
    )
    torch.testing.assert_close(
        adapter.optimizer.states[0].step[replacement_page],
        torch.tensor(0.0),
    )
    adapter.complete_reads_synchronous(reacquired.selection)

    pending = arena.plan_admission(
        (RequestIdentity("third", 3, 0),),
        (0,),
    )
    adapter.poison()
    with pytest.raises(RuntimeError, match="poisoned"):
        adapter.prepare_admission(pending)
    arena.abort_admission(pending, poison=False)


def test_stable_view_bridge_blocks_step_finish_until_completion() -> None:
    bank = _ScalarBank(capacity=1)
    with torch.no_grad():
        bank.weight.fill_(2.0)
        bank.bias.fill_(0.5)
    store = RequestTrainableStateStore[_ScalarBank].fixed_batched(
        resource="state",
        key_fields=(RequestKeyField.REQUEST_ID,),
        epoch=1,
        bank=bank,
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    service = RequestStateRuntimeService(store)
    packed_layout = FrozenPackedTensorLayout(
        PackedLayoutId(2),
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
    views = service.stable_view_service(
        view_epoch=1,
        layout=packed_layout,
        buffer_count=1,
    )
    identity = RequestIdentity("bridge", 0, 0)
    service.stage_identities(0, RequestIdentityBatch((identity,)))
    selection = service.selection_for_step(0)
    audit = service.before(StateEffectOperation.READ, selection, 0)
    view = views.create_synchronous(
        0,
        selection,
        (torch.tensor(((3.0,),)),),
    )
    with pytest.raises(RuntimeError, match="uncompleted stable packed view"):
        service.finish_step(0)
    torch.testing.assert_close(
        view.packed_tensors.tensors[0],
        torch.tensor(((3.0,),)),
    )
    views.complete_synchronous(view)
    service.after(audit)
    service.finish_step(0)
    service.retire_requests(RequestFinishedBatch((identity,)))
    store.replace_epoch(2)
    assert service.epoch == 2
