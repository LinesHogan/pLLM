"""Numerical and lifecycle oracle for native request-scoped ESamp.

The retained legacy engine groups parallel samples by prompt index.  Native
request state instead keys a group by the stable parent request id.  This file
locks the migration contract at the RequestStateRuntimeService boundary until
the ESamp workload declares its distiller as request-trainable state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from tllm.consumers.esamp.engine import group_row_indices_by_prompt
from tllm.consumers.esamp.model import (
    BatchedLowRankGatedResidualModel,
    LowRankGatedResidualModel,
)
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


_DEVICE = torch.device("cpu")
_DTYPE = torch.float32
_HIDDEN_SIZE = 4
_RANK = 2
_LEARNING_RATE = 1e-2


def test_batched_low_rank_bank_matches_repeated_independent_slots() -> None:
    torch.manual_seed(904)
    bank = BatchedLowRankGatedResidualModel(
        capacity=2,
        hidden_size=_HIDDEN_SIZE,
        rank=_RANK,
        use_output_layernorm=True,
        device=_DEVICE,
        dtype=_DTYPE,
    )
    references = tuple(_model() for _ in range(2))
    with torch.no_grad():
        for slot, reference in enumerate(references):
            reference.a.copy_(bank.a[slot])
            reference.g.copy_(bank.g[slot])
            reference.b.copy_(bank.b[slot])
            reference.gate_bias.copy_(bank.gate_bias[slot])
            match reference.out_ln:
                case torch.nn.LayerNorm() as output_norm:
                    output_norm.weight.copy_(bank.out_ln_weight[slot])
                    output_norm.bias.copy_(bank.out_ln_bias[slot])
    source = torch.tensor(
        (
            (0.1, -0.2, 0.3, -0.4),
            (0.5, 0.6, -0.7, 0.8),
            (-0.9, 1.0, 1.1, -1.2),
        ),
        dtype=_DTYPE,
    )
    slot_ids = torch.tensor((0, 1, 0), dtype=torch.long)
    expected = torch.cat(
        tuple(
            references[slot](source[row : row + 1])
            for row, slot in enumerate((0, 1, 0))
        ),
        dim=0,
    )
    torch.testing.assert_close(
        bank(slot_ids, source), expected, rtol=0.0, atol=0.0
    )


def _model() -> LowRankGatedResidualModel:
    return LowRankGatedResidualModel(
        hidden_size=_HIDDEN_SIZE,
        rank=_RANK,
        use_output_layernorm=True,
        device=_DEVICE,
        dtype=_DTYPE,
    ).train()


def _optimizer(
    module: LowRankGatedResidualModel,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(module.parameters(), lr=_LEARNING_RATE)


@dataclass
class _ServiceFixture:
    service: RequestStateRuntimeService[LowRankGatedResidualModel]
    optimizers: list[torch.optim.Optimizer]


def _service(capacity: int) -> _ServiceFixture:
    optimizers: list[torch.optim.Optimizer] = []

    def optimizer_factory(
        module: LowRankGatedResidualModel,
    ) -> torch.optim.Optimizer:
        optimizer = _optimizer(module)
        optimizers.append(optimizer)
        return optimizer

    service = RequestStateRuntimeService(
        RequestTrainableStateStore(
            resource="distiller",
            key_fields=(RequestKeyField.REQUEST_ID,),
            capacity=capacity,
            epoch=1,
            device=_DEVICE,
            module_factory=_model,
            optimizer_factory=optimizer_factory,
        )
    )
    return _ServiceFixture(service, optimizers)


def _group_tensors(
    selection: RequestStateSelectionBatch,
    source: torch.Tensor,
    target: torch.Tensor,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for lease in selection.leases:
        row_indices = torch.tensor(
            [
                row_index
                for row_index, row in enumerate(selection.rows)
                if row.lease == lease
            ],
            dtype=torch.long,
        )
        yield (
            source.index_select(0, row_indices),
            target.index_select(0, row_indices),
        )


def _parameter_copies(
    module: LowRankGatedResidualModel,
) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def _optimizer_for(
    module: LowRankGatedResidualModel,
    optimizers: Iterable[torch.optim.Optimizer],
) -> torch.optim.Optimizer:
    parameters = tuple(module.parameters())
    # Admission reconstructs the optimizer when a physical slot receives a
    # new generation, so the newest matching optimizer is the live one.
    for optimizer in reversed(tuple(optimizers)):
        candidate_parameters = optimizer.param_groups[0]["params"]
        if len(candidate_parameters) == len(parameters) and all(
            candidate is parameter
            for candidate, parameter in zip(
                candidate_parameters, parameters, strict=True
            )
        ):
            return optimizer
    raise AssertionError("ESamp request group has no owned optimizer")


def _optimizer_copies(
    module: LowRankGatedResidualModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            optimizer.state[parameter]["step"].detach().clone(),
            optimizer.state[parameter]["exp_avg"].detach().clone(),
            optimizer.state[parameter]["exp_avg_sq"].detach().clone(),
        )
        for parameter in module.parameters()
    )


def _assert_tensor_sequences_close(
    actual: Iterable[torch.Tensor],
    expected: Iterable[torch.Tensor],
) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-6, atol=1e-7)


def _assert_optimizer_states_close(
    actual: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    expected: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    for actual_state, expected_state in zip(actual, expected, strict=True):
        _assert_tensor_sequences_close(actual_state, expected_state)


def test_native_request_oracle_groups_samples_and_predicts_before_training() -> None:
    """Freeze stable-parent grouping, exact AdamW math, and isolation.

    Both samples of each parent are one group and one optimizer transition;
    the two parent groups stay isolated.  The prediction retained for the
    current token is computed under version zero, before its own update.
    """

    torch.manual_seed(2718)
    fixture = _service(capacity=2)
    service = fixture.service
    identities = (
        RequestIdentity("parent-a", 41, 0),
        RequestIdentity("parent-a", 41, 1),
        RequestIdentity("parent-b", 92, 0),
        RequestIdentity("parent-b", 92, 1),
    )
    source = torch.tensor(
        (
            (0.1, -0.2, 0.3, -0.4),
            (0.5, 0.6, -0.7, 0.8),
            (-0.9, 1.0, 1.1, -1.2),
            (1.3, -1.4, 1.5, -1.6),
        ),
        dtype=_DTYPE,
    )
    target = torch.tensor(
        (
            (-0.3, 0.2, 0.1, 0.4),
            (0.8, -0.6, 0.5, -0.2),
            (0.7, 0.9, -1.0, 1.1),
            (-1.2, 1.3, -1.4, 1.5),
        ),
        dtype=_DTYPE,
    )

    # Frozen legacy behavior is prompt keyed; the native service's unique
    # lease order is the stable parent request-id order for this same generate.
    assert group_row_indices_by_prompt([41, 41, 92, 92], 4) == {
        41: [0, 1],
        92: [2, 3],
    }
    service.stage_identities(101, RequestIdentityBatch(identities))
    selection = service.selection_for_step(101)
    assert tuple(lease.key.request_id for lease in selection.leases) == (
        "parent-a",
        "parent-b",
    )
    assert tuple(row.lease.slot for row in selection.rows) == (
        selection.leases[0].slot,
        selection.leases[0].slot,
        selection.leases[1].slot,
        selection.leases[1].slot,
    )

    read_audit = service.before(StateEffectOperation.READ, selection, 101)
    groups = service.groups(selection)
    references: list[tuple[LowRankGatedResidualModel, torch.optim.Optimizer]] = []
    predictions: list[torch.Tensor] = []
    expected_losses: list[torch.Tensor] = []
    for group, (group_source, group_target) in zip(
        groups,
        _group_tensors(selection, source, target),
        strict=True,
    ):
        reference = _model()
        reference.load_state_dict(group.module.state_dict())
        reference_optimizer = _optimizer(reference)
        prediction = group.module(group_source)
        expected_prediction = reference(group_source)
        loss = F.mse_loss(group.module(group_source).float(), group_target.float())
        expected_loss = F.mse_loss(
            expected_prediction.float(), group_target.float()
        )
        torch.testing.assert_close(prediction, expected_prediction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(loss, expected_loss, rtol=0.0, atol=0.0)
        predictions.append(prediction.detach().clone())
        expected_losses.append(expected_loss)
        references.append((reference, reference_optimizer))
    service.after(read_audit)

    transition_audit = service.before(
        StateEffectOperation.TRANSITION, selection, 101
    )
    losses = tuple(
        F.mse_loss(group.module(group_source).float(), group_target.float())
        for group, (group_source, group_target) in zip(
            service.groups(selection),
            _group_tensors(selection, source, target),
            strict=True,
        )
    )
    for (reference, reference_optimizer), expected_loss in zip(
        references, expected_losses, strict=True
    ):
        reference_optimizer.zero_grad(set_to_none=True)
        expected_loss.backward()
        reference_optimizer.step()

    transition = service.optimize(selection, losses)
    service.after(transition_audit)

    # A native loss/update count is one scalar loss and one version transition
    # per unique parent group, never per parallel child row.
    assert len(losses) == 2
    assert len(transition.slots) == 2
    assert tuple(
        (effect.before_version, effect.after_version) for effect in transition.slots
    ) == ((0, 1), (0, 1))
    assert tuple(version.version for version in service.versions()) == (1, 1)

    for group, (reference, reference_optimizer), prediction, (group_source, _) in zip(
        service.groups(selection),
        references,
        predictions,
        _group_tensors(selection, source, target),
        strict=True,
    ):
        _assert_tensor_sequences_close(
            _parameter_copies(group.module), _parameter_copies(reference)
        )
        _assert_optimizer_states_close(
            _optimizer_copies(
                group.module, _optimizer_for(group.module, fixture.optimizers)
            ),
            _optimizer_copies(reference, reference_optimizer),
        )
        assert not torch.equal(group.module(group_source), prediction)

    parent_b_before_second_parent_a_update = _parameter_copies(groups[1].module)
    service.finish_step(101)

    service.stage_identities(
        102,
        RequestIdentityBatch(
            (
                RequestIdentity("parent-a", 41, 0),
                RequestIdentity("parent-a", 41, 1),
            )
        ),
    )
    second_selection = service.selection_for_step(102)
    second_audit = service.before(
        StateEffectOperation.TRANSITION, second_selection, 102
    )
    parent_a_group = service.groups(second_selection)[0]
    second_loss = F.mse_loss(
        parent_a_group.module(source[:2]).float(), target[:2].float()
    )
    service.optimize(second_selection, (second_loss,))
    service.after(second_audit)
    assert tuple(
        (version.key.request_id, version.version) for version in service.versions()
    ) == (("parent-a", 2), ("parent-b", 1))
    _assert_tensor_sequences_close(
        _parameter_copies(groups[1].module), parent_b_before_second_parent_a_update
    )
    service.finish_step(102)


def test_native_request_oracle_retires_last_alias_and_reinitializes_reused_slot(
) -> None:
    """Freeze alias completion, fresh initialization, and ABA-safe reuse."""

    torch.manual_seed(31415)
    expected_first_parameters = _parameter_copies(_model())
    expected_second_parameters = _parameter_copies(_model())
    torch.manual_seed(31415)
    fixture = _service(capacity=1)
    service = fixture.service
    first_alias = RequestIdentity("parent", 7, 0)
    final_alias = RequestIdentity("parent", 7, 1)
    service.stage_identities(201, RequestIdentityBatch((first_alias, final_alias)))
    selection = service.selection_for_step(201)
    original_slot = selection.leases[0].slot
    initial_parameters = _parameter_copies(service.groups(selection)[0].module)
    _assert_tensor_sequences_close(
        initial_parameters, expected_first_parameters
    )

    audit = service.before(StateEffectOperation.TRANSITION, selection, 201)
    group = service.groups(selection)[0]
    train_loss = F.mse_loss(
        group.module(torch.ones((2, _HIDDEN_SIZE))).float(),
        torch.zeros((2, _HIDDEN_SIZE)),
    )
    service.optimize(selection, (train_loss,))
    service.after(audit)
    assert not torch.equal(
        _parameter_copies(group.module)[0], initial_parameters[0]
    )
    assert _optimizer_copies(
        group.module, _optimizer_for(group.module, fixture.optimizers)
    )
    service.finish_step(201)

    first_retirement = service.retire_requests(RequestFinishedBatch((first_alias,)))
    assert first_retirement.slots == ()
    assert service.versions()[0].slot == original_slot
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        service.stage_identities(
            202, RequestIdentityBatch((RequestIdentity("blocked", 8, 0),))
        )
        service.selection_for_step(202)
    service.finish_step(202)

    final_retirement = service.retire_requests(RequestFinishedBatch((final_alias,)))
    assert len(final_retirement.slots) == 1
    assert final_retirement.slots[0].slot == original_slot
    assert service.versions() == ()

    replacement = RequestIdentity("replacement", 8, 0)
    service.stage_identities(203, RequestIdentityBatch((replacement,)))
    replacement_selection = service.selection_for_step(203)
    replacement_group = service.groups(replacement_selection)[0]
    assert replacement_selection.leases[0].slot.slot_id == original_slot.slot_id
    assert (
        replacement_selection.leases[0].slot.generation
        == original_slot.generation + 1
    )
    _assert_tensor_sequences_close(
        _parameter_copies(replacement_group.module),
        expected_second_parameters,
    )
    assert not torch.equal(
        expected_first_parameters[0], expected_second_parameters[0]
    )
    assert not _optimizer_for(
        replacement_group.module, fixture.optimizers
    ).state
    service.finish_step(203)
