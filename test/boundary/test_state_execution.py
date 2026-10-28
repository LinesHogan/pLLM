"""PTTS lifecycle across request, route and instance resources, without region execution."""

import pytest
import torch

from pllm.contracts.request_identity import RequestIdentity
from pllm.execution.state_execution import StateSettlementError
from test.state_fixtures import (
    AUX,
    DENSE,
    GLOBAL,
    assert_multistate_reclaimed,
    increment_resources,
    multistate,
    rows_for,
)


def test_heterogeneous_resources_admit_publish_and_pin_in_one_invocation():
    execution, task = multistate()
    (global_ref,) = execution.admit(())
    a, b = RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0)
    refs = execution.admit((a, b))
    rows = rows_for(a, b)
    selected = execution.directory.select(rows)
    assert selected.ref_for(b, AUX) == refs[3] and selected.ref_for(a, GLOBAL) == global_ref
    task.update = increment_resources
    execution.step(rows)
    context = task.context
    for old, expected in zip(refs, (2, 2, 11, 11, 101), strict=True):
        current = context.current_owned_ref(old)
        assert current.version_id == old.version_id + 1
        value = execution.snapshot(current)[0]
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    # Request inference still owns the original global version after publication.
    pin = execution.directory.inference_pin(a)
    assert tuple(pin.refs) == (global_ref,)
    torch.testing.assert_close(execution.arena.snapshot(pin, global_ref)[0], torch.full((2, 5), 100.0), rtol=0, atol=0)
    execution.retire((a, b))
    execution.retire(())
    assert_multistate_reclaimed(execution)


def test_admission_failure_rolls_back_every_resource_and_inference_pin():
    execution, _ = multistate(failing=(AUX,))
    execution.admit(())
    a, b = RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0)
    with pytest.raises(RuntimeError):
        execution.admit((a, b))
    execution.arena.collect_completed(wait=True)
    assert execution.directory.active_count == execution.arena.active_lineages == 1
    assert execution.directory.inference_pin_count == 0
    assert execution.arena.usage().private_versions == execution.arena.usage().pinned_versions == 0
    execution.retire(())
    assert_multistate_reclaimed(execution)


def test_shared_route_state_persists_until_dormant_retirement():
    execution, _ = multistate(route=True, instance=False)
    a, b = RequestIdentity("a", 0, 0, "math"), RequestIdentity("b", 1, 0, "math")
    execution.admit((a,))
    joined = execution.admit((b,))
    left, right = execution.directory.select(rows_for(a)), execution.directory.select(rows_for(b))
    assert left.ref_for(a, AUX) == right.ref_for(b, AUX) == joined[-1]
    invocation = execution.begin(rows_for(a))
    execution.cancel_step(invocation)
    invocation = execution.begin(rows_for(b))
    execution.retire((a,))
    assert invocation.page_pins[invocation.pin] is None
    execution.cancel_step(invocation)
    execution.retire((b,))
    assert execution.directory.active_count == 1  # The shared route persists dormant.
    c = RequestIdentity("c", 2, 0, "math")
    assert execution.admit((c,))[-1] == joined[-1]
    execution.retire((c,))
    execution.retire_dormant()
    assert_multistate_reclaimed(execution)


def test_retiring_a_request_cancels_its_invocation_and_releases_only_its_pins():
    execution, _ = multistate(route=True)
    (global_ref,) = execution.admit(())
    a, b = RequestIdentity("a", 0, 0, "math"), RequestIdentity("b", 1, 0, "math")
    execution.admit((a, b))
    invocation = execution.begin(rows_for(a))
    assert invocation.page_pins[invocation.pin] is None
    execution.retire((a,))
    assert invocation.cancel_requested.is_set() and invocation.page_pins[invocation.pin] is not None
    assert not execution.directory.has_inference_pin(a)
    assert tuple(execution.directory.inference_pin(b).refs) == (global_ref,)
    execution.cancel_step(execution.begin(rows_for(b)))
    execution.retire((b,))
    # An offline instance invocation has no request roots, even in a mixed directory.
    invocation = execution.begin(())
    assert tuple(invocation.pin.refs) == (global_ref,)
    assert tuple(invocation.selection.for_resource(DENSE).refs) == ()
    execution.retire(())
    assert invocation.page_pins[invocation.pin] is not None
    execution.retire_dormant()
    assert_multistate_reclaimed(execution)


def test_cancellation_inside_a_loop_keeps_the_first_commit_and_stops():
    execution, task = multistate(instance=False)
    identities = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    refs = execution.admit(identities)
    rows = rows_for(*identities)
    calls = []

    def update(context):
        for _ in range(6):
            calls.append(1)
            if len(calls) == 2:
                context.cancel_requested.set()
            increment_resources(context)

    task.update = update
    invocation = execution.start_step(rows)
    with pytest.raises(StateSettlementError):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert len(calls) == 2 and execution.arena.latest(refs[0].lineage_id).version_id == 1
    execution.cancel_step(invocation)
    execution.retire(identities)
    assert_multistate_reclaimed(execution)


@pytest.mark.parametrize("committed", (False, True))
def test_task_failure_keeps_the_commit_boundary(committed):
    execution, task = multistate(instance=False)
    identity = RequestIdentity("a", 0, 0)
    refs = execution.admit((identity,))
    calls = []

    def fail(context):
        calls.append(1)
        if not committed:
            raise ValueError("injected failure before publication")
        increment_resources(context)
        raise RuntimeError("injected failure after publication")

    task.update = fail
    invocation = execution.start_step(rows_for(identity))
    with pytest.raises(StateSettlementError if committed else ValueError):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert len(calls) == 1
    assert execution.arena.latest(refs[0].lineage_id).version_id == int(committed)
    if committed:
        execution.cancel_step(invocation)
    execution.retire((identity,))
    assert_multistate_reclaimed(execution)
