"""Evidence 19: real host-state rollback and post-publication receipt settlement."""

from dataclasses import replace
from itertools import count

import pytest
import torch

import pllm
from pllm import make_llm
from pllm.execution import trace as trace_module
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.host_runtime import HostExecution, HostSettlementError
from pllm.execution.trace import TRACE_SCHEMA_VERSION, TracePhase, TraceSnapshot
from pllm.host import HostReceipt, HostRuntimeFacts
from pllm.workloads.gtta import state as gtta_math
from pllm.workloads.gtta.environment import GTTAEnvironmentService
from pllm.workloads.gtta.state import GTTADynamicsSnapshot, GTTADynamicsState, GTTAStepContext
from pllm.workloads.gtta.types import (
    GTTAConfig,
    GTTAHostRequest,
    GTTAHostTransaction,
    GTTAPromptPosition,
    GTTAQuery,
    GTTAResult,
    TrajectoryFeedback,
)
from pllm.workloads.gtta.workload import GTTAWorkload


def test_typed_host_transaction_abort_requeue_and_postpublish_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = count(0, 10)
    monkeypatch.setattr(trace_module, "monotonic_ns", lambda: next(clock))
    service = GTTAEnvironmentService(GTTAConfig())
    event = TrajectoryFeedback("task", "act", "start", "end", "change", 1.0)
    service.enqueue_feedback(event)
    request = GTTAHostRequest(GTTAQuery(0, "task", ("act",), "start", "", "", 0, GTTAPromptPosition.SUFFIX))
    original_state = GTTADynamicsState()

    def update(context: GTTAStepContext) -> None:
        assert context.dynamics is original_state
        gtta_math.update(service.config, context)

    def failed_query(context: GTTAStepContext) -> None:
        assert context.dynamics is original_state and original_state.version == 1
        raise RuntimeError("query interrupted after update")

    facts = pllm.host.HostRuntimeFacts(torch.device("cpu"))
    binding = pllm.host.transaction(
        "local.dynamics",
        facts,
        config=service.config,
        state_factory=lambda: original_state,
        service=service,
        update=update,
        query=failed_query,
        publish=gtta_math.publish,
    )
    assert binding.service is service
    local = binding.compile()
    assert local.service is service
    with pytest.raises(RuntimeError, match="query interrupted"):
        local.run(request)
    assert original_state.version == 0 and service.queued_trajectories == 1
    coordinator = make_llm(workload=GTTAWorkload(service), config=HostRuntimeFacts(torch.device("cpu")))
    execution = coordinator.execution
    original_service_commit = service.commit

    def reject_publication(
        self: GTTAEnvironmentService,
        transaction: GTTAHostTransaction,
        receipt: HostReceipt[GTTAResult],
    ) -> None:
        raise RuntimeError("publication rejected before commit")

    with monkeypatch.context() as fault:
        fault.setattr(GTTAEnvironmentService, "commit", reject_publication)
        with pytest.raises(RuntimeError, match="publication rejected"):
            coordinator.run(request)
    assert execution.diagnostics().state_version == 0
    assert service.queued_trajectories == 1

    # Save the issued object, not a reconstructed ID: receipt identity is the capability.
    published: list[HostReceipt[GTTAResult]] = []

    def record_publication(
        self: GTTAEnvironmentService,
        transaction: GTTAHostTransaction,
        receipt: HostReceipt[GTTAResult],
    ) -> None:
        original_service_commit(transaction, receipt)
        published.append(receipt)

    def reject_settlement(
        self: HostExecution[GTTAHostTransaction, GTTAStepContext, GTTADynamicsSnapshot, GTTAResult],
        receipt: HostReceipt[GTTAResult],
    ) -> None:
        raise RuntimeError("settlement interrupted after publication")

    with monkeypatch.context() as fault:
        fault.setattr(GTTAEnvironmentService, "commit", record_publication)
        fault.setattr(HostExecution, "commit", reject_settlement)
        with pytest.raises(HostSettlementError) as failure:
            coordinator.run(request)
    (receipt,) = published
    assert failure.value.receipt is receipt
    assert service.queued_trajectories == 0
    assert execution.diagnostics().state_version == 1
    assert execution.diagnostics().transaction_count == 0
    phase = [event.phase for event in execution.trace.snapshot().events if event.invocation_id == receipt.receipt_id]
    assert phase[-3:] == [TracePhase.PREPARED, TracePhase.PUBLISH, TracePhase.FAILED]
    with pytest.raises(RuntimeError, match="not active"):
        execution.commit(replace(receipt))
    execution.commit(receipt)
    with pytest.raises(RuntimeError, match="not active"):
        execution.commit(receipt)
    result = coordinator.run(request)
    assert result is not None and result.update.processed == 0
    assert result.update.state_version == 1
    assert execution.diagnostics().transaction_count == 2
    assert execution.diagnostics().fallback_count == 0
    before = execution.trace.snapshot()
    assert before.schema_version == TRACE_SCHEMA_VERSION and before.dropped == 0
    assert before.pending_device_timings == 0 and not before.device_task_times()
    assert any(event.phase is TracePhase.ABORT for event in before.events)
    assert sum(event.phase is TracePhase.SETTLED for event in before.events) == 2
    assert all(elapsed == calls * 10 for calls, elapsed in before.host_task_times().values())
    assert decode_boundary(encode_boundary(before), TraceSnapshot) == before
    for _ in range(60):
        coordinator.run(request)
    after = execution.trace.snapshot()
    assert after.execution_id == before.execution_id and after.process_id == before.process_id
    assert len(after.events) == after.capacity == 256 and after.dropped > 0
    assert after.dropped == after.events[0].sequence - 1
    assert after.events[-1].sequence == after.dropped + after.capacity
    assert all(elapsed == calls * 10 for calls, elapsed in after.host_task_times().values())
