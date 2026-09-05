"""Evidence 19: real host-state rollback and post-publication receipt settlement."""

from dataclasses import replace

import pytest
import torch

from tllm.canonical.host_runtime import (
    CanonicalHostRuntimeFacts, HostExecution, HostReceipt, HostSettlementError, HostTransactionCoordinator,
)
from tllm.canonical.workloads.gtta.environment import GTTAEnvironmentService
from tllm.canonical.workloads.gtta.types import (
    GTTAConfig, GTTAHostRequest, GTTAHostTransaction, GTTAPromptPosition, GTTAQuery, GTTAResult, TrajectoryFeedback,
)
from tllm.canonical.workloads.gtta.state import GTTADynamicsSnapshot, GTTAStepContext
from tllm.canonical.workloads.gtta.workload import GTTAWorkload


def test_typed_host_transaction_abort_requeue_and_postpublish_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    execution = GTTAWorkload().compile(CanonicalHostRuntimeFacts(torch.device("cpu")).timeline())
    service = GTTAEnvironmentService(GTTAConfig())
    event = TrajectoryFeedback("task", "act", "start", "end", "change", 1.0)
    service.enqueue_feedback(event)
    request = GTTAHostRequest(GTTAQuery(0, "task", ("act",), "start", "", "", 0, GTTAPromptPosition.SUFFIX))
    coordinator = HostTransactionCoordinator(service, execution)
    original_service_commit = service.commit

    def reject_publication(
        self: GTTAEnvironmentService, transaction: GTTAHostTransaction, receipt: HostReceipt[GTTAResult],
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
        self: GTTAEnvironmentService, transaction: GTTAHostTransaction, receipt: HostReceipt[GTTAResult],
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
    receipt, = published
    assert failure.value.receipt is receipt
    assert service.queued_trajectories == 0
    assert execution.diagnostics().state_version == 1
    assert execution.diagnostics().transaction_count == 0
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
