"""Identity contracts for shared staged-workload port vocabulary."""

from tllm.contracts.workload_ports import (
    RuntimeValueMarker,
    SKIPPED,
    StagedLifecycle,
    StagedPortTransactionResolution,
)
from tllm.runtime import (
    StagedPortTransactionResolution as RuntimeTransactionResolution,
)
from tllm.workload import StagedLifecycle as WorkloadStagedLifecycle


def test_marker_remains_closed_and_is_not_string_coercible() -> None:
    assert SKIPPED is RuntimeValueMarker.SKIPPED
    assert SKIPPED != "skipped"
    assert str(SKIPPED) == "RuntimeValueMarker.SKIPPED"


def test_public_facades_export_the_canonical_contract_objects() -> None:
    assert WorkloadStagedLifecycle is StagedLifecycle
    assert RuntimeTransactionResolution is StagedPortTransactionResolution
