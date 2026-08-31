"""CPU contracts for runner-owned eager packed hidden capture."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from tllm.backends.protocol import RunnerBinding
from tllm.backends.vllm.packed_hidden_capture import (
    RunnerPackedHiddenCapture,
    synchronize_runner_packed_hidden_capture,
)
from tllm.backends.vllm.v027_mrv1 import V027MRv1Driver
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshotBatch,
)
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
)
from tllm.runtime.packed_phase_row_staging import PackedPhaseRowStaging
from tllm.runtime.specialization import RuntimeSpecializationSpec


_DEVICE = torch.device("cpu")
_DTYPE = torch.float32
_PLAN_EPOCH = 5


class _SharedModule(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden + 1.0, hidden.sum(dim=1)


class _TailModule(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 3.0


class _CaptureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = _SharedModule()
        self.tail = _TailModule()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shared_hidden, _ = self.shared(hidden)
        return self.tail(shared_hidden)


class _InvalidTail(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> str:
        _ = hidden
        return "invalid"


class _InvalidCaptureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = _SharedModule()
        self.tail = _InvalidTail()

    def forward(self, hidden: torch.Tensor) -> str:
        shared_hidden, _ = self.shared(hidden)
        return self.tail(shared_hidden)


@dataclass
class _SnapshotExecution:
    batches: list[PackedHiddenSnapshotBatch]

    def stage_packed_hidden_snapshots(
        self, batch: PackedHiddenSnapshotBatch
    ) -> None:
        self.batches.append(batch)


class _FailingSnapshotExecution:
    def stage_packed_hidden_snapshots(
        self, batch: PackedHiddenSnapshotBatch
    ) -> None:
        _ = batch
        raise RuntimeError("execution rejected packed snapshots")


def _resources() -> tuple[PackedHiddenResourceBinding, ...]:
    return (
        PackedHiddenResourceBinding(
            layer="0",
            layer_kind=HiddenLayerKind.INDEX,
            site=HiddenSite.BLOCK_OUTPUT,
            module_path="shared",
        ),
        PackedHiddenResourceBinding(
            layer="1",
            layer_kind=HiddenLayerKind.INDEX,
            site=HiddenSite.BLOCK_OUTPUT,
            module_path="shared",
        ),
        PackedHiddenResourceBinding(
            layer="2",
            layer_kind=HiddenLayerKind.INDEX,
            site=HiddenSite.BLOCK_OUTPUT,
            module_path="tail",
        ),
    )


def _specialization() -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=2,
        hidden_size=2,
        dtype=_DTYPE,
        device=_DEVICE,
        max_prefill_tokens=4,
    )


def _rows(step_id: int) -> PackedPhaseRowBatch:
    return PackedPhaseRowStaging().stage(
        _PLAN_EPOCH,
        step_id,
        PackedRowPlanInputs(
            request_ids=("request",),
            scheduled_tokens=(3,),
            prompt_tokens=(3,),
            computed_tokens=(0,),
        ),
        prompt_indices={"request": 7},
        sample_indices={"request": 11},
    )


def test_capture_preserves_exact_resource_tensor_provenance() -> None:
    resources = _resources()
    model = _CaptureModel()
    with pytest.raises(AttributeError):
        RunnerPackedHiddenCapture(
            model=model,
            resources=(
                replace(resources[0], module_path="model.shared"),
            ),
            specialization=_specialization(),
            plan_epoch=_PLAN_EPOCH,
        )
    capture = RunnerPackedHiddenCapture(
        model=model,
        resources=resources,
        specialization=_specialization(),
        plan_epoch=_PLAN_EPOCH,
    )
    hidden = torch.arange(6, dtype=_DTYPE).reshape(3, 2)

    rows = _rows(13)
    capture.begin(rows)
    model(hidden)
    execution = _SnapshotExecution([])
    snapshot = capture.publish(execution)

    assert execution.batches == [snapshot]
    assert snapshot.row_batch is rows
    shared_expected = hidden + 1.0
    for resource in resources[:2]:
        torch.testing.assert_close(
            snapshot.hidden_for(resource), shared_expected
        )
    torch.testing.assert_close(
        snapshot.hidden_for(resources[2]), hidden + 4.0
    )
    capture.finish()

    empty_rows = PackedPhaseRowStaging().stage(
        _PLAN_EPOCH,
        14,
        PackedRowPlanInputs(("request",), (0,), (3,), (0,)),
        prompt_indices={"request": 7},
        sample_indices={"request": 11},
    )
    capture.begin(empty_rows)
    empty_snapshot = capture.publish()
    assert all(
        tuple(item.hidden.shape) == (0, 2)
        for item in empty_snapshot.snapshots
    )
    capture.finish()
    capture.close()


def test_capture_aborts_partial_forward_and_publish_failures_then_removes_hooks() -> None:
    resources = _resources()
    hidden = torch.ones((3, 2), dtype=_DTYPE)
    invalid_model = _InvalidCaptureModel()
    failed_forward = RunnerPackedHiddenCapture(
        model=invalid_model,
        resources=resources,
        specialization=_specialization(),
        plan_epoch=_PLAN_EPOCH,
    )
    rows = _rows(17)
    failed_forward.begin(rows)
    with pytest.raises(RuntimeError, match="Tensor output"):
        invalid_model(hidden)
    with pytest.raises(RuntimeError, match="already finished"):
        failed_forward.begin(rows)
    failed_forward.close()
    assert invalid_model(hidden) == "invalid"

    model = _CaptureModel()
    failed_publish = RunnerPackedHiddenCapture(
        model=model,
        resources=resources,
        specialization=_specialization(),
        plan_epoch=_PLAN_EPOCH,
    )
    publish_rows = _rows(19)
    failed_publish.begin(publish_rows)
    model(hidden)
    with pytest.raises(RuntimeError, match="execution rejected"):
        failed_publish.publish(_FailingSnapshotExecution())
    with pytest.raises(RuntimeError, match="already finished"):
        failed_publish.begin(publish_rows)
    driver = V027MRv1Driver()
    binding = RunnerBinding(
        driver=driver,
        runner=model,
        capabilities=driver.capabilities(),
        packed_hidden_capture=failed_publish,
    )
    assert model.shared._forward_hooks
    synchronize_runner_packed_hidden_capture(
        binding=binding,
        execution=None,
    )
    assert binding.packed_hidden_capture is None
    assert not model.shared._forward_hooks
    assert not model.tail._forward_hooks
