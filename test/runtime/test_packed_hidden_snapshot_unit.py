"""Strict eager snapshot contracts for packed source/target hidden rows."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tllm.contracts.packed_hidden import (
    PackedHiddenRole,
    PackedHiddenSnapshotBatch,
    PackedHiddenSnapshotSpecialization,
)
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
)
from tllm.runtime.packed_hidden_snapshot import PackedHiddenSnapshotService
from tllm.runtime.packed_phase_row_staging import PackedPhaseRowStaging


_DEVICE = torch.device("cpu")
_DTYPE = torch.float32


def _row_batch(*, plan_epoch: int = 3, step_id: int = 7) -> PackedPhaseRowBatch:
    return PackedPhaseRowStaging().stage(
        plan_epoch,
        step_id,
        PackedRowPlanInputs(
            request_ids=("request-a", "request-b"),
            scheduled_tokens=(3, 3),
            prompt_tokens=(3, 3),
            computed_tokens=(0, 0),
        ),
        prompt_indices={"request-a": 11, "request-b": 19},
        sample_indices={},
    )


def _service(
    *, plan_epoch: int = 3, max_packed_rows: int = 8
) -> PackedHiddenSnapshotService:
    return PackedHiddenSnapshotService(
        PackedHiddenSnapshotSpecialization(
            plan_epoch=plan_epoch,
            max_packed_rows=max_packed_rows,
            hidden_size=2,
            dtype=_DTYPE,
            device=_DEVICE,
        )
    )


def test_snapshot_owns_exact_ordinary_tensors_and_batch_provenance() -> None:
    batch = _row_batch()
    service = _service()
    expected_source = torch.arange(12, dtype=_DTYPE).reshape(6, 2)
    expected_target = -expected_source
    with torch.inference_mode():
        source = expected_source.clone()
        target = expected_target.clone()
    assert torch.is_inference(source)
    assert torch.is_inference(target)

    service.stage(batch, PackedHiddenRole.SOURCE, source)
    service.stage(batch, PackedHiddenRole.TARGET, target)
    with torch.inference_mode():
        source.fill_(101.0)
        target.fill_(202.0)
    snapshot = service.snapshot_for_step(3, 7)

    assert snapshot.row_batch is batch
    assert (snapshot.plan_epoch, snapshot.step_id) == (3, 7)
    assert not torch.is_inference(snapshot.source_hidden)
    assert not torch.is_inference(snapshot.target_hidden)
    torch.testing.assert_close(snapshot.source_hidden, expected_source)
    torch.testing.assert_close(snapshot.target_hidden, expected_target)
    assert snapshot.source_hidden.data_ptr() != source.data_ptr()
    assert snapshot.target_hidden.data_ptr() != target.data_ptr()
    assert snapshot.source_hidden.data_ptr() != snapshot.target_hidden.data_ptr()

    snapshot.source_hidden.zero_()
    fresh = service.snapshot_for_step(3, 7)
    torch.testing.assert_close(fresh.source_hidden, expected_source)


def test_source_and_target_accept_equal_batches_and_keep_canonical_ref() -> None:
    canonical = _row_batch()
    equal_reconstruction = replace(canonical)
    assert equal_reconstruction == canonical
    assert equal_reconstruction is not canonical
    service = _service()
    hidden = torch.arange(12, dtype=_DTYPE).reshape(6, 2)

    service.stage(canonical, PackedHiddenRole.SOURCE, hidden)
    service.stage(
        equal_reconstruction,
        PackedHiddenRole.TARGET,
        -hidden,
    )
    snapshot = service.snapshot_for_step(3, 7)
    assert snapshot.row_batch is canonical
    assert snapshot.source_hidden.shape == snapshot.target_hidden.shape

    changed_rows = (
        replace(canonical.rows[0], token_offset=99),
        *canonical.rows[1:],
    )
    different = replace(canonical, rows=changed_rows)
    different_service = _service()
    different_service.stage(
        canonical, PackedHiddenRole.SOURCE, hidden
    )
    with pytest.raises(RuntimeError, match="same row batch"):
        different_service.stage(
            different,
            PackedHiddenRole.TARGET,
            -hidden,
        )


def test_target_may_arrive_before_source_with_one_canonical_batch() -> None:
    canonical = _row_batch(step_id=8)
    equal_reconstruction = replace(canonical)
    service = _service()
    hidden = torch.arange(12, dtype=_DTYPE).reshape(6, 2)

    service.stage(
        equal_reconstruction,
        PackedHiddenRole.TARGET,
        -hidden,
    )
    service.stage(canonical, PackedHiddenRole.SOURCE, hidden)
    snapshot = service.snapshot_for_step(3, 8)
    assert snapshot.row_batch is equal_reconstruction
    torch.testing.assert_close(snapshot.source_hidden, hidden)
    torch.testing.assert_close(snapshot.target_hidden, -hidden)


def test_snapshot_rejects_capacity_shape_dtype_epoch_and_duplicate_role() -> None:
    batch = _row_batch()
    hidden = torch.zeros((6, 2), dtype=_DTYPE)
    with pytest.raises(RuntimeError, match="specialized capacity"):
        _service(max_packed_rows=5).stage(
            batch, PackedHiddenRole.SOURCE, hidden
        )

    service = _service()
    with pytest.raises(RuntimeError, match="tensor shape"):
        service.stage(
            batch,
            PackedHiddenRole.SOURCE,
            torch.zeros((7, 2), dtype=_DTYPE),
        )
    with pytest.raises(RuntimeError, match="tensor shape"):
        service.stage(
            batch,
            PackedHiddenRole.SOURCE,
            torch.zeros((6, 3), dtype=_DTYPE),
        )
    with pytest.raises(RuntimeError, match="dtype or device"):
        service.stage(
            batch,
            PackedHiddenRole.SOURCE,
            torch.zeros((6, 2), dtype=torch.float64),
        )
    with pytest.raises(RuntimeError, match="another plan epoch"):
        _service(plan_epoch=4).stage(
            batch, PackedHiddenRole.SOURCE, hidden
        )

    service.stage(batch, PackedHiddenRole.SOURCE, hidden)
    with pytest.raises(RuntimeError, match="already staged"):
        service.stage(batch, PackedHiddenRole.SOURCE, hidden)
    with pytest.raises(RuntimeError, match="not both staged"):
        service.finish_step(3, 7)
    service.stage(batch, PackedHiddenRole.TARGET, hidden)
    service.finish_step(3, 7)


def test_finish_high_watermark_prevents_step_aba() -> None:
    finished = _row_batch(step_id=10)
    older = _row_batch(step_id=9)
    service = _service()
    hidden = torch.zeros((6, 2), dtype=_DTYPE)
    service.stage(finished, PackedHiddenRole.SOURCE, hidden)
    service.stage(finished, PackedHiddenRole.TARGET, hidden)
    service.finish_step(3, 10)

    with pytest.raises(RuntimeError, match="already finished"):
        service.stage(finished, PackedHiddenRole.SOURCE, hidden)
    with pytest.raises(RuntimeError, match="already finished"):
        service.stage(older, PackedHiddenRole.SOURCE, hidden)
    with pytest.raises(RuntimeError, match="not both staged"):
        service.snapshot_for_step(3, 10)


def test_high_watermark_does_not_invalidate_an_older_active_provenance() -> None:
    older = _row_batch(step_id=11)
    newer = _row_batch(step_id=12)
    service = _service()
    hidden = torch.zeros((6, 2), dtype=_DTYPE)
    service.stage(older, PackedHiddenRole.SOURCE, hidden)
    service.stage(newer, PackedHiddenRole.SOURCE, hidden)
    service.stage(newer, PackedHiddenRole.TARGET, hidden)
    service.finish_step(3, 12)

    service.stage(older, PackedHiddenRole.TARGET, hidden)
    service.finish_step(3, 11)
    with pytest.raises(RuntimeError, match="already finished"):
        service.stage(older, PackedHiddenRole.SOURCE, hidden)


def test_snapshot_specialization_requires_explicit_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max packed rows"):
        PackedHiddenSnapshotSpecialization(
            plan_epoch=1,
            max_packed_rows=0,
            hidden_size=2,
            dtype=_DTYPE,
            device=_DEVICE,
        )


def test_public_snapshot_batch_rejects_borrowed_inference_tensors() -> None:
    batch = _row_batch()
    with torch.inference_mode():
        source = torch.zeros((6, 2), dtype=_DTYPE)
        target = torch.zeros((6, 2), dtype=_DTYPE)
    with pytest.raises(ValueError, match="ordinary tensors"):
        PackedHiddenSnapshotBatch(batch, source, target)
