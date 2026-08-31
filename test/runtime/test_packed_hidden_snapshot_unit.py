"""Strict eager snapshot contracts for resource-keyed packed hidden rows."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
    PackedHiddenSnapshotSpecialization,
)
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
)
from tllm.runtime.packed_hidden_snapshot import PackedHiddenSnapshotService
from tllm.runtime.packed_phase_row_staging import PackedPhaseRowStaging


_DEVICE = torch.device("cpu")
_DTYPE = torch.float32
_FIRST_RESOURCE = PackedHiddenResourceBinding(
    layer="-1",
    layer_kind=HiddenLayerKind.INDEX,
    site=HiddenSite.BLOCK_OUTPUT,
    module_path="model.layers.-1",
)
_SECOND_RESOURCE = PackedHiddenResourceBinding(
    layer="0",
    layer_kind=HiddenLayerKind.INDEX,
    site=HiddenSite.BLOCK_OUTPUT,
    module_path="model.layers.0",
)
_RESOURCES = (_FIRST_RESOURCE, _SECOND_RESOURCE)


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
    *,
    plan_epoch: int = 3,
    max_packed_rows: int = 8,
    resources: tuple[PackedHiddenResourceBinding, ...] = _RESOURCES,
) -> PackedHiddenSnapshotService:
    return PackedHiddenSnapshotService(
        PackedHiddenSnapshotSpecialization(
            plan_epoch=plan_epoch,
            max_packed_rows=max_packed_rows,
            hidden_size=2,
            resources=resources,
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

    service.stage(batch, _FIRST_RESOURCE, source)
    service.stage(batch, _SECOND_RESOURCE, target)
    with torch.inference_mode():
        source.fill_(101.0)
        target.fill_(202.0)
    snapshot = service.snapshot_for_step(3, 7)

    assert snapshot.row_batch is batch
    assert (snapshot.plan_epoch, snapshot.step_id) == (3, 7)
    source_snapshot = snapshot.hidden_for(_FIRST_RESOURCE)
    target_snapshot = snapshot.hidden_for(_SECOND_RESOURCE)
    assert not torch.is_inference(source_snapshot)
    assert not torch.is_inference(target_snapshot)
    torch.testing.assert_close(source_snapshot, expected_source)
    torch.testing.assert_close(target_snapshot, expected_target)
    assert source_snapshot.data_ptr() != source.data_ptr()
    assert target_snapshot.data_ptr() != target.data_ptr()
    assert source_snapshot.data_ptr() != target_snapshot.data_ptr()

    source_snapshot.zero_()
    fresh = service.snapshot_for_step(3, 7)
    torch.testing.assert_close(
        fresh.hidden_for(_FIRST_RESOURCE), expected_source
    )


def test_resources_accept_equal_batches_and_keep_canonical_ref() -> None:
    canonical = _row_batch()
    equal_reconstruction = replace(canonical)
    assert equal_reconstruction == canonical
    assert equal_reconstruction is not canonical
    service = _service()
    hidden = torch.arange(12, dtype=_DTYPE).reshape(6, 2)

    service.stage(canonical, _FIRST_RESOURCE, hidden)
    service.stage(
        equal_reconstruction,
        _SECOND_RESOURCE,
        -hidden,
    )
    snapshot = service.snapshot_for_step(3, 7)
    assert snapshot.row_batch is canonical
    assert (
        snapshot.hidden_for(_FIRST_RESOURCE).shape
        == snapshot.hidden_for(_SECOND_RESOURCE).shape
    )

    changed_rows = (
        replace(canonical.rows[0], token_offset=99),
        *canonical.rows[1:],
    )
    different = replace(canonical, rows=changed_rows)
    different_service = _service()
    different_service.stage(
        canonical, _FIRST_RESOURCE, hidden
    )
    with pytest.raises(RuntimeError, match="same row batch"):
        different_service.stage(
            different,
            _SECOND_RESOURCE,
            -hidden,
        )


def test_resources_may_arrive_in_any_order_with_one_canonical_batch() -> None:
    canonical = _row_batch(step_id=8)
    equal_reconstruction = replace(canonical)
    service = _service()
    hidden = torch.arange(12, dtype=_DTYPE).reshape(6, 2)

    service.stage(
        equal_reconstruction,
        _SECOND_RESOURCE,
        -hidden,
    )
    service.stage(canonical, _FIRST_RESOURCE, hidden)
    snapshot = service.snapshot_for_step(3, 8)
    assert snapshot.row_batch is equal_reconstruction
    torch.testing.assert_close(snapshot.hidden_for(_FIRST_RESOURCE), hidden)
    torch.testing.assert_close(snapshot.hidden_for(_SECOND_RESOURCE), -hidden)


def test_snapshot_rejects_capacity_shape_dtype_epoch_and_duplicate_resource() -> None:
    batch = _row_batch()
    hidden = torch.zeros((6, 2), dtype=_DTYPE)
    with pytest.raises(RuntimeError, match="specialized capacity"):
        _service(max_packed_rows=5).stage(
            batch, _FIRST_RESOURCE, hidden
        )

    service = _service()
    with pytest.raises(RuntimeError, match="tensor shape"):
        service.stage(
            batch,
            _FIRST_RESOURCE,
            torch.zeros((7, 2), dtype=_DTYPE),
        )
    with pytest.raises(RuntimeError, match="tensor shape"):
        service.stage(
            batch,
            _FIRST_RESOURCE,
            torch.zeros((6, 3), dtype=_DTYPE),
        )
    with pytest.raises(RuntimeError, match="dtype or device"):
        service.stage(
            batch,
            _FIRST_RESOURCE,
            torch.zeros((6, 2), dtype=torch.float64),
        )
    with pytest.raises(RuntimeError, match="another plan epoch"):
        _service(plan_epoch=4).stage(
            batch, _FIRST_RESOURCE, hidden
        )

    service.stage(batch, _FIRST_RESOURCE, hidden)
    with pytest.raises(RuntimeError, match="already staged"):
        service.stage(batch, _FIRST_RESOURCE, hidden)
    with pytest.raises(RuntimeError, match="not all signed"):
        service.finish_step(3, 7)
    service.stage(batch, _SECOND_RESOURCE, hidden)
    service.finish_step(3, 7)


def test_finish_high_watermark_prevents_step_aba() -> None:
    finished = _row_batch(step_id=10)
    older = _row_batch(step_id=9)
    service = _service()
    hidden = torch.zeros((6, 2), dtype=_DTYPE)
    service.stage(finished, _FIRST_RESOURCE, hidden)
    service.stage(finished, _SECOND_RESOURCE, hidden)
    service.finish_step(3, 10)

    with pytest.raises(RuntimeError, match="already finished"):
        service.stage(finished, _FIRST_RESOURCE, hidden)
    with pytest.raises(RuntimeError, match="already finished"):
        service.stage(older, _FIRST_RESOURCE, hidden)
    with pytest.raises(RuntimeError, match="not staged"):
        service.snapshot_for_step(3, 10)


def test_high_watermark_does_not_invalidate_an_older_active_provenance() -> None:
    older = _row_batch(step_id=11)
    newer = _row_batch(step_id=12)
    service = _service()
    hidden = torch.zeros((6, 2), dtype=_DTYPE)
    service.stage(older, _FIRST_RESOURCE, hidden)
    service.stage(newer, _FIRST_RESOURCE, hidden)
    service.stage(newer, _SECOND_RESOURCE, hidden)
    service.finish_step(3, 12)

    service.stage(older, _SECOND_RESOURCE, hidden)
    service.finish_step(3, 11)
    with pytest.raises(RuntimeError, match="already finished"):
        service.stage(older, _FIRST_RESOURCE, hidden)


def test_snapshot_specialization_requires_explicit_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max packed rows"):
        PackedHiddenSnapshotSpecialization(
            plan_epoch=1,
            max_packed_rows=0,
            hidden_size=2,
            resources=_RESOURCES,
            dtype=_DTYPE,
            device=_DEVICE,
        )


def test_public_snapshot_batch_rejects_borrowed_inference_tensors() -> None:
    batch = _row_batch()
    with torch.inference_mode():
        source = torch.zeros((6, 2), dtype=_DTYPE)
        target = torch.zeros((6, 2), dtype=_DTYPE)
    with pytest.raises(ValueError, match="ordinary tensors"):
        PackedHiddenSnapshotBatch(
            batch,
            (
                PackedHiddenSnapshot(_FIRST_RESOURCE, source),
                PackedHiddenSnapshot(_SECOND_RESOURCE, target),
            ),
        )


def test_snapshot_service_supports_generic_resource_arity() -> None:
    third = PackedHiddenResourceBinding(
        layer="teacher",
        layer_kind=HiddenLayerKind.PATH,
        site=HiddenSite.BLOCK_OUTPUT,
        module_path="model.teacher",
    )
    resources = (*_RESOURCES, third)
    service = _service(resources=resources)
    batch = _row_batch()
    tensors = tuple(
        torch.full((6, 2), float(index), dtype=_DTYPE)
        for index in range(len(resources))
    )

    for resource, tensor in zip(resources, tensors, strict=True):
        service.stage(batch, resource, tensor)
    snapshot = service.snapshot_for_step(3, 7)

    assert tuple(item.resource for item in snapshot.snapshots) == resources
    for resource, expected in zip(resources, tensors, strict=True):
        torch.testing.assert_close(snapshot.hidden_for(resource), expected)


def test_abort_retires_partial_snapshot_without_publication() -> None:
    service = _service()
    batch = _row_batch(step_id=21)
    hidden = torch.zeros((6, 2), dtype=_DTYPE)
    service.stage(batch, _FIRST_RESOURCE, hidden)

    service.abort_step(3, 21)

    with pytest.raises(RuntimeError, match="not staged"):
        service.snapshot_for_step(3, 21)
    with pytest.raises(RuntimeError, match="already finished"):
        service.stage(batch, _SECOND_RESOURCE, hidden)
