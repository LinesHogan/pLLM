"""Strict contracts for Rust-planned packed request rows."""

from __future__ import annotations

import pytest

from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.packed_rows import PackedRowPlanInputs
from tllm.contracts.request_identity import RequestIdentity
from tllm.runtime.packed_phase_row_staging import PackedPhaseRowStaging


def test_staging_expands_two_complete_three_token_prompts() -> None:
    staging = PackedPhaseRowStaging()
    batch = staging.stage(
        1,
        11,
        PackedRowPlanInputs(
            request_ids=("request-a", "request-b"),
            scheduled_tokens=(3, 3),
            prompt_tokens=(3, 3),
            computed_tokens=(0, 0),
        ),
        prompt_indices={"request-a": 17, "request-b": 29},
        sample_indices={},
    )

    assert (batch.plan_epoch, batch.step_id, batch.total_rows) == (1, 11, 6)
    assert batch.prefill_rows == batch.rows
    assert batch.decode_rows == ()
    assert tuple(row.packed_row_index for row in batch.rows) == tuple(range(6))
    assert tuple(row.request_ordinal for row in batch.rows) == (0, 0, 0, 1, 1, 1)
    assert tuple(row.raw_request_id for row in batch.rows) == (
        ("request-a",) * 3 + ("request-b",) * 3
    )
    assert tuple(row.token_offset for row in batch.rows) == (0, 1, 2, 0, 1, 2)
    assert tuple(row.phase for row in batch.rows) == (HiddenPhase.PREFILL,) * 6
    assert batch.identities == (
        (RequestIdentity("request-a", 17, 0),) * 3
        + (RequestIdentity("request-b", 29, 0),) * 3
    )
    assert tuple(row.decode_plan_ordinal for row in batch.rows) == (None,) * 6
    assert staging.batch_for_step(1, 11) is batch
    staging.finish_step(1, 11)
    with pytest.raises(RuntimeError, match="not staged"):
        staging.batch_for_step(1, 11)


def test_staging_preserves_decode_and_mixed_plan_order() -> None:
    decode = PackedPhaseRowStaging().stage(
        1,
        12,
        PackedRowPlanInputs(
            request_ids=("request-a", "request-b"),
            scheduled_tokens=(2, 1),
            prompt_tokens=(3, 2),
            computed_tokens=(3, 2),
        ),
        prompt_indices={"request-a": 17, "request-b": 29},
        sample_indices={},
    )
    assert decode.prefill_rows == ()
    assert tuple(row.phase for row in decode.rows) == (HiddenPhase.DECODE,) * 3
    assert tuple(row.decode_plan_ordinal for row in decode.rows) == (0, 1, 2)
    assert tuple(row.token_offset for row in decode.rows) == (3, 4, 2)

    mixed = PackedPhaseRowStaging().stage(
        1,
        13,
        PackedRowPlanInputs(
            request_ids=("request-a", "request-b"),
            scheduled_tokens=(3, 3),
            prompt_tokens=(3, 3),
            computed_tokens=(0, 2),
        ),
        prompt_indices={"request-a": 17, "request-b": 29},
        sample_indices={},
    )
    assert tuple(row.phase for row in mixed.rows) == (
        HiddenPhase.PREFILL,
        HiddenPhase.PREFILL,
        HiddenPhase.PREFILL,
        HiddenPhase.PREFILL,
        HiddenPhase.DECODE,
        HiddenPhase.DECODE,
    )
    assert tuple(row.packed_row_index for row in mixed.prefill_rows) == (0, 1, 2, 3)
    assert tuple(row.packed_row_index for row in mixed.decode_rows) == (4, 5)
    assert tuple(row.decode_plan_ordinal for row in mixed.rows) == (
        None,
        None,
        None,
        None,
        0,
        1,
    )


def test_staging_preserves_raw_children_and_stable_parent_identities() -> None:
    batch = PackedPhaseRowStaging().stage(
        1,
        14,
        PackedRowPlanInputs(
            request_ids=("0_parent", "1_parent"),
            scheduled_tokens=(1, 1),
            prompt_tokens=(1, 1),
            computed_tokens=(0, 0),
        ),
        prompt_indices={"parent": 41},
        sample_indices={},
    )

    assert tuple(row.raw_request_id for row in batch.rows) == (
        "0_parent",
        "1_parent",
    )
    assert batch.identities == (
        RequestIdentity("parent", 41, 0),
        RequestIdentity("parent", 41, 1),
    )


def test_staging_epoch_identity_is_exact_and_restage_is_tamper_evident() -> None:
    columns = PackedRowPlanInputs(
        request_ids=("request-a",),
        scheduled_tokens=(1,),
        prompt_tokens=(1,),
        computed_tokens=(0,),
    )
    staging = PackedPhaseRowStaging()
    epoch_one = staging.stage(
        1,
        7,
        columns,
        prompt_indices={"request-a": 3},
        sample_indices={},
    )
    assert staging.batch_for_step(1, 7) is epoch_one
    with pytest.raises(RuntimeError, match="active plan epoch"):
        staging.stage(
            2,
            7,
            columns,
            prompt_indices={"request-a": 3},
            sample_indices={},
        )
    staging.finish_step(1, 7)
    epoch_two = staging.stage(
        2,
        7,
        columns,
        prompt_indices={"request-a": 3},
        sample_indices={},
    )
    assert staging.batch_for_step(2, 7) is epoch_two

    changed = PackedRowPlanInputs(
        request_ids=("request-a",),
        scheduled_tokens=(2,),
        prompt_tokens=(2,),
        computed_tokens=(0,),
    )
    with pytest.raises(RuntimeError, match="differs"):
        staging.stage(
            2,
            7,
            changed,
            prompt_indices={"request-a": 3},
            sample_indices={},
        )
    staging.finish_step(2, 7)
    with pytest.raises(RuntimeError, match="already retired"):
        staging.stage(
            2,
            7,
            changed,
            prompt_indices={"request-a": 3},
            sample_indices={},
        )
    with pytest.raises(RuntimeError, match="prior plan epoch"):
        staging.stage(
            1,
            8,
            columns,
            prompt_indices={"request-a": 3},
            sample_indices={},
        )


def test_staging_fails_closed_on_bad_identity_epoch_and_column_shape() -> None:
    columns = PackedRowPlanInputs(
        request_ids=("request-a",),
        scheduled_tokens=(1,),
        prompt_tokens=(1,),
        computed_tokens=(0,),
    )
    staging = PackedPhaseRowStaging()
    with pytest.raises(RuntimeError, match="stable request identity"):
        staging.stage(
            1,
            15,
            columns,
            prompt_indices={},
            sample_indices={},
        )
    with pytest.raises(ValueError, match="plan epoch"):
        staging.stage(
            0,
            15,
            columns,
            prompt_indices={"request-a": 1},
            sample_indices={},
        )
    with pytest.raises(ValueError, match="resolved request identity"):
        staging.stage(
            1,
            16,
            columns,
            prompt_indices={"request-a": -1},
            sample_indices={},
        )
    with pytest.raises(ValueError, match="equal lengths"):
        PackedRowPlanInputs(
            request_ids=("request-a", "request-b"),
            scheduled_tokens=(1,),
            prompt_tokens=(1,),
            computed_tokens=(0,),
        )
    with pytest.raises(ValueError, match="unique"):
        PackedRowPlanInputs(
            request_ids=("request-a", "request-a"),
            scheduled_tokens=(1, 1),
            prompt_tokens=(1, 1),
            computed_tokens=(0, 0),
        )
