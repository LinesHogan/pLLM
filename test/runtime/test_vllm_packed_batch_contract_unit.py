"""Closed vLLM 0.27 packed-batch contract for Region planning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tllm.backends.vllm import runner_adapter
from tllm.backends.vllm.v027_mrv1 import V027MRv1Driver
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver


def _mrv1_batch(*, indices: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=["request-z", "request-a"],
        num_reqs=2,
        req_id_to_index=indices,
        num_prompt_tokens=[7, 3],
        num_computed_tokens_cpu=[2, 0],
    )


def _mrv2_batch() -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=["request-z", "request-a"],
        num_reqs=2,
        prefill_len_np=[7, 3],
        num_computed_tokens_np=[2, 0],
    )


def test_v027_drivers_preserve_the_explicit_compact_request_order() -> None:
    """Region planning receives vLLM's compact prefix, never a repaired one."""

    mrv1 = V027MRv1Driver().packed_row_plan_inputs(
        runner=SimpleNamespace(),
        num_scheduled_tokens=[2, 1],
        input_batch=_mrv1_batch(indices={"request-z": 0, "request-a": 1}),
    )
    mrv2 = V027MRv2Driver().packed_row_plan_inputs(
        runner=SimpleNamespace(),
        num_scheduled_tokens=[2, 1],
        input_batch=_mrv2_batch(),
    )

    expected = runner_adapter.PackedRowPlanInputs(
        request_ids=("request-z", "request-a"),
        scheduled_tokens=(2, 1),
        prompt_tokens=(7, 3),
        computed_tokens=(2, 0),
    )
    assert mrv1 == expected
    assert mrv2 == expected


def test_empty_batches_do_not_require_family_private_columns() -> None:
    empty = SimpleNamespace(req_ids=(), num_reqs=0)

    assert runner_adapter.mrv1_packed_row_plan_inputs(
        runner=SimpleNamespace(),
        num_scheduled_tokens=(),
        input_batch=empty,
    ) == runner_adapter.PackedRowPlanInputs((), (), (), ())
    assert runner_adapter.mrv2_packed_row_plan_inputs(
        runner=SimpleNamespace(),
        num_scheduled_tokens=(),
        input_batch=empty,
    ) == runner_adapter.PackedRowPlanInputs((), (), (), ())


def test_noncompact_request_indices_fail_instead_of_becoming_python_ordinals() -> None:
    with pytest.raises(RuntimeError, match="compact request-ordered prefix"):
        runner_adapter.mrv1_packed_row_plan_inputs(
            runner=SimpleNamespace(),
            num_scheduled_tokens=[2, 1],
            input_batch=_mrv1_batch(indices={"request-z": 4, "request-a": 9}),
        )


def test_incomplete_active_prefix_fails_instead_of_compacting_slots_in_python() -> None:
    with pytest.raises(RuntimeError, match="complete active request prefix"):
        runner_adapter.mrv2_packed_row_plan_inputs(
            runner=SimpleNamespace(),
            num_scheduled_tokens=[2, 1],
            input_batch=SimpleNamespace(
                req_ids=["request-z"],
                num_reqs=2,
                prefill_len_np=[7, 3],
                num_computed_tokens_np=[2, 0],
            ),
        )


def test_empty_active_request_id_is_rejected_without_string_coercion() -> None:
    with pytest.raises(RuntimeError, match="empty active request slot"):
        runner_adapter.mrv2_packed_row_plan_inputs(
            runner=SimpleNamespace(),
            num_scheduled_tokens=[2, 1],
            input_batch=SimpleNamespace(
                req_ids=["request-z", None],
                num_reqs=2,
                prefill_len_np=[7, 3],
                num_computed_tokens_np=[2, 0],
            ),
        )


def test_mrv1_rejects_mrv2_private_batch_layout() -> None:
    with pytest.raises(AttributeError, match="req_id_to_index"):
        runner_adapter.mrv1_packed_row_plan_inputs(
            runner=SimpleNamespace(),
            num_scheduled_tokens=[2, 1],
            input_batch=_mrv2_batch(),
        )


def test_mrv2_rejects_mrv1_private_batch_layout() -> None:
    with pytest.raises(AttributeError, match="prefill_len_np"):
        runner_adapter.mrv2_packed_row_plan_inputs(
            runner=SimpleNamespace(),
            num_scheduled_tokens=[2, 1],
            input_batch=_mrv1_batch(indices={"request-z": 0, "request-a": 1}),
        )
