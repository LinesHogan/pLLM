"""Row roles follow the prompt length; only the final row of a recomputed history is a new decode."""

from pllm import _core
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.request_identity import RequestIdentity


def test_resumed_history_is_generated_but_only_its_final_row_is_new():
    # vLLM resumes a preempted request (prompt 4, three outputs) by recomputing all seven tokens.
    plan = _core.plan_packed_prefill_rows([7], [4], [0], 7)
    identity = RequestIdentity("r", 0, 0)
    rows = tuple(
        PackedPhaseRow(packed, 0, offset, HiddenPhase(phase), identity)
        for packed, offset, phase in zip(plan.packed_row_indices, plan.token_offsets, plan.phases, strict=True)
    )
    assert [row.phase for row in rows] == [HiddenPhase.PREFILL] * 4 + [HiddenPhase.DECODE] * 3
    assert PackedPhaseRowBatch(1, 0, 7, rows).latest_decode_rows == (rows[6],)
