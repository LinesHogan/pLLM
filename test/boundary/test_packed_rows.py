"""Row roles follow the prompt length; only the final row of a recomputed history is a new decode."""

import pytest

from pllm import _core
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.request_identity import RequestIdentity


def test_resumed_history_is_generated_but_only_its_final_row_is_new():
    # vLLM resumes a preempted request (prompt 4, three outputs) by recomputing all seven tokens.
    plan = _core.plan_packed_prefill_rows([7], [4], [0], 7)
    identity = RequestIdentity("r", 0, 0)
    rows = tuple(
        PackedPhaseRow(packed, 0, offset, HiddenPhase(phase), identity, plan.prompt_lengths[packed])
        for packed, offset, phase in zip(plan.packed_row_indices, plan.token_offsets, plan.phases, strict=True)
    )
    assert [row.phase for row in rows] == [HiddenPhase.PREFILL] * 4 + [HiddenPhase.DECODE] * 3
    assert PackedPhaseRowBatch(1, 0, 7, rows).latest_decode_rows == (rows[6],)

    assert plan.prompt_lengths == [4] * 7
    assert PackedPhaseRowBatch(1, 0, 7, rows).producer_rows(HiddenPhase.PREFILL) == (rows[3],)
    assert _core.plan_packed_prefill_rows([2], [4], [0], 2).prompt_lengths == [4, 4]


@pytest.mark.parametrize("length", [1, 75, 127, 128, 129, 2047, 2048, 2049])
@pytest.mark.parametrize("budget", [1, 127, 128, 129, 512])
def test_prompt_chunks_select_one_completion_among_reordered_decode_rows(length, budget):
    identities = (RequestIdentity("decode", 1, 0), RequestIdentity("prompt", 0, 0))
    endings = []
    for offset in range(0, length, budget):
        count = min(budget, length - offset)
        plan = _core.plan_packed_prefill_rows([1, count], [1, length], [10 + offset, offset], count + 1)
        rows = tuple(
            PackedPhaseRow(packed, request, token, HiddenPhase(phase), identities[request], prompt)
            for packed, request, token, phase, prompt in zip(
                plan.packed_row_indices,
                plan.request_ordinals,
                plan.token_offsets,
                plan.phases,
                plan.prompt_lengths,
                strict=True,
            )
        )
        batch = PackedPhaseRowBatch(1, offset, len(rows), rows)
        endings.extend(batch.producer_rows(HiddenPhase.PREFILL))
        assert batch.producer_rows(HiddenPhase.DECODE) == (rows[0],)
    assert len(endings) == 1 and endings[0].token_offset == length - 1
