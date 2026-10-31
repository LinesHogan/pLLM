"""Row roles follow the prompt length; only the final row of a recomputed history is a new decode."""

import pytest

from pllm.backends.vllm.packed_rows import label_rows
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.request_identity import RequestIdentity


def test_resumed_history_is_generated_but_only_its_final_row_is_new():
    # vLLM resumes a preempted request (prompt 4, three outputs) by recomputing all seven tokens.
    batch = label_rows((RequestIdentity("r", 0, 0),), [7], [4], [0], 0).row_batch
    rows = batch.rows
    assert [row.phase for row in rows] == [HiddenPhase.PREFILL] * 4 + [HiddenPhase.DECODE] * 3
    assert batch.latest_decode_rows == (rows[6],)
    assert [row.prompt_length for row in rows] == [4] * 7
    assert batch.producer_rows(HiddenPhase.PREFILL) == (rows[3],)


@pytest.mark.parametrize("length", [1, 75, 127, 128, 129, 2047, 2048, 2049])
@pytest.mark.parametrize("budget", [1, 127, 128, 129, 512])
def test_prompt_chunks_select_one_completion_among_reordered_decode_rows(length, budget):
    identities = (RequestIdentity("decode", 1, 0), RequestIdentity("prompt", 0, 0))
    endings = []
    for offset in range(0, length, budget):
        count = min(budget, length - offset)
        batch = label_rows(identities, [1, count], [1, length], [10 + offset, offset], offset).row_batch
        endings.extend(batch.producer_rows(HiddenPhase.PREFILL))
        assert batch.producer_rows(HiddenPhase.DECODE) == (batch.rows[0],)
    assert len(endings) == 1 and endings[0].token_offset == length - 1
