"""Row roles follow the prompt length; only the final row of a recomputed history is a new decode."""

import pytest

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import label_rows, tap_selection
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


def test_bulk_labels_links_and_tap_slots_match_retained_replay_and_slot_reuse():
    """Compare all native facts with the prior record semantics as membership and generations change."""
    import numpy as np

    from pllm.contracts.packed_rows import PackedPhaseRow, PackedSamplerRowLink, RowLabeler

    labeler = RowLabeler()
    identities = tuple(RequestIdentity(key, index, 0) for index, key in enumerate(("a", "b", "c")))
    for identity in identities[:2]:
        labeler.admit(identity)
    frontiers = {}
    for step, (members, counts, lengths, done) in enumerate(
        (
            ((0, 1), (2, 1), (4, 1), (0, 0)),
            ((1, 0), (1, 3), (1, 4), (1, 2)),
            ((0,), (3,), (4,), (0,)),  # partial retained replay
            ((0, 1), (3, 1), (4, 1), (3, 2)),  # replay crosses the prompt ending
            ((2, 0), (1, 1), (1, 4), (0, 6)),  # cancelled b's native ID is reused by c
        )
    ):
        if step == 4:
            labeler.retire(identities[1])
            labeler.admit(identities[2])
        active = tuple(identities[index] for index in members)
        ids = np.array([labeler.ids[identity.request_id] for identity in active], dtype=np.int64)
        actual = labeler.prepare(
            *(np.array(values, dtype=np.int32) for values in (counts, lengths, done)), ids, step, True
        )
        rows, links = [], []
        for ordinal, (identity, count, length, start) in enumerate(zip(active, counts, lengths, done, strict=True)):
            for offset in range(start, start + count):
                rows.append(
                    PackedPhaseRow(
                        len(rows),
                        ordinal,
                        offset,
                        HiddenPhase.PREFILL if offset < length else HiddenPhase.DECODE,
                        identity,
                        length,
                        offset < frontiers.get(identity, 0),
                    )
                )
            links.append(PackedSamplerRowLink(ordinal, len(rows) - 1))
            frontiers[identity] = max(frontiers.get(identity, 0), start + count)
        assert actual.row_batch.rows == tuple(rows) and actual.links == tuple(links)
        last = {row.request_ordinal: row for row in rows}
        for phases in (
            (HiddenPhase.DECODE,),
            (HiddenPhase.PREFILL,),
            (HiddenPhase.DECODE, HiddenPhase.PREFILL),
            (HiddenPhase.ALL,),
        ):
            kind = tap_selection(phases)
            selected = [
                row.packed_row_index
                for phase in phases
                for row in rows
                if not row.recomputed
                and (
                    phase is HiddenPhase.ALL
                    or row.phase is phase
                    and (
                        row is last[row.request_ordinal]
                        if phase is HiddenPhase.DECODE
                        else row.token_offset == row.prompt_length - 1
                    )
                )
            ]
            np.testing.assert_array_equal(actual.row_batch.tap_selected[kind], selected)
            slots = np.arange(len(selected), dtype=np.int64)[::-1] + step * 7
            expected = np.full(len(rows), slots[0] if slots.size else 0, dtype=np.int64)
            expected[selected] = slots
            mapped = slots[actual.row_batch.tap_ranks[kind]] if slots.size else np.zeros(len(rows), dtype=np.int64)
            np.testing.assert_array_equal(mapped, expected)
