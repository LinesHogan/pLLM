"""Borrow vLLM's host columns for the native row-label and hidden-map operation.

The session interns identities at admission; MRv2 borrows its persistent slot mapping.
"""

import numpy as np
from vllm.v1.worker.gpu.input_batch import InputBatch

from pllm.contracts.packed_rows import PackedSamplerRowBatch, RowLabeler


def encode_mrv2_rows(
    input_batch: InputBatch,
    prompt_lengths: np.ndarray,
    labeler: RowLabeler,
    *,
    step_id: int,
    keep: bool,
) -> PackedSamplerRowBatch:
    """Label actual prompt positions, including generated history in recomputed prefills."""
    count = input_batch.num_reqs
    assert count >= 1 and input_batch.num_reqs_after_padding >= count
    assert input_batch.num_tokens_after_padding >= input_batch.num_tokens
    assert input_batch.num_draft_tokens == 0 and input_batch.num_draft_tokens_per_req is None
    assert np.array_equal(input_batch.cu_num_logits_np, np.arange(count + 1))
    logits = input_batch.logits_indices
    assert logits.ndim == 1 and logits.device.type == "cuda" and logits.numel() == count
    return labeler.prepare(
        input_batch.num_scheduled_tokens,
        prompt_lengths,
        input_batch.num_computed_tokens_np,
        labeler.slots[input_batch.idx_mapping_np],
        step_id,
        keep,
    )
