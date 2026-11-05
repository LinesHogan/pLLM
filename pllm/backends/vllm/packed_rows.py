"""Borrow vLLM's host columns for the native row-label and hidden-map operation.

The session interns identities at admission; MRv2 borrows its persistent slot mapping.
"""

import numpy as np
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu_input_batch import InputBatch as Mrv1InputBatch

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


def encode_mrv1_rows(
    batch: Mrv1InputBatch,
    scheduler: SchedulerOutput,
    logits_indices: torch.Tensor,
    labeler: RowLabeler,
    *,
    step_id: int,
    keep: bool,
) -> PackedSamplerRowBatch:
    """Adapt MRv1's persistent request order to the same native bulk operation."""
    count = batch.num_reqs
    assert count >= 1 and not scheduler.scheduled_spec_decode_tokens
    assert logits_indices.shape == (count,) and logits_indices.device.type == "cuda"
    scheduled = np.fromiter(map(scheduler.num_scheduled_tokens.__getitem__, batch.req_ids), dtype=np.int32, count=count)
    ids = np.fromiter(map(labeler.ids.__getitem__, batch.req_ids), dtype=np.int64, count=count)
    return labeler.prepare(
        scheduled,
        batch.num_prompt_tokens[:count],
        batch.num_computed_tokens_cpu[:count],
        ids,
        step_id,
        keep,
    )
