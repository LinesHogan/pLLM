"""Label vLLM MRv1/MRv2 input batches as packed phase rows; used by the runners.

compat.py admits only the vLLM releases in SUPPORTED_VLLM; the batch layout asserted here is the one they produce.
"""

import numpy as np
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu_input_batch import InputBatch as Mrv1InputBatch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import (
    PackedPhaseRow,
    PackedPhaseRowBatch,
    PackedSamplerRowBatch,
    PackedSamplerRowLink,
)
from pllm.contracts.request_identity import RequestIdentity


def encode_mrv2_rows(
    input_batch: InputBatch,
    prompt_lengths: np.ndarray,
    identities: tuple[RequestIdentity, ...],
    *,
    step_id: int,
) -> PackedSamplerRowBatch:
    """Label rows against each request's prompt; vLLM's prefill length also covers recomputed output history."""
    count = input_batch.num_reqs
    # An n=1, non-speculative batch: one logits row per request, no draft tokens.
    assert count >= 1 and input_batch.num_reqs_after_padding >= count
    assert input_batch.num_tokens_after_padding >= input_batch.num_tokens
    assert input_batch.num_draft_tokens == 0 and input_batch.num_draft_tokens_per_req is None
    columns = (
        input_batch.num_scheduled_tokens,
        input_batch.prefill_len_np,
        input_batch.num_computed_tokens_np,
        input_batch.is_prefilling_np,
        prompt_lengths,
    )
    assert all(len(column) == count for column in columns)
    assert np.array_equal(input_batch.cu_num_logits_np, np.arange(count + 1))
    logits = input_batch.logits_indices
    assert logits.ndim == 1 and logits.device.type == "cuda" and logits.numel() == count
    request_ids = tuple(identity.request_id for identity in identities)
    assert tuple(input_batch.req_ids) == request_ids and all(request_ids) and len(set(request_ids)) == count
    assert (input_batch.num_scheduled_tokens >= 1).all()
    assert np.array_equal(input_batch.is_prefilling_np, input_batch.num_computed_tokens_np < input_batch.prefill_len_np)
    assert input_batch.num_scheduled_tokens.sum() == input_batch.num_tokens
    return label_rows(
        identities,
        input_batch.num_scheduled_tokens.tolist(),
        prompt_lengths.tolist(),
        input_batch.num_computed_tokens_np.tolist(),
        step_id,
    )


def encode_mrv1_rows(
    batch: Mrv1InputBatch,
    scheduler: SchedulerOutput,
    logits_indices: torch.Tensor,
    identities: tuple[RequestIdentity, ...],
    *,
    step_id: int,
) -> PackedSamplerRowBatch:
    count = batch.num_reqs
    assert count >= 1 and tuple(batch.req_ids) == tuple(identity.request_id for identity in identities)
    assert len(set(batch.req_ids)) == count == len(identities) and not scheduler.scheduled_spec_decode_tokens
    assert logits_indices.shape == (count,) and logits_indices.device.type == "cuda"
    scheduled = [scheduler.num_scheduled_tokens[key] for key in batch.req_ids]
    assert min(scheduled) >= 1 and sum(scheduled) == scheduler.total_num_scheduled_tokens
    return label_rows(
        identities,
        scheduled,
        [int(value) for value in batch.num_prompt_tokens[: batch.num_reqs]],
        [int(value) for value in batch.num_computed_tokens_cpu[: batch.num_reqs]],
        step_id,
    )


def label_rows(
    identities: tuple[RequestIdentity, ...],
    scheduled: list[int],
    prompt: list[int],
    computed: list[int],
    step_id: int,
) -> PackedSamplerRowBatch:
    """One row per scheduled token in request order; token positions below the prompt length are prefill."""
    rows: list[PackedPhaseRow] = []
    links: list[PackedSamplerRowLink] = []
    for request, (count, length, done) in enumerate(zip(scheduled, prompt, computed, strict=True)):
        for offset in range(done, done + count):
            phase = HiddenPhase.PREFILL if offset < length else HiddenPhase.DECODE
            rows.append(PackedPhaseRow(len(rows), request, offset, phase, identities[request], length))
        # vLLM samples each request's last scheduled token, which is the last packed row of its span.
        links.append(PackedSamplerRowLink(request, len(rows) - 1))
    return PackedSamplerRowBatch(PackedPhaseRowBatch(step_id, len(rows), tuple(rows)), tuple(links))
