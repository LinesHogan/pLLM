"""Label vLLM MRv1/MRv2 input batches as packed phase rows through the Rust row plan; used by the runners.

create_vllm_llm pins the vLLM version, so the batch layout asserted here is the one this version produces.
"""

import numpy as np
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu_input_batch import InputBatch as Mrv1InputBatch

from pllm import _core
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
    assert [int(value) for value in input_batch.cu_num_logits_np] == list(range(count + 1))
    logits = input_batch.logits_indices
    assert logits.ndim == 1 and logits.device.type == "cuda" and logits.numel() == count
    request_ids = tuple(identity.request_id for identity in identities)
    assert tuple(input_batch.req_ids) == request_ids and all(request_ids) and len(set(request_ids)) == count
    scheduled = [int(value) for value in input_batch.num_scheduled_tokens]
    prompt = [int(value) for value in prompt_lengths]
    computed = [int(value) for value in input_batch.num_computed_tokens_np]
    assert all(
        tokens >= 1 and bool(prefilling) == (done < length)
        for tokens, length, done, prefilling in zip(
            scheduled, input_batch.prefill_len_np, computed, input_batch.is_prefilling_np, strict=True
        )
    )
    return _planned_rows(
        identities, scheduled, prompt, computed, input_batch.logits_indices, input_batch.num_tokens, step_id
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
    assert min(scheduled) >= 1
    return _planned_rows(
        identities,
        scheduled,
        [int(value) for value in batch.num_prompt_tokens[: batch.num_reqs]],
        [int(value) for value in batch.num_computed_tokens_cpu[: batch.num_reqs]],
        logits_indices,
        scheduler.total_num_scheduled_tokens,
        step_id,
    )


def _planned_rows(
    identities: tuple[RequestIdentity, ...],
    scheduled: list[int],
    prompt: list[int],
    computed: list[int],
    logits_indices: torch.Tensor,
    total: int,
    step_id: int,
) -> PackedSamplerRowBatch:
    plan = _core.plan_packed_prefill_rows(scheduled, prompt, computed, total)
    rows: list[PackedPhaseRow] = []
    for request, packed, offset, raw_phase in zip(
        plan.request_ordinals, plan.packed_row_indices, plan.token_offsets, plan.phases, strict=True
    ):
        phase = HiddenPhase(raw_phase)
        rows.append(PackedPhaseRow(packed, request, offset, phase, identities[request], plan.prompt_lengths[packed]))
    phase_rows = PackedPhaseRowBatch(step_id, plan.total_rows, tuple(rows))
    # vLLM samples each request's last scheduled token, which is the last packed row of its span.
    links = tuple(PackedSamplerRowLink(span.request_ordinal, span.packed_row_end - 1) for span in plan.request_spans)
    return PackedSamplerRowBatch(phase_rows, links)
