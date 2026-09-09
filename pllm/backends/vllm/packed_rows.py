"""Concrete MRv2 ABI checks followed by the original Rust row plan."""

import torch

from pllm import _core
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import (
    PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink,
)
from pllm.contracts.request_identity import RequestIdentity
from vllm import __version__ as _VLLM_VERSION
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu_input_batch import InputBatch as Mrv1InputBatch
from vllm.v1.core.sched.output import SchedulerOutput


def encode_mrv2_rows(input_batch: InputBatch, identities: tuple[RequestIdentity, ...],
                     *, plan_epoch: int, step_id: int) -> PackedSamplerRowBatch:
    if _VLLM_VERSION != "0.27.1":
        raise RuntimeError("canonical MRv2 packed rows require vLLM 0.27.1 exactly")
    if type(input_batch) is not InputBatch:
        raise TypeError("MRv2 packed rows require concrete signed inputs")
    if torch.compiler.is_compiling() or (
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError("canonical MRv2 packed rows are eager-only")
    count = input_batch.num_reqs
    if (count < 1 or len(input_batch.req_ids) != count
        or input_batch.num_reqs_after_padding != count
        or input_batch.num_tokens_after_padding != input_batch.num_tokens
        or input_batch.num_draft_tokens != 0 or input_batch.num_draft_tokens_per_req is not None
        or any(len(column) != count for column in (input_batch.num_scheduled_tokens,
            input_batch.prefill_len_np, input_batch.num_computed_tokens_np, input_batch.is_prefilling_np))
        or len(input_batch.cu_num_logits_np) != count + 1
        or input_batch.logits_indices.ndim != 1 or input_batch.logits_indices.device.type != "cuda"
        or input_batch.logits_indices.numel() != count):
        raise RuntimeError("vLLM MRv2 InputBatch is not an eager n=1 non-speculative batch")
    request_ids = tuple(identity.request_id for identity in identities)
    if (tuple(input_batch.req_ids) != request_ids or any(not value for value in request_ids)
        or len(set(request_ids)) != count):
        raise RuntimeError("canonical request identities differ from the active MRv2 InputBatch")
    if any(int(value) != index for index, value in enumerate(input_batch.cu_num_logits_np)):
        raise RuntimeError("canonical MRv2 adapter requires exactly one sampler row per request")
    scheduled = [int(value) for value in input_batch.num_scheduled_tokens]
    prompt = [int(value) for value in input_batch.prefill_len_np]
    computed = [int(value) for value in input_batch.num_computed_tokens_np]
    if any(tokens < 1 or bool(prefilling) != (done < length)
        for tokens, length, done, prefilling in zip(scheduled, prompt, computed,
            input_batch.is_prefilling_np, strict=True)):
        raise RuntimeError("vLLM MRv2 InputBatch request columns do not match the signed ABI")
    return _planned_rows(identities, scheduled, prompt, computed, input_batch.logits_indices,
        input_batch.num_tokens, plan_epoch, step_id)


def encode_mrv1_rows(batch: Mrv1InputBatch, scheduler: SchedulerOutput, logits_indices: torch.Tensor,
                     identities: tuple[RequestIdentity, ...], *, plan_epoch: int, step_id: int) -> PackedSamplerRowBatch:
    if (type(batch) is not Mrv1InputBatch or _VLLM_VERSION != "0.27.1"
        or batch.num_reqs < 1 or tuple(batch.req_ids) != tuple(identity.request_id for identity in identities)
        or len(set(batch.req_ids)) != batch.num_reqs or len(identities) != batch.num_reqs
        or scheduler.scheduled_spec_decode_tokens or logits_indices.shape != (batch.num_reqs,)
        or logits_indices.device.type != "cuda"):
        raise ValueError("MRv1 rows require the exact non-speculative persistent input batch")
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MRv1 workload packed rows are eager-only")
    scheduled = [scheduler.num_scheduled_tokens[key] for key in batch.req_ids]
    if any(count < 1 for count in scheduled):
        raise ValueError("MRv1 input batch contains an unscheduled request")
    return _planned_rows(identities, scheduled, [int(value) for value in batch.num_prompt_tokens[:batch.num_reqs]],
        [int(value) for value in batch.num_computed_tokens_cpu[:batch.num_reqs]], logits_indices,
        scheduler.total_num_scheduled_tokens, plan_epoch, step_id)


def _planned_rows(identities: tuple[RequestIdentity, ...], scheduled: list[int], prompt: list[int],
                  computed: list[int], logits_indices: torch.Tensor, total: int,
                  plan_epoch: int, step_id: int) -> PackedSamplerRowBatch:
    plan = _core.plan_packed_prefill_rows(scheduled, prompt, computed, total)
    rows: list[PackedPhaseRow] = []
    for request, packed, offset, raw_phase in zip(plan.request_ordinals, plan.packed_row_indices,
        plan.token_offsets, plan.phases, strict=True):
        phase = HiddenPhase(raw_phase)
        rows.append(PackedPhaseRow(packed, request, offset, phase, identities[request]))
    phase_rows = PackedPhaseRowBatch(plan_epoch, step_id, plan.total_rows, tuple(rows))
    cpu_indices = logits_indices.to(device="cpu", dtype=torch.int64)
    links: list[PackedSamplerRowLink] = []
    for span in plan.request_spans:
        index = span.packed_row_end - 1
        if int(cpu_indices[span.request_ordinal]) != index:
            raise RuntimeError("backend logits row does not select its request's final packed row")
        links.append(PackedSamplerRowLink(span.request_ordinal, index))
    return PackedSamplerRowBatch(phase_rows, tuple(links))
