"""Closed vLLM 0.27.1 MRv2 packed-row boundary.

This module is deliberately an in-process, eager-only adapter.  It consumes
the concrete MRv2 :class:`vllm.v1.worker.gpu.input_batch.InputBatch`, asks the
Rust planner for the complete packed-row expansion, and validates every
returned column before exposing canonical rows.  There is no fallback layout
guessing and no request-id suffix parsing at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm import _core
from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.packed_rows import (
    PackedPhaseRow,
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
    PackedSamplerRowBatch,
    PackedSamplerRowLink,
)
from tllm.contracts.request_identity import RequestIdentity

from vllm import __version__ as _VLLM_VERSION
from vllm.v1.worker.gpu.input_batch import InputBatch


_SUPPORTED_VLLM_VERSION = "0.27.1"


@dataclass(frozen=True, slots=True)
class Mrv2RequestIdBinding:
    """One exact active MRv2 request id and its canonical n=1 identity."""

    raw_request_id: str
    identity: RequestIdentity

    def __post_init__(self) -> None:
        if (
            type(self.raw_request_id) is not str
            or not self.raw_request_id
            or type(self.identity) is not RequestIdentity
            or self.identity.sample_idx != 0
        ):
            raise ValueError(
                "MRv2 packed rows require an exact n=1 request-id binding"
            )


@dataclass(frozen=True, slots=True)
class Mrv2RequestIdBatch:
    """Request-id keys in exactly the active InputBatch request order."""

    bindings: tuple[Mrv2RequestIdBinding, ...]

    def __post_init__(self) -> None:
        if (
            type(self.bindings) is not tuple
            or not self.bindings
            or any(type(binding) is not Mrv2RequestIdBinding for binding in self.bindings)
        ):
            raise ValueError(
                "MRv2 request-id batch must contain unique exact bindings"
            )
        request_ids = tuple(binding.raw_request_id for binding in self.bindings)
        canonical_request_ids = tuple(
            binding.identity.request_id for binding in self.bindings
        )
        if (
            len(request_ids) != len(set(request_ids))
            or len(canonical_request_ids) != len(set(canonical_request_ids))
        ):
            raise ValueError(
                "MRv2 request-id batch must preserve one n=1 identity per request"
            )

    @property
    def raw_request_ids(self) -> tuple[str, ...]:
        return tuple(binding.raw_request_id for binding in self.bindings)


@dataclass(frozen=True, slots=True)
class Mrv2PackedStepRows:
    """One exact canonical row batch and its one-row-per-request sampler map."""

    phase_rows: PackedPhaseRowBatch
    sampler_rows: PackedSamplerRowBatch

    def __post_init__(self) -> None:
        if (
            type(self.phase_rows) is not PackedPhaseRowBatch
            or type(self.sampler_rows) is not PackedSamplerRowBatch
            or self.sampler_rows.row_batch is not self.phase_rows
        ):
            raise ValueError(
                "MRv2 packed step rows must retain one exact row provenance"
            )


class Vllm027Mrv2PackedRowCodec:
    """Strict vLLM 0.27.1 MRv2 input-batch to canonical row codec.

    The codec deliberately has no transport mode or speculative variant: an
    attached caller constructs it inside the MRv2 worker process and calls it
    only around an ordinary eager model step.  The method rejects CUDA-graph
    capture, padded batches, speculation, and any sampler shape other than
    one logit row for each active request.
    """

    __slots__ = ("_plan_epoch",)

    def __init__(self, *, plan_epoch: int) -> None:
        if type(plan_epoch) is not int or plan_epoch < 1:
            raise ValueError("MRv2 packed-row plan epoch must be positive")
        if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
            raise RuntimeError(
                "canonical MRv2 packed rows require vLLM 0.27.1 exactly"
            )
        self._plan_epoch = plan_epoch

    @property
    def plan_epoch(self) -> int:
        return self._plan_epoch

    def encode(
        self,
        input_batch: InputBatch,
        request_ids: Mrv2RequestIdBatch,
        *,
        step_id: int,
    ) -> Mrv2PackedStepRows:
        """Expand and validate one concrete eager MRv2 InputBatch."""

        self._require_eager_input(input_batch, request_ids, step_id)
        active_request_count = input_batch.num_reqs
        scheduled_tokens: list[int] = []
        prompt_tokens: list[int] = []
        computed_tokens: list[int] = []
        for request_ordinal in range(active_request_count):
            scheduled = int(input_batch.num_scheduled_tokens[request_ordinal])
            prompt = int(input_batch.prefill_len_np[request_ordinal])
            computed = int(input_batch.num_computed_tokens_np[request_ordinal])
            is_prefilling = bool(input_batch.is_prefilling_np[request_ordinal])
            if (
                scheduled < 1
                or prompt < 0
                or computed < 0
                or is_prefilling != (computed < prompt)
            ):
                raise RuntimeError(
                    "vLLM MRv2 InputBatch request columns do not match the signed ABI"
                )
            scheduled_tokens.append(scheduled)
            prompt_tokens.append(prompt)
            computed_tokens.append(computed)

        if sum(scheduled_tokens) != input_batch.num_tokens:
            raise RuntimeError(
                "vLLM MRv2 InputBatch scheduled tokens do not match num_tokens"
            )
        columns = PackedRowPlanInputs(
            request_ids=request_ids.raw_request_ids,
            scheduled_tokens=tuple(scheduled_tokens),
            prompt_tokens=tuple(prompt_tokens),
            computed_tokens=tuple(computed_tokens),
        )
        row_plan = _core.plan_packed_prefill_rows(
            list(columns.scheduled_tokens),
            list(columns.prompt_tokens),
            list(columns.computed_tokens),
            input_batch.num_tokens,
        )
        total_rows = int(row_plan.total_rows)
        if total_rows != input_batch.num_tokens:
            raise RuntimeError("Rust packed-row plan total differs from MRv2 input")
        if (
            len(row_plan.request_ordinals) != total_rows
            or len(row_plan.packed_row_indices) != total_rows
            or len(row_plan.token_offsets) != total_rows
            or len(row_plan.phases) != total_rows
        ):
            raise RuntimeError("Rust packed-row plan has an incomplete row column")

        rows: list[PackedPhaseRow] = []
        request_row_counts = [0 for _ in range(active_request_count)]
        request_row_ends = [0 for _ in range(active_request_count)]
        decode_plan_ordinal = 0
        for row_index, (
            raw_request_ordinal,
            raw_packed_row_index,
            raw_token_offset,
            raw_phase,
        ) in enumerate(
            zip(
                row_plan.request_ordinals,
                row_plan.packed_row_indices,
                row_plan.token_offsets,
                row_plan.phases,
                strict=True,
            )
        ):
            request_ordinal = int(raw_request_ordinal)
            packed_row_index = int(raw_packed_row_index)
            token_offset = int(raw_token_offset)
            if (
                request_ordinal < 0
                or request_ordinal >= active_request_count
                or packed_row_index != row_index
                or request_row_counts[request_ordinal]
                >= scheduled_tokens[request_ordinal]
            ):
                raise RuntimeError("Rust packed-row plan changed its signed row order")
            expected_token_offset = (
                computed_tokens[request_ordinal]
                + request_row_counts[request_ordinal]
            )
            expected_phase = (
                HiddenPhase.PREFILL
                if expected_token_offset < prompt_tokens[request_ordinal]
                else HiddenPhase.DECODE
            )
            phase = HiddenPhase(raw_phase)
            if token_offset != expected_token_offset or phase is not expected_phase:
                raise RuntimeError("Rust packed-row plan changed a signed phase column")
            request_row_counts[request_ordinal] += 1
            request_row_ends[request_ordinal] = packed_row_index + 1
            decode_ordinal: int | None = None
            if phase is HiddenPhase.DECODE:
                decode_ordinal = decode_plan_ordinal
                decode_plan_ordinal += 1
            binding = request_ids.bindings[request_ordinal]
            rows.append(
                PackedPhaseRow(
                    packed_row_index=packed_row_index,
                    request_ordinal=request_ordinal,
                    raw_request_id=binding.raw_request_id,
                    token_offset=token_offset,
                    phase=phase,
                    identity=binding.identity,
                    decode_plan_ordinal=decode_ordinal,
                )
            )
        if tuple(request_row_counts) != tuple(scheduled_tokens):
            raise RuntimeError("Rust packed-row plan omitted a scheduled request row")

        phase_rows = PackedPhaseRowBatch(
            plan_epoch=self._plan_epoch,
            step_id=step_id,
            total_rows=total_rows,
            rows=tuple(rows),
        )
        sampler_rows = self._sampler_rows(
            input_batch,
            phase_rows,
            request_row_ends,
        )
        return Mrv2PackedStepRows(phase_rows, sampler_rows)

    @staticmethod
    def _require_eager_input(
        input_batch: InputBatch,
        request_ids: Mrv2RequestIdBatch,
        step_id: int,
    ) -> None:
        if (
            type(input_batch) is not InputBatch
            or type(request_ids) is not Mrv2RequestIdBatch
            or type(step_id) is not int
            or step_id < 0
        ):
            raise TypeError("MRv2 packed rows require concrete signed inputs")
        if torch.compiler.is_compiling() or (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("canonical MRv2 packed rows are eager-only")
        if (
            input_batch.num_reqs < 1
            or len(input_batch.req_ids) != input_batch.num_reqs
            or input_batch.num_reqs_after_padding != input_batch.num_reqs
            or input_batch.num_tokens_after_padding != input_batch.num_tokens
            or input_batch.num_draft_tokens != 0
            or input_batch.num_draft_tokens_per_req is not None
            or len(input_batch.num_scheduled_tokens) != input_batch.num_reqs
            or len(input_batch.prefill_len_np) != input_batch.num_reqs
            or len(input_batch.num_computed_tokens_np) != input_batch.num_reqs
            or len(input_batch.is_prefilling_np) != input_batch.num_reqs
            or len(input_batch.cu_num_logits_np) != input_batch.num_reqs + 1
            or input_batch.logits_indices.ndim != 1
            or input_batch.logits_indices.device.type != "cuda"
        ):
            raise RuntimeError(
                "vLLM MRv2 InputBatch is not an eager n=1 non-speculative batch"
            )
        raw_request_ids = tuple(input_batch.req_ids)
        if (
            raw_request_ids != request_ids.raw_request_ids
            or any(not request_id for request_id in raw_request_ids)
            or len(set(raw_request_ids)) != len(raw_request_ids)
        ):
            raise RuntimeError(
                "canonical request-id keys differ from the active MRv2 InputBatch"
            )
        if int(input_batch.cu_num_logits_np[0]) != 0:
            raise RuntimeError("MRv2 sampler cumulative rows must start at zero")
        for request_ordinal in range(input_batch.num_reqs):
            if (
                int(input_batch.cu_num_logits_np[request_ordinal + 1])
                != int(input_batch.cu_num_logits_np[request_ordinal]) + 1
            ):
                raise RuntimeError(
                    "canonical MRv2 adapter requires exactly one sampler row per request"
                )
        if int(input_batch.logits_indices.numel()) != input_batch.num_reqs:
            raise RuntimeError("MRv2 logits indices differ from one-row sampler ABI")

    @staticmethod
    def _sampler_rows(
        input_batch: InputBatch,
        phase_rows: PackedPhaseRowBatch,
        request_row_ends: list[int],
    ) -> PackedSamplerRowBatch:
        cpu_indices = input_batch.logits_indices.to(
            device=torch.device("cpu"),
            dtype=torch.int64,
        )
        links: list[PackedSamplerRowLink] = []
        for sampler_row_index in range(input_batch.num_reqs):
            packed_row_index = int(cpu_indices[sampler_row_index])
            expected_packed_row_index = request_row_ends[sampler_row_index] - 1
            if packed_row_index != expected_packed_row_index:
                raise RuntimeError(
                    "MRv2 logits row does not select its request's final packed row"
                )
            links.append(
                PackedSamplerRowLink(
                    sampler_row_index=sampler_row_index,
                    packed_row_index=packed_row_index,
                )
            )
        return PackedSamplerRowBatch(phase_rows, tuple(links))


__all__ = [
    "Mrv2PackedStepRows",
    "Mrv2RequestIdBatch",
    "Mrv2RequestIdBinding",
    "Vllm027Mrv2PackedRowCodec",
]
