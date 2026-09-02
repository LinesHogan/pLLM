#!/usr/bin/env python3
"""Decode-row localization for the MRv2 persistent InputBatch.

MRv2 keeps a persistent ``InputBatch`` (``vllm.v1.worker.gpu.input_batch``)
that is rebuilt by ``GPUModelRunner.prepare_inputs`` before each forward.
For every logits row we know the owning request (``idx_mapping_np`` /
``req_ids``) and whether the request is still prefilling
(``is_prefilling_np``). Decode rows map back to packed hidden rows through
``logits_indices``.

Under speculative decoding a request contributes multiple logits rows; the
first row (the position of the first sampled token) is used for hidden
capture until the strict multi-row mapping lands with the spec-decode work.

MRv2 additionally runs *real* engine steps during kernel warmup
(``vllm.v1.worker.gpu.warmup``) with synthetic requests named
``_warmup_{i}_``; those must never enter the decode scratch (there is no
matching prompt state and the batch is padded to ``max_num_reqs``).
"""

from __future__ import annotations

import os
from typing import Any

import torch

from tllm.runtime.torch_adapter import torch_regular_position_view

from tllm.runtime.vllm_patch.port_runtime_hooks import (
    _clear_decode_localization,
    store_decode_localization,
)

_WARMUP_REQ_PREFIX = "_warmup_"


def prepare_decode_localization_mrv2(*, core: Any, runner: Any, input_batch: Any) -> tuple:
    """Fill decode scratch from the persistent InputBatch.

    Returns the strict spec-decode row mapping (empty when every decode
    request contributes exactly one logits row). The training scratch keeps
    the first-row-per-request view; the full mapping is attached to the
    StepSession by the driver for spec-decode-aware consumers.
    """
    from tllm.backends.vllm._step_session import SpecRowMapping

    runtime = core.RUNTIME
    if input_batch is None:
        _clear_decode_localization(core)
        return ()

    num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
    logits_indices = getattr(input_batch, "logits_indices", None)
    is_prefilling = getattr(input_batch, "is_prefilling_np", None)
    cu_num_logits = getattr(input_batch, "cu_num_logits_np", None)
    req_ids_raw = list(getattr(input_batch, "req_ids", []) or [])
    if (
        num_reqs <= 0
        or not isinstance(logits_indices, torch.Tensor)
        or is_prefilling is None
        or cu_num_logits is None
        or len(req_ids_raw) < num_reqs
    ):
        _clear_decode_localization(core)
        return ()

    logit_rows: list[int] = []
    prompt_idxs: list[int] = []
    sample_idxs: list[int] = []
    request_ids: list[str] = []
    spec_rows: list = []
    multi_row = False
    for batch_idx in range(num_reqs):
        if bool(is_prefilling[batch_idx]):
            continue
        start = int(cu_num_logits[batch_idx])
        end = int(cu_num_logits[batch_idx + 1])
        if end <= start:
            continue
        req_id = str(req_ids_raw[batch_idx])
        if req_id.startswith(_WARMUP_REQ_PREFIX):
            continue
        prompt_idx, sample_idx = core._resolve_prompt_sample_for_req_id(req_id)
        if end - start > 1:
            multi_row = True
            for offset, logit_row in enumerate(range(start, end)):
                spec_rows.append(
                    SpecRowMapping(
                        request_id=req_id,
                        prompt_idx=int(prompt_idx),
                        sample_idx=int(sample_idx),
                        sampled_index=offset,
                        logit_row=logit_row,
                        hidden_row=int(logits_indices[logit_row].item()),
                    )
                )
        # First logits row of the request = position of the first sampled
        # token; the training scratch keeps this first-row view.
        logit_rows.append(start)
        prompt_idxs.append(int(prompt_idx))
        sample_idxs.append(int(sample_idx))
        request_ids.append(req_id)

    if os.environ.get("TLLM_DEBUG_MRV2_LOC"):
        import sys as _sys

        print(
            f"[mrv2_loc] num_reqs={num_reqs} req_ids[:6]={req_ids_raw[:6]} "
            f"prefilling={int(sum(bool(x) for x in is_prefilling[:num_reqs]))} "
            f"cu_logits={list(cu_num_logits[: min(8, len(cu_num_logits))])}",
            file=_sys.stderr,
        )
    if not logit_rows:
        _clear_decode_localization(core)
        return ()

    row_idx = torch_regular_position_view(logits_indices, logit_rows)
    if row_idx is None:
        selector = runtime.decode_logit_row_selector_buf
        selector_staging = runtime.decode_logit_row_selector_staging
        decode_row_idx = runtime.decode_row_idx
        count = len(logit_rows)
        if selector is None or selector_staging is None or decode_row_idx is None:
            raise RuntimeError(
                "MRv2 localization requires persistent row-selector buffers"
            )
        selector_staging.copy(logit_rows, selector)
        torch.index_select(
            logits_indices,
            0,
            selector[:count],
            out=decode_row_idx[:count],
        )
        row_idx = decode_row_idx[:count]
    store_decode_localization(
        core=core,
        row_idx=row_idx,
        prompt_idxs=prompt_idxs,
        sample_idxs=sample_idxs,
        request_ids=request_ids,
    )
    return tuple(spec_rows) if multi_row else ()
