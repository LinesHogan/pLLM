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
"""

from __future__ import annotations

from typing import Any

import torch

from tllm.runtime.vllm_patch.port_runtime_hooks import (
    _clear_decode_localization,
    store_decode_localization,
)


def prepare_decode_localization_mrv2(*, core: Any, runner: Any, input_batch: Any) -> None:
    runtime = core.RUNTIME
    if input_batch is None:
        _clear_decode_localization(core)
        return

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
        return

    logit_rows: list[int] = []
    prompt_idxs: list[int] = []
    sample_idxs: list[int] = []
    request_ids: list[str] = []
    for batch_idx in range(num_reqs):
        if bool(is_prefilling[batch_idx]):
            continue
        start = int(cu_num_logits[batch_idx])
        end = int(cu_num_logits[batch_idx + 1])
        if end <= start:
            continue
        # First logits row of the request = position of the first sampled
        # token. Additional spec-decode rows are handled in the spec-decode
        # mapping work; the hidden row for training is the first one.
        logit_rows.append(start)
        req_id = str(req_ids_raw[batch_idx])
        prompt_idx, sample_idx = core._resolve_prompt_sample_for_req_id(req_id)
        prompt_idxs.append(int(prompt_idx))
        sample_idxs.append(int(sample_idx))
        request_ids.append(req_id)

    if not logit_rows:
        _clear_decode_localization(core)
        return

    row_selector = torch.as_tensor(logit_rows, dtype=torch.long, device=logits_indices.device)
    row_idx = logits_indices.index_select(0, row_selector)
    store_decode_localization(
        core=core,
        row_idx=row_idx,
        prompt_idxs=prompt_idxs,
        sample_idxs=sample_idxs,
        request_ids=request_ids,
    )
