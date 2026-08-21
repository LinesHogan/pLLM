#!/usr/bin/env python3
"""Decode row localization: select decode rows and export capture-layer hidden."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

import torch

from tllm.common.state import (
    STATE,
    find_capture_layer,
    resolve_prompt_sample_for_req_id,
)

DECODE_ROW_IDX_BUFFER = "decode_row_idx"
DECODE_VALID_MASK_BUFFER = "decode_valid_mask"


def compute_decode_localization(
    req_ids: List[str],
    is_decode_req: List[bool],
    logits_indices: torch.Tensor | None,
    num_actual_tokens: int,
    resolve_prompt_sample_fn: Callable[[str], Tuple[int, int]],
    max_decode_rows: int = 0,
) -> Tuple[torch.Tensor, List[int], List[int], List[int]]:
    """Pure decode-localization function (no model/runner dependency).

    Position selection (host-known integer math) is delegated to the Rust
    core; the row gather and bounds diagnostics stay on the tensor side so
    the CUDA hot path never gains a host sync.

    Returns:
    - row_idx: packed hidden row index for each decode request in current step.
    - prompt_idxs: prompt index per localized row.
    - sample_idxs: sample index per localized row (n>1 aware).
    - decode_positions: request positions selected from current req order.
    """
    if num_actual_tokens <= 0 or logits_indices is None or logits_indices.numel() == 0:
        return torch.empty((0,), dtype=torch.long), [], [], []

    from tllm import core as _core

    located = _core.localize_decode_positions(
        [bool(x) for x in is_decode_req],
        int(logits_indices.numel()),
        len(req_ids),
        int(max_decode_rows),
    )
    decode_positions = [int(p) for p in located.positions]
    if not decode_positions:
        return torch.empty((0,), device=logits_indices.device, dtype=torch.long), [], [], []

    if located.contiguous:
        row_idx = logits_indices.narrow(0, int(decode_positions[0]), len(decode_positions))
    else:
        decode_pos = torch.tensor(decode_positions, device=logits_indices.device, dtype=torch.long)
        row_idx = logits_indices.index_select(0, decode_pos)

    # Keep the explicit bounds diagnostic on CPU, but avoid forcing a host
    # sync on CUDA decode hot paths.
    if row_idx.device.type != "cuda":
        min_idx = int(row_idx.min().item())
        max_idx = int(row_idx.max().item())
        if min_idx < 0 or max_idx >= int(num_actual_tokens):
            raise RuntimeError(
                f"logits_indices out of range in prepare_inputs: min={min_idx} "
                f"max={max_idx} num_actual_tokens={int(num_actual_tokens)}"
            )

    prompt_idxs: List[int] = []
    sample_idxs: List[int] = []
    for req_pos in decode_positions:
        req_id = req_ids[req_pos]
        prompt_idx, sample_idx = resolve_prompt_sample_fn(req_id)
        prompt_idxs.append(int(prompt_idx))
        sample_idxs.append(int(sample_idx))

    return row_idx, prompt_idxs, sample_idxs, decode_positions


def prepare_decode_localization(runner: Any) -> None:
    """Compute decode row indices from vLLM packed layout.

    Why this is needed:
    - vLLM packs tokens for all active requests into one dense hidden tensor.
    - `logits_indices[i]` points to the row used for request `i` sampling.
    - only requests in decode phase should be consumed by decode producer.

    Example:
    - req_ids        = [A, B, C, D]
    - is_decode_req  = [F, T, T, F]
    - logits_indices = [4, 7, 11, 15]
    -> decode_positions = [1, 2]
    -> decode rows      = [7, 11]

    The rows above are written into fixed GPU buffers so capture-layer forward can
    run a graph-safe gather/mask sequence.
    """
    step = STATE.step

    layer, _ = find_capture_layer(runner.model)
    decode_row_idx = getattr(layer, DECODE_ROW_IDX_BUFFER)
    decode_valid_mask = getattr(layer, DECODE_VALID_MASK_BUFFER)

    step.decode_count = 0
    step.decode_prompt_idxs = []
    step.decode_sample_idxs = []

    with torch.no_grad():
        decode_row_idx.zero_()
        decode_valid_mask.zero_()

    row_idx, prompt_idxs, sample_idxs, _decode_positions = compute_decode_localization(
        req_ids=step.req_ids,
        is_decode_req=step.is_decode_req,
        logits_indices=step.logits_indices,
        num_actual_tokens=step.num_actual_tokens,
        resolve_prompt_sample_fn=resolve_prompt_sample_for_req_id,
    )
    if row_idx.numel() == 0:
        return

    if row_idx.device != decode_row_idx.device:
        row_idx = row_idx.to(decode_row_idx.device)

    k = int(row_idx.numel())
    if k > int(decode_row_idx.numel()):
        raise RuntimeError(
            f"decode rows exceed graph scratch rows: decode_rows={k} "
            f"capacity={int(decode_row_idx.numel())}; increase graph_scratch_rows"
        )

    with torch.no_grad():
        decode_row_idx[:k].copy_(row_idx)
        decode_valid_mask[:k].fill_(1.0)

    step.decode_count = k
    step.decode_prompt_idxs = [int(x) for x in prompt_idxs]
    step.decode_sample_idxs = [int(x) for x in sample_idxs]
