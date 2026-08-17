#!/usr/bin/env python3
"""GPU resource buffers for the sampler precompute pipeline.

These classes own the reusable scratch buffers, per-step caches, CUDA
streams/events, and diagnostic counters used to move distiller inputs/outputs
between the main stream and the adaptation side streams. They are pure
resource holders: no hook, dispatch, or consumer logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplerPrecomputeBuffers:
    pred_hidden: torch.Tensor
    valid_mask: torch.Tensor
    source_hidden: torch.Tensor
    prompt_idx: torch.Tensor
    all_row_ids: torch.Tensor
    dense_logits: torch.Tensor | None = None

    def covers(self, *, rows: int, hidden: int, device: torch.device, dtype: torch.dtype) -> bool:
        return (
            tuple(self.pred_hidden.shape) == (rows, hidden)
            and self.pred_hidden.device == device
            and self.pred_hidden.dtype == dtype
            and int(self.valid_mask.numel()) == rows
            and self.valid_mask.device == device
            and tuple(self.source_hidden.shape) == (rows, hidden)
            and self.source_hidden.device == device
            and self.source_hidden.dtype == dtype
            and int(self.prompt_idx.numel()) == rows
            and self.prompt_idx.device == device
            and int(self.all_row_ids.numel()) == rows
            and self.all_row_ids.device == device
        )

    def dense_covers(self, *, rows: int, vocab: int, device: torch.device) -> bool:
        return (
            self.dense_logits is not None
            and tuple(self.dense_logits.shape) == (rows, vocab)
            and self.dense_logits.device == device
            and self.dense_logits.dtype == torch.float32
        )


@dataclass
class SamplerPrecomputeCache:
    step_id: int
    row_ids: torch.Tensor
    pred_hidden: torch.Tensor
    dense_logits: torch.Tensor | None = None
    pred_hidden_row_map: torch.Tensor | None = None
    all_rows: bool = False


@dataclass
class SamplerPrecomputeState:
    stream: torch.cuda.Stream | None = None
    event: torch.cuda.Event | None = None
    buffers: SamplerPrecomputeBuffers | None = None
    cache: SamplerPrecomputeCache | None = None
    source_enabled: bool = False
    source_capture_step_id: int = -1
    port_enabled: bool = True
    port_capture_step_id: int = -1
    port_publish_step_id: int = -1
    port_consume_step_id: int = -1
    precomputed_step_id: int = -1
    all_rows: bool = False
    timing_enabled: bool = False
    precompute_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    wait_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    fallback_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    port_publish_attempt_count: int = 0
    port_publish_hit_count: int = 0
    schedule_attempt_count: int = 0
    schedule_hit_count: int = 0
    candidate_sample_count: int = 0
    candidate_token_count: int = 0
    candidate_row_count: int = 0
    candidate_max_count: int = 0
    candidate_kernel_triton_count: int = 0
    candidate_kernel_torch_count: int = 0
    candidate_kernel_fallback_count: int = 0

    def reset_step(self) -> None:
        self.source_enabled = False
        self.source_capture_step_id = -1
        self.port_capture_step_id = -1
        self.port_publish_step_id = -1
        self.port_consume_step_id = -1
        self.precomputed_step_id = -1
        self.cache = None
        self.all_rows = False

    def reset_decode_step(self, step_id: int) -> None:
        self.source_enabled = False
        self.source_capture_step_id = -1
        self.port_capture_step_id = -1
        self.port_publish_step_id = -1
        self.port_consume_step_id = -1
        self.precomputed_step_id = int(step_id)
        self.cache = None
        self.all_rows = False
        if self.buffers is not None:
            self.buffers.valid_mask.zero_()

    def ensure_buffers(
        self,
        *,
        source_hidden: torch.Tensor,
        dense_vocab: int | None = None,
    ) -> SamplerPrecomputeBuffers:
        rows = int(source_hidden.shape[0])
        hidden = int(source_hidden.shape[1])
        device = source_hidden.device
        dtype = source_hidden.dtype
        if self.buffers is None or not self.buffers.covers(rows=rows, hidden=hidden, device=device, dtype=dtype):
            self.buffers = SamplerPrecomputeBuffers(
                pred_hidden=torch.empty((rows, hidden), device=device, dtype=dtype),
                valid_mask=torch.zeros((rows,), device=device, dtype=torch.bool),
                source_hidden=torch.empty((rows, hidden), device=device, dtype=dtype),
                prompt_idx=torch.empty((rows,), device=device, dtype=torch.long),
                all_row_ids=torch.arange(rows, device=device, dtype=torch.long),
            )
        if dense_vocab is not None and not self.buffers.dense_covers(rows=rows, vocab=int(dense_vocab), device=device):
            self.buffers.dense_logits = torch.empty((rows, int(dense_vocab)), device=device, dtype=torch.float32)
        return self.buffers

    def cache_for_step(self, step_id: int) -> SamplerPrecomputeCache | None:
        if self.cache is None or int(self.cache.step_id) != int(step_id):
            return None
        return self.cache

    def store_cache(
        self,
        *,
        step_id: int,
        row_ids: torch.Tensor,
        pred_hidden: torch.Tensor,
        dense_logits: torch.Tensor | None = None,
        pred_hidden_row_map: torch.Tensor | None = None,
        all_rows: bool = False,
    ) -> SamplerPrecomputeCache:
        self.precomputed_step_id = int(step_id)
        self.cache = SamplerPrecomputeCache(
            step_id=int(step_id),
            row_ids=row_ids,
            pred_hidden=pred_hidden,
            dense_logits=dense_logits,
            pred_hidden_row_map=pred_hidden_row_map,
            all_rows=bool(all_rows),
        )
        self.all_rows = bool(all_rows)
        return self.cache

    def captured_rows_for_step(
        self,
        *,
        step_id: int,
        decode_count: int,
    ) -> SamplerPrecomputeCache | None:
        buffers = self.buffers
        count = int(decode_count)
        if (
            buffers is None
            or count <= 0
            or int(self.source_capture_step_id) != int(step_id)
            or int(buffers.pred_hidden.shape[0]) < count
            or int(buffers.valid_mask.numel()) < count
        ):
            return None
        dense_logits = buffers.dense_logits[:count] if buffers.dense_logits is not None and int(buffers.dense_logits.shape[0]) >= count else None
        if self.all_rows:
            row_ids = buffers.all_row_ids[:count]
            return SamplerPrecomputeCache(
                step_id=int(step_id),
                row_ids=row_ids,
                pred_hidden=buffers.pred_hidden[:count],
                dense_logits=dense_logits,
                all_rows=True,
            )
        row_ids = buffers.valid_mask[:count].nonzero(as_tuple=False).view(-1)
        if int(row_ids.numel()) <= 0:
            return None
        selected_dense = dense_logits.index_select(0, row_ids) if dense_logits is not None else None
        return SamplerPrecomputeCache(
            step_id=int(step_id),
            row_ids=row_ids,
            pred_hidden=buffers.pred_hidden[:count].index_select(0, row_ids),
            dense_logits=selected_dense,
            all_rows=False,
        )
