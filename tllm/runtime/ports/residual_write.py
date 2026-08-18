#!/usr/bin/env python3
"""Staged residual-stream write channel (sync-free, capture-safe).

Consumers declare residual writes in their flows (``ResidualStream.write``);
the runtime owns all mechanics. A writer stages its delta into an
address-stable :class:`ResidualWriteLease` before the forward pass; the
tapped layer's forward wrapper applies pending leases in-place
(``index_copy_`` / ``index_add_``) after the original forward and before
the capture ``index_select``, so both downstream layers and the tap
observe the written stream.

The same staged path serves eager and CUDA-graph modes — buffer addresses
never change across steps, so the apply sequence is replayable as-is.
Consumers never select an implementation; ``TLLM_DEBUG_GRAPH_MODE`` (see
``tllm.runtime.graph_policy``) exists only for tLLM developers validating
the graph path.

Semantics: staging is one-shot. A lease applies on the next tapped forward
after ``stage()`` and then clears itself; rows staged but never applied
(e.g. a skipped step) are dropped on the following ``stage()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import torch

from tllm.ports.base import WriteMode, WriteRows
from tllm.ports.residual_stream import ResidualLocator


@dataclass(frozen=True)
class WriteLeaseKey:
    layer_path: str
    writer_id: str
    order: int


class ResidualWriteLease:
    """Address-stable staging buffers for one writer at one tapped layer."""

    def __init__(
        self,
        *,
        key: WriteLeaseKey,
        mode: WriteMode,
        rows: WriteRows,
        max_rows: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if mode not in (WriteMode.REPLACE, WriteMode.ADD):
            raise ValueError(f"write leases support replace/add modes, got `{mode}`")
        if int(max_rows) <= 0 or int(hidden_size) <= 0:
            raise ValueError("write lease requires max_rows > 0 and hidden_size > 0")
        self.key = key
        self.mode = mode
        self.rows = rows
        self.delta = torch.zeros((int(max_rows), int(hidden_size)), device=device, dtype=dtype)
        self.row_idx = torch.zeros((int(max_rows),), device=device, dtype=torch.long)
        self.active_mask = torch.zeros((int(max_rows), 1), device=device, dtype=dtype)
        self.count = 0

    @property
    def max_rows(self) -> int:
        return int(self.delta.shape[0])

    def stage(self, *, delta: torch.Tensor, row_idx: torch.Tensor, validate_rows: bool = False) -> None:
        """Copy a pending write into the persistent buffers (sync-free).

        ``validate_rows=True`` enables the deterministic-row debug check
        (device sync; tests and debugging only — never on the hot path).
        """
        n = int(row_idx.numel())
        if n != int(delta.shape[0]):
            raise ValueError(
                f"staged write row mismatch: delta_rows={int(delta.shape[0])} row_idx={n}"
            )
        if n > self.max_rows:
            raise ValueError(
                f"staged write exceeds lease capacity: rows={n} capacity={self.max_rows}"
            )
        if int(delta.shape[-1]) != int(self.delta.shape[1]):
            raise ValueError(
                f"staged write hidden width mismatch: delta={int(delta.shape[-1])} "
                f"lease={int(self.delta.shape[1])}"
            )
        if validate_rows and n > 0:
            unique = int(torch.unique(row_idx).numel())
            if unique != n:
                raise ValueError(
                    f"staged write rows must be unique for deterministic apply: rows={n} unique={unique}"
                )
        self.active_mask.zero_()
        if n > 0:
            self.delta[:n].copy_(delta, non_blocking=True)
            self.row_idx[:n].copy_(row_idx.to(dtype=torch.long), non_blocking=True)
            self.active_mask[:n].fill_(1)
        self.count = n

    def clear(self) -> None:
        self.active_mask.zero_()
        self.count = 0


def leases_for_path(
    leases: Mapping[WriteLeaseKey, ResidualWriteLease],
    layer_path: str,
) -> List[ResidualWriteLease]:
    """Leases staged for one tapped layer, in deterministic apply order."""
    selected = [lease for key, lease in leases.items() if key.layer_path == layer_path]
    selected.sort(key=lambda lease: lease.key.order)
    return selected


def apply_pending_writes(
    leases: Mapping[WriteLeaseKey, ResidualWriteLease],
    layer_path: str,
    tensor: torch.Tensor,
) -> None:
    """Apply staged writes to the live layer output, then clear them.

    Runs inside the tapped forward wrapper after the original forward and
    before residual capture, on persistent buffers only — capture-safe
    under piecewise/full CUDA graphs and identical under eager.
    """
    for lease in leases_for_path(leases, layer_path):
        # CUDA graph replay only replays Tensor operations captured from this
        # block; it never re-evaluates a Python row count. Always launch the
        # fixed-capacity operation and gate active rows with device state.
        rows = lease.row_idx
        if lease.mode == WriteMode.REPLACE:
            current = tensor.index_select(0, rows)
            updates = (lease.delta - current) * lease.active_mask
            tensor.index_add_(0, rows, updates)
        else:
            tensor.index_add_(0, rows, lease.delta * lease.active_mask)
        lease.clear()


def build_write_leases(
    *,
    writes: Iterable[tuple[str, ResidualLocator, WriteMode, WriteRows, int]],
    resolve_path,
    device: torch.device,
    max_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> Dict[WriteLeaseKey, ResidualWriteLease]:
    """Create one lease per (layer path, writer, order) declared write.

    ``writes`` yields ``(writer_id, locator, mode, rows, order)``;
    ``resolve_path`` maps a locator to its resolved layer path.
    """
    leases: Dict[WriteLeaseKey, ResidualWriteLease] = {}
    for writer_id, locator, mode, rows, order in writes:
        layer_path = str(resolve_path(locator) or "").strip()
        if not layer_path:
            raise RuntimeError(f"residual write locator was not resolved to a tapped layer: {locator}")
        key = WriteLeaseKey(layer_path=layer_path, writer_id=str(writer_id), order=int(order))
        if key in leases:
            raise ValueError(f"duplicate residual write lease for key {key}")
        leases[key] = ResidualWriteLease(
            key=key,
            mode=mode,
            rows=rows,
            max_rows=int(max_rows),
            hidden_size=int(hidden_size),
            device=device,
            dtype=dtype,
        )
    return leases
