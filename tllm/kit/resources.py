"""Training-stats and state-version accounting shared by Region workloads.

Every stateful native workload used to hand-write the same ~70 lines: loss
accumulation, ``read_and_reset_stats`` with the CUDA sync guard, the
``{None: state_version}`` runtime snapshot, and the update counters.  This
base class owns that bookkeeping once; an algorithm's resource object
subclasses it and adds only its own publication records.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from tllm.contracts.consumer_stats import TrainingStats

if TYPE_CHECKING:
    from tllm.kit.algorithm import RegionAlgorithm


class RegionResourcesBase:
    """Loss/version bookkeeping for one native Region workload.

    Subclasses add algorithm-specific publication storage (guidance,
    reweight, staged rows).  The version accounting reads
    ``workload.state_version`` so the signed audit and the stats pipe always
    observe the same counter.
    """

    def __init__(self, workload: "RegionAlgorithm") -> None:  # type: ignore[type-arg]
        self._workload = workload
        self._loss_count = 0
        self._update_count = 0
        self._loss_sum: torch.Tensor | None = None

    # -- signed state-version accounting ---------------------------------

    def state_snapshot(self) -> Mapping[str, int]:
        return MappingProxyType({"replica": self._workload.state_version})

    def runtime_state_snapshot(self, _step_id: int) -> Mapping[Hashable, int]:
        snapshot: dict[Hashable, int] = {None: self._workload.state_version}
        return MappingProxyType(snapshot)

    # -- training stats ----------------------------------------------------

    def extra_stats(self) -> Mapping[str, int]:
        """Algorithm-specific counters merged into :meth:`stats`."""
        return MappingProxyType({})

    def stats(self) -> Mapping[str, int]:
        merged = dict(self.extra_stats())
        merged["loss_count"] = self._loss_count
        merged["update_count"] = self._update_count
        return MappingProxyType(merged)

    def read_and_reset_stats(self, sync: bool = True) -> TrainingStats:
        loss_sum = self._loss_sum
        if loss_sum is None or self._loss_count <= 0:
            return TrainingStats(loss_avg=0.0, loss_count=0)
        if sync and loss_sum.device.type == "cuda":
            torch.cuda.current_stream(loss_sum.device).synchronize()
        result = TrainingStats(
            loss_avg=float(loss_sum.item()) / self._loss_count,
            loss_count=self._loss_count,
        )
        self._loss_sum = None
        self._loss_count = 0
        return result

    def record_update(self, loss: torch.Tensor) -> None:
        detached = loss.detach().float()
        self._loss_sum = (
            detached.clone() if self._loss_sum is None else self._loss_sum + detached
        )
        self._loss_count += 1
        self._update_count += 1


__all__ = ["RegionResourcesBase"]
