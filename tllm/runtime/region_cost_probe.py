"""Read-only runtime cost probe for signed ExecutionRegions (Phase D).

Measures per-region execute/capture/replay durations and keeps an in-process
EWMA per (region, phase).  The probe is **read-only by construction**: nothing
in the compiler reads it, it never mutates the signed Decision, and the
planner keeps signing ``Cost.main_path_extension_ns = 0``.  The report only
backfills the *observed* counterpart of that field for diagnostics and
explanations, so decision determinism (and every signature) is unchanged.
Feeding measurements back into ``CompileOptions`` (D3) is explicitly deferred
until the data is trustworthy.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

__all__ = [
    "RegionCostProbe",
    "RegionCostReading",
    "RegionCostReport",
]

#: Phases whose GPU work extends the main path when the region is signed onto
#: the default stream.  One-time "capture" phases are reported but excluded.
_MAIN_PATH_PHASES = frozenset({"execute", "replay"})


@dataclass(frozen=True, slots=True)
class RegionCostReading:
    """One EWMA time series for a (region, phase) pair."""

    region_id: str
    phase: str
    stream: str
    samples: int
    ewma_ns: float
    last_ns: int


@dataclass(frozen=True, slots=True)
class RegionCostReport:
    """Read-only observed-cost snapshot for one native execution epoch."""

    readings: Mapping[str, RegionCostReading]
    estimated_main_path_extension_ns: int
    signed_main_path_extension_ns: int
    explanations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PendingEvents:
    region_id: str
    phase: str
    stream: str
    start: object
    end: object


class RegionCostProbe:
    """In-process EWMA store for per-region execution durations.

    CUDA hosts record event pairs around the GPU work and harvest them
    non-blockingly (``query``); CPU hosts take a ``perf_counter_ns`` delta
    directly.  Event pairs that are still in flight stay queued until the next
    harvest (the next measurement, ``finish_step``, or an explicit read).
    """

    def __init__(self, *, alpha: float = 0.2) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("RegionCostProbe.alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._series: dict[str, RegionCostReading] = {}
        self._pending: deque[_PendingEvents] = deque()

    @staticmethod
    def _key(region_id: str, phase: str) -> str:
        return f"{region_id}/{phase}"

    @contextlib.contextmanager
    def measure(
        self, region_id: str, phase: str, stream: str
    ) -> Iterator[None]:
        """Time one GPU region execution with CUDA events when available."""
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.submit_events(region_id, phase, stream, start, end)
            return
        with self.measure_host(region_id, phase, stream):
            yield

    @contextlib.contextmanager
    def measure_host(
        self, region_id: str, phase: str, stream: str
    ) -> Iterator[None]:
        """Time one host-side span (capture, CPU execution) with perf_counter."""
        begin = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record_elapsed_ns(
                region_id, phase, stream, time.perf_counter_ns() - begin
            )

    def submit_events(
        self,
        region_id: str,
        phase: str,
        stream: str,
        start_event: object,
        end_event: object,
    ) -> None:
        """Queue one recorded event pair for harvesting (internal seam)."""
        self._pending.append(
            _PendingEvents(str(region_id), str(phase), str(stream), start_event, end_event)
        )
        self.harvest()

    def record_elapsed_ns(
        self, region_id: str, phase: str, stream: str, elapsed_ns: int
    ) -> RegionCostReading:
        """Fold one observed duration into the (region, phase) EWMA."""
        elapsed = max(0, int(elapsed_ns))
        key = self._key(region_id, phase)
        previous = self._series.get(key)
        if previous is None:
            reading = RegionCostReading(
                region_id=str(region_id),
                phase=str(phase),
                stream=str(stream),
                samples=1,
                ewma_ns=float(elapsed),
                last_ns=elapsed,
            )
        else:
            reading = RegionCostReading(
                region_id=previous.region_id,
                phase=previous.phase,
                stream=str(stream),
                samples=previous.samples + 1,
                ewma_ns=previous.ewma_ns + self._alpha * (elapsed - previous.ewma_ns),
                last_ns=elapsed,
            )
        self._series[key] = reading
        return reading

    def harvest(self, *, block: bool = False) -> int:
        """Drain completed event pairs; return how many were folded in."""
        harvested = 0
        while self._pending:
            pending = self._pending[0]
            end = pending.end
            if block:
                end.synchronize()  # type: ignore[attr-defined]
            elif not end.query():  # type: ignore[attr-defined]
                break
            start = pending.start
            elapsed_ns = int(start.elapsed_time(end) * 1e6)  # type: ignore[attr-defined]
            self.record_elapsed_ns(
                pending.region_id, pending.phase, pending.stream, elapsed_ns
            )
            self._pending.popleft()
            harvested += 1
        return harvested

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def readings(self, *, block: bool = False) -> Mapping[str, RegionCostReading]:
        self.harvest(block=block)
        return MappingProxyType(dict(self._series))

    def report(
        self, *, signed_main_path_extension_ns: int, block: bool = False
    ) -> RegionCostReport:
        """Snapshot the observed readings against the signed (placeholder) cost."""
        self.harvest(block=block)
        readings = MappingProxyType(dict(self._series))
        main_path = [
            reading
            for reading in readings.values()
            if reading.phase in _MAIN_PATH_PHASES and reading.stream == "default"
        ]
        estimate = int(round(sum(reading.ewma_ns for reading in main_path)))
        unsigned = [
            reading
            for reading in readings.values()
            if reading.phase in _MAIN_PATH_PHASES and reading.stream != "default"
        ]
        explanations = [
            "observed main-path extension estimate "
            f"{estimate} ns across {len(main_path)} default-stream region phase(s) "
            "(EWMA over sampled executions; read-only, decisions unchanged)",
            f"signed Cost.main_path_extension_ns remains {int(signed_main_path_extension_ns)} "
            "(planner placeholder); observed readings are diagnostics and never "
            "feed CompileOptions",
        ]
        if unsigned:
            overlapped = ", ".join(
                f"{reading.region_id}/{reading.phase}~{int(reading.ewma_ns)}ns"
                for reading in unsigned
            )
            explanations.append(
                "side-stream region phase(s) overlap the main path by signature "
                f"and are excluded from the estimate: {overlapped}"
            )
        return RegionCostReport(
            readings=readings,
            estimated_main_path_extension_ns=estimate,
            signed_main_path_extension_ns=int(signed_main_path_extension_ns),
            explanations=tuple(explanations),
        )
