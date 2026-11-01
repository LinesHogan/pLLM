"""CUDA graph capture for fixed-table tensor bodies.

StepCapture captures one body per tensor task (TensorExecutor, FieldWriteExecutor) and overlay tile
(OverlayBatch) and replays it; capture_graph is the warmup-and-capture routine it shares with vLLM layer replay.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

from pllm.contracts.semantic_graph import TaskResourceFact
from pllm.execution.torch_private import CudaPoolHandle, cuda_graph_pool_bytes

R = TypeVar("R")


def _ready() -> None:
    pass


def capture_graph(
    device: torch.device,
    body: Callable[[], R],
    *,
    prepare: Callable[[], None] = _ready,
    restore: Callable[[], None] = _ready,
    pool: CudaPoolHandle | None = None,
) -> tuple[torch.cuda.CUDAGraph, R, int]:
    """Warm `body` up three times on a private stream, capture it once and report the graph pool's bytes.

    `prepare` runs before every warmup and before the capture, for state the body mutates.
    """
    stream = torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            restore()
            prepare()
            body()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    collecting = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.stream(stream):
            restore()
        # Cyclic collection can release tensors from another CUDA stream and invalidate capture.
        # Torch's context collects before starting; postpone automatic collection until capture ends.
        with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle() if pool is None else pool):
            prepare()
            result = body()
    finally:
        if collecting:
            gc.enable()
    current.wait_stream(stream)
    return graph, result, cuda_graph_pool_bytes(graph.pool(), device.index or 0)


@dataclass
class StepCapture:
    """One task's fixed capacity, private graph pool and replay stream ordering."""

    entitlement: TaskResourceFact
    states: int
    graph: torch.cuda.CUDAGraph | None = None
    addresses: tuple[int, ...] = ()
    resident_bytes: int = 0
    stream: torch.cuda.Stream | None = None

    def run(
        self,
        body: Callable[[], None],
        table: torch.Tensor,
        destinations: torch.Tensor,
        addresses: tuple[int, ...],
        refill: Callable[[], None],
        buffers: tuple[torch.Tensor, ...] = (),
    ) -> None:
        """Capture when addresses change; warmup never writes live destinations. The caller has called `order`
        before refilling the table and buffers."""
        if self.graph is None or self.addresses != addresses:
            self._capture(body, table, destinations, buffers)
            refill()
        self.addresses = addresses
        assert self.graph is not None
        self.graph.replay()

    def order(self, stream: torch.cuda.Stream | None) -> None:
        """Order buffer refills as well as replay after the preceding replay; call before refilling."""
        if stream is not None and self.stream is not None and stream != self.stream:
            stream.wait_stream(self.stream)
        self.stream = stream

    def _capture(
        self,
        body: Callable[[], None],
        table: torch.Tensor,
        destinations: torch.Tensor,
        buffers: tuple[torch.Tensor, ...],
    ) -> None:
        # Release the invalidated pool before allocating its replacement.
        self.graph = None
        # Inactive destinations make warmup and capture read-only with respect to the arena.
        destinations.fill_(-1)
        self.graph, _, pool = capture_graph(table.device, body)
        self.resident_bytes = sum(value.untyped_storage().nbytes() for value in (table, *buffers)) + pool
        if self.resident_bytes > self.entitlement.resident_bytes:
            raise ValueError(
                f"capture requires {self.resident_bytes} resident bytes, reserved {self.entitlement.resident_bytes}"
            )
