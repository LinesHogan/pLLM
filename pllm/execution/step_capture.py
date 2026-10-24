"""CUDA graph addresses, capture and replay used by the shared tensor executor."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass

import torch

from pllm.contracts.semantic_graph import TaskResourceFact
from pllm.execution.codec import cuda_graph_pool_bytes


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
        stream: torch.cuda.Stream | None,
        refill: Callable[[], None],
    ) -> None:
        """Capture when addresses change; warmup never writes live destinations."""
        if stream is not None and self.stream is not None and stream != self.stream:
            stream.wait_stream(self.stream)
        self.stream = stream
        if self.graph is None or self.addresses != addresses:
            self._capture(body, table, destinations)
            refill()
        self.addresses = addresses
        assert self.graph is not None
        self.graph.replay()

    def _capture(self, body: Callable[[], None], table: torch.Tensor, destinations: torch.Tensor) -> None:
        stream = torch.cuda.Stream(device=table.device)
        current = torch.cuda.current_stream(table.device)
        stream.wait_stream(current)
        # Inactive destinations make warmup and capture read-only with respect to the arena.
        with torch.cuda.stream(stream):
            destinations.fill_(-1)
            for _ in range(3):
                body()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        collecting = gc.isenabled()
        gc.disable()
        try:
            # Cyclic collection can release tensors from another CUDA stream
            # and invalidate capture. Torch's context collects before starting;
            # postpone automatic collection until capture has ended as well.
            with torch.cuda.graph(self.graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
                body()
        finally:
            if collecting:
                gc.enable()
        current.wait_stream(stream)
        self.resident_bytes = table.untyped_storage().nbytes() + cuda_graph_pool_bytes(
            self.graph.pool(), table.device.index or 0
        )
        if self.resident_bytes > self.entitlement.resident_bytes:
            raise ValueError(
                f"capture requires {self.resident_bytes} resident bytes, reserved {self.entitlement.resident_bytes}"
            )
