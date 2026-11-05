"""Device-selected CUDA graph branches shared by scans and adaptive model slices."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from pllm.execution.torch_private import CudaPoolHandle, release_graph_pool_scope

_pool: ContextVar[CudaPoolHandle | None] = ContextVar("conditional_graph_pool", default=None)


@contextmanager
def conditional_graph_pool(pool: CudaPoolHandle) -> Generator[None]:
    """Name the owning pool while Torch switches allocation scopes for conditional nodes."""
    token = _pool.set(pool)
    try:
        yield
    finally:
        _pool.reset(token)


def _release_scope() -> None:
    # Torch's conditional begin/end each reopen the same pool, incrementing its
    # use count, but CUDAGraph.reset releases only the capture's first reference.
    # Balance those extra references; the live graph still owns the pool.
    pool = _pool.get()
    if pool is not None:
        release_graph_pool_scope(pool)


@contextmanager
def device_branch(predicate: torch.Tensor) -> Generator[None]:
    """Capture a CUDA IF node; cold warmup runs every branch without reading its predicate."""
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph() if torch.cuda.is_current_stream_capturing() else None
    if graph is not None:
        graph.begin_capture_to_if_node(predicate)  # pyright: ignore[reportUnknownMemberType]
        _release_scope()
    try:
        yield
    finally:
        if graph is not None:
            graph.end_capture_to_conditional_node()  # pyright: ignore[reportUnknownMemberType]
            _release_scope()
