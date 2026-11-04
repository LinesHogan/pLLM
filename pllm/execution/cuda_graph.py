"""Device-selected CUDA graph branches shared by scans and adaptive model slices."""

from collections.abc import Generator
from contextlib import contextmanager

import torch


@contextmanager
def device_branch(predicate: torch.Tensor) -> Generator[None]:
    """Capture a CUDA IF node; cold warmup runs every branch without reading its predicate."""
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph() if torch.cuda.is_current_stream_capturing() else None
    if graph is not None:
        graph.begin_capture_to_if_node(predicate)  # pyright: ignore[reportUnknownMemberType]
    try:
        yield
    finally:
        if graph is not None:
            graph.end_capture_to_conditional_node()  # pyright: ignore[reportUnknownMemberType]
