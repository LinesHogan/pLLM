#!/usr/bin/env python3
"""GPU race gate for shape-keyed pinned host staging across graph captures."""

from __future__ import annotations

import json

import torch

from tllm.runtime.torch_adapter import CudaLongHostStaging


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    staging = CudaLongHostStaging(4, device)
    target = torch.zeros(4, dtype=torch.long, device=device)

    graphs: list[torch.cuda.CUDAGraph] = []
    with torch.cuda.stream(stream):
        for _capture in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                staging.copy([1, 2], target)
            graphs.append(graph)
        for graph in graphs:
            graph.replay()
    stream.synchronize()
    torch.testing.assert_close(
        target[:2], torch.tensor([1, 2], device=device)
    )

    with torch.cuda.stream(stream):
        staging.copy([7, 9], target)
        for graph in graphs:
            graph.replay()
    stream.synchronize()
    torch.testing.assert_close(
        target[:2], torch.tensor([7, 9], device=device)
    )
    print(json.dumps({"captures": 2, "reused_values": [7, 9]}))


if __name__ == "__main__":
    main()
