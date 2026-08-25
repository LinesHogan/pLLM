"""Runtime-neutral snapshots of static device resources."""

from __future__ import annotations

import torch


def static_device_headroom_bytes(
    device: torch.device, *, configured_budget: int
) -> int:
    """Read device headroom at cold setup, never from a scheduling step."""
    try:
        if device.type != "cuda":
            raise RuntimeError("runner device is not CUDA")
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        headroom = int(free_bytes)
        if headroom <= 0:
            raise RuntimeError("CUDA reported no free memory")
        return headroom
    except Exception as exc:
        if configured_budget > 0:
            return configured_budget
        raise RuntimeError(
            "cannot derive static workload memory headroom; configure "
            "workload_memory_budget_bytes explicitly"
        ) from exc
