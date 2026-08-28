"""Author-facing ergonomics kit for native Region workloads.

The kit owns the process-local lifecycle boilerplate (resource assembly,
specialization identity, cold specialization, optimizer construction, update
finalization, worker reconstruction, host channels) so an algorithm
implementation reads like a description of its logic.  The signed
declaration surface (``State``/``@task``/timing/effects) stays explicit —
that *is* the algorithm.
"""

from tllm.kit.algorithm import RegionAlgorithm
from tllm.kit.channel import HostChannelBox, clear_host_channels, host_channel
from tllm.kit.resources import RegionResourcesBase

__all__ = [
    "HostChannelBox",
    "RegionAlgorithm",
    "RegionResourcesBase",
    "clear_host_channels",
    "host_channel",
]
