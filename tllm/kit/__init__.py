"""Author-facing ergonomics kit for native Region workloads.

The kit owns the process-local lifecycle boilerplate (resource assembly,
specialization identity, cold specialization, optimizer construction, update
finalization, worker reconstruction, host channels) so an algorithm
implementation reads like a description of its logic.  The signed
declaration surface (``State``/``@task``/timing/effects) stays explicit —
that *is* the algorithm.
"""

from tllm.kit.algorithm import RegionAlgorithm
from tllm.kit.channel import (
    HostChannelBox,
    clear_host_channels,
    host_channel,
    typed_host_channel,
)
from tllm.kit.lm_head import ResolvedLMHead, resolve_lm_head
from tllm.kit.resources import RegionResourcesBase
from tllm.kit.sampler import SamplerModifier, candidate_modifier_state
from tllm.kit.sink import SinkRecord, SinkStore, StepRowIdentity

__all__ = [
    "HostChannelBox",
    "RegionAlgorithm",
    "RegionResourcesBase",
    "ResolvedLMHead",
    "SamplerModifier",
    "SinkRecord",
    "SinkStore",
    "StepRowIdentity",
    "candidate_modifier_state",
    "clear_host_channels",
    "host_channel",
    "resolve_lm_head",
    "typed_host_channel",
]
