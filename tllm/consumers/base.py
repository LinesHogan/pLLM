#!/usr/bin/env python3
"""Base consumer interface for hidden producer-consumer framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Mapping, Protocol, Sequence, TypeVar

from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow

ConfigT = TypeVar("ConfigT")


class FusionGroupHandle(Protocol):
    """One executable artifact for an exact compiler-selected flow group."""

    def execute(
        self,
        bundles: Mapping[str, PortBundle],
        ctx: RuntimeContext,
    ) -> None: ...

    def replay_count(self) -> int: ...


class BaseConsumer(ABC, Generic[ConfigT]):
    #: Each consumer narrows this to its own config dataclass; declared here
    #: so shared infra (registry spec snapshots) can reach it without probing.
    config: ConfigT

    @property
    @abstractmethod
    def consumer_id(self) -> str:
        raise NotImplementedError

    def flows(self) -> Sequence[ConsumerFlow]:
        """Public consumer declaration surface."""
        return ()

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        """Public flow-based consumption surface."""
        _ = (bundle, ctx)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        """Optional hook called after the model step is complete."""
        _ = ctx

    def retire_device_lease(self, ctx: RuntimeContext) -> None:
        """Fence any step-lifetime device inputs before producer reuse."""
        _ = ctx

    def fusion_replay_count(self) -> int:
        """Host-visible replay counter for a compiler-selected fused callable."""
        return 0

    def fusion_group_handle(
        self,
        flow_ids: tuple[str, ...],
    ) -> FusionGroupHandle | None:
        """Return the single executable artifact owning ``flow_ids``."""
        _ = flow_ids
        return None

    def synchronize(self) -> None:
        """Optional flush hook for async/background consumers."""
        return None
