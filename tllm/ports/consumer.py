#!/usr/bin/env python3
"""Public consumer base class and fusion-group handle.

Canonical home of ``BaseConsumer`` and ``FusionGroupHandle``. They are port
contracts for runtime-owned consumer dispatch; ``tllm.consumers.base``
re-exports them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Mapping, Protocol, Sequence, TypeVar

from tllm.contracts.consumer_stats import TrainingStats
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow

if TYPE_CHECKING:
    from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
    from tllm.contracts.recurrence import RecurrenceGraph, RecurrenceRequirements

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

    def recurrence_requirements(self) -> "RecurrenceRequirements | None":
        """Declare recurrent effects; the compiler owns schedule selection."""

        return None

    def recurrence_graph(
        self, *, token_count: int, layer_count: int
    ) -> "RecurrenceGraph":
        """Instantiate a finite recurrence after token/layer sizes are known."""

        _ = (token_count, layer_count)
        raise RuntimeError("consumer does not declare a recurrence graph")

    def recurrence_runtime_adapter(self, *, layer_count: int) -> object | None:
        """Return pure non-model op implementations for a recurrence graph.

        Model-layer execution and state materialization remain backend-owned.
        The adapter only evaluates consumer-declared executors such as a mix
        operation after the compiler has selected their order.
        """

        _ = layer_count
        return None

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        """Public flow-based consumption surface."""
        _ = (bundle, ctx)

    def consume_flow(self, flow_ordinal: int, bundle: PortBundle, ctx: RuntimeContext) -> None:
        """Per-flow entry point; defaults to the legacy whole-bundle surface.

        Program-generated consumers override this and dispatch on
        ``flow_ordinal`` (carried by ``FlowDispatchTarget``); legacy consumers
        keep working unchanged.
        """
        _ = flow_ordinal
        self.consume_bundle(bundle, ctx)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        """Optional hook called after the model step is complete."""
        _ = ctx

    def sampler_modifier_provider(self) -> "SamplerModifierProvider | None":
        """Optionally expose the provider for same-step sampler intervention."""

        return None

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

    def set_enabled(self, enabled: bool) -> None:
        """Apply the runtime training toggle when this consumer supports it."""

        _ = enabled
        raise RuntimeError("consumer does not support runtime enable updates")

    def read_and_reset_stats(self, *, sync: bool = True) -> TrainingStats:
        """Return training statistics for the shared runtime control plane."""

        _ = sync
        return TrainingStats(loss_avg=0.0, loss_count=0)

    def read_and_reset_per_request_stats(
        self,
        *,
        sync: bool = True,
    ) -> Mapping[int, TrainingStats]:
        """Return request-scoped training statistics when the consumer has them."""

        _ = sync
        return {}
