#!/usr/bin/env python3
"""Public consumer base class and fusion-group handle.

Canonical home of ``BaseConsumer`` and ``FusionGroupHandle``. They are port
contracts — the runtime and workgraph annotate against them — so they live
in the ports layer; ``tllm.consumers.base`` re-exports them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Mapping, Protocol, Sequence, TypeVar

from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.contracts.runtime_stats import WorkGraphRuntimeStats
from tllm.ports.base import ConsumerFlow

if TYPE_CHECKING:
    from tllm.runtime.specialization import RuntimeSpecializationSpec
    from tllm.runtime.scheduling.plan_store import PlanResources
    from tllm.workgraph.frontend import CapturedFlow
    from tllm.workgraph.hierarchy import CaptureAttachment
    from tllm.workgraph.physical import PhysicalCaptureRegion, PhysicalRegionExecutorDesc

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

    def workgraph_captures(self) -> tuple["CapturedFlow", ...]:
        """Return cold-path capture artifacts already specialized by this consumer."""

        return ()

    def workgraph_attachments(self) -> tuple["CaptureAttachment", ...]:
        """Return explicit runtime-node bindings for ``workgraph_captures``."""

        return ()

    def workgraph_region_executors(self) -> tuple["PhysicalRegionExecutorDesc", ...]:
        """Return frozen typed implementations for explicitly migrated regions."""

        return ()

    def cold_specialize_workgraph(self, spec: "RuntimeSpecializationSpec") -> None:
        """Optionally specialize capture metadata after the backend loads a model."""

        _ = spec

    def activate_workgraph_regions(
        self,
        regions: tuple["PhysicalCaptureRegion", ...],
    ) -> None:
        """Atomically replace cold-compiled regions attached to this consumer."""

        _ = regions

    def activate_workgraph_region_resources(
        self,
        regions: tuple["PhysicalCaptureRegion", ...],
        resources: tuple["PlanResources", ...],
    ) -> None:
        """Install placement-aware resources for explicitly migrated regions."""

        if resources:
            raise RuntimeError(
                "consumer does not support WorkGraph placement resources"
            )
        self.activate_workgraph_regions(regions)

    def drain_workgraph_specializations(self) -> int:
        """Run queued physical specialization work at a prepare safe point."""

        return 0

    def prepare_workgraph_step(self, step_id: int) -> None:
        """Acquire this consumer's staged physical region at a prepare safe point."""

        _ = step_id

    def finish_workgraph_step(self, step_id: int) -> None:
        """Release one prepared physical region after its legacy boundary runs."""

        _ = step_id

    def read_workgraph_runtime_stats(self) -> WorkGraphRuntimeStats:
        """Return a non-resetting snapshot for physical execution gates."""

        return WorkGraphRuntimeStats()

    def synchronize(self) -> None:
        """Optional flush hook for async/background consumers."""
        return None
