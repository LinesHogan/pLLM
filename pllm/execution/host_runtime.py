"""Host execution without a model: the settings a host workload compiles against (HostConfig), planning and binding
its graph (bind_host), and the completion rows a generate call hands to host feedback (CompletionInput).

pllm.host.make_host compiles a pllm.Workload with this config; the vLLM session writes CompletionInput rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeVar

import torch

from pllm.contracts.feedback import CompletionBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    SchedulerTimeline,
    SemanticGraph,
    TaskResourceFact,
    TimelineAnchor,
    ValueResourceFact,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorLayoutRequest
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.declaration import Call
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.torch_private import MemoryUsage

ResultT = TypeVar("ResultT")
InputT = TypeVar("InputT")
DataT = TypeVar("DataT")


def bind_host(
    graph: SemanticGraph,
    layouts: TensorLayoutRequest,
    math: MathBinding[InputT, DataT, ResultT],
    config: HostConfig,
    timeline: SchedulerTimeline,
) -> StateExecution[InputT, DataT, ResultT]:
    """Plan a host graph against `timeline` and bind its math to a page arena on `config.device`."""
    providers = (
        (PageBundleProviderFacts(f"page_bundle_{config.device.type}@1", StateDeviceKind(config.device.type)),)
        if layouts.layouts
        else ()
    )
    plan = compile_page_bundle_plan((graph, layouts), timeline, providers, 1, config.state_page_budget_bytes)
    (anchor,) = (AnchorId(item.ordinal) for item in timeline.anchors if item.symbol == "host.next")
    return math.bind(PageBundleArena(plan, device=config.device), anchor)


@dataclass(frozen=True, slots=True)
class CompletionInput:
    """The completions of one generate call and each one's final hidden row (float32, one row per completion),
    shared by host consumers; generate has finished writing `hidden` when it returns."""

    completions: CompletionBatch
    hidden: torch.Tensor


class CompletionRows:
    """Write final sampling rows directly into the original owned output tensor."""

    def __init__(self, identities: tuple[RequestIdentity, ...], hidden: torch.Tensor) -> None:
        self.identities, self.hidden = identities, hidden
        self._indices = {identity: index for index, identity in enumerate(identities)}
        self._written: set[RequestIdentity] = set()
        self._readiness = (
            CudaStreamReadiness(torch.cuda.current_stream(hidden.device)) if hidden.device.type == "cuda" else None
        )

    def write(self, identities: tuple[RequestIdentity, ...], rows: torch.Tensor) -> None:
        assert rows.shape == (len(identities), self.hidden.shape[1])
        with torch.inference_mode(False), torch.no_grad():
            if self._readiness is not None:
                self._readiness.order_on(torch.cuda.current_stream(self.hidden.device))
                self.hidden.record_stream(torch.cuda.current_stream(self.hidden.device))
            for identity, row in zip(identities, rows.unbind(), strict=True):
                self.hidden[self._indices[identity]].copy_(row.detach(), non_blocking=False)
                self._written.add(identity)
            if self.hidden.device.type == "cuda":
                self._readiness = CudaStreamReadiness(torch.cuda.current_stream(self.hidden.device))

    def require_complete(self) -> None:
        if len(self._written) != len(self.identities):
            raise RuntimeError("completion capture is missing submitted identities")

    def synchronize(self) -> None:
        if self._readiness is not None:
            self._readiness.synchronize()


@dataclass(frozen=True, slots=True)
class HostConfig:
    """What pllm.host.make_host compiles a workload against: the device, the width of hidden rows in host batches
    and the shape of each row's target (() for one value per row), the seed of initial state values, and the state
    and per-step memory limits (max_input_rows bounds a batch)."""

    device: torch.device
    hidden_size: int = 0
    target_shape: tuple[int, ...] = ()
    seed: int = 0
    state_page_budget_bytes: int = 64 * 1024 * 1024
    max_input_rows: int | None = None
    execution_peak_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            self.device.type not in ("cpu", "cuda")
            or self.hidden_size < 0
            or any(size < 1 for size in self.target_shape)
            or self.seed < 0
            or self.state_page_budget_bytes < 0
            or self.max_input_rows is not None
            and self.max_input_rows < 0
            or self.execution_peak_budget_bytes is not None
            and self.execution_peak_budget_bytes < 0
        ):
            raise ValueError(
                "HostConfig needs a cpu or cuda device, positive target_shape sizes and nonnegative hidden_size, "
                f"seed, state_page_budget_bytes, max_input_rows and execution_peak_budget_bytes; got {self}"
            )

    def timeline(self) -> SchedulerTimeline:
        compute_capability = "compute/cpu@1" if self.device.type == "cpu" else "compute/cuda@1"
        return SchedulerTimeline(
            execution_peak_budget_bytes=self.execution_peak_budget_bytes,
            anchors=(
                TimelineAnchor(AnchorSymbol("host.input.ready"), 0),
                TimelineAnchor(AnchorSymbol("host.compute.end"), 1),
                TimelineAnchor(AnchorSymbol("host.next"), 2),
            ),
            capabilities=tuple(
                sorted(
                    (
                        compute_capability,
                        "optimizer/adamw@1",
                        "transition/full_root@1",
                    )
                )
            ),
        )


def measured_timeline(
    config: HostConfig,
    reserved_bytes: int,
    costs: list[tuple[Call, MemoryUsage]],
    input_shapes: tuple[tuple[int, ...], ...] = (),
) -> SchedulerTimeline:
    """The host timeline with each call's measured workspace and the bytes its one output keeps."""
    timeline = config.timeline()
    costs = sorted(costs, key=lambda cost: cost[0].symbol)
    return replace(
        timeline,
        reserved_bytes=(reserved_bytes,) * len(timeline.anchors),
        task_resources=tuple(
            TaskResourceFact(call.symbol, usage.workspace, input_shapes=input_shapes) for call, usage in costs
        ),
        value_resources=tuple(ValueResourceFact(call.symbol, call.outputs[0], usage.output) for call, usage in costs),
    )


__all__ = [
    "CompletionInput",
    "HostConfig",
    "bind_host",
    "measured_timeline",
]
