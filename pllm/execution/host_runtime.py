"""Host execution without a model: the facts a host workload compiles against (HostConfig), its compiled binding,
and the completion rows a generate call hands to host feedback (CompletionInput).

pllm.host.make_host compiles a pllm.Workload with these facts; the vLLM session writes CompletionInput rows.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import torch

from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason
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


@dataclass(frozen=True, slots=True)
class TensorHostBinding(Generic[InputT, DataT, ResultT]):
    graph: SemanticGraph
    layouts: TensorLayoutRequest
    math: MathBinding[InputT, DataT, ResultT]
    facts: HostConfig
    timeline: SchedulerTimeline

    def compile(self) -> StateExecution[InputT, DataT, ResultT]:
        facts = self.facts
        limits = tuple(
            value for value in (self.timeline.memory_budget_bytes, facts.memory_budget_bytes) if value is not None
        )
        timeline = replace(self.timeline, memory_budget_bytes=min(limits, default=None))
        providers = (
            (
                PageBundleProviderFacts(
                    f"canonical.page_bundle_{facts.device.type}@1",
                    3,
                    StateDeviceKind(facts.device.type),
                ),
            )
            if self.layouts.layouts
            else ()
        )
        plan = compile_page_bundle_plan(
            (self.graph, self.layouts),
            timeline,
            providers,
            1,
            facts.state_memory_budget_bytes,
            facts.device,
        )
        (anchor,) = (AnchorId(item.ordinal) for item in timeline.anchors if item.symbol == "host.next")
        return self.math.bind(PageBundleArena(plan, device=facts.device), anchor)


@dataclass(frozen=True, slots=True)
class CompletionInput:
    """One owned completion/hidden boundary, shared by host consumers."""

    completions: CompletionBatch
    hidden: torch.Tensor
    readiness: CudaStreamReadiness | None = None

    @classmethod
    def example(cls, facts: HostConfig, device: torch.device = torch.device("meta")) -> CompletionInput:
        """A shape-only input with domain-neutral completion metadata."""
        if facts.max_input_rows is None:
            raise ValueError("finite completion costs require max_input_rows")
        entries = tuple(
            Completion(RequestIdentity(str(i), i, 0), i, str(i), (), "", FinishReason.LENGTH)
            for i in range(facts.max_input_rows)
        )
        return cls(
            CompletionBatch(entries), torch.empty((len(entries), facts.hidden_size), dtype=torch.float32, device=device)
        )

    @classmethod
    def capture(
        cls, completions: CompletionBatch, hidden: torch.Tensor, stream: torch.cuda.Stream | None = None
    ) -> CompletionInput:
        if (
            (hidden.device.type == "cuda") != (stream is not None)
            or (stream is not None and stream.device != hidden.device)
            or hidden.ndim != 2
            or hidden.shape[0] != len(completions.entries)
            or hidden.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("completion input requires rows aligned to completions on its producer stream")
        with nullcontext() if stream is None else torch.cuda.stream(stream), torch.inference_mode(False):
            owned = hidden.detach().to(dtype=torch.float32, copy=True)
            return cls(completions, owned, CudaStreamReadiness(stream) if stream is not None else None)


class CompletionCapture:
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

    def finish(self, completions: CompletionBatch) -> CompletionInput:
        self.require_complete()
        assert tuple(entry.identity for entry in completions.entries) == self.identities
        return CompletionInput(completions, self.hidden, self._readiness)


@dataclass(frozen=True, slots=True)
class HostConfig:
    """What pllm.host.make_host compiles a workload against: the device, the width of hidden rows in host batches,
    the seed of initial state values, and the state and per-step memory limits (max_input_rows bounds a batch)."""

    device: torch.device
    hidden_size: int = 0
    initialization_seed: int = 0
    state_memory_budget_bytes: int = 64 * 1024 * 1024
    max_input_rows: int | None = None
    memory_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            self.device.type not in ("cpu", "cuda")
            or self.hidden_size < 0
            or self.initialization_seed < 0
            or self.state_memory_budget_bytes < 0
            or self.max_input_rows is not None
            and self.max_input_rows < 0
            or self.memory_budget_bytes is not None
            and self.memory_budget_bytes < 0
        ):
            raise ValueError(
                "HostConfig needs a cpu or cuda device and nonnegative hidden_size, initialization_seed, "
                f"state_memory_budget_bytes, max_input_rows and memory_budget_bytes; got {self}"
            )

    def timeline(self) -> SchedulerTimeline:
        compute_capability = "compute/cpu@1" if self.device.type == "cpu" else "compute/cuda@1"
        return SchedulerTimeline(
            memory_budget_bytes=self.memory_budget_bytes,
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
    facts: HostConfig,
    reserved_bytes: int,
    costs: list[tuple[Call, MemoryUsage]],
    input_shapes: tuple[tuple[int, ...], ...] = (),
) -> SchedulerTimeline:
    """The host timeline with each call's measured workspace and the bytes its one output keeps."""
    timeline = facts.timeline()
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
    "TensorHostBinding",
    "measured_timeline",
]
