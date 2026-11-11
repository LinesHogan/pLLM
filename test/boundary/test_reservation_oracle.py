"""The planned state pages are exactly what the engine can hold at once: a dry run of the real ledger reaches them.

Each workload is planned as `pllm.explain` plans it, then the real StateExecution, request directory, post-sample
overlap, arena and Rust ledger run with exactly the planned pages, driven in VllmSession's order of state calls.
Task bodies are stubbed at the ledger level: a writer opens a private transition of every row it may write and
publishes as its effect says; a writer fed by a host batch runs only in a generate call's first step. An adversarial
schedule must use every planned page of every field (the reservation is tight) and random schedules must never
need more (a transition that found no free page would raise).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch

import pllm
from pllm import testing
from pllm.backends.vllm.compile_facts import compile_facts
from pllm.backends.vllm.hidden_taps import HiddenTapAnnouncement
from pllm.backends.vllm.model_shape import ModelShape
from pllm.config import ModelConfig
from pllm.contracts.backend_attachment import AttachmentGraphMode
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.host_batch import HOST_BATCH_SCHEMAS
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, HostServiceInputSource, ImplementationId, StateEffectKind
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import StateTensorLayout, TensorLayoutRequest
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_frame import TensorFrame
from pllm.workloads import ESampConfig, ESampWorkload, OverRIDEConfig, OverRIDEWorkload, TLMConfig, TLMWorkload
from pllm.workloads.tlm.workload import TLMMode
from pllm.workloads.video import VideoWorkload
from test.boundary.test_writer_fields import InstanceDecodeFactors

SEQS = 4
PUBLISHES = (StateEffectKind.UPDATE, StateEffectKind.UPDATE_MAYBE, StateEffectKind.PUBLISH)


def plan(workload: pllm.Workload) -> PlanBundle:
    """The CUDA plan pllm.explain derives from config.json, moved to CPU pages of one element per field."""
    config = ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=SEQS)
    requirements = workload.backend_requirements()
    shape = ModelShape.of_config(config, requirements.parameter_targets)
    taps = HiddenTapAnnouncement.declared(requirements, config).resolve(
        shape.layer_count, shape.hidden_size, shape.vocab_size
    )
    facts = compile_facts(
        shape,
        requirements,
        config,
        model_identity=shape.architecture,
        hidden_layouts=taps,
        overlay_graph_mode=AttachmentGraphMode.CUDA_GRAPH,
        memory_bytes=1 << 40,
    )
    bundle = workload.plan_backend(*workload.specialize(facts, config.execution)).plan
    layouts = TensorLayoutRequest(
        tuple(
            StateTensorLayout(
                entry.resource, replace(entry.layout, fields=tuple(replace(f, shape=()) for f in entry.layout.fields))
            )
            for entry in bundle.tensor_layout_request.layouts
        )
    )
    plans = tuple(replace(p, device_kind=StateDeviceKind.CPU, host_bytes=0) for p in bundle.states.plans)
    return replace(bundle, tensor_layout_request=layouts, states=replace(bundle.states, plans=plans))


class PeakArena(PageBundleArena):
    """The real arena, recording after every allocation the most pages each field held."""

    def __init__(self, bundle: PlanBundle) -> None:
        super().__init__(bundle, device=torch.device("cpu"))
        self.peak = {resource: [0] * len(pools.tensors) for resource, pools in self.pools.items()}

    def _sample(self) -> None:
        for resource, pools in self.pools.items():
            for field, capacity in enumerate(self._plans[resource].page_capacities):
                self.peak[resource][field] = max(self.peak[resource][field], capacity - pools.free_count(field))

    def prepare_admission(self, resources, stream=None):
        token = super().prepare_admission(resources, stream)
        self._sample()
        return token

    def prepare_transition(self, sources, stream=None, *, fields=None, copy_fields=None):
        token = super().prepare_transition(sources, stream, fields=fields, copy_fields=copy_fields)
        self._sample()
        return token


class StubMath:
    """StateMath without tensors: a step's data is its StepInput."""

    parameter_overlay = None
    parameter_phases = ()
    model_sites = ()
    attention = None

    def initialize(self, roots, addresses) -> None:
        pass

    def rows(self, value: StepInput):
        return value.rows.rows

    def prepare(self, value: StepInput, context) -> TensorFrame:
        return TensorFrame(value, value.rows.rows)

    def output(self, context) -> None:
        return None


@dataclass(frozen=True)
class StubTask:
    """One task's page effects: a private write of every row it may write, published as its effect says."""

    implementation_id: ImplementationId
    effects: tuple[tuple[str, StateEffectKind], ...]
    host_batch: bool

    def execute(self, context: StateTaskContext[TensorFrame]) -> dict:
        if self.host_batch and context.data().step.batch is None:
            return {}
        for resource, kind in self.effects:
            write = context.pending_write(resource)
            if kind is not StateEffectKind.PUBLISH and write is None:
                refs = context.current_owned_refs(context.selection.for_resource(resource).refs)
                write = context.prepare_owned_write(refs) if len(refs) else None
            if write is not None and kind in PUBLISHES:
                context.publish_owned_write(write)
        return {}


@dataclass(frozen=True)
class Step:
    """Requests vLLM finished or preempted before a step, requests it admits, and requests it runs."""

    retire: tuple[int, ...] = ()
    admit: tuple[int, ...] = ()
    run: tuple[int, ...] | None = None  # None runs every live request
    new_generate: bool = False  # the step starts a generate call, whose host batch is staged


class Engine:
    """VllmSession's state calls: retire and admit (before_execute), open the step before the forward for model
    readers and host-batch training with admission pins after request.prepare (prepare_input) or after it with the
    overlap history (complete_forward), every anchor, commit; finish_requests retires everything."""

    def __init__(self, bundle: PlanBundle) -> None:
        graph = bundle.semantic_graph
        tasks = tuple(
            StubTask(
                physical.implementation_id,
                tuple((str(e.resource), e.kind) for e in task.effects if e.kind is not StateEffectKind.READ),
                any(
                    isinstance(port.source, HostServiceInputSource) and port.source.schema in HOST_BATCH_SCHEMAS
                    for port in task.inputs
                ),
            )
            for physical in sorted(bundle.schedule.tasks, key=lambda item: item.implementation_id)
            for task in (graph.tasks[physical.task_id],)
        )
        self.arena = PeakArena(bundle)
        sampler = next(anchor.ordinal for anchor in bundle.timeline.anchors if anchor.symbol == "sampler.pre_draw")
        directory = RequestStateDirectory(self.arena, graph.states)
        self.execution = StateExecution(self.arena, directory, StubMath(), tasks, AnchorId(sampler))
        self.execution.admit_instances()
        self.early = bool(graph.model_reads) or graph.trains_on_batch
        self.live: list[int] = []
        self.batch: object | None = None
        self.step_id = 0

    def retire(self, requests: tuple[int, ...]) -> None:
        directory = self.execution.directory
        held = tuple(
            identity
            for identity in map(_identity, requests)
            if directory.request_states or directory.has_inference_pin(identity)
        )
        if held:
            self.execution.retire(held)
        self.live = [i for i in self.live if i not in requests]

    def run(self, step: Step | None) -> None:
        execution = self.execution
        if step is None:
            execution.release_history()
            self.retire(tuple(self.live))
            return
        if step.new_generate:
            self.batch = object()
        self.retire(step.retire)
        if step.admit and execution.directory.request_states:
            execution.admit(tuple(map(_identity, step.admit)))
        self.live += step.admit
        rows = tuple(map(_identity, sorted(self.live if step.run is None else step.run)))
        self.step_id += 1
        if not rows:
            return
        value = replace(testing.step_input(rows, step_id=self.step_id), batch=self.batch)
        overlap = execution.post_sample
        if self.early:
            context = execution.start_step(value)
            execution.advance_to(context, "request.prepare")
            pending = tuple(
                identity
                for identity in rows
                if execution.directory.pinned_states and not execution.directory.has_inference_pin(identity)
            )
            if pending:
                execution.admit(pending)
        else:
            context = execution.start_step(value, history=() if overlap is None else overlap.history)
        self.batch = None  # instance states consume the host batch in the call's first step
        for anchor in execution.arena.plan.timeline.anchors:
            execution.advance_to(context, anchor.symbol)
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(context)
        execution.finish_step(context)


def _identity(index: int) -> RequestIdentity:
    return RequestIdentity(f"r{index}", index, 0)


def adversary() -> Iterator[Step | None]:
    """All requests at once; one admitted per step, then two steps without admission (the latest version leaves every
    admission pin behind) and churn; one preempted and re-admitted per step. None ends a generate call."""
    yield Step(admit=tuple(range(SEQS)), new_generate=True)
    yield from (Step() for _ in range(4))
    yield None
    yield Step(admit=(SEQS,), new_generate=True)
    yield from (Step(admit=(i,)) for i in range(SEQS + 1, 2 * SEQS))
    yield from (Step(), Step())
    for i in range(2 * SEQS, 2 * SEQS + 4):
        yield Step(retire=(i - SEQS,), admit=(i,))
        yield Step()
    yield None
    yield Step(admit=tuple(range(100, 100 + SEQS)), new_generate=True)
    yield from (Step(retire=(100 + i % SEQS,), admit=(100 + i % SEQS,)) for i in range(SEQS + 2))
    yield None


def random_schedule(rng: random.Random, steps: int) -> Iterator[Step | None]:
    """Random retirement, admission and partial steps; a generate call ends when its requests are gone."""
    live: list[int] = []
    next_request = 0
    for _ in range(steps):
        new_generate = not live
        retire = tuple(i for i in live if rng.random() < 0.3)
        live = [i for i in live if i not in retire]
        count = rng.randint(0 if live else 1, SEQS - len(live))
        admit = tuple(range(next_request, next_request + count))
        next_request += count
        live += admit
        run = None if rng.random() < 0.6 else tuple(i for i in live if rng.random() < 0.7)
        yield Step(retire, admit, run, new_generate)
        if not live or rng.random() < 0.1:
            yield None
            live = []
    yield None


def peaks(bundle: PlanBundle, schedule: Iterator[Step | None]) -> dict[str, list[int]]:
    engine = Engine(bundle)
    for step in schedule:
        engine.run(step)
    return engine.arena.peak


@pytest.mark.parametrize(
    "workload",
    [
        ESampWorkload(ESampConfig(rank=2)),
        ESampWorkload(ESampConfig(rank=2, intervention_beta=0.1)),
        OverRIDEWorkload(OverRIDEConfig(rank=2, num_iterations=3)),
        TLMWorkload(TLMConfig(), rank=2),
        TLMWorkload(TLMConfig(mode=TLMMode.OFFLINE), rank=2),
        VideoWorkload(),
        InstanceDecodeFactors(),
    ],
    ids=("esamp", "esamp-guided", "override", "tlm-online", "tlm-offline", "video", "decode-factors"),
)
def test_planned_pages_are_reached_and_never_exceeded(workload):
    bundle = plan(workload)
    planned = {p.resource: list(p.page_capacities) for p in bundle.states.plans}
    fed = {str(r) for group in bundle.semantic_graph.input_groups for r in group.borrowed_resources}
    reached = peaks(bundle, adversary())
    assert {r: pages for r, pages in reached.items() if r not in fed} == {
        r: pages for r, pages in planned.items() if r not in fed
    }
    rng = random.Random(0)
    for _ in range(20):
        for resource, pages in peaks(bundle, random_schedule(rng, 12)).items():
            assert all(used <= limit for used, limit in zip(pages, planned[resource], strict=True))
