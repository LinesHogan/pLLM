"""Prototype for docs/cleanup/pttsdryrun: dry-run PTTS page reservations and per-field reads/writes on CPU.

For each workload the script plans the real 7B schedule (as `pllm.explain` does), traces every tensor task on meta
inputs to find the state fields it reads and writes, then drives the real StateExecution, RequestStateDirectory,
PostSampleOverlap, PageBundleArena and Rust PageLedger through adversarial engine schedules (the order of state
calls VllmSession makes). Task bodies are stubbed at the ledger level: a writer opens a private transition of every
row it may write, whenever its trigger allows, and publishes as its effect says. Field shapes shrink to scalars;
page counts do not depend on them. After every allocation the arena records, per field, the pages today's ledger
holds and the pages a field-aware ledger would hold (old versions keep only the fields some pin reads).

    PYTHONPATH=. /workspace/pllm/.venv/bin/python tools/ptts_dryrun.py [--seqs 128] [--fuzz 300]
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import pllm
from pllm import testing
from pllm.backends.vllm.compile_facts import compile_facts
from pllm.backends.vllm.hidden_taps import HiddenTapAnnouncement
from pllm.backends.vllm.model_shape import ModelShape
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.backend_attachment import AttachmentGraphMode
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, HostServiceInputSource, StateEffectKind, StateScope
from pllm.contracts.state_feed import FEED_SLOTS
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId, StateTensorLayout, TensorLayoutRequest
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import page_stride_bytes
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.step_input import StepInput
from pllm.execution.task_memory import example, task_example
from pllm.execution.tensor_frame import TensorFrame
from pllm.explanation import _specialize
from pllm.workloads import ESampConfig, ESampWorkload, OverRIDEWorkload, TLMConfig, TLMWorkload
from pllm.workloads.state_producers import DecodeFactorsWorkload, seed_factors
from pllm.workloads.tlm.workload import TLMMode
from pllm.workloads.video import VideoWorkload

WRITES = (
    StateEffectKind.TRANSITION,
    StateEffectKind.TRANSITION_MAYBE,
    StateEffectKind.UPDATE,
    StateEffectKind.UPDATE_MAYBE,
)
PUBLISHES = (StateEffectKind.UPDATE, StateEffectKind.UPDATE_MAYBE, StateEffectKind.PUBLISH)


class InstanceDecodeFactors(DecodeFactorsWorkload):
    """DecodeFactors with one factor state shared by every request's Adapter (test_state_producers)."""

    @property
    def factors(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"a": pllm.Field(lambda i, o: (self.rank, i)), "b": pllm.Field(lambda i, o: (o, self.rank))},
            optimizer=pllm.AdamW(lr=1e-3),
            init=pllm.Computed(seed_factors),
        )


WORKLOADS: dict[str, pllm.Workload] = {
    "ESamp b=0": ESampWorkload(ESampConfig(rank=4)),
    "ESamp b>0": ESampWorkload(ESampConfig(rank=4, intervention_beta=0.1)),
    "ESamp b=0 r128": ESampWorkload(ESampConfig(rank=128)),
    "ESamp b>0 r128": ESampWorkload(ESampConfig(rank=128, intervention_beta=0.1)),
    "OverRIDE": OverRIDEWorkload(),
    "TLM online": TLMWorkload(TLMConfig()),
    "TLM offline": TLMWorkload(TLMConfig(mode=TLMMode.OFFLINE)),
    "video": VideoWorkload(),
    "DecodeFactors inst": InstanceDecodeFactors(),
}


# ---------------------------------------------------------------------------------------------------------------
# Planning: the facts pllm.explain uses for the model at max_num_seqs, without weights or CUDA.


def plan_for(workload: pllm.Workload, model: str, seqs: int):
    config = ModelConfig(model=model, max_num_seqs=seqs)
    requirements = workload.backend_requirements()
    shape = ModelShape.of_config(config, requirements.parameter_targets)
    layouts = HiddenTapAnnouncement.declared(requirements, config).resolve(
        shape.layer_count, shape.hidden_size, shape.vocab_size
    )
    facts = compile_facts(
        shape,
        requirements,
        config,
        model_identity=shape.architecture,
        hidden_layouts=layouts,
        overlay_graph_mode=AttachmentGraphMode.CUDA_GRAPH,
        memory_bytes=1 << 60,
    )
    facts = replace(facts, probe=replace(facts.probe, blas_workspace_bytes=0, device_index=0))
    facts, binding, _ = _specialize(workload, facts, config)
    return facts, binding, workload.plan_backend(facts, binding).plan


def scalar_plan(plan: PlanBundle, capacity: Callable[[str], int]) -> PlanBundle:
    """The same plan on CPU with one-element fields and `capacity(resource)` pages per field."""
    layouts = TensorLayoutRequest(
        tuple(
            StateTensorLayout(
                entry.resource, replace(entry.layout, fields=tuple(replace(f, shape=()) for f in entry.layout.fields))
            )
            for entry in plan.tensor_layout_request.layouts
        )
    )
    plans = tuple(
        replace(
            p,
            device_kind=StateDeviceKind.CPU,
            page_capacities=(capacity(p.resource),) * len(p.page_capacities),
            host_bytes=0,
        )
        for p in plan.states.plans
    )
    return replace(plan, tensor_layout_request=layouts, states=replace(plan.states, plans=plans))


# ---------------------------------------------------------------------------------------------------------------
# Field analysis: which fields each task reads as data and writes, from a meta trace of its planning body.


class FieldTrace(TorchDispatchMode):
    """Record the state fields an operation reads as data or writes, from each aten schema's alias facts.

    A view argument (aliasing, not written) is neither; an argument the schema writes is a write; any other
    argument is a data read. Shape-only uses such as `.numel()` and `new_*`/`*_like` factories read no data.
    """

    def __init__(self, fields: dict[StorageWeakRef, tuple[str, int]]) -> None:
        super().__init__()
        self.fields = fields
        self.reads: set[tuple[str, int]] = set()
        self.writes: set[tuple[str, int]] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = func._schema
        name = schema.name.split("::")[-1]
        if name.startswith("new_") or name.endswith("_like"):
            return func(*args, **kwargs)
        values = (*args, *(kwargs.get(a.name) for a in schema.arguments[len(args) :]))
        for argument, value in zip(schema.arguments, values, strict=False):
            for tensor in tree_leaves(value):
                key = (
                    self.fields.get(StorageWeakRef(tensor.untyped_storage()))
                    if isinstance(tensor, torch.Tensor)
                    else None
                )
                if key is None:
                    continue
                alias = argument.alias_info
                if alias is not None and alias.is_write:
                    self.writes.add(key)
                elif alias is None:
                    self.reads.add(key)
        return func(*args, **kwargs)


@dataclass(frozen=True)
class StateFields:
    """One state's field facts: fields some pin reads, fields some writer modifies, and every field."""

    pinned: frozenset[int]
    written: frozenset[int]
    width: int


def state_fields(binding, facts, plan: PlanBundle) -> tuple[dict[str, StateFields], list[str]]:
    """Trace every tensor task on meta inputs; return per-state field facts and a log of each task's sets.

    Reads by a state's own writer happen inside its transition, under the write's source hold, so only the reads of
    other tasks and of Adapters/ActivationPatches need old versions pinned. A writer whose math is not traced (a
    host-batch causal update) is taken to write every field.
    """
    data, batch = example(binding, facts, ExecutionOptions())
    fields = {
        StorageWeakRef(tensor.untyped_storage()): (str(resource), index)
        for resource, scoped in batch.resources.items()
        for index, tensor in enumerate(scoped.roots[0])
    }
    reads: dict[str, set[int]] = {}
    writes: dict[str, set[int]] = {}
    log = []
    writers = {
        str(task.symbol): {str(e.resource) for e in task.effects if e.kind in WRITES}
        for task in plan.semantic_graph.tasks
    }
    for symbol, operation in binding.executables(facts, ()).items():
        task = binding.tensor_tasks.get(symbol)
        task_data = task_example(data, None if task is None else task.phase)
        scoped = operation.scoped(batch)
        with FieldTrace(fields) as body:
            output = operation.trace_body(task_data, scoped)
        with FieldTrace(fields) as update:
            if operation.optimizer is not None and output.gradients:
                operation.apply_gradients(scoped.roots, scoped.field_slots, scoped.roots, output)
        written = body.writes | update.writes
        owner = writers.get(str(symbol).removesuffix(".write"), set())
        for resource, field in body.reads:
            if resource not in owner:
                reads.setdefault(resource, set()).add(field)
        for resource, field in written:
            writes.setdefault(resource, set()).add(field)
        log.append(f"{symbol}: reads {sorted(body.reads)} writes {sorted(written)}")
    for (resource, _), view in binding.field_views.items():
        if resource in {
            str(b.resource) for b in (*binding.effects.overlay_bindings, *binding.effects.activation_bindings)
        }:
            reads.setdefault(resource, set()).add(int(view.field))
    log.append(f"model reads {sorted((r, f) for r, fs in reads.items() for f in fs)}")
    result = {}
    for entry in plan.tensor_layout_request.layouts:
        width = len(entry.layout.fields)
        written = writes.get(entry.resource) or (
            set(range(width)) if any(entry.resource in owner for owner in writers.values()) else set()
        )
        result[entry.resource] = StateFields(frozenset(reads.get(entry.resource, ())), frozenset(written), width)
    return result, log


# ---------------------------------------------------------------------------------------------------------------
# Ledger-level stubs around the real StateExecution and arena.


class PeakArena(PageBundleArena):
    """The real arena, recording after every allocation the most pages each field holds, today and field-aware.

    Field-aware: the latest version and private writes hold every field; an older version keeps only the fields
    some pin reads (`pinned`), so its other pages return to the pool at publication.
    """

    def __init__(self, plan: PlanBundle, pinned: dict[str, frozenset[int]]) -> None:
        super().__init__(plan, device=torch.device("cpu"))
        self.pinned = pinned
        self.peak = {r: [0] * len(p.tensors) for r, p in self.pools.items()}
        self.field_peak = {r: [0] * len(p.tensors) for r, p in self.pools.items()}

    def _sample(self) -> None:
        for resource, pools in self.pools.items():
            for f, capacity in enumerate(self._plans[resource].page_capacities):
                self.peak[resource][f] = max(self.peak[resource][f], capacity - pools.free_count(f))
        current: dict[str, list[set[int]]] = {r: [set() for _ in p.tensors] for r, p in self.pools.items()}
        latest = set()
        for category in ("live", "private", "held"):
            for group in self._inspect(category):
                for i, ref in enumerate(group.refs):
                    if category == "live":
                        latest.add(ref)
                    old = category == "held" and ref not in latest
                    for f, slot in enumerate(group.slots[i].tolist()):
                        if not old or f in self.pinned.get(group.resource, ()):
                            current[group.resource][f].add(slot)
        for resource, sets in current.items():
            for f, slots in enumerate(sets):
                self.field_peak[resource][f] = max(self.field_peak[resource][f], len(slots))

    def prepare_admission(self, resources, stream=None):  # type: ignore[override]
        token = super().prepare_admission(resources, stream)
        self._sample()
        return token

    def prepare_transition(self, sources, stream=None, *, fields=None, copy_fields=None):  # type: ignore[override]
        token = super().prepare_transition(sources, stream, fields=fields, copy_fields=copy_fields)
        self._sample()
        return token


class StubMath:
    """StateMath with no tensors: a step's data is its StepInput in a TensorFrame."""

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


@dataclass
class StubTask:
    """One scheduled task's page effects: a private write of every row it may write, published as declared.

    A task reading a host-service input (CausalTrainBatch, WeightedRowBatch, StateImport) runs only in steps that
    carry a host batch; every other writer writes in every step (the adversary).
    """

    implementation_id: object
    effects: tuple[tuple[str, StateEffectKind], ...]
    host_gated: bool
    written: dict[str, tuple[LayoutFieldId, ...]] | None

    def execute(self, context: StateTaskContext[TensorFrame]) -> dict:
        if self.host_gated and context.data().step.batch is None:
            return {}
        for resource, kind in self.effects:
            write = context.pending_write(resource)
            if kind is not StateEffectKind.PUBLISH and write is None:
                refs = context.current_owned_refs(context.selection.for_resource(resource).refs)
                if not len(refs):
                    continue
                fields = None if self.written is None else self.written[resource]
                write = context.prepare_owned_write(refs, None if fields is None else (fields,) * len(refs))
            if write is not None and kind in PUBLISHES:
                context.publish_owned_write(write)
        return {}


def stub_execution(plan: PlanBundle, facts: dict[str, StateFields], field_aware: bool) -> StateExecution:
    arena = PeakArena(plan, {r: f.pinned for r, f in facts.items()})
    graph = plan.semantic_graph
    written = {r: tuple(LayoutFieldId(i) for i in sorted(f.written)) for r, f in facts.items()} if field_aware else None
    tasks = []
    for physical in sorted(plan.schedule.tasks, key=lambda t: t.implementation_id):
        task = graph.tasks[physical.task_id]
        effects = tuple((str(e.resource), e.kind) for e in task.effects if e.kind in (*WRITES, *PUBLISHES))
        gated = any(isinstance(port.source, HostServiceInputSource) for port in task.inputs)
        tasks.append(StubTask(physical.implementation_id, effects, gated, written))
    sampler = next(a.ordinal for a in plan.timeline.anchors if a.symbol == "sampler.pre_draw")
    execution = StateExecution(
        arena, RequestStateDirectory(arena, graph.states), StubMath(), tuple(tasks), AnchorId(sampler)
    )
    execution.admit_instances()
    return execution


# ---------------------------------------------------------------------------------------------------------------
# The engine model: the order of state calls VllmSession makes per step, and adversarial schedules.


@dataclass(frozen=True)
class Step:
    """One engine step: requests vLLM finished or preempted before it, requests it admits, requests it runs."""

    retire: tuple[int, ...] = ()
    admit: tuple[int, ...] = ()
    run: tuple[int, ...] | None = None  # None runs every live request
    new_generate: bool = False  # the step starts a generate() call, whose host batch is staged


class Engine:
    """VllmSession's state calls: before_execute (retire, admit), prepare_input or complete_forward (start the
    step, admission pins after request.prepare), every anchor, commit; finish_requests retires everything."""

    def __init__(self, execution: StateExecution) -> None:
        self.execution = execution
        graph = execution.arena.plan.semantic_graph
        self.early = bool(graph.model_reads) or graph.trains_on_batch
        self.has_request_states = bool(execution.directory.request_states)
        self.pinned = bool(execution.directory.pinned_states)
        self.live: list[int] = []
        self.batch: object | None = None
        self.step_id = 0

    @staticmethod
    def identity(i: int) -> RequestIdentity:
        return RequestIdentity(f"r{i}", i, 0)

    def retire(self, requests) -> None:
        """Without request states only requests that pinned at admission hold state (VllmSession._retire)."""
        directory = self.execution.directory
        held = tuple(
            self.identity(i)
            for i in requests
            if self.has_request_states or directory.has_inference_pin(self.identity(i))
        )
        if held:
            self.execution.retire(held)

    def run(self, step: Step) -> None:
        execution, ids = self.execution, self.identity
        if step.new_generate:
            assert not self.live
            self.batch = object()
        if step.retire:
            self.retire(step.retire)
            self.live = [i for i in self.live if i not in step.retire]
        if step.admit and self.has_request_states:
            execution.admit(tuple(ids(i) for i in step.admit))
        self.live.extend(step.admit)
        rows = sorted(self.live if step.run is None else step.run)
        self.step_id += 1
        if not rows:
            return
        value = replace(testing.step_input(tuple(ids(i) for i in rows), step_id=self.step_id), batch=self.batch)
        overlap = execution.post_sample
        if self.early:
            context = execution.start_step(value)
            execution.advance_to(context, "request.prepare")
            pending = tuple(ids(i) for i in rows if self.pinned and not execution.directory.has_inference_pin(ids(i)))
            if pending:
                execution.admit(pending)
        else:
            context = execution.start_step(value, history=() if overlap is None else overlap.history)
        self.batch = None  # instance states consume the batch in the generate call's first step
        for anchor in execution.arena.plan.timeline.anchors:
            execution.advance_to(context, anchor.symbol)
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(context)
        execution.finish_step(context)

    def finish_generate(self) -> None:
        self.execution.release_history()
        self.retire(self.live)
        self.live = []


def burst(n: int, steps: int) -> Iterator[Step]:
    """One generate call admitting n requests at once, all decoding."""
    yield Step(admit=tuple(range(n)), new_generate=True)
    for _ in range(steps):
        yield Step()


def stagger(n: int, steps: int) -> Iterator[Step]:
    """One request admitted per step until n are live, two steps without admission, then churn with idle steps."""
    yield Step(admit=(0,), new_generate=True)
    for i in range(1, n):
        yield Step(admit=(i,))
    yield Step()
    yield Step()
    for i in range(n, n + steps):
        yield Step(retire=(i - n,), admit=(i,))
        yield Step()


def preempt(n: int, steps: int) -> Iterator[Step]:
    """All n admitted at once; every step one request is preempted and re-admitted (recompute)."""
    yield Step(admit=tuple(range(n)), new_generate=True)
    for s in range(steps):
        yield Step(retire=(s % n,), admit=(s % n,))


def structured(n: int) -> Iterator[Step | None]:
    """Each pattern twice, separated by finish_requests (None)."""
    for pattern, steps in ((burst, 6), (stagger, 4), (preempt, n + 2)):
        for _ in range(2):
            yield from pattern(n, steps)
            yield None


def fuzz(n: int, steps: int, rng: random.Random) -> Iterator[Step | None]:
    """Random retire/admit/run choices; a generate call ends whenever its requests are gone or at random."""
    live: list[int] = []
    nxt = 0
    for _ in range(steps):
        new_generate = not live
        retire = tuple(i for i in live if rng.random() < 0.3)
        live = [i for i in live if i not in retire]
        count = rng.randint(0 if live else 1, n - len(live))
        admit = tuple(range(nxt, nxt + count))
        nxt += count
        live += admit
        run = None if rng.random() < 0.6 else tuple(i for i in live if rng.random() < 0.7)
        yield Step(retire, admit, run, new_generate)
        if not live or rng.random() < 0.1:
            yield None
            live = []


def drive(plan, facts, schedule, capacity, *, field_aware: bool) -> PeakArena:
    execution = stub_execution(scalar_plan(plan, capacity), facts, field_aware)
    engine = Engine(execution)
    for step in schedule:
        engine.finish_generate() if step is None else engine.run(step)
    engine.finish_generate()
    arena = execution.arena
    assert isinstance(arena, PeakArena)
    return arena


# ---------------------------------------------------------------------------------------------------------------
# Closed-form bounds.


def proposed_bound(plan: PlanBundle, resource: str, fields: StateFields, n: int, history: int) -> list[int]:
    """Pages per field: today's per-lineage terms, the admission term only for writers that run while requests
    live (not host-batch writers, which run in a generate call's first step), and field-aware old versions."""
    graph = plan.semantic_graph
    state = next(s for s in graph.states if s.resource == resource)
    if any(resource in group.borrowed_resources for group in graph.input_groups):
        return [FEED_SLOTS] * fields.width
    writers = [t for t in graph.tasks if any(str(e.resource) == resource and e.kind in WRITES for e in t.effects)]
    if not writers:
        return [1 if state.scope is StateScope.INSTANCE else n] * fields.width
    per_step = any(not any(isinstance(p.source, HostServiceInputSource) for p in t.inputs) for t in writers)
    lineages = 1 if state.scope is StateScope.INSTANCE else n
    pins = n if state.scope is StateScope.INSTANCE and resource in graph.admission_reads and per_step else 0
    result = []
    for f in range(fields.width):
        if f not in fields.written:
            result.append(lineages)
        elif f in fields.pinned:
            result.append((2 + history) * lineages + pins)
        else:
            result.append(2 * lineages)
    return result


def mib(pages: list[int], strides: list[int]) -> float:
    return sum(p * s for p, s in zip(pages, strides, strict=True)) / 2**20


def without_gradient(pages: list[int], strides: list[int]) -> float:
    """Field-aware bytes if AdamW states dropped the gradient field, which no reader or kernel reads back.

    An AdamW layout is (parameters, gradient, step, exp_avg, exp_avg_sq, *carry): field 2 is the scalar step.
    """
    adamw = len(strides) >= 5 and strides[2] == 16 and len({strides[0], strides[1], strides[3], strides[4]}) == 1
    return mib(pages, strides) - (pages[1] * strides[1] / 2**20 if adamw else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seqs", type=int, default=128)
    parser.add_argument("--fuzz", type=int, default=300)
    parser.add_argument("--small", type=int, default=4, help="max_num_seqs for the fuzz cross-check")
    parser.add_argument("--only", default=None)
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()
    n = args.seqs
    roomy = lambda resource: 4 * n + 16  # noqa: E731
    header = (
        f"{'workload / state':34} {'today pages':>14} {'today MiB':>10} {'dry-run pages':>14} {'bound pages':>16}"
        f" {'field-aware pages':>22} {'field MiB':>10} {'-grad MiB':>10}  fuzz(n={args.small})  t"
    )
    print(header)
    for name, workload in WORKLOADS.items():
        if args.only and args.only not in name:
            continue
        facts, binding, plan = plan_for(workload, args.model, n)
        fields, log = state_fields(binding, facts, plan)
        if args.traces:
            print("\n".join(f"    {line}" for line in log))
        history = max((t.read_distance for t in plan.schedule.tasks), default=0)
        strides = {
            e.resource: [page_stride_bytes(f) for f in e.layout.fields] for e in plan.tensor_layout_request.layouts
        }
        today = {p.resource: max(p.page_capacities) for p in plan.states.plans}
        started = time.perf_counter()
        native = drive(plan, fields, structured(n), roomy, field_aware=False)
        aware = drive(plan, fields, structured(n), roomy, field_aware=True)
        elapsed = time.perf_counter() - started
        # Fuzz at a small max_num_seqs: the structured adversary must reach what random schedules reach.
        small_plan = plan_for(workload, args.model, args.small)[2]
        rng = random.Random(0)
        fuzzed: dict[str, list[int]] = {}
        small_structured = drive(small_plan, fields, structured(args.small), roomy, field_aware=True).field_peak
        for _ in range(args.fuzz):
            peak = drive(small_plan, fields, fuzz(args.small, 14, rng), roomy, field_aware=True).field_peak
            for resource, values in peak.items():
                fuzzed[resource] = [max(a, b) for a, b in zip(fuzzed.get(resource, values), values, strict=True)]
        # The proposed bound must run every structured schedule on today's ledger without waiting for a page.
        bound = {r: proposed_bound(plan, r, fields[r], n, history) for r in today}
        drive(plan, fields, structured(n), lambda r: max(bound[r]), field_aware=False)
        for resource in today:
            pages = native.peak[resource]
            field_pages = aware.field_peak[resource]
            small_bound = proposed_bound(small_plan, resource, fields[resource], args.small, history)
            flag = (
                ""
                if all(
                    f <= min(b, s)
                    for f, b, s in zip(fuzzed[resource], small_bound, small_structured[resource], strict=True)
                )
                else " !"
            )
            print(
                f"{name + ' / ' + resource:34} {today[resource]:>14} {today[resource] * sum(strides[resource]) / 2**20:>10.1f}"
                f" {max(pages):>14} {str(bound[resource]):>16} {str(field_pages):>22}"
                f" {mib(field_pages, strides[resource]):>10.1f} {without_gradient(field_pages, strides[resource]):>10.1f}"
                f"  {fuzzed[resource]}/{small_bound}{flag}  {elapsed:.2f}s"
            )


if __name__ == "__main__":
    main()
