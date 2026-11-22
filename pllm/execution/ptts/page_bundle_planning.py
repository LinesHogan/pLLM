"""Cold physical capacity facts for the shared CPU/CUDA PageBundle arena."""

from dataclasses import dataclass, replace
from functools import partial
from math import prod
from pathlib import Path

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import (
    SchedulerTimeline,
    SemanticGraph,
    StateEffectKind,
    StateResource,
    StateScope,
)
from pllm.contracts.state_feed import FEED_SLOTS
from pllm.contracts.state_representation import (
    StateDeviceKind,
    StateProviderPlan,
    StateRepresentationRequest,
)
from pllm.contracts.tensor_layout import (
    TensorFieldSpec,
    TensorLayoutBinding,
    TensorLayoutDecision,
    TensorLayoutRequest,
    TensorLayoutSpec,
)
from pllm.execution.scheduler import compile_state_schedule, intern_tensor_layouts


def compile_page_bundle_plan(
    declaration: tuple[SemanticGraph, TensorLayoutRequest],
    timeline: SchedulerTimeline,
    providers: tuple["PageBundleProviderFacts", ...],
    max_admitted_lineages: int,
    state_page_budget_bytes: int,
) -> PlanBundle:
    """Compile original declarations once; all choices remain Rust-owned.

    How many old versions the post-sample overlap retains is a schedule fact (its largest read
    distance) that does not depend on reservations or budgets. A first plan without budgets finds it;
    the second plan reserves it and checks the budgets.
    """
    graph, layouts = declaration
    layout_decision = intern_tensor_layouts(layouts)
    request = partial(_state_request, graph, layouts, layout_decision, providers, max_admitted_lineages)
    _, unbudgeted = compile_state_schedule(graph, replace(timeline, execution_peak_budget_bytes=None), request(0, 0))
    history = max((task.read_distance for task in unbudgeted.tasks), default=0)
    states, schedule = compile_state_schedule(graph, timeline, request(history, state_page_budget_bytes))
    return PlanBundle(graph, timeline, schedule, states, layouts, layout_decision)


def _state_request(
    graph: SemanticGraph,
    layouts: TensorLayoutRequest,
    layout_decision: TensorLayoutDecision,
    providers: tuple["PageBundleProviderFacts", ...],
    max_admitted_lineages: int,
    history: int,
    state_page_budget_bytes: int,
) -> StateRepresentationRequest:
    """Every provider's reservation for every state, after checking host tables fit host memory."""
    candidates = tuple(
        provider.materialize_plan(
            resource=entry.resource,
            layout=entry.layout,
            binding=binding,
            max_admitted_lineages=max_admitted_lineages,
            graph=graph,
            history=history,
        )
        for entry, binding in zip(layouts.layouts, layout_decision.bindings, strict=True)
        for provider in providers
    )
    host_bytes = sum(
        max(candidate.host_bytes for candidate in candidates if candidate.resource == entry.resource)
        for entry in layouts.layouts
    )
    if host_bytes:
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        if host_bytes > available:
            raise ValueError(
                'host tables do not fit available host memory; reduce the tables or declare placement="device"'
            )
    return StateRepresentationRequest(
        state_page_budget_bytes, tuple(binding.resource for binding in layout_decision.bindings), candidates
    )


def page_stride_bytes(field: TensorFieldSpec) -> int:
    size = prod(field.shape) * field.dtype.itemsize
    return (size + field.alignment_bytes - 1) // field.alignment_bytes * field.alignment_bytes


def version_capacity(lineages: int, history: int, *, written: bool) -> int:
    """Pages per field that the admitted lineages of one state can hold at the same time.

    Each admitted lineage holds:
    - 1, the version its step reads, pinned until the step commits;
    - `history`, the older versions the post-sample overlap keeps for lagged readers. Its newest pin is
      the latest published version, which therefore needs no root of its own. A retired lineage still
      pinned there shares its seat with the lineage admitted next, which has no history yet;
    - 1 if a task writes the state, the private version a transition fills before it publishes.

    Released pages are reused behind GPU completion fences, so in-flight GPU reads reserve nothing.
    """
    return (1 + history + written) * lineages


def written_while_requests_run(graph: SemanticGraph, resource: StateResource) -> bool:
    """Whether a task may write `resource` while requests pin an earlier instance version.

    Both step-driven updates and mid-flight control-plane updates (`import_state`, `train_batch`) can publish
    a new version while in-flight requests hold an older instance state across decode steps.
    """
    return any(
        effect.resource == resource and effect.kind not in (StateEffectKind.READ, StateEffectKind.PUBLISH)
        for task in graph.tasks
        for effect in task.effects
    )


@dataclass(frozen=True, slots=True)
class PageBundleProviderFacts:
    """Where state pages live. Test harnesses that hold extra pins set aside `minimum_pages` per state."""

    provider_id: str
    device_kind: StateDeviceKind = StateDeviceKind.CPU
    minimum_pages: int = 0

    def materialize_plan(
        self,
        *,
        resource: str,
        layout: TensorLayoutSpec,
        binding: TensorLayoutBinding,
        max_admitted_lineages: int,
        graph: SemanticGraph,
        history: int,
    ) -> StateProviderPlan:
        """Reserve every version one state can hold at once, so a transition never waits for a page.

        Older versions are held only by pins, which keep the fields readers read; fields only the state's writer
        reads (AdamW's step and moments) need the latest version and the private one per lineage.
        """
        state = next(state for state in graph.states if state.resource == resource)
        instance = state.scope is StateScope.INSTANCE
        lineages = 1 if instance else max_admitted_lineages
        if any(state.resource in group.borrowed_resources for group in graph.input_groups):
            capacity = FEED_SLOTS
        elif instance and state.immutable:
            capacity = 1
        elif instance:
            # One lineage, plus the version each admitted request pins when an Adapter or ActivationPatch reads it
            # (requests keep their admission version) and a task writes the state while they run. Admitting one
            # request per written step pins distinct versions; after 1 + history more writes all of them are older
            # than the lineage's own versions. A state written only from host batches changes once per generate
            # call: its requests pin the version before or after that write, both within the lineage's own.
            pinned = state.resource in graph.admission_reads and written_while_requests_run(graph, state.resource)
            pins = max_admitted_lineages if pinned else 0
            capacity = version_capacity(1, history, written=True) + pins
        else:
            capacity = version_capacity(max_admitted_lineages, history, written=not state.immutable)
        writer_capacity = version_capacity(lineages, 0, written=True)
        capacities = tuple(
            max(self.minimum_pages, writer_capacity if field in state.writer_fields else capacity)
            for field in range(len(layout.fields))
        )
        reserved_bytes = sum(
            pages * page_stride_bytes(field) + max(field.alignment_bytes - field.dtype.itemsize, 0)
            for pages, field in zip(capacities, layout.fields, strict=True)
        )
        return StateProviderPlan(
            resource=resource,
            provider_id=self.provider_id,
            device_kind=self.device_kind,
            layout_id=binding.layout_id,
            max_admitted_lineages=lineages,
            page_capacities=capacities,
            reserved_bytes=0 if state.placement == "host" else reserved_bytes,
            host_bytes=reserved_bytes if state.placement == "host" else 0,
            capture_stable=self.device_kind is StateDeviceKind.CUDA,
            capabilities=tuple(
                sorted(
                    (
                        f"device/{self.device_kind.value}@1",
                        "optimizer/adamw@1",
                        "state/exact_versions@1",
                        "transition/full_root@1",
                        *(("state/fenced_reclaim@1",) if self.device_kind is StateDeviceKind.CUDA else ()),
                    )
                )
            ),
        )
