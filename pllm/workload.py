"""pllm.Workload, the base class every workload subclasses, and reflection: reading a workload's declared states,
decorated methods and input markers once into a WorkloadContext, which pllm.execution.workload_lowering lowers into
a TensorDeclaration. Workload also compiles itself against a model backend (specialize, plan_backend,
bind_backend) into a BackendDeployment. Used by pllm (make_llm, explain), the vLLM backend and worker, pllm.host
and pllm.testing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from graphlib import CycleError, TopologicalSorter
from inspect import Parameter as SignatureParameter, signature
from itertools import chain, pairwise
from types import FunctionType
from typing import Any, ClassVar, TypeVar

import torch

from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import AttachmentGraphMode, BackendAttachment
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.semantic_graph import AnchorId, SchedulerTimeline, SemanticGraph, TaskId
from pllm.contracts.tensor_policy import TensorPolicy
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.backend_attachment import (
    attachment_buffer_bytes,
    bind_backend_attachment,
    derive_attachment_resources,
)
from pllm.execution.causal_batch import CausalBinder
from pllm.execution.compile_facts import BackendCompileFacts, BackendRequirements, prefix_cache_reason
from pllm.execution.declaration import When
from pllm.execution.host_runtime import HostConfig
from pllm.execution.method_dependencies import check_dependencies
from pllm.execution.method_effects import (
    CustomAttention,
    CustomResidual,
    Effect,
    Gradient,
    TrainInForward,
    Write,
)
from pllm.execution.method_inputs import (
    Attention,
    InputMarker,
    InputTokens,
    Parameter,
    Result,
    StateIndex,
    StateVersion,
    Value,
)
from pllm.execution.overlay_task import ChunkUpdate
from pllm.execution.position_state import PositionState
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import compile_page_bundle_plan, version_capacity
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.state_init import Fed, Imported
from pllm.execution.task_memory import custom_forward_capture_bytes
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.trace import TraceEvent
from pllm.execution.workload_lowering import (
    Operation,
    WorkloadContext,
    bind_host_declaration,
    finish_model_declaration,
    host_declaration,
    lower_workload,
    model_declaration,
    reserve_resources,
)
from pllm.usage import ActivationPatch, Adapter, InstanceState, Phase, RequestState, SharedLayers

C = TypeVar("C")


@dataclass(frozen=True, slots=True)
class BackendDeployment:
    """A compiled workload: its bound execution and the backend attachment it was planned with."""

    execution: StateExecution
    attachment: BackendAttachment

    def __post_init__(self) -> None:
        assert self.attachment.plan is self.execution.arena.plan


class Workload:
    """Python methods and effects compiled once, with independent sub-workloads in dataclass fields.

    Sampler additions follow member declaration order (parent methods, then fields recursively), before
    vLLM filters and at most one guide. Set required_publish_lag when the algorithm requires a fixed
    value; every component must agree with the engine's ExecutionOptions.publish_lag.
    """

    required_publish_lag: ClassVar[int | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject members that would silently replace pLLM's own Workload methods; on_export and
        required_publish_lag are the ones an author overrides."""
        super().__init_subclass__(**kwargs)
        framework = {name for name in vars(Workload) if not name.startswith("_")}
        reserved = framework - {"on_export", "required_publish_lag"}
        for name in (*vars(cls), *vars(cls).get("__annotations__", {})):
            if name in reserved:
                raise TypeError(
                    f"{cls.__name__}.{name}: pLLM's Workload uses the name {name!r}; rename it "
                    f"(reserved: {', '.join(sorted(reserved))})"
                )

    def specialize(
        self,
        facts: BackendCompileFacts,
        options: ExecutionOptions,
        *,
        causal: CausalBinder | None = None,
        prefix_caching: bool = False,
    ) -> tuple[BackendCompileFacts, TensorDeclaration]:
        """Measure and fund the declaration on this backend, then declare against the funded facts as workers do;
        planning and binding reuse both."""
        for path, component in component_workloads(self):
            required = component.required_publish_lag
            if required is not None and required != options.publish_lag:
                raise ValueError(
                    f"{path or type(component).__name__} requires publish_lag={required}; the engine uses publish_lag={options.publish_lag}; all components must use the engine-wide lag"
                )
        facts = _apply_options(facts, options)
        measured = self.build(facts, causal=causal)
        if prefix_caching:
            BackendRequirements.from_model_reads(
                measured.graph,
                measured.effects,
                external_states={*measured.imports, *(feed.resource for feed in measured.feeds)},
                custom_attention=bool(measured.attention.methods),
            ).check_prefix_cache()
        timeline = _fund_attachment(measured, facts, reserve_resources(measured, facts, options), options)
        facts = replace(facts, timeline=timeline)
        declaration = self.funded_declaration(facts, causal=causal)
        declaration.gather_bytes = measured.gather_bytes
        return facts, declaration

    def funded_declaration(
        self, facts: BackendCompileFacts, *, causal: CausalBinder | None = None
    ) -> TensorDeclaration:
        """The declaration on funded facts: tensor tasks funded resident memory run as captured CUDA graphs. The
        driver and every worker process declare here, so their graphs agree."""
        declaration = self.build(facts, causal=causal)
        declaration.select_capture(facts.timeline.task_resources)
        return declaration

    def _export_handler(
        self, declaration: TensorDeclaration, graph: SemanticGraph
    ) -> Callable[[TraceEvent], None] | None:
        """Bind export ownership by task identity; author labels may contain dots."""
        if not declaration.has_exports:
            return None
        owners = dict(component_workloads(self))
        routes = {
            TaskId(index): (path, owners[path])
            for index, task in enumerate(graph.tasks)
            if task.symbol in declaration.tensor_tasks and declaration.tensor_tasks[task.symbol].export is not None
            for path in (str(task.symbol).rpartition(".")[0],)
        }
        return partial(self._route_export, routes)

    @staticmethod
    def _route_export(routes: Mapping[TaskId, tuple[str, Workload]], event: TraceEvent) -> None:
        """Deliver each component's export with its local label, even when labels overlap."""
        assert event.task_id is not None and event.tensor_name is not None
        path, owner = routes[event.task_id]
        owner.on_export(
            replace(event, tensor_name=event.tensor_name.removeprefix(path + ".") if path else event.tensor_name)
        )

    def on_export(self, event: TraceEvent) -> None:
        """Consume ordered CPU snapshots in the execution process; never reenter execution.

        The default retains snapshot history only. Own any retained tensors and I/O resources.
        Tensor-parallel execution delivers exports only from the output rank (rank 0).
        """

    def build(self, facts: BackendCompileFacts, *, causal: CausalBinder | None = None) -> TensorDeclaration:
        """Build the graph and tensor tasks against the backend's model dimensions."""
        declaration = model_declaration(type(self).__qualname__, facts, causal)
        lower_workload(read_workload(self), declaration, facts.timeline.capabilities)
        finish_model_declaration(declaration, facts)
        return declaration

    def backend_requirements(self) -> BackendRequirements:
        """Read model dependencies from declarations and markers before dimensions or tensor readers exist."""
        from pllm.execution.method_inputs import Batch

        context = read_workload(self)
        definitions = context.state_definitions
        inputs: dict[str, tuple[InputMarker, ...]] = {}
        trains_on_batch = False
        for operation in context.operations:
            if operation.effect.reads_taps:
                with operation.context.declaring(operation.name):
                    sources = operation.sources()
                if any(isinstance(marker, Batch) for markers in sources.values() for marker in markers.values()):
                    trains_on_batch = True
                inputs.update((name, tuple(arguments.values())) for name, arguments in sources.items())
        declared = BackendRequirements(
            shared_layer_groups=tuple(group for shared in context.shared_layers.values() for group in shared.groups),
            parameter_targets=tuple(
                target for delta in context.adapters.values() if delta.route is None for target in delta.targets
            ),
            routed_targets=tuple(
                (name, delta.route.over)
                for delta in context.adapters.values()
                if delta.route is not None
                for name in delta.target_names
            ),
            activation_layers=tuple(
                layer
                for patch in context.activations.values()
                for layer in (patch.route.over if patch.route is not None else patch.layers)
            ),
            position_states=context.position_states(),
            custom_forward=bool(context.custom_forwards),
            trains_on_batch=trains_on_batch,
        )
        return BackendRequirements.combine(
            (
                declared,
                _model_site_requirements(context, inputs),
                *(state.init.requirements(state) for state in definitions.values()),
                *(operation.effect.requirements() for operation in context.operations),
                *(source.requirements() for name in sorted(inputs) for source in inputs[name]),
            )
        )

    def bind_math(
        self,
        facts: BackendCompileFacts,
        plan: PlanBundle,
        declaration: TensorDeclaration,
        *,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> MathBinding:
        """Bind the planned declaration's tensor tasks to the implementations the plan selected."""
        declaration.attention.bind(facts.device)
        operations = declaration.executables(facts, plan.timeline.task_resources, warmup=True, map_host=map_host)
        declaration.require_resident_side_tasks(plan, operations)
        return MathBinding(
            declaration.function_math(facts.device, facts.initialization_seed),
            declaration.bind_all(plan, model_probe=facts.probe, operations=operations),
            feeds=declaration.feeds,
            on_export=self._export_handler(declaration, plan.semantic_graph),
        )

    def plan_backend(self, facts: BackendCompileFacts, declaration: TensorDeclaration) -> BackendAttachment:
        """Plan the specialized declaration; binding reuses the same declaration."""
        graph, layouts, attachment = declaration.finish()
        plan = compile_page_bundle_plan(
            (graph, layouts),
            facts.timeline,
            facts.providers,
            facts.max_admitted_lineages,
            facts.state_page_budget_bytes,
        )
        return bind_backend_attachment(
            plan, facts.probe, replace(attachment, guidance_capacity=facts.guidance_capacity)
        )

    def bind_backend(
        self,
        facts: BackendCompileFacts,
        attachment: BackendAttachment,
        declaration: TensorDeclaration,
        *,
        is_output_rank: bool = True,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> BackendDeployment:
        plan = attachment.plan
        math = self.bind_math(facts, plan, declaration, map_host=map_host)
        if not is_output_rank:
            math = replace(math, on_export=None)
        arena = PageBundleArena(
            plan, device=facts.device, owned_resources=math.local_resources(plan), map_host=map_host
        )
        (sampler,) = (
            AnchorId(anchor.ordinal) for anchor in plan.timeline.anchors if anchor.symbol == "sampler.pre_draw"
        )
        return BackendDeployment(math.bind(arena, sampler), attachment)

    def compile_backend(
        self,
        facts: BackendCompileFacts,
        options: ExecutionOptions = ExecutionOptions(),
        *,
        causal: CausalBinder | None = None,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> BackendDeployment:
        """Specialize, plan and bind: the whole compile on probed backend facts."""
        facts, declaration = self.specialize(facts, options, causal=causal)
        return self.bind_backend(facts, self.plan_backend(facts, declaration), declaration, map_host=map_host)

    def compile_host(self, config: HostConfig) -> StateExecution:
        """Build, plan and bind the graph without a model (see pllm.host.make_host); training reads host batches."""
        declaration = host_declaration(type(self).__qualname__, config)
        lower_workload(read_workload(self), declaration)
        return bind_host_declaration(declaration, config, self._export_handler(declaration, declaration.graph))


def _model_site_requirements(
    context: WorkloadContext, inputs: Mapping[str, tuple[InputMarker, ...]]
) -> BackendRequirements:
    """Read the model-site signatures before layer counts and storage layouts are available."""
    reads = {item.state for item in (*context.adapters.values(), *context.activations.values())}
    methods = [name for name, _, _ in context.custom_forwards]
    writers = {op.effect.state for op in context.operations if op.effect.training or isinstance(op.effect, Write)}
    value_site = False
    custom_attention = False
    for op in context.operations:
        effect = op.effect
        if isinstance(effect, CustomResidual):
            methods.append(op.name)
            if effect.read:
                methods.append(effect.read)
            value_site |= effect.site == "value"
            if effect.state:
                reads.add(effect.state)
        elif isinstance(effect, CustomAttention):
            custom_attention = True
            methods.append(op.name)
    markers = tuple(
        arg.default.bind(context.component(name))
        for name in methods
        for arg in signature(context.method(name)).parameters.values()
        if isinstance(arg.default, InputMarker)
    )
    reads.update(_states_read_by_sites(markers, inputs))
    attention_facts = any(isinstance(marker, Attention) for marker in chain(markers, *inputs.values()))
    reason = prefix_cache_reason(
        (
            (name, state.scope, isinstance(state.init, (Fed, Imported)) or name in writers)
            for name, state in context.state_definitions.items()
            if name in reads
        ),
        custom_attention=custom_attention or attention_facts,
        in_forward=any(isinstance(op.effect, TrainInForward) for op in context.operations),
        last_tokens=any(patch.last_tokens > 0 for patch in context.activations.values()),
    )
    return BackendRequirements(
        custom_value_site=value_site or any(isinstance(marker, Value) for marker in markers),
        custom_attention_facts=attention_facts,
        prefix_cache_unsafe_reason=reason,
    )


def _states_read_by_sites(markers: tuple[InputMarker, ...], inputs: Mapping[str, tuple[InputMarker, ...]]) -> set[str]:
    """Follow pre-forward result dependencies to their state inputs."""
    reads: set[str] = set()
    pending = list(markers)
    visited: set[str] = set()
    while pending:
        source = pending.pop()
        if isinstance(source, (Parameter, StateVersion, StateIndex)):
            reads.add(source.state)
        elif isinstance(source, Result) and source.operation not in visited:
            visited.add(source.operation)
            pending.extend(inputs.get(source.operation, ()))
    return reads


def _apply_options(facts: BackendCompileFacts, options: ExecutionOptions) -> BackendCompileFacts:
    """The execution options the declaration and plan depend on: capabilities, budgets, tensor policy, visibility."""
    if options.state_page_budget_bytes > facts.state_page_budget_bytes:
        raise ValueError(
            f"ExecutionOptions.state_page_budget_bytes={options.state_page_budget_bytes} exceeds the "
            f"{facts.state_page_budget_bytes} bytes free on the device; lower it, or lower gpu_memory_utilization "
            "so vLLM leaves more memory free"
        )
    capabilities = set(facts.timeline.capabilities)
    if "compute/cuda@1" in capabilities:
        capabilities.add("compute/cuda_graph@1")
    timeline = replace(
        facts.timeline,
        capabilities=tuple(sorted(capabilities)),
        execution_peak_budget_bytes=min(
            (
                n
                for n in (facts.timeline.execution_peak_budget_bytes, options.execution_peak_budget_bytes)
                if n is not None
            ),
            default=None,
        ),
        tensor_policy=TensorPolicy(options.allow_numerical_transforms, options.tuning),
        visibility=VisibilitySource.post_sampler(options.publish_lag),
    )
    return replace(
        facts,
        timeline=timeline,
        state_page_budget_bytes=options.state_page_budget_bytes or facts.state_page_budget_bytes,
        guidance_capacity=options.guidance_capacity if facts.guidance_rejection is None else None,
    )


def _decode_tail_buffer_bytes(
    declaration: TensorDeclaration, facts: BackendCompileFacts, graph: SemanticGraph, options: ExecutionOptions
) -> int:
    """Fund DecodeOverlay's per-page dirty bounds and per-lane generation staging tables."""
    math, probe = declaration.overlay_math, facts.probe
    if (
        math is None
        or not isinstance(math.chunk, ChunkUpdate)
        or not graph.updates_in_forward
        or math.custom_forward
        or HiddenPhase.DECODE not in math.phases
        or probe.graph_mode is not AttachmentGraphMode.CUDA_GRAPH
    ):
        return 0
    pages = max(
        version_capacity(facts.max_admitted_lineages, options.publish_lag, written=True),
        max((provider.minimum_pages for provider in facts.providers), default=0),
    )
    requests = probe.max_packed_rows if probe.causal_domain is None else probe.causal_domain.requests
    return len(math.targets) * (len(math.chunk.arguments) - 1) * (16 * pages + 24 * requests)


def _fund_attachment(
    declaration: TensorDeclaration, facts: BackendCompileFacts, timeline: SchedulerTimeline, options: ExecutionOptions
) -> SchedulerTimeline:
    """Reserve the attachment's effect buffers and fund the custom-forward task."""
    graph, layouts, attachment = declaration.finish()
    probe = facts.probe
    buffers = attachment_buffer_bytes(probe, graph, layouts, attachment) + _decode_tail_buffer_bytes(
        declaration, facts, graph, options
    )
    if buffers:
        timeline = replace(
            timeline,
            reserved_bytes=tuple(
                value + buffers for value in (timeline.reserved_bytes or (0,) * len(timeline.anchors))
            ),
        )
    capture = options.cuda_graph_pool_budget_bytes or custom_forward_capture_bytes(declaration, probe)
    return derive_attachment_resources(probe, graph, timeline, cuda_graph_pool_budget_bytes=capture)


def component_workloads(workload: Workload) -> tuple[tuple[str, Workload], ...]:
    """This workload and its nested dataclass fields, in field declaration order."""
    children = (
        tuple((item.name, object.__getattribute__(workload, item.name)) for item in fields(workload))
        if is_dataclass(workload)
        else ()
    )
    return (
        ("", workload),
        *(
            (f"{name}.{path}" if path else name, owner)
            for name, child in children
            if isinstance(child, Workload)
            for path, owner in component_workloads(child)
        ),
    )


def read_workload(workload: Workload) -> WorkloadContext:
    """Flatten independent field components into one declaration and one conflict domain."""
    contexts = tuple(_read_component(owner, path) for path, owner in component_workloads(workload))
    root = contexts[0]
    root.components = contexts
    root.state_definitions = {key: value for context in contexts for key, value in context.state_definitions.items()}
    root.adapters = {key: value for context in contexts for key, value in context.adapters.items()}
    root.activations = {key: value for context in contexts for key, value in context.activations.items()}
    root.shared_layers = {key: value for context in contexts for key, value in context.shared_layers.items()}
    root.position_definitions = {
        key: value for context in contexts for key, value in context.position_definitions.items()
    }
    roles: dict[str, str] = {}
    for context in contexts:
        for name, tokens, _ in context.custom_forwards:
            for role in ("prefill", "decode"):
                if tokens not in (role, "all"):
                    continue
                if role in roles:
                    raise ValueError(f"{roles[role]} and {name}: @pllm.custom_forward tokens overlap: {role}")
                roles[role] = name
    root.operations = [operation for context in contexts for operation in context.operations]
    root.custom_forwards = [forward for context in contexts for forward in context.custom_forwards]
    root.in_forward_states = {name for context in contexts for name in context.in_forward_states}
    for context in contexts:
        context.components = contexts
        context.state_definitions = root.state_definitions
        context.adapters = root.adapters
        context.activations = root.activations
        context.shared_layers = root.shared_layers
        context.position_definitions = root.position_definitions
        context.operations = root.operations
        context.custom_forwards = root.custom_forwards
        context.states = root.states
        context.parameters = root.parameters
        context.writes = root.writes
        context.trainers = root.trainers
        context.predictions = root.predictions
        context.in_forward_states = root.in_forward_states
    return root


def _read_component(workload: Workload, prefix: str) -> WorkloadContext:
    """Copy the workload, read its operations and check its methods and config before evaluating state properties."""
    owner = deepcopy(workload)
    values, members = _values(owner), _members(owner)
    _check_base_collisions(owner)
    context = WorkloadContext(owner, prefix=prefix)
    for name, member in members.items():
        with context.declaring(name):
            _read_operation(context, name, member)
    _declare_result_helpers(context)
    _reject_result_cycles(context)
    for name in _helpers(owner, values, members):
        with context.declaring(name):
            check_dependencies(context.method(name))
    check_dependencies({key: value for key, value in values.items() if not isinstance(value, Workload)})
    for name, member in members.items():
        if _property(member):
            values[name] = object.__getattribute__(owner, name)
    context.state_definitions = _of_type(values, InstanceState)
    context.adapters = {key: value.delta for key, value in _of_type(values, Adapter).items()}
    context.activations = _of_type(values, ActivationPatch)
    context.shared_layers = _of_type(values, SharedLayers)
    context.position_definitions = _of_type(values, PositionState)
    _namespace(context)
    return context


def _read_operation(context: WorkloadContext, name: str, member: object) -> None:
    """Read decorator attributes once; marker values never drive runtime reflection."""
    if (function := _typed(member, FunctionType)) is None:
        return
    markers = vars(function)
    if (tokens := markers.get("__pllm_tokens__")) is not None:
        if tuple(signature(function).parameters)[:2] != ("self", "layers") or any(
            markers.get(f"__pllm_{marker}__") is not None for marker in ("phase", "effect", "gradient", "rows")
        ):
            raise ValueError(
                "@pllm.custom_forward methods take (self, layers, marker inputs...) and no other pLLM decorator; use layers.save instead of @write"
            )
        overlap = {
            role
            for _, selected, _ in context.custom_forwards
            for role in ("prefill", "decode")
            if selected in (role, "all") and tokens in (role, "all")
        }
        if overlap:
            raise ValueError(
                f"{context.qualify(name)}: @pllm.custom_forward tokens overlap: {', '.join(sorted(overlap))}"
            )
        context.custom_forwards.append((name, tokens, markers["__pllm_over__"]))
        return
    phase, marked, gradient, rows = (
        None if (on := markers.get("__pllm_phase__")) is None else _typed(on(context.owner), Phase),
        None if (build := markers.get("__pllm_effect__")) is None else _typed(build(context.owner), Effect),
        _typed(markers.get("__pllm_gradient__"), Gradient),
        _typed(markers.get("__pllm_rows__", ""), str),
    )
    if phase is None and marked is None:
        return
    effect = marked or Effect()
    if (trained := _typed(effect, TrainInForward)) is not None:
        context.in_forward_states.add(trained.state)
    if (residual := _typed(effect, CustomResidual)) is not None and residual.state:
        context.in_forward_states.add(residual.state)
    when = phase.value if phase is not None else effect.phase or When.AFTER_SAMPLE
    assert rows is not None
    context.operations.append(Operation(context, name, when, effect, gradient, rows, phase is not None))


def _declare_result_helpers(context: WorkloadContext) -> None:
    """Declare implicit result producers, giving model-site consumers the pre-forward row domain."""
    pending = [(op.function, op.phase) for op in context.operations if not isinstance(op.effect, CustomResidual)]
    pending.extend(
        (op.function, When.BEFORE_FORWARD) for op in context.operations if isinstance(op.effect, CustomResidual)
    )
    pending.extend(
        (context.method(op.effect.read), When.BEFORE_FORWARD)
        for op in context.operations
        if isinstance(op.effect, CustomResidual) and op.effect.read
    )
    declared = {op.name for op in context.operations}
    while pending:
        function, phase = pending.pop()
        for parameter in signature(function).parameters.values():
            source = parameter.default
            if not isinstance(source, Result) or source.operation in declared:
                continue
            helper = context.method(source.operation)
            arguments = tuple(parameter.default for parameter in signature(helper).parameters.values())
            helper_phase = (
                When.AUTOMATIC
                if phase is not When.BEFORE_FORWARD
                and any(isinstance(arg, InputTokens) for arg in arguments)
                and all(isinstance(arg, (InputTokens, Parameter)) for arg in arguments)
                else phase
            )
            gradient = _typed(vars(helper).get("__pllm_gradient__"), Gradient)
            context.operations.append(Operation(context, source.operation, helper_phase, Effect(), gradient, ""))
            declared.add(source.operation)
            pending.append((helper, phase))


def _reject_result_cycles(context: WorkloadContext) -> None:
    """A step runs each method once, so results must flow one way; name the methods that read each other."""
    reads = {
        operation.name: [
            parameter.default.operation
            for parameter in signature(operation.function).parameters.values()
            if isinstance(parameter.default, Result)
        ]
        for operation in context.operations
    }
    try:
        TopologicalSorter(reads).prepare()
    except CycleError as error:
        cycle = error.args[1][::-1]  # graphlib lists each producer before its reader
        chain = " and ".join(
            f"{context.qualify(reader)} reads pllm.result({producer!r})" for reader, producer in pairwise(cycle)
        )
        raise ValueError(f"{chain}; results can only flow one way within a step, remove one of the reads") from None


def _namespace(context: WorkloadContext) -> None:
    """Qualify emitted names while retaining each method's bound owner and local configuration."""
    context.state_definitions = {context.qualify(key): value for key, value in context.state_definitions.items()}
    context.adapters = {
        context.qualify(key): replace(
            value,
            state=context.qualify(value.state),
            route=None if value.route is None else replace(value.route, resource=context.qualify(value.route.resource)),
        )
        for key, value in context.adapters.items()
    }
    context.activations = {
        context.qualify(key): replace(
            value,
            state=context.qualify(value.state),
            route=None if value.route is None else replace(value.route, resource=context.qualify(value.route.resource)),
        )
        for key, value in context.activations.items()
    }
    context.shared_layers = {context.qualify(key): value for key, value in context.shared_layers.items()}
    context.position_definitions = {context.qualify(key): value for key, value in context.position_definitions.items()}
    context.in_forward_states = {context.qualify(key) for key in context.in_forward_states}
    context.custom_forwards = [(context.qualify(key), tokens, over) for key, tokens, over in context.custom_forwards]
    for operation in context.operations:
        operation.name = context.qualify(operation.name)
        operation.rows = context.qualify(operation.rows) if operation.rows else ""
        effect = operation.effect
        if is_dataclass(effect):
            references = {
                item.name: context.qualify(value)
                for item in fields(effect)
                if item.name in ("state", "update_rows", "reset", "read") and (value := vars(effect)[item.name])
            }
            operation.effect = replace(effect, **references)
        if operation.gradient is not None:
            gradient = operation.gradient
            operation.gradient = replace(
                gradient,
                input_rows=context.qualify(gradient.input_rows) if gradient.input_rows else "",
                parameter_rows=context.qualify(gradient.parameter_rows) if gradient.parameter_rows else "",
            )


def _members(owner: object) -> dict[str, object]:
    """Every class attribute of the owner's MRO, base classes first and subclasses overriding."""
    return {name: item for cls in reversed(type(owner).__mro__) for name, item in vars(cls).items()}


def _values(owner: object) -> dict[str, object]:
    """Authoring values: scalar and declaration class attributes plus instance and dataclass fields."""
    markers = (InstanceState, ActivationPatch, Adapter, PositionState, SharedLayers, Workload)
    result: dict[str, object] = {
        name: member
        for name, member in _members(owner).items()
        if not name.startswith("__") and isinstance(member, (str, int, float, bool, tuple, *markers))
    }
    result.update(vars(owner))
    if is_dataclass(owner) and not isinstance(owner, type):
        result.update((item.name, getattr(owner, item.name)) for item in fields(owner))
    return result


def _of_type(values: dict[str, object], expected: type[C]) -> dict[str, C]:
    """The values of one declared type, in declaration order."""
    return {key: typed for key, value in values.items() if (typed := _typed(value, expected)) is not None}


def _typed(value: object, expected: type[C]) -> C | None:
    """Narrow an untrusted boundary marker to its declared type."""
    return value if isinstance(value, expected) else None


def _return_type(getter: FunctionType) -> type | None:
    """The class the getter is annotated to return. A string annotation (`from __future__ import annotations`) is
    looked up as a dotted name in the getter's globals, never evaluated."""
    annotation = signature(getter, eval_str=False).return_annotation
    if type(annotation) is str:
        head, *path = annotation.split(".")
        annotation = getter.__globals__.get(head)
        for name in path:
            annotation = getattr(annotation, name, None)
    return _typed(annotation, type)


def _property(member: object) -> FunctionType | None:
    """Find declaration getters without evaluating arbitrary owner properties."""
    getter = _typed(member.fget, FunctionType) if isinstance(member, property) else None
    if getter is not None and signature(getter).return_annotation is SignatureParameter.empty:
        raise TypeError(
            f"property {getter.__name__!r} needs a return annotation; annotate state declarations with "
            "-> pllm.RequestState or -> pllm.InstanceState, and other properties with their return type"
        )
    declared = (InstanceState, RequestState, ActivationPatch, Adapter, PositionState, SharedLayers)
    return getter if getter is not None and _return_type(getter) in declared else None


def _helpers(owner: Workload, values: dict[str, object], members: dict[str, object]) -> dict[str, FunctionType]:
    """The author's methods in MRO order, excluding framework methods and value bindings."""
    classes = type(owner).__mro__[: type(owner).__mro__.index(Workload)]
    names = dict.fromkeys(name for cls in classes for name in vars(cls))
    return {
        name: member
        for name in names
        if not name.startswith("__")
        and name not in values
        and not any(name in vars(base) for base in Workload.__mro__)
        and (member := _typed(members[name], FunctionType)) is not None
    }


def _own_fields(base: type) -> set[str]:
    """Dataclass fields introduced by this base, excluding inherited declarations."""
    if not is_dataclass(base):
        return set()
    return {item.name for item in fields(base) if item.name in vars(base).get("__annotations__", {})}


def _check_base_collisions(owner: Workload) -> None:
    """Reject unrelated declaration owners whose common name Python would silently discard."""
    declarations: dict[str, list[type]] = {}
    for base in type(owner).__mro__:
        if base in (Workload, object):
            continue
        own_fields = _own_fields(base)
        for name in dict.fromkeys((*vars(base), *sorted(own_fields))):
            value = vars(base).get(name)
            if (
                name in own_fields
                or _property(value) is not None
                or isinstance(value, (InstanceState, ActivationPatch, Adapter, PositionState, SharedLayers))
                or isinstance(value, FunctionType)
                and any(key.startswith("__pllm_") for key in vars(value))
            ):
                declarations.setdefault(name, []).append(base)
    for name, bases in declarations.items():
        for index, left in enumerate(bases):
            for right in bases[index + 1 :]:
                if not issubclass(left, right) and not issubclass(right, left):
                    raise ValueError(
                        f"{left.__name__} and {right.__name__} both declare {name!r}; compose them as fields instead of inheriting from both"
                    )
