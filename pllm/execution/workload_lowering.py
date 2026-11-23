"""Lower a reflected workload into a TensorDeclaration. pllm.workload reads an author's class into a
WorkloadContext holding its Operations and declared states; this module allocates the states, has each operation's
effect declare its tasks, then composes and signs the graph. It also prepares and finishes the declaration of a model
or host build around that lowering."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from inspect import signature
from math import prod
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Literal

import torch

from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import ActivationPatchOperator, DecodeRefresh, DecodeWindow, MissingPolicy
from pllm.contracts.hidden_locator import (
    HiddenLayerKind,
    HiddenSite,
    LayerRouteSelector,
    resolve_route_layers,
    route_locator_tag,
)
from pllm.contracts.host_batch import StateImport, validate_host_batch
from pllm.contracts.member import MemberDeclaration
from pllm.contracts.parameter_overlay import (
    OverlayRoute,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ProjectionDelta,
)
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    HiddenInputSource,
    HostServiceInputSource,
    MathContract,
    SamplerFeedbackInputSource,
    SchedulerTimeline,
    SemanticOutputKind,
    SemanticTask,
    StateEffectKind,
    StateResource,
    StateScope,
    TaskResultInputSource,
    TaskSymbol,
    TimingConstraint,
    TrainingMode,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.binding_base import Field, layout_field, overlay_fields, routed_overlay_fields
from pllm.execution.causal_batch import CausalBinder
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.declaration import Call, Dimensions, LayoutField, State, When
from pllm.execution.host_runtime import HostConfig, bind_host
from pllm.execution.method_effects import TRAINING_INPUTS, Effect, Gradient, Sampler
from pllm.execution.method_inputs import (
    Carried,
    Carry,
    InputMarker,
    Parameter,
    Result,
    RoutedHidden,
    RoutedLayerLogits,
    method_inputs,
)
from pllm.execution.model_effect_binding import declare_activation_patch, declare_custom_forward
from pllm.execution.model_sites import StateInput
from pllm.execution.position_state import PositionState
from pllm.execution.row_task_binding import declare_logits, declare_parameters, declare_tensor, validate_gradients
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.state_init import Computed, Fed, Imported, Pretrained, Zeros
from pllm.execution.step_input import StepInput
from pllm.execution.task_memory import derive_host_resources, derive_resources, shape_site_results
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.tensor_frame import TensorFrame, TensorPort, UpdateOutput, result_port
from pllm.execution.tensor_program import TaskOutput, TensorBatch
from pllm.execution.tensor_task import TensorExport
from pllm.execution.token_program import Over, Segment, Tokens
from pllm.execution.trace import TraceEvent

if TYPE_CHECKING:
    from pllm.usage import ActivationPatch, InstanceState
    from pllm.workload import Workload


@dataclass
class Operation:
    """A reflected method, with marker types checked once at the boundary."""

    context: WorkloadContext
    name: str
    phase: When
    effect: Effect
    gradient: Gradient | None
    rows: str
    phase_override: bool = False

    @property
    def plain(self) -> bool:
        """A method without an effect: a result helper, or a method placed only by @pllm.on."""
        return type(self.effect) is Effect

    @property
    def function(self) -> FunctionType | MethodType:
        return self.context.method(self.name)

    @property
    def identity(self) -> str:
        """Identify a method implementation by its qualified method name, unique in the workload."""
        return f"python:{self.name}"

    @property
    def transition_window(self) -> When:
        """Where this method's state transition runs; host workloads have their own step anchors."""
        return When.HOST_COMPUTE if self.context.binding.host else self.phase

    def reads_field(self, state: str, field: str) -> bool:
        """Follow declared parameter and method-result inputs to a previous field version."""
        pending, visited = [self.name], set[str]()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            sources, _ = next(operation for operation in self.context.operations if operation.name == name).arguments()
            for source in sources.values():
                if isinstance(source, Parameter) and (source.state, source.field) == (state, field):
                    return True
                if isinstance(source, Result):
                    pending.append(source.operation)
        return False

    def arguments(self) -> tuple[dict[str, InputMarker], dict[str, Carry]]:
        """Split carried inputs into their read source and the carried field the update writes."""
        sources, carries = self.context.arguments(self.function), dict[str, Carry]()
        for key, source in sources.items():
            if isinstance(source, Carried):
                sources[key], carries[key] = source.source, source.carry
        return sources, carries

    def tensor(
        self,
        arguments: dict[str, InputMarker] | None = None,
        carries: dict[str, Carry] | None = None,
        *,
        when: When | None = None,
        update_kind: StateEffectKind = StateEffectKind.UPDATE_MAYBE,
        output_kind: SemanticOutputKind = SemanticOutputKind.TEMPORARY,
        result_schema: str = "",
        export: TensorExport | None = None,
        sampler_stage: SamplerStage | None = None,
        sampler_beta: float = 0.0,
    ) -> Call:
        """Declare tensor mathematics, derivative domains and memory publication."""
        if arguments is None:
            arguments, carries = self.arguments()
        context, effect = self.context, self.effect
        inputs = {key: source.port(context) for key, source in arguments.items()}
        routed_sources = tuple(
            source for source in arguments.values() if isinstance(source, (RoutedHidden, RoutedLayerLogits))
        )
        unique_routes = tuple(dict.fromkeys(source.route for source in routed_sources))
        if export is not None and len(unique_routes) == 1:
            first_source = routed_sources[0]
            resolved = context.bind_layer_route(first_source.route)
            export = replace(
                export,
                route_locator_key=(
                    route_locator_tag(resolved),
                    HiddenLayerKind.WILDCARD.value,
                    first_source.resolve().site.value,
                ),
            )
        demand = self.gradient or Gradient((), effect.training)
        math = demand.contract(self, arguments)
        update = context.states[effect.state] if effect.training else None
        writes = {key: carry.declare(effect.state, update) for key, carry in (carries or {}).items()}
        math = self.selectors(arguments, inputs, math)
        return declare_tensor(
            context.binding,
            self.name,
            self.function,
            inputs=inputs,
            when=self.phase if when is None else when,
            identity=self.identity,
            update=update,
            math=math,
            carries=writes,
            outputs=effect.outputs,
            tuple_output=effect.tuple_output,
            effect=update_kind,
            output_kind=output_kind,
            result_schema=result_schema,
            export=export,
            sampler_stage=sampler_stage,
            sampler_beta=sampler_beta,
        )

    @property
    def domains(self) -> dict[str, str]:
        """Selector methods whose inputs contribute to the operation's model requirements."""
        demand = self.gradient or Gradient((), self.effect.training)
        return {
            "forward_rows": self.rows,
            "input_gradient_rows": demand.input_rows,
            "parameter_gradient_rows": demand.parameter_rows,
            "update_rows": self.effect.update_rows,
        }

    def sources(self) -> dict[str, dict[str, InputMarker]]:
        """Each task this operation declares, by name, with the sources it reads: the method and its row selectors."""
        tasks = {self.name: self.arguments()[0]}
        for method in self.domains.values():
            if method:
                tasks[self.context.selector_name(self.name, method)] = self.context.arguments(
                    self.context.method(method)
                )
        return tasks

    def selectors(
        self,
        arguments: dict[str, InputMarker],
        inputs: dict[str, TensorPort],
        math: MathContract,
    ) -> MathContract:
        """Declare each selector once and connect its row-domain result; selectors read their own inputs."""
        selectors: dict[str, Call] = {}
        sources = self.sources()
        for domain, method in self.domains.items():
            if not method:
                continue
            if method not in selectors:
                selectors[method] = declare_tensor(
                    self.context.binding,
                    self.context.selector_name(self.name, method),
                    self.context.method(method),
                    when=self.phase,
                    identity=f"python:{self.name}:{method}",
                    inputs={
                        key: inputs[key] if arguments.get(key) == source else source.port(self.context)
                        for key, source in sources[self.context.selector_name(self.name, method)].items()
                    },
                    gated=inputs,
                )
            key = f"__{domain}"
            inputs[key] = result_port(TaskResultInputSource(selectors[method].symbol, "value"))
            math = replace(math, **{domain: key})
        return math


@dataclass
class WorkloadContext:
    """One workload component during one build: what reflection read from it (owner, definitions, operations) and
    what lowering has declared so far (the binding, allocated states and their ports, writes, trainers). Components
    of one workload share every collection but the owner and prefix."""

    owner: Workload
    binding: TensorDeclaration = field(init=False)
    prefix: str = ""
    components: tuple[WorkloadContext, ...] = ()
    # The owner's values, signed; the declared states and model values among them, by type.
    state_definitions: dict[str, InstanceState] = field(default_factory=lambda: {})
    # Each declared pllm.model.Adapter as the projection delta it applies.
    adapters: dict[str, ProjectionDelta] = field(default_factory=lambda: {})
    activations: dict[str, ActivationPatch] = field(default_factory=lambda: {})
    position_definitions: dict[str, PositionState] = field(default_factory=lambda: {})
    # Ordered (name, digest) entries folded into the graph's execution parameters digest.
    operations: list[Operation] = field(default_factory=lambda: [])
    custom_forwards: list[tuple[str, Tokens, Over]] = field(default_factory=lambda: [])
    positions: tuple[tuple[str, float], ...] = ()
    in_forward_states: set[str] = field(default_factory=lambda: set[str]())
    states: dict[str, State] = field(default_factory=lambda: {})
    parameters: dict[tuple[str, str], TensorPort] = field(default_factory=lambda: {})
    writes: dict[str, tuple[Call, Call]] = field(default_factory=lambda: {})
    predictions: list[tuple[Call, Sampler]] = field(default_factory=lambda: [])
    # Each @pllm.train method and the state it trains.
    trainers: list[tuple[Call, State]] = field(default_factory=lambda: [])

    def validate_pretrained(self) -> None:
        """Validate declared weights on CPU after every lowering has resolved its field shapes."""
        for resource in self.binding.pretrained:
            if resource in self.binding.initial_values:
                self.binding.initial_values[resource](self.binding.dimensions, 0)
            else:
                layout = next(item.layout for item in self.binding.layouts.layouts if item.resource == resource)
                shards = self.binding.field_shards[resource]
                tensors = tuple(
                    torch.empty(shards[i].shape if i in shards else field.shape, dtype=field.dtype.torch, device="meta")
                    for i, field in enumerate(layout.fields)
                )
                self.binding.initializers[resource](tensors, torch.Generator())

    def field_dtype(self, state: str, field: str, dtype: torch.dtype | Literal["auto"]) -> TensorDType:
        """Choose storage once from the declared field role; explicit precision always wins."""
        if dtype != "auto":
            return TensorDType(str(dtype).removeprefix("torch."))
        definition = self.state_definitions[state]
        if definition.optimizer is not None or any(
            operation.effect.state == state and operation.effect.accumulates(operation, field)
            for operation in self.operations
        ):
            return TensorDType.FLOAT32
        return self.binding.dimensions.dtype

    @property
    def sampler(self) -> Sampler | None:
        """Find active guidance before either prediction or training is emitted."""
        samplers = tuple(
            operation.effect
            for operation in self.operations
            if isinstance(operation.effect, Sampler) and operation.effect.enabled
        )
        return next(
            (sampler for sampler in samplers if sampler.stage is SamplerStage.PACKED_POST_FILTER_EXACT),
            next(iter(samplers), None),
        )

    @property
    def has_model_effect(self) -> bool:
        """Model effects own training unless a sampler pairs with it."""
        return bool(self.custom_forwards or self.adapters or self.activations) or any(
            operation.effect.model_effect for operation in self.operations
        )

    def adapter_bindings(self, delta: ProjectionDelta) -> tuple[ParameterOverlayFieldBinding, ...]:
        """The field views an adapter applies, shared by inference and differentiable model calls."""
        definition = self.state_definitions[delta.state]
        if delta.route is None:
            return delta.field_bindings(lambda name: self.binding.field_views[delta.state, name])
        resolved = self.bind_layer_route(delta.route)
        if delta.refresh == "admission":
            _reject_mutable_request_admission(self, delta.state)
            _reject_mutable_request_admission(self, resolved.resource)
        if definition.layers and definition.layers != resolved.over:
            raise ValueError(
                f"{delta.state}.layers={definition.layers} must match LayerRoute(over={resolved.over}) or be omitted"
            )
        route_view = self.binding.field_views[resolved.resource, resolved.field]
        return delta.field_bindings(
            lambda name: self.binding.field_views[delta.state, name],
            layer_bank=bool(definition.layers),
            route=OverlayRoute(resolved.resource, route_view, delta.refresh),
        )

    def bind_layer_route(self, route: LayerRouteSelector) -> LayerRouteSelector:
        """Validate a LayerRoute's int64 scalar control field and resolve its candidate layers."""
        definition = self.state_definition(route.resource, "pllm.model.LayerRoute needs a declared state")
        if definition.optimizer is not None:
            raise ValueError(
                f"LayerRoute state {route.resource!r} cannot declare optimizer=...: discrete int64 layer indices are "
                "not differentiable; put the int64 route field in a separate state without an optimizer"
            )
        self.parameter(route.resource, route.field)
        view = self.binding.field_views[route.resource, route.field]
        layout = next(item.layout for item in self.binding.layouts.layouts if item.resource == route.resource)
        spec = layout.fields[view.field]
        if view.shape != () or spec.dtype is not TensorDType.INT64:
            raise ValueError(
                f"LayerRoute field {route.resource}.{route.field} must be a 0-d int64 scalar: pllm.Field((), torch.int64)"
            )
        over = resolve_route_layers(route.over, self.binding.dimensions.layer_count)
        self.binding.record_route_bound(route.resource, route.field, over)
        return replace(route, over=over)

    def qualify(self, name: str) -> str:
        """Keep references within one component; dots are reserved for framework paths."""
        if "." in name:
            raise ValueError(
                f"{self.prefix or type(self.owner).__name__}: {name!r}: components are independent; to share state, put both methods in one workload"
            )
        return self.emitted(name)

    def emitted(self, name: str) -> str:
        """Prefix generated names and export labels, which may themselves contain dots."""
        return f"{self.prefix}.{name}" if self.prefix else name

    def selector_name(self, operation: str, method: str) -> str:
        """Give generated selectors the same component namespace as their method."""
        prefix = self.prefix + "." if self.prefix else ""
        return self.emitted(f"__{operation.removeprefix(prefix)}_{method.removeprefix(prefix)}")

    def arguments(
        self, function: Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | UpdateOutput | None]
    ) -> dict[str, InputMarker]:
        """Resolve local marker names against this method's component."""
        return {key: source.bind(self) for key, source in method_inputs(function).items()}

    def component(self, name: str) -> WorkloadContext:
        """Find the owner of a qualified member, including nested field paths."""
        return next(context for context in reversed(self.components) if name.rpartition(".")[0] == context.prefix)

    def method(self, name: str) -> FunctionType | MethodType:
        """Bind a named authoring method, including an instance-local replacement."""
        try:
            context = self.component(name) if self.components else self
            return object.__getattribute__(
                context.owner, name.removeprefix(f"{context.prefix}.") if context.prefix else name
            )
        except AttributeError:
            raise ValueError(f"{name!r} names no method of {type(self.owner).__name__}") from None

    @contextmanager
    def declaring(self, member: str) -> Generator[None]:
        """Name the workload member whose declaration failed; the exception keeps its type."""
        try:
            yield
        except (ValueError, TypeError) as error:
            if self.prefix:
                error.args = (f"{member}: {error}", *error.args[1:])
            error.add_note(f"while declaring {member if self.prefix else type(self.owner).__name__ + chr(46) + member}")
            raise

    def state_definition(self, name: str, error: str) -> InstanceState:
        definition = self.state_definitions.get(name)
        if definition is None:
            declared = ", ".join(self.state_definitions)
            raise ValueError(
                f"{error}: {name!r} is not a declared state of {type(self.owner).__name__} (declared: {declared or 'none'}); "
                "declare a state as a property annotated `-> pllm.RequestState` or `-> pllm.InstanceState`"
            )
        return definition

    def state(self, name: str, error: str) -> State:
        """The allocated state `name`; a missing one lists the declared states."""
        self.state_definition(name, error)
        return self.states[name]

    def parameter(self, state: str, field: str) -> TensorPort:
        """The port of one declared state field; a missing one lists the declared fields."""
        if (state, field) not in self.parameters:
            declared = ", ".join(f"{resource}.{key}" for resource, key in self.parameters)
            raise ValueError(f"pllm.parameter({state!r}, {field!r}) names no declared field; declared: {declared}")
        if self.binding.field_views[state, field].field in self.binding.field_shards.get(state, {}):
            raise ValueError(
                f"{state}.{field}: TP sharded fields are projection-local; other readers need tensor_parallel_size=1"
            )
        return self.parameters[state, field]

    def declare_values(self) -> None:
        """Allocate ordinary state before effects and operations consume its ports."""
        if self.binding.host and (self.custom_forwards or self.adapters or self.activations):
            raise ValueError(
                "pllm.model.Adapter, pllm.model.ActivationPatch and @pllm.custom_forward act inside a model forward; compile this "
                "workload with make_llm, not make_host"
            )
        self.positions = self.position_states()
        for name, state in self.state_definitions.items():
            with self.declaring(name):
                declare_state(self, name, state)

    def attach_imports(self) -> None:
        """Make every imported-state reader consume the call's before-forward publication."""
        for task in self.binding.graph.tasks:
            effects = tuple(
                replace(effect, version=self.writes[str(effect.resource)][1].symbol)
                if effect.kind is StateEffectKind.READ and str(effect.resource) in self.binding.imports
                else effect
                for effect in task.effects
            )
            if effects != task.effects:
                self.binding.update_task(task.symbol, lambda task, effects=effects: replace(task, effects=effects))

    def attach_effects(self) -> None:
        """Connect publications before the custom forward and missing-input branches."""
        for name, patch in self.activations.items():
            with self.declaring(name), self.binding.member(name, MemberDeclaration("ActivationPatch")):
                declare_patch(self, name, patch)
        for name, delta in self.adapters.items():
            with self.declaring(name), self.binding.member(name, MemberDeclaration("Adapter")):
                self.binding.overlay(self.adapter_bindings(delta))
        # The before-forward writes of states activation patches or routed adapters read: a custom forward reads their
        # versions, and a fed chain (feed -> @pllm.write -> patch) closes its conditional input at that publication.
        tasks_by_symbol = {task.symbol: task for task in self.binding.graph.tasks}
        publications = tuple(
            dict.fromkeys(
                self.writes[state][1]
                for state in (
                    *(patch.state for patch in self.activations.values()),
                    *(patch.route.resource for patch in self.activations.values() if patch.route is not None),
                    *(delta.state for delta in self.adapters.values() if delta.route is not None),
                    *(delta.route.resource for delta in self.adapters.values() if delta.route is not None),
                )
                if state in self.writes and _can_precede_forward(tasks_by_symbol, self.writes[state][1].symbol)
            )
        )
        if self.custom_forwards:
            declare_custom_forward(
                self.binding,
                tuple(
                    (self.component(name).prefix, tokens, over, *self.forward_inputs(name))
                    for name, tokens, over in self.custom_forwards
                ),
                self.positions,
                publications,
            )
            self.binding.members.update(
                (name, MemberDeclaration("custom_forward", tokens=tokens)) for name, tokens, _ in self.custom_forwards
            )
            task = self.binding.effects.custom_forward
            assert task is not None
            self.binding.task_members[task] = tuple(name for name, _, _ in self.custom_forwards)
        if self.binding.feeds:
            if len(publications) != 1:
                raise ValueError(
                    "a fed state reaches the model through one @pllm.write into the state a "
                    "pllm.model.ActivationPatch reads"
                )
            self.binding.conditional_input(tuple(feed.resource for feed in self.binding.feeds), publications[0])

    def forward_inputs(self, name: str) -> tuple[Callable[..., torch.Tensor], tuple[tuple[str, StateInput], ...]]:
        """Bind each forward's instance fields."""
        function = self.method(name)
        inputs: list[tuple[str, StateInput]] = []
        for key, parameter in tuple(signature(function).parameters.items())[1:]:
            source = parameter.default
            if not isinstance(source, Parameter):
                raise TypeError(
                    f"{name}.{key}: use pllm.parameter or pllm.module after layers; read plain settings from self"
                )
            binding = source.bind(self.component(name)).site_input(self)
            if binding.request or source.per_row:
                raise ValueError(
                    f"{name}.{key}: custom forwards read shared InstanceState; use InstanceState and remove per_row=True"
                )
            inputs.append((key, binding))
        return function, tuple(inputs)

    def position_states(self) -> tuple[tuple[str, float], ...]:
        """Resolve the pre-allocation schema without tracing tensors."""
        states = tuple((name, value.initial) for name, value in sorted(self.position_definitions.items()))
        if states and not self.custom_forwards:
            raise ValueError(
                "PositionState is read and written by layers.load/save inside a @pllm.custom_forward method"
            )
        return states

    def train(self, call: Call, state: str, mode: TrainingMode) -> None:
        mode_now = self.binding.graph.training_mode
        if mode_now is not None and mode_now != mode:
            raise ValueError(
                f"{call.symbol} trains on {TRAINING_INPUTS[mode]} but {self.trainers[0][0].symbol} on "
                f"{TRAINING_INPUTS[mode_now]}; pLLM funds one training mode per workload, so move one of them to a "
                "separate workload"
            )
        self.trainers.append((call, self.states[state]))
        self.binding.graph = replace(self.binding.graph, training_mode=mode)

    def attach_training(self) -> None:
        """Validate gradients and record the sampler and training operations."""
        validate_gradients(self.binding, tuple(call for call, _ in self.trainers))
        guides = [call for call, sampler in self.predictions if sampler.stage is SamplerStage.PACKED_POST_FILTER_EXACT]
        if len(guides) > 1:
            raise ValueError(
                f"{guides[0].symbol} and {guides[1].symbol} both declare @pllm.guide_sampling; keep at most one guidance method"
            )
        for call, sampler in self.predictions:
            declare_logits(self.binding, call, stage=sampler.stage)
        for call, _ in self.trainers:
            self.binding.trains(call)

    def overlap_sampler(self) -> None:
        """Overlap an independent private update with sampling, retaining the token-commit publication boundary."""
        b = self.binding
        # The invocation owns one pending write. Other writers, model effects and feeds retain their placement.
        if len(self.trainers) != 1 or self.has_model_effect or self.writes or b.feeds:
            return
        ((call, state),) = self.trainers
        if call.symbol not in b.tensor_tasks:
            return  # Host-batch training has its own execution and publication policy.
        operation = next(operation for operation in self.operations if operation.name == call.symbol)
        if operation.phase is not When.AUTOMATIC:
            return
        dependencies = b.tensor_tasks[call.symbol].dependencies(b.tensor_tasks)
        late = {task.symbol for task in b.graph.tasks if task.timing.after == AnchorSymbol("token.commit")}
        if any(task.symbol in late for task in dependencies[:-1]) or any(
            isinstance(source, SamplerFeedbackInputSource) for task in dependencies for source in task.sources.values()
        ):
            return

        def private(task: SemanticTask) -> SemanticTask:
            return replace(
                task,
                timing=TimingConstraint(AnchorSymbol("sampler.pre_draw"), AnchorSymbol("next.prepare")),
                effects=tuple(
                    replace(
                        effect,
                        kind=StateEffectKind.TRANSITION
                        if effect.kind is StateEffectKind.UPDATE
                        else StateEffectKind.TRANSITION_MAYBE,
                    )
                    if effect.kind in (StateEffectKind.UPDATE, StateEffectKind.UPDATE_MAYBE)
                    else effect
                    for effect in task.effects
                ),
            )

        b.update_task(call.symbol, private)
        with b.member(operation.name, b.members[operation.name]):
            b.publish(call, state, when=When.AFTER_SAMPLE, name=f"{call.symbol}.publish")

    def sign_composition(self) -> None:
        """Validate composed effects and sign ordered definitions and owner values."""
        b = self.binding
        b.sites.check(
            b.dimensions.layer_count,
            bool(self.custom_forwards),
            any(
                isinstance(port.source, HiddenInputSource)
                and port.source.site in (HiddenSite.RESIDUAL, HiddenSite.ATTENTION_RESIDUAL, HiddenSite.LAYER_LOGITS)
                for task in b.graph.tasks
                for port in task.inputs
            ),
        )
        if len(b.imports) > 1:
            raise ValueError(
                f"StateImport supplies one imported resource per workload; {', '.join(b.imports)} are imported"
            )
        results = tuple(dict.fromkeys(source.task for rewrite in b.sites.rewrites for source in rewrite.results))
        if results or not b.graph.tasks and b.graph.model_reads:
            states = tuple(self.states[str(resource)] for resource in b.graph.model_reads)
            b.compute(
                "model.read",
                inputs={
                    **{str(state.declaration.resource): state.read() for state in states},
                    **{str(task): TaskResultInputSource(task, "value") for task in results},
                },
                reads=tuple(state.version() for state in states),
                when=When.AFTER_FORWARD if results else When.IN_FORWARD,
                function=lambda data, batch: TaskOutput(),
            )

        written = {
            str(effect.resource)
            for task in b.graph.tasks
            for effect in task.effects
            if effect.kind is not StateEffectKind.READ
        }
        for name, definition in self.state_definitions.items():
            definition.init.check_use(name, written=name in written, model_read=name in b.graph.model_reads)
        # Fed and imported states change from outside the graph; trained or written states change inside it.
        frozen = {
            name
            for name, state in self.state_definitions.items()
            if state.optimizer is None and not isinstance(state.init, (Fed, Imported)) and name not in written
        }
        b.graph = replace(
            b.graph,
            states=tuple(
                replace(
                    state,
                    immutable=state.resource in frozen,
                    placement=self.state_definitions[state.resource].init.placement,
                )
                for state in b.graph.states
            ),
            model_results=tuple(sorted(b.model_site_tasks)),
        )


def declare_state(context: WorkloadContext, name: str, definition: InstanceState) -> None:
    """Allocate tensor parameters; the layout of fields an adapter reads follows its targets."""
    definition.init.register(context.binding, name, definition.scope)
    if name in context.in_forward_states:
        # The in-forward transition allocates the state and runs only the source's initializer.
        if not isinstance(definition.init, (Zeros, Computed, Pretrained)):
            raise ValueError(
                f"{name} is written inside the forward (@pllm.train_in_forward or @pllm.custom_residual(state=)), "
                f"which cannot apply pllm.{type(definition.init).__name__}; start it from pllm.Zeros(), "
                "pllm.Computed(...) or pllm.Pretrained(...)"
            )
        return
    if name in context.states:
        return
    fields = _layout_fields(context, name, definition)
    if context.binding.dimensions.tensor_parallel_size > 1 and any(field.shard is not None for field in fields):
        # Initialization runs on the whole field before each rank takes its slice.
        if definition.optimizer is not None or not isinstance(definition.init, (Zeros, Computed, Pretrained, Imported)):
            raise ValueError(
                "TP Adapter fields support Zeros, Computed, Pretrained and Imported initialization and shard-local "
                "@pllm.train_in_forward updates; optimizers and other init= sources need tensor_parallel_size=1"
            )
    context.states[name], ports = declare_parameters(
        context.binding,
        name,
        fields,
        optimizer=definition.optimizer,
        initialize=definition.init.initializer(
            context.binding, name, tuple(item.name for item in fields) + tuple(definition.carry)
        ),
        scope=definition.scope,
        keys=definition.request_keys,
        carry=tuple(
            LayoutField(key, shape, context.field_dtype(name, key, definition.dtype), definition.alignment)
            for key, shape in definition.carry.items()
        ),
    )
    context.parameters.update({(name, field): port for field, port in ports.items()})
    definition.init.declare(context, name, definition)


def _layout_fields(context: WorkloadContext, name: str, definition: InstanceState) -> tuple[LayoutField, ...]:
    """The allocated fields: plain shapes in the state's dtype, or pllm.Field values sized by the hidden width or,
    when a pllm.model.Adapter reads the state, once per adapter target."""
    consumer = next((adapter for adapter in context.adapters.values() if adapter.state == name), None)
    if definition.layers:
        if definition.layers[-1] >= context.binding.dimensions.layer_count:
            raise ValueError(f"{name}.layers exceeds the model layer count")
        if consumer is not None and consumer.route is not None:
            resolved_over = resolve_route_layers(consumer.route.over, context.binding.dimensions.layer_count)
            if definition.layers != resolved_over:
                raise ValueError(f"{name}.layers={definition.layers} must match LayerRoute(over={resolved_over})")
            context.binding.overlay_targets = consumer.targets
            return routed_overlay_fields(
                definition.field_specs(),
                consumer,
                context.binding.dimensions,
                lambda key, value: context.field_dtype(name, key, value.dtype),
                layer_bank=True,
            )
        return tuple(
            layout_field(
                f"{key}_{layer}",
                value,
                lambda m: (m.hidden_size, m.hidden_size),
                context.field_dtype(name, f"{key}_{layer}", value.dtype),
            )
            for layer in definition.layers
            for key, value in definition.field_specs().items()
        )
    if consumer is None:
        return tuple(
            layout_field(
                key, value, lambda m: (m.hidden_size, m.hidden_size), context.field_dtype(name, key, value.dtype)
            )
            if isinstance(value, Field)
            else LayoutField(key, value, context.field_dtype(name, key, definition.dtype), definition.alignment)
            for key, value in definition.fields.items()
        )
    if (
        consumer.operator is ParameterOverlayOperator.LOW_RANK_ADD
        and definition.optimizer is not None
        and isinstance(definition.init, Zeros)
    ):
        raise ValueError(
            f"{name}: add init=pllm.Computed(...) (e.g. random A, zero B); with both low_rank factors at zero "
            "every gradient is zero and the adapter never learns"
        )
    context.binding.overlay_targets = consumer.targets
    specs = definition.field_specs()
    if consumer.route is not None:
        route_key = consumer.route.field if consumer.route.resource == name else ""
        if route_key and definition.optimizer is not None:
            raise ValueError(
                f"LayerRoute state {name!r} cannot declare optimizer=...: discrete int64 layer indices are "
                "not differentiable; put the int64 route field in a separate state without an optimizer"
            )
        extra: tuple[LayoutField, ...] = ()
        if route_key and route_key in specs:
            raw_route = definition.fields[route_key]
            route_dtype = context.field_dtype(name, route_key, specs[route_key].dtype)
            valid_shape = raw_route == () or (
                isinstance(raw_route, Field) and not callable(raw_route.shape) and raw_route.shape == ()
            )
            if not valid_shape or route_dtype is not TensorDType.INT64:
                raise ValueError(
                    f"LayerRoute field {name}.{route_key} must be a 0-d int64 scalar: pllm.Field((), torch.int64)"
                )
            extra = (LayoutField(route_key, (), route_dtype, 8),)
        adapter_specs = {k: v for k, v in specs.items() if k != route_key}
        routed = routed_overlay_fields(
            adapter_specs,
            consumer,
            context.binding.dimensions,
            lambda key, value: context.field_dtype(name, key, value.dtype),
            layer_bank=False,
        )
        if not extra:
            return routed
        keys = tuple(specs)
        pos = keys.index(route_key)
        if len(consumer.target_names) == 1:
            return (*routed[:pos], *extra, *routed[pos:])
        if pos not in (0, len(keys) - 1):
            raise ValueError(
                f"{name}.{route_key}: place the LayerRoute control field first or last when adapting multiple projections"
            )
        return (*extra, *routed) if pos == 0 else (*routed, *extra)
    return overlay_fields(
        specs,
        consumer,
        context.binding.dimensions,
        lambda key, value: context.field_dtype(name, key, value.dtype),
    )


def _can_precede_forward(tasks_by_symbol: Mapping[TaskSymbol, SemanticTask], symbol: TaskSymbol) -> bool:
    """Whether `symbol` and its upstream task-result closure can execute before `model.execute`."""
    pending, visited = [symbol], set[TaskSymbol]()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        task = tasks_by_symbol[current]
        if task.timing.after != AnchorSymbol("request.prepare") or task.custom_forward is not None:
            return False
        for port in task.inputs:
            if isinstance(port.source, (HiddenInputSource, SamplerFeedbackInputSource)):
                return False
            if isinstance(port.source, TaskResultInputSource):
                pending.append(port.source.task)
    return True


def _reject_mutable_request_admission(context: WorkloadContext, state_name: str) -> None:
    definition = context.state_definitions[state_name]
    if definition.scope is not StateScope.REQUEST:
        return
    written_by_task = any(
        any(
            effect.resource == state_name and effect.kind not in (StateEffectKind.READ, StateEffectKind.PUBLISH)
            for effect in task.effects
        )
        and not any(
            isinstance(port.source, HostServiceInputSource) and port.source.schema == StateImport.schema
            for port in task.inputs
        )
        for task in context.binding.graph.tasks
    )
    shared_import = isinstance(definition.init, Imported) and RequestKeyField.REQUEST_ID not in definition.request_keys
    if definition.optimizer is not None or written_by_task or shared_import:
        raise ValueError(
            f'{state_name}: refresh="admission" on RequestState requires a frozen per-request state '
            '(key_fields including "request_id" without @pllm.write/@pllm.train); use refresh="step" or InstanceState'
        )


def declare_patch(context: WorkloadContext, name: str, patch: ActivationPatch) -> None:
    """Read the state at the activation sites; its writer, if the workload has one, is connected later."""
    definition = context.state_definition(patch.state, "pllm.model.ActivationPatch needs a declared state")
    if isinstance(definition.init, Fed):
        raise ValueError(
            "pllm.model.ActivationPatch reads a state the workload writes; it cannot read a fed state directly"
        )
    layers = patch.layers
    route_resource: StateResource | None = None
    route_field = None
    route_refresh = DecodeRefresh.STEP
    layer_bank = True
    if patch.route is not None:
        resolved = context.bind_layer_route(patch.route)
        if patch.refresh == "admission":
            _reject_mutable_request_admission(context, patch.state)
            _reject_mutable_request_admission(context, resolved.resource)
        layers = resolved.over
        route_resource = StateResource(resolved.resource)
        route_field = context.binding.field_views[resolved.resource, resolved.field]
        route_refresh = DecodeRefresh.STEP if patch.refresh == "step" else DecodeRefresh.INHERIT
        if definition.layers and definition.layers != layers:
            raise ValueError(
                f"{patch.state}.layers={definition.layers} must match LayerRoute(over={layers}) or be omitted"
            )
        layer_bank = bool(definition.layers)
    declare_activation_patch(
        context.binding,
        name,
        context.states[patch.state],
        layers=layers,
        operator=ActivationPatchOperator(patch.operator),
        first_tokens=patch.first_tokens,
        last_tokens=patch.last_tokens,
        decode=DecodeWindow.ALL if patch.decode else DecodeWindow.NONE,
        decode_refresh=DecodeRefresh.STEP if patch.refresh == "step" else DecodeRefresh.INHERIT,
        on_missing=MissingPolicy(patch.on_missing),
        route_resource=route_resource,
        route_field=route_field,
        route_refresh=route_refresh,
        layer_bank=layer_bank,
    )


def lower_workload(context: WorkloadContext, b: TensorDeclaration, capabilities: tuple[str, ...] = ()) -> None:
    """Declare a reflected workload's states and operations into `b`, then compose and sign their graph."""
    adapters = {key: value.expand(b.dimensions.layer_count) for key, value in context.adapters.items()}
    for component in context.components:
        component.binding = b
        component.adapters = adapters
    context.declare_values()
    ordered = sorted(context.operations, key=lambda operation: not operation.effect.declared_first)
    for operation in ordered:
        description = operation.effect.description()
        if operation.phase_override:
            description = replace(description, phase=operation.phase.name.lower())
        with operation.context.declaring(operation.name), b.member(operation.name, description):
            operation.effect.declare(operation)
    context.attach_imports()
    context.attach_effects()
    context.attach_training()
    if b.dimensions.tensor_parallel_size > 1 and b.effects.in_forward is None:
        for task in b.graph.tasks:
            if any(
                isinstance(port.source, HostServiceInputSource) and port.source.schema == StateImport.schema
                for port in task.inputs
            ):
                continue
            if any(
                b.field_shards.get(effect.resource)
                and effect.kind not in (StateEffectKind.READ, StateEffectKind.PUBLISH)
                for effect in task.effects
            ):
                raise ValueError(
                    "TP Adapter updates need shard-local @pllm.train_in_forward; side-lane updates need tensor_parallel_size=1"
                )
    if all(
        capability in capabilities
        for capability in ("compute/cuda@1", "transition/full_root@1", "backend/tensor_parallel@1")
    ):
        context.overlap_sampler()
    context.validate_pretrained()
    context.sign_composition()


def model_declaration(name: str, facts: BackendCompileFacts, causal: CausalBinder | None) -> TensorDeclaration:
    """An empty declaration against the backend's model dimensions, ready for lower_workload."""
    dimensions = Dimensions(
        facts.probe.hidden_size,
        facts.vocab_size,
        facts.probe.targets,
        facts.probe.tensor_dtype,
        facts.layer_count,
        facts.probe.tensor_parallel_size,
    )
    b = TensorDeclaration(name, dimensions)
    b.probe = facts.probe
    b.checkpoint = facts.checkpoint
    b.causal = causal
    return b


def finish_model_declaration(b: TensorDeclaration, facts: BackendCompileFacts) -> None:
    """Shape model-site results, plan custom attention and configure scans after lowering."""
    shape_site_results(b, facts)
    # Executable ports retain their dependencies; an unused factory must not pin the decoder.
    b.causal = None
    b.attention.plan(b, facts)
    if any(
        isinstance(step, Segment)
        and step.kv == "peek"
        and any(step.start <= layer < step.stop for layer in b.attention.methods)
        for program in (b.custom_forward_programs or {}).values()
        for step in program.steps
    ):
        raise ValueError(
            "@pllm.custom_attention inside @pllm.custom_forward reads only layers called with kv='commit'; use kv='commit'"
        )
    b.configure_scans(facts)


def reserve_resources(
    declaration: TensorDeclaration, facts: BackendCompileFacts, options: ExecutionOptions
) -> SchedulerTimeline:
    """Reserve ordinary tensor scratch and each requested graph's private pool."""
    if declaration.graph.trains_on_batch:
        if options.execution_peak_budget_bytes is not None and not facts.timeline.task_resources:
            raise ValueError(
                "training on CausalTrainBatch/WeightedRowBatch inputs cannot be funded under "
                "ExecutionOptions.execution_peak_budget_bytes yet; leave execution_peak_budget_bytes unset"
            )
        timeline = facts.timeline
    else:
        timeline = derive_resources(declaration, facts, options)
    resident = (
        declaration.template_bytes
        + declaration.attention.storage_bytes
        + facts.probe.max_packed_rows
        * sum(
            prod(source.shape) * source.dtype.itemsize
            for source in {source for rewrite in declaration.sites.rewrites for source in rewrite.results}
        )
    )
    return replace(
        timeline,
        reserved_bytes=tuple(value + resident for value in (timeline.reserved_bytes or (0,) * len(timeline.anchors))),
    )


def host_declaration(name: str, config: HostConfig) -> TensorDeclaration:
    """An empty declaration for a host loop without a model (pllm.host.make_host), ready for lower_workload."""
    return TensorDeclaration(name, Dimensions(config.hidden_size, 0), host=True)


def bind_host_declaration(
    b: TensorDeclaration, config: HostConfig, on_export: Callable[[TraceEvent], None] | None
) -> StateExecution:
    """Plan and bind a lowered host declaration; each step checks its host batch before reading it."""
    timeline = config.timeline()
    if config.execution_peak_budget_bytes is not None and b.graph.states:
        timeline = derive_host_resources(b, config)
    math = b.function_math(config.device, config.seed)
    prepare = math.prepare_data

    def checked_prepare(value: StepInput, batch: TensorBatch) -> TensorFrame:
        validate_host_batch(
            value.batch,
            b.graph,
            b.layouts,
            timeline,
            SharedTensor.describe,
            hidden_size=config.hidden_size,
            vocab_size=0,
            target_shape=config.target_shape,
            prompt_count=len(set(value.import_rows.values())),
            import_fields=b.effects.import_fields,
        )
        return prepare(value, batch)

    return bind_host(
        b.graph,
        b.layouts,
        MathBinding(
            replace(math, prepare_data=checked_prepare),
            b.bind_all(operations=b.executables(None, ())),
            on_export=on_export,
        ),
        config,
        timeline,
    )
