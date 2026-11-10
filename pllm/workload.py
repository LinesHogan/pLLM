"""pllm.Workload, the base class every workload subclasses, and reflection: reading a workload's declared states,
decorated methods and input markers once into a WorkloadContext, which pllm.execution.workload_lowering lowers into
a TensorDeclaration. Used by pllm (make_llm, explain), pllm.host and pllm.testing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields, is_dataclass, replace
from functools import partial
from graphlib import CycleError, TopologicalSorter
from inspect import Parameter as SignatureParameter, signature
from itertools import pairwise
from types import FunctionType, MethodType
from typing import ClassVar, TypeVar

import torch

from pllm.authoring import BackendWorkload
from pllm.config import ExecutionOptions
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import SchedulerTimeline, SemanticGraph, StateScope, TaskId
from pllm.execution.causal_batch import CausalBinder
from pllm.execution.compile_facts import BackendCompileFacts, BackendRequirements
from pllm.execution.declaration import When
from pllm.execution.freeze import (
    fingerprint,
    freeze,
    members as _members,
    owner_snapshot,
    owner_values,
    reachable_functions,
)
from pllm.execution.host_runtime import HostConfig
from pllm.execution.method_effects import (
    CustomResidual,
    Effect,
    Gradient,
    TrainInForward,
)
from pllm.execution.method_inputs import (
    InputMarker,
    InputTokens,
    Parameter,
    Result,
)
from pllm.execution.position_state import PositionState
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.state_init import Computed
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_declaration import TensorDeclaration, TensorOutput
from pllm.execution.tensor_frame import TensorFrame
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
from pllm.usage import ActivationPatch, Adapter, InstanceState, Phase, RequestState

C = TypeVar("C")


class Workload(BackendWorkload[TensorFrame, TensorOutput]):
    """Python methods and effects compiled once, with independent sub-workloads in dataclass fields.

    Sampler additions follow member declaration order (parent methods, then fields recursively), before
    vLLM filters and at most one guide. Set required_publish_lag when the algorithm requires a fixed
    value; every component must agree with the engine's ExecutionOptions.publish_lag.
    """

    required_publish_lag: ClassVar[int | None] = None

    def specialize(
        self, facts: BackendCompileFacts, options: ExecutionOptions, *, causal: CausalBinder | None = None
    ) -> tuple[BackendCompileFacts, TensorDeclaration]:
        """Check component timing requirements against the single engine setting."""
        for path, component in component_workloads(self):
            required = component.required_publish_lag
            if required is not None and required != options.publish_lag:
                raise ValueError(
                    f"{path or type(component).__name__} requires publish_lag={required}; the engine uses publish_lag={options.publish_lag}; all components must use the engine-wide lag"
                )
        return super().specialize(facts, options, causal=causal)

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
        context = read_workload(self)
        definitions = context.state_definitions
        inputs: dict[str, tuple[InputMarker, ...]] = {}
        for operation in context.operations:
            if operation.effect.reads_taps:
                with operation.context.declaring(operation.name):
                    inputs.update((name, tuple(arguments.values())) for name, arguments in operation.sources().items())
        declared = BackendRequirements(
            parameter_targets=tuple(target for delta in context.adapters.values() for target in delta.targets),
            activation_layers=tuple(layer for patch in context.activations.values() for layer in patch.layers),
            position_states=context.position_states(),
            custom_forward=bool(context.custom_forwards),
            instance_adapter=any(
                definitions[delta.state].scope is StateScope.INSTANCE
                for delta in context.adapters.values()
                if delta.state in definitions
            ),
        )
        return BackendRequirements.combine(
            (
                declared,
                *(state.init.requirements(state) for state in definitions.values()),
                *(operation.effect.requirements() for operation in context.operations),
                *(source.requirements() for name in sorted(inputs) for source in inputs[name]),
            )
        )

    def resource_facts(
        self, declaration: TensorDeclaration, facts: BackendCompileFacts, options: ExecutionOptions
    ) -> SchedulerTimeline:
        """Reserve ordinary tensor scratch and each requested graph's private pool."""
        return reserve_resources(declaration, facts, options)

    def bind_math(
        self,
        facts: BackendCompileFacts,
        plan: PlanBundle,
        declaration: TensorDeclaration,
        *,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> MathBinding[StepInput, TensorFrame, TensorOutput]:
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

    def compile_host(self, config: HostConfig) -> StateExecution[StepInput, TensorFrame, TensorOutput]:
        """Build, plan and bind the graph without a model (see pllm.host.make_host); training reads host batches."""
        declaration = host_declaration(type(self).__qualname__, config)
        lower_workload(read_workload(self), declaration)
        return bind_host_declaration(declaration, config, self._export_handler(declaration, declaration.graph))


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
    root.signature = [item for context in contexts for item in context.signature]
    root.bindings = tuple(item for context in contexts for item in context.bindings)
    root.in_forward_states = {name for context in contexts for name in context.in_forward_states}
    for context in contexts:
        context.components = contexts
        context.state_definitions = root.state_definitions
        context.adapters = root.adapters
        context.activations = root.activations
        context.position_definitions = root.position_definitions
        context.operations = root.operations
        context.custom_forwards = root.custom_forwards
        context.signature = root.signature
        context.states = root.states
        context.parameters = root.parameters
        context.writes = root.writes
        context.trainers = root.trainers
        context.predictions = root.predictions
        context.in_forward_states = root.in_forward_states
    return root


def _read_component(workload: Workload, prefix: str) -> WorkloadContext:
    """Snapshot and sign reachable mathematics before evaluating state properties."""
    owner = _snapshot(workload)
    values, members = _values(owner), _members(owner)
    _check_base_collisions(owner)
    context, candidates = WorkloadContext(owner, prefix=prefix), _candidates(owner, values, members)
    seeds: list[FunctionType] = []
    for name, member in members.items():
        with context.declaring(name):
            _read_operation(context, name, member, candidates, seeds)
        if (getter := _property(member)) is not None:
            seeds.append(getter)
    for value in values.values():
        if (state := _typed(value, InstanceState)) is not None and isinstance(state.init, Computed):
            method = _typed(state.init.function, MethodType)
            if (
                function := _typed(state.init.function if method is None else method.__func__, FunctionType)
            ) is not None:
                seeds.append(function)
    _declare_result_helpers(context, candidates, seeds)
    _reject_result_cycles(context)
    _freeze_helpers(context, candidates, seeds)
    context.bindings = tuple(
        (key, fingerprint(value)) for key, value in sorted(values.items()) if not isinstance(value, Workload)
    )
    for name, member in members.items():
        if _property(member):
            values[name] = object.__getattribute__(owner, name)
    context.state_definitions = _of_type(values, InstanceState)
    context.adapters = {key: value.delta for key, value in _of_type(values, Adapter).items()}
    context.activations = _of_type(values, ActivationPatch)
    context.position_definitions = _of_type(values, PositionState)
    _namespace(context)
    return context


def _read_operation(
    context: WorkloadContext, name: str, member: object, candidates: dict[str, FunctionType], seeds: list[FunctionType]
) -> None:
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
        seeds.append(function)
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
    seeds.extend(
        _named_methods((*vars(effect).values(), *vars(gradient or Gradient((), False)).values(), rows), candidates)
    )
    if (trained := _typed(effect, TrainInForward)) is not None:
        context.in_forward_states.add(trained.state)
    if (residual := _typed(effect, CustomResidual)) is not None and residual.state:
        context.in_forward_states.add(residual.state)
    when = phase.value if phase is not None else effect.phase or When.AFTER_SAMPLE
    seeds.append(function)
    assert rows is not None
    context.operations.append(Operation(context, name, when, effect, gradient, rows, phase is not None))


def _declare_result_helpers(
    context: WorkloadContext, candidates: dict[str, FunctionType], seeds: list[FunctionType]
) -> None:
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
            seeds.append(candidates[source.operation])
            seeds.extend(_named_methods(vars(gradient or Gradient((), False)).values(), candidates))
            pending.append((helper, phase))


def _named_methods(values: Iterable[object], candidates: dict[str, FunctionType]) -> list[FunctionType]:
    """The methods a decorator names by string (rows=, input_rows=, ...), which must be frozen with the workload."""
    return [candidates[value] for value in values if type(value) is str and value in candidates]


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


def _freeze_helpers(context: WorkloadContext, candidates: dict[str, FunctionType], seeds: list[FunctionType]) -> None:
    """Freeze only helpers reachable from operations, state properties or markers."""
    reachable = reachable_functions(candidates, tuple(seeds)) | {
        name for name, member in candidates.items() if any(member is seed for seed in seeds)
    }
    for name in candidates:
        if name in reachable:
            function = context.method(name)
            with context.declaring(name):
                context.signature.append((name, fingerprint(function)))
            object.__setattr__(context.owner, name, freeze(function))


def _namespace(context: WorkloadContext) -> None:
    """Qualify emitted names while retaining each method's bound owner and local configuration."""
    context.bindings = tuple((context.qualify(key), value) for key, value in context.bindings)
    context.state_definitions = {context.qualify(key): value for key, value in context.state_definitions.items()}
    context.adapters = {
        context.qualify(key): replace(value, state=context.qualify(value.state))
        for key, value in context.adapters.items()
    }
    context.activations = {
        context.qualify(key): replace(value, state=context.qualify(value.state))
        for key, value in context.activations.items()
    }
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


def _values(owner: object) -> dict[str, object]:
    return owner_values(owner, (InstanceState, ActivationPatch, Adapter, PositionState, Workload))


def _snapshot(owner: C, borrowed: tuple[object, ...] = ()) -> C:
    return owner_snapshot(owner, (InstanceState, ActivationPatch, Adapter, PositionState, Workload), borrowed)


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
    declared = (InstanceState, RequestState, ActivationPatch, Adapter, PositionState)
    return getter if getter is not None and _return_type(getter) in declared else None


def _candidates(owner: Workload, values: dict[str, object], members: dict[str, object]) -> dict[str, FunctionType]:
    """Find helper methods in MRO order, excluding framework and value bindings."""
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
        if base in (Workload, BackendWorkload, object):
            continue
        own_fields = _own_fields(base)
        for name in dict.fromkeys((*vars(base), *sorted(own_fields))):
            value = vars(base).get(name)
            if (
                name in own_fields
                or _property(value) is not None
                or isinstance(value, (InstanceState, ActivationPatch, Adapter, PositionState))
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
