"""What a workload method's result does: the effect its decorator marks (@pllm.write, @pllm.export,
@pllm.train, @pllm.add_logits / guide_sampling, @pllm.train_in_forward, @pllm.custom_attention,
@pllm.custom_residual, and @pllm.differentiate's gradient demand), and how each effect declares its tasks into the
TensorDeclaration. pllm.usage's decorators build the effects; pllm.execution.workload_lowering calls `declare`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from inspect import Parameter as SignatureParameter, signature
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.host_batch import CausalTrainBatch, WeightedCEBatch, WeightedRowBatch
from pllm.contracts.kv_layout import Keep, KVLayout, expand_layouts
from pllm.contracts.member import MemberDeclaration
from pllm.contracts.parameter_overlay import FieldView, ParameterOverlayOperator, Projection, ProjectionDelta
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    MathContract,
    SemanticOutputKind,
    StateEffectKind,
    StateScope,
    TrainingMode,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.contracts.visibility import VisibilityKind
from pllm.execution.attention_methods import AttentionFact, AttentionMethod
from pllm.execution.binding_base import SGD, Field, TargetRows
from pllm.execution.causal_batch import declare_batch_train
from pllm.execution.compile_facts import BackendRequirements
from pllm.execution.declaration import Dimensions, LayoutField, Shape, When
from pllm.execution.method_inputs import (
    Attention,
    Batch,
    Carry,
    CausalModelInput,
    InputMarker,
    Operand,
    OperandName,
    Parameter,
)
from pllm.execution.model_effect_binding import (
    declare_parameter_overlay,
    declare_site_memory,
    write_result,
)
from pllm.execution.model_sites import (
    DepthInput,
    EarlierInput,
    PatchInput,
    Place,
    ResidualRewrite,
    ResultInput,
    Site,
    SiteInput,
    SiteMemory,
    StateInput,
)
from pllm.execution.overlay_parallel import admit_update
from pllm.execution.overlay_task import ChunkUpdate, GradientUpdate, overlay_gradient
from pllm.execution.row_task_binding import declare_parameters, declare_weighted_train
from pllm.execution.state_init import Fed
from pllm.execution.tensor_frame import Tensors, UpdateOutput
from pllm.execution.tensor_program import PoolSGD
from pllm.execution.tensor_task import TensorExport
from pllm.execution.token_program import Tokens
from pllm.kernels.adamw import AdamWConfig

if TYPE_CHECKING:
    from pllm.execution.workload_lowering import Operation, WorkloadContext
    from pllm.usage import InstanceState

# How @pllm.train reduces the selected per-row losses within each state in this step.
TrainReduction = Literal["mean", "sum"]
Initializer = Callable[[Tensors, torch.Generator], None]
# The tokens whose projection operands train a @pllm.train_in_forward state.
TOKEN_PHASES: dict[str, tuple[HiddenPhase, ...]] = {
    "prefill": (HiddenPhase.PREFILL,),
    "decode": (HiddenPhase.DECODE,),
    "all": (HiddenPhase.PREFILL, HiddenPhase.DECODE),
}


class Effect:
    """An operation effect owns its declaration and attachment policy."""

    kind: ClassVar[str] = "compute"
    training: ClassVar[bool] = False
    # Whether the effect acts inside the model forward, so it owns request training unless a sampler pairs with it.
    model_effect: ClassVar[bool] = False
    tuple_output: ClassVar[bool] = False
    # The phase the effect runs in when no `@pllm.on` names one; None means the author must name it.
    phase: ClassVar[When | None] = None

    state: str = ""
    update_rows: str = ""
    reduction: TrainReduction

    def description(self) -> MemberDeclaration:
        """The public effect name and its timing settings."""
        return MemberDeclaration(self.kind)

    def requirements(self) -> BackendRequirements:
        """What the backend must provide for this effect, beyond what the method's inputs need."""
        return BackendRequirements()

    @property
    def reads_taps(self) -> bool:
        """Whether the backend taps the rows the method's inputs read."""
        return True

    @property
    def declared_first(self) -> bool:
        """Whether other methods' declarations read what this one declares, so it is declared before them."""
        return False

    def accumulates(self, operation: Operation, field: str) -> bool:
        """Whether this effect carries the field's previous value into its next version."""
        return False

    def declare(self, operation: Operation) -> None:
        operation.tensor()

    @property
    def outputs(self) -> tuple[str, ...]:
        return ("value",)


@dataclass(frozen=True, kw_only=True)
class Write(Effect):
    kind: ClassVar[str] = "write"

    tuple_output: ClassVar[bool] = True

    state: str
    fields: tuple[str, ...]

    phase: ClassVar[When | None] = When.AUTOMATIC

    def accumulates(self, operation: Operation, field: str) -> bool:
        return field in self.fields and operation.reads_field(self.state, field)

    def declare(self, operation: Operation) -> None:
        context = operation.context
        definition = context.state_definition(self.state, "write needs a declared state")
        if isinstance(definition.init, Fed):
            raise ValueError(f"@pllm.write: {self.state!r} is filled by its feed; write a state without init=pllm.Fed")
        if not self.fields or len(set(self.fields)) != len(self.fields) or operation.rows:
            raise ValueError("@pllm.write needs distinct field names and no rows=; it writes every row")
        for key in self.fields:
            context.parameter(self.state, key)
        call = operation.tensor()
        state = context.states[self.state]
        update = write_result(context.binding, call, state, self.fields, operation.transition_window)
        context.writes[self.state] = (
            update,
            context.binding.publish(
                update, state, when=operation.transition_window.publication, name=f"{update.symbol}.publish"
            ),
        )

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.fields


@dataclass(frozen=True)
class Export(Effect):
    kind: ClassVar[str] = "export"

    name: str
    every: int
    max_rows: int
    max_cols: int
    enabled: bool
    overflow: Literal["drop", "block"]

    phase: ClassVar[When | None] = When.AFTER_FORWARD

    def declare(self, operation: Operation) -> None:
        """Publish bounded tensor observations to the host."""
        export = TensorExport(
            name=operation.context.emitted(self.name),
            every=self.every,
            max_rows=self.max_rows,
            max_cols=self.max_cols,
            enabled=self.enabled,
            overflow=self.overflow,
        )
        operation.context.binding.exports(operation.tensor(result_schema="hidden_export@1", export=export))


TRAINING_INPUTS: dict[TrainingMode, str] = {
    "request": "rows of running requests (no pllm.batch fields)",
    "batch": "WeightedRowBatch or CausalTrainBatch fields (or a causal model)",
    "weighted_batch": "WeightedCEBatch fields (base_logits or labels with hidden rows)",
}


def _training_schema(arguments: Mapping[str, InputMarker]) -> str:
    """Choose one batch from the method's complete field set; shared field names keep that schema."""
    fields = {source.field for source in arguments.values() if isinstance(source, Batch)}
    if not fields:
        return ""
    for batch in (WeightedRowBatch, CausalTrainBatch, WeightedCEBatch):
        if fields <= set(batch.names):
            return batch.schema
    raise ValueError(
        f"pllm.batch fields {sorted(fields)} must all come from one batch: WeightedRowBatch {WeightedRowBatch.names}, "
        f"CausalTrainBatch {CausalTrainBatch.names} or WeightedCEBatch {WeightedCEBatch.names}"
    )


def _training_mode(arguments: Mapping[str, InputMarker]) -> TrainingMode:
    """Batch fields choose the batch; without them or a causal model, train on request rows."""
    schema = _training_schema(arguments)
    if not schema:
        return "batch" if any(isinstance(source, CausalModelInput) for source in arguments.values()) else "request"
    return "weighted_batch" if schema == WeightedCEBatch.schema else "batch"


@dataclass(frozen=True, kw_only=True)
class Train(Effect):
    kind: ClassVar[str] = "train"

    state: str
    update_rows: str
    reduction: TrainReduction
    steps: int
    reset: str

    training: ClassVar[bool] = True
    phase: ClassVar[When | None] = When.AUTOMATIC

    def declare(self, operation: Operation) -> None:
        """Train on what the method reads (see `_training_mode`); the state's optimizer must suit it."""
        arguments, carries = operation.arguments()
        definition = operation.context.state_definition(self.state, "@pllm.train needs a declared state")
        mode = _training_mode(arguments)
        if mode != "request" and self.reduction != "mean":
            raise ValueError("host-batch training returns one scalar and owns its reduction; remove reduction=")
        if mode != "request" and operation.phase is When.AUTOMATIC:
            operation = replace(operation, phase=When.AFTER_SAMPLE)
        match mode, definition.optimizer:
            case "weighted_batch", SGD() as optimizer:
                self.weighted(operation, arguments, carries, definition, optimizer)
            case "batch", AdamWConfig():
                self.batch(operation, arguments, carries)
            case "request", AdamWConfig():
                self.request(operation, arguments, carries)
            case _:
                expected = "pllm.SGD(lr)" if mode == "weighted_batch" else "pllm.AdamW(lr)"
                raise ValueError(
                    f"@pllm.train reads {TRAINING_INPUTS[mode]}, which trains with {expected}; declare "
                    f"{self.state!r} with optimizer={expected}"
                )

    def batch(self, operation: Operation, arguments: dict[str, InputMarker], carries: dict[str, Carry]) -> None:
        """One AdamW step on a host CausalTrainBatch or WeightedRowBatch."""
        context = operation.context
        if carries or operation.gradient or operation.rows or self.update_rows or self.reset or self.steps != 1:
            raise ValueError(
                "training on WeightedRowBatch or CausalTrainBatch fields takes one AdamW step over the whole batch: "
                "remove rows=, update_rows=, steps=, reset=, @pllm.differentiate and pllm.carry"
            )
        schema = _training_schema(arguments) or CausalTrainBatch.schema
        context.train(
            declare_batch_train(
                context.binding,
                operation.name,
                context.states[self.state],
                ports={key: source.training_port(context, schema) for key, source in arguments.items()},
                function=operation.function,
                identity=operation.identity,
                when=operation.transition_window,
            ),
            self.state,
            "batch",
        )

    def request(self, operation: Operation, arguments: dict[str, InputMarker], carries: dict[str, Carry]) -> None:
        """Train each request's state on its rows; token commit owns the step unless a model effect does."""
        if self.reset or self.steps != 1:
            raise ValueError(
                "steps= and reset= apply to training on WeightedCEBatch fields (pllm.batch); remove them to train on "
                "request rows"
            )
        context = operation.context
        sampler = context.sampler
        request_training = sampler is not None or not context.has_model_effect
        unconditional = request_training and (sampler is None or sampler.stage is SamplerStage.PACKED_POST_FILTER_EXACT)
        if unconditional and (operation.rows or self.update_rows):
            raise ValueError(
                "rows= and update_rows= on request training require a model effect without sampler guidance, "
                "or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state, "
                "so remove rows= and update_rows="
            )
        # Selectors keep their declared phase.
        when = When.AFTER_SAMPLE if request_training and operation.phase is When.AUTOMATIC else operation.phase
        context.train(
            operation.tensor(
                arguments,
                carries,
                when=when,
                update_kind=StateEffectKind.UPDATE if unconditional else StateEffectKind.UPDATE_MAYBE,
            ),
            self.state,
            "request",
        )

    def weighted(
        self,
        operation: Operation,
        arguments: dict[str, InputMarker],
        carries: dict[str, Carry],
        definition: InstanceState,
        optimizer: SGD,
    ) -> None:
        """Declare the weighted SGD batch transition, including its reset and input bindings."""
        if carries:
            raise ValueError("SGD batch training manages its own state writes")
        if not self.reset:
            raise ValueError(
                "training on WeightedCEBatch fields restarts each state before its SGD steps: give "
                "@pllm.train(reset=...) naming a method that sets the starting values"
            )
        if (
            operation.gradient is not None
            or operation.rows
            or self.update_rows
            or operation.phase is not When.AFTER_SAMPLE
        ):
            raise ValueError(
                "training on WeightedCEBatch fields uses the whole batch after sampling: remove rows=, update_rows=, "
                "@pllm.differentiate and @pllm.on"
            )
        context = operation.context
        sources = {key: source.batch(self.state, definition) for key, source in arguments.items()}
        reset = context.method(self.reset)
        context.train(
            declare_weighted_train(
                context.binding,
                operation.name,
                context.states[self.state],
                identity=operation.identity,
                optimizer=optimizer,
                steps=self.steps,
                reset=reset,
                function=operation.function,
                arguments=sources,
            ),
            self.state,
            "weighted_batch",
        )


@dataclass(frozen=True)
class Sampler(Effect):
    stage: SamplerStage
    beta: float | None = None  # None for additive logits, which have no guidance strength

    phase: ClassVar[When | None] = When.BEFORE_SAMPLE

    def description(self) -> MemberDeclaration:
        return MemberDeclaration("add_logits" if self.beta is None else "guide_sampling")

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(sampler_stages=(self.stage,) if self.enabled else ())

    @property
    def reads_taps(self) -> bool:
        """Zero-strength guidance reads nothing."""
        return self.enabled

    @property
    def enabled(self) -> bool:
        """Zero-strength guidance has no sampler attachment."""
        return self.beta != 0

    def declare(self, operation: Operation) -> None:
        """Zero-strength guidance declares no computation or sampler attachment."""
        if self.enabled:
            call = operation.tensor(
                output_kind=SemanticOutputKind.SAMPLER,
                sampler_stage=self.stage,
                sampler_beta=0.0 if self.beta is None else self.beta,
            )
            binding = operation.context.binding
            binding.tensor_tasks[call.symbol] = replace(
                binding.tensor_tasks[call.symbol], sampler_order=len(operation.context.predictions)
            )
            operation.context.predictions.append((call, self))


@dataclass(frozen=True)
class Gradient:
    inputs: tuple[str, ...]
    parameters: bool
    input_rows: str = ""
    parameter_rows: str = ""

    def contract(self, operation: Operation, arguments: dict[str, InputMarker]) -> MathContract:
        """Identify parameter resources and detachments."""
        parameters = {key: source for key, source in arguments.items() if isinstance(source, Parameter)}
        math = (
            MathContract(
                input_gradients=self.inputs,
                parameter_gradients=tuple(
                    dict.fromkeys(
                        operation.context.states[source.state].declaration.resource
                        for source in parameters.values()
                        if not operation.effect.training or source.state == operation.effect.state
                    )
                )
                if self.parameters
                else (),
                detached_inputs=tuple(key for key in arguments if key not in parameters and key not in self.inputs),
            )
            if operation.gradient is not None or operation.effect.training
            else MathContract()
        )

        return (
            replace(math, update_rows=next(iter(parameters), "__update_state"), reduction=operation.effect.reduction)
            if operation.effect.training
            else math
        )


def _author_fields(
    initialize: Initializer, names: tuple[str, ...], values: Tensors, generator: torch.Generator
) -> None:
    """Run the author's initializer on the fields it declared, not on the generated pending ones."""
    initialize({name: values[name] for name in names}, generator)


def _chunked_in_forward(
    function: Callable[..., Any],
    initialize: Initializer | None,
    bindings: dict[str, str],
    fields: dict[str, Field],
    effect: ProjectionDelta,
    chunk: int,
    optimizer: SGD | None,
    flush: str | None,
    invariants: tuple[str, ...],
    dimensions: Dimensions,
) -> tuple[Callable[..., UpdateOutput], Initializer | None]:
    """Generate pending state before layout interning and bind the existing chunk executor."""
    if initialize is not None:
        initialize = partial(_author_fields, initialize, tuple(fields))
    arguments = tuple(key for key in bindings if key not in invariants)
    sources = tuple(bindings[key] for key in arguments)
    if chunk < 1:
        raise ValueError("@pllm.train_in_forward(chunk=) must be positive")
    if optimizer is None:
        if (
            len(sources) < 2
            or sources[0] != effect.weight
            or any(s not in ("input", "target", "embedding") for s in sources[1:])
        ):
            raise ValueError(
                "a chunked update without an optimizer takes the weight carry, then input/target/embedding operands"
            )
        rows = dimensions.dtype.torch
        pending = {
            f"_pending_rows_{name}": TargetRows((chunk,), dtype=rows)
            if source == "target"
            else Field(
                lambda inputs, _outputs, source=source: (
                    chunk,
                    inputs if source == "input" else dimensions.hidden_size,
                ),
                dtype=rows,
            )
            for name, source in zip(arguments[1:], sources[1:], strict=True)
        }
        if tuple(bindings) != (*arguments, *invariants):
            raise ValueError("put scan invariant arguments after the weight carry and input/target/embedding operands")
        update = ChunkUpdate(function, chunk, arguments, invariants)
    else:
        if (
            "input" not in sources
            or any(
                source not in (*effect.applied_fields, "input", "target", "embedding", "weight") for source in sources
            )
            or any(sources.count(name) != 1 for name in effect.applied_fields)
        ):
            raise ValueError('a chunked loss reads pllm.operand("input") and each applied state field exactly once')
        pending = {
            f"_pending_grad_{name}": replace(
                fields[source],
                dtype=torch.float32
                if fields[source].dtype in (torch.float16, torch.bfloat16)
                else fields[source].dtype,
            )
            for name, source in bindings.items()
            if source in effect.applied_fields
        }
        update = GradientUpdate(
            overlay_gradient(function),
            overlay_gradient(function, batched=True),
            PoolSGD(optimizer.lr),
            effect,
            chunk,
            flush == "prompt_end",
            dict(bindings),
        )
    pending["_pending_length"] = Field((), dtype=torch.int64)
    if (pending.keys() | {"_prompt_end"}) & (fields.keys() | bindings.keys()):
        raise ValueError("rename the state fields or arguments that start with _pending_; pLLM reserves them")
    fields.update(pending)
    bindings.update((name, name) for name in pending)
    if optimizer is not None:
        bindings["_prompt_end"] = "prompt_end"
    return update, initialize


@dataclass(frozen=True, kw_only=True)
class TrainInForward(Effect):
    kind: ClassVar[str] = "train_in_forward"

    state: str
    targets: tuple[Projection, ...]
    weight: str
    operator: ParameterOverlayOperator
    factors: tuple[str, str]
    scale: float
    tokens: Tokens
    visibility: VisibilityKind
    chunk: int | None
    flush: Literal["prompt_end"] | None

    phase: ClassVar[When | None] = When.IN_FORWARD
    model_effect: ClassVar[bool] = True

    def accumulates(self, operation: Operation, field: str) -> bool:
        return (self.chunk is not None and field == self.weight) or operation.reads_field(self.state, field)

    def projection_delta(self, layers: int) -> ProjectionDelta:
        """The configured projection delta, with all-layer targets resolved against the model."""
        return ProjectionDelta(self.state, self.targets, self.weight, self.operator, self.factors, self.scale).expand(
            layers
        )

    def bindings(
        self, operation: Operation, fields: Mapping[str, Field], targets: tuple[Projection, ...], optimizer: SGD | None
    ) -> tuple[dict[str, str], dict[str, tuple[str, tuple[FieldView, ...]]], tuple[str, ...]]:
        """Resolve carried operands and shared, per-layer invariant views at declaration time."""
        context = operation.context
        sources = context.arguments(operation.function)
        invariants: dict[str, tuple[str, tuple[FieldView, ...]]] = {}
        bindings: dict[str, str] = {}
        for key, source in sources.items():
            if isinstance(source, Parameter) and source.state != self.state:
                declared = context.state_definition(source.state, "scan invariant needs a declared state")
                if (
                    declared.scope is not StateScope.INSTANCE
                    or source.per_row
                    or self.chunk is None
                    or optimizer is not None
                ):
                    raise ValueError(
                        "chunk recurrences read other InstanceState fields as invariants; remove per_row and optimizer"
                    )
                views = tuple(
                    context.binding.field_views[
                        source.state, f"{source.field}_{target.layer}" if declared.layers else source.field
                    ]
                    for target in targets
                )
                name = f"__invariant_{key}"
                if name in fields:
                    raise ValueError("rename fields starting with __invariant_; pLLM reserves them")
                invariants[name] = (source.state, views)
                bindings[key] = name
            elif isinstance(source, Operand):
                bindings[key] = source.name
            else:
                bindings[key] = source.in_forward_binding(self.state, fields)
        invariant_names = tuple(
            key
            for key, source in sources.items()
            if (isinstance(source, Parameter) and source.state != self.state) or bindings[key] == "weight"
        )
        return bindings, invariants, invariant_names

    def description(self) -> MemberDeclaration:
        return MemberDeclaration(self.kind, tokens=self.tokens, visibility=self.visibility)

    def requirements(self) -> BackendRequirements:
        """Exact visibility makes the forward read chunks trained earlier in the same forward."""
        return BackendRequirements(
            parameter_targets=self.targets,
            exact_phases=TOKEN_PHASES[self.tokens] if self.visibility == "exact" else (),
        )

    @property
    def reads_taps(self) -> bool:
        """Its inputs are projection operands inside the forward."""
        return False

    def declare(self, operation: Operation) -> None:
        """Declare the request state's in-forward transition and the delta it applies at the target projections."""
        if operation.gradient is not None or operation.rows:
            raise ValueError("@pllm.train_in_forward updates its own state: remove rows= and @pllm.differentiate")
        if operation.phase is not When.IN_FORWARD:
            raise ValueError("@pllm.train_in_forward runs in Phase.IN_FORWARD; remove @pllm.on")
        context = operation.context
        definition = context.state_definition(self.state, "@pllm.train_in_forward needs a declared request state")
        if definition.scope is not StateScope.REQUEST:
            raise ValueError(f"@pllm.train_in_forward trains per request; declare {self.state!r} as pllm.RequestState")
        optimizer = definition.optimizer
        if isinstance(optimizer, AdamWConfig):
            raise ValueError(
                f"@pllm.train_in_forward trains {self.state!r} with SGD once per chunk; declare it with optimizer=pllm.SGD(lr), "
                "or with no optimizer when the method returns the next carry itself"
            )
        if optimizer is not None and self.chunk is None:
            raise ValueError(
                f"{self.state!r} has an optimizer, so @pllm.train_in_forward trains it per chunk; give chunk="
            )
        if self.flush is not None and optimizer is None:
            raise ValueError(f"flush='prompt_end' trains a short final chunk; declare {self.state!r} with an optimizer")
        fields = {
            key: replace(value, dtype=context.field_dtype(self.state, key, value.dtype).torch)
            for key, value in definition.field_specs().items()
        }
        effect = self.projection_delta(context.binding.dimensions.layer_count)
        bindings, invariants, invariant_names = self.bindings(operation, fields, effect.targets, optimizer)
        if set(fields) & set(get_args(OperandName)):
            raise ValueError(
                "rename the state fields that repeat an operand name (input, target, embedding, weight, prompt_end)"
            )
        admit_update(
            operation.function,
            fields,
            effect,
            context.binding.dimensions,
            bindings,
            {
                key: tuple(torch.empty(view.shape, device="meta") for view in views)
                for key, (_, views) in invariants.items()
            },
            gradient=optimizer is not None,
            chunked=self.chunk is not None,
        )
        function: Callable[..., UpdateOutput] = operation.function
        initialize = definition.init.initializer(context.binding, self.state, tuple(definition.fields))
        if self.chunk is not None:
            function, initialize = _chunked_in_forward(
                function,
                initialize,
                bindings,
                fields,
                effect,
                self.chunk,
                optimizer,
                self.flush,
                invariant_names,
                context.binding.dimensions,
            )
        declare_parameter_overlay(
            context.binding,
            operation.name,
            effect,
            fields,
            initialize=initialize,
            keys=definition.request_keys,
            phases=TOKEN_PHASES[self.tokens],
            identity=operation.identity,
            function=function,
            arguments=bindings,
            visibility=self.visibility,
            invariants=invariants,
            reads=tuple(
                context.states[name] for name in dict.fromkeys(resource for resource, _ in invariants.values())
            ),
        )


SITE_TENSORS: dict[str, int] = {"attention": 2, "mlp": 2, "both": 2, "value": 1, "final": 1}
# The places a site owns in each chosen layer.
_SITE_KINDS: dict[str, tuple[Literal["attention", "mlp", "value"], ...]] = {
    "attention": ("attention",),
    "mlp": ("mlp",),
    "both": ("attention", "mlp"),
    "value": ("value",),
}


def _site_inputs(
    function: Callable[..., torch.Tensor], leading: int, context: WorkloadContext, method: str
) -> tuple[tuple[str, SiteInput], ...]:
    """The marker inputs after a method's `leading` site tensors (the stream and branch output, or value rows)."""
    parameters = tuple(signature(function, eval_str=False).parameters.items())
    if len(parameters) < leading or any(
        parameter.default is not SignatureParameter.empty for _, parameter in parameters[:leading]
    ):
        tensors = ("x", "y") if leading == 2 else ("x",)
        raise TypeError(f"{method}: take the site tensors {', '.join(tensors)} first, without defaults, then markers")
    inputs: list[tuple[str, SiteInput]] = []
    for name, parameter in parameters[leading:]:
        if not isinstance(parameter.default, InputMarker):
            raise TypeError(f"{name}: give every input after the site tensors a source as its default")
        inputs.append((name, parameter.default.bind(context).site_input(context)))
    return tuple(inputs)


@dataclass(frozen=True, kw_only=True)
class CustomAttention(Effect):
    kind: ClassVar[str] = "custom_attention"
    layers: tuple[int, ...] | None
    keep: Keep
    window: int
    source: int
    sink: bool
    phase: ClassVar[When | None] = When.IN_FORWARD
    model_effect: ClassVar[bool] = True

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(custom_attention=True, kv_layouts=self.layouts())

    @property
    def reads_taps(self) -> bool:
        """It reads state and earlier attention facts inside the forward."""
        return False

    @property
    def declared_first(self) -> bool:
        """Methods reading pllm.attention(...) facts read the attention method this registers."""
        return True

    def layouts(self) -> tuple[KVLayout, ...]:
        """Validate storage declarations at the authoring boundary."""
        layers, window, source, sink = (
            (-1,) if self.layers is None else self.layers,
            self.window,
            self.source,
            self.sink,
        )
        if self.keep not in ("all", "window", "none"):
            raise ValueError('custom_attention keep must be "all", "window" or "none"')
        if self.keep == "window":
            if window < 1 or source:
                raise ValueError('keep="window" needs window > 0 and source=0')
        elif window or sink:
            raise ValueError('window and sink need keep="window"')
        if self.keep == "none":
            if source < 1 or not layers or any(layer < source for layer in layers):
                raise ValueError(
                    'keep="none" needs explicit layers and source >= 1 pointing to an earlier layer; use @pllm.custom_forward with kv="peek" for top-layer sources'
                )
        elif source:
            raise ValueError('source needs keep="none"')
        return tuple(KVLayout(layer, self.keep, window, source, sink) for layer in layers)

    def declare(self, operation: Operation) -> None:
        context, binding = operation.context, operation.context.binding
        if binding.host or operation.gradient is not None or operation.rows or operation.phase is not When.IN_FORWARD:
            raise ValueError(
                "@pllm.custom_attention runs every model row; remove rows=, on and differentiate, and use make_llm"
            )
        count = binding.dimensions.layer_count
        layers = tuple(range(count)) if self.layers is None else self.layers
        if (
            not layers
            or any(type(layer) is not int or not 0 <= layer < count for layer in layers)
            or tuple(sorted(set(layers))) != layers
        ):
            raise ValueError(f"@pllm.custom_attention needs sorted distinct layers in 0..{count - 1}")
        parameters = tuple(signature(operation.function, eval_str=False).parameters.items())
        if len(parameters) < 2 or any(p.default is not SignatureParameter.empty for _, p in parameters[:2]):
            raise TypeError(f"{operation.name}: take q and kv first, without defaults, then markers")
        inputs: list[tuple[str, StateInput | DepthInput | AttentionFact]] = []
        for name, parameter in parameters[2:]:
            source = parameter.default
            if not isinstance(source, InputMarker):
                raise TypeError(f"{name}: give every attention input after q and kv a marker default")
            if isinstance(source, Attention):
                value = AttentionFact(source.layer, source.fact)
            else:
                value = source.bind(context).site_input(context)
                if not isinstance(value, (StateInput, DepthInput)):
                    raise TypeError(
                        "@pllm.custom_attention reads instance parameters, modules, depth and earlier attention facts"
                    )
                if isinstance(value, StateInput) and value.request:
                    raise ValueError(
                        f"{name} reads RequestState {value.resource!r}, but @pllm.custom_attention runs for every "
                        "scheduled token with instance-wide inputs; read an InstanceState and select per request "
                        "inside the math"
                    )
            inputs.append((name, value))
        binding.attention.add(AttentionMethod(operation.name, layers, operation.function, tuple(inputs)))
        binding.attention.layouts += expand_layouts(self.layouts(), count)
        binding.model_reads(tuple(source.resource for _, source in inputs if isinstance(source, StateInput)))


@dataclass(frozen=True, kw_only=True)
class CustomResidual(Effect):
    kind: ClassVar[str] = "custom_residual"

    site: Site
    layers: tuple[int, ...] | None
    read: str
    streams: int
    state: str
    chunk: int | None

    phase: ClassVar[When | None] = When.IN_FORWARD
    model_effect: ClassVar[bool] = True

    def accumulates(self, operation: Operation, field: str) -> bool:
        return bool(self.state) and self.chunk is not None

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(custom_residual=True, stateful_residual=bool(self.state))

    @property
    def declared_first(self) -> bool:
        """A stateful method allocates the carry state that other methods may read."""
        return bool(self.state)

    @property
    def reads_taps(self) -> bool:
        """It reads state and earlier values of the forward, never tapped rows."""
        return False

    def places(self, count: int) -> tuple[Place, ...]:
        """The owned places: the chosen layers' sublayer adds or value rows, or the final stream."""
        if self.site == "final":
            if self.layers is not None:
                raise ValueError('@pllm.custom_residual(site="final") rewrites the one final stream; remove layers=')
            return (("final", count),)
        layers = tuple(range(count)) if self.layers is None else self.layers
        if (
            not layers
            or tuple(sorted(set(layers))) != layers
            or not all(type(layer) is int and 0 <= layer < count for layer in layers)
        ):
            raise ValueError(f"@pllm.custom_residual(layers={layers!r}): give sorted distinct layers in 0..{count - 1}")
        return tuple((kind, layer) for layer in layers for kind in _SITE_KINDS[self.site])

    def declare(self, operation: Operation) -> None:
        """Own the chosen sites' residual updates (or value rows) with the method; it reads states and earlier values."""
        context = operation.context
        if operation.gradient is not None or operation.rows:
            raise ValueError(
                "@pllm.custom_residual rewrites every row of its sites: remove rows= and @pllm.differentiate"
            )
        if operation.phase is not When.IN_FORWARD:
            raise ValueError("@pllm.custom_residual runs inside the forward; remove @pllm.on")
        if context.binding.host:
            raise ValueError("@pllm.custom_residual acts inside a model forward; compile this workload with make_llm")
        places = self.places(context.binding.dimensions.layer_count)
        memory = None if not self.state else self.memory(operation, places)
        reader = context.method(self.read) if self.read else None
        rewrite = ResidualRewrite(
            operation.name,
            self.site,
            places,
            operation.function,
            _site_inputs(operation.function, SITE_TENSORS[self.site], context, operation.name),
            reader,
            () if reader is None else _site_inputs(reader, 1, context, self.read),
            self.streams,
            memory,
        )
        if rewrite.streams < 1 or (self.read and self.site in ("value", "final")):
            raise ValueError("@pllm.custom_residual takes streams >= 1, and read= only at attention/MLP sites")
        if memory is not None and (
            rewrite.read is not None
            or any(
                isinstance(source, (EarlierInput, PatchInput, ResultInput)) or source.request and source != memory.carry
                for _, source in rewrite.inputs
                if not isinstance(source, DepthInput)
            )
        ):
            raise ValueError(
                f"a stateful @pllm.custom_residual reads its carry pllm.parameter({self.state!r}, ...), instance "
                "state and pllm.depth(), without read="
            )
        context.binding.sites.add(rewrite)
        context.binding.model_reads(
            tuple(
                source.resource
                for _, source in (*rewrite.inputs, *rewrite.read_inputs)
                if isinstance(source, StateInput) and (memory is None or source != memory.carry)
            )
        )

    def memory(self, operation: Operation, places: tuple[Place, ...]) -> SiteMemory:
        """Declare the request state the method carries chunk by chunk, with the framework's unfinished-chunk rows,
        and the transition that advances it inside the forward."""
        context, b = operation.context, operation.context.binding
        definition = context.state_definition(self.state, "@pllm.custom_residual(state=...) needs a declared state")
        if (
            definition.scope is not StateScope.REQUEST
            or definition.optimizer is not None
            or len(definition.fields) != 1
            or self.chunk is None
            or len(places) != 1
            or places[0][0] not in ("attention", "mlp")
        ):
            raise ValueError(
                f"@pllm.custom_residual(state={self.state!r}) needs one attention or MLP site of one layer, chunk=, "
                "and a pllm.RequestState with one field, its carry, and no optimizer"
            )
        chunk = self.chunk
        if chunk < 1:
            raise ValueError("@pllm.custom_residual(chunk=) must be positive")
        ((name, shape),) = definition.fields.items()
        if isinstance(shape, Field):
            raise TypeError(f"{name}: give the carry a fixed shape or a function of the model dimensions")
        rows: Shape = lambda dimensions: (max(1, chunk - 1), dimensions.hidden_size)  # noqa: E731
        initialize = definition.init.initializer(context.binding, self.state, (name,))
        if initialize is not None:
            initialize = partial(_author_fields, initialize, (name,))
        state, ports = declare_parameters(
            b,
            self.state,
            (
                LayoutField(name, shape, context.field_dtype(self.state, name, definition.dtype), definition.alignment),
                LayoutField("_pending_rows_x", rows, lambda dimensions: dimensions.dtype),
                LayoutField("_pending_rows_y", rows, lambda dimensions: dimensions.dtype),
                LayoutField("_pending_length", (), TensorDType.INT64),
            ),
            optimizer=None,
            initialize=initialize,
            scope=StateScope.REQUEST,
            keys=definition.request_keys,
        )
        context.states[self.state] = state
        context.parameters.update({(self.state, field): port for field, port in ports.items()})
        declare_site_memory(b, operation.name, state)
        views = b.field_views
        return SiteMemory(
            StateInput(state.declaration.resource, views[self.state, name], True),
            (
                views[self.state, "_pending_rows_x"],
                views[self.state, "_pending_rows_y"],
                views[self.state, "_pending_length"],
            ),
            chunk,
        )
