"""Cold Python authoring boundary. Reflection stops at the original typed graph."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field, fields, is_dataclass, replace
from enum import Enum
from functools import partial
from inspect import Parameter as SignatureParameter, signature
from math import prod
from types import FunctionType, MethodType
from typing import Any, ClassVar, Literal, TypeVar, get_args

import torch

from pllm.authoring import BackendRequirements, BackendWorkload
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import (
    ActivationPatchOperator,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.codec import boundary_digest
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.host_batch import CausalTrainBatch, WeightedCEBatch, WeightedRowBatch, validate_host_batch
from pllm.contracts.member import MemberDeclaration
from pllm.contracts.parameter_overlay import (
    FieldView,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    Projection,
    ProjectionDelta,
)
from pllm.contracts.request_identity import RequestKeyField, ordered_key_fields
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    HiddenInputSource,
    MathContract,
    SamplerFeedbackInputSource,
    SchedulerTimeline,
    SemanticGraph,
    SemanticOutputKind,
    SemanticTask,
    StateEffectKind,
    StateScope,
    TaskId,
    TaskResultInputSource,
    TaskSymbol,
    TimingConstraint,
    TrainingMode,
)
from pllm.contracts.state_feed import FeedBinding, FeedOrder
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.contracts.visibility import VisibilityKind
from pllm.execution.binding_base import SGD as SGD, Field as Field, layout_field, overlay_fields
from pllm.execution.causal_batch import CausalBinder, CausalModel, CausalPort, TrainingBatch, declare_batch_train
from pllm.execution.checkpoint import PretrainedFile, initialize_checkpoint
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.declaration import Call, Dimensions, LayoutField, Shape, State, When
from pllm.execution.freeze import (
    fingerprint,
    freeze,
    members as _members,
    owner_snapshot,
    owner_values,
    reachable_functions,
)
from pllm.execution.host_runtime import HostConfig, bind_host
from pllm.execution.model_effect_binding import (
    declare_activation_patch,
    declare_custom_forward,
    declare_parameter_overlay,
    declare_site_memory,
    import_update,
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
from pllm.execution.module_graph import ModuleCall
from pllm.execution.module_state import (
    ModuleFactory,
    initial_values,
    parameter_count,
    skeleton,
)
from pllm.execution.overlay_task import ChunkUpdate, GradientUpdate, overlay_gradient
from pllm.execution.position_state import PositionState as PositionState
from pllm.execution.row_task_binding import (
    declare_logits,
    declare_parameters,
    declare_tensor,
    declare_weighted_train,
    parameter_rows,
    sampled_token_port,
    state_metadata,
    validate_gradients,
)
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.step_input import StepInput
from pllm.execution.task_memory import derive_host_resources, derive_resources, shape_site_results
from pllm.execution.tensor_binding import TensorBinding, TensorOutput as TensorOutput
from pllm.execution.tensor_frame import (
    TensorFrame,
    TensorPort,
    Tensors,
    UpdateOutput as UpdateOutput,
    feedback_port,
    hidden_port,
    input_tokens_port,
    layer_logits_port,
    result_port,
    sample_index_port,
)
from pllm.execution.tensor_program import PoolSGD, TaskOutput, TensorBatch
from pllm.execution.tensor_task import TensorExport
from pllm.execution.token_program import Over, Tokens
from pllm.execution.trace import TraceEvent
from pllm.kernels.adamw import AdamWConfig

F = TypeVar("F", bound=Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | UpdateOutput | None])
Initializer = Callable[[Tensors, torch.Generator], None]
C = TypeVar("C")
# How @pllm.train averages per-row losses: over all rows, or per state and then summed over states.
TrainReduction = Literal["mean", "state_mean"]
L = TypeVar("L", bound=str)
# The tokens whose projection operands train a @pllm.train_in_forward state.
_TOKEN_PHASES: dict[str, tuple[HiddenPhase, ...]] = {
    "prefill": (HiddenPhase.PREFILL,),
    "decode": (HiddenPhase.DECODE,),
    "all": (HiddenPhase.PREFILL, HiddenPhase.DECODE),
}
# How an adapter's fields change a projection: a dense (out_features, in_features) delta, low-rank factors
# (rank, in_features) and (out_features, rank), or one (width, width) block per attention head.
AdapterOperator = Literal["dense", "low_rank", "per_head"]
_OPERATORS: dict[str, ParameterOverlayOperator] = {
    "dense": ParameterOverlayOperator.DENSE_ADD,
    "low_rank": ParameterOverlayOperator.LOW_RANK_ADD,
    "per_head": ParameterOverlayOperator.HEAD_ADD,
}
# The fields that key a request state: the vLLM request, the prompt, the sample of a prompt, a user route.
KeyField = Literal["request_id", "prompt_index", "sample_index", "route"]


class Phase(Enum):
    BEFORE_FORWARD = When.BEFORE_FORWARD
    AFTER_FORWARD = When.AFTER_FORWARD
    BEFORE_SAMPLE = When.BEFORE_SAMPLE
    AFTER_SAMPLE = When.AFTER_SAMPLE
    IN_FORWARD = When.IN_FORWARD
    AUTOMATIC = When.AUTOMATIC


class _Source:
    """An argument marker owns its graph input and derivative eligibility."""

    differentiable: ClassVar[bool] = True

    def qualify(self, context: _Reflection) -> _Source:
        """Resolve component-local references once at the declaration boundary."""
        return self

    def port(self, context: _Reflection) -> TensorPort:
        raise TypeError(f"unsupported tensor source: {type(self).__name__}")

    def training_port(self, context: _Reflection, schema: str) -> CausalPort:
        raise ValueError("causal training inputs must name Batch, Parameter or causal_model ports")

    def in_forward_binding(self, state: str, fields: Mapping[str, Field]) -> str:
        raise TypeError("@pllm.train_in_forward inputs are pllm.operand(...) or pllm.parameter fields of its own state")

    def batch(self, state: str, definition: InstanceState) -> tuple[str, str]:
        raise ValueError("batch training inputs must name its batch tensors or own parameters")

    def site_input(self, context: _Reflection) -> SiteInput:
        raise TypeError(
            "@pllm.custom_residual methods read pllm.parameter, pllm.module, pllm.depth, pllm.value, pllm.residual and pre-forward pllm.result"
        )


@dataclass(frozen=True)
class Attr:
    """A reference to a workload attribute by dotted path, e.g. pllm.Attr("config.chunk_size"), resolved once when
    the workload is built."""

    name: str

    @staticmethod
    def resolve(value: object, owner: object) -> object:
        """Resolve an attribute path against the snapshotted owner."""
        match value:
            case Attr(name=name):
                for part in name.split("."):
                    try:
                        owner = object.__getattribute__(owner, part)
                    except AttributeError:
                        raise ValueError(
                            f"pllm.Attr({name!r}): {type(owner).__name__} has no attribute {part!r}"
                        ) from None
                return owner
            case _:
                return value


@dataclass(frozen=True)
class Hidden(_Source):
    layer: int | Attr = 0
    site: HiddenSite = HiddenSite.BLOCK_OUTPUT
    phase: HiddenPhase = HiddenPhase.DECODE

    def resolve(self, owner: Workload) -> HiddenInputSource:
        """Resolve the model dependency without building a tensor binding."""
        return HiddenInputSource(_setting(self.layer, owner, int), self.site, self.phase)

    def port(self, context: _Reflection) -> TensorPort:
        return hidden_port(self.resolve(context.owner))

    def site_input(self, context: _Reflection) -> SiteInput:
        """Inside a custom residual, pllm.residual reads the same token's stream earlier in this forward."""
        if self.site not in (HiddenSite.RESIDUAL, HiddenSite.ATTENTION_RESIDUAL):
            return super().site_input(context)
        layer = _setting(self.layer, context.owner, int) % context.binding.dimensions.layer_count
        return EarlierInput("residual" if self.site is HiddenSite.RESIDUAL else "attention_residual", layer)


@dataclass(frozen=True)
class LayerLogits(_Source):
    layers: tuple[int, ...] | Attr

    def resolve(self, owner: Workload) -> HiddenInputSource:
        """All layer-logit readers of a workload share one input holding every layer any of them names, for every
        row the sampler draws (phase NONE: prompt ends and decode rows)."""
        return HiddenInputSource(-1, HiddenSite.LAYER_LOGITS, HiddenPhase.NONE)

    def layer_tuple(self, owner: Workload) -> tuple[int, ...]:
        layers = _setting(self.layers, owner, tuple)
        if not layers or any(type(layer) is not int for layer in layers):
            raise TypeError(f"pllm.layer_logits({layers!r}) takes a non-empty tuple of layer indices")
        return layers

    def port(self, context: _Reflection) -> TensorPort:
        probe = context.binding.probe
        if probe is None:
            raise ValueError("pllm.layer_logits reads the model's LM head; compile this workload with make_llm")
        count = context.binding.dimensions.layer_count
        columns = tuple(probe.layer_logits.index(layer % count) for layer in self.layer_tuple(context.owner))
        return layer_logits_port(self.resolve(context.owner), columns)


@dataclass(frozen=True)
class SampleIndex(_Source):
    limit: int | Attr | None = None

    differentiable: ClassVar[bool] = False

    def port(self, context: _Reflection) -> TensorPort:
        return sample_index_port(context.binding.sampler_rows())


@dataclass(frozen=True)
class Batch(_Source):
    field: str

    def training_port(self, context: _Reflection, schema: str) -> CausalPort:
        return CausalPort(
            context.binding.input("training", self.field, schema),
            self.training_value,
        )

    def training_value(self, batch: TrainingBatch, parameters: Tensors) -> torch.Tensor | None:
        """Hand training math floating inputs in at least FP32, retaining integer IDs and absent masks."""
        value = batch.tensor(self.field)
        if value is None or not value.is_floating_point():
            return value
        return value.to(torch.promote_types(value.dtype, torch.float32))

    def batch(self, state: str, definition: InstanceState) -> tuple[str, str]:
        return "batch", self.field


@dataclass(frozen=True)
class Parameter(_Source):
    state: str
    field: str
    per_row: bool = False

    def qualify(self, context: _Reflection) -> Parameter:
        return replace(self, state=context.qualify(self.state))

    def training_port(self, context: _Reflection, schema: str) -> CausalPort:
        context.parameter(self.state, self.field)
        view = context.binding.field_views[self.state, self.field]
        return CausalPort(
            context.states[self.state].read(), lambda batch, parameters: view.read(parameters["parameters"])
        )

    def port(self, context: _Reflection) -> TensorPort:
        parameter = context.parameter(self.state, self.field)
        return parameter_rows(parameter) if self.per_row else parameter

    def in_forward_binding(self, state: str, fields: Mapping[str, Field]) -> str:
        if self.state == state and self.field in fields:
            return self.field
        return super().in_forward_binding(state, fields)

    def batch(self, state: str, definition: InstanceState) -> tuple[str, str]:
        if self.state != state or self.field not in definition.fields or self.per_row:
            return super().batch(state, definition)
        return "parameter", self.field

    def site_input(self, context: _Reflection) -> StateInput:
        context.parameter(self.state, self.field)
        return StateInput(
            context.states[self.state].declaration.resource,
            context.binding.field_views[self.state, self.field],
            context.state_definitions[self.state].scope is StateScope.REQUEST,
        )


@dataclass(frozen=True)
class _CausalModelMarker(_Source):
    """An adapter's differentiable views adapted to a worker-local model callable."""

    name: str

    def qualify(self, context: _Reflection) -> _CausalModelMarker:
        return replace(self, name=context.qualify(self.name))

    def training_port(self, context: _Reflection, schema: str) -> CausalPort:
        adapter = context.adapters.get(self.name)
        if adapter is None:
            raise ValueError(f"pllm.causal_model({self.name!r}) must name a pllm.model.Adapter of the workload")
        state = context.states[adapter.state]
        bindings = context.adapter_bindings(adapter)
        forward = None if context.binding.causal is None else context.binding.causal(bindings)
        return CausalPort(
            state.read(),
            lambda batch, parameters: CausalModel(
                forward, tuple(view.read(parameters["parameters"]) for binding in bindings for view in binding.fields)
            ),
        )


@dataclass(frozen=True)
class _Module(Parameter):
    """A module state, handed to the method as a callable with the module's interface."""

    field: str = "parameters"

    def port(self, context: _Reflection) -> TensorPort:
        return replace(super().port(context), adapt=self.adapt(context))

    def adapt(self, context: _Reflection) -> Callable[[torch.Tensor], ModuleCall]:
        return partial(ModuleCall, self.skeleton(context), per_row=self.request(context))

    def skeleton(self, context: _Reflection) -> torch.nn.Module:
        definition = context.state_definition(self.state, f"{self.state}: pllm.module needs a declared module state")
        if definition.module is None:
            raise TypeError(f"{self.state}: pllm.module needs a state declared with module=")
        return skeleton(definition.module, context.binding.dimensions)

    def request(self, context: _Reflection) -> bool:
        return context.state_definition(self.state, "pllm.module needs a declared state").scope is StateScope.REQUEST

    def site_input(self, context: _Reflection) -> StateInput:
        return replace(super().site_input(context), module=self.skeleton(context))


@dataclass(frozen=True)
class Carry:
    """Where a training method's pllm.carry input is written at publication: one carried field of its own state,
    which the next step reads as the input version. Per request the last selected row wins."""

    state: str
    field: str

    def declare(self, state: str, update: State | None) -> LayoutFieldId:
        """Check the carried field against the training state's fields."""
        if self.state != state:
            raise ValueError("pllm.carry writes into the training method's own state")
        assert update is not None
        if self.field not in {field.name for field in update.carry_fields}:
            raise ValueError(f"{self.field}: pllm.carry names a field declared in the state's carry=")
        return update.field(self.field)


@dataclass(frozen=True)
class StateIndex(_Source):
    state: str

    differentiable: ClassVar[bool] = False

    def qualify(self, context: _Reflection) -> StateIndex:
        return replace(self, state=context.qualify(self.state))

    def port(self, context: _Reflection) -> TensorPort:
        return state_metadata(context.state(self.state, "pllm.index needs a declared state"), version=False)


@dataclass(frozen=True)
class StateVersion(_Source):
    state: str

    differentiable: ClassVar[bool] = False

    def qualify(self, context: _Reflection) -> StateVersion:
        return replace(self, state=context.qualify(self.state))

    def port(self, context: _Reflection) -> TensorPort:
        return state_metadata(context.state(self.state, "pllm.version needs a declared state"), version=True)


@dataclass(frozen=True)
class Result(_Source):
    operation: str

    def qualify(self, context: _Reflection) -> Result:
        return replace(self, operation=context.qualify(self.operation))

    def port(self, context: _Reflection) -> TensorPort:
        return result_port(TaskResultInputSource(TaskSymbol(self.operation), "value"))

    def site_input(self, context: _Reflection) -> SiteInput:
        operation = next(op for op in context.operations if op.name == self.operation)
        if operation.phase is not Phase.BEFORE_FORWARD or type(operation.effect) is not _Effect:
            raise ValueError(f"{self.operation}: custom residuals read results of read-only BEFORE_FORWARD methods")
        return ResultInput(TaskSymbol(self.operation))


@dataclass(frozen=True)
class InputTokens(_Source):
    """The request token ids read by each row, with a fixed number of lookbacks."""

    history: int = 0
    differentiable: ClassVar[bool] = False

    def port(self, context: _Reflection) -> TensorPort:
        return input_tokens_port(self.history)


@dataclass(frozen=True)
class Depth(_Source):
    differentiable: ClassVar[bool] = False

    def site_input(self, context: _Reflection) -> SiteInput:
        return DepthInput()


@dataclass(frozen=True)
class Value(_Source):
    layer: int | Attr

    def site_input(self, context: _Reflection) -> SiteInput:
        return EarlierInput("value", _setting(self.layer, context.owner, int) % context.binding.dimensions.layer_count)


@dataclass(frozen=True)
class BaseLogits(_Source):
    def port(self, context: _Reflection) -> TensorPort:
        return feedback_port("base_logits")


@dataclass(frozen=True)
class SampledTokens(_Source):
    differentiable: ClassVar[bool] = False

    def port(self, context: _Reflection) -> TensorPort:
        return sampled_token_port()


OperandName = Literal["input", "target", "embedding", "weight", "prompt_end"]


@dataclass(frozen=True)
class Operand(_Source):
    """A named model operand, selected when the workload is declared."""

    name: OperandName | Attr


@dataclass(frozen=True)
class _Carried(_Source):
    """An input whose value a training task also writes into a carried field."""

    source: _Source
    carry: Carry

    def qualify(self, context: _Reflection) -> _Carried:
        return replace(
            self,
            source=self.source.qualify(context),
            carry=replace(self.carry, state=context.qualify(self.carry.state)),
        )


# Inputs are declared as parameter defaults: `h: torch.Tensor = pllm.hidden(0)`. The factories return the
# source marker typed `Any`, so the annotation can name the value the method body receives.


def hidden(layer: int | Attr = 0, *, tokens: Literal["decode", "prefill"] = "decode") -> Any:
    """The MLP output before the residual add in decoder layer `layer` (negative from the end).

    One row per request: the latest decode token, or with tokens="prefill" the last prompt token of completed prefill.
    """
    return Hidden(layer, HiddenSite.BLOCK_OUTPUT, HiddenPhase(tokens))


def residual(
    layer: int | Attr,
    *,
    after: Literal["layer", "attention"] = "layer",
    tokens: Literal["decode", "prefill"] = "decode",
) -> Any:
    """The residual stream after decoder layer `layer` (negative from the end): the embedding plus every earlier
    sublayer's output, which the next sublayer normalizes. This is what steering vectors, sparse autoencoders and the
    logit lens read; pllm.hidden(layer) is only the layer's MLP output before it is added to this stream.
    after="attention" reads the stream between the layer's attention and its MLP. One row per request, chosen as in
    pllm.hidden."""
    if after not in ("layer", "attention"):
        raise ValueError(f'pllm.residual(after={after!r}): use "layer" or "attention"')
    return Hidden(
        layer, HiddenSite.RESIDUAL if after == "layer" else HiddenSite.ATTENTION_RESIDUAL, HiddenPhase(tokens)
    )


def layer_logits(layers: tuple[int, ...] | Attr) -> Any:
    """The model's final norm and LM head applied to the residual stream after each of `layers` (negative from the
    end), as float32 [rows, len(layers), vocab]: the logit lens. The last layer's logits are the model's own. One row
    for each token the sampler draws: a prompt's last token and every decode token."""
    return LayerLogits(layers)


def final_hidden(*, tokens: Literal["decode", "prefill"] = "decode") -> Any:
    """The hidden state after the model's final norm, which the LM head reads (Hugging Face's last_hidden_state);
    one row per request, chosen as in pllm.hidden."""
    return Hidden(-1, HiddenSite.MODEL_OUTPUT, HiddenPhase(tokens))


def parameter(state: str, field: str, *, per_row: bool = False) -> Any:
    """One tensor field of a declared state, aligned to the operation's rows for request state."""
    return Parameter(state, field, per_row)


def causal_model(adapter: str) -> Any:
    """The causal LM with the named pllm.model.Adapter applied, as a pllm.model.CausalModel; .base skips it."""
    return _CausalModelMarker(adapter)


def module(state: str) -> Any:
    """A state declared with `module=`, called like the module; request state uses each row's own parameters."""
    return _Module(state)


def version(state: str) -> Any:
    """How many updates each row's state has received (0 before the first update)."""
    return StateVersion(state)


def index(state: str) -> Any:
    """Each row's slot in the state pool, for indexing pooled tensors."""
    return StateIndex(state)


def sample_index(limit: int | Attr | None = None) -> Any:
    """The sample index of each row's request: 0 for a prompt's first sample.

    With `limit`, generate rejects requests whose sample index is `limit` or more.
    """
    return SampleIndex(limit)


def result(operation: str) -> Any:
    """The value another operation of this workload returned in the same step."""
    return Result(operation)


def depth() -> Any:
    """Inside a @pllm.custom_residual method, the index of the place it runs at, a Python int: the layer, or
    2 * layer (+ 1 for the MLP) with site="both"."""
    return Depth()


def value(layer: int | Attr) -> Any:
    """Inside a @pllm.custom_residual method, the value rows [rows, kv_heads, head_dim] layer `layer` computed for the
    same token in this forward (after its own value rewrite)."""
    return Value(layer)


def base_logits() -> Any:
    """The model's logits before any workload adjustment."""
    return BaseLogits()


def input_tokens(*, history: int = 0) -> Any:
    """Int64 [rows, history + 1] token ids: current token first, then lookbacks; -1 before the request starts.

    Results consumed by custom_residual use every scheduled token; other methods use sampled rows.
    Requires runner="mrv2".
    """
    if type(history) is not int or history < 0:
        raise ValueError("pllm.input_tokens(history=) needs a nonnegative integer")
    return InputTokens(history)


def sampled_tokens() -> Any:
    """The tokens the sampler drew this step."""
    return SampledTokens()


def operand(name: OperandName | Attr) -> Any:
    """A projection input/target/weight, input token embedding, or scalar prompt-end bool.

    An Attr selects the operand name at declaration time. Embeddings are the residual stream
    entering the first decoder layer, before its input norm, aligned with each projection row.
    """
    return Operand(name)


def carry(source: _Source, *, state: str, field: str) -> Any:
    """Read `source` and, when the training update publishes, write its value into the carried `state.field`.

    The next step reads that field as its input version; per request the last selected row wins.
    """
    return _Carried(source, Carry(state, field))


def batch(
    field: Literal["hidden", "base_logits", "labels", "weights", "targets", "input_ids", "attention_mask"],
) -> Any:
    """One field of the host batch passed to generate.

    The fields a @pllm.train method reads pick its batch: WeightedRowBatch (hidden, targets, weights) or
    CausalTrainBatch (input_ids, labels, attention_mask, weights) take one AdamW step;
    WeightedCEBatch (hidden, base_logits, labels, weights) runs SGD steps per state.
    """
    return Batch(field)


@dataclass(frozen=True)
class InstanceState:
    """State shared by requests; pretrained= supplies safetensors starting values instead of initialize=.

    Plain states load tensor names matching their fields. Module states load a strict state_dict on meta,
    skipping random initialization. Values convert to the declared storage dtype, including trainable states.
    """

    fields: Mapping[str, Shape | Field] = field(default_factory=lambda: dict[str, Shape | Field]())
    # How @pllm.train or @pllm.train_in_forward updates the state. Training on request rows or on WeightedRowBatch
    # and CausalTrainBatch fields takes pllm.AdamW; WeightedCEBatch training and @pllm.train_in_forward take pllm.SGD.
    optimizer: AdamWConfig | SGD | None = field(default=None, kw_only=True)
    # Run once per resource when binding; every admission copies the same seeded initial tensors.
    initialize: Callable[[Tensors, torch.Generator], None] | None = None
    shape: Shape | None = None
    dtype: torch.dtype | Literal["auto"] | None = "auto"
    alignment: int = field(default=16, kw_only=True)
    # A fed state is filled by a producer process (VllmLlm.feed): "ordered" consumes every publication in order,
    # "latest" only the newest.
    feed: Literal["ordered", "latest"] | None = field(default=None, kw_only=True)
    # Plain per-request tensors carried across steps (written by pllm.carry); never optimizer-managed.
    carry: Mapping[str, Shape] = field(default_factory=lambda: dict[str, Shape](), kw_only=True)
    # An `nn.Module` built from the model's dimensions; its parameters become the state, initialized as the
    # module initializes them (every request starts from the same values).
    module: ModuleFactory | None = field(default=None, kw_only=True)
    imported: bool = field(default=False, kw_only=True)
    pretrained: str = field(default="", kw_only=True)
    checkpoint: Mapping[str, str] = field(default_factory=lambda: dict[str, str](), kw_only=True)
    layers: tuple[int, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if self.pretrained and (self.initialize is not None or self.checkpoint or self.feed or self.imported):
            raise ValueError(
                "pretrained= supplies starting values; drop initialize=, checkpoint=, feed= and imported=True"
            )
        if self.checkpoint and (
            self.scope is not StateScope.INSTANCE
            or self.optimizer is not None
            or self.initialize is not None
            or self.module is not None
            or self.feed
            or self.imported
            or set(self.checkpoint) != set(self.fields)
        ):
            raise ValueError("checkpoint state needs an immutable InstanceState with a tensor name for every field")
        if self.layers and (
            self.scope is not StateScope.INSTANCE
            or tuple(sorted(set(self.layers))) != self.layers
            or self.layers[0] < 0
        ):
            raise ValueError("layers needs sorted unique nonnegative indices on an InstanceState")
        if self.feed not in (None, "ordered", "latest"):
            raise ValueError(f'feed={self.feed!r}: use "ordered", "latest" or None')
        if self.module is not None:
            if self.fields or self.shape is not None or self.initialize is not None or self.feed:
                raise ValueError(
                    "a module state takes its fields and initial values from the module; drop fields/shape/initialize"
                )
            object.__setattr__(self, "fields", {"parameters": parameter_count(self.module)})
        if self.feed is not None and (
            self.scope is not StateScope.INSTANCE
            or self.optimizer is not None
            or self.initialize is not None
            or self.carry
            or self.imported
        ):
            raise ValueError("fed state needs InstanceState without an optimizer, initializer or carry fields")
        if self.shape is not None:
            object.__setattr__(self, "fields", {"value": self.shape})
        if self.carry and self.optimizer is None:
            raise ValueError("carry fields ride on the training transition; give the state an optimizer")
        if set(self.carry) & set(self.fields):
            raise ValueError("rename the carry fields that repeat a parameter field name")

    scope: ClassVar[StateScope] = StateScope.INSTANCE

    @property
    def request_keys(self) -> tuple[RequestKeyField, ...]:
        return ()

    def declare(self, name: str, context: _Reflection) -> None:
        """Allocate tensor parameters; the layout of fields an adapter reads follows its targets."""
        pretrained = PretrainedFile.read(self.pretrained, name) if self.pretrained else None
        context.signature.append(
            (
                name,
                boundary_digest(fingerprint(self), tuple(self.fields), "" if pretrained is None else pretrained.sha256),
            )
        )
        context.binding.state_sources[name] = self.source() if pretrained is None else pretrained.source
        if pretrained is not None:
            context.pretrained[name] = pretrained
        if name in context.in_forward_states or name in context.states:
            return
        if self.feed is not None:
            if context.binding.feeds:
                raise ValueError(f"{context.binding.feeds[0].resource} and {name}: declare one fed state per workload")
            context.binding.feeds = (FeedBinding(name, FeedOrder(self.feed), tuple(self.fields)),)
        fields = self.layout_fields(name, context)
        context.states[name], ports = declare_parameters(
            context.binding,
            name,
            fields,
            optimizer=self.optimizer,
            initialize=context.initializer(name, tuple(item.name for item in fields) + tuple(self.carry))
            if self.module is None
            else None,
            scope=self.scope,
            keys=self.request_keys,
            carry=tuple(
                LayoutField(key, shape, context.field_dtype(name, key, self.dtype), self.alignment)
                for key, shape in self.carry.items()
            ),
        )
        context.parameters.update({(name, field): port for field, port in ports.items()})
        if self.checkpoint:
            context.binding.initializers[name] = partial(
                initialize_checkpoint, context.binding.checkpoint, self.checkpoint_names()
            )
        if self.module is not None:
            context.binding.initial_values[name] = partial(
                initial_values,
                self.module,
                dtype=context.field_dtype(name, "parameters", self.dtype).torch,
                pretrained=pretrained,
            )

    def source(self) -> str:
        """Describe where this state's version-zero values come from."""
        if self.imported:
            return (
                "imported: "
                + ("one row per prompt" if self.scope is StateScope.REQUEST else "field tensors")
                + " via generate(batch=pllm.io.StateImport(...)), applied before the request's first forward"
            )
        if self.feed:
            return "fed"
        if self.checkpoint:
            return "base checkpoint"
        if self.module is not None:
            return "module init"
        return "initialize" if self.initialize is not None else "zeros"

    def checkpoint_names(self) -> tuple[str, ...]:
        """Exact checkpoint tensor names, in the same order as the allocated fields."""
        if not self.checkpoint:
            return ()
        return (
            tuple(
                pattern.format(layer=layer)
                for layer in self.layers
                for pattern in (self.checkpoint[key] for key in self.fields)
            )
            if self.layers
            else tuple(self.checkpoint[key] for key in self.fields)
        )

    def layout_fields(self, name: str, context: _Reflection) -> tuple[LayoutField, ...]:
        """The allocated fields: plain shapes in the state's dtype, or pllm.Field values sized by the hidden width or,
        when a pllm.model.Adapter reads the state, once per adapter target."""
        if self.layers:
            if self.layers[-1] >= context.binding.dimensions.layer_count:
                raise ValueError(f"{name}.layers exceeds the model layer count")
            return tuple(
                layout_field(
                    f"{key}_{layer}",
                    value,
                    lambda m: (m.hidden_size, m.hidden_size),
                    context.field_dtype(name, f"{key}_{layer}", value.dtype),
                )
                for layer in self.layers
                for key, value in _fields(self).items()
            )
        consumer = next((adapter for adapter in context.adapters.values() if adapter.state == name), None)
        shapes = {key: spec for key, spec in self.fields.items() if not isinstance(spec, Field)}
        if consumer is None and len(shapes) == len(self.fields):
            return tuple(
                LayoutField(key, shape, context.field_dtype(name, key, self.dtype), self.alignment)
                for key, shape in shapes.items()
            )
        fields = _fields(self)
        if consumer is None:
            return tuple(
                layout_field(
                    key, value, lambda m: (m.hidden_size, m.hidden_size), context.field_dtype(name, key, value.dtype)
                )
                for key, value in fields.items()
            )
        context.binding.overlay_targets = consumer.targets
        return overlay_fields(
            fields, consumer, context.binding.dimensions, lambda key, value: context.field_dtype(name, key, value.dtype)
        )


@dataclass(frozen=True)
class RequestState(InstanceState):
    """Per-request parameters, read as tensors aligned to the current operation's rows."""

    # Requests with equal values of these fields share one copy of the state, listed in KeyField order.
    key_fields: tuple[KeyField, ...] = ("request_id",)

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(name not in get_args(KeyField) for name in self.key_fields):
            raise ValueError(f"key_fields={self.key_fields}: use names from {get_args(KeyField)}")
        ordered_key_fields(self.request_keys)

    scope: ClassVar[StateScope] = StateScope.REQUEST

    @property
    def request_keys(self) -> tuple[RequestKeyField, ...]:
        return tuple(RequestKeyField(name) for name in self.key_fields)


@dataclass(frozen=True)
class ActivationPatch:
    """Edit the hidden state at `layers` from a state, whoever writes it (ReFT-style): the first and last prompt
    tokens, and with decode=True every decode token.

    operator="low_rank_affine" reads three fields per layer, "add" one delta per layer. refresh="admission" keeps
    the version a request was admitted with; "step" reads the newest version at every decode step, and on_missing
    says what a step without a new version does ("stop" the request or "hold" the last version).
    """

    state: str
    layers: tuple[int, ...]
    first_tokens: int = 0
    last_tokens: int = 2
    operator: Literal["low_rank_affine", "add"] = "low_rank_affine"
    _: KW_ONLY
    decode: bool = False
    refresh: Literal["admission", "step"] = "admission"
    on_missing: Literal["stop", "hold"] = "stop"

    def __post_init__(self) -> None:
        if (
            self.operator not in ("low_rank_affine", "add")
            or self.refresh not in ("admission", "step")
            or self.on_missing not in ("stop", "hold")
        ):
            raise ValueError(
                'ActivationPatch takes operator "low_rank_affine" or "add", refresh "admission" or "step" and '
                f'on_missing "stop" or "hold"; got {self}'
            )
        if not self.decode and (self.refresh != "admission" or self.on_missing != "stop"):
            raise ValueError("refresh= and on_missing= apply to decode tokens; set decode=True")

    def declare(self, name: str, context: _Reflection) -> None:
        """Read the state at the activation sites; its writer, if the workload has one, is connected later."""
        definition = context.state_definition(self.state, "pllm.model.ActivationPatch needs a declared state")
        if definition.feed is not None:
            raise ValueError(
                "pllm.model.ActivationPatch reads a state the workload writes; it cannot read a fed state directly"
            )
        declare_activation_patch(
            context.binding,
            name,
            context.states[self.state],
            layers=self.layers,
            operator=ActivationPatchOperator(self.operator),
            first_tokens=self.first_tokens,
            last_tokens=self.last_tokens,
            decode=DecodeWindow.ALL if self.decode else DecodeWindow.NONE,
            decode_refresh=DecodeRefresh.STEP if self.refresh == "step" else DecodeRefresh.INHERIT,
            on_missing=MissingPolicy(self.on_missing),
        )


@dataclass(frozen=True)
class Adapter:
    """Apply a state as a weight delta at the target projections, whoever trains it (a PEFT-style adapter).

    `weight` names the field of a "dense" or "per_head" delta, `factors` the two fields of a "low_rank" one; the
    delta is multiplied by `scale`.
    """

    state: str
    targets: tuple[Projection, ...]
    weight: str = "value"
    operator: AdapterOperator = "dense"
    factors: tuple[str, str] = ("a", "b")
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.operator not in _OPERATORS:
            raise ValueError(f'Adapter(operator={self.operator!r}): use "dense", "low_rank" or "per_head"')

    @property
    def delta(self) -> ProjectionDelta:
        return ProjectionDelta(
            self.state, self.targets, self.weight, _OPERATORS[self.operator], self.factors, self.scale
        )


def _frozen(initialize: Initializer | None) -> Initializer | None:
    return None if initialize is None else freeze(initialize)


def _fields(state: InstanceState) -> dict[str, Field]:
    """The state's projection-sized fields as pllm.Field values;
    a fixed shape takes the state's dtype."""
    fields: dict[str, Field] = {}
    for key, value in state.fields.items():
        match value:
            case Field():
                fields[key] = value
            case tuple():
                fields[key] = Field(value, state.dtype)
            case _:
                raise TypeError(
                    f"{key}: a shape that depends on the projection is pllm.Field(lambda in_features, out_features: ...)"
                )
    return fields


class _Effect:
    """An operation effect owns its declaration and attachment policy."""

    kind: ClassVar[str] = "compute"
    training: ClassVar[bool] = False
    # Whether the effect acts inside the model forward, so it owns request training unless a sampler pairs with it.
    model_effect: ClassVar[bool] = False
    tuple_output: ClassVar[bool] = False
    # The phase the effect runs in when no `@pllm.on` names one; None means the author must name it.
    phase: ClassVar[Phase | None] = None

    state: str = ""
    update_rows: str = ""
    reduction: TrainReduction = "mean"

    def description(self, owner: Workload) -> MemberDeclaration:
        """The public effect name, with timing settings resolved at the authoring boundary."""
        return MemberDeclaration(self.kind)

    def accumulates(self, operation: _Operation, field: str) -> bool:
        """Whether this effect carries the field's previous value into its next version."""
        return False

    def declare(self, operation: _Operation) -> None:
        operation.tensor()

    @property
    def outputs(self) -> tuple[str, ...]:
        return ("value",)


@dataclass(frozen=True, kw_only=True)
class _Write(_Effect):
    kind: ClassVar[str] = "write"

    tuple_output: ClassVar[bool] = True

    state: str
    fields: tuple[str, ...]

    phase: ClassVar[Phase | None] = Phase.AUTOMATIC

    def accumulates(self, operation: _Operation, field: str) -> bool:
        return field in self.fields and operation.reads_field(self.state, field)

    def declare(self, operation: _Operation) -> None:
        context = operation.context
        definition = context.state_definition(self.state, "write needs a declared state")
        if definition.feed is not None:
            raise ValueError(f"@pllm.write: {self.state!r} is filled by its feed; write a state without feed=")
        if not self.fields or len(set(self.fields)) != len(self.fields) or operation.rows:
            raise ValueError("@pllm.write needs distinct field names and no rows=; it writes every row")
        for key in self.fields:
            context.parameter(self.state, key)
        context.signature.append((operation.name, fingerprint(self)))
        call = operation.tensor()
        state = context.states[self.state]
        update = write_result(context.binding, call, state, self.fields, operation.transition_window)
        context.writes[self.state] = (
            update,
            context.binding.publish(
                update, state, when=operation.transition_window.publication, name=f"{self.state}.publish"
            ),
        )

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.fields


def write(*, state: str, fields: tuple[str, ...]) -> Callable[[F], F]:
    """Atomically publish a tuple of computed fields; request values have a leading request axis."""
    return _mark(effect=_Write(state=state, fields=fields))


@dataclass(frozen=True)
class _Export(_Effect):
    kind: ClassVar[str] = "export"

    name: str | Attr
    every: int | Attr
    max_rows: int | Attr
    max_cols: int | Attr
    enabled: bool | Attr
    overflow: Literal["drop", "block"]

    phase: ClassVar[Phase | None] = Phase.AFTER_FORWARD
    model_effect: ClassVar[bool] = True

    def declare(self, operation: _Operation) -> None:
        """Publish bounded tensor observations to the host."""
        owner = operation.context.owner
        export = TensorExport(
            name=operation.context.emitted(_setting(self.name, owner, str)),
            every=_setting(self.every, owner, int),
            max_rows=_setting(self.max_rows, owner, int),
            max_cols=_setting(self.max_cols, owner, int),
            enabled=_setting(self.enabled, owner, bool),
            overflow=self.overflow,
        )
        operation.context.binding.exports(operation.tensor(result_schema="hidden_export@1", export=export))


_TRAINING_INPUTS: dict[TrainingMode, str] = {
    "request": "rows of running requests (no pllm.batch fields)",
    "batch": "WeightedRowBatch or CausalTrainBatch fields (or a causal model)",
    "weighted_batch": "WeightedCEBatch fields (base_logits or labels with hidden rows)",
}


def _training_schema(arguments: Mapping[str, _Source]) -> str:
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


def _training_mode(arguments: Mapping[str, _Source]) -> TrainingMode:
    """Batch fields choose the batch; without them or a causal model, train on request rows."""
    schema = _training_schema(arguments)
    if not schema:
        return "batch" if any(isinstance(source, _CausalModelMarker) for source in arguments.values()) else "request"
    return "weighted_batch" if schema == WeightedCEBatch.schema else "batch"


@dataclass(frozen=True, kw_only=True)
class _Train(_Effect):
    kind: ClassVar[str] = "train"

    state: str
    update_rows: str
    reduction: TrainReduction
    steps: int | Attr
    reset: str

    training: ClassVar[bool] = True
    phase: ClassVar[Phase | None] = Phase.AUTOMATIC

    def declare(self, operation: _Operation) -> None:
        """Train on what the method reads (see `_training_mode`); the state's optimizer must suit it."""
        arguments, carries = operation.arguments()
        definition = operation.context.state_definition(self.state, "@pllm.train needs a declared state")
        mode = _training_mode(arguments)
        if mode != "request" and operation.phase is Phase.AUTOMATIC:
            operation = replace(operation, phase=Phase.AFTER_SAMPLE)
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
                    f"@pllm.train reads {_TRAINING_INPUTS[mode]}, which trains with {expected}; declare "
                    f"{self.state!r} with optimizer={expected}"
                )

    def batch(self, operation: _Operation, arguments: dict[str, _Source], carries: dict[str, Carry]) -> None:
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
                function=freeze(operation.function),
                identity=operation.identity,
                when=operation.transition_window,
            ),
            self.state,
            "batch",
        )

    def request(self, operation: _Operation, arguments: dict[str, _Source], carries: dict[str, Carry]) -> None:
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
        when = When.AFTER_SAMPLE if request_training and operation.phase is Phase.AUTOMATIC else operation.phase.value
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
        operation: _Operation,
        arguments: dict[str, _Source],
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
            or operation.phase is not Phase.AFTER_SAMPLE
        ):
            raise ValueError(
                "training on WeightedCEBatch fields uses the whole batch after sampling: remove rows=, update_rows=, "
                "@pllm.differentiate and @pllm.on"
            )
        context = operation.context
        sources = {key: source.batch(self.state, definition) for key, source in arguments.items()}
        reset = context.method(self.reset)
        identity = boundary_digest(fingerprint(operation.function), fingerprint(reset), sources, self)
        context.signature.append((operation.name, identity))
        context.train(
            declare_weighted_train(
                context.binding,
                operation.name,
                context.states[self.state],
                identity=f"{operation.identity}:{identity}",
                optimizer=optimizer,
                steps=_setting(self.steps, context.owner, int),
                reset=freeze(reset),
                function=freeze(operation.function),
                arguments=sources,
            ),
            self.state,
            "weighted_batch",
        )


@dataclass(frozen=True)
class _Sampler(_Effect):
    stage: SamplerStage
    beta: float | Attr | None = None  # None for additive logits, which have no guidance strength

    phase: ClassVar[Phase | None] = Phase.BEFORE_SAMPLE

    def description(self, owner: Workload) -> MemberDeclaration:
        return MemberDeclaration("add_logits" if self.beta is None else "guide_sampling")

    def enabled(self, owner: Workload) -> bool:
        """Zero-strength guidance has no sampler attachment."""
        return self.beta is None or _setting(self.beta, owner, float) != 0

    def declare(self, operation: _Operation) -> None:
        """Zero-strength guidance declares no computation or sampler attachment."""
        if self.enabled(operation.context.owner):
            call = operation.tensor(
                output_kind=SemanticOutputKind.SAMPLER,
                sampler_stage=self.stage,
                sampler_beta=0.0 if self.beta is None else _setting(self.beta, operation.context.owner, float),
            )
            binding = operation.context.binding
            binding.tensor_tasks[call.symbol] = replace(
                binding.tensor_tasks[call.symbol], sampler_order=len(operation.context.predictions)
            )
            operation.context.predictions.append((call, self))


@dataclass(frozen=True)
class _Gradient:
    inputs: tuple[str, ...]
    parameters: bool
    input_rows: str = ""
    parameter_rows: str = ""

    def contract(self, operation: _Operation, arguments: dict[str, _Source]) -> MathContract:
        """Identify parameter resources and detachments."""
        parameters = {
            key: parameter for key, source in arguments.items() if (parameter := _typed(source, Parameter)) is not None
        }
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


def _mark(**markers: object) -> Callable[[F], F]:
    """Attach authoring markers; reflection consumes them once during declaration."""

    def decorate(function: F) -> F:
        # A second marker of one kind would silently replace the first, e.g. @export over @train never trains.
        if any(f"__pllm_{name}__" in vars(function) for name in markers):
            raise ValueError(
                "a method takes one pllm decorator of each kind; split stacked effects into one method per effect"
            )
        for name, marker in markers.items():
            setattr(function, f"__pllm_{name}__", marker)
        return function

    return decorate


def differentiate(
    *, inputs: tuple[str, ...] = (), parameters: bool = False, input_rows: str = "", parameter_rows: str = ""
) -> Callable[[F], F]:
    """Keep derivative edges; parameters may stay frozen while inputs differentiate."""
    return _mark(gradient=_Gradient(inputs, parameters, input_rows, parameter_rows))


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
    hidden_size: int,
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
        pending = {
            f"_pending_rows_{name}": Field(
                lambda inputs, outputs, source=source: (
                    chunk,
                    hidden_size if source == "embedding" else inputs if source == "input" else outputs,
                ),
                dtype=None,
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
        raise ValueError("rename the state fields or arguments that start with _pending_; pllm reserves them")
    fields.update(pending)
    bindings.update((name, name) for name in pending)
    if optimizer is not None:
        bindings["_prompt_end"] = "prompt_end"
    return update, initialize


@dataclass(frozen=True, kw_only=True)
class _TrainInForward(_Effect):
    kind: ClassVar[str] = "train_in_forward"

    state: str
    targets: tuple[Projection, ...] | Attr
    weight: str
    operator: AdapterOperator | Attr
    factors: tuple[str, str] | Attr
    scale: float | Attr
    tokens: Tokens | Attr
    visibility: VisibilityKind | Attr
    chunk: int | Attr | None
    flush: Literal["prompt_end"] | None

    phase: ClassVar[Phase | None] = Phase.IN_FORWARD
    model_effect: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.flush not in (None, "prompt_end"):
            raise ValueError("@pllm.train_in_forward(flush=...) takes 'prompt_end' or None")

    def accumulates(self, operation: _Operation, field: str) -> bool:
        return (self.chunk is not None and field == self.weight) or operation.reads_field(self.state, field)

    def projection_delta(self, owner: Workload, layers: int) -> ProjectionDelta:
        """The configured projection delta, with all-layer targets resolved against the model."""
        return ProjectionDelta(
            self.state,
            _targets(_setting(self.targets, owner, tuple)),
            self.weight,
            _OPERATORS[_option(self.operator, owner, get_args(AdapterOperator), "operator")],
            _setting(self.factors, owner, tuple),
            _setting(self.scale, owner, float),
        ).expand(layers)

    def bindings(
        self, operation: _Operation, fields: Mapping[str, Field], targets: tuple[Projection, ...], optimizer: SGD | None
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
                    raise ValueError("rename fields starting with __invariant_; pllm reserves them")
                invariants[name] = (source.state, views)
                bindings[key] = name
            elif isinstance(source, Operand):
                bindings[key] = _option(source.name, context.owner, get_args(OperandName), "operand")
            else:
                bindings[key] = source.in_forward_binding(self.state, fields)
        invariant_names = tuple(
            key
            for key, source in sources.items()
            if (isinstance(source, Parameter) and source.state != self.state) or bindings[key] == "weight"
        )
        return bindings, invariants, invariant_names

    def description(self, owner: Workload) -> MemberDeclaration:
        return MemberDeclaration(
            self.kind,
            tokens=_option(self.tokens, owner, get_args(Tokens), "tokens"),
            visibility=_option(self.visibility, owner, ("forward_pinned", "exact"), "visibility"),
        )

    def declare(self, operation: _Operation) -> None:
        """Declare the request state's in-forward transition and the delta it applies at the target projections."""
        if operation.gradient is not None or operation.rows:
            raise ValueError("@pllm.train_in_forward updates its own state: remove rows= and @pllm.differentiate")
        if operation.phase is not Phase.IN_FORWARD:
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
            for key, value in _fields(definition).items()
        }
        effect = self.projection_delta(context.owner, context.binding.dimensions.layer_count)
        bindings, invariants, invariant_names = self.bindings(operation, fields, effect.targets, optimizer)
        if set(fields) & set(get_args(OperandName)):
            raise ValueError(
                "rename the state fields that repeat an operand name (input, target, embedding, weight, prompt_end)"
            )
        identity = fingerprint(operation.function)
        function: Callable[..., UpdateOutput] = freeze(operation.function)
        initialize = context.initializer(self.state, tuple(definition.fields))
        if self.chunk is not None:
            function, initialize = _chunked_in_forward(
                function,
                initialize,
                bindings,
                fields,
                effect,
                _setting(self.chunk, context.owner, int),
                optimizer,
                self.flush,
                invariant_names,
                context.binding.dimensions.hidden_size,
            )
        declare_parameter_overlay(
            context.binding,
            operation.name,
            effect,
            fields,
            initialize=initialize,
            keys=definition.request_keys,
            phases=_TOKEN_PHASES[_option(self.tokens, context.owner, get_args(Tokens), "tokens")],
            identity=operation.identity,
            function=function,
            arguments=bindings,
            visibility=_option(self.visibility, context.owner, ("forward_pinned", "exact"), "visibility"),
            equations_digest=boundary_digest(context.signature, context.bindings, self.chunk, self.flush),
            invariants=invariants,
            reads=tuple(
                context.states[name] for name in dict.fromkeys(resource for resource, _ in invariants.values())
            ),
        )
        context.signature.append(
            (
                operation.name,
                boundary_digest(
                    identity,
                    bindings,
                    effect.operator,
                    effect.scale,
                    effect.applied_fields,
                    self.visibility,
                    self.chunk,
                    self.flush,
                ),
            )
        )


_SITE_TENSORS: dict[str, int] = {"attention": 2, "mlp": 2, "both": 2, "value": 1, "final": 1}
# The places a site owns in each chosen layer.
_SITE_KINDS: dict[str, tuple[Literal["attention", "mlp", "value"], ...]] = {
    "attention": ("attention",),
    "mlp": ("mlp",),
    "both": ("attention", "mlp"),
    "value": ("value",),
}


def _site_inputs(
    function: Callable[..., torch.Tensor], leading: int, context: _Reflection, method: str
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
        source = _typed(parameter.default, _Source)
        if source is None:
            raise TypeError(f"{name}: give every input after the site tensors a source as its default")
        inputs.append((name, source.qualify(context).site_input(context)))
    return tuple(inputs)


@dataclass(frozen=True, kw_only=True)
class _CustomResidual(_Effect):
    kind: ClassVar[str] = "custom_residual"

    site: Site
    layers: tuple[int, ...] | Attr | None
    read: str
    streams: int | Attr
    state: str
    chunk: int | Attr | None

    phase: ClassVar[Phase | None] = Phase.IN_FORWARD
    model_effect: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.site not in _SITE_TENSORS:
            raise ValueError(
                f"@pllm.custom_residual(site={self.site!r}): use one of {', '.join(map(repr, _SITE_TENSORS))}"
            )

    def accumulates(self, operation: _Operation, field: str) -> bool:
        return bool(self.state) and self.chunk is not None

    def places(self, owner: Workload, count: int) -> tuple[Place, ...]:
        """The owned places: the chosen layers' sublayer adds or value rows, or the final stream."""
        if self.site == "final":
            if self.layers is not None:
                raise ValueError('@pllm.custom_residual(site="final") rewrites the one final stream; remove layers=')
            return (("final", count),)
        layers = tuple(range(count)) if self.layers is None else _setting(self.layers, owner, tuple)
        if (
            not layers
            or tuple(sorted(set(layers))) != layers
            or not all(type(layer) is int and 0 <= layer < count for layer in layers)
        ):
            raise ValueError(f"@pllm.custom_residual(layers={layers!r}): give sorted distinct layers in 0..{count - 1}")
        return tuple((kind, layer) for layer in layers for kind in _SITE_KINDS[self.site])

    def declare(self, operation: _Operation) -> None:
        """Own the chosen sites' residual updates (or value rows) with the method; it reads states and earlier values."""
        context = operation.context
        if operation.gradient is not None or operation.rows:
            raise ValueError(
                "@pllm.custom_residual rewrites every row of its sites: remove rows= and @pllm.differentiate"
            )
        if operation.phase is not Phase.IN_FORWARD:
            raise ValueError("@pllm.custom_residual runs inside the forward; remove @pllm.on")
        if context.binding.host:
            raise ValueError("@pllm.custom_residual acts inside a model forward; compile this workload with make_llm")
        owner, count = context.owner, context.binding.dimensions.layer_count
        places = self.places(owner, count)
        memory = None if not self.state else self.memory(operation, places)
        reader = context.method(self.read) if self.read else None
        rewrite = ResidualRewrite(
            operation.name,
            self.site,
            places,
            freeze(operation.function),
            _site_inputs(operation.function, _SITE_TENSORS[self.site], context, operation.name),
            None if reader is None else freeze(reader),
            () if reader is None else _site_inputs(reader, 1, context, self.read),
            _setting(self.streams, owner, int),
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
        context.signature.append(
            (operation.name, boundary_digest(fingerprint(operation.function), fingerprint(reader), self))
        )

    def memory(self, operation: _Operation, places: tuple[Place, ...]) -> SiteMemory:
        """Declare the request state the method carries chunk by chunk, with the framework's unfinished-chunk rows,
        and the transition that advances it inside the forward."""
        context, b = operation.context, operation.context.binding
        definition = context.state_definition(self.state, "@pllm.custom_residual(state=...) needs a declared state")
        if (
            definition.scope is not StateScope.REQUEST
            or definition.optimizer is not None
            or definition.module is not None
            or len(definition.fields) != 1
            or self.chunk is None
            or len(places) != 1
            or places[0][0] not in ("attention", "mlp")
        ):
            raise ValueError(
                f"@pllm.custom_residual(state={self.state!r}) needs one attention or MLP site of one layer, chunk=, "
                "and a pllm.RequestState with one field, its carry, and no optimizer"
            )
        chunk = _setting(self.chunk, context.owner, int)
        if chunk < 1:
            raise ValueError("@pllm.custom_residual(chunk=) must be positive")
        ((name, shape),) = definition.fields.items()
        if isinstance(shape, Field):
            raise TypeError(f"{name}: give the carry a fixed shape or a function of the model dimensions")
        rows: Shape = lambda dimensions: (max(1, chunk - 1), dimensions.hidden_size)  # noqa: E731
        initialize = context.initializer(self.state, (name,))
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


def custom_residual(
    *,
    site: Site,
    layers: tuple[int, ...] | Attr | None = None,
    read: str = "",
    streams: int | Attr = 1,
    state: str = "",
    chunk: int | Attr | None = None,
) -> Callable[[F], F]:
    """Own the residual update at `site` of `layers` (default: every layer) with this method.

    site="attention" or "mlp" (or "both", in forward order) take the stream before the sublayer and its branch
    output, `write(x, y, ...) -> stream`, replacing the plain `x + y`; the native branch always runs. site="value"
    maps a full-attention layer's value rows [rows, kv_heads, head_dim] before attention reads and caches them;
    site="final" maps the stream entering the final norm to one stream. `read` names a method `(x, ...) -> branch
    input` that the sublayer normalizes instead of the stream. With `streams=n` the stream is [rows, n, hidden] from the
    embedding to a "final" method that collapses it. After the site tensors, methods read pllm.parameter, pllm.module,
    pllm.depth(), pllm.value(layer) and pllm.residual(layer) (the same token, earlier in this forward). Tensors are in the
    model dtype; compute in float32 inside the method when needed.

    With `state` (a pllm.RequestState with one field) and `chunk`, the method is a stateful sublayer at one site: it
    sees one chunk of one request, `write(x, y, carry=pllm.parameter(state, ...), ...) -> (stream, next carry)`,
    where every row reads the carry its chunk started with and a chunk's update applies once its last row has run.
    Each row's stream should depend only on that row and the carry.
    """
    return _mark(effect=_CustomResidual(site=site, layers=layers, read=read, streams=streams, state=state, chunk=chunk))


def train_in_forward(
    *,
    state: str,
    targets: tuple[Projection, ...] | Attr,
    weight: str = "value",
    operator: AdapterOperator | Attr = "dense",
    factors: tuple[str, str] | Attr = ("a", "b"),
    scale: float | Attr = 1.0,
    tokens: Tokens | Attr = "prefill",
    visibility: Literal["forward_pinned", "exact"] | Attr = "forward_pinned",
    chunk: int | Attr | None = None,
    flush: Literal["prompt_end"] | None = None,
) -> Callable[[F], F]:
    """Train a request state inside the forward and add its residual at each target projection.

    `tokens` picks the tokens whose projection operands train the state: "prefill", "decode" or "all";
    `operator`, `weight`, `factors` and `scale` say how the state changes each projection, as in
    pllm.model.Adapter. visibility="forward_pinned" reads the state the forward started with; "exact" also reads
    every chunk already trained in this forward.

    With ``chunk``, the function takes the weight carry followed by input/target/embedding row operands and returns
    (next carry, readouts). Pending rows belong to the framework. When ``state`` declares
    ``optimizer=pllm.SGD(lr)``, return one loss per row instead; gradients accumulate until a chunk completes,
    including a short final chunk with ``flush="prompt_end"``.
    """
    return _mark(
        effect=_TrainInForward(
            state=state,
            targets=targets,
            weight=weight,
            operator=operator,
            factors=factors,
            scale=scale,
            tokens=tokens,
            visibility=visibility,
            chunk=chunk,
            flush=flush,
        )
    )


def on(phase: Phase | Attr) -> Callable[[F], F]:
    """Run the method in `phase`; needed for methods without an effect or to move an effect's phase."""
    return _mark(phase=phase)


def export(
    *,
    name: str | Attr = "value",
    every: int | Attr = 1,
    max_rows: int | Attr = 1,
    max_cols: int | Attr = 16,
    enabled: bool | Attr = True,
    overflow: Literal["drop", "block"] = "drop",
) -> Callable[[F], F]:
    """Deliver bounded observations of executed forward rows to on_export.

    Async scheduling can execute one extra decode row before learning that a request
    stopped at EOS; that row is observed too. Callbacks run asynchronously and close()
    drains their delivery. Block on a full export queue only when explicitly requested.
    """
    if overflow not in ("drop", "block"):
        raise ValueError("export overflow must be 'drop' or 'block'")
    return _mark(effect=_Export(name, every, max_rows, max_cols, enabled, overflow))


def add_logits(*, rows: str = "") -> Callable[[F], F]:
    """Add the returned tensor to the logits before sampling.

    `rows` (here and on the other effects) names a method returning a boolean mask: only those rows run.
    """
    return _mark(effect=_Sampler(SamplerStage.PACKED_PRE_FILTER_ADD), rows=rows)


def guide_sampling(*, beta: float | Attr, rows: str = "") -> Callable[[F], F]:
    """Push sampling away from the returned hidden-state prediction for positive `beta`.

    On finite top-k/top-p survivors, use `(1 + beta) * logits - beta * pred_logits`, where the model's LM head
    projects the returned hidden states to `pred_logits`.
    """
    return _mark(effect=_Sampler(SamplerStage.PACKED_POST_FILTER_EXACT, beta), rows=rows)


def train(
    *,
    state: str,
    update_rows: str = "",
    reduction: TrainReduction = "state_mean",
    steps: int | Attr = 1,
    reset: str = "",
    rows: str = "",
) -> Callable[[F], F]:
    """Minimize the returned loss with `state`'s optimizer; updates publish after sampling by default.

    With tensor parallelism, independent automatic request training can compute a private update before the
    draw. Publication still follows sampling; cancellation discards the private version. Use
    `@pllm.on(pllm.Phase.AFTER_SAMPLE)` to require the computation itself to follow sampling.

    The method's inputs say what it trains on. Without pllm.batch fields it trains each request's state on that
    request's rows (pllm.AdamW). WeightedRowBatch or CausalTrainBatch fields, or a causal model, take one AdamW
    step on the host batch. WeightedCEBatch fields run `steps` SGD steps per state, each state first restored by
    the `reset` method (pllm.SGD).

    The default `reduction="state_mean"` averages each state's rows, then sums over states, so RequestState
    updates are independent of neighboring requests. `"mean"` averages over all rows and couples request updates.
    Host-batch training ignores it. `rows=` names a selector whose mask picks the rows that count; the method
    still runs on every row. For request training, rows= and update_rows= require a model effect without sampler
    guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state.
    """
    if reduction not in ("mean", "state_mean"):
        raise ValueError(f"@pllm.train(reduction={reduction!r}): use 'mean' or 'state_mean'")
    return _mark(
        effect=_Train(state=state, update_rows=update_rows, reduction=reduction, steps=steps, reset=reset), rows=rows
    )


def _option(value: L | Attr, owner: Workload, options: tuple[L, ...], name: str) -> L:
    """Resolve a closed string option, possibly a pllm.Attr, and check it is one of `options`."""
    resolved = Attr.resolve(value, owner)
    for option in options:
        if resolved == option:
            return option
    raise ValueError(f"{name}={resolved!r}: use one of {', '.join(map(repr, options))}")


def _setting(value: C | Attr, config: Workload, expected: type[C]) -> C:
    resolved = Attr.resolve(value, config)
    if expected is float and type(resolved) is int:
        resolved = float(resolved)
    if (typed := _typed(resolved, expected)) is None or type(resolved) is not expected:
        raise TypeError(f"{value!r} is {type(resolved).__name__}; expected {expected.__name__}")
    return typed


def _targets(value: tuple[object, ...]) -> tuple[Projection, ...]:
    targets = tuple(target for item in value if (target := _typed(item, Projection)) is not None)
    if len(targets) != len(value):
        raise TypeError("targets must be pllm.model.Projection values")
    return targets


def custom_forward(*, tokens: Tokens = "all", over: Over = "token") -> Callable[[F], F]:
    """Replace the forward of the selected tokens with this method's program over model layer slices and tensor math.

    ``tokens`` follows the token's prompt/generated role, including after preemption.
    ``over="token"`` runs each position independently in causal order. ``over="span"``
    packs each request's contiguous scheduled tokens into [M, H] hidden tensors;
    position and generated flags have shape [M, 1]. Spans cannot load predecessors.
    Inputs after layers use pllm.parameter / pllm.module to read InstanceState, shared
    by every token, or pllm.Attr for a static constant.
    """
    if tokens not in ("prefill", "decode", "all"):
        raise ValueError('set @pllm.custom_forward(tokens=) to "prefill", "decode" or "all"')
    if over not in ("token", "span"):
        raise ValueError('set @pllm.custom_forward(over=) to "token" or "span"')
    return _mark(tokens=tokens, over=over)


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


def _arguments(
    function: Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | UpdateOutput | None],
) -> dict[str, _Source]:
    """Read each parameter's source from its default value, e.g. `h: torch.Tensor = pllm.hidden(0)`."""
    result: dict[str, _Source] = {}
    for name, parameter in signature(function, eval_str=False).parameters.items():
        source = _typed(parameter.default, _Source)
        if source is None or parameter.kind is SignatureParameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{name}: give every input a source as its default, e.g. `{name}: torch.Tensor = pllm.hidden(0)`"
            )
        result[name] = source
    return result


@dataclass
class _Operation:
    """A reflected method, with marker types checked once at the boundary."""

    context: _Reflection
    name: str
    phase: Phase
    effect: _Effect
    gradient: _Gradient | None
    rows: str
    phase_override: bool = False

    @property
    def function(self) -> FunctionType | MethodType:
        return self.context.method(self.name)

    @property
    def identity(self) -> str:
        """Identify a method implementation within its component's namespace."""
        return f"python:{self.name}:{fingerprint(self.function)}"

    @property
    def transition_window(self) -> When:
        """Where this method's state transition runs; host workloads have their own step anchors."""
        return When.HOST_COMPUTE if self.context.binding.host else self.phase.value

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

    def arguments(self) -> tuple[dict[str, _Source], dict[str, Carry]]:
        """Split carried inputs into their read source and the carried field the update writes."""
        sources, carries = self.context.arguments(self.function), dict[str, Carry]()
        for key, source in sources.items():
            if (carried := _typed(source, _Carried)) is not None:
                sources[key], carries[key] = carried.source, carried.carry
        return sources, carries

    def tensor(
        self,
        arguments: dict[str, _Source] | None = None,
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
        inputs, identity = {key: source.port(context) for key, source in arguments.items()}, fingerprint(self.function)
        context.signature.append(
            (
                self.name,
                boundary_digest(
                    identity,
                    tuple((key, fingerprint(source)) for key, source in arguments.items()),
                    fingerprint(carries),
                ),
            )
        )
        demand = self.gradient or _Gradient((), effect.training)
        math = demand.contract(self, arguments)
        update = context.states[effect.state] if effect.training else None
        writes = {key: carry.declare(effect.state, update) for key, carry in (carries or {}).items()}
        math = self.selectors(arguments, inputs, math)
        return declare_tensor(
            context.binding,
            self.name,
            inputs=inputs,
            when=self.phase.value if when is None else when,
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
        )(freeze(self.function))

    @property
    def domains(self) -> dict[str, str]:
        """Selector methods whose inputs contribute to the operation's model requirements."""
        demand = self.gradient or _Gradient((), self.effect.training)
        return {
            "forward_rows": self.rows,
            "input_gradient_rows": demand.input_rows,
            "parameter_gradient_rows": demand.parameter_rows,
            "update_rows": self.effect.update_rows,
        }

    def sources(self) -> dict[str, dict[str, _Source]]:
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
        arguments: dict[str, _Source],
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
                selector = self.context.method(method)
                identity = fingerprint(selector)
                self.context.signature.append((method, identity))
                selectors[method] = declare_tensor(
                    self.context.binding,
                    self.context.selector_name(self.name, method),
                    when=self.phase.value,
                    identity=f"python:{identity}:{self.name}:{method}",
                    inputs={
                        key: inputs[key] if arguments.get(key) == source else source.port(self.context)
                        for key, source in sources[self.context.selector_name(self.name, method)].items()
                    },
                )(freeze(selector))
            key = f"__{domain}"
            inputs[key] = result_port(TaskResultInputSource(selectors[method].symbol, "value"))
            math = replace(math, **{domain: key})
        return math


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


@dataclass
class _Reflection:
    """Own the binding, reflected owner and ordered definitions during one build."""

    owner: Workload
    binding: TensorBinding = field(init=False)
    prefix: str = ""
    components: tuple[_Reflection, ...] = ()
    # The owner's values, signed; the declared states and model values among them, by type.
    bindings: tuple[tuple[str, str], ...] = ()
    state_definitions: dict[str, InstanceState] = field(default_factory=lambda: {})
    pretrained: dict[str, PretrainedFile] = field(default_factory=lambda: {})
    # Each declared pllm.model.Adapter as the projection delta it applies.
    adapters: dict[str, ProjectionDelta] = field(default_factory=lambda: {})
    activations: dict[str, ActivationPatch] = field(default_factory=lambda: {})
    position_definitions: dict[str, PositionState] = field(default_factory=lambda: {})
    # Ordered (name, digest) entries folded into the graph's execution parameters digest.
    signature: list[tuple[str, str]] = field(default_factory=lambda: [])
    operations: list[_Operation] = field(default_factory=lambda: [])
    custom_forwards: list[tuple[str, Tokens, Over]] = field(default_factory=lambda: [])
    positions: tuple[tuple[str, float], ...] = ()
    in_forward_states: set[str] = field(default_factory=lambda: set[str]())
    states: dict[str, State] = field(default_factory=lambda: {})
    parameters: dict[tuple[str, str], TensorPort] = field(default_factory=lambda: {})
    writes: dict[str, tuple[Call, Call]] = field(default_factory=lambda: {})
    predictions: list[tuple[Call, _Sampler]] = field(default_factory=lambda: [])
    # Each @pllm.train method and the state it trains.
    trainers: list[tuple[Call, State]] = field(default_factory=lambda: [])

    def initializer(self, name: str, fields: tuple[str, ...]) -> Initializer | None:
        """One starting-value source for ordinary state and model-side state lowerings."""
        if name in self.pretrained:
            return partial(self.pretrained[name].initialize, fields)
        return _frozen(self.state_definitions[name].initialize)

    def validate_pretrained(self) -> None:
        """Validate declared weights on CPU after every lowering has resolved its field shapes."""
        for resource in self.pretrained:
            if resource in self.binding.initial_values:
                self.binding.initial_values[resource](self.binding.dimensions, 0)
            else:
                layout = next(item.layout for item in self.binding.layouts.layouts if item.resource == resource)
                tensors = tuple(
                    torch.empty(field.shape, dtype=field.dtype.torch, device="meta") for field in layout.fields
                )
                self.binding.initializers[resource](tensors, torch.Generator())

    def field_dtype(self, state: str, field: str, dtype: torch.dtype | Literal["auto"] | None) -> TensorDType:
        """Choose storage once from the declared field role; explicit precision always wins."""
        if dtype != "auto":
            return self.binding.dimensions.dtype if dtype is None else TensorDType(str(dtype).removeprefix("torch."))
        definition = self.state_definitions[state]
        if definition.optimizer is not None or any(
            operation.effect.state == state and operation.effect.accumulates(operation, field)
            for operation in self.operations
        ):
            return TensorDType.FLOAT32
        return self.binding.dimensions.dtype

    @property
    def sampler(self) -> _Sampler | None:
        """Find active guidance before either prediction or training is emitted."""
        samplers = tuple(
            sampler
            for operation in self.operations
            if (sampler := _typed(operation.effect, _Sampler)) is not None and sampler.enabled(operation.context.owner)
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
        return delta.field_bindings(lambda name: self.binding.field_views[delta.state, name])

    @classmethod
    def read_local(cls, workload: Workload, prefix: str) -> _Reflection:
        """Snapshot and sign reachable mathematics before evaluating state properties."""
        owner = _snapshot(workload)
        values, members = _values(owner), _members(owner)
        _check_base_collisions(owner)
        context, candidates = cls(owner, prefix=prefix), _candidates(owner, values, members)
        seeds: list[FunctionType] = []
        for name, member in members.items():
            context.read_operation(name, member, candidates, seeds)
            if (getter := _property(member)) is not None:
                seeds.append(getter)
        for value in values.values():
            if (state := _typed(value, InstanceState)) is not None and state.initialize is not None:
                method = _typed(state.initialize, MethodType)
                if (
                    function := _typed(state.initialize if method is None else method.__func__, FunctionType)
                ) is not None:
                    seeds.append(function)
        context.declare_result_helpers(candidates, seeds)
        context.freeze_helpers(candidates, seeds)
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
        context.namespace()
        return context

    @classmethod
    def read(cls, workload: Workload) -> _Reflection:
        """Flatten independent field components into one declaration and one conflict domain."""
        contexts = tuple(cls.read_local(owner, path) for path, owner in component_workloads(workload))
        root = contexts[0]
        root.components = contexts
        root.state_definitions = {
            key: value for context in contexts for key, value in context.state_definitions.items()
        }
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
    ) -> dict[str, _Source]:
        """Resolve local marker names against this method's component."""
        return {key: source.qualify(self) for key, source in _arguments(function).items()}

    def namespace(self) -> None:
        """Qualify emitted names while retaining each method's bound owner and local configuration."""
        self.bindings = tuple((self.qualify(key), value) for key, value in self.bindings)
        self.state_definitions = {self.qualify(key): value for key, value in self.state_definitions.items()}
        self.adapters = {
            self.qualify(key): replace(value, state=self.qualify(value.state)) for key, value in self.adapters.items()
        }
        self.activations = {
            self.qualify(key): replace(value, state=self.qualify(value.state))
            for key, value in self.activations.items()
        }
        self.position_definitions = {self.qualify(key): value for key, value in self.position_definitions.items()}
        self.in_forward_states = {self.qualify(key) for key in self.in_forward_states}
        self.custom_forwards = [(self.qualify(key), tokens, over) for key, tokens, over in self.custom_forwards]
        for operation in self.operations:
            operation.name = self.qualify(operation.name)
            operation.rows = self.qualify(operation.rows) if operation.rows else ""
            effect = operation.effect
            if is_dataclass(effect):
                references = {
                    item.name: self.qualify(value)
                    for item in fields(effect)
                    if item.name in ("state", "update_rows", "reset", "read") and (value := vars(effect)[item.name])
                }
                operation.effect = replace(effect, **references)
            if operation.gradient is not None:
                gradient = operation.gradient
                operation.gradient = replace(
                    gradient,
                    input_rows=self.qualify(gradient.input_rows) if gradient.input_rows else "",
                    parameter_rows=self.qualify(gradient.parameter_rows) if gradient.parameter_rows else "",
                )

    def component(self, name: str) -> _Reflection:
        """Find the owner of a qualified member, including nested field paths."""
        return next(context for context in reversed(self.components) if name.rpartition(".")[0] == context.prefix)

    def read_operation(
        self, name: str, member: object, candidates: dict[str, FunctionType], seeds: list[FunctionType]
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
                    "@pllm.custom_forward methods take (self, layers, marker inputs...) and no other pllm decorator; use layers.save instead of @write"
                )
            overlap = {
                role
                for _, selected, _ in self.custom_forwards
                for role in ("prefill", "decode")
                if selected in (role, "all") and tokens in (role, "all")
            }
            if overlap:
                raise ValueError(
                    f"{self.qualify(name)}: @pllm.custom_forward tokens overlap: {', '.join(sorted(overlap))}"
                )
            self.custom_forwards.append((name, tokens, markers["__pllm_over__"]))
            seeds.append(function)
            return
        phase, marked, gradient, rows = (
            _typed(Attr.resolve(markers.get("__pllm_phase__"), self.owner), Phase),
            _typed(markers.get("__pllm_effect__"), _Effect),
            _typed(markers.get("__pllm_gradient__"), _Gradient),
            _typed(markers.get("__pllm_rows__", ""), str),
        )
        if phase is None and marked is None:
            return
        effect = marked or _Effect()
        texts = (*vars(effect).values(), *vars(gradient or _Gradient((), False)).values(), rows)
        seeds.extend(candidates[text] for text in texts if type(text) is str and text in candidates)
        if (trained := _typed(effect, _TrainInForward)) is not None:
            self.in_forward_states.add(trained.state)
        if (residual := _typed(effect, _CustomResidual)) is not None and residual.state:
            self.in_forward_states.add(residual.state)
        phase_override = phase is not None
        phase = phase or effect.phase or Phase.AFTER_SAMPLE
        seeds.append(function)
        assert rows is not None
        self.operations.append(_Operation(self, name, phase, effect, gradient, rows, phase_override))

    def method(self, name: str) -> FunctionType | MethodType:
        """Bind a named authoring method, including an instance-local replacement."""
        try:
            context = self.component(name) if self.components else self
            return object.__getattribute__(
                context.owner, name.removeprefix(f"{context.prefix}.") if context.prefix else name
            )
        except AttributeError:
            raise ValueError(f"{name!r} names no method of {type(self.owner).__name__}") from None

    def declare_result_helpers(self, candidates: dict[str, FunctionType], seeds: list[FunctionType]) -> None:
        """Declare implicit result producers, giving model-site consumers the pre-forward row domain."""
        pending = [(op.function, op.phase) for op in self.operations if not isinstance(op.effect, _CustomResidual)]
        pending.extend(
            (op.function, Phase.BEFORE_FORWARD) for op in self.operations if isinstance(op.effect, _CustomResidual)
        )
        pending.extend(
            (self.method(op.effect.read), Phase.BEFORE_FORWARD)
            for op in self.operations
            if isinstance(op.effect, _CustomResidual) and op.effect.read
        )
        declared = {op.name for op in self.operations}
        while pending:
            function, phase = pending.pop()
            for parameter in signature(function).parameters.values():
                source = parameter.default
                if not isinstance(source, Result) or source.operation in declared:
                    continue
                helper = self.method(source.operation)
                self.operations.append(_Operation(self, source.operation, phase, _Effect(), None, ""))
                declared.add(source.operation)
                seeds.append(candidates[source.operation])
                pending.append((helper, phase))

    def freeze_helpers(self, candidates: dict[str, FunctionType], seeds: list[FunctionType]) -> None:
        """Freeze only helpers reachable from operations, state properties or markers."""
        reachable = reachable_functions(candidates, tuple(seeds)) | {
            name for name, member in candidates.items() if any(member is seed for seed in seeds)
        }
        for name in candidates:
            if name in reachable:
                function = self.method(name)
                with self.declaring(name):
                    self.signature.append((name, fingerprint(function)))
                object.__setattr__(self.owner, name, freeze(function))

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
                state.declare(name, self)
            if state.imported:
                allocated = self.states[name]
                self.binding.effects = replace(
                    self.binding.effects,
                    import_fields=tuple(
                        (field, view)
                        for (resource, field), view in self.binding.field_views.items()
                        if resource == name
                    ),
                )
                self.binding.imports[name] = tuple(
                    view for (resource, _), view in self.binding.field_views.items() if resource == name
                )
                update = import_update(self.binding, allocated, f"{name}:import@1")
                self.writes[name] = (
                    update,
                    self.binding.publish(update, allocated, when=When.BEFORE_FORWARD, name=f"{name}.publish"),
                )
        self.signature.extend(
            (name, fingerprint(value)) for name, value in {**self.activations, **self.adapters}.items()
        )

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
                patch.declare(name, self)
        for name, delta in self.adapters.items():
            with self.declaring(name), self.binding.member(name, MemberDeclaration("Adapter")):
                self.binding.overlay(self.adapter_bindings(delta))
        # The writes of states activation patches read: a custom forward reads their versions, and a fed chain
        # (feed -> @pllm.write -> patch) closes its conditional input at that publication.
        publications = tuple(
            self.writes[patch.state][1] for patch in self.activations.values() if patch.state in self.writes
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
        """Bind each forward's instance fields and resolve static attribute defaults."""
        function = self.method(name)
        inputs: list[tuple[str, StateInput]] = []
        for key, parameter in tuple(signature(function).parameters.items())[1:]:
            source = parameter.default
            if isinstance(source, Attr):
                function = partial(function, **{key: Attr.resolve(source, self.component(name).owner)})
                continue
            if not isinstance(source, Parameter):
                raise TypeError(
                    f"{name}.{key}: use pllm.parameter or pllm.module after layers, or pllm.Attr for a constant"
                )
            binding = source.qualify(self.component(name)).site_input(self)
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
                f"{call.symbol} trains on {_TRAINING_INPUTS[mode]} but {self.trainers[0][0].symbol} on "
                f"{_TRAINING_INPUTS[mode_now]}; pllm funds one training mode per workload, so move one of them to a "
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
        if operation.phase is not Phase.AUTOMATIC:
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

        for name, state in self.state_definitions.items():
            if state.checkpoint and any(
                effect.resource == name and effect.kind is not StateEffectKind.READ
                for task in b.graph.tasks
                for effect in task.effects
            ):
                raise ValueError(f"checkpoint state {name!r} is frozen; write to a separate state")
        frozen = {
            name
            for name, state in self.state_definitions.items()
            if state.optimizer is None
            and not state.imported
            and state.feed is None
            and not any(
                effect.resource == name and effect.kind is not StateEffectKind.READ
                for task in b.graph.tasks
                for effect in task.effects
            )
        }
        b.graph = replace(
            b.graph,
            states=tuple(replace(state, immutable=state.resource in frozen) for state in b.graph.states),
            execution_parameters_digest=boundary_digest(
                b.graph.execution_parameters_digest, self.signature, self.bindings
            ),
        )


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


class Workload(BackendWorkload[TensorFrame, TensorOutput]):
    """Python methods and effects compiled once, with independent sub-workloads in dataclass fields.

    Sampler additions follow member declaration order (parent methods, then fields recursively), before
    vLLM filters and at most one guide. Set required_publish_lag when the algorithm requires a fixed
    value; every component must agree with the engine's ExecutionOptions.publish_lag.
    """

    required_publish_lag: ClassVar[int | None] = None

    def specialize(
        self, facts: BackendCompileFacts, options: ExecutionOptions, *, causal: CausalBinder | None = None
    ) -> tuple[BackendCompileFacts, TensorBinding]:
        """Check component timing requirements against the single engine setting."""
        for path, component in component_workloads(self):
            required = component.required_publish_lag
            if required is not None and required != options.publish_lag:
                raise ValueError(
                    f"{path or type(component).__name__} requires publish_lag={required}; the engine uses publish_lag={options.publish_lag}; all components must use the engine-wide lag"
                )
        return super().specialize(facts, options, causal=causal)

    def _export_handler(self, declaration: TensorBinding, graph: SemanticGraph) -> Callable[[TraceEvent], None] | None:
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

    def _declare(self, b: TensorBinding, *, capabilities: tuple[str, ...] = ()) -> None:
        """Declare owner values and operations, then compose and sign their graph."""
        context = _Reflection.read(self)
        adapters = {key: value.expand(b.dimensions.layer_count) for key, value in context.adapters.items()}
        for component in context.components:
            component.binding = b
            component.adapters = adapters
        context.declare_values()
        for operation in context.operations:
            description = operation.effect.description(operation.context.owner)
            if operation.phase_override:
                description = replace(description, phase=operation.phase.name.lower())
            with operation.context.declaring(operation.name), b.member(operation.name, description):
                operation.effect.declare(operation)
        context.attach_imports()
        context.attach_effects()
        context.attach_training()
        if all(
            capability in capabilities
            for capability in ("compute/cuda@1", "transition/full_root@1", "backend/tensor_parallel@1")
        ):
            context.overlap_sampler()
        context.validate_pretrained()
        context.sign_composition()

    def build(self, facts: BackendCompileFacts, *, causal: CausalBinder | None = None) -> TensorBinding:
        """Build the graph and tensor tasks against the backend's model dimensions."""
        dimensions = Dimensions(
            facts.probe.hidden_size,
            facts.vocab_size,
            tuple((target.input_size, target.output_size) for target in facts.probe.targets),
            facts.probe.tensor_dtype,
            facts.layer_count,
        )
        b = TensorBinding(type(self).__qualname__, boundary_digest(dimensions, facts.initialization_seed), dimensions)
        b.probe = facts.probe
        b.checkpoint = facts.checkpoint
        b.causal = causal
        self._declare(b, capabilities=facts.timeline.capabilities)
        shape_site_results(b, facts)
        # Executable ports retain their dependencies; an unused factory must not pin the decoder.
        b.causal = None
        b.configure_scans(facts)
        return b

    def backend_requirements(self) -> BackendRequirements:
        """Read model dependencies from markers before dimensions or tensor readers exist."""
        context = _Reflection.read(self)
        sources: dict[str, tuple[tuple[_Source, Workload], ...]] = {}
        targets = tuple(target for value in context.adapters.values() for target in value.targets)
        for operation in context.operations:
            effect = operation.effect
            if isinstance(effect, _TrainInForward):
                targets += _targets(_setting(effect.targets, operation.context.owner, tuple))
                continue
            # Custom residual methods read state and earlier values of the forward, never tapped rows.
            if isinstance(effect, _CustomResidual) or (
                isinstance(effect, _Sampler) and not effect.enabled(operation.context.owner)
            ):
                continue
            with operation.context.declaring(operation.name):
                sources.update(
                    (name, tuple((source, operation.context.owner) for source in arguments.values()))
                    for name, arguments in operation.sources().items()
                )
        owned_inputs = tuple(source for name in sorted(sources) for source in sources[name])
        inputs = tuple(source for source, _ in owned_inputs)
        return BackendRequirements(
            input_token_history=max(
                (source.history for source in inputs if isinstance(source, InputTokens)), default=-1
            ),
            checkpoint_names=tuple(
                name for state in context.state_definitions.values() for name in state.checkpoint_names()
            ),
            exact_phases=tuple(
                dict.fromkeys(
                    phase
                    for op in context.operations
                    if isinstance(op.effect, _TrainInForward)
                    and _option(op.effect.visibility, op.context.owner, ("exact", "forward_pinned"), "visibility")
                    == "exact"
                    for phase in _TOKEN_PHASES[_option(op.effect.tokens, op.context.owner, get_args(Tokens), "tokens")]
                )
            ),
            hidden_inputs=tuple(
                dict.fromkeys(
                    (
                        *(source.resolve(owner) for source, owner in owned_inputs if isinstance(source, Hidden)),
                        *(
                            HiddenInputSource(-1, HiddenSite.LAYER_LOGITS, phase)
                            for source in inputs
                            if isinstance(source, LayerLogits)
                            for phase in (HiddenPhase.PREFILL, HiddenPhase.DECODE)
                        ),
                    )
                )
            ),
            layer_logits=tuple(
                layer
                for source, owner in owned_inputs
                if isinstance(source, LayerLogits)
                for layer in source.layer_tuple(owner)
            ),
            parameter_targets=targets,
            activation_layers=tuple(layer for value in context.activations.values() for layer in value.layers),
            sampler_stages=tuple(
                op.effect.stage
                for op in context.operations
                if isinstance(op.effect, _Sampler) and op.effect.enabled(op.context.owner)
            ),
            sampler_feedback=any(isinstance(source, (BaseLogits, SampledTokens)) for source in inputs),
            sample_limit=min(
                (
                    _setting(source.limit, owner, int)
                    for source, owner in owned_inputs
                    if isinstance(source, SampleIndex) and source.limit is not None
                ),
                default=None,
            ),
            position_states=context.position_states(),
            custom_forward=bool(context.custom_forwards),
            custom_residual=any(isinstance(operation.effect, _CustomResidual) for operation in context.operations),
            stateful_residual=any(
                isinstance(operation.effect, _CustomResidual) and bool(operation.effect.state)
                for operation in context.operations
            ),
            instance_adapter=any(
                context.state_definitions[value.state].scope is StateScope.INSTANCE
                for value in context.adapters.values()
                if value.state in context.state_definitions
            ),
            causal_training=any(isinstance(source, _CausalModelMarker) for source in inputs),
            feeds=any(state.feed is not None for state in context.state_definitions.values()),
        )

    def resource_facts(
        self, declaration: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions
    ) -> SchedulerTimeline:
        """Reserve ordinary tensor scratch and each requested graph's private pool."""
        if declaration.graph.trains_on_batch:
            if options.memory_budget_bytes is not None and not facts.timeline.task_resources:
                raise ValueError(
                    "training on CausalTrainBatch/WeightedRowBatch inputs cannot be funded under "
                    "ExecutionOptions.memory_budget_bytes yet; leave memory_budget_bytes unset"
                )
            timeline = facts.timeline
        else:
            timeline = derive_resources(declaration, facts, options)
        resident = declaration.template_bytes + facts.probe.max_packed_rows * sum(
            prod(source.shape) * source.dtype.itemsize
            for source in {source for rewrite in declaration.sites.rewrites for source in rewrite.results}
        )
        return replace(
            timeline,
            reserved_bytes=tuple(
                value + resident for value in (timeline.reserved_bytes or (0,) * len(timeline.anchors))
            ),
        )

    def bind_math(
        self, facts: BackendCompileFacts, plan: PlanBundle, declaration: TensorBinding
    ) -> MathBinding[StepInput, TensorFrame, TensorOutput]:
        """Bind the planned declaration's tensor tasks to the implementations the plan selected."""
        operations = declaration.executables(facts, plan.timeline.task_resources)
        declaration.require_resident_side_tasks(plan, operations)
        return MathBinding(
            declaration.function_math(facts.device, facts.initialization_seed),
            declaration.bind_all(plan, model_probe=facts.probe, operations=operations),
            feeds=declaration.feeds,
            on_export=self._export_handler(declaration, plan.semantic_graph),
        )

    def compile_host(self, config: HostConfig) -> StateExecution[StepInput, TensorFrame, TensorOutput]:
        """Build, plan and bind the graph without a model (see pllm.host.make_host); training reads host batches."""
        b = TensorBinding(
            type(self).__qualname__,
            boundary_digest(config.hidden_size, config.seed),
            Dimensions(config.hidden_size, 0),
            host=True,
        )
        self._declare(b)
        timeline = config.timeline()
        if config.memory_budget_bytes is not None and b.graph.states:
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
                on_export=self._export_handler(b, b.graph),
            ),
            config,
            timeline,
        )


__all__ = [
    "Workload",
    "custom_forward",
    "Phase",
    "on",
    "write",
    "export",
    "add_logits",
    "guide_sampling",
    "train",
    "Attr",
    "hidden",
    "residual",
    "layer_logits",
    "final_hidden",
    "parameter",
    "module",
    "version",
    "index",
    "sample_index",
    "result",
    "base_logits",
    "sampled_tokens",
    "operand",
    "carry",
    "causal_model",
    "batch",
    "InstanceState",
    "RequestState",
    "train_in_forward",
    "custom_residual",
    "depth",
    "value",
    "Field",
    "UpdateOutput",
    "differentiate",
    "ActivationPatch",
    "Adapter",
    "SGD",
]
