"""What a workload method reads: the markers given as its parameter defaults (`h = pllm.hidden(0)`,
`w = pllm.parameter("state", "w")`, ...), and how each lowers to the tensor port, training port or model-site input
the method's task reads. pllm.usage's factories build the markers; pllm.execution.workload_lowering binds and
lowers them."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from inspect import Parameter as SignatureParameter, signature
from typing import TYPE_CHECKING, Literal

import torch

from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.semantic_graph import HiddenInputSource, StateScope, TaskResultInputSource, TaskSymbol
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.attention_methods import AttentionFact, Fact
from pllm.execution.binding_base import Field
from pllm.execution.causal_batch import CausalModel, CausalPort, TrainingBatch
from pllm.execution.compile_facts import BackendRequirements
from pllm.execution.declaration import State, When
from pllm.execution.model_sites import DepthInput, EarlierInput, ResultInput, SiteInput, StateInput
from pllm.execution.module_graph import ModuleCall
from pllm.execution.module_state import skeleton
from pllm.execution.row_task_binding import parameter_rows, sampled_token_port, state_metadata
from pllm.execution.state_init import FromModule
from pllm.execution.tensor_frame import (
    TensorPort,
    Tensors,
    UpdateOutput,
    feedback_port,
    hidden_port,
    input_tokens_port,
    layer_logits_port,
    result_port,
    sample_index_port,
)

if TYPE_CHECKING:
    from pllm.execution.workload_lowering import WorkloadContext
    from pllm.usage import InstanceState


class InputMarker:
    """What one parameter of a workload method reads, given as its default value; each kind lowers itself to the
    port the method's task reads, or rejects the uses it does not support."""

    def bind(self, context: WorkloadContext) -> InputMarker:
        """Resolve the marker against the bound workload once: settings read from it, state and method names
        qualified within its component."""
        return self

    def requirements(self) -> BackendRequirements:
        """What the backend must provide for a method to read this input."""
        return BackendRequirements()

    def port(self, context: WorkloadContext) -> TensorPort:
        raise TypeError(f"unsupported tensor source: {type(self).__name__}")

    def training_port(self, context: WorkloadContext, schema: str) -> CausalPort:
        raise ValueError("causal training inputs must name Batch, Parameter or causal_model ports")

    def in_forward_binding(self, state: str, fields: Mapping[str, Field]) -> str:
        raise TypeError("@pllm.train_in_forward inputs are pllm.operand(...) or pllm.parameter fields of its own state")

    def batch(self, state: str, definition: InstanceState) -> tuple[str, str]:
        raise ValueError("batch training inputs must name its batch tensors or own parameters")

    def site_input(self, context: WorkloadContext) -> SiteInput:
        raise TypeError(
            "@pllm.custom_residual methods read pllm.parameter, pllm.module, pllm.depth, pllm.value, pllm.residual and pre-forward pllm.result"
        )


@dataclass(frozen=True)
class Hidden(InputMarker):
    layer: int
    site: HiddenSite
    phase: HiddenPhase

    def resolve(self) -> HiddenInputSource:
        """Resolve the model dependency without building a tensor binding."""
        return HiddenInputSource(self.layer, self.site, self.phase)

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(hidden_inputs=(self.resolve(),))

    def port(self, context: WorkloadContext) -> TensorPort:
        return hidden_port(self.resolve())

    def site_input(self, context: WorkloadContext) -> SiteInput:
        """Inside a custom residual, pllm.residual reads the same token's stream earlier in this forward."""
        if self.site not in (HiddenSite.RESIDUAL, HiddenSite.ATTENTION_RESIDUAL):
            return super().site_input(context)
        layer = self.layer % context.binding.dimensions.layer_count
        return EarlierInput("residual" if self.site is HiddenSite.RESIDUAL else "attention_residual", layer)


@dataclass(frozen=True)
class Attention(InputMarker):
    layer: int
    fact: Fact

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(custom_attention=True)

    def port(self, context: WorkloadContext) -> TensorPort:
        return context.binding.attention.port(context.binding, AttentionFact(self.layer, self.fact))


@dataclass(frozen=True)
class LayerLogits(InputMarker):
    layers: tuple[int, ...]

    def resolve(self) -> HiddenInputSource:
        """All layer-logit readers of a workload share one input holding every layer any of them names, for every
        row the sampler draws (phase NONE: prompt ends and decode rows)."""
        return HiddenInputSource(-1, HiddenSite.LAYER_LOGITS, HiddenPhase.NONE)

    def requirements(self) -> BackendRequirements:
        """The backend taps the residual stream after each layer, for prompt-end and decode rows."""
        return BackendRequirements(
            hidden_inputs=tuple(
                HiddenInputSource(-1, HiddenSite.LAYER_LOGITS, phase)
                for phase in (HiddenPhase.PREFILL, HiddenPhase.DECODE)
            ),
            layer_logits=self.layers,
        )

    def port(self, context: WorkloadContext) -> TensorPort:
        probe = context.binding.probe
        if probe is None:
            raise ValueError("pllm.layer_logits reads the model's LM head; compile this workload with make_llm")
        count = context.binding.dimensions.layer_count
        columns = tuple(probe.layer_logits.index(layer % count) for layer in self.layers)
        return layer_logits_port(self.resolve(), columns)


@dataclass(frozen=True)
class SampleIndex(InputMarker):
    limit: int | None

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(sample_limit=self.limit)

    def port(self, context: WorkloadContext) -> TensorPort:
        return sample_index_port(context.binding.sampler_rows())


@dataclass(frozen=True)
class Batch(InputMarker):
    field: str

    def training_port(self, context: WorkloadContext, schema: str) -> CausalPort:
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
class Parameter(InputMarker):
    state: str
    field: str
    per_row: bool = False

    def bind(self, context: WorkloadContext) -> Parameter:
        return replace(self, state=context.qualify(self.state))

    def training_port(self, context: WorkloadContext, schema: str) -> CausalPort:
        context.parameter(self.state, self.field)
        view = context.binding.field_views[self.state, self.field]
        return CausalPort(
            context.states[self.state].read(), lambda batch, parameters: view.read(parameters["parameters"])
        )

    def port(self, context: WorkloadContext) -> TensorPort:
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

    def site_input(self, context: WorkloadContext) -> StateInput:
        context.parameter(self.state, self.field)
        return StateInput(
            context.states[self.state].declaration.resource,
            context.binding.field_views[self.state, self.field],
            context.state_definitions[self.state].scope is StateScope.REQUEST,
        )


@dataclass(frozen=True)
class CausalModelInput(InputMarker):
    """An adapter's differentiable views adapted to a worker-local model callable."""

    name: str

    def bind(self, context: WorkloadContext) -> CausalModelInput:
        return replace(self, name=context.qualify(self.name))

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(causal_training=True)

    def training_port(self, context: WorkloadContext, schema: str) -> CausalPort:
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
class ModuleInput(Parameter):
    """A module state, handed to the method as a callable with the module's interface."""

    field: str = "parameters"

    def port(self, context: WorkloadContext) -> TensorPort:
        return replace(super().port(context), adapt=self.adapt(context))

    def adapt(self, context: WorkloadContext) -> Callable[[torch.Tensor], ModuleCall]:
        return partial(ModuleCall, self.skeleton(context), per_row=self.request(context))

    def skeleton(self, context: WorkloadContext) -> torch.nn.Module:
        definition = context.state_definition(self.state, f"{self.state}: pllm.module needs a declared module state")
        if not isinstance(definition.init, FromModule):
            raise TypeError(f"{self.state}: pllm.module needs a state declared with init=pllm.FromModule(...)")
        return skeleton(definition.init.factory, context.binding.dimensions)

    def request(self, context: WorkloadContext) -> bool:
        return context.state_definition(self.state, "pllm.module needs a declared state").scope is StateScope.REQUEST

    def site_input(self, context: WorkloadContext) -> StateInput:
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
class StateIndex(InputMarker):
    state: str

    def bind(self, context: WorkloadContext) -> StateIndex:
        return replace(self, state=context.qualify(self.state))

    def port(self, context: WorkloadContext) -> TensorPort:
        return state_metadata(context.state(self.state, "pllm.index needs a declared state"), version=False)


@dataclass(frozen=True)
class StateVersion(InputMarker):
    state: str

    def bind(self, context: WorkloadContext) -> StateVersion:
        return replace(self, state=context.qualify(self.state))

    def port(self, context: WorkloadContext) -> TensorPort:
        return state_metadata(context.state(self.state, "pllm.version needs a declared state"), version=True)


@dataclass(frozen=True)
class Result(InputMarker):
    operation: str

    def bind(self, context: WorkloadContext) -> Result:
        return replace(self, operation=context.qualify(self.operation))

    def port(self, context: WorkloadContext) -> TensorPort:
        return result_port(TaskResultInputSource(TaskSymbol(self.operation), "value"))

    def site_input(self, context: WorkloadContext) -> SiteInput:
        operation = next(op for op in context.operations if op.name == self.operation)
        if operation.phase is not When.BEFORE_FORWARD or not operation.plain:
            raise ValueError(f"{self.operation}: custom residuals read results of read-only BEFORE_FORWARD methods")
        return ResultInput(TaskSymbol(self.operation))


@dataclass(frozen=True)
class InputTokens(InputMarker):
    """The request token ids read by each row, with a fixed number of lookbacks."""

    history: int = 0

    def requirements(self) -> BackendRequirements:
        return BackendRequirements(input_token_history=self.history)

    def port(self, context: WorkloadContext) -> TensorPort:
        return input_tokens_port(self.history)


@dataclass(frozen=True)
class Depth(InputMarker):
    def site_input(self, context: WorkloadContext) -> SiteInput:
        return DepthInput()


@dataclass(frozen=True)
class Value(InputMarker):
    layer: int

    def site_input(self, context: WorkloadContext) -> SiteInput:
        return EarlierInput("value", self.layer % context.binding.dimensions.layer_count)


@dataclass(frozen=True)
class BaseLogits(InputMarker):
    def requirements(self) -> BackendRequirements:
        return BackendRequirements(sampler_feedback=True)

    def port(self, context: WorkloadContext) -> TensorPort:
        return feedback_port("base_logits")


@dataclass(frozen=True)
class SampledTokens(InputMarker):
    def requirements(self) -> BackendRequirements:
        return BackendRequirements(sampler_feedback=True)

    def port(self, context: WorkloadContext) -> TensorPort:
        return sampled_token_port()


OperandName = Literal["input", "target", "embedding", "weight", "prompt_end"]


@dataclass(frozen=True)
class Operand(InputMarker):
    """A named model operand, selected when the workload is declared."""

    name: OperandName


@dataclass(frozen=True)
class Carried(InputMarker):
    """An input whose value a training task also writes into a carried field."""

    source: InputMarker
    carry: Carry

    def bind(self, context: WorkloadContext) -> Carried:
        return replace(
            self,
            source=self.source.bind(context),
            carry=replace(self.carry, state=context.qualify(self.carry.state)),
        )


def method_inputs(
    function: Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | UpdateOutput | None],
) -> dict[str, InputMarker]:
    """Read each parameter's marker from its default value, e.g. `h: torch.Tensor = pllm.hidden(0)`."""
    result: dict[str, InputMarker] = {}
    for name, parameter in signature(function, eval_str=False).parameters.items():
        if not isinstance(parameter.default, InputMarker) or parameter.kind is SignatureParameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{name}: give every input a source as its default, e.g. `{name}: torch.Tensor = pllm.hidden(0)`"
            )
        result[name] = parameter.default
    return result
