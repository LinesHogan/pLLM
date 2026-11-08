"""Where a declared state's starting values come from, given as `init=` of pllm.InstanceState / RequestState:
pllm.Zeros (the default), pllm.Computed, pllm.Pretrained, pllm.ModelCheckpoint, pllm.FromModule, pllm.Fed and
pllm.Imported. Each source checks the state it fills when the state is declared, and registers its values with the
TensorDeclaration when pllm.execution.workload_lowering allocates the state. Re-exported by pllm."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Literal

import torch

from pllm.contracts.semantic_graph import StateScope
from pllm.contracts.state_feed import FeedBinding, FeedOrder
from pllm.execution.binding_base import Field
from pllm.execution.checkpoint import PretrainedFile, initialize_checkpoint
from pllm.execution.compile_facts import BackendRequirements
from pllm.execution.declaration import Shape, When
from pllm.execution.freeze import freeze
from pllm.execution.model_effect_binding import import_update
from pllm.execution.module_state import ModuleFactory, initial_values, parameter_count
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.tensor_frame import Tensors

if TYPE_CHECKING:
    from pllm.execution.workload_lowering import WorkloadContext
    from pllm.usage import InstanceState

Initializer = Callable[[Tensors, torch.Generator], None]


class StateInit:
    """Where a state's version-zero values come from. Every admission copies the same values."""

    # Where the state lives; only a pllm.Pretrained table may stay in pinned host memory.
    placement: Literal["device", "host"] = "device"

    def check(self, state: InstanceState) -> None:
        """Reject the state declarations this source cannot fill."""

    def fields(self, declared: Mapping[str, Shape | Field]) -> Mapping[str, Shape | Field]:
        """The state's fields; most sources fill the fields the author declared."""
        return declared

    def describe(self, scope: StateScope) -> str:
        """Where the values come from, as pllm.explain reports it."""
        return "zeros"

    def register(self, binding: TensorDeclaration, name: str, scope: StateScope) -> str:
        """Record the source of the state `name`; returns the digest of any file the values are read from."""
        binding.state_sources[name] = self.describe(scope)
        return ""

    def initializer(self, binding: TensorDeclaration, name: str, fields: tuple[str, ...]) -> Initializer | None:
        """The function that fills the named fields of a newly allocated state; None leaves them at zero."""
        return None

    def declare(self, context: WorkloadContext, name: str, state: InstanceState) -> None:
        """Connect the values to the allocated state `name`, beyond its initializer."""

    def check_use(self, name: str, *, written: bool, model_read: bool) -> None:
        """Reject uses the source cannot serve, once every operation of the workload is declared."""

    def requirements(self, state: InstanceState) -> BackendRequirements:
        """What the backend must provide to fill the state."""
        return BackendRequirements()


@dataclass(frozen=True)
class Zeros(StateInit):
    """Every field starts at zero (the default)."""


@dataclass(frozen=True)
class Computed(StateInit):
    """`function(fields, generator)` fills the named field tensors in place, once per build with a seeded
    generator; every admission copies the result."""

    function: Initializer

    def describe(self, scope: StateScope) -> str:
        return "computed"

    def initializer(self, binding: TensorDeclaration, name: str, fields: tuple[str, ...]) -> Initializer | None:
        return freeze(self.function)


@dataclass(frozen=True)
class Pretrained(StateInit):
    """Tensors from a safetensors file: a plain state loads the tensors named like its fields, a module state
    (pllm.FromModule(..., pretrained=path)) a strict state_dict. Values convert to the declared storage dtype.

    placement="host" keeps a read-only table of gathered fields in pinned host memory; "device" (default) never
    falls back to host. Budgets count device residency separately.
    """

    path: str
    placement: Literal["device", "host"] = "device"

    def check(self, state: InstanceState) -> None:
        if self.placement not in ("device", "host"):
            raise ValueError('pllm.Pretrained(placement=) takes "device" or "host"')
        if self.placement == "host" and (
            state.scope is not StateScope.INSTANCE or state.optimizer is not None or state.carry
        ):
            raise ValueError(
                'placement="host" needs a read-only InstanceState of gathered fields; use placement="device" for '
                "mutable state"
            )

    def register(self, binding: TensorDeclaration, name: str, scope: StateScope) -> str:
        file = PretrainedFile.read(self.path, name)
        binding.state_sources[name] = file.source
        binding.pretrained[name] = file
        return file.sha256

    def initializer(self, binding: TensorDeclaration, name: str, fields: tuple[str, ...]) -> Initializer | None:
        return partial(binding.pretrained[name].initialize, fields)

    def check_use(self, name: str, *, written: bool, model_read: bool) -> None:
        if self.placement == "host" and model_read:
            raise ValueError(
                f"{name}: gather host rows in a helper and read pllm.result in the model site; "
                'or use placement="device"'
            )
        if self.placement == "host" and written:
            raise ValueError(f'{name}: placement="host" needs read-only state; use placement="device" for writes')


@dataclass(frozen=True)
class ModelCheckpoint(StateInit):
    """Tensors of the served model's own checkpoint, one name per field (with `layers=`, a pattern formatted with
    each `{layer}`); vLLM skips them when it loads the model. The state is read-only."""

    names: Mapping[str, str]

    def check(self, state: InstanceState) -> None:
        if (
            state.scope is not StateScope.INSTANCE
            or state.optimizer is not None
            or set(self.names) != set(state.fields)
        ):
            raise ValueError("pllm.ModelCheckpoint needs an InstanceState without an optimizer and a name per field")

    def describe(self, scope: StateScope) -> str:
        return "base checkpoint"

    def tensor_names(self, state: InstanceState) -> tuple[str, ...]:
        """Exact checkpoint tensor names, in the same order as the allocated fields."""
        patterns = tuple(self.names[key] for key in state.fields)
        if not state.layers:
            return patterns
        return tuple(pattern.format(layer=layer) for layer in state.layers for pattern in patterns)

    def declare(self, context: WorkloadContext, name: str, state: InstanceState) -> None:
        context.binding.initializers[name] = partial(
            initialize_checkpoint, context.binding.checkpoint, self.tensor_names(state)
        )

    def check_use(self, name: str, *, written: bool, model_read: bool) -> None:
        if written:
            raise ValueError(f"{name}: a pllm.ModelCheckpoint state is read-only; write to a separate state")

    def requirements(self, state: InstanceState) -> BackendRequirements:
        return BackendRequirements(checkpoint_names=self.tensor_names(state))


@dataclass(frozen=True)
class FromModule(StateInit):
    """The parameters of an `nn.Module` built from the model's dimensions become the state's one field, initialized as
    the module initializes them, or from a safetensors state_dict at `pretrained`. pllm.module(state) calls it."""

    factory: ModuleFactory
    pretrained: str = ""

    def fields(self, declared: Mapping[str, Shape | Field]) -> Mapping[str, Shape | Field]:
        if declared:
            raise ValueError("a module state takes its fields from the module; drop fields= and shape=")
        return {"parameters": parameter_count(self.factory)}

    def describe(self, scope: StateScope) -> str:
        return "module init"

    def register(self, binding: TensorDeclaration, name: str, scope: StateScope) -> str:
        if not self.pretrained:
            return super().register(binding, name, scope)
        return Pretrained(self.pretrained).register(binding, name, scope)

    def declare(self, context: WorkloadContext, name: str, state: InstanceState) -> None:
        context.binding.initial_values[name] = partial(
            initial_values,
            self.factory,
            dtype=context.field_dtype(name, "parameters", state.dtype).torch,
            pretrained=context.binding.pretrained.get(name),
        )


@dataclass(frozen=True)
class Fed(StateInit):
    """Filled by a producer process (VllmLlm.feed): "ordered" consumes every publication in order, "latest" only the
    newest. A workload declares at most one fed state."""

    order: Literal["ordered", "latest"]

    def check(self, state: InstanceState) -> None:
        if self.order not in ("ordered", "latest"):
            raise ValueError(f'pllm.Fed({self.order!r}): use "ordered" or "latest"')
        if state.scope is not StateScope.INSTANCE or state.optimizer is not None or state.carry:
            raise ValueError("a fed state needs an InstanceState without an optimizer or carry fields")

    def describe(self, scope: StateScope) -> str:
        return "fed"

    def declare(self, context: WorkloadContext, name: str, state: InstanceState) -> None:
        if context.binding.feeds:
            raise ValueError(f"{context.binding.feeds[0].resource} and {name}: declare one fed state per workload")
        context.binding.feeds = (FeedBinding(name, FeedOrder(self.order), tuple(state.fields)),)

    def requirements(self, state: InstanceState) -> BackendRequirements:
        return BackendRequirements(feeds=True)


@dataclass(frozen=True)
class Imported(StateInit):
    """Supplied by generate(batch=pllm.io.StateImport(...)) and applied before the request's first forward: one row
    per prompt for a RequestState, the field tensors for an InstanceState."""

    def describe(self, scope: StateScope) -> str:
        rows = "one row per prompt" if scope is StateScope.REQUEST else "field tensors"
        return (
            f"imported: {rows} via generate(batch=pllm.io.StateImport(...)), applied before the request's first forward"
        )

    def declare(self, context: WorkloadContext, name: str, state: InstanceState) -> None:
        b, allocated = context.binding, context.states[name]
        views = tuple((field, view) for (resource, field), view in b.field_views.items() if resource == name)
        b.effects = replace(b.effects, import_fields=views)
        b.imports[name] = tuple(view for _, view in views)
        update = import_update(b, allocated, f"{name}:import@1")
        context.writes[name] = (update, b.publish(update, allocated, when=When.BEFORE_FORWARD, name=f"{name}.publish"))
