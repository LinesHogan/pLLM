"""Declaration values and task identities used by tensor workload authoring and binding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import FieldView, Projection, ProjectionShape
from pllm.contracts.semantic_graph import (
    ImplementationId,
    SemanticEffect,
    SemanticState,
    SemanticTask,
    StateEffectKind,
    StateSelectionInputSource,
    TaskImplementationSpec,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.tensor_layout import (
    LayoutFieldId,
    TensorDType,
)
from pllm.execution.tensor_program import (
    ADAM,
    ADAM_FIELDS,
    PageUpdate,
)
from pllm.kernels.adamw import AdamWConfig

OperationMode = Literal["read", "transition", "update", "gradient"]


@dataclass(frozen=True, slots=True)
class Dimensions:
    """Global model widths for workload declarations, and the probed shape of every projection the workload's
    adapters and @pllm.train_in_forward target."""

    hidden_size: int
    vocab_size: int
    targets: tuple[ProjectionShape, ...] = ()
    dtype: TensorDType = TensorDType.FLOAT32
    layer_count: int = 0
    tensor_parallel_size: int = 1

    @property
    def target_shapes(self) -> tuple[tuple[int, int], ...]:
        """(in_features, out_features) of every targeted projection, before tensor-parallel partitioning."""
        return tuple(target.global_shape for target in self.targets)

    def projections(self, targets: tuple[Projection, ...]) -> tuple[ProjectionShape, ...]:
        """The rank-local shapes of `targets`, one projection-delta declaration's subset of all targets."""
        shapes = {shape.target: shape for shape in self.targets}
        return tuple(shapes[target] for target in targets)


Shape = tuple[int, ...] | Callable[[Dimensions], tuple[int, ...]]


class When(Enum):
    AUTOMATIC = ("request.prepare", "next.prepare")
    BEFORE_SAMPLE = ("model.execute", "sampler.pre_draw")
    AFTER_SAMPLE = ("token.commit", "next.prepare")
    BEFORE_FORWARD = ("request.prepare", "model.execute")
    IN_FORWARD = ("model.execute", "state.publish")
    PUBLISH = ("state.publish", "next.prepare")
    HOST_COMPUTE = ("host.input.ready", "host.compute.end")
    HOST_COMMIT = ("host.compute.end", "host.next")
    HOST = ("host.input.ready", "host.next")
    AFTER_FORWARD = ("state.publish", "sampler.pre_draw")

    @property
    def publication(self) -> When:
        """Where a transition computed in this window publishes; a host step publishes only after its compute
        window closes, so an abandoned step publishes nothing."""
        return When.HOST_COMMIT if self is When.HOST_COMPUTE else self


@dataclass(frozen=True, slots=True)
class FieldShard:
    """A contiguous axis slice of one globally initialized field."""

    shape: tuple[int, ...]
    axis: int
    replicas: int = 1

    def read(self, tensor: torch.Tensor, shape: tuple[int, ...], rank: int) -> torch.Tensor:
        """Borrow this rank's slice of a full tensor."""
        return tensor.narrow(self.axis, (rank // self.replicas) * shape[self.axis], shape[self.axis])


@dataclass(frozen=True, slots=True)
class LayoutField:
    name: str
    shape: Shape
    dtype: TensorDType | Callable[[Dimensions], TensorDType] = TensorDType.FLOAT32
    alignment: int = 16
    shard: FieldShard | None = None


@dataclass(frozen=True, slots=True)
class State:
    declaration: SemanticState
    fields: tuple[LayoutField, ...]
    # With AdamW, the first ADAM fields are one flat parameter field and its optimizer fields.
    optimizer: AdamWConfig | None = None

    def read(self, phase: HiddenPhase = HiddenPhase.DECODE) -> StateSelectionInputSource:
        return StateSelectionInputSource(self.declaration.resource, phase)

    def field(self, name: str) -> LayoutFieldId:
        return LayoutFieldId(next(i for i, field in enumerate(self.fields) if field.name == name))

    def view(self, name: str, dimensions: Dimensions | None) -> FieldView:
        """Describe a whole allocated field in the model's resolved layout."""
        index = self.field(name)
        shape = self.fields[index].shape
        return FieldView(
            index,
            0,
            shape(dimensions) if callable(shape) and dimensions is not None else () if callable(shape) else shape,
        )

    def version(self, producer: Call | None = None) -> SemanticEffect:
        return SemanticEffect(
            self.declaration.resource, StateEffectKind.READ, None if producer is None else producer.symbol
        )

    def page_update(self) -> PageUpdate:
        """The AdamW update over this state's optimizer fields."""
        assert self.optimizer is not None
        return PageUpdate(self.optimizer)

    @property
    def carry_fields(self) -> tuple[LayoutField, ...]:
        """The plain carried fields that follow the AdamW fields."""
        return self.fields[0 if self.optimizer is None else ADAM :]

    @property
    def copy_fields(self) -> tuple[LayoutFieldId, ...]:
        """The fields an optimizer update carries into the next version: all but the gradient."""
        gradient = ADAM_FIELDS.index("gradient")
        return tuple(LayoutFieldId(i) for i in range(len(self.fields)) if self.optimizer is None or i != gradient)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The flat AdamW parameter field's name."""
        return (self.fields[0].name.removesuffix(".parameters"),)


def operation_mode(changes: State | None, *, transition: bool) -> OperationMode:
    """How an operation changes its state: not at all, in place, or through a private transition that the state's
    optimizer applies gradients to or the operation writes itself."""
    if changes is None:
        return "read"
    return ("gradient" if changes.optimizer is not None else "transition") if transition else "update"


def planned_implementation(task: SemanticTask, plan: PlanBundle | None) -> ImplementationId:
    """The implementation the plan selected for `task`, or its declared one when binding without a plan."""
    if plan is None:
        return task.implementations[0].implementation_id
    return next(
        item.implementation_id for item in plan.schedule.tasks if plan.semantic_graph.tasks[item.task_id] is task
    )


@dataclass(frozen=True, slots=True)
class Call:
    symbol: TaskSymbol
    outputs: tuple[str, ...]

    def __getitem__(self, name: str) -> TaskResultInputSource:
        if name not in self.outputs:
            raise ValueError(f"{self.symbol}: unknown output {name}")
        return TaskResultInputSource(self.symbol, name)


def native(identity: str, *capabilities: str) -> TaskImplementationSpec:
    return TaskImplementationSpec(ImplementationId(identity), capabilities)
