"""Structured learning surfaces: host transactions and completion learning.

These classes compose the ordinary declaration/binding machinery for learning-shaped
workloads whose mathematics arrives as frozen user functions with a signed optimizer
transition. They are ordinary clients of the DSL, not a second authoring stack.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, TypeVar

import torch

from pllm.contracts.codec import boundary_digest
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    MathContract,
    StateEffectKind,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.binding_base import Buffer
from pllm.execution.declaration import Builder, Dimensions, Field, Shape, When, native
from pllm.execution.freeze import fingerprint, freeze, owner_snapshot, owner_values
from pllm.execution.host_runtime import (
    CompletionInput,
    HostRuntimeFacts,
    HostWorkload,
    TensorHostBinding,
    measured_timeline,
)
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.tensor_frame import Tensors
from pllm.execution.tensor_program import (
    FunctionMath,
    InvocationOutput,
    Roots,
    TaskOutput,
    TensorAdamW,
    TensorBatch,
    autograd_loss,
)
from pllm.execution.torch_private import MemoryUsage
from pllm.kernels.adamw import AdamWConfig
from pllm.usage import Activation, InstanceState, Overlay, RequestState

ContextT = TypeVar("ContextT")
ResultT = TypeVar("ResultT")
TransactionT = TypeVar("TransactionT")
SnapshotT = TypeVar("SnapshotT")
RequestT = TypeVar("RequestT")

_MARKERS = (InstanceState, Activation, Overlay)


class CompletionWorkload(HostWorkload[StateExecution[CompletionInput, ContextT, ResultT]]):
    """Structured completion math with named parameters and one original instance root."""

    name: ClassVar[str] = "completion.learning"

    @property
    @abstractmethod
    def parameters(self) -> InstanceState: ...

    @abstractmethod
    def prepare(self, value: CompletionInput) -> ContextT: ...

    @abstractmethod
    def loss(self, data: ContextT, parameters: Tensors) -> torch.Tensor | None: ...

    @abstractmethod
    def trained_rows(self, data: ContextT) -> int: ...

    @abstractmethod
    def output(self, data: ContextT, result: InvocationOutput) -> ResultT: ...

    def training_example(self, facts: HostRuntimeFacts) -> ContextT:
        """Maximum-shape host input; override when host metadata selects training rows."""
        return self.prepare(CompletionInput.example(facts))

    def declare_host(self, facts: HostRuntimeFacts) -> TensorHostBinding[CompletionInput, ContextT, ResultT]:
        owner = owner_snapshot(self, _MARKERS)
        definition = owner.parameters
        shapes: dict[str, Shape] = {}
        for name, field in definition.fields.items():
            match field:
                case Buffer():
                    break
                case shape:
                    shapes[name] = shape
        if (
            type(definition) is RequestState
            or type(definition.optimizer) is not AdamWConfig
            or len(shapes) != len(definition.fields)
        ):
            raise ValueError(
                f"{type(self).__name__}.parameters must be a pllm.InstanceState of tensor shapes (no Buffer fields) "
                "with optimizer=pllm.AdamW(...)"
            )
        if facts.hidden_size < 1:
            raise ValueError("completion learning needs HostRuntimeFacts.hidden_size >= 1")
        if facts.memory_budget_bytes is not None and facts.max_input_rows is None:
            raise ValueError("set HostRuntimeFacts.max_input_rows when memory_budget_bytes is set")
        identity = fingerprint(owner.loss)
        b = Builder[ContextT](
            self.name,
            boundary_digest(
                facts,
                fingerprint(owner_values(owner, _MARKERS)),
                fingerprint(definition),
                identity,
                fingerprint(owner.trained_rows),
            ),
            Dimensions(facts.hidden_size, 0),
        )
        state = b.state(
            "parameters",
            tuple(
                Field(name, shape, TensorDType(str(definition.dtype).removeprefix("torch.")), definition.alignment)
                for name, shape in shapes.items()
            ),
            namespace=definition.namespace,
            optimizer=definition.optimizer,
        )
        (layout,) = b.layouts.layouts
        optimizer = TensorAdamW(definition.optimizer, len(layout.layout.fields) // 5)
        trained_rows = freeze(owner.trained_rows)
        update = b.compute(
            "train",
            function=autograd_loss(
                freeze(owner.loss),
                tuple(shapes),
                lambda data: TaskOutput(loss_count=1, update_count=1, trained_rows=trained_rows(data)),
            ),
            inputs={"batch": b.input("input", "completions", f"{self.name}.training@1"), "state": state.read()},
            outputs=("control",),
            changes=state,
            effect=StateEffectKind.TRANSITION_MAYBE,
            when=When.HOST_COMPUTE,
            implementations=(
                native("tensor.completion_adamw@1:" + boundary_digest(identity, layout, definition.optimizer)),
            ),
            optimizer=optimizer,
            math=MathContract(
                parameter_gradients=(state.declaration.resource,),
                detached_inputs=("batch",),
                forward_rows="batch",
                parameter_gradient_rows="batch",
                update_states="state",
                reduction="custom",
            ),
        )
        published = b.publish(
            update,
            state,
            output="control",
            schema=f"{self.name}.result@1",
            when=When.HOST_COMMIT,
            implementations=(native("canonical.state:publish@1"),),
        )
        generator = torch.Generator(device=facts.device).manual_seed(facts.initialization_seed)
        initialize_parameters = None if definition.initialize is None else freeze(definition.initialize)
        prepare_data, output_data = freeze(owner.prepare), freeze(owner.output)

        def initialize(roots: Roots, keys: tuple[RequestStateKey | None, ...]) -> None:
            if initialize_parameters is not None:
                (root,) = roots
                initialize_parameters({name: root[5 * index] for index, name in enumerate(shapes)}, generator)

        def prepare(value: CompletionInput, batch: TensorBatch) -> ContextT:
            if (
                value.hidden.shape[1] != facts.hidden_size
                or value.hidden.device != facts.device
                or facts.max_input_rows is not None
                and len(value.hidden) > facts.max_input_rows
            ):
                raise ValueError("completion input shape, device or row count differs from specialization")
            if value.readiness is not None:
                assert batch.stream is not None
                value.readiness.order_on(batch.stream)
                value.hidden.record_stream(batch.stream)
            return prepare_data(value)

        timeline = facts.timeline()
        if facts.memory_budget_bytes is not None:
            assert facts.max_input_rows is not None
            data = owner.training_example(facts)
            root = tuple(
                torch.empty(field.shape, dtype=field.dtype.torch, device="meta") for field in layout.layout.fields
            )
            batch = TensorBatch(RequestStateSelection((), ()), (root,), (), None)
            usage = b.operations[update.symbol].memory(data, batch)
            timeline = measured_timeline(
                facts,
                facts.max_input_rows * facts.hidden_size * 4,
                [(published, MemoryUsage(0, 0)), (update, usage)],
            )
        return TensorHostBinding(
            b.graph,
            b.layouts,
            MathBinding(
                FunctionMath({str(state.declaration.resource): initialize}, lambda value: (), prepare, output_data),
                b.bind_all(operations=b.operations),
            ),
            facts,
            AnchorSymbol("host.compute.end"),
            timeline,
        )
