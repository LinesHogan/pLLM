"""Compile and execute ModelScan through the shared region, optimizer and PTTS machinery."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import torch

from pllm.contracts.backend_attachment import BackendPlacementProbe
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_program import (
    AffineIndex,
    BankType,
    Block,
    Call,
    Coordinate,
    ExternalResource,
    Function,
    If,
    Program,
    Publish,
    Region,
    Resource,
    ScalarType,
    Scan,
    SizeParameter,
    Snapshot,
    TensorType,
    Value,
)
from pllm.contracts.semantic_graph import ImplementationId, SemanticTask, TaskId
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import NativePrimitive, PrimitiveResources, external_primitive
from pllm.execution.region_resources import ExternalFunction, RegionPageOwner, RegionResources, ResourcePageCell
from pllm.execution.region_values import Bank, Data
from pllm.execution.tensor_program import Optimizer

if TYPE_CHECKING:
    from pllm.execution.state_execution import StateTaskContext

D = TypeVar("D")


def _tensors(values: tuple[Data, ...]) -> tuple[torch.Tensor, ...]:
    result: list[torch.Tensor] = []
    for value in values:
        assert isinstance(value, torch.Tensor)
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class ChunkTraining:
    """Local derivatives and the declared optimizer; publication belongs to the region."""

    loss: Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
    fields: tuple[str, ...]
    parameters: tuple[int, ...]
    optimizer: Optimizer

    def __call__(self, inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
        x, target, weight, *root = _tensors(inputs)
        with torch.inference_mode(False), torch.enable_grad():
            values = tuple(root[i].detach().clone().requires_grad_(True) for i in self.parameters)
            loss = self.loss(
                {
                    "input": x.detach().clone(),
                    "target": target.detach().clone(),
                    "weight": weight.detach().clone(),
                    **dict(zip(self.fields, values, strict=True)),
                }
            )
            gradients = torch.autograd.grad(loss, values)
        self.optimizer.update((tuple(root),), lambda field: (root[field].unsqueeze(0), (0,)), gradients)
        return tuple(root)


def _names(values: tuple[Value, ...]) -> tuple[str, ...]:
    return tuple(value.name for value in values)


@dataclass(frozen=True)
class ModelScanRegion:
    """One valid row shape, specialized before any model invocation."""

    program: Program
    execution: RegionExecution
    forward: Function
    observe: Function
    fields: tuple[LayoutFieldId, ...]

    @classmethod
    def compile(
        cls,
        chunk: int,
        max_chunks: int,
        hidden: int,
        weight: TensorType,
        state: tuple[TensorType, ...],
        fields: tuple[LayoutFieldId, ...],
        training: ChunkTraining,
    ) -> "ModelScanRegion":
        """Build serial forward, optimizer and atomic publication statements."""
        index = ScalarType.INDEX
        x, h = (TensorType((chunk, size), weight.dtype, weight.device) for size in (weight.shape[1], hidden))
        parameters = tuple(state[i] for i in training.parameters)
        model = Function("model_scan.forward", (index, *parameters), (h, x, h), ("model",))
        train = Function("model_scan.train", (x, h, weight, *state), state)
        observe = Function("model_scan.observe", (index, h, *parameters), (), ("trace",))
        args = (Value("chunk", index), Value("weight", weight), Value("learning", ScalarType.BOOL))
        old, new = (tuple(Value(f"{prefix}_{i}", t) for i, t in enumerate(state)) for prefix in ("old", "new"))
        inputs = (args[0], Value("hidden", h), Value("x", x), Value("desired", h), args[1], *old)
        learned = Region(
            Block(
                inputs,
                (
                    Coordinate((), (Value("target", index),), AffineIndex(0)),
                    Call(("x", "desired", "weight", *_names(old)), new, train.name),
                    Publish((*_names(new), "target"), (), "adapters"),
                    Call(("chunk", "hidden", *(new[i].name for i in training.parameters)), (), observe.name),
                ),
                (),
            )
        )
        body = Region(
            Block(
                args,
                (
                    Coordinate((), (Value("target", index),), AffineIndex(0)),
                    Snapshot(("target",), old, "adapters"),
                    Call(("chunk", *(old[i].name for i in training.parameters)), inputs[1:4], model.name),
                    If(("learning", *_names(inputs)), (), learned, Region(Block(inputs, (), ()))),
                ),
                ("weight", "learning", "hidden"),
            )
        )
        program = Program(
            Region(
                Block(
                    args[1:],
                    (
                        Scan(
                            ("weight", "learning"),
                            (
                                Value("base", weight),
                                Value("learned", ScalarType.BOOL),
                                Value("outputs", BankType(h, "chunks")),
                            ),
                            "chunks",
                            body,
                        ),
                    ),
                    ("outputs",),
                )
            ),
            (model, train, observe),
            (Resource("adapters", state, (1,)), ExternalResource("model"), ExternalResource("trace")),
            (SizeParameter("chunks", 0, max_chunks),),
        )
        schedule = analyze_region_program(program, optimize=False).reference_schedule
        execution = RegionExecution(
            program,
            lower_region_schedule(program, schedule),
            lower_region_bindings(program, schedule),
            {
                model.name: external_primitive(model),
                train.name: NativePrimitive(train, training),
                observe.name: external_primitive(observe),
            },
        )
        return cls(program, execution, model, observe, fields)

    def run(
        self,
        owner: RegionPageOwner,
        ref: StateRef,
        weight: torch.Tensor,
        chunks: int,
        learning: bool,
        forward: Callable[[tuple[Data, ...]], tuple[Data, ...]],
        observe: Callable[[tuple[Data, ...]], tuple[Data, ...]],
    ) -> tuple[torch.Tensor, ...]:
        """Read exact versions and publish through the original invocation owner."""
        pages = RegionResources(
            self.program,
            owner,
            (ResourcePageCell("adapters", (0,), ref, self.fields),),
            (ExternalFunction(self.forward, forward), ExternalFunction(self.observe, observe)),
            sizes={"chunks": chunks},
        )
        (outputs,) = self.execution.run(
            {"weight": weight, "learning": learning}, pages=pages, sizes={"chunks": chunks}
        ).values
        assert isinstance(outputs, Bank)
        return _tensors(outputs.elements)


class ModelScanPrimitive(Protocol):
    """Prepared model inputs; no state directory or optimizer authority."""

    @property
    def row(self) -> PackedPhaseRow: ...
    @property
    def length(self) -> int: ...
    @property
    def weight(self) -> torch.Tensor: ...
    def forward(
        self, start: int, stop: int, parameters: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None: ...


@dataclass(frozen=True)
class ModelScanTask(Generic[D]):
    """Serial region implementation selected by the signed model attachment."""

    task: SemanticTask
    implementation_id: ImplementationId
    chunk_size: int
    regions: tuple[ModelScanRegion, ...]
    fields: tuple[str, ...]

    @classmethod
    def bind(
        cls,
        task: SemanticTask,
        implementation_id: ImplementationId,
        plan: PlanBundle,
        probe: BackendPlacementProbe,
        chunk_size: int,
        training: ChunkTraining,
    ) -> "ModelScanTask[D]":
        """Compile every valid tail shape before the backend starts requests."""
        (layout,) = plan.tensor_layout_request.layouts
        device = "cpu" if probe.device_kind.value == "cpu" else f"cuda:{probe.device_ordinal}"
        state = tuple(TensorType(field.shape, "torch." + field.dtype.value, device) for field in layout.layout.fields)
        a, b = (state[i] for i in training.parameters)
        weight = TensorType((b.shape[0], a.shape[1]), a.dtype, device)
        regions = tuple(
            ModelScanRegion.compile(
                rows,
                probe.max_packed_rows // rows,
                probe.hidden_size,
                weight,
                state,
                tuple(field.field_id for field in layout.layout.fields),
                training,
            )
            for rows in range(1, min(chunk_size, probe.max_packed_rows) + 1)
        )
        return cls(task, implementation_id, chunk_size, regions, training.fields)

    def execute(self, context: "StateTaskContext[D]") -> None:
        """Use exact snapshots for prefill and the frozen decode forward."""
        model = context.model_scan
        assert model is not None
        learning = model.row.phase is HiddenPhase.PREFILL
        chunk = self.chunk_size if learning else 1
        whole, tail = divmod(model.length, chunk)
        ref = context.selection.ref_for(model.row.identity, self.task.effects[0].resource)
        task_id = TaskId(context.arena.plan.semantic_graph.tasks.index(self.task))
        outputs: tuple[torch.Tensor, ...] = ()
        for offset, rows, count in ((0, chunk, whole), (whole * chunk, tail, 1)):
            if not rows or not count:
                continue

            def forward(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
                index = inputs[0]
                assert type(index) is int
                return model.forward(offset + index * rows, offset + (index + 1) * rows, _tensors(inputs[1:]))

            def observe(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
                index = inputs[0]
                assert type(index) is int
                for name, value in zip(("hidden", *self.fields), _tensors(inputs[1:]), strict=True):
                    context.trace.observe(
                        context.invocation_id,
                        task_id,
                        name,
                        value,
                        row=model.row,
                        iteration=(offset + index * rows) // chunk,
                        refs=(context.current_owned_ref(ref),),
                    )
                context.update_count += 1
                context.loss_count += 1
                context.trained_rows += rows
                return ()

            outputs += self.regions[rows - 1].run(context, ref, model.weight, count, learning, forward, observe)
        model.finish(outputs)


def model_scan_workspace(probe: BackendPlacementProbe, chunk: int, shapes: tuple[tuple[int, ...], ...]) -> int:
    """Bound native/autograd copies, optimizer fields, scratch and packed outputs."""
    a, b = shapes
    width = 8 if probe.tensor_dtype is TensorDType.FLOAT64 else 4
    rank, inputs = a
    outputs = b[0]
    return width * (
        2 * inputs * outputs
        + 24 * rank * (inputs + outputs)
        + min(chunk, probe.max_packed_rows) * (8 * inputs + 12 * outputs + 6 * rank)
        + 2 * probe.max_packed_rows * probe.hidden_size
    )
