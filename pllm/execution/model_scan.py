"""Build complete chunk/layer sources and execute them with the shared region and PTTS owners."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import torch

from pllm.contracts.backend_attachment import BackendPlacementProbe
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import ModelScan
from pllm.contracts.region_bindings import RegionBindings
from pllm.contracts.region_program import (
    AccessMode,
    AccessRange,
    AffineIndex,
    BankType,
    Block,
    BorrowedResource,
    Call,
    Function,
    Op,
    Program,
    Publish,
    Region,
    Resource,
    ResourceAccess,
    ScalarType,
    Scan,
    Select,
    SizeParameter,
    Snapshot,
    TensorType,
    Value,
)
from pllm.contracts.schedule_instructions import LoweredSchedule
from pllm.contracts.semantic_graph import ImplementationId, SemanticTask, StateResource, TaskId, TaskSourceRegion
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorLayoutRequest
from pllm.execution.region_compilation import lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import NativePrimitive, PrimitiveResources, external_primitive
from pllm.execution.region_resources import ExternalFunction, RegionResources, ResourcePageCell, TensorResourceBinding
from pllm.execution.region_values import Bank, Data
from pllm.execution.tensor_borrow import TensorResourceView
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
        rows = inputs[0]
        assert type(rows) is int
        x, target, weight, *root = _tensors(inputs[1:])
        with torch.inference_mode(False), torch.enable_grad():
            values = tuple(root[i].detach().clone().requires_grad_(True) for i in self.parameters)
            loss = self.loss(
                {
                    "input": x[:rows].detach().clone(),
                    "target": target[:rows].detach().clone(),
                    "weight": weight.detach().clone(),
                    **dict(zip(self.fields, values, strict=True)),
                }
            )
            gradients = torch.autograd.grad(loss, values)
        self.optimizer.update((tuple(root),), lambda field: (root[field].unsqueeze(0), (0,)), gradients)
        return tuple(root)


def _names(values: tuple[Value, ...]) -> tuple[str, ...]:
    return tuple(value.name for value in values)


def target_resource(state: str, layer: int) -> StateResource:
    """Stable target identity, independent of its position in a declaration."""
    return StateResource(f"{state}.layer{layer:04d}")


def _layer_source(
    layer: int,
    hidden: TensorType,
    resource: Resource | None,
    parameters: tuple[int, ...],
    learning: bool,
) -> tuple[tuple[Value, ...], tuple[Op, ...], tuple[Function, ...]]:
    """One original layer, followed by its optional local update and publication."""
    index = ScalarType.INDEX
    state = () if resource is None else resource.payload
    old = tuple(Value(f"old{layer}_{i}", t) for i, t in enumerate(state))
    payload = tuple(state[i] for i in parameters) if resource is not None else ()
    operand = hidden
    if payload:
        a = payload[0]
        assert isinstance(a, TensorType)
        operand = TensorType((hidden.shape[0], a.shape[1]), hidden.dtype, hidden.device)
    outputs = (Value(f"h{layer + 1}", hidden), Value(f"r{layer}", hidden))
    if resource is not None:
        outputs += (Value(f"x{layer}", operand), Value(f"t{layer}", hidden))
    access = (AccessRange(AffineIndex(layer), AffineIndex(layer + 1)),)
    forward = Function(
        f"layer.{layer}",
        (index, index, hidden, hidden, *payload),
        tuple(v.type for v in outputs),
        ("kv",),
        tuple(ResourceAccess("kv", mode, access) for mode in (AccessMode.READ, AccessMode.WRITE)),
    )
    operations: tuple[Op, ...] = () if resource is None else (Snapshot((), old, resource.name),)
    operations += (
        Call(
            (
                "c",
                "rows",
                f"h{layer}",
                "h0" if layer == 0 else f"r{layer - 1}",
                *(old[i].name for i in parameters if resource is not None),
            ),
            outputs,
            forward.name,
        ),
    )
    if resource is None or not learning:
        return (), operations, (forward,)
    assert len(payload) == 2
    a, b = payload
    assert isinstance(a, TensorType) and isinstance(b, TensorType)
    weight = Value(f"weight{layer}", TensorType((b.shape[0], a.shape[1]), hidden.dtype, hidden.device))
    train = Function(f"train.{layer}", (index, operand, hidden, weight.type, *state), state)
    new = tuple(Value(f"new{layer}_{i}", t) for i, t in enumerate(state))
    return (
        (weight,),
        (
            *operations,
            Call(("rows", f"x{layer}", f"t{layer}", weight.name, *_names(old)), new, train.name),
            Publish(_names(new), (), resource.name),
        ),
        (forward, train),
    )


def model_scan_source(
    scan: ModelScan,
    layouts: TensorLayoutRequest,
    probe: BackendPlacementProbe,
    layers: int,
    parameters: tuple[int, ...],
    *,
    learning: bool,
) -> Program:
    """Expose the complete grid before partitioning, with statically typed layers.

    The chunk axis stays symbolic, including the tail. Padding only fixes the
    value-frame layout; the original kernels and local loss use valid row counts.
    """
    chunk = min(scan.chunk_size, probe.max_packed_rows) if learning else 1
    device = "cpu" if probe.device_kind.value == "cpu" else f"cuda:{probe.device_ordinal}"
    hidden = TensorType((chunk, probe.hidden_size), "torch." + probe.tensor_dtype.value, device)
    bank = BankType(hidden, "chunks")
    args = (Value("inputs", bank), Value("counts", BankType(ScalarType.INDEX, "chunks")))
    ops: tuple[Op, ...] = (
        Select(("inputs", "c"), (Value("h0", hidden),)),
        Select(("counts", "c"), (Value("rows", ScalarType.INDEX),)),
    )
    functions: tuple[Function, ...] = ()
    targets = {
        target.layer: Resource(
            layout.resource, tuple(TensorType(f.shape, "torch." + f.dtype.value, device) for f in layout.layout.fields)
        )
        for target, layout in zip(scan.apply.targets, layouts.layouts, strict=True)
    }
    for layer in range(layers):
        arguments, operations, declarations = _layer_source(layer, hidden, targets.get(layer), parameters, learning)
        args += arguments
        ops += operations
        functions += declarations
    body = Region(Block((Value("c", ScalarType.INDEX), *args), ops, (f"h{layers}", f"r{layers - 1}")))
    entry = Region(
        Block(
            args,
            (
                Scan(
                    _names(args),
                    (Value("outputs", bank), Value("residuals", bank)),
                    "chunks",
                    body,
                    invariants=len(args),
                ),
            ),
            ("outputs", "residuals"),
        )
    )
    return Program(
        entry,
        functions,
        (BorrowedResource("kv", (layers,)), *targets.values()),
        (SizeParameter("chunks", 1, (probe.max_packed_rows + chunk - 1) // chunk),),
    )


class ModelScanPrimitive(Protocol):
    """Prepared model endpoints; the caller owns state, scheduling and optimization."""

    @property
    def row(self) -> PackedPhaseRow: ...
    @property
    def length(self) -> int: ...
    def inputs(self, chunk: int) -> tuple[torch.Tensor, ...]: ...
    def weight(self, layer: int) -> torch.Tensor: ...
    def kv_view(self, resource: BorrowedResource) -> TensorResourceView: ...
    def forward(
        self,
        layer: int,
        start: int,
        rows: int,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        parameters: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]: ...
    def finish(self, outputs: tuple[torch.Tensor, ...], residuals: tuple[torch.Tensor, ...], chunk: int) -> None: ...


@dataclass(frozen=True)
class ModelScanTask(Generic[D]):
    """Run the selected whole-grid instructions on the original invocation stream."""

    task: SemanticTask
    implementation_id: ImplementationId
    scan: ModelScan
    training: ChunkTraining
    sources: tuple[TaskSourceRegion, ...]
    lowered: tuple[LoweredSchedule, ...]
    bindings: tuple[RegionBindings, ...]
    fields: tuple[tuple[StateResource, tuple[LayoutFieldId, ...]], ...]

    @classmethod
    def bind(
        cls,
        task: SemanticTask,
        implementation_id: ImplementationId,
        plan: PlanBundle,
        scan: ModelScan,
        training: ChunkTraining,
    ) -> "ModelScanTask[D]":
        """Lower the complete signed source; no worker constructs a new source graph."""
        sources = tuple(
            next(
                source
                for source in plan.semantic_graph.source_regions
                if source.task == task.symbol and source.name == phase
            )
            for phase in ("prefill", "decode")
        )
        schedules = tuple(
            next(
                selected.schedule_map
                for selected in plan.schedule_physical.source_schedules
                if selected.task == source.task and selected.name == source.name
            )
            for source in sources
        )
        return cls(
            task,
            implementation_id,
            scan,
            training,
            sources,
            tuple(
                lower_region_schedule(source.program, schedule)
                for source, schedule in zip(sources, schedules, strict=True)
            ),
            tuple(
                lower_region_bindings(source.program, schedule)
                for source, schedule in zip(sources, schedules, strict=True)
            ),
            tuple(
                (StateResource(layout.resource), tuple(f.field_id for f in layout.layout.fields))
                for layout in plan.tensor_layout_request.layouts
            ),
        )

    def execute(self, context: "StateTaskContext[D]") -> None:
        """Bind exact per-target refs, then execute the precompiled instruction order."""
        model = context.model_scan
        assert model is not None
        learning = model.row.phase is HiddenPhase.PREFILL
        ordinal = 0 if learning else 1
        source = self.sources[ordinal].program
        chunk = source.entry.block.arguments[0].type
        assert isinstance(chunk, BankType) and isinstance(chunk.element, TensorType)
        rows = chunk.element.shape[0]
        inputs = model.inputs(rows)
        count = len(inputs)
        refs = {
            effect.resource: context.selection.ref_for(model.row.identity, effect.resource)
            for effect in self.task.effects
        }
        cells = tuple(ResourcePageCell(resource, (), refs[resource], fields) for resource, fields in self.fields)
        invocation = ModelScanInvocation(self, context, model, rows, refs)
        functions = {
            f.name: external_primitive(f)
            if f.resources
            else NativePrimitive(f, partial(invocation.train, int(f.name.split(".")[1])))
            for f in source.functions
        }
        kv = source.resources[0]
        assert isinstance(kv, BorrowedResource)
        pages = RegionResources(
            source,
            context,
            cells,
            tuple(
                ExternalFunction(f, partial(invocation.forward, int(f.name.split(".")[1])))
                for f in source.functions
                if f.resources
            ),
            sizes={"chunks": count},
            borrows=(TensorResourceBinding(model.kv_view(kv), context),),
        )
        arguments: dict[str, Data] = {
            "inputs": Bank(inputs),
            "counts": Bank(tuple(min(rows, model.length - c * rows) for c in range(count))),
        }
        if learning:
            arguments.update(
                (f"weight{target.layer}", model.weight(target.layer)) for target in self.scan.apply.targets
            )
        execution = RegionExecution(source, self.lowered[ordinal], self.bindings[ordinal], functions)
        h, r = execution.run(arguments, pages=pages, sizes={"chunks": count}).values
        assert isinstance(h, Bank) and isinstance(r, Bank)
        model.finish(_tensors(h.elements), _tensors(r.elements), rows)


@dataclass
class ModelScanInvocation(Generic[D]):
    """Invocation-local endpoint bindings and diagnostic observations."""

    task: ModelScanTask[D]
    owner: "StateTaskContext[D]"
    model: ModelScanPrimitive
    chunk: int
    refs: Mapping[StateResource, StateRef]

    def forward(self, layer: int, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        c, rows = inputs[:2]
        assert type(c) is int and type(rows) is int
        h, r, *parameters = _tensors(inputs[2:])
        for name, value in zip(("input", "residual_in", "a_read", "b_read"), (h, r, *parameters)):
            self.observe(layer, c, name, value)
        outputs = self.model.forward(layer, c * self.chunk, rows, h, r, tuple(parameters))
        names = ("hidden", "residual", "x", "target") if parameters else ("hidden", "residual")
        for name, value in zip(names, outputs, strict=True):
            self.observe(layer, c, name, value)
        return outputs

    def observe(self, layer: int, chunk: int, name: str, value: torch.Tensor) -> None:
        resource = target_resource(self.task.scan.apply.state, layer)
        refs = () if resource not in self.refs else (self.owner.current_owned_ref(self.refs[resource]),)
        self.owner.trace.observe(
            self.owner.invocation_id,
            TaskId(self.owner.arena.plan.semantic_graph.tasks.index(self.task.task)),
            f"{layer}.{name}",
            value,
            row=self.model.row,
            iteration=chunk,
            refs=refs,
        )

    def train(self, layer: int, inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
        result = self.task.training(inputs, resources)
        # The per-layer exact chain supplies the logical chunk without a device read.
        ref = self.owner.current_owned_ref(self.refs[target_resource(self.task.scan.apply.state, layer)])
        for name, field in zip(self.task.training.fields, self.task.training.parameters, strict=True):
            self.observe(layer, int(ref.version_id), name, _tensors((result[field],))[0])
        self.owner.update_count += 1
        self.owner.loss_count += 1
        rows = inputs[0]
        assert type(rows) is int
        self.owner.trained_rows += rows
        return result


def model_scan_workspace(probe: BackendPlacementProbe, chunk: int, shapes: tuple[tuple[int, ...], ...]) -> int:
    """Bound local derivatives, optimizer scratch and the complete hidden frontier."""
    a, b = shapes
    width = 8 if probe.tensor_dtype is TensorDType.FLOAT64 else 4
    rank, inputs = a
    outputs = b[0]
    return width * (
        2 * inputs * outputs
        + 24 * rank * (inputs + outputs)
        + min(chunk, probe.max_packed_rows) * (8 * inputs + 12 * outputs + 6 * rank)
        + 4 * probe.max_packed_rows * probe.hidden_size
    )
