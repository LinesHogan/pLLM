"""Build complete chunk/layer sources and execute them with the shared region and PTTS owners."""

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.backend_attachment import BackendPlacementProbe
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import ModelScan
from pllm.contracts.region_composition import inline_region
from pllm.contracts.region_imports import KernelAllocation, NativeKernelImport, region_import_name
from pllm.contracts.region_program import (
    AccessMode,
    AccessRange,
    AffineIndex,
    BankType,
    Block,
    BorrowedResource,
    Call,
    Coordinate,
    Function,
    IndexTerm,
    Op,
    ParameterTerm,
    Program,
    Publish,
    Region,
    Resource,
    ResourceAccess,
    ResourceType,
    ScalarType,
    Scan,
    Select,
    SizeConstraint,
    SizeParameter,
    Snapshot,
    TensorType,
    Value,
)
from pllm.contracts.region_proposal import ClosedRegionWindow
from pllm.contracts.semantic_graph import ImplementationId, SemanticTask, StateResource, TaskId
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorLayoutRequest
from pllm.execution.codec import boundary_digest
from pllm.execution.exported_cell import ExportedCell, TensorSignature
from pllm.execution.memory import measure
from pllm.execution.region_execution import RegionExecutionPlan, RegionInstructionExecution
from pllm.execution.region_native import BorrowedTensorPrimitive, NativePrimitive, source_region_functions
from pllm.execution.region_resources import (
    ExternalFunction,
    HostResourceBinding,
    ReadonlyResourceBinding,
    RegionPageOwner,
    RegionResources,
    ResourcePageCell,
    TensorResourceBinding,
)
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.region_values import Bank, Data
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.tensor_access import import_tensor_access_graph
from pllm.execution.tensor_borrow import MutableTensorView, ReadonlyTensorView
from pllm.execution.tensor_program import Optimizer
from pllm.execution.trace import MAX_SNAPSHOT_BYTES

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

    def gradients(self, rows: int, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Differentiate only the declared local parameters, retaining the original loss."""
        x, target, weight, *root = inputs
        with torch.inference_mode(False), torch.enable_grad():
            values = tuple(root[i].detach().clone().requires_grad_(True) for i in self.parameters)
            loss = self.loss(
                {
                    "input": x[:rows].detach().clone(),
                    "target": target[:rows].detach().clone(),
                    "weight": weight.detach(),
                    **dict(zip(self.fields, values, strict=True)),
                }
            )
            gradients = torch.autograd.grad(loss, values)
        return gradients

    def capture(self, declaration: Function) -> "CapturedChunkTraining":
        """Import every valid tail shape and the shared optimizer's measured storage."""
        cases = self._gradient_cases(declaration)
        sources = tuple(import_tensor_access_graph(cell) for cell in cases)
        device = cases[0].input_signatures[0].device
        return CapturedChunkTraining(
            cases,
            self.optimizer,
            NativeKernelImport(
                declaration,
                boundary_digest(sources, self.optimizer),
                str(device),
                self._optimizer_allocations(cases[0].input_signatures[3:]),
                host_reads=device.type == "cpu",
                tensor_sources=sources,
            ),
        )

    def _gradient_cases(self, declaration: Function) -> tuple[ExportedCell, ...]:
        with FakeTensorMode():
            operands: list[torch.Tensor] = []
            for type_ in declaration.inputs[4:]:
                assert isinstance(type_, TensorType)
                operands.append(
                    torch.empty(
                        type_.shape,
                        dtype=tensor_dtype(TensorDType(type_.dtype.removeprefix("torch."))),
                        device=type_.device,
                    )
                )
            examples = tuple(operands)
            captured: list[ExportedCell] = []
            for rows in range(1, examples[0].shape[0] + 1):

                def gradients(*values: torch.Tensor, count: int = rows) -> tuple[torch.Tensor, ...]:
                    return self.gradients(count, *values)

                captured.append(
                    ExportedCell.capture(
                        gradients,
                        tuple(f"arg{i}" for i in range(len(examples))),
                        tuple(f"gradient{i}" for i in range(len(self.parameters))),
                        examples,
                    )
                )
        return tuple(captured)

    def _optimizer_allocations(self, types: tuple[TensorSignature, ...]) -> tuple[KernelAllocation, ...]:
        device = types[0].device
        # CPU Adam reads its step on the CPU; CUDA uses the shared capturable path.
        with torch.no_grad(), FakeTensorMode() if device.type == "cuda" else nullcontext():
            roots = tuple(torch.zeros(type_.shape, dtype=type_.dtype, device=type_.device) for type_ in types)
            gradients = tuple(torch.zeros_like(roots[i]) for i in self.parameters)
            scratch = measure(
                lambda: self.optimizer.update((roots,), lambda field: (roots[field].unsqueeze(0), (0,)), gradients)
            )
        return (
            KernelAllocation("optimizer", str(device), scratch.workspace),
            KernelAllocation("observations", str(device), 9 * MAX_SNAPSHOT_BYTES),
            KernelAllocation(
                "blas",
                str(device),
                torch.backends.cuda.cublas_workspace_size() + torch.backends.cuda.cublaslt_workspace_size()
                if device.type == "cuda"
                else 0,
            ),
        )


@dataclass(frozen=True)
class CapturedChunkTraining:
    """A bounded row-count dispatch over imported loss/optimizer tensor graphs."""

    cases: tuple[ExportedCell, ...]
    optimizer: Optimizer
    implementation: NativeKernelImport

    def update(self, rows: int, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate one imported gradient graph and the declared shared optimizer."""
        assert 1 <= rows <= len(self.cases)
        gradients = self.cases[rows - 1].invoke(values)
        root = values[3:]
        self.optimizer.update((root,), lambda field: (root[field].unsqueeze(0), (0,)), gradients)
        return root


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
) -> tuple[tuple[Op, ...], tuple[Function, ...]]:
    """One original layer, followed by its optional local update and publication."""
    index = ScalarType.INDEX
    phase = "prefill" if learning else "decode"
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
    cell = AccessRange(AffineIndex(0, (IndexTerm(0),)), AffineIndex(1, (IndexTerm(0),)))
    layer_cell = AccessRange(AffineIndex(layer), AffineIndex(layer + 1))
    access = (cell, layer_cell)
    forward = Function(
        f"{phase}.layer.{layer}",
        (index, index, index, index, hidden, hidden, *payload),
        tuple(v.type for v in outputs),
        ("kv", "weights", "inputs"),
        (
            *tuple(ResourceAccess("kv", mode, access) for mode in (AccessMode.READ, AccessMode.WRITE)),
            ResourceAccess("weights", AccessMode.READ, (layer_cell,)),
            ResourceAccess("inputs", AccessMode.READ, access),
        ),
    )
    operations: tuple[Op, ...] = () if resource is None else (Snapshot(("request",), old, resource.name),)
    operations += (
        Call(
            (
                "request",
                "c",
                "rows",
                "offset",
                f"h{layer}",
                "h0" if layer == 0 else f"r{layer - 1}",
                *(old[i].name for i in parameters if resource is not None),
            ),
            outputs,
            forward.name,
        ),
    )
    if resource is None or not learning:
        return operations, (forward,)
    assert len(payload) == 2
    a, b = payload
    assert isinstance(a, TensorType) and isinstance(b, TensorType)
    weight = Value(f"weight{layer}", TensorType((b.shape[0], a.shape[1]), hidden.dtype, hidden.device))
    train = Function(
        "train." + boundary_digest(operand, hidden, weight.type, state),
        (index, index, index, index, operand, hidden, weight.type, *state),
        state,
        ("invocation",),
        (ResourceAccess("invocation", AccessMode.READ, ()),),
    )
    load = Function(
        f"weight.{layer}",
        (index,),
        (weight.type,),
        ("weights",),
        (ResourceAccess("weights", AccessMode.READ, (cell,)),),
    )
    new = tuple(Value(f"new{layer}_{i}", t) for i, t in enumerate(state))
    return (
        (
            *operations,
            Coordinate((), (Value(f"layer_id{layer}", index),), AffineIndex(layer)),
            Call((f"layer_id{layer}",), (weight,), load.name),
            Call(
                ("request", f"layer_id{layer}", "c", "rows", f"x{layer}", f"t{layer}", weight.name, *_names(old)),
                new,
                train.name,
            ),
            Publish((*_names(new), "request"), (), resource.name),
        ),
        (forward, train, load),
    )


def _request_source(
    scan: ModelScan,
    layouts: TensorLayoutRequest,
    probe: BackendPlacementProbe,
    layers: int,
    parameters: tuple[int, ...],
    *,
    learning: bool,
    request: int,
    requests: int,
    count: str,
) -> Program:
    """Expose the complete grid before partitioning, with statically typed layers.

    The chunk axis stays symbolic, including the tail. Padding only fixes the
    value-frame layout; the original kernels and local loss use valid row counts.
    """
    assert probe.causal_domain is not None
    chunk = scan.chunk_size if learning else 1
    device = "cpu" if probe.device_kind.value == "cpu" else f"cuda:{probe.device_ordinal}"
    hidden = TensorType((chunk, probe.hidden_size), "torch." + probe.tensor_dtype.value, device)
    bank = BankType(hidden, count)
    args = (
        Value("inputs", bank),
        Value("counts", BankType(ScalarType.INDEX, count)),
        Value("offset", ScalarType.INDEX),
    )
    ops: tuple[Op, ...] = (
        Coordinate((), (Value("request", ScalarType.INDEX),), AffineIndex(request)),
        Select(("inputs", "c"), (Value("h0", hidden),)),
        Select(("counts", "c"), (Value("rows", ScalarType.INDEX),)),
    )
    functions: tuple[Function, ...] = ()
    targets = {
        target.layer: Resource(
            layout.resource,
            tuple(TensorType(f.shape, "torch." + f.dtype.value, device) for f in layout.layout.fields),
            (requests,),
        )
        for target, layout in zip(scan.apply.targets, layouts.layouts, strict=True)
    }
    for layer in range(layers):
        operations, declarations = _layer_source(layer, hidden, targets.get(layer), parameters, learning)
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
                    count,
                    body,
                    invariants=len(args),
                ),
            ),
            ("outputs", "residuals"),
        )
    )
    return Program(
        entry,
        tuple({function.name: function for function in functions}.values()),
        (
            BorrowedResource("kv", (requests, layers)),
            BorrowedResource("weights", (layers,)),
            BorrowedResource("inputs", (requests, layers)),
            BorrowedResource("invocation", ()),
            *targets.values(),
        ),
        (SizeParameter(count, 0, (probe.causal_domain.positions + chunk - 1) // chunk if learning else 1),),
    )


def model_scan_source(
    scan: ModelScan,
    layouts: TensorLayoutRequest,
    probe: BackendPlacementProbe,
    layers: int,
    parameters: tuple[int, ...],
    requests: int,
) -> Program:
    """Compose the full request/phase/chunk/layer source before closed-window selection."""
    if probe.causal_domain is None:
        raise ValueError("ModelScan needs the backend's complete causal model domain")
    arguments: dict[str, Value] = {}
    operations: list[Op] = []
    results: list[str] = []
    functions: dict[str, Function] = {}
    sizes: list[SizeParameter] = []
    constraints: list[SizeConstraint] = []
    resources: tuple[ResourceType, ...] = ()
    maximum = (probe.causal_domain.positions + scan.chunk_size - 1) // scan.chunk_size
    for request in range(requests):
        mode = SizeParameter(f"learning{request}", 0, 1)
        sizes.append(mode)
        for phase, learning in (("prefill", True), ("decode", False)):
            count = f"{phase}{request}"
            part = _request_source(
                scan,
                layouts,
                probe,
                layers,
                parameters,
                learning=learning,
                request=request,
                requests=requests,
                count=count,
            )
            bound = tuple(
                replace(value, name=f"{count}_{value.name}") if value.name in ("inputs", "counts", "offset") else value
                for value in part.entry.block.arguments
            )
            arguments.update((value.name, value) for value in bound)
            body = inline_region(part, bound, count)
            operations.extend(body.operations)
            results.extend(value.name for value in body.results)
            functions.update((function.name, function) for function in part.functions)
            sizes.extend(part.parameters)
            resources = part.resources
        constraints.extend(
            (
                SizeConstraint((ParameterTerm(f"prefill{request}"), ParameterTerm(mode.name, -maximum)), 0),
                SizeConstraint((ParameterTerm(f"decode{request}"), ParameterTerm(mode.name)), 1),
            )
        )
    return Program(
        Region(Block(tuple(arguments.values()), tuple(operations), tuple(results))),
        tuple(functions.values()),
        resources,
        tuple(sizes),
        tuple(constraints),
    )


@dataclass(frozen=True)
class ModelScanSpan:
    """One original request's contiguous engine admission."""

    row: PackedPhaseRow
    length: int
    prompt_length: int

    @property
    def learning(self) -> bool:
        return self.row.phase is HiddenPhase.PREFILL


class ModelScanPrimitive(Protocol):
    """Model endpoints borrow the original inputs, weights, streams and KV owner."""

    @property
    def spans(self) -> tuple[ModelScanSpan, ...]: ...
    def inputs(self, request: int, chunk: int) -> tuple[torch.Tensor, ...]: ...
    def kv_view(self, resource: BorrowedResource) -> MutableTensorView: ...
    def weights_view(self, resource: BorrowedResource) -> ReadonlyTensorView:
        """Original layer tensors, with the adapted projection weight in field zero."""
        ...

    def input_view(self, resource: BorrowedResource) -> ReadonlyTensorView: ...
    def borrow_streams(self, owner: RegionPageOwner, lanes: int) -> RegionStreamBindings: ...
    def forward(
        self,
        request: int,
        layer: int,
        start: int,
        rows: int,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        parameters: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]: ...
    def finish(
        self,
        outputs: tuple[tuple[torch.Tensor, ...], ...],
        residuals: tuple[tuple[torch.Tensor, ...], ...],
        chunks: tuple[int, ...],
    ) -> None: ...


@dataclass(frozen=True)
class _InvocationView:
    resource: BorrowedResource
    namespace: int

    def validate(self) -> None:
        assert not self.resource.domain

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        assert mode is AccessMode.READ and not ranges


@dataclass
class ModelScanTask(Generic[D]):
    """Bound shared executors; every window gets fresh original-owner resource bindings."""

    task: SemanticTask
    implementation_id: ImplementationId
    scan: ModelScan
    window: ClosedRegionWindow
    execution: RegionInstructionExecution
    training: Mapping[str, CapturedChunkTraining]
    fields: tuple[tuple[StateResource, tuple[LayoutFieldId, ...]], ...]
    requests: int

    @classmethod
    def bind(
        cls,
        task: SemanticTask,
        implementation_id: ImplementationId,
        plan: PlanBundle,
        scan: ModelScan,
        training: ChunkTraining,
    ) -> "ModelScanTask[D]":
        """Bind the native-selected cut and exactly the imported endpoint implementations."""
        source = next(source for source in plan.semantic_graph.source_regions if source.task == task.symbol)
        selected = next(item for item in plan.schedule_physical.source_schedules if item.task == source.task)
        window = selected.window
        assert window is not None
        imports = {region_import_name(item): item for item in source.imports}
        captured = {
            function.name: training.capture(function)
            for function in window.program.functions
            if function.name.startswith("train.")
        }
        for name, cell in captured.items():
            if cell.implementation != imports[name]:
                raise ValueError("local loss/optimizer capture differs from its signed source import")
        functions = source_region_functions(window.program.functions, {}, source.imports)
        execution = RegionExecutionPlan(
            window.program, source.imports, window.proposal, window.proposal.peak_bytes, window.state_windows
        ).bind(functions)
        return cls(
            task,
            implementation_id,
            scan,
            window,
            execution,
            MappingProxyType(captured),
            tuple(
                (StateResource(layout.resource), tuple(field.field_id for field in layout.layout.fields))
                for layout in plan.tensor_layout_request.layouts
            ),
            len(window.request.counts),
        )

    def execute(self, context: "StateTaskContext[D]") -> None:
        model = context.model_scan
        assert model is not None
        spans = model.spans
        refs = {
            (request, resource): context.selection.ref_for(span.row.identity, resource)
            for request, span in enumerate(spans)
            for resource, _ in self.fields
        }
        for (request, _), ref in refs.items():
            span = spans[request]
            expected = (
                span.row.token_offset // self.scan.chunk_size
                if span.learning
                else (span.prompt_length + self.scan.chunk_size - 1) // self.scan.chunk_size
            )
            if ref.version_id != expected:
                raise ValueError("admit the exact complete adapter frontier before this model window")
        invocation = ModelScanInvocation(self, context, model, refs)
        arguments, sizes = invocation.arguments()
        resources = {
            resource.name: resource
            for resource in self.window.program.resources
            if isinstance(resource, BorrowedResource)
        }
        pages = RegionResources(
            self.window.program,
            context,
            tuple(
                ResourcePageCell(resource, (request,), refs[request, resource], fields)
                for request in range(len(spans))
                for resource, fields in self.fields
            ),
            invocation.endpoints(),
            sizes=sizes,
            tensor_borrows=tuple(
                cell.implementation
                for cell in self.execution.cells.values()
                if isinstance(cell, BorrowedTensorPrimitive)
            ),
            borrows=(
                TensorResourceBinding(model.kv_view(resources["kv"]), context),
                ReadonlyResourceBinding(model.weights_view(resources["weights"]), context),
                ReadonlyResourceBinding(model.input_view(resources["inputs"]), context),
                HostResourceBinding(_InvocationView(resources["invocation"], id(context)), context),
            ),
        )
        physical = self.execution.stream_plan
        streams = None if physical is None else model.borrow_streams(context, physical.lanes)
        values = self.execution.run(arguments, pages=pages, sizes=sizes, streams=streams).values
        outputs: list[tuple[torch.Tensor, ...]] = []
        residuals: list[tuple[torch.Tensor, ...]] = []
        chunks: list[int] = []
        for request, span in enumerate(spans):
            index = 4 * request + (0 if span.learning else 2)
            hidden, residual = values[index : index + 2]
            assert isinstance(hidden, Bank) and isinstance(residual, Bank)
            outputs.append(_tensors(hidden.elements))
            residuals.append(_tensors(residual.elements))
            chunks.append(self.scan.chunk_size if span.learning else 1)
        model.finish(tuple(outputs), tuple(residuals), tuple(chunks))


@dataclass
class ModelScanInvocation(Generic[D]):
    """Per-window endpoint bindings and diagnostic observations; no page ownership."""

    task: ModelScanTask[D]
    owner: "StateTaskContext[D]"
    model: ModelScanPrimitive
    refs: Mapping[tuple[int, StateResource], StateRef]

    def arguments(self) -> tuple[dict[str, Data], dict[str, int]]:
        arguments: dict[str, Data] = {}
        sizes: dict[str, int] = {}
        for request in range(self.task.requests):
            span = self.model.spans[request] if request < len(self.model.spans) else None
            sizes[f"learning{request}"] = int(span is not None and span.learning)
            for phase, learning in (("prefill", True), ("decode", False)):
                count = f"{phase}{request}"
                chunk = self.task.scan.chunk_size if learning else 1
                active = span is not None and span.learning == learning
                values = self.model.inputs(request, chunk) if active else ()
                sizes[count] = len(values)
                arguments[f"{count}_inputs"] = Bank(values)
                arguments[f"{count}_counts"] = (
                    Bank(tuple(min(chunk, span.length - index * chunk) for index in range(len(values))))
                    if span is not None
                    else Bank(())
                )
                arguments[f"{count}_offset"] = span.row.token_offset if span is not None else 0
        return arguments, sizes

    def endpoints(self) -> tuple[ExternalFunction, ...]:
        result: list[ExternalFunction] = []
        for function in self.task.window.program.functions:
            cell = self.task.execution.cells[function.name]
            if isinstance(cell, BorrowedTensorPrimitive):
                continue
            assert isinstance(cell, NativePrimitive)
            if function.name.startswith("train."):
                callback = partial(self.train, function.name)
            else:
                tensor = function.inputs[4]
                assert isinstance(tensor, TensorType)
                callback = partial(self.forward, int(function.name.split(".")[-1]), tensor.shape[0])
            result.append(ExternalFunction(function, callback, cell.implementation))
        return tuple(result)

    def forward(self, layer: int, chunk: int, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        request, c, rows, offset = inputs[:4]
        assert type(request) is int and type(c) is int and type(rows) is int and type(offset) is int
        h, r, *parameters = _tensors(inputs[4:])
        iteration = offset // chunk + c if self.model.spans[request].learning else 0
        for name, value in zip(("input", "residual_in", "a_read", "b_read"), (h, r, *parameters)):
            self.observe(request, layer, iteration, name, value)
        outputs = self.model.forward(request, layer, offset + c * chunk, rows, h, r, tuple(parameters))
        names = ("hidden", "residual", "x", "target") if parameters else ("hidden", "residual")
        for name, value in zip(names, outputs, strict=True):
            self.observe(request, layer, iteration, name, value)
        return outputs

    def observe(self, request: int, layer: int, chunk: int, name: str, value: torch.Tensor) -> None:
        if not self.owner.trace.observing:
            return
        resource = target_resource(self.task.scan.apply.state, layer)
        key = request, resource
        refs = () if key not in self.refs else (self.owner.current_owned_ref(self.refs[key]),)
        self.owner.trace.observe(
            self.owner.invocation_id,
            TaskId(self.owner.arena.plan.semantic_graph.tasks.index(self.task.task)),
            f"{layer}.{name}",
            value,
            row=self.model.spans[request].row,
            iteration=chunk,
            refs=refs,
        )

    def train(self, name: str, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        request, layer, _, rows = inputs[:4]
        assert type(request) is int and type(layer) is int and type(rows) is int
        cell = self.task.training[name]
        root = cell.update(rows, *_tensors(inputs[4:]))
        ref = self.owner.current_owned_ref(self.refs[request, target_resource(self.task.scan.apply.state, layer)])
        stride = len(root) // cell.optimizer.groups
        for field, label in zip((0, stride), self.task.scan.apply.factors, strict=True):
            self.observe(request, layer, int(ref.version_id), label, root[field])
        self.owner.update_count += 1
        self.owner.loss_count += 1
        self.owner.trained_rows += rows
        return root
