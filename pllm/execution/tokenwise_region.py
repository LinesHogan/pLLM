"""A single token/layer source program borrowing the original model KV owner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import replace
from functools import lru_cache
from math import prod
from types import MappingProxyType

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_composition import RegionFunctionBody, inline_function_bodies, inline_region
from pllm.contracts.region_imports import (
    BorrowedTensorImport,
    NativeKernelImport,
    RegionTensorImport,
    region_import_name,
)
from pllm.contracts.region_program import (
    AccessMode,
    AffineIndex,
    BankType,
    Block,
    BorrowedResource,
    Call,
    Coordinate,
    Function,
    If,
    IndexTerm,
    Op,
    Parameter,
    ParameterTerm,
    Program,
    Region,
    ResourceAccess,
    ScalarType,
    Scan,
    Select,
    SizeConstraint,
    SizeParameter,
    TensorType,
    Value,
)
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_retention import RetainedAllocation, RetentionEndpoint
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.semantic_graph import TaskSourceRegion, TokenProgramEffect
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.codec import boundary_digest
from pllm.execution.decoder_region import (
    DecoderActivation,
    DecoderSource,
    RowLayerForward,
    activation_selected_function,
    decoder_normalize_function,
    index_cell,
    kv_resource,
)
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.position_state import POSITION_RESOURCE, position_functions
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_execution import RegionExecutionPlan, RegionInstructionExecution, RegionInvocation
from pllm.execution.region_native import RegionFunction, source_region_functions
from pllm.execution.region_resources import (
    ExternalFunction,
    HostResourceBinding,
    PagedResourceBinding,
    PinnedResourceBinding,
    ReadonlyResourceBinding,
    RegionPageOwner,
    RegionResources,
    TensorResourceBinding,
)
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_values import Bank, Data
from pllm.execution.shared_tensor import tensor_dtype

Begin = Callable[[int], tuple[torch.Tensor, torch.Tensor]]
Normalize = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
Observe = Callable[[int, torch.Tensor], None]


def tokenwise_retained_bytes(hidden: int, width: int, layers: int, positions: int) -> int:
    # Seed and position buffers plus fixed single-row attention counters.
    return hidden * width + 8 + layers * (32 + 8 * positions)


class TokenwiseRegion:
    def __init__(
        self,
        tensor: TensorType,
        max_rows: int,
        domain: CausalModelDomain,
        effect: TokenProgramEffect,
        source: TaskSourceRegion,
        *,
        source_imports: tuple[RegionTensorImport, ...] = (),
        function_bodies: tuple[RegionFunctionBody, ...] = (),
        activations: tuple[DecoderActivation, ...] = (),
    ) -> None:
        assert domain.layers == effect.layer_count
        self.tensor = tensor
        self.weights = BorrowedResource("weights", (domain.layers + 1,))
        self.context = BorrowedResource("context", (max_rows,))
        self.trace = BorrowedResource("trace", ())
        self.activations = activations
        self.activation_resources = tuple(item.resource(max_rows) for item in activations)
        self.overlay = next((resource for resource in source.program.resources if resource.name == "overlay"), None)
        assert self.overlay is None or isinstance(self.overlay, BorrowedResource)
        self.normal = DecoderSource(
            tensor,
            tuple(range(effect.layer_count)),
            domain,
            weights=self.weights,
            context=self.context,
            source_imports=source_imports,
            function_bodies=function_bodies,
            activations=activations,
            overlay=self.overlay,
        )
        self.function_bodies = function_bodies
        index, boolean = ScalarType.INDEX, ScalarType.BOOL
        positions = TensorType((1,), "torch.int64", tensor.device)
        self.domain = domain
        self.kv = kv_resource(domain)
        digest = boundary_digest(tensor, max_rows, domain, effect)
        context_read = ResourceAccess(self.context.name, AccessMode.READ, (index_cell(0),))
        context_write = ResourceAccess(self.context.name, AccessMode.WRITE, (index_cell(0),))
        trace_write = ResourceAccess(self.trace.name, AccessMode.WRITE, ())
        self.begin = Function(
            "tokenwise.begin@3:" + digest,
            (index, index, index),
            (positions, tensor),
            ("context",),
            (context_write,),
        )
        self.normalize = decoder_normalize_function(tensor, domain.layers)
        self.observe = Function(
            "tokenwise.observe@3:" + digest, (index, tensor), (), ("context", "trace"), (context_read, trace_write)
        )
        self.complete = Function(
            "tokenwise.complete@2:" + digest, (index, boolean), (), ("context", "trace"), (context_write, trace_write)
        )
        if self.overlay is not None:
            from pllm.execution.token_program import overlay_accesses

            self.begin, self.complete = (
                replace(
                    function,
                    resources=(*function.resources, self.overlay.name),
                    accesses=(*function.accesses, *overlay_accesses(self.overlay)),
                )
                for function in (self.begin, self.complete)
            )
        args = (
            Value("row", index),
            Value("request", index),
            Value("offset", index),
            Value("positions", positions),
            Value("seed", tensor),
        )
        route = inline_region(source.program, args, "route")
        normal = inline_region(self.normal.program, args, "normal")

        def zero(hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (torch.full(hidden.shape, 0, dtype=hidden.dtype, device=hidden.device),)

        with FakeTensorMode():
            example = torch.empty(
                tensor.shape, dtype=tensor_dtype(TensorDType(tensor.dtype.removeprefix("torch."))), device=tensor.device
            )
            zeros = ExportedCell.capture(zero, ("hidden",), ("zero",), (example,))
        route_body = Region(
            Block(
                args,
                (*route.operations, Call((route.results[0].name,), (Value("zero", tensor),), zeros.digest)),
                (route.results[0].name, "zero"),
            )
        )
        normal_body = Region(Block(args, normal.operations, tuple(value.name for value in normal.results)))
        self.position = POSITION_RESOURCE in source.program.resources
        if self.position:
            entry, store = position_functions(tensor, len(effect.position_states))
            ports = tuple(Value(f"position{i}", type_) for i, type_ in enumerate(entry.outputs))
            normal_body = Region(
                Block(
                    args,
                    (
                        Call((), ports, entry.name),
                        *normal.operations,
                        Call(tuple(value.name for value in ports[1:]), (), store.name),
                    ),
                    tuple(value.name for value in normal.results),
                )
            )
        parameters: list[SizeParameter] = []
        constraints: list[SizeConstraint] = []
        arguments: list[Value] = []
        operations: list[Op] = [Coordinate((), (Value("base0", index),), AffineIndex(0))]
        rows: list[str] = []
        for request in range(domain.requests):
            count = SizeParameter(f"tokens{request}", 0, min(max_rows, domain.positions))
            prefix = SizeParameter(f"prefix{request}", 0, domain.positions - 1)
            parameters.extend((count, prefix))
            constraints.append(
                SizeConstraint((ParameterTerm(prefix.name), ParameterTerm(count.name)), domain.positions)
            )
            flags = BankType(boolean, count.name)
            arguments.append(Value(f"eligible{request}", flags))
            invariants = (
                Value("eligible", flags),
                Value("base", index),
                Value("prefix", index),
            )
            body = Region(
                Block(
                    (Value("token", index), *invariants),
                    (
                        Coordinate((), (Value("request", index),), AffineIndex(request)),
                        Coordinate(
                            ("base", "token"), (Value("row", index),), AffineIndex(0, (IndexTerm(0), IndexTerm(1)))
                        ),
                        Coordinate(
                            ("prefix", "token"), (Value("offset", index),), AffineIndex(0, (IndexTerm(0), IndexTerm(1)))
                        ),
                        Select(("eligible", "token"), (Value("enabled", boolean),)),
                        Call(
                            ("row", "request", "offset"),
                            (Value("positions", positions), Value("seed", tensor)),
                            self.begin.name,
                        ),
                        If(
                            ("enabled", *(value.name for value in args)),
                            (Value("sampled", tensor), Value("residual", tensor)),
                            route_body,
                            normal_body,
                        ),
                        Call(("row", "sampled", "residual"), (Value("normal_output", tensor),), self.normalize.name),
                        Call(("row", "sampled"), (), self.observe.name),
                        Call(("row", "enabled"), (), self.complete.name),
                    ),
                    ("normal_output",),
                )
            )
            outputs = (Value(f"rows{request}", BankType(tensor, count.name)),)
            operations.extend(
                (
                    Parameter((), (Value(prefix.name, index),), prefix.name),
                    Parameter((), (Value(count.name, index),), count.name),
                    Scan(
                        (
                            f"eligible{request}",
                            f"base{request}",
                            prefix.name,
                        ),
                        outputs,
                        count.name,
                        body,
                        invariants=len(invariants),
                    ),
                    Coordinate(
                        (f"base{request}", count.name),
                        (Value(f"base{request + 1}", index),),
                        AffineIndex(0, (IndexTerm(0), IndexTerm(1))),
                    ),
                )
            )
            rows.append(outputs[-1].name)
        constraints.append(
            SizeConstraint(tuple(ParameterTerm(f"tokens{request}") for request in range(domain.requests)), max_rows)
        )
        endpoints = (self.begin, self.normalize, self.observe, self.complete)
        declarations: dict[str, Function] = {}
        for declaration in (*endpoints, *self.normal.program.functions, *source.program.functions, zeros.declaration):
            assert declaration.name not in declarations or declarations[declaration.name] == declaration
            declarations[declaration.name] = declaration
        cells = {**self.normal.functions, zeros.digest: zeros}
        catalog = {region_import_name(imported): imported for imported in (*source.imports, *source_imports)}
        self.program = Program(
            Region(Block(tuple(arguments), tuple(operations), tuple(rows))),
            tuple(declarations.values()),
            (
                self.context,
                self.kv,
                self.trace,
                self.weights,
                *self.activation_resources,
                *(() if self.overlay is None else (self.overlay,)),
                *((POSITION_RESOURCE,) if self.position else ()),
            ),
            tuple(parameters),
            tuple(constraints),
        )
        self.program = inline_function_bodies(self.program, function_bodies) if function_bodies else self.program
        self.functions: Mapping[str, RegionFunction] = MappingProxyType(
            source_region_functions(self.program.functions, cells, tuple(catalog.values()))
        )
        self.imports = import_region_functions(self.program, self.functions, catalog=tuple(catalog.values()))
        schedule = analyze_region_program(self.program, optimize=False).reference_schedule
        width = 8 if tensor.dtype == "torch.float64" else 4
        retained = tokenwise_retained_bytes(tensor.shape[1], width, domain.layers, domain.positions) + sum(
            prod(field.shape) * width for activation in activations for field in activation.fields
        )
        retained += 0 if self.overlay is None else self.overlay.domain[0]
        retained += (3 * prod(tensor.shape) * width + 16) * len(effect.position_states) + 2 * self.position
        self.retained = (
            RetainedAllocation(
                "tokenwise.buffers",
                RetentionEndpoint(self.begin.name, 0),
                RetentionEndpoint(self.complete.name, 0),
                tensor.device,
                retained,
            ),
        )
        self.payload = analyze_region_payload(self.program, schedule, self.imports, self.retained)
        self._funded_limits: tuple[RegionPayloadLimit, ...] | None = None
        self.retention = self.payload.retention
        if self.retention.allocations[0].peak_bytes_upper_bound > retained:
            raise ValueError("tokenwise schedule exceeds the fixed buffer reservation")
        self._plan = RegionExecutionPlan(
            self.program, self.imports, RegionScheduleProposal(self.payload), self.payload_limits
        )
        self._execution: RegionInstructionExecution | None = None
        self._bind: Callable[[Mapping[str, RegionFunction]], RegionInstructionExecution] = self._plan.bind

    def _install_plan(self, proposal: RegionScheduleProposal) -> None:
        self._plan = RegionExecutionPlan(self.program, self.imports, proposal, self.payload_limits)
        self._bind = self._plan.bind

    def bind_functions(self, sources: Mapping[str, RegionFunction]) -> TokenwiseRegion:
        """Bind local source functions, preparing execution when all are available."""
        missing = {function.name for function in self.program.functions} - self.functions.keys()
        if not set(sources) <= missing or self._execution is not None:
            raise ValueError("bind only missing tensor implementations before preparing execution")
        result = copy(self)
        result.functions = MappingProxyType({**self.functions, **sources})
        if set(result.functions) == {function.name for function in result.program.functions}:
            _ = result.execution
        return result

    @property
    def execution(self) -> RegionInstructionExecution:
        if self._execution is None:  # Lazy binding boundary; all invocations use the same concrete executor.
            self._execution = self._bind(self.functions)
        return self._execution

    @property
    def payload_limits(self) -> tuple[RegionPayloadLimit, ...]:
        if self._funded_limits is not None:
            return self._funded_limits
        return RegionScheduleProposal(self.payload).peak_bytes

    def within_payload_budget(self, budget: int) -> TokenwiseRegion:
        """Fund the serial reference plan with the original task workspace fact."""
        if type(budget) is not int or budget < 0 or self._execution is not None:
            raise ValueError("tokenwise payload budgeting requires an original source plan and nonnegative bytes")
        if any(device.peak_bytes_upper_bound > budget for device in self.payload.devices):
            raise ValueError("tokenwise reference payload exceeds the original task workspace budget")
        result = copy(self)
        result._funded_limits = tuple(RegionPayloadLimit(device.device, budget) for device in self.payload.devices)
        result._install_plan(RegionScheduleProposal(result.payload))
        return result

    def run(
        self,
        owner: RegionPageOwner,
        eligible: tuple[bool, ...],
        requests: tuple[int, ...],
        offsets: tuple[int, ...],
        *,
        kv: PagedResourceBinding,
        weights: ReadonlyResourceBinding,
        context: HostResourceBinding,
        trace: HostResourceBinding,
        begin: Begin,
        layer: RowLayerForward,
        normalize: Normalize,
        observe: Observe,
        commit: Callable[[int, Callable[[], None]], None],
        record: Callable[[int], None],
        run_region: Callable[[RegionInvocation], tuple[torch.Tensor, ...]] = RegionInvocation.run,
        kernel_imports: tuple[NativeKernelImport, ...] = (),
        function_bodies: tuple[RegionFunctionBody, ...] = (),
        model_endpoints: tuple[ExternalFunction, ...] = (),
        activation_borrows: tuple[PinnedResourceBinding | ReadonlyResourceBinding, ...] = (),
        overlay_borrows: tuple[TensorResourceBinding, ...] = (),
        position_borrows: tuple[TensorResourceBinding, ...] = (),
        activation_selected: Callable[[int, int], bool] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        assert function_bodies == self.function_bodies

        def begin_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, _request, _token = inputs
            assert type(row) is int
            return begin(row)

        def normalize_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, hidden, residual = inputs
            assert type(row) is int and isinstance(hidden, torch.Tensor) and isinstance(residual, torch.Tensor)
            return (normalize(row, hidden, residual),)

        def observe_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, hidden = inputs
            assert type(row) is int and isinstance(hidden, torch.Tensor)
            if invocation.cold:
                invocation.observation = hidden
            else:
                observe(row, hidden)
            return ()

        def complete_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, enabled = inputs
            assert type(row) is int and type(enabled) is bool

            def record_() -> None:
                if enabled:
                    record(row)

            if not invocation.cold:
                commit(row, record_)
            return ()

        sizes: dict[str, int] = {}
        inputs: dict[str, Data] = {}
        start = 0
        for request in range(self.domain.requests):
            end = start
            while end < len(requests) and requests[end] == request:
                end += 1
            prefix = offsets[start] if end > start else 0
            sizes[f"tokens{request}"], sizes[f"prefix{request}"] = end - start, prefix
            inputs[f"eligible{request}"] = Bank(eligible[start:end])
            start = end
        kernels = {kernel.declaration.name: kernel for kernel in kernel_imports}
        layer_endpoints = {
            endpoint.declaration.name: endpoint
            for endpoint in (*self.normal.externals(layer, kernel_imports), *model_endpoints)
        }

        if self.overlay is not None:
            layer_endpoints.update(
                (function.name, ExternalFunction(function, self.normal.externals(layer)[0].function))
                for function in self.program.functions
                if function.name.startswith(("decoder.first_layer@", "decoder.next_layer@"))
            )

        def activation_endpoint(activation: DecoderActivation) -> ExternalFunction:
            def selected(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
                (row,) = inputs
                assert type(row) is int and activation_selected is not None
                return (activation_selected(row, activation.binding.layer),)

            return ExternalFunction(activation_selected_function(activation), selected)

        resources = RegionResources(
            self.program,
            owner,
            (),
            (
                *(
                    ExternalFunction(declaration, callback, kernels.get(declaration.name))
                    for declaration, callback in (
                        (self.begin, begin_),
                        (self.normalize, normalize_),
                        (self.observe, observe_),
                        (self.complete, complete_),
                    )
                    if declaration.name in self.functions
                ),
                *layer_endpoints.values(),
                *(activation_endpoint(item) for item in self.activations),
            ),
            sizes=sizes,
            borrows=(kv, weights, context, trace, *activation_borrows, *overlay_borrows, *position_borrows),
            tensor_borrows=tuple(item for item in self.imports if isinstance(item, BorrowedTensorImport)),
        )
        invocation = RegionInvocation(self.execution, inputs, resources, sizes, None, eligible, observe, record)
        return run_region(invocation)


@lru_cache(maxsize=8)
def compile_tokenwise_source(
    tensor: TensorType,
    max_rows: int,
    domain: CausalModelDomain,
    effect: TokenProgramEffect,
    source: TaskSourceRegion,
    *,
    source_imports: tuple[RegionTensorImport, ...] = (),
    function_bodies: tuple[RegionFunctionBody, ...] = (),
    activations: tuple[DecoderActivation, ...] = (),
) -> TokenwiseRegion:
    """Build the region once per source; budgeting and binding return copies, so the cached value stays shared."""

    return TokenwiseRegion(
        tensor,
        max_rows,
        domain,
        effect,
        source,
        source_imports=source_imports,
        function_bodies=function_bodies,
        activations=activations,
    )


def tokenwise_attachment_bytes(hidden: int, width: int, max_rows: int) -> int:
    """Original adapter allocations outside the source region workspace.

    Embeddings, packed graph outputs and final concatenation belong to the adapter.
    Route math is accounted by its exported cells. Model/layer/attention library
    workspace remains separately owned.
    """
    return 3 * max_rows * hidden * width + 12 * max_rows
