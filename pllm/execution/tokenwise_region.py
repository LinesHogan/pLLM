"""A single token/layer source program borrowing the original model KV owner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from functools import lru_cache
from itertools import islice, zip_longest
from types import MappingProxyType
import struct

import torch

from pllm.contracts.region_composition import RegionFunctionBody, inline_function_bodies, inline_region
from pllm.contracts.region_program import (
    AccessMode, AccessRange, AffineIndex, BankType, Block, BorrowedResource, Call, Constant, Function, If, Op,
    Parameter, ParameterTerm, Program, Region, ResourceAccess, Scalar, ScalarKind, ScalarType, Scan, Select,
    SizeConstraint, SizeParameter, TensorType, Value,
)
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.semantic_graph import TokenwiseLayerReplay
from pllm.contracts.region_retention import RetainedAllocation, RetentionEndpoint
from pllm.contracts.region_imports import BorrowedTensorImport, NativeKernelImport, RegionTensorImport
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_streams import RegionStreamPlan
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelection, RegionSelectionPolicy, RegionSelectionRequest
from pllm.contracts.tensor_rewrite import TensorRewrite
from pllm.execution.codec import boundary_digest
from pllm.execution.decoder_region import DecoderSource, RowLayerForward, decoder_normalize_function, index_cell, kv_resource
from pllm.execution.region_compilation import analyze_region_program, explore_region_schedules
from pllm.execution.region_execution import RegionExecutionPlan, RegionInstructionExecution
from pllm.execution.region_optimization import SelectedRegion, bind_region_candidate, bind_selected_region
from pllm.execution.region_native import NativePrimitive, PrimitiveResources, RegionFunction, original_region_sources, source_region_functions
from pllm.execution.region_resources import ExternalFunction, HostResourceBinding, PagedResourceBinding, ReadonlyResourceBinding, RegionPageOwner, RegionResources
from pllm.execution.region_values import Bank, Data
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_streams import RegionStreamBindings, explore_region_streams


Mix = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
Begin = Callable[[int], tuple[torch.Tensor, torch.Tensor]]
Normalize = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
Observe = Callable[[int, torch.Tensor], None]


def mix_layer_outputs(effect: TokenwiseLayerReplay, source: torch.Tensor,
                      destination: torch.Tensor, position: int) -> torch.Tensor:
    alpha = struct.unpack("!d", struct.pack("!Q", effect.alpha_bits))[0]
    beta = None if effect.beta_bits is None else struct.unpack("!d", struct.pack("!Q", effect.beta_bits))[0]
    if effect.ramp_tokens:
        alpha *= min(position, effect.ramp_tokens) / effect.ramp_tokens
    source_float, destination_float = source.float(), destination.float()
    normalized = source_float / source_float.square().sum(-1, keepdim=True).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
    normalized *= destination_float.square().sum(-1, keepdim=True).sqrt()
    return ((1.0 - alpha if beta is None else beta) * destination_float + alpha * normalized).to(destination.dtype)



def tokenwise_retained_bytes(hidden: int, width: int, layers: int) -> int:
    # Original backend: one cloned current KV slot per layer, one shared int32
    # sequence and at most three converted int64 metadata elements per layer.
    return 12 + layers * (2 * hidden * width + 24)


class TokenwiseRegion:
    def __init__(self, tensor: TensorType, max_rows: int, domain: CausalModelDomain, effect: TokenwiseLayerReplay, mix: Mix, *,
                 source_imports: tuple[RegionTensorImport, ...] = (),
                 function_bodies: tuple[RegionFunctionBody, ...] = ()) -> None:
        if domain.layers != effect.layer_count:
            raise ValueError("tokenwise source differs from the signed model layer domain")
        self.weights = BorrowedResource("weights", (domain.layers + 1,))
        self.context = BorrowedResource("context", (max_rows,))
        self.trace = BorrowedResource("trace", ())
        self.normal = DecoderSource(tensor, tuple(range(effect.layer_count)), domain,
            taps=(effect.source_layer, effect.destination_layer), weights=self.weights, context=self.context,
            source_imports=source_imports, function_bodies=function_bodies)
        self.replay = DecoderSource(tensor, tuple(range(effect.destination_layer + 1, effect.layer_count)), domain,
            weights=self.weights, context=self.context, source_imports=source_imports, function_bodies=function_bodies)
        self.function_bodies = function_bodies
        index, boolean = ScalarType.INDEX, ScalarType.BOOL
        positions = TensorType((1,), "torch.int64", tensor.device)
        self.domain = domain
        self.kv = kv_resource(domain)
        normal_layers, replay_layers = BankType(index, len(self.normal.layers)), BankType(index, len(self.replay.layers))
        digest = boundary_digest(tensor, max_rows, domain, effect)
        backups = ResourceAccess("kv", AccessMode.READ,
            (index_cell(1), AccessRange(AffineIndex(), AffineIndex(domain.layers)), index_cell(2)))
        context_read = ResourceAccess(self.context.name, AccessMode.READ, (index_cell(0),))
        context_write = ResourceAccess(self.context.name, AccessMode.WRITE, (index_cell(0),))
        trace_write = ResourceAccess(self.trace.name, AccessMode.WRITE, ())
        self.begin = Function("tokenwise.begin@3:" + digest, (index, index, index), (positions, tensor), ("context", "kv"), (context_write, backups))
        self.normalize = decoder_normalize_function(tensor, domain.layers)
        self.observe = Function("tokenwise.observe@3:" + digest, (index, tensor), (), ("context", "trace"), (context_read, trace_write))
        self.complete = Function("tokenwise.complete@2:" + digest, (index, boolean), (), ("context", "trace"), (context_write, trace_write))
        blend = Function("tokenwise.mix@1:" + digest, (tensor, tensor, index), (tensor,))
        def blend_(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
            source, destination, position = inputs
            if not isinstance(source, torch.Tensor) or not isinstance(destination, torch.Tensor) or type(position) is not int:
                raise ValueError("token mixture requires hidden tensors and the original token offset")
            return (mix(source, destination, position),)

        args = (Value("row", index), Value("request", index), Value("positions", positions), Value("source", tensor), Value("destination", tensor),
                Value("offset", index), Value("layer_ids", replay_layers))
        replay = inline_region(self.replay.program,
            (Value("row", index), Value("request", index), Value("offset", index), Value("layer_ids", replay_layers),
             Value("positions", positions), Value("mixed", tensor)), "replay")
        replay_body = Region(Block(args, (
            Call(("source", "destination", "offset"), (Value("mixed", tensor),), blend.name),
            Call(("row", "mixed"), (), self.observe.name),
            *replay.operations,
        ), ()))
        skip = Region(Block(args, (), ()))
        normal = inline_region(self.normal.program,
            (Value("row", index), Value("request", index), Value("offset", index), Value("normal_layers", normal_layers),
             Value("positions", positions), Value("seed", tensor)), "normal")
        hidden, residual, source, destination = normal.results
        parameters: list[SizeParameter] = []
        constraints: list[SizeConstraint] = []
        arguments = [Value("normal_layers", normal_layers), Value("replay_layers", replay_layers)]
        operations: list[Op] = [Constant((), (Value("base0", index),), 0)]
        rows: list[str] = []
        for request in range(domain.requests):
            count = SizeParameter(f"tokens{request}", 0, min(max_rows, domain.positions))
            prefix = SizeParameter(f"prefix{request}", 0, domain.positions - 1)
            parameters.extend((count, prefix))
            constraints.append(SizeConstraint((ParameterTerm(prefix.name), ParameterTerm(count.name)), domain.positions))
            flags = BankType(boolean, count.name)
            arguments.append(Value(f"eligible{request}", flags))
            invariants = (Value("eligible", flags), Value("normal_layers", normal_layers), Value("replay_layers", replay_layers),
                     Value("base", index), Value("prefix", index))
            body = Region(Block((Value("token", index), *invariants), (
                Constant((), (Value("request", index),), request),
                Scalar(("base", "token"), (Value("row", index),), ScalarKind.ADD),
                Scalar(("prefix", "token"), (Value("offset", index),), ScalarKind.ADD),
                Select(("eligible", "token"), (Value("enabled", boolean),)),
                Call(("row", "request", "offset"),
                    (Value("positions", positions), Value("seed", tensor)), self.begin.name),
                *normal.operations,
                Call(("row", hidden.name, residual.name),
                    (Value("normal_output", tensor),), self.normalize.name),
                If(("enabled", "row", "request", "positions", source.name, destination.name, "offset", "replay_layers"), (), replay_body, skip),
                Call(("row", "enabled"), (), self.complete.name),
            ), ("normal_output",)))
            outputs = (Value(f"rows{request}", BankType(tensor, count.name)),)
            operations.extend((
                Parameter((), (Value(prefix.name, index),), prefix.name),
                Parameter((), (Value(count.name, index),), count.name),
                Scan((f"eligible{request}", "normal_layers", "replay_layers", f"base{request}", prefix.name),
                    outputs, count.name, body, invariants=len(invariants)),
                Scalar((f"base{request}", count.name), (Value(f"base{request + 1}", index),), ScalarKind.ADD),
            ))
            rows.append(outputs[-1].name)
        constraints.append(SizeConstraint(tuple(ParameterTerm(f"tokens{request}") for request in range(domain.requests)), max_rows))
        endpoints = (self.begin, self.normalize, self.observe, self.complete)
        declarations: dict[str, Function] = {}
        for declaration in (*endpoints, *self.normal.program.functions, *self.replay.program.functions, blend):
            if declaration.name in declarations and declarations[declaration.name] != declaration:
                raise ValueError("tokenwise sources contain conflicting function declarations")
            declarations[declaration.name] = declaration
        cells: dict[str, RegionFunction] = dict(self.normal.functions)
        cells.update(self.replay.functions)
        cells[blend.name] = NativePrimitive(blend, blend_)
        self.program = Program(Region(Block(tuple(arguments), tuple(operations), tuple(rows))),
            tuple(declarations.values()),
            (self.context, self.kv, self.trace, self.weights), tuple(parameters), tuple(constraints))
        self.program = inline_function_bodies(self.program, function_bodies) if function_bodies else self.program
        self.functions: Mapping[str, RegionFunction] = MappingProxyType(
            source_region_functions(self.program.functions, cells, source_imports))
        self.imports = import_region_functions(self.program, self.functions, catalog=source_imports)
        schedule = analyze_region_program(self.program, optimize=False).reference_schedule
        width = 8 if tensor.dtype == "torch.float64" else 4
        retained = tokenwise_retained_bytes(tensor.shape[1], width, domain.layers)
        self.retained = (
            RetainedAllocation("tokenwise.recovery", RetentionEndpoint(self.begin.name, 0),
                RetentionEndpoint(self.complete.name, 0), tensor.device, retained),)
        # Reference evidence remains available after selecting a schedule.
        # The catalog's per-device maxima fund the attachment; the selected
        # schedule's actual witness lives in SelectedRegion.selection.
        self.payload = analyze_region_payload(self.program, schedule, self.imports, self.retained)
        self.candidate_proposals: tuple[RegionScheduleProposal, ...] = (RegionScheduleProposal(self.payload),)
        # A cold catalog estimates funding; attachment supplies the actual
        # signed workspace envelope before worker implementations are explored.
        self._funded_limits: tuple[RegionPayloadLimit, ...] | None = None
        self.retention = self.payload.retention
        # Reference execution retains one row. Alternate schedules must fit
        # the combined tensor/recovery reservation checked by with_schedule.
        if self.retention.allocations[0].peak_bytes_upper_bound > retained:
            raise ValueError("tokenwise schedule exceeds the original recovery reservation")
        self._plan: RegionExecutionPlan | SelectedRegion = RegionExecutionPlan(self.program, self.imports,
            self.candidate_proposals[0], self.payload_limits, self.retained)
        self._execution: RegionInstructionExecution | SelectedRegion | None = None

    @property
    def stream_plan(self) -> RegionStreamPlan | None:
        return self._plan.stream_plan if self._execution is None else self._execution.stream_plan

    def bind_functions(self, sources: Mapping[str, RegionFunction]) -> TokenwiseRegion:
        """Bind this worker's original graphs and kernels before measurement."""
        missing = {function.name for function in self.program.functions} - self.functions.keys()
        if set(sources) != missing or isinstance(self._plan, SelectedRegion):
            raise ValueError("worker must supply exactly the unbound original tensor implementations")
        result = copy(self)
        result.functions = MappingProxyType({**self.functions, **sources})
        result._execution = self._plan.bind(result.functions)
        return result

    @property
    def execution(self) -> RegionInstructionExecution | SelectedRegion:
        if self._execution is None:
            self._execution = self._plan if isinstance(self._plan, SelectedRegion) else self._plan.bind(self.functions)
        return self._execution

    def with_schedule(self, request: RegionSelectionRequest, selection: RegionSelection,
                      policy: RegionSelectionPolicy, records: tuple[TensorRewrite, ...] = ()) -> TokenwiseRegion:
        """Bind measured work within this attachment's original payload envelope.

        The source plan stays reusable. The returned plan borrows the same
        invocation resources and keeps the same recovery/commit callbacks.
        """
        if request.payload_limits != self.payload_limits or request.retention_imports != self.retained:
            raise ValueError("tokenwise selection differs from the original attachment payload reservation")
        execution = bind_selected_region(self.program, original_region_sources(self.functions),
            request, selection, records, options=policy)
        result = copy(self)
        result._plan = execution
        result._execution = None
        return result

    @property
    def payload_limits(self) -> tuple[RegionPayloadLimit, ...]:
        if self._funded_limits is not None:
            return self._funded_limits
        amounts: dict[str, int] = {}
        for proposal in self.candidate_proposals:
            for device in proposal.peak_bytes:
                amounts[device.device] = max(amounts.get(device.device, 0), device.bytes)
        return tuple(RegionPayloadLimit(device, amount) for device, amount in sorted(amounts.items()))

    def within_payload_budget(self, budget: int) -> TokenwiseRegion:
        """Keep source candidates funded by the original task workspace fact."""
        if type(budget) is not int or budget < 0 or isinstance(self._plan, SelectedRegion):
            raise ValueError("tokenwise payload budgeting requires an original source plan and nonnegative bytes")
        if any(device.peak_bytes_upper_bound > budget for device in self.payload.devices):
            raise ValueError("tokenwise reference payload exceeds the original task workspace budget")
        result = copy(self)
        result.candidate_proposals = tuple(proposal for proposal in self.candidate_proposals
            if all(device.bytes <= budget for device in proposal.peak_bytes))
        result._funded_limits = tuple(RegionPayloadLimit(device.device, budget) for device in self.payload.devices)
        return result

    def with_catalog(self, proposals: tuple[RegionScheduleProposal, ...], budget: int, *, stream_lanes: int = 1) -> TokenwiseRegion:
        """Validate driver proposals against this source and the signed workspace.

        Search is not authority: a worker checks each submitted schedule's
        dependence and lifetime evidence without searching for it again.
        """
        if not proposals:
            raise ValueError("native schedule catalog requires its original reference")
        reference, *_ = proposals
        if (isinstance(self._plan, SelectedRegion) or len(proposals) > 8
            or reference != RegionScheduleProposal(self.payload) or type(budget) is not int or budget < 0
            or type(stream_lanes) is not int or not 1 <= stream_lanes <= 8):
            raise ValueError("native schedule catalog requires the original reference and workspace")
        if len({(p.payload.capacity.schedule_map, None if p.stream_plan is None else p.stream_plan.signature)
                for p in proposals}) != len(proposals):
            raise ValueError("native schedule catalog contains duplicate physical proposals")
        limits = tuple(RegionPayloadLimit(device.device, budget) for device in self.payload.devices)
        for proposal in proposals:
            if proposal.stream_plan is not None:
                if proposal.stream_plan.lanes != stream_lanes:
                    raise ValueError("native proposal exceeds the original backend stream capability")
            RegionExecutionPlan(self.program, self.imports, proposal, limits, self.retained)
        result = copy(self)
        result.candidate_proposals = proposals
        result._funded_limits = limits
        return result

    def measurement_candidate(self, request: RegionSelectionRequest, index: int, records: tuple[TensorRewrite, ...],
                              policy: RegionSelectionPolicy) -> TokenwiseRegion:
        """Cold-bind a source-derived implementation for the owning engine's trial."""
        if request.payload_limits != self.payload_limits or request.retention_imports != self.retained:
            raise ValueError("tokenwise trial differs from the original attachment payload reservation")
        result = copy(self)
        result._execution = bind_region_candidate(self.program, original_region_sources(self.functions),
            request, index, records, options=policy)
        return result

    def measurement_schedule(self, index: int) -> TokenwiseRegion:
        """Bind a catalog candidate for an isolated cold invocation, before timing.

        Production installation requires with_schedule and an AER decision.
        This operation only lowers the already funded source schedule.
        """
        if type(index) is not int or not 0 <= index < len(self.candidate_proposals):
            raise ValueError("tokenwise measurement has no such funded candidate")
        proposal = self.candidate_proposals[index]
        result = copy(self)
        result._plan = RegionExecutionPlan(self.program, self.imports, proposal, self.payload_limits, self.retained)
        result._execution = None
        return result

    def run(self, owner: RegionPageOwner, eligible: tuple[bool, ...], requests: tuple[int, ...], offsets: tuple[int, ...], *,
            kv: PagedResourceBinding, weights: ReadonlyResourceBinding, context: HostResourceBinding, trace: HostResourceBinding,
            begin: Begin, layer: RowLayerForward, normalize: Normalize, observe: Observe,
            commit: Callable[[int, Callable[[], None]], None], record: Callable[[int], None],
            restore: Callable[[], None], streams: RegionStreamBindings | None = None,
            kernel_imports: tuple[NativeKernelImport, ...] = (), function_bodies: tuple[RegionFunctionBody, ...] = (),
            model_endpoints: tuple[ExternalFunction, ...] = ()) -> tuple[torch.Tensor, ...]:
        if function_bodies != self.function_bodies:
            raise ValueError("local model source differs from its signed function bodies")
        def begin_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, request, token = inputs
            if (type(row) is not int or type(request) is not int or type(token) is not int
                or (request, token) != (requests[row], offsets[row])):
                raise ValueError("token preparation differs from its original KV coordinates")
            return begin(row)
        def normalize_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, hidden, residual = inputs
            if (type(row) is not int
                or not isinstance(hidden, torch.Tensor) or not isinstance(residual, torch.Tensor)):
                raise ValueError("normal output requires its row and residual carry")
            return (normalize(row, hidden, residual),)
        def observe_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, hidden = inputs
            if type(row) is not int or not isinstance(hidden, torch.Tensor):
                raise ValueError("token observation requires its row and hidden value")
            observe(row, hidden)
            return ()
        def complete_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, enabled = inputs
            if type(row) is not int or type(enabled) is not bool:
                raise ValueError("token completion requires an index and branch predicate")
            def record_() -> None:
                if enabled:
                    record(row)
            commit(row, record_)
            return ()
        if len(eligible) != len(requests) or len(requests) != len(offsets):
            raise ValueError("token metadata must describe the same packed rows")
        if any(type(request) is not int or not 0 <= request < self.domain.requests for request in requests):
            raise ValueError("packed request is outside the signed model domain")
        if any(a > b for a, b in zip(requests, requests[1:])):
            raise ValueError("token rows differ from the original request-major packed plan")
        sizes: dict[str, int] = {}
        inputs: dict[str, Data] = {"normal_layers": Bank(self.normal.layers), "replay_layers": Bank(self.replay.layers)}
        start = 0
        for request in range(self.domain.requests):
            end = start
            while end < len(requests) and requests[end] == request:
                end += 1
            prefix = offsets[start] if end > start else 0
            if (type(prefix) is not int or any(type(offsets[row]) is not int or offsets[row] != prefix + row - start
                                              for row in range(start, end))):
                raise ValueError("token offsets must form the original consecutive request span")
            sizes[f"tokens{request}"], sizes[f"prefix{request}"] = end - start, prefix
            inputs[f"eligible{request}"] = Bank(eligible[start:end])
            start = end
        kernels = {kernel.declaration.name: kernel for kernel in kernel_imports}
        if len(kernels) != len(kernel_imports):
            raise ValueError("local backend kernel catalog contains duplicate declarations")
        layer_endpoints: dict[str, ExternalFunction] = {}
        for endpoint in (*self.normal.externals(layer, kernel_imports), *self.replay.externals(layer, kernel_imports), *model_endpoints):
            if endpoint.declaration.name in layer_endpoints:
                if layer_endpoints[endpoint.declaration.name].declaration != endpoint.declaration:
                    raise ValueError("model endpoints contain conflicting declarations")
            else:
                layer_endpoints[endpoint.declaration.name] = endpoint
        resources = RegionResources(self.program, owner, (), (
            *(ExternalFunction(declaration, callback, kernels.get(declaration.name)) for declaration, callback in (
                (self.begin, begin_), (self.normalize, normalize_), (self.observe, observe_), (self.complete, complete_))
                if declaration.name in self.functions),
            *layer_endpoints.values()),
            sizes=sizes, borrows=(kv, weights, context, trace),
            tensor_borrows=tuple(item for item in self.imports if isinstance(item, BorrowedTensorImport)))
        try:
            results = self.execution.run(inputs, pages=resources, sizes=sizes, streams=streams).values
        except Exception:
            restore()
            raise
        outputs: list[torch.Tensor] = []
        for result in results:
            if not isinstance(result, Bank):
                raise RuntimeError("token region did not return its ordered normal outputs")
            for value in result.elements:
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError("token region returned a non-tensor output")
                outputs.append(value)
        return tuple(outputs)


@lru_cache(maxsize=8)
def compile_tokenwise_source(tensor: TensorType, max_rows: int, domain: CausalModelDomain,
                             effect: TokenwiseLayerReplay, *, source_imports: tuple[RegionTensorImport, ...] = (),
                             function_bodies: tuple[RegionFunctionBody, ...] = ()) -> TokenwiseRegion:
    """Reconstruct source and reference lifetimes without optimization search."""
    def mix(source: torch.Tensor, destination: torch.Tensor, position: int) -> torch.Tensor:
        return mix_layer_outputs(effect, source, destination, position)
    return TokenwiseRegion(tensor, max_rows, domain, effect, mix, source_imports=source_imports, function_bodies=function_bodies)


@lru_cache(maxsize=8)
def compile_tokenwise_region(tensor: TensorType, max_rows: int, domain: CausalModelDomain,
                             effect: TokenwiseLayerReplay, policy: RegionSelectionPolicy = RegionSelectionPolicy(), *,
                             stream_lanes: int = 1, source_imports: tuple[RegionTensorImport, ...] = (),
                             function_bodies: tuple[RegionFunctionBody, ...] = ()) -> TokenwiseRegion:
    """Reuse cold source/lifetime/lowering work across attachment and task binding.

    The cached plan has no invocation, model tensors, page capabilities or RNG.
    Every run creates fresh frames and borrows its original owner's resources.
    """
    if type(stream_lanes) is not int or not 1 <= stream_lanes <= 8:
        raise ValueError("native catalog requires a bounded original stream capability")
    region = copy(compile_tokenwise_source(tensor, max_rows, domain, effect, source_imports=source_imports, function_bodies=function_bodies))
    if policy.schedule_policy == "auto":
        catalog = explore_region_schedules(region.program)
        serial = (RegionScheduleProposal(analyze_region_payload(region.program,
            candidate.schedule_map, region.imports, region.retained)) for candidate in catalog.candidates[1:])
        # The reference plus seven interleaved proposals is the funded catalog.
        # Derive lifetimes only for serial entries that can actually enter it;
        # likewise do not construct a fourth parallel plan that will be dropped.
        parallel_limit = min(4, 7 - min(4, len(catalog.candidates) - 1))
        parallel = (() if stream_lanes == 1 else tuple(RegionScheduleProposal(plan.payload, plan)
            for plan in explore_region_streams(region.program, region.imports, lanes=stream_lanes,
                retained=region.retained, max_candidates=parallel_limit, max_pipeline_depth=2).candidates))
        region.candidate_proposals = (RegionScheduleProposal(region.payload), *islice((proposal
            for pair in zip_longest(serial, parallel) for proposal in pair if proposal is not None), 7))
    return region


def tokenwise_attachment_bytes(hidden: int, width: int, max_rows: int) -> int:
    """Original adapter allocations outside the source region workspace.

    Embeddings and final concatenation remain in the original model adapter.
    The native mixture's internal temporaries retain their existing conservative
    allowance. Model/layer/attention library workspace remains separately owned.
    """
    outside = 2 * max_rows * hidden * width
    mixture = 16 * hidden * width
    return outside + mixture
