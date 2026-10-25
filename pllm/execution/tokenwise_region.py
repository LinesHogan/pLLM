"""A single token/layer source program borrowing the original model KV owner."""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping
from copy import copy
from functools import lru_cache, partial
from math import prod
from types import MappingProxyType

import torch

from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_composition import RegionFunctionBody, inline_function_bodies, inline_region
from pllm.contracts.region_imports import BorrowedTensorImport, KernelAllocation, NativeKernelImport, RegionTensorImport
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
from pllm.contracts.region_selection import (
    RegionPayloadLimit,
    RegionSelectionPolicy,
)
from pllm.contracts.region_streams import RegionStreamPlan
from pllm.contracts.semantic_graph import TokenwiseLayerReplay
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
from pllm.execution.region_compilation import analyze_region_program, explore_region_schedules
from pllm.execution.region_execution import RegionExecutionPlan, RegionInstructionExecution, RegionInvocation
from pllm.execution.region_native import NativePrimitive, PrimitiveResources, RegionFunction, source_region_functions
from pllm.execution.region_optimization import prepare_region_execution
from pllm.execution.region_resources import (
    ExternalFunction,
    HostResourceBinding,
    PagedResourceBinding,
    PinnedResourceBinding,
    ReadonlyResourceBinding,
    RegionPageOwner,
    RegionResources,
)
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_streams import RegionStreamBindings, arrange_region_schedule, plan_region_schedule
from pllm.execution.region_values import Bank, Data

Begin = Callable[[int], tuple[torch.Tensor, torch.Tensor]]
Normalize = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
Observe = Callable[[int, torch.Tensor], None]


def mixture_weights(effect: TokenwiseLayerReplay, offsets: tuple[int, ...], device: str) -> torch.Tensor:
    """Upload independently rounded coefficients using the source's double arithmetic."""
    alpha = struct.unpack("!d", struct.pack("!Q", effect.alpha_bits))[0]
    beta = None if effect.beta_bits is None else struct.unpack("!d", struct.pack("!Q", effect.beta_bits))[0]
    alphas = tuple(
        alpha * (min(offset, effect.ramp_tokens) / effect.ramp_tokens) if effect.ramp_tokens else alpha
        for offset in offsets
    )
    return torch.tensor([(a, 1.0 - a if beta is None else beta) for a in alphas], dtype=torch.float32, device=device)


def mix_layer_outputs(source: torch.Tensor, destination: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mix normalized layer taps with the current row's device coefficients."""
    source_float, destination_float = source.float(), destination.float()
    normalized = source_float / source_float.square().sum(-1, keepdim=True).sqrt().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized *= destination_float.square().sum(-1, keepdim=True).sqrt()
    return (weights[1] * destination_float + weights[0] * normalized).to(destination.dtype)


def tokenwise_retained_bytes(hidden: int, width: int, layers: int, positions: int) -> int:
    # Seed, position and coefficient buffers plus fixed single-row attention counters.
    return hidden * width + 16 + layers * (32 + 8 * positions)


class TokenwiseRegion:
    def __init__(
        self,
        tensor: TensorType,
        max_rows: int,
        domain: CausalModelDomain,
        effect: TokenwiseLayerReplay,
        *,
        source_imports: tuple[RegionTensorImport, ...] = (),
        function_bodies: tuple[RegionFunctionBody, ...] = (),
        activations: tuple[DecoderActivation, ...] = (),
    ) -> None:
        if domain.layers != effect.layer_count:
            raise ValueError("tokenwise source differs from the signed model layer domain")
        self.effect = effect
        self.device = tensor.device
        coefficients = TensorType((2,), "torch.float32", tensor.device)
        self.weights = BorrowedResource("weights", (domain.layers + 1,))
        self.context = BorrowedResource("context", (max_rows,))
        self.trace = BorrowedResource("trace", ())
        self.activations = activations
        self.activation_resources = tuple(item.resource(max_rows) for item in activations)
        self.normal = DecoderSource(
            tensor,
            tuple(range(effect.layer_count)),
            domain,
            taps=(effect.source_layer, effect.destination_layer),
            weights=self.weights,
            context=self.context,
            source_imports=source_imports,
            function_bodies=function_bodies,
            activations=activations,
        )
        self.replay = DecoderSource(
            tensor,
            tuple(range(effect.destination_layer + 1, effect.layer_count)),
            domain,
            weights=self.weights,
            context=self.context,
            source_imports=source_imports,
            function_bodies=function_bodies,
            activations=activations,
        )
        self.function_bodies = function_bodies
        index, boolean = ScalarType.INDEX, ScalarType.BOOL
        positions = TensorType((1,), "torch.int64", tensor.device)
        self.domain = domain
        self.kv = kv_resource(domain)
        normal_layers, replay_layers = (
            BankType(index, len(self.normal.layers)),
            BankType(index, len(self.replay.layers)),
        )
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
        blend = Function("tokenwise.mix@2:" + digest, (tensor, tensor, coefficients), (tensor,))

        def blend_(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
            source, destination, weights = inputs
            assert (
                isinstance(source, torch.Tensor)
                and isinstance(destination, torch.Tensor)
                and isinstance(weights, torch.Tensor)
            )
            return (mix_layer_outputs(source, destination, weights),)

        args = (
            Value("row", index),
            Value("request", index),
            Value("positions", positions),
            Value("source", tensor),
            Value("destination", tensor),
            Value("offset", index),
            Value("layer_ids", replay_layers),
            Value("coefficients", coefficients),
        )
        replay = inline_region(
            self.replay.program,
            (
                Value("row", index),
                Value("request", index),
                Value("offset", index),
                Value("layer_ids", replay_layers),
                Value("positions", positions),
                Value("mixed", tensor),
            ),
            "replay",
        )
        replay_body = Region(
            Block(
                args,
                (
                    Call(("source", "destination", "coefficients"), (Value("mixed", tensor),), blend.name),
                    Call(("row", "mixed"), (), self.observe.name),
                    *replay.operations,
                ),
                (),
            )
        )
        skip = Region(Block(args, (), ()))
        normal = inline_region(
            self.normal.program,
            (
                Value("row", index),
                Value("request", index),
                Value("offset", index),
                Value("normal_layers", normal_layers),
                Value("positions", positions),
                Value("seed", tensor),
            ),
            "normal",
        )
        hidden, residual, source, destination = normal.results
        parameters: list[SizeParameter] = []
        constraints: list[SizeConstraint] = []
        arguments = [Value("normal_layers", normal_layers), Value("replay_layers", replay_layers)]
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
            arguments.extend(
                (Value(f"eligible{request}", flags), Value(f"weights{request}", BankType(coefficients, count.name)))
            )
            invariants = (
                Value("eligible", flags),
                Value("weights", BankType(coefficients, count.name)),
                Value("normal_layers", normal_layers),
                Value("replay_layers", replay_layers),
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
                        Select(("weights", "token"), (Value("coefficients", coefficients),)),
                        Call(
                            ("row", "request", "offset"),
                            (Value("positions", positions), Value("seed", tensor)),
                            self.begin.name,
                        ),
                        *normal.operations,
                        Call(
                            ("row", hidden.name, residual.name), (Value("normal_output", tensor),), self.normalize.name
                        ),
                        If(
                            (
                                "enabled",
                                "row",
                                "request",
                                "positions",
                                source.name,
                                destination.name,
                                "offset",
                                "replay_layers",
                                "coefficients",
                            ),
                            (),
                            replay_body,
                            skip,
                        ),
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
                            f"weights{request}",
                            "normal_layers",
                            "replay_layers",
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
        for declaration in (*endpoints, *self.normal.program.functions, *self.replay.program.functions, blend):
            if declaration.name in declarations and declarations[declaration.name] != declaration:
                raise ValueError("tokenwise sources contain conflicting function declarations")
            declarations[declaration.name] = declaration
        cells: dict[str, RegionFunction] = dict(self.normal.functions)
        cells.update(self.replay.functions)
        width = 8 if tensor.dtype == "torch.float64" else 4
        cells[blend.name] = NativePrimitive(
            blend,
            blend_,
            NativeKernelImport(
                blend,
                "tokenwise.mix@2",
                tensor.device,
                (KernelAllocation("mixture", tensor.device, 16 * tensor.shape[1] * width),),
                host_reads=False,
            ),
        )
        self.program = Program(
            Region(Block(tuple(arguments), tuple(operations), tuple(rows))),
            tuple(declarations.values()),
            (self.context, self.kv, self.trace, self.weights, *self.activation_resources),
            tuple(parameters),
            tuple(constraints),
        )
        self.program = inline_function_bodies(self.program, function_bodies) if function_bodies else self.program
        self.functions: Mapping[str, RegionFunction] = MappingProxyType(
            source_region_functions(self.program.functions, cells, source_imports)
        )
        self.imports = import_region_functions(self.program, self.functions, catalog=source_imports)
        schedule = analyze_region_program(self.program, optimize=False).reference_schedule
        width = 8 if tensor.dtype == "torch.float64" else 4
        retained = tokenwise_retained_bytes(tensor.shape[1], width, domain.layers, domain.positions) + sum(
            prod(field.shape) * width for activation in activations for field in activation.fields
        )
        self.retained = (
            RetainedAllocation(
                "tokenwise.buffers",
                RetentionEndpoint(self.begin.name, 0),
                RetentionEndpoint(self.complete.name, 0),
                tensor.device,
                retained,
            ),
        )
        # Reference evidence remains available after selecting a schedule.
        # The catalog's per-device maxima fund the attachment; the selected
        # selected execution retains its own physical stream plan.
        self.payload = analyze_region_payload(self.program, schedule, self.imports, self.retained)
        self.candidate_proposals: tuple[RegionScheduleProposal, ...] = (RegionScheduleProposal(self.payload),)
        # A cold catalog estimates funding; attachment supplies the actual
        # signed workspace envelope before worker implementations are explored.
        self._funded_limits: tuple[RegionPayloadLimit, ...] | None = None
        self.retention = self.payload.retention
        # Reference execution retains one row. Alternate schedules must fit
        # the original combined tensor/buffer reservation.
        if self.retention.allocations[0].peak_bytes_upper_bound > retained:
            raise ValueError("tokenwise schedule exceeds the fixed buffer reservation")
        self._plan: RegionExecutionPlan = RegionExecutionPlan(
            self.program, self.imports, self.candidate_proposals[0], self.payload_limits
        )
        self._execution: RegionInstructionExecution | None = None
        self._policy = RegionSelectionPolicy(allow_numerical_transforms=False, schedule_policy="serial")
        self._bind: Callable[[Mapping[str, RegionFunction]], RegionInstructionExecution] = self._plan.bind
        self._parallel = False
        self._capture = False

    def planned(self, policy: RegionSelectionPolicy, *, stream_lanes: int, token_capture: bool) -> TokenwiseRegion:
        if type(stream_lanes) is not int or not 1 <= stream_lanes <= 8:
            raise ValueError("native catalog requires a bounded original stream capability")
        region = copy(self)
        region._policy = policy
        region._capture = token_capture
        if region._capture:
            return region
        region._parallel = stream_lanes > 1 and any(d.device.startswith("cuda") for d in self.payload.devices)
        proposal = plan_region_schedule(
            region.program,
            region.imports,
            retained=region.retained,
            parallel=region._parallel,
            policy=policy.schedule_policy,
        )
        if proposal is None:
            raise ValueError("tokenwise region has no legal source arrangement")
        proposals = [RegionScheduleProposal(region.payload)]
        if proposal != proposals[0]:
            proposals.append(proposal)
        if policy.schedule_policy == "auto" and policy.tuning == "measured":
            for candidate in explore_region_schedules(region.program, max_candidates=4).candidates:
                if len(proposals) == 4:
                    break
                extra = arrange_region_schedule(
                    region.program,
                    candidate.schedule_map,
                    region.imports,
                    parallel=region._parallel,
                    retained=region.retained,
                )
                if extra not in proposals:
                    proposals.append(extra)
        region.candidate_proposals = tuple(proposals)
        region._install_plan(proposal if policy.tuning == "heuristic" else proposals[0])
        return region

    def _install_plan(self, proposal: RegionScheduleProposal) -> None:
        self._plan = RegionExecutionPlan(self.program, self.imports, proposal, self.payload_limits)
        self._bind = (
            partial(
                prepare_region_execution,
                self.program,
                payload_limits=self.payload_limits,
                options=self._policy,
                parallel=self._parallel,
                retained=self.retained,
            )
            if self._policy.tuning == "heuristic" and not self._capture
            else self._plan.bind
        )

    @property
    def stream_plan(self) -> RegionStreamPlan | None:
        return self._plan.stream_plan if self._execution is None else self._execution.stream_plan

    def bind_functions(self, sources: Mapping[str, RegionFunction]) -> TokenwiseRegion:
        """Bind this worker's original graphs and kernels before measurement."""
        missing = {function.name for function in self.program.functions} - self.functions.keys()
        if set(sources) != missing or self._execution is not None:
            raise ValueError("worker must supply exactly the unbound original tensor implementations")
        result = copy(self)
        result.functions = MappingProxyType({**self.functions, **sources})
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
        amounts: dict[str, int] = {}
        for proposal in self.candidate_proposals:
            for device in proposal.peak_bytes:
                amounts[device.device] = max(amounts.get(device.device, 0), device.bytes)
        return tuple(RegionPayloadLimit(device, amount) for device, amount in sorted(amounts.items()))

    def within_payload_budget(self, budget: int) -> TokenwiseRegion:
        """Keep source candidates funded by the original task workspace fact."""
        if type(budget) is not int or budget < 0 or self._execution is not None:
            raise ValueError("tokenwise payload budgeting requires an original source plan and nonnegative bytes")
        if any(device.peak_bytes_upper_bound > budget for device in self.payload.devices):
            raise ValueError("tokenwise reference payload exceeds the original task workspace budget")
        result = copy(self)
        result.candidate_proposals = tuple(
            proposal
            for proposal in self.candidate_proposals
            if all(device.bytes <= budget for device in proposal.peak_bytes)
        )
        result._funded_limits = tuple(RegionPayloadLimit(device.device, budget) for device in self.payload.devices)
        if self._capture:
            result._install_plan(RegionScheduleProposal(self.payload))
        elif self._policy.tuning == "heuristic":
            proposal = plan_region_schedule(
                self.program,
                self.imports,
                payload_limits=result.payload_limits,
                retained=self.retained,
                parallel=self._parallel,
                policy=self._policy.schedule_policy,
            )
            if proposal is None:
                raise ValueError("tokenwise region exceeds the signed payload budget")
            result.candidate_proposals = tuple(dict.fromkeys((RegionScheduleProposal(self.payload), proposal)))
            result._install_plan(proposal)
        return result

    def with_catalog(
        self, proposals: tuple[RegionScheduleProposal, ...], budget: int, *, stream_lanes: int = 1
    ) -> TokenwiseRegion:
        """Check driver proposals against this source and the signed workspace.

        The wire checkpoint is codec parsing plus local consistency: the
        reference proposal must equal the locally derived reference, budgets
        and stream payloads must agree. Proposal evidence is not re-derived.
        """
        if not proposals:
            raise ValueError("native schedule catalog requires its original reference")
        reference, *_ = proposals
        if (
            len(proposals) > 8
            or reference != RegionScheduleProposal(self.payload)
            or type(budget) is not int
            or budget < 0
            or type(stream_lanes) is not int
            or not 1 <= stream_lanes <= 8
        ):
            raise ValueError("native schedule catalog requires the original reference and workspace")
        if len(
            {
                (p.payload.capacity.schedule_map, None if p.stream_plan is None else p.stream_plan.signature)
                for p in proposals
            }
        ) != len(proposals):
            raise ValueError("native schedule catalog contains duplicate physical proposals")
        limits = tuple(RegionPayloadLimit(device.device, budget) for device in self.payload.devices)
        for proposal in proposals:
            if proposal.stream_plan is not None:
                if proposal.stream_plan.lanes != stream_lanes:
                    raise ValueError("native proposal exceeds the original backend stream capability")
            RegionExecutionPlan(self.program, self.imports, proposal, limits)
        result = copy(self)
        result.candidate_proposals = proposals
        result._funded_limits = limits
        return result

    def with_execution(self, execution: RegionInstructionExecution) -> TokenwiseRegion:
        """Attach the existing worker's admitted executor, retaining the source and owners."""
        if execution.payload_limits != self.payload_limits:
            raise ValueError("bound execution differs from the original attachment payload reservation")
        result = copy(self)
        result._execution = execution
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
        streams: RegionStreamBindings | None = None,
        run_region: Callable[[RegionInvocation], tuple[torch.Tensor, ...]] = RegionInvocation.run,
        kernel_imports: tuple[NativeKernelImport, ...] = (),
        function_bodies: tuple[RegionFunctionBody, ...] = (),
        model_endpoints: tuple[ExternalFunction, ...] = (),
        activation_borrows: tuple[PinnedResourceBinding | ReadonlyResourceBinding, ...] = (),
        activation_selected: Callable[[int, int], bool] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if function_bodies != self.function_bodies:
            raise ValueError("local model source differs from its signed function bodies")

        def begin_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, request, token = inputs
            if (
                type(row) is not int
                or type(request) is not int
                or type(token) is not int
                or (request, token) != (requests[row], offsets[row])
            ):
                raise ValueError("token preparation differs from its original KV coordinates")
            return begin(row)

        def normalize_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, hidden, residual = inputs
            if type(row) is not int or not isinstance(hidden, torch.Tensor) or not isinstance(residual, torch.Tensor):
                raise ValueError("normal output requires its row and residual carry")
            return (normalize(row, hidden, residual),)

        def observe_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, hidden = inputs
            if type(row) is not int or not isinstance(hidden, torch.Tensor):
                raise ValueError("token observation requires its row and hidden value")
            if invocation.cold:
                invocation.observation = hidden
            else:
                observe(row, hidden)
            return ()

        def complete_(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, enabled = inputs
            if type(row) is not int or type(enabled) is not bool:
                raise ValueError("token completion requires an index and branch predicate")

            def record_() -> None:
                if enabled:
                    record(row)

            if not invocation.cold:
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
        coefficients = mixture_weights(self.effect, offsets, self.device)
        start = 0
        for request in range(self.domain.requests):
            end = start
            while end < len(requests) and requests[end] == request:
                end += 1
            prefix = offsets[start] if end > start else 0
            if type(prefix) is not int or any(
                type(offsets[row]) is not int or offsets[row] != prefix + row - start for row in range(start, end)
            ):
                raise ValueError("token offsets must form the original consecutive request span")
            sizes[f"tokens{request}"], sizes[f"prefix{request}"] = end - start, prefix
            inputs[f"eligible{request}"] = Bank(eligible[start:end])
            inputs[f"weights{request}"] = Bank(tuple(coefficients[start:end].unbind()))
            start = end
        kernels = {kernel.declaration.name: kernel for kernel in kernel_imports}
        if len(kernels) != len(kernel_imports):
            raise ValueError("local backend kernel catalog contains duplicate declarations")
        layer_endpoints: dict[str, ExternalFunction] = {}
        for endpoint in (
            *self.normal.externals(layer, kernel_imports),
            *self.replay.externals(layer, kernel_imports),
            *model_endpoints,
        ):
            if endpoint.declaration.name in layer_endpoints:
                if layer_endpoints[endpoint.declaration.name].declaration != endpoint.declaration:
                    raise ValueError("model endpoints contain conflicting declarations")
            else:
                layer_endpoints[endpoint.declaration.name] = endpoint

        def activation_endpoint(activation: DecoderActivation) -> ExternalFunction:
            def selected(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
                (row,) = inputs
                if type(row) is not int or activation_selected is None:
                    raise ValueError("activation selection requires its original packed row")
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
            borrows=(kv, weights, context, trace, *activation_borrows),
            tensor_borrows=tuple(item for item in self.imports if isinstance(item, BorrowedTensorImport)),
        )
        invocation = RegionInvocation(self.execution, inputs, resources, sizes, streams, eligible, observe, record)
        return run_region(invocation)


@lru_cache(maxsize=8)
def compile_tokenwise_source(
    tensor: TensorType,
    max_rows: int,
    domain: CausalModelDomain,
    effect: TokenwiseLayerReplay,
    *,
    source_imports: tuple[RegionTensorImport, ...] = (),
    function_bodies: tuple[RegionFunctionBody, ...] = (),
    activations: tuple[DecoderActivation, ...] = (),
) -> TokenwiseRegion:
    """Reconstruct source and reference lifetimes without optimization search."""

    return TokenwiseRegion(
        tensor,
        max_rows,
        domain,
        effect,
        source_imports=source_imports,
        function_bodies=function_bodies,
        activations=activations,
    )


@lru_cache(maxsize=8)
def compile_tokenwise_region(
    tensor: TensorType,
    max_rows: int,
    domain: CausalModelDomain,
    effect: TokenwiseLayerReplay,
    policy: RegionSelectionPolicy = RegionSelectionPolicy(),
    *,
    stream_lanes: int = 1,
    token_capture: bool = False,
    source_imports: tuple[RegionTensorImport, ...] = (),
    function_bodies: tuple[RegionFunctionBody, ...] = (),
    activations: tuple[DecoderActivation, ...] = (),
) -> TokenwiseRegion:
    """Reuse cold source/lifetime/lowering work across attachment and task binding.

    The cached plan has no invocation, model tensors, page capabilities or RNG.
    Every run creates fresh frames and borrows its original owner's resources.
    """
    return compile_tokenwise_source(
        tensor,
        max_rows,
        domain,
        effect,
        source_imports=source_imports,
        function_bodies=function_bodies,
        activations=activations,
    ).planned(policy, stream_lanes=stream_lanes, token_capture=token_capture)


def tokenwise_attachment_bytes(hidden: int, width: int, max_rows: int) -> int:
    """Original adapter allocations outside the source region workspace.

    Embeddings, packed graph outputs and final concatenation belong to the adapter.
    The mixture is accounted by its region kernel import. Model/layer/attention
    library workspace remains separately owned.
    """
    return 3 * max_rows * hidden * width + 12 * max_rows
