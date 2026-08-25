#!/usr/bin/env python3
"""Program runtime: compile a declared Program into a BaseConsumer.

``compile_program`` is the control-plane entrypoint: it validates the
declaration (``tllm.program.decl``), captures and slices each flow
(``tllm.program.stages``), binds sinks, derives the functional optimizer
transition from the instance's reference optimizer, and runs the
cloned-state reference oracle — the extern's Python body is executed once on
a deep-cloned instance and must match the functional transition within
dtype-aware tolerance, or compilation fails (PG145). There is no silent
fallback to eager Python anywhere on the trained path.

The generated ``ProgramConsumer`` dispatches ``consume_flow(ordinal, ...)``:
it gathers declared inputs from the bundle's ``device_lease`` (entry key =
flow parameter name), executes stages eagerly (per state-slot row group for
REQUEST banks), publishes sink outputs, and runs due externs as functional
optimizer transitions on the versioned state bank.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Hashable, Protocol, cast, runtime_checkable

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.ports.consumer import BaseConsumer
from tllm.program.decl import ProgramDecl, ProgramNode, StateSpec, declare_program
from tllm.program.optim import (
    AdamScalars,
    OptState,
    SGDScalars,
    adam_step_inplace_,
    functional_step,
    validate_optimizer,
)
from tllm.program.stages import FlowStagePlan, StagedFlowExecutor, slice_flow_stages
from tllm.program.state import ProgramStateBank, StateSlot
from tllm.workgraph.frontend import (
    DynamicDim,
    ExternProgramDecl,
    FlowProgramDecl,
    ResultRef,
    SinkRef,
    StateScope,
)
from tllm.workgraph.lowering import WorkGraphError

_STATE_PREFIX = "_program_instance."


@runtime_checkable
class ProgramSink(Protocol):
    """A registered consumer of a flow's named sink output."""

    @property
    def name(self) -> str: ...

    def publish(self, value: torch.Tensor, meta: RowBatchMeta | None) -> None: ...


class _FlowAdapter(torch.nn.Module):
    """functional_call wrapper exposing one flow method as ``forward``."""

    def __init__(self, instance: torch.nn.Module, method_name: str) -> None:
        super().__init__()
        self.program_instance = instance
        self._method_name = method_name

    def forward(self, *args: torch.Tensor) -> object:
        method = getattr(self.program_instance, self._method_name)
        out = method(*args)
        # A flow with one declared output may return a bare tensor; the
        # training/oracle paths index outputs positionally, so normalize.
        decl = method.flow_program  # type: ignore[attr-defined]
        if len(decl.outputs) == 1 and isinstance(out, torch.Tensor):
            return (out,)
        return out


@dataclass(slots=True)
class _FlowRecord:
    """Owned per-step stash of one flow's inputs/outputs (stage buffers)."""

    step_id: int
    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]
    meta: RowBatchMeta | None


@dataclass(slots=True)
class _ExternBinding:
    node: ProgramNode
    decl: ExternProgramDecl
    source_flow: str
    source_ordinal: int
    source_output: str
    bank: str  # state name
    last_step: int = -1


@dataclass(slots=True)
class _CapturedStep:
    """One CUDA-graph-captured training step for a (flow, rows) bucket.

    The graph reads ``static_inputs`` and the slot's persistent buffers,
    applies one functional update in place, and leaves the flow outputs in
    ``static_outputs``. ``step_buf`` holds the 0-based step count and is
    incremented inside the graph; the host re-pins it from ``slot.step``
    before every replay.
    """

    graph: torch.cuda.CUDAGraph
    static_inputs: dict[str, torch.Tensor]
    static_outputs: dict[str, torch.Tensor]
    loss_buffers: tuple[torch.Tensor, ...]
    step_buf: torch.Tensor
    slot: StateSlot
    keepalive: tuple[object, ...]


class _StreamLane:
    """Side-stream execution lane for CUDA bundles.

    Flow execution and functional updates run on a dedicated stream so they
    overlap the main decode stream. Ordering is event-based: the lane waits
    on the producing stream before reading bundle tensors; readers
    (``retire_device_lease`` at lease end, ``run_flow`` at sampler time) wait
    on the lane's latest completion event. All entry points run on the engine
    thread, so no host locking is involved.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._event = torch.cuda.Event(blocking=False)
        self._context: torch.cuda.StreamContext | None = None

    def __enter__(self) -> "_StreamLane":
        self._stream.wait_stream(torch.cuda.current_stream(device=self._device))
        context = torch.cuda.stream(self._stream)
        context.__enter__()
        self._context = context
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._context is not None
        self._context.__exit__(exc_type, exc, tb)
        self._context = None
        self._event.record(self._stream)

    def wait_for_completion(self) -> None:
        """Order the current stream after the lane's latest recorded work."""
        torch.cuda.current_stream(device=self._device).wait_event(self._event)

    @property
    def completion_event(self) -> torch.cuda.Event:
        return self._event


def _is_identity_group(row_ids: tuple[int, ...], rows: int) -> bool:
    """True when one group spans the whole batch in order (no slicing needed)."""
    return (
        len(row_ids) == rows
        and rows > 0
        and row_ids[0] == 0
        and row_ids[-1] == rows - 1
    )


def _merge_row_outputs(
    row_outputs: dict[str, list[tuple[tuple[int, ...], torch.Tensor]]], rows: int
) -> dict[str, torch.Tensor]:
    """Scatter per-group outputs back into batch-row order."""
    outputs: dict[str, torch.Tensor] = {}
    for name, pieces in row_outputs.items():
        merged = torch.empty(
            (rows, *pieces[0][1].shape[1:]),
            dtype=pieces[0][1].dtype,
            device=pieces[0][1].device,
        ) if pieces[0][1].ndim > 0 else torch.empty(
            (len(pieces),), dtype=pieces[0][1].dtype, device=pieces[0][1].device
        )
        if pieces[0][1].ndim > 0:
            for row_ids, value in pieces:
                merged[torch.as_tensor(row_ids, device=merged.device)] = value
            outputs[name] = merged
        else:
            outputs[name] = torch.stack([value for _ids, value in pieces])
    return outputs


def _dtype_tolerance(dtype: torch.dtype) -> float:
    if dtype in (torch.float64,):
        return 1e-10
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2
    return 1e-5


def _optimizer_of(instance: torch.nn.Module) -> torch.optim.Optimizer:
    found = [
        value for value in vars(instance).values() if isinstance(value, torch.optim.Optimizer)
    ]
    if len(found) != 1:
        raise WorkGraphError(
            "PG160",
            f"Program training requires exactly one torch.optim.Optimizer attribute, "
            f"found {len(found)}",
        )
    return found[0]


def _trainable_names(
    instance: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[str, ...]:
    by_id = {id(param): name for name, param in instance.named_parameters()}
    names: list[str] = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = by_id.get(id(param))
            if name is None:
                raise WorkGraphError(
                    "PG130", "optimizer parameter is not an instance nn.Parameter"
                )
            names.append(name)
    if not names:
        raise WorkGraphError("PG160", "optimizer has no parameters")
    return tuple(names)


def _check_oracle(
    *,
    instance: torch.nn.Module,
    extern: _ExternBinding,
    flow_node: ProgramNode,
    flow_decl: FlowProgramDecl,
    scalars: SGDScalars | AdamScalars,
    trainable: tuple[str, ...],
    example_args: tuple[torch.Tensor, ...],
) -> None:
    """Run the extern's Python body on a cloned state; compare with the
    functional transition. Any divergence rejects the specialization."""
    output_names = tuple(name for name, _ref in flow_decl.outputs)
    try:
        loss_position = output_names.index(extern.source_output)
    except ValueError as exc:
        raise WorkGraphError(
            "PG100", f"extern {extern.node.name!r} references an undeclared output"
        ) from exc

    reference = copy.deepcopy(instance)
    # Compile may run under the engine's inference_mode; deepcopy preserves
    # inference tensors, which cannot be saved for backward. Rebuild the
    # reference's parameters as plain tensors before the oracle step and
    # rebind the (deepcopied) optimizer's param groups to them.
    with torch.inference_mode(False):
        replaced: dict[int, torch.nn.Parameter] = {}
        for module in reference.modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.is_inference():
                    new_param = torch.nn.Parameter(
                        param.detach().clone(), requires_grad=param.requires_grad
                    )
                    module._parameters[param_name] = new_param
                    replaced[id(param)] = new_param
        if replaced:
            for group in _optimizer_of(reference).param_groups:
                group["params"] = [
                    replaced.get(id(param), param) for param in group["params"]
                ]
        normal_args = tuple(
            arg.clone() if torch.is_tensor(arg) and arg.is_inference() else arg
            for arg in example_args
        )
        with torch.enable_grad():
            reference_out = getattr(reference, flow_node.name)(*normal_args)
            if len(output_names) == 1 and isinstance(reference_out, torch.Tensor):
                reference_out = (reference_out,)
            loss_ref = cast(tuple[torch.Tensor, ...], reference_out)[loss_position]
            if not torch.is_tensor(loss_ref) or loss_ref.numel() != 1:
                raise WorkGraphError(
                    "PG130", "extern training requires a scalar loss output"
                )
            getattr(reference, extern.node.name)(loss_ref)
    reference_optimizer = _optimizer_of(reference)
    reference_params = {
        name: param.detach().clone()
        for name, param in reference.named_parameters()
        if name in trainable
    }
    by_id = {id(param): name for name, param in reference.named_parameters()}
    # The "step" counter is excluded: the functional transition tracks it on
    # the state slot (slot.step), not as an optimizer-state tensor.
    reference_state: OptState = {}
    for param, param_state in reference_optimizer.state.items():
        name = by_id.get(id(param))
        if name is None:
            continue
        reference_state[name] = {
            key: value
            for key, value in param_state.items()
            if torch.is_tensor(value) and key != "step"
        }

    adapter = _FlowAdapter(instance, flow_node.name)
    with torch.inference_mode(False):
        params0 = {
            f"program_instance.{name}": (
                param.detach().clone()
                if not param.is_inference()
                else param.clone()
            )
            for name, param in instance.named_parameters()
            if name in trainable
        }

        def loss_fn(
            params: dict[str, torch.Tensor], *args: torch.Tensor
        ) -> torch.Tensor:
            out = torch.func.functional_call(adapter, params, args)
            return cast(tuple[torch.Tensor, ...], out)[loss_position]

        grads = torch.func.grad(loss_fn)(params0, *normal_args)
        new_params, new_state = functional_step(
            scalars,
            {name[len("program_instance."):]: value for name, value in params0.items()},
            {name[len("program_instance."):]: value for name, value in grads.items()},
            {},
            step=1,
        )

    for name in trainable:
        tolerance = _dtype_tolerance(new_params[name].dtype)
        if not torch.allclose(
            new_params[name], reference_params[name], rtol=tolerance, atol=tolerance
        ):
            raise WorkGraphError(
                "PG145",
                f"reference oracle mismatch on parameter {name!r}: functional "
                "transition diverges from the extern Python body",
            )
        for state_key, reference_value in reference_state.get(name, {}).items():
            functional_value = new_state.get(name, {}).get(state_key)
            if functional_value is None or not torch.allclose(
                functional_value, reference_value, rtol=tolerance, atol=tolerance
            ):
                raise WorkGraphError(
                    "PG145",
                    f"reference oracle mismatch on optimizer state {name!r}.{state_key}",
                )
    del reference


def compile_program(
    instance: torch.nn.Module,
    *,
    states: Mapping[str, StateSpec],
    examples: Mapping[str, tuple[torch.Tensor, ...]],
    sinks: Iterable[ProgramSink] = (),
    dynamic_dims: Mapping[str, tuple[DynamicDim, ...]] = {},
) -> ProgramConsumer:
    """Compile a declared Program instance into a consumer.

    ``examples`` maps each flow name to cold-specialization example inputs
    (real hidden sizes). Failure anywhere raises a PG-coded WorkGraphError;
    no partial consumer is created.
    """
    if not isinstance(instance, torch.nn.Module):
        raise TypeError("compile_program.instance must be an nn.Module")
    decl = declare_program(instance, states=states)
    sink_registry = {sink.name: sink for sink in sinks}
    if len(sink_registry) != len(tuple(sinks)):
        raise WorkGraphError("PG120", "duplicate sink registration")
    for value in decl.values:
        if isinstance(value.ref, SinkRef) and value.ref.name not in sink_registry:
            raise WorkGraphError(
                "PG100", f"sink {value.ref.name!r} has no registered implementation"
            )

    flow_nodes = {node.name: node for node in decl.nodes if node.kind == "flow"}
    extern_nodes = [node for node in decl.nodes if node.kind == "extern"]

    plans: dict[int, tuple[ProgramNode, FlowStagePlan, StagedFlowExecutor]] = {}
    for node in flow_nodes.values():
        flow_decl = cast(FlowProgramDecl, node.decl)
        example_args = examples.get(node.name)
        if example_args is None:
            raise WorkGraphError(
                "PG130", f"flow {node.name!r} has no cold-specialization examples"
            )
        plan, executor = slice_flow_stages(
            flow=node.name,
            flow_ordinal=node.ordinal,
            decl=flow_decl,
            function=node.function,
            example_args=tuple(cast(tuple[object, ...], example_args)),
            dynamic_dims=dynamic_dims.get(node.name, ()),
        )
        plans[node.ordinal] = (node, plan, executor)

    scalars: SGDScalars | AdamScalars | None = None
    trainable: tuple[str, ...] = ()
    externs: list[_ExternBinding] = []
    if extern_nodes:
        optimizer = _optimizer_of(instance)
        scalars = validate_optimizer(optimizer)
        trainable = _trainable_names(instance, optimizer)

    banks: dict[str, ProgramStateBank] = {}
    actual_cls: type | None = None
    if scalars is not None:
        actual_cls = (
            torch.optim.SGD
            if isinstance(scalars, SGDScalars)
            else (torch.optim.AdamW if cast(AdamScalars, scalars).decoupled else torch.optim.Adam)
        )
    for state in decl.states:
        initial = {
            name: param.detach().clone()
            for name, param in instance.named_parameters()
            if name in trainable
        }
        banks[state.name] = ProgramStateBank(
            state.name, state.spec.scope, initial, key_spec=state.spec.key
        )
        expected = state.spec.optimizer
        if expected is not None and actual_cls is not None:
            expected_cls = expected if isinstance(expected, type) else type(expected)
            if expected_cls is not actual_cls:
                raise WorkGraphError(
                    "PG160",
                    f"state {state.name!r} declares optimizer "
                    f"{expected_cls.__name__} but the instance uses {actual_cls.__name__}",
                )

    for node in extern_nodes:
        extern_decl = cast(ExternProgramDecl, node.decl)
        result_refs = [
            (param, value)
            for param, value in extern_decl.inputs
            if isinstance(value, ResultRef)
        ]
        if len(result_refs) != 1 or len(extern_decl.inputs) != 1:
            raise WorkGraphError(
                "PG130",
                f"extern {node.name!r}: v1 requires exactly one result() input",
            )
        _param, source = result_refs[0]
        if extern_decl.state is None:
            raise WorkGraphError("PG100", f"extern {node.name!r} has no state")
        binding = _ExternBinding(
            node=node,
            decl=extern_decl,
            source_flow=source.flow,
            source_ordinal=flow_nodes[source.flow].ordinal,
            source_output=source.output,
            bank=extern_decl.state,
        )
        source_node = flow_nodes[source.flow]
        _check_oracle(
            instance=instance,
            extern=binding,
            flow_node=source_node,
            flow_decl=cast(FlowProgramDecl, source_node.decl),
            scalars=cast(SGDScalars | AdamScalars, scalars),
            trainable=trainable,
            example_args=tuple(
                cast(torch.Tensor, arg) for arg in examples[source.flow]
            ),
        )
        externs.append(binding)

    return ProgramConsumer(
        instance=instance,
        decl=decl,
        flows=plans,
        externs=tuple(externs),
        banks=banks,
        sinks=sink_registry,
        scalars=scalars,
        trainable=trainable,
    )


class ProgramConsumer(BaseConsumer[Mapping[str, object]]):
    """Generated consumer: stage execution + functional state transitions."""

    def __init__(
        self,
        *,
        instance: torch.nn.Module,
        decl: ProgramDecl,
        flows: Mapping[int, tuple[ProgramNode, FlowStagePlan, StagedFlowExecutor]],
        externs: tuple[_ExternBinding, ...],
        banks: Mapping[str, ProgramStateBank],
        sinks: Mapping[str, ProgramSink],
        scalars: SGDScalars | AdamScalars | None,
        trainable: tuple[str, ...],
    ) -> None:
        self.config: Mapping[str, object] = {}
        self._instance = instance
        self._decl = decl
        self._flows = dict(flows)
        self._externs = externs
        self._banks = dict(banks)
        self._sinks = dict(sinks)
        self._scalars = scalars
        self._trainable = trainable
        self._records: dict[str, _FlowRecord] = {}
        self.update_count = 0
        self.loss_count = 0
        self._loss_sum: torch.Tensor | None = None
        self.read_versions: dict[tuple[str, int], int] = {}
        self._slot_loss_sums: dict[tuple[str, int], torch.Tensor] = {}
        self._slot_loss_counts: dict[tuple[str, int], int] = {}
        self._lane: _StreamLane | None = None
        self._adapters: dict[str, _FlowAdapter] = {}
        self._reader_stream: torch.cuda.Stream | None = None
        self._reader_event: torch.cuda.Event | None = None
        self._captures: dict[tuple[str, int], _CapturedStep] = {}
        self.graph_capture_count = 0
        self.graph_replay_count = 0

    @property
    def loss_sum(self) -> float:
        # Syncs the device accumulator on read (control plane only).
        if self._loss_sum is None:
            return 0.0
        return float(self._loss_sum)

    @property
    def consumer_id(self) -> str:
        return f"program:{type(self._instance).__name__}"

    def bank(self, name: str) -> ProgramStateBank:
        return self._banks[name]

    def flow_ordinal(self, flow_name: str) -> int:
        """Resolve a declared flow name to its class-definition ordinal."""
        for ordinal, (node, _plan, _executor) in self._flows.items():
            if node.name == flow_name:
                return ordinal
        raise WorkGraphError(
            "PG100", f"unknown flow {flow_name!r} for {self.consumer_id}"
        )

    def run_flow(
        self,
        flow_name: str,
        *inputs: torch.Tensor,
        prompt_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Invoke one flow on demand against the current bank state.

        Read-only: no sink publication, no record stash, no extern scheduling.
        REQUEST banks require ``prompt_idx``.
        """
        entry: tuple[ProgramNode, StagedFlowExecutor] | None = None
        for node, _plan, executor in self._flows.values():
            if node.name == flow_name:
                entry = (node, executor)
                break
        if entry is None:
            raise WorkGraphError(
                "PG100", f"unknown flow {flow_name!r} for {self.consumer_id}"
            )
        node, executor = entry
        flow_decl = cast(FlowProgramDecl, node.decl)
        bank = (
            self._banks.get(flow_decl.state) if flow_decl.state is not None else None
        )

        def _invoke() -> dict[str, torch.Tensor]:
            with torch.no_grad():
                if bank is None:
                    return executor.run(*inputs)
                slot = bank.slot_for(prompt_idx)
                return executor.run(*inputs, state_override=self._slot_override(slot))

        lane = self._lane
        device = inputs[0].device if inputs else None
        if lane is None or device is None or device.type != "cuda":
            return _invoke()
        # Read-only invocations go to a dedicated reader lane: the only edges
        # are input readiness (caller stream) and state freshness (training
        # lane's completion event). The caller's stream waits only on the
        # produced outputs, never inline-executes the flow, and never blocks
        # behind unrelated decode work.
        if self._reader_stream is None:
            self._reader_stream = torch.cuda.Stream(device=device)
            self._reader_event = torch.cuda.Event(blocking=False)
        reader = self._reader_stream
        event = self._reader_event
        assert event is not None
        current = torch.cuda.current_stream(device=device)
        reader.wait_stream(current)
        reader.wait_event(lane.completion_event)
        with torch.cuda.stream(reader):
            out = _invoke()
        event.record(reader)
        current.wait_event(event)
        for value in out.values():
            # Allocated on the reader lane, consumed on the caller stream.
            value.record_stream(current)
        return out

    def read_and_reset_slot_losses(self) -> dict[tuple[str, int], tuple[float, int]]:
        """Per-slot (loss_sum, loss_count) since the last read; resets counters."""
        out = {
            key: (float(self._slot_loss_sums[key]), self._slot_loss_counts[key])
            for key in self._slot_loss_sums
        }
        self._slot_loss_sums.clear()
        self._slot_loss_counts.clear()
        return out

    def _groups(
        self, bank: ProgramStateBank, meta: RowBatchMeta | None, rows: int
    ) -> tuple[tuple[Hashable | None, tuple[int, ...]], ...]:
        if bank.scope is StateScope.REPLICA:
            return ((None, tuple(range(rows))),)
        if meta is None:
            raise WorkGraphError(
                "PG180", "REQUEST state requires request_meta in the bundle"
            )
        return tuple(
            (key, row_ids)
            for key, row_ids in bank.groups_for_metadata(
                request_ids=meta.request_ids[:rows],
                prompt_idxs=meta.prompt_idxs[:rows],
                sample_idxs=meta.sample_idxs[:rows],
            )
        )

    def _slot_override(self, slot: StateSlot) -> dict[str, torch.Tensor]:
        return {f"{_STATE_PREFIX}{name}": value for name, value in slot.params.items()}

    def consume_flow(self, flow_ordinal: int, bundle: PortBundle, ctx: object) -> None:
        del ctx
        entry = self._flows.get(flow_ordinal)
        if entry is None:
            raise WorkGraphError(
                "PG100", f"unknown flow ordinal {flow_ordinal} for {self.consumer_id}"
            )
        node, plan, executor = entry
        flow_decl = cast(FlowProgramDecl, node.decl)
        lease = bundle.entries.get("device_lease")
        if not isinstance(lease, DeviceTensorLease):
            raise WorkGraphError(
                "PG100", f"flow {node.name!r} bundle has no device_lease entry"
            )
        meta_payload = bundle.entries.get("request_meta")
        meta = meta_payload if isinstance(meta_payload, RowBatchMeta) else None
        rows = int(lease.active_rows)
        try:
            inputs = {
                param: lease.entries[param][:rows]
                for param, _source in flow_decl.inputs
            }
        except KeyError as exc:
            raise WorkGraphError(
                "PG100", f"flow {node.name!r} input {exc} missing from device_lease"
            ) from exc

        device = next(iter(inputs.values())).device
        step_id = int(bundle.key.engine_step_id)
        if device.type == "cuda":
            if self._lane is None:
                self._lane = _StreamLane(device)
            with self._lane:
                self._consume_inner(node, executor, flow_decl, inputs, meta, rows, step_id)
        else:
            self._consume_inner(node, executor, flow_decl, inputs, meta, rows, step_id)

    def retire_device_lease(self, ctx: object) -> None:
        del ctx
        # Lease buffers may be reused by the producer after retirement; order
        # the current stream after the lane's pending reads first.
        if self._lane is not None:
            self._lane.wait_for_completion()

    def _consume_inner(
        self,
        node: ProgramNode,
        executor: StagedFlowExecutor,
        flow_decl: FlowProgramDecl,
        inputs: dict[str, torch.Tensor],
        meta: RowBatchMeta | None,
        rows: int,
        step_id: int,
    ) -> None:
        outputs: dict[str, torch.Tensor] = {}
        bank = self._banks.get(flow_decl.state) if flow_decl.state is not None else None
        # Grad mode is derived from the declared DAG: when an extern whose
        # input edge reads this flow's result is due this step, the flow runs
        # once with grad enabled and the edge carries both the detached
        # values (sink/record) and the autograd graph (functional update).
        # Flows without a due training extern run under no_grad as before.
        due = self._due_externs(node.name, step_id) if bank is not None else ()
        if bank is None:
            with torch.no_grad():
                outputs = executor.run(*tuple(inputs.values()))
        elif due:
            outputs = self._train_step_outputs(
                node, flow_decl, bank, due, inputs, meta, rows, step_id
            )
        else:
            groups = self._groups(bank, meta, rows)
            row_outputs: dict[str, list[tuple[tuple[int, ...], torch.Tensor]]] = {}
            with torch.no_grad():
                for prompt_idx, row_ids in groups:
                    slot = bank.slot_for(prompt_idx)
                    self.read_versions[(bank.name, slot.slot_id)] = slot.version
                    if _is_identity_group(row_ids, rows):
                        group_inputs = tuple(inputs.values())
                    else:
                        group_inputs = tuple(
                            value.index_select(
                                0, torch.as_tensor(row_ids, device=value.device)
                            )
                            for value in inputs.values()
                        )
                    group_out = executor.run(
                        *group_inputs, state_override=self._slot_override(slot)
                    )
                    if _is_identity_group(row_ids, rows):
                        # Single group spanning the whole batch: group outputs
                        # already are the merged outputs.
                        outputs = group_out
                        continue
                    for name, value in group_out.items():
                        row_outputs.setdefault(name, []).append((row_ids, value))
            outputs = outputs or _merge_row_outputs(row_outputs, rows)

        for name, value in outputs.items():
            ref = dict(flow_decl.outputs)[name]
            if isinstance(ref, SinkRef):
                self._sinks[ref.name].publish(value, meta)

        self._records[node.name] = _FlowRecord(
            step_id=step_id,
            inputs=inputs,
            outputs=outputs,
            meta=meta,
        )
        self._run_due_externs(step_id)

    def _due_externs(self, flow_name: str, step_id: int) -> tuple[_ExternBinding, ...]:
        """Externs whose declared dependencies make them due at this flow's
        execution: source edge reads this flow, not yet run this step, and
        every ``after`` flow already executed this step (self-after counts)."""
        due: list[_ExternBinding] = []
        for extern in self._externs:
            if extern.source_flow != flow_name or extern.last_step == step_id:
                continue
            ready = True
            for after in extern.decl.after:
                if after == flow_name:
                    continue
                record = self._records.get(after)
                if record is None or record.step_id != step_id:
                    ready = False
                    break
            if ready:
                due.append(extern)
        return tuple(due)

    def _train_step_outputs(
        self,
        node: ProgramNode,
        flow_decl: FlowProgramDecl,
        bank: ProgramStateBank,
        due: tuple[_ExternBinding, ...],
        inputs: dict[str, torch.Tensor],
        meta: RowBatchMeta | None,
        rows: int,
        step_id: int,
    ) -> dict[str, torch.Tensor]:
        """Single-pass grad execution: one forward per group serves both the
        flow's published outputs and the due externs' functional updates."""
        if self._scalars is None:
            raise WorkGraphError("PG160", "extern without a validated optimizer")
        adapter = self._adapters.get(node.name)
        if adapter is None:
            adapter = _FlowAdapter(self._instance, node.name)
            self._adapters[node.name] = adapter
        output_names = tuple(name for name, _ in flow_decl.outputs)
        loss_positions = tuple(
            output_names.index(extern.source_output) for extern in due
        )

        groups = self._groups(bank, meta, rows)
        identity = len(groups) == 1 and _is_identity_group(groups[0][1], rows)
        device = next(iter(inputs.values())).device
        if self._capture_eligible(bank, due, identity, device):
            return self._captured_train_step(
                node, flow_decl, bank, due, inputs, rows, step_id
            )
        outputs: dict[str, torch.Tensor] = {}
        row_outputs: dict[str, list[tuple[tuple[int, ...], torch.Tensor]]] = {}
        for prompt_idx, row_ids in groups:
            slot = bank.slot_for(prompt_idx)
            self.read_versions[(bank.name, slot.slot_id)] = slot.version
            if identity:
                group_inputs = tuple(inputs.values())
            else:
                group_inputs = tuple(
                    value.index_select(0, torch.as_tensor(row_ids, device=value.device))
                    for value in inputs.values()
                )
            # The whole region lifts inference_mode explicitly: the engine
            # executes consumers under inference_mode, where enable_grad alone
            # records no autograd graph and inference tensors cannot be saved
            # for backward.
            with torch.inference_mode(False):
                with torch.enable_grad():
                    grad_params = {
                        f"program_instance.{name}": slot.params[name]
                        .detach()
                        .requires_grad_(True)
                        for name in self._trainable
                    }
                    grad_inputs = tuple(
                        value.clone() if value.is_inference() else value
                        for value in group_inputs
                    )
                    out = cast(
                        tuple[torch.Tensor, ...],
                        torch.func.functional_call(adapter, grad_params, grad_inputs),
                    )
                    group_values = dict(zip(output_names, out, strict=True))
                    for index, extern in enumerate(due):
                        group_loss = out[loss_positions[index]]
                        grad_list = torch.autograd.grad(
                            group_loss,
                            tuple(grad_params.values()),
                            retain_graph=index < len(due) - 1,
                        )
                        grads = dict(zip(self._trainable, grad_list, strict=True))
                        ext_bank = self._banks[extern.bank]
                        ext_slot = ext_bank.slot_for(prompt_idx)
                        with torch.no_grad():
                            new_params, new_state = functional_step(
                                self._scalars,
                                {
                                    name: grad_params[
                                        f"program_instance.{name}"
                                    ].detach()
                                    for name in self._trainable
                                },
                                grads,
                                ext_slot.opt_state,
                                step=ext_slot.step + 1,
                            )
                        # Published inside the lifted region so the new slot
                        # tensors are normal (autograd-able) tensors.
                        ext_bank.publish_update(ext_slot, new_params, new_state)
                        self.update_count += 1
                        slot_key = (ext_bank.name, ext_slot.slot_id)
                        slot_loss = group_loss.detach().float()
                        prior = self._slot_loss_sums.get(slot_key)
                        self._slot_loss_sums[slot_key] = (
                            slot_loss if prior is None else prior + slot_loss
                        )
                        self._slot_loss_counts[slot_key] = (
                            self._slot_loss_counts.get(slot_key, 0) + 1
                        )
                        extern.last_step = step_id
                        step_loss = slot_loss.sum()
                        self._loss_sum = (
                            step_loss
                            if self._loss_sum is None
                            else self._loss_sum + step_loss
                        )
            detached = {name: value.detach() for name, value in group_values.items()}
            if identity:
                outputs = detached
            else:
                for name, value in detached.items():
                    row_outputs.setdefault(name, []).append((row_ids, value))
        # One training event per due extern per step (legacy stats semantics);
        # per-group detail lives in the slot loss counters.
        self.loss_count += len(due)
        return outputs if identity else _merge_row_outputs(row_outputs, rows)

    def _capture_eligible(
        self,
        bank: ProgramStateBank,
        due: tuple[_ExternBinding, ...],
        identity: bool,
        device: torch.device,
    ) -> bool:
        """A step is capturable when it is a fixed DAG over static buffers:
        single-group REPLICA bank, exactly one due extern on that bank, AdamW
        scalars, CUDA inputs. Anything else runs the eager functional path —
        an internal execution-mode decision, not a correctness fallback."""
        return (
            identity
            and device.type == "cuda"
            and bank.scope is StateScope.REPLICA
            and len(due) == 1
            and isinstance(self._scalars, AdamScalars)
            and self._banks[due[0].bank] is bank
        )

    def _captured_train_step(
        self,
        node: ProgramNode,
        flow_decl: FlowProgramDecl,
        bank: ProgramStateBank,
        due: tuple[_ExternBinding, ...],
        inputs: dict[str, torch.Tensor],
        rows: int,
        step_id: int,
    ) -> dict[str, torch.Tensor]:
        key = (node.name, rows)
        slot = bank.slot_for(None)
        self.read_versions[(bank.name, slot.slot_id)] = slot.version
        captured = self._captures.get(key)
        if captured is None:
            captured = self._capture_train_step(node, flow_decl, due, inputs, slot)
            self._captures[key] = captured
            self.graph_capture_count += 1
            # The verification replay inside capture already applied this
            # step's update; bookkeeping happens in the shared tail below.
        else:
            for name, value in inputs.items():
                captured.static_inputs[name].copy_(value)
            captured.step_buf.fill_(float(slot.step))
            captured.graph.replay()
            slot.step += 1
            slot.version += 1
            self.graph_replay_count += 1
        for extern, loss_buf in zip(due, captured.loss_buffers, strict=True):
            extern.last_step = step_id
            self.update_count += 1
            slot_key = (bank.name, slot.slot_id)
            slot_loss = loss_buf.detach().float()
            prior = self._slot_loss_sums.get(slot_key)
            self._slot_loss_sums[slot_key] = (
                slot_loss if prior is None else prior + slot_loss
            )
            self._slot_loss_counts[slot_key] = (
                self._slot_loss_counts.get(slot_key, 0) + 1
            )
            self._loss_sum = (
                slot_loss if self._loss_sum is None else self._loss_sum + slot_loss
            )
        self.loss_count += len(due)
        return {
            name: buf.detach().clone()
            for name, buf in captured.static_outputs.items()
        }

    def _capture_train_step(
        self,
        node: ProgramNode,
        flow_decl: FlowProgramDecl,
        due: tuple[_ExternBinding, ...],
        inputs: dict[str, torch.Tensor],
        slot: StateSlot,
    ) -> _CapturedStep:
        """Capture one training step as a CUDA graph, then verify the first
        replay against the eager functional transition (PG146 on mismatch)."""
        scalars = self._scalars
        if not isinstance(scalars, AdamScalars):
            raise WorkGraphError("PG160", "capture requires validated Adam scalars")
        adapter = self._adapters.get(node.name)
        if adapter is None:
            adapter = _FlowAdapter(self._instance, node.name)
            self._adapters[node.name] = adapter
        output_names = tuple(name for name, _ in flow_decl.outputs)
        loss_position = output_names.index(due[0].source_output)
        device = next(iter(inputs.values())).device

        with torch.inference_mode(False):
            static_inputs = {
                name: torch.empty_like(value) for name, value in inputs.items()
            }
            for name, value in inputs.items():
                static_inputs[name].copy_(value)
            # The slot's parameter/state tensors become the graph's fixed
            # write targets; they stay persistent buffers from now on.
            for name in self._trainable:
                param = slot.params[name]
                if not param.requires_grad:
                    param.requires_grad_(True)
                buffers = slot.opt_state.setdefault(name, {})
                if "exp_avg" not in buffers:
                    buffers["exp_avg"] = torch.zeros_like(param)
                if "exp_avg_sq" not in buffers:
                    buffers["exp_avg_sq"] = torch.zeros_like(param)
            beta1 = torch.full((), scalars.betas[0], dtype=torch.float32, device=device)
            beta2 = torch.full((), scalars.betas[1], dtype=torch.float32, device=device)
            step_buf = torch.zeros((), dtype=torch.float32, device=device)

            def step_fn(
                params: dict[str, torch.Tensor],
                state: OptState,
                input_values: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, ...]:
                with torch.enable_grad():
                    fn_params = {
                        f"program_instance.{name}": params[name]
                        for name in self._trainable
                    }
                    out = cast(
                        tuple[torch.Tensor, ...],
                        torch.func.functional_call(adapter, fn_params, input_values),
                    )
                    loss = out[loss_position]
                    grad_list = torch.autograd.grad(loss, tuple(fn_params.values()))
                grads = dict(zip(self._trainable, grad_list, strict=True))
                with torch.no_grad():
                    step_buf.add_(1.0)
                    adam_step_inplace_(
                        params,
                        grads,
                        state,
                        scalars,
                        step=step_buf,
                        beta1=beta1,
                        beta2=beta2,
                    )
                return out

            # Warmup on throwaway state (initializes backend workspaces off
            # the capture path), then capture against the live slot buffers.
            warm_params = {
                name: slot.params[name].detach().clone().requires_grad_(True)
                for name in self._trainable
            }
            warm_state: OptState = {
                name: {key: value.clone() for key, value in slot.opt_state[name].items()}
                for name in self._trainable
            }
            step_buf.fill_(float(slot.step))
            step_fn(warm_params, warm_state, tuple(static_inputs.values()))
            step_buf.fill_(float(slot.step))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = step_fn(slot.params, slot.opt_state, tuple(static_inputs.values()))
            static_outputs = {
                name: out[index].detach() for index, name in enumerate(output_names)
            }

            # Verify the first replay against the eager functional transition
            # on a snapshot of the pre-step state. The replay doubles as this
            # step's real update.
            snap_params = {
                name: slot.params[name].detach().clone() for name in self._trainable
            }
            snap_state: OptState = {
                name: {
                    key: value.detach().clone()
                    for key, value in slot.opt_state[name].items()
                }
                for name in self._trainable
            }
            with torch.enable_grad():
                snap_fn_params = {
                    f"program_instance.{name}": snap_params[name]
                    .detach()
                    .requires_grad_(True)
                    for name in self._trainable
                }
                snap_out = cast(
                    tuple[torch.Tensor, ...],
                    torch.func.functional_call(
                        adapter, snap_fn_params, tuple(static_inputs.values())
                    ),
                )
                snap_loss = snap_out[loss_position]
                snap_grads = torch.autograd.grad(
                    snap_loss, tuple(snap_fn_params.values())
                )
            exp_params, exp_state = functional_step(
                scalars,
                snap_params,
                dict(zip(self._trainable, snap_grads, strict=True)),
                snap_state,
                step=slot.step + 1,
            )
            step_buf.fill_(float(slot.step))
            graph.replay()
            tolerance = _dtype_tolerance(next(iter(slot.params.values())).dtype)
            for name in self._trainable:
                if not torch.allclose(
                    slot.params[name], exp_params[name], rtol=tolerance, atol=tolerance
                ):
                    raise WorkGraphError(
                        "PG146",
                        f"captured train step diverges from the eager functional "
                        f"transition on parameter {name!r}",
                    )
                for state_key, expected in exp_state.get(name, {}).items():
                    if not torch.allclose(
                        slot.opt_state[name][state_key],
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                    ):
                        raise WorkGraphError(
                            "PG146",
                            f"captured train step diverges on optimizer state "
                            f"{name!r}.{state_key}",
                        )
            slot.step += 1
            slot.version += 1
        return _CapturedStep(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
            loss_buffers=(static_outputs[due[0].source_output],),
            step_buf=step_buf,
            slot=slot,
            # Every tensor the captured kernels read must stay alive for the
            # graph's lifetime; a dropped reference frees the block back to
            # the allocator and later replays read reused memory.
            keepalive=(out, snap_out, beta1, beta2),
        )

    def _run_due_externs(self, step_id: int) -> None:
        """Fallback for externs whose ``after`` edges were satisfied only by a
        later flow this step: re-run the source forward under grad (the flow
        itself already executed in no_grad mode before they became due)."""
        for extern in self._externs:
            if extern.last_step == step_id:
                continue
            record = self._records.get(extern.source_flow)
            if record is None or record.step_id != step_id:
                continue
            if any(
                self._records.get(after) is None
                or cast(_FlowRecord, self._records.get(after)).step_id != step_id
                for after in extern.decl.after
            ):
                continue
            loss = record.outputs[extern.source_output]
            self._functional_update(extern, record, loss)
            extern.last_step = step_id

    def _functional_update(
        self, extern: _ExternBinding, record: _FlowRecord, loss: torch.Tensor
    ) -> None:
        if self._scalars is None:
            raise WorkGraphError("PG160", "extern without a validated optimizer")
        bank = self._banks[extern.bank]
        source_node, _source_plan, _executor = self._flows[extern.source_ordinal]
        rows = next(iter(record.inputs.values())).shape[0]
        groups = self._groups(bank, record.meta, int(rows))
        adapter = self._adapters.get(source_node.name)
        if adapter is None:
            adapter = _FlowAdapter(self._instance, source_node.name)
            self._adapters[source_node.name] = adapter
        flow_decl = cast(FlowProgramDecl, source_node.decl)
        loss_position = tuple(name for name, _ in flow_decl.outputs).index(
            extern.source_output
        )

        loss_values = record.outputs[extern.source_output].detach().float().view(-1)
        for group_index, (prompt_idx, row_ids) in enumerate(groups):
            slot = bank.slot_for(prompt_idx)
            if _is_identity_group(row_ids, int(rows)):
                group_inputs = tuple(record.inputs.values())
            else:
                group_inputs = tuple(
                    value.index_select(0, torch.as_tensor(row_ids, device=value.device))
                    for value in record.inputs.values()
                )

            # Single forward pass: run the flow on grad-enabled views of the
            # slot params, then reverse-mode straight to those views.
            # Equivalent to torch.func.grad over a re-invoked loss_fn but
            # skips the grad-transform wrapper. The whole region lifts
            # inference_mode explicitly: the engine executes consumers under
            # inference_mode, where enable_grad alone records no autograd
            # graph and inference tensors cannot be saved for backward.
            with torch.inference_mode(False):
                with torch.enable_grad():
                    grad_params = {
                        f"program_instance.{name}": slot.params[name]
                        .detach()
                        .requires_grad_(True)
                        for name in self._trainable
                    }
                    grad_inputs = tuple(
                        value.clone() if value.is_inference() else value
                        for value in group_inputs
                    )
                    out = torch.func.functional_call(adapter, grad_params, grad_inputs)
                    group_loss = cast(tuple[torch.Tensor, ...], out)[loss_position]
                    grad_list = torch.autograd.grad(
                        group_loss, tuple(grad_params.values())
                    )
                grads = dict(zip(self._trainable, grad_list, strict=True))
                with torch.no_grad():
                    new_params, new_state = functional_step(
                        self._scalars,
                        {
                            name: grad_params[f"program_instance.{name}"].detach()
                            for name in self._trainable
                        },
                        grads,
                        slot.opt_state,
                        step=slot.step + 1,
                    )
            bank.publish_update(slot, new_params, new_state)
            self.update_count += 1
            slot_key = (bank.name, slot.slot_id)
            slot_loss = loss_values[group_index]
            prior = self._slot_loss_sums.get(slot_key)
            self._slot_loss_sums[slot_key] = (
                slot_loss if prior is None else prior + slot_loss
            )
            self._slot_loss_counts[slot_key] = (
                self._slot_loss_counts.get(slot_key, 0) + 1
            )
        self.loss_count += 1
        step_loss = loss.detach().float().sum()
        self._loss_sum = step_loss if self._loss_sum is None else self._loss_sum + step_loss


__all__ = ["ProgramConsumer", "ProgramSink", "compile_program"]
