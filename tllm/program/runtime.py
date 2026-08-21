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
from typing import Protocol, cast, runtime_checkable

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
    functional_step,
    validate_optimizer,
)
from tllm.program.stages import FlowStagePlan, StagedFlowExecutor, slice_flow_stages
from tllm.program.state import ProgramStateBank, StateSlot, group_rows_by_prompt
from tllm.workgraph.frontend import (
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
        return method(*args)


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
    with torch.enable_grad():
        reference_out = getattr(reference, flow_node.name)(*example_args)
        loss_ref = reference_out[loss_position]
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
    reference_state: OptState = {}
    for param, param_state in reference_optimizer.state.items():
        name = by_id.get(id(param))
        if name is None:
            continue
        reference_state[name] = {
            key: value
            for key, value in param_state.items()
            if torch.is_tensor(value)
        }

    adapter = _FlowAdapter(instance, flow_node.name)
    params0 = {
        f"program_instance.{name}": param.detach().clone()
        for name, param in instance.named_parameters()
        if name in trainable
    }

    def loss_fn(params: dict[str, torch.Tensor], *args: torch.Tensor) -> torch.Tensor:
        out = torch.func.functional_call(adapter, params, args)
        return cast(tuple[torch.Tensor, ...], out)[loss_position]

    grads = torch.func.grad(loss_fn)(params0, *example_args)
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
        banks[state.name] = ProgramStateBank(state.name, state.spec.scope, initial)
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
        self.loss_sum = 0.0
        self.read_versions: dict[tuple[str, int], int] = {}

    @property
    def consumer_id(self) -> str:
        return f"program:{type(self._instance).__name__}"

    def bank(self, name: str) -> ProgramStateBank:
        return self._banks[name]

    def _groups(
        self, bank: ProgramStateBank, meta: RowBatchMeta | None, rows: int
    ) -> tuple[tuple[int | None, tuple[int, ...]], ...]:
        if bank.scope is StateScope.REPLICA:
            return ((None, tuple(range(rows))),)
        if meta is None:
            raise WorkGraphError(
                "PG180", "REQUEST state requires request_meta in the bundle"
            )
        return tuple(
            (prompt_idx, row_ids)
            for prompt_idx, row_ids in group_rows_by_prompt(meta.prompt_idxs[:rows])
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

        outputs: dict[str, torch.Tensor] = {}
        bank = self._banks.get(flow_decl.state) if flow_decl.state is not None else None
        if bank is None:
            with torch.no_grad():
                outputs = executor.run(*tuple(inputs.values()))
        else:
            groups = self._groups(bank, meta, rows)
            row_outputs: dict[str, list[tuple[tuple[int, ...], torch.Tensor]]] = {}
            with torch.no_grad():
                for prompt_idx, row_ids in groups:
                    slot = bank.slot_for(prompt_idx)
                    self.read_versions[(bank.name, slot.slot_id)] = slot.version
                    group_inputs = tuple(
                        value.index_select(
                            0, torch.as_tensor(row_ids, device=value.device)
                        )
                        for value in inputs.values()
                    )
                    group_out = executor.run(
                        *group_inputs, state_override=self._slot_override(slot)
                    )
                    for name, value in group_out.items():
                        row_outputs.setdefault(name, []).append((row_ids, value))
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

        for name, value in outputs.items():
            ref = dict(flow_decl.outputs)[name]
            if isinstance(ref, SinkRef):
                self._sinks[ref.name].publish(value, meta)

        self._records[node.name] = _FlowRecord(
            step_id=int(bundle.key.engine_step_id),
            inputs=inputs,
            outputs=outputs,
            meta=meta,
        )
        self._run_due_externs(int(bundle.key.engine_step_id))

    def _run_due_externs(self, step_id: int) -> None:
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
        adapter = _FlowAdapter(self._instance, source_node.name)
        flow_decl = cast(FlowProgramDecl, source_node.decl)
        loss_position = tuple(name for name, _ in flow_decl.outputs).index(
            extern.source_output
        )

        for prompt_idx, row_ids in groups:
            slot = bank.slot_for(prompt_idx)
            params = {
                f"program_instance.{name}": slot.params[name] for name in self._trainable
            }
            group_inputs = tuple(
                value.index_select(0, torch.as_tensor(row_ids, device=value.device))
                for value in record.inputs.values()
            )

            def loss_fn(
                fn_params: dict[str, torch.Tensor], *args: torch.Tensor
            ) -> torch.Tensor:
                out = torch.func.functional_call(adapter, fn_params, args)
                return cast(tuple[torch.Tensor, ...], out)[loss_position]

            with torch.enable_grad():
                grads = torch.func.grad(loss_fn)(params, *group_inputs)
            new_params, new_state = functional_step(
                self._scalars,
                {name: params[f"program_instance.{name}"] for name in self._trainable},
                {
                    name: grads[f"program_instance.{name}"]
                    for name in self._trainable
                },
                slot.opt_state,
                step=slot.step + 1,
            )
            bank.publish_update(slot, new_params, new_state)
            self.update_count += 1
        self.loss_count += 1
        self.loss_sum += float(loss.detach().float().sum())


__all__ = ["ProgramConsumer", "ProgramSink", "compile_program"]
