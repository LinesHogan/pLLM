"""Qwen2 parameter primitives borrowing the original signed arena fields and pins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import Parameter
from torch.nn import functional as F
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

from pllm.authoring import ParameterTargetShape
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterModuleKind,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterRole,
    ParameterTarget,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import StateScope
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_stream import upload
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.state_execution import DataT, StateTaskContext

_LinearOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]
_LinearForward = Callable[[torch.Tensor], _LinearOutput]
_MlpForward = Callable[[torch.Tensor], torch.Tensor]


class _Linear(Protocol):
    weight: Parameter
    bias: Parameter | None
    tp_size: int
    return_bias: bool
    forward: _LinearForward


class _Mlp(Protocol):
    forward: _MlpForward


def _locate(model: Qwen2ForCausalLM, target: ParameterTarget) -> tuple[_Linear, int, int, _Mlp | None]:
    layer = model.model.layers[target.layer]
    if target.module is ParameterModuleKind.MLP and target.role is ParameterRole.DOWN_PROJECTION_WEIGHT:
        module = layer.mlp.down_proj
        if module.bias is not None:
            raise ValueError("dense operand capture requires a bias-free down projection")
        return module, 0, int(module.weight.shape[0]), layer.mlp
    if target.module is ParameterModuleKind.ATTENTION:
        attention = layer.self_attn
        if target.role is ParameterRole.QUERY_PROJECTION_WEIGHT:
            return attention.qkv_proj, 0, attention.q_size, None
        if target.role is ParameterRole.VALUE_PROJECTION_WEIGHT:
            return attention.qkv_proj, attention.q_size + attention.kv_size, attention.kv_size, None
    raise ValueError("Qwen2 overlay requires an MLP down or attention q/v projection")


def inspect_parameter_targets(
    model: Qwen2ForCausalLM, targets: tuple[ParameterTarget, ...], dtype: torch.dtype
) -> tuple[ParameterTargetShape, ...]:
    specs: list[ParameterTargetShape] = []
    if len(set(targets)) != len(targets):
        raise ValueError("parameter targets must be unique")
    for target in targets:
        module, offset, size, _ = _locate(model, target)
        if (
            module.tp_size != 1
            or not module.return_bias
            or module.weight.dtype is not dtype
            or offset + size > module.weight.shape[0]
        ):
            raise ValueError("Qwen2 overlay requires an exact unsharded projection")
        specs.append(ParameterTargetShape(target, int(module.weight.shape[1]), size))
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class _PreparedRows:
    source: PackedPhaseRow
    pin: PagePin
    rows: torch.Tensor | slice
    fields: tuple[torch.Tensor, ...]
    binding: ParameterOverlayFieldBinding
    offset: int
    size: int


class _ParameterModule:
    def __init__(self, module: _Linear, mlp: _Mlp | None) -> None:
        self.module, self.mlp = module, mlp
        self.bindings: list[tuple[ParameterOverlayFieldBinding, int, int]] = []
        self.original_forward = module.forward
        self.original_mlp_forward = None if mlp is None else mlp.forward
        self.prepared: tuple[_PreparedRows, ...] = ()
        self.captures: tuple[CapturedParameterOperand, ...] = ()
        self.mlp_input: torch.Tensor | None = None
        self.total_rows = 0
        self.apply_count = self.updated_apply_count = 0
        self.capture_operands = True

    def forward_mlp(self, value: torch.Tensor) -> torch.Tensor:
        if self.original_mlp_forward is None:
            raise RuntimeError("operand capture requires an MLP")
        self.mlp_input = value.detach()
        try:
            return self.original_mlp_forward(value)
        finally:
            self.mlp_input = None

    def forward(self, hidden: torch.Tensor) -> _LinearOutput:
        base = self.original_forward(hidden)
        if not self.prepared:
            return base
        if hidden.shape[0] != self.total_rows or self.captures:
            raise RuntimeError("parameter application differs from the prepared model step")
        # The unquantized TP1 projection owns this fresh output, not an input or state page.
        output = base[0] if isinstance(base, tuple) else base
        captures: list[CapturedParameterOperand] = []
        capturing = self.capture_operands and self.original_mlp_forward is not None
        for item in self.prepared:
            selected = hidden[item.rows] if isinstance(item.rows, slice) else hidden.index_select(0, item.rows)
            if item.binding.operator is ParameterOverlayOperator.DENSE_ADD:
                delta = F.linear(selected, item.fields[0]) * item.binding.scale
            else:
                a, b = item.fields
                delta = F.linear(F.linear(selected, a), b) * item.binding.scale
            if capturing:
                if (
                    self.mlp_input is None
                    or self.mlp_input.shape[0] != self.total_rows
                    or not isinstance(item.rows, torch.Tensor)
                ):
                    raise RuntimeError("operand capture requires owned operand rows and the original MLP input")
                captures.append(
                    CapturedParameterOperand(
                        item.source.identity,
                        item.binding.target,
                        item.source.phase,
                        selected.detach(),
                        self.mlp_input.index_select(0, item.rows).detach(),
                        self.module.weight.detach(),
                    )
                )
            columns = output[:, item.offset : item.offset + item.size]
            if isinstance(item.rows, slice):
                columns[item.rows].add_(delta)
            else:
                columns.index_add_(0, item.rows, delta)
            self.apply_count += selected.shape[0]
            if any(ref.version_id > 0 for ref in item.pin.refs):
                self.updated_apply_count += selected.shape[0]
        self.captures = tuple(captures)
        return base

    def clear(self) -> None:
        self.prepared, self.captures = (), ()
        self.mlp_input = None
        self.total_rows = 0
        self.capture_operands = True


class VllmParameterOverlay:
    """One hook per physical module, including fused q/v; no version selection."""

    def __init__(
        self, model: Qwen2ForCausalLM, bindings: tuple[ParameterOverlayFieldBinding, ...], *, capture: bool = True
    ) -> None:
        modules: dict[int, _ParameterModule] = {}
        for binding in bindings:
            module, offset, size, mlp = _locate(model, binding.target)
            if id(module) not in modules:
                modules[id(module)] = _ParameterModule(module, mlp)
            modules[id(module)].bindings.append((binding, offset, size))
        self._modules = tuple(modules.values())
        for handle in self._modules:
            handle.capture_operands = capture
            handle.module.forward = handle.forward
            if handle.mlp is not None:
                handle.mlp.forward = handle.forward_mlp

    @property
    def apply_count(self) -> int:
        return sum(handle.apply_count for handle in self._modules)

    @property
    def updated_apply_count(self) -> int:
        return sum(handle.updated_apply_count for handle in self._modules)

    def prepare(
        self, rows: PackedPhaseRowBatch, context: StateTaskContext[DataT], directory: RequestStateDirectory
    ) -> None:
        identities = tuple(dict.fromkeys(row.identity for row in rows.rows))
        resolved: list[tuple[tuple[PackedPhaseRow, ...], PagePin, StateRef]] = []
        for identity in identities:
            source = tuple(row for row in rows.rows if row.identity == identity)
            if len({row.phase for row in source}) != 1:
                raise ValueError("parameter application requires one phase per request")
            if directory.scope is StateScope.INSTANCE:
                pin = directory.inference_pin(identity, context.pin.stream)
                (ref,) = pin.refs
            else:
                pin = context.pin
                ref = context.selection.ref_for(identity)
            resolved.append((source, pin, ref))
        for handle in self._modules:
            prepared: list[_PreparedRows] = []
            # Operand capture owns per-request rows on MLP targets; shared launches
            # group only rows reading the exact same admitted root without capture.
            capturing = handle.capture_operands and handle.mlp is not None
            groups: dict[
                tuple[ParameterOverlayFieldBinding, StateRef, RequestIdentity | None],
                tuple[PagePin, list[PackedPhaseRow], int, int],
            ] = {}
            for source, pin, ref in resolved:
                for binding, offset, size in handle.bindings:
                    key = (binding, ref, source[0].identity if capturing else None)
                    if key not in groups:
                        groups[key] = (pin, [], offset, size)
                    groups[key][1].extend(source)
            for (binding, ref, _), (pin, source, offset, size) in groups.items():
                packed = tuple(row.packed_row_index for row in source)
                indices = (
                    upload(packed, handle.module.weight.device)
                    if capturing or packed != tuple(range(packed[0], packed[0] + len(packed)))
                    else slice(packed[0], packed[0] + len(packed))
                )
                fields = tuple(context.arena.borrow_field(pin, ref, field) for field in binding.fields)
                shapes = (
                    [(size, handle.module.weight.shape[1])]
                    if binding.operator is ParameterOverlayOperator.DENSE_ADD
                    else [(fields[0].shape[0], handle.module.weight.shape[1]), (size, fields[0].shape[0])]
                )
                if len(fields) != len(shapes) or any(
                    value.shape != shape
                    or value.dtype != handle.module.weight.dtype
                    or value.device != handle.module.weight.device
                    for value, shape in zip(fields, shapes, strict=True)
                ):
                    raise ValueError("signed parameter fields differ from the physical projection")
                prepared.append(_PreparedRows(source[0], pin, indices, fields, binding, offset, size))
            handle.prepared, handle.total_rows = tuple(prepared), rows.total_rows

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        captures: list[CapturedParameterOperand] = []
        for handle in self._modules:
            expected = (
                len(handle.prepared) if handle.capture_operands and handle.original_mlp_forward is not None else 0
            )
            if len(handle.captures) != expected:
                raise RuntimeError("parameter target did not capture every signed operand")
            captures.extend(handle.captures)
            handle.captures = ()
        return tuple(captures)

    def prepare_replay(
        self, row: PackedPhaseRow, layer: int, arena: PageBundleArena, pin: PagePin, ref: StateRef
    ) -> None:
        self.clear()
        for handle in self._modules:
            handle.capture_operands = False
            handle.total_rows = 1
            handle.prepared = tuple(
                _PreparedRows(
                    row,
                    pin,
                    slice(0, 1),
                    tuple(arena.borrow_field(pin, ref, field) for field in binding.fields),
                    binding,
                    offset,
                    size,
                )
                for binding, offset, size in handle.bindings
                if binding.target.layer == layer
            )

    def clear(self) -> None:
        for handle in self._modules:
            handle.clear()

    def close(self) -> None:
        self.clear()
        for handle in self._modules:
            handle.module.forward = handle.original_forward
            if handle.mlp is not None and handle.original_mlp_forward is not None:
                handle.mlp.forward = handle.original_mlp_forward
