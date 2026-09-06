"""Typed eager DENSE_ADD application and operand capture for vLLM MRv2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from tllm.canonical.backend_program import ParameterTargetShape
from tllm.canonical.state_execution import DataT, StateTaskContext
from tllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from tllm.contracts.state_ref import StateRef
from torch.nn import Parameter
from torch.nn import functional as F

from tllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterModuleKind,
    ParameterOverlayFieldBinding,
    ParameterRole,
    ParameterTarget,
)
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM


_LinearOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]
_LinearForward = Callable[[torch.Tensor], _LinearOutput]
_MlpForward = Callable[[torch.Tensor], torch.Tensor]


class _DenseLinear(Protocol):
    weight: Parameter
    bias: Parameter | None
    tp_size: int
    return_bias: bool
    forward: _LinearForward


class _Mlp(Protocol):
    forward: _MlpForward

    @property
    def down_proj(self) -> _DenseLinear: ...


@dataclass(frozen=True, slots=True)
class _PreparedRows:
    source: PackedPhaseRow
    ref: StateRef
    delta: torch.Tensor
    rows: torch.Tensor


class VllmDenseAddTarget:
    """One restored-on-close target with no version-selection authority."""

    __slots__ = (
        "_apply_count",
        "_updated_apply_count",
        "_base_weight",
        "_captures",
        "_mlp",
        "_mlp_input",
        "_module",
        "_original_forward",
        "_original_mlp_forward",
        "_prepared",
        "_binding",
        "_total_rows",
    )

    def __init__(
        self,
        binding: ParameterOverlayFieldBinding,
        mlp: _Mlp,
        module: _DenseLinear,
    ) -> None:
        self._binding = binding
        self._mlp = mlp
        self._module = module
        self._original_mlp_forward: _MlpForward = mlp.forward
        self._original_forward: _LinearForward = module.forward
        self._base_weight = module.weight.detach()
        self._prepared: tuple[_PreparedRows, ...] = ()
        self._captures: tuple[CapturedParameterOperand, ...] = ()
        self._mlp_input: torch.Tensor | None = None
        self._total_rows = 0
        self._apply_count = 0
        self._updated_apply_count = 0
        mlp.forward = self.forward_mlp
        module.forward = self.forward

    @property
    def target(self) -> ParameterTarget:
        return self._binding.target

    @property
    def base_weight(self) -> torch.Tensor:
        return self._base_weight

    @property
    def input_size(self) -> int:
        return int(self._module.weight.shape[1])

    @property
    def output_size(self) -> int:
        return int(self._module.weight.shape[0])

    @property
    def apply_count(self) -> int:
        return self._apply_count

    @property
    def updated_apply_count(self) -> int:
        return self._updated_apply_count

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
    ) -> None:
        prepared: list[_PreparedRows] = []
        for selected in context.selection.rows:
            source_rows = tuple(row for row in rows.rows if row.identity == selected.identity)
            if not source_rows or len({row.phase for row in source_rows}) != 1:
                raise RuntimeError("dense-add request requires one nonempty packed phase")
            ref = context.pin.refs[selected.state_index]
            value = context.arena.borrow_field(context.pin, ref, self._binding.fields[0])
            if (
                value.shape != self._module.weight.shape
                or value.dtype is not self._module.weight.dtype
                or value.device != self._module.weight.device
            ):
                raise ValueError("dense-add delta differs from target weight shape")
            prepared.append(
                _PreparedRows(
                    source_rows[0],
                    ref,
                    value,
                    torch.tensor(
                        tuple(row.packed_row_index for row in source_rows),
                        dtype=torch.long,
                        device=self._module.weight.device,
                    ),
                )
            )
        self._prepared = tuple(prepared)
        self._captures = ()
        self._total_rows = rows.total_rows

    def forward_mlp(self, mlp_input: torch.Tensor) -> torch.Tensor:
        if self._prepared:
            self._mlp_input = mlp_input.detach()
        try:
            return self._original_mlp_forward(mlp_input)
        finally:
            self._mlp_input = None

    def forward(self, hidden: torch.Tensor) -> _LinearOutput:
        base = self._original_forward(hidden)
        if not self._prepared:
            return base
        mlp_input = self._mlp_input
        if (
            mlp_input is None
            or int(hidden.shape[0]) != self._total_rows
            or int(mlp_input.shape[0]) != self._total_rows
        ):
            raise RuntimeError("dense-add capture rows differ from model operands")
        match base:
            case torch.Tensor() as base_tensor:
                output = base_tensor.clone()
                bias: Parameter | None = None
                tuple_output = False
            case (torch.Tensor() as base_tensor, bias):
                output = base_tensor.clone()
                tuple_output = True
        captures: list[CapturedParameterOperand] = []
        if self._captures:
            raise RuntimeError("dense-add target executed twice in one model step")
        for prepared in self._prepared:
            selected_hidden = hidden.index_select(0, prepared.rows)
            selected_desired = mlp_input.index_select(0, prepared.rows)
            output.index_add_(
                0,
                prepared.rows,
                F.linear(selected_hidden, prepared.delta),
            )
            captures.append(
                CapturedParameterOperand(
                    prepared.source.identity,
                    self.target,
                    prepared.source.phase,
                    selected_hidden.detach().clone(),
                    selected_desired.detach().clone(),
                    self._base_weight,
                )
            )
            self._apply_count += prepared.rows.numel()
            if int(prepared.ref.version_id) > 0:
                self._updated_apply_count += prepared.rows.numel()
        self._captures = tuple(captures)
        return (output, bias) if tuple_output else output

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        captures = self._captures
        self._captures = ()
        if self._prepared and len(captures) != len(self._prepared):
            raise RuntimeError("dense-add target did not capture every row assignment")
        return captures

    def clear(self) -> None:
        self._prepared = ()
        self._captures = ()
        self._mlp_input = None
        self._total_rows = 0

    def close(self) -> None:
        self.clear()
        self._mlp.forward = self._original_mlp_forward
        self._module.forward = self._original_forward


class VllmDenseAddAdapter:
    """Mechanical collection of typed targets selected by a signed attachment."""

    __slots__ = ("_targets",)

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        bindings: tuple[ParameterOverlayFieldBinding, ...],
    ) -> None:
        handles: list[VllmDenseAddTarget] = []
        modules: set[int] = set()
        for binding in bindings:
            target = binding.target
            if (
                target.module is not ParameterModuleKind.MLP
                or target.role is not ParameterRole.DOWN_PROJECTION_WEIGHT
            ):
                raise ValueError("vLLM dense-add requires an MLP down projection")
            mlp = model.model.layers[target.layer].mlp
            module = mlp.down_proj
            if (
                id(module) in modules
                or module.tp_size != 1
                or not module.return_bias
                or module.bias is not None
            ):
                raise RuntimeError("vLLM dense-add target is not an exact TP1 weight")
            modules.add(id(module))
            handles.append(VllmDenseAddTarget(binding, mlp, module))
        self._targets = tuple(handles)

    @property
    def targets(self) -> tuple[VllmDenseAddTarget, ...]:
        return self._targets

    @property
    def apply_count(self) -> int:
        return sum(target.apply_count for target in self._targets)

    @property
    def updated_apply_count(self) -> int:
        return sum(target.updated_apply_count for target in self._targets)

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
    ) -> None:
        for target in self._targets:
            target.prepare(rows, context)

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        return tuple(
            capture
            for target in self._targets
            for capture in target.take_captures()
        )

    def clear(self) -> None:
        for target in self._targets:
            target.clear()

    def close(self) -> None:
        for target in self._targets:
            target.close()


def inspect_dense_add_targets(
    model: Qwen2ForCausalLM,
    targets: tuple[ParameterTarget, ...],
    dtype: torch.dtype,
) -> tuple[ParameterTargetShape, ...]:
    specs: list[ParameterTargetShape] = []
    modules: set[int] = set()
    for target in targets:
        if (
            target.module is not ParameterModuleKind.MLP
            or target.role is not ParameterRole.DOWN_PROJECTION_WEIGHT
        ):
            raise ValueError("vLLM dense-add requires an MLP down projection")
        try:
            module = model.model.layers[target.layer].mlp.down_proj
        except IndexError as exc:
            raise ValueError("dense-add target layer is outside the model") from exc
        if (
            id(module) in modules
            or module.tp_size != 1
            or not module.return_bias
            or module.bias is not None
            or module.weight.dtype is not dtype
        ):
            raise RuntimeError("vLLM dense-add target is not an exact TP1 weight")
        modules.add(id(module))
        specs.append(
            ParameterTargetShape(
                target,
                int(module.weight.shape[1]),
                int(module.weight.shape[0]),
            )
        )
    return tuple(specs)


__all__ = [
    "VllmDenseAddAdapter",
    "VllmDenseAddTarget",
    "inspect_dense_add_targets",
]
