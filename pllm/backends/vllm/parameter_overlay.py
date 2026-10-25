"""Qwen2 target inspection for both readers, and the eager Exact projection-task driver."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import Parameter
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

from pllm.backends.vllm.region_layers import original_layer
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterModuleKind,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterRole,
    ParameterTarget,
)
from pllm.contracts.state_ref import StateRef
from pllm.execution.compile_facts import ParameterTargetShape
from pllm.execution.cuda_stream import upload
from pllm.execution.module_graph import capture_module_calls
from pllm.execution.overlay_task import ProjectionTransition, RouteOverlay
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


def locate_parameter(model: Qwen2ForCausalLM, target: ParameterTarget) -> tuple[_Linear, int, int, _Mlp | None]:
    """Resolve the supported projection and its disjoint output slice."""
    layer = model.model.layers[target.layer]
    if target.module is ParameterModuleKind.MLP and target.role is ParameterRole.DOWN_PROJECTION_WEIGHT:
        module = layer.mlp.down_proj
        return module, 0, int(module.weight.shape[0]), layer.mlp
    if target.module is ParameterModuleKind.ATTENTION:
        attention = layer.self_attn
        if target.role is ParameterRole.QUERY_PROJECTION_WEIGHT:
            return attention.qkv_proj, 0, attention.q_size, None
        if target.role is ParameterRole.VALUE_PROJECTION_WEIGHT:
            return attention.qkv_proj, attention.q_size + attention.kv_size, attention.kv_size, None
    raise ValueError("Qwen2 overlay requires an MLP down or attention q/v projection")


def validate_projection_sources(model: Qwen2ForCausalLM, targets: tuple[ParameterTarget, ...]) -> None:
    """Bind Exact to original Qwen projections; the query slice cannot write same-layer K/V."""
    for layer_index in dict.fromkeys(target.layer for target in targets):
        layer = model.model.layers[layer_index]
        if not original_layer(layer):
            raise ValueError("Exact overlays need the original Qwen2 layer source without implicit hooks")
        source = capture_module_calls(layer.self_attn)
        nodes = tuple(source.graph.nodes)
        projection = next(node for node in nodes if node.op == "call_module" and node.target == "qkv_proj")
        packed = next(node for node in nodes if node.target is operator.getitem and node.args == (projection, 0))
        split = next(node for node in nodes if node.op == "call_method" and node.target == "split")
        attention = layer.self_attn
        if split.args != (packed, [attention.q_size, attention.kv_size, attention.kv_size]) or split.kwargs != {
            "dim": -1
        }:
            raise ValueError("Exact query overlays need disjoint query, key and value slices from the fused projection")
        # Original Qwen2 independently normalizes Q/K and rotates each with the same positions.
        # Only qkv[:q_size] is changed; K/V here retain history, while higher layers see corrected hidden.
        slices = tuple(node.args[1] for node in split.users if node.target is operator.getitem)
        if slices != (0, 1, 2) or tuple(packed.users) != (split,):
            raise ValueError("Exact overlays need the original Qwen2 Q/K/V consumers")
        query = next(node for node in split.users if node.args[1] == 0)
        rotary = next(node for node in nodes if node.op == "call_module" and node.target == "rotary_emb")
        consumer = next(node for node in nodes if node.op == "call_module" and node.target == "attn")
        dependent: set[torch.fx.Node] = {query}
        for node in nodes:
            parents = node.all_input_nodes
            if node.target is operator.getitem and node.args[0] is rotary:
                # The original RoPE endpoint rotates Q and K independently with shared positions.
                output = node.args[1]
                assert isinstance(output, int)
                parent = rotary.args[1 + output]
                assert isinstance(parent, torch.fx.Node)
                parents = [parent]
            if any(parent in dependent for parent in parents):
                dependent.add(node)
        if consumer.args[1] in dependent or consumer.args[2] in dependent:
            raise ValueError("Exact query overlays need a Qwen2 source where the query correction cannot feed K/V")


def inspect_parameter_targets(
    model: Qwen2ForCausalLM, targets: tuple[ParameterTarget, ...], dtype: torch.dtype
) -> tuple[ParameterTargetShape, ...]:
    specs: list[ParameterTargetShape] = []
    if len(set(targets)) != len(targets):
        raise ValueError("parameter targets must be unique")
    for target in targets:
        module, offset, size, _ = locate_parameter(model, target)
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
    ref: StateRef
    rows: torch.Tensor
    binding: ParameterOverlayFieldBinding
    offset: int
    size: int


class _ExactProjection:
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
        self.transition: ProjectionTransition | None = None

    def captures_binding(self, binding: ParameterOverlayFieldBinding) -> bool:
        return self.original_mlp_forward is not None or binding.operator is ParameterOverlayOperator.HEAD_ADD

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
        inputs = tuple(hidden.index_select(0, item.rows) for item in self.prepared)
        captures = tuple(
            self.capture(item, selected)
            for item, selected in zip(self.prepared, inputs, strict=True)
            if self.captures_binding(item.binding)
        )
        assert self.transition is not None
        residuals = self.transition.run(captures)
        for item, selected in zip(self.prepared, inputs, strict=True):
            delta = (residuals[item.source.identity, item.binding.target] * item.binding.scale).to(output.dtype)
            columns = output[:, item.offset : item.offset + item.size]
            columns.index_add_(0, item.rows, delta)
            self.apply_count += selected.shape[0]
            if item.ref.version_id > 0:
                self.updated_apply_count += selected.shape[0]
        self.captures = () if isinstance(self.transition, RouteOverlay) else captures
        return base

    def capture(self, item: _PreparedRows, selected: torch.Tensor) -> CapturedParameterOperand:
        if item.binding.operator is ParameterOverlayOperator.HEAD_ADD:
            desired = selected
        else:
            assert self.mlp_input is not None and self.mlp_input.shape[0] == self.total_rows
            desired = self.mlp_input.index_select(0, item.rows)
        return CapturedParameterOperand(
            item.source.identity,
            item.binding.target,
            item.source.phase,
            selected.detach(),
            desired.detach(),
            self.module.weight[item.offset : item.offset + item.size].detach(),
        )

    def clear(self) -> None:
        self.prepared, self.captures = (), ()
        self.mlp_input = None
        self.total_rows = 0
        self.transition = None


class ExactParameterOverlay:
    """Eager Exact projection-task driver; pinned reads use the shared tensor op."""

    def __init__(self, model: Qwen2ForCausalLM, bindings: tuple[ParameterOverlayFieldBinding, ...]) -> None:
        modules: dict[int, _ExactProjection] = {}
        for binding in bindings:
            module, offset, size, mlp = locate_parameter(model, binding.target)
            if id(module) not in modules:
                modules[id(module)] = _ExactProjection(module, mlp)
            modules[id(module)].bindings.append((binding, offset, size))
        self._modules = tuple(modules.values())
        self._token_index = torch.zeros(1, dtype=torch.int64, device=self._modules[0].module.weight.device)
        for handle in self._modules:
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
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
        transition: ProjectionTransition | None = None,
    ) -> None:
        assert transition is not None
        identities = tuple(dict.fromkeys(row.identity for row in rows.rows))
        for handle in self._modules:
            prepared: list[_PreparedRows] = []
            for identity in identities:
                source = tuple(row for row in rows.rows if row.identity == identity)
                ref = context.selection.ref_for(identity)
                indices = upload(tuple(row.packed_row_index for row in source), handle.module.weight.device)
                prepared.extend(
                    _PreparedRows(source[0], ref, indices, binding, offset, size)
                    for binding, offset, size in handle.bindings
                )
            handle.transition = transition
            handle.prepared, handle.total_rows = tuple(prepared), rows.total_rows

    def prepare_token(self, row: PackedPhaseRow, transition: RouteOverlay) -> None:
        """Bind one row to the shared projection capture integration for this route occurrence."""
        index = transition.batch.selection.state_indices[transition.row]
        for handle in self._modules:
            handle.transition = transition
            handle.prepared = tuple(
                _PreparedRows(row, transition.batch.after_refs[index], self._token_index, binding, offset, size)
                for binding, offset, size in handle.bindings
            )
            handle.captures, handle.total_rows = (), 1

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        for handle in self._modules:
            if isinstance(handle.transition, RouteOverlay):
                continue
            expected = sum(handle.captures_binding(item.binding) for item in handle.prepared)
            assert len(handle.captures) == expected
            handle.captures = ()
        return ()

    def clear(self) -> None:
        for handle in self._modules:
            handle.clear()

    def close(self) -> None:
        self.clear()
        for handle in self._modules:
            handle.module.forward = handle.original_forward
            if handle.mlp is not None and handle.original_mlp_forward is not None:
                handle.mlp.forward = handle.original_mlp_forward
