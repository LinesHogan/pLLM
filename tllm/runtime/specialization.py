"""Typed cold-path specialization facts supplied by runtime backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import TYPE_CHECKING, cast

import torch

from tllm.contracts.state_ref import LayoutId

if TYPE_CHECKING:
    from tllm.workload.frontend import WorkloadDeclaration


@dataclass(frozen=True, slots=True)
class _StateLayoutBinding:
    workload_id: str
    state_name: str
    layout_id: LayoutId


@dataclass(frozen=True, slots=True)
class RuntimeSpecializationSpec:
    rows: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device
    model_layer_count: int | None = None
    max_prefill_tokens: int | None = None
    _state_layouts: tuple[_StateLayoutBinding, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.rows, int) or isinstance(self.rows, bool) or self.rows < 1:
            raise ValueError("RuntimeSpecializationSpec.rows must be a positive int")
        if (
            not isinstance(self.hidden_size, int)
            or isinstance(self.hidden_size, bool)
            or self.hidden_size < 1
        ):
            raise ValueError(
                "RuntimeSpecializationSpec.hidden_size must be a positive int"
            )
        if not isinstance(cast(object, self.dtype), torch.dtype):
            raise TypeError("RuntimeSpecializationSpec.dtype must be torch.dtype")
        if not isinstance(cast(object, self.device), torch.device):
            raise TypeError("RuntimeSpecializationSpec.device must be torch.device")
        if self.model_layer_count is not None and (
            not isinstance(self.model_layer_count, int)
            or isinstance(self.model_layer_count, bool)
            or self.model_layer_count < 1
        ):
            raise ValueError(
                "RuntimeSpecializationSpec.model_layer_count must be a positive int or None"
            )
        if self.max_prefill_tokens is not None and (
            not isinstance(self.max_prefill_tokens, int)
            or isinstance(self.max_prefill_tokens, bool)
            or self.max_prefill_tokens < 1
        ):
            raise ValueError(
                "RuntimeSpecializationSpec.max_prefill_tokens must be a positive int or None"
            )
    def state_layout_id(
        self,
        workload_id: str,
        state_name: str,
    ) -> LayoutId:
        for binding in self._state_layouts:
            if (
                binding.workload_id == workload_id
                and binding.state_name == state_name
            ):
                return binding.layout_id
        raise RuntimeError("cold specialization has no interned state layout")


def bind_state_layout_ids(
    spec: RuntimeSpecializationSpec,
    graph_json: str,
    declarations: tuple["WorkloadDeclaration", ...],
) -> RuntimeSpecializationSpec:
    """Parse the Rust interner result once at the Python–Rust boundary."""

    from tllm import _core

    decoded = cast(object, json.loads(_core.intern_state_layout_ids(graph_json)))
    if type(decoded) is not list:
        raise TypeError("state layout interner must return a JSON array")
    decoded_bindings: list[tuple[str, LayoutId]] = []
    for position, item in enumerate(cast(list[object], decoded)):
        if type(item) is not dict:
            raise TypeError(
                f"state layout interner entry {position} must be an object"
            )
        fields = cast(dict[object, object], item)
        if set(fields) != {"resource", "layout_id"}:
            raise ValueError("state layout interner entry has the wrong fields")
        resource = fields["resource"]
        layout_id = fields["layout_id"]
        if type(resource) is not str or type(layout_id) is not int:
            raise TypeError("state layout interner entry has invalid field types")
        decoded_bindings.append((resource, LayoutId(layout_id)))
    composed = len(declarations) > 1
    declared_states = tuple(
        (
            declaration.ir.workload_id,
            state.name,
            (
                f"{declaration.ir.workload_id}/{state.name}"
                if composed
                else state.name
            ),
        )
        for declaration in declarations
        for state in declaration.ir.states
    )
    if len(decoded_bindings) != len(declared_states):
        raise RuntimeError(
            "state layout interner result differs from workload declarations"
        )
    bindings: list[_StateLayoutBinding] = []
    for (
        workload_id,
        state_name,
        expected_resource,
    ), (resource, layout_id) in zip(
        declared_states,
        decoded_bindings,
        strict=True,
    ):
        if resource != expected_resource:
            raise RuntimeError(
                "state layout interner order differs from workload declarations"
            )
        bindings.append(_StateLayoutBinding(workload_id, state_name, layout_id))
    return replace(spec, _state_layouts=tuple(bindings))


__all__ = [
    "RuntimeSpecializationSpec",
    "bind_state_layout_ids",
]
