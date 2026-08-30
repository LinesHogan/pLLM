"""Typed eager vLLM application of request-scoped dense overlay versions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import Parameter
from torch.nn import functional as F
from tllm.runtime.parameter_overlay import (
    OverlayStepPlan,
    OverlayTarget,
    OverlayVersion,
    OverlayVersionRef,
)


_LinearOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]
_LinearForward = Callable[[torch.Tensor], _LinearOutput]


class _DenseLinear(Protocol):
    weight: Parameter
    bias: Parameter | None
    tp_size: int
    return_bias: bool
    forward: _LinearForward


class _MLP(Protocol):
    @property
    def down_proj(self) -> _DenseLinear: ...


class _DecoderLayer(Protocol):
    @property
    def mlp(self) -> _MLP: ...


class _DecoderModel(Protocol):
    @property
    def layers(self) -> Sequence[_DecoderLayer]: ...


class _CausalModel(Protocol):
    @property
    def model(self) -> _DecoderModel: ...


class _ParallelConfig(Protocol):
    @property
    def tensor_parallel_size(self) -> int: ...

    @property
    def pipeline_parallel_size(self) -> int: ...


class _ModelConfig(Protocol):
    @property
    def quantization(self) -> str | None: ...


class _VllmConfig(Protocol):
    @property
    def model_config(self) -> _ModelConfig: ...


class DenseOverlayRunner(Protocol):
    @property
    def model(self) -> _CausalModel: ...

    @property
    def parallel_config(self) -> _ParallelConfig: ...

    @property
    def vllm_config(self) -> _VllmConfig: ...


@dataclass(frozen=True, slots=True)
class _PreparedRows:
    version: OverlayVersionRef
    rows: torch.Tensor


class VLLMDenseReplacementHandle:
    """One restored-on-close wrapper for a concrete decoder weight target."""

    def __init__(
        self,
        *,
        target: OverlayTarget,
        module: _DenseLinear,
    ) -> None:
        self._target = target
        self.module = module
        self._original_forward: _LinearForward = module.forward
        self._versions: dict[OverlayVersionRef, torch.Tensor] = {}
        self._prepared: tuple[_PreparedRows, ...] = ()
        self._total_rows = 0
        self._apply_count = 0
        module.forward = self.forward

    @property
    def target(self) -> OverlayTarget:
        return self._target

    @property
    def apply_count(self) -> int:
        return self._apply_count

    def has_version(self, ref: OverlayVersionRef) -> bool:
        return ref in self._versions

    def install_version(
        self,
        version: OverlayVersion,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        materialized = dict(tensors)
        if tuple(materialized) != ("weight",):
            raise RuntimeError(
                "dense replacement overlay must contain exactly one weight tensor"
            )
        weight = materialized["weight"]
        if tuple(weight.shape) != tuple(self.module.weight.shape):
            raise RuntimeError(
                "dense replacement shape does not match the target weight"
            )
        self._versions[version.ref] = weight.to(
            device=self.module.weight.device,
            dtype=self.module.weight.dtype,
            copy=True,
        ).contiguous()

    def retain_versions(self, refs: frozenset[OverlayVersionRef]) -> None:
        self._versions = {
            ref: weight for ref, weight in self._versions.items() if ref in refs
        }

    def prepare_step(self, plan: OverlayStepPlan) -> None:
        assignments = tuple(
            assignment
            for assignment in plan.assignments
            if assignment.target == self.target
        )
        self._prepared = tuple(
            _PreparedRows(
                assignment.version,
                torch.tensor(
                    assignment.packed_rows,
                    dtype=torch.long,
                    device=self.module.weight.device,
                ),
            )
            for assignment in assignments
        )
        self._total_rows = plan.total_rows

    def clear_step(self) -> None:
        self._prepared = ()
        self._total_rows = 0

    def forward(self, hidden: torch.Tensor) -> _LinearOutput:
        base = self._original_forward(hidden)
        if not self._prepared:
            return base
        if hidden.shape[0] != self._total_rows:
            raise RuntimeError(
                "dense replacement row plan does not match the model input"
            )
        output = base[0].clone()
        for prepared in self._prepared:
            weight = self._versions[prepared.version]
            replacement = F.linear(
                hidden.index_select(0, prepared.rows), weight
            )
            output.index_copy_(0, prepared.rows, replacement)
            self._apply_count += prepared.rows.numel()
        return output, None

    def restore(self) -> None:
        self.clear_step()
        self._versions.clear()
        self.module.forward = self._original_forward


def _resolve_dense_replacement_target(
    runner: DenseOverlayRunner, target: OverlayTarget
) -> _DenseLinear:
    if target.module != "mlp" or target.parameter != "down_proj.weight":
        raise RuntimeError(
            "vLLM dense overlays support only decoder MLP down_proj.weight"
        )
    layers = runner.model.model.layers
    layer = target.layer if target.layer >= 0 else len(layers) + target.layer
    if layer < 0 or layer >= len(layers):
        raise RuntimeError("dense replacement layer index is out of range")
    down_proj = layers[layer].mlp.down_proj
    if not down_proj.return_bias or down_proj.bias is not None:
        raise RuntimeError(
            "dense replacement requires tuple-returning biasless RowParallelLinear"
        )
    return down_proj


def validate_dense_replacement_targets(
    runner: DenseOverlayRunner, targets: tuple[OverlayTarget, ...]
) -> None:
    if runner.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError(
            "dense replacement currently requires tensor parallel size 1"
        )
    if runner.parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError(
            "dense replacement currently requires pipeline parallel size 1"
        )
    if runner.vllm_config.model_config.quantization is not None:
        raise RuntimeError("dense replacement requires an unquantized model")
    resolved: dict[int, OverlayTarget] = {}
    for target in targets:
        module = _resolve_dense_replacement_target(runner, target)
        module_identity = id(module)
        existing = resolved.get(module_identity)
        if existing is not None:
            raise RuntimeError(
                "dense overlay targets alias the same physical parameter: "
                f"{existing!r} and {target!r}"
            )
        resolved[module_identity] = target


def install_dense_replacement_target(
    runner: DenseOverlayRunner,
    target: OverlayTarget,
) -> VLLMDenseReplacementHandle:
    validate_dense_replacement_targets(runner, (target,))
    down_proj = _resolve_dense_replacement_target(runner, target)
    if down_proj.tp_size != 1:
        raise RuntimeError(
            "dense target shard requires tensor parallel size 1"
        )
    return VLLMDenseReplacementHandle(target=target, module=down_proj)


__all__ = [
    "DenseOverlayRunner",
    "VLLMDenseReplacementHandle",
    "install_dense_replacement_target",
    "validate_dense_replacement_targets",
]
