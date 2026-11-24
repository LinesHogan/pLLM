"""Parameter effects and model advancement declarations shared by authoring and backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from math import prod
from typing import Literal, Self, get_args

import torch

from pllm.contracts.hidden_locator import HiddenPhase, LayerRouteSelector, resolve_route_layers
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.tensor_layout import LayoutFieldId

OverlayRefresh = Literal["step", "admission"]


class ParameterOverlayOperator(StrEnum):
    DENSE_ADD = "dense_add"
    LOW_RANK_ADD = "low_rank_add"
    HEAD_ADD = "head_add"
    CUSTOM = "custom"


# Hugging Face module names of the projections pLLM can adapt, in their order inside one decoder layer.
ProjectionName = Literal["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_PROJECTION_ORDER: tuple[ProjectionName, ...] = get_args(ProjectionName)


@dataclass(frozen=True, slots=True)
class Projection:
    """One linear projection of the base model by its Hugging Face module name; `layer=None` means every layer."""

    name: ProjectionName
    layer: int | None = None

    def __post_init__(self) -> None:
        if self.name not in _PROJECTION_ORDER or self.layer is not None and self.layer < 0:
            raise ValueError(f"Projection needs name in {_PROJECTION_ORDER} and layer >= 0 or None; got {self}")

    def __lt__(self, other: Projection) -> bool:
        """Model order: by layer, then attention and MLP projections within a layer."""
        assert self.layer is not None and other.layer is not None
        return (self.layer, _PROJECTION_ORDER.index(self.name)) < (other.layer, _PROJECTION_ORDER.index(other.name))


def expand_targets(targets: tuple[Projection, ...], layers: int) -> tuple[Projection, ...]:
    """Resolve layer=None declarations against the loaded model, in projection order."""
    if all(target.layer is not None for target in targets):
        return targets
    return tuple(
        sorted(
            Projection(target.name, layer)
            for target in targets
            for layer in (range(layers) if target.layer is None else (target.layer,))
        )
    )


@dataclass(frozen=True, slots=True)
class ProjectionShape:
    """Rank-local projection widths and their partitions, shared by model probes and workload planning."""

    target: Projection
    input_size: int
    output_size: int
    input_partitions: int = 1
    output_partitions: int = 1

    @property
    def global_shape(self) -> tuple[int, int]:
        """Input and output widths before tensor parallel partitioning."""
        return self.input_size * self.input_partitions, self.output_size * self.output_partitions

    @property
    def target_size(self) -> int:
        """Width of operand('target'): MLP input for down, projection input for columns."""
        return self.output_size if self.target.name == "down_proj" else self.input_size

    @property
    def partitions(self) -> int:
        """Number of distinct shards; KV heads can replicate across adjacent ranks."""
        return max(self.input_partitions, self.output_partitions)


@dataclass(frozen=True, slots=True)
class FieldView:
    """A logical tensor inside one PTTS field, shared by producers and model readers."""

    field: LayoutFieldId
    offset: int
    shape: tuple[int, ...]

    def read(self, tensor: torch.Tensor, *, pooled: bool = False) -> torch.Tensor:
        """Borrow the named slice while preserving a pool's leading slot axis."""
        prefix = (len(tensor),) if pooled else ()
        flat = tensor.reshape(*prefix, -1)
        # Even a full-width narrow allocates and copies a full gradient in backward.
        if self.offset or prod(self.shape) != flat.shape[-1]:
            flat = flat.narrow(int(pooled), self.offset, prod(self.shape))
        return flat.view((*prefix, *self.shape))


@dataclass(frozen=True, slots=True)
class OverlayRoute:
    """Device field view selecting the active physical layer for a routed projection overlay."""

    resource: str
    field: FieldView
    refresh: OverlayRefresh = "step"


@dataclass(frozen=True, slots=True)
class ParameterOverlayFieldBinding:
    target: Projection
    operator: ParameterOverlayOperator
    fields: tuple[FieldView, ...]
    resource: str
    scale: float = 1.0
    # The overlay function reads operand("weight"): only then does a quantized projection get a dense copy.
    reads_weight: bool = False
    reads_embedding: bool = False
    route: OverlayRoute | None = None
    # Stable identifier for ParameterOverlayOperator.CUSTOM callables; empty for built-in operators.
    operator_id: str = ""


@dataclass(frozen=True, slots=True)
class ProjectionOperands:
    """Projection tensors and a scalar prompt-end fact, live through captured replay."""

    identity: RequestIdentity
    target: Projection
    phase: HiddenPhase
    hidden: torch.Tensor
    desired: torch.Tensor
    base_weight: torch.Tensor
    prompt_end: torch.Tensor
    embedding: torch.Tensor


@dataclass(frozen=True)
class ProjectionDelta:
    """Apply request state at original model projections."""

    state: str
    targets: tuple[Projection, ...]
    weight: str = "value"
    operator: ParameterOverlayOperator = ParameterOverlayOperator.DENSE_ADD
    factors: tuple[str, ...] = ("a", "b")
    scale: float = 1.0
    route: LayerRouteSelector | None = None
    refresh: OverlayRefresh = "admission"
    target_names: tuple[ProjectionName, ...] = ()
    custom_operator: Callable[..., torch.Tensor] | None = None
    # Per-field layer groups that share one PTTS LayoutField per projection name: ((field, ((l0, l1, ...), ...)), ...).
    share: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = ()

    @property
    def applied_fields(self) -> tuple[str, ...]:
        """The fields consumed by the declared projection operator."""
        return (
            self.factors
            if self.operator in (ParameterOverlayOperator.LOW_RANK_ADD, ParameterOverlayOperator.CUSTOM)
            else (self.weight,)
        )

    def shared_leader_index(self) -> dict[tuple[str, int], int]:
        """Map (field_name, target_index) to its owning leader target index for shared fields."""
        if not self.share:
            return {}
        target_index = {target: i for i, target in enumerate(self.targets)}
        leaders: dict[tuple[str, int], int] = {}
        for field_name, groups in self.share:
            layer_to_leader = {layer: group[0] for group in groups for layer in group[1:]}
            for i, target in enumerate(self.targets):
                if target.layer is not None and target.layer in layer_to_leader:
                    leader_target = Projection(target.name, layer_to_leader[target.layer])
                    if leader_target in target_index:
                        leaders[field_name, i] = target_index[leader_target]
        return leaders

    def expand(self, layers: int) -> Self:
        """The same effect with layer=None or routed targets resolved against a model of `layers` layers."""
        if self.route is not None:
            over = resolve_route_layers(self.route.over, layers)
            targets = tuple(Projection(name, layer) for layer in over for name in self.target_names)
            return replace(self, targets=targets, route=replace(self.route, over=over))
        return replace(self, targets=expand_targets(self.targets, layers))

    def field_bindings(
        self,
        view: Callable[[str], FieldView],
        *,
        reads_weight: bool = False,
        reads_embedding: bool = False,
        layer_bank: bool = False,
        route: OverlayRoute | None = None,
        operator_id: str = "",
    ) -> tuple[ParameterOverlayFieldBinding, ...]:
        """One binding per target; `view` finds each applied field's copy for target i, named `{field}_{i}`."""
        if self.route is not None and not layer_bank:
            names = self.target_names
            return tuple(
                ParameterOverlayFieldBinding(
                    target,
                    self.operator,
                    tuple(view(f"{key}_{names.index(target.name)}") for key in self.applied_fields),
                    self.state,
                    self.scale,
                    reads_weight,
                    reads_embedding,
                    route=route,
                    operator_id=operator_id,
                )
                for target in self.targets
            )
        if self.route is not None and layer_bank and len(self.target_names) == 1:
            return tuple(
                ParameterOverlayFieldBinding(
                    target,
                    self.operator,
                    tuple(view(f"{key}_{target.layer}") for key in self.applied_fields),
                    self.state,
                    self.scale,
                    reads_weight,
                    reads_embedding,
                    route=route,
                    operator_id=operator_id,
                )
                for target in self.targets
            )
        leaders = self.shared_leader_index()
        return tuple(
            ParameterOverlayFieldBinding(
                target,
                self.operator,
                tuple(view(f"{key}_{leaders.get((key, i), i)}") for key in self.applied_fields),
                self.state,
                self.scale,
                reads_weight,
                reads_embedding,
                route=route,
                operator_id=operator_id,
            )
            for i, target in enumerate(self.targets)
        )


__all__ = [
    "OverlayRefresh",
    "OverlayRoute",
    "ProjectionDelta",
    "ParameterOverlayFieldBinding",
    "ParameterOverlayOperator",
    "Projection",
    "ProjectionName",
    "ProjectionOperands",
]
