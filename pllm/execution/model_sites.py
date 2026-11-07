"""@pllm.custom_residual methods as the model sees them: the sublayer residual adds (or value rows) each owns, the
inputs it reads, and the owner table that makes two methods of one site a conflict.

usage.py declares them into a TensorDeclaration; the vLLM backend's model_sites module installs them on the model's
norm modules and value projections and asks `SiteValues` for their inputs at every call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.semantic_graph import StateResource, TaskSymbol
from pllm.execution.module_graph import ModuleCall
from pllm.execution.overlay_task import ChunkUpdate

# Where a method applies: one sublayer's residual add (attention or MLP), both of a layer's adds, the value rows
# attention reads and caches, or the stream entering the model's final norm.
Site = Literal["attention", "mlp", "both", "value", "final"]
# One owned place: the residual add after a layer's attention or MLP, a layer's value rows, or the final stream.
Place = tuple[Literal["attention", "mlp", "value", "final"], int]


class SiteValues(Protocol):
    """What the backend hands a method at a call: state tensors aligned to the call's rows, and earlier values of the
    same tokens in this forward."""

    def state(self, source: StateInput) -> torch.Tensor: ...
    def earlier(self, source: EarlierInput) -> torch.Tensor: ...
    def result(self, source: ResultInput) -> torch.Tensor: ...
    def patch(self, source: PatchInput, layer: int) -> Callable[[torch.Tensor], torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class StateInput:
    """A state field: an instance state's field as declared, a request state's field per row (leading row axis).

    `module` makes the flat parameters of a module state callable like the module.
    """

    resource: StateResource
    view: FieldView
    request: bool
    module: torch.nn.Module | None = None

    def value(self, values: SiteValues, depth: int) -> torch.Tensor | ModuleCall:
        tensor = values.state(self)
        return tensor if self.module is None else ModuleCall(self.module, tensor, self.request)


@dataclass(frozen=True, slots=True)
class DepthInput:
    """The place's index: the layer, or 2 * layer (+ 1 for the MLP) for site="both"; the layer count at "final"."""

    def value(self, values: SiteValues, depth: int) -> int:
        return depth


@dataclass(frozen=True, slots=True)
class EarlierInput:
    """The same token's value at an earlier place of this forward: a layer's value rows, or the stream after a layer
    (after its attention for "attention_residual")."""

    kind: Literal["value", "residual", "attention_residual"]
    layer: int

    def value(self, values: SiteValues, depth: int) -> torch.Tensor:
        return values.earlier(self)

    @property
    def position(self) -> int:
        """Where in the forward the value exists (see `position`)."""
        kind: Literal["value", "attention", "mlp"] = (
            "value" if self.kind == "value" else "attention" if self.kind == "attention_residual" else "mlp"
        )
        return position((kind, self.layer))


@dataclass(frozen=True, slots=True)
class PatchInput:
    """The edit a pllm.model.ActivationPatch on `resource` applies to the MLP output of the place's layer, on the rows
    it selects with the fields of the versions they read."""

    resource: StateResource

    def value(self, values: SiteValues, depth: int) -> Callable[[torch.Tensor], torch.Tensor]:
        return values.patch(self, depth)


def activation_write(x: torch.Tensor, y: torch.Tensor, patch: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """An activation patch as a custom residual: the edited MLP output joins the stream with the plain add."""
    return x + patch(y)


@dataclass(frozen=True, slots=True)
class ResultInput:
    """A read-only method's pre-forward result, aligned to the model's scheduled rows."""

    task: TaskSymbol
    shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.float32

    def value(self, values: SiteValues, depth: int) -> torch.Tensor:
        return values.result(self)


SiteInput = StateInput | DepthInput | EarlierInput | PatchInput | ResultInput


def position(place: Place, *, read: bool = False) -> int:
    """Forward order of the places: layer l's input norm (3l, which completes the MLP of layer l - 1), its value rows
    (3l + 1), its post-attention norm (3l + 2), the final norm (3L). A residual add at a place is complete at its
    norm; a branch input is formed right after that norm completes, so `read` places sit at the same position."""
    kind, layer = place
    match kind:
        case "attention":
            return 3 * layer + 2 if not read else 3 * layer
        case "value":
            return 3 * layer + 1
        case "mlp":
            return 3 * (layer + 1) if not read else 3 * layer + 2
        case "final":
            return 3 * layer


@dataclass(frozen=True, slots=True)
class ResidualRewrite:
    """One @pllm.custom_residual method: the places it owns, its write (and optional read) method and their inputs.

    `write(x, y, ...)` returns the stream after a sublayer from the stream `x` before it and the branch output `y`;
    at "value" it maps value rows to value rows, at "final" the stream to one stream. `read(x, ...)` returns the
    branch input a sublayer normalizes instead of the stream. `streams` residual streams travel in vLLM's residual
    variable from the embedding to the final norm.
    """

    name: str
    site: Site
    places: tuple[Place, ...]
    write: Callable[..., torch.Tensor]
    inputs: tuple[tuple[str, SiteInput], ...]
    read: Callable[..., torch.Tensor] | None = None
    read_inputs: tuple[tuple[str, SiteInput], ...] = ()
    streams: int = 1
    # A stateful method's per-request memory; its write then returns (stream, next carry) for one chunk.
    memory: SiteMemory | None = None

    def depth(self, place: Place) -> int:
        kind, layer = place
        return 2 * layer + (kind == "mlp") if self.site == "both" else layer

    def call_write(self, place: Place, values: SiteValues, *tensors: torch.Tensor) -> torch.Tensor:
        depth = self.depth(place)
        return self.write(*tensors, **{name: source.value(values, depth) for name, source in self.inputs})

    def call_read(self, place: Place, values: SiteValues, stream: torch.Tensor) -> torch.Tensor:
        assert self.read is not None
        depth = self.depth(place)
        return self.read(stream, **{name: source.value(values, depth) for name, source in self.read_inputs})

    @property
    def results(self) -> tuple[ResultInput, ...]:
        """The pre-forward values the write or read method consumes."""
        return tuple(source for _, source in (*self.inputs, *self.read_inputs) if isinstance(source, ResultInput))

    @property
    def earlier(self) -> tuple[EarlierInput, ...]:
        """Every earlier value of the forward the methods read; the backend keeps a copy of each."""
        return tuple(source for _, source in (*self.inputs, *self.read_inputs) if isinstance(source, EarlierInput))

    def check_order(self) -> None:
        """A method reads only values the forward has computed before it runs at each of its places."""
        for place in self.places:
            for inputs, latest in ((self.inputs, position(place) - 1), (self.read_inputs, position(place, read=True))):
                for source in (source for _, source in inputs if isinstance(source, EarlierInput)):
                    if source.position > latest:
                        raise ValueError(
                            f"{self.name} reads pllm.{'value' if source.kind == 'value' else 'residual'}"
                            f"({source.layer}), which the forward computes after its {place[0]} site of layer "
                            f"{place[1]}; read an earlier layer"
                        )


@dataclass(frozen=True, slots=True)
class SiteMemory:
    """The per-request memory of a stateful custom residual (state= and chunk=): the carry field the method advances
    chunk by chunk, and the rows of a chunk not yet complete, which the framework keeps.

    Every row reads the carry its chunk started with; a chunk's update applies once its last row has run, so the
    result does not depend on how vLLM splits prompts into steps. A transition before the forward copies each
    request's version into a private write; the method's hook advances it; a publication after the forward makes it
    visible.
    """

    carry: StateInput
    # The fields of the unfinished chunk: its streams, branch outputs and length.
    pending: tuple[FieldView, FieldView, FieldView]
    chunk: int

    def invariants(self, rewrite: ResidualRewrite) -> tuple[StateInput, ...]:
        """The instance state the method reads besides its carry, passed to every chunk unchanged."""
        return tuple(source for _, source in rewrite.inputs if isinstance(source, StateInput) and source != self.carry)

    def update(self, rewrite: ResidualRewrite, place: Place) -> ChunkUpdate:
        """The chunk recurrence over one request's (stream, branch output) rows."""
        invariants = self.invariants(rewrite)
        return ChunkUpdate(
            MemoryStep(rewrite, place, self, invariants),
            self.chunk,
            ("carry", "x", "y"),
            tuple(f"__invariant_{index}" for index in range(len(invariants))),
        )


@dataclass(frozen=True)
class MemoryStep:
    """pllm.scan's step for one chunk: the method's (stream, next carry) as (next carry, stream rows)."""

    rewrite: ResidualRewrite
    place: Place
    memory: SiteMemory
    invariants: tuple[StateInput, ...]

    def __call__(
        self, carry: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *invariants: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = _ChunkValues(self.memory.carry, carry, dict(zip(self.invariants, invariants, strict=True)))
        stream, updated = self.rewrite.call_write(self.place, values, x, y)
        return updated, stream


@dataclass(frozen=True)
class _ChunkValues:
    """A chunk's inputs: the carry it started with and the instance state the method reads."""

    source: StateInput
    carry: torch.Tensor
    invariants: dict[StateInput, torch.Tensor]

    def state(self, source: StateInput) -> torch.Tensor:
        return self.carry if source == self.source else self.invariants[source]

    def earlier(self, source: EarlierInput) -> torch.Tensor:
        raise AssertionError("stateful custom residuals read no earlier values")

    def result(self, source: ResultInput) -> torch.Tensor:
        raise AssertionError("stateful custom residuals read no method results")

    def patch(self, source: PatchInput, layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
        raise AssertionError("stateful custom residuals apply no activation patch")


class SiteOwners:
    """The rewrites of one workload, keyed by the places they own; two owners of one place are a conflict."""

    def __init__(self) -> None:
        self.rewrites: list[ResidualRewrite] = []
        self.owners: dict[Place, ResidualRewrite] = {}

    def add(self, rewrite: ResidualRewrite) -> None:
        for place in rewrite.places:
            other = self.owners.get(place)
            if other is not None:
                kind, layer = place
                raise ValueError(
                    f'layer {layer} "{kind}" already has a custom residual ({other.name}); one method owns a '
                    "sublayer's residual update"
                    if kind != "final"
                    else f"the final stream already has a custom residual ({other.name})"
                )
        rewrite.check_order()
        self.rewrites.append(rewrite)
        self.owners.update((place, rewrite) for place in rewrite.places)

    def check(self, layer_count: int, custom_forward: bool, reads_streams: bool) -> None:
        """Rewrites the rest of the workload conflicts with. A stateful method advances its memory once per token,
        which a custom forward's layer visits do not respect. n > 1 residual streams exist from the embedding to the
        final norm, so every sublayer and the final stream need an owner, and nothing else may read the one-stream
        residual; one method declares streams=n and the others see the same n streams."""
        stateful = next((rewrite.name for rewrite in self.rewrites if rewrite.memory is not None), None)
        if stateful is not None and custom_forward:
            raise ValueError(
                f"{stateful} advances its memory once per token, but a @pllm.custom_forward may run its layer several "
                "times or not at all per token; drop state= or the custom forward"
            )
        counts = {rewrite.streams for rewrite in self.rewrites if rewrite.streams > 1}
        if not counts:
            return
        if len(counts) > 1:
            raise ValueError(f"custom residual methods declare different streams= {sorted(counts)}; declare one count")
        (count,) = counts
        places = {(kind, layer) for layer in range(layer_count) for kind in ("attention", "mlp")} | {
            ("final", layer_count)
        }
        if not places <= self.owners.keys():
            raise ValueError(
                f"streams={count} keeps {count} residual streams from the embedding to the final norm; own every "
                'attention and MLP site (site="both" on all layers) and add a @pllm.custom_residual(site="final") '
                "method that collapses them to one"
            )
        if custom_forward or reads_streams:
            raise ValueError(
                f"streams={count} keeps several residual streams; pllm.residual, pllm.layer_logits and "
                "@pllm.custom_forward read one stream, so read pllm.residual inside a custom residual method instead"
            )
