"""Shape-only inputs for explain's traces of the production model-site hooks.

Native normalization already belongs to the base model. Its stand-in returns the
same ports so only the hook's added math, gathers and copies are counted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import overload

import torch
from torch import Tensor

from pllm.backends.vllm.model_sites import ModelSites, SublayerBoundary, ValueProjection
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.model_sites import PatchInput, Place, ResidualRewrite, StateInput, position
from pllm.execution.site_memory import SiteMemorySchedule
from pllm.execution.task_memory import meta_site_state
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.torch_private import KernelCountMode


class _PortNorm:
    @overload
    def __call__(self, x: Tensor, residual: None = None) -> Tensor: ...
    @overload
    def __call__(self, x: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]: ...

    def __call__(self, x: Tensor, residual: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        return x if residual is None else (x, residual)


class _MemoryExample(SiteMemorySchedule):
    def __init__(self, rewrite: ResidualRewrite, binding: TensorDeclaration, rows: int) -> None:
        memory = rewrite.memory
        assert memory is not None
        self.memory = memory
        self.update = memory.update(rewrite, rewrite.places[0])
        self.views = (memory.carry.view, *memory.pending)
        layout = next(item.layout for item in binding.layouts.layouts if item.resource == memory.carry.resource)
        self.pools = tuple(
            view.read(torch.empty((rows, *field.shape), dtype=field.dtype.torch, device="meta"), pooled=True)
            for view in self.views
            for field in (layout.fields[view.field],)
        )
        self.table = torch.empty((2 + len(self.views), rows), dtype=torch.long, device="meta")
        self.rows = torch.empty(rows, dtype=torch.long, device="meta")
        # Each decode request contributes exactly one row; wider conditional
        # buckets do not execute, even though warmup/capture visits their bodies.
        self._counts = (1,) * rows
        self._buckets = {}


class _SiteExample(ModelSites):
    def __init__(self, binding: TensorDeclaration, facts: BackendCompileFacts) -> None:
        rewrites = binding.sites.rewrites
        self.streams = max((rewrite.streams for rewrite in rewrites), default=1)
        self.hidden, self.count = facts.probe.hidden_size, facts.max_admitted_lineages
        self.rows, self.active = None, None
        self._prefetch = {}
        self._requests_of = torch.empty(self.count, dtype=torch.long, device="meta")
        carries = {rewrite.memory.carry for rewrite in rewrites if rewrite.memory is not None}
        self.state_inputs = tuple(
            dict.fromkeys(
                source
                for rewrite in rewrites
                for _, source in (*rewrite.inputs, *rewrite.read_inputs)
                if isinstance(source, StateInput) and source not in carries
            )
        )
        self.state_values = {}
        for source in self.state_inputs:
            value = meta_site_state(binding, replace(source, module=None), self.count if source.request else None)
            assert isinstance(value, Tensor)
            self.state_values[source] = value
        self._result_sources = tuple(dict.fromkeys(source for rewrite in rewrites for source in rewrite.results))
        self.result_values = {
            source: torch.empty((self.count, *source.shape), dtype=source.dtype, device="meta")
            for source in self._result_sources
        }
        head, heads = facts.probe.head_dim, facts.probe.kv_heads
        self.earlier_values = {
            source: torch.empty(
                (self.count, heads, head)
                if source.kind == "value"
                else (self.count, self.streams, self.hidden)
                if self.streams > 1
                else (self.count, self.hidden),
                dtype=facts.probe.tensor_dtype.torch,
                device="meta",
            )
            for rewrite in rewrites
            for source in rewrite.earlier
        }
        self._memories = {
            rewrite.name: _MemoryExample(rewrite, binding, self.count)
            for rewrite in rewrites
            if rewrite.memory is not None
        }

    def patch(self, source: PatchInput, layer: int) -> Callable[[Tensor], Tensor]:
        # Activation-patch math is already traced by its tensor task.
        return _identity


def _identity(value: Tensor) -> Tensor:
    return value


def _boundaries(sites: _SiteExample, binding: TensorDeclaration, facts: BackendCompileFacts) -> list[SublayerBoundary]:
    owners = {place: rewrite for rewrite in binding.sites.rewrites for place in rewrite.places}
    earlier = tuple(sites.earlier_values)
    result: list[SublayerBoundary] = []
    for index in range(2 * facts.layer_count + 1):
        layer, mlp = divmod(index, 2)
        before: Place | None = ("attention", layer) if mlp else ("mlp", layer - 1) if layer else None
        after: Place | None = None if layer == facts.layer_count else ("mlp" if mlp else "attention", layer)
        reader = owners.get(after) if after is not None else None
        result.append(
            SublayerBoundary(
                sites,
                _PortNorm(),
                before,
                owners.get(before) if before is not None else None,
                after,
                reader if reader is not None and reader.read is not None else None,
                owners.get(("final", facts.layer_count)) if after is None else None,
                tuple(
                    source
                    for source in earlier
                    if source.kind != "value" and before is not None and source.position == position(before)
                ),
                (None, None),
            )
        )
    return result


def estimate_sites(binding: TensorDeclaration, facts: BackendCompileFacts) -> dict[str, tuple[int, int]]:
    """Trace added hook operators and eager staging copies once per decode step.

    Shared boundaries and retained values are charged to their first consumer.
    Body fusion is deliberately ignored, as with explain's logical byte counts.
    """
    if not binding.sites.rewrites:
        return {}
    sites = _SiteExample(binding, facts)
    sample = torch.empty(
        (facts.max_admitted_lineages, facts.probe.hidden_size), dtype=facts.probe.tensor_dtype.torch, device="meta"
    )
    counts = {rewrite.name: (0, 0) for rewrite in binding.sites.rewrites}
    for boundary in _boundaries(sites, binding, facts):
        sites.count = 1
        boundary.bind(sample[0], {})
        users = tuple(rewrite for rewrite in (boundary.owner, boundary.reader, boundary.final) if rewrite is not None)
        consumers = users or tuple(
            rewrite for rewrite in binding.sites.rewrites if any(source in rewrite.earlier for source in boundary.keeps)
        )
        if not consumers and sites.streams == 1:
            continue
        name = (consumers[0] if consumers else binding.sites.rewrites[0]).name
        stream = torch.empty((len(sample), sites.hidden * sites.streams), dtype=sample.dtype, device="meta")
        width = sites.hidden * (sites.streams if boundary.final is None else 1)
        inputs = [sample] if boundary.before is None else [sample, stream]
        outputs = [torch.empty_like(sample)]
        if boundary.before is not None:
            outputs.append(sample.new_empty((len(sample), width)))
        boundary.emit(inputs, outputs)  # Warm shape specializations outside the count.
        with KernelCountMode() as counter:
            boundary.emit(inputs, outputs)
        graph, eager = counts[name]
        counts[name] = graph + counter.count, eager
    _estimate_values(sites, binding, facts, sample, counts)
    _estimate_staging(sites, binding, counts)
    return counts


def _estimate_values(
    sites: _SiteExample,
    binding: TensorDeclaration,
    facts: BackendCompileFacts,
    sample: Tensor,
    counts: dict[str, tuple[int, int]],
) -> None:
    from pllm.execution.model_sites import EarlierInput

    owners = {place: rewrite for rewrite in binding.sites.rewrites for place in rewrite.places}
    layers = {layer for kind, layer in owners if kind == "value"} | {
        source.layer for source in sites.earlier_values if source.kind == "value"
    }
    width = facts.probe.kv_heads * facts.probe.head_dim
    for layer in sorted(layers):
        source = EarlierInput("value", layer)
        keeps = source if source in sites.earlier_values else None
        hook = ValueProjection(
            sites, _identity, layer, owners.get(("value", layer)), keeps, 0, width, facts.probe.head_dim
        )
        sites.count = 1
        hook.bind(sample[0], {})
        if hook.owner is None and keeps is None:
            continue
        name = (hook.owner or next(rewrite for rewrite in binding.sites.rewrites if source in rewrite.earlier)).name
        projected = sample.new_empty((len(sample), width))
        hook.emit([], [projected])
        with KernelCountMode() as counter:
            hook.emit([], [projected])
        graph, eager = counts[name]
        counts[name] = graph + counter.count, eager


def _estimate_staging(
    sites: _SiteExample,
    binding: TensorDeclaration,
    counts: dict[str, tuple[int, int]],
) -> None:
    for source in sites.state_inputs:
        name = next(
            rewrite.name
            for rewrite in binding.sites.rewrites
            if any(value == source for _, value in (*rewrite.inputs, *rewrite.read_inputs))
        )
        value = sites.state_values[source]
        with KernelCountMode() as counter:
            if source.request:
                for row in value:
                    row.copy_(torch.empty_like(row))
            else:
                value.copy_(torch.empty_like(value))
        graph, eager = counts[name]
        counts[name] = graph, eager + counter.count
    for source, value in sites.result_values.items():
        name = next(rewrite.name for rewrite in binding.sites.rewrites if source in rewrite.results)
        with KernelCountMode() as counter:
            value.copy_(torch.empty_like(value))
        graph, eager = counts[name]
        counts[name] = graph, eager + counter.count
