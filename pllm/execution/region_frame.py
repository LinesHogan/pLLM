"""An invocation's mathematical values; never a store of mutable resource state.

Access and last-use expressions were derived by the native compiler. This
frame releases Torch references only. Page pins, publication and GPU completion
remain with their original owners and are not represented by schedule times.
"""

from __future__ import annotations

from collections.abc import Mapping
from heapq import heappop, heappush
from itertools import product

import torch

from pllm.contracts.region_bindings import AccessTable, RegionBindings
from pllm.contracts.region_program import BankType, DataType, ResourceType, resolve_extent
from pllm.execution.region_schedule import checked_schedule_integer, evaluate_schedule_expression
from pllm.execution.region_values import Bank, Data, check_data, copy_data
from pllm.execution.region_window import RegionWindow

ValuePoint = tuple[int, tuple[int, ...]]


def lookup(table: AccessTable | None, point: tuple[int, ...], sizes: Mapping[str, int] = {}) -> ValuePoint | None:
    if table is None:
        return None
    variables = {f"__size_{name}": value for name, value in sizes.items()}
    variables.update((f"p{i}", coordinate) for i, coordinate in enumerate(point))
    result: ValuePoint | None = None
    for case in table.cases:
        if evaluate_schedule_expression(case.condition, variables):
            result = (
                case.target_id,
                tuple(checked_schedule_integer(evaluate_schedule_expression(c, variables)) for c in case.coordinates),
            )
    return result


Routes = dict[ValuePoint, ValuePoint]


def expand_accesses(
    tables: tuple[AccessTable, ...], domains: Mapping[int, tuple[int | str, ...]], sizes: Mapping[str, int]
) -> Routes:
    """Resolve a signed accessor over this window's concrete domain once."""
    result: Routes = {}
    for table in tables:
        for point in product(*(range(resolve_extent(n, sizes)) for n in domains[table.input_id])):
            target = lookup(table, point, sizes)
            if target is not None:
                result[table.input_id, point] = target
    return result


class RegionValueLayout:
    """Concrete routes and lifetimes shared by fresh frames of a fixed window."""

    def __init__(self, bindings: RegionBindings, sizes: Mapping[str, int], resources: tuple[ResourceType, ...]) -> None:
        domains = {site.id: (*site.extents, *site.bank_shape) for site in bindings.values}
        self.bindings = bindings
        self.sources = expand_accesses(bindings.sources, domains, sizes)
        self.scalars = expand_accesses(bindings.scalars, domains, sizes)
        self.last_uses = expand_accesses(bindings.last_uses, domains, sizes)
        self.retained = expand_accesses(bindings.retained, domains, sizes)
        self.resource_uses = expand_accesses(
            bindings.resource_uses, {i: r.domain for i, r in enumerate(resources)}, sizes
        )


class RegionValueFrame:
    """One execution's live SSA leaves, including exact old bank elements."""

    def __init__(
        self,
        layout: RegionValueLayout,
        inputs: Mapping[str, Data],
        sizes: Mapping[str, int],
        window: RegionWindow | None = None,
    ) -> None:
        # The executor supplies its checked invocation-entry snapshot. This
        # frame consumes value routes and layouts, not the source Program.
        self._window = window
        self._sizes = sizes
        self._bindings = bindings = layout.bindings
        self._layout = layout
        self._live: dict[ValuePoint, Data] = {}
        self._releases: list[tuple[tuple[int, ...], ValuePoint]] = []
        self.peak_live_values = 0
        self.released_values = 0
        arguments = tuple(bindings.values[i] for i in bindings.arguments)
        if set(inputs) != {site.name for site in arguments}:
            raise ValueError("region input bindings must be complete and exact")
        # The caller retains ownership. Detached private inputs prevent an
        # exported view result from exposing a borrowed caller or arena tensor.
        # Validate all shapes before copying any payload: a wrong bank length
        # must not allocate outside the bound invocation's workspace envelope.
        for site in arguments:
            check_data(inputs[site.name], site.type, self._sizes)
        for site in arguments:
            self.put(site.id, (), copy_data(inputs[site.name]))

    def advance(self, time: tuple[int, ...]) -> None:
        # Equal-time invocations may appear in either order. Keep their values
        # until the next time, regardless of ISL's ordering of independent leaves.
        while self._releases and self._releases[0][0] < time:
            _, point = heappop(self._releases)
            if self._window is not None:
                self._window.release(point)
            del self._live[point]
            self.released_values += 1

    def _read_leaf(self, value_id: int, coordinates: tuple[int, ...]) -> Data:
        source = self._layout.sources.get((value_id, coordinates))
        assert source is not None
        scalar = self._layout.scalars.get(source)
        if scalar is not None:
            return scalar[1][0]
        value = self._live[source]
        if self._window is not None:
            self._window.read(source, value)
        return value

    def read(self, value_id: int, coordinates: tuple[int, ...]) -> Data:
        site = self._bindings.values[value_id]
        # Loop invariants keep their original definition in an enclosing
        # scope; only that scope's coordinate prefix identifies the value.
        coordinates = coordinates[: len(site.extents)]

        def assemble(type_: DataType, indices: tuple[int, ...]) -> Data:
            if isinstance(type_, BankType):
                return Bank(
                    tuple(
                        assemble(type_.element, (*indices, i)) for i in range(resolve_extent(type_.size, self._sizes))
                    )
                )
            return self._read_leaf(value_id, (*coordinates, *indices))

        return assemble(site.type, ())

    def put(self, value_id: int, coordinates: tuple[int, ...], value: Data) -> None:
        def insert(item: Data, indices: tuple[int, ...]) -> None:
            if isinstance(item, Bank):
                for i, child in enumerate(item.elements):
                    insert(child, (*indices, i))
                return
            point = (*coordinates, *indices)
            key = (value_id, point)
            retained = key in self._layout.retained
            last = self._layout.last_uses.get(key)
            if not retained and last is None:
                return
            if self._window is not None and isinstance(item, torch.Tensor):
                item = self._window.store(key, item)
            self._live[key] = item
            if not retained and last is not None:
                heappush(self._releases, (last[1], key))
            self.peak_live_values = max(self.peak_live_values, len(self._live))

        insert(value, ())

    def results(self) -> tuple[Data, ...]:
        return tuple(self.read(i, ()) for i in self._bindings.results)

    def hold_live_values(self) -> tuple[Data, ...]:
        """Keep references through a physical completion, without copying data.

        The caller must fund the pending phase window and release these holds
        only after completion. Logical last use continues to retire frame slots.
        """
        return tuple(self._live.values())

    def close(self) -> None:
        self._live.clear()
        self._releases.clear()
