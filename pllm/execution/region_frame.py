"""An invocation's mathematical values; never a store of mutable resource state.

Access and last-use expressions were derived by the native compiler. This
frame releases Torch references only. Page pins, publication and GPU completion
remain with their original owners and are not represented by schedule times.
"""

from __future__ import annotations

from collections.abc import Mapping
from heapq import heappop, heappush

from pllm.contracts.region_bindings import AccessTable, RegionBindings
from pllm.contracts.region_program import BankType, DataType, resolve_extent
from pllm.execution.region_schedule import checked_schedule_integer, evaluate_schedule_expression
from pllm.execution.region_values import Bank, Data, check_data, copy_data

ValuePoint = tuple[int, tuple[int, ...]]


def lookup(table: AccessTable | None, point: tuple[int, ...], sizes: Mapping[str, int] = {}) -> ValuePoint | None:
    if table is None:
        return None
    if len(point) != table.rank:
        raise ValueError("value accessor coordinate rank mismatch")
    variables = {f"__size_{name}": value for name, value in sizes.items()}
    variables.update((f"p{i}", coordinate) for i, coordinate in enumerate(point))
    result: ValuePoint | None = None
    for case in table.cases:
        if evaluate_schedule_expression(case.condition, variables):
            if result is not None:
                raise RuntimeError("verified value accessor has overlapping source cases")
            result = (
                case.target_id,
                tuple(checked_schedule_integer(evaluate_schedule_expression(c, variables)) for c in case.coordinates),
            )
    return result


class RegionValueFrame:
    """One execution's live SSA leaves, including exact old bank elements."""

    def __init__(self, bindings: RegionBindings, inputs: Mapping[str, Data], sizes: Mapping[str, int]) -> None:
        # The executor supplies its checked invocation-entry snapshot. This
        # frame consumes value routes and layouts, not the source Program.
        self._sizes = sizes
        self._bindings = bindings
        self._sources = {t.input_id: t for t in bindings.sources}
        self._scalars = {t.input_id: t for t in bindings.scalars}
        self._last_uses = {t.input_id: t for t in bindings.last_uses}
        self._retained = {t.input_id: t for t in bindings.retained}
        self._times = {t.input_id: t for t in bindings.times}
        self._live: dict[ValuePoint, Data] = {}
        self._releases: list[tuple[tuple[int, ...], ValuePoint]] = []
        self._time: tuple[int, ...] | None = None
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

    def time_for(self, statement_id: int, coordinates: tuple[int, ...]) -> tuple[int, ...]:
        mapped = lookup(self._times.get(statement_id), coordinates, self._sizes)
        if mapped is None:
            raise RuntimeError("invocation has no verified schedule time")
        return mapped[1]

    def advance(self, statement_id: int, coordinates: tuple[int, ...]) -> None:
        time = self.time_for(statement_id, coordinates)
        if self._time is not None and time < self._time:
            raise RuntimeError("execution cursor moved backwards in schedule time")
        self._time = time
        # Equal-time invocations may appear in either order. Keep their values
        # until the next time, regardless of ISL's ordering of independent leaves.
        while self._releases and self._releases[0][0] < time:
            _, point = heappop(self._releases)
            del self._live[point]
            self.released_values += 1

    def _read_leaf(self, value_id: int, coordinates: tuple[int, ...]) -> Data:
        source = lookup(self._sources.get(value_id), coordinates, self._sizes)
        if source is None:
            raise RuntimeError("value has no verified original source")
        source_id, source_coordinates = source
        scalar = lookup(self._scalars.get(source_id), source_coordinates, self._sizes)
        if scalar is not None:
            return scalar[1][0]
        try:
            return self._live[source]
        except KeyError as error:
            raise RuntimeError("source value was not produced or was released too early") from error

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
        site = self._bindings.values[value_id]
        check_data(value, site.type, self._sizes)

        def insert(item: Data, indices: tuple[int, ...]) -> None:
            if isinstance(item, Bank):
                for i, child in enumerate(item.elements):
                    insert(child, (*indices, i))
                return
            point = (*coordinates, *indices)
            key = (value_id, point)
            retained = lookup(self._retained.get(value_id), point, self._sizes) is not None
            last = lookup(self._last_uses.get(value_id), point, self._sizes)
            if not retained and last is None:
                return
            if key in self._live:
                raise RuntimeError("SSA leaf was produced more than once")
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
