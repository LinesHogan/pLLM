"""Red differential contracts for the Phase 4 distributed-plan validator.

Malformed candidates are frozen, attribute-only proxies.  This is important:
the Python dataclass constructors reject many corruptions before they reach
the PyO3 boundary, while Rust must validate the same structural descriptors
that a future producer or another process could provide.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import importlib
from typing import Iterable
import unittest

from tllm.workgraph import phase3
from tllm.workgraph.frontend import StateScope
from tllm.workgraph.phase4 import (
    CollectiveBackend,
    CollectiveKey,
    CollectiveKind,
    CollectiveRequestDesc,
    CollectiveWindow,
    DistributedTopology,
    ParallelGroupKind,
    RankRole,
    TopologyRankDesc,
)


@dataclass(frozen=True, slots=True)
class _FrozenAttrs:
    """Frozen structural proxy accepted by attribute-based PyO3 readers."""

    entries: tuple[tuple[str, object], ...]

    def __getattr__(self, name: str) -> object:
        for key, value in self.entries:
            if key == name:
                return value
        raise AttributeError(name)


def _attrs(value: object) -> dict[str, object]:
    if isinstance(value, _FrozenAttrs):
        return dict(value.entries)
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _proxy(value: object, **changes: object) -> _FrozenAttrs:
    raw = _attrs(value)
    raw.update(changes)
    return _FrozenAttrs(tuple(raw.items()))


def _topology() -> DistributedTopology:
    ranks = tuple(
        TopologyRankDesc(
            rank=index,
            node_id=f"node-{index}",
            local_rank=0,
            placement=phase3.Placement("cpu"),
            role=RankRole.MODEL,
            tp_rank=index,
            pp_rank=0,
            dp_rank=0,
        )
        for index in range(2)
    )
    return DistributedTopology(ranks=ranks, tp_size=2, pp_size=1, dp_size=1)


def _request(
    name: str,
    group: ParallelGroupKind,
    *,
    ordinal: int = 0,
    window: CollectiveWindow = CollectiveWindow.AFTER_COMPUTE,
    depends_on: tuple[CollectiveKey, ...] = (),
) -> CollectiveRequestDesc:
    return CollectiveRequestDesc(
        key=CollectiveKey("phase4.oracle", name, ordinal),
        kind=CollectiveKind.ALL_REDUCE,
        group=group,
        backend=CollectiveBackend.GLOO,
        window=window,
        depends_on=depends_on,
    )


def _plan():
    return importlib.import_module("tllm.workgraph.phase4").compile_distributed_flow_plan(
        plan_epoch=12,
        topology=_topology(),
        collective_requests=(
            _request("second", ParallelGroupKind.GLOBAL, depends_on=(CollectiveKey("phase4.oracle", "first", 0),)),
            _request("first", ParallelGroupKind.TENSOR, window=CollectiveWindow.PREPARE),
        ),
    )


def _text(value: object) -> str:
    value = getattr(value, "value", value)
    if not isinstance(value, str) or not value:
        raise ValueError("expected non-empty enum/text value")
    return value


def _seq(value: object, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid {label}")
    return value


def _key(value: object) -> tuple[str, str, int]:
    return (
        _text(getattr(value, "namespace")),
        _text(getattr(value, "name")),
        _nonnegative_int(getattr(value, "ordinal"), "collective ordinal"),
    )


def _key_value(value: object) -> str:
    namespace, name, ordinal = _key(value)
    return f"{namespace}:{name}/{ordinal}"


def _oracle_validate(plan: object) -> None:
    """Closed-world Python oracle for the frozen distributed descriptor graph."""

    if _nonnegative_int(getattr(plan, "plan_epoch"), "plan epoch") < 1:
        raise ValueError("plan epoch must be positive")
    if getattr(plan, "schema_version") != 1:
        raise ValueError("unsupported schema")

    topology = getattr(plan, "topology")
    ranks = _seq(getattr(topology, "ranks"), "topology ranks")
    tp_size = _nonnegative_int(getattr(topology, "tp_size"), "tp size")
    pp_size = _nonnegative_int(getattr(topology, "pp_size"), "pp size")
    dp_size = _nonnegative_int(getattr(topology, "dp_size"), "dp size")
    if min(tp_size, pp_size, dp_size) < 1 or not ranks:
        raise ValueError("invalid topology dimensions")
    rank_ids: list[int] = []
    coordinates: set[tuple[int, int, int]] = set()
    model_count = 0
    for rank in ranks:
        rank_id = _nonnegative_int(getattr(rank, "rank"), "rank")
        rank_ids.append(rank_id)
        _text(getattr(rank, "node_id"))
        _nonnegative_int(getattr(rank, "local_rank"), "local rank")
        role = _text(getattr(rank, "role"))
        if role == RankRole.MODEL.value:
            model_count += 1
            coords = tuple(
                _nonnegative_int(getattr(rank, field), field)
                for field in ("tp_rank", "pp_rank", "dp_rank")
            )
            coordinates.add(coords)
        elif role == RankRole.HELPER.value:
            if any(getattr(rank, field) is not None for field in ("tp_rank", "pp_rank", "dp_rank")):
                raise ValueError("helper coordinates are forbidden")
        else:
            raise ValueError("unknown rank role")
    if sorted(rank_ids) != list(range(len(ranks))):
        raise ValueError("rank ids must be contiguous")
    expected_coordinates = {
        (tp, pp, dp)
        for dp in range(dp_size)
        for pp in range(pp_size)
        for tp in range(tp_size)
    }
    if model_count != tp_size * pp_size * dp_size or coordinates != expected_coordinates:
        raise ValueError("topology mesh mismatch")

    model_ranks = tuple(
        rank for rank in ranks if _text(getattr(rank, "role")) == RankRole.MODEL.value
    )

    def groups(kind: str) -> tuple[tuple[int, ...], ...]:
        if kind == ParallelGroupKind.GLOBAL.value:
            return (tuple(getattr(item, "rank") for item in model_ranks),)
        if kind == ParallelGroupKind.HELPER.value:
            helper = tuple(
                getattr(item, "rank")
                for item in ranks
                if _text(getattr(item, "role")) == RankRole.HELPER.value
            )
            return () if not helper else (helper,)
        grouped: dict[tuple[int, int], list[object]] = {}
        for rank in model_ranks:
            tp = getattr(rank, "tp_rank")
            pp = getattr(rank, "pp_rank")
            dp = getattr(rank, "dp_rank")
            if kind == ParallelGroupKind.TENSOR.value:
                group_key, order = (dp, pp), tp
            elif kind == ParallelGroupKind.PIPELINE.value:
                group_key, order = (dp, tp), pp
            elif kind == ParallelGroupKind.DATA.value:
                group_key, order = (pp, tp), dp
            else:
                raise ValueError("unknown communicator group")
            grouped.setdefault(group_key, []).append(rank)
            grouped[group_key].sort(key=lambda item: (order, getattr(item, "rank")))
        return tuple(
            tuple(getattr(item, "rank") for item in grouped[key])
            for key in sorted(grouped)
        )

    requirements = _seq(getattr(plan, "state_requirements"), "state requirements")
    requirement_keys: set[object] = set()
    for requirement in requirements:
        key = getattr(requirement, "key")
        requirement_key = (
            _text(getattr(key, "namespace")),
            _text(getattr(key, "name")),
            _nonnegative_int(getattr(key, "version"), "state version"),
        )
        if requirement_key in requirement_keys:
            raise ValueError("duplicate state requirement")
        requirement_keys.add(requirement_key)
        scope = _text(getattr(requirement, "scope"))
        if scope in {StateScope.NONE.value, StateScope.REQUEST.value}:
            raise ValueError("invalid distributed state scope")
        shard = getattr(requirement, "shard")
        shard_kind = _text(getattr(shard, "kind"))
        partitions = _nonnegative_int(getattr(shard, "partitions"), "shard partitions")
        expected = {
            "replicated": 1,
            "tensor": tp_size,
            "pipeline": pp_size,
            "data": dp_size,
        }.get(shard_kind)
        if expected is None or partitions != expected:
            raise ValueError("state shard does not match topology")
        if shard_kind == "replicated" and scope == StateScope.SHARD.value:
            raise ValueError("replicated state cannot be shard-scoped")

    residences = _seq(getattr(plan, "state_residences"), "state residences")
    for residence in residences:
        residence_key = getattr(residence, "key")
        signature = (
            _text(getattr(residence_key, "namespace")),
            _text(getattr(residence_key, "name")),
            _nonnegative_int(getattr(residence_key, "version"), "residence version"),
        )
        if signature not in requirement_keys:
            raise ValueError("residence has no requirement")
        rank = _nonnegative_int(getattr(residence, "rank"), "residence rank")
        if rank >= len(ranks):
            raise ValueError("residence rank is outside topology")
        _nonnegative_int(getattr(residence, "shard_index"), "shard index")
        _nonnegative_int(getattr(residence, "replica_index"), "replica index")

    requests = _seq(getattr(plan, "collective_requests"), "collective requests")
    request_by_key: dict[tuple[str, str, int], object] = {}
    for request in requests:
        key = _key(getattr(request, "key"))
        if key in request_by_key:
            raise ValueError("duplicate collective request")
        request_by_key[key] = request
        group = _text(getattr(request, "group"))
        group_index = _nonnegative_int(getattr(request, "group_index"), "group index")
        available_groups = groups(group)
        if group_index >= len(available_groups):
            raise ValueError("collective group is outside topology")
        kind = _text(getattr(request, "kind"))
        if kind == CollectiveKind.SEND_RECV.value and len(available_groups[group_index]) != 2:
            raise ValueError("send_recv requires two ranks")
        root = getattr(request, "root_rank")
        if root is not None and root not in available_groups[group_index]:
            raise ValueError("collective root is outside communicator")
        for dependency in _seq(getattr(request, "depends_on"), "collective dependencies"):
            if _key(dependency) not in request_by_key and _key(dependency) not in {
                _key(getattr(other, "key")) for other in requests
            }:
                raise ValueError("unknown collective dependency")

    remaining = {key: {_key(item) for item in _seq(getattr(request, "depends_on"), "dependencies")} for key, request in request_by_key.items()}
    ordered: list[object] = []
    while remaining:
        ready = sorted((key for key, deps in remaining.items() if not deps))
        if not ready:
            raise ValueError("collective dependency cycle")
        chosen = ready[0]
        ordered.append(request_by_key[chosen])
        del remaining[chosen]
        for deps in remaining.values():
            deps.discard(chosen)

    communicators = _seq(getattr(plan, "communicators"), "communicators")
    communicator_by_key: dict[tuple[str, int], object] = {}
    for communicator in communicators:
        key = getattr(communicator, "key")
        signature = (_text(getattr(key, "kind")), _nonnegative_int(getattr(key, "group_index"), "communicator group"))
        if signature in communicator_by_key:
            raise ValueError("duplicate communicator")
        communicator_by_key[signature] = communicator
        members = tuple(_nonnegative_int(rank, "communicator rank") for rank in _seq(getattr(communicator, "ranks"), "communicator ranks"))
        expected_groups = groups(signature[0])
        if signature[1] >= len(expected_groups) or members != expected_groups[signature[1]]:
            raise ValueError("communicator topology mismatch")
        _text(getattr(communicator, "backend"))

    collectives = _seq(getattr(plan, "collectives"), "collectives")
    if len(collectives) != len(ordered):
        raise ValueError("collective count mismatch")
    for sequence, (actual, request) in enumerate(zip(collectives, ordered)):
        if _key(getattr(actual, "key")) != _key(getattr(request, "key")):
            raise ValueError("collective order mismatch")
        if getattr(actual, "sequence") != sequence:
            raise ValueError("collective sequence mismatch")
        communicator = getattr(actual, "communicator")
        signature = (_text(getattr(communicator, "kind")), _nonnegative_int(getattr(communicator, "group_index"), "collective group"))
        expected_signature = (_text(getattr(request, "group")), _nonnegative_int(getattr(request, "group_index"), "request group"))
        if signature != expected_signature or signature not in communicator_by_key:
            raise ValueError("collective communicator mismatch")
        if _text(getattr(actual, "kind")) != _text(getattr(request, "kind")):
            raise ValueError("collective kind mismatch")
        if _text(getattr(actual, "window")) != _text(getattr(request, "window")):
            raise ValueError("collective window mismatch")

    schedules = _seq(getattr(plan, "local_schedules"), "local schedules")
    if len(schedules) != len(ranks):
        raise ValueError("local schedule count mismatch")
    for expected_rank, schedule in enumerate(schedules):
        if getattr(schedule, "rank") != expected_rank:
            raise ValueError("local schedule rank mismatch")
        actual_keys = tuple(_key(key) for key in _seq(getattr(schedule, "collectives"), "local schedule collectives"))
        expected_keys = tuple(
            _key(getattr(collective, "key"))
            for collective in collectives
            if expected_rank
            in tuple(
                item
                for item in _seq(
                    getattr(
                        communicator_by_key[
                            (
                                _text(getattr(getattr(collective, "communicator"), "kind")),
                                getattr(getattr(collective, "communicator"), "group_index"),
                            )
                        ],
                        "ranks",
                    ),
                    "communicator members",
                )
            )
        )
        if actual_keys != expected_keys:
            raise ValueError("local schedule does not project global order")


def _rust_outcome(test_case: unittest.TestCase, candidate: object) -> tuple[str, object | None]:
    try:
        from tllm import core

        validate = getattr(core, "validate_distributed_flow_plan")
    except (AttributeError, ImportError) as exc:
        test_case.fail("tllm.core.validate_distributed_flow_plan is required for Phase 4")
        raise AssertionError from exc
    try:
        return ("ok", validate(candidate))
    except (TypeError, ValueError, RuntimeError) as exc:
        return ("err", str(exc))


def _oracle_outcome(candidate: object) -> tuple[str, object | None]:
    try:
        _oracle_validate(candidate)
    except (TypeError, ValueError, RuntimeError) as exc:
        return ("err", str(exc))
    return ("ok", None)


def _assert_same(test_case: unittest.TestCase, candidate: object, *, expected: str) -> None:
    oracle = _oracle_outcome(candidate)
    rust = _rust_outcome(test_case, candidate)
    test_case.assertEqual(oracle[0], expected)
    test_case.assertEqual(rust[0], oracle[0])


class Phase4DistributedValidatorOracleUnitTest(unittest.TestCase):
    def test_rust_accepts_canonical_attribute_descriptors(self) -> None:
        _assert_same(self, _plan(), expected="ok")

    def test_python_and_rust_reject_malformed_collective_order(self) -> None:
        plan = _plan()
        first, second = plan.collectives
        malformed = _proxy(plan, collectives=(second, first))
        _assert_same(self, malformed, expected="err")

    def test_python_and_rust_reject_malformed_topology(self) -> None:
        plan = _plan()
        ranks = tuple(plan.topology.ranks)
        malformed_topology = _proxy(
            plan.topology,
            ranks=(_proxy(ranks[0], rank=1), ranks[1]),
        )
        _assert_same(self, _proxy(plan, topology=malformed_topology), expected="err")

    def test_python_and_rust_reject_malformed_distributed_state(self) -> None:
        plan = _plan()
        bad_requirement = _FrozenAttrs(
            (
                ("key", _FrozenAttrs((("namespace", "state"), ("name", "weights"), ("version", 1)))),
                ("scope", StateScope.REQUEST),
                ("shard", _FrozenAttrs((("kind", "replicated"), ("partitions", 1), ("axis", None)))),
                ("affinity", _FrozenAttrs((("ranks", ()), ("node_ids", ()), ("role", RankRole.MODEL)))),
            )
        )
        _assert_same(self, _proxy(plan, state_requirements=(bad_requirement,)), expected="err")

    def test_python_and_rust_reject_missing_or_wrong_communicator(self) -> None:
        plan = _plan()
        _assert_same(self, _proxy(plan, communicators=()), expected="err")
        communicator = plan.communicators[0]
        malformed = _proxy(communicator, ranks=(0,))
        _assert_same(self, _proxy(plan, communicators=(malformed, *plan.communicators[1:])), expected="err")

    def test_python_and_rust_reject_malformed_local_schedule(self) -> None:
        plan = _plan()
        schedule = plan.local_schedules[0]
        malformed = _proxy(schedule, collectives=())
        _assert_same(self, _proxy(plan, local_schedules=(malformed, *plan.local_schedules[1:])), expected="err")


if __name__ == "__main__":
    unittest.main()
