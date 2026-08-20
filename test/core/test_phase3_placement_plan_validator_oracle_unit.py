"""Red differential contracts for the Phase 3 placement-plan validator.

The compiler tests use real :class:`PlacementPlan` instances.  Rejected plans
use frozen attribute-only proxies so malformed fields cross the Python/Rust
boundary instead of being rejected by the local dataclass constructors first.
No Tensor or CUDA runtime is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import importlib
import unittest

from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph import phase3
from tllm.workgraph.physical import (
    FenceDependencyDesc,
    FenceKey,
    FenceKind,
    PhysicalFlowPlan,
    PhysicalNodePlan,
    StreamClass,
)


@dataclass(frozen=True, slots=True)
class _FrozenAttrs:
    """Frozen structural proxy accepted by attribute-based PyO3 descriptors."""

    entries: tuple[tuple[str, object], ...]

    def __getattr__(self, name: str) -> object:
        for key, value in self.entries:
            if key == name:
                return value
        raise AttributeError(name)


def _values(obj: object) -> dict[str, object]:
    if isinstance(obj, _FrozenAttrs):
        return dict(obj.entries)
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def _proxy(obj: object, **changes: object) -> _FrozenAttrs:
    raw = _values(obj)
    raw.update(changes)
    return _FrozenAttrs(tuple(raw.items()))


@dataclass
class _FakeImplementation:
    implementation_id: str

    def execute(self, *args: object) -> None:
        del args


def _phase3(test_case: unittest.TestCase):
    try:
        return importlib.import_module("tllm.workgraph.phase3")
    except (ImportError, ModuleNotFoundError) as exc:
        test_case.fail("tllm.workgraph.phase3 is required for Phase 3 placement tests")
        raise AssertionError from exc


def _physical_plan() -> tuple[PhysicalFlowPlan, NodeKey, NodeKey, ExecutorKey, ExecutorKey]:
    source_node = NodeKey.producer("phase3.validator.source")
    target_node = NodeKey.consumer_flow("phase3.validator.target", 0)
    source_executor = ExecutorKey("logical", "phase3.validator.source")
    target_executor = ExecutorKey("logical", "phase3.validator.target")
    dependency = FenceDependencyDesc(
        fence=FenceKey("phase3", "source-ready"),
        producer=source_node,
        consumer=target_node,
        kind=FenceKind.READY,
    )
    physical = PhysicalFlowPlan(
        plan_epoch=7,
        nodes=(
            PhysicalNodePlan(
                node_key=source_node,
                executor=source_executor,
                stream=StreamClass.DEFAULT,
            ),
            PhysicalNodePlan(
                node_key=target_node,
                executor=target_executor,
                stream=StreamClass.SIDE_TRAIN,
                fences=(dependency,),
            ),
        ),
    )
    return physical, source_node, target_node, source_executor, target_executor


def _registry(
    phase3_module: object,
    *,
    source: object,
    destination: object,
):
    _physical, _source_node, _target_node, source_executor, target_executor = _physical_plan()
    registry = phase3_module.ImplementationRegistry()
    registry.register(
        phase3_module.ImplementationSpec(
            key=ExecutorKey("implementation", "phase3.validator.source"),
            logical_executor=source_executor,
            placement=source,
            implementation=_FakeImplementation("source"),
            duration=phase3_module.DurationProfile(3),
        )
    )
    registry.register(
        phase3_module.ImplementationSpec(
            key=ExecutorKey("implementation", "phase3.validator.target"),
            logical_executor=target_executor,
            placement=destination,
            implementation=_FakeImplementation("target"),
            duration=phase3_module.DurationProfile(4),
        )
    )
    return registry


def _compile(
    test_case: unittest.TestCase,
    *,
    source: object,
    destination: object,
    peer_access: frozenset[tuple[int, int]] = frozenset(),
    topology: object | None = None,
) -> object:
    phase3_module = _phase3(test_case)
    physical, source_node, target_node, _source_executor, _target_executor = _physical_plan()
    value = phase3_module.ValueKey("hidden", 1)
    compiled_topology = topology
    if compiled_topology is None:
        compiled_topology = phase3_module.DeviceTopology(
            peer_access=peer_access, cuda_devices=(0, 1)
        )
    return phase3_module.compile_placement_plan(
        physical,
        registry=_registry(phase3_module, source=source, destination=destination),
        topology=compiled_topology,
        value_flows=(
            phase3_module.ValueFlowDesc(
                value=value,
                producer=source_node,
                consumer=target_node,
                byte_count=64,
            ),
        ),
    )


def _validator(test_case: unittest.TestCase):
    try:
        core = importlib.import_module("tllm.core")
        return core.validate_placement_plan
    except (AttributeError, ImportError) as exc:
        test_case.fail("tllm.core.validate_placement_plan is required for Phase 3")
        raise AssertionError from exc


def _assert_rejected(test_case: unittest.TestCase, plan: object) -> None:
    validate = _validator(test_case)
    with test_case.assertRaises((TypeError, ValueError, RuntimeError)):
        validate(plan)


class Phase3PlacementPlanValidatorTest(unittest.TestCase):
    def test_python_and_rust_reject_runtime_text_in_node_keys(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        original = plan.selections[0].node_key
        runtime_key = _FrozenAttrs(
            (
                ("namespace", "runtime"),
                ("owner", "<Tensor object at 0xdeadbeef>_runtime_node"),
                ("ordinal", original.ordinal),
            )
        )
        runtime_selection = _proxy(plan.selections[0], node_key=runtime_key)
        runtime_schedule = _proxy(plan.schedule[0], node_key=runtime_key)
        proxy = _proxy(
            plan,
            selections=(runtime_selection, plan.selections[1]),
            schedule=(runtime_schedule, plan.schedule[1]),
        )
        with self.assertRaises((TypeError, ValueError)):
            replace(
                plan,
                selections=(runtime_selection, plan.selections[1]),
                schedule=(runtime_schedule, plan.schedule[1]),
            )
        _assert_rejected(self, proxy)

    def test_python_and_rust_reject_unstable_value_key_separators(self) -> None:
        phase3 = _phase3(self)
        with self.assertRaises(ValueError):
            phase3.ValueKey("value/with/slash", 1)
        with self.assertRaises(ValueError):
            phase3.ValueKey("value\\with\\slash", 1)
        with self.assertRaises(ValueError):
            phase3.ValueKey("hidden0Xdeadbeef", 1)

    def test_python_and_rust_reject_route_endpoints_that_differ_from_nodes(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        scheduled = plan.moves[0]
        wrong_destination = phase3.Residency(
            scheduled.move.value,
            phase3.Placement("cuda", 1),
        )
        wrong_move = replace(scheduled.move, destination=wrong_destination)
        malformed = replace(scheduled, move=wrong_move)
        proxy = _proxy(
            plan,
            residencies=(*plan.residencies, wrong_destination),
            moves=(malformed,),
            leases=(
                *plan.leases,
                phase3.PlacementLeaseDesc(
                    wrong_destination,
                    scheduled.start_at,
                    scheduled.end_at,
                ),
            ),
        )
        with self.assertRaises((TypeError, ValueError)):
            replace(
                plan,
                residencies=(*plan.residencies, wrong_destination),
                moves=(malformed,),
                leases=(
                    *plan.leases,
                    phase3.PlacementLeaseDesc(
                        wrong_destination,
                        scheduled.start_at,
                        scheduled.end_at,
                    ),
                ),
            )
        _assert_rejected(self, proxy)

    def test_rust_rejects_mixed_oracle_keys_before_cross_placement_selection(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        logical = ExecutorKey("logical", "phase3.validator.shared")
        oracle = phase3.SemanticOracleKey("oracle", "phase3.validator.shared")
        source = _proxy(
            plan.selections[0],
            logical_executor=logical,
            oracle_key=oracle,
        )
        middle_node = NodeKey.consumer_flow("phase3.validator.middle", 0)
        middle_key = ExecutorKey("implementation", "phase3.validator.middle")
        middle = phase3.ImplementationSelectionDesc(
            node_key=middle_node,
            logical_executor=logical,
            key=middle_key,
            placement=phase3.Placement("cpu"),
            duration=phase3.DurationProfile(1),
            oracle_key=None,
        )
        target = _proxy(
            plan.selections[1],
            logical_executor=logical,
            oracle_key=oracle,
        )
        middle_schedule = phase3.PlacedNodeDesc(
            node_key=middle_node,
            implementation=middle_key,
            placement=phase3.Placement("cpu"),
            start_at=0,
            end_at=1,
        )
        _assert_rejected(
            self,
            _proxy(
                plan,
                selections=(source, middle, target),
                schedule=(plan.schedule[0], middle_schedule, plan.schedule[1]),
            ),
        )

    def test_validator_accepts_real_compiled_placement_plan(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        result = _validator(self)(plan)
        self.assertIsNotNone(result)

    def test_compiled_plan_owns_a_frozen_topology_snapshot(self) -> None:
        phase3 = _phase3(self)
        source_topology = phase3.DeviceTopology(
            peer_access=frozenset({(0, 1)}),
            cuda_devices=(0, 1),
        )
        plan = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
            topology=source_topology,
        )

        object.__setattr__(source_topology, "cuda_devices", (99,))
        object.__setattr__(source_topology, "peer_access", frozenset())

        self.assertEqual(plan.topology.cuda_devices, (0, 1))
        self.assertEqual(plan.topology.peer_access, frozenset({(0, 1)}))

    def test_validator_rejects_schema_and_epoch_corruption(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        _assert_rejected(self, _proxy(plan, plan_epoch=0))
        _assert_rejected(self, _proxy(plan, schema_version=0))
        _assert_rejected(self, _proxy(plan, schema_version=2))

    def test_validator_rejects_placement_and_memory_corruption(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        residency = plan.residencies[0]
        invalid_placement = _proxy(
            residency.placement,
            kind="xpu",
            index=0,
        )
        invalid_residency = _proxy(residency, placement=invalid_placement)
        residencies = (invalid_residency, *plan.residencies[1:])
        _assert_rejected(self, _proxy(plan, residencies=residencies))

        cuda = next(item for item in plan.residencies if item.placement.kind.value == "cuda")
        bad_memory = _proxy(cuda, memory="host")
        bad_residencies = tuple(bad_memory if item == cuda else item for item in plan.residencies)
        _assert_rejected(self, _proxy(plan, residencies=bad_residencies))

    def test_validator_rejects_selection_schedule_mismatch_and_bad_order(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        selection = plan.selections[0]
        bad_selection = _proxy(
            selection,
            key=ExecutorKey("implementation", "phase3.validator.missing"),
        )
        _assert_rejected(
            self,
            _proxy(plan, selections=(bad_selection, *plan.selections[1:])),
        )

        _assert_rejected(self, _proxy(plan, schedule=tuple(reversed(plan.schedule))))

    def test_python_and_rust_reject_schedule_duration_mismatch(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        selected = plan.selections[1]
        scheduled = plan.schedule[1]
        malformed_schedule = _proxy(
            scheduled,
            end_at=scheduled.start_at + selected.duration.microseconds + 1,
        )
        malformed_plan = _proxy(
            plan,
            schedule=(plan.schedule[0], malformed_schedule),
        )

        with self.assertRaises(ValueError):
            replace(
                plan,
                schedule=(
                    plan.schedule[0],
                    replace(
                        scheduled,
                        end_at=scheduled.start_at + selected.duration.microseconds + 1,
                    ),
                ),
            )
        _assert_rejected(self, malformed_plan)

    def test_validator_rejects_move_route_engine_and_dependency_order(self) -> None:
        phase3 = _phase3(self)
        staged = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
        )
        self.assertEqual(
            tuple(item.move.kind for item in staged.moves),
            (phase3.MoveKind.D2H, phase3.MoveKind.H2D),
        )
        first, second = staged.moves
        bad_move = _proxy(
            first.move,
            kind=phase3.MoveKind.H2D,
        )
        _assert_rejected(
            self,
            _proxy(
                staged,
                moves=(_proxy(first, move=bad_move), second),
            ),
        )

        bad_engine = _proxy(first, engine=phase3.CopyEngine.H2D)
        _assert_rejected(
            self,
            _proxy(staged, moves=(bad_engine, second)),
        )

        _assert_rejected(self, _proxy(staged, moves=tuple(reversed(staged.moves))))

        consumer = next(
            item for item in staged.schedule if item.node_key == second.consumer
        )
        late_second = _proxy(second, end_at=consumer.start_at + 1)
        _assert_rejected(self, _proxy(staged, moves=(first, late_second)))

    def test_python_and_rust_reject_synchronous_scheduled_moves(self) -> None:
        phase3 = _phase3(self)
        staged = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
        )
        first, second = staged.moves
        with self.assertRaises(ValueError):
            replace(first.move, asynchronous=False)
        synchronous_move = _proxy(first.move, asynchronous=False)
        malformed_first = _proxy(first, move=synchronous_move)
        malformed_plan = _proxy(
            staged,
            moves=(malformed_first, second),
        )

        _assert_rejected(self, malformed_plan)

    def test_validator_rejects_unknown_lease_residency_and_invalid_lifetime(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        self.assertTrue(plan.leases)
        lease = plan.leases[0]
        unknown_value = phase3.ValueKey("missing", 1)
        unknown_residency = _proxy(lease.residency, value=unknown_value)
        _assert_rejected(
            self,
            _proxy(plan, leases=(_proxy(lease, residency=unknown_residency), *plan.leases[1:])),
        )
        _assert_rejected(self, _proxy(plan, leases=(_proxy(lease, live_end=0), *plan.leases[1:])))

    def test_python_and_rust_reject_the_same_missing_lease(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        self.assertGreaterEqual(len(plan.leases), 2)
        missing_lease = plan.leases[:-1]

        with self.assertRaises(ValueError):
            replace(plan, leases=missing_lease)
        _assert_rejected(self, _proxy(plan, leases=missing_lease))

    def test_validator_rejects_copy_engine_capacity_corruption(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        capacity = plan.copy_engines[0]
        _assert_rejected(
            self,
            _proxy(
                plan,
                copy_engines=(
                    _proxy(capacity, capacity=0),
                    *plan.copy_engines[1:],
                ),
            ),
        )
        _assert_rejected(
            self,
            _proxy(plan, copy_engines=(capacity, capacity)),
        )

    def test_validator_rejects_runtime_repr_in_structural_identity(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        selection = plan.selections[0]
        runtime_key = _proxy(
            selection.key,
            name="<Tensor object at 0xdeadbeef>",
        )
        runtime_selection = _proxy(selection, key=runtime_key)
        runtime_schedule = _proxy(plan.schedule[0], implementation=runtime_key)
        _assert_rejected(
            self,
            _proxy(
                plan,
                selections=(runtime_selection, *plan.selections[1:]),
                schedule=(runtime_schedule, *plan.schedule[1:]),
            ),
        )

    def test_python_and_rust_reject_placements_outside_frozen_topology(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        cuda99 = phase3.Placement("cuda", 99)

        bad_selection = replace(plan.selections[1], placement=cuda99)
        bad_schedule = replace(plan.schedule[1], placement=cuda99)
        with self.assertRaises(ValueError):
            replace(
                plan,
                selections=(plan.selections[0], bad_selection),
                schedule=(plan.schedule[0], bad_schedule),
            )
        _assert_rejected(
            self,
            _proxy(
                plan,
                selections=(
                    plan.selections[0],
                    _proxy(plan.selections[1], placement=cuda99),
                ),
                schedule=(
                    plan.schedule[0],
                    _proxy(plan.schedule[1], placement=cuda99),
                ),
            ),
        )

        destination = next(
            item
            for item in plan.residencies
            if item.placement == phase3.Placement("cuda", 0)
        )
        bad_residency = phase3.Residency(destination.value, cuda99)
        bad_residencies = tuple(
            bad_residency if item == destination else item for item in plan.residencies
        )
        scheduled = plan.moves[0]
        bad_move = replace(scheduled.move, destination=bad_residency)
        bad_scheduled = replace(scheduled, move=bad_move)
        bad_moves = (bad_scheduled, *plan.moves[1:])
        bad_leases = tuple(
            replace(lease, residency=bad_residency)
            if lease.residency == destination
            else lease
            for lease in plan.leases
        )
        with self.assertRaises(ValueError):
            replace(
                plan,
                residencies=bad_residencies,
                moves=bad_moves,
                leases=bad_leases,
            )
        _assert_rejected(
            self,
            _proxy(
                plan,
                residencies=bad_residencies,
                moves=tuple(
                    _proxy(item, move=bad_move) if item is scheduled else item
                    for item in plan.moves
                ),
                leases=bad_leases,
            ),
        )

    def test_python_and_rust_reject_reverse_directional_peer_edge(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
            peer_access=frozenset({(0, 1)}),
        )
        self.assertEqual(
            tuple(item.move.kind for item in plan.moves),
            (phase3.MoveKind.P2P,),
        )
        reverse_only = phase3.DeviceTopology(
            peer_access=frozenset({(1, 0)}),
            cuda_devices=(0, 1),
        )
        with self.assertRaises(ValueError):
            replace(plan, topology=reverse_only)
        _assert_rejected(self, _proxy(plan, topology=reverse_only))

    def test_python_replace_recursively_rejects_mutated_move_contract(self) -> None:
        phase3 = _phase3(self)

        synchronous = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
        )
        object.__setattr__(synchronous.moves[0].move, "asynchronous", False)
        with self.assertRaises(ValueError):
            replace(synchronous)
        _assert_rejected(self, synchronous)

        wrong_kind = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
        )
        object.__setattr__(wrong_kind.moves[0].move, "kind", phase3.MoveKind.H2D)
        with self.assertRaises(ValueError):
            replace(wrong_kind)
        _assert_rejected(self, wrong_kind)

        wrong_endpoint = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
        )
        object.__setattr__(
            wrong_endpoint.moves[0].move,
            "destination",
            phase3.Residency(
                wrong_endpoint.moves[0].move.value,
                phase3.Placement("cuda", 99),
            ),
        )
        with self.assertRaises(ValueError):
            replace(wrong_endpoint)
        _assert_rejected(self, wrong_endpoint)


class Phase3PlacementPlanCompilerRouteTest(unittest.TestCase):
    def test_compile_placement_plan_same_gpu_has_no_move(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 0),
        )
        self.assertEqual(plan.moves, ())
        self.assertIsNotNone(_validator(self)(plan))

    def test_compile_placement_plan_cpu_fallback_contains_h2d_route(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cpu"),
            destination=phase3.Placement("cuda", 0),
        )
        self.assertEqual(tuple(item.move.kind for item in plan.moves), (phase3.MoveKind.H2D,))
        self.assertEqual(tuple(item.engine for item in plan.moves), (phase3.CopyEngine.H2D,))

    def test_compile_placement_plan_peer_gpu_contains_one_p2p_move(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
            peer_access=frozenset({(0, 1)}),
        )
        self.assertEqual(tuple(item.move.kind for item in plan.moves), (phase3.MoveKind.P2P,))
        self.assertEqual(tuple(item.engine for item in plan.moves), (phase3.CopyEngine.P2P,))

    def test_compile_placement_plan_without_peer_uses_ordered_staged_route(self) -> None:
        phase3 = _phase3(self)
        plan = _compile(
            self,
            source=phase3.Placement("cuda", 0),
            destination=phase3.Placement("cuda", 1),
        )
        self.assertEqual(
            tuple(item.move.kind for item in plan.moves),
            (phase3.MoveKind.D2H, phase3.MoveKind.H2D),
        )
        self.assertEqual(
            plan.moves[0].move.destination.memory,
            phase3.MemoryKind.PINNED_HOST,
        )
        self.assertEqual(plan.moves[1].depends_on, (plan.moves[0].key,))
        self.assertTrue(all(item.producer == plan.schedule[0].node_key for item in plan.moves))
        self.assertTrue(all(item.consumer == plan.schedule[1].node_key for item in plan.moves))
        self.assertLessEqual(plan.moves[-1].end_at, plan.schedule[1].start_at)


if __name__ == "__main__":
    unittest.main()
