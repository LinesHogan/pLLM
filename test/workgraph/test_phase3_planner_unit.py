"""Red contracts for the Phase 3 placement planner.

These tests deliberately stay on the metadata side of the WorkGraph boundary.
They exercise implementation selection, residency routing, copy-engine timing,
and deterministic scheduling without requiring CUDA or a runtime Tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import importlib
import unittest

from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.physical import (
    FenceDependencyDesc,
    FenceKey,
    FenceKind,
    PhysicalFlowPlan,
    PhysicalNodePlan,
    StreamClass,
)
from tllm.workgraph.signature import structural_signature


def _phase3(test_case: unittest.TestCase):
    """Load Phase 3 while keeping the baseline failure explicit."""

    try:
        return importlib.import_module("tllm.workgraph.phase3")
    except (ImportError, ModuleNotFoundError) as exc:
        test_case.fail("Phase 3 module tllm.workgraph.phase3 is required")
        raise AssertionError from exc


@dataclass
class _FakeImplementation:
    implementation_id: str
    calls: list[tuple[object, ...]]

    def execute(self, *args: object) -> None:
        self.calls.append(args)


@dataclass
class _SemanticImplementation:
    implementation_id: str

    def execute(self, *args: object) -> object:
        return tuple(int(value) + 1 for value in args)


def _physical_plan() -> tuple[PhysicalFlowPlan, ExecutorKey, ExecutorKey]:
    source_key = NodeKey.producer("phase3.source")
    target_key = NodeKey.consumer_flow("phase3.target", 0)
    source_executor = ExecutorKey("logical", "phase3.source")
    target_executor = ExecutorKey("logical", "phase3.target")
    dependency = FenceDependencyDesc(
        fence=FenceKey("phase3", "source-ready"),
        producer=source_key,
        consumer=target_key,
        kind=FenceKind.READY,
    )
    return (
        PhysicalFlowPlan(
            plan_epoch=1,
            nodes=(
                PhysicalNodePlan(
                    node_key=source_key,
                    executor=source_executor,
                    stream=StreamClass.DEFAULT,
                ),
                PhysicalNodePlan(
                    node_key=target_key,
                    executor=target_executor,
                    stream=StreamClass.SIDE_TRAIN,
                    fences=(dependency,),
                ),
            ),
        ),
        source_executor,
        target_executor,
    )


def _implementation(
    phase3: object,
    *,
    key: ExecutorKey,
    logical_executor: ExecutorKey,
    placement: object,
    duration: int,
) -> object:
    return phase3.ImplementationSpec(  # type: ignore[attr-defined]
        key=key,
        logical_executor=logical_executor,
        placement=placement,
        implementation=_FakeImplementation(str(key), []),
        duration=phase3.DurationProfile(duration),  # type: ignore[attr-defined]
    )


class Phase3PlacementPlanTest(unittest.TestCase):
    def test_frozen_plan_selects_typed_implementations_from_dag_deterministically(self) -> None:
        phase3 = _phase3(self)
        physical, source_executor, target_executor = _physical_plan()
        source_cpu = ExecutorKey("implementation", "phase3.source.cpu")
        source_gpu = ExecutorKey("implementation", "phase3.source.cuda0")
        target_cpu = ExecutorKey("implementation", "phase3.target.cpu")

        specs = (
            _implementation(
                phase3,
                key=source_gpu,
                logical_executor=source_executor,
                placement=phase3.Placement("cuda", 0),
                duration=4,
            ),
            _implementation(
                phase3,
                key=target_cpu,
                logical_executor=target_executor,
                placement=phase3.Placement("cpu"),
                duration=3,
            ),
            _implementation(
                phase3,
                key=source_cpu,
                logical_executor=source_executor,
                placement=phase3.Placement("cpu"),
                duration=40,
            ),
        )
        first_registry = phase3.ImplementationRegistry()
        for spec in specs:
            first_registry.register(spec)
        second_registry = phase3.ImplementationRegistry()
        for spec in reversed(specs):
            second_registry.register(spec)

        topology = phase3.DeviceTopology(peer_access=frozenset())
        first = phase3.compile_placement_plan(  # type: ignore[attr-defined]
            physical,
            registry=first_registry,
            topology=topology,
        )
        second = phase3.compile_placement_plan(  # type: ignore[attr-defined]
            physical,
            registry=second_registry,
            topology=topology,
        )

        self.assertIsInstance(first, phase3.PlacementPlan)
        self.assertEqual(first, second)
        self.assertTrue(dataclasses.is_dataclass(first))
        self.assertTrue(getattr(type(first), "__dataclass_params__").frozen)
        self.assertEqual(
            tuple(selection.key for selection in first.selections),
            (source_gpu, target_cpu),
        )
        self.assertEqual(
            tuple(item.node_key for item in first.schedule),
            (physical.nodes[0].node_key, physical.nodes[1].node_key),
        )
        self.assertFalse(any(hasattr(selection, "implementation") for selection in first.selections))

    def test_critical_path_is_prioritized_after_dependency_readiness(self) -> None:
        phase3 = _phase3(self)
        candidates = (
            phase3.ScheduleCandidate("a.wide", priority=0, ready_at=0, duration=7),
            phase3.ScheduleCandidate("z.root", priority=0, ready_at=0, duration=2),
            phase3.ScheduleCandidate(
                "z.tail",
                priority=0,
                ready_at=0,
                duration=9,
                depends_on=("z.root",),
            ),
        )

        schedule = phase3.schedule_candidates(candidates)

        self.assertEqual(
            tuple(item.key for item in schedule),
            ("z.root", "z.tail", "a.wide"),
        )

    def test_cpu_and_gpu_registered_implementations_share_a_semantic_oracle(self) -> None:
        phase3 = _phase3(self)
        logical = ExecutorKey("logical", "phase3.semantic")
        cpu_key = ExecutorKey("implementation", "phase3.semantic.cpu")
        gpu_key = ExecutorKey("implementation", "phase3.semantic.cuda0")
        registry = phase3.ImplementationRegistry()
        registry.register(
            phase3.ImplementationSpec(
                cpu_key,
                phase3.Placement("cpu"),
                _SemanticImplementation("cpu"),
                logical_executor=logical,
            )
        )
        registry.register(
            phase3.ImplementationSpec(
                gpu_key,
                phase3.Placement("cuda", 0),
                _SemanticImplementation("cuda0"),
                logical_executor=logical,
            )
        )

        cpu_result = registry.resolve(cpu_key).execute(1, 4, 9)
        gpu_result = registry.resolve(gpu_key).execute(1, 4, 9)
        self.assertEqual(cpu_result, gpu_result)


class Phase3TransferSchedulingTest(unittest.TestCase):
    def _request(self, phase3: object, source: object, destination: object, *, bytes_: int = 64):
        value = phase3.ValueKey("hidden", 1)  # type: ignore[attr-defined]
        return phase3.TransferRequest(  # type: ignore[attr-defined]
            value=value,
            source=phase3.Residency(value, source),  # type: ignore[attr-defined]
            destination=phase3.Residency(value, destination),  # type: ignore[attr-defined]
            byte_count=bytes_,
        )

    def test_scheduled_moves_preserve_typed_residencies_and_serialize_one_engine(self) -> None:
        phase3 = _phase3(self)
        cpu = phase3.Placement("cpu")
        gpu = phase3.Placement("cuda", 0)
        requests = (
            self._request(phase3, cpu, gpu),
            self._request(phase3, gpu, cpu),
            self._request(phase3, cpu, gpu),
        )
        moves = phase3.compile_moves(  # type: ignore[attr-defined]
            requests,
            topology=phase3.DeviceTopology(peer_access=frozenset()),
        )
        profile = phase3.TransferProfile(  # type: ignore[attr-defined]
            h2d=phase3.DurationProfile(7),  # type: ignore[attr-defined]
            d2h=phase3.DurationProfile(11),  # type: ignore[attr-defined]
            p2p=phase3.DurationProfile(3),  # type: ignore[attr-defined]
        )

        scheduled = phase3.schedule_moves(  # type: ignore[attr-defined]
            moves,
            profile=profile,
            start_at=10,
        )

        self.assertTrue(all(isinstance(item, phase3.ScheduledMove) for item in scheduled))
        self.assertEqual(tuple(item.move for item in scheduled), moves)
        self.assertEqual(tuple(item.engine for item in scheduled), (
            phase3.CopyEngine.H2D,
            phase3.CopyEngine.D2H,
            phase3.CopyEngine.H2D,
        ))
        self.assertEqual((scheduled[0].start_at, scheduled[0].end_at), (10, 17))
        self.assertEqual((scheduled[1].start_at, scheduled[1].end_at), (10, 21))
        self.assertEqual((scheduled[2].start_at, scheduled[2].end_at), (17, 24))
        self.assertIsInstance(scheduled[0].move.source, phase3.Residency)
        self.assertIsInstance(scheduled[0].move.destination, phase3.Residency)

    def test_missing_profile_uses_a_conservative_typed_fallback(self) -> None:
        phase3 = _phase3(self)
        request = self._request(
            phase3,
            phase3.Placement("cpu"),
            phase3.Placement("cuda", 0),
        )
        move = phase3.compile_moves(  # type: ignore[attr-defined]
            (request,),
            topology=phase3.DeviceTopology(peer_access=frozenset()),
        )[0]
        known = phase3.schedule_moves(  # type: ignore[attr-defined]
            (move,),
            profile=phase3.TransferProfile(h2d=phase3.DurationProfile(8)),  # type: ignore[attr-defined]
            start_at=0,
        )[0]
        missing = phase3.schedule_moves(  # type: ignore[attr-defined]
            (move,),
            profile=phase3.TransferProfile(),  # type: ignore[attr-defined]
            start_at=0,
        )[0]

        self.assertGreaterEqual(missing.end_at, known.end_at)
        self.assertTrue(missing.used_fallback)

    def test_invalid_topology_and_transfer_profiles_fail_closed(self) -> None:
        phase3 = _phase3(self)
        with self.assertRaises(ValueError):
            phase3.DeviceTopology(peer_access=frozenset({(0, 0)}))
        with self.assertRaises(ValueError):
            phase3.DeviceTopology(peer_access=frozenset({(-1, 1)}))
        with self.assertRaises(ValueError):
            phase3.DeviceTopology(peer_access=frozenset(), cuda_devices=(0, 0))
        with self.assertRaises(ValueError):
            phase3.DurationProfile(0)
        with self.assertRaises((TypeError, ValueError)):
            phase3.TransferProfile(h2d=object())  # type: ignore[arg-type]

    def test_route_specific_transfer_profile_drives_placement_move_duration(self) -> None:
        phase3 = _phase3(self)
        physical, source_executor, target_executor = _physical_plan()
        registry = phase3.ImplementationRegistry()
        registry.register(
            _implementation(
                phase3,
                key=ExecutorKey("implementation", "phase3.profile.source.cpu"),
                logical_executor=source_executor,
                placement=phase3.Placement("cpu"),
                duration=2,
            )
        )
        registry.register(
            _implementation(
                phase3,
                key=ExecutorKey("implementation", "phase3.profile.target.cuda"),
                logical_executor=target_executor,
                placement=phase3.Placement("cuda", 0),
                duration=3,
            )
        )
        value = phase3.ValueKey("profiled", 1)
        plan = phase3.compile_placement_plan(
            physical,
            registry=registry,
            topology=phase3.DeviceTopology(peer_access=frozenset()),
            value_flows=(
                phase3.ValueFlowDesc(
                    value,
                    physical.nodes[0].node_key,
                    physical.nodes[1].node_key,
                    64,
                ),
            ),
            transfer_profiles=(
                phase3.TransferProfileDesc(
                    phase3.MoveKind.H2D,
                    phase3.Placement("cpu"),
                    phase3.Placement("cuda", 0),
                    bytes_per_microsecond=64,
                    fixed_latency_microseconds=2,
                    samples=5,
                ),
            ),
        )

        self.assertEqual(len(plan.moves), 1)
        self.assertEqual(plan.moves[0].end_at - plan.moves[0].start_at, 3)
        self.assertFalse(plan.moves[0].used_fallback)


class Phase3SignatureTest(unittest.TestCase):
    def test_structural_signature_changes_when_selected_placement_changes(self) -> None:
        phase3 = _phase3(self)
        physical, source_executor, target_executor = _physical_plan()
        registry = phase3.ImplementationRegistry()
        source_cpu = ExecutorKey("implementation", "phase3.signature.cpu")
        source_gpu = ExecutorKey("implementation", "phase3.signature.cuda0")
        target_cpu = ExecutorKey("implementation", "phase3.signature.target.cpu")
        registry.register(
            _implementation(
                phase3,
                key=source_cpu,
                logical_executor=source_executor,
                placement=phase3.Placement("cpu"),
                duration=2,
            )
        )
        registry.register(
            _implementation(
                phase3,
                key=source_gpu,
                logical_executor=source_executor,
                placement=phase3.Placement("cuda", 0),
                duration=3,
            )
        )
        registry.register(
            _implementation(
                phase3,
                key=target_cpu,
                logical_executor=target_executor,
                placement=phase3.Placement("cpu"),
                duration=1,
            )
        )

        cpu_plan = phase3.compile_placement_plan(  # type: ignore[attr-defined]
            physical,
            registry=registry,
            topology=phase3.DeviceTopology(peer_access=frozenset()),
            placement_overrides={source_executor: source_cpu},
        )
        gpu_plan = phase3.compile_placement_plan(  # type: ignore[attr-defined]
            physical,
            registry=registry,
            topology=phase3.DeviceTopology(peer_access=frozenset()),
            placement_overrides={source_executor: source_gpu},
        )

        self.assertNotEqual(structural_signature(cpu_plan), structural_signature(gpu_plan))


if __name__ == "__main__":
    unittest.main()
