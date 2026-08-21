"""Red tests for the Phase 3 WorkGraph placement/execution contracts.

Phase 1/2 deliberately stop at a single-device physical plan.  These tests
specify the next boundary in terms of small, typed values so they can run on
CPU-only builders.  The test module imports the planned Phase 3 surface from
inside each test: on the Phase 2 baseline this is an explicit red failure,
instead of making collection itself fail before the individual contracts are
reported.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import unittest

from tllm.workgraph.ir import ExecutorKey


def _phase3(test_case: unittest.TestCase):
    """Load the future Phase 3 module and turn a missing surface into a red test."""

    try:
        return importlib.import_module("tllm.workgraph.placement")
    except (ImportError, ModuleNotFoundError) as exc:
        test_case.fail(
            "Phase 3 WorkGraph contracts are not implemented: "
            "tllm.workgraph.placement is required"
        )
        raise AssertionError from exc


@dataclass
class _FakeImplementation:
    """Typed runtime implementation used by the registry test."""

    implementation_id: str
    calls: list[tuple[object, ...]]

    def execute(self, *args: object) -> None:
        self.calls.append(args)


class _FakeCopyRuntime:
    """Semantic oracle for transfer selection; it has no CUDA dependency."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def copy(self, kind: str, source: str, destination: str) -> None:
        self.calls.append((kind, source, destination))


class Phase3ImplementationRegistryTest(unittest.TestCase):
    def test_registry_rejects_ambiguous_executor_key_separators(self) -> None:
        phase3 = _phase3(self)
        registry = phase3.ImplementationRegistry()
        with self.assertRaises(ValueError):
            registry.register(
                phase3.ImplementationSpec(
                    ExecutorKey("implementation:ambiguous", "candidate"),
                    phase3.Placement("cpu"),
                    _FakeImplementation("ambiguous", []),
                )
            )

    def test_registry_is_typed_and_rejects_duplicate_executor_keys(self) -> None:
        phase3 = _phase3(self)
        registry = phase3.ImplementationRegistry()
        key = ExecutorKey("region", "residual_copy")
        implementation = _FakeImplementation("residual_copy.cpu", [])
        spec = phase3.ImplementationSpec(
            key=key,
            placement=phase3.Placement("cpu"),
            implementation=implementation,
        )

        registry.register(spec)
        self.assertIs(registry.resolve(key), implementation)
        self.assertEqual(registry.keys(), (key,))

        with self.assertRaises((TypeError, ValueError)):
            registry.register(object())
        with self.assertRaises((KeyError, RuntimeError, ValueError)):
            registry.register(spec)

    def test_registry_does_not_resolve_by_untyped_string_or_runtime_repr(self) -> None:
        phase3 = _phase3(self)
        registry = phase3.ImplementationRegistry()
        key = ExecutorKey("region", "residual_copy")
        registry.register(
            phase3.ImplementationSpec(
                key=key,
                placement=phase3.Placement("cpu"),
                implementation=_FakeImplementation("residual_copy.cpu", []),
            )
        )

        with self.assertRaises((TypeError, KeyError, ValueError)):
            registry.resolve("region/residual_copy")
        with self.assertRaises((TypeError, KeyError, ValueError)):
            registry.resolve("<ExecutorKey at 0xdeadbeef>")


class Phase3PlacementResidencyTest(unittest.TestCase):
    def _request(self, phase3, *, source: object, destination: object):
        value = phase3.ValueKey("hidden", 1)
        return phase3.TransferRequest(
            value=value,
            source=phase3.Residency(value, source),
            destination=phase3.Residency(value, destination),
        )

    def test_same_gpu_residency_requires_no_move(self) -> None:
        phase3 = _phase3(self)
        gpu0 = phase3.Placement("cuda", 0)
        request = self._request(phase3, source=gpu0, destination=gpu0)

        moves = phase3.compile_moves(
            (request,),
            topology=phase3.DeviceTopology(
                peer_access=frozenset(), cuda_devices=(0,)
            ),
        )
        self.assertEqual(moves, ())

    def test_cpu_gpu_boundary_selects_directional_h2d_and_d2h_moves(self) -> None:
        phase3 = _phase3(self)
        cpu = phase3.Placement("cpu")
        gpu0 = phase3.Placement("cuda", 0)
        topology = phase3.DeviceTopology(
            peer_access=frozenset(), cuda_devices=(0,)
        )

        h2d = phase3.compile_moves(
            (self._request(phase3, source=cpu, destination=gpu0),),
            topology=topology,
        )
        d2h = phase3.compile_moves(
            (self._request(phase3, source=gpu0, destination=cpu),),
            topology=topology,
        )
        self.assertEqual(tuple(move.kind.value for move in h2d), ("h2d",))
        self.assertEqual(tuple(move.kind.value for move in d2h), ("d2h",))

    def test_peer_gpu_residency_uses_one_p2p_typed_move(self) -> None:
        phase3 = _phase3(self)
        gpu0 = phase3.Placement("cuda", 0)
        gpu1 = phase3.Placement("cuda", 1)
        request = self._request(phase3, source=gpu0, destination=gpu1)

        moves = phase3.compile_moves(
            (request,),
            topology=phase3.DeviceTopology(peer_access=frozenset({(0, 1)})),
        )
        self.assertEqual(tuple(move.kind.value for move in moves), ("p2p",))
        self.assertEqual(moves[0].source.placement, gpu0)
        self.assertEqual(moves[0].destination.placement, gpu1)

    def test_non_peer_gpu_residency_is_host_staged_d2h_then_h2d(self) -> None:
        phase3 = _phase3(self)
        gpu0 = phase3.Placement("cuda", 0)
        gpu1 = phase3.Placement("cuda", 1)
        request = self._request(phase3, source=gpu0, destination=gpu1)

        moves = phase3.compile_moves(
            (request,),
            topology=phase3.DeviceTopology(
                peer_access=frozenset(), cuda_devices=(0, 1)
            ),
        )
        self.assertEqual(tuple(move.kind.value for move in moves), ("d2h", "h2d"))
        self.assertEqual(moves[0].destination.placement, phase3.Placement("cpu"))
        self.assertEqual(moves[1].source.placement, phase3.Placement("cpu"))

    def test_move_descriptor_is_typed_and_cannot_carry_a_runtime_tensor(self) -> None:
        phase3 = _phase3(self)
        gpu0 = phase3.Placement("cuda", 0)
        gpu1 = phase3.Placement("cuda", 1)
        request = self._request(phase3, source=gpu0, destination=gpu1)
        move = phase3.compile_moves(
            (request,),
            topology=phase3.DeviceTopology(peer_access=frozenset({(0, 1)})),
        )[0]

        self.assertIsInstance(move, phase3.Move)
        self.assertFalse(hasattr(move, "tensor"))
        self.assertEqual(move.value, phase3.ValueKey("hidden", 1))


class Phase3SchedulingAdmissionTest(unittest.TestCase):
    def test_candidate_list_schedule_is_independent_of_declaration_order(self) -> None:
        phase3 = _phase3(self)
        candidates = (
            phase3.ScheduleCandidate("decode.b", priority=1, ready_at=0),
            phase3.ScheduleCandidate("decode.a", priority=1, ready_at=0),
            phase3.ScheduleCandidate("decode.c", priority=0, ready_at=0),
        )

        first = phase3.schedule_candidates(candidates)
        second = phase3.schedule_candidates(tuple(reversed(candidates)))
        self.assertEqual(first, second)
        self.assertEqual(tuple(item.key for item in first), ("decode.a", "decode.b", "decode.c"))

    def test_candidate_schedule_respects_dependencies_before_priority_ties(self) -> None:
        phase3 = _phase3(self)
        candidates = (
            phase3.ScheduleCandidate("consumer", priority=100, ready_at=0, depends_on=("producer",)),
            phase3.ScheduleCandidate("producer", priority=0, ready_at=0),
        )

        schedule = phase3.schedule_candidates(candidates)
        self.assertEqual(tuple(item.key for item in schedule), ("producer", "consumer"))

    def test_lease_blocks_reuse_and_copy_until_retirement(self) -> None:
        phase3 = _phase3(self)
        lease = phase3.Lease(
            key="hidden.slot.0",
            live_start=10,
            live_end=20,
            ready=True,
            retired=False,
        )
        self.assertFalse(phase3.admit_reuse(lease, live_start=15, live_end=18))
        self.assertTrue(phase3.admit_reuse(lease, live_start=21, live_end=25))

        copy = phase3.CopyAdmission(
            source="hidden.cpu",
            destination="hidden.slot.0",
            at_step=15,
        )
        self.assertFalse(phase3.admit_copy(copy, leases=(lease,)))
        retired = phase3.Lease(
            key="hidden.slot.0",
            live_start=10,
            live_end=20,
            ready=True,
            retired=True,
        )
        self.assertTrue(phase3.admit_copy(copy, leases=(retired,)))

    def test_fake_runtime_executes_only_admitted_typed_moves(self) -> None:
        phase3 = _phase3(self)
        runtime = _FakeCopyRuntime()
        moves = (
            phase3.Move(
                value=phase3.ValueKey("hidden", 1),
                source=phase3.Residency(
                    phase3.ValueKey("hidden", 1), phase3.Placement("cpu")
                ),
                destination=phase3.Residency(
                    phase3.ValueKey("hidden", 1), phase3.Placement("cuda", 0)
                ),
                kind=phase3.MoveKind.H2D,
            ),
        )

        phase3.execute_moves(moves, runtime=runtime)
        self.assertEqual(runtime.calls, [("h2d", "cpu", "cuda:0")])


if __name__ == "__main__":
    unittest.main()
