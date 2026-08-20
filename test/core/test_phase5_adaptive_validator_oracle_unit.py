"""Differential acceptance and rejection tests for the Rust Phase 5 validator."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import unittest

from tllm import core
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.phase3 import (
    CopyEngine,
    CopyEngineCapacityDesc,
    DeviceKind,
    DeviceTopology,
    DurationProfile,
    ImplementationSelectionDesc,
    MemoryKind,
    Move,
    MoveKey,
    MoveKind,
    Placement,
    PlacementLeaseDesc,
    PlacementPlan,
    PlacedNodeDesc,
    Residency,
    ScheduledMove,
    ValueFlowDesc,
    ValueKey,
)
from tllm.workgraph.phase5 import (
    AdmissionMode,
    BudgetUseDesc,
    CandidateActionDesc,
    CandidateActionKey,
    CandidateActionKind,
    CandidateMoveSetDesc,
    InterferenceBudgetDesc,
    InterferenceBudgetKey,
    MoveSetKey,
    Phase5ControlPlan,
    ShapeBucketDesc,
    ShapeBucketKey,
    WorkloadBucketDesc,
    WorkloadBucketKey,
)


@dataclass(frozen=True, slots=True)
class _FrozenAttrs:
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
    values = _values(obj)
    values.update(changes)
    return _FrozenAttrs(tuple(values.items()))


def _empty_placement(epoch: int = 1) -> PlacementPlan:
    return PlacementPlan(epoch, (), (), DeviceTopology(frozenset()))


def _two_hop_placement(
    epoch: int = 1,
    *,
    value_name: str = "rust-value",
) -> tuple[PlacementPlan, ScheduledMove, ScheduledMove]:
    source_node = NodeKey.consumer_flow("rust.source", 0)
    target_node = NodeKey.consumer_flow("rust.target", 0)
    source_impl = ExecutorKey("rust", "source")
    target_impl = ExecutorKey("rust", "target")
    source = Placement(DeviceKind.CUDA, 0)
    target = Placement(DeviceKind.CUDA, 1)
    host = Placement(DeviceKind.CPU)
    value = ValueKey(value_name, 1)
    source_residency = Residency(value, source)
    host_residency = Residency(value, host, MemoryKind.PINNED_HOST)
    target_residency = Residency(value, target)
    first = ScheduledMove(
        MoveKey(value, 0),
        Move(value, source_residency, host_residency, MoveKind.D2H, 8),
        CopyEngine.D2H,
        1,
        2,
        producer=source_node,
        consumer=target_node,
    )
    second = ScheduledMove(
        MoveKey(value, 1),
        Move(value, host_residency, target_residency, MoveKind.H2D, 8),
        CopyEngine.H2D,
        2,
        3,
        producer=source_node,
        consumer=target_node,
        depends_on=(first.key,),
    )
    plan = PlacementPlan(
        epoch,
        (
            ImplementationSelectionDesc(source_node, source_impl, source_impl, source, DurationProfile(1)),
            ImplementationSelectionDesc(target_node, target_impl, target_impl, target, DurationProfile(1)),
        ),
        (
            PlacedNodeDesc(source_node, source_impl, source, 0, 1),
            PlacedNodeDesc(target_node, target_impl, target, 5, 6),
        ),
        DeviceTopology(frozenset(), (0, 1)),
        (ValueFlowDesc(value, source_node, target_node, 8),),
        (source_residency, host_residency, target_residency),
        (first, second),
        (
            PlacementLeaseDesc(source_residency, 0, 6),
            PlacementLeaseDesc(host_residency, 0, 6),
            PlacementLeaseDesc(target_residency, 0, 6),
        ),
        (CopyEngineCapacityDesc(CopyEngine.D2H), CopyEngineCapacityDesc(CopyEngine.H2D)),
    )
    return plan, first, second


def _phase5(
    *,
    placement: PlacementPlan | None = None,
    move_sets: tuple[CandidateMoveSetDesc, ...] = (),
    actions: tuple[CandidateActionDesc, ...] | None = None,
    shape_buckets: tuple[ShapeBucketDesc, ...] | None = None,
    workload_buckets: tuple[WorkloadBucketDesc, ...] | None = None,
    buffer_capacity: int = 2,
    densification_limit: int = 64,
) -> Phase5ControlPlan:
    placement = placement or _empty_placement()
    shape = ShapeBucketKey("shape", "pair")
    workload = WorkloadBucketKey("work", "decode")
    fallback_key = CandidateActionKey("action", "fallback")
    execute_key = CandidateActionKey("action", "execute")
    fallback = CandidateActionDesc(
        fallback_key,
        CandidateActionKind.AUTHORITATIVE_FALLBACK,
        duration=1,
        deadline=8,
        unknown_bucket=True,
    )
    execute = CandidateActionDesc(
        execute_key,
        CandidateActionKind.EXECUTE,
        workload_bucket=workload,
        shape_bucket=shape,
        fallback_key=fallback_key,
        duration=1,
        deadline=8,
        buffer_credits=1,
        utility=2,
    )
    return Phase5ControlPlan(
        plan_epoch=placement.plan_epoch,
        placement_plan=placement,
        scheduled_moves=placement.moves,
        move_sets=move_sets,
        workload_buckets=workload_buckets or (WorkloadBucketDesc(workload, (shape,)),),
        shape_buckets=shape_buckets or (ShapeBucketDesc(shape, shape=(2, 4)),),
        budgets=(),
        actions=actions or (fallback, execute),
        unknown_bucket_fallback=fallback_key,
        buffer_capacity=buffer_capacity,
        mode=AdmissionMode.STATIC,
        densification_limit=densification_limit,
    )


class Phase5AdaptiveValidatorOracleTest(unittest.TestCase):
    def test_rust_accepts_canonical_adaptive_plan_and_returns_action_order(self) -> None:
        placement, first, second = _two_hop_placement()
        move_set = CandidateMoveSetDesc(MoveSetKey("moves", "chain"), (second, first))
        plan = _phase5(placement=placement, move_sets=(move_set,))
        validation = core.validate_adaptive_flow_plan(plan)
        self.assertEqual(
            validation.topological_order,
            [action.key.value for action in plan.actions],
        )

    def test_rust_signature_matches_python_ascii_json_for_unicode(self) -> None:
        placement, first, second = _two_hop_placement(value_name="value-\u503c-\U0001f600")
        move_set = CandidateMoveSetDesc(MoveSetKey("moves", "unicode"), (first, second))
        plan = _phase5(placement=placement, move_sets=(move_set,))
        self.assertIsNotNone(core.validate_adaptive_flow_plan(plan))

    def test_rust_rejects_epoch_signature_and_move_set_anchor_changes(self) -> None:
        placement, first, second = _two_hop_placement()
        move_set = CandidateMoveSetDesc(MoveSetKey("moves", "chain"), (first, second))
        plan = _phase5(placement=placement, move_sets=(move_set,))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(_proxy(plan, plan_epoch=2))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(_proxy(plan, placement_signature="bad-signature"))
        rewritten = replace(first, end_at=4)
        rewritten_set = _proxy(move_set, moves=(rewritten, second))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(_proxy(plan, move_sets=(rewritten_set,)))

    def test_rust_rejects_non_closed_and_non_subset_existing_move_sets(self) -> None:
        placement, first, second = _two_hop_placement()
        plan = _phase5(placement=placement)
        open_set = _proxy(
            CandidateMoveSetDesc(MoveSetKey("moves", "open"), (first,)),
            moves=(second,),
        )
        foreign = _proxy(first, key=MoveKey(ValueKey("foreign", 1), 0))
        foreign_set = _proxy(
            CandidateMoveSetDesc(MoveSetKey("moves", "foreign"), (first,)),
            moves=(foreign,),
        )
        for candidate in (open_set, foreign_set):
            with self.assertRaises((TypeError, ValueError, RuntimeError)):
                core.validate_adaptive_flow_plan(_proxy(plan, move_sets=(candidate,)))

    def test_rust_rejects_densification_and_bucket_reference_violations(self) -> None:
        plan = _phase5(densification_limit=2)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(_proxy(plan, densification_limit=1))

        shape_a = ShapeBucketKey("shape", "a")
        shape_b = ShapeBucketKey("shape", "b")
        workload = WorkloadBucketKey("work", "decode")
        fallback = next(action for action in plan.actions if action.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK)
        execute = next(action for action in plan.actions if action.kind == CandidateActionKind.EXECUTE)
        bad_execute = replace(execute, shape_bucket=shape_b)
        bad_plan = _proxy(
            plan,
            actions=(fallback, bad_execute),
            shape_buckets=(ShapeBucketDesc(shape_a, shape=(2, 4)), ShapeBucketDesc(shape_b, shape=(3, 4))),
            workload_buckets=(WorkloadBucketDesc(workload, (shape_a,)),),
        )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(bad_plan)

    def test_rust_rejects_required_shedding_and_nonsemantic_fallbacks(self) -> None:
        plan = _phase5()
        fallback = next(action for action in plan.actions if action.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK)
        execute = next(action for action in plan.actions if action.kind == CandidateActionKind.EXECUTE)
        shed = CandidateActionDesc(
            CandidateActionKey("action", "shed"),
            CandidateActionKind.SHED,
            duration=0,
        )
        bad_shed = _proxy(shed, required=True)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(
                _proxy(plan, actions=(fallback, execute, bad_shed), densification_limit=3)
            )
        mismatched = replace(execute, semantic_key="different")
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(_proxy(plan, actions=(fallback, mismatched)))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(
                _proxy(plan, actions=(_proxy(fallback, authoritative=0), execute))
            )

    def test_rust_rejects_unknown_fallback_and_unstable_or_noncanonical_keys(self) -> None:
        plan = _phase5()
        fallback = next(action for action in plan.actions if action.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK)
        execute = next(action for action in plan.actions if action.kind == CandidateActionKind.EXECUTE)
        known_fallback = replace(fallback, key=CandidateActionKey("action", "known"), unknown_bucket=False)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(
                _proxy(plan, actions=(known_fallback, execute), unknown_bucket_fallback=known_fallback.key)
            )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(_proxy(plan, placement_signature="A" * 64))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(
                _proxy(plan, actions=(fallback, execute))
            )

    def test_rust_rejects_invalid_budget_use_and_accepts_sorted_budget_use(self) -> None:
        budget_key = InterferenceBudgetKey("budget", "shared")
        budget = InterferenceBudgetDesc(budget_key, 2)
        use = BudgetUseDesc(budget_key, 1)
        plan = _phase5()
        fallback = next(action for action in plan.actions if action.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK)
        execute = next(action for action in plan.actions if action.kind == CandidateActionKind.EXECUTE)
        budgeted_execute = replace(execute, budget_uses=(use,))
        budgeted = _proxy(
            plan,
            budgets=(budget,),
            actions=(budgeted_execute, fallback),
        )
        self.assertIsNotNone(core.validate_adaptive_flow_plan(budgeted))
        unknown_use = BudgetUseDesc(InterferenceBudgetKey("budget", "missing"), 1)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            core.validate_adaptive_flow_plan(
                _proxy(
                    plan,
                    budgets=(budget,),
                    actions=(replace(execute, budget_uses=(unknown_use,)), fallback),
                )
            )


if __name__ == "__main__":
    unittest.main()
