"""Unit contracts for the immutable Phase 5 adaptive admission plan."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.placement import (
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
from tllm.workgraph.admission import (
    AdmissionMode,
    BudgetUseDesc,
    CandidateActionDesc,
    CandidateActionKey,
    CandidateActionKind,
    CandidateMoveSetDesc,
    EWMAProfileKey,
    FixedRatio,
    IntegerEWMAProfile,
    InterferenceBudgetDesc,
    InterferenceBudgetKey,
    MoveSetKey,
    AdmissionControlPlan,
    ShapeBucketDesc,
    ShapeBucketKey,
    WorkItemDesc,
    WorkItemKey,
    WorkloadBucketDesc,
    WorkloadBucketKey,
    approved_actions,
    exhaustive_admission_oracle,
    greedy_admission,
    static_admission,
    validate_candidate_move_sets,
)
from tllm.workgraph.signature import structural_signature
from tllm.runtime.scheduling.adaptive import (
    AdaptiveDecisionKind,
    AdaptiveAdmissionScheduler,
)


def _empty_placement(epoch: int = 1) -> PlacementPlan:
    return PlacementPlan(
        plan_epoch=epoch,
        selections=(),
        schedule=(),
        topology=DeviceTopology(frozenset()),
    )


def _two_hop_placement(epoch: int = 1) -> tuple[PlacementPlan, ScheduledMove, ScheduledMove]:
    source_node = NodeKey.consumer_flow("admission.source", 0)
    target_node = NodeKey.consumer_flow("admission.target", 0)
    source_impl = ExecutorKey("admission", "source")
    target_impl = ExecutorKey("admission", "target")
    source = Placement(DeviceKind.CUDA, 0)
    target = Placement(DeviceKind.CUDA, 1)
    host = Placement(DeviceKind.CPU)
    value = ValueKey("admission-value", 1)
    source_residency = Residency(value, source)
    host_residency = Residency(value, host, MemoryKind.PINNED_HOST)
    target_residency = Residency(value, target)

    first = ScheduledMove(
        key=MoveKey(value, 0),
        move=Move(value, source_residency, host_residency, MoveKind.D2H, 8),
        engine=CopyEngine.D2H,
        start_at=1,
        end_at=2,
        producer=source_node,
        consumer=target_node,
    )
    second = ScheduledMove(
        key=MoveKey(value, 1),
        move=Move(value, host_residency, target_residency, MoveKind.H2D, 8),
        engine=CopyEngine.H2D,
        start_at=2,
        end_at=3,
        producer=source_node,
        consumer=target_node,
        depends_on=(first.key,),
    )
    plan = PlacementPlan(
        plan_epoch=epoch,
        selections=(
            ImplementationSelectionDesc(
                source_node,
                source_impl,
                source_impl,
                source,
                DurationProfile(1),
            ),
            ImplementationSelectionDesc(
                target_node,
                target_impl,
                target_impl,
                target,
                DurationProfile(1),
            ),
        ),
        schedule=(
            PlacedNodeDesc(source_node, source_impl, source, 0, 1),
            PlacedNodeDesc(target_node, target_impl, target, 5, 6),
        ),
        topology=DeviceTopology(frozenset(), (0, 1)),
        value_flows=(ValueFlowDesc(value, source_node, target_node, 8),),
        residencies=(source_residency, host_residency, target_residency),
        moves=(first, second),
        leases=(
            PlacementLeaseDesc(source_residency, 0, 6),
            PlacementLeaseDesc(host_residency, 0, 6),
            PlacementLeaseDesc(target_residency, 0, 6),
        ),
        copy_engines=(
            CopyEngineCapacityDesc(CopyEngine.D2H),
            CopyEngineCapacityDesc(CopyEngine.H2D),
        ),
    )
    return plan, first, second


def _plan(
    *,
    placement: PlacementPlan | None = None,
    actions: tuple[CandidateActionDesc, ...] | None = None,
    shape_buckets: tuple[ShapeBucketDesc, ...] | None = None,
    workload_buckets: tuple[WorkloadBucketDesc, ...] | None = None,
    budgets: tuple[InterferenceBudgetDesc, ...] = (),
    profiles: tuple[IntegerEWMAProfile, ...] = (),
    buffer_capacity: int = 2,
    densification_limit: int = 64,
    mode: AdmissionMode = AdmissionMode.STATIC,
) -> AdmissionControlPlan:
    shape = ShapeBucketKey("shape", "pair")
    workload = WorkloadBucketKey("work", "decode")
    fallback_key = CandidateActionKey("action", "fallback")
    execute_key = CandidateActionKey("action", "execute")
    fallback = CandidateActionDesc(
        key=fallback_key,
        kind=CandidateActionKind.AUTHORITATIVE_FALLBACK,
        semantic_key="default",
        duration=1,
        deadline=8,
        unknown_bucket=True,
    )
    execute = CandidateActionDesc(
        key=execute_key,
        kind=CandidateActionKind.EXECUTE,
        semantic_key="default",
        workload_bucket=workload,
        shape_bucket=shape,
        fallback_key=fallback_key,
        duration=1,
        deadline=8,
        buffer_credits=1,
        utility=3,
    )
    if actions is None:
        actions = (fallback, execute)
    if shape_buckets is None:
        shape_buckets = (ShapeBucketDesc(shape, shape=(2, 4)),)
    if workload_buckets is None:
        workload_buckets = (WorkloadBucketDesc(workload, (shape,)),)
    return AdmissionControlPlan(
        plan_epoch=(placement or _empty_placement()).plan_epoch,
        placement_plan=placement or _empty_placement(),
        scheduled_moves=(placement or _empty_placement()).moves,
        move_sets=(),
        workload_buckets=workload_buckets,
        shape_buckets=shape_buckets,
        budgets=budgets,
        actions=actions,
        unknown_bucket_fallback=fallback_key,
        buffer_capacity=buffer_capacity,
        mode=mode,
        profiles=profiles,
        densification_limit=densification_limit,
    )


class Phase5AdaptivePlanUnitTest(unittest.TestCase):
    def test_descriptors_are_frozen_and_keys_are_stable_injective(self) -> None:
        keys = (
            EWMAProfileKey("stats", "latency"),
            WorkloadBucketKey("bucket", "decode"),
            ShapeBucketKey("bucket", "tokens"),
            InterferenceBudgetKey("budget", "shared"),
            MoveSetKey("moves", "fast", 2),
            CandidateActionKey("action", "run", 2),
            WorkItemKey("item", "request", 2),
        )
        self.assertEqual(len({key.value for key in keys}), len(keys))
        self.assertEqual(EWMAProfileKey("stats", "latency", 2).value, "stats:latency@2")
        self.assertEqual(MoveSetKey("moves", "fast", 2).value, "moves:fast/2")
        with self.assertRaises(FrozenInstanceError):
            keys[0].name = "other"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            _plan().actions[0].deadline = 9  # type: ignore[misc]
        for key_type in (EWMAProfileKey, WorkloadBucketKey, ShapeBucketKey):
            with self.assertRaises(ValueError):
                key_type("runtime", "<object at 0xdeadbeef>")
        with self.assertRaises(ValueError):
            CandidateActionKey("bad/name", "run")

    def test_required_execute_has_authoritative_fallback_and_cannot_be_shed(self) -> None:
        plan = _plan()
        required = WorkItemDesc(
            WorkItemKey("item", "required"),
            (2, 4),
            WorkloadBucketKey("work", "decode"),
            required=True,
            allow_shedding=True,
        )
        self.assertFalse(required.allow_shedding)
        candidates = approved_actions(plan, required)
        self.assertTrue(candidates)
        self.assertTrue(
            all(
                candidate.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK
                or candidate.fallback_key is not None
                for candidate in candidates
            )
        )
        result = greedy_admission(plan, (required,))
        self.assertTrue(result.feasible)
        self.assertEqual(result.shed, ())
        self.assertEqual(len(result.selected), 1)

    def test_non_sheddable_and_workload_required_items_are_mandatory(self) -> None:
        base = _plan(buffer_capacity=1)
        workload = base.workload_buckets[0]
        required_by_default = _plan(
            buffer_capacity=1,
            workload_buckets=(
                WorkloadBucketDesc(
                    workload.key,
                    workload.shape_buckets,
                    required_by_default=True,
                ),
            ),
        )
        leading = WorkItemDesc(
            WorkItemKey("item", "leading"),
            (2, 4),
            workload.key,
        )
        non_sheddable = WorkItemDesc(
            WorkItemKey("item", "non-sheddable"),
            (2, 4),
            workload.key,
            allow_shedding=False,
        )
        default_required = WorkItemDesc(
            WorkItemKey("item", "default-required"),
            (2, 4),
            workload.key,
        )

        first = greedy_admission(base, (leading, non_sheddable))
        second = greedy_admission(required_by_default, (leading, default_required))

        self.assertTrue(first.feasible)
        self.assertEqual(first.shed, ())
        self.assertEqual(len(first.selected), 2)
        self.assertTrue(second.feasible)
        self.assertEqual(second.shed, ())
        self.assertEqual(len(second.selected), 2)

    def test_fallback_is_authoritative_and_preserves_semantics(self) -> None:
        plan = _plan()
        execute = next(
            action
            for action in plan.actions
            if action.kind == CandidateActionKind.EXECUTE
        )
        fallback = next(action for action in plan.actions if action.unknown_bucket)
        with self.assertRaises(ValueError):
            _plan(
                actions=(
                    execute,
                    replace(fallback, semantic_key="different"),
                )
            )
        unknown = WorkItemDesc(
            WorkItemKey("item", "unknown"),
            (99, 99),
            WorkloadBucketKey("work", "decode"),
        )
        candidates = approved_actions(plan, unknown)
        self.assertEqual(candidates, (fallback,))
        self.assertTrue(candidates[0].authoritative)
        self.assertTrue(candidates[0].unknown_bucket)
        with self.assertRaises(RuntimeError):
            approved_actions(plan, replace(unknown, semantic_key="other"))
        with self.assertRaises(RuntimeError):
            static_admission(plan, (replace(unknown, semantic_key="other"),))
        with self.assertRaises(TypeError):
            replace(fallback, authoritative=0)  # type: ignore[arg-type]

    def test_placement_epoch_signature_and_existing_move_anchoring(self) -> None:
        placement, first, second = _two_hop_placement(epoch=7)
        move_set = CandidateMoveSetDesc(MoveSetKey("moves", "chain"), (second, first))
        plan = _plan(placement=placement)
        anchored = AdmissionControlPlan(
            plan_epoch=7,
            placement_plan=placement,
            scheduled_moves=placement.moves,
            move_sets=(move_set,),
            workload_buckets=plan.workload_buckets,
            shape_buckets=plan.shape_buckets,
            budgets=(),
            actions=plan.actions,
            unknown_bucket_fallback=plan.unknown_bucket_fallback,
            buffer_capacity=2,
        )
        self.assertEqual(anchored.placement_signature, structural_signature(placement))
        self.assertEqual(anchored.scheduled_moves, placement.moves)
        self.assertEqual(anchored.move_sets[0].moves, (first, second))
        with self.assertRaises(ValueError):
            AdmissionControlPlan(
                plan_epoch=8,
                placement_plan=placement,
                scheduled_moves=placement.moves,
                move_sets=(),
                workload_buckets=plan.workload_buckets,
                shape_buckets=plan.shape_buckets,
                budgets=(),
                actions=plan.actions,
                unknown_bucket_fallback=plan.unknown_bucket_fallback,
                buffer_capacity=2,
            )
        with self.assertRaises(ValueError):
            AdmissionControlPlan(
                plan_epoch=7,
                placement_plan=placement,
                scheduled_moves=(replace(first, end_at=4), second),
                move_sets=(),
                workload_buckets=plan.workload_buckets,
                shape_buckets=plan.shape_buckets,
                budgets=(),
                actions=plan.actions,
                unknown_bucket_fallback=plan.unknown_bucket_fallback,
                buffer_capacity=2,
            )

    def test_candidate_move_sets_are_dependency_closed_subsets(self) -> None:
        placement, first, second = _two_hop_placement()
        self.assertEqual(
            CandidateMoveSetDesc(MoveSetKey("moves", "closed"), (second, first)).move_keys,
            (first.key, second.key),
        )
        with self.assertRaises(ValueError):
            CandidateMoveSetDesc(MoveSetKey("moves", "open"), (second,))
        self.assertEqual(tuple(item.key for item in placement.moves), (first.key, second.key))

    def test_control_plan_rejects_move_set_descriptor_that_differs_from_placement(self) -> None:
        placement, first, second = _two_hop_placement()
        base = _plan(placement=placement)
        move_set = CandidateMoveSetDesc(
            MoveSetKey("moves", "descriptor-mismatch"),
            (replace(first, end_at=4), second),
        )

        with self.assertRaises(ValueError):
            AdmissionControlPlan(
                plan_epoch=base.plan_epoch,
                placement_plan=placement,
                scheduled_moves=placement.moves,
                move_sets=(move_set,),
                workload_buckets=base.workload_buckets,
                shape_buckets=base.shape_buckets,
                budgets=base.budgets,
                actions=base.actions,
                unknown_bucket_fallback=base.unknown_bucket_fallback,
                buffer_capacity=base.buffer_capacity,
            )

    def test_candidate_move_set_validator_rejects_descriptor_mismatch(self) -> None:
        placement, first, second = _two_hop_placement()
        move_set = CandidateMoveSetDesc(
            MoveSetKey("moves", "descriptor-mismatch"),
            (replace(first, end_at=4), second),
        )

        with self.assertRaises(ValueError):
            validate_candidate_move_sets(placement.moves, (move_set,))

    def test_move_deadline_is_identical_in_runtime_greedy_and_oracle(self) -> None:
        placement, first, second = _two_hop_placement()
        base = _plan(placement=placement)
        move_set = CandidateMoveSetDesc(
            MoveSetKey("moves", "deadline"),
            (first, second),
        )
        base_execute = next(
            action
            for action in base.actions
            if action.kind == CandidateActionKind.EXECUTE
        )
        fallback = next(action for action in base.actions if action.unknown_bucket)
        execute = replace(
            base_execute,
            move_set=move_set.key,
            ready_at=5,
            duration=1,
            deadline=7,
        )
        plan = AdmissionControlPlan(
            plan_epoch=base.plan_epoch,
            placement_plan=placement,
            scheduled_moves=placement.moves,
            move_sets=(move_set,),
            workload_buckets=base.workload_buckets,
            shape_buckets=base.shape_buckets,
            budgets=base.budgets,
            actions=(fallback, execute),
            unknown_bucket_fallback=base.unknown_bucket_fallback,
            buffer_capacity=base.buffer_capacity,
            mode=AdmissionMode.ADAPTIVE,
        )
        item = WorkItemDesc(
            WorkItemKey("item", "move-deadline"),
            (2, 4),
            WorkloadBucketKey("work", "decode"),
        )

        greedy = greedy_admission(plan, (item,))
        oracle = exhaustive_admission_oracle(plan, (item,))
        runtime = AdaptiveAdmissionScheduler(plan).decide(item)

        self.assertEqual(greedy.selected, ((item.key, execute.key),))
        self.assertEqual(oracle.selected, ((item.key, execute.key),))
        self.assertEqual(runtime.action, AdaptiveDecisionKind.RUN)
        self.assertEqual(runtime.action_key, execute.key)

    def test_workload_and_shape_buckets_match_exact_and_bounded_shapes(self) -> None:
        exact = ShapeBucketKey("shape", "exact")
        bounded = ShapeBucketKey("shape", "bounded")
        workload = WorkloadBucketKey("work", "decode")
        exact_desc = ShapeBucketDesc(exact, shape=(2, 4))
        bounded_desc = ShapeBucketDesc(bounded, lower_bounds=(1, 2), upper_bounds=(4, 8))
        self.assertTrue(exact_desc.matches((2, 4)))
        self.assertFalse(exact_desc.matches((2, 5)))
        self.assertTrue(bounded_desc.matches((3, 7)))
        self.assertFalse(bounded_desc.matches((5, 7)))
        action = CandidateActionDesc(
            CandidateActionKey("action", "decode"),
            CandidateActionKind.EXECUTE,
            workload_bucket=workload,
            shape_bucket=bounded,
            fallback_key=CandidateActionKey("action", "fallback"),
        )
        fallback = CandidateActionDesc(
            CandidateActionKey("action", "fallback"),
            CandidateActionKind.AUTHORITATIVE_FALLBACK,
            unknown_bucket=True,
        )
        plan = _plan(
            actions=(fallback, action),
            shape_buckets=(bounded_desc,),
            workload_buckets=(WorkloadBucketDesc(workload, (bounded,)),),
        )
        known = WorkItemDesc(WorkItemKey("item", "known"), (3, 7), workload)
        self.assertEqual(approved_actions(plan, known)[0].key, action.key)
        unknown = replace(known, shape=(9, 9))
        self.assertEqual(approved_actions(plan, unknown), (fallback,))

    def test_integer_ewma_uses_integer_rounding_and_bounds(self) -> None:
        key = EWMAProfileKey("stats", "latency")
        profile = IntegerEWMAProfile(
            key=key,
            estimate=10,
            alpha=FixedRatio(1, 2),
            minimum=4,
            maximum=12,
        )
        observed = profile.observe(11)
        self.assertEqual(observed.estimate, 11)
        self.assertEqual(observed.samples, 1)
        self.assertEqual(observed.value, observed.estimate)
        self.assertEqual(observed.observe(4).estimate, 8)
        self.assertEqual(observed.observe(12).estimate, 12)
        with self.assertRaises(ValueError):
            profile.observe(99)
        with self.assertRaises(ValueError):
            FixedRatio(3, 2)

    def test_buffer_deadline_and_interference_constraints_make_selection_feasible(self) -> None:
        budget_key = InterferenceBudgetKey("budget", "shared")
        budget = InterferenceBudgetDesc(budget_key, capacity=1)
        use = (BudgetUseDesc(budget_key, 1),)
        base = _plan(budgets=(budget,), buffer_capacity=1)
        execute = replace(
            next(
                action
                for action in base.actions
                if action.kind == CandidateActionKind.EXECUTE
            ),
            budget_uses=use,
        )
        fallback = replace(
            next(action for action in base.actions if action.unknown_bucket),
            buffer_credits=1,
            budget_uses=use,
        )
        shed = CandidateActionDesc(
            CandidateActionKey("action", "shed"),
            CandidateActionKind.SHED,
            workload_bucket=execute.workload_bucket,
            shape_bucket=execute.shape_bucket,
            duration=0,
        )
        constrained = _plan(
            actions=(fallback, execute, shed),
            budgets=(budget,),
            buffer_capacity=1,
        )
        items = (
            WorkItemDesc(WorkItemKey("item", "a"), (2, 4), WorkloadBucketKey("work", "decode")),
            WorkItemDesc(WorkItemKey("item", "b"), (2, 4), WorkloadBucketKey("work", "decode")),
        )
        result = greedy_admission(constrained, items)
        self.assertTrue(result.feasible)
        self.assertEqual(len(result.selected), 1)
        too_late = replace(items[0], deadline=1)
        self.assertEqual(greedy_admission(constrained, (too_late,)).shed, (too_late.key,))
        required = replace(too_late, required=True)
        self.assertFalse(greedy_admission(constrained, (required,)).feasible)

    def test_static_admission_is_deterministic_and_ignores_observed_profile_values(self) -> None:
        profile_key = EWMAProfileKey("stats", "latency")
        profile = IntegerEWMAProfile(profile_key, estimate=2)
        plan = _plan(profiles=(profile,), mode=AdmissionMode.STATIC)
        item = WorkItemDesc(
            WorkItemKey("item", "one"),
            (2, 4),
            WorkloadBucketKey("work", "decode"),
        )
        first = static_admission(plan, (item,))
        observed_plan = replace(plan, profiles=(profile.observe(10_000),))
        second = static_admission(observed_plan, (item,))
        self.assertEqual(first, second)
        self.assertEqual(static_admission(plan, (item,)), static_admission(plan, (item,)))

    def test_greedy_and_exhaustive_small_graph_oracle_agree_on_required_work(self) -> None:
        plan = _plan(buffer_capacity=1)
        items = (
            WorkItemDesc(
                WorkItemKey("item", "required"),
                (2, 4),
                WorkloadBucketKey("work", "decode"),
                required=True,
            ),
            WorkItemDesc(
                WorkItemKey("item", "optional"),
                (2, 4),
                WorkloadBucketKey("work", "decode"),
            ),
        )
        greedy = greedy_admission(plan, items)
        oracle = exhaustive_admission_oracle(plan, items, max_items=2)
        self.assertTrue(greedy.feasible)
        self.assertTrue(oracle.feasible)
        self.assertIn(items[0].key, {key for key, _ in oracle.selected})
        self.assertGreaterEqual(oracle.objective, greedy.objective)
        self.assertEqual(oracle, exhaustive_admission_oracle(plan, tuple(reversed(items)), max_items=2))


if __name__ == "__main__":
    unittest.main()
