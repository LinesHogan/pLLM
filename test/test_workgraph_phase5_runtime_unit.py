"""Focused unit contracts for the Python-owned Phase 5 adaptive runtime."""

from __future__ import annotations

import unittest

from tllm.runtime.scheduling.phase5_adaptive import (
    AdaptiveDecisionKind,
    AdaptiveMode,
    AdmissionContext,
    Phase5AdaptiveScheduler,
)
from tllm.workgraph.phase5 import (
    AdmissionMode,
    BudgetUseDesc,
    CandidateActionDesc,
    CandidateActionKey,
    CandidateActionKind,
    EWMAProfileKey,
    FixedRatio,
    IntegerEWMAProfile,
    InterferenceBudgetDesc,
    InterferenceBudgetKey,
    Phase5ControlPlan,
    ShapeBucketDesc,
    ShapeBucketKey,
    WorkItemDesc,
    WorkItemKey,
    WorkloadBucketDesc,
    WorkloadBucketKey,
)
from tllm.workgraph.placement import DeviceTopology, PlacementPlan


class WorkGraphPhase5RuntimeUnitTest(unittest.TestCase):
    @staticmethod
    def _plan(*, with_budget: bool = False, deadline: int = 8, duration: int = 1) -> tuple[Phase5ControlPlan, WorkItemDesc, WorkItemDesc, InterferenceBudgetKey | None]:
        shape_key = ShapeBucketKey("shape", "two")
        workload_key = WorkloadBucketKey("work", "decode")
        fallback_key = CandidateActionKey("action", "fallback")
        execute_key = CandidateActionKey("action", "execute")
        fallback = CandidateActionDesc(
            key=fallback_key,
            kind=CandidateActionKind.AUTHORITATIVE_FALLBACK,
            semantic_key="default",
            deadline=deadline,
            duration=1,
            unknown_bucket=True,
        )
        budget_key = InterferenceBudgetKey("budget", "shared") if with_budget else None
        uses = () if budget_key is None else (BudgetUseDesc(budget_key, 1),)
        execute = CandidateActionDesc(
            key=execute_key,
            kind=CandidateActionKind.EXECUTE,
            semantic_key="default",
            workload_bucket=workload_key,
            shape_bucket=shape_key,
            fallback_key=fallback_key,
            deadline=deadline,
            duration=duration,
            buffer_credits=1,
            budget_uses=uses,
        )
        budgets = () if budget_key is None else (InterferenceBudgetDesc(budget_key, 1),)
        placement_plan = PlacementPlan(
            plan_epoch=1,
            selections=(),
            schedule=(),
            topology=DeviceTopology(frozenset()),
        )
        plan = Phase5ControlPlan(
            plan_epoch=1,
            placement_plan=placement_plan,
            scheduled_moves=(),
            move_sets=(),
            workload_buckets=(WorkloadBucketDesc(workload_key, (shape_key,)),),
            shape_buckets=(ShapeBucketDesc(shape_key, shape=(2,)),),
            budgets=budgets,
            actions=(fallback, execute),
            unknown_bucket_fallback=fallback_key,
            buffer_capacity=1,
            mode=AdmissionMode.ADAPTIVE,
        )
        optional = WorkItemDesc(
            key=WorkItemKey("item", "optional"),
            shape=(2,),
            workload_bucket=workload_key,
        )
        required = WorkItemDesc(
            key=WorkItemKey("item", "required"),
            shape=(2,),
            workload_bucket=workload_key,
            required=True,
        )
        return plan, optional, required, budget_key

    def test_required_work_is_never_shed_when_admission_is_full(self) -> None:
        plan, _optional, required, _budget = self._plan()
        scheduler = Phase5AdaptiveScheduler(plan)

        decision = scheduler.decide(required, context=AdmissionContext(available_buffers=0))

        self.assertNotEqual(decision.action, AdaptiveDecisionKind.DROP)
        self.assertIn(decision.action, (AdaptiveDecisionKind.RUN, AdaptiveDecisionKind.FALLBACK))

    def test_non_sheddable_and_workload_required_work_use_fallback(self) -> None:
        plan, optional, _required, _budget = self._plan()
        non_sheddable = WorkItemDesc(
            key=WorkItemKey("item", "non-sheddable"),
            shape=optional.shape,
            workload_bucket=optional.workload_bucket,
            allow_shedding=False,
        )
        workload = plan.workload_buckets[0]
        required_by_default = Phase5ControlPlan(
            plan_epoch=plan.plan_epoch,
            placement_plan=plan.placement_plan,
            scheduled_moves=plan.scheduled_moves,
            move_sets=plan.move_sets,
            workload_buckets=(
                WorkloadBucketDesc(
                    workload.key,
                    workload.shape_buckets,
                    required_by_default=True,
                ),
            ),
            shape_buckets=plan.shape_buckets,
            budgets=plan.budgets,
            actions=plan.actions,
            unknown_bucket_fallback=plan.unknown_bucket_fallback,
            buffer_capacity=plan.buffer_capacity,
            mode=plan.mode,
        )

        first = Phase5AdaptiveScheduler(plan).decide(
            non_sheddable,
            context=AdmissionContext(available_buffers=0),
        )
        second = Phase5AdaptiveScheduler(required_by_default).decide(
            optional,
            context=AdmissionContext(available_buffers=0),
        )

        self.assertEqual(first.action, AdaptiveDecisionKind.FALLBACK)
        self.assertEqual(second.action, AdaptiveDecisionKind.FALLBACK)

    def test_explicit_action_must_be_the_frozen_action_approved_for_item(self) -> None:
        plan, optional, _required, _budget = self._plan()
        execute = next(
            action
            for action in plan.actions
            if action.kind == CandidateActionKind.EXECUTE
        )
        rewritten = CandidateActionDesc(
            key=execute.key,
            kind=execute.kind,
            semantic_key=execute.semantic_key,
            workload_bucket=execute.workload_bucket,
            shape_bucket=execute.shape_bucket,
            fallback_key=execute.fallback_key,
            deadline=execute.deadline,
            duration=execute.duration + 1,
            buffer_credits=execute.buffer_credits,
        )
        other_shape = WorkItemDesc(
            key=WorkItemKey("item", "other-shape"),
            shape=(99,),
            workload_bucket=optional.workload_bucket,
        )
        scheduler = Phase5AdaptiveScheduler(plan)

        with self.assertRaises(ValueError):
            scheduler.decide(optional, action=rewritten)
        with self.assertRaises(ValueError):
            scheduler.decide(other_shape, action=execute.key)

    def test_unknown_shape_bucket_uses_compiler_fallback(self) -> None:
        plan, _optional, _required, _budget = self._plan()
        unknown = WorkItemDesc(
            key=WorkItemKey("item", "unknown"),
            shape=(99,),
            workload_bucket=WorkloadBucketKey("work", "decode"),
        )
        decision = Phase5AdaptiveScheduler(plan).decide(unknown)

        self.assertEqual(decision.action, AdaptiveDecisionKind.FALLBACK)
        self.assertIsNotNone(decision.candidate)
        assert decision.candidate is not None
        self.assertTrue(decision.candidate.unknown_bucket)

    def test_buffer_deadline_and_interference_admission(self) -> None:
        plan, optional, _required, budget_key = self._plan(
            with_budget=True,
            deadline=2,
            duration=2,
        )
        self.assertIsNotNone(budget_key)
        assert budget_key is not None

        buffer_drop = Phase5AdaptiveScheduler(plan).decide(
            optional,
            context=AdmissionContext(available_buffers=0),
        )
        deadline_drop = Phase5AdaptiveScheduler(plan).decide(
            optional,
            context=AdmissionContext(available_buffers=1, step=1, deadline_step=2),
        )
        interference_drop = Phase5AdaptiveScheduler(plan).decide(
            optional,
            context=AdmissionContext(
                available_buffers=1,
                interference={budget_key: 1},
            ),
        )

        self.assertEqual(buffer_drop.action, AdaptiveDecisionKind.FALLBACK)
        self.assertEqual(buffer_drop.reason, "shedding not approved")
        self.assertEqual(deadline_drop.action, AdaptiveDecisionKind.FALLBACK)
        self.assertEqual(interference_drop.action, AdaptiveDecisionKind.FALLBACK)

    def test_action_level_required_prevents_approved_shedding(self) -> None:
        plan, optional, _required, _budget = self._plan()
        fallback = next(action for action in plan.actions if action.unknown_bucket)
        execute = next(
            action
            for action in plan.actions
            if action.kind == CandidateActionKind.EXECUTE
        )
        required_execute = CandidateActionDesc(
            key=execute.key,
            kind=execute.kind,
            semantic_key=execute.semantic_key,
            workload_bucket=execute.workload_bucket,
            shape_bucket=execute.shape_bucket,
            fallback_key=execute.fallback_key,
            deadline=execute.deadline,
            duration=execute.duration,
            buffer_credits=execute.buffer_credits,
            required=True,
        )
        shed = CandidateActionDesc(
            CandidateActionKey("action", "shed"),
            CandidateActionKind.SHED,
            workload_bucket=execute.workload_bucket,
            shape_bucket=execute.shape_bucket,
            duration=0,
        )
        required_plan = Phase5ControlPlan(
            plan_epoch=plan.plan_epoch,
            placement_plan=plan.placement_plan,
            scheduled_moves=plan.scheduled_moves,
            move_sets=plan.move_sets,
            workload_buckets=plan.workload_buckets,
            shape_buckets=plan.shape_buckets,
            budgets=plan.budgets,
            actions=(fallback, required_execute, shed),
            unknown_bucket_fallback=fallback.key,
            buffer_capacity=plan.buffer_capacity,
            mode=plan.mode,
        )

        decision = Phase5AdaptiveScheduler(required_plan).decide(
            optional,
            context=AdmissionContext(available_buffers=0),
        )

        self.assertEqual(decision.action, AdaptiveDecisionKind.FALLBACK)

    def test_drop_reports_the_high_priority_approved_shed_candidate(self) -> None:
        plan, optional, _required, _budget = self._plan()
        fallback = next(action for action in plan.actions if action.unknown_bucket)
        execute = next(
            action
            for action in plan.actions
            if action.kind == CandidateActionKind.EXECUTE
        )
        shed = CandidateActionDesc(
            key=CandidateActionKey("action", "shed"),
            kind=CandidateActionKind.SHED,
            semantic_key=execute.semantic_key,
            workload_bucket=execute.workload_bucket,
            shape_bucket=execute.shape_bucket,
            duration=0,
            priority=100,
        )
        plan_with_shed = Phase5ControlPlan(
            plan_epoch=plan.plan_epoch,
            placement_plan=plan.placement_plan,
            scheduled_moves=plan.scheduled_moves,
            move_sets=plan.move_sets,
            workload_buckets=plan.workload_buckets,
            shape_buckets=plan.shape_buckets,
            budgets=plan.budgets,
            actions=(fallback, execute, shed),
            unknown_bucket_fallback=plan.unknown_bucket_fallback,
            buffer_capacity=plan.buffer_capacity,
            mode=plan.mode,
        )

        decision = Phase5AdaptiveScheduler(plan_with_shed).decide(
            optional,
            context=AdmissionContext(available_buffers=0),
            allow_defer=False,
        )

        self.assertEqual(decision.action, AdaptiveDecisionKind.DROP)
        self.assertEqual(decision.action_key, shed.key)
        self.assertIsNotNone(decision.candidate)
        assert decision.candidate is not None
        self.assertEqual(decision.candidate.key, shed.key)

    def test_runtime_never_adds_or_rewrites_phase3_moves(self) -> None:
        plan, optional, _required, _budget = self._plan()
        prepared = Phase5AdaptiveScheduler(plan).prepared
        assert prepared is not None

        self.assertEqual(prepared.move_keys, ())
        self.assertEqual(plan.scheduled_moves, ())
        decision = Phase5AdaptiveScheduler(plan).decide(
            optional,
            context=AdmissionContext(available_buffers=1),
        )
        self.assertFalse(hasattr(decision, "moves"))

    def test_static_mode_is_deterministic_and_ignores_observations(self) -> None:
        plan, optional, _required, budget = self._plan(with_budget=True)
        scheduler = Phase5AdaptiveScheduler(plan, mode=AdaptiveMode.STATIC)
        first = scheduler.decide(optional, context=AdmissionContext(available_buffers=1))
        assert budget is not None
        scheduler.observe_interference(budget, 10_000)
        second = scheduler.decide(optional, context=AdmissionContext(available_buffers=1))

        self.assertEqual(first, second)
        self.assertEqual(first.action, AdaptiveDecisionKind.RUN)

    def test_integer_ewma_observation_is_python_owned(self) -> None:
        plan, optional, _required, _budget = self._plan()
        scheduler = Phase5AdaptiveScheduler(plan)
        action_key = next(iter(scheduler.prepared.action_by_key))  # type: ignore[union-attr]

        self.assertEqual(scheduler.observe(action_key, 7), 7)
        self.assertEqual(scheduler.observe(action_key, 9), 8)
        self.assertIsInstance(scheduler.ewma[action_key], int)
        self.assertEqual(plan.actions[0].duration, 1)
        _ = optional

    def test_prepared_interference_profile_seeds_and_updates_runtime_ewma(self) -> None:
        plan, optional, _required, budget_key = self._plan(with_budget=True)
        assert budget_key is not None
        profile_key = EWMAProfileKey("profile", "shared")
        profile = IntegerEWMAProfile(
            profile_key,
            estimate=4,
            alpha=FixedRatio(1, 4),
            minimum=2,
            maximum=10,
        )
        budget = InterferenceBudgetDesc(budget_key, capacity=10, profile=profile_key)
        profiled = Phase5ControlPlan(
            plan_epoch=plan.plan_epoch,
            placement_plan=plan.placement_plan,
            scheduled_moves=plan.scheduled_moves,
            move_sets=plan.move_sets,
            workload_buckets=plan.workload_buckets,
            shape_buckets=plan.shape_buckets,
            budgets=(budget,),
            actions=plan.actions,
            unknown_bucket_fallback=plan.unknown_bucket_fallback,
            buffer_capacity=plan.buffer_capacity,
            mode=plan.mode,
            profiles=(profile,),
        )
        scheduler = Phase5AdaptiveScheduler(profiled)

        self.assertEqual(scheduler.interference[profile_key], 4)
        self.assertEqual(scheduler.observe_interference(profile_key, 8), 5)
        self.assertEqual(scheduler.interference[profile_key], 5)
        self.assertEqual(
            scheduler.decide(optional).action,
            AdaptiveDecisionKind.RUN,
        )

    def test_authoritative_fallback_reserves_its_declared_resources(self) -> None:
        plan, optional, required, _budget = self._plan()
        fallback = next(
            action
            for action in plan.actions
            if action.kind == CandidateActionKind.AUTHORITATIVE_FALLBACK
        )
        execute = next(
            action
            for action in plan.actions
            if action.kind == CandidateActionKind.EXECUTE
        )
        fallback = CandidateActionDesc(
            key=fallback.key,
            kind=fallback.kind,
            semantic_key=fallback.semantic_key,
            deadline=fallback.deadline,
            duration=fallback.duration,
            buffer_credits=1,
            unknown_bucket=True,
        )
        shed = CandidateActionDesc(
            CandidateActionKey("action", "shed"),
            CandidateActionKind.SHED,
            workload_bucket=execute.workload_bucket,
            shape_bucket=execute.shape_bucket,
            duration=0,
        )
        with_fallback_credit = Phase5ControlPlan(
            plan_epoch=plan.plan_epoch,
            placement_plan=plan.placement_plan,
            scheduled_moves=plan.scheduled_moves,
            move_sets=plan.move_sets,
            workload_buckets=plan.workload_buckets,
            shape_buckets=plan.shape_buckets,
            budgets=plan.budgets,
            actions=(fallback, execute, shed),
            unknown_bucket_fallback=fallback.key,
            buffer_capacity=1,
            mode=plan.mode,
        )
        scheduler = Phase5AdaptiveScheduler(with_fallback_credit)
        scheduler.observe(execute.key, execute.deadline + 1)

        first = scheduler.decide(
            required,
            context=AdmissionContext(available_buffers=1),
            reserve=True,
        )
        second = scheduler.decide(
            optional,
            context=AdmissionContext(available_buffers=1),
            reserve=True,
        )

        self.assertEqual(first.action, AdaptiveDecisionKind.FALLBACK)
        self.assertEqual(scheduler.state.reserved_buffers, 1)
        self.assertEqual(second.action, AdaptiveDecisionKind.DROP)

    def test_authoritative_fallback_must_pass_runtime_feasibility(self) -> None:
        plan, optional, required, budget_key = self._plan(with_budget=True)
        assert budget_key is not None
        fallback = next(action for action in plan.actions if action.unknown_bucket)
        execute = next(
            action
            for action in plan.actions
            if action.kind == CandidateActionKind.EXECUTE
        )
        constrained_fallback = CandidateActionDesc(
            key=fallback.key,
            kind=fallback.kind,
            semantic_key=fallback.semantic_key,
            deadline=2,
            duration=2,
            buffer_credits=1,
            budget_uses=(BudgetUseDesc(budget_key, 1),),
            unknown_bucket=True,
        )
        constrained = Phase5ControlPlan(
            plan_epoch=plan.plan_epoch,
            placement_plan=plan.placement_plan,
            scheduled_moves=plan.scheduled_moves,
            move_sets=plan.move_sets,
            workload_buckets=plan.workload_buckets,
            shape_buckets=plan.shape_buckets,
            budgets=plan.budgets,
            actions=(constrained_fallback, execute),
            unknown_bucket_fallback=constrained_fallback.key,
            buffer_capacity=plan.buffer_capacity,
            mode=plan.mode,
        )
        unknown = WorkItemDesc(
            key=WorkItemKey("item", "unknown-constrained"),
            shape=(99,),
            workload_bucket=optional.workload_bucket,
        )

        for item, context, selected_action in (
            (required, AdmissionContext(available_buffers=0), None),
            (required, AdmissionContext(available_buffers=0), execute.key),
            (
                unknown,
                AdmissionContext(available_buffers=1, deadline_slack=1),
                None,
            ),
            (
                unknown,
                AdmissionContext(
                    available_buffers=1,
                    interference={budget_key: 1},
                ),
                None,
            ),
        ):
            with self.subTest(
                item=item.key,
                context=context,
                selected_action=selected_action,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "no feasible compiler-approved action",
                ):
                    Phase5AdaptiveScheduler(constrained).decide(
                        item,
                        context=context,
                        action=selected_action,
                    )

        deferred = Phase5AdaptiveScheduler(constrained).decide(
            unknown,
            context=AdmissionContext(available_buffers=0),
            allow_defer=True,
        )
        self.assertEqual(deferred.action, AdaptiveDecisionKind.KEEP)


if __name__ == "__main__":
    unittest.main()
