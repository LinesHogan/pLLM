#!/usr/bin/env python3
"""Unit tests for the shared deferred flow-work queue."""

from __future__ import annotations

from collections.abc import Callable
import unittest

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.runtime.scheduling.deferred import (
    DeferredWork,
    DeferredWorkDecision,
    DeferredWorkQueue,
)
from tllm.runtime.scheduling.plan_store import PlanResources
from tllm.workgraph.physical import PhysicalFlowPlan


def _bundle(step: int, *, version: int | None = None) -> PortBundle:
    return PortBundle(
        key=BundleKey(engine_step_id=step, phase="decode", request_id="r", sample_idx=0),
        entries={"value": step},
        version=step if version is None else version,
    )


def _context() -> RuntimeContext:
    import torch

    return RuntimeContext(
        runner=None,
        model=None,
        device=torch.device("cpu"),
        main_stream=None,
        is_compiling=False,
        uses_cudagraph=False,
        event_name="flow:decode",
    )


def _owned_work(
    resources: PlanResources,
    *,
    run: Callable[[PortBundle, RuntimeContext], None],
    flow_id: str = "owned",
) -> DeferredWork:
    resources.retain_deferred_handoff()
    return DeferredWork(
        flow_id=flow_id,
        bundle=_bundle(4),
        context=_context(),
        run=run,
        enqueued_step=4,
        eligible_step=4,
        eligible_hook="sample_tokens.pre",
        deadline_step=5,
        deadline_hook="prepare_inputs.pre",
        plan_resources=resources,
    )


class DeferredWorkQueueUnitTest(unittest.TestCase):
    def test_empty_queue_fast_path(self) -> None:
        queue = DeferredWorkQueue()

        self.assertFalse(queue)
        self.assertEqual(queue.drain_on_bubble(step_id=4, hook="sample_tokens.pre"), 0)
        self.assertEqual(queue.flush_expired(step_id=4, hook="prepare_inputs.pre"), 0)

    def test_ready_work_drains_in_fifo_order(self) -> None:
        seen: list[tuple[str, int]] = []
        queue = DeferredWorkQueue()
        for flow_id, step in (("a", 3), ("b", 4)):
            queue.enqueue(
                DeferredWork(
                    flow_id=flow_id,
                    bundle=_bundle(step),
                    context=_context(),
                    run=lambda bundle, _ctx, fid=flow_id: seen.append((fid, bundle.version)),
                    enqueued_step=step,
                    eligible_step=step,
                    eligible_hook="sample_tokens.pre",
                    deadline_step=step + 1,
                    deadline_hook="prepare_inputs.pre",
                )
            )

        self.assertEqual(queue.drain_on_bubble(step_id=3, hook="execute_model.post"), 0)
        self.assertEqual(queue.drain_on_bubble(step_id=4, hook="sample_tokens.pre"), 2)
        self.assertEqual(seen, [("a", 3), ("b", 4)])
        self.assertFalse(queue)

    def test_bubble_predecessor_events_are_injected_into_runtime_context(self) -> None:
        seen: list[tuple[object, ...]] = []
        queue = DeferredWorkQueue()
        queue.enqueue(
            DeferredWork(
                flow_id="fenced",
                bundle=_bundle(3),
                context=_context(),
                run=lambda _bundle, ctx: seen.append(ctx.predecessor_events),
                enqueued_step=3,
                eligible_step=3,
                eligible_hook="sampler.pre_draw",
                deadline_step=4,
                deadline_hook="prepare_inputs.pre",
            )
        )
        fence = object()

        self.assertEqual(
            queue.drain_on_bubble(
                step_id=3,
                hook="sampler.pre_draw",
                predecessor_events=(fence,),
            ),
            1,
        )
        self.assertEqual(seen, [(fence,)])

    def test_ready_work_behind_not_ready_work_is_not_head_of_line_blocked(self) -> None:
        seen: list[str] = []
        queue = DeferredWorkQueue()
        queue.enqueue(
            DeferredWork(
                flow_id="later",
                bundle=_bundle(4),
                context=_context(),
                run=lambda _bundle, _ctx: seen.append("later"),
                enqueued_step=4,
                eligible_step=4,
                eligible_hook="sample_tokens.post",
                deadline_step=5,
                deadline_hook="prepare_inputs.pre",
            )
        )
        queue.enqueue(
            DeferredWork(
                flow_id="ready",
                bundle=_bundle(4),
                context=_context(),
                run=lambda _bundle, _ctx: seen.append("ready"),
                enqueued_step=4,
                eligible_step=4,
                eligible_hook="sample_tokens.pre",
                deadline_step=5,
                deadline_hook="prepare_inputs.pre",
            )
        )

        self.assertEqual(queue.drain_on_bubble(step_id=4, hook="sample_tokens.pre"), 1)
        self.assertEqual(seen, ["ready"])
        self.assertEqual([work.flow_id for work in queue], ["later"])

    def test_expired_work_behind_unexpired_ready_work_preserves_order(self) -> None:
        seen: list[str] = []
        queue = DeferredWorkQueue()
        for flow_id, deadline_step in (("predecessor", 6), ("expired", 5)):
            queue.enqueue(
                DeferredWork(
                    flow_id=flow_id,
                    bundle=_bundle(4),
                    context=_context(),
                    run=lambda _bundle, _ctx, fid=flow_id: seen.append(fid),
                    enqueued_step=4,
                    eligible_step=4,
                    eligible_hook="sample_tokens.pre",
                    deadline_step=deadline_step,
                    deadline_hook="prepare_inputs.pre",
                )
            )

        self.assertEqual(queue.flush_expired(step_id=5, hook="prepare_inputs.pre"), 2)
        self.assertEqual(seen, ["predecessor", "expired"])
        self.assertFalse(queue)

    def test_next_prepare_forces_borrowed_bundle_before_storage_reuse(self) -> None:
        seen: list[int] = []
        queue = DeferredWorkQueue()
        queue.enqueue(
            DeferredWork(
                flow_id="flow",
                bundle=_bundle(9),
                context=_context(),
                run=lambda bundle, _ctx: seen.append(bundle.version),
                enqueued_step=9,
                eligible_step=9,
                eligible_hook="sample_tokens.pre",
                deadline_step=10,
                deadline_hook="prepare_inputs.pre",
            )
        )

        self.assertEqual(queue.flush_expired(step_id=9, hook="sample_tokens.post"), 0)
        self.assertEqual(queue.flush_expired(step_id=10, hook="prepare_inputs.pre"), 1)
        self.assertEqual(seen, [9])

    def test_budgeted_bubble_drain_leaves_remaining_work_queued(self) -> None:
        seen: list[int] = []
        queue = DeferredWorkQueue()
        for index in range(1, 4):
            queue.enqueue(
                DeferredWork(
                    flow_id=str(index),
                    bundle=_bundle(index),
                    context=_context(),
                    run=lambda bundle, _ctx: seen.append(bundle.version),
                    enqueued_step=0,
                    eligible_step=0,
                    eligible_hook="execute_model.post",
                    deadline_step=1,
                    deadline_hook="prepare_inputs.pre",
                )
            )

        self.assertEqual(queue.drain_on_bubble(step_id=0, hook="execute_model.post", max_items=2), 2)
        self.assertEqual(seen, [1, 2])
        self.assertEqual(queue.depth, 1)
        self.assertEqual(queue.flush_all(), 1)
        self.assertEqual(seen, [1, 2, 3])

    def test_bubble_failure_keeps_failed_work_and_followers_queued(self) -> None:
        attempts = 0
        seen: list[str] = []
        queue = DeferredWorkQueue()

        def run_first(_bundle, _ctx) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("retryable")
            seen.append("first")

        for flow_id, run in (
            ("first", run_first),
            ("second", lambda _bundle, _ctx: seen.append("second")),
        ):
            queue.enqueue(
                DeferredWork(
                    flow_id=flow_id,
                    bundle=_bundle(4),
                    context=_context(),
                    run=run,
                    enqueued_step=4,
                    eligible_step=4,
                    eligible_hook="sample_tokens.pre",
                    deadline_step=5,
                    deadline_hook="prepare_inputs.pre",
                )
            )

        with self.assertRaisesRegex(RuntimeError, "retryable"):
            queue.drain_on_bubble(step_id=4, hook="sample_tokens.pre")
        self.assertEqual([work.flow_id for work in queue], ["first", "second"])

        self.assertEqual(queue.drain_on_bubble(step_id=4, hook="sample_tokens.pre"), 2)
        self.assertEqual(seen, ["first", "second"])

    def test_deadline_failure_keeps_failed_work_and_followers_queued(self) -> None:
        attempts = 0
        seen: list[str] = []
        queue = DeferredWorkQueue()

        def run_first(_bundle, _ctx) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("retryable deadline")
            seen.append("first")

        for flow_id, run in (
            ("first", run_first),
            ("second", lambda _bundle, _ctx: seen.append("second")),
        ):
            queue.enqueue(
                DeferredWork(
                    flow_id=flow_id,
                    bundle=_bundle(4),
                    context=_context(),
                    run=run,
                    enqueued_step=4,
                    eligible_step=4,
                    eligible_hook="sample_tokens.pre",
                    deadline_step=5,
                    deadline_hook="prepare_inputs.pre",
                )
            )

        with self.assertRaisesRegex(RuntimeError, "retryable deadline"):
            queue.flush_expired(step_id=5, hook="prepare_inputs.pre")
        self.assertEqual([work.flow_id for work in queue], ["first", "second"])

        self.assertEqual(queue.flush_expired(step_id=5, hook="prepare_inputs.pre"), 2)
        self.assertEqual(seen, ["first", "second"])

    def test_flush_all_failure_keeps_failed_work_and_followers_queued(self) -> None:
        attempts = 0
        seen: list[str] = []
        queue = DeferredWorkQueue()

        def run_first(_bundle, _ctx) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("retryable flush")
            seen.append("first")

        for flow_id, run in (
            ("first", run_first),
            ("second", lambda _bundle, _ctx: seen.append("second")),
        ):
            queue.enqueue(
                DeferredWork(
                    flow_id=flow_id,
                    bundle=_bundle(4),
                    context=_context(),
                    run=run,
                    enqueued_step=4,
                    eligible_step=4,
                    eligible_hook="sample_tokens.pre",
                    deadline_step=5,
                    deadline_hook="prepare_inputs.pre",
                )
            )

        with self.assertRaisesRegex(RuntimeError, "retryable flush"):
            queue.flush_all()
        self.assertEqual([work.flow_id for work in queue], ["first", "second"])

        self.assertEqual(queue.flush_all(), 2)
        self.assertEqual(seen, ["first", "second"])

    def test_successful_run_releases_owner_after_callback(self) -> None:
        resources = PlanResources(plan=PhysicalFlowPlan(plan_epoch=1, nodes=()))
        queue = DeferredWorkQueue()
        callback_reusability: list[bool] = []
        queue.enqueue(
            _owned_work(
                resources,
                run=lambda _bundle, _ctx: callback_reusability.append(
                    resources.reusable
                ),
            )
        )

        self.assertFalse(resources.reusable)
        result = queue.drain_selected(
            step_id=4,
            hook="sample_tokens.pre",
            select=lambda _work, _must_resolve: DeferredWorkDecision.RUN,
        )

        self.assertEqual(result.ran, 1)
        self.assertEqual(callback_reusability, [False])
        self.assertTrue(resources.reusable)
        self.assertFalse(queue)

    def test_failed_run_remains_queued_and_retains_owner(self) -> None:
        resources = PlanResources(plan=PhysicalFlowPlan(plan_epoch=1, nodes=()))
        queue = DeferredWorkQueue()

        def fail(_bundle: PortBundle, _ctx: RuntimeContext) -> None:
            self.assertFalse(resources.reusable)
            raise RuntimeError("retryable owned callback")

        work = _owned_work(resources, run=fail)
        queue.enqueue(work)

        with self.assertRaisesRegex(RuntimeError, "retryable owned callback"):
            queue.drain_selected(
                step_id=4,
                hook="sample_tokens.pre",
                select=lambda _work, _must_resolve: DeferredWorkDecision.RUN,
            )

        self.assertEqual(tuple(queue), (work,))
        self.assertFalse(resources.reusable)
        queue.clear()
        self.assertTrue(resources.reusable)

    def test_drop_releases_retained_owner(self) -> None:
        resources = PlanResources(plan=PhysicalFlowPlan(plan_epoch=1, nodes=()))
        queue = DeferredWorkQueue()
        callbacks: list[str] = []
        queue.enqueue(
            _owned_work(
                resources,
                run=lambda _bundle, _ctx: callbacks.append("run"),
            )
        )

        self.assertFalse(resources.reusable)
        result = queue.drain_selected(
            step_id=4,
            hook="sample_tokens.pre",
            select=lambda _work, _must_resolve: DeferredWorkDecision.DROP,
        )

        self.assertEqual(result.dropped, 1)
        self.assertEqual(callbacks, [])
        self.assertTrue(resources.reusable)
        self.assertFalse(queue)

    def test_clear_releases_every_retained_owner_reference(self) -> None:
        resources = PlanResources(plan=PhysicalFlowPlan(plan_epoch=1, nodes=()))
        queue = DeferredWorkQueue()
        for flow_id in ("owned-a", "owned-b"):
            queue.enqueue(
                _owned_work(
                    resources,
                    flow_id=flow_id,
                    run=lambda _bundle, _ctx: None,
                )
            )

        self.assertFalse(resources.reusable)
        queue.clear()

        self.assertFalse(queue)
        self.assertTrue(resources.reusable)

    def test_invalid_deadline_and_version_are_rejected_at_enqueue(self) -> None:
        queue = DeferredWorkQueue()
        with self.assertRaisesRegex(ValueError, "deadline precedes enqueue"):
            queue.enqueue(
                DeferredWork(
                    flow_id="bad",
                    bundle=_bundle(5),
                    context=_context(),
                    run=lambda _bundle, _ctx: None,
                    enqueued_step=5,
                    eligible_step=5,
                    eligible_hook="sample_tokens.pre",
                    deadline_step=4,
                    deadline_hook="prepare_inputs.pre",
                )
            )
        with self.assertRaisesRegex(ValueError, "positive bundle version"):
            queue.enqueue(
                DeferredWork(
                    flow_id="legacy",
                    bundle=_bundle(5, version=0),
                    context=_context(),
                    run=lambda _bundle, _ctx: None,
                    enqueued_step=5,
                    eligible_step=5,
                    eligible_hook="sample_tokens.pre",
                    deadline_step=6,
                    deadline_hook="prepare_inputs.pre",
                )
            )


if __name__ == "__main__":
    unittest.main()
