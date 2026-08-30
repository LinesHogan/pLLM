#!/usr/bin/env python3
"""Unit contracts for adapter-neutral request lifecycle expansion."""

from __future__ import annotations

import unittest
from unittest import mock

from tllm.backends.protocol import OverlayBackendState, RunnerBinding
from tllm.contracts.request_identity import RequestFinishedBatch, RequestIdentity
from tllm.runtime.native_request_finish import NativeRequestFinishCleanup
from tllm.runtime.request_lifecycle import (
    DuplicateRequestFinishError,
    RequestLifecycleError,
    RequestLifecycleLedger,
    UnknownRequestAliasError,
    finish_registered_requests,
    observe_request_aliases,
    register_request_alias,
)
from tllm.runtime.state import ResidualRuntimeState


class _CleanupService:
    def __init__(self, state: ResidualRuntimeState, *, fail_confirm: bool = False) -> None:
        self.state = state
        self.fail_confirm = fail_confirm
        self.events: list[str] = []

    def stage(self, batch: RequestFinishedBatch) -> int:
        _ = batch
        self.events.append("stage")
        self._assert_mapping_live()
        return 1

    def confirm(self, batch: RequestFinishedBatch, staged_ports: int) -> None:
        _ = batch
        self.events.append(f"confirm:{staged_ports}")
        self._assert_mapping_live()
        if self.fail_confirm:
            raise RuntimeError("signed cleanup failed")

    def _assert_mapping_live(self) -> None:
        if "request" not in self.state.reqid_to_promptidx:
            raise AssertionError("mapping was removed before lifecycle commit")


class _FailingOverlayDriver:
    def __init__(self) -> None:
        self.events: list[str] = []

    def consume_parameter_overlay_finished_batch(self, **_kwargs: object) -> int:
        self.events.append("consume")
        return 1

    def synchronize_parameter_overlays(self, **_kwargs: object) -> int:
        self.events.append("sync")
        raise RuntimeError("overlay sync failed")


class RequestLifecycleLedgerUnitTest(unittest.TestCase):
    def test_parent_finish_expands_parallel_children_and_enginecore_suffix(self) -> None:
        ledger = RequestLifecycleLedger()
        parent = RequestIdentity("parent_with_underscores", 4, 0)
        child = RequestIdentity("parent_with_underscores", 4, 1)
        ledger.register_alias(raw_request_id="parent_with_underscores", identity=parent)
        ledger.register_alias(raw_request_id="0_parent_with_underscores", identity=parent)
        ledger.register_alias(raw_request_id="1_parent_with_underscores", identity=child)

        receipt = ledger.prepare_finish(("parent_with_underscores-a1B2c3D4",))

        self.assertEqual(
            tuple(trigger.raw_request_id for trigger in receipt.triggers),
            ("parent_with_underscores-a1B2c3D4",),
        )
        self.assertEqual(
            tuple(alias.raw_request_id for alias in receipt.aliases),
            (
                "parent_with_underscores",
                "0_parent_with_underscores",
                "1_parent_with_underscores",
            ),
        )
        self.assertEqual(receipt.batch.requests, (parent, child))
        self.assertEqual(len(ledger.active_aliases), 3)
        ledger.commit_finish(receipt)
        self.assertEqual(ledger.active_aliases, ())

    def test_child_finish_expands_the_registered_parent_family(self) -> None:
        ledger = RequestLifecycleLedger()
        first = RequestIdentity("request", 3, 0)
        second = RequestIdentity("request", 3, 1)
        ledger.register_alias(raw_request_id="0_request", identity=first)
        ledger.register_alias(raw_request_id="1_request", identity=second)

        receipt = ledger.prepare_finish(("1_request-Z9Y8X7W6",))

        self.assertEqual(
            receipt.triggers[0].alias.raw_request_id,
            "1_request",
        )
        self.assertEqual(receipt.batch.requests, (first, second))
        self.assertEqual(len(ledger.active_aliases), 2)
        ledger.commit_finish(receipt)
        self.assertEqual(ledger.active_aliases, ())

    def test_finish_receipt_preserves_all_aliases_before_terminal_removal(self) -> None:
        ledger = RequestLifecycleLedger()
        first = RequestIdentity("shared", 1, 0)
        second = RequestIdentity("shared", 1, 1)
        ledger.register_alias(raw_request_id="shared", identity=first)
        ledger.register_alias(raw_request_id="0_shared", identity=first)
        ledger.register_alias(raw_request_id="1_shared", identity=second)

        receipt = ledger.prepare_finish(("0_shared", "1_shared"))

        self.assertEqual(
            tuple(trigger.alias.raw_request_id for trigger in receipt.triggers),
            ("0_shared", "1_shared"),
        )
        self.assertEqual(len(receipt.aliases), 3)
        self.assertEqual(receipt.batch.requests, (first, second))
        self.assertEqual(len(ledger.active_aliases), 3)
        ledger.commit_finish(receipt)
        self.assertEqual(ledger.active_aliases, ())

    def test_unknown_or_duplicate_finish_has_no_fallback(self) -> None:
        ledger = RequestLifecycleLedger()
        identity = RequestIdentity("known", 0, 0)
        ledger.register_alias(raw_request_id="known", identity=identity)

        with self.assertRaises(UnknownRequestAliasError):
            ledger.prepare_finish(("unknown",))
        self.assertEqual(ledger.active_aliases[0].identity, identity)

        receipt = ledger.prepare_finish(("known",))
        ledger.commit_finish(receipt)
        with self.assertRaises(DuplicateRequestFinishError):
            ledger.prepare_finish(("known",))
        with self.assertRaises(DuplicateRequestFinishError):
            ledger.prepare_finish(("known-a1B2c3D4",))

    def test_registration_requires_one_stable_resolved_mapping(self) -> None:
        ledger = RequestLifecycleLedger()
        identity = RequestIdentity("request", 2, 0)
        same = ledger.register_alias(raw_request_id="request", identity=identity)
        self.assertIs(
            same,
            ledger.register_alias(raw_request_id="request", identity=identity),
        )
        with self.assertRaises(RequestLifecycleError):
            ledger.register_alias(
                raw_request_id="request",
                identity=RequestIdentity("request", 2, 1),
            )
        with self.assertRaises(ValueError):
            ledger.register_alias(
                raw_request_id="unknown",
                identity=RequestIdentity("unknown", -1, 0),
            )

    def test_finish_batch_rejects_a_repeated_raw_event(self) -> None:
        ledger = RequestLifecycleLedger()
        ledger.register_alias(
            raw_request_id="request",
            identity=RequestIdentity("request", 0, 0),
        )

        with self.assertRaises(DuplicateRequestFinishError):
            ledger.prepare_finish(("request", "request"))
        self.assertEqual(len(ledger.active_aliases), 1)

    def test_finish_mappings_survive_abort_and_only_commit_once(self) -> None:
        ledger = RequestLifecycleLedger()
        identity = RequestIdentity("request", 0, 0)
        ledger.register_alias(raw_request_id="request", identity=identity)
        receipt = ledger.prepare_finish(("request",))

        self.assertEqual(ledger.active_aliases[0].identity, identity)
        with self.assertRaises(RequestLifecycleError):
            ledger.prepare_finish(("request",))
        ledger.abort_finish(receipt)
        self.assertEqual(ledger.active_aliases[0].identity, identity)

        replacement = ledger.prepare_finish(("request",))
        ledger.commit_finish(replacement)
        with self.assertRaises(RequestLifecycleError):
            ledger.commit_finish(replacement)

    def test_parent_assigned_finish_id_ignores_local_counter_history(self) -> None:
        ledger = RequestLifecycleLedger()
        identity = RequestIdentity("request", 0, 0)
        ledger.register_alias(raw_request_id="request", identity=identity)
        local = ledger.prepare_finish(("request",))
        ledger.abort_finish(local)

        assigned = ledger.prepare_finish(("request",), finish_id=987654321)

        self.assertEqual(assigned.finish_id, 987654321)

    def test_empty_finish_batch_is_not_a_cleanup_transaction(self) -> None:
        ledger = RequestLifecycleLedger()
        with self.assertRaises(ValueError):
            ledger.prepare_finish(())

    def test_generic_finish_commits_before_mapping_removal(self) -> None:
        state = ResidualRuntimeState(
            reqid_to_promptidx={"request": 2},
            reqid_to_sampleidx={"request": 1},
        )
        register_request_alias(state, "request")
        cleanup = _CleanupService(state)

        outcome = finish_registered_requests(
            runtime=state,
            raw_request_ids=("request",),
            cleanup=cleanup,
        )

        self.assertEqual(cleanup.events, ["stage", "confirm:1"])
        self.assertEqual(
            outcome.receipt.batch.requests,
            (RequestIdentity("request", 2, 1),),
        )
        self.assertEqual(state.reqid_to_promptidx, {})
        self.assertEqual(state.reqid_to_sampleidx, {})
        self.assertEqual(state.request_lifecycle.active_aliases, ())

    def test_generic_finish_failure_stays_pending_without_releasing_mapping(self) -> None:
        state = ResidualRuntimeState(reqid_to_promptidx={"request": 2})
        register_request_alias(state, "request")

        with self.assertRaisesRegex(RuntimeError, "signed cleanup failed"):
            finish_registered_requests(
                runtime=state,
                raw_request_ids=("request",),
                cleanup=_CleanupService(state, fail_confirm=True),
            )

        self.assertEqual(state.reqid_to_promptidx, {"request": 2})
        self.assertEqual(
            state.request_lifecycle.active_aliases[0].identity,
            RequestIdentity("request", 2, 0),
        )
        self.assertIsNotNone(state.request_lifecycle.pending_finish)

    def test_overlay_sync_failure_keeps_pending_receipt_after_cleanup_begins(self) -> None:
        state = ResidualRuntimeState(reqid_to_promptidx={"request": 2})
        register_request_alias(state, "request")
        driver = _FailingOverlayDriver()
        binding = mock.create_autospec(RunnerBinding, instance=True)
        binding.driver = driver
        binding.runner = object()
        binding.parameter_overlay_state = OverlayBackendState()

        with self.assertRaisesRegex(RuntimeError, "overlay sync failed"):
            finish_registered_requests(
                runtime=state,
                raw_request_ids=("request",),
                cleanup=NativeRequestFinishCleanup(state, binding),
            )

        self.assertEqual(driver.events, ["consume", "sync"])
        self.assertEqual(state.reqid_to_promptidx, {"request": 2})
        self.assertIsNotNone(state.request_lifecycle.pending_finish)

    def test_adapter_observation_registers_parent_child_and_internal_aliases(self) -> None:
        state = ResidualRuntimeState(reqid_to_promptidx={"parent": 4})

        aliases = observe_request_aliases(
            state,
            (
                "parent",
                "0_parent",
                "1_parent",
                "1_parent-a1B2c3D4",
            ),
        )

        self.assertEqual(
            tuple(alias.identity for alias in aliases),
            (
                RequestIdentity("parent", 4, 0),
                RequestIdentity("parent", 4, 0),
                RequestIdentity("parent", 4, 1),
                RequestIdentity("parent", 4, 1),
            ),
        )
        receipt = state.request_lifecycle.prepare_finish(("parent",))
        self.assertEqual(
            receipt.batch.requests,
            (RequestIdentity("parent", 4, 0), RequestIdentity("parent", 4, 1)),
        )


if __name__ == "__main__":
    unittest.main()
