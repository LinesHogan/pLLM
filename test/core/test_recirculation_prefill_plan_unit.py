"""Closed Rust contract for one Recirculation prefill binding batch."""

from __future__ import annotations

import json
import unittest

import pytest

from tllm import _core


_ORDERED_ROW_FUSION_POLICY = (
    "max_rows=2;internalized_edges=kv_overwrite_read"
)


def _plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "prefill",
        "layer_count": 3,
        "total_rows": 3,
        "requests": [
            {
                "request_id": "req-a",
                "request_ordinal": 0,
                "scheduled_tokens": 2,
                "prompt_tokens": 5,
                "computed_tokens": 1,
                "packed_row_start": 0,
                "packed_row_end": 2,
                "packed_row_indices": [0, 1],
                "token_position_start": 1,
                "token_position_end": 3,
                "token_positions": [1, 2],
            },
            {
                "request_id": "req-b",
                "request_ordinal": 1,
                "scheduled_tokens": 1,
                "prompt_tokens": 2,
                "computed_tokens": 1,
                "packed_row_start": 2,
                "packed_row_end": 3,
                "packed_row_indices": [2],
                "token_position_start": 1,
                "token_position_end": 2,
                "token_positions": [1],
            },
        ],
    }


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class RecirculationPrefillPlanUnitTest(unittest.TestCase):
    def test_rust_compiles_closed_typed_recirculation_operation_groups(self) -> None:
        schedule = _core.compile_recirculation_prefill_schedule(
            token_count=2,
            layer_count=4,
            source_layer=2,
            destination_layer=0,
            ordered_row_fusion_policy=_ORDERED_ROW_FUSION_POLICY,
        )

        self.assertEqual(schedule.kind, "wavefront")
        self.assertEqual(schedule.source_layer, 2)
        self.assertEqual(schedule.destination_layer, 0)
        operations = [
            operation
            for group in schedule.groups
            for operation in group.operations
        ]
        self.assertEqual(
            sorted(operation.ordinal for operation in operations),
            list(range(len(operations))),
        )
        self.assertTrue(
            all(
                operation.kind
                in {
                    _core.RecirculationPrefillOpKind.Normal,
                    _core.RecirculationPrefillOpKind.Mix,
                    _core.RecirculationPrefillOpKind.Recurrent,
                }
                for operation in operations
            )
        )
        mixes = [
            operation
            for operation in operations
            if operation.kind == _core.RecirculationPrefillOpKind.Mix
        ]
        self.assertEqual(len(mixes), 2)
        for operation in mixes:
            self.assertIn(
                operation.mix_source_ordinal,
                operation.data_predecessor_ordinals,
            )
            self.assertIn(
                operation.mix_destination_ordinal,
                operation.data_predecessor_ordinals,
            )
            self.assertIn(
                operation.mix_final_normal_ordinal,
                operation.data_predecessor_ordinals,
            )
        by_ordinal = {operation.ordinal: operation for operation in operations}
        final_ops = [by_ordinal[ordinal] for ordinal in schedule.final_normal_ordinals]
        self.assertEqual(
            [operation.kind for operation in final_ops],
            [_core.RecirculationPrefillOpKind.Normal] * 2,
        )
        self.assertEqual([operation.token for operation in final_ops], [0, 1])
        self.assertEqual([operation.layer for operation in final_ops], [3, 3])
        self.assertEqual(schedule.retained_ordinals, schedule.final_normal_ordinals)
        for group_index, group in enumerate(schedule.groups):
            self.assertEqual(group.ordinal, group_index)
            if group.kind == _core.RecirculationPrefillGroupKind.ModelLayer:
                self.assertIsNotNone(group.model_layer)
                self.assertTrue(
                    all(
                        operation.kind
                        in {
                            _core.RecirculationPrefillOpKind.Normal,
                            _core.RecirculationPrefillOpKind.Recurrent,
                        }
                        and operation.layer == group.model_layer
                        for operation in group.operations
                    )
                )
            else:
                self.assertEqual(group.kind, _core.RecirculationPrefillGroupKind.Mix)
                self.assertIsNone(group.model_layer)
                self.assertEqual(len(group.operations), 1)
                self.assertEqual(
                    group.operations[0].kind, _core.RecirculationPrefillOpKind.Mix
                )
        positions = {
            operation.ordinal: (group_index, operation_index)
            for group_index, group in enumerate(schedule.groups)
            for operation_index, operation in enumerate(group.operations)
        }
        for operation in operations:
            for predecessor in operation.data_predecessor_ordinals:
                self.assertLess(positions[predecessor], positions[operation.ordinal])
            if operation.kind == _core.RecirculationPrefillOpKind.Mix:
                self.assertIsNone(operation.model_input_kind)
                self.assertIsNone(operation.model_input_ordinal)
            elif operation.kind == _core.RecirculationPrefillOpKind.Normal and operation.layer == 0:
                self.assertEqual(operation.model_input_kind, _core.RecirculationInputKind.Seed)
                self.assertIsNone(operation.model_input_ordinal)
            else:
                self.assertEqual(operation.model_input_kind, _core.RecirculationInputKind.Ordinal)
                self.assertEqual(
                    operation.model_input_ordinal,
                    operation.data_predecessor_ordinals[0],
                )
        expected_releases = {
            operation.ordinal
            for group in schedule.groups
            for operation in group.operations
            if operation.ordinal not in schedule.retained_ordinals
        }
        actual_releases = {
            ordinal for group in schedule.groups for ordinal in group.release_after
        }
        self.assertEqual(actual_releases, expected_releases)
        release_group = {
            ordinal: group_index
            for group_index, group in enumerate(schedule.groups)
            for ordinal in group.release_after
        }
        for operation in operations:
            if operation.ordinal in schedule.retained_ordinals:
                self.assertNotIn(operation.ordinal, release_group)
                continue
            expected_last_use = max(
                [positions[operation.ordinal][0]]
                + [
                    positions[consumer.ordinal][0]
                    for consumer in operations
                    if operation.ordinal in consumer.data_predecessor_ordinals
                ]
            )
            self.assertEqual(release_group[operation.ordinal], expected_last_use)
        self.assertFalse(hasattr(operations[0], "op_id"))
        with pytest.raises(AttributeError):
            operations[0].layer = 99
        copied_groups = schedule.groups
        copied_groups.clear()
        self.assertNotEqual(schedule.groups, [])

    def test_rust_schedule_rejects_invalid_shape_or_layer_contract(self) -> None:
        for args, message in (
            ((0, 4, 2, 0, "none"), "token_count"),
            ((1, 0, 0, 0, "none"), "layer_count"),
            ((1, 4, 0, 0, "none"), "destination_layer"),
            ((1, 4, 4, 0, "none"), "source_layer"),
            ((1, 4, -5, 0, "none"), "source_layer"),
        ):
            with self.subTest(message=message):
                with pytest.raises(ValueError, match=message):
                    _core.compile_recirculation_prefill_schedule(*args)

    def test_rust_rejects_noncanonical_ordered_row_fusion_policy(self) -> None:
        for policy in (
            "true",
            "max_rows=02;internalized_edges=kv_overwrite_read",
            "max_rows=1;internalized_edges=kv_overwrite_read",
            "max_rows=2;internalized_edges=kv_overwrite_read,control",
            "max_rows=2;internalized_edges=",
        ):
            with self.subTest(policy=policy):
                with pytest.raises(ValueError, match="ordered-row fusion policy"):
                    _core.compile_recirculation_prefill_schedule(
                        2,
                        4,
                        2,
                        0,
                        policy,
                    )

    def test_builder_derives_the_entire_typed_prefill_plan_in_rust(self) -> None:
        plan = _core.build_recirculation_prefill_plan(
            ["req-a", "req-b"],
            [2, 1],
            [5, 2],
            [1, 1],
            3,
        )

        self.assertEqual(plan.schema_version, 1)
        self.assertEqual(plan.phase, "prefill")
        self.assertEqual(plan.layer_count, 3)
        self.assertEqual(plan.total_rows, 3)
        self.assertEqual(
            [request.request_id for request in plan.requests], ["req-a", "req-b"]
        )
        self.assertEqual(plan.requests[0].packed_row_indices, [0, 1])
        self.assertEqual(plan.requests[0].token_positions, [1, 2])
        self.assertEqual(plan.requests[1].packed_row_indices, [2])
        self.assertEqual(plan.requests[1].token_positions, [1])
        self.assertEqual(
            _core.normalize_recirculation_prefill_plan(
                json.dumps(
                    {
                        "schema_version": plan.schema_version,
                        "phase": plan.phase,
                        "layer_count": plan.layer_count,
                        "total_rows": plan.total_rows,
                        "requests": [
                            {
                                "request_id": request.request_id,
                                "request_ordinal": request.request_ordinal,
                                "scheduled_tokens": request.scheduled_tokens,
                                "prompt_tokens": request.prompt_tokens,
                                "computed_tokens": request.computed_tokens,
                                "packed_row_start": request.packed_row_start,
                                "packed_row_end": request.packed_row_end,
                                "packed_row_indices": request.packed_row_indices,
                                "token_position_start": request.token_position_start,
                                "token_position_end": request.token_position_end,
                                "token_positions": request.token_positions,
                            }
                            for request in plan.requests
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            _json(_plan()),
        )

    def test_builder_returns_validated_noop_for_decode_only_batch(self) -> None:
        plan = _core.build_recirculation_prefill_plan(
            ["req-a"], [2], [3], [5], 3
        )

        self.assertEqual(plan.total_rows, 0)
        self.assertEqual(plan.requests, [])

    def test_builder_rejects_mixed_and_invalid_scheduler_columns(self) -> None:
        cases = (
            ((["req-a"], [2], [2], [1], 3), "mix"),
            ((["req-a", "req-b"], [1, 1], [2, 2], [1, 2], 3), "mix"),
            ((["req-a", "req-a"], [1, 1], [2, 2], [1, 1], 3), "request_id"),
            ((["req-a"], [1], [2], [1], 0), "layer_count"),
            ((["req-a"], [1], [2, 2], [1], 3), "equal lengths"),
        )
        for args, message in cases:
            with self.subTest(message=message):
                with pytest.raises(ValueError, match=message):
                    _core.build_recirculation_prefill_plan(*args)

    def test_normalizes_one_complete_prefill_batch_to_canonical_json(self) -> None:
        plan = _plan()
        raw = json.dumps(plan, separators=(", ", ": "), ensure_ascii=True)

        canonical = _core.normalize_recirculation_prefill_plan(raw)

        self.assertEqual(canonical, _json(plan))
        self.assertEqual(
            _core.normalize_recirculation_prefill_plan(canonical), canonical
        )

    def test_parses_the_same_validated_plan_as_a_read_only_typed_view(self) -> None:
        parsed = _core.parse_recirculation_prefill_plan(_json(_plan()))

        self.assertEqual(parsed.schema_version, 1)
        self.assertEqual(parsed.phase, "prefill")
        self.assertEqual(parsed.layer_count, 3)
        self.assertEqual(parsed.total_rows, 3)
        self.assertEqual(len(parsed.requests), 2)
        first = parsed.requests[0]
        self.assertEqual(first.request_id, "req-a")
        self.assertEqual(first.request_ordinal, 0)
        self.assertEqual(first.scheduled_tokens, 2)
        self.assertEqual(first.packed_row_indices, [0, 1])
        self.assertEqual(first.token_positions, [1, 2])
        with pytest.raises(AttributeError):
            parsed.total_rows = 4
        with pytest.raises(AttributeError):
            first.request_id = "changed"

    def test_allows_empty_zero_row_prefill_as_a_decode_no_op_only(self) -> None:
        no_op = {
            "schema_version": 1,
            "phase": "prefill",
            "layer_count": 3,
            "total_rows": 0,
            "requests": [],
        }

        canonical = _core.normalize_recirculation_prefill_plan(_json(no_op))
        parsed = _core.parse_recirculation_prefill_plan(canonical)
        self.assertEqual(canonical, _json(no_op))
        self.assertEqual(parsed.total_rows, 0)
        self.assertEqual(parsed.requests, [])

        nonzero = dict(no_op, total_rows=1)
        with pytest.raises(ValueError, match="empty.*total_rows"):
            _core.parse_recirculation_prefill_plan(_json(nonzero))

    def test_rejects_mixed_or_non_prefill_rows(self) -> None:
        mixed_phase = _plan()
        mixed_phase["phase"] = "decode"
        with pytest.raises(ValueError, match="phase"):
            _core.normalize_recirculation_prefill_plan(_json(mixed_phase))

        decode_row = _plan()
        requests = decode_row["requests"]
        assert isinstance(requests, list)
        first = requests[0]
        assert isinstance(first, dict)
        first["prompt_tokens"] = 2
        with pytest.raises(ValueError, match="prefill"):
            _core.normalize_recirculation_prefill_plan(_json(decode_row))

    def test_rejects_duplicate_or_inconsistent_packed_rows(self) -> None:
        duplicate_request = _plan()
        requests = duplicate_request["requests"]
        assert isinstance(requests, list)
        second = requests[1]
        assert isinstance(second, dict)
        second["request_id"] = "req-a"
        with pytest.raises(ValueError, match="request_id"):
            _core.normalize_recirculation_prefill_plan(_json(duplicate_request))

        duplicate_row = _plan()
        requests = duplicate_row["requests"]
        assert isinstance(requests, list)
        second = requests[1]
        assert isinstance(second, dict)
        second["packed_row_start"] = 1
        second["packed_row_end"] = 2
        second["packed_row_indices"] = [1]
        with pytest.raises(ValueError, match="packed"):
            _core.normalize_recirculation_prefill_plan(_json(duplicate_row))

        positions = _plan()
        requests = positions["requests"]
        assert isinstance(requests, list)
        first = requests[0]
        assert isinstance(first, dict)
        first["token_positions"] = [1, 3]
        with pytest.raises(ValueError, match="token"):
            _core.normalize_recirculation_prefill_plan(_json(positions))

    def test_rejects_invalid_counts_ordinals_and_closed_wire_drift(self) -> None:
        for mutate, message in (
            (lambda plan: plan.__setitem__("layer_count", 0), "layer_count"),
            (lambda plan: plan.__setitem__("total_rows", 4), "total_rows"),
            (
                lambda plan: plan["requests"][1].__setitem__("request_ordinal", 2),  # type: ignore[index,union-attr]
                "ordinal",
            ),
            (
                lambda plan: plan["requests"][0].__setitem__("packed_row_end", 3),  # type: ignore[index,union-attr]
                "packed",
            ),
        ):
            plan = _plan()
            mutate(plan)
            with self.subTest(message=message):
                with pytest.raises(ValueError, match=message):
                    _core.normalize_recirculation_prefill_plan(_json(plan))

        unknown = _plan()
        unknown["unexpected"] = True
        with pytest.raises(ValueError, match="unknown field"):
            _core.normalize_recirculation_prefill_plan(_json(unknown))


if __name__ == "__main__":
    unittest.main()
