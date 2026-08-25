from __future__ import annotations

import json
import unittest
from unittest import mock

import torch

from tllm import _core
from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.contracts.shape_specialization import (
    RegionShapeSpecializationInput,
    RegionShapeSpecializationRequest,
)
from tllm.runtime.runtime import TllmRuntime
from tllm.schedule import CompileOptions
from tllm.workload import Between, provided, task, workload


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _request() -> RegionShapeSpecializationRequest:
    return RegionShapeSpecializationRequest(
        base_cache_key=_DIGEST_A,
        base_signature=_DIGEST_B,
        observed_plan_epoch=4,
        region_id="esamp.forward",
        artifact_key=_DIGEST_C,
        inputs=(
            RegionShapeSpecializationInput(
                binding_id="hidden",
                shape=(8, 896),
                dtype="torch.bfloat16",
                device="cuda:0",
            ),
        ),
    )


class RegionShapeSpecializationContractTest(unittest.TestCase):
    def test_round_trip_is_stable_json_and_dedup_is_content_addressed(self) -> None:
        request = _request()

        self.assertEqual(RegionShapeSpecializationRequest.from_json(request.to_json()), request)
        self.assertEqual(request.to_json(), RegionShapeSpecializationRequest.from_wire(request.to_wire()).to_json())
        self.assertEqual(len(request.dedup_key), 64)

    def test_rejects_unknown_or_missing_wire_fields(self) -> None:
        payload = _request().to_wire()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "invalid fields"):
            RegionShapeSpecializationRequest.from_wire(payload)

    def test_rejects_duplicate_json_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            RegionShapeSpecializationRequest.from_json(
                '{"version":1,"version":1}'
            )

        payload = _request().to_wire()
        del payload["region_id"]
        with self.assertRaisesRegex(ValueError, "invalid fields"):
            RegionShapeSpecializationRequest.from_wire(payload)

    def test_inputs_are_unique_and_binding_id_ordered(self) -> None:
        first = RegionShapeSpecializationInput("a", (1,), "torch.float32", "cpu")
        second = RegionShapeSpecializationInput("b", (1,), "torch.float32", "cpu")
        with self.assertRaisesRegex(ValueError, "ordered"):
            RegionShapeSpecializationRequest(
                _DIGEST_A, _DIGEST_B, 1, "region", _DIGEST_C, (second, first)
            )
        with self.assertRaisesRegex(ValueError, "ordered"):
            RegionShapeSpecializationRequest(
                _DIGEST_A, _DIGEST_B, 1, "region", _DIGEST_C, (first, first)
            )

    def test_rejects_non_json_runtime_values(self) -> None:
        with self.assertRaises(TypeError):
            RegionShapeSpecializationInput("input", (1,), "torch.float32", object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RegionShapeSpecializationRequest(
                "not-a-digest", _DIGEST_B, 1, "region", _DIGEST_C,
                (RegionShapeSpecializationInput("input", (), "torch.float32", "cpu"),),
            )

    def test_rust_normalizer_owns_closed_json_semantics(self) -> None:
        request = _request()
        self.assertEqual(
            _core.normalize_region_shape_specialization_request(request.to_json()),
            request.to_json(),
        )
        _core.validate_region_shape_specialization_request(request.to_json())

        invalid = request.to_json().replace(
            '"observed_plan_epoch":4', '"observed_plan_epoch":0'
        )
        with self.assertRaisesRegex(ValueError, "observed_plan_epoch"):
            _core.validate_region_shape_specialization_request(invalid)

    def test_runtime_recompiles_matching_request_with_a_new_epoch(self) -> None:
        @workload(id="shape.specialization.runtime")
        class Auxiliary:
            @task(
                inputs={"value": provided("value")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: torch.Tensor) -> None:
                _ = value

        runtime = TllmRuntime(
            workloads=(Auxiliary(),),
            workload_memory_budget_bytes=1 << 20,
            workload_provided_inputs={"value": lambda _step: torch.ones((2, 4))},
        )
        timeline = export_main_workload_timeline(
            BackendCapabilities(
                backend_id="shape-specialization-test",
                runner_family="mrv2",
                sampler_stages=frozenset(),
                native_min_p=False,
                tuple_sampler_return=False,
                split_execute_and_sample=True,
                graph_safety=GraphSafety.CAPTURE_SAFE,
            ),
            graph_mode="piecewise",
            static_headroom_bytes=1 << 20,
        )
        current = runtime.compile_workloads(timeline)
        assert current.physical is not None
        region = current.decision.execution_regions[0]
        binding_id = region.input_bindings[0].binding_id
        request = RegionShapeSpecializationRequest(
            base_cache_key=current.cache_key,
            base_signature=current.decision.signature,
            observed_plan_epoch=current.physical.plan_epoch,
            region_id=region.region_id,
            artifact_key=region.artifact_key,
            inputs=(
                RegionShapeSpecializationInput(
                    binding_id, (2, 4), "torch.float32", "cuda:0"
                ),
            ),
        )
        requests_json = json.dumps(
            [request.to_wire()], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        expected_options = CompileOptions.from_json(
            _core.apply_region_shape_specializations(
                current.cache_key,
                current.graph.to_json(),
                current.timeline.to_json(),
                current.options.to_json(),
                current.decision.to_json(),
                current.physical.plan_epoch,
                requests_json,
            )
        )

        with mock.patch(
            "tllm.runtime.residual_runtime.publish_workload_compilation"
        ) as publish:
            self.assertTrue(runtime.apply_region_shape_specializations((request,)))

        replacement = runtime.workload_compilation
        assert replacement is not None and replacement.physical is not None
        self.assertEqual(runtime.workload_compile_options, expected_options)
        self.assertEqual(
            (current.physical.plan_epoch, replacement.physical.plan_epoch),
            (1, 2),
        )
        self.assertNotEqual(current.cache_key, replacement.cache_key)
        self.assertNotEqual(
            region.artifact_key,
            replacement.decision.execution_regions[0].artifact_key,
        )
        publish.assert_called_once()

        # A Region-shape key belongs to the active immutable epoch.  Retrying
        # either the same observation or a different one must be rejected by
        # Rust before Python mutates compiler/runtime state.
        occupied_region = replacement.decision.execution_regions[0]
        before_options = runtime.workload_compile_options
        before_compilation = runtime.workload_compilation
        for shape in ((2, 4), (4, 4)):
            occupied_request = RegionShapeSpecializationRequest(
                base_cache_key=replacement.cache_key,
                base_signature=replacement.decision.signature,
                observed_plan_epoch=replacement.physical.plan_epoch,
                region_id=occupied_region.region_id,
                artifact_key=occupied_region.artifact_key,
                inputs=(
                    RegionShapeSpecializationInput(
                        occupied_region.input_bindings[0].binding_id,
                        shape,
                        "torch.float32",
                        "cuda:0",
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "would overwrite an active Region input"):
                _core.apply_region_shape_specializations(
                    replacement.cache_key,
                    replacement.graph.to_json(),
                    replacement.timeline.to_json(),
                    replacement.options.to_json(),
                    replacement.decision.to_json(),
                    replacement.physical.plan_epoch,
                    json.dumps(
                        [occupied_request.to_wire()],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "would overwrite an active Region input"):
                runtime.apply_region_shape_specializations((occupied_request,))
            self.assertIs(runtime.workload_compilation, before_compilation)
            self.assertEqual(runtime.workload_compile_options, before_options)

    def test_rust_shape_merge_rejects_conflicting_region_observations(self) -> None:
        @workload(id="shape.specialization.conflict")
        class Auxiliary:
            @task(
                inputs={"value": provided("value")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: torch.Tensor) -> None:
                _ = value

        runtime = TllmRuntime(
            workloads=(Auxiliary(),),
            workload_memory_budget_bytes=1 << 20,
            workload_provided_inputs={"value": lambda _step: torch.ones((2, 4))},
        )
        timeline = export_main_workload_timeline(
            BackendCapabilities(
                backend_id="shape-specialization-conflict-test",
                runner_family="mrv2",
                sampler_stages=frozenset(),
                native_min_p=False,
                tuple_sampler_return=False,
                split_execute_and_sample=True,
                graph_safety=GraphSafety.CAPTURE_SAFE,
            ),
            graph_mode="piecewise",
            static_headroom_bytes=1 << 20,
        )
        current = runtime.compile_workloads(timeline)
        assert current.physical is not None
        region = current.decision.execution_regions[0]
        request = RegionShapeSpecializationRequest(
            base_cache_key=current.cache_key,
            base_signature=current.decision.signature,
            observed_plan_epoch=current.physical.plan_epoch,
            region_id=region.region_id,
            artifact_key=region.artifact_key,
            inputs=(
                RegionShapeSpecializationInput(
                    region.input_bindings[0].binding_id,
                    (2, 4),
                    "torch.float32",
                    "cuda:0",
                ),
            ),
        )
        conflicting = request.to_wire()
        conflicting["inputs"] = [
            {
                **request.inputs[0].to_wire(),
                "shape": [4, 4],
            }
        ]
        with self.assertRaisesRegex(ValueError, "disagree for one Region input"):
            _core.apply_region_shape_specializations(
                current.cache_key,
                current.graph.to_json(),
                current.timeline.to_json(),
                current.options.to_json(),
                current.decision.to_json(),
                current.physical.plan_epoch,
                json.dumps(
                    [request.to_wire(), conflicting],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
            )


if __name__ == "__main__":
    unittest.main()
