from __future__ import annotations

from dataclasses import replace
import unittest
from unittest import mock

import torch

from tllm.schedule import (
    CompileOptions,
    MainWorkloadTimeline,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
)
from tllm.schedule.contracts import ScheduleEffect, ScheduleEffectKind
from tllm.workload import (
    ApproximationCriteria,
    ApproximationPolicy,
    Between,
    declare_workload,
    materialize_region_artifacts,
    materialize_region_callable,
    provided,
    sink,
    task,
    workload,
)
from tllm.workload.region_compiler import (
    RegionParityError,
    compare_region_outputs,
    region_dtype_tolerance,
)


def _timeline() -> MainWorkloadTimeline:
    return MainWorkloadTimeline(
        backend_id="vllm",
        backend_version="0.27.1",
        runner_family="mrv2",
        graph_mode="eager",
        anchors=(
            TimelineAnchor("model.execute", 0),
            TimelineAnchor("step.end", 1),
        ),
        regions=(),
        static_headroom_bytes=64,
    )


_REAL_TORCH_COMPILE = torch.compile


def _eager_backend_compile(function: object, **kwargs: object) -> object:
    return _REAL_TORCH_COMPILE(function, backend="eager", **kwargs)  # type: ignore[arg-type]


class RegionParityGateUnitTest(unittest.TestCase):
    def _compile_single_region(self, declaration: object):
        compilation = WorkloadCompilerCoordinator(
            (declaration,), options=CompileOptions(memory_budget_bytes=1 << 20)
        ).compile(_timeline())
        return compilation

    def test_first_call_parity_gate_passes_and_runs_oracle_once(self) -> None:
        calls = {"count": 0}

        @workload(id="region.parity-pass")
        class Faithful:
            @task(
                inputs={"value": provided("value")},
                outputs={"result_value": sink("result")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                calls["count"] += 1
                assert isinstance(value, torch.Tensor)
                return value * 2

        compilation = self._compile_single_region(declare_workload(Faithful()))
        with mock.patch("torch.compile", side_effect=_eager_backend_compile):
            (artifact,) = materialize_region_artifacts(
                (declare_workload(Faithful()),), compilation.decision, compilation.options
            )
        self.assertTrue(artifact.compiled)
        self.assertEqual(artifact.parity, "first_call")

        inputs = {artifact.input_keys[0]: torch.tensor([3.0])}
        first = artifact.execute(inputs)
        second = artifact.execute(inputs)

        torch.testing.assert_close(
            first[artifact.output_keys[0]], torch.tensor([6.0])
        )
        torch.testing.assert_close(
            second[artifact.output_keys[0]], torch.tensor([6.0])
        )
        # First call runs oracle + compiled; the second call is compiled-only.
        self.assertEqual(calls["count"], 3)

    def test_first_call_parity_gate_fails_closed_on_divergence(self) -> None:
        @workload(id="region.parity-diverge")
        class Drifting:
            @task(
                inputs={"value": provided("value")},
                outputs={"result_value": sink("result")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                assert isinstance(value, torch.Tensor)
                return value * 2

        declaration = declare_workload(Drifting())
        compilation = self._compile_single_region(declaration)

        def corrupting_compile(function: object, **kwargs: object) -> object:
            compiled = _REAL_TORCH_COMPILE(function, backend="eager", **kwargs)  # type: ignore[arg-type]

            def drifted(*args: object) -> object:
                outputs = compiled(*args)
                return tuple(
                    output + 1.0 if torch.is_tensor(output) else output
                    for output in outputs
                )

            return drifted

        with mock.patch("torch.compile", side_effect=corrupting_compile):
            (artifact,) = materialize_region_artifacts(
                (declaration,), compilation.decision, compilation.options
            )
        with self.assertRaises(RegionParityError):
            artifact.execute({artifact.input_keys[0]: torch.tensor([3.0])})

    def test_parity_gate_restores_rng_for_the_compiled_run(self) -> None:
        @workload(id="region.parity-rng")
        class Random:
            @task(
                inputs={"value": provided("value")},
                outputs={"result_value": sink("result")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                assert isinstance(value, torch.Tensor)
                return value + torch.randn_like(value)

        declaration = declare_workload(Random())
        compilation = self._compile_single_region(declaration)
        with mock.patch("torch.compile", side_effect=_eager_backend_compile):
            (artifact,) = materialize_region_artifacts(
                (declaration,), compilation.decision, compilation.options
            )
        # A faithful compiled run on restored RNG must not trip the gate.
        output = artifact.execute({artifact.input_keys[0]: torch.zeros(4)})
        self.assertEqual(tuple(output[artifact.output_keys[0]].shape), (4,))

    def test_mutating_effects_defer_the_parity_gate(self) -> None:
        @workload(id="region.parity-effects")
        class Stateful:
            @task(
                inputs={"value": provided("value")},
                outputs={"result_value": sink("result")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                assert isinstance(value, torch.Tensor)
                return value * 2

        declaration = declare_workload(Stateful())
        compilation = self._compile_single_region(declaration)
        region = compilation.decision.execution_regions[0]
        effectful = replace(
            region,
            effects=(
                ScheduleEffect(kind=ScheduleEffectKind.STATE_WRITE, resource="bank"),
            ),
        )
        with mock.patch("torch.compile", side_effect=_eager_backend_compile):
            artifact = materialize_region_callable(
                (declaration,), effectful, compilation.options
            )
        self.assertTrue(artifact.compiled)
        self.assertEqual(artifact.parity, "deferred_state_effects")

    def test_eager_adapter_marks_parity_eager(self) -> None:
        @workload(id="region.parity-eager")
        class Plain:
            @task(
                inputs={"value": provided("value")},
                outputs={"result_value": sink("result")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                assert isinstance(value, torch.Tensor)
                return value * 2

        declaration = declare_workload(Plain())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20, compile_policy="eager"
            ),
        ).compile(_timeline())
        (artifact,) = materialize_region_artifacts(
            (declaration,), compilation.decision, compilation.options
        )
        self.assertFalse(artifact.compiled)
        self.assertEqual(artifact.parity, "eager")


class ApproximationParityUnitTest(unittest.TestCase):
    @staticmethod
    def _declaration(criteria: ApproximationCriteria | None):
        @workload(
            id="region.parity-approx",
            approximation=ApproximationPolicy.JACOBI,
            approximation_criteria=criteria,
        )
        class Approx:
            @task(
                inputs={"value": provided("value")},
                outputs={"result_value": sink("result")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                assert isinstance(value, torch.Tensor)
                return value * 2

        return declare_workload(Approx())

    @staticmethod
    def _drifting_compile(delta: float):
        def corrupting_compile(function: object, **kwargs: object) -> object:
            compiled = _REAL_TORCH_COMPILE(function, backend="eager", **kwargs)  # type: ignore[arg-type]

            def drifted(*args: object) -> object:
                outputs = compiled(*args)
                return tuple(
                    output + delta if torch.is_tensor(output) else output
                    for output in outputs
                )

            return drifted

        return corrupting_compile

    def _compile(self, declaration: object):
        return WorkloadCompilerCoordinator(
            (declaration,), options=CompileOptions(memory_budget_bytes=1 << 20)
        ).compile(_timeline())

    def test_approximation_without_criteria_fails_closed(self) -> None:
        declaration = self._declaration(None)
        compilation = self._compile(declaration)
        with mock.patch("torch.compile", side_effect=_eager_backend_compile):
            with self.assertRaisesRegex(RegionParityError, "no approximation criteria"):
                materialize_region_artifacts(
                    (declaration,), compilation.decision, compilation.options
                )

    def test_approximation_gate_uses_declared_criteria(self) -> None:
        declaration = self._declaration(ApproximationCriteria(rtol=0.2, atol=0.2))
        compilation = self._compile(declaration)
        with mock.patch("torch.compile", side_effect=self._drifting_compile(0.1)):
            (artifact,) = materialize_region_artifacts(
                (declaration,), compilation.decision, compilation.options
            )
        self.assertEqual(artifact.parity, "first_call_approximation")
        self.assertEqual(artifact.parity_tolerance, (0.2, 0.2))
        # +0.1 drift on values of order 6 stays inside rtol=atol=0.2.
        output = artifact.execute({artifact.input_keys[0]: torch.tensor([3.0])})
        torch.testing.assert_close(
            output[artifact.output_keys[0]], torch.tensor([6.1])
        )

    def test_approximation_gate_rejects_drift_beyond_criteria(self) -> None:
        declaration = self._declaration(ApproximationCriteria(rtol=1e-4, atol=1e-4))
        compilation = self._compile(declaration)
        with mock.patch("torch.compile", side_effect=self._drifting_compile(0.1)):
            (artifact,) = materialize_region_artifacts(
                (declaration,), compilation.decision, compilation.options
            )
        with self.assertRaises(RegionParityError):
            artifact.execute({artifact.input_keys[0]: torch.tensor([3.0])})


class CompareRegionOutputsUnitTest(unittest.TestCase):
    def test_matching_outputs_pass(self) -> None:
        compare_region_outputs(
            "region-ok",
            (torch.tensor([1.0]), "meta"),
            (torch.tensor([1.0]), "meta"),
        )

    def test_output_count_mismatch_fails(self) -> None:
        with self.assertRaises(RegionParityError):
            compare_region_outputs(
                "region-count", (torch.tensor([1.0]),), (torch.tensor([1.0]),) * 2
            )

    def test_type_mismatch_fails(self) -> None:
        with self.assertRaises(RegionParityError):
            compare_region_outputs("region-type", (torch.tensor([1.0]),), (1.0,))

    def test_shape_mismatch_fails(self) -> None:
        with self.assertRaises(RegionParityError):
            compare_region_outputs(
                "region-shape", (torch.zeros(2),), (torch.zeros(3),)
            )

    def test_dtype_tolerance_is_dtype_aware(self) -> None:
        self.assertEqual(region_dtype_tolerance(torch.float64), 1e-10)
        self.assertEqual(region_dtype_tolerance(torch.float16), 2e-2)
        self.assertEqual(region_dtype_tolerance(torch.float32), 1e-5)
        # fp16 drift within tolerance passes; the same drift in fp32 fails.
        compare_region_outputs(
            "region-fp16",
            (torch.ones(4, dtype=torch.float16),),
            (torch.full((4,), 1.001, dtype=torch.float16),),
        )
        with self.assertRaises(RegionParityError):
            compare_region_outputs(
                "region-fp32",
                (torch.ones(4, dtype=torch.float32),),
                (torch.full((4,), 1.001, dtype=torch.float32),),
            )

    def test_float64_drift_fails(self) -> None:
        with self.assertRaises(RegionParityError):
            compare_region_outputs(
                "region-fp64",
                (torch.ones(2, dtype=torch.float64),),
                (torch.full((2,), 1.0 + 1e-8, dtype=torch.float64),),
            )


if __name__ == "__main__":
    unittest.main()
