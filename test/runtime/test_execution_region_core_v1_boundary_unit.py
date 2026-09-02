"""Fail-closed boundaries for ExecutionRegion Core v1."""

from __future__ import annotations

import unittest
from unittest import mock

from tllm.schedule.compiler import effective_compile_options
from tllm.schedule.contracts import CompileOptions
from tllm.schedule.coordinator import install_workload_compilation


class ExecutionRegionCoreV1BoundaryUnitTest(unittest.TestCase):
    def test_coordinator_rejects_a_supplied_foreign_environment_fingerprint(self) -> None:
        with mock.patch(
            "tllm.schedule.compiler.compiler_environment_fingerprint",
            return_value="a" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "compiler environment fingerprint mismatch"):
                effective_compile_options(
                    CompileOptions(compiler_fingerprint="b" * 64)
                )

    def test_worker_install_rejects_a_foreign_environment_before_local_binding(self) -> None:
        with mock.patch(
            "tllm.schedule.compiler.compiler_environment_fingerprint",
            return_value="a" * 64,
        ), mock.patch("tllm._core.validate_workload_compilation_wire") as validate_wire:
            with self.assertRaisesRegex(RuntimeError, "compiler environment fingerprint mismatch"):
                install_workload_compilation(
                    cache_key="c" * 64,
                    graph_json="{}",
                    timeline_json="{}",
                    options_json=CompileOptions(
                        compiler_fingerprint="b" * 64
                    ).to_json(),
                    decision_json="{}",
                )
        validate_wire.assert_called_once()


if __name__ == "__main__":
    unittest.main()
