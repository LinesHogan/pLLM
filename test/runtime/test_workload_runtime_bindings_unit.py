"""Contract tests for closed native workload runtime bindings."""

from __future__ import annotations

import unittest
from collections.abc import Hashable, Mapping
from typing import cast

from tllm.contracts.consumer_stats import TrainingStats
from tllm.runtime.workload_bindings import (
    LocalRuntimeServices,
    NoPerRequestStats,
    NoRuntimeServices,
    NoSamplerBinding,
    NoWorkerBinding,
    RuntimeInputBinding,
    RuntimeOutputBinding,
    RuntimePortRegistry,
    RuntimeStateRegistry,
    StateSnapshot,
    WorkloadRuntimeBinding,
)
from tllm.runtime.workload_plugins import NativeWorkload


class _BindingOwner(NativeWorkload):
    def __init__(self) -> None:
        self.binding = WorkloadRuntimeBinding(owner=self)

    def native_workload_resources(self) -> WorkloadRuntimeBinding:
        return self.binding


def _input(step_id: int) -> object:
    return step_id


def _output(_value: object, _step_id: int) -> None:
    return None


def _snapshot(_step_id: int) -> Mapping[Hashable, int]:
    return {None: 3}


class WorkloadRuntimeBindingTest(unittest.TestCase):
    def test_empty_variants_have_no_registered_endpoints(self) -> None:
        binding = _BindingOwner().binding

        self.assertIsInstance(binding.services, NoRuntimeServices)
        self.assertIsInstance(binding.sampler, NoSamplerBinding)
        self.assertIsInstance(binding.worker, NoWorkerBinding)
        self.assertEqual(binding.ports.inputs, {})
        self.assertEqual(binding.states.snapshots, {})
        self.assertEqual(binding.packed_hidden_resources, ())

    def test_registries_are_immutable_and_lifecycle_is_closed(self) -> None:
        owner = _BindingOwner()
        inputs: dict[str, RuntimeInputBinding[object]] = {
            "input": RuntimeInputBinding(_input)
        }
        outputs: dict[str, RuntimeOutputBinding[object]] = {
            "output": RuntimeOutputBinding(_output)
        }
        snapshots: dict[str, StateSnapshot] = {"state": _snapshot}
        services = LocalRuntimeServices(
            cold_specialize=lambda _spec: None,
            diagnostics=lambda: {"updates": 3},
            stats=lambda _sync: TrainingStats(loss_avg=0.0, loss_count=3),
            set_enabled=lambda _enabled: None,
            is_enabled=lambda: True,
            synchronize=lambda: None,
        )
        binding = WorkloadRuntimeBinding(
            owner=owner,
            ports=RuntimePortRegistry(inputs=inputs, outputs=outputs),
            states=RuntimeStateRegistry(snapshots=snapshots),
            services=services,
        )

        inputs["other"] = RuntimeInputBinding(_input)
        outputs["other"] = RuntimeOutputBinding(_output)
        snapshots["other"] = _snapshot
        self.assertEqual(tuple(binding.ports.inputs), ("input",))
        self.assertEqual(tuple(binding.ports.outputs), ("output",))
        self.assertEqual(tuple(binding.states.snapshots), ("state",))
        self.assertIsInstance(services.per_request_stats, NoPerRequestStats)
        self.assertEqual(services.diagnostics(), {"updates": 3})

        with self.assertRaises(TypeError):
            cast(
                dict[str, RuntimeInputBinding[object]], binding.ports.inputs
            )["other"] = RuntimeInputBinding(_input)


if __name__ == "__main__":
    unittest.main()
