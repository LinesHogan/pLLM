#!/usr/bin/env python3
"""Kit ergonomics: mechanized lifecycle pieces behave exactly as hand-written.

Each test pins one piece of boilerplate the kit now owns: specialization-key
encoding, resource assembly, worker-spec derivation and reconstruction, the
cold-specialize guard preamble, optimizer construction from the signed
spec, and the update finalization (backward/step/version/stats).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import unittest

import torch

from tllm.kit import RegionAlgorithm, RegionResourcesBase
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.workload_plugins import NativeWorkloadResources
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    State,
    StateScope,
    hidden,
    provided,
    read,
    sink,
    task,
    workload,
)


@dataclass
class _KitConfig:
    implementation: str = "region"
    rank: int = 4
    learning_rate: float = 1e-3
    label: str = "probe"
    enabled: bool = True


class _KitResources(RegionResourcesBase):
    pass


def _spec(hidden_size: int = 8) -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=2,
        hidden_size=hidden_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


@workload(id="kit.probe")
class _ProbeWorkload(RegionAlgorithm[_KitConfig]):
    algorithm = "kit-probe-v1"
    config_type = _KitConfig
    specialization_fields = {
        "rank": "rank",
        "learning_rate": "learning_rate",
        "label": "label",
        "training_enabled": "enabled",
    }
    resources_type = _KitResources
    enabled_field = "enabled"

    heads = State(
        scope=StateScope.REPLICA,
        parameters=("scale",),
        optimizer=AdamWSpec(lr=1e-4),
    )

    def init_parameters(
        self, spec: RuntimeSpecializationSpec
    ) -> Mapping[str, torch.nn.Parameter]:
        return {
            "scale": torch.nn.Parameter(
                torch.zeros((int(spec.hidden_size),), device=spec.device, dtype=spec.dtype)
            )
        }

    def optimizer_overrides(self) -> Mapping[str, object]:
        return {"lr": float(self.config.learning_rate)}

    def provided_inputs(self) -> Mapping[str, object]:
        return {"probe_input": lambda _step_id: torch.ones((1, 1))}

    @task(
        inputs={"value": provided("probe_input")},
        outputs={"result_value": sink("result")},
        effects=(read(heads),),
        timing=Between("model.execute", "step.end"),
    )
    def run(self, value: torch.Tensor) -> torch.Tensor:
        return value


class KitAssemblyUnitTest(unittest.TestCase):
    def test_subclass_must_declare_algorithm_and_config_type(self) -> None:
        with self.assertRaises(TypeError):

            class _Missing(RegionAlgorithm[_KitConfig]):
                pass

    def test_specialization_key_encoding_convention(self) -> None:
        instance = _ProbeWorkload(_KitConfig())
        key = instance.native_workload_resources().specialization_key
        self.assertEqual(
            dict(key),
            {
                "algorithm": "kit-probe-v1",
                "rank": "4",
                "learning_rate": float(1e-3).hex(),
                "label": "probe",
                "training_enabled": "True",
            },
        )

    def test_default_worker_factory_and_reconstruct_roundtrip(self) -> None:
        instance = _ProbeWorkload(_KitConfig(rank=7))
        spec = instance.native_workload_resources().worker_spec
        assert spec is not None
        self.assertEqual(
            spec.factory, f"{__name__}:_ProbeWorkload.reconstruct"
        )
        rebuilt = instance.reconstruct(spec.config)
        self.assertIsInstance(rebuilt, NativeWorkloadResources)
        self.assertEqual(
            dict(rebuilt.specialization_key),
            dict(instance.native_workload_resources().specialization_key),
        )

    def test_state_driven_resource_fields(self) -> None:
        resources = _ProbeWorkload(_KitConfig()).native_workload_resources()
        self.assertEqual(sorted(resources.state_snapshots), ["heads"])
        self.assertIsNotNone(resources.cold_specializer)
        self.assertIsNotNone(resources.stats_reader)
        self.assertIsNotNone(resources.enabled_setter)
        self.assertIsNotNone(resources.enabled_predicate)
        self.assertEqual(sorted(resources.provided_inputs), ["probe_input"])

    def test_enabled_epoch_rebuilds_resource_snapshot(self) -> None:
        instance = _ProbeWorkload(_KitConfig())
        first = instance.native_workload_resources()
        assert first.enabled_predicate is not None
        self.assertTrue(first.enabled_predicate())
        instance.set_enabled(False)
        second = instance.native_workload_resources()
        self.assertIsNot(first, second)
        assert second.enabled_predicate is not None
        self.assertFalse(second.enabled_predicate())
        self.assertEqual(second.specialization_key["training_enabled"], "False")


class KitColdSpecializeUnitTest(unittest.TestCase):
    def test_allocates_parameters_and_optimizer_from_signed_state(self) -> None:
        instance = _ProbeWorkload(_KitConfig())
        instance.cold_specialize(_spec())
        scale = instance.params["scale"]
        self.assertIsInstance(scale, torch.nn.Parameter)
        self.assertEqual(tuple(scale.shape), (8,))
        optimizer = instance.optimizer
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        assert optimizer is not None
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertIs(instance.state_version, 0)

    def test_repeat_same_spec_is_noop_different_spec_rejected(self) -> None:
        instance = _ProbeWorkload(_KitConfig())
        instance.cold_specialize(_spec())
        instance.cold_specialize(_spec())
        with self.assertRaises(RuntimeError):
            instance.cold_specialize(_spec(hidden_size=16))

    def test_parameter_names_must_match_signed_state(self) -> None:
        @workload(id="kit.bad-params")
        class _Bad(RegionAlgorithm[_KitConfig]):
            algorithm = "kit-bad-v1"
            config_type = _KitConfig
            worker_factory = "test.unused:reconstruct"
            heads = State(
                scope=StateScope.REPLICA,
                parameters=("scale",),
                optimizer=AdamWSpec(lr=1e-4),
            )

            def init_parameters(
                self, spec: RuntimeSpecializationSpec
            ) -> Mapping[str, torch.nn.Parameter]:
                return {"wrong": torch.nn.Parameter(torch.zeros((1,)))}

            @task(
                inputs={"value": provided("probe_input")},
                outputs={"result_value": sink("result")},
                effects=(read(heads),),
                timing=Before("step.end"),
            )
            def run(self, value: torch.Tensor) -> torch.Tensor:
                return value

        with self.assertRaises(RuntimeError):
            _Bad(_KitConfig()).cold_specialize(_spec())

    def test_step_update_advances_version_and_records_stats(self) -> None:
        instance = _ProbeWorkload(_KitConfig())
        instance.cold_specialize(_spec())
        scale = instance.params["scale"]
        assert isinstance(scale, torch.nn.Parameter)
        loss = (scale * 2.0 + 1.0).sum()
        instance.step_update(loss)
        self.assertEqual(instance.state_version, 1)
        resources = instance.resources
        assert isinstance(resources, RegionResourcesBase)
        stats = resources.read_and_reset_stats(sync=False)
        self.assertEqual(stats.loss_count, 1)
        self.assertGreater(stats.loss_avg, 0.0)
        self.assertNotEqual(
            float(scale.detach().abs().sum()), 0.0, "optimizer step must move parameters"
        )

    def test_step_update_requires_grad_and_specialization(self) -> None:
        instance = _ProbeWorkload(_KitConfig())
        with self.assertRaises(RuntimeError):
            instance.step_update(torch.tensor(1.0))
        instance.cold_specialize(_spec())
        with self.assertRaises(ValueError):
            instance.step_update(torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
