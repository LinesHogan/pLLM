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
from tllm.ports.sampler import CandidateModifierState
from tllm.runtime.sampler_bridge.types import SamplerStepView
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


class _AutoSpecConfig(_KitConfig):
    pass


@workload(id="kit.auto-spec")
class _AutoSpecWorkload(RegionAlgorithm[_AutoSpecConfig]):
    """No specialization_fields: the key derives from every config field."""

    algorithm = "kit-auto-spec-v1"
    config_type = _AutoSpecConfig
    specialization_exclude = ("implementation",)

    @task(inputs={"value": provided("probe_input")}, timing=Before("step.end"))
    def run(self, value: torch.Tensor) -> torch.Tensor:
        return value


@workload(id="kit.modifier")
class _ModifierWorkload(RegionAlgorithm[_KitConfig]):
    """Sink + modify_candidates: provider, store, and checks come from the kit."""

    algorithm = "kit-modifier-v1"
    config_type = _KitConfig
    enabled_field = "enabled"

    @task(
        inputs={"value": provided("probe_input")},
        outputs={"guidance": sink("guidance")},
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def predict(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def modify_candidates(self, value: torch.Tensor, view: object) -> object:
        rows = int(value.shape[0])
        return CandidateModifierState(
            beta=1.0,
            backend="post_filter_exact",
            affected_row_ids=torch.arange(rows, dtype=torch.long),
            pred_hidden=value,
            lm_head_weight=torch.eye(4),
            lm_head_bias=None,
        )


def _set_decode_metadata(rows: int) -> None:
    from tllm.runtime import residual_runtime

    runtime = residual_runtime.RUNTIME
    runtime.decode_request_ids = [f"r{i}" for i in range(rows)]
    runtime.decode_prompt_idxs = [0] * rows
    runtime.decode_sample_idxs = [3] * rows


def _clear_decode_metadata() -> None:
    from tllm.runtime import residual_runtime

    runtime = residual_runtime.RUNTIME
    runtime.decode_request_ids = []
    runtime.decode_prompt_idxs = []
    runtime.decode_sample_idxs = []


def _view(step_id: int, rows: int, device: torch.device) -> SamplerStepView:
    return SamplerStepView(
        engine_step_id=step_id,
        phase="decode",
        logits=torch.zeros((rows, 4), device=device),
        sampling_metadata=None,
        decode_count=rows,
        request_ids=tuple(f"r{i}" for i in range(rows)),
        prompt_idxs=(0,) * rows,
        sample_idxs=(3,) * rows,
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=torch.zeros((rows, 4), device=device),
        device=device,
        model=None,
        runner=None,
    )


class KitV2AutoDerivationTest(unittest.TestCase):
    def test_specialization_key_derives_all_config_fields(self) -> None:
        key = dict(_AutoSpecWorkload(_AutoSpecConfig()).native_workload_resources().specialization_key)
        self.assertEqual(
            key,
            {
                "algorithm": "kit-auto-spec-v1",
                "rank": "4",
                "learning_rate": float(1e-3).hex(),
                "label": "probe",
                "enabled": "True",
            },
        )

    def test_unencodable_config_field_fails_closed(self) -> None:
        @dataclass
        class _BadConfig:
            callback: object = None

        @workload(id="kit.bad-spec")
        class _Bad(RegionAlgorithm[_BadConfig]):
            algorithm = "kit-bad-spec-v1"
            config_type = _BadConfig

            @task(inputs={"value": provided("probe_input")})
            def run(self, value: torch.Tensor) -> torch.Tensor:
                return value

        with self.assertRaises(TypeError):
            _Bad(_BadConfig())

    def test_param_and_model_accessors_never_optional(self) -> None:
        instance = _ProbeWorkload(_KitConfig())
        with self.assertRaises(RuntimeError):
            instance.param("scale")  # before cold_specialize
        instance.cold_specialize(_spec())
        self.assertIsInstance(instance.param("scale"), torch.nn.Parameter)
        with self.assertRaises(RuntimeError):
            instance.param("missing")
        with self.assertRaises(RuntimeError):
            instance.model  # zero Module parameters

    def test_read_task_runs_under_grad_automatically(self) -> None:
        seen: dict[str, bool] = {}

        @workload(id="kit.grad-wrap")
        class _GradProbe(RegionAlgorithm[_KitConfig]):
            algorithm = "kit-grad-wrap-v1"
            config_type = _KitConfig
            worker_factory = "test.unused:reconstruct"  # <locals> is not importable
            heads = State(
                scope=StateScope.REPLICA,
                parameters=("scale",),
                optimizer=AdamWSpec(lr=1e-4),
            )

            def init_parameters(self, spec: RuntimeSpecializationSpec) -> Mapping[str, torch.nn.Parameter]:
                return {"scale": torch.nn.Parameter(torch.zeros(1))}

            @task(inputs={"value": provided("probe_input")}, effects=(read(heads),))
            def run(self, value: torch.Tensor) -> torch.Tensor:
                seen["grad"] = torch.is_grad_enabled()
                seen["inference"] = torch.is_inference_mode_enabled()
                return value

        instance = _GradProbe(_KitConfig())
        with torch.inference_mode():
            instance.run(torch.ones(1))
        self.assertEqual(seen, {"grad": True, "inference": False})


class KitV2SinkAndModifierTest(unittest.TestCase):
    def tearDown(self) -> None:
        _clear_decode_metadata()

    def test_declared_sink_auto_binds_into_store_with_identity(self) -> None:
        instance = _ModifierWorkload(_KitConfig())
        sinks = instance.native_workload_resources().output_sinks
        self.assertEqual(sorted(sinks), ["guidance"])
        _set_decode_metadata(2)
        payload = torch.ones((2, 4))
        sinks["guidance"](payload, 7)
        resources = instance.resources
        assert isinstance(resources, RegionResourcesBase)
        record = resources.sinks.for_step("guidance", 7)
        assert record is not None
        self.assertEqual(record.identity.request_ids, ("r0", "r1"))
        self.assertEqual(record.identity.sample_idxs, (3, 3))
        self.assertEqual(record.identity.phase, "decode")
        self.assertTrue(torch.equal(record.value, payload))
        self.assertIsNot(record.value, payload, "store must isolate the payload")
        self.assertIsNone(resources.sinks.for_step("guidance", 8))

    def test_provider_auto_installed_and_aligns_rows(self) -> None:
        instance = _ModifierWorkload(_KitConfig())
        provider = instance.native_workload_resources().sampler_provider
        self.assertIsNotNone(provider)
        assert provider is not None
        _set_decode_metadata(2)
        instance.native_workload_resources().output_sinks["guidance"](
            torch.ones((2, 4)), 7
        )
        state = provider.prepare_step(_view(7, 2, torch.device("cpu")))
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.beta, 1.0)
        # Wrong step: no publication -> no modifier.
        self.assertIsNone(provider.prepare_step(_view(8, 2, torch.device("cpu"))))
        # Row identity mismatch is a framework-checked invariant violation.
        with self.assertRaises(RuntimeError):
            provider.prepare_step(_view(7, 1, torch.device("cpu")))

    def test_disabled_workflow_never_modifies(self) -> None:
        instance = _ModifierWorkload(_KitConfig(enabled=False))
        provider = instance.native_workload_resources().sampler_provider
        assert provider is not None
        self.assertFalse(provider.is_active())

    def test_modify_candidates_requires_exactly_one_sink(self) -> None:
        @workload(id="kit.no-sink")
        class _NoSink(RegionAlgorithm[_KitConfig]):
            algorithm = "kit-no-sink-v1"
            config_type = _KitConfig

            @task(inputs={"value": provided("probe_input")})
            def run(self, value: torch.Tensor) -> torch.Tensor:
                return value

            def modify_candidates(self, value: torch.Tensor, view: object) -> object:
                raise AssertionError("unused")

        with self.assertRaises(RuntimeError):
            _NoSink(_KitConfig())


if __name__ == "__main__":
    unittest.main()
