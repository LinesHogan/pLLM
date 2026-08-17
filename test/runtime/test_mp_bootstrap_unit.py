#!/usr/bin/env python3
"""Unit tests for the multiprocess bootstrap spec and general plugin."""

from __future__ import annotations

import dataclasses
import os
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import torch

from tllm.common.dataclass_json import dataclass_from_dict, dataclass_to_jsonable
from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.consumers.inplace_ttt.config import InPlaceTTTConsumerConfig
from tllm.consumers import registry as consumer_registry
from tllm.runtime import residual_runtime
from tllm.runtime.mp import bootstrap
from tllm.runtime.mp.bootstrap import BootstrapSpec, ConsumerSpec
from tllm.runtime.mp.plugin import tllm_general_plugin
from tllm.runtime.state import ResidualRuntimeConfig


class DataclassJsonUnitTest(unittest.TestCase):
    def test_builtin_default_configs_round_trip(self) -> None:
        for name in consumer_registry.registered_consumers():
            with self.subTest(consumer=name):
                spec = consumer_registry.spec_for_consumer(consumer_registry.build_consumer(name, None))
                rebuilt = consumer_registry.build_consumer_from_spec(spec)
                self.assertEqual(type(rebuilt).__name__, type(consumer_registry.build_consumer(name, None)).__name__)
                self.assertEqual(dataclass_to_jsonable(rebuilt.config), spec.config)

    def test_esamp_config_with_nested_initializer_round_trips(self) -> None:
        config = ESampConsumerConfig(
            per_request_model_bank=True,
            model_bank_slots=8,
            model_bank_initializer=SVDModelBankInitializerConfig(method="ridge_svd", min_rows=16),
            adaptation_stream_mode="serial",
            enable_distiller_intervention=True,
            distiller_beta=0.5,
        )
        data = dataclass_to_jsonable(config)
        self.assertIsInstance(data["model_bank_initializer"], dict)
        self.assertIsInstance(data["model_bank_initializer"]["method"], str)
        rebuilt = dataclass_from_dict(ESampConsumerConfig, data)
        self.assertEqual(rebuilt.model_bank_initializer, config.model_bank_initializer)
        self.assertEqual(rebuilt.adaptation_stream_mode, "serial")
        self.assertAlmostEqual(rebuilt.distiller_beta, 0.5)

    def test_tuple_literal_field_round_trips(self) -> None:
        config = InPlaceTTTConsumerConfig(phases=("prefill", "decode"), target_layer=-1)
        data = dataclass_to_jsonable(config)
        self.assertEqual(data["phases"], ["prefill", "decode"])
        rebuilt = dataclass_from_dict(InPlaceTTTConsumerConfig, data)
        self.assertEqual(rebuilt.phases, ("prefill", "decode"))

    def test_unsafe_values_raise_at_capture_time(self) -> None:
        with self.assertRaises(TypeError):
            dataclass_to_jsonable(_UnsafeDataclass(payload=torch.zeros(1)))
        with self.assertRaises(TypeError):
            dataclass_to_jsonable(_UnsafeDataclass(payload=lambda: 1))

    def test_unknown_consumer_spec_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no registered types"):
            consumer_registry.build_consumer_from_spec(ConsumerSpec(name="nope", config={}))


@dataclasses.dataclass
class _UnsafeDataclass:
    payload: object = None


class BootstrapSpecUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        runtime = residual_runtime.RUNTIME
        self._saved = (
            runtime.config,
            runtime.consumer,
            runtime.consumer_registry,
            runtime.dispatch_plan,
            runtime.execution_plan,
            runtime.test_time_lora_enabled,
        )

    def tearDown(self) -> None:
        runtime = residual_runtime.RUNTIME
        (
            runtime.config,
            runtime.consumer,
            runtime.consumer_registry,
            runtime.dispatch_plan,
            runtime.execution_plan,
            runtime.test_time_lora_enabled,
        ) = self._saved
        os.environ.pop(bootstrap.TLLM_BOOTSTRAP_SPEC_ENV, None)

    def _configure_dummy_runtime(self) -> None:
        residual_runtime.configure_runtime(
            graph_scratch_rows=4,
            tap_layer_paths=("model.model.layers[0].input_layernorm",),
            source_layer_path="model.model.layers[0].input_layernorm",
            target_layer_path="model.model.layers[-1].input_layernorm",
        )
        consumer_registry.activate_consumer(
            consumer_registry.build_consumer("dummy", DummyConsumerConfig(consumer_id="dummy_mp")),
            primary=True,
        )

    def test_spec_json_round_trip(self) -> None:
        spec = BootstrapSpec(
            runtime_config={"graph_scratch_rows": 4},
            primary_consumer=ConsumerSpec(name="dummy", config={"consumer_id": "d"}),
            consumers=(ConsumerSpec(name="gtta", config={"consumer_id": "g"}),),
            enable_flags={"test_time_lora_enabled": True},
        )
        path = bootstrap.write_bootstrap_spec(spec)
        try:
            loaded = bootstrap.read_bootstrap_spec(path)
        finally:
            os.unlink(path)
        self.assertEqual(loaded.to_json_dict(), spec.to_json_dict())

    def test_schema_version_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version"):
            BootstrapSpec.from_json_dict({"schema_version": 999})

    def test_capture_apply_round_trip_matches(self) -> None:
        self._configure_dummy_runtime()
        spec = bootstrap.capture_bootstrap_spec()
        self.assertIsNotNone(spec.primary_consumer)
        self.assertEqual(spec.primary_consumer.name, "dummy")

        # Simulate a fresh child process runtime.
        residual_runtime.set_runtime_consumer(None)
        residual_runtime.clear_dispatch_consumers()
        residual_runtime.RUNTIME.config = ResidualRuntimeConfig()
        self.assertFalse(bootstrap.runtime_matches_spec(spec))

        bootstrap.apply_bootstrap_spec(spec)
        self.assertTrue(bootstrap.runtime_matches_spec(spec))
        consumer = residual_runtime.RUNTIME.consumer
        self.assertEqual(getattr(consumer.config, "consumer_id"), "dummy_mp")
        self.assertEqual(residual_runtime.RUNTIME.config.graph_scratch_rows, 4)

    def test_plugin_no_spec_is_noop(self) -> None:
        os.environ.pop(bootstrap.TLLM_BOOTSTRAP_SPEC_ENV, None)
        with mock.patch("tllm.runtime.lifecycle.install_runner_patch") as install:
            tllm_general_plugin()
        install.assert_not_called()

    def test_plugin_applies_spec_and_is_idempotent(self) -> None:
        self._configure_dummy_runtime()
        path = bootstrap.export_bootstrap_spec_env()
        residual_runtime.set_runtime_consumer(None)
        residual_runtime.clear_dispatch_consumers()
        residual_runtime.RUNTIME.config = ResidualRuntimeConfig()

        with mock.patch("tllm.runtime.lifecycle.install_runner_patch") as install:
            tllm_general_plugin()
            self.assertTrue(bootstrap.runtime_matches_spec(bootstrap.read_bootstrap_spec(path)))
            tllm_general_plugin()  # second run must be a no-op beyond checks
        self.assertEqual(install.call_count, 2)
        self.assertEqual(getattr(residual_runtime.RUNTIME.consumer.config, "consumer_id"), "dummy_mp")

    def test_spec_is_frozen(self) -> None:
        spec = BootstrapSpec(runtime_config={})
        with self.assertRaises(FrozenInstanceError):
            spec.schema_version = 2  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
