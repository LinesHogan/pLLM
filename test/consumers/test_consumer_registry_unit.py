#!/usr/bin/env python3
"""Every registered consumer name must build and compile through the default plan."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

from tllm.consumers.base import BaseConsumer
from tllm.consumers.plugin import ConsumerPluginSpec
from tllm.consumers import registry as consumer_registry
from tllm.consumers.registry import (
    build_consumer,
    register_consumer_plugin,
    registered_consumers,
    unregister_consumer_plugin,
)
from tllm.runtime import consumer_plugins
from tllm.runtime.execution_plan import compile_default_plan


class ConsumerRegistryUnitTest(unittest.TestCase):
    def test_entry_point_name_mismatch_does_not_pollute_registry(self) -> None:
        @dataclass
        class Config:
            pass

        class Consumer(BaseConsumer[Config]):
            consumer_id = "external.mismatch"

        plugin = ConsumerPluginSpec("declared_name", Consumer, Config, Consumer)
        entry_point = SimpleNamespace(load=lambda: plugin)
        with mock.patch.object(
            consumer_plugins,
            "_consumer_entry_points",
            return_value={"requested_name": entry_point},
        ):
            with self.assertRaisesRegex(ValueError, "declared mismatched name"):
                build_consumer("requested_name", None)
            self.assertNotIn("declared_name", registered_consumers())

    def test_external_typed_plugin_builds_without_editing_builtin_catalog(self) -> None:
        @dataclass
        class Config:
            value: int = 7

        class Consumer(BaseConsumer[Config]):
            consumer_id = "external.test"

            def __init__(self, config: Config) -> None:
                self.config = config

        plugin = ConsumerPluginSpec("external_test", Consumer, Config, Consumer)
        register_consumer_plugin(plugin)
        try:
            consumer = build_consumer("external_test", None)
            self.assertIsInstance(consumer, Consumer)
            self.assertEqual(consumer.config.value, 7)
        finally:
            unregister_consumer_plugin("external_test")

    def test_builtin_consumers_are_registered(self) -> None:
        self.assertEqual(
            registered_consumers(),
            (
                "dummy",
                "esamp",
                "fttt",
                "gtta",
                "inplace_ttt",
                "override",
                "recirculation",
                "tlm",
                "ttrl",
            ),
        )

    def test_every_registered_consumer_flows_compile(self) -> None:
        for name in registered_consumers():
            with self.subTest(consumer=name):
                consumer = build_consumer(name, None)
                plan = compile_default_plan(consumers=[consumer])
                self.assertIsNotNone(plan)


if __name__ == "__main__":
    unittest.main()
