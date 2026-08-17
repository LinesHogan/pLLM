#!/usr/bin/env python3
"""Every registered consumer name must build and compile through the default plan."""

from __future__ import annotations

import unittest

from tllm.consumers.registry import build_consumer, registered_consumers
from tllm.runtime.execution_plan import compile_default_plan


class ConsumerRegistryUnitTest(unittest.TestCase):
    def test_builtin_consumers_are_registered(self) -> None:
        self.assertEqual(
            registered_consumers(),
            ("dummy", "esamp", "fttt", "gtta", "inplace_ttt", "override", "tlm", "ttrl"),
        )

    def test_every_registered_consumer_builds_with_default_config(self) -> None:
        for name in registered_consumers():
            with self.subTest(consumer=name):
                consumer = build_consumer(name, None)
                self.assertEqual(consumer.consumer_id, consumer.config.consumer_id)

    def test_every_registered_consumer_flows_compile(self) -> None:
        for name in registered_consumers():
            with self.subTest(consumer=name):
                consumer = build_consumer(name, None)
                plan = compile_default_plan(consumers=[consumer])
                self.assertIsNotNone(plan)

    def test_unknown_consumer_name_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            build_consumer("no_such_consumer", None)


if __name__ == "__main__":
    unittest.main()
