#!/usr/bin/env python3
"""Unit tests for the OverRIDE-inspired consumer."""

from __future__ import annotations

import unittest


class OverRIDEConsumerUnitTest(unittest.TestCase):
    def test_sampler_only_consumer_declares_no_flows(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig())

        self.assertEqual(tuple(consumer.flows()), ())


if __name__ == "__main__":
    unittest.main()
