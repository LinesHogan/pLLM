#!/usr/bin/env python3
"""Unit tests for the generic residual runtime host."""

from __future__ import annotations

import importlib
import unittest


class RuntimeResidualRuntimeHostUnitTest(unittest.TestCase):
    def test_residual_runtime_host_exposes_runtime_state_and_patch_entrypoints(self) -> None:
        host = importlib.import_module("tllm.runtime.residual_runtime")

        self.assertTrue(hasattr(host, "RUNTIME"))
        self.assertTrue(callable(host.configure_runtime))
        self.assertTrue(callable(host.make_llm))
        self.assertTrue(callable(host.register_dispatch_consumer))
        self.assertTrue(callable(host.replace_dispatch_consumers))
        self.assertTrue(callable(host.clear_dispatch_consumers))
        self.assertTrue(callable(host.set_test_time_lora_enabled))

    def test_residual_runtime_host_exposes_test_time_lora_enable_switch(self) -> None:
        host = importlib.import_module("tllm.runtime.residual_runtime")
        old_value = host.RUNTIME.test_time_lora_enabled
        try:
            host.set_test_time_lora_enabled(True)
            self.assertTrue(host.RUNTIME.test_time_lora_enabled)
            host.set_test_time_lora_enabled(False)
            self.assertFalse(host.RUNTIME.test_time_lora_enabled)
        finally:
            host.RUNTIME.test_time_lora_enabled = old_value

    def test_public_tllm_make_llm_is_runtime_hook_installing_entrypoint(self) -> None:
        tllm = importlib.import_module("tllm")
        host = importlib.import_module("tllm.runtime.residual_runtime")

        # The public façade routes through the default TllmRuntime, whose
        # make_llm delegates to the hook-installing residual_runtime host.
        self.assertEqual(tllm.default_runtime().make_llm.__func__.__qualname__, "TllmRuntime.make_llm")
        from unittest import mock

        with mock.patch.object(host, "make_llm", return_value="sentinel") as host_make:
            self.assertEqual(tllm.make_llm(model_name="m"), "sentinel")
        host_make.assert_called_once()

    def test_util_tools_exposes_only_explicit_plain_llm_constructor(self) -> None:
        tools = importlib.import_module("tllm.util.tools")

        self.assertTrue(callable(tools.make_plain_llm))
        self.assertFalse(hasattr(tools, "make_llm"))


if __name__ == "__main__":
    unittest.main()
