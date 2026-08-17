#!/usr/bin/env python3
"""Unit tests for runtime typing contracts on retained framework code."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import typing
import unittest

import torch

from tllm.contracts.runtime_context import RuntimeContext
from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.consumers.esamp.engine import ESampTrainEngine
from tllm.runtime import consumer_compat
from tllm.runtime import residual_runtime as esamp_runtime
from tllm.runtime.ports.provider_registry import ProviderRegistry
from tllm.runtime.vllm_patch import port_runtime_hooks
from tllm.runtime.vllm_patch import common_hooks


class RuntimeTypingContractsUnitTest(unittest.TestCase):
    def test_runtime_context_runner_and_model_are_not_any(self) -> None:
        hints = typing.get_type_hints(RuntimeContext)

        self.assertIsNot(hints["runner"], typing.Any)
        self.assertIsNot(hints["model"], typing.Any)

    def test_common_hooks_build_runtime_context_runner_is_not_any(self) -> None:
        hints = typing.get_type_hints(common_hooks.build_runtime_context)

        self.assertIsNot(hints["runner"], typing.Any)

    def test_consumer_compat_does_not_expose_any_consumer_parameters(self) -> None:
        for fn in (
            consumer_compat.consumer_subscriptions,
            consumer_compat.on_step_end,
            consumer_compat.dispatch_consumer_event,
            consumer_compat.synchronize,
        ):
            hints = typing.get_type_hints(fn)
            for name, hint in hints.items():
                if name == "return":
                    continue
                if name == "ctx":
                    continue
                self.assertIsNot(
                    hint,
                    typing.Any,
                    msg=f"{fn.__module__}.{fn.__name__} parameter `{name}` should not be Any",
                )

    def test_provider_registry_is_generic_over_provider_type(self) -> None:
        register_hints = typing.get_type_hints(ProviderRegistry.register)
        get_hints = typing.get_type_hints(ProviderRegistry.get)

        self.assertIsNot(register_hints["provider"], typing.Any)
        self.assertIsNot(get_hints["return"], typing.Any)

    def test_runtime_context_exposes_runner_protocol_alias(self) -> None:
        self.assertTrue(hasattr(inspect.getmodule(RuntimeContext), "RunnerLike"))

    def test_esamp_consumer_model_annotations_are_not_any(self) -> None:
        resolve_hints = typing.get_type_hints(ESampConsumer._resolve_layer_with_fallback)
        template_hints = typing.get_type_hints(ESampConsumer._maybe_prepare_initializer)

        self.assertIsNot(resolve_hints["model"], typing.Any)
        self.assertIsNot(template_hints["model"], typing.Any)

    def test_esamp_engine_sampling_prompt_inputs_are_tensors(self) -> None:
        assign_hints = typing.get_type_hints(ESampTrainEngine.assign_sampling_model_bank_slots)
        predict_hints = typing.get_type_hints(ESampTrainEngine.predict_hidden_for_sampling)

        self.assertIs(assign_hints["prompt_idxs"], torch.Tensor)
        self.assertIs(predict_hints["prompt_idxs"], torch.Tensor)

    def test_esamp_engine_device_layout_state_is_not_nullable(self) -> None:
        state_hints = typing.get_type_hints(type(ESampTrainEngine(hidden_dim=1, lr=0.0).state))
        engine_text = Path("tllm/consumers/esamp/engine.py").read_text(encoding="utf-8")

        self.assertIs(state_hints["device"], torch.device)
        self.assertIs(state_hints["hidden_dtype"], torch.dtype)
        self.assertNotIn("device is None", engine_text)
        self.assertNotIn("hidden_dtype is None", engine_text)

    def test_esamp_engine_graph_storage_does_not_probe_external_tensor_types(self) -> None:
        text = Path("tllm/consumers/esamp/engine.py").read_text(encoding="utf-8")
        body = text.split("    def _ensure_graph_storage(", 1)[1].split("    def _capture_graph(", 1)[0]

        self.assertNotIn("isinstance(src_input, torch.Tensor)", body)
        self.assertNotIn("isinstance(tgt_input, torch.Tensor)", body)

    def test_esamp_engine_hot_path_uses_resource_requirements_not_nullable_fallbacks(self) -> None:
        text = Path("tllm/consumers/esamp/engine.py").read_text(encoding="utf-8")
        prediction_body = text.split("    def predict_hidden_for_sampling_capture(", 1)[1].split("    def _replay_graph_with_disable_fallback(", 1)[0]

        self.assertNotIn("s.pipeline is not None else", text)
        self.assertNotIn("shared is None", prediction_body)
        self.assertIn("_require_shared()", prediction_body)
        self.assertIn("_require_model_bank_train_resources()", text)

    def test_esamp_runtime_public_runtime_helpers_are_not_any(self) -> None:
        runner_hints = typing.get_type_hints(esamp_runtime.runner_uses_compilation_or_cudagraph)
        resolve_hints = typing.get_type_hints(esamp_runtime.resolve_module_by_path_with_fallback)
        runtime_hints = typing.get_type_hints(esamp_runtime.ResidualRuntimeState)

        self.assertIsNot(runner_hints["runner"], typing.Any)
        self.assertIsNot(resolve_hints["model"], typing.Any)
        self.assertIsNot(runtime_hints["consumer"], typing.Any)

    def test_esamp_runtime_exposes_runtime_consumer_alias(self) -> None:
        self.assertTrue(hasattr(esamp_runtime, "RuntimeConsumer"))

    def test_port_runtime_hooks_decode_entrypoints_are_not_any(self) -> None:
        prepare_hints = typing.get_type_hints(port_runtime_hooks.prepare_decode_localization)
        tap_hints = typing.get_type_hints(port_runtime_hooks.build_tap_path_list)
        dispatch_hints = typing.get_type_hints(port_runtime_hooks.dispatch_decode_port_bundles)

        self.assertIsNot(prepare_hints["core"], typing.Any)
        self.assertIsNot(prepare_hints["runner"], typing.Any)
        self.assertIsNot(tap_hints["core"], typing.Any)
        self.assertIsNot(dispatch_hints["core"], typing.Any)
        self.assertIsNot(dispatch_hints["runner"], typing.Any)

    def test_getattr_probes_outside_quarantine_only_decrease(self) -> None:
        """AST gate: no new getattr duck-typing probes outside the quarantine dirs.

        ``tllm/runtime/vllm_patch`` and ``tllm/backends`` are the sanctioned
        quarantine for probing foreign (vLLM) objects. Everywhere else, tLLM
        internal objects must be reached through typed attributes; a
        ``getattr(obj, "name", default)`` probe on an internal object is a
        typing-discipline violation. Existing sites are recorded as
        ``file::attribute`` counts in ``tools/getattr_probe_baseline.json``
        (count-based so line drift is not flagged); the counts may only
        shrink.
        """
        quarantine = ("tllm/runtime/vllm_patch", "tllm/backends")
        current: dict[str, int] = {}
        for path in sorted(Path("tllm").rglob("*.py")):
            sp = str(path)
            if "__pycache__" in sp or any(sp.startswith(q) for q in quarantine):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=sp)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"):
                    continue
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    key = f"{sp}::{node.args[1].value}"
                    current[key] = current.get(key, 0) + 1

        baseline: dict[str, int] = json.loads(Path("tools/getattr_probe_baseline.json").read_text(encoding="utf-8"))
        violations = sorted(
            f"{key}: {count} > baseline {baseline.get(key, 0)}" for key, count in current.items() if count > baseline.get(key, 0)
        )
        self.assertEqual(
            violations,
            [],
            msg=(
                "new getattr duck-typing probes outside the quarantine dirs "
                "(tllm/runtime/vllm_patch, tllm/backends) are forbidden; reach "
                "tLLM internals through typed attributes instead:\n" + "\n".join(violations)
            ),
        )


if __name__ == "__main__":
    unittest.main()
