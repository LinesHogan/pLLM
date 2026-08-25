#!/usr/bin/env python3
"""Fail-closed admission tests for the fast-weight overlay provider."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety, RunnerBinding
from tllm.backends.vllm.v027_mrv1 import V027MRv1Driver
from tllm.ports.module_weight_view import ModuleWeightView
from tllm.runtime.fast_weight.runtime import (
    maybe_install_fast_weight_wrappers,
    restore_fast_weight_wrappers,
)


class _DownProj:
    def __init__(self) -> None:
        self.weight = torch.eye(2)
        self.bias = None

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(hidden, self.weight), None


class _GateUpProj:
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden, None


class _MLP:
    def __init__(self, *, supported: bool = True) -> None:
        self.gate_up_proj = _GateUpProj()
        self.act_fn = torch.nn.Identity()
        if supported:
            self.down_proj = _DownProj()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj.forward(hidden)
        activated = self.act_fn(gate_up)
        output, _ = self.down_proj.forward(activated)
        return output


class _DispatchPlan:
    def __init__(self, writes: tuple[object, ...]) -> None:
        self._flow = SimpleNamespace(writes=writes)

    def module_weight_writes(self) -> list[tuple[str, object, object]]:
        return [("test", self._flow, write) for write in self._flow.writes]


def _runtime(*, enabled: bool = True, layers: tuple[int, ...] = (0,), declared_layer: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            fast_weight_enabled=enabled,
            fast_weight_layers=layers,
            fast_weight_mode="exact",
            fast_weight_chunk_size=2,
            fast_weight_lr=0.1,
            fast_weight_decode_update_enabled=False,
            fast_weight_key_mode="request_sample",
            fast_weight_static_request_id="static",
            fast_weight_max_slots=64,
            fast_weight_max_bytes=0,
        ),
        dispatch_plan=_DispatchPlan(
            (ModuleWeightView.write(layer=declared_layer, module="mlp", phase="prefill"),)
        ),
        fast_weight_handles={},
        fast_weight_binding=None,
        fast_weight_hook_spec=None,
        fast_weight_state=None,
        fast_weight_phase="prefill",
        fast_weight_current_token_batch_meta=None,
        fast_weight_installed_count=0,
        fast_weight_skipped_count=0,
        reqid_to_promptidx={},
        reqid_to_sampleidx={},
        event_step_id=0,
    )


def _runner(*, tp_size: int = 1, graph_level: int = 0, supported: tuple[bool, ...] = (True,)) -> SimpleNamespace:
    layers = [SimpleNamespace(mlp=_MLP(supported=value)) for value in supported]
    return SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        compilation_config=SimpleNamespace(level=graph_level, cudagraph_mode=0),
        input_batch=SimpleNamespace(req_ids=[], num_reqs=0),
    )


def _binding(runner: object, *, capabilities: BackendCapabilities | None = None) -> RunnerBinding:
    driver = V027MRv1Driver()
    return RunnerBinding(
        driver=driver,
        runner=runner,
        capabilities=capabilities or driver.capabilities(),
    )


class FastWeightRuntimeUnitTest(unittest.TestCase):
    def test_repeated_install_is_noop_and_runtime_restore_recovers_forward(self) -> None:
        runtime = _runtime()
        runner = _runner()
        binding = _binding(runner)
        mlp = runner.model.model.layers[0].mlp
        original_func = mlp.forward.__func__

        self.assertEqual(
            maybe_install_fast_weight_wrappers(runtime=runtime, runner=runner, binding=binding),
            1,
        )
        self.assertEqual(
            maybe_install_fast_weight_wrappers(runtime=runtime, runner=runner, binding=binding),
            0,
        )
        self.assertEqual(restore_fast_weight_wrappers(runtime), 1)
        self.assertIs(mlp.forward.__func__, original_func)
        self.assertEqual(restore_fast_weight_wrappers(runtime), 0)

    def test_active_module_weight_write_requires_enabled_provider(self) -> None:
        runtime = _runtime(enabled=False)
        runner = _runner()
        with self.assertRaisesRegex(RuntimeError, "overlay provider is disabled"):
            maybe_install_fast_weight_wrappers(runtime=runtime, runner=runner, binding=_binding(runner))

    def test_active_module_weight_write_requires_backend_capability(self) -> None:
        runtime = _runtime()
        runner = _runner()
        caps = BackendCapabilities(
            backend_id="no-overlay",
            runner_family="mrv1",
            sampler_stages=frozenset(),
            native_min_p=True,
            tuple_sampler_return=True,
            split_execute_and_sample=True,
        )
        with self.assertRaisesRegex(RuntimeError, "does not support module_weight_view/wrapper/prefill"):
            maybe_install_fast_weight_wrappers(
                runtime=runtime,
                runner=runner,
                binding=_binding(runner, capabilities=caps),
            )

    def test_active_module_weight_write_requires_eager_tp1_and_matching_layers(self) -> None:
        cases = (
            (_runner(graph_level=1), _runtime(), "requires eager graph mode"),
            (_runner(tp_size=2), _runtime(), "tensor_parallel_size=1"),
            (_runner(supported=(True, True)), _runtime(layers=(0,), declared_layer=1), "configured layers"),
        )
        for runner, runtime, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    maybe_install_fast_weight_wrappers(
                        runtime=runtime,
                        runner=runner,
                        binding=_binding(runner),
                    )

    def test_install_failure_restores_every_module_installed_in_attempt(self) -> None:
        runtime = _runtime(layers=(0, 1), declared_layer=0)
        runtime.dispatch_plan = _DispatchPlan(
            (
                ModuleWeightView.write(layer=0, module="mlp", phase="prefill"),
                ModuleWeightView.write(layer=1, module="mlp", phase="prefill"),
            )
        )
        runner = _runner(supported=(True, False))
        first = runner.model.model.layers[0].mlp
        original = first.forward

        with self.assertRaisesRegex(RuntimeError, "down_proj"):
            maybe_install_fast_weight_wrappers(runtime=runtime, runner=runner, binding=_binding(runner))

        self.assertEqual(first.forward(torch.tensor([[1.0, 0.0]])).tolist(), [[1.0, 0.0]])
        self.assertIs(getattr(first.forward, "__func__", None), getattr(original, "__func__", None))
        self.assertEqual(runtime.fast_weight_handles, {})
        self.assertEqual(runtime.fast_weight_installed_count, 0)


if __name__ == "__main__":
    unittest.main()
