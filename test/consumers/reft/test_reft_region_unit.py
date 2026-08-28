#!/usr/bin/env python3
"""Unit tests for the ReFT (LoReFT) intervention workload and primitive.

The publish path is driven through full signed steps (compile -> publish ->
prepare -> advance anchors -> version audit).  The wrapper math and row
selection are tested directly against pyreft's LoReFT formula.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.reft.config import ReftConsumerConfig
from tllm.consumers.reft.loader import load_loreft_interventions
from tllm.consumers.reft.region_workload import (
    ReftInterventionWorkload,
    build_reft_region_workload,
)
from tllm.contracts.activation_patch import ActivationPatchSpec
from tllm.kit import clear_host_channels
from tllm.runtime import residual_runtime
from tllm.runtime.activation_patch import LoreftBank, loreft_delta, prefill_row_mask
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_plugins import reconstruct_worker_bindings
from tllm.workload import declare_workload

_HIDDEN = 8
_RANK = 2
_LAYERS = (3, 7)


def _config(**overrides: object) -> ReftConsumerConfig:
    values: dict[str, object] = {
        "consumer_id": "reft",
        "layers": _LAYERS,
        "rank": _RANK,
        "last_tokens": 2,
    }
    values.update(overrides)
    return ReftConsumerConfig(**values)  # type: ignore[arg-type]


def _spec(rows: int = 4) -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(rows, _HIDDEN, torch.float32, torch.device("cpu"))


def _compile(workload: ReftInterventionWorkload) -> tuple[TllmRuntime, object]:
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    runtime.cold_specialize_workloads(_spec())
    compilation = runtime.compile_workloads(
        export_main_workload_timeline(
            BackendCapabilities(
                backend_id="vllm-0.27-mrv2",
                runner_family="mrv2",
                sampler_stages=frozenset(),
                native_min_p=False,
                tuple_sampler_return=False,
                split_execute_and_sample=True,
                graph_safety=GraphSafety.EAGER_ONLY,
                hidden_locators=frozenset(),
            ),
            graph_mode="eager",
            static_headroom_bytes=1 << 20,
        )
    )
    return runtime, compilation


def _drive_signed_step(runtime: TllmRuntime, compilation: object, step_id: int = 7):
    state = ResidualRuntimeState()
    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation, bindings=runtime.workload_execution_bindings
        )
        residual_runtime.prepare_physical_workgraph_step(step_id)
        for region in compilation.decision.execution_regions:
            residual_runtime.advance_workload_anchor(region.placement_anchor)
        execution = state.workload_execution
        assert execution is not None
        version = execution.state_versions.observed_version("intervention", None)
    return version


def _trained_params(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    source_w = torch.randn(len(_LAYERS), _RANK, _HIDDEN, generator=generator)
    source_b = torch.randn(len(_LAYERS), _RANK, generator=generator)
    rotate = torch.randn(len(_LAYERS), _HIDDEN, _RANK, generator=generator)
    return source_w, source_b, rotate


class _ReftTestBase(unittest.TestCase):
    def setUp(self) -> None:
        clear_host_channels()


class TestActivationPatchSpec(_ReftTestBase):
    def test_spec_validation_fails_closed(self) -> None:
        with self.assertRaises(TypeError):
            ActivationPatchSpec(layers=())
        with self.assertRaises(ValueError):
            ActivationPatchSpec(layers=(2, 2), last_tokens=1)
        with self.assertRaises(TypeError):
            ActivationPatchSpec(layers=(-1,), last_tokens=1)
        with self.assertRaises(ValueError):
            ActivationPatchSpec(layers=(1,))
        with self.assertRaises(ValueError):
            ActivationPatchSpec(layers=(1,), last_tokens=1, phase="decode")

    def test_spec_sorts_layers(self) -> None:
        spec = ActivationPatchSpec(layers=(7, 3), first_tokens=1)
        self.assertEqual(spec.layers, (3, 7))


class TestReftDeclaration(_ReftTestBase):
    def test_declaration_signs_one_extern_transition_region(self) -> None:
        workload = build_reft_region_workload(_config())
        _, compilation = _compile(workload)
        regions = compilation.decision.execution_regions
        self.assertEqual(tuple(region.tasks for region in regions), (("publish",),))
        self.assertEqual(
            tuple(region.implementation_kind for region in regions), ("extern",)
        )
        declared = declare_workload(workload)
        effects = {
            task.name: tuple(effect.kind for effect in task.effects)
            for task in declared.ir.tasks
        }
        self.assertEqual(effects["publish"], ("transition_maybe",))

    def test_specialization_key_pins_layers_rank_and_positions(self) -> None:
        workload = build_reft_region_workload(_config())
        key = workload.native_workload_resources().specialization_key
        self.assertEqual(key["layers"], "3,7")
        self.assertEqual(key["rank"], "2")
        self.assertEqual(key["last_tokens"], "2")

    def test_validate_config_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive rank"):
            ReftInterventionWorkload(_config(rank=0))
        with self.assertRaisesRegex(ValueError, "positive learning_rate"):
            ReftInterventionWorkload(_config(learning_rate=0.0))
        with self.assertRaises(TypeError):
            ReftInterventionWorkload(_config(layers=()))

    def test_cold_specialize_shapes_and_orthonormal_rotate(self) -> None:
        workload = build_reft_region_workload(_config())
        workload.cold_specialize(_spec())
        params = workload.params
        self.assertEqual(tuple(params["source_w"].shape), (2, _RANK, _HIDDEN))
        self.assertEqual(tuple(params["source_b"].shape), (2, _RANK))
        rotate = params["rotate"]
        self.assertEqual(tuple(rotate.shape), (2, _HIDDEN, _RANK))
        for slot in range(2):
            gram = rotate[slot].T @ rotate[slot]
            self.assertTrue(
                torch.allclose(gram, torch.eye(_RANK), atol=1e-5),
                f"rotate[{slot}] is not orthonormal: {gram}",
            )


class TestReftSignedPublish(_ReftTestBase):
    def test_push_lands_through_signed_step(self) -> None:
        workload = build_reft_region_workload(_config())
        runtime, compilation = _compile(workload)
        source_w, source_b, rotate = _trained_params()
        workload.push_intervention(source_w, source_b, rotate)

        version = _drive_signed_step(runtime, compilation)

        self.assertEqual(version, 1)
        self.assertEqual(workload.state_version, 1)
        self.assertTrue(torch.equal(workload.params["source_w"].detach(), source_w))
        self.assertTrue(torch.equal(workload.params["rotate"].detach(), rotate))
        bank = workload.bank_getter()()
        self.assertIsNotNone(bank)
        assert bank is not None
        self.assertEqual(bank.version, 1)
        self.assertTrue(bank.active)
        # The bank mirrors the signed State parameters (synced, not aliased:
        # engine steps can execute on a reconstructed instance).
        self.assertTrue(torch.equal(bank.source_w, workload.params["source_w"]))

    def test_zero_push_step_creates_inactive_baseline_bank(self) -> None:
        workload = build_reft_region_workload(_config())
        runtime, compilation = _compile(workload)

        version = _drive_signed_step(runtime, compilation)

        self.assertEqual(version, 0)
        bank = workload.bank_getter()()
        assert bank is not None
        self.assertEqual(bank.version, 0)
        self.assertFalse(bank.active)  # unpatched baseline until the first push

    def test_push_fails_closed(self) -> None:
        workload = build_reft_region_workload(_config())
        source_w, source_b, rotate = _trained_params()

        with self.assertRaisesRegex(RuntimeError, "cold_specialize"):
            workload.push_intervention(source_w, source_b, rotate)

        workload.cold_specialize(_spec())
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            workload.push_intervention(source_w[:, :1], source_b, rotate)

        disabled = build_reft_region_workload(_config(enabled=False))
        disabled.cold_specialize(_spec())
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            disabled.push_intervention(source_w, source_b, rotate)

        workload.push_intervention(source_w, source_b, rotate)
        with self.assertRaisesRegex(RuntimeError, "already has staged weights"):
            workload.push_intervention(source_w, source_b, rotate)

    def test_worker_spec_roundtrip(self) -> None:
        workload = build_reft_region_workload(_config())
        _, compilation = _compile(workload)
        worker_spec = workload.native_workload_resources().worker_spec
        self.assertIsNotNone(worker_spec)
        bindings = reconstruct_worker_bindings(
            compilation, (worker_spec,), specialization=_spec()
        )
        self.assertEqual(
            bindings.declarations[0].ir.semantic_key,
            declare_workload(workload).ir.semantic_key,
        )
        self.assertEqual(set(bindings.state_versions), {"intervention"})


class TestLoreftPrimitive(_ReftTestBase):
    def test_delta_matches_pyreft_loreft_forward(self) -> None:
        spec = ActivationPatchSpec(layers=_LAYERS, last_tokens=2)
        source_w, source_b, rotate = _trained_params()
        bank = LoreftBank(spec=spec, source_w=source_w, source_b=source_b, rotate=rotate)
        rows = torch.randn(5, _HIDDEN)
        for slot in range(2):
            # Reference: pyreft LoreftIntervention.forward with linear act_fn:
            #   base + (W h + b - h R) @ R^T, so the delta is the second term.
            expected = (
                torch.nn.functional.linear(rows, source_w[slot], source_b[slot])
                - rows @ rotate[slot]
            ) @ rotate[slot].T
            self.assertTrue(
                torch.allclose(loreft_delta(bank, slot, rows), expected, atol=1e-6)
            )

    def test_bank_publish_validates_version_and_shape(self) -> None:
        spec = ActivationPatchSpec(layers=_LAYERS, last_tokens=2)
        source_w, source_b, rotate = _trained_params()
        bank = LoreftBank(spec=spec, source_w=source_w, source_b=source_b, rotate=rotate)
        new = _trained_params(seed=1)
        bank.publish(*new, version=1)
        self.assertTrue(torch.equal(bank.source_w, new[0]))
        with self.assertRaisesRegex(ValueError, "monotonically increasing"):
            bank.publish(*new, version=1)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            bank.publish(new[0][:, :1], new[1], new[2], version=2)

    def test_prefill_row_mask_selects_prompt_edges(self) -> None:
        # Two requests: rows 0-2 (prompt len 3) and rows 3-7 (prompt len 5).
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
        query_start_loc = torch.tensor([0, 3, 8])
        seq_lens = torch.tensor([3, 5])
        computed = torch.tensor([0, 0])
        mask = prefill_row_mask(
            positions, query_start_loc, seq_lens, computed,
            first_tokens=1, last_tokens=2,
        )
        self.assertEqual(
            mask.tolist(),
            # req0: pos0 (first) + pos1,2 (last 2); req1: pos0 + pos3,4
            [True, True, True, True, False, False, True, True],
        )

    def test_prefill_row_mask_excludes_decode_and_continuation_chunks(self) -> None:
        positions = torch.tensor([0, 1, 2, 9, 10])
        query_start_loc = torch.tensor([0, 3, 5])
        seq_lens = torch.tensor([3, 11])
        # Request 1 is mid-chunk (computed>0): excluded rather than mis-selected.
        computed = torch.tensor([0, 8])
        mask = prefill_row_mask(
            positions, query_start_loc, seq_lens, computed,
            first_tokens=1, last_tokens=1,
        )
        self.assertEqual(
            mask.tolist(), [True, False, True, False, False]
        )

    def test_loader_roundtrip(self) -> None:
        spec_layers = (3, 7)
        with tempfile.TemporaryDirectory() as tmp:
            for layer in spec_layers:
                torch.save(
                    {
                        "weight": torch.randn(_RANK, _HIDDEN),
                        "bias": torch.randn(_RANK),
                        "rotate_layer": torch.randn(_HIDDEN, _RANK),
                    },
                    Path(tmp) / f"intkey_layer.{layer}.comp.block_output.unit.pos.nunit.1#0.bin",
                )
            loaded = load_loreft_interventions(tmp, spec_layers)
            self.assertEqual(sorted(loaded), [3, 7])
            self.assertEqual(tuple(loaded[3]["source_w"].shape), (_RANK, _HIDDEN))
            self.assertEqual(tuple(loaded[7]["rotate"].shape), (_HIDDEN, _RANK))
            with self.assertRaisesRegex(FileNotFoundError, "layers \\[9\\]"):
                load_loreft_interventions(tmp, (9,))


if __name__ == "__main__":
    unittest.main()
