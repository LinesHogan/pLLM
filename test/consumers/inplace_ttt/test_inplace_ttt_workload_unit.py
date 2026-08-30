"""Native exact In-Place-TTT and overlay lifecycle contracts."""

from __future__ import annotations

from typing import cast
import unittest
from unittest import mock

import torch

from tllm.backends.protocol import (
    BackendCapabilities,
    GraphSafety,
    VersionedParameterOverlayCapabilities,
)
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.inplace_ttt.workload import (
    InPlaceTTTBatch,
    InPlaceTTTConfig,
    InPlaceTTTFastWeightState,
    InPlaceTTTKey,
    InPlaceTTTPhase,
    InPlaceTTTWorkload,
)
from test.core.oracles.inplace_ttt import oracle_exact_step
from tllm.contracts.request_identity import RequestFinishedBatch, RequestIdentity
from tllm.runtime.parameter_overlay import (
    BaseWeightLease,
    DenseOperandCapture,
    DenseOperandCaptureBatch,
    OverlayCaptureRows,
    OverlayLeaseKey,
    OverlayRowPhase,
    OverlayTarget,
)
from tllm.runtime import residual_runtime
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_bindings import LocalRuntimeServices
from tllm.schedule import MainWorkloadTimeline, WorkloadCompilation


def _timeline(graph_mode: str = "eager") -> MainWorkloadTimeline:
    return export_main_workload_timeline(
        BackendCapabilities(
            backend_id="vllm-0.27-mrv2",
            runner_family="mrv2",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.EAGER_ONLY,
            parameter_overlay=VersionedParameterOverlayCapabilities(
                representations=frozenset({"dense_replace"}),
                scopes=frozenset({"request"}),
                phases=frozenset({"prefill", "decode"}),
                pinning=frozenset({"explicit_request_advance"}),
                mixed_versions=True,
                operand_captures=frozenset({"mlp_down_proj"}),
                graph_safety=GraphSafety.EAGER_ONLY,
                tensor_parallel_sizes=frozenset({1}),
            ),
        ),
        graph_mode=graph_mode,
        static_headroom_bytes=1 << 20,
    )


def _build(
    config: InPlaceTTTConfig,
) -> tuple[InPlaceTTTWorkload, TllmRuntime[InPlaceTTTWorkload], WorkloadCompilation]:
    workload = InPlaceTTTWorkload(config)
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=1 << 20
    )
    runtime.cold_specialize_workloads(
        RuntimeSpecializationSpec(2, 3, torch.float64, torch.device("cpu"))
    )
    return workload, runtime, runtime.compile_workloads(_timeline())


def _drive(
    runtime: TllmRuntime[InPlaceTTTWorkload],
    compilation: WorkloadCompilation,
    step_id: int,
    finished: RequestFinishedBatch | None = None,
    captured: DenseOperandCaptureBatch | None = None,
) -> tuple[int, int]:
    state = ResidualRuntimeState()
    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation, bindings=runtime.workload_execution_bindings
        )
        execution = state.workload_execution
        assert execution is not None
        if finished is not None and execution.stage_request_finished(finished) != 1:
            raise AssertionError("In-Place lifecycle port was not staged")
        if (
            captured is not None
            and execution.stage_dense_operand_captured(captured) != 1
        ):
            raise AssertionError("In-Place operand capture port was not staged")
        residual_runtime.prepare_physical_workgraph_step(step_id)
        for region in compilation.decision.execution_regions:
            residual_runtime.advance_workload_anchor(region.placement_anchor)
        return (
            execution.state_versions.observed_version("fast_weights", None),
            execution.state_versions.observed_version("overlay_versions", None),
        )


def _captured_batch(
    request: RequestIdentity,
    *,
    layers: tuple[int, ...],
    phase: OverlayRowPhase,
    hidden: torch.Tensor,
    target: torch.Tensor,
    base_weight: torch.Tensor,
    step_id: int,
) -> DenseOperandCaptureBatch:
    packed_rows = tuple(range(hidden.shape[0]))
    return DenseOperandCaptureBatch(
        tuple(
            DenseOperandCapture(
                step_id=step_id,
                target=OverlayTarget(layer, "mlp", "down_proj.weight"),
                rows=(
                    OverlayCaptureRows(
                        target=OverlayTarget(
                            layer, "mlp", "down_proj.weight"
                        ),
                        request=request,
                        phase=phase,
                        packed_rows=packed_rows,
                        token_offsets=packed_rows,
                    ),
                ),
                down_proj_input=hidden,
                mlp_input=target,
                base_weight=BaseWeightLease(
                    OverlayTarget(layer, "mlp", "down_proj.weight"),
                    base_weight,
                ),
            )
            for layer in layers
        )
    )


class InPlaceTTTWorkloadUnitTest(unittest.TestCase):
    def test_native_multistep_tail_matches_frozen_historical_oracle(self) -> None:
        state = InPlaceTTTFastWeightState()
        key = InPlaceTTTKey(RequestIdentity("oracle", 0, 0), 0)
        base = torch.tensor(
            [[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]],
            dtype=torch.float64,
        )
        first_hidden = torch.tensor(
            [[1.0, -0.5, 0.25]], dtype=torch.float64
        )
        first_target = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
        first_oracle = oracle_exact_step(
            first_hidden,
            first_target,
            base,
            chunk_size=2,
            learning_rate=0.3,
        )
        first = state.execute(
            InPlaceTTTBatch(
                key,
                InPlaceTTTPhase.PREFILL,
                first_hidden,
                first_target,
                base,
            ),
            chunk_size=2,
            learning_rate=0.3,
            update_enabled=True,
        )
        torch.testing.assert_close(first.output, first_oracle.output)
        torch.testing.assert_close(first.weight, first_oracle.weight)

        second_hidden = torch.tensor(
            [[0.2, 0.4, -0.3]], dtype=torch.float64
        )
        second_target = torch.tensor([[0.1, 0.75]], dtype=torch.float64)
        second_oracle = oracle_exact_step(
            second_hidden,
            second_target,
            base,
            chunk_size=2,
            learning_rate=0.3,
            past_hidden=first_oracle.tail_hidden,
            past_target=first_oracle.tail_target,
            past_weight=first_oracle.weight,
        )
        second = state.execute(
            InPlaceTTTBatch(
                key,
                InPlaceTTTPhase.PREFILL,
                second_hidden,
                second_target,
                base,
            ),
            chunk_size=2,
            learning_rate=0.3,
            update_enabled=True,
        )
        torch.testing.assert_close(second.output, second_oracle.output)
        torch.testing.assert_close(second.weight, second_oracle.weight)
        self.assertEqual(second.update_count, second_oracle.update_count)
        self.assertEqual(second.trained_rows, second_oracle.trained_rows)

    def test_dense_overlay_rejects_non_eager_backend_timeline(self) -> None:
        workload = InPlaceTTTWorkload(InPlaceTTTConfig())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(
                2, 3, torch.float64, torch.device("cpu")
            )
        )

        with self.assertRaisesRegex(RuntimeError, "eager_only"):
            runtime.compile_workloads(_timeline("piecewise"))

    def test_captured_bundle_updates_multiple_requests_and_layers_once(self) -> None:
        workload, runtime, compilation = _build(
            InPlaceTTTConfig(layers=(0, 1), chunk_size=1)
        )
        first = RequestIdentity("first", 0, 0)
        second = RequestIdentity("second", 1, 0)
        hidden = torch.tensor(
            [[1.0, 0.0, 0.5], [0.5, 1.0, -0.5], [2.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        target = torch.tensor(
            [[0.5, 1.0], [1.0, -0.5], [0.25, 0.75]],
            dtype=torch.float64,
        )
        base = torch.zeros((2, 3), dtype=torch.float64)
        captures = tuple(
            DenseOperandCapture(
                step_id=1,
                target=OverlayTarget(layer, "mlp", "down_proj.weight"),
                rows=(
                    OverlayCaptureRows(
                        target=OverlayTarget(
                            layer, "mlp", "down_proj.weight"
                        ),
                        request=first,
                        phase=OverlayRowPhase.PREFILL,
                        packed_rows=(0, 1),
                        token_offsets=(0, 1),
                    ),
                    OverlayCaptureRows(
                        target=OverlayTarget(
                            layer, "mlp", "down_proj.weight"
                        ),
                        request=second,
                        phase=OverlayRowPhase.DECODE,
                        packed_rows=(2,),
                        token_offsets=(3,),
                    ),
                ),
                down_proj_input=hidden,
                mlp_input=target,
                base_weight=BaseWeightLease(
                    OverlayTarget(layer, "mlp", "down_proj.weight"), base
                ),
            )
            for layer in (0, 1)
        )

        self.assertEqual(
            _drive(
                runtime,
                compilation,
                1,
                captured=DenseOperandCaptureBatch(captures),
            ),
            (1, 1),
        )
        (step_bundle,) = runtime.drain(InPlaceTTTWorkload.captured_steps)
        self.assertEqual(len(step_bundle.steps), 4)
        self.assertEqual(
            tuple(step.update_count for step in step_bundle.steps),
            (2, 0, 2, 0),
        )
        (publication_bundle,) = runtime.drain(
            InPlaceTTTWorkload.captured_publications
        )
        versions = tuple(
            publication.version
            for publication in publication_bundle.publications
            if publication.version is not None
        )
        self.assertEqual(len(versions), 2)
        self.assertEqual(workload.update_count, 4)
        self.assertEqual(len(workload.fast_weights.slots), 2)
        self.assertEqual(len(workload.overlay_versions.store.lineages()), 2)

        _drive(
            runtime,
            compilation,
            2,
            finished=RequestFinishedBatch((first,)),
        )
        self.assertEqual(workload.fast_weights.slots, {})
        self.assertEqual(workload.overlay_versions.store.lineages(), ())

    def test_adjacent_mixed_phases_share_one_enabled_update_group(self) -> None:
        workload, runtime, compilation = _build(
            InPlaceTTTConfig(
                chunk_size=2,
                learning_rate=0.3,
                update_prefill=True,
                update_decode=True,
            )
        )
        request = RequestIdentity("mixed", 0, 0)
        target_spec = OverlayTarget(0, "mlp", "down_proj.weight")
        hidden = torch.tensor(
            [[1.0, -0.5, 0.25], [0.2, 0.4, -0.3]],
            dtype=torch.float64,
        )
        target = torch.tensor(
            [[0.5, -0.25], [0.1, 0.75]], dtype=torch.float64
        )
        base = torch.tensor(
            [[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]],
            dtype=torch.float64,
        )
        capture = DenseOperandCapture(
            step_id=1,
            target=target_spec,
            rows=(
                OverlayCaptureRows(
                    target_spec,
                    request,
                    OverlayRowPhase.PREFILL,
                    (0,),
                    (0,),
                ),
                OverlayCaptureRows(
                    target_spec,
                    request,
                    OverlayRowPhase.DECODE,
                    (1,),
                    (1,),
                ),
            ),
            down_proj_input=hidden,
            mlp_input=target,
            base_weight=BaseWeightLease(target_spec, base),
        )

        self.assertEqual(
            _drive(
                runtime,
                compilation,
                1,
                captured=DenseOperandCaptureBatch((capture,)),
            ),
            (1, 1),
        )
        (step_bundle,) = runtime.drain(InPlaceTTTWorkload.captured_steps)
        self.assertEqual(len(step_bundle.steps), 1)
        self.assertEqual(step_bundle.steps[0].update_count, 1)
        expected = oracle_exact_step(
            hidden,
            target,
            base,
            chunk_size=2,
            learning_rate=0.3,
        )
        torch.testing.assert_close(step_bundle.steps[0].weight, expected.weight)
        (publication_bundle,) = runtime.drain(
            InPlaceTTTWorkload.captured_publications
        )
        self.assertEqual(len(publication_bundle.publications), 1)
        lineage = publication_bundle.publications[0].version
        assert lineage is not None
        self.assertEqual(
            len(workload.overlay_versions.store.versions(lineage.lineage)), 1
        )

    def test_exact_chunk_output_precedes_update_and_publishes_owned_weight(self) -> None:
        config = InPlaceTTTConfig(
            layers=(0,), chunk_size=2, learning_rate=0.3
        )
        workload, runtime, compilation = _build(config)
        request = RequestIdentity("req-0", 0, 0)
        hidden = torch.tensor(
            [[1.0, -0.5, 0.25], [0.2, 0.4, -0.3]], dtype=torch.float64
        )
        target = torch.tensor(
            [[0.5, -0.25], [0.1, 0.75]], dtype=torch.float64
        )
        base = torch.tensor(
            [[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]], dtype=torch.float64
        )
        expected_output = hidden @ base.transpose(0, 1)
        expected_weight = base + 0.3 * (target.transpose(0, 1) @ hidden)
        self.assertEqual(
            _drive(
                runtime,
                compilation,
                1,
                captured=_captured_batch(
                    request,
                    layers=(0,),
                    phase=OverlayRowPhase.PREFILL,
                    hidden=hidden,
                    target=target,
                    base_weight=base,
                    step_id=1,
                ),
            ),
            (1, 1),
        )
        step = runtime.drain(InPlaceTTTWorkload.captured_steps)[0].steps[0]
        torch.testing.assert_close(step.output, expected_output)
        torch.testing.assert_close(step.weight, expected_weight)
        publication = runtime.drain(
            InPlaceTTTWorkload.captured_publications
        )[0].publications[0]
        assert publication.version is not None
        materialized = workload.overlay_versions.store.materialize(
            publication.version
        )
        torch.testing.assert_close(materialized["weight"], expected_weight)
        materialized["weight"].zero_()
        torch.testing.assert_close(
            workload.overlay_versions.store.materialize(publication.version)["weight"],
            expected_weight,
        )
        self.assertEqual(workload.update_count, 1)

    def test_tail_and_decode_disabled_are_conditional_state_transitions(self) -> None:
        config = InPlaceTTTConfig(chunk_size=4, update_decode=False)
        workload, runtime, compilation = _build(config)
        request = RequestIdentity("req-tail", 1, 0)
        hidden = torch.ones((2, 3), dtype=torch.float64)
        target = torch.ones((2, 2), dtype=torch.float64)
        base = torch.ones((2, 3), dtype=torch.float64)
        self.assertEqual(
            _drive(
                runtime,
                compilation,
                1,
                captured=_captured_batch(
                    request,
                    layers=(0,),
                    phase=OverlayRowPhase.PREFILL,
                    hidden=hidden,
                    target=target,
                    base_weight=base,
                    step_id=1,
                ),
            ),
            (1, 0),
        )
        self.assertEqual(
            runtime.drain(InPlaceTTTWorkload.captured_steps)[0]
            .steps[0]
            .tail_rows,
            2,
        )
        self.assertIsNone(
            runtime.drain(InPlaceTTTWorkload.captured_publications)[0]
            .publications[0]
            .version
        )

        decode_request = RequestIdentity("req-decode", 2, 0)
        self.assertEqual(
            _drive(
                runtime,
                compilation,
                2,
                captured=_captured_batch(
                    decode_request,
                    layers=(0,),
                    phase=OverlayRowPhase.DECODE,
                    hidden=hidden,
                    target=target,
                    base_weight=base,
                    step_id=2,
                ),
            ),
            (1, 0),
        )
        self.assertFalse(
            runtime.drain(InPlaceTTTWorkload.captured_steps)[0]
            .steps[0]
            .state_changed
        )
        self.assertEqual(workload.update_count, 0)

    def test_finished_request_releases_tail_and_overlay_lease(self) -> None:
        workload, runtime, compilation = _build(
            InPlaceTTTConfig(chunk_size=2)
        )
        request = RequestIdentity("req-finish", 0, 0)
        _drive(
            runtime,
            compilation,
            1,
            captured=_captured_batch(
                request,
                layers=(0,),
                phase=OverlayRowPhase.PREFILL,
                hidden=torch.ones((2, 3), dtype=torch.float64),
                target=torch.ones((2, 2), dtype=torch.float64),
                base_weight=torch.ones((2, 3), dtype=torch.float64),
                step_id=1,
            ),
        )
        publication = runtime.drain(
            InPlaceTTTWorkload.captured_publications
        )[0].publications[0]
        assert publication.version is not None
        lease_key = OverlayLeaseKey(request, publication.version.lineage)
        self.assertIsNotNone(workload.overlay_versions.store.pinned(lease_key))
        self.assertEqual(
            _drive(
                runtime,
                compilation,
                2,
                RequestFinishedBatch((request,)),
            ),
            (2, 2),
        )
        receipt = runtime.drain(InPlaceTTTWorkload.cleanup)[0]
        self.assertEqual(receipt.cleared_slots, 1)
        self.assertEqual(receipt.released_versions, 1)
        self.assertIsNone(workload.overlay_versions.store.pinned(lease_key))

    def test_finished_request_retires_every_layer_lineage_and_payload(self) -> None:
        workload, runtime, compilation = _build(
            InPlaceTTTConfig(layers=(0, 1), chunk_size=2)
        )
        request = RequestIdentity("req-multi-layer", 0, 0)
        lease_keys: list[OverlayLeaseKey] = []
        for step_id, layer in enumerate((0, 1), start=1):
            _drive(
                runtime,
                compilation,
                step_id,
                captured=_captured_batch(
                    request,
                    layers=(layer,),
                    phase=OverlayRowPhase.PREFILL,
                    hidden=torch.ones((2, 3), dtype=torch.float64),
                    target=torch.ones((2, 2), dtype=torch.float64),
                    base_weight=torch.ones((2, 3), dtype=torch.float64),
                    step_id=step_id,
                ),
            )
            publication = runtime.drain(
                InPlaceTTTWorkload.captured_publications
            )[0].publications[0]
            assert publication.version is not None
            lease_keys.append(
                OverlayLeaseKey(request, publication.version.lineage)
            )

        for lease_key in lease_keys:
            self.assertIsNotNone(workload.overlay_versions.store.pinned(lease_key))
        self.assertEqual(len(workload.overlay_versions.store.lineages()), 2)
        self.assertGreater(workload.overlay_versions.store.byte_size, 0)

        _drive(
            runtime,
            compilation,
            3,
            RequestFinishedBatch((request,)),
        )

        receipt = runtime.drain(InPlaceTTTWorkload.cleanup)[0]
        self.assertEqual(receipt.cleared_slots, 2)
        self.assertEqual(receipt.released_versions, 2)
        self.assertEqual(workload.overlay_versions.store.lineages(), ())
        self.assertEqual(workload.overlay_versions.store.byte_size, 0)
        for lease_key in lease_keys:
            self.assertIsNone(workload.overlay_versions.store.pinned(lease_key))

    def test_config_contains_algorithm_semantics_not_scheduler_controls(self) -> None:
        self.assertEqual(
            InPlaceTTTConfig.field_names(),
            (
                "enabled",
                "layers",
                "chunk_size",
                "learning_rate",
                "update_prefill",
                "update_decode",
            ),
        )
        services = cast(
            LocalRuntimeServices,
            InPlaceTTTWorkload().native_workload_resources().services,
        )
        self.assertEqual(services.diagnostics()["update_count"], 0)


if __name__ == "__main__":
    unittest.main()
