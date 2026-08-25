"""CPU proof for compiler-owned exact Recirculation Regions."""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest
import torch

from tllm.backends.protocol import (
    BackendCapabilities,
    BackendCapability,
    GraphSafety,
)
from tllm.backends.timeline import export_main_workload_timeline
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
from tllm.consumers.recirculation import (
    RecirculationConfig,
    RecirculationConsumer,
    RecirculationExecutionRequest,
    RecirculationRegionWorkload,
    run_serial_recirculation,
)
from tllm.runtime import residual_runtime
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_plugins import reconstruct_worker_bindings
from tllm.workgraph.recurrence import JacobiPolicy
from tllm.workload import declare_workload
from tllm.consumers.recirculation import region_workload


class _TinyCausalDecoder:
    layer_count = 3

    def __init__(self) -> None:
        self.kv: dict[tuple[int, int], torch.Tensor] = {}
        self.calls: list[tuple[int, int, bool]] = []

    def embed(self, token_id: int, position: int) -> torch.Tensor:
        return torch.tensor([[float(token_id), float(position + 1)]])

    def run_layer(
        self,
        layer: int,
        hidden: torch.Tensor,
        *,
        position: int,
        overwrite_kv: bool,
    ) -> torch.Tensor:
        self.calls.append((position, layer, overwrite_kv))
        previous = sum(
            (
                value
                for (cached_layer, cached_position), value in self.kv.items()
                if cached_layer == layer and cached_position < position
            ),
            torch.zeros_like(hidden),
        )
        output = hidden + previous + float(layer + 1)
        self.kv[(layer, position)] = output.detach().clone()
        return output

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.sum(dim=-1)


def _config() -> RecirculationConfig:
    return RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2)


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="recirculation-region-test",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.EAGER_ONLY,
        extensions=frozenset({BackendCapability.RECIRCULATION_FINITE_REQUEST}),
    )


def _timeline():
    return export_main_workload_timeline(
        _capabilities(), graph_mode="eager", static_headroom_bytes=1 << 20
    )


def _cold_specialize(
    workload: RecirculationRegionWorkload, *, rows: int = 3
) -> RuntimeSpecializationSpec:
    spec = RuntimeSpecializationSpec(
        rows=rows,
        hidden_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        model_layer_count=_TinyCausalDecoder.layer_count,
        max_prefill_tokens=rows,
    )
    workload.native_workload_resources().cold_specializer(spec)
    return spec


def test_finite_request_capability_is_closed_and_exported_to_timeline() -> None:
    capabilities = _capabilities()

    assert capabilities.capability_names() == frozenset(
        {BackendCapability.RECIRCULATION_FINITE_REQUEST.value}
    )
    assert BackendCapability.RECIRCULATION_FINITE_REQUEST.value in _timeline().capabilities
    with pytest.raises(TypeError, match=r"frozenset\[BackendCapability\]"):
        replace(capabilities, extensions=frozenset({"recirculation/finite_request"}))


def test_region_uses_one_rust_selected_grouping_and_matches_serial_oracle() -> None:
    config = _config()
    oracle = run_serial_recirculation(
        _TinyCausalDecoder(), RecirculationConsumer(config), (1, 2, 3)
    )
    workload = RecirculationRegionWorkload(config)
    _cold_specialize(workload)

    with (
        mock.patch.object(
            region_workload,
            "build_recirculation_graph",
            side_effect=AssertionError("runtime rebuilt a Recirculation graph"),
        ),
        mock.patch.object(
            region_workload._core,
            "compile_typed_recurrence_plan",
            side_effect=AssertionError("runtime compiled a typed recurrence plan"),
        ),
    ):
        native = workload.execute_exact(
            RecirculationExecutionRequest(_TinyCausalDecoder(), (1, 2, 3)), 4
        )

    assert native.schedule_kind == "wavefront"
    assert any(len(group) == 2 for group in native.schedule_groups)
    for actual, expected in zip(native.logits, oracle.logits, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(native.recurrent_states, oracle.recurrent_states, strict=True):
        torch.testing.assert_close(actual, expected)


def test_native_request_has_no_runtime_grouping_selector() -> None:
    with pytest.raises(TypeError):
        RecirculationExecutionRequest(_TinyCausalDecoder(), (1,), "serial")


def test_region_workload_has_one_affine_region_and_no_legacy_authority() -> None:
    workload = RecirculationRegionWorkload(_config())
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    compilation = runtime.compile_workloads(_timeline())

    assert runtime.consumers == ()
    assert not hasattr(workload, "legacy_consumer")
    assert declare_workload(workload).ir.approximation.value == "exact"
    assert compilation.decision.schedule_tree.kind == "hybrid"
    assert len(compilation.decision.schedule_tree.affine_regions) == 1
    assert tuple(region.tasks for region in compilation.decision.execution_regions) == (
        ("execute_exact",),
    )
    assert compilation.decision.materializations == ()


def test_region_rejects_normal_v027_timeline_before_execution() -> None:
    workload = RecirculationRegionWorkload(_config())
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    timeline = export_main_workload_timeline(
        V027MRv2Driver().capabilities(),
        graph_mode="eager",
        static_headroom_bytes=1 << 20,
    )

    assert BackendCapability.RECIRCULATION_FINITE_REQUEST.value not in timeline.capabilities
    with pytest.raises(
        RuntimeError, match="requires unavailable capabilities.*recirculation/finite_request"
    ):
        runtime.compile_workloads(timeline)


def test_native_region_runtime_executes_complete_request_without_retained_intermediate() -> None:
    workload = RecirculationRegionWorkload(_config())
    _cold_specialize(workload, rows=2)
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    compilation = runtime.compile_workloads(_timeline())
    state = ResidualRuntimeState()
    request = RecirculationExecutionRequest(_TinyCausalDecoder(), (1, 2))

    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation,
            bindings=runtime.workload_execution_bindings,
        )
        workload.resources.publish_request(request)
        residual_runtime.prepare_physical_workgraph_step(17)
        for region in compilation.decision.execution_regions:
            assert residual_runtime.advance_workload_anchor(region.placement_anchor)

        execution = state.workload_execution
        assert execution is not None
        assert execution.executor.stats.execute_count == 1
        assert execution.retained_value_count == 0

    result = workload.resources.result()
    assert result is not None
    assert result.step_id == 17
    assert result.schedule_kind == "wavefront"


def test_worker_reconstruction_preserves_signed_native_region() -> None:
    workload = RecirculationRegionWorkload(_config())
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    specialization = RuntimeSpecializationSpec(
        rows=2,
        hidden_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        model_layer_count=_TinyCausalDecoder.layer_count,
        max_prefill_tokens=2,
    )
    runtime.cold_specialize_workloads(specialization)
    compilation = runtime.compile_workloads(_timeline())
    spec = workload.native_workload_resources().worker_spec
    assert spec is not None
    assert "stage_artifact_bundle" in spec.config
    assert "serial_stage_artifact_bundle" not in spec.config
    assert "wavefront_stage_artifact_bundle" not in spec.config

    bindings = reconstruct_worker_bindings(
        compilation, (spec,), specialization=specialization
    )
    assert bindings.declarations[0].ir.semantic_key == declare_workload(workload).ir.semantic_key
    assert set(bindings.provided_inputs) == {"recirculation_request"}
    assert set(bindings.output_sinks) == {"recirculation_result"}


def test_exact_region_fails_closed_until_cold_specialized() -> None:
    workload = RecirculationRegionWorkload(_config())

    with pytest.raises(RuntimeError, match="specialization-required"):
        workload.execute_exact(
            RecirculationExecutionRequest(_TinyCausalDecoder(), (1,)), 1
        )


def test_worker_reconstruction_requires_one_signed_artifact_bundle() -> None:
    workload = RecirculationRegionWorkload(_config())
    _cold_specialize(workload, rows=2)
    spec = workload.native_workload_resources().worker_spec
    assert spec is not None

    with (
        mock.patch.object(
            region_workload,
            "build_recirculation_graph",
            side_effect=AssertionError("worker rebuilt a Recirculation graph"),
        ),
        mock.patch.object(
            region_workload._core,
            "compile_typed_recurrence_stage_artifact_json",
            side_effect=AssertionError("worker compiled a stage artifact"),
        ),
        mock.patch.object(
            region_workload._core,
            "load_typed_recurrence_stage_artifact_bundle",
            wraps=region_workload._core.load_typed_recurrence_stage_artifact_bundle,
        ) as load_bundle,
    ):
        rebuilt = region_workload.reconstruct_recirculation_region_workload(spec.config)

    assert load_bundle.call_count == 1
    assert rebuilt.specialization_key["stage_artifact_bundle"] == (
        workload.native_workload_resources().specialization_key["stage_artifact_bundle"]
    )
    missing = dict(spec.config)
    missing.pop("stage_artifact_bundle")
    with pytest.raises(RuntimeError, match="requires a signed cold stage artifact"):
        region_workload.reconstruct_recirculation_region_workload(missing)


def test_region_fails_closed_without_request_or_for_approximation() -> None:
    workload = RecirculationRegionWorkload(_config())
    with pytest.raises(RuntimeError, match="explicit finite execution request"):
        workload.resources.request_for_step(0)
    with pytest.raises(ValueError, match="enabled=True"):
        RecirculationRegionWorkload(
            RecirculationConfig(source_layer=1, destination_layer=0, enabled=False)
        )
    with pytest.raises(ValueError, match="exact execution only"):
        RecirculationRegionWorkload(
            RecirculationConfig(
                source_layer=1,
                destination_layer=0,
                jacobi=JacobiPolicy(max_rounds=2),
            )
        )
