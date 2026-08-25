#!/usr/bin/env python3
"""Eager vLLM model-region materialization and lifecycle contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from tllm import _core
from tllm.backends.protocol import BackendCapability, GraphSafety, RunnerBinding
from tllm.backends.timeline import export_main_workload_timeline
from tllm.backends.vllm import module_weight_overlay
from tllm.backends.vllm import model_region
from tllm.backends.vllm.model_region import (
    PreparedRegionRequest,
    PreparedRegionStep,
    VllmModelRegionHandle,
    install_model_region_executor,
    prepare_model_region_step,
    restore_model_region_executors,
    stage_signed_backend_region_event,
)
from tllm.backends.vllm.typed_layer_replay import VllmTypedLayerReplayHandle
from tllm.backends.vllm.v027_mrv1 import V027MRv1Driver
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
from tllm.consumers.recirculation import (
    PREFILL_BINDER_IMPLEMENTATION_ID,
    RecirculationConfig,
    RecirculationConsumer,
    VllmRecirculationRegionWorkload,
)
from tllm.consumers.recirculation import vllm_region_workload
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.graph_policy import GraphMode
from tllm.runtime import residual_runtime
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.schedule import BackendObservableEvent, BackendRegionBinderStage
from tllm.runtime.workload_execution import BackendObservableEvent


@dataclass
class _Metadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    use_cascade: bool = False
    common_prefix_len: int = 0
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0


@dataclass
class _BatchDescriptor:
    num_tokens: int


@dataclass
class _ForwardContext:
    attn_metadata: dict[str, _Metadata]
    slot_mapping: dict[str, torch.Tensor]
    batch_descriptor: _BatchDescriptor | None = None
    is_padding: torch.Tensor | None = None


class _Norm(torch.nn.Module):
    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, None]:
        return (hidden if residual is None else hidden + residual), None


class _Layer(torch.nn.Module):
    def __init__(self, index: int, calls: list[tuple[int, tuple[int, ...]]]) -> None:
        super().__init__()
        self.index = index
        self.calls = calls
        self.self_attn = SimpleNamespace(attn=SimpleNamespace())

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((self.index, tuple(int(value) for value in positions)))
        stream = hidden if residual is None else hidden + residual
        return stream + float(self.index + 1), torch.zeros_like(stream)


class _Inner(torch.nn.Module):
    def __init__(self, layer_count: int = 3) -> None:
        super().__init__()
        self.calls: list[tuple[int, tuple[int, ...]]] = []
        self.layers = torch.nn.ModuleList(
            [_Layer(index, self.calls) for index in range(layer_count)]
        )
        self.norm = _Norm()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = input_ids.float().unsqueeze(-1)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return self.norm(hidden, residual)[0]


def _runner() -> tuple[SimpleNamespace, _Inner]:
    inner = _Inner()
    runner = SimpleNamespace(
        model=SimpleNamespace(model=inner),
        enforce_eager=True,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    return runner, inner


def _context(rows: int) -> _ForwardContext:
    metadata = _Metadata(
        num_actual_tokens=rows,
        max_query_len=rows,
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32),
        max_seq_len=rows,
        seq_lens=torch.tensor([rows], dtype=torch.int32),
        block_table=torch.tensor([[1, 0]], dtype=torch.int32),
        slot_mapping=torch.arange(16, 16 + rows, dtype=torch.long),
    )
    return _ForwardContext(
        attn_metadata={"layer": metadata},
        slot_mapping={"layer": metadata.slot_mapping},
        batch_descriptor=_BatchDescriptor(rows),
    )


class _ForbiddenLegacyDispatchPlan:
    """Make accidental native-path reads of legacy scheduling observable."""

    def recurrence_targets(self) -> tuple[object, ...]:
        raise AssertionError(
            "native Recirculation binder path must not read DispatchPlan"
        )


def test_any_installed_native_execution_excludes_the_legacy_region_adapter() -> None:
    runtime = SimpleNamespace(
        workload_execution=object(),
        dispatch_plan=_ForbiddenLegacyDispatchPlan(),
    )

    assert (
        prepare_model_region_step(
            runtime=runtime,
            runner=object(),
            num_scheduled_tokens=None,
        )
        is None
    )


def _native_recirculation_runtime() -> tuple[
    TllmRuntime,
    VllmRecirculationRegionWorkload,
]:
    workload = VllmRecirculationRegionWorkload(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2)
    )
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=1 << 20
    )
    runtime.cold_specialize_workloads(
        RuntimeSpecializationSpec(
            rows=2,
            hidden_size=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
            model_layer_count=3,
            max_prefill_tokens=4,
        )
    )
    return runtime, workload


def _publish_native_recirculation_step(
    *,
    state: ResidualRuntimeState,
    runtime: TllmRuntime,
    step_id: int,
) -> None:
    timeline = export_main_workload_timeline(
        V027MRv2Driver().capabilities(),
        graph_mode="eager",
        static_headroom_bytes=1 << 20,
    )
    compilation = runtime.compile_workloads(timeline)
    assert runtime.workload_execution_bindings is not None
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(residual_runtime, "RUNTIME", state)
        residual_runtime.publish_workload_compilation(
            compilation,
            bindings=runtime.workload_execution_bindings,
        )
        residual_runtime.prepare_physical_workgraph_step(step_id)


def test_v027_drivers_publish_eager_tp1_prefill_model_region() -> None:
    for driver in (V027MRv1Driver(), V027MRv2Driver()):
        capabilities = driver.capabilities()
        capability = capabilities.model_region
        assert capability is not None
        assert (
            BackendCapability.BACKEND_REGION_BINDER.value
            in capabilities.capability_names()
        )
        assert (
            BackendCapability.RECIRCULATION_PREFILL_BINDING.value
            in capabilities.capability_names()
        )
        assert capability.phases == frozenset({"prefill"})
        assert capability.replay and capability.kv_overwrite
        assert not capability.kv_snapshot
        assert capability.graph_safety is GraphSafety.EAGER_ONLY
        assert capability.tensor_parallel_sizes == frozenset({1})
        assert capability.ordered_row_fusion is not None
        assert capability.ordered_row_fusion.internalized_edges == frozenset(
            {"kv_overwrite_read"}
        )


def test_compiled_wavefront_drives_layer_groups_and_restores() -> None:
    driver = V027MRv2Driver()
    runner, inner = _runner()
    binding = RunnerBinding(
        driver=driver,
        runner=runner,
        capabilities=driver.capabilities(),
    )
    runtime = SimpleNamespace(model_region_binding=None)
    original_model_forward = inner.forward
    original_first_forward = inner.layers[0].forward
    handle = install_model_region_executor(binding=binding, runtime=runtime)
    consumer = RecirculationConsumer(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2)
    )
    compiled = DispatchPlan.build([consumer]).compile_recurrences(
        token_count=2,
        layer_count=3,
        backend_capabilities=driver.capabilities(),
        graph_mode=GraphMode.EAGER,
    )[0][1]
    assert compiled.schedule.kind == "wavefront"
    assert ("t0/recurrent/l1", "t1/normal/l1") in compiled.schedule.groups
    handle.prepare(
        PreparedRegionStep(
            total_rows=2,
            requests=(
                PreparedRegionRequest(
                    request_id="req",
                    request_ordinal=0,
                    packed_rows=(0, 1),
                    token_positions=(0, 1),
                    compiled=compiled,
                    adapter=consumer.recurrence_runtime_adapter(layer_count=3),
                ),
            ),
        )
    )
    from vllm.forward_context import override_forward_context

    with override_forward_context(_context(2)):
        output = inner(torch.tensor([2, 3]), torch.tensor([0, 1]))

    assert output.shape == (2, 1)
    assert handle.stats.request_count == 1
    assert handle.stats.replay_count == 1
    assert handle.stats.fused_group_count == 2
    assert (1, (0, 1)) in inner.calls
    assert (2, (0, 1)) in inner.calls

    assert restore_model_region_executors(binding) == 1
    assert restore_model_region_executors(binding) == 0
    assert inner.forward == original_model_forward
    assert inner.layers[0].forward == original_first_forward


def test_prepare_fails_closed_with_prefix_cache_before_forward() -> None:
    driver = V027MRv1Driver()
    runner, _inner = _runner()
    runner.cache_config.enable_prefix_caching = True
    module_weight_overlay.ensure_runner_binding(driver=driver, runner=runner)
    consumer = RecirculationConsumer(
        RecirculationConfig(source_layer=1, destination_layer=0)
    )
    runtime = SimpleNamespace(
        dispatch_plan=DispatchPlan.build([consumer]),
        model_region_binding=None,
    )

    with pytest.raises(RuntimeError, match="prefix caching"):
        prepare_model_region_step(
            runtime=runtime,
            runner=runner,
            num_scheduled_tokens=[1],
            input_batch=SimpleNamespace(
                req_ids=["req"],
                num_reqs=1,
                prefill_len_np=[1],
                num_computed_tokens_np=[0],
            ),
        )


def test_prepare_fails_closed_with_engine_multiprocessing(monkeypatch) -> None:
    from vllm import envs as vllm_envs

    monkeypatch.setattr(vllm_envs, "VLLM_ENABLE_V1_MULTIPROCESSING", True)
    driver = V027MRv2Driver()
    runner, _inner = _runner()
    binding = module_weight_overlay.ensure_runner_binding(
        driver=driver, runner=runner
    )
    consumer = RecirculationConsumer(
        RecirculationConfig(source_layer=1, destination_layer=0)
    )
    runtime = SimpleNamespace(
        dispatch_plan=DispatchPlan.build([consumer]),
        model_region_binding=None,
    )
    with pytest.raises(RuntimeError, match="MULTIPROCESSING=0"):
        prepare_model_region_step(
            runtime=runtime,
            runner=runner,
            num_scheduled_tokens=[1],
            input_batch=SimpleNamespace(
                req_ids=["req"],
                num_reqs=1,
                prefill_len_np=[1],
                num_computed_tokens_np=[0],
            ),
        )
    assert binding.model_region_handles == {}


def test_unknown_model_structure_rolls_back_install() -> None:
    driver = V027MRv2Driver()
    runner = SimpleNamespace(model=SimpleNamespace())
    binding = RunnerBinding(
        driver=driver,
        runner=runner,
        capabilities=driver.capabilities(),
    )
    with pytest.raises(RuntimeError, match="Qwen-like"):
        install_model_region_executor(
            binding=binding,
            runtime=SimpleNamespace(model_region_binding=None),
        )
    assert binding.model_region_handles == {}


def test_native_prefill_binder_stages_without_reading_legacy_dispatch_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The signed Region, not a RecirculationConsumer, owns native binding."""

    monkeypatch.setattr(
        __import__("vllm.envs", fromlist=["VLLM_ENABLE_V1_MULTIPROCESSING"]),
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        False,
    )
    runtime, workload = _native_recirculation_runtime()
    state = ResidualRuntimeState(event_step_id=7)
    _publish_native_recirculation_step(
        state=state,
        runtime=runtime,
        step_id=7,
    )
    state.dispatch_plan = _ForbiddenLegacyDispatchPlan()  # type: ignore[assignment]
    runner, _inner = _runner()
    binding = module_weight_overlay.ensure_runner_binding(
        driver=V027MRv2Driver(), runner=runner
    )

    prepared = prepare_model_region_step(
        runtime=state,
        runner=runner,
        num_scheduled_tokens=[2],
        input_batch=SimpleNamespace(
            req_ids=["req"],
            num_reqs=1,
            prefill_len_np=[2],
            num_computed_tokens_np=[0],
        ),
    )

    # `prepare_model_region_step` may install the restorable handle for
    # validation, but the Region callback is the first place it may prepare it.
    assert prepared is None
    assert all(
        handle.prepared is None
        for handle in binding.model_region_handles.values()
        if isinstance(handle, VllmModelRegionHandle)
    )
    assert stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_POST,
        runtime=state,
        runner=runner,
        num_scheduled_tokens=[2],
        input_batch=SimpleNamespace(
            req_ids=["req"],
            num_reqs=1,
            prefill_len_np=[2],
            num_computed_tokens_np=[0],
        ),
    )
    staged = workload.binder.plan_for_step(7)
    assert len(staged.requests) == 1
    assert isinstance(staged.requests[0].bound_stage, _core.TypedBoundRecurrenceStage)


def test_native_typed_stage_does_not_coerce_request_identity() -> None:
    """The native boundary rejects an opaque request ID without ``str()``."""

    class _DisplayOnlyRequestId:
        def __str__(self) -> str:
            return "request-that-must-not-be-coerced"

    runtime, _workload = _native_recirculation_runtime()
    state = ResidualRuntimeState(event_step_id=9)
    _publish_native_recirculation_step(state=state, runtime=runtime, step_id=9)
    runner, _inner = _runner()
    binding = module_weight_overlay.ensure_runner_binding(
        driver=V027MRv2Driver(), runner=runner
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            __import__("vllm.envs", fromlist=["VLLM_ENABLE_V1_MULTIPROCESSING"]),
            "VLLM_ENABLE_V1_MULTIPROCESSING",
            False,
        )
        with pytest.raises(RuntimeError, match="string vLLM request identifiers"):
            stage_signed_backend_region_event(
                event=BackendObservableEvent.PREPARE_INPUTS_POST,
                runtime=state,
                runner=runner,
                num_scheduled_tokens=[1],
                input_batch=SimpleNamespace(
                    req_ids=[_DisplayOnlyRequestId()],
                    num_reqs=1,
                    prefill_len_np=[1],
                    num_computed_tokens_np=[0],
                ),
            )
    assert binding.model_region_handles == {}


def test_native_prefill_handle_is_prepared_only_by_signed_prepare_post_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        __import__("vllm.envs", fromlist=["VLLM_ENABLE_V1_MULTIPROCESSING"]),
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        False,
    )
    runtime, workload = _native_recirculation_runtime()
    state = ResidualRuntimeState(event_step_id=11)
    _publish_native_recirculation_step(
        state=state,
        runtime=runtime,
        step_id=11,
    )
    state.dispatch_plan = _ForbiddenLegacyDispatchPlan()  # type: ignore[assignment]
    runner, _inner = _runner()
    binding = module_weight_overlay.ensure_runner_binding(
        driver=V027MRv2Driver(), runner=runner
    )

    prepare_model_region_step(
        runtime=state,
        runner=runner,
        num_scheduled_tokens=[2],
        input_batch=SimpleNamespace(
            req_ids=["req"],
            num_reqs=1,
            prefill_len_np=[2],
            num_computed_tokens_np=[0],
        ),
    )
    assert all(
        handle.prepared is None
        for handle in binding.model_region_handles.values()
        if isinstance(handle, VllmModelRegionHandle)
    )
    assert stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_POST,
        runtime=state,
        runner=runner,
        num_scheduled_tokens=[2],
        input_batch=SimpleNamespace(
            req_ids=["req"],
            num_reqs=1,
            prefill_len_np=[2],
            num_computed_tokens_np=[0],
        ),
    )

    with monkeypatch.context() as runtime_patch:
        runtime_patch.setattr(residual_runtime, "RUNTIME", state)
        assert residual_runtime.advance_workload_anchor("prepare.post")

    handle = next(
        handle
        for handle in binding.model_region_handles.values()
        if isinstance(handle, VllmTypedLayerReplayHandle)
    )
    assert handle.prepared is not None
    assert handle.prepared.total_rows == 2
    with pytest.raises(RuntimeError, match="already completed"):
        workload.binder.invoke(11)


def test_native_prefill_region_uses_rust_typed_plan_without_legacy_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native path may declare a graph but cannot enter legacy replay."""

    def _forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("native typed Recirculation must not load legacy replay")

    monkeypatch.setattr(
        model_region,
        "_legacy",
        _forbidden,
    )
    monkeypatch.setattr(
        __import__("vllm.envs", fromlist=["VLLM_ENABLE_V1_MULTIPROCESSING"]),
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        False,
    )
    runtime, _workload = _native_recirculation_runtime()
    state = ResidualRuntimeState(event_step_id=17)
    _publish_native_recirculation_step(
        state=state,
        runtime=runtime,
        step_id=17,
    )
    state.dispatch_plan = _ForbiddenLegacyDispatchPlan()  # type: ignore[assignment]
    runner, inner = _runner()
    binding = module_weight_overlay.ensure_runner_binding(
        driver=V027MRv2Driver(), runner=runner
    )

    prepare_model_region_step(
        runtime=state,
        runner=runner,
        num_scheduled_tokens=[2],
        input_batch=SimpleNamespace(
            req_ids=["req"],
            num_reqs=1,
            prefill_len_np=[2],
            num_computed_tokens_np=[0],
        ),
    )
    assert stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_POST,
        runtime=state,
        runner=runner,
        num_scheduled_tokens=[2],
        input_batch=SimpleNamespace(
            req_ids=["req"],
            num_reqs=1,
            prefill_len_np=[2],
            num_computed_tokens_np=[0],
        ),
    )
    with monkeypatch.context() as runtime_patch:
        runtime_patch.setattr(residual_runtime, "RUNTIME", state)
        assert residual_runtime.advance_workload_anchor("prepare.post")

    handle = next(
        handle
        for handle in binding.model_region_handles.values()
        if isinstance(handle, VllmTypedLayerReplayHandle)
    )
    with __import__("vllm.forward_context", fromlist=["override_forward_context"]).override_forward_context(
        _context(2)
    ):
        output = inner(torch.tensor([2, 3]), torch.tensor([0, 1]))
    assert output.shape == (2, 1)
    assert handle.prepared is None
    assert handle.stats.prepare_count == 1
    assert handle.stats.replay_count == 1
    assert handle.stats.replay_row_count == 2
    assert handle.stats.request_replays == ("req",)


def test_native_prefill_does_not_reselect_artifact_policy_from_worker_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hook binds the signed serial artifact without choosing a policy."""

    monkeypatch.setattr(
        __import__("vllm.envs", fromlist=["VLLM_ENABLE_V1_MULTIPROCESSING"]),
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        False,
    )
    runtime, _workload = _native_recirculation_runtime()
    state = ResidualRuntimeState(event_step_id=19)
    _publish_native_recirculation_step(
        state=state,
        runtime=runtime,
        step_id=19,
    )
    runner, inner = _runner()
    binding = module_weight_overlay.ensure_runner_binding(
        driver=V027MRv2Driver(), runner=runner
    )
    signed_region = binding.capabilities.model_region
    assert signed_region is not None
    binding.capabilities = replace(
        binding.capabilities,
        model_region=replace(signed_region, ordered_row_fusion=None),
    )

    assert stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_POST,
        runtime=state,
        runner=runner,
        num_scheduled_tokens=[2],
        input_batch=SimpleNamespace(
            req_ids=["req"],
            num_reqs=1,
            prefill_len_np=[2],
            num_computed_tokens_np=[0],
        ),
    )
    assert inner.calls == []


def test_native_prefill_rejects_mixed_decode_before_model_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        __import__("vllm.envs", fromlist=["VLLM_ENABLE_V1_MULTIPROCESSING"]),
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        False,
    )
    runtime, _workload = _native_recirculation_runtime()
    state = ResidualRuntimeState(event_step_id=13)
    _publish_native_recirculation_step(
        state=state,
        runtime=runtime,
        step_id=13,
    )
    state.dispatch_plan = _ForbiddenLegacyDispatchPlan()  # type: ignore[assignment]
    runner, inner = _runner()
    module_weight_overlay.ensure_runner_binding(driver=V027MRv2Driver(), runner=runner)

    with pytest.raises(ValueError, match="mixed backend phases"):
        stage_signed_backend_region_event(
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            runtime=state,
            runner=runner,
            num_scheduled_tokens=[2],
            input_batch=SimpleNamespace(
                req_ids=["req"],
                num_reqs=1,
                # One row is prefill (prompt - computed) and one is decode.
                prefill_len_np=[1],
                num_computed_tokens_np=[0],
            ),
        )
    assert inner.calls == []


def test_native_prefill_shape_miss_never_compiles_in_the_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        __import__("vllm.envs", fromlist=["VLLM_ENABLE_V1_MULTIPROCESSING"]),
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        False,
    )
    runtime, _workload = _native_recirculation_runtime()
    state = ResidualRuntimeState(event_step_id=23)
    _publish_native_recirculation_step(state=state, runtime=runtime, step_id=23)
    runner, inner = _runner()
    binding = module_weight_overlay.ensure_runner_binding(
        driver=V027MRv2Driver(), runner=runner
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("vLLM hook must never cold compile a stage artifact")

    monkeypatch.setattr(_core, "compile_typed_recurrence_stage_artifact", forbidden)
    monkeypatch.setattr(_core, "compile_typed_recurrence_stage_artifact_json", forbidden)
    monkeypatch.setattr(
        _core, "compile_typed_recurrence_stage_artifact_bundle_json", forbidden
    )
    monkeypatch.setattr(vllm_region_workload, "build_recirculation_graph", forbidden)

    with pytest.raises(RuntimeError, match="specialization-required"):
        stage_signed_backend_region_event(
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            runtime=state,
            runner=runner,
            num_scheduled_tokens=[5],
            input_batch=SimpleNamespace(
                req_ids=["req"],
                num_reqs=1,
                prefill_len_np=[5],
                num_computed_tokens_np=[0],
            ),
        )
    assert inner.calls == []
    assert binding.model_region_handles == {}


def test_native_prefill_rejects_a_signed_binder_without_a_local_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        __import__("vllm.envs", fromlist=["VLLM_ENABLE_V1_MULTIPROCESSING"]),
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        False,
    )
    runner, _inner = _runner()
    module_weight_overlay.ensure_runner_binding(driver=V027MRv2Driver(), runner=runner)
    runtime = SimpleNamespace(
        event_step_id=17,
        dispatch_plan=_ForbiddenLegacyDispatchPlan(),
        model_region_binding=None,
        workload_execution=SimpleNamespace(
            backend_region_binders={"wrong.backend:bind_prefill@1": object()},
            backend_region_binder_stages=lambda event: (
                (
                    BackendRegionBinderStage(
                        backend_event=BackendObservableEvent.PREPARE_INPUTS_POST,
                        region_id="region-0000-test",
                        implementation_id="test.backend:missing-local-slot@1",
                    ),
                )
                if event is BackendObservableEvent.PREPARE_INPUTS_POST
                else ()
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="no local binder slot"):
        stage_signed_backend_region_event(
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            runtime=runtime,
            runner=runner,
            num_scheduled_tokens=[1],
            input_batch=SimpleNamespace(
                req_ids=["req"],
                num_reqs=1,
                prefill_len_np=[1],
                num_computed_tokens_np=[0],
            ),
        )


def test_mrv2_cold_compile_precedes_native_step_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first MRv2 prefill must prepare the execution cold-compiled in it."""

    from tllm.backends.vllm import v027_mrv2
    from tllm.runtime.vllm_patch import common_hooks
    from tllm.runtime.workload_execution import BackendObservableEvent

    class _Execution:
        def __init__(self) -> None:
            self.prepared_steps: list[int] = []
            self.prepare_post_count = 0

        def prepare_step(self, step_id: int) -> None:
            self.prepared_steps.append(step_id)

        def advance_backend_event(self, event: BackendObservableEvent) -> bool:
            if event is BackendObservableEvent.PREPARE_INPUTS_POST:
                if not self.prepared_steps:
                    return False
                self.prepare_post_count += 1
            return True

    execution = _Execution()
    runtime = SimpleNamespace(
        event_step_id=0,
        workload_execution=None,
        dispatch_plan=None,
        deferred_work_queue=None,
        launch_consumer_from_hooks=False,
    )
    runner = SimpleNamespace(model=SimpleNamespace())
    calls: list[str] = []

    def cold_compile(*, core: object, runner: object) -> None:
        assert core is residual_runtime
        assert runner is not None
        calls.append("cold_compile")
        if runtime.workload_execution is None:
            runtime.workload_execution = execution

    def prepare_physical(step_id: int) -> None:
        calls.append("prepare_physical")
        if runtime.workload_execution is not None:
            runtime.workload_execution.prepare_step(step_id)

    def original_execute(_runner: object, *_args: object, **_kwargs: object) -> object:
        # This is the nested prepare-inputs boundary reached from MRv2's
        # original execute_model implementation.
        v027_mrv2.cold_compile_loaded_workloads(
            core=residual_runtime,
            runner=runner,
        )
        common_hooks.dispatch_runtime_event(
            runtime=runtime,
            runner=runner,
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            phase="prefill",
            layer_path=None,
            capture_enabled=False,
        )
        return object()

    with monkeypatch.context() as patch:
        patch.setattr(residual_runtime, "RUNTIME", runtime)
        patch.setattr(v027_mrv2, "cold_compile_loaded_workloads", cold_compile)
        patch.setattr(
            residual_runtime,
            "prepare_physical_workgraph_step",
            prepare_physical,
        )
        patch.setattr(v027_mrv2.active_targets, "runtime_has_active_targets", lambda _: True)
        patch.setattr(v027_mrv2._hooks, "setup_runtime_hooks_if_active", lambda **_: None)
        v027_mrv2._wrapped_execute_model(
            core=residual_runtime,
            runner=runner,
            args=(),
            kwargs={},
            orig=original_execute,
        )

    assert calls.index("cold_compile") < calls.index("prepare_physical")
    assert execution.prepared_steps == [1]
    assert execution.prepare_post_count == 1


def test_model_forward_failure_clears_prepared_region_before_decode_noop() -> None:
    """A failed prefill cannot replay its plan during a later decode step."""

    driver = V027MRv2Driver()
    runner, inner = _runner()
    binding = RunnerBinding(
        driver=driver,
        runner=runner,
        capabilities=driver.capabilities(),
    )
    handle = install_model_region_executor(
        binding=binding,
        runtime=SimpleNamespace(model_region_binding=None),
    )
    consumer = RecirculationConsumer(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2)
    )
    compiled = DispatchPlan.build([consumer]).compile_recurrences(
        token_count=1,
        layer_count=3,
        backend_capabilities=driver.capabilities(),
        graph_mode=GraphMode.EAGER,
    )[0][1]
    handle.prepare(
        PreparedRegionStep(
            total_rows=1,
            requests=(
                PreparedRegionRequest(
                    request_id="prefill",
                    request_ordinal=0,
                    packed_rows=(0,),
                    token_positions=(0,),
                    compiled=compiled,
                    adapter=consumer.recurrence_runtime_adapter(layer_count=3),
                ),
            ),
        )
    )
    original_model_forward = handle.original_model_forward

    def failing_model_forward(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Reach the patched first layer so the stale seed path is exercised.
        handle.layers[0](positions, input_ids.float().unsqueeze(-1), None)
        raise RuntimeError("injected original-model failure")

    try:
        handle.original_model_forward = failing_model_forward
        with pytest.raises(RuntimeError, match="injected original-model failure"):
            inner(torch.tensor([2]), torch.tensor([0]))

        assert handle.prepared is None
        assert handle.seed_hidden is None
        assert handle.seed_residual is None
        assert handle.seed_positions is None

        handle.original_model_forward = original_model_forward
        from vllm.forward_context import override_forward_context

        with override_forward_context(_context(1)):
            inner(torch.tensor([3]), torch.tensor([0]))

        assert handle.stats.replay_count == 0
        assert handle.stats.request_replays == {}
    finally:
        handle.original_model_forward = original_model_forward
        restore_model_region_executors(binding)
