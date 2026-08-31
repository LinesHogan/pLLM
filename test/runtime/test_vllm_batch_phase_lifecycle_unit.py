"""Rust-derived batch phase must govern every vLLM hook in one step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tllm.backends.vllm import v027_mrv1, v027_mrv2
from tllm.backends.vllm import scheduler_batch_phase
from tllm import _core
from tllm.contracts.packed_rows import PackedPhaseRowBatch, PackedRowPlanInputs
from tllm.runtime.packed_phase_row_staging import PackedPhaseRowStaging
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.contracts.subscription import ConsumerSubscription
from tllm.runtime.backend_region_binders import (
    BackendRegionBinderSlot,
    BackendRegionStageContext,
    BackendRegionStagerRegistry,
)
from tllm.runtime.dispatch_plan import DispatchPlan, DispatchTarget
from tllm.runtime.vllm_patch import common_hooks
from tllm.runtime.workload_execution import BackendObservableEvent
from tllm.schedule import BackendRegionBinderStage


@dataclass
class _PreparedPackedRowsExecution:
    order: list[str]
    plan_epoch: int = 17
    requires_packed_rows: bool = True
    requires_packed_hidden: bool = True
    parameter_overlays: dict[str, int] = field(default_factory=dict)
    step_ids: list[int] = field(default_factory=list)
    columns: list[PackedRowPlanInputs] = field(default_factory=list)

    def stage_packed_row_plan(
        self,
        step_id: int,
        columns: PackedRowPlanInputs,
        prompt_indices: Mapping[str, int],
        sample_indices: Mapping[str, int],
    ) -> PackedPhaseRowBatch:
        self.order.append("stage:packed_rows")
        self.step_ids.append(step_id)
        self.columns.append(columns)
        return PackedPhaseRowStaging().stage(
            self.plan_epoch,
            step_id,
            columns,
            prompt_indices=prompt_indices,
            sample_indices=sample_indices,
        )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        event_step_id=0,
        decode_request_ids=(),
        launch_consumer_from_hooks=False,
        path_hotspot_enabled=False,
        workload_execution=None,
        reqid_to_promptidx={},
        reqid_to_sampleidx={},
    )


def _prefill_batch_mrv1() -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        req_id_to_index={"request": 0},
        num_prompt_tokens=[4],
        num_computed_tokens_cpu=[0],
    )


def _prefill_batch_mrv2() -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        prefill_len_np=[4],
        num_computed_tokens_np=[0],
        num_scheduled_tokens=[2],
        logits_indices=None,
    )


@pytest.mark.parametrize("driver", ("mrv1", "mrv2"))
def test_prefill_phase_is_rust_derived_for_entire_hook_lifecycle(
    driver: str,
) -> None:
    runtime = _runtime()
    runtime.reqid_to_promptidx = {"request": 3}
    runtime.reqid_to_sampleidx = {"request": 7}
    runner = SimpleNamespace(model=SimpleNamespace())
    phases: list[tuple[BackendObservableEvent, str | None]] = []
    order: list[str] = []
    packed_execution = _PreparedPackedRowsExecution(order)
    runtime.workload_execution = packed_execution
    specialization = RuntimeSpecializationSpec(
        rows=1,
        hidden_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_prefill_tokens=4,
    )

    def dispatch(**kwargs: object) -> int:
        order.append(f"dispatch:{kwargs['event'].value}")
        phases.append((kwargs["event"], kwargs["phase"]))
        return 0

    if driver == "mrv1":
        runner.input_batch = _prefill_batch_mrv1()
        view = SimpleNamespace(
            logits_indices=None,
            spec_decode_metadata=None,
            num_scheduled_tokens_np=[2],
        )
        with (
            mock.patch.object(v027_mrv1.active_targets, "runtime_has_active_targets", return_value=True),
            mock.patch.object(v027_mrv1._common_hooks, "dispatch_runtime_event", side_effect=dispatch),
            mock.patch.object(v027_mrv1._hooks, "setup_runtime_hooks_if_active"),
            mock.patch.object(v027_mrv1._hooks, "run_prepare_inputs_post_producers"),
            mock.patch.object(v027_mrv1._model_region, "prepare_model_region_step"),
            mock.patch.object(
                v027_mrv1._model_region,
                "stage_signed_backend_region_event",
                return_value=False,
            ),
            mock.patch.object(v027_mrv1._sampler_patch, "maybe_prepare_sampler_decode_step"),
            mock.patch.object(v027_mrv1._ADAPTER, "unpack_prepare_inputs_output", return_value=view),
            mock.patch("tllm.runtime.request_lifecycle.observe_request_aliases"),
            mock.patch.object(
                v027_mrv1,
                "runner_model_module",
                return_value=torch.nn.Identity(),
            ),
            mock.patch.object(
                v027_mrv1,
                "runner_runtime_specialization",
                return_value=specialization,
            ),
            mock.patch.object(
                v027_mrv1._packed_hidden_capture,
                "begin_runner_packed_hidden_capture",
                side_effect=lambda **_kwargs: order.append(
                    "begin:packed_hidden"
                ),
            ),
        ):
            core = SimpleNamespace(RUNTIME=runtime)
            v027_mrv1._wrapped_prepare_inputs(
                core=core,
                runner=runner,
                scheduler_output=object(),
                num_scheduled_tokens=[2],
                orig=mock.Mock(return_value=(None, None)),
            )
            core._ORIG_EXECUTE_MODEL = mock.Mock(return_value=object())
            core._runner_uses_compilation_or_cudagraph = lambda _runner: False
            with (
                mock.patch.object(v027_mrv1._hooks, "maybe_launch_post_logits_decode_work"),
                mock.patch.object(v027_mrv1._hooks, "run_execute_model_post_producers"),
                mock.patch.object(v027_mrv1._hooks._hidden_bridge, "dispatch_deferred_layer_batches"),
            ):
                v027_mrv1._hooks.wrapped_execute_model(
                    core=core,
                    runner=runner,
                    args=(),
                    kwargs={},
                    split_execute_and_sample=True,
                    phase=None,
                    execute_pre_already_dispatched=True,
                    phase_after_inputs=lambda: scheduler_batch_phase.session_hook_phase(
                        v027_mrv1.get_step_session(runner)
                    ),
                )
            v027_mrv1._wrapped_sample_tokens(
                core=core,
                runner=runner,
                grammar_output=None,
                orig=mock.Mock(return_value=object()),
            )
    else:
        batch = _prefill_batch_mrv2()

        def execute_orig(_runner: object) -> object:
            return v027_mrv2._wrapped_prepare_inputs(
                core=core,
                runner=runner,
                args=(),
                kwargs={},
                orig=mock.Mock(return_value=batch),
            )

        with (
            mock.patch.object(v027_mrv2.active_targets, "runtime_has_active_targets", return_value=True),
            mock.patch.object(v027_mrv2._common_hooks, "dispatch_runtime_event", side_effect=dispatch),
            mock.patch.object(v027_mrv2._hooks, "setup_runtime_hooks_if_active"),
            mock.patch.object(v027_mrv2._hooks, "run_prepare_inputs_post_producers"),
            mock.patch.object(v027_mrv2._hooks, "run_execute_model_post_producers"),
            mock.patch.object(v027_mrv2._hooks, "maybe_launch_post_logits_decode_work"),
            mock.patch.object(v027_mrv2._model_region, "prepare_model_region_step"),
            mock.patch.object(
                v027_mrv2._model_region,
                "stage_signed_backend_region_event",
                return_value=False,
            ),
            mock.patch.object(
                v027_mrv2._model_region,
                "abort_model_region_step",
                return_value=False,
            ),
            mock.patch.object(v027_mrv2._sampler_patch, "maybe_prepare_sampler_decode_step"),
            mock.patch.object(v027_mrv2, "_decode_localization_from_mrv2", return_value=()),
            mock.patch("tllm.runtime.request_lifecycle.observe_request_aliases"),
            mock.patch.object(
                v027_mrv2,
                "runner_model_module",
                return_value=torch.nn.Identity(),
            ),
            mock.patch.object(
                v027_mrv2,
                "runner_runtime_specialization",
                return_value=specialization,
            ),
            mock.patch.object(
                v027_mrv2._packed_hidden_capture,
                "begin_runner_packed_hidden_capture",
                side_effect=lambda **_kwargs: order.append(
                    "begin:packed_hidden"
                ),
            ),
            mock.patch.object(
                v027_mrv2._packed_hidden_capture,
                "publish_runner_packed_hidden_capture",
                side_effect=lambda **_kwargs: order.append(
                    "publish:packed_hidden"
                ),
            ),
        ):
            core = SimpleNamespace(RUNTIME=runtime)
            v027_mrv2._wrapped_execute_model(
                core=core,
                runner=runner,
                args=(),
                kwargs={},
                orig=execute_orig,
            )
            v027_mrv2._wrapped_sample_tokens(
                core=core,
                runner=runner,
                grammar_output=None,
                orig=mock.Mock(return_value=object()),
            )

    assert [event for event, _phase in phases] == [
        BackendObservableEvent.PREPARE_INPUTS_PRE,
        BackendObservableEvent.PREPARE_INPUTS_POST,
        BackendObservableEvent.EXECUTE_MODEL_PRE,
        BackendObservableEvent.EXECUTE_MODEL_POST,
        BackendObservableEvent.SAMPLE_TOKENS_PRE,
        BackendObservableEvent.SAMPLE_TOKENS_POST,
    ]
    assert [phase for _event, phase in phases] == ["prefill"] * len(phases)
    assert packed_execution.step_ids == [1]
    assert packed_execution.columns == [
        PackedRowPlanInputs(
            request_ids=("request",),
            scheduled_tokens=(2,),
            prompt_tokens=(4,),
            computed_tokens=(0,),
        )
    ]
    assert order[:3] == [
        "stage:packed_rows",
        "begin:packed_hidden",
        "dispatch:prepare_inputs.pre",
    ]
    if driver == "mrv2":
        assert order.index("publish:packed_hidden") < order.index(
            "dispatch:execute_model.post"
        )


def test_mixed_scheduler_batch_is_rust_derived_for_the_full_mrv2_lifecycle() -> None:
    """One decode request plus one prefill request remains Mixed end-to-end."""

    runtime = _runtime()
    runner = SimpleNamespace(model=SimpleNamespace())
    batch = SimpleNamespace(
        req_ids=["prefill", "decode"],
        num_reqs=2,
        prefill_len_np=[2, 1],
        num_computed_tokens_np=[0, 1],
        num_scheduled_tokens=[1, 1],
    )
    events: list[tuple[BackendObservableEvent, str | None]] = []

    def execute_orig(_runner: object) -> object:
        return v027_mrv2._wrapped_prepare_inputs(
            core=core,
            runner=runner,
            args=(),
            kwargs={},
            orig=mock.Mock(return_value=batch),
        )

    with (
        mock.patch.object(v027_mrv2.active_targets, "runtime_has_active_targets", return_value=True),
        mock.patch.object(
            v027_mrv2._common_hooks,
            "dispatch_runtime_event",
            side_effect=lambda **kwargs: events.append((kwargs["event"], kwargs["phase"])),
        ),
        mock.patch.object(v027_mrv2._hooks, "setup_runtime_hooks_if_active"),
        mock.patch.object(v027_mrv2._hooks, "run_prepare_inputs_post_producers"),
        mock.patch.object(v027_mrv2._hooks, "run_execute_model_post_producers"),
        mock.patch.object(v027_mrv2._hooks, "maybe_launch_post_logits_decode_work"),
        mock.patch.object(v027_mrv2._model_region, "prepare_model_region_step"),
        mock.patch.object(v027_mrv2._sampler_patch, "maybe_prepare_sampler_decode_step"),
        mock.patch.object(v027_mrv2, "_decode_localization_from_mrv2", return_value=()),
    ):
        core = SimpleNamespace(RUNTIME=runtime)
        v027_mrv2._wrapped_execute_model(
            core=core,
            runner=runner,
            args=(),
            kwargs={},
            orig=execute_orig,
        )
        v027_mrv2._wrapped_sample_tokens(
            core=core,
            runner=runner,
            grammar_output=None,
            orig=mock.Mock(return_value=object()),
        )

    assert v027_mrv2.get_step_session(runner).batch_phase == _core.SchedulerBatchPhase.Mixed
    assert [event for event, _phase in events] == [
        BackendObservableEvent.PREPARE_INPUTS_PRE,
        BackendObservableEvent.PREPARE_INPUTS_POST,
        BackendObservableEvent.EXECUTE_MODEL_PRE,
        BackendObservableEvent.EXECUTE_MODEL_POST,
        BackendObservableEvent.SAMPLE_TOKENS_PRE,
        BackendObservableEvent.SAMPLE_TOKENS_POST,
    ]
    assert [phase for _event, phase in events] == ["mixed"] * len(events)


def test_hook_phase_maps_only_rust_pure_phases_and_preserves_empty() -> None:
    assert scheduler_batch_phase.hook_phase(_core.SchedulerBatchPhase.Prefill) == "prefill"
    assert scheduler_batch_phase.hook_phase(_core.SchedulerBatchPhase.Decode) == "decode"
    assert scheduler_batch_phase.hook_phase(_core.SchedulerBatchPhase.Empty) == "empty"
    assert scheduler_batch_phase.hook_phase(_core.SchedulerBatchPhase.Mixed) == "mixed"


def test_mixed_phase_advances_native_regions_without_a_legacy_dispatch_plan() -> None:
    execution = mock.Mock()
    runtime = SimpleNamespace(
        dispatch_plan=None,
        workload_execution=execution,
        deferred_work_queue=None,
        event_step_id=1,
    )

    count = common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=SimpleNamespace(),
        event=BackendObservableEvent.EXECUTE_MODEL_POST,
        phase="mixed",
    )

    assert count == 0
    execution.advance_backend_event.assert_called_once_with(
        BackendObservableEvent.EXECUTE_MODEL_POST
    )


def test_mixed_phase_rejects_legacy_event_subscribers_before_native_advance() -> None:
    subscription = ConsumerSubscription(
        consumer_id="legacy",
        event_name=BackendObservableEvent.EXECUTE_MODEL_POST.value,
        phase_filter="decode",
    )
    plan = DispatchPlan(
        {
            BackendObservableEvent.EXECUTE_MODEL_POST.value: [
                DispatchTarget(consumer=object(), subscription=subscription)
            ]
        },
        [],
    )
    execution = mock.Mock()
    runtime = SimpleNamespace(
        dispatch_plan=plan,
        workload_execution=execution,
        deferred_work_queue=None,
        event_step_id=1,
    )

    with pytest.raises(RuntimeError, match="mixed scheduler batch"):
        common_hooks.dispatch_runtime_event(
            runtime=runtime,
            runner=SimpleNamespace(),
            event=BackendObservableEvent.EXECUTE_MODEL_POST,
            phase="mixed",
        )

    execution.advance_backend_event.assert_not_called()


def test_mrv1_execute_model_creates_rust_phase_inside_nested_prepare_inputs() -> None:
    """MRv1 calls ``_prepare_inputs`` from inside ``execute_model``.

    The execute wrapper must consequently not consume a session before the
    nested input boundary has adapted scheduler columns and Rust has
    classified the phase.
    """

    runtime = _runtime()
    runtime.reqid_to_promptidx = {"request": 3}
    runtime.reqid_to_sampleidx = {"request": 7}
    order: list[str] = []
    runtime.workload_execution = _PreparedPackedRowsExecution(order)
    runner = SimpleNamespace(
        model=SimpleNamespace(),
        input_batch=_prefill_batch_mrv1(),
    )
    view = SimpleNamespace(
        logits_indices=None,
        spec_decode_metadata=None,
        num_scheduled_tokens_np=[2],
    )
    events: list[tuple[BackendObservableEvent, str | None]] = []
    specialization = RuntimeSpecializationSpec(
        rows=1,
        hidden_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_prefill_tokens=4,
    )

    class NestedPrepareRunner:
        def __init__(self) -> None:
            self.model = runner.model
            self.input_batch = runner.input_batch
            self.prepare_inputs = True

        def load_model(self) -> None:
            return None

        def _prepare_inputs(self, scheduler_output: object, num_scheduled_tokens: object) -> tuple[None, None]:
            return (None, None)

        def execute_model(self) -> object:
            # This is the installed vLLM MRv1 lifecycle: the outer execute
            # wrapper starts before the input boundary exists.
            if self.prepare_inputs:
                self._prepare_inputs(object(), [2])
            return object()

        def sample_tokens(self, grammar_output: object = None) -> object:
            return object()

    actual_runner = NestedPrepareRunner()
    core = SimpleNamespace(
        RUNTIME=runtime,
        _runner_uses_compilation_or_cudagraph=lambda _runner: False,
    )
    info = SimpleNamespace(
        mrv1_runner_cls=NestedPrepareRunner,
        mrv1_has_sample_tokens=True,
        mrv1_prepare_inputs_arity=2,
        sampler_returns_tuple=True,
    )

    with (
        mock.patch.object(v027_mrv1.active_targets, "runtime_has_active_targets", return_value=True),
        mock.patch.object(
            v027_mrv1._common_hooks,
            "dispatch_runtime_event",
            side_effect=lambda **kwargs: (
                order.append(f"dispatch:{kwargs['event'].value}"),
                events.append((kwargs["event"], kwargs["phase"])),
            ),
        ),
        mock.patch.object(v027_mrv1._hooks, "setup_runtime_hooks_if_active"),
        mock.patch.object(v027_mrv1._hooks, "run_prepare_inputs_post_producers"),
        mock.patch.object(v027_mrv1._hooks, "run_execute_model_post_producers"),
        mock.patch.object(v027_mrv1._hooks, "maybe_launch_post_logits_decode_work"),
        mock.patch.object(v027_mrv1._hooks._hidden_bridge, "dispatch_deferred_layer_batches"),
        mock.patch.object(v027_mrv1._model_region, "prepare_model_region_step"),
        mock.patch.object(v027_mrv1._model_region, "stage_signed_backend_region_event"),
        mock.patch.object(v027_mrv1._sampler_patch, "maybe_prepare_sampler_decode_step"),
        mock.patch.object(v027_mrv1._ADAPTER, "unpack_prepare_inputs_output", return_value=view),
        mock.patch("tllm.runtime.request_lifecycle.observe_request_aliases"),
        mock.patch.object(
            v027_mrv1,
            "runner_model_module",
            return_value=torch.nn.Identity(),
        ),
        mock.patch.object(
            v027_mrv1,
            "runner_runtime_specialization",
            return_value=specialization,
        ),
        mock.patch.object(
            v027_mrv1._packed_hidden_capture,
            "begin_runner_packed_hidden_capture",
            side_effect=lambda **_kwargs: order.append(
                "begin:packed_hidden"
            ),
        ),
        mock.patch.object(
            v027_mrv1._packed_hidden_capture,
            "publish_runner_packed_hidden_capture",
            side_effect=lambda **_kwargs: order.append(
                "publish:packed_hidden"
            ),
        ),
    ):
        v027_mrv1.V027MRv1Driver().install(core=core, info=info)
        actual_runner.execute_model()

    assert v027_mrv1.get_step_session(actual_runner).batch_phase == _core.SchedulerBatchPhase.Prefill
    assert [event for event, _phase in events] == [
        BackendObservableEvent.PREPARE_INPUTS_PRE,
        BackendObservableEvent.PREPARE_INPUTS_POST,
        BackendObservableEvent.EXECUTE_MODEL_PRE,
        BackendObservableEvent.EXECUTE_MODEL_POST,
    ]
    assert [phase for _event, phase in events] == ["prefill"] * len(events)
    assert order.index("publish:packed_hidden") < order.index(
        "dispatch:execute_model.post"
    )

    # A later empty scheduler batch never reaches nested _prepare_inputs.
    # It must not inherit the previous step's phase or emit an execute post.
    actual_runner.prepare_inputs = False
    with (
        mock.patch.object(v027_mrv1.active_targets, "runtime_has_active_targets", return_value=True),
        mock.patch.object(v027_mrv1._common_hooks, "dispatch_runtime_event", side_effect=AssertionError),
        mock.patch.object(v027_mrv1._hooks, "setup_runtime_hooks_if_active"),
        mock.patch.object(v027_mrv1._hooks, "run_execute_model_post_producers"),
        mock.patch.object(v027_mrv1._hooks, "maybe_launch_post_logits_decode_work"),
    ):
        actual_runner.execute_model()
    assert v027_mrv1.get_step_session(actual_runner) is None


@pytest.mark.parametrize("driver", ("mrv1", "mrv2"))
def test_signed_prepare_post_binder_stages_after_pre_and_prepare_producers(
    driver: str,
) -> None:
    """A post-bound extern cannot observe the batch before its signed event."""

    implementation_id = "test.backend:prepare-post@1"
    slot = BackendRegionBinderSlot(implementation_id)
    order: list[str] = []

    def stage(context: BackendRegionStageContext) -> None:
        order.append("stage:prepare_inputs.post")
        context.slot.stage({}, lambda _plan, _inputs, _step_id: None, context.step_id)

    runtime = _runtime()
    runtime.workload_execution = SimpleNamespace(
        parameter_overlays={},
        requires_packed_rows=False,
        backend_region_binders={implementation_id: slot},
        backend_region_binder_stages=lambda event: (
            (
                BackendRegionBinderStage(
                    backend_event=BackendObservableEvent.PREPARE_INPUTS_POST,
                    region_id="region-post",
                    implementation_id=implementation_id,
                ),
            )
            if event is BackendObservableEvent.PREPARE_INPUTS_POST
            else ()
        ),
    )
    runtime.backend_region_stager_registry = BackendRegionStagerRegistry(
        {implementation_id: stage}
    )
    runner = SimpleNamespace(model=SimpleNamespace())

    def dispatch(**kwargs: object) -> int:
        order.append(f"dispatch:{kwargs['event'].value}")
        return 0

    if driver == "mrv1":
        runner.input_batch = _prefill_batch_mrv1()
        view = SimpleNamespace(
            logits_indices=None,
            spec_decode_metadata=None,
            num_scheduled_tokens_np=[2],
        )
        with (
            mock.patch.object(v027_mrv1.active_targets, "runtime_has_active_targets", return_value=True),
            mock.patch.object(v027_mrv1._common_hooks, "dispatch_runtime_event", side_effect=dispatch),
            mock.patch.object(v027_mrv1._hooks, "setup_runtime_hooks_if_active"),
            mock.patch.object(
                v027_mrv1._hooks,
                "run_prepare_inputs_post_producers",
                side_effect=lambda **_kwargs: order.append("producers"),
            ),
            mock.patch.object(v027_mrv1._sampler_patch, "maybe_prepare_sampler_decode_step"),
            mock.patch.object(v027_mrv1._ADAPTER, "unpack_prepare_inputs_output", return_value=view),
        ):
            v027_mrv1._wrapped_prepare_inputs(
                core=SimpleNamespace(RUNTIME=runtime),
                runner=runner,
                scheduler_output=object(),
                num_scheduled_tokens=[2],
                orig=mock.Mock(return_value=(None, None)),
            )
    else:
        batch = _prefill_batch_mrv2()
        runtime.event_step_id = 1
        with (
            mock.patch.object(v027_mrv2.active_targets, "runtime_has_active_targets", return_value=True),
            mock.patch.object(v027_mrv2._common_hooks, "dispatch_runtime_event", side_effect=dispatch),
            mock.patch.object(
                v027_mrv2._hooks,
                "run_prepare_inputs_post_producers",
                side_effect=lambda **_kwargs: order.append("producers"),
            ),
            mock.patch.object(v027_mrv2._sampler_patch, "maybe_prepare_sampler_decode_step"),
        ):
            from tllm.runtime.step_session import StepSession, set_step_session

            set_step_session(runner, StepSession(step_id=1, backend_id=driver))
            v027_mrv2._wrapped_prepare_inputs(
                core=SimpleNamespace(RUNTIME=runtime),
                runner=runner,
                args=(),
                kwargs={},
                orig=mock.Mock(return_value=batch),
            )

    expected = [
        "dispatch:prepare_inputs.pre",
        "producers",
        "stage:prepare_inputs.post",
        "dispatch:prepare_inputs.post",
        "dispatch:execute_model.pre",
    ]
    assert order == expected
