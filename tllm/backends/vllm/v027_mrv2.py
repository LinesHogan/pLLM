#!/usr/bin/env python3
"""Backend driver for vLLM 0.25+ MRv2 (``vllm.v1.worker.gpu.model_runner``).

MRv2 builds the input batch inside the runner, stores an
``ExecuteModelState`` on the runner between ``execute_model`` and
``sample_tokens``, and samples through a plain callable
``vllm.v1.worker.gpu.sample.sampler.Sampler`` instance. The driver:

* creates/refreshes a :class:`StepSession` at ``execute_model.pre``;
* fills hidden state / InputBatch references from ``execute_model_state``
  at ``execute_model.post``;
* wraps ``runner.sample``/``sample_tokens`` for the sampling lifecycle;
* replaces ``runner.sampler`` with an instance-scoped delegating
  interceptor (see :mod:`tllm.backends.vllm.sampler_mrv2`) — the Sampler
  class is never globally patched.
"""

from __future__ import annotations

from typing import Any

from tllm.backends.protocol import (
    BackendCapabilities,
    GraphSafety,
    ModelRegionCapabilities,
    OverlayBackendState,
    OrderedRowFusionCapabilities,
    RunnerBinding,
    SamplerStage,
    VersionedParameterOverlayCapabilities,
    runner_model_module,
)
from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.contracts.request_identity import RequestFinishedBatch
from tllm.contracts.backend_capability import BackendCapability
from tllm.backends.vllm import runner_adapter as _runner_adapter
from tllm.backends.vllm import model_region as _model_region
from tllm.backends.vllm import packed_hidden_capture as _packed_hidden_capture
from tllm.backends.vllm import scheduler_batch_phase as _scheduler_batch_phase
from tllm.backends.vllm import versioned_parameter_overlay as _parameter_overlay
from tllm.backends.vllm import sampler_mrv2 as _sampler_mrv2
from tllm.runtime import active_targets
from tllm.backends.vllm._step_session import (
    StepSession,
    StepState,
    get_step_session,
    set_step_session,
)
from tllm.runtime.vllm_patch import common_hooks as _common_hooks
from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks
from tllm.runtime.vllm_patch import sampler_patch as _sampler_patch
from tllm.backends.protocol import BackendObservableEvent


def _decode_localization_from_mrv2(*, core: Any, runner: Any, input_batch: Any) -> tuple:
    """Fill decode scratch buffers from the MRv2 persistent InputBatch.

    MRv2 keeps one logits row per scheduled request (more under spec
    decode); ``input_batch.logits_indices`` maps packed hidden rows to the
    token positions that produce logits and ``idx_mapping_np`` maps logits
    rows to request slots. Returns the strict spec-decode row mapping.
    """
    from tllm.backends.vllm.mrv2_localization import prepare_decode_localization_mrv2

    return prepare_decode_localization_mrv2(core=core, runner=runner, input_batch=input_batch)


def _wrapped_prepare_inputs(
    *,
    core: Any,
    runner: Any,
    args: tuple,
    kwargs: dict,
    orig: Any,
    driver: Any | None = None,
) -> Any:
    runtime = core.RUNTIME
    binding = _runner_adapter.get_runner_binding(runner)
    overlay_controller = _parameter_overlay.maybe_bind_publisher(
        runtime=runtime,
        runner=runner,
        state=None if binding is None else binding.parameter_overlay_state,
    )
    input_batch = orig(runner, *args, **kwargs)
    has_active_targets = active_targets.runtime_has_active_targets(runtime)
    active_driver = driver or V027MRv2Driver()
    batch_columns = active_driver.packed_row_plan_inputs(
        runner=runner,
        num_scheduled_tokens=getattr(input_batch, "num_scheduled_tokens", None),
        input_batch=input_batch,
    )
    if (
        has_active_targets
        and _runner_adapter.is_v027_warmup_packed_rows(batch_columns)
    ):
        session = get_step_session(runner)
        if session is None or session.state is not StepState.CREATED:
            raise RuntimeError(
                "MRv2 backend warmup requires its newly-created step session"
            )
        session.transition(StepState.BACKEND_WARMUP)
        return input_batch
    from tllm.runtime.request_lifecycle import observe_request_aliases

    observe_request_aliases(
        runtime,
        _runner_adapter.mrv2_active_request_ids(runner, input_batch),
    )
    if overlay_controller is not None:
        overlay_controller.sync_versions()
        overlay_controller.apply_mrv2_input_batch(input_batch)
        if overlay_controller.requires_dense_step():
            overlay_controller.prepare_dense_step(
                step_id=int(runtime.event_step_id), columns=batch_columns
            )

    if not has_active_targets:
        return input_batch

    # InputBatch is the first point where all three scheduler columns are
    # available in MRv2. Rust classifies them before any lifecycle event is
    # advanced through native workload execution.
    packed_execution = None
    batch_phase = _scheduler_batch_phase.scheduler_batch_phase(
        columns=batch_columns
    )
    _packed_hidden_capture.synchronize_runner_packed_hidden_capture(
        binding=binding,
        execution=None,
    )
    packed_rows = _scheduler_batch_phase.stage_prepared_packed_rows(
        execution=None,
        step_id=int(runtime.event_step_id),
        columns=batch_columns,
        prompt_indices=runtime.reqid_to_promptidx,
        sample_indices=runtime.reqid_to_sampleidx,
    )
    if (
        packed_rows is not None
        and packed_execution is not None
        and packed_execution.requires_packed_hidden
    ):
        from tllm.runtime.graph_policy import resolve_graph_mode

        _packed_hidden_capture.begin_runner_packed_hidden_capture(
            binding=binding,
            model=runner_model_module(runner),
            execution=packed_execution,
            specialization=_runner_adapter.runner_runtime_specialization(runner),
            graph_mode=resolve_graph_mode(runner),
            row_batch=packed_rows,
        )
    if (
        packed_rows is not None
        and packed_execution is not None
        and packed_execution.requires_packed_sampler_rows
    ):
        packed_execution.stage_packed_sampler_rows(
            _runner_adapter.mrv2_packed_sampler_row_batch(
                row_batch=packed_rows,
                input_batch=input_batch,
            )
        )
    session = get_step_session(runner)
    if session is None or session.state is not StepState.CREATED:
        raise RuntimeError(
            "MRv2 prepare_inputs requires a newly-created step session before "
            "Rust-derived batch phase dispatch"
        )
    session.batch_phase = batch_phase
    # MRv2 forms InputBatch inside execute_model. Delay PRE until the exact
    # scheduler columns exist, while still emitting it before model compute.
    _model_region.stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_PRE,
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=getattr(input_batch, "num_scheduled_tokens", None),
        input_batch=input_batch,
    )
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        event=BackendObservableEvent.PREPARE_INPUTS_PRE,
    )
    spec_rows: tuple = ()

    def _localize_decode() -> None:
        nonlocal spec_rows
        spec_rows = _decode_localization_from_mrv2(
            core=core,
            runner=runner,
            input_batch=input_batch,
        )

    _hooks.prepare_native_decode_metadata(
        core=core,
        runner=runner,
        localize_decode=_localize_decode,
    )
    _sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)
    if session.state == StepState.CREATED:
        request_ids = tuple(str(x) for x in getattr(runtime, "decode_request_ids", ()))
        session.request_ids = request_ids
        session.request_index = {req_id: i for i, req_id in enumerate(request_ids)}
        session.num_scheduled_tokens = getattr(input_batch, "num_scheduled_tokens", None)
        session.logits_indices = getattr(input_batch, "logits_indices", None)
        session.spec_rows = tuple(spec_rows)
        from tllm.runtime import graph_policy as _graph_policy

        session.graph_mode = _graph_policy.resolve_graph_mode(runner)
        session.transition(StepState.INPUTS_READY)
        # MRv2 prepares its InputBatch inside execute_model.  Keep the typed
        # lifecycle ordered as PREPARE_INPUTS_PRE -> POST -> EXECUTE_MODEL_PRE
        # around that construction and its post-prepare native metadata.
        _model_region.stage_signed_backend_region_event(
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            runtime=runtime,
            runner=runner,
            num_scheduled_tokens=getattr(input_batch, "num_scheduled_tokens", None),
            input_batch=input_batch,
        )
        _common_hooks.dispatch_runtime_event(
            runtime=runtime,
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
        )
        _model_region.stage_signed_backend_region_event(
            event=BackendObservableEvent.EXECUTE_MODEL_PRE,
            runtime=runtime,
            runner=runner,
            num_scheduled_tokens=getattr(input_batch, "num_scheduled_tokens", None),
            input_batch=input_batch,
        )
        _common_hooks.dispatch_runtime_event(
            runtime=runtime,
            event=BackendObservableEvent.EXECUTE_MODEL_PRE,
        )
    return input_batch


def _wrapped_execute_model(*, core: Any, runner: Any, args: tuple, kwargs: dict, orig: Any) -> Any:
    runtime = core.RUNTIME
    binding = _runner_adapter.get_runner_binding(runner)
    _packed_hidden_capture.synchronize_runner_packed_hidden_capture(
        binding=binding,
        execution=None,
    )
    overlay_controller = _parameter_overlay.maybe_bind_publisher(
        runtime=runtime,
        runner=runner,
        state=None if binding is None else binding.parameter_overlay_state,
    )
    if overlay_controller is not None:
        overlay_controller.sync_versions()
    if not active_targets.runtime_has_active_targets(runtime):
        return orig(runner, *args, **kwargs)

    runtime.event_step_id += 1
    from tllm.runtime import residual_runtime as _residual_runtime

    if core is _residual_runtime:
        _residual_runtime.prepare_physical_workgraph_step(int(runtime.event_step_id))
    session = StepSession(
        step_id=int(runtime.event_step_id),
        backend_id=V027MRv2Driver.backend_id,
    )
    set_step_session(runner, session)
    model_hook_flag = getattr(core, "MODEL_HOOK_FLAG", "_tllm_residual_hooks_installed")
    if getattr(getattr(runner, "model", None), model_hook_flag, False):
        runtime.launch_consumer_from_hooks = not core._runner_uses_compilation_or_cudagraph(runner)
    else:
        _hooks.setup_runtime_hooks_if_active(core=core, runner=runner)
    try:
        out = orig(runner, *args, **kwargs)
        if session.state is StepState.BACKEND_WARMUP:
            return out
        if session.state is not StepState.CREATED:
            _packed_hidden_capture.publish_runner_packed_hidden_capture(
                binding=binding,
                execution=None,
            )
        if overlay_controller is not None:
            overlay_controller.stage_dense_operand_captured()
    except Exception as exc:
        if session.state is StepState.BACKEND_WARMUP:
            raise
        _packed_hidden_capture.abort_runner_packed_hidden_capture(binding)
        session.fail(str(exc))
        _model_region.abort_model_region_step(runtime, int(runtime.event_step_id))
        raise
    finally:
        if overlay_controller is not None:
            overlay_controller.clear_dense_step()

    state = getattr(runner, "execute_model_state", None)
    if state is not None:
        session.hidden_states = getattr(state, "hidden_states", None)
        input_batch = getattr(state, "input_batch", None)
        if input_batch is not None:
            session.expanded_idx_mapping = getattr(input_batch, "expanded_idx_mapping", None)
            session.logits_indices = getattr(input_batch, "logits_indices", None)
    if session.state == StepState.INPUTS_READY:
        session.transition(StepState.FORWARD_DONE)
    elif session.state == StepState.CREATED:
        # prepare_inputs was not wrapped into this step (e.g. dummy run);
        # there is nothing to consume.
        return out

    if not bool(getattr(getattr(runner, "model", None), "_tllm_compute_logits_wrapped", False)):
        _hooks.maybe_launch_post_logits_decode_work(core=core, runner=runner)
    _model_region.stage_signed_backend_region_event(
        event=BackendObservableEvent.EXECUTE_MODEL_POST,
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=session.num_scheduled_tokens,
        input_batch=getattr(getattr(runner, "execute_model_state", None), "input_batch", None),
    )
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        event=BackendObservableEvent.EXECUTE_MODEL_POST,
    )
    return out


def _wrapped_sample_tokens(*, core: Any, runner: Any, grammar_output: Any, orig: Any) -> Any:
    runtime = core.RUNTIME
    if not active_targets.runtime_has_active_targets(runtime):
        return orig(runner, grammar_output)
    session = get_step_session(runner)
    if session is not None and session.state is StepState.BACKEND_WARMUP:
        return orig(runner, grammar_output)
    _model_region.stage_signed_backend_region_event(
        event=BackendObservableEvent.SAMPLE_TOKENS_PRE,
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=None,
    )
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        event=BackendObservableEvent.SAMPLE_TOKENS_PRE,
    )
    if session is not None and session.state == StepState.FORWARD_DONE:
        session.transition(StepState.SAMPLING)
    try:
        out = orig(runner, grammar_output)
    except Exception as exc:
        if session is not None:
            session.fail(str(exc))
        raise
    if session is not None and session.state == StepState.SAMPLING:
        session.transition(StepState.COMPLETED)
    _model_region.stage_signed_backend_region_event(
        event=BackendObservableEvent.SAMPLE_TOKENS_POST,
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=None,
    )
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        event=BackendObservableEvent.SAMPLE_TOKENS_POST,
    )
    return out


def _wrapped_load_model(*, core: Any, runner: Any, args: tuple, kwargs: dict, orig: Any) -> Any:
    runtime = core.RUNTIME

    if not active_targets.runtime_has_active_targets(runtime):
        out = orig(runner, *args, **kwargs)
        _sampler_mrv2.bind_runner_sampler_mrv2(runtime=runtime, runner=runner)
        return out
    out = orig(runner, *args, **kwargs)
    _sampler_mrv2.bind_runner_sampler_mrv2(runtime=runtime, runner=runner)
    _hooks.setup_runtime_hooks_if_active(core=core, runner=runner)
    return out


class V027MRv2Driver:
    backend_id = "vllm-0.27-mrv2"
    _HOOK_FLAG = "_tllm_v027_mrv2_hooks"

    def supports(self, info: Any) -> bool:
        return bool(info.mrv2_runner_cls is not None and info.mrv2_has_sample_tokens)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            runner_family="mrv2",
            sampler_stages=frozenset(
                {
                    SamplerStage.RAW_LOGITS,
                    SamplerStage.AFTER_TEMPERATURE,
                    SamplerStage.AFTER_MIN_P,
                    SamplerStage.AFTER_TOP_K_TOP_P,
                    SamplerStage.BEFORE_DRAW,
                    SamplerStage.AFTER_DRAW,
                }
            ),
            native_min_p=True,
            tuple_sampler_return=True,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.CAPTURE_SAFE,
            extensions=frozenset(
                {
                    BackendCapability.BACKEND_REGION_BINDER,
                    BackendCapability.TYPED_LAYER_REPLAY,
                }
            ),
            eager_extensions=frozenset(
                {
                    BackendCapability.PACKED_PHASE_ROWS,
                    BackendCapability.PACKED_SAMPLER_ROWS,
                    BackendCapability.PACKED_HIDDEN_BLOCK_OUTPUT,
                }
            ),
            hidden_locators=frozenset(
                {HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}
            ),
            parameter_overlay=VersionedParameterOverlayCapabilities(
                representations=frozenset(
                    {"low_rank_add", "dense_replace"}
                ),
                scopes=frozenset({"global", "request"}),
                phases=frozenset({"prefill", "decode"}),
                pinning=frozenset(
                    {"latest_at_admission", "explicit_request_advance"}
                ),
                mixed_versions=True,
                operand_captures=frozenset({"mlp_down_proj"}),
                graph_safety=GraphSafety.EAGER_ONLY,
                tensor_parallel_sizes=frozenset({1}),
            ),
            model_region=ModelRegionCapabilities(
                phases=frozenset({"prefill"}),
                replay=True,
                kv_snapshot=False,
                kv_overwrite=True,
                ordered_row_fusion=OrderedRowFusionCapabilities(
                    internalized_edges=frozenset({"kv_overwrite_read"}),
                    max_rows=2,
                ),
                graph_safety=GraphSafety.EAGER_ONLY,
                tensor_parallel_sizes=frozenset({1}),
            ),
        )

    def install_typed_layer_replay(self, *, runtime: Any, runner: Any) -> Any:
        from tllm.backends.vllm import typed_layer_replay

        return typed_layer_replay.install_typed_layer_replay(
            runtime=runtime, runner=runner
        )

    def restore_typed_layer_replay(self, *, runtime: Any) -> int:
        from tllm.backends.vllm import typed_layer_replay

        return typed_layer_replay.restore_typed_layer_replay(runtime)

    def runner_model_identity(self, runner: Any) -> int:
        return _runner_adapter.runner_model_identity(runner)

    def runner_active_request_count(self, runner: Any) -> int:
        return _runner_adapter.runner_active_request_count(runner)

    def runner_model_layer_count(self, runner: Any) -> int:
        return _runner_adapter.runner_model_layer_count(runner)

    def runner_tensor_parallel_size(self, runner: Any) -> int:
        return _runner_adapter.runner_tensor_parallel_size(runner)

    def packed_row_plan_inputs(
        self,
        *,
        runner: Any,
        num_scheduled_tokens: Any | None,
        input_batch: Any | None,
    ) -> Any:
        return _runner_adapter.mrv2_packed_row_plan_inputs(
            runner=runner,
            num_scheduled_tokens=num_scheduled_tokens,
            input_batch=input_batch,
        )

    def install(self, *, core: Any, info: Any) -> None:
        cls = info.mrv2_runner_cls
        if getattr(cls, self._HOOK_FLAG, False):
            return
        orig_load_model = cls.load_model
        orig_prepare_inputs = cls.prepare_inputs
        orig_execute_model = cls.execute_model
        orig_sample_tokens = cls.sample_tokens

        def _load_model(runner: Any, *args: Any, **kwargs: Any) -> Any:
            binding = _runner_adapter.ensure_runner_binding(
                driver=self, runner=runner
            )
            out = _wrapped_load_model(
                core=core,
                runner=runner,
                args=args,
                kwargs=kwargs,
                orig=orig_load_model,
            )
            _parameter_overlay.maybe_bind_publisher(
                runtime=core.RUNTIME,
                runner=runner,
                state=binding.parameter_overlay_state,
            )
            return out

        def _prepare_inputs(runner: Any, *args: Any, **kwargs: Any) -> Any:
            return _wrapped_prepare_inputs(
                core=core,
                runner=runner,
                args=args,
                kwargs=kwargs,
                orig=orig_prepare_inputs,
                driver=self,
            )

        def _execute_model(runner: Any, *args: Any, **kwargs: Any) -> Any:
            return _wrapped_execute_model(core=core, runner=runner, args=args, kwargs=kwargs, orig=orig_execute_model)

        def _sample_tokens(runner: Any, grammar_output: Any = None) -> Any:
            return _wrapped_sample_tokens(core=core, runner=runner, grammar_output=grammar_output, orig=orig_sample_tokens)

        cls.load_model = _load_model
        cls.prepare_inputs = _prepare_inputs
        cls.execute_model = _execute_model
        cls.sample_tokens = _sample_tokens
        setattr(cls, self._HOOK_FLAG, True)

    def unbind(self, binding: RunnerBinding) -> None:
        _packed_hidden_capture.close_runner_packed_hidden_capture(binding)
        _parameter_overlay.unbind_parameter_overlays(
            runner=binding.runner,
            state=binding.parameter_overlay_state,
        )
        _runner_adapter.unbind_runner(binding)

    def consume_parameter_overlay_finished_batch(
        self,
        *,
        runtime: Any,
        runner: Any,
        state: OverlayBackendState,
        batch: RequestFinishedBatch,
    ) -> int:
        return _parameter_overlay.consume_parameter_overlay_finished_batch(
            runtime=runtime,
            runner=runner,
            state=state,
            batch=batch,
        )

    def synchronize_parameter_overlays(
        self,
        *,
        runtime: Any,
        runner: Any,
        state: OverlayBackendState,
    ) -> int:
        return _parameter_overlay.synchronize_parameter_overlays(
            runtime=runtime, runner=runner, state=state
        )
