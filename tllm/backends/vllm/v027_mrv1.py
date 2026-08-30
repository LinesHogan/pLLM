#!/usr/bin/env python3
"""Backend driver for vLLM 0.25+ MRv1 (``vllm.v1.worker.gpu_model_runner``).

Structural deltas versus the legacy driver:

* ``_prepare_inputs(scheduler_output, num_scheduled_tokens)`` and the return
  tuple is ``(logits_indices, spec_decode_metadata)``.
* One logical step spans ``execute_model`` (forward + logits) and
  ``sample_tokens`` (sampling); a :class:`StepSession` carries state across
  the two and the step only completes at ``sample_tokens.post``.
* min-p is native (built-in logits processor) — no InputBatch patch.
* ``Sampler.sample`` returns ``(sampled, processed_logprobs)`` — handled by
  :mod:`tllm.backends.vllm.sampler_mrv1`.
"""

from __future__ import annotations

from typing import Any

import torch

from tllm import _core
from tllm.backends.protocol import (
    BackendCapabilities,
    GraphSafety,
    ModelRegionCapabilities,
    OverlayBackendState,
    OrderedRowFusionCapabilities,
    RunnerBinding,
    SamplerStage,
    UnsupportedBackendError,
    VersionedParameterOverlayCapabilities,
)
from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.contracts.request_identity import RequestFinishedBatch
from tllm.contracts.backend_capability import BackendCapability
from tllm.backends.vllm import runner_adapter as _runner_adapter
from tllm.backends.vllm import model_region as _model_region
from tllm.backends.vllm import scheduler_batch_phase as _scheduler_batch_phase
from tllm.backends.vllm import versioned_parameter_overlay as _parameter_overlay
from tllm.backends.vllm import sampler_mrv1 as _sampler_mrv1
from tllm.backends.vllm.workgraph_runtime import (
    cold_compile_loaded_workloads,
    cold_specialize_loaded_runtime,
    record_resolved_backend,
    runner_tensor_parallel_size,
)
from tllm.runtime import active_targets
from tllm.runtime.step_session import (
    StepSession,
    StepState,
    get_step_session,
    set_step_session,
)
from tllm.runtime.vllm_patch import common_hooks as _common_hooks
from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks
from tllm.runtime.vllm_patch import sampler_patch as _sampler_patch
from tllm.runtime.vllm_patch.adapters.v027 import V027PrepareInputsAdapter
from tllm.runtime.workload_execution import BackendObservableEvent

_ADAPTER = V027PrepareInputsAdapter()


def _new_step_session(
    *,
    core: Any,
    runner: Any,
    view: Any,
    num_scheduled_tokens: Any,
    batch_phase: _core.SchedulerBatchPhase,
) -> StepSession:
    runtime = core.RUNTIME
    request_ids = tuple(str(x) for x in getattr(runtime, "decode_request_ids", ()))
    session = StepSession(
        step_id=int(getattr(runtime, "event_step_id", 0)),
        backend_id=V027MRv1Driver.backend_id,
        request_ids=request_ids,
        request_index={req_id: i for i, req_id in enumerate(request_ids)},
        num_scheduled_tokens=num_scheduled_tokens,
        batch_phase=batch_phase,
        logits_indices=view.logits_indices,
    )
    from tllm.runtime import graph_policy as _graph_policy

    session.graph_mode = _graph_policy.resolve_graph_mode(runner)
    session.transition(StepState.INPUTS_READY)
    set_step_session(runner, session)
    return session


def _build_spec_rows_mrv1(*, core: Any, view: Any, session: StepSession) -> tuple:
    """Strict per-logits-row mapping when MRv1 ran speculative decoding.

    MRv1's ``SpecDecodeMetadata`` carries ``num_draft_tokens`` per sampled
    request (batch order); request i contributes ``1 + num_draft_tokens[i]``
    logits rows, delimited by ``cu_num_sampled_tokens`` and mapped to packed
    hidden rows through ``logits_indices``.
    """
    metadata = getattr(view, "spec_decode_metadata", None)
    if metadata is None:
        return ()
    num_draft = list(getattr(metadata, "num_draft_tokens", ()) or ())
    request_ids = list(session.request_ids)
    logits_indices = view.logits_indices
    if (
        not num_draft
        or len(num_draft) != len(request_ids)
        or logits_indices is None
        or max(num_draft) == 0
    ):
        return ()
    from tllm.runtime.step_session import SpecRowMapping

    rows: list = []
    logit_row = 0
    for req_id, drafts in zip(request_ids, num_draft):
        # Consistent with the MRv2 mapping: only requests contributing more
        # than one logits row appear in the strict mapping.
        if int(drafts) > 0:
            prompt_idx, sample_idx = core._resolve_prompt_sample_for_req_id(req_id)
            for sampled_index in range(int(drafts) + 1):
                rows.append(
                    SpecRowMapping(
                        request_id=req_id,
                        prompt_idx=int(prompt_idx),
                        sample_idx=int(sample_idx),
                        sampled_index=sampled_index,
                        logit_row=logit_row + sampled_index,
                        hidden_row=int(logits_indices[logit_row + sampled_index].item()),
                    )
                )
        logit_row += int(drafts) + 1
    return tuple(rows)


def _restore_first_row_localization_mrv1(*, core: Any, view: Any, session: StepSession) -> None:
    """Rebuild decode scratch with correct first-row indices under spec decode.

    The legacy ``compute_decode_localization`` indexes ``logits_indices`` by
    request ordinal, which is only valid when every request contributes one
    logits row. With draft tokens present, request i owns rows
    ``[start_i, start_i + 1 + num_draft_tokens[i])``; the first row is the
    target position used for training. No-op when the batch has no drafts.
    """
    metadata = getattr(view, "spec_decode_metadata", None)
    if metadata is None:
        return
    num_draft = list(getattr(metadata, "num_draft_tokens", ()) or ())
    if not num_draft or max(num_draft) == 0:
        return
    runtime = core.RUNTIME
    request_ids = list(getattr(runtime, "decode_request_ids", ()) or ())
    logits_indices = view.logits_indices
    if len(num_draft) != len(request_ids) or logits_indices is None:
        import warnings

        warnings.warn(
            "tllm: spec-decode draft counts do not align with localized decode "
            "requests; keeping legacy first-row localization (training rows may "
            "be misaligned under speculative decoding)",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    first_rows: list[int] = []
    row = 0
    for drafts in num_draft:
        first_rows.append(row)
        row += int(drafts) + 1
    selector = torch.as_tensor(first_rows, dtype=torch.long, device=logits_indices.device)
    row_idx = logits_indices.index_select(0, selector)
    from tllm.runtime.vllm_patch.port_runtime_hooks import store_decode_localization

    store_decode_localization(
        core=core,
        row_idx=row_idx,
        prompt_idxs=list(getattr(runtime, "decode_prompt_idxs", ()) or ()),
        sample_idxs=list(getattr(runtime, "decode_sample_idxs", ()) or ()),
        request_ids=request_ids,
    )


def _wrapped_prepare_inputs(
    *,
    core: Any,
    runner: Any,
    scheduler_output: Any,
    num_scheduled_tokens: Any,
    orig: Any,
    driver: Any | None = None,
) -> Any:
    runtime = core.RUNTIME
    cold_compile_loaded_workloads(core=core, runner=runner)
    if num_scheduled_tokens is None:
        raise UnsupportedBackendError(
            "vLLM 0.25+ MRv1 _prepare_inputs requires num_scheduled_tokens; "
            "the engine called it with a legacy single-argument shape. "
            "Refusing to guess: install a matching backend driver."
        )
    binding = _runner_adapter.get_runner_binding(runner)
    overlay_controller = _parameter_overlay.maybe_bind_publisher(
        runtime=runtime,
        runner=runner,
        state=None if binding is None else binding.parameter_overlay_state,
    )
    if overlay_controller is not None:
        overlay_controller.finish_requests(
            _runner_adapter.mrv1_finished_request_ids(scheduler_output)
        )
        overlay_controller.sync_versions()
        overlay_controller.apply_mrv1_input_batch(runner.input_batch)
    if not active_targets.runtime_has_active_targets(runtime):
        out = orig(runner, scheduler_output, num_scheduled_tokens)
        view = _ADAPTER.unpack_prepare_inputs_output(
            runner=runner,
            scheduler_output=scheduler_output,
            out=out,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        _model_region.prepare_model_region_step(
            runtime=runtime,
            runner=runner,
            num_scheduled_tokens=view.num_scheduled_tokens_np,
        )
        return out

    active_driver = driver or V027MRv1Driver()
    batch_columns = active_driver.packed_row_plan_inputs(
        runner=runner,
        num_scheduled_tokens=num_scheduled_tokens,
        input_batch=None,
    )
    if (
        overlay_controller is not None
        and overlay_controller.has_dense_versions()
    ):
        overlay_controller.prepare_dense_step(
            step_id=int(runtime.event_step_id) + 1,
            columns=batch_columns,
        )
    batch_phase = _scheduler_batch_phase.scheduler_batch_phase(columns=batch_columns)
    hook_phase = _scheduler_batch_phase.hook_phase(batch_phase)

    from tllm.runtime import residual_runtime as _residual_runtime

    if core is _residual_runtime:
        _residual_runtime.prepare_physical_workgraph_step(
            # MRv1 increments the public step id after input construction.
            # Prepare that next step before its PREPARE lifecycle event.
            int(runtime.event_step_id) + 1
        )
    _model_region.stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_PRE,
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=num_scheduled_tokens,
        step_id=runtime.event_step_id + 1,
    )
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event=BackendObservableEvent.PREPARE_INPUTS_PRE,
        phase=hook_phase,
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
    )
    model_hook_flag = getattr(core, "MODEL_HOOK_FLAG", "_tllm_residual_hooks_installed")
    if getattr(getattr(runner, "model", None), model_hook_flag, False):
        runtime.launch_consumer_from_hooks = not core._runner_uses_compilation_or_cudagraph(runner)
    else:
        _hooks.setup_runtime_hooks_if_active(core=core, runner=runner)
    out = orig(runner, scheduler_output, num_scheduled_tokens)
    runtime.event_step_id += 1
    view = _ADAPTER.unpack_prepare_inputs_output(
        runner=runner,
        scheduler_output=scheduler_output,
        out=out,
        num_scheduled_tokens=num_scheduled_tokens,
    )
    _hooks.run_prepare_inputs_post_producers(
        core=core,
        runner=runner,
        localize_decode=lambda: _hooks.prepare_decode_localization(
            core=core,
            runner=runner,
            out=out,
            prepare_inputs_view=view,
        ),
    )
    _model_region.prepare_model_region_step(
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=view.num_scheduled_tokens_np,
    )
    try:
        _sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)
        session = _new_step_session(
            core=core,
            runner=runner,
            view=view,
            num_scheduled_tokens=num_scheduled_tokens,
            batch_phase=batch_phase,
        )
        session.spec_rows = _build_spec_rows_mrv1(core=core, view=view, session=session)
        _restore_first_row_localization_mrv1(core=core, view=view, session=session)
        _model_region.stage_signed_backend_region_event(
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            runtime=runtime,
            runner=runner,
            num_scheduled_tokens=view.num_scheduled_tokens_np,
        )
        _common_hooks.dispatch_runtime_event(
            runtime=runtime,
            runner=runner,
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            phase=hook_phase,
            layer_path=None,
            capture_enabled=bool(runtime.launch_consumer_from_hooks),
        )
        # MRv1 constructs inputs from inside its original execute_model.
        # The Rust-signed phase therefore only exists here, before the model
        # body continues, rather than at the outer execute wrapper entry.
        _model_region.stage_signed_backend_region_event(
            event=BackendObservableEvent.EXECUTE_MODEL_PRE,
            runtime=runtime,
            runner=runner,
            num_scheduled_tokens=view.num_scheduled_tokens_np,
            step_id=session.step_id,
        )
        _common_hooks.dispatch_runtime_event(
            runtime=runtime,
            runner=runner,
            event=BackendObservableEvent.EXECUTE_MODEL_PRE,
            phase=hook_phase,
            layer_path=None,
            capture_enabled=bool(runtime.launch_consumer_from_hooks),
        )
    except Exception:
        _model_region.abort_model_region_step(runtime, int(runtime.event_step_id))
        raise
    return out


def _wrapped_sample_tokens(*, core: Any, runner: Any, grammar_output: Any, orig: Any) -> Any:
    runtime = core.RUNTIME
    if not active_targets.runtime_has_active_targets(runtime):
        return orig(runner, grammar_output)
    session = get_step_session(runner)
    hook_phase = _scheduler_batch_phase.session_hook_phase(session)
    _model_region.stage_signed_backend_region_event(
        event=BackendObservableEvent.SAMPLE_TOKENS_PRE,
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=None if session is None else session.num_scheduled_tokens,
    )
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event=BackendObservableEvent.SAMPLE_TOKENS_PRE,
        phase=hook_phase,
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
    )
    if session is not None and session.state in (StepState.INPUTS_READY, StepState.FORWARD_DONE):
        if session.state == StepState.INPUTS_READY:
            session.transition(StepState.FORWARD_DONE)
        state = getattr(runner, "execute_model_state", None)
        if state is not None:
            session.hidden_states = getattr(state, "hidden_states", None)
            session.sample_hidden_states = getattr(state, "sample_hidden_states", None)
            session.logits = getattr(state, "logits", None)
        session.transition(StepState.SAMPLING)
    try:
        out = orig(runner, grammar_output)
    except Exception as exc:
        if session is not None:
            session.fail(str(exc))
        raise
    if session is not None and session.state == StepState.SAMPLING:
        session.sampled_tokens = getattr(out, "sampled_token_ids", None)
        session.transition(StepState.COMPLETED)
    _model_region.stage_signed_backend_region_event(
        event=BackendObservableEvent.SAMPLE_TOKENS_POST,
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=None if session is None else session.num_scheduled_tokens,
    )
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event=BackendObservableEvent.SAMPLE_TOKENS_POST,
        phase=hook_phase,
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
    )
    from tllm.runtime import residual_runtime as _residual_runtime

    if core is _residual_runtime:
        _residual_runtime.finish_workgraph_distributed_step(
            int(runtime.event_step_id)
        )
    return out


class V027MRv1Driver:
    backend_id = "vllm-0.27-mrv1"
    _HOOK_FLAG = "_tllm_v027_mrv1_hooks"

    def supports(self, info: Any) -> bool:
        return bool(
            info.mrv1_runner_cls is not None
            and info.mrv1_has_sample_tokens
            and info.mrv1_prepare_inputs_arity == 2
            and info.sampler_returns_tuple
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            runner_family="mrv1",
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

    def export_main_workload_timeline(
        self, *, graph_mode: str, static_headroom_bytes: int
    ) -> Any:
        from tllm.backends.timeline import export_main_workload_timeline

        return export_main_workload_timeline(
            self.capabilities(),
            graph_mode=graph_mode,
            static_headroom_bytes=static_headroom_bytes,
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
        return _runner_adapter.mrv1_packed_row_plan_inputs(
            runner=runner,
            num_scheduled_tokens=num_scheduled_tokens,
            input_batch=input_batch,
        )

    def install(self, *, core: Any, info: Any) -> None:
        cls = info.mrv1_runner_cls
        if getattr(cls, self._HOOK_FLAG, False):
            return
        orig_load_model = cls.load_model
        orig_prepare_inputs = cls._prepare_inputs
        orig_execute_model = cls.execute_model
        orig_sample_tokens = cls.sample_tokens

        # Legacy-compat attributes on the core module (existing tests and the
        # shared hook helpers read them).
        core.GPUModelRunner = cls
        core._ORIG_LOAD_MODEL = orig_load_model
        core._ORIG_PREPARE_INPUTS = orig_prepare_inputs
        core._ORIG_EXECUTE_MODEL = orig_execute_model
        core._ORIG_SAMPLE_TOKENS = orig_sample_tokens

        def _load_model(runner: Any, *args: Any, **kwargs: Any) -> Any:
            binding = _runner_adapter.ensure_runner_binding(
                driver=self, runner=runner
            )
            out = _hooks.wrapped_load_model(core=core, runner=runner, args=args, kwargs=kwargs)
            _parameter_overlay.maybe_bind_publisher(
                runtime=core.RUNTIME,
                runner=runner,
                state=binding.parameter_overlay_state,
            )
            from tllm.runtime.graph_policy import resolve_graph_mode

            record_resolved_backend(
                core=core,
                capabilities=self.capabilities(),
                graph_mode=resolve_graph_mode(runner),
                tp_size=runner_tensor_parallel_size(runner),
            )
            cold_specialize_loaded_runtime(core=core, runner=runner)
            return out

        def _prepare_inputs(runner: Any, scheduler_output: Any, num_scheduled_tokens: Any = None) -> Any:
            return _wrapped_prepare_inputs(
                core=core,
                runner=runner,
                scheduler_output=scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens,
                orig=orig_prepare_inputs,
                driver=self,
            )

        def _execute_model(runner: Any, *args: Any, **kwargs: Any) -> Any:
            active = active_targets.runtime_has_active_targets(core.RUNTIME)
            binding = _runner_adapter.get_runner_binding(runner)
            overlay_controller = _parameter_overlay.maybe_bind_publisher(
                runtime=core.RUNTIME,
                runner=runner,
                state=(
                    None
                    if binding is None
                    else binding.parameter_overlay_state
                ),
            )
            if active:
                # The next session is created by the nested _prepare_inputs
                # call inside vLLM's original execute_model. Clearing the
                # completed prior step also prevents an empty batch from
                # borrowing its phase.
                set_step_session(runner, None)

            def _phase_after_nested_prepare() -> str | None:
                session = get_step_session(runner)
                if session is None:
                    return None
                return _scheduler_batch_phase.session_hook_phase(session)

            def _stage_event(event: BackendObservableEvent) -> None:
                session = get_step_session(runner)
                _model_region.stage_signed_backend_region_event(
                    event=event,
                    runtime=core.RUNTIME,
                    runner=runner,
                    num_scheduled_tokens=(
                        None if session is None else session.num_scheduled_tokens
                    ),
                    step_id=None if session is None else session.step_id,
                )

            try:
                out = _hooks.wrapped_execute_model(
                    core=core,
                    runner=runner,
                    args=args,
                    kwargs=kwargs,
                    split_execute_and_sample=True,
                    phase=None,
                    before_runtime_event=_stage_event,
                    execute_pre_already_dispatched=active,
                    phase_after_inputs=(
                        _phase_after_nested_prepare if active else None
                    ),
                )
            finally:
                if overlay_controller is not None:
                    overlay_controller.clear_dense_step()
            session = get_step_session(runner)
            if session is not None and session.state == StepState.INPUTS_READY:
                session.transition(StepState.FORWARD_DONE)
            return out

        def _sample_tokens(runner: Any, grammar_output: Any = None) -> Any:
            return _wrapped_sample_tokens(
                core=core,
                runner=runner,
                grammar_output=grammar_output,
                orig=orig_sample_tokens,
            )

        cls.load_model = _load_model
        cls._prepare_inputs = _prepare_inputs
        cls.execute_model = _execute_model
        cls.sample_tokens = _sample_tokens
        setattr(cls, self._HOOK_FLAG, True)

        _sampler_mrv1.install_sampler_patch_v027(core=core)

    def unbind(self, binding: RunnerBinding) -> None:
        _parameter_overlay.unbind_parameter_overlays(
            runner=binding.runner,
            state=binding.parameter_overlay_state,
        )
        _runner_adapter.unbind_runner(binding)

    def finish_parameter_overlay_requests(
        self,
        *,
        runtime: Any,
        runner: Any,
        state: OverlayBackendState,
        request_ids: tuple[str, ...],
    ) -> RequestFinishedBatch:
        return _parameter_overlay.finish_parameter_overlay_requests(
            runtime=runtime,
            runner=runner,
            state=state,
            request_ids=request_ids,
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
