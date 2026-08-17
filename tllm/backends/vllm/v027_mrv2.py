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
    RunnerBinding,
    SamplerStage,
)
from tllm.backends.vllm import sampler_mrv2 as _sampler_mrv2
from tllm.runtime import active_targets
from tllm.runtime.fast_weight import runtime as _fast_weight
from tllm.runtime.vllm_patch import test_time_lora_hooks as _test_time_lora
from tllm.runtime.step_session import (
    StepSession,
    StepState,
    get_step_session,
    set_step_session,
)
from tllm.runtime.vllm_patch import common_hooks as _common_hooks
from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks
from tllm.runtime.vllm_patch import sampler_patch as _sampler_patch


def _test_time_lora_enabled(runtime: Any) -> bool:
    return bool(
        getattr(runtime, "test_time_lora_enabled", False)
        or getattr(runtime, "test_time_lora_service", None) is not None
    )


def _decode_localization_from_mrv2(*, core: Any, runner: Any, input_batch: Any) -> tuple:
    """Fill decode scratch buffers from the MRv2 persistent InputBatch.

    MRv2 keeps one logits row per scheduled request (more under spec
    decode); ``input_batch.logits_indices`` maps packed hidden rows to the
    token positions that produce logits and ``idx_mapping_np`` maps logits
    rows to request slots. Returns the strict spec-decode row mapping.
    """
    from tllm.backends.vllm.mrv2_localization import prepare_decode_localization_mrv2

    return prepare_decode_localization_mrv2(core=core, runner=runner, input_batch=input_batch)


def _wrapped_prepare_inputs(*, core: Any, runner: Any, args: tuple, kwargs: dict, orig: Any) -> Any:
    runtime = core.RUNTIME
    input_batch = orig(runner, *args, **kwargs)
    if not active_targets.runtime_has_active_targets(runtime):
        return input_batch
    session = get_step_session(runner)
    spec_rows = _decode_localization_from_mrv2(core=core, runner=runner, input_batch=input_batch)
    _sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)
    if session is not None and session.state == StepState.CREATED:
        request_ids = tuple(str(x) for x in getattr(runtime, "decode_request_ids", ()))
        session.request_ids = request_ids
        session.request_index = {req_id: i for i, req_id in enumerate(request_ids)}
        session.num_scheduled_tokens = getattr(input_batch, "num_scheduled_tokens", None)
        session.logits_indices = getattr(input_batch, "logits_indices", None)
        session.spec_rows = tuple(spec_rows)
        from tllm.runtime import graph_policy as _graph_policy

        session.graph_mode = _graph_policy.resolve_graph_mode(runner)
        session.transition(StepState.INPUTS_READY)
    return input_batch


def _wrapped_execute_model(*, core: Any, runner: Any, args: tuple, kwargs: dict, orig: Any) -> Any:
    runtime = core.RUNTIME
    if not active_targets.runtime_has_active_targets(runtime):
        return orig(runner, *args, **kwargs)

    runtime.event_step_id += 1
    session = StepSession(
        step_id=int(runtime.event_step_id),
        backend_id=V027MRv2Driver.backend_id,
    )
    set_step_session(runner, session)
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="execute_model.pre",
        phase="decode",
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
    )
    model_hook_flag = getattr(core, "MODEL_HOOK_FLAG", "_tllm_residual_hooks_installed")
    if getattr(getattr(runner, "model", None), model_hook_flag, False):
        runtime.launch_consumer_from_hooks = not core._runner_uses_compilation_or_cudagraph(runner)
    else:
        _hooks.setup_runtime_hooks_if_active(core=core, runner=runner)
    _test_time_lora.maybe_commit_and_apply_request_mapping(runtime=runtime, runner=runner)

    try:
        out = orig(runner, *args, **kwargs)
    except Exception as exc:
        session.fail(str(exc))
        raise

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
    _hooks.dispatch_decode_port_bundles(core=core, runner=runner)
    if not runtime.launch_consumer_from_hooks:
        from tllm.runtime import hidden_event_bridge as _hidden_bridge

        _hidden_bridge.dispatch_deferred_layer_batches(core=core, runner=runner)
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="execute_model.post",
        phase="decode",
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
    )
    return out


def _wrapped_sample_tokens(*, core: Any, runner: Any, grammar_output: Any, orig: Any) -> Any:
    runtime = core.RUNTIME
    if not active_targets.runtime_has_active_targets(runtime):
        return orig(runner, grammar_output)
    session = get_step_session(runner)
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="sample_tokens.pre",
        phase="decode",
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
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
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="sample_tokens.post",
        phase="decode",
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
    )
    return out


def _wrapped_load_model(*, core: Any, runner: Any, args: tuple, kwargs: dict, orig: Any) -> Any:
    runtime = core.RUNTIME
    if not active_targets.runtime_has_active_targets(runtime):
        out = orig(runner, *args, **kwargs)
        if _test_time_lora_enabled(runtime):
            _test_time_lora.maybe_bind_test_time_lora_service(runtime=runtime, runner=runner)
        _fast_weight.maybe_install_fast_weight_wrappers(runtime=runtime, runner=runner)
        _sampler_mrv2.bind_runner_sampler_mrv2(runtime=runtime, runner=runner)
        return out
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="load_model.pre",
        phase=None,
        layer_path=None,
        capture_enabled=False,
    )
    out = orig(runner, *args, **kwargs)
    _sampler_mrv2.bind_runner_sampler_mrv2(runtime=runtime, runner=runner)
    _test_time_lora.maybe_bind_test_time_lora_service(runtime=runtime, runner=runner)
    _fast_weight.maybe_install_fast_weight_wrappers(runtime=runtime, runner=runner)
    _hooks.setup_runtime_hooks_if_active(core=core, runner=runner)
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="load_model.post",
        phase=None,
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
    )
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
            return _wrapped_load_model(core=core, runner=runner, args=args, kwargs=kwargs, orig=orig_load_model)

        def _prepare_inputs(runner: Any, *args: Any, **kwargs: Any) -> Any:
            return _wrapped_prepare_inputs(core=core, runner=runner, args=args, kwargs=kwargs, orig=orig_prepare_inputs)

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
        binding.active = False
