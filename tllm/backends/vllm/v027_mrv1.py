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

from tllm.backends.protocol import (
    BackendCapabilities,
    GraphSafety,
    RunnerBinding,
    SamplerStage,
    UnsupportedBackendError,
)
from tllm.backends.vllm import sampler_mrv1 as _sampler_mrv1
from tllm.backends.vllm.workgraph_runtime import record_resolved_backend
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
from tllm.runtime.vllm_patch.adapters.v027 import V027PrepareInputsAdapter

_ADAPTER = V027PrepareInputsAdapter()


def _test_time_lora_enabled(runtime: Any) -> bool:
    return bool(
        getattr(runtime, "test_time_lora_enabled", False)
        or getattr(runtime, "test_time_lora_service", None) is not None
    )


def _new_step_session(*, core: Any, runner: Any, view: Any, num_scheduled_tokens: Any) -> StepSession:
    runtime = core.RUNTIME
    request_ids = tuple(str(x) for x in getattr(runtime, "decode_request_ids", ()))
    session = StepSession(
        step_id=int(getattr(runtime, "event_step_id", 0)),
        backend_id=V027MRv1Driver.backend_id,
        request_ids=request_ids,
        request_index={req_id: i for i, req_id in enumerate(request_ids)},
        num_scheduled_tokens=num_scheduled_tokens,
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
) -> Any:
    runtime = core.RUNTIME
    if num_scheduled_tokens is None:
        raise UnsupportedBackendError(
            "vLLM 0.25+ MRv1 _prepare_inputs requires num_scheduled_tokens; "
            "the engine called it with a legacy single-argument shape. "
            "Refusing to guess: install a matching backend driver."
        )
    if not active_targets.runtime_has_active_targets(runtime):
        if _test_time_lora_enabled(runtime):
            _test_time_lora.maybe_commit_and_apply_request_mapping(runtime=runtime, runner=runner)
        out = orig(runner, scheduler_output, num_scheduled_tokens)
        view = _ADAPTER.unpack_prepare_inputs_output(
            runner=runner,
            scheduler_output=scheduler_output,
            out=out,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        _fast_weight.update_fast_weight_metadata_from_runner(
            runtime,
            runner,
            num_scheduled_tokens_np=view.num_scheduled_tokens_np,
        )
        return out

    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="prepare_inputs.pre",
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
    _fast_weight.update_fast_weight_metadata_from_runner(
        runtime,
        runner,
        num_scheduled_tokens_np=view.num_scheduled_tokens_np,
    )
    _sampler_patch.maybe_prepare_sampler_decode_step(runtime=runtime, runner=runner)
    session = _new_step_session(core=core, runner=runner, view=view, num_scheduled_tokens=num_scheduled_tokens)
    session.spec_rows = _build_spec_rows_mrv1(core=core, view=view, session=session)
    _restore_first_row_localization_mrv1(core=core, view=view, session=session)
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="prepare_inputs.post",
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
    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name="sample_tokens.post",
        phase="decode",
        layer_path=None,
        capture_enabled=bool(runtime.launch_consumer_from_hooks),
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
            out = _hooks.wrapped_load_model(core=core, runner=runner, args=args, kwargs=kwargs)
            from tllm.runtime.graph_policy import resolve_graph_mode

            record_resolved_backend(
                core=core,
                capabilities=self.capabilities(),
                graph_mode=resolve_graph_mode(runner),
            )
            return out

        def _prepare_inputs(runner: Any, scheduler_output: Any, num_scheduled_tokens: Any = None) -> Any:
            return _wrapped_prepare_inputs(
                core=core,
                runner=runner,
                scheduler_output=scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens,
                orig=orig_prepare_inputs,
            )

        def _execute_model(runner: Any, *args: Any, **kwargs: Any) -> Any:
            out = _hooks.wrapped_execute_model(core=core, runner=runner, args=args, kwargs=kwargs)
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
        binding.active = False
