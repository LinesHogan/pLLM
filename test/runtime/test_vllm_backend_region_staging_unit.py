"""vLLM staging must follow signed backend-Region identities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tllm.backends.vllm.model_region import (
    stage_signed_backend_region_event,
)
from tllm.runtime.backend_region_binders import (
    BackendRegionBinderSlot,
    BackendRegionStageContext,
    BackendRegionStagerRegistry,
)
from tllm.runtime.workload_execution import BackendObservableEvent
from tllm.schedule import BackendRegionBinderStage


def test_model_region_stages_the_signed_binder_at_its_signed_event() -> None:
    implementation_id = "test.backend:second-extern@1"
    region_id = "region-0000-test"
    observed: list[BackendRegionStageContext] = []

    def stage(context: BackendRegionStageContext) -> None:
        observed.append(context)
        context.slot.stage(
            {"opaque": context.implementation_id},
            lambda _plan, _inputs, _step_id: None,
            context.step_id,
        )

    slot = BackendRegionBinderSlot(implementation_id, backend_stager=stage)
    runtime = SimpleNamespace(
        event_step_id=19,
        dispatch_plan=None,
        model_region_binding=None,
        workload_execution=SimpleNamespace(
            backend_region_binders={implementation_id: slot},
            backend_region_binder_stages=lambda event: (
                (
                    BackendRegionBinderStage(
                        backend_event=BackendObservableEvent.PREPARE_INPUTS_POST,
                        region_id=region_id,
                        implementation_id=implementation_id,
                    ),
                )
                if event is BackendObservableEvent.PREPARE_INPUTS_POST
                else ()
            ),
        ),
    )
    input_batch = object()

    result = stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_POST,
        runtime=runtime,
        runner=object(),
        num_scheduled_tokens=(3,),
        input_batch=input_batch,
    )

    assert result
    assert [context.implementation_id for context in observed] == [implementation_id]
    assert observed[0].runtime is runtime
    assert observed[0].backend_event == "prepare_inputs.post"
    assert observed[0].region_id == region_id
    assert observed[0].runner is not None
    assert observed[0].input_batch is input_batch
    assert observed[0].num_scheduled_tokens == (3,)
    assert slot.plan_for_step(19) == {"opaque": implementation_id}


def test_model_region_stages_only_the_current_signed_event_once() -> None:
    post_id = "test.backend:post@1"
    pre_id = "test.backend:pre@1"
    observed: list[BackendRegionStageContext] = []

    def stage(context: BackendRegionStageContext) -> None:
        observed.append(context)
        context.slot.stage({}, lambda _plan, _inputs, _step_id: None, context.step_id)

    slots = {
        post_id: BackendRegionBinderSlot(post_id),
        pre_id: BackendRegionBinderSlot(pre_id),
    }
    stages = {
        BackendObservableEvent.PREPARE_INPUTS_POST: (
            BackendRegionBinderStage(
                backend_event=BackendObservableEvent.PREPARE_INPUTS_POST,
                region_id="region-post",
                implementation_id=post_id,
            ),
        ),
        BackendObservableEvent.EXECUTE_MODEL_PRE: (
            BackendRegionBinderStage(
                backend_event=BackendObservableEvent.EXECUTE_MODEL_PRE,
                region_id="region-pre",
                implementation_id=pre_id,
            ),
        ),
    }
    runtime = SimpleNamespace(
        event_step_id=23,
        backend_region_stager_registry=BackendRegionStagerRegistry(
            {post_id: stage, pre_id: stage}
        ),
        workload_execution=SimpleNamespace(
            backend_region_binders=slots,
            backend_region_binder_stages=lambda event: stages.get(event, ()),
        ),
    )

    assert not stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_PRE,
        runtime=runtime,
        runner=object(),
        num_scheduled_tokens=(3,),
    )
    assert [context.implementation_id for context in observed] == []
    assert stage_signed_backend_region_event(
        event=BackendObservableEvent.EXECUTE_MODEL_PRE,
        runtime=runtime,
        runner=object(),
        num_scheduled_tokens=(3,),
    )
    assert [context.implementation_id for context in observed] == [pre_id]
    assert stage_signed_backend_region_event(
        event=BackendObservableEvent.PREPARE_INPUTS_POST,
        runtime=runtime,
        runner=object(),
        num_scheduled_tokens=(3,),
    )
    assert [context.implementation_id for context in observed] == [pre_id, post_id]
    with pytest.raises(RuntimeError, match="duplicate backend Region stage"):
        stage_signed_backend_region_event(
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            runtime=runtime,
            runner=object(),
            num_scheduled_tokens=(3,),
        )


def test_model_region_rejects_a_coercible_backend_step_id() -> None:
    implementation_id = "test.backend:typed-step@1"
    slot = BackendRegionBinderSlot(implementation_id)
    runtime = SimpleNamespace(
        event_step_id=True,
        backend_region_stager_registry=BackendRegionStagerRegistry(
            {implementation_id: lambda _context: None}
        ),
        workload_execution=SimpleNamespace(
            backend_region_binders={implementation_id: slot},
            backend_region_binder_stages=lambda event: (
                (
                    BackendRegionBinderStage(
                        backend_event=BackendObservableEvent.PREPARE_INPUTS_POST,
                        region_id="region-0000-test",
                        implementation_id=implementation_id,
                    ),
                )
                if event is BackendObservableEvent.PREPARE_INPUTS_POST
                else ()
            ),
        ),
    )

    with pytest.raises(ValueError, match="step_id must be a non-negative int"):
        stage_signed_backend_region_event(
            event=BackendObservableEvent.PREPARE_INPUTS_POST,
            runtime=runtime,
            runner=object(),
            num_scheduled_tokens=None,
            step_id=True,
        )
