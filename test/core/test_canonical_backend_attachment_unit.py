"""One full-plan Python-Rust proof for a signed backend attachment bundle."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tllm.canonical.backend_attachment import (
    compile_backend_attachment,
    lower_backend_attachment,
    validate_backend_attachment,
    validate_backend_attachment_physical,
    validate_backend_attachment_reconstruction,
)
from tllm.canonical.scheduler import compile_schedule, lower_physical_plan
from tllm.canonical.state_representation import (
    compile_state_representation,
    lower_state_physical_plan,
)
from tllm.contracts.backend_attachment import (
    AttachmentCleanupStage,
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentKind,
    AttachmentSamplerStage,
    AttachmentTaskBinding,
    AttachmentTaskRole,
    AttachmentTopology,
    BackendAttachmentCompilerInputs,
    BackendAttachmentRequest,
    BackendPlacementProbe,
    StreamDependency,
)
from tllm.contracts.hidden_locator import (
    HiddenLayerKind,
    HiddenPhase,
    HiddenSite,
)
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.request_identity import RequestKeyField
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    ExecutionKind,
    HiddenInputSource,
    ImplementationId,
    PackedRowsInputSource,
    PackedSamplerRowsInputSource,
    SchedulerTimeline,
    SemanticEdge,
    SemanticEffect,
    SemanticGraph,
    SemanticInput,
    SemanticOutput,
    SemanticOutputKind,
    SemanticState,
    SemanticTask,
    StateEffectKind,
    StateResource,
    StateScope,
    StateSelectionInputSource,
    TaskImplementationSpec,
    TaskSymbol,
    TaskResultInputSource,
    TimelineAnchor,
    TimingConstraint,
    WorkloadId,
)
from tllm.contracts.state_ref import LayoutId
from tllm.contracts.state_representation import (
    StateDeviceKind,
    StateLayoutRequirement,
    StateProviderPlan,
    StateRepresentation,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import TensorDType


def test_rust_signs_full_backend_attachment_bundle() -> None:
    predict_implementation = ImplementationId("canonical.test:cuda_predict@1")
    train_implementation = ImplementationId("canonical.test:cuda_train@1")
    graph = SemanticGraph(
        workload_id=WorkloadId("test.attachment"),
        tasks=(
            SemanticTask(
                symbol=TaskSymbol("predict"),
                timing=TimingConstraint(
                    AnchorSymbol("model.execute"),
                    AnchorSymbol("sampler.pre_draw"),
                ),
                inputs=(
                    SemanticInput(
                        "source_hidden",
                        HiddenInputSource(
                            -1,
                            HiddenSite.BLOCK_OUTPUT,
                            HiddenPhase.DECODE,
                        ),
                    ),
                    SemanticInput(
                        "target_hidden",
                        HiddenInputSource(
                            0,
                            HiddenSite.BLOCK_OUTPUT,
                            HiddenPhase.DECODE,
                        ),
                    ),
                    SemanticInput("rows", PackedRowsInputSource()),
                    SemanticInput(
                        "sampler_rows",
                        PackedSamplerRowsInputSource(),
                    ),
                    SemanticInput(
                        "selection",
                        StateSelectionInputSource(
                            StateResource("request_distiller"),
                            HiddenPhase.DECODE,
                        ),
                    ),
                ),
                outputs=(
                    SemanticOutput("guidance", SemanticOutputKind.SAMPLER),
                    SemanticOutput("loss_batch", SemanticOutputKind.TEMPORARY),
                ),
                implementations=(
                    TaskImplementationSpec(
                        predict_implementation,
                        ExecutionKind.NATIVE_TASK,
                        ("compute/cuda@1",),
                    ),
                ),
                effects=(
                    SemanticEffect(
                        StateResource("request_distiller"),
                        StateEffectKind.READ,
                    ),
                ),
            ),
            SemanticTask(
                symbol=TaskSymbol("train"),
                timing=TimingConstraint(
                    AnchorSymbol("token.commit"),
                    AnchorSymbol("next.prepare"),
                ),
                inputs=(
                    SemanticInput(
                        "loss_batch",
                        TaskResultInputSource(TaskSymbol("predict"), "loss_batch"),
                    ),
                    SemanticInput(
                        "selection",
                        StateSelectionInputSource(
                            StateResource("request_distiller"),
                            HiddenPhase.DECODE,
                        ),
                    ),
                ),
                outputs=(),
                implementations=(
                    TaskImplementationSpec(
                        train_implementation,
                        ExecutionKind.NATIVE_TASK,
                        ("compute/cuda@1",),
                    ),
                ),
                effects=(
                    SemanticEffect(
                        StateResource("request_distiller"),
                        StateEffectKind.TRANSITION,
                    ),
                ),
            ),
        ),
        states=(
            SemanticState(
                StateResource("request_distiller"),
                StateScope.REQUEST,
                (RequestKeyField.REQUEST_ID,),
            ),
        ),
        edges=(SemanticEdge(TaskSymbol("predict"), TaskSymbol("train")),),
    )
    timeline = SchedulerTimeline(
        anchors=(
            TimelineAnchor(AnchorSymbol("model.execute"), 0),
            TimelineAnchor(AnchorSymbol("sampler.pre_draw"), 1),
            TimelineAnchor(AnchorSymbol("token.commit"), 2),
            TimelineAnchor(AnchorSymbol("next.prepare"), 3),
        ),
        capabilities=("compute/cuda@1",),
    )
    schedule_decision = compile_schedule(graph, timeline)
    schedule_physical = lower_physical_plan(
        graph,
        timeline,
        schedule_decision,
        plan_epoch=3,
    )
    state_provider = StateProviderPlan(
        resource="request_distiller",
        provider_id="canonical.cuda_dense_page@1",
        representation=StateRepresentation.DENSE_PAGE_ARENA,
        device_kind=StateDeviceKind.CUDA,
        device_ordinal=0,
        layout_id=LayoutId(0),
        layout_namespace="1" * 64,
        layout_digest="2" * 64,
        max_admitted_lineages=2,
        max_live_versions_per_lineage=3,
        root_page_count=1,
        reserved_bytes=128,
        capture_stable=False,
        capabilities=("device/cuda@1",),
    )
    state_request = StateRepresentationRequest(
        memory_budget_bytes=128,
        states=(
            StateLayoutRequirement(
                resource="request_distiller",
                layout_id=LayoutId(0),
                layout_namespace="1" * 64,
                layout_digest="2" * 64,
                max_admitted_lineages=2,
                required_capabilities=("device/cuda@1",),
            ),
        ),
        candidates=(state_provider,),
    )
    state_decision = compile_state_representation(state_request)
    state_physical = lower_state_physical_plan(
        state_request,
        state_decision,
        plan_epoch=3,
    )
    hidden_resources = (
        AttachmentHiddenResource(
            PackedHiddenResourceBinding(
                "-1",
                HiddenLayerKind.INDEX,
                HiddenSite.BLOCK_OUTPUT,
                "model.layers[-1]",
            ),
            HiddenPhase.DECODE,
        ),
        AttachmentHiddenResource(
            PackedHiddenResourceBinding(
                "0",
                HiddenLayerKind.INDEX,
                HiddenSite.BLOCK_OUTPUT,
                "model.layers[0]",
            ),
            HiddenPhase.DECODE,
        ),
    )
    capabilities = (
        "backend/vllm_mrv2@1",
        "producer/packed_hidden@1",
        "sampler/post_filter_exact@1",
    )
    probe = BackendPlacementProbe(
        adapter_id="canonical.vllm_mrv2@1",
        backend_abi_digest="a" * 64,
        runner_fingerprint="b" * 64,
        model_fingerprint="c" * 64,
        topology=AttachmentTopology.IN_PROCESS,
        graph_mode=AttachmentGraphMode.EAGER,
        device_kind=StateDeviceKind.CUDA,
        device_ordinal=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_dtype=TensorDType.FLOAT32,
        hidden_size=4,
        max_packed_rows=8,
        hidden_resources=hidden_resources,
        capabilities=capabilities,
    )
    request = BackendAttachmentRequest(
        adapter_id=probe.adapter_id,
        attachment_kind=AttachmentKind.REQUEST_SAMPLER_TRAINING,
        backend_abi_digest=probe.backend_abi_digest,
        runner_fingerprint=probe.runner_fingerprint,
        model_fingerprint=probe.model_fingerprint,
        topology=probe.topology,
        graph_mode=probe.graph_mode,
        device_kind=probe.device_kind,
        device_ordinal=probe.device_ordinal,
        tensor_parallel_size=probe.tensor_parallel_size,
        pipeline_parallel_size=probe.pipeline_parallel_size,
        tensor_dtype=probe.tensor_dtype,
        hidden_size=probe.hidden_size,
        max_packed_rows=probe.max_packed_rows,
        packed_row_schema_version=1,
        hidden_resources=hidden_resources,
        task_bindings=(
            AttachmentTaskBinding(
                AttachmentTaskRole.COMPUTE,
                schedule_physical.tasks[0].task_id,
                predict_implementation,
                AnchorId(0),
            ),
            AttachmentTaskBinding(
                AttachmentTaskRole.TRANSITION,
                schedule_physical.tasks[1].task_id,
                train_implementation,
                AnchorId(2),
            ),
        ),
        sampler_anchor_id=AnchorId(1),
        sampler_anchor_symbol="sampler.pre_draw",
        stream_dependencies=(
            StreamDependency.HIDDEN_PRODUCER_TO_COMPUTE,
            StreamDependency.COMPUTE_TO_SAMPLER,
            StreamDependency.SAMPLER_TO_TRANSITION,
            StreamDependency.TRANSITION_TO_RELEASE,
        ),
        sampler_stage=AttachmentSamplerStage.PACKED_POST_FILTER_EXACT,
        cleanup_stage=AttachmentCleanupStage.REQUEST_FINISH,
        request_key_fields=(RequestKeyField.REQUEST_ID,),
        state_resource="request_distiller",
        plan_epoch=3,
        schedule_signature=schedule_physical.schedule_signature,
        state_request_digest=state_physical.request_digest,
        state_representation_signature=state_physical.representation_signature,
        capabilities=capabilities,
    )
    inputs = BackendAttachmentCompilerInputs(
        request=request,
        probe=probe,
        graph=graph,
        timeline=timeline,
        schedule_decision=schedule_decision,
        schedule_physical=schedule_physical,
        state_request=state_request,
        state_decision=state_decision,
        state_physical=state_physical,
    )
    decision = compile_backend_attachment(inputs)
    validate_backend_attachment(inputs, decision)
    physical = lower_backend_attachment(inputs, decision)
    validate_backend_attachment_physical(inputs, decision, physical)
    validate_backend_attachment_reconstruction(
        inputs,
        decision,
        physical,
        inputs,
        physical,
    )
    assert physical.bundle_signature != physical.attachment_signature
    with pytest.raises(ValueError):
        compile_backend_attachment(
            replace(
                inputs,
                schedule_physical=replace(
                    schedule_physical,
                    tasks=(
                        replace(schedule_physical.tasks[0], anchor_id=AnchorId(2)),
                        schedule_physical.tasks[1],
                    ),
                ),
            )
        )
    graph_without_sampler_rows = replace(
        graph,
        tasks=(
            replace(
                graph.tasks[0],
                inputs=tuple(
                    task_input
                    for task_input in graph.tasks[0].inputs
                    if task_input.name != "sampler_rows"
                ),
            ),
            graph.tasks[1],
        ),
    )
    decision_without_sampler_rows = compile_schedule(
        graph_without_sampler_rows,
        timeline,
    )
    physical_without_sampler_rows = lower_physical_plan(
        graph_without_sampler_rows,
        timeline,
        decision_without_sampler_rows,
        plan_epoch=3,
    )
    with pytest.raises(ValueError, match="packed sampler semantics"):
        compile_backend_attachment(
            replace(
                inputs,
                request=replace(
                    request,
                    schedule_signature=physical_without_sampler_rows.schedule_signature,
                ),
                graph=graph_without_sampler_rows,
                schedule_decision=decision_without_sampler_rows,
                schedule_physical=physical_without_sampler_rows,
            )
        )
