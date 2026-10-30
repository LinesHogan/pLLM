"""Effect buffer facts and the Python-Rust input check used by attachment planning."""

from __future__ import annotations

from dataclasses import replace
from math import prod

import torch

from pllm import _core
from pllm.contracts.backend_attachment import (
    TOKEN_CAPTURE_CAPABILITY,
    AttachmentGraphMode,
    AttachmentTaskBinding,
    AttachmentTaskRole,
    BackendAttachment,
    BackendAttachmentSpec,
    BackendPlacementProbe,
)
from pllm.contracts.codec import encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.contracts.semantic_graph import (
    AnchorId,
    SchedulerTimeline,
    SemanticGraph,
    TaskResourceFact,
    TaskSymbol,
    TokenProgramEffect,
    ValueResourceFact,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType, TensorLayoutRequest
from pllm.execution.parameter_projection import ROWS_PER_BLOCK, projection_blocks, uses_pinned_projection


def attachment_buffer_bytes(
    probe: BackendPlacementProbe,
    graph: SemanticGraph,
    layouts: TensorLayoutRequest,
    attachment: BackendAttachmentSpec,
) -> int:
    """Fund effect buffers and pinned projection scratch; excludes model scratch."""
    rows, hidden = probe.max_packed_rows, probe.hidden_size
    width = 8 if probe.tensor_dtype is TensorDType.FLOAT64 else 4
    fields = {
        field.field_id: field
        for layout in layouts.layouts
        if layout.resource == attachment.state_resource
        for field in layout.layout.fields
    }
    buffers = 1024 if attachment.overlay_bindings else 0
    pinned = uses_pinned_projection(
        in_forward=graph.updates_in_forward, graphs=probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH
    )
    if attachment.overlay_bindings and probe.device_kind is StateDeviceKind.CUDA and pinned:
        buffers += torch.backends.cuda.cublas_workspace_size() + torch.backends.cuda.cublaslt_workspace_size()
    for binding in attachment.overlay_bindings:
        buffers += rows  # Scalar prompt-end operands for the largest request set.
        first = binding.fields[0].shape
        if binding.operator is ParameterOverlayOperator.DENSE_ADD:
            outputs, inputs = first
            # Owned hidden/MLP snapshots survive forward; base weights are borrowed.
            buffers += rows * (width * (inputs + hidden + outputs) + 8)
        elif binding.operator is ParameterOverlayOperator.HEAD_ADD:
            heads, width_per_head, _ = first
            inputs = outputs = heads * width_per_head
            # Owned capture, fp32 input/correction, layout copy, cast and addition.
            buffers += rows * (6 * width * heads * width_per_head + 8)
        else:
            rank, inputs = first
            outputs = binding.fields[1].shape[0]
            buffers += rows * (width * (inputs + rank + 2 * outputs) + 8)
        if probe.causal_domain is not None and pinned:
            blocks = projection_blocks(rows, probe.causal_domain.requests)
            buffers += 24 * (blocks * (ROWS_PER_BLOCK + len(binding.fields)) + rows)
            buffers += 2 * blocks * sum(prod(field.shape) * width for field in binding.fields)
            buffers += blocks * ROWS_PER_BLOCK * 2 * width * (inputs + outputs)
    for binding in attachment.activation_bindings:
        shape = binding.fields[0].shape
        rank = shape[0] if len(shape) == 2 else 0
        buffers += rows * (width * (2 * hidden + 3 * rank) + 8)
    for task in graph.tasks:
        match task.replay:
            case TokenProgramEffect():
                buffers += 2 * rows * hidden * width
                buffers += (3 * hidden * width + 16) * len(task.replay.position_states) + 2
                if probe.causal_domain is not None and attachment.overlay_bindings:
                    requests = probe.causal_domain.requests
                    # Two phase signatures at every exact width, plus one ordinary span.
                    buffers += (requests * (requests + 1) + requests) * (
                        3 * sum(prod(field.shape) * width for field in fields.values()) + 1
                    )
            case None:
                pass
    return buffers


def derive_attachment_resources(
    probe: BackendPlacementProbe,
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    *,
    capture_memory_bytes: int = 0,
) -> SchedulerTimeline:
    """Fund the tokenwise reference plan through its task and value resource facts.

    Preserve producer task/value facts when the model reads a publication.
    The adapter keeps embeddings and output concatenation in reserved_bytes;
    the replay task pays for its own tensors and recovery lifetimes.
    """
    tasks = tuple(task for task in graph.tasks if isinstance(task.replay, TokenProgramEffect))
    if not tasks:
        return timeline
    if len(tasks) != 1:
        raise ValueError("tokenwise attachment costs require one model replay task")
    (task,) = tasks
    # Builder._attach composes the replay with its publication; a conditional input group keeps
    # all producer facts, so the complete DAG funds the worst case.
    resident = capture_memory_bytes if TOKEN_CAPTURE_CAPABILITY in probe.capabilities else 0
    workspace = probe.max_packed_rows * probe.hidden_size * 8
    existing = next((fact for fact in timeline.task_resources if fact.task == task.symbol), None)
    if existing is not None:
        task_facts = tuple(
            replace(fact, resident_bytes=max(fact.resident_bytes, resident)) if fact.task == task.symbol else fact
            for fact in timeline.task_resources
        )
    else:
        task_facts = (*timeline.task_resources, TaskResourceFact(task.symbol, workspace, resident))
    if timeline.memory_budget_bytes is not None and (
        max(timeline.reserved_bytes, default=0)
        + sum(fact.workspace_bytes + fact.resident_bytes for fact in task_facts)
        + sum(fact.bytes for fact in timeline.value_resources)
        > timeline.memory_budget_bytes
    ):
        raise ValueError("increase memory_budget_bytes to fund route workspace and captures")
    # The output lives in the original adapter's concatenation reservation;
    # the outer task runner does not allocate a second copy of this value.
    known = {(fact.task, fact.output) for fact in timeline.value_resources}
    values = (
        *timeline.value_resources,
        *tuple(
            ValueResourceFact(task.symbol, output.name, 0)
            for output in task.outputs
            if (task.symbol, output.name) not in known
        ),
    )
    return replace(
        timeline,
        task_resources=tuple(sorted(task_facts, key=lambda fact: fact.task)),
        value_resources=tuple(sorted(values, key=lambda fact: (fact.task, fact.output))),
    )


def check_backend_attachment(attachment: BackendAttachment) -> None:
    """Reject, in Rust, attachments a workload can declare but the backend cannot run."""
    _core.check_backend_attachment(encode_boundary(attachment))


def bind_attachment_tasks(
    plan_bundle: PlanBundle,
    roles: tuple[tuple[AttachmentTaskRole, TaskSymbol], ...],
) -> tuple[AttachmentTaskBinding, ...]:
    bindings: list[AttachmentTaskBinding] = []
    for role, symbol in roles:
        (physical,) = (
            task
            for task in plan_bundle.schedule.tasks
            if plan_bundle.semantic_graph.tasks[task.task_id].symbol == symbol
        )
        bindings.append(
            AttachmentTaskBinding(
                role,
                physical.task_id,
                physical.implementation_id,
                physical.anchor_id,
            )
        )
    return tuple(bindings)


def bind_backend_attachment(
    plan: PlanBundle,
    probe: BackendPlacementProbe,
    spec: BackendAttachmentSpec,
) -> BackendAttachment:
    anchors = {anchor.symbol: anchor for anchor in plan.timeline.anchors}
    sampler = anchors[spec.sampler_anchor]
    attachment = BackendAttachment(
        spec=spec,
        task_bindings=bind_attachment_tasks(plan, spec.tasks),
        sampler_anchor_id=AnchorId(sampler.ordinal),
        probe=probe,
        plan=plan,
    )
    check_backend_attachment(attachment)
    return attachment


__all__ = [
    "bind_attachment_tasks",
    "bind_backend_attachment",
    "check_backend_attachment",
]
