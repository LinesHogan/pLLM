"""CUDA dense-fast-weight tasks selected by the canonical scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.kernels.dense_fast_weight import (
    transition_dense_fast_weight,
)
from tllm.canonical.ptts.cuda_publication import (
    CudaRequestStatePublicationCoordinator,
    CudaRequestStatePreparedPublication,
    CudaRequestStatePublicationReceipt,
)
from tllm.canonical.ptts.dense_page_cuda import CudaDensePageStateProvider
from tllm.canonical.ptts.dense_page_cuda_types import (
    CudaDensePageTransition,
    CudaDensePageVersionPin,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateSelection,
    RequestStateTransitionBatch,
    RequestStateTransitionLease,
)
from tllm.canonical.workloads.inplace_ttt.layout import InPlaceTTTStateLayout
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTConfig,
    InPlaceTTTStepInput,
    InPlaceTTTStepResult,
    InPlaceTTTTargetResult,
)
from tllm.contracts.parameter_overlay import ParameterOverlayPublication
from tllm.contracts.semantic_graph import ImplementationId


INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION = ImplementationId(
    "canonical.inplace_ttt:cuda_transition@1"
)
INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION = ImplementationId(
    "canonical.state:cuda_publish_parameter_overlay@1"
)


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTTransitionPending:
    pass


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTNoTransition:
    targets: tuple[InPlaceTTTTargetResult, ...]


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTPreparedTransition:
    logical: RequestStateTransitionBatch
    physical: CudaDensePageTransition
    target_root: tuple[torch.Tensor, ...]
    targets: tuple[InPlaceTTTTargetResult, ...]
    weight_changed: bool


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTCommittedTransition:
    receipt: CudaRequestStatePublicationReceipt


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTValidatedTransition:
    token: CudaRequestStatePreparedPublication


CudaInPlaceTTTTransitionState = (
    CudaInPlaceTTTTransitionPending
    | CudaInPlaceTTTNoTransition
    | CudaInPlaceTTTPreparedTransition
    | CudaInPlaceTTTValidatedTransition
    | CudaInPlaceTTTCommittedTransition
)


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTPublicationPending:
    pass


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTPublished:
    result: InPlaceTTTStepResult


CudaInPlaceTTTPublicationState = (
    CudaInPlaceTTTPublicationPending | CudaInPlaceTTTPublished
)


@dataclass(slots=True)
class CudaInPlaceTTTStepContext:
    step: InPlaceTTTStepInput
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    state_pin: CudaDensePageVersionPin
    stream: torch.cuda.Stream
    transition: CudaInPlaceTTTTransitionState
    publication: CudaInPlaceTTTPublicationState


class CudaInPlaceTTTTransitionTask:
    __slots__ = ("_config", "_directory", "_layout", "_provider", "_publication")

    def __init__(
        self,
        config: InPlaceTTTConfig,
        layout: InPlaceTTTStateLayout,
        directory: RequestStateDirectory,
        provider: CudaDensePageStateProvider,
        publication: CudaRequestStatePublicationCoordinator,
    ) -> None:
        self._config = config
        self._layout = layout
        self._directory = directory
        self._provider = provider
        self._publication = publication

    @property
    def implementation_id(self) -> ImplementationId:
        return INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION

    def execute(self, context: CudaInPlaceTTTStepContext) -> None:
        (_source_ref,) = context.selection.refs
        source_root = tuple(
            self._provider.pinned_field_rows(
                context.state_pin,
                field.field_id,
                context.stream,
            )[0]
            for field in self._layout.tensor_layout.fields
        )
        if not any(self._config.updates(operand.phase) for operand in context.step.operands):
            with torch.cuda.stream(context.stream):
                context.transition = CudaInPlaceTTTNoTransition(
                    tuple(
                        InPlaceTTTTargetResult(
                            operand.target,
                            operand.hidden
                            @ (
                                operand.base_weight
                                + source_root[
                                    int(
                                        self._layout.fields_for(
                                            operand.target
                                        ).delta_weight
                                    )
                                ]
                            ).transpose(0, 1),
                            0,
                            0,
                            int(
                                source_root[
                                    int(
                                        self._layout.fields_for(
                                            operand.target
                                        ).tail_length
                                    )
                                ].item()
                            ),
                        )
                        for operand in context.step.operands
                    )
                )
            return
        logical = self._directory.prepare_transition(
            context.selection,
            context.transition_lease,
        )
        try:
            physical = self._provider.prepare_transition_batch(
                logical.before_refs,
                logical.after_refs,
                context.stream,
            )
        except Exception:
            self._directory.abort_transition(logical)
            raise
        target_root = list(source_root)
        results: list[InPlaceTTTTargetResult] = []
        weight_changed = False
        try:
            with torch.cuda.stream(context.stream):
                for operand in context.step.operands:
                    fields = self._layout.fields_for(operand.target)
                    delta = source_root[int(fields.delta_weight)]
                    if self._config.updates(operand.phase):
                        transition = transition_dense_fast_weight(
                            base_weight=operand.base_weight,
                            delta_weight=delta,
                            tail_input=source_root[int(fields.tail_input)],
                            tail_target=source_root[int(fields.tail_target)],
                            tail_length=int(
                                source_root[int(fields.tail_length)].item()
                            ),
                            hidden=operand.hidden,
                            target=operand.desired,
                            chunk_size=self._config.chunk_size,
                            learning_rate=self._config.learning_rate,
                        )
                        target_root[int(fields.delta_weight)] = (
                            transition.delta_weight
                        )
                        target_root[int(fields.tail_input)] = transition.tail_input
                        target_root[int(fields.tail_target)] = transition.tail_target
                        next_tail_length = source_root[
                            int(fields.tail_length)
                        ].clone()
                        next_tail_length.fill_(transition.tail_length)
                        target_root[int(fields.tail_length)] = next_tail_length
                        weight_changed = weight_changed or bool(
                            transition.update_count
                        )
                        results.append(
                            InPlaceTTTTargetResult(
                                operand.target,
                                transition.output,
                                transition.update_count,
                                transition.trained_rows,
                                transition.tail_length,
                            )
                        )
                    else:
                        results.append(
                            InPlaceTTTTargetResult(
                                operand.target,
                                operand.hidden
                                @ (operand.base_weight + delta).transpose(0, 1),
                                0,
                                0,
                                int(
                                    source_root[int(fields.tail_length)].item()
                                ),
                            )
                        )
        except Exception:
            self._publication.abort_reservation(logical, physical)
            raise
        context.transition = CudaInPlaceTTTPreparedTransition(
            logical,
            physical,
            tuple(target_root),
            tuple(results),
            weight_changed,
        )


class CudaInPlaceTTTPublishTask:
    __slots__ = ("_layout", "_publication")

    def __init__(
        self,
        layout: InPlaceTTTStateLayout,
        publication: CudaRequestStatePublicationCoordinator,
    ) -> None:
        self._layout = layout
        self._publication = publication

    @property
    def implementation_id(self) -> ImplementationId:
        return INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION

    def execute(self, context: CudaInPlaceTTTStepContext) -> None:
        (before_ref,) = context.selection.refs
        match context.transition:
            case CudaInPlaceTTTNoTransition(targets=targets):
                context.publication = CudaInPlaceTTTPublished(
                    InPlaceTTTStepResult(
                        context.step.identity,
                        before_ref,
                        before_ref,
                        targets,
                        None,
                    )
                )
            case CudaInPlaceTTTPreparedTransition(
                logical=logical,
                physical=physical,
                target_root=target_root,
                targets=targets,
                weight_changed=weight_changed,
            ):
                (after_ref,) = logical.after_refs
                overlay = (
                    None
                    if not weight_changed
                    else ParameterOverlayPublication(
                        context.step.identity,
                        after_ref,
                        self._layout.overlay_bindings,
                    )
                )
                result = InPlaceTTTStepResult(
                    context.step.identity,
                    before_ref,
                    after_ref,
                    targets,
                    overlay,
                )
                prepared = self._publication.prepare_full_root(
                    logical,
                    physical,
                    (target_root,),
                    context.stream,
                )
                context.transition = CudaInPlaceTTTValidatedTransition(prepared)
                receipt = self._publication.commit_full_root(prepared)
                context.transition = CudaInPlaceTTTCommittedTransition(receipt)
                context.publication = CudaInPlaceTTTPublished(result)
            case CudaInPlaceTTTTransitionPending():
                raise RuntimeError("CUDA publication requires transition output")
            case CudaInPlaceTTTCommittedTransition():
                raise RuntimeError("CUDA transition is already committed")
            case CudaInPlaceTTTValidatedTransition():
                raise RuntimeError("CUDA transition submission is already validated")


__all__ = [
    "CudaInPlaceTTTCommittedTransition",
    "CudaInPlaceTTTNoTransition",
    "CudaInPlaceTTTPreparedTransition",
    "CudaInPlaceTTTPublicationPending",
    "CudaInPlaceTTTPublishTask",
    "CudaInPlaceTTTPublished",
    "CudaInPlaceTTTStepContext",
    "CudaInPlaceTTTTransitionPending",
    "CudaInPlaceTTTTransitionTask",
    "CudaInPlaceTTTValidatedTransition",
    "INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION",
    "INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION",
]
