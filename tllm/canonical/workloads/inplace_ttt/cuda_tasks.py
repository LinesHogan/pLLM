"""Batched CUDA dense-fast-weight tasks selected by the canonical scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.ptts.cuda_publication import (
    CudaRequestStatePreparedPublication,
    CudaRequestStatePublicationCoordinator,
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
from tllm.canonical.workloads.inplace_ttt.tasks import (
    InPlaceTTTComputedTransition,
    compute_inplace_ttt_transition,
)
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTConfig,
    InPlaceTTTStepBatch,
    InPlaceTTTStepBatchResult,
    InPlaceTTTStepResult,
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
    result: InPlaceTTTStepBatchResult


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTPreparedTransition:
    logical: RequestStateTransitionBatch
    physical: CudaDensePageTransition
    target_roots: tuple[tuple[torch.Tensor, ...], ...]
    entries: tuple[InPlaceTTTComputedTransition, ...]


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTValidatedTransition:
    token: CudaRequestStatePreparedPublication


@dataclass(frozen=True, slots=True)
class CudaInPlaceTTTCommittedTransition:
    receipt: CudaRequestStatePublicationReceipt


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
    result: InPlaceTTTStepBatchResult


CudaInPlaceTTTPublicationState = (
    CudaInPlaceTTTPublicationPending | CudaInPlaceTTTPublished
)


@dataclass(slots=True)
class CudaInPlaceTTTStepContext:
    batch: InPlaceTTTStepBatch
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
        field_rows = tuple(
            self._provider.pinned_field_rows(
                context.state_pin,
                field.field_id,
                context.stream,
            )
            for field in self._layout.tensor_layout.fields
        )
        source_roots = tuple(
            tuple(rows[state_index] for rows in field_rows)
            for state_index in range(len(context.selection.refs))
        )
        changed_indices = tuple(
            index
            for index, entry in enumerate(context.batch.entries)
            if any(self._config.updates(operand.phase) for operand in entry.operands)
        )
        if not changed_indices:
            with torch.cuda.stream(context.stream):
                entries = tuple(
                    compute_inplace_ttt_transition(
                        self._config,
                        self._layout,
                        entry,
                        source_roots[index],
                    )
                    for index, entry in enumerate(context.batch.entries)
                )
            context.transition = CudaInPlaceTTTNoTransition(
                InPlaceTTTStepBatchResult(
                    tuple(
                        InPlaceTTTStepResult(
                            entry.identity,
                            context.selection.refs[index],
                            context.selection.refs[index],
                            computed.targets,
                            None,
                        )
                        for index, (entry, computed) in enumerate(
                            zip(context.batch.entries, entries, strict=True)
                        )
                    )
                )
            )
            return
        logical = self._directory.prepare_transition_subset(
            context.selection,
            context.transition_lease,
            changed_indices,
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
        try:
            with torch.cuda.stream(context.stream):
                computed = tuple(
                    compute_inplace_ttt_transition(
                        self._config,
                        self._layout,
                        entry,
                        source_roots[index],
                    )
                    for index, entry in enumerate(context.batch.entries)
                )
        except Exception:
            self._publication.abort_reservation(logical, physical)
            raise
        context.transition = CudaInPlaceTTTPreparedTransition(
            logical,
            physical,
            tuple(computed[index].target_root for index in changed_indices),
            computed,
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
        match context.transition:
            case CudaInPlaceTTTNoTransition(result=result):
                context.publication = CudaInPlaceTTTPublished(result)
            case CudaInPlaceTTTPreparedTransition(
                logical=logical,
                physical=physical,
                target_roots=target_roots,
                entries=entries,
            ):
                after_by_before = {
                    transition.before: transition.after
                    for transition in logical.transitions
                }
                results = tuple(
                    InPlaceTTTStepResult(
                        entry.identity,
                        context.selection.refs[index],
                        after_by_before.get(
                            context.selection.refs[index],
                            context.selection.refs[index],
                        ),
                        computed.targets,
                        ParameterOverlayPublication(
                            entry.identity,
                            after_by_before[context.selection.refs[index]],
                            self._layout.overlay_bindings,
                        )
                        if computed.weight_changed
                        else None,
                    )
                    for index, (entry, computed) in enumerate(
                        zip(context.batch.entries, entries, strict=True)
                    )
                )
                result = InPlaceTTTStepBatchResult(results)
                prepared = self._publication.prepare_full_root(
                    logical,
                    physical,
                    target_roots,
                    context.stream,
                )
                context.transition = CudaInPlaceTTTValidatedTransition(prepared)
                receipt = self._publication.commit_full_root(prepared)
                context.transition = CudaInPlaceTTTCommittedTransition(receipt)
                context.publication = CudaInPlaceTTTPublished(result)
            case CudaInPlaceTTTTransitionPending():
                raise RuntimeError("CUDA publication requires transition output")
            case CudaInPlaceTTTValidatedTransition():
                raise RuntimeError("CUDA transition submission is already validated")
            case CudaInPlaceTTTCommittedTransition():
                raise RuntimeError("CUDA transition is already committed")


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
