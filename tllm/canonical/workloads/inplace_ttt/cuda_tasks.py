"""Batched CUDA dense-fast-weight tasks selected by the canonical scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.kernels.dense_fast_weight import transition_dense_fast_weight
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
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTConfig,
    InPlaceTTTStepBatch,
    InPlaceTTTStepBatchResult,
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
class _ComputedEntry:
    state_index: int
    targets: tuple[InPlaceTTTTargetResult, ...]
    weight_changed: bool


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
    entries: tuple[_ComputedEntry, ...]


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
            entries = tuple(
                self._compute_entry(
                    index,
                    entry,
                    source_roots[index],
                    context.stream,
                )[0]
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
            computed_with_roots = tuple(
                self._compute_entry(
                    index,
                    entry,
                    source_roots[index],
                    context.stream,
                )
                for index, entry in enumerate(context.batch.entries)
            )
        except Exception:
            self._publication.abort_reservation(logical, physical)
            raise
        computed = tuple(item[0] for item in computed_with_roots)
        roots_by_index = {
            item[0].state_index: item[1] for item in computed_with_roots
        }
        context.transition = CudaInPlaceTTTPreparedTransition(
            logical,
            physical,
            tuple(roots_by_index[index] for index in changed_indices),
            computed,
        )

    def _compute_entry(
        self,
        state_index: int,
        entry: InPlaceTTTStepInput,
        source_root: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream,
    ) -> tuple[_ComputedEntry, tuple[torch.Tensor, ...]]:
        target_root = list(source_root)
        results: list[InPlaceTTTTargetResult] = []
        weight_changed = False
        with torch.cuda.stream(stream):
            for operand in entry.operands:
                fields = self._layout.fields_for(operand.target)
                delta = source_root[int(fields.delta_weight)]
                if self._config.updates(operand.phase):
                    transition = transition_dense_fast_weight(
                        base_weight=operand.base_weight,
                        delta_weight=delta,
                        tail_input=source_root[int(fields.tail_input)],
                        tail_target=source_root[int(fields.tail_target)],
                        tail_length=int(source_root[int(fields.tail_length)].item()),
                        hidden=operand.hidden,
                        target=operand.desired,
                        chunk_size=self._config.chunk_size,
                        learning_rate=self._config.learning_rate,
                    )
                    target_root[int(fields.delta_weight)] = transition.delta_weight
                    target_root[int(fields.tail_input)] = transition.tail_input
                    target_root[int(fields.tail_target)] = transition.tail_target
                    tail_length = source_root[int(fields.tail_length)].clone()
                    tail_length.fill_(transition.tail_length)
                    target_root[int(fields.tail_length)] = tail_length
                    weight_changed = weight_changed or bool(transition.update_count)
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
                            int(source_root[int(fields.tail_length)].item()),
                        )
                    )
        return (
            _ComputedEntry(state_index, tuple(results), weight_changed),
            tuple(target_root),
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
