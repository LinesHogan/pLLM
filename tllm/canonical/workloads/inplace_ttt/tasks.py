"""One dense-fast-weight transition; device bindings only perform storage I/O."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch

from tllm.canonical.kernels.dense_fast_weight import (
    transition_dense_fast_weight,
)
from tllm.canonical.ptts.dense_page import (
    DensePageStateProvider,
    DensePageTransitionBatch,
)
from tllm.canonical.ptts.cuda_publication import (
    CudaRequestStatePreparedPublication,
    CudaRequestStatePublicationCoordinator,
)
from tllm.canonical.ptts.dense_page_cuda import CudaDensePageStateProvider
from tllm.canonical.ptts.dense_page_cuda_types import CudaDensePageVersionPin
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateSelection,
    RequestStateTransitionBatch,
    RequestStateTransitionLease,
)
from tllm.canonical.ptts.publication import (
    RequestStatePublicationCoordinator,
    RequestStatePublicationReceipt,
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
from tllm.contracts.parameter_overlay import (
    ParameterOverlayFieldBinding,
    ParameterOverlayPublication,
)
from tllm.contracts.state_ref import StateRef
from tllm.contracts.semantic_graph import ImplementationId


INPLACE_TTT_TRANSITION_IMPLEMENTATION = ImplementationId(
    "canonical.inplace_ttt:cpu_transition@1"
)
INPLACE_TTT_PUBLISH_IMPLEMENTATION = ImplementationId(
    "canonical.state:publish_parameter_overlay@1"
)
INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION = ImplementationId(
    "canonical.inplace_ttt:cuda_transition@1"
)
INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION = ImplementationId(
    "canonical.state:cuda_publish_parameter_overlay@1"
)


@dataclass(frozen=True, slots=True)
class InPlaceTTTTransitionPending:
    pass


@dataclass(frozen=True, slots=True)
class InPlaceTTTNoTransition:
    result: InPlaceTTTStepBatchResult


@dataclass(frozen=True, slots=True)
class _CpuWrite:
    publication: RequestStatePublicationCoordinator
    physical: DensePageTransitionBatch


@dataclass(frozen=True, slots=True)
class _CudaWrite:
    publication: CudaRequestStatePublicationCoordinator
    token: CudaRequestStatePreparedPublication


@dataclass(frozen=True, slots=True)
class InPlaceTTTPreparedTransition:
    logical: RequestStateTransitionBatch
    write: _CpuWrite | _CudaWrite
    result: InPlaceTTTStepBatchResult


@dataclass(frozen=True, slots=True)
class InPlaceTTTSubmitting:
    """CUDA submission may have started; rollback is no longer allowed."""


@dataclass(frozen=True, slots=True)
class InPlaceTTTCommittedTransition:
    receipt: RequestStatePublicationReceipt
    result: InPlaceTTTStepBatchResult


InPlaceTTTTransitionState = (
    InPlaceTTTTransitionPending
    | InPlaceTTTNoTransition
    | InPlaceTTTPreparedTransition
    | InPlaceTTTSubmitting
    | InPlaceTTTCommittedTransition
)


@dataclass(frozen=True, slots=True)
class InPlaceTTTCpuStorage:
    provider: DensePageStateProvider
    publication: RequestStatePublicationCoordinator
    stream: None = None

    def read(
        self, layout: InPlaceTTTStateLayout, refs: tuple[StateRef, ...],
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        return tuple(self.provider.snapshot(ref) for ref in refs)

    def prepare(
        self, logical: RequestStateTransitionBatch,
        roots: tuple[tuple[torch.Tensor, ...], ...],
    ) -> _CpuWrite:
        physical = self.provider.prepare_transition_batch(
            logical.before_refs, logical.after_refs,
        )
        # CPU compute rolls back its reservation on failure.
        self.provider.compute_full_root_transition(physical, roots)
        return _CpuWrite(self.publication, physical)


@dataclass(frozen=True, slots=True)
class InPlaceTTTCudaStorage:
    provider: CudaDensePageStateProvider
    publication: CudaRequestStatePublicationCoordinator
    pin: CudaDensePageVersionPin
    stream: torch.cuda.Stream

    def read(
        self, layout: InPlaceTTTStateLayout, refs: tuple[StateRef, ...],
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        fields = tuple(
            self.provider.pinned_field_rows(self.pin, field.field_id, self.stream)
            for field in layout.tensor_layout.fields
        )
        return tuple(tuple(rows[index] for rows in fields) for index in range(len(refs)))

    def prepare(
        self, logical: RequestStateTransitionBatch,
        roots: tuple[tuple[torch.Tensor, ...], ...],
    ) -> _CudaWrite:
        physical = self.provider.prepare_transition_batch(
            logical.before_refs, logical.after_refs, self.stream,
        )
        try:
            token = self.publication.prepare_full_root(
                logical, physical, roots, self.stream,
            )
        except Exception:
            self.provider.abort_unsubmitted_transition(physical)
            raise
        return _CudaWrite(self.publication, token)


@dataclass(slots=True)
class InPlaceTTTStepContext:
    batch: InPlaceTTTStepBatch
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    storage: InPlaceTTTCpuStorage | InPlaceTTTCudaStorage
    transition: InPlaceTTTTransitionState

    def abort(self) -> None:
        match self.transition:
            case InPlaceTTTPreparedTransition(logical=logical, write=write):
                match write:
                    case _CpuWrite(publication=publication, physical=physical):
                        publication.abort(logical, physical)
                    case _CudaWrite(publication=publication, token=token):
                        publication.abort_prepared(token)
                self.transition = InPlaceTTTTransitionPending()
            case InPlaceTTTSubmitting() | InPlaceTTTCommittedTransition():
                raise RuntimeError("cannot abort a submitted parameter transition")
            case InPlaceTTTTransitionPending() | InPlaceTTTNoTransition():
                pass

    def result(self) -> InPlaceTTTStepBatchResult:
        match self.transition:
            case InPlaceTTTNoTransition(result=result) | InPlaceTTTCommittedTransition(result=result):
                return result
            case _:
                raise RuntimeError("signed dense overlay graph did not publish")


@dataclass(frozen=True, slots=True)
class InPlaceTTTComputedTransition:
    target_root: tuple[torch.Tensor, ...]
    targets: tuple[InPlaceTTTTargetResult, ...]
    weight_changed: bool


def build_inplace_ttt_step_result(
    step: InPlaceTTTStepInput,
    before_ref: StateRef,
    after_ref: StateRef,
    targets: tuple[InPlaceTTTTargetResult, ...],
    overlay_bindings: tuple[ParameterOverlayFieldBinding, ...],
    *,
    weight_changed: bool,
) -> InPlaceTTTStepResult:
    return InPlaceTTTStepResult(
        step.identity,
        before_ref,
        after_ref,
        targets,
        ParameterOverlayPublication(
            step.identity,
            after_ref,
            overlay_bindings,
        )
        if weight_changed
        else None,
    )


def compute_inplace_ttt_transition(
    config: InPlaceTTTConfig,
    layout: InPlaceTTTStateLayout,
    step: InPlaceTTTStepInput,
    source_root: tuple[torch.Tensor, ...],
) -> InPlaceTTTComputedTransition:
    """Shared CPU/CUDA tensor transition; providers own storage mechanics."""

    target_root = list(source_root)
    results: list[InPlaceTTTTargetResult] = []
    weight_changed = False
    for operand in step.operands:
        fields = layout.fields_for(operand.target)
        delta = source_root[int(fields.delta_weight)]
        if config.updates(operand.phase):
            transition = transition_dense_fast_weight(
                base_weight=operand.base_weight,
                delta_weight=delta,
                tail_input=source_root[int(fields.tail_input)],
                tail_target=source_root[int(fields.tail_target)],
                tail_length=int(source_root[int(fields.tail_length)].item()),
                hidden=operand.hidden,
                target=operand.desired,
                chunk_size=config.chunk_size,
                learning_rate=config.learning_rate,
            )
            target_root[int(fields.delta_weight)] = transition.delta_weight
            target_root[int(fields.tail_input)] = transition.tail_input
            target_root[int(fields.tail_target)] = transition.tail_target
            next_tail_length = source_root[int(fields.tail_length)].clone()
            next_tail_length.fill_(transition.tail_length)
            target_root[int(fields.tail_length)] = next_tail_length
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
                    operand.hidden @ (operand.base_weight + delta).transpose(0, 1),
                    0,
                    0,
                    int(source_root[int(fields.tail_length)].item()),
                )
            )
    return InPlaceTTTComputedTransition(
        tuple(target_root), tuple(results), weight_changed
    )


class InPlaceTTTTransitionTask:
    __slots__ = ("_config", "_directory", "_layout", "implementation_id")

    def __init__(
        self,
        config: InPlaceTTTConfig,
        layout: InPlaceTTTStateLayout,
        directory: RequestStateDirectory,
        implementation_id: ImplementationId,
    ) -> None:
        self._config = config
        self._layout = layout
        self._directory = directory
        self.implementation_id = implementation_id

    def execute(self, context: InPlaceTTTStepContext) -> None:
        match context.transition:
            case InPlaceTTTTransitionPending():
                pass
            case _:
                raise RuntimeError("parameter transition already computed")
        source_roots = context.storage.read(self._layout, context.selection.refs)
        stream = context.storage.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            computed = tuple(
                compute_inplace_ttt_transition(self._config, self._layout, entry, root)
                for entry, root in zip(context.batch.entries, source_roots, strict=True)
            )
        changed = tuple(
            index for index, entry in enumerate(context.batch.entries)
            if any(self._config.updates(operand.phase) for operand in entry.operands)
        )
        logical = self._directory.prepare_transition_subset(
            context.selection, context.transition_lease, changed,
        ) if changed else None
        try:
            after = {} if logical is None else {
                transition.before: transition.after for transition in logical.transitions
            }
            result = InPlaceTTTStepBatchResult(
                tuple(
                    build_inplace_ttt_step_result(
                        entry, before, after.get(before, before), values.targets,
                        self._layout.overlay_bindings, weight_changed=values.weight_changed,
                    )
                    for entry, before, values in zip(
                        context.batch.entries, context.selection.refs, computed, strict=True,
                    )
                )
            )
            if logical is None:
                context.transition = InPlaceTTTNoTransition(result)
                return
            write = context.storage.prepare(
                logical, tuple(computed[index].target_root for index in changed),
            )
        except Exception:
            if logical is not None:
                self._directory.abort_transition(logical)
            raise
        context.transition = InPlaceTTTPreparedTransition(logical, write, result)


class InPlaceTTTPublishTask:
    __slots__ = ("implementation_id",)

    def __init__(self, implementation_id: ImplementationId) -> None:
        self.implementation_id = implementation_id

    def execute(self, context: InPlaceTTTStepContext) -> None:
        match context.transition:
            case InPlaceTTTNoTransition():
                return
            case InPlaceTTTPreparedTransition(logical=logical, write=write, result=result):
                match write:
                    case _CpuWrite(publication=publication, physical=physical):
                        receipt = publication.publish(logical, physical)
                    case _CudaWrite(publication=publication, token=token):
                        context.transition = InPlaceTTTSubmitting()
                        receipt = publication.commit_full_root(token)
                context.transition = InPlaceTTTCommittedTransition(receipt, result)
            case InPlaceTTTTransitionPending():
                raise RuntimeError("parameter publication requires transition output")
            case InPlaceTTTSubmitting() | InPlaceTTTCommittedTransition():
                raise RuntimeError("parameter transition is already committed")
