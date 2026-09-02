"""CPU dense-fast-weight tasks selected by the canonical scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.kernels.dense_fast_weight import (
    transition_dense_fast_weight,
)
from tllm.canonical.ptts.dense_page import (
    DensePageStateProvider,
    DensePageTransitionBatch,
)
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


@dataclass(frozen=True, slots=True)
class InPlaceTTTTransitionPending:
    pass


@dataclass(frozen=True, slots=True)
class InPlaceTTTNoTransition:
    targets: tuple[InPlaceTTTTargetResult, ...]


@dataclass(frozen=True, slots=True)
class InPlaceTTTPreparedTransition:
    logical: RequestStateTransitionBatch
    physical: DensePageTransitionBatch
    targets: tuple[InPlaceTTTTargetResult, ...]
    weight_changed: bool


@dataclass(frozen=True, slots=True)
class InPlaceTTTCommittedTransition:
    receipt: RequestStatePublicationReceipt


InPlaceTTTTransitionState = (
    InPlaceTTTTransitionPending
    | InPlaceTTTNoTransition
    | InPlaceTTTPreparedTransition
    | InPlaceTTTCommittedTransition
)


@dataclass(frozen=True, slots=True)
class InPlaceTTTPublicationPending:
    pass


@dataclass(frozen=True, slots=True)
class InPlaceTTTPublished:
    result: InPlaceTTTStepResult


InPlaceTTTPublicationState = InPlaceTTTPublicationPending | InPlaceTTTPublished


@dataclass(slots=True)
class InPlaceTTTStepContext:
    step: InPlaceTTTStepInput
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    transition: InPlaceTTTTransitionState
    publication: InPlaceTTTPublicationState


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
    __slots__ = (
        "_config",
        "_directory",
        "_layout",
        "_provider",
        "_publication",
    )

    def __init__(
        self,
        config: InPlaceTTTConfig,
        layout: InPlaceTTTStateLayout,
        directory: RequestStateDirectory,
        provider: DensePageStateProvider,
        publication: RequestStatePublicationCoordinator,
    ) -> None:
        self._config = config
        self._layout = layout
        self._directory = directory
        self._provider = provider
        self._publication = publication

    @property
    def implementation_id(self) -> ImplementationId:
        return INPLACE_TTT_TRANSITION_IMPLEMENTATION

    def execute(self, context: InPlaceTTTStepContext) -> None:
        (source_ref,) = context.selection.refs
        source_root = self._provider.snapshot(source_ref)
        if not any(
            self._config.updates(operand.phase) for operand in context.step.operands
        ):
            computed = compute_inplace_ttt_transition(
                self._config,
                self._layout,
                context.step,
                source_root,
            )
            context.transition = InPlaceTTTNoTransition(computed.targets)
            return
        logical = self._directory.prepare_transition(
            context.selection,
            context.transition_lease,
        )
        try:
            physical = self._provider.prepare_transition_batch(
                logical.before_refs,
                logical.after_refs,
            )
        except Exception:
            self._directory.abort_transition(logical)
            raise
        try:
            computed = compute_inplace_ttt_transition(
                self._config,
                self._layout,
                context.step,
                source_root,
            )
        except Exception:
            self._publication.abort(logical, physical)
            raise
        try:
            self._provider.compute_full_root_transition(
                physical,
                (computed.target_root,),
            )
        except Exception:
            self._directory.abort_transition(logical)
            raise
        context.transition = InPlaceTTTPreparedTransition(
            logical,
            physical,
            computed.targets,
            computed.weight_changed,
        )


class InPlaceTTTPublishTask:
    __slots__ = ("_layout", "_publication")

    def __init__(
        self,
        layout: InPlaceTTTStateLayout,
        publication: RequestStatePublicationCoordinator,
    ) -> None:
        self._layout = layout
        self._publication = publication

    @property
    def implementation_id(self) -> ImplementationId:
        return INPLACE_TTT_PUBLISH_IMPLEMENTATION

    def execute(self, context: InPlaceTTTStepContext) -> None:
        (before_ref,) = context.selection.refs
        match context.transition:
            case InPlaceTTTNoTransition(targets=targets):
                context.publication = InPlaceTTTPublished(
                    build_inplace_ttt_step_result(
                        context.step,
                        before_ref,
                        before_ref,
                        targets,
                        self._layout.overlay_bindings,
                        weight_changed=False,
                    )
                )
            case InPlaceTTTPreparedTransition(
                logical=logical,
                physical=physical,
                targets=targets,
                weight_changed=weight_changed,
            ):
                (after_ref,) = logical.after_refs
                result = build_inplace_ttt_step_result(
                    context.step,
                    before_ref,
                    after_ref,
                    targets,
                    self._layout.overlay_bindings,
                    weight_changed=weight_changed,
                )
                receipt = self._publication.publish(logical, physical)
                context.transition = InPlaceTTTCommittedTransition(receipt)
                context.publication = InPlaceTTTPublished(result)
            case InPlaceTTTTransitionPending():
                raise RuntimeError("parameter publication requires transition output")
            case InPlaceTTTCommittedTransition():
                raise RuntimeError("parameter transition is already committed")


__all__ = [
    "INPLACE_TTT_PUBLISH_IMPLEMENTATION",
    "INPLACE_TTT_TRANSITION_IMPLEMENTATION",
    "InPlaceTTTComputedTransition",
    "InPlaceTTTCommittedTransition",
    "InPlaceTTTNoTransition",
    "InPlaceTTTPreparedTransition",
    "InPlaceTTTPublicationPending",
    "InPlaceTTTPublishTask",
    "InPlaceTTTPublished",
    "InPlaceTTTStepContext",
    "InPlaceTTTTransitionPending",
    "InPlaceTTTTransitionTask",
    "build_inplace_ttt_step_result",
    "compute_inplace_ttt_transition",
]
