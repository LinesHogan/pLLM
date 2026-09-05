"""One dense-fast-weight transition; device bindings only perform storage I/O."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch

from tllm.canonical.kernels.dense_fast_weight import (
    transition_dense_fast_weight,
)
from tllm.canonical.ptts.page_bundle import PageBundleArena, PagePin, PagePublication, PageWrite
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory, RequestStateSelection, RequestStateTransitionLease,
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
class InPlaceTTTPreparedTransition:
    write: PageWrite
    result: InPlaceTTTStepBatchResult


@dataclass(frozen=True, slots=True)
class InPlaceTTTCommittedTransition:
    receipt: PagePublication
    result: InPlaceTTTStepBatchResult


InPlaceTTTTransitionState = (
    InPlaceTTTTransitionPending
    | InPlaceTTTNoTransition
    | InPlaceTTTPreparedTransition
    | InPlaceTTTCommittedTransition
)


@dataclass(slots=True)
class InPlaceTTTStepContext:
    batch: InPlaceTTTStepBatch | None
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    arena: PageBundleArena
    pin: PagePin
    transition: InPlaceTTTTransitionState

    def abort(self) -> None:
        match self.transition:
            case InPlaceTTTPreparedTransition(write=write):
                self.arena.abort(write)
                self.transition = InPlaceTTTTransitionPending()
            case InPlaceTTTCommittedTransition():
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
        batch = context.batch
        if batch is None:
            raise RuntimeError("parameter transition requires captured operands")
        match context.transition:
            case InPlaceTTTTransitionPending():
                pass
            case _:
                raise RuntimeError("parameter transition already computed")
        source_roots = tuple(context.arena.snapshot(context.pin, ref) for ref in context.selection.refs)
        stream = context.pin.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            computed = tuple(
                compute_inplace_ttt_transition(self._config, self._layout, entry, root)
                for entry, root in zip(batch.entries, source_roots, strict=True)
            )
        changed = tuple(
            index for index, entry in enumerate(batch.entries)
            if any(self._config.updates(operand.phase) for operand in entry.operands)
        )
        logical = self._directory.prepare_transition_subset(
            context.selection, context.transition_lease, changed, stream,
        ) if changed else None
        try:
            after = {} if logical is None else {
                before: after for before, after in zip(logical.before_refs, logical.after_refs, strict=True)
            }
            result = InPlaceTTTStepBatchResult(
                tuple(
                    build_inplace_ttt_step_result(
                        entry, before, after.get(before, before), values.targets,
                        self._layout.overlay_bindings, weight_changed=values.weight_changed,
                    )
                    for entry, before, values in zip(
                        batch.entries, context.selection.refs, computed, strict=True,
                    )
                )
            )
            if logical is None:
                context.transition = InPlaceTTTNoTransition(result)
                return
            context.arena.copy_roots(logical, tuple(computed[index].target_root for index in changed))
        except Exception:
            if logical is not None:
                context.arena.abort(logical)
            raise
        context.transition = InPlaceTTTPreparedTransition(logical, result)


class InPlaceTTTPublishTask:
    __slots__ = ("implementation_id",)

    def __init__(self, implementation_id: ImplementationId) -> None:
        self.implementation_id = implementation_id

    def execute(self, context: InPlaceTTTStepContext) -> None:
        match context.transition:
            case InPlaceTTTNoTransition():
                return
            case InPlaceTTTPreparedTransition(write=write, result=result):
                receipt = context.arena.publish(write)
                context.transition = InPlaceTTTCommittedTransition(receipt, result)
            case InPlaceTTTTransitionPending():
                raise RuntimeError("parameter publication requires transition output")
            case InPlaceTTTCommittedTransition():
                raise RuntimeError("parameter transition is already committed")
