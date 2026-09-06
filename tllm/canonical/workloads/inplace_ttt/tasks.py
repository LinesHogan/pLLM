"""One dense-fast-weight transition; device bindings only perform storage I/O."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch

from tllm.canonical.kernels.dense_fast_weight import (
    transition_dense_fast_weight,
)
from tllm.canonical.state_execution import StateTaskContext
from tllm.canonical.workloads.inplace_ttt.layout import InPlaceTTTStateLayout
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTConfig,
    InPlaceTTTStepBatchResult,
    InPlaceTTTStepResult,
    InPlaceTTTTargetResult,
)
from tllm.contracts.parameter_overlay import CapturedParameterOperand
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


@dataclass(slots=True)
class InPlaceTTTData:
    operands: tuple[CapturedParameterOperand, ...]
    result: InPlaceTTTStepBatchResult | None = None


@dataclass(frozen=True, slots=True)
class InPlaceTTTComputedTransition:
    target_root: tuple[torch.Tensor, ...]
    targets: tuple[InPlaceTTTTargetResult, ...]
    weight_changed: bool


def compute_inplace_ttt_transition(
    config: InPlaceTTTConfig,
    layout: InPlaceTTTStateLayout,
    operands: tuple[CapturedParameterOperand, ...],
    source_root: tuple[torch.Tensor, ...],
) -> InPlaceTTTComputedTransition:
    """Shared CPU/CUDA tensor transition; providers own storage mechanics."""

    target_root = list(source_root)
    results: list[InPlaceTTTTargetResult] = []
    weight_changed = False
    for operand in operands:
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
    __slots__ = ("_config", "_layout", "implementation_id")

    def __init__(
        self,
        config: InPlaceTTTConfig,
        layout: InPlaceTTTStateLayout,
        implementation_id: ImplementationId,
    ) -> None:
        self._config = config
        self._layout = layout
        self.implementation_id = implementation_id

    def execute(self, context: StateTaskContext[InPlaceTTTData]) -> None:
        data = context.data()
        operands = tuple(tuple(sorted(
            (operand for operand in data.operands if operand.identity == row.identity),
            key=lambda operand: operand.target,
        )) for row in context.selection.rows)
        fields = self._layout.tensor_layout.fields
        source_roots = tuple(tuple(context.arena.borrow_field(context.pin, ref, field.field_id)
            for field in fields) for ref in context.selection.refs)
        stream = context.pin.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            computed = tuple(
                compute_inplace_ttt_transition(self._config, self._layout, entry, root)
                for entry, root in zip(operands, source_roots, strict=True)
            )
        changed = tuple(
            index for index, entry in enumerate(operands)
            if any(self._config.updates(operand.phase) for operand in entry)
        )
        write = context.arena.prepare_transition(tuple(context.pin.refs[index] for index in changed), stream) if changed else None
        context.write = write
        after = {} if write is None else dict(zip(write.before_refs, write.after_refs, strict=True))
        data.result = InPlaceTTTStepBatchResult(tuple(InPlaceTTTStepResult(
            row.identity, before, after.get(before, before), values.targets, values.weight_changed,
        ) for row, before, values in zip(context.selection.rows, context.pin.refs, computed, strict=True)))
        if write is not None:
            context.arena.copy_roots(write, tuple(computed[index].target_root for index in changed))
        context.update_count = sum(target.update_count for values in computed for target in values.targets)
        context.trained_rows = sum(target.trained_rows for values in computed for target in values.targets)
