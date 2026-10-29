"""Sequence batch ports and ordinary training transitions over worker-bound causal forwards."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import torch

from pllm.contracts.auxiliary import CausalTrainBatch
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding
from pllm.contracts.semantic_graph import MathContract, SemanticInputSource, StateEffectKind, StateSelectionInputSource
from pllm.execution.declaration import Call, State, When, native
from pllm.execution.tensor_frame import TensorFrame, Tensors, TensorUpdate
from pllm.execution.tensor_program import TaskOutput, TensorBatch, autograd_loss

if TYPE_CHECKING:
    from pllm.execution.tensor_binding import TensorBinding


class CausalForward(Protocol):
    """A bound decoder forward; factors are borrowed from the current invocation."""

    def __call__(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None, factors: tuple[torch.Tensor, ...] | None
    ) -> torch.Tensor: ...


CausalBinder = Callable[[tuple[ParameterOverlayFieldBinding, ...]], CausalForward]


@dataclass(frozen=True)
class CausalCall:
    """The method's callable model argument and its overlay-free base forward."""

    forward: CausalForward | None
    factors: tuple[torch.Tensor, ...]

    def __call__(self, ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert self.forward is not None, "bind the causal port in the model-owning worker"
        return self.forward(ids, attention_mask, self.factors)

    def base(self, ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate the frozen base without the declared overlay."""
        assert self.forward is not None
        return self.forward(ids, attention_mask, None)


@dataclass(frozen=True)
class CausalPort:
    """A signed batch/state read and its invocation-local argument adapter."""

    source: SemanticInputSource
    adapt: Callable[[CausalTrainBatch[torch.Tensor], Tensors], torch.Tensor | CausalCall | None]


def declare_causal_train(
    b: TensorBinding,
    name: str,
    state: State,
    *,
    ports: Mapping[str, CausalPort],
    function: Callable[..., torch.Tensor | TensorUpdate],
    identity: str,
    when: When,
) -> Call:
    """Bind sequence inputs, explicit parameter gradients and the shared AdamW transition."""
    if state.optimizer is None or state.declaration.key_fields or when not in (When.BEFORE_MODEL, When.AFTER_SAMPLE):
        raise ValueError("causal training needs an instance AdamW state and BEFORE_MODEL or AFTER_SAMPLE")
    if any(
        isinstance(port.source, StateSelectionInputSource) and port.source.resource != state.declaration.resource
        for port in ports.values()
    ):
        raise ValueError("causal training must read the overlay's own optimizer state")
    limit = next((item for item in b.resources if item.task == name), None)
    if limit is not None and (
        len(limit.input_shapes) != 1
        or len(limit.input_shapes[0]) != 2
        or limit.input_shapes[0][0] < 1
        or limit.input_shapes[0][1] < 2
    ):
        raise ValueError("causal training costs require signed [sequences, tokens] input bounds")
    names = tuple(field.name.removesuffix(".parameters") for field in state.fields[: state.optimizer_fields : 5])

    def loss(data: TensorFrame, parameters: Tensors) -> torch.Tensor | None:
        batch = data.step.auxiliary
        if batch is None:
            return None
        assert isinstance(batch, CausalTrainBatch)
        result = function(**{key: port.adapt(batch, parameters) for key, port in ports.items()})
        if isinstance(result, TensorUpdate):
            data.training = replace(result, output=result.output.detach())
            return result.output
        data.trained_rows = batch.input_ids.shape[0]
        return result

    def count(data: TensorFrame) -> int:
        if data.training is None:
            return data.trained_rows
        return int(torch.where(data.training.changed, data.training.trained_rows, 0).item())

    compute = autograd_loss(loss, names, count)

    def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        value = data.step.auxiliary
        if value is not None:
            if not isinstance(value, CausalTrainBatch):
                raise ValueError("causal training consumes CausalTrainBatch")
            device = batch.roots[0][0].device
            if (
                value.input_ids.ndim != 2
                or value.input_ids.shape != value.labels.shape
                or value.input_ids.shape[0] < 1
                or value.input_ids.shape[1] < 2
                or value.input_ids.dtype != torch.int64
                or value.labels.dtype != torch.int64
                or value.input_ids.device != device
                or value.labels.device != device
                or value.attention_mask is not None
                and (value.attention_mask.shape != value.input_ids.shape or value.attention_mask.device != device)
                or limit is not None
                and any(size > bound for size, bound in zip(value.input_ids.shape, limit.input_shapes[0], strict=True))
            ):
                raise ValueError(
                    "causal batch must contain aligned int64 token/label tensors within signed input bounds"
                )
        return compute(data, batch)

    b.causal_state = str(state.declaration.resource)
    call = b.compute(
        name,
        function=execute,
        inputs={key: port.source for key, port in ports.items()},
        outputs=("updated",),
        changes=state,
        effect=StateEffectKind.TRANSITION_MAYBE,
        when=when,
        implementations=(native(identity, "optimizer/adamw@1", "transition/full_root@1"),),
        math=MathContract(
            parameter_gradients=(state.declaration.resource,),
            detached_inputs=tuple(
                key for key, port in ports.items() if not isinstance(port.source, StateSelectionInputSource)
            ),
            reduction="custom",
        ),
    )
    b.publish(call, state, when=when)
    return call
