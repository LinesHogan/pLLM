"""Training-batch ports and the ordinary AdamW transition over them, with worker-bound causal forwards."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import torch

from pllm.contracts.auxiliary import CausalTrainBatch, WeightedRowBatch
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding
from pllm.contracts.semantic_graph import MathContract, SemanticInputSource, StateEffectKind, StateSelectionInputSource
from pllm.execution.declaration import Call, State, When, native
from pllm.execution.tensor_frame import TensorFrame, Tensors, TensorUpdate
from pllm.execution.tensor_program import TaskOutput, autograd_loss

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


TrainingBatch = CausalTrainBatch[torch.Tensor] | WeightedRowBatch[torch.Tensor]


@dataclass(frozen=True)
class CausalPort:
    """A signed batch/state read and its invocation-local argument adapter."""

    source: SemanticInputSource
    adapt: Callable[[TrainingBatch, Tensors], torch.Tensor | CausalCall | None]


_TRAINING_WINDOWS = (When.BEFORE_MODEL, When.AFTER_SAMPLE, When.HOST_COMPUTE)


def declare_batch_train(
    b: TensorBinding,
    name: str,
    state: State,
    *,
    ports: Mapping[str, CausalPort],
    function: Callable[..., torch.Tensor | TensorUpdate | None],
    identity: str,
    when: When,
) -> Call:
    """Bind batch inputs, explicit parameter gradients and the shared AdamW transition.

    The method returns a scalar loss (every row trains), a TensorUpdate whose `changed` flag is a device
    tensor, or None to skip the step.
    """
    if state.optimizer is None or state.declaration.key_fields or when not in _TRAINING_WINDOWS:
        raise ValueError("batch training needs an instance AdamW state and BEFORE_MODEL, AFTER_SAMPLE or HOST_COMPUTE")
    if any(
        isinstance(port.source, StateSelectionInputSource) and port.source.resource != state.declaration.resource
        for port in ports.values()
    ):
        raise ValueError("batch training must read its own optimizer state")
    names = state.parameter_names

    def loss(data: TensorFrame, parameters: Tensors) -> torch.Tensor | None:
        batch = data.step.auxiliary
        if batch is None:
            return None
        assert isinstance(batch, (CausalTrainBatch, WeightedRowBatch))
        result = function(**{key: port.adapt(batch, parameters) for key, port in ports.items()})
        if result is None:
            return None
        if not isinstance(result, TensorUpdate):
            rows = (batch.input_ids if isinstance(batch, CausalTrainBatch) else batch.hidden).shape[0]
            one = torch.ones((), dtype=torch.int64, device=result.device)
            result = TensorUpdate(result, one.bool(), one, one * rows)
        data.training = replace(result, output=result.output.detach())
        return result.output

    def summary(data: TensorFrame) -> TaskOutput:
        training = data.training
        assert training is not None
        return TaskOutput(
            loss_count=1,
            update_mask=training.changed,
            counters=torch.stack((training.update_count, torch.where(training.changed, training.trained_rows, 0))),
        )

    compute = autograd_loss(loss, names, summary)

    call = b.compute(
        name,
        function=compute,
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
    # A host step publishes only after its compute window closes, so an abandoned step publishes nothing.
    b.publish(call, state, when=When.HOST_COMMIT if when is When.HOST_COMPUTE else when)
    return call
