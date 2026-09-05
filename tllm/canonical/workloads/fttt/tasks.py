"""Functional CPU tasks for the canonical FTTT signed graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import torch
from torch.nn import functional as F

from tllm.canonical.workloads.fttt.math import edit_inputs, target_delta
from tllm.canonical.workloads.fttt.types import (
    FTTTClosedLoopIntent,
    FTTTConfig,
    FTTTCpuReadiness,
    FTTTCudaReadiness,
    FTTTHostTransaction,
    FTTTResult,
    FTTTSpecialization,
)
from tllm.contracts.semantic_graph import ImplementationId


FTTT_PREPARE_IMPLEMENTATION = ImplementationId("canonical.fttt:cpu_prepare@1")
FTTT_LOSS_IMPLEMENTATION = ImplementationId("canonical.fttt:cpu_loss@1")
FTTT_OPTIMIZE_IMPLEMENTATION = ImplementationId("canonical.fttt:cpu_adam@1")
FTTT_PREDICT_IMPLEMENTATION = ImplementationId("canonical.fttt:cpu_predict@1")
FTTT_PUBLISH_IMPLEMENTATION = ImplementationId("canonical.fttt:cpu_publish@1")
FTTT_CUDA_PREPARE_IMPLEMENTATION = ImplementationId(
    "canonical.fttt:cuda_prepare@1"
)
FTTT_CUDA_LOSS_IMPLEMENTATION = ImplementationId("canonical.fttt:cuda_loss@1")
FTTT_CUDA_OPTIMIZE_IMPLEMENTATION = ImplementationId(
    "canonical.fttt:cuda_adam@1"
)
FTTT_CUDA_PREDICT_IMPLEMENTATION = ImplementationId(
    "canonical.fttt:cuda_predict@1"
)


@dataclass(slots=True)
class FTTTEditState:
    down: torch.Tensor
    up: torch.Tensor
    down_exp_avg: torch.Tensor
    down_exp_avg_sq: torch.Tensor
    up_exp_avg: torch.Tensor
    up_exp_avg_sq: torch.Tensor
    specialization: FTTTSpecialization
    step: int = 0
    version: int = 0

    def __post_init__(self) -> None:
        if (
            self.down.ndim != 2
            or self.up.ndim != 2
            or int(self.down.shape[0]) != int(self.up.shape[1])
            or self.down.device.type not in ("cpu", "cuda")
            or self.up.device != self.down.device
            or self.down.dtype is not torch.float32
            or self.up.dtype is not self.down.dtype
            or self.down_exp_avg.shape != self.down.shape
            or self.down_exp_avg_sq.shape != self.down.shape
            or self.up_exp_avg.shape != self.up.shape
            or self.up_exp_avg_sq.shape != self.up.shape
            or type(self.step) is not int
            or self.step < 0
            or type(self.version) is not int
            or self.version < 0
        ):
            raise ValueError("FTTT edit state is not layout-aligned")

    def prepare(self, transaction: FTTTHostTransaction) -> FTTTStepContext:
        if (
            int(transaction.hidden_rows.shape[1]) != self.specialization.hidden_size
            or transaction.hidden_rows.dtype is not torch.float32
            or transaction.hidden_rows.device != self.specialization.device
        ):
            raise ValueError("FTTT transaction differs from specialization")
        match transaction.readiness:
            case FTTTCpuReadiness() if self.specialization.device.type == "cpu":
                pass
            case FTTTCudaReadiness() as readiness if self.specialization.device.type == "cuda":
                readiness.order_on(torch.cuda.current_stream(self.specialization.device))
            case _:
                raise ValueError("FTTT readiness differs from placement")
        return FTTTStepContext(
            transaction, self, FTTTPreparePending(), FTTTLossPending(),
            FTTTOptimizationPending(), FTTTPredictionPending(), FTTTPublicationPending(),
        )

    def snapshot(self) -> FTTTEditState:
        return replace(
            self, down=self.down.detach().clone(), up=self.up.detach().clone(),
            down_exp_avg=self.down_exp_avg.clone(), down_exp_avg_sq=self.down_exp_avg_sq.clone(),
            up_exp_avg=self.up_exp_avg.clone(), up_exp_avg_sq=self.up_exp_avg_sq.clone(),
        )

    def restore(self, snapshot: FTTTEditState) -> None:
        self.down, self.up = snapshot.down, snapshot.up
        self.down_exp_avg, self.down_exp_avg_sq = snapshot.down_exp_avg, snapshot.down_exp_avg_sq
        self.up_exp_avg, self.up_exp_avg_sq = snapshot.up_exp_avg, snapshot.up_exp_avg_sq
        self.step, self.version = snapshot.step, snapshot.version

    def output(self, context: FTTTStepContext) -> FTTTResult:
        match context.prediction, context.publication, context.optimization:
            case (
                FTTTPredicted(delta=delta), FTTTPublished(intent=intent),
                FTTTNotUpdated(version=version, update_count=updates),
            ):
                losses: tuple[torch.Tensor, ...] = ()
            case (
                FTTTPredicted(delta=delta), FTTTPublished(intent=intent),
                FTTTOptimized(loss=loss, version=version, update_count=updates),
            ):
                losses = (loss,)
            case _:
                raise RuntimeError("FTTT signed graph did not produce all outputs")
        return FTTTResult(
            context.transaction.plan.intent.transaction_id, delta, intent, losses, updates, version,
        )


def initialize_edit_state(
    config: FTTTConfig,
    specialization: FTTTSpecialization,
) -> FTTTEditState:
    input_size = (
        specialization.hidden_size + config.feedback_feature_size + 1
    )
    generator = torch.Generator(device=specialization.device)
    generator.manual_seed(specialization.initialization_seed)
    down = torch.empty(
        (config.rank, input_size),
        dtype=torch.float32,
        device=specialization.device,
    )
    down.normal_(
        mean=0.0,
        std=1.0 / sqrt(float(input_size)),
        generator=generator,
    )
    up = torch.zeros(
        (specialization.hidden_size, config.rank),
        dtype=torch.float32,
        device=specialization.device,
    )
    return FTTTEditState(
        down,
        up,
        torch.zeros_like(down),
        torch.zeros_like(down),
        torch.zeros_like(up),
        torch.zeros_like(up),
        specialization,
    )


def low_rank_edit(
    edit_input: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    return F.linear(F.linear(edit_input, down), up)


@dataclass(frozen=True, slots=True)
class FTTTPreparePending:
    pass


@dataclass(frozen=True, slots=True)
class _Prepared:
    edit_input: torch.Tensor
    target: torch.Tensor
    intent: FTTTClosedLoopIntent


@dataclass(frozen=True, slots=True)
class FTTTLossPending:
    pass


@dataclass(frozen=True, slots=True)
class _NoGradients:
    edit_input: torch.Tensor
    intent: FTTTClosedLoopIntent


@dataclass(frozen=True, slots=True)
class _Gradients:
    down: torch.Tensor
    up: torch.Tensor
    loss: torch.Tensor
    edit_input: torch.Tensor
    intent: FTTTClosedLoopIntent


@dataclass(frozen=True, slots=True)
class FTTTOptimizationPending:
    pass


@dataclass(frozen=True, slots=True)
class FTTTNotUpdated:
    edit_input: torch.Tensor
    intent: FTTTClosedLoopIntent
    version: int
    update_count: int = 0


@dataclass(frozen=True, slots=True)
class FTTTOptimized:
    loss: torch.Tensor
    edit_input: torch.Tensor
    intent: FTTTClosedLoopIntent
    version: int
    update_count: int


@dataclass(frozen=True, slots=True)
class FTTTPredictionPending:
    pass


@dataclass(frozen=True, slots=True)
class FTTTPredicted:
    delta: torch.Tensor


@dataclass(frozen=True, slots=True)
class FTTTPublicationPending:
    pass


@dataclass(frozen=True, slots=True)
class FTTTPublished:
    intent: FTTTClosedLoopIntent


_PrepareState = FTTTPreparePending | _Prepared
_LossState = FTTTLossPending | _NoGradients | _Gradients
_OptimizationState = FTTTOptimizationPending | FTTTNotUpdated | FTTTOptimized
_PredictionState = FTTTPredictionPending | FTTTPredicted
_PublicationState = FTTTPublicationPending | FTTTPublished


@dataclass(slots=True)
class FTTTStepContext:
    transaction: FTTTHostTransaction
    edit_state: FTTTEditState
    prepared: _PrepareState
    gradients: _LossState
    optimization: _OptimizationState
    prediction: _PredictionState
    publication: _PublicationState


class FTTTPrepareTask:
    __slots__ = ("_config", "_implementation_id")

    def __init__(
        self,
        config: FTTTConfig,
        implementation_id: ImplementationId,
    ) -> None:
        if implementation_id not in (
            FTTT_PREPARE_IMPLEMENTATION,
            FTTT_CUDA_PREPARE_IMPLEMENTATION,
        ):
            raise ValueError("FTTT prepare implementation is not closed")
        self._config = config
        self._implementation_id = implementation_id

    @property
    def implementation_id(self) -> ImplementationId:
        return self._implementation_id

    def execute(self, context: FTTTStepContext) -> None:
        plan = context.transaction.plan
        hidden_rows = context.transaction.hidden_rows
        if plan.selected_rows:
            selected = hidden_rows.detach().index_select(
                0,
                torch.tensor(
                    plan.selected_rows,
                    dtype=torch.long,
                    device=hidden_rows.device,
                ),
            )
            edit_input = edit_inputs(
                selected,
                plan.intent.feedback_records,
                feature_size=self._config.feedback_feature_size,
            )
            target = target_delta(
                selected,
                plan.intent.feedback_records,
                feature_size=self._config.feedback_feature_size,
                edit_scale=self._config.edit_scale,
            )
        else:
            hidden_size = int(hidden_rows.shape[1])
            edit_input = hidden_rows.new_empty(
                (0, hidden_size + self._config.feedback_feature_size + 1)
            )
            target = hidden_rows.new_empty((0, hidden_size))
        context.prepared = _Prepared(edit_input, target, plan.intent)


class FTTTLossTask:
    __slots__ = ("_implementation_id",)

    def __init__(self, implementation_id: ImplementationId) -> None:
        if implementation_id not in (
            FTTT_LOSS_IMPLEMENTATION,
            FTTT_CUDA_LOSS_IMPLEMENTATION,
        ):
            raise ValueError("FTTT loss implementation is not closed")
        self._implementation_id = implementation_id

    @property
    def implementation_id(self) -> ImplementationId:
        return self._implementation_id

    def execute(self, context: FTTTStepContext) -> None:
        match context.prepared:
            case _Prepared(edit_input=edit_input, target=target, intent=intent):
                pass
            case FTTTPreparePending():
                raise RuntimeError("FTTT loss requires a prepared host transaction")
        if int(edit_input.shape[0]) == 0:
            context.gradients = _NoGradients(edit_input, intent)
            return
        down = context.edit_state.down.detach().requires_grad_(True)
        up = context.edit_state.up.detach().requires_grad_(True)
        loss = F.mse_loss(low_rank_edit(edit_input, down, up), target)
        down_gradient, up_gradient = torch.autograd.grad(loss, (down, up))
        context.gradients = _Gradients(
            down_gradient.detach(),
            up_gradient.detach(),
            loss.detach(),
            edit_input,
            intent,
        )


class FTTTOptimizeTask:
    __slots__ = ("_config", "_implementation_id")

    def __init__(
        self,
        config: FTTTConfig,
        implementation_id: ImplementationId,
    ) -> None:
        if implementation_id not in (
            FTTT_OPTIMIZE_IMPLEMENTATION,
            FTTT_CUDA_OPTIMIZE_IMPLEMENTATION,
        ):
            raise ValueError("FTTT optimizer implementation is not closed")
        self._config = config
        self._implementation_id = implementation_id

    @property
    def implementation_id(self) -> ImplementationId:
        return self._implementation_id

    def execute(self, context: FTTTStepContext) -> None:
        match context.gradients:
            case _NoGradients(edit_input=edit_input, intent=intent):
                context.optimization = FTTTNotUpdated(
                    edit_input,
                    intent,
                    context.edit_state.version,
                    0,
                )
                return
            case _Gradients(
                down=down_gradient,
                up=up_gradient,
                loss=loss,
                edit_input=edit_input,
                intent=intent,
            ):
                pass
            case FTTTLossPending():
                raise RuntimeError("FTTT optimizer requires a completed loss task")
        state = context.edit_state
        state.step += 1
        beta_one, beta_two = self._config.betas
        with torch.no_grad():
            for parameter, gradient, exp_avg, exp_avg_sq in (
                (state.down, down_gradient, state.down_exp_avg, state.down_exp_avg_sq),
                (state.up, up_gradient, state.up_exp_avg, state.up_exp_avg_sq),
            ):
                exp_avg.mul_(beta_one).add_(gradient, alpha=1.0 - beta_one)
                exp_avg_sq.mul_(beta_two).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta_two,
                )
                denominator = exp_avg_sq.sqrt().div_(
                    (1.0 - beta_two**state.step) ** 0.5
                ).add_(self._config.eps)
                step_size = self._config.learning_rate / (
                    1.0 - beta_one**state.step
                )
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)
        state.version += 1
        context.optimization = FTTTOptimized(
            loss,
            edit_input,
            intent,
            state.version,
            1,
        )


class FTTTPredictTask:
    __slots__ = ("_implementation_id",)

    def __init__(self, implementation_id: ImplementationId) -> None:
        if implementation_id not in (
            FTTT_PREDICT_IMPLEMENTATION,
            FTTT_CUDA_PREDICT_IMPLEMENTATION,
        ):
            raise ValueError("FTTT predict implementation is not closed")
        self._implementation_id = implementation_id

    @property
    def implementation_id(self) -> ImplementationId:
        return self._implementation_id

    def execute(self, context: FTTTStepContext) -> None:
        match context.optimization:
            case FTTTNotUpdated(edit_input=edit_input) | FTTTOptimized(
                edit_input=edit_input
            ):
                pass
            case FTTTOptimizationPending():
                raise RuntimeError("FTTT prediction requires optimizer completion")
        context.prediction = FTTTPredicted(
            low_rank_edit(
                edit_input,
                context.edit_state.down,
                context.edit_state.up,
            ).detach()
        )


class FTTTPublishTask:
    __slots__ = ()

    @property
    def implementation_id(self) -> ImplementationId:
        return FTTT_PUBLISH_IMPLEMENTATION

    def execute(self, context: FTTTStepContext) -> None:
        match context.optimization:
            case FTTTNotUpdated(intent=intent, version=version) | FTTTOptimized(
                intent=intent,
                version=version,
            ):
                pass
            case FTTTOptimizationPending():
                raise RuntimeError("FTTT publication requires optimizer completion")
        context.publication = FTTTPublished(
            replace(intent, adapter_version=version)
        )


__all__ = [
    "FTTTEditState",
    "FTTT_CUDA_LOSS_IMPLEMENTATION",
    "FTTT_CUDA_OPTIMIZE_IMPLEMENTATION",
    "FTTT_CUDA_PREDICT_IMPLEMENTATION",
    "FTTT_CUDA_PREPARE_IMPLEMENTATION",
    "FTTTLossPending",
    "FTTTNotUpdated",
    "FTTTOptimizationPending",
    "FTTTOptimized",
    "FTTTPredicted",
    "FTTTPredictionPending",
    "FTTTPreparePending",
    "FTTTPublicationPending",
    "FTTTPublished",
    "FTTT_LOSS_IMPLEMENTATION",
    "FTTT_OPTIMIZE_IMPLEMENTATION",
    "FTTT_PREDICT_IMPLEMENTATION",
    "FTTT_PREPARE_IMPLEMENTATION",
    "FTTT_PUBLISH_IMPLEMENTATION",
    "FTTTLossTask",
    "FTTTOptimizeTask",
    "FTTTPredictTask",
    "FTTTPrepareTask",
    "FTTTPublishTask",
    "FTTTStepContext",
    "initialize_edit_state",
    "low_rank_edit",
]
