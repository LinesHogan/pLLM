"""Request lifecycle for canonical CPU dense fast weights."""

from __future__ import annotations

from threading import RLock

import torch

from tllm.canonical.execution import TaskGraphExecution
from tllm.canonical.ptts.dense_page import (
    DensePageAdmission,
    DensePageStateProvider,
)
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.canonical.ptts.publication import RequestStatePublicationCoordinator
from tllm.canonical.scheduler import validate_physical_plan, validate_schedule
from tllm.canonical.workloads.inplace_ttt.layout import InPlaceTTTStateLayout
from tllm.canonical.workloads.inplace_ttt.tasks import (
    INPLACE_TTT_PUBLISH_IMPLEMENTATION,
    INPLACE_TTT_TRANSITION_IMPLEMENTATION,
    InPlaceTTTPreparedTransition,
    InPlaceTTTPublicationPending,
    InPlaceTTTPublishTask,
    InPlaceTTTPublished,
    InPlaceTTTStepContext,
    InPlaceTTTTransitionPending,
    InPlaceTTTTransitionTask,
)
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTCleanupReceipt,
    InPlaceTTTConfig,
    InPlaceTTTDiagnostics,
    InPlaceTTTSpecialization,
    InPlaceTTTStepInput,
    InPlaceTTTStepResult,
)
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.request_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
)
from tllm.contracts.semantic_graph import AnchorId, ImplementationId, PhysicalPlan
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import StatePhysicalPlan


class InPlaceTTTExecution:
    """Own request roots and mechanically execute one signed task graph."""

    __slots__ = (
        "_config",
        "_directory",
        "_layout",
        "_lock",
        "_plan_bundle",
        "_provider",
        "_publication",
        "_runner",
        "_specialization",
        "_trained_rows",
        "_update_count",
    )

    def __init__(
        self,
        config: InPlaceTTTConfig,
        specialization: InPlaceTTTSpecialization,
        plan_bundle: CanonicalPlanBundle,
        layout: InPlaceTTTStateLayout,
        provider: DensePageStateProvider,
        directory: RequestStateDirectory,
        publication: RequestStatePublicationCoordinator,
    ) -> None:
        if (
            plan_bundle.semantic_graph.execution_parameters_digest
            != specialization.execution_parameters_digest(config)
            or plan_bundle.schedule_physical.plan_epoch
            != plan_bundle.state_physical.plan_epoch
        ):
            raise ValueError("In-Place-TTT execution artifacts disagree")
        validate_schedule(
            plan_bundle.semantic_graph,
            plan_bundle.timeline,
            plan_bundle.schedule_decision,
        )
        validate_physical_plan(
            plan_bundle.semantic_graph,
            plan_bundle.timeline,
            plan_bundle.schedule_decision,
            plan_bundle.schedule_physical,
        )
        implementations_by_task = {
            int(task.task_id): task.implementation_id
            for task in plan_bundle.schedule_physical.tasks
        }
        if tuple(sorted(implementations_by_task)) != (0, 1):
            raise ValueError("In-Place-TTT plan does not cover dense task IDs")
        implementation_ids: tuple[ImplementationId, ImplementationId] = (
            implementations_by_task[0],
            implementations_by_task[1],
        )
        if implementation_ids != (
            INPLACE_TTT_PUBLISH_IMPLEMENTATION,
            INPLACE_TTT_TRANSITION_IMPLEMENTATION,
        ):
            raise ValueError("In-Place-TTT implementations differ from placement")
        self._config = config
        self._specialization = specialization
        self._plan_bundle = plan_bundle
        self._layout = layout
        self._provider = provider
        self._directory = directory
        self._publication = publication
        self._runner = TaskGraphExecution(
            plan_bundle.schedule_physical,
            (
                InPlaceTTTPublishTask(layout, publication),
                InPlaceTTTTransitionTask(config, layout, directory, provider),
            ),
        )
        self._update_count = 0
        self._trained_rows = 0
        self._lock = RLock()

    @property
    def plan(self) -> PhysicalPlan:
        return self._plan_bundle.schedule_physical

    @property
    def state_plan(self) -> StatePhysicalPlan:
        return self._plan_bundle.state_physical

    @property
    def config(self) -> InPlaceTTTConfig:
        return self._config

    @property
    def specialization(self) -> InPlaceTTTSpecialization:
        return self._specialization

    @property
    def execution_parameters_digest(self) -> str:
        return self._specialization.execution_parameters_digest(self._config)

    def admit(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            logical = self._directory.prepare_admission(identities)
            try:
                self._provider.admit_batch(
                    tuple(
                        DensePageAdmission(
                            admission.ref,
                            self._provider.new_zero_root_tensors(),
                        )
                        for admission in logical.admissions
                    )
                )
            except Exception:
                self._directory.abort_admission(logical)
                raise
            self._directory.commit_admission(logical)
            return tuple(admission.ref for admission in logical.admissions)

    def step(self, step: InPlaceTTTStepInput) -> InPlaceTTTStepResult:
        with self._lock:
            return self._step(step)

    def _step(self, step: InPlaceTTTStepInput) -> InPlaceTTTStepResult:
        self._validate_step(step)
        selection = self._directory.select(
            IndexedRequestIdentityBatch(
                (IndexedRequestIdentityRow(0, step.identity),)
            )
        )
        transition_lease = self._directory.acquire_transition_lease(selection)
        try:
            pin = self._provider.pin_exact(selection.refs)
        except Exception:
            self._directory.release_transition_lease(transition_lease)
            raise
        context = InPlaceTTTStepContext(
            step,
            selection,
            transition_lease,
            InPlaceTTTTransitionPending(),
            InPlaceTTTPublicationPending(),
        )
        invocation = self._runner.start(context)
        try:
            anchor_ids = tuple(
                dict.fromkeys(AnchorId(int(task.anchor_id)) for task in self.plan.tasks)
            )
            for anchor_id in anchor_ids:
                self._runner.execute_anchor(invocation, anchor_id)
            completed = self._runner.finish(invocation)
        except Exception:
            match context.transition, context.publication:
                case InPlaceTTTPreparedTransition(
                    logical=logical,
                    physical=physical,
                ), InPlaceTTTPublicationPending():
                    self._publication.abort(logical, physical)
                case _:
                    pass
            self._provider.release_pin(pin)
            self._directory.release_transition_lease(transition_lease)
            raise
        match completed.publication:
            case InPlaceTTTPublished(result=result):
                pass
            case InPlaceTTTPublicationPending():
                self._provider.release_pin(pin)
                self._directory.release_transition_lease(transition_lease)
                raise RuntimeError("signed In-Place-TTT graph did not publish")
        self._provider.release_pin(pin)
        self._directory.release_transition_lease(transition_lease)
        self._update_count += sum(target.update_count for target in result.targets)
        self._trained_rows += sum(target.trained_rows for target in result.targets)
        return result

    def retire(
        self,
        identities: tuple[RequestIdentity, ...],
    ) -> InPlaceTTTCleanupReceipt:
        with self._lock:
            logical = self._directory.prepare_retirement(identities)
            try:
                self._provider.retire_batch(logical.refs)
            except Exception:
                self._directory.abort_retirement(logical)
                raise
            self._directory.commit_retirement(logical)
            return InPlaceTTTCleanupReceipt(logical.refs)

    def snapshot(self, ref: StateRef) -> tuple[torch.Tensor, ...]:
        with self._lock:
            return self._provider.snapshot(ref)

    def diagnostics(self) -> InPlaceTTTDiagnostics:
        with self._lock:
            return InPlaceTTTDiagnostics(
                self._directory.active_count,
                self._update_count,
                self._trained_rows,
                self.state_plan,
                self.plan.fallback_count,
            )

    def _validate_step(self, step: InPlaceTTTStepInput) -> None:
        specialization_by_target = {
            target.target: target for target in self._specialization.targets
        }
        for operand in step.operands:
            try:
                target = specialization_by_target[operand.target]
            except KeyError as exc:
                raise ValueError("operand target is outside specialization") from exc
            if (
                operand.hidden.ndim != 2
                or operand.desired.ndim != 2
                or operand.base_weight.ndim != 2
                or tuple(operand.hidden.shape[1:]) != (target.input_size,)
                or tuple(operand.desired.shape)
                != (int(operand.hidden.shape[0]), target.output_size)
                or tuple(operand.base_weight.shape)
                != (target.output_size, target.input_size)
                or operand.hidden.dtype is not operand.desired.dtype
                or operand.hidden.dtype is not operand.base_weight.dtype
                or operand.hidden.dtype is not self._specialization.torch_dtype
                or int(operand.hidden.shape[0]) < 1
                or operand.hidden.device != self._specialization.device
                or operand.desired.device != self._specialization.device
                or operand.base_weight.device != self._specialization.device
            ):
                raise ValueError("operand tensors differ from specialization")


__all__ = ["InPlaceTTTExecution"]
