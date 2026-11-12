"""The order of one engine step over a StateExecution: what runs before, inside and after the model forward.

StepDriver is the one place that sequences requests and steps on the model timeline. VllmSession translates vLLM's
runner callbacks into its calls, pllm.testing.check drives its CPU decoder through the same calls and the
reservation oracle replays schedules through it. Model-free steps (host training, unit tests) run every anchor of
their own timeline with StateExecution.step instead.
"""

from __future__ import annotations

from collections.abc import Collection, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from traceback import extract_tb
from typing import TYPE_CHECKING, Literal

import torch

from pllm.contracts.backend_attachment import BackendAttachment
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.host_batch import HostBatch, StateImport
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch, RowLabeler
from pllm.contracts.request_identity import RequestIdentity, prompt_rows
from pllm.contracts.sampler import SamplerFeedback
from pllm.contracts.semantic_graph import StateScope, TaskId
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.model_sites import state_inputs
from pllm.execution.preemption import preemption_policy
from pllm.execution.ptts.slot_pools import SlotPools, Slots
from pllm.execution.state_execution import (
    HeldInput,
    ProjectionResiduals,
    StateExecution,
    StateTaskContext,
)
from pllm.execution.state_feed import Feed, FeedReceiver, ReadonlyStepAbsent, ReadonlyStepEnd
from pllm.execution.step_input import StepInput

if TYPE_CHECKING:
    from pllm.contracts.semantic_graph import AnchorId
    from pllm.execution.tensor_declaration import TensorOutput
    from pllm.execution.tensor_program import FunctionMath


class StepOpen(Enum):
    """When a step's state invocation opens, decided once from what the model forward reads."""

    # Nothing in the forward reads the step's state: open once the forward produced the step's inputs.
    AFTER_FORWARD = "after_forward"
    # Model effects, token history, a custom forward or host-batch training read the step before the forward.
    BEFORE_FORWARD = "before_forward"
    # An exact overlay trains inside the forward: open before it, with inputs arriving in projection parts.
    IN_FORWARD = "in_forward"

    @staticmethod
    def of(attachment: BackendAttachment, math: FunctionMath) -> StepOpen:
        graph, spec = attachment.plan.semantic_graph, attachment.spec
        custom_forward = graph.custom_forward_task is not None
        if graph.updates_in_forward and not (spec.in_forward is not None and custom_forward):
            return StepOpen.IN_FORWARD
        rewrites, attention = math.model_sites, math.attention
        reads = (
            spec.overlay_bindings
            or spec.activation_bindings
            or custom_forward
            or graph.trains_on_batch
            or graph.input_token_history >= 0
            or attention is not None
            and attention.facts
            or state_inputs(rewrites, attention)
            or any(rewrite.results or rewrite.memory is not None for rewrite in rewrites)
        )
        return StepOpen.BEFORE_FORWARD if reads else StepOpen.AFTER_FORWARD


@dataclass(slots=True)
class ActiveStep:
    """The step between `begin` and `commit`: its rows, its invocation once open and whether the forward is over."""

    rows: PackedSamplerRowBatch
    context: StateTaskContext | None = None
    forward_done: bool = False


class StepDriver:
    """Requests and steps of one attached workload, in the order the engine runs them.

    Per generate call: `stage`, then per engine step `schedule`, `begin`, the forward (whose effects call `anchor`
    and `part`), `after_forward`, `sample`, `commit`; `finish_call` retires the call's requests.
    """

    def __init__(
        self,
        execution: StateExecution,
        attachment: BackendAttachment,
    ) -> None:
        self.execution = execution
        self.open = StepOpen.of(attachment, execution.math)
        self.preemption: Literal["keep", "reset"] = preemption_policy(attachment, execution.math.model_sites)
        self._recompute_rejection = attachment.plan.schedule.recompute_rejection
        graph = attachment.plan.semantic_graph
        self._tapped_phases = frozenset(resource.phase for resource in attachment.probe.hidden_resources)
        imported = graph.host_state(StateImport.schema)
        self._request_import = imported is not None and imported.scope is StateScope.REQUEST
        self.receiver: FeedReceiver | None = None
        if execution.feeds:
            (binding,) = execution.feeds
            self.receiver = FeedReceiver(execution.arena, binding)
        self.labeler = RowLabeler()
        self.admitted: dict[str, RequestIdentity] = {}
        self._pending: dict[str, RequestIdentity] = {}
        self._host_batch: HostBatch[torch.Tensor] | None = None
        self._import_rows: dict[str, int] = {}
        self.step: ActiveStep | None = None
        self.next_step_id = 0
        self._failed = False
        self._sync_widths: dict[int, int] | None = None
        self._steady = False
        self.steady_steps = 0

    # Requests.

    def stage(
        self,
        identities: tuple[RequestIdentity, ...],
        batch: HostBatch[torch.Tensor] | None,
        samples_per_prompt: int = 1,
    ) -> None:
        """Stage one generate call's requests and the host batch its first step trains on."""
        self._require_live()
        # Planning reserves no admission pins for states written only from host batches: no request may hold one.
        assert not self.admitted, "generate stages a call's inputs after the previous call retired"
        self._host_batch = batch
        self._import_rows = prompt_rows(identities, samples_per_prompt)
        self._pending = {identity.request_id: identity for identity in identities}

    def schedule(self, finished: Collection[str], preempted: Collection[str], new: Iterable[str]) -> None:
        """Before a step: retire finished requests, apply the preemption policy and admit newly scheduled ones."""
        self._require_live()
        execution = self.execution
        execution.arena.collect_completed()
        self._retire(tuple(self.admitted[request] for request in sorted(self.admitted.keys() & finished)))
        if self.preemption == "reset":
            for request in finished:
                self._pending.pop(request, None)
            self._preempt(preempted)
        admitted = tuple(self._pending[request] for request in new if request not in self.admitted)
        if not admitted:
            return
        if execution.directory.request_states:
            device = execution.device
            execution.admit(admitted, torch.cuda.current_stream(device) if device.type == "cuda" else None)
        for identity in admitted:
            self.labeler.admit(identity)
            self.admitted[identity.request_id] = identity
            self._pending.pop(identity.request_id)

    def _preempt(self, requests: Collection[str]) -> None:
        identities = tuple(self.admitted[request] for request in sorted(self.admitted.keys() & requests))
        if not identities:
            return
        if self._recompute_rejection is not None:
            self._failed = True
            raise RuntimeError(
                "vLLM preempted a request and would recompute its history; "
                f"{self._recompute_rejection}. To avoid preemption, give vLLM more KV cache "
                "(gpu_memory_utilization) or fewer concurrent sequences (max_num_seqs)"
            )
        self._retire(identities)
        self._pending.update((identity.request_id, identity) for identity in identities)

    def _retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        execution = self.execution
        held = (
            identities
            if execution.directory.request_states
            else tuple(identity for identity in identities if execution.directory.has_inference_pin(identity))
        )
        if held:
            # Released pages remain fenced in the arena. Finishing a subset of the
            # requests must not make the host wait for unrelated side-stream work.
            execution.retire(held)
        for identity in identities:
            self.admitted.pop(identity.request_id)
            self.labeler.retire(identity)

    def finish_call(self, *, wait: bool = False) -> None:
        """Retire every request of the generate call."""
        self.execution.release_history()
        self._retire(tuple(self.admitted[request] for request in sorted(self.admitted)))
        self._pending.clear()
        self.execution.arena.collect_completed(wait=wait)

    def require_idle(self) -> None:
        """Workload replacement cannot discard admitted or staged requests."""
        if self.step is not None or self.admitted or self._pending:
            raise RuntimeError("finish or abort active requests before replacing the workload")

    def feed(self, name: str) -> Feed:
        """The declared writer of a fed state, obtained while the consumer is idle."""
        if self.receiver is None or self.next_step_id or self.step is not None:
            raise RuntimeError("declare a fed state and obtain its writer before generate")
        return self.receiver.take(name)

    def release(self) -> None:
        """After the last call's `finish_call`: retire the instance and dormant states too."""
        execution = self.execution
        if execution.directory.instance_states:
            execution.retire((), wait=True)
        if execution.directory.request_states:
            execution.retire_dormant(wait=True)

    def close(self) -> None:
        try:
            self.execution.close()
        finally:
            if self.receiver is not None:
                self._failed = True
                self.execution.discard_borrows()
                self.receiver.close()

    def _require_live(self) -> None:
        if self._failed:
            raise RuntimeError("this engine stopped after an earlier error; create a new one with pllm.make_llm")

    # One step.

    def begin(self, value: StepInput, stream: torch.cuda.Stream | None) -> StateTaskContext | None:
        """Start a step on `value`'s rows; open its invocation now when the forward reads it."""
        self._require_live()
        self.next_step_id += 1
        rows = value.sampler_rows
        step = self.step = ActiveStep(rows)
        execution = self.execution
        if self.open is StepOpen.AFTER_FORWARD:
            return None
        if self.open is StepOpen.IN_FORWARD:
            step.context = execution.begin(rows.row_batch.request_rows, stream, retain_gradients=False)
            execution.advance_to(step.context, "request.prepare")
            return step.context
        held_input = self._receive_feed(rows, stream)
        step.context = execution.start_step(self._staged(value), stream, held_input=held_input, retain_gradients=False)
        self._consume_host_batch()
        execution.advance_to(step.context, "request.prepare")
        # Offline publication precedes admission; existing request pins remain unchanged.
        pending = tuple(
            identity
            for identity in sorted(
                rows.identities, key=lambda item: (item.request_id, item.prompt_index, item.sample_index)
            )
            if execution.directory.pinned_states and not execution.directory.has_inference_pin(identity)
        )
        if pending:
            execution.admit(pending, stream)
        return step.context

    def _receive_feed(self, rows: PackedSamplerRowBatch, stream: torch.cuda.Stream | None) -> HeldInput | None:
        if self.receiver is None:
            return None
        try:
            received = self.receiver(rows, stream)
            assert received.step_id == rows.step_id
            if isinstance(received, (ReadonlyStepAbsent, ReadonlyStepEnd)):
                reason = "end" if isinstance(received, ReadonlyStepEnd) else "absent"
                return self.execution.stage_hold(tuple(rows.identities), reason, step_id=rows.step_id)
            # Ownership becomes the execution's pending borrow.
            self.execution.stage_readonly(received.roots)
            return None
        except BaseException:
            self._failed = True
            raise

    def anchor(self, context: StateTaskContext, anchor: AnchorId) -> None:
        """Run an anchor the model forward reaches (a custom forward's)."""
        with self._checked():
            self.execution.execute_anchor(context, anchor)

    def part(self, context: StateTaskContext, value: StepInput, task: TaskId, *, complete: bool) -> ProjectionResiduals:
        """Run one projection segment of an in-forward update."""
        with self._checked():
            return self.execution.execute_part(context, value, task, complete=complete)

    def after_forward(
        self,
        value: StepInput,
        stream: torch.cuda.Stream | None,
        holds: tuple[tuple[SlotPools, Slots], ...] = (),
    ) -> None:
        """Hand the forward's outputs to the step and run it up to the sampler; `holds` keep producer slots alive
        for the step's readers. A step whose forward produced nothing the workload reads ends here."""
        step = self.step
        if step is None or step.forward_done:
            return
        step.forward_done = True
        execution = self.execution
        value = self._staged(value)
        with self._checked():
            if step.context is None:
                if not self._tapped(step.rows.row_batch):
                    self.step = None
                    return
                overlap = execution.post_sample
                if overlap is not None and not overlap.lag:
                    stream = overlap.stream
                history = () if overlap is None else overlap.history
                step.context = execution.start_step(value, stream, history=history, retain_gradients=False)
                self._consume_host_batch()
            elif step.context.value is None or value.hidden is not None or value.operands:
                readiness = None if stream is None else CudaStreamReadiness(stream)
                execution.submit_input(step.context, value, readiness)
            if holds:
                step.context.hold_slots(holds)
            execution.advance_to(step.context, "sampler.pre_draw")

    def _consume_host_batch(self) -> None:
        """Instance states train on a host batch once per generate call; request imports reach each request's
        first step, so their batch stays until every staged request was admitted."""
        if not self._pending or self.execution.directory.instance_states and not self._request_import:
            self._host_batch = None

    def _tapped(self, rows: PackedPhaseRowBatch) -> bool:
        """Whether the forward produced rows a hidden tap keeps; a step without them has nothing to run."""
        return HiddenPhase.ALL in self._tapped_phases or any(
            (rows.phases == int(phase is HiddenPhase.PREFILL)).any() for phase in self._tapped_phases
        )

    def sample(self, feedback: SamplerFeedback | None, stream: torch.cuda.Stream | None) -> None:
        """Complete the sampler anchor with what the sampler drew."""
        if self.step is None:
            return
        context = self.context()
        with self._checked():
            self.execution.execute_anchor(context, self.execution.sampler_anchor_id)
            self.execution.complete_sampler(context, stream, feedback=feedback)

    def commit(self, stream: torch.cuda.Stream | None = None) -> TensorOutput:
        """Run the step's remaining anchors and publish it; pass the stream that reads its result, if any."""
        if self.step is None:
            return None
        context = self.context()
        with self._checked():
            self.execution.advance_to(context, "next.prepare")
            output = self.execution.finish_step(context, stream)
        self.step = None
        self.execution.arena.collect_completed()
        return output

    def abort(self) -> None:
        """Cancel the active step and drop the staged host batch."""
        self.execution.cancel_active()
        self._host_batch = None
        self.step = None

    def context(self) -> StateTaskContext:
        assert self.step is not None and self.step.context is not None
        return self.step.context

    def _staged(self, value: StepInput) -> StepInput:
        return replace(value, batch=self._host_batch, import_rows=self._import_rows)

    # Host-sync check (pllm.testing.check_on_gpu).

    def check_host_syncs(self) -> None:
        """From now on, raise on host synchronization inside pLLM's per-step calls of steady decode steps: every
        row decoding, at a width two earlier steps had. `steady_steps` counts the checked steps."""
        self._sync_widths = {}

    @contextmanager
    def step_scope(self, rows: PackedPhaseRowBatch) -> Generator[None]:
        """Scope a step's preparation, counting it towards the host-sync check."""
        self._steady = False
        if self._sync_widths is not None and rows.rows and all(row.role is HiddenPhase.DECODE for row in rows.rows):
            seen = self._sync_widths[len(rows.rows)] = self._sync_widths.get(len(rows.rows), 0) + 1
            self._steady = seen > 2
            self.steady_steps += self._steady
        with self._checked():
            yield

    @contextmanager
    def _checked(self) -> Generator[None]:
        if not self._steady:
            yield
            return
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            yield
        except RuntimeError as error:
            frames = "; ".join(f"{f.filename}:{f.lineno} ({f.name})" for f in extract_tb(error.__traceback__)[-5:])
            raise RuntimeError(f"{error}; {frames}") from error
        finally:
            torch.cuda.set_sync_debug_mode(mode)
