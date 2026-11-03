"""Side-stream input ownership and retained reader versions for vLLM task launches.

The backend protects reusable inputs before a signed writer launch. With lag L, readers rewind
the last L step inputs (new lineages keep their admission version); writers always extend
the latest version. CUDA completion never determines which version a reader selects.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

import torch

from pllm.contracts.host_batch import map_host_batch
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.tensor_frame import TensorFrame

if TYPE_CHECKING:
    from pllm.execution.state_execution import StateTaskContext

_DataT = TypeVar("_DataT")


class PostSampleOverlap:
    """One serial writer lane and a bounded history of exact pins; the step hands off before `anchor`, the first
    anchor with a task that reads the latest version."""

    def __init__(
        self,
        arena: PageBundleArena,
        directory: RequestStateDirectory,
        lag: int,
        stream: torch.cuda.Stream | None,
        anchor: int,
    ) -> None:
        self.arena, self.directory, self.lag, self.stream, self.anchor = arena, directory, lag, stream, anchor
        self._history: deque[PagePin] = deque()

    @property
    def history(self) -> tuple[StateRefs, ...]:
        """Writer inputs, oldest first, retained independently of GPU completion."""
        return tuple(pin.refs for pin in self._history)

    def hand_off(self, context: StateTaskContext[_DataT]) -> None:
        """Protect reusable inputs and move the step to the writer stream, once per step."""
        if context.handed_off:
            return
        self._copy_inputs(context)
        if self.lag or context.pin.stream != self.stream:
            with nullcontext() if self.stream is None else torch.cuda.stream(self.stream):
                self._resume_latest(context)
                if self.lag:
                    # Expired history is released on the writer stream that owns it.
                    self._history.append(self.arena.pin_exact(context.pin.refs, self.stream))
                    while len(self._history) > self.lag:
                        self.arena.release_pin(self._history.popleft())
        context.handed_off = True

    def _resume_latest(self, context: StateTaskContext[_DataT]) -> None:
        """Move the step's pin to the writer stream without losing exact read holds.

        Old reader pages stay pinned on the writer stream until the step commits, including aliases
        consumed by later tasks. Equal reader and writer refs share the new pin; retained
        history owns its own hold.
        """
        assert not context.writing and not context.publications and not context.borrowed_roots
        reader = context.pin
        context.selection = self.directory.select(context.selection.rows)
        if reader.refs == context.selection.refs:
            if self.stream is not None:
                context.pin = context.move_owned_pin(reader, self.stream)
        else:
            context.pin = self.arena.pin_exact(context.selection.refs, self.stream)
            context.register_pin(context.pin)
            if self.stream is not None:
                context.move_owned_pin(reader, self.stream)
        context.move_slots()

    def _copy_inputs(self, context: StateTaskContext[_DataT]) -> None:
        """Own sampler feedback and results before the writer consumes them.

        Hidden rows stay in held slots. Only copies of reusable feedback and result
        tensors fence the main stream; writer execution follows on its serial lane.
        """
        if self.stream is None:
            data = context.data()
            assert isinstance(data, TensorFrame)
            context.value = replace(data, plan=None)
            return
        main = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            data = context.data()
            assert isinstance(data, TensorFrame)
            value = data.step
            if value.batch is not None:
                # Worker-local device copies of external inputs can be dropped at
                # the host commit while their side-stream consumers are still running.
                map_host_batch(value.batch, self._record_input)
            feedback = context.sampler_feedback
            if feedback is not None:
                context.sampler_feedback = replace(
                    feedback, base_logits=feedback.base_logits.clone(), sampled_tokens=feedback.sampled_tokens.clone()
                )
                feedback.base_logits.record_stream(self.stream)
                feedback.sampled_tokens.record_stream(self.stream)
            for key, result in context.results.items():
                # Captured results may be overwritten on the next reader invocation.
                context.results[key] = result.clone()
                result.record_stream(self.stream)
            if feedback is not None or context.results:
                main.wait_stream(self.stream)
            # Cached row versions belong to the reader's selection.
            context.value = replace(data, step=value, plan=None)

    def _record_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda:
            assert self.stream is not None
            tensor.record_stream(self.stream)
        return tensor

    def release_history(self) -> None:
        """Release history before retirement/drain, also on cancellation."""
        while self._history:
            self.arena.release_pin(self._history.popleft())
