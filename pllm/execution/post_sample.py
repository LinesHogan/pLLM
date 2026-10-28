"""Side-stream input ownership and retained reader versions for vLLM task launches.

The backend protects reusable inputs before a signed writer launch. With lag L, readers rewind
the last L step inputs (new lineages keep their admission version); writers always extend
the latest version. CUDA completion never determines which version a reader selects.
"""

from collections import deque
from dataclasses import replace

import torch

from pllm.contracts.auxiliary import map_auxiliary
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import DataT, StateTaskContext
from pllm.execution.tensor_frame import TensorFrame


class PostSampleOverlap:
    """One serial writer lane and a bounded history of exact pins."""

    def __init__(self, arena: PageBundleArena, lag: int, stream: torch.cuda.Stream) -> None:
        self.arena, self.lag, self.stream = arena, lag, stream
        self._history: deque[PagePin] = deque()

    @property
    def history(self) -> tuple[StateRefs, ...]:
        """Writer inputs, oldest first, retained independently of GPU completion."""
        return tuple(pin.refs for pin in self._history)

    def copy_inputs(self, context: StateTaskContext[DataT]) -> None:
        """Own sampler feedback and results before the writer consumes them.

        Hidden rows stay in held slots. Only copies of reusable feedback and result
        tensors fence the main stream; writer execution follows on its serial lane.
        """
        main = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            data = context.data()
            assert isinstance(data, TensorFrame)
            value = data.step
            if value.auxiliary is not None:
                # Worker-local device copies of external inputs can be dropped at
                # host settlement while their side-stream consumers are still running.
                map_auxiliary(value.auxiliary, self._record_input)
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
            # Cached row versions and parameter leaves belong to the reader's selection.
            context.value = replace(data, step=value, index=None, plan=None, leaf=None, fields=(), hidden_inputs=None)

    def _record_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda:
            tensor.record_stream(self.stream)
        return tensor

    def retain(self, refs: StateRefs) -> None:
        """Retain this transition's input; release expired history on its owning stream."""
        if not self.lag:
            return
        self._history.append(self.arena.pin_exact(refs, self.stream))
        while len(self._history) > self.lag:
            self.arena.release_pin(self._history.popleft())

    def clear(self) -> None:
        """Release history before retirement/drain, also on cancellation."""
        while self._history:
            self.arena.release_pin(self._history.popleft())
