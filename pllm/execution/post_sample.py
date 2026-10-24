"""Side-stream input ownership and retained reader versions for vLLM's post-sampler phase.

The backend uses this for read-only pre-sampler tasks without deferred gradients. With lag L, readers rewind
the last L step inputs (new lineages keep their admission version); writers always extend
the latest version. CUDA completion never determines which version a reader selects.
"""

from collections import deque
from dataclasses import replace

import torch

from pllm.contracts.auxiliary import map_auxiliary
from pllm.contracts.packed_hidden import PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.tensor_frame import TensorFrame


class PostSampleOverlap:
    """One serial writer lane, fixed hidden storage and a bounded history of exact pins."""

    def __init__(
        self, arena: PageBundleArena, visibility: VisibilitySource, device: torch.device, *, readers: bool = True
    ) -> None:
        self.arena, self.visibility = arena, visibility
        self.readers = readers
        self.stream = torch.cuda.Stream(device=device)
        self._history: deque[PagePin] = deque()
        self._hidden: dict[tuple[str, str, str], torch.Tensor] = {}

    @property
    def history(self) -> tuple[StateRefs, ...]:
        """Writer inputs, oldest first, retained independently of GPU completion."""
        return tuple(pin.refs for pin in self._history)

    def copy_inputs(self, context: StateTaskContext[TensorFrame]) -> None:
        """Copy reusable backend tensors, then let the next forward overwrite them.

        Only this copy prefix fences the main stream. Graph replay, page transitions and
        releases are enqueued afterwards on the writer lane. Fixed hidden addresses avoid
        recapture; only live rows are copied from the larger model tap buffers.
        """
        main = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            data = context.data()
            value = data.step
            if value.auxiliary is not None:
                # Worker-local device copies of external inputs can be dropped at
                # host settlement while their side-stream consumers are still running.
                map_auxiliary(value.auxiliary, self._record_input)
            if value.hidden is not None:
                value = replace(
                    value,
                    hidden=self._copy_hidden(value.hidden),
                    hidden_readiness=CudaStreamReadiness(self.stream),
                )
            feedback = context.sampler_feedback
            if feedback is not None:
                context.sampler_feedback = replace(
                    feedback, base_logits=feedback.base_logits.clone(), sampled_tokens=feedback.sampled_tokens.clone()
                )
                feedback.base_logits.record_stream(self.stream)
                feedback.sampled_tokens.record_stream(self.stream)
            for key, result in context.results.items():
                # Eager methods may return views of backend inputs. Stream recording
                # alone protects allocation lifetime, not writes to a reused tap.
                context.results[key] = result.clone()
                result.record_stream(self.stream)
            main.wait_stream(self.stream)
            # Cached row versions and parameter leaves belong to the reader's selection.
            context.value = replace(data, step=value, index=None, plan=None, leaf=None, fields=(), hidden_inputs=None)

    def _copy_hidden(self, batch: PackedHiddenSnapshotBatch) -> PackedHiddenSnapshotBatch:
        snapshots: list[PackedHiddenSnapshot] = []
        for snapshot in batch.snapshots:
            key = snapshot.resource.locator_key
            if key not in self._hidden:
                self._hidden[key] = torch.empty_like(snapshot.storage)
            buffer = self._hidden[key]
            hidden = buffer[: len(snapshot.hidden)]
            hidden.copy_(snapshot.hidden)
            snapshot.hidden.record_stream(self.stream)
            snapshots.append(PackedHiddenSnapshot(snapshot.resource, hidden, buffer))
        return PackedHiddenSnapshotBatch(batch.row_batch, tuple(snapshots))

    def _record_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda:
            tensor.record_stream(self.stream)
        return tensor

    def retain(self, pin: PagePin) -> None:
        """Retain this transition's input; release expired history on its owning stream."""
        self._history.append(pin)
        while len(self._history) > self.visibility.lag:
            self.arena.release_pin(self._history.popleft())

    def clear(self) -> None:
        """Release history before retirement/drain, also on cancellation."""
        while self._history:
            self.arena.release_pin(self._history.popleft())
