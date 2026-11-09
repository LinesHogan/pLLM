"""A deterministic CPU decoder for testing.check's custom-forward schedule comparisons.

It exercises real state bindings and token programs; its pointwise layers do not model attention or KV caches.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pllm.execution.attention import DenseAttention, PagedKV
    from pllm.execution.attention_methods import Argument
    from pllm.execution.compile_facts import BackendCompileFacts
    from pllm.execution.tensor_declaration import TensorDeclaration

import torch

from pllm.contracts.kv_layout import storage_layout
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.execution.custom_forward import CustomForwardTask, SerialCustomForward
from pllm.execution.state_execution import StateExecution
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_declaration import TensorOutput
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.token_program import KV


class CheckForward(SerialCustomForward):
    """Run authored token math against a small deterministic stack, with production state reads."""

    def __init__(self, execution: StateExecution[StepInput, TensorFrame, TensorOutput]) -> None:
        task = execution.arena.plan.semantic_graph.custom_forward_task
        assert task is not None
        scheduled = execution.arena.plan.scheduled(task.symbol)
        binding = CustomForwardTask[TensorFrame].from_implementation(
            execution.implementation(scheduled.implementation_id)
        )
        execution.rebind_implementation(replace(binding, primitive=self))
        self.attention: CheckAttention | None = None
        self._rows: Sequence[PackedPhaseRow] = ()
        self.inputs = torch.empty(0)
        self.output = torch.empty(0)

    def prepare(self, value: StepInput, inputs: torch.Tensor) -> None:
        """Set this step's independently generated token embeddings."""
        self._rows, self.inputs = value.rows.rows, inputs

    @property
    def rows(self) -> Sequence[PackedPhaseRow]:
        return self._rows

    def begin(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([row.token_offset]), self.inputs[row.packed_row_index : row.packed_row_index + 1]

    def layer(
        self,
        row: PackedPhaseRow,
        ordinal: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        kv: KV = "commit",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream = hidden if residual is None else hidden + residual
        if self.attention is not None and ordinal in self.attention.program.methods:
            return self.attention.layer((row,), ordinal, stream), torch.zeros_like(stream)
        return (stream.float() * 0.5 + (ordinal + 1) * 0.01).tanh().to(hidden.dtype), torch.zeros_like(hidden)

    def propagate(self, row: PackedPhaseRow, ordinal: int, positions: torch.Tensor, hidden: torch.Tensor) -> None:
        """Pointwise fake layers have no KV; the causal decoder oracle checks propagation."""

    def normalize(self, row: PackedPhaseRow, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return hidden + residual

    def commit(self, row: PackedPhaseRow, record: Callable[[], None]) -> None:
        record()

    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None:
        self.output = torch.cat(outputs)


class CheckAttention:
    """Tiny dict-backed paged attention for CPU permutation, chunk and recompute checks."""

    def __init__(
        self, binding: "TensorDeclaration", facts: "BackendCompileFacts", *, device_equivalence: bool = False
    ) -> None:
        from pllm.execution.model_sites import StateInput
        from pllm.execution.task_memory import seeded_site_state

        self.program, self.probe = binding.attention, facts.probe
        self.cache: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.parameters = {
            source: seeded_site_state(binding, source, None, facts.initialization_seed)
            for method in self.program.methods.values()
            for _, source in method.inputs
            if isinstance(source, StateInput)
        }
        self.device_equivalence = device_equivalence

    def prepare(self, value: StepInput) -> None:
        """Stage sampled-to-packed indices for the fact snapshot tasks."""
        from pllm.execution.cuda_stream import upload

        mapping = tuple(
            link.packed_row_index for link in sorted(value.sampler_rows.links, key=lambda link: link.sampler_row_index)
        )
        if self.program.sample_rows.numel():
            self.program.sample_rows[: len(mapping)].copy_(upload(mapping, torch.device("cpu")))

    def step(self, value: StepInput, tokens: torch.Tensor) -> dict[tuple[str, int, str], torch.Tensor]:
        """Replay scheduled queries against request prefixes, including same-step causal K/V."""
        self.prepare(value)
        outputs: dict[tuple[str, int, str], torch.Tensor] = {}
        hidden = tokens
        for layer in range(max(self.program.methods) + 1):
            hidden = self.layer(value.rows.rows, layer, hidden)
            for row, tensor in zip(value.rows.rows, hidden, strict=True):
                outputs[row.identity.request_id, row.token_offset, f"attention {layer}"] = tensor.detach().clone()
        return outputs

    def paged(self, rows: Sequence[PackedPhaseRow], layer: int, q: torch.Tensor, read: "_CheckRead") -> "PagedKV":
        """Stage request-local dictionaries as fixed-size pages, including all current K/V writes."""
        from pllm.execution.attention import CHECK_PAGE_SIZE, PagedKV

        kv_heads, dim = self.probe.kv_heads, self.probe.head_dim
        requests = tuple(dict.fromkeys(row.identity.request_id for row in rows))
        groups = tuple(tuple(row for row in rows if row.identity.request_id == request) for request in requests)
        lengths = tuple(max(row.token_offset for row in group) + 1 for group in groups)
        starts = [0]
        for group in groups:
            starts.append(starts[-1] + len(group))
        layout = storage_layout(self.program.layouts, layer)
        for row, query in zip(rows, q, strict=True):
            if layout.layer != layer:
                continue
            key = query[:kv_heads].clone()
            val = (key.float() * 0.7 + layer * 0.01).to(q.dtype)
            self.cache[row.identity.request_id, layer, row.token_offset] = key, val
        domain = self.probe.causal_domain
        assert domain is not None
        pages = (domain.positions + CHECK_PAGE_SIZE - 1) // CHECK_PAGE_SIZE
        keys, values = (q.new_zeros((len(groups) * pages, CHECK_PAGE_SIZE, kv_heads, dim)) for _ in (0, 1))
        for request, name in enumerate(requests):
            for position in range(lengths[request]):
                cached = self.cache.get((name, layout.layer, position))
                if cached is not None:
                    block, offset = request * pages + position // CHECK_PAGE_SIZE, position % CHECK_PAGE_SIZE
                    keys[block, offset], values[block, offset] = cached
        return PagedKV(
            keys,
            values,
            torch.arange(len(groups) * pages, dtype=torch.int32).reshape(len(groups), pages),
            torch.tensor(lengths, dtype=torch.int32),
            torch.tensor(starts, dtype=torch.int32),
            len(rows),
            read,
            layout.window,
            torch.tensor(tuple(group[0].prompt_length for group in groups)) if layout.sink else None,
        )

    def layer(self, rows: Sequence[PackedPhaseRow], layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """Execute the actual method over the whole scheduled batch, so cross-request reads fail isolation."""
        from pllm.execution.attention import DenseAttention
        from pllm.execution.model_sites import DepthInput, StateInput

        heads, dim = self.probe.attention_heads, self.probe.head_dim
        width = hidden.shape[-1]
        q = hidden.repeat(1, (heads * dim + width - 1) // width)[:, : heads * dim].reshape(len(rows), heads, dim)
        trace = _CheckRead()
        kv = self.paged(rows, layer, q, trace)
        trace.kernel = DenseAttention(kv, dim**-0.5, int(kv.seq_lens.max()))
        method = self.program.methods.get(layer)
        if method is None:
            out = kv.attend(q)[0].flatten(1)
            trace.kernel = None
            return out
        kwargs = {
            arg: self.parameters[source]
            if isinstance(source, StateInput)
            else layer
            if isinstance(source, DepthInput)
            else self.program.facts[source][: len(rows)]
            for arg, source in method.inputs
        }
        try:
            result = method.function(q, kv, **kwargs)
            if self.device_equivalence:
                _compare_attention_device(method.function, q, kv, kwargs, result)
            out, aux = result if isinstance(result, tuple) else (result, None)
            for source, target in self.program.facts.items():
                if source.layer == layer:
                    tensor = trace.lse if source.fact == "lse" else aux
                    assert tensor is not None
                    target[: len(rows)].copy_(tensor)
            return out.flatten(1)
        finally:
            trace.kernel = None


class _CheckRead:
    """Retain the first native read's LSE for the check decoder's fact writer."""

    def __init__(self) -> None:
        self.kernel: "DenseAttention | None" = None
        self.lse: torch.Tensor | None = None

    def __call__(self, q: torch.Tensor, window: int, topk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.kernel is not None
        result = self.kernel(q, window, topk)
        if self.lse is None:
            self.lse = result[1]
        return result


def _compare_attention_device(
    function: "Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]",
    q: torch.Tensor,
    kv: "PagedKV",
    kwargs: "dict[str, Argument]",
    expected: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Run user kernels on identical CPU/CUDA paged inputs, under sync-debug."""
    from pllm.execution.attention import DenseAttention, PagedKV
    from pllm.execution.module_graph import ModuleCall

    fields = tuple(
        value.to("cuda") for value in (kv.key_cache, kv.value_cache, kv.block_table, kv.seq_lens, kv.query_start)
    )
    kernel = _CheckRead()
    keys, values, table, lengths, starts = fields
    gpu = PagedKV(
        keys,
        values,
        table,
        lengths,
        starts,
        kv.query_count,
        kernel,
        kv.storage_window,
        kv.prompt_lens.to("cuda") if kv.prompt_lens is not None else None,
    )
    kernel.kernel = DenseAttention(gpu, q.shape[-1] ** -0.5, int(kv.seq_lens.max()))
    arguments = {
        name: value.to(torch.device("cuda")) if isinstance(value, (torch.Tensor, ModuleCall)) else value
        for name, value in kwargs.items()
    }
    gpu_q = q.to("cuda")
    mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = function(gpu_q, gpu, **arguments)
    finally:
        torch.cuda.set_sync_debug_mode(mode)
        kernel.kernel = None
    torch.testing.assert_close(actual, expected, check_device=False)
