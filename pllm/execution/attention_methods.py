"""Declare attention methods and snapshot their facts for ordinary tensor-task readers.

TensorBinding owns this program; a backend installs its reads and CPU checks replay
it on seeded paged KV. Fact snapshots use the normal task-result lifetime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

import torch

from pllm.contracts.kv_layout import KVLayout, storage_layout
from pllm.contracts.semantic_graph import PackedSamplerRowsInputSource, TaskResultInputSource, TaskSymbol
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.attention import CHECK_PAGE_SIZE, PAGE_SIZE, PagedKV
from pllm.execution.model_sites import DepthInput, StateInput
from pllm.execution.module_graph import ModuleCall
from pllm.execution.tensor_frame import TensorFrame, TensorPort, result_port
from pllm.execution.tensor_program import TensorBatch

if TYPE_CHECKING:
    from pllm.execution.compile_facts import BackendCompileFacts
    from pllm.execution.tensor_binding import TensorBinding

Fact = Literal["aux", "lse"]
Argument = torch.Tensor | ModuleCall | int


@dataclass(frozen=True)
class AttentionFact:
    """A same-token fact computed by a strictly earlier attention layer."""

    layer: int
    fact: Fact


@dataclass(frozen=True)
class AttentionMethod:
    """A frozen user read and its layer and marker declarations."""

    name: str
    layers: tuple[int, ...]
    function: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
    inputs: tuple[tuple[str, StateInput | DepthInput | AttentionFact], ...]


@dataclass(frozen=True)
class AttentionCost:
    """Shape-derived attention work per token, including all auxiliary reads."""

    layer: int
    method: str
    reads: tuple[str, ...]
    flops_per_token: int
    kv_bytes_per_token: int


class TraceRead:
    """Shape-only native reads for planning and explain's per-layer cost rows."""

    def __init__(self, context: int) -> None:
        self.context = context
        self.reads: list[str] = []
        self.tokens = 0
        self.head_tokens = 0
        self.shared_tokens = 0
        self.sink = False

    def __call__(self, q: torch.Tensor, window: int, topk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if window < 0:
            raise ValueError("kv.attend needs window >= 0")
        if topk is not None and (
            topk.dtype not in (torch.int32, torch.int64)
            or topk.ndim not in (2, 3)
            or topk.shape[0] != q.shape[0]
            or topk.ndim == 3
            and topk.shape[1] != q.shape[1]
            or topk.shape[-1] < 1
        ):
            raise ValueError("kv.attend topk needs integer positions [rows, k] or [rows, heads, k], padded with -1")
        count = min(self.context, window) if window else self.context
        if topk is not None:
            count = min(count, topk.shape[-1])
        self.tokens += count
        if topk is not None and topk.ndim == 3:
            self.head_tokens += count * q.shape[1]
        else:
            self.shared_tokens += count
        self.reads.append(
            f"topk <= {count}/head"
            if topk is not None
            else f"window {window}"
            if window
            else "prompt sink + window (gathered)"
            if self.sink
            else "all positions natively"
        )
        return torch.empty_like(q), torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)


def identity(value: torch.Tensor) -> torch.Tensor:
    """Snapshot a gathered fact through the framework's ordinary task output."""
    return value


def native(q: torch.Tensor, kv: PagedKV) -> torch.Tensor:
    """Default attention installed when a reader asks only for an LSE fact."""
    return kv.attend(q)[0]


class AttentionProgram:
    """Attention declarations, explicit per-step scratch and sampled fact snapshots."""

    def __init__(self) -> None:
        self.methods: dict[int, AttentionMethod] = {}
        self.facts: dict[AttentionFact, torch.Tensor] = {}
        self.costs: tuple[AttentionCost, ...] = ()
        self.layouts: tuple[KVLayout, ...] = ()
        self.sample_rows = torch.empty(0, dtype=torch.long, device="meta")
        self.prompt_lens = torch.empty(0, dtype=torch.long, device="meta")

    def add(self, method: AttentionMethod) -> None:
        """Reject overlapping owners and forward references at the authoring boundary."""
        for layer in method.layers:
            if layer in self.methods:
                raise ValueError(f"layer {layer} already has custom attention `{self.methods[layer].name}`")
            for _, source in method.inputs:
                if isinstance(source, AttentionFact) and source.layer >= layer:
                    raise ValueError(
                        f"{method.name}: pllm.attention({source.layer}, {source.fact!r}) needs a strictly earlier layer"
                    )
                if isinstance(source, StateInput) and source.request:
                    raise ValueError(
                        "@pllm.custom_attention reads InstanceState; use stateless selection or instance parameters"
                    )
            self.methods[layer] = method

    def port(self, binding: TensorBinding, source: AttentionFact) -> TensorPort:
        """Snapshot this layer's sampled facts once, before tensor-task consumers run."""
        from pllm.execution.declaration import When
        from pllm.execution.row_task_binding import declare_tensor

        symbol = TaskSymbol(f"__attention_{source.layer}_{source.fact}")
        if symbol not in binding.tensor_tasks:
            self.facts[source] = torch.empty(0, device="meta")
            declare_tensor(
                binding,
                str(symbol),
                inputs={"value": TensorPort(PackedSamplerRowsInputSource(), partial(self.read, source))},
                when=When.AFTER_FORWARD,
            )(identity)
        return result_port(TaskResultInputSource(symbol, "value"))

    def read(self, source: AttentionFact, data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        """Gather the fact for each sampled row into the task's owned result."""
        indices = data.table.index.sampler
        value = self.facts[source]
        if indices.is_meta:
            return torch.empty((len(indices), *value.shape[1:]), dtype=value.dtype, device="meta")
        return value.index_select(0, self.sample_rows.index_select(0, indices.clamp_min(0)))

    def plan(self, binding: TensorBinding, facts: BackendCompileFacts) -> None:
        """Trace methods in forward order to infer fact shapes and attention work."""
        for method in tuple(self.methods.values()):
            for _, source in method.inputs:
                if isinstance(source, AttentionFact):
                    self.facts.setdefault(source, torch.empty(0, device="meta"))
        for source in tuple(self.facts):
            if source.layer not in self.methods:
                if source.fact == "aux":
                    raise ValueError(
                        f"pllm.attention({source.layer}, 'aux') needs custom attention returning (output, aux)"
                    )
                self.methods[source.layer] = AttentionMethod("native", (source.layer,), native, ())
        probe = facts.probe
        rows, heads, dim = probe.max_packed_rows, probe.attention_heads, probe.head_dim
        if not self.methods:
            return
        if not heads or not dim:
            raise ValueError("@pllm.custom_attention needs dense GQA/MHA model geometry")
        context = probe.causal_domain.positions if probe.causal_domain is not None else 1
        costs: list[AttentionCost] = []
        for layer, method in sorted(self.methods.items()):
            if not 0 <= layer < binding.dimensions.layer_count:
                raise ValueError(f"pllm.attention layer {layer} is outside the model")
            trace = TraceRead(context)
            q, kv, kwargs = attention_example(binding, facts, self, layer, rows, trace)
            result = method.function(q, kv, **kwargs)
            output, aux = result if isinstance(result, tuple) else (result, None)
            if output.shape != q.shape or output.dtype != q.dtype:
                raise ValueError(f"{method.name}: return attention with q's shape and model dtype")
            for source in tuple(self.facts):
                if source.layer != layer:
                    continue
                if source.fact == "lse":
                    if not trace.reads:
                        raise ValueError(f"{method.name}: call kv.attend to expose pllm.attention({layer}, 'lse')")
                    value = q.new_empty(q.shape[:2], dtype=torch.float32)
                else:
                    if aux is None or aux.ndim < 1 or aux.shape[0] != rows:
                        raise ValueError(f"{method.name}: return (attention, aux) with aux shaped [rows, ...]")
                    value = aux
                self.facts[source] = value
            costs.append(
                AttentionCost(
                    layer,
                    method.name,
                    tuple(trace.reads),
                    4 * heads * dim * trace.tokens,
                    2 * dim * q.element_size() * (probe.kv_heads * trace.shared_tokens + trace.head_tokens),
                )
            )
        self.costs = tuple(costs)
        self.sample_rows = torch.empty(rows, dtype=torch.long, device="meta")
        self.prompt_lens = torch.empty(
            rows if any(layout.sink for layout in self.layouts) else 0, dtype=torch.long, device="meta"
        )

    @property
    def storage_bytes(self) -> int:
        """Persistent fact and sampled-index scratch, charged before planning."""
        return sum(
            value.numel() * value.element_size() for value in (*self.facts.values(), self.sample_rows, self.prompt_lens)
        )

    def bind(self, device: torch.device) -> None:
        """Allocate scratch once; fact snapshots are retained by ordinary task results."""
        self.facts = {
            source: torch.zeros(value.shape, dtype=value.dtype, device=device) for source, value in self.facts.items()
        }
        self.sample_rows = torch.zeros(self.sample_rows.shape, dtype=torch.long, device=device)
        self.prompt_lens = torch.zeros(self.prompt_lens.shape, dtype=torch.long, device=device)


def attention_example(
    binding: TensorBinding,
    facts: BackendCompileFacts,
    program: AttentionProgram,
    layer: int,
    rows: int,
    trace: TraceRead,
) -> tuple[torch.Tensor, PagedKV, dict[str, Argument]]:
    """Build meta operands from declared input types; instance fields never gain or lose a row axis."""
    from pllm.execution.task_memory import site_state

    probe = facts.probe
    q = torch.empty((rows, probe.attention_heads, probe.head_dim), dtype=probe.tensor_dtype.torch, device="meta")
    size = CHECK_PAGE_SIZE if probe.device_kind is StateDeviceKind.CPU else PAGE_SIZE
    pages = (trace.context + size - 1) // size
    cache = q.new_empty((rows * pages, size, probe.kv_heads, probe.head_dim))
    integer = torch.empty(rows, dtype=torch.int32, device="meta")
    layout = storage_layout(program.layouts, layer)
    trace.sink = layout.sink
    kv = PagedKV(
        cache,
        cache,
        integer.new_empty((rows, pages)),
        integer,
        integer.new_empty(rows + 1),
        integer,
        trace,
        layout.window,
        integer if layout.sink else None,
    )
    arguments = {
        name: site_state(binding, source, None, None)
        if isinstance(source, StateInput)
        else layer
        if isinstance(source, DepthInput)
        else torch.empty((rows, *program.facts[source].shape[1:]), dtype=program.facts[source].dtype, device="meta")
        for name, source in program.methods[layer].inputs
    }
    return q, kv, arguments
