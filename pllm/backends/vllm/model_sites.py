"""@pllm.custom_residual on vLLM: hooks on the norms between sublayers and on the value projections.

Every admitted family adds each sublayer's output into the residual stream inside the next norm, a fused in-place
kernel that returns (normalized branch input, stream). A hook on that norm completes the sublayer with the owning
method's write instead of the plain add, keeps copies the methods read later in the forward, forms the next
branch input with a read method, and normalizes with the kernel the model uses. Boundaries nothing applies to are
not wrapped. The value hook replaces a layer's V slice of its qkv projection before attention reads and caches it.
Used by VllmSession; a custom forward points `rows` at the packed rows of the layers it runs. Stateless hooks enter
vLLM's per-capacity model graphs through one opaque custom op per hook. State fields and the row-to-request table
are staged before replay; the same eager kernels are captured, preserving arithmetic and leaving capacity graphs,
their allocation pool and memory budget with the upstream graph runner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.architecture import Decoder, LinearOutput, NormForward, hidden_binding, sublayer_norms
from pllm.backends.vllm.hidden_taps import HiddenTaps, Tap
from pllm.contracts.hidden_locator import HiddenSite
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.execution.cuda_stream import upload
from pllm.execution.model_sites import (
    EarlierInput,
    PatchInput,
    Place,
    ResidualRewrite,
    ResultInput,
    StateInput,
    position,
)
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.site_memory import SiteMemorySchedule
from pllm.execution.state_execution import DataT, StateTaskContext

Forward = Callable[..., Tensor | tuple[Tensor, ...]] | Callable[[Tensor], LinearOutput]

_SITES: dict[int, "ModelSites"] = {}


@torch.library.custom_op("pllm::model_site", mutates_args={"inputs", "outputs", "buffers"})
def model_site(inputs: list[Tensor], outputs: list[Tensor], buffers: list[Tensor], hook: int) -> None:
    """Run a bound hook inside vLLM's capture, with stable state and same-forward scratch ports."""
    _SITES[buffers[0].data_ptr()].hooks[hook].emit(inputs, outputs)


def _model_site_meta(inputs: list[Tensor], outputs: list[Tensor], buffers: list[Tensor], hook: int) -> None:
    return None


model_site.register_fake(_model_site_meta)  # pyright: ignore[reportUnknownMemberType]


class ModelSites:
    """One session's installed rewrites, the step's copies of the states they read and the earlier values of the
    forward they read. Every copy lives in a stable buffer, so custom-forward graph replays see the step's values."""

    def __init__(
        self,
        decoder: Decoder,
        rewrites: tuple[ResidualRewrite, ...],
        taps: HiddenTaps | None,
        activation: VllmActivationPatch | None,
        rows: int,
        requests: int,
        arena: PageBundleArena,
        *,
        graphs: bool = False,
    ) -> None:
        body = decoder.model.model
        count, weight = len(body.layers), body.norm.weight
        owners = {place: rewrite for rewrite in rewrites for place in rewrite.places}
        earlier = {source for rewrite in rewrites for source in rewrite.earlier}
        self.streams, self.hidden = max((rewrite.streams for rewrite in rewrites), default=1), weight.shape[0]
        self.taps = taps
        self.active: Tensor | None = None
        # The packed rows of the layer call a custom forward is running (on the device and the host); None while
        # vLLM runs the whole step.
        self.rows: Tensor | None = None
        self.packed: tuple[int, ...] = ()
        self._activation = activation
        self.count = 0  # rows of the current hook call (vLLM may pad the step's rows)
        self._requests_of = torch.zeros(rows, dtype=torch.long, device=weight.device)
        carries = {rewrite.memory.carry for rewrite in rewrites if rewrite.memory is not None}
        self._sources = tuple(
            dict.fromkeys(
                source
                for rewrite in rewrites
                for _, source in (*rewrite.inputs, *rewrite.read_inputs)
                if isinstance(source, StateInput) and source not in carries
            )
        )
        self._memories = {
            rewrite.name: SiteMemorySchedule(rewrite, arena, requests, rows)
            for rewrite in rewrites
            if rewrite.memory is not None
        }
        self._result_sources = tuple(dict.fromkeys(source for rewrite in rewrites for source in rewrite.results))
        self._results = {
            source: torch.zeros((rows, *source.shape), device=weight.device, dtype=source.dtype)
            for source in self._result_sources
        }
        self._prefetch = {
            source: torch.cuda.Event(external=True)
            for source in self._result_sources
            if any(
                task.lane == "side"
                and task.read_origin != "latest"
                and arena.plan.semantic_graph.tasks[task.task_id].symbol == source.task
                for task in arena.plan.schedule.tasks
            )
        }
        # An uncreated event makes wait_event a no-op, so create it before model graph capture.
        for ready in self._prefetch.values():
            ready.record()
        self._states: dict[StateInput, Tensor] = {}
        for source in self._sources:
            field = source.view.read(arena.field_pool(source.resource, source.view.field), pooled=True)[0]
            leading = (requests,) if source.request else ()
            self._states[source] = field.new_zeros((*leading, *field.shape))
        self.hooks: list[_Boundary | _Value] = []
        self._installed: list[tuple[torch.nn.Module, Forward]] = []
        self._ends: dict[int, _Boundary] = {}
        values = {
            layer: decoder.value_rows(layer)
            for layer in sorted(
                {place[1] for place in owners if place[0] == "value"}
                | {source.layer for source in earlier if source.kind == "value"}
            )
        }
        # Each earlier value in the shape methods read: value rows [rows, kv_heads, head_dim] or the stream.
        stream_shape = (self.streams, self.hidden) if self.streams > 1 else (self.hidden,)
        self._earlier = {
            source: weight.new_zeros(
                (rows, values[source.layer][2] // values[source.layer][3], values[source.layer][3])
                if source.kind == "value"
                else (rows, *stream_shape)
            )
            for source in earlier
        }
        for index, (norm, kernel) in enumerate(zip(sublayer_norms(decoder.model), decoder.norms, strict=True)):
            layer, attention = divmod(index, 2)
            before: Place | None = ("attention", layer) if attention else ("mlp", layer - 1) if layer else None
            after: Place | None = None if layer == count else ("mlp", layer) if attention else ("attention", layer)
            reader = owners.get(after) if after is not None else None
            path = (
                "model.norm"
                if layer == count
                else f"model.layers.{layer}.{'post_attention_layernorm' if attention else 'input_layernorm'}"
            )
            boundary = _Boundary(
                self,
                kernel,
                before,
                None if before is None else owners.get(before),
                after,
                reader if reader is not None and reader.read is not None else None,
                owners.get(("final", count)) if layer == count else None,
                tuple(
                    source
                    for source in earlier
                    if source.kind != "value" and before is not None and source.position == position(before)
                ),
                (None, None) if taps is None else (taps.taps.get((path, 0)), taps.taps.get((path, 1))),
            )
            if before is not None and before[0] == "mlp":
                self._ends[before[1]] = boundary
            if (
                boundary.owner
                or boundary.reader
                or boundary.final
                or boundary.keeps
                or any(boundary.taps)
                or self.streams > 1
            ):
                boundary.index = len(self.hooks)
                self._install(norm, boundary.graph_forward if graphs else boundary)
                self.hooks.append(boundary)
        for layer, (module, start, width, head_dim) in values.items():
            keeps = EarlierInput("value", layer)
            hook = _Value(
                self,
                module.forward,
                layer,
                owners.get(("value", layer)),
                keeps if keeps in earlier else None,
                start,
                width,
                head_dim,
            )
            hook.index = len(self.hooks)
            self._install(module, hook.graph_forward if graphs else hook)
            self.hooks.append(hook)
        if graphs:
            _SITES[self._requests_of.data_ptr()] = self

    @property
    def buffers(self) -> list[Tensor]:
        """The uploaded row-to-request table, staged fields and same-forward scratch used by captured hooks."""
        return [
            self._requests_of,
            *self._states.values(),
            *self._results.values(),
            *self._earlier.values(),
            *(buffer for memory in self._memories.values() for buffer in memory.buffers),
        ]

    def run_hook(self, hook: int, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        """Keep orchestration opaque to Dynamo; upstream owns each capacity's graph and its allocations."""
        model_site(inputs, outputs, self.buffers, hook)

    def _install(self, module: torch.nn.Module, hook: Forward) -> None:
        self._installed.append((module, module.forward))
        module.forward = hook

    def prepare(
        self, rows: PackedPhaseRowBatch, context: StateTaskContext[DataT], directory: RequestStateDirectory
    ) -> None:
        """Copy the step's version of every state input, and each packed row's request, into the stable buffers.

        Request state is read at each request's version; instance state at the step's newest version, the one value
        every row of the forward reads.
        """
        for source in self._result_sources:
            value = context.results[source.task, "value"]
            if source in self._prefetch:
                assert context.prefetch_stream is not None
                with torch.cuda.stream(context.prefetch_stream):
                    self._results[source][: len(value)].copy_(value)
                    self._prefetch[source].record(context.prefetch_stream)
            else:
                self._results[source][: len(value)].copy_(value)
        requests = tuple(dict.fromkeys(row.identity for row in rows.rows))
        # Each packed row's index into `requests`, the order request state is staged in below.
        index = {identity: position for position, identity in enumerate(requests)}
        positions = tuple(index[row.identity] for row in rows.rows)
        self._requests_of[: len(positions)].copy_(upload(positions, self._requests_of.device))
        for source in self._sources:
            versions = (
                context.read_versions(requests, source.resource, directory)
                if source.request
                else context.read_versions(requests[:1], source.resource, directory, latest=True)
            )
            for index, (pin, ref) in enumerate(versions):
                field = source.view.read(context.arena.borrow_field(pin, ref, source.view.field))
                (self._states[source][index] if source.request else self._states[source]).copy_(field)
        for memory in self._memories.values():
            memory.prepare(rows, context)

    def remember(self, rewrite: ResidualRewrite, stream: Tensor, branch: Tensor) -> Tensor:
        """Advance private PTTS memory through device-counted request buckets."""
        memory = self._memories[rewrite.name]
        invariants = tuple(self.state(source) for source in memory.memory.invariants(rewrite))
        return memory(stream, branch, invariants)

    def _rows_of(self, values: Tensor) -> Tensor:
        """The current call's rows of a per-row buffer."""
        return values[: self.count] if self.rows is None else values.index_select(0, self.rows)

    def state(self, source: StateInput) -> Tensor:
        staged = self._states[source]
        return staged.index_select(0, self._rows_of(self._requests_of)) if source.request else staged

    def result(self, source: ResultInput) -> Tensor:
        if source in self._prefetch:
            torch.cuda.current_stream().wait_event(self._prefetch[source])
        return self._rows_of(self._results[source])

    def earlier(self, source: EarlierInput) -> Tensor:
        return self._rows_of(self._earlier[source])

    def patch(self, source: PatchInput, layer: int) -> Callable[[Tensor], Tensor]:
        assert self._activation is not None
        return self._activation.patch(source.resource, layer, self.rows, self.packed)

    def keep(self, source: EarlierInput, values: Tensor) -> None:
        """Copy the current rows' value at an earlier place for methods later in this forward."""
        buffer = self._earlier[source]
        shaped = values.reshape(len(values), *buffer.shape[1:])
        if self.rows is None:
            buffer[: len(shaped)].copy_(shaped)
        else:
            if self.active is not None:
                shaped = torch.where(self.active, shaped, buffer.index_select(0, self.rows))
            buffer.index_copy_(0, self.rows, shaped)

    def frozen(self, start: int, stop: int, hidden: Tensor) -> None:
        """Seed skipped layers' readouts with the frozen stream before computed rows overwrite them."""
        for source in self._earlier:
            if source.kind != "value" and start <= source.layer < stop:
                self.keep(source, hidden)
        if self.taps is not None:
            for layer in range(start, stop):
                for site in (HiddenSite.RESIDUAL, HiddenSite.ATTENTION_RESIDUAL):
                    tap = self.taps.taps.get(hidden_binding(layer, site, len(self._ends)).tap)
                    if tap is not None:
                        tap.copy(hidden)

    def stream_after(self, layer: int, branch: Tensor, stream: Tensor) -> Tensor:
        """The stream after layer `layer`, for a custom forward that adds the last layer of a slice itself."""
        self.count = len(branch)
        boundary = self._ends[layer]
        completed = boundary.complete(branch, stream)
        boundary.keep_stream(completed)
        return completed

    def close(self) -> None:
        for module, forward in reversed(self._installed):
            module.forward = forward
        self._installed.clear()
        _SITES.pop(self._requests_of.data_ptr(), None)


@dataclass
class _Boundary:
    """One norm between sublayers: completes the sublayer `before` it, normalizes the input of the one `after` it."""

    sites: ModelSites
    # The norm's own forward, past any tap wrapping the module; this hook copies tapped rows itself.
    norm: NormForward
    before: Place | None
    owner: ResidualRewrite | None
    after: Place | None
    reader: ResidualRewrite | None
    final: ResidualRewrite | None
    keeps: tuple[EarlierInput, ...]
    # The taps of the module's normalized output and residual stream, which this hook copies itself.
    taps: tuple[Tap | None, Tap | None]
    index: int = 0

    def graph_forward(self, branch: Tensor, stream: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        """Expose fresh outputs with the shapes vLLM's compiled model carries between boundaries."""
        normed = torch.empty_like(branch)
        if stream is None:
            self.sites.run_hook(self.index, [branch], [normed])
            return normed
        width = self.sites.hidden * (self.sites.streams if self.final is None else 1)
        residual = branch.new_empty((branch.shape[0], width))
        self.sites.run_hook(self.index, [branch, stream], [normed, residual])
        return normed, residual

    def emit(self, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        """Materialize the hook's results into the graph's declared outputs."""
        if len(inputs) == 1:
            result = self(inputs[0])
            assert isinstance(result, Tensor)
            outputs[0].copy_(result)
        else:
            result = self(inputs[0], inputs[1])
            assert isinstance(result, tuple)
            outputs[0].copy_(result[0])
            outputs[1].copy_(result[1])

    def streams_of(self, stream: Tensor) -> Tensor:
        """The stream as [rows, n, d] with n residual streams. vLLM passes the embedding to the first layer's
        post-attention norm, so one stream there is expanded too."""
        count = self.sites.streams
        if count == 1:
            return stream
        if stream.shape[-1] == self.sites.hidden:
            stream = stream.unsqueeze(1).expand(-1, count, -1)
        return stream.reshape(len(stream), count, -1)

    def complete(self, branch: Tensor, stream: Tensor) -> Tensor:
        """The stream after the sublayer `before`: the owner's write, or the plain add."""
        if self.owner is None:
            return stream + branch
        assert self.before is not None
        expected = self.streams_of(stream)
        written = (
            self.owner.call_write(self.before, self.sites, expected, branch)
            if self.owner.memory is None
            else self.sites.remember(self.owner, stream, branch)
        )
        if written.shape != expected.shape or written.dtype != stream.dtype:
            shape = ", ".join(map(str, ("rows", *expected.shape[1:])))
            raise ValueError(
                f"{self.owner.name}: return the stream as [{shape}] {stream.dtype} (the model dtype); compute in "
                "float32 inside the method and cast at the end"
            )
        return written.reshape(len(stream), -1)

    def __call__(self, branch: Tensor, stream: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        self.sites.count = len(branch)
        if stream is None:
            # The model's first norm, a custom forward entering mid-model, or a custom forward normalizing its result.
            x = branch if self.before is not None else self.streams_of(branch).reshape(len(branch), -1)
        elif self.owner is None and self.sites.streams == 1 and self.reader is None and self.final is None:
            normed, x = self.norm(branch, stream)
            self._keep(normed, x)
            return normed, x
        else:
            x = self.complete(branch, stream)
        if self.final is not None:
            x = self._final(self.final, x)
        u = x if self.reader is None else self._read(self.reader, x)
        # The fused kernel over (u, 0) normalizes exactly u; the first norm has no residual and uses the plain one.
        normed = self.norm(u) if stream is None else self.norm(u.clone(), torch.zeros_like(u))[0]
        if stream is None:
            normalized = self.taps[0]
            if normalized is not None:
                normalized.copy(normed)
        else:
            self._keep(normed, x)
        return normed if stream is None else (normed, x)

    def _read(self, reader: ResidualRewrite, stream: Tensor) -> Tensor:
        assert self.after is not None
        branch = reader.call_read(self.after, self.sites, self.streams_of(stream))
        if branch.shape != (len(stream), self.sites.hidden) or branch.dtype != stream.dtype:
            raise ValueError(
                f"{reader.name}: the read method returns one branch input [rows, {self.sites.hidden}] {stream.dtype}"
            )
        return branch

    def _final(self, final: ResidualRewrite, stream: Tensor) -> Tensor:
        collapsed = final.call_write(final.places[0], self.sites, self.streams_of(stream))
        if collapsed.shape != (len(stream), self.sites.hidden) or collapsed.dtype != stream.dtype:
            raise ValueError(
                f"{final.name}: return one stream [rows, {self.sites.hidden}] {stream.dtype} (the model dtype) from the "
                "final site"
            )
        return collapsed

    def keep_stream(self, stream: Tensor) -> None:
        """Publish a completed stream even when a slice ends before the next norm runs."""
        for source in self.keeps:
            self.sites.keep(source, stream)
        residual = self.taps[1]
        if residual is not None:
            residual.copy(stream)

    def _keep(self, normed: Tensor, stream: Tensor) -> None:
        self.keep_stream(stream)
        normalized = self.taps[0]
        if normalized is not None:
            normalized.copy(normed)


@dataclass
class _Value:
    """A layer's qkv projection: the owner rewrites its value rows before attention reads and caches them."""

    sites: ModelSites
    native: Callable[[Tensor], LinearOutput]
    layer: int
    owner: ResidualRewrite | None
    keeps: EarlierInput | None
    start: int
    width: int
    head_dim: int
    index: int = 0

    def __call__(self, hidden: Tensor) -> LinearOutput:
        output = self.native(hidden)
        projected = output if isinstance(output, Tensor) else output[0]
        self.emit([projected], [])
        return output

    def graph_forward(self, hidden: Tensor) -> LinearOutput:
        """Keep the native projection compiled and capture just its value rewrite as a model-site op."""
        output = self.native(hidden)
        projected = output if isinstance(output, Tensor) else output[0]
        self.sites.run_hook(self.index, [projected], [])
        return output

    def emit(self, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        """Rewrite and retain the value slice in place before attention reads it."""
        (projected,) = inputs
        self.sites.count = len(projected)
        value = projected[:, self.start : self.start + self.width].view(len(projected), -1, self.head_dim)
        if self.owner is not None:
            written = self.owner.call_write(("value", self.layer), self.sites, value)
            if written.shape != value.shape or written.dtype != value.dtype:
                raise ValueError(
                    f"{self.owner.name}: return value rows [rows, {value.shape[1]}, {self.head_dim}] {value.dtype}"
                )
            value.copy_(written)
        if self.keeps is not None:
            self.sites.keep(self.keeps, value)
