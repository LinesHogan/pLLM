"""@pllm.custom_residual on vLLM: hooks on the norms between sublayers and on the value projections.

Every admitted family adds each sublayer's output into the residual stream inside the next norm, a fused in-place
kernel that returns (normalized branch input, stream). A hook on that norm completes the sublayer with the owning
method's write instead of the plain add, keeps copies the methods read later in the forward, forms the next
branch input with a read method, and normalizes with the kernel the model uses. Boundaries nothing applies to are
not wrapped. The value hook replaces a layer's V slice of its qkv projection before attention reads and caches it.
Used by VllmSession; a custom forward points `rows` at the packed rows of the layers it runs. Stateless hooks enter
vLLM's per-capacity model graphs through one opaque custom op per hook. State fields and the row-to-request table
are staged before replay; compiled site bodies are captured, leaving capacity graphs,
their allocation pool and memory budget with the upstream graph runner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from vllm.model_executor.layers.linear import LinearBase

from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.architecture import (
    Decoder,
    LinearOutput,
    NormForward,
    hidden_binding,
    post_attention_norm_name,
    sublayer_norms,
)
from pllm.backends.vllm.hidden_taps import HiddenTaps, Tap
from pllm.backends.vllm.site_body import SiteBody, compile_site_method
from pllm.contracts.hidden_locator import HiddenSite
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.execution.attention_methods import AttentionProgram
from pllm.execution.cuda_stream import upload
from pllm.execution.model_sites import (
    EarlierInput,
    PatchInput,
    Place,
    ResidualRewrite,
    ResultInput,
    StateInput,
    position,
    state_inputs,
)
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.site_memory import SiteMemorySchedule
from pllm.execution.state_execution import StateTaskContext

Forward = Callable[..., Tensor | tuple[Tensor, ...]] | Callable[[Tensor], LinearOutput]

# Graph-mode sessions by the address of their row-to-request table; see model_site.
_SITES: dict[int, "ModelSites"] = {}


@torch.library.custom_op("pllm::model_site", mutates_args={"outputs", "writes"})
def model_site(
    inputs: list[Tensor], outputs: list[Tensor], reads: list[Tensor], writes: list[Tensor], hook: int
) -> None:
    """Run a bound hook inside vLLM's capture, with stable state and same-forward scratch ports.

    A custom op takes only tensors and scalars, and vLLM's compiled (and AOT-cached) model graph bakes scalars in,
    so the session travels as `reads[0]`, its row table: a runtime graph input whose address finds the session
    without specializing the graph on one. vLLM routes its own opaque ops through a name registry the same way
    (forward_context.no_compile_layers). A session registers in ModelSites.__init__ and leaves in close(); it
    owns the table meanwhile, so the address cannot be reused while registered.
    """
    _SITES[reads[0].data_ptr()].hooks[hook].emit(inputs, outputs)


def _model_site_meta(
    inputs: list[Tensor], outputs: list[Tensor], reads: list[Tensor], writes: list[Tensor], hook: int
) -> None:
    return None


model_site.register_fake(_model_site_meta)  # pyright: ignore[reportUnknownMemberType]


def _prefetch_events(sources: tuple[ResultInput, ...], arena: PageBundleArena) -> dict[ResultInput, torch.cuda.Event]:
    """A readiness event per task result a side-lane task produces from an earlier version."""
    events = {
        source: torch.cuda.Event(external=True)
        for source in sources
        if any(
            task.lane == "side"
            and task.read_origin != "latest"
            and arena.plan.semantic_graph.tasks[task.task_id].symbol == source.task
            for task in arena.plan.schedule.tasks
        )
    }
    # An uncreated event makes wait_event a no-op, so create it before model graph capture.
    for ready in events.values():
        ready.record()
    return events


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
        attention_program: AttentionProgram | None = None,
    ) -> None:
        weight = decoder.model.model.norm.weight
        owners = {place: rewrite for rewrite in rewrites for place in rewrite.places}
        earlier = {source for rewrite in rewrites for source in rewrite.earlier}
        self.streams, self.hidden = max((rewrite.streams for rewrite in rewrites), default=1), weight.shape[0]
        self.post_attn_norm = post_attention_norm_name(type(decoder.model).__name__)
        self.taps = taps
        self.active: Tensor | None = None
        # The packed rows of the layer call a custom forward is running (on the device and the host); None while
        # vLLM runs the whole step.
        self.rows: Tensor | None = None
        self._activation = activation
        self.count = 0  # rows of the current hook call (vLLM may pad the step's rows)
        self._requests_of = torch.zeros(rows, dtype=torch.long, device=weight.device)
        self.state_inputs = state_inputs(rewrites, attention_program)
        self._memories = {
            rewrite.name: SiteMemorySchedule(rewrite, arena, requests, rows)
            for rewrite in rewrites
            if rewrite.memory is not None
        }
        self._result_sources = tuple(dict.fromkeys(source for rewrite in rewrites for source in rewrite.results))
        self.result_values = {
            source: torch.zeros((rows, *source.shape), device=weight.device, dtype=source.dtype)
            for source in self._result_sources
        }
        self._prefetch = _prefetch_events(self._result_sources, arena)
        self.state_values: dict[StateInput, Tensor] = {}
        for source in self.state_inputs:
            field = source.view.read(arena.field_pool(source.resource, source.view.field), pooled=True)[0]
            leading = (requests,) if source.request else ()
            self.state_values[source] = field.new_zeros((*leading, *field.shape))
        self.hooks: list[SublayerBoundary | ValueProjection] = []
        self._installed: list[tuple[torch.nn.Module, Forward]] = []
        self._ends: dict[int, SublayerBoundary] = {}
        values = {
            layer: decoder.value_rows(layer)
            for layer in sorted(
                {place[1] for place in owners if place[0] == "value"}
                | {source.layer for source in earlier if source.kind == "value"}
            )
        }
        # Each earlier value in the shape methods read: value rows [rows, kv_heads, head_dim] or the stream.
        stream_shape = (self.streams, self.hidden) if self.streams > 1 else (self.hidden,)
        self.earlier_values = {
            source: weight.new_zeros(
                (rows, values[source.layer][2] // values[source.layer][3], values[source.layer][3])
                if source.kind == "value"
                else (rows, *stream_shape)
            )
            for source in earlier
        }
        self.count = 1
        numerics = arena.plan.timeline.tensor_policy.allow_numerical_transforms and weight.is_cuda
        methods = {
            (rewrite.name, read): compile_site_method(rewrite, read=read)
            for rewrite in rewrites
            if numerics and rewrite.memory is None
            for read in ((False, True) if rewrite.read is not None else (False,))
        }
        self._install_boundaries(decoder, owners, earlier, methods, graphs=graphs)
        for layer, value_rows in values.items():
            self._install_value(layer, value_rows, owners, earlier, methods, weight, graphs=graphs)
        if numerics:
            sample = weight.new_zeros((1, self.hidden))
            for rewrite in rewrites:
                if rewrite.memory is not None:
                    self.remember(rewrite, sample, sample)
        if graphs:
            _SITES[self._requests_of.data_ptr()] = self

    def _install_boundaries(
        self,
        decoder: Decoder,
        owners: dict[Place, ResidualRewrite],
        earlier: set[EarlierInput],
        methods: dict[tuple[str, bool], Callable[..., None]],
        *,
        graphs: bool,
    ) -> None:
        """Hook each norm between sublayers that completes, reads, keeps or taps something."""
        count, weight = len(decoder.model.model.layers), decoder.model.model.norm.weight
        for index, (norm, kernel) in enumerate(zip(sublayer_norms(decoder.model), decoder.norms, strict=True)):
            layer, attention = divmod(index, 2)
            before: Place | None = ("attention", layer) if attention else ("mlp", layer - 1) if layer else None
            after: Place | None = None if layer == count else ("mlp", layer) if attention else ("attention", layer)
            reader = owners.get(after) if after is not None else None
            path = (
                "model.norm"
                if layer == count
                else f"model.layers.{layer}.{self.post_attn_norm if attention else 'input_layernorm'}"
            )
            boundary = SublayerBoundary(
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
                (None, None) if self.taps is None else (self.taps.taps.get((path, 0)), self.taps.taps.get((path, 1))),
            )
            boundary.bind(weight, methods)
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

    def _install_value(
        self,
        layer: int,
        value_rows: tuple[LinearBase, int, int, int],
        owners: dict[Place, ResidualRewrite],
        earlier: set[EarlierInput],
        methods: dict[tuple[str, bool], Callable[..., None]],
        weight: Tensor,
        *,
        graphs: bool,
    ) -> None:
        """Hook a layer's value projection when a method writes or keeps its V rows."""
        module, start, width, head_dim = value_rows
        keeps = EarlierInput("value", layer)
        hook = ValueProjection(
            self,
            module.forward,
            layer,
            owners.get(("value", layer)),
            keeps if keeps in earlier else None,
            start,
            width,
            head_dim,
        )
        hook.bind(weight, methods)
        if hook.owner is None and hook.keeps is None:
            return
        hook.index = len(self.hooks)
        self._install(module, hook.graph_forward if graphs else hook)
        self.hooks.append(hook)

    @property
    def needs_step(self) -> bool:
        """Whether hooks consume versioned state, task results or retained request memory."""
        return bool(self.state_inputs or self._result_sources or self._memories)

    def run_hook(self, hook: int, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        """Keep orchestration opaque to Dynamo; declare only scratch and private pools writable."""
        reads = [self._requests_of, *self.state_values.values(), *self.result_values.values()]
        if self._activation is not None:
            reads.extend(self._activation.buffers)
        writes = [*self.earlier_values.values()]
        for memory in self._memories.values():
            reads.extend((memory.table, memory.rows))
            writes.extend(memory.pools)
        model_site(inputs, outputs, reads, writes, hook)

    def _install(self, module: torch.nn.Module, hook: Forward) -> None:
        self._installed.append((module, module.forward))
        module.forward = hook

    def prepare(self, rows: PackedPhaseRowBatch, context: StateTaskContext, directory: RequestStateDirectory) -> None:
        """Copy the step's version of every state input, and each packed row's request, into the stable buffers.

        Request state is read at each request's version; instance state at the step's newest version, the one value
        every row of the forward reads.
        """
        if not self.needs_step:
            return
        for source in self._result_sources:
            value = context.results[source.task, "value"]
            if source in self._prefetch:
                assert context.prefetch_stream is not None
                with torch.cuda.stream(context.prefetch_stream):
                    self.result_values[source][: len(value)].copy_(value)
                    self._prefetch[source].record(context.prefetch_stream)
            else:
                self.result_values[source][: len(value)].copy_(value)
        requests = tuple(dict.fromkeys(row.identity for row in rows.rows))
        # Each packed row's index into `requests`, the order request state is staged in below.
        index = {identity: position for position, identity in enumerate(requests)}
        positions = tuple(index[row.identity] for row in rows.rows)
        self._requests_of[: len(positions)].copy_(upload(positions, self._requests_of.device))
        for source in self.state_inputs:
            versions = (
                context.read_versions(requests, source.resource, directory)
                if source.request
                else context.read_versions(requests[:1], source.resource, directory, latest=True)
            )
            for index, (pin, ref) in enumerate(versions):
                field = source.view.read(context.arena.borrow_field(pin, ref, source.view.field))
                (self.state_values[source][index] if source.request else self.state_values[source]).copy_(field)
        for memory in self._memories.values():
            memory.prepare(rows, context)

    def remember(self, rewrite: ResidualRewrite, stream: Tensor, branch: Tensor) -> Tensor:
        """Advance private PTTS memory through device-counted request buckets."""
        memory = self._memories[rewrite.name]
        invariants = tuple(self.state(source) for source in memory.memory.invariants(rewrite))
        return memory(stream, branch, invariants)

    def current_rows(self, values: Tensor) -> Tensor:
        """The current call's rows of a per-row buffer."""
        return values[: self.count] if self.rows is None else values.index_select(0, self.rows)

    def state(self, source: StateInput) -> Tensor:
        staged = self.state_values[source]
        return staged.index_select(0, self.current_rows(self._requests_of)) if source.request else staged

    def result(self, source: ResultInput) -> Tensor:
        if source in self._prefetch:
            torch.cuda.current_stream().wait_event(self._prefetch[source])
        return self.current_rows(self.result_values[source])

    def earlier(self, source: EarlierInput) -> Tensor:
        return self.current_rows(self.earlier_values[source])

    def patch(self, source: PatchInput, layer: int) -> Callable[[Tensor], Tensor]:
        assert self._activation is not None
        return self._activation.patch(source.resource, layer, self.rows)

    def store_rows(self, buffer: Tensor, values: Tensor) -> None:
        """Write current rows, preserving destinations of inactive custom-forward lanes."""
        shaped = values.reshape(len(values), *buffer.shape[1:])
        if self.rows is None:
            buffer[: len(shaped)].copy_(shaped)
        else:
            if self.active is not None:
                mask = self.active.reshape(len(values), *((1,) * (values.ndim - 1)))
                shaped = torch.where(mask, shaped, buffer.index_select(0, self.rows))
            buffer.index_copy_(0, self.rows, shaped)

    def keep(self, source: EarlierInput, values: Tensor) -> None:
        """Copy the current rows' value at an earlier place for methods later in this forward."""
        self.store_rows(self.earlier_values[source], values)

    def frozen(self, start: int, stop: int, hidden: Tensor) -> None:
        """Seed skipped layers' readouts with the frozen stream before computed rows overwrite them."""
        for source in self.earlier_values:
            if source.kind != "value" and start <= source.layer < stop:
                self.keep(source, hidden)
        if self.taps is not None:
            for layer in range(start, stop):
                for site in (HiddenSite.RESIDUAL, HiddenSite.ATTENTION_RESIDUAL):
                    tap = self.taps.taps.get(hidden_binding(layer, site, len(self._ends), self.post_attn_norm).tap)
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
class SublayerBoundary:
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
    write: SiteBody | None = None
    read: SiteBody | None = None
    collapse: SiteBody | None = None

    def bind(self, weight: Tensor, methods: dict[tuple[str, bool], Callable[..., None]]) -> None:
        """Bind the same authored bodies for execution and shape-only cost tracing."""
        if self.owner is not None and self.owner.memory is None:
            assert self.before is not None
            sample = weight.new_empty(
                (1, self.sites.streams, self.sites.hidden) if self.sites.streams > 1 else (1, self.sites.hidden)
            )
            branch = weight.new_empty((1, self.sites.hidden))
            self.write = SiteBody(
                self.owner, self.before, self.sites, (sample, branch), compiled=methods.get((self.owner.name, False))
            )
            if self.write.identity:
                self.owner = None
        sample = weight.new_empty(
            (1, self.sites.streams, self.sites.hidden) if self.sites.streams > 1 else (1, self.sites.hidden)
        )
        if self.reader is not None:
            assert self.after is not None
            self.read = SiteBody(
                self.reader,
                self.after,
                self.sites,
                (sample,),
                compiled=methods.get((self.reader.name, True)),
                read=True,
            )
        if self.final is not None:
            self.collapse = SiteBody(
                self.final,
                self.final.places[0],
                self.sites,
                (sample,),
                compiled=methods.get((self.final.name, False)),
            )

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
        """Run the norm directly in its declared ports, without copying its outputs."""
        branch, stream = inputs[0], inputs[1] if len(inputs) == 2 else None
        self.sites.count = len(branch)
        if stream is None:
            result = self(branch)
            assert isinstance(result, Tensor)
            outputs[0].copy_(result)
            return
        normed, residual = outputs
        if (
            self.reader is None
            and self.final is None
            and (self.owner is None or (self.write is not None and self.write.split(stream, branch)))
        ):
            if self.owner is None:
                residual.copy_(stream)
                normed.copy_(branch)
            else:
                assert self.write is not None
                self.write.emit((stream, branch), (residual, normed))
            self._normalize(normed, residual)
        else:
            x = (
                residual
                if self.final is None
                else stream.new_empty((len(stream), self.sites.streams * self.sites.hidden))
            )
            self.complete_into(branch, stream, x)
            if self.final is not None:
                assert self.collapse is not None
                self.collapse.emit((self.streams_of(x),), (residual,))
            if self.reader is None:
                normed.zero_()
                self._normalize(normed, residual)
            else:
                assert self.read is not None
                self.read.emit((self.streams_of(residual),), (normed,))
                self._normalize(normed, torch.zeros_like(normed))
        self._keep(normed, residual)

    def _normalize(self, branch: Tensor, stream: Tensor) -> None:
        normed, residual = self.norm(branch, stream)
        branch.copy_(normed)
        stream.copy_(residual)

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
        """The completed stream for a custom-forward slice that owns its outgoing port."""
        output = torch.empty_like(self.streams_of(stream)).reshape(len(stream), -1)
        self.complete_into(branch, stream, output)
        return output

    def complete_into(self, branch: Tensor, stream: Tensor, output: Tensor) -> None:
        """Write the completed stream directly into the model's residual port."""
        if self.owner is None:
            torch.add(stream, branch, out=output)
        elif self.owner.memory is None:
            assert self.write is not None
            expected = self.streams_of(stream)
            self.write.emit((expected, branch), (output.view_as(expected),))
        else:
            output.copy_(self.sites.remember(self.owner, stream, branch))

    def __call__(self, branch: Tensor, stream: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        self.sites.count = len(branch)
        if stream is not None:
            width = self.sites.hidden * (self.sites.streams if self.final is None else 1)
            # Eager native boundaries donate their inputs; captured ops use fresh ports.
            outputs = (
                [branch, stream]
                if self.owner is None and self.reader is None and self.final is None and self.sites.streams == 1
                else [torch.empty_like(branch), branch.new_empty((len(branch), width))]
            )
            self.emit([branch, stream], outputs)
            return outputs[0], outputs[1]
        x = branch if self.before is not None else self.streams_of(branch).reshape(len(branch), -1)
        if self.final is not None:
            x = self._final(x)
        u = x if self.reader is None else self._read(x)
        normed = self.norm(u)
        normalized = self.taps[0]
        if normalized is not None:
            normalized.copy(normed)
        return normed

    def _read(self, stream: Tensor) -> Tensor:
        assert self.after is not None
        assert self.read is not None
        branch = stream.new_empty((len(stream), self.sites.hidden))
        self.read.emit((self.streams_of(stream),), (branch,))
        return branch

    def _final(self, stream: Tensor) -> Tensor:
        assert self.collapse is not None
        collapsed = stream.new_empty((len(stream), self.sites.hidden))
        self.collapse.emit((self.streams_of(stream),), (collapsed,))
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
class ValueProjection:
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
    write: SiteBody | None = None

    def bind(self, weight: Tensor, methods: dict[tuple[str, bool], Callable[..., None]]) -> None:
        """Bind a value rewrite, omitting shape-independent identity methods."""
        if self.owner is not None:
            sample = weight.new_empty((1, self.width // self.head_dim, self.head_dim))
            self.write = SiteBody(
                self.owner,
                ("value", self.layer),
                self.sites,
                (sample,),
                compiled=methods.get((self.owner.name, False)),
            )
            if self.write.identity:
                self.owner = None

    def __call__(self, hidden: Tensor) -> LinearOutput:
        output = self.native(hidden)
        projected = output if isinstance(output, Tensor) else output[0]
        self.emit([], [projected])
        return output

    def graph_forward(self, hidden: Tensor) -> LinearOutput:
        """Keep the native projection compiled and capture just its value rewrite as a model-site op."""
        output = self.native(hidden)
        projected = output if isinstance(output, Tensor) else output[0]
        self.sites.run_hook(self.index, [], [projected])
        return output

    def emit(self, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        """Rewrite and retain the value slice in place before attention reads it."""
        (projected,) = outputs
        self.sites.count = len(projected)
        value = projected[:, self.start : self.start + self.width].view(len(projected), -1, self.head_dim)
        if self.owner is not None:
            assert self.write is not None
            self.write.emit((value,), (value,))
        if self.keeps is not None:
            self.sites.keep(self.keeps, value)
