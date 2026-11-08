"""The vocabulary for writing a workload: the input markers a method's parameters read (pllm.hidden,
pllm.parameter, ...), the decorators that say what its result does (@pllm.train, @pllm.write, ...), the state
types it declares (pllm.InstanceState, pllm.RequestState) and the model changes (pllm.model.ActivationPatch,
pllm.model.Adapter). Each checks what the author wrote and records it; pllm.workload reads the records from a
Workload class and pllm.execution lowers them. Re-exported by pllm and pllm.model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, get_args

import torch

from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.kv_layout import Keep
from pllm.contracts.parameter_overlay import ParameterOverlayOperator, Projection, ProjectionDelta
from pllm.contracts.request_identity import RequestKeyField, ordered_key_fields
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import StateScope
from pllm.contracts.visibility import VisibilityKind
from pllm.execution.attention_methods import Fact
from pllm.execution.binding_base import SGD as SGD, Field as Field
from pllm.execution.declaration import Shape, When
from pllm.execution.method_effects import (
    SITE_TENSORS,
    CustomAttention,
    CustomResidual,
    Effect,
    Export,
    Gradient,
    Sampler,
    Train,
    TrainInForward,
    TrainReduction,
    Write,
)
from pllm.execution.method_inputs import (
    Attention,
    BaseLogits,
    Batch,
    Carried,
    Carry,
    CausalModelInput,
    Depth,
    Hidden,
    InputMarker,
    InputTokens,
    LayerLogits,
    ModuleInput,
    Operand,
    OperandName,
    Parameter,
    Result,
    SampledTokens,
    SampleIndex,
    StateIndex,
    StateVersion,
    Value,
)
from pllm.execution.model_sites import Site
from pllm.execution.module_state import ModuleFactory, parameter_count
from pllm.execution.position_state import PositionState as PositionState
from pllm.execution.tensor_frame import Tensors, UpdateOutput as UpdateOutput
from pllm.execution.token_program import Over, Tokens
from pllm.kernels.adamw import AdamWConfig

if TYPE_CHECKING:
    from pllm.execution.workload_lowering import WorkloadContext

F = TypeVar("F", bound=Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | UpdateOutput | None])
C = TypeVar("C")
L = TypeVar("L", bound=str)
# How an adapter's fields change a projection: a dense (out_features, in_features) delta, low-rank factors
# (rank, in_features) and (out_features, rank), or one (width, width) block per attention head.
AdapterOperator = Literal["dense", "low_rank", "per_head"]
_OPERATORS: dict[str, ParameterOverlayOperator] = {
    "dense": ParameterOverlayOperator.DENSE_ADD,
    "low_rank": ParameterOverlayOperator.LOW_RANK_ADD,
    "per_head": ParameterOverlayOperator.HEAD_ADD,
}
# The fields that key a request state: the vLLM request, the prompt, the sample of a prompt, a user route.
KeyField = Literal["request_id", "prompt_index", "sample_index", "route"]


class Phase(Enum):
    """The windows of one engine step a method can run in (@pllm.on); each effect has a default window."""

    BEFORE_FORWARD = When.BEFORE_FORWARD
    AFTER_FORWARD = When.AFTER_FORWARD
    BEFORE_SAMPLE = When.BEFORE_SAMPLE
    AFTER_SAMPLE = When.AFTER_SAMPLE
    IN_FORWARD = When.IN_FORWARD
    AUTOMATIC = When.AUTOMATIC


# Inputs: what a method's parameters read, declared as their defaults: `h: torch.Tensor = pllm.hidden(0)`. The
# factories return the marker typed `Any`, so the annotation can name the value the method body receives.
#
# A setting that depends on the workload's configuration is a function of the workload instead of a value: usually
# a method defined above, `pllm.hidden(source_layer)` with `def source_layer(self) -> int`, which the IDE follows and
# the type checker checks. pllm calls it once when the workload is bound.


def hidden(layer: int | Callable[[Any], int] = 0, *, tokens: Literal["decode", "prefill", "all"] = "decode") -> Any:
    """The MLP output before the residual add in decoder layer `layer` (negative from the end).

    One row per request: the latest decode token, or with tokens="prefill" the last prompt token of completed prefill.
    tokens="all" reads one row per token the step computes, prompt chunks included, in token order within each
    request; tokens a preempted request recomputes are read once. Only @pllm.export methods read these rows: each
    step can carry max_num_batched_tokens of them, so give the export max_rows that large (pllm.explain reports the
    host copy volume).
    """
    phase = HiddenPhase(tokens)
    return _Deferred(lambda owner: Hidden(_setting(layer, owner, int), HiddenSite.BLOCK_OUTPUT, phase))


def residual(
    layer: int | Callable[[Any], int],
    *,
    after: Literal["layer", "attention"] = "layer",
    tokens: Literal["decode", "prefill", "all"] = "decode",
) -> Any:
    """The residual stream after decoder layer `layer` (negative from the end): the embedding plus every earlier
    sublayer's output, which the next sublayer normalizes. This is what steering vectors, sparse autoencoders and the
    logit lens read; pllm.hidden(layer) is only the layer's MLP output before it is added to this stream.
    after="attention" reads the stream between the layer's attention and its MLP. Rows are chosen by `tokens` as in
    pllm.hidden."""
    if after not in ("layer", "attention"):
        raise ValueError(f'pllm.residual(after={after!r}): use "layer" or "attention"')
    site, phase = HiddenSite.RESIDUAL if after == "layer" else HiddenSite.ATTENTION_RESIDUAL, HiddenPhase(tokens)
    return _Deferred(lambda owner: Hidden(_setting(layer, owner, int), site, phase))


def layer_logits(layers: tuple[int, ...] | Callable[[Any], tuple[int, ...]]) -> Any:
    """The model's final norm and LM head applied to the residual stream after each of `layers` (negative from the
    end), as float32 [rows, len(layers), vocab]: the logit lens. The last layer's logits are the model's own. One row
    for each token the sampler draws: a prompt's last token and every decode token."""
    return _Deferred(lambda owner: LayerLogits(_layers(layers, owner)))


def final_hidden(*, tokens: Literal["decode", "prefill", "all"] = "decode") -> Any:
    """The hidden state after the model's final norm, which the LM head reads (Hugging Face's last_hidden_state);
    rows are chosen by `tokens` as in pllm.hidden."""
    return Hidden(-1, HiddenSite.MODEL_OUTPUT, HiddenPhase(tokens))


def attention(layer: int, fact: Fact) -> Any:
    """Read sampled attention aux or the first kv.attend call's float32 LSE; inside attention, use earlier layers."""
    if type(layer) is not int or layer < 0 or fact not in ("aux", "lse"):
        raise ValueError(
            "pllm.attention needs a nonnegative layer and fact='aux' or 'lse'; DSA topk facts are deferred"
        )
    return Attention(layer, fact)


def parameter(state: str, field: str, *, per_row: bool = False) -> Any:
    """One tensor field of a declared state, aligned to the operation's rows for request state."""
    return Parameter(state, field, per_row)


def module(state: str) -> Any:
    """A state declared with `module=`, called like the module; request state uses each row's own parameters."""
    return ModuleInput(state)


def version(state: str) -> Any:
    """How many updates each row's state has received (0 before the first update)."""
    return StateVersion(state)


def index(state: str) -> Any:
    """Each row's slot in the state pool, for indexing pooled tensors."""
    return StateIndex(state)


def sample_index(limit: int | Callable[[Any], int] | None = None) -> Any:
    """The sample index of each row's request: 0 for a prompt's first sample.

    With `limit`, generate rejects requests whose sample index is `limit` or more.
    """
    return _Deferred(lambda owner: SampleIndex(None if limit is None else _setting(limit, owner, int)))


def result(operation: str) -> Any:
    """The value another operation of this workload returned in the same step."""
    return Result(operation)


def base_logits() -> Any:
    """The model's logits before any workload adjustment."""
    return BaseLogits()


def sampled_tokens() -> Any:
    """The tokens the sampler drew this step."""
    return SampledTokens()


def input_tokens(*, history: int = 0) -> Any:
    """Int64 [rows, history + 1] token ids: current token first, then lookbacks; -1 before the request starts.

    Results consumed by custom_residual use every scheduled token; other methods use sampled rows.
    Requires runner="mrv2".
    """
    if type(history) is not int or history < 0:
        raise ValueError("pllm.input_tokens(history=) needs a nonnegative integer")
    return InputTokens(history)


def operand(name: OperandName | Callable[[Any], OperandName]) -> Any:
    """A projection input/target/weight, input token embedding, or scalar prompt-end bool.

    Embeddings are the residual stream entering the first decoder layer, before its input norm, aligned with each
    projection row.
    """
    return _Deferred(lambda owner: Operand(_option(name, owner, get_args(OperandName), "operand")))


def carry(source: InputMarker, *, state: str, field: str) -> Any:
    """Read `source` and, when the training update publishes, write its value into the carried `state.field`.

    The next step reads that field as its input version; per request the last selected row wins.
    """
    return Carried(source, Carry(state, field))


def batch(
    field: Literal["hidden", "base_logits", "labels", "weights", "targets", "input_ids", "attention_mask"],
) -> Any:
    """One field of the host batch passed to generate.

    The fields a @pllm.train method reads pick its batch: WeightedRowBatch (hidden, targets, weights) or
    CausalTrainBatch (input_ids, labels, attention_mask, weights) take one AdamW step;
    WeightedCEBatch (hidden, base_logits, labels, weights) runs SGD steps per state.
    """
    return Batch(field)


def causal_model(adapter: str) -> Any:
    """The causal LM with the named pllm.model.Adapter applied, as a pllm.model.CausalModel; .base skips it."""
    return CausalModelInput(adapter)


def depth() -> Any:
    """Inside a @pllm.custom_residual method, the index of the place it runs at, a Python int: the layer, or
    2 * layer (+ 1 for the MLP) with site="both"."""
    return Depth()


def value(layer: int | Callable[[Any], int]) -> Any:
    """Inside a @pllm.custom_residual method, the value rows [rows, kv_heads, head_dim] layer `layer` computed for the
    same token in this forward (after its own value rewrite)."""
    return _Deferred(lambda owner: Value(_setting(layer, owner, int)))


# States: tensors a workload owns, and the changes to the base model that read them.


@dataclass(frozen=True)
class InstanceState:
    """State shared by requests; pretrained= supplies safetensors starting values instead of initialize=.

    Plain states load tensor names matching their fields. Module states load a strict state_dict on meta,
    skipping random initialization. Values convert to the declared storage dtype, including trainable states.
    placement="host" explicitly keeps read-only pretrained gathered fields in pinned host memory;
    placement="device" (default) never falls back to host. Budgets count device residency separately.
    """

    fields: Mapping[str, Shape | Field] = field(default_factory=lambda: dict[str, Shape | Field]())
    # How @pllm.train or @pllm.train_in_forward updates the state. Training on request rows or on WeightedRowBatch
    # and CausalTrainBatch fields takes pllm.AdamW; WeightedCEBatch training and @pllm.train_in_forward take pllm.SGD.
    optimizer: AdamWConfig | SGD | None = field(default=None, kw_only=True)
    # Run once per resource when binding; every admission copies the same seeded initial tensors.
    initialize: Callable[[Tensors, torch.Generator], None] | None = None
    shape: Shape | None = None
    dtype: torch.dtype | Literal["auto"] = "auto"
    alignment: int = field(default=16, kw_only=True)
    # A fed state is filled by a producer process (VllmLlm.feed): "ordered" consumes every publication in order,
    # "latest" only the newest.
    feed: Literal["ordered", "latest"] | None = field(default=None, kw_only=True)
    # Plain per-request tensors carried across steps (written by pllm.carry); never optimizer-managed.
    carry: Mapping[str, Shape] = field(default_factory=lambda: dict[str, Shape](), kw_only=True)
    # An `nn.Module` built from the model's dimensions; its parameters become the state, initialized as the
    # module initializes them (every request starts from the same values).
    module: ModuleFactory | None = field(default=None, kw_only=True)
    imported: bool = field(default=False, kw_only=True)
    pretrained: str = field(default="", kw_only=True)
    placement: Literal["device", "host"] = field(default="device", kw_only=True)
    checkpoint: Mapping[str, str] = field(default_factory=lambda: dict[str, str](), kw_only=True)
    layers: tuple[int, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if self.placement not in ("device", "host"):
            raise ValueError('placement needs "device" or "host"')
        if self.placement == "host" and (
            self.scope is not StateScope.INSTANCE
            or not self.pretrained
            or self.optimizer is not None
            or self.module is not None
            or self.carry
        ):
            raise ValueError(
                'placement="host" needs a read-only pretrained InstanceState of gathered fields; '
                'use placement="device" for modules or mutable state'
            )
        if self.pretrained and (self.initialize is not None or self.checkpoint or self.feed or self.imported):
            raise ValueError(
                "pretrained= supplies starting values; drop initialize=, checkpoint=, feed= and imported=True"
            )
        if self.checkpoint and (
            self.scope is not StateScope.INSTANCE
            or self.optimizer is not None
            or self.initialize is not None
            or self.module is not None
            or self.feed
            or self.imported
            or set(self.checkpoint) != set(self.fields)
        ):
            raise ValueError("checkpoint state needs an immutable InstanceState with a tensor name for every field")
        if self.layers and (
            self.scope is not StateScope.INSTANCE
            or tuple(sorted(set(self.layers))) != self.layers
            or self.layers[0] < 0
        ):
            raise ValueError("layers needs sorted unique nonnegative indices on an InstanceState")
        if self.feed not in (None, "ordered", "latest"):
            raise ValueError(f'feed={self.feed!r}: use "ordered", "latest" or None')
        if self.module is not None:
            if self.fields or self.shape is not None or self.initialize is not None or self.feed:
                raise ValueError(
                    "a module state takes its fields and initial values from the module; drop fields/shape/initialize"
                )
            object.__setattr__(self, "fields", {"parameters": parameter_count(self.module)})
        if self.feed is not None and (
            self.scope is not StateScope.INSTANCE
            or self.optimizer is not None
            or self.initialize is not None
            or self.carry
            or self.imported
        ):
            raise ValueError("fed state needs InstanceState without an optimizer, initializer or carry fields")
        if self.shape is not None:
            object.__setattr__(self, "fields", {"value": self.shape})
        if self.carry and self.optimizer is None:
            raise ValueError("carry fields ride on the training transition; give the state an optimizer")
        if set(self.carry) & set(self.fields):
            raise ValueError("rename the carry fields that repeat a parameter field name")

    scope: ClassVar[StateScope] = StateScope.INSTANCE

    def field_specs(self) -> dict[str, Field]:
        """Each field as a pllm.Field, a fixed shape in the state's dtype; for states an adapter or a
        @pllm.train_in_forward method reads per projection."""
        fields: dict[str, Field] = {}
        for key, value in self.fields.items():
            match value:
                case Field():
                    fields[key] = value
                case tuple():
                    fields[key] = Field(value, self.dtype)
                case _:
                    raise TypeError(
                        f"{key}: a shape that depends on the projection is pllm.Field(lambda in_features, out_features: ...)"
                    )
        return fields

    @property
    def request_keys(self) -> tuple[RequestKeyField, ...]:
        return ()

    def source(self) -> str:
        """Describe where this state's version-zero values come from."""
        if self.imported:
            return (
                "imported: "
                + ("one row per prompt" if self.scope is StateScope.REQUEST else "field tensors")
                + " via generate(batch=pllm.io.StateImport(...)), applied before the request's first forward"
            )
        if self.feed:
            return "fed"
        if self.checkpoint:
            return "base checkpoint"
        if self.module is not None:
            return "module init"
        return "initialize" if self.initialize is not None else "zeros"

    def checkpoint_names(self) -> tuple[str, ...]:
        """Exact checkpoint tensor names, in the same order as the allocated fields."""
        if not self.checkpoint:
            return ()
        return (
            tuple(
                pattern.format(layer=layer)
                for layer in self.layers
                for pattern in (self.checkpoint[key] for key in self.fields)
            )
            if self.layers
            else tuple(self.checkpoint[key] for key in self.fields)
        )


@dataclass(frozen=True)
class RequestState(InstanceState):
    """Per-request parameters, read as tensors aligned to the current operation's rows."""

    # Requests with equal values of these fields share one copy of the state, listed in KeyField order.
    key_fields: tuple[KeyField, ...] = ("request_id",)

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(name not in get_args(KeyField) for name in self.key_fields):
            raise ValueError(f"key_fields={self.key_fields}: use names from {get_args(KeyField)}")
        ordered_key_fields(self.request_keys)

    scope: ClassVar[StateScope] = StateScope.REQUEST

    @property
    def request_keys(self) -> tuple[RequestKeyField, ...]:
        return tuple(RequestKeyField(name) for name in self.key_fields)


@dataclass(frozen=True)
class ActivationPatch:
    """Edit the hidden state at `layers` from a state, whoever writes it (ReFT-style): the first and last prompt
    tokens, and with decode=True every decode token.

    operator="low_rank_affine" reads three fields per layer, "add" one delta per layer. refresh="admission" keeps
    the version a request was admitted with; "step" reads the newest version at every decode step, and on_missing
    says what a step without a new version does ("stop" the request or "hold" the last version).
    """

    state: str
    layers: tuple[int, ...]
    first_tokens: int = 0
    last_tokens: int = 2
    operator: Literal["low_rank_affine", "add"] = "low_rank_affine"
    _: KW_ONLY
    decode: bool = False
    refresh: Literal["admission", "step"] = "admission"
    on_missing: Literal["stop", "hold"] = "stop"

    def __post_init__(self) -> None:
        if (
            self.operator not in ("low_rank_affine", "add")
            or self.refresh not in ("admission", "step")
            or self.on_missing not in ("stop", "hold")
        ):
            raise ValueError(
                'ActivationPatch takes operator "low_rank_affine" or "add", refresh "admission" or "step" and '
                f'on_missing "stop" or "hold"; got {self}'
            )
        if not self.decode and (self.refresh != "admission" or self.on_missing != "stop"):
            raise ValueError("refresh= and on_missing= apply to decode tokens; set decode=True")


@dataclass(frozen=True)
class Adapter:
    """Apply a state as a weight delta at the target projections, whoever trains it (a PEFT-style adapter).

    `weight` names the field of a "dense" or "per_head" delta, `factors` the two fields of a "low_rank" one; the
    delta is multiplied by `scale`.

    A weight delta applies on every invocation of its projection, including repeated layer calls and replay in
    @pllm.custom_forward. Each call uses its actual input and each row's request state, pinned at forward entry.
    Compacted active= rows and reordered waves retain that ownership through device indices; skipped projections
    receive no correction. Native QKV propagation for skipped layers still applies the Q/V deltas when it runs.
    This lets custom depth schedules use the same adapted model as the standard forward.

    At TP>1 (vLLM 0.32 MRv2), dense and low_rank deltas follow each projection's shards.
    Declare fields with Field(lambda inputs, outputs: ...). Down deltas partition dense weights and A
    by input columns, with replicated B. Attention q/k/v and MLP gate/up deltas partition dense weights and B
    by output rows, with replicated A; KV head replicas follow vLLM's head placement. Small fields replicate.
    Initialization runs globally before slicing, and explain reports per-rank storage. vLLM's existing row
    reduction sums corrected down partials; no extra collective is added. Updates must be shard-local: training
    replicated A from column-sharded outputs, or B from row-sharded inputs, requires tensor_parallel_size=1.
    Sharded fields are read only in projection transitions; replicated factors remain available to other methods.
    TP updates reject captured tensor constants; use Python scalars, new_full/new_ones or replicated state fields.
    per_head deltas, imported/checkpoint adapters and side-lane adapter training still require TP=1.
    """

    state: str
    targets: tuple[Projection, ...]
    weight: str = "value"
    operator: AdapterOperator = "dense"
    factors: tuple[str, str] = ("a", "b")
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.operator not in _OPERATORS:
            raise ValueError(f'Adapter(operator={self.operator!r}): use "dense", "low_rank" or "per_head"')

    @property
    def delta(self) -> ProjectionDelta:
        return ProjectionDelta(
            self.state, self.targets, self.weight, _OPERATORS[self.operator], self.factors, self.scale
        )


# Decorators: what a method's result does, and when the method runs.


def on(phase: Phase | Callable[[Any], Phase]) -> Callable[[F], F]:
    """Run the method in `phase`; needed for methods without an effect or to move an effect's phase."""
    return _mark(phase=lambda owner: _setting(phase, owner, Phase))


def write(*, state: str, fields: tuple[str, ...]) -> Callable[[F], F]:
    """Atomically publish a tuple of computed fields; request values have a leading request axis."""
    return _mark(effect=lambda owner: Write(state=state, fields=fields))


def export(
    *,
    name: str | Callable[[Any], str] = "value",
    every: int | Callable[[Any], int] = 1,
    max_rows: int | Callable[[Any], int] = 1,
    max_cols: int | Callable[[Any], int] = 16,
    enabled: bool | Callable[[Any], bool] = True,
    overflow: Literal["drop", "block"] = "drop",
) -> Callable[[F], F]:
    """Deliver bounded observations of executed forward rows to on_export.

    A step delivers at most max_rows rows, split into events of whole rows of at most 32 KiB; each event lists
    every row's request identity and token position. Async scheduling can execute one extra decode row before
    learning that a request stopped at EOS; that row is observed too. Callbacks run asynchronously and close()
    drains their delivery. Block on a full export queue only when explicitly requested.
    """
    if overflow not in ("drop", "block"):
        raise ValueError("export overflow must be 'drop' or 'block'")
    return _mark(
        effect=lambda owner: Export(
            _setting(name, owner, str),
            _setting(every, owner, int),
            _setting(max_rows, owner, int),
            _setting(max_cols, owner, int),
            _setting(enabled, owner, bool),
            overflow,
        )
    )


def add_logits(*, rows: str = "") -> Callable[[F], F]:
    """Add the returned tensor to the logits before sampling.

    `rows` (here and on the other effects) names a method returning a boolean mask: only those rows run.
    """
    return _mark(effect=lambda owner: Sampler(SamplerStage.PACKED_PRE_FILTER_ADD), rows=rows)


def guide_sampling(*, beta: float | Callable[[Any], float], rows: str = "") -> Callable[[F], F]:
    """Push sampling away from the returned hidden-state prediction for positive `beta`.

    On finite top-k/top-p survivors, use `(1 + beta) * logits - beta * pred_logits`, where the model's LM head
    projects the returned hidden states to `pred_logits`.
    """
    return _mark(
        effect=lambda owner: Sampler(SamplerStage.PACKED_POST_FILTER_EXACT, _setting(beta, owner, float)), rows=rows
    )


def train(
    *,
    state: str,
    update_rows: str = "",
    reduction: TrainReduction = "mean",
    steps: int | Callable[[Any], int] = 1,
    reset: str = "",
    rows: str = "",
) -> Callable[[F], F]:
    """Minimize the returned loss with `state`'s optimizer; updates publish after sampling by default.

    With tensor parallelism, independent automatic request training can compute a private update before the
    draw. Publication still follows sampling; cancellation discards the private version. Use
    `@pllm.on(pllm.Phase.AFTER_SAMPLE)` to require the computation itself to follow sampling.

    The method's inputs say what it trains on. Without pllm.batch fields it trains each request's state on that
    request's rows (pllm.AdamW). WeightedRowBatch or CausalTrainBatch fields, or a causal model, take one AdamW
    step on the host batch. WeightedCEBatch fields run `steps` SGD steps per state, each state first restored by
    the `reset` method (pllm.SGD).

    Request-row methods return a [rows] loss. The default `reduction="mean"` averages each state's selected
    rows in this engine step; `"sum"` sums them. Both keep states independent. Each state takes one optimizer
    step per engine step; prompt rows train once per prefill chunk, so prompt training depends on chunking.
    Reduction matters for prompt rows, shared key_fields states, InstanceState, or any step with several rows
    per state. Plain request_id decode training has one row per state, so both reductions are equal.
    Choose mean for AdamW by default; choose sum for a per-token SGD learning rate or shared-key online group
    training, where each surviving token should contribute without division by the group's current row count.

    Host-batch methods return one scalar and own their reduction; a non-default reduction= is rejected.
    `rows=` names a selector whose mask picks the rows that count; zero-weight losses still count toward the
    mean, and the method still runs on every row. For request training, rows= and update_rows= require a model
    effect without sampler guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update
    every selected state.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"@pllm.train(reduction={reduction!r}): use 'mean' or 'sum'")
    return _mark(
        effect=lambda owner: Train(
            state=state, update_rows=update_rows, reduction=reduction, steps=_setting(steps, owner, int), reset=reset
        ),
        rows=rows,
    )


def differentiate(
    *, inputs: tuple[str, ...] = (), parameters: bool = False, input_rows: str = "", parameter_rows: str = ""
) -> Callable[[F], F]:
    """Keep derivative edges; parameters may stay frozen while inputs differentiate."""
    return _mark(gradient=Gradient(inputs, parameters, input_rows, parameter_rows))


def train_in_forward(
    *,
    state: str,
    targets: tuple[Projection, ...] | Callable[[Any], tuple[Projection, ...]],
    weight: str = "value",
    operator: AdapterOperator | Callable[[Any], AdapterOperator] = "dense",
    factors: tuple[str, str] | Callable[[Any], tuple[str, str]] = ("a", "b"),
    scale: float | Callable[[Any], float] = 1.0,
    tokens: Tokens | Callable[[Any], Tokens] = "prefill",
    visibility: VisibilityKind | Callable[[Any], VisibilityKind] = "forward_pinned",
    chunk: int | Callable[[Any], int] | None = None,
    flush: Literal["prompt_end"] | None = None,
) -> Callable[[F], F]:
    """Train a request state inside the forward and add its residual at each target projection.

    `tokens` picks the tokens whose projection operands train the state: "prefill", "decode" or "all";
    `operator`, `weight`, `factors` and `scale` say how the state changes each projection, as in
    pllm.model.Adapter. visibility="forward_pinned" reads the state the forward started with; "exact" also reads
    every chunk already trained in this forward. Column projections expose their replicated input as
    operand("target"); readouts have the projection's output width, which can differ from that input width.
    At TP>1, state initialization and shard-local update rules are the same as pllm.model.Adapter.

    With ``chunk``, the function takes the weight carry followed by input/target/embedding row operands and returns
    (next carry, readouts). Pending rows belong to the framework. When ``state`` declares
    ``optimizer=pllm.SGD(lr)``, return one loss per row instead; gradients accumulate until a chunk completes,
    including a short final chunk with ``flush="prompt_end"``.
    """
    if flush not in (None, "prompt_end"):
        raise ValueError("@pllm.train_in_forward(flush=...) takes 'prompt_end' or None")
    return _mark(
        effect=lambda owner: TrainInForward(
            state=state,
            targets=_targets(_setting(targets, owner, tuple)),
            weight=weight,
            operator=_OPERATORS[_option(operator, owner, get_args(AdapterOperator), "operator")],
            factors=_setting(factors, owner, tuple),
            scale=_setting(scale, owner, float),
            tokens=_option(tokens, owner, get_args(Tokens), "tokens"),
            visibility=_option(visibility, owner, get_args(VisibilityKind), "visibility"),
            chunk=None if chunk is None else _setting(chunk, owner, int),
            flush=flush,
        )
    )


def custom_attention(
    *,
    layers: tuple[int, ...] | Callable[[Any], tuple[int, ...]] | None = None,
    keep: Keep = "all",
    window: int | Callable[[Any], int] = 0,
    source: int | Callable[[Any], int] = 0,
    sink: bool | Callable[[Any], bool] = False,
) -> Callable[[F], F]:
    """Replace dense FlashAttention reads, retaining native projections and KV writes.

    The method receives post-RoPE q [rows, heads, head_dim] and pllm.model.PagedKV.
    Return attention with q's shape/dtype, optionally paired with aux [rows, ...].
    kv.attend returns (output, float32 LSE); window and per-head topk select reads
    without freeing storage. Methods may call pure torch.library.custom_op kernels.
    keep="window", window=N frees old blocks through vLLM's SlidingWindowSpec.
    sink=True retains the entire prompt plus that window (vLLM RSWASpec).
    keep="none", source=k allocates no cache and reads the layer k below.
    Layouts restrict every read, including topk; window_start marks the retained tail.
    Requires TP=1, unquantized FA2 attention and prefix caching disabled.
    Static layouts need MRv2 and cannot accompany PositionState or custom forwards.
    The prompt-sink FA2 read uses a gathered FP32 softmax; storage and reclamation
    are owned by vLLM's RSWAManager. Score-based memory eviction is unsupported.
    """
    return _mark(
        effect=lambda owner: CustomAttention(
            layers=None if layers is None else _setting(layers, owner, tuple),
            keep=keep,
            window=_setting(window, owner, int),
            source=_setting(source, owner, int),
            sink=_setting(sink, owner, bool),
        )
    )


def custom_residual(
    *,
    site: Site,
    layers: tuple[int, ...] | Callable[[Any], tuple[int, ...]] | None = None,
    read: str = "",
    streams: int | Callable[[Any], int] = 1,
    state: str = "",
    chunk: int | Callable[[Any], int] | None = None,
) -> Callable[[F], F]:
    """Own the residual update at `site` of `layers` (default: every layer) with this method.

    site="attention" or "mlp" (or "both", in forward order) take the stream before the sublayer and its branch
    output, `write(x, y, ...) -> stream`, replacing the plain `x + y`; the native branch always runs. site="value"
    maps a full-attention layer's value rows [rows, kv_heads, head_dim] before attention reads and caches them;
    site="final" maps the stream entering the final norm to one stream. `read` names a method `(x, ...) -> branch
    input` that the sublayer normalizes instead of the stream. With `streams=n` the stream is [rows, n, hidden] from the
    embedding to a "final" method that collapses it. After the site tensors, methods read pllm.parameter, pllm.module,
    pllm.depth(), pllm.value(layer) and pllm.residual(layer) (the same token, earlier in this forward). Tensors are in the
    model dtype; compute in float32 inside the method when needed.

    With `state` (a pllm.RequestState with one field) and `chunk`, the method is a stateful sublayer at one site: it
    sees one chunk of one request, `write(x, y, carry=pllm.parameter(state, ...), ...) -> (stream, next carry)`,
    where every row reads the carry its chunk started with and a chunk's update applies once its last row has run.
    Each row's stream should depend only on that row and the carry.
    """
    if site not in SITE_TENSORS:
        raise ValueError(f"@pllm.custom_residual(site={site!r}): use one of {', '.join(map(repr, SITE_TENSORS))}")
    return _mark(
        effect=lambda owner: CustomResidual(
            site=site,
            layers=None if layers is None else _setting(layers, owner, tuple),
            read=read,
            streams=_setting(streams, owner, int),
            state=state,
            chunk=None if chunk is None else _setting(chunk, owner, int),
        )
    )


def custom_forward(*, tokens: Tokens = "all", over: Over = "token") -> Callable[[F], F]:
    """Replace the forward of the selected tokens with this method's program over model layer slices and tensor math.

    ``tokens`` follows the token's prompt/generated role, including after preemption.
    ``over="token"`` runs each position independently in causal order. ``over="span"``
    packs each request's contiguous scheduled tokens into [M, H] hidden tensors;
    position and generated flags have shape [M, 1]. Spans cannot load predecessors.
    Inputs after layers use pllm.parameter / pllm.module to read InstanceState, shared
    by every token; the method body reads plain settings from self.
    """
    if tokens not in ("prefill", "decode", "all"):
        raise ValueError('set @pllm.custom_forward(tokens=) to "prefill", "decode" or "all"')
    if over not in ("token", "span"):
        raise ValueError('set @pllm.custom_forward(over=) to "token" or "span"')
    return _mark(tokens=tokens, over=over)


# Settings and markers read from the bound workload, and the attributes the decorators leave on a method.


@dataclass(frozen=True)
class _Deferred(InputMarker):
    """A marker whose settings read the workload, e.g. pllm.hidden(source_layer); built when the workload is bound."""

    build: Callable[[Any], InputMarker]

    def bind(self, context: WorkloadContext) -> InputMarker:
        return self.build(context.owner).bind(context)


def _setting(value: C | Callable[[Any], C], owner: object, expected: type[C]) -> C:
    """Read a decorator or marker setting, calling it on the workload when it is a function, and check its type."""
    resolved = value(owner) if callable(value) else value
    if expected is float and type(resolved) is int:
        resolved = float(resolved)
    if not isinstance(resolved, expected) or type(resolved) is not expected:
        raise TypeError(f"{value!r} is {type(resolved).__name__}; expected {expected.__name__}")
    return resolved


def _option(value: L | Callable[[Any], L], owner: object, options: tuple[L, ...], name: str) -> L:
    """Read a closed string option, calling it on the workload when it is a function, and check it is one of
    `options`."""
    resolved = value(owner) if callable(value) else value
    for option in options:
        if resolved == option:
            return option
    raise ValueError(f"{name}={resolved!r}: use one of {', '.join(map(repr, options))}")


def _layers(layers: tuple[int, ...] | Callable[[Any], tuple[int, ...]], owner: object) -> tuple[int, ...]:
    resolved = _setting(layers, owner, tuple)
    if not resolved or any(type(layer) is not int for layer in resolved):
        raise TypeError(f"pllm.layer_logits({resolved!r}) takes a non-empty tuple of layer indices")
    return resolved


def _targets(value: tuple[object, ...]) -> tuple[Projection, ...]:
    targets = tuple(item for item in value if isinstance(item, Projection))
    if len(targets) != len(value):
        raise TypeError("targets must be pllm.model.Projection values")
    return targets


def _mark(
    *,
    effect: Callable[[Any], Effect] | None = None,
    phase: Callable[[Any], Phase] | None = None,
    **markers: object,
) -> Callable[[F], F]:
    """Attach authoring markers; reflection consumes them once during declaration, building the effect and phase
    from the bound workload."""
    if effect is not None:
        markers["effect"] = effect
    if phase is not None:
        markers["phase"] = phase

    def decorate(function: F) -> F:
        # A second marker of one kind would silently replace the first, e.g. @export over @train never trains.
        if any(f"__pllm_{name}__" in vars(function) for name in markers):
            raise ValueError(
                "a method takes one pllm decorator of each kind; split stacked effects into one method per effect"
            )
        for name, marker in markers.items():
            setattr(function, f"__pllm_{name}__", marker)
        return function

    return decorate


__all__ = [
    "custom_forward",
    "Phase",
    "on",
    "write",
    "export",
    "add_logits",
    "guide_sampling",
    "train",
    "hidden",
    "residual",
    "layer_logits",
    "final_hidden",
    "parameter",
    "module",
    "version",
    "index",
    "sample_index",
    "result",
    "base_logits",
    "sampled_tokens",
    "operand",
    "carry",
    "causal_model",
    "batch",
    "InstanceState",
    "RequestState",
    "train_in_forward",
    "custom_residual",
    "depth",
    "value",
    "Field",
    "UpdateOutput",
    "differentiate",
    "ActivationPatch",
    "Adapter",
    "SGD",
]
