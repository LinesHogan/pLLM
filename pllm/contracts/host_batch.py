"""Original structured inputs shared by authoring, execution and transport."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import (
    HostServiceInputSource,
    SchedulerTimeline,
    SemanticGraph,
    SemanticState,
    StateScope,
)
from pllm.contracts.tensor_layout import TensorDType, TensorFieldSpec, TensorLayoutRequest, TensorLayoutSpec

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True, slots=True)
class CausalTrainBatch(Generic[T]):
    """Token sequences for host training; optional floating weights have one value per sequence."""

    schema: ClassVar[str] = "causal_tokens@2"
    names: ClassVar[tuple[str, ...]] = ("input_ids", "labels", "attention_mask", "weights")
    input_ids: T
    labels: T
    attention_mask: T | None = None
    weights: T | None = None

    def tensor(self, name: str) -> T | None:
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "attention_mask": self.attention_mask,
            "weights": self.weights,
        }[name]


@dataclass(frozen=True, slots=True)
class WeightedRowBatch(Generic[T]):
    """[rows, hidden_size] features with one target (a value or a fixed-shape tensor) and one weight per row; weight 0
    skips a row."""

    schema: ClassVar[str] = "weighted_rows@1"
    names: ClassVar[tuple[str, ...]] = ("hidden", "targets", "weights")
    hidden: T
    targets: T
    weights: T

    def tensor(self, name: str) -> T | None:
        return {"hidden": self.hidden, "targets": self.targets, "weights": self.weights}[name]


@dataclass(frozen=True, slots=True)
class StateImport(Generic[T]):
    """External tensors in field order, visible from the importing call's first step.

    RequestState values have one leading row per prompt, in generate order, shared by its n samples.
    InstanceState values are the field tensors themselves. Floating values convert to storage precision.
    """

    schema: ClassVar[str] = "state_tensors@1"
    values: tuple[T, ...]


@dataclass(frozen=True, slots=True)
class WeightedCEBatch(Generic[T]):
    """Shared [rows, ...] or independently keyed [adapters, rows, ...] weighted CE."""

    schema: ClassVar[str] = "weighted_ce@2"
    names: ClassVar[tuple[str, ...]] = ("hidden", "base_logits", "labels", "weights")
    hidden: T
    base_logits: T
    labels: T
    weights: T
    keys: tuple[RequestStateKey, ...] = ()

    @staticmethod
    def for_prompts(
        hidden: U, base_logits: U, labels: U, weights: U, prompt_indices: tuple[int, ...], *, sample_index: int = 0
    ) -> "WeightedCEBatch[U]":
        """Key the adapter axis by the same prompt indices supplied to generate."""
        keys = tuple(
            RequestStateKey.project(
                RequestIdentity("", index, sample_index), (RequestKeyField.PROMPT_INDEX, RequestKeyField.SAMPLE_INDEX)
            )
            for index in prompt_indices
        )
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("independent training requires nonempty, unique prompt keys")
        return WeightedCEBatch(hidden, base_logits, labels, weights, keys)


HostBatch = CausalTrainBatch[T] | StateImport[T] | WeightedCEBatch[T] | WeightedRowBatch[T]


def validate_host_batch(
    value: HostBatch[T] | None,
    graph: SemanticGraph,
    layouts: TensorLayoutRequest,
    timeline: SchedulerTimeline,
    describe: Callable[[T], TensorFieldSpec],
    *,
    hidden_size: int,
    vocab_size: int,
    target_shape: tuple[int, ...] = (),
    prompt_count: int = 0,
    import_fields: tuple[tuple[str, FieldView], ...] = (),
) -> None:
    """Check a generate call's host batch once, before export, mapping or device copies; tasks trust it.

    `target_shape` is the shape of each WeightedRowBatch row's target (HostConfig.target_shape).
    """
    if value is None:
        return
    consumers = tuple(
        task
        for task in graph.tasks
        if any(
            isinstance(port.source, HostServiceInputSource) and port.source.schema == value.schema
            for port in task.inputs
        )
    )
    if not consumers:
        raise ValueError(
            f"this workload reads no {type(value).__name__}; pass batch=None or add a method that reads it"
            + (" or declare a state with init=pllm.Imported()" if isinstance(value, StateImport) else "")
        )

    def shape(tensor: T) -> tuple[int, ...]:
        return describe(tensor).shape

    match value:
        case CausalTrainBatch(input_ids=ids, labels=labels, attention_mask=mask, weights=weights):
            tokens = shape(ids)
            if (
                len(tokens) != 2
                or tokens[0] < 1
                or tokens[1] < 2
                or shape(labels) != tokens
                or mask is not None
                and shape(mask) != tokens
                or describe(ids).dtype is not TensorDType.INT64
                or describe(labels).dtype is not TensorDType.INT64
            ):
                raise ValueError("causal batch must contain aligned token/label tensors")
            if weights is not None and (
                shape(weights) != tokens[:1] or not describe(weights).dtype.torch.is_floating_point
            ):
                raise ValueError(
                    f"CausalTrainBatch weights need one floating value per sequence: shape {tokens[:1]}, "
                    f"got {shape(weights)} ({describe(weights).dtype})"
                )
            for task in consumers:
                if weights is None and any(
                    isinstance(port.source, HostServiceInputSource) and port.source.port == "weights"
                    for port in task.inputs
                ):
                    raise ValueError(
                        f"{task.symbol} reads pllm.batch('weights'); pass CausalTrainBatch(..., weights=...)"
                    )
            actual = (tokens,)
        case WeightedCEBatch(hidden=hidden, base_logits=logits, labels=labels, weights=weights, keys=keys):
            prefix = () if keys else (1,)
            (layout,) = layouts.layouts
            dtype = layout.layout.fields[0].dtype
            hidden_shape, logits_shape = prefix + shape(hidden), prefix + shape(logits)
            rows = hidden_shape[:2]
            if (
                len(hidden_shape) != 3
                or min(hidden_shape) < 1
                or hidden_shape[2] != hidden_size
                or len(logits_shape) != 3
                or logits_shape[:2] != rows
                or logits_shape[2] != vocab_size
                or prefix + shape(labels) != rows
                or prefix + shape(weights) != rows
                or hidden_shape[0] != (len(keys) if keys else 1)
                or len(set(keys)) != len(keys)
                or describe(labels).dtype is not TensorDType.INT64
                or any(describe(tensor).dtype is not dtype for tensor in (hidden, logits, weights))
            ):
                raise ValueError(
                    f"WeightedCEBatch needs {dtype} hidden [..., tokens, {hidden_size}] and base_logits "
                    f"[..., tokens, {vocab_size}], int64 labels and {dtype} weights [..., tokens], and one leading "
                    "row per distinct key when keys are given"
                )
            actual = (hidden_shape, logits_shape)
        case WeightedRowBatch(hidden=hidden, targets=targets, weights=weights):
            features = shape(hidden)
            if (
                len(features) != 2
                or features[0] < 1
                or features[1] != hidden_size
                or shape(targets) != features[:1] + target_shape
                or shape(weights) != features[:1]
                or describe(targets).dtype is not describe(hidden).dtype
                or describe(weights).dtype is not describe(hidden).dtype
            ):
                rows = ", ".join(("rows", *map(str, target_shape)))
                raise ValueError(
                    f"weighted rows need hidden [rows, {hidden_size}], targets [{rows}] and weights [rows] of one "
                    "dtype; set HostConfig(target_shape=...) to the shape of each row's target"
                )
            actual = (features,)
        case StateImport():
            state = graph.host_state(StateImport.schema)
            assert state is not None
            layout = next(entry.layout for entry in layouts.layouts if entry.resource == state.resource)
            _validate_state_import(value, state, layout, import_fields, describe, prompt_count)
            return
    for task in consumers:
        limit = next((fact for fact in timeline.task_resources if fact.task == task.symbol), None)
        if limit is None:
            continue  # No cost declaration: the original unbounded execution contract.
        if len(limit.input_shapes) != len(actual) or any(
            len(dims) != len(bounds) or any(size > bound for size, bound in zip(dims, bounds))
            for dims, bounds in zip(actual, limit.input_shapes)
        ):
            raise ValueError(
                f"{task.symbol}: host batch shapes {list(actual)} exceed the planned {list(limit.input_shapes)}; "
                "raise ExecutionOptions(max_host_training_rows=N) for more training rows per state, or "
                "HostConfig(max_input_rows=N) for a host workload"
            )


def _validate_state_import(
    value: StateImport[T],
    state: SemanticState,
    layout: TensorLayoutSpec,
    fields: tuple[tuple[str, FieldView], ...],
    describe: Callable[[T], TensorFieldSpec],
    prompt_count: int,
) -> None:
    """Validate logical import fields before storage mapping or execution."""
    values = value.values
    label = f"StateImport for {('RequestState' if state.scope is StateScope.REQUEST else 'InstanceState')} {state.resource!r}"
    if len(values) != len(fields):
        raise ValueError(f"{label} needs {len(fields)} field tensors in declaration order, got {len(values)}")
    for tensor, (name, view) in zip(values, fields, strict=True):
        spec = describe(tensor)
        prefix = (prompt_count,) if state.scope is StateScope.REQUEST else ()
        if prefix and spec.shape[:1] != prefix:
            raise ValueError(
                f"{label} needs one row per prompt: field {name!r} has {spec.shape[0] if spec.shape else 0} rows for {prompt_count} prompts"
            )
        if spec.shape != prefix + view.shape:
            raise ValueError(
                f"{label} takes {'one row per prompt' if prefix else 'the field tensors themselves'}: field {name!r} has shape {spec.shape}, declared {prefix + view.shape}"
            )
        dtype = layout.fields[view.field].dtype
        if spec.dtype != dtype and not (spec.dtype.torch.is_floating_point and dtype.torch.is_floating_point):
            raise ValueError(f"{label}: field {name!r} needs floating or matching {dtype} values")


def map_host_batch(value: HostBatch[T] | None, convert: Callable[[T], U]) -> HostBatch[U] | None:
    """Transform tensor storage without introducing a transport-specific input."""
    match value:
        case None:
            return None
        case CausalTrainBatch(input_ids=ids, labels=labels, attention_mask=mask, weights=weights):
            return CausalTrainBatch(
                convert(ids),
                convert(labels),
                None if mask is None else convert(mask),
                None if weights is None else convert(weights),
            )
        case StateImport(values=values):
            return StateImport(tuple(convert(item) for item in values))
        case WeightedCEBatch(hidden=hidden, base_logits=logits, labels=labels, weights=weights, keys=keys):
            return WeightedCEBatch(convert(hidden), convert(logits), convert(labels), convert(weights), keys)
        case WeightedRowBatch(hidden=hidden, targets=targets, weights=weights):
            return WeightedRowBatch(convert(hidden), convert(targets), convert(weights))
