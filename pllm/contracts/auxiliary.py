"""Original structured inputs shared by authoring, execution and transport."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import HostServiceInputSource, StateScope
from pllm.contracts.tensor_layout import TensorDType, TensorFieldSpec

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True, slots=True)
class CausalTrainBatch(Generic[T]):
    schema: ClassVar[str] = "canonical.causal_tokens@1"
    input_ids: T
    labels: T
    attention_mask: T | None = None


@dataclass(frozen=True, slots=True)
class StateTensorUpdate(Generic[T]):
    """Original external tensors in the signed layout's field order."""

    schema: ClassVar[str] = "canonical.state_tensors@1"
    values: tuple[T, ...]
    keys: tuple[RequestStateKey, ...] = ()


@dataclass(frozen=True, slots=True)
class WeightedCEBatch(Generic[T]):
    """Shared [rows, ...] or independently keyed [adapters, rows, ...] weighted CE."""

    schema: ClassVar[str] = "canonical.weighted_ce@2"
    hidden: T
    base_logits: T
    labels: T
    weights: T
    keys: tuple[RequestStateKey, ...] = ()

    @staticmethod
    def for_prompts(
        hidden: U, base_logits: U, labels: U, weights: U, prompt_indices: tuple[int, ...], *, sample_idx: int = 0
    ) -> "WeightedCEBatch[U]":
        """Key the adapter axis by the same prompt indices supplied to generate."""
        keys = tuple(
            RequestStateKey.project(
                RequestIdentity("", index, sample_idx), (RequestKeyField.PROMPT_IDX, RequestKeyField.SAMPLE_IDX)
            )
            for index in prompt_indices
        )
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("independent training requires nonempty, unique prompt keys")
        return WeightedCEBatch(hidden, base_logits, labels, weights, keys)


AuxiliaryInput = CausalTrainBatch[T] | StateTensorUpdate[T] | WeightedCEBatch[T]


def validate_auxiliary(
    value: AuxiliaryInput[T] | None, plan: PlanBundle, describe: Callable[[T], TensorFieldSpec]
) -> None:
    """Preflight original storage metadata before export, mapping or device copies."""
    if value is None:
        return
    consumers = tuple(
        task
        for task in plan.semantic_graph.tasks
        if any(
            isinstance(port.source, HostServiceInputSource) and port.source.schema == value.schema
            for port in task.inputs
        )
    )
    if not consumers:
        raise ValueError("signed workload does not consume this auxiliary input")

    def shape(tensor: T) -> tuple[int, ...]:
        return describe(tensor).shape

    match value:
        case CausalTrainBatch(input_ids=ids, labels=labels, attention_mask=mask):
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
            actual = (tokens,)
        case WeightedCEBatch(hidden=hidden, base_logits=logits, labels=labels, weights=weights, keys=keys):
            prefix = () if keys else (1,)
            (layout,) = plan.tensor_layout_request.layouts
            dtype = layout.layout.fields[0].dtype
            hidden_shape, logits_shape = prefix + shape(hidden), prefix + shape(logits)
            rows = hidden_shape[:2]
            if (
                len(hidden_shape) != 3
                or min(hidden_shape) < 1
                or len(logits_shape) != 3
                or logits_shape[:2] != rows
                or logits_shape[2] < 1
                or prefix + shape(labels) != rows
                or prefix + shape(weights) != rows
                or hidden_shape[0] != (len(keys) if keys else 1)
                or len(set(keys)) != len(keys)
                or describe(labels).dtype is not TensorDType.INT64
                or any(describe(tensor).dtype is not dtype for tensor in (hidden, logits, weights))
            ):
                raise ValueError("weighted-CE batch requires aligned states, rows and tensors")
            actual = (hidden_shape, logits_shape)
        case StateTensorUpdate(values=values, keys=keys):
            if not values:
                return
            (state,) = plan.semantic_graph.states
            (layout,) = plan.tensor_layout_request.layouts
            if (
                len(values) != len(layout.layout.fields)
                or state.scope is StateScope.INSTANCE
                and keys
                or state.scope is StateScope.REQUEST
                and not keys
                or len(set(keys)) != len(keys)
                or any(
                    shape(tensor) != ((len(keys),) if keys else ()) + field.shape
                    or describe(tensor).dtype is not field.dtype
                    for tensor, field in zip(values, layout.layout.fields)
                )
            ):
                raise ValueError("replacement tensors differ from the signed layout or state scope")
            return
    for task in consumers:
        limit = next((fact for fact in plan.timeline.task_resources if fact.task == task.symbol), None)
        if limit is None:
            continue  # No cost declaration: the original unbounded execution contract.
        if len(limit.input_shapes) != len(actual) or any(
            len(dims) != len(bounds) or any(size > bound for size, bound in zip(dims, bounds))
            for dims, bounds in zip(actual, limit.input_shapes)
        ):
            raise ValueError(f"{task.symbol}: auxiliary input exceeds its signed input bounds")


def map_auxiliary(value: AuxiliaryInput[T] | None, convert: Callable[[T], U]) -> AuxiliaryInput[U] | None:
    """Transform tensor storage without introducing a transport-specific input."""
    match value:
        case None:
            return None
        case CausalTrainBatch(input_ids=ids, labels=labels, attention_mask=mask):
            return CausalTrainBatch(convert(ids), convert(labels), None if mask is None else convert(mask))
        case StateTensorUpdate(values=values, keys=keys):
            return StateTensorUpdate(tuple(convert(item) for item in values), keys)
        case WeightedCEBatch(hidden=hidden, base_logits=logits, labels=labels, weights=weights, keys=keys):
            return WeightedCEBatch(convert(hidden), convert(logits), convert(labels), convert(weights), keys)
