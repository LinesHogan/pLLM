"""One closed wire codec shared by source validation and execution boundaries."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import IntEnum, StrEnum
from hashlib import sha256
from typing import Protocol, TypeVar, cast, get_args, get_origin

import torch
from cattrs import Converter, gen
from cattrs.errors import ForbiddenExtraKeysError
from cattrs.strategies import configure_tagged_union

from pllm.contracts.control_instructions import (
    ControlAction,
    ControlBlock,
    ControlIf,
    ControlMap,
    ControlPrimitive,
    ControlScan,
    ControlWhile,
)
from pllm.contracts.region_imports import (
    AdditiveScanImport,
    BorrowedTensorImport,
    CompiledTensorImport,
    ExportedTensorImport,
    FunctionalKernelImport,
    NativeKernelImport,
    NativeTensorImport,
    RegionTensorImport,
)
from pllm.contracts.region_program import (
    AccessMode,
    BankType,
    BorrowedResource,
    Call,
    Constant,
    Coordinate,
    DataType,
    ExternalResource,
    If,
    Map,
    Op,
    Parameter,
    ParameterTerm,
    Publish,
    Region,
    Resource,
    ResourceType,
    Scalar,
    ScalarKind,
    ScalarType,
    Scan,
    Select,
    SizeParameter,
    Snapshot,
    TensorType,
    Update,
    While,
)
from pllm.contracts.schedule_instructions import (
    ScheduleApply,
    ScheduleExpression,
    ScheduleGuard,
    ScheduleInstruction,
    ScheduleInteger,
    ScheduleInvoke,
    ScheduleIterator,
    ScheduleLoop,
    ScheduleSequence,
)
from pllm.contracts.semantic_graph import (
    ConditionalLayerReplay,
    HiddenInputSource,
    HostServiceInputSource,
    PackedRowsInputSource,
    PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource,
    SemanticInputSource,
    StateSelectionInputSource,
    StepIdInputSource,
    TaskResultInputSource,
    TokenwiseLayerReplay,
)

T = TypeVar("T")
E = TypeVar("E", bound=StrEnum)
_converter = Converter(forbid_extra_keys=True, detailed_validation=False)
_converter.register_unstructure_hook(torch.device, str)
_converter.register_structure_hook(torch.device, lambda value, _target: torch.device(boundary_scalar(value, str)))


def boundary_scalar(value: object, target: type[T]) -> T:
    if target is float and type(value) is int:
        return cast(T, float(value))
    if type(value) is not target:
        raise TypeError("boundary scalar has the wrong type")
    return cast(T, value)


for _type in (str, int, float, bool):
    _converter.register_structure_hook(_type, boundary_scalar)


def _enum(value: object, target: type[E]) -> E:
    return target(boundary_scalar(value, str))


_converter.register_structure_hook(StrEnum, _enum)


def _enum_encoder(target: type[E]) -> Callable[[E], str]:
    def encode(value: E) -> str:
        if type(value) is not target:
            raise TypeError("boundary enum has the wrong type")
        return value.value

    return encode


for _source_enum in (ScalarKind, AccessMode):
    _converter.register_unstructure_hook(_source_enum, _enum_encoder(_source_enum))

# Source indices must not be objects that merely behave like Python integers.
_converter.register_unstructure_hook(int, lambda value: boundary_scalar(value, int))
_converter.register_unstructure_hook(bool, lambda value: boundary_scalar(value, bool))
_converter.register_unstructure_hook(IntEnum, lambda value: value.value)


class _TupleCodegen(Protocol):
    def make_iterable_unstructure_fn(
        self, target: object, converter: Converter, unstructure_to: type[tuple[object, ...]]
    ) -> Callable[[tuple[object, ...]], object]: ...

    def make_hetero_tuple_unstructure_fn(
        self, target: object, converter: Converter, unstructure_to: type[tuple[object, ...]]
    ) -> Callable[[tuple[object, ...]], object]: ...


def _tuple_encoder(target: object) -> Callable[[object], object]:
    args = get_args(target)
    variable = not args or args[-1] is Ellipsis
    codegen = cast(_TupleCodegen, gen)
    encode = (
        codegen.make_iterable_unstructure_fn(target, _converter, tuple)
        if variable
        else codegen.make_hetero_tuple_unstructure_fn(target, _converter, tuple)
    )

    def checked(value: object) -> object:
        if type(value) is not tuple:
            raise TypeError("boundary sequence must be a tuple")
        items = cast(tuple[object, ...], value)
        if not variable and len(items) != len(args):
            raise ValueError("boundary tuple has the wrong arity")
        return encode(items)

    return checked


_converter.register_unstructure_hook_factory(lambda target: get_origin(target) is tuple, _tuple_encoder)

# Keep cattrs' typed tuple handler, but reject iterable objects that are not JSON arrays.
_tuple_handler = _converter.get_structure_hook(tuple)


def _array(value: object, target: object) -> object:
    if type(value) is not list:
        raise TypeError("boundary sequence must be a JSON array")
    return _tuple_handler(value, target)


_converter.register_structure_hook_func(lambda target: get_origin(target) is tuple, _array)

_input_tags: dict[type[object], str] = {
    HiddenInputSource: "hidden",
    StepIdInputSource: "step_id",
    PackedRowsInputSource: "packed_rows",
    PackedSamplerRowsInputSource: "packed_sampler_rows",
    SamplerFeedbackInputSource: "sampler_feedback",
    HostServiceInputSource: "host_service",
    StateSelectionInputSource: "state_selection",
    TaskResultInputSource: "result",
}
configure_tagged_union(SemanticInputSource, _converter, tag_generator=_input_tags.__getitem__, tag_name="kind")


def _replay(value: object, _target: object) -> ConditionalLayerReplay | TokenwiseLayerReplay | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise TypeError("replay effect must be a closed object")
    if "layers" in value:
        return _converter.structure(value, ConditionalLayerReplay)
    return _converter.structure(value, TokenwiseLayerReplay)


_converter.register_structure_hook(ConditionalLayerReplay | TokenwiseLayerReplay | None, _replay)


def _region_type_wire(value: DataType) -> dict[str, object]:
    if type(value) is ScalarType:
        return {"kind": "scalar", "name": value.value}
    if type(value) is TensorType:
        return {
            "kind": "tensor",
            "shape": [boundary_scalar(d, int) for d in value.shape],
            "dtype": value.dtype,
            "device": value.device,
        }
    if type(value) is BankType:
        return {
            "kind": "bank",
            "element": _region_type_wire(value.element),
            "size": _region_extent(value.size, int | str),
        }
    raise ValueError("unknown region data type")


def _region_type(value: object, _target: object) -> DataType:
    if type(value) is not dict:
        raise ValueError("region type must be a closed tagged object")
    fields = cast(dict[str, object], value)
    kind = fields.get("kind")
    if kind == "scalar" and set(fields) == {"kind", "name"}:
        return ScalarType(boundary_scalar(fields["name"], str))
    if kind == "tensor" and set(fields) == {"kind", "shape", "dtype", "device"}:
        return TensorType(
            _converter.structure(fields["shape"], tuple[int, ...]),
            boundary_scalar(fields["dtype"], str),
            boundary_scalar(fields["device"], str),
        )
    if kind == "bank" and set(fields) == {"kind", "element", "size"}:
        return BankType(_region_type(fields["element"], DataType), _region_extent(fields["size"], int | str))
    raise ValueError("unknown region type or type fields")


def _region_literal(value: object, _target: object) -> int | float | bool:
    if type(value) not in (int, float, bool):
        raise ValueError("region constant must be a scalar literal")
    if type(value) is int and not -(2**63) <= value < 2**63:
        raise ValueError("integer constant exceeds signed 64-bit range")
    return cast(int | float | bool, value)


_converter.register_unstructure_hook(DataType, _region_type_wire)
_converter.register_structure_hook(DataType, _region_type)
_converter.register_structure_hook(int | float | bool, _region_literal)
_converter.register_unstructure_hook(int | float | bool, lambda value: _region_literal(value, int | float | bool))


def _region_extent(value: object, _target: object) -> int | str:
    if type(value) not in (int, str):
        raise ValueError("region extent must be an integer or size parameter name")
    return cast(int | str, value)


_converter.register_structure_hook(int | str, _region_extent)
_converter.register_unstructure_hook(int | str, lambda value: _region_extent(value, int | str))
_converter.register_unstructure_hook(
    SizeParameter,
    lambda value: {
        "name": boundary_scalar(value.name, str),
        "minimum": boundary_scalar(value.minimum, int),
        "maximum": boundary_scalar(value.maximum, int),
    },
)
_converter.register_unstructure_hook(
    ParameterTerm,
    lambda value: {
        "parameter": boundary_scalar(value.parameter, str),
        "coefficient": boundary_scalar(value.coefficient, int),
    },
)


_resource_tags: dict[type[object], str] = {
    Resource: "state",
    ExternalResource: "external",
    BorrowedResource: "borrowed",
}
configure_tagged_union(ResourceType, _converter, tag_generator=_resource_tags.__getitem__, tag_name="kind")


def _region_wire(value: Region) -> dict[str, object]:
    # Resolve the operation union at invocation time. Eager generation of a
    # recursive union's handlers otherwise caches untagged nested operations.
    block = value.block
    return {
        "block": {
            "arguments": [_converter.unstructure(v) for v in block.arguments],
            "operations": [_converter.unstructure(op, unstructure_as=Op) for op in block.operations],
            "yields": list(block.yields),
        }
    }


_converter.register_unstructure_hook(Region, _region_wire)
_region_op_tags: dict[type[object], str] = {
    Constant: "constant",
    Parameter: "parameter",
    Coordinate: "coordinate",
    Scalar: "scalar",
    Call: "call",
    Select: "select",
    Update: "update",
    Snapshot: "snapshot",
    Publish: "publish",
    Scan: "scan",
    Map: "map",
    If: "if",
    While: "while",
}
configure_tagged_union(Op, _converter, tag_generator=_region_op_tags.__getitem__, tag_name="op")


def _control_block_wire(value: ControlBlock) -> dict[str, object]:
    return {
        "arguments": [_converter.unstructure(a) for a in value.arguments],
        "steps": [
            {
                "inputs": list(s.inputs),
                "outputs": [_converter.unstructure(o) for o in s.outputs],
                "action": _converter.unstructure(s.action, unstructure_as=ControlAction),
                "releases": list(s.releases),
            }
            for s in value.steps
        ],
        "results": list(value.results),
        "discard_arguments": list(value.discard_arguments),
    }


_converter.register_unstructure_hook(ControlBlock, _control_block_wire)
_control_tags: dict[type[object], str] = {
    ControlPrimitive: "primitive",
    ControlIf: "if",
    ControlWhile: "while",
    ControlScan: "scan",
    ControlMap: "map",
}
configure_tagged_union(ControlAction, _converter, tag_generator=_control_tags.__getitem__, tag_name="kind")


def _schedule_apply_wire(value: ScheduleApply) -> dict[str, object]:
    return {
        "operator": value.operator.value,
        "arguments": [_converter.unstructure(a, unstructure_as=ScheduleExpression) for a in value.arguments],
    }


_converter.register_unstructure_hook(ScheduleApply, _schedule_apply_wire)
_expression_tags: dict[type[object], str] = {
    ScheduleInteger: "integer",
    ScheduleIterator: "iterator",
    ScheduleApply: "apply",
}
configure_tagged_union(ScheduleExpression, _converter, tag_generator=_expression_tags.__getitem__, tag_name="kind")


def _schedule_loop_wire(value: ScheduleLoop) -> dict[str, object]:
    return {
        "iterator": value.iterator,
        "initial": _converter.unstructure(value.initial, unstructure_as=ScheduleExpression),
        "condition": _converter.unstructure(value.condition, unstructure_as=ScheduleExpression),
        "step": _converter.unstructure(value.step, unstructure_as=ScheduleExpression),
        "body": _converter.unstructure(value.body, unstructure_as=ScheduleInstruction),
    }


def _schedule_sequence_wire(value: ScheduleSequence) -> dict[str, object]:
    return {"children": [_converter.unstructure(n, unstructure_as=ScheduleInstruction) for n in value.children]}


def _schedule_guard_wire(value: ScheduleGuard) -> dict[str, object]:
    return {
        "condition": _converter.unstructure(value.condition, unstructure_as=ScheduleExpression),
        "if_true": _converter.unstructure(value.if_true, unstructure_as=ScheduleInstruction),
        "if_false": _converter.unstructure(value.if_false, unstructure_as=ScheduleInstruction),
    }


_converter.register_unstructure_hook(ScheduleLoop, _schedule_loop_wire)
_converter.register_unstructure_hook(ScheduleSequence, _schedule_sequence_wire)
_converter.register_unstructure_hook(ScheduleGuard, _schedule_guard_wire)
_instruction_tags: dict[type[object], str] = {
    ScheduleInvoke: "invoke",
    ScheduleLoop: "loop",
    ScheduleSequence: "sequence",
    ScheduleGuard: "guard",
}
configure_tagged_union(ScheduleInstruction, _converter, tag_generator=_instruction_tags.__getitem__, tag_name="kind")


_tensor_import_tags = {
    ExportedTensorImport: "exported",
    NativeTensorImport: "native",
    NativeKernelImport: "native_kernel",
    FunctionalKernelImport: "functional_kernel",
    CompiledTensorImport: "compiled",
    BorrowedTensorImport: "borrowed",
    AdditiveScanImport: "additive_scan",
}
configure_tagged_union(RegionTensorImport, _converter, tag_generator=_tensor_import_tags.__getitem__, tag_name="kind")
# Imports cross the boundary as a heterogeneous tuple. Runtime tuple encoding
# dispatches on the concrete member, so retain its closed union discriminator.
_converter.register_unstructure_hook(
    ExportedTensorImport, lambda value: _converter.unstructure(value, unstructure_as=RegionTensorImport)
)
_converter.register_unstructure_hook(
    NativeTensorImport, lambda value: _converter.unstructure(value, unstructure_as=RegionTensorImport)
)
_converter.register_unstructure_hook(
    NativeKernelImport, lambda value: _converter.unstructure(value, unstructure_as=RegionTensorImport)
)
_converter.register_unstructure_hook(
    FunctionalKernelImport, lambda value: _converter.unstructure(value, unstructure_as=RegionTensorImport)
)
_converter.register_unstructure_hook(
    CompiledTensorImport, lambda value: _converter.unstructure(value, unstructure_as=RegionTensorImport)
)
_converter.register_unstructure_hook(
    BorrowedTensorImport, lambda value: _converter.unstructure(value, unstructure_as=RegionTensorImport)
)


_converter.register_unstructure_hook(
    AdditiveScanImport, lambda value: _converter.unstructure(value, unstructure_as=RegionTensorImport)
)


def _native_kernel_import(value: object, _target: object) -> NativeKernelImport:
    result = _converter.structure(value, cast(type[RegionTensorImport], RegionTensorImport))
    if not isinstance(result, NativeKernelImport):
        raise ValueError("backend kernel catalog requires a native kernel import")
    return result


_converter.register_structure_hook(NativeKernelImport, _native_kernel_import)


def _functional_kernel_import(value: object, _target: object) -> FunctionalKernelImport:
    result = _converter.structure(value, cast(type[RegionTensorImport], RegionTensorImport))
    if not isinstance(result, FunctionalKernelImport):
        raise ValueError("functional kernel catalog requires a functional kernel import")
    return result


_converter.register_structure_hook(FunctionalKernelImport, _functional_kernel_import)


def _additive_scan_import(value: object, _target: object) -> AdditiveScanImport:
    result = _converter.structure(value, cast(type[RegionTensorImport], RegionTensorImport))
    if not isinstance(result, AdditiveScanImport):
        raise ValueError("additive scan needs its verified kernel import")
    return result


_converter.register_structure_hook(AdditiveScanImport, _additive_scan_import)


def _borrowed_tensor_import(value: object, _target: object) -> BorrowedTensorImport:
    result = _converter.structure(value, cast(type[RegionTensorImport], RegionTensorImport))
    if not isinstance(result, BorrowedTensorImport):
        raise ValueError("tensor borrowing requires an original resource import")
    result.validate()
    return result


_converter.register_structure_hook(BorrowedTensorImport, _borrowed_tensor_import)


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate boundary field")
    return result


def _nonfinite(value: str) -> object:
    raise ValueError("non-finite boundary scalar: " + value)


def encode_boundary(value: object) -> str:
    try:
        return json.dumps(_converter.unstructure(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, AttributeError, KeyError) as error:
        raise ValueError("invalid closed boundary value: " + str(error)) from error


def boundary_digest(*values: object) -> str:
    """Bind original typed parameters and cold facts with the same closed encoding."""
    return sha256(encode_boundary(values).encode("utf-8")).hexdigest()


def decode_boundary(payload: str, target: type[T]) -> T:
    """The caller supplies a statically bound type, never a type named by payload."""
    try:
        return _converter.structure(json.loads(payload, object_pairs_hook=_object, parse_constant=_nonfinite), target)
    except (TypeError, AttributeError, KeyError, ValueError, ForbiddenExtraKeysError) as exc:
        raise ValueError("invalid closed boundary payload") from exc
