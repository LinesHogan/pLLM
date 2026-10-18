"""Structured semantic programs, before task selection or storage lowering.

This is the architecture prototype's frontend contract. It has no page handles,
executable callbacks, schedules, or production runtime authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class ScalarType(StrEnum):
    INDEX = "index"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


@dataclass(frozen=True, slots=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class BankType:
    """An immutable, logically indexed sequence; no physical layout is implied."""

    element: DataType
    size: int | str


DataType: TypeAlias = ScalarType | TensorType | BankType


@dataclass(frozen=True, slots=True)
class Value:
    name: str
    type: DataType


@dataclass(frozen=True, slots=True)
class Operation:
    inputs: tuple[str, ...]
    outputs: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class Constant(Operation):
    value: int | float | bool


@dataclass(frozen=True, slots=True)
class Parameter(Operation):
    """Read a declared invocation size as an index value, including inside regions."""

    parameter: str


@dataclass(frozen=True, slots=True)
class Coordinate(Operation):
    """Declare an affine index value: constant + sum(coefficient * input).

    Terms reference this op's inputs by position, the same convention as
    resource access ranges referencing call inputs. Inputs must be in-scope
    index values: block arguments, loop cursors, Parameter results or other
    Coordinates. The combined value must stay within the checked index range
    over the enclosing domain; out-of-range or non-affine operands are
    rejected at verification/analysis, not silently degraded.
    """

    affine: AffineIndex


class ScalarKind(StrEnum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    LESS = "less"
    EQUAL = "equal"


@dataclass(frozen=True, slots=True)
class Scalar(Operation):
    kind: ScalarKind


@dataclass(frozen=True, slots=True)
class Call(Operation):
    function: str


@dataclass(frozen=True, slots=True)
class Select(Operation):
    pass


@dataclass(frozen=True, slots=True)
class Update(Operation):
    pass


@dataclass(frozen=True, slots=True)
class Snapshot(Operation):
    resource: str


@dataclass(frozen=True, slots=True)
class Publish(Operation):
    resource: str


@dataclass(frozen=True, slots=True)
class Scan(Operation):
    """Thread carry, keep trailing inputs fixed, and collect extra body results.

    The body receives the index followed by all inputs. It returns only the
    next carry and collected values; invariant inputs are not returned.
    """

    count: int | str
    body: Region
    invariants: int = 0


@dataclass(frozen=True, slots=True)
class Map(Operation):
    count: int | str
    body: Region


@dataclass(frozen=True, slots=True)
class If(Operation):
    then_region: Region
    else_region: Region


@dataclass(frozen=True, slots=True)
class While(Operation):
    """Test before each step, execute at most limit steps, return the final carry."""

    limit: int
    condition: Region
    body: Region


Op: TypeAlias = (
    Constant | Parameter | Coordinate | Scalar | Call | Select | Update | Snapshot | Publish | Scan | Map | If | While
)


@dataclass(frozen=True, slots=True)
class Block:
    arguments: tuple[Value, ...]
    operations: tuple[Op, ...]
    yields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Region:
    block: Block


@dataclass(frozen=True, slots=True)
class IndexTerm:
    argument: int
    coefficient: int = 1


@dataclass(frozen=True, slots=True)
class AffineIndex:
    constant: int = 0
    terms: tuple[IndexTerm, ...] = ()


@dataclass(frozen=True, slots=True)
class AccessRange:
    """A half-open interval in one logical resource dimension."""

    lower: AffineIndex
    upper: AffineIndex


class AccessMode(StrEnum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class ResourceAccess:
    resource: str
    mode: AccessMode
    ranges: tuple[AccessRange, ...] = ()


@dataclass(frozen=True, slots=True)
class Function:
    """A compiler-imported function signature, not a user assertion of purity.

    Inputs and outputs contain only data. Resources and indexed accesses
    describe effects; dependencies preserve their reference ordering.
    Empty resources permits dataflow use, but does not imply speculation safety.
    """

    name: str
    inputs: tuple[DataType, ...]
    outputs: tuple[DataType, ...]
    resources: tuple[str, ...] = ()
    accesses: tuple[ResourceAccess, ...] = ()


@dataclass(frozen=True, slots=True)
class Resource:
    """A family of atomic cells; each domain coordinate identifies one cell.

    Empty domain means one atomic resource. All payload fields snapshot and
    publish together; their types may differ. A BankType field is still atomic
    as a whole, unlike a family of independently published cells.
    The native binder must establish alias separation for the logical identities.
    """

    name: str
    payload: tuple[DataType, ...]
    domain: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ExternalResource:
    """An opaque effect owned by a backend; accessed only through imported calls."""

    name: str
    domain: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class BorrowedResource:
    """External indexed storage requiring an invocation-scoped alias proof.

    The original backend owns its storage and recovery. A runtime binding must
    prove writable-cell separation and enforce every imported access interval.
    """

    name: str
    domain: tuple[int, ...]


ResourceType: TypeAlias = Resource | ExternalResource | BorrowedResource


@dataclass(frozen=True, slots=True)
class SizeParameter:
    """An invocation-invariant logical extent, bounded before any effect runs."""

    name: str
    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class ParameterTerm:
    parameter: str
    coefficient: int = 1


@dataclass(frozen=True, slots=True)
class SizeConstraint:
    """An entry guard: the weighted sum of named sizes must not exceed upper."""

    terms: tuple[ParameterTerm, ...]
    upper: int


@dataclass(frozen=True, slots=True)
class Program:
    entry: Region
    functions: tuple[Function, ...] = ()
    resources: tuple[ResourceType, ...] = ()
    parameters: tuple[SizeParameter, ...] = ()
    constraints: tuple[SizeConstraint, ...] = ()


class InvalidProgram(ValueError):
    pass


def bind_sizes(program: Program, supplied: Mapping[str, int]) -> dict[str, int]:
    """Bind exactly the declared shape parameters; never infer sizes mid-program."""
    if set(supplied) != {parameter.name for parameter in program.parameters}:
        raise ValueError("size parameter bindings must be complete and exact")
    sizes = dict(supplied)
    for parameter in program.parameters:
        value = sizes[parameter.name]
        if type(value) is not int or not parameter.minimum <= value <= parameter.maximum:
            raise ValueError("size parameter is outside its declared bounds")
    for constraint in program.constraints:
        if sum(term.coefficient * sizes[term.parameter] for term in constraint.terms) > constraint.upper:
            raise ValueError("size parameter bindings violate a declared relation")
    return sizes


def resolve_extent(extent: int | str, sizes: Mapping[str, int] = {}) -> int:
    if type(extent) is int:
        return extent
    if isinstance(extent, str) and extent in sizes:
        return sizes[extent]
    raise ValueError("symbolic extent requires its original size parameter binding")


def verify(program: Program, *, declaration: Function | None = None) -> tuple[DataType, ...]:
    """Check source semantics with the same authority used by native planning.

    The codec enforces the Python representation at this boundary. Rust checks
    SSA, operation signatures, resources and the optional outer declaration.
    """
    from pllm import _core
    from pllm.contracts.codec import encode_boundary

    try:
        _core.validate_region_program(
            encode_boundary(program), None if declaration is None else encode_boundary(declaration)
        )
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        raise InvalidProgram("invalid source payload: " + str(error)) from error
    block = program.entry.block
    values = {value.name: value.type for value in block.arguments}
    values.update((value.name, value.type) for operation in block.operations for value in operation.outputs)
    return tuple(values[name] for name in block.yields)
