"""Connect an original exported graph to fields of an existing readonly owner."""

from collections.abc import Mapping

from pllm.contracts.region_composition import RegionFunctionBody, inline_region
from pllm.contracts.region_imports import BorrowedTensorImport
from pllm.contracts.region_program import (
    AccessMode,
    AccessRange,
    AffineIndex,
    Block,
    BorrowedResource,
    Call,
    DataType,
    Function,
    IndexTerm,
    Program,
    Region,
    ResourceAccess,
    ScalarType,
    TensorType,
    Value,
    verify,
)
from pllm.execution.codec import boundary_digest
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.exported_region import ExportedRegion
from pllm.execution.region_native import BorrowedTensorPrimitive, TensorSource


def borrow_exported_inputs(
    source: ExportedCell | ExportedRegion, resource: BorrowedResource, fields: Mapping[str, int]
) -> tuple[RegionFunctionBody, dict[str, TensorSource]]:
    """Outline a call whose lifted tensor ports borrow one original resource cell.

    ``fields`` maps source port names (normally export parameter/buffer targets)
    to the owner's existing field ordinals. Remaining tensor ports come first,
    followed by the resource coordinates. Inline the returned body in its caller
    and bind the same owner's readonly view through RegionResources.

    This only reads original versions. Functional mutation outputs stay visible
    to the caller; publishing a new version still belongs to the original owner.
    No tensor or module is captured in the generated accessors.
    """
    if isinstance(source, ExportedCell):
        inputs = tuple(Value(name, type_) for name, type_ in zip(source.inputs, source.declaration.inputs, strict=True))
        prefix = "output"
        while any(name.startswith(prefix) for name in source.inputs):
            prefix += "_"
        outputs = tuple(Value(f"{prefix}{i}", type_) for i, type_ in enumerate(source.declaration.outputs))
        program = Program(
            Region(
                Block(
                    inputs,
                    (Call(tuple(value.name for value in inputs), outputs, source.digest),),
                    tuple(value.name for value in outputs),
                )
            ),
            (source.declaration,),
        )
        functions: dict[str, TensorSource] = {source.digest: source}
    else:
        program, functions = source.program, dict(source.functions)
    inputs = program.entry.block.arguments
    input_names = tuple(value.name for value in inputs)

    def tensor_type(type_: DataType) -> TensorType:
        if not isinstance(type_, TensorType):
            raise ValueError("export borrowing requires tensor ports")
        return type_

    input_types = tuple(tensor_type(value.type) for value in inputs)
    output_types = tuple(tensor_type(type_) for type_ in verify(program))
    if program.resources or program.parameters or program.constraints:
        raise ValueError("export borrowing requires a static tensor region with lifted state")
    if not fields or set(fields) - set(input_names):
        raise ValueError("export borrowing requires known source input ports")
    if any(type(field) is not int or not 0 <= field < 2**63 for field in fields.values()):
        raise ValueError("export borrowing requires nonnegative original field ordinals")
    ordinary = tuple(i for i, name in enumerate(input_names) if name not in fields)
    selected = tuple((i, fields[name]) for i, name in enumerate(input_names) if name in fields)
    signatures = tuple(source.input_signatures[i] for i, _ in selected) if isinstance(source, ExportedCell) else ()
    for signature in signatures:
        stride = 1
        for size, actual in zip(signature.shape[::-1], signature.stride[::-1], strict=True):
            if actual != stride:
                raise ValueError("export borrowing requires the owner's contiguous tensor field ABI")
            stride *= max(1, size)
    identity = boundary_digest("borrow-export-inputs@2", program, resource, selected)
    coordinate_types = (ScalarType.INDEX,) * len(resource.domain)
    coordinates = tuple(Value(f"coordinate{i}", type_) for i, type_ in enumerate(coordinate_types))
    arguments = tuple(Value(f"input{i}", inputs[i].type) for i in ordinary)

    def access(offset: int) -> ResourceAccess:
        return ResourceAccess(
            resource.name,
            AccessMode.READ,
            tuple(
                AccessRange(AffineIndex(0, (IndexTerm(i + offset),)), AffineIndex(1, (IndexTerm(i + offset),)))
                for i in range(len(resource.domain))
            ),
        )

    declaration = Function(
        "borrowed_graph@1:" + identity,
        (*tuple(input_types[i] for i in ordinary), *coordinate_types),
        output_types,
        (resource.name,),
        (access(len(arguments)),),
    )
    operations: list[Call] = []
    tensor_names = {i: f"input{i}" for i in ordinary}
    # Equal fields may feed tied parameter ports. Reuse the same read and SSA
    # value when their signatures agree; do not manufacture independent copies.
    borrowed: dict[BorrowedTensorImport, str] = {}
    for ordinal, field in selected:
        getter = Function(
            "borrowed_field@1:" + boundary_digest(resource, field, input_types[ordinal]),
            coordinate_types,
            (input_types[ordinal],),
            (resource.name,),
            (access(0),),
        )
        imported = BorrowedTensorImport(getter, field)
        if imported in borrowed:
            tensor_names[ordinal] = borrowed[imported]
            continue
        primitive = BorrowedTensorPrimitive(imported)
        functions[getter.name] = primitive
        name = f"input{ordinal}"
        operations.append(
            Call(tuple(value.name for value in coordinates), (Value(name, inputs[ordinal].type),), getter.name)
        )
        tensor_names[ordinal], borrowed[imported] = name, name
    inlined = inline_region(
        program, tuple(Value(tensor_names[i], value.type) for i, value in enumerate(inputs)), "export"
    )
    body = RegionFunctionBody(
        declaration,
        Program(
            Region(
                Block(
                    (*arguments, *coordinates),
                    (*operations, *inlined.operations),
                    tuple(value.name for value in inlined.results),
                )
            ),
            tuple(function.declaration for function in functions.values()),
            (resource,),
        ),
    )
    body.validate()
    return body, functions
