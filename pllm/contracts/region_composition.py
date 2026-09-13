"""Lexical source composition before dependence analysis or scheduling."""

from dataclasses import dataclass, replace

from pllm.contracts.region_program import (
    AffineIndex, Block, Call, Coordinate, Function, If, Map, Op, Program, Region, Scalar, ScalarKind,
    ScalarType, Scan, Value, While, verify,
)


@dataclass(frozen=True, slots=True)
class RegionFunctionBody:
    """Backend source definition, before scheduling; not an equivalence claim.

    The original provider must reconstruct the same definition when binding its
    endpoints. Core verification checks the typed body and resource interface;
    it cannot prove that arbitrary Python implements this source definition.
    """

    declaration: Function
    program: Program

    def validate(self) -> None:
        verify(self.program, declaration=self.declaration)


@dataclass(frozen=True, slots=True)
class InlinedRegion:
    operations: tuple[Op, ...]
    results: tuple[Value, ...]


def select_function_bodies(bodies: tuple[RegionFunctionBody, ...], *, selector: str) -> RegionFunctionBody:
    """Compose source variants at an index in the caller's bounded domain.

    The caller establishes ``0 <= selector < len(bodies)`` (for example through
    an existing scan). Selection preserves the complete tensor/resource ABI.
    Equal contiguous definitions stay inlined; other ranges use ordinary If
    regions. This does not add a runtime registry or claim affine scheduling
    through the resulting branches.
    """
    if not bodies:
        raise ValueError("source selection needs at least one body")
    first = bodies[0]
    arguments = first.program.entry.block.arguments
    if not any(value.name == selector and value.type is ScalarType.INDEX for value in arguments):
        raise ValueError("source selector must be an index argument")
    functions: dict[str, Function] = {}
    for body in bodies:
        body.validate()
        if (body.declaration != first.declaration or body.program.resources != first.program.resources):
            raise ValueError("source variants must share their external ports and resource domain")
        for function in body.program.functions:
            if function.name in functions and functions[function.name] != function:
                raise ValueError("source variants disagree on an imported function")
            functions[function.name] = function
    if all(body == first for body in bodies):
        return first
    used = {value.name for value in arguments}
    def fresh(name: str) -> str:
        while name in used:
            name += "_"
        used.add(name)
        return name
    results = tuple(Value(fresh(f"selected{i}"), type_) for i, type_ in enumerate(first.declaration.outputs))
    split, before = Value(fresh("split_index"), ScalarType.INDEX), Value(fresh("before_split"), ScalarType.BOOL)
    def select(start: int, stop: int) -> Region:
        if all(body == bodies[start] for body in bodies[start:stop]):
            inlined = _inline_block(bodies[start].program.entry.block, arguments, "source")
            return Region(Block(arguments, inlined.operations, tuple(value.name for value in inlined.results)))
        middle = (start + stop) // 2
        return Region(Block(arguments, (
            Coordinate((), (split,), AffineIndex(middle)),
            Scalar((selector, split.name), (before,), ScalarKind.LESS),
            If((before.name, *(value.name for value in arguments)), results, select(start, middle), select(middle, stop)),
        ), tuple(value.name for value in results)))
    result = RegionFunctionBody(first.declaration, Program(select(0, len(bodies)),
        tuple(functions[name] for name in sorted(functions)), first.program.resources))
    result.validate()
    return result


def inline_region(program: Program, arguments: tuple[Value, ...], prefix: str) -> InlinedRegion:
    """Splice a closed source block into its caller without an execution boundary.

    The caller merges the original function, resource, parameter and constraint declarations
    and verifies the resulting Program. Child regions have their own explicit
    block arguments, so their lexical names need no substitution.
    """
    verify(program)
    return _inline_block(program.entry.block, arguments, prefix)


def _inline_block(block: Block, arguments: tuple[Value, ...], prefix: str) -> InlinedRegion:
    """Rename a checked block; the enclosing composition verifies the result."""
    if tuple(value.type for value in arguments) != tuple(value.type for value in block.arguments):
        raise ValueError("composed region arguments differ from its source signature")
    values = {formal.name: actual for formal, actual in zip(block.arguments, arguments, strict=True)}
    used = {value.name for value in arguments}
    operations: list[Op] = []
    for operation in block.operations:
        inputs = tuple(values[name].name for name in operation.inputs)
        outputs: list[Value] = []
        for value in operation.outputs:
            name = prefix + "_" + value.name
            while name in used:
                name += "_"
            used.add(name)
            renamed = Value(name, value.type)
            values[value.name] = renamed
            outputs.append(renamed)
        operations.append(replace(operation, inputs=inputs, outputs=tuple(outputs)))
    return InlinedRegion(tuple(operations), tuple(values[name] for name in block.yields))


def inline_function_bodies(program: Program, bodies: tuple[RegionFunctionBody, ...]) -> Program:
    """Expand source definitions into the caller's existing regions and owner.

    Definitions are one level deep; recursive or mutually dependent imports are
    rejected. The expanded Program, rather than the former call's aggregate
    effects, is the source of dependence and storage analysis.
    """
    verify(program)
    definitions = {body.declaration.name: body for body in bodies}
    if len(definitions) != len(bodies):
        raise ValueError("source definitions repeat a function name")
    functions = {function.name: function for function in program.functions}
    resources = {resource.name: resource for resource in program.resources}
    for body in bodies:
        body.validate()
        if body.declaration.name in functions and functions[body.declaration.name] != body.declaration:
            raise ValueError("function definition differs from the caller declaration")
        if any(function.name in definitions for function in body.program.functions):
            raise ValueError("nested or recursive source definitions are unsupported")
    expanded: set[str] = set()

    def compose(source: Region) -> Region:
        block = source.block
        names = {value.name: value.name for value in block.arguments}
        types = {value.name: value.type for value in block.arguments}
        used = {value.name for value in block.arguments}
        used.update(value.name for operation in block.operations for value in operation.outputs)
        operations: list[Op] = []
        for ordinal, original in enumerate(block.operations):
            operation = replace(original, inputs=tuple(names[name] for name in original.inputs))
            if isinstance(operation, (Scan, Map)):
                operation = replace(operation, body=compose(operation.body))
            elif isinstance(operation, If):
                operation = replace(operation, then_region=compose(operation.then_region), else_region=compose(operation.else_region))
            elif isinstance(operation, While):
                operation = replace(operation, condition=compose(operation.condition), body=compose(operation.body))
            body = definitions.get(operation.function) if isinstance(operation, Call) else None
            if body is None:
                operations.append(operation)
                names.update((value.name, value.name) for value in original.outputs)
            else:
                expanded.add(body.declaration.name)
                for resource in body.program.resources:
                    if resources.get(resource.name) != resource:
                        raise ValueError("function body differs from the original resource family")
                for function in body.program.functions:
                    if function.name in functions and functions[function.name] != function:
                        raise ValueError("function body imports conflicting declarations")
                    functions[function.name] = function
                arguments = tuple(Value(names[name], types[name]) for name in original.inputs)
                prefix = f"body{ordinal}"
                while True:
                    inlined = _inline_block(body.program.entry.block, arguments, prefix)
                    introduced = {value.name for item in inlined.operations for value in item.outputs}
                    if not introduced.intersection(used):
                        break
                    prefix += "_"
                used.update(introduced)
                operations.extend(inlined.operations)
                names.update((old.name, value.name) for old, value in zip(original.outputs, inlined.results, strict=True))
            types.update((value.name, value.type) for value in original.outputs)
        return Region(Block(block.arguments, tuple(operations), tuple(names[name] for name in block.yields)))

    entry = compose(program.entry)
    result = replace(program, entry=entry, functions=tuple(value for name, value in functions.items() if name not in expanded))
    verify(result)
    return result
