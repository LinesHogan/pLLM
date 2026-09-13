"""Small serial oracle for structured IR, never a production execution path.

Copies make value semantics explicit. The in-memory resource map models visible
contents only: it makes no claims about PTTS permissions, COW, or GPU completion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import torch

from pllm.execution.region_values import Bank, Data, check_data, copy_data

from pllm.contracts.region_program import (
    BankType, Call, Constant, Coordinate, DataType, If, Map, Parameter, Program, Resource, Publish,
    Region, Scalar, ScalarKind, Scan, Select, Snapshot,
    Update, verify, bind_sizes, resolve_extent,
)


ReferenceFunction: TypeAlias = Callable[[tuple[Data, ...]], tuple[Data, ...]]


@dataclass(frozen=True, slots=True)
class EffectEvent:
    kind: str
    resource: str
    sequence: int
    coordinates: tuple[int, ...]
    indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    values: tuple[Data, ...]
    resources: dict[str, tuple[Data, ...]]
    effects: tuple[EffectEvent, ...]
    calls: tuple[tuple[str, tuple[int, ...]], ...]


def _scalar(op: Scalar, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
    left, right = inputs
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        raise ValueError("scalar operation received nonscalar data")
    if op.kind is ScalarKind.ADD:
        return (left + right,)
    if op.kind is ScalarKind.SUB:
        return (left - right,)
    if op.kind is ScalarKind.MUL:
        return (left * right,)
    if op.kind is ScalarKind.LESS:
        return (left < right,)
    return (left == right,)


def interpret(program: Program, inputs: Mapping[str, Data], *,
              resources: Mapping[str, tuple[Data, ...]] | None = None,
              functions: Mapping[str, ReferenceFunction] | None = None,
              sizes: Mapping[str, int] = {}) -> ReferenceResult:
    """Execute verified reference order; all input tensors remain caller-owned.

    Function implementations here are trusted test/import adapters. For an
    effectful call, the whole family for each payload field is appended to data
    inputs in declared resource/field order; the adapter returns the updated
    fields after its data outputs. Resource dictionaries hold tuples of these
    field families, so heterogeneous atomic payloads need no runtime tuple value. The interpreter
    checks and records them. These extra arguments are oracle adapters, not the
    production primitive ABI. This is never used as a worker fallback.
    """
    verify(program)
    sizes = bind_sizes(program, sizes)
    supplied: Mapping[str, tuple[Data, ...]] = {} if resources is None else resources
    implementations: Mapping[str, ReferenceFunction] = {} if functions is None else functions
    if set(supplied) != {r.name for r in program.resources if isinstance(r, Resource)}:
        raise ValueError("reference resource bindings must be complete and exact")
    arguments = program.entry.block.arguments
    ranks = {r.name: len(r.domain) for r in program.resources}
    if set(inputs) != {v.name for v in arguments}:
        raise ValueError("reference input bindings must be complete and exact")
    declarations = {f.name: f for f in program.functions}
    if set(implementations) != set(declarations):
        raise ValueError("reference function bindings must be complete and exact")
    state = {r.name: tuple(copy_data(v) for v in supplied[r.name]) for r in program.resources if isinstance(r, Resource)}
    storage_types: dict[str, tuple[DataType, ...]] = {}
    for r in program.resources:
        if not isinstance(r, Resource):
            continue
        fields: list[DataType] = []
        for type_ in r.payload:
            for size in reversed(r.domain):
                type_ = BankType(type_, size)
            fields.append(type_)
        for value, type_ in zip(state[r.name], fields, strict=True):
            check_data(value, type_, sizes)
        storage_types[r.name] = tuple(fields)
    sequences = {r.name: 0 for r in program.resources}
    effects: list[EffectEvent] = []
    calls: list[tuple[str, tuple[int, ...]]] = []

    def advance(resource: str, kind: str, coordinates: tuple[int, ...],
                indices: tuple[int, ...] = ()) -> None:
        sequences[resource] += 1
        effects.append(EffectEvent(kind, resource, sequences[resource], coordinates, indices))

    def indices_of(values: tuple[Data, ...]) -> tuple[int, ...]:
        result: list[int] = []
        for value in values:
            if type(value) is not int:
                raise ValueError("resource family indices must be integers")
            result.append(value)
        return tuple(result)

    def select_cell(value: Data, indices: tuple[int, ...]) -> Data:
        for index in indices:
            if not isinstance(value, Bank) or not 0 <= index < len(value.elements):
                raise ValueError("resource family index out of bounds")
            value = value.elements[index]
        return value

    def replace_cell(value: Data, indices: tuple[int, ...], replacement: Data) -> Data:
        if not indices:
            return copy_data(replacement)
        index = indices[0]
        if not isinstance(value, Bank) or not 0 <= index < len(value.elements):
            raise ValueError("resource family index out of bounds")
        cell = replace_cell(value.elements[index], indices[1:], replacement)
        return Bank((*value.elements[:index], cell, *value.elements[index + 1:]))

    def run(region: Region, args: tuple[Data, ...],
            coordinates: tuple[int, ...]) -> tuple[Data, ...]:
        block = region.block
        env: dict[str, Data] = {}
        for spec, value in zip(block.arguments, args, strict=True):
            check_data(value, spec.type, sizes)
            env[spec.name] = value
        for op in block.operations:
            ins = tuple(env[name] for name in op.inputs)
            result: tuple[Data, ...]
            if isinstance(op, Constant):
                result = (op.value,)
            elif isinstance(op, Parameter):
                result = (sizes[op.parameter],)
            elif isinstance(op, Coordinate):
                value = op.affine.constant
                for term in op.affine.terms:
                    operand = ins[term.argument]
                    if type(operand) is not int:
                        raise ValueError("coordinate operands must be integers")
                    value += term.coefficient * operand
                result = (value,)
            elif isinstance(op, Scalar):
                result = _scalar(op, ins)
            elif isinstance(op, (Select, Update)):
                bank, index = ins[:2]
                if not isinstance(bank, Bank) or type(index) is not int:
                    raise ValueError("invalid reference bank access")
                if not 0 <= index < len(bank.elements):
                    raise ValueError("logical bank index is out of bounds")
                if isinstance(op, Select):
                    result = (bank.elements[index],)
                else:
                    replacement = ins[2]
                    result = (Bank((*bank.elements[:index], replacement, *bank.elements[index + 1:])),)
            elif isinstance(op, Snapshot):
                indices = indices_of(ins[:ranks[op.resource]])
                contents = tuple(copy_data(select_cell(v, indices)) for v in state[op.resource])
                advance(op.resource, "snapshot", coordinates, indices)
                result = contents
            elif isinstance(op, Publish):
                width = len(storage_types[op.resource])
                values = ins[:width]
                indices = indices_of(ins[width:width + ranks[op.resource]])
                contents = tuple(replace_cell(old, indices, value) for old, value in
                    zip(state[op.resource], values, strict=True))
                advance(op.resource, "publish", coordinates, indices)
                state[op.resource] = contents
                result = ()
            elif isinstance(op, Call):
                f = declarations[op.function]
                data = tuple(copy_data(v) for v in ins)
                resource_data = tuple(copy_data(v) for r in f.resources if r in state for v in state[r])
                returned = implementations[f.name]((*data, *resource_data))
                if len(returned) != len(f.outputs) + len(resource_data):
                    raise ValueError("reference function result arity mismatch")
                data_outputs, changed = returned[:len(f.outputs)], returned[len(f.outputs):]
                for value, type_ in zip(data_outputs, f.outputs, strict=True):
                    check_data(value, type_, sizes)
                replacements: dict[str, tuple[Data, ...]] = {}
                offset = 0
                for name in f.resources:
                    if name not in storage_types:
                        continue
                    types = storage_types[name]
                    contents = changed[offset:offset + len(types)]
                    offset += len(types)
                    for value, type_ in zip(contents, types, strict=True):
                        check_data(value, type_, sizes)
                    replacements[name] = tuple(copy_data(v) for v in contents)
                for name in f.resources:
                    advance(name, f.name, coordinates)
                state.update(replacements)
                result = tuple(copy_data(v) for v in data_outputs)
                calls.append((f.name, coordinates))
            elif isinstance(op, Scan):
                width = len(ins) - op.invariants
                carry, fixed = ins[:width], ins[width:]
                collected: list[list[Data]] = [[] for _ in op.outputs[width:]]
                for i in range(resolve_extent(op.count, sizes)):
                    row = run(op.body, (i, *carry, *fixed), (*coordinates, i))
                    carry = row[:width]
                    for column, value in zip(collected, row[width:], strict=True):
                        column.append(value)
                result = (*carry, *(Bank(tuple(column)) for column in collected))
            elif isinstance(op, Map):
                columns: list[list[Data]] = [[] for _ in op.outputs]
                for i in range(resolve_extent(op.count, sizes)):
                    row = run(op.body, (i, *ins), (*coordinates, i))
                    for column, value in zip(columns, row, strict=True):
                        column.append(value)
                result = tuple(Bank(tuple(column)) for column in columns)
            elif isinstance(op, If):
                branch = op.then_region if ins[0] else op.else_region
                result = run(branch, ins[1:], coordinates)
            else:  # The verified closed union leaves only While.
                result = ins
                for i in range(op.limit):
                    condition = run(op.condition, result, (*coordinates, i))
                    if not condition[0]:
                        break
                    result = run(op.body, result, (*coordinates, i))
            for output, value in zip(op.outputs, result, strict=True):
                check_data(value, output.type, sizes)
                env[output.name] = value
        return tuple(env[name] for name in block.yields)

    entry_args = tuple(copy_data(inputs[v.name]) for v in arguments)
    with torch.no_grad():
        returned = run(program.entry, entry_args, ())
    return ReferenceResult(tuple(copy_data(v) for v in returned),
                           state, tuple(effects), tuple(calls))
