"""Mechanical execution of native-lowered structural actions and their leaf ABI."""

from collections.abc import Mapping

import torch

from pllm.contracts.control_instructions import (
    ControlAction, ControlBlock, ControlIf, ControlPrimitive, ControlScan, ControlStep, ControlWhile,
)
from pllm.contracts.region_program import Call, Constant, Coordinate, Op, Parameter, Program, Publish, Resource, Scalar, ScalarKind, Select, Snapshot, Update, resolve_extent
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_native import RegionFunction
from pllm.execution.region_resources import RegionResources
from pllm.execution.region_values import Bank, Data, check_data


class RegionActions:
    """Stateless action binding shared by affine and dynamic parts of one plan."""

    def __init__(self, program: Program, cells: Mapping[str, RegionFunction]) -> None:
        self._cells = dict(cells)
        self._payload_sizes = {r.name: len(r.payload) for r in program.resources if isinstance(r, Resource)}

    def execute(self, action: ControlAction, inputs: tuple[Data, ...],
                pages: RegionResources | None, sizes: Mapping[str, int] = {}) -> tuple[Data, ...]:
        if pages is not None:
            pages.before_statement()
        if isinstance(action, ControlPrimitive):
            return self._primitive(action.operation, inputs, pages, sizes)
        if isinstance(action, ControlIf):
            predicate = inputs[0]
            if type(predicate) is not bool:
                raise ValueError("region branch predicate must be boolean")
            return self._block(action.then_block if predicate else action.else_block, inputs[1:], pages, sizes)
        if isinstance(action, ControlWhile):
            carry = inputs
            for _ in range(action.limit):
                predicate, = self._block(action.condition, carry, pages, sizes)
                if type(predicate) is not bool:
                    raise ValueError("region loop predicate must be boolean")
                if not predicate:
                    break
                carry = self._block(action.body, carry, pages, sizes)
            return carry
        if isinstance(action, ControlScan):
            width = len(inputs) - action.invariants
            carry, fixed = inputs[:width], inputs[width:]
            collected: list[list[Data]] = [[] for _ in action.body.results[width:]]
            for index in range(resolve_extent(action.count, sizes)):
                row = self._block(action.body, (index, *carry, *fixed), pages, sizes)
                carry = row[:width]
                for column, value in zip(collected, row[width:], strict=True):
                    column.append(value)
            return (*carry, *(Bank(tuple(column)) for column in collected))
        # The closed action union leaves only Map.
        columns: list[list[Data]] = [[] for _ in action.body.results]
        for index in range(resolve_extent(action.count, sizes)):
            row = self._block(action.body, (index, *inputs), pages, sizes)
            for column, value in zip(columns, row, strict=True):
                column.append(value)
        return tuple(Bank(tuple(column)) for column in columns)

    def _block(self, block: ControlBlock, arguments: tuple[Data, ...],
               pages: RegionResources | None, sizes: Mapping[str, int] = {}) -> tuple[Data, ...]:
        if pages is not None:
            pages.before_statement()
        values: dict[int, Data] = {}
        for slot, value in zip(block.arguments, arguments, strict=True):
            check_data(value, slot.type, sizes)
            if slot.id not in block.discard_arguments:
                values[slot.id] = value
        try:
            for step in block.steps:
                self._step(step, values, pages, sizes)
            return tuple(values[i] for i in block.results)
        finally:
            values.clear()

    def _step(self, step: ControlStep, values: dict[int, Data], pages: RegionResources | None,
              sizes: Mapping[str, int]) -> None:
        # Match the affine cursor's per-invocation scope: released slots must
        # not remain alive in this loop's previous result or value locals.
        inputs = tuple(values[i] for i in step.inputs)
        result = self.execute(step.action, inputs, pages, sizes)
        for slot, value in zip(step.outputs, result, strict=True):
            check_data(value, slot.type, sizes)
            values[slot.id] = value
        for slot_id in step.releases:
            del values[slot_id]

    def _primitive(self, op: Op, ins: tuple[Data, ...],
                   pages: RegionResources | None, sizes: Mapping[str, int] = {}) -> tuple[Data, ...]:
        if isinstance(op, Constant):
            return (op.value,)
        if isinstance(op, Parameter):
            return (sizes[op.parameter],)
        if isinstance(op, Coordinate):
            value = op.affine.constant
            for term in op.affine.terms:
                operand = ins[term.argument]
                if type(operand) is not int:
                    raise ValueError("coordinate operands must be integers")
                value += term.coefficient * operand
            return (value,)
        if isinstance(op, Call):
            function = self._cells[op.function]
            if not isinstance(function, ExportedCell):
                return function.invoke(ins, pages, sizes)
            tensors: list[torch.Tensor] = []
            for value in ins:
                if not isinstance(value, torch.Tensor):
                    raise ValueError("exported cell received a non-tensor operand")
                tensors.append(value)
            return function.invoke(tuple(tensors))
        if isinstance(op, (Snapshot, Publish)):
            if pages is None:
                raise RuntimeError("resource statement has no PTTS binding")
            width = self._payload_sizes[op.resource]
            index_values = ins if isinstance(op, Snapshot) else ins[width:]
            indices: list[int] = []
            for index in index_values:
                if type(index) is not int:
                    raise ValueError("resource indices must be integers")
                indices.append(index)
            if isinstance(op, Snapshot):
                return pages.snapshot(op.resource, tuple(indices))
            pages.publish(op.resource, tuple(indices), ins[:width])
            return ()
        if isinstance(op, Scalar):
            left, right = ins
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("scalar statement received a non-scalar operand")
            if op.kind is ScalarKind.ADD:
                return (left + right,)
            if op.kind is ScalarKind.SUB:
                return (left - right,)
            if op.kind is ScalarKind.MUL:
                return (left * right,)
            if op.kind is ScalarKind.LESS:
                return (left < right,)
            return (left == right,)
        if isinstance(op, (Select, Update)):
            bank, index = ins[:2]
            if not isinstance(bank, Bank) or type(index) is not int or not 0 <= index < len(bank.elements):
                raise ValueError("logical bank index is out of bounds")
            if isinstance(op, Select):
                return (bank.elements[index],)
            return (Bank((*bank.elements[:index], ins[2], *bank.elements[index + 1:])),)
        raise RuntimeError("nested region cannot execute as a primitive instruction")
