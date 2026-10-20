"""Cold traversal of native instructions for the shared executor's instance tables.

This cursor only issues statement identities and coordinates during preparation.
It neither owns state nor chooses placement, transforms or stream policy.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping

from pllm.contracts.region_program import Program, bind_sizes, resolve_extent
from pllm.contracts.schedule_instructions import (
    LoweredSchedule,
    ScheduleExpression,
    ScheduleInstruction,
    ScheduleInteger,
    ScheduleInvoke,
    ScheduleIterator,
    ScheduleLoop,
    ScheduleOperator,
    ScheduleSequence,
    StatementInvocation,
)

_ARITHMETIC = {
    ScheduleOperator.ADD: operator.add,
    ScheduleOperator.SUB: operator.sub,
    ScheduleOperator.MUL: operator.mul,
}
_COMPARISONS = {
    ScheduleOperator.EQ: operator.eq,
    ScheduleOperator.LE: operator.le,
    ScheduleOperator.LT: operator.lt,
    ScheduleOperator.GE: operator.ge,
    ScheduleOperator.GT: operator.gt,
}


def checked_schedule_integer(value: int | bool) -> int:
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("schedule integer exceeds its signed 64-bit contract")
    return value


def evaluate_schedule_expression(expression: ScheduleExpression, variables: dict[str, int]) -> int | bool:
    if isinstance(expression, ScheduleInteger):
        return checked_schedule_integer(expression.value)
    if isinstance(expression, ScheduleIterator):
        return variables[expression.name]
    op, args = expression.operator, expression.arguments
    if op in (ScheduleOperator.AND, ScheduleOperator.AND_THEN):
        return bool(evaluate_schedule_expression(args[0], variables)) and bool(
            evaluate_schedule_expression(args[1], variables)
        )
    if op in (ScheduleOperator.OR, ScheduleOperator.OR_ELSE):
        return bool(evaluate_schedule_expression(args[0], variables)) or bool(
            evaluate_schedule_expression(args[1], variables)
        )
    if op in (ScheduleOperator.CONDITIONAL, ScheduleOperator.SELECT):
        return evaluate_schedule_expression(
            args[1] if evaluate_schedule_expression(args[0], variables) else args[2], variables
        )
    numbers = tuple(checked_schedule_integer(evaluate_schedule_expression(a, variables)) for a in args)
    if op is ScheduleOperator.MIN:
        return min(numbers)
    if op is ScheduleOperator.MAX:
        return max(numbers)
    if op is ScheduleOperator.NEG:
        return checked_schedule_integer(-numbers[0])
    left, right = numbers
    if op in _ARITHMETIC:
        return checked_schedule_integer(_ARITHMETIC[op](left, right))
    if op in _COMPARISONS:
        return _COMPARISONS[op](left, right)
    if op is ScheduleOperator.EXACT_DIV:
        if right == 0 or left % right:
            raise ValueError("schedule exact division precondition failed")
        return checked_schedule_integer(left // right)
    if op is ScheduleOperator.FLOOR_DIV:
        if right <= 0:
            raise ValueError("schedule floor division requires a positive divisor")
        return checked_schedule_integer(left // right)
    if op in (ScheduleOperator.POSITIVE_DIV, ScheduleOperator.POSITIVE_MOD):
        if left < 0 or right <= 0:
            raise ValueError("schedule positive division precondition failed")
        return left // right if op is ScheduleOperator.POSITIVE_DIV else left % right
    if op is ScheduleOperator.ZERO_MOD:
        if right <= 0:
            raise ValueError("schedule modulo requires a positive divisor")
        quotient = abs(left) // abs(right)
        if (left < 0) != (right < 0):
            quotient = -quotient
        return checked_schedule_integer(left - quotient * right)
    raise ValueError("unknown schedule expression operator")


def _walk(instruction: ScheduleInstruction, variables: dict[str, int]) -> Iterator[StatementInvocation]:
    if isinstance(instruction, ScheduleInvoke):
        yield StatementInvocation(
            instruction.statement_id,
            tuple(
                checked_schedule_integer(evaluate_schedule_expression(c, variables)) for c in instruction.coordinates
            ),
        )
    elif isinstance(instruction, ScheduleSequence):
        for child in instruction.children:
            yield from _walk(child, variables)
    elif isinstance(instruction, ScheduleLoop):
        previous = variables.get(instruction.iterator)
        variables[instruction.iterator] = checked_schedule_integer(
            evaluate_schedule_expression(instruction.initial, variables)
        )
        try:
            while evaluate_schedule_expression(instruction.condition, variables):
                yield from _walk(instruction.body, variables)
                step = checked_schedule_integer(evaluate_schedule_expression(instruction.step, variables))
                if step <= 0:
                    raise ValueError("lowered schedule loop must advance")
                variables[instruction.iterator] = checked_schedule_integer(variables[instruction.iterator] + step)
        finally:
            if previous is None:
                del variables[instruction.iterator]
            else:
                variables[instruction.iterator] = previous
    else:
        branch = (
            instruction.if_true
            if evaluate_schedule_expression(instruction.condition, variables)
            else instruction.if_false
        )
        yield from _walk(branch, variables)


class RegionScheduleExecution:
    """Bind the source-checked instructions once, then open independent cursors."""

    def __init__(self, program: Program, lowered: LoweredSchedule) -> None:
        self._lowered = lowered
        self._program = program

    def invocations(self, sizes: Mapping[str, int] = {}) -> Iterator[StatementInvocation]:
        sizes = bind_sizes(self._program, sizes)
        for invocation in _walk(self._lowered.body, {f"__size_{name}": value for name, value in sizes.items()}):
            statement = self._lowered.statements[invocation.statement_id]
            if any(
                not 0 <= coordinate < resolve_extent(extent, sizes)
                for coordinate, extent in zip(invocation.coordinates, statement.extents, strict=True)
            ):
                raise RuntimeError("lowered invocation is outside its verified source domain")
            yield invocation
