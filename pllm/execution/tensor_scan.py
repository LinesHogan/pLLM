"""Lower an imported tensor recurrence to a structured region scan.

The cell receives carry, invariants, then one row from each sequence. Its first
results replace carry; remaining results are collected in iteration order.
This builds the reference algorithm, which can later be composed into a larger
region before global scheduling. It owns no state pages or publication rights.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass

import torch

from pllm.contracts.region_program import (
    BankType,
    Block,
    Call,
    Program,
    Region,
    ScalarType,
    Scan,
    Select,
    SizeParameter,
    TensorType,
    Value,
    verify,
)
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelectionPolicy
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution, RegionInstructionExecution
from pllm.execution.region_native import RegionFunction
from pllm.execution.region_optimization import (
    RegionOptimizationOptions,
    SelectedRegion,
    optimize_region,
    prepare_region_execution,
)
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.region_values import Bank, Data


@dataclass(frozen=True, slots=True)
class TensorScanResult:
    carry: tuple[torch.Tensor, ...]
    columns: tuple[tuple[torch.Tensor, ...], ...]


class TensorScan:
    def __init__(
        self,
        cell: RegionFunction | Program,
        count: int | SizeParameter,
        *,
        carry: int,
        invariants: int,
        functions: Mapping[str, RegionFunction] | None = None,
    ) -> None:
        """Keep a tensor region's operations inside the loop, visible to ISL.

        A single imported function remains shorthand for a one-call region.
        A Program body uses its own lexical names; only its tensor ports are
        lifted to carry, invariant and sequence arguments. No nested executor
        or callable wrapping the entire body is introduced.
        """
        if isinstance(cell, Program):
            source = cell
            if functions is None:
                raise ValueError("tensor scan region requires its imported functions")
            cells = dict(functions)
        else:
            if functions is not None:
                raise ValueError("a single tensor import supplies its own function binding")
            signature = cell.declaration
            args = tuple(Value(f"arg{i}", type_) for i, type_ in enumerate(signature.inputs))
            results = tuple(Value(f"result{i}", type_) for i, type_ in enumerate(signature.outputs))
            source = Program(
                Region(
                    Block(
                        args,
                        (Call(tuple(v.name for v in args), results, signature.name),),
                        tuple(v.name for v in results),
                    )
                ),
                (signature,),
            )
            cells = {signature.name: cell}
        output_types = verify(source)
        block = source.entry.block
        input_types = tuple(v.type for v in block.arguments)
        if (
            source.resources
            or source.parameters
            or any(not isinstance(type_, TensorType) for type_ in (*input_types, *output_types))
        ):
            raise ValueError("tensor scan requires a pure tensor region with static body ports")
        if set(cells) != {f.name for f in source.functions} or any(
            cells[f.name].declaration != f for f in source.functions
        ):
            raise ValueError("tensor scan imports differ from its source region")
        if (
            not isinstance(count, SizeParameter)
            and (type(count) is not int or count < 0)
            or type(carry) is not int
            or carry < 0
            or type(invariants) is not int
            or invariants < 0
            or carry + invariants > len(input_types)
            or output_types[:carry] != input_types[:carry]
        ):
            raise ValueError("tensor scan requires a nonnegative extent and matching carry types")
        fixed = carry + invariants
        extent = count.name if isinstance(count, SizeParameter) else count
        tensors = tuple(type_ for type_ in input_types if isinstance(type_, TensorType))
        result_tensors = tuple(type_ for type_ in output_types if isinstance(type_, TensorType))
        arguments = tuple(
            Value(f"arg{i}", type_ if i < fixed else BankType(type_, extent)) for i, type_ in enumerate(tensors)
        )
        names = {v.name for v in block.arguments} | {v.name for op in block.operations for v in op.outputs}

        def fresh(prefix: str) -> str:
            name = prefix
            while name in names:
                name += "_"
            names.add(name)
            return name

        index = fresh("scan_index")
        body_arguments = block.arguments[:fixed] + tuple(
            Value(fresh(f"scan_sequence{i}"), BankType(type_, extent)) for i, type_ in enumerate(tensors[fixed:])
        )
        body = Region(
            Block(
                (Value(index, ScalarType.INDEX), *body_arguments),
                (
                    *(
                        Select((bank.name, index), (row,))
                        for bank, row in zip(body_arguments[fixed:], block.arguments[fixed:], strict=True)
                    ),
                    *block.operations,
                ),
                block.yields,
            )
        )
        outputs = tuple(Value(f"final{i}", v.type) for i, v in enumerate(arguments[:carry])) + tuple(
            Value(f"column{i}", BankType(type_, extent)) for i, type_ in enumerate(result_tensors[carry:])
        )
        program = Program(
            Region(
                Block(
                    arguments,
                    (Scan(tuple(v.name for v in arguments), outputs, extent, body, invariants=len(arguments) - carry),),
                    tuple(v.name for v in outputs),
                )
            ),
            source.functions,
            parameters=(count,) if isinstance(count, SizeParameter) else (),
            constraints=source.constraints,
        )
        analysis = analyze_region_program(program, optimize=False)
        schedule = analysis.reference_schedule
        self.program = program
        self._cells = cells
        self._execution: RegionInstructionExecution = RegionExecution(
            program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule), cells
        )
        self._carry, self._invariants = carry, invariants
        self._sequences = len(input_types) - fixed
        self._names = tuple(v.name for v in arguments)
        self.count = count

    def _arguments(
        self,
        carry: tuple[torch.Tensor, ...],
        invariants: tuple[torch.Tensor, ...],
        sequences: tuple[tuple[torch.Tensor, ...], ...],
        count: int | None,
    ) -> tuple[Mapping[str, Data], dict[str, int]]:
        extent = (
            (count if count is not None else len(sequences[0]) if sequences else None)
            if isinstance(self.count, SizeParameter)
            else self.count
        )
        if (
            len(carry) != self._carry
            or len(invariants) != self._invariants
            or type(extent) is not int
            or count is not None
            and (type(count) is not int or count != extent)
            or len(sequences) != self._sequences
            or any(len(s) != extent for s in sequences)
        ):
            raise ValueError("tensor scan operands differ from its source signature")
        sizes = {self.count.name: extent} if isinstance(self.count, SizeParameter) else {}
        inputs: Mapping[str, Data] = dict(
            zip(self._names, (*carry, *invariants, *(Bank(s) for s in sequences)), strict=True)
        )
        return inputs, sizes

    def optimized(
        self,
        carry: tuple[torch.Tensor, ...],
        invariants: tuple[torch.Tensor, ...],
        sequences: tuple[tuple[torch.Tensor, ...], ...],
        *,
        payload_limits: tuple[RegionPayloadLimit, ...],
        count: int | None = None,
        options: RegionOptimizationOptions = RegionSelectionPolicy(tuning="measured"),
        warmup_rounds: int = 1,
        measurement_rounds: int = 3,
        max_candidates: int = 8,
        max_rewrite_depth: int = 2,
        streams: RegionStreamBindings | None = None,
    ) -> tuple["TensorScan", SelectedRegion]:
        """Return a cold-tuned scan with the same callable and source contract.

        Examples are privately copied. One selected symbolic plan serves every
        declared count; timings describe only the example count. The original
        scan stays available, and calling either scan does not tune or compile.
        Limits cover tensor payloads, not the complete device/owner budget.
        """
        sources: dict[str, ExportedCell] = {}
        for name, cell in self._cells.items():
            if not isinstance(cell, ExportedCell):
                raise ValueError("tensor scan tuning currently requires original exported cells")
            sources[name] = cell
        inputs, sizes = self._arguments(carry, invariants, sequences, count)
        execution = optimize_region(
            self.program,
            sources,
            inputs,
            sizes=sizes,
            payload_limits=payload_limits,
            options=options,
            warmup_rounds=warmup_rounds,
            measurement_rounds=measurement_rounds,
            max_candidates=max_candidates,
            max_rewrite_depth=max_rewrite_depth,
            streams=streams,
        )
        result = copy(self)
        result._execution = execution.execution
        return result, execution

    def prepared(
        self,
        *,
        payload_limits: tuple[RegionPayloadLimit, ...],
        options: RegionOptimizationOptions,
        parallel: bool = False,
    ) -> "TensorScan":
        """Static preparation needs no example invocation or measurement."""
        result = copy(self)
        result._execution = prepare_region_execution(
            self.program, self._cells, payload_limits=payload_limits, options=options, parallel=parallel
        )
        return result

    @property
    def parallel(self) -> bool:
        return self._execution.stream_plan is not None

    def __call__(
        self,
        carry: tuple[torch.Tensor, ...],
        invariants: tuple[torch.Tensor, ...],
        sequences: tuple[tuple[torch.Tensor, ...], ...],
        *,
        count: int | None = None,
        streams: RegionStreamBindings | None = None,
    ) -> TensorScanResult:
        inputs, sizes = self._arguments(carry, invariants, sequences, count)
        if streams is not None:
            streams.validate_available()
        with nullcontext() if streams is None else torch.cuda.stream(streams.owner):
            result = self._execution.run(inputs, sizes=sizes, streams=streams if self.parallel else None).values
        final: list[torch.Tensor] = []
        columns: list[tuple[torch.Tensor, ...]] = []
        for value in result[: self._carry]:
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("lowered tensor carry is not a tensor")
            final.append(value)
        for value in result[self._carry :]:
            if not isinstance(value, Bank):
                raise RuntimeError("lowered tensor collection is not a bank")
            column: list[torch.Tensor] = []
            for item in value.elements:
                if not isinstance(item, torch.Tensor):
                    raise RuntimeError("lowered tensor collection contains a non-tensor")
                column.append(item)
            columns.append(tuple(column))
        return TensorScanResult(tuple(final), tuple(columns))
