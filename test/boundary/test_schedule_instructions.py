"""The emitted loop tree must implement the fixed schedule, without expansion."""

from dataclasses import replace
from itertools import islice

import pytest

from pllm.contracts.region_program import BankType, Call, Function, Map, Program, ScalarType, Value
from pllm.contracts.schedule_instructions import LoweredSchedule, ScheduleLoop, ScheduleSequence
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_compilation import analyze_region_program, lower_region_schedule, preview_region_schedule
from pllm.execution.region_schedule import RegionScheduleExecution
from test.boundary.test_region_program import grid_program, region


@pytest.mark.parametrize("chunks, layers", [(0, 3), (3, 0), (1, 1), (2, 7), (5, 3), (6, 5)])
def test_loop_instructions_match_independent_isl_point_evaluation(chunks, layers):
    program = grid_program(chunks, layers)
    for candidate in analyze_region_program(program).candidates:
        lowered = lower_region_schedule(program, candidate.schedule_map)
        assert decode_boundary(encode_boundary(lowered), LoweredSchedule) == lowered
        actual = tuple(RegionScheduleExecution(program, lowered).invocations())
        expected = preview_region_schedule(program, candidate.schedule_map)
        assert [(x.statement_id, x.coordinates) for x in actual] == [(x.statement_id, x.coordinates) for x in expected]


def test_tiled_schedule_handles_partial_tiles_with_exact_coverage():
    program = grid_program(5, 7)
    schedule = "{ S0[c,l] -> [floor(c/2),floor(l/3),c,l] : 0<=c<5 and 0<=l<7 }"
    lowered = lower_region_schedule(program, schedule)
    actual = tuple(RegionScheduleExecution(program, lowered).invocations())
    expected = preview_region_schedule(program, schedule)
    assert [x.coordinates for x in actual] == [x.coordinates for x in expected]
    assert len({x.coordinates for x in actual}) == 35


def test_negative_time_and_nonunit_steps_preserve_integer_division():
    index = ScalarType.INDEX
    body = region((Value("i", index),), (Call(("i",), (Value("y", index),), "f"),), ("y",))
    program = Program(region((), (Map((), (Value("ys", BankType(index, 6)),), 6, body),), ("ys",)),
                      (Function("f", (index,), (index,)),))
    schedule = "{ S0[i] -> [-2*i] : 0<=i<6 }"
    lowered = lower_region_schedule(program, schedule)
    actual = tuple(RegionScheduleExecution(program, lowered).invocations())
    assert [x.coordinates for x in actual] == [(i,) for i in reversed(range(6))]


def test_large_lowering_and_cursor_are_lazy_and_repeatable():
    program = grid_program(1_000_000, 16)
    candidate = next(c for c in analyze_region_program(program).candidates if c.source == "isl-parallel")
    lowered = lower_region_schedule(program, candidate.schedule_map)
    assert isinstance(lowered.body, ScheduleLoop)
    assert len(encode_boundary(lowered)) < 5000
    execution = RegionScheduleExecution(program, lowered)
    first = tuple(islice(execution.invocations(), 6))
    assert [x.coordinates for x in first] == [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
    assert tuple(islice(execution.invocations(), 6)) == first


def test_modified_instructions_are_rejected_before_opening_a_cursor():
    program = grid_program(3, 3)
    analysis = analyze_region_program(program)
    lowered = lower_region_schedule(program, analysis.candidates[-1].schedule_map)
    for changed in (
        replace(lowered, body=ScheduleSequence(())),
        replace(lowered, statements=()),
        replace(lowered, program_digest="a" * 64),
        replace(lowered, signature="b" * 64),
        replace(lowered, version=lowered.version + 1),
    ):
        with pytest.raises(ValueError, match="original program"):
            RegionScheduleExecution(program, changed)
