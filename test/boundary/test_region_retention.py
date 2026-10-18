"""Backend retention bounds follow the actual lowered order across sizes."""

from dataclasses import replace

import pytest

from pllm.contracts.region_program import (
    AccessMode,
    BorrowedResource,
    Call,
    Function,
    Program,
    ResourceAccess,
    ScalarType,
    Scan,
    SizeParameter,
    Value,
)
from pllm.contracts.region_retention import RetainedAllocation, RetentionEndpoint
from pllm.execution.decoder_region import index_cell
from pllm.execution.region_compilation import (
    analyze_region_program,
    analyze_region_retention,
    lower_region_schedule,
    validate_region_retention,
)
from pllm.execution.region_schedule import RegionScheduleExecution
from test.boundary.test_region_program import region


def source():
    index = ScalarType.INDEX
    functions = tuple(
        Function(name, (index,), (), ("context",), (ResourceAccess("context", mode, (index_cell(0),)),))
        for name, mode in (("begin", AccessMode.WRITE), ("complete", AccessMode.READ))
    )
    body = region(
        (Value("i", index),),
        (
            Call(("i",), (), "begin"),
            Call(("i",), (), "complete"),
        ),
        (),
    )
    return Program(
        region((), (Scan((), (), "n", body),), ()),
        functions,
        (BorrowedResource("context", (1_000_000,)),),
        (SizeParameter("n", 0, 1_000_000),),
    )


IMPORTS = (RetainedAllocation("backup", RetentionEndpoint("begin", 0), RetentionEndpoint("complete", 0), "cpu", 128),)


def test_serial_and_hoisted_acquisitions_have_different_capacity():
    program = source()
    serial = analyze_region_program(program, optimize=False).reference_schedule
    guard = "0 <= __size_n <= 1000000 and 0 <= i < __size_n"
    hoisted = f"[__size_n] -> {{ S0[i] -> [0,i] : {guard}; S1[i] -> [1,i] : {guard} }}"
    for schedule, expected in ((serial, 1), (hoisted, 1_000_000)):
        bound = analyze_region_retention(program, schedule, IMPORTS)
        (allocation,) = bound.allocations
        assert allocation.peak_instances_upper_bound == expected
        assert allocation.peak_bytes_upper_bound == expected * 128
        validate_region_retention(program, IMPORTS, bound)
        cursor = RegionScheduleExecution(program, lower_region_schedule(program, schedule))
        for n in (0, 1, 7, 31):
            live, peak = {}, 0
            for action in cursor.invocations({"n": n}):
                (identity,) = action.coordinates
                if action.statement_id == 0:
                    live[identity] = bytearray(128)
                    peak = max(peak, sum(len(value) for value in live.values()))
                else:
                    del live[identity]
            assert not live
            assert peak == (min(n, 1) if schedule == serial else n) * 128
            assert peak <= allocation.peak_bytes_upper_bound
        with pytest.raises(ValueError, match="retention bound"):
            validate_region_retention(
                program, IMPORTS, replace(bound, allocations=(replace(allocation, peak_bytes_upper_bound=0),))
            )


def test_retention_rejects_unmatched_or_unresolved_endpoint_identities():
    program = source()
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    for endpoint in (RetentionEndpoint("missing", 0), RetentionEndpoint("complete", 1), RetentionEndpoint("begin", 0)):
        with pytest.raises(ValueError):
            analyze_region_retention(program, schedule, (replace(IMPORTS[0], release=endpoint),))
    # A source index passed at entry has no affine identity for the planner.
    index = ScalarType.INDEX
    unknown = replace(
        program,
        entry=region(
            (Value("i", index),),
            (
                Call(("i",), (), "begin"),
                Call(("i",), (), "complete"),
            ),
            (),
        ),
        parameters=(),
    )
    with pytest.raises(ValueError, match="identities"):
        analyze_region_retention(unknown, analyze_region_program(unknown, optimize=False).reference_schedule, IMPORTS)


def test_retention_endpoints_keep_host_ownership_even_with_native_kernel_imports():
    from pllm.contracts.region_imports import NativeKernelImport
    from pllm.execution.region_native import external_primitive
    from pllm.execution.region_streams import lower_region_streams

    program = source()
    cells = {
        f.name: external_primitive(f, NativeKernelImport(f, "test.current_stream@1", "cpu", ()))
        for f in program.functions
    }
    guard = "0 <= __size_n <= 1000000 and 0 <= i < __size_n"
    schedule = f"[__size_n] -> {{ S0[i] -> [0,i] : {guard}; S1[i] -> [1,i] : {guard} }}"
    ordinary = lower_region_streams(program, schedule, cells, axis=1, lanes=2, pipeline_depth=2)
    assert not ordinary.host_fences
    # A kernel contract does not move the original retention owner to another
    # lane. With both calls now being lifetime endpoints, no parallel work remains.
    with pytest.raises(ValueError, match="no independent work"):
        lower_region_streams(program, schedule, cells, axis=1, lanes=2, pipeline_depth=2, retained=IMPORTS)
