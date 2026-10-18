"""A separate planner consumes closed imports without source graphs or weights."""

import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.region_imports import BorrowedTensorImport
from pllm.contracts.region_payload import RegionPayload
from pllm.contracts.region_program import (
    AffineIndex,
    Block,
    BorrowedResource,
    Coordinate,
    Region,
    ScalarType,
    TensorType,
)
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_streams import RegionStreamExploration
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.exported_resources import borrow_exported_inputs
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_execution import RegionExecutionPlan
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_streams import explore_region_streams


def test_import_only_planner_roundtrip_matches_local_graphs_without_initializing_cuda():
    def computation(x, a, b):
        return (x + a + b,)

    with FakeTensorMode():
        tensors = tuple(torch.empty((4, 4), device="cuda:0") for _ in range(3))
        source = ExportedCell.capture(computation, ("x", "a", "b"), ("out",), tensors)
    body, functions = borrow_exported_inputs(source, BorrowedResource("original", (1,)), {"a": 0, "b": 1})
    block = body.program.entry.block
    value, coordinate = block.arguments
    assert coordinate.type is ScalarType.INDEX
    program = replace(
        body.program,
        entry=Region(Block((value,), (Coordinate((), (coordinate,), AffineIndex(0)), *block.operations), block.yields)),
    )
    imports = import_region_functions(program, functions)
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    expected = analyze_region_payload(program, schedule, functions)
    expected_streams = explore_region_streams(
        program, functions, schedule=analyze_region_program(program, parallel_only=True).candidates[-1].schedule_map
    )
    assert expected_streams.candidates
    script = """
import sys
from unittest.mock import patch
import torch
from pllm.contracts.region_imports import RegionTensorImport
from pllm.contracts.region_program import Program
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_storage import analyze_region_payload
from pllm.execution.region_streams import explore_region_streams
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.execution.region_execution import RegionExecutionPlan
program, imports = decode_boundary(sys.stdin.read(), tuple[Program, tuple[RegionTensorImport, ...]])
assert not torch.cuda.is_initialized()
def forbidden(*args, **kwargs):
    raise AssertionError("planning must not reconstruct a tensor graph or allocate tensors")
with patch('pllm.execution.region_storage.import_tensor_access_graph', forbidden), patch.object(torch, 'empty', forbidden):
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    payload = analyze_region_payload(program, schedule, imports)
    streams = explore_region_streams(program, imports, schedule=analyze_region_program(program, parallel_only=True).candidates[-1].schedule_map)
    proposal = RegionScheduleProposal(payload)
    plan = RegionExecutionPlan(program, imports, proposal, proposal.peak_bytes)
    assert plan.stream_plan is None
assert not torch.cuda.is_initialized()
print(encode_boundary((payload, streams)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=encode_boundary((program, imports)),
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    payload, streams = decode_boundary(completed.stdout, tuple[RegionPayload, RegionStreamExploration])
    assert payload == expected and streams == expected_streams
    parallel = RegionScheduleProposal(streams.candidates[0].payload, streams.candidates[0])
    plan = RegionExecutionPlan(program, imports, parallel, parallel.peak_bytes)
    execution = plan.bind(functions)
    assert execution.stream_plan == streams.candidates[0]
    with pytest.raises(ValueError, match="payload reservation"):
        replace(plan, payload_limits=())
    with patch("pllm.execution.region_storage.import_tensor_access_graph", side_effect=AssertionError("no reimport")):
        assert analyze_region_payload(program, schedule, imports) == expected
    # Descriptor-only input still passes through native ABI validation.
    borrowed = next(item for item in imports if isinstance(item, BorrowedTensorImport))
    forged = replace(
        borrowed, declaration=replace(borrowed.declaration, outputs=(TensorType((999,), "torch.float32", "cuda:0"),))
    )
    with pytest.raises(ValueError):
        analyze_region_payload(program, schedule, tuple(forged if item == borrowed else item for item in imports))
    with pytest.raises(ValueError, match="complete distinct"):
        analyze_region_payload(program, schedule, (*imports, imports[0]))
