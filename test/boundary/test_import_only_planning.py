"""A separate planner process consumes closed region imports without source graphs, weights or CUDA.

The parent captures a CUDA region on fake tensors and ships only the program and its
imports; the child's payload and stream plans must equal the parent's.
"""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.region_payload import RegionPayload
from pllm.contracts.region_program import AffineIndex, Block, BorrowedResource, Coordinate, Region
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_streams import RegionStreamExploration
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.exported_resources import borrow_exported_inputs
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_execution import RegionExecutionPlan
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_streams import explore_region_streams


@pytest.mark.slow
def test_import_only_planner_roundtrip_matches_local_graphs_without_initializing_cuda():
    def computation(x, a, b):
        return (x + a + b,)

    with FakeTensorMode():
        tensors = tuple(torch.empty((4, 4), device="cuda:0") for _ in range(3))
        source = ExportedCell.capture(computation, ("x", "a", "b"), ("out",), tensors)
    body, functions = borrow_exported_inputs(source, BorrowedResource("original", (1,)), {"a": 0, "b": 1})
    block = body.program.entry.block
    value, coordinate = block.arguments
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
    raise AssertionError("planning must not allocate tensors or initialize CUDA")
with patch.object(torch, 'empty', forbidden):
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    payload = analyze_region_payload(program, schedule, imports)
    streams = explore_region_streams(program, imports, schedule=analyze_region_program(program, parallel_only=True).candidates[-1].schedule_map)
    proposal = RegionScheduleProposal(payload)
    plan = RegionExecutionPlan(program, imports, proposal, proposal.peak_bytes)
    assert plan.stream_plan is None
assert not torch.cuda.is_initialized()
from pllm.execution.memory import measure
from pllm import scan
from pllm.execution.memory import apply_adamw_or_reserve
from pllm.kernels.adamw import AdamWConfig, AdamWFields
with patch.object(torch.cuda, '_lazy_init', forbidden):
    roots = ((torch.empty(32, device='meta'),) * 2 + (torch.empty((), device='meta'),) + (torch.empty(32, device='meta'),) * 2,)
    measure(lambda: apply_adamw_or_reserve(roots, AdamWFields(0, 1, 2, 3, 4), AdamWConfig(.01), (torch.empty(32, device='meta'),)))
    scan(lambda carry, x: (carry + x.sum(0), x + carry), carry=torch.empty(3, device="meta"), over=(torch.empty(4, 3, device="meta"),), chunk=2)
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
