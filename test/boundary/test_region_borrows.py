"""Borrowed external storage requires physical evidence from the original invocation."""

from dataclasses import replace

import pytest
import torch

from pllm.contracts.region_program import (
    AccessMode, AffineIndex, BorrowedResource, Call, Coordinate, Function, Program, ResourceAccess, ScalarType, TensorType, Value,
)
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.contracts.region_selection import RegionCandidate, RegionPayloadLimit, RegionSelectionRequest
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.decoder_region import index_cell
from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule, validate_region_program
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import external_primitive
from pllm.execution.region_resources import ExternalFunction, PagedResourceBinding, ReadonlyResourceBinding, RegionResources
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.region_optimization import select_region_candidate, PreparedRegionCandidate, bind_region_candidate, assess_region_candidates
from pllm.execution.region_storage import import_region_functions
from test.boundary.test_paged_access import rows as paged_rows
from test.boundary.test_region_pages import owner_fixture
from test.boundary.test_region_program import region


X = ScalarType.INDEX
KV = BorrowedResource("kv", (2, 1, 8))


def source():
    tensor = TensorType((2,), "torch.float32")
    ranges = tuple(index_cell(i) for i in range(3))
    read = Function("read", (X, X, X), (tensor,), ("kv",), (ResourceAccess("kv", AccessMode.READ, ranges),))
    write = Function("write", (X, X, X), (), ("kv",), (ResourceAccess("kv", AccessMode.WRITE, ranges),))
    return Program(region((), (
        Coordinate((), (Value("zero", X),), AffineIndex(0)), Coordinate((), (Value("one", X),), AffineIndex(1)),
        Coordinate((), (Value("token", X),), AffineIndex(4)),
        Call(("zero", "zero", "token"), (Value("old", tensor),), read.name),
        Call(("one", "zero", "token"), (), write.name),
        Call(("zero", "zero", "token"), (), write.name),
        Call(("zero", "zero", "token"), (Value("new", tensor),), read.name),
    ), ("old", "new")), (read, write), (KV,))


def paged_view():
    cache = torch.arange(32, dtype=torch.float32).reshape(4, 1, 4, 2)
    layer = PagedLayer("layer", cache, torch.tensor([[0, 1], [0, 2]]), torch.tensor([4, 5, 8, 9]))
    return cache, PagedResourceView(KV, bind_paged_access(paged_rows(), (layer,)), ((0, "layer"),))


def test_readonly_tensor_borrows_allow_shared_reads_and_reject_writable_aliases():
    execution, _, rows, refs = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    weights = BorrowedResource("weights", (2,))
    tensor = TensorType((2,), "torch.float32")
    read = Function("weights.read", (X,), (tensor,), ("weights",), (
        ResourceAccess("weights", AccessMode.READ, (index_cell(0),)),))
    program = Program(region((), (
        Coordinate((), (Value("zero", X),), AffineIndex(0)), Coordinate((), (Value("one", X),), AffineIndex(1)),
        Call(("zero",), (Value("a", tensor),), read.name),
        Call(("one",), (Value("b", tensor),), read.name),
    ), ("a", "b")), (read,), (KV, weights))
    cache, kv = paged_view()
    weight = torch.tensor([3., 7.])
    def bind(value):
        view = ReadonlyTensorView(weights, tuple(ReadonlyTensorCell.bind((i,), (value,)) for i in range(2)))
        return RegionResources(program, invocation.context, (), (ExternalFunction(read, lambda inputs: (value,)),),
            borrows=(PagedResourceBinding(kv, invocation.context), ReadonlyResourceBinding(view, invocation.context)))
    pages = bind(weight)
    schedule = "{ S0[] -> [1]; S1[] -> [0] }"
    runner = RegionExecution(program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule),
        {read.name: external_primitive(read)})
    for value in runner.run({}, pages=pages).values:
        torch.testing.assert_close(value, weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match="cannot authorize a write"):
        pages.bind_accesses(replace(read, accesses=(replace(read.accesses[0], mode=AccessMode.WRITE),)), (0,))
    with pytest.raises(ValueError, match="writable paged"):
        bind(cache.reshape(-1)[:2])
    arena_value = invocation.context.arena.borrow_field(invocation.context.pin, refs[0], LayoutFieldId(0))
    with pytest.raises(ValueError, match="arena storage"):
        bind(arena_value)
    weight.add_(1)
    with pytest.raises(ValueError, match="changed after binding"):
        runner.run({}, pages=pages)
    assert not invocation.context.page_publications
    execution.cancel_step(invocation)
    execution.retire(tuple(row.identity for row in rows))


def test_borrowed_schedule_uses_original_storage_and_rejects_unproved_accesses():
    execution, _, rows, _ = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    program = source()
    assert decode_boundary(encode_boundary(program), Program) == program
    assert not any(s.opaque for s in analyze_region_program(program, optimize=False).statements)
    cache, view = paged_view()
    old = cache[1, 0, 0].clone()
    calls = []
    def read(inputs):
        request, _, token = inputs
        calls.append(("read", request))
        return (cache[request + 1, 0, token - 4],)
    def write(inputs):
        request, _, token = inputs
        calls.append(("write", request))
        cache[request + 1, 0, token - 4].fill_(10. + request)
        return ()
    endpoints = tuple(ExternalFunction(f, callback) for f, callback in zip(program.functions, (read, write), strict=True))
    with pytest.raises(ValueError, match="proofs must be complete"):
        RegionResources(program, invocation.context, (), endpoints)
    binding = PagedResourceBinding(view, invocation.context)
    pages = RegionResources(program, invocation.context, (), endpoints, borrows=(binding,))
    with pytest.raises(ValueError, match="original read/write interval"):
        pages.external(program.functions[1], (0, 0, 0))
    assert not calls
    schedule = "{ S0[] -> [1]; S1[] -> [0]; S2[] -> [2]; S3[] -> [3] }"
    cells = {f.name: external_primitive(f) for f in program.functions}
    imports = import_region_functions(program, cells)
    reference = analyze_region_program(program, optimize=False).reference_schedule
    request = RegionSelectionRequest(imports, (RegionCandidate(reference, imports, (20,)), RegionCandidate(schedule, imports, (10,))),
        (RegionPayloadLimit("cpu", 1024),), True, "schedule selection oracle")
    selection = select_region_candidate(program, request)
    assert selection.selected == 1
    runner = bind_region_candidate(PreparedRegionCandidate(program, request.candidates[selection.selected], (), cells),
        request.payload_limits, selection.candidates[selection.selected])
    result = runner.run({}, pages=pages)
    assert calls == [("write", 1), ("read", 0), ("write", 0), ("read", 0)]
    torch.testing.assert_close(result.values[0], old, rtol=0, atol=0)
    torch.testing.assert_close(result.values[1], torch.full((2,), 10.), rtol=0, atol=0)
    torch.testing.assert_close(cache[2, 0, 0], torch.full((2,), 11.), rtol=0, atol=0)
    assert not invocation.context.page_publications
    invalid = replace(program, functions=(replace(program.functions[0], accesses=()), program.functions[1]))
    with pytest.raises(ValueError, match="explicit indexed accesses"):
        validate_region_program(invalid)
    execution.cancel_step(invocation)
    with pytest.raises(RuntimeError, match="released"):
        runner.run({}, pages=pages)
    execution.retire(tuple(row.identity for row in rows))


def test_borrow_names_cannot_hide_aliases_with_each_other_or_arena_allocations():
    execution, _, rows, refs = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    program, (_, view) = source(), paged_view()
    endpoints = tuple(ExternalFunction(f, lambda inputs: ()) for f in program.functions)
    other = BorrowedResource("other", KV.domain)
    aliased = replace(program, resources=(KV, other))
    with pytest.raises(ValueError, match="aliases"):
        RegionResources(aliased, invocation.context, (), endpoints, borrows=(PagedResourceBinding(view, invocation.context),
            PagedResourceBinding(replace(view, resource=other), invocation.context)))
    # This view overlaps the original pool, including its future COW capacity.
    tensor = execution.arena.borrow_field(invocation.context.pin, refs[0], LayoutFieldId(0))
    layer = PagedLayer("layer", tensor.view(1, 1, 2, 3), torch.tensor([[0]]), torch.tensor([0]))
    pool_view = PagedResourceView(KV, bind_paged_access((rows[0],), (layer,)), ((0, "layer"),))
    with pytest.raises(ValueError, match="overlaps original arena"):
        RegionResources(program, invocation.context, (), endpoints, borrows=(PagedResourceBinding(pool_view, invocation.context),))
    assert not invocation.context.page_publications
    execution.cancel_step(invocation)
    execution.retire(tuple(row.identity for row in rows))
