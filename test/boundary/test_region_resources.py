"""Borrowed and external resources: original storage, explicit effects, no hidden aliases.

Regions read model weights, KV pages and backend endpoints that the invocation
does not own. A ``RegionResources`` binding needs physical proof from the
original invocation; it must reject aliases with writable or arena storage,
expire with the invocation, and never copy or let callers mutate the weights.
"""

from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pllm.contracts.region_composition import inline_function_bodies
from pllm.contracts.region_imports import BorrowedTensorImport
from pllm.contracts.region_program import (
    AccessMode,
    AffineIndex,
    BorrowedResource,
    Call,
    Coordinate,
    ExternalResource,
    Function,
    If,
    Program,
    ResourceAccess,
    ScalarType,
    TensorType,
    Value,
)
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelectionPolicy
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.decoder_region import index_cell
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.exported_region import import_exported_region
from pllm.execution.exported_resources import borrow_exported_inputs
from pllm.execution.module_graph import exported_parameter_fields
from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
from pllm.execution.region_native import BorrowedTensorPrimitive, external_primitive
from pllm.execution.region_optimization import optimize_region
from pllm.execution.region_resources import (
    ExternalFunction,
    PagedResourceBinding,
    ReadonlyResourceBinding,
    RegionResources,
)
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.region_streams import RegionStreamBindings, lower_region_streams
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from test.region_fixtures import bind, limits, owner_fixture, paged_rows, region, stream_execution
from test.region_reference import interpret

X, I = ScalarType.INDEX, ScalarType.INT
KV = BorrowedResource("kv", (2, 1, 8))


class Copies(TorchDispatchMode):
    """Record the shapes of every tensor clone."""

    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.clone.default:
            self.shapes.append(tuple(args[0].shape))
        return func(*args, **(kwargs or {}))


def invocation_scope(stream=None):
    execution, _, rows, refs = owner_fixture(rounds=1, stream=stream)
    return execution, rows, refs, execution.start_step(rows, stream)


def close_scope(execution, rows, invocation):
    execution.cancel_step(invocation)
    execution.retire(tuple(row.identity for row in rows))


def test_external_endpoints_run_in_order_and_expire_with_the_invocation():
    first = Function("owner.first", (), (I,), ("first",))
    second = Function("owner.second", (), (I,), ("second",))
    program = Program(
        region((), (Call((), (Value("x", I),), first.name), Call((), (Value("y", I),), second.name)), ("x", "y")),
        (first, second),
        (ExternalResource("first"), ExternalResource("second")),
    )
    execution, rows, refs, invocation = invocation_scope()
    events = []

    def call(inputs):
        events.append(len(events) + 1)
        return (events[-1],)

    resources = RegionResources(
        program, invocation.context, (), tuple(ExternalFunction(f, call) for f in program.functions)
    )
    result = bind(program, {f.name: external_primitive(f) for f in program.functions}).run({}, pages=resources)
    assert result.values == (1, 2) and events == [1, 2]
    assert not invocation.context.page_publications
    assert tuple(execution.arena.latest(ref.lineage_id) for ref in refs) == refs
    events.clear()
    assert interpret(program, {}, functions={f.name: call for f in program.functions}).values == result.values
    execution.cancel_step(invocation)
    with pytest.raises(RuntimeError):
        resources.external(program.functions[0], ())
    execution.retire(tuple(row.identity for row in rows))


def paged_view():
    cache = torch.arange(32, dtype=torch.float32).reshape(4, 1, 4, 2)
    layer = PagedLayer("layer", cache, torch.tensor([[0, 1], [0, 2]]), torch.tensor([4, 5, 8, 9]))
    return cache, PagedResourceView(KV, bind_paged_access(paged_rows(), (layer,)), ((0, "layer"),))


def test_readonly_borrows_allow_shared_reads_and_reject_writable_aliases():
    execution, rows, _, invocation = invocation_scope()
    weights = BorrowedResource("weights", (2,))
    tensor = TensorType((2,), "torch.float32")
    read = Function(
        "weights.read", (X,), (tensor,), ("weights",), (ResourceAccess("weights", AccessMode.READ, (index_cell(0),)),)
    )
    program = Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Coordinate((), (Value("one", X),), AffineIndex(1)),
                Call(("zero",), (Value("a", tensor),), read.name),
                Call(("one",), (Value("b", tensor),), read.name),
            ),
            ("a", "b"),
        ),
        (read,),
        (KV, weights),
    )
    cache, kv = paged_view()
    weight = torch.tensor([3.0, 7.0])

    def resources(value):
        view = ReadonlyTensorView(weights, tuple(ReadonlyTensorCell.bind((i,), (value,)) for i in range(2)))
        return RegionResources(
            program,
            invocation.context,
            (),
            (ExternalFunction(read, lambda inputs: (value,)),),
            borrows=(PagedResourceBinding(kv, invocation.context), ReadonlyResourceBinding(view, invocation.context)),
        )

    pages = resources(weight)
    runner = bind(program, {read.name: external_primitive(read)}, "{ S0[] -> [1]; S1[] -> [0] }")
    for value in runner.run({}, pages=pages).values:
        torch.testing.assert_close(value, weight, rtol=0, atol=0)
    with pytest.raises(ValueError):
        resources(cache.reshape(-1)[:2])
    weight.add_(1)
    with pytest.raises(ValueError):
        runner.run({}, pages=pages)
    assert not invocation.context.page_publications
    close_scope(execution, rows, invocation)


def kv_program():
    tensor = TensorType((2,), "torch.float32")
    ranges = tuple(index_cell(i) for i in range(3))
    read = Function("read", (X, X, X), (tensor,), ("kv",), (ResourceAccess("kv", AccessMode.READ, ranges),))
    write = Function("write", (X, X, X), (), ("kv",), (ResourceAccess("kv", AccessMode.WRITE, ranges),))
    return Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Coordinate((), (Value("one", X),), AffineIndex(1)),
                Coordinate((), (Value("token", X),), AffineIndex(4)),
                Call(("zero", "zero", "token"), (Value("old", tensor),), read.name),
                Call(("one", "zero", "token"), (), write.name),
                Call(("zero", "zero", "token"), (), write.name),
                Call(("zero", "zero", "token"), (Value("new", tensor),), read.name),
            ),
            ("old", "new"),
        ),
        (read, write),
        (KV,),
    )


def test_paged_borrows_use_original_kv_storage_and_reject_unproved_accesses():
    execution, rows, _, invocation = invocation_scope()
    program = kv_program()
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
        cache[request + 1, 0, token - 4].fill_(10.0 + request)
        return ()

    endpoints = tuple(ExternalFunction(f, cb) for f, cb in zip(program.functions, (read, write), strict=True))
    with pytest.raises(ValueError):
        RegionResources(program, invocation.context, (), endpoints)
    pages = RegionResources(
        program, invocation.context, (), endpoints, borrows=(PagedResourceBinding(view, invocation.context),)
    )
    with pytest.raises(ValueError):
        pages.external(program.functions[1], (0, 0, 0))
    assert not calls
    # Request 1's write is proven disjoint from request 0's accesses, so it may move first.
    cells = {f.name: external_primitive(f) for f in program.functions}
    moved = "{ S0[] -> [1]; S1[] -> [0]; S2[] -> [2]; S3[] -> [3] }"
    runner = bind(program, cells, moved)
    result = runner.run({}, pages=pages)
    assert calls == [("write", 1), ("read", 0), ("write", 0), ("read", 0)]
    torch.testing.assert_close(result.values[0], old, rtol=0, atol=0)
    torch.testing.assert_close(result.values[1], torch.full((2,), 10.0), rtol=0, atol=0)
    torch.testing.assert_close(cache[2, 0, 0], torch.full((2,), 11.0), rtol=0, atol=0)
    assert not invocation.context.page_publications
    execution.cancel_step(invocation)
    with pytest.raises(RuntimeError):
        runner.run({}, pages=pages)
    execution.retire(tuple(row.identity for row in rows))


def test_borrow_names_cannot_hide_aliases_with_each_other_or_the_arena():
    execution, rows, refs, invocation = invocation_scope()
    program, (_, view) = kv_program(), paged_view()
    endpoints = tuple(ExternalFunction(f, lambda inputs: ()) for f in program.functions)
    other = BorrowedResource("other", KV.domain)
    with pytest.raises(ValueError):
        RegionResources(
            replace(program, resources=(KV, other)),
            invocation.context,
            (),
            endpoints,
            borrows=(
                PagedResourceBinding(view, invocation.context),
                PagedResourceBinding(replace(view, resource=other), invocation.context),
            ),
        )
    # This view overlaps the arena pool, including its future COW capacity.
    tensor = execution.arena.borrow_field(invocation.context.pin, refs[0], LayoutFieldId(0))
    layer = PagedLayer("layer", tensor.view(1, 1, 2, 3), torch.tensor([[0]]), torch.tensor([0]))
    pool_view = PagedResourceView(KV, bind_paged_access((rows[0],), (layer,)), ((0, "layer"),))
    with pytest.raises(ValueError):
        RegionResources(
            program, invocation.context, (), endpoints, borrows=(PagedResourceBinding(pool_view, invocation.context),)
        )
    close_scope(execution, rows, invocation)


def weight_reader(name, tensor, field):
    declaration = Function(
        name, (X,), (tensor,), ("weights",), (ResourceAccess("weights", AccessMode.READ, (index_cell(0),)),)
    )
    return BorrowedTensorPrimitive(BorrowedTensorImport(declaration, field))


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_swiglu_reads_original_weights_without_copies_and_owns_returned_views(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dtype = torch.float64 if device == "cpu" else torch.bfloat16
    hidden, gate_up, down = (torch.randn(shape, device=device, dtype=dtype) for shape in ((2, 64), (128, 64), (64, 64)))
    originals = gate_up.clone(), down.clone()

    def mlp(h, gw, dw):
        gate, up = torch.nn.functional.linear(h, gw).chunk(2, -1)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, dw), gw[:1]

    cell = ExportedCell.capture(mlp, ("hidden", "gate_up", "down"), ("output", "weight_view"), (hidden, gate_up, down))
    h, g, d = cell.declaration.inputs
    out, view_type = cell.declaration.outputs
    load_gate, load_down = weight_reader("model.gate_up", g, 0), weight_reader("model.down", d, 1)
    resource = BorrowedResource("weights", (1,))
    program = Program(
        region(
            (Value("hidden", h),),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Call(("zero",), (Value("gw", g),), load_gate.declaration.name),
                Call(("zero",), (Value("dw", d),), load_down.declaration.name),
                Call(("hidden", "gw", "dw"), (Value("result", out), Value("slice", view_type)), cell.digest),
            ),
            ("result", "slice"),
        ),
        (load_gate.declaration, load_down.declaration, cell.declaration),
        (resource,),
    )
    functions = {value.declaration.name: value for value in (load_gate, load_down, cell)}
    schedule = "{ S0[] -> [0]; S1[] -> [1]; S2[] -> [3] }"
    assert analyze_region_capacity(program, schedule, functions).devices[0].peak_tensor_bytes_upper_bound < (
        gate_up.numel() * gate_up.element_size()
    )
    streams = None
    if device == "cpu":
        runner = bind(program, functions, schedule)
    else:
        streams = RegionStreamBindings(torch.cuda.Stream(device=device), (torch.cuda.Stream(device=device),))
        streams.owner.wait_stream(torch.cuda.current_stream(device))
        plan = lower_region_streams(program, schedule, functions, axis=0, lanes=2, pipeline_depth=2)
        runner = stream_execution(program, functions, plan, payload_limits=limits(plan))
    owner = None if streams is None else streams.owner
    execution, rows, _, invocation = invocation_scope(owner)
    proof = ReadonlyResourceBinding(
        ReadonlyTensorView(resource, (ReadonlyTensorCell.bind((0,), (gate_up, down)),)), invocation.context
    )
    loans = (load_gate.implementation, load_down.implementation)
    pages = RegionResources(program, invocation.context, (), borrows=(proof,), tensor_borrows=loans)
    assert load_gate.invoke((0,), pages)[0].data_ptr() == gate_up.data_ptr()
    with Copies() as copies:
        result = runner.run({"hidden": hidden}, pages=pages, streams=streams).values
    assert tuple(gate_up.shape) not in copies.shapes and tuple(down.shape) not in copies.shapes
    for actual, expected in zip(result, mlp(hidden, gate_up, down), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result[1].zero_()
    torch.testing.assert_close(gate_up, originals[0], rtol=0, atol=0)
    torch.testing.assert_close(down, originals[1], rtol=0, atol=0)
    close_scope(execution, rows, invocation)


def test_weight_borrowed_through_control_flow_cannot_escape_as_a_mutable_alias():
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    before = weight.clone()
    tensor = TensorType((8, 8), "torch.float32")
    load = weight_reader("model.weight", tensor, 0)
    branch = region(
        (),
        (
            Coordinate((), (Value("zero", X),), AffineIndex(0)),
            Call(("zero",), (Value("w", tensor),), load.declaration.name),
        ),
        ("w",),
    )
    resource = BorrowedResource("weights", (1,))
    program = Program(
        region(
            (Value("choose", ScalarType.BOOL),), (If(("choose",), (Value("out", tensor),), branch, branch),), ("out",)
        ),
        (load.declaration,),
        (resource,),
    )
    runner = bind(program, {load.declaration.name: load})
    execution, rows, _, invocation = invocation_scope()
    proof = ReadonlyResourceBinding(
        ReadonlyTensorView(resource, (ReadonlyTensorCell.bind((0,), (weight,)),)), invocation.context
    )
    pages = RegionResources(program, invocation.context, (), borrows=(proof,), tensor_borrows=(load.implementation,))
    for predicate in (False, True):
        (output,) = runner.run({"choose": predicate}, pages=pages).values
        torch.testing.assert_close(output, before, rtol=0, atol=0)
        output.fill_(-1)
        torch.testing.assert_close(weight, before, rtol=0, atol=0)
    close_scope(execution, rows, invocation)


def test_exported_parameters_and_buffers_are_borrowed_from_the_module():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((64, 64), 1 / 64))
            self.register_buffer("offset", torch.full((64,), 0.25))

        def forward(self, x):
            return torch.nn.functional.silu(x @ self.weight + self.offset), self.weight[:1]

    model, x = Projection(), torch.full((2, 64), 2.0)
    resource = BorrowedResource("parameters", (1,))
    exported = torch.export.export(model, (x,))
    source = import_exported_region(exported, bind_kernel=lambda partition: pytest.fail("unexpected kernel adapter"))
    body, functions = borrow_exported_inputs(
        source, resource, exported_parameter_fields(exported, (model.weight, model.offset))
    )
    declaration = body.declaration
    outputs = tuple(Value(f"out{i}", type_) for i, type_ in enumerate(declaration.outputs))
    wrapper = Program(
        region(
            (Value("x", declaration.inputs[0]),),
            (
                Coordinate((), (Value("layer", X),), AffineIndex(0)),
                Call(("x", "layer"), outputs, declaration.name),
            ),
            tuple(value.name for value in outputs),
        ),
        (declaration,),
        (resource,),
    )
    program = inline_function_bodies(wrapper, (body,))
    loans = tuple(f.implementation for f in functions.values() if isinstance(f, BorrowedTensorPrimitive))
    execution, rows, _, invocation = invocation_scope()
    original = model.weight.clone(), model.offset.clone()
    proof = ReadonlyResourceBinding(
        ReadonlyTensorView(resource, (ReadonlyTensorCell.bind((0,), (model.weight, model.offset)),)), invocation.context
    )
    pages = RegionResources(program, invocation.context, (), borrows=(proof,), tensor_borrows=loans)
    options = dict(
        payload_limits=(RegionPayloadLimit("cpu", 1 << 20),), options=RegionSelectionPolicy(tuning="measured")
    )
    with pytest.raises(ValueError):
        optimize_region(program, functions, {"x": x}, max_candidates=1, **options)
    with pages.scope():
        runner = optimize_region(
            program,
            functions,
            {"x": x},
            pages=pages,
            warmup_rounds=0,
            measurement_rounds=1,
            max_candidates=8,
            **options,
        )
        with Copies() as copies:
            actual = runner.execution.run({"x": x}, pages=pages).values
        assert tuple(model.weight.shape) not in copies.shapes
        for value, reference in zip(actual, model(x), strict=True):
            torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
        actual[1].zero_()
        torch.testing.assert_close(model.weight, original[0], rtol=0, atol=0)
        torch.testing.assert_close(model.offset, original[1], rtol=0, atol=0)
    close_scope(execution, rows, invocation)


def test_meta_memory_tracks_aliases_saved_gradients_and_real_cuda_peaks():
    from pllm.execution.memory import measure

    source = torch.empty(1024, device="meta")
    assert measure(lambda: source.view(32, 32)).workspace == 0

    def views():
        value = source.clone()
        return value, value.view(32, 32), value[1:]

    assert measure(views).workspace == 4096

    def temporary():
        source.clone()
        return source.clone()

    assert measure(temporary).workspace == 4096

    def program(x):
        leaf = x.detach().requires_grad_(True)
        loss = leaf.sin().square().mean()
        return loss.detach(), torch.autograd.grad(loss, leaf)[0]

    bound = measure(lambda: program(source)).workspace
    assert bound > source.numel() * source.element_size()
    if torch.cuda.is_available():
        actual = torch.randn(source.shape, device="cuda")
        program(actual)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = program(actual)
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() - baseline <= bound
        assert result[1].shape == actual.shape
