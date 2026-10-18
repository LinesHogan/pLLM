"""Storage identity, last-use envelopes, and traffic facts consumed by AER."""

import weakref
from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pllm import _core
from pllm.contracts.tensor_access import TensorAccessKind, TensorMemoryProfile
from pllm.execution.codec import decode_boundary, encode_boundary, tensor_node_metadata
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.tensor_access import import_tensor_access_graph, profile_tensor_memory, validate_tensor_memory
from test.boundary.test_tensor_rewrites import reassociation_candidates


class StorageObserver(TorchDispatchMode):
    """Observe live real tensor storages, excluding allocator and kernel scratch."""

    def __init__(self, inputs):
        super().__init__()
        self.tensors = [weakref.ref(t) for t in inputs]
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        outputs = result if isinstance(result, tuple) else (result,)
        self.tensors.extend(weakref.ref(t) for t in outputs if isinstance(t, torch.Tensor))
        alive = [ref() for ref in self.tensors]
        storages = {
            (str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
            for t in alive
            if t is not None and t.untyped_storage().nbytes()
        }
        self.peak = max(self.peak, sum(storages.values()))
        return result


def capture(function, inputs, outputs=("out",)):
    return ExportedCell.capture(function, tuple(f"in{i}" for i in range(len(inputs))), outputs, inputs)


def chain(a, b, c):
    return ((a @ b) @ c,)


def test_less_arithmetic_can_increase_memory_traffic_and_peak_storage():
    inputs = tuple(torch.ones(shape, dtype=torch.float64) for shape in ((4, 1), (1, 4), (4, 32)))
    source = capture(chain, inputs)
    (candidate,) = reassociation_candidates(source)
    (before,) = profile_tensor_memory(source).devices
    (after,) = profile_tensor_memory(candidate.cell).devices
    assert candidate.rewrite.after.flops == 512 < candidate.rewrite.before.flops == 1056
    assert before.logical_read_bytes + before.logical_write_bytes == 2368
    assert after.logical_read_bytes + after.logical_write_bytes == 2624
    assert before.peak_new_tensor_bytes == 1152 < after.peak_new_tensor_bytes == 1280
    assert before.input_bytes == after.input_bytes == 1088
    assert before.result_bytes == after.result_bytes == 1024


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_aliases_and_last_uses_match_real_unfused_graph_storage(device):
    def function(x):
        a = x + 1
        b = a.T[:2].clone()
        c = b * 2
        return c, c.T

    inputs = (torch.ones(3, 4, device=device),)
    source = capture(function, inputs, ("c", "ct"))
    imported = import_tensor_access_graph(source)
    original_graph = str(source.graph.graph)
    profile = profile_tensor_memory(source)
    (facts,) = profile.devices
    assert facts.input_bytes == 48
    assert facts.result_bytes == 24  # Two outputs share one allocation.
    assert facts.peak_new_tensor_bytes == 72
    assert facts.peak_live_tensor_bytes == 120
    assert facts.logical_read_bytes == 48 + 24 + 24
    assert facts.logical_write_bytes == 48 + 24 + 24
    assert not profile.unknown_accesses
    views = [op for op in imported.operations if op.kind is TensorAccessKind.VIEW]
    assert len(views) == 3
    for op in views:
        result = next(v for v in imported.values if v.name == op.node)
        assert any(v.name in op.inputs and v.storage == result.storage for v in imported.values)
    with StorageObserver(inputs) as observed:
        outputs = source(*inputs)
    assert observed.peak == facts.peak_live_tensor_bytes
    assert outputs[0].untyped_storage().data_ptr() == outputs[1].untyped_storage().data_ptr()
    assert str(source.graph.graph) == original_graph
    validate_tensor_memory(source, profile)
    assert decode_boundary(encode_boundary(profile), TensorMemoryProfile) == profile


def test_tiny_returned_slice_retains_the_whole_base_storage():
    def function(x):
        return ((x + 1)[:1, :1],)

    source = capture(function, (torch.ones(64, 64),))
    (facts,) = profile_tensor_memory(source).devices
    assert source.output_bytes == 4
    assert facts.result_bytes == 64 * 64 * 4
    assert facts.peak_new_tensor_bytes == facts.result_bytes


def test_views_of_input_allocate_nothing_and_keep_original_storage():
    def function(x):
        return x.T, x[:1]

    source = capture(function, (torch.ones(3, 4),), ("transposed", "slice"))
    (facts,) = profile_tensor_memory(source).devices
    assert facts.input_bytes == facts.result_bytes == facts.peak_live_tensor_bytes == 48
    assert facts.peak_new_tensor_bytes == facts.logical_read_bytes == facts.logical_write_bytes == 0


def test_rewrite_refreshes_downstream_view_aliases_and_import_ignores_stale_metadata():
    def function(a, b, c):
        product = (a @ b) @ c
        return product.T, product[:1]

    inputs = (torch.ones(4, 2), torch.ones(2, 8), torch.ones(8, 3))
    source = capture(function, inputs, ("transpose", "slice"))
    (candidate,) = reassociation_candidates(source)
    nodes = {node.name: node for node in candidate.cell.graph.graph.nodes}
    product = tensor_node_metadata(nodes[candidate.rewrite.replacement_nodes[1]])
    view = next(n for n in nodes.values() if str(n.target) == "aten.permute.default")
    assert tensor_node_metadata(view).untyped_storage() is product.untyped_storage()
    profile = profile_tensor_memory(candidate.cell)
    (facts,) = profile.devices
    assert facts.result_bytes == 4 * 3 * 4
    # Reimport must replay fake semantics, not use arbitrary cached view metadata.
    view.meta["val"] = tensor_node_metadata(next(n for n in nodes.values() if n.op == "placeholder"))
    assert profile_tensor_memory(candidate.cell) == profile


def test_unknown_access_pattern_is_explicit_not_a_zero_traffic_estimate():
    def function(x):
        return (x.sin(),)

    source = capture(function, (torch.ones(8),))
    profile = profile_tensor_memory(source)
    assert profile.unknown_accesses == ("sin",)
    (facts,) = profile.devices
    assert facts.peak_new_tensor_bytes == 32
    # These are only the known subtotal; the missing operation is mandatory
    # evidence that this profile cannot supply a complete traffic estimate.
    assert facts.logical_read_bytes == facts.logical_write_bytes == 0


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float64),
        pytest.param(
            "cuda", torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
        ),
    ],
)
def test_swiglu_split_keeps_shared_projection_storage_and_known_accesses(device, dtype):
    def mlp(hidden, gate_up_weight, down_weight):
        gate, up = torch.nn.functional.linear(hidden, gate_up_weight).chunk(2, dim=-1)
        return (torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_weight),)

    inputs = tuple(torch.randn(shape, device=device, dtype=dtype) for shape in ((3, 4), (12, 4), (4, 6)))
    saved = tuple(tensor.clone() for tensor in inputs)
    source = capture(mlp, inputs)
    imported = import_tensor_access_graph(source)
    profile = profile_tensor_memory(source)
    assert not profile.unknown_accesses
    assert source.transformations == ()
    assert not any("getitem" in str(node.target) for node in source.graph.graph.nodes)
    views = [operation for operation in imported.operations if operation.kind is TensorAccessKind.VIEW]
    assert len(views) == 2
    values = {value.name: value for value in imported.values}
    assert values[views[0].node].storage == values[views[1].node].storage
    assert {values[view.node].offset for view in views} == {0, 6}
    with StorageObserver(inputs) as observed:
        (actual,) = source(*inputs)
    torch.testing.assert_close(actual, mlp(*inputs)[0], rtol=0, atol=0)
    assert observed.peak == profile.devices[0].peak_live_tensor_bytes
    for value, original in zip(inputs, saved, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    validate_tensor_memory(source, profile)


def test_cold_import_does_not_execute_source_math_or_profile_compiled_graph_as_unfused():
    def forbidden(*args):
        raise AssertionError("real tensor math was evaluated")

    source = capture(chain, (torch.ones(4, 2), torch.ones(2, 8), torch.ones(8, 3)))
    assert profile_tensor_memory(replace(source, function=forbidden)) == profile_tensor_memory(source)
    with pytest.raises(ValueError, match="backend access model"):
        profile_tensor_memory(replace(source, transformations=("torch.compile",)))


def test_native_witness_rejects_tampered_costs_releases_and_signatures():
    source = capture(chain, (torch.ones(4, 2), torch.ones(2, 8), torch.ones(8, 3)))
    profile = profile_tensor_memory(source)
    for forged in (
        replace(profile, devices=(replace(profile.devices[0], peak_new_tensor_bytes=0),)),
        replace(profile, steps=(replace(profile.steps[0], release_storages=(0,)), *profile.steps[1:])),
        replace(profile, access_digest="a" * 64),
        replace(profile, signature="b" * 64),
    ):
        with pytest.raises(ValueError, match="memory witness"):
            validate_tensor_memory(source, forged)


def test_native_import_checks_storage_spans_aliases_and_source_order():
    source = capture(chain, (torch.ones(4, 2), torch.ones(2, 8), torch.ones(8, 3)))
    graph = import_tensor_access_graph(source)
    forged = (
        replace(graph, storages=(replace(graph.storages[0], bytes=1), *graph.storages[1:])),
        replace(graph, values=(replace(graph.values[0], stride=(2**63, 1)), *graph.values[1:])),
        replace(graph, operations=tuple(reversed(graph.operations))),
        replace(graph, operations=(replace(graph.operations[0], kind=TensorAccessKind.VIEW), *graph.operations[1:])),
        replace(graph, values=(replace(graph.values[0], shape=(2**63, 4)), *graph.values[1:])),
        replace(graph, inputs=(graph.inputs[0], graph.inputs[0], graph.inputs[2])),
    )
    for imported in forged:
        with pytest.raises(ValueError):
            _core.profile_tensor_memory(encode_boundary(imported))


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), ()])
def test_empty_and_scalar_storages(shape):
    def function(x):
        return (x + 1,)

    inputs = (torch.ones(shape),)
    (facts,) = profile_tensor_memory(capture(function, inputs)).devices
    size = inputs[0].numel() * inputs[0].element_size()
    assert facts.input_bytes == facts.result_bytes == facts.peak_new_tensor_bytes == size
    assert facts.peak_live_tensor_bytes == size * 2
