"""Independent frozen Huber/AdamW oracle through the ordinary workload API."""

import torch

from examples.huber import HuberWorkload
from pllm import testing
from pllm.authoring import TensorResult
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from test import reference_huber as oracle
from test.backend_facts import backend_facts


def effective(guidance):
    """Rows the guidance actually changes (nonzero weight) and their predictions."""
    if guidance.row_weights is None:
        return guidance.affected_row_ids, guidance.values
    keep = [index for index, weight in enumerate(guidance.row_weights.tolist()) if weight]
    return tuple(guidance.affected_row_ids[index] for index in keep), guidance.values[keep]


def test_huber() -> None:
    for device_name in ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",):
        for hidden, rank in ((4, 2), (8, 3)):
            _huber_lifecycle(hidden, rank, device_name)


def _huber_lifecycle(hidden: int, rank: int, device_name: str) -> None:
    device = torch.device(device_name)
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    bindings = tuple(
        PackedHiddenResourceBinding(
            str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]"
        )
        for layer in (0, -1)
    )
    facts = backend_facts(
        hidden,
        bindings,
        RuntimeFacts(device).timeline(),
        PageBundleProviderFacts("test.huber@1", 12, StateDeviceKind(device.type)),
        max_lineages=3,
        memory_bytes=100000,
    )
    execution = HuberWorkload(rank).compile_backend(facts).execution
    identities = tuple(RequestIdentity(str(i), 0, 0) for i in range(3))
    latest = list(execution.admit(identities, stream))
    initial = tuple(latest)
    held = execution.arena.pin_exact(initial, stream)
    originals = tuple(execution.snapshot(ref, stream) for ref in initial)
    generator = torch.Generator(device=device).manual_seed(7)
    parameters = [oracle.initialize(hidden, rank, generator, device) for _ in identities]
    optimizers = [oracle.optimizer(pair) for pair in parameters]
    for snapshot, pair in zip(originals, parameters, strict=True):
        torch.testing.assert_close(snapshot[0], torch.cat([p.flatten() for p in pair]), rtol=0, atol=0)
        for field in snapshot[1:]:
            assert torch.count_nonzero(field).item() == 0

    updates = 0
    trained_rows = 0
    for step_id, owners in enumerate(((1, 0, 1), (2, 1, 1, 0), (0, 2, 0, 2, 2), (1,), (2, 2, 2))):
        count = len(owners)
        source = torch.linspace(-0.8 + step_id * 0.1, 0.9, count * hidden, device=device).reshape(count, hidden)
        target = source.flip(1) * -0.6 + 0.13
        step = testing.step_input(
            tuple(identities[owner] for owner in owners),
            {bindings[0]: source, bindings[1]: target},
            step_id=step_id,
            token_offset=step_id,
            sampler_order=tuple(reversed(range(count))),
            stream=stream,
        )
        before = tuple(execution.snapshot(ref, stream) for ref in latest)
        for last_anchor in (0, execution.sampler_anchor_id):
            cancelled = execution.start_step(step, stream)
            for anchor in range(last_anchor + 1):
                execution.execute_anchor(cancelled, AnchorId(anchor))
            execution.cancel_step(cancelled)
            assert cancelled.value is None and cancelled.loss is None
            for ref, snapshot in zip(latest, before, strict=True):
                assert execution.arena.latest(ref.lineage_id) == ref
                for actual, expected in zip(execution.snapshot(ref, stream), snapshot, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        active = tuple(link for link in step.sampler_rows.links if latest[owners[link.packed_row_index]].version_id > 0)
        predicted = (
            torch.cat(
                [
                    oracle.predict(
                        source[link.packed_row_index : link.packed_row_index + 1],
                        parameters[owners[link.packed_row_index]],
                    ).detach()
                    for link in active
                ]
            )
            if active
            else source.new_empty((0, hidden))
        )
        handle = execution.start_step(step, stream)
        for anchor in range(execution.sampler_anchor_id + 1):
            execution.execute_anchor(handle, AnchorId(anchor))
        guidance = execution.sampler_guidance(handle, stream)
        rows, values = effective(guidance)
        assert rows == tuple(link.sampler_row_index for link in active)
        assert guidance.beta == 0.2
        torch.testing.assert_close(values, predicted)
        for ref, snapshot in zip(latest, before, strict=True):
            for actual, expected in zip(execution.snapshot(ref, stream), snapshot, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        execution.complete_sampler(handle, stream)
        pointers = set()
        saved_parameters = []

        def save(tensor):
            if tensor.shape == (count, rank, hidden):
                pointers.add(handle.value.leaf.untyped_storage().data_ptr())
                saved_parameters.append((tensor.untyped_storage().data_ptr(), tensor.stride(0)))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
            for timeline_anchor in execution.arena.plan.timeline.anchors:
                if timeline_anchor.ordinal > execution.sampler_anchor_id:
                    execution.execute_anchor(handle, AnchorId(timeline_anchor.ordinal))
        if len(set(owners)) == 1:
            assert saved_parameters and all(
                pointer in pointers and (count == 1 or stride == 0) for pointer, stride in saved_parameters
            )
        result = execution.finish_step(handle)
        assert isinstance(result, TensorResult)
        assert handle.value is None and handle.loss is None
        assert result.publication is not None and result.loss is not None
        assert result.guidance is guidance
        for old, new, loss in zip(
            result.publication.before_refs, result.publication.after_refs, result.loss, strict=True
        ):
            owner = latest.index(old)
            selected = [i for i, candidate in enumerate(owners) if candidate == owner]
            pair, optimizer = parameters[owner], optimizers[owner]
            expected_loss = oracle.step(source[selected], target[selected], pair, optimizer)
            torch.testing.assert_close(loss, expected_loss)
            gradients: list[torch.Tensor] = []
            for parameter in pair:
                assert parameter.grad is not None
                gradients.append(parameter.grad.flatten())
            gradient = torch.cat(gradients)
            torch.testing.assert_close(result.parameter_gradients[old], gradient, rtol=1e-4, atol=1e-7)
            state_a, state_b = (optimizer.state[p] for p in pair)
            expected_fields = (
                torch.cat([p.flatten() for p in pair]),
                gradient,
                state_a["step"].to(device),
                torch.cat([state_a["exp_avg"].flatten(), state_b["exp_avg"].flatten()]),
                torch.cat([state_a["exp_avg_sq"].flatten(), state_b["exp_avg_sq"].flatten()]),
            )
            torch.testing.assert_close(state_a["step"], state_b["step"])
            for actual, expected in zip(execution.snapshot(new, stream), expected_fields, strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)
            assert new.version_id == old.version_id + 1
            latest[owner] = new
        updates += len(set(owners))
        trained_rows += count
        for ref, snapshot in zip(initial, originals, strict=True):
            for actual, expected in zip(execution.arena.snapshot(held, ref), snapshot, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    execution.arena.wait_release(execution.arena.release_pin(held))
    execution.retire(identities)
    diagnostics = execution.diagnostics()
    assert diagnostics.loss_count == diagnostics.update_count == updates
    assert diagnostics.trained_rows == trained_rows
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
    assert execution.arena.free_pages == 12
