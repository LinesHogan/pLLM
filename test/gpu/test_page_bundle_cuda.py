"""Evidence 13: actual CUDA PageBundle COW and exact event-gated reclamation."""


from pllm.execution.ptts.page_bundle_planning import compile_page_bundle_plan
from pllm.contracts.semantic_graph import AnchorId
from dataclasses import replace
import multiprocessing as mp
from unittest.mock import patch

import pytest
import torch

from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_bundle import PageAccess, PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.workloads.esamp_request import ESampRequestWorkload
from pllm.workloads.esamp_request import ESampRequestConfig
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.semantic_graph import AnchorSymbol, SchedulerTimeline, TimelineAnchor
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId
from test.backend_facts import backend_facts


def _plan(shared=False, rank=2, memory_bytes=10000):
    bindings = tuple(PackedHiddenResourceBinding(str(layer),
        HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]") for layer in (0, -1))
    facts = backend_facts(4, bindings, SchedulerTimeline(tuple(TimelineAnchor(AnchorSymbol(name), index)
            for index, name in enumerate(("model.execute", "sampler.pre_draw", "token.commit", "next.prepare"))),
            ("compute/cuda@1", "state/dense_page_cuda@1")),
        PageBundleProviderFacts("test.page_bundle@1", 6, StateDeviceKind.CUDA, 0, shared_memory=shared), memory_bytes=memory_bytes)
    declaration = ESampRequestWorkload(ESampRequestConfig(rank=rank)).declaration(facts)
    return compile_page_bundle_plan((declaration.graph, declaration.layouts), facts.timeline, facts.providers,
        facts.max_admitted_lineages, facts.state_memory_budget_bytes, facts.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ptts_waits_for_exact_fence_before_reclaim() -> None:
    arena = PageBundleArena(_plan())
    device = torch.device("cuda", 0)
    writer, reader = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    write = arena.prepare_admission(("request_distiller",) * 2, writer)
    arena.abort(write)
    assert arena.free_pages == 6
    write = arena.prepare_admission(("request_distiller",) * 2, writer)
    roots = arena.writable_roots(write)
    with torch.cuda.stream(writer):
        initial = tuple(torch.arange(root[0].numel(), device=device, dtype=torch.float32) / 10 + index
            for index, root in enumerate(roots))
        for root, value in zip(roots, initial, strict=True):
            root[0].copy_(value)
    admitted = arena.publish(write)
    arena.settle(admitted)
    pin = arena.pin_exact(admitted.after_refs, reader)
    borrowed = arena.borrow_field(pin, admitted.after_refs[0], LayoutFieldId(0))
    original = arena.snapshot(pin, admitted.after_refs[0])
    update = arena.prepare_transition(admitted.after_refs, writer)
    fields = AdamWFields(*(LayoutFieldId(index) for index in range(5)))
    with torch.cuda.stream(writer):
        gradients = tuple(torch.full_like(value, gradient) for value, gradient in zip(initial, (0.25, -0.5), strict=True))
        apply_adamw(arena.writable_roots(update), fields, AdamWConfig(0.01, (0.9, 0.999), 1e-8, 0.0), gradients)
    publication = arena.publish(update)
    arena.settle(publication)
    updated = arena.pin_exact(publication.after_refs, reader)
    result = arena.snapshot(updated, publication.after_refs[0])
    # Old roots must remain live while their original consumer stream still runs.
    with torch.cuda.stream(reader):
        torch.cuda._sleep(100_000_000)
        consumed = borrowed.clone()
    release = arena.release_pin(pin)
    assert arena.pending_pin_releases == 1
    assert arena.free_pages == 2
    usage = arena.usage()
    assert usage.live_field_pages == usage.retiring_field_pages == usage.free_field_pages == 10
    assert usage.private_versions == 0 and usage.pending_receipts >= 1
    with pytest.raises(RuntimeError, match="no new reads"):
        arena.snapshot(pin, admitted.after_refs[0])
    with pytest.raises(RuntimeError, match="issued"):
        arena.wait_release(replace(release))
    addresses = arena.pool_addresses
    pressure = arena.publish(arena.prepare_transition(publication.after_refs, writer))
    arena.settle(pressure)
    # All six slots are owned until the handed-back reader fence completes.
    update = arena.prepare_transition(pressure.after_refs, writer)
    assert arena.release_complete(release)
    arena.abort(update)
    assert arena.pool_addresses == addresses and arena.free_pages == 2
    torch.testing.assert_close(original[0], initial[0])
    torch.testing.assert_close(consumed, initial[0])
    torch.testing.assert_close(result[0], initial[0] - 0.01)
    assert result[2].item() == 1
    arena.retire(pressure.after_refs)
    replacement = arena.publish(arena.prepare_admission(("request_distiller",) * 2, writer))
    arena.settle(replacement)
    torch.testing.assert_close(arena.snapshot(updated, publication.after_refs[0])[0], result[0])
    arena.wait_release(arena.release_pin(updated))
    arena.retire(replacement.after_refs)
    arena.collect_completed(wait=True)
    assert arena.free_pages == 6 and arena.active_lineages == 0
    _cuda_ipc_lifecycle()
    _paged_linear_lifecycle()
    _paged_adamw_lifecycle()
    _paged_adamw_lifecycle(captured=False)


def _paged_linear_lifecycle():
    from pllm.kernels.paged_gemm import split_linear
    from pllm.kernels.paged_linear import PagedRows, TensorRows
    from pllm.execution.ptts.page_bundle import PagedKernel

    for transposed in (False, True):
        pool = torch.randn((4, 7, 129) if transposed else (4, 129, 7), device="cuda")
        matrices = pool.transpose(1, 2) if transposed else pool
        inputs = torch.randn((3, 129), device="cuda")
        inputs[2].fill_(float("nan"))
        slots = torch.tensor([0, 2], device="cuda")
        states = torch.tensor([1, 0, 1], device="cuda")
        active = torch.tensor([True, True, False], device="cuda")
        result = split_linear(inputs, matrices, slots, states, active)
        expected = torch.stack((inputs[0] @ matrices[2], inputs[1] @ matrices[0], inputs.new_zeros(7)))
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    arena = PageBundleArena(_plan())
    stream = torch.cuda.Stream(device=arena.device)
    initial = arena.prepare_admission(("request_distiller",) * 2, stream)
    with torch.cuda.stream(stream):
        for index, root in enumerate(arena.writable_roots(initial)):
            root[0].copy_(torch.linspace(-1, 1, root[0].numel(), device=arena.device) + index)
    initial = arena.publish(initial)
    arena.settle(initial)
    pin = arena.pin_exact(initial.after_refs, stream)
    with torch.cuda.stream(stream):
        rows, moments = arena.paged_rows((pin, pin), pin.refs, (LayoutFieldId(0), LayoutFieldId(3)), (1, 0, 1))
        assert rows.order is moments.order and rows.offsets is moments.offsets and rows.states is moments.states
        assert rows.slots.numel() == moments.slots.numel() == len(pin.refs)
        x = torch.arange(12, device=arena.device, dtype=torch.float32).reshape(3, 4).requires_grad_()
        weights = [arena.snapshot(pin, ref)[0][:16].view(4, 4).requires_grad_() for ref in pin.refs]
        tensor_rows = TensorRows(torch.stack([weight.detach().flatten() for weight in weights]),
            torch.tensor([1, 0, 1], device=arena.device))
        reference = torch.stack([x[i] @ weights[state] for i, state in enumerate((1, 0, 1))])
        with pytest.raises(ValueError, match="explicit derivatives"):
            rows.linear(x, 0, (4, 4))
        torch.testing.assert_close(rows.linear(x.detach(), 0, (4, 4)), reference)
        torch.testing.assert_close(tensor_rows.linear(x.detach(), 0, (4, 4)), reference)
        dy = torch.linspace(-1, 1, 12, device=arena.device).view(3, 4)
        dx_mask = torch.tensor([[1.], [0.], [1.]], device=arena.device)
        dw_mask = 1 - dx_mask
        dx, = torch.autograd.grad(reference, x, dy * dx_mask, retain_graph=True)
        dw = torch.autograd.grad(reference, weights, dy * dw_mask)
        torch.testing.assert_close(rows.input_gradient(dy * dx_mask, 0, (4, 4)), dx)
        torch.testing.assert_close(rows.parameter_gradient(x.detach(), dy * dw_mask), torch.stack(dw))
        selected_states = torch.tensor([1, -1, 0, -1], device=arena.device)[::2]
        for selection in (selected_states, selected_states[:1], selected_states[:0]):
            for implementation in (rows, tensor_rows):
                compact = implementation.parameter_gradient(x.detach(), dy * dw_mask, selection)
                torch.testing.assert_close(compact, torch.stack(dw).index_select(0, selection))
                assert compact.untyped_storage().nbytes() == compact.numel() * compact.element_size()
        repeated, = arena.paged_rows((pin, pin), pin.refs, (LayoutFieldId(0),), (1, 0, 1) * 11)
        torch.testing.assert_close(repeated.parameter_gradient(x.detach().repeat(11, 1),
            (dy * dw_mask).repeat(11, 1), selected_states), torch.stack(dw).flip(0) * 11)
        strided_mask = torch.stack((dx_mask.flatten().bool(), dw_mask.flatten().bool()), dim=1)[:, 0]
        dx_rows, dx_moments = PagedRows.select_rows((rows, moments), strided_mask)
        dw_rows, = PagedRows.select_rows((rows,), dw_mask.flatten().bool())
        restricted, = PagedRows.select_rows((dx_rows,), torch.tensor(True, device=arena.device).expand(3))
        torch.testing.assert_close(restricted.active, dx_rows.active)
        torch.testing.assert_close(restricted.offsets, dx_rows.offsets)
        assert dx_rows.order is dx_moments.order and dx_rows.offsets is dx_moments.offsets
        torch.testing.assert_close(dx_rows.input_gradient(dy.masked_fill(~dx_mask.bool(), float("nan")), 0, (4, 4)), dx)
        torch.testing.assert_close(dw_rows.parameter_gradient(x.detach().masked_fill(~dw_mask.bool(), float("nan")),
            dy.masked_fill(~dw_mask.bool(), float("nan"))), torch.stack(dw))
        with pytest.raises(ValueError, match="boolean"):
            PagedRows.select_rows((rows,), dx_mask.flatten())
        with pytest.raises(RuntimeError, match="private PageWrite"):
            dx_rows.write(rows.pool.new_zeros((2, rows.pool.shape[1])), torch.arange(2, device=arena.device))
        torch.testing.assert_close(tensor_rows.input_gradient(dy * dx_mask, 0, (4, 4)), dx)
        torch.testing.assert_close(tensor_rows.parameter_gradient(x.detach(), dy * dw_mask), torch.stack(dw))
        with pytest.raises(RuntimeError, match="private"):
            tensor_rows.write(torch.zeros((1, 4, 4), device=arena.device), torch.tensor([0], device=arena.device))
        before = rows.linear(x.detach(), 0, (4, 4))
        with pytest.raises(RuntimeError, match="private PageWrite"):
            rows.write(rows.pool.new_zeros((2, rows.pool.shape[1])), torch.arange(2, device=arena.device))
        inputs = (x.detach(), dy, torch.tensor([1, 0, 1], device=arena.device),
            *(torch.ones(3, device=arena.device, dtype=torch.bool) for _ in range(3)), selected_states[:1].clone())
        def selective(fields, inputs):
            forward, = PagedRows.select_rows(fields, inputs[3])
            dx, = PagedRows.select_rows(fields, inputs[4])
            dw, = PagedRows.select_rows(fields, inputs[5])
            return (forward.linear(inputs[0], 0, (4, 4)), dx.input_gradient(inputs[1], 0, (4, 4)),
                dw.parameter_gradient(inputs[0], inputs[1]), dw.parameter_gradient(inputs[0], inputs[1], inputs[6]))
        capture = arena.capture_kernel(pin, pin.refs, (LayoutFieldId(0),), inputs, PagedKernel.rows(2, selective))
        stable = tuple(tensor.data_ptr() for tensor in capture.results)
        for masks in (((True, False, True), (False, True, False), (True, True, False)),
                      ((False,) * 3,) * 3, ((True,) * 3,) * 3):
            for buffer, mask in zip(inputs[3:6], masks, strict=True):
                buffer.copy_(torch.tensor(mask, device=arena.device))
            inputs[6].sub_(1).neg_()
            arena.replay_captured(capture)
            expected_dx = torch.stack([dy[i] @ weights[state].detach().T for i, state in enumerate((1, 0, 1))])
            expected_dw = torch.stack([sum((torch.outer(x[i].detach(), dy[i]) * masks[2][i]
                for i, state in enumerate((1, 0, 1)) if state == owner), torch.zeros((4, 4), device=arena.device))
                for owner in range(2)])
            for actual, expected in zip(capture.results,
                    (reference.detach() * inputs[3].unsqueeze(1), expected_dx * inputs[4].unsqueeze(1),
                     expected_dw, expected_dw.index_select(0, inputs[6])), strict=True):
                torch.testing.assert_close(actual, expected)
            assert stable == tuple(tensor.data_ptr() for tensor in capture.results)
        for count in (0, 65537):
            state_rows = torch.arange(count, device=arena.device).remainder(2)
            mask = torch.arange(count, device=arena.device).remainder(3) == 0
            def select(fields, inputs):
                chosen, = PagedRows.select_rows(fields, inputs[1])
                return chosen.order, chosen.offsets, chosen.active
            selected = arena.capture_kernel(pin, pin.refs, (LayoutFieldId(0),),
                (state_rows, mask), PagedKernel.rows(0, select))
            for _ in range(2):
                arena.replay_captured(selected)
                expected = torch.cat(tuple(torch.where(mask & (state_rows == state))[0] for state in range(2)))
                torch.testing.assert_close(selected.results[0][:len(expected)], expected)
                torch.testing.assert_close(selected.results[1], torch.stack(tuple((mask & (state_rows <= state)).sum()
                    for state in range(2))).int())
                torch.testing.assert_close(selected.results[2], mask)
                mask.logical_not_()
    update = arena.prepare_transition(pin.refs, stream)
    with torch.cuda.stream(stream):
        for root in arena.writable_roots(update):
            root[0].add_(2)
        private, = arena.paged_rows((update, update), update.after_refs, (LayoutFieldId(0),), (0, 1, 0))
        assert rows.pool.data_ptr() == private.pool.data_ptr()
        private.linear(x.detach(), 0, (4, 4))
        selected = torch.tensor([1, -1, 0, -1], device=arena.device)[::2]
        values = torch.arange(2 * private.pool.shape[1], device=arena.device, dtype=torch.float32).reshape(2, -1)
        private.write(values, selected, scale=0.5)
        for root, expected in zip(arena.writable_roots(update), values.flip(0) * 0.5, strict=True):
            torch.testing.assert_close(root[0].flatten(), expected)
        torch.testing.assert_close(rows.linear(x.detach(), 0, (4, 4)), before, rtol=0, atol=0)
    arena.abort(update)
    with torch.cuda.stream(stream), pytest.raises(RuntimeError, match="consumed"):
        private.linear(x.detach(), 0, (4, 4))

    update = arena.prepare_transition(pin.refs, stream)
    with torch.cuda.stream(stream):
        selected = torch.arange(2, device=arena.device)
        values = rows.pool.new_full((2, rows.pool.shape[1]), 3)
        row_indices = torch.arange(2, device=arena.device)
        originals = tuple(root[0].clone() for root in arena.writable_roots(update))

        def increment(fields, inputs):
            fields[0].write(inputs[0], selected, add=True)
            return ()

        with pytest.raises(ValueError, match="private admission"):
            arena.capture_kernel(update, update.after_refs, (LayoutFieldId(0),),
                (values, row_indices), PagedKernel.rows(1, increment), preserve_fields=False)
        capture = arena.capture_kernel(update, update.after_refs, (LayoutFieldId(0),),
            (values, row_indices), PagedKernel.rows(1, increment))
        assert capture.buffers == ()
        for root, original in zip(arena.writable_roots(update), originals, strict=True):
            torch.testing.assert_close(root[0], original, rtol=0, atol=0)
        addresses = (*arena.pool_addresses, *capture.addresses)
        arena.replay_captured(capture)
        for root, original in zip(arena.writable_roots(update), originals, strict=True):
            torch.testing.assert_close(root[0], original + 3)
    arena.abort(update)
    other = torch.cuda.Stream(device=arena.device)
    update = arena.prepare_transition((pin.refs[1],), other)
    arena.select_captured(capture, update, update.after_refs)
    arena.replay_captured(capture)
    publication = arena.publish(update)
    arena.settle(publication)
    assert addresses == (*arena.pool_addresses, *capture.addresses)
    current = arena.pin_exact(publication.after_refs, other)
    with torch.cuda.stream(other):
        torch.testing.assert_close(arena.snapshot(current, current.refs[0])[0], originals[1] + 3)
    arena.wait_release(arena.release_pin(current))
    with pytest.raises(RuntimeError, match="consumed"):
        arena.replay_captured(capture)
    with torch.cuda.stream(stream):
        torch.testing.assert_close(rows.linear(x.detach(), 0, (4, 4)), before, rtol=0, atol=0)
    arena.wait_release(arena.release_pin(pin))
    with torch.cuda.stream(stream), pytest.raises(RuntimeError):
        rows.linear(x.detach(), 0, (4, 4))
    arena.retire((initial.after_refs[0], publication.after_refs[0]))
    arena.collect_completed(wait=True)
    assert arena.free_pages == 6
    cold = arena.prepare_admission(("request_distiller",) * 2, stream)
    disposable = arena.capture_kernel(cold, cold.after_refs, (LayoutFieldId(0),),
        (values, row_indices), PagedKernel.rows(1, increment), preserve_fields=False)
    with torch.cuda.stream(stream):
        for root in arena.writable_roots(cold):
            torch.testing.assert_close(root[0], torch.full_like(root[0], 9))
    arena.abort(cold)
    with pytest.raises(RuntimeError, match="consumed"):
        arena.replay_captured(disposable)
    arena.collect_completed(wait=True)
    assert arena.free_pages == 6 and arena.active_lineages == 0


def _paged_adamw_lifecycle(captured=True):
    from pllm.execution.ptts.page_bundle import PagedKernel
    from pllm.kernels.paged_linear import PagedRows
    for rank in (2, 41):
        arena = PageBundleArena(_plan(rank=rank, memory_bytes=100000))
        stream = torch.cuda.Stream(device=arena.device)
        with torch.cuda.stream(stream):
            fields = tuple(LayoutFieldId(index) for index in range(5))
            config = AdamWConfig(0.03, (0.8, 0.95), 1e-8, 0.1)
            admission = arena.prepare_admission(("request_distiller",) * 2, stream)
            with torch.cuda.stream(stream):
                for index, root in enumerate(arena.writable_roots(admission)):
                    root[0].copy_(torch.linspace(-0.3, 0.6, root[0].numel(), device=arena.device) + index)
            publication = arena.publish(admission)
            arena.settle(publication)
            initial = arena.pin_exact(publication.after_refs, stream)
            originals = tuple(arena.snapshot(initial, ref) for ref in initial.refs)
            parameters = [value[0].clone().requires_grad_() for value in originals]
            reference = torch.optim.AdamW(parameters, lr=config.lr, betas=config.betas,
                eps=config.eps, weight_decay=config.weight_decay, foreach=False)
            current = list(initial.refs)
            with torch.cuda.stream(stream):
                gradients = torch.zeros((2, parameters[0].numel()), device=arena.device)
            def compute(view, inputs):
                view.arena.adamw_captured(view, inputs[0], config)
                return ()
            kernel = PagedKernel(compute)
            with pytest.raises(RuntimeError, match="private PageWrite"):
                arena.capture_kernel(initial, initial.refs, fields, (gradients,), kernel)
            with pytest.raises(RuntimeError, match="private PageWrite"):
                PagedRows.adamw(arena.paged_rows((initial,) * 2, initial.refs, fields, ()), gradients, config)
            view = None
            for iteration, owners in enumerate(((1, 0), (0,), (0, 1))):
                write = arena.prepare_transition(tuple(current[i] for i in owners), stream)
                before = tuple(tuple(t.clone() for t in root) for root in arena.writable_roots(write))
                if not captured:
                    pages = arena.paged_rows((write,) * len(owners), write.after_refs, fields, ())
                    assert len({page.slots.untyped_storage().data_ptr() for page in pages}) == 1
                    assert all(page.slots.data_ptr() == page.write_slots.data_ptr() for page in pages)
                elif view is None:
                    view = arena.capture_kernel(write, write.after_refs, fields, (gradients,), kernel)
                    assert view.buffers == ()
                    for root, old in zip(arena.writable_roots(write), before, strict=True):
                        for value, expected in zip(root, old, strict=True):
                            torch.testing.assert_close(value, expected, rtol=0, atol=0)
                else:
                    arena.select_captured(view, write, write.after_refs)
                with torch.cuda.stream(stream):
                    gradients.fill_(float("nan"))
                    reference.zero_grad(set_to_none=True)
                    for row, owner in enumerate(owners):
                        gradients[row].fill_(0 if iteration == 2 else (owner + 1) * 0.05)
                        parameters[owner].grad = gradients[row].clone()
                    if captured:
                        arena.replay_captured(view)
                    else:
                        PagedRows.adamw(pages, gradients[:len(owners)], config)
                    reference.step()
                publication = arena.publish(write)
                arena.settle(publication)
                pin = arena.pin_exact(publication.after_refs, stream)
                for row, owner in enumerate(owners):
                    parameter = parameters[owner]
                    state = reference.state[parameter]
                    expected = (parameter, parameter.grad, state["step"].to(arena.device), state["exp_avg"], state["exp_avg_sq"])
                    for actual, value in zip(arena.snapshot(pin, pin.refs[row]), expected, strict=True):
                        torch.testing.assert_close(actual, value, rtol=1e-5, atol=1e-7)
                    current[owner] = pin.refs[row]
                arena.wait_release(arena.release_pin(pin))
            write = arena.prepare_transition(tuple(current), stream)
            if captured:
                arena.select_captured(view, write, write.after_refs)
                arena.replay_captured(view)
            else:
                pages = arena.paged_rows((write,) * 2, write.after_refs, fields, ())
                PagedRows.adamw(pages, gradients, config)
            arena.abort(write)
            if not captured:
                with pytest.raises(RuntimeError, match="consumed"):
                    PagedRows.adamw(pages, gradients, config)
            assert tuple(arena.latest(ref.lineage_id) for ref in current) == tuple(current)
            for ref, old in zip(initial.refs, originals, strict=True):
                for actual, expected in zip(arena.snapshot(initial, ref), old, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            arena.wait_release(arena.release_pin(initial))
            arena.retire(tuple(current))
            arena.collect_completed(wait=True)
            assert arena.free_pages == 6 and arena.active_lineages == 0


def _cuda_ipc_worker(control, read_channel, write_channel, wire):
    try:
        plan = decode_boundary(wire, PlanBundle)
        stream = torch.cuda.Stream(device=torch.device("cuda", 0))
        with torch.cuda.stream(stream), patch("pllm.execution.ptts.page_bundle.map_cpu_storage",
            side_effect=AssertionError("CUDA page transport must not stage through CPU storage")):
            read = PageBundleArena.receive_field(plan, read_channel, stream=stream)
            write = PageBundleArena.receive_field(plan, write_channel, access=PageAccess.WRITE, stream=stream)
            assert read.tensor.is_cuda and write.tensor.is_cuda
            assert read.descriptor.cuda_storage.handle == write.descriptor.cuda_storage.handle
            assert read.descriptor.storage_offset != write.descriptor.storage_offset
            storage = read.descriptor.cuda_storage
            for invalid in (replace(storage, device_uuid="different GPU"), replace(storage, size_bytes=storage.size_bytes + 4),
                            replace(storage, handle="")):
                with pytest.raises(ValueError):
                    invalid.map(storage.size_bytes, read.tensor.dtype, read.descriptor.storage_offset,
                        tuple(read.tensor.shape), stream)
            control.send_bytes(b"prepared")
            assert control.recv_bytes() == b"run"
            expected = torch.arange(read.tensor.numel(), device=stream.device, dtype=read.tensor.dtype)
            torch.testing.assert_close(read.tensor, expected)
            torch.cuda._sleep(100_000_000)
            write.tensor.copy_(read.tensor + 17)
            done = torch.cuda.Event()
            done.record(stream)
            assert not done.query()
            control.send_bytes(b"queued")
            assert control.recv_bytes() == b"close"
            write.close()
            assert done.query()
            read.close()
            control.send_bytes(b"completed")
        assert control.recv_bytes() == b"exit"
    except Exception as exc:
        control.send_bytes(f"{type(exc).__name__}: {exc}".encode())
        raise
    finally:
        control.close()
        read_channel.close()
        write_channel.close()


def _cuda_ipc_lifecycle():
    context = mp.get_context("spawn")
    plan = _plan(shared=True)
    for outcome in ("commit", "abort"):
        arena = PageBundleArena(plan)
        stream = torch.cuda.Stream(device=arena.device)
        initial = arena.prepare_admission(("request_distiller",), stream)
        with torch.cuda.stream(stream):
            field = arena.write_field(initial, 0, LayoutFieldId(0))
            expected = torch.arange(field.numel(), device=arena.device, dtype=field.dtype)
            field.copy_(expected)
        initial = arena.publish(initial)
        arena.settle(initial)
        ref, = initial.after_refs
        pin = arena.pin_exact((ref,), stream)
        observer = arena.pin_exact((ref,), stream)
        write = arena.prepare_transition((ref,), stream, fields=((LayoutFieldId(0),),))
        control, child_control = context.Pipe()
        read_channel, child_read = context.Pipe()
        write_channel, child_write = context.Pipe()
        process = context.Process(target=_cuda_ipc_worker,
            args=(child_control, child_read, child_write, encode_boundary(plan)))
        process.start()
        child_control.close()
        child_read.close()
        child_write.close()
        try:
            descriptor = arena.send_field(pin, ref, LayoutFieldId(0), read_channel, process.pid)
            arena.send_write_field(write, 0, LayoutFieldId(0), write_channel, process.pid)
            assert descriptor.ref is ref and descriptor.cuda_storage is not None
            assert control.poll(20) and control.recv_bytes() == b"prepared"
            with pytest.raises(RuntimeError, match="exclusively"):
                arena.write_field(write, 0, LayoutFieldId(0))
            released = arena.release_pin(pin)
            assert not arena.release_complete(released)
            control.send_bytes(b"run")
            assert control.poll(20) and control.recv_bytes() == b"queued"
            with pytest.raises(RuntimeError, match="quorum"):
                arena.publish(write)
            control.send_bytes(b"close")
            assert control.poll(20) and control.recv_bytes() == b"completed"
            if outcome == "commit":
                publication = arena.publish(write)
                arena.settle(publication)
                latest, = publication.after_refs
                current = arena.pin_exact((latest,), stream)
                torch.testing.assert_close(arena.snapshot(current, latest)[0], expected + 17)
                arena.wait_release(arena.release_pin(current))
            else:
                arena.abort(write)
                latest = arena.latest(ref.lineage_id)
                assert latest == ref
            torch.testing.assert_close(arena.snapshot(observer, ref)[0], expected)
            arena.wait_release(released)
            arena.wait_release(arena.release_pin(observer))
            arena.retire((latest,))
            arena.collect_completed(wait=True)
            assert arena.active_lineages == arena.pending_pin_releases == 0 and arena.free_pages == 6
            control.send_bytes(b"exit")
            process.join(10)
            assert process.exitcode == 0
        finally:
            control.close()
            read_channel.close()
            write_channel.close()
            if process.is_alive():
                process.terminate()
                process.join(10)
            process.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_graph_pagebundle_keeps_stable_addresses_across_versions() -> None:
    arena = PageBundleArena(_plan())
    assert all(item.capture_stable for item in arena.plan.state_physical.representations)
    stream = torch.cuda.Stream(device=arena.device)
    fields = (LayoutFieldId(0), LayoutFieldId(2))
    write = arena.prepare_admission(("request_distiller",), stream)
    with torch.cuda.stream(stream):
        initial = torch.arange(arena.write_field(write, 0, fields[0]).numel(), device=arena.device,
            dtype=torch.float32).reshape_as(arena.write_field(write, 0, fields[0]))
        arena.write_field(write, 0, fields[0]).copy_(initial)
        arena.write_field(write, 0, fields[1]).fill_(1)
    admitted = arena.publish(write)
    arena.settle(admitted)
    current, = admitted.after_refs
    original = arena.pin_exact((current,), stream)
    read = arena.capture_kernel(original, (current,), fields, (),
        lambda values, inputs: (values[0][0] * values[1][0] + 3,))
    result, = read.results
    indices, buffers, graph = read.indices, read.buffers, read.graph
    stable = (*arena.pool_addresses, *read.addresses, result.data_ptr())
    arena.replay_captured(read)
    stream.synchronize()
    torch.testing.assert_close(result, initial + 3)
    with pytest.raises(RuntimeError, match="private PageWrite"):
        arena.write_captured(read)
    with pytest.raises(RuntimeError, match="issued"):
        arena.select_captured(read, replace(original), (current,))
    for invalid in ((), (current, current)):
        with pytest.raises(ValueError, match="distinct nonempty"):
            arena.select_captured(read, original, invalid)
    with pytest.raises(ValueError, match="alias"):
        arena.capture_fields(original, (current,), fields, indices,
            (arena.borrow_field(original, current, fields[0]).unsqueeze(0), buffers[1]), graph)
    with pytest.raises(ValueError, match="address"):
        arena.select_captured(replace(read, indices=indices.clone()), original, (current,))
    other_stream = torch.cuda.Stream(device=arena.device)
    previous_done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        result.zero_()
        torch.cuda._sleep(100_000_000)
    arena.replay_captured(read)
    previous_done.record(stream)
    assert not previous_done.query()
    foreign = arena.pin_exact((current,), other_stream)
    arena.select_captured(read, foreign, (current,))
    arena.replay_captured(read)
    arena.wait_release(arena.release_pin(foreign))
    assert previous_done.query()
    torch.testing.assert_close(result, initial + 3)
    arena.select_captured(read, original, (current,))

    # The same captured gather switches exact versions; unmodified scalar fields share a slot.
    for iteration in range(1, 5):
        update = arena.prepare_transition((current,), stream, fields=((fields[0],),))
        with torch.cuda.stream(stream):
            arena.write_field(update, 0, fields[0]).add_(2)
        publication = arena.publish(update)
        arena.settle(publication)
        current, = publication.after_refs
        pin = arena.pin_exact((current,), stream)
        arena.select_captured(read, pin, (current,))
        arena.replay_captured(read)
        arena.wait_release(arena.release_pin(pin))
        torch.testing.assert_close(result, initial + 2 * iteration + 3)
        with pytest.raises(RuntimeError, match="no new reads"):
            arena.replay_captured(read)
        assert stable == (*arena.pool_addresses, *read.addresses, result.data_ptr())
        with torch.cuda.stream(stream):
            torch.testing.assert_close(arena.borrow_field(original, admitted.after_refs[0], fields[0]), initial)
    arena.wait_release(arena.release_pin(original))

    # Captured writes only target original private pages; publication stays outside the graph.
    update = arena.prepare_transition((current,), stream, fields=(fields,))
    def increment(values, inputs):
        values[0].add_(5)
        values[1].add_(1)
        return ()
    private = arena.capture_kernel(update, update.after_refs, fields, (), increment)
    abandoned_done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
    arena.replay_captured(private)
    abandoned_done.record(stream)
    assert not abandoned_done.query()
    arena.abort(update)
    assert abandoned_done.query() and arena.latest(current.lineage_id) == current
    arena.collect_completed(wait=True)
    for iteration in range(1, 4):
        old = arena.pin_exact((current,), stream)
        update = arena.prepare_transition((current,), stream, fields=(fields,))
        arena.select_captured(private, update, update.after_refs)
        arena.replay_captured(private)
        publication = arena.publish(update)
        arena.settle(publication)
        with pytest.raises(RuntimeError, match="consumed"):
            arena.replay_captured(private)
        current, = publication.after_refs
        pin = arena.pin_exact((current,), stream)
        arena.select_captured(read, pin, (current,))
        arena.replay_captured(read)
        arena.wait_release(arena.release_pin(pin))
        torch.testing.assert_close(result, (initial + 8 + 5 * iteration) * (1 + iteration) + 3)
        with torch.cuda.stream(stream):
            torch.testing.assert_close(arena.borrow_field(old, old.refs[0], fields[0]), initial + 8 + 5 * (iteration - 1))
        arena.wait_release(arena.release_pin(old))
        assert stable == (*arena.pool_addresses, *read.addresses, result.data_ptr())
    arena.retire((current,))
    arena.collect_completed(wait=True)
    assert arena.active_lineages == arena.pending_pin_releases == 0 and arena.free_pages == 6
