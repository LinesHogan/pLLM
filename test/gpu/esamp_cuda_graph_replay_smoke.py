"""Evidence 37: CUDA graph replay preserves canonical ESamp training."""

from __future__ import annotations


from pllm.execution.ptts.page_bundle_planning import compile_page_bundle_plan
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import pytest

from pllm.kernels import low_rank_gated
from pllm.kernels.paged_linear import TensorRows
from pllm.kernels.adamw import apply_adamw
from pllm.execution.scheduler import compile_schedule
from pllm.execution.trace import TracePhase, TraceSelection

from pllm.execution.state_execution import StateExecution
from pllm.authoring import TensorStepInput
from pllm.execution.tensor_binding import TensorFrame
from pllm.usage import TensorOutput
from pllm.workloads.esamp_request import (
    ADAM_FIELDS,
)
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.workloads.esamp_request import (
    ESampRequestWorkload,
)
from pllm.workloads.esamp_request import (
    ESampRequestConfig,
)
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
)
from pllm.contracts.packed_rows import (
    PackedPhaseRow,
    PackedPhaseRowBatch,
    PackedSamplerRowBatch,
    PackedSamplerRowLink,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch
from pllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    SchedulerTimeline,
    TaskResourceFact,
    TimelineAnchor,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId
from test.backend_facts import backend_facts


def _captured_training(workload, facts, plan) -> list[float]:
    arena = PageBundleArena(plan)
    execution = workload.bind_math(facts, arena.plan).bind(arena, next(AnchorId(item.ordinal) for item in arena.plan.timeline.anchors if item.symbol == "sampler.pre_draw"))
    stream = torch.cuda.Stream(device=arena.device)
    identities = tuple(RequestIdentity(f"captured-training-{index}", index, 0) for index in range(2))
    current = execution.admit(identities, stream)
    original = arena.pin_exact(current, stream)
    config = workload.optimizer
    binding = ADAM_FIELDS
    layout = low_rank_gated.LowRankGatedLayout(4, 2, True)
    fields = tuple(LayoutFieldId(index) for index in range(5))
    with torch.cuda.stream(stream):
        initial = tuple(arena.borrow_field(original, ref, fields[0]).clone() for ref in current)
        reference = {ref.lineage_id: value.clone().requires_grad_(True)
            for ref, value in zip(current, initial, strict=True)}
        source = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2],
            [0.1, 0.3, -0.2, 0.8]], device=arena.device)
        target = source.flip(1).contiguous()
        rows = torch.tensor([0, 1, 0], dtype=torch.long, device=arena.device)
    optimizer = torch.optim.AdamW(tuple(reference.values()), lr=config.lr, betas=config.betas,
        eps=config.eps, weight_decay=config.weight_decay)
    calls = [0, 0]

    def loss_kernel(values, inputs):
        calls[0] += 1
        return low_rank_gated.low_rank_gated_loss_and_gradients(
            inputs[0], inputs[1], TensorRows(values[0], inputs[2]), layout)

    def update_kernel(values, inputs):
        calls[1] += 1
        apply_adamw(tuple(zip(*(value.unbind() for value in values), strict=True)),
            binding, config, tuple(inputs[0].unbind()))
        return ()

    read = arena.capture_kernel(original, current, (fields[0],), (source, target, rows), loss_kernel)
    arena.replay_captured(read)
    write = arena.prepare_transition(current, stream)
    update = arena.capture_kernel(write, write.after_refs, fields, (read.results[1],), update_kernel)
    resident = {"loss": read.resident_bytes(), "train": update.resident_bytes()}
    assert all(size > 0 for size in resident.values())
    timeline = replace(plan.timeline, task_resources=tuple(
        TaskResourceFact(task.symbol, 0, resident.get(task.symbol, 0)) for task in plan.semantic_graph.tasks))
    baseline = compile_schedule(plan.semantic_graph, plan.timeline, state=plan.state_physical)
    measured = compile_schedule(plan.semantic_graph, timeline, state=plan.state_physical)
    assert measured.peak_memory_bytes == baseline.peak_memory_bytes + sum(resident.values())
    print({"captured_resident_bytes": resident, "aer_resident_increment": sum(resident.values())})
    cold_calls = tuple(calls)
    stable = (*arena.pool_addresses, *read.addresses, *update.addresses,
        *(value.data_ptr() for value in read.results))
    # Cold warmup/capture must not update private or published optimizer state.
    with torch.cuda.stream(stream):
        for index, value in enumerate(initial):
            assert arena.write_field(write, index, fields[2]).item() == 0
            torch.testing.assert_close(arena.write_field(write, index, fields[0]), value)
    arena.replay_captured(update)
    arena.abort(write)
    assert all(arena.latest(ref.lineage_id) == ref for ref in current)
    latest = {ref.lineage_id: ref for ref in current}
    losses = []
    for iteration, state_count in enumerate((2, 1, 2)):
        current = tuple(latest.values())
        if iteration != 1:
            current = tuple(reversed(current))
        current = current[:state_count]
        pin = arena.pin_exact(current, stream)
        with torch.cuda.stream(stream):
            rows.copy_(torch.tensor([0, 1, 0] if state_count == 2 else [0, 0, -1], device=arena.device))
            source[2].copy_(torch.tensor([0.1, 0.3, -0.2, 0.8] if state_count == 2 else [float("nan")] * 4,
                device=arena.device))
            target.add_(0.03)
            optimizer.zero_grad()
            expected_losses = []
            for index, ref in enumerate(current):
                selected = source[rows == index]
                up, gate, down, bias, scale, shift = reference[ref.lineage_id].split((8, 8, 8, 2, 4, 4))
                residual = ((selected @ up.reshape(4, 2)) * F.silu(selected @ gate.reshape(4, 2) + bias)) @ down.reshape(2, 4)
                expected_loss = (F.layer_norm(selected + residual, (4,)) * scale + shift - target[rows == index]).square().mean()
                expected_loss.backward()
                expected_losses.append(expected_loss.detach())
        arena.select_captured(read, pin, current)
        write = arena.prepare_transition(current, stream)
        arena.select_captured(update, write, write.after_refs)
        with patch.object(low_rank_gated, "low_rank_gated_loss_and_gradients", side_effect=AssertionError("eager loss")), \
             patch(__name__ + ".apply_adamw", side_effect=AssertionError("eager optimizer")):
            arena.replay_captured(read)
            arena.replay_captured(update)
        publication = arena.publish(write)
        arena.settle(publication)
        current = publication.after_refs
        latest.update((ref.lineage_id, ref) for ref in current)
        with torch.cuda.stream(stream):
            torch.testing.assert_close(read.results[0][:state_count], torch.stack(expected_losses))
            assert torch.count_nonzero(read.results[0][state_count:]) == 0
            assert torch.count_nonzero(read.results[1][state_count:]) == 0
            for index, ref in enumerate(current):
                torch.testing.assert_close(read.results[1][index], reference[ref.lineage_id].grad,
                    rtol=1e-5, atol=1e-7)
                # Separate float32 GEMM rounding from AdamW's small-gradient amplification.
                reference[ref.lineage_id].grad = read.results[1][index].clone()
            optimizer.step()
            updated = arena.pin_exact(current, stream)
            for ref in current:
                parameter = reference[ref.lineage_id]
                actual = arena.snapshot(updated, ref)
                state = optimizer.state[parameter]
                for field, (result, expected) in enumerate(zip(actual, (parameter, parameter.grad, state["step"],
                        state["exp_avg"], state["exp_avg_sq"]), strict=True)):
                    torch.testing.assert_close(result, expected.to(result.device),
                        msg=lambda message: f"iteration={iteration}, lineage={ref.lineage_id}, field={field}: {message}")
                assert actual[2].item() == state["step"].item()
            for ref in latest.values():
                if ref not in current:
                    unchanged = arena.pin_exact((ref,), stream)
                    torch.testing.assert_close(arena.borrow_field(unchanged, ref, fields[0]), reference[ref.lineage_id])
                    arena.wait_release(arena.release_pin(unchanged))
            for ref, value in zip(original.refs, initial, strict=True):
                torch.testing.assert_close(arena.borrow_field(original, ref, fields[0]), value)
                assert not torch.equal(reference[ref.lineage_id].detach(), value)
            losses.extend(read.results[0][:state_count].tolist())
        arena.wait_release(arena.release_pin(updated))
        arena.wait_release(arena.release_pin(pin))
        assert tuple(calls) == cold_calls
        assert stable == (*arena.pool_addresses, *read.addresses, *update.addresses,
            *(value.data_ptr() for value in read.results))
    assert all(loss > 0 for loss in losses)
    arena.wait_release(arena.release_pin(original))
    arena.retire(tuple(latest.values()))
    arena.collect_completed(wait=True)
    assert arena.active_lineages == arena.pending_pin_releases == 0
    return losses


def _timeline() -> SchedulerTimeline:
    return SchedulerTimeline(
        anchors=(
            TimelineAnchor(AnchorSymbol("model.execute"), 0),
            TimelineAnchor(AnchorSymbol("sampler.pre_draw"), 1),
            TimelineAnchor(AnchorSymbol("token.commit"), 2),
            TimelineAnchor(AnchorSymbol("next.prepare"), 3),
        ),
        capabilities=(
            "compute/cuda@1",
            "state/dense_page_cuda@1",
        ),
    )


def _rows(
    step_id: int,
    identities: tuple[RequestIdentity, RequestIdentity],
) -> PackedPhaseRowBatch:
    return PackedPhaseRowBatch(
        1,
        step_id,
        2,
        (
            PackedPhaseRow(
                0,
                0,
                step_id,
                HiddenPhase.DECODE,
                identities[0],
            ),
            PackedPhaseRow(
                1,
                1,
                step_id,
                HiddenPhase.DECODE,
                identities[1],
            ),
        ),
    )


def _step_input(
    *,
    step_id: int,
    identities: tuple[RequestIdentity, RequestIdentity],
    hidden_bindings: tuple[PackedHiddenResourceBinding, ...],
    source: torch.Tensor,
    target: torch.Tensor,
    producer_stream: torch.cuda.Stream,
) -> TensorStepInput:
    rows = _rows(step_id, identities)
    hidden_by_resource = {
        hidden_bindings[0]: source,
        hidden_bindings[1]: target,
    }
    return TensorStepInput(
        rows,
        PackedSamplerRowBatch(
            rows,
            (
                PackedSamplerRowLink(0, 0),
                PackedSamplerRowLink(1, 1),
            ),
        ),
        PackedHiddenSnapshotBatch(
            rows,
            tuple(
                PackedHiddenSnapshot(resource, hidden_by_resource[resource])
                for resource in sorted(hidden_bindings, key=lambda item: item.locator_key)
            ),
        ),
        hidden_readiness=CudaStreamReadiness(producer_stream),
    )


def _borrowed_hidden_lifetime() -> None:
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    bindings = tuple(PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX,
        HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]") for layer in (0, -1))
    identities = (RequestIdentity("borrow-a", 0, 0), RequestIdentity("borrow-b", 0, 1))
    with torch.cuda.stream(producer):
        value = _step_input(step_id=0, identities=identities, hidden_bindings=bindings,
            source=torch.full((2, 32768), 3.0, device="cuda"),
            target=torch.full((2, 32768), 7.0, device="cuda"), producer_stream=producer)
    with torch.cuda.stream(consumer):
        hidden = value.hidden_batch(consumer)
        source = hidden.hidden_for(bindings[0])
        torch.cuda._sleep(500_000_000)
        observed = source.clone()
    del value, hidden, source
    # Reuse the producer's freed blocks while the consumer is still reading.
    with torch.cuda.stream(producer):
        pressure = [torch.full((2, 32768), -1.0, device="cuda") for _ in range(32)]
    consumer.synchronize()
    producer.synchronize()
    torch.testing.assert_close(observed, torch.full_like(observed, 3.0))
    del pressure


def _run_step(
    *,
    execution: StateExecution[TensorStepInput, TensorFrame, TensorOutput],
    step_id: int,
    identities: tuple[RequestIdentity, RequestIdentity],
    hidden_bindings: tuple[PackedHiddenResourceBinding, ...],
    source: torch.Tensor,
    target: torch.Tensor,
    execution_stream: torch.cuda.Stream,
    sampler_stream: torch.cuda.Stream,
) -> tuple[float, ...]:
    for last_anchor in (0, 1):
        cancelled = execution.start_step(
            _step_input(
                step_id=step_id, identities=identities, hidden_bindings=hidden_bindings,
                source=source, target=target, producer_stream=execution_stream,
            ),
            execution_stream,
        )
        for anchor in range(last_anchor + 1):
            execution.execute_anchor(cancelled, AnchorId(anchor))
        execution.cancel_step(cancelled)
    handle = execution.start_step(
        _step_input(
            step_id=step_id,
            identities=identities,
            hidden_bindings=hidden_bindings,
            source=source,
            target=target,
            producer_stream=execution_stream,
        ),
        execution_stream,
    )
    prediction_done = torch.cuda.Event()
    def delayed_guidance(rows: PackedSamplerRowBatch, prediction: torch.Tensor,
                         affected: tuple[int, ...], beta: float) -> PackedSamplerGuidanceBatch:
        if affected:
            torch.cuda._sleep(500_000_000)
            prediction.add_(1)
        prediction_done.record(execution_stream)
        return PackedSamplerGuidanceBatch(rows, prediction, affected, beta)

    with patch("pllm.workloads.esamp_request.PackedSamplerGuidanceBatch", delayed_guidance):
        execution.execute_anchor(handle, AnchorId(0))
    if step_id and not prediction_done.query():
        with patch.object(torch.cuda.Event, "synchronize", side_effect=AssertionError("trace synchronized CUDA")), \
             patch.object(torch.cuda, "synchronize", side_effect=AssertionError("trace synchronized CUDA")):
            assert execution.trace.snapshot().pending_device_timings > 0
    guidance = execution.sampler_guidance(handle, sampler_stream)
    if guidance.affected_row_ids and guidance.values.data_ptr() == 0:
        raise RuntimeError("canonical ESamp guidance did not materialize on CUDA")
    with torch.cuda.stream(sampler_stream):
        consumed = guidance.values.clone()
    sampler_stream.synchronize()
    if not prediction_done.query():
        execution_stream.synchronize()
        raise RuntimeError("sampler read completed before prediction readiness")
    torch.testing.assert_close(consumed, guidance.values)
    execution.execute_anchor(handle, AnchorId(1))
    execution.complete_sampler(handle, sampler_stream)
    execution.execute_anchor(handle, AnchorId(2))
    result = execution.finish_step(handle)
    execution_stream.synchronize()
    sampler_stream.synchronize()
    assert result.loss is not None
    return tuple(float(loss.item()) for loss in result.loss)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical ESamp CUDA graph replay smoke requires CUDA")
    _borrowed_hidden_lifetime()
    device = torch.device("cuda", 0)
    hidden_bindings = tuple(PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX,
        HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]") for layer in (0, -1))
    workload = ESampRequestWorkload(
        ESampRequestConfig(
            rank=2,
            intervention=True,
            intervention_beta=0.2,
            learning_rate=0.05,
        )
    )
    facts = backend_facts(4, hidden_bindings, _timeline(), PageBundleProviderFacts(
        "canonical.cuda_dense_page@1", device_kind=StateDeviceKind.CUDA, device_ordinal=0,
        page_capacity=6))
    declaration = workload.declaration(facts)
    plan = compile_page_bundle_plan((declaration.graph, declaration.layouts), facts.timeline, facts.providers,
        facts.max_admitted_lineages, facts.state_memory_budget_bytes, device)
    captured_losses = _captured_training(workload, facts, plan)
    from pllm.config import ExecutionOptions
    facts = workload.specialize(facts, ExecutionOptions(capture=True, capture_memory_bytes=4 * 1024 * 1024))
    timeline = facts.timeline
    declaration = workload.declaration(facts)
    plan = compile_page_bundle_plan((declaration.graph, declaration.layouts), timeline, facts.providers,
        facts.max_admitted_lineages, facts.state_memory_budget_bytes, device)
    loss_limit = next(item for item in timeline.task_resources if item.task == "loss")
    for loss_fact in (replace(loss_limit, resident_bytes=32), replace(loss_limit, resident_bytes=4096),
            replace(loss_limit, input_shapes=((2, 5), (2, 4), (2, 34), (2,)))):
        invalid_facts = replace(facts, timeline=replace(timeline, task_resources=tuple(
            loss_fact if item.task == "loss" else item for item in timeline.task_resources)))
        declaration = workload.declaration(invalid_facts)
        invalid = compile_page_bundle_plan((declaration.graph, declaration.layouts), invalid_facts.timeline,
            facts.providers, facts.max_admitted_lineages, facts.state_memory_budget_bytes, device)
        rejected = PageBundleArena(invalid)
        with pytest.raises(ValueError, match="resident bytes|shapes disagree"):
            workload.bind_math(invalid_facts, rejected.plan).bind(rejected, next(AnchorId(item.ordinal) for item in rejected.plan.timeline.anchors if item.symbol == "sampler.pre_draw"))
        assert rejected.active_lineages == rejected.pending_pin_releases == 0 and rejected.free_pages == 6
    execution = workload.bind_math(facts, plan).bind(PageBundleArena(plan), next(AnchorId(item.ordinal) for item in plan.timeline.anchors if item.symbol == "sampler.pre_draw"))
    execution.trace.profile_tasks(tuple(task.task_id for task in execution.plan.tasks))
    assert execution.arena.active_lineages == execution.arena.pending_pin_releases == 0
    assert sum("cuda_graph" in task.implementation_id for task in execution.plan.tasks) == 2
    if execution.device.type != "cuda":
        raise RuntimeError("canonical ESamp did not compile CUDA execution")
    if execution.plan.fallback_count != 0:
        raise RuntimeError("canonical ESamp CUDA plan used fallback tasks")

    execution_stream = torch.cuda.Stream(device=device)
    sampler_stream = torch.cuda.Stream(device=device)
    identities = (
        RequestIdentity("cuda-graph-a", 0, 0),
        RequestIdentity("cuda-graph-b", 1, 0),
    )
    with torch.cuda.stream(execution_stream):
        source = torch.tensor(
            ((0.2, -0.1, 0.4, 0.3), (-0.5, 0.7, 0.1, -0.2)),
            device=device,
            dtype=torch.float32,
        )
        target = torch.tensor(
            ((0.6, 0.0, -0.3, 0.2), (0.4, -0.2, 0.8, 0.1)),
            device=device,
            dtype=torch.float32,
        )
    execution.admit(identities, execution_stream)
    execution_stream.synchronize()

    hidden_addresses = (source.data_ptr(), target.data_ptr())
    state_addresses = execution.arena.pool_addresses
    all_losses: list[float] = []
    for step_id in range(2):
        execution.trace.configure(TraceSelection(tasks=tuple(task.task_id for task in execution.plan.tasks
            if plan.semantic_graph.tasks[task.task_id].symbol == "loss"),
            tensors=step_id == 1, gradients=step_id == 1, kernel_calls=int(step_id == 0)))
        if (
            hidden_addresses != (source.data_ptr(), target.data_ptr())
            or state_addresses != execution.arena.pool_addresses
        ):
            raise RuntimeError("CUDA graph replay changed canonical buffer addresses")
        with patch("pllm.workloads.esamp_request.low_rank_gated_loss_and_gradients",
                side_effect=AssertionError("eager loss on captured task")), \
             patch("pllm.workloads.esamp_request.apply_adamw",
                side_effect=AssertionError("eager optimizer on captured task")):
            all_losses.extend(_run_step(
                execution=execution,
                step_id=step_id,
                identities=identities,
                hidden_bindings=hidden_bindings,
                source=source,
                target=target,
                execution_stream=execution_stream,
                sampler_stream=sampler_stream,
            ))
        execution.arena.collect_completed()

    diagnostics = execution.diagnostics()
    if (
        len(all_losses) != 4
        or not all(loss > 0.0 for loss in all_losses)
        or diagnostics.loss_count != 4
        or diagnostics.update_count != 4
        or diagnostics.completed_sampler_pins != 2
        or diagnostics.active_lineages != 2
        or execution.plan.fallback_count != 0
    ):
        raise RuntimeError("canonical ESamp CUDA graph replay evidence is incomplete")
    cancelled = execution.start_step(
        _step_input(
            step_id=2, identities=identities, hidden_bindings=hidden_bindings,
            source=source, target=target, producer_stream=execution_stream,
        ),
        execution_stream,
    )
    execution.execute_anchor(cancelled, AnchorId(0))
    guidance = execution.sampler_guidance(cancelled, sampler_stream)
    sampler_done = torch.cuda.Event()
    with torch.cuda.stream(sampler_stream):
        torch.cuda._sleep(500_000_000)
        consumed = guidance.values.clone()
        sampler_done.record()
    if sampler_done.query():
        raise RuntimeError("cancellation evidence requires an unfinished sampler read")
    execution.retire(identities)
    if not sampler_done.query():
        sampler_stream.synchronize()
        raise RuntimeError("retirement released state before the cancelled sampler completed")
    torch.testing.assert_close(consumed, guidance.values)
    execution.arena.collect_completed()
    retired = execution.diagnostics()
    if retired.active_lineages != 0 or retired.pending_state_releases != 0:
        raise RuntimeError("cancelled request retirement left live state or release receipts")
    trace = execution.trace.snapshot()
    assert trace.schema_version == 5 and trace.pending_device_timings == 0
    assert len(trace.device_task_times()) == len(execution.plan.tasks)
    assert all(count > 0 and elapsed > 0 for count, elapsed in trace.device_task_times().values())
    kernels = [event for event in trace.events if event.phase is TracePhase.KERNEL]
    assert kernels and all(event.kernel_name and event.device_duration_ns > 0 for event in kernels)
    gradients = [event for event in trace.events if event.phase is TracePhase.GRADIENT]
    assert len(gradients) == 2 and {event.row.identity for event in gradients} == set(identities), (len(gradients), trace.dropped)
    assert all(torch.isfinite(event.tensor()).all() and event.tensor().abs().sum() > 0 for event in gradients)
    execution.trace.configure(TraceSelection())
    execution.trace.profile_tasks(())
    for _ in range(trace.capacity):
        execution.trace.record(TracePhase.RETIRE, 0)
    evicted = execution.trace.snapshot()
    assert evicted.dropped > 0 and evicted.pending_device_timings == 0 and not evicted.device_task_times()
    print(
        {
            "captured_training_losses": captured_losses,
            "losses": all_losses,
            "loss_count": diagnostics.loss_count,
            "update_count": diagnostics.update_count,
            "fallback_count": execution.plan.fallback_count,
            "hidden_addresses": hidden_addresses,
            "state_addresses": state_addresses,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
