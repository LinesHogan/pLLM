"""Evidence 43: archived shared weighted CE/SGD, original state and sampler boundaries."""

from contextlib import ExitStack, nullcontext
from dataclasses import replace
from typing import Annotated
import os

import pytest
import torch
import pllm
from torch.nn import functional as F

from pllm.authoring import AuxiliaryInput, BackendCompileFacts, TensorStepInput, WeightedCEBatch, map_auxiliary
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.config import ExecutionOptions
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.trace import TracePhase, TraceSelection
from pllm.workloads.lora import OutputLoRAConfig, OutputLoRAWorkload
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentHiddenResource, AttachmentTopology, BackendPlacementProbe
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import AnchorId, StateScope, TaskId, TaskResultInputSource
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from test.backend_facts import assert_capture_preflight, assert_auxiliary_preflight


def test_weighted_ce_shared_update_matches_sgd_and_preserves_original_pins(monkeypatch) -> None:
    binding = PackedHiddenResourceBinding("-1", HiddenLayerKind.INDEX, HiddenSite.MODEL_OUTPUT, "model.norm")
    facts = BackendCompileFacts(BackendPlacementProbe("test.sampler@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CPU, 0, 1, 1,
        TensorDType.FLOAT32, 4, 8, (AttachmentHiddenResource(binding, HiddenPhase.DECODE),), ()),
        RuntimeFacts(torch.device("cpu")).timeline(), (PageBundleProviderFacts("test.page@1", 3),),
        3, 65536, 7, vocab_size=7)
    config = OutputLoRAConfig(rank=2, alpha=3.0, learning_rate=0.3, steps=3, seed=7)
    workload = OutputLoRAWorkload(config)
    fixed = ExecutionOptions(training_rows=4, state_memory_bytes=32768)
    fixed_plan = workload.plan_backend(workload.specialize(facts, fixed)).inputs.plan
    assert fixed_plan == workload.plan_backend(workload.specialize(
        replace(facts, state_memory_budget_bytes=131072), fixed)).inputs.plan
    assert fixed_plan.state_request.memory_budget_bytes == 32768
    bounded = replace(fixed, memory_budget_bytes=fixed_plan.schedule_decision.peak_memory_bytes)
    bounded_facts = workload.specialize(facts, bounded)
    assert bounded_facts.timeline.memory_budget_bytes == bounded.memory_budget_bytes
    assert workload.plan_backend(bounded_facts).inputs.plan.schedule_decision.peak_memory_bytes == bounded.memory_budget_bytes
    assert decode_boundary(encode_boundary(bounded), ExecutionOptions) == bounded
    for budget in (0, fixed_plan.schedule_decision.peak_memory_bytes - 1):
        with pytest.raises(ValueError):
            workload.plan_backend(workload.specialize(facts, replace(bounded, memory_budget_bytes=budget)))
    with pytest.raises(ValueError, match="memory limits"):
        replace(bounded, memory_budget_bytes=-1)
    # Resource declarations can tighten limits, but cannot drop or loosen an original/user limit.
    with monkeypatch.context() as changed:
        for original, declared, requested, expected in ((None, None, 100, 100), (50, None, 100, 50),
                (100, 50, None, 50), (0, 100, None, 0), (None, None, None, None)):
            changed.setattr(OutputLoRAWorkload, "resource_facts", lambda self, facts, options:
                replace(bounded_facts.timeline, memory_budget_bytes=declared))
            constrained = workload.specialize(replace(facts, timeline=replace(facts.timeline,
                memory_budget_bytes=original)), replace(fixed, memory_budget_bytes=requested))
            assert constrained.timeline.memory_budget_bytes == expected
        for missing in ("task_resources", "value_resources"):
            changed.setattr(OutputLoRAWorkload, "resource_facts", lambda self, facts, options:
                replace(bounded_facts.timeline, **{missing: ()}))
            with pytest.raises(ValueError, match="no joint state representation and AER placement"):
                workload.plan_backend(workload.specialize(facts, bounded))
    with pytest.raises(ValueError, match="exceeds available backend memory"):
        workload.specialize(replace(facts, state_memory_budget_bytes=16384), fixed)
    with pytest.raises(ValueError):
        workload.plan_backend(workload.specialize(facts, replace(fixed, state_memory_bytes=1)))
    for paged in (False, True):
        assert_capture_preflight(OutputLoRAWorkload(replace(config, paged=paged)), facts, monkeypatch)
    facts = workload.specialize(facts, ExecutionOptions(training_rows=4))
    signed = workload.plan_backend(facts)
    for rows, vocab in ((5, 7), (4, 8)):
        assert_auxiliary_preflight(signed, WeightedCEBatch(torch.empty((rows, 4), device="meta"),
            torch.empty((rows, vocab), device="meta"), torch.empty(rows, device="meta", dtype=torch.int64), torch.empty(rows, device="meta")))
    preparation, prediction, training = facts.timeline.task_resources
    assert prediction.task == "predict" and prediction.input_shapes == ((3, 4),)
    assert prediction.workspace_bytes == 932 and training.workspace_bytes == 1096
    with pytest.raises(ValueError, match="predict workspace"):
        workload.compile_backend(replace(facts, timeline=replace(facts.timeline,
            task_resources=(preparation, replace(prediction, workspace_bytes=931), training))))
    with pytest.raises(ValueError, match="prepare workspace"):
        workload.compile_backend(replace(facts, timeline=replace(facts.timeline,
            task_resources=(replace(preparation, workspace_bytes=0), prediction, training))))
    with pytest.raises(ValueError, match="output storage"):
        workload.compile_backend(replace(facts, timeline=replace(facts.timeline,
            value_resources=tuple(replace(item, bytes=0) for item in facts.timeline.value_resources))))
    roomy_facts = replace(facts, timeline=replace(facts.timeline,
        task_resources=(preparation, replace(prediction, workspace_bytes=4096), training)))
    roomy = workload.plan_backend(roomy_facts).inputs.plan
    assert roomy.schedule_decision.peak_memory_bytes >= roomy.state_physical.reserved_bytes + 4096
    workload.plan_backend(replace(roomy_facts, timeline=replace(roomy_facts.timeline,
        memory_budget_bytes=roomy.schedule_decision.peak_memory_bytes)))
    with pytest.raises(ValueError):
        workload.plan_backend(replace(roomy_facts, timeline=replace(roomy_facts.timeline,
            memory_budget_bytes=roomy.state_physical.reserved_bytes + 4095)))
    execution = workload.compile_backend(facts).execution
    initial, = execution.admit(())
    original = execution.snapshot(initial)
    old_pin = execution.arena.pin_exact((initial,))
    hidden = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2], [0.1, 0.3, -0.2, 0.8]])
    batch = WeightedCEBatch(torch.cat((hidden, torch.full((1, 4), float("nan")))),
        torch.linspace(-1, 1, 28).reshape(4, 7), torch.tensor([2, 4, 1, -100]), torch.tensor([1., 3., 0., 1.]))
    validate_auxiliary(batch, signed.inputs.plan, SharedTensor.describe)
    assert_auxiliary_preflight(signed, replace(batch, labels=batch.labels.float()))
    rows = PackedPhaseRowBatch(1, 0, 3, tuple(PackedPhaseRow(i, i, 0,
        HiddenPhase.DECODE, RequestIdentity(str(i), i, 0)) for i in range(3)))
    value = TensorStepInput(rows, PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(3))),
        PackedHiddenSnapshotBatch(rows, (PackedHiddenSnapshot(binding, hidden),)), auxiliary=batch)
    def constant_reset(self, parameters, generator):
        parameters["a"].fill_(0.02)
        parameters["b"].zero_()
    with monkeypatch.context() as changed:
        changed.setattr(OutputLoRAWorkload, "reset", constant_reset)
        with pytest.raises(ValueError, match="resource_facts declaration"):
            workload.specialize(facts, ExecutionOptions(training_rows=4))
        with pytest.raises(ValueError, match="resource_facts declaration"):
            workload.compile_backend(facts)
        # This constant reset stays inside the original envelope; explicitly own that claim.
        changed.setattr(OutputLoRAWorkload, "resource_facts", lambda self, facts, options: facts.timeline)
        custom = workload.compile_backend(facts).execution
        custom.admit(())
        prepared = custom.start_step(value)
        for anchor in range(4):
            custom.execute_anchor(prepared, AnchorId(anchor))
        assert torch.all(prepared.context.data().parameters[0] == 0.02)
        custom.cancel_step(prepared)
        custom.retire(())
        explicit = replace(facts, timeline=replace(facts.timeline,
            capabilities=tuple(sorted((*facts.timeline.capabilities, "compute/explicit_math@1")))))
        with pytest.raises(ValueError, match="native implementation"):
            workload.compile_backend(explicit)
    excess = replace(rows, total_rows=4, rows=(*rows.rows,
        PackedPhaseRow(3, 3, 0, HiddenPhase.DECODE, RequestIdentity("extra", 3, 0))))
    with pytest.raises(ValueError, match="prediction envelope"):
        execution.step(replace(value, rows=excess, sampler_rows=PackedSamplerRowBatch(excess,
            tuple(PackedSamplerRowLink(i, i) for i in range(4))), hidden=PackedHiddenSnapshotBatch(excess,
                (PackedHiddenSnapshot(binding, torch.cat((hidden, hidden[:1]))),))))
    assert execution.arena.latest(initial.lineage_id) == initial
    mixed = replace(excess, rows=(*rows.rows, replace(excess.rows[-1], phase=HiddenPhase.PREFILL)))
    skipped = execution.step(replace(value, rows=mixed, auxiliary=None,
        sampler_rows=PackedSamplerRowBatch(mixed, tuple(PackedSamplerRowLink(i, i) for i in range(4))),
        hidden=PackedHiddenSnapshotBatch(mixed, (PackedHiddenSnapshot(binding, torch.cat((hidden, hidden[:1]))),))))
    assert skipped.publication is None
    for wrong_binding, wrong_hidden in ((replace(binding, module_path="model.other"), hidden),
            (binding, hidden.to(torch.float64)), (binding, hidden[:, :3])):
        invalid = replace(value, hidden=PackedHiddenSnapshotBatch(rows, (PackedHiddenSnapshot(wrong_binding, wrong_hidden),)))
        with pytest.raises(ValueError, match="sampler hidden input"):
            execution.step(invalid)
        assert execution.arena.latest(initial.lineage_id) == initial
    with ExitStack() as lifetime:
        wire = map_auxiliary(batch, lambda tensor: SharedTensor.export(tensor, lifetime))
        restored = decode_boundary(encode_boundary(wire), AuxiliaryInput[SharedTensor])
        mapped = map_auxiliary(restored, lambda tensor: tensor.map(os.getpid()))
        assert isinstance(mapped, WeightedCEBatch)
        torch.testing.assert_close(mapped.hidden, batch.hidden, equal_nan=True)
    # Unequal row weights are normalized together, not averaged per request.
    a = torch.nn.Parameter(torch.randn((4, 2), generator=torch.Generator().manual_seed(7)) * 0.01)
    b = torch.nn.Parameter(torch.zeros(2, 7))
    optimizer = torch.optim.SGD((a, b), lr=config.learning_rate)
    for _ in range(config.steps):
        optimizer.zero_grad()
        losses = F.cross_entropy(batch.base_logits[:2] + hidden[:2] @ a @ b * 1.5, batch.labels[:2], reduction="none")
        ((losses[0] + 3 * losses[1]) / 4).backward()
        optimizer.step()

    train_task = next(TaskId(index) for index, task in enumerate(execution.arena.plan.semantic_graph.tasks) if task.symbol == "train")
    execution.trace.configure(TraceSelection(tasks=(train_task,), iterations=(config.steps - 1,), gradients=True))
    invocation = execution.start_step(value)
    frame = invocation.context.data()
    assert frame.batch_inputs == frame.parameters == ()
    training_task = execution.arena.plan.semantic_graph.tasks[train_task]
    assert any(port.source == TaskResultInputSource("__train_prepare", "batch") for port in training_task.inputs)
    for anchor in range(4):
        execution.execute_anchor(invocation, AnchorId(anchor))
    assert len(frame.batch_inputs) == 4 and len(frame.parameters) == 2
    execution.complete_sampler(invocation)
    for anchor in range(4, 8):
        execution.execute_anchor(invocation, AnchorId(anchor))
    assert frame.batch_inputs == frame.parameters == () and frame.parameter_gradient_fields
    result = execution.finish_step(invocation)
    publication = result.publication
    assert result.loss is not None
    torch.testing.assert_close(result.loss[0], (losses[0] + 3 * losses[1]).detach() / 4)
    gradients = [event for event in execution.trace.snapshot().events if event.phase is TracePhase.GRADIENT]
    assert [event.tensor_name for event in gradients] == ["a.grad", "b.grad"]
    for event, parameter in zip(gradients, (a, b), strict=True):
        torch.testing.assert_close(event.tensor(), parameter.grad)
    execution.trace.configure(TraceSelection())
    assert publication is not None and publication.before_refs == (initial,)
    latest, = publication.after_refs
    for actual, expected in zip(execution.snapshot(latest), (a, b), strict=True):
        torch.testing.assert_close(actual, expected)
    # Another batch resets the optimizer parameters, but absence of input preserves the head.
    second = execution.step(value).publication
    assert second is not None
    latest, = second.after_refs
    for actual, expected in zip(execution.snapshot(latest), (a, b), strict=True):
        torch.testing.assert_close(actual, expected)
    invocation = execution.start_step(replace(value, auxiliary=None))
    for anchor in range(4):
        execution.execute_anchor(invocation, AnchorId(anchor))
    guidance = execution.sampler_guidance(invocation)
    torch.testing.assert_close(guidance.values, hidden @ a @ b * 1.5)
    assert torch.count_nonzero(guidance.values) > 0
    execution.complete_sampler(invocation)
    for anchor in range(4, 8):
        execution.execute_anchor(invocation, AnchorId(anchor))
    idle = execution.finish_step(invocation)
    assert idle.publication is None and idle.loss is None and idle.guidance is guidance
    capacity = execution.arena.free_pages
    with pytest.raises(ValueError, match="at least one"):
        execution.step(replace(value, auxiliary=replace(batch, weights=torch.zeros(4))))
    def fail_publish(self, write):
        raise RuntimeError("publication failed")
    with monkeypatch.context() as fault:
        fault.setattr(PageBundleArena, "publish", fail_publish)
        with pytest.raises(RuntimeError, match="publication failed"):
            execution.step(value)
    assert execution.arena.latest(latest.lineage_id) == latest and execution.arena.free_pages == capacity
    for actual, expected in zip(execution.arena.snapshot(old_pin, initial), original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    diagnostics = execution.diagnostics()
    assert diagnostics.update_count == 2 and diagnostics.loss_count == 6 and diagnostics.trained_rows == 4
    execution.arena.wait_release(execution.arena.release_pin(old_pin))
    execution.retire(())
    assert execution.diagnostics().active_lineages == execution.diagnostics().fallback_count == 0


def test_independent_lora_bmm_and_capture_preserve_keyed_mean_and_private_publication(monkeypatch) -> None:
    # Archived batched semantics: seeded full bank, per-adapter normalized CE,
    # mean across adapters, one SGD step per iteration. No production math helper.
    binding = PackedHiddenResourceBinding("-1", HiddenLayerKind.INDEX, HiddenSite.MODEL_OUTPUT, "model.norm")
    identities = tuple(RequestIdentity(str(index), index, 0) for index in range(3))
    keys = tuple(RequestStateKey.project(identity, (RequestKeyField.PROMPT_IDX, RequestKeyField.SAMPLE_IDX)) for identity in identities)
    for device, paged, captured, explicit in ((torch.device("cpu"), False, False, False), (torch.device("cpu"), False, False, True), *(
        [(torch.device("cuda", 0), False, False, False), (torch.device("cuda", 0), False, False, True),
         (torch.device("cuda", 0), False, True, False), (torch.device("cuda", 0), True, False, False),
         (torch.device("cuda", 0), True, True, False)] if torch.cuda.is_available() else [])):
        config = OutputLoRAConfig(rank=2, alpha=3.0, learning_rate=0.2, steps=3, seed=11,
            scope=StateScope.REQUEST, paged=paged)
        cuda = device.type == "cuda"
        stream = torch.cuda.Stream(device=device) if cuda else None
        facts = BackendCompileFacts(BackendPlacementProbe("test.sampler@1", "a" * 64, "b" * 64, "c" * 64,
            AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CUDA if cuda else StateDeviceKind.CPU,
            0, 1, 1, TensorDType.FLOAT32, 4, 8, (AttachmentHiddenResource(binding, HiddenPhase.DECODE),), ()),
            RuntimeFacts(device).timeline(),
            (PageBundleProviderFacts("canonical.cuda_dense_page@1" if cuda else "test.page@1", 8,
                StateDeviceKind.CUDA if cuda else StateDeviceKind.CPU, 0),), 3, 32 * 1024**2, 11, vocab_size=7)
        workload = OutputLoRAWorkload(config)
        facts = workload.specialize(facts, ExecutionOptions(training_rows=4, capture=captured,
            capture_memory_bytes=16 * 1024**2, explicit_math=explicit))
        if paged or captured or explicit:
            with monkeypatch.context() as changed:
                changed.setitem(OutputLoRAWorkload.train.__annotations__, "hidden",
                    Annotated[torch.Tensor, pllm.Batch("base_logits")])
                with pytest.raises(ValueError, match="customized mathematics"):
                    workload.compile_backend(facts)
        execution = workload.compile_backend(facts).execution
        if captured:
            assert any("cuda_graph" in task.implementation_id for task in execution.plan.tasks)
        initial = execution.admit(identities, stream)
        old_pin = execution.arena.pin_exact(initial, stream)
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            original = tuple(execution.arena.snapshot(old_pin, ref) for ref in initial)
            hidden = torch.linspace(-0.7, 0.9, 32, device=device).reshape(2, 4, 4)
            logits = torch.linspace(-1, 1, 56, device=device).reshape(2, 4, 7)
            labels = torch.tensor([[2, 4, -100, 1], [0, 6, 3, 2]], device=device)
            weights = torch.tensor([[1., 3., 1., 0.], [2., 1., 0., 4.]], device=device)
            hidden[0, 2].fill_(float("nan"))
            batch = pllm.WeightedCEBatch.for_prompts(hidden, logits, labels, weights, (1, 0))
            assert batch.keys == (keys[1], keys[0]) and batch.hidden is hidden and batch.weights is weights
            for invalid in ((), (0, 0), (-1,)):
                with pytest.raises(ValueError):
                    pllm.WeightedCEBatch.for_prompts(hidden, logits, labels, weights, invalid)
            rows = PackedPhaseRowBatch(1, 0, 3, tuple(PackedPhaseRow(i, i, 0, HiddenPhase.DECODE, identity)
                for i, identity in enumerate(identities)))
            value = TensorStepInput(rows, PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(3))),
                PackedHiddenSnapshotBatch(rows, (PackedHiddenSnapshot(binding, torch.ones((3, 4), device=device)),)),
                auxiliary=batch, hidden_readiness=CudaStreamReadiness(stream) if stream is not None else None)
            seed = torch.randn((2, 4, 2), device=device, generator=torch.Generator(device=device).manual_seed(config.seed)) * 0.01
            references = []
            reference_gradients = []
            for index in range(2):
                a, b = torch.nn.Parameter(seed[index].clone()), torch.nn.Parameter(torch.zeros((2, 7), device=device))
                optimizer = torch.optim.SGD((a, b), lr=config.learning_rate)
                active = (labels[index] >= 0) & (weights[index] > 0)
                for _ in range(config.steps):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(logits[index, active] + hidden[index, active] @ a @ b * 1.5,
                        labels[index, active], reduction="none")
                    ((loss * weights[index, active]).sum() / weights[index, active].sum() / 2).backward()
                    optimizer.step()
                references.append((a.detach(), b.detach()))
                reference_gradients.extend((a.grad.clone(), b.grad.clone()))
            execution.trace.configure(TraceSelection(gradients=True))
            result = execution.step(value, stream)
            publication = result.publication
            assert publication is not None
            events = [event for event in execution.trace.snapshot().events if event.phase is TracePhase.GRADIENT]
            assert {identity for event in events for identity in event.identities} == set(identities[:2])
            observed = [event.tensor() for event in events]
            for actual, target in zip(observed, reference_gradients, strict=True):
                torch.testing.assert_close(actual, target.cpu(), rtol=1e-5, atol=1e-7)
            execution.trace.configure(TraceSelection())
            assert publication.before_refs == (initial[1], initial[0])
            for ref, expected in zip(publication.after_refs, references, strict=True):
                for actual, target in zip(execution.snapshot(ref, stream), expected, strict=True):
                    torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-7)
            assert execution.arena.latest(initial[2].lineage_id) == initial[2]
            # Reordered original keys, fewer states/rows, and a new version reuse
            # the same captured graph; group mean now divides by one, not capacity.
            one = WeightedCEBatch(hidden[1:2, :2], logits[1:2, :2], labels[1:2, :2], weights[1:2, :2], (keys[0],))
            a = torch.nn.Parameter(torch.randn((1, 4, 2), device=device,
                generator=torch.Generator(device=device).manual_seed(config.seed))[0] * 0.01)
            b = torch.nn.Parameter(torch.zeros((2, 7), device=device))
            optimizer = torch.optim.SGD((a, b), lr=config.learning_rate)
            for _ in range(config.steps):
                optimizer.zero_grad()
                loss = F.cross_entropy(one.base_logits[0] + one.hidden[0] @ a @ b * 1.5, one.labels[0], reduction="none")
                ((loss * one.weights[0]).sum() / one.weights[0].sum()).backward()
                optimizer.step()
            invocation = execution.start_step(replace(value, auxiliary=one), stream)
            for anchor in execution.arena.plan.timeline.anchors:
                execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
                if anchor.ordinal == execution.sampler_anchor_id:
                    prepared = invocation.context.data().batch_inputs
                    assert tuple(tensor.data_ptr() for tensor in prepared) == tuple(tensor.data_ptr()
                        for tensor in (one.hidden, one.base_logits, one.labels, one.weights))
                    execution.complete_sampler(invocation, stream)
            second = execution.finish_step(invocation).publication
            assert second is not None
            # Inspection after another replay still reads the first result's owned gradients.
            for index, ref in enumerate((initial[1], initial[0])):
                expected = reference_gradients[index * 2:index * 2 + 2]
                for actual, target in zip(result.parameter_gradient_fields[ref], expected, strict=True):
                    torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(result.parameter_gradients[ref],
                    torch.cat(tuple(target.flatten() for target in expected)), rtol=1e-5, atol=1e-7)
            for actual, target in zip(execution.snapshot(second.after_refs[0], stream), (a, b), strict=True):
                torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-7)
            with torch.inference_mode():
                inference_input = replace(one, hidden=one.hidden.clone(), base_logits=one.base_logits.clone(),
                    labels=one.labels.clone(), weights=one.weights.clone())
            second = execution.step(replace(value, auxiliary=inference_input), stream).publication
            assert second is not None
            for actual, target in zip(execution.snapshot(second.after_refs[0], stream), (a, b), strict=True):
                torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-7)
            free = execution.arena.free_pages
            def fail_publish(arena, write):
                raise RuntimeError("abort private LoRA batch")
            with monkeypatch.context() as fault:
                fault.setattr(PageBundleArena, "publish", fail_publish)
                with pytest.raises(RuntimeError, match="abort private"):
                    execution.step(replace(value, auxiliary=one), stream)
            assert execution.arena.free_pages == free
            assert execution.arena.latest(second.after_refs[0].lineage_id) == second.after_refs[0]
            invocation = execution.start_step(replace(value, auxiliary=None), stream)
            for anchor in execution.arena.plan.timeline.anchors:
                execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
                if anchor.ordinal == execution.sampler_anchor_id:
                    guidance = execution.sampler_guidance(invocation, stream)
                    expected = torch.stack([torch.ones(1, 4, device=device) @ factors[0] @ factors[1] * 1.5
                        for factors in ((a, b), references[0], original[2])]).squeeze(1)
                    torch.testing.assert_close(guidance.values, expected, rtol=1e-5, atol=1e-7)
                    execution.complete_sampler(invocation, stream)
            idle = execution.finish_step(invocation)
            assert idle.publication is None and idle.loss is None
            for ref, expected in zip(initial, original, strict=True):
                for actual, target in zip(execution.arena.snapshot(old_pin, ref), expected, strict=True):
                    torch.testing.assert_close(actual, target, rtol=0, atol=0)
        diagnostics = execution.diagnostics()
        assert diagnostics.loss_count == 12 and diagnostics.update_count == 4 and diagnostics.trained_rows == 9
        execution.arena.wait_release(execution.arena.release_pin(old_pin))
        execution.retire(identities)
        assert execution.diagnostics().active_lineages == execution.diagnostics().fallback_count == 0
