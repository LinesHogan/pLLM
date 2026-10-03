"""Evidence 20: original hidden inputs, stateless attachment and bounded owned export."""

from dataclasses import replace
from typing import Annotated
from weakref import ref

import pytest
import torch
from pllm import make_llm
import pllm
from pllm.authoring import BackendCompileFacts, TensorStepInput
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.host_runtime import HostRuntimeFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.state_execution import StateDiagnostics
from pllm.execution.tensor_program import TensorOperation
from pllm.execution.trace import ExecutionTrace, MAX_SNAPSHOT_BYTES, TracePhase, TraceSelection, TraceSnapshot
from pllm.workloads.dummy import DummyConfig, DummyWorkload, hidden_rows
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentHiddenResource, AttachmentTopology, BackendPlacementProbe
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.contracts.semantic_graph import AnchorId, TaskSymbol
from test.backend_facts import assert_finite_resources


def test_dummy_runs_graph_sign_plan_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.centered_hidden import centered_hidden
    workload = DummyWorkload(DummyConfig(export_every_n_steps=2, export_max_rows=2, export_max_cols=3))
    execution = make_llm(workload=workload, config=HostRuntimeFacts(torch.device("cpu")))
    hidden = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    expected = hidden[1:3, :3].clone()
    binding = PackedHiddenResourceBinding("0", HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, "model.layers.0")
    def value(step):
        rows = PackedPhaseRowBatch(1, step, 3, tuple(PackedPhaseRow(index, index, 0,
            HiddenPhase.PREFILL if index == 0 else HiddenPhase.DECODE, RequestIdentity(str(index), index, 0))
            for index in range(3)))
        return TensorStepInput(rows, PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(3))),
            PackedHiddenSnapshotBatch(rows, (PackedHiddenSnapshot(binding, hidden),)))
    assert execution.step(value(1)) is None
    exported = execution.step(value(2))
    assert isinstance(exported, torch.Tensor)
    hidden.fill_(-1)
    torch.testing.assert_close(exported, expected)
    assert exported.device.type == "cpu" and execution.plan.fallback_count == 0
    hidden.copy_(torch.arange(15, dtype=torch.float32).reshape(3, 5))
    variant = make_llm(HostRuntimeFacts(torch.device("cpu")), workload=centered_hidden(workload.config))
    torch.testing.assert_close(variant.step(value(2)), expected - expected.mean(dim=-1, keepdim=True))
    snapshot = execution.trace.snapshot()
    assert decode_boundary(encode_boundary(snapshot), TraceSnapshot) == snapshot
    observed, = (event for event in snapshot.events if event.phase is TracePhase.TENSOR)
    assert observed.step_id == 2 and tuple(identity.prompt_idx for identity in observed.identities) == (1, 2)
    torch.testing.assert_close(observed.tensor(), expected)
    observed.tensor().fill_(123)
    torch.testing.assert_close(observed.tensor(), expected)
    trace = ExecutionTrace(execution.plan, capacity=2)
    task, = execution.plan.tasks
    for tensor in (torch.tensor([2**60 + 1]), torch.tensor([True, False]),
                   torch.tensor([float("nan"), float("inf")], dtype=torch.bfloat16)):
        trace.record(TracePhase.TENSOR, 1, task_id=task.task_id, tensor=tensor, tensor_name="selected")
        event = decode_boundary(encode_boundary(trace.snapshot()), TraceSnapshot).events[-1]
        torch.testing.assert_close(event.tensor(), tensor, equal_nan=True)
    assert trace.snapshot().dropped == 1 and len(trace.snapshot().events) == 2
    with pytest.raises(ValueError, match="bounded"):
        trace.record(TracePhase.TENSOR, 1, task_id=task.task_id, tensor=torch.zeros(MAX_SNAPSHOT_BYTES), tensor_name="too_large")
    with pytest.raises(ValueError, match="shape"):
        replace(observed, tensor_shape=(99,)).tensor()
    trace = ExecutionTrace(execution.plan, capacity=4)
    row = value(2).rows.rows[1]
    selection = TraceSelection(tasks=(task.task_id,), identities=(row.identity,), iterations=(2,),
        names=("activation", "weight.grad"), tensors=True, gradients=True, max_tensor_bytes=8)
    trace.configure(decode_boundary(encode_boundary(selection), TraceSelection))
    original_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    trace.observe(1, task.task_id, "activation", original_tensor, row=row, iteration=1)
    trace.observe(1, task.task_id, "activation", original_tensor, row=value(2).rows.rows[2], iteration=2)
    trace.observe(1, task.task_id, "activation", original_tensor, identities=(row.identity, value(2).rows.rows[2].identity), iteration=2)
    assert not trace.snapshot().events
    trace.observe(1, task.task_id, "activation", original_tensor, row=row, iteration=2)
    trace.observe(1, task.task_id, "weight.grad", original_tensor + 1, row=row, iteration=2, gradient=True)
    observed, grad = decode_boundary(encode_boundary(trace.snapshot()), TraceSnapshot).events
    assert observed.source_shape == (2, 3) and observed.tensor_shape == (2,) and observed.row == row
    assert grad.phase is TracePhase.GRADIENT
    torch.testing.assert_close(observed.tensor(), original_tensor.flatten()[:2])
    torch.testing.assert_close(grad.tensor(), (original_tensor + 1).flatten()[:2])
    trace.configure(TraceSelection())
    trace.observe(1, task.task_id, "activation", original_tensor)
    assert len(trace.snapshot().events) == 2
    # Bounded snapshots must not first materialize a full strided/broadcast source.
    trace.configure(TraceSelection(tensors=True, max_tensor_bytes=128))
    for device in ("cpu", *(("cuda",) if torch.cuda.is_available() else ())):
        for transposed in (False, True):
            for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.bool):
                # Convert the small base before expanding, keeping the large input a view.
                small = torch.arange(2049, device=device).to(dtype)[1:]
                tensor = small.expand(2048, -1).T if transposed else small.expand(2048, -1)
                if device == "cuda":
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                trace.observe(2, task.task_id, "bounded", tensor)
                event = trace.snapshot().events[-1]
                count = 128 // tensor.element_size()
                prefix = small[:count] if tensor.stride(0) == 0 else small[:1].expand(count)
                torch.testing.assert_close(event.tensor(), prefix.cpu())
                assert event.source_shape == (2048, 2048)
                if device == "cuda":
                    assert torch.cuda.max_memory_allocated() - baseline < 64 * 1024
    disabled = DummyWorkload(replace(workload.config, enabled=False)).compile_host(HostRuntimeFacts(torch.device("cpu")))
    assert disabled.step(value(2)) is None
    assert not any(event.phase is TracePhase.TENSOR for event in disabled.trace.snapshot().events)
    facts = BackendCompileFacts(BackendPlacementProbe("test.observer@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CPU, 0, 1, 1,
        TensorDType.FLOAT32, 5, 3, (AttachmentHiddenResource(binding, HiddenPhase.DECODE),), ()),
        RuntimeFacts(torch.device("cpu")).timeline(), (), 1, 0, 7, layer_count=1)
    bounded = assert_finite_resources(workload, facts)
    deployment = workload.compile_backend(bounded)
    with pytest.raises(ValueError, match="resource_facts"):
        centered_hidden(workload.config).specialize(facts, pllm.ExecutionOptions(memory_budget_bytes=1 << 30))
    assert deployment.execution.arena.active_lineages == 0
    assert deployment.execution.step(value(2)).shape == (2, 3)
    original = value(2)
    excess = replace(original.rows, total_rows=4, rows=(*original.rows.rows,
        PackedPhaseRow(3, 0, 1, HiddenPhase.PREFILL, original.rows.rows[0].identity)))
    oversized = TensorStepInput(excess, PackedSamplerRowBatch(excess, original.sampler_rows.links),
        PackedHiddenSnapshotBatch(excess, (PackedHiddenSnapshot(binding, torch.cat((hidden, hidden[:1]))),)))
    # Only two decode rows and three requests, but four original packed tokens exceed the signed cap.
    before = deployment.execution.trace.snapshot()
    with pytest.raises(ValueError, match="signed backend row capacity"):
        deployment.execution.start_step(oversized)
    assert deployment.execution.trace.snapshot() == before
    invocation = deployment.execution.begin(original.rows.decode_rows)
    with pytest.raises(ValueError, match="signed backend row capacity"):
        deployment.execution.submit_input(invocation, oversized)
    assert invocation.context.value is None
    diagnostics = deployment.execution.diagnostics()
    assert diagnostics.pending_state_releases == 0
    assert diagnostics.state_usage.reserved_bytes == diagnostics.state_usage.logical_live_bytes == 0
    assert decode_boundary(encode_boundary(diagnostics), StateDiagnostics) == diagnostics
    wrong_width = replace(original, hidden=PackedHiddenSnapshotBatch(original.rows,
        (PackedHiddenSnapshot(binding, torch.zeros(3, 6)),)))
    with pytest.raises(ValueError, match="parameter specialization"):
        deployment.execution.step(wrong_width)
    torch.testing.assert_close(deployment.execution.step(original), expected)
    with pytest.raises(ValueError, match="signed declaration"):
        DummyWorkload(replace(workload.config, export_every_n_steps=1)).bind_backend(bounded, deployment.attachment)
    with pytest.raises(ValueError, match="hidden inputs"):
        workload.plan_backend(replace(facts, probe=replace(facts.probe, hidden_resources=(
            AttachmentHiddenResource(replace(binding, layer="1"), HiddenPhase.DECODE),))))
    assert workload.transform is hidden_rows
    assert variant.arena.plan.semantic_graph != execution.arena.plan.semantic_graph
    torch.testing.assert_close(execution.step(value(2)), expected)
    for modified in (centered_hidden(workload.config), replace(workload, export_name="renamed")):
        with pytest.raises(ValueError, match="signed declaration"):
            modified.bind_backend(bounded, deployment.attachment)
    class Chained(DummyWorkload):
        @pllm.on(pllm.Phase.OBSERVE)
        def shift(self, x: Annotated[torch.Tensor, pllm.Hidden(0)]) -> torch.Tensor:
            return x + 1

        @pllm.on(pllm.Phase.OBSERVE)
        @pllm.export(name="chain", max_rows=2, max_cols=3)
        def export_hidden(self, x: Annotated[torch.Tensor, pllm.Result("shift")]) -> torch.Tensor:
            return x * 2

    chained = make_llm(HostRuntimeFacts(torch.device("cpu")), workload=Chained())
    invocation = chained.start_step(value(2))
    saved = {}
    execute = TensorOperation.execute
    def observe(self, context):
        execute(self, context)
        saved.update((key, ref(tensor)) for key, tensor in context.results.items())
    with monkeypatch.context() as watch:
        watch.setattr(TensorOperation, "execute", observe)
        for anchor in chained.arena.plan.timeline.anchors:
            chained.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.ordinal == chained.sampler_anchor_id:
                chained.complete_sampler(invocation)
    assert (TaskSymbol("shift"), "value") not in invocation.context.results
    assert (TaskSymbol("export_hidden"), "value") in invocation.context.results
    assert saved[TaskSymbol("shift"), "value"]() is None
    assert saved[TaskSymbol("export_hidden"), "value"]() is not None
    torch.testing.assert_close(chained.finish_step(invocation), (expected + 1) * 2)
    assert not invocation.context.results
    assert chained.diagnostics().fallback_count == 0

    scale = [2.0]
    def scaled(hidden: torch.Tensor) -> torch.Tensor:
        return hidden * scale[0]
    custom = replace(workload, transform=scaled)
    frozen = custom.compile_host(HostRuntimeFacts(torch.device("cpu")))
    scale[0] = 3.0
    torch.testing.assert_close(frozen.step(value(2)), expected * 2)
    fresh = custom.compile_host(HostRuntimeFacts(torch.device("cpu")))
    torch.testing.assert_close(fresh.step(value(2)), expected * 3)
    assert frozen.arena.plan.semantic_graph != fresh.arena.plan.semantic_graph
    import pllm.usage as authoring_binding
    def unexpected_inspection(*args, **kwargs):
        raise AssertionError("reflection reached execution")
    with monkeypatch.context() as cold_only:
        cold_only.setattr(authoring_binding, "signature", unexpected_inspection)
        torch.testing.assert_close(frozen.step(value(2)), expected * 2)

    class Ambiguous(DummyWorkload):
        @pllm.on(pllm.Phase.OBSERVE)
        @pllm.hook(0)
        @pllm.export()
        def export_hidden(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            return first + second
    with pytest.raises(TypeError, match="multiple Tensor inputs"):
        Ambiguous().backend_requirements()

    # A frozen state — declared once, initialized at admission — is readable from an
    # observation without ever becoming an update target, and the signed export sees it.
    class FrozenEncoder(pllm.Workload):
        encoder = pllm.InstanceState({"weight": lambda model: (4, model.hidden_size)},
            dtype=torch.float32,
            initialize=lambda tensors, generator: tensors["weight"].copy_(torch.eye(4)))

        @pllm.on(pllm.Phase.OBSERVE)
        @pllm.hook(0)
        @pllm.export(name="features", every=1, max_rows=2, max_cols=4)
        def encode(self, hidden: torch.Tensor,
                   weight: Annotated[torch.Tensor, pllm.Parameter("encoder", "weight")]) -> torch.Tensor:
            return hidden @ weight.T

    from test.backend_facts import backend_facts
    from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
    frozen_facts = backend_facts(4, (binding,), RuntimeFacts(torch.device("cpu")).timeline(),
        PageBundleProviderFacts("test.plain@1", 4, StateDeviceKind.CPU), max_lineages=2)
    frozen_exec = FrozenEncoder().compile_backend(frozen_facts).execution
    identity = RequestIdentity("req", 0, 0)
    frozen_exec.admit(())
    frozen_exec.admit((identity,))
    row_batch = PackedPhaseRowBatch(1, 0, 1, (PackedPhaseRow(0, 0, 0, HiddenPhase.DECODE, identity),))
    source = (torch.arange(4, dtype=torch.float32) + 1).unsqueeze(0)
    frozen_exec.step(TensorStepInput(row_batch, PackedSamplerRowBatch(row_batch, ()),
        PackedHiddenSnapshotBatch(row_batch, (PackedHiddenSnapshot(binding, source),))))
    events = [event.tensor() for event in frozen_exec.trace.snapshot().events if event.tensor_name == "features"]
    assert len(events) == 1 and torch.allclose(events[0][0], source)
    assert frozen_exec.diagnostics().fallback_count == 0
    frozen_exec.retire((identity,))


def test_unused_helpers_are_neither_signed_nor_frozen() -> None:
    class Lazy(pllm.Workload):
        def unused(self):
            import os  # Forbidden dependency: signing this helper would raise.
            return os.getcwd()

        @pllm.on(pllm.Phase.OBSERVE)
        def observe(self, hidden: Annotated[torch.Tensor, pllm.Hidden(0)]) -> torch.Tensor:
            return hidden * 2

    execution = make_llm(workload=Lazy(), config=HostRuntimeFacts(torch.device("cpu")))
    assert execution.plan.fallback_count == 0
