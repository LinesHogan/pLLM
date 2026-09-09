"""Evidence 18: a spawned worker consumes the exact plan, without selection."""

from dataclasses import replace
import json
import multiprocessing as mp
from multiprocessing.connection import Connection

import pytest
import torch

from pllm import make_llm
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.execution import TaskGraphExecution
from pllm.execution.host_runtime import HostRuntimeFacts
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.scheduler import compile_state_schedule, compile_schedule, lower_physical_plan, validate_physical_plan
from pllm.execution.trace import ExecutionTrace
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.state_ref import StateRef
from pllm.contracts.semantic_graph import (
    AnchorSymbol, ExecutionDomainFact, ExecutionDomainId, ExecutionKind, ImplementationId, PhysicalPlan,
    SchedulerTimeline, SemanticGraph, SemanticInput, SemanticOutput, SemanticOutputKind, SemanticTask,
    TaskImplementationSpec, TaskResourceFact, TaskResultInputSource, TaskSymbol, TimelineAnchor,
    TimingConstraint, TransferLinkFact, ValueResourceFact, WorkloadId,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId


def _worker(connection: Connection, expected_signature: str) -> None:
    try:
        for attempt in range(5):
            payload = connection.recv_bytes().decode()
            try:
                plan = decode_boundary(payload, PlanBundle)
                provider = PageBundleProviderFacts("canonical.page_bundle_cpu@1", 4 if attempt == 2 else 3)
                catalog = tuple(provider.materialize_plan(resource=entry.resource, layout=entry.layout,
                    binding=binding, max_admitted_lineages=1)
                    for entry, binding in zip(plan.tensor_layout_request.layouts, plan.tensor_layout_decision.bindings, strict=True))
                arena = PageBundleArena.reconstruct(payload, expected_signature, catalog)
                resource, = arena.plan.semantic_graph.states
                write = arena.prepare_admission((resource.resource,))
                arena.write_field(write, 0, LayoutFieldId(0)).fill_(3)
                receipt = arena.publish(write)
                arena.settle(receipt)
                ref, = receipt.after_refs
                pin = arena.pin_exact((ref,))
                assert torch.all(arena.borrow_field(pin, ref, LayoutFieldId(0)) == 3)
                arena.retire((ref,))
                assert arena.active_lineages == 1
                arena.wait_release(arena.release_pin(pin))
                assert arena.active_lineages == 0
                connection.send_bytes(encode_boundary(ref).encode())
            except ValueError:
                connection.send_bytes(b"rejected")
    finally:
        connection.close()


def test_worker_reconstructs_exact_signed_plan_and_rejects_local_drift() -> None:
    from examples.centered_hidden import centered_hidden
    from pllm.worker import encode_workload
    from pllm.workloads.dummy import DummyConfig, DummyWorkload
    from pllm.workloads.override import OverRIDEConfig, OverRIDEWorkload, cross_entropy
    from pllm.workloads.esamp_request import ESampRequestWorkload
    from pllm.workloads.lora import OutputLoRAWorkload
    assert decode_boundary(encode_workload(DummyWorkload()), DummyConfig) == DummyConfig()
    assert decode_boundary(encode_workload(OverRIDEWorkload()), OverRIDEConfig) == OverRIDEConfig()
    def custom_objective(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits, labels) * 0
    for custom in (centered_hidden(), replace(DummyWorkload(), export_name="custom"),
                   replace(OverRIDEWorkload(), objective=custom_objective)):
        with pytest.raises(ValueError, match="local tensor replacements"):
            encode_workload(custom)
    class CustomESamp(ESampRequestWorkload):
        pass
    class CustomLoRA(OutputLoRAWorkload):
        pass
    for custom in (CustomESamp(), CustomLoRA()):
        with pytest.raises(ValueError, match="custom native"):
            encode_workload(custom)
    execution = make_llm(workload=TTRLWorkload(TTRLConfig(rank=2)),
        config=HostRuntimeFacts(torch.device("cpu"), hidden_size=4))
    plan = execution.arena.plan
    timeline = replace(plan.timeline, task_resources=tuple(
        TaskResourceFact(task.symbol, 0, index + 1) for index, task in enumerate(plan.semantic_graph.tasks)))
    state_decision, state, decision, physical = compile_state_schedule(plan.semantic_graph, timeline, plan.state_request)
    plan = replace(plan, timeline=timeline, state_decision=state_decision, state_physical=state,
        schedule_decision=decision, schedule_physical=physical)
    wire = encode_boundary(plan)
    restored = decode_boundary(wire, PlanBundle)
    assert restored == plan
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_worker, args=(child, plan.schedule_decision.signature))
    worker.start()
    child.close()
    try:
        payloads = (wire,
            encode_boundary(replace(plan, schedule_decision=replace(plan.schedule_decision, signature="0" * 64))),
            wire, wire.replace('"plan_epoch":1', '"plan_epoch":2', 1),
            wire.replace('"resident_bytes":1', '"resident_bytes":0', 1))
        for index, payload in enumerate(payloads):
            parent.send_bytes(payload.encode())
            assert parent.poll(20), "worker did not finish reconstruction"
            answer = parent.recv_bytes().decode()
            if index == 0:
                ref = decode_boundary(answer, StateRef)
                assert ref.version_id == 0
            else:
                assert answer == "rejected"
        worker.join(10)
        assert worker.exitcode == 0
    finally:
        parent.close()
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        worker.close()
    iterable_object = json.loads(wire)
    iterable_object["timeline"]["capabilities"] = dict.fromkeys(plan.timeline.capabilities, 0)
    for payload in (json.dumps(iterable_object), wire[:-1] + ',"module":"os"}', wire.replace('"plan_epoch":1', '"plan_epoch":true'),
                    wire.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)):
        with pytest.raises(ValueError):
            decode_boundary(payload, PlanBundle)
    anchors = tuple(TimelineAnchor(AnchorSymbol(f"a{i}"), i) for i in range(3))
    kernel = (TaskImplementationSpec(ImplementationId("kernel"), ExecutionKind.NATIVE_TASK),)
    graph = SemanticGraph(WorkloadId("transfer"), (
        SemanticTask(TaskSymbol("emit"), TimingConstraint(anchors[0].symbol, anchors[1].symbol), (),
            (SemanticOutput("x", SemanticOutputKind.TEMPORARY),), kernel),
        SemanticTask(TaskSymbol("read"), TimingConstraint(anchors[1].symbol, anchors[2].symbol),
            (SemanticInput("x", TaskResultInputSource(TaskSymbol("emit"), "x")),), (), kernel),
    ), "f" * 64)
    timeline = SchedulerTimeline(anchors, domains=tuple(
        ExecutionDomainFact(i, StateDeviceKind.CPU, 0, memory_budget_bytes=32) for i in range(2)),
        task_resources=tuple(TaskResourceFact(task.symbol, 0, domains=(ExecutionDomainId(i),))
            for i, task in enumerate(graph.tasks)),
        value_resources=(ValueResourceFact(graph.tasks[0].symbol, "x", 16),),
        transfer_links=(TransferLinkFact(ExecutionDomainId(0), ExecutionDomainId(1), 2, 1),))
    decision = compile_schedule(graph, timeline)
    physical = lower_physical_plan(graph, timeline, decision, plan_epoch=7)
    decoded = decode_boundary(encode_boundary(physical), PhysicalPlan)
    validate_physical_plan(graph, timeline, decision, decoded)
    assert decoded == physical and decision.transfer_cost == 32
    assert tuple(task.release_values for task in decoded.tasks) == ((), (0,))
    with pytest.raises(ValueError, match="Rust projection"):
        validate_physical_plan(graph, timeline, decision, replace(decoded, tasks=(decoded.tasks[0],
            replace(decoded.tasks[1], release_values=()))))
    assert len(decoded.transfers) == 1
    with pytest.raises(ValueError, match="distributed physical plan"):
        TaskGraphExecution(decoded, (), ExecutionTrace(decoded))
