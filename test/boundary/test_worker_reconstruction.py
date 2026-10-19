"""Evidence 18: a spawned worker consumes the exact plan, without selection."""

import json
import multiprocessing as mp
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection

import pytest
import torch

import pllm
from pllm import make_llm
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import TaskResourceFact
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.host_runtime import HostRuntimeFacts
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.scheduler import compile_state_schedule
from pllm.workloads.dummy import DummyWorkload
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload


def doubled_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    from pllm.workloads.override import cross_entropy

    return cross_entropy(logits, labels) * 2


class PlainProbe(pllm.Workload):
    """Not a frozen dataclass: EngineCore reference transport rejects it with guidance."""


@dataclass(frozen=True, slots=True)
class FakeTransportConfig:
    value: int


class FakeTransport:
    """WorkerTransportable test double; rebuild proves the worker-side hook ran."""

    def __init__(self, config: FakeTransportConfig) -> None:
        self.config = config

    def transport_payload(self) -> FakeTransportConfig:
        return self.config

    @classmethod
    def transport_payload_type(cls) -> type[FakeTransportConfig]:
        return FakeTransportConfig

    @classmethod
    def transport_rebuild(cls, payload: FakeTransportConfig, *, device: torch.device, seed: int) -> "FakeTransport":
        return cls(FakeTransportConfig(payload.value + seed))


@dataclass(frozen=True)
class TransportHolder(DummyWorkload):
    extra: FakeTransport = FakeTransport(FakeTransportConfig(1))


def _worker(connection: Connection, expected_signature: str) -> None:
    try:
        for attempt in range(2):
            payload = connection.recv_bytes().decode()
            try:
                plan = decode_boundary(payload, PlanBundle)
                provider = PageBundleProviderFacts("canonical.page_bundle_cpu@1", 4 if attempt == 1 else 3)
                catalog = tuple(
                    provider.materialize_plan(
                        resource=entry.resource, layout=entry.layout, binding=binding, max_admitted_lineages=1
                    )
                    for entry, binding in zip(
                        plan.tensor_layout_request.layouts, plan.tensor_layout_decision.bindings, strict=True
                    )
                )
                if plan.schedule_decision.signature != expected_signature:
                    raise ValueError("worker plan differs from the submitted compilation")
                arena = PageBundleArena(plan, catalog)
                (resource,) = arena.plan.semantic_graph.states
                write = arena.prepare_admission((resource.resource,))
                arena.write_field(write, 0, LayoutFieldId(0)).fill_(3)
                receipt = arena.publish(write)
                arena.settle(receipt)
                (ref,) = receipt.after_refs
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


@pytest.mark.slow
def test_worker_reconstructs_exact_signed_plan_and_rejects_local_drift() -> None:
    from examples.centered_hidden import centered_hidden
    from examples.huber import HuberWorkload
    from pllm.contracts.workload_reference import WorkloadReference
    from pllm.workload_transport import decode_workload, encode_workload
    from pllm.workloads.esamp_request import ESampRequestWorkload
    from pllm.workloads.lora import OutputLoRAWorkload
    from pllm.workloads.override import OverRIDEWorkload

    # Built-ins and user workloads travel identically: class coordinates plus frozen fields.
    # Module-level function replacements are imported by the worker and re-signed there.
    for workload in (
        DummyWorkload(),
        OverRIDEWorkload(),
        ESampRequestWorkload(),
        OutputLoRAWorkload(),
        HuberWorkload(3),
        centered_hidden(),
        replace(DummyWorkload(), export_name="custom"),
        replace(OverRIDEWorkload(), objective=doubled_cross_entropy),
        TransportHolder(extra=FakeTransport(FakeTransportConfig(2))),
    ):
        restored = decode_workload(encode_workload(workload), device=torch.device("cpu"), seed=0)
        assert type(restored) is type(workload)
        if isinstance(workload, TransportHolder):
            assert isinstance(restored, TransportHolder)
            assert restored.extra.config == FakeTransportConfig(2)
        else:
            assert restored == workload

    class CustomESamp(ESampRequestWorkload):
        pass

    for custom in (PlainProbe(), CustomESamp()):
        with pytest.raises(ValueError, match="cannot import|frozen dataclass"):
            encode_workload(custom)
    with pytest.raises(ValueError, match="closure or local"):
        encode_workload(replace(DummyWorkload(), transform=lambda hidden: hidden))
    reference = decode_boundary(encode_workload(DummyWorkload()), WorkloadReference)
    tampered = encode_boundary(replace(reference, module_digest="0" * 64))
    with pytest.raises(ValueError, match="differs from the driver"):
        decode_workload(tampered, device=torch.device("cpu"), seed=0)
    execution = make_llm(
        workload=TTRLWorkload(TTRLConfig(rank=2)), config=HostRuntimeFacts(torch.device("cpu"), hidden_size=4)
    )
    plan = execution.arena.plan
    timeline = replace(
        plan.timeline,
        task_resources=tuple(
            TaskResourceFact(task.symbol, 0, index + 1) for index, task in enumerate(plan.semantic_graph.tasks)
        ),
    )
    state_decision, state, decision, physical = compile_state_schedule(
        plan.semantic_graph, timeline, plan.state_request
    )
    plan = replace(
        plan,
        timeline=timeline,
        state_decision=state_decision,
        state_physical=state,
        schedule_decision=decision,
        schedule_physical=physical,
    )
    wire = encode_boundary(plan)
    restored = decode_boundary(wire, PlanBundle)
    assert restored == plan
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_worker, args=(child, plan.schedule_decision.signature))
    worker.start()
    child.close()
    try:
        # The second attempt sees different local provider facts on the worker.
        for index, payload in enumerate((wire, wire)):
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
    for payload in (
        json.dumps(iterable_object),
        wire[:-1] + ',"module":"os"}',
        wire.replace('"plan_epoch":1', '"plan_epoch":true'),
        wire.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1),
    ):
        with pytest.raises(ValueError):
            decode_boundary(payload, PlanBundle)
