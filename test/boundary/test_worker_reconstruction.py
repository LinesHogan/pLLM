"""Evidence 18: a spawned worker consumes the exact plan, without selection."""

import json
import multiprocessing as mp
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection

import pytest
import torch

import pllm
from examples.centered_hidden import centered_hidden
from examples.huber import HuberWorkload
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
from pllm.workload_transport import decode_workload, encode_workload
from pllm.workloads.dummy import DummyWorkload
from pllm.workloads.esamp_request import ESampRequestWorkload
from pllm.workloads.lora import OutputLoRAWorkload
from pllm.workloads.override import OverRIDEWorkload
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


# Built-ins and user workloads travel identically: class coordinates plus frozen fields.
@pytest.mark.parametrize(
    "workload",
    (
        DummyWorkload(),
        OverRIDEWorkload(),
        ESampRequestWorkload(),
        OutputLoRAWorkload(),
        HuberWorkload(3),
        centered_hidden(),
        replace(DummyWorkload(), export_name="custom"),
        TransportHolder(extra=FakeTransport(FakeTransportConfig(2))),
    ),
)
def test_workload_transport_round_trips_frozen_workloads(workload) -> None:
    restored = decode_workload(encode_workload(workload), device=torch.device("cpu"), seed=0)
    assert type(restored) is type(workload)
    if isinstance(workload, TransportHolder):
        assert isinstance(restored, TransportHolder)
        assert restored.extra.config == FakeTransportConfig(2)
    else:
        assert restored == workload


def test_module_level_function_replacement_round_trips() -> None:
    # Module-level function replacements are imported by the worker and re-signed there.
    workload = replace(OverRIDEWorkload(), objective=doubled_cross_entropy)
    restored = decode_workload(encode_workload(workload), device=torch.device("cpu"), seed=0)
    assert isinstance(restored, OverRIDEWorkload)
    assert restored.objective is doubled_cross_entropy


def test_transport_rejects_workloads_the_worker_cannot_import() -> None:
    class CustomDummy(DummyWorkload):
        pass

    for custom in (PlainProbe(), CustomDummy()):
        with pytest.raises(ValueError):
            encode_workload(custom)
    with pytest.raises(ValueError):
        encode_workload(replace(DummyWorkload(), transform=lambda hidden: hidden))


def test_transport_rejects_a_tampered_module_digest() -> None:
    from pllm.contracts.workload_reference import WorkloadReference

    reference = decode_boundary(encode_workload(DummyWorkload()), WorkloadReference)
    tampered = encode_boundary(replace(reference, module_digest="0" * 64))
    with pytest.raises(ValueError):
        decode_workload(tampered, device=torch.device("cpu"), seed=0)


def _drifted_plan() -> PlanBundle:
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
    return replace(
        plan,
        timeline=timeline,
        state_decision=state_decision,
        state_physical=state,
        schedule_decision=decision,
        schedule_physical=physical,
    )


def test_plan_bundle_codec_round_trips_and_rejects_malformed_wire() -> None:
    wire = encode_boundary(_drifted_plan())
    plan = decode_boundary(wire, PlanBundle)
    assert decode_boundary(wire, PlanBundle) == plan
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
    plan = _drifted_plan()
    wire = encode_boundary(plan)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_worker, args=(child, plan.schedule_decision.signature))
    worker.start()
    child.close()
    try:
        # The second attempt sees different local provider facts on the worker.
        for index in range(2):
            parent.send_bytes(wire.encode())
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
