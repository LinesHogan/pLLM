"""Evidence 18: a spawned worker consumes the exact plan, without selection."""

import multiprocessing as mp
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection

import pytest
import torch

import pllm
from examples.huber import HuberWorkload
from pllm import make_llm
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.contracts.workload_reference import WorkloadReference
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.host_runtime import HostRuntimeFacts
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.workload_transport import decode_workload, encode_workload
from pllm.workloads.dummy import DummyWorkload
from pllm.workloads.override import OverRIDEWorkload
from pllm.workloads.prefill_adapters import PrefillAdapters
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


def test_workload_transport_round_trips_frozen_workloads_and_rejects_what_the_worker_cannot_import() -> None:
    # Built-ins and user workloads travel identically: class coordinates plus frozen fields;
    # module-level function replacements are imported by the worker.
    for workload in (
        DummyWorkload(),
        replace(OverRIDEWorkload(), objective=doubled_cross_entropy),
        HuberWorkload(3),
        PrefillAdapters(dtype=torch.float64),
    ):
        assert decode_workload(encode_workload(workload), device=torch.device("cpu"), seed=0) == workload
    holder = TransportHolder(extra=FakeTransport(FakeTransportConfig(2)))
    restored = decode_workload(encode_workload(holder), device=torch.device("cpu"), seed=0)
    assert isinstance(restored, TransportHolder) and restored.extra.config == FakeTransportConfig(2)
    for workload in (PlainProbe(), replace(DummyWorkload(), transform=lambda hidden: hidden)):
        with pytest.raises(ValueError):
            encode_workload(workload)
    reference = decode_boundary(encode_workload(DummyWorkload()), WorkloadReference)
    with pytest.raises(ValueError):
        decode_workload(encode_boundary(replace(reference, module_digest="0" * 64)), device=torch.device("cpu"), seed=0)


def _plan() -> PlanBundle:
    config = HostRuntimeFacts(torch.device("cpu"), hidden_size=4)
    return make_llm(workload=TTRLWorkload(TTRLConfig(rank=2)), config=config).arena.plan


def test_plan_bundle_codec_round_trips_and_rejects_injected_fields() -> None:
    wire = encode_boundary(_plan())
    assert decode_boundary(wire, PlanBundle) == _plan()
    with pytest.raises(ValueError):
        decode_boundary(wire[:-1] + ',"module":"os"}', PlanBundle)


def _worker(connection: Connection) -> None:
    try:
        for page_count in (3, 4):
            plan = decode_boundary(connection.recv_bytes().decode(), PlanBundle)
            provider = PageBundleProviderFacts("canonical.page_bundle_cpu@1", page_count)
            try:
                catalog = tuple(
                    provider.materialize_plan(
                        resource=entry.resource, layout=entry.layout, binding=binding, max_admitted_lineages=1
                    )
                    for entry, binding in zip(
                        plan.tensor_layout_request.layouts, plan.tensor_layout_decision.bindings, strict=True
                    )
                )
                arena = PageBundleArena(plan, catalog)
                (resource,) = arena.plan.semantic_graph.states
                write = arena.prepare_admission((resource.resource,))
                arena.write_field(write, 0, LayoutFieldId(0)).fill_(3)
                receipt = arena.publish(write)
                arena.settle(receipt)
                (ref,) = receipt.after_refs
                pin = arena.pin_exact((ref,))
                assert torch.all(arena.borrow_field(pin, ref, LayoutFieldId(0)) == 3)
                connection.send_bytes(encode_boundary(ref).encode())
            except ValueError:
                connection.send_bytes(b"rejected")
    finally:
        connection.close()


@pytest.mark.slow
def test_worker_reconstructs_the_shipped_plan_and_rejects_local_drift() -> None:
    wire = encode_boundary(_plan()).encode()
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_worker, args=(child,))
    worker.start()
    child.close()
    try:
        # The second attempt sees different local provider facts on the worker.
        answers = []
        for _ in range(2):
            parent.send_bytes(wire)
            assert parent.poll(20), "worker did not finish reconstruction"
            answers.append(parent.recv_bytes().decode())
        assert decode_boundary(answers[0], StateRef).version_id == 0 and answers[1] == "rejected"
        worker.join(10)
        assert worker.exitcode == 0
    finally:
        parent.close()
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        worker.close()
