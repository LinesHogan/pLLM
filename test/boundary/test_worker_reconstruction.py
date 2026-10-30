"""Evidence 18: a spawned worker consumes the exact plan, without selection."""

import multiprocessing as mp
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch

from examples.huber import HuberWorkload
from pllm import make_llm
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.backends.vllm.runtime import BackendInstall
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.state_refs import StateRefs
from pllm.host import HostRuntimeFacts
from pllm.worker import GPUWorker
from pllm.workload_transport import decode_workload, encode_workload
from pllm.workloads.dummy import DummyWorkload
from pllm.workloads.override import OverRIDEWorkload
from pllm.workloads.prefill_adapters import PrefillAdapters
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload
from test.boundary.test_backend_attachment import overlay_facts


def doubled_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    from pllm.workloads.override import cross_entropy

    return cross_entropy(logits, labels) * 2


@dataclass(frozen=True, slots=True)
class FakeTemplate:
    """Stands in for a template model: travels as its config and is rebuilt by the worker."""

    value: int

    def __reduce__(self) -> tuple[type["FakeTemplate"], tuple[int]]:
        return FakeTemplate, (self.value + 1,)


@dataclass(frozen=True)
class TemplateHolder(DummyWorkload):
    extra: FakeTemplate = FakeTemplate(1)


def test_workload_transport_round_trips_workloads_and_rejects_what_the_worker_cannot_import() -> None:
    # Built-ins and user workloads travel identically; module-level function replacements are imported by the worker.
    for workload in (
        DummyWorkload(),
        replace(OverRIDEWorkload(), objective=doubled_cross_entropy),
        HuberWorkload(3),
        PrefillAdapters(dtype=torch.float64),
    ):
        assert decode_workload(encode_workload(workload)) == workload
    restored = decode_workload(encode_workload(TemplateHolder(extra=FakeTemplate(2))))
    assert isinstance(restored, TemplateHolder) and restored.extra == FakeTemplate(3)
    with pytest.raises(ValueError):
        encode_workload(replace(DummyWorkload(), transform=lambda hidden: hidden))


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
        workload, _, _ = overlay_facts()
        for imported in (workload, replace(workload, config=replace(workload.config, learning_rate=0.5))):
            install = decode_boundary(connection.recv_bytes().decode(), BackendInstall)
            runner = Mock()
            try:
                with patch.object(GPUWorker, "workload_runner", new_callable=PropertyMock, return_value=runner):
                    GPUWorker._install(object.__new__(GPUWorker), imported, install)
                arena = runner.attach_workload.call_args.args[0].execution.arena
                (resource,) = arena.plan.semantic_graph.states
                write = arena.prepare_admission((resource.resource,))
                arena.write_field(write, 0, LayoutFieldId(0)).fill_(3)
                receipt = arena.publish(write)
                arena.settle(receipt)
                (ref,) = receipt.after_refs
                pin = arena.pin_exact(StateRefs.of((ref,)))
                assert torch.all(arena.borrow_field(pin, ref, LayoutFieldId(0)) == 3)
                connection.send_bytes(encode_boundary(ref).encode())
            except ValueError as error:
                assert "differs from the driver's" in str(error)
                connection.send_bytes(b"rejected")
    finally:
        connection.close()


@pytest.mark.slow
def test_worker_reconstructs_the_shipped_plan_and_rejects_source_drift() -> None:
    workload, _, facts = overlay_facts()
    attachment, _ = workload.plan_backend(facts)
    policy = CachePolicy.for_attachment(attachment, "reuse")
    wire = encode_boundary(BackendInstall(facts, attachment, "", policy)).encode()
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_worker, args=(child,))
    worker.start()
    child.close()
    try:
        # A worker whose imported workload declares different math is rejected before attaching.
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
