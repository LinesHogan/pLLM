"""Evidence 18: a spawned worker rebuilds the shipped workload, plans it and binds it."""

import multiprocessing as mp
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from threading import Lock
from unittest.mock import Mock

import pytest
import torch

from examples.huber import HuberWorkload
from pllm import ExecutionOptions
from pllm.backends.vllm.runner import BackendInstall, WorkloadModelRunner
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.state_refs import StateRefs
from pllm.host import HostConfig
from pllm.workload_transport import decode_workload, encode_workload
from pllm.workloads.dummy import DummyWorkload
from pllm.workloads.override import OverRIDEWorkload
from pllm.workloads.prefill_adapters import PrefillAdaptersWorkload
from pllm.workloads.reward_head import RewardHeadConfig, RewardHeadWorkload
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


def test_workload_transport_round_trips_workloads_and_rejects_unpicklable_fields() -> None:
    # Built-ins and user workloads travel identically; module-level function replacements are imported by the worker.
    for workload in (
        DummyWorkload(),
        replace(OverRIDEWorkload(), objective=doubled_cross_entropy),
        HuberWorkload(3),
        PrefillAdaptersWorkload(dtype=torch.float64),
    ):
        assert decode_workload(encode_workload(workload)) == workload
    restored = decode_workload(encode_workload(TemplateHolder(extra=FakeTemplate(2))))
    assert isinstance(restored, TemplateHolder) and restored.extra == FakeTemplate(3)
    with pytest.raises(ValueError):
        encode_workload(replace(DummyWorkload(), transform=lambda hidden, lock=Lock(): hidden))


def _plan() -> PlanBundle:
    config = HostConfig(torch.device("cpu"), hidden_size=4)
    return RewardHeadWorkload(RewardHeadConfig(rank=2)).compile_host(config).arena.plan


def test_plan_bundle_codec_round_trips_and_rejects_injected_fields() -> None:
    wire = encode_boundary(_plan())
    assert decode_boundary(wire, PlanBundle) == _plan()
    with pytest.raises(ValueError):
        decode_boundary(wire[:-1] + ',"module":"os"}', PlanBundle)


def _worker(connection: Connection) -> None:
    try:
        install = decode_boundary(connection.recv_bytes().decode(), BackendInstall)
        runner = Mock(installs=0)
        WorkloadModelRunner.install_workload(
            runner, decode_workload(install.definition), install.facts, install.options, install.generated_kv
        )
        arena = runner.attach_workload.call_args.args[0].execution.arena
        (resource,) = arena.plan.semantic_graph.states
        write = arena.prepare_admission((resource.resource,))
        arena.write_field(write, 0, LayoutFieldId(0)).fill_(3)
        receipt = arena.publish(write)
        (ref,) = receipt.after_refs
        pin = arena.pin_exact(StateRefs.of((ref,)))
        assert torch.all(arena.borrow_field(pin, ref, LayoutFieldId(0)) == 3)
        connection.send_bytes(encode_boundary(ref).encode())
    finally:
        connection.close()


@pytest.mark.slow
def test_worker_plans_and_binds_the_shipped_workload() -> None:
    workload, _, facts = overlay_facts()
    wire = encode_boundary(BackendInstall(facts, ExecutionOptions(), encode_workload(workload), "reuse")).encode()
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_worker, args=(child,))
    worker.start()
    child.close()
    try:
        parent.send_bytes(wire)
        assert parent.poll(20), "worker did not finish installing"
        assert decode_boundary(parent.recv_bytes().decode(), StateRef).version_id == 0
        worker.join(10)
        assert worker.exitcode == 0
    finally:
        parent.close()
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        worker.close()
