"""The dummy observer exports bounded copies of hidden rows, on the host and attached to a backend.

Also covers the execution trace it reports through (bounded, selectable snapshots
that survive the process boundary) and authoring rules shared by every workload:
signed declarations, chained results, values frozen at compile time and frozen state.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import pllm
from examples.centered_hidden import centered_hidden
from examples.sae_export import SAE
from pllm import make_llm, testing
from pllm.contracts.backend_attachment import AttachmentHiddenResource
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.tensor_task import TensorTask
from pllm.execution.trace import ExecutionTrace, TracePhase, TraceSelection, TraceSnapshot
from pllm.host import HostRuntimeFacts
from pllm.workloads.dummy import DummyConfig, DummyWorkload
from test.backend_facts import assert_finite_resources, backend_facts

BINDING = testing.hidden_binding(0, HiddenSite.BLOCK_OUTPUT, 1)
CPU = HostRuntimeFacts(torch.device("cpu"))


def step_input(step, hidden, count=3):
    """Row 0 is a prefill row; the rest decode, one request each."""
    phases = (HiddenPhase.PREFILL,) + (HiddenPhase.DECODE,) * (count - 1)
    identities = tuple(RequestIdentity(str(i), i, 0) for i in range(count))
    return testing.step_input(identities, {BINDING: hidden[:count]}, step_id=step, phases=phases)


def exported(execution, value):
    """Exports are terminal effects; explicit snapshot reads may wait."""
    assert execution.step(value) is None
    execution.trace.drain()
    return next(event.tensor() for event in reversed(execution.trace.snapshot().events) if event.tensor_name)


def test_dummy_runs_graph_sign_plan_execute() -> None:
    workload = DummyWorkload(DummyConfig(export_every_n_steps=2, export_max_rows=2, export_max_cols=3))
    execution = make_llm(workload=workload, config=CPU)
    hidden = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    expected = hidden[1:3, :3].clone()
    value = lambda step: step_input(step, hidden)
    # Every second step exports the decode rows, bounded to 2 x 3, as an owned copy.
    with patch.object(TensorTask, "read", autospec=True, side_effect=TensorTask.read) as reads:
        assert execution.step(value(1)) is None
        assert execution.step(step_input(2, hidden, count=1)) is None
        reads.assert_not_called()
        result = exported(execution, value(2))
        reads.assert_called_once()
    hidden.fill_(-1)
    torch.testing.assert_close(result, expected)
    hidden.copy_(torch.arange(15, dtype=torch.float32).reshape(3, 5))
    variant = make_llm(CPU, workload=centered_hidden(workload.config))
    torch.testing.assert_close(exported(variant, value(2)), expected - hidden[1:3].mean(dim=-1, keepdim=True))
    disabled = DummyWorkload(replace(workload.config, enabled=False)).compile_host(CPU)
    with patch.object(TensorTask, "read", autospec=True, side_effect=TensorTask.read) as reads:
        assert disabled.step(value(2)) is None
        reads.assert_not_called()

    snapshot = execution.trace.snapshot()
    assert decode_boundary(encode_boundary(snapshot), TraceSnapshot) == snapshot
    (observed,) = (event for event in snapshot.events if event.phase is TracePhase.TENSOR)
    assert observed.step_id == 2 and tuple(identity.prompt_idx for identity in observed.identities) == (1, 2)
    torch.testing.assert_close(observed.tensor(), expected)
    _trace_selection_and_bounds(execution.plan)

    facts = backend_facts(
        5,
        (BINDING,),
        model_timeline(torch.device("cpu")),
        None,
        max_lineages=1,
        memory_bytes=0,
        max_rows=3,
        layer_count=1,
    )
    bounded = assert_finite_resources(workload, facts)
    deployment = workload.compile_backend(bounded)
    attached = deployment.execution
    torch.testing.assert_close(exported(attached, value(2)), expected)
    torch.testing.assert_close(exported(attached, value(2)), expected)
    other_layer = (AttachmentHiddenResource(replace(BINDING, layer="1"), HiddenPhase.DECODE),)
    with pytest.raises(ValueError):  # the backend does not provide the hidden layer the workload reads
        workload.plan_backend(replace(facts, probe=replace(facts.probe, hidden_resources=other_layer)))

    class Chained(DummyWorkload):
        @pllm.on(pllm.Phase.OBSERVE)
        def shift(self, x: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
            return x + 1

        @pllm.export(name="chain", max_rows=2, max_cols=3)
        def export_hidden(self, x: torch.Tensor = pllm.result("shift")) -> torch.Tensor:
            return x * 2

    torch.testing.assert_close(exported(make_llm(CPU, workload=Chained()), value(2)), (expected + 1) * 2)

    # A transform's closure values are frozen when the workload compiles.
    scale = [2.0]
    custom = replace(workload, transform=lambda rows: rows * scale[0])
    frozen = custom.compile_host(CPU)
    scale[0] = 3.0
    torch.testing.assert_close(exported(frozen, value(2)), expected * 2)
    torch.testing.assert_close(exported(custom.compile_host(CPU), value(2)), expected * 3)

    class Unsourced(DummyWorkload):
        @pllm.export()
        def export_hidden(self, first: torch.Tensor = pllm.hidden(0), second: torch.Tensor = 0) -> torch.Tensor:
            return first + second

    with pytest.raises(TypeError):  # every input names its source
        Unsourced().backend_requirements()
    _frozen_state_is_readable_by_an_observation()


def _trace_selection_and_bounds(plan) -> None:
    (task,) = plan.tasks
    trace = ExecutionTrace(plan, capacity=2)
    special = torch.tensor([float("nan"), float("inf")], dtype=torch.bfloat16)
    for _ in range(3):
        trace.record(TracePhase.TENSOR, 1, task_id=task.task_id, tensor=special, tensor_name="selected")
    snapshot = decode_boundary(encode_boundary(trace.snapshot()), TraceSnapshot)
    assert snapshot.dropped == 1 and len(snapshot.events) == 2
    torch.testing.assert_close(snapshot.events[-1].tensor(), special, equal_nan=True)

    row = PackedPhaseRow(1, 1, 0, HiddenPhase.DECODE, RequestIdentity("1", 1, 0))
    other = replace(row, identity=RequestIdentity("2", 2, 0))
    selection = TraceSelection(
        tasks=(task.task_id,),
        identities=(row.identity,),
        iterations=(2,),
        names=("activation", "weight.grad"),
        tensors=True,
        gradients=True,
        max_tensor_bytes=8,
    )
    trace = ExecutionTrace(plan, capacity=4)
    trace.configure(decode_boundary(encode_boundary(selection), TraceSelection))
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    # Wrong iteration, wrong request and a mixed-request batch are all filtered out.
    trace.observe(1, task.task_id, "activation", tensor, row=row, iteration=1)
    trace.observe(1, task.task_id, "activation", tensor, row=other, iteration=2)
    trace.observe(1, task.task_id, "activation", tensor, identities=(row.identity, other.identity), iteration=2)
    assert not trace.snapshot().events
    trace.observe(1, task.task_id, "activation", tensor, row=row, iteration=2)
    trace.observe(1, task.task_id, "weight.grad", tensor + 1, row=row, iteration=2, gradient=True)
    observed, grad = decode_boundary(encode_boundary(trace.snapshot()), TraceSnapshot).events
    assert observed.source_shape == (2, 3) and observed.row == row and grad.phase is TracePhase.GRADIENT
    torch.testing.assert_close(observed.tensor(), tensor.flatten()[:2])
    torch.testing.assert_close(grad.tensor(), (tensor + 1).flatten()[:2])
    # A bounded snapshot of a huge broadcast view must not materialize the source first.
    trace.configure(TraceSelection(tensors=True, max_tensor_bytes=128))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    small = torch.arange(1, 2049, device=device, dtype=torch.float32)
    if device == "cuda":
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    trace.observe(2, task.task_id, "bounded", small.expand(2048, -1).T)
    event = trace.snapshot().events[-1]
    torch.testing.assert_close(event.tensor(), small[:1].expand(32).cpu())
    assert event.source_shape == (2048, 2048)
    if device == "cuda":
        assert torch.cuda.max_memory_allocated() - baseline < 64 * 1024


class Encoder(torch.nn.Module):
    """A frozen PyTorch module whose forward method an observation calls."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(20, width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.weight.T


def _frozen_state_is_readable_by_an_observation() -> None:
    """A module state without an optimizer starts from the module's values and is read by an export."""

    delivered = []

    class FrozenEncoder(SAE):
        def on_export(self, event):
            delivered.append(event)

        encoder = pllm.InstanceState(module=lambda model: Encoder(model.hidden_size))

    bindings = tuple(testing.hidden_binding(layer, HiddenSite.BLOCK_OUTPUT, 18) for layer in (3, 5, 17))
    timeline = model_timeline(torch.device("cpu"))
    provider = PageBundleProviderFacts("test.plain@1", 4, StateDeviceKind.CPU)
    execution = (
        FrozenEncoder().compile_backend(backend_facts(32, bindings, timeline, provider, layer_count=18)).execution
    )
    identity = RequestIdentity("req", 0, 0)
    execution.admit(())
    execution.admit((identity,))
    source = (torch.arange(32, dtype=torch.float32) + 1).unsqueeze(0)
    execution.step(
        testing.step_input((identity,), {binding: source + i for i, binding in enumerate(bindings)}, sampler_order=())
    )
    execution.trace.drain()
    assert (
        len(delivered) == 3
        and {(event.tensor_name, event.step_id, event.identities) for event in delivered}
        == {(f"sae.{layer}", 0, (identity,)) for layer in (3, 5, 17)}
        and [event.sequence for event in delivered] == sorted(event.sequence for event in delivered)
    )
    for event in delivered:
        torch.testing.assert_close(
            event.tensor(), (source + (3, 5, 17).index(int(event.tensor_name.split(".")[1])))[:, :16]
        )
    execution.close()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]
)
@pytest.mark.parametrize("overflow", ["drop", "block"])
def test_export_queue_delivery_and_lifecycle(device, overflow) -> None:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    entered, release, delivered = Event(), Event(), []

    def consume(event):
        entered.set()
        assert release.wait(10)
        delivered.append(event)

    execution = DummyWorkload().compile_host(CPU)
    trace = ExecutionTrace(execution.plan, capacity=1, on_export=consume)
    value = torch.ones((2, 3), device=device)
    record = lambda policy: trace.record(
        TracePhase.TENSOR,
        1,
        task_id=execution.plan.tasks[0].task_id,
        tensor=value,
        tensor_name="features",
        export=policy,
    )
    try:
        torch.cuda._sleep(20_000_000) if device == "cuda" else None
        record(overflow)
        value.fill_(2)  # Same-stream reuse must follow the DtoH, including on CUDA.
        assert entered.wait(10)
        trace.record(TracePhase.BEGIN, 2)
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(record, overflow)
            try:
                if overflow == "drop":
                    pending.result(10)
                    assert trace.snapshot().export_dropped == 1
                else:
                    with pytest.raises(TimeoutError):
                        pending.result(0.05)
                assert trace.snapshot().dropped == 1
            finally:
                release.set()
            pending.result(10)
        trace.drain()
        assert len(delivered) == (1 if overflow == "drop" else 2)
        torch.testing.assert_close(delivered[0].tensor(), torch.ones((2, 3)))
        assert [event.sequence for event in delivered] == sorted(event.sequence for event in delivered)
        with trace.private():
            record(overflow)
        assert len(delivered) == (1 if overflow == "drop" else 2)
        trace.configure(TraceSelection(max_events=2))
        record(overflow)
        trace.configure(TraceSelection(max_events=1))
    finally:
        release.set()
        trace.close()
        execution.close()
    assert trace._thread is not None and not trace._thread.is_alive()


def test_export_failure_and_meta() -> None:
    class InvalidConsumer(DummyWorkload):
        def on_export(self):
            pass

    execution = InvalidConsumer().compile_host(CPU)
    trace = execution.trace
    record = lambda device: trace.record(
        TracePhase.TENSOR,
        1,
        task_id=execution.plan.tasks[0].task_id,
        tensor=torch.ones(1, device=device),
        tensor_name="features",
        export="block",
    )
    record("meta")
    assert not trace.snapshot().events
    record("cpu")
    for action in (trace.drain, lambda: record("cpu"), execution.close):
        with pytest.raises(TypeError, match="positional argument"):
            action()
