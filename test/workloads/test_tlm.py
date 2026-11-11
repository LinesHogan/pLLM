"""TLM evidence 11/24: independent factor AdamW and causal objective semantics.

The borrowed vLLM model path is covered by test/gpu/vllm_tlm_smoke.py.
"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from pllm.authoring import CausalTrainBatch
from pllm.config import ExecutionOptions
from pllm.contracts.host_batch import validate_host_batch
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.parameter_overlay import Projection, ProjectionShape
from pllm.contracts.semantic_graph import TaskResourceFact, TaskSymbol
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.causal_batch import LMHead
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.step_input import StepInput
from pllm.kernels.adamw import AdamWConfig
from pllm.testing import RequestIdentity, backend_facts
from pllm.workloads.tlm import TLMConfig, TLMMode, TLMWorkload


class TinyCausalPort:
    device = torch.device("cpu")
    tensor_dtype = TensorDType.FLOAT64
    targets = tuple(ProjectionShape(Projection(name, 0), 3, 3) for name in ("q_proj", "v_proj"))

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(23)
        self.embedding = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        self.base = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64)
        self.head = LMHead(torch.randn(3, 5, generator=generator, dtype=torch.float64).T, 5, 1.0)

    def __deepcopy__(self, memo):
        raise AssertionError("binding must retain the original frozen causal model")

    resident_bytes = 4 * 2 * 3 * 4  # Only factor templates are owned by the workload.

    def bind(self, bindings):
        return self

    def hidden(self, ids, attention_mask, factors) -> torch.Tensor:
        hidden = self.embedding[ids]
        weights = (
            self.base
            if factors is None
            else torch.stack(
                tuple(self.base[index] + factors[2 * index + 1] @ factors[2 * index] for index in range(2))
            )
        )
        return torch.tanh((hidden @ weights[0].T) + (hidden @ weights[1].T))


def causal_facts(model, *, reserved=0, task_resources=()) -> BackendCompileFacts:
    """Single-device facts before workload template reservations."""
    facts = backend_facts(
        model.targets[0].input_size,
        device=model.device.type,
        pages=4,
        memory_bytes=32768,
        max_rows=8,
        dtype=model.tensor_dtype.torch,
        layer_count=1,
        targets={shape.target: (shape.input_size, shape.output_size) for shape in model.targets},
    )
    timeline = replace(
        facts.timeline, reserved_bytes=(reserved,) * len(facts.timeline.anchors), task_resources=task_resources
    )
    return replace(facts, timeline=timeline)


def _compile(model, config: TLMConfig):
    return TLMWorkload(config, 2, 2).compile_backend(causal_facts(model), causal=model.bind).execution


def batch_input(batch):
    """Wrap a host training batch in a rowless step input."""
    rows = PackedPhaseRowBatch.from_rows(0, 0, ())
    return StepInput(rows, PackedSamplerRowBatch.from_links(rows, ()), batch=batch)


def _reference(model, parameters, batch, config):
    hidden = model.embedding[batch.input_ids]
    q = model.base[0] + parameters[1] @ parameters[0]
    v = model.base[1] + parameters[3] @ parameters[2]
    logits = model.head(torch.tanh(F.linear(hidden, q) + F.linear(hidden, v)))
    selector = (
        model.head(torch.tanh(F.linear(hidden, model.base[0]) + F.linear(hidden, model.base[1])))
        if config.mode is TLMMode.OFFLINE
        else logits.detach()
    )
    target = batch.labels[:, 1:]
    valid = target != config.ignore_index

    def ce(value):
        token = -value[:, :-1].log_softmax(-1).gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return (token * valid).sum(-1) / valid.sum(-1).clamp_min(1)

    sentence = ce(selector).detach()
    chosen = sentence > config.threshold
    return (ce(logits) * config.lamb * (sentence - config.threshold).exp() * chosen).sum() / chosen.sum().clamp_min(1)


def test_global_multifactor_adamw_matches_oracle_and_publishes_one_version() -> None:
    model = TinyCausalPort()
    config = TLMConfig(optimizer=AdamWConfig(0.01, (0.8, 0.95), 1e-7, 0.1), threshold=0.0)
    execution = _compile(model, config)
    (ref,) = execution.admit(())
    first, second = RequestIdentity("first", 0, 0), RequestIdentity("second", 1, 0)
    execution.admit((first,))
    old_pin = execution.directory.inference_pin(first)
    before = execution.arena.snapshot(old_pin, ref)
    reference = tuple(
        torch.nn.Parameter(value.reshape(2, 3) if index % 2 == 0 else value.reshape(3, 2))
        for index, value in enumerate(before[0].clone().split(6))
    )
    optimizer = torch.optim.AdamW(reference, lr=0.01, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.1, foreach=False)
    batch = CausalTrainBatch(torch.tensor([[0, 1, 2, 3], [2, 4, 1, 0]]), torch.tensor([[0, 1, -100, 3], [2, 4, 1, 0]]))
    for version in (1, 2):
        loss = _reference(model, reference, batch, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result = execution.step(batch_input(batch))
        (current,) = result.publication.after_refs
        assert current.version_id == version and result.metrics["selected_count"] == 2
        if version == 1:
            execution.admit((second,))
            second_pin = execution.directory.inference_pin(second)
        assert execution.directory.inference_pin(first) is old_pin
        assert tuple(old_pin.refs) == (ref,)
        actual = execution.snapshot(current)
        for index, parameter in enumerate(reference):
            fields = tuple(
                value if field == 1 else value[index * 6 : (index + 1) * 6].reshape(parameter.shape)
                for field, value in enumerate(actual)
            )
            state = optimizer.state[parameter]
            for value, expected in zip(
                fields, (parameter, state["step"], state["exp_avg"], state["exp_avg_sq"]), strict=True
            ):
                torch.testing.assert_close(value, expected)
        torch.testing.assert_close(result.loss, loss.detach())
        for actual_old, expected in zip(execution.arena.snapshot(old_pin, ref), before, strict=True):
            torch.testing.assert_close(actual_old, expected)
    assert second_pin.refs[0].version_id == 1  # A later request pins the version current at its admission.
    execution.retire((first, second))
    execution.retire(())
    assert execution.diagnostics().active_states == 0


def test_tlm_matches_objective_online_offline_graph_and_qv_publication() -> None:
    model = TinyCausalPort()
    batch = CausalTrainBatch(torch.tensor([[0, 2, 3]]), torch.tensor([[0, 2, 3]]))
    for mode in TLMMode:
        config = TLMConfig(mode=mode, threshold=0.0)
        limits = (TaskResourceFact(TaskSymbol("update"), 4096, input_shapes=((1, 3),)),)
        workload, facts = TLMWorkload(config, 2, 2), causal_facts(model)
        timeline = workload.resource_facts(workload.build(facts), facts, ExecutionOptions())
        assert min(timeline.reserved_bytes) == model.resident_bytes
        execution = workload.compile_backend(causal_facts(model, task_resources=limits), causal=model.bind).execution
        (initial,) = execution.admit(())
        tokens = torch.zeros((2, 3), dtype=torch.int64)
        with pytest.raises(ValueError):  # Larger than the planned input shape.
            validate_host_batch(
                CausalTrainBatch(tokens, tokens),
                execution.arena.plan.semantic_graph,
                execution.arena.plan.tensor_layout_request,
                execution.arena.plan.timeline,
                SharedTensor.describe,
                hidden_size=0,
                vocab_size=0,
            )
        anchors = {
            execution.arena.plan.semantic_graph.tasks[task.task_id].symbol: task.anchor_id
            for task in execution.arena.plan.schedule.tasks
        }
        timeline = {item.symbol: item.ordinal for item in execution.arena.plan.timeline.anchors}
        assert anchors["update"] <= anchors["factors.publish"]
        assert (anchors["factors.publish"] < timeline["model.execute"]) == (mode is TLMMode.OFFLINE)
        assert (anchors["update"] >= timeline["token.commit"]) == (mode is TLMMode.ONLINE)
        expected = _reference(
            model,
            tuple(
                value.reshape(2, 3) if index % 2 == 0 else value.reshape(3, 2)
                for index, value in enumerate(execution.snapshot(initial)[0].split(6))
            ),
            batch,
            config,
        )
        result = execution.step(batch_input(batch))
        torch.testing.assert_close(result.loss, expected)
        assert result.publication is not None and result.metrics["selected_count"] == 1
        execution.retire(())
        skipped = _compile(model, replace(config, threshold=1e6))
        (initial,) = skipped.admit(())
        result = skipped.step(batch_input(batch))
        assert result.publication is not None and result.metrics["selected_count"] == 0
        assert skipped.arena.latest(initial.lineage_id).version_id == initial.version_id + 1
        skipped.retire(())
