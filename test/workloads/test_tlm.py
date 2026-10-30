"""TLM evidence 11/24: independent factor AdamW and causal objective semantics.

The borrowed vLLM model path is covered by test/gpu/vllm_tlm_smoke.py.
"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from pllm.authoring import CausalTrainBatch
from pllm.config import ExecutionOptions
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentTopology,
    BackendPlacementProbe,
)
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget, ParameterTargetShape
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import TaskResourceFact, TaskSymbol
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.step_input import TensorStepInput
from pllm.kernels.adamw import AdamWConfig
from pllm.workloads.tlm import TLMConfig, TLMMode, TLMWorkload


class TinyCausalPort:
    device = torch.device("cpu")
    tensor_dtype = TensorDType.FLOAT64
    targets = tuple(
        ParameterTargetShape(ParameterTarget(0, ParameterModuleKind.ATTENTION, role), 3, 3)
        for role in (ParameterRole.QUERY_PROJECTION_WEIGHT, ParameterRole.VALUE_PROJECTION_WEIGHT)
    )

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(23)
        self.embedding = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        self.base = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64)
        self.head = torch.randn(3, 5, generator=generator, dtype=torch.float64)

    def __deepcopy__(self, memo):
        raise AssertionError("binding must retain the original frozen causal model")

    resident_bytes = 4 * 2 * 3 * 8  # Only factor templates are owned by the workload.

    def bind(self, bindings):
        return self

    def __call__(self, ids, attention_mask, factors) -> torch.Tensor:
        hidden = self.embedding[ids]
        weights = (
            self.base
            if factors is None
            else torch.stack(
                tuple(self.base[index] + factors[2 * index + 1] @ factors[2 * index] for index in range(2))
            )
        )
        return torch.tanh((hidden @ weights[0].T) + (hidden @ weights[1].T)) @ self.head


def _facts(model, *, reserved=0, task_resources=()) -> BackendCompileFacts:
    """Single-device facts before workload template reservations."""
    kind, ordinal = StateDeviceKind(model.device.type), model.device.index or 0
    timeline = model_timeline(model.device, ("optimizer/adamw@1", "transition/full_root@1"))
    timeline = replace(timeline, reserved_bytes=(reserved,) * len(timeline.anchors), task_resources=task_resources)
    probe = BackendPlacementProbe(
        "test.tlm@1",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        AttachmentTopology.IN_PROCESS,
        AttachmentGraphMode.EAGER,
        kind,
        ordinal,
        1,
        1,
        model.tensor_dtype,
        model.targets[0].input_size,
        8,
        (),
        (),
    )
    provider = PageBundleProviderFacts("test.tlm@1", 4, kind, ordinal)
    return BackendCompileFacts(replace(probe, targets=model.targets), timeline, (provider,), 2, 32768, 7, layer_count=1)


def _compile(model, config: TLMConfig):
    return TLMWorkload(config, 2, 2).compile_backend(_facts(model), causal=model.bind).execution


def _input(batch):
    rows = PackedPhaseRowBatch(0, 0, ())
    return TensorStepInput(rows, PackedSamplerRowBatch(rows, ()), auxiliary=batch)


def _reference(model, parameters, batch, config):
    hidden = model.embedding[batch.input_ids]
    q = model.base[0] + parameters[1] @ parameters[0]
    v = model.base[1] + parameters[3] @ parameters[2]
    logits = torch.tanh(F.linear(hidden, q) + F.linear(hidden, v)) @ model.head
    selector = (
        torch.tanh(F.linear(hidden, model.base[0]) + F.linear(hidden, model.base[1])) @ model.head
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
        result = execution.step(_input(batch))
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
                value if field == 2 else value[index * 6 : (index + 1) * 6].reshape(parameter.shape)
                for field, value in enumerate(actual)
            )
            state = optimizer.state[parameter]
            for value, expected in zip(
                fields, (parameter, parameter.grad, state["step"], state["exp_avg"], state["exp_avg_sq"]), strict=True
            ):
                torch.testing.assert_close(value, expected)
        torch.testing.assert_close(result.loss, loss.detach())
        for actual_old, expected in zip(execution.arena.snapshot(old_pin, ref), before, strict=True):
            torch.testing.assert_close(actual_old, expected)
    assert second_pin.refs[0].version_id == 1  # A later request pins the version current at its admission.
    execution.retire((first, second))
    with pytest.raises(AssertionError):
        execution.arena.snapshot(old_pin, ref)
    execution.retire(())
    assert execution.diagnostics().active_lineages == 0


def test_tlm_matches_objective_online_offline_graph_and_qv_publication() -> None:
    model = TinyCausalPort()
    if torch.cuda.is_available():
        gpu_model = TinyCausalPort()
        gpu_model.device = torch.device("cuda", 0)
        online = TLMWorkload(TLMConfig(mode=TLMMode.ONLINE), 2, 2)
        attachment = online.plan_backend(online.specialize(_facts(gpu_model), ExecutionOptions()))
        assert attachment.spec.overlay_bindings
        assert all(task.lane == "main" for task in attachment.plan.schedule.tasks)
    batch = CausalTrainBatch(torch.tensor([[0, 2, 3]]), torch.tensor([[0, 2, 3]]))
    for mode in TLMMode:
        config = TLMConfig(mode=mode, threshold=0.0)
        limits = (TaskResourceFact(TaskSymbol("update"), 4096, input_shapes=((1, 3),)),)
        assert (
            min(TLMWorkload(config, 2, 2).resource_facts(_facts(model, reserved=0), ExecutionOptions()).reserved_bytes)
            == model.resident_bytes
        )
        facts = _facts(model, task_resources=limits)
        execution = TLMWorkload(config, 2, 2).compile_backend(facts, causal=model.bind).execution
        (initial,) = execution.admit(())
        tokens = torch.zeros((2, 3), dtype=torch.int64)
        with pytest.raises(ValueError):  # Larger than the planned input shape.
            validate_auxiliary(
                CausalTrainBatch(tokens, tokens),
                execution.arena.plan,
                SharedTensor.describe,
                hidden_size=0,
                vocab_size=0,
            )
        anchors = {
            execution.arena.plan.semantic_graph.tasks[task.task_id].symbol: task.anchor_id
            for task in execution.arena.plan.schedule.tasks
        }
        timeline = {item.symbol: item.ordinal for item in execution.arena.plan.timeline.anchors}
        assert anchors["update"] <= anchors["publish"]
        assert (anchors["publish"] < timeline["model.execute"]) == (mode is TLMMode.OFFLINE)
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
        result = execution.step(_input(batch))
        torch.testing.assert_close(result.loss, expected)
        assert result.publication is not None and result.metrics["selected_count"] == 1
        execution.retire(())
        skipped = _compile(model, replace(config, threshold=1e6))
        (initial,) = skipped.admit(())
        result = skipped.step(_input(batch))
        assert result.publication is not None and result.metrics["selected_count"] == 0
        assert skipped.arena.latest(initial.lineage_id).version_id == initial.version_id + 1
        skipped.retire(())
